import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch.distributed as dist

from collections import OrderedDict

from hydra.utils import instantiate
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

try:
    from mmaction.apis import init_recognizer
except (ImportError, ModuleNotFoundError):
    pass


def set_weight_decay(model, weight_decay, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1 or
            name.endswith('.bias') or
            name in skip_list or
            check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [
        {'params': has_decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.}
    ]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


class MMActionWrapper(nn.Module):
    def __init__(self, config, checkpoint=None, device='cpu'):
        super().__init__()
        model = init_recognizer(
            config,
            checkpoint,
            device=device
        )
        self.module = nn.Sequential(OrderedDict([
            ('backbone', model.backbone),
            ('cls_head', model.cls_head)
        ]))

    def forward(self, x):
        return self.module(x)


class VideoClassificationLightningModule(pl.LightningModule):
    def __init__(
            self,
            net,
            criterion,
            num_classes,
            batch_key,
            optimizer,
            scheduler_dict,
            scheduler
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = instantiate(
            net
        )
        self.criterion = instantiate(
            criterion
        )
        self.train_accuracy = MulticlassAccuracy(self.hparams.num_classes)
        self.val_accuracy = MulticlassAccuracy(self.hparams.num_classes)

    def on_train_epoch_start(self):
        """
        For distributed training we need to set the datasets video sampler epoch so
        that shuffling is done correctly
        """
        epoch = self.trainer.current_epoch
        use_ddp = dist.is_available() and dist.is_initialized()
        if use_ddp:
            self.trainer.datamodule.train_dataset.dataset.video_sampler.set_epoch(epoch)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.hparams.batch_key]
        batch_size = x.shape[0]
        y_hat = self.net(x)
        loss = self.criterion(y_hat, batch['label'])
        acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch['label'])
        self.log(
            'train_loss', loss,
            batch_size=batch_size
        )
        self.log(
            'train_acc', acc,
            on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
            batch_size=batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.hparams.batch_key]
        batch_size = x.shape[0]
        y_hat = self.net(x)
        loss = self.criterion(y_hat, batch['label'])
        acc = self.val_accuracy(F.softmax(y_hat, dim=-1), batch['label'])
        self.log(
            'val_loss', loss,
            batch_size=batch_size
        )
        self.log(
            'val_acc', acc,
            on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
            batch_size=batch_size
        )
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(
            self.hparams.optimizer,
            params=(
                set_weight_decay(self.trainer.model, self.hparams.optimizer['weight_decay'])
                if 'weight_decay' in self.hparams.optimizer
                else self.trainer.model.parameters()
            )
        )
        scheduler_dict = self.hparams.scheduler_dict
        if scheduler_dict.get('interval', None) == 'step':
            scheduler_dict['scheduler'] = instantiate(
                self.hparams.scheduler,
                optimizer=optimizer,
                warmup_epochs=int(
                    self.hparams.scheduler['warmup_epochs'] /
                    self.hparams.scheduler['max_epochs'] *
                    self.trainer.estimated_stepping_batches
                ),
                max_epochs=self.trainer.estimated_stepping_batches
            )
        else:
            scheduler_dict['scheduler'] = instantiate(
                self.hparams.scheduler,
                optimizer=optimizer
            )
        return [optimizer], [self.hparams.scheduler_dict]


class DistillationLightningModule(pl.LightningModule):
    def __init__(
            self,
            student,
            student_chekpoint,
            num_classes,
            student_loss,
            teacher,
            temperature,
            distillation_loss,
            distillation_loss_weight,
            batch_key,
            optimizer,
            scheduler_dict,
            scheduler
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.student = instantiate(
            student
        )
        if student_chekpoint is not None:
            self.student.load_state_dict(torch.load(student_chekpoint))
        self.teacher = instantiate(
            teacher
        )

        self.student_loss = instantiate(
            student_loss
        )
        self.distillation_loss = instantiate(
            distillation_loss
        )

        self.train_accuracy = MulticlassAccuracy(self.hparams.num_classes)
        self.val_accuracy = MulticlassAccuracy(self.hparams.num_classes)

    def on_train_epoch_start(self):
        """
        For distributed training we need to set the datasets video sampler epoch so
        that shuffling is done correctly
        """
        epoch = self.trainer.current_epoch
        use_ddp = dist.is_available() and dist.is_initialized()
        if use_ddp:
            self.trainer.datamodule.train_dataset.dataset.video_sampler.set_epoch(epoch)

    def forward(self, x):
        return self.student(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.hparams.batch_key]
        batch_size = x.shape[0]

        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(x.unsqueeze(1), stage='head')
            soft_targets = F.softmax(teacher_logits / self.hparams.temperature, dim=-1)
        student_logits = self.student(x[:, :, ::2])
        dist_loss = self.distillation_loss(F.log_softmax(student_logits / self.hparams.temperature, dim=-1), soft_targets)
        stud_loss = self.student_loss(student_logits, batch['label'])
        loss = (
            self.hparams.distillation_loss_weight * dist_loss +
            (1 - self.hparams.distillation_loss_weight) * stud_loss
        )
        acc = self.train_accuracy(F.softmax(student_logits, dim=-1), batch['label'])
        self.log(
            'train_loss', loss,
            batch_size=batch_size
        )
        self.log(
            'train_acc', acc,
            on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
            batch_size=batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.hparams.batch_key]
        batch_size = x.shape[0]
        student_logits = self.student(x)
        loss = self.student_loss(student_logits, batch['label'])
        acc = self.val_accuracy(F.softmax(student_logits, dim=-1), batch['label'])
        self.log(
            'val_loss', loss,
            batch_size=batch_size
        )
        self.log(
            'val_acc', acc,
            on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
            batch_size=batch_size
        )
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(
            self.hparams.optimizer,
            params=(
                set_weight_decay(self.trainer.model, self.hparams.optimizer['weight_decay'])
                if 'weight_decay' in self.hparams.optimizer
                else self.trainer.model.parameters()
            )
        )
        scheduler_dict = self.hparams.scheduler_dict
        if scheduler_dict.get('interval', None) == 'step':
            scheduler_dict['scheduler'] = instantiate(
                self.hparams.scheduler,
                optimizer=optimizer,
                warmup_epochs=int(
                    self.hparams.scheduler['warmup_epochs'] /
                    self.hparams.scheduler['max_epochs'] *
                    self.trainer.estimated_stepping_batches
                ),
                max_epochs=self.trainer.estimated_stepping_batches
            )
        else:
            scheduler_dict['scheduler'] = instantiate(
                self.hparams.scheduler,
                optimizer=optimizer
            )
        return [optimizer], [self.hparams.scheduler_dict]
