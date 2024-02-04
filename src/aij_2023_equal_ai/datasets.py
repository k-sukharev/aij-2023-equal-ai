import itertools
import os

import lightning.pytorch as pl
import pytorchvideo
import torch.distributed as dist

from typing import Any, Dict

from pytorchvideo.data.clip_sampling import ClipInfo, ClipSampler, RandomClipSampler
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler
from torchvision.transforms import Compose


class FromStartClipSampler(ClipSampler):
    """
    Samples clip of size clip_duration from start of the videos.
    """

    def __call__(
        self, last_clip_time: float, video_duration: float, annotation: Dict[str, Any]
    ) -> ClipInfo:
        """
        Args:
            last_clip_time (float): Not used for FromStartClipSampler.
            video_duration: (float): the duration (in seconds) for the video that's
                being sampled
            annotation (Dict): Not used by this sampler.
        Returns:
            clip_info (ClipInfo): includes the clip information of (clip_start_time,
            clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
            clip_index, aux_index and is_last_clip are always 0, 0 and True, respectively.

        """
        clip_start_sec = 0.0
        return ClipInfo(
            clip_start_sec, clip_start_sec + self._clip_duration, 0, 0, True
        )


class LimitDataset(Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos


class KineticsDataModule(pl.LightningDataModule):
    """
    This LightningDataModule implementation constructs a PyTorchVideo Kinetics dataset for both
    the train and val partitions. It defines each partition's augmentation and
    preprocessing transforms and configures the PyTorch DataLoaders.
    """

    def __init__(
            self,
            data_path: str,
            video_path_prefix: str,
            train_transforms: Compose,
            val_transforms: Compose,
            clip_duration: float = 2.67,
            seed: int = 42,
            batch_size: int = 32,
            num_workers: int = os.cpu_count(),
            pin_memory: bool = False
        ) -> None:
        self.data_path = data_path
        self.video_path_prefix = video_path_prefix
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.clip_duration = clip_duration
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        super().__init__()

    def setup(self, stage: str | None = None) -> None:
        if stage == 'fit' or stage is None:
            use_ddp = dist.is_available() and dist.is_initialized()
            sampler = DistributedSampler if use_ddp else RandomSampler
            self.train_dataset = LimitDataset(
                pytorchvideo.data.Kinetics(
                    data_path=os.path.join(self.data_path, 'train.csv'),
                    clip_sampler=RandomClipSampler(clip_duration=self.clip_duration),
                    video_sampler=sampler,
                    transform=self.train_transforms,
                    video_path_prefix=self.video_path_prefix,
                    decode_audio=False
                )
            )
            self.val_dataset = LimitDataset(
                pytorchvideo.data.Kinetics(
                    data_path=os.path.join(self.data_path, 'val.csv'),
                    clip_sampler=FromStartClipSampler(clip_duration=self.clip_duration),
                    video_sampler=sampler,
                    transform=self.val_transforms,
                    video_path_prefix=self.video_path_prefix,
                    decode_audio=False
                )
            )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
