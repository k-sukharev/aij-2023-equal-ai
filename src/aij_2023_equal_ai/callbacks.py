import logging
from typing import Optional


import lightning.pytorch as pl
from lightning.fabric.utilities.rank_zero import _get_rank
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.rank_zero import rank_prefixed_message

log = logging.getLogger(__name__)


class EpochStopping(Callback):
    r"""Stop training when it reach epoch threshold.
    """
    def __init__(
        self,
        epoch_threshold: int,
        verbose: bool = False,
        log_rank_zero_only: bool = False,
    ):
        super().__init__()
        self.epoch_threshold = epoch_threshold
        self.verbose = verbose
        self.log_rank_zero_only = log_rank_zero_only

    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        from lightning.pytorch.trainer.states import TrainerFn

        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_skip_check(trainer):
            return
        self._run_stopping_check(trainer)

    def _run_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if trainer.fast_dev_run:
            return

        if trainer.current_epoch == self.epoch_threshold:
            should_stop, reason = True, f'Epoch threshold reached: {trainer.current_epoch}'
        else:
            should_stop, reason = False, None

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)

    @staticmethod
    def _log_info(trainer: Optional["pl.Trainer"], message: str, log_rank_zero_only: bool) -> None:
        rank = _get_rank(
            strategy=(trainer.strategy if trainer is not None else None),  # type: ignore[arg-type]
        )
        if trainer is not None and trainer.world_size <= 1:
            rank = None
        message = rank_prefixed_message(message, rank)
        if rank is None or not log_rank_zero_only or rank == 0:
            log.info(message)
