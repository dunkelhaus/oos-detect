import torch
from typing import Any, List, Dict
from allennlp.training.trainer import Trainer
from allennlp.training.trainer import TrainerCallback
from utils.exceptions import UnskippableSituationError
from allennlp.data.data_loaders.data_loader import TensorDict


class LogMetricsToWandb(TrainerCallback):
    def __init__(
            self,
            wbrun: Any,
            epoch_end_log_freq: int = 1
    ) -> None:
        # import wandb here to be sure that it was initialized
        # before this line was executed
        super().__init__()
        # import wandb  # type: ignore

        self.wandb = wbrun
        self.config = self.wandb.config

        self.batch_end_log_freq = 1
        self.epoch_end_log_freq = 1

        self.current_batch_num = -1
        self.current_epoch_num = -1

        self.previous_logged_batch = -1
        self.previous_logged_epoch = -1

    def update_config(self, trainer: Trainer) -> None:
        if self.config is None:
            print("Config is none. How did this happen?")
            raise UnskippableSituationError()

    def on_batch(
        self,
        trainer: Trainer,
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool
    ) -> None:
        """
        This should run after all the epoch end metrics have
        been computed by the metric_tracker callback.
        """
        if self.config is None:
            self.update_config(trainer)

        self.current_batch_num += 1

        if (is_primary
                and (self.current_batch_num - self.previous_logged_batch)
                >= self.batch_end_log_freq):
            print("Writing metrics for the batch to wandb.")

            batch_outputs = [{
                key: value.cpu()
                for key, value
                in batch_output.items()
                if isinstance(value, torch.Tensor)
            } for batch_output in batch_outputs]

            self.wandb.log(
                {
                    **batch_outputs[0],
                    **batch_metrics
                }
            )
            self.previous_logged_batch = self.current_batch_num

    def on_epoch(
            self,
            trainer: Trainer,
            metrics: Dict[str, Any],
            epoch: int,
            is_primary: bool,
    ) -> None:
        """ This should run after all the epoch end metrics have
        been computed by the metric_tracker callback.
        """

        if self.config is None:
            self.update_config(trainer)

        self.current_epoch_num += 1

        if (is_primary
                and (self.current_epoch_num - self.previous_logged_epoch)
                >= self.epoch_end_log_freq):
            # print("Writing metrics for the epoch to wandb.")
            print(f"Epoch metrics are: {metrics!r}")
            self.wandb.log(
                {
                    **metrics,
                }
            )
            self.previous_logged_epoch = self.current_epoch_num
