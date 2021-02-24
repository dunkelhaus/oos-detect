import torch
import logging.config
from typing import Any, List, Dict, Optional
from allennlp.training.trainer import Trainer
from allennlp.data.dataloader import TensorDict
from allennlp.training.trainer import EpochCallback, BatchCallback

# Logger setup.
log = logging.getLogger(__name__)


class LogBatchMetricsToWandb(BatchCallback):
    def __init__(
            self,
            wbrun: Any,
            epoch_end_log_freq: int = 1
    ) -> None:
        # import wandb here to be sure that it was initialized
        # before this line was executed
        super().__init__()
        # import wandb  # type: ignore

        self.config: Optional[Dict[str, Any]] = None

        self.wandb = wbrun
        self.batch_end_log_freq = 1
        self.current_batch_num = -1
        self.previous_logged_batch = -1

    def update_config(self, trainer: Trainer) -> None:
        if self.config is None:
            # we assume that allennlp train pipeline would have written
            # the entire config to the file by this time
            log.info("Updating config in callback init...")
            wbconf = {}
            wbconf["batch_size"] = 64
            wbconf["lr"] = 0.0001
            wbconf["num_epochs"] = 3
            wbconf["no_cuda"] = False
            wbconf["log_interval"] = 10
            self.config = wbconf
            self.wandb.config.update(self.config)

    def __call__(
        self,
        trainer: Trainer,
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_master: bool
    ) -> None:
        """
        This should run after all the epoch end metrics have
        been computed by the metric_tracker callback.
        """
        if self.config is None:
            self.update_config(trainer)

        self.current_batch_num += 1

        if (is_master
                and (self.current_batch_num - self.previous_logged_batch)
                >= self.batch_end_log_freq):
            # print("Writing metrics for the batch to wandb.")
            # print(f"Batch outputs are: {batch_outputs!r}")
            # print(f"Batch metrics are: {batch_metrics!r}")

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


class LogMetricsToWandb(EpochCallback):
    def __init__(
            self,
            wbrun: Any,
            epoch_end_log_freq: int = 1
    ) -> None:
        # import wandb here to be sure that it was initialized
        # before this line was executed
        super().__init__()
        # import wandb  # type: ignore

        self.config: Optional[Dict[str, Any]] = None

        self.wandb = wbrun
        self.epoch_end_log_freq = 1
        self.current_batch_num = -1
        self.current_epoch_num = -1
        self.previous_logged_epoch = -1

    def update_config(self, trainer: Trainer) -> None:
        if self.config is None:
            # we assume that allennlp train pipeline would have written
            # the entire config to the file by this time
            log.info("Updating config in callback init...")
            wbconf = {}
            wbconf["batch_size"] = 64
            wbconf["lr"] = 0.0001
            wbconf["num_epochs"] = 3
            wbconf["no_cuda"] = False
            wbconf["log_interval"] = 10
            self.config = wbconf
            self.wandb.config.update(self.config)

    def __call__(
            self,
            trainer: Trainer,
            metrics: Dict[str, Any],
            epoch: int,
            is_master: bool,
    ) -> None:
        """ This should run after all the epoch end metrics have
        been computed by the metric_tracker callback.
        """

        if self.config is None:
            self.update_config(trainer)

        self.current_epoch_num += 1

        if (is_master
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
