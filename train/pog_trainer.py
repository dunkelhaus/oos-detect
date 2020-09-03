import torch
from allennlp.models import Model
from allennlp.data import DataLoader
from allennlp.training import Trainer


class POGTrainer(Trainer):

    def __init__(
            self,
            model: Model,
            optimizer: torch.optim.Optimizer,
            data_loader: DataLoader,
            patience: Optional[int] = None,
            validation_metric: str = "-loss",
            validation_data_loader: DataLoader = None,
            num_epochs: int = 20,
            serialization_dir: Optional[str] = None,
            checkpointer: Checkpointer = None,
            cuda_device: Optional[Union[int, torch.device]] = None,
            grad_norm: Optional[float] = None,
            grad_clipping: Optional[float] = None,
            learning_rate_scheduler: Optional[LearningRateScheduler] = None,
            momentum_scheduler: Optional[MomentumScheduler] = None,
            tensorboard_writer: TensorboardWriter = None,
            moving_average: Optional[MovingAverage] = None,
            batch_callbacks: List[BatchCallback] = None,
            epoch_callbacks: List[EpochCallback] = None,
            distributed: bool = False,
            local_rank: int = 0,
            world_size: int = 1,
            num_gradient_accumulation_steps: int = 1,
            use_amp: bool = False,
    ):
        super().__init__(serialization_dir, cuda_device, distributed, local_rank, world_size)

        
