import torch
import logging.config
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.data import PyTorchDataLoader
from allennlp.data.dataloader import TensorDict
from datasets.readers.oos_eval import OOSEvalReader
from allennlp.data import DatasetReader, DataLoader, Instance
from models.bert_linear_classifier import BertLinearClassifier
from allennlp.training.trainer import EpochCallback, BatchCallback
from allennlp.training.optimizers import HuggingfaceAdamWOptimizer
from typing import Any, List, Dict, Tuple, Value, Iterable, Optional
from allennlp.modules.seq2vec_encoders.bert_pooler import BertPooler
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder

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

        self.config: Optional[Dict[str, Value]] = None

        self.wandb = wbrun
        self.batch_end_log_freq = 1
        self.current_batch_num = -1
        self.previous_logged_batch = -1

    def update_config(self, trainer: GradientDescentTrainer) -> None:
        if self.config is None:
            # we assume that allennlp train pipeline would have written
            # the entire config to the file by this time
            log.info("Updating config in callback init...")
            wbconf = {}
            wbconf["batch_size"] = 64
            wbconf["lr"] = 0.0001
            wbconf["num_epochs"] = 5
            wbconf["no_cuda"] = False
            wbconf["log_interval"] = 10
            self.config = wbconf
            self.wandb.config.update(self.config)

    def __call__(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
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
            log.info("Writing metrics for the batch to wandb")
            print(f"Batch outputs are: {batch_outputs!r}")

            batch_outputs = [{
                key: value.cpu()
                for key, value
                in batch_output.items()
                if isinstance(value, torch.Tensor)
            } for batch_output in batch_outputs]

            self.wandb.log(
                {
                    **batch_outputs[0],
                },
                step=self.current_batch_num,
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

        self.config: Optional[Dict[str, Value]] = None

        self.wandb = wbrun
        self.epoch_end_log_freq = 1
        self.current_batch_num = -1
        self.current_epoch_num = -1
        self.previous_logged_epoch = -1

    def update_config(self, trainer: GradientDescentTrainer) -> None:
        if self.config is None:
            # we assume that allennlp train pipeline would have written
            # the entire config to the file by this time
            log.info("Updating config in callback init...")
            wbconf = {}
            wbconf["batch_size"] = 64
            wbconf["lr"] = 0.0001
            wbconf["num_epochs"] = 5
            wbconf["no_cuda"] = False
            wbconf["log_interval"] = 10
            self.config = wbconf
            self.wandb.config.update(self.config)

    def __call__(
            self,
            trainer: GradientDescentTrainer,
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
            log.info("Writing metrics for the epoch to wandb")
            log.debug(f"Metrics are: {metrics!r}")
            self.wandb.log(
                {
                    **metrics,
                },
                step=self.current_epoch_num,
            )
            self.previous_logged_epoch = self.current_epoch_num


def build_dataset_reader() -> DatasetReader:
    """
    Instantiate dataset reader - factory method.

    :return OOSEvalReader: Instantiated DatasetReader object.
    """
    return OOSEvalReader()


def build_epoch_callbacks(wbrun) -> EpochCallback:
    """
    Instantiate callback - factory method.

    :return LogMetricsToWandb: Instantiated LogMetricsToWandb object.
    """
    return [LogMetricsToWandb(wbrun=wbrun)]


def build_batch_callbacks(wbrun) -> BatchCallback:
    """
    Instantiate callback - factory method.

    :return LogBatchMetricsToWandb: Instantiated LogBatchMetricsToWandb object.
    """
    return [LogBatchMetricsToWandb(wbrun=wbrun)]


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    """
    Build the Vocabulary object from the instances.

    :param instances: Iterable of allennlp instances.
    :return Vocabulary: The Vocabulary object.
    """
    log.debug("Building the vocabulary.")
    return Vocabulary.from_instances(instances)


def build_model(vocab: Vocabulary, wbrun: Any) -> Model:
    """
    Build the Model object, along with the embedder and encoder.

    :param vocab: The pre-instantiated vocabulary object.
    :return Model: The model object itself.
    """
    log.debug("Building the model.")
    # vocab_size = vocab.get_vocab_size("tokens")

    # TokenEmbedder object.
    bert_embedder = PretrainedTransformerEmbedder("bert-base-uncased")

    # TextFieldEmbedder that wraps TokenEmbedder objects. Each
    # TokenEmbedder output from one TokenIndexer--the data produced
    # by a TextField is a dict {names:representations}, hence
    # TokenEmbedders have corresponding names.
    embedder: TextFieldEmbedder = BasicTextFieldEmbedder(
        {"tokens": bert_embedder}
    )

    log.debug("Embedder built.")
    encoder = BertPooler("bert-base-uncased", requires_grad=True)
    # encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(768,20,batch_first=True))
    log.debug("Encoder built.")
    return BertLinearClassifier(vocab, embedder, encoder, wbrun).cuda(0)


def build_data_loader(
    data: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """
    Build an AllenNLP DataLoader.

    :param train_data: The training dataset, torch object.
    :param dev_data: The dev dataset, torch object.
    :return train_loader, dev_loader: The train and dev data loaders as a
            tuple.
    """
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    log.debug("Building DataLoader.")
    loader = PyTorchDataLoader(
        data,
        batch_size=batch_size,
        shuffle=True
    )
    log.debug("DataLoader built.")
    return loader


def build_train_data_loaders(
    train_data: torch.utils.data.Dataset,
    dev_data: torch.utils.data.Dataset,
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Build the AllenNLP DataLoaders.

    :param train_data: The training dataset, torch object.
    :param dev_data: The dev dataset, torch object.
    :return train_loader, dev_loader: The train and dev data loaders as a
            tuple.
    """
    log.debug("Building Training DataLoaders.")
    train_loader = build_data_loader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    dev_loader = build_data_loader(
        dev_data,
        batch_size=batch_size,
        shuffle=False
    )
    log.debug("Training DataLoaders built.")
    return train_loader, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    lr: float,
    num_epochs: int,
    wbrun: Any
) -> Trainer:
    """
    Build the model trainer. Includes instantiating the optimizer as well.

    :param model: The model object to be trained.
    :param serialization_dir: The serialization directory to output
            results to.
    :param train_loader: The training data loader.
    :param dev_loader: The dev data loader.
    :return trainer: The Trainer object.
    """
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = HuggingfaceAdamWOptimizer(parameters, lr=lr)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        epoch_callbacks=build_epoch_callbacks(wbrun),
        batch_callbacks=build_batch_callbacks(wbrun)
    )
    return trainer
