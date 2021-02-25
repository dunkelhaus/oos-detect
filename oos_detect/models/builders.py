import torch
from allennlp.models import Model
from allennlp.data import Instance
from allennlp.data import Vocabulary
from allennlp.data import DataLoader
from typing import Any, Tuple, Iterable
from allennlp.training.trainer import Trainer
from allennlp.data import MultiProcessDataLoader
from allennlp.training.trainer import TrainerCallback
from oos_detect.train.callbacks import LogMetricsToWandb
from allennlp.training.trainer import GradientDescentTrainer
from allennlp.training.optimizers import HuggingfaceAdamWOptimizer


def build_callbacks(wbrun) -> TrainerCallback:
    """
    Instantiate callback - factory method.

    :param wbrun: WandB object.
    :return LogMetricsToWandb: Instantiated LogMetricsToWandb object.
    """

    return [LogMetricsToWandb(wbrun=wbrun)]


def build_vocab(
        instances: Iterable[Instance],
        from_transformer: bool = False
) -> Vocabulary:
    """
    Build the Vocabulary object from the instances only,
    or from the pretrained transformer, based on boolean flag

    :param instances: Iterable of allennlp instances.
    :param from_transformer: Whether to initialize vocab from
                             pretrained transformer, or from
                             instances directly.
    :return Vocabulary: The Vocabulary object.
    """
    # log.debug("Building the vocabulary.")
    print(f"Adding {len(instances)} instances data to vocab.")

    if from_transformer:
        vocab = Vocabulary.from_pretrained_transformer(
            model_name="bert-base-uncased"
        )
        vocab.extend_from_instances(instances)
    else:
        vocab = Vocabulary.from_instances(instances)

    return vocab


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
    # log.debug("Building DataLoader.")
    loader = MultiProcessDataLoader(
        data,
        batch_size=batch_size,
        shuffle=True
    )
    # log.debug("DataLoader built.")

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
    # log.debug("Building Training DataLoaders.")
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
    # log.debug("Training DataLoaders built.")

    return train_loader, dev_loader


def build_grad_desc_with_adam_trainer(
        model: Model,
        serialization_dir: str,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        lr: float,
        num_epochs: int,
        wbrun: Any = None
) -> Trainer:
    """
    Build the model trainer.
    Includes instantiating the optimizer as well.
    This builder uses the GradientDescentTrainer &
    HuggingfaceAdamWOptimizer combo.
    Also allows setting callbacks (atm for WandB mainly).

    :param model: The model object to be trained.
    :param serialization_dir: The serialization directory to output
            results to.
    :param train_loader: The training data loader.
    :param dev_loader: The dev data loader.
    :param lr: Learning rate.
    :param num_epochs: Number of epochs to train for.
    :param wbrun: WandB object to use for callbacks.
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
        callbacks=build_callbacks(wbrun) if wbrun else None
    )

    return trainer
