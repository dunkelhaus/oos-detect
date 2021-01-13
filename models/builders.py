import torch
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.data import DataLoader
from typing import Any, Tuple, Iterable
from allennlp.data import PyTorchDataLoader
from allennlp.data import DatasetReader, Instance
from allennlp.training.trainer import EpochCallback
from allennlp.training.trainer import BatchCallback
from utilities.wandb_loggers import LogMetricsToWandb
from utilities.wandb_loggers import LogBatchMetricsToWandb


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


def build_vocab(
        instances: Iterable[Instance]
) -> Vocabulary:
    """
    Build the Vocabulary object from the instances.

    :param instances: Iterable of allennlp instances.
    :return Vocabulary: The Vocabulary object.
    """
    # log.debug("Building the vocabulary.")

    return Vocabulary.from_instances(instances)


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
    loader = PyTorchDataLoader(
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
