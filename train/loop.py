import os
import wandb
import torch
import logging
from train.builders import build_model
from train.builders import build_vocab
from train.builders import build_trainer
from utilities.locate import locate_oos_data
from train.builders import build_data_loaders
from typing import List, Dict, Tuple, Iterable
from train.builders import build_dataset_reader
from utilities.locate import locate_results_dir
from configs.log.log_conf import LOGGING_CONFIG
from allennlp.data import Instance, DatasetReader

# Logger setup.
log = logging.getLogger(__name__)


def read_data(
        reader: DatasetReader
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    path = locate_oos_data()
    training_data = reader.read(path/"data_full_train.json")
    validation_data = reader.read(path/"data_full_val.json")
    return training_data, validation_data


def run_training_loop():
    wandb.init(entity="dunkelhaus", project="oos-detect")
    wbconf = wandb.config

    wbconf.batch_size = 64
    wbconf.lr = 0.0001
    wbconf.num_epochs = 5
    wbconf.no_cuda = False
    wbconf.log_interval = 10
    dataset_reader = build_dataset_reader()

    # These are a subclass of pytorch Datasets, with some allennlp-specific
    # functionality added.
    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab)

    wandb.watch(model, log="all")

    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and this
    # is how we do it.
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    # These are again a subclass of pytorch DataLoaders, with an
    # allennlp-specific collate function, that runs our indexing and
    # batching code.
    train_loader, dev_loader = build_data_loaders(train_data, dev_data, wbconf.batch_size)

    # Locate serialization directory.
    serialization_dir = locate_results_dir()

    if not serialization_dir:
        log.info("Failed to locate results directory, stopping.")
        return

    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this guide, we need to do this.
    trainer = build_trainer(
        model,
        serialization_dir,
        train_loader,
        dev_loader,
        lr=wbconf.lr,
        num_epochs=wbconf.num_epochs
    )

    trainer.train()
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "linear-bert-uncased.pt"))

    return model, dataset_reader
