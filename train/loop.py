import os
import wandb
import torch
import logging
from train.builders import build_model
from train.builders import build_vocab
from train.builders import build_trainer
from allennlp.training.util import evaluate
from utilities.locate import locate_oos_data
from train.builders import build_data_loader
from typing import List, Dict, Tuple, Iterable
from train.builders import build_dataset_reader
from utilities.locate import locate_results_dir
from configs.log.log_conf import LOGGING_CONFIG
from allennlp.data import Instance, DatasetReader
from train.builders import build_train_data_loaders

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


def run_training_loop(run_test: bool = False):
    wbrun = wandb.init(project="oos-detect", sync_tensorboard=False, name="dunkrun")
    # wandb.tensorboard.patch(save=True, tensorboardX=False)
    batch_size = 64
    lr = 0.0001
    num_epochs = 5

    """wbconf = wandb.config

    wbconf.no_cuda = False
    wbconf.log_interval = 10
    log.debug(f"WandB config: {wandb.config!r}")"""
    dataset_reader = build_dataset_reader()

    # These are a subclass of pytorch Datasets, with some allennlp-specific
    # functionality added.
    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab, wbrun)

    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and this
    # is how we do it.
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    # These are again a subclass of pytorch DataLoaders, with an
    # allennlp-specific collate function, that runs our indexing and
    # batching code.
    train_loader, dev_loader = build_train_data_loaders(train_data, dev_data, batch_size)

    # Locate serialization directory.
    serialization_dir = locate_results_dir()

    if not serialization_dir:
        log.info("Failed to locate results directory, stopping.")
        return

    wandb.watch(model, log="all")

    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this guide, we need to do this.
    trainer = build_trainer(
        model,
        serialization_dir,
        train_loader,
        dev_loader,
        lr=lr,
        num_epochs=num_epochs,
        wbrun=wbrun
    )

    trainer.train()

    if run_test:
        test_data = dataset_reader.read(locate_oos_data()/"data_full_test.json")
        test_data.index_with(model.vocab)
        test_data_loader = build_data_loader(
            test_data,
            batch_size=8,
            shuffle=False
        )
        results = evaluate(model, test_data_loader, cuda_device=0)
        print(results)
        #log.info(results)


    # wandb.join()
    # torch.save(model.state_dict(), os.path.join(wandb.run.dir, "linear-bert-uncased.pt"))

    return model, dataset_reader


if __name__ == '__main__':
    """import logging.config
    from configs.log.log_conf import LOGGING_CONFIG

    # --- Universal logger setup - startup task ---
    logging.config.dictConfig(LOGGING_CONFIG)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    # logging.getLogger("transformers").setLevel(logging.DEBUG)

    # Logger setup.
    log = logging.getLogger(__name__)
    log.debug("Logging is configured.")"""

    model, dataset_reader = run_training_loop()
