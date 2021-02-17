import wandb
import logging
from typing import Tuple
from typing import Iterable
from allennlp.models import Model
from allennlp.data import Instance
from oos_detect.models.builders import build_vocab
from allennlp.training.util import evaluate
from oos_detect.models.builders import build_data_loader
from oos_detect.utilities.locate import locate_results_dir
from oos_detect.models.builders import build_train_data_loaders

# Logger setup.
# from configs.log.log_conf import LOGGING_CONFIG
log = logging.getLogger(__name__)


def run_training(
        data: Tuple[Iterable[Instance], Iterable[Instance]],
        model_builder,
        run_name: str
) -> Model:
    wbrun = wandb.init(
        project="oos-detect",
        sync_tensorboard=False,
        name=run_name
    )
    print("Running over training set.")
    # wandb.tensorboard.patch(save=True, tensorboardX=False)
    batch_size = 64
    lr = 0.001
    num_epochs = 80
    train_data, dev_data = data

    # wbconf = wandb.config

    # wbconf.no_cuda = False
    # wbconf.log_interval = 10
    # log.debug(f"WandB config: {wandb.config!r}")

    print(f"Example training instance: {train_data[0]}.")

    vocab = build_vocab(train_data + dev_data)
    print(f"Vocab size: {vocab.get_index_to_token_vocabulary()}")

    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and
    # this is how we do it.
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    # These are again a subclass of pytorch DataLoaders, with an
    # allennlp-specific collate function, that runs our indexing and
    # batching code.
    train_loader, dev_loader = build_train_data_loaders(
        train_data,
        dev_data,
        batch_size
    )

    # Locate serialization directory.
    serialization_dir = locate_results_dir()

    if not serialization_dir:
        log.info("Failed to locate results directory, stopping.")
        return

    model, trainer = model_builder(
        serialization_dir,
        train_loader,
        dev_loader,
        lr,
        num_epochs,
        vocab,
        wbrun
    )

    wandb.watch(model, log="all")

    trainer.train()

    # wandb.join()
    # torch.save(model.state_dict(), os.path.join(wandb.run.dir,
    # "linear-bert-uncased.pt"))

    return model


def run_testing(
        data: Iterable[Instance],
        model: Model
) -> Model:
    print("Running over test set.")

    data.index_with(model.vocab)
    test_data_loader = build_data_loader(
        data,
        batch_size=8,
        shuffle=False
    )
    results = evaluate(model, test_data_loader, cuda_device=0)
    print(f"Test results: {results}.")
    # log.info(results)

    return model


if __name__ == '__main__':
    # import logging.config
    # from configs.log.log_conf import LOGGING_CONFIG

    # --- Universal logger setup - startup task ---
    # logging.config.dictConfig(LOGGING_CONFIG)
    # logging.getLogger("transformers").setLevel(logging.ERROR)
    # logging.getLogger("transformers").setLevel(logging.DEBUG)

    # Logger setup.
    # log = logging.getLogger(__name__)
    # log.debug("Logging is configured.")

    model, dataset_reader = run_training()
