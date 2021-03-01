import wandb
import logging
from typing import Any
from typing import Dict
from typing import Tuple
from pathlib import Path
from typing import Callable
from allennlp.models import Model
from allennlp.data import DatasetReader
from allennlp.training.util import evaluate
from oos_detect.models.builders import build_vocab
from oos_detect.utilities.filesystem import empty_dir
from oos_detect.models.builders import build_data_loader
from oos_detect.utilities.filesystem import locate_results_dir
from oos_detect.models.builders import build_train_data_loaders

# Logger setup.
# from configs.log.log_conf import LOGGING_CONFIG
log = logging.getLogger(__name__)


def run_training(
        data_reader: DatasetReader,
        data_paths: Tuple[Path, Path],
        model_builder: Callable,
        run_name: str,
        hyperparams: Dict[str, Any],
        clear_serialization_dir: bool = False
) -> Model:
    wbrun = wandb.init(
        project="oos-detect",
        sync_tensorboard=False,
        name=run_name,
        config=hyperparams
    )
    print("Running over training set.")
    # wandb.tensorboard.patch(save=True, tensorboardX=False)
    train_path, val_path = data_paths

    # wbconf = wandb.config

    # wbconf.no_cuda = False
    # wbconf.log_interval = 10
    # log.debug(f"WandB config: {wandb.config!r}")

    vocab = build_vocab(from_transformer=True)

    print(f"\nVocab size (num tokens): "
          f"{vocab.get_vocab_size('tokens')}")
    print(f"Vocab check: {vocab}. Post init.")

    train_loader, val_loader = build_train_data_loaders(
        data_reader=data_reader,
        train_path=train_path,
        val_path=val_path,
        batch_size=hyperparams["batch_size"]
    )

    # This is the allennlp-specific functionality in the Dataset
    # object; we need to be able convert strings in the data to
    # integers, and this is how we do it.
    # Note: changed for v2.1.0 of allennlp, since the
    # AllennlpDataset has now been removed. Dataloaders have
    # an index_with function instead.
    vocab.extend_from_instances(train_loader.iter_instances())
    vocab.extend_from_instances(val_loader.iter_instances())
    print(f"Vocab check: {vocab}. Post extending.")

    train_loader.index_with(vocab)
    val_loader.index_with(vocab)
    print(f"Vocab check: {vocab}. Post indexing.")

    # Locate serialization directory.
    serialization_dir = locate_results_dir()

    if clear_serialization_dir:
        empty_dir(serialization_dir)

    model, trainer = model_builder(
        serialization_dir,
        train_loader,
        val_loader,
        hyperparams["lr"],
        hyperparams["num_epochs"],
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
        data_reader: DatasetReader,
        data_path: Path,
        model: Model
) -> Model:
    print("Running over test set.")

    test_loader = build_data_loader(
        data_reader=data_reader,
        data_path=data_path,
        batch_size=8,
        shuffle=False
    )
    model.vocab.extend_from_instances(test_loader.iter_instances())
    test_loader.index_with(model.vocab)

    results = evaluate(model, test_loader, cuda_device=0)
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
