from allennlp
import logging.config
from configs.log.log_conf import LOGGING_CONFIG
from oos_detect.training.builders import build_model
from oos_detect.training.builders import build_vocab
from oos_detect.training.builders import build_trainer
from oos_detect.utilities.locate import locate_oos_data
from oos_detect.training.builders import build_data_loaders
from oos_detect.training.builders import build_dataset_reader


def read_data(
        reader: DatasetReader
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    path = locate_oos_data()
    training_data = reader.read(path/"data_full_train.json")
    validation_data = reader.read(path/"data_full_val.json")
    return training_data, validation_data


def run_training_loop():
    dataset_reader = build_dataset_reader()

    # These are a subclass of pytorch Datasets, with some allennlp-specific
    # functionality added.
    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab)

    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and this
    # is how we do it.
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    # These are again a subclass of pytorch DataLoaders, with an
    # allennlp-specific collate function, that runs our indexing and
    # batching code.
    train_loader, dev_loader = build_data_loaders(train_data, dev_data)

    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this guide, we need to do this.
    trainer = build_trainer(
        model,
        serialization_dir,
        train_loader,
        dev_loader
    )
    trainer.train()

    return model, dataset_reader
