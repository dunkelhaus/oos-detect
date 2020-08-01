from allennlp
from oos_detect.utilities.locate import locate_oos_data
from oos_detect.dataset.readers.oos_eval import OOSEvalReader


def build_dataset_reader() -> DatasetReader:
    """
    Instantiate dataset reader - factory method.

    :return OOSEvalReader: Instantiated DatasetReader object.
    """
    return OOSEvalReader()


def read_data(
        reader: DatasetReader
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    path = locate_oos_data()
    training_data = reader.read(path/"data_full_train.json")
    validation_data = reader.read(path/"data_full_val.json")
    return training_data, validation_data
