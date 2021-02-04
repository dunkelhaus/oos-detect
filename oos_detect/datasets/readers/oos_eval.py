import json
import logging
import numpy as np
from allennlp.data import Instance
from allennlp.data import DatasetReader
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import LabelField, TextField
from oos_detect.utilities.locate import locate_oos_data
from typing import Dict, List, Iterator, Tuple, Iterable
from oos_detect.utilities.exceptions import ReqdFileNotInSetError
from oos_detect.utilities.exceptions import DataSetPortionMissingError
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer

# Logger setup.
log = logging.getLogger(__name__)


def read_oos_data(
        reader: DatasetReader,
        set: str
) -> Tuple[Tuple[Iterable[Instance], Iterable[Instance]], Iterable[Instance]]:
    if set not in {"oos_plus", "imbalanced", "full", "small"}:
        print("Incorrect dataset mentioned, ending.")
        return

    print(f"Reading the oos_{set} dataset.")
    train_data = read_train_data(reader, set)
    test_data = read_test_data(reader, set)

    return train_data, test_data


def read_train_data(
        reader: DatasetReader,
        set: str
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    train_file_name = f"data_{set}_train.json"
    val_file_name = f"data_{set}_val.json"
    print(f"Reading training & validation data from "
          f"{train_file_name} and {val_file_name}.")

    path = locate_oos_data()
    training_data = reader.read(path/train_file_name)
    validation_data = reader.read(path/val_file_name)

    return training_data, validation_data


def read_test_data(
        reader: DatasetReader,
        set: str
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    test_file_name = f"data_{set}_test.json"
    print(f"Reading test data from {test_file_name}.")

    path = locate_oos_data()
    test_data = reader.read(path/test_file_name)

    return test_data


# @DatasetReader.register('oos-eval-reader')
class OOSEvalReader(DatasetReader):
    """
    Read the data_full json dataset from the CLINC OOS dataset.
    """

    def __init__(
            self,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            max_length: int = 512
    ) -> None:
        """
        Parametrized constructor.

        :param token_indexers: Dict containing token indexer, string.
        """
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {
            "tokens": PretrainedTransformerIndexer(
                model_name="bert-base-uncased",
                max_length=max_length
            )
        }
        self.tokenizer = tokenizer or PretrainedTransformerTokenizer(
            model_name="bert-base-uncased",
            max_length=max_length
        )

    def text_to_instance(
            self,
            sentence: str,
            label: List[str] = None
    ) -> Instance:
        tokens = self.tokenizer.tokenize(sentence)
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        # lab = LabelField(label)
        fields["label"] = LabelField(label)
        # print(f"Just read: {lab.label}, {type(lab.label)}")

        return Instance(fields)

    def _read(
            self,
            file_path: str
    ) -> Iterator[Instance]:
        """
        AllenNLP DatasetReader read method. Generator for Instances.

        :param set_type: The type of dataset being read - the JSON
                file's name without extension; full, imbalanced,
                oos_plus, or small.
        :param set_portion: The portion of the dataset being used;
                Whether it is the train, val, or dev set.
        """
        # fpath = self.path/(set_type + ".json")

        try:
            # file_path.resolve(strict=True)
            log.debug(file_path)
            # assert os.path.isfile(file_path)

        except FileNotFoundError as fe:
            print(f"Mentioned file at path not found: {file_path}.")
            print(f"Error: {fe!r}.")
            raise ReqdFileNotInSetError()

        else:
            with open(file_path, "r") as f:
                data_f = json.load(f)["data"]
                # data = self._clinc_json_to_np(data_f)

                for line in data_f:
                    sentence, label = line[0], line[1]
                    yield self.text_to_instance(sentence, label)

    def _clinc_json_to_np(
            self,
            loaded_json: json
    ) -> np.array:
        """
        Convert a particular CLINC JSON file to a numpy array.
        :param loaded_json: JSON object - must be loaded using json.load.
        :param portion: The portion of the dataset requested; train,
                dev, etc.
        :return sentences, labels: A tuple containing sentences, and labels.
        """
        try:
            data = np.array(loaded_json["data"])
        except KeyError as ke:
            print("Invalid data.")
            print(f"Error: {ke!r}.")
            raise DataSetPortionMissingError()

        return data