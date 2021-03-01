import json
import logging
import numpy as np
from pathlib import Path
from allennlp.data import Instance
from allennlp.data import DatasetReader
from allennlp.data.tokenizers import Tokenizer
from typing import Dict, List, Iterator, Tuple
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import LabelField, TextField
from oos_detect.utilities.filesystem import locate_oos_data
from oos_detect.utilities.errors import ReqdFileNotInSetError
from oos_detect.utilities.errors import DataSetPortionMissingError
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer

# Logger setup.
log = logging.getLogger(__name__)


def oos_data_paths(
        set: str
) -> Tuple[Tuple[Path, Path], Path]:
    if set not in {"oos_plus", "imbalanced", "full", "small"}:
        print("Incorrect dataset mentioned, ending.")
        return

    print(f"Retrieving paths for oos_{set} dataset.")
    train_file_name = f"data_{set}_train.json"
    val_file_name = f"data_{set}_val.json"
    test_file_name = f"data_{set}_test.json"

    path = locate_oos_data()
    train_path = path/train_file_name
    val_path = path/val_file_name
    test_path = path/test_file_name

    return (train_path, val_path), test_path


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
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True
        )
        self.token_indexers = token_indexers or {
            "tokens": PretrainedTransformerIndexer(
                model_name="bert-base-uncased",
                max_length=max_length,
                namespace="tokens"
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
        sentence_field = TextField(tokens)
        fields = {"tokens": sentence_field}

        # lab = LabelField(label)
        fields["label"] = LabelField(
            label,
            label_namespace="labels"
        )
        # print(f"Just read: {lab.label}, {type(lab.label)}")

        return Instance(fields)

    def apply_token_indexers(
            self,
            instance: Instance
    ) -> None:
        instance.fields["tokens"].token_indexers = self.token_indexers

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
            print(f"Looking for data in path: {file_path}")
            Path(file_path).resolve(strict=True)
            # assert os.path.isfile(file_path)

        except FileNotFoundError as fe:
            print(f"Mentioned file at path not found: {file_path}.")
            print(f"Error: {fe!r}.")
            raise ReqdFileNotInSetError()

        else:
            with open(file_path, "r") as f:
                data_f = json.load(f)["data"]
                # data = self._clinc_json_to_np(data_f)

                for line in self.shard_iterable(data_f):
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
