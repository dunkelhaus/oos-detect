import os
import sys
import logging
import pandas as pd
from pathlib import Path
from utilities.exceptions import ReqdFileNotInSetError
from dataload import clinc_json_to_df
from allennlp.data.tokenizers import Token
from allennlp.data import DatasetReader, instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import LabelField, TextField
from typing import Dict, List, Tuple, Iterable, Iterator
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerTokenIndexer

# Logger setup.
log = logging.getLogger(__name__)
log.debug("Logging is configured.")


@DatasetReader.register('oos-eval-reader')
class OOSEvalReader(DatasetReader):
    """
    Read the data_full json dataset from the CLINC OOS dataset.
    """

    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = None
    ) -> None:
        """
        Parametrized constructor.

        :param token_indexers: Dict containing token indexer, string.
        """
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {
            "tokens": PretrainedTransformerTokenIndexer()
        }
        self.path = Path(__file__).parent.absolute()/"oos-eval/data/"


    def text_to_instance(
            self,
            tokens: List[Token],
            tags: List[str] = None
    ) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["label"] = label_field

        return Instance(fields)


    def _read(self, set_type: str, set_portion: str) -> Iterator[Instance]:
        """
        AllenNLP DatasetReader read method. Generator for Instances.

        :param set_type: The type of dataset being read - the JSON
                file's name without extension; full, imbalanced, oos_plus, or small.
        :param set_portion: The portion of the dataset being used;
                Whether it is the train, val, or dev set.
        """
        fpath = self.path/(set_type + ".json")

        try:
            fpath.resolve(strict=True)
            assert fpath.is_file()

        except FileNotFoundError as fe:
            print(f"Mentioned set type not found: {set_type}.")
            print(f"Error: {ke!r}.")
            raise ReqdFileNotInSetError()

        else:
            with open(self.path/fpath, "r") as f:
                data_f = json.load(f)
                data_dfs = clinc_json_to_df(data_f)

                for line in f:
                    pairs = line.strip().split()
                    sentence, tags = zip(*(pair.split("###") for pair in pairs))
                    yield self.text_to_instance([Token(word) for word in sentence], tags)
