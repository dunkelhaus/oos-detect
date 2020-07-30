import os
import sys
import pandas as pd
from pathlib import Path
from dataload import clinc_json_to_df
from allennlp.data.tokenizers import Token
from allennlp.data import DatasetReader, instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import LabelField, TextField
from typing import Dict, List, Tuple, Iterable, Iterator
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerTokenIndexer


@DatasetReader.register('data-full')
class DataFullReader(DatasetReader):
    """
    Read the data_full json dataset from the CLINC OOS data.
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
        self.token_indexers = token_indexers or {"tokens": PretrainedTransformerTokenIndexer()}


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


    def _read(self, file_path: str) -> Iterator[Instance]:
        """
        AllenNLP DatasetReader read method.

        :param file_path: Path to file being read, as a string.
        """
        with open(file_path, "r") as f:
            data_f = json.load(f)
            data_dfs = clinc_json_to_df(data_f)

            for line in f:
                pairs = line.strip().split()
                sentence, tags = zip(*(pair.split("###") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in sentence], tags)
