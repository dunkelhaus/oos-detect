import torch
import logging
from typing import Dict
from allennlp.nn import util
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.data import TextFieldTensors
from allennlp.modules import Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.token_embedders import TokenEmbedder

# Logger setup.
log = logging.getLogger(__name__)


# @Model.register('single_layer_lstm')
class SingleLayerLSTMClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TokenEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(
            encoder.get_output_dim(),
            num_labels
        )
        self.accuracy = CategoricalAccuracy()
        log.debug("Model init complete.")

    def forward(self,
                sentence: TextFieldTensors,
                label: torch.Tensor = None
        ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        # log.debug(f"Forward pass starting. Sentence Dict: {sentence!r}")

        embedded_text = self.embedder(
            token_ids=sentence["tokens"]["token_ids"],
            mask=sentence["tokens"]["mask"],
            type_ids=sentence["tokens"]["type_ids"],
            segment_concat_mask=sentence["tokens"]["segment_concat_mask"]
        )
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(sentence)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {'probs': probs}

        # log.debug(f"Forward pass complete. Probabilities: {probs!r}")

        if label is not None:
            self.accuracy(logits, label)
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
