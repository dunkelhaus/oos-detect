import wandb
import torch
import logging
from typing import Any, Dict
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
                 encoder: Seq2VecEncoder,
                 wbrun: Any):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        log.debug(f"Labels: {num_labels}.")
        self.classifier = torch.nn.Linear(
            encoder.get_output_dim(),
            num_labels
        )
        self.accuracy = CategoricalAccuracy()
        wbrun.watch(self.classifier, log=all)
        log.debug("Model init complete.")

    def forward(self,
                sentence: TextFieldTensors,
                label: torch.Tensor = None
        ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        # log.debug(f"Forward pass starting. Sentence Dict: {sentence!r}")

        mask = util.get_text_field_mask(sentence)
        embedded_text = self.embedder(sentence)
        # Shape: (batch_size, num_tokens)
        # mask = sentence["tokens"]["mask"]

        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        # log.debug(f"Running the classifier. "
        #         f"{mask}")
        logits = self.classifier(encoded_text)
        log.debug("Ran the classifier.")
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {'probs': probs}

        # log.debug(f"Forward pass complete. Probabilities: {probs!r}")

        if label is not None:
            self.accuracy(logits, label)
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
            """log.debug("Calling wandb.log")
            wandb.log({
                "loss": output['loss'],
                "accuracy": self.accuracy.get_metric(reset=False)
            })"""
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
