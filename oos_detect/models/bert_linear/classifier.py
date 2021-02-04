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


# @Model.register('bert_linear_classifier')
class BertLinearClassifier(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            embedder: TokenEmbedder,
            seq2vec_encoder: Seq2VecEncoder,
            wbrun: Any
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.seq2vec_encoder = seq2vec_encoder
        num_labels = vocab.get_vocab_size("labels")
        log.debug(f"Labels: {num_labels}.")
        self.classifier = torch.nn.Linear(
            seq2vec_encoder.get_output_dim(),
            num_labels
        )
        self.accuracy = CategoricalAccuracy()
        wbrun.watch(self.classifier, log=all)
        log.debug("Model init complete.")

    def forward(
            self,
            example: TextFieldTensors,
            target: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        # log.debug(f"Forward pass starting. Sentence Dict: {sentence!r}")

        mask = util.get_text_field_mask(example)
        embedded_text = self.embedder(example)
        # Shape: (batch_size, num_tokens)
        # mask = sentence["tokens"]["mask"]

        # Shape: (batch_size, encoding_dim)
        encoded_text = self.seq2vec_encoder(
            embedded_text,
            mask
        )
        # Shape: (batch_size, num_labels)
        # log.debug(f"Running the classifier. "
        #         f"{mask}")
        logits = self.classifier(encoded_text)
        log.debug("Ran the classifier.")
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=1)
        # Shape: (1,)
        output = {'probs': probs}

        # log.debug(f"Forward pass complete. Probabilities: {probs!r}")

        if target is not None:
            self.accuracy(logits, target)
            output['loss'] = torch.nn.functional.cross_entropy(logits, target)
            """log.debug("Calling wandb.log")
            wandb.log({
                "loss": output['loss'],
                "accuracy": self.accuracy.get_metric(reset=False)
            })"""
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
