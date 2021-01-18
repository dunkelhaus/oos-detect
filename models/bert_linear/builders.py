import logging.config
from typing import Any, Tuple
from allennlp.models import Model
from allennlp.data import DataLoader
from allennlp.data import Vocabulary
from allennlp.training.trainer import Trainer
from models.builders import build_epoch_callbacks
from models.builders import build_batch_callbacks
from allennlp.training.trainer import GradientDescentTrainer
from models.bert_linear.classifier import BertLinearClassifier
from allennlp.training.optimizers import HuggingfaceAdamWOptimizer
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders.bert_pooler import BertPooler
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder

# Logger setup.
log = logging.getLogger(__name__)


def bert_linear_builders(
        serialization_dir: str,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        lr: float,
        num_epochs: int,
        vocab: Vocabulary,
        wbrun: Any
) -> Tuple[Model, Trainer]:
    """
    Simple wrapper for both model-specific builder fns.

    :returns Model, Trainer: The Model & Trainer objects
                                respectively.
    """
    model = build_model(vocab, wbrun)

    trainer = build_trainer(
        model,
        serialization_dir,
        train_loader,
        dev_loader,
        lr=lr,
        num_epochs=num_epochs,
        wbrun=wbrun
    )

    return model, trainer


def build_model(vocab: Vocabulary, wbrun: Any) -> Model:
    """
    Build the Model object, along with the embedder and encoder.

    :param vocab: The pre-instantiated vocabulary object.
    :return Model: The model object itself.
    """
    log.debug("Building the model.")
    # vocab_size = vocab.get_vocab_size("tokens")

    # TokenEmbedder object.
    bert_embedder = PretrainedTransformerEmbedder("bert-base-uncased")

    # TextFieldEmbedder that wraps TokenEmbedder objects. Each
    # TokenEmbedder output from one TokenIndexer--the data produced
    # by a TextField is a dict {names:representations}, hence
    # TokenEmbedders have corresponding names.
    embedder: TextFieldEmbedder = BasicTextFieldEmbedder(
        {"tokens": bert_embedder}
    )

    log.debug("Embedder built.")
    encoder = BertPooler("bert-base-uncased", requires_grad=True)
    # encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(768,20,batch_first=True))
    log.debug("Encoder built.")

    return BertLinearClassifier(vocab, embedder, encoder, wbrun).cuda(0)


def build_trainer(
        model: Model,
        serialization_dir: str,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        lr: float,
        num_epochs: int,
        wbrun: Any
) -> Trainer:
    """
    Build the model trainer. Includes instantiating the optimizer as well.

    :param model: The model object to be trained.
    :param serialization_dir: The serialization directory to output
            results to.
    :param train_loader: The training data loader.
    :param dev_loader: The dev data loader.
    :return trainer: The Trainer object.
    """
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = HuggingfaceAdamWOptimizer(parameters, lr=lr)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        epoch_callbacks=build_epoch_callbacks(wbrun),
        batch_callbacks=build_batch_callbacks(wbrun)
    )

    return trainer
