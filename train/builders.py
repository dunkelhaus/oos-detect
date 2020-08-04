import torch
import logging.config
from allennlp.models import Model
from allennlp.data import PyTorchDataLoader
from utilities.locate import locate_oos_data
from typing import List, Dict, Tuple, Iterable
from datasets.readers.oos_eval import OOSEvalReader
from models.single_layer_lstm import SingleLayerLSTMClassifier
from allennlp.training.optimizers import HuggingfaceAdamWOptimizer
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders.bert_pooler import BertPooler
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.data import DatasetReader, DataLoader, Instance, Vocabulary
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder

# Logger setup.
log = logging.getLogger(__name__)


def build_dataset_reader() -> DatasetReader:
    """
    Instantiate dataset reader - factory method.

    :return OOSEvalReader: Instantiated DatasetReader object.
    """
    return OOSEvalReader()


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    """
    Build the Vocabulary object from the instances.

    :param instances: Iterable of allennlp instances.
    :return Vocabulary: The Vocabulary object.
    """
    log.debug("Building the vocabulary.")
    return Vocabulary.from_instances(instances)


def build_model(vocab: Vocabulary) -> Model:
    """
    Build the Model object, along with the embedder and encoder.

    :param vocab: The pre-instantiated vocabulary object.
    :return Model: The model object itself.
    """
    log.debug("Building the model.")
    vocab_size = vocab.get_vocab_size("tokens")
    bert_embedder = PretrainedTransformerEmbedder("bert-base-uncased")
    embedder: TextFieldEmbedder = BasicTextFieldEmbedder(
        {"tokens": bert_embedder}
    )
    log.debug("Embedder built.")
    encoder = BertPooler("bert-base-uncased", requires_grad=True)
    # encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(768,20,batch_first=True))
    log.debug("Encoder built.")
    return SingleLayerLSTMClassifier(vocab, embedder, encoder).cuda(0)


def build_data_loaders(
    train_data: torch.utils.data.Dataset,
    dev_data: torch.utils.data.Dataset,
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Build the AllenNLP DataLoaders.

    :param train_data: The training dataset, torch object.
    :param dev_data: The dev dataset, torch object.
    :return train_loader, dev_loader: The train and dev data loaders as a
            tuple.
    """
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    log.debug("Building DataLoaders.")
    train_loader = PyTorchDataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    dev_loader = PyTorchDataLoader(
        dev_data,
        batch_size=batch_size,
        shuffle=False
    )
    log.debug("DataLoaders built.")
    return train_loader, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    lr: float,
    num_epochs: int
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
        optimizer=optimizer
    )
    return trainer
