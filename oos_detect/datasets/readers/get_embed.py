import torch
import numpy as np
from pandas import pd
from typing import List, Tuple
from transformers import BertTokenizer, BertModel

# import logging
# logging.basicConfig(level=logging.INFO)


def get_hidden_states(
        model: BertModel,
        sent_data: np.array
) -> torch.Tensor:
    """
    Grab the hidden state values from the model using the sentence data,
    and return the tensor.

    :param model: A BertModel object, already instantiated.
    :param sent_data: The sentence data extracted from the tokenizer.
    :return hidden_states: The torch.Tensor containing the hidden layer
                    values for each sentence, for all sentences.
    """
    # Disable backprop; not building compute graph
    with torch.no_grad():
        hidden_states = [
            model(sent_tensor[0], sent_tensor[1])[2]
            for sent_tensor
            in sent_data
        ]

        print(f"{len(hidden_states)} sentences have states each.")

        print(f"Number of sentences: {len(hidden_states)}")
        sent_i = 0

        print(f"Number of layers: {len(hidden_states[sent_i])} "
              "(initial embeddings + 12 BERT layers)")
        layer_i = 0

        print(f"Number of batches: "
              f"{len(hidden_states[sent_i][layer_i])}")
        batch_i = 0

        print(f"Number of tokens: "
              f"{len(hidden_states[sent_i][layer_i][batch_i])}")
        token_i = 0

        print(f"Number of hidden units: "
              f"{len(hidden_states[sent_i][layer_i][batch_i][token_i])}")

    return hidden_states


def load_bert() -> BertModel:
    """
    Load the BertModel object, on the GPU (add flag later if needed).

    :return model: The BertModel object, on GPU.
    """
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(
        'bert-base-uncased',
        output_hidden_states=True,
        # Whether the model returns all hidden-states.
    )

    model.to('cuda:0')

    return model


def tokenize_bert(sents: pd.DataFrame) -> List:
    """
    Given a pd.Series containing strings of sentences, returns the
    sentences tokenized using the BertTokenizer, and with segment
    IDs for them, to load into BERT.

    :param sents: The sentences, as a pandas Series.
    :return List: A list of tuples, one per sentence in the Series;
                    each containing a token tensor, and segment tensor.
    """
    # Load pretrained model tokenizer (vocab)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Run for all sentences in the bert tokenizer.
    sent_data = np.array([
        _tokenize_bert_sentence(
            text=sentence,
            tokenizer=tokenizer
        )
        for sentence
        in sents[1]
    ])

    return sent_data


def _tokenize_bert_sentence(
        text: str,
        tokenizer: BertTokenizer
) -> Tuple:
    """
    Given a sentence and a BertTokenizer, tokenizes the text, maps to
    BERT vocab indices, and make the segment IDs for the tokens before
    returning the tensors on GPU (add flag for GPU disable soon).

    :param text: The sentence being tokenized. A single sentence as str.
    :param tokenizer: The BertTokenizer object instantiated.
    :return token_tensor, segments_tensor: The tensors containing the
                    segment IDs and the tokens themselves. On GPU.
    """
    # Split the sentence into tokens.
    tokenized_text = tokenizer.encode(
        text,
        add_special_tokens=True
    )

    # Map the token strings to their vocabulary indices.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the tokens as belonging to sentence "1" (single
    #  sentence).
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors, place on GPU.
    token_tensor = torch.tensor([indexed_tokens]).to('cuda:0')
    segments_tensor = torch.tensor([segments_ids]).to('cuda:0')

    return token_tensor, segments_tensor
