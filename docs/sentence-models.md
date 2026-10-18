# Sentence models

The supported models use plain PyTorch. They do not import AllenNLP. All model
parameters, input tensors, word tables, random noise, and model tests require
CUDA. There is no CPU fallback. The parent trainer owns optimizers and parameter
freezing. These modules do not update parameters.

## Dimensions and word tables

| Setting | Paper profile | Tiny example |
| --- | --- | --- |
| AE word dimension | 100 | 8 |
| Sentence code dimension | 100 | 7 |
| Decoder hidden dimension | 100 | 7 |
| CNN word dimension | 100 | 8 |
| CNN widths | 2, 3, 4, 5 | 2, 3, 4, 5 |
| Channels per width | 128 | 4 |
| Intent MLP | 512 → 512 → 512 → 150 | 16 → 12 → 12 → 3 |

The AE encoder and decoder share one trainable word table. Each CNN has its own
table and parameters. Supply CUDA `embeddings=[vocabulary_size, word_dimension]`
to copy pretrained values. Construction copies the values; it does not share the
source tensor. The PAD row starts at zero. Token lookup and expected embeddings
exclude PAD gradients. Use no weight decay if the optimizer must preserve that
row exactly. Do not edit its values after construction.

The CNN uses parallel registered `Conv1d` layers, ReLU, and masked maximum
pooling. A valid window must contain only valid tokens. Short inputs receive
right padding to the largest kernel width. A branch with no valid window returns
a zero pooled vector. Thus, an all-PAD input has finite logits from the MLP
biases. A one-token input has no valid width-2/3/4/5 window. This policy avoids
padding-dependent features. The MLP has three affine layers and two ReLU layers.

## Construction

```python
from pseudo_ood_generation.components.ae.pog_ae import POGAutoEncoder
from pseudo_ood_generation.components.auxilliary_classifier import SentenceCNN

# Paper dimensions. Replace vocabulary sizes with the prepared artifact sizes.
ae = POGAutoEncoder(vocab_size=5800, device="cuda:0")
aux = SentenceCNN(vocab_size=8300, device="cuda:0")

# Non-paper smoke dimensions.
tiny_ae = POGAutoEncoder(11, embedding_dim=8, code_dim=7, device="cuda:0")
tiny_aux = SentenceCNN(13, embedding_dim=8, num_classes=3,
                       out_channels=4, mlp_hidden_size=12, device="cuda:0")
```

Vocabulary sizes above are examples, not fixed paper counts. `SentenceCNN`,
`POGAuxiliaryClassifier`, and the original `POGAuxilliaryClassifier` name refer
to the same class. Construct a new CNN for the final classifier. Do not copy the
auxiliary checkpoint into it.

## Encoder and reconstruction

`ae.encode(tokens)` accepts right-padded CUDA long IDs `[B,T]` and returns clean
codes `[B,C]`. It selects the final valid encoder output with CUDA gather. It
does not pack sequences or copy lengths to the CPU. All-PAD rows return zero.
Internal or left padding is an error.

`ae(tokens, add_noise=True, temperature=1.0, generator=None)` returns a dictionary:

- `codes`, `noisy_codes`: `[B,C]`.
- `logits`, `probabilities`: `[B,T-1,V]`.
- `mask`: boolean `[B,T-1]`, true where the next-token target is not PAD.
- `loss`: scalar mean cross entropy over non-PAD targets.

Inputs include BOS and EOS. The decoder receives `tokens[:, :-1]`; targets are
`tokens[:, 1:]`. The code passes through one affine layer and tanh to initialize
both decoder LSTM states. The decoder then predicts each next token from the
previous word and recurrent state. This conditioning method is an engineering
assumption. The decoder and encoder each use one unidirectional LSTM layer.

Reconstruction adds independent standard Gaussian noise to clean codes by
default. `noise_std` controls its scale. `add_noise=False` or `noise_std=0`
disables only this noise. Noise is explicit, including in evaluation mode; clean
encoding never adds noise. Supply a CUDA `torch.Generator` for a separate random
stream. All-PAD targets produce a differentiable zero loss.

Teacher-forced logits are the default source for auxiliary training. Use
`aux.forward_soft(output["logits"], output["mask"], temperature, token_mapping)`.
The loss retains the AE graph. The parent must step only the auxiliary optimizer
and then clear unused AE gradients.

## Rollout modes and EOS policy

`ae.rollout(codes, max_steps=63, temperature=1.0, mode="soft", generator=None)`
returns `logits`, `probabilities` of shape `[B,S,V]` and a boolean `mask` `[B,S]`.
All modes start from BOS. PAD and BOS are prohibited output tokens. EOS remains
eligible. Returned logits are raw vocabulary projections; returned probabilities
apply the output restriction and temperature.

- `soft`: Feed `q @ E_AE` to the next decoder step, where `q` is temperature-scaled
  softmax. Run all steps and use an all-true mask. EOS probability does not stop
  or mask the rollout. There is no argmax or hard EOS decision in this path.
- `sample`: Sample each next token. Include the first EOS in the mask and set
  later tokens to PAD. Return `tokens`, masked `log_probs` `[B,S]`, and their sum
  `sequence_log_probs` `[B]`. These log probabilities retain code gradients for
  REINFORCE. Sampling itself is discrete, not a straight-through estimator.
- `greedy`: Use argmax with the same discrete EOS policy. This is not the entropy
  training path.

All modes return fixed tensor shapes. Discrete modes continue internal computation
after EOS but mask it out. A 64-token cap including initial BOS permits 63 output
steps, including EOS when present. If EOS is absent at the last step, the export
layer must record truncation. Text removal of boundaries and empty-output retries
belong to the export layer.

## Auxiliary vocabulary mapping and gradient ownership

`aux.forward(tokens, mask=None)` accepts classifier-vocabulary IDs and returns
`[B,K]` logits. `aux.forward_embeddings(embeddings, mask)` accepts `[B,S,E]`.
`aux.forward_soft(logits, mask, temperature=1.0, token_mapping=None)` applies
softmax and the expected table lookup.
`aux.forward_probabilities(probabilities, mask, token_mapping=None)` accepts
probabilities that already include temperature and rollout restrictions. Do not
apply temperature a second time to rollout probabilities.

`token_mapping` is a CUDA long tensor `[V_generator]`. Each entry selects the
corresponding auxiliary table row. The expected embedding is
`q @ E_aux[token_mapping]`. Duplicate mapping IDs are valid. Without a mapping,
the probability vocabulary must match the table vocabulary.

For generator entropy, freeze AE and CNN parameters with `requires_grad_(False)`
but retain gradient recording. Pass generated codes through soft rollout and
`forward_probabilities`. Minimize `mean(sum(p * log(p)))` over intent classes.
Do not use `no_grad`, detach the codes, or use hard tokens in this path. For
REINFORCE, use sampled `sequence_log_probs` and a detached reward with the
trainer-owned baseline. The trainer owns baseline state and gradient clearing.

`annealed_temperature(step, total_steps, start=1.0, floor=0.1)` in
`components.sentence_utils` provides a linear schedule with a positive floor.
Invalid temperatures fail before softmax.

## Acceptance checks

`tests/reproduction/test_sentence_models.py` covers the paper and tiny shapes,
masked CE, right-padding invariance, all-PAD inputs, shared and independent word
tables, tiny no-noise overfit, seeded noise and sampling, EOS masks, CNN parameter
updates, mapped expected embeddings, soft feedback, and frozen-model code
gradients for soft and sampled training paths. Run these tests only on CUDA.

The Colab T4 run passed all 14 tests in 5.04 seconds with Python 3.13.15 and
PyTorch 2.11.0+cu130. This is an earlier supplemental check. The final isolated
Python 3.12.14/PyTorch 2.14.1+cu132 T4 run passed all 258 reproduction tests,
including these checks. See the [acceptance report](acceptance-report.md).
