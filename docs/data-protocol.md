# CLINC data protocol

## Inputs and splits

Use `oos_detect.reproduce.data`. The module reads these original files:

- `oos_detect/datasets/oos_eval/data/original/data_full.json`
- `oos_detect/datasets/oos_eval/data/original/data_oos_plus.json`

Do not use the converted 151-class files. Each original record must contain
`[text, label]`. The reader checks all six partitions, label sets, and counts.
The plus source must match full-source IND and held-out OOD records in order.

| Partition | Full count | Plus count | Use |
| --- | ---: | ---: | --- |
| IND train | 15,000 | 15,000 | Supervised training |
| IND validation | 3,000 | 3,000 | Model selection |
| IND test | 4,500 | 4,500 | Evaluation |
| OOD train | 100 | 250 | Text source; Dmix uses plus only |
| OOD validation | 100 | 100 | OOD validation |
| OOD test | 1,000 | 1,000 | OOD evaluation |

Each of the 150 IND intents has 100 train, 20 validation, and 30 test records.
Intent IDs follow sorted IND training labels. OOD has `is_ood=true` and
`target=null`; it never becomes an intent class. Test targets are for evaluation
only, not optimization or checkpoint selection.

## Dmix and record identity

`build_dmix(corpus, seed)` samples 10,000 full-source IND training records without
replacement. It adds all 250 plus-source OOD training records, then shuffles with
a private seeded Python RNG. IND overlap with supervised training is intentional.
No validation or test IDs enter Dmix.

IDs have the form `source:partition:index`. Index means the zero-based position
in the original partition. Repeated text is not proof of shared record identity.
`Mixture.records` exposes only `id` and `text`, with no intent or OOD status.
The audit manifest stores ordered IDs and hashes of ordered and sorted IDs.
Explicit smaller counts and custom `SplitContract` values are for non-paper
fixtures and smoke runs only.

## Tokens and vocabulary

Tokenization uses Python Unicode regular expression `\w+|[^\w\s]` after
`str.lower()`. It applies no Unicode normalization. Word runs include digits and
underscores. Each punctuation symbol is a separate token. For example,
`can't` becomes `can`, `'`, `t`. These are chosen assumptions, not paper claims.

Both vocabularies reserve `<pad>`, `<unk>`, `<bos>`, `<eos>` at IDs 0, 1, 2, 3.
Other tokens follow Unicode lexical order. Generator vocabulary uses only full
IND training text. The paper classifier vocabulary uses text from all six full
partitions and plus OOD training. This is a **text-only transductive** policy:
held-out text affects token membership, but held-out labels do not. The manifest
lists each text source. The optional `train-only` profile excludes held-out text
and is not the paper profile. Vocabulary sizes are measured, not forced.

`encode_text` adds BOS and EOS, truncates content to 62 tokens by default, and
right-pads to 64 IDs. EOS is retained. Valid length includes both boundaries.
Unknown tokens map to UNK without vocabulary growth. Encoded records include
original and truncated token counts. The manifest also stores truncation totals.
`generator_to_classifier` maps every generator token ID to the classifier ID
for that same token. Consumers must use this mapping for expected embeddings.

## Embedding source and CUDA

GloVe 6B 100d is a chosen pretrained source, not a source specified by the paper.
Obtain `glove.6B.zip` from the [official GloVe page](https://nlp.stanford.edu/projects/glove/).
Extract `glove.6B.100d.txt` and pass its path to `load_embeddings`. This archive
uses Wikipedia 2014 and Gigaword 5, with 6 billion tokens and uncased vectors.
Do not substitute the newer 2024 vectors without recording a different source.

The official page distributes pretrained vectors under the
[Public Domain Dedication and License v1.0](https://opendatacommons.org/licenses/pddl/1-0/).
The GloVe software has separate license terms. Review the source terms before
redistribution. This project does not download or bundle vectors.

`load_embeddings(vocabulary, path, seed=0, device='cuda:0')` checks every source
line for 100 finite dimensions, hashes the file, and records lexical coverage.
Coverage excludes the four special tokens. Missing and special vectors use a
seeded normal distribution with mean 0 and standard deviation 0.1. PAD is then
set to zero. Every tensor is created on the device returned by `require_cuda`;
there is no CPU fallback. File parsing can use CPU. Same-file, same-vocabulary,
same-seed tensors are repeatable in the same CUDA/software environment.
Cross-version bitwise identity is not promised.

`random_only=True` requires no file and marks the result
`random-only-smoke-non-paper`. It is never an implicit replacement for missing
pretrained vectors. AE, auxiliary, and final models must own independent tables;
the data API returns initial weights, not shared trained model parameters.
Consumers must keep PAD fixed during training.

## Reusable API

```python
from oos_detect.reproduce.data import (
    load_corpus, build_dmix, build_vocabularies, load_embeddings, prepare_data,
)

corpus = load_corpus()
mixture = build_dmix(corpus, seed=0)
vocabularies = build_vocabularies(corpus, profile="paper")
generator = load_embeddings(vocabularies.generator, "/data/glove.6B.100d.txt", seed=0)
classifier = load_embeddings(vocabularies.classifier, "/data/glove.6B.100d.txt", seed=1)
prepared = prepare_data(
    "runs/prepared-seed-0",
    seed=0,
    embedding_manifests={"generator": generator.manifest, "classifier": classifier.manifest},
)
```

The parent CLI can call these functions directly. This module does not add a CLI
or start training. `prepare_data` can run as a text-only stage without embedding
metadata. In that case its manifest states `required-before-training`.
Training consumers must require both pretrained tables with matching vocabulary
and source hashes for paper runs. Embedding weights are rebuilt from the recorded
file and seed; this text preparation stage does not save tensor files.

## Artifact contract, version 1

`prepare_data` requires a new output directory. An existing directory, file,
or symlink causes an error. It never overwrites outputs or changes source files.
An interrupted write can leave a partial directory; preserve it for inspection
and use a new path. `manifest.json` is written last.

| File | Content |
| --- | --- |
| `manifest.json` | Source hashes, contract, seed, policies, split hashes, truncation, embedding metadata, artifact hashes |
| `labels.json` | Sorted intent-to-ID map |
| `split-*.json` | Original full-source records, IDs, separate status and target |
| `dmix.json` | Unlabeled ID/text records |
| `dmix-membership.json` | Dmix IDs, counts, seeds, membership hashes |
| `vocab-generator.json`, `vocab-classifier.json` | Ordered token lists |
| `generator-to-classifier.json` | Ordered classifier IDs for generator IDs |
| `encoded-*.json` | ID, padded IDs, valid length, and truncation counts; no targets |

All files use sorted-key, indented UTF-8 JSON with one final newline. Hashes use
SHA-256. `manifest_sha256` hashes the manifest before that field is inserted.
Other file hashes cover exact bytes. There are no timestamps or absolute source
paths in artifact identity. Two runs with the same inputs and settings in fresh
output directories produce identical files. Consumers can join encoded full
records to split metadata by ID; they must not join Dmix to supervised targets.

## Verification

Use the isolated interpreter through the approved Colab MCP session, not the
local host. Follow [runtime setup](runtime.md), then run from the checkout root:

```sh
snip "$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest tests/reproduction/test_data.py -q
```

For text and file validation only, exclude the CUDA embedding test:

```sh
snip "$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest tests/reproduction/test_data.py -q -k 'not cuda_embedding'
```

Tests create tiny original-format fixtures locally and also validate the checked-out
real files. No test downloads data or runs training.

## Preparation command

Run this command from the repository root in the locked environment:

```bash
snip "$UV_PROJECT_ENVIRONMENT/bin/python" -m oos_detect.reproduce prepare --output prepared/clinc-seed0 --seed 0
```

The output directory must not exist. To compare two preparations, use two new
directories with the same arguments. The command writes text artifacts only.
It does not download embeddings or start training. A paper training run must
load the declared pretrained embeddings before it initializes its models.
