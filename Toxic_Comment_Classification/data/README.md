# Data

The dataset is not included in this repository.

## Source

Jigsaw Toxic Comment Classification (Wikipedia comments labeled for six types of toxicity),
hosted on Hugging Face:
https://huggingface.co/datasets/jhan21/jigsaw-toxic-comment-classification

The data belongs to its original owners and is not covered by this project's license.

## Files

- `train.csv` (about 69 MB): comments with six binary labels
  (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`)

## Setup

Download `train.csv` from the link above and place it in this folder:

```
data/
└── train.csv
```

The notebook reads it from `../data/train.csv`. Files in this folder are ignored by Git
(see `.gitignore`), except this README.