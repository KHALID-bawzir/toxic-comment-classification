# Toxic Comment Classification

Multi-label classification of online comments into six toxicity categories:
`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

The project is built in stages. Stage 1 is a classical machine learning baseline
(TF-IDF + Logistic Regression). Stage 2 (deep learning and transfer learning) is coming soon.

## Project Structure

```
Toxic_Comment_Classification/
├── notebooks/
│   └── 01_classical_ml_tfidf_logreg.ipynb
├── outputs/
│   └── classical_ml/
│       ├── tfidf_logreg.joblib
│       ├── validation_results.csv
│       ├── test_results.csv
│       └── examples_results.csv
├── data/
│   └── README.md
├── requirements.txt
└── README.md
```

## Data

Wikipedia comments labeled for six types of toxicity (Jigsaw Toxic Comment
Classification dataset). The data is not included in this repository, see
`data/README.md` for the download source.

Key characteristics:
- About 90% of comments are clean, so the classes are highly imbalanced.
- Rare labels such as `threat` and `identity_hate` make the task harder.
- A comment can have several labels at once.

## Pipeline

1. Load the data and drop the `id` column.
2. Explore it: row count, missing values, label distribution, text lengths, label correlation.
3. Split into train / validation / test (80 / 10 / 10) with a fixed random state.
4. Clean the text: remove URLs, mentions, punctuation, and extra whitespace.
5. Vectorize with TF-IDF (10,000 features), fitted on the training set only.
6. Train one Logistic Regression model per label.
7. Evaluate with AUC and F1.
8. Tune a separate decision threshold per label on the validation set.
9. Save the models, vectorizer, and thresholds in one file.

## Evaluation

Accuracy is not used because about 90% of the data is clean: a model that predicts
"clean" for everything would score about 90%.

- **AUC** measures how well the model ranks toxic comments above clean ones (independent of the threshold).
- **F1** balances precision and recall (depends on the threshold).

## Results (test set)

| Label | Threshold | AUC | F1 (0.5) | F1 (tuned) |
|---|---|---|---|---|
| toxic | .. | .. | .. | .. |
| severe_toxic | .. | .. | .. | .. |
| obscene | .. | .. | .. | .. |
| threat | .. | .. | .. | .. |
| insult | .. | .. | .. | .. |
| identity_hate | .. | .. | .. | .. |
| **mean** | | **XX** | **XX** | **XX** |

AUC is high, but F1 at the default 0.5 threshold is much lower. The models are trained
without class weights, so they output low probabilities for rare labels. Tuning the
threshold per label (chosen on the validation set only) recovers a large part of the gap.

## Limitations

- TF-IDF sees words only, without context, so disguised toxicity (no explicit bad words) is often missed.
- Rare labels (`severe_toxic`, `threat`, `identity_hate`) have few training examples.
- Text cleaning removes punctuation and casing information that may carry signal.

## Setup

```bash
pip install -r requirements.txt
```

Python version: 3.XX

Then open `notebooks/01_classical_ml_tfidf_logreg.ipynb` and run all cells.

## Using the saved model

```python
import joblib

saved = joblib.load("outputs/classical_ml/tfidf_logreg.joblib")
models, vec, thresholds = saved["models"], saved["vectorizer"], saved["thresholds"]
```

Apply the same text cleaning used in training before calling `vec.transform`.

## Next Steps

- Notebook 02: deep learning and transfer learning, compared against this baseline.

## License

MIT. The dataset has its own license, see its source page.