# Fake News Detection — India & Global News (2020–2026)

A text classification project that labels a news article as **REAL** or **FAKE** using classic NLP + ML: TF-IDF features with Logistic Regression, Passive Aggressive, and Random Forest classifiers.

**Files:**
- `fake_news_detection.ipynb` — full, already-executed notebook (EDA → cleaning → modeling → evaluation → error analysis)
- `fake_news_dataset_india_global_2020_2026.csv` — the dataset (5,000 articles)

Open the notebook in Jupyter or upload both files to Google Colab (same folder) and run top to bottom — every cell has already been executed once so you can also just read the saved outputs directly.

---

## Dataset

5,000 news articles (`title`, `text`, `category`, `date`, `label`, `note`), balanced almost exactly 50/50 between REAL (2,504) and FAKE (2,496), spread across 12 categories (Politics, Health, Sports, Technology, Crime/Law, etc.) and dated 2020–2026, mixing Indian and global stories. No missing values. No meaningful duplicates (9 repeated headlines turned out to be different articles about the same real event, not copy-pastes).

## Approach

1. **EDA** — checked class balance, category balance, missing values/duplicates, title/body length distributions, and date coverage before writing a single line of modeling code.
2. **Caught a data-leakage bug** (see below) before it made it into the "real" pipeline.
3. **Text preprocessing** — lowercase, strip URLs/numbers/punctuation, remove NLTK English stopwords, drop tokens ≤2 chars.
4. **Features** — TF-IDF, unigrams + bigrams, `max_features=5000`, `min_df=2`.
5. **Models** — Logistic Regression, Passive Aggressive Classifier, Random Forest (300 trees), all trained on identical features for a fair comparison.
6. **Evaluation** — accuracy, macro precision/recall/F1, full classification report, confusion matrices, per-category accuracy breakdown, and misclassified-example inspection.

## Results (test set, 1,000 held-out articles, 80/20 stratified split)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Logistic Regression | 0.998 | 0.998 | 0.998 | 0.998 |
| **Passive Aggressive** | **0.999** | **0.999** | **0.999** | **0.999** |
| Random Forest | 0.998 | 0.998 | 0.998 | 0.998 |

All three models converge in the 99.8–99.9% range. Per-category accuracy is ≥98.6% everywhere (worst category: Crime/Law at 98.75%; most categories hit 100%). Only 1–2 articles out of 1,000 are misclassified per model.

## The bug I caught (best interview story here)

My first pass concatenated `title` + `text` + `note` into one feature blob, since `note` is also text and I figured more text = more signal. A quick sanity-check Logistic Regression on that combined feature scored **100.00% accuracy** on the held-out test set.

That number itself was the red flag — a real-world fake-news classifier should never be perfect. I checked what was actually in `note` and found it's a fact-check annotation written by whoever labeled the dataset: **100% of FAKE articles' notes contain the word "false"**, and 92% contain "hoax," while REAL notes almost never do. `note` is effectively the answer key leaking into the input features — information that would never be available before you've already decided whether an article is fake.

**Fix:** dropped `note` entirely and rebuilt the pipeline on `title` + `text` only. This is the same root-cause pattern as the data-leakage bug I found and fixed in my CNN eye-disease detection project (there, patient images leaked across train/test splits; here, the label's own justification text leaked into the input) — a feature that's only available *because* the answer is already known.

## What the model actually learned (and why 99.9% isn't the whole story)

Even without `note`, accuracy stayed at 99.8–99.9%, which is still unusually high for genuine fact-verification. Looking at the Logistic Regression coefficients showed why: the model isn't checking facts, it's detecting **writing style**.

- **FAKE-indicating terms:** `hoax`, `fabricated`, `falsely`, `viral`, `claims`, `claiming`, `leaked`, `alert` — sensational, rumor/tip-off language.
- **REAL-indicating terms:** `announced`, `expansion`, `infrastructure`, `initiative`, `comprehensive`, `demonstrated` — flat, formal, wire-service language.

This tracks with how the dataset appears to have been constructed: FAKE articles are written in a deliberately sensational register and REAL articles in a deliberately neutral one, so a bag-of-words model picks up the register almost immediately. It says little about how the model would handle real misinformation that's *written to sound credible* — which is increasingly how actual fake news operates.

## Error analysis

The handful of mistakes cluster in **Sports** and **Crime/Law** — categories where genuinely REAL headlines (e.g. a celebrity found liable in a defamation suit, a viral sporting moment) can themselves read as sensational, momentarily overlapping the vocabulary the model associates with FAKE. That's a reassuring failure pattern: the model fails exactly where its style heuristic gets genuinely ambiguous, not randomly.

## Honest limitations

1. Near-perfect accuracy reflects a strong, learnable stylistic split baked into this (likely synthetically constructed) dataset — not proof the model can fact-check real-world news.
2. Should be stress-tested on adversarial/out-of-distribution examples (professionally written misinformation, satire, opinion pieces) before trusting these numbers in any real deployment.
3. Article bodies are short (21–68 words) — this is closer to a headline/snippet classifier than a full-article one.

## Predicting on new text

The notebook ends with a `predict_news(title, text)` function that runs any headline/article through the same cleaning + TF-IDF pipeline used for training and returns a label + confidence score, e.g.:

```python
predict_news(
    title="Reserve Bank of India Announces New Digital Payment Guidelines",
    text="The RBI issued updated guidelines for digital payment providers, focusing on security..."
)
# -> {'label': 'REAL', 'confidence': 0.7845, ...}
```

The final cell wraps this in an `input()` loop so you can type in your own headlines interactively — run it as its own cell in Jupyter/Colab (it's intentionally left un-executed in the saved notebook since it waits on live input).

## Tech stack

Python, pandas, NumPy, scikit-learn, NLTK (stopwords), matplotlib, seaborn.

## How to run

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
python -c "import nltk; nltk.download('stopwords')"
jupyter notebook fake_news_detection.ipynb
```

Or upload `fake_news_detection.ipynb` and `fake_news_dataset_india_global_2020_2026.csv` to the same Google Colab folder and run all cells (Colab already has pandas/scikit-learn/matplotlib preinstalled; you'll still need the two lines above to install NLTK's stopwords).
