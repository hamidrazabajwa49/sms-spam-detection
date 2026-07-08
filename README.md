<div align="center">

# 📱 SMS Spam Detection

### TF-IDF + hand-crafted features → spam/ham classification

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8-4CAF50?style=flat-square)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-4CAF50?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)

**5,572 SMS messages · UCI SMS Spam Collection · 13.4% spam rate**

</div>

---

## 📌 TL;DR

> Classic imbalanced NLP classification: TF-IDF (1-2 grams, 5000 features) combined with 8 hand-crafted stylistic features (length, digit ratio, uppercase ratio, punctuation, currency symbols). Naive Bayes, Logistic Regression, and Random Forest compared; **tuned Logistic Regression wins with F1 0.952, ROC-AUC 0.993.**

<div align="center">

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| Naive Bayes | 0.979 | 0.985 | 0.859 | 0.918 | 0.981 |
| **Logistic Regression** | 0.983 | 0.928 | 0.946 | **0.937** | **0.992** |
| Random Forest | 0.984 | **1.000** | 0.879 | 0.936 | 0.993 |
| **Logistic Regression (tuned)** | **0.987** | 0.966 | 0.940 | **0.952** | 0.993 |

</div>

---

## 🗂️ Table of Contents

- [Dataset](#-dataset)
- [Preprocessing](#-preprocessing)
- [Feature Engineering](#-feature-engineering)
- [Modeling](#-modeling)
- [Results](#-results)
- [Live Inference](#-live-inference-examples)
- [Honest Take](#-honest-take)
- [Project Structure](#-project-structure)
- [Quickstart](#-quickstart)

---

## 📊 Dataset

<details>
<summary><b>UCI SMS Spam Collection v.1 — click to expand</b></summary>

| Attribute | Value |
|---|---|
| Total messages | 5,572 |
| Ham | 4,825 (86.6%) |
| Spam | 747 (13.4%) |
| Source | Grumbletext (UK), NUS SMS Corpus, Caroline Tag PhD corpus, SMS Spam Corpus v0.1 |

The original dataset `readme` (with full provenance and citation) ships alongside the raw file in `data/`.

</details>

## 🧹 Preprocessing

Each message goes through:

1. **Lowercase** — collapses case variants (`Free` / `FREE` / `free`)
2. **Strip URLs and phone numbers** — common spam call-to-action patterns
3. **Strip punctuation and digits** — reduces vocabulary noise
4. **Tokenize + remove stopwords**
5. **Stem** (Porter Stemmer) — `winning/winner/win` → `win`

```
Original : Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005...
Cleaned  : free entri wkli comp win fa cup final tkt st may text fa receiv entri...
```

## ⚙️ Feature Engineering

**Two feature families, combined via `scipy.sparse.hstack`:**

| Family | Details |
|---|---|
| **TF-IDF** | 1–2 grams, top 5,000 terms, `min_df=2`, `sublinear_tf=True` |
| **Hand-crafted (8)** | `msg_length`, `word_count`, `digit_ratio`, `upper_ratio`, `punct_count`, `has_currency`, `exclaim_count`, `question_count` |

Meta features are `MinMaxScaler`-fit on train only, then applied to test — no leakage.

## 🤖 Modeling

<div align="center">

| Model | Features Used | Why |
|---|---|---|
| Naive Bayes | TF-IDF only | Classic text baseline; requires non-negative input |
| Logistic Regression | TF-IDF + meta, `class_weight='balanced'` | Handles the 87/13 imbalance directly |
| Random Forest | TF-IDF + meta, `class_weight='balanced'` | Non-linear baseline |

</div>

Best model selected by **F1** (not accuracy — with an 87/13 imbalance, accuracy is misleading), then tuned with `GridSearchCV` under 5-fold stratified CV.

<details>
<summary><b>Best hyperparameters found</b></summary>

```
Logistic Regression: C=100, penalty='l2'
Best CV F1: 0.9471
```

</details>

## 📈 Results

**Tuned Logistic Regression — held-out test set (1,115 messages):**

```
              precision    recall  f1-score   support

         Ham       0.99      0.99      0.99       966
        Spam       0.97      0.94      0.95       149

    accuracy                           0.99      1115
```

**Notable trade-off across untuned models:** Random Forest hits perfect precision (1.00) but the lowest recall (0.879) — it never flags a real ham message as spam, but it misses more actual spam than Logistic Regression does. Depending on the cost of a false positive (blocking a real message) vs. a false negative (letting spam through), either model could be the right choice in production.

## 💬 Live Inference Examples

| Message | Prediction | Confidence |
|---|:---:|:---:|
| "Congratulations! You've won a FREE iPhone. Click here to claim now!" | 🚨 SPAM | 73.2% |
| "Hey, are we still on for dinner tonight at 7?" | ✅ HAM | 99.99% |
| "URGENT: Your account has been compromised. Call 08001234567 immediately." | 🚨 SPAM | 88.5% |
| "Can you pick up some milk on your way home?" | ✅ HAM | 100.0% |

## 🤔 Honest Take

- **Precision/recall trade-off matters more than any single headline number** — the notebook deliberately compares all three models on both metrics rather than picking a winner by accuracy alone, since false positives and false negatives carry different real-world costs here.
- **Random Forest's perfect precision is a double-edged result** — great for never annoying users with a blocked real message, but it comes at the cost of missing more spam than Logistic Regression. This is a genuine deployment decision, not something to paper over.
- **The confidence scores in the live examples are well-calibrated for clear-cut cases** (near 100% for obvious ham) but the "URGENT... call immediately" example only scored 88.5% — a reminder that adversarial/borderline spam phrasing still leaves real uncertainty.

## 📁 Project Structure

```
sms-spam-detection/
├── data/
│   ├── SMSSpamCollection
│   └── readme                 (original UCI dataset documentation)
├── sms_spam_detection.ipynb
├── README.md
└── requirements.txt
```

## 🚀 Quickstart

```bash
git clone <this-repo>
cd sms-spam-detection
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
jupyter notebook sms_spam_detection.ipynb
```

---

<div align="center">

**Stack:** Python · pandas · NumPy · NLTK · scikit-learn · matplotlib · seaborn

</div>
