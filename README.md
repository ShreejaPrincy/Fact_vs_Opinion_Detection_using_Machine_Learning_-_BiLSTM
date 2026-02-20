# Fact vs Opinion Detection using Classical ML & BiLSTM

## 📌 Overview

This project implements and compares multiple machine learning and deep learning models for **Fact vs Opinion classification** at sentence level.

The objective is to determine whether a sentence expresses:

- 🟢 Fact (Objective statement)
- 🔵 Opinion (Subjective statement)

The study evaluates both traditional machine learning models and a Bidirectional LSTM deep learning model, followed by statistical significance testing.

---

## 📂 Dataset

Dataset Used: **MPQA Subjectivity Dataset**

- 5000 Objective sentences (Fact)
- 5000 Subjective sentences (Opinion)
- Balanced binary classification problem

Source:
https://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_3_0/

Files used:
- `plot.tok.gt9.5000` → Objective (Fact)
- `quote.tok.gt9.5000` → Subjective (Opinion)

---

## ⚙️ Project Pipeline

### 1️⃣ Data Loading
- Custom function to load labeled text files
- Combined into single dataframe

### 2️⃣ Text Preprocessing
- Lowercasing
- URL removal
- Special character removal

### 3️⃣ Train-Test Split
- 80% Training
- 20% Testing
- Stratified split

---

## 🧠 Models Implemented

### 🔹 Classical Machine Learning Models

- Naive Bayes (MultinomialNB)
- Logistic Regression
- Linear SVM

Vectorization:
- TF-IDF (max_features=5000, ngrams=(1,2))

---

### 🔹 Deep Learning Model

Bidirectional LSTM Architecture:

- Embedding Layer (15000 vocab, 200 dim)
- BiLSTM (64 units, return_sequences=True)
- Dropout (0.5)
- BiLSTM (32 units)
- Dropout (0.5)
- Dense (Sigmoid)

Training:
- EarlyStopping
- ReduceLROnPlateau
- 15 epochs (with validation split)

---

## 📊 Results

### Accuracy Comparison

| Model | Accuracy |
|-------|----------|
| BiLSTM | **91.70%** |
| Naive Bayes | 91.55% |
| Logistic Regression | 90.00% |
| SVM | 89.20% |

---

### F1 Score Comparison

| Model | F1 Score |
|-------|----------|
| BiLSTM | **0.9177** |
| Naive Bayes | 0.9167 |
| Logistic Regression | 0.9000 |
| SVM | 0.8914 |

---

### 5-Fold Cross Validation (Classical Models)

- Naive Bayes: 90.18%
- Logistic Regression: 89.34%
- SVM: 88.94%

---

### ROC-AUC Comparison

ROC curves were plotted for:
- Naive Bayes
- Logistic Regression
- BiLSTM

All models demonstrated strong separability performance.

---

### Statistical Significance Test (McNemar Test)

Contingency table built between Naive Bayes and BiLSTM.

p-value = 0.8748

Conclusion:
There is **no statistically significant difference** between Naive Bayes and BiLSTM performance.

This indicates that traditional ML remains highly competitive for structured text classification on moderate-sized datasets.

---

## 📈 Evaluation Metrics Used

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC
- 5-Fold Cross Validation
- McNemar Statistical Test

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Statsmodels

---

## 🚀 How to Run

1. Clone repository:

```bash
git clone https://github.com/ShreejaPrincy/Fact_vs_Opinion_Detection_using_Machine_Learning_-_BiLSTM.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run notebook:
```
new_fact_vs_opinion (2).ipynb
```

---

## 🔬 Key Observations

- BiLSTM achieved highest accuracy (91.70%)
- Naive Bayes performed nearly identical
- Statistical test confirms no significant difference
- Deep learning advantage is marginal on moderate-sized datasets

---

## 🚀 Future Work

- Add Attention mechanism
- Use pretrained embeddings (GloVe/FastText)
- Transformer-based models (BERT)
- Multilingual fact-opinion classification

---

## 👩‍💻 Author

Shreeja Princy  

