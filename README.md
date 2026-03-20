# Fraud-Detection-System
# 🔍 Real-Time Fraud Detection System

An end-to-end fraud detection pipeline trained on 590,540 real financial transactions using a two-layer detection system with explainable AI.


---

## 📊 Results

| Metric | Score |
|--------|-------|
| ROC AUC Score | 0.9370 |
| Fraud Detection Rate | 81.93% |
| Transactions Trained On | 590,540 |

---

## 🏗️ How It Works

- **Layer 1 — XGBoost** detects known fraud patterns from labelled transactions
- **Layer 2 — Isolation Forest** flags unusual transactions that don't match normal behaviour
- **SHAP** explains why each transaction was flagged in plain English

---

## 🛠️ Tech Stack
Python, XGBoost, Isolation Forest, SHAP, Pandas, Scikit-learn, Streamlit

---

## 📂 Dataset
[IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)

---

## 👩‍💻 Author
**Vaishnavi Patil** — MSc Data Science & Analytics, University of Leeds

