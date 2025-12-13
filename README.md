
# Personality-Based Career Prediction Model  
**AI Studio Final Project – Break Through Tech**

This project explores how natural language processing (NLP) and supervised machine learning can be used to predict personality traits from text data. Using social media posts labeled with Myers-Briggs Type Indicator (MBTI) personality types, we built and evaluated multiple models to understand how linguistic patterns correlate with personality dimensions.

Our final approach reframes MBTI prediction as four binary classification problems (I/E, N/S, T/F, J/P), leading to significantly improved performance compared to direct 16-class classification.

---

## 👥 Team Members

| Name | Affiliation | Role & Contributions |
|---|---|---|
| **Joseann Boneo** | New York Institute of Technology |  |
| **Hodan Ali** | Smith College | |
| **Trinity Dhillon** | Fordham University |  |
| **Sumaiyah Rahman** | Columbia University |  |

---

## 🎯 Project Highlights

- Built a **supervised NLP pipeline** to predict MBTI personality traits from text
- Compared **16-class classification vs. binary classification approaches**
- Achieved **~82% accuracy** on multiple personality axes using **Word + Character TF-IDF with Linear SVM**
- Identified key challenges in predicting **Intuition vs. Sensing (N/S)** due to subtle linguistic differences
- Designed a **reproducible, modular Jupyter Notebook workflow**
- Explored **bias, fairness, and ethical considerations** in personality prediction models

---

## 📌 Project Overview

This project was completed as part of the **Break Through Tech AI Studio Program**. The challenge focused on building an end-to-end machine learning solution that applies NLP techniques to a real-world problem.

### Problem Statement
Can we accurately predict personality traits from written text using machine learning?

### Objective
- Develop a predictive model for MBTI personality traits
- Learn and apply NLP techniques including TF-IDF vectorization
- Evaluate multiple machine learning models
- Analyze trade-offs between model complexity, accuracy, and interpretability

### Real-World Significance
Accurate personality prediction models can support:
- Improved career and talent matching
- Program placement and personalization
- Automation of assessment tools
- Inclusive growth across diverse populations

---

## 📊 Dataset

**Source:** Myers-Briggs Personality Dataset  
- ~8,675 user-generated text posts  
- Each post labeled with one of **16 MBTI personality types**

**Features:**
- `posts`: Social media text content
- `type`: MBTI personality label

To address class imbalance, additional extraverted posts were incorporated to create a more balanced dataset across personality types.

---

## 🧹 Data Preprocessing

Key preprocessing steps included:
- Removal of URLs, emojis, special characters, and stopwords
- Text normalization (lowercasing, whitespace standardization)
- Tokenization and vectorization
- One-hot encoding for personality labels
- Binary splitting of MBTI into four axes:
  - Introversion / Extraversion (I/E)
  - Intuition / Sensing (N/S)
  - Thinking / Feeling (T/F)
  - Judging / Perceiving (J/P)

---

## 🧠 Feature Engineering

- **TF-IDF Vectorization**
  - Word-level TF-IDF
  - Character-level TF-IDF
  - `max_features = 10,000`
  - `ngram_range = (1,3)`
  - `min_df = 5`

Combining word and character features allowed the models to capture both semantic meaning and stylistic patterns in text.

---

## 🤖 Models Used

### Baseline
- **Logistic Regression + TF-IDF**
  - ~77% accuracy on I/E, T/F, J/P
  - Lower performance on N/S (~57%)

### Advanced Models
- **Linear SVM + Word TF-IDF**
- **Linear SVM + Word + Character TF-IDF** (Best Model)

### Key Insight
Direct 16-class MBTI prediction performed poorly. Treating MBTI as **four binary classification tasks** significantly improved accuracy and robustness.

---

## 📈 Results

| Axis | Best Accuracy |
|---|---|
| I / E | ~82% |
| T / F | ~82% |
| J / P | ~80% |
| N / S | ~66% |

- Binary classifiers consistently outperformed 16-class models
- Macro-F1 scores improved with richer feature representations
- N/S remains the most challenging dimension to predict

---

## ⚠️ Challenges & Limitations

- Subtle linguistic differences make N/S classification difficult
- Personality prediction from text raises ethical and fairness concerns
- MBTI labels may oversimplify human personality traits

---

## 🔮 Next Steps

- Experiment with **Transformer-based models** (e.g., RoBERTa)
- Improve contextual understanding for N/S prediction
- Combine four binary predictions into a final MBTI label
- Conduct deeper bias and fairness evaluations
- Explore deployment as an interactive personality insight tool

---

## 🛠️ Setup & Installation

### Requirements
- Python 3.8+
- Jupyter Notebook

### Install Dependencies
```bash
pip install -r requirements.txt
