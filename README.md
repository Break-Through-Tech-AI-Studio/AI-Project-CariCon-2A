# Personality-Based Career Prediction Model (CariCon 2A) 
**AI Studio Final Project – Break Through Tech**

This project explores how natural language processing (NLP) and supervised machine learning can be used to predict personality traits from text data. Using social media posts labeled with Myers-Briggs Type Indicator (MBTI) personality types, we built and evaluated multiple models to understand how linguistic patterns correlate with personality dimensions.

Our final approach reframes MBTI prediction as four binary classification problems (I/E, N/S, T/F, J/P), leading to significantly improved performance compared to direct 16-class classification.

---

## 👥 Team Members

| Name | GitHub Handles| Affiliation | Role & Contributions |
|---|---|---|---|
| **Joseann Boneo** | @J-O-S-I-E | New York Institute of Technology | NLP pipline roadmapping, data exploration, EDA, model training results analysis, documentation, feature encoding |
| **Hodan Ali** | @hodanali7 | Smith College | Data exploration, EDA, results analysis, documentation, model training, visualization |
| **Trinity Dhillon** | @tdhillon113 | Fordham University | Feature engineering, model training, hyperparameter tuning, performance analysis, results analysis, documentation |
| **Sumaiyah Rahman** | @sumaiyahr2004 | Columbia University | Data preprocessing, NLP pipeline development, results analysis, documentation, model training, hyperparameter tuning, performance analysis |

---
## Project Highlights

- Built a **supervised NLP pipeline** to predict MBTI personality traits from text
- Compared **16-class classification vs. binary classification approaches**
- Achieved **~82% accuracy** on multiple personality axes using **Word + Character TF-IDF with Linear SVM**
- Identified key challenges in predicting **Intuition vs. Sensing (N/S)** due to subtle linguistic differences
- Designed a **reproducible, modular Jupyter Notebook workflow**
- Explored **bias, fairness, and ethical considerations** in personality prediction models

---

## Project Overview

Caricon is a nonprofit organization dedicated to celebrating Caribbean literature, and their motivation for this project was to explore how AI can support more inclusive talent recognition, storytelling, and program matching across the Caribbean diaspora.

This project was completed as part of the **Break Through Tech AI Studio Program**. The challenge focused on building an end-to-end machine learning solution that applies NLP techniques to a real-world problem.

### Problem Statement
How can we accurately predict personality traits from written text using machine learning?

### Objective
- Develop a predictive model for MBTI personality traits
- Learn and apply NLP techniques including TF-IDF vectorization
- Evaluate multiple machine learning models
- Analyze trade-offs between model complexity, accuracy, and interpretability

### Business Impact
Accurate personality prediction models can support:
- Improved career and talent matching
- Program placement and personalization
- Automation of assessment tools
- Inclusive growth across diverse populations

---

## Dataset

**Source:** Myers-Briggs Personality Dataset  
- ~8,675 user-generated text posts  
- Each post labeled with one of **16 MBTI personality types**

**Features:**
- `posts`: Social media text content
- `type`: MBTI personality label

To address class imbalance, additional extraverted posts were incorporated to create a more balanced dataset across personality types.

---

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) was conducted to understand the structure, quality, and distribution of the MBTI personality dataset prior to modeling. This step was critical for identifying potential biases, data imbalances, and linguistic patterns that could impact model performance.


### Key Observations
- **Post Length Consistency:** Average text length was relatively consistent across MBTI types, reducing concerns about length-based bias.
- **Subjective Language:** Posts frequently contained introspective, emotional, and opinion-based language, making the dataset suitable for personality inference.
- **Class Imbalance:** Introverted personality types were overrepresented in the original dataset, which posed a risk of biased predictions.

<p>
  <img width="1072" height="251" alt="Screenshot 2025-12-14 133626" src="https://github.com/user-attachments/assets/aafbe184-0ee0-4aab-b232-fa827c6212f4" />
</p>

### Dataset Balancing
To address imbalance, additional extraverted posts were incorporated into the dataset. After augmentation:
- The distribution across MBTI types became more balanced
- Extraverted types (e.g., ENFP, ESTP, ESFJ) gained stronger representation
- Newly added posts exhibited more action-oriented language and shorter sentence structures
<p>
  <img width="972" height="299" alt="Screenshot 2025-12-14 133636" src="https://github.com/user-attachments/assets/a17a5b07-7a82-41e8-a267-d8419c4d3832" />
</p>

### Impact on Modeling
EDA insights directly informed downstream decisions:
- Motivated the shift from **16-class classification** to **binary classification by MBTI axis**
- Influenced feature engineering choices, including the use of **word- and character-level TF-IDF**
- Helped identify **Intuition vs. Sensing (N/S)** as the most challenging dimension to predict due to subtle linguistic differences


---

## Data Preprocessing

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

## Feature Engineering

- **TF-IDF Vectorization**
  - Word-level TF-IDF
  - Character-level TF-IDF
  - `max_features = 10,000`
  - `ngram_range = (1,3)`
  - `min_df = 5`

---

## Models Used & Technical Approach

### Model Development: Method Selection & Justification
We approached personality prediction as a **supervised text classification problem**. Initial experimentation with direct **16-class MBTI classification** resulted in poor performance due to high class complexity and overlapping linguistic features. Based on insights from EDA, we reframed the task into **four independent binary classification problems** (I/E, N/S, T/F, J/P), which significantly reduced complexity and improved predictive accuracy.

The following models and tools were selected due to their strong performance on high-dimensional text data and interpretability:

- **TF-IDF Vectorization**: Converts text into numerical feature representations while preserving important word and character patterns.
- **Logistic Regression**: Used as a baseline model due to its simplicity, interpretability, and efficiency.
- **Linear Support Vector Machine (SVM)**: Chosen for its ability to handle sparse, high-dimensional feature spaces common in NLP tasks.
- **Word + Character-Level TF-IDF**: Combined to capture both semantic meaning and stylistic writing patterns linked to personality traits.

---

### Technical Architecture & Training Pipeline
The modeling pipeline followed a modular, end-to-end machine learning workflow:

1. **Text Preprocessing**
   - Removal of URLs, emojis, special characters, and stopwords
   - Text normalization (lowercasing and spacing)
2. **Feature Engineering**
   - Word-level TF-IDF vectorization
   - Character-level TF-IDF vectorization
   - One-hot encoding of personality labels
3. **Model Training**
   - Baseline: Logistic Regression + TF-IDF
   - Advanced: Linear SVM + Word TF-IDF
   - Optimized: Linear SVM + Word + Character TF-IDF
4. **Evaluation**
   - Accuracy and Macro-F1 score
   - Comparison across personality axes
   - Performance benchmarking against baseline models

All models were trained and evaluated using **scikit-learn**, with consistent train/test splits to ensure fair comparisons.

---

## Results

| Model | Task | Key Results |
|---|---|---|
| Logistic Regression + TF-IDF | Binary Axes | ~77% accuracy on I/E, T/F, J/P; lower on N/S |
| Linear SVM + Word TF-IDF | Binary Axes | ~78% accuracy overall |
| **Linear SVM + Word + Char TF-IDF** | **Binary Axes** | **Up to ~82% accuracy**, highest Macro-F1 |

#### Linear SVM + Word + Char TF-IDF Results

| Axis | Best Accuracy |
|---|---|
| I / E | ~82% |
| T / F | ~82% |
| J / P | ~80% |
| N / S | ~66% |


#### Key Insights
- **Binary classification consistently outperformed 16-class prediction**
- Word + character-level features significantly improved performance
- N/S remained the most difficult axis due to subtle contextual cues
- Macro-F1 scores improved with richer feature representations, indicating more balanced class performance

---

### Baseline Comparison
The Logistic Regression + TF-IDF model served as a baseline. While effective for simpler axes, it struggled with nuanced personality dimensions. The optimized SVM model demonstrated clear performance gains across all evaluated metrics, validating the chosen architecture and training strategy.

---

## Limitations

- Subtle linguistic differences make N/S classification difficult
- Personality prediction from text raises ethical and fairness concerns
- MBTI labels may oversimplify human personality traits

---

## Next Steps

- Experiment with **Transformer-based models** (e.g., RoBERTa)
- Improve contextual understanding for N/S prediction
- Combine four binary predictions into a final MBTI label
- Conduct deeper bias and fairness evaluations
- Explore deployment as an interactive personality insight tool

---

## Code Highlights

The project is developed primarily in a **Jupyter Notebook**, which serves as the central environment for experimentation, analysis, and documentation. The notebook is organized into clearly labeled sections that follow the end-to-end machine learning lifecycle, enabling transparency and reproducibility.

```
AI-PROJECT-CARICON-2A/
├── .venv/ # Local virtual environment (ignored in Git)
├── .virtual_documents/ # Jupyter/IDE-generated files
├── anaconda_projects/ # Anaconda-related project files
├── dataset/ # Dataset storage directory
│
├── EDA Notebooks/ # Exploratory Data Analysis notebooks
│ ├── EDA-additional_dataset.ipynb
│ └── EDA-original_dataset.ipynb
│
├── mbti_axis_models/ # Trained binary classification models
│ ├── IE_y_svm.joblib # Introversion / Extraversion SVM model
│ ├── JP_y_svm.joblib # Judging / Perceiving SVM model
│ ├── NS_y_svm.joblib # Intuition / Sensing SVM model
│ └── TF_y_svm.joblib # Thinking / Feeling SVM model
│
├── .gitignore # Git ignore rules
├── BTT-AI Interview Prep Guide.pdf
├── FINAL_Caricon_2A_Workspace.ipynb # Final end-to-end modeling notebook
├── Initial Caricon_2A_Workspace.ipynb # Initial experimentation notebook
├── myer-briggs-data.csv # Original MBTI dataset
└── updated_mbti.csv # Balanced and cleaned MBTI dataset
```

In addition to the Jupyter Notebook, **GitHub is used for version control and collaboration**. The repository tracks incremental changes to code, documentation, and experiments, enabling efficient teamwork, maintaining a clear development history, and ensuring reproducibility of results. GitHub also serves as the central hub for project documentation, including the README and dependency specifications.

---

## 🛠️ Setup & Installation

### Requirements
To run this project locally, ensure you have the following installed:
- Python 3.8 or higher
- Jupyter Notebook

### Installation
Install all required Python dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt

```
