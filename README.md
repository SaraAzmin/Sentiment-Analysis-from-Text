# Sentiment Analysis of YouTube Comments

A comparative study of Machine Learning approaches for sentiment classification of real-world YouTube comments.

---

## Project Overview

This project implements and compares multiple Machine Learning techniques for sentiment analysis on YouTube video comments. Comments were extracted directly from YouTube using the YouTube Data API, manually labelled, and used to train and evaluate different models — ranging from traditional ML classifiers to neural networks.

The goal was to determine which approach performs best on informal, noisy, real-world text data such as social media comments.

---

## Dataset

- **Source:** YouTube comments from a real video using the YouTube Data API
- **Collection:** Custom Python script (`Comment_Extraction.py`) using the YouTube Data API v3
- **Labelling:** Manually labelled into sentiment classes (Positive / Negative / Neutral)
- **Files:** `comments.csv` (raw), `Labelled_Dataset.csv` (processed & labelled)

---

## Approaches Compared

### 1. Traditional ML Models (`Traditional_model_comparison.ipynb`)
Baseline classifiers trained on bag-of-words features:
- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest

### 2. TF-IDF with ML Classifiers (`TF_IDF_Final_Khalid.ipynb`)
TF-IDF vectorization combined with ML classifiers to capture term importance across the corpus, improving on simple bag-of-words representations.

### 3. Neural Network (`Neural_Network.ipynb`)
A deep learning approach using a neural network trained on word embeddings to capture semantic context within comments.

---

## Tech Stack

- **Language:** Python
- **Libraries:** scikit-learn, TensorFlow/Keras, pandas, NumPy, NLTK, matplotlib
- **Tools:** Jupyter Notebook, YouTube Data API v3
- **Environment:** Google Colab / Jupyter

---

## Project Structure

```
├── Comment_Extraction.py           # YouTube API comment scraper
├── comments.csv                    # Raw extracted comments
├── Labelled_Dataset.csv            # Manually labelled dataset
├── Traditional_model_comparison.ipynb  # Baseline ML classifiers
├── TF_IDF_Final_Khalid.ipynb       # TF-IDF based approach
├── Neural_Network.ipynb            # Deep learning approach
└── README.md
```

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/SaraAzmin/Sentiment-Analysis-from-Text.git
cd Sentiment-Analysis-from-Text
```

2. Install dependencies:
```bash
pip install scikit-learn pandas numpy nltk tensorflow matplotlib
```

3. Open any notebook in Jupyter or Google Colab and run all cells.

> To extract new comments, add your YouTube Data API key to `Comment_Extraction.py` before running.

---

## Key Learnings

- Real-world social media text is noisy and informal, making preprocessing critical
- TF-IDF significantly outperformed simple bag-of-words on this dataset
- Neural networks require more data to outperform traditional ML on small datasets
- Manual labelling introduces subjectivity, especially for neutral comments

---
