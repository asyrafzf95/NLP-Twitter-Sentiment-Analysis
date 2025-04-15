# 🐦 Natural Language Processing_Twitter Sentiment Analysis

This project performs sentiment classification on tweets to detect toxic content (racist/sexist). It includes data cleaning, EDA, hashtag analysis, and a machine learning model trained on Bag-of-Words features.

---

## 📁 Dataset

- **Name**: Twitter Sentiments
- **Rows**: 31,962
- **Columns**:
  - `id`: Unique ID
  - `label`: 0 (non-toxic), 1 (toxic)
  - `tweet`: Raw tweet text

---

## 🧼 Preprocessing Pipeline

- Removed Twitter handles (`@user`)
- Removed punctuation, numbers, short words
- Tokenized and stemmed words using NLTK
- Combined tokens back into cleaned text

---

## 📊 Exploratory Data Analysis

### 🔠 Word Clouds

Word clouds were generated for both toxic and non-toxic tweets, revealing interesting patterns in vocabulary use.

- **Positive Tweets (non-toxic)**: Words like `thank`, `love`, `life`
- **Negative Tweets (toxic)**: Hashtags like `#tcot`, `#cnn`, `#australia` appeared frequently

### 📈 Hashtag Frequency

Bar charts show top hashtags for both classes:

- **Non-toxic**: `#model`, `#run`, `#motiv`
- **Toxic**: `#tcot`, `#cnn`, `#australia`

This suggests hashtags can help differentiate between tweet sentiments.

---

## 🧠 Machine Learning

### 🔧 Features
- **Vectorization**: Bag of Words (`CountVectorizer`)
  - `max_features=1000`
  - Removed stopwords

### 🤖 Model
- **Logistic Regression**
- **Train/Test Split**: 75/25

### 📊 Performance

| Threshold | F1 Score | Accuracy |
|-----------|----------|----------|
| 0.5       | 0.507    | 94.8%    |
| 0.3       | 0.558 ✅ | 94.3%    |

> Lowering the threshold improved F1 score by catching more toxic tweets, trading a bit of accuracy for better balance.

---

## 🛠️ Dependencies

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `nltk`
- `scikit-learn`
- `wordcloud`

