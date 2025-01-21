# Amazon Fine Food Reviews Sentiment Analysis

This project analyzes the sentiment of Amazon Fine Food Reviews using a Naive Bayes classifier. It categorizes reviews into three sentiment classes: Positive, Neutral, and Negative.

---

## Prerequisites

Install the required libraries:

```bash
pip install kaggle pandas matplotlib scikit-learn nltk -q
```

---

## Steps to Run the Project

### Step 1: Upload Kaggle API Key

1. Download your Kaggle API key (`kaggle.json`) from [Kaggle](https://www.kaggle.com/account).
2. Upload the `kaggle.json` file in your environment:

```python
from google.colab import files
files.upload()
```

### Step 2: Configure Kaggle

Move the API key to the proper directory and set the permissions:

```bash
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Download Dataset

Download the Amazon Fine Food Reviews dataset:

```bash
!kaggle datasets download -d snap/amazon-fine-food-reviews
```

### Step 4: Extract Dataset

Unzip the dataset:

```bash
!unzip amazon-fine-food-reviews.zip
```

---

## Sentiment Analysis Workflow

### 1. Load and Simplify Dataset

- Load the dataset and select relevant columns (`Text` and `Score`).
- Simplify scores:
  - Positive: 4 and 5
  - Neutral: 3
  - Negative: 1 and 2

### 2. Preprocess Text

- Convert text to lowercase.
- Remove stopwords and non-alphabetic tokens.

### 3. Vectorize Text

- Use `CountVectorizer` to transform text into numerical format.

### 4. Train and Evaluate Model

- Train a Naive Bayes classifier (`MultinomialNB`) on the processed data.
- Evaluate the model using:
  - Accuracy score
  - Classification report
  - Confusion matrix

### 5. Visualize Results

- Plot the confusion matrix to analyze model performance.
- Display the distribution of sentiments in the dataset.

---

## Key Results

### Example Predictions

```python
print(predict_sentiment("The product is amazing and works perfectly!"))  # Positive
print(predict_sentiment("This is the worst purchase I have ever made."))  # Negative
```

### Accuracy and Classification Report

The model achieves high accuracy in predicting sentiments, with detailed precision, recall, and F1-score for each class.

### Confusion Matrix

Visualizes the performance of the model for each sentiment class.

### Sentiment Distribution

Displays the number of reviews in each sentiment category.

---

## Visualizations

### Confusion Matrix

![Confusion Matrix](#)

### Sentiment Distribution

![Sentiment Distribution](#)

---

## Notes

- Ensure the Kaggle API key is uploaded correctly to access the dataset.
- Preprocessing steps like stopword removal and tokenization are crucial for better model performance.
- The project can be extended with other machine learning models or deep learning techniques for improved accuracy.

---

## License

This project is open-source and available under the MIT License.

