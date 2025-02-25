# News Classification Project

## Overview
This project is a **News Classification** system that uses **Natural Language Processing (NLP)** techniques to classify news articles into different categories. The model is trained using the **Naïve Bayes classifier** and **TF-IDF vectorization** to transform text data into numerical features.

## Features
- Cleans and preprocesses text data
- Encodes labels for classification
- Splits data into training and testing sets
- Uses **TF-IDF Vectorization** for feature extraction
- Trains a **Multinomial Naïve Bayes** model for classification
- Evaluates model performance using accuracy and classification report
- Classifies new unseen news articles

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install pandas numpy scikit-learn
```

## Dataset
- The dataset should be a CSV file containing news articles.
- The file should have at least two columns:
  - `text`: The actual news content.
  - `label`: The category of the news article.

## File Structure
```
|-- news_classification.py  # Main script
|-- news_dataset/           # Directory containing dataset
    |-- news.csv            # News dataset file
|-- README.md               # Project documentation
```

## Usage

### 1. Run the Script
Execute the script to train and evaluate the model:
```bash
python news_classification.py
```

### 2. Classify New Documents
The script includes an option to classify new unseen news articles. Modify the `new_documents` list with new articles to test the model's predictions.

## Code Explanation

### Step 1: Import Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
```

### Step 2: Load and Clean Data
- Loads the dataset from CSV.
- Removes punctuation and stopwords.

### Step 3: Encode Labels & Split Data
- Encodes categorical labels into numerical format.
- Splits the dataset into training (80%) and testing (20%) sets.

### Step 4: Train the Model
- Uses **TF-IDF Vectorizer** and **MultinomialNB** in a pipeline.
- Trains the model on the training data.

### Step 5: Evaluate the Model
- Predicts on the test set.
- Prints accuracy and classification report.

### Step 6: Classify New Documents
- Cleans and predicts the category for new unseen news articles.

## Example Output
```plaintext
Accuracy: 85.34%

Classification Report:
               precision    recall  f1-score   support

     Business       0.88      0.84      0.86       200
      Sports       0.83      0.87      0.85       180
      Science       0.84      0.82      0.83       150

Document: Breaking news: Scientists discovered new species of fish!
Predicted Label: Science
--------------------------------------------------
Document: Exclusive: This new gadget will revolutionize the tech world.
Predicted Label: Business
--------------------------------------------------
```

## Conclusion
This project demonstrates a basic text classification pipeline using machine learning and NLP. It can be extended with more advanced models like deep learning for better performance.

## License
This project is open-source and available for use under the MIT License.
