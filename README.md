# Stock Sentiment Analysis

This project performs sentiment analysis on stock market news headlines using machine learning techniques. The goal is to classify news headlines as positive or negative sentiment and evaluate the performance of different classification models.

## Contents

- **Data Preparation and Cleaning**: Importing and preprocessing the data
- **Feature Extraction**: Transforming text data into features for modeling
- **Model Training**: Training classification models
- **Model Evaluation**: Evaluating models using confusion matrix and classification report

## Requirements

To run this notebook, you will need the following Python libraries:
- `pandas`
- `numpy`
- `sklearn`
- `matplotlib`

You can install these libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Getting Started

1. **Download the Data**

   Ensure you have the dataset `Data.csv`. Update the path in the code to the location where your dataset is stored.

2. **Run the Notebook**

   The notebook is designed to be run in a Jupyter notebook environment, such as Google Colab or Jupyter Notebook. To run the notebook:

   - Open the notebook in Google Colab or Jupyter Notebook.
   - Ensure that all necessary libraries are installed.
   - Execute each cell in the notebook sequentially.

## Data Description

The dataset consists of stock market news headlines with the following columns:
- `Date`: Date of the news headline
- `Label`: Sentiment label (positive or negative)
- 25 columns containing the actual headlines

## Steps Overview

1. **Data Loading and Preprocessing**
   - Load the dataset and split it into training and testing sets based on the date.
   - Clean the headlines by removing non-alphabetic characters and converting text to lowercase.
   - Combine headlines for each row into a single string for text processing.

2. **Feature Extraction**
   - Use `CountVectorizer` to convert text data into a matrix of token counts (bigrams).

3. **Model Training and Prediction**
   - Train a `RandomForestClassifier` and `MultinomialNB` (Naive Bayes) on the training data.
   - Make predictions on the test data.

4. **Model Evaluation**
   - Evaluate model performance using confusion matrices and classification reports.
   - Visualize the confusion matrices.

## Results

The notebook includes evaluation metrics such as accuracy score and classification report for both `RandomForestClassifier` and `MultinomialNB`. The confusion matrices are plotted to visualize the performance of each model.


## Acknowledgments

- This project uses publicly available datasets for stock market news sentiment analysis.
- Special thanks to the developers of the Python libraries used in this project.

Feel free to contribute to this project or use it as a reference for your own work! If you have any questions or issues, please open an issue on this repository.
