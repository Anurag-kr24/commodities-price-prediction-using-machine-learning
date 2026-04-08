# Commodity Price Prediction using Machine Learning

## Overview
This project focuses on predicting the prices of commodities like gold and silver using Machine Learning. The idea is to use historical data to understand patterns and build a model that can estimate future prices.

It was built as a practical project to explore how ML can be applied to real-world financial data.

---

## Features
- End-to-end machine learning workflow  
- Data preprocessing and feature engineering  
- Exploratory Data Analysis (EDA)  
- Regression-based prediction models  
- Model saving and reuse  
- Flask-based web app for predictions  

---

## Workflow

### Data Preprocessing
The raw data was cleaned and prepared by handling missing values and formatting it properly for model training.

### Feature Engineering
Relevant features were created to improve the performance of the model.

### Exploratory Data Analysis
Trends and patterns in gold and silver prices were analyzed using visualizations.

### Model Building
Regression models were trained on the processed data to predict commodity prices.

### Model Evaluation
Predictions were compared with actual values and improved through iteration.

### Deployment
A simple Flask application was built to make predictions based on user input.

---

## Tech Stack
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Flask  

---

## Project Structure

Data_analysis/
Datasets/
Preprocessing_and_feature_engineering/
Models/
Flask_app/
README.md


---

## How to Run

```bash
git clone https://github.com/Anurag-kr24/commodities-price-prediction-using-machine-learning
cd commodities-price-prediction-using-machine-learning

pip install -r requirements.txt

cd Flask_app
python app.py


Limitations-
Uses only historical data
Does not consider external real-time factors
Accuracy depends on dataset quality
Future Improvements
Use advanced models like LSTM
Add real-time data integration
Improve feature engineering
Deploy as a full web application

Author-
Anurag Kumar
https://github.com/Anurag-kr24
