import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
with open("Models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


gold_df = pd.read_csv("Datasets/gold_data.csv")
silver_df = pd.read_csv("Datasets/silver_data.csv")


gold_df['date'] = pd.to_datetime(gold_df['date'], errors='coerce')
silver_df['date'] = pd.to_datetime(silver_df['date'], errors='coerce')


gold_df = gold_df.dropna(subset=['date']).sort_values('date')
silver_df = silver_df.dropna(subset=['date']).sort_values('date')


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

gold_df = remove_outliers(gold_df, 'price')
silver_df = remove_outliers(silver_df, 'price')

scaler = MinMaxScaler()
gold_df['price'] = scaler.fit_transform(gold_df[['price']])
silver_df['price'] = scaler.fit_transform(silver_df[['price']])


gold_df.to_csv("Datasets/cleaned_gold_data.csv", index=False)
silver_df.to_csv("Datasets/cleaned_silver_data.csv", index=False)

print("Data preprocessing completed! Cleaned datasets saved.")

