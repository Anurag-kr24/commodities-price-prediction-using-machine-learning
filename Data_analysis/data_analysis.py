import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller


gold_df = pd.read_csv('Datasets/cleaned_gold_data.csv')
silver_df = pd.read_csv('Datasets/cleaned_silver_data.csv')


gold_df['date'] = pd.to_datetime(gold_df['date'])
silver_df['date'] = pd.to_datetime(silver_df['date'])

scaler = StandardScaler()
gold_df['scaled_price'] = scaler.fit_transform(gold_df[['price']])
silver_df['scaled_price'] = scaler.fit_transform(silver_df[['price']])


gold_df['30_day_MA'] = gold_df['price'].rolling(window=30).mean()
silver_df['30_day_MA'] = silver_df['price'].rolling(window=30).mean()


gold_df['price_diff1'] = gold_df['price'].diff()
silver_df['price_diff1'] = silver_df['price'].diff()


gold_df['price_diff2'] = gold_df['price_diff1'].diff()
silver_df['price_diff2'] = silver_df['price_diff1'].diff()


def adf_test(series, title):
    result = adfuller(series.dropna())  
    print(f'ADF Test for {title}')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Stationary' if result[1] < 0.05 else 'Non-Stationary')
    print('-' * 50)


adf_test(gold_df['price_diff1'], 'Gold First Differenced Prices')
adf_test(silver_df['price_diff1'], 'Silver First Differenced Prices')


plt.figure(figsize=(14, 6))
sns.lineplot(x=gold_df['date'], y=gold_df['price'], label='Gold Price', color='gold')
sns.lineplot(x=gold_df['date'], y=gold_df['30_day_MA'], label='Gold 30-day MA', color='red')
plt.title('Gold Prices with 30-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize=(14, 6))
sns.lineplot(x=silver_df['date'], y=silver_df['price'], label='Silver Price', color='silver')
sns.lineplot(x=silver_df['date'], y=silver_df['30_day_MA'], label='Silver 30-day MA', color='blue')
plt.title('Silver Prices with 30-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


gold_df.to_csv('Datasets/analyzed_gold_data.csv', index=False)
silver_df.to_csv('Datasets/analyzed_silver_data.csv', index=False)

print("Data analysis completed and saved!")
