import pandas as pd


gold_df = pd.read_csv('Datasets/cleaned_gold_data.csv')
silver_df = pd.read_csv('Datasets/cleaned_silver_data.csv')


gold_df['date'] = pd.to_datetime(gold_df['date'])
silver_df['date'] = pd.to_datetime(silver_df['date'])


gold_df.sort_values('date', inplace=True)
silver_df.sort_values('date', inplace=True)


for lag in range(1, 6): 
    gold_df[f'gold_price_lag_{lag}'] = gold_df['price'].shift(lag)
    silver_df[f'silver_price_lag_{lag}'] = silver_df['price'].shift(lag)


gold_df['gold_MA7'] = gold_df['price'].rolling(window=7).mean()
gold_df['gold_MA30'] = gold_df['price'].rolling(window=30).mean()

silver_df['silver_MA7'] = silver_df['price'].rolling(window=7).mean()
silver_df['silver_MA30'] = silver_df['price'].rolling(window=30).mean()


gold_df['gold_volatility'] = gold_df['price'].rolling(window=7).std()
silver_df['silver_volatility'] = silver_df['price'].rolling(window=7).std()


gold_df.dropna(inplace=True)
silver_df.dropna(inplace=True)


gold_df.to_csv('Datasets/featured_gold_data.csv', index=False)
silver_df.to_csv('Datasets/featured_silver_data.csv', index=False)

print("Feature Engineering Completed! Data saved in 'Datasets' folder.")
