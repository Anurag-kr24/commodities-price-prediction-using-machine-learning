import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
import pickle 


base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Datasets"))

gold_data_path = os.path.join(base_path, "featured_gold_data.csv")
silver_data_path = os.path.join(base_path, "featured_silver_data.csv")


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

gold_data = load_data(gold_data_path)
silver_data = load_data(silver_data_path)


def create_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    return df

gold_data = create_features(gold_data)
silver_data = create_features(silver_data)


def train_xgboost(df, target_col):
    X = df[['year', 'month', 'day', 'weekday']]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"MAE for {target_col}: {mae}")
    
    return model

gold_model = train_xgboost(gold_data, 'price')
silver_model = train_xgboost(silver_data, 'price')


with open('gold_xgboost_model.pkl', 'wb') as f:
    pickle.dump(gold_model, f)
with open('silver_xgboost_model.pkl', 'wb') as f:
    pickle.dump(silver_model, f)

print("Models saved as .pkl files successfully.")


future_days = int(input("Enter the number of days to predict: "))
def predict_future(model, last_date, future_days):
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
    future_df = pd.DataFrame({
        'year': future_dates.year,
        'month': future_dates.month,
        'day': future_dates.day,
        'weekday': future_dates.weekday
    })
    predictions = model.predict(future_df)
    return future_dates, predictions

gold_future_dates, gold_future_prices = predict_future(gold_model, gold_data['date'].iloc[-1], future_days)
silver_future_dates, silver_future_prices = predict_future(silver_model, silver_data['date'].iloc[-1], future_days)


def plot_predictions(df, future_dates, future_prices, title):
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['price'], label='Historical Prices')
    plt.plot(future_dates, future_prices, label='Predicted Prices', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.show()

plot_predictions(gold_data, gold_future_dates, gold_future_prices, "Gold Price Prediction")
plot_predictions(silver_data, silver_future_dates, silver_future_prices, "Silver Price Prediction")


future_gold_df = pd.DataFrame({'Date': gold_future_dates, 'Predicted Gold Price': gold_future_prices})
future_silver_df = pd.DataFrame({'Date': silver_future_dates, 'Predicted Silver Price': silver_future_prices})

print("\nFuture Gold Price Predictions:")
print(future_gold_df.head(future_days))
print("\nFuture Silver Price Predictions:")
print(future_silver_df.head(future_days))

