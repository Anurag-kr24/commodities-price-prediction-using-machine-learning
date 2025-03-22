import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os


base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Datasets"))

gold_data_path = os.path.join(base_path, "featured_gold_data.csv")
silver_data_path = os.path.join(base_path, "featured_silver_data.csv")


gold_data = pd.read_csv(gold_data_path, parse_dates=["date"], index_col="date")
silver_data = pd.read_csv(silver_data_path, parse_dates=["date"], index_col="date")

print("Gold and Silver data loaded successfully!")


gold_data = gold_data.sort_index()
silver_data = silver_data.sort_index()


gold_data.index = pd.DatetimeIndex(gold_data.index).to_period("D")
silver_data.index = pd.DatetimeIndex(silver_data.index).to_period("D")


gold_prices = gold_data["price"]
silver_prices = silver_data["price"]

# Order (p, d, q)
gold_order = (5, 1, 2)  
silver_order = (4, 1, 2)  


gold_model = ARIMA(gold_prices, order=gold_order)
gold_fit = gold_model.fit()


silver_model = ARIMA(silver_prices, order=silver_order)
silver_fit = silver_model.fit()


def predict_future_prices(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast


while True:
    try:
        future_days = int(input("Enter the number of days you want to predict (e.g., 30): "))
        if future_days <= 0:
            print("Please enter a valid positive number.")
        else:
            break
    except ValueError:
        print("Invalid input. Please enter a valid number.")


gold_future = predict_future_prices(gold_fit, future_days)
silver_future = predict_future_prices(silver_fit, future_days)


gold_forecast_index = pd.date_range(start=gold_prices.index[-1].to_timestamp(), periods=future_days + 1, freq="D")[1:]
silver_forecast_index = pd.date_range(start=silver_prices.index[-1].to_timestamp(), periods=future_days + 1, freq="D")[1:]

gold_future.index = gold_forecast_index
silver_future.index = silver_forecast_index


print("\nFuture Gold Prices Prediction:")
print(gold_future)

print("\nFuture Silver Prices Prediction:")
print(silver_future)


def plot_forecast(actual, future_forecast, title):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index.to_timestamp(), actual, label="Actual Prices", color="blue")
    plt.plot(future_forecast.index, future_forecast, label="Predicted Prices", linestyle="dashed", color="red")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


plot_forecast(gold_prices, gold_future, "Gold Price Prediction")


plot_forecast(silver_prices, silver_future, "Silver Price Prediction")
