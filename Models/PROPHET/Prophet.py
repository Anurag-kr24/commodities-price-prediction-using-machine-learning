import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os
import pickle 


base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Datasets"))


gold_data_path = os.path.join(base_path, "featured_gold_data.csv")
silver_data_path = os.path.join(base_path, "featured_silver_data.csv")


gold_data = pd.read_csv(gold_data_path, parse_dates=["date"])
silver_data = pd.read_csv(silver_data_path, parse_dates=["date"])


def train_prophet(data, future_days):
    data = data.rename(columns={"date": "ds", "price": "y"})
    data["ds"] = pd.to_datetime(data["ds"])

    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=future_days)
    forecast = model.predict(future)

    all_predictions = forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Predicted Price"})
    last_historical_date = data["ds"].max()
    future_predictions = all_predictions[all_predictions["Date"] > last_historical_date]
    
    return model, forecast, all_predictions, future_predictions


while True:
    try:
        future_days = int(input("Enter the number of days to predict: "))
        if future_days <= 0:
            print("Please enter a positive number.")
            continue
        break
    except ValueError:
        print("Invalid input. Please enter a valid number.")


gold_model, gold_forecast, gold_all_predictions, gold_future_predictions = train_prophet(gold_data, future_days)
silver_model, silver_forecast, silver_all_predictions, silver_future_predictions = train_prophet(silver_data, future_days)


with open('gold_prophet_model.pkl', 'wb') as f:
    pickle.dump(gold_model, f)
with open('silver_prophet_model.pkl', 'wb') as f:
    pickle.dump(silver_model, f)

print("Models saved as .pkl files successfully.")


def plot_forecast(model, forecast, title):
    model.plot(forecast)
    plt.title(title)
    plt.show()


plot_forecast(gold_model, gold_forecast, f"Gold Price Prediction ({future_days} days)")
plot_forecast(silver_model, silver_forecast, f"Silver Price Prediction ({future_days} days)")


print("\n📜 Gold Price Predictions (Past & Future):")
print(gold_all_predictions.to_string(index=False))

print("\n📜 Silver Price Predictions (Past & Future):")
print(silver_all_predictions.to_string(index=False))


print("\n🚀 Future Gold Price Predictions:")
print(gold_future_predictions.to_string(index=False))

print("\n🚀 Future Silver Price Predictions:")
print(silver_future_predictions.to_string(index=False))
