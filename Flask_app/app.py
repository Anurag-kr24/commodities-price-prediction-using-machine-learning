from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle 
import os

app = Flask(__name__)


base_path = os.path.abspath(os.path.dirname(__file__))


gold_arima_model_path = os.path.join(base_path, "..", "Models", "ARIMA", "arima_model_gold.pkl")
silver_arima_model_path = os.path.join(base_path, "..", "Models", "ARIMA", "arima_model_silver.pkl")

gold_prophet_model_path = os.path.join(base_path, "..", "Models", "PROPHET", "gold_prophet_model.pkl")
silver_prophet_model_path = os.path.join(base_path, "..", "Models", "PROPHET", "silver_prophet_model.pkl")

gold_xgboost_model_path = os.path.join(base_path, "..", "Models", "XGBoost", "gold_xgboost_model.pkl")
silver_xgboost_model_path = os.path.join(base_path, "..", "Models", "XGBoost", "silver_xgboost_model.pkl")


print(f"Gold ARIMA Model Path: {gold_arima_model_path}")
print(f"Silver ARIMA Model Path: {silver_arima_model_path}")


for model_path in [gold_arima_model_path, silver_arima_model_path, gold_prophet_model_path, silver_prophet_model_path,
                   gold_xgboost_model_path, silver_xgboost_model_path]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")


with open(gold_arima_model_path, "rb") as f:
    gold_arima_model = pickle.load(f)
with open(silver_arima_model_path, "rb") as f:
    silver_arima_model = pickle.load(f)

with open(gold_prophet_model_path, "rb") as f:
    gold_prophet_model = pickle.load(f)
with open(silver_prophet_model_path, "rb") as f:
    silver_prophet_model = pickle.load(f)

with open(gold_xgboost_model_path, "rb") as f:
    gold_xgboost_model = pickle.load(f)
with open(silver_xgboost_model_path, "rb") as f:
    silver_xgboost_model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        days = int(data["days"])  
        model_type = data.get("model", "arima") 

      
        future_dates = pd.date_range(start=pd.Timestamp.today(), periods=days)

        if model_type == "arima":
            
            gold_predictions = gold_arima_model.forecast(steps=days)
            silver_predictions = silver_arima_model.forecast(steps=days)

        elif model_type == "prophet":
            
            gold_future = gold_prophet_model.make_future_dataframe(periods=days)
            silver_future = silver_prophet_model.make_future_dataframe(periods=days)
            gold_predictions = gold_prophet_model.predict(gold_future)["yhat"].tail(days).values
            silver_predictions = silver_prophet_model.predict(silver_future)["yhat"].tail(days).values

        elif model_type == "xgboost":
           
            future_df = pd.DataFrame({
                'year': future_dates.year,
                'month': future_dates.month,
                'day': future_dates.day,
                'weekday': future_dates.weekday
            })
            gold_predictions = gold_xgboost_model.predict(future_df)
            silver_predictions = silver_xgboost_model.predict(future_df)

        else:
            return jsonify({"error": "Invalid model type. Choose between 'arima', 'prophet', or 'xgboost'."})

       
        response = {
            "gold": gold_predictions.tolist(),
            "silver": silver_predictions.tolist(),
            "dates": future_dates.strftime("%Y-%m-%d").tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
