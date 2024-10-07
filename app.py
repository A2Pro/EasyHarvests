import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import requests
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
from flask import Flask, render_template, request, jsonify
import statistics

load_dotenv()

openAIClient = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_gpt(prompt):
    completion = openAIClient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer the following question to the best of your ability."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def get_precipitation_data(latitude, longitude, start_date, end_date):
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR&community=AG&longitude={longitude}&latitude={latitude}&start={start_date}&end={end_date}&format=JSON"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        try:
            return data['properties']['parameter']['PRECTOTCORR']
        except KeyError:
            print("Unexpected JSON structure:", data)
            return None
    else:
        print("Failed to retrieve data:", response.status_code)
        return None

def predict_precipitation(data):
    valid_dates = []
    values = []

    for date_str, value in data.items():
        try:
            date = pd.to_datetime(date_str, format='%Y%m%d')
            values.append(float(value))
            valid_dates.append(date)
        except (ValueError, TypeError):
            continue
    df = pd.DataFrame({'Date': valid_dates, 'Precipitation': values})
    df.dropna(inplace=True)
    df.set_index('Date', inplace=True)

    model = ARIMA(df['Precipitation'].astype(float), order=(5, 1, 0))
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=30)
    forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Precipitation': forecast})
    forecast_df.set_index('Date', inplace=True)

    predicted_temps = forecast.tolist() 
    temp_forecast_dict = {date.strftime('%Y-%m-%d'): temp for date, temp in zip(forecast_dates, predicted_temps)}

    temps = []
    for date, temp in temp_forecast_dict.items():
        temps.append(temp)

    return forecast_df, temp_forecast_dict, temps

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        data = request.get_json()
        print("Received data:", data) 
        latitude = data.get("latitude")
        longitude = data.get("longitude")

        if latitude is None or longitude is None:
            return jsonify({"error": "Latitude and Longitude are required"}), 400

        start_date = "20240101"
        end_date = "20240630"

        precipitation_data = get_precipitation_data(latitude, longitude, start_date, end_date)

        if precipitation_data:
            forecast_df, temp_forecast_dict, temp_list = predict_precipitation(precipitation_data)
            print(temp_list)
            mean = statistics.fmean(temp_list)
            response = ask_gpt("Here I have a list of rainfall for the next 30 days. Assume I do agriculture. Tell me if there's going to be a lot, a little, any increase or decrease. Tell me what steps I should do to prepare for this. Here's the list: " + str(temp_list)) 
            print(response)
            return jsonify({"message": str(response), "mean" :  mean})  
        else:
            return jsonify({"error": "Failed to retrieve precipitation data"}), 400

    return jsonify({"error": "Invalid request"}), 400

if __name__ == "__main__":
    app.run(debug=True)
