from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pandas as pd
import joblib
import requests

app = Flask(__name__)
cors = CORS(app)

# Load the model
model = joblib.load('river_discharge_model_tomorrow.pkl')

# Function to fetch hourly weather data from the API
def fetch_hourly_weather_data(latitude, longitude):
    latitude = 14.3304
    longitude = 100.5296
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,cloud_cover,wind_speed_10m,soil_temperature_0cm&timezone=Asia%2FBangkok&forecast_days=1"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        # Extract today's and past data (24 hours each)
        hourly_data_today = { 
            'temperature_2m': data['hourly']['temperature_2m'][:24],
            'relative_humidity_2m': data['hourly']['relative_humidity_2m'][:24],
            'dew_point_2m': data['hourly']['dew_point_2m'][:24],
            'pressure_msl': data['hourly']['pressure_msl'][:24],
            'cloud_cover': data['hourly']['cloud_cover'][:24],
            'wind_speed_10m': data['hourly']['wind_speed_10m'][:24],
            'soil_temperature_0cm': data['hourly']['soil_temperature_0cm'][:24]
        }
        
        return hourly_data_today
    else:
        raise Exception("Failed to fetch data from the API")

# Function to get the last available data point for each weather parameter
def get_latest_weather_data(hourly_data):
    # Get the last value from each parameter's list
    latest_data = {key: value[-1] for key, value in hourly_data.items()}
    return latest_data

# Function to calculate averages for all parameters
def calculate_averages(hourly_data):
    # Convert lists to pandas Series and calculate the mean for each parameter
    averages = {key: pd.Series(value).mean() for key, value in hourly_data.items()}
    return averages

# Function to fetch river discharge data from the API
def fetch_river_discharge_data(latitude, longitude):
    latitude = 14.3304
    longitude = 100.5296
    url = f"https://flood-api.open-meteo.com/v1/flood?latitude={latitude}&longitude={longitude}&daily=river_discharge&past_days=1&forecast_days=7"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract river discharge data for the last 7 days
        river_discharge_7day = data['daily']['river_discharge'][-7:]  # Last 7 days (including today)
        
        # Get today's discharge (last element) and yesterday's discharge (second-to-last element)
        river_discharge_today = river_discharge_7day[-1]
        river_discharge_past = river_discharge_7day[-2]  # Yesterday's discharge
        
        return river_discharge_today, river_discharge_past, river_discharge_7day
    else:
        app.logger.error(f"Failed to fetch data from the API: {response.status_code} - {response.text}")
        raise Exception("Failed to fetch data from the API")



@app.route('/')
@cross_origin()
def home():
    return "Welcome to the Flask server!"

@app.route('/api/predict', methods=['GET'])
@cross_origin()
def get_predict():
    # Fetch latitude and longitude from query parameters if needed
    latitude = 14.3304
    longitude = 100.5296
    
    # Get the weather and river discharge data
    river_discharge_today, river_discharge_past, river_discharge_7day = fetch_river_discharge_data(latitude, longitude)
    hourly_data_today = fetch_hourly_weather_data(latitude, longitude)

    # Calculate averages for today
    averages_today = calculate_averages(hourly_data_today)

    app.logger.info(f"river_discharge_7day: {river_discharge_7day}")

    # Prepare input data for today's predictions
    manual_input_today = [
        [averages_today['temperature_2m'], 
         averages_today['relative_humidity_2m'], 
         averages_today['dew_point_2m'], 
         averages_today['pressure_msl'], 
         averages_today['cloud_cover'], 
         averages_today['wind_speed_10m'], 
         averages_today['soil_temperature_0cm'], 
         river_discharge_today, 
         river_discharge_past]
    ]

    # Log the input values for debugging
    app.logger.info(f"Manual input for today's prediction: {manual_input_today}")

    # Define input columns
    columns = [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'pressure_msl',
        'cloud_cover', 'wind_speed_10m', 'soil_temperature_0_to_7cm',
        'river_discharge', 'river_discharge_past'
    ]

    # Create DataFrame for today's input
    input_df_today = pd.DataFrame(manual_input_today, columns=columns)

    # Make prediction for today
    predictions_today = model.predict(input_df_today)

    # Prepare the response data
    result = {
        'predictions_tomorrow': predictions_today.tolist()
    }

    return jsonify(result)

@app.route('/api/river_discharge_7day', methods=['GET'])
@cross_origin()
def get_river_discharge_7day():
    # Fetch latitude and longitude from query parameters or set default values
    latitude = 14.3304
    longitude = 100.5296
    
    try:
        # Get the river discharge data for the last 7 days
        _, _, river_discharge_7day = fetch_river_discharge_data(latitude, longitude)
        
        # Return the river discharge data as JSON
        result = {
            'river_discharge_7day': river_discharge_7day
        }
        
        return jsonify(result)
    
    except Exception as e:
        # If there's an error, return the error message
        return jsonify({'error': str(e)}), 500

# New route to get the latest weather and river discharge data (not averaged)
@app.route('/api/weather/latest', methods=['GET'])
@cross_origin()
def get_latest_weather():
    # Fetch latitude and longitude from query parameters if needed
    latitude = 14.3304
    longitude = 100.5296
    
    # Get the hourly weather data
    hourly_data_today = fetch_hourly_weather_data(latitude, longitude)
    
    # Get the latest available data point for each weather parameter
    latest_weather_data = get_latest_weather_data(hourly_data_today)
    
    # Get the latest river discharge data
    river_discharge_today, river_discharge_past, river_discharge_7day = fetch_river_discharge_data(latitude, longitude)
    
    # Combine both weather and river discharge data
    latest_data = {
        'latest_weather_data': latest_weather_data,
        'latest_river_discharge': river_discharge_today
    }
    
    # Return the latest data as JSON
    return jsonify(latest_data)


if __name__ == '__main__':
    app.run(debug=True)
