import pandas as pd
import numpy as np
import joblib

# Load the model
model = joblib.load('river_discharge_model_tomorrow.pkl')


manual_input = [
    [25.1, 65, 18.0, 1013, 50, 2.8, 21, 160.0, 150.0],  
    # Add more rows as needed
]

# Convert the list to a DataFrame
columns = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'pressure_msl',
    'cloud_cover', 'wind_speed_10m', 'soil_temperature_0_to_7cm',
    'river_discharge', 'river_discharge_past'
]

input_df = pd.DataFrame(manual_input, columns=columns)

# Make predictions
predictions = model.predict(input_df)

# Print the predictions
print("Predicted River Discharge for Tomorrow:", predictions)