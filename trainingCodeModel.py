import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# Load the datasets
weather_data = pd.read_csv('weather_data.csv')
river_data = pd.read_csv('river_data.csv')

# Preprocess the weather data
weather_data['time'] = pd.to_datetime(weather_data['time'])
weather_data.set_index('time', inplace=True)

# Preprocess the river discharge data
river_data['time'] = pd.to_datetime(river_data['time'])
river_data.set_index('time', inplace=True)

# Drop the 'rain' column from weather data
weather_data.drop(columns=['rain'], inplace=True)

# Resample weather data to daily averages
daily_weather = weather_data.resample('D').mean()

# Merge the daily weather data with river discharge data
merged_data = daily_weather.join(river_data)

# Impute missing values (forward fill)
merged_data.ffill(inplace=True)

# Drop rows with remaining missing values
merged_data.dropna(inplace=True)

# Split features (X) and target (y)
X = merged_data.drop(columns=['river_discharge_tomorrow'])
y = merged_data['river_discharge_tomorrow']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Hyperparameter tuning using RandomizedSearchCV
param_distributions = {
    'n_estimators': np.arange(100, 501, 50),  # Range of estimators
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
}

# Create the RandomizedSearchCV object
rf_tuned = RandomizedSearchCV(rf, param_distributions, n_iter=20, cv=5,
                               random_state=42, n_jobs=-1, verbose=2)

# Train the model
rf_tuned.fit(X_train, y_train)

# Make predictions
predictions = rf_tuned.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'Mean Absolute Error: {mae:.2f}')
print(f'R² Score: {r2:.2f}')

# Plot actual vs predicted river discharge for tomorrow
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual (Tomorrow)', color='blue', linewidth=2)
plt.plot(y_test.index, predictions, label='Predicted (Tomorrow)', color='orange', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted River Discharge for Tomorrow', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('River Discharge', fontsize=12)
plt.legend()
plt.grid()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout for better fit
plt.show()

# Feature importance
importance = rf_tuned.best_estimator_.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

# Plot feature importances
plt.figure(figsize=(10, 5))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.title('Feature Importances', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.grid(axis='x')
plt.tight_layout()
plt.show()

# Save the model for future use
joblib.dump(rf_tuned, 'river_discharge_model_tomorrow.pkl')
