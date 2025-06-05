import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import json
import os

# Load model and feature names
@st.cache_resource
def load_model_and_features():
    # Load model
    model_path = 'tanzania_climate_model.joblib'
    feature_path = 'feature_names.json'
    
    if not os.path.exists(model_path):
        st.error("Model file not found!")
        raise FileNotFoundError(f"Model file {model_path} does not exist")
    
    if not os.path.exists(feature_path):
        st.error("Feature names file not found!")
        raise FileNotFoundError(f"Feature file {feature_path} does not exist")
    
    model = load(model_path)
    
    # Load feature names
    with open(feature_path, 'r') as f:
        feature_names = json.load(f)
    
    return model, feature_names

model, feature_names = load_model_and_features()

# Load processed data
data_path = r'C:\Users\hp\OMDENA\capstone-project-1962vickyrena\data\tanzania_climate_data.csv'
df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')

# --- EDA Section ---
st.title('Tanzania Climate Predictions')
st.header('Historical Climate Trends')

# Interactive time period selector
year_range = st.slider(
    "Select Year Range",
    min_value=2000,
    max_value=2020,
    value=(2000, 2020)
)

# Filter data and display chart
filtered_df = df.loc[f"{year_range[0]}":f"{year_range[1]}"]
st.line_chart(filtered_df[['Average_Temperature_C', 'Total_Rainfall_mm']])

# --- Prediction Interface ---
st.header('Make Predictions')

# Create future dates for forecasting
last_date = df.index[-1]
n_months = st.slider('Months to Predict', 1, 24, 12)

if st.button('Generate Forecast'):
    # Prepare future dates
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=n_months,
        freq='MS'
    )
    
    # Create feature matrix for predictions
    future_data = []
    for i in range(n_months):
        # Customize this section with your actual feature logic
        row = {
            'Temp_lag_1': df['Average_Temperature_C'].iloc[-1],
            'Rain_lag_1': df['Total_Rainfall_mm'].iloc[-1],
            'Season_Dry': 1 if future_dates[i].month in [12, 1, 2, 6, 7, 8] else 0,
            # Add all other required features here
        }
        future_data.append(row)
    
    future_df = pd.DataFrame(future_data, index=future_dates)
    
    # Ensure correct feature order
    try:
        future_df = future_df[feature_names]
    except KeyError as e:
        st.error(f"Missing feature in prediction data: {e}")
        st.stop()
    
    # Make predictions
    predictions = model.predict(future_df)
    
    # Display results
    st.subheader('Temperature Forecast')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index[-24:], df['Average_Temperature_C'][-24:], label='Historical')
    ax.plot(future_dates, predictions, label='Forecast', color='red', linestyle='--')
    ax.set_title('Temperature Forecast')
    ax.set_ylabel('Temperature (°C)')
    ax.legend()
    st.pyplot(fig)
    
    # Show forecast table
    st.write("Forecast Details:")
    forecast_df = pd.DataFrame({
        'Date': future_dates.strftime('%Y-%m'),
        'Predicted Temperature (°C)': predictions.round(2)
    })
    st.dataframe(forecast_df.set_index('Date'))