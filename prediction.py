import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load your trained model
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# Load the dataset for reference
@st.cache_data
def load_data():
    return pd.read_csv("./data/tanzania_climate_data.csv")

# Preprocessing function (replicate from your notebook)
def preprocess_data(df):
    # Handle missing values
    df = df.dropna()
    
    # Create datetime feature
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    
    # Feature engineering (add your own features here)
    df['Season'] = df['Month'].apply(lambda x: 'Dry' if x in [6, 7, 8, 9] else 'Wet')
    
    return df

# Main app
def main():
    st.title("🌦️ Tanzania Climate Prediction Dashboard")
    st.write("Predict climate conditions in Tanzania using historical weather data")
    
    # Load model and data
    try:
        model = load_model("tanzania_temp_predictor.pkl")
        df = load_data()
        processed_df = preprocess_data(df)
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return
    
    # Sidebar
    st.sidebar.header("User Input Features")
    st.sidebar.info("Adjust parameters to predict climate conditions")
    
    # User input features
    year = st.sidebar.slider("Year", 2021, 2030, 2023)
    month = st.sidebar.selectbox("Month", range(1, 13), format_func=lambda x: datetime(2023, x, 1).strftime('%B'))
    rainfall = st.sidebar.slider("Total Rainfall (mm)", 0.0, 300.0, 50.0)
    max_temp = st.sidebar.slider("Max Temperature (°C)", 20.0, 40.0, 30.0)
    min_temp = st.sidebar.slider("Min Temperature (°C)", 10.0, 30.0, 20.0)
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Year': [year],
        'Month': [month],
        'Total_Rainfall_mm': [rainfall],
        'Max_Temperature_C': [max_temp],
        'Min_Temperature_C': [min_temp]
    })
    
    # Preprocess input
    input_processed = preprocess_data(input_data)
    
    # Prediction
    if st.sidebar.button("Predict Climate"):
        try:
            # Prepare features (ensure same as training)
            features = input_processed[['Year', 'Month', 'Total_Rainfall_mm', 
                                        'Max_Temperature_C', 'Min_Temperature_C']]
            
            prediction = model.predict(features)
            
            st.subheader("Prediction Result")
            st.success(f"Predicted Average Temperature: **{prediction[0]:.1f}°C**")
            
            # Interpretation
            st.write("### Climate Interpretation")
            if prediction[0] > 28:
                st.warning("⚠️ High temperature expected - Heat advisory")
            elif prediction[0] < 22:
                st.info("ℹ️ Cool temperatures expected - Pack warm clothing")
            else:
                st.success("✅ Moderate temperatures expected - Ideal conditions")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    
    # Data Visualization Section
    st.header("Historical Climate Analysis")
    
    # Time Series Plot
    st.subheader("Temperature Trends Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=processed_df, x='Date', y='Average_Temperature_C', ax=ax)
    plt.xticks(rotation=45)
    plt.ylabel("Average Temperature (°C)")
    plt.title("Historical Temperature Trends")
    st.pyplot(fig)
    
    # Seasonal Analysis
    st.subheader("Seasonal Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Rainfall by Season")
        fig_rain = plt.figure(figsize=(10, 5))
        sns.boxplot(data=processed_df, x='Season', y='Total_Rainfall_mm')
        plt.title("Rainfall Distribution by Season")
        st.pyplot(fig_rain)
    
    with col2:
        st.write("#### Temperature by Season")
        fig_temp = plt.figure(figsize=(10, 5))
        sns.boxplot(data=processed_df, x='Season', y='Average_Temperature_C')
        plt.title("Temperature Distribution by Season")
        st.pyplot(fig_temp)
    
    # Data Table
    st.subheader("Historical Climate Data")
    st.dataframe(processed_df[['Date', 'Average_Temperature_C', 'Total_Rainfall_mm', 
                              'Max_Temperature_C', 'Min_Temperature_C']].tail(10))

if __name__ == "__main__":
    main()