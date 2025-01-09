# app.py
import streamlit as st
import pandas as pd
import pickle

# Load the model and preprocessor
with open('best_model.pkl', 'rb') as f:
    model, preprocessor = pickle.load(f)

# Streamlit app
st.title("California House Price Prediction")

# Input fields
st.sidebar.header("Input Features")
med_inc = st.sidebar.number_input("Median Income (MedInc)", min_value=0.0, value=8.3252)
house_age = st.sidebar.number_input("House Age (HouseAge)", min_value=0.0, value=41.0)
ave_rooms = st.sidebar.number_input("Average Rooms (AveRooms)", min_value=0.0, value=6.984127)
ave_bedrms = st.sidebar.number_input("Average Bedrooms (AveBedrms)", min_value=0.0, value=1.023810)
population = st.sidebar.number_input("Population", min_value=0.0, value=322.0)
ave_occup = st.sidebar.number_input("Average Occupancy (AveOccup)", min_value=0.0, value=2.555556)
latitude = st.sidebar.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.88)
longitude = st.sidebar.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-122.23)

# Create input DataFrame
input_data = {
    "MedInc": med_inc,
    "HouseAge": house_age,
    "AveRooms": ave_rooms,
    "AveBedrms": ave_bedrms,
    "Population": population,
    "AveOccup": ave_occup,
    "Latitude": latitude,
    "Longitude": longitude
}
input_df = pd.DataFrame([input_data])

# Preprocess input data
input_processed = preprocessor.transform(input_df)

# Make prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_processed)
    st.success(f"Predicted Median House Value: {prediction[0]:.4f}")