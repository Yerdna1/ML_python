# predict.py
# Load the saved model and make predictions.

import pickle
import pandas as pd

# Load the model and preprocessor
with open('best_model.pkl', 'rb') as f:
    model, preprocessor = pickle.load(f)

# Example input for prediction
example_input = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.023810,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
}

# Convert input to DataFrame
input_df = pd.DataFrame([example_input])

# Preprocess the input data
input_processed = preprocessor.transform(input_df)

# Make prediction
prediction = model.predict(input_processed)
print(f"Predicted Median House Value: {prediction[0]:.4f}")