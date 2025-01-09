# House Price Prediction Project
# A complete machine learning pipeline from data preprocessing to model deployment.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import pickle
from flask import Flask, request, jsonify

# Step 1: Load and Explore the Dataset
def load_and_explore_data():
    # Load dataset
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # Display dataset info
    print("Dataset Head:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())

    # Visualize correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig('correlation_matrix.png')  # Save the plot as an image
    print("Correlation matrix saved as 'correlation_matrix.png'.")

    return df

# Step 2: Data Preprocessing
def preprocess_data(df):
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Split features and target
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipeline
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Apply preprocessing
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor

# Step 3: Model Selection and Training
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(random_state=42),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "R2": r2}
        print(f"{name}: MSE = {mse:.4f}, R2 = {r2:.4f}")

    return results

# Step 4: Hyperparameter Tuning
def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }

    gb_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print("\nBest Parameters:", best_params)

    return best_model

# Step 5: Model Evaluation and Comparison
def evaluate_and_compare(results):
    results_df = pd.DataFrame(results).T
    print("\nModel Comparison:")
    print(results_df)

    results_df.plot(kind='bar', y='R2', legend=False)
    plt.title("Model Comparison (R2 Score)")
    plt.ylabel("R2 Score")
    plt.savefig('model_comparison.png')  # Save the plot as an image
    print("Model comparison plot saved as 'model_comparison.png'.")

# Step 6: Save the Best Model
def save_model(model, preprocessor):
    with open('best_model.pkl', 'wb') as f:
        pickle.dump((model, preprocessor), f)
    print("\nBest model saved as 'best_model.pkl'.")

# Step 7: Deployment (Flask API)
def deploy_model():
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        input_data = preprocessor.transform([data])
        prediction = model.predict(input_data)
        return jsonify({'prediction': prediction[0]})

    print("\nStarting Flask server...")
    app.run(debug=True)

# Main Function
if __name__ == '__main__':
    # Step 1: Load and explore data
    df = load_and_explore_data()

    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    # Step 3: Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Step 4: Hyperparameter tuning
    best_model = hyperparameter_tuning(X_train, y_train)

    # Step 5: Evaluate and compare models
    evaluate_and_compare(results)

    # Step 6: Save the best model
    save_model(best_model, preprocessor)

    # Step 7: Deploy the model (uncomment to run)
    #deploy_model()