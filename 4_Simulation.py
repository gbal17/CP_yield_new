# # Setting up the environment
# Import necessary libraries
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib  # For loading the saved model
import json

# Start tracking time for performance evaluation
start_time = time.time()

# # Load Pre-Trained Model and Selected Features
# The model and selected features are loaded from files in the `Models` folder.
model_file = 'Models/PD_n1_c5_delta_xb30.joblib'
features_file = 'Models/PD_n1_c5_delta_xb30.json'

# Load pre-trained model
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file {model_file} not found.")
model = joblib.load(model_file)

# Load selected features
if not os.path.exists(features_file):
    raise FileNotFoundError(f"Features file {features_file} not found.")
with open(features_file, 'r') as f:
    selected_features = json.load(f)['features']

# # Dataset Configuration
# Set paths to input files.
dataset_name = 'PD'
reduction_method = '_n1_c5_delta'
input_file = os.path.join('Input', f'{dataset_name}{reduction_method}.csv')
output_file = f'{dataset_name}{reduction_method}_xb30'
output_folder = 'Output'
os.makedirs(output_folder, exist_ok=True)

# Load data from CSV, raising an error if file is missing
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file {input_file} not found.")
df = pd.read_csv(input_file)

# # Feature Engineering
# This function adds new features such as sine/cosine transformations for the 'week' column and growth rates for vegetation indices.
def add_features(df):
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
    df['growth_rate_ndvi'] = df['veg_max_ndvi'] - df['veg_min_ndvi']
    df['growth_rate_evi'] = df['veg_max_evi'] - df['veg_min_evi']
    df['growth_rate_lai'] = df['veg_max_lai'] - df['veg_min_lai']
    df['mean_combined_vegetation_index'] = (df['veg_mean_evi'] + df['veg_mean_ndvi'] + df['veg_mean_lai']) / 3
    return df

# Apply feature engineering
df = add_features(df)

# # Using Full Dataset for Model Evaluation
# Fit the model on the entire dataset and calculate metrics without splitting the data.
X = df[selected_features]
y = df["Mean"]

# Using the full dataset for training and evaluation
model.fit(X, y)
y_pred = model.predict(X)

# # Save Results DataFrame
# Create and save a DataFrame that includes observed and predicted yields, along with crop type, field ID, year, and week.
def create_results_df(X, y, model):
    return pd.DataFrame({
        'Yield_mean': y,
        'Predicted_Yield': model.predict(X),
        'Crop_type': df['Crop_type'].reset_index(drop=True),
        'FIELDID': df['FIELDID'].reset_index(drop=True),
        'Year': df['Year'].reset_index(drop=True),
        'Week': df['week'].reset_index(drop=True),
    })

# Create results DataFrame
results_df = create_results_df(X, y, model)

# # Save Results
# Save the results DataFrame to a CSV file.
results_df.to_csv(os.path.join(output_folder, f'{output_file}_results.csv'), index=False)
