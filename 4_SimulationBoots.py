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
from sklearn.utils import resample  # For bootstrapping
from scipy.ndimage import gaussian_filter1d  # To smooth the data

# Start tracking time for performance evaluation
start_time = time.time()

# # Load Pre-Trained Model and Selected Features
# The model and selected features are loaded from files in the `Models` folder.
model_file = 'Models/PD_CF1_n1_c5_delta_xb40.joblib'
features_file = 'Models/PD_CF1_n1_c5_delta_xb40.json'

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
dataset_name = 'PD_CF1'
reduction_method = '_n1_c5_delta'
input_file = os.path.join('Input', f'{dataset_name}{reduction_method}.csv')
output_file = f'{dataset_name}{reduction_method}_xb40'
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

# Bootstrapping to generate prediction uncertainties
n_bootstraps = 100  # Number of bootstrap samples to estimate uncertainty
predictions_list = []

for _ in range(n_bootstraps):
    # Resample the dataset with replacement
    X_resampled, y_resampled = resample(X, y)
    
    # Fit the model on the resampled dataset
    model.fit(X_resampled, y_resampled)
    
    # Predict using the original X
    y_pred = model.predict(X)
    
    # Store predictions for uncertainty estimation
    predictions_list.append(y_pred)

# Convert the list of predictions into a 2D array (n_samples, n_bootstraps)
predictions_array = np.array(predictions_list)

# Calculate mean and standard deviation of the predictions across bootstraps
y_pred_mean = np.mean(predictions_array, axis=0)
y_pred_std = np.std(predictions_array, axis=0)


# # Save Results DataFrame
# Create and save a DataFrame that includes observed and predicted yields, along with crop type, field ID, year, week, and prediction uncertainties.

# Create and save a DataFrame that includes observed and predicted yields, along with crop type, field ID, week, and prediction uncertainties.
def create_results_df(X, y, y_pred_mean, y_pred_std):
    return pd.DataFrame({
        'Yield_mean': y,
        'Predicted_Yield': y_pred_mean,
        'Prediction_Uncertainty': y_pred_std,  # Add the prediction uncertainty (standard deviation)
        'Crop_type': df['Crop_type'].reset_index(drop=True),
        'FIELDID': df['FIELDID'].reset_index(drop=True),
        'Week': df['week'].reset_index(drop=True),  # Keep Week, but remove Year
    })


# Create results DataFrame
results_df = create_results_df(X, y, y_pred_mean, y_pred_std)

# # Save Results
# Save the results DataFrame to a CSV file.
results_df.to_csv(os.path.join(output_folder, f'{output_file}_boots_results.csv'), index=False)


# # Calculate Uncertainties Per Week and Crop Type
# Group the results by 'Week' and 'Crop_type', and calculate the mean uncertainty for each group.
def calculate_weekly_uncertainty(results_df):
    weekly_uncertainty = results_df.groupby(['Week', 'Crop_type']).agg(
        mean_uncertainty=('Prediction_Uncertainty', 'mean'),
        std_uncertainty=('Prediction_Uncertainty', 'std'),
        count=('Prediction_Uncertainty', 'count')
    ).reset_index()
    
    return weekly_uncertainty

# Calculate uncertainties per week and crop type
weekly_uncertainty_df = calculate_weekly_uncertainty(results_df)

# Save the weekly uncertainties to a CSV file for further analysis
weekly_uncertainty_file = os.path.join(output_folder, f'{output_file}_weekly_uncertainties.csv')
weekly_uncertainty_df.to_csv(weekly_uncertainty_file, index=False)

# Save the weekly uncertainties to a CSV file for further analysis
weekly_uncertainty_file = os.path.join(output_folder, f'{output_file}_weekly_uncertainties.csv')
weekly_uncertainty_df.to_csv(weekly_uncertainty_file, index=False)



# # Plot Uncertainties by Week and Crop Type
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

def plot_weekly_uncertainty_with_improvements(weekly_uncertainty_df):
    # Create a figure and axis
    plt.figure(figsize=(10, 6))
    
    # Get the unique crop types
    crop_types = weekly_uncertainty_df['Crop_type'].unique()
    
    # Define a color map for each crop type
    color_map = {
        'Maize': 'blue',
        'Soy': 'orange',
        'Sunflower': 'green',
        'Lucern': 'red',
        'Wheat': 'purple'
    }
    
    # Plot each crop type with smoothing
    for crop_type in crop_types:
        crop_data = weekly_uncertainty_df[weekly_uncertainty_df['Crop_type'] == crop_type]
        
        # Convert Week and mean_uncertainty columns to numpy arrays
        weeks = crop_data['Week'].to_numpy()
        mean_uncertainty = crop_data['mean_uncertainty'].to_numpy()
        
        # Apply Gaussian smoothing to the uncertainty values for a smoother plot
        smooth_uncertainty = gaussian_filter1d(mean_uncertainty, sigma=1)
        
        # Plot the smoothed uncertainty with line style and color
        plt.plot(weeks, smooth_uncertainty, label=crop_type, color=color_map.get(crop_type, 'gray'), linewidth=2)
        
        # Optionally shade around the line to indicate variability (standard deviation)
        std_uncertainty = crop_data['std_uncertainty'].to_numpy()
        plt.fill_between(weeks, smooth_uncertainty - std_uncertainty, smooth_uncertainty + std_uncertainty,
                         color=color_map.get(crop_type, 'gray'), alpha=0.2)

    # Add labels and title
    plt.xlabel('Week of the Year')
    plt.ylabel('Mean Prediction Uncertainty')
    plt.title('Prediction Uncertainty (t/ha) per Week for Each Crop Type (Smoothed)')
    plt.ylim(0, 1)
    
    # Add a legend
    plt.legend(title='Crop Type')
    
    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    # plt.show()


# Call the function to plot the uncertainty with improvements
plot_weekly_uncertainty_with_improvements(weekly_uncertainty_df)
# Save the plot to a file
plt.savefig(os.path.join(output_folder, f'{output_file}_uncertainty_plot.png'))

