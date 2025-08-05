# Crop Yield Prediction New

This project contains a series of Python notebooks designed to facilitate data preparation, model fitting, and evaluation for predicting crop yield of Maize and Soy based on the Producer Data dataset. The goal is to enable effective data processing, model training, and prediction of crop yield using machine learning techniques.

## Notebooks and Scripts

### 1. [`1_Data_Preparation.ipynb`](./1_Data_Preparation.ipynb)
This notebook is designed to prepare crop yield data before feeding it into machine learning models, focusing on cleaning, feature engineering, and balancing class distribution.

- **Main Tasks:**
  - Loading and cleaning data based on NDVI and coverage thresholds.
  - Feature engineering with vegetation index derivatives (e.g., NDVI, EVI).
  - Rebalancing the class distribution for Maize and Soy.

### 2. [`2_Fit_Classifier.ipynb`](./2_Fit_Classifier.ipynb)
This notebook outlines the process of predicting crop yield using an XGBoost model with Sequential Forward Selection (SFS) for feature selection.

- **Main Tasks:**
  - Importing and shuffling the dataset.
  - Adding cyclical features (e.g., week_sin, week_cos) and vegetation growth rates (e.g., NDVI, EVI).
  - Dynamically selecting predictors and defining labels.
  - Training the XGBoost model.
  - Performing feature selection and evaluating the model using R² and RMSE.
  - Saving the trained model and selected features for future use.

### 3. [`3_Simulation.ipynb`](./3_Simulation.ipynb)
This notebook loads a pre-trained XGBoost model and selected features to perform crop yield prediction and visualize results for Maize and Soy.

- **Main Tasks:**
  - Loading pre-trained model and features.
  - Adding cyclical week features and vegetation growth rates.
  - Evaluating the model and plotting results using time series scatter plots.

### 4. [`4_Simulation.py`](./4_Simulation.py)
This script performs crop yield prediction and visualization using a pre-trained XGBoost model.

- **Main Tasks:**
  - Loading the pre-trained model and features.
  - Creating sine/cosine transformations for seasonal patterns and adding growth rates for vegetation indices.
  - Predicting crop yield using the pre-trained model.
  - Calculating weekly performance metrics and saving results to CSV.

### 5. [`4_SimulationBoots.py`](./4_SimulationBoots.py)
This script evaluates a pre-trained XGBoost model and calculates bootstrapped accuracy for yield prediction on a weekly basis.

- **Main Tasks:**
  - Bootstrapping accuracy calculation per crop, per week.
  - Applying Gaussian smoothing to accuracy results.
  - Saving results and accuracy plots.

## General Instructions

### Environment Setup
Ensure that the following Python packages are installed: `pandas`, `scikit-learn`, `XGBoost`, and `matplotlib`. The notebooks are intended to be run in a Jupyter environment.

### Workflow
1. Start with [`1_Data_Preparation.ipynb`](./1_Data_Preparation.ipynb) to prepare the data.
2. Use [`2_Fit_Classifier.ipynb`](./2_Fit_Classifier.ipynb) to train the model.
3. Use [`3_Simulation.ipynb`](./3_Simulation.ipynb) for yield predictions and evaluations.
4. Additionally, use [`4_Simulation.py`](./4_Simulation.py) for Python-based yield predictions, and [`4_SimulationBoots.py`](./4_SimulationBoots.py) to add bootstrapped accuracy on a weekly basis.

### Outputs
- The notebooks generate outputs such as CSV files, scatter plots, and saved models.
- Each notebook saves its output in the specified directory.

