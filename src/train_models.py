import warnings
import yaml

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

from eda import *
from feature_engeneering import feature_engeneering
from metrics import get_metrics, make_log_data, append_to_json

# Open the config file and load the options
options_path = 'src/config.yml'
with open(options_path, 'r') as option_file:
    options = yaml.safe_load(option_file)

# Set the format for displaying floating point numbers
pd.set_option('display.float_format', '{:.2f}'.format)

# Filter out potential warning messages
warnings.filterwarnings('ignore')

# Extract the paths for the training data from the loaded options
train_path = options['train_path']
train_deals_path = options['train_deals_path']
train_add_info_path = options['train_add_info_path']

# Load the training data and perform EDA on the data
df = pd.read_csv(train_path)
df = eda(df)

# Split the training data into features and targets
features, target = split_features_target(df)

# Add additional features derived from an LSTM model
features = add_from_LSTM(features)

# Load additional data from CSV files
features = get_from_csv_files(features, train_add_info_path)

# Load information on training deals
features = get_train_deals(features, train_deals_path)

# Add information on each tool for all training deals
features, all_tools_list = add_each_tool(features, train_deals_path)

# Fill in any missing values with 0s
features.fillna(0, inplace=True)

# Return the features dataframe without the identifier column
features = return_without_id(features)

if options['feature_engeneering']:
    # Perform feature engineering
    features = feature_engeneering(features)

# Extract model name from options
model_name = options['model']['model_name']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = split(features, target, model_name)

# Print dimensions of the training set
print(X_train.shape)

# Initialize the ML model based on the extracted model name
if model_name == 'catboost':
    model = CatBoostClassifier()
elif model_name == 'xgboost':
    model = XGBClassifier()
elif model_name == 'gradientboost':
    model = GradientBoostingClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate performance metrics of the trained model on the testing set
metrics = get_metrics(y_pred, y_test)

try:
    # Check if the user opted to save the training history
    if options['to_history']:
        # Get the logs of the current training session
        data = make_log_data(model_name, model.get_params(), metrics, options['random_state'])
        # Append the logs for the current session to the training history JSON file
        append_to_json(data, options['history_file_path'])
except:
    # If an error occurs in the try block, print a message indicating the failure
    print('could not add this train to history')