import os
import warnings
import yaml
import zipfile
from io import StringIO

import pandas as pd
import numpy as np


LEN_OF_FEATURES = 200 # len of a sequence that's will be loaded to LSMT model

# Open config YAML file and load the contents into 'options' variable
options_path = 'src/config.yml'
with open(options_path, 'r') as option_file:
    options = yaml.safe_load(option_file)

# Set a custom float output format
pd.set_option('display.float_format', '{:.2f}'.format)

# Ignore possible warnings that may arise during runtime
warnings.filterwarnings('ignore')

# Extract the paths for train data files from the 'options' variable
train_path = options['train_path']
train_deals_path = options['train_deals_path']
train_add_info_path = options['train_add_info_path']

# Read data from csv file for train data and create a new pandas dataframe 'df'
df = pd.read_csv(train_path)

def get_all_tools_list(train_deals_path):
    """
    Retrieves all the unique tools present in the CSV files within the given directory.

    Args:
        train_deals_path (str): The path to the directory where CSV files are stored.

    Returns:
        A numpy array containing all the unique tools present in the CSV files.

    Raises:
        OSError: If unable to read an existing file within the given directory.
    """
    alltools_list = np.array([])

    for filename in os.listdir(train_deals_path):
        if filename.endswith('.zip'): #take only zip files
            with zipfile.ZipFile(os.path.join(train_deals_path, filename), 'r') as zip_ref:
                for file in zip_ref.namelist():
                    with zip_ref.open(file) as f_in:#opening and reading files
                        file_content = f_in.read().decode('utf-8') #reading file's content 
                        content = StringIO(file_content) #making it csv file, so we can read it with "read_csv"
                        auxiliary_df = pd.read_csv(content, sep=";", names=['time','tool','quantity','sum'])

                        # Get the unique tools in the current CSV file and concatenate with existing list
                        alltools_list = np.concatenate((alltools_list,  auxiliary_df['tool'].unique()))

    # Return the unique tools in the entire directory
    return np.unique(alltools_list)

# Retrieve all the unique tools in the given directory 
all_tools_list = get_all_tools_list(train_deals_path)

# Create a pandas dataframe from the unique list of all the tools
all_tools_df = pd.DataFrame(all_tools_list, columns=['tool'])

# Reset the index of the dataframe, rename the column representing the index to 'id', and update the dataframe
all_tools_df= all_tools_df.reset_index().rename(columns={'index': 'id'})


def change_tools_columns(train_deals_path, all_tools_df):
    """
    Changes a specified column in each CSV file within a directory to make it compatible with a given dataframe.

    Args:
        train_deals_path (str): The path to the directory where CSV files are stored.
        all_tools_df (pandas.DataFrame): The dataframe containing all tool information.

    Returns:
        A pandas dataframe containing all the modified CSV file data.

    Raises:
        ValueError: If the specified CSV file column does not exist.
        OSError: If unable to read an existing file within the given directory.
    """
    main_df = pd.DataFrame()
        
    for filename in os.listdir(train_deals_path):
        filename_id = filename[2:-4] # getting user id from folder's name
        if filename.endswith('.zip'): #take only zip files
            with zipfile.ZipFile(os.path.join(train_deals_path, filename), 'r') as zip_ref: 
                for file in zip_ref.namelist():
                    with zip_ref.open(file) as f_in:#opening and reading files
                        file_content = f_in.read().decode('utf-8') #reading file's content 
                        content = StringIO(file_content) #making it csv file, so we can read it with "read_csv"
                        auxiliary_df = pd.read_csv(content, sep=";", names=['time','tool','quantity','sum'])

                        auxiliary_df['time'] = pd.to_datetime(auxiliary_df['time'])
                        # create a timedelta column representing the difference from a specific year (1953 in this case) in seconds
                        auxiliary_df['time_delta'] = (auxiliary_df['time'] - pd.to_datetime('1953-01-01 00:00:00.000')).dt.total_seconds()
                        # drop the original 'time' column
                        auxiliary_df = auxiliary_df.drop('time', axis=1)

                        new_auxiliary_df = pd.merge(auxiliary_df, all_tools_df, on='tool', how='left')
                        new_auxiliary_df = new_auxiliary_df.drop('tool', axis=1)

                        if len(new_auxiliary_df) < LEN_OF_FEATURES:
                            add_rows = LEN_OF_FEATURES - len(new_auxiliary_df)
                            zero_df = pd.DataFrame(np.zeros((add_rows, len(new_auxiliary_df.columns))), columns=new_auxiliary_df.columns)
                            new_auxiliary_df = pd.concat([new_auxiliary_df, zero_df], ignore_index=True)
                            
                        new_auxiliary_df.insert(0, 'user_id', filename_id)
                        main_df = main_df._append(new_auxiliary_df.iloc[:LEN_OF_FEATURES], ignore_index=True)                   
        
        return main_df
    
features_df = change_tools_columns( train_deals_path, all_tools_df)

def add_class(features_df, origin_df):
    """
    Adds a class label to a given dataframe based on the 'id' column in another dataframe.

    Args:
        features_df (pandas.DataFrame): The features dataframe to which the class information is to be added.
        origin_df (pandas.DataFrame): The dataframe containing ID and class information.

    Returns:
        A modified pandas dataframe containing the class label added to the features dataframe.

    Raises:
        TypeError: If the 'user_id' column is not numeric.
    """
    # Rename 'id' column to 'user_id' in the origin dataframe
    origin_df = origin_df.rename(columns={'id': 'user_id'})

    # Convert the 'user_id' column in the features dataframe to numeric values
    features_df['user_id'] = pd.to_numeric(features_df['user_id'], downcast='integer')

    # Merge the origin dataframe with the features dataframe on the 'user_id' column
    auxiliary_df = pd.merge(features_df, origin_df, on='user_id', how='left')

    # Raise an error if the 'user_id' column in the features dataframe is not numeric
    if not pd.api.types.is_numeric_dtype(auxiliary_df['user_id']):
        raise TypeError("The 'user_id' column must be numeric.")

    return auxiliary_df

# Add the class label to the features dataframe
final_df = add_class(features_df, df)

# Drop unnecessary columns from the final dataframe
final_df  = final_df.drop(columns=['nickname','broker','start_sum', 'request', 'deals', 'income_rub', 'income_percent'], axis=1)

#save final df
final_df.to_csv('data/processed data/prepared_features_for_LSTM.csv')