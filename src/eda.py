import os
import yaml
import zipfile
from io import StringIO

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


#opening config file
options_path = 'src/config.yml'
with open(options_path, 'r') as option_file:
    options = yaml.safe_load(option_file)

RANDOM_STATE = options['random_state'] #getting random state from options file into a variable 

def eda(df):
    """
    Modify DataFrame data types to int or float instead of object.

    Args:
        df (pandas.DataFrame): a pandas dataframe with 'income_rub', 'start_sum', and 'income_percent' columns.

    Returns:
        pandas.DataFrame: The modified DataFrame.

    Raises:
        None.
    """
    df['income_rub'] = df['income_rub'].replace(',', '.', regex=True).replace(' ', '', regex=True).astype('float')
    df[df['income_percent'] == '-'] = 0  
    df['income_percent'] = df['income_percent'].replace(',', '.', regex=True).astype('float')
    df['start_sum'] = df['start_sum'].replace(',', '.', regex=True).replace(' ', '', regex=True).astype('float')
    return df


def add_from_LSTM(features):
    
    final_list = []
    predected_lables = pd.read_csv('Final_final.csv')
    predected_lables['prediction_of_lstm'] = predected_lables['prediction_of_lstm'] + 1
    
    def search_by_id(features, id):
        # filter the dataframe to retain only the rows where the 'id' column matches the given id
        result = features[features.iloc[:, 0] == id]
        # return the first (and hopefully only) row of the filtered dataframe
        return result.iloc[0]['prediction_of_lstm'] if len(result) > 0 else 0
    
    for  index, row in features.iterrows():
        final_list.append(search_by_id(predected_lables, row['id']))
            
    features['prediction_of_lstm'] = final_list
    
    return features


def split_features_target(df):
    """
    Splits DataFrame into features and target values.

    Args:
        df (pandas.DataFrame): A pandas DataFrame with 'id', 'start_sum', "request", "deals", "income_rub", "income_percent", and 'class' columns.

    Returns:
        pandas.DataFrame: The features DataFrame with 'id', 'start_sum', "request", "deals", "income_rub", and "income_percent" columns.
        pandas.Series: The target Series with 'class' column values.

    Raises:
        None.
    """
    features = df[['id', 'start_sum', "request", "deals", "income_rub", "income_percent"]]
    target = df["class"]
    return features, target


def return_without_id(df):
    """
    Returns a DataFrame without the 'id' column.

    Args:
        df (pandas.DataFrame): A pandas DataFrame with 'id' and other columns.

    Returns:
        pandas.DataFrame: The same input data, but with the 'id' column dropped.

    Raises:
        None.
    """
    return df.drop('id', axis=1)


# return train_test_split(features, target, test_size=368, shuffle=False)
def split(features, target, mode):
    """
    Splits the input data into training and test sets using train_test_split() function from sklearn.

    Args:
        features (pandas.DataFrame or array-like): An object with the independent variables.
        target (pandas.Series or array-like): An object with the dependent variable.
        mode (str): A string indicating which model to use ('xgboost' or other).

    Returns:
        tuple of arrays or pandas.DataFrames: (X_train, X_test, y_train, y_test).

    Raises:
        None.
    """
    if mode == 'xgboost':
        return train_test_split(features.values, target.values.reshape(-1, 1), train_size=0.8, shuffle=True, random_state=RANDOM_STATE)
    else:
        return train_test_split(features, target, train_size=0.8, shuffle=True, random_state= RANDOM_STATE) 
    

def get_train_deals(features, train_deals_path):
    """
    Extracts data from other tables and adds it to the existing features dataset.

    Args:
        features (pandas.DataFrame): The data that requires additional features.
        train_deals_path (str): The path to the folder with the train_deals files.
    
    Returns:
        pandas.DataFrame: The data with the new extracted features.

    Raises:
        None.
    """
    for filename in os.listdir(train_deals_path):
        filename_id = filename[2:-4] # getting user id from folder's name
        
        if filename.endswith('.zip'): # take only zip files
            with zipfile.ZipFile(os.path.join(train_deals_path, filename), 'r') as zip_ref:
                for file in zip_ref.namelist():
                    with zip_ref.open(file) as f_in: # opening and reading files
                        file_content = f_in.read().decode('utf-8') # reading file's content
                        content = StringIO(file_content) # making it csv file, so we can read it with "read_csv"
                        auxiliary_df = pd.read_csv(content, sep=";", names=['time','tool','quantity','sum'])
                        auxiliary_df['date'] = auxiliary_df['time'].str[:10] # extracting date from datetime column to group by date

                        # getting how many tools a user was using
                        features.loc[features['id'] == int(filename_id), ['amount_of_tools']] = len(auxiliary_df['tool'].unique())
                    
                        # getting user's mean sum by days
                        features.loc[features['id'] == int(filename_id), ['mean_sum_by_days']] = auxiliary_df['sum'].mean().round(2)

                        # getting user's max sum by days
                        features.loc[features['id'] == int(filename_id), ['max_sum_by_days']] = auxiliary_df['sum'].max()

    return features

def get_from_csv_files(features, folder_path):
    """
    Extracts data from csv files and adds it to the existing features dataset.

    Args:
        features (pandas.DataFrame): The data that requires additional features.
        folder_path (str): The path to the folder with the csv files.
    
    Returns:
        pandas.DataFrame: The data with the new extracted features.

    Raises:
        None.
    """
    
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            filename_id = dir_name[3:] # get user's id from dir's name
            for file in os.listdir(dir_path):
                if filename_id.isdigit(): #check that's filename contains user's id by checking special part of it's name
                    if file.endswith(".csv"): # get only csv files
                        file_path = os.path.join(dir_path, file)
                        #work with stats table file
                        if 'stats_table' in file:
                            auxiliary_df = pd.read_csv(file_path, sep=';', header=0) #reading file
                            features.loc[features['id'] == int(filename_id), ['money_turnover']] = auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'all_markets', ['money_turnover']].values[0]
                            features.loc[features['id'] == int(filename_id), ['ration_of_stock_market']] = auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'stock_market', ['deals']].values[0][0]*100/ auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'all_markets', ['deals']].values[0][0]
                            features.loc[features['id'] == int(filename_id), ['ration_of_forts_market']] = auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'forts_market', ['deals']].values[0][0]*100/ auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'all_markets', ['deals']].values[0][0]
                        #work with stats table file
                        if 'account_condition' in file:
                            auxiliary_df = pd.read_csv(file_path, sep=';', header=0)
                            #clearing column 'free_funds'
                            auxiliary_df['free_funds_clear'] = auxiliary_df['free_funds'].apply(lambda x: x.split('(', 1)[-1].split(')', 1)[0]).apply(lambda x: 0 if (x == '-' or x =='+') else x).astype('float')
                            # getting average free funds 
                            features.loc[features['id'] == int(filename_id), ['average_free_funds']] = float(auxiliary_df['free_funds_clear'].diff().mean())       
                        #work with reference point file
                        if 'reference_point' in file:
                            auxiliary_df = pd.read_csv(file_path, sep=';', header=0)
                            features.loc[features['id'] == int(filename_id), ['average_end_day_balance']] = auxiliary_df['end_day_balance'].diff().mean()
    return features


def add_each_tool(features, train_deals_path):
    """
    adds all tools(shares) from train_deals_path csv files to main df as one hot encodint

    Args:
        train_deals_path (str): The path to the folder with the train_deals files.
        
    Returns:
        numpy.ndarray: features with additional columns which are one hot encoded tools

    Raises:
        None.
    """

    def get_all_tools_list(train_deals_path):
        """
        Creates a list of unique values for all tools from all train_deals files.

        Args:
            features (pandas.DataFrame): Unused argument.
            train_deals_path (str): The path to the folder with the train_deals files.
        
        Returns:
            pandas.DataFrame: The data with the modified tool columns.

        Raises:
            None.
        """
        alltools_list = np.array([])
        for filename in os.listdir(train_deals_path):
            if filename.endswith('.zip'): # take only zip files
                with zipfile.ZipFile(os.path.join(train_deals_path, filename), 'r') as zip_ref:
                    for file in zip_ref.namelist():
                        with zip_ref.open(file) as f_in: # opening and reading files
                            file_content = f_in.read().decode('utf-8') # reading file's content
                            content = StringIO(file_content) # making it csv file, so we can read it with "read_csv"
                            auxiliary_df = pd.read_csv(content, sep=";", names=['time','tool','quantity','sum'])
                            alltools_list = np.concatenate((alltools_list,  auxiliary_df[auxiliary_df['tool'].str.len() <= 5]['tool'].unique()))
        return  np.unique(alltools_list)
    
    def change_tools_columns(features, train_deals_path):
        """
        Modifies tools columns by replacing zeros with ones if the user had any deals related to a particular tool.

        Args:
            features (pandas.DataFrame): The data that requires tool columns modification.
            train_deals_path (str): The path to the folder with the train_deals files.
        
        Returns:
            pandas.DataFrame: The data with the modified tool columns.

        Raises:
            None.
        """
        for filename in os.listdir(train_deals_path):
            filename_id = filename[2:-4] # getting user id from folder's name
            if filename.endswith('.zip'): # take only zip files
                with zipfile.ZipFile(os.path.join(train_deals_path, filename), 'r') as zip_ref:
                    for file in zip_ref.namelist():
                        with zip_ref.open(file) as f_in: # opening and reading files
                            file_content = f_in.read().decode('utf-8') # reading file's content 
                            content = StringIO(file_content) # making it csv file, so we can read it with "read_csv"
                            auxiliary_df = pd.read_csv(content, sep=";", names=['time','tool','quantity','sum'])

                            columns_to_modify = list(auxiliary_df[auxiliary_df['tool'].str.len() <= 5]['tool'].unique())
                            
                            features.loc[features['id'] == int(filename_id), columns_to_modify] = 1

        return features

    alltools_list = get_all_tools_list(train_deals_path)

    for col in alltools_list:
        features[col] = 0

    return change_tools_columns(features, train_deals_path), alltools_list
