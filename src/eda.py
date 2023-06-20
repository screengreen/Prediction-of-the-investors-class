import os
from io import StringIO
import yaml

from sklearn.model_selection import train_test_split
import zipfile
import pandas as pd
import numpy as np

#opening config file
options_path = '/Users/andreisuhov/Desktop/проект лаба/src/config.yml'
with open(options_path, 'r') as option_file:
    options = yaml.safe_load(option_file)

RANDOM_STATE = options['random_state']

def eda(df):
    # editing data in df, making int or float data types instead of object type
    df['income_rub'] = (df['income_rub'].replace(',','.', regex=True).replace(' ','', regex=True).astype('float'))
    df[df['income_percent'] == '-'] = 0  #replacing all "-" with zeros 
    df['income_percent'] = (df['income_percent'].replace(',','.', regex=True).astype('float'))
    df['start_sum'] = (df['start_sum'].replace(',','.', regex=True).replace(' ','', regex=True).astype('float'))

    return df

def split_features_target(df):
    #spliting df into features and target values 
    features, target = df[['id','start_sum',"request","deals","income_rub","income_percent"]], df["class"]
    return features, target


def split(features, target, mode):
    if mode == 'xgboost':
       return train_test_split(features.values, target.values.reshape(-1,1), train_size = 0.8 , shuffle = True, random_state= RANDOM_STATE )
    else:
        return train_test_split(features, target, train_size = 0.8 , shuffle = True, random_state= RANDOM_STATE) #random_state = 65 (try - 56)

"""## Вытаскием данные из других табличек"""
def get_train_deals(features, train_deals_path):
    for filename in os.listdir(train_deals_path):
        filename_id = filename[2:-4] # getting user id from folder's name
        #print(filename_id)
        
        if filename.endswith('.zip'): #take only zip files
            with zipfile.ZipFile(os.path.join(train_deals_path, filename), 'r') as zip_ref: 
                for file in zip_ref.namelist():
                    with zip_ref.open(file) as f_in:#opening and reading files
                        file_content = f_in.read().decode('utf-8') #reading file's content 
                        content = StringIO(file_content) #making it csv file, so we can read it with "read_csv"
                        auxiliary_df = pd.read_csv(content, sep=";", names=['time','tool','quantity','sum'])
                        auxiliary_df['date'] = auxiliary_df['time'].str[:10] # extracting date from datetime column to group by date

                        #getting how many tools a user was using
                        features.loc[features['id'] == int(filename_id), ['amount_of_tools']] = len(auxiliary_df['tool'].unique()) 

                        #getting user's everage deals by days 
                        #features.loc[features['id'] == int(filename_id), ['everage_deals_by_day']] = auxiliary_df.groupby('date').count().mean().iloc[1]

                        #getting user's mean sum by days 
                        features.loc[features['id'] == int(filename_id), ['mean_sum_by_days']] = auxiliary_df['sum'].mean().round(2)

                        #getting user's max sum by days 
                        features.loc[features['id'] == int(filename_id), ['max_sum_by_days']] = auxiliary_df['sum'].max()
    
    return features

def get_from_csv_files(features, folder_path):

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
                            #print('sdf - ', auxiliary_df)
                            #getting all values from stats table row "all_markets"
                            features.loc[features['id'] == int(filename_id), ['money_turnover']] = auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'all_markets', ['money_turnover']].values[0]
                            #print('\n\n stock_market  - ',type(auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'stock_market', ['deals']].values[0][0]),'\n\n fonts - ', auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'forts_market', ['deals']].values[0][0])
                            features.loc[features['id'] == int(filename_id), ['ration_of_stock_market']] = auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'stock_market', ['deals']].values[0][0]*100/ auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'all_markets', ['deals']].values[0][0]
                            features.loc[features['id'] == int(filename_id), ['ration_of_forts_market']] = auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'forts_market', ['deals']].values[0][0]*100/ auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'all_markets', ['deals']].values[0][0]
                            #features.loc[features['id'] == int(filename_id), ['ration_of_income_stock_market']] = auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'stock_market', ['income_rubles']].values[0][0]*100/ auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'all_markets', ['income_rubles']].values[0][0]
                            #features.loc[features['id'] == int(filename_id), ['ration_of_income_forts_market']] = auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'forts_market', ['income_rubles']].values[0][0]*100/ auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'all_markets', ['income_rubles']].values[0][0]
                            #features.loc[features['id'] == int(filename_id), ['ration_of_turnover_stock_market']] = auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'stock_market', ['money_turnover']].values[0][0]*100/ auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'all_markets', ['money_turnover']].values[0][0]
                            #features.loc[features['id'] == int(filename_id), ['ration_of_turnover_forts_market']] = auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'forts_market', ['money_turnover']].values[0][0]*100/ auxiliary_df.loc[ auxiliary_df['Unnamed: 0'] == 'all_markets', ['money_turnover']].values[0][0]
                            
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

    def get_all_tools_list(features, train_deals_path):
        alltools_list = np.array([])
        for filename in os.listdir(train_deals_path):
            if filename.endswith('.zip'): #take only zip files
                with zipfile.ZipFile(os.path.join(train_deals_path, filename), 'r') as zip_ref: 
                    for file in zip_ref.namelist():
                        with zip_ref.open(file) as f_in:#opening and reading files
                            file_content = f_in.read().decode('utf-8') #reading file's content 
                            content = StringIO(file_content) #making it csv file, so we can read it with "read_csv"
                            auxiliary_df = pd.read_csv(content, sep=";", names=['time','tool','quantity','sum'])
                            alltools_list = np.concatenate((alltools_list,  auxiliary_df[auxiliary_df['tool'].str.len() <= 5]['tool'].unique()))
        return  np.unique(alltools_list)
    
    def change_tools_columns(features, train_deals_path):
        
        for filename in os.listdir(train_deals_path):
            filename_id = filename[2:-4] # getting user id from folder's name
            if filename.endswith('.zip'): #take only zip files
                with zipfile.ZipFile(os.path.join(train_deals_path, filename), 'r') as zip_ref: 
                    for file in zip_ref.namelist():
                        with zip_ref.open(file) as f_in:#opening and reading files
                            file_content = f_in.read().decode('utf-8') #reading file's content 
                            content = StringIO(file_content) #making it csv file, so we can read it with "read_csv"
                            auxiliary_df = pd.read_csv(content, sep=";", names=['time','tool','quantity','sum'])

                            columns_to_modify = list(auxiliary_df[auxiliary_df['tool'].str.len() <= 5]['tool'].unique())
                            
                            features.loc[features['id'] == int(filename_id), columns_to_modify] = 1                    

        return features

    alltools_list = get_all_tools_list(features, train_deals_path)

    for col in alltools_list:
        features[col] = 0

    return change_tools_columns(features, train_deals_path), alltools_list
    

                            
                        

