import warnings
import os
from io import StringIO

import zipfile
import yaml
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

from eda import *
from feature_engeneering import feature_engeneering
from metrics import get_metrics, make_log_data, append_to_json

#opening config file
options_path = '/Users/andreisuhov/Desktop/проект лаба/src/config.yml'
with open(options_path, 'r') as option_file:
    options = yaml.safe_load(option_file)

#сделаем красивый формат для дробных чисел
pd.set_option('display.float_format', '{:.2f}'.format)

#скроем возможные предупреждения
warnings.filterwarnings('ignore')

train_path = options['train_path']
train_deals_path = options['train_deals_path']
train_add_info_path = options['train_add_info_path']

df = pd.read_csv(train_path) #reading data from csv file train data

df = eda(df)
features , target = split_features_target(df)
features = get_from_csv_files(features, train_add_info_path)
features = get_train_deals(features, train_deals_path)
features, all_tools_list = add_each_tool(features, train_deals_path)
features.fillna(0, inplace=True)

if options['feature_engeneering']:
    feature_engeneering(features)

model_name = options['model']['model_name']

X_train, X_test, y_train, y_test = split(features, target, model_name)

print(X_train.shape)

if model_name == 'catboost':
    model = CatBoostClassifier()
elif model_name == 'xgboost':
    model = XGBClassifier()
elif model_name == 'gradientboost':
    model = GradientBoostingClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
metrics = get_metrics(y_pred, y_test)

try:
    if options['to_history']:
        data = make_log_data(model_name, model.get_params(), metrics, options['random_state'])
        append_to_json(data, options['history_file_path'])
except:
    print('could not add this train to history')
