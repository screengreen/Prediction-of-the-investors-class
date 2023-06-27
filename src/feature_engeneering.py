import numpy as np
import pandas as pd

def feature_engeneering(features):
    """
    Performs feature engineering on the given dataframe 'features'.

    Args:
        features (pandas.DataFrame): The features dataframe to perform the feature engineering.

    Returns:
        A pandas dataframe containing the updated features after feature engineering.
    """
    
    features_engeneered = features.copy()

    features_engeneered['income_percent_plus_min'] = features_engeneered['income_percent'] - features_engeneered['income_percent'].min()
    features_engeneered['income_percent_log'] = np.log(features_engeneered['income_percent_plus_min'])
    features_engeneered = features_engeneered.drop('income_percent_plus_min', axis=1)

    features_engeneered['income_rub_plus_min'] = features_engeneered['income_rub'] - features_engeneered['income_rub'].min()
    features_engeneered['income_rub_log'] = np.log(features_engeneered['income_rub_plus_min'])
    features_engeneered = features_engeneered.drop('income_rub_plus_min', axis=1)

    features_engeneered['money_turnover_min'] = features_engeneered['money_turnover'] - features_engeneered['money_turnover'].min()
    features_engeneered['money_turnover_log'] = np.log(features_engeneered['money_turnover_min'])
    features_engeneered = features_engeneered.drop('money_turnover_min', axis=1)

    features_engeneered['average_free_funds_min'] = features_engeneered['average_free_funds'] - features_engeneered['average_free_funds'].min()
    features_engeneered['average_free_funds_log'] = np.log(features_engeneered['average_free_funds_min'])
    features_engeneered = features_engeneered.drop('average_free_funds_min', axis=1)

    features_engeneered=features_engeneered.fillna(0)

    return feature_engeneering