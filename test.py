import pandas as pd

df = pd.read_csv('data/raw data/train/train.csv')


def add_from_LSTM(features):
    
    final_list = []
    predected_lables = pd.read_csv('Final_final.csv')
    predected_lables['prediction_of_lstm'] = predected_lables['prediction_of_lstm'] + 1
    
    def search_by_id(df, id):
        # filter the dataframe to retain only the rows where the 'id' column matches the given id
        result = df[df.iloc[:, 0] == id]
        # return the first (and hopefully only) row of the filtered dataframe
        return result.iloc[0]['prediction_of_lstm'] if len(result) > 0 else 0
    
    for  index, row in df.iterrows():
        final_list.append(search_by_id(predected_lables, row['id']))
            
    df['prediction_of_lstm'] = final_list
    
    return df


print(add_from_LSTM(df))