import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from train_models import df
"""## Анализ"""

data = df
data.info() #изучим общую информацию о данных

# приведем данные к нужному типу
data['start_sum'] = data['start_sum'].str.replace(" ", "").str.replace(',','.').astype(float)
data['income_rub'] = data['income_rub'].str.replace(" ", "").str.replace(',','.').astype(float)
data['income_percent'] = data['income_percent'].str.replace("-", "0").str.replace(',','.').astype(float)

#посмотрим долю пропусков
pd.DataFrame(data.isna().mean()*100)

print('Число дубликатов:', data.duplicated().sum()) #проверим нет ли явных дубликатов

data.describe()

def draw_hist_and_boxplot(column):
    figure, ax = plt.subplots(1, 2, figsize=(15,5))
    data[column].plot(kind='hist', bins=50, ax=ax[0])
    data[column].plot(kind='box', ax=ax[1]);
    plt.show()

draw_hist_and_boxplot('start_sum')

"""У стартовой суммы значения от 100 тысяч до 3 миллионов, в большинстве случаев стартовая сумма составляет как раз минимальные 100 тысяч, выбросы - значения после одного миллиона."""

data[data['start_sum'] > data['start_sum'].std() ]['start_sum'].describe()

data[data['start_sum'] > 4* data['start_sum'].std()]

"""выбросов нет


"""

draw_hist_and_boxplot('request')

"""request - это количество заявок для сделки и здесь явно много выбросов и аномальных значений. В основном до 500 заявок, однако есть человек у которого было и 28 тысяч заявок **(по-моему это странно и надо об этом спросить).**"""

data[data['request'] > data['request'].std()]['request'].describe()

draw_hist_and_boxplot('deals')

"""То же самое и с количеством сделок, в основном их около 600, но встречаются и люди, у которых их 43 тысячи, **(что также странно и об этом, наверное, нужно спросить)**"""

draw_hist_and_boxplot('income_rub')

"""В полне логично, что у всех разный доход, но можно заметить, что большинство людей выходят в 0. Судя по выбросам, есть супер богатые и банкроты )))"""

# вернуть дф, в котором удалены строчки, где значения превышают 3 std
def del_rows_that_bigger(df, col):
  return df[(df[col] > df[col].mean() - 3*df[col].std()) & (df[col] < df[col].mean() + 3*df[col].std())]

new_df = del_rows_that_bigger(data, 'income_rub')
# Сколько удалилось:
(data.shape[0] - new_df.shape[0])/data.shape[0]

df['class'].value_counts() # check how many target classes

