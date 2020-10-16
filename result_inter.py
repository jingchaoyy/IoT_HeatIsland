"""
Created on  9/28/20
@author: Jingchao Yang
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import numpy as np

# result = pd.read_csv('/Volumes/Samsung_T5/IoT_HeatIsland_Data/data/zmodel_evaluation'
#                      '/testerror_boxplot_Chicago_Chicago_allattr.csv')
#
# # sns.boxplot(x='RMSE', y='Temperature (F)', data=result)
# result.boxplot()
# plt.xlabel('input-->forecast')
# plt.ylabel('Temperature (C)')
# plt.title('Prediction RMSE (Chicago)')
# plt.show()

loc = 'LA_LA_20neuron'
model_load_path = f'/Volumes/Samsung_T5/IoT_HeatIsland_Data/data/LA/exp_data/{loc}/'

experiments = [(24, 1), (24, 4), (24, 8), (24, 12), (36, 12), (48, 12), (72, 12), (72, 24), (72, 36), (72, 48),
               (120, 48), (144, 48), (120, 72), (144, 72), (168, 120)]

rmse_df = pd.DataFrame()
for exp in tqdm(range(len(experiments))):
    train_window, output_size = experiments[exp]

    rmse_path = glob.glob(model_load_path + f'{str(train_window)}_{str(output_size)}/' + 'testScores_C.csv')
    try:
        rmse_df[f'{str(train_window)}_{str(output_size)}'] = pd.read_csv(rmse_path[0], usecols=['0']).values.flatten()
    except:
        rmse_df[f'{str(train_window)}_{str(output_size)}'] = np.NaN

rmse_df.boxplot()
plt.xlabel('input-->forecast')
plt.ylabel('Temperature (C)')
plt.title(f'Prediction RMSE ({loc})')
plt.show()
