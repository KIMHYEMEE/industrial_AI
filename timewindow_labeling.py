# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:50:47 2021

@author: USER
"""

import os
import pandas as pd
import numpy as np

# 1. Load data

data_dir = 'C:/BaseData/Class/2021-2/Industrial AI/Progress'

os.chdir(data_dir)

df_feature = pd.read_csv('lstm_autoencoder_result.csv')
df_outlier = pd.read_csv('clustering_paper.csv')
df_outlier = df_outlier[['TIME_STAMP','Result']]


# 2. score calcuation
ts = 12
n_row = np.shape(df_outlier)[0]-ts+1

for i in range(ts):
    tmp = np.asarray(df_outlier.iloc[i:i+n_row,1:])
    
    try:
        df_idx = np.append(df_idx, tmp, axis=1)
    except:
        df_idx = tmp

score = df_idx.mean(axis=1)

# 3. scale the score

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

score = np.transpose(np.array([score]))
score = scaler.fit_transform(score)

# 4. concat the score

df_feature['score'] = score

# 5. save the result

df_feature.to_csv('feature_score.csv',index=False)
