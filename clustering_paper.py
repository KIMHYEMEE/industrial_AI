# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:46:50 2021

@author: USER
"""

import os
import pandas as pd
import numpy as np


missing_cond = 'drop' # {mean, drop}

# 1. Load data

data_dir = 'C:/BaseData/Class/2021-2/Industrial AI/Progress'

os.chdir(data_dir)

df = pd.read_csv('data_10min.csv')
#df.TIME_STAMP = pd.to_datetime(df.TIME_STAMP , format='%Y-%m-%d %H:%M:%S')

df_des = df.describe()

df_col = df.columns[np.where(df_des.loc['count'] >= 50000)]

# 2. RPM Condition

rpm = 'ME1_RPM_ECC'
df_working = df[(df[rpm] >= 80) & (df[rpm] <=120)]


# 3. Select columns to train
df_clustering = df_working[df_col]
df_clustering.index = range(np.shape(df_clustering)[0])

# df_clustering = df[df_col]

# missing data processing

if missing_cond == 'mean':

    # (1) Fill average
    clustering_idx = df_clustering.index.tolist()
    df_clustering = df_clustering.fillna(df.mean())

elif missing_cond == 'drop': 
    # (2) Drop null
    clustering_idx = df_clustering.dropna(axis=0).index.tolist()
    df_clustering = df_clustering.iloc[clustering_idx,:]

# 4. Modeling function
def modeling(model_name):
    if model_name == 'KNN':
        from pyod.models.knn import KNN as pyod_model
        model = pyod_model()
    
    elif model_name == 'ABOD':
        from pyod.models.abod import ABOD as pyod_model
        model = pyod_model()
    
    elif model_name == 'LOF':
        from pyod.models.lof import LOF as pyod_model
        model = pyod_model()
        
    elif model_name == 'CBLOF':
        from pyod.models.cblof import CBLOF as pyod_model
        model = pyod_model()    
    elif model_name == 'LODA':
        from pyod.models.loda import LODA as pyod_model
        model = pyod_model()
        
    elif model_name == 'IF': # Isolation Forest
        from pyod.models.iforest import IForest as pyod_model
        model = pyod_model()
        
    elif model_name == 'OCSVM':
        from pyod.models.ocsvm import OCSVM as pyod_model
        model = pyod_model()
        
    elif model_name == 'auto_encoder': # torch ..? 현재 실행 안됨
        from pyod.models.auto_encoder import AutoEncoder as pyod_model
        model = pyod_model()

    return model

# 5. Get outlier result from outlier score

def determine_outlier(rlt): # top 1% outlier score
    tmp = rlt.copy()
    tmp.sort()
    target_idx = len(tmp) - int(0.01 / (1/len(tmp)))
    target_val = tmp[target_idx]
    
    
    rlt_tf = np.array([0 for _ in range(len(rlt))])
    rlt_tf[rlt > target_val] = 1

    return rlt_tf
    


# 6. Training each model and save detection result

model_list = ['ABOD','LOF','CBLOF','LODA','IF']

rlt_df = pd.DataFrame({'TIME_STAMP': df['TIME_STAMP'][clustering_idx],
                       'Result':[np.nan for _ in range(np.shape(df_clustering.iloc[:,1:])[0])]})

for model_name in model_list:
    model = modeling(model_name)
    model.fit(df_clustering.iloc[:,1:])
    
    rlt = model.decision_scores_
    rlt_df[model_name] = determine_outlier(rlt)
    
    
# 7. Calculate final result

tmp = [0 for _ in range(np.shape(rlt_df)[0])]
idx = np.where(rlt_df.iloc[:,1:].sum(axis=1) >= 3)[0]

for i in idx:
    tmp[i] = 1

rlt_df['Result'] =  tmp


# 8. Svae the result

# rlt_df.to_csv('clustering_paper.csv', index = False)

# 9. Result join

rlt_join = pd.DataFrame({'TIME_STAMP': df_clustering['TIME_STAMP'],
                         'Cluster': rlt_df['Result']})

rlt_final = df[['TIME_STAMP', rpm]]

rlt_final = pd.merge(left = rlt_final, right=rlt_join, how='left', on='TIME_STAMP')


# 10. Plot
import matplotlib.pyplot as plt

l = 15000

for i in range(int(np.shape(rlt_final)[0]/l)+1):
    try:
        tmp = rlt_final.iloc[(i*l):((i+1)*l),:]
    except:
        tmp = rlt_final.iloc[(i*l):,:]
    
    fig_name = f'./1125/fig/{missing_cond}/outlier_{str(tmp["TIME_STAMP"].tolist()[0])[:10]}-{str(tmp["TIME_STAMP"].tolist()[-1])[:10]}.png'
    
    tmp['Cluster'] = tmp['Cluster'] * tmp[rpm]
    tmp['Cluster'][tmp['Cluster']==0] = np.nan
    
    plt.rcParams['axes.grid'] = True
    plt.figure(figsize=(30,5))
    plt.plot(tmp[rpm])
    plt.plot(tmp['Cluster'], 'r^')
    plt.title(f'{tmp["TIME_STAMP"].tolist()[0]} - {tmp["TIME_STAMP"].tolist()[-1]}')
    plt.margins(x=0)
    
    plt.savefig(fig_name, dpi=200,transparent=True,bbox_inches='tight')
    plt.show()
