# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 19:23:57 2021

@author: USER
"""

import os
import pandas as pd
from datetime import timedelta
import numpy as np
from sklearn.cluster import KMeans


del_depth = 'non_depth'
missing_cond = 'interpolate'

save_dir = 'C:/BaseData/Class/2021-2/Industrial AI/Progress/1125/rlt'
fn_dir = 'C:/GIT/industrial_AI'
data_dir = 'C:/BaseData/Class/2021-2/Industrial AI/Progress'
file_name = 'data_10min.csv'

os.chdir(fn_dir)
from data_function import *

# load data

for del_depth in ['non_depth', 'depth']:
    for missing_cond in ['mean', 'interpolate']:
        result_dir = f'C:/BaseData/Class/2021-2/Industrial AI/Progress/1125/rlt/clustering_LSTM-{del_depth}-{missing_cond}.csv'
        
        df_result = pd.read_csv(result_dir)
        df_clustering, clustering_idx, df = load_preprocessing(data_dir, file_name, del_depth, missing_cond)
        
        df['TIME_STAMP'] = pd.to_datetime(df['TIME_STAMP'], format='%Y-%m-%d %H:%M:%S')
        df_clustering['TIME_STAMP'] = pd.to_datetime(df_clustering['TIME_STAMP'], format='%Y-%m-%d %H:%M:%S')
        
        # get timesteps from data
        
        ts = np.shape(df_clustering)[0] - np.shape(df_result)[0] + 1
        
        
        # extract outlier case
        
        df_result_outlier= pd.DataFrame({'idx': df_result[df_result['Result'] == 1].index})
        
        for i in range(ts):
            df_result_outlier[f'ts{i}'] = df_clustering.loc[df_result_outlier['idx'] + i]['TIME_STAMP'].tolist()
            df_result_outlier[f'ts{i}'] = pd.to_datetime(df_result_outlier[f'ts{i}'], format='%Y-%m-%d %H:%M:%S')
        
        df_result_outlier['gap'] = (df_result_outlier[f'ts{ts-1}'] - df_result_outlier['ts0']).dt.seconds/60
        
        
        # get only continuous case
        
        df_result_outlier = df_result_outlier.drop(np.where(df_result_outlier['gap'] > ((ts-1)*10))[0])
        df_result_outlier.index = range(np.shape(df_result_outlier)[0])
        
        
        for i in range(np.shape(df_result_outlier)[0]):
            tmp = np.array(df_clustering.loc[np.where(df_clustering['TIME_STAMP'].isin([df_result_outlier.loc[i][ts]]))[0]])

            #tmp = np.array(df_clustering.loc[np.where(df_clustering['TIME_STAMP'].isin([df_result_outlier.loc[i][ts]]))[0]][df_clustering.columns[1:]])
            #tmp = np.array(df_clustering.loc[np.where(df_clustering['TIME_STAMP'].isin(df_result_outlier.loc[i][1:(ts+1)]))[0]][df_clustering.columns[1:]])
            #tmp = np.array([tmp.flatten()])
            try:
                case = np.append(case,tmp, axis = 0)
            except:
                case = tmp
        
        case = pd.DataFrame(case)
        case.columns = df_clustering.columns
        
        os.chdir(save_dir)
        case['Result'] = 1
        case.to_csv(f'case_{del_depth}-{missing_cond}.csv', index = False)
        
        # kmeans
        """
        kmeans_rlt = pd.DataFrame({'TIME_STAMP':df_result_outlier[f'ts{ts-1}']})
        
        for k in range(2,11):
            kmeans = KMeans(n_clusters = k)
            kmeans.fit(case)
            
            kmeans_rlt[f'cluster{k}'] = kmeans.labels_
        
        os.chdir(save_dir)
        kmeans_rlt.to_csv(f'kmeans_rlt-{del_depth}-{missing_cond}.csv', index = False)
        
        """

        del case
