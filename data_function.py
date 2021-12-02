# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 14:37:06 2021

@author: USER
"""

import os
import pandas as pd
import numpy as np

def load_preprocessing(data_dir, file_name, del_depth, missing_cond):
    # 1. Load data    
    
    os.chdir(data_dir)
    
    df = pd.read_csv(file_name)
    #df.TIME_STAMP = pd.to_datetime(df.TIME_STAMP , format='%Y-%m-%d %H:%M:%S')
    
    df_des = df.describe()
    
    df_col = df.columns[np.where(df_des.loc['count'] >= 50000)].tolist()
    rpm = 'ME1_RPM_ECC'
    
    df_col.append(rpm)
    
    # 2. RPM Condition
    
    df_working = df[(df[rpm] >= 80) & (df[rpm] <=120)] 
    
    
    # 3. Select columns to train
    df_clustering = df_working[df_col]
    df_clustering.index = range(np.shape(df_clustering)[0]) 
    
    # df_clustering = df[df_col]
    
    df_clustering.loc[df_clustering.SPEED_LG>100,'SPEED_LG'] = np.nan
    df_clustering.loc[df_clustering.SPEED_TG>100,'SPEED_TG'] = np.nan
    
    
    if del_depth == 'non_depth':
        del df_clustering['WATER_DEPTH']
    
    
    # missing data processing
    
    if missing_cond == 'mean':
    
        # (1) Fill average
        clustering_idx = df_clustering.index.tolist()
        df_clustering = df_clustering.fillna(df.mean())
    
    elif missing_cond == 'drop': 
        # (2) Drop null
        clustering_idx = df_clustering.dropna(axis=0).index.tolist()
        df_clustering = df_clustering.iloc[clustering_idx,:]
        
    elif missing_cond == 'interpolate':
        
        clustering_idx = df_clustering.index.tolist()
        df_clustering = df_clustering.interpolate()
        
    return df_clustering, clustering_idx, df