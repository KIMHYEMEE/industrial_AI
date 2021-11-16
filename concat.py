# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 16:44:39 2021

@author: USER
"""

import os
import pandas as pd
import numpy as np

path = 'C:/BaseData/Class/2021-2/Industrial AI/Progress'

os.chdir(path)

# 1. Get file list

file_list = os.listdir('./upzip_data')

file_group = pd.DataFrame(file_list)
file_group.columns = ['name']

file_group = np.unique(file_group['name'].str[:9])

# 2. Data concat

data_group = {file: np.nan for file in file_group}

col = pd.read_csv('./upzip_data/'+file_list[0]).columns

for f in file_list:
    print(f)
    tmp = pd.read_csv('./upzip_data/'+f)
    tmp = tmp.drop([0])
    tmp.columns = col
    
    try:
        
        data_group[f[:9]] = pd.concat((data_group[f[:9]], tmp))
    except:
        data_group[f[:9]] = tmp.copy()


for f in file_group:
    try:
        df = pd.concat((df, data_group[f]))
    except:
        df = data_group[f].copy()
    

        
# 3. Data Formating
df.index = range(np.shape(df)[0])

df.TIME_STAMP = pd.to_datetime(df.TIME_STAMP , format='%Y-%m-%d %H:%M:%S')
#df.UTC = pd.to_datetime(df.UTC , format='%Y-%m-%d %H:%M:%S')

# 4. Save data frame
df.to_csv('data.csv',index = False)
