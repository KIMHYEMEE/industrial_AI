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


# 2. Data concat

for f in file_list:
    tmp = pd.read_csv('./upzip_data/'+f)
    tmp = tmp.drop([0])
    
    try:
        df = pd.concat((df, tmp))
    except:
        df = tmp.copy()
        
        
# 3. Data Formating
df.index = range(np.shape(df)[0])

df.TIME_STAMP = pd.to_datetime(df.TIME_STAMP , format='%Y-%m-%d %H:%M:%S')
df.UTC = pd.to_datetime(df.UTC , format='%Y-%m-%d %H:%M:%S')

# 4. Save data frame
df.to_csv('data.csv',index = False)