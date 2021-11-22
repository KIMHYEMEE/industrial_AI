# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:52:57 2021

@author: USER
"""

import os
import pandas as pd
import numpy as np

data_dir = 'C:/BaseData/Class/2021-2/Industrial AI/Progress'

os.chdir(data_dir)

# 1. Load data

df = pd.read_csv('data.csv')
df = pd.DataFrame(df)
df_col= df.columns

# 2. Convert the timestamp to every 10 minutes

df['TIME_STAMP_idx'] = df['TIME_STAMP'].str[:15] + ['0:00' for _ in range(len(df['TIME_STAMP']))]

# 3. Get target timestamp

ts_idx = np.unique(df['TIME_STAMP_idx'])

# 4. Make a column list

c_list = ['TIME_STAMP']

for c in df_col[2:]:
    c_list.append(c)

# 5. Convert -9999 to nan(-9999 is error value)

for c in c_list[1:]:
    i = df[c].isin([-9999])
    df[c][i] = [np.nan for _ in range(sum(i))]

# 6. Define result data frame

df_10min = pd.DataFrame({str(c):[] for c in c_list})


# 7. Loop for result data frame

for ts in ts_idx:
    df_tmp = pd.DataFrame(df[df['TIME_STAMP_idx'] == ts][df_col[2:]])
    
    try:
        df_tmp['RUDDER_ANGLE'] = df_tmp['RUDDER_ANGLE'].astype(float)
    except:
        pass
    
    mean_tmp = df_tmp.mean(numeric_only=None)
    mean_tmp['TIME_STAMP'] = ts
    
    
    c_nan = np.array(c_list)[-pd.Series(c_list).isin(mean_tmp.index)]
    for c in c_nan:
        mean_tmp[c] = np.nan
    
    mean_tmp = pd.DataFrame(mean_tmp)
    mean_tmp = mean_tmp.transpose()
    df_10min = df_10min.append(mean_tmp)
    
    print(str(np.shape(df_10min)[0]),"/",str(len(ts_idx)),"(",str(np.shape(df_10min)[0]/len(ts_idx)),")")
    
    
# 8. Rearrange index and change timestamp data type

df_10min.index = range(np.shape(df_10min)[0])
df_10min.TIME_STAMP = pd.to_datetime(df_10min.TIME_STAMP , format='%Y-%m-%d %H:%M:%S')

# 9. Save data frame

df_10min.to_csv('data_10min.csv', index=False)