# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 16:15:21 2021

@author: USER
"""

import zipfile
import os
import shutil

base_path = 'C:/BaseData/Class/2021-2/Industrial AI/Progress/data'
save_path = 'C:/BaseData/Class/2021-2/Industrial AI/Progress/upzip_data'
os.chdir(base_path)

year_path = os.listdir(base_path)

month_path = {}
day_path = {}
file_names = {}


for y in year_path:
    month_path[y] = os.listdir('./'+y)
    
    for m in month_path[y]:
        day_path[y+'-'+m] = os.listdir('./'+y+'/'+m)
        
        for d in day_path[y+'-'+m]:
            file_names[y+'-'+m+'-'+d] = os.listdir('./'+y+'/'+m+'/'+d)



for y in year_path:
    for m in month_path[y]:
        for d in day_path[y+'-'+m]:
            for f in file_names[y+'-'+m+'-'+d]:
                data_path = './'+y+'/'+m+'/'+d+'/'+f
                
                try:
                    with zipfile.ZipFile(data_path, 'r') as zip_ref:
                        zip_ref.extractall(save_path)
                except:
                    shutil.copy(data_path, save_path)