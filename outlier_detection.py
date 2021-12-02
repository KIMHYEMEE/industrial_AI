# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:44:05 2021

@author: USER
"""

import numpy as np

# 4. Modeling function
def clustering_modeling(model_name, n_features):
    if model_name == 'KNN':
        from pyod.models.knn import KNN as pyod_model
        clustering_model = pyod_model()
    
    elif model_name == 'ABOD':
        from pyod.models.abod import ABOD as pyod_model
        clustering_model = pyod_model()
    
    elif model_name == 'LOF':
        from pyod.models.lof import LOF as pyod_model
        clustering_model = pyod_model()
        
    elif model_name == 'CBLOF':
        from pyod.models.cblof import CBLOF as pyod_model
        clustering_model = pyod_model()    
    elif model_name == 'LODA':
        from pyod.models.loda import LODA as pyod_model
        clustering_model = pyod_model()
        
    elif model_name == 'IF': # Isolation Forest
        from pyod.models.iforest import IForest as pyod_model
        clustering_model = pyod_model()
        
    elif model_name == 'OCSVM':
        from pyod.models.ocsvm import OCSVM as pyod_model
        clustering_model = pyod_model()
        
    elif model_name == 'auto_encoder': 
        from pyod.models.auto_encoder import AutoEncoder as pyod_model
        clustering_model = pyod_model(epochs=100, contamination=0.1,hidden_neurons=[int(n_features/2)])


    return clustering_model


def determine_outlier(rlt): # top 1% outlier score
    tmp = rlt.copy()
    tmp.sort()
    target_idx = len(tmp) - int(0.01 / (1/len(tmp)))
    target_val = tmp[target_idx]
    
    
    rlt_tf = np.array([0 for _ in range(len(rlt))])
    rlt_tf[rlt > target_val] = 1

    return rlt_tf