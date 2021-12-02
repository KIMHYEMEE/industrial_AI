# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:39:17 2021

@author: USER

Ref.: https://www.kaggle.com/dimitreoliveira/time-series-forecasting-with-lstm-autoencoders

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# 1. Load data

fn_dir = 'C:/GIT/industrial_AI'
data_dir = 'C:/BaseData/Class/2021-2/Industrial AI/Progress'
save_dir = 'C:/BaseData/Class/2021-2/Industrial AI/Progress'
file_name = 'data_10min.csv'
rpm = 'ME1_RPM_ECC'

lr= 0.0001
batch = 128
epochs = 100
ts = 6

os.chdir(fn_dir)

from data_function import *
from LSTM_autoencoder_modeling import *
from outlier_detection import *

for del_depth in ['non_depth', 'depth']:
    for missing_cond in ['interpolate']:#['mean', 'interpolate']: # when you drop the data, it is impossible to consider the time series characteristics

        df, clustering_idx, df_origin = load_preprocessing(data_dir, file_name, del_depth, missing_cond)
        df_clustering = df.copy()
        
        # 4. Data scaling
        
        scaler = MinMaxScaler()
        df_clustering.iloc[:,1:] = scaler.fit_transform(df_clustering.iloc[:,1:])
        
        
        # 5. Data formating for training
        
        n_feature = np.shape(df_clustering.iloc[:,1:])[1]
        n_row = np.shape(df_clustering)[0]-ts+1
        
        
        
        for i in range(ts):
            tmp = np.asarray([df_clustering.iloc[i:i+n_row,1:]])
            tmp = np.swapaxes(tmp, 0,1)
            
            try:
                train_x = np.append(train_x, tmp, axis=1)
            except:
                train_x = tmp
        
        
        # 6. Modeling: LSTM Autoencoder
        
        encoder_decoder = modeling(ts, n_feature)
        
        
        adam = optimizers.Adam(lr)
        encoder_decoder.compile(loss='mse', optimizer=adam)
        
        
        encoder_decoder_history = encoder_decoder.fit(train_x, train_x, 
                                                      batch_size=batch, 
                                                      epochs=epochs, 
                                                      verbose=2)
        
        encoder = Model(inputs=encoder_decoder.inputs, outputs=encoder_decoder.layers[2].output)
        
        # 7. Prediction
        
        pred = encoder.predict(train_x)
        df_pred = pd.DataFrame({'TIME_STAMP':np.asarray(df_clustering['TIME_STAMP'][(ts-1):].tolist())})
        pred = pd.DataFrame(pred)
        pred.columns = [f'feature{i+1}' for i in range(len(pred.columns))]
        
        for  c in pred.columns:
            df_pred[c] = pred[c]
        
        
        df_pred[rpm] = df[rpm][(ts-1):].tolist()
        
        # 8. Save compression results
        
        os.chdir(save_dir)
        df_pred.to_csv(f'./1125/rlt/lstm_autoencoder_result_{del_depth}_{missing_cond}.csv',index = False)
        
        
        # 9. Training each model and save detection result
        
        model_list = ['ABOD','LOF','CBLOF','LODA','IF','auto_encoder','OCSVM']
        
        clustering_idx = clustering_idx[(ts-1):]
        
        rlt_df = pd.DataFrame({'TIME_STAMP': df['TIME_STAMP'][clustering_idx],
                               'Result':[np.nan for _ in range(len(clustering_idx))]})
        
        
        n_features_clustering = int(ts/2)
        
        for model_name in model_list:
            model = clustering_modeling(model_name,n_features_clustering )
            model.fit(df_pred.iloc[:,1:(1+n_features_clustering)])
            
            rlt = model.decision_scores_
            rlt_df[model_name] = determine_outlier(rlt)
            
            
        # 10. Calculate final result
         
        tmp = [0 for _ in range(np.shape(rlt_df)[0])]
        idx = np.where(rlt_df.iloc[:,1:].sum(axis=1) >= 4)[0]
         
        for i in idx:
            tmp[i] = 1
         
        rlt_df['Result'] =  tmp
        
        # 11. Svae the result
        
        rlt_df.to_csv(f'./1125/rlt/clustering_LSTM-{del_depth}-{missing_cond}.csv',index = False)
        
        # 12. Result join
        
        rlt_join = pd.DataFrame({'TIME_STAMP': df_clustering['TIME_STAMP'],
                                 'Cluster': rlt_df['Result']})
        
        rlt_final = df_origin[['TIME_STAMP', rpm]]
        
        rlt_final = pd.merge(left = rlt_final, right=rlt_join, how='left', on='TIME_STAMP')
         
         
        # 10. Plot
        
        
        l = 15000
        
        for i in range(int(np.shape(rlt_final)[0]/l)+1):
            try:
                tmp = rlt_final.iloc[(i*l):((i+1)*l),:]
            except:
                tmp = rlt_final.iloc[(i*l):,:]
             
            fig_name = f'./1125/fig/LSTM/{del_depth}/{missing_cond}/outlier_{str(tmp["TIME_STAMP"].tolist()[0])[:10]}-{str(tmp["TIME_STAMP"].tolist()[-1])[:10]}.png'
             
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