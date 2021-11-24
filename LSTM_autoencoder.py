# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:39:17 2021

@author: USER

Ref.: https://www.kaggle.com/dimitreoliveira/time-series-forecasting-with-lstm-autoencoders

"""

import os
import pandas as pd
import numpy as np

# 1. Load data

data_dir = 'C:/BaseData/Class/2021-2/Industrial AI/Progress'

os.chdir(data_dir)

df = pd.read_csv('data_10min.csv')

df_des = df.describe()

df_col = df.columns[np.where(df_des.loc['count'] >= 50000)]

# 3. Select columns to train
df_clustering = df[df_col]


# missing data processing

# (1) Fill average

# clustering_idx = df_clustering.index.tolist()
#df_clustering = df_clustering.fillna(df.mean())

# (2) Fill regression

clustering_idx = df_clustering.dropna(axis=0).index.tolist()
df_clustering = df_clustering.iloc[clustering_idx,:]

# 4. Data scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_clustering.iloc[:,1:] = scaler.fit_transform(df_clustering.iloc[:,1:])


# 5. Data formating for training
ts = 12
n_feature = np.shape(df_clustering.iloc[:,1:])[1]
n_row = np.shape(df_clustering)[0]-ts+1

"""
for i in range(n_row):
    print(str(i),'/',str(n_row))
    tmp = np.asarray(df_clustering.iloc[i:(i+ts),1:])
    
    try:
        train_x = np.append(train_x, np.array([tmp]), axis=0)
    except:
        train_x = np.array([tmp])
"""

for i in range(ts):
    tmp = np.asarray([df_clustering.iloc[i:i+n_row,1:]])
    tmp = np.swapaxes(tmp, 0,1)
    
    try:
        train_x = np.append(train_x, tmp, axis=1)
    except:
        train_x = tmp


# 6. Modeling: LSTM Autoencoder
import tensorflow as tf
import keras.layers as L
from tensorflow.keras import optimizers, Sequential, Model

lr= 0.0001
batch = 128
epochs = 100

encoder_decoder = Sequential()
encoder_decoder.add(L.LSTM(ts, activation='relu', input_shape = (ts, n_feature), return_sequences=True))
encoder_decoder.add(L.LSTM(int(ts/2), activation='relu', return_sequences=True))
encoder_decoder.add(L.LSTM(int(ts/2), activation='relu'))
encoder_decoder.add(L.RepeatVector(ts))
encoder_decoder.add(L.LSTM(int(ts/2), activation='relu', return_sequences=True))
encoder_decoder.add(L.LSTM(int(ts/2), activation='relu', return_sequences=True))
encoder_decoder.add(L.TimeDistributed(L.Dense(n_feature)))
encoder_decoder.summary()

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

# 8. Save results
    
df_pred.to_csv('lstm_autoencoder_result.csv',index = False)
