# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:37:10 2021

@author: USER
"""
import tensorflow as tf
import keras.layers as L
from tensorflow.keras import optimizers, Sequential, Model

def modeling(ts, n_feature):
    encoder_decoder = Sequential()
    encoder_decoder.add(L.LSTM(ts, activation='relu', input_shape = (ts, n_feature), return_sequences=True))
    encoder_decoder.add(L.LSTM(int(ts/2), activation='relu', return_sequences=True))
    encoder_decoder.add(L.LSTM(int(ts/2), activation='sigmoid'))
    encoder_decoder.add(L.RepeatVector(ts))
    encoder_decoder.add(L.LSTM(int(ts/2), activation='relu', return_sequences=True))
    encoder_decoder.add(L.LSTM(int(ts/2), activation='relu', return_sequences=True))
    encoder_decoder.add(L.TimeDistributed(L.Dense(n_feature)))
    encoder_decoder.summary()
    
    return encoder_decoder