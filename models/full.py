import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, TimeDistributed, LSTM, Dense, Reshape, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CosineSimilarity
from tensorflow.keras.models import load_model
from data_utils import *
import tensorflow as tf
from pathlib import Path

seq_len = 50
output_len = 5

def model_cnn_lstm(static_model_path, learning_rate, train_backbone=False):
    # Loading our pre-trained static backbone model    
    resnet = load_model(static_model_path, compile=False)

    #Remove the dense layers from the model
    resnet_no_top = Model(inputs=resnet.inputs, outputs=resnet.layers[-4].output)

    if train_backbone == False:
        for layer in resnet_no_top.layers:
            layer.trainable = False
    
    input_layer = Input(shape=(seq_len, 400, 640, 1))
    curr_layer = TimeDistributed(resnet_no_top)(input_layer)
    curr_layer = Reshape(target_shape=(seq_len, 2048))(curr_layer)    
    lstm_enc = LSTM(128)(curr_layer)
    repeater = RepeatVector(output_len)(lstm_enc)
    lstm_dec = LSTM(128, return_sequences=True)(repeater)
    fc_1 = TimeDistributed(Dense(64, activation='relu'))(lstm_dec)
    fc_2 = TimeDistributed(Dense(3))(fc_1)
    model = Model(inputs=input_layer, outputs=fc_2)   
    
    opt = Adam(learning_rate=learning_rate)
    
    model.compile(metrics=['cosine_similarity','mse'], optimizer=opt, loss=CosineSimilarity())   

    return model
