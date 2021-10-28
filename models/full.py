import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, TimeDistributed, LSTM, Dense, Reshape, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CosineSimilarity
from tensorflow.keras.models import load_model
from data_utils import *
import tensorflow as tf
from pathlib import Path
from .static_backbone import *

output_len = 5

def model_cnn_lstm(static_model_path, learning_rate, train_backbone=False, seq_len=50):
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

def model_cnn_lstm_v2(static_model_path, learning_rate, train_backbone=False, seq_len=50):
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
    lstm_enc = LSTM(200)(curr_layer)
    repeater = RepeatVector(output_len)(lstm_enc)
    lstm_dec = LSTM(200, return_sequences=True)(repeater)
    fc_1 = TimeDistributed(Dense(100, activation='relu'))(lstm_dec)
    fc_2 = TimeDistributed(Dense(3))(fc_1)
    model = Model(inputs=input_layer, outputs=fc_2)   
    
    opt = Adam(learning_rate=learning_rate)
    
    model.compile(metrics=['cosine_similarity','mse'], optimizer=opt, loss=CosineSimilarity())   

    return model

def model_cnn_lstm_v3(static_model_path, learning_rate, train_backbone=False, seq_len=50):
    # Loading our pre-trained static backbone model    
    
    resnet = model_resnet50_v1(512, 0.3, 0.001)
    resnet.load_weights(static_model_path)    

    #Remove the dense layers from the model
    resnet_no_top = Model(inputs=resnet.inputs, outputs=resnet.layers[-4].output)

    if train_backbone == False:
        for layer in resnet_no_top.layers:
            layer.trainable = False
    
    input_layer = Input(shape=(seq_len, 400, 640, 1))
    curr_layer = TimeDistributed(resnet_no_top)(input_layer)
    curr_layer = Reshape(target_shape=(seq_len, 2048))(curr_layer)    
    lstm_enc = LSTM(256)(curr_layer)
    repeater = RepeatVector(output_len)(lstm_enc)
    lstm_dec = LSTM(256, return_sequences=True)(repeater)
    fc_1 = TimeDistributed(Dense(128, activation='relu'))(lstm_dec)
    fc_2 = TimeDistributed(Dense(3))(fc_1)
    model = Model(inputs=input_layer, outputs=fc_2)   
    
    opt = Adam(learning_rate=learning_rate)
    
    model.compile(metrics=['cosine_similarity','mse'], optimizer=opt, loss=CosineSimilarity())   

    return model

def model_cnn_lstm_v4(static_model_path, learning_rate, train_backbone=False, seq_len=50):
    # Loading our pre-trained static backbone model    
    resnet = load_model(static_model_path, compile=False)

    #Remove the dense layers from the model
    resnet_no_top = Model(inputs=resnet.inputs, outputs=resnet.layers[-4].output)

    if train_backbone == False:
        for layer in resnet_no_top.layers:
            if (not layer.name.startswith("conv5")):
                layer.trainable = False
    
    input_layer = Input(shape=(seq_len, 400, 640, 1))
    curr_layer = TimeDistributed(resnet_no_top)(input_layer)
    curr_layer = Reshape(target_shape=(seq_len, 2048))(curr_layer)    
    lstm_enc = LSTM(32)(curr_layer)
    repeater = RepeatVector(output_len)(lstm_enc)
    lstm_dec = LSTM(32, return_sequences=True)(repeater)
    fc_1 = TimeDistributed(Dense(32, activation='relu'))(lstm_dec)
    fc_2 = TimeDistributed(Dense(3))(fc_1)
    model = Model(inputs=input_layer, outputs=fc_2)   
    
    opt = Adam(learning_rate=learning_rate)
    
    model.compile(metrics=['cosine_similarity','mse'], optimizer=opt, loss=distributed_euclidean_distance)   

    return model

def model_cnn_lstm_v5(static_model_path, learning_rate, train_backbone=False, seq_len=50):
    # Loading our pre-trained static backbone model    
    resnet = load_model(static_model_path, compile=False)

    #Remove the dense layers from the model
    resnet_no_top = Model(inputs=resnet.inputs, outputs=resnet.layers[-4].output)

    if train_backbone == False:
        for layer in resnet_no_top.layers:
            if (not (layer.name.startswith("conv5") or layer.name.startswith("conv4"))):
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
    
    model.compile(metrics=['cosine_similarity','mse'], optimizer=opt, loss=distributed_euclidean_distance)   

    return model