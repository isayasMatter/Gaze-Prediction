import tensorflow as tf
import numpy as np

from tensorflow.python.keras.applications import resnet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CosineSimilarity


from data_utils import *
from pathlib import Path

def stack_fn(x):
    x = resnet.stack1(x, 64, 2, stride1=1, name='conv2')
    x = resnet.stack1(x, 128, 2, name='conv3')
    x = resnet.stack1(x, 256, 2, name='conv4')
    return resnet.stack1(x, 512, 2, name='conv5')

def get_model_resnet18_v1(fc1_units, drop_out, learning_rate):
    init_model = resnet.ResNet(stack_fn, 
                               include_top=False, 
                               weights=None, 
                               model_name='resnet18', 
                               preact=False, 
                               use_bias=True, 
                               input_shape=(400,640,1), 
                               pooling='avg')

    x = init_model.output

    x = Dense(fc1_units, activation='relu')(x)
    x = Dropout(drop_out)(x)
    out = Dense(3)(x)
    model = Model(inputs = init_model.input, outputs=out)
    
    opt = Adam(learning_rate=learning_rate)
    model.compile(metrics=['cosine_similarity','mse'], optimizer=opt, loss=distributed_euclidean_distance) 
        
    return model

def get_model_resnet18_v2(fc1_units, drop_out, learning_rate):
    init_model = resnet.ResNet(stack_fn, 
                               include_top=False, 
                               weights=None, 
                               model_name='resnet18', 
                               preact=False, 
                               use_bias=True, 
                               input_shape=(400,640,1), 
                               pooling='avg')

    x = init_model.output

    x = Dense(fc1_units, activation='relu')(x)
    x = Dropout(drop_out)(x)
    out = Dense(3)(x)
    model = Model(inputs = init_model.input, outputs=out)
    
    opt = Adam(learning_rate=learning_rate)
    model.compile(metrics=['cosine_similarity','mse'], optimizer=opt, loss=CosineSimilarity()) 
        
    return model
        
def get_model_resnet50_v1(fc1_units, drop_out, learning_rate):
    init_model = ResNet50(include_top=False, input_shape = (400,640,1), weights=None)

    x = init_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(fc1_units, activation='relu')(x)
    x = Dropout(args.drop_out)(x)
    out = Dense(3)(x)
    model = Model(inputs = init_model.input, outputs=out)

    opt = Adam(learning_rate=learning_rate)
    model.compile(metrics=['cosine_similarity','mse'], optimizer=opt, loss=CosineSimilarity()) 

    return model
