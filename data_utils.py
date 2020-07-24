import numpy as np
import os
from tensorflow.keras.preprocessing import image
import copy
import random
import tensorflow as tf
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

AUTOTUNE = tf.data.experimental.AUTOTUNE

def generate_ids(data_files: str, label_files = '', bin_by_sequence=True, inference=False):
    data_list = []
    labels_list = []
    sequence_ids = sorted(os.listdir(data_files))
    label_ids = [x + ".txt" for x in sequence_ids]
    
    for sequence, label in zip(sequence_ids, label_ids):        
        samples = sorted(os.listdir(os.path.join(data_files, sequence)))
        if inference == False:
            labels = pd.read_csv(os.path.join(label_files,label), 
                                 delimiter=',', 
                                 skiprows=0, 
                                 names=('id','x','y','z'), 
                                 dtype={'id':str})

            labels.set_index('id', inplace=True)
            targets_list = []
        
        images = []        
        
        for sample in samples:
            sample_name = os.path.splitext(sample)[0]
            sample_id = sequence + "_" + sample_name
            images.append(os.path.join(data_files, sequence, sample))
            
            if inference == False:
                targets_list.append(labels.loc[[sample_name]].values[0])
                
        data_list.append(images)
        
        if inference ==False:
            labels_list.append(targets_list)
        
    if bin_by_sequence==False:
        flat_data_list = [item for sublist in data_list for item in sublist]
        flat_labels_list = [item for sublist in labels_list for item in sublist]
        return flat_data_list, flat_labels_list
        
    return data_list, labels_list 

def to_supervised(train, labels='', n_input=50, n_out=5, inference=False):
    X, y = list(), list()
    for i in range(len(train)):
        data = train[i]        
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance            
            lbls = []
            if out_end <= len(data):
                X.append(data[in_start:in_end])
                
                if inference == False:
                    lbls = labels[i][in_end:out_end]
                    y.append(lbls)
                    
            # move along one time step
            in_start += 1
            
    return np.array(X), np.array(y)

def euclidean_distance(y_true, y_pred):
    """
    Compute average euclidean distance
    :param y_true: list of ground truth labels
    :param y_pred: list of predicted labels
    :return: euclidean distance
    """
    return K.mean(K.sqrt(K.sum(K.square(y_true - y_pred), axis=1, keepdims=True)))    

def distributed_euclidean_distance(y_true, y_pred):
    """
    Compute average euclidean distance for distributed strategy training
    :param y_true: list of ground truth labels
    :param y_pred: list of predicted labels
    :return: euclidean distance
    """
    per_example_loss = K.sqrt(K.sum(K.square(y_true - y_pred), axis=1, keepdims=True))
    return tf.nn.compute_average_loss(per_example_loss)

@tf.function
def load_and_preprocess_image(path):
    """
    Loads and normalizes an image
    :param path: path to image file    
    :return: normalized image tensor
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
#     image = tf.image.resize(image, [HEIGHT, WIDTH])
    image /= 255.0
    return image

def load_sequence(seq):
    imgs = tf.map_fn(
        load_and_preprocess_image, seq,  dtype=tf.float32, parallel_iterations=10)
    return imgs

def configure_for_performance(ds, buffer_size=1000, batch_size=16, enable_cache = False):      
    
    if enable_cache == True:
        ds = ds.cache()
        
    ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def prepare_dataset(x, y, sequence=False, buffer_size=1000, batch_size=16, enable_cache=False):
    images_dataset = tf.data.Dataset.from_tensor_slices(x)
    labels_dataset = tf.data.Dataset.from_tensor_slices(y)

    if sequence == True:
        images_dataset = images_dataset.map(load_sequence, num_parallel_calls=AUTOTUNE)
    else:
        images_dataset = images_dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    
    data_label_ds = tf.data.Dataset.zip((images_dataset, labels_dataset))

    data_label_ds = configure_for_performance(data_label_ds, 
                                              buffer_size=buffer_size, 
                                              batch_size=batch_size, 
                                              enable_cache = enable_cache)
    return data_label_ds

def get_dataset(base_path, sequence=False, buffer_size=1000, batch_size=16):
    data_path = os.path.join(base_path, 'sequences')
    labels_path = os.path.join(base_path, 'labels')

    data_paths_list, labels_list = generate_ids( data_path, labels_path, bin_by_sequence=sequence)    
    
    if sequence == True:
        x, y = to_supervised(data_paths_list, labels_list)    
        dataset = prepare_dataset(x, y, sequence=True, buffer_size=buffer_size, batch_size = batch_size, enable_cache=False)
        samples = len(x)
    else:
        dataset = prepare_dataset(data_paths_list, labels_list, sequence=False, buffer_size=buffer_size, batch_size = batch_size, enable_cache=True)
        samples = len(data_paths_list)

    return (dataset, samples)
   