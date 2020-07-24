import tensorflow as tf
import argparse
from pathlib import Path
import json

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from tensorflow.keras import Model
import numpy as np

from data_utils import *
from models.static_backbone import *



#Set random seed for reproducibility
tf.random.set_seed(12)

TRAIN_PATH = '/datastore/Openedsdata2020/openEDS2020-GazePrediction/train'
TEST_PATH = '/datastore/Openedsdata2020/openEDS2020-GazePrediction/test'
VALIDATION_PATH = '/datastore/Openedsdata2020/openEDS2020-GazePrediction/validation'


def init_main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m", "--model_name", dest="model_name", default="", help="The model version.")
    parser.add_argument("-c", "--callback_path", dest="callback_path", default="", help="The base directory to save data from callbacks. ")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="The batch size for inference")
    parser.add_argument("-bf", "--buffer_size", dest="buffer_size", default=10000, type=int, help="The shuffle buffer size")
    parser.add_argument("-lr", "--learning_rate", dest="learning_rate", default=0.0005, type=float, help="The training learning rate.")
    parser.add_argument("-d", "--dropout", dest="dropout", default=0.3, type=float, help="The drop out after FC1.")
    parser.add_argument("-u", "--fc1_units", dest="fc1_units", default=256, type=int, help="The shuffle buffer size")
    parser.add_argument("-e", "--epochs", dest="epochs", default=50, type=int, help="The number of epochs to train.")    
    
    return parser.parse_args()


if __name__ == '__main__':

    args = init_main() 

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        if args.model_name == "resnet18_v1":
            model = model_resnet18_v1(args.fc1_units, args.dropout, args.learning_rate)           
        elif  args.model_name == "resnet18_v2":
            model = model_resnet18_v2(args.fc1_units, args.dropout, args.learning_rate)   
        elif  args.model_name == "resnet50_v1":
            model = model_resnet50_v1(args.fc1_units, args.dropout, args.learning_rate)      
        elif  args.model_name == "vgg16":
            model = model_vgg16(args.fc1_units, args.dropout, args.learning_rate)                   
        else:           
            raise ValueError("Model {0} not found.".format(args.model_name))
                             
            
    #create callback paths
    Path(os.path.join(args.callback_path,"Logs")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.callback_path,"CheckPoints")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.callback_path,"TensorBoard")).mkdir(parents=True, exist_ok=True)
    
    # CSV logger
    csvlogger_callback = CSVLogger(os.path.join(args.callback_path,"Logs", "log.csv"), append=True, separator=";")

    # Model checkpoints
    filepath = os.path.join(args.callback_path, "CheckPoints", "checkpoint" + "-{epoch:03d}-{val_loss:.5f}" + ".h5")
    checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False)

    #Tensorboard
    tensorboard_callback = TensorBoard(log_dir=os.path.join(args.callback_path,"TensorBoard"))

    #Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

    training_dataset, training_samples = get_dataset(TRAIN_PATH, sequence=False, buffer_size=args.buffer_size, batch_size=args.batch_size)
    validation_dataset, validation_samples = get_dataset(VALIDATION_PATH, sequence=False, buffer_size=args.buffer_size, batch_size=args.batch_size)
    
    model_params_file = os.path.join(args.callback_path,"model_params.json")

    with open(model_params_file, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    model.fit(
        x=training_dataset, 
        validation_data=validation_dataset, 
        epochs = args.epochs, 
        callbacks=[csvlogger_callback, checkpoint_callback, tensorboard_callback, early_stopping],
        steps_per_epoch = training_samples//args.batch_size,
        validation_steps = validation_samples//args.batch_size
    )