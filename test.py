from tensorflow.keras.models import load_model
from data_utils import *
import json
import argparse
import logging

TEST_PATH = '/datastore/Openedsdata2020/openEDS2020-GazePrediction/test'
VALIDATION_PATH = '/datastore/Openedsdata2020/openEDS2020-GazePrediction/validation'

def init_main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m", "--model_path", dest="model_path", default="", help="The path to the saved model")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=8, type=int, help="The batch size for inference")
    parser.add_argument("-tf", "--test_files_directory", dest="test_dir", default="", help="The base directory of the test files")
    parser.add_argument("-s", "--steps", dest="prediction_steps", default=None, type=int, help="The number of steps to run")
    parser.add_argument("-pf", "--predictions_file_name", dest="predictions_file", default="", help="The number of steps to run")
    return parser.parse_args()
    
def load_saved_model(model_path, load_custom_objects=True):       
    if load_custom_objects == False:
        model = load_model(model_path)
    else:
        model = load_model(model_path, custom_objects={'distributed_euclidean_distance': distributed_euclidean_distance}, compile=False)
        logging.info("Loaded model with custom loss function.")
    return model



if __name__ == '__main__':

    args = init_main() 

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = load_saved_model(args.model_path)   
    
    logging.info("Preparing data...")
    test_dataset, sequence_names = get_dataset(TEST_PATH, sequence=True, batch_size=args.batch_size, inference=True)
    
    logging.info("Starting predictions...")
    preds = model.predict(test_dataset, verbose=1)
    
    c = {}
    for i in range(len(preds)):
        d = {}
        for j in range(len(preds[i])):
            d[str(j+50)] = preds[i][j].tolist()
        seq_name = sequence_names[i][0].split("/")[-2]
        c[seq_name] = d

    with open(args.predictions_file, "w") as outfile:
        json.dump(c, outfile)