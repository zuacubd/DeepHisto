import os
import sys
import gc
import math
import argparse
import time

from config.config import config_class_level
#from patch.patch_extraction import get_patch_extract
from patch.patch_extraction_high_resolution import get_patch_extract

#from model.training import training_model
#from model.testing import testing_model
#from model.prediction import prediction_model

#from predict.merger_models_predict_patch import ensemble_model
#from predict.model_predict_slide import predict_slide_model
#from predict.merger_models_predict_slide import combine_predict_slide_model

parser = argparse.ArgumentParser(description='A tool used to parse user arguments')
parser.add_argument('-extract', '--patch_extract', nargs='?', type=bool, required=False, help='Extract patch of slides (p)')
parser.add_argument('-train', '--train_model', nargs='?', type=bool, required=False, help='Train model (tr)')
parser.add_argument('-test','--test_model', nargs='?', type=bool, required=False, help='Test model (te)')
parser.add_argument('-predict','--predict_model', nargs='?', type=bool, required=False, help='Prediction model (pr)')
parser.add_argument('-combine','--combine_model', nargs='?', type=bool, required=False, help='Combination of predictions')
parser.add_argument('-criteria','--combine_criteria', nargs='?', type=str, required=False, help='Combination criteria')
parser.add_argument('-slide','--slide_predict', nargs='?', type=bool, required=False, help='Prediction of slide')
parser.add_argument('-combine-slide','--combine_slide', nargs='?', type=bool, required=False, help='Combination of slide model')
parser.add_argument('-network', '--network_architecture', nargs='?', type=str, required=False, help='Network architecture')
parser.add_argument('-level', '--resolution_level', nargs='?', type=int, required=False, help='Resolution level (l)')
parser.add_argument('-size', '--patch_size', nargs='?', type=int, required=False, help='Patch size (s)')
parser.add_argument('-overlap', '--patch_overlap', nargs='?', type=bool, required=False, help='Patch overlap (po)')
parser.add_argument('-ext', '--save_ext', nargs='?', type=str, required=False, help='Extension of the Patch file')

patch_extract = None
train_model =  None
test_model = None
predict_model = None
combine_model = None
combine_criteria = None
predict_slide = None
combine_slide_model = None
network_name = None
resolution_level = None
patch_size = None
patch_overlap = None
save_ext = None

def parse_arguments():
    """ Function used to parse the arguments provided to the script"""
    # Parsing the args
    args = parser.parse_args()

    global patch_extract
    global train_model
    global test_model
    global predict_model
    global combine_model
    global combine_criteria
    global predict_slide
    global combine_slide_model
    global network_name
    global resolution_level
    global patch_size
    global patch_overlap
    global save_ext

    # Retrieving the args
    patch_extract = args.patch_extract
    print("Patch extaction: {0}".format(patch_extract))

    train_model = args.train_model
    print("Train model: {0}".format(train_model))

    test_model = args.test_model
    print("Test model: {0}".format(test_model))

    predict_model = args.predict_model
    print("Prediction model: {0}".format(predict_model))

    combine_model = args.combine_model
    print("Combination of prediction: {0}".format(combine_model))

    combine_criteria = args.combine_criteria
    print("Combination critera: {0}".format(combine_criteria))

    predict_slide = args.slide_predict
    print("Prediction of slide: {0}".format(predict_slide))
    combine_slide_model = args.combine_slide
    print("Combination of model for slide level: {0}".format(combine_slide_model))

    network_name = args.network_architecture
    print("Network architecture: {0}".format(network_name))

    resolution_level = args.resolution_level
    print("Resolution level: {0}".format(resolution_level))

    patch_size = args.patch_size
    print("Patch size: {0}".format(patch_size))

    patch_overlap = args.patch_overlap
    print("Patch overlap: {0}".format(patch_overlap))

    save_ext = args.save_ext
    print("Patch ext: {0}".format(save_ext))


if __name__ == '__main__':
    print ("start at {0}".format(time.time()))
    parse_arguments()

    if patch_extract:
        get_patch_extract(config_class_level, resolution_level, patch_size, patch_overlap, save_ext)

    elif train_model:
        training_model(network_name)

    elif test_model:
        testing_model(network_name)

    elif predict_model:
        prediction_model(network_name)

    elif combine_model:
        ensemble_model(combine_criteria)

    elif predict_slide:
        predict_slide_model(network_name)

    elif combine_slide_model:
        combine_predict_slide_model()

    else:
        print ("Wrong command.")

    print ("done at {0}".format(time.time()))
