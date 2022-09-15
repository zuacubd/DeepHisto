'''
    This program is used to make prediction of each slide (i.e. GC or NGC) using the prediction probability of patch
'''

import os
import time
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report

from reader.dataset import get_all_lines

from config.config import config_class_level
from config.config import train_parameters
from config.config import output_parameters

def combine_predict_slide_model():
    ''' predicting the global level of each slides by fusing multiple networks '''

    num_classes = train_parameters['num_classes']
    test_slides_patch_gc_ngc_path = config_class_level['test_slides_patch_gc_ngc_bg_path']
    test_slides_gc_ngc_path = config_class_level['test_slides_gc_ngc_path']

    prediction_folder = output_parameters['prediction_folder']
    #network_names = ["alexnet", "vgg16", "vgg19", "googlenet", "resnet50", "resnet101"]
    network_names = ["alexnet", "vgg16", "googlenet", "resnet50"]
    all_network_names = "_".join(network_names)
    result_folder = output_parameters['result_folder']

    aggregators = ["voting", "sum", "mean", "median", "std", "var", "max"]
    slides_patches_actual_lines = get_all_lines(test_slides_patch_gc_ngc_path, False)
    slides, slides_actual, slides_patches_actual = get_slides_patches_labels(slides_patches_actual_lines)

    for agg_func in aggregators:

        print ("Processing {}".format(agg_func))
        result_file_path = result_folder + '/' + all_network_names + '_' + agg_func + '_' + 'classification.res'
        slides_patches_predicted_lines = get_combined_patches_predicted_labels(network_names, prediction_folder,\
                                                                               agg_func, num_classes)
        slides_patches_predicted = get_slides_patches_predicted(slides_patches_actual_lines, \
                                                                slides_patches_predicted_lines, num_classes)

        slides_predicted = get_slides_predicted_label(slides_patches_predicted, config_class_level)
        slides_true_labels = get_slides_true_labels(test_slides_gc_ngc_path, config_class_level)

        predicted_cls = get_predicted_cls(slides, slides_predicted)
        actual_cls = get_actual_cls(slides, slides_true_labels)
        cls_rep = classification_report(actual_cls, predicted_cls)

        with open(result_file_path, 'w') as fwriter:
            fwriter.write(cls_rep)


def get_slides_true_labels(test_slides_gc_ngc_path, config_class_level):
    slides_cls = {}
    with open(test_slides_gc_ngc_path, 'r') as fr:
        lines = fr.readlines()

    for line in lines:
        cline = line.rstrip()
        parts = cline.split("\t")
        slide_path = parts[0]
        label = parts[1]
        cls = config_class_level[label]
        slides_cls[slide_path] = cls

    return slides_cls



def get_combined_patches_predicted_labels(network_names, prediction_folder, agg_func, num_classes):

    networks_predicted_lines = []
    for network_name in network_names:
        predicted_path = prediction_folder + '/' + network_name + '.pred'
        patches_predicted_lines = get_all_lines(predicted_path, header=True)
        networks_predicted_lines.append(patches_predicted_lines)

    #print (len(networks_predicted_lines))

    merged_patches_predicted_lines = []
    num_lines = len(networks_predicted_lines[0])
    #print (num_lines)

    for idx in range(num_lines):
        networks_patch_predicted = []
        for ndx in range(len(network_names)):
            networks_patch_predicted.append(networks_predicted_lines[ndx][idx])

        fusion_patch_predicted = get_fusion_patch_predicted(network_names, networks_patch_predicted, agg_func, num_classes)
        merged_patches_predicted_lines.append(fusion_patch_predicted)

    return merged_patches_predicted_lines


def get_fusion_patch_predicted(network_names, networks_patch_predicted, agg_func, num_classes):

    networks_prob = np.zeros((num_classes, len(network_names)), dtype=np.float32)

    for idx in range(len(network_names)):
        line = networks_patch_predicted[idx]
        parts = line.split("\t")

        for cdx in range(num_classes):
            prob = float(parts[cdx+1])
            networks_prob[cdx, idx] = prob

    line = networks_patch_predicted[0]
    parts = line.split("\t")
    agg_line = []
    agg_line.append(parts[0])

    for cdx in range(num_classes):

        if agg_func == "mean":
            agg_prob = np.mean(networks_prob[cdx, :])

        elif agg_func == "median":
            agg_prob = np.median(networks_prob[cdx, :])

        elif agg_func == "std":
            agg_prob = np.std(networks_prob[cdx, :])

        elif agg_func == "var":
            agg_prob = np.var(networks_prob[cdx, :])

        elif agg_func == "max":
            agg_prob = np.max(networks_prob[cdx, :])

        elif agg_func == "sum":
            agg_prob = np.sum(networks_prob[cdx, :])

        elif agg_func == "voting":
            agg_prob = (networks_prob[cdx, :]>=0.5).sum()

        else:
            print ("wrong aggregator")

        agg_line.append(str(agg_prob))

    startIndex = num_classes + 1
    for k in range(startIndex, len(parts)):
        agg_line.append(parts[k])

    return "\t".join(agg_line)


def get_actual_cls(slides, slides_actual):

    actual = []
    for slide in slides:
        cls = slides_actual.get(slide)
        actual.append(int(cls))

    return actual


def get_predicted_cls(slides, slides_predicted):

    pred = []
    for slide in slides:
        cls = slides_predicted.get(slide)
        pred.append(int(cls))

    return pred


def get_slides_predicted_label(slides_patches_predicted, config_class_level):

    slides_predicted = {}
    for slide_path in slides_patches_predicted:
        patches_predicted = slides_patches_predicted.get(slide_path)

        cls_predicted_by_num_patch = {}
        for patch_path in patches_predicted:
            cls_prob = patches_predicted.get(patch_path)
            best_predicted_cls = get_best_predicted_cls(cls_prob)

            num_patch = cls_predicted_by_num_patch.get(best_predicted_cls)
            if num_patch is None:
                num_patch = 0

            cls_predicted_by_num_patch[best_predicted_cls] = num_patch + 1

        best_cls = get_slide_best_predicted_cls(cls_predicted_by_num_patch, config_class_level)
        slides_predicted[slide_path] = best_cls

    return slides_predicted


def get_best_predicted_cls(cls_prob):
    best_cls = 0
    max_prob = 0.0
    for cls in cls_prob:
        prob = cls_prob.get(cls)
        if max_prob <prob:
            best_cls = cls
            max_prob = prob

    return best_cls


def get_slide_best_predicted_cls(cls_prob, config_class_level):
    bg_cls = config_class_level['BG']
    gc_cls = config_class_level['GC']
    ngc_cls = config_class_level['NGC']

    if gc_cls in cls_prob:
        gc_num_patch = cls_prob[gc_cls]
    else:
        gc_num_patch = 0

    if ngc_cls in cls_prob:
        ngc_num_patch = cls_prob[ngc_cls]
    else:
        ngc_num_patch = 0

    if gc_num_patch >= ngc_num_patch/4:
        return gc_cls
    else:
        return ngc_cls


def get_slides_patches_predicted(all_slides_patches_actual_lines, all_slides_patches_predicted_lines, num_classes):

    slides_patches_predicted = {}
    for idx in range(0, len(all_slides_patches_actual_lines)):
        line = all_slides_patches_actual_lines[idx]

        #print (line)
        parts = line.split("\t")
        slide_path = parts[0]
        patch_path = parts[1]
        actual = float(parts[2])

        predicted_line = all_slides_patches_predicted_lines[idx]
        parts = predicted_line.split("\t")
        cls_prob = {}
        for jdx in range(1, num_classes + 1):
            cls_prob[jdx-1] = float(parts[jdx])

        patches_predicted = slides_patches_predicted.get(slide_path)
        if patches_predicted is None:
            patches_predicted = {}

        if patch_path not in patches_predicted:
            patches_predicted[patch_path] = cls_prob
        slides_patches_predicted[slide_path] = patches_predicted

    return slides_patches_predicted


def get_slides_patches_labels(all_slides_patches_lines):
    ''' loading the slides's path and the corresponding patch path and label '''
    slides = []
    slides_label = {}
    slides_patches_labels = {}

    for line in all_slides_patches_lines:

        parts = line.split("\t")
        #I know three parts here (Slide-path, patch-path, label)
        slide_path = parts[0]
        patch_path = parts[1]
        label = float(parts[2])

        if slide_path not in slides:
            slides.append(slide_path)

        slides_label[slide_path] = label

        patches_labels = slides_patches_labels.get(slide_path)
        if patches_labels is None:
            patches_labels = {}
        patches_labels[patch_path] = label
        slides_patches_labels[slide_path] = patches_labels

    return slides, slides_label, slides_patches_labels


