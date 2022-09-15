'''
    This program is used to fuse the predictions of several models based on either sum of probability or voting
'''

import os
import time
import numpy as np

from sklearn import metrics
from sklearn.metrics import classification_report
from config.config import config_class_level
from config.config import train_parameters
from config.config import output_parameters

def ensemble_model(criteria):
    num_classes = train_parameters['num_classes']
    #network_names = ["alexnet", "vgg16", "vgg19", "googlenet", "resnet50", "resnet101"]
    network_names = ["alexnet", "vgg16", "googlenet", "resnet50"]
    #network_names = ["alexnet", "googlenet", "resnet50", "resnet101"]
    aggregators = ["voting", "sum", "mean", "median", "std", "var", "max"]

    prediction_folder = output_parameters['prediction_folder']
    result_folder = output_parameters['result_folder']
    all_network_names = "_".join(network_names)

    if criteria == "Probability":

        for agg_func in aggregators:

            fused_prediction_path = prediction_folder + '/' + all_network_names + '_' + agg_func + '.pred'
            fused_prediction_result_path = result_folder + '/' + all_network_names + '_' + agg_func + '.res'

            ensembled_results, actual = ensemble_probabilities(prediction_folder, network_names, agg_func, num_classes)
            N, C = ensembled_results.shape

            with open(fused_prediction_path, 'w') as writer:
                header = 'Serial'
                for j in range(num_classes):
                    header = header + '\t' + 'Class-' + str(j)
                header = header + '\t' + 'Actual'
                writer.write(header + '\n')

                for idx in range(N):
                    ensemble_result = ensembled_results[idx, ]
                    line = str(idx)
                    for l in range(len(ensemble_result)):
                        line = line + "\t" + str(ensemble_result[l])
                    line = line + "\t" + str(actual[idx,0])
                    writer.write(line + '\n')


            # Evaluation op: Accuracy of the model
            #with tf.name_scope("accuracy"):
            #    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
            #    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            print (actual.shape)
            print (ensembled_results.shape)
            #np.savetxt('actual.out', actual, delimiter=',')
            #np.savetxt('pred.out', ensembled_results, delimiter=',')

            am_actual = np.argmax(actual, axis=1)
            es_result = np.argmax(ensembled_results, axis=1)
            print (am_actual.shape)
            print (am_actual)
            print (es_result.shape)
            print (es_result)
            #np.savetxt('actual_idx.out', am_actual, delimiter=',')
            #np.savetxt('pred_idx.out', es_result, delimiter=',')

            accuracy = np.equal(am_actual, es_result).mean()
            print(accuracy)
            # Add the accuracy to the summary
            #tf.summary.scalar('Accuracy', accuracy)

            #with tf.name_scope("confusion"):
            #    confusion = tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(score, 1), num_classes)
            confusion = metrics.confusion_matrix(am_actual, es_result)
            print(confusion)
            report = metrics.classification_report(am_actual, es_result)
            print (report)
            # Add the accuracy to the summary
            #tf.summary.scalar('Confusion', confusion)
            with open(fused_prediction_result_path, 'w') as fw:
                fw.write(report)



def ensemble_probabilities(predict_path, network_names, agg_func, num_classes):
    ''' combine the predicted probabilities of several networks for an image '''

    networks_prediction = []
    networks_actual = []

    for idx in range(len(network_names)):
        network_name = network_names[idx]
        network_predict_path = predict_path + '/' + network_name + '.pred'

        network_result = get_prediction_result(network_predict_path)
        predictions, ys = get_prediction_actual(network_result, num_classes)
        networks_prediction.append(predictions)
        networks_actual.append(ys)

    N, C = networks_prediction[0].shape
    ensemble_results = np.zeros((N, C), dtype=np.float32)
    actual = networks_actual[0]

    for c in range(C):
        cls_predictions = np.zeros((N, len(network_names)), dtype=np.float32)

        for ndx in range(len(network_names)):
            pred = networks_prediction[ndx][:, c]
            cls_predictions[:, ndx] = pred

        if agg_func == "mean":
            agg_prob = np.mean(cls_predictions, axis=1)

        elif agg_func == "median":
            agg_prob = np.median(cls_predictions, axis=1)

        elif agg_func == "std":
            agg_prob = np.std(cls_predictions, axis=1)

        elif agg_func == "var":
            agg_prob = np.var(cls_predictions, axis=1)

        elif agg_func == "max":
            agg_prob = np.max(cls_predictions, axis=1)

        elif agg_func == "sum":
            agg_prob = np.sum(cls_predictions, axis=1)

        elif agg_func == "voting":
            agg_prob = (cls_predictions >= 0.5).sum(axis=1)

        else:
            print ("wrong aggregator")

        ensemble_results[:, c] = agg_prob

    return ensemble_results, actual


def get_sum_of_probability(networks_prediction):
    ''' compute the sum of log probabilities for each class'''

    C = len(networks_prediction[0]) - 2
    M = len(networks_prediction)
    eps = 1e-10
    serial = networks_prediction[0][0]
    actual = networks_prediction[0][len(networks_prediction[0])-1]

    fused_result = [serial]
    for i in range(C):

        total_prob = 0
        class_index = 1 + i
        for j in range(M):
            pred_prob = float(networks_prediction[j][class_index])
            #score = math.log(pred_prob + eps)
            total_prob += pred_prob

        fused_result.append(str(total_prob))
    fused_result.append(actual)
    return fused_result


def get_prediction_result(prediction_path):
    '''load prediction results'''

    with open(prediction_path, 'r') as reader:
        lines = reader.readlines()

    #skip header
    data_lines = lines[1:]
    result_lines = [line.rstrip().split("\t") for line in data_lines]
    return result_lines


def get_prediction_actual(result, num_classes):
    ''' Converting the prediction results into numpy matrix'''

    num_sample = len(result)
    predictions = np.zeros((num_sample, num_classes), dtype=np.float32)
    actual = np.zeros((num_sample, num_classes), dtype=np.float32)

    for i in range(num_sample):
        line = result[i]

        for j in range(1, len(line) - num_classes):
            pred = float(line[j])
            predictions[i, j-1] = pred

        k = 0
        for j in range(1+num_classes, len(line)):
            y = float(line[j])
            actual[i, k] = y
            k = k + 1

    return predictions, actual
