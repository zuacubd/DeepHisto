import sys
import os
import tensorflow as tf


path_to_events_file = sys.argv[1]

for e in tf.train.summary_iterator(path_to_events_file):
    #print (e)
    for v in e.summary.value:
        #print (v)
        if v.tag == 'Loss' or v.tag == 'Accuracy_0' or v.tag == 'Accuracy_1':
            print("{0} : {1}".format(v.tag, v.simple_value))

