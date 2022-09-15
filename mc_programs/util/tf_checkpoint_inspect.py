import os
import sys

import tensorflow
from tensorflow.python.tools import inspect_checkpoint as chkp

path = sys.argv[1]
print (path)
chkp.print_tensors_in_checkpoint_file(path, tensor_name='', all_tensors=True)
