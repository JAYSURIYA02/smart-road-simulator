import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically allocate memory
sess = tf.Session(config=config)

import tensorflow as tf
print(tf.__version__)
print(tf.test.is_gpu_available())