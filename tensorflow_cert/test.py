import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import required libraries
import tensorflow as tf

s = tf.constant("Hello, World!")

print(s.numpy())
