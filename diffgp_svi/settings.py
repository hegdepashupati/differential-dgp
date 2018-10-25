import numpy as np
import tensorflow as tf

class Settings:
    def __init__(self):
        self.tf_float_type   = tf.float32
        self.tf_int_type   = tf.int32
        self.np_float_type   = np.float32
        self.np_int_type   = np.int32
        self.jitter_level    = 1e-4
        self.summ = True
