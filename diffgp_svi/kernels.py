from .param import Param

import tensorflow as tf
import numpy as np
from gpflow import transforms
from gpflow import settings

float_type = settings.float_type
jitter_level = settings.jitter

class Kernel:
    '''
    Based on GPflow Kernel class
    '''
    def __init__(self,init_lengthscales,init_variance,input_dim=None,name="kernel"):
        with tf.name_scope(name):
            lengthscales = Param(init_lengthscales,
                                      transform=transforms.Log1pe(),
                                      name="lengthscale")
            variance     = Param(init_variance,
                                      transform=transforms.Log1pe(),
                                      name="variance")
        self.lengthscales = lengthscales()
        self.input_dim = input_dim
        self.variance = variance()

    def square_dist(self,X, X2=None):
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, X, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscales
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, X2, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

class RBF(Kernel):
    '''
    Based on GPflow Kernel class
    '''
    def __init__(self,init_lengthscales,init_variance,input_dim,name="rbf"):
        super().__init__(init_lengthscales = init_lengthscales,
                         init_variance = init_variance,
                         input_dim = input_dim,
                         name = name)

    def K(self,X,X2=None):
        if X2 is None:
            return self.variance * tf.exp(-self.square_dist(X) / 2)
        else:
            return self.variance * tf.exp(-self.square_dist(X, X2) / 2)

    def Ksymm(self,X):
        return self.variance * tf.exp(-self.square_dist(X) / 2)

    def Kdiag(self,X):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))
