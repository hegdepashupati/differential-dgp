import os
import numpy as np
import tensorflow as tf
import gpflow as gp
from gpflow import transforms
from tensorflow import Variable
from gpflow import settings
from .settings import Settings


class Param:
    '''
    Based on Param class in GPflow
    '''
    def __init__(self,value,transform = transforms.Identity(),name='param',fixed=False):
        self.value = value
        self.fixed = fixed
        self.transform = transform
        self.name = name

        if self.fixed:
            self.tf_opt_var = tf.constant(self.value,name=name,dtype=settings.float_type)
        else:
            self.tf_opt_var = Variable(self.transform.backward(self.value),name=name,dtype=settings.float_type)

        if Settings().summ:
            self.variable_summaries(self.transform.forward_tensor(self.tf_opt_var))

    def __call__(self):
        return self.transform.forward_tensor(self.tf_opt_var)

    def variable_summaries(self,var):
      tf.summary.histogram(self.name, var)

    @property
    def shape(self):
        return self.value.shape
