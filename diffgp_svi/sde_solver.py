import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from gpflow import settings

float_type = settings.float_type
jitter_level = settings.jitter

class EulerMaruyama:
    def __init__(self,f,total_time,nsteps):
        self.ts = np.linspace(0,total_time,nsteps)
        self.f = f

    def forward(self,y0,save_intermediate=False):
        time_grid = ops.convert_to_tensor(self.ts, preferred_dtype=float_type, name='t')
        y0 = ops.convert_to_tensor(y0, name='y0')
        time_delta_grid = time_grid[1:] - time_grid[:-1]
        time_grid = time_grid[1:]
        time_combined = tf.concat([time_grid[:,None],time_delta_grid[:,None]],axis=1)
        scan_func = self._make_scan_func(self.f)

        if save_intermediate:
            y_grid = functional_ops.scan(scan_func, time_combined,y0)
            y_s = array_ops.concat([[y0], y_grid], axis=0)
            y_t = y_s[-1,:,:,:]
            return y_t, y_s
        else:
            y_t = functional_ops.foldl(scan_func, time_combined,y0)
            return y_t, None

    def _step_func(self, evol_func, t_and_dt, y):
        t = t_and_dt[0];dt = t_and_dt[1]
        mu,var = evol_func(y, t) #NXD, NXD
        if var.get_shape().ndims == 3:
            raise NotImplementedError
        dt_cast = math_ops.cast(dt, y.dtype)
        dy = mu*dt_cast + tf.sqrt(dt_cast)*tf.sqrt(var)*tf.random_normal(shape=[tf.shape(y)[0],tf.shape(y)[1]], dtype=y.dtype) #NXD
        return dy

    def _make_scan_func(self, evol_func):
        def scan_func(y, t_and_dt):
            dy = self._step_func(evol_func, t_and_dt, y)
            dy = math_ops.cast(dy, dtype=y.dtype)
            return y + dy
        return scan_func
