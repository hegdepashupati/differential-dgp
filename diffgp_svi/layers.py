# Borrowed and modified from: https://github.com/ICL-SML/Doubly-Stochastic-DGP/

from .param import Param
from .reparameterize import reparameterize

import tensorflow as tf
import numpy as np
from gpflow.conditionals import conditional
from gpflow.kullback_leiblers import gauss_kl
from gpflow import transforms
from gpflow import settings


class Layer:
    def __init__(self):
        pass
    def conditional_ND(self, X, full_cov=False):
        raise NotImplementedError

    def KL(self):
        return tf.cast(0., dtype=settings.float_type)

    def conditional_SND(self, X, full_cov=False):
        """
        A multisample conditional, where X is shape (S,U,N,D_out), independent over samples S

        if full_cov is True
            mean is (S,U,N,D_out), var is (S,N,N,D_out)

        if full_cov is False
            mean is (S,U,N,D_out) var is (S,N,D_out)

        :param X:  The input locations (S,U,N,D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S,U,N,D_out), var (S,N,D_out or S,N,N,D_out)
        """
        S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]

        f = lambda X: self.conditional_ND(X,t=None, full_cov=full_cov)
        mean, var = tf.map_fn(f,X, dtype=(settings.float_type,settings.float_type))

        if full_cov is True:
            return tf.reshape(tf.stack(mean),[S,N,self.num_outputs]), tf.reshape(tf.stack(var),[S,N,N,self.num_outputs])
        else:
            return [tf.reshape(m, [S, N, self.num_outputs]) for m in [mean, var]]

    def sample_from_conditional(self, X, z=None, full_cov=False):
        """
        Calculates self.conditional and also draws a sample, adding input propagation if necessary

        If z=None then the tensorflow random_normal function is used to generate the
        N(0, 1) samples, otherwise z are used for the whiteed sample points

        :param X: Input locations (S,N,D_in)
        :param full_cov: Whether to compute correlations between outputs
        :param z: None, or the sampled points in whiteed representation
        :return: mean (S,N,D), var (S,N,N,D or S,N,D), samples (S,N,D)
        """
        mean, var = self.conditional_SND(X, full_cov=full_cov)

        # set shapes
        S = tf.shape(X)[0]
        N = tf.shape(X)[1]
        D = self.num_outputs

        if z is None:
            z = tf.random_normal(tf.shape(mean), dtype=settings.float_type)
        samples = reparameterize(mean, var, z, full_cov=full_cov)

        return samples, mean, var

class SVGP_Layer(Layer):
    def __init__(self,kern,Um,Us_sqrt,Z,num_outputs,white=True):
        self.white = white
        self.kern = kern
        self.num_outputs = num_outputs
        self.num_inducing = Z.shape[0]
        self.q_diag = True if Us_sqrt.ndim == 2 else False
        with tf.name_scope("inducing"):
            self.Z  = Param(Z, # MxM
                      name="z")()
            self.Um  = Param(Um, #DxM
                       name="u")()
            if self.q_diag:
                self.Us_sqrt = Param(Us_sqrt, # DxM
                                transforms.positive,
                                name="u_variance")()
            else:
                self.Us_sqrt = Param(Us_sqrt, # DxMxM
                                  transforms.LowerTriangular(Us_sqrt.shape[1],Us_sqrt.shape[0]),
                                  name="u_variance")()

        self.Ku = self.kern.Ksymm(self.Z) + tf.eye(tf.shape(self.Z)[0],dtype=self.Z.dtype)*settings.jitter
        self.Lu = tf.cholesky(self.Ku)
        self.Ku_tiled = tf.tile(self.Ku[None, :, :], [self.num_outputs, 1, 1]) # DxMxM
        self.Lu_tiled = tf.tile(self.Lu[None, :, :], [self.num_outputs, 1, 1])

    def conditional_ND(self, X,t, full_cov=False):

        Kuf = self.kern.K(self.Z, X) # MxN

        A = tf.matrix_triangular_solve(self.Lu, Kuf, lower=True) # MxM*MxN --> MxN
        if not self.white:
            A = tf.matrix_triangular_solve(tf.transpose(self.Lu), A, lower=False) #MxN

        mean = tf.matmul(A,self.Um,transpose_a=True) # NxM * MxD = NxD

        A_tiled = tf.tile(A[None, :, :], [self.num_outputs, 1, 1]) #DxMxN
        I = tf.eye(self.num_inducing, dtype=settings.float_type)[None, :, :] #1xMxM

        if self.white:
            SK = -I
        else:
            SK = -self.Ku_tiled

        Us_sqrt = self.Us_sqrt[:,:,None] if self.q_diag else self.Us_sqrt
        SK += tf.matmul(Us_sqrt, tf.transpose(Us_sqrt,[0,2,1])) # DxMxM
        B = tf.matmul(SK, A_tiled) # DxMxN

        if full_cov:
            # (num_latent, num_X, num_X)
            delta_cov = tf.matmul(tf.transpose(A_tiled,[0,2,1]), B) # DxNxN
            Kff = self.kern.K(X) # NxN
        else:
            # (num_latent, num_X)
            delta_cov = tf.reduce_sum(A_tiled * B, 1) # DxN
            Kff = self.kern.Kdiag(X) #NX,

        # either (1, num_X) + (num_latent, num_X) or (1, num_X, num_X) + (num_latent, num_X, num_X)
        var = tf.expand_dims(Kff, 0) + delta_cov
        var = tf.transpose(var)
        return mean, var # NxD, NxD or NxNxD


    def KL(self):
        M = tf.cast(self.num_inducing, settings.float_type)
        B = tf.cast(self.num_outputs, settings.float_type)

        logdet_pcov, logdet_qcov, mahalanobis, trace = self.get_kl_terms(self.Um,tf.transpose(self.Us_sqrt) if self.q_diag else self.Us_sqrt) #scalar, Dx1, Dx1, Dx1
        constant  = -M
        twoKL = logdet_pcov - logdet_qcov + mahalanobis + trace + constant
        kl = 0.5*tf.reduce_sum(twoKL)

        return kl

    def get_kl_terms(self,q_mu,q_sqrt):
        if self.white:
            alpha = q_mu  # MxD
        else:
            alpha = tf.matrix_triangular_solve(self.Lu, q_mu, lower=True)  # MxD

        if self.q_diag:
            Lq = Lq_diag = q_sqrt # MxD
            Lq_full = tf.matrix_diag(tf.transpose(q_sqrt))  # DxMxM
        else:
            Lq = Lq_full = tf.matrix_band_part(q_sqrt, -1, 0)  # force lower triangle # DxMxM
            Lq_diag = tf.transpose(tf.matrix_diag_part(Lq))  # MxD

        # Mahalanobis term: μqᵀ Σp⁻¹ μq
        mahalanobis = tf.reduce_sum(tf.square(alpha),axis=0)[:,None] # Dx1

        # Log-determinant of the covariance of q(x):
        logdet_qcov = tf.reduce_sum(tf.log(tf.square(Lq_diag)),axis=0)[:,None] # Dx1

        # Trace term: tr(Σp⁻¹ Σq)
        if self.white:
            if self.q_diag:
                trace = tf.reduce_sum(tf.square(Lq),axis=0)[:,None] # MxD --> Dx1
            else:
                trace = tf.reduce_sum(tf.square(Lq),axis=[1,2])[:,None] # DxMxM --> Dx1
        else:
            if self.q_diag:
                Lp     = self.Lu
                LpT    = tf.transpose(Lp)  # MxM
                Lp_inv = tf.matrix_triangular_solve(Lp, tf.eye(self.num_inducing, dtype=settings.float_type),lower=True)  # MxM
                K_inv  = tf.matrix_diag_part(tf.matrix_triangular_solve(LpT, Lp_inv, lower=False))[:, None]  # MxM -> Mx1
                trace  = tf.reduce_sum(K_inv * tf.square(q_sqrt),axis=0)[:,None] # Mx1*MxD --> Dx1
            else:
                Lp_full = self.Lu_tiled
                LpiLq   = tf.matrix_triangular_solve(Lp_full, Lq_full, lower=True) # DxMxM
                trace   = tf.reduce_sum(tf.square(LpiLq),axis=[1,2])[:,None] # Dx1

        # Log-determinant of the covariance of p(x):
        if not self.white:
            log_sqdiag_Lp = tf.log(tf.square(tf.matrix_diag_part(self.Lu)))
            logdet_pcov = tf.reduce_sum(log_sqdiag_Lp)
        else:
            logdet_pcov = 0

        return logdet_pcov, logdet_qcov, mahalanobis, trace
