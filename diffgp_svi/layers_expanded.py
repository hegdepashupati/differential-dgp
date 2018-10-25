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
    def __init__(self,kerns,Um,Us_sqrt,Z,num_outputs,white=True):
        self.white = white
        self.kerns = kerns
        self.num_outputs = num_outputs
        self.num_inducing = Z.shape[0]
        self.q_diag = True if Us_sqrt.ndim == 2 else False
        with tf.name_scope("inducing"):
            self.Z  = Param(Z, # MxM
                      name="z")()
            self.Um  = Param(Um, #MxD
                       name="u")()
            if self.q_diag:
                self.Us_sqrt = Param(Us_sqrt, # DxM
                                transforms.positive,
                                name="u_variance")()
            else:
                self.Us_sqrt = Param(Us_sqrt, # DxMxM
                                  transforms.LowerTriangular(Us_sqrt.shape[1],Us_sqrt.shape[0]),
                                  name="u_variance")()

        self.Ku_tiled = tf.stack([kern.Ksymm(self.Z) + tf.eye(tf.shape(self.Z)[0],dtype=self.Z.dtype)*settings.jitter for kern in self.kerns],axis=0) # DxMxM
        self.Lu_tiled = tf.cholesky(self.Ku_tiled) # DxMxM

    def conditional_ND(self, X,t, full_cov=False):

        Kuf = tf.stack([kern.K(self.Z, X) for kern in self.kerns],axis=0) # DxMxN

        A = tf.matrix_triangular_solve(self.Lu_tiled, Kuf, lower=True) # DxMxM*DxMxN --> DxMxN
        if not self.white:
            A = tf.matrix_triangular_solve(tf.transpose(self.Lu_tiled,[0,2,1]), A, lower=False) #DxMxN


        mean = tf.transpose(tf.reduce_sum(tf.multiply(A,tf.transpose(self.Um)[:,:,None]),axis=1)) # DxMxN * DxMxN --> DxN --> NxD
        I = tf.eye(self.num_inducing, dtype=settings.float_type)[None, :, :] #1xMxM

        if self.white:
            SK = -I
        else:
            SK = -self.Ku_tiled

        Us_sqrt = self.Us_sqrt[:,:,None] if self.q_diag else self.Us_sqrt
        SK += tf.matmul(Us_sqrt, tf.transpose(Us_sqrt,[0,2,1])) # DxMxM
        B = tf.matmul(SK, A) # DxMxN

        if full_cov:
            # (num_latent, num_X, num_X)
            delta_cov = tf.matmul(tf.transpose(A,[0,2,1]), B) # DxNxN
            Kff = tf.stack([kern.K(X) for kern in self.kerns],axis=0) # DxNxN
        else:
            # (num_latent, num_X)
            delta_cov = tf.reduce_sum(A * B, 1) # DxN
            Kff = tf.stack([kern.Kdiag(X) for kern in self.kerns],axis=0) #DxN

        var = Kff + delta_cov
        var = tf.transpose(var)
        return mean, var # UxNxD, UxNxD or UxNxNxD


    def KL(self):
        M = tf.cast(self.num_inducing, settings.float_type)
        B = tf.cast(self.num_outputs, settings.float_type)

        logdet_pcov, logdet_qcov, mahalanobis, trace = self.get_kl_terms(self.Um,tf.transpose(self.Us_sqrt) if self.q_diag else self.Us_sqrt) #Dx1, Dx1, Dx1, Dx1
        constant  = -M
        twoKL = logdet_pcov - logdet_qcov + mahalanobis + trace + constant
        kl = 0.5*tf.reduce_sum(twoKL)

        return kl

    def get_kl_terms(self,q_mu,q_sqrt):
        if self.white:
            alpha = tf.transpose(q_mu)[:,:,None]  # DxMx1
        else:
            alpha = tf.matrix_triangular_solve(self.Lu_tiled,tf.transpose(q_mu)[:,:,None], lower=True)  # DxMxM * DxMx1 --> DxMx1

        if self.q_diag:
            Lq = Lq_diag = q_sqrt # MxD
            Lq_full = tf.matrix_diag(tf.transpose(q_sqrt))  # DxMxM
        else:
            Lq = Lq_full = tf.matrix_band_part(q_sqrt, -1, 0)  # force lower triangle # DxMxM
            Lq_diag = tf.transpose(tf.matrix_diag_part(Lq))  # MxD

        # Mahalanobis term: μqᵀ Σp⁻¹ μq
        mahalanobis = tf.reduce_sum(tf.square(alpha),axis=[1,2])[:,None] # Dx1

        # Log-determinant of the covariance of q(x):
        logdet_qcov = tf.reduce_sum(tf.log(tf.square(Lq_diag)),axis=0)[:,None] # Dx1

        # Trace term: tr(Σp⁻¹ Σq)
        if self.white:
            if self.q_diag:
                trace = tf.reduce_sum(tf.square(Lq),axis=0)[:,None] # MxD --> Dx1
            else:
                trace = tf.reduce_sum(tf.square(Lq),axis=[1,2])[:,None] # DxMxM --> Dx1
        else:
            LpiLq   = tf.matrix_triangular_solve(self.Lu_tiled, Lq_full, lower=True) # DxMxM
            trace   = tf.reduce_sum(tf.square(LpiLq),axis=[1,2])[:,None] # Dx1

        # Log-determinant of the covariance of p(x):
        if not self.white:
            log_sqdiag_Lp = tf.stack([tf.log(tf.square(tf.matrix_diag_part(self.Lu_tiled[d]))) for d in range(self.num_outputs)],axis=0) #DxM
            logdet_pcov = tf.reduce_sum(log_sqdiag_Lp,axis=1)[:,None] #Dx1
        else:
            logdet_pcov = 0

        return logdet_pcov, logdet_qcov, mahalanobis, trace
