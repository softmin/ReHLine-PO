### LEGACY CODE: not used anymore
import numpy as np
import numpy.linalg as LA

from .base import CovarianceBase


class FactorModel(CovarianceBase):
    def __init__(self, L, Psi, Sigma_f, Sigma_f_inv=None):
        """Factor model to express the covariance matrix

        Parameters
        ----------

        L: ndarray of shape (N, F)
            factor loadings

        Psi: ndarray of shape (N,)
            diagonal matrix expressing noise variances

        Sigma_f: ndarray of shape (F, F)
            covariance matrix of factors

        Sigma: ndarray of shape (F, F) [optional]
            inverse of a covariance matrix of factors

        Methods are implemented to ensure all linear algebra operations (finding 
        inverse, solving system of linear equation) are done in O(N*F^2) = O(N) time
        """
        self.L = L
        self.Psi = Psi
        self.Sigma_f = Sigma_f
        self.Sigma_f_inv = Sigma_f_inv if Sigma_f_inv else LA.inv(Sigma_f)
        self.N = L.shape[0]
        self.F = L.shape[1]

    def cov_rightmult(self, x):
        x = np.array(x)
        if x.shape[0] != self.N:
            raise ValueError(f"Can input vector of size {self.N} only")
        # x @ (L @ Sigma_f @ L.T + Psi)
        factor_cov = x @ self.L @ self.Sigma_f @ self.L.T 
        noise_cov = np.diag(x * self.Psi)
        return factor_cov + noise_cov
        
    def invcov_rightmult(self, x):
        """Woodbury's inverse formula is used here:
            A more efficient multiplication of an inverse of covariance matrix
            that runs in O(N*F^2) time instead of O(N^3)

            See https://en.wikipedia.org/wiki/Woodbury_matrix_identity for the 
            formula.

            (Psi + L Sigma_f L^T)^{-1} = 
                Psi^{-1} - Psi^{-1} L (Sigma_f^{-1} + L^T Psi^{-1} L)^{-1} L^T Psi^{-1}
        """

        Psi_inv = 1.0 / self.Psi
        B_inv = LA.inv(self.Sigma_f_inv + self.L.T * Psi_inv @ self.L)
        return x * Psi_inv - (x * Psi_inv) @ self.L @ B_inv @ self.L.T * Psi_inv

    def get_cov(self):
        return self.L @ self.Sigma_f @ self.L.T + np.diag(self.Psi)

    def get_invcov(self):
        Psi_inv = 1.0 / self.Psi
        B_inv = LA.inv(self.Sigma_f_inv + (self.L.T * Psi_inv) @ self.L)
        return np.diag(Psi_inv) - (Psi_inv.reshape(-1, 1) * self.L) @ B_inv @ self.L.T * Psi_inv


