import numpy as np
import numpy.linalg as LA

class CovarianceBase:
    """Base class for covariance matrices"""
    pass


class Covariance(CovarianceBase):
    def __init__(self, cov):
        self.cov = cov
        self.inv_cov = None

    def cov_rightmult(self, x):
        x = np.array(x)
        return x @ self.cov
        
    def invcov_rightmult(self, x):
        if self.inv_cov is None:
            self.inv_cov = LA.inv(self.cov)
        return x @ self.inv_cov

    def get_cov(self):
        return self.cov

    def get_invcov(self):
        if self.inv_cov is None:
            self.inv_cov = LA.inv(self.cov)
        return self.inv_cov

