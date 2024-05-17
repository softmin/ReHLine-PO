import numpy as np
from numpy import linalg as LA

from rehline_po.optimization._base import Portfolio
from typing import Dict, List, Tuple, Callable
from rehline import ReHLineLinear, ReHLoss
from plqcom import PLQLoss, plq_to_rehloss
        

class MeanVariance(Portfolio):
    def __init__(self,
                 mu: np.ndarray,
                 cov: np.ndarray,
                 A: np.ndarray,
                 b: np.ndarray,
                 w_prev: np.ndarray = None,
                 transaction_costs = None):
        self.mu = mu
        self.cov = cov
        # self.linear_constraints = linear_constraints
        self.A = A
        self.b = b
        self.w_prev = w_prev
        self.transaction_costs = transaction_costs
        self.n_assets = len(self.mu)


    def max_quad_util_portf(self, risk_aversion: float = 1.0, tol: float = 1e-5) -> np.ndarray:
        # Solution provided by ReHLine
        L = LA.cholesky(self.cov)
        X = LA.inv(L.T)
        A_tilde = self.A @ X
        mu_tilde = X.T @ self.mu
        U, V = self._construct_relu_coefs()

        markowitz = ReHLineLinear(
            C=risk_aversion, 
            A=A_tilde, 
            b=self.b,
            U=U,
            V=V,
            mu=mu_tilde,
            tol=tol,
        )
        markowitz.fit(X=X)
        return X @ markowitz.coef_


    def max_return_portf(self, max_risk: float) -> np.ndarray:
        pass


    def min_risk_portf(self, min_return: float) -> np.ndarray:
        pass


    def efficient_frontier(self) -> Tuple[np.ndarray, np.ndarray]:
        return None, None


    def _construct_relu_coefs(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.transaction_costs is None:
            return np.zeros(shape=(0,0)), np.zeros(shape=(0,0))
        if isinstance(self.transaction_costs, float):
            plqs = [
                PLQLoss(
                    {'a': np.array([0., 0.]), 'b': np.array([-1.0*self.transaction_costs, 1.0*self.transaction_costs]), 'c': np.array([0., 0.])}, 
                    cutpoints=np.array([0.])
                ) for _ in range(self.n_assets)]
        elif isinstance(self.transaction_costs, List):
            plqs = self.transaction_costs
        else:
            raise TypeError("transaction_costs is supposed to be either float or a list of PLQ objects")

        rehlosses = []
        for i, plq in enumerate(plqs):
            rehloss: ReHLoss = plq_to_rehloss(plq)
            assert rehloss.H == 0, "Transaction cost function is supposed to be constructed of linear pieces only"
            # w[i] -> w[i] - w_prev[i]
            for j in range(rehloss.L):
                rehloss.relu_intercept[j] = rehloss.relu_intercept[j] - rehloss.relu_coef[j] * self.w_prev[i]
            rehlosses.append(rehloss)

        # for i, plq in enumerate(self.holding_costs):
        #     rehloss: ReHLoss = plq_to_rehloss(plq)
        #     assert rehloss.H == 0, "Holding cost function is supposed to be constructed of linear pieces only"
        #     rehlosses.append(rehloss)

        L = max([rehloss.L for rehloss in rehlosses])
        U, V = np.zeros((L, self.n_assets)), np.zeros((L, self.n_assets))
        for i, rehloss in enumerate(rehlosses):
            U[:rehloss.L, i], V[:rehloss.L, i] = rehloss.relu_coef.flatten(), rehloss.relu_intercept.flatten()
        return U, V
