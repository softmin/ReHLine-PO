import numpy as np
from numpy import linalg as LA

from rehline_po.optimization._base import Portfolio
from typing import Dict, List, Tuple, Callable
from rehline import ReHLineLinear, ReHLoss
from plqcom import PLQLoss, plq_to_rehloss, affine_transformation
        

class MeanVariance(Portfolio):
    def __init__(self,
                 mu: np.ndarray,
                 cov: np.ndarray,
                 A: np.ndarray,
                 b: np.ndarray,
                 cov_sqrt: np.ndarray = None,
                 w_prev: np.ndarray = None,
                 buy_cost: float = None,
                 sell_cost: float = None,
                 transaction_cost: Dict = None):
        self.mu = mu
        self.cov = cov
        self.n_assets = len(self.mu)
        # self.linear_constraints = linear_constraints
        self.A = A
        self.b = b
        self.cov_sqrt = cov_sqrt
        self.cov_sqrt_inv = None
        self.w_prev = w_prev if w_prev is not None else np.zeros(self.n_assets)
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.transaction_cost = transaction_cost
        if self.cov_sqrt:
            assert np.isclose(2.0*LA.norm(self.cov_sqrt @ self.cov_sqrt.T - self.cov)
                                / (LA.norm(self.cov_sqrt @ self.cov_sqrt.T) + LA.norm(self.cov)), 1e-4), \
                    "Invalid sqrt of a covariance matrix supplied!"
        self.U, self.V = self._construct_relu_coefs()
        


    def max_quad_util_portf(self, risk_aversion: float = 1.0, max_iter=1000, 
                            tol: float = 1e-4, verbose=False, trace_freq=100) -> np.ndarray:
        # Solution provided by ReHLine
        if not self.cov_sqrt:
            self.cov_sqrt = LA.cholesky(self.cov)
        if not self.cov_sqrt_inv:
            self.cov_sqrt_inv = LA.inv(self.cov_sqrt)

        X = self.cov_sqrt_inv.T / np.sqrt(risk_aversion)
        A_tilde = self.A @ X
        mu_tilde = X.T @ self.mu

        self._optimizer = ReHLineLinear(
            C=risk_aversion, 
            A=A_tilde, 
            b=self.b,
            U=self.U,
            V=self.V,
            mu=mu_tilde,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            trace_freq=trace_freq,
        )
        self._optimizer.fit(X=X)
        return X @ self._optimizer.coef_


    def max_return_portf(self, max_risk: float) -> np.ndarray:
        pass


    def min_risk_portf(self, min_return: float) -> np.ndarray:
        pass


    def efficient_frontier(self) -> Tuple[np.ndarray, np.ndarray]:
        return None, None


    def _construct_relu_coefs(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.buy_cost is None and self.transaction_cost is None and self.sell_cost is None:
            return np.zeros(shape=(0,0)), np.zeros(shape=(0,0))
        if self.buy_cost is not None and self.sell_cost is not None:
            sell_cost = self.sell_cost * np.ones(self.n_assets)
            buy_cost = self.buy_cost * np.ones(self.n_assets)
            plqs = [PLQLoss({'a': np.array([0., 0.]), 
                     'b': np.array([-1.0*sell_cost[i], 1.0*buy_cost[i]]),
                     'c': np.array([0., 0.])
                     }, 
                    cutpoints=np.array([0.])) for i in range(self.n_assets)]
        elif self.transaction_cost is not None:
            plqs = self.transaction_costs
        else:
            raise TypeError("Invalid argument for transaction costs. \
                            Please, check documentation.")

        rehlosses = []
        for i, plq in enumerate(plqs):
            rehloss: ReHLoss = plq_to_rehloss(plq)
            assert rehloss.H == 0, "Transaction cost function is supposed to \
                                    be constructed of linear pieces only"
            # w[i] -> w[i] - w_prev[i]
            for j in range(rehloss.L):
                rehloss.relu_intercept[j] = rehloss.relu_intercept[j] - rehloss.relu_coef[j] * self.w_prev[i]
            rehlosses.append(rehloss)

        # for i, plq in enumerate(self.holding_costs):
        #     rehloss: ReHLoss = plq_to_rehloss(plq)
        #     assert rehloss.H == 0, "Holding cost function is \
        #           supposed to be constructed of linear pieces only"
        #     rehlosses.append(rehloss)

        L = max([rehloss.L for rehloss in rehlosses])
        U, V = np.zeros((L, self.n_assets)), np.zeros((L, self.n_assets))
        for i, rehloss in enumerate(rehlosses):
            U[:rehloss.L, i] = rehloss.relu_coef.flatten()
            V[:rehloss.L, i] = rehloss.relu_intercept.flatten()
        return U, V
