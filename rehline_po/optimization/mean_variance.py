import numpy as np
from numpy import linalg as LA

from rehline_po.optimization.base import Portfolio
from rehline_po.risk_models.factor_model import FactorModel
from typing import Dict, List, Tuple, Callable, Union
from rehline import ReHLineLinear, ReHLoss, ReHLineQuad
from plqcom import PLQLoss, plq_to_rehloss, affine_transformation
        

class MeanVariance(Portfolio):
    r"""Mean Variance Efficient Frontier
    
    Class that solves mean-variance portfolio problem of the type:
        min_x quadratic_utility(x) + sum_i transaction_cost(x_i)
    """
    def __init__(self,
                 mu: np.ndarray,
                 cov: Union[np.ndarray, FactorModel],
                 A: np.ndarray,
                 b: np.ndarray,
                 cov_sqrt: np.ndarray = None,
                 w_prev: np.ndarray = None,
                 buy_cost: float = None,
                 sell_cost: float = None,
                 transaction_costs: List[PLQLoss] = None):
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
        self.transaction_costs = transaction_costs
        # if self.cov_sqrt:
        #     assert np.isclose(2.0*LA.norm(self.cov_sqrt @ self.cov_sqrt.T - self.cov)
        #                         / (LA.norm(self.cov_sqrt @ self.cov_sqrt.T) + LA.norm(self.cov)), 1e-4), \
        #             "Invalid sqrt of a covariance matrix supplied!"
        self.U, self.V = self._construct_relu_coefs()
        


    def max_quad_util_portf(self, risk_aversion: float = 1.0, max_iter=1000, 
                            tol: float = 1e-4, verbose=False, trace_freq=100) -> np.ndarray:
        r"""Solve maximum quadratic utility portfolio using ReHLine algorithm

        .. math::

            \min_{\mathbf{w} \in \mathbb{R}^n} \frac{C}{2} \mathbf{w}^T G \mathbf{w} - \mathbf{\mu}^T \mathbf{w} + \sum_{i=1}^n \phi_i(w_i),

        where :math:`\phi_i` is a piecewise linear convex function (typically modelling 
        transaction cost + market impact)
        """
        if not isinstance(self.cov, np.ndarray):
            U_scaled, V_scaled = self.U / risk_aversion, self.V / risk_aversion
            mu_scaled = self.mu / risk_aversion
            if verbose > 0:
                self._optimizer = ReHLineQuad(
                    loss='custom',
                    C=risk_aversion,
                    A=self.A,
                    b=self.b,
                    U=U_scaled,
                    V=V_scaled,
                    G=self.cov.get_cov(),
                    invG=self.cov.get_invcov(),
                    rightmult_invG=self.cov.invcov_rightmult,
                    rightmult_G=self.cov.cov_rightmult,
                    mu=mu_scaled,
                    max_iter=max_iter,
                    tol=tol,
                    verbose=verbose,
                    trace_freq=trace_freq,
                )
            else:
                self._optimizer = ReHLineQuad(
                    loss='custom',
                    C=risk_aversion,
                    A=self.A,
                    b=self.b,
                    U=U_scaled,
                    V=V_scaled,
                    rightmult_invG=self.cov.invcov_rightmult,
                    rightmult_G=self.cov.cov_rightmult,
                    mu=mu_scaled,
                    max_iter=max_iter,
                    tol=tol,
                    verbose=verbose,
                    trace_freq=trace_freq,
                )
            self._optimizer.fit(X=np.eye(self.n_assets))
            return self._optimizer.coef_
        else:
            if not self.cov_sqrt:
                self.cov_sqrt = LA.cholesky(self.cov)
            if not self.cov_sqrt_inv:
                self.cov_sqrt_inv = LA.inv(self.cov_sqrt)

            X = self.cov_sqrt_inv.T / np.sqrt(risk_aversion)
            A_tilde = self.A @ X
            mu_tilde = X.T @ self.mu

            # print("C:", risk_aversion)
            # print("A:", A_tilde)
            # print("b:", self.b)
            # print("U:", self.U)
            # print("V:", self.V)
            # print("mu:", mu_tilde)
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


    def _construct_relu_coefs(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.buy_cost is None and self.transaction_costs is None and self.sell_cost is None:
            return np.zeros(shape=(0,0)), np.zeros(shape=(0,0))
        if self.buy_cost is not None and self.sell_cost is not None:
            sell_cost = self.sell_cost * np.ones(self.n_assets)
            buy_cost = self.buy_cost * np.ones(self.n_assets)
            plqs = [PLQLoss({'a': np.array([0., 0.]), 
                     'b': np.array([-1.0*sell_cost[i], 1.0*buy_cost[i]]),
                     'c': np.array([0., 0.])
                     }, 
                    cutpoints=np.array([0.])) for i in range(self.n_assets)]
        elif self.transaction_costs is not None:
            plqs = self.transaction_costs
        else:
            raise TypeError("Invalid argument for transaction costs. \
                            Please, check documentation.")

        rehlosses = []
        # print(plqs)
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
