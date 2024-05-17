import os

import pandas as pd
import numpy as np
from numpy import linalg as LA

from pypfopt import expected_returns, risk_models
from pypfopt import EfficientFrontier, objective_functions

from .context import MeanVariance


def resource(name):
    return os.path.join(os.path.dirname(__file__), "resources", name)


def get_data():
    return pd.read_csv(resource("stock_prices.csv"), parse_dates=True, index_col="date")

"""
min_{w in R^n} -mu' w + alpha/2 * w' S w + sum_i phi_i(w_i) s.t. w'1 = 1 & w >= 0
where phi_i(w) = transaction_cost * |w - w^pre_i|
"""

prices = get_data()
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)
risk_aversion = 1.0
transaction_cost = 0.01
# assume we have started with equally weighted portfolio
tickers = prices.columns
n = len(tickers)
initial_weights = np.array([1/n] * n)

# Solution provided by PyPortfolio
ef = EfficientFrontier(mu, S)
ef.add_objective(objective_functions.transaction_cost, w_prev=initial_weights, k=transaction_cost)
ef.max_quadratic_utility(risk_aversion=risk_aversion)
weights_pyportf = np.array(list(ef.clean_weights().values()))
print("Solution provided by PyPortfolio: ", weights_pyportf)

# Solution provided by ReHLine-PO
A = np.r_[np.c_[np.ones(n), np.ones(n)*-1.0].T, np.eye(n)]
b = np.r_[-1.0, 1.0, np.zeros(n)]
tol = 1e-6

pf = MeanVariance(mu, S, A, b, w_prev=initial_weights, transaction_costs=transaction_cost)
weights_rehlinepo = pf.max_quad_util_portf(tol=tol, risk_aversion=risk_aversion)
print("Solution provided by ReHLine: ", weights_rehlinepo)
print("L2 distance between solutions", LA.norm(weights_rehlinepo - weights_pyportf))

assert np.isclose(LA.norm(weights_pyportf - weights_rehlinepo), 0.0, atol=1e-5)