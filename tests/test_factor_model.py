import numpy as np
import numpy.linalg as LA
from .context import FactorModel, MeanVariance

n_assets, n_factors = 10, 5
lb, ub = -1.0, 1.0
buy_cost, sell_cost = 0.01*np.ones(n_assets), 0.02*np.ones(n_assets)
risk_aversion = 2.0

max_iter = 1000
tol = 1e-4
verbose = 0
trace_freq = 10

L = np.array([
    [0.12458073, 0.11162239, 0.36462851, 0.95092074, 0.36054079],
    [0.25599588, 0.96516357, 0.43746946, 0.15322412, 0.57586066],
    [0.4147078, 0.45902409, 0.43758612, 0.33774256, 0.64955405],
    [0.12068682, 0.52805477, 0.63648499, 0.66386672, 0.59191421],
    [0.25648868, 0.98494758, 0.7140777,  0.19115454, 0.07275046],
    [0.90974659, 0.53366267, 0.59984997, 0.63719311, 0.83501116],
    [0.91224402, 0.47440557, 0.25703101, 0.74850448, 0.62705515],
    [0.14938886, 0.99058464, 0.23918362, 0.02195245, 0.73148935],
    [0.48560735, 0.61765321, 0.6534312,  0.8287803, 0.63791695],
    [0.5934457, 0.42036697, 0.26251893, 0.90422348, 0.29167714]
])
lam = 0.2
Psi = lam * np.ones(n_assets)
mu = np.array([0.31307052, -0.07482042, -0.3441124, 0.3052823, 0.05707814, 
               -0.06578429, -0.40086533, -0.39854555,  0.23378517, -0.07748241])
cov_fm = FactorModel(L, Psi, np.eye(n_factors))
cov = cov_fm.get_cov()

A = np.r_[
    np.c_[np.ones(n_assets), np.ones(n_assets)*-1.0].T, 
    np.eye(n_assets),
    -np.eye(n_assets)]
b = np.r_[-1.0, 1.0, -np.ones(n_assets)*lb, np.ones(n_assets)*ub]

pf1 = MeanVariance(mu=mu, cov=cov_fm, A=A, b=b, buy_cost=buy_cost, sell_cost=sell_cost)
weights_fm = pf1.max_quad_util_portf(tol=tol, risk_aversion=risk_aversion, 
                    max_iter=max_iter, verbose=verbose, trace_freq=trace_freq)

pf1 = MeanVariance(mu=mu, cov=cov, A=A, b=b, buy_cost=buy_cost, sell_cost=sell_cost)
weights_lt = pf1.max_quad_util_portf(tol=tol, risk_aversion=risk_aversion, 
                    max_iter=max_iter, verbose=verbose, trace_freq=trace_freq)

print("l2_dist(reh_factor, reh_vanilla)", 
      LA.norm(weights_fm - weights_lt))

    