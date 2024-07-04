import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import time
from typing import Iterable, Union, Tuple, List

import mosek
from mosek.fusion import *
import numpy.linalg as LA
import cvxpy as cp
import gurobipy as gp
from rehline_po import MeanVariance
from plqcom import PLQLoss


import pandas as pd
import numpy as np
import numpy.linalg as LA

import matplotlib.pyplot as plt
import seaborn as sns

from pypfopt import expected_returns, risk_models
from pypfopt import EfficientFrontier, objective_functions


def rel_diff(x: np.ndarray, y: np.ndarray):
    return 2.0*LA.norm(x - y) / (LA.norm(x) + LA.norm(y))


def random_mu(N):
    return np.random.uniform(low=-0.5, high=0.5, size=N)


def random_beta(N, F):
    return np.random.uniform(low=-0.5, high=0.5, size=(N, F))


def dense_random_cov(N):
    C = np.random.uniform(low=-0.5, high=0.5, size=(N, N))
    return C.T @ C
        

def max_quad_util_portf_pyportf(N, mu, cov, risk_aversion=1.0, transaction_cost=0.0, 
                                lb=None, ub=None, tol=1e-6, debug=False):
    """Solves mean-variance problem with transaction cost with PyportfolioOpt"""
    ef = EfficientFrontier(mu, cov, weight_bounds=(lb, ub))
    ef.add_objective(objective_functions.transaction_cost, w_prev=np.zeros(N), k=transaction_cost)
    ef.max_quadratic_utility(risk_aversion=risk_aversion)
    weights_pyportf = np.array(list(ef.clean_weights().values()))
    return weights_pyportf
    

def max_quad_util_portf_gurobi_plq(mu, cov, risk_aversion, transaction_costs, lb, ub):
    """Solves mean-variance problem with transaction cost with Gurobi"""
    with gp.Model() as m:
        m.Params.OutputFlag = 0
        n_assets = len(mu)
        L = (len(transaction_costs[0].cutpoints) - 1) // 2

        # dp[n][i]: length of i-th positive-side linear piece
        # dm[n][i]: length of i-th negative-side linear piece
        dp, dm = np.zeros((n_assets, L)), np.zeros((n_assets, L))
        for i, transaction_cost in enumerate(transaction_costs):
            dp_ = np.diff(transaction_cost.cutpoints[L:-1])
            dm_ = np.diff(transaction_cost.cutpoints[1:L+1])[::-1]
            dp[i, :(L-1)], dm[i, :(L-1)] = dp_, dm_
        dp[:, L-1], dm[:, L-1] = np.ones(n_assets)*float("inf"), np.ones(n_assets)*float("inf")

        # variables = [w, wp[0][0..L-1], ..., wp[N-1][0..L-1], wm[0][0..L-1], ..., wm[N-1][0..L-1]]
        w = m.addMVar(n_assets, lb=lb, ub=ub)
        # 0 <= wp[n][i] <= DP[n][i] if i < L-1 else 0 <= wp[n][i]
        wp = m.addMVar((n_assets, L), lb=0.0, ub=dp)
        # 0 <= wm[n][i] <= DM[n][i] if i < L-1 else 0 <= wm[n][i]
        wm = m.addMVar((n_assets, L), lb=0.0, ub=dm)
        
        # constraints
        m.addConstr(w.sum() == 1)
        m.addConstr(w + wm.sum(axis=1) - wp.sum(axis=1) == 0)

        vp, vm = np.zeros((n_assets, L)), np.zeros((n_assets, L))
        for i, transaction_cost in enumerate(transaction_costs):
            vp[i, :] = np.array(transaction_cost.quad_coef['b'][L:])
            vm[i, :] = np.array(-transaction_cost.quad_coef['b'][:L][::-1])

        # objective
        m.setObjective(
            w @ mu - risk_aversion / 2.0 * w @ cov @ w - (wp * vp + wm * vm).sum(),
            gp.GRB.MAXIMIZE,
        )

        m.optimize()
        return w.X


def max_quad_util_portf_gurobi_fm_plq(mu, beta, Psi, risk_aversion, 
                                      transaction_costs: Iterable[PLQLoss], lb, ub):
    """Mean-Variance with the assumption of Factor Modeled covariance using Gurobi

    Optimization is based on 
    https://gurobi-finance.readthedocs.io/en/latest/modeling_notebooks/factor_models_objective.html

    Works extremely fast in practice.
    """
    with gp.Model() as m:
        m.Params.OutputFlag = 0
        n_assets, n_factors = beta.shape[0], beta.shape[1]
        L = (len(transaction_costs[0].cutpoints) - 1) // 2

        # dp[n][i]: length of i-th positive-side linear piece
        # dm[n][i]: length of i-th negative-side linear piece
        dp, dm = np.zeros((n_assets, L)), np.zeros((n_assets, L))
        for i, transaction_cost in enumerate(transaction_costs):
            dp_ = np.diff(transaction_cost.cutpoints[L:-1])
            dm_ = np.diff(transaction_cost.cutpoints[1:L+1])[::-1]
            dp[i, :(L-1)], dm[i, :(L-1)] = dp_, dm_
        dp[:, L-1], dm[:, L-1] = np.ones(n_assets)*float("inf"), np.ones(n_assets)*float("inf")

        # variables = [w, wp[0][0..L-1], ..., wp[N-1][0..L-1], wm[0][0..L-1], ..., wm[N-1][0..L-1]]
        w = m.addMVar(n_assets, lb=lb, ub=ub)
        # 0 <= wp[n][i] <= DP[n][i] if i < L-1 else 0 <= wp[n][i]
        wp = m.addMVar((n_assets, L), lb=0.0, ub=dp)
        # 0 <= wm[n][i] <= DM[n][i] if i < L-1 else 0 <= wm[n][i]
        wm = m.addMVar((n_assets, L), lb=0.0, ub=dm)
        y = m.addMVar(n_factors, lb=-float("inf"), ub=float("inf"))
        
        # constraints
        m.addConstr(w.sum() == 1)
        m.addConstr(beta.T @ w - y == 0)
        m.addConstr(w + wm.sum(axis=1) - wp.sum(axis=1) == 0)

        vp, vm = np.zeros((n_assets, L)), np.zeros((n_assets, L))
        for i, transaction_cost in enumerate(transaction_costs):
            vp[i, :] = np.array(transaction_cost.quad_coef['b'][L:])
            vm[i, :] = np.array(-transaction_cost.quad_coef['b'][:L][::-1])
        # objective
        m.setObjective(
            w @ mu - risk_aversion / 2.0 * y @ y - ((risk_aversion / 2.0 * Psi) * w**2).sum()
                - (wp * vp + wm * vm).sum(),
            gp.GRB.MAXIMIZE,
        )

        m.optimize()
        return w.X


# Since the actual value of Infinity is ignored, we define it solely
# for symbolic purposes:
inf = 0.0


# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def determine_volume(N, risk_aversion=1.0):
    """An attempt to estimate "cap volume" (max volume a stock can be) for the
    simulated experiment."""
    n_tries = 100
    pos = np.zeros(n_tries)
    for _ in range(n_tries):
        mu, cov = random_mu(N), dense_random_cov(N)
        pos = (LA.inv(cov) @ mu / risk_aversion).abs().mean()
    return pos.mean() * 100.0


def create_plqloss_from_volume(N, V, L, vp, vm, d_vp, d_vm) -> Iterable[PLQLoss]:
    """Creates PLQLoss object given total volume and #pieces
    
    Divides positive (and negative) parts of the function into `num_pieces`
    linear pieces with i-th one having slope vp + d_vp*(i-1) 
    [or from vm + d_vm*(i-1) for negative part]. The last piece starts from V
    """

    from typing import Iterable
    def create_func(vp, vm, d_vp, d_vm):
        xp, yp = np.zeros(L+1), np.zeros(L+1)
        for i in range(1, L+1):
            xp[i] = i/L*V
            yp[i] = yp[i-1] + (xp[i] - xp[i-1])*(vp + d_vp*(i-1))
            
        xm, ym = np.zeros(L+1), np.zeros(L+1)
        for i in range(1, L+1):
            xm[i] = -i/L*V
            ym[i] = ym[i-1] - (xm[i] - xm[i-1])*(vm + d_vm*(i-1))

        # remove (0, 0) since we have it in (xp, yp) # and reverse
        xm, ym = xm[1:], ym[1:]
        xm, ym = xm[::-1], ym[::-1]
        points = np.c_[np.r_[xm, xp], np.r_[ym, yp]]
        return PLQLoss(points=points, form="points")

    if isinstance(vp, Iterable) and isinstance(vm, Iterable) \
            and isinstance(d_vp, Iterable) and isinstance(d_vm, Iterable):
        plqlosses = []
        for _vp, _vm, _d_vp, _d_vm in zip(vp, vm, d_vp, d_vm):
            plqlosses.append(create_func(_vp, _vm, _d_vp, _d_vm))
        return plqlosses
    elif isinstance(vp, float) and isinstance(vm, float) \
            and isinstance(d_vp, float) and isinstance(d_vm, float):
        return [create_func(vp, vm, d_vp, d_vm)]*N
    else:
        raise ValueError("Expected vp, vm, d_vp, d_vm to be either all iterable or floats.")

# inf here doesn't hold numerical meaning
inf = 0.0 

def max_quad_util_portf_mosek_plq(N, mu, cov, transaction_costs: Iterable[PLQLoss],
                                risk_aversion=1.0, lb=None, ub=None, tol=1e-6, debug=False):
    """Solves mean-variance problem with transaction cost with MOSEK"""
    with mosek.Env() as env:
        # Attach a printer to the environment
        env.set_Stream(mosek.streamtype.log, streamprinter)
        # Create a task
        with env.Task() as task:
            # Set log level (integer parameter)
            if not debug:
                task.putintparam(mosek.iparam.log, 0)
            else:
                task.putintparam(mosek.iparam.log, 1)

            task.set_Stream(mosek.streamtype.log, streamprinter)
            # Set up and input bounds and linear coefficients
            L = (len(transaction_costs[0].cutpoints) - 1) // 2
            n_assets = len(mu)

            dp, dm = np.zeros((n_assets, L-1)), np.zeros((n_assets, L-1))
            for i, transaction_cost in enumerate(transaction_costs):
                dp[i, :] = np.diff(transaction_cost.cutpoints[L:-1])
                dm[i, :] = np.diff(transaction_cost.cutpoints[1:L+1])[::-1]
                
            # print(dp)
            # print(dm)
            numvar = N + 2*L*N
            # variables = [x, xp[0][0..L-1], ..., xp[N-1][0..L-1], xm[0][0..L-1], ..., xm[N-1][0..L-1]]

            # lb <= x <= ub
            if lb is None and ub is None:
                bkx = [mosek.boundkey.fr]*N
                blx = [-inf]*N
                bux = [inf]*N
            elif lb is None:
                bkx = [mosek.boundkey.up]*N
                blx = [-inf]*N
                bux = [ub]*N
            elif ub is None:
                bkx = [mosek.boundkey.lo]*N
                blx = [lb]*N
                bux = [inf]*N
            else:       
                bkx = [mosek.boundkey.ra]*N
                blx = [lb]*N
                bux = [ub]*N
            # dp[l] >= xp[i][l] >= 0 for l = 0...L-2
            #          xp[i][l] >= 0 for l = L-1
            for i in range(N):
                for l in range(L):
                    if l < (L - 1):
                        bkx.append(mosek.boundkey.ra)
                        blx.append(0)
                        bux.append(dp[i, l])
                    else:
                        bkx.append(mosek.boundkey.lo)
                        blx.append(0)
                        bux.append(inf)
            # dm[l] >= xm[i][l] >= 0 for l = 0...L-2
            #          xm[i][l] >= 0 for l = L-1
            for i in range(N):
                for l in range(L):
                    if l < (L - 1):
                        bkx.append(mosek.boundkey.ra)
                        blx.append(0)
                        bux.append(dm[i, l])
                    else:
                        bkx.append(mosek.boundkey.lo)
                        blx.append(0)
                        bux.append(inf)

            # [1] x1+...+xN = 1
            # [2...N+1] \sum_l xp[i][l] - \sum_l xm[i][l] = x_i
            bkc = [mosek.boundkey.fx]*(N+1)
            blc = [1] + [0]*(N)
            buc = [1] + [0]*(N)

            # Note below matrix structure for constraints [1..N+1]
            # x[0..N-1]
            asub, aval = [], []
            for j in range(N):
                # j-th column
                ids, vals = [0, j+1], [1.0, 1.0]
                asub.append(ids)
                aval.append(vals)
            # xp[0..N-1][0..L-1]
            for i in range(N):
                for l in range(L):
                    ids, vals = [i+1], [-1.0]
                    asub.append(ids)
                    aval.append(vals)
            # xm[0..N-1][0..L-1]
            for i in range(N):
                for l in range(L):
                    ids, vals = [i+1], [1.0]
                    asub.append(ids)
                    aval.append(vals)

            numvar = len(bkx)
            numcon = len(bkc)

            # Append 'numcon' empty constraints.
            # The constraints will initially have no bounds.
            task.appendcons(numcon)

            # Append 'numvar' variables.
            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numvar)

            # linear term
            vp, vm = np.zeros((n_assets, L)), np.zeros((n_assets, L))
            for i, transaction_cost in enumerate(transaction_costs):   
                vp[i, :] = np.array(transaction_cost.quad_coef['b'][L:])
                vm[i, :] = np.array(-transaction_cost.quad_coef['b'][:L][::-1])
            c = np.r_[-mu, vp.flatten(), vm.flatten()] / risk_aversion
            
            for j in range(numvar):
                # Set the linear term c_j in the objective.
                task.putcj(j, c[j])
                # Set the bounds on variable j
                # blx[j] <= x_j <= bux[j]
                task.putvarbound(j, bkx[j], blx[j], bux[j])
                # Input column j of A
                task.putacol(j,                  # Variable (column) index.
                             # Row index of non-zeros in column j.
                             asub[j],
                             aval[j])            # Non-zero Values of column j.
            for i in range(numcon):
                task.putconbound(i, bkc[i], blc[i], buc[i])
            
            # Set up and input quadratic objective
            qsubi = []
            qsubj = []
            qval = []
            for i in range(N):
                for j in range(i+1):
                    qsubi.append(i)
                    qsubj.append(j)
                    qval.append(cov[i, j])
            task.putqobj(qsubi, qsubj, qval)

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)

            # Optimize
            task.optimize()
            # Print a summary containing information
            # about the solution for debugging purposes
            if debug:
                task.solutionsummary(mosek.streamtype.msg)

            prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)

            # Output a solution
            res = task.getxx(mosek.soltype.itr)

            if debug:
                if solsta == mosek.solsta.optimal:
                    print("Optimal solution: %s" % res)
                elif solsta == mosek.solsta.dual_infeas_cer:
                    print("Primal or dual infeasibility.\n")
                elif solsta == mosek.solsta.prim_infeas_cer:
                    print("Primal or dual infeasibility.\n")
                elif mosek.solsta.unknown:
                    print("Unknown solution status")
                else:
                    print("Other solution status")

            return np.array(res[:N])



def max_quad_util_portf_rehline_plq(N, mu, cov, transaction_costs: Iterable[PLQLoss], 
                                    risk_aversion=1.0, lb=None, ub=None, 
                                    max_iter=1000, tol=1e-6, verbose=False, trace_freq=100):
    """Solves mean-variance problem with transaction cost with ReHLine"""
    if lb is None and ub is None:
        A = np.r_[np.c_[np.ones(N), np.ones(N)*-1.0].T]
        b = np.r_[-1.0, 1.0]
    elif lb is None:
        A = np.r_[np.c_[np.ones(N), np.ones(N)*-1.0].T, -np.eye(N)]
        b = np.r_[-1.0, 1.0, np.ones(N)*ub]
    elif ub is None:
        A = np.r_[np.c_[np.ones(N), np.ones(N)*-1.0].T, np.eye(N)]
        b = np.r_[-1.0, 1.0, -np.ones(N)*lb]
    else:
        A = np.r_[np.c_[np.ones(N), np.ones(N)*-1.0].T, np.eye(N), -np.eye(N)]
        b = np.r_[-1.0, 1.0, -np.ones(N)*lb, np.ones(N)*ub]
    
    pf = MeanVariance(mu, cov, A, b, transaction_costs=transaction_costs)
    weights_rehlinepo = pf.max_quad_util_portf(
        tol=tol, risk_aversion=risk_aversion, 
        max_iter=max_iter, verbose=verbose, 
        trace_freq=trace_freq,
    )
    return weights_rehlinepo, pf._optimizer


def eval_quad_util_plq(w, N, mu, cov, risk_aversion, transaction_costs: Iterable[PLQLoss]):
    tc = 0
    for i in range(N):
        tc += transaction_costs[i](w[i])
    return 0.5*risk_aversion*(w.T @ cov @ w) - mu.T @ w + tc