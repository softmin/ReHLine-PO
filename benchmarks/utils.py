import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import time

import mosek
import numpy.linalg as LA
import cvxpy as cp
from rehline_po import MeanVariance

import pandas as pd
import numpy as np
import numpy.linalg as LA

import matplotlib.pyplot as plt
import seaborn as sns

from pypfopt import expected_returns, risk_models
from pypfopt import EfficientFrontier, objective_functions


def random_sparse_spd_matrix(
    dim: int,
    density: float,
    chol_entry_min: float = 0.1,
    chol_entry_max: float = 1.0,
) -> np.ndarray:
    """Generate random sparse semi positive-definite matrix"""
    if not 0 <= density <= 1:
        raise ValueError(f"Density must be between 0 and 1, but is {density}.")
    G = np.eye(dim)
    num_elements = int(dim * dim)
    num_nonzero_entries = int(num_elements * density)

    if num_nonzero_entries > 0:
        # Draw entries of lower triangle (below diagonal) according to sparsity level
        idx_samples = np.random.choice(
            a=dim*dim, size=num_nonzero_entries, replace=False
        )
        nonzero_entry_ids = (idx_samples % dim, idx_samples // dim)

        # Fill Cholesky factor
        G[nonzero_entry_ids] = np.random.uniform(
            low=chol_entry_min, high=chol_entry_max, size=num_nonzero_entries
        )

    return G @ G.T


def eval_quad_util(w, N, m, cov, risk_aversion, vp, vm):
    return 0.5*risk_aversion*(w.T @ cov @ w) - m.T @ w + np.where(w > 0, w*vp, -w*vm).sum()


def rel_diff(x: np.ndarray, y: np.ndarray):
    return 2.0*LA.norm(x - y) / (LA.norm(x) + LA.norm(y))


def max_meanstdrisk_util_portf_mosek(N, m, G, risk_aversion=1.0, vp=0.0, vm=0.0, 
                                     lb=0.0, ub=None, tol=1e-6):
    """Optimizes the problem:
        min_x 1/2*\sqrt{x'Sx} - m'x + \sum_i \phi_i(x_i)
            s.t. lb <= x <= ub & x'1 = 1
        where \phi_i(y) = vp_i*y if y >= 0 else vm_i*y
    """
    with Model("Case study") as M:   
        # Model settings
        M.setSolverParam("intpntCoTolRelGap", tol)
        # Real variables
        if lb is None and ub is None: 
            x = M.variable("x", N, Domain.unbounded())
        elif lb is None:
            x = M.variable("x", N, Domain.lowerThan(ub))
        elif ub is None:
            x = M.variable("x", N, Domain.greaterThan(lb))
        else:
            x = M.variable("x", N, Domain.inRange(lb, ub))
        xp = M.variable("xp", N, Domain.greaterThan(0.0))
        xm = M.variable("xm", N, Domain.greaterThan(0.0))
        s = M.variable("s", 1, Domain.unbounded())
        
        # Constraint assigning xp and xm to the pos. and neg. part of x.
        M.constraint('pos-neg-part', Expr.sub(x, Expr.sub(xp, xm)),
                   Domain.equalsTo(0.0))
        
        # Conic constraint for the portfolio variance
        M.constraint('risk', Expr.vstack(s, Expr.mul(G.T, x)), Domain.inQCone())
        
        # Budget constraint
        M.constraint('budget', Expr.sum(x), Domain.equalsTo(1.0))
        
        # Objective (quadratic utility version)
        delta = M.parameter()
        delta.setValue(risk_aversion)
        varcost_terms = Expr.add([Expr.dot(vp, xp), Expr.dot(vm, xm)])
        M.objective('obj', ObjectiveSense.Maximize, Expr.sub(
            Expr.sub(Expr.dot(m, x), varcost_terms), Expr.mul(delta, s)))

        M.solve()
        portfolio_weights = x.level()
    return portfolio_weights


def max_quad_util_portf_pyportf(N, m, cov, risk_aversion=1.0, transaction_cost=0.0, 
                                lb=None, ub=None, tol=1e-6, debug=False):
    # Solution provided by PyPortfolio
    ef = EfficientFrontier(m, cov, weight_bounds=(lb, ub))
    ef.add_objective(objective_functions.transaction_cost, w_prev=np.zeros(N), k=transaction_cost)
    ef.max_quadratic_utility(risk_aversion=risk_aversion)
    weights_pyportf = np.array(list(ef.clean_weights().values()))
    return weights_pyportf


def max_quad_util_portf_rehline(N, m, cov, 
                                risk_aversion=1.0, vp=0.0, vm=0.0, 
                                lb=None, ub=None, 
                                max_iter=1000, tol=1e-6, verbose=False, trace_freq=100):
    if lb is None and ub is None:
        A = np.r_[np.c_[np.ones(N), np.ones(N)*-1.0].T]
        b = np.r_[-1.0, 1.0]
    elif lb is None:
        A = np.r_[np.c_[np.ones(N), np.ones(N)*-1.0].T, -np.eye(n)]
        b = np.r_[-1.0, 1.0, np.ones(N)*ub]
    elif ub is None:
        A = np.r_[np.c_[np.ones(N), np.ones(N)*-1.0].T, np.eye(n)]
        b = np.r_[-1.0, 1.0, -np.ones(N)*lb]
    else:
        A = np.r_[np.c_[np.ones(N), np.ones(N)*-1.0].T, np.eye(N), -np.eye(N)]
        b = np.r_[-1.0, 1.0, -np.ones(N)*lb, np.ones(N)*ub]
    pf = MeanVariance(m, cov, A, b, buy_cost=vp, sell_cost=vm)
    weights_rehlinepo = pf.max_quad_util_portf(tol=tol, risk_aversion=risk_aversion, max_iter=max_iter, verbose=verbose, trace_freq=trace_freq)
    return weights_rehlinepo, pf._optimizer


# Since the actual value of Infinity is ignored, we define it solely
# for symbolic purposes:
inf = 0.0


# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

    
def max_quad_util_portf_mosek(N, m, cov, risk_aversion=1.0, vp=0.0, vm=0.0, 
                              lb=None, ub=None, tol=1e-6, debug=False):
    # Open MOSEK and create an environment and task
    # Make a MOSEK environment
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
            numvar = 3*N
            # variables = [x, x+, x-]

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
            # x+ >= 0
            bkx = bkx + [mosek.boundkey.lo]*N
            blx = blx + [0]*N
            bux = bux + [inf]*N
            # x- >= 0
            bkx = bkx + [mosek.boundkey.lo]*N
            blx = blx + [0]*N
            bux = bux + [inf]*N

            # [1] x1+...+xN = 1
            # [2..N+1] x+ - x- = x
            bkc = [mosek.boundkey.fx]*(N+1)
            blc = [1] + [0]*(N)
            buc = [1] + [0]*(N)

            # Note below matrix structure for constraints [1..N+1]
            asub, aval = [], []
            for j in range(3*N):
                # j-th column
                if j < N: 
                    ids, vals = [0, j+1], [1.0, 1.0]
                elif j < 2*N: 
                    ids, vals = [j+1-N], [-1.0]
                else: 
                    ids, vals = [j+1-2*N], [1.0]
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
            c = np.r_[-m/risk_aversion, vp/risk_aversion, vm/risk_aversion]
            
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
                    print("Optimal solution: %s" % xx)
                elif solsta == mosek.solsta.dual_infeas_cer:
                    print("Primal or dual infeasibility.\n")
                elif solsta == mosek.solsta.prim_infeas_cer:
                    print("Primal or dual infeasibility.\n")
                elif mosek.solsta.unknown:
                    print("Unknown solution status")
                else:
                    print("Other solution status")

            return np.array(res[:N])