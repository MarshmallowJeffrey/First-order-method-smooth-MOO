"""Exact float64 objectives and analytic policy gradients of a finite MDP.

F_k(theta) = (1-gamma) E[sum_t gamma^t (tau KL(pi(.|s_t) || pi_ref) - r_k)] under the normalized
discounted occupancy; pi = softmax of the state-action logits theta.  One sparse LU factorization
serves the occupancy solve and all K value solves.
"""
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import logsumexp


class Oracle:
    def __init__(self, model):
        self.model = model
        self.S, self.A, self.K = model['S'], model['A'], model['K']
        self.gamma, self.tau = model['gamma'], model['tau']
        self.R, self.rho = model['R'], model['rho0']
        self.logref = np.log(model['pi_ref'])
        self.si, self.ai, self.ni = np.nonzero(model['P'])
        self.pdata = model['P'][self.si, self.ai, self.ni]
        self.eye = sparse.eye(self.S, format='csc')
        self.flatP = sparse.csr_matrix(model['P'].reshape(self.S * self.A, self.S))
        self.calls = 0; self.seconds = 0.

    def _system(self, pi):
        pp = sparse.coo_matrix((self.pdata * pi[self.si, self.ai], (self.si, self.ni)), shape=(self.S, self.S)).tocsc()
        return splu(self.eye - self.gamma * pp)

    def evaluate(self, theta, gradient=True):
        """gradient=True: (F, Jacobian K x d); gradient=False: (F, [rewards_1..K, KL term])."""
        t0 = time.perf_counter()
        logits = np.asarray(theta).reshape(self.S, self.A)
        logpi = logits - logsumexp(logits, axis=1, keepdims=True)
        pi = np.exp(logpi)
        lu = self._system(pi)
        d = lu.solve((1 - self.gamma) * self.rho, trans='T')
        c = self.tau * (logpi - self.logref)[:, :, None] - self.R
        cp = np.einsum('sa,sak->sk', pi, c)
        fv = d @ cp
        rewards = np.einsum('s,sa,sak->k', d, pi, self.R)
        reg = self.tau * np.einsum('s,sa,sa->', d, pi, logpi - self.logref)
        if gradient:
            v = lu.solve(cp)
            q = c + self.gamma * (self.flatP @ v).reshape(self.S, self.A, self.K)
            g = d[:, None, None] * pi[:, :, None] * (q - v[:, None, :])
            jac = g.transpose(2, 0, 1).reshape(self.K, -1)
            self.calls += 1
            self.seconds += time.perf_counter() - t0
            return fv, jac
        return fv, np.r_[rewards, reg]

    __call__ = evaluate
