"""Checks of abm/ccp_cg.py (Algorithm 2 with constraint generation) on random bundles (CPU, a few seconds):

1. the constraint-generation LP returns an optimal solution of the full LP (value and feasibility, against scipy),
   also when it needs several rounds;
2. with the same seeds, the selector makes the same decisions with constraint generation as with the full LP;
3. Algorithm 2: the returned value is phi at the returned lambda and no worse than the best seed; a bundle of one point
   gives its best vertex; with an eps, the certificate ends the call exactly when val(A) <= eps^2;
4. the step rule keeps its state when ccp_cg returns the same point up to rounding (methods.lambda_changed), and the
   envelope selection still compares exactly.

    python tests/test_ccp_cg.py        (or: python -m pytest tests)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abm.ccp import _phi_terms, _scipy_game  # noqa: E402
from abm.ccp_cg import CCPCGConfig, CCPCGSelector, CGGameLP  # noqa: E402
from abm.methods import lambda_changed  # noqa: E402


def random_bundle(m, K=3, d=20, seed=0):
    """Gram matrices of Jacobians whose rows share a common direction more and more (as along a run)."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(d)
    grams = []
    for i in range(m):
        scale = 3.0 / (1.0 + 0.05 * i)
        J = scale * (rng.uniform(0.5, 1.5, K)[:, None] * base[None, :] + 0.3 * rng.standard_normal((K, d)))
        grams.append(J @ J.T)
    return np.asarray(grams)


def phi_exact(Q, lam):
    return float(np.min(np.einsum("k,ikl,l->i", lam, Q, lam)))


def test_cg_lp_is_optimal_for_the_full_lp():
    Q = random_bundle(3000, seed=1)
    rng = np.random.default_rng(2)
    for work, add in ((200, 200), (10, 5)):
        lp = CGGameLP(3, work=work, add=add)
        for _ in range(8):
            lam_c = rng.dirichlet(np.ones(3))
            G, phis = _phi_terms(Q, lam_c)
            M = 2.0 * G - phis[:, None]
            t, lam = lp.solve(M, phis)
            t_full, _ = _scipy_game(M)
            assert abs(t - t_full) <= 1e-9 * max(1.0, abs(t_full)), (t, t_full)
            assert abs(lam.sum() - 1.0) <= 1e-12 and lam.min() >= -1e-12
            assert float((M @ lam).min()) >= t - 1e-12 * max(1.0, abs(t))
        print(f"  constraint generation (work {work}, add {add}): {lp.solves} LPs in {lp.rounds} rounds, "
              f"largest working set {lp.max_rows} of 3000 rows")
    assert lp.rounds > lp.solves                       # the small working set needed more than one round


def test_cg_and_full_lp_make_the_same_decisions():
    Q = random_bundle(520, seed=3)
    out = {}
    for mode in ("cg", "full"):
        sel = CCPCGSelector(3, CCPCGConfig(lp=mode, N=500, r=5, work=50, add=20, seed=4))
        out[mode] = [sel.solve(Q[:m]) for m in range(400, 521, 5)]
    for (v1, l1), (v2, l2) in zip(out["cg"], out["full"]):
        assert np.max(np.abs(l1 - l2)) <= 1e-8 and abs(v1 - v2) <= 1e-10 * abs(v2)
    print(f"  {len(out['cg'])} decisions: the same lambda with constraint generation and with the full LP")


def test_algorithm_2():
    Q = random_bundle(300, seed=5)
    sel = CCPCGSelector(3, CCPCGConfig(N=300, r=4, seed=6))
    for m in (100, 105, 110):
        v, lam = sel.solve(Q[:m])
        assert abs(v - phi_exact(Q[:m], lam)) <= 1e-12 * abs(v)
        seeds_best = max([phi_exact(Q[:m], e) for e in np.eye(3)] + [phi_exact(Q[:m], sel.lam_A)])
        assert v >= seeds_best * (1.0 - 1e-12)
        assert sel.upper >= v and len(sel.prev) <= 4
    one = CCPCGSelector(3).solve(Q[:1])
    k = int(np.argmax(np.diagonal(Q[0])))
    assert abs(one[0] - Q[0, k, k]) <= 1e-12 * Q[0, k, k] and np.allclose(one[1], np.eye(3)[k])
    probe = CCPCGSelector(3)
    probe.solve(Q[:50])
    val_A = probe.upper
    cert = CCPCGSelector(3, CCPCGConfig(eps=float(np.sqrt(val_A * (1.0 + 1e-9)))))
    v, lam = cert.solve(Q[:50])
    assert cert.certified and lam is None and abs(v - val_A) <= 1e-12 * val_A
    no_cert = CCPCGSelector(3, CCPCGConfig(eps=float(np.sqrt(val_A * (1.0 - 1e-6)))))
    v, lam = no_cert.solve(Q[:50])
    assert not no_cert.certified and lam is not None
    print("  Algorithm 2: value = phi(lambda) >= best seed; one point -> best vertex; certificate iff val(A) <= eps^2")


def test_same_lambda_rule():
    lam = np.array([0.2, 0.3, 0.5])
    noise = np.array([4e-10, -3e-10, -1e-10])
    assert lambda_changed(lam, None, "ccp_cg")
    assert not lambda_changed(lam + noise, lam, "ccp_cg")
    assert lambda_changed(lam + np.array([1e-6, -1e-6, 0.0]), lam, "ccp_cg")
    assert not lambda_changed(lam.copy(), lam, "envelope") and lambda_changed(lam + noise, lam, "envelope")
    print("  same lambda: ccp_cg up to 1e-8 in l1, the envelope exactly")


if __name__ == "__main__":
    for t in (test_cg_lp_is_optimal_for_the_full_lp, test_cg_and_full_lp_make_the_same_decisions, test_algorithm_2,
              test_same_lambda_rule):
        print(t.__name__, flush=True)
        t()
    print("all checks passed")
