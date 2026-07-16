"""Tier-0 fast replacements for the gate's Python loops (bit-exact targets).

fast_densify : mapaccum fold of the gate's densify() (stride x n_sub RK4
               substeps per coarse node). Graph composes the SAME m.f calls
               in the SAME order as NLinkCartPole.rk4_step -> bit-identical
               (verified by check_bitexact_densify).
FastDTVLQR   : DiscreteTVLQR with the per-tick zoh_AB loop replaced by ONE
               batched CasADi call for all (A,B) pairs + the same scipy expm
               and backward Riccati. linearize values are identical (same
               compiled functions, batched); expm/Riccati unchanged code path.
"""
from __future__ import annotations

import casadi as ca
import numpy as np
import scipy.linalg as sla


# ----------------------------------------------------------------------
def make_densifier(model, control_dt: float, n_sub: int, stride: int,
                   n_coarse: int):
    """Build the mapaccum densifier once. Returns densify(Xp, Up) -> (Xd, Ud).

    Reproduces exactly:
        for k: xx = Xp[k]; repeat stride times { n_sub RK4 substeps of u_k };
        appending state after each stride-group of substeps.
    i.e. per coarse node k there are `stride` dense states, integrated from
    the COARSE node state Xp[k] (not chained across nodes) — matching the
    gate's densify().
    """
    nx = model.nx
    dt_sub = control_dt / n_sub
    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u")

    def rk4(xx):
        k1 = model.f(xx, u)
        k2 = model.f(xx + 0.5 * dt_sub * k1, u)
        k3 = model.f(xx + 0.5 * dt_sub * k2, u)
        k4 = model.f(xx + dt_sub * k3, u)
        return xx + (dt_sub / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # one control tick = n_sub substeps; output state after the tick
    xx = x
    for _ in range(n_sub):
        xx = rk4(xx)
    tick = ca.Function("tick", [x, u], [xx])
    # one coarse node = stride ticks from the node state, log each tick end
    tick_chain = tick.mapaccum(stride)          # (nx, stride) per node
    node = ca.Function("node", [x, u], [tick_chain(x, ca.repmat(u, 1, stride))])
    node_map = node.map(n_coarse)               # all nodes in one call

    def densify(Xp: np.ndarray, Up: np.ndarray):
        N = len(Up)
        assert N == n_coarse
        out = np.asarray(node_map(np.asarray(Xp)[:N].T,
                                  np.asarray(Up).reshape(1, N)))
        # out is (nx, N*stride), node-major: states after each tick
        Xd = np.empty((N * stride + 1, model.nx))
        Xd[0] = Xp[0]
        Xd[1:] = out.T
        Ud = np.repeat(np.asarray(Up, dtype=float), stride)
        return Xd, Ud

    return densify


def check_bitexact_densify(model, densify_fast, Xp, Up, control_dt, n_sub,
                           stride, n_check=10):
    """Compare against the gate's Python-loop densify on the first n_check
    coarse nodes. Returns (bit_identical, max_abs_diff)."""
    dt_sub = control_dt / n_sub
    Xd_f, Ud_f = densify_fast(Xp, Up)
    md = 0.0
    same = True
    idx = 1
    for k in range(n_check):
        xx = Xp[k].astype(float).copy()
        for _ in range(stride):
            for _ in range(n_sub):
                xx = model.rk4_step(xx, float(Up[k]), dt_sub)
            if not np.array_equal(xx, Xd_f[idx]):
                same = False
                md = max(md, float(np.max(np.abs(xx - Xd_f[idx]))))
            idx += 1
    return same, md


# ----------------------------------------------------------------------
class FastDTVLQR:
    """Drop-in DiscreteTVLQR with batched linearization.

    Same math, same expm, same backward Riccati loop; the only change is
    evaluating ALL (A_k, B_k) in one batched CasADi call instead of 2N
    individual Python-dispatch calls.
    """

    def __init__(self, model, X, U, dt, Qf=None, Q=None, R=None):
        from cartpole_race.lqr import make_Q, make_R, static_lqr, \
            wrap_state_error
        self._wrap = wrap_state_error
        n = model.n
        nx = model.nx
        N = len(U)
        assert len(X) == N + 1
        Q = make_Q(n) if Q is None else Q
        R = make_R() if R is None else R
        Rv = float(np.asarray(R).reshape(-1)[0])
        if Qf is None:
            _, Qf = static_lqr(model)
        self.model, self.n, self.X, self.U, self.dt, self.N = \
            model, n, X, U, dt, N

        # ---- batched continuous linearization (identical functions) ----
        Amap = model._A_fn.map(N)
        Bmap = model._B_fn.map(N)
        Xa = np.asarray(X)[:N].T          # (nx, N)
        Ua = np.asarray(U).reshape(1, N)
        Aall = np.asarray(Amap(x=Xa, u=Ua)["A"]).reshape(nx, N, nx)
        Ball = np.asarray(Bmap(x=Xa, u=Ua)["B"]).reshape(nx, N)
        # map output is horizontally stacked: columns k*nx:(k+1)*nx
        Ad = np.empty((N, nx, nx))
        Bd = np.empty((N, nx))
        M = np.zeros((nx + 1, nx + 1))
        for k in range(N):
            A = Aall[:, k, :]
            B = Ball[:, k]
            M[:nx, :nx] = A * dt
            M[:nx, nx] = B * dt
            E = sla.expm(M)
            Ad[k] = E[:nx, :nx]
            Bd[k] = E[:nx, nx]

        # ---- identical backward Riccati ----
        Kk = np.empty((N, nx))
        S = Qf.copy()
        for k in range(N - 1, -1, -1):
            a, b = Ad[k], Bd[k]
            sb = S @ b
            den = Rv + b @ sb
            kk = (a.T @ sb) / den
            Kk[k] = kk
            Acl = a - np.outer(b, kk)
            S = Q + Rv * np.outer(kk, kk) + Acl.T @ S @ Acl
            S = 0.5 * (S + S.T)
        self.K = Kk
        self.Ad, self.Bd = Ad, Bd
        self.S0 = S

    def policy(self, x, t):
        k = min(max(int(round(t / self.dt)), 0), self.N - 1)
        e = self._wrap(x, self.X[k], self.n)
        return float(self.U[k] - self.K[k] @ e)
