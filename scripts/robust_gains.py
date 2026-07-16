"""Hold-gain computation that survives high n (float64 DARE dies at n=13).

Three tiers (measured on this plant, .working/conditioning_ladder.py):
  n<=12 : plain scipy DARE works.
  n=13  : matrix_balance + DARE works (plain fails).
  n>=14 : extended-precision structure-preserving doubling (SDA) in mpmath;
          returns float64 gain after high-precision verification.

API:
  hold_gain(model) -> (Krow, info)   # picks the right tier for model.n
  sda_dare_mp(Ad,Bd,Q,Rv,dps) -> (P64,K64,info)
"""
from __future__ import annotations
import numpy as np
import scipy.linalg as sla


def _zoh(A, B, dt=1e-3):
    nx = A.shape[0]
    M = np.zeros((nx+1, nx+1)); M[:nx,:nx] = A*dt; M[:nx,nx] = B.reshape(-1)*dt
    E = sla.expm(M)
    return E[:nx,:nx], E[:nx,nx:nx+1]


def dare_balanced(Ad, Bd, Q, Rv):
    """DARE after diagonal balancing similarity (recovers n=13)."""
    _, T = sla.matrix_balance(Ad)
    Ab = np.linalg.solve(T, Ad@T); Bb = np.linalg.solve(T, Bd)
    Qb = T.T@Q@T
    P = sla.solve_discrete_are(Ab, Bb, Qb, np.array([[Rv]]))
    Kb = np.linalg.solve(Rv + Bb.T@P@Bb, Bb.T@P@Ab)
    K = (Kb@np.linalg.inv(T)).reshape(-1)     # gain back in original coords
    return K, P, T


def sda_dare_mp(Ad, Bd, Q, Rv, dps=50, iters=60, tol=None):
    """Structure-preserving doubling for DARE in mpmath precision.
    A_{k+1}=A(I+GH)^-1 A ; G_{k+1}=G+A(I+GH)^-1 G A^T ; H_{k+1}=H+A^T H(I+GH)^-1 A
    H -> P. Returns float64 (P, K) plus a high-precision verification report."""
    import mpmath as mp
    mp.mp.dps = dps
    nx = Ad.shape[0]
    def M(a): return mp.matrix(a.tolist())
    A = M(Ad); B = M(Bd.reshape(nx,1)); Qm = M(Q)
    G = B*(B.T/mp.mpf(Rv))
    H = Qm.copy(); Ak = A.copy()
    I = mp.eye(nx)
    tol = mp.mpf(10)**(-(dps-10)) if tol is None else mp.mpf(tol)
    for k in range(iters):
        W = I + G*H
        Winv = W**-1
        AW = Ak*Winv
        A1 = AW*Ak
        G1 = G + AW*G*Ak.T
        H1 = H + Ak.T*H*Winv*Ak
        dH = max(abs(H1[i,j]-H[i,j]) for i in range(nx) for j in range(nx))
        sH = max(abs(H1[i,j]) for i in range(nx) for j in range(nx))
        Ak, G, H = A1, G1, H1
        if dH/sH < tol:
            break
    P = H
    # K = (R + B'PB)^-1 B'PA  in mp, then downcast
    BtPB = (B.T*P*B)[0,0] + mp.mpf(Rv)
    K = (B.T*P*A)/BtPB
    K64 = np.array([float(K[0,j]) for j in range(nx)])
    P64 = np.array([[float(P[i,j]) for j in range(nx)] for i in range(nx)])
    # verify with the FLOAT64-ROUNDED gain in high precision: rho(A - B K64)
    Kmp = M(K64.reshape(1,nx))
    Acl = A - B*Kmp
    ev = mp.eig(Acl, left=False, right=False)
    rho = max(abs(e) for e in ev)
    return P64, K64, dict(doubling_iters=k+1, rho_cl=float(rho),
                          margin=float(1-rho))


def hold_gain(model, Q=None, Rv=None, dps=50):
    """Pick the right tier for model.n; returns (Krow, info)."""
    from cartpole_race.lqr import make_Q, make_R
    n = model.n
    Q = make_Q(n) if Q is None else Q
    Rv = float(make_R()[0,0]) if Rv is None else Rv
    xup = model.x_equilibrium("up")
    A, B = model.linearize(xup, 0.0)
    Ad, Bd = _zoh(A, B, model.spec.control_dt_s)
    if n <= 12:
        P = sla.solve_discrete_are(Ad, Bd, Q, np.array([[Rv]]))
        K = np.linalg.solve(Rv+Bd.T@P@Bd, Bd.T@P@Ad).reshape(-1)
        rho = float(np.max(np.abs(np.linalg.eigvals(Ad-Bd@K.reshape(1,-1)))))
        return K, dict(tier="plain", margin=1-rho)
    if n == 13:
        try:
            K, P, T = dare_balanced(Ad, Bd, Q, Rv)
            rho = float(np.max(np.abs(np.linalg.eigvals(Ad-Bd@K.reshape(1,-1)))))
            if 1-rho > 0:
                return K, dict(tier="balanced", margin=1-rho)
        except Exception:
            pass
    P64, K64, info = sda_dare_mp(Ad, Bd, Q, Rv, dps=dps)
    info["tier"] = f"mpmath{dps}"
    return K64, info


def hold_gain_and_P(model, dps=50):
    """(Krow, P, info) for hold + TVLQR terminal cost, tier-selected by n.
    P from the same computation as K so they are consistent."""
    from cartpole_race.lqr import make_Q, make_R
    n = model.n
    Q = make_Q(n); Rv = float(make_R()[0, 0])
    xup = model.x_equilibrium("up")
    A, B = model.linearize(xup, 0.0)
    Ad, Bd = _zoh(A, B, model.spec.control_dt_s)
    if n <= 12:
        P = sla.solve_discrete_are(Ad, Bd, Q, np.array([[Rv]]))
        K = np.linalg.solve(Rv+Bd.T@P@Bd, Bd.T@P@Ad).reshape(-1)
        rho = float(np.max(np.abs(np.linalg.eigvals(Ad-Bd@K.reshape(1,-1)))))
        return K, P, dict(tier="plain", margin=1-rho)
    P64, K64, info = sda_dare_mp(Ad, Bd, Q, Rv, dps=dps)
    info["tier"] = f"mpmath{dps}"
    return K64, P64, info


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, "src")
    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec
    for n in ([int(sys.argv[1])] if len(sys.argv) > 1 else [13, 14, 15, 16]):
        spec = CartPoleSpec(n_links=n, cart_mass_kg=1.0, link_masses_kg=[0.10]*n,
                            link_lengths_m=[0.50]*n,
                            damping_links_n_m_s_rad=[0.0]*n, force_bound_n=150.0)
        m = NLinkCartPole(spec)
        t0 = time.time()
        K, info = hold_gain(m)
        print(f"n={n}: tier={info['tier']} 1-rho={info['margin']:+.3e} "
              f"|K|max={np.max(np.abs(K)):.2e} ({time.time()-t0:.0f}s)", flush=True)
