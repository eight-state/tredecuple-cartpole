"""iLQG / DDP trajectory optimizer for the n-link cart-pole swing-up.

Why this solver (vs direct collocation):
    Collocation poses one big feasibility NLP (all defects + bounds at once).
    For the STIFF, unstable n>=4 swing-up the simulator's 1ms ZOH defects make
    the NLP either uninvertible at coarse meshes (coarse can't replay) or crawl /
    stall on feasibility at fine meshes. iLQG/DDP is shooting-based: it ALWAYS
    holds a dynamically feasible rollout (it integrates the real discrete map),
    and improves it with a regularized backward Riccati pass + forward
    line search. There is no global feasibility constraint to get stuck on, so it
    degrades gracefully (it makes monotone cost progress) instead of NaN-ing.

1ms-replayable BY CONSTRUCTION:
    The discrete dynamics used here ARE the simulator's single ZOH control tick:
    ``control_dt`` held constant, integrated with ``n_sub`` RK4 substeps of size
    ``control_dt / n_sub`` -- byte-for-byte the same arithmetic as
    :meth:`NLinkCartPole.rollout_zoh`. So a dense 1ms ZOH replay of the planned
    controls reproduces the planned states to round-off. Single-source dynamics
    (Principle 1): the discrete step is assembled from ``model.f``.

Control saturation:
    Box-constrained DDP. The backward pass solves the per-step QP
    min_du 0.5 du' Quu du + Qu' du  s.t.  u_lo <= u+du <= u_hi
    in closed form for the scalar control (clamp of the unconstrained step), and
    the forward pass clamps to the force bound -- which the discrete map also
    clamps, keeping plan and replay identical at saturation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from cartpole_race.dynamics import NLinkCartPole


# ----------------------------------------------------------------------------
# Discrete dynamics = the simulator's ZOH tick, plus exact Jacobians.
# ----------------------------------------------------------------------------
def build_discrete_step(model: NLinkCartPole):
    """Compile F(x,u), Fx=dF/dx, Fu=dF/du for ONE simulator ZOH control tick.

    Mirrors :meth:`NLinkCartPole.rollout_zoh` exactly: control held over
    ``control_dt``, integrated with ``n_sub`` RK4 substeps of ``control_dt/n_sub``.
    Returns (F, Fx, Fu, control_dt, n_sub). Force is NOT clamped inside F so the
    linearization stays honest; the caller clamps at the box bound (matching the
    simulator boundary).
    """
    spec = model.spec
    nx = model.nx
    control_dt = spec.control_dt_s
    n_sub = max(1, int(np.ceil(control_dt / spec.rk4_max_step_s)))
    dt_sub = control_dt / n_sub

    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u")

    def rk4(xx, uu, h):
        k1 = model.f(xx, uu)
        k2 = model.f(xx + 0.5 * h * k1, uu)
        k3 = model.f(xx + 0.5 * h * k2, uu)
        k4 = model.f(xx + h * k3, uu)
        return xx + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    xn = x
    for _ in range(n_sub):
        xn = rk4(xn, u, dt_sub)

    F = ca.Function("F", [x, u], [xn], ["x", "u"], ["xn"])
    Fx = ca.Function("Fx", [x, u], [ca.jacobian(xn, x)], ["x", "u"], ["Fx"])
    Fu = ca.Function("Fu", [x, u], [ca.jacobian(xn, u)], ["x", "u"], ["Fu"])
    # Base Jacobian Functions; the solver wraps these in .map(N) once the
    # horizon N is known (one C++ call evaluates the whole trajectory's
    # Jacobians, avoiding N Python<->CasADi round trips -- key for larger n).
    FxB = ca.Function("FxB", [x, u], [ca.jacobian(xn, x)])
    FuB = ca.Function("FuB", [x, u], [ca.jacobian(xn, u)])
    return F, Fx, Fu, control_dt, n_sub, FxB, FuB


# ----------------------------------------------------------------------------
# Cost (quadratic): track upright + bound cart + minimize force.
# ----------------------------------------------------------------------------
@dataclass
class QuadCost:
    """Stage + terminal quadratic cost about a target state (default upright).

    Stage cost per step k:  0.5 (x-xt)'Q(x-xt) + 0.5 R u^2
    Terminal cost:          0.5 (x-xt)'Qf(x-xt)
    Angle errors are wrapped to (-pi, pi] so a 2pi swing-up is not penalized as
    a huge error.
    """

    n: int
    xt: np.ndarray
    Q: np.ndarray
    R: float
    Qf: np.ndarray
    ramp_steps: int = 0   # last `ramp_steps` stages ramp Q -> Qf (a hold funnel)

    def _err(self, x: np.ndarray) -> np.ndarray:
        n = self.n
        e = np.asarray(x, float) - self.xt
        # wrap link-angle errors (indices 1..n)
        e[1:1 + n] = (e[1:1 + n] + np.pi) % (2 * np.pi) - np.pi
        return e

    def _Qk(self, k: int, N: int) -> np.ndarray:
        """Stage weight at step k. With a terminal ramp, the last
        ``ramp_steps`` stages blend Q -> Qf so the trajectory is pulled to AND
        HELD at upright (a funnel), letting iLQG land tightly without a single
        absurdly-stiff terminal node."""
        if self.ramp_steps <= 0 or N - k > self.ramp_steps:
            return self.Q
        # frac: 0 at the start of the ramp, ->1 at the terminal node.
        frac = (k - (N - self.ramp_steps)) / max(self.ramp_steps, 1)
        frac = min(max(frac, 0.0), 1.0)
        return self.Q + frac * (self.Qf - self.Q)

    def stage(self, x, u, k=0, N=1):
        e = self._err(x)
        Qk = self._Qk(k, N)
        return 0.5 * e @ Qk @ e + 0.5 * self.R * u * u

    def terminal(self, x):
        e = self._err(x)
        return 0.5 * e @ self.Qf @ e

    # Derivatives (gradient uses wrapped error; Hessian = (ramped) Q).
    def stage_derivs(self, x, u, k=0, N=1):
        e = self._err(x)
        Qk = self._Qk(k, N)
        lx = Qk @ e
        lu = np.array([self.R * u])
        lxx = Qk
        luu = np.array([[self.R]])
        lux = np.zeros((1, len(x)))
        return lx, lu, lxx, luu, lux

    def terminal_derivs(self, x):
        e = self._err(x)
        return self.Qf @ e, self.Qf


@dataclass
class ILQGResult:
    x: np.ndarray            # (N+1, nx) planned states (the rollout)
    u: np.ndarray            # (N,) planned controls
    cost: float
    cost_history: list
    iters: int
    converged: bool
    wall_s: float
    peak_force: float
    info: dict = field(default_factory=dict)


def default_cost(model: NLinkCartPole, *, w_cart=2.0, w_angle=20.0,
                 w_cartrate=1.0, w_anglerate=2.0, w_u=1e-3,
                 wf_cart=20.0, wf_angle=2000.0, wf_cartrate=200.0,
                 wf_anglerate=400.0, ramp_steps=0, x_target=None) -> QuadCost:
    """Build a QuadCost with sensible upright-tracking weights.

    Terminal weights are heavy (drive all links upright + zero rate, cart bounded)
    so the swing-up actually lands; stage weights are light so the solver is free
    to swing through large angles. Tuned to keep peak force well under 150 N for
    n=1..3 (see __main__ validation)."""
    n = model.n
    nx = model.nx
    xt = (np.asarray(x_target, float) if x_target is not None
          else np.asarray(model.x_equilibrium("up")))
    q = np.concatenate([[w_cart], w_angle * np.ones(n),
                        [w_cartrate], w_anglerate * np.ones(n)])
    qf = np.concatenate([[wf_cart], wf_angle * np.ones(n),
                         [wf_cartrate], wf_anglerate * np.ones(n)])
    return QuadCost(n=n, xt=xt, Q=np.diag(q), R=float(w_u), Qf=np.diag(qf),
                    ramp_steps=int(ramp_steps))


def rollout(F, x0, u, fbound):
    """Forward rollout of the discrete map with force clamped to +-fbound."""
    N = len(u)
    nx = len(x0)
    xs = np.empty((N + 1, nx))
    xs[0] = x0
    uc = np.clip(u, -fbound, fbound)
    for k in range(N):
        xs[k + 1] = np.asarray(F(xs[k], uc[k])).reshape(-1)
    return xs, uc


def total_cost(cost: QuadCost, xs, u):
    J = 0.0
    N = len(u)
    for k in range(N):
        J += cost.stage(xs[k], u[k], k, N)
    J += cost.terminal(xs[-1])
    return float(J)


def solve_ilqg(
    model: NLinkCartPole,
    x0: np.ndarray,
    *,
    horizon_s: float,
    cost: QuadCost | None = None,
    u_init: np.ndarray | None = None,
    force_bound: float | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    mu_init: float = 1e-6,
    mu_min: float = 1e-9,
    mu_max: float = 1e10,
    mu_factor: float = 2.0,
    alphas=None,
    verbose: bool = False,
    discrete=None,
) -> ILQGResult:
    """Box-constrained iLQG/DDP swing-up.

    Args:
        model: shared dynamics.
        x0: initial state (e.g. hanging down).
        horizon_s: trajectory duration; N = round(horizon_s/control_dt) ticks.
        cost: QuadCost; default upright tracker if None.
        u_init: initial control sequence (N,); zeros if None.
        force_bound: |u| clamp (default spec bound).
        max_iter, tol: outer-loop limits.
        mu_*: Levenberg-Marquardt regularization schedule on Quu.
        alphas: line-search step fractions (default 1.1^-(0..9)^2).
        discrete: optional precompiled (F,Fx,Fu,control_dt,n_sub) tuple.
    """
    spec = model.spec
    nx = model.nx
    fbound = float(force_bound if force_bound is not None else spec.force_bound_n)
    if cost is None:
        cost = default_cost(model)
    if discrete is None:
        F, Fx, Fu, control_dt, n_sub, FxB, FuB = build_discrete_step(model)
    else:
        F, Fx, Fu, control_dt, n_sub, FxB, FuB = discrete
    N = int(round(horizon_s / control_dt))
    FxN = FxB.map(N)  # batched Jacobian over the whole horizon
    FuN = FuB.map(N)
    if u_init is None:
        u = np.zeros(N)
    else:
        u = np.array(u_init, float).reshape(-1)[:N].copy()
        if len(u) < N:
            u = np.concatenate([u, np.zeros(N - len(u))])
    if alphas is None:
        alphas = 1.1 ** (-np.arange(10) ** 2)

    x0 = np.asarray(x0, float).reshape(-1)
    xs, u = rollout(F, x0, u, fbound)
    J = total_cost(cost, xs, u)
    Jhist = [J]
    mu = mu_init
    t0 = time.time()
    converged = False
    bp_fail = 0

    for it in range(max_iter):
        # ---- Backward pass: build per-step gains with regularization. -----
        # Precompute per-step Jacobians + cost derivs at the current rollout.
        lx = np.empty((N, nx)); lu = np.empty((N, 1))
        lxx = np.empty((N, nx, nx)); luu = np.empty((N, 1, 1))
        lux = np.empty((N, 1, nx))
        for k in range(N):
            lx[k], lu[k], lxx[k], luu[k], lux[k] = cost.stage_derivs(xs[k], u[k], k, N)
        # Batched Jacobians: one CasADi call each for the whole horizon.
        Xin = xs[:N].T                      # (nx, N)
        Uin = u.reshape(1, N)               # (1, N)
        Ak_flat = np.asarray(FxN(Xin, Uin))  # (nx, nx*N)
        Bk_flat = np.asarray(FuN(Xin, Uin))  # (nx, N)
        Ak = Ak_flat.reshape(nx, N, nx).transpose(1, 0, 2)  # (N, nx, nx)
        Bk = Bk_flat.T.reshape(N, nx, 1)                     # (N, nx, 1)
        Vx, Vxx = cost.terminal_derivs(xs[-1])

        k_ff = np.empty((N, 1))
        K_fb = np.empty((N, 1, nx))
        dV1 = 0.0  # expected decrease, linear term
        dV2 = 0.0  # expected decrease, quadratic term
        backward_ok = True

        for k in range(N - 1, -1, -1):
            A = Ak[k]; B = Bk[k]
            Qx = lx[k] + A.T @ Vx
            Qu = lu[k] + B.T @ Vx
            Qxx = lxx[k] + A.T @ Vxx @ A
            Qux = lux[k] + B.T @ Vxx @ A
            Quu = luu[k] + B.T @ Vxx @ B
            # Levenberg regularization on the control Hessian (state-dim reg via
            # Vxx would also work; control reg is robust and cheap for scalar u).
            Quu_reg = Quu + mu * np.eye(1)
            if not np.all(np.isfinite(Quu_reg)) or Quu_reg[0, 0] <= 0:
                backward_ok = False
                break
            quu = Quu_reg[0, 0]
            # Unconstrained step, then box-clamp (projected Newton for scalar u).
            ulo, uhi = -fbound - u[k], fbound - u[k]  # bounds on du
            k_unc = -Qu[0] / quu
            kff = min(max(k_unc, ulo), uhi)
            clamped = (kff != k_unc)
            # Feedback gain: zero the control direction if clamped (free only
            # when interior), standard box-DDP treatment.
            if clamped:
                Kk = np.zeros((1, nx))
            else:
                Kk = -(Qux / quu)
            k_ff[k] = kff
            K_fb[k] = Kk
            # Expected cost reduction (Tassa et al.): uses ff term.
            dV1 += float(kff * Qu[0])
            dV2 += float(0.5 * kff * quu * kff)
            # Value update (use regularized Quu in the cross terms).
            Vx = Qx + Kk.T @ Quu @ k_ff[k] + Kk.T @ Qu + Qux.T @ k_ff[k]
            Vx = Vx.reshape(-1)
            Vxx = Qxx + Kk.T @ Quu @ Kk + Kk.T @ Qux + Qux.T @ Kk
            Vxx = 0.5 * (Vxx + Vxx.T)

        if not backward_ok:
            mu = min(mu * mu_factor, mu_max)
            bp_fail += 1
            if mu >= mu_max:
                if verbose:
                    print(f"[{it}] backward pass non-PD, mu maxed -> stop")
                break
            continue

        # ---- Forward pass: line search over alpha. ------------------------
        improved = False
        for alpha in alphas:
            xnew = np.empty_like(xs)
            unew = np.empty_like(u)
            xnew[0] = x0
            for k in range(N):
                du = alpha * k_ff[k, 0] + float((K_fb[k] @ (xnew[k] - xs[k]))[0])
                uk = np.clip(u[k] + du, -fbound, fbound)
                unew[k] = uk
                xnew[k + 1] = np.asarray(F(xnew[k], uk)).reshape(-1)
            Jnew = total_cost(cost, xnew, unew)
            exp_dec = -(alpha * dV1 + alpha * alpha * dV2)
            # Accept on actual decrease (ratio test when expectation meaningful).
            if Jnew < J:
                if exp_dec > 1e-12:
                    ratio = (J - Jnew) / exp_dec
                    accept = ratio > 1e-4
                else:
                    accept = True
                if accept:
                    improved = True
                    break

        if improved:
            dJ = J - Jnew
            xs, u, J = xnew, unew, Jnew
            Jhist.append(J)
            mu = max(mu * (1.0 / mu_factor), mu_min)  # relax regularization
            if verbose:
                print(f"[{it}] J={J:.6e} dJ={dJ:.3e} alpha={alpha:.3f} "
                      f"mu={mu:.1e} |u|max={np.max(np.abs(u)):.1f}", flush=True)
            if dJ < tol * (1.0 + abs(J)):
                converged = True
                break
        else:
            mu = min(mu * mu_factor, mu_max)
            if verbose:
                print(f"[{it}] no improvement, mu->{mu:.1e}", flush=True)
            if mu >= mu_max:
                break

    wall = time.time() - t0
    return ILQGResult(
        x=xs, u=u, cost=J, cost_history=Jhist, iters=len(Jhist) - 1,
        converged=converged, wall_s=wall, peak_force=float(np.max(np.abs(u))),
        info={"N": N, "mu_final": mu, "bp_fail": bp_fail},
    )


# ----------------------------------------------------------------------------
# Validation: replay through the REAL simulator rollout_zoh.
# ----------------------------------------------------------------------------
def replay_error(model: NLinkCartPole, res: ILQGResult, x0):
    """Dense 1ms ZOH replay of the planned controls; max state mismatch."""
    u = res.u
    K = len(u)

    def policy(x, t):
        k = min(int(round(t / model.spec.control_dt_s)), K - 1)
        return u[k]

    t_log, x_log, u_log = model.rollout_zoh(
        x0, policy, t_final=K * model.spec.control_dt_s,
        control_dt=model.spec.control_dt_s,
        rk4_max_step=model.spec.rk4_max_step_s,
    )
    err = np.max(np.abs(x_log[:len(res.x)] - res.x))
    return err, x_log


def upright_report(model: NLinkCartPole, x_final):
    n = model.n
    th = np.asarray(x_final)[1:1 + n]
    th_w = (th + np.pi) % (2 * np.pi) - np.pi
    rates = np.asarray(x_final)[1 + n:]
    return np.degrees(th_w), rates


if __name__ == "__main__":
    import argparse
    from cartpole_race.env_spec import CartPoleSpec

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--horizon", type=float, default=None)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--force", type=float, default=150.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    spec = CartPoleSpec().with_n_links(args.n)
    model = NLinkCartPole(spec)
    x0 = model.x_equilibrium("down")
    H = args.horizon if args.horizon is not None else {1: 2.0, 2: 3.5, 3: 4.5, 4: 6.0}.get(args.n, 5.0)
    cost = default_cost(model)
    print(f"n={args.n} horizon={H}s force_bound={args.force}N building discrete...")
    res = solve_ilqg(model, x0, horizon_s=H, cost=cost, force_bound=args.force,
                     max_iter=args.iters, verbose=args.verbose)
    err, _ = replay_error(model, res, x0)
    deg, rates = upright_report(model, res.x[-1])
    print(f"  iters={res.iters} conv={res.converged} wall={res.wall_s:.1f}s "
          f"J={res.cost:.4e} peakF={res.peak_force:.2f}N replay_err={err:.2e}")
    print(f"  final angles(deg)={np.round(deg,2)} rates={np.round(rates,3)}")
    if args.save:
        np.savez(args.save, x=res.x, u=res.u, horizon=H, n=args.n,
                 force=args.force, peak_force=res.peak_force, cost=res.cost)
        print(f"  saved -> {args.save}")
