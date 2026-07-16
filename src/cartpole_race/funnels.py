"""Empirical capture funnels: unstable-eigenspace target + sampled ``rho(t)``.

Two pieces, both from the proposal 'Controllers and handoff':

1. **Unstable-eigenspace terminal target.** Left-eigenvector basis ``W_u`` of
   the upright Jacobian ``A_upright`` for ``Re(eig) > 1e-6``; the terminal
   modal coordinate ``z_u = W_u^T e`` (``e = wrap_state_error``). The catch
   targets ``||z_u|| <= eps_u``: zero unstable modal content, not "all angles
   tiny" (stable-mode residual is absorbed downstream).

2. **Empirical ``rho(t)`` (no SOS).** ``rho_static`` first from shell sampling
   on ``dx'P dx = rho`` + CEM/CMA-ES; then the BACKWARD-IN-TIME binary-search
   recursion: ``rho(tf) = rho_static``; for ``k = N-1 .. 0``, the max ``rho_k``
   such that sampled points on ``V_k = rho_k`` flow under TVLQR to
   ``V_{k+1} <= rho_{k+1}`` while respecting force/track. Shell samples 512 for
   2-3 link.

Every failing sample is recorded with the proposal's inspectable schema
(see :func:`counterexample_record`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import scipy.linalg

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.lqr import wrap_state_error
from cartpole_race.tvlqr import TVLQR

SHELL_SAMPLES_2_3LINK = 512
RHO_BINARY_SEARCH_ITERS = 12


# ----------------------------------------------------------------------
# Unstable-eigenspace terminal target
# ----------------------------------------------------------------------
def unstable_left_basis(
    model: NLinkCartPole, tol: float = 1e-6
) -> np.ndarray:
    """Real left-eigenvector basis ``W_u`` for ``Re(eig(A_up)) > tol``.

    Args:
        model: Shared dynamics object.
        tol: Real-part threshold above which a mode is "unstable".

    Returns:
        ``W_u`` of shape ``(nx, m)`` where ``m`` is the number of unstable
        modes (real basis; complex conjugate pairs split into real/imag).
    """
    x_up = model.x_equilibrium("up")
    A, _ = model.linearize(x_up, 0.0)
    evals, left = scipy.linalg.eig(A, left=True, right=False)
    cols = []
    used = np.zeros(len(evals), dtype=bool)
    for i in range(len(evals)):
        if used[i] or evals[i].real <= tol:
            continue
        v = left[:, i]
        if abs(evals[i].imag) < 1e-12:
            cols.append(v.real)
        else:
            # Complex pair -> two real basis vectors (real and imag parts).
            cols.append(v.real)
            cols.append(v.imag)
        used[i] = True
    if not cols:
        return np.zeros((model.nx, 0))
    W = np.column_stack(cols)
    # Orthonormalize for a stable modal coordinate.
    Q, _ = np.linalg.qr(W)
    return Q


def unstable_modal_coord(
    model: NLinkCartPole, x: np.ndarray, W_u: np.ndarray
) -> np.ndarray:
    """Unstable modal coordinate ``z_u = W_u^T wrap_state_error(x, x_up)``."""
    x_up = model.x_equilibrium("up")
    e = wrap_state_error(x, x_up, model.n)
    return W_u.T @ e


# ----------------------------------------------------------------------
# Success predicate (locked) and shell sampling
# ----------------------------------------------------------------------
def in_success_set(
    model: NLinkCartPole,
    x: np.ndarray,
    theta_tol: float = np.deg2rad(5.0),
    thetad_tol: float = 0.5,
    x_tol: float = 2.0,
    xdot_tol: float = 0.5,
) -> bool:
    """The locked upright hold set (proposal success predicate, per-instant).

    ``|wrap(theta_i)| <= 5deg``, ``|thetad_i| <= 0.5``, ``|x| <= 2``,
    ``|xdot| <= 0.5``.
    """
    n = model.n
    x = np.asarray(x).reshape(-1)
    x_up = model.x_equilibrium("up")
    e = wrap_state_error(x, x_up, n)
    theta_err = e[1 : 1 + n]
    thetad = x[1 + n + 1 :]
    return bool(
        np.all(np.abs(theta_err) <= theta_tol)
        and np.all(np.abs(thetad) <= thetad_tol)
        and abs(x[0]) <= x_tol
        and abs(x[1 + n]) <= xdot_tol
    )


def sample_shell(
    P: np.ndarray, rho: float, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample states on the ellipsoid shell ``e'P e = rho`` (as error vectors).

    Draws uniform directions on the sphere, scales each to land exactly on the
    P-ellipsoid shell of level ``rho``.

    Args:
        P: SPD matrix defining the ellipsoid.
        rho: Level value.
        n_samples: Number of shell points.
        rng: Numpy random generator.

    Returns:
        Array of error vectors, shape ``(n_samples, nx)``.
    """
    nx = P.shape[0]
    # WHITENED sampling: uniform on the ellipsoid surface in the P-metric, so
    # every principal axis (stiff AND soft) is covered in proportion. The old
    # isotropic-ray method (d~N(0,I) scaled to the shell) collapses nearly all
    # samples into the stiff-dominated sliver and never reaches the soft-axis
    # extremes -> it hid large soft-direction failures on this 1e8-condition
    # ellipsoid (adversarial-checkpoint finding). e = sqrt(rho) L^-T z, P=L L^T,
    # z uniform on the unit sphere => e' P e = rho.
    L = np.linalg.cholesky(P)
    g = rng.standard_normal((n_samples, nx))
    z = g / np.linalg.norm(g, axis=1, keepdims=True)
    e = np.sqrt(rho) * np.linalg.solve(L.T, z.T).T
    return e


# ----------------------------------------------------------------------
# rho_static via shell sampling (CEM/CMA-ES refinement lives in funnels.refine)
# ----------------------------------------------------------------------
@dataclass
class StaticFunnel:
    """Result of the static-LQR funnel estimation."""

    rho_static: float
    interior_fraction: float
    P: np.ndarray


def estimate_rho_static(
    model: NLinkCartPole,
    policy,
    P: np.ndarray,
    hold_time_s: float = 5.0,
    n_shell: int = SHELL_SAMPLES_2_3LINK,
    rho_grid: np.ndarray | None = None,
    interior_threshold: float = 0.99,
    seed: int = 0,
) -> StaticFunnel:
    """Largest ``rho`` whose shell holds the success set for >= threshold.

    For each candidate ``rho`` (ascending grid), sample the shell, roll the
    static policy for ``hold_time_s``, and require >= ``interior_threshold`` of
    samples to (a) end in the locked success set, (b) never violate
    force/track. The accepted ``rho_static`` is the largest passing level
    (the search stops at the first failing level).

    Args:
        model: Shared dynamics object.
        policy: Static-LQR ``(x, t) -> u`` policy.
        P: Static Riccati matrix.
        hold_time_s: Hold horizon for the convergence test.
        n_shell: Shell sample count.
        rho_grid: Optional ascending candidate levels; default geometric grid.
        interior_threshold: Required interior-convergence fraction.
        seed: RNG seed.

    Returns:
        :class:`StaticFunnel`.
    """
    rng = np.random.default_rng(seed)
    spec = model.spec
    x_up = model.x_equilibrium("up")
    if rho_grid is None:
        rho_grid = np.geomspace(0.02, 50.0, 24)

    control_dt = spec.control_dt_s
    rk4_max = spec.rk4_max_step_s
    track = spec.track_half_length_m
    # Allow a settling transient before the hold window, then require the FINAL
    # hold_time_s continuously in-set (the locked "enters AND remains"
    # predicate). One settling second past the hold horizon.
    settle_time_s = 1.0
    total_t = hold_time_s + settle_time_s
    hold_ticks = int(round(hold_time_s / control_dt))

    best_rho = 0.0
    best_frac = 0.0
    for rho in rho_grid:
        shell = sample_shell(P, rho, n_shell, rng)
        ok = 0
        for e in shell:
            x0 = x_up + e
            t_log, x_log, u_log = model.rollout_zoh(
                x0, policy, total_t, control_dt, rk4_max
            )
            # force/track respected across the whole rollout
            if np.any(np.abs(x_log[:, 0]) > track):
                continue
            if np.any(np.abs(u_log) > spec.force_bound_n + 1e-6):
                continue
            # Final hold_time_s window must be continuously in the success set.
            window = x_log[-(hold_ticks + 1) :]
            if all(in_success_set(model, xx) for xx in window):
                ok += 1
        frac = ok / n_shell
        if frac >= interior_threshold:
            best_rho = float(rho)
            best_frac = frac
        else:
            # Monotone assumption: once a shell fails, larger shells fail too.
            break
    return StaticFunnel(rho_static=best_rho, interior_fraction=best_frac, P=P)


# ----------------------------------------------------------------------
# Backward-in-time rho(t) recursion under TVLQR
# ----------------------------------------------------------------------
@dataclass
class Funnel:
    """Time-varying funnel ``rho(t)`` over the TVLQR horizon."""

    t_grid: np.ndarray
    rho: np.ndarray  # rho(t_k), same length as t_grid

    def rho_at(self, t: float) -> float:
        """Interpolated funnel level at time ``t``."""
        return float(np.interp(t, self.t_grid, self.rho))


def estimate_rho_of_t(
    model: NLinkCartPole,
    tvlqr: TVLQR,
    rho_tf: float,
    n_shell: int = SHELL_SAMPLES_2_3LINK,
    time_spacing_s: float = 0.02,
    bsearch_iters: int = RHO_BINARY_SEARCH_ITERS,
    flow_fraction: float = 0.95,
    seed: int = 0,
) -> Funnel:
    """Backward binary-search recursion for the TVLQR funnel ``rho(t)``.

    ``rho(tf) = rho_tf``. For ``k = N-1 .. 0``, binary-search the max
    ``rho_k`` such that >= ``flow_fraction`` of points on ``V_k = rho_k`` flow
    one time-step forward under the TVLQR closed loop to
    ``V_{k+1} <= rho_{k+1}`` while respecting force/track.

    Args:
        model: Shared dynamics object.
        tvlqr: Built TVLQR providing ``S(t)``, ``value(t,x)`` and ``policy``.
        rho_tf: Terminal level (``rho_static``).
        n_shell: Shell sample count per time index.
        time_spacing_s: Funnel time grid spacing.
        bsearch_iters: Binary-search iterations per index.
        flow_fraction: Required fraction of shell points that flow inward.
        seed: RNG seed.

    Returns:
        :class:`Funnel`.
    """
    rng = np.random.default_rng(seed)
    spec = model.spec
    control_dt = spec.control_dt_s
    rk4_max = spec.rk4_max_step_s
    track = spec.track_half_length_m
    fbound = spec.force_bound_n

    t0, tf = tvlqr.t0, tvlqr.tf
    n_steps = max(2, int(round((tf - t0) / time_spacing_s)) + 1)
    t_grid = np.linspace(t0, tf, n_steps)
    rho = np.zeros(n_steps)
    rho[-1] = rho_tf

    for k in range(n_steps - 2, -1, -1):
        tk = t_grid[k]
        tk1 = t_grid[k + 1]
        dt_step = tk1 - tk
        Sk = tvlqr.S_at(tk)
        x_nom_k, _ = tvlqr._nom_at(tk)

        def flows_inward(rho_k: float) -> bool:
            shell = sample_shell(Sk, rho_k, n_shell, rng)
            ok = 0
            for dx in shell:
                x0 = x_nom_k + dx
                # roll the TVLQR closed loop for one funnel time-step
                n_sub = max(1, int(round(dt_step / control_dt)))
                t_local = tk
                xc = x0.copy()
                viol = False
                for _ in range(n_sub):
                    _, x_log, u_log = model.rollout_zoh(
                        xc,
                        lambda xx, tt, t_local=t_local: tvlqr.policy(
                            xx, t_local + tt
                        ),
                        control_dt,
                        control_dt,
                        rk4_max,
                    )
                    xc = x_log[-1]
                    if abs(xc[0]) > track or abs(u_log[0]) > fbound + 1e-6:
                        viol = True
                        break
                    t_local += control_dt
                if viol:
                    continue
                if tvlqr.value(tk1, xc) <= rho[k + 1] + 1e-12:
                    ok += 1
            return ok / n_shell >= flow_fraction

        # Binary search for max rho_k in (0, rho_hi].
        lo, hi = 0.0, max(rho[k + 1] * 4.0, 1e-3)
        # Expand hi while it still flows (cap a few doublings).
        for _ in range(4):
            if flows_inward(hi):
                lo = hi
                hi *= 2.0
            else:
                break
        for _ in range(bsearch_iters):
            mid = 0.5 * (lo + hi)
            if flows_inward(mid):
                lo = mid
            else:
                hi = mid
        rho[k] = lo

    return Funnel(t_grid=t_grid, rho=rho)


# ----------------------------------------------------------------------
# Inspectable counterexample schema (Principle 4)
# ----------------------------------------------------------------------
@dataclass
class Counterexample:
    """Inspectable counterexample record (proposal schema)."""

    seed: int
    n_links: int
    t_i: float
    rho: float
    state: list[float]
    mode: str
    max_force: float
    min_track_margin: float
    final_state: list[float]
    failure_reason: str


def counterexample_record(**kwargs) -> dict:
    """Build a counterexample dict with the locked schema/key order."""
    return asdict(Counterexample(**kwargs))
