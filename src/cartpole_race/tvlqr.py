"""Finite-horizon time-varying LQR (TVLQR) catch controller.

Per the proposal 'Controllers and handoff -> Finite-horizon TVLQR catch':
    Backward Riccati ``-Sdot = A'S + SA - SBR^-1B'S + Q``, ``S(tf) = Qf``,
    ``K(t) = R^-1 B' S(t)``, ``V(t, x) = dx' S(t) dx``, integrated with
    ``scipy.integrate.solve_ivp`` (DOP853). ``A(t), B(t)`` come ONLY from the
    shared :meth:`NLinkCartPole.linearize`.

    ``Q_tvlqr = Q_static``, ``R_tvlqr = [[0.02]]``, ``Qf = 25 * P_static``.

The nominal trajectory is supplied as time-sampled ``(t, x_nom, u_nom)``; we
interpolate it to evaluate ``A(t), B(t)`` at arbitrary Riccati integration
times. For the M1 catch the nominal is the upright fixed point held with the
static policy (a near-constant nominal), but the construction supports any
feasible nominal so M2+ can reuse it unchanged.
"""

from __future__ import annotations

import numpy as np
import scipy.integrate

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.lqr import make_Q, make_R, static_lqr, wrap_state_error


class TVLQR:
    """Backward-Riccati finite-horizon LQR about a nominal trajectory.

    Attributes:
        t_grid: Ascending time samples ``[t0 .. tf]`` of the stored solution.
        S_grid: Stacked ``S(t)`` matrices, shape ``(len(t_grid), nx, nx)``.
        K_grid: Stacked gains ``K(t)``, shape ``(len(t_grid), 1, nx)``.
    """

    def __init__(
        self,
        model: NLinkCartPole,
        t_nom: np.ndarray,
        x_nom: np.ndarray,
        u_nom: np.ndarray,
        Qf: np.ndarray,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        n_eval: int = 200,
    ) -> None:
        """Integrate the backward Riccati ODE along a nominal trajectory.

        Args:
            model: Shared dynamics object.
            t_nom: Nominal time samples, ascending, shape ``(N,)``.
            x_nom: Nominal states, shape ``(N, nx)``.
            u_nom: Nominal controls, shape ``(N,)`` or ``(N, 1)``.
            Qf: Terminal cost ``S(tf)``, shape ``(nx, nx)``.
            Q: Running state cost (default proposal-locked).
            R: Running control cost (default ``[[0.02]]``).
            n_eval: Number of dense output points stored across the horizon.
        """
        self.model = model
        self.n = model.n
        self.nx = model.nx
        if Q is None:
            Q = make_Q(self.n)
        if R is None:
            R = make_R()
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.Rinv = np.linalg.inv(self.R)

        self.t_nom = np.asarray(t_nom, dtype=float).reshape(-1)
        self.x_nom = np.asarray(x_nom, dtype=float).reshape(len(self.t_nom), -1)
        self.u_nom = np.asarray(u_nom, dtype=float).reshape(-1)
        self.t0 = float(self.t_nom[0])
        self.tf = float(self.t_nom[-1])

        self._integrate_backward(np.asarray(Qf, dtype=float), n_eval)

    def _nom_at(self, t: float) -> tuple[np.ndarray, float]:
        """Linearly interpolate the nominal ``(x, u)`` at time ``t``."""
        t = float(np.clip(t, self.t0, self.tf))
        # Constant-nominal fast path (M1 upright catch): both endpoints equal.
        if len(self.t_nom) == 2 and np.array_equal(self.x_nom[0], self.x_nom[1]):
            return self.x_nom[0], float(self.u_nom[0])
        idx = np.searchsorted(self.t_nom, t, side="left")
        if idx <= 0:
            return self.x_nom[0].copy(), float(self.u_nom[0])
        if idx >= len(self.t_nom):
            return self.x_nom[-1].copy(), float(self.u_nom[-1])
        lo = idx - 1
        denom = self.t_nom[idx] - self.t_nom[lo]
        w = 0.0 if denom == 0 else (t - self.t_nom[lo]) / denom
        x = (1.0 - w) * self.x_nom[lo] + w * self.x_nom[idx]
        u = (1.0 - w) * self.u_nom[lo] + w * self.u_nom[idx]
        return x, float(u)

    def _AB_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Linearize the shared dynamics at the nominal point at time ``t``."""
        x, u = self._nom_at(t)
        return self.model.linearize(x, u)

    def _riccati_rhs(self, t: float, s_flat: np.ndarray) -> np.ndarray:
        """RHS of ``Sdot = -(A'S + SA - SBR^-1B'S + Q)`` (forward in time t).

        We integrate backward by running ``solve_ivp`` from ``tf`` to ``t0``,
        so the natural ``Sdot`` we hand the integrator is the time derivative
        of ``S`` itself; the backward direction is handled by the integration
        span, not a sign flip here.
        """
        S = s_flat.reshape(self.nx, self.nx)
        S = 0.5 * (S + S.T)  # keep symmetric against numerical drift
        A, B = self._AB_at(t)
        BRiB = B @ self.Rinv @ B.T
        Sdot = -(A.T @ S + S @ A - S @ BRiB @ S + self.Q)
        return Sdot.reshape(-1)

    def _integrate_backward(self, Qf: np.ndarray, n_eval: int) -> None:
        """Solve the Riccati ODE from ``tf`` back to ``t0`` and store ``S, K``."""
        t_eval = np.linspace(self.tf, self.t0, n_eval)
        sol = scipy.integrate.solve_ivp(
            self._riccati_rhs,
            (self.tf, self.t0),
            Qf.reshape(-1),
            method="DOP853",
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(f"TVLQR backward Riccati failed: {sol.message}")

        # Re-order ascending in time for storage/interpolation.
        order = np.argsort(sol.t)
        self.t_grid = sol.t[order]
        S_stack = sol.y.T[order].reshape(-1, self.nx, self.nx)
        # Symmetrize each slice.
        S_stack = 0.5 * (S_stack + np.transpose(S_stack, (0, 2, 1)))
        self.S_grid = S_stack

        K_list = []
        for k in range(len(self.t_grid)):
            A, B = self._AB_at(self.t_grid[k])
            del A
            S = self.S_grid[k]
            K_list.append(self.Rinv @ B.T @ S)
        self.K_grid = np.stack(K_list)  # (N, 1, nx)

    def _interp_weights(self, t: float) -> tuple[int, int, float]:
        """Return ``(lo, hi, w)`` so ``y = (1-w)*y[lo] + w*y[hi]`` for time t."""
        tg = self.t_grid
        t = float(np.clip(t, tg[0], tg[-1]))
        hi = int(np.searchsorted(tg, t, side="left"))
        if hi <= 0:
            return 0, 0, 0.0
        if hi >= len(tg):
            return len(tg) - 1, len(tg) - 1, 0.0
        lo = hi - 1
        denom = tg[hi] - tg[lo]
        w = 0.0 if denom == 0 else (t - tg[lo]) / denom
        return lo, hi, w

    def S_at(self, t: float) -> np.ndarray:
        """Interpolated ``S(t)`` (clamped to the stored horizon)."""
        lo, hi, w = self._interp_weights(t)
        S = (1.0 - w) * self.S_grid[lo] + w * self.S_grid[hi]
        return 0.5 * (S + S.T)

    def K_at(self, t: float) -> np.ndarray:
        """Interpolated gain ``K(t)`` of shape ``(1, nx)``."""
        lo, hi, w = self._interp_weights(t)
        return (1.0 - w) * self.K_grid[lo] + w * self.K_grid[hi]

    def value(self, t: float, x: np.ndarray) -> float:
        """Cost-to-go ``V(t, x) = dx' S(t) dx`` with angle-wrapped error."""
        x_nom, _ = self._nom_at(t)
        dx = wrap_state_error(x, x_nom, self.n)
        S = self.S_at(t)
        return float(dx @ S @ dx)

    def policy(self, x: np.ndarray, t: float) -> float:
        """TVLQR feedback ``u = u_nom(t) - K(t) dx`` (unsaturated).

        Saturation is applied at the simulator boundary in ``rollout_zoh``.
        """
        x_nom, u_nom = self._nom_at(t)
        dx = wrap_state_error(x, x_nom, self.n)
        K = self.K_at(t)
        return float(u_nom - (K @ dx).item())


def build_upright_tvlqr(
    model: NLinkCartPole,
    horizon: float,
    qf_scale: float = 25.0,
    n_eval: int = 200,
) -> TVLQR:
    """Build a TVLQR whose nominal is the upright fixed point held constant.

    The M1 catch nominal is the upright equilibrium (``x_nom = x_up``,
    ``u_nom = 0``) over ``[0, horizon]``; ``Qf = qf_scale * P_static``.

    Args:
        model: Shared dynamics object.
        horizon: Catch horizon ``tf`` (seconds).
        qf_scale: Terminal scaling on the static Riccati ``P`` (proposal 25).
        n_eval: Dense storage resolution.

    Returns:
        A ready :class:`TVLQR`.
    """
    _, P = static_lqr(model)
    Qf = qf_scale * P
    x_up = model.x_equilibrium("up")
    t_nom = np.array([0.0, horizon])
    x_nom = np.vstack([x_up, x_up])
    u_nom = np.array([0.0, 0.0])
    return TVLQR(model, t_nom, x_nom, u_nom, Qf=Qf, n_eval=n_eval)
