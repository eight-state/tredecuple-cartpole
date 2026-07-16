"""Static infinite-horizon LQR about the upright equilibrium.

Per the proposal 'Controllers and handoff -> Static LQR':
    ``A, B = linearize(x_upright, 0)``;
    ``P = scipy.linalg.solve_continuous_are(A, B, Q, R)``;
    ``K = R^-1 B' P``;
    ``u = clip(-K @ wrap_state_error(x, x_upright), -Fmax, Fmax)``.

Q/R (proposal-locked):
    Q diag: cart_pos 1.0 | each angle 80.0 | cart_vel 1.0 | each ang_vel 5.0
    R = [[0.02]]

All linearization comes from the single shared dynamics module
(:meth:`NLinkCartPole.linearize`); no EOM is duplicated here (Principle 1).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from cartpole_race.dynamics import NLinkCartPole

# Proposal-locked LQR weights.
Q_CART_POS = 1.0
Q_ANGLE = 80.0
Q_CART_VEL = 1.0
Q_ANG_VEL = 5.0
R_STATIC = 0.02


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to the interval ``(-pi, pi]``."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_state_error(
    x: np.ndarray, x_ref: np.ndarray, n_links: int
) -> np.ndarray:
    """State error ``x - x_ref`` with the angle components wrapped to pi.

    The state layout is ``[x_cart, theta_1..theta_n, xdot, thetad_1..thetad_n]``.
    Only the configuration angles (indices ``1 .. n``) are wrapped; cart
    position, cart velocity and angular velocities are plain differences.

    Args:
        x: Current state, length ``2*(n+1)``.
        x_ref: Reference state, same length.
        n_links: Number of links ``n``.

    Returns:
        Error vector of the same length, angle entries wrapped to ``(-pi, pi]``.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)
    e = x - x_ref
    # Angle indices are 1 .. n (the configuration angles).
    e[1 : 1 + n_links] = wrap_to_pi(e[1 : 1 + n_links])
    return e


def make_Q(n_links: int) -> np.ndarray:
    """Build the proposal-locked diagonal state-cost matrix ``Q``.

    Order: ``[cart_pos, angle*n, cart_vel, ang_vel*n]`` matching the state
    layout ``[q, qdot]`` with ``q = [x_cart, theta_1..theta_n]``.

    Args:
        n_links: Number of links ``n``.

    Returns:
        Diagonal ``Q`` of shape ``(2*(n+1), 2*(n+1))``.
    """
    diag = np.concatenate(
        [
            [Q_CART_POS],
            np.full(n_links, Q_ANGLE),
            [Q_CART_VEL],
            np.full(n_links, Q_ANG_VEL),
        ]
    )
    return np.diag(diag)


def make_R() -> np.ndarray:
    """Return the proposal-locked control-cost matrix ``R = [[0.02]]``."""
    return np.array([[R_STATIC]])


def static_lqr(
    model: NLinkCartPole,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the static upright LQR via continuous algebraic Riccati.

    Args:
        model: Shared dynamics object.
        Q: Optional override for the state cost (default proposal-locked).
        R: Optional override for the control cost (default ``[[0.02]]``).

    Returns:
        ``(K, P)`` where ``K`` has shape ``(1, nx)`` (gain) and ``P`` is the
        ``(nx, nx)`` Riccati solution defining ``V(x) = e'P e``.
    """
    n = model.n
    if Q is None:
        Q = make_Q(n)
    if R is None:
        R = make_R()
    x_up = model.x_equilibrium("up")
    A, B = model.linearize(x_up, 0.0)
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)  # R^-1 B' P, shape (1, nx)
    return K, P


class StaticLQRPolicy:
    """Saturated static-LQR feedback policy ``u = clip(-K e, -Fmax, Fmax)``.

    Implements the ``(state, t) -> force`` policy interface consumed by
    :meth:`NLinkCartPole.rollout_zoh`.
    """

    def __init__(self, model: NLinkCartPole, K: np.ndarray | None = None) -> None:
        """Build the policy, solving the static LQR if ``K`` is not supplied.

        Args:
            model: Shared dynamics object.
            K: Optional precomputed gain ``(1, nx)``. If ``None``, solved here.
        """
        self.model = model
        self.n = model.n
        self.x_up = model.x_equilibrium("up")
        self.fbound = model.spec.force_bound_n
        if K is None:
            K, P = static_lqr(model)
            self.P = P
        self.K = np.asarray(K).reshape(1, -1)

    def __call__(self, x: np.ndarray, t: float) -> float:
        """Return the saturated cart force for state ``x`` (time unused)."""
        del t
        e = wrap_state_error(x, self.x_up, self.n)
        u = float(-(self.K @ e).item())
        return float(np.clip(u, -self.fbound, self.fbound))

    def value(self, x: np.ndarray) -> float:
        """Quadratic cost-to-go ``V(x) = e'P e`` (requires ``P`` available)."""
        e = wrap_state_error(x, self.x_up, self.n)
        return float(e @ self.P @ e)
