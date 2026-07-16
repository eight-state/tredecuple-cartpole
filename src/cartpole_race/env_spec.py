"""Frozen environment spec for the n-link cart-pole.

This module is the single home of the physical/timing spec. ``dynamics.py``
stays pure equations-of-motion and consumes a :class:`CartPoleSpec` instance.
Keeping the spec here (per the proposal's package layout) avoids leaking
configuration concerns into the symbolic dynamics graph.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class CartPoleSpec(BaseModel):
    """Immutable physical + timing specification for an n-link cart-pole.

    Coordinate convention (authoritative, from the proposal):
        ``q = [x_cart, theta_1, ..., theta_n]``, ``state = [q, qdot]``,
        length ``2 * (n_links + 1)``. ``theta_i = 0`` => link points UP,
        ``theta_i = pi`` => DOWN. Angles are ABSOLUTE world angles.

    All per-link lists must have exactly ``n_links`` entries.
    """

    model_config = ConfigDict(frozen=True)

    n_links: int = 6
    cart_mass_kg: float = 1.0
    link_masses_kg: list[float] = [0.10] * 6
    link_lengths_m: list[float] = [0.50] * 6
    gravity_m_s2: float = 9.81
    damping_cart_n_s_m: float = 0.0
    damping_links_n_m_s_rad: list[float] = [0.0] * 6
    force_bound_n: float = 150.0
    track_half_length_m: float = 10.0

    # Simulation / control timing.
    control_rate_hz: float = 1000.0
    rk4_max_step_s: float = 0.00025

    @field_validator("n_links")
    @classmethod
    def _n_links_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_links must be >= 1")
        return v

    @model_validator(mode="after")
    def _check_list_lengths(self) -> CartPoleSpec:
        """Ensure every per-link list matches ``n_links``."""
        n = self.n_links
        for name in ("link_masses_kg", "link_lengths_m", "damping_links_n_m_s_rad"):
            vals = getattr(self, name)
            if len(vals) != n:
                raise ValueError(
                    f"{name} has length {len(vals)}, expected n_links={n}"
                )
        if self.control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be > 0")
        if self.rk4_max_step_s <= 0.0:
            raise ValueError("rk4_max_step_s must be > 0")
        return self

    @property
    def nx(self) -> int:
        """State dimension ``2 * (n_links + 1)``."""
        return 2 * (self.n_links + 1)

    @property
    def nq(self) -> int:
        """Configuration dimension ``n_links + 1``."""
        return self.n_links + 1

    @property
    def control_dt_s(self) -> float:
        """Zero-order-hold control period (seconds)."""
        return 1.0 / self.control_rate_hz

    def with_n_links(self, n: int) -> CartPoleSpec:
        """Return a copy resized to ``n`` links, reusing per-link defaults.

        Per-link properties (mass, length, damping) are taken from the first
        entry of the current spec and broadcast across ``n`` links. Used by
        tests that sweep ``n = 1, 2, 3, 6`` against one base spec.

        Args:
            n: Target number of links (must be >= 1).

        Returns:
            A new frozen :class:`CartPoleSpec` with ``n_links = n``.
        """
        return CartPoleSpec(
            n_links=n,
            cart_mass_kg=self.cart_mass_kg,
            link_masses_kg=[self.link_masses_kg[0]] * n,
            link_lengths_m=[self.link_lengths_m[0]] * n,
            gravity_m_s2=self.gravity_m_s2,
            damping_cart_n_s_m=self.damping_cart_n_s_m,
            damping_links_n_m_s_rad=[self.damping_links_n_m_s_rad[0]] * n,
            force_bound_n=self.force_bound_n,
            track_half_length_m=self.track_half_length_m,
            control_rate_hz=self.control_rate_hz,
            rk4_max_step_s=self.rk4_max_step_s,
        )


# Mapping from YAML field names (proposal spec) to CartPoleSpec field names.
_YAML_ALIASES = {
    "n_links": "n_links",
    "cart_mass_kg": "cart_mass_kg",
    "link_masses_kg": "link_masses_kg",
    "link_lengths_m": "link_lengths_m",
    "gravity_m_s2": "gravity_m_s2",
    "damping_cart_n_s_m": "damping_cart_n_s_m",
    "damping_links_n_m_s_rad": "damping_links_n_m_s_rad",
    "force_bound_n": "force_bound_n",
    "track_half_length_m": "track_half_length_m",
    "control_rate_hz": "control_rate_hz",
    "rk4_max_step_s": "rk4_max_step_s",
}

# Keys that appear in the proposal YAML but are not part of the M0 dynamics
# spec (they govern controllers/proof, out of M0 scope). Silently ignored.
_IGNORED_YAML_KEYS = {
    "link_inertia",
    "logging_dt_s",
    "hold_time_s",
}


def load_spec(path: str | Path) -> CartPoleSpec:
    """Load a :class:`CartPoleSpec` from a YAML file.

    Unknown keys that belong to later milestones (e.g. ``logging_dt_s``) are
    ignored so a single config file can carry both M0 and downstream fields.

    Args:
        path: Path to a YAML config file.

    Returns:
        A validated, frozen :class:`CartPoleSpec`.

    Raises:
        ValueError: If the YAML contains an unrecognized key.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    kwargs: dict = {}
    for key, value in raw.items():
        if key in _IGNORED_YAML_KEYS:
            continue
        if key not in _YAML_ALIASES:
            raise ValueError(f"Unknown config key {key!r} in {path}")
        kwargs[_YAML_ALIASES[key]] = value
    return CartPoleSpec(**kwargs)
