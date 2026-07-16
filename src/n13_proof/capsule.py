"""Shared authority checks and deterministic N13 fresh-rollout machinery."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

N_LINKS = 13
NX = 28
DT = 0.001
QUARTER_DT = 0.00025
TRACKER_TICKS = 10_000
SWITCH_TICK = 5_577
HOLD_TICKS = 12_000
FORCE_BOUND_N = 150.0
RAIL_LIMIT_M = 10.0

BUNDLE_REL = Path(
    ".working/n13/n13-b2-affine-defect-tracker-20260713T053956542276Z-278512-00"
)
BASE_REL = Path("runs/r2/nom_n13_4ms_n13_base60r3.npz")
ARM_A_REL = Path(
    ".working/n13/n13-live-parent-bridge-20260713T031033509051Z-222644-00/arm-a.npz"
)

EXCLUSIONS = (
    "perturbation robustness",
    "release seed gates",
    "72/72",
    "promotion",
    "statistical robustness",
    "hardware",
)


class AuthorityError(RuntimeError):
    """An immutable input is absent, malformed, or has drifted."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def byte_equal(left: np.ndarray, right: np.ndarray) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return bool(
        a.dtype == b.dtype
        and a.shape == b.shape
        and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    )


def max_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    delta = a - b
    return float(np.max(np.abs(delta))) if np.all(np.isfinite(delta)) else float("inf")


def configure_local_imports(root: Path) -> None:
    for path in (root / "src", root / "scripts"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuthorityError(f"cannot read {path}: {error}") from error
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def authority_record(root: Path, relative: Path, role: str, tier: str) -> dict[str, str]:
    path = root / relative
    return {
        "tier": tier,
        "role": role,
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
    }


@dataclass(frozen=True)
class PreparedRun:
    root: Path
    model: Any
    hanging: np.ndarray
    upright: np.ndarray
    x_ref: np.ndarray
    u_ref: np.ndarray
    feedback: np.ndarray
    feedforward: np.ndarray
    static_k: np.ndarray
    switch_tick: int
    authority_inputs: tuple[dict[str, str], ...]


def _validate_declared_source_closure(root: Path, manifest: dict[str, Any]) -> None:
    sources = manifest.get("source_files")
    require(isinstance(sources, dict), "manifest source_files is absent")
    for relative, record in sources.items():
        require(isinstance(record, dict), f"manifest source record is invalid: {relative}")
        expected = record.get("sha256")
        path = root / relative
        require(isinstance(expected, str), f"manifest source hash is absent: {relative}")
        require(path.is_file(), f"manifest source is absent: {relative}")
        require(sha256_file(path) == expected, f"manifest source hash mismatch: {relative}")


def _load_npz(path: Path, expected_keys: set[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == expected_keys, f"{path.name}: key set drift")
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        raise AuthorityError(f"cannot load {path}: {error}") from error


def prepare_run(root: Path | None = None) -> PreparedRun:
    """Load and integrity-check only the B0/B2 inputs used by the demo."""
    root = project_root() if root is None else root.resolve()
    bundle = root / BUNDLE_REL
    manifest_path = bundle / "00-source-manifest.json"
    controller_path = bundle / "01-affine-controller.npz"
    classification_json_path = bundle / "05-b2-classification.json"
    classification_npz_path = bundle / "05-b2-classification.npz"
    base_path = root / BASE_REL
    arm_a_path = root / ARM_A_REL

    manifest = read_json(manifest_path)
    require(manifest.get("schema") == "n13-b2-source-manifest-v1", "manifest schema drift")
    _validate_declared_source_closure(root, manifest)
    require(
        sha256_file(base_path) == manifest["immutable_base60_source"]["required_sha256"],
        "B0 base archive hash mismatch",
    )
    require(
        sha256_file(arm_a_path) == manifest["immutable_b0_arm_a_source"]["required_sha256"],
        "B0 Arm-A archive hash mismatch",
    )

    classification = read_json(classification_json_path)
    require(
        classification.get("classification") == "N13_ONE_RUN_PASS",
        "B2 classification claim is not N13_ONE_RUN_PASS",
    )
    with np.load(classification_npz_path, allow_pickle=False) as archive:
        require(set(archive.files) == {"selected_switch_tick"}, "classification NPZ key set drift")
        switch_tick = int(np.asarray(archive["selected_switch_tick"]))
    require(switch_tick == SWITCH_TICK, f"B2 switch tick is {switch_tick}, expected {SWITCH_TICK}")

    controller = _load_npz(
        controller_path,
        {
            "x_ref",
            "u_ref",
            "defect_raw",
            "defect_wrapped",
            "feedback_k",
            "feedforward",
            "static_default_sda_k",
            "static_default_sda_p",
            "q",
            "r",
        },
    )
    expected_shapes = {
        "x_ref": (TRACKER_TICKS + 1, NX),
        "u_ref": (TRACKER_TICKS,),
        "feedback_k": (TRACKER_TICKS, 1, NX),
        "feedforward": (TRACKER_TICKS,),
        "static_default_sda_k": (NX,),
        "static_default_sda_p": (NX, NX),
    }
    for name, shape in expected_shapes.items():
        value = controller[name]
        require(
            value.dtype == np.dtype(np.float64) and value.shape == shape and np.all(np.isfinite(value)),
            f"controller contract failed: {name}",
        )

    configure_local_imports(root)
    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec
    from fast_pieces import make_densifier

    spec = CartPoleSpec(
        n_links=N_LINKS,
        cart_mass_kg=1.0,
        link_masses_kg=[0.10] * N_LINKS,
        link_lengths_m=[0.50] * N_LINKS,
        damping_links_n_m_s_rad=[0.0] * N_LINKS,
        force_bound_n=FORCE_BOUND_N,
        track_half_length_m=RAIL_LIMIT_M,
        control_rate_hz=1000.0,
        rk4_max_step_s=QUARTER_DT,
    )
    model = NLinkCartPole(spec)
    hanging = np.asarray(model.x_equilibrium("down"), dtype=np.float64)
    upright = np.asarray(model.x_equilibrium("up"), dtype=np.float64)

    try:
        with np.load(base_path, allow_pickle=False) as archive:
            require(
                set(archive.files) == {"x", "u", "horizon", "n", "force", "n_nodes"},
                "B0 base archive key set drift",
            )
            coarse_x = np.asarray(archive["x"])
            coarse_u = np.asarray(archive["u"])
            require(
                coarse_x.dtype == np.dtype(np.float64) and coarse_x.shape == (2501, NX),
                "B0 base state shape drift",
            )
            require(
                coarse_u.dtype == np.dtype(np.float64) and coarse_u.shape == (2500,),
                "B0 base control shape drift",
            )
    except (OSError, ValueError) as error:
        raise AuthorityError(f"cannot load B0 base archive: {error}") from error
    dense_x, dense_u = make_densifier(model, DT, 4, 4, 2500)(coarse_x, coarse_u)
    require(byte_equal(dense_x, controller["x_ref"]), "B0 dense reference states differ from B2")
    require(byte_equal(dense_u, controller["u_ref"]), "B0 dense reference controls differ from B2")

    try:
        with np.load(arm_a_path, allow_pickle=False) as archive:
            arm_k = np.asarray(archive["static_default_sda_k"])
            arm_p = np.asarray(archive["tracker_terminal_p"])
    except (OSError, KeyError, ValueError) as error:
        raise AuthorityError(f"cannot load B0 Arm-A archive: {error}") from error
    require(byte_equal(arm_k, controller["static_default_sda_k"]), "B0 Arm-A K differs from B2")
    require(byte_equal(arm_p, controller["static_default_sda_p"]), "B0 Arm-A P differs from B2")

    authority_inputs = (
        authority_record(root, BASE_REL, "dense base reference", "B0"),
        authority_record(root, ARM_A_REL, "static SDA K/P", "B0"),
        authority_record(root, BUNDLE_REL / "00-source-manifest.json", "source closure lock", "B2"),
        authority_record(root, BUNDLE_REL / "01-affine-controller.npz", "affine feedback/feedforward", "B2"),
        authority_record(root, BUNDLE_REL / "05-b2-classification.json", "claimed B2 classification", "B2"),
        authority_record(root, BUNDLE_REL / "05-b2-classification.npz", "selected switch tick", "B2"),
    )
    return PreparedRun(
        root=root,
        model=model,
        hanging=hanging,
        upright=upright,
        x_ref=np.asarray(dense_x, dtype=np.float64),
        u_ref=np.asarray(dense_u, dtype=np.float64),
        feedback=np.asarray(controller["feedback_k"], dtype=np.float64),
        feedforward=np.asarray(controller["feedforward"], dtype=np.float64),
        static_k=np.asarray(arm_k, dtype=np.float64),
        switch_tick=switch_tick,
        authority_inputs=authority_inputs,
    )


def trailing_success_seconds(values: np.ndarray) -> tuple[float, int]:
    count = 0
    for value in np.asarray(values, dtype=bool)[::-1]:
        if not value:
            break
        count += 1
    return max(0.0, (count - 1) * DT), count


def fresh_composed_rollout(prepared: PreparedRun) -> dict[str, Any]:
    """Execute one policy-controlled RK4 rollout without loading saved states."""
    from cartpole_race.funnels import in_success_set
    from cartpole_race.lqr import wrap_state_error

    ticks = prepared.switch_tick + HOLD_TICKS
    state = prepared.hanging.copy()
    states = np.empty((ticks + 1, NX), dtype=np.float64)
    raw = np.empty(ticks, dtype=np.float64)
    applied = np.empty(ticks, dtype=np.float64)
    phase = np.empty(ticks, dtype="U24")
    states[0] = state
    node_cart_peak = abs(float(state[0]))
    quarter_cart_peak = node_cart_peak
    first_nonfinite: dict[str, int | str] | None = None
    first_rail_violation: dict[str, int | float] | None = None

    for tick in range(ticks):
        if tick < prepared.switch_tick:
            error = wrap_state_error(state, prepared.x_ref[tick], N_LINKS)
            command = float(
                prepared.u_ref[tick]
                - (prepared.feedback[tick] @ error).reshape(-1)[0]
                - prepared.feedforward[tick]
            )
            phase[tick] = "affine_defect_tracker"
        else:
            command = -float(prepared.static_k @ wrap_state_error(state, prepared.upright, N_LINKS))
            phase[tick] = "static_default_sda"
        raw[tick] = command
        force = float(np.clip(command, -FORCE_BOUND_N, FORCE_BOUND_N))
        applied[tick] = force
        if not np.isfinite(command) and first_nonfinite is None:
            first_nonfinite = {"tick": tick, "quarter": 0, "kind": "raw_control"}
        for quarter in range(4):
            state = np.asarray(prepared.model.rk4_step(state, force, QUARTER_DT), dtype=np.float64)
            if not np.all(np.isfinite(state)) and first_nonfinite is None:
                first_nonfinite = {"tick": tick, "quarter": quarter + 1, "kind": "state"}
            cart_abs = abs(float(state[0]))
            quarter_cart_peak = max(quarter_cart_peak, cart_abs)
            if cart_abs > RAIL_LIMIT_M and first_rail_violation is None:
                first_rail_violation = {
                    "tick": tick,
                    "quarter": quarter + 1,
                    "cart_abs_m": cart_abs,
                }
        states[tick + 1] = state
        node_cart_peak = max(node_cart_peak, abs(float(state[0])))

    success = np.asarray(
        [in_success_set(prepared.model, value) for value in states[prepared.switch_tick :]],
        dtype=bool,
    )
    trailing_s, trailing_samples = trailing_success_seconds(success)
    all_finite = bool(np.all(np.isfinite(states)) and np.all(np.isfinite(raw)))
    gates = {
        "fresh_exact_hanging_start": max_abs_delta(states[0], prepared.hanging) <= 1e-12,
        "all_values_finite": all_finite,
        "raw_equals_applied": byte_equal(raw, applied),
        "raw_peak_le_150_n": bool(np.max(np.abs(raw)) <= FORCE_BOUND_N),
        "node_cart_le_10_m": node_cart_peak <= RAIL_LIMIT_M,
        "quarter_cart_le_10_m": first_nonfinite is None and first_rail_violation is None,
        "switch_in_instantaneous_success_set": bool(success[0]),
        "trailing_in_set_ge_5_s": trailing_s >= 5.0,
    }
    return {
        "t": np.arange(ticks + 1, dtype=np.float64) * DT,
        "x": states,
        "u_raw": raw,
        "u_applied": applied,
        "phase": phase,
        "gates": gates,
        "passed": bool(all(gates.values())),
        "switch_tick": prepared.switch_tick,
        "switch_time_s": prepared.switch_tick * DT,
        "raw_peak_n": float(np.max(np.abs(raw))),
        "node_cart_peak_m": node_cart_peak,
        "quarter_cart_peak_m": quarter_cart_peak,
        "first_nonfinite": first_nonfinite,
        "first_rail_violation": first_rail_violation,
        "trailing_success_s": trailing_s,
        "trailing_success_samples": trailing_samples,
        "saved_tracking_reference_loaded": True,
        "saved_b2_rollout_trace_loaded": False,
    }
