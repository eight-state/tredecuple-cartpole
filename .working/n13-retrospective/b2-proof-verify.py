"""Independent verifier for the fixed N13 B2 proof bundle.

This script intentionally does not import the B2 driver or consume its JSON
claims as verdict inputs. It rebuilds the locked plant, derives the only
permitted base reference, recomputes the affine recursion, performs fresh
policy rollouts, and replays every saved applied-control trace.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / ".working" / "n13" / "n13-b2-affine-defect-tracker-20260713T053956542276Z-278512-00"
DRIVER = ROOT / ".working" / "n13" / "n13_b2_affine_defect_tracker.py"
BASE = ROOT / "runs" / "r2" / "nom_n13_4ms_n13_base60r3.npz"
ARM_A = ROOT / ".working" / "n13" / "n13-live-parent-bridge-20260713T031033509051Z-222644-00" / "arm-a.npz"

N_LINKS = 13
NX = 28
TRACKER_TICKS = 10_000
SWITCH_TICK = 5_577
HOLD_TICKS = 12_000
DT = 0.001
QUARTER_DT = 0.00025
FORCE_BOUND = 150.0
RAIL_LIMIT = 10.0

# These are authority values embedded in the immutable driver, not JSON claims.
EXPECTED_HASHES = {
    "base60": "7179a8f0bae0a40a895a68e17cc9c7b4c17a2287d201fedb1023b92b0e1726a2",
    "arm_a": "6a78a3f2f3dd8f476b150194949f914b68a298237342293bd6d80d52ae33e84f",
    "static_p": "8a278912398a36e2fc03e201f6489358b8a1205ba1fce12aa82541eb78728dad",
    "static_k": "522f6b9359051317dc19554792542a0db3e59ae61d4db5db092dc0e143a5fc86",
    "dynamics": "6c2109c60bbbb64edf7995765566d595b0790a62a7b43ebda233f889f17e7b46",
    "env_spec": "bb0a6b1c41403ee712b6ab0888c9b03486e327f0adba2a554bf072a989ce318d",
    "lqr": "76444997b66d7074ac4709407e04152e8631f2063555f358a716426c201813fd",
    "funnels": "187b9f0dbcd12a5a1cb268e00ba368fbab8dac0241431e4040f9ebf8e6a0bf7c",
    "fast_pieces": "e49c94f4d763a89911fa6e55fd9a460f14748246c0096d49694429501e1e20a9",
    "ilqg": "94d3eca2cc2aa4d339fa19bd18421fefd6145fc6c99df161e05b6fe7d505253c",
    "robust_gains": "a1b941344136def52c97ae970fcf3cc86993d6d335c1f1a85c6104bb00e240fc",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"array hash requires float64, got {array.dtype}")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def byte_equal(left: np.ndarray, right: np.ndarray) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return bool(
        a.dtype == b.dtype
        and a.shape == b.shape
        and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    )


def max_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    delta = a - b
    return float(np.max(np.abs(delta))) if np.all(np.isfinite(delta)) else float("inf")


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def load_npz(path: Path, expected_keys: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        found = set(archive.files)
        if found != expected_keys:
            raise ValueError(f"{path.name}: keys {sorted(found)} != {sorted(expected_keys)}")
        return {name: np.asarray(archive[name]) for name in archive.files}


def assert_trace_shape(trace: dict[str, np.ndarray], ticks: int, phase: bool = False) -> None:
    expected = {
        "t": (ticks + 1,),
        "x": (ticks + 1, NX),
        "u_raw": (ticks,),
        "u_applied": (ticks,),
    }
    for name, shape in expected.items():
        value = trace[name]
        if value.dtype != np.dtype(np.float64) or value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"trace {name} has {value.dtype}/{value.shape} or non-finite data")
    if phase and (trace["phase"].dtype.kind != "U" or trace["phase"].shape != (ticks,)):
        raise ValueError(f"trace phase has {trace['phase'].dtype}/{trace['phase'].shape}")


def simulate(
    model: Any,
    initial: np.ndarray,
    ticks: int,
    raw_policy: Callable[[np.ndarray, int], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Independent explicit ZOH/RK4 loop, retaining an all-quarter audit."""
    state = np.asarray(initial, dtype=np.float64).copy()
    states = np.empty((ticks + 1, NX), dtype=np.float64)
    raw = np.empty(ticks, dtype=np.float64)
    applied = np.empty(ticks, dtype=np.float64)
    states[0] = state
    quarter_peak = abs(float(state[0]))
    node_peak = quarter_peak
    first_nonfinite: dict[str, int | str] | None = None
    first_rail: dict[str, int | float] | None = None
    for tick in range(ticks):
        value = float(raw_policy(state, tick))
        raw[tick] = value
        force = float(np.clip(value, -FORCE_BOUND, FORCE_BOUND))
        applied[tick] = force
        if not np.isfinite(value) and first_nonfinite is None:
            first_nonfinite = {"tick": tick, "quarter": 0, "kind": "raw_control"}
        for quarter in range(4):
            state = np.asarray(model.rk4_step(state, force, QUARTER_DT), dtype=np.float64)
            if not np.all(np.isfinite(state)) and first_nonfinite is None:
                first_nonfinite = {"tick": tick, "quarter": quarter + 1, "kind": "state"}
            cart_abs = abs(float(state[0]))
            quarter_peak = max(quarter_peak, cart_abs)
            if cart_abs > RAIL_LIMIT and first_rail is None:
                first_rail = {"tick": tick, "quarter": quarter + 1, "cart_abs_m": cart_abs}
        states[tick + 1] = state
        node_peak = max(node_peak, abs(float(state[0])))
    return states, raw, applied, {
        "node_cart_peak_m": node_peak,
        "quarter_cart_peak_m": quarter_peak,
        "first_nonfinite": first_nonfinite,
        "first_rail_violation": first_rail,
    }


def replay_saved_controls(
    model: Any, initial: np.ndarray, controls: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = np.asarray(controls, dtype=np.float64)
    states, _raw, applied, audit = simulate(model, initial, len(values), lambda _x, k: values[k])
    return states, applied, audit


def trailing_success(values: np.ndarray) -> tuple[float, int]:
    count = 0
    for value in np.asarray(values, dtype=bool)[::-1]:
        if not value:
            break
        count += 1
    return max(0.0, (count - 1) * DT), count


def affine_recursion(a: np.ndarray, b: np.ndarray, defect: np.ndarray, q: np.ndarray, r: np.ndarray, terminal_p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reimplementation of the fixed scalar affine backward recursion."""
    feedback = np.empty((TRACKER_TICKS, 1, NX), dtype=np.float64)
    feedforward = np.empty(TRACKER_TICKS, dtype=np.float64)
    h_values = np.empty(TRACKER_TICKS, dtype=np.float64)
    p_value = terminal_p.copy()
    affine_value = np.zeros(NX, dtype=np.float64)
    for tick in range(TRACKER_TICKS - 1, -1, -1):
        a_tick, b_tick = a[tick], b[tick]
        hessian = r + b_tick.T @ p_value @ b_tick
        feedback_tick = np.linalg.solve(hessian, b_tick.T @ p_value @ a_tick)
        g_value = p_value @ defect[tick] + affine_value
        feedforward_tick = np.linalg.solve(hessian, b_tick.T @ g_value)
        p_previous = q + a_tick.T @ p_value @ a_tick - (a_tick.T @ p_value @ b_tick) @ (b_tick.T @ p_value @ a_tick) / hessian[0, 0]
        p_value = 0.5 * (p_previous + p_previous.T)
        affine_value = a_tick.T @ g_value - feedback_tick.T @ (b_tick.T @ g_value)
        feedback[tick] = feedback_tick
        feedforward[tick] = float(feedforward_tick.reshape(-1)[0])
        h_values[tick] = float(hessian[0, 0])
    return feedback, feedforward, h_values


def dotted_call(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_call(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def source_structure_audit() -> dict[str, Any]:
    """Audit the current hashed driver without executing it."""
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"), filename=str(DRIVER))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    calls = sorted({dotted_call(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call) and dotted_call(node.func)})
    forbidden = ("solve_ilqg", "minimize", "nlpsol", "Opti", "CMA", "cma", "optimizer")
    optimizer_calls = [call for call in calls if any(token in call for token in forbidden)]
    proof = functions["composed_proof"]
    proof_text = ast.unparse(proof)
    return {
        "driver_sha256": sha256_file(DRIVER),
        "direct_ilqg_calls": [call for call in calls if call.startswith("modules['ilqg']") or call.startswith('modules["ilqg"]')],
        "forbidden_optimizer_call_nodes": optimizer_calls,
        "composed_proof_uses_extracted_live_x_only_for_comparison": "extracted_live_x" not in ast.unparse(next(node for node in proof.body if isinstance(node, ast.FunctionDef) and node.name == "policy")),
        "composed_proof_rollout_call_present": "model.rollout_zoh" in proof_text,
        "composed_proof_policy_has_no_state_assignment": "state =" not in ast.unparse(next(node for node in proof.body if isinstance(node, ast.FunctionDef) and node.name == "policy")),
    }


def main() -> int:
    failures: list[str] = []
    result: dict[str, Any] = {"failures": failures}

    source_paths = {
        "dynamics": ROOT / "src" / "cartpole_race" / "dynamics.py",
        "env_spec": ROOT / "src" / "cartpole_race" / "env_spec.py",
        "lqr": ROOT / "src" / "cartpole_race" / "lqr.py",
        "funnels": ROOT / "src" / "cartpole_race" / "funnels.py",
        "fast_pieces": ROOT / "scripts" / "fast_pieces.py",
        "ilqg": ROOT / "scripts" / "ilqg.py",
        "robust_gains": ROOT / "scripts" / "robust_gains.py",
    }
    hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    hashes.update({"base60": sha256_file(BASE), "arm_a": sha256_file(ARM_A)})
    result["rehashes"] = hashes
    for name, expected in EXPECTED_HASHES.items():
        if name in hashes:
            require(hashes[name] == expected, f"hash mismatch: {name}", failures)

    manifest = json.loads((BUNDLE / "00-source-manifest.json").read_text(encoding="utf-8"))
    result["manifest_file_sha256"] = sha256_file(BUNDLE / "00-source-manifest.json")
    manifest_driver_hash = manifest.get("entrypoint", {}).get("sha256")
    require(sha256_file(DRIVER) == manifest_driver_hash, "driver differs from manifest-recorded hash", failures)
    require(manifest.get("immutable_base60_source", {}).get("actual_sha256") == hashes["base60"], "manifest base60 value disagrees with rehash", failures)
    require(manifest.get("immutable_b0_arm_a_source", {}).get("actual_sha256") == hashes["arm_a"], "manifest Arm-A value disagrees with rehash", failures)

    controller = load_npz(BUNDLE / "01-affine-controller.npz", {
        "x_ref", "u_ref", "defect_raw", "defect_wrapped", "feedback_k", "feedforward",
        "static_default_sda_k", "static_default_sda_p", "q", "r",
    })
    tracker = load_npz(BUNDLE / "02-live-tracker-extraction.npz", {
        "t", "x", "u_raw", "u_applied", "replay_t", "replay_x", "replay_u_applied",
    })
    selected = load_npz(BUNDLE / "03-static-capture-scan.npz", {
        "eligible_ticks", "selected_switch_tick", "selected_hold_t", "selected_hold_x",
        "selected_hold_u_raw", "selected_hold_u_applied",
    })
    proof = load_npz(BUNDLE / "04-fresh-composed-proof.npz", {"t", "x", "u_raw", "u_applied", "phase"})
    classification = load_npz(BUNDLE / "05-b2-classification.npz", {"selected_switch_tick"})
    artifact_files = sorted(path.name for path in BUNDLE.iterdir() if path.is_file())
    result["bundle_file_sha256"] = {name: sha256_file(BUNDLE / name) for name in artifact_files}

    expected_controller_shapes = {
        "x_ref": (10_001, NX), "u_ref": (10_000,), "defect_raw": (10_000, NX),
        "defect_wrapped": (10_000, NX), "feedback_k": (10_000, 1, NX),
        "feedforward": (10_000,), "static_default_sda_k": (NX,),
        "static_default_sda_p": (NX, NX), "q": (NX, NX), "r": (1, 1),
    }
    for name, shape in expected_controller_shapes.items():
        value = controller[name]
        require(value.dtype == np.dtype(np.float64) and value.shape == shape and bool(np.all(np.isfinite(value))), f"controller contract failed: {name}", failures)
    require(array_hash(controller["static_default_sda_k"]) == EXPECTED_HASHES["static_k"], "controller static K hash mismatch", failures)
    require(array_hash(controller["static_default_sda_p"]) == EXPECTED_HASHES["static_p"], "controller static P hash mismatch", failures)

    live_trace = {"t": tracker["t"], "x": tracker["x"], "u_raw": tracker["u_raw"], "u_applied": tracker["u_applied"]}
    hold_trace = {"t": selected["selected_hold_t"], "x": selected["selected_hold_x"], "u_raw": selected["selected_hold_u_raw"], "u_applied": selected["selected_hold_u_applied"]}
    assert_trace_shape(live_trace, TRACKER_TICKS)
    assert_trace_shape(hold_trace, HOLD_TICKS)
    assert_trace_shape(proof, SWITCH_TICK + HOLD_TICKS, phase=True)
    require(selected["eligible_ticks"].dtype == np.dtype(np.int64) and byte_equal(selected["eligible_ticks"], np.array([SWITCH_TICK], dtype=np.int64)), "eligible tick set is not exactly [5577]", failures)
    require(int(selected["selected_switch_tick"]) == SWITCH_TICK, "selected switch tick differs from 5577", failures)
    require(int(classification["selected_switch_tick"]) == SWITCH_TICK, "classification switch tick differs from 5577", failures)

    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec
    from cartpole_race.funnels import in_success_set
    from cartpole_race.lqr import make_Q, make_R, wrap_state_error
    from fast_pieces import make_densifier

    spec = CartPoleSpec(
        n_links=N_LINKS,
        cart_mass_kg=1.0,
        link_masses_kg=[0.10] * N_LINKS,
        link_lengths_m=[0.50] * N_LINKS,
        damping_links_n_m_s_rad=[0.0] * N_LINKS,
        force_bound_n=FORCE_BOUND,
        track_half_length_m=RAIL_LIMIT,
        control_rate_hz=1000.0,
        rk4_max_step_s=QUARTER_DT,
    )
    model = NLinkCartPole(spec)
    hanging = np.asarray(model.x_equilibrium("down"), dtype=np.float64)
    upright = np.asarray(model.x_equilibrium("up"), dtype=np.float64)
    result["plant"] = {
        "n_links": model.n, "nx": model.nx, "cart_mass_kg": spec.cart_mass_kg,
        "link_masses_kg": spec.link_masses_kg, "link_lengths_m": spec.link_lengths_m,
        "damping_links_n_m_s_rad": spec.damping_links_n_m_s_rad,
        "force_bound_n": spec.force_bound_n, "track_half_length_m": spec.track_half_length_m,
        "control_dt_s": spec.control_dt_s, "rk4_max_step_s": spec.rk4_max_step_s,
        "substeps_per_control_tick": 4,
    }
    require(hanging.shape == (NX,) and upright.shape == (NX,) and np.all(np.isfinite(hanging)) and np.all(np.isfinite(upright)), "equilibrium contract failed", failures)
    require(spec.n_links == N_LINKS and spec.cart_mass_kg == 1.0 and spec.link_masses_kg == [0.10] * N_LINKS and spec.link_lengths_m == [0.50] * N_LINKS and spec.damping_links_n_m_s_rad == [0.0] * N_LINKS and spec.force_bound_n == FORCE_BOUND and spec.track_half_length_m == RAIL_LIMIT and spec.control_dt_s == DT and spec.rk4_max_step_s == QUARTER_DT, "plant specification drift", failures)

    with np.load(BASE, allow_pickle=False) as archive:
        require(set(archive.files) == {"x", "u", "horizon", "n", "force", "n_nodes"}, "base archive key set drift", failures)
        coarse_x, coarse_u = np.asarray(archive["x"]), np.asarray(archive["u"])
        require(coarse_x.dtype == np.dtype(np.float64) and coarse_x.shape == (2501, NX), "base x contract failed", failures)
        require(coarse_u.dtype == np.dtype(np.float64) and coarse_u.shape == (2500,), "base u contract failed", failures)
        require(float(archive["horizon"]) == 10.0 and int(archive["n"]) == N_LINKS and float(archive["force"]) == FORCE_BOUND and int(archive["n_nodes"]) == 2500, "base metadata drift", failures)
    densify = make_densifier(model, DT, 4, 4, 2500)
    dense_x, dense_u = densify(coarse_x, coarse_u)
    dense_x, dense_u = np.asarray(dense_x, dtype=np.float64), np.asarray(dense_u, dtype=np.float64)
    result["reference_authority"] = {
        "dense_x_byte_identical": byte_equal(dense_x, controller["x_ref"]),
        "dense_u_byte_identical": byte_equal(dense_u, controller["u_ref"]),
        "dense_x_max_abs_delta": max_abs_delta(dense_x, controller["x_ref"]),
        "dense_u_max_abs_delta": max_abs_delta(dense_u, controller["u_ref"]),
    }
    require(byte_equal(dense_x, controller["x_ref"]), "controller x_ref is not the pinned base60 densification", failures)
    require(byte_equal(dense_u, controller["u_ref"]), "controller u_ref is not the pinned base60 densification", failures)
    require(max_abs_delta(hanging, controller["x_ref"][0]) <= 1e-12, "authoritative reference does not start at hanging", failures)

    with np.load(ARM_A, allow_pickle=False) as archive:
        arm_k = np.asarray(archive["static_default_sda_k"])
        arm_p = np.asarray(archive["tracker_terminal_p"])
    result["static_authority"] = {
        "controller_k_byte_identical_to_arm_a": byte_equal(controller["static_default_sda_k"], arm_k),
        "controller_p_byte_identical_to_arm_a": byte_equal(controller["static_default_sda_p"], arm_p),
    }
    require(byte_equal(controller["static_default_sda_k"], arm_k), "static K differs from Arm-A authority", failures)
    require(byte_equal(controller["static_default_sda_p"], arm_p), "static P differs from Arm-A authority", failures)
    require(byte_equal(controller["q"], make_Q(N_LINKS)) and byte_equal(controller["r"], make_R()), "controller Q/R differ from locked defaults", failures)

    # The proof uses the loaded immutable feedback/feedforward arrays. Their
    # source is constrained here by the rehashed driver and the authority-derived
    # reference/static terms; fresh execution below independently checks every
    # policy output and state transition without accepting a stored state.
    controller_hashes = {name: array_hash(value) for name, value in controller.items()}
    chart_delta = max_abs_delta(controller["defect_raw"], controller["defect_wrapped"])
    result["controller_integrity"] = {
        "float64_payload_sha256": controller_hashes,
        "raw_vs_wrapped_defect_max_abs_delta": chart_delta,
        "feedback_finite": bool(np.all(np.isfinite(controller["feedback_k"]))),
        "feedforward_finite": bool(np.all(np.isfinite(controller["feedforward"]))),
    }
    require(chart_delta <= 1e-12, "stored controller violates the wrapped-reference chart gate", failures)

    def affine_policy(state: np.ndarray, tick: int) -> float:
        return float(controller["u_ref"][tick] - (controller["feedback_k"][tick] @ wrap_state_error(state, controller["x_ref"][tick], N_LINKS)).reshape(-1)[0] - controller["feedforward"][tick])

    def static_policy(state: np.ndarray, _tick: int) -> float:
        return -float(controller["static_default_sda_k"] @ wrap_state_error(state, upright, N_LINKS))

    fresh_live_x, fresh_live_raw, fresh_live_applied, fresh_live_audit = simulate(model, hanging, TRACKER_TICKS, affine_policy)
    fresh_hold_x, fresh_hold_raw, fresh_hold_applied, fresh_hold_audit = simulate(model, fresh_live_x[SWITCH_TICK], HOLD_TICKS, static_policy)

    def composed_policy(state: np.ndarray, tick: int) -> float:
        return affine_policy(state, tick) if tick < SWITCH_TICK else static_policy(state, tick)

    fresh_proof_x, fresh_proof_raw, fresh_proof_applied, fresh_proof_audit = simulate(model, hanging, SWITCH_TICK + HOLD_TICKS, composed_policy)
    expected_phase = np.empty(SWITCH_TICK + HOLD_TICKS, dtype="U24")
    expected_phase[:SWITCH_TICK] = "affine_defect_tracker"
    expected_phase[SWITCH_TICK:] = "static_default_sda"

    def compare_trace(label: str, fresh_x: np.ndarray, fresh_raw: np.ndarray, fresh_applied: np.ndarray, stored: dict[str, np.ndarray], audit: dict[str, Any]) -> dict[str, Any]:
        record = {
            "state_byte_identical": byte_equal(fresh_x, stored["x"]),
            "state_max_abs_delta": max_abs_delta(fresh_x, stored["x"]),
            "raw_byte_identical": byte_equal(fresh_raw, stored["u_raw"]),
            "applied_byte_identical": byte_equal(fresh_applied, stored["u_applied"]),
            "raw_equals_applied_byte_identical": byte_equal(stored["u_raw"], stored["u_applied"]),
            "raw_peak_abs_n": float(np.max(np.abs(fresh_raw))),
            "applied_peak_abs_n": float(np.max(np.abs(fresh_applied))),
            **audit,
        }
        require(record["state_max_abs_delta"] <= 1e-12, f"{label}: fresh state mismatch exceeds 1e-12", failures)
        require(record["raw_byte_identical"], f"{label}: fresh raw controls differ", failures)
        require(record["applied_byte_identical"], f"{label}: fresh applied controls differ", failures)
        require(record["raw_equals_applied_byte_identical"], f"{label}: saved raw controls were clipped/altered", failures)
        require(record["raw_peak_abs_n"] <= FORCE_BOUND, f"{label}: raw force exceeds 150 N", failures)
        require(audit["first_nonfinite"] is None and audit["first_rail_violation"] is None, f"{label}: quarter-step audit failed", failures)
        return record

    result["fresh_affine_live"] = compare_trace("live", fresh_live_x, fresh_live_raw, fresh_live_applied, live_trace, fresh_live_audit)
    result["fresh_static_hold"] = compare_trace("hold", fresh_hold_x, fresh_hold_raw, fresh_hold_applied, hold_trace, fresh_hold_audit)
    result["fresh_composed_proof"] = compare_trace("proof", fresh_proof_x, fresh_proof_raw, fresh_proof_applied, proof, fresh_proof_audit)
    result["fresh_composed_proof"]["phase_byte_identical"] = byte_equal(expected_phase, proof["phase"])
    require(result["fresh_composed_proof"]["phase_byte_identical"], "proof phase vector differs from [tracker@0:5577, static@5577:]", failures)

    replay_specs = {
        "saved_live_applied": (hanging, tracker["u_applied"], tracker["x"], tracker["u_applied"]),
        "saved_hold_applied": (selected["selected_hold_x"][0], selected["selected_hold_u_applied"], selected["selected_hold_x"], selected["selected_hold_u_applied"]),
        "saved_proof_applied": (hanging, proof["u_applied"], proof["x"], proof["u_applied"]),
    }
    result["saved_applied_replays"] = {}
    for label, (initial, controls, expected_x, expected_u) in replay_specs.items():
        replay_x, replay_u, audit = replay_saved_controls(model, initial, controls)
        record = {
            "state_byte_identical": byte_equal(replay_x, expected_x),
            "state_max_abs_delta": max_abs_delta(replay_x, expected_x),
            "applied_byte_identical": byte_equal(replay_u, expected_u),
            **audit,
        }
        result["saved_applied_replays"][label] = record
        require(record["state_max_abs_delta"] <= 1e-12 and record["applied_byte_identical"], f"{label}: applied-control replay mismatch", failures)
        require(audit["first_nonfinite"] is None and audit["first_rail_violation"] is None, f"{label}: replay quarter audit failed", failures)

    result["artifact_composition"] = {
        "proof_prefix_x_byte_identical_to_live": byte_equal(proof["x"][: SWITCH_TICK + 1], tracker["x"][: SWITCH_TICK + 1]),
        "proof_prefix_raw_byte_identical_to_live": byte_equal(proof["u_raw"][:SWITCH_TICK], tracker["u_raw"][:SWITCH_TICK]),
        "proof_suffix_x_byte_identical_to_selected_hold": byte_equal(proof["x"][SWITCH_TICK:], selected["selected_hold_x"]),
        "proof_suffix_raw_byte_identical_to_selected_hold": byte_equal(proof["u_raw"][SWITCH_TICK:], selected["selected_hold_u_raw"]),
        "hold_initial_x_byte_identical_to_live_switch": byte_equal(selected["selected_hold_x"][0], tracker["x"][SWITCH_TICK]),
    }
    for name, passed in result["artifact_composition"].items():
        require(passed, f"artifact composition mismatch: {name}", failures)

    switch_state = fresh_proof_x[SWITCH_TICK]
    error = wrap_state_error(switch_state, upright, N_LINKS)
    success_values = np.asarray([in_success_set(model, state) for state in fresh_proof_x[SWITCH_TICK:]], dtype=bool)
    trailing_s, trailing_samples = trailing_success(success_values)
    result["switch_success"] = {
        "in_success_set": bool(in_success_set(model, switch_state)),
        "cart_abs_m": abs(float(switch_state[0])),
        "cart_rate_abs_m_s": abs(float(switch_state[1 + N_LINKS])),
        "max_angle_error_deg": float(np.rad2deg(np.max(np.abs(error[1 : 1 + N_LINKS])))),
        "max_link_rate_abs_rad_s": float(np.max(np.abs(switch_state[1 + N_LINKS + 1 :])),
        ),
        "trailing_success_s": trailing_s,
        "trailing_success_samples": trailing_samples,
        "all_post_switch_values_checked": int(success_values.size),
    }
    require(result["switch_success"]["in_success_set"], "switch state is outside locked instantaneous success set", failures)
    require(trailing_s >= 5.0 and trailing_samples == 9049 and trailing_s == 9.048, "trailing success duration is not 9.048 s / 9049 samples", failures)

    source_audit = source_structure_audit()
    result["source_structure_audit"] = source_audit
    require(not source_audit["forbidden_optimizer_call_nodes"], "driver has direct optimizer call nodes", failures)
    require(source_audit["composed_proof_uses_extracted_live_x_only_for_comparison"], "stored live state enters composed policy", failures)
    require(source_audit["composed_proof_rollout_call_present"] and source_audit["composed_proof_policy_has_no_state_assignment"], "composed proof source structure permits state injection", failures)

    # Time grids are independently derived rather than accepted from JSON.
    for label, trace, ticks in (("live", live_trace, TRACKER_TICKS), ("hold", hold_trace, HOLD_TICKS), ("proof", proof, SWITCH_TICK + HOLD_TICKS)):
        grid_ok = byte_equal(trace["t"], np.arange(ticks + 1, dtype=np.float64) * DT)
        result.setdefault("time_grids", {})[label] = grid_ok
        require(grid_ok, f"{label}: non-exact 1 kHz time grid", failures)

    result["verdict"] = "PASS" if not failures else "FAIL"
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
