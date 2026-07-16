"""Disposable N13 B2 exact-map affine-defect tracker classifier.

Default mode performs the one fixed B2 classifier only.  It never invokes an
optimizer, never modifies production sources, and publishes immutable artifacts
under a new run directory.  ``--self-test`` is bounded: it imports no local
N13 source, constructs no SDA gain, and runs no plant rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
RUNS_DIRECTORY = WORKSPACE / ".working" / "n13"
DRIVER_PATH = Path(__file__).resolve()
BASE60_SOURCE = WORKSPACE / "runs" / "r2" / "nom_n13_4ms_n13_base60r3.npz"
B0_ARM_A_SOURCE = (
    RUNS_DIRECTORY
    / "n13-live-parent-bridge-20260713T031033509051Z-222644-00"
    / "arm-a.npz"
)

DT_S = 0.001
QUARTER_DT_S = 0.00025
HORIZON_S = 10.0
N_TICKS = 10_000
N_STATES = 10_001
N_LINKS = 13
STATE_SIZE = 28
COARSE_TICKS = 2_500
COARSE_STATES = 2_501
FORCE_BOUND_N = 150.0
RAIL_LIMIT_M = 10.0
HOLD_TICKS = 12_000

BASE60_SHA256 = "7179a8f0bae0a40a895a68e17cc9c7b4c17a2287d201fedb1023b92b0e1726a2"
B0_ARM_A_SHA256 = "6a78a3f2f3dd8f476b150194949f914b68a298237342293bd6d80d52ae33e84f"
STATIC_P_SHA256 = "8a278912398a36e2fc03e201f6489358b8a1205ba1fce12aa82541eb78728dad"
STATIC_K_SHA256 = "522f6b9359051317dc19554792542a0db3e59ae61d4db5db092dc0e143a5fc86"

SOURCE_PATHS = (
    WORKSPACE / "src" / "cartpole_race" / "__init__.py",
    WORKSPACE / "src" / "cartpole_race" / "dynamics.py",
    WORKSPACE / "src" / "cartpole_race" / "env_spec.py",
    WORKSPACE / "src" / "cartpole_race" / "lqr.py",
    WORKSPACE / "src" / "cartpole_race" / "funnels.py",
    WORKSPACE / "src" / "cartpole_race" / "tvlqr.py",
    WORKSPACE / "scripts" / "fast_pieces.py",
    WORKSPACE / "scripts" / "ilqg.py",
    WORKSPACE / "scripts" / "robust_gains.py",
    DRIVER_PATH,
)
RETAINED_SOURCE_SHA256 = {
    "src/cartpole_race/dynamics.py": "6c2109c60bbbb64edf7995765566d595b0790a62a7b43ebda233f889f17e7b46",
    "src/cartpole_race/env_spec.py": "bb0a6b1c41403ee712b6ab0888c9b03486e327f0adba2a554bf072a989ce318d",
    "src/cartpole_race/lqr.py": "76444997b66d7074ac4709407e04152e8631f2063555f358a716426c201813fd",
    "src/cartpole_race/funnels.py": "187b9f0dbcd12a5a1cb268e00ba368fbab8dac0241431e4040f9ebf8e6a0bf7c",
    "scripts/fast_pieces.py": "e49c94f4d763a89911fa6e55fd9a460f14748246c0096d49694429501e1e20a9",
    "scripts/ilqg.py": "94d3eca2cc2aa4d339fa19bd18421fefd6145fc6c99df161e05b6fe7d505253c",
    "scripts/robust_gains.py": "a1b941344136def52c97ae970fcf3cc86993d6d335c1f1a85c6104bb00e240fc",
}

SOURCE_CONTRACT_FAILURE = "N13_B2_SOURCE_CONTRACT_FAILURE"
REFERENCE_FAILURE = "N13_B2_REFERENCE_FAILURE"
REFERENCE_CHART_REJECTED = "N13_B2_REFERENCE_CHART_REJECTED"
AFFINE_RECURSION_REJECTED = "N13_B2_AFFINE_RECURSION_REJECTED"
TRACKER_CLIP = "N13_B2_TRACKER_CLIP"
TRACKER_RAIL = "N13_B2_TRACKER_RAIL"
TRACKER_NONFINITE = "N13_B2_TRACKER_NONFINITE"
TRACKER_EXTRACTION_PASS = "N13_B2_TRACKER_EXTRACTION_PASS"
NO_CAPTURE_ENTRY = "N13_NO_CAPTURE_ENTRY"
CAPTURE_REJECTED = "N13_CAPTURE_REJECTED"
CAPTURE_READY = "N13_B2_CAPTURE_READY_FOR_COMPOSED_PROOF"
ONE_RUN_PASS = "N13_ONE_RUN_PASS"
CHILD_RESULT_PREFIX = "N13_B2_CHILD_RESULT="


class B2Failure(RuntimeError):
    """A classified stop that must not authorize a second B2 tracker run."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def relative_path(path: Path) -> str:
    """Return a workspace-relative authority path."""
    try:
        return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError as error:
        raise B2Failure(
            SOURCE_CONTRACT_FAILURE, f"authority path is outside workspace: {path}"
        ) from error


def sha256_file(path: Path) -> str:
    """Hash one file without importing any local source."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contiguous_float64_bytes(array: np.ndarray) -> bytes:
    """Return the native-float64 C-order payload used by numerical locks."""
    values = np.asarray(array)
    if values.dtype != np.dtype(np.float64):
        raise ValueError(f"expected native float64, got {values.dtype}")
    return np.ascontiguousarray(values).tobytes()


def array_sha256(array: np.ndarray) -> str:
    """Hash a contiguous native-float64 payload; callers own shape checks."""
    return hashlib.sha256(contiguous_float64_bytes(array)).hexdigest()


def arrays_byte_identical(left: np.ndarray, right: np.ndarray) -> bool:
    """Require equal float64 shape and exactly equal C-order bytes."""
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    return bool(
        left_values.dtype == np.dtype(np.float64)
        and right_values.dtype == np.dtype(np.float64)
        and left_values.shape == right_values.shape
        and contiguous_float64_bytes(left_values)
        == contiguous_float64_bytes(right_values)
    )


def finite_peak(values: np.ndarray) -> float:
    """Return the absolute peak, preserving an empty trace as zero."""
    array = np.asarray(values, dtype=float)
    return float(np.max(np.abs(array))) if array.size else 0.0


def max_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    """Return infinity rather than laundering incompatible/non-finite arrays."""
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.shape != right_values.shape:
        return float("inf")
    delta = left_values - right_values
    if not np.all(np.isfinite(delta)):
        return float("inf")
    return finite_peak(delta)


def json_value(value: Any) -> Any:
    """Convert diagnostics to strict JSON values without hiding non-finites."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish one JSON artifact atomically and never replace it."""
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(json_value(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, **arrays: Any) -> None:
    """Publish one NPZ artifact atomically and never replace it."""
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        np.savez(temporary, **arrays)
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def make_run_root() -> Path:
    """Allocate an exclusive disposable B2 directory without replacement."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    for suffix in range(100):
        run_root = RUNS_DIRECTORY / (
            f"n13-b2-affine-defect-tracker-{stamp}-{os.getpid()}-{suffix:02d}"
        )
        try:
            run_root.mkdir()
        except FileExistsError:
            continue
        return run_root
    raise RuntimeError("could not allocate a fresh immutable B2 run directory")


def source_records() -> dict[str, dict[str, str | None]]:
    """Hash the complete local source closure before any local import."""
    records: dict[str, dict[str, str | None]] = {}
    for path in SOURCE_PATHS:
        name = relative_path(path)
        actual = sha256_file(path) if path.is_file() else "MISSING"
        expected = RETAINED_SOURCE_SHA256.get(name)
        records[name] = {
            "sha256": actual,
            "retained_expected_sha256": expected,
            "retained_hash_matches": actual == expected if expected is not None else None,
        }
    return records


def runtime_versions() -> dict[str, str]:
    """Record external runtime versions without importing local modules."""
    try:
        casadi_version = importlib.metadata.version("casadi")
    except importlib.metadata.PackageNotFoundError:
        casadi_version = "MISSING"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "casadi": casadi_version,
    }


def build_source_manifest() -> dict[str, Any]:
    """Build the pre-import source and immutable-base authority record."""
    sources = source_records()
    entrypoint = relative_path(DRIVER_PATH)
    base_actual = sha256_file(BASE60_SOURCE) if BASE60_SOURCE.is_file() else "MISSING"
    arm_a_actual = sha256_file(B0_ARM_A_SOURCE) if B0_ARM_A_SOURCE.is_file() else "MISSING"
    return {
        "schema": "n13-b2-source-manifest-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "entrypoint": {"path": entrypoint, "sha256": sources[entrypoint]["sha256"]},
        "runtime_versions": runtime_versions(),
        "complete_local_import_closure": list(sources),
        "source_files": sources,
        "immutable_base60_source": {
            "path": relative_path(BASE60_SOURCE),
            "required_sha256": BASE60_SHA256,
            "actual_sha256": base_actual,
        },
        "immutable_b0_arm_a_source": {
            "path": relative_path(B0_ARM_A_SOURCE),
            "required_sha256": B0_ARM_A_SHA256,
            "actual_sha256": arm_a_actual,
        },
    }


def assert_initial_source_lock(manifest: dict[str, Any]) -> None:
    """Reject retained-source or base drift before a child can import local code."""
    failures: list[str] = []
    for name, record in manifest["source_files"].items():
        expected = record["retained_expected_sha256"]
        if expected is not None and record["sha256"] != expected:
            failures.append(name)
    base = manifest["immutable_base60_source"]
    if base["actual_sha256"] != base["required_sha256"]:
        failures.append(base["path"])
    arm_a = manifest["immutable_b0_arm_a_source"]
    if arm_a["actual_sha256"] != arm_a["required_sha256"]:
        failures.append(arm_a["path"])
    if failures:
        raise B2Failure(
            SOURCE_CONTRACT_FAILURE,
            f"initial source or base60 hash lock failed: {', '.join(failures)}",
        )


def validate_source_manifest(run_root: Path) -> dict[str, Any]:
    """Rehash the exact declared local closure and base archive in each process."""
    manifest_path = run_root / "00-source-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise B2Failure(
            SOURCE_CONTRACT_FAILURE,
            f"cannot read source manifest: {type(error).__name__}: {error}",
        ) from error

    expected_names = [relative_path(path) for path in SOURCE_PATHS]
    source_files = manifest.get("source_files")
    if (
        not isinstance(source_files, dict)
        or manifest.get("complete_local_import_closure") != expected_names
        or set(source_files) != set(expected_names)
    ):
        raise B2Failure(
            SOURCE_CONTRACT_FAILURE,
            "source manifest does not declare the exact approved local closure",
        )

    failures: list[str] = []
    for path in SOURCE_PATHS:
        name = relative_path(path)
        actual = sha256_file(path) if path.is_file() else "MISSING"
        recorded = source_files[name].get("sha256")
        retained = RETAINED_SOURCE_SHA256.get(name)
        if actual != recorded:
            failures.append(f"manifest drift: {name}")
        if retained is not None and actual != retained:
            failures.append(f"retained hash mismatch: {name}")
    entrypoint = manifest.get("entrypoint", {})
    if (
        entrypoint.get("path") != relative_path(DRIVER_PATH)
        or entrypoint.get("sha256") != sha256_file(DRIVER_PATH)
    ):
        failures.append("entrypoint hash mismatch")
    base = manifest.get("immutable_base60_source", {})
    base_actual = sha256_file(BASE60_SOURCE) if BASE60_SOURCE.is_file() else "MISSING"
    if (
        base.get("path") != relative_path(BASE60_SOURCE)
        or base.get("required_sha256") != BASE60_SHA256
        or base.get("actual_sha256") != base_actual
        or base_actual != BASE60_SHA256
    ):
        failures.append("base60 source hash mismatch")
    arm_a = manifest.get("immutable_b0_arm_a_source", {})
    arm_a_actual = sha256_file(B0_ARM_A_SOURCE) if B0_ARM_A_SOURCE.is_file() else "MISSING"
    if (
        arm_a.get("path") != relative_path(B0_ARM_A_SOURCE)
        or arm_a.get("required_sha256") != B0_ARM_A_SHA256
        or arm_a.get("actual_sha256") != arm_a_actual
        or arm_a_actual != B0_ARM_A_SHA256
    ):
        failures.append("B0 Arm-A controller source hash mismatch")
    if failures:
        raise B2Failure(SOURCE_CONTRACT_FAILURE, "; ".join(failures))
    return manifest


def configure_approved_imports() -> None:
    """Expose only the approved local source roots to an isolated child."""
    for source_root in reversed((WORKSPACE / "src", WORKSPACE / "scripts")):
        source = str(source_root)
        if source in sys.path:
            sys.path.remove(source)
        sys.path.insert(0, source)


def assert_import_closure() -> None:
    """Reject extra or missing workspace Python imports; .venv is out of scope."""
    allowed = {path.resolve() for path in SOURCE_PATHS}
    observed: set[Path] = {DRIVER_PATH.resolve()}
    local_roots = (
        (WORKSPACE / "src" / "cartpole_race").resolve(),
        (WORKSPACE / "scripts").resolve(),
        (WORKSPACE / ".working" / "n13").resolve(),
    )
    for module in tuple(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            path = Path(module_file).resolve()
        except OSError:
            continue
        if path.suffix != ".py":
            continue
        if any(path.is_relative_to(root) for root in local_roots):
            observed.add(path)
    extras = sorted(str(path) for path in observed - allowed)
    missing = sorted(str(path) for path in allowed - observed)
    if extras or missing:
        detail: list[str] = []
        if extras:
            detail.append(f"extra local imports: {extras}")
        if missing:
            detail.append(f"missing declared imports: {missing}")
        raise B2Failure(SOURCE_CONTRACT_FAILURE, "; ".join(detail))


def load_approved_sources() -> dict[str, Any]:
    """Import the declared closure after the manifest was rechecked."""
    local_names = (
        "fast_pieces",
        "ilqg",
        "robust_gains",
        "cartpole_race",
        "cartpole_race.dynamics",
        "cartpole_race.env_spec",
        "cartpole_race.lqr",
        "cartpole_race.funnels",
        "cartpole_race.tvlqr",
    )
    preloaded = [name for name in local_names if name in sys.modules]
    if preloaded:
        raise B2Failure(
            SOURCE_CONTRACT_FAILURE,
            f"local modules were pre-imported before source verification: {preloaded}",
        )
    configure_approved_imports()
    try:
        fast_pieces = importlib.import_module("fast_pieces")
        ilqg = importlib.import_module("ilqg")
        robust_gains = importlib.import_module("robust_gains")
        dynamics = importlib.import_module("cartpole_race.dynamics")
        env_spec = importlib.import_module("cartpole_race.env_spec")
        lqr = importlib.import_module("cartpole_race.lqr")
        funnels = importlib.import_module("cartpole_race.funnels")
        importlib.import_module("cartpole_race.tvlqr")
    except Exception as error:
        raise B2Failure(
            SOURCE_CONTRACT_FAILURE,
            f"approved source import failed: {type(error).__name__}: {error}",
        ) from error
    assert_import_closure()
    return {
        "make_densifier": fast_pieces.make_densifier,
        "ilqg": ilqg,
        "hold_gain_and_P": robust_gains.hold_gain_and_P,
        "NLinkCartPole": dynamics.NLinkCartPole,
        "CartPoleSpec": env_spec.CartPoleSpec,
        "make_Q": lqr.make_Q,
        "make_R": lqr.make_R,
        "wrap_state_error": lqr.wrap_state_error,
        "in_success_set": funnels.in_success_set,
    }


def make_n13_model(modules: dict[str, Any]) -> tuple[Any, np.ndarray, np.ndarray]:
    """Construct the locked all-0.10 kg N13 plant and exact equilibria."""
    spec = modules["CartPoleSpec"](
        n_links=N_LINKS,
        cart_mass_kg=1.0,
        link_masses_kg=[0.10] * N_LINKS,
        link_lengths_m=[0.50] * N_LINKS,
        damping_links_n_m_s_rad=[0.0] * N_LINKS,
        force_bound_n=FORCE_BOUND_N,
        track_half_length_m=RAIL_LIMIT_M,
        control_rate_hz=1_000.0,
        rk4_max_step_s=QUARTER_DT_S,
    )
    model = modules["NLinkCartPole"](spec)
    hanging = np.asarray(model.x_equilibrium("down"), dtype=np.float64)
    upright = np.asarray(model.x_equilibrium("up"), dtype=np.float64)
    if (
        hanging.shape != (STATE_SIZE,)
        or upright.shape != (STATE_SIZE,)
        or not np.all(np.isfinite(hanging))
        or not np.all(np.isfinite(upright))
    ):
        raise B2Failure(SOURCE_CONTRACT_FAILURE, "N13 equilibrium contract failed")
    return model, hanging, upright


def load_and_densify_reference(
    model: Any, make_densifier: Callable[..., Any], hanging: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load only pinned base60r3, then densify it once at the locked schedule."""
    try:
        with np.load(BASE60_SOURCE, allow_pickle=False) as archive:
            coarse_x = np.asarray(archive["x"])
            coarse_u = np.asarray(archive["u"])
            source_keys = sorted(archive.files)
    except (OSError, KeyError, ValueError) as error:
        raise B2Failure(
            REFERENCE_FAILURE,
            f"cannot read pinned base60r3 source: {type(error).__name__}: {error}",
        ) from error

    source_gates = {
        "x_native_float64": coarse_x.dtype == np.dtype(np.float64),
        "u_native_float64": coarse_u.dtype == np.dtype(np.float64),
        "x_shape": coarse_x.shape == (COARSE_STATES, STATE_SIZE),
        "u_shape": coarse_u.shape == (COARSE_TICKS,),
        "finite": bool(np.all(np.isfinite(coarse_x)) and np.all(np.isfinite(coarse_u))),
        "exact_hanging_start": max_abs_delta(coarse_x[0], hanging) <= 1e-12
        if coarse_x.shape == (COARSE_STATES, STATE_SIZE)
        else False,
    }
    if not all(source_gates.values()):
        raise B2Failure(REFERENCE_FAILURE, f"base60r3 source gates failed: {source_gates}")

    densify = make_densifier(model, DT_S, 4, 4, COARSE_TICKS)
    try:
        dense_x, dense_u = densify(coarse_x, coarse_u)
    except BaseException as error:
        raise B2Failure(
            REFERENCE_FAILURE,
            f"pinned base60r3 densification failed: {type(error).__name__}: {error}",
        ) from error
    x_ref = np.asarray(dense_x)
    u_ref = np.asarray(dense_u)
    dense_gates = {
        "x_native_float64": x_ref.dtype == np.dtype(np.float64),
        "u_native_float64": u_ref.dtype == np.dtype(np.float64),
        "x_shape": x_ref.shape == (N_STATES, STATE_SIZE),
        "u_shape": u_ref.shape == (N_TICKS,),
        "finite": bool(np.all(np.isfinite(x_ref)) and np.all(np.isfinite(u_ref))),
        "exact_hanging_start": max_abs_delta(x_ref[0], hanging) <= 1e-12
        if x_ref.shape == (N_STATES, STATE_SIZE)
        else False,
        "control_repeat_exact": arrays_byte_identical(u_ref, np.repeat(coarse_u, 4)),
    }
    if not all(dense_gates.values()):
        raise B2Failure(REFERENCE_FAILURE, f"base60r3 dense reference gates failed: {dense_gates}")
    return np.ascontiguousarray(x_ref), np.ascontiguousarray(u_ref), {
        "base60_path": relative_path(BASE60_SOURCE),
        "base60_sha256": BASE60_SHA256,
        "source_keys": source_keys,
        "source_gates": source_gates,
        "densifier": {
            "builder": "make_densifier(model, 0.001, 4, 4, 2500)",
            "dense_x_shape": list(x_ref.shape),
            "dense_u_shape": list(u_ref.shape),
            "dense_x_contiguous_float64_sha256": array_sha256(x_ref),
            "dense_u_contiguous_float64_sha256": array_sha256(u_ref),
            "gates": dense_gates,
        },
    }


def reshape_exact_map_jacobians(
    fx_flat: np.ndarray, fu_flat: np.ndarray, nx: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Use the same CasADi map-axis reshape as ``solve_ilqg`` exactly."""
    fx_values = np.asarray(fx_flat)
    fu_values = np.asarray(fu_flat)
    if fx_values.shape != (nx, nx * horizon) or fu_values.shape != (nx, horizon):
        raise B2Failure(
            AFFINE_RECURSION_REJECTED,
            f"exact-map Jacobian output shapes are {fx_values.shape}/{fu_values.shape}",
        )
    if (
        fx_values.dtype != np.dtype(np.float64)
        or fu_values.dtype != np.dtype(np.float64)
        or not np.all(np.isfinite(fx_values))
        or not np.all(np.isfinite(fu_values))
    ):
        raise B2Failure(AFFINE_RECURSION_REJECTED, "exact-map Jacobians are not finite float64")
    a = fx_values.reshape(nx, horizon, nx).transpose(1, 0, 2)
    b = fu_values.T.reshape(horizon, nx, 1)
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def exact_map_and_defect(
    modules: dict[str, Any], model: Any, x_ref: np.ndarray, u_ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Bind all seven map values, batch Jacobians, and enforce the local chart."""
    try:
        F, Fx, Fu, control_dt, n_sub, FxB, FuB = modules["ilqg"].build_discrete_step(model)
    except BaseException as error:
        raise B2Failure(
            AFFINE_RECURSION_REJECTED,
            f"complete build_discrete_step binding failed: {type(error).__name__}: {error}",
        ) from error
    if control_dt != DT_S or n_sub != 4:
        raise B2Failure(
            AFFINE_RECURSION_REJECTED,
            f"exact map schedule is not 1 ms/four substeps: {(control_dt, n_sub)}",
        )
    if not all(callable(item) for item in (F, Fx, Fu, FxB, FuB)):
        raise B2Failure(AFFINE_RECURSION_REJECTED, "exact-map return is not seven callable/map values")

    try:
        fx_map = FxB.map(N_TICKS)
        fu_map = FuB.map(N_TICKS)
        x_input = x_ref[:-1].T
        u_input = u_ref.reshape(1, N_TICKS)
        fx_flat = np.asarray(fx_map(x_input, u_input))
        fu_flat = np.asarray(fu_map(x_input, u_input))
        a, b = reshape_exact_map_jacobians(fx_flat, fu_flat, STATE_SIZE, N_TICKS)
        x_next_raw = np.empty((N_TICKS, STATE_SIZE), dtype=np.float64)
        for tick in range(N_TICKS):
            x_next_raw[tick] = np.asarray(F(x_ref[tick], u_ref[tick])).reshape(-1)
    except B2Failure:
        raise
    except BaseException as error:
        raise B2Failure(
            AFFINE_RECURSION_REJECTED,
            f"exact-map defect/Jacobian evaluation failed: {type(error).__name__}: {error}",
        ) from error

    if not np.all(np.isfinite(x_next_raw)):
        raise B2Failure(AFFINE_RECURSION_REJECTED, "exact-map reference successors are non-finite")
    d_raw = x_next_raw - x_ref[1:]
    wrap_state_error = modules["wrap_state_error"]
    defect = np.asarray(
        [wrap_state_error(x_next_raw[tick], x_ref[tick + 1], N_LINKS) for tick in range(N_TICKS)],
        dtype=np.float64,
    )
    chart_delta = max_abs_delta(d_raw, defect)
    if not np.all(np.isfinite(defect)) or chart_delta > 1e-12:
        raise B2Failure(
            REFERENCE_CHART_REJECTED,
            f"raw/wrapped defect chart mismatch is {chart_delta}",
        )
    return a, b, np.ascontiguousarray(d_raw), np.ascontiguousarray(defect), {
        "builder": "ilqg.build_discrete_step(model)",
        "bound_return": "(F, Fx, Fu, control_dt, n_sub, FxB, FuB)",
        "control_dt_s": control_dt,
        "rk4_substeps_per_tick": n_sub,
        "fx_map_output_shape": list(fx_flat.shape),
        "fu_map_output_shape": list(fu_flat.shape),
        "a_shape": list(a.shape),
        "b_shape": list(b.shape),
        "a_contiguous_float64_sha256": array_sha256(a),
        "b_contiguous_float64_sha256": array_sha256(b),
        "raw_defect_contiguous_float64_sha256": array_sha256(d_raw),
        "wrapped_defect_contiguous_float64_sha256": array_sha256(defect),
        "raw_vs_wrapped_max_abs_delta": chart_delta,
        "raw_chart_equivalent_le_1e-12": chart_delta <= 1e-12,
    }


def static_sda_contract(modules: dict[str, Any], model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build the one default-Q/R 50-digit SDA contract used twice by B2."""
    try:
        q = np.asarray(modules["make_Q"](N_LINKS), dtype=np.float64)
        r = np.asarray(modules["make_R"](), dtype=np.float64)
        k, runtime_p, info = modules["hold_gain_and_P"](model, dps=50)
        if sha256_file(B0_ARM_A_SOURCE) != B0_ARM_A_SHA256:
            raise ValueError("pinned B0 Arm-A controller artifact drifted")
        with np.load(B0_ARM_A_SOURCE, allow_pickle=False) as arm_a:
            p = np.asarray(arm_a["tracker_terminal_p"])
            stored_k = np.asarray(arm_a["static_default_sda_k"])
    except BaseException as error:
        raise B2Failure(
            AFFINE_RECURSION_REJECTED,
            f"default 50-digit SDA construction failed: {type(error).__name__}: {error}",
        ) from error
    gain = np.ascontiguousarray(np.asarray(k, dtype=np.float64).reshape(-1))
    runtime_p = np.ascontiguousarray(np.asarray(runtime_p, dtype=np.float64))
    stored_k = np.ascontiguousarray(np.asarray(stored_k, dtype=np.float64).reshape(-1))
    terminal_p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    q = np.ascontiguousarray(q)
    r = np.ascontiguousarray(r)
    gates = {
        "q_shape": q.shape == (STATE_SIZE, STATE_SIZE),
        "r_shape": r.shape == (1, 1),
        "k_shape": gain.shape == (STATE_SIZE,),
        "p_shape": terminal_p.shape == (STATE_SIZE, STATE_SIZE),
        "finite": bool(
            np.all(np.isfinite(q))
            and np.all(np.isfinite(r))
            and np.all(np.isfinite(gain))
            and np.all(np.isfinite(runtime_p))
            and np.all(np.isfinite(stored_k))
            and np.all(np.isfinite(terminal_p))
        ),
        "tier_mpmath50": info.get("tier") == "mpmath50",
        "p_hash": array_sha256(terminal_p) == STATIC_P_SHA256,
        "k_hash": array_sha256(gain) == STATIC_K_SHA256,
        "stored_k_matches_runtime": arrays_byte_identical(stored_k, gain),
        "runtime_p_numerically_matches_authority": bool(
            runtime_p.shape == terminal_p.shape
            and np.allclose(runtime_p, terminal_p, rtol=2e-15, atol=1e-12)
        ),
    }
    if not all(gates.values()):
        raise B2Failure(
            AFFINE_RECURSION_REJECTED,
            f"default 50-digit SDA contract failed: {gates}; info={info}",
        )
    return q, r, gain, terminal_p, {
        "dps": 50,
        "info": info,
        "q_contiguous_float64_sha256": array_sha256(q),
        "r_contiguous_float64_sha256": array_sha256(r),
        "k_contiguous_float64_sha256": array_sha256(gain),
        "p_contiguous_float64_sha256": array_sha256(terminal_p),
        "runtime_p_contiguous_float64_sha256": array_sha256(runtime_p),
        "runtime_p_vs_authority_max_abs_delta": max_abs_delta(runtime_p, terminal_p),
        "authority_artifact": relative_path(B0_ARM_A_SOURCE),
        "authority_artifact_sha256": B0_ARM_A_SHA256,
        "gates": gates,
    }


def affine_backward_recursion(
    a: np.ndarray,
    b: np.ndarray,
    defect: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    terminal_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform the fixed affine LQR recursion with terminal p[N] = 0."""
    horizon, nx, nx_second = a.shape
    valid_shapes = bool(
        nx == nx_second
        and b.shape == (horizon, nx, 1)
        and defect.shape == (horizon, nx)
        and q.shape == (nx, nx)
        and r.shape == (1, 1)
        and terminal_p.shape == (nx, nx)
    )
    if not valid_shapes or not all(
        np.all(np.isfinite(item)) for item in (a, b, defect, q, r, terminal_p)
    ):
        raise B2Failure(AFFINE_RECURSION_REJECTED, "affine recursion input contract failed")

    feedback = np.empty((horizon, 1, nx), dtype=np.float64)
    feedforward = np.empty(horizon, dtype=np.float64)
    h_values = np.empty(horizon, dtype=np.float64)
    p_value = terminal_p.copy()
    affine_value = np.zeros(nx, dtype=np.float64)
    for tick in range(horizon - 1, -1, -1):
        a_tick = a[tick]
        b_tick = b[tick]
        hessian = r + b_tick.T @ p_value @ b_tick
        if hessian.shape != (1, 1) or not np.all(np.isfinite(hessian)) or hessian[0, 0] <= 0.0:
            raise B2Failure(
                AFFINE_RECURSION_REJECTED,
                f"non-positive or non-finite H at tick {tick}: {hessian}",
            )
        try:
            feedback_tick = np.linalg.solve(hessian, b_tick.T @ p_value @ a_tick)
            g_value = p_value @ defect[tick] + affine_value
            feedforward_tick = np.linalg.solve(hessian, b_tick.T @ g_value)
            p_previous = (
                q
                + a_tick.T @ p_value @ a_tick
                - (a_tick.T @ p_value @ b_tick) @ (b_tick.T @ p_value @ a_tick) / hessian[0, 0]
            )
            affine_previous = a_tick.T @ g_value - feedback_tick.T @ (b_tick.T @ g_value)
        except np.linalg.LinAlgError as error:
            raise B2Failure(
                AFFINE_RECURSION_REJECTED,
                f"affine recursion solve failed at tick {tick}: {error}",
            ) from error
        p_previous = 0.5 * (p_previous + p_previous.T)
        if not all(
            np.all(np.isfinite(item))
            for item in (feedback_tick, g_value, feedforward_tick, p_previous, affine_previous)
        ):
            raise B2Failure(AFFINE_RECURSION_REJECTED, f"non-finite affine recursion at tick {tick}")
        feedback[tick] = feedback_tick
        feedforward[tick] = float(feedforward_tick.reshape(-1)[0])
        h_values[tick] = float(hessian[0, 0])
        p_value = p_previous
        affine_value = affine_previous
    return feedback, feedforward, h_values, p_value


def build_affine_controller(
    modules: dict[str, Any], model: Any, hanging: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Construct exactly one fixed affine controller from the pinned reset reference."""
    x_ref, u_ref, reference_report = load_and_densify_reference(
        model, modules["make_densifier"], hanging
    )
    a, b, d_raw, defect, map_report = exact_map_and_defect(modules, model, x_ref, u_ref)
    q, r, static_k, terminal_p, static_report = static_sda_contract(modules, model)
    feedback, feedforward, h_values, initial_p = affine_backward_recursion(
        a, b, defect, q, r, terminal_p
    )
    controller = {
        "x_ref": x_ref,
        "u_ref": u_ref,
        "defect_raw": d_raw,
        "defect_wrapped": defect,
        "feedback_k": feedback,
        "feedforward": feedforward,
        "static_default_sda_k": static_k,
        "static_default_sda_p": terminal_p,
        "q": q,
        "r": r,
    }
    report = {
        "reference": reference_report,
        "exact_map": map_report,
        "default_sda": static_report,
        "affine_backward_recursion": {
            "terminal_affine_value_p_is_zero": True,
            "terminal_quadratic_value_is_default_sda_p": True,
            "feedback_shape": list(feedback.shape),
            "feedforward_shape": list(feedforward.shape),
            "feedback_contiguous_float64_sha256": array_sha256(feedback),
            "feedforward_contiguous_float64_sha256": array_sha256(feedforward),
            "h_min": float(np.min(h_values)),
            "h_max": float(np.max(h_values)),
            "all_h_positive": bool(np.all(h_values > 0.0)),
            "initial_p_symmetric_max_abs_delta": max_abs_delta(initial_p, initial_p.T),
            "initial_p_finite": bool(np.all(np.isfinite(initial_p))),
        },
        "controller_policy": "u_raw[k] = u_ref[k] - K[k] @ wrap_error(x[k], x_ref[k]) - kff[k]",
        "optimizer_calls": 0,
    }
    return controller, report


def write_controller_artifact(run_root: Path, controller: dict[str, np.ndarray], report: dict[str, Any]) -> None:
    """Retain the locked controller before the one live tracker rollout."""
    atomic_npz(run_root / "01-affine-controller.npz", **controller)
    atomic_json(
        run_root / "01-affine-controller.json",
        {
            "classification": "N13_B2_AFFINE_CONTROLLER_CONSTRUCTED",
            "manifest_sha256": sha256_file(run_root / "00-source-manifest.json"),
            **report,
        },
    )


def rollout_controls(
    model: Any, initial_state: np.ndarray, controls: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay an exact applied-control vector through the native ZOH boundary."""
    vector = np.asarray(controls)
    if vector.dtype != np.dtype(np.float64) or vector.shape != (N_TICKS,):
        raise ValueError(f"controls must be float64 shape {(N_TICKS,)}, got {vector.dtype}/{vector.shape}")
    command = np.ascontiguousarray(vector)

    def policy(_state: np.ndarray, elapsed_s: float) -> float:
        tick = min(int(round(elapsed_s / DT_S)), N_TICKS - 1)
        return float(command[tick])

    times, states, applied = model.rollout_zoh(
        initial_state, policy, HORIZON_S, DT_S, QUARTER_DT_S
    )
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(states, dtype=np.float64),
        np.asarray(applied, dtype=np.float64),
    )


def quarter_step_audit(model: Any, initial_state: np.ndarray, controls: np.ndarray) -> dict[str, Any]:
    """Audit every actual-control quarter-step without altering the policy path."""
    state = np.asarray(initial_state, dtype=np.float64).copy()
    first_nonfinite: dict[str, Any] | None = None
    first_rail: dict[str, Any] | None = None
    peak = abs(float(state[0]))
    for tick, control in enumerate(np.asarray(controls, dtype=np.float64).reshape(-1)):
        if not np.isfinite(control):
            first_nonfinite = {"tick": tick, "quarter": None, "kind": "control"}
            break
        for quarter in range(4):
            state = np.asarray(model.rk4_step(state, float(control), QUARTER_DT_S), dtype=np.float64)
            if not np.all(np.isfinite(state)):
                first_nonfinite = {"tick": tick, "quarter": quarter + 1, "kind": "state"}
                break
            cart_abs = abs(float(state[0]))
            peak = max(peak, cart_abs)
            if first_rail is None and cart_abs > RAIL_LIMIT_M:
                first_rail = {"tick": tick, "quarter": quarter + 1, "cart_abs_m": cart_abs}
        if first_nonfinite is not None:
            break
    return {
        "cart_peak_m": peak,
        "first_nonfinite": first_nonfinite,
        "first_rail_violation": first_rail,
    }


def first_trace_failure(
    raw: np.ndarray, applied: np.ndarray, states: np.ndarray, quarter: dict[str, Any]
) -> dict[str, Any] | None:
    """Choose the first causal clip, rail, or non-finite event in tick order."""
    events: list[tuple[int, int, str, dict[str, Any]]] = []
    raw_values = np.asarray(raw, dtype=np.float64).reshape(-1)
    applied_values = np.asarray(applied, dtype=np.float64).reshape(-1)
    for tick, value in enumerate(raw_values):
        if not np.isfinite(value):
            events.append((tick, 0, TRACKER_NONFINITE, {"kind": "raw_control"}))
        elif abs(float(value)) > FORCE_BOUND_N:
            events.append((tick, 1, TRACKER_CLIP, {"raw_n": float(value)}))
    if raw_values.shape != (N_TICKS,) or applied_values.shape != (N_TICKS,):
        events.append(
            (
                min(len(raw_values), len(applied_values)),
                0,
                TRACKER_NONFINITE,
                {"kind": "control_trace_shape", "raw_shape": list(raw_values.shape), "applied_shape": list(applied_values.shape)},
            )
        )
    else:
        for tick, (raw_value, applied_value) in enumerate(zip(raw_values, applied_values, strict=True)):
            if not np.isfinite(applied_value):
                events.append((tick, 0, TRACKER_NONFINITE, {"kind": "applied_control"}))
            elif raw_value != applied_value:
                events.append(
                    (tick, 1, TRACKER_CLIP, {"raw_n": float(raw_value), "applied_n": float(applied_value)})
                )
    state_values = np.asarray(states, dtype=np.float64)
    if state_values.shape != (N_STATES, STATE_SIZE):
        events.append((0, 3, TRACKER_NONFINITE, {"kind": "state_trace_shape", "shape": list(state_values.shape)}))
    else:
        for node, state in enumerate(state_values):
            tick = max(0, node - 1)
            if not np.all(np.isfinite(state)):
                events.append((tick, 3, TRACKER_NONFINITE, {"kind": "node_state", "node": node}))
                break
            if abs(float(state[0])) > RAIL_LIMIT_M:
                events.append((tick, 3, TRACKER_RAIL, {"kind": "node", "node": node, "cart_abs_m": abs(float(state[0]))}))
                break
    nonfinite = quarter.get("first_nonfinite")
    if nonfinite is not None:
        events.append((int(nonfinite["tick"]), 2, TRACKER_NONFINITE, {"kind": "quarter", **nonfinite}))
    rail = quarter.get("first_rail_violation")
    if rail is not None:
        events.append((int(rail["tick"]), 2, TRACKER_RAIL, {"kind": "quarter", **rail}))
    if not events:
        return None
    tick, _phase, classification, detail = min(events, key=lambda item: (item[0], item[1]))
    return {"classification": classification, "tick": tick, "detail": detail}


def raw_tracker_rollout(
    model: Any,
    hanging: np.ndarray,
    controller: dict[str, np.ndarray],
    wrap_state_error: Callable[..., np.ndarray],
) -> dict[str, Any]:
    """Run the one unclipped raw affine tracker rollout from exact hanging."""
    raw_log: list[float] = []
    x_ref = controller["x_ref"]
    u_ref = controller["u_ref"]
    feedback = controller["feedback_k"]
    feedforward = controller["feedforward"]

    def policy(state: np.ndarray, elapsed_s: float) -> float:
        tick = int(round(elapsed_s / DT_S))
        if not 0 <= tick < N_TICKS:
            raw = float("nan")
        else:
            error = wrap_state_error(state, x_ref[tick], N_LINKS)
            raw = float(u_ref[tick] - (feedback[tick] @ error).reshape(-1)[0] - feedforward[tick])
        raw_log.append(raw)
        return raw

    exception: str | None = None
    try:
        times, states, applied = model.rollout_zoh(
            hanging, policy, HORIZON_S, DT_S, QUARTER_DT_S
        )
        times = np.asarray(times, dtype=np.float64)
        states = np.asarray(states, dtype=np.float64)
        applied = np.asarray(applied, dtype=np.float64)
    except BaseException as error:
        exception = f"{type(error).__name__}: {error}"
        times = np.empty(0, dtype=np.float64)
        states = np.asarray([hanging], dtype=np.float64)
        applied = np.empty(0, dtype=np.float64)
    raw = np.asarray(raw_log, dtype=np.float64)
    quarter = quarter_step_audit(model, hanging, applied) if applied.size else {
        "cart_peak_m": float("nan"),
        "first_nonfinite": {"tick": 0, "quarter": None, "kind": "unavailable_after_rollout_exception"},
        "first_rail_violation": None,
    }
    first_failure = first_trace_failure(raw, applied, states, quarter)
    exact_shape = bool(
        times.shape == (N_STATES,)
        and states.shape == (N_STATES, STATE_SIZE)
        and raw.shape == (N_TICKS,)
        and applied.shape == (N_TICKS,)
    )
    finite = bool(
        np.all(np.isfinite(times))
        and np.all(np.isfinite(states))
        and np.all(np.isfinite(raw))
        and np.all(np.isfinite(applied))
    )
    gates: dict[str, bool] = {
        "rollout_exception_free": exception is None,
        "exact_shape": exact_shape,
        "exact_time_grid": bool(
            exact_shape and np.array_equal(times, np.arange(N_STATES, dtype=np.float64) * DT_S)
        ),
        "exact_hanging_start": bool(
            exact_shape and max_abs_delta(states[0], hanging) <= 1e-12
        ),
        "finite": finite,
        "raw_equals_applied": bool(exact_shape and finite and arrays_byte_identical(raw, applied)),
        "raw_peak_le_150_n": bool(finite and finite_peak(raw) <= FORCE_BOUND_N),
        "node_cart_le_10_m": bool(
            finite and states.shape == (N_STATES, STATE_SIZE) and finite_peak(states[:, 0]) <= RAIL_LIMIT_M
        ),
        "quarter_cart_le_10_m": bool(
            quarter["first_nonfinite"] is None and quarter["first_rail_violation"] is None
        ),
    }
    replay_times = np.empty(0, dtype=np.float64)
    replay_states = np.empty((0, STATE_SIZE), dtype=np.float64)
    replay_applied = np.empty(0, dtype=np.float64)
    replay_delta = float("inf")
    if all(gates.values()):
        try:
            replay_times, replay_states, replay_applied = rollout_controls(model, hanging, applied)
            replay_delta = max_abs_delta(replay_states, states)
            gates.update(
                {
                    "applied_replay_exact_shape": replay_states.shape == (N_STATES, STATE_SIZE)
                    and replay_times.shape == (N_STATES,)
                    and replay_applied.shape == (N_TICKS,),
                    "applied_replay_exact_time_grid": np.array_equal(
                        replay_times, np.arange(N_STATES, dtype=np.float64) * DT_S
                    ),
                    "applied_replay_controls_byte_identical": arrays_byte_identical(replay_applied, applied),
                    "applied_replay_state_match_le_1e-12": replay_delta <= 1e-12,
                }
            )
        except BaseException as error:
            gates["applied_replay_exception_free"] = False
            exception = exception or f"applied replay {type(error).__name__}: {error}"
    passed = bool(all(gates.values()))
    classification = TRACKER_EXTRACTION_PASS if passed else (
        first_failure["classification"] if first_failure is not None else TRACKER_NONFINITE
    )
    return {
        "classification": classification,
        "passed": passed,
        "exception": exception,
        "gates": gates,
        "first_causal_failure": first_failure,
        "raw_peak_n": finite_peak(raw),
        "applied_peak_n": finite_peak(applied),
        "cart_peak_node_m": finite_peak(states[:, 0]) if states.ndim == 2 and states.shape[1:] == (STATE_SIZE,) else float("nan"),
        "quarter_step_rail_audit": quarter,
        "applied_replay_state_max_abs_delta": replay_delta,
        "t": times,
        "x": states,
        "u_raw": raw,
        "u_applied": applied,
        "replay_t": replay_times,
        "replay_x": replay_states,
        "replay_u_applied": replay_applied,
    }


def without_arrays(value: Any) -> Any:
    """Keep JSON inspectable while immutable NPZs retain numerical traces."""
    if isinstance(value, dict):
        return {
            key: without_arrays(item)
            for key, item in value.items()
            if key not in {"t", "x", "u_raw", "u_applied", "replay_t", "replay_x", "replay_u_applied", "phase"}
        }
    if isinstance(value, list):
        return [without_arrays(item) for item in value]
    return value


def write_tracker_artifact(run_root: Path, tracker: dict[str, Any]) -> None:
    """Retain live-only candidate evidence and its exact applied-control replay."""
    atomic_npz(
        run_root / "02-live-tracker-extraction.npz",
        t=tracker["t"],
        x=tracker["x"],
        u_raw=tracker["u_raw"],
        u_applied=tracker["u_applied"],
        replay_t=tracker["replay_t"],
        replay_x=tracker["replay_x"],
        replay_u_applied=tracker["replay_u_applied"],
    )
    atomic_json(
        run_root / "02-live-tracker-extraction.json",
        {
            "classification": tracker["classification"],
            "live_states_are_only_candidate_evidence": tracker["passed"],
            "reference_states_are_not_candidate_evidence": True,
            **without_arrays(tracker),
        },
    )


def trailing_success_seconds(values: np.ndarray) -> tuple[float, int]:
    """Measure the time span of the trailing in-set state sequence."""
    samples = 0
    for value in np.asarray(values, dtype=bool)[::-1]:
        if not value:
            break
        samples += 1
    return max(0.0, (samples - 1) * DT_S), samples


def static_hold(
    model: Any,
    initial_state: np.ndarray,
    static_k: np.ndarray,
    upright: np.ndarray,
    wrap_state_error: Callable[..., np.ndarray],
    in_success_set: Callable[..., bool],
) -> dict[str, Any]:
    """Run exactly twelve seconds of the unchanged unclipped default SDA law."""
    raw_log: list[float] = []

    def policy(state: np.ndarray, _elapsed_s: float) -> float:
        raw = -float(static_k @ wrap_state_error(state, upright, N_LINKS))
        raw_log.append(raw)
        return raw

    exception: str | None = None
    try:
        times, states, applied = model.rollout_zoh(
            initial_state, policy, HOLD_TICKS * DT_S, DT_S, QUARTER_DT_S
        )
        times = np.asarray(times, dtype=np.float64)
        states = np.asarray(states, dtype=np.float64)
        applied = np.asarray(applied, dtype=np.float64)
    except BaseException as error:
        exception = f"{type(error).__name__}: {error}"
        times = np.empty(0, dtype=np.float64)
        states = np.asarray([initial_state], dtype=np.float64)
        applied = np.empty(0, dtype=np.float64)
    raw = np.asarray(raw_log, dtype=np.float64)
    exact_shape = bool(
        times.shape == (HOLD_TICKS + 1,)
        and states.shape == (HOLD_TICKS + 1, STATE_SIZE)
        and raw.shape == (HOLD_TICKS,)
        and applied.shape == (HOLD_TICKS,)
    )
    finite = bool(
        np.all(np.isfinite(times))
        and np.all(np.isfinite(states))
        and np.all(np.isfinite(raw))
        and np.all(np.isfinite(applied))
    )
    in_set = (
        np.asarray([in_success_set(model, state) for state in states], dtype=bool)
        if exact_shape and finite
        else np.empty(0, dtype=bool)
    )
    trailing_s, trailing_samples = trailing_success_seconds(in_set)
    quarter = quarter_step_audit(model, initial_state, applied) if applied.size else {
        "cart_peak_m": float("nan"),
        "first_nonfinite": {"tick": 0, "quarter": None, "kind": "unavailable_after_hold_exception"},
        "first_rail_violation": None,
    }
    gates = {
        "exact_12_second_shape": exact_shape,
        "exact_time_grid": bool(
            exact_shape and np.array_equal(times, np.arange(HOLD_TICKS + 1, dtype=np.float64) * DT_S)
        ),
        "finite": finite,
        "raw_equals_applied": bool(exact_shape and finite and arrays_byte_identical(raw, applied)),
        "raw_peak_le_150_n": bool(finite and finite_peak(raw) <= FORCE_BOUND_N),
        "node_cart_le_10_m": bool(
            finite and states.shape == (HOLD_TICKS + 1, STATE_SIZE) and finite_peak(states[:, 0]) <= RAIL_LIMIT_M
        ),
        "quarter_cart_le_10_m": bool(
            quarter["first_nonfinite"] is None and quarter["first_rail_violation"] is None
        ),
        "trailing_in_set_ge_5_s": trailing_s >= 5.0,
    }
    return {
        "passed": bool(exception is None and all(gates.values())),
        "exception": exception,
        "gates": gates,
        "raw_peak_n": finite_peak(raw),
        "applied_peak_n": finite_peak(applied),
        "cart_peak_node_m": finite_peak(states[:, 0]) if states.ndim == 2 and states.shape[1:] == (STATE_SIZE,) else float("nan"),
        "trailing_success_s": trailing_s,
        "trailing_success_samples": trailing_samples,
        "quarter_step_rail_audit": quarter,
        "t": times,
        "x": states,
        "u_raw": raw,
        "u_applied": applied,
    }


def capture_scan(
    model: Any,
    live_x: np.ndarray,
    controller: dict[str, np.ndarray],
    upright: np.ndarray,
    wrap_state_error: Callable[..., np.ndarray],
    in_success_set: Callable[..., bool],
) -> dict[str, Any]:
    """Scan actual live states in order and stop at the first passing 12 s hold."""
    static_k = controller["static_default_sda_k"]
    eligible_ticks: list[int] = []
    tested_holds: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for tick, state in enumerate(live_x):
        if not in_success_set(model, state):
            continue
        entry_raw = -float(static_k @ wrap_state_error(state, upright, N_LINKS))
        if not np.isfinite(entry_raw) or abs(entry_raw) > FORCE_BOUND_N:
            continue
        eligible_ticks.append(tick)
        hold = static_hold(model, state, static_k, upright, wrap_state_error, in_success_set)
        tested_holds.append({"tick": tick, "initial_static_raw_n": entry_raw, **without_arrays(hold)})
        if hold["passed"]:
            selected = {"tick": tick, "hold": hold}
            break
    if not eligible_ticks:
        classification = NO_CAPTURE_ENTRY
    elif selected is None:
        classification = CAPTURE_REJECTED
    else:
        classification = CAPTURE_READY
    return {
        "classification": classification,
        "eligible_ticks": eligible_ticks,
        "tested_holds": tested_holds,
        "selected_switch_tick": selected["tick"] if selected is not None else None,
        "selected_hold": without_arrays(selected["hold"]) if selected is not None else None,
        "selected_hold_arrays": selected["hold"] if selected is not None else None,
    }


def write_capture_artifact(run_root: Path, capture: dict[str, Any]) -> None:
    """Retain the ordered scan, including only the earliest successful hold trace."""
    selected = capture["selected_hold_arrays"]
    if selected is None:
        selected_t = np.empty(0, dtype=np.float64)
        selected_x = np.empty((0, STATE_SIZE), dtype=np.float64)
        selected_raw = np.empty(0, dtype=np.float64)
        selected_applied = np.empty(0, dtype=np.float64)
    else:
        selected_t = selected["t"]
        selected_x = selected["x"]
        selected_raw = selected["u_raw"]
        selected_applied = selected["u_applied"]
    atomic_npz(
        run_root / "03-static-capture-scan.npz",
        eligible_ticks=np.asarray(capture["eligible_ticks"], dtype=np.int64),
        selected_switch_tick=np.asarray(
            -1 if capture["selected_switch_tick"] is None else capture["selected_switch_tick"], dtype=np.int64
        ),
        selected_hold_t=selected_t,
        selected_hold_x=selected_x,
        selected_hold_u_raw=selected_raw,
        selected_hold_u_applied=selected_applied,
    )
    atomic_json(
        run_root / "03-static-capture-scan.json",
        {
            "classification": capture["classification"],
            "ordered_live_state_scan": True,
            "static_law": "default-Q mpmath50 SDA K from 01-affine-controller.npz",
            "eligible_ticks": capture["eligible_ticks"],
            "tested_holds": capture["tested_holds"],
            "selected_switch_tick": capture["selected_switch_tick"],
            "selected_hold": capture["selected_hold"],
        },
    )


def load_controller_artifact(run_root: Path) -> dict[str, np.ndarray]:
    """Load and revalidate the immutable fixed controller for the fresh proof."""
    try:
        with np.load(run_root / "01-affine-controller.npz", allow_pickle=False) as archive:
            controller = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, KeyError, ValueError) as error:
        raise B2Failure(CAPTURE_REJECTED, f"cannot load controller artifact: {type(error).__name__}: {error}") from error
    expected_shapes = {
        "x_ref": (N_STATES, STATE_SIZE),
        "u_ref": (N_TICKS,),
        "defect_raw": (N_TICKS, STATE_SIZE),
        "defect_wrapped": (N_TICKS, STATE_SIZE),
        "feedback_k": (N_TICKS, 1, STATE_SIZE),
        "feedforward": (N_TICKS,),
        "static_default_sda_k": (STATE_SIZE,),
        "static_default_sda_p": (STATE_SIZE, STATE_SIZE),
        "q": (STATE_SIZE, STATE_SIZE),
        "r": (1, 1),
    }
    for name, shape in expected_shapes.items():
        value = controller.get(name)
        if value is None or value.dtype != np.dtype(np.float64) or value.shape != shape or not np.all(np.isfinite(value)):
            raise B2Failure(CAPTURE_REJECTED, f"controller artifact contract failed for {name}")
    if (
        array_sha256(controller["static_default_sda_k"]) != STATIC_K_SHA256
        or array_sha256(controller["static_default_sda_p"]) != STATIC_P_SHA256
    ):
        raise B2Failure(CAPTURE_REJECTED, "controller artifact SDA hashes drifted")
    if max_abs_delta(controller["defect_raw"], controller["defect_wrapped"]) > 1e-12:
        raise B2Failure(CAPTURE_REJECTED, "controller artifact no longer satisfies chart gate")
    return {name: np.ascontiguousarray(value) for name, value in controller.items()}


def load_live_states(run_root: Path) -> np.ndarray:
    """Load the one accepted live tracker state trace for fresh-proof comparison."""
    try:
        with np.load(run_root / "02-live-tracker-extraction.npz", allow_pickle=False) as archive:
            live_x = np.asarray(archive["x"])
    except (OSError, KeyError, ValueError) as error:
        raise B2Failure(CAPTURE_REJECTED, f"cannot load live tracker evidence: {type(error).__name__}: {error}") from error
    if (
        live_x.dtype != np.dtype(np.float64)
        or live_x.shape != (N_STATES, STATE_SIZE)
        or not np.all(np.isfinite(live_x))
    ):
        raise B2Failure(CAPTURE_REJECTED, "live tracker evidence shape/dtype/finite contract failed")
    return np.ascontiguousarray(live_x)


def composed_proof(
    model: Any,
    hanging: np.ndarray,
    upright: np.ndarray,
    controller: dict[str, np.ndarray],
    extracted_live_x: np.ndarray,
    switch_tick: int,
    wrap_state_error: Callable[..., np.ndarray],
    in_success_set: Callable[..., bool],
) -> dict[str, Any]:
    """Run the one fresh tracker-to-default-SDA composition from exact hanging."""
    if not 0 <= switch_tick <= N_TICKS:
        raise B2Failure(CAPTURE_REJECTED, f"invalid selected switch tick: {switch_tick}")
    raw_log: list[float] = []
    phase_log: list[str] = []
    x_ref = controller["x_ref"]
    u_ref = controller["u_ref"]
    feedback = controller["feedback_k"]
    feedforward = controller["feedforward"]
    static_k = controller["static_default_sda_k"]

    def policy(state: np.ndarray, elapsed_s: float) -> float:
        tick = int(round(elapsed_s / DT_S))
        if tick < switch_tick:
            raw = float(
                u_ref[tick]
                - (
                    feedback[tick]
                    @ wrap_state_error(state, x_ref[tick], N_LINKS)
                ).reshape(-1)[0]
                - feedforward[tick]
            )
            phase = "affine_defect_tracker"
        else:
            raw = -float(static_k @ wrap_state_error(state, upright, N_LINKS))
            phase = "static_default_sda"
        raw_log.append(raw)
        phase_log.append(phase)
        return raw

    total_ticks = switch_tick + HOLD_TICKS
    exception: str | None = None
    try:
        times, states, applied = model.rollout_zoh(
            hanging, policy, total_ticks * DT_S, DT_S, QUARTER_DT_S
        )
        times = np.asarray(times, dtype=np.float64)
        states = np.asarray(states, dtype=np.float64)
        applied = np.asarray(applied, dtype=np.float64)
    except BaseException as error:
        exception = f"{type(error).__name__}: {error}"
        times = np.empty(0, dtype=np.float64)
        states = np.asarray([hanging], dtype=np.float64)
        applied = np.empty(0, dtype=np.float64)
    raw = np.asarray(raw_log, dtype=np.float64)
    phase = np.asarray(phase_log, dtype="U24")
    exact_shape = bool(
        times.shape == (total_ticks + 1,)
        and states.shape == (total_ticks + 1, STATE_SIZE)
        and raw.shape == (total_ticks,)
        and applied.shape == (total_ticks,)
        and phase.shape == (total_ticks,)
    )
    finite = bool(
        np.all(np.isfinite(times))
        and np.all(np.isfinite(states))
        and np.all(np.isfinite(raw))
        and np.all(np.isfinite(applied))
    )
    quarter = quarter_step_audit(model, hanging, applied) if applied.size else {
        "cart_peak_m": float("nan"),
        "first_nonfinite": {"tick": 0, "quarter": None, "kind": "unavailable_after_proof_exception"},
        "first_rail_violation": None,
    }
    prefix_complete = bool(exact_shape and extracted_live_x.shape == (N_STATES, STATE_SIZE))
    prefix_delta = (
        max_abs_delta(states[: switch_tick + 1], extracted_live_x[: switch_tick + 1])
        if prefix_complete
        else float("inf")
    )
    switch_in_set = bool(prefix_complete and finite and in_success_set(model, states[switch_tick]))
    post_switch_set = (
        np.asarray([in_success_set(model, state) for state in states[switch_tick:]], dtype=bool)
        if exact_shape and finite
        else np.empty(0, dtype=bool)
    )
    trailing_s, trailing_samples = trailing_success_seconds(post_switch_set)
    gates = {
        "rollout_exception_free": exception is None,
        "exact_shape": exact_shape,
        "exact_time_grid": bool(
            exact_shape and np.array_equal(times, np.arange(total_ticks + 1, dtype=np.float64) * DT_S)
        ),
        "exact_hanging_start": bool(exact_shape and max_abs_delta(states[0], hanging) <= 1e-12),
        "finite": finite,
        "raw_equals_applied": bool(exact_shape and finite and arrays_byte_identical(raw, applied)),
        "raw_peak_le_150_n": bool(finite and finite_peak(raw) <= FORCE_BOUND_N),
        "node_cart_le_10_m": bool(
            finite and states.shape == (total_ticks + 1, STATE_SIZE) and finite_peak(states[:, 0]) <= RAIL_LIMIT_M
        ),
        "quarter_cart_le_10_m": bool(
            quarter["first_nonfinite"] is None and quarter["first_rail_violation"] is None
        ),
        "prefix_agrees_le_1e-12": prefix_delta <= 1e-12,
        "switch_in_instantaneous_success_set": switch_in_set,
        "trailing_in_set_ge_5_s": trailing_s >= 5.0,
    }
    return {
        "passed": bool(all(gates.values())),
        "exception": exception,
        "switch_tick": switch_tick,
        "switch_time_s": switch_tick * DT_S,
        "stored_live_state_injected": False,
        "tracker_policy_recomputed_live": True,
        "gates": gates,
        "prefix_max_abs_delta": prefix_delta,
        "raw_peak_n": finite_peak(raw),
        "applied_peak_n": finite_peak(applied),
        "cart_peak_node_m": finite_peak(states[:, 0]) if states.ndim == 2 and states.shape[1:] == (STATE_SIZE,) else float("nan"),
        "trailing_success_s": trailing_s,
        "trailing_success_samples": trailing_samples,
        "quarter_step_rail_audit": quarter,
        "t": times,
        "x": states,
        "u_raw": raw,
        "u_applied": applied,
        "phase": phase,
    }


def write_proof_artifact(run_root: Path, proof: dict[str, Any]) -> None:
    """Retain the single fresh composition regardless of its classification."""
    atomic_npz(
        run_root / "04-fresh-composed-proof.npz",
        t=proof["t"],
        x=proof["x"],
        u_raw=proof["u_raw"],
        u_applied=proof["u_applied"],
        phase=proof["phase"],
    )
    atomic_json(
        run_root / "04-fresh-composed-proof.json",
        {
            "classification": ONE_RUN_PASS if proof["passed"] else CAPTURE_REJECTED,
            "only_this_artifact_can_authorize_n13_one_run_pass": True,
            **without_arrays(proof),
        },
    )


def tracker_worker(run_root: Path) -> int:
    """Construct once, run one live tracker, then perform the ordered capture scan."""
    try:
        validate_source_manifest(run_root)
        modules = load_approved_sources()
        model, hanging, upright = make_n13_model(modules)
        controller, controller_report = build_affine_controller(modules, model, hanging)
        write_controller_artifact(run_root, controller, controller_report)
        tracker = raw_tracker_rollout(model, hanging, controller, modules["wrap_state_error"])
        write_tracker_artifact(run_root, tracker)
        if not tracker["passed"]:
            capture = {
                "classification": tracker["classification"],
                "eligible_ticks": [],
                "tested_holds": [],
                "selected_switch_tick": None,
                "selected_hold": None,
                "selected_hold_arrays": None,
            }
            write_capture_artifact(run_root, capture)
            print(CHILD_RESULT_PREFIX + json.dumps({"classification": tracker["classification"]}))
            return 0
        capture = capture_scan(
            model,
            tracker["x"],
            controller,
            upright,
            modules["wrap_state_error"],
            modules["in_success_set"],
        )
        write_capture_artifact(run_root, capture)
        print(
            CHILD_RESULT_PREFIX
            + json.dumps(
                {
                    "classification": capture["classification"],
                    "selected_switch_tick": capture["selected_switch_tick"],
                }
            )
        )
        return 0
    except B2Failure as error:
        print(CHILD_RESULT_PREFIX + json.dumps({"classification": error.classification, "detail": str(error)}))
        return 1
    except BaseException as error:
        print(
            CHILD_RESULT_PREFIX
            + json.dumps(
                {
                    "classification": SOURCE_CONTRACT_FAILURE,
                    "detail": f"{type(error).__name__}: {error}",
                }
            )
        )
        return 1


def proof_worker(run_root: Path, switch_tick: int) -> int:
    """Run one fresh proof child after the manifest and immutable inputs recheck."""
    try:
        validate_source_manifest(run_root)
        modules = load_approved_sources()
        controller = load_controller_artifact(run_root)
        extracted_live_x = load_live_states(run_root)
        model, hanging, upright = make_n13_model(modules)
        proof = composed_proof(
            model,
            hanging,
            upright,
            controller,
            extracted_live_x,
            switch_tick,
            modules["wrap_state_error"],
            modules["in_success_set"],
        )
        write_proof_artifact(run_root, proof)
        classification = ONE_RUN_PASS if proof["passed"] else CAPTURE_REJECTED
        print(CHILD_RESULT_PREFIX + json.dumps({"classification": classification}))
        return 0
    except B2Failure as error:
        print(CHILD_RESULT_PREFIX + json.dumps({"classification": error.classification, "detail": str(error)}))
        return 1
    except BaseException as error:
        print(
            CHILD_RESULT_PREFIX
            + json.dumps(
                {
                    "classification": CAPTURE_REJECTED,
                    "detail": f"{type(error).__name__}: {error}",
                }
            )
        )
        return 1


def child_payload(stdout: str) -> dict[str, Any] | None:
    """Read only the explicit child result line, not incidental output."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(CHILD_RESULT_PREFIX):
            try:
                return json.loads(line[len(CHILD_RESULT_PREFIX) :])
            except json.JSONDecodeError:
                return None
    return None


def invoke_child(run_root: Path, mode: str, *extra: str) -> subprocess.CompletedProcess[str]:
    """Launch an isolated child so its manifest recheck precedes local import."""
    command = [sys.executable, "-I", str(DRIVER_PATH), mode, str(run_root), *extra]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(command, text=True, capture_output=True, check=False, env=environment)


def write_final_classification(
    run_root: Path,
    classification: str,
    tracker_payload: dict[str, Any] | None,
    proof_payload: dict[str, Any] | None,
) -> None:
    """Write the parent-level immutable classification without rerunning B2."""
    selected = (tracker_payload or {}).get("selected_switch_tick")
    atomic_npz(
        run_root / "05-b2-classification.npz",
        selected_switch_tick=np.asarray(-1 if selected is None else selected, dtype=np.int64),
    )
    atomic_json(
        run_root / "05-b2-classification.json",
        {
            "classification": classification,
            "tracker_child": tracker_payload,
            "proof_child": proof_payload,
            "optimizer_calls": 0,
            "second_tracker_run_authorized": False,
            "retry_authorized": False,
            "physical_rollout_performed_by_parent": False,
        },
    )


def parent_main() -> int:
    """Coordinate the one tracker classifier and, only if selected, one proof."""
    run_root = make_run_root()
    manifest = build_source_manifest()
    atomic_json(run_root / "00-source-manifest.json", manifest)
    try:
        assert_initial_source_lock(manifest)
        validate_source_manifest(run_root)
    except B2Failure as error:
        write_final_classification(run_root, error.classification, None, None)
        print(f"{error.classification}: {error}")
        return 1

    tracker_child = invoke_child(run_root, "--tracker-child")
    tracker_payload = child_payload(tracker_child.stdout)
    if tracker_child.returncode != 0 or tracker_payload is None:
        classification = str((tracker_payload or {}).get("classification", SOURCE_CONTRACT_FAILURE))
        write_final_classification(run_root, classification, tracker_payload, None)
        print(f"{classification}: tracker child failed")
        return 1
    classification = str(tracker_payload.get("classification", TRACKER_NONFINITE))
    if classification != CAPTURE_READY:
        write_final_classification(run_root, classification, tracker_payload, None)
        print(classification)
        return 0
    selected = tracker_payload.get("selected_switch_tick")
    if not isinstance(selected, int):
        write_final_classification(run_root, CAPTURE_REJECTED, tracker_payload, None)
        print(f"{CAPTURE_REJECTED}: capture child omitted an integer switch tick")
        return 1

    proof_child = invoke_child(run_root, "--proof-child", str(selected))
    proof_payload = child_payload(proof_child.stdout)
    if proof_child.returncode != 0 or proof_payload is None:
        classification = str((proof_payload or {}).get("classification", CAPTURE_REJECTED))
        write_final_classification(run_root, classification, tracker_payload, proof_payload)
        print(f"{classification}: proof child failed")
        return 1
    classification = str(proof_payload.get("classification", CAPTURE_REJECTED))
    write_final_classification(run_root, classification, tracker_payload, proof_payload)
    print(classification)
    return 0 if classification == ONE_RUN_PASS else 1


def local_wrap_error(x: np.ndarray, reference: np.ndarray, n_links: int) -> np.ndarray:
    """Mirror the production wrapped coordinate for bounded no-import tests."""
    error = np.asarray(x, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    error = error.copy()
    error[1 : 1 + n_links] = (error[1 : 1 + n_links] + np.pi) % (2.0 * np.pi) - np.pi
    return error


def self_test() -> int:
    """Exercise pure contracts without N13 source imports, SDA construction, or rollout."""
    nx, horizon = 2, 3
    fx_flat = np.arange(nx * nx * horizon, dtype=np.float64).reshape(nx, nx * horizon)
    fu_flat = np.arange(nx * horizon, dtype=np.float64).reshape(nx, horizon)
    a, b = reshape_exact_map_jacobians(fx_flat, fu_flat, nx, horizon)
    assert np.array_equal(a, fx_flat.reshape(nx, horizon, nx).transpose(1, 0, 2))
    assert np.array_equal(b, fu_flat.T.reshape(horizon, nx, 1))

    reference_state = np.array([1.0, 0.0, 3.0, 0.0], dtype=np.float64)
    reference_successor = np.array([1.0, 0.0, 3.0, 0.0], dtype=np.float64)
    mapped_reference = np.array([1.3, 0.2, 2.6, 0.1], dtype=np.float64)
    wrapped_defect = local_wrap_error(mapped_reference, reference_successor, 1)
    assert np.allclose(wrapped_defect, np.array([0.3, 0.2, -0.4, 0.1]))
    assert np.any(wrapped_defect != 0.0)
    error_now = local_wrap_error(
        np.array([1.05, 0.04, 2.98, -0.03]), reference_state, 1
    )
    mapped_live = mapped_reference + error_now
    error_next = local_wrap_error(mapped_live, reference_successor, 1)
    assert np.allclose(error_next, error_now + wrapped_defect)
    chart_crossing_raw = np.array([0.0, 2.0 * np.pi - 0.2, 0.0, 0.0])
    chart_crossing_wrapped = local_wrap_error(chart_crossing_raw, np.zeros(4), 1)
    assert max_abs_delta(chart_crossing_raw, chart_crossing_wrapped) > 1e-12

    a_one = np.array([[[2.0]]])
    b_one = np.array([[[3.0]]])
    d_one = np.array([[13.0]])
    q_one = np.array([[5.0]])
    r_one = np.array([[7.0]])
    terminal_one = np.array([[11.0]])
    feedback, feedforward, h_values, initial_p = affine_backward_recursion(
        a_one, b_one, d_one, q_one, r_one, terminal_one
    )
    assert np.allclose(h_values, [106.0])
    assert np.allclose(feedback, [[[66.0 / 106.0]]])
    assert np.allclose(feedforward, [429.0 / 106.0])
    assert np.allclose(initial_p, [[5.0 + 44.0 - (66.0 * 66.0) / 106.0]])
    assert np.allclose(-feedback[0, 0, 0] * 0.0 - feedforward[0], [-429.0 / 106.0])
    try:
        affine_backward_recursion(a_one, b_one, d_one, q_one, np.array([[-100.0]]), terminal_one)
    except B2Failure as error:
        assert error.classification == AFFINE_RECURSION_REJECTED
    else:
        raise AssertionError("non-positive affine H was accepted")

    clean_quarter = {"first_nonfinite": None, "first_rail_violation": None}
    clip = first_trace_failure(
        np.array([0.0, 151.0] + [0.0] * (N_TICKS - 2)),
        np.array([0.0, 150.0] + [0.0] * (N_TICKS - 2)),
        np.zeros((N_STATES, STATE_SIZE), dtype=np.float64),
        clean_quarter,
    )
    assert clip == {
        "classification": TRACKER_CLIP,
        "tick": 1,
        "detail": {"raw_n": 151.0},
    }
    rail = first_trace_failure(
        np.zeros(N_TICKS, dtype=np.float64),
        np.zeros(N_TICKS, dtype=np.float64),
        np.zeros((N_STATES, STATE_SIZE), dtype=np.float64),
        {"first_nonfinite": None, "first_rail_violation": {"tick": 2, "quarter": 1, "cart_abs_m": 10.1}},
    )
    assert rail is not None and rail["classification"] == TRACKER_RAIL and rail["tick"] == 2
    nonfinite_raw = np.zeros(N_TICKS, dtype=np.float64)
    nonfinite_raw[3] = np.nan
    nonfinite = first_trace_failure(
        nonfinite_raw,
        np.zeros(N_TICKS, dtype=np.float64),
        np.zeros((N_STATES, STATE_SIZE), dtype=np.float64),
        clean_quarter,
    )
    assert nonfinite is not None and nonfinite["classification"] == TRACKER_NONFINITE and nonfinite["tick"] == 3

    payload = np.array([1.0, -2.0], dtype=np.float64)
    assert arrays_byte_identical(payload, payload.copy())
    assert array_sha256(payload) == hashlib.sha256(payload.tobytes()).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        atomic_json(root / "test.json", {"payload": payload})
        atomic_npz(root / "test.npz", payload=payload)
        with np.load(root / "test.npz", allow_pickle=False) as archive:
            assert np.array_equal(archive["payload"], payload)
        try:
            atomic_json(root / "test.json", {"replacement": True})
        except FileExistsError:
            pass
        else:
            raise AssertionError("immutable JSON writer allowed replacement")
    print("N13_B2_SELF_TEST_PASS")
    return 0


def parse_args() -> argparse.Namespace:
    """Keep child modes internal so default mode remains the single B2 path."""
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--tracker-child", type=Path, metavar="RUN_ROOT")
    group.add_argument("--proof-child", nargs=2, metavar=("RUN_ROOT", "SWITCH_TICK"))
    return parser.parse_args()


def main() -> int:
    """Route default B2 mode or an explicit isolated child mode."""
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.tracker_child is not None:
        return tracker_worker(args.tracker_child.resolve())
    if args.proof_child is not None:
        run_root_text, switch_tick_text = args.proof_child
        try:
            switch_tick = int(switch_tick_text)
        except ValueError:
            print(CHILD_RESULT_PREFIX + json.dumps({"classification": CAPTURE_REJECTED, "detail": "invalid switch tick"}))
            return 1
        return proof_worker(Path(run_root_text).resolve(), switch_tick)
    return parent_main()


if __name__ == "__main__":
    raise SystemExit(main())
