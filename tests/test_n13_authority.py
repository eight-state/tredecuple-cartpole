from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from n13_proof.capsule import (
    B2_FIXED_FILE_SHA256,
    B2_SOURCE_SHA256,
    AuthorityError,
    project_root,
    validate_b2_authority,
)
from n13_proof.verify import portable_gate


def copy_fixed_authority(tmp_path: Path) -> Path:
    source_root = project_root()
    for relative in {**B2_SOURCE_SHA256, **B2_FIXED_FILE_SHA256}:
        source = source_root / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return tmp_path


def passing_preserved_result() -> dict[str, object]:
    trace = {
        "first_nonfinite": None,
        "first_rail_violation": None,
        "raw_equals_applied_byte_identical": True,
        "raw_peak_abs_n": 40.0,
        "node_cart_peak_m": 7.0,
        "quarter_cart_peak_m": 7.0,
    }
    return {
        "verdict": "PASS",
        "_exit_code": 0,
        "failures": [],
        "reference_authority": {
            "dense_x_max_abs_delta": 0.0,
            "dense_u_max_abs_delta": 0.0,
        },
        "fresh_affine_live": trace,
        "fresh_static_hold": trace,
        "fresh_composed_proof": trace,
        "artifact_composition": {"locked": True},
        "time_grids": {"locked": True},
        "switch_success": {
            "in_success_set": True,
            "trailing_success_samples": 9049,
            "trailing_success_s": 9.048,
        },
    }


def test_fixed_b2_authority_passes_for_release_bytes() -> None:
    authority = validate_b2_authority()
    assert authority["passed"] is True
    assert set(authority["files"]) == set(B2_SOURCE_SHA256) | set(B2_FIXED_FILE_SHA256)
    assert all(record["passed"] for record in authority["files"].values())
    assert all(record["passed"] for record in authority["controller_payloads"].values())


@pytest.mark.parametrize(
    "relative",
    [
        "src/cartpole_race/__init__.py",
        "src/cartpole_race/tvlqr.py",
        ".working/n13/n13_b2_affine_defect_tracker.py",
        (
            ".working/n13/n13-b2-affine-defect-tracker-"
            "20260713T053956542276Z-278512-00/00-source-manifest.json"
        ),
    ],
)
def test_fixed_b2_authority_rejects_source_or_manifest_drift(
    tmp_path: Path, relative: str
) -> None:
    root = copy_fixed_authority(tmp_path)
    path = root / relative
    path.write_bytes(path.read_bytes() + b"\n# integrity test change\n")
    with pytest.raises(AuthorityError, match="fixed B2 authority hash mismatch"):
        validate_b2_authority(root)


def test_fixed_b2_authority_rejects_controller_drift(tmp_path: Path) -> None:
    root = copy_fixed_authority(tmp_path)
    relative = next(
        value for value in B2_FIXED_FILE_SHA256 if value.endswith("01-affine-controller.npz")
    )
    path = root / relative
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["feedforward"] = arrays["feedforward"].copy()
    arrays["feedforward"][100] += 1e-6
    np.savez(path, **arrays)
    with pytest.raises(AuthorityError, match="fixed B2 authority hash mismatch"):
        validate_b2_authority(root)


def test_portable_gate_accepts_consistent_exact_byte_result() -> None:
    result = portable_gate(passing_preserved_result(), {"passed": True})
    assert result["verdict"] == "PASS"
    assert result["mode"] == "exact_byte_pass"


def test_portable_gate_accepts_only_coherent_numeric_drift() -> None:
    preserved = passing_preserved_result()
    preserved.update(
        {
            "verdict": "FAIL",
            "_exit_code": 1,
            "failures": ["live: fresh state mismatch exceeds 1e-12"],
        }
    )
    result = portable_gate(preserved, {"passed": True})
    assert result["verdict"] == "PASS"
    assert result["mode"] == "numeric_drift_only"


@pytest.mark.parametrize(
    ("authority", "verdict", "exit_code", "failures"),
    [
        ({"passed": False}, "PASS", 0, []),
        ({"passed": True}, "PASS", 0, ["live: fresh state mismatch exceeds 1e-12"]),
        ({"passed": True}, "FAIL", 1, []),
        ({"passed": True}, "PASS", 0, "not a list"),
    ],
)
def test_portable_gate_rejects_integrity_or_result_incoherence(
    authority: dict[str, bool], verdict: str, exit_code: int, failures: object
) -> None:
    preserved = passing_preserved_result()
    preserved.update({"verdict": verdict, "_exit_code": exit_code, "failures": failures})
    result = portable_gate(preserved, authority)
    assert result["verdict"] == "FAIL"
    assert result["mode"] == "integrity_or_semantic_failure"
