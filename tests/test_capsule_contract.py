from __future__ import annotations

import hashlib
import json
from pathlib import Path

from n13_proof.capsule import EXCLUSIONS, project_root
from n13_proof.verify import (
    INDEPENDENT_VERIFIER_REL,
    INDEPENDENT_VERIFIER_SHA256,
    portable_gate,
)


def sha256_file(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def test_copy_inventory_rehashes_every_copied_input() -> None:
    root = project_root()
    inventory = json.loads((root / "evidence/copy-manifest.json").read_text(encoding="utf-8"))
    files = inventory["files"]
    assert inventory["copied_file_count"] == len(files) == 27
    assert inventory["copied_total_bytes"] == sum(item["bytes"] for item in files)
    for item in files:
        path = root / item["path"]
        assert path.is_file(), item["path"]
        assert path.stat().st_size == item["bytes"], item["path"]
        assert sha256_file(path) == item["sha256"], item["path"]


def test_scope_is_exactly_the_stated_one_run_boundary() -> None:
    assert EXCLUSIONS == (
        "perturbation robustness",
        "release seed gates",
        "72/72",
        "promotion",
        "statistical robustness",
        "hardware",
    )


def test_demo_loads_the_reference_but_never_the_saved_b2_rollout() -> None:
    root = project_root()
    demo_source = (root / "src/n13_proof/demo.py").read_text(encoding="utf-8")
    capsule_source = (root / "src/n13_proof/capsule.py").read_text(encoding="utf-8")
    assert "04-fresh-composed-proof" not in demo_source
    assert "04-fresh-composed-proof" not in capsule_source
    assert '"saved_tracking_reference_loaded": True' in capsule_source
    assert '"saved_b2_rollout_trace_loaded": False' in capsule_source


def test_preserved_independent_verifier_is_byte_locked() -> None:
    path = project_root() / INDEPENDENT_VERIFIER_REL
    assert sha256_file(path) == INDEPENDENT_VERIFIER_SHA256


def test_portable_gate_rejects_failures_outside_the_numeric_allowlist() -> None:
    trace = {
        "first_nonfinite": None,
        "first_rail_violation": None,
        "raw_equals_applied_byte_identical": True,
        "raw_peak_abs_n": 40.0,
        "node_cart_peak_m": 7.0,
        "quarter_cart_peak_m": 7.0,
    }
    result = {
        "verdict": "FAIL",
        "_exit_code": 1,
        "failures": ["plant specification drift"],
        "reference_authority": {
            "dense_x_max_abs_delta": 1e-14,
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
    portable = portable_gate(result, {"passed": True})
    assert portable["verdict"] == "FAIL"
    assert portable["unexpected_failures"] == ["plant specification drift"]
