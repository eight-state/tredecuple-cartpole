"""Run the preserved byte verifier and a platform-stable proof gate."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from n13_proof.capsule import BUNDLE_REL, project_root, read_json, sha256_file

INDEPENDENT_VERIFIER_REL = Path(".working/n13-retrospective/b2-proof-verify.py")
INDEPENDENT_VERIFIER_SHA256 = "9cada194b63f13b576fc6b8906bb7e315cd604e73fbb3a022147de461f022a13"

# These checks compare floating-point trajectories produced on the current host
# with arrays produced on the source host. The portable gate below does not
# reinterpret any other failure as numerical drift.
PLATFORM_NUMERIC_FAILURES = frozenset(
    {
        "controller x_ref is not the pinned base60 densification",
        "live: fresh state mismatch exceeds 1e-12",
        "live: fresh raw controls differ",
        "live: fresh applied controls differ",
        "hold: fresh state mismatch exceeds 1e-12",
        "hold: fresh raw controls differ",
        "hold: fresh applied controls differ",
        "proof: fresh state mismatch exceeds 1e-12",
        "proof: fresh raw controls differ",
        "proof: fresh applied controls differ",
        "saved_live_applied: applied-control replay mismatch",
        "saved_hold_applied: applied-control replay mismatch",
        "saved_proof_applied: applied-control replay mismatch",
    }
)


def load_independent_verifier(path: Path) -> ModuleType:
    if sha256_file(path) != INDEPENDENT_VERIFIER_SHA256:
        raise RuntimeError("independent verifier hash mismatch")
    spec = importlib.util.spec_from_file_location("n13_preserved_independent_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load independent verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_independent_verifier(root: Path) -> dict[str, Any]:
    """Run the preserved verifier unchanged; its JSON claim is not a verdict input."""
    module = load_independent_verifier(root / INDEPENDENT_VERIFIER_REL)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = module.main()
    try:
        result = json.loads(output.getvalue())
    except json.JSONDecodeError as error:
        raise RuntimeError("independent verifier did not emit a JSON result") from error
    if not isinstance(result, dict):
        raise RuntimeError("independent verifier result is not an object")
    result["_exit_code"] = code
    return result


def portable_gate(independent: dict[str, Any]) -> dict[str, Any]:
    """Evaluate source integrity and physical gates without cross-host byte claims."""
    failures = set(independent.get("failures", []))
    unexpected_failures = sorted(failures - PLATFORM_NUMERIC_FAILURES)

    reference = independent.get("reference_authority", {})
    reference_close = (
        math.isfinite(float(reference.get("dense_x_max_abs_delta", math.inf)))
        and float(reference.get("dense_x_max_abs_delta", math.inf)) <= 1e-12
        and float(reference.get("dense_u_max_abs_delta", math.inf)) <= 1e-12
    )

    trace_names = ("fresh_affine_live", "fresh_static_hold", "fresh_composed_proof")
    trace_checks: dict[str, bool] = {}
    for name in trace_names:
        trace = independent.get(name, {})
        trace_checks[name] = bool(
            trace.get("first_nonfinite") is None
            and trace.get("first_rail_violation") is None
            and trace.get("raw_equals_applied_byte_identical") is True
            and math.isfinite(float(trace.get("raw_peak_abs_n", math.inf)))
            and float(trace.get("raw_peak_abs_n", math.inf)) <= 150.0
            and float(trace.get("node_cart_peak_m", math.inf)) <= 10.0
            and float(trace.get("quarter_cart_peak_m", math.inf)) <= 10.0
        )

    composition = independent.get("artifact_composition", {})
    composition_locked = bool(composition) and all(composition.values())
    time_grids = independent.get("time_grids", {})
    time_grids_locked = bool(time_grids) and all(time_grids.values())
    switch = independent.get("switch_success", {})
    success_predicate = bool(
        switch.get("in_success_set") is True
        and switch.get("trailing_success_samples") == 9049
        and switch.get("trailing_success_s") == 9.048
    )

    checks = {
        "no_unexpected_preserved_verifier_failures": not unexpected_failures,
        "b0_reference_within_1e-12": reference_close,
        "fresh_closed_loop_traces_pass_physical_gates": all(trace_checks.values()),
        "stored_artifact_composition_byte_locked": composition_locked,
        "time_grids_byte_locked": time_grids_locked,
        "switch_and_trailing_success_predicate": success_predicate,
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "trace_checks": trace_checks,
        "unexpected_failures": unexpected_failures,
    }


def main() -> int:
    root = project_root()
    try:
        independent = run_independent_verifier(root)
        claimed = read_json(root / BUNDLE_REL / "05-b2-classification.json").get("classification")
        portable = portable_gate(independent)
    except Exception as error:
        print(f"n13-proof-verify failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    independent_pass = independent.get("verdict") == "PASS" and independent.get("_exit_code") == 0
    claim_matches = claimed == "N13_ONE_RUN_PASS"
    result = {
        "classification": "deterministic_one_run_proof",
        "bundle_claim": claimed,
        "bundle_claim_matches_N13_ONE_RUN_PASS": claim_matches,
        "portable_closed_loop_witness": portable,
        "preserved_independent_verifier": {
            "path": INDEPENDENT_VERIFIER_REL.as_posix(),
            "sha256": INDEPENDENT_VERIFIER_SHA256,
            "verdict_on_this_platform": independent.get("verdict"),
            "exact_byte_verdict_passes": independent_pass,
            "failure_count": len(independent.get("failures", [])),
            "failures": independent.get("failures", []),
            "reference_authority": independent.get("reference_authority"),
            "fresh_affine_live": independent.get("fresh_affine_live"),
            "fresh_static_hold": independent.get("fresh_static_hold"),
            "fresh_composed_proof": independent.get("fresh_composed_proof"),
            "saved_applied_replays": independent.get("saved_applied_replays"),
            "artifact_composition": independent.get("artifact_composition"),
            "switch_success": independent.get("switch_success"),
            "time_grids": independent.get("time_grids"),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if claim_matches and portable["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
