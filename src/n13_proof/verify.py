"""Entrypoint that runs the preserved independent N13 B2 verifier unchanged."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from n13_proof.capsule import BUNDLE_REL, project_root, read_json, sha256_file

INDEPENDENT_VERIFIER_REL = Path(".working/n13-retrospective/b2-proof-verify.py")
INDEPENDENT_VERIFIER_SHA256 = "9cada194b63f13b576fc6b8906bb7e315cd604e73fbb3a022147de461f022a13"


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
    """Run its original logic; the B2 JSON claim is not a verdict input."""
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


def main() -> int:
    root = project_root()
    try:
        independent = run_independent_verifier(root)
        claimed = read_json(root / BUNDLE_REL / "05-b2-classification.json").get("classification")
    except Exception as error:
        print(f"n13-proof-verify failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    independent_pass = independent.get("verdict") == "PASS" and independent.get("_exit_code") == 0
    comparison = {
        "bundle_claim": claimed,
        "bundle_claim_matches_N13_ONE_RUN_PASS": claimed == "N13_ONE_RUN_PASS",
        "independent_verdict": independent.get("verdict"),
        "independent_verdict_matches_PASS": independent_pass,
    }
    result = {
        "classification": "deterministic_one_run_proof",
        "comparison": comparison,
        "independent_verifier": {
            "path": INDEPENDENT_VERIFIER_REL.as_posix(),
            "sha256": INDEPENDENT_VERIFIER_SHA256,
            "fresh_composed_state_byte_identical": independent.get("fresh_composed_proof", {}).get(
                "state_byte_identical"
            ),
            "fresh_composed_phase_byte_identical": independent.get("fresh_composed_proof", {}).get(
                "phase_byte_identical"
            ),
            "failure_count": len(independent.get("failures", [])),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if comparison["bundle_claim_matches_N13_ONE_RUN_PASS"] and independent_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
