"""Public fresh-run demonstration and GIF renderer for the N13 capsule."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from n13_proof.capsule import EXCLUSIONS, N_LINKS, fresh_composed_rollout, prepare_run, sha256_file


def _screen_point(cart_x: float, height_m: float, width: int, baseline: int) -> tuple[int, int]:
    x = int(round(width / 2 + cart_x * (width - 80) / 20.0))
    y = int(round(baseline - height_m * 26.0))
    return x, y


def _frame(state: np.ndarray, tick: int, width: int = 720, height: int = 440) -> Image.Image:
    image = Image.new("RGB", (width, height), "#111827")
    draw = ImageDraw.Draw(image)
    baseline = 265
    draw.line((30, baseline, width - 30, baseline), fill="#64748b", width=2)
    for rail_x in (-10.0, 10.0):
        x, _ = _screen_point(rail_x, 0.0, width, baseline)
        draw.line((x, baseline - 8, x, baseline + 8), fill="#f59e0b", width=2)

    cart_x = float(state[0])
    cart_screen_x, cart_screen_y = _screen_point(cart_x, 0.0, width, baseline)
    draw.rectangle(
        (cart_screen_x - 15, cart_screen_y - 8, cart_screen_x + 15, cart_screen_y + 8),
        fill="#38bdf8",
    )
    point_x, point_y = cart_x, 0.0
    previous = _screen_point(point_x, point_y, width, baseline)
    for theta in np.asarray(state[1 : 1 + N_LINKS], dtype=np.float64):
        point_x += 0.5 * math.sin(float(theta))
        point_y += 0.5 * math.cos(float(theta))
        current = _screen_point(point_x, point_y, width, baseline)
        draw.line((*previous, *current), fill="#f8fafc", width=3)
        draw.ellipse(
            (current[0] - 3, current[1] - 3, current[0] + 3, current[1] + 3),
            fill="#f43f5e",
        )
        previous = current
    draw.text((24, 20), f"N13 fresh composed rollout  t={tick / 1000:.3f}s", fill="#f8fafc")
    return image


def write_fresh_gif(states: np.ndarray, output: Path) -> dict[str, Any]:
    """Render only the state array produced by this process's fresh rollout."""
    ticks = np.linspace(0, len(states) - 1, num=120, dtype=np.int64)
    frames = [_frame(states[tick], int(tick)) for tick in ticks]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=False,
    )
    return {
        "path": output.as_posix(),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "frames": len(frames),
        "source": "in-memory fresh rollout only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gif",
        type=Path,
        default=None,
        help="output GIF path under this repository (default: .working/n13-demo.gif)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        prepared = prepare_run()
        output = (prepared.root / ".working/n13-demo.gif") if args.gif is None else args.gif.resolve()
        if not output.resolve().is_relative_to((prepared.root / ".working").resolve()):
            raise ValueError("GIF output must stay under .working/")
        rollout = fresh_composed_rollout(prepared)
        gif = write_fresh_gif(rollout["x"], output)
        gif["path"] = output.resolve().relative_to(prepared.root).as_posix()
    except Exception as error:
        print(f"n13-demo failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    result = {
        "classification": "N13_ONE_RUN_PASS" if rollout["passed"] else "N13_ONE_RUN_FAIL",
        "scope": "deterministic_one_run_proof",
        "authority_inputs_loaded": prepared.authority_inputs,
        "fresh_rollout": {
            "switch_tick": rollout["switch_tick"],
            "switch_time_s": rollout["switch_time_s"],
            "raw_peak_n": rollout["raw_peak_n"],
            "node_cart_peak_m": rollout["node_cart_peak_m"],
            "quarter_cart_peak_m": rollout["quarter_cart_peak_m"],
            "trailing_success_s": rollout["trailing_success_s"],
            "trailing_success_samples": rollout["trailing_success_samples"],
            "saved_tracking_reference_loaded": rollout["saved_tracking_reference_loaded"],
            "saved_b2_rollout_trace_loaded": rollout["saved_b2_rollout_trace_loaded"],
            "gates": rollout["gates"],
        },
        "gif": gif,
        "exclusions": list(EXCLUSIONS),
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if rollout["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
