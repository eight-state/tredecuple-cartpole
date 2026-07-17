# N13 deterministic one-run proof

![N13 cart-pole swing-up and balance](docs/n13-demo.gif)

This README limits scope to one deterministic, unperturbed, model-only N13 simulation from exact hanging to a final 1 kHz-sampled success-set hold (`deterministic_one_run_proof`).

```text
uv sync --locked
uv run n13-demo
uv run n13-proof-verify
```

`n13-demo` loads the frozen B0 nominal states and controls as its tracking reference, the Arm-A static gain, and the B2 affine controller. It computes each control from the current state, advances the native plant with four 0.25 ms RK4 steps per 1 kHz tick, and renders only the newly integrated states to `.working/n13-demo.gif`. The saved B2 proof rollout never supplies animation states.

`n13-proof-verify` first checks literal SHA-256 authority for the complete ten-file B2 source closure, the historical manifest, the controller archive, and the stored feedback/feedforward payloads. It then checks the SHA-256 of the separately preserved `.working/n13-retrospective/b2-proof-verify.py` and runs it unchanged. Numeric platform drift is eligible for the portable gate only after fixed authority and preserved-verifier result semantics pass. The verifier executes the fixed archived controller but does not claim to regenerate its 10,000-step affine derivation.

The switch node at 5.577 s is inside the success set, but the run later leaves and re-enters it. The final 9,049 in-set nodes cover ticks 8,529 through 17,577 and span `(9049 - 1) × 0.001 = 9.048 s`. Success is evaluated at 1 kHz node states. Quarter steps are checked for finite state and rail position; this is not an analytic continuous-time membership claim.

`evidence/copy-manifest.json` records all 27 copied inputs with source path, SHA-256, size, and category. `evidence/problem-definition.md` is an archived pre-delivery planning record: its statement that no retained N13 nominal existed predates the frozen B0 N13 nominal now shipped under `runs/r2/`. The capsule reports every authority input it loads.

The commands above are supported from a full source checkout after `uv sync --locked`. The wheel omits the repository-relative proof authority and is not a standalone proof capsule.

This README excludes perturbation robustness, release seed gates, `72/72` promotion, statistical robustness, formal guarantees, and hardware behavior. The capsule contains no optimizer run or promotion decision.
