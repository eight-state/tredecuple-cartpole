# N13 B2 proof verification

**Verdict: PASS.** The claimed `N13_ONE_RUN_PASS` is reproduced from the immutable controller and proof data without relying on the bundle’s JSON outcome claims.

## Authority and integrity

- Rehashed base authority `runs/r2/nom_n13_4ms_n13_base60r3.npz`: `7179a8f0bae0a40a895a68e17cc9c7b4c17a2287d201fedb1023b92b0e1726a2`.
- Rehashed B0 Arm-A SDA authority: `6a78a3f2f3dd8f476b150194949f914b68a298237342293bd6d80d52ae33e84f`.
- Rehashed all declared local source dependencies. The locked dynamics, environment, LQR, funnel, densifier, exact-map, and SDA source hashes all match their retained values. The driver is `1168aafa6ff6650ccd282f59b3a6705110b01c30f2bb67e0bbba8c765ac1dc1b`, matching the manifest-recorded value.
- Controller artifact SHA-256: `7d57677e6858113d908334e19b37b6286341129e5f8f8f9e93bb36653fbdefb8`; proof artifact SHA-256: `fffb9466ee0be82646867ed6c8f13748827a2d157144eb0c81cfe642fc0a005b`.
- Re-densifying the pinned base archive on the reconstructed plant yields controller `x_ref` and `u_ref` byte-for-byte. Controller static K and P are byte-for-byte equal to Arm-A authority (`522f…a5fc86`, `8a27…128dad`). The stored raw/wrapped defect discrepancy is `2.220446049250313e-16`.

## Plant and timing

The verifier rebuilt N13 from exact hanging with 13 × 0.10 kg links, 0.50 m links, zero link damping, 1.0 kg cart, ±150 N force bound, ±10 m rail, 1 kHz ZOH, and four 0.25 ms RK4 substeps per tick. All three stored time vectors are exact 1 kHz grids.

## Independent fresh execution

The verifier directly stepped the policy and native RK4 boundary, rather than accepting stored state values.

| Trace | State/raw/applied match | State max delta | Raw/applied peak | Node / quarter cart peak |
|---|---:|---:|---:|---:|
| 10,000-tick affine live run | byte-identical | 0.0 | 39.83765056127226 N | 6.900186930927419 / 6.900188430319146 m |
| 12,000-tick static SDA hold | byte-identical | 0.0 | 38.10396084189415 N | 0.6467318637216906 / 0.6467318637216906 m |
| 17,577-tick composed proof | byte-identical | 0.0 | 39.83765056127226 N | 6.900186930927419 / 6.900188430319146 m |

For every trace, raw and applied controls are byte-identical: no clipping occurred. Every node and every quarter step was finite and within the 10 m rail. The composed proof phase vector is byte-identical to 5,577 affine ticks followed by 12,000 static-SDA ticks.

Independent replay of each saved applied-control vector also reproduced its saved states and applied controls byte-for-byte, with zero state delta: the live tracker, selected static hold, and full composed proof all passed.

## Switch and success

- Switch: tick 5577, 5.577 s.
- Instantaneous switch metrics: cart `0.6467318637216906 m`; cart rate `0.2503705543357863 m/s`; maximum wrapped link-angle error `4.997237741964539°`; maximum link-rate magnitude `0.13331785830210413 rad/s`.
- The switch state satisfies the locked success predicate.
- Recomputed trailing in-set duration: **9.048 s**, **9049 samples**, across all 12,001 post-switch states.

## Injection, optimizer, and reference-path audit

The current hashed driver’s `composed_proof` policy has no assignment to the live state and does not reference `extracted_live_x`; that saved trace is used only for the prefix comparison. Fresh policy rollout and independent applied-control replay both reproduce every stored proof state exactly, ruling out state injection in the saved proof path.

AST inspection found no direct optimizer call nodes in the driver. The proof uses the loaded immutable affine controller, whose reference is independently tied to the sole pinned base60 archive and whose static K/P are tied to the Arm-A authority. No alternate/reference-only N13 state is needed to reproduce the proof.

## Tool checks

- Python compilation: `COMPILE_PASS`
- Ruff: `All checks passed!`
- Driver self-test: `N13_B2_SELF_TEST_PASS`

The machine-readable independent result is `.working/n13-retrospective/b2-proof-verification-result.json`; the verifier is `.working/n13-retrospective/b2-proof-verify.py`. No driver or target proof artifact was modified.
