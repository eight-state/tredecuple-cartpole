# N13 current-target problem definition

## Problem

Produce one deterministic, unperturbed N13 run that starts from exact hanging, swings upright under the unchanged plant, and then remains continuously inside the locked success set for at least five seconds.

This is a bounded next-rung problem. N13 has no retained nominal. The nearest proven ingredients are the N12 `nom_n12_4ms_fast.npz` swing family and N12's exact early handoff at tick 9700.

## Locked constraints

- Thirteen identical 0.10 kg, 0.50 m, undamped links and 1.0 kg cart.
- Force bound `±150 N`.
- Exact 1 kHz control.
- Four 0.25 ms RK4 substeps per control tick.
- Unchanged success predicate: all wrapped angles `<=5°`, link rates `<=0.5 rad/s`, `|cart|<=2 m`, `|cart rate|<=0.5 m/s`, continuously for at least five seconds.
- No change to lower-rung defaults or retained artifacts.

## Scope

In scope: one nominal/warm construction, exact tracking classification, a numerically verified N13 upright hold gain, capture/handoff selection, and one exact end-to-end proof.

Out of scope for this target: perturbation robustness, release seeds, `72/72`, the separate `0.05°` extraction/promotion screen, N14+, and broad controller redesign unless the existing pipeline is numerically invalid at N13.

## Known evidence

- No solved N13 nominal exists. `.working/n13/n13-from-n12-fast-warm.npz` is a finite `(2501,28)/(2500,)` copied warm artifact, not a candidate nominal.
- The copied warm has exact reset-node defect `0.8179348550`; its exact chained controls end at `162.8571359°`. Its planned terminal is therefore not physical evidence.
- N13 uses the 50-digit MPMath SDA path from `hold_gain_and_P()`, downcast to float64. The retained gain has `rho=0.9995711475`, margin `4.2885255e-4`, and peak magnitude `2.5701032e9`. Balanced DARE is not the active controller contract.
- The copied planned terminal requests `-45,876.2652 N` from that gain, so pointwise planned success is outside the force-feasible capture basin.
- N12 succeeded with Qv×0.25 and a 9.700 s continuous-CARE handoff. Neither setting transfers directly: N13's supported terminal P and hold controller come from discrete MPMath SDA.

## Unknowns to settle before a long solve

1. The downcast DARE residual, conditioning, and nonlinear local-hold behavior of the active MPMath SDA gain.
2. The exact N13 tracker contract: Q, terminal P source, and acceptance defect for a solved candidate.
3. Which bounded polish turns the copied warm's `0.8179348550` exact defect into a physical N13 trajectory.
4. Whether the resulting live trajectory contains a force-feasible static-handoff interval and where it lies. The search can begin only after exact tracking passes.

## Success evidence

A fresh process must execute one uninterrupted exact rollout from hanging. It passes only if all values are finite, commanded/applied force stays within 150 N, cart stays within 10 m, the handoff state is inside the success set, and the final continuous in-set duration is at least five seconds. Solver defects and planned endpoints are diagnostics, never the verdict.
