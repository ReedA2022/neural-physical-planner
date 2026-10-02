# Reproduced software validation

The supplied run bundles were produced using Python 3.12 with NumPy and Pydantic 2. The component coefficients are synthetic. Results below establish behavior of this implementation under its declared model; they do not establish photonic feasibility, fabrication readiness, measured device performance or research novelty.

## End-to-end cases

`python scripts/run_experiments.py --out-dir runs/new-experiments --samples 5000` reproduces these cases. Each retained plan is replay-checked and independently sampled. Refer to each run's input snapshots for the precise request and library.

| Case | Evaluated | Feasible | Retained | Complete search |
|---|---:|---:|---:|---|
| Full reconvergence example | 512 | 512 | 10 | Yes |
| Digital only | 8 | 8 | 1 | Yes |
| Passive splitting and digital copy only | 192 | 192 | 8 | Yes |
| Editable linear weights required | 128 | 128 | 10 | Yes |
| Beam width/evaluation limit 24 | 24 | 24 | 6 | No |
| ReLU example | 32 | 32 | 3 | Yes |
| Zero-energy impossible request | 448 | 0 | 0 | Yes |

All retained plans in these saved runs pass replay checking and the reported Monte Carlo consistency diagnostic. The impossible request forbids serialization, explaining its smaller choice space. Counts describe these small examples; they are not scalability results. Beam search can miss feasible or preferable solutions.

## Independent numerical checks

Run `python -m unittest discover -s tests -v`. The suite checks:

- Covariance against manually derived shared-source and independent-source formulas, including reconvergent cancellation and repeated uses.
- Optical passive, boosted, amplified, regenerated and serialized paths against closed-form energy/error expectations.
- Headroom, domain, editability, serialization, input-structure and cost-budget rejection.
- Conservative ReLU bounds and independent primitive-level Monte Carlo agreement.
- A separately derived finite search frontier, nonconvex nondominated alternatives, and complete versus incomplete infeasibility claims.
- Replay rejection of changed weights, hashes, components, topology, metrics and malformed numerical claims.
- Tiny nonzero quantities against zero budgets, extreme-scale sampling cancellation and stable beam ranking for large finite weights.
- CLI run export, subprocess replay checking and infeasible-request exit status.

The checker shares its evaluator with planning; it catches altered/stale plans but cannot discover every mistake in that evaluator. The analytical oracles and separate forward sampler target this remaining risk. The sampler and evaluator still share the primitive parameter specification, so neither independently validates real device physics.

## Reproducibility boundaries

Normalized inputs are hashed; decisions and retained ordering are deterministic for fixed inputs and software. Search elapsed time and local output paths differ across runs. Monte Carlo seeds are explicit. Numerical results can vary slightly across NumPy/platform versions. `requirements-tested.txt` records the versions used for the bundled results; the package supports a wider declared dependency range.

Finite precision is not directed interval arithmetic. Reported affine covariance is exact algebraically within the declared model, evaluated numerically. Nonlinear RMS bounds do not bound every sample or classification error. The source's floating-point limits and physical assumptions remain part of the interpretation.
