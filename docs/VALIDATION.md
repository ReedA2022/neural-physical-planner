# Reproduced software validation

The supplied run bundles were produced using Python 3.12 with NumPy and Pydantic 2. The component coefficients are synthetic. Results below establish behavior of this implementation under its declared model; they do not establish photonic feasibility, fabrication readiness, measured device performance or research novelty.

## Version 0.3 milestone verification

All four parts passed independent critique in the required order. Parts 1, 2
and 3 each needed a second review after fixes; Part 4 passed its first review.
The findings, regression tests and final judgments are preserved in
`docs/reviews/`. Automated review covers software and declared-model consistency;
the reference technology remains illustrative and unreviewed by a photonics expert.

Final tests on 2026-10-03 used Python 3.12.14:

- With all importers, Keras and SAX installed: **268 tests run, 267 passed,
  one expected skip**, no failures. The skipped test specifically requires SAX
  to be absent.
- Core-only environment: **268 tests run, 227 passed, 41 expected optional
  dependency skips**, no failures. The missing-SAX diagnostic path passed.
- The original macro demo still evaluates 512 candidates and retains 10 plans.
  Normalized input hashes and every retained plan record exactly match the
  bundled original demo. All plans replay and pass the 10000-sample diagnostic.

The final physical integration harness is reproducible with:

```sh
python -m pip install '.[photonics]'
python scripts/validate_physical.py --out-dir runs/new-physical-validation
```

It exercises passive and regenerative fanout 2/4/8 with unequal route lengths,
then five designer scenarios. Every retained physical plan is replayed and
compared with actual SAX/KLU over 1540, 1545, 1550, 1555 and 1560 nm. Across
**11 circuits and 370 detector/termination power comparisons**, all pass the
declared tolerances. The largest absolute difference is approximately
**1.11e-16 mW**. This checks composition of shared illustrative model assumptions,
not calibration, electronics, noise, timing or complete NN inference.

| Designer case | Submitted | Evaluated | Excluded | Unevaluated | Feasible | Pareto | Complete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Unlocked | 9 | 9 | 0 | 0 | 8 | 3 | Yes |
| Launch locked to baseline | 9 | 3 | 6 | 0 | 2 | 2 | Yes |
| Locked; regeneration forbidden | 9 | 1 | 8 | 0 | 0 | 0 | Yes |
| One evaluation allowed | 9 | 1 | 0 | 8 | 0 | 0 | No |
| Zero energy budget | 9 | 9 | 0 | 0 | 0 | 0 | Yes |

Saved evidence is in `runs/validation-v0.3/`: normalized inputs, design records,
circuit and comparison reports, optical exports, per-boundary solver results,
environment versions, test counts and legacy compatibility checks. These are
small functional examples, not scalability benchmarks.

The tested optical stack is SAX 0.18.2, JAX/JAXlib 0.9.2 and klujax 0.5.2;
`requirements-photonics-tested.txt` records it. Python 3.11+ is required for SAX;
core planning and export still support Python 3.10+. No external solver is
silently substituted when the photonics dependencies are missing.

The 0.3.0 wheel was built, installed into a separate target and exercised from
outside the source checkout. Its technology/schema exports, physical planning,
baseline replanning, replay and actual SAX comparison all passed. The installed
package also replayed a design and exported its optical netlist with the core-only
Python environment. The recorded module location confirms the wheel was used.

## Version 0.2 input verification

The complete suite passed **110 tests with no skips** using all optional import dependencies and Keras for export verification. A separate core-only environment passed 79 tests and skipped the 31 tests that require optional packages. Importing Keras weights requires only `h5py`; Keras itself is used solely to verify genuine framework exports in the test suite.

Verified input behavior includes:

- Actual saved NumPy, SafeTensors, PyTorch, HDF5, Keras and ONNX files; checkpoint wrappers; exact tensor-name selection; relative paths and structured rejection of corrupt or mismatched inputs.
- Deliberately nonsymmetric matrices, including square matrices, to detect incorrect transposition. Independent NumPy, PyTorch, ONNX reference and Keras predictions agree within the applicable numerical tolerance.
- Genuine `.keras` and `.weights.h5` files written by Keras 3 using its torch backend. Both import correctly while Keras and TensorFlow imports are blocked in the tensor loader's process.
- The same parameters imported from equivalent formats yield identical Pareto decisions and metrics. Every retained plan passes replay checking.
- CLI execution from a different working directory and continued plan/replay operation after normalization and deletion of the source weights.
- Existing canonical JSON inputs, duplicate-key and unknown-field rejection, physical units, dimension checks, optional-dependency diagnostics and preservation of existing starter files.

The new `examples/friendly/project.yaml` run evaluates 32 candidates and retains 3 Pareto plans. All three pass replay checking and the consistency diagnostic with 10,000 independent forward samples per plan. The actual results are in `runs/friendly-v0.2/`.

The original demo was also rerun with version 0.2: its input hashes, all 10 retained plan records, decisions and evaluation fields exactly match the bundled version 0.1 run. It still evaluates 512 candidates. All retained plans pass replay and the 10,000-sample consistency diagnostic.

A wheel installation was tested outside the source checkout in an environment without optional import frameworks: `npp init`, tensor inspection and the complete project planning/replay/sampling workflow succeed.

Tested optional versions were PyTorch 2.14.1+cpu, SafeTensors 0.8.0, h5py 3.16.0, ONNX 1.23.1 and Keras 3.15.1. Core versions are recorded in `requirements-tested.txt`. To reproduce all tests, install the project with its `all` extra and also install Keras; the Keras test selects the torch backend in an isolated subprocess. With just the `all` extra, that one framework-export test skips if Keras is absent. The supported dependency ranges are wider than this tested set.

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
