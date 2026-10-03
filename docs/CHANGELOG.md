# Changes

## 0.3.1 — stress-tested research-prototype maintenance

- Corrected legacy frontier selection near its relative comparison tolerance;
  tiny worsening in one objective can no longer be forgiven by dominance.
- Guarded positive noise, covariance and physical contributions against silent
  floating-point range loss. Stabilized nonlinear error aggregation, optical
  launch energy and Monte Carlo uncertainty; unsupported ranges fail explicitly.
- Added simulation allocation/work preflight and strict seed validation.
- Made equivalent decimal unit strings normalize identically, rejected nonzero
  quantity strings rounding to zero and mixed boolean/numeric inline tensors,
  and bounded repeated YAML alias expansion.
- Kept report coordinates finite for extreme finite metrics and hardened the
  JSON decoder depth-error path.
- Added five seeded stress campaigns, a source-hashed aggregate release gate,
  adversarial regressions and independent critique. See
  [STRESS_VALIDATION.md](STRESS_VALIDATION.md) for executed evidence and limits.

Schemas and the supported v0.3 scope are unchanged. Extreme numeric records and
near-tied legacy frontiers affected by these correctness fixes can intentionally
change. Frozen ordinary v0.3 artifacts remain the compatibility gate.

## M0 — multiphysics specification reconciliation (on 0.3.0)

- Preserved the supplied specification, including equations and its original
  DOCX, with a full REQ-01–37 inventory against the actual v0.3 implementation.
- Added a versioned migration roadmap and H1–H6 experiment definitions with
  explicit baselines, negative controls, unresolved assumptions and dependencies.
- Added an immutable v0.3 baseline manifest and executable compatibility checks
  for legacy planning, physical realizations, designer cases and schema meaning.
- Recorded M0 tests and independent critique in
  [the gate record](multiphysics/M0_GATE.md).

At the M0 gate the runtime remained 0.3.0. M1–M6 are planned work; the proposed multiphysics
configuration and experiment registry are not current CLI inputs. This integration
does not add mixed neural operators, acoustic/quantum models or decoder execution.

## 0.3.0 — inspectable optical circuits and designer interaction

- Added strict physical technology packs with typed ports, operating envelopes,
  component equations, parameter evidence and explicit review/calibration status.
- Added NN-linked optical fanout graphs with paid sources, fixed splitter trees,
  waveguides, detectors, regenerative carriers/modulators and explicit unused
  termination boundaries. Costs, complete-symbol timing, receiver margins and
  shared first-order noise estimates are recorded per physical instance.
- Added portable physical realization replay and offline circuit reports.
- Added optional real SAX/KLU composition of exported optical segments, wavelength
  sweeps, per-boundary power comparisons and reports. Electrical dynamics, noise,
  timing and calibration are explicitly outside that external check.
- Added bounded multiobjective design search, readable quantities, recipe/power
  and component-instance locks, forbidden recipes, hard requirements, baseline
  replanning, rejection explanations and replay of the complete design result.
- Added independent critique gates, adversarial regressions and a reproducible
  physical integration harness. Reviews are preserved in `docs/reviews/`.

The physical schemas use `0.3`; existing neural macro schemas remain `0.1`.
The two workflows keep separate metric boundaries. This release demonstrates
one NN node's optical distribution subcircuit, with illustrative device values.
It does not provide calibrated hardware performance, signed physical NN
operators, arbitrary topology synthesis, fabrication layout or task accuracy.

## 0.2.0 — friendly inputs and saved weights

- Added a single YAML or JSON project file connecting the model, hardware and design settings. The original three JSON input options remain supported.
- Added commented, runnable starter projects with `npp init`, sequential layer descriptions, inferred dimensions, explicit weight layouts and relative paths resolved from the file that declares them.
- Added saved tensors from NumPy, SafeTensors, restricted PyTorch state dictionaries, HDF5, Keras weight archives and ONNX initializers. Each optional framework dependency is installed separately. Inspection lists exact tensor names and dimensions.
- Added supported ONNX graph import for vector/batch-one dense models, including affine operations, ReLU and residual addition. Unsupported operators and inconsistent types are rejected.
- Added physical units, short objective names, editable-weight options, checked synthetic-library overrides and structured input diagnostics. Duplicate YAML/JSON keys and unknown specification fields are rejected.
- Added `npp normalize` to export portable canonical JSON with resolved weights embedded.
- Added independent import, framework prediction, Pareto equivalence, replay and CLI integration checks. See [VALIDATION.md](VALIDATION.md).

The canonical input/output schema remains `0.1`. Existing valid canonical inputs and saved-plan replay remain compatible. The physical model and planning search are unchanged. This release expands input usability; it does not add physical layout generation, new device mechanisms, production-scale optimization or support for arbitrary neural architectures.

The format guide and examples are in [INPUT_GUIDE.md](INPUT_GUIDE.md) and `examples/friendly/`. The bundled `runs/friendly-v0.2/` directory contains an actual checked and sampled result from that example.
