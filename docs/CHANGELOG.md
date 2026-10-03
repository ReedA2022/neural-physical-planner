# Changes

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
