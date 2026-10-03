# Neural Physical Planner — research prototype v0.3

A runnable compiler for exploring alternative **hardware macro-plans** for a neural computation graph. It takes explicit weights, a physical component and compensation-rule library, and a design request. It returns feasible implementations, a Pareto set, predicted costs and error, an expanded implementation graph, and a schedule.

Version 0.3 adds an explicit physical layer for one NN node's optical distribution
circuit, with component evidence, typed connections, SAX power comparisons and
constrained designer replanning. The macro planner retains its additive Gaussian
surrogate and schema 0.1. Physical subcircuit metrics use separate boundaries.
The example hardware values are **illustrative and uncalibrated**. This is not a
fabrication tool or a whole-NN physical implementation, and the examples establish
no research novelty or measured hardware-performance claim.

## Start with a friendly project

Version 0.2 adds commented YAML specifications, external saved weights, physical units and a single project file. Existing v0.1 JSON inputs and saved-plan replay remain supported; the physical model is unchanged.

```bash
python -m pip install .
npp init my-design
npp inspect-weights my-design/weights.npz
npp plan --project my-design/project.yaml --out-dir runs/my-design --samples 10000
```

The starter writes `project.yaml`, `model.yaml`, `hardware.yaml`, `design.yaml` and a small `weights.npz`. Relative paths are resolved beside the file that declares them, so the command works from another directory. YAML comments explain the settings. The starter hardware is explicitly synthetic.

A model can refer to trained tensors by name:

```yaml
name: My trained network
weights: trained.pth
layout: out_in
input:
  size: 2
  bounds: [-1, 1]
layers:
  - name: hidden
    type: linear
    weights: hidden.weight
    bias: hidden.bias
  - name: activation
    type: relu
  - name: readout
    type: linear
    weights: readout.weight
    bias: readout.bias
```

The layer list defines the actual architecture. The compiler infers output dimensions and connections for this sequential form; it does not guess activations from weights. Residual/DAG networks can still use the explicit graph form. Input bounds describe the operating range of the data supplied at inference.

Design requests can use readable names and units:

```yaml
minimize: [energy, latency, error]
limits:
  latency: 20 ns
  energy: 2 nJ
  error: 0.1
editable: true
search: exhaustive
```

See [the input guide](docs/INPUT_GUIDE.md) for full examples, hardware overrides, matrix orientation and troubleshooting. To inspect exactly what will be compiled, run `npp normalize --project my-design/project.yaml --out-dir normalized`. The resulting canonical JSON embeds the resolved weights and is independent of the source weight files.

## Inspect a physical optical distribution circuit

To search alternatives under designer constraints, start with:

```bash
npp plan-physical --network examples/physical/fanout4.json \
  --design examples/physical/design.yaml --out-dir runs/physical-design
npp check-physical-design --result runs/physical-design/design-result.json
```

The [designer guide](docs/DESIGNER.md) explains readable units, component and
power locks, forbidden recipes, Pareto alternatives, rejection explanations and
replanning from an existing realization. The output report distinguishes a
complete finite search from a truncated one and links to each retained circuit.

The v0.3 physical layer expands one bounded NN node's fanout into explicit sources,
splitters, waveguides, detectors and optional regeneration components. It supplies
typed connections, per-instance resource costs, receiver power margins and a
portable replay-checked record. These physical subcircuit metrics are separate from
the macro planner's whole-graph estimates.

```bash
npp realize --network examples/physical/network.json \
  --spec examples/physical/realization.yaml --out-dir runs/physical
npp check-realization --realization runs/physical/realization.json
npp export-optical --realization runs/physical/realization.json --out optical-netlist.json
```

For an external SAX power-composition comparison, use Python 3.11+ and install the
optional solver. Export and the core physical planner do not require SAX.

```bash
python -m pip install '.[photonics]'
npp simulate-realization --realization runs/physical/realization.json \
  --wavelengths-nm 1540 1550 1560 --out-dir runs/optical-comparison
```

Open each output directory's `report.html` to inspect the circuit or comparison.
The technology values remain illustrative and uncalibrated. SAX checks composition
of declared full-scale optical transfers; it does not validate the electronics,
noise, timing, NN task accuracy or fabrication layout. See
[technology contracts](docs/TECHNOLOGY.md), [physical implementations](docs/IMPLEMENTATION.md)
and [external simulation](docs/SIMULATION.md) for the supported boundaries.

## Saved weight formats

| File format | Support | Install extra |
|---|---|---|
| NumPy `.npz`, `.npy` | Named arrays, or a single array named `array` | Included |
| PyTorch `.pt`, `.pth`, `.bin` | Tensor `state_dict`; common checkpoint wrappers; explicit custom wrapper key | `pytorch` |
| SafeTensors `.safetensors` | Named dense tensors | `safetensors` |
| HDF5 `.h5`, `.hdf5`, `.weights.h5` | Numeric datasets selected by their exact paths | `hdf5` |
| Keras `.keras` | Embedded HDF5 weights; architecture described separately | `hdf5` |
| ONNX `.onnx` | Embedded initializer weights, or supported graph import | `onnx` |

Install the extras you need, for example `python -m pip install '.[pytorch,safetensors]'`, or `python -m pip install '.[all]'` for every importer. The core installation does not require PyTorch, TensorFlow or ONNX. Use `npp inspect-weights PATH` to list names, shapes and dtypes; add `--json` for agents/scripts. PyTorch loading always uses the restricted weights-only CPU path. Full pickled Python models and TorchScript archives are not loaded.

Keras Dense kernels use `[input, output]`, so specify `layout: in_out`. The default `out_in` follows PyTorch Linear. No transpose is guessed from a square matrix. HDF5 is a container format: use the convention of the software that saved it.

ONNX can include the architecture as well:

```bash
npp import-model network.onnx --out network.json --input-min -1 --input-max 1
```

The graph importer supports one vector/batch-one input and standard `Gemm`, constant-weight `MatMul`, supported `Add`, `Relu`, `Identity` and constants. Unsupported operators are rejected. This does not add CNN, recurrent-network or Transformer synthesis to the backend. External-data ONNX models and sharded checkpoints require a self-contained export. Import size limits are documented in the input guide.

## Run the original working demo

Requires Python 3.10+ with NumPy, Pydantic 2 and PyYAML. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
npp demo --out-dir runs/my-demo --samples 10000
```

On Windows, activate with `.venv\Scripts\activate`. If dependencies are already installed, `python -m npp` works directly from this directory in place of `npp`.

Open `runs/my-demo/report.html`. The bundled reconvergence example enumerates 512 implementation choices and retains 10 Pareto designs. It checks every retained plan and samples its primitive noise sources. `runs/demo/` contains a precomputed run for immediate inspection.

```bash
python -m unittest discover -s tests -v
python scripts/run_experiments.py --out-dir runs/my-experiments
```

The experiments compare the full design space with digital-only, passive-only, editable-weight and beam-search variants, exercise a ReLU graph, and demonstrate an infeasible budget. They are software/model demonstrations, not device benchmarks.

## The three inputs

| Input | File | What it specifies |
|---|---|---|
| Neural program | `examples/split_recombine.json` | Topologically ordered graph, weights, biases, finite input bounds and output nodes. |
| Hardware knowledge base | `examples/components.json` | Compatible macros, physical parameters, domain converters and guarded fan-out recipes. |
| Design request | `examples/request_default.json` | Objectives, budgets, editable weight nodes, allowed domains, serialization permission and search limits. |

Weights alone do not describe a network: graph structure and activation semantics are also required. Version 0.1 supports one bounded vector input, dense linear maps, addition, identity and digital ReLU. Components provide declared width capacity and output headroom. Each graph node receives a dedicated macro.

The output is a set of alternatives rather than a single hidden weighted score. Any nonempty subset of energy per inference, latency, macro area and RMS output error can be minimized; hard budgets are separate. Editability is a required capability on specified linear nodes. Serialization can be forbidden. General memory/random-access synthesis, parameter training, resource sharing and throughput optimization are not implemented.

## Compile your own request

```bash
npp validate --network examples/split_recombine.json \
  --library examples/components.json --request examples/request_default.json

npp plan --network examples/split_recombine.json \
  --library examples/components.json --request examples/request_default.json \
  --out-dir runs/custom --samples 20000

npp schemas --out-dir exported-schemas
```

Command results are JSON except the human-readable `inspect-weights` table (use `--json` for structured output). Exit codes are `0` for success, `2` for invalid input or a failed plan check, `3` for infeasibility established within the finite declared choice space, and `4` for an incomplete search that found no feasible plan. A successful heuristic search can return useful plans with `search.complete=false`; success does not assert optimality. Output directories must be new or empty unless `--overwrite` is explicit. The run manifest lists the current plans; overwriting does not delete unrelated or stale files.

## Inspect and verify outputs

| Artifact | Contents |
|---|---|
| `inputs/*.json` | Normalized snapshots of the three inputs. |
| `report.json` | Pareto plans, input hashes, search coverage, rejection counts and assumptions. |
| `plans/plan-<hash>.json` | One decision and its full evaluation, physical graph and schedule. |
| `report.html` | Offline report, trade-off plot, neural graph, design choices and model details. |
| `pareto.csv` | Metrics and decisions for comparison in other tools. |
| `checks.json` | Replay-check results for every retained plan. |
| `simulations.json` | Optional forward Monte Carlo results with sampling uncertainty. |
| `manifest.json` | Current run files and software version. |

Set `PLAN` to one path listed in `manifest.json`:

```bash
npp check --network runs/custom/inputs/network.json \
  --library runs/custom/inputs/library.json --request runs/custom/inputs/request.json \
  --plan "$PLAN"

npp simulate --network runs/custom/inputs/network.json \
  --library runs/custom/inputs/library.json --request runs/custom/inputs/request.json \
  --plan "$PLAN" --samples 50000 --seed 7
```

The checker validates normalized input hashes and the decision ID, recomputes every evaluation field and rejects changed metrics, topology or schedule. It reuses the evaluator; it is **not** an independent physics proof or a cryptographic authorship signature. Independent forward sampling provides a separate numerical consistency check of the implemented noise model.

## What the compiler implements

1. Strict versioned input validation: dimensions, graph order, names, bounded values and supported capabilities.
2. Compatible component selection and fan-out recipe generation, including repeated operands and final output sinks.
3. Guarded digital copy, passive split, source power boost, noisy amplification, regeneration and serial re-encoding. Domain conversions are inserted when required.
4. Physical evaluation: costs, launch energy, range and gain headroom, expanded macrograph and dependency schedule.
5. Exact joint covariance for affine graphs, retaining shared errors across split/recombine structures. Graphs with ReLU receive a conservative RMS bound using operator norms and Lipschitz/Minkowski inequalities.
6. Exhaustive enumeration or bounded heuristic beam search, with nondominated alternatives and explicit search coverage.
7. Replay checking, independent primitive-level Monte Carlo, portable reports and a JSON command interface.

New instances of supported components/rules can be added as data. A new physical mechanism needs an implemented, reviewed evaluator and guards: arbitrary natural-language rules are not executed. This creates a useful boundary for engineer- or agent-proposed changes.

## Scientific scope and limitations

Optical channels use normalized signal-independent additive Gaussian noise with a photon-count-dependent variance. Gain changes the photon budget while preserving the ideal numerical signal. Shared pre-split error is never erased. Signed optical encoding, coherent interference, saturation, detailed converter internals and device fabrication are abstracted. Nominal interval checks do not constrain Gaussian tails.

Costs are library assumptions per macro or complete fan-out recipe; they do not include physical routing, port-limited synthesis, thermal control, packaging, static/leakage power or yield unless explicitly folded into supplied values. Latency is for one inference under the modeled schedule, not throughput. Input intervals and nonlinear error bounds can be conservative. Float64 computations are numerical estimates, not formally rounded proofs.

Exhaustive search establishes the frontier only over the supplied finite choices and numerical comparison tolerance. Beam search has no admissible pruning guarantee. Dense covariance propagation and enumeration are intended for small research graphs; production-size networks require sparse representations, decomposition and stronger search methods.

The supplied experiments validate software semantics under these assumptions. They cannot validate real hardware parameters. A next research milestone is a characterized component library and comparison with an external device/circuit simulator on a narrowly chosen encoding.

## Project map and extension points

- `npp/models.py`: strict contracts and schema generation.
- `npp/config.py`, `npp/templates.py`: friendly specifications, unit conversion and starter projects.
- `npp/weights.py`, `npp/onnx_import.py`: external tensor loading and supported ONNX graph lowering.
- `npp/physics.py`: guards, interval analysis, noise propagation, scheduling and sampling.
- `npp/planner.py`: choice construction, enumeration, beam search and Pareto filtering.
- `npp/checker.py`: saved-plan replay checking.
- `npp/cli.py`, `npp/reports.py`: run bundles and offline reports.
- `examples/`, `npp/data/`: editable examples and installed demo data.
- `schemas/`: input and output contracts; semantic checks additionally require Python validation.
- `tests/`: independent numerical oracles, rejection, optimization, tampering and CLI checks.
- [System specification](docs/SYSTEM_SPEC.md), [physical model](docs/PHYSICS.md), [agent interface](docs/AGENT_INTERFACE.md).

The immediate engineering path is to choose one real hardware family, characterize macro/rule coefficients and uncertainty, add port/routing constraints, and implement a lowering backend with external validation. Broader operator lowering, storage synthesis and additional device mechanisms should build on that checked core.
