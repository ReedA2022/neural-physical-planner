# Neural Physical Planner — research prototype v0.1

A runnable compiler for exploring alternative **hardware macro-plans** for a neural computation graph. It takes explicit weights, a physical component and compensation-rule library, and a design request. It returns feasible implementations, a Pareto set, predicted costs and error, an expanded implementation graph, and a schedule.

This release establishes a concrete input/output contract and an end-to-end implementation. The example hardware coefficients are **synthetic**. The backend is an explicitly defined additive Gaussian surrogate for digital/optical computation, **not a fabrication netlist generator or a calibrated photonic simulator**. No novelty or hardware-performance claim follows from these examples.

## Run the working demo

Requires Python 3.10+ with NumPy and Pydantic 2. From this directory:

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

All command results are JSON. Exit codes are `0` for success, `2` for invalid input or a failed plan check, `3` for infeasibility established within the finite declared choice space, and `4` for an incomplete search that found no feasible plan. A successful heuristic search can return useful plans with `search.complete=false`; success does not assert optimality. Output directories must be new or empty unless `--overwrite` is explicit. The run manifest lists the current plans; overwriting does not delete unrelated or stale files.

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
- `npp/physics.py`: guards, interval analysis, noise propagation, scheduling and sampling.
- `npp/planner.py`: choice construction, enumeration, beam search and Pareto filtering.
- `npp/checker.py`: saved-plan replay checking.
- `npp/cli.py`, `npp/reports.py`: run bundles and offline reports.
- `examples/`, `npp/data/`: editable examples and installed demo data.
- `schemas/`: input and output contracts; semantic checks additionally require Python validation.
- `tests/`: independent numerical oracles, rejection, optimization, tampering and CLI checks.
- [System specification](docs/SYSTEM_SPEC.md), [physical model](docs/PHYSICS.md), [agent interface](docs/AGENT_INTERFACE.md).

The immediate engineering path is to choose one real hardware family, characterize macro/rule coefficients and uncertainty, add port/routing constraints, and implement a lowering backend with external validation. ONNX/PyTorch import, storage synthesis and broader device mechanisms should build on that checked core.
