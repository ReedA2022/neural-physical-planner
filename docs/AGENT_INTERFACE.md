# Agent integration and automation

## Division of responsibility

An agent can assemble a bounded NN specification, select an approved library, express requirements, invoke the planner and inspect diagnostics. The physical model and executable rule implementations remain trusted engineering code. Natural-language requests and notes are not executed.

The intended loop is:

1. Translate requirements into project/YAML inputs or the canonical three JSON inputs.
2. Inspect external tensor names, normalize the inputs and fix explicit schema or graph errors.
3. Generate a candidate family under a declared search budget.
4. Inspect feasibility, search completeness and the requested trade-offs.
5. Replay-check selected plans using the exact normalized input snapshots.
6. Compare forward simulation with the analytic prediction when relevant.
7. Present alternatives and unresolved modeling assumptions for engineering review.

A budget violation is not permission to change the user's budget. An agent may propose a relaxation explicitly, but must preserve the original request and label the resulting alternative.

## Command-line interface

Invoke the installed `npp` entry point or `python -m npp`. The project root contains working examples. These commands perform local computation and write local artifacts; no API keys or remote services are required.

### Friendly project workflow

```bash
python -m npp init my-project
python -m npp inspect-weights my-project/weights.npz --json
python -m npp validate --project my-project/project.yaml
python -m npp normalize --project my-project/project.yaml --out-dir runs/normalized
python -m npp plan --project my-project/project.yaml --out-dir runs/project --samples 10000
```

`init` writes a commented, runnable example: `project.yaml`, `model.yaml`, `hardware.yaml`, `design.yaml` and numeric `weights.npz`. It checks target collisions before writing and requires explicit `--overwrite` to replace existing files. These generated values are synthetic demonstrations, not inferred hardware measurements.

`inspect-weights` returns a human-readable table by default; agents should pass `--json`. Its output contains `format`, `tensors` (each with `name`, `shape`, `dtype`) and `note`. Bind exact tensor names. Do not infer network topology or tensor orientation from names or square matrix dimensions. `--state-dict-key KEY` selects an exact top-level PyTorch checkpoint wrapper; use the corresponding `{file: checkpoint.pt, state_dict_key: KEY}` source in the model specification.

`normalize` writes `network.json`, `library.json` and `request.json` with concrete matrices and canonical units. Inspect these before launching expensive searches, especially when converting Keras `in_out` kernels to the planner's `out_in` convention. `validate`, `normalize`, `plan`, `check` and `simulate` accept either `--project PATH` or the complete `--network`/`--library`/`--request` triple; do not combine both forms. Model and checkpoint paths are anchored to the declaring file.

To lower a supported ONNX graph:

```bash
python -m npp import-model trained.onnx --out model.json --input-min -1 --input-max 1
```

This operation requires explicit finite input bounds. Set the project model path to the emitted JSON, or use `model: {file: trained.onnx, input_bounds: [-1, 1]}` directly in a project file. Import supports only the documented dense-vector subset; unknown operators, shapes or attributes are errors. It is not a general framework-to-hardware compiler. For weights-only PyTorch, SafeTensors, NumPy or Keras/HDF5 files, use tensor inspection plus explicit network structure. No model class, custom Keras layer or arbitrary pickled module is executed.

The friendly syntax, format-specific restrictions and optional dependency extras are described in [INPUT_GUIDE.md](INPUT_GUIDE.md). Software v0.2 lowers to the existing canonical schema `"0.1"`; do not change input `schema_version` to `"0.2"`.

### Canonical input workflow

The original three-input interface remains valid. Each specification may now be JSON or YAML. Validate a complete input set:

```bash
python -m npp validate \
  --network examples/split_recombine.json \
  --library examples/components.json \
  --request examples/request_default.json
```

Generate plans and reports:

```bash
python -m npp plan \
  --network examples/split_recombine.json \
  --library examples/components.json \
  --request examples/request_default.json \
  --out-dir runs/split-recombine \
  --samples 10000
```

The optional `--samples` requests Monte Carlo diagnostics in addition to analytic evaluation. For numerical-budget screening, the analytic metric is authoritative; a favorable Monte Carlo sample cannot override an exceeded analytic bound.

A completed run preserves normalized input snapshots, the search report, standalone plan records, replay-check results and human-readable report/CSV artifacts. Use the actual generated plan filename for subsequent operations; IDs are derived from decisions and should not be guessed.

| Artifact | Contents |
|---|---|
| `inputs/network.json`, `inputs/library.json`, `inputs/request.json` | Normalized snapshots whose hashes bind the plan to its assumptions. |
| `report.json` | Complete search telemetry and retained plan records. |
| `plans/plan-<decision-hash>.json` | One standalone record per retained plan. |
| `checks.json` | Compact replay-check outcomes for the saved plans. |
| `simulations.json` | Sampling results if requested, otherwise an empty object. |
| `pareto.csv`, `report.html` | Comparison table and portable visual report. |
| `manifest.json` | Current-run inputs, software version and listed plan files. |

Output directories must be empty unless `--overwrite` is supplied. Overwriting can leave older, unreferenced files, so use `manifest.json` to identify the current run rather than globbing every historical plan file.

Replay-check a selected record:

```bash
python -m npp check \
  --network runs/split-recombine/inputs/network.json \
  --library runs/split-recombine/inputs/library.json \
  --request runs/split-recombine/inputs/request.json \
  --plan runs/split-recombine/plans/SELECTED_PLAN_ID.json
```

Run independent forward sampling for that decision:

```bash
python -m npp simulate \
  --network runs/split-recombine/inputs/network.json \
  --library runs/split-recombine/inputs/library.json \
  --request runs/split-recombine/inputs/request.json \
  --plan runs/split-recombine/plans/SELECTED_PLAN_ID.json \
  --samples 20000
```

Discover command-specific options using `--help`. The `demo` command produces a reproducible demonstration from bundled examples; `schemas` exports the versioned JSON input and output schemas. Static schema files are also available under `schemas/`.

```bash
python -m npp demo --out-dir runs/demo --samples 10000
python -m npp schemas --out-dir runs/schemas
```

Command results are JSON on standard output, except `inspect-weights` uses a table unless `--json` is supplied. Help/version output also uses ordinary text. Exit codes are `0` for a successful command or a plan run with feasible results, `2` for invalid input or failed replay checking, `3` for model-relative infeasibility, and `4` for an incomplete search that found no feasible plan. A sampling inconsistency is recorded in the returned JSON; agents must inspect it rather than rely only on the process exit code. `simulate --seed N` overrides the request seed for that invocation.

## Python API

The planner accepts normalized dictionaries. Always validate user-supplied inputs before calling it.

For project files and friendly specifications, use the configuration loader, which also performs canonical validation:

```python
from npp.config import load_inputs
from npp.weights import inspect_weights
from npp.templates import create_project

created = create_project("my-project")
metadata = inspect_weights("my-project/weights.npz")
network, library, request = load_inputs(project_path="my-project/project.yaml")
# Alternatively: load_inputs(network_path=..., library_path=..., request_path=...)
```

For already canonical JSON files, the original API remains available:

```python
from npp.models import InputValidationError, load_json, validate_inputs
from npp.planner import plan
from npp.checker import check_record
from npp.physics import simulate

try:
    network, library, request = validate_inputs(
        load_json("examples/split_recombine.json"),
        load_json("examples/components.json"),
        load_json("examples/request_default.json"),
    )
except InputValidationError as exc:
    # JSON-serializable records: code, path, message.
    print(exc.diagnostics)
    raise

report = plan(network, library, request)
for candidate in report["plans"]:
    checked = check_record(network, library, request, candidate)
    assert checked["valid"], checked["diagnostics"]
    sampled = simulate(
        network, library, request, candidate["decision"],
        samples=10000, seed=request["seed"],
    )
```

`npp.models.schema_bundle()` returns a dictionary with keys `network`, `library` and `request`. JSON Schema is useful for constructing valid syntax, but is not sufficient for topology, dimensions, semantic capabilities or cross-input references. `validate_inputs` performs those checks, rejects extra fields, rejects nonfinite numeric values and returns normalized defaults.

`load_json` rejects malformed top-level values, duplicate JSON keys and explicit nonfinite constants. Component parameters, rule settings and network weights remain declarative values; loading never evaluates Python or shell expressions.

For controlled experiments, `npp.physics.evaluate(network, library, request, decision)` evaluates a specific decision without search. `outgoing_uses(network)` provides canonical branch ordering and `infer_bounds(network)` supplies nominal interval enclosures.

## Decision format

A decision has two maps:

```json
{
  "components": {
    "x": "optical_encoder",
    "left": "optical_fixed",
    "right": "optical_programmable",
    "sum": "digital_fixed"
  },
  "fanout_rules": {
    "x": "source_boost_2"
  }
}
```

This is a structurally illustrative choice for `split_recombine.json`, not a claim that it is nondominated or feasible under every request. Every node needs one component. Every node with more than one outgoing use needs one rule; nodes without fan-out must not receive one. Component/rule IDs must come from the supplied library.

Agents can propose such decisions directly and receive evaluator diagnostics. To obtain a checked plan artifact with provenance, use the generated plan records or the documented Python record format. Do not fabricate cached metrics, IDs or hashes.

## Diagnostics and interpretation

Treat the structured `code`, `path` and `message` as the explanation of a failure. Typical classes include:

| Diagnostic family | Appropriate response |
|---|---|
| Missing importer dependency | Install the declared optional extra; do not silently switch to another file format or interpretation. |
| Missing tensor, unsupported checkpoint, wrong matrix layout | Inspect exact names/shapes, select the intended state dictionary and confirm orientation; do not guess missing numerical values. |
| Unsupported ONNX operator or shape | Explain the unsupported graph construct or provide a supported export; do not silently delete it. |
| Unit mismatch or unknown friendly configuration key | Correct the units or field name while preserving the requested physical quantity. |
| Schema error, extra field, invalid number | Correct the input representation; do not silently discard unsupported requirements. |
| Missing graph predecessor or dimension mismatch | Correct the graph from the intended NN computation. |
| No compatible macro | Report the missing operation, domain, width or editability capability; a new validated library entry may be needed. |
| Source scaling forbidden | Do not use launch-power adjustment or exact source re-encoding for an unknown intermediate signal. |
| Nominal range/headroom exceeded | Explore applicable alternatives or request a characterized component with greater range. |
| Budget exceeded | Explore other candidates; any user-budget relaxation must be explicit. |
| Stored evaluation/hash mismatch | Treat the plan as changed or inconsistent and recompute from the intended snapshots. |
| Search incomplete with no plan | Report “no feasible plan found within this search,” not “no feasible hardware exists.” |

Use the actual diagnostic codes in returned artifacts; textual families above explain handling rather than freeze every code string.

## What agents may and may not infer

A successful `validate` means inputs are structurally supported. It does not mean there is a feasible plan. A successful replay check means a record agrees with recomputation under the given model. It does not certify a physical device, prove global optimality, validate a component characterization or establish scientific novelty.

The output error is an aggregate RMS model bound. It is not a deterministic maximum, a classification error rate or a guarantee against rare saturation. A Monte Carlo agreement flag is a diagnostic, not independent experimental hardware evidence.

Read both `report.status` and `report.search.complete`. `ok` can describe an incomplete search. Only completed exhaustive enumeration supports a complete frontier claim within the supplied finite choices. Beam results are approximate. Search hashes identify input content; they are not signatures or secure attestations.

An agent must not infer unmodeled properties from component names. In particular, `editable` concerns fixed-shape linear weights only. Parallel branch execution does not establish a throughput bound. The implementation has no general memory architecture, random-access guarantee or arbitrary resource-sharing scheduler.

## Extending the library through an agent

A safe, useful extension workflow is:

1. Propose a new entry with explicit source, units, operating limits and intended fan-out size.
2. Have a domain expert verify characterization and applicability.
3. If the mechanism uses an existing supported kind, add parameter alternatives to a new library version.
4. If the physics differs, implement the new backend semantics, guards and independent forward sampling, and add targeted tests.
5. Regenerate plans and compare with previous artifacts using the same original requirements.

The current schema does not implement a free-form theorem prover or symbolic physical-law engine. Descriptions such as “recover clean signal” or “free copy” have no executable authority. The rule kind and backend code determine the modeled behavior.

## Reproducibility checklist

Preserve the exact three normalized inputs, software version/source, search mode and limits, full report, selected standalone plans, checks and any sampling results. Record external component characterization separately. When comparing strategies, use the same allowed components and modeling assumptions unless the changed assumption is the experiment itself.

The bundled `request_editable.json` targets linear nodes `left` and `right` of `split_recombine.json`. It is not directly applicable to the differently named nodes in `simple_chain.json` or `small_relu.json`. `request_impossible.json` deliberately imposes zero energy to exercise infeasibility reporting.
