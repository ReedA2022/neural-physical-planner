# Friendly project files and saved weights

You can keep learned tensors in their original checkpoint file and describe the network and design choices in short, commented YAML files. The loader converts these files into the same validated graph, library and request used by the planner. Existing JSON inputs remain supported.

Weight import does not expand the physical backend: it still supports bounded vector inputs, dense linear maps, addition, identity and ReLU. Loading a large checkpoint does not make an arbitrary CNN or Transformer graph compilable.

## Start with a complete working project

From the repository root after installation:

```bash
npp init my-project
npp inspect-weights my-project/weights.npz
npp validate --project my-project/project.yaml
npp plan --project my-project/project.yaml --out-dir runs/my-project --samples 10000
```

Open `runs/my-project/report.html`. `npp init` creates five files:

| File | What you normally edit |
|---|---|
| `project.yaml` | Paths to the other three specifications |
| `model.yaml` | Checkpoint path, tensor names, layer order and input bounds |
| `weights.npz` | Demonstration tensors; replace this with your own checkpoint |
| `hardware.yaml` | Component/rule library and physical parameter overrides |
| `design.yaml` | Objectives, hard limits and required capabilities |

The generated network weights and hardware parameters are **synthetic examples**. Hardware predictions only become meaningful when the component parameters and physical model match the device family being studied.

Initialization checks all target files before writing. It refuses to replace existing project files unless `--overwrite` is explicit. Unrelated files are preserved. Existing target directories and symlinks are rejected.

## Connect the files

The generated `project.yaml` is deliberately short:

```yaml
name: My neural hardware project
model: model.yaml
hardware: hardware.yaml
design: design.yaml
```

Paths in the project file resolve relative to that file. A checkpoint path in the model specification resolves relative to the model specification, so running the command from a different working directory does not change which checkpoint is loaded.

Both JSON and YAML are accepted for specifications. YAML supports explanatory comments; duplicate mapping keys and unknown configuration fields are rejected rather than silently ignored.

## Attach learned weights to a network

For a PyTorch-style dense network saved with matrices shaped `[output_size, input_size]`:

```yaml
name: My trained dense network
weights: trained-model.safetensors  # Also .pt, .pth, .bin, .npz, .npy, .keras or .h5
layout: out_in

input:
  name: x
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

`weights` and `bias` inside a layer refer to exact tensor names, not filenames. Use `npp inspect-weights FILE` to list available names and shapes. Layer widths are inferred from the matrices; incompatible dimensions produce an error. An omitted `weights` reference defaults to `LAYER_NAME.weight`. For an omitted bias, a `PREFIX.weight` reference looks for `PREFIX.bias`; other references look for `LAYER_NAME.bias`, always in the same declared tensor source. If that key is absent the bias is zero. Set `bias: null` explicitly when you want zeros even if a conventional bias tensor exists. Explicit tensor names, as shown above, are recommended for Keras and make every binding easier to review. Sequential `layers` support `linear`, `relu` and `identity`; addition and branching use an explicit graph.

**A weights-only file does not specify the activation graph.** The importer does not guess ReLU placement or recover branching from parameter names. Write the structure explicitly, or use an ONNX graph that fits the supported operator subset.

Bounds describe the numerical values presented to the neural program after preprocessing. For normalized input coordinates in `[-1, 1]`, use `bounds: [-1, 1]`. They are required for nominal range checks; they are not inferred from trained weights or treated as a training dataset.

For different bounds on each coordinate, use:

```yaml
input:
  name: x
  size: 2
  bounds:
    lower: [-1, 0]
    upper: [1, 2]
```

### Supported tensor containers

| Format | Supported content | Install extra | Important limit |
|---|---|---|---|
| NumPy `.npz` | Named numeric arrays | None | Object arrays are rejected; the ZIP archive must contain arrays, not serialized Python objects. |
| NumPy `.npy` | One numeric array, exposed as `array` | None | Use `weights: array` for its layer reference. A separate bias can be inline in the specification. |
| SafeTensors `.safetensors` | Named tensors | `safetensors` | A tensor container does not supply layer order or activation semantics. |
| PyTorch `.pt`, `.pth`, `.bin` | Tensor state dictionaries | `pytorch` | Restricted `weights_only=True` loading; arbitrary pickled models and TorchScript archives are not supported. |
| HDF5 `.h5`, `.hdf5` | Numeric tensor datasets | `hdf5` | Use the exact dataset path reported by inspection. Reading datasets does not recover the Keras model architecture. |
| Keras `.keras` | Numeric tensors from the embedded HDF5 weights | `hdf5` | Tensor extraction only; no Keras model deserialization or custom-layer execution. |
| ONNX `.onnx` | Named dense initializers | `onnx` | Tensor inspection alone does not lower the graph; use `import-model` for the supported graph subset. |

Install only the format you need, from the source checkout:

```bash
python -m pip install '.[safetensors]'
python -m pip install '.[pytorch]'
python -m pip install '.[hdf5]'
python -m pip install '.[onnx]'
```

Or install `'.[all]'` for all optional importers. PyTorch can be a large dependency; it is not required for NumPy, SafeTensors, Keras/HDF5 or ONNX import.

Numeric tensors are converted to the planner's numerical representation and checked for supported shape and finite values. Import is inference-oriented: it does not resume training, use optimizer states, execute custom model classes or train missing weights.

The planner evaluates the imported parameters in float64. It preserves the supported mathematical operations, but does not reproduce the original framework's float16/float32 rounding, device kernels or quantization behavior. Framework comparisons therefore use appropriate numerical tolerances; bit-for-bit framework inference is not promised.

The PyTorch importer reads on the CPU and automatically recognizes a plain state dictionary or a checkpoint containing `state_dict` or `model_state_dict`. If both wrapper keys exist, select one explicitly when inspecting:

```bash
npp inspect-weights checkpoint.pth --state-dict-key model_state_dict
```

Use the same explicit selection in `model.yaml`:

```yaml
weights:
  file: checkpoint.pth
  state_dict_key: model_state_dict
```

HDF5 inspection lists numeric datasets, which can include optimizer variables. Bind the network's learned tensors explicitly; do not select datasets solely because their shapes match. The saved-tensor loader limits file size and expanded tensor data to 512 MiB and the tensor count to 10,000. Those limits do not describe a total process-memory guarantee or the separate ONNX graph-import path. External/soft HDF5 links, external/virtual datasets and ONNX external initializers are rejected: export a self-contained file. Sparse and quantized PyTorch tensors must be exported as dense, dequantized tensors.

`inspect-weights` prints a readable table by default. Add `--json` for machine-readable names, shapes, dtypes and format metadata; it does not print the learned numerical arrays.

### Keras matrix orientation must be explicit

Keras Dense kernels usually have shape `[input_size, output_size]`; the planner uses `[output_size, input_size]`. Set `layout: in_out` when your selected tensors follow the Keras convention:

```yaml
weights: trained.keras
layout: in_out
input: {name: x, size: 2, bounds: [-1, 1]}
layers:
  - name: dense
    type: linear
    weights: layers/dense/vars/0  # Example only: copy the actual inspected name.
    bias: layers/dense/vars/1
```

Do not infer orientation from matching dimensions. For the square kernel `[[1, 2], [3, 4]]`, `in_out` explicitly yields planner weights `[[1, 3], [2, 4]]`. Both shapes are `2 × 2`, but they implement different functions without that transpose. Checkpoint suffixes do not automatically select an orientation.

An individual layer may override the top-level `layout` when tensors in the same checkpoint use different conventions.

### Branching and multiple parameter files

For branching, retain the canonical `nodes`/`outputs` graph structure from [SYSTEM_SPEC.md](SYSTEM_SPEC.md). It can also use external tensor references instead of embedded matrices:

```yaml
name: Residual dense example
weights: residual.npz
layout: out_in
nodes:
  - id: x
    op: input
    size: 2
    input_bounds: {lower: [-1, -1], upper: [1, 1]}
  - id: transformed
    op: linear
    inputs: [x]
    size: 2
    weights: projection.weight
    bias: projection.bias
  - id: residual
    op: add
    inputs: [x, transformed]
    size: 2
outputs: [residual]
```

This example needs a `residual.npz` containing `projection.weight` with shape `[2, 2]` and `projection.bias` with shape `[2]`. Explicit graph nodes retain their declared sizes and topological order. Unlike friendly sequential layers, a canonical graph node with no bias field retains the original contract's zero-bias behavior. A tensor reference may also specify its own source file:

```yaml
weights: {file: encoder.safetensors, tensor: projection.weight, layout: out_in}
bias: {file: biases.npz, tensor: projection.bias}
```

These last two fields belong inside a linear layer or node. Relative source paths are anchored to the containing model specification. `state_dict_key` is also accepted in such a reference for PyTorch checkpoint wrappers.

## Import a supported ONNX graph

ONNX contains a graph as well as parameters, so the importer can recover supported structure:

```bash
npp import-model trained.onnx --out model.json --input-min -1 --input-max 1
```

Then set `model: model.json` in `project.yaml`. `--input-min` and `--input-max` specify a common finite bound for each input coordinate; edit the normalized graph if coordinates have different bounds.

Alternatively, reference ONNX directly from the project:

```yaml
name: Imported dense network
model: {file: trained.onnx, input_bounds: [-1, 1]}
hardware: hardware.yaml
design: design.yaml
```

The importer supports the implemented dense-vector subset of **Gemm, MatMul, Add, Relu and Identity**, plus numeric **Constant** nodes. It accepts one floating-point input shaped `[features]` or `[1, features]`; an unspecified or symbolic batch dimension is interpreted as one sample. Feature widths must be fixed. MatMul requires constant right-hand weights, Gemm requires batch-one input with `transA=0`, and dynamic Add operands must have identical shapes. Supported constant bias broadcasts are explicitly lowered. Custom operators/functions, sparse initializers and externally stored tensors are unsupported.

Unsupported nodes fail explicitly. The importer does not silently remove convolution, normalization, attention, reshaping or other operators to make a graph load. See diagnostics for the offending node/operator. A dynamic batch dimension does not provide a batching or throughput model.

Importing an ONNX file is not a promise that every graph in the ONNX ecosystem is supported. Large network planning also remains limited by exhaustive search and dense covariance costs.

## Describe objectives and constraints in ordinary units

```yaml
minimize: [energy, latency, error]
limits:
  error: 0.1
  latency: 20 ns
  energy: 50 pJ
  area: 1000 um2

editable: false
domains: [digital, optical]
serialization: false
search: exhaustive
seed: 7
```

The objectives are separate from the hard limits. Including `energy` in `minimize` asks the planner to retain nondominated energy trade-offs. Adding `limits.energy` rejects candidates above that budget. The planner does not hide the trade-offs inside a guessed weighted sum.

| Friendly name | Canonical metric | Meaning |
|---|---|---|
| `energy` | `energy_pj` | Predicted energy per inference |
| `latency` | `latency_ns` | Predicted elapsed time for one inference |
| `area` | `area_um2` | Sum of modeled macro areas |
| `error` | `error_rms_bound` | RMS bound on the stacked output-vector error |

Bare numbers use the canonical units above. Unit-bearing strings make physical quantities explicit; error remains a dimensionless normalized numerical value. Error is neither classification error nor a per-coordinate absolute worst-case guarantee.

Supported latency units are `ps`, `ns`, `us`/`µs`, `ms` and `s`; energy units are `fJ`, `pJ`, `nJ`, `uJ`/`µJ`, `mJ` and `J`. Area supports squared length units from `nm2` to `m2`, including `um2`/`µm2`; `^2` and `²` spellings are also accepted. Wavelengths in hardware entries accept `nm`, `um`/`µm`, `mm` and `m`. Unsupported or dimensionally inappropriate units fail explicitly.

`editable: true` requires editable implementations for every linear node. `editable: [hidden]` requires editability only for the named linear node. `editable: false` means no editability constraint; it does not force all weights to be immutable.

`domains: [digital]` restricts placement to digital components. `serialization: false` excludes serial re-encoding rules. `serialization: true` merely permits those rules; it does not force sequential execution of the whole network.

For larger finite choice spaces:

```yaml
search:
  mode: beam
  max_evaluations: 10000
  beam_width: 64
```

The report states whether the search was complete. Beam search is heuristic; its returned plans are not a globally certified Pareto frontier.

## Customize a hardware library

The generated hardware file extends the installed synthetic example and overrides selected fields:

```yaml
# SYNTHETIC demonstration assumptions: replace with characterized values.
extends: example
overrides:
  components:
    optical_fixed:
      latency: 0.5 ns
      energy: 0.2 pJ
      area: 8 um2
```

An override names an existing component ID. A typo must not silently create a new hardware component. You can instead supply a full library specification using the canonical library fields documented in [SYSTEM_SPEC.md](SYSTEM_SPEC.md). Shorter files do not remove the need to provide scientifically justified provenance, capacities, dynamic range, efficiencies and noise assumptions.

New numerical instances of supported component and rule kinds can be supplied as data. New physical mechanisms still require implemented semantics and validation code. A YAML description or an LLM-proposed rule is not executable physics.

## Inspect the exact inputs used by the planner

```bash
npp normalize --project my-project/project.yaml --out-dir normalized-inputs
```

Normalization produces self-contained JSON inputs with concrete matrices, resolved defaults and canonical units. This is the best place to verify matrix orientation, tensor selection, inferred widths and physical units before a long search. Planning also saves normalized snapshots under the run's `inputs/` directory for subsequent checking and simulation.

Equivalent decimal unit strings normalize identically, including small quantities such as `9 fJ` and `0.009 pJ`. Explicit quantity strings whose nonzero values would underflow to zero are rejected; true zero and representable subnormal values remain valid parser inputs. Plain JSON/YAML numeric literals and Python floats use ordinary binary64 decoding, which can round a sufficiently small literal to zero before validation. Use quantity strings when this distinction matters. Evaluation may reject accepted small values if later arithmetic cannot preserve their range.

Small YAML aliases are supported, but recursive aliases and repeated aliases introducing more than 1,000,000 expanded values and containers are rejected explicitly. This alias limit does not restrict plain JSON/YAML trees or portable records without aliases. Put large parameter arrays in a supported weight file instead of repeating them through YAML aliases.

The original three-file interface continues to work, including YAML specifications:

```bash
npp plan --network my-project/model.yaml --library my-project/hardware.yaml \
  --request my-project/design.yaml --out-dir runs/three-files
```

Check a saved plan against the saved normalized snapshots, which do not need the original checkpoint:

```bash
npp check --network runs/my-project/inputs/network.json \
  --library runs/my-project/inputs/library.json \
  --request runs/my-project/inputs/request.json \
  --plan runs/my-project/plans/PLAN_FILENAME.json
```

Replace `PLAN_FILENAME.json` with an actual path from that run's `manifest.json`.

## Resolve common input errors

| Error | What to check |
|---|---|
| Missing optional dependency | Install the matching extra above in the same Python environment as `npp`. |
| Tensor name not found | Run `inspect-weights` and copy an exact tensor key, including dots or dataset path separators. |
| Weight dimensions disagree | Check input width, layer order and explicit `layout`; square matrices can conceal an orientation mistake. |
| Unsupported PyTorch checkpoint | Export a plain tensor `state_dict`; do not disable restricted loading or substitute an arbitrary pickle loader. |
| Unsupported ONNX operator or shape | Export a supported dense-vector submodel or add a reviewed lowering implementation; import will not silently approximate it. |
| Invalid YAML field or duplicate key | Fix spelling/indentation; defaults do not justify ignoring unknown instructions. |
| Valid input but no feasible plans | Inspect rejection diagnostics: hardware capacity, range, required editability or budgets may exclude every candidate. |
| Output directory already exists | Choose another output directory, or deliberately pass `--overwrite`. |

Validation distinguishes malformed input from an otherwise valid but physically infeasible request. Successful import verifies a representation; it does not establish the accuracy of supplied hardware measurements or the suitability of the Gaussian surrogate for a real device.
