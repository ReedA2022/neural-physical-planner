# Explicit optical fanout implementation

The v0.3 backend expands **one selected NN node's fanout** into inspectable physical
instances, ports, connections, receiver boundaries and unused-port terminations.
It does not implement the selected NN operation or its consumers. This is a bounded
physical distribution subcircuit, separate from the v0.2 neural macro planner.

The bundled technology remains illustrative and unreviewed by a photonics expert.
Passing the structural checker, component guards or replay checker establishes
consistency under the declared model, not measured-device accuracy or fabrication
readiness. See [TECHNOLOGY.md](TECHNOLOGY.md) for the component equations and evidence.

## Run the example

From the repository directory:

```bash
npp realize --network examples/physical/network.json \
  --spec examples/physical/realization.yaml --out-dir runs/physical
```

Use `--technology your-technology.yaml` to supply another strict pack. Alternatively,
`--project your-project.yaml` loads a friendly project and uses its normalized NN.
The existing macro hardware library in that project is not a physical technology
pack and does not override `--technology`. The command exports a portable record,
input snapshots and an inspection report. It refuses existing output files unless
`--overwrite` is supplied. Exit codes distinguish feasible (0), infeasible (3),
and invalid input (2).

The example selects the two-dimensional input `x`. `x` occurs twice in the `add`
operator and once as an NN output, so it requires **three uses per scalar lane**.
Repeated operands and repeated NN output occurrences are distinct sinks; none are
silently deduplicated. The separate `add` output need not fit the optical encoding
range because this subcircuit realizes only `x`'s distribution.

Minimal specification:

```yaml
schema_version: "0.3"
source_node: x
recipe: passive
source_power_mw: 1.0
lengths_um: [1000.0, 2000.0, 3000.0]
operating:
  wavelength_nm: 1550.0
  temperature_c: 25.0
  symbol_duration_ns: 1.0
```

`lengths_um` is one nonnegative number broadcast to every route, or one length per
ordered NN use: first consumer nodes in network order and their input positions,
then network output occurrences in output order. The same ordered route lengths
apply to each scalar lane. Default route length is 1000 µm; default operation is
1550 nm, 25 °C and a 1 ns rectangular symbol. These low-level unit-bearing fields
accept finite numbers, not strings such as `"1 ns"`. Unknown fields are errors.

The source's inferred nominal bounds must lie wholly inside `[0,1]` on every lane.
Signed values and values greater than one are rejected. The compiler does not
silently clip, translate, normalize or rescale the NN. A user may choose a suitable
bounded node or supply a separately specified encoder outside this backend.

Component choices default to IDs `source`, `splitter`, `waveguide`, `detector` and
`modulator`; another pack can override these IDs via the `components` mapping. The
selected components must have the corresponding kinds. Tree synthesis currently
requires a declared **fixed 50:50 splitter**, not a tunable-ratio device.

## Two explicit recipes

**Passive.** For each scalar lane, create one externally driven encoded launch,
a balanced binary splitter tree, one waveguide per NN use and one detector per use.
For fanout `f`, the tree has depth `ceil(log2(f))`, `2^depth - 1` splitters and
`2^depth` leaves. Excess leaves terminate at explicit ideal matched external
boundaries. Their full-scale power is reported and is never redistributed among
used leaves. The physical termination device and its packaging area/cost are
outside this model; the graph does not imply they are fabricated for free.

**Regenerate.** Use `recipe: regenerate` and supply `regeneration_power_mw`. Split
the initially encoded signal first, then give each used branch its own early
detector, fresh constant-carrier source and modulator before its long waveguide
and final receiver. The detector output drives exactly one modulator. The carrier
port must receive an unencoded constant optical carrier, while the sample port
must receive the matching encoded NN sample's electrical descendant. Feeding an
encoded copy into the carrier would multiply data, so the checker rejects it.

```yaml
source_node: x
recipe: regenerate
source_power_mw: 1.0
regeneration_power_mw: 0.5
lengths_um: [1000.0, 2000.0, 3000.0]
```

Regeneration adds energy, area, latency and independent noise. It does not erase
upstream noise or manufacture a missing sample. In particular, an early receiver
below its full-scale sensitivity makes the regenerative candidate infeasible.

Each generated source has an explicit role. `encoded_launch` binds to the selected
NN node and scalar lane at the externally encoded boundary. `constant_carrier`
is an independent power supply for reencoding, without an NN data binding.
Source wall-plug energy is charged to every physical source instance.

## Graph and checks

The graph embeds its canonical NN, selected source, recipe and operating point.
Instances name technology components; the technology's named ports authoritatively
define direction, domain, quantity, encoding and connection cardinality. Connections
use explicit `{instance, port}` endpoints. The graph lists launch boundaries,
receivers with `{source_node, lane, use_index}` bindings, and matched terminations.

`validate_implementation` checks unique IDs, component kinds/parameters, every
port, input completeness, output use or termination, one connection per port,
matching physical types, acyclicity and exact sample lineage. There must be exactly
one encoded launch per lane and exactly one receiver for every ordered NN use and
lane. The graph cannot smuggle in unlimited electrical broadcast or multiply
samples through an encoded carrier. It must use the selected recipe's number of
regeneration stages. It cannot realize arbitrary signed or unbounded NN outputs.

Bounds for this backend are at most 64 uses, 64 scalar lanes, 256 scalar receivers
and 4096 physical instances. NN context is limited to 1024 nodes and 100000 total
scalar node values. Larger networks need an explicitly partitioned workflow;
limits prevent accidental graph and covariance expansion.

Structural validity is separate from physical feasibility. `evaluate_implementation`
checks each instance's operating envelope, source range, optical input ceiling,
waveguide length and detector full-scale sensitivity. Failed instances have
structured diagnostics and downstream instances are marked blocked. Aggregate
metrics are null when evaluation is incomplete; a partial ledger is never presented
as a full cost. No guard silently clamps an operating point to make it pass.

## Cost, timing and noise interpretation

The resource ledger charges every instantiated component, including per-lane trees
and per-branch regenerative sources. Energy is electrical pJ per full-scale symbol
under the nominal encoding model. Waveguide area is declared fixed area plus core
width times length; bends, spacing, control, memory, consumer computation and
external termination devices are excluded. These metrics are not added to the
legacy macro estimates, which have different boundaries and could double-count.

Latency means **complete-symbol availability**, measured from the initial launch
start. An encoded launch completes after its fixed overhead plus one symbol
duration `T`; passive propagation and receiver overheads add to that time. A
regenerative modulator waits for its early receiver's complete sample and its
carrier to be available, then adds its own overhead and another `T`. A final
receiver adds its overhead; it does not count the same optical integration interval
a second time. With the reference pack, a passive branch is ready at
`T + 0.3 ns + route delay`, and a regenerative branch at
`2*T + 0.7 ns + route delay` for the usual non-blocking carrier setup.

Carrier fixed startup availability runs in parallel with the initial stage; the
source is scheduled for one symbol when reencoding occurs. Continuous standby,
turn-on transients, bandwidth waveform distortion and steady-state throughput are
not modeled. Every such assumption appears in the record's scope.

Each independent normalized noise source has a unique loading identifier.
The initial source's full-scale relative-intensity noise loading is shared across
all its fanout branches. Separate scalar lanes use separate physical sources.
Each detector adds its own shot/electronic contribution. A regenerated sample
retains all upstream loadings and adds the fresh carrier's relative-intensity noise
and the modulator's added sample noise. Thus off-diagonal receiver covariance
preserves shared ancestry instead of treating all output errors as independent.

This is a **first-order, full-scale noise estimate**. Signal/carrier noise products,
nonlinear transfer, saturation and clipping tails are excluded. Shot noise depends
on signal level; the reported receiver formula uses the full-scale operating point.
No claim is made that the resulting covariance certifies NN error, physical clipping
absence or inference accuracy.

`noise_rms_estimate` in aggregate metrics is the largest scalar receiver RMS.
`stacked_noise_rms_estimate` is the square root of the sum of scalar receiver
variances. `receiver_covariance_estimate` uses the exact order of the `receivers`
list. Receiver rows identify their NN consumer/input position and scalar lane,
full-scale power, sensitivity/overload margins and complete-symbol ready time.

## Python and portable replay

```python
from npp.config import read_spec
from npp.technology import load_technology
from npp.implementation import realize, check_realization

record = realize(read_spec("examples/physical/network.json"),
                 load_technology(),
                 read_spec("examples/physical/realization.yaml"))
assert record["evaluation"]["feasible"]
assert check_realization(record)["valid"]
```

The same replay is available as `npp check-realization --realization runs/physical/realization.json`.

The record stores normalized network/technology/spec snapshots, generated graph,
SHA256 content hashes and evaluation. `check_realization(record)` regenerates the
graph from embedded inputs, reevaluates it and checks the complete record. It detects
changed costs, receiver bindings, graph topology or hashes even if a changed graph's
hash was recomputed. These hashes bind content; they do not authenticate measurement
sources or a reviewer's identity. No original weight files are required for replay.

Individual APIs are `validate_realization_spec`, `build_implementation`,
`validate_implementation`, `evaluate_implementation`, `realize` and
`check_realization`. `realization_schema()` and `implementation_schema()` expose
JSON Schema for editors; graph/NN cross-field checks also run in Python.
