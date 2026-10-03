# Implemented multiphysics runtime

This document records new executable capabilities separately from the historical
M0 inventory and proposed experiment registry. See the per-stage records in
`docs/reviews/` for independent judgments. The authoritative acceptance scope is
the supplied specification and `IMPLEMENTATION_ACCEPTANCE.md`.

## M1: contracts and local reference models

The opt-in namespace is `npp mp`; old commands, schemas and saved outputs retain
their existing meaning. No new optional package is required for M1.

```sh
python -m npp mp init my-multiphysics-project
python -m npp mp validate --project my-multiphysics-project/project.yaml
python -m npp mp normalize --project my-multiphysics-project/project.yaml --out-dir portable
python -m npp mp evaluate --project portable/project.json --out-dir digital-evaluation
python -m npp mp check --record digital-evaluation/evaluation.json
python -m npp mp technology --out technology.json
python -m npp mp tile --spec examples/multiphysics/analog-tile.json --out-dir analog-evaluation
```

The starter separates model, workload, technology, environment and constraints.
It references saved NumPy weights by explicit tensor name and orientation. The
same loader supports the previously documented optional formats; it never
infers architecture from a checkpoint. Relative paths resolve from the file
declaring them. Normalized projects embed all weights and parameter evidence.
Explicit quantities accept time, energy, distance and binary memory units.

Common immutable contracts separate carrier from regime, field from intensity,
shape from encoding, and evidence from model fidelity. Unknown quantities remain
unknown, bounded costs remain bounded, and ledger contributions have one owner.
Backend IDs and versions select registered code; technology files cannot execute
equations. Signal/state/resource declarations and context-bound identities are
shared without sharing device equations. Actual mixed graphs and state scheduling
belong to M2, not the M1 local evaluator.

The digital reference executes a complete biased SwiGLU block with independent
scalar reference outputs. Float32/float64 formats declare nearest-even rounding,
non-fused accumulation, overflow rejection and gradual underflow. Dimensions are
at most 256, batch at most 64, and work at most 10 million matrix MACs. Its ledger
includes arithmetic, materialized intermediate SRAM traffic, external weight
loads, local links, configuration, launch and static energy. A serial local
duration is not a multi-device schedule or measured CPU/GPU performance.

The analog family is a differential-conductance matrix primitive with bipolar
input voltages, positive/negative weight rails and explicit DAC/ADC readout.
Conductance and input ranges, finite precision, programming noise, drift, read
history, lumped line loading and temperature limits are declared. A diagnostic
ideal mode checks algebra; it is not a claim of physically infinite precision.
One tile is bounded to 64 rows/columns. The complete analog FFN is a later graph
composition, not a relabeled digital reference.

The example technology pack is wholly hypothetical. Its local active-component
boundary explicitly excludes chip startup, global clock distribution and package
infrastructure. System-level comparisons must add their declared owners or keep
those totals incomplete. Reference values are not measurements or feasibility
certificates. Out-of-envelope inputs fail; unknown cost or unresolved comparisons
remain conditional. Replay is explicitly separate from independent validation.

Track E preserves the ideal supplied function and still measures finite-precision
error. Track A enforces both the model's declared absolute error budget and the
workload's RMS budget. Track R metadata can describe an externally adapted model,
but hardware execution rejects R as unsupported: this runtime has no adaptation
cost ledger and matched adapted-model study. Metadata alone cannot certify that
the adaptation requirement has been met.

Exit codes are 0 for an executed model result (including explicit conditional
results), 2 for invalid/unsupported/numerical errors or failed replay, and 3 for
a resource/quality budget violation. A conditional result carries unresolved
obligations and must not be interpreted as a complete numerical cost estimate.

## M2: composed plans (independently reviewed)

The new compilation command takes implementation choices separately from the
model and technology. The starter writes `implementation.yaml` with one mixed
FFN: photonic up projection, analog gate projection and digital down projection.
The schema supports explicit tiles or ordered input/output cut grids; all matrix
cells must occur exactly once. Bias belongs to the complete operator, and signed
hidden permutations carry a proof record with the corresponding weight/bias
transforms.

`geometry`, `system` and `interfaces` may each be embedded mappings or paths to
separate YAML/JSON files, resolved beside the implementation file. Compilation
embeds those records so later replay does not read external specification files.

```sh
python -m npp mp compile --project my-multiphysics-project/project.yaml --implementation my-multiphysics-project/implementation.yaml --out-dir mixed-plan
python -m npp mp compile --project my-multiphysics-project/project.yaml --kind digital --out-dir digital-plan
python -m npp mp check-implementation --record mixed-plan/implementation.json
python -m npp mp interfaces --out interfaces.json
python -m npp mp compile --project examples/multiphysics/ffn/project-fused.yaml --implementation examples/multiphysics/ffn/fused.yaml --out-dir fused-plan
```

`--kind` selects one explicit reference arrangement: `digital`, `analog`,
`photonic`, `hybrid`, or `fused`. Fused up/down photonic engines need distinct
physical instances; resource capacities are enforced. There is no automatic
search in this command. A plan pins its model, device and interface coefficients,
system assumptions, geometry, resources, schedule and random seed. Replay uses
those embedded snapshots and needs no original checkpoint files.

The compiled bundle contains the complete `implementation.json`, normalized
inputs, semantic plan, typed physical graph, geometry, reservations/state events,
cost ledger and assumptions. `report.html` displays modeled totals, quality and
the actual schedule; its expandable record retains all details. Unknown costs
remain visible. Budget-violating plans are saved for inspection and return exit
code 3; malformed specifications do not create an output bundle.

The classical photonic primitive represents signed coherent amplitudes through a
passive normalized matrix. Digital encoding, sources, reference fields, coherent
bias injection, homodyne detection and ADC readout have separate declared costs.
A field is not a free digital tensor. A signed electro-optic gate uses a bounded
signed transfer; an intensity-only attenuator rejects negative control. The
unfused detection/multiply/re-encoding path is retained as a comparison.

System coefficients live in a versioned `system` section. The named
`hypothetical_system` preset expands every package, global clock, route and
buffer assumption into parameter records with evidence and units. For example:

```yaml
system:
  preset: hypothetical_system
  overrides:
    clock_startup_ns: 20 ns
    package_static_power_mw:
      state: unavailable
      unit: mW
      reason: Package power has not been characterized.
```

This makes the total conditional; it does not grant a free package. Device-local
and system-level owners remain distinct. Static support energy is charged over
elapsed schedule time, including waiting. Geometry is an illustrative placement
and interconnect skeleton, not fabrication layout. These models are deliberately
bounded and conservative; a software review does not establish calibrated device
physics or a hardware advantage.

Shared fixed analog/photonic arrays are sized to the largest row and column
requirements of their assigned operations. Each operation executes the full
zero-padded physical array; unused analog cells retain their modeled conductance
and loading. Digital padding and output extraction are explicit paid operations.
Records expose active versus installed dimensions and utilization. The shared
electrical bus's route/port capacity limits the schedule even when more logical
link capacity is available in the project constraints.
