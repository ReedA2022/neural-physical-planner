# Designer constraints and physical replanning

The designer explores a finite grid of the physical fanout circuits described in
[IMPLEMENTATION.md](IMPLEMENTATION.md). It produces alternatives, a Pareto set,
explicit rejection reasons and portable records. It distributes one selected NN
node's values; it does not synthesize the NN's operators or generate a layout.
The reference technology remains illustrative and uncalibrated.

## Run the example

From the repository after `python -m pip install .`:

```sh
npp plan-physical --network examples/physical/fanout4.json \
  --design examples/physical/design.yaml --out-dir runs/physical-design
npp check-physical-design --result runs/physical-design/design-result.json
```

Open `runs/physical-design/report.html`. It compares energy, complete-symbol
latency, area and normalized full-scale noise. Each retained candidate links to
its circuit report and `realization.json`. An energy/latency projection is only
a projection of the selected objectives; points can overlap or differ in noise.
Use `--project your-project.yaml` instead of `--network` to load an existing
friendly project with external weights. Physical component models come from
`--technology pack.yaml`, or the bundled reference, separately from macro models.

To preserve a baseline's launch power while redesigning its distribution:

```sh
npp realize --network examples/physical/fanout4.json \
  --spec examples/physical/baseline.yaml --out-dir runs/physical-baseline
# Exit 3 is expected: this baseline is physically infeasible but replayable.
npp replan-physical --baseline runs/physical-baseline/realization.json \
  --design examples/physical/replan.yaml --out-dir runs/physical-replan
npp check-physical-design --result runs/physical-replan/design-result.json
```

This example locks the launch at 50 microW and distributes it over four 10 mm
routes. Passive distribution misses the detector sensitivity. Regeneration
detects each branch before its long route, then pays for a fresh carrier and
modulator. Under the reference assumptions, 100 microW carriers yield 3.45 pJ
per full-scale symbol, and 200 microW carriers yield 5.45 pJ with lower estimated
noise. Both take about 2.8334 ns and occupy 26260 square micrometres in this model.
Unlocking the launch permits a 100 microW passive alternative at 0.9 pJ and
about 1.4334 ns. These are modeled subcircuit costs, not measured performance.

## Specify the search

The commented examples contain every common field. A minimal design is:

```yaml
source_node: x
grid:
  recipes: [passive, regenerate]
  source_powers_mw: [50 uW, 100 uW, 200 uW]
  regeneration_powers_mw: [100 uW, 200 uW]
  route_lengths_um: [10 mm]
objectives: [energy_pj, noise_rms_estimate]
constraints:
  max_energy_pj: 6 pJ
  min_receiver_margin_mw: 0 uW
locks:
  source_power_mw: 50 uW
forbidden_recipes: []
max_evaluations: 128
```

Numbers use the units in their field names. Power, length, wavelength, time,
energy, area and Celsius temperature also accept explicit unit strings. `u`,
`µ` and `μ` prefixes are accepted, as are `um2`, `µm²` and `um^2` for area.
Noise is unitless; it is not an NN error bound. Unknown fields, wrong units,
booleans as numbers, nonfinite values and duplicate normalized choices fail
validation. `npp schemas --out-dir schemas-copy` exports the canonical numeric
`physical-design.schema.json`; unit normalization precedes that schema.

Each entry of `route_lengths_um` is a candidate route option: either one length
for all uses or a list with one length per ordered use. Thus `[1 mm, 2 mm]`
means two uniform options; `[[1 mm, 2 mm, 3 mm, 4 mm]]` means one asymmetric
four-use option. The same option applies to every scalar lane. The software
does not infer routing geometry or make long routes shorter automatically.

`grid.components` maps each component kind to a list of technology IDs. Defaults
are the reference IDs `source`, `splitter`, `waveguide`, `detector`, `modulator`.
One selected ID per kind applies to every instance of that kind. Modulator and
regeneration-power alternatives expand the regenerative recipe only. Other
packs must supply valid corresponding IDs. Unsupported physical recipes remain
unsupported; the search does not invent device models.

## Locks, requirements and explanations

| Field | Meaning |
| --- | --- |
| `locks.recipe` | Require `passive`, `regenerate`, or the baseline recipe |
| `locks.source_power_mw` | Require a launch power or the baseline value |
| `locks.regeneration_power_mw` | Require a carrier power or baseline value; a passive baseline's null value permits passive candidates only |
| `locks.instances` | Map a stable physical instance ID to a component ID or `baseline`; require its presence and constrain its global component family |
| `forbidden_recipes` | Exclude named supported recipes |
| `constraints.max_energy_pj`, `max_latency_ns`, `max_area_um2`, `max_noise_rms_estimate` | Upper bounds on the corresponding physical metric |
| `constraints.min_receiver_margin_mw` | Minimum sensitivity margin at every final receiver; all early detectors still obey technology guards |

For example, `lane0.use0.modulator: modulator` requires that regeneration
instance to exist. `lane0.use0.receiver: baseline` preserves its baseline
component ID. These are constraints on the declared global family choices,
not independent component overrides. Locks filter the submitted grid; they do
not add missing choices. A locked value absent from the grid can exclude every
candidate, with explicit reasons. The literal `baseline` requires a saved,
replayable realization and, for instance locks, that instance in the baseline.

Replanning requires the same canonical NN and selected source. An optional new
technology pack or operating point is allowed, but comparisons flag the changed
assumptions and do not label the deltas as like-for-like design improvements.
An instance-ID lock preserves the component identifier, not an old calibration
snapshot. Metric deltas are unavailable when either realization is infeasible.

Each evaluated candidate records physical guards or failed numeric requirements,
the tightest evaluated final receiver margin, and the five largest per-instance
energy/area cost contributors. An infeasible circuit can have a partial ledger;
its aggregate costs remain null. `single_constraint_relaxations` names only
evaluated, physically feasible candidates that violate exactly one numeric
requirement. Their actual value, limit and required relaxation are recorded.
There is no claim of a globally minimal relaxation or an untested repair.

## Completeness, portability and resource limits

All selected objectives are minimized without a weighted score. A candidate is
dominated only if another evaluated feasible candidate is no worse on every
selected axis and strictly better on at least one. Equal objective values may
retain distinct implementations. Enumeration and candidate IDs are deterministic
for the same normalized input ordering; IDs are local to that submitted grid.

`search` distinguishes submitted, excluded, evaluated, unevaluated, feasible and
Pareto counts. Exclusions need no physics run because they already violate a
hard choice constraint. When `max_evaluations` is reached, remaining eligible
candidates stay explicitly unevaluated. Retained plans are then nondominated
only in the evaluated subset; an untested candidate could dominate them.

Limits are 4096 submitted combinations, 128 physical evaluations, bounded
instance/covariance work and serialized input/output budgets. Large requests
fail with structured diagnostics; reduce the grid or evaluation budget. These
bounds also protect replay and report generation. The physical graph's own
fanout, lane and operating limits remain in force.

`design-result.json` embeds normalized network, technology, design and optional
baseline inputs, their hashes, every candidate summary and all retained physical
realizations. Original weight files and baseline paths are no longer needed.
`check-physical-design` reruns the bounded search and checks every saved claim,
including exclusions, costs, Pareto membership and explanations. The comparison
report performs that same replay before displaying success claims or links.
Replay binds contents and verifies this model's computation; it does not
authenticate authorship, run SAX or establish physical calibration.

Exit codes are 0 when plans exist, 3 for a complete grid without a feasible plan,
4 for an unfinished search with no feasible plan yet, and 2 for invalid input.
A successful exit can still have `search.complete=false`; always inspect it.
Existing output directories are preserved unless `--overwrite` is supplied.
Use [SIMULATION.md](SIMULATION.md) to externally check a retained realization.

Python API:

```python
from npp.designer import plan_physical, check_physical_design, validate_design_spec
result = plan_physical(network, technology, design, baseline=optional_baseline)
assert check_physical_design(result)["valid"]
```
