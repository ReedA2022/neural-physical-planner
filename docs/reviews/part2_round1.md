# Part 2 independent review — round 1

Date: 2026-10-03 (Europe/Paris)

Decision: **BLOCK. Do not begin Part 3 until the report-layer failures below are fixed and independently rechecked.**

## Reviewed boundary

This gate reviews the explicit, typed implementation graph for one selected neural
node's intensity-encoded fanout distribution. It includes graph generation and
validation, physical-model evaluation, portable replay, the command-line interface,
and offline HTML/SVG inspection. Whole-NN hardware, signed encoding, fabrication,
calibrated devices, independent optical simulation, and designer replanning are not
claimed by this part. The technology remains illustrative and unreviewed by a
photonics expert. Those limitations are explicit rather than concealed failures.

## Reproduced blocking findings

### R2.1 — incomplete or malformed evaluations are presented as complete (medium)

`render_implementation` checks only that metrics are a nonempty dictionary,
ledger length equals instance count, and remaining rows carry `evaluated` status.
That is insufficient for the stated conservative presentation contract.

Starting from a valid two-way realization, each of the following independently
still yields `Recorded feasible within the declared model` and displays a feasible
aggregate:

- Replace every ledger row with the first row, keeping the original length.
- Replace one ledger row's instance ID with a nonexistent ID.
- Remove or duplicate an evaluated receiver row, or change its boundary port to
  a nonexistent instance.
- Replace metrics with `{"arbitrary": 0}`.
- Replace energy with `true`, `NaN`, `Infinity`, a string, or null.

The replay checker correctly rejects these records, and the renderer clearly says
it does not replay them. This is nevertheless a report-completeness bug: obvious
missing coverage and malformed numbers are promoted to a complete feasible report.
A hardware reviewer reading the report should not have to spot missing or duplicated
rows to discover this discrepancy.

Required fix: determine presentation completeness from exact unique instance
coverage, valid matching receiver boundaries/bindings, the required finite numeric
metric fields, and consistent counts. Preserve inspectability and the disclaimer
that rendering is not independent simulation or provenance authentication. Full
physical reevaluation inside the renderer is not required.

### R2.2 — malformed identifiers crash an otherwise inspectable report (medium)

Set `record["technology"]["components"][0]["id"] = ["invalid identifier"]` and call
`render_implementation`. `_display_graph` raises `TypeError: unhashable type: 'list'`
while building its component lookup. The documented malformed/partial-record
inspection behavior is not met.

Required fix: treat malformed/unresolved identifiers as incomplete data, provide
inert readable fallback text/tables, and avoid an unhandled exception. Do not silently
promote the malformed record to success.

### R2.3 — regenerated circuit drawings obscure carrier wiring (medium)

Independent raster inspection of actual two-way and eight-way regenerative graphs
shows long constant-carrier wires passing behind early-detector boxes and then
overlapping the electrical detector-to-modulator connection. The typed connection
table is correct, but the diagram can visually imply that the carrier feeds the
detector. Passive eight-way rendering is readable.

Required fix: reposition or route the fresh carrier sources so their edges visibly
end at the modulator without crossing unrelated component boxes or hiding the
sample connection. Reinspect the actual regenerative drawings after the change.
This is a schematic clarity requirement, not a fabrication-layout requirement.

## Independent verification performed

- Existing implementation, report and physical CLI suites: **32 tests pass**.
- Added `tests/test_implementation_review.py`: **10 independent tests pass**.
  These cover eight-way unequal route losses and energy conservation; five-way
  explicit dumped power; an interior bounded affine node derived from signed input;
  repeated operands and output order; duration and fixed source costs;
  a manually constructed shared-carrier splitter retaining correlated noise;
  early detection failure despite later regeneration; every top-level replay field;
  Boolean/numeric equality and nonfinite/oversized JSON values; strict indices; and
  forbidden electrical sample broadcasting.
- Added `tests/test_report_review.py`: **5 completeness tests reproduce 11 failing subcases and
  1 exception** before fixes. A sixth geometric regression reproduces carrier
  wire intersections with unrelated boxes for regenerative fanout 2, 4 and 8.
  Together these encode R2.1–R2.3.
- Independent CLI smoke passed feasible realization/replay (exit 0), overwrite
  refusal and tampered metric replay (exit 2), and infeasible realization (exit 3).
  A faithfully saved infeasible record correctly passes replay: replay validity
  means consistent reconstruction, not physical feasibility.
- Inspected source-level guards for typed ports, complete connectivity, cycles,
  launch/receiver lineage, unused leaves, encoded-carrier rejection, component
  operating envelopes, source costs, acquisition timing and noise scope.

No blocking defect was found in the core graph generator, physical evaluator or
portable replay checker in this round. First-order covariance and nominal cost
consistency do not establish real-device accuracy. Report corrections and a passing
second gate are required before Part 3.
