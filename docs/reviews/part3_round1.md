# Part 3 independent review — round 1

Date: 2026-10-03 (UTC)

Decision: **BLOCK. Do not begin Part 4 until the optical report contract defects
below are fixed and independently retested.**

## Blocking findings

### R3.1 — Unsupported tolerances can receive an agreement label

`render_optical_simulation()` checks nonnegative finite tolerances and comparison
arithmetic but does not enforce the simulator's acceptance limits. Starting with
a complete stored-result fixture, set `rtol=1`, double every SAX power, and update
the recorded error, allowance and summary arithmetic. The report then displays
“Recorded optical-power agreement within tolerance” despite a 100% power
discrepancy and a tolerance that `simulate_realization()` rejects. The analogous
absolute-tolerance ceiling must also be enforced.

Required fix: presentation consistency must reject tolerances outside the
declared adapter contract (`rtol <= 0.01`, `atol_mw <= 1e-4`). Retain the raw record
for inspection and mark its status incomplete or inconsistent. This requires no
optical simulation or claim of authentication.

### R3.2 — Incomplete or unsupported exports can receive an agreement label

The same report still claims a complete, internally consistent agreement when:

- Every segment's `netlist` and `instance_mapping` are removed.
- The technology snapshot and model definitions are removed, and provenance
  hashes and technology status are replaced by `{"fake": 0}`.
- Both comparison and export `schema_version` values become `"999"`.
- A declared netlist output points to `nonexistent,p_out`.
- The instance mapping is emptied while its netlist remains.

Required fix: check supported versions and meaningful required artifact fields;
validate the structural coverage and relationships among exported model ports,
netlists, mappings, boundary descriptors, technology and provenance. A report
must remain inspectable but must not promote these incomplete or inconsistent
records to the complete agreement status. This is a presentation contract check,
not a demand to rerun physics or cryptographically authenticate authorship.

Regression coverage is in `tests/test_optical_review.py`, class
`OpticalReviewReportTests`. The initial run fails the tolerance test and all five
structural subcases. Production code was not edited by the reviewer.

## Independent validation completed

The reviewed implementation does use actual external SAX circuit composition with
separate field models; it does not substitute the scalar evaluator as the optical
solver. The implementation's 25 optical module/report/CLI tests ran in the SAX
environment with 24 passing and one expected dependency-absence branch skipped.

The reviewer added seven passing non-report tests covering:

- Exact component, source, detector, unused-termination and final NN-use coverage
  for a two-lane, five-use regenerated graph; every local model port is consumed
  exactly once and all ten electrical links are explicit excluded boundaries.
- Custom technology identifiers and punctuation in NN source identifiers, with
  safe reversible local SAX names.
- A detector-only wavelength restriction: an otherwise supported sweep is
  rejected before the engine starts, with the failing sweep index and guards.
- Malformed top-level and nested realization snapshots rejected before optional
  engine import.
- Independent hand-calculated power with non-default splitter excess loss,
  non-default guide attenuation, unequal route lengths, an unused leaf, and
  requested wavelength ordering preserved by the real SAX sweep.
- Deliberately perturbed scalar expected powers produce disagreement while SAX
  output remains equal to the independent optical oracle.
- Nonfinite or missing external transfer coefficients yield structured errors,
  never a pass.

The initial attempt to use a 30:70 splitter was correctly rejected by the
previously declared fixed-50:50 graph-builder scope. It is not a Part 3 defect;
the independent supported-circuit oracle uses custom losses instead.

Commands used for the main independent checks:

```sh
MPLCONFIGDIR=/tmp/npp-mpl /tmp/npp-v02-env/bin/python -m unittest \
  tests.test_optical_simulation tests.test_optical_report tests.test_optical_cli -v
MPLCONFIGDIR=/tmp/npp-mpl /tmp/npp-v02-env/bin/python -m unittest \
  tests.test_optical_review -q
python -m unittest tests.test_optical_review.OpticalReviewContractTests -q
```

The new review suite ran nine methods: seven core/export/engine tests pass; the
two report methods expose six initial failures across their executed cases. The
four dependency-free review contract tests also pass with base Python.

A separate read-only report/CLI subreview ran both installed-SAX and absent-SAX
paths, confirmed output preservation, replay before export/simulation, actionable
missing-dependency diagnostics and correct exit codes. It also rendered 3,960
JSON-compatible malformed substitutions without finding an unhandled renderer
exception. HTML escaping and lack of network assets looked sound. Those positive
checks do not resolve R3.1 and R3.2.

## Scope judgment

No remaining core optical-simulation blocker was identified. The scope is
appropriately limited to full-scale optical power composition at detector inputs
and explicit matched terminations. Regeneration is checked as separate optical
segments with electrical cuts and modulators clamped to full-scale sample one.
The exporter, results and documentation explicitly exclude calibration,
electrical dynamics, noise, timing, phase/group-delay validation, whole-NN
computation and fabrication readiness. The technology remains illustrative and
unreviewed by a photonics expert. These exclusions are honest bounds, not missing
requirements for this gate.

Part 3 cannot pass while the report presents unsupported or incomplete saved
results as consistent agreement. Fixes must pass the unchanged independent
regressions and relevant existing tests, followed by a fresh independent judgment.

## Reviewed source fingerprints

- `npp/optical_simulation.py`: `e57c57e8a35cafef2b1daf8b9ca53afe2d02e2c283e36497ef83e10fc2886c9e`
- `npp/optical_report.py`: `9ce4d97e2c08db676330ab2f7bc8399b0fcf031c33d290162a78287d51755f00`
- `npp/cli.py`: `86c4727dc8f5765a57aabab57a33b18b7a4a8bfe90a324ac8385e5d9bc8e416c`
