# Part 2 independent review — round 2

Date: 2026-10-03 (Europe/Paris)

Decision: **PASS. Part 2 is complete enough within its declared one-node intensity-fanout scope; Part 3 may begin.**

## Resolution of the blocking findings

- **R2.1 resolved.** Presentation consistency now requires unique exact instance
  coverage, matching component bindings, exact declared/evaluated receiver coverage,
  NN lane/use bindings, finite typed cost/noise/timing fields and matching aggregate
  counts. Duplicated rows, missing or unknown receivers, fake aggregate fields,
  Boolean/string/null/nonfinite energies no longer receive a complete feasible
  label or normal aggregate presentation. Stored claims remain inspectable, with
  specific presentation issues and a clear distinction from replay or simulation.
- **R2.2 resolved.** List-valued component identifiers are handled as malformed,
  unresolved data. The report remains readable and explicitly incomplete instead
  of raising an unhandled `TypeError`.
- **R2.3 resolved.** Fresh carrier sources are displayed beside the early
  detectors, immediately before their modulators. Independent SVG path checks for
  regenerated fanout 2, 4 and 8 find no paths through unrelated component boxes.
  Fresh raster inspection of regenerated fanout 2 and 8 confirms distinct optical
  carrier and dashed electrical sample paths with correct visible endpoints.
  The eight-way graph is wide and uses horizontal scrolling at native size; full
  instance and typed-connection tables retain exact labels.

## Independent retest evidence

The reviewer reran the regression tests without changing their expected behavior:

```bash
python -m unittest tests.test_implementation tests.test_implementation_review \
  tests.test_implementation_report tests.test_report_review \
  tests.test_physical_cli -q
```

Result: **48 tests pass** in this focused implementation/report/CLI suite, including
10 independent graph/evaluation tests and 6 independent report regression tests.

The prerequisite technology and technology-review suites were also rerun:

```bash
python -m unittest tests.test_technology tests.test_technology_review -q
```

Result: **24 tests pass**. Together, **72 focused tests pass** in the reviewed state.

The independent report subreview additionally executed the CLI workflow: feasible
realization and replay, overwrite refusal, tampered metric rejection, and the
creation/replay of an infeasible realization. Exit statuses are consistent with the
documented distinction between feasibility and deterministic replay validity.

## Completion judgment

The implementation now provides a usable bounded physical graph below the NN:
component instances, typed ports and explicit connections; guarded passive and
regenerative fanout; complete ordered NN use/lane bindings; explicit unused-leaf
termination; instance cost and timing ledgers; first-order covariance retaining
shared ancestry; portable deterministic replay; and an inspectable offline report.
No remaining blocker was identified within this part's declared scope.

The approval does not certify measured devices, a whole-NN hardware realization,
calibration, clipping absence, task accuracy, routing/fabrication readiness or
throughput. The reference technology remains illustrative and unreviewed by a
photonics expert. Independent optical simulation is deliberately reserved for
Part 3; designer edits/replanning and explanations are reserved for Part 4.

## Reviewed source fingerprints

- `npp/implementation.py`: `2b36b4ef0b275083b4c7e88689e4b34d64d75a2b7d861bf52c533c03e11fddbe`
- `npp/implementation_report.py`: `350621cc3d3bcfc5058cd31a74328244fe457c22ae937f41f5b1ba3147c4bb2a`
- `npp/cli.py`: `93870109f300d80ea33c0cc1895f4e430f9ecccf56ecce0ee11ca4c2e2711221`
