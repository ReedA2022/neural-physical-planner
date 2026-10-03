# Part 4 independent review — round 1

**Judgment: PASS.** Part 4 is complete enough in its declared bounded form to
proceed to final milestone integration and release validation. No blocking
defect was found in this review. No production code was changed by the reviewer.

## Review scope

Reviewed the finite-grid designer, requirements and choice locks, baseline
replanning, Pareto selection, constraint explanations, portable replay, comparison
report, CLI and shipped examples. The reviewed implementation is deliberately a
physical fanout distribution subcircuit planner for one NN node. It supports
passive and regenerative distribution, explicit component-family choices and
route-length alternatives under the embedded component models.

This judgment does not imply arbitrary NN synthesis, geometry or fabrication
closure, measured device calibration, task-accuracy validation, or a global
optimum outside the evaluated submitted grid. Instance locks constrain the
documented global family for that component kind; they do not create independent
per-instance component overrides. These limits are consistently disclosed.

## Evidence and independent tests

The reviewer read `npp/designer.py`, the strict physical contracts and
`docs/DESIGNER.md`, and independently ran all 38 submitted designer, oracle,
report and CLI tests: **38 passed**.

Added `tests/test_designer_review.py`, with **9 passing test methods**:

1. All 15 nonempty subsets of the four objectives agree with an independently
   calculated all-pairs Pareto frontier over separately evaluated circuits.
2. Distinct route implementations with equal selected objective values both
   remain eligible for retention.
3. Conflicting component-family locks produce a complete empty feasible set,
   explain the conflicting choices, and perform no physical evaluations.
4. A changed symbol duration flags operating assumptions as changed, disables
   like-for-like comparison, and produces the expected energy and latency deltas.
5. Exact numeric requirement boundaries pass. A candidate violating two numeric
   requirements does not generate a misleading single-requirement repair.
6. Hard exclusions occurring after the evaluation cap remain exclusions, while
   eligible untested candidates remain explicitly unevaluated.
7. Decimal unit equivalence preserves locks and duplicate detection, including
   `0.0001 W`, `100 uW` and numeric `0.1 mW`.
8. A generated record exceeding the output budget is rejected even when its
   input snapshots pass the earlier allowance; replay also enforces the limit.
9. Tampering with resolved locks, cost drivers, receiver identity, relaxation
   witnesses or retained realization metrics fails reconstruction.

A separately delegated read-only report/CLI audit added
`tests/test_designer_report_review.py`, with **3 passing test methods**:

- Stale or duplicated retained plans and fabricated rejection explanations lose
  the verified status, plots and candidate links.
- Boolean, nonfinite and oversized metric tampering remains inspectable while
  invalidating the saved claims.
- A truncated search through paths containing spaces returns the documented
  incomplete-search exit code, visibly remains incomplete, and can be replayed
  after deletion of the original design file.

The independent tests ran together: **12 passed**. This gate therefore exercised
**50 passing focused test methods** across the submitted and independent suites.
The objective-subset method additionally covers all 15 selections as subcases.

## End-to-end examples

The report/CLI auditor executed the shipped examples:

- Planning: 9 evaluated candidates, 8 feasible, 3 retained. The saved result
  passed `check-physical-design`.
- Replanning from the physically infeasible passive 50 uW launch / 10 mm route
  baseline: 3 evaluated, 6 excluded by locks, 2 retained regenerated circuits.
  Both retained the baseline launch power. The saved result passed replay, and
  all retained realization and report files were present.

## Assessment

The designer exposes the promised constrained alternatives and lets a hardware
designer inspect why a candidate is excluded, physically infeasible, over budget,
or retained. Search completion, physical feasibility and nondominance are
distinguished correctly. Changes of technology or operating assumptions cannot
silently masquerade as like-for-like improvements. The replay checker binds the
saved inputs and reconstructs all output claims, including explanations and
retained plans; the report uses that verification before presenting success.

The required next step is the already planned final integrated validation across
the four parts, including external SAX checks of retained supported circuits and
legacy regression/package checks. A second Part 4 edit/review round is not
required by this gate because no blocker remains.
