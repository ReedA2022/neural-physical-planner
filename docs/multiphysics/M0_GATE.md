# M0: baseline reconciliation gate

Date: 2026-10-03. Baseline: `3ac1f30337753b1930fc538dc9ac58cbc1ac4c70`.
Runtime: 0.3.0. Gate status: **PASS**. Next eligible stage: **M1**.

This gate adopts the supplied multiphysics requirements and establishes the
compatibility boundary for future work. It does not implement M1–M6.

## Requirements addressed at this gate

- Section 23 M0: capability inventory, pinned legacy outputs, migration plan and
  experiment definitions.
- REQ-01/02/30/31: register workload/baseline/evidence/uncertainty obligations and
  H1–H6 comparison plans, without claiming an executable research campaign.
- REQ-32/33: retain existing inputs, meanings and runtime schemas; plan a separate
  versioned multiphysics format.
- REQ-36/37: executed software compatibility checks, independent critique,
  recorded issues/fixes and explicit evidence boundaries.
- All REQ-01–37 are inventoried; none of their broader runtime obligations is
  marked complete just because its documentation has been added.

## Changes

Preserved the original specification and a readable transcription containing all
37 requirements, seven embedded equations, ten tables, H1–H6 and source links.
Added code/test traceability, staged architecture/migration decisions and a
documentation-only experiment registry. Added a frozen baseline manifest and
checker with negative tests. Updated the root README and changelog to expose the
roadmap and its implementation status.

## Executed evidence

| Check | Executed result |
| --- | --- |
| Frozen baseline checker | 30 comparisons pass; 57 artifact hashes match |
| Full dependency suite | 285 tests run: 284 pass, 1 expected skip, no failures/errors |
| Core-only suite | 285 tests run: 244 pass, 41 expected optional-dependency skips, no failures/errors |
| New baseline fault tests | 11 pass, including altered goldens, runtime metrics, search completeness, exceptions and overwrite prevention |
| Independent review tests | 6 pass, including source preservation, proposed-input rejection, normalization drift and schema-only drift |
| Unchanged runtime/input/schema bytes | 56 files match baseline commit |
| Source transcription | 37 requirements, 7 equations, 10 tables, 15 external links and all 480 non-equation paragraphs/table cells preserved |

Python 3.12.14 was used in both environments. The single full-suite skip tests the
missing-SAX path and is expected when SAX is installed; that path passed in the
core-only suite. The 17 new tests are included in the 285-test totals.

The [machine-readable summary](validation/summary.json) records environment
versions and scope. The [baseline report](validation/baseline-check.json),
[core log](validation/core-tests.txt) and [full dependency log](validation/full-tests.txt)
preserve the executed evidence. Reproduce from the repository root:

```sh
python scripts/check_multiphysics_baseline.py
python -m unittest tests.test_multiphysics_baseline tests.test_multiphysics_m0_review -v
python -m unittest discover -s tests -v
```

Full optional coverage uses the existing tested importer, Keras and photonics
dependencies documented in [VALIDATION.md](../VALIDATION.md). The final suite
exercises the existing SAX integration tests as regressions. M0 adds no new
device or coupling validation. The historical 370-comparison v0.3 integration
bundle is retained and integrity-checked rather than relabeled a new campaign.

## Review

Independent reviewer: `/root/critique_multiphysics_m0`, an automated critique
agent separate from the implementation/document authors. Its
[round-one record](../reviews/multiphysics_m0_round1.md) gives a **PASS** after one
minor correction: the Markdown now retains the original title/subtitle, and the
conversion metadata accurately describes the editorial steps. The reviewer
verified the correction and added the six independent tests listed above.
There are no remaining required changes for M0.

This judgment covers software compatibility and scope. It is not human hardware
review, model calibration, or a new external device comparison. The final combined
suites passed after the correction. M1 may proceed through its own implementation,
critique and correction gate; no M1–M6 runtime gate passes by inheritance.

## Remaining limitations

The runtime remains bounded optical distribution plus the legacy macro planner.
There is no new complete digital FFN/decoder path, general backend registry,
resource scheduler, mixed neural operator, acoustic mechanism or quantum model.
H1–H6 remain proposed and blocked on declared models, resource envelopes,
numerical sweep bounds, tolerances and quality criteria. The supplied literature
descriptions are preserved, not independently re-audited in M0. No speedup or
novelty claim is made. Archived SAX evidence retains its original optical power
scope; the baseline checker does not execute an external simulator.
