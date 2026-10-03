# Multiphysics M0: independent review, round 1

Date: 2026-10-03. Reviewed baseline:
`3ac1f30337753b1930fc538dc9ac58cbc1ac4c70`, plus the M0 working-tree additions.

Reviewer: `/root/critique_multiphysics_m0`, an independent automated critique
agent. The reviewer did not author the inventory, migration, registry or baseline
checker. This is a software and scope review, not human hardware review,
parameter characterization, a literature audit or external physics validation.

**Decision: PASS for M0. The project may advance to M1.** This decision does not
approve M1–M6 or claim that any proposed multiphysics experiment is executable.
The final integrated regression counts are recorded by the implementer in
`docs/multiphysics/M0_GATE.md`.

## Scope and requirement traceability

- Section 23 M0: inspected the capability inventory, pinned outputs, compatibility
  checker, migration plan and H1–H6 definitions against the supplied specification.
- REQ-01/02/30/31: checked registered baselines, resource/quality/evidence
  obligations, unresolved numerical domains, conditional outcomes and preserved
  failure controls. Registration does not count as execution.
- REQ-03–29: reviewed all inventory rows and their narrower v0.3 boundaries;
  digital, analog electrical, classical photonic, acoustic/phononic and quantum
  families all remain in the full scope. Existing intensity fanout is not
  represented as a signed matrix tile, mixed FFN or complete digital baseline.
- REQ-32/33: verified preserved source, proposed-input rejection, unchanged
  production files and existing fixtures, exact schema comparisons and complete
  saved-output comparisons with the stated finite-float tolerances.
- REQ-34/35: checked that missing future artifacts and explanations remain open;
  no local optical power comparison is promoted to complete neural computation.
- REQ-36/37: independently ran the gate and negative tests, added focused review
  tests, recorded the correction below and bounded this approval to M0.

Every REQ-01–37 has exactly one reconciled status row. None is marked complete
merely because the specification has been adopted. Migration retains both legacy
hash/evaluator meanings, separates new schemas, and requires sequential critique
gates. H1/H2 are assigned to the bounded M4 campaign, acoustic alignment waits
for M5, and stateful H3–H6 studies wait for M6.

## Finding and correction

**F1 — minor transcription provenance mismatch, resolved.** The initial readable
transcription replaced the original title/subtitle with a combined navigation
heading. Metadata described only an added title and note. An independent
paragraph comparison found the original subtitle missing, although no technical
requirement or equation was missing.

The implementer restored both original title paragraphs verbatim and updated
the editorial-change record and transcription hash. The reviewer reran the source
test: all 480 non-equation paragraphs/table-cell paragraphs now occur in the
transcription after removing presentation syntax. There are no outstanding
required changes.

## Executed evidence

The reviewer executed these checks, rather than relying only on the implementer's
reported test results:

| Command or check | Result |
| --- | --- |
| `python scripts/check_multiphysics_baseline.py --report /tmp/npp-m0-independent-report.json` | PASS: 30 checks, 57 pinned files; external simulator explicitly not executed |
| `python -m unittest discover -s tests -p 'test_multiphysics_baseline.py' -v` | 11 tests passed |
| `python -m unittest discover -s tests -p 'test_multiphysics_m0_review.py' -v` | 6 independent review tests passed |
| Byte comparison of archived DOCX against the supplied uploaded attachment | Identical |
| `git show` comparison of every manifest entry with Git provenance against baseline commit | All 56 original files identical |
| `git diff 3ac1f30337753b1930fc538dc9ac58cbc1ac4c70 -- npp pyproject.toml schemas examples runs` | Empty: no runtime/schema/existing fixture changes |

The new review tests use standard-library ZIP/XML inspection and established core
dependencies. They add no optional document or simulator requirement. They check:

- Source and transcription digests, all 480 non-equation paragraphs/table cells,
  seven equation blocks, ten tables, 15 external links, and all 37 requirement IDs.
  The seven mathematical expressions were also inspected manually against the
  source math content; the equations retain their stated meaning.
- Exactly H1–H6, no results or executed baselines, unresolved sweep domains and
  quality thresholds, unmet blockers and resolvable internal registry references.
- Actual proposed project and acoustic-component examples, and the experiment
  registry, fail current planner entry points rather than gaining accidental
  runtime support.
- Fault injection into friendly-input normalization is detected independently
  of replaying already normalized legacy snapshots.
- A schema-only constraint change is detected even when every evaluated fixture
  still fits within the altered constraint.

The supplied negative tests additionally demonstrated detection of changed or
missing golden artifacts, weakened inventory, historical SAX-result tampering,
changed physical costs, changed search completion, runtime exceptions, discrete
type substitutions, non-finite values and attempted report overwrites. Optional
SAX/JAX imports are forbidden during the core gate test.

## Judgment and remaining limits

M0 satisfies its deliverable and acceptance boundary: the active v0.3 scope is
accurately reconciled, existing example meanings are pinned and checked, source
requirements are retained, and future work has explicit gates and experiment
definitions. The baseline gate compares full deterministic structures instead of
selected favorable metrics; only finite nonnegative legacy wall-clock duration
is excluded. It cannot refresh its own expected answers.

The baseline gate is a regression check using production evaluators. It is not
an independent semantic oracle. Historical SAX evidence is checked for byte
integrity only and keeps its original full-scale optical-power scope. The new
tests establish no device calibration, signed arithmetic, stateful scheduling,
acoustic/quantum behavior, inference speedup or research novelty.

M1 must still implement and review common contracts, unknown-cost/evidence and
ownership semantics, a complete bounded digital reference, and a selected
electronic/photonic mechanism. The experiment registry is documentation-only;
numeric domains, resource caps, datasets, tolerances and physical models remain
unresolved prerequisites for any research campaign. Full regression execution
after the review additions remains the final integration check before delivery.
