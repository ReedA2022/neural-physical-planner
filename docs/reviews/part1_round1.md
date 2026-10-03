# Part 1 independent review — round 1

**Judgment: BLOCK.** The reduced technology model is a coherent software foundation, but two input-validation defects must be fixed before the sequential gate is approved. No part 2 implementation is approved by this review.

## Scope and evidence boundary

Reviewed `npp/technology.py`, the packaged reference JSON, technology CLI/schema integration, `docs/TECHNOLOGY.md`, and the part 1 acceptance boundary in `docs/MILESTONE_V03.md`. This is an independent software/model-consistency review, not a human photonics review or calibration. The illustrative numeric values and unreviewed reference status are clearly disclosed and must remain so.

The reference defines a deliberately narrow scalar normalized-intensity model. Source energy, splitter power loss, waveguide attenuation/delay, photon counts, and detector shot/electronic noise formulas are internally consistent in their declared units. The docs correctly distinguish field from power transmission, full-scale calibration from instantaneous sample value, fixed overhead from symbol acquisition, and independent simulation from device calibration.

## Checks performed

- `python -m unittest tests.test_technology tests.test_physical_cli -v`: **15 tests passed**.
- Added independent `tests/test_technology_review.py` with power-conservation/asymmetric-division checks, source-energy scaling with a nonzero fixed cost, receiver noise scaling, waveguide cascade composition, inclusive envelope boundaries checked against adjacent floating-point values, order invariance, named-review representation, and malformed public inputs.
- `python -m unittest tests.test_technology_review -v`: **10 test methods; 7 passed, 3 exposed defects**. The failures comprise 5 failed subtests and 6 error subtests described below.
- Built a wheel using `python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/npp-gate1-wheel .`, installed it into `/tmp/npp-gate1-installed`, and invoked it from `/tmp` with that installation on `PYTHONPATH`. Confirmed imported module path was the installed package, not the checkout.
- Installed-wheel smoke checks passed: `technology`, export to JSON, reload exported JSON, and `schemas`; the generated technology schema declares version `0.3`. Package distribution metadata still says `0.2.0`; release versioning is a final packaging task, not a gate-1 defect.

## Required corrections

### R1 — Empty review/evidence text passes the claimed traceability checks

**Severity: medium.** A whitespace-only reviewer satisfies `status: reviewed`, although the contract requires a named reviewer. Required evidence citation, version, coverage and limitations fields also accept whitespace-only values. Thus a structurally accepted pack can claim a named review without a name and can satisfy evidence requirements with empty human-readable content.

Reproduction: start with `load_technology()`, append a valid `review_report` source, set `review.status = "reviewed"`, `review.reviewer = " \t\n "`, and `review.report_source_id` to that source. `validate_technology` currently succeeds. Independently, replacing any evidence `citation`, `version`, `coverage` or `limitations` with `" \t\n "` also succeeds.

Required fix: reject whitespace-only required review/evidence text with structured input diagnostics. Applying this consistently to required text in the new technology contract is preferable. This cannot authenticate a real review and should not be described as doing so; it simply enforces the advertised minimum declaration.

Regression methods: `test_blank_reviewer_cannot_satisfy_named_review_requirement` and `test_blank_evidence_is_not_a_traceable_source`.

### R2 — Oversized integer public inputs leak arithmetic exceptions

**Severity: low; required robustness correction.** `_number` and `check_operating_point` call `math.isfinite` directly on Python integers. A legitimate Python integer outside binary64 range causes `OverflowError` before the helper can return its documented validation error or diagnostic.

Reproduction: `db_transmission(10 ** 400)`, `photon_count(10 ** 400, 1.0, 1550.0)`, or `check_operating_point(detector, wavelength_nm=10 ** 400, temperature_c=25.0, symbol_duration_ns=1.0)` currently leak `OverflowError: int too large to convert to float`. The same operating-point issue affects temperature, duration and power arguments.

Required fix: handle nonrepresentable values without leaking arithmetic exceptions. Numeric helper APIs should raise `ValueError`; `check_operating_point` should return its structured infeasibility diagnostic. Do not silently clamp such values.

Regression method: `test_large_integer_public_inputs_fail_as_validation_errors`.

## Obligations for later parts, not gate-1 blockers

- A source component has no instance-level signal role yet. Part 2 must distinguish the already-encoded launch boundary from constant full-scale carrier sources. A modulator must not accidentally multiply two independently encoded samples.
- Enforce port connectivity/cardinality, both modulator inputs, unused splitter output termination, and propagation of shared noise contributions in the implementation graph.
- Include complete-symbol acquisition and explicit regeneration scheduling; fixed component overhead alone is not end-to-end availability latency.
- Preserve the separation between physical subcircuit metrics and legacy neural macro estimates.
- The bundled technology remains illustrative/unreviewed even after all automated review gates pass.

The next round should rerun the unchanged independent regressions and focused implementation/CLI tests after the required fixes. Further features are not needed for this gate.
