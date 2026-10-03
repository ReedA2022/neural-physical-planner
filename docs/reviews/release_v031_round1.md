# v0.3.1 independent release critique

Date: 2026-10-03. Baseline: `cd351be`.
Reviewer: `/root/critique_release`, an automated agent separate from the
production-code authors. Narrow CLI/report reviewer:
`/root/critique_release/review_cli_outputs`.

Final gate status: **PASS for the documented v0.3.1 research prototype**.
The first critique required corrections; the recorded counterexamples now pass,
and the frozen release evidence below satisfies the stated software gate.
This is software review of the documented research prototype, not human hardware
review, component calibration, fabrication signoff, or approval of M1–M6.

## Initial judgment and corrections

The initial shipping judgment was **REVISE**. Passing the initial campaigns did
not justify approval: independent counterexamples exposed four defect classes,
including a second correction to the optical-energy fix.

| Finding | Independently reproduced failure | Required correction and retest |
| --- | --- | --- |
| Nonlinear error underflow | Input noise RMS `1e-100`, linear weight `1e-160`, then ReLU produced an aggregate error bound of zero despite a stored per-output bound of `1e-260`; a zero error budget passed. Chaining another small weight could erase the bound earlier. | Stable Euclidean aggregation and explicit scalar-product range guards. Both positive representable RMS and unrepresentable chained attenuation are covered by independent tests. |
| Quantity-string underflow | `1e-999 pJ` became zero energy; `-1e-999 um` became negative zero and passed a nonnegative route contract. Extremely small Decimal exponents could underflow before float conversion. | Both configuration and designer parsers trap Decimal underflow and reject nonzero scaled values rounded to float zero. Tests cover both signs, extreme exponents, explicit zero and the representable smallest subnormal. |
| Optical-energy intermediate range loss | `1e-300` photons gave zero energy after premature SI underflow. The first fix missed nonzero subnormal intermediates: `3.7e-299` photons at 1550 nm yielded `3.1875188613658446e-306` pJ instead of the independent Decimal value `4.741838497710346e-306` pJ, incorrectly passing a `4e-306` pJ budget. | Bounded mantissa/exponent fallback whenever an intermediate is subnormal or nonfinite; unsupported final ranges fail explicitly. The second counterexample verifies the nonzero-budget decision against an independently evaluated SI expression. |
| Infinite sampling uncertainty reported as agreement | A zero-valued digital input with noise RMS `1e150`, 20 samples and seed zero returned infinite MSE standard error and `consistent_with_bound: true`, with no diagnostics. Strict JSON serialization failed. | Scale squared errors before their standard-deviation calculation, check modeled MSE range, and reject nonfinite final statistics. The independent test requires either explicit numerical failure or finite serializable uncertainty. |

The reviewer sent concrete reproductions before the corresponding authors edited
production. The reviewer only added independent tests and this record. The
authors also broadened their own regressions and campaigns.

## Independent executed checks

The following were executed by the critique agents, rather than inferred from
the implementation authors' test counts:

- Eight tests in `tests/test_release_review.py`: the failures above, chained
  attenuation, true-zero preservation and representable subnormal inputs.
- Two tests in `tests/test_release_cli_review.py`: six end-to-end CLI
  export/replay cases from zero through the largest finite float, and 25
  malformed JSON/record-command combinations. Metrics survive export and replay,
  manifests enumerate existing files, plot coordinates stay finite, and rejected
  commands leave no output artifacts.
- The narrow report reviewer additionally executed 3,200 seeded nested mutations
  across implementation and optical reports without an uncaught exception.
  These probes are supplementary review evidence, not additional distinct device
  validations, and are not included in the aggregate release-campaign count.
- The combined targeted suite covering the independent tests, harness and all
  five stress areas ran 53 tests: 52 passed, with one expected ONNX-dependency
  skip in the core environment, and no failures or errors.
- After the final harness correction, the reviewer reran the ten critique tests
  plus seven aggregate-harness tests: all 17 passed.

Reproduce the independently added tests from the checkout root:

```sh
python -m unittest tests.test_release_review tests.test_release_cli_review -v
```

## Other implementation and harness findings

The revised legacy dominance rule requires exact componentwise non-worsening and
a significant relative improvement in at least one objective; exact objective
ties alone share a canonical representative. This avoids the prior cyclic
relaxed relation and loss of necessary dominance witnesses. It deliberately
differs from the designer's exact Pareto dominance and retained tied alternatives.
The independent mathematical review and adversarial regression cases support
the distinction; near-equal historical frontier outputs can intentionally change.

The review also inspected positive physical contribution guards, covariance and
shared-noise tests, unit equivalence, YAML alias accounting, mixed boolean tensor
rejection, simulation allocation/work preflight, report escaping and numerical
plot coordinates, and saved-record replay boundaries.

The aggregate harness preserves failures and skips separately, can require
optional dependencies, refuses optimized Python that would disable assertions,
records evaluated source hashes, and fails if sources change during a run.
Actual SAX outcomes remain separate from local model checks; replay is not
reported as independent physics validation. Existing baseline artifacts are not
refreshed to hide regressions.

The first aggregate run correctly failed when the legacy campaign returned
`actual_scenario_count` instead of the aggregate's required `scenario_count`.
This integration failure was preserved in `full-stress/`. A canonical alias,
preservation of malformed campaign payloads and a real all-five-campaign contract
test corrected the boundary. A complete rerun used `full-stress-final/`; the
earlier failed run was not relabeled or counted as final passing evidence.

## Frozen release evidence and decision

The reviewer read the final aggregate and all 15 campaign reports, verified that
their counts agree and their failure lists are empty, checked the full/core logs
and frozen baseline report, and inspected the installed-wheel record. The
reviewer independently compared all 28 installed Python/data files with the
current checkout: no differences. The aggregate's individual source hashes also
match the checkout after execution.

| Gate | Final observed outcome |
| --- | --- |
| Seeded aggregate | 6,075 scenarios passed; zero failures and zero skips; source unchanged during execution |
| Actual SAX comparison | 183 boundary/wavelength comparisons passed over four circuits; maximum absolute discrepancy `8.33e-17 mW` |
| Full dependency suite | 340 tests: 339 passed, one expected missing-SAX-path skip |
| Core suite | 340 tests: 298 passed, 42 explicit optional-dependency skips |
| Frozen compatibility | All 30 comparisons passed, including the 57-artifact integrity check; no refreshed goldens |
| Installed wheel | 18 checks passed outside the checkout, including a separate 24-comparison actual SAX run |

Evidence is in `runs/validation-v0.3.1/`, consolidated by
`release-summary.json` and described in `docs/STRESS_VALIDATION.md`. The evaluated
source digest is
`9886fae30b229f798326045e013391f5fa97ade0396b812a0b2ab6e0092b5fbe`.
The reported limits and distinction between repeatable software evidence,
declared-model consistency and actual hardware validation are accurate.

**Decision: ship v0.3.1 within its documented research-prototype scope.** No
unresolved shipping blocker remains from this review. Further scenario expansion
is not required for this maintenance gate; the proposed multiphysics work needs
its own staged implementation and review.

## Remaining limits relevant to shipping

- This release remains the legacy macro planner plus bounded optical fanout
  implementations. It does not physically synthesize complete neural networks,
  provide fabrication layouts, implement the proposed multiphysics stages, or
  establish a speedup or research novelty claim.
- Technology numbers remain illustrative and uncalibrated. SAX checks compare
  declared optical power under matched assumptions; they do not verify energy,
  electronics, noise, timing, task accuracy or actual devices.
- Numerical range failures can conservatively reject mathematically meaningful
  scenarios. Default float decoding of plain JSON/YAML numeric literals follows
  binary64; the new explicit nonzero-underflow parser guarantee applies to
  quantity strings. Loading large plain specifications is not a general process
  memory guarantee.
- Seeded campaigns and unit tests are finite evidence. They do not establish that
  every accepted graph, library and floating-point value is free of defects.

These limits remain visible acceptance boundaries, rather than capabilities
implicitly conferred by the passing maintenance gate.
