# M1 independent implementation gate

**Verdict: PASS for the bounded M1 scope. M2 may begin.**

This final verdict follows two repair rounds after the initial HOLD. All
counterexamples below now reject, classify, or evaluate correctly. Review was
completed against the opt-in runtime carrying software version 0.4.0; this is
not a claim that its later M2–M6 milestones have passed.

Reviewer: independent `critique_m1` agent, with a separate `cli_probe` agent
checking public interfaces. The reviewer did not edit production code. Review
regressions are retained in `tests/test_mp_m1_review.py`.

## Bounded scope

Reviewed the common contract layer, complete digital biased SwiGLU execution
and local cost model, differential-conductance matrix primitive, technology
catalog, local lifecycle, friendly project files and deterministic replay.
The gate covers M1 in `IMPLEMENTATION_ACCEPTANCE.md`, not a mixed graph,
system scheduler, photonic matrix family, adaptation engine or device calibration.

This is the first substantive slice of REQ-01–06, REQ-08–12, REQ-14–17,
REQ-27–28, REQ-32–35 and REQ-37. Spanning requirements remain open for their
later gates. REQ-12 here means explicit declarations; runtime cross-domain
connections are M2 work. No M1 pass completes the multiphysics specification.

## Findings and repair rounds

The original 50 implementation tests passed, but adversarial review found
behavior those tests did not cover:

| Finding | Counterexample | Required repair |
| --- | --- | --- |
| Ignored component metadata | Restrict the input/output range, change a digital port to current, declare dimensions in GHz, or declare a billion-nanosecond integration time; the result stayed feasible and unchanged. | Enforce bound values and canonical units; reject unsupported signal/dynamics/accounting declarations. |
| Lifecycle contract bypass | Instantiate a one-row envelope, then evaluate a two-row model in an incompatible environment without power. | Apply the same component/environment validators to catalog and lifecycle execution. |
| Track A budget ignored | Float32 arithmetic violates an explicit zero absolute-error budget, but result status remained feasible. | Propagate semantic-budget failure to the public result and lifecycle status with diagnostics. |
| Malformed replay crash | Replace embedded `inputs` with a list; actual CLI raised `AttributeError`. | Validate the mapping before access; return invalid replay and exit 2. |
| Malformed sampling crash | Supply JSON null/list/number/string as sampling context. | Return a structured invalid lifecycle outcome. |
| Non-reusable digital state | First execution returned a prefixed hash which the next execution rejected. | Normalize identity representation and retain changed-weight rejection. |
| Unsupported adaptation accepted | Label unchanged weights Track R with arbitrary adaptation names and unknown adaptation cost. | Reject hardware execution as unsupported; metadata representation is not an adaptation mechanism. |
| False zero RMS | One error of `5e-324` among four outputs underflowed the reported RMS to zero and passed a zero budget. | Fail explicitly when nonzero error cannot be represented as a positive RMS; retain representable subnormals and genuine zeros. |

The integration audit additionally hardened evaluator-registry failure
classification and added explicit category coverage for storage, calibration
and other locally absent operations. Referencing an existing owner does not
add its energy a second time.

## Independent evidence

- The review executes the public interfaces and actual command line, rather
  than relying only on author-reported outcomes.
- A separately written scalar analog oracle reconstructs conductance
  quantization, bipolar DACs, column loading, both current rails, differential
  ADC readout, and device/line Joule energy. It does not call production
  arithmetic helpers.
- The existing semantic tests were inspected for independence: they include
  high-precision/scalar FFN and explicit float32 rounding references.
- A 1,105-case nested/type replay mutation probe found four crashes with the
  same nonmapping-input cause; the other cases were handled. This is stress
  evidence, not a universal proof of parser robustness.
- The frozen legacy check independently passed all 30 checks, including the
  integrity of all 57 pinned artifacts, without refreshing any baseline.

## Commands and final results

Independent final execution passed **63 multiphysics tests**, including the
reviewer's **11 test methods**, with no failures or skips. The frozen legacy
gate again passed **30 checks / 57 artifact integrity checks**, unchanged.
Commands:

```sh
python -m unittest discover -s tests -p 'test_mp_*.py' -q
python scripts/check_multiphysics_baseline.py --report /tmp/npp-m1-independent-baseline-final.json
```

Actual command-line negative cases were also re-executed after the final fixes:
the violated Track A budget exits 3 with `resource_infeasible`; unsupported
Track R exits 2 with `unsupported`; positive-error RMS reporting underflow
exits 2 with `numerical_failure`. None produced a traceback. The independent
replay regression verifies malformed embedded inputs exit 2 with `valid=false`.
Successful digital state reuse and changed-weight rejection are both checked.

Root integration evidence is retained under
`runs/multiphysics-validation/m1/`, including the final test log, baseline
record, digital replay and analog evaluation. These records are production
execution/replay evidence; the scalar oracle tests provide the separate
numerical comparison evidence.

## Limits of the gate

Reference coefficients are hypothetical. The independent scalar tests check
declared-model numerical semantics; they are not external hardware measurements
or independent circuit simulation. M1 durations are local serial models.
Startup, global package infrastructure and unsupported calibration processes
remain outside the active-component boundary and must not be presented as
free system resources. Track R execution is unsupported. M2 must implement
typed composition, system reservations/state hazards and the second classical
matrix family before mixed-system claims are justified.
