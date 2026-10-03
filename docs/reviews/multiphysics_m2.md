# M2 independent implementation gate

**Verdict: PASS for the bounded M2 scope. M3 may begin.**

This verdict follows the initial HOLD and targeted repair/review iterations.
All retained counterexamples now pass their expected rejection, accounting or
semantic checks. The later M3–M6 gates remain separate.

Reviewer: independent `critique_m2` agent, with independent specialist
`schedule_review` for scheduling, state and geometry. Neither reviewer edited
production code. Retained review tests are `tests/test_mp_m2_review.py` and
`tests/test_mp_m2_review_schedule.py`. Author integration tests and self-audits
are identified separately; an author-reported pass is not this gate verdict.

## Scope and requirement slices

Reviewed the bounded complete mixed SwiGLU compiler, digital/analog/coherent
photonic primitives and explicit conversions, input/output/two-dimensional
partitions, signed hidden permutations, signed field gating and four alignment
alternatives, state ownership/lifetimes, capacity reservations, physical geometry,
system costs, friendly compilation files and deterministic portable replay.

The M2 slices are REQ-01, REQ-03–09, REQ-11–18, REQ-20, REQ-23–28,
REQ-32–35 and REQ-37. These are bounded composition/scheduler/workflow slices,
not completion of later search, acoustic, quantum, decoder or experimental
requirements. M3 independent device comparisons and M4–M6 studies remain later
gates. The acceptance document and supplied specification control this scope.

## Findings and repair iterations

The first independent execution of the expanded 163-test multiphysics suite
failed the author-discovered missing-clock regression. Adversarial review then
found defects not covered by the original author test set:

| Finding | Counterexample | Required repair and review status |
| --- | --- | --- |
| Native operation contracts were weaker than wire equality | Relabel every digital port with volts, an unknown encoding, or scale two; both endpoints agreed and the graph validated. Coherent field ports could uniformly claim mW. | Registered digital/field/voltage/current representations now enforce their native units, encoding and scale requirements. Independent regressions pass. |
| Inclusive disclosure ownership was not bound | Change a nested child's accounting parent to an unrelated identifier. | Nested ownership and schedule IDs now bind to the actual enclosing composite. Independent regression passes. |
| Malformed explicit implementations silently selected defaults | Pass `[]`, `{}`, an empty string, false or zero to the compiler. | Default only on an omitted/None specification; malformed values reject. Independent regressions pass. |
| Undeclared geometry forwarding | Connect compute blocks a–b–c with bidirectional routes; the bus-distance helper silently joined different ports inside b at no cost. Off-center junctions also retained a center-based distance. | Require the explicit supported shared-bus topology and centered, correctly oriented junction ports. Both negative regressions pass. |
| Abstract capacity invented physical bus channels | Set all resource capacities to four; up/gate transferred concurrently over a one-channel bus. | Bind effective link capacity to validated route and port capacity; report available versus physical/effective capacity. Independent capacity regression passes. |
| Required clock service ignored | Remove clock from the environment while digital graph ports and clock startup still required it. This was disclosed by an author integration audit. | Enforce required services recursively over top-level and internal graph ports. Author regression independently rerun. |
| Shape-changing fixed arrays underprovisioned | A 4×1 projection and a 1×4 projection shared a physical array whose area was only the largest local element count. This was disclosed by author self-audit. | Provision maximum independent row and column dimensions and separately account input/output devices and holding power. |
| Padding altered the physical loading | After fixing rectangular area, evaluating only an active analog row still removed inactive rows' nonzero baseline conductance for free. Independent review identified this residual defect. | Evaluate the complete zero-padded physical array and pay padding/output extraction, full programming and converters. Independent finite-conductance FFN oracle passes after actual full-array evaluation. |

The final envelope probe found an additional bypass: a separately selected
fused down-projection component with an exact-zero input range still accepted a
nonzero coherent input because the graph skipped range validation for field
payload dictionaries. The repaired compiler validates the selected semantic
input envelope from the physical quadrature diagnostic without exposing it as
an available free readout, while preserving the field-amplitude envelope.
`test_fused_field_input_honors_selected_component_envelope` retains this
counterexample. It now passes; there are no remaining blockers within this
bounded M2 gate.

The initial six independent core review methods produced ten failing subcases
across the malformed-input, native-representation and ownership defects; those
subcases now pass. Phase-quadrature and signed passive-gate controls passed from
the initial review. A subsequent full rerun found two author tests directly
calling an internal nested-validator entry without its newly required parent;
the tests were updated to supply the true enclosing parent, without weakening
production ownership validation.

## Independent evidence

The core review checks that fixed-phase homodyne does not turn an orthogonal
quadrature into an absolute-amplitude result, and that negative signed gating
uses phase/sign transfer with passive power and paid conversion/control.

The scheduler specialist retained eight tests, including 32 seeded fractional
schedules checked by a separately written event sweep. The sweep verifies every
precedence, release time and resource capacity independently of the production
allocation helper. State tests exercise expiration caused by lock waiting,
bounded and unknown retention, destructive second reads, reset/write/version
chains and unauthorized mutation. Their final focused execution passed 31 tests
(eight review, twelve scheduler, eleven geometry), without skips.

A scalar finite-conductance FFN oracle is retained for the padded-array repair.
It constructs both conductance rails, converter quantization, all installed-row
loads, differential readout and SiLU independently, without importing production
arithmetic helpers. Its high-loading scenario distinguishes full physical
padding from free inactive-row isolation.

Independent frozen compatibility execution passed all **30 checks**, including
integrity of **57 pinned artifacts**, without refreshing goldens. This checks
software compatibility; archived SAX evidence remains archived and is not
claimed as a newly run mixed-matrix comparison.

## Commands and final results

Final independent execution passed **187 multiphysics tests**, including the
**16 independent core and scheduler review methods**, with zero failures or
skips. The selected field-input envelope fix and finite-conductance padded FFN
oracle are included. The final frozen compatibility rerun again passed
**30 checks / 57 artifact integrity checks** with no golden refresh.
Commands executed during review include:

```sh
python -m unittest discover -s tests -p 'test_mp_*.py' -q
python -m unittest tests.test_mp_m2_review tests.test_mp_m2_review_schedule -v
python -m unittest tests.test_mp_m2_review_schedule tests.test_mp_scheduling tests.test_mp_geometry -q
python scripts/check_multiphysics_baseline.py --report /tmp/npp-m2-independent-baseline-final.json
```

Final independent logs and the compatibility JSON are retained under
`runs/multiphysics-validation/m2/independent-review-*`. Earlier failed rerun logs
are retained with round-specific names. Root's broader optional-dependency
integration tests and compiled/replayed example bundles are additional evidence
in the same stage directory; their command logs disclose their own scope.

## Limits

All reference coefficients remain hypothetical. This is review of the declared
software and bounded physical equations, not fabricated-device calibration,
independent circuit validation or a speedup claim. The scheduler uses conservative
non-preemptive whole-task reservations and locks; it does not promise an optimal
schedule. Geometry is an explicit planar electrical-bus skeleton and a local
inclusive coherent assembly, not a manufacturable layout. Unsupported external
optical routing, arbitrary fanout, absent sources/references and unsupported
integration domains reject explicitly. Track R adaptation remains unsupported.
Replay establishes deterministic self-consistency, not independent physics.
