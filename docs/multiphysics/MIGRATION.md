# Migration from v0.3 to the multiphysics specification

Status: design decisions for future implementation, reconciled against baseline
commit `3ac1f30337753b1930fc538dc9ac58cbc1ac4c70`. M0 adds documentation and
regression tooling. It does not change runtime semantics, package version, input
schemas or supported physical mechanisms. Requirement IDs refer to
[the supplied specification](SPECIFICATION_V1.md).

## Preserve the working baseline

The legacy macro planner accepts canonical schema `0.1` and friendly project
files. Its operations are input, linear, ReLU, add and identity. Its digital and
optical macro estimates and Gaussian error surrogate keep their existing meaning.
The v0.3 physical commands use schema `0.3` for bounded intensity fanout and
regeneration at one selected NN node. They do not implement the NN's operators.

Preserve both APIs and existing weight importers. New workloads must not be
accepted by weakening legacy validators or silently inserting unsupported
operators. A future multiphysics project needs a separate, versioned schema and
an explicit entry point. The string `proposed/npp-multiphysics-1` in the supplied
document is a design placeholder, not a supported schema version.

Saved inputs and plans remain replayable with the evaluator version that produced
them. A migration creates a new artifact with an input-to-output mapping and
unresolved obligations; it never mutates the old record or changes its evidence.
The v0.3 technology pack stays illustrative and unreviewed. Formula references
and historical SAX agreement do not promote its coefficients to measurements.

The baseline checker pins example inputs, schemas, complete deterministic output
records and historical validation evidence. Its default operation cannot refresh
the golden data. Any future intentional compatibility change requires a migration
description, independent review and a separate reviewed baseline update. A failing
check must not be fixed by silently regenerating its expected answer.

## Reuse without changing what existing results mean

| Existing implementation | Reuse in the extension | Required change before broader claims |
| --- | --- | --- |
| `config.py`, `weights.py`, `onnx_import.py` | Friendly units, path handling, tensor loading, strict unsupported-input diagnostics | New workload/model contracts; weights alone still do not define architecture |
| `technology.py` | Evidence references, envelope validation, finite numerical guards | Parameter-level evidence/uncertainty, fidelity axes, accounting ownership, integration obligations and backend IDs |
| `implementation.py` | Explicit component instances, ports, optical fanout, lineage and cost ledger | General signal contracts, state ownership, interfaces, physical neural operators and resource reservations |
| `optical_simulation.py` | Actual SAX adapter and matched optical power comparison | A validated adapter for each new mechanism; existing power checks cannot validate signed arithmetic, noise, electronics or timing |
| `designer.py` | Bounded enumeration, locks, Pareto filtering, rejection reasons and reproducible replay | Partition/rewrite choices, geometry, state/schedule constraints, uncertainty cohorts and crossover reports |
| `checker.py`, output reports and review tests | Tamper detection, strict output validation and explicit claim boundaries | Version-pinned multiphysics records, dependency invalidation and new independent semantic oracles |

Adapters may wrap the current implementation at its existing boundary. They must
not add the old macro total to physical subcircuit totals, count the same source
twice, or imply that the optical fanout represents the linear operator itself.

## Common contracts and separate physical models

Add an opt-in package, provisionally `npp/multiphysics/`, rather than moving every
legacy module. The directory layout below is a proposed allocation of ownership,
not a claim that these modules exist.

| Layer | Owns | Dependency constraint |
| --- | --- | --- |
| Contracts/evidence | Immutable versioned signals, parameters, outcomes, resource and provenance records | No backend libraries or device equations |
| Semantic IR/rewrites | Operators, E/A/R track, ideal relations, proof guards and model identity | No implicit physical carrier or encoding |
| Device backends | Declared capabilities, local equations/state, envelopes and reference exports | Shared contracts and its own optional libraries only |
| Interfaces/couplings | Directional conversions, control inputs and genuinely joint solves | Own joint state; no access to another backend's private state |
| Geometry/environment | Regions, routes, compatibility, shared disturbances and solver context | Explicit contextual inputs to backends |
| Scheduler/state store | Time, capacity reservations, versioned lifetimes and hazards | No inferred physics or hidden state evolution |
| Search/reporting | Legal choices, budgets, cohorts, explanations and bundles | Calls declared interfaces; never invents conversions or costs |

The backend protocol is versioned and exposes describe, validate, instantiate,
evaluate, advance/sample and export/compare responsibilities (REQ-14). Initial
backends can explicitly reject unsupported dynamic evolution. Do not supply fake
successful implementations just to satisfy an interface. A joint coupling block
owns its simultaneous equations and reports solver residuals/convergence.

No pack may execute arbitrary Python, formula strings or natural-language rules.
Packs select registered, reviewed evaluator IDs and supply instance data. Adding
a new mechanism requires code and tests; adding a characterized instance of an
existing mechanism may only require data. Optional solvers load lazily.

## Decisions to settle in M1 and M2

1. **Signals and outcomes.** Separate carrier from classical/quantum regime,
   semantic payload from physical variable, and field amplitude from intensity.
   Declare shape/order, encoding/sign/scale, physical units, bandwidth, clock,
   lifetime and shared-disturbance IDs. Compatibility requires agreement or an
   explicit interface. Use structured invalid, unsupported, conditional and
   model-feasible outcomes, with numerical failures distinct from infeasibility.
2. **Unknowns and costs.** A quantity is known, bounded/symbolic, or unavailable;
   missing information is never zero. Distinguish evidence from model fidelity.
   Every contribution has one owner and a declared system boundary. Inclusive
   composites expose complete reservations; expanded composites exclude child
   costs from the parent. Incomplete totals stay incomplete.
3. **State and time.** Version state references and declare owners, lifetimes,
   read/write modes, consumption and reset. Reserve compute, converters, ports,
   links, buffers and controls. Waiting can invalidate retention. The scheduler
   determines latency; resource throughput numbers do not determine a schedule.
4. **Identity and randomness.** Pin model, pack, component, interface and evaluator
   versions plus normalized inputs. Key evaluations by workload, environment,
   geometry, schedule-dependent context, fidelity and seed policy. Separate
   search and physical streams; shared disturbances use one source identity.
5. **Reference semantics.** Define a small complete digital FFN including SiLU,
   multiplication and bias. Specify numeric formats, accumulation and overflow.
   Check ideal identities separately from finite-precision and physical error.
   Add decoder semantics in M6 through the same operator/state contracts.

The bounded first mixed target is a linear/FFN block. Signed gating is an explicit
new mechanism: the current attenuation-only intensity modulator is insufficient
for arbitrary signed SwiGLU. The first interface comparison must include an
unfused reference, gate/control costs and branch alignment. Acoustic storage is
an optional later branch, not a required ingredient or a free delay.

## Ordered implementation and critique gates

Every row ends with independent agent critique, requested fixes, re-executed
tests and a recorded PASS/HOLD judgment. The reviewer reports its role; agent
review is neither human device calibration nor external physics validation.
Do not begin a dependent stage before its predecessor's gate passes.

| Gate | Implement and deliver | Evidence needed to pass |
| --- | --- | --- |
| M0 | Full source specification, REQ inventory, baseline lock/check, migration plan, H1–H6 definitions | Existing inputs/results preserve meaning; tests pass; requirements/equations retained; independent critique recorded |
| M1 | Shared component/evidence contracts, complete digital FFN reference, one bounded electronic/photonic family | Unknown costs, envelopes, evidence/fidelity, ownership and isolation negative tests; independently computed digital semantics; clearly hypothetical coefficients |
| M2 | Typed implementation graph, directional interfaces, owned state, resources, reviewed partition/representation rules, basic scheduler | Illegal encodings/conversions rejected; mixed linear/FFN fixtures execute; no double bias, hidden sign conversion, overbooking or consumed/expired-state reuse |
| M3 | Independent semantic oracles and a genuine device/coupling comparison for the new mixed path | Actual matched-condition comparison, tolerances, discrepancies, convergence and limits; replay explicitly separated from independent validation |
| M4 | Designer edits, bounded search, evidence cohorts, crossover explanations and H1/H2 runner | Reproducible complete block comparisons, tiny exhaustive oracle, retained losing cases, constraint visibility; no favorable tuning after selecting a winner |
| M5 | Acoustic delay/coupling and quantum logical/small-reference backends | Retention/retrieval/pump accounting; valid quantum channels/instruments, no cloning or simulator-only readout; conditional resources where physical costs remain unknown |
| M6 | Small stateful decoder, local attention, timing/cache semantics and H3–H6 campaign | End-to-end quality/resource accounting, independent attention controls, correlation controls, committed-token measurement and output-aware quantum comparison |

The detailed requirement-to-gate mapping is in [REQUIREMENTS.md](REQUIREMENTS.md).
Some requirements span gates: e.g. M1 defines state contracts, M2 checks a basic
stateful schedule, and M6 implements decoder cache workloads. A gate passing its
bounded slice does not mark the whole requirement complete.

Selecting one matrix family at M1 does not drop the other classical family from
scope. The remaining analog-electrical or photonic tile must receive the same
component-contract review in M2 and an appropriate reference check in M3 before
its mixed fixtures or H1/H2 comparisons can run. The current optical fanout pack
cannot fill the photonic matrix-tile requirement. All five families remain
mandatory for completion of the full initial specification.

M4's H2 campaign initially compares supported non-acoustic alignment paths and
records acoustic options as unsupported. M5 can add a valid acoustic option, and
M6 extends the study. Likewise H6 is defined at M0, gains small reference models
in M5, and receives its workload-integrated campaign in M6. These dependencies
must remain explicit in reports instead of silently dropping unavailable cases.

## Research release and scientific claims

A first coherent research release needs common contracts, a complete digital
path, a genuinely mixed implementation, explicit storage/timing, complete or
explicitly incomplete accounting, replay, independent semantic checks and a
reproducible crossover study. M4 is a candidate point for the bounded FFN study
only if all those conditions are met. It does not complete the decoder or the
full five-family specification.

The full initial scope requires five families expressible through the common
contracts, legal mixed subgraphs evaluated at their declared model level,
diagnostics for unsupported mixtures, and executable H1–H6 where their declared
models are supplied. All outputs expose missing evidence. No release gate
requires a speedup; a reproducible loss or conditional threshold is a valid
research result. A draft paper must identify which experiment actually ran and
must not infer novelty from adding more technologies.
