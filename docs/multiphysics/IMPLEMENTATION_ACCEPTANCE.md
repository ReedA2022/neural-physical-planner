# M1–M6 implementation acceptance matrix

This is a planning checklist derived from the complete supplied
[specification](SPECIFICATION_V1.md), [migration decisions](MIGRATION.md),
[37-requirement inventory](REQUIREMENTS.md), and the complete
[H1–H6 registry](experiments.json). It does not pass any implementation gate.
The source specification remains authoritative. Each stage requires a separate
reviewer to execute tests, inspect the implementation and issue PASS or HOLD
before implementation of the next dependent stage begins.

The intended deliverable is a bounded, executable theoretical planner. Small
reference models and explicitly hypothetical parameter sets are acceptable;
fabrication-ready layouts, device calibration, a full-size LLM and a demonstrated
speedup are not required. Restricting dimensions, mechanisms and search budgets
is acceptable when guards and reports state the bounds. Replacing required
mechanisms with unexecuted examples, free resources or hard-coded winning scores
is not acceptable. Track R adaptation is optional; unsupported adaptation must
be rejected instead of relabeling an unchanged-model result.

## Common gate record

Every gate records the requirement IDs addressed, exact bounded mechanism,
public API/CLI/example inputs, independent reviewer identity or role, initial
issues, fixes, executed commands and results, and remaining limitations. A
reviewer PASS is software/model review, not measurement or human device review.
Preserve the frozen legacy baseline; never regenerate goldens to hide drift.

All executed scenarios pin normalized model/workload identity, parameter and
evaluator versions, finite input domains, resource limits, E/A/R track, quality
criteria, evidence policy, tolerance/confidence and separated random streams.
An executable scenario must resolve the registry's null obligations explicitly.
New executed records should link to the historical M0 definitions instead of
rewriting the unexecuted M0 registration as historical experimental evidence.

## M1 — contracts, digital reference and first matrix family

Required mechanisms:

- Separate carrier, regime and payload; declare shape/order/multiplexing,
  encoding/sign/scale, units, range/bandwidth, timing/clock/lifetime, shared-error
  identities and controls/environment in the signal contract.
- Versioned component identity, envelope, local state and lifecycle, validation
  references, integration/package assumptions and registered evaluator ID.
  Packs contain data, never executable Python, formula strings or natural
  language interpreted as device equations. Backends depend on common contracts
  and their own libraries, with lazy optional imports.
- Parameter-level provenance/source location, evidence distinct from fidelity,
  known/bounded/unavailable quantities, uncertainty type and correlated-instance
  dependencies. Invalid, unsupported, conditional and model-feasible are distinct;
  numerical failure is not physical infeasibility.
- Unique accounting owners; inclusive composites expose reservations, expanded
  composites exclude children. Unknown costs make totals incomplete. The ledger
  covers compute, movement, conversion, storage, programming, calibration,
  control and support, including explicitly absent/not-applicable categories.
- Complete deterministic digital linear/FFN: bias, SiLU, elementwise product,
  down-projection; declared format, rounding/accumulation/overflow and bounded
  vector/batch conventions. Digital resources include memory, transfers,
  programming, control and support rather than only multiply-add counts.
- One actual bounded analog-electrical or classical-photonic matrix primitive
  with explicit signed representation and readout. Analog conductances require
  range/scaling, converters, programming/drift/loading/read-disturbance semantics;
  photonics requires field/intensity distinction, signed readout, passivity,
  sources, reference/driver and calibration assumptions.
- Immutable or defensively copied ownership boundaries, pinned evaluation
  identity including workload/environment/fidelity/seed/geometry/schedule context,
  and a versioned lifecycle whose unsupported methods reject honestly.

Minimum evidence: independent scalar/digital FFN oracle; zero-imperfection tile
semantics; wrong shape/range/encoding rejection; missing quantity versus true zero;
duplicate ledger effects; envelope violations; finite numerical guards; private
state alias mutation; deterministic seeds; contextual identity changes; optional
dependency isolation. A legacy fanout circuit is not a matrix primitive.

## M2 — typed graph, composition, rewrites and scheduler

Required mechanisms:

- Versioned graph with real typed edges and registered directional DAC/ADC,
  modulation/detection and computing interfaces. Add and review the other
  classical matrix family if M1 supplied only one. Required sources, controls,
  reference signals, clock crossings and runtime transformations are explicit.
- Execute complete mixed linear and FFN paths. Signed fused gating must have a
  transfer law and bounds; retain the unfused detection/multiply/re-encode path.
  An attenuation-only intensity modulator cannot implement arbitrary signed
  gating. Bias is applied once, and branch alignment is enforced.
- Reviewed input/output/two-dimensional partition rules with accumulation and
  ordered concatenation; complete coverage/disjointness, padding/utilization and
  partial-sum precision. Store source/destination/proof guards and runtime costs.
  Signed hidden permutations transform the ungated branch, down-projection and
  corresponding biases, never commute sign through SiLU without a valid rule.
- State owners, versions, creation/expiry/reset, repeatable or destructive reads;
  actual read/write/consume hazards and explicit lifetime while waiting.
- A precedence and capacity scheduler reserving compute, converters, links,
  memory ports, sources, controls and buffers. Include programming, calibration,
  synchronization and static support over elapsed time. Readiness timestamps
  alone do not satisfy the scheduler requirement.
- Geometry skeleton with regions, component/support/state placement, dimensions,
  non-overlap/ports/orientation, legal route/resource context and environmental
  compatibility. Route length affects the selected physical mechanism, and
  geometry/schedule edits invalidate their dependent evaluations.
- Separate multiphysics entry point and normalized friendly files, declaring-file
  relative paths and units; normalized interfaces and assumptions; pinned replay.

Minimum evidence: valid mixed FFN against independent digital outputs; negative
connections for amplitude/intensity, scale/sign, current/voltage, units and clock;
missing source/control and phase-reference mismatches; illegal partitions and
nonlinear rewrites; duplicate bias; finite-capacity contention; stale, expired,
consumed and simultaneous state access; mutually incompatible environments;
tampered replay; legacy compatibility.

## M3 — independent semantic and device validation

Required mechanisms:

- Separate numerical semantic oracles for partitions, signed rewrites, fused
  versus unfused complete FFN and scheduler/state behavior. Do not import the
  production arithmetic through a helper in the alleged oracle.
- A genuine independent device/coupling implementation for the new mixed path,
  using matched input encoding, operating points and output boundaries. Both
  classical matrix families need suitable reference checks. An external solver
  is useful but not mandatory; an independently implemented reduced-order
  simulation is permitted. Prior SAX fanout validates only its historical scope.
- Discrepancy/tolerance reports with retained failures, numerical convergence or
  residual evidence where a solve requires it, validity envelopes and explicit
  limitations. Perturbing either production or reference must detect disagreement.
- Independent stochastic sampling for supported physical error models, explicit
  shared versus independent streams, confidence conventions and adversarial
  input-range tests. No task-quality claim from only a covariance proxy.

Minimum evidence: actual executed matched comparisons and outputs, not exports
alone; ideal/noise/boundary cases; phase/sign/passivity tests; comparison engine
errors and nonconvergence remain failures. Replay and self-consistency results
are labeled separately from independent comparison.

## M4 — designer, bounded search and H1/H2

Required mechanisms:

- Designer edits pin devices, forbid technologies, constrain regions, require
  editable weights, alter physical assumptions and set quality/evidence limits.
  Changes trigger normalization, compatibility, schedule and quality rechecking.
- Bounded hierarchical enumeration over the supported partition, representation,
  device/precision, placement/route, interface/alignment and schedule choices;
  semantic/evidence/envelope pruning before expensive checks. Disclose coverage,
  seed, budget and termination. Exact optimality is only for the enumerated space.
- Cost/quality frontiers distinguish evidence cohorts; conditional structural
  archive is separate from the performance frontier. Report joint uncertainty,
  existential versus robust wins and unresolved interval overlap. No technology
  count reward or silent tuning to reach a requested speedup.
- Complete normalized/semantic/physical/schedule/ledger/assumption/comparison/
  verification bundle and human-readable explanation of crossings, rejections,
  critical assumptions and counterfactual measurements.
- Executable H1: fixed digital, each eligible single compute family with support,
  whole-operator hybrid and intra-layer hybrid; exact tiny search oracle; dimension,
  reuse/range/precision/resource sweeps and partition/representation/merge ablations.
- Executable H2: digital/unfused hybrid, signed fused/no-added-storage, optical
  delay, electronic buffer/re-encode and recomputation alternatives; skew,
  gate/precision/error/loss/control/retention sweeps; acoustic remains explicitly
  unsupported until M5. Derive crossover conditions from full schedules and
  ledgers. Preserve cases where fusion/partitioning loses.

Minimum evidence: an independent exhaustive frontier on identical choices;
input edits actually alter dependent candidates; resource/quality violations
and unknown costs cannot win as zero; reproducible registered H1/H2 with complete
baselines and losing/low-utilization/control-dominated cases. Synthetic studies
are software evidence, not hardware calibration or established scientific gains.

## M5 — acoustic coupling and quantum reference families

Required mechanisms:

- A declared acoustic delay/dynamical model with mode normalization, passband,
  loss, temperature/control/reset, finite independently recoverable capacity and
  fidelity versus retention. Write/store/read and pumps/excitation are charged.
  Joint optoacoustic interaction owns the coupled states/equations once, exports
  typed ports and solver/convergence evidence, and has an independent reference.
- Acoustic H2 alignment integrates with graph state and scheduler: conflicting
  reads/writes, waiting-induced expiration and destructive retrieval matter.
  Compare against already available optical/electronic/recompute paths without
  granting storage gain or random access absent a specified mechanism.
- Quantum regime and physical platform stay separate. A bounded small state or
  density model owns entangled state jointly; channels/instruments have valid
  positivity, normalization/completeness, outcome and post-measurement semantics.
- Logical resources report circuit, qubits, gates/depth, accesses, success and
  output type. Preparation, matrix/data encoding, ancillas, uncomputation,
  repeated attempts, measurement/reset and classical processing are represented.
  Physical timing/energy stay unavailable or conditional without complete mapping.
- Explicit preparation and measurement/estimator boundaries; shots follow an
  error/confidence contract. Selected observables, classical samples, complete
  classical vectors and internal simulator amplitudes remain different outputs.

Minimum evidence: independent acoustic response/coupling and quantum channel/
instrument small references; retention-quality, reread, pump and capacity failures;
invalid channel/state, cloning, hidden-amplitude readout, free preparation/oracle,
failed postselection accounting and missing mapping tests; removing either new
backend leaves old backends unchanged. Declaring quantum physical totals unknown
is acceptable; using a quantum simulator's elapsed time as device latency is not.

## M6 — decoder, state-local attention and H3–H6

Required mechanisms:

- Small executable decoder with explicit architecture and symbolic/tiled shapes:
  linear, residual, chosen normalization, exact chosen position encoding,
  scaled/masked attention, cache append/read, SiLU and multiplication. Reject
  unsupported operations or insert a declared digital fallback.
- Distinct prefill/decode, batch/prompt/generation, state versions and ownership,
  weights residency/reuse, capacity and arrival policy. Resource-constrained
  schedules include programming, updates, rollback, calibration, movement and
  waiting. Persistent cache is not acoustic temporary storage.
- Complete cold/steady cost and quality reports: first-token, inter-token and
  request latency, committed tokens/s under a declared latency bound, energy per
  committed token, peak/average power, memory/traffic and occupancy. Quantiles
  need actual traces and a declared distribution.
- H3: fixed, sensitivity-aware, transforms-only, geometry-only and joint plans;
  re-evaluated physical covariance, PSD validation, independent sampling,
  held-out quality and operating-point costs. Isotropic additive and
  signal-following multiplicative controls must show no spurious transform gain;
  shared-mode cancellation must retain unrelated noise.
- H4: centralized, strong fixed locality-aware, placement-only and joint
  cache/merge/interface plans. Stable query-dependent local attention summaries
  handle empty/fully masked partitions; charge local scans, query/partial traffic,
  cache updates and merge under matched resources. Compare with an independent
  full attention oracle and retain traffic wins that fail end-to-end.
- H5: target-only, digital-draft and physical-draft greedy decoding with fixed
  target and deterministic tie-breaking. Measure acceptance and committed tokens
  after all proposal/verifier/rejection/rollback costs; test zero acceptance and
  expensive interfaces. Independent target-only outputs must match. No claim of
  distribution preservation from greedy correctness.
- H6: a concrete small subproblem related mathematically to the enclosing neural
  workload; selected estimates and complete-vector outputs compared separately
  with dense and relevant randomized classical methods at matched access,
  precision/confidence. Include preparation, shots, failures, resets and output
  reconstruction. Logical resources remain separate from unmapped physical costs.

Minimum evidence: executable, preregistered finite H3–H6 scenario suites, held-out
inputs/tasks separate from search/calibration, explicit synthetic-quality scope,
independent decoder/attention/sequence/quantum controls, retained failures and
losses, fresh-process replay, full prior regression and baseline preservation.
Small synthetic task quality is acceptable for bounded software validation; it
does not establish real-language perplexity or production LLM quality.

## Requirement-to-gate traceability

Each row identifies the first substantive slice and required completion/extension;
passing the first slice alone does not complete a spanning requirement.

| Requirement | First slice | Required extension or completion evidence |
| --- | --- | --- |
| REQ-01 workload/track/resources/quality/evidence | M1 contracts | M2 binding; M4 enforced preregistration; M6 decoder |
| REQ-02 obligations and unknown quantities | M1 quantities/status | M4 joint uncertainty and crossover; all outputs |
| REQ-03 shared contracts, separate physics | M1 architecture | M2 orchestration; M5 genuinely joint coupling |
| REQ-04 single owner per effect/reservation | M1 ledger ownership | M2 reservations/composites; M6 amortization |
| REQ-05 dependency and state isolation | M1 backend boundary | M2 mutation tests; M5 backend-removal tests |
| REQ-06 complete signals | M1 types | M2 checked edges; M5 acoustic/quantum regimes |
| REQ-07 owned versioned state | M2 scheduler/state | M5 buffers/registers; M6 persistent cache |
| REQ-08 explicit representations/conversions | M1 type distinctions | M2 legal explicit operations and proof guards |
| REQ-09 mechanisms and component contracts | M1 components | M2 dynamic integration; M5 new families |
| REQ-10 parameter provenance/uncertainty | M1 parameters | M4 joint uncertainty, sources and sensitivity |
| REQ-11 integration/environment compatibility | M1 assumptions | M2 composition; M5 specialized environments |
| REQ-12 directional registered crossings | M1 declarations | M2 electronic/photonic; M5 broader interfaces |
| REQ-13 guarded interface fusion | M2 rewrite/interface | M3 independent oracle; M4 full H2 comparison |
| REQ-14 lifecycle and coupling protocol | M1 protocol | M2 external time; M5 local dynamics/joint solve |
| REQ-15 full dependency identity | M1 version/hash | M2 geometry/schedule invalidation; M4 cache |
| REQ-16 complete digital baseline | M1 FFN | M2 resource model; M6 complete decoder |
| REQ-17 analog electrical mechanism | M1 or M2 tile | M2 chain/programming; M3 reference |
| REQ-18 classical photonic mechanism | M1 or M2 tile | M2 signed composition; M3 reference |
| REQ-19 acoustic/phononic mechanism | M5 | Separate physics, excitation/readout and isolation |
| REQ-20 usable buffer and retention | M2 generic state | M5 real delay/coupling; M6 wait/quality effects |
| REQ-21 quantum state/channel/regime | M5 | Valid channels/instruments and no-cloning/readout |
| REQ-22 quantum data access/output cost | M5 boundaries | M6 output-aware H6, failures and repetitions |
| REQ-23 reviewed exact transformations | M2 rewrites | M3 independent algebra; M4/M6 guarded use |
| REQ-24 computing interface search | M2 mixed FFN | M3 compare; M4 H2; M5 acoustic option |
| REQ-25 full physical geometry/state | M2 skeleton | M4 legal exploration; M6 local attention |
| REQ-26 decoder and stateful scheduling | M2 scheduler | M6 complete small decoder/prefill/decode |
| REQ-27 categorized complete accounting | M1 ledger | M2 scheduled time; M6 system/reuse/support |
| REQ-28 distinct errors/correlation | M1 taxonomy | M2 shared propagation; M6 H3 controls |
| REQ-29 sampling and held-out quality | M3 sampling | M5 measurement boundary; M6 task evaluation |
| REQ-30 hierarchical joint search | M4 bounded engine | M6 stateful choices; staged checks and coverage |
| REQ-31 separate structural archive | M4 | M5 conditional broader candidates |
| REQ-32 friendly separated inputs | M1 schema | M2 normalization; M4 runnable workflow |
| REQ-33 legacy meaning/pinned versions | M0 baseline; M1 new schema | All gates preserve baseline and replay identities |
| REQ-34 reproducible workflow/artifacts | M2 plan/schedule | M3 validation; M4 bundle; M6 system metrics |
| REQ-35 structured reasons/counterfactuals | M1 statuses | M4 designer explanations; M5/H6 missing mechanisms |
| REQ-36 falsification and independence | Every gate | Negative tests before search broadening; actual independent runs |
| REQ-37 independent reviewed gate record | Every gate | Named role, issues, fixes, executed tests, PASS/HOLD and limits |

The final scope is complete only when all five families are expressible through
these contracts, legal supported mixtures execute at their declared fidelity,
unsupported mixtures fail with reasons, and the chosen H1–H6 scenarios execute
with visible missing evidence. A measured performance improvement is never a
software acceptance condition.
