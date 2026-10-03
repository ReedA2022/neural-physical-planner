# Proposed multiphysics experiment registry

This is the M0 registration of H1–H6 in the user-supplied *Neural Physical
Planner — Multiphysics Specification v1.0*, dated 3 October 2026, Sections
12–17 and 20–23. The companion [experiments.json](experiments.json) records
the same contracts in machine-readable form. Its schema is proposed and
documentation-only: neither file is accepted as an executable planner input.

**All six experiments are proposed and not executable in the current
implementation. No experiment has results.** Existing v0.3 optical fanout,
replay and SAX checks do not execute these neural-block or stateful
experiments. No claim of performance improvement, device validation or
novelty follows from this registration. The proposed baselines below have
not been run.

## Shared protocol and unresolved choices

Before running or searching, a scenario must pin its model/checkpoint and
workload identity, semantic track, resource envelope, quality criteria and
evidence policy (REQ-01). A saved checkpoint does not specify architecture,
operator semantics, state policy or request arrivals. Record tensor shapes,
batch/reuse, numeric representations, input ranges, hardware capacities,
weight residency and amortization horizon explicitly. Stateful scenarios
also pin prefill/decode phases, masks, positional encoding, cache policy,
prompt/generated lengths and arrival assumptions.

The registry deliberately leaves numerical sweep domains, resource caps,
quality thresholds, tolerances, task datasets and uncertainty distributions
unresolved. `null` means **an unmet obligation**, never zero, unbounded
permission or an optimistic estimate. Source locations and validity envelopes
must justify any later selected values. These missing choices block execution.
Scenarios may use synthetic fixtures to test software, but research results
must remain separate and label source-parameterized or hypothetical models.

Every experiment must follow these rules:

- Freeze a complete baseline and candidate system boundary before selecting
  winners. Charge compute, distribution, movement, conversion, storage,
  programming, calibration, control and support. Attribute each physical
  effect once, including shared sources/pumps and composite coupling blocks.
- Match memory capacity, compute capacity, communication capacity, power and
  quality constraints where applicable. Report any resource difference
  explicitly. Do not compare logical quantum counts with physical electronic
  latency or energy as if they were the same kind of estimate.
- Derive elapsed time from a resource-constrained schedule, including startup,
  input readiness, storage expiry, contention and reset. Report cold-start and
  steady-state costs with a declared reuse horizon. A saved converter, smaller
  exported tensor or lower local error is not itself an end-to-end speedup.
- Preserve E (unchanged ideal function), A (declared approximation budget) and
  R (new adapted model identity) as separate tracks. Ideal equivalence does not
  imply bitwise equivalence or equal physical error. R additionally needs an
  adaptation/training budget and an adapted digital comparator.
- Separate calibration/search inputs from held-out task evaluation. Publish
  tolerance and confidence conventions before comparisons; report seeds,
  independent random streams, evaluator versions and convergence checks.
  Retain rejected candidates, negative results and known counterexamples.
- Verify semantic behavior with an independent implementation that does not
  call the production evaluator. Verify device/coupling behavior separately
  under matched conditions. Replay is useful but is not independent validation.
- Keep evidence and model fidelity as separate labels, including at parameter
  level. Expose missing costs as incomplete totals. Use separate evidence
  cohorts; propagate shared uncertainties jointly. Distinguish a win at some
  allowed settings from a win at all settings. Overlapping intervals alone
  do not order candidates.
- For small declared spaces, compare the search frontier with an exhaustive
  oracle. Larger searches report budget, seed, coverage and termination; a
  single-technology winner is valid. Do not reward a technology count.

Each executed study must retain normalized inputs, semantic plan, physical
plan, schedule, cost/quality ledger, assumption ledger, matched comparison
and verification artifacts, with input/model hashes and failures. Output
files must distinguish executed measurements, simulated research results,
synthetic software fixtures and proposed scenarios (REQ-36).

## H1 — Partitions inside a linear or feed-forward block

**Question.** Can partitions matched to physical capabilities beat both
single-technology execution and conventional whole-operator mapping?

**Workload and track.** Start with a small linear block, then a complete FFN
with explicit nonlinearities, bias, output ordering and precision. Track E
partitions must preserve the ideal function. Any truncation or approximate
operator creates a separately budgeted Track A scenario; adaptation is not
an implicit fix for a failed equivalent plan.

For disjoint input-index sets, the exact identity is

$$Wx = \sum_j W_{:,I_j}x_{I_j}.$$

Input partitions require accumulation, output partitions ordered
concatenation, and two-dimensional tiling both. Bias is added once. Include
padding, lane utilization, distribution, partial-sum precision and merge.

**Baselines.** Fixed digital; each eligible single compute backend with all
required support; conventional operator-level hybrid; intra-layer hybrid
with complete interfaces; and an exhaustive small-instance oracle using the
same choices, resource boundary and metrics. Unsupported families cannot be
inserted merely to fill a comparison row.

**Sweeps and measures.** Dimensions, batch/reuse, input/weight dynamic range,
converter precision and hardware allocation; complete block latency, energy,
memory and physical resources, output error and held-out task quality. Ablate
partitioning, representation transforms and merge placement independently.

**Negative controls and failure.** Reject overlapping/missing partitions,
incorrect ordering, repeated bias, illegal nonlinear commutations and
unbudgeted approximation. Include a conversion/distribution/merge-dominated
regime and low-utilization cases. A gain that disappears after these costs,
or needs resources outside the matched envelope, is a recorded failure.

**Blocked by.** Complete digital reference, typed mixed operator execution,
reviewed rewrites, explicit converters, resource scheduler, owned accounting,
independent block oracle and bounded search/crossover runner (M1–M4).

## H2 — Interface-fused gating and branch alignment

**Question.** Can a legal gating interface avoid intermediate readout,
electronic multiplication and re-encoding at equal semantic output and
quality?

**Workload and track.** Begin with an explicit, bias-free SwiGLU block:

$$z=\operatorname{SiLU}(W_gx)\odot(W_ux),\qquad y=W_dz.$$

The proposed Track E mechanism keeps the ungated branch optically encoded,
computes the gate electronically and drives a physical modulator before an
optical down-projection. Signed values, gate range, scale, transfer law and
synchronization are obligations. The v0.3 nonnegative intensity modulator
does not establish this signed multiplication contract. Any approximate
transfer law requires a separate Track A quality budget.

**Baselines and alternatives.** Complete digital block; ordinary unfused
hybrid; fused gating without added storage (including delayed launch where
legal); fused gating with optical delay; electronic buffering/re-encoding;
recomputation; and, only once its own model exists, a specified acoustic
buffer. Storage is an optional design choice, not a required acoustic win.

**Sweeps and measures.** Branch skew, gate range, interface nonlinearity/error,
insertion loss, driver/control cost and storage fidelity versus retention;
complete latency and energy, resource use and task/output quality. Report a
crossover surface and maximum tolerable interface overhead or required
storage fidelity. Include writing, retrieval, reset, pumps, sources and
synchronization. A coupling backend owns a joint optoacoustic interaction;
independent optical/acoustic formulas must not count it twice.

**Negative controls and failure.** An attenuation-only nonnegative intensity
modulator must fail arbitrary signed multiplication. Fail incompatible
encoding/scale, absent gain, late or expired input, conflicting storage access
and illegal second retrieval. Include regimes where conversion, alignment,
loss or retrieval makes fusion lose. Moving readout elsewhere does not remove
its charge. An acoustic win dependent on hypothetical retention is a
conditional design target, not a demonstrated acoustic advantage.

**Blocked by.** Signed operator/interface semantics, gate oracle, scheduler,
complete baseline and crossover study (M1–M4); acoustic variant additionally
requires M5 coupling, state, storage and independent reference models.

## H3 — Representation-aware correlated-error placement

**Question.** Can exact channel transformations and placement improve a useful
operating point beyond sensitivity-aware assignment?

**Workload and track.** Use the same complete FFN with held-out inputs and
Track E signed permutations. For permutation $P$ and diagonal sign matrix
$D$, the bias-free SwiGLU transformation is

$$W'_g=PW_g,\qquad W'_u=DPW_u,\qquad
W'_d=W_dP^{\mathsf T}D^{-1}.$$

The sign acts on the ungated branch, never through SiLU; transform any biases
consistently. Re-encoding can change range, programmability and covariance.
For sensitivity $J$, bias $b$ and covariance $\Sigma$, a local proxy is

$$\mathbb E\|\delta y\|_2^2\approx\|Jb\|_2^2+
\operatorname{tr}(J\Sigma J^{\mathsf T}).$$

For signed permutation $S$ and placement $\pi$, evaluate

$$\operatorname{tr}\!\left(JS^{-1}\Sigma_{\pi,S}
S^{-\mathsf T}J^{\mathsf T}\right).$$

Covariance is re-evaluated after encoding changes; the proxy is not a general
quality or nonlinear-noise guarantee.

**Baselines.** Fixed representation; sensitivity-aware assignment;
transformations alone; geometry alone; joint optimization. Match physical
resources and held-out task-quality targets. Sweep shared/local noise,
correlation structure, representation, placement, integration time,
concurrency and calibration policy within valid device envelopes.

**Mandatory controls.** (1) Isotropic additive noise under unchanged resources
must not show a false benefit from signed permutations. (2) For
$e=\xi Sz$, inverse representation gives $S^{-1}e=\xi z$; this signal-following
multiplicative mechanism alone cannot improve. Also include physically shared
additive disturbances and a device-specific model, reject non-PSD covariance,
and retain unrelated local noise when shared modes cancel. Check propagation
with independently sampled noise and test clipping/range changes separately.

**Success and failure.** Lower harmful error must enable a modeled improvement
in integration time, concurrency, calibration or energy at equal task quality
to support an inference-performance claim. Record a proxy-only gain as such;
pure robustness is robustness. A benefit that vanishes after covariance
re-evaluation fails the mechanism.

**Blocked by.** Reviewed signed rewrites, geometry-dependent shared-error
models, independent sampling, held-out quality evaluation and full operating
point/schedule comparison (M6, using M1–M4 foundations).

## H4 — State-local attention and reduction placement

**Question.** Can placement and conversion boundaries improve end-to-end
performance beyond a strong locality-aware architecture?

**Workload and track.** Track E stateful attention with declared prefill and
decode, causal/other masks, scaling, positional encoding, numeric formats and
cache append/read semantics. Regions produce query-dependent summaries
$(m_j,l_j,z_j)$, combined as

$$M=\max_jm_j,\quad L=\sum_j e^{m_j-M}l_j,\quad
Z=\sum_j e^{m_j-M}z_j,\quad y=Z/L.$$

Empty or fully masked partitions need explicit valid-state semantics.
These summaries do not replace an arbitrary persistent cache with constant
storage. Finite-precision changes remain part of quality evaluation.

**Baselines.** Centralized attention; a strong fixed near-memory/local-reduction
implementation; placement alone; joint cache/merge/interface placement. Pin
memory capacity, compute and link resources or expose differences.

**Sweeps and measures.** Context length, cache partitioning, batch size, head
dimensions, local bandwidth and inter-region links. Charge query distribution,
local scans, cache writes/updates, partial traffic and merge. Measure boundary
traffic and local bandwidth alongside total latency and energy; stateful
reports include cold-start/steady-state, time to first token, inter-token and
request latency, throughput under a latency bound, energy per committed token,
power and resource occupancy. Quantiles need declared arrivals and traces.

**Negative controls and failure.** Independent attention oracle covers empty
and masked partitions, cache versions and numerically difficult scores. Test
read/write hazards, expired state and contention. Reject acoustic delay as
persistent cache absent a separately valid model. Traffic reduction without
latency/energy gain, or a win from uncharged memory/compute, fails the claim.

**Blocked by.** Decoder semantics, persistent storage/versioning, locality-aware
digital baseline, resource/geometry scheduler, independent attention oracle
and whole-cycle accounting (M6).

## H5 — Approximate physical draft and digital verifier

**Question.** Does a low-fidelity physical proposer improve committed-token
throughput after full verification costs?

**Workload and tracks.** Start with greedy decoding against an authoritative
digital target, with deterministic tie-breaking and its model identity fixed.
The draft is Track A with a declared fidelity/quality contract. The committed
greedy output must match the target under the specified verification algorithm;
draft approximation is not permission to change the target output. A trained
or distilled draft additionally requires Track R identity, cost and comparator.

**Baselines.** Digital target without speculation; digital draft plus the same
digital verifier; analog/photonic draft plus that verifier. Match memory,
power, quality, model roles and resource capacities; expose any parallel
hardware difference. Include rejected proposals, verifier computation,
interfaces, cache updates/rollback, synchronization and state reuse.

**Sweeps and measures.** Draft fidelity, proposal length, observed acceptance,
interface overhead and state-reuse policy. Acceptance is measured under each
fidelity/workload setting, not freely chosen to manufacture a win. A separate
symbolic acceptance sweep may be reported as conditional. Measure accepted and
committed tokens per full cycle, committed tokens/s and energy per committed
token with end-to-end latency and resource use; never substitute draft tokens/s.

**Negative controls and failure.** Low/zero acceptance, costly interfaces,
proposal rejection/rollback and target/draft disagreements must be handled.
Compare committed greedy tokens with an independent target-only execution.
If acceptance or communication erases the gain, retain the loss. Greedy
correctness does not establish distribution-preserving stochastic generation:
that extension is blocked until a valid probabilistic verification algorithm
and actual proposal probabilities are supplied. Noisy hardware samples alone
are insufficient.

**Blocked by.** Decoder/cache semantics, proposal-verifier algorithm, explicit
draft physical model, scheduler, independent sequence oracle and task suite
(M6). Stochastic distribution preservation is a separate future contract.

## H6 — Output-aware quantum/classical crossover

**Question.** For what specified data access, precision and outputs could a
quantum subroutine help a classical neural workload?

**Workload and tracks.** Choose a small, mathematically specified subproblem
and prove its relationship to the enclosing graph. Register two distinct
output cases: selected classical estimates/observables, and a complete
classical vector. Neither substitutes for the other. Declare Track E ideal
semantics and any Track A estimator approximation/confidence budget per case;
quantum states are not ordinary classical tensors.

**Baselines.** Classical dense and relevant randomized alternatives for each
same output requirement and data-access assumption. Include the enclosing
classical interfaces. A logical-resource study reports queries, depth and
qubits separately from a hardware-mapped study with a named platform, noise,
connectivity, control and (when applicable) error-correction assumptions.

**Sweeps and measures.** State/data preparation, queries, circuit depth, success
probability, estimator precision, shots, measurement/readout and hardware
mapping. Include repeated preparations, ancillas, uncomputation, failed
postselection, resets and classical postprocessing. Shots follow a declared
estimator and confidence target. Physical latency/energy remain incomplete
when mapping or support costs are missing; simulator runtime is not target
device latency.

**Negative controls and failure.** Reject free preparation, unsupported
QRAM/oracles, cloning, simulator-only state readout and free complete-vector
output. Verify normalized positive states, valid channels/instruments and
success/failure probabilities using a small independent reference. Compare
selected-estimate and full-vector costs separately, including input loading
and repetitions. Any advantage requiring a prohibited assumption fails. A
conditional crossover or principled rejection is a valid research result.

**Blocked by.** Separate logical/reference quantum backend and typed classical
boundaries (M5), concrete subproblem/algorithm/access contract, independent
oracle, estimator design and full surrounding-workload accounting (M6).

## Release gates and optional adaptation

M0 registers these definitions; it does not execute them. The minimum coherent
research release needs common contracts, a complete digital path, a genuinely
mixed implementation, explicit storage/timing, evidence-aware accounting,
replay, independent semantic checks and a reproducible crossover study. H1
and non-acoustic H2 are the focused M4 studies. An acoustic H2 variant depends
on M5. H3–H6 research execution is M6 work.

The full initial scope additionally requires all five families—digital
electronics, analog electronics, classical photonics, acoustics/phononics and
quantum—to be expressible without core changes, legal mixed subgraphs
evaluated at their declared model level, illegal mixtures diagnosed, H1–H6
executable when the chosen models are supplied, and missing evidence visible.
Acoustic and quantum packs may begin with validated small semantics and
conditional resources. A measured speedup is not a software acceptance gate.

An acoustic recurrent/filtering extension belongs **only to Track R**. It
needs a new adapted model identity, training/adaptation budget, held-out task
evaluation and an adapted digital model under the same budget. It must never
be described as compilation of an unchanged Transformer.
