# Neural Physical Planner: multiphysics exploration specification

> User-supplied requirements, version 1.0, 3 October 2026. This readable transcription
> was generated from the [original DOCX](source/Neural_Physical_Planner_Multiphysics_Specification_v1.docx).
> The original is retained unchanged, including its equations and source links.
> The document's v0.2 snapshot is historical; see [REQUIREMENTS.md](REQUIREMENTS.md)
> for reconciliation against the actual v0.3 code. Proposed configuration is **not
> accepted by the current CLI**. Publication descriptions below are the supplied
> specification's claims, not a new literature audit or novelty assessment.

**Neural Physical Planner**

**Multi-physics exploration specification**

**Theory-first composition, placement, geometry and scheduling**

Digital electronics · Analog electronics · Classical photonics · Acoustics and phononics · Quantum computation

**Version 1.0 \| 3 October 2026**

Prepared for the **neural-physical-planner** project.

The objective is to discover whether heterogeneous physical implementations can reduce the cost of neural inference by changing the decomposition of the computation, the placement of persistent state, and the operations performed at physical interfaces.

The planner must permit unconventional combinations without permitting inconsistent physics. Mathematical semantics, device behavior and cross-domain coupling remain separately specified and separately testable.

**Document status: proposed requirements.** This document does not report an implemented multi-physics planner, a fabricated device, a validated hardware speedup, or a verified novelty claim. Proposed designs are hypotheses to evaluate. Literature precedents are identified where relevant.

**Primary deliverable:** reproducible candidate implementations, their assumptions, their trade-offs, and the conditions under which they outperform or lose to strong baselines.

# 01 \| Purpose and success criteria

The planner should answer a more ambitious question than “which accelerator should execute this layer?” It should ask: **which equivalent computation, physical representation, component mixture, geometry and execution schedule provides the best justified trade-off for a specified workload?** A single layer may be partitioned among several technologies; successive operations may remain in one physical representation; and an interface may perform useful arithmetic instead of only converting a signal.

## Questions the system must make testable

We want to investigate intra-layer heterogeneous partitions, interface-fused computation, representation-aware placement under correlated errors, state-local attention and reductions, approximate proposal engines with precise verification, and carefully delimited quantum or acoustic subroutines. A winning architecture is not required to use every available technology. A single-technology plan is a valid outcome.

## Three separate semantic tracks

| **Track**                       | **What may change**                                                           | **Required claim boundary**                                                                 |
|---------------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| **E: equivalent computation**   | Algebraic rewrites and physical realization, with an unchanged ideal function | Exactness is mathematical; actual finite-precision and physical errors are still evaluated. |
| **A: controlled approximation** | Quantization, truncation, approximate operators or reduced fidelity           | A declared error or task-quality budget is required; approximation is never labeled exact.  |
| **R: model adaptation**         | Fine-tuning, distillation, altered nonlinearities or architectures            | Export a new model identity and include adaptation cost and task evaluation.                |

## Definition of a useful result

A useful theoretical result identifies a mechanism and a crossover region: for example, when a fused interface saves more conversion work than its synchronization and control cost. It includes a strong baseline, matched system boundaries, uncertainty analysis and a counterexample where the mechanism loses. A negative result that rules out a design family under stated assumptions is also useful.

**REQ-01.** Every exploration must specify its workload, semantic track, hardware-resource envelope, quality criteria and evidence policy before search. These fields cannot be inferred from a desired speedup.

**REQ-02.** Report both benefits and unresolved obligations. Unknown parameters remain symbolic or bounded assumptions, never zero-cost defaults. A theoretical advantage is not a measured deployment advantage.

**First target:** one feed-forward block and one stateful attention workload, with modular electronic and photonic implementations; acoustic and quantum models enter through the same contracts but retain their own physics.

# 02 \| Scope, baseline and reading map

## What is established in the inspected project

The repository README inspected on 3 October 2026 describes v0.2: user-friendly project specifications, saved-weight import, a small neural graph, digital/optical macro choices, fan-out recipes, a dependency schedule, Pareto search, error analysis, replay checking and sampling. It explicitly identifies its example coefficients and optical noise model as synthetic. It does not claim Transformer synthesis, general persistent storage, detailed routing, resource sharing or throughput optimization. This is a documentation snapshot, not a fresh code audit. \[1\]

Ongoing implementation work may already exceed that snapshot. The implementer must inspect the active branch and reconcile this specification with the existing four-part engineering milestone: reviewed technology packs; typed implementation graphs; independent simulator adapters; and designer edits with explanations. Preserve working capabilities rather than rebuild them under new names.

## Scope of the proposed extension

| **In scope for theoretical exploration**                                                             | **Outside the initial claim**                                              |
|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| **Symbolic resource models, reduced-order device models and small executable reference simulations** | A universal first-principles simulator spanning all technologies           |
| **Heterogeneous blocks within layers and across subgraphs**                                          | Fabrication-ready layouts, sign-off timing or guaranteed manufacturability |
| **Explicit geometry, interconnect, storage, control and noise dependencies**                         | Performance inferred only from a device’s advertised operations per second |
| **Classical and quantum subgraphs with different semantics**                                         | A quantum block treated as a drop-in classical matrix engine               |
| **Hypothetical devices with bounded, labeled assumptions**                                           | Hypothetical coefficients mixed silently with characterized coefficients   |

## Reading map

Sections **03–07** define the compartmentalized architecture and interfaces. Sections **08–11** specify each physics family. Sections **12–17** define transformations, geometry, execution, errors and search. Sections **18–19** define the user workflow and outputs. Sections **20–23** define experiments, tests and implementation gates. Section **24** provides a worked contract. Sections **25–26** contain sources and claim boundaries.

## Prior-art boundary

Fine-grained heterogeneous layer mapping already exists in ODiMO; hybrid photonic–digital attention appears in HyAtten; and model adaptation for analog–digital mapping has also been studied. The research contribution must therefore be a particular mechanism or demonstrated capability, not merely the presence of multiple technologies. \[2,3,14\]

# 03 \| Compartmentalize the physics

**REQ-03. The planner must have one shared contract layer, not one shared physics model.** Each technology family owns its equations, internal state, validity conditions and local validation. The search engine manipulates declared capabilities; it does not invent physical conversions.

| **Compartment**                     | **Owns**                                                                          | **Must not do**                                              |
|-------------------------------------|-----------------------------------------------------------------------------------|--------------------------------------------------------------|
| **Neural semantics**                | Tensor operations, state transitions, permitted rewrites and quality requirements | Assume a carrier, voltage, photon budget or quantum encoding |
| **Device backends**                 | Electronic, optical, acoustic or quantum state evolution and local constraints    | Read or mutate another backend’s private state               |
| **Interface and coupling backends** | Transduction, measurement, modulation and genuinely joint interactions            | Act as an unchecked type cast or silently erase errors       |
| **System orchestration**            | Placement, routing, resources, memory, clocks, scheduling and cost aggregation    | Replace missing device physics with a favorable score        |
| **Evidence and verification**       | Provenance, model versions, assumptions, tests and independent comparisons        | Promote a simulated result to a measured one                 |

## Isolation is not an assumption of independence

Shared temperature, power supplies, lasers, vibrations and clocks must appear as explicit environmental variables or coupling models. A photonic backend may expose its response to temperature while a thermal model computes that temperature from the layout and power schedule. Neither should import the other’s private implementation.

When an interaction cannot be represented by sequential black boxes, use a **joint coupling block**. A Brillouin interaction, for example, may jointly own optical and acoustic state. Its internal equations are solved together; its external ports remain typed. This is coherent compartmentalization, not an obligation to pretend the physics is separable.

## Single ownership of every physical effect

**REQ-04.** Each loss, delay, stored-energy term, noise source and resource reservation has exactly one accounting owner. A composite block may hide its internal parts from the global scheduler only if it exports a complete resource summary. Otherwise, expose those parts and exclude them from the parent’s incremental cost. Never count both.

**REQ-05.** The core may depend on backend interfaces; backends may depend on shared schemas and their own libraries. A backend must not depend on another backend’s implementation module. Optional dependencies are loaded lazily. Adding an acoustic component must not require editing photonic equations or installing a quantum simulator.

**Acceptance:** a deliberately invalid cross-domain connection fails before optimization; a valid joint block executes through its declared coupling contract; adding a new backend leaves unchanged-backend regression results unchanged within fixed tolerances.

# 04 \| Typed ports, signals and state

**REQ-06. Every edge must carry a complete signal contract.** A tensor shape is necessary but insufficient. Identical floating-point arrays can represent optical field amplitude, electrical current, acoustic displacement or simulator-only quantum amplitudes; these are not interchangeable physical signals.

| **Contract field**              | **Required meaning**                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------|
| **Semantic payload**            | Tensor, scalar, control message, probability sample, quantum register or state reference |
| **Physical carrier and regime** | Electrical, optical or acoustic; classical digital, classical analog or quantum          |
| **Encoding and decoding**       | Mapping between semantic values and physical variables, including scale and sign         |
| **Units and normalization**     | SI dimensions or an explicit reference normalization with physical units                 |
| **Shape and multiplexing**      | Tensor axes, lanes, wavelengths, frequencies, time slots and ordering                    |
| **Range and bandwidth**         | Valid input envelope, spectral support, sample or symbol rate, saturation limits         |
| **Timing and lifetime**         | Arrival interval, integration window, clock relation, retention and reset conditions     |
| **Error and provenance**        | Relevant bias/noise model and identifiers of inherited shared disturbances               |
| **Control and environment**     | Bias supplies, pumps, references, temperature and other required services                |

## Different output types stay different

A classical numerical vector, samples from a distribution, and an amplitude-encoded quantum state are different interface types. A quantum simulator’s internal statevector is diagnostic information, not a physically available classical output. Converting it into a tensor requires an explicitly modeled measurement or reconstruction procedure.

Classical fan-out is also physical: digital copies consume transport/storage resources; analog splits alter loading or signal power unless compensation is modeled. Quantum fan-out must not clone an arbitrary unknown state. A preparation recipe may be repeated at its actual cost; copying known classical preparation parameters is a different operation.

## Stateful objects

**REQ-07.** Model weights, cache entries, stored charges, optical/acoustic excitations, quantum registers and calibration state as versioned objects with ownership and lifetime. Declare whether reads are destructive, repeatable, nondestructive within a bound, or unsupported. A second read cannot silently recover a consumed state.

**REQ-08.** Every implicit normalization, transpose, sign encoding, sampling change or domain conversion must either be proven compile-time-only or become an explicit graph operation. Reject connections that require undocumented conventions.

**Acceptance:** an intensity port cannot feed a coherent-field input without a legal encoder; an unknown quantum state cannot feed two independent consumers as copied data; unit and clock mismatches produce actionable diagnostics.

# 05 \| Component and evidence contracts

**REQ-09. A technology pack must describe mechanisms, not just performance numbers.** A component declaration must contain the following groups.

| **Group**                 | **Minimum contents**                                                                     |
|---------------------------|------------------------------------------------------------------------------------------|
| **Identity**              | Pack/component versions, mechanism, implementation family and evidence references        |
| **Mathematical behavior** | Supported operators, ideal transfer map or state transition, parameters and conventions  |
| **Physical realization**  | Ports, encoding, dimensions, capacity, geometry assumptions and required controls        |
| **Operating envelope**    | Ranges of shape, temperature, signal power, rate, precision and configuration            |
| **Dynamic behavior**      | Startup, programming, settling, integration, reset, retention and calibration            |
| **Imperfections**         | Bias, random errors, signal dependence, drift, clipping, coupling and uncertainty        |
| **Accounting**            | Incremental resources and costs, inclusive/exclusive boundaries and amortization rules   |
| **Validation**            | Reference implementation, numerical tests, simulator comparisons and known failure cases |

## Evidence and fidelity are separate axes

**Evidence labels:** hypothetical assumption; source-parameterized; independently simulated; characterized against measured data; measured instance. A parameter may have its own label. A whole component does not inherit “measured” status because one coefficient was measured.

**Model fidelity:** algebraic ideal; analytical cost model; reduced-order dynamical model; circuit/device model; measured surrogate. Greater detail does not automatically mean stronger evidence. A fitted surrogate outside its characterized envelope is not validated there.

**REQ-10.** Store each parameter’s value or domain, unit, source location, uncertainty type, valid range and dependencies. Separate irreducible/random variability from uncertainty about the model. Preserve correlations between parameters; independently combining the best number from incompatible devices is forbidden.

## Missing information and composability

A missing quantity may remain symbolic, yield a bounded result, or make a candidate non-evaluable. It must never become zero by default. Distinguish **invalid**, **unsupported**, **conditional** and **model-feasible** outcomes. “Model-feasible” means feasible under the stated model, not fabrication-ready.

**REQ-11.** Component packs must include integration assumptions: substrate/package compatibility, temperature region, available controls and interface family. Two individually plausible devices do not form a plausible system unless their interface and operating conditions also match.

**Acceptance:** a report can trace every cost-driving assumption to its source or explicit hypothesis and can explain which missing measurement would most change the conclusion.

# 06 \| Explicit cross-domain interfaces

**REQ-12. All domain crossings must be registered blocks with direction-specific contracts.** Reversing an arrow does not imply the existence of a reverse device with identical cost, bandwidth or noise.

| **Connection**                  | **Candidate mechanism**                                | **Required obligations**                                                   |
|---------------------------------|--------------------------------------------------------|----------------------------------------------------------------------------|
| **Digital ↔ analog electrical** | DAC / ADC with drivers and readout                     | Precision, sampling, loading, settling, power and saturation               |
| **Electrical → optical**        | Modulation and source/driver path                      | Optical reference, sign/phase encoding, bandwidth and control cost         |
| **Optical → electrical**        | Detection and receiver                                 | Field-versus-intensity semantics, noise, bandwidth and readout             |
| **Optical ↔ acoustic**          | Pump-assisted optoacoustic interaction                 | Coupled modes, matching conditions, pumps, retention and retrieval         |
| **Electrical ↔ acoustic**       | Specified piezoelectric/electromechanical transducer   | Impedance, coupling, passband, loss, excitation and sensing                |
| **Classical → quantum**         | State preparation or coherent data-access construction | Encoding, normalization, ancillas, preparation cost and access assumptions |
| **Quantum → classical**         | Measurement instrument and estimator                   | Output type, shots, error/confidence, postselection and reset              |

These are families to model, not blanket declarations of compatibility. Photoelectric computation and coherent photonic–phononic storage have experimental precedents, but their particular mechanisms and envelopes cannot be generalized into arbitrary free conversions. \[4,8\]

## Interfaces that compute

A computational interface declares both its physical action and its semantic function. A modulator controlled by an electrical gate may implement a multiplication on an optical representation only if the required transfer law, sign handling and timing are satisfied. A detector with a nonlinear response is not an exact SiLU merely because it is nonlinear.

**REQ-13.** The planner may replace a sequence of operators and conversions with a fused interface only through a reviewed rewrite rule with shape, range, representation and timing guards. Store the unfused reference, all consumed controls, and the physical error introduced by the fused form.

## Common invariants

A passive block cannot create signal energy without a declared source. Sources, pumps and controls are counted at the selected system boundary. Routing and buffering cannot precede their causes. Global error-source identities survive splits, conversions and reconvergence unless a physically justified state transformation changes them. Measurement is a state-changing operation, not transparent transport.

**Acceptance:** any claimed interface-fusion advantage must improve the complete implementation cost, not merely remove conversion nodes while leaving their work unaccounted for elsewhere.

# 07 \| Backend API and coupling discipline

**REQ-14.** Use a small, versioned backend protocol. The following names express required responsibilities, not existing public APIs.

| **Responsibility**   | **Contract**                                                                                |
|----------------------|---------------------------------------------------------------------------------------------|
| **Describe**         | Return capabilities, port types, supported model levels and validity guards.                |
| **Validate**         | Accept or reject an instance with structured reasons and unmet obligations.                 |
| **Instantiate**      | Bind parameters and create backend-owned state without hidden global mutation.              |
| **Evaluate**         | Return ideal behavior, cost model, error model, resources and declared dependencies.        |
| **Advance / sample** | Evolve state or produce reference samples over an explicit time interval and random stream. |
| **Export / compare** | Produce an adapter input and compare independent outputs under matched conditions.          |

## The orchestrator owns time; the backend owns local evolution

An event scheduler reserves resources and advances logical time. A continuous-time backend may integrate between events. Clock-domain crossings and multi-rate resampling are explicit. Internal timesteps may differ, but the coupling contract specifies exchange times, interpolation, accuracy tolerances and causality.

If two components form a tightly coupled physical system, their coupling block owns the simultaneous solve. If thermal, electrical or acoustic feedback requires iteration, record solver convergence and residuals. A failed fixed point is an unresolved or invalid evaluation, not a favorable approximate result.

## Deterministic orchestration and reproducible randomness

Randomness must enter through explicit seeds and stream identifiers. Shared physical noise is sampled once per declared event/trajectory and distributed to its consumers. Independent noise uses independent streams. Search randomness and physical randomness are recorded separately.

**REQ-15.** Cached evaluations are keyed by component version, parameterization, workload, environmental state assumptions, fidelity, seed policy and geometry-dependent context. A layout or schedule change invalidates dependent costs and coupling estimates.

## Failure semantics

Backends return structured categories: domain mismatch; envelope violation; unavailable model; numerical failure; resource infeasibility; or successful conditional evaluation. A numerical failure cannot be reported as a proof of physical infeasibility. Unsupported candidates may be retained in a separate research backlog with the missing mechanism identified.

**Acceptance:** the same normalized inputs reproduce the same deterministic outputs; a changed coupling assumption invalidates the correct cache entries; no backend can alter another backend’s private state through shared mutable objects.

# 08 \| Electronic backends

## Digital electronics: reference path and real resource model

**REQ-16.** Provide digital operators for the complete initial workload, including unsupported-elsewhere nonlinearities and control. Specify numerical format, rounding, accumulation precision and overflow. Declare the distinction between matrix–matrix and matrix–vector execution, supported tiling, resource sharing and data locality.

Account for arithmetic, SRAM/DRAM accesses, weight transfers, cache updates, interconnect, synchronization, launch/control overhead and static power over elapsed time. The same digital reference must be available as a complete baseline, not only as an expensive fallback that makes unconventional devices look better.

A CPU or GPU reference execution establishes semantics and, when actually measured, a software performance point. It does not calibrate a hypothetical ASIC simply because both execute the same operator.

## Analog electronics and compute-in-memory

**REQ-17.** A conductance-based matrix primitive may use a relation such as currents obtained from a conductance matrix and applied voltages, but its model must also account for the physical realization of negative weights, input scaling and output readout. Declare programmed conductance range, tile size, precision, programming error, drift, read disturbance and relevant line/loading effects.

DACs, ADCs, current integration, reference generation, accumulation and weight programming must remain visible in the resource ledger. A partial sum from several tiles needs a declared accumulation path. Charge-domain, current-mode and resistive-memory implementations must not share one unexplained “analog efficiency” coefficient.

## Stored weights versus dynamic operands

Weights may be persistent over many inference calls; queries, keys and attention scores vary by request and token. A component optimized for slowly programmed weights cannot receive a fresh dynamic matrix at zero cost. Programming, reuse count, update latency and endurance assumptions must be explicit.

Mixed analog–digital mapping and adaptation already have direct precedents; their existence motivates strong mapping baselines rather than a novelty claim for the mixture. \[14\]

## Initial reference models

Start with a deterministic digital kernel, a bounded-size analog matrix tile with an explicit converter chain, and a shared memory/interconnect model. Use synthetic coefficients only in labeled validation fixtures. Add source-parameterized packs separately.

**Acceptance:** zero-noise analog execution agrees with its ideal operator under supported encoding; the same tile cannot be used concurrently by two tasks without declared parallel capacity; changing readout precision changes both the numerical and resource models.

# 09 \| Classical photonic backend

**REQ-18.** Treat optical carrier, encoding and computational mechanism separately. At minimum distinguish coherent-field computation, intensity-based computation and any time/frequency multiplexing. Classical optical coherence is not quantum entanglement.

## Signals and operators

A field-linear device acts on complex field amplitudes. Direct intensity detection is nonlinear in that field. A signed real output requires a specified representation and detection arrangement; it cannot be recovered from intensity by an undocumented sign convention. Passive transformations must respect the chosen physical normalization and energy constraint. Gain requires an active resource with its own noise and power model.

Initial primitives should include a specified linear tile, split/combination, propagation, modulation, detection and an optical source/control path. Optical linear tiles may have different realizations; their dimensional constraints, programmability, losses and calibration costs must remain distinct.

## Geometry and operating point

Declare wavelength or band, supported mode structure, routing loss, bends, crossings, coupling losses, fan-out and permitted phase-reference relationships. Capacity is not merely a number of abstract channels: wavelength spacing, passband and crosstalk may limit concurrent channels.

Latency includes data preparation, weight configuration when needed, propagation, settling, integration and readout. Source electrical power, modulation drivers, tuning, stabilization and static support costs belong in the selected system boundary. Do not claim system energy from signal photons alone.

## Noise and drift

Use a mechanism-specific model for relevant source, detector, phase and thermal disturbances. Retain correlated errors where components share sources or environmental influences. Calibrating a component consumes time and resources and may be required repeatedly during a workload.

DOCTOR provides a relevant precedent for photonic variation modeling, sensitivity-aware remapping and remediation. It is a baseline inspiration, not proof that our broader representation-and-geometry hypothesis works. \[5\]

## Initial support boundary

Provide one analytically transparent classical optical tile and one independent reference simulation for small instances before adding multiple competing tile families. The legacy Gaussian optical surrogate remains available for regression, but is explicitly a legacy surrogate rather than the universal photonic model. \[1\]

**Acceptance:** field and intensity are never confused; passive fan-out accounts for power; phase-sensitive combinations reject incompatible references; all assumed converters and calibration operations appear in the schedule and costs.

# 10 \| Acoustic and phononic backend

**REQ-19.** Include sound-based components as a real exploratory family, with a narrower initial role than a general-purpose LLM arithmetic engine. Distinguish bulk/surface acoustic waves, confined mechanical modes and optoacoustic interactions when their equations differ. “Sound” is not one universal transfer function.

## Useful hypotheses to investigate

The first candidate is a **temporary physical buffer or delay** that aligns two branches without an optical-to-digital-to-optical round trip. A second is an explicitly specified linear filter or weighted temporal combination. A third, in the model-adaptation track only, is a dynamical recurrent element.

These are motivated by existing mechanisms, not newly established effects. Merklein and colleagues demonstrated coherent optical information storage in acoustic waves on a chip. The OREO work demonstrated an optoacoustic recurrent operator using acoustic persistence to link optical pulses. Neither establishes a faster unchanged LLM or an arbitrary-capacity key–value cache. \[7,8\]

## Required model contents

Specify mode amplitudes and their normalization, dispersion/passband, propagation or resonant dynamics, loss, coupling, excitation/readout, temperature dependence and reset. For a reduced-order model, identify whether it is a delay-line response, impulse response or finite-dimensional dynamical system, and state where that approximation is valid.

**REQ-20.** Every buffer declares usable delay, bandwidth, number of independently recoverable symbols/modes, storage fidelity versus time, read/write conflicts and retrieval behavior. Retention must be sufficient at the required quality, not merely nonzero. A long lifetime alone does not imply large usable memory capacity.

A proposed acoustic alignment path must pay for writing, storage loss, reading, control pulses or electrical excitation, and synchronization. Compare it with optical delay, electronic buffering and recomputation. Slow propagation can enable delay in a compact path; it does not make arithmetic intrinsically faster.

## Strict semantic boundary

A lossy delay cannot masquerade as exact random-access memory. A temporal filter cannot replace a dense matrix or attention operator without a valid decomposition or an explicitly approximate/adapted model. Classical acoustic amplitudes are not quantum states merely because phonons are involved.

**Acceptance:** a branch delayed beyond its declared retention fails its quality guard; a second retrieval obeys the component’s read semantics; removing acoustic support leaves the electronic/photonic system unchanged; no acoustic gain is assumed in the baseline fixtures.

# 11 \| Quantum backend and classical boundaries

**REQ-21.** Quantum is a computational regime, not a peer carrier to “optical” or “electrical.” A quantum backend must specify its physical platform separately. A classical optical field is not automatically a photonic quantum register, and a classical acoustic mode is not automatically a quantum memory.

## Two model levels, clearly separated

A logical-resource model reports circuit structure, logical qubits, depth, gate families, data-access assumptions, success probability and output specification. A hardware-mapped model additionally supplies physical qubits or modes, connectivity, operations, noise, control, error-correction resources where applicable, measurement/reset and system-support costs. Logical counts alone do not produce credible joules or wall-clock seconds.

Quantum Transformer is a relevant theoretical precedent: it constructs quantum subroutines under fault-tolerant assumptions and represents outputs in quantum form. It is not evidence that an arbitrary quantum block inserted between classical layers accelerates inference. \[9\]

## State and operation contracts

Represent small reference states as statevectors or density operators inside the backend. Physical operations must be valid channels; measurements are instruments with outcome probabilities and conditional post-measurement states. For a finite-dimensional channel represented by Kraus operators, require the appropriate completeness relation. \[10\]

$$\mathcal{E}(\rho) = \sum_{k}^{}K_{k}\rho K_{k}^{\dagger},\quad\quad\sum_{k}^{}K_{k}^{\dagger}K_{k} = I.$$

Trace-decreasing branches require explicit success/failure outcomes and probabilities. Entangled subsystems retain their joint-state ownership. They cannot be treated as independent classical noise vectors.

## Data movement is part of the algorithm

**REQ-22.** Charge for state preparation, matrix/data encoding, ancillas, uncomputation, measurement, repeated preparations, resets and classical postprocessing. QRAM or an oracle is a declared resource assumption, never free implicit access to model weights. Full-vector reconstruction and estimating a few observables are different tasks with different costs.

Postselection must include failed attempts and its effect on output statistics. Shot requirements must follow a declared estimator and confidence target. Nonlinear amplitude manipulation requires a concrete admissible subroutine, not an arbitrary nonlinear map on an unknown state.

**Acceptance:** preserve positive, normalized quantum states within stated numerical tolerances; reject cloning and simulator-only readout; distinguish simulator execution time from target-device latency; show query-count results separately when physical timing or energy remains unknown.

# 12 \| Equivalent transformations and layer partitioning

**REQ-23.** Search over reviewed transformation rules with explicit mathematical preconditions. Store each rewrite’s source subgraph, destination subgraph, proof rationale, numerical test oracle and cost of any runtime transformation.

## Intra-layer partitions

For a partition of input indices into disjoint sets, a linear operation may be implemented by separate physical branches and a reduction:

$$Wx = \sum_{j}^{}W_{:,I_{j}}x_{I_{j}}.$$

Input partitions require accumulation; output-channel partitions require concatenation in the correct order. Two-dimensional tiling requires both. Bias is added exactly once. Include input distribution, lane utilization, padding, partial-sum precision and the merge path. ODiMO already explores intra-layer heterogeneous mapping, so this is a foundational capability and baseline, not a novelty claim. \[2\]

## Exact representation freedom in a feed-forward block

For a bias-free SwiGLU block, write:

$$z = SiLU\left( W_{g}x \right) \odot \left( W_{u}x \right),\quad\quad y = W_{d}z.$$

With a hidden-channel permutation P and a diagonal sign matrix D, the following transformed weights preserve the ideal function:

$$W\prime_{g} = PW_{g},\quad W\prime_{u} = DPW_{u},\quad W\prime_{d} = W_{d}P^{\mathsf{T}}D^{- 1}.$$

The sign is applied on the ungated branch, not pushed through SiLU. Biases, when present, need the corresponding transformation. These identities are algebraic design freedoms; their physical usefulness depends on encoding, error structure and placement.

## General transformations and approximation

An invertible change of basis can rewrite a linear map, but its implementation, inverse conditioning, range expansion and compatibility with neighboring operations must be modeled. Do not commute it through nonlinearities or normalization without a valid rule. A transformation absorbed into weights may still alter their physical programmability and dynamic range.

Low-rank-plus-residual or structured decompositions must retain the complete residual for mathematical exactness. Truncating or approximating it selects Track A or R. Keeping a full dense residual can eliminate the hoped-for savings; the planner must account for that rather than reward the decomposition symbolically.

**Acceptance:** verify rewrites with an independent digital oracle and adversarial shapes/ranges. Never equate ideal algebraic equivalence with bitwise equality under changed rounding or with equal physical error after re-encoding.

# 13 \| Computational interfaces: the first worked hypothesis

**REQ-24.** Make interface fusion a first-class search decision. The first focused example is a mixed-physics feed-forward block, not an entire speculative multi-technology chip.

## Candidate: electronically controlled optical gating

The required computation is the SwiGLU block in Section 12. One proposed implementation keeps the ungated branch u optically encoded, computes the gate g electronically, and uses g to control a physical modulator acting on u. The resulting representation feeds an optical down-projection.

The alternative to beat is an ordinary hybrid chain that reads out intermediate values, multiplies them electronically and re-encodes the result. The hypothesis is that a valid computational interface avoids enough intermediate materialization and conversion to offset its control, encoding and timing costs.

ACCEL demonstrates that photoelectric interfaces can participate in useful neural computation, but in a vision architecture with its own training and physical transfer functions. It does not validate this SwiGLU implementation. \[4\]

## Obligations before a candidate is considered valid

The implementation must specify signed values and gate range. An attenuation-only intensity modulator is insufficient for arbitrary signed multiplication. The optical representation and electrical control must have compatible scaling and synchronization. Driver response, phase/amplitude effects, insertion loss, calibration, noise and any required gain must be modeled.

The planner must decide how to align branch arrivals: start one branch later, use a physical delay, buffer and re-encode, or recompute. An acoustic buffer is one optional candidate, with retention and retrieval costs; it is not mandatory. A joint optoacoustic interaction must use a coupling backend, not independent optical and acoustic formulas that double-count the same interaction.

## What counts as an advantage

For an idealized parallel partition, latency has a distribution/input term, the slowest branch and the merge term. The real evaluator must additionally model contention, calibration and resource reservations. If removing a converter only transfers its work to a different readout circuit, that work remains charged.

**Acceptance:** compare fused and unfused implementations with identical semantic output and matched resource/quality budgets. Demonstrate a crossover surface over interface overhead, branch skew and precision. Include cases where conversion or synchronization makes fusion lose. Report any remaining device parameters as explicit feasibility conditions, not estimated measurements.

# 14 \| Geometry, persistent state and reductions

**REQ-25.** Placement must include compute blocks, weight storage, cache banks, converters, pumps/sources, controls, accumulators and interconnect. The geometry is a physical graph embedded in declared regions, not only coordinates for arithmetic nodes.

## Geometry contracts

Model dimensions, non-overlap, ports and orientation; legal routes and channel capacity; distance-dependent delay/loss; bends/crossings where relevant; fan-out; shared-resource contention; and environmental coupling. Use technology-specific route models. A Manhattan length is not, by itself, a photonic propagation or acoustic storage model.

Allow chiplets or distinct physical regions with explicit links, interfaces and temperature domains. A room-temperature and cryogenic subsystem cannot share an assumed zero-cost local connection. Co-location is a hypothesis requiring compatible packaging and environmental contracts, not a default.

## Place state before moving it

A key objective is to execute useful reductions near stored data. For attention, a region may produce query-dependent partial states rather than export its complete keys and values. Composable attention states are established; FlashAttention and FlashInfer are important IO-aware precedents. \[11,12\]

For local scores s and values v, a region may compute a maximum m, a stabilized sum l and a weighted sum z. Combining regions can use:

$$M = \max_{j}m_{j},\quad L = \sum_{j}^{}e^{m_{j} - M}l_{j},\quad Z = \sum_{j}^{}e^{m_{j} - M}z_{j},\quad y = Z/L.$$

The planner chooses where local work, merging and conversions occur. Empty or fully masked partitions require explicit valid-state handling, not undefined divisions or exponentials of invalid values.

## Accounting and claim limits

Local cache reads, query distribution, partial-result traffic, merge latency and cache writes all remain charged. These summaries depend on the query; they are not a permanent constant-size replacement for an arbitrary cache. Digital/near-memory persistent cache and short-lived acoustic buffering are different storage classes.

**Acceptance:** compare boundary traffic, local bandwidth, total latency and energy against a strong locality-aware baseline. Hold memory capacity, available compute and communication resources fixed or expose their differences. A shorter route or smaller exported tensor is only an intermediate result; report the end-to-end effect.

# 15 \| Stateful LLM workloads and scheduling

**REQ-26.** Support an explicit decoder workload with weights, activations and evolving key–value state. Begin with symbolic shapes and a small executable model rather than materializing a full physical component for every scalar in a large checkpoint.

## Semantic workload coverage

The initial decoder path needs linear maps, residual addition, RMSNorm or the chosen normalization, the exact positional encoding in the reference model, attention including masks and scaling, cache append/read, SiLU and elementwise multiplication. Unsupported operators must be rejected or implemented through an explicit digital fallback.

Prefill and decode are separate execution phases. Declare batch size, prompt and generated lengths, heads and head dimensions, cache policy, numeric formats, model/checkpoint identity, weight residency and request-arrival assumptions. Do not derive architecture solely from saved weights.

## Time and resources

A schedule must reserve compute units, converters, memory ports, links, sources, buffers and calibration facilities. Tasks wait for inputs and resources. Include warm-up, weight programming, clock crossings, reconfiguration, integration, transfers, reset and synchronization. Physical state may expire while a task waits.

Persistent weights can be amortized over a declared reuse horizon; dynamic attention data cannot be amortized as though it were a fixed model parameter. Report cold-start and steady-state costs separately. Calibration frequency is part of the workload, not an invisible offline step.

Autoregressive dependencies limit single-request progress. Spatial pipelines may improve throughput across requests without proportionately improving one request’s latency. The scheduler must distinguish these cases.

## Reported metrics

Report time to first token, inter-token latency, request latency, tokens per second under a latency limit, energy per committed token, peak/average power, memory capacity, interconnect usage and physical resource counts. Quantiles require a declared workload distribution and sufficient traces; they cannot be inferred from one deterministic pass.

**REQ-27.** Maintain a cost ledger separating compute, movement, conversion, storage, programming, calibration, control and support costs. Sum energies at a consistent boundary; calculate time from the schedule, not from a sum of component throughputs. Report incomplete totals as incomplete.

**Acceptance:** no two tasks exceed a shared resource’s capacity; every cache read sees the correct version; no expired analog/acoustic/quantum state is consumed; a measured software baseline and a simulated device baseline are labeled separately.

# 16 \| Error, uncertainty and correlation-aware placement

**REQ-28.** Distinguish mathematical approximation, finite-precision arithmetic, physical bias, stochastic noise and uncertainty in the model itself. Do not combine them into one unexplained “accuracy penalty.”

## Classical errors

For a local linearization with sensitivity J, bias b and zero-mean covariance Sigma, a squared output-error proxy is:

$$\mathbb{E} \parallel \delta y \parallel_{2}^{2} \approx \parallel Jb \parallel_{2}^{2} + tr\left( J\Sigma J^{\mathsf{T}} \right).$$

This is an approximation unless stronger conditions establish exactness. Nonlinear saturation, rare tails and autoregressive error accumulation need separate checks. A low block-level error does not establish preserved perplexity or generation quality.

Shared disturbances must be represented by persistent source identities. A scalable classical model may use a small set of common modes plus local noise, rather than a dense covariance over the full model. Geometry and scheduling may change the couplings, temporal correlation and noise distribution.

## Representation-aware hypothesis

For a signed permutation S and an additive disturbance e in the transformed representation, the effective perturbation includes the inverse transformation. One candidate proxy is:

$$tr\left( JS^{- 1}\Sigma_{\pi,S}S^{- \mathsf{T}}J^{\mathsf{T}} \right).$$

The covariance must be re-evaluated when the physical encoding changes. It cannot be held fixed merely because doing so predicts a benefit. DOCTOR already exploits sensitivity and spatial variation; our extra hypothesis concerns exact representation freedom and correlation structure jointly with geometry. \[5\]

## Mandatory negative controls

For the specified isotropic additive-noise test with unchanged physical resources, signed permutations must not create a spurious benefit. For a signal-following multiplicative disturbance e = xi S z, applying the inverse representation leaves xi z; this mechanism alone does not improve. Correlation coefficients must define a positive-semidefinite covariance, and cancellation must not erase unrelated noise.

**REQ-29.** Use independent sampling to check classical propagation models, stress input ranges and evaluate held-out model quality. Quantum error uses channels and measurements, not this Gaussian covariance formula. Where a quantum output becomes a classical estimator, its classical statistics enter only after the declared measurement boundary.

**Acceptance:** uncertainty and quality reports identify their scope, confidence/tolerance conventions and untested conditions. “Zero observed failures” is not a proof of zero failure probability.

# 17 \| Search, multi-fidelity evaluation and crossover analysis

**REQ-30.** Search jointly over supported decompositions, representations, component instances, operating points, placement, routes, memory allocation, interfaces and schedules. Use a hierarchical search so large neural tensors are represented by tiles or groups rather than independently enumerated scalar devices.

## Evaluation pipeline

First check semantic legality, port compatibility and evidence policy. Then apply cheap resource and physical-envelope bounds. Next evaluate a complete schedule and error/cost model. Finally send selected candidates to finer local simulations and held-out neural evaluation. Record which stage produced each metric and rejection.

Exhaustive search is required only for small declared choice spaces used as oracles. Larger heuristic or surrogate searches must report seeds, budgets, coverage and termination status. They must not claim optimality or infeasibility beyond what their method establishes.

## Compare under explicit uncertainty

Use separate frontiers for evidence cohorts or prominently distinguish their members. Do not silently rank an optimistic hypothetical quantum device against measured electronic hardware as though their estimates had equal status.

For uncertain parameters, report nominal results, sensitivity and the conditions for a crossover. Distinguish **there exists** an allowed parameter setting where a plan wins from **for all** allowed settings it wins. Interval overlap means unresolved ordering unless a justified probabilistic model supplies a different criterion. Shared uncertainties must be propagated jointly.

## Break-even accounting

Let T_removed denote work avoided by a fused design, and T_added its new overhead after interactions with the critical path are modeled. A necessary local screening condition is that the avoided work exceed added work; the actual end-to-end comparison still comes from the full schedule. The analogous energy test includes all source/control and support terms.

Report the maximum tolerable interface overhead, required storage fidelity or required converter efficiency for each promising candidate. These are useful technology targets even when no present device is assumed to satisfy them.

**REQ-31.** Maintain a diversity archive of structurally different conditional candidates for research, separate from the performance Pareto set. The number of physical technologies used is not a performance reward and is not a novelty score.

**Acceptance:** tiny instances reproduce exhaustive frontiers; heuristics can return a one-technology winner; assumptions cannot become more optimistic automatically to satisfy a requested speedup; independent validation failures remain visible rather than discarded.

# 18 \| User-facing specifications and project layout

**REQ-32.** Preserve the friendly project-file workflow. The user should describe the neural program, workload, technology packs and exploration constraints in separate files. Resolve relative paths consistently, support physical units and produce a fully normalized snapshot.

The following is **proposed configuration syntax**, not a claim that these fields run in the current version:

>     schema_version: proposed/npp-multiphysics-1
>     model: model.yaml
>     workload: workloads/decode.yaml
>     technology_packs:
>       - packs/digital_reference.yaml
>       - packs/analog_electrical.yaml
>       - packs/classical_photonic.yaml
>       - packs/acoustic_exploratory.yaml
>       - packs/quantum_exploratory.yaml
>     interfaces: interfaces/registry.yaml
>     environment: system/environment.yaml
>     exploration:
>       semantic_track: equivalent
>       granularity: intra_operator
>       allow_interface_fusion: true
>       allow_representation_rewrites: true
>       allow_single_technology_solution: true
>       uncertainty_policy: report_conditions
>       unmodeled_connection_policy: reject
>     constraints: constraints.yaml
>     experiments: experiments.yaml

## Separate mechanism code from instance data

Technology packs reference reviewed evaluator identifiers; they do not execute arbitrary Python, formulas or natural-language rules. A new mechanism requires code, guards and tests. A new characterized instance of a supported mechanism may be supplied as data.

Keep the modules separate: core contracts and intermediate representations; semantic rewrites; device backends; interfaces/couplings; environment and geometry; scheduling; search; evidence; validation adapters; and reporting. Names may follow the existing repository conventions rather than impose a disruptive directory rewrite.

**REQ-33.** Legacy inputs must either retain their previous meaning or undergo an explicit, versioned migration. A legacy optical noise surrogate must never silently become a calibrated device pack. Saved plans pin model, component, interface and evaluator versions.

**Acceptance:** a user can switch an acoustic pack off without rewriting the neural model; normalization exposes every inserted interface and assumption; unsupported quantum metrics remain absent/conditional instead of receiving invented defaults.

# 19 \| Workflow, outputs and explanations

**REQ-34.** The normal workflow is: validate and normalize; enumerate legal rewrites and compositions; evaluate resources and physics; explore geometry and scheduling; refine selected candidates; compare against independent references; and export a reproducible result bundle.

## Required outputs

| **Artifact**                | **Required contents**                                                                  |
|-----------------------------|----------------------------------------------------------------------------------------|
| **Normalized inputs**       | Model/workload hashes; pack versions; environment; evidence and search policies        |
| **Semantic plan**           | Operator partition, rewrite trace, model identity and exact/approximate/adapted status |
| **Physical plan**           | Typed ports, devices, interfaces, routes, shared controls and state ownership          |
| **Schedule**                | Events, dependencies, reservations, state lifetimes and calibration/programming        |
| **Cost and quality ledger** | Per-category resources, complete/incomplete totals, error and task metrics             |
| **Assumption ledger**       | Source or hypothesis for each critical parameter, validity envelope and uncertainty    |
| **Comparison report**       | Matched baseline, deltas, crossover conditions, ablations and rejected candidates      |
| **Verification bundle**     | Reference outputs, independent comparisons, test status, seeds and failures            |

A human-readable report must make the semantic graph, device graph, domain crossings, critical path and noise dependencies distinguishable. A diagram does not replace the machine-readable contracts. Result tables must say whether numbers are symbolic, analytically estimated, simulated or measured.

## Designer interaction

The user must be able to pin a device, forbid a technology, constrain a region, require editable weights, select a quality/evidence threshold, change a physical assumption, and compare the resulting plans. Every edit invalidates dependent evaluations and triggers rechecking.

Explanations should answer: why was an interface inserted; why did one technology disappear; which assumption makes this candidate win; what prevents a quantum output from feeding the next layer; and what measurement would most reduce the uncertainty?

**REQ-35.** Export structured rejection reasons and counterfactuals, not only the winning plan. Never display “quantum advantage,” “LLM speedup” or “physically feasible” as an unconditional badge when the result is conditional or only local to one block.

**Acceptance:** another process can replay the normalized plan and reproduce its deterministic outputs. A separate reference evaluator can check selected behavior without importing the production evaluator’s formulas. Replaying the same formulas is a consistency check, not independent validation.

# 20 \| Initial experiments: composition and geometry

The first campaign should establish or reject mechanisms, not optimize for a desired headline. Use synthetic fixtures for software tests and source-parameterized or explicitly conditional models for research comparisons.

## H1 \| Heterogeneous partitions inside a linear/FFN block

**Question:** can partitions matched to physical capabilities beat both single-technology execution and conventional whole-operator mapping?

Compare fixed digital, each eligible single compute backend with required support circuitry, operator-level hybrid, intra-layer hybrid with full interfaces, and an exact small-instance search oracle. Sweep dimensions, batch/reuse, dynamic range, converter precision and hardware-resource allocation. Record complete block latency, energy, resources and output/task quality. Ablate partitioning, representation transforms and merge placement separately.

**Failure condition:** the gain disappears when distribution, conversions, utilization and merging are charged, or requires more resources than the comparison permits. A loss is recorded rather than repaired with selectively favorable coefficients.

## H2 \| Interface-fused gating with optional acoustic alignment

**Question:** can a legal physical gating interface replace intermediate readout, multiplication and re-encoding?

Compare unfused hybrid, fused gating without extra storage, fused gating with optical delay, electronic buffering/re-encoding, recomputation and a specified acoustic buffer. Sweep branch skew, gate range, interface nonlinearity/error, insertion loss, control cost and storage retention. Hold the semantic track and quality target fixed.

**Success criterion:** a reproducible parameter region where the complete fused plan wins, with explicit minimum requirements on the interface and buffer. **Failure condition:** a required sign/transfer law is absent, or timing and retrieval negate the saving. An acoustic buffer winning only under hypothetical retention is a conditional design target, not a demonstrated acoustic advantage.

## H3 \| Representation-aware correlated-error placement

**Question:** do exact channel transformations and placement jointly improve a useful physical operating point beyond sensitivity-aware mapping?

Compare fixed representation, sensitivity-aware assignment, transformations alone, geometry alone and joint optimization. Include isotropic additive noise, physically shared additive disturbances, signal-following multiplicative disturbances and a device-specific model. Evaluate held-out inputs and a range of physical operating points.

**Success criterion:** lower harmful error enables a real improvement in modeled integration time, concurrency, calibration or energy at the same task quality. **Failure condition:** only the error proxy improves, or the benefit vanishes when covariance is re-evaluated after the transformation. Pure robustness gains are reported as robustness, not inference acceleration.

# 21 \| Initial experiments: state, roles and quantum

## H4 \| State-local attention and reduction placement

**Question:** can geometry and conversion boundaries reduce communication more effectively than an existing locality-aware architecture?

Compare centralized execution, a strong fixed near-memory/local-reduction baseline, placement optimization alone, and joint cache/merge/interface placement. Sweep context length, cache partitioning, batch size, head dimensions, local bandwidth and inter-region links. Charge cache updates and local scans as well as exported traffic. Validate attention outputs, including masked and empty partitions, against an independent implementation. \[11,12\]

**Failure condition:** traffic reduction does not improve total latency/energy, or wins only by adding uncharged memory or compute. Acoustic delay is not substituted for persistent cache storage without a separately valid storage model.

## H5 \| Approximate physical draft, digital verifier

**Question:** does a low-fidelity physical proposer improve committed-token throughput when verification is fully charged?

Speculative decoding already supplies a proposer/verifier algorithm; our hypothesis concerns physical operating points and system scheduling. Begin with greedy decoding and an authoritative digital target. Compare a digital draft with an analog/photonic draft at matched memory, power and quality conventions. Sweep draft fidelity, proposal length, acceptance, interfaces and state reuse. Measure accepted/committed tokens per full cycle, not draft tokens per second. \[13\]

**Failure condition:** acceptance or communication overhead erases the gain. Distribution-preserving stochastic generation requires its own valid verification algorithm and actual proposal probabilities; it is not inferred from the greedy result or from noisy hardware sampling.

## H6 \| Quantum–classical crossover with output-aware costs

**Question:** under what explicit data-access, precision and output requirements could a quantum subroutine be useful inside a classical neural workload?

Use a specified small subproblem with a classical reference: for example, estimating selected linear-algebra quantities versus returning a complete classical vector. Verify its mathematical relationship to the surrounding graph. Compare classical dense and relevant randomized alternatives under comparable access assumptions. Sweep preparation, queries, depth, success probability, estimator precision, shots, readout and hardware mapping. Separate logical-resource results from physical estimates. \[9,10\]

**Failure condition:** the claimed advantage requires free preparation, free full-vector output, an unsupported oracle, or simulator-only state access. A valid result may be a conditional crossover or a principled rejection of this placement, not a quantum win.

**Optional extension, Track R only:** explore acoustic recurrent/filtering blocks through adaptation or distillation. Compare against an adapted digital model under the same task and training budget; never present it as unchanged-Transformer compilation. Physical-network training has established precedents. \[6,7\]

# 22 \| Verification and falsification suite

**REQ-36.** Build the tests before broadening the search space. Use deterministic or statistically justified pass criteria, published tolerances and small independent oracles. Random tests alone do not establish universal physical correctness.

| **Test family**               | **Required failure or invariant**                                                                        |
|-------------------------------|----------------------------------------------------------------------------------------------------------|
| **Semantic rewrites**         | Correct partitions, bias placement, permutations and sign rules; illegal nonlinear commutations rejected |
| **Units and encodings**       | No implicit amplitude/intensity, current/voltage, classical/quantum or clock-domain cast                 |
| **Passive/active accounting** | No passive gain or free compensation; all sources and controls have owners                               |
| **Backend isolation**         | Removing one optional backend does not change unrelated semantics or require its dependencies            |
| **Shared disturbances**       | Reconvergent paths retain common errors; covariance stays physically admissible                          |
| **Representation controls**   | No false gain under the specified isotropic and signal-following noise controls                          |
| **Scheduling**                | Precedence, capacity, read/write hazards, synchronization and programming costs enforced                 |
| **Storage**                   | Retention, destructive reads, reset and versioning enforced for every state type                         |
| **Interfaces**                | Directionality, operating envelope, pump/reference and sampling requirements enforced                    |
| **Quantum semantics**         | State validity, instrument probabilities, no-cloning and physical readout enforced                       |
| **Evidence**                  | Missing costs never become zero; out-of-envelope use changes validation status                           |
| **Search**                    | Small exhaustive oracle matches; heuristic coverage and incompleteness are explicit                      |
| **Reproducibility**           | Stable hashes, pinned evaluators, separated random streams and cache invalidation                        |
| **Accounting**                | Component and composite ledgers contain no omitted or double-counted contribution                        |

## Independent validation levels

First check exact algebra with a separate numerical oracle. Then compare reduced-order device models with an independent implementation or external simulator on matched small cases. Validate coupled blocks and system schedules separately. Where possible, characterize parameters against measurements, without treating that as proof of the entire system.

The reference must not simply call the production evaluator. Report tolerances, convergence checks, discrepancies and model limitations. Retain known counterexamples as regression fixtures.

## Quality and research controls

Separate calibration/search inputs from held-out task evaluation. Register baselines and resource normalization before selecting winners. Keep negative results, failed candidates and sensitivity sweeps. Distinguish actual executed measurements from proposed experiments and synthetic fixtures in every output file.

**Acceptance:** no claimed capability is complete without its negative tests, and no external validation is claimed without an actual independently obtained comparison result.

# 23 \| Implementation sequence and release gates

This sequence extends the project’s four-part milestone; it does not assert that those parts are absent or already complete. Inspect and reuse the active implementation first. Every stage ends with implementation, tests, review, correction and a recorded gate decision.

| **Stage**                            | **Deliverable**                                                                          | **Gate before expansion**                                                               |
|--------------------------------------|------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| **M0: baseline reconciliation**      | Capability inventory, pinned legacy outputs, migration plan and experiment definitions   | Current branch scope is documented; existing tests and example meaning are preserved.   |
| **M1: reviewed technology packs**    | Component/evidence contracts, digital reference and one electronic/photonic model family | Unknown parameters, envelopes and accounting ownership are tested.                      |
| **M2: typed implementation graph**   | Ports, interfaces, state, resources, semantic rewrites and basic scheduler               | Illegal cross-domain connections fail; mixed linear/FFN fixtures execute coherently.    |
| **M3: independent validation**       | Reference oracles and at least one genuine independent device/coupling comparison        | Discrepancies and validity limits are documented; replay is not called independence.    |
| **M4: designer exploration**         | Edits, explanations, Pareto/crossover reports and bounded search                         | H1/H2 can run reproducibly; constraints and assumptions are visible.                    |
| **M5: broader theoretical families** | Acoustic delay/coupling and quantum logical/reference models                             | Each has separate physics, interface costs, negative tests and conditional outputs.     |
| **M6: stateful research campaign**   | Decoder scheduling, local attention and H3–H6 experiments                                | End-to-end resource/quality accounting works; local gains are not extrapolated blindly. |

## Minimum coherent research release

A release is useful before all device families are highly characterized. It must, however, contain the common contracts, a complete digital path, at least one genuinely mixed implementation, explicit storage/timing, evidence-aware accounting, replay, independent semantic checks and a reproducible crossover study. Acoustic and quantum packs may initially provide validated small semantics plus conditional resource estimates rather than full physical estimates.

## Completion of this specification’s initial scope

The initial multi-physics scope is complete when all five families can be expressed without changing the core, legal mixed subgraphs can be evaluated at their declared model level, illegal or unsupported mixtures are diagnosed, H1–H6 are representable and executable where their chosen models are supplied, and every output exposes missing evidence. Performance improvement is an experimental result, not a software acceptance requirement.

**REQ-37.** Each gate records requirement IDs, reviewer identity or role, issues, changes, executed tests and remaining limitations. Automated self-checks are useful but must not be mislabeled as an independent reviewer or external physics validation.

# 24 \| Worked contract: an acoustic alignment block

The following is a **schema-design example**, not current executable input and not a characterized device. It illustrates the requested compartmentalization. The block is a composite: optical writing, acoustic evolution and optical retrieval belong to one coupling implementation. Numerical coefficients live in an explicitly selected scenario, not in hidden defaults.

>     component:
>       id: optoacoustic_alignment_candidate
>       schema_version: proposed/npp-multiphysics-1
>       backend: interfaces.optoacoustic_buffer
>       kind: joint_coupling_block
>       evidence: hypothetical
>       semantic_contract:
>         operation: delayed_tensor_transfer
>         ideal_relation: output_equals_delayed_input
>         physical_fidelity: scenario_defined
>       ports:
>         data_in: {type_ref: classical.optical.field_vector}
>         data_out: {type_ref: classical.optical.field_vector}
>         write_control: {type_ref: classical.optical.pump}
>         read_control: {type_ref: classical.optical.pump}
>       owned_state:
>         acoustic_modes: {model_ref: scenarios/acoustic_modes.yaml}
>       parameters_ref: scenarios/buffer_assumptions.yaml
>       guards:
>         - spectral_and_mode_matching
>         - compatible_phase_and_amplitude_encoding
>         - arrival_within_storage_fidelity_envelope
>         - no_conflicting_write_or_read
>       accounting:
>         ownership: composite_inclusive
>         includes: [write, storage, read, local_control, local_loss]
>         external_services: [pump_supply, routing, thermal_support]
>       validation:
>         ideal_oracle: delayed_identity
>         reference_model: independent_coupled_mode_reference
>         required_tests: [retention, loss, timing, energy, reread]

**How to read it.** The ideal transfer states the intended semantic operation; it does not declare zero physical error. The coupling backend owns the joint dynamics. Referenced port types provide units, shape and encoding. Shared pump supply and external routing are counted once outside the composite; local interactions are counted inside. A source-parameterized replacement needs provenance and a valid envelope for every required coefficient.

**Quantum contrast.** A quantum buffer would require quantum ports, a quantum state/channel model and a compatible preparation/readout contract. Renaming the classical acoustic state “quantum” would be rejected. The common framework enables both without equating their internal physics.

# 25 \| Sources and prior-art boundaries I

These sources motivate requirements and identify nearby work. They do not establish originality of the proposed combination. The specification is an original design proposal; no reported device performance has been transplanted into its hypothetical components. Sources were checked on 3 October 2026. Links point to primary publications, author manuscripts, project documentation or developer documentation.

**\[1\] Neural Physical Planner, repository README, v0.2 description.** Inspected project documentation. README blob SHA: `3715a76a8e87db698af6aa07b72e7d75b665cf25`. Relevant for the current documented scope, synthetic coefficients, supported importers, noise model and limitations. This is not a code/test audit. [Repository source](https://github.com/ReedA2022/neural-physical-planner/blob/main/README.md)

**\[2\] Risso et al. Precision-aware Latency and Energy Balancing on Multi-Accelerator Platforms for DNN Inference.** ODiMO, ISLPED 2023; author manuscript arXiv:2306.05060. Direct precedent for splitting individual layers across heterogeneous compute units with precision-aware trade-offs. Read the mapping model and communication assumptions before defining a baseline. [Author manuscript](https://arxiv.org/abs/2306.05060)

**\[3\] HyAtten: Hybrid Photonic-Digital Architecture for Accelerating Attention Mechanism.** DATE 2025, IEEE document 10993031. Direct architectural precedent for hybrid photonic/digital attention and converter-related precision handling. It motivates a stronger baseline than assigning a single technology to each layer. [Publication record](https://ieeexplore.ieee.org/document/10993031/)

**\[4\] Chen et al. All-analog photoelectronic chip for high-speed vision tasks.** Nature 623, 48–57 (2023). DOI: 10.1038/s41586-023-06558-8. ACCEL is a physical vision demonstration combining optical and analog electronic computation, including a computational photoelectric interface. Do not transfer its vision results or encoding directly to LLM inference. [Published paper](https://www.nature.com/articles/s41586-023-06558-8)

**\[5\] DOCTOR: Dynamic On-Chip Temporal Variation Remediation Toward Self-Corrected Photonic Tensor Accelerators.** Author manuscript arXiv:2403.02688 (2024), version 2 consulted. Relevant to thermal/temporal variation, sensitivity-aware remapping and calibration overhead. It is close prior art for noise-aware mapping, not evidence for our proposed exact-representation advantage. [Author manuscript](https://arxiv.org/abs/2403.02688)

**\[6\] Wright et al. Deep physical neural networks trained with backpropagation.** Nature 601, 549–555 (2022). DOI: 10.1038/s41586-021-04223-6. Demonstrates physics-aware training with different physical systems. Relevant to the separate model-adaptation track; it does not establish compilation equivalence for an unchanged checkpoint. [Published paper](https://www.nature.com/articles/s41586-021-04223-6)

**\[7\] An optoacoustic field-programmable perceptron for recurrent neural networks.** Nature Communications (2024). DOI: 10.1038/s41467-024-47053-6. OREO demonstrates an optoacoustic recurrent operator. Relevant to coupled dynamics and short-lived physical memory; not evidence for arbitrary persistent storage or faster LLM decoding. [Published paper](https://www.nature.com/articles/s41467-024-47053-6)

# 26 \| Sources and prior-art boundaries II

**\[8\] Merklein et al. A chip-integrated coherent photonic-phononic memory.** Nature Communications (2017). DOI: 10.1038/s41467-017-00717-y. Relevant experimental precedent for coherent conversion of optical information into acoustic storage and retrieval. Its specific retention, bandwidth and control requirements must be preserved when parameterizing a model. [Published paper](https://www.nature.com/articles/s41467-017-00717-y) · [Author manuscript](https://arxiv.org/abs/1608.08767)

**\[9\] Guo et al. Quantum Transformer: Accelerating model inference via quantum linear algebra.** arXiv:2402.16714; version 3 consulted. Theoretical fault-tolerant quantum treatment, with explicit input/output representations and data-access assumptions. Read the problem formulation, state preparation, output extraction and runtime discussion. The consulted manuscript is not a demonstrated physical LLM accelerator. [Author manuscript](https://arxiv.org/abs/2402.16714)

**\[10\] IBM Quantum Learning. Channel representations.** Primary technical documentation, accessed 3 October 2026. Relevant to valid channel representations, complete positivity and trace preservation. It supports the state/channel contract, not any hardware advantage estimate. [Technical reference](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/quantum-channels/representations-of-channels)

**\[11\] Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.** NeurIPS 2022; author manuscript arXiv:2205.14135. Relevant to exact attention, tiling and IO-aware baselines. Reducing communication against a naive centralized design is not enough when stronger locality-aware methods exist. [Paper](https://arxiv.org/abs/2205.14135)

**\[12\] FlashInfer. Cascade Inference: Memory Bandwidth Efficient Shared Prefix Batch Decoding.** Developer technical article, 2 February 2024. Primary explanation of composable attention states and merge operations. It motivates the reduction interface; the algebra itself is not proposed as new. This source is an engineering article, not a peer-reviewed hardware demonstration. [Developer article](https://flashinfer.ai/2024/02/02/cascade-inference.html)

**\[13\] Leviathan, Kalman and Matias. Fast Inference from Transformers via Speculative Decoding.** ICML 2023, PMLR 202, 19274–19286. Established proposer/verifier algorithmic precedent. Our proposed experiment changes physical implementation and operating point, not the foundational speculative-decoding algorithm. [Published paper](https://proceedings.mlr.press/v202/leviathan23a.html)

**\[14\] Benmeziane et al. Supernetwork-based efficient mapping of deep learning applications to mixed-precision hardware using model adaptation.** Nature Communications 17, 4501 (27 March 2026). DOI: 10.1038/s41467-026-71071-1. Relevant recent precedent for joint model adaptation and heterogeneous analog–digital mapping. Treat adaptation as a separate comparison track. [Published full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13187305/)

## Final design principle

**Permit combinations through explicit contracts; never invent compatibility to complete a graph.** The planner succeeds when it explains what can be composed, what cannot, and which physical improvements would make a currently conditional design worthwhile.
