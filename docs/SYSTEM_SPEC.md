# Neural Physical Planner v0.1 — system specification

## Purpose and implementation boundary

NPP accepts a neural computation, a parameterized physical-component library and a design request. It generates alternative abstract hardware implementation plans, evaluates their costs and numerical behavior, and retains feasible nondominated choices.

A plan is a **macro graph**, with selected components, compensation recipes, domain conversions, connections and an execution schedule. It is not a transistor netlist, placed-and-routed layout, optical mask, fabrication package or bill of commercially purchasable parts. The current backend does not emit synthesizable RTL. Component internals and concrete optical encodings remain engineering inputs to a future lower-level backend.

The bundled library is explicitly synthetic. Its parameters exercise the algorithms; they do not support claims about the performance of real hardware. The implemented model is `normalized_additive_gaussian_v1`, not a coherent optical simulator. Signed real arithmetic is abstracted; signal encodings for negative weights, digital rounding, wavelength interference and device nonlinearities are outside this model.

## Input 1: neural computation

The JSON object has these fields:

| Field | Meaning |
|---|---|
| `schema_version` | `"0.1"`; supplied automatically when omitted. |
| `name` | Nonempty descriptive name. |
| `nodes` | Nonempty, explicitly topologically ordered list of nodes. |
| `outputs` | Nonempty ordered list of node IDs. Outputs are delivered digitally. |

Each node has `id`, `op`, `inputs` and positive integer `size`. `size` is the output vector dimension. IDs are unique and follow `^[A-Za-z][A-Za-z0-9_.:-]*$`. Exactly one input node is supported. Every node must reach an output. Repeated input references and repeated output references are legal and count as separate physical uses.

| Operation | Inputs and additional fields | Ideal numerical semantics |
|---|---|---|
| `input` | No predecessors; `input_bounds.lower` and `.upper` vectors of length `size`. | Bounded real input vector. Each lower bound must not exceed the corresponding upper bound. |
| `linear` | One predecessor; rectangular `weights` with `size` rows and input-dimension columns; optional length-`size` `bias`. | `weights @ input + bias`; omitted bias becomes a zero vector. |
| `relu` | One predecessor of the same size. | Coordinatewise maximum with zero. Only digital components support this operation in v0.1. |
| `identity` | One predecessor of the same size. | Unchanged vector. |
| `add` | At least two predecessors, all of the same size. | Coordinatewise sum, including repeated operands. |

Weights and bias are accepted only on `linear` nodes; bounds only on the input. Dimensions, references, ordering and reachability are validated before search. There is no arbitrary Python expression, executable model importer or implicit broadcasting. Weights alone are not a complete input: the graph and input domain are also required.

The reference values use NumPy floating-point arithmetic. They are not bit-exact fixed-point hardware semantics. An additional quantization contract and backend would be needed for that claim.

## Input 2: component and rule knowledge base

The library object contains `schema_version`, `name`, the fixed model identifier, nonempty `provenance`, `components`, `rules` and `conversions`. Provenance should identify how parameters were obtained, their intended operating conditions and their uncertainty. It is currently descriptive metadata, not a verified calibration record.

### Components

Each component provides these fields:

| Field | Type and units | Meaning |
|---|---|---|
| `id`, `ops` | ID and nonempty list of supported operations | A reusable macro type. |
| `domain` | `digital` or `optical` | Numerical transport domain. |
| `editable` | Boolean | Whether the component permits weight changes when it implements a linear node. |
| `source_scalable` | Boolean | Whether the input optical encoder can change launch power or be reused for sequential encoding. Only an optical component with `ops=["input"]` may set it true. |
| `area_um2` | Nonnegative square micrometres | Area of one instantiated macro. |
| `latency_ns` | Nonnegative nanoseconds | Latency of one macro operation. |
| `energy_pj` | Nonnegative picojoules | Base dynamic energy of one macro operation. |
| `energy_per_mac_pj` | Nonnegative picojoules/MAC | Additional energy for `linear` only. |
| `noise_std` | Nonnegative normalized numerical units | Independent isotropic additive noise at scalar outputs; a macro's output noise is shared by its outgoing uses unless source re-encoding generates separate instances. |
| `max_abs_value` | Positive normalized numerical units | Nominal output range and the declared headroom used by gain guards. |
| `photons` | Positive count | Baseline photon budget per scalar optical value. |
| `optical_efficiency` | `(0,1]` | Efficiency used in the branch noise model. |
| `wall_plug_efficiency` | `(0,1]` | Efficiency used in the launch energy model. |
| `wavelength_nm` | Positive nanometres | Photon-energy conversion wavelength. |
| `capacity` | Positive integer | Maximum node **output** dimension supported by this macro; not a general transistor, memory or MAC-capacity model. |
| `notes` | String | Engineering explanation; never executable. |

Cost and optical fields are scalar parameters, not expressions. Costs are per complete macro, not automatically proportional to vector length. Optical launch energy is additionally proportional to output dimension. The solver does not infer a realistic circuit cost from weights; an engineer must supply or derive macro alternatives with appropriate parameters.

Default values are materialized by `validate_inputs`. Digital components still receive normalized optical fields, but those fields do not generate optical launch energy for digital nodes. Input encoders are separate component types; a type cannot mix `input` and other operations.

### Trusted compensation kinds

Every rule has `id`, `kind`, `domains`, `split_policy`, `gain`, `added_noise_std`, `area_um2`, `latency_ns`, `energy_pj` and `notes`. Costs cover one complete fan-out recipe. v0.1 has no automatic splitter-tree synthesis, port limit or fan-out-dependent area scaling, so supplied costs must be appropriate for the intended number of branches.

A rule is selected exactly where a node has more than one outgoing use. Output delivery counts as a use. Rules do not apply to arbitrary locations in this version.

| `kind` | Preconditions | Implemented behavior |
|---|---|---|
| `digital_copy` | Digital node; `gain=1`. | Nonzero-cost distribution if declared; inherited errors remain correlated. |
| `passive_split` | Optical node; `gain=1`. | Divides available photons over branches. Fractions are equal or use the sensitivity heuristic. |
| `source_boost` | Source-scalable optical input encoder. | Increases launch photon budget by `gain`; preserves the ideal decoded value and pays the modeled launch energy. |
| `amplify` | Optical node. | Increases effective photon budget; adds declared shared pre-split noise and energy overhead. It does not erase incoming errors. |
| `regenerate` | Optical node. | Adds shared measurement noise, including a detector shot surrogate, then independent full-power branch re-encoding noise. Incoming errors are preserved. |
| `serial_reencode` | Source-scalable optical input encoder; request permits serialization; `gain=1`. | Re-encodes the exact source input independently for successive branches. Encoding energy and delay repeat. It cannot regenerate an unknown intermediate numerical value for free. |

`digital_copy` accepts only `domains=["digital"]`; all other kinds accept only `["optical"]`. `gain` is at least one. For the three kinds that increase photon budget, the nominal field-amplitude surrogate must satisfy

`sqrt(gain) * maximum nominal absolute value <= max_abs_value`.

This headroom test is a declared abstraction, not a universal optical-device law. Rule noise may be zero in the mathematical input format; that does not establish that a noiseless physical amplifier exists.

`split_policy="sensitivity"` allocates positive fractions according to downstream Lipschitz sensitivities. It is a deterministic heuristic; it is not a continuous optimizer or a proof that the allocation is globally optimal. Regeneration and serial re-encoding use full-power branches rather than passive fractional splitting.

### Conversions

Exactly one conversion is provided in each direction, digital-to-optical and optical-to-digital. Its fields are `id`, `from_domain`, `to_domain`, `area_um2`, `latency_ns`, `energy_pj`, `noise_std` and `notes`. A conversion is instantiated on each edge that changes domains, including a final optical output delivered to the digital sink. It adds its costs and independent noise; it does not reset inherited error.

There is currently no search among alternative conversion types for the same direction. Supporting that requires extending the decision representation and enumerator.

### What “extending the knowledge base” means

Adding component entries or parameterized alternatives for an existing rule kind requires only JSON changes. The same search engine can consider them immediately.

A new physical mechanism requires more than a note in JSON. It needs executable applicability guards, numerical semantics, noise-source sharing semantics, cost/schedule semantics, forward-simulation behavior and tests. Those changes belong in the trusted backend. Unknown rule kinds are rejected. No LLM-produced text is treated as a verified physical law.

## Input 3: design request

| Field | Meaning |
|---|---|
| `schema_version`, `name` | Version and descriptive request name. |
| `objectives` | Nonempty subset of `energy_pj`, `latency_ns`, `area_um2`, `error_rms_bound`; all minimized. |
| `budgets` | Optional nonnegative upper limits on any of the same four metrics. A zero budget is meaningful. |
| `required_editable` | IDs of linear nodes whose weights must remain editable; selected macros must declare that capability. |
| `allowed_domains` | Nonempty subset of `digital`, `optical`. |
| `allow_serialization` | Permits the source-only sequential re-encoding rule. |
| `search.mode` | `exhaustive` or `beam`. |
| `search.max_evaluations` | Positive limit on full candidate evaluations. |
| `search.beam_width` | Positive retained-prefix width for beam search. |
| `seed` | Nonnegative integer used for reproducible sampling; search itself is deterministic. |

Editability concerns weights and bias of a fixed-shape linear macro. It does not guarantee arbitrary graph changes, a particular programming protocol, update latency, update precision, nonvolatile storage, lifetime rebuilding costs or random-access memory. Those properties need additional contracts.

There is one dedicated macro per neural node. Dependencies follow an ASAP schedule. Independent compatible branches can overlap. The only synthesized resource reuse is source re-encoding when enabled. Arbitrary shared-resource scheduling, batching, throughput optimization, interconnect contention and random-access guarantees are not supported.

## Numerical semantics and guarantees

Each primitive source is an independent zero-mean isotropic Gaussian vector. A source that flows down two paths remains the same source; it is not independently resampled merely because paths separate. This distinction matters at reconvergence.

For an optical branch with photon budget `N`, split fraction `s`, efficiency `eta` and effective gain `g`, the normalized additive shot surrogate has variance

`1 / (N * s * eta)` divided by `g`.

It is independent of the numerical signal value. This is a specified Gaussian approximation, not Poisson photo-detection or coherent-wave simulation. Amplification changes the photon/error model, not the ideal decoded NN value.

The error objective is

`sqrt(E[||stacked_digital_outputs - ideal_outputs||_2^2])`.

It is an aggregate output-vector RMS error, **not** RMS per scalar output, classification accuracy, worst-case deterministic error or a confidence interval.

For an affine DAG, independent-source loading matrices propagate through the graph. Summing their outer products gives the exact output covariance **under the declared additive model**, including cancellation and reinforcement of shared noise. The error is the square root of its trace. Such plans report `analysis.kind="exact_affine_gaussian"`.

With ReLU, the evaluator uses Lipschitz and Minkowski propagation to bound the aggregate RMS error conservatively. Such plans report `conservative_lipschitz_rms`. They do not report the affine covariance as an exact nonlinear covariance.

Nominal operating ranges are conservative interval enclosures of the noiseless NN over the input box. These intervals can be loose because they omit some input correlations. A rejected range check does not prove a real device must fail. Conversely, passing the nominal check does not bound unbounded Gaussian tails; the simulator does not clip them. The checker therefore does not certify absence of stochastic saturation.

Optical launch energy per macro operation uses

`photons * output_dimension * h * c / wavelength / wall_plug_efficiency`.

Gain, regeneration and serial rules modify launch counts as reported in each plan. Amplifier/regenerator replenishment uses the originating macro's efficiency as a stated surrogate, plus the rule's overhead. Digital energy includes base energy plus per-MAC cost for linear operations. Every inference is one graph execution.

## Output and provenance contract

A search report contains:

- `schema_version`, `status` and hashes of the normalized three inputs.
- `search` telemetry: mode, algorithm, structural candidate cardinality, evaluated/feasible/rejected counts, limits, completion flag and beam pruning counts.
- `plans`: retained feasible nondominated records.
- `rejection_summary`, diagnostics and modeling caveats.

Each plan contains its decision-derived ID, the three input hashes, a `decision` and an `evaluation`. Decisions map every node ID to a component ID and every fan-out node ID to a rule ID. Outgoing uses are ordered first by destination-node order and input position, then by output position. This ordering defines serial branch order.

Each evaluation contains:

| Field | Contents |
|---|---|
| `feasible` | Whether all implemented physical guards and requested budgets passed. |
| `metrics` | Energy, latency, area and aggregate RMS error bound. |
| `diagnostics` | Structured code/path/message records. |
| `analysis` | Analysis kind, nominal bounds, model assumptions; exact covariance for affine plans. |
| `trace` | Chosen compensation rules, gains, fractions and guard information. |
| `hardware` | Macro instances, connections and schedule; `device_netlist=false`. |
| `noise_sources` | Source identities, dimensions, variance and role for auditing correlation. |

Candidate feasibility and input validity are different. A well-formed zero-energy request can yield no feasible plan. A malformed DAG fails input validation before search.

Input hashes and plan IDs use canonical JSON and SHA-256. They support consistency checks, not cryptographic authenticity, proof of ownership or externally attested results. A plan ID hashes the decision only; the associated input hashes must also match.

## Search claims

Exhaustive search lazily enumerates the finite Cartesian product of structurally compatible component/rule assignments. Full evaluations apply physical and budget guards. If it finishes, the returned nondominated set is complete within that finite candidate family and the numerical comparison tolerance. It is not complete over all physical circuits or continuously variable component parameters.

If enumeration hits `max_evaluations`, `search.complete` is false. Beam search ranks partial assignments with cost/noise surrogates and retains a diverse mixture of objective rankings. These are not admissible bounds. It may discard every feasible continuation and miss better designs; its set is approximate even when all retained complete candidates were evaluated.

| Status | Interpretation |
|---|---|
| `ok` | At least one feasible plan was found. Inspect `search.complete` separately. |
| `infeasible` | No feasible plan exists in the exhausted finite candidate family, or the structural candidate family is empty. This is model-relative. |
| `search_exhausted` | No feasible plan was found by the incomplete search. Nonexistence is not proved. |

Pareto comparisons minimize the requested objectives, with relative tolerance `1e-9` and no absolute tolerance. Equivalent objective vectors retain a deterministic representative. Budgets are constraints, not objectives that can be traded away.

## Replay checking and simulation

`check_record` validates the inputs, verifies hashes and plan ID, recomputes the evaluation from the decision and compares stored fields. It does not trust cached metrics or rerun the search. It uses the same physical model as the planner; it is not an independent physical proof system. A valid replay does not establish Pareto optimality.

`simulate` independently executes ideal and noisy operations using primitive noise samples. It does not draw from the analytically computed final covariance. Inputs are sampled uniformly from the declared box. The output contains empirical MSE/RMS error, estimated standard errors, an analytic bound and output summaries. Affine consistency requires two-sided agreement with the predicted MSE within four estimated standard errors plus a relative numerical tolerance. Nonlinear consistency checks only the upper bound. These are diagnostic heuristics, not formal confidence guarantees; extreme scales can cause floating-point cancellation, which the affine check can expose.

The software can be used as a reproducible planning and model-consistency baseline. Claims about characterized devices, fabrication or research novelty require additional evidence.
