# Physical evaluation model

This is the retained macro model. For the v0.3 intensity fanout component models,
complete-symbol timing and optical comparisons, see [TECHNOLOGY.md](TECHNOLOGY.md),
[IMPLEMENTATION.md](IMPLEMENTATION.md) and [SIMULATION.md](SIMULATION.md).
The two evaluation scopes have separate metrics and must not be summed.

This document describes the implemented `normalized_additive_gaussian_v1`
backend in `npp/physics.py`. It is an auditable computational model for comparing
abstract hardware implementation plans. It is not a coherent-wave simulator, a
transistor-level model, or a fabrication-ready circuit generator.

All guarantees below are conditional on the declared model. The example library
contains illustrative synthetic parameters, not measured devices.

## 1. Numerical meaning and scope

A neural node operates on a vector of signed real numbers. Supported operations
are input, affine linear transformation, identity, addition and ReLU. A selected
component implements the ideal operation and then introduces its declared
output noise. The software does not synthesize an optical representation of
negative values, the internal implementation of a matrix multiplication, a
physical ReLU, or digital fixed-point rounding.

Optical gain changes the available photon budget. It does not multiply the
represented numerical value. Consequently, increasing power or adding an
amplifier never removes numerical error that has already entered a signal.

Every delivered optical edge is treated as an effective detected and normalized
channel, including an edge whose receiving macro is optical. This is a strong
abstraction. The graph must not be interpreted as a physical assertion that
every optical-to-optical connection contains a particular detector, nor as a
complete specification of the devices needed to realize that connection.

The reported error is

\[
R = \sqrt{\mathbb E\!\left[\|\widehat y-y\|_2^2\right]},
\]

where all requested digital output vectors are stacked into one vector. This is
the RMS of the joint Euclidean error, **not** the mean per-coordinate RMSE. An
output listed twice appears twice in that stack.

## 2. Primitive noise sources

Every primitive source is a zero-mean isotropic Gaussian vector. Its
`noise_std` or `added_noise_std` is a per-coordinate standard deviation in
normalized numerical units. Distinct primitive sources are independent of one
another and of the external input.

Independence does not mean that every downstream signal has independent noise.
When two branches inherit one primitive source, they inherit the same random
vector. The evaluator preserves that sharing across branches, repeated
operands, output sinks and reconvergence.

For a component of output dimension \(d\), its usual contribution is

\[
\widehat z=f(\widehat u_1,\ldots,\widehat u_m)+\sigma_c\xi_c,
\qquad \xi_c\sim\mathcal N(0,I_d).
\]

Serial source re-encoding is the exception: it performs independent encoding
operations on the exact external input and draws component noise separately for
each branch.

For an optical branch with source photon budget \(P\) per scalar, transmission
efficiency \(\eta\), allocation fraction \(p_j\) and effective optical gain
\(g\), the branch's shot-noise surrogate is

\[
v_{\mathrm{shot},j}=\frac{1}{P\,p_j\,\eta\,g}.
\]

This additive Gaussian approximation is independent of the signal's value. It
does not model signal-dependent Poisson statistics, coherent interference,
quantum states, bandwidth-dependent noise spectra, or a detailed detector.

## 3. Fan-out rules

A fan-out rule is required when a node has more than one outgoing use. A use is
an operand occurrence or an output sink; two appearances of the same source in
an `add` count as two uses. Ordering is deterministic: network-node order,
operand index, then output-list order.

Below, \(z\) is the noisy normalized value after the source macro,
\(k\) is its number of outgoing uses, and \(\sigma_r\) is the rule's
`added_noise_std`. The sources denoted by different symbols are independent.
Cross-domain conversion noise, when present, is added after these operations.

### Digital copying

`digital_copy` sends

\[
z_j=z+\sigma_r\xi_r.
\]

The rule noise is shared across branches. With `added_noise_std = 0`, the
numerical copy itself is exact in this backend. Its declared area, energy and
latency still apply. Electrical copying is therefore not assumed to have zero
physical resource cost.

### Passive splitting

`passive_split` uses \(g=1\) and fractions summing to one:

\[
z_j=z+\sigma_r\xi_r+
\sqrt{\frac{1}{Pp_j\eta}}\,\xi_j.
\]

For equal splitting, \(p_j=1/k\). A nonzero rule noise describes a shared
technical disturbance in this model; it is not additional attenuation.

### Source boost

`source_boost` uses the same normalized branch equation with the configured
\(g\geq1\):

\[
z_j=z+\sigma_r\xi_r+
\sqrt{\frac{1}{Pp_j\eta g}}\,\xi_j.
\]

It requires an optical input encoder whose component has
`source_scalable = true`. It is forbidden for intermediate computed values.
The source photon-energy contribution is multiplied by \(g\). Existing macro
noise and rule noise are not divided by \(g\).

### Amplification

`amplify` has the same normalized branch equation as source boost, but can be
applied to an intermediate optical value. Its rule noise is inserted once,
before the split, and remains shared by every branch.

The distinction is a physical implementation option and its parameters, guards
and costs, rather than a different ideal numerical function. Equal parameters
can make source boosting and amplification score identically in this surrogate.
No law in the current schema derives amplifier noise from gain: the library
must supply a credible `added_noise_std`. A zero-noise amplifier entry is an
abstract user-supplied assumption, not a claim of physical realizability.

### Regeneration

`regenerate` first measures the noisy incoming signal once:

\[
m=z+\sigma_r\xi_m+
\sqrt{\frac{1}{P\eta}}\,\xi_{\mathrm{det}}.
\]

It then re-encodes that same measured value independently for each branch:

\[
z_j=m+\sigma_r\xi_{\mathrm{enc},j}+
\sqrt{\frac{1}{P\eta g}}\,\xi_j.
\]

Each new branch receives a full photon budget \(Pg\), so its allocation
fraction is one. `split_policy` does not affect regeneration. There are two
shared new sources: measurement noise and measurement shot noise. Each branch
also gets independent re-encoding and detection noise. The same library
parameter \(\sigma_r\) sets the measurement and re-encoding standard
deviations, but those draws are independent.

Crucially, \(z\) still contains all inherited analog error. Regeneration is
not an oracle that reconstructs the ideal value. This implementation also does
not introduce a quantizer or a nonlinear error-correcting decision.

### Serial re-encoding

`serial_reencode` is allowed only for a source-scalable optical input encoder,
and only when the request permits serialization. For exact external input
\(x\), it produces

\[
z_j=x+\sigma_c\xi_{c,j}+\sigma_r\xi_{r,j}+
\sqrt{\frac{1}{P\eta}}\,\xi_j.
\]

All three new sources are independent across branches. This is a repeated
encoding of an available external value, not cloning an intermediate noisy
state. Each branch receives the full budget \(P\); `gain` must be one and
`split_policy` is unused. The source macro is reused sequentially, charging
\(k\) encoding operations and increasing branch completion times.

## 4. Split allocation heuristic

The `equal` policy sets \(p_j=1/k\). The `sensitivity` policy estimates
downstream influence using a reverse pass:

- An output-sink use has influence one.
- A use feeding node \(v\) has influence \(L_v S_v\), where
  \(L_v=\|W_v\|_2\) for a linear node and one otherwise.
- \(S_v\) is the sum of the influences of all outgoing uses of \(v\).

The outgoing fractions are proportional to these influence estimates. Before
normalizing, each estimate is floored at
\(10^{-9}\max(1,\max_j s_j)\), so even a numerically zero-influence branch
receives a strictly positive allocation.

This is a fixed, deterministic Lipschitz heuristic. It can be useful for
unequal branch sensitivities, but it is not a continuous global optimizer and
does not account for all cancellations at reconvergence. Search completeness
over discrete library entries does not make these fractions globally optimal.

## 5. Exact covariance for affine graphs

If the graph has no ReLU, each error vector is represented as a sum of loadings
on independent unit Gaussian sources:

\[
e_v=\sum_s A_{v,s}\xi_s.
\]

The representation propagates as follows:

\[
\begin{aligned}
\text{linear:}\quad&A_{v,s}=W_vA_{u,s},\\
\text{add:}\quad&A_{v,s}=\sum_{u\text{ operand of }v}A_{u,s},\\
\text{new noise:}\quad&A_{v,\mathrm{new}}=\sigma I.
\end{aligned}
\]

The sum at an `add` is by source identifier. It does not replace a shared source
with fresh independent copies. Repeated occurrences of an operand are included
repeatedly.

Stack the loadings of all output blocks into \(B_s\). Then

\[
\Sigma_{\mathrm{out}}=\sum_s B_sB_s^\top,
\qquad
R=\sqrt{\operatorname{tr}(\Sigma_{\mathrm{out}})}.
\]

The report includes the full output covariance and labels the analysis
`exact_affine_gaussian`. Exactness refers to these equations under the model;
calculations use floating-point arithmetic.

For example, if \(a=x+\epsilon\) and two noiseless branches compute \(a\)
and \(-a\), their sum cancels the shared error exactly. Independent branch
noise would not cancel. For separate outputs, the same example produces a
negative off-diagonal covariance. Keeping only marginal variances loses both
facts.

## 6. Conservative RMS bound for ReLU graphs

If any node is a ReLU, the backend uses a scalar RMS upper bound throughout the
graph. It does not report a Gaussian output distribution or an exact output
covariance for that case.

If \(r_u\) bounds \(\sqrt{\mathbb E\|e_u\|_2^2}\), the propagation is

\[
\begin{aligned}
\text{linear:}\quad&r_v\leq\|W_v\|_2r_u,\\
\text{ReLU or identity:}\quad&r_v\leq r_u,\\
\text{add:}\quad&r_v\leq\sum_{u\text{ operand of }v}r_u.
\end{aligned}
\]

The addition rule is the Minkowski inequality and is valid even for correlated
operands. ReLU is nonexpansive in the Euclidean norm.

A newly added independent, zero-mean isotropic source permits the tighter
quadrature update

\[
r_{\mathrm{new}}\leq\sqrt{r_{\mathrm{old}}^2+d\sigma^2}.
\]

The cross term has zero expectation by independence and zero mean. This update
is not used to declare two inherited branch errors independent. For output
blocks, the final bound is

\[
R\leq\sqrt{\sum_o r_o^2},
\]

because the squared norm of a stacked vector is the sum of its block squared
norms, regardless of cross-block correlations.

The result is labeled `conservative_lipschitz_rms`. It can be loose, especially
when paths cancel or when ReLU frequently suppresses an error. It bounds RMS
under the model for every admissible deterministic input, and hence also for
any supported random input independent of the primitive noise sources. It does
not bound every individual noisy inference.

## 7. Nominal interval and headroom guards

`infer_bounds` computes noiseless interval enclosures. With \(W_+\) and
\(W_-\) denoting the nonnegative and nonpositive parts of a weight matrix,

\[
\begin{aligned}
\ell_v&=W_+\ell_u+W_-h_u+b,\\
h_v&=W_+h_u+W_-\ell_u+b.
\end{aligned}
\]

Addition sums endpoints; ReLU applies `max(0, ·)` to both endpoints; identity
passes them through. Correlations between inputs to an operation are not
retained, so the intervals can overestimate the reachable range.

A candidate is rejected when its maximum absolute nominal interval endpoint
exceeds its selected macro's `max_abs_value`. Source boost, amplification and
regeneration additionally require

\[
\sqrt g\,\max(|\ell_v|,|h_v|)\leq M_v.
\]

This is a declared field-amplitude headroom surrogate using the originating
macro's range \(M_v\). The current rule schema does not carry a separate
amplifier or regenerator saturation curve. It is not a complete derivation for
every possible optical data encoding.

These are guards on **noiseless nominal values**. Gaussian noise has unbounded
support, and the simulator applies no clipping. Passing the guards cannot imply
that stochastic saturation never occurs. The intervals and comparisons use
ordinary floating-point arithmetic, with a small comparison tolerance, rather
than directed-rounding interval arithmetic or a formal numerical proof system.

Other guards enforce operation support, allowed domains, macro output width,
required editability, valid cross-domain conversions, source scalability and
permission for serialization. No input-port count or internal macro signal
range is independently synthesized or verified.

## 8. Conversion semantics

Whenever source and destination domains differ, exactly one declared conversion
is instantiated for that use. Each conversion adds independent isotropic
`noise_std` and its declared area, latency and energy. All final output sinks
are digital; an optical output therefore requires conversion.

For an optical source, the effective branch shot term is already included.
Conversion `noise_std` is additional residual noise in the declared model;
library authors should not enter a measured total that unintentionally counts
the same shot term twice. A digital-to-optical edge has conversion noise, while
its receiving optical macro's output later receives that macro's usual optical
channel treatment.

## 9. Resource accounting

All resource parameters are nonnegative and measured in the units shown in
their names: square micrometers, nanoseconds and picojoules.

For an optical macro producing \(d\) scalars, the baseline launch energy is

\[
E_\gamma=
\frac{Pd\,hc}{\lambda\eta_{\mathrm{wp}}}\,10^{12}
\quad\text{pJ},
\]

where \(\lambda\) is converted from nanometers to meters and
\(\eta_{\mathrm{wp}}\) is wall-plug efficiency. Transmission efficiency
\(\eta\) affects delivered shot noise, while wall-plug efficiency affects
electrical launch energy.

The launch-energy multiplier is:

| Rule | Multiplier on \(E_\gamma\) | Meaning |
|---|---:|---|
| No fan-out rule or passive split | \(1\) | One baseline launch |
| Source boost | \(g\) | Stronger source launch |
| Amplify | \(g\) | Baseline plus replenished optical energy |
| Regenerate | \(1+kg\) | Original launch plus \(k\) fresh launches |
| Serial re-encode | \(k\) | \(k\) full-power source encodings |

Amplifier and regeneration replenishment use the source macro's wall-plug
efficiency as a surrogate. A device-specific amplifier efficiency model is not
implemented.

Each neural macro also pays `energy_pj`, and a linear macro pays
`energy_per_mac_pj` times input width times output width. Serial re-encoding
repeats the source macro's dynamic cost \(k\) times. Other macros execute once.

Each fan-out recipe pays its declared area once. Its `energy_pj` is paid once,
except that a serial recipe pays it per encoding. Every cross-domain edge pays
its conversion costs. The total area is the sum of macro, recipe and conversion
areas; zero-cost channel and output records are bookkeeping objects.

Library `energy_pj` entries should represent the declared residual overhead,
excluding launch energy already added by these equations. Otherwise the model
double-counts energy.

**Arity caveat:** scalar rule costs describe one complete fan-out recipe. This
version does not synthesize a \(k\)-port splitter tree, add an area term per
regeneration encoder, or enforce a maximum number of output ports. Its recipe
costs must therefore be characterized for the intended fan-out sizes. Photon
energy and serial dynamic costs scale as stated, but that does not make the
area model automatically valid for arbitrary \(k\).

## 10. Schedule model

There is one dedicated macro per neural node. Nodes start as soon as all
operand deliveries have completed. Each ordinary fan-out recipe starts after
its macro completes, and each branch conversion follows its recipe. Branches
otherwise proceed in parallel. There is no shared memory controller,
interconnect contention, placement-dependent delay, clock-domain analysis or
arbitrary resource reuse.

For serial source re-encoding, let source latency be \(L_c\) and recipe
latency be \(L_r\). Branch \(j\), indexed from zero, is delivered before
any conversion at

\[
t_j=t_{\mathrm{start}}+(j+1)(L_c+L_r).
\]

The hardware schedule records the repeated source operations and rule
operations on their shared instances. Total latency is the latest final
digital-output completion time. It describes a single inference under this
schedule, not steady-state throughput or a general optimal schedule.

## 11. Independent forward Monte Carlo

`simulate` draws inputs uniformly from their declared box and executes the
ideal and noisy computations separately. It draws primitive sources while
executing the physical graph: a pre-split source is drawn once, and branch
sources are drawn independently. It does not sample from the output covariance
computed by `evaluate`.

For each sample \(i\), define \(q_i=\|\widehat y_i-y_i\|_2^2\). The
reported MSE and estimated standard error are

\[
\widehat{\mathrm{MSE}}=\frac1n\sum_i q_i,
\qquad
\widehat{\mathrm{SE}}_{\mathrm{MSE}}=
\frac{\operatorname{sd}(q_1,\ldots,q_n)}{\sqrt n}.
\]

Empirical RMS is the square root of empirical MSE. Its reported standard error
uses the delta-method approximation
\(\widehat{\mathrm{SE}}_{\mathrm{RMS}}=
\widehat{\mathrm{SE}}_{\mathrm{MSE}}/(2\widehat{\mathrm{RMS}})\)
when RMS is positive.

For an affine graph, `consistent_with_bound` checks **two-sided agreement**:
the absolute difference between empirical and analytic MSE must be at most four
estimated standard errors, or fall within a relative-only roundoff tolerance
of \(10^{-12}\). For a nonlinear graph it checks only that empirical MSE is
no greater than the squared upper bound plus four estimated standard errors,
with the same relative-only tolerance. A positive quantity is never treated as
zero by an absolute tolerance. These are heuristic sampling diagnostics, not
formal confidence statements or proofs of a bound. A finite Monte Carlo run
cannot validate a physical model against real hardware.

Finite-precision forward sampling can erase small noise added to large nominal
signals, and subtracting nearly equal outputs can erase error information.
For example, unit noise around a signal of magnitude \(10^{20}\) can disappear
in floating-point arithmetic. An affine empirical MSE of zero with a positive
analytic prediction is therefore a discrepancy, even if the estimated standard
error is also zero. In that case `mse_difference_standard_errors` is `null`,
`consistent_with_bound` is false, and a diagnostic records the mismatch.

For an affine graph, the error distribution is input-independent under this
additive model, and the analytic MSE is `trace(output_covariance)`. For a
nonlinear graph, the uniform-input experiment samples only one input
distribution, while the Lipschitz result is distribution-independent under its
stated assumptions.

## 12. Interpretation of a feasible report

A feasible report means that the declared discrete decisions satisfy the
implemented guards and requested metric budgets under this model. It includes
the assumptions, source descriptions, rule trace, nominal intervals,
implementation graph and schedule needed to inspect that conclusion.

It does not establish manufacturability, physical encoding correctness,
stochastic saturation safety, calibrated device accuracy, layout feasibility,
hardware throughput, or global optimality beyond the finite search space and
its declared evaluation model. Such claims require additional backends,
characterization and validation.
