# Optical export and independent SAX comparison

Part 3 checks the composition of the planner's declared optical power model with
the external **SAX** circuit solver. It verifies full-scale optical power at every
detector input and unused splitter termination. It does not validate the device
parameters against measurements, the electronics, timing, noise, or a whole NN.
The bundled technology remains illustrative and unreviewed.

## Install and run

Export needs only the core package. Simulation additionally needs Python 3.11 or
newer and the optional photonics dependencies:

```sh
python -m pip install '.[photonics]'

npp realize --network examples/physical/network.json \
  --spec examples/physical/realization.yaml --out-dir runs/physical

npp export-optical --realization runs/physical/realization.json \
  --out runs/physical/optical-netlist.json

npp simulate-realization --realization runs/physical/realization.json \
  --wavelengths-nm 1540 1550 1560 --out-dir runs/optical-check
```

The adapter is tested with SAX **0.18.2**, JAX **0.9.2**, and KLU/JAX **0.5.2**.
The optional extra selects SAX `>=0.18.2,<0.19`. The solver backend is explicitly
`klu`; each optical segment is compiled through a real `sax.circuit(...)` call.
There is no in-house solver fallback. Missing dependencies, solver exceptions,
invalid saved records, or infeasible sweep points produce structured errors.

Commands preserve existing output files unless `--overwrite` is provided. The
simulation directory contains a portable numerical comparison, the optical
export, and an offline HTML report. CLI success is `0`; numerical disagreement
is `3`; invalid input or an unavailable/failed solver is `2`.

## What crosses the external-solver boundary

Each source launches a separate connected optical segment. The source is an
external excitation port with its recorded full-scale power; its electrical
energy, RIN, startup and sample-encoding model are outside SAX. Detectors are
external measurement boundaries at their **optical input**, with no photodetector
S matrix. Unused splitter leaves are exposed matched termination boundaries.
Every nonexcited external port is matched and has zero incoming field.

| Planner component | Exported optical model |
| --- | --- |
| Splitter | Three-port reciprocal model; branch amplitudes are `sqrt(r*T)` and `sqrt((1-r)*T)`, where `T=10**(-excess_loss_db/10)` |
| Waveguide | Reciprocal attenuator with propagation phase; amplitude is `10**(-loss_db_per_um*length_um/20)` |
| Regeneration modulator | Reciprocal attenuator at electrical sample **fixed to 1**; amplitude is `10**(-insertion_loss_db/20)` |
| Source | External excitation at the source's specified full-scale power |
| Detector | External optical measurement port; no electrical response or noise simulation |

Absent scattering entries are zero. These are phenomenological, reflection-free
component models, not PDK compact models or electromagnetic solutions. Independent
sources are never summed coherently: supported generated graphs have one source
per optical segment and no optical recombination.

The `regenerate` recipe is split at every electrical connection. SAX separately
checks (1) the launch and splitter tree up to the early detectors and (2) each
fresh carrier through its fixed-sample modulator and guide to the final detector.
Passing these checks does **not** verify that the early detector and modulator
actually regenerate a waveform or that the electrical signal arrives in time.
The result explicitly labels coverage `partial_optical_only` for both recipes.

The waveguide phase uses a first-order effective-index model around the recorded
operating wavelength. With wavelengths and lengths in micrometres:

```text
neff(wl) = neff0 - (wl-wl0)*(ng-neff0)/wl0
phase(wl) = 2*pi*neff(wl)*length/wl
```

This keeps the declared effective and group indices distinct, but neither phase
nor group delay is a comparison target. Attenuation and splitting ratios are
constant across wavelength within the declared operating envelope. A wavelength
sweep exercises composition and guard handling; it is not evidence that a real
device has a flat spectral response.

## Numerical comparison and acceptance

Before export or simulation, `check_realization` deterministically reconstructs
the realization from its embedded inputs and checks every saved field. Changing
a stored metric, netlist, source power, or technology hash cannot bypass replay.
Hashes provide reproducibility and content binding, not author authentication.

At each requested wavelength the scalar planner reevaluates the same graph with
that operating wavelength. Every source, detector and passive component must be
within its envelope. A valid but physically infeasible record can be exported for
inspection; it cannot receive a simulation pass. If any sweep point is infeasible,
the whole request returns an error before starting the external engine.

For every exported detector/termination boundary, the comparison uses
`SAX_power = abs(S[output,input])**2 * source_power_mw`. Expected power comes from
the planner's reevaluated per-instance ledger. The independently encoded SAX field
models do not call the scalar component evaluator or its transmission helper.
The acceptance rule is:

```text
abs(SAX_power - planner_power) <= atol_mw + rtol*abs(planner_power)
```

Defaults are `rtol=1e-5`, `atol_mw=1e-9`. Options must be finite numbers, excluding
booleans and numeric strings, with `0 <= rtol <= 0.01` and
`0 <= atol_mw <= 1e-4`. Both may be zero for a deliberately exact comparison.
The chosen tolerances and the allowed error for each boundary are saved. A failed
comparison returns `status="failed"` and individual `optical_power_mismatch`
diagnostics; it never becomes a physical validity claim.

The default is one evaluation at the recorded wavelength. An explicit sweep
accepts 1–33 distinct positive finite wavelengths in nanometres. The exporter is
bounded to 320 source segments, 128 optical component instances per segment, and
20,000 summed external scattering-matrix entries. It composes one segment at a
time rather than constructing one matrix for all lanes and fresh carriers.

## Python API and portable artifacts

```python
from npp.optical_simulation import export_optical_netlist, simulate_realization

optical_export = export_optical_netlist(realization_record)
comparison = simulate_realization(
    realization_record, wavelengths_nm=[1540.0, 1550.0, 1560.0],
    rtol=1e-5, atol_mw=1e-9,
)
assert comparison["passed"]
```

`export_optical_netlist` returns JSON-compatible data containing the original
realization hashes, operating point, full technology snapshot and evidence,
scope, model definitions, excluded electrical connections, and the segments.
Each segment contains:

- A plain SAX `netlist` with `instances`, `connections`, `ports` and numeric model
  settings. Safe local instance names (`i0`, `i1`, ...) support NN IDs containing
  punctuation without name collisions in SAX.
- An `instance_mapping` from local SAX names to original physical instance and
  technology component IDs.
- The source role, original output endpoint, NN source/lane binding where
  applicable, excitation power and external input-port name.
- Every external output's original endpoint, kind and final NN receiver binding
  (or matched termination binding). Early regeneration detector inputs have no
  final NN-use binding because they are intermediate electrical boundaries.

The exported model identifiers are adapter contracts, not built-in SAX names.
Their formulas and parameter settings are included in the artifact; executable
implementations are in `npp.optical_simulation`. Consumers may supply equivalent
SAX model functions or substitute independently calibrated component models for
a separate analysis. The comparison API accepts a replayable realization, not an
arbitrary edited export.

`simulate_realization` returns `kind="sax_optical_comparison"`, engine/version,
tolerances, the export, and one `samples` row per wavelength. Each sample contains
all boundary comparisons with expected power, SAX power, absolute/relative
errors, allowed error and pass status. The summary records segment, wavelength
and comparison counts and maximum errors. The scope and unchecked properties
travel with the numerical result. No runtime credentials or external service
are needed once the optional packages are installed.

## Validation evidence

The adapter tests include independent hand calculations for two-, four- and
eight-way distribution trees, an unused third-tree leaf, unequal route lengths,
multiple scalar lanes, repeated NN operand bindings, and separate regenerative
carrier paths. Tests assert that real `sax.circuit` calls occur. A deliberately
incorrect field model must produce numerical disagreement; engine errors and
missing dependencies must produce errors rather than passes. Tampered records,
infeasible launch powers, out-of-envelope sweeps, strict numeric inputs and size
limits are also exercised. Real solver tests skip explicitly when the photonics
extra is absent; export and input-validation tests still run.

Primary API/model references (accessed 2026-10-02):

- [SAX circuit API and solver backends](https://gdsfactory.github.io/sax/api/)
- [SAX YAML/netlist format](https://gdsfactory.github.io/sax/nbs/examples/03_circuit_from_yaml/)
- [SAX component model reference](https://gdsfactory.github.io/sax/models/)
- [GDSFactory Plugins SAX composition example](https://gdsfactory.github.io/gplugins/notebooks/sax_01_sax/)

These sources support the solver workflow and model conventions. They do not
provide experimental calibration for the bundled numerical technology values.
