# Physical technology packs

The v0.3 physical backend has a separate, strict technology contract. The existing
v0.1 network and macro-library contracts remain unchanged. A technology pack
describes one scalar optical distribution model; it does not describe the internal
implementation of signed matrix multiplication or fabricate a neural network.

**The bundled pack is illustrative and unreviewed.** All numeric device parameters,
costs and operating ranges are assumptions chosen for software experiments. No
photonics collaborator has reviewed them, no foundry process is claimed, and passing
software tests does not establish predictive accuracy. An independent circuit
solver using these same parameters can check composition, not calibrate the devices.

## Inspect and supply a pack

```bash
npp technology
npp technology --out technology.json
npp technology --file technology.yaml
```

`--out` requires `--overwrite` to replace an existing file. JSON and safe YAML are
accepted. Keys carry canonical units, for example `maximum_output_power_mw: 10.0`
and `latency_ns: 0.2`. Numeric strings such as `"10 mW"` are deliberately rejected
in this low-level contract; friendly project specifications still have their own
unit-aware frontend. Unknown fields, duplicate keys, nonfinite values, inconsistent
ranges and broken evidence references produce structured diagnostics.

Python API:

```python
from npp.technology import load_technology, get_component, evaluate_component

pack = load_technology()  # or a JSON/YAML filename
source = get_component(pack, "source")
metrics = evaluate_component(source, source_power_mw=2.0,
                             symbol_duration_ns=1.0)
assert metrics["energy_pj"] == 10.0
```

`validate_technology(mapping)` returns a normalized copy without changing its input.
`technology_schema()` returns JSON Schema; cross-field and evidence checks also run
in Python. `get_component` and `evaluate_component` expect an already validated pack
and component. `check_operating_point` returns diagnostics for an operating envelope;
it does not extrapolate outside it. Device-specific requirements such as the source
power range, receiver sensitivity and waveguide length are additionally enforced by
`evaluate_component`.

## Representation and ports

The only implemented encoding is `normalized_intensity`: a logical sample
`0 <= x <= 1` gives optical power `P = P_full_scale * x`. The full-scale power is an
explicit calibration parameter. The detector minimum is a requirement on that
full-scale value, not a claim that an encoded zero must emit light. Negative values,
coherent phase encoding, quantum states and wavelength multiplexing are unsupported.

The distribution interface assumes a sample has already been encoded at its launch
boundary. It does not supply a free implementation of the preceding NN operation.
Energy estimates use the full-scale value throughout a rectangular symbol; they are
upper bounds under that model, not workload-averaged measurements.

| Kind | Input ports | Output ports | Meaning |
| --- | --- | --- | --- |
| `source` | none inside the distribution boundary | optical `out` | Externally driven, already encoded launch; electrical energy includes wall-plug conversion. |
| `splitter` | optical `in` | optical `out1`, `out2` | Two-way division with fixed ratio and excess insertion loss. |
| `waveguide` | optical `in` | optical `out` | Uniform matched propagation with specified length, loss and delay. |
| `detector` | optical `in` | electrical `out` | Calibrated normalized sample plus an explicit noise contribution. |
| `modulator` | optical `carrier`, electrical `sample` | optical `out` | Ideal intensity reencoding using a supplied fresh carrier. |

Every port allows **one connection**, including electrical outputs. A detector may
not drive an unlimited number of modulators. Regeneration therefore needs dedicated
per-branch detection/reencoding, or a separately modeled electrical distribution
network. A graph must connect both modulator inputs; evaluating a modulator alone
only reports its full-scale transfer and overhead, not graph completeness.

The reference splitter has `power_ratio: 0.5` and `ratio_tunability: fixed`. Its
ratio is not a free search variable. A different ratio means another declared
device. Unused splitter branches must be exposed as explicitly terminated outputs
by a graph; their power is not redistributed to the used outputs.

## Equations and units

These equations are executable built-in models; packs cannot supply Python code or
arbitrary expressions. Let `t` be symbol duration in ns and `P` power in mW.

- Power transmission for loss `L` dB is `T = 10**(-L/10)`. A scattering adapter uses
  field transmission `sqrt(T)`, not `T`.
- A splitter produces `P*T*r` and `P*T*(1-r)`; insertion loss is in addition to the
  ideal division. The reference `r=0.5` is fixed.
- A waveguide of length `l` micrometres has `T = 10**(-alpha*l/10)` and delay
  `n_group*l*1e3/c` ns. Its illustrative core area is `l*width` plus fixed area;
  routing spacing and bends are excluded. `n_effective` is a phase-model parameter
  for a scattering adapter and is distinct from `n_group`.
- Source electrical energy is `P*t/eta_wallplug` pJ plus fixed energy. There is no
  free source-power multiplication; maximum output power is checked.
- The mean photon count is `N = (P*1e-3)*(t*1e-9)/(h*c/(lambda_nm*1e-9))`.
- A receiver with quantum efficiency `eta` has mean photoelectron count `eta*N`.
  At full scale its normalized added noise variance is
  `1/(eta*N) + (input_referred_noise_mw_rms/P)**2`. This assumes Poisson photons,
  ideal normalized gain and the stated independent electronic RMS noise. At smaller
  `x`, the shot contribution scales with `x`; the full-scale value is a conditional
  upper bound over `[0,1]` in this simplified receiver model.
- Source relative-intensity noise and modulator sample noise are reported as separate
  variance contributions. A graph must propagate shared upstream contributions;
  this scalar evaluator does not promise independence across branches or prove an
  NN error bound after regeneration.
- Modulation at full-scale sample one transmits carrier power times its insertion
  transmission. Costs include an illustrative receiver/driver overhead where stated.

All costs are per physical instance. Passive components may have zero fixed dynamic
energy in this model but still lose optical power, require source energy and occupy
area. The physical backend must not silently add these numbers to an existing macro
estimate that already accounts for the same devices.

`costs.latency_ns` is a fixed pipeline/propagation overhead and excludes acquisition
of the complete symbol. Waveguide delay is added to that overhead by the evaluator.
A graph scheduler must include the symbol duration when reporting when a complete
integrated sample is available; a 1 ns symbol is not delivered in less than 1 ns
merely because its component overheads sum to less than 1 ns. Regenerative stages
must also state their acquisition and scheduling assumptions explicitly.

## Evidence and review

Each pack has a version, sources, calibration status, assumptions, excluded effects
and review record. Each component binds evidence to four groups: `parameters`,
`operating_envelope`, `costs`, and `model`. A binding covers every field in its group;
the source's `coverage` and `limitations` explain exactly what it supports. Formula
references and standards alone cannot support numeric device parameters or costs.
Illustrative packs must explicitly cite assumptions for each numeric group.

`calibration_status: externally_supplied` identifies user-supplied data; it does not
certify it. A `reviewed` record requires a named reviewer and a `review_report`
source. The loader validates references and consistency, but cannot authenticate a
measurement, review or source. The bundled pack keeps `status: unreviewed`.

Primary sources inspected for this implementation:

- [NIST SI definitions](https://www.nist.gov/si-redefinition/definitions-si-base-units)
  support the exact constants `c = 299792458 m/s`, `h = 6.62607015e-34 J s` only.
- [GDSFactory SAX workflow](https://gdsfactory.github.io/gplugins/notebooks/sax_01_sax/)
  documents composing component scattering models and netlists. It supplies no
  calibrated component values for this pack.
- [Ansys photodetector compact-model scope](https://optics.ansys.com/hc/en-us/articles/360034710353-photodetector-simple-CML-Compiler-Model)
  describes distinct optical/electrical ports, shot and thermal noise, and saturation.
  The current reduced model does not embed or reproduce Ansys's implementation.

Not modeled: reflections/interference, dispersion, optical coupling and packaging,
modulator nonlinearity/extinction, transients/bandwidth distortion, temperature/process
variation, drift, foundry layout/DRC, complete IO/control/memory overhead, amplifiers,
switches, storage, or task accuracy. These omissions remain explicit even when the
pack passes validation or is used by an independent circuit solver.
