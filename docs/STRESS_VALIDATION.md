# v0.3.1 stress validation

Date: 2026-10-03. This maintenance release targets the documented research
prototype: legacy neural macro planning and bounded optical distribution circuit
design. It does not implement multiphysics stages M1–M6.

## Shipping criteria

The release gate requires no unexplained failures in deterministic stress
campaigns, the complete installed optional-dependency test suite, the core suite
with explicit dependency skips, the unchanged v0.3 compatibility checks, and a
wheel installation exercised outside the checkout. An independent critique
agent must reproduce the defects it finds, review the corrections and approve
this bounded software release. A passing local replay alone does not constitute
independent physics validation.

## Campaign coverage

Each case is one recorded scenario, not one assertion. Randomized scenarios are
seeded; some deterministic boundary cases intentionally repeat across seeds.
Failures preserve their index, family and a reproduction recipe or fixture.

| Area | Scenarios and independent checks |
| --- | --- |
| Inputs | Equivalent units and projects; NumPy round trips; restricted optional weight formats; ONNX predictions against ONNX reference and NumPy; malformed specs, aliases, booleans and extreme quantities |
| Legacy planner | Affine and nonlinear DAGs; shared primitive noise and reconvergence; small exhaustive searches against a separate oracle; optical fanout; altered records; numeric extremes and near-tie frontiers |
| Physical circuits | Separate scalar, power, energy, delay and covariance calculations; source regeneration; loss conservation; envelope edges; supported large fanouts; malformed graphs and replay; positive underflow |
| Designer | Independent finite-grid enumeration and all-pairs dominance; locks and baselines; equivalent quantities; malformed requests; altered saved results; search and size boundaries |
| Reports and CLI | Nested record corruption, missing evidence and HTML injection; actual portable project, macro plan, physical, designer and export/replay flows; overwrite preservation and exit-status distinctions |

The report campaign adds 25 actual CLI scenarios per seed to its requested
mutation count. An actual SAX run separately compares four passive/regenerative
circuits at three wavelengths; these comparisons are not counted as software
stress scenarios.

## Defects found and corrected

The first campaigns required revisions. The first independent critique also
required revisions, and found another optical-energy counterexample after the
initial fix. The final gate therefore depends on retests of those concrete
counterexamples as well as new seeded campaigns.

| Defect | Correction and behavioral consequence |
| --- | --- |
| Equivalent decimal quantities produced different input hashes | Convert decimal unit strings before binary64 conversion; equivalent values now have identical normalized records. |
| Nested YAML aliases caused exponential traversal | Memoize expanded container counts and reject aliases introducing more than 1,000,000 repeated values/containers. Plain trees are not subject to this alias limit. |
| Boolean tensor entries were coerced to numbers | Reject booleans in inline weights/bias and imported numeric arguments before NumPy conversion. |
| Nonzero quantity strings became zero or negative zero | Trap Decimal underflow and nonzero-to-zero float conversion in both friendly configuration and designer parsers. |
| Tiny nonzero noise or loss became a free/zero-error contribution | Reject unsupported positive arithmetic ranges. Stable Euclidean aggregation preserves representable nonlinear RMS bounds. |
| Intermediate SI photon-energy arithmetic changed budget decisions | Recover normal representable final energy through bounded mantissa/exponent arithmetic when ordinary intermediates are subnormal, zero or nonfinite; otherwise reject unsupported final ranges. |
| Relaxed legacy dominance lost valid frontier alternatives | Require exact componentwise non-worsening plus a relative-significant improvement; deduplicate exact objective ties only. The designer retains its separate exact Pareto relation and tied alternatives. |
| Monte Carlo uncertainty overflowed but reported agreement | Scale squared errors before estimating uncertainty, guard modeled MSE range and reject nonfinite result trees. |
| Sampling accepted requests too large to allocate safely | Preflight at 1,000,000 samples, 16,000,000 estimated array elements and 100,000,000 multiply-accumulates; validate nonnegative integer seeds. |
| Extreme finite report metrics generated infinite SVG coordinates | Normalize before pixel arithmetic and form axis labels without overflowing binary64. |
| Decoder depth errors lacked a dedicated structured error path | Translate decoder `RecursionError` to an input diagnostic; test both the observed CLI rejection and an injected decoder failure. |

## Reproduce

From the checkout root, without Python optimization (`-O`/`-OO` disables the
assertions used by the campaigns and is explicitly rejected):

```sh
python -m pip install '.[all,photonics]'
python scripts/stress_release.py --out-dir runs/my-release-stress \
  --seeds 20261003 731 987654321 --cases 400 --require-optional --sax
python -m unittest discover -s tests -v
python scripts/check_multiphysics_baseline.py --report baseline-check.json
```

Use a new output directory. Individual scripts in `scripts/stress_*.py` can
reproduce a particular family/seed. The aggregate records every campaign's
failures and skips, dependency requirements, source hashes and actual SAX
execution. Missing required optional cases produce an incomplete gate, not PASS;
an exception or any source change during execution fails the gate. The source
hashes identify the evaluated code even when the run precedes its final commit.

## Final evidence

The final source freeze passed the integrated checks below. Machine-readable
results and logs are kept in
[`runs/validation-v0.3.1`](../runs/validation-v0.3.1/); the independent shipping
judgment and its limits are recorded in
[`release_v031_round1.md`](reviews/release_v031_round1.md).

| Final gate | Executed outcome | Evidence |
| --- | --- | --- |
| Five campaigns, three seeds, 400 cases per area/seed plus CLI flows | **6,075 passed; 0 failed; 0 skipped**; evaluated source unchanged during run | `full-stress-final/summary.json` and its 15 individual reports |
| Actual SAX comparison | **183 boundary/wavelength comparisons passed** across four circuits; maximum absolute discrepancy `8.33e-17 mW` | `full-stress-final/external-sax.json` |
| Full dependency suite | **340 tests: 339 passed, 1 skipped**; the skipped test specifically requires SAX to be absent | `tests-full-final.log` |
| Core dependency suite | **340 tests: 298 passed, 42 dependency skips**; no errors or failures | `tests-core-final.log` |
| Frozen compatibility | **30 checks passed**, including integrity of 57 preserved artifacts; no golden files changed | `baseline-check.json` |
| Installed wheel outside checkout | **18 checks passed**, including bundled resources, project portability, planning/replay/simulation and a separate actual SAX run with 24 comparisons | `package-install.json` |

The evaluated source digest is
`9886fae30b229f798326045e013391f5fa97ade0396b812a0b2ab6e0092b5fbe`.
The aggregate contains the individual Python, technology-data and campaign
hashes; the reviewer independently compared all 28 installed Python/data files
with the checkout. Tests ran on Linux with Python 3.12.14. Exact full and core
dependency versions are preserved in `environment-full.json` and
`environment-core.json`; the release summary pins the recorded evidence.

The first aggregate run deliberately remains in `full-stress/` with a failed
status. Its three recorded execution failures exposed a harness integration
mistake: the legacy campaign used `actual_scenario_count` while the aggregate
required `scenario_count`. The canonical field was added and a real all-five
campaign contract test now covers this boundary. The final integrated run was
repeated in a new directory; the earlier failed record was not relabeled or
included in the final passing scenario count.

## Limits

All results concern software and the declared illustrative models. Hardware
parameters remain uncalibrated. SAX validates optical power composition under
matched assumptions; it does not validate noise, electronics, timing, neural
task accuracy, fabrication readiness or a hardware-performance claim.

Unsupported numerical ranges may be conservatively rejected even if a
mathematical answer exists. Plain JSON/YAML numeric literals follow ordinary
binary64 decoding; only explicit quantity strings have the new parser guarantee
against nonzero values rounding to zero. Sampling limits estimate work rather
than guarantee process memory. This campaign does not qualify every supported
Python/dependency version or constitute an exhaustive proof over all inputs.
