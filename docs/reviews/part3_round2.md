# Part 3 independent review — round 2

Date: 2026-10-03 (UTC)

Decision: **PASS. Part 3 is complete enough within its declared full-scale optical
power-composition scope; Part 4 may begin.**

## Resolution of the blocking findings

- **R3.1 resolved.** The report now uses the adapter's shared tolerance ceilings.
  A stored comparison with `rtol=1` and a 100% power discrepancy cannot receive the
  complete agreement label even when its error arithmetic is coordinated. The
  absolute-tolerance ceiling is also checked. Both unchanged review probes pass.
- **R3.2 resolved.** Supported comparison/export versions, backend and adapter
  identifiers are checked. A complete technology snapshot must validate, its
  status must agree with that snapshot, and all four source hash fields must have
  the required SHA-256 shape. Complete supported model definitions, netlists,
  instance mappings, local model settings and exact port coverage are required.
  Missing bodies, fake provenance objects, unknown versions, nonexistent netlist
  endpoints and empty mappings are now marked incomplete or inconsistent.
- During this round, an additional content-consistency probe found that a valid
  but modified embedded technology snapshot could retain its old recorded hash.
  The reviewer added a regression before the fix. The final renderer now checks
  the exact embedded snapshot using the same finite canonical JSON hash function
  as physical realization records. Both a changed snapshot and a false digest
  suppress agreement. A separate valid Unicode realization/export regression
  confirms the correct hashing convention. The report expressly distinguishes
  this check from authentication and identifies the other hashes whose bodies
  are absent and therefore cannot be recomputed.

The fixes preserve raw fields for inspection and do not execute SAX, propagate
optical signals, or authenticate the author. Correct supported comparisons retain
their recorded agreement status.

## Independent retest evidence

The reviewer reran the original report regressions unchanged, the new hash
regression, all independent optical review tests, and the existing module/report/
CLI integration tests on the final source tree:

```sh
MPLCONFIGDIR=/tmp/npp-mpl /tmp/npp-v02-env/bin/python -m unittest \
  tests.test_optical_review tests.test_optical_simulation \
  tests.test_optical_report tests.test_optical_cli -q
```

Result: **36 tests ran; 35 pass and one expected dependency-absence branch is
skipped**. This environment uses real SAX 0.18.2, JAX 0.9.2 and KLU/JAX 0.5.2.
The suite includes actual `sax.circuit` composition, independent power oracles,
regenerative segment boundaries, tamper rejection, deliberately wrong models and
scalar expected powers, nonfinite/missing engine results, guarded sweeps, report
consistency, and CLI output/exit behavior.

The identical command with base `python` also ran on the final tree:

```sh
python -m unittest tests.test_optical_review tests.test_optical_simulation \
  tests.test_optical_report tests.test_optical_cli -q
```

Result: **36 tests ran; 26 pass and ten optional-engine tests are explicitly
skipped**. The absent-engine CLI branch returns its actionable dependency error;
export, replay, presentation and input-validation checks remain available.

The independent read-only report subreview reproduced every original report
blocker after the fix and confirmed the agreement label is suppressed. Its valid
fixture remains consistent. Repeating 3,960 JSON-compatible malformed substitutions
across 660 record paths found no unhandled renderer exceptions. Inspection of the
final hash fix and the fresh targeted tests completes that subreview's separate
content-binding follow-up.

## Completion judgment

No remaining blocker was identified within Part 3's declared scope. The product
provides portable source-rooted optical netlists, explicit electrical cuts and
matched terminations, independent external composition using field models,
per-boundary comparison across guarded wavelengths, reproducible engine/version
records, and an inspectable offline report with conservative status handling.

Passing SAX comparison checks agreement of full-scale optical power under the
shared declared component assumptions. It does not calibrate devices or validate
electrical regeneration, noise, timing, phase/group delay, whole-NN computation,
task accuracy or fabrication readiness. The reference technology remains
illustrative and unreviewed by a photonics expert. These limitations are retained
in the artifacts and documentation and are not promoted into validation claims.

## Reviewed final source fingerprints

- `npp/optical_simulation.py`: `e57c57e8a35cafef2b1daf8b9ca53afe2d02e2c283e36497ef83e10fc2886c9e`
- `npp/optical_report.py`: `f823d39dc9f84c5cf093df25c84b1e4b6758b7c3111f0c00fb2aa7846e9b06f7`
- `npp/cli.py`: `86c4727dc8f5765a57aabab57a33b18b7a4a8bfe90a324ac8385e5d9bc8e416c`
- `tests/test_optical_review.py`: `a52367818e488302d6b42625313f2d16e2d1060582e5b018a6a928c568124274`
