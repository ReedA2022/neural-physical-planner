# Multiphysics extension

This directory adopts the supplied 3 October 2026 specification. M0 was
reconciled against version **0.3.0**; **0.3.1** retained that runtime scope. The
**0.4.0 development** implementation adds an opt-in `npp mp` runtime while
preserving those frozen baselines. M1 and M2 have passed independent review.
This is not yet a completed M1–M6 release.

The extension explores digital electronics, analog electronics, classical
photonics, acoustics/phononics, and quantum computation through common contracts
and separate physical models. It requires explicit interfaces, persistent state,
resource scheduling, evidence and complete cost boundaries. Mathematical
equivalence, controlled approximation and model adaptation are separate tracks.

## Documents and executable checks

| File | Purpose |
| --- | --- |
| [SPECIFICATION_V1.md](SPECIFICATION_V1.md) | Full supplied requirements, equations, tables and source links |
| [Original specification](source/Neural_Physical_Planner_Multiphysics_Specification_v1.docx) | Unmodified authoritative attachment |
| [REQUIREMENTS.md](REQUIREMENTS.md) | REQ-01–37 mapped to actual v0.3 capabilities and remaining work |
| [MIGRATION.md](MIGRATION.md) | Compatibility rules, architecture decisions and ordered M0–M6 gates |
| [EXPERIMENTS.md](EXPERIMENTS.md) | H1–H6 hypotheses, baselines, controls and blockers |
| [experiments.json](experiments.json) | Proposed experiment registry; not executable planner input |
| [baseline_v03.json](baseline_v03.json) | Immutable baseline manifest for software regression checks |
| [M0_GATE.md](M0_GATE.md) | Executed checks, critique decisions and limits of the M0 delivery |
| [IMPLEMENTED.md](IMPLEMENTED.md) | Current executable runtime, commands and bounds |
| [IMPLEMENTATION_ACCEPTANCE.md](IMPLEMENTATION_ACCEPTANCE.md) | Concrete mechanisms, negative tests and requirement traceability for every gate |

Run the baseline compatibility gate from the repository root:

```sh
python scripts/check_multiphysics_baseline.py
python -m unittest discover -s tests -v
```

The first command checks preserved v0.3 results and input/schema meaning. It does
not validate new physical mechanisms. The full test suite requires optional
dependencies for all importer and external-solver tests; absent optional
dependencies produce explicit skips.

## Current delivery boundary

M1 implements shared component/evidence contracts, a complete bounded digital
FFN, and a signed analog matrix model. See [its independent review](../reviews/multiphysics_m1.md).
M2 adds the typed mixed graph and scheduling layer; [its independent gate passed](../reviews/multiphysics_m2.md).
M3–M6 are later gates. No acoustic, quantum or decoder support is claimed yet.
H1–H6 remain registered research questions until their executable studies run.
The old four-part milestone passed for its optical distribution scope; those
approvals do not approve the broader stages.
