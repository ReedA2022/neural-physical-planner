# Multiphysics extension

This directory adopts the supplied 3 October 2026 specification as the next
development roadmap. It does not change the v0.3 runtime or advertise a completed
multiphysics compiler. The current runtime remains version **0.3.0**.

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

M0 integrates and reconciles the specification. M1–M6 are future implementation
gates. The old four-part milestone passed for its documented optical distribution
scope; those approvals are useful evidence but do not approve the broader stages.
Read [M0_GATE.md](M0_GATE.md) for the actual gate decision and executed evidence.

The next runtime deliverable is **M1**: versioned shared component/evidence
contracts, an explicit digital reference, and one bounded electronic/photonic
model family. Interface and state checks must precede wider search. No acoustic,
quantum, decoder or mixed-FFN support is claimed by this integration. H1–H6 are
registered research questions with unresolved models and sweep bounds, not results.
