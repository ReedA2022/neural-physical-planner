# Version 0.3: inspectable optical distribution designs

This milestone develops a physical implementation layer for optical fanout subcircuits linked to a neural graph. Signed neural operators, complete electronic circuits, fabrication layout and full-network task accuracy remain separate engineering work. Physical subcircuit metrics must not silently replace or be added to the existing macro metrics.

## Ordered acceptance gates

1. **Technology pack:** strict, versioned components and physical signal contracts; explicit evidence and operating ranges; runnable inspection/export; independent reviewer tests and approval.
2. **Implementation graphs:** explicit component instances, typed ports, connections and fanout-dependent resource accounting; physical applicability checks; independent reviewer tests and approval.
3. **Independent simulation:** export supported optical circuits to an external circuit solver; compare transfer and loss predictions over declared conditions; separate external checks from unsupported electronics/noise claims; independent reviewer tests and approval.
4. **Designer interaction:** lock and forbid implementation choices, change requirements, regenerate alternatives and explain consequences; portable snapshots and replay; independent reviewer tests and approval.

Each gate is reviewed only after implementation. Review findings, fixes and final judgments are recorded in `docs/reviews/`. Later parts must preserve earlier contracts. A final integrated regression run follows all four approvals.

## Completed gates

| Part | Independent judgment | Review rounds |
| --- | --- | ---: |
| Technology pack | PASS | 2 |
| Physical implementation graphs | PASS | 2 |
| External optical simulation | PASS | 2 |
| Designer constraints and replanning | PASS | 1 |

The implementation, CLI examples and regression evidence are delivered together.
See [VALIDATION.md](VALIDATION.md) for final test counts, legacy compatibility and
the actual external-SAX integration run. These approvals concern the bounded
scope and declared models below.

## Evidence boundary

Automated critique verifies software and model consistency. It does not constitute human photonics review, measured calibration, manufacturing sign-off or a guarantee of physical prediction accuracy. Illustrative numbers must remain identified as assumptions. External simulation agreement is reported only for the properties and operating conditions actually checked.
