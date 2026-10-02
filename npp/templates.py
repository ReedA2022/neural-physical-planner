"""Small, fully runnable project templates with human-editable specifications."""
from __future__ import annotations

import io
from pathlib import Path
import shlex

import numpy as np


_PROJECT = """# Start here: all paths are relative to this file.
# The bundled hardware values are SYNTHETIC, not measured device specifications.
name: My neural hardware project
model: model.yaml
hardware: hardware.yaml
design: design.yaml
"""

_MODEL = """# Network structure and learned parameters are separate.
# Inspect available tensor names with: npp inspect-weights weights.npz
name: Small dense network
weights: weights.npz

# out_in: each dense matrix has shape [output_size, input_size] (PyTorch).
# Use in_out for Keras dense kernels; this transposes even square matrices.
layout: out_in

input:
  name: x
  size: 2
  bounds: [-1, 1]  # The same lower/upper bound applies to every input coordinate.

# Layers form a sequential chain. Widths are inferred from the saved matrices.
# Weights-only checkpoints do not specify activations: list those explicitly.
layers:
  - name: hidden
    type: linear
    weights: hidden.weight
    bias: hidden.bias
  - name: activation
    type: relu
  - name: readout
    type: linear
    weights: readout.weight
    bias: readout.bias
"""

_HARDWARE = """# SYNTHETIC example library for testing the workflow.
# Replace it with characterized hardware values before interpreting predictions.
extends: example

# Override only what changes. Component IDs come from the selected library.
# Numbers use canonical units; strings can include a supported unit explicitly.
overrides:
  components:
    optical_fixed:
      latency: 0.5 ns

# Example alternatives (uncomment inside the component entry above):
#     energy: 0.2 pJ
#     area: 8 um2
"""

_DESIGN = """# Minimize these metrics together: the planner returns Pareto alternatives.
minimize: [energy, latency, error]

# Hard limits are separate from optimization objectives. Omit unwanted limits.
# Error is an RMS bound in normalized output units, not a classification score.
limits:
  error: 0.1
  latency: 20 ns

# true requires every linear layer to have editable weights.
# A list such as [hidden] requires editability only on those named layers.
# false imposes no editability requirement; it does not forbid editable macros.
editable: false
domains: [digital, optical]
serialization: false

# Exhaustive is suitable for this small example. For larger choice spaces use
# search: {mode: beam, max_evaluations: 10000, beam_width: 64}
search: exhaustive
seed: 7
"""


def create_project(directory: str | Path, *, overwrite: bool = False) -> dict:
    """Write a complete example without silently replacing existing files.

    Every planned path is checked before any content is written. Unrelated files
    are retained, including when ``overwrite`` is requested. The template's
    synthetic learned weights and synthetic hardware are demonstrations only.
    """
    root = Path(directory).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"Project destination is not a directory: {root}")

    payloads: dict[str, bytes] = {
        "project.yaml": _PROJECT.encode("utf-8"),
        "model.yaml": _MODEL.encode("utf-8"),
        "hardware.yaml": _HARDWARE.encode("utf-8"),
        "design.yaml": _DESIGN.encode("utf-8"),
    }
    weights = io.BytesIO()
    np.savez(weights, **{
        "hidden.weight": np.array([[0.5, -0.2], [0.1, 0.4], [-0.3, 0.2]], dtype=np.float64),
        "hidden.bias": np.array([0.1, 0.0, -0.05], dtype=np.float64),
        "readout.weight": np.array([[0.6, -0.4, 0.2]], dtype=np.float64),
        "readout.bias": np.array([0.01], dtype=np.float64),
    })
    payloads["weights.npz"] = weights.getvalue()

    targets = [root / name for name in payloads]
    # Treat dangling symlinks as existing entries and never follow any target
    # symlink when writing, even if overwrite was explicitly requested.
    invalid = [path for path in targets if path.is_symlink() or (path.exists() and not path.is_file())]
    if invalid:
        raise ValueError("Project targets must be regular files: " + ", ".join(str(p) for p in invalid))
    collisions = [path for path in targets if path.exists()]
    if collisions and not overwrite:
        raise ValueError("Project files already exist: " + ", ".join(str(p) for p in collisions)
                         + ". Choose another directory or use --overwrite.")

    root.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        (root / name).write_bytes(content)

    quote = lambda value: shlex.quote(str(value))
    project = quote(root / "project.yaml")
    return {
        "status": "ok",
        "project_directory": str(root),
        "project": str(root / "project.yaml"),
        "written_files": [str(path) for path in targets],
        "commands": {
            "inspect_weights": f"npp inspect-weights {quote(root / 'weights.npz')}",
            "validate": f"npp validate --project {project}",
            "normalize": f"npp normalize --project {project} --out-dir {quote(root / 'normalized')}",
            "plan": f"npp plan --project {project} --out-dir {quote(root / 'runs' / 'first')} --samples 10000",
        },
    }
