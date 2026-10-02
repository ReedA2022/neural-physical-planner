"""Reproduce small synthetic comparisons; run from any working directory."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from npp.cli import compile_run
from npp.models import load_json, validate_inputs


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out-dir", default="runs/experiments")
    cli.add_argument("--samples", type=int, default=5000)
    args = cli.parse_args()
    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        cli.error("Choose a new or empty output directory")
    if args.samples < 2:
        cli.error("--samples must be >= 2")
    out.mkdir(parents=True, exist_ok=True)
    base = load_json(ROOT / "examples/split_recombine.json")
    library = load_json(ROOT / "examples/components.json")
    request = load_json(ROOT / "examples/request_default.json")
    cases = []
    cases.append(("full", base, library, request))
    digital = deepcopy(request)
    digital["allowed_domains"] = ["digital"]
    cases.append(("digital_only", base, library, digital))
    passive = deepcopy(library)
    passive["rules"] = [r for r in passive["rules"] if r["kind"] in ("digital_copy", "passive_split")]
    cases.append(("passive_only", base, passive, request))
    cases.append(("editable", base, library, load_json(ROOT / "examples/request_editable.json")))
    beam = deepcopy(request)
    beam["search"] = {"mode": "beam", "max_evaluations": 24, "beam_width": 24}
    cases.append(("beam_24", base, library, beam))
    cases.append(("relu", load_json(ROOT / "examples/small_relu.json"), library, request))
    cases.append(("impossible", base, library, load_json(ROOT / "examples/request_impossible.json")))
    summary = []
    for name, network, lib, req in cases:
        n, l, r = validate_inputs(network, lib, req)
        result = compile_run(n, l, r, out / name, args.samples)
        result["case"] = name
        summary.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
    (out / "summary.json").write_text(json.dumps({
        "interpretation": "Synthetic model comparison; changes across libraries/domains are not measured device improvements.",
        "results": summary}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
