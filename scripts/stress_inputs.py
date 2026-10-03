"""Bounded seeded input/import campaign with independent numerical expectations.

Run with core dependencies or with optional import extras. Missing extras are
reported as skipped scenarios, never as successful format validation. No
unrestricted pickle loader or saved model constructor is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from decimal import Decimal
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import zipfile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npp.config import _quantity, load_inputs, read_spec
from npp.models import InputValidationError
from npp.planner import input_hash
from npp.weights import load_weights

FAMILIES = ("unit_equivalence", "project_equivalence", "numpy_roundtrip",
            "malformed_spec", "invalid_weights", "optional_weights",
            "onnx_semantics", "invalid_project")


class MissingOptional(Exception):
    pass


def _require(module):
    if importlib.util.find_spec(module) is None:
        raise MissingOptional(module)


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _reject(call):
    try:
        call()
    except InputValidationError as exc:
        _assert(bool(exc.diagnostics), "rejection lacked structured diagnostics")
        return
    raise AssertionError("invalid input was accepted")


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(value, allow_unicode=True) if path.suffix == ".yaml" else json.dumps(value)
    path.write_text(text, encoding="utf-8")
    return path


def _project(matrix, bias):
    return {"name": "Seeded input – unités", "hardware": "example", "model": {
        "input": {"name": "x", "size": len(matrix[0]), "bounds": [-2, 2]},
        "layers": [{"name": "affine", "type": "linear", "weights": matrix, "bias": bias},
                   {"name": "activation", "type": "relu"}]},
        "design": {"minimize": ["energy", "latency"], "limits": {"energy": "9 fJ"}}}


def _scenario(family, rng, root, detail, variant):
    if family == "unit_equivalence":
        units = [("energy_pj", "fJ", "pJ", Decimal(".001")),
                 ("latency_ns", "ps", "ns", Decimal(".001")),
                 ("area_um2", "nm²", "um2", Decimal(".000001")),
                 ("wavelength_nm", "µm", "nm", Decimal("1000")),
                 ("energy_per_mac_pj", "nJ", "pJ", Decimal("1000"))]
        field, source, target, factor = units[variant % len(units)]
        number = Decimal(rng.randint(1, 100_000)) / Decimal(100)
        left, right = f"{number} {source}", f"{number * factor} {target}"
        detail.update(field=field, left=left, right=right)
        a, b = _quantity(left, field, "left"), _quantity(right, field, "right")
        _assert(a == b == float(number * factor), f"unit conversion differed: {a!r}, {b!r}")
        return

    width, height = rng.randint(1, 6), rng.randint(1, 6)
    matrix = np.array([[rng.randint(-16, 16) / 8 for _ in range(width)] for _ in range(height)])
    bias = np.array([rng.randint(-8, 8) / 8 for _ in range(height)])
    detail.update(shape=[height, width], variant=variant)

    if family == "project_equivalence":
        project = _project(matrix.tolist(), bias.tolist())
        expected = load_inputs(project_path=_write(root / "inline.json", project))
        alternate = deepcopy(project)
        model = alternate.pop("model")
        model["weights"] = "parameters.npz"
        transpose = variant % 2 == 0
        model["layout"] = "in_out" if transpose else "out_in"
        model["layers"][0].update(weights="affine.weight", bias="affine.bias")
        model["input"]["bounds"] = {"lower": [-2] * width, "upper": [2] * width}
        _write(root / "nested" / "model.yaml", model)
        np.savez(root / "nested" / "parameters.npz", **{
            "affine.weight": matrix.T if transpose else matrix, "affine.bias": bias})
        alternate["model"] = "nested/model.yaml"
        alternate["design"]["limits"]["energy"] = ".009 pJ"
        actual = load_inputs(project_path=_write(root / "project.yaml", alternate))
        _assert(expected == actual, "JSON/inline and YAML/relative-weight normalization differ")
        _assert(input_hash(expected) == input_hash(actual), "equivalent inputs have different hashes")
        return

    if family == "numpy_roundtrip":
        dtype = (np.float16, np.float32, np.float64, np.int16, np.uint16)[variant % 5]
        array = np.asfortranarray(matrix.astype(dtype))
        kind = variant % 3
        path = root / ("tensor.npy" if kind == 0 else "tensor.npz")
        if kind == 0:
            np.save(path, array)
            wanted = {"array": array}
        else:
            wanted = {"weight": array, "weight.npy": bias, "nested/µ.bias": -bias}
            (np.savez if kind == 1 else np.savez_compressed)(path, **wanted)
        actual = load_weights(path)
        _assert(set(actual) == set(wanted), "tensor names changed")
        for key, value in wanted.items():
            np.testing.assert_array_equal(actual[key], value)
            _assert(actual[key].dtype == value.dtype, "tensor dtype changed")
        return

    if family == "malformed_spec":
        texts = [('.json', '{"x": 1, "x": 2}'), ('.json', '{"x": 1e999}'),
                 ('.json', '{"x": NaN}'), ('.json', '[1, 2]'),
                 ('.yaml', 'x: .inf\n'), ('.yaml', 'x: &x [*x]\n'),
                 ('.yaml', 'x: !!python/object/apply:builtins.str [bad]\n'),
                 ('.yaml', 'x: 2026-10-03\n'), ('.yaml', '1: value\n'),
                 ('.yaml', 'x: &x {a: 1}\ny: {<<: *x}\n'),
                 ('.yaml', 'a0: &a0 [0]\n' + ''.join(
                     f'a{i}: &a{i} [*a{i-1}, *a{i-1}]\n' for i in range(1, 23))),
                 ('.json', '{"x": [1,}')]
        suffix, text = texts[variant % len(texts)]
        detail["input_text"] = text
        path = root / ("invalid" + suffix)
        path.write_text(text)
        _reject(lambda: read_spec(path))
        return

    if family == "invalid_weights":
        invalids = [np.array([np.nan]), np.array([np.inf]), np.array([1j]),
                    np.array([True]), np.array(["data"]), np.array([{}], dtype=object)]
        choice = variant % 9
        path = root / "invalid.npy"
        if choice < 6:
            np.save(path, invalids[choice])
        elif choice == 6:
            path.write_bytes(rng.randbytes(rng.randint(1, 90)))
        elif choice == 7:
            stream = io.BytesIO()
            np.save(stream, matrix)
            path.write_bytes(stream.getvalue()[:-1])
        else:
            path = root / "invalid.npz"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("unexpected.txt", "not a tensor")
        _reject(lambda: load_weights(path))
        return

    if family == "optional_weights":
        fmt = ("safetensors", "torch", "h5py", "keras_archive")[variant % 4]
        detail["format"] = fmt
        key = "dense.weight"
        if fmt == "safetensors":
            _require("safetensors")
            from safetensors.numpy import save_file
            path = root / "weights.safetensors"
            save_file({key: matrix}, str(path))
        elif fmt == "torch":
            _require("torch")
            import torch
            path = root / "weights.pt"
            state = {key: torch.tensor(matrix)}
            torch.save({"model_state_dict": state, "epoch": 3}, path)
        else:
            _require("h5py")
            import h5py
            path = root / "weights.h5"
            with h5py.File(path, "w") as archive:
                archive.create_dataset(key, data=matrix)
            if fmt == "keras_archive":
                container = root / "model.keras"
                with zipfile.ZipFile(container, "w") as archive:
                    archive.write(path, "model.weights.h5")
                    archive.writestr("config.json", "No model code is deserialized")
                path = container
        np.testing.assert_array_equal(load_weights(path)[key], matrix)
        return

    if family == "onnx_semantics":
        _require("onnx")
        import onnx
        from onnx.reference import ReferenceEvaluator
        from npp.onnx_import import import_onnx
        h = onnx.helper
        tp = onnx.TensorProto.DOUBLE
        gemm = variant % 3 != 0
        transpose = variant % 2
        weight = matrix if gemm and transpose else matrix.T
        initializers = [onnx.numpy_helper.from_array(weight, name="W"),
                        onnx.numpy_helper.from_array(bias, name="B")]
        if gemm:
            operations = [h.make_node("Gemm", ["x", "W", "B"], ["affine"],
                                      transB=transpose, alpha=0.5, beta=-0.25)]
        else:
            operations = [h.make_node("MatMul", ["x", "W"], ["matmul"]),
                          h.make_node("Add", ["B", "matmul"], ["affine"])]
        operations += [h.make_node("Relu", ["affine"], ["activation"]),
                       h.make_node("Add", ["activation", "activation"], ["y"])]
        graph = h.make_graph(operations, "seeded", [h.make_tensor_value_info("x", tp, [1, width])],
                            [h.make_tensor_value_info("y", tp, [1, height])], initializers)
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
        path = root / "network.onnx"
        onnx.save(model, path)
        network = import_onnx(path, input_bounds=(-2, 2))
        sample = np.array([[rng.randint(-16, 16) / 8 for _ in range(width)]])
        expected = (0.5 * sample @ matrix.T - 0.25 * bias) if gemm else sample @ matrix.T + bias
        expected = 2 * np.maximum(expected, 0)
        reference = ReferenceEvaluator(model).run(None, {"x": sample})[0]
        np.testing.assert_allclose(reference, expected, rtol=1e-13, atol=1e-13)
        values = {}
        for node in network["nodes"]:
            op, inputs = node["op"], [values[k] for k in node["inputs"]]
            if op == "input":
                value = sample
            elif op == "linear":
                value = inputs[0] @ np.asarray(node["weights"]).T + node["bias"]
            elif op == "relu":
                value = np.maximum(inputs[0], 0)
            elif op == "add":
                value = sum(inputs)
            else:
                raise AssertionError(f"unexpected operation {op}")
            values[node["id"]] = value
        np.testing.assert_allclose(values[network["outputs"][0]], reference, rtol=1e-13, atol=1e-13)
        return

    if family == "invalid_project":
        project = _project(matrix.tolist(), bias.tolist())
        mutations = [lambda p: p.update(desgin={}),
                     lambda p: p["model"]["input"].update(bounds=[2, -2]),
                     lambda p: p["model"]["input"].update(size=True),
                     lambda p: p["model"]["layers"][0].update(weights=[[1] * (width + 1)]),
                     lambda p: p["model"]["layers"][0].update(layout="guess"),
                     lambda p: p["design"].update(limits={"energy": "1 ns"}),
                     lambda p: p["design"].update(limits={"energy": True}),
                     lambda p: p["model"]["layers"][1].update(type="sigmoid"),
                     lambda p: p["model"]["layers"][0].update(weights="missing"),
                     lambda p: p["model"]["layers"][0]["weights"][0].__setitem__(0, True),
                     lambda p: p["model"]["layers"][0]["bias"].__setitem__(0, False),
                     lambda p: p["design"].update(limits={"energy": "1e-999 pJ"}),
                     lambda p: p["design"].update(limits={"energy": "-1e-999 pJ"}),
                     lambda p: p["design"].update(limits={"energy": "1e-999999999 pJ"})]
        mutations[variant % len(mutations)](project)
        detail["project"] = project
        _reject(lambda: load_inputs(project_path=_write(root / "bad.yaml", project)))
        return
    raise AssertionError(f"unknown family {family}")


def run_campaign(seed=1703, cases=400):
    """Return one counted scenario per independent fixture, not per assertion."""
    if not __debug__:
        raise ValueError("Stress campaigns require assertions; run Python without -O or -OO.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(cases, bool) or not isinstance(cases, int) or not 1 <= cases <= 5000:
        raise ValueError("cases must be between 1 and 5000")
    start = time.monotonic()
    failures, skipped = [], []
    counts = {family: Counter() for family in FAMILIES}
    chooser = random.Random(seed)
    with tempfile.TemporaryDirectory(prefix="npp-input-stress-") as directory:
        for index in range(cases):
            family = FAMILIES[index % len(FAMILIES)]
            case_seed = chooser.getrandbits(64)
            detail = {"index": index, "family": family, "case_seed": case_seed,
                      "variant": index // len(FAMILIES)}
            root = Path(directory) / f"case-{index}"
            root.mkdir()
            counts[family]["attempted"] += 1
            try:
                _scenario(family, random.Random(case_seed), root, detail, detail["variant"])
            except MissingOptional as exc:
                counts[family]["skipped"] += 1
                skipped.append({**detail, "reason": f"optional dependency missing: {exc}"})
            except Exception as exc:
                counts[family]["failed"] += 1
                failures.append({**detail, "exception": type(exc).__name__, "message": str(exc)})
            else:
                counts[family]["passed"] += 1
    return {"campaign": "input_import_stress_v1", "seed": seed, "scenario_count": cases,
            "passed_count": sum(c["passed"] for c in counts.values()),
            "skipped_count": len(skipped), "failed_count": len(failures),
            "counts_by_family": {k: dict(v) for k, v in counts.items()},
            "failures": failures, "skipped": skipped,
            "elapsed_seconds": time.monotonic() - start,
            "status": "failed" if failures else "passed",
            "scope": "Software input and tensor/ONNX semantics only; no device physics validation."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1703)
    parser.add_argument("--cases", type=int, default=400)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = run_campaign(seed=args.seed, cases=args.cases)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report:
        args.report.write_text(encoded)
    print(encoded, end="")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
