"""Human-friendly specifications lowered to the strict planner contracts.

JSON and YAML are accepted; paths are relative to the file declaring them.
This module only expands explicit syntax and known presets. It does not guess
network architecture, repair misspelled keys or invent hardware measurements.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, DecimalException, Underflow, localcontext
from importlib.resources import files
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import yaml

from .models import InputValidationError, validate_inputs


_METRICS = {"energy": "energy_pj", "latency": "latency_ns", "area": "area_um2", "error": "error_rms_bound"}
_HARDWARE_ALIASES = {"area": "area_um2", "latency": "latency_ns", "energy": "energy_pj", "energy_per_mac": "energy_per_mac_pj"}
_UNITS = {
    "latency_ns": {"ps": 0.001, "ns": 1.0, "us": 1000.0, "ms": 1e6, "s": 1e9},
    "energy_pj": {"fJ": 0.001, "pJ": 1.0, "nJ": 1000.0, "uJ": 1e6, "mJ": 1e9, "J": 1e12},
    "area_um2": {"nm2": 1e-6, "um2": 1.0, "mm2": 1e6, "cm2": 1e8, "m2": 1e12},
    "wavelength_nm": {"nm": 1.0, "um": 1000.0, "mm": 1e6, "m": 1e9},
    "error_rms_bound": {},
}
_QUANTITY = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\s]*)$")
# Bound values introduced by repeated YAML aliases: a tiny source can encode
# an exponentially large tree. Plain JSON/YAML data retains its existing limits.
MAX_SPEC_ALIAS_VALUES = 1_000_000


def _fail(path: str, message: str, code: str = "config_error") -> None:
    raise InputValidationError([{"code": code, "path": path, "message": message}])


def _mapping(value: Any, path: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _fail(path, "expected an object with string keys")
    return value


def _keys(value: dict, allowed: set[str], path: str) -> None:
    extra = sorted(set(value) - allowed)
    if extra:
        _fail(path, f"unknown field(s): {', '.join(extra)}", "unknown_field")


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _yaml_mapping(loader: _UniqueSafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    # Explicit merge/override behavior belongs to our hardware syntax. YAML merge
    # keys would hide duplicate keys after parser-dependent flattening.
    result = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise yaml.constructor.ConstructorError(None, None, "YAML merge keys are unsupported; use hardware defaults/overrides", key_node.start_mark)
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise yaml.constructor.ConstructorError(None, None, "mapping keys must be strings (quote names such as 'on' or 'yes')", key_node.start_mark)
        if key in result:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping)


def read_spec(path: str | Path) -> dict:
    """Read a JSON/YAML object with duplicate-key and nonfinite-value checks."""
    path = Path(path)
    def unique(pairs):
        value = {}
        for key, entry in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = entry
        return value

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON numeric constant {value!r}")

    try:
        raw = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".json":
            result = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
        elif path.suffix.lower() in (".yaml", ".yml"):
            result = yaml.load(raw, Loader=_UniqueSafeLoader)
        else:
            _fail(str(path), "specification extension must be .json, .yaml or .yml", "unsupported_spec_format")
        _mapping(result, str(path))
        _check_plain_tree(result, str(path), set(), {}, [0])
        return result
    except InputValidationError:
        raise
    except (OSError, UnicodeError, ValueError, yaml.YAMLError, RecursionError) as exc:
        _fail(str(path), str(exc), "spec_load_error")


def _check_plain_tree(value: Any, path: str, active: set[int], memo: dict[int, int], alias_values: list[int]) -> int:
    count = 1
    if isinstance(value, (dict, list)):
        if id(value) in active:
            _fail(path, "recursive YAML aliases are unsupported")
        if id(value) in memo:
            alias_values[0] += memo[id(value)]
            if alias_values[0] > MAX_SPEC_ALIAS_VALUES:
                _fail(path, f"YAML aliases introduce more than {MAX_SPEC_ALIAS_VALUES} expanded values; reduce repeated aliases", "spec_resource_limit")
            return memo[id(value)]
        active.add(id(value))
        if isinstance(value, dict):
            _mapping(value, path)
            items = ((f"{path}.{key}", child) for key, child in value.items())
        else:
            items = ((f"{path}[{index}]", child) for index, child in enumerate(value))
        for child_path, child in items:
            count += _check_plain_tree(child, child_path, active, memo, alias_values)
        active.remove(id(value))
        memo[id(value)] = count
    elif isinstance(value, float) and not math.isfinite(value):
        _fail(path, "values must be finite", "nonfinite_value")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        _fail(path, "unsupported YAML value; use JSON-compatible scalars, lists and objects")
    return count


def _quantity(value: Any, field: str, path: str) -> float:
    dimension = "energy_pj" if field == "energy_per_mac_pj" else field
    if isinstance(value, bool):
        _fail(path, "expected a finite number or a number with a unit", "invalid_quantity")
    if isinstance(value, (int, float)):
        try:
            result = float(value)
        except OverflowError:
            _fail(path, "quantity is outside the supported numeric range", "invalid_quantity")
    elif isinstance(value, str):
        match = _QUANTITY.fullmatch(value.strip())
        if not match:
            _fail(path, "expected a number such as 2.5 or a quantity such as '2.5 ns'", "invalid_quantity")
        number, unit = match.groups()
        unit = unit.replace("µ", "u").replace("μ", "u").replace("²", "2").replace("^2", "2")
        units = _UNITS[dimension]
        if unit and unit not in units:
            accepted = ", ".join(units) or "no unit (normalized RMS error)"
            _fail(path, f"invalid unit {unit!r} for {field}; expected {accepted}", "unit_mismatch")
        # Round only once, after decimal unit conversion, so e.g. 9 fJ and
        # 0.009 pJ yield identical canonical inputs and replay hashes.
        try:
            with localcontext() as context:
                context.prec = max(28, len(number) + 16)
                context.traps[Underflow] = True
                scaled = Decimal(number) * Decimal(str(units.get(unit, 1.0)))
                result = float(scaled)
                if scaled != 0 and result == 0:
                    _fail(path, "nonzero quantity is too small for the supported numeric range", "invalid_quantity")
        except (ValueError, OverflowError, DecimalException):
            _fail(path, "quantity is outside the supported numeric range", "invalid_quantity")
    else:
        _fail(path, "expected a finite number or a number with a unit", "invalid_quantity")
    if not math.isfinite(result):
        _fail(path, "quantity must be finite", "invalid_quantity")
    return result


def _alias_fields(value: dict, aliases: dict[str, str], path: str) -> dict:
    result = deepcopy(_mapping(value, path))
    for short, canonical in aliases.items():
        if short in result:
            if canonical in result:
                _fail(path, f"use either {short!r} or {canonical!r}, not both", "conflicting_fields")
            result[canonical] = result.pop(short)
    return result


def _hardware_fields(value: dict, path: str) -> dict:
    result = _alias_fields(value, _HARDWARE_ALIASES, path)
    for field in (*_UNITS, "energy_per_mac_pj"):
        if field in result:
            result[field] = _quantity(result[field], field, f"{path}.{field}")
    return result


def _resolve(base: Path, value: Any, path: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(path, "file must be a nonempty path string")
    target = Path(value).expanduser()
    return target if target.is_absolute() else base / target


def _bounds(value: Any, size: Any, path: str) -> dict:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        _fail(path, "input size must be a positive integer")
    if isinstance(value, list) and len(value) == 2:
        low, high = value
    elif isinstance(value, dict):
        _keys(value, {"lower", "upper"}, path)
        if not {"lower", "upper"} <= value.keys():
            _fail(path, "bounds require lower and upper")
        low, high = value["lower"], value["upper"]
    else:
        _fail(path, "bounds must be [lower, upper] or {lower: ..., upper: ...}; scalar endpoints apply to every input")
    result = {}
    for key, endpoint in (("lower", low), ("upper", high)):
        array = endpoint if isinstance(endpoint, list) else [endpoint] * size
        if len(array) != size:
            _fail(f"{path}.{key}", f"expected {size} entries, got {len(array)}", "shape_mismatch")
        result[key] = [_quantity(item, "error_rms_bound", f"{path}.{key}[{i}]") for i, item in enumerate(array)]
    return result


class _TensorResolver:
    def __init__(self, base: Path, source: Any, layout: str, path: str):
        self.base, self.source, self.path = base, source, path
        self.layout = self._layout(layout, f"{path}.layout")
        self.cache: dict[tuple[str, str | None], dict] = {}
        if source is not None:
            self._source(source, f"{path}.weights")

    @staticmethod
    def _layout(value: Any, path: str) -> str:
        if value not in ("out_in", "in_out"):
            _fail(path, "layout must be 'out_in' or 'in_out'")
        return value

    def _source(self, source: Any, path: str) -> tuple[Path, str | None]:
        if isinstance(source, str):
            return _resolve(self.base, source, path), None
        _mapping(source, path)
        _keys(source, {"file", "state_dict_key"}, path)
        if "file" not in source:
            _fail(path, "weights source requires file")
        key = source.get("state_dict_key")
        if key is not None and (not isinstance(key, str) or not key):
            _fail(f"{path}.state_dict_key", "expected a nonempty checkpoint dictionary key")
        return _resolve(self.base, source["file"], f"{path}.file"), key

    def _load(self, source: Any, path: str) -> dict:
        filename, state_key = self._source(source, path)
        cache_key = (str(filename.resolve()), state_key)
        if cache_key not in self.cache:
            from .weights import load_weights
            self.cache[cache_key] = load_weights(filename, state_dict_key=state_key)
        return self.cache[cache_key]

    def conventional_bias(self, reference: Any, node_id: str, path: str) -> list | None:
        """Look only for the conventional bias in the same declared source."""
        source = self.source
        tensor_key = reference
        if isinstance(reference, dict):
            tensor_key = reference.get("tensor")
            if "file" in reference:
                source = {key: reference[key] for key in ("file", "state_dict_key") if key in reference}
        key = tensor_key[:-7] + ".bias" if isinstance(tensor_key, str) and tensor_key.endswith(".weight") else f"{node_id}.bias"
        if source is None or key not in self._load(source, path):
            return None
        result = {"tensor": key}
        if isinstance(source, str):
            result["file"] = source
        else:
            result.update(source)
        return self.tensor(result, path, matrix=False)

    def tensor(self, value: Any, path: str, *, matrix: bool) -> list:
        layout = self.layout if matrix else "out_in"
        if isinstance(value, dict):
            _keys(value, {"file", "tensor", "state_dict_key", "layout"} if matrix else {"file", "tensor", "state_dict_key"}, path)
            if "tensor" not in value:
                _fail(path, "tensor reference requires a tensor key")
            if "file" in value:
                source = {key: value[key] for key in ("file", "state_dict_key") if key in value}
            else:
                if "state_dict_key" in value:
                    _fail(path, "state_dict_key requires an explicit file")
                source = self.source
            if matrix:
                layout = self._layout(value.get("layout", self.layout), f"{path}.layout")
            value = self._get(value["tensor"], source, path)
        elif isinstance(value, str):
            value = self._get(value, self.source, path)
        # NumPy promotes [True, 2] to an integer array. Reject boolean leaves
        # before that coercion can conceal a malformed inline weight or bias.
        pending = [value] if isinstance(value, (list, tuple)) else []
        visited = set()
        while pending:
            entry = pending.pop()
            if isinstance(entry, (list, tuple)):
                if id(entry) not in visited:
                    visited.add(id(entry))
                    pending.extend(entry)
            elif isinstance(entry, (bool, np.bool_)):
                _fail(path, "tensor must contain finite real numbers, not booleans", "invalid_tensor")
        try:
            array = np.asarray(value)
        except (ValueError, TypeError) as exc:
            _fail(path, f"expected a rectangular numeric tensor: {exc}", "invalid_tensor")
        expected = 2 if matrix else 1
        if array.ndim != expected or not array.size:
            _fail(path, f"expected a nonempty rank-{expected} tensor; got shape {array.shape}", "shape_mismatch")
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            _fail(path, "tensor must contain finite real numbers", "invalid_tensor")
        if matrix and layout == "in_out":
            array = array.T
        return array.astype(float).tolist()

    def _get(self, tensor: Any, source: Any, path: str) -> Any:
        if not isinstance(tensor, str) or not tensor:
            _fail(path, "tensor key must be a nonempty string")
        if source is None:
            _fail(path, "tensor references require a top-level weights file or {file: ..., tensor: ...}")
        tensors = self._load(source, path)
        if tensor not in tensors:
            available = ", ".join(sorted(tensors)[:12])
            _fail(path, f"unknown tensor {tensor!r}; available keys: {available}", "unknown_tensor")
        return tensors[tensor]


def _network(value: dict, base: Path, name: str, path: str) -> dict:
    value = deepcopy(_mapping(value, path))
    resolver = _TensorResolver(base, value.pop("weights", None), value.pop("layout", "out_in"), path)
    value.setdefault("name", name)
    if "nodes" in value:
        nodes = value["nodes"]
        if not isinstance(nodes, list):
            _fail(f"{path}.nodes", "expected a list")
        for index, node in enumerate(nodes):
            node_path = f"{path}.nodes[{index}]"
            _mapping(node, node_path)
            for field in ("weights", "bias"):
                if field in node and node[field] is not None:
                    node[field] = resolver.tensor(node[field], f"{node_path}.{field}", matrix=field == "weights")
        return value

    _keys(value, {"schema_version", "name", "input", "layers"}, path)
    if "input" not in value or "layers" not in value:
        _fail(path, "model requires either canonical nodes/outputs or friendly input/layers")
    inp = _mapping(value["input"], f"{path}.input")
    _keys(inp, {"name", "size", "bounds"}, f"{path}.input")
    if not {"size", "bounds"} <= inp.keys():
        _fail(f"{path}.input", "input requires size and bounds")
    size = inp["size"]
    prior = inp.get("name", "x")
    nodes = [{"id": prior, "op": "input", "size": size, "input_bounds": _bounds(inp["bounds"], size, f"{path}.input.bounds")}]
    layers = value["layers"]
    if not isinstance(layers, list) or not layers:
        _fail(f"{path}.layers", "expected at least one sequential layer")
    for index, layer in enumerate(layers):
        layer_path = f"{path}.layers[{index}]"
        _mapping(layer, layer_path)
        _keys(layer, {"name", "type", "weights", "bias", "layout"}, layer_path)
        op = layer.get("type")
        node_id = layer.get("name", f"layer_{index + 1}")
        if op not in ("linear", "relu", "identity"):
            _fail(f"{layer_path}.type", "sequential layers support linear, relu and identity; use canonical nodes for branching", "unsupported_operation")
        node = {"id": node_id, "op": op, "inputs": [prior], "size": size}
        if op == "linear":
            reference = layer.get("weights", f"{node_id}.weight")
            if "layout" in layer:
                local = _TensorResolver(base, resolver.source, layer["layout"], layer_path)
                local.cache = resolver.cache
            else:
                local = resolver
            matrix = local.tensor(reference, f"{layer_path}.weights", matrix=True)
            if len(matrix[0]) != size:
                _fail(f"{layer_path}.weights", f"weight input dimension is {len(matrix[0])}, but previous layer has size {size}; check tensor and layout", "shape_mismatch")
            size = len(matrix)
            node.update(size=size, weights=matrix)
            if "bias" in layer:
                if layer["bias"] is not None:
                    node["bias"] = resolver.tensor(layer["bias"], f"{layer_path}.bias", matrix=False)
            else:
                bias = resolver.conventional_bias(reference, node_id, f"{layer_path}.bias")
                if bias is not None:
                    node["bias"] = bias
        elif any(key in layer for key in ("weights", "bias", "layout")):
            _fail(layer_path, "weights, bias and layout are only valid on linear layers")
        nodes.append(node)
        prior = node_id
    return {"schema_version": value.get("schema_version", "0.1"), "name": value["name"], "nodes": nodes, "outputs": [prior]}


def _load_network(value: Any, base: Path, name: str, path: str) -> dict:
    if isinstance(value, str):
        value = {"file": value}
    if isinstance(value, dict) and "file" in value:
        _keys(value, {"file", "name", "input_bounds"}, path)
        filename = _resolve(base, value["file"], f"{path}.file")
        name = value.get("name", name)
        if filename.suffix.lower() == ".onnx":
            if "input_bounds" not in value:
                _fail(path, "ONNX models require explicit input_bounds, for example [-1, 1]")
            from .onnx_import import import_onnx
            return import_onnx(filename, input_bounds=value["input_bounds"], name=name)
        if filename.suffix.lower() in (".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".npz", ".npy", ".h5", ".hdf5", ".keras"):
            _fail(path, "saved weights do not define a supported computation graph by themselves; describe input/layers in a model YAML/JSON and set its weights field to this file, or import a supported ONNX model", "architecture_required")
        if "input_bounds" in value:
            _fail(path, "input_bounds beside file is only supported for ONNX; specify bounds inside a JSON/YAML model")
        return _network(read_spec(filename), filename.parent, name, str(filename))
    return _network(value, base, name, path)


def _entries(value: Any, defaults: dict, path: str) -> list[dict]:
    if isinstance(value, dict):
        entries = []
        for key, entry in value.items():
            entry = deepcopy(_mapping(entry, f"{path}.{key}"))
            if "id" in entry and entry["id"] != key:
                _fail(f"{path}.{key}.id", "entry id must match its mapping key")
            entry["id"] = key
            entries.append(entry)
    elif isinstance(value, list):
        entries = value
    else:
        _fail(path, "expected a list of entries or an object keyed by entry id")
    return [{**defaults, **_hardware_fields(entry, f"{path}[{index}]")} for index, entry in enumerate(entries)]


def _library(value: Any, base: Path, path: str) -> dict:
    if isinstance(value, str):
        if value == "example":
            return json.loads(files("npp").joinpath("data/components.json").read_text(encoding="utf-8"))
        filename = _resolve(base, value, path)
        return _library(read_spec(filename), filename.parent, str(filename))
    value = deepcopy(_mapping(value, path))
    if "extends" in value:
        _keys(value, {"extends", "overrides", "name", "provenance"}, path)
        if value["extends"] != "example":
            _fail(f"{path}.extends", "the only bundled hardware preset is 'example' (synthetic parameters)")
        result = _library("example", base, path)
        overrides = _mapping(value.get("overrides", {}), f"{path}.overrides")
        _keys(overrides, {"components", "rules", "conversions"}, f"{path}.overrides")
        for group, patches in overrides.items():
            existing = {entry["id"]: entry for entry in result[group]}
            for key, patch in _mapping(patches, f"{path}.overrides.{group}").items():
                location = f"{path}.overrides.{group}.{key}"
                if key not in existing:
                    _fail(location, f"unknown {group} id {key!r}", "unknown_hardware_id")
                patch = _hardware_fields(patch, location)
                if "id" in patch and patch["id"] != key:
                    _fail(f"{location}.id", "an override cannot rename an entry")
                existing[key].update(patch)
        result["name"] = value.get("name", "Customized synthetic demonstration library")
        if overrides:
            result["provenance"] += " User-provided overrides have been applied."
        if "provenance" in value:
            if not isinstance(value["provenance"], str):
                _fail(f"{path}.provenance", "expected a string")
            result["provenance"] += " User provenance: " + value["provenance"]
        return result

    defaults = _mapping(value.pop("defaults", {}), f"{path}.defaults")
    _keys(defaults, {"components", "rules", "conversions"}, f"{path}.defaults")
    for group in ("components", "rules", "conversions"):
        group_defaults = _hardware_fields(defaults.get(group, {}), f"{path}.defaults.{group}")
        if "id" in group_defaults:
            _fail(f"{path}.defaults.{group}.id", "ids cannot be supplied as defaults")
        if group in value:
            value[group] = _entries(value[group], group_defaults, f"{path}.{group}")
    return value


def _request(value: Any, base: Path, name: str, network: dict, path: str) -> dict:
    if isinstance(value, str):
        filename = _resolve(base, value, path)
        return _request(read_spec(filename), filename.parent, name, network, str(filename))
    result = _alias_fields(value, {"minimize": "objectives", "limits": "budgets", "editable": "required_editable", "domains": "allowed_domains", "serialization": "allow_serialization"}, path)
    result.setdefault("name", name + " design")
    if "minimize" in value:
        if not isinstance(result["objectives"], list):
            _fail(f"{path}.minimize", "expected a list such as [energy, latency, error]")
        mapped = []
        for index, field in enumerate(result["objectives"]):
            if not isinstance(field, str):
                _fail(f"{path}.minimize[{index}]", "expected an objective name")
            mapped.append(_METRICS.get(field, field))
        result["objectives"] = mapped
    if "budgets" in result:
        budget = _alias_fields(result["budgets"], _METRICS if "limits" in value else {}, f"{path}.budgets")
        for field in budget:
            if field in _UNITS and budget[field] is not None:
                budget[field] = _quantity(budget[field], field, f"{path}.budgets.{field}")
        result["budgets"] = budget
    if "editable" in value and isinstance(result["required_editable"], bool):
        result["required_editable"] = [node.get("id") for node in network["nodes"] if node.get("op") == "linear"] if result["required_editable"] else []
    if isinstance(result.get("search"), str):
        result["search"] = {"mode": result["search"]}
    return result


def load_inputs(*, network_path: str | Path | None = None, library_path: str | Path | None = None,
                request_path: str | Path | None = None, project_path: str | Path | None = None) -> tuple[dict, dict, dict]:
    """Load a project or three separate specifications and validate the result.

    A project contains model, hardware and optional design. A saved-weight file
    contains tensors only, so its architecture must be described in model.layers
    (or canonical nodes); an ONNX file supplies its own supported architecture.
    """
    if project_path is not None:
        if any(value is not None for value in (network_path, library_path, request_path)):
            _fail("inputs", "use project_path or three separate specification paths, not both", "conflicting_inputs")
        project_path = Path(project_path)
        project = read_spec(project_path)
        _keys(project, {"schema_version", "name", "model", "hardware", "design"}, str(project_path))
        if project.get("schema_version", "0.1") != "0.1":
            _fail(f"{project_path}.schema_version", "supported project schema_version is '0.1'")
        if not {"model", "hardware"} <= project.keys():
            _fail(str(project_path), "project requires model and hardware; use hardware: example for explicitly synthetic demonstration data")
        name = project.get("name", project_path.stem)
        if not isinstance(name, str) or not name:
            _fail(f"{project_path}.name", "expected a nonempty string")
        base = project_path.parent
        network = _load_network(project["model"], base, name, f"{project_path}.model")
        library = _library(project["hardware"], base, f"{project_path}.hardware")
        request = _request(project.get("design", {}), base, name, network, f"{project_path}.design")
    else:
        if any(value is None for value in (network_path, library_path, request_path)):
            _fail("inputs", "provide project_path or all of network_path, library_path and request_path", "missing_input")
        network_path, library_path, request_path = Path(network_path), Path(library_path), Path(request_path)
        network = _load_network(str(network_path), Path.cwd(), network_path.stem, str(network_path))
        library = _library(str(library_path), Path.cwd(), str(library_path))
        request = _request(str(request_path), Path.cwd(), network_path.stem, network, str(request_path))
    return validate_inputs(network, library, request)
