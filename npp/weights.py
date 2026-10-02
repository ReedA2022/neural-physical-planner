"""Read named numeric tensors without constructing or executing saved models.

External formats are optional dependencies.  Loading weights does not recover a
network architecture: the friendly network specification binds tensor names to
linear operations explicitly.  PyTorch's restricted loader is used exclusively;
this is not a sandbox for arbitrary files or a guarantee against parser bugs.
"""
from __future__ import annotations

from collections.abc import Mapping
import importlib
import io
import math
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from .models import InputValidationError


MAX_WEIGHT_BYTES = 512 * 1024 * 1024
MAX_TENSORS = 10_000
_FORMATS = {
    ".npy": "numpy_npy", ".npz": "numpy_npz",
    ".safetensors": "safetensors",
    ".pt": "pytorch", ".pth": "pytorch", ".bin": "pytorch",
    ".h5": "hdf5", ".hdf5": "hdf5", ".keras": "keras",
    ".onnx": "onnx",
}


def _fail(path: Path | str, message: str, code: str = "weights_invalid") -> None:
    raise InputValidationError([{"code": code, "path": str(path), "message": message}])


def _dependency(module: str, extra: str, path: Path):
    try:
        return importlib.import_module(module)
    except ImportError:
        _fail(path, f"This weight format needs an optional dependency. From the project source directory, run "
              f"python -m pip install '.[{extra}]'.", "weights_dependency")


def _check_size(size: int, path: Path | str) -> None:
    if size > MAX_WEIGHT_BYTES:
        _fail(path, f"Weight data exceeds the {MAX_WEIGHT_BYTES // 2**20} MiB import limit. "
              "Export only the tensors needed by this network.", "weights_resource_limit")


def _check_shape(shape, dtype, path: Path | str) -> None:
    dtype = np.dtype(dtype)
    if dtype.kind not in "iuf":
        _fail(path, f"Expected real numeric tensors; dtype {dtype} is unsupported. "
              "Export floating-point or integer arrays; object, boolean, string and complex arrays are not weights.")
    _check_size(math.prod(shape) * dtype.itemsize, path)


def _validate(tensors: Mapping, path: Path) -> dict[str, np.ndarray]:
    if not tensors:
        _fail(path, "No tensors were found in this file.")
    if len(tensors) > MAX_TENSORS:
        _fail(path, f"At most {MAX_TENSORS} tensors can be imported.", "weights_resource_limit")
    result: dict[str, np.ndarray] = {}
    total = 0
    for key, value in tensors.items():
        if not isinstance(key, str) or not key:
            _fail(path, "Every tensor must have a nonempty string name.")
        location = f"{path}:{key}"
        array = np.asarray(value)
        _check_shape(array.shape, array.dtype, location)
        total += array.nbytes
        _check_size(total, path)
        if not np.isfinite(array).all():
            _fail(location, "Tensor contains NaN or infinity; all imported values must be finite.")
        result[key] = array
    return result


def _npy_header(stream, path: Path | str) -> int:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    elif version == (3, 0):
        # V3 differs only in header encoding. Numeric dtype/shape headers are
        # ASCII, a subset of both Latin-1 (the V2 reader) and UTF-8.
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        _fail(path, f"Unsupported NumPy format version {version}.")
    _check_shape(shape, dtype, path)
    return math.prod(shape) * dtype.itemsize


def _load_numpy(path: Path, fmt: str) -> dict[str, np.ndarray]:
    if fmt == "numpy_npy":
        with path.open("rb") as source:
            _npy_header(source, path)
        return {"array": np.load(path, allow_pickle=False)}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_TENSORS:
            _fail(path, f"At most {MAX_TENSORS} tensors can be imported.", "weights_resource_limit")
        if len({entry.filename for entry in entries}) != len(entries):
            _fail(path, "The NumPy archive contains duplicate tensor entries.")
        _check_size(sum(entry.file_size for entry in entries), path)
        total = 0
        for entry in entries:
            if not entry.filename.endswith(".npy") or entry.is_dir():
                _fail(path, "A .npz weights archive must contain only .npy tensor entries.")
            with archive.open(entry) as source:
                total += _npy_header(source, f"{path}:{entry.filename}")
            _check_size(total, path)
    # Read exact members rather than NpzFile.__getitem__: that API treats a
    # key ending in '.npy' as an archive filename first, so keys 'weight' and
    # 'weight.npy' can otherwise silently resolve to the same tensor.
    tensors = {}
    with zipfile.ZipFile(path) as archive:
        for entry in entries:
            with archive.open(entry) as source:
                tensors[entry.filename[:-4]] = np.load(source, allow_pickle=False)
    return tensors


def _load_pytorch(path: Path, state_dict_key: str | None) -> dict[str, np.ndarray]:
    torch = _dependency("torch", "pytorch", path)
    # torch.load may otherwise dispatch TorchScript archives to torch.jit.load.
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if any(entry.filename.split("/")[-1] == "constants.pkl" for entry in entries):
                _fail(path, "TorchScript archives are unsupported. Export model.state_dict() to a .pt file.")
            _check_size(sum(entry.file_size for entry in entries), path)
    try:
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as exc:
        _fail(path, "Restricted PyTorch loading failed. Use a plain tensor state_dict saved with "
              "torch.save(model.state_dict(), path), or export safetensors/NPZ. "
              "Full pickled modules and custom Python objects are unsupported; unsafe loading is never retried. "
              f"Loader error type: {type(exc).__name__}.")
    if not isinstance(checkpoint, Mapping):
        _fail(path, "Expected a tensor state_dict or a checkpoint containing state_dict/model_state_dict.")
    if state_dict_key is not None:
        if state_dict_key not in checkpoint:
            _fail(path, f"Checkpoint has no top-level key {state_dict_key!r}.")
        checkpoint = checkpoint[state_dict_key]
    else:
        wrappers = [key for key in ("state_dict", "model_state_dict")
                    if key in checkpoint and isinstance(checkpoint[key], Mapping)]
        if len(wrappers) > 1:
            _fail(path, "Checkpoint contains both state_dict and model_state_dict. Set state_dict_key explicitly.")
        if wrappers:
            checkpoint = checkpoint[wrappers[0]]
    if not isinstance(checkpoint, Mapping):
        _fail(path, "Selected checkpoint entry must be a mapping from tensor names to tensors.")
    if len(checkpoint) > MAX_TENSORS:
        _fail(path, f"At most {MAX_TENSORS} tensors can be imported.", "weights_resource_limit")
    result = {}
    total = 0
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            _fail(f"{path}:{key}", "State dictionary contains a non-tensor entry. Select the correct state_dict_key.")
        if value.layout != torch.strided or value.is_quantized:
            _fail(f"{path}:{key}", "Sparse and quantized tensors are unsupported; export dense dequantized weights.")
        if value.is_complex() or value.dtype == torch.bool:
            _fail(f"{path}:{key}", "Weights must be real floating-point or integer tensors.")
        tensor = value.detach().cpu()
        total += tensor.numel() * max(tensor.element_size(), 4 if tensor.is_floating_point() else 1)
        _check_size(total, path)
        try:
            array = tensor.numpy()
        except (TypeError, RuntimeError):
            if not tensor.is_floating_point():
                _fail(f"{path}:{key}", f"Tensor dtype {tensor.dtype} cannot be converted to NumPy.")
            # NumPy lacks native support for torch.bfloat16 / float8. Conversion
            # to float32 is exact for their representable finite values.
            array = tensor.to(dtype=torch.float32).numpy()
        result[key] = array
    return result


def _read_hdf5(source: Any, path: Path) -> dict[str, np.ndarray]:
    h5py = _dependency("h5py", "hdf5", path)
    tensors = {}
    total = 0
    visited: set[int] = set()
    with h5py.File(source, "r") as archive:
        def walk(group, prefix=""):
            nonlocal total
            address = h5py.h5o.get_info(group.id).addr
            if address in visited:
                _fail(path, "HDF5 group aliases/cycles are unsupported; export a simple tensor tree.")
            visited.add(address)
            for name in group:
                key = prefix + name
                link = group.get(name, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    _fail(f"{path}:{key}", "HDF5 external and soft links are unsupported. Export self-contained tensor datasets.")
                entry = group[name]
                if isinstance(entry, h5py.Group):
                    walk(entry, key + "/")
                elif isinstance(entry, h5py.Dataset):
                    if entry.is_virtual or entry.external:
                        _fail(f"{path}:{key}", "HDF5 external and virtual datasets are unsupported. Export self-contained tensors.")
                    if entry.shape is None:
                        _fail(f"{path}:{key}", "HDF5 null datasets are not numeric tensors.")
                    _check_shape(entry.shape, entry.dtype, f"{path}:{key}")
                    total += math.prod(entry.shape) * entry.dtype.itemsize
                    _check_size(total, path)
                    if len(tensors) >= MAX_TENSORS:
                        _fail(path, f"At most {MAX_TENSORS} tensors can be imported.", "weights_resource_limit")
                    tensors[key] = entry[()]
        walk(archive)
    return tensors


def _load_keras(path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        entries = [entry for entry in archive.infolist() if entry.filename == "model.weights.h5"]
        if len(entries) != 1:
            _fail(path, "Expected exactly one model.weights.h5 entry in this .keras archive. "
                  "Export Keras weights to a .weights.h5 file if this is a different archive format.")
        _check_size(entries[0].file_size, path)
        # Never extract paths or deserialize the model configuration/custom code.
        with archive.open(entries[0]) as source:
            content = source.read(MAX_WEIGHT_BYTES + 1)
        _check_size(len(content), path)
    return _read_hdf5(io.BytesIO(content), path)


def _load_onnx(path: Path) -> dict[str, np.ndarray]:
    onnx = _dependency("onnx", "onnx", path)
    model = onnx.load_model(str(path), load_external_data=False)
    if model.graph.sparse_initializer:
        _fail(path, "Sparse ONNX initializers are unsupported; export dense weights.")
    if len(model.graph.initializer) > MAX_TENSORS:
        _fail(path, f"At most {MAX_TENSORS} tensors can be imported.", "weights_resource_limit")
    result = {}
    total = 0
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
            _fail(f"{path}:{tensor.name}", "External ONNX tensor files are unsupported here. "
                  "Export a self-contained ONNX file or a separate NPZ/safetensors archive.")
        if tensor.name in result:
            _fail(path, f"Duplicate ONNX initializer name {tensor.name!r}.")
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
        _check_shape(tuple(tensor.dims), dtype, f"{path}:{tensor.name}")
        total += math.prod(tensor.dims) * np.dtype(dtype).itemsize
        _check_size(total, path)
        result[tensor.name] = onnx.numpy_helper.to_array(tensor)
    return result


def load_weights(path: str | Path, *, state_dict_key: str | None = None) -> dict[str, np.ndarray]:
    """Return finite real tensors keyed exactly as stored (``array`` for NPY).

    Only PyTorch checkpoints accept ``state_dict_key``; it names an exact
    top-level dictionary entry.  No graph architecture or custom code is loaded.
    """
    path = Path(path)
    fmt = _FORMATS.get(path.suffix.lower())
    if fmt is None:
        _fail(path, "Unsupported weight extension. Use .npz, .npy, .safetensors, .pt, .pth, .bin, "
              ".h5, .hdf5, .keras or .onnx.", "weights_format")
    if state_dict_key is not None and (not isinstance(state_dict_key, str) or not state_dict_key):
        _fail(path, "state_dict_key must be a nonempty string.")
    if state_dict_key is not None and fmt != "pytorch":
        _fail(path, "state_dict_key is only supported for PyTorch checkpoints.")
    try:
        if not path.is_file():
            _fail(path, "Weight file does not exist or is not a regular file.", "weights_file")
        _check_size(path.stat().st_size, path)
        if fmt in {"numpy_npy", "numpy_npz"}:
            tensors = _load_numpy(path, fmt)
        elif fmt == "safetensors":
            safetensors = _dependency("safetensors.numpy", "safetensors", path)
            tensors = safetensors.load_file(str(path))
        elif fmt == "pytorch":
            tensors = _load_pytorch(path, state_dict_key)
        elif fmt == "hdf5":
            tensors = _read_hdf5(path, path)
        elif fmt == "keras":
            tensors = _load_keras(path)
        else:
            tensors = _load_onnx(path)
        return _validate(tensors, path)
    except InputValidationError:
        raise
    except Exception as exc:
        _fail(path, f"Could not read {fmt} weights: {type(exc).__name__}: {exc}", "weights_read")


def inspect_weights(path: str | Path, state_dict_key: str | None = None) -> dict[str, Any]:
    """Load and validate weights, returning JSON-compatible tensor metadata."""
    tensors = load_weights(path, state_dict_key=state_dict_key)
    fmt = _FORMATS[Path(path).suffix.lower()]
    return {
        "format": fmt,
        "tensors": [{"name": name, "shape": list(array.shape), "dtype": str(array.dtype)}
                    for name, array in sorted(tensors.items())],
        "note": "Tensor names are preserved. Bind them explicitly to network layers; weights alone do not define an architecture."
                + (" The single .npy tensor is named 'array'." if fmt == "numpy_npy" else "")
                + (" All numeric datasets are listed, including any optimizer state; select the model weights explicitly."
                   if fmt in {"hdf5", "keras"} else ""),
    }
