"""Strict, optional ONNX import into the planner's vector computation graph.

Only the explicitly supported inference operators are translated. Batch-one
axes are removed from the resulting vector IR; dynamic batch dimensions mean
one sample per invocation, not a throughput or batching model.
"""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
from pydantic import ValidationError

from .models import InputValidationError, Network


def _fail(code: str, path: str, message: str):
    raise InputValidationError([{"code": code, "path": path, "message": message}])


def _tensor_values(graph):
    """Find tensor attributes as well as initializers without loading payloads."""
    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            if attribute.HasField("g"):
                yield from _tensor_values(attribute.g)
            for child in attribute.graphs:
                yield from _tensor_values(child)


def import_onnx(path, *, input_bounds=(-1.0, 1.0), name=None) -> dict:
    """Import a self-contained, single-input dense ONNX inference model.

    Supported operations: Gemm, MatMul with a constant right operand, Add
    (bias or two equal-shaped dynamic operands), Relu, Identity and Constant.
    Input tensors must have shape ``[features]`` or ``[1, features]``; a
    symbolic/unspecified batch dimension is evaluated as one. Bounds may be
    scalar ``(lower, upper)`` or two per-feature vectors.

    No pickle or framework code is executed. External ONNX tensor files,
    overridable initializer inputs, custom operators, shape transformations
    and unsupported activations are rejected with structured diagnostics.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        _fail("missing_dependency", "onnx", "ONNX import requires the optional 'onnx' package. From the project source directory, run: python -m pip install '.[onnx]'")

    model_path = Path(path)
    try:
        model = onnx.load_model(str(model_path), load_external_data=False)
    except Exception as exc:
        _fail("onnx_read", str(model_path), f"Cannot read ONNX model: {exc}")
    for tensor in _tensor_values(model.graph):
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
            _fail("onnx_external_data", str(model_path), "External ONNX tensor files are unsupported. Export a self-contained ONNX model with embedded weights.")
    if model.graph.sparse_initializer:
        _fail("onnx_sparse_initializer", "graph.initializer", "Sparse ONNX initializers are unsupported; export dense weights.")
    if model.functions:
        _fail("onnx_function", "functions", "Model-local ONNX functions are unsupported; export standard dense operators directly.")
    supported = {"Constant", "Gemm", "MatMul", "Add", "Relu", "Identity"}
    for index, node in enumerate(model.graph.node):
        if node.domain not in ("", "ai.onnx"):
            _fail("onnx_domain", f"graph.node[{index}]", f"Custom operator domain {node.domain!r} is unsupported ({node.op_type}).")
        if node.op_type not in supported:
            _fail("onnx_operator", f"graph.node[{index}]", f"Unsupported ONNX operator {node.op_type!r} at {node.name or index!r}. Supported: Constant, Gemm, MatMul, Add, Relu, Identity. No operator was skipped.")
    for imported in model.opset_import:
        if imported.domain in ("", "ai.onnx") and imported.version < 7:
            _fail("onnx_opset", "opset_import", "ONNX opsets before 7 use unsupported broadcasting semantics. Re-export with a current opset.")
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        _fail("onnx_invalid", str(model_path), f"ONNX validation failed: {exc}")

    initializer_names = {item.name for item in model.graph.initializer}
    parameter_inputs = initializer_names & {item.name for item in model.graph.input}
    if parameter_inputs:
        _fail("onnx_parameter_input", "graph.input", f"Initializers are also exposed as overridable inputs: {sorted(parameter_inputs)}. Re-export with fixed weights (keep_initializers_as_inputs=False).")
    if len(model.graph.input) != 1:
        _fail("onnx_input_count", "graph.input", "The planner requires exactly one data input; combine data inputs explicitly before export.")
    if not model.graph.output:
        _fail("onnx_outputs", "graph.output", "The model must declare at least one output.")

    def shape_of(value, location):
        if not value.type.HasField("tensor_type"):
            _fail("onnx_type", location, "Only floating-point tensor values are supported.")
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT16):
            _fail("onnx_dtype", location, "Only float16, float32 or float64 computation is supported; quantized and integer models need a dedicated backend.")
        if not tensor_type.HasField("shape"):
            _fail("onnx_shape", location, "Tensor shape must be declared; export a vector or batch-one tensor with fixed feature count.")
        dims = tensor_type.shape.dim
        if len(dims) not in (1, 2):
            _fail("onnx_shape", location, "Expected [features] or [batch, features]. Image, sequence and higher-rank tensors are unsupported.")
        if not dims[-1].HasField("dim_value") or dims[-1].dim_value <= 0:
            _fail("onnx_shape", location, "The feature dimension must be a fixed positive integer.")
        if len(dims) == 2:
            if dims[0].HasField("dim_value") and dims[0].dim_value != 1:
                _fail("onnx_batch", location, "Only batch size one is supported. Symbolic batch dimensions are imported explicitly as batch one.")
            return (1, int(dims[-1].dim_value))
        return (int(dims[-1].dim_value),)

    def numeric(value, location):
        pending = [value] if isinstance(value, (list, tuple)) else []
        visited = set()
        while pending:
            entry = pending.pop()
            if isinstance(entry, (list, tuple)):
                if id(entry) not in visited:
                    visited.add(id(entry))
                    pending.extend(entry)
            elif isinstance(entry, (bool, np.bool_)):
                _fail("onnx_constant_type", location, "Numeric values must not contain booleans.")
        array = np.asarray(value)
        if array.dtype.kind not in "fiu":
            _fail("onnx_constant_type", location, "Weights and bias constants must contain real numeric values.")
        array = array.astype(np.float64)
        if not np.isfinite(array).all():
            _fail("onnx_nonfinite", location, "Weights, biases and constants must be finite.")
        return array

    constants = {}
    for item in model.graph.initializer:
        try:
            constants[item.name] = numeric(numpy_helper.to_array(item), f"initializer.{item.name}")
        except InputValidationError:
            raise
        except Exception as exc:
            _fail("onnx_tensor", f"initializer.{item.name}", f"Cannot decode tensor: {exc}")

    ids: set[str] = set()

    def identifier(raw):
        base = re.sub(r"[^A-Za-z0-9_.:-]", "_", raw) or "node"
        if not base[0].isalpha() or not base[0].isascii():
            base = "node_" + base
        candidate = base
        suffix = 2
        while candidate in ids:
            candidate = f"{base}_{suffix}"
            suffix += 1
        ids.add(candidate)
        return candidate

    first = model.graph.input[0]
    first_shape = shape_of(first, "graph.input[0]")
    size = first_shape[-1]
    try:
        if len(input_bounds) != 2:
            raise ValueError("expected a lower/upper pair")
        lower = np.broadcast_to(numeric(input_bounds[0], "input_bounds.lower"), (size,)).copy()
        upper = np.broadcast_to(numeric(input_bounds[1], "input_bounds.upper"), (size,)).copy()
        if np.any(lower > upper):
            raise ValueError("each lower bound must be <= its upper bound")
    except (TypeError, ValueError, KeyError) as exc:
        if isinstance(exc, InputValidationError):
            raise
        _fail("onnx_input_bounds", "input_bounds", f"Expected scalar or per-feature lower/upper bounds: {exc}")
    input_id = identifier(first.name)
    nodes = [{"id": input_id, "op": "input", "inputs": [], "size": size, "input_bounds": {"lower": lower.tolist(), "upper": upper.tolist()}}]
    values = {first.name: (input_id, first_shape)}

    def dynamic(tensor_name, location):
        if tensor_name not in values:
            _fail("onnx_operand", location, f"Expected an earlier dynamic tensor, received {tensor_name!r}.")
        return values[tensor_name]

    def constant(tensor_name, location):
        if tensor_name not in constants:
            _fail("onnx_operand", location, f"Expected constant weights/bias for {tensor_name!r}; dynamic parameters are unsupported.")
        return constants[tensor_name]

    def bias_vector(value, output_shape, location):
        try:
            return np.broadcast_to(value, output_shape).reshape(-1)
        except ValueError:
            _fail("onnx_broadcast", location, f"Constant shape {value.shape} cannot broadcast to {output_shape} without changing the dynamic tensor shape.")

    for index, operation in enumerate(model.graph.node):
        location = f"graph.node[{index}]({operation.name or operation.op_type})"
        if len(operation.output) != 1 or not operation.output[0]:
            _fail("onnx_output_count", location, "Each supported operator must have exactly one nonempty output.")
        result_name = operation.output[0]
        if result_name in values or result_name in constants:
            _fail("onnx_duplicate_tensor", location, f"Tensor {result_name!r} is defined more than once.")
        attributes = {item.name: onnx.helper.get_attribute_value(item) for item in operation.attribute}
        op = operation.op_type
        allowed_attributes = {"alpha", "beta", "transA", "transB"} if op == "Gemm" else {"value", "value_float", "value_floats", "value_int", "value_ints"} if op == "Constant" else set()
        unknown = set(attributes) - allowed_attributes
        if unknown:
            _fail("onnx_attribute", location, f"Unsupported attributes for {op}: {sorted(unknown)}.")
        if op == "Constant":
            if operation.input or len(attributes) != 1:
                _fail("onnx_constant", location, "Constant must have no inputs and exactly one numeric value attribute.")
            key, value = next(iter(attributes.items()))
            try:
                decoded = numpy_helper.to_array(value) if key == "value" else value
                constants[result_name] = numeric(decoded, location)
            except InputValidationError:
                raise
            except Exception as exc:
                _fail("onnx_constant", location, f"Cannot decode Constant: {exc}")
            continue

        inputs = list(operation.input)
        if op in ("Identity", "Relu"):
            if len(inputs) != 1:
                _fail("onnx_operand_count", location, f"{op} requires one input.")
            predecessor, output_shape = dynamic(inputs[0], location)
            mapped = {"op": "identity" if op == "Identity" else "relu", "inputs": [predecessor], "size": output_shape[-1]}
        elif op in ("Gemm", "MatMul"):
            if len(inputs) not in ((2, 3) if op == "Gemm" else (2,)):
                _fail("onnx_operand_count", location, f"Unexpected number of operands for {op}.")
            predecessor, source_shape = dynamic(inputs[0], location)
            matrix = constant(inputs[1], location)
            if matrix.ndim != 2:
                _fail("onnx_weight_shape", location, "Dense weights must be a rank-two constant matrix.")
            if op == "Gemm":
                if len(source_shape) != 2:
                    _fail("onnx_shape", location, "Gemm requires a rank-two batch-one input.")
                if attributes.get("transA", 0) != 0:
                    _fail("onnx_transpose", location, "Gemm transA is unsupported; export row-vector inference with transA=0.")
                if attributes.get("transB", 0) not in (0, 1):
                    _fail("onnx_transpose", location, "Gemm transB must be zero or one.")
                if attributes.get("transB", 0):
                    matrix = matrix.T
            if matrix.shape[0] != source_shape[-1] or matrix.shape[1] < 1:
                _fail("onnx_weight_shape", location, f"Weight shape {matrix.shape} is incompatible with input shape {source_shape}.")
            output_shape = (*source_shape[:-1], matrix.shape[1])
            alpha = float(attributes.get("alpha", 1.0))
            beta = float(attributes.get("beta", 1.0))
            numeric([alpha, beta], location)
            with np.errstate(over="ignore", invalid="ignore"):
                weights = numeric(alpha * matrix.T, location)
                bias = numeric(beta * bias_vector(constant(inputs[2], location), output_shape, location), location) if len(inputs) == 3 and inputs[2] else np.zeros(matrix.shape[1])
            mapped = {"op": "linear", "inputs": [predecessor], "size": matrix.shape[1], "weights": weights.tolist(), "bias": bias.tolist()}
        else:  # Add
            if len(inputs) != 2:
                _fail("onnx_operand_count", location, "Add requires two operands.")
            dynamic_inputs = [item for item in inputs if item in values]
            if len(dynamic_inputs) == 2:
                left, output_shape = dynamic(inputs[0], location)
                right, right_shape = dynamic(inputs[1], location)
                if right_shape != output_shape:
                    _fail("onnx_broadcast", location, f"Dynamic Add requires identical shapes; got {output_shape} and {right_shape}. Broadcasting between dynamic tensors is unsupported.")
                mapped = {"op": "add", "inputs": [left, right], "size": output_shape[-1]}
            elif len(dynamic_inputs) == 1:
                dynamic_name = dynamic_inputs[0]
                predecessor, output_shape = dynamic(dynamic_name, location)
                fixed_name = inputs[1] if inputs[0] == dynamic_name else inputs[0]
                bias = bias_vector(constant(fixed_name, location), output_shape, location)
                mapped = {"op": "linear", "inputs": [predecessor], "size": output_shape[-1], "weights": np.eye(output_shape[-1]).tolist(), "bias": bias.tolist()}
            else:
                _fail("onnx_operand", location, "Add requires one or two dynamic vector operands; constant-only subgraphs must be folded before export.")
        node_id = identifier(result_name)
        nodes.append({"id": node_id, **mapped})
        values[result_name] = (node_id, output_shape)

    outputs = []
    for index, output in enumerate(model.graph.output):
        node_id, actual_shape = dynamic(output.name, f"graph.output[{index}]")
        declared_shape = shape_of(output, f"graph.output[{index}]")
        if declared_shape != actual_shape:
            _fail("onnx_shape", f"graph.output[{index}]", f"Declared output shape {declared_shape} differs from computed shape {actual_shape}.")
        outputs.append(node_id)
    # Shape annotations on intermediate dynamic tensors must agree too.
    for index, annotation in enumerate(model.graph.value_info):
        if annotation.name in values:
            declared_shape = shape_of(annotation, f"graph.value_info[{index}]")
            if declared_shape != values[annotation.name][1]:
                _fail("onnx_shape", f"graph.value_info[{index}]", "Intermediate tensor shape disagrees with operator-derived shape.")
    # The structural checker alone accepts some dtype-inconsistent graphs.
    # Check the original typed graph before returning the float-valued IR so
    # conversion of numeric constants cannot conceal an invalid ONNX program.
    # Running this after our supported-subset checks preserves specific shape
    # and operator diagnostics while still validating original type semantics.
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        _fail("onnx_invalid", str(model_path), f"ONNX type/shape validation failed: {exc}")
    network = {"schema_version": "0.1", "name": name or model.graph.name or model_path.stem, "nodes": nodes, "outputs": outputs}
    try:
        return Network.model_validate(network).model_dump(mode="json", exclude_none=True)
    except ValidationError as exc:
        _fail("onnx_network", str(model_path), f"Imported graph does not satisfy the planner contract: {exc}")
