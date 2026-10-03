"""Independent decimal/scalar oracles for carrier-free M1 digital semantics."""
import copy
from decimal import Decimal, localcontext
import json
import math
import random
import struct
import unittest

import numpy as np

from npp.multiphysics.semantic import (
    FFNWorkload, NumericPolicy, NumericalFailure, SemanticError, add,
    evaluate_ffn, linear, multiply, operation_counts, silu,
)


def fixture():
    return {
        "W_up": [[1, -2], [0.5, 1.5], [-1, 0.2]],
        "W_gate": [[-0.4, 0.8], [1.2, -0.1], [0.3, 0.7]],
        "W_down": [[0.6, -0.7, 0.2], [-0.4, 1.1, 0.3]],
        "b_up": [0.2, -0.3, 0.4], "b_gate": [0.1, 0.4, -0.2], "b_down": [0.5, -0.6],
    }


def decimal_oracle(model, vector):
    """No production kernels or NumPy; independent high precision exp algebra."""
    with localcontext() as context:
        context.prec = 80
        x = [Decimal.from_float(float(v)) for v in vector]
        def mat(name, bias, values):
            return [sum((Decimal.from_float(float(a)) * b for a, b in zip(row, values)),
                        Decimal.from_float(float(offset)))
                    for row, offset in zip(model[name], model[bias])]
        u = mat("W_up", "b_up", x)
        g = mat("W_gate", "b_gate", x)
        hidden = [(v / (1 + (-v).exp())) * a for v, a in zip(g, u)]
        return [float(v) for v in mat("W_down", "b_down", hidden)]


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def float32_scalar_oracle(model, vector, accumulation):
    """Pack/unpack float32 rounding, independent of NumPy implementation."""
    accum = fp32 if accumulation == "float32" else float
    def mat(name, bias, values):
        result = []
        for row, offset in zip(model[name], model[bias]):
            total = accum(0.0)
            for a, x in zip(row, values):
                total = accum(total + accum(fp32(fp32(a) * fp32(x))))
            result.append(fp32(accum(total + accum(fp32(offset)))))
        return result
    u = mat("W_up", "b_up", vector)
    g = mat("W_gate", "b_gate", vector)
    gate = [fp32(v / (1 + math.exp(-v))) for v in g]
    return mat("W_down", "b_down", [fp32(a * b) for a, b in zip(u, gate)])


class SemanticReferenceTests(unittest.TestCase):
    def test_seeded_shapes_biases_batches_against_decimal_oracle(self):
        rng = random.Random(701)
        for n, h, m in ((1, 1, 1), (2, 3, 2), (5, 2, 4), (3, 7, 1)):
            for _ in range(5):
                matrix = lambda rows, cols: [[rng.uniform(-3, 3) for _ in range(cols)] for _ in range(rows)]
                vector = lambda width: [rng.uniform(-2, 2) for _ in range(width)]
                model = dict(W_up=matrix(h, n), W_gate=matrix(h, n), W_down=matrix(m, h),
                             b_up=vector(h), b_gate=vector(h), b_down=vector(m))
                batch = [vector(n) for _ in range(3)]
                result = evaluate_ffn(model, batch)
                expected = [decimal_oracle(model, x) for x in batch]
                np.testing.assert_allclose(result["ideal_output"], expected, rtol=2e-13, atol=2e-13)
                np.testing.assert_allclose(result["output"], expected, rtol=2e-13, atol=2e-13)
                json.dumps(result, allow_nan=False)

    def test_float32_rounding_and_accumulation_match_scalar_oracle(self):
        model = fixture()
        vector = [0.123456781, -1.987654321]
        for accumulation in ("float32", "float64"):
            policy = NumericPolicy(format="float32", accumulation=accumulation)
            result = evaluate_ffn(model, vector, policy)
            self.assertEqual(result["output"], float32_scalar_oracle(model, vector, accumulation))
            self.assertGreater(result["absolute_error"]["max"], 0)
            self.assertEqual(result["numeric_policy"]["accumulation"], accumulation)

    def test_reference_keeps_compensated_sum_separate_from_finite_execution(self):
        model = dict(W_up=[[1e16, 1, -1e16]], W_gate=[[0, 0, 0]], W_down=[[1]],
                     b_gate=[1])
        result = evaluate_ffn(model, [1, 1, 1])
        self.assertEqual(result["output"], [0])
        self.assertAlmostEqual(result["ideal_output"][0], 1 / (1 + math.exp(-1)))
        self.assertGreater(result["absolute_error"]["max"], 0.7)

    def test_linear_orientation_and_no_broadcast(self):
        self.assertEqual(linear([[1, 2], [3, 4]], [5, 6], [7, 8]).tolist(), [24, 47])
        self.assertEqual(linear([[1, 2]], [[3, 4], [5, 6]], [1]).tolist(), [[12], [18]])
        for call in (lambda: linear([[1, 2]], [1]), lambda: add([[1, 2]], [1, 2]),
                     lambda: multiply([1], [1, 2]), lambda: linear([[1]], [1], [1, 2])):
            with self.assertRaises(SemanticError):
                call()

    def test_silu_stable_extremes_and_declared_gradual_underflow(self):
        output = silu([-1e308, -1000, 0, 1000, 1e308])
        self.assertTrue(np.isfinite(output).all())
        self.assertEqual(output.tolist(), [-0.0, -0.0, 0.0, 1000.0, 1e308])
        self.assertEqual(multiply([1e-300], [1e-300]).tolist(), [0])
        self.assertEqual(NumericPolicy().underflow, "gradual")

    def test_nonfinite_arithmetic_reports_numerical_failure(self):
        for call in (lambda: linear([[1e308]], [2]), lambda: add([1e308], [1e308]),
                     lambda: multiply([1e308], [2]),
                     lambda: linear([[1e100]], [1], policy=NumericPolicy(format="float32")),
                     lambda: evaluate_ffn(dict(W_up=[[1e308]], W_gate=[[1]], W_down=[[1]]), [2])):
            with self.assertRaises(NumericalFailure):
                call()

    def test_positive_subnormal_rms_cannot_be_reported_as_zero(self):
        model = dict(W_up=[[1]], W_gate=[[1]], W_down=[[5e-324]],
                     semantic_track="A", quality_budget=0)
        policy = NumericPolicy(format="float32", accumulation="float32")
        with self.assertRaisesRegex(NumericalFailure, "RMS error underflows"):
            evaluate_ffn(model, [[1], [0], [0], [0]], policy)
        # The smallest positive representable RMS remains visible when it is
        # not divided below range; an actually zero error remains valid.
        positive = evaluate_ffn(model, [1], policy)
        self.assertEqual(positive["absolute_error"], {"max": 5e-324, "rms": 5e-324})
        self.assertFalse(positive["quality_passed"])
        zero = evaluate_ffn(model, [[0], [0], [0], [0]], policy)
        self.assertEqual(zero["absolute_error"], {"max": 0.0, "rms": 0.0})
        self.assertTrue(zero["quality_passed"])

    def test_work_counts_expose_logical_boundary(self):
        w = FFNWorkload.from_dict(fixture())
        counts = operation_counts(w, 4)
        self.assertEqual(counts["matrix_mac"], 4 * (2 * 2 * 3 + 3 * 2))
        self.assertEqual(counts["bias_add"], 4 * (2 * 3 + 2))
        self.assertEqual(counts["silu"], 12)
        self.assertEqual(counts["elementwise_multiply"], 12)
        self.assertEqual(counts["weight_elements"], 18)
        self.assertIn("no assumed hardware reuse", counts["count_boundary"])
        self.assertEqual(evaluate_ffn(w, [1, 2])["operations"]["execution"], "matrix_vector")

    def test_workload_is_immutable_and_normalized(self):
        raw = fixture()
        w = FFNWorkload.from_dict(raw)
        ident = w.model_identity
        raw["W_up"][0][0] = 888
        exported = w.to_dict()
        exported["W_up"][0][0] = 999
        self.assertEqual(w.W_up[0][0], 1)
        self.assertEqual(w.model_identity, ident)
        zeros = FFNWorkload.from_dict(dict(W_up=[[1]], W_gate=[[2]], W_down=[[3]]))
        explicit = FFNWorkload.from_dict(dict(W_up=[[1]], W_gate=[[2]], W_down=[[3]], b_up=[0], b_gate=[0], b_down=[0]))
        self.assertEqual(zeros.model_identity, explicit.model_identity)
        self.assertEqual(FFNWorkload.from_dict(w.to_dict()), w)
        self.assertEqual([node["bias"] for node in w.semantic_ir()["nodes"] if "bias" in node], ["b_up", "b_gate", "b_down"])

    def test_adversarial_inputs(self):
        mutations = [
            {"W_up": [[True, 2], [1, 2], [1, 2]]}, {"W_up": [["1", 2], [1, 2], [1, 2]]},
            {"W_up": [[math.nan, 2], [1, 2], [1, 2]]}, {"W_gate": [[1]]},
            {"W_down": [[1, 2]]}, {"b_up": [1]}, {"b_gate": [True, 0, 0]},
            {"W_up": [[1, 2], [3]]}, {"W_up": []}, {"W_up": [[1 + 1j, 2]]},
            {"name": ""}, {"semantic_track": "exact-ish"}, {"quality_budget": True},
            {"quality_budget": math.inf}, {"quality_budget": 10 ** 1000}, {"unknown_option": 1},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(SemanticError):
                FFNWorkload.from_dict({**fixture(), **mutation})
        for x in ([True, 1], [math.inf, 1], [], [[1, 2], [1]], [1, 2, 3], [[1, 2]] * 65):
            with self.subTest(x=x), self.assertRaises(SemanticError):
                evaluate_ffn(fixture(), x)
        with self.assertRaises(SemanticError):
            FFNWorkload.from_dict({"W_up": [[1]]})

    def test_track_contracts_and_quality_scope(self):
        for extras in ({"semantic_track": "A"}, {"semantic_track": "R"},
                       {"parent_model_identity": "parent"},
                       {"semantic_track": "R", "parent_model_identity": "same", "adaptation_identity": "same", "adaptation_cost": "unknown"}):
            with self.assertRaises(SemanticError):
                FFNWorkload.from_dict({**fixture(), **extras})
        approximate = {**fixture(), "semantic_track": "A", "quality_budget": 0}
        result = evaluate_ffn(approximate, [0.123456, 0.999], NumericPolicy(format="float32"))
        self.assertFalse(result["quality_passed"])
        self.assertEqual(result["status"], "evaluated")
        adapted = {**fixture(), "semantic_track": "R", "parent_model_identity": "old", "adaptation_identity": "new", "adaptation_cost": "unavailable: training not measured"}
        result = evaluate_ffn(adapted, [1, 2])
        self.assertIn("no task-quality or parent-model equivalence", result["quality_scope"])

    def test_numeric_contract_rejects_unsupported_modes(self):
        for value in ({"format": "int8"}, {"accumulation": "float16"},
                      {"format": "float64", "accumulation": "float32"}, {"overflow": "saturate"},
                      {"rounding": "stochastic"}, {"underflow": "flush_to_zero"}, {"typo": 1}):
            with self.assertRaises(SemanticError):
                NumericPolicy.from_dict(value)

    def test_algebraic_signed_permutation_with_bias_independent_oracle(self):
        """Review the identity's precondition; transformation engine comes in M2."""
        original = fixture()
        perm, signs = [2, 0, 1], [-1, 1, -1]
        transformed = copy.deepcopy(original)
        transformed["W_gate"] = [original["W_gate"][k] for k in perm]
        transformed["b_gate"] = [original["b_gate"][k] for k in perm]
        transformed["W_up"] = [[s * x for x in original["W_up"][k]] for k, s in zip(perm, signs)]
        transformed["b_up"] = [s * original["b_up"][k] for k, s in zip(perm, signs)]
        transformed["W_down"] = [[s * row[k] for k, s in zip(perm, signs)] for row in original["W_down"]]
        for vector in ([0, 0], [1, -2], [-4, 1], [0.125, 8]):
            np.testing.assert_allclose(decimal_oracle(original, vector), decimal_oracle(transformed, vector), rtol=1e-14, atol=1e-14)
        invalid = copy.deepcopy(transformed)
        invalid["W_gate"][0] = [-x for x in invalid["W_gate"][0]]
        invalid["b_gate"][0] *= -1
        self.assertGreater(max(abs(a - b) for a, b in zip(decimal_oracle(original, [1, -2]), decimal_oracle(invalid, [1, -2]))), 1e-3)


if __name__ == "__main__":
    unittest.main()
