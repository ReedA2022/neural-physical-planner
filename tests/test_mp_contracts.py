"""Negative contract tests: invalid composition must fail before backend search."""
import json
import math
import subprocess
import sys
import unittest

from pydantic import ValidationError

from npp.multiphysics.contracts import (
    AccountingOwner, AccountingPolicy, BackendDescriptor, ClockContract,
    ComponentContract, CostEntry, CostLedger, Diagnostic, DynamicsContract,
    EnvelopeAxis, EvaluationIdentity, EvaluationOutcome, EvaluatorRegistry,
    Evidence, IntegrationContract, Interval, LifetimeContract, Parameter,
    PortContract, Quantity, RandomStream, Reservation, ResourceContract,
    SignalContract, StateContract, TechnologyPack, Uncertainty,
    ValidationContract, canonical_hash, canonical_json, known, signal_mismatches,
    unavailable,
)


def signal(**changes):
    data = dict(payload="tensor", carrier="electrical", regime="classical_digital",
                shape=(2,), axis_order=("features",), encoding="float64",
                physical_variable="digital_word", signed=True, scale=1.0,
                unit="1", value_range=Interval(minimum=-10.0, maximum=10.0),
                bandwidth_hz=known(1e9, "Hz"),
                clock=ClockContract(domain="main", mode="synchronous", period_ns=1.0),
                lifetime=LifetimeContract(arrival_ns=Interval(minimum=0.0, maximum=0.0),
                                          integration_ns=1.0, retention_ns=known(100.0, "ns"),
                                          reset_required=False))
    data.update(changes)
    return SignalContract(**data)


def parameter(name="energy", **changes):
    data = dict(name=name, quantity=known(1.0, "pJ"),
                evidence=Evidence(kind="hypothetical", source="fixture assumptions", location="test",
                                  version="1", scope="Synthetic coefficient; not a measurement"),
                uncertainty=Uncertainty(kind="epistemic", description="Assumed coefficient", bounds=Interval(minimum=0.5, maximum=2.0)),
                envelope=Interval(minimum=0.0, maximum=10.0), instance_group="fixture")
    data.update(changes)
    return Parameter(**data)


def component(**changes):
    data = dict(id="digital", version="1", family="digital", mechanism="deterministic arithmetic",
                evaluator_id="digital.reference", evaluator_version="1", ideal_relation="y=x",
                fidelity="algebraic_ideal", parameters=(parameter(),),
                ports=(PortContract(name="in", direction="in", signal=signal()),),
                operating_envelope=(EnvelopeAxis(name="temperature_c", minimum=0.0, maximum=50.0, unit="degC"),),
                integration=IntegrationContract(substrate="hypothetical CMOS", package="test package",
                                                temperature_c=Interval(minimum=0.0, maximum=50.0), required_services=("power",),
                                                interface_families=("digital",), geometry_assumptions=("bounded tile",)),
                dynamics=DynamicsContract(**{name: unavailable("ns", "not supplied") for name in DynamicsContract.model_fields}),
                imperfections=("No calibrated hardware error model",),
                accounting=AccountingPolicy(mode="leaf", boundary="system", cost_categories=("compute",),
                                             exclusions=(), amortization="one operation"),
                validation=ValidationContract(reference_implementations=(), tests=("identity fixture",), known_failures=(),
                                              limitations=("Only a schema fixture",)))
    data.update(changes)
    return ComponentContract(**data)


def entry(name, value, owner="a", **changes):
    data = dict(contribution_id=name, owner=owner, boundary="system", category="compute",
                metric="energy", quantity=value, basis="one operation")
    data.update(changes)
    return CostEntry(**data)


def ledger(entries, owners=None, **changes):
    data = dict(boundary="system", owners=owners or (AccountingOwner(id="a", mode="leaf"),),
                entries=entries, expected_contributions=tuple(e.contribution_id for e in entries))
    data.update(changes)
    return CostLedger(**data)


def identity(**changes):
    data = dict(model_version="model:1", pack_versions=("pack:1",), component_versions=("component:1",),
                interface_versions=("interface:1",), evaluator_versions=("evaluator:1",),
                normalized_inputs_hash=canonical_hash({"x": [1.0]}), workload_hash=canonical_hash("FFN"),
                environment_hash=canonical_hash({"temperature": 25}), geometry_hash=canonical_hash({"distance": 1}),
                schedule_hash=canonical_hash({"start": 0}), fidelity="analytical_cost",
                random_streams=(), seed_policy="deterministic")
    data.update(changes)
    return EvaluationIdentity(**data)


class MultiphysicsContractsTests(unittest.TestCase):
    def test_unknown_is_not_zero_and_bounds_are_explicit(self):
        unknown = unavailable("pJ", "measurement missing")
        self.assertIsNone(unknown.value)
        for bad in [dict(state="known", unit="pJ"),
                    dict(state="unavailable", unit="pJ", reason="missing", value=0.0),
                    dict(state="bounded", unit="pJ"),
                    dict(state="bounded", unit="pJ", lower=2.0, upper=1.0)]:
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                Quantity(**bad)
        self.assertEqual(Quantity(state="bounded", unit="pJ", symbol="unmeasured").symbol, "unmeasured")
        for value in [True, "1", float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                known(value, "pJ")

    def test_parameter_evidence_fidelity_and_envelope_are_distinct(self):
        ideal = component(fidelity="circuit_device")
        self.assertEqual(ideal.parameters[0].evidence.kind, "hypothetical")
        with self.assertRaises(ValidationError):
            parameter(quantity=known(11.0, "pJ"))
        with self.assertRaises(ValidationError):
            parameter(dependencies=("energy",))
        with self.assertRaises(ValidationError):
            component(parameters=(parameter(dependencies=("missing",)),))
        with self.assertRaises(ValidationError):
            component(parameters=(parameter(dependencies=("other",)), parameter("other", dependencies=("energy",))))
        with self.assertRaises(ValidationError):
            component(parameters=(parameter(dependencies=("other",)), parameter("other", instance_group="incompatible_device")))

    def test_roundtrip_strict_json_and_deep_immutability(self):
        pack = TechnologyPack(id="fixture", version="1", components=(component(),), assumptions=("hypothetical",))
        self.assertEqual(pack, TechnologyPack.model_validate_json(pack.model_dump_json()))
        raw = pack.model_dump(mode="json")
        raw["components"][0]["parameters"][0]["quantity"]["value"] = True
        with self.assertRaises(ValidationError):
            TechnologyPack.model_validate_json(json.dumps(raw))
        with self.assertRaises(ValidationError):
            pack.components[0].parameters[0].quantity.value = 2.0
        with self.assertRaises(TypeError):
            pack.components[0] = component()
        with self.assertRaises(ValidationError):
            TechnologyPack(id="fixture", version="1", components=(component(), component()), assumptions=())
        raw = pack.model_dump(mode="json")
        raw["components"][0]["formula"] = "__import__('os').system('false')"
        with self.assertRaises(ValidationError):
            TechnologyPack.model_validate_json(json.dumps(raw))

    def test_signal_contracts_reject_implicit_field_sign_and_quantum_casts(self):
        with self.assertRaises(ValidationError):
            signal(shape=(2, 2))
        with self.assertRaises(ValidationError):
            signal(signed=False)
        optical = dict(carrier="optical", regime="classical_analog", encoding="intensity",
                       physical_variable="power", signed=False,
                       value_range=Interval(minimum=0.0, maximum=1.0))
        intensity = signal(**optical)
        with self.assertRaises(ValidationError):
            signal(**{**optical, "signed": True})
        with self.assertRaises(ValidationError):
            signal(**{**optical, "physical_variable": "field_amplitude"})
        with self.assertRaises(ValidationError):
            signal(carrier="optical", regime="classical_analog", encoding="coherent_field", physical_variable="field_amplitude")
        field = signal(carrier="optical", regime="classical_analog", encoding="coherent_field",
                       physical_variable="field_amplitude", phase_reference="laser")
        self.assertIn("encoding", signal_mismatches(intensity, field))
        with self.assertRaises(ValidationError):
            signal(regime="quantum")
        with self.assertRaises(ValidationError):
            signal(payload="quantum_register")
        with self.assertRaises(ValidationError):
            signal(disturbance_ids=("laser", "laser"))
        self.assertIn("clock", signal_mismatches(signal(), signal(clock=ClockContract(domain="other", mode="asynchronous"))))
        self.assertEqual(signal_mismatches(signal(), signal()), ())

    def test_unknown_totals_stay_incomplete_and_units_are_not_cast(self):
        costs = ledger((entry("arith", known(2.0, "pJ")), entry("readout", unavailable("pJ", "not measured"))))
        total = costs.aggregate("energy", "pJ")
        self.assertFalse(total.complete)
        self.assertEqual(total.known_subtotal, 2.0)
        self.assertEqual(total.quantity.state, "unavailable")
        self.assertIsNone(total.quantity.value)
        self.assertEqual(total.unresolved, ("readout",))
        self.assertFalse(costs.aggregate("area", "um2").complete)
        with self.assertRaises(ValueError):
            costs.aggregate("energy", "J")
        bounded = ledger((entry("a", known(2.0, "pJ")), entry("b", Quantity(state="bounded", lower=3.0, upper=4.0, unit="pJ"))))
        self.assertEqual(bounded.aggregate("energy", "pJ").quantity.lower, 5.0)
        self.assertEqual(bounded.aggregate("energy", "pJ").quantity.upper, 6.0)
        symbolic = ledger((entry("a", Quantity(state="bounded", symbol="x", unit="pJ")),))
        self.assertIsNone(symbolic.aggregate("energy", "pJ").quantity.upper)
        with self.assertRaises(ValueError):
            ledger((entry("a", known(1.7e308, "pJ")), entry("b", known(1.7e308, "pJ")))).aggregate("energy", "pJ")

    def test_ledger_enforces_single_ownership_complete_expected_set_and_boundary(self):
        e = entry("same_effect", known(1.0, "pJ"))
        with self.assertRaises(ValidationError):
            ledger((e, e))
        with self.assertRaises(ValidationError):
            ledger((e,), expected_contributions=("same_effect", "omitted_readout"))
        with self.assertRaises(ValidationError):
            ledger((entry("a", known(1.0, "pJ"), boundary="another_system"),))
        with self.assertRaises(ValidationError):
            ledger((entry("a", known(1.0, "pJ"), owner="undeclared"),))
        with self.assertRaises(ValidationError):
            entry("a", known(-1.0, "pJ"))
        owners = (AccountingOwner(id="parent", mode="expanded", children=("a",)), AccountingOwner(id="a", mode="leaf"))
        self.assertEqual(ledger((e, entry("overhead", known(2.0, "pJ"), owner="parent")), owners).aggregate("energy", "pJ").quantity.value, 3.0)
        with self.assertRaises(ValidationError):
            AccountingOwner(id="parent", mode="inclusive", children=("a",))
        inclusive = (AccountingOwner(id="parent", mode="inclusive", children=("a",), resource_summary_complete=True), owners[1])
        with self.assertRaises(ValidationError):
            ledger((e, entry("all", known(3.0, "pJ"), owner="parent")), inclusive)
        self.assertEqual(ledger((entry("all", known(3.0, "pJ"), owner="parent"),), inclusive).aggregate("energy", "pJ").quantity.value, 3.0)
        cycle = (AccountingOwner(id="parent", mode="expanded", children=("a",)), AccountingOwner(id="a", mode="expanded", children=("parent",)))
        with self.assertRaises(ValidationError):
            ledger((), cycle)

    def test_inclusive_composites_export_real_child_reservations(self):
        child_reservation = Reservation(resource_id="dac", owner="a", start_ns=0.0, end_ns=2.0, amount=1)
        child = AccountingOwner(id="a", mode="leaf", reservations=(child_reservation,))
        parent = AccountingOwner(id="parent", mode="inclusive", children=("a",), resource_summary_complete=True)
        with self.assertRaises(ValidationError):
            ledger((), (parent, child))
        exported = Reservation(resource_id="dac", owner="parent", start_ns=0.0, end_ns=2.0, amount=1)
        complete = AccountingOwner(id="parent", mode="inclusive", children=("a",), resource_summary_complete=True,
                                   reservations=(exported,))
        self.assertEqual(len(ledger((), (complete, child)).owners), 2)
        corrupt = exported.model_copy(update={"amount": 0})
        with self.assertRaises(ValidationError):
            AccountingOwner(id="parent", mode="inclusive", children=("a",), resource_summary_complete=True,
                            reservations=(corrupt,))

    def test_state_and_resource_contracts_explicit_no_fake_retention(self):
        data = dict(id="buffer", owner="tile", version=0, kind="excitation", created_ns=0.0,
                    retention_ns=known(10.0, "ns"), read_mode="bounded", max_reads=1,
                    reset_required=True, capacity=1, capacity_unit="symbol")
        self.assertEqual(StateContract(**data).max_reads, 1)
        with self.assertRaises(ValidationError):
            StateContract(**{**data, "max_reads": None})
        with self.assertRaises(ValidationError):
            StateContract(**{**data, "retention_ns": known(-1.0, "ns")})
        with self.assertRaises(ValidationError):
            StateContract(**{**data, "kind": "quantum_register", "read_mode": "repeatable", "max_reads": None})
        with self.assertRaises(ValidationError):
            Reservation(resource_id="dac", owner="task", start_ns=2.0, end_ns=1.0, amount=1)
        with self.assertRaises(ValidationError):
            ResourceContract(id="dac", owner="tile", kind="converter", capacity=True, capacity_unit="lane", region="chip")

    def test_cache_changes_for_every_dependency_and_streams_are_separated(self):
        base = identity()
        self.assertEqual(base.cache_key(), identity().cache_key())
        for field in ("normalized_inputs_hash", "workload_hash", "environment_hash", "geometry_hash", "schedule_hash"):
            self.assertNotEqual(base.cache_key(), identity(**{field: canonical_hash("changed")}).cache_key())
        for field in ("pack_versions", "component_versions", "interface_versions", "evaluator_versions"):
            self.assertNotEqual(base.cache_key(), identity(**{field: ("changed:2",)}).cache_key())
        self.assertNotEqual(base.cache_key(), identity(fidelity="circuit_device").cache_key())
        streams = (RandomStream(id="search", seed=1, role="search"), RandomStream(id="physics", seed=2, role="physical"))
        self.assertNotEqual(base.cache_key(), identity(random_streams=streams, seed_policy="explicit_streams").cache_key())
        with self.assertRaises(ValidationError):
            identity(random_streams=streams)
        with self.assertRaises(ValidationError):
            identity(random_streams=(streams[0], RandomStream(id="physics", seed=1, role="physical")), seed_policy="explicit_streams")
        with self.assertRaises(ValidationError):
            identity(random_streams=(RandomStream(id="a", seed=1, role="shared_disturbance", source_id="laser"),
                                     RandomStream(id="b", seed=2, role="shared_disturbance", source_id="laser")), seed_policy="explicit_streams")

    def test_outcomes_do_not_convert_unknown_or_numerical_failure_to_feasible(self):
        incomplete = ledger((entry("unknown", unavailable("pJ", "not measured")),))
        with self.assertRaises(ValidationError):
            EvaluationOutcome(status="model_feasible", diagnostics=(), ledger=incomplete)
        with self.assertRaises(ValidationError):
            EvaluationOutcome(status="conditional", diagnostics=())
        with self.assertRaises(ValidationError):
            EvaluationOutcome(status="unsupported", diagnostics=())
        output = EvaluationOutcome(status="conditional", diagnostics=(), obligations=("measure readout",), ledger=incomplete,
                                   payload_json='{"y": [2.0], "x": 1}')
        self.assertEqual(output.payload_json, '{"x":1,"y":[2.0]}')
        self.assertEqual(output.result_digest, canonical_hash({"x": 1, "y": [2.0]}))
        with self.assertRaises(ValidationError):
            EvaluationOutcome(status="model_feasible", diagnostics=(), payload_json='{"bad": NaN}')
        with self.assertRaises(ValidationError):
            EvaluationOutcome(status="model_feasible", diagnostics=(), payload_json='{}', result_digest="0" * 64)

    def test_registry_dispatches_ids_only_and_snapshots_inputs(self):
        registry = EvaluatorRegistry()
        missing = registry.evaluate("not.installed", "1", {}, {})
        self.assertEqual(missing.status, "unsupported")
        descriptor = BackendDescriptor(id="fixture", version="1", families=("digital",), capabilities=("identity",), limitations=("fixture only",))
        seen = []
        def evaluator(inputs_json, context_json):
            values = json.loads(inputs_json)
            values["x"][0] = 999
            seen.append(context_json)
            return EvaluationOutcome(status="model_feasible", diagnostics=(), payload_json=canonical_json(values))
        registry.register(descriptor, evaluator)
        values = {"x": [1.0]}
        self.assertEqual(registry.evaluate("fixture", "1", values, {"a": 1}).status, "model_feasible")
        self.assertEqual(values, {"x": [1.0]})
        self.assertEqual(seen, ['{"a":1}'])
        with self.assertRaises(ValueError):
            registry.register(descriptor, evaluator)
        other = descriptor.model_copy(update={"id": "bad"})
        def numerical_failure(*args):
            raise FloatingPointError("solver did not converge")
        registry.register(other, numerical_failure)
        self.assertEqual(registry.evaluate("bad", "1", {}, {}).status, "numerical_failure")

    def test_registry_separates_invalid_inputs_contracts_and_numerical_failures(self):
        registry = EvaluatorRegistry()
        calls = []
        descriptor = BackendDescriptor(id="fixture", version="1", families=("digital",),
                                       capabilities=("identity",), limitations=("fixture only",))
        def evaluator(inputs_json, context_json):
            calls.append(inputs_json)
            return EvaluationOutcome(status="model_feasible", diagnostics=())
        registry.register(descriptor, evaluator)
        for inputs, context in [({"x": float("nan")}, {}), ({}, {"temperature": float("inf")}),
                                ({1: "not a JSON key"}, {}), ({}, {"object": object()})]:
            with self.subTest(inputs=inputs, context=context):
                result = registry.evaluate("fixture", "1", inputs, context)
                self.assertEqual(result.status, "invalid")
                self.assertEqual(result.diagnostics[0].code, "invalid_evaluation_input")
        self.assertEqual(calls, [])
        def invalid_contract(*args):
            raise ValueError("input dimension violates registered contract")
        registry.register(descriptor.model_copy(update={"id": "invalid"}), invalid_contract)
        self.assertEqual(registry.evaluate("invalid", "1", {}, {}).status, "invalid")
        registry.register(descriptor.model_copy(update={"id": "malformed"}), lambda *args: {"status": "fake_success"})
        self.assertEqual(registry.evaluate("malformed", "1", {}, {}).status, "invalid")
        def arithmetic_failure(*args):
            raise OverflowError("finite numerical range exceeded")
        registry.register(descriptor.model_copy(update={"id": "overflow"}), arithmetic_failure)
        self.assertEqual(registry.evaluate("overflow", "1", {}, {}).status, "numerical_failure")

    def test_canonical_json_and_optional_backend_isolation(self):
        self.assertEqual(canonical_hash({"b": 1, "a": [2]}), canonical_hash({"a": [2], "b": 1}))
        for value in [float("nan"), float("inf"), {1: 2}, {"x": object()}]:
            with self.assertRaises(ValueError):
                canonical_hash(value)
        cycle = []
        cycle.append(cycle)
        with self.assertRaises(ValueError):
            canonical_hash(cycle)
        code = "import sys; import npp.multiphysics.contracts; assert not ({'sax','torch','qiskit','scipy'} & set(sys.modules))"
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
