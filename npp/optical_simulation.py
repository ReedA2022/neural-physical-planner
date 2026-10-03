"""Optional SAX verification of the declared full-scale optical power model.

SAX independently composes phenomenological component S matrices. It does not
calibrate those matrices, simulate the electronics, or validate an entire NN.
No optional simulator is imported by the export path.
"""
from __future__ import annotations

from copy import deepcopy
from importlib import metadata
import math

from .implementation import check_realization, evaluate_implementation
from .models import InputValidationError

MAX_WAVELENGTHS = 33
MAX_SEGMENTS = 320
MAX_SEGMENT_INSTANCES = 128
MAX_MATRIX_ELEMENTS = 20000
MAX_RTOL = 0.01
MAX_ATOL_MW = 1e-4
BACKEND = "klu"
ADAPTER_VERSION = "npp_scalar_optics_v1"

MODEL_DEFINITIONS = {
    "npp_splitter_v1": {
        "ports": ["p_in", "p_out1", "p_out2"],
        "power": "T=10**(-loss_db/10); P_out1=P_in*ratio*T; P_out2=P_in*(1-ratio)*T",
        "field": "reciprocal amplitudes sqrt(ratio*T) and sqrt((1-ratio)*T); other entries zero",
    },
    "npp_waveguide_v1": {
        "ports": ["p_in", "p_out"],
        "power": "P_out=P_in*10**(-loss_db_per_um*length_um/10)",
        "field": "reciprocal sqrt(transmission)*exp(2j*pi*neff(wl)*length_um/wl_um)",
        "dispersion": "neff(wl)=neff-(wl_um-reference_wavelength_um)*(ng-neff)/reference_wavelength_um",
    },
    "npp_modulator_full_scale_v1": {
        "ports": ["p_in", "p_out"],
        "power": "P_out=P_carrier*10**(-loss_db/10), electrical sample clamped to 1",
        "field": "reciprocal attenuation sqrt(transmission); no electrical or transient model",
    },
}

SCOPE = {
    "checked": ["SAX composition of full-scale optical power at every detector input and unused matched optical termination", "agreement with planner powers at each requested wavelength"],
    "unchecked": ["physical calibration or validity of component parameters", "NN computation or task accuracy", "electrical detection and regeneration dynamics", "noise, clipping and saturation statistics", "energy, area, acquisition time, latency and throughput", "optical phase and group delay", "reflections, coherent recombination, crosstalk and fabrication layout"],
    "boundary_conditions": "one external source excitation per disconnected optical segment; all other ports matched and unexcited; sources are full scale and modulators have electrical sample fixed to 1",
    "independence": "external SAX circuit solver and independently encoded field models; the same declared technology assumptions are used by both engines, so agreement checks composition, not experimental truth",
    "model_limitations": "phenomenological reciprocal matched splitter/guide/attenuator models; absent scattering entries are zero; attenuation and splitting ratios are wavelength independent within their declared envelopes",
    "phase": "first-order effective-index dispersion about the recorded wavelength retains distinct neff and ng; phase and group delay are not comparison targets",
}


def _error(code, path, message):
    raise InputValidationError([{"code": code, "path": path, "message": message}])


def _checked_record(record):
    result = check_realization(record)
    if not result["valid"]:
        raise InputValidationError(result["diagnostics"])
    return deepcopy(record)


def _endpoint(value):
    return value["instance"], value["port"]


def _port_name(original):
    return {"in": "p_in", "carrier": "p_in", "out": "p_out", "out1": "p_out1", "out2": "p_out2"}[original]


def _instance_model(instance, component, wavelength_nm):
    parameters = component["parameters"]
    kind = component["kind"]
    if kind == "splitter":
        return {"component": "npp_splitter_v1", "settings": {"ratio": parameters["power_ratio"], "loss_db": parameters["insertion_loss_db"]}}
    if kind == "waveguide":
        return {"component": "npp_waveguide_v1", "settings": {
            "length_um": instance["parameters"]["length_um"],
            "loss_db_per_um": parameters["attenuation_db_per_um"],
            "neff": parameters["effective_index"], "ng": parameters["group_index"],
            "reference_wavelength_um": wavelength_nm / 1000.0,
        }}
    if kind == "modulator":
        return {"component": "npp_modulator_full_scale_v1", "settings": {"loss_db": parameters["insertion_loss_db"]}}
    _error("unsupported_optical_model", instance["id"], f"no SAX field model for {kind}")


def export_optical_netlist(record: dict) -> dict:
    """Replay-check and export JSON netlists without requiring SAX.

    Valid but physically infeasible realizations may be exported for inspection;
    simulation rejects them. Dots and other NN ID characters never enter SAX's
    identifier grammar: each segment uses reversible local names i0, i1, ... .
    """
    record = _checked_record(record)
    graph, technology = record["graph"], record["technology"]
    instances = {i["id"]: i for i in graph["instances"]}
    components = {c["id"]: c for c in technology["components"]}
    outgoing, electrical = {}, []
    for edge in graph["connections"]:
        component = components[instances[edge["source"]["instance"]]["component"]]
        port = next(p for p in component["ports"] if p["name"] == edge["source"]["port"])
        if port["domain"] == "electrical":
            electrical.append(edge)
        else:
            outgoing[_endpoint(edge["source"])] = _endpoint(edge["target"])
    terminations = {_endpoint(t["port"]): t for t in graph["terminations"]}
    receivers = {r["port"]["instance"]: r for r in graph["receivers"]}
    segments, covered = [], set()
    for source in (i for i in graph["instances"] if components[i["component"]]["kind"] == "source"):
        root_target = outgoing.get((source["id"], "out"))
        if root_target is None or components[instances[root_target[0]]["component"]]["kind"] == "detector":
            _error("unsupported_optical_segment", source["id"], "a source must feed at least one supported optical component")
        pending, local_ids, internal_edges, outputs = [root_target[0]], {}, [], []
        while pending:
            iid = pending.pop(0)
            if iid in local_ids:
                continue
            if iid in covered:
                _error("unsupported_optical_segment", iid, "multiple optical source segments cannot share a component")
            instance = instances[iid]
            component = components[instance["component"]]
            if component["kind"] not in {"splitter", "waveguide", "modulator"}:
                _error("unsupported_optical_model", iid, "only splitter, waveguide and fixed-sample modulator field models are available")
            local_ids[iid] = f"i{len(local_ids)}"
            covered.add(iid)
            for port in component["ports"]:
                if port["domain"] != "optical" or port["direction"] != "out":
                    continue
                origin = (iid, port["name"])
                if origin in terminations:
                    outputs.append({"endpoint": {"instance": iid, "port": port["name"]}, "kind": "matched_termination", "binding": terminations[origin], "origin": origin})
                    continue
                target = outgoing[origin]
                target_component = components[instances[target[0]]["component"]]
                if target_component["kind"] == "detector":
                    outputs.append({"endpoint": {"instance": target[0], "port": target[1]}, "kind": "receiver_input" if target[0] in receivers else "regeneration_detector_input", "binding": receivers.get(target[0]), "origin": origin})
                else:
                    internal_edges.append((origin, target))
                    pending.append(target[0])
        if len(local_ids) > MAX_SEGMENT_INSTANCES:
            _error("optical_simulation_size_limit", source["id"], f"at most {MAX_SEGMENT_INSTANCES} optical instances per segment are supported")
        sax_netlist = {
            "instances": {local: _instance_model(instances[iid], components[instances[iid]["component"]], graph["operating"]["wavelength_nm"]) for iid, local in local_ids.items()},
            "connections": {f"{local_ids[a[0]]},{_port_name(a[1])}": f"{local_ids[b[0]]},{_port_name(b[1])}" for a, b in internal_edges},
            "ports": {"input": f"{local_ids[root_target[0]]},{_port_name(root_target[1])}"},
        }
        for index, output in enumerate(outputs):
            name = f"output{index}"
            origin = output.pop("origin")
            output["sax_port"] = name
            sax_netlist["ports"][name] = f"{local_ids[origin[0]]},{_port_name(origin[1])}"
        segments.append({
            "id": f"segment{len(segments)}", "source": {
                "endpoint": {"instance": source["id"], "port": "out"}, "role": source["role"],
                "source_node": source["source_node"], "lane": source["lane"],
                "power_mw": source["parameters"]["power_mw"], "sax_port": "input",
            },
            "netlist": sax_netlist,
            "instance_mapping": {local: {"instance": iid, "technology_component": instances[iid]["component"]} for iid, local in local_ids.items()},
            "outputs": outputs,
        })
    expected = {iid for iid, i in instances.items() if components[i["component"]]["kind"] in {"splitter", "waveguide", "modulator"}}
    if covered != expected:
        _error("unsupported_optical_segment", "graph.instances", "some optical components are not covered exactly once by source-rooted segments")
    if len(segments) > MAX_SEGMENTS or sum(len(s["netlist"]["ports"]) ** 2 for s in segments) > MAX_MATRIX_ELEMENTS:
        _error("optical_simulation_size_limit", "graph", "optical segment count or external scattering-matrix budget exceeded")
    return {"schema_version": "0.3", "kind": "sax_optical_export", "adapter_version": ADAPTER_VERSION,
            "realization_hashes": record["hashes"], "recipe": graph["recipe"],
            "operating": graph["operating"], "baseline_feasible": record["evaluation"]["feasible"],
            "technology_status": record["evaluation"]["technology_status"],
            "technology": technology,
            "scope": {**deepcopy(SCOPE), "coverage": "partial_optical_only"},
            "model_definitions": deepcopy(MODEL_DEFINITIONS), "segments": segments,
            "excluded_electrical_connections": electrical}


def _finite_number(value, path, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error("invalid_optical_simulation_option", path, "expected a finite number, not a boolean or numeric string")
    try:
        number = float(value)
    except (ValueError, OverflowError):
        _error("invalid_optical_simulation_option", path, "number exceeds finite supported range")
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        _error("invalid_optical_simulation_option", path, "expected a strictly positive finite number" if positive else "expected a nonnegative finite number")
    return number


def _load_sax():
    try:
        import sax
        import jax.numpy as jnp
    except ImportError:
        _error("optional_dependency_missing", "photonics", "SAX and its dependencies are required; install neural-physical-planner[photonics] with Python 3.11 or newer")
    except Exception as exc:
        _error("optical_engine_error", "photonics", f"SAX dependencies could not initialize ({type(exc).__name__}): {exc}")
    return sax, jnp


def _sax_models(sax, jnp):
    # These field models deliberately do not call the planner's scalar evaluator
    # or transmission helper. SAX composes their netlist connections externally.
    def splitter(*, wl=1.55, ratio=0.5, loss_db=0.0):
        attenuation = 10.0 ** (-loss_db / 20.0)
        return sax.reciprocal({("p_in", "p_out1"): jnp.sqrt(ratio) * attenuation,
                               ("p_in", "p_out2"): jnp.sqrt(1.0 - ratio) * attenuation})

    def waveguide(*, wl=1.55, length_um=0.0, loss_db_per_um=0.0, neff=2.4, ng=4.0, reference_wavelength_um=1.55):
        effective_index = neff - (wl - reference_wavelength_um) * (ng - neff) / reference_wavelength_um
        phase = 2.0 * jnp.pi * effective_index * length_um / wl
        transmission = 10.0 ** (-loss_db_per_um * length_um / 20.0) * jnp.exp(1j * phase)
        return sax.reciprocal({("p_in", "p_out"): transmission})

    def modulator(*, wl=1.55, loss_db=0.0):
        return sax.reciprocal({("p_in", "p_out"): jnp.asarray(10.0 ** (-loss_db / 20.0), dtype=complex)})

    return {"npp_splitter_v1": splitter, "npp_waveguide_v1": waveguide, "npp_modulator_full_scale_v1": modulator}


def simulate_realization(record: dict, wavelengths_nm=None, *, rtol=1e-5, atol_mw=1e-9) -> dict:
    """Compare real SAX optical powers against replayed planner powers.

    Returns status ``passed``/``failed`` for numerical agreement. Invalid input,
    infeasible operating points, absent dependencies and engine failures raise
    InputValidationError with structured diagnostics. No local solver fallback.
    """
    exported = export_optical_netlist(record)
    record = deepcopy(record)
    rtol = _finite_number(rtol, "rtol")
    atol_mw = _finite_number(atol_mw, "atol_mw")
    if rtol > MAX_RTOL or atol_mw > MAX_ATOL_MW:
        _error("invalid_optical_simulation_option", "tolerances", f"comparison tolerances are limited to rtol <= {MAX_RTOL:g} and atol_mw <= {MAX_ATOL_MW:g}")
    if wavelengths_nm is None:
        wavelengths_nm = [exported["operating"]["wavelength_nm"]]
    if not isinstance(wavelengths_nm, (list, tuple)) or not 1 <= len(wavelengths_nm) <= MAX_WAVELENGTHS:
        _error("invalid_optical_simulation_option", "wavelengths_nm", f"expected a list of 1..{MAX_WAVELENGTHS} wavelengths in nanometres")
    wavelengths = [_finite_number(w, f"wavelengths_nm.{i}", positive=True) for i, w in enumerate(wavelengths_nm)]
    if len(set(wavelengths)) != len(wavelengths):
        _error("invalid_optical_simulation_option", "wavelengths_nm", "duplicate wavelengths are not allowed")
    if not exported["baseline_feasible"]:
        _error("infeasible_optical_realization", "evaluation", "export is available for inspection, but simulation requires a feasible recorded realization")
    evaluations = []
    for index, wavelength in enumerate(wavelengths):
        graph = deepcopy(record["graph"])
        graph["operating"]["wavelength_nm"] = wavelength
        evaluation = evaluate_implementation(graph, record["technology"])
        if not evaluation["feasible"]:
            raise InputValidationError([{"code": "infeasible_optical_sweep", "path": f"wavelengths_nm.{index}", "message": f"{wavelength:g} nm is infeasible: {d['message']}", "guard": d} for d in evaluation["diagnostics"]])
        evaluations.append({r["instance"]: r for r in evaluation["ledger"]})
    sax, jnp = _load_sax()
    models = _sax_models(sax, jnp)
    samples = [{"wavelength_nm": wavelength, "comparisons": [], "passed": True} for wavelength in wavelengths]
    diagnostics = []
    for segment in exported["segments"]:
        try:
            circuit, _ = sax.circuit(segment["netlist"], models=models, backend=BACKEND,
                                        ignore_impossible_connections=False, return_type="SDict")
            for index, wavelength in enumerate(wavelengths):
                scattering = circuit(wl=wavelength / 1000.0)
                for output in segment["outputs"]:
                    pair = (output["sax_port"], segment["source"]["sax_port"])
                    coefficient = complex(scattering[pair])
                    actual = abs(coefficient) ** 2 * segment["source"]["power_mw"]
                    endpoint = output["endpoint"]
                    ledger = evaluations[index][endpoint["instance"]]
                    expected = ledger["output_powers_mw"][endpoint["port"]] if output["kind"] == "matched_termination" else ledger["input_power_mw"]
                    if not math.isfinite(actual):
                        _error("optical_engine_nonfinite", segment["id"], "SAX returned a nonfinite output power")
                    absolute_error = abs(actual - expected)
                    allowed_error = atol_mw + rtol * abs(expected)
                    passed = absolute_error <= allowed_error
                    samples[index]["comparisons"].append({
                        "segment": segment["id"], "source": segment["source"]["endpoint"],
                        "sax_port": output["sax_port"], "endpoint": endpoint, "kind": output["kind"], "binding": output["binding"],
                        "expected_power_mw": expected, "sax_power_mw": actual,
                        "absolute_error_mw": absolute_error, "relative_error": absolute_error / expected if expected else None,
                        "allowed_error_mw": allowed_error, "passed": passed,
                    })
                    if not passed:
                        samples[index]["passed"] = False
                        diagnostics.append({"code": "optical_power_mismatch", "path": f"{segment['id']}.{output['sax_port']}", "message": f"SAX and planner powers disagree at {wavelength:g} nm"})
        except InputValidationError:
            raise
        except Exception as exc:
            _error("optical_engine_error", segment["id"], f"SAX circuit composition failed ({type(exc).__name__}): {exc}")
    comparisons = [c for sample in samples for c in sample["comparisons"]]
    passed = all(sample["passed"] for sample in samples)
    return {"schema_version": "0.3", "kind": "sax_optical_comparison", "status": "passed" if passed else "failed", "passed": passed,
            "engine": {"name": "SAX", "version": metadata.version("sax"), "backend": BACKEND,
                       "jax_version": metadata.version("jax"), "adapter_version": ADAPTER_VERSION},
            "tolerances": {"rtol": rtol, "atol_mw": atol_mw, "rule": "absolute_error_mw <= atol_mw + rtol*abs(expected_power_mw)"},
            "scope": deepcopy(exported["scope"]), "export": exported, "samples": samples,
            "summary": {"segment_count": len(exported["segments"]), "wavelength_count": len(wavelengths),
                        "comparison_count": len(comparisons), "max_absolute_error_mw": max(c["absolute_error_mw"] for c in comparisons),
                        "max_relative_error": max(c["relative_error"] for c in comparisons if c["relative_error"] is not None)},
            "diagnostics": diagnostics}
