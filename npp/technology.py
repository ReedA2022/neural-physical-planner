"""Traceable scalar intensity technology contracts for the physical backend.

The bundled pack is illustrative and unreviewed. Validation verifies a declared
contract, not the accuracy or authenticity of its supporting evidence.
"""
from __future__ import annotations

import math
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from .models import (Efficiency, Finite, Identifier, InputValidationError,
                     Nonnegative, Positive, StrictModel)

C_M_PER_S = 299_792_458.0
H_J_S = 6.62607015e-34
Kind = Literal["source", "splitter", "waveguide", "detector", "modulator"]
Text = Annotated[str, Field(strict=True, min_length=1, pattern=r"\S")]


class Interval(StrictModel):
    minimum: Finite
    maximum: Finite

    @model_validator(mode="after")
    def ordered(self):
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self


class OperatingEnvelope(StrictModel):
    wavelength_nm: Interval
    temperature_c: Interval
    symbol_duration_ns: Interval
    max_input_power_mw: Positive | None = None

    @model_validator(mode="after")
    def physical_ranges(self):
        if self.wavelength_nm.minimum <= 0 or self.symbol_duration_ns.minimum <= 0:
            raise ValueError("wavelength and symbol duration must be strictly positive")
        if self.temperature_c.minimum < -273.15:
            raise ValueError("temperature cannot be below absolute zero")
        return self


class Encoding(StrictModel):
    id: Literal["normalized_intensity"] = "normalized_intensity"
    optical_quantity: Literal["power_mw"] = "power_mw"
    electrical_quantity: Literal["normalized_sample"] = "normalized_sample"
    logical_minimum: Annotated[float, Field(strict=True, ge=0, le=0, allow_inf_nan=False)] = 0.0
    logical_maximum: Annotated[float, Field(strict=True, ge=1, le=1, allow_inf_nan=False)] = 1.0
    interpretation: Literal["P(t)=full_scale_power_mw*x; 0<=x<=1"] = "P(t)=full_scale_power_mw*x; 0<=x<=1"
    energy_accounting: Literal["full_scale_per_symbol_upper_bound"] = "full_scale_per_symbol_upper_bound"


class Port(StrictModel):
    name: Identifier
    direction: Literal["in", "out"]
    domain: Literal["optical", "electrical"]
    quantity: Literal["power_mw", "normalized_sample"]
    encoding: Literal["normalized_intensity"] = "normalized_intensity"
    max_connections: Annotated[int, Field(strict=True, ge=1, le=1)] = 1

    @model_validator(mode="after")
    def compatible_quantity(self):
        expected = "power_mw" if self.domain == "optical" else "normalized_sample"
        if self.quantity != expected:
            raise ValueError(f"{self.domain} ports require quantity={expected}")
        return self


class Evidence(StrictModel):
    id: Identifier
    kind: Literal["illustrative_assumption", "model_reference", "measurement", "simulation", "standard", "review_report"]
    citation: Text
    url: str | None = None
    version: Text
    coverage: Text
    limitations: Text

    @model_validator(mode="after")
    def locator(self):
        if self.url is not None and not self.url.startswith(("https://", "http://")):
            raise ValueError("evidence URL must be an http(s) URL; local evidence belongs in the citation")
        return self


class EvidenceBindings(StrictModel):
    """Each key covers every leaf of that group; references do not calibrate values."""
    parameters: list[Identifier] = Field(min_length=1)
    operating_envelope: list[Identifier] = Field(min_length=1)
    costs: list[Identifier] = Field(min_length=1)
    model: list[Identifier] = Field(min_length=1)


class Review(StrictModel):
    status: Literal["unreviewed", "reviewed"]
    reviewer: Text | None = None
    report_source_id: Identifier | None = None
    notes: Text

    @model_validator(mode="after")
    def review_evidence(self):
        if self.status == "reviewed" and (not self.reviewer or not self.report_source_id):
            raise ValueError("reviewed status requires a named reviewer and report_source_id")
        if self.status == "unreviewed" and (self.reviewer is not None or self.report_source_id is not None):
            raise ValueError("unreviewed status cannot contain a completed review")
        return self


class Costs(StrictModel):
    area_um2: Nonnegative
    fixed_energy_pj: Nonnegative
    latency_ns: Nonnegative


class SourceParameters(StrictModel):
    minimum_output_power_mw: Positive
    maximum_output_power_mw: Positive
    wall_plug_efficiency: Efficiency
    relative_intensity_noise_rms: Nonnegative

    @model_validator(mode="after")
    def power_range(self):
        if self.minimum_output_power_mw > self.maximum_output_power_mw:
            raise ValueError("minimum output power exceeds maximum output power")
        return self


class SplitterParameters(StrictModel):
    power_ratio: Annotated[float, Field(strict=True, gt=0, lt=1, allow_inf_nan=False)]
    insertion_loss_db: Nonnegative
    ratio_tunability: Literal["fixed"] = "fixed"


class WaveguideParameters(StrictModel):
    attenuation_db_per_um: Nonnegative
    group_index: Positive
    effective_index: Positive
    width_um: Positive
    maximum_length_um: Positive


class DetectorParameters(StrictModel):
    minimum_full_scale_power_mw: Positive
    quantum_efficiency: Efficiency
    input_referred_noise_mw_rms: Nonnegative


class ModulatorParameters(StrictModel):
    insertion_loss_db: Nonnegative
    added_sample_noise_rms: Nonnegative


_PORTS = {
    "source": [("out", "out", "optical")],
    "splitter": [("in", "in", "optical"), ("out1", "out", "optical"), ("out2", "out", "optical")],
    "waveguide": [("in", "in", "optical"), ("out", "out", "optical")],
    "detector": [("in", "in", "optical"), ("out", "out", "electrical")],
    "modulator": [("carrier", "in", "optical"), ("sample", "in", "electrical"), ("out", "out", "optical")],
}


class PhysicalComponent(StrictModel):
    id: Identifier
    kind: Kind
    model_version: Literal["scalar_intensity_v1"]
    ports: list[Port] = Field(min_length=1)
    operating_envelope: OperatingEnvelope
    costs: Costs
    evidence: EvidenceBindings
    notes: Text

    @model_validator(mode="after")
    def component_contract(self):
        observed = [(p.name, p.direction, p.domain) for p in self.ports]
        if sorted(observed) != sorted(_PORTS[self.kind]):
            raise ValueError(f"{self.kind} requires exactly these named ports: {_PORTS[self.kind]}")
        if self.kind == "source":
            if self.operating_envelope.max_input_power_mw is not None:
                raise ValueError("source has no optical input; max_input_power_mw must be null")
        elif self.operating_envelope.max_input_power_mw is None:
            raise ValueError("optical input components require max_input_power_mw")
        return self


class Source(PhysicalComponent):
    kind: Literal["source"]
    parameters: SourceParameters


class Splitter(PhysicalComponent):
    kind: Literal["splitter"]
    parameters: SplitterParameters


class Waveguide(PhysicalComponent):
    kind: Literal["waveguide"]
    parameters: WaveguideParameters


class Detector(PhysicalComponent):
    kind: Literal["detector"]
    parameters: DetectorParameters

    @model_validator(mode="after")
    def receiver_range(self):
        if self.parameters.minimum_full_scale_power_mw > self.operating_envelope.max_input_power_mw:
            raise ValueError("receiver minimum full-scale power exceeds maximum input power")
        return self


class Modulator(PhysicalComponent):
    kind: Literal["modulator"]
    parameters: ModulatorParameters


Component = Annotated[Source | Splitter | Waveguide | Detector | Modulator, Field(discriminator="kind")]


class TechnologyPack(StrictModel):
    schema_version: Literal["0.3"] = "0.3"
    id: Identifier
    name: Text
    version: Text
    calibration_status: Literal["illustrative", "externally_supplied"]
    encoding: Encoding
    review: Review
    sources: list[Evidence] = Field(min_length=1)
    components: list[Component] = Field(min_length=1)
    assumptions: list[Text] = Field(min_length=1)
    excluded_effects: list[Text] = Field(min_length=1)

    @model_validator(mode="after")
    def references(self):
        ids = [source.id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence source ids must be unique")
        component_ids = [component.id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component ids must be unique")
        source_map = {source.id: source for source in self.sources}
        for component in self.components:
            for group, references in component.evidence.model_dump().items():
                if len(references) != len(set(references)):
                    raise ValueError(f"{component.id}: duplicate {group} evidence references")
                unknown = set(references) - set(ids)
                if unknown:
                    raise ValueError(f"{component.id}: unknown {group} evidence: {sorted(unknown)}")
                if group != "model" and all(source_map[ref].kind in ("model_reference", "standard", "review_report") for ref in references):
                    raise ValueError(f"{component.id}: {group} needs parameter evidence or explicit illustrative assumptions, not only a formula/review reference")
                if self.calibration_status == "illustrative" and group != "model" and not any(source_map[ref].kind == "illustrative_assumption" for ref in references):
                    raise ValueError(f"{component.id}: illustrative {group} must cite illustrative assumptions")
        if self.review.report_source_id is not None:
            report = source_map.get(self.review.report_source_id)
            if report is None or report.kind != "review_report":
                raise ValueError("review report_source_id must reference a review_report source")
        return self


def validate_technology(value: dict) -> dict:
    """Return a normalized copy or user-readable structured diagnostics."""
    try:
        return TechnologyPack.model_validate(value).model_dump(mode="json")
    except ValidationError as exc:
        diagnostics = [{"code": "technology_validation_error", "path": ".".join(str(x) for x in err["loc"]) or "technology", "message": err["msg"]}
                       for err in exc.errors(include_url=False, include_context=False)]
        raise InputValidationError(diagnostics) from exc


def load_technology(path: str | Path | None = None) -> dict:
    """Load strict JSON/YAML, defaulting to the packaged illustrative reference."""
    from .config import read_spec
    source = Path(path) if path is not None else Path(str(files("npp").joinpath("data/technology_reference.json")))
    return validate_technology(read_spec(source))


def technology_schema() -> dict:
    return TechnologyPack.model_json_schema()


def get_component(technology: dict, component_id: str) -> dict:
    """Look up a component in an already validated technology pack."""
    for component in technology["components"]:
        if component["id"] == component_id:
            return component
    raise InputValidationError([{"code": "unknown_component", "path": component_id, "message": "component does not occur in this technology pack"}])


def _finite_number(value: object) -> bool:
    """A Python integer can be finite but too large for the backend's float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _numeric_description(value: object) -> str:
    # Avoid Python's integer-to-string digit limit while constructing diagnostics.
    if isinstance(value, int) and value.bit_length() > 1024:
        return "<integer outside binary64 range>"
    return repr(value)


def _number(value: float, name: str, *, positive: bool = False) -> float:
    if not _finite_number(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")
    return float(value)


def _evaluated_number(value: float, name: str, *, positive: bool = False) -> float:
    """Do not turn a positive physical contribution into a free/ideal zero.

    Binary64 underflow is a model-range failure, just like overflow. The caller
    states whether its exact expression is strictly positive, independently of
    the rounded intermediate result. True zero inputs remain supported.
    """
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} exceeds supported finite numeric range")
    if positive and value == 0:
        raise ValueError(f"{name} underflows supported finite numeric range")
    return value


def db_transmission(loss_db: float) -> float:
    """Power transmission (field amplitude transmission is its square root)."""
    return _evaluated_number(10.0 ** (-_number(loss_db, "loss_db") / 10.0),
                             "power transmission", positive=True)


def photon_count(power_mw: float, symbol_duration_ns: float, wavelength_nm: float) -> float:
    """Mean incident photons for a constant-power rectangular symbol."""
    power = _number(power_mw, "power_mw")
    duration = _number(symbol_duration_ns, "symbol_duration_ns", positive=True)
    wavelength = _number(wavelength_nm, "wavelength_nm", positive=True)
    value = power * duration * wavelength * 1e-21 / (H_J_S * C_M_PER_S)
    return _evaluated_number(value, "photon count", positive=power > 0)


def check_operating_point(component: dict, *, wavelength_nm: float, temperature_c: float, symbol_duration_ns: float,
                          input_power_mw: float | None = None) -> list[dict]:
    """Return infeasibility diagnostics; never silently extrapolate an envelope."""
    envelope = component["operating_envelope"]
    diagnostics = []
    for name, value in (("wavelength_nm", wavelength_nm), ("temperature_c", temperature_c), ("symbol_duration_ns", symbol_duration_ns)):
        bounds = envelope[name]
        if not _finite_number(value) or not bounds["minimum"] <= value <= bounds["maximum"]:
            diagnostics.append({"code": "outside_operating_envelope", "path": f"{component['id']}.{name}", "message": f"{_numeric_description(value)} outside [{bounds['minimum']}, {bounds['maximum']}]"})
    if input_power_mw is not None:
        maximum = envelope["max_input_power_mw"]
        if not _finite_number(input_power_mw) or input_power_mw < 0 or maximum is None or input_power_mw > maximum:
            diagnostics.append({"code": "input_power_out_of_range", "path": component["id"], "message": f"input power {_numeric_description(input_power_mw)} must be nonnegative and <= {maximum!r} mW"})
    return diagnostics


def evaluate_component(component: dict, *, wavelength_nm: float = 1550.0, temperature_c: float = 25.0,
                       symbol_duration_ns: float = 1.0, input_power_mw: float | None = None,
                       source_power_mw: float | None = None, length_um: float = 0.0) -> dict:
    """Evaluate one validated instance at full scale, with explicit unit-bearing keys.

    Power is the calibration full scale (x=1), not an assertion that every NN
    sample equals one. Receiver noise is normalized to that full scale. Source
    and modulator noise are separate contributions for a graph to propagate.
    """
    diagnostics = check_operating_point(component, wavelength_nm=wavelength_nm, temperature_c=temperature_c,
                                        symbol_duration_ns=symbol_duration_ns, input_power_mw=input_power_mw)
    if diagnostics:
        raise InputValidationError(diagnostics)
    kind, p, costs = component["kind"], component["parameters"], component["costs"]
    if kind != "source" and input_power_mw is None:
        raise ValueError(f"{kind} requires input_power_mw")
    if kind != "source" and source_power_mw is not None:
        raise ValueError("source_power_mw only applies to sources")
    if kind != "waveguide" and length_um != 0:
        raise ValueError("length_um only applies to waveguides")
    result = {"output_powers_mw": {}, "area_um2": costs["area_um2"], "energy_pj": costs["fixed_energy_pj"],
              "latency_ns": costs["latency_ns"], "added_noise_variance": 0.0}
    if kind == "source":
        power = _number(source_power_mw, "source_power_mw", positive=True)
        if not p["minimum_output_power_mw"] <= power <= p["maximum_output_power_mw"]:
            raise ValueError("source power outside declared output range")
        result["output_powers_mw"] = {"out": power}
        result["energy_pj"] += _evaluated_number(power * symbol_duration_ns / p["wall_plug_efficiency"],
                                                 "source energy", positive=True)
        result["added_noise_variance"] = _evaluated_number(
            p["relative_intensity_noise_rms"] * p["relative_intensity_noise_rms"],
            "source noise variance", positive=p["relative_intensity_noise_rms"] > 0)
    elif kind == "splitter":
        delivered = input_power_mw * db_transmission(p["insertion_loss_db"])
        result["output_powers_mw"] = {"out1": delivered * p["power_ratio"], "out2": delivered * (1 - p["power_ratio"])}
    elif kind == "waveguide":
        length = _number(length_um, "length_um")
        if length > p["maximum_length_um"]:
            raise ValueError("waveguide length outside declared range")
        loss = _evaluated_number(p["attenuation_db_per_um"] * length, "waveguide loss",
                                  positive=p["attenuation_db_per_um"] > 0 and length > 0)
        result["output_powers_mw"] = {"out": input_power_mw * db_transmission(loss)}
        result["latency_ns"] += _evaluated_number(p["group_index"] * length * 1e3 / C_M_PER_S,
                                                  "waveguide delay", positive=length > 0)
        result["area_um2"] += _evaluated_number(length * p["width_um"], "waveguide area", positive=length > 0)
    elif kind == "detector":
        if input_power_mw < p["minimum_full_scale_power_mw"]:
            raise ValueError("receiver full-scale power is below declared sensitivity")
        electrons = p["quantum_efficiency"] * photon_count(input_power_mw, symbol_duration_ns, wavelength_nm)
        if electrons <= 0 or not math.isfinite(electrons):
            raise ValueError("photoelectron count exceeds supported numeric range")
        noise_ratio = p["input_referred_noise_mw_rms"] / input_power_mw
        electronic_variance = _evaluated_number(noise_ratio * noise_ratio, "detector electronic noise variance",
                                                 positive=p["input_referred_noise_mw_rms"] > 0)
        result["added_noise_variance"] = 1.0 / electrons + electronic_variance
        result["mean_photoelectrons"] = electrons
        result["electrical_output_full_scale"] = 1.0
    elif kind == "modulator":
        result["output_powers_mw"] = {"out": input_power_mw * db_transmission(p["insertion_loss_db"])}
        result["added_noise_variance"] = _evaluated_number(p["added_sample_noise_rms"] * p["added_sample_noise_rms"],
                                                          "modulator noise variance", positive=p["added_sample_noise_rms"] > 0)
    # Every supported optical map has strictly positive transmission for finite
    # loss and positive input. A zero output would silently erase positive power.
    for output in result["output_powers_mw"].values():
        _evaluated_number(output, "output optical power", positive=kind == "source" or input_power_mw > 0)
    if not all(math.isfinite(result[key]) for key in ("area_um2", "energy_pj", "latency_ns", "added_noise_variance")):
        raise ValueError("component evaluation exceeds finite numeric range")
    return result
