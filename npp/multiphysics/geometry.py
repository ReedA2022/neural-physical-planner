"""Bounded placement/route geometry, separate from all technology equations.

Lengths are geometric measurements in micrometres. They imply no propagation
speed, loss, legal physical crossing, thermal coupling or fabrication readiness.
"""
from __future__ import annotations

import math
import heapq
from typing import Literal

from pydantic import Field, model_validator

from .contracts import (Carrier, Contract, Count, Identifier, Interval, Nonnegative,
                        Positive, Text, canonical_json)


MAX_ROUTE_SEGMENTS = 1024
MAX_FOOTPRINT_SEGMENT_WORK = 1_000_000


class GeometryLimitError(ValueError):
    pass


class Point(Contract):
    x_um: Nonnegative
    y_um: Nonnegative


class Region(Contract):
    id: Identifier
    x_um: Nonnegative
    y_um: Nonnegative
    width_um: Positive
    height_um: Positive
    substrate: Text
    package: Text
    temperature_c: float = Field(strict=True, ge=-273.15, allow_inf_nan=False)
    services: tuple[Identifier, ...]


class GeometryPort(Contract):
    id: Identifier
    side: Literal["north", "south", "east", "west"]
    offset_um: Nonnegative
    capacity: Count = 1


class Placement(Contract):
    id: Identifier
    component_id: Identifier
    kind: Literal["compute", "storage", "interface", "pump", "control", "merge", "source", "buffer"]
    region: Identifier
    x_um: Nonnegative
    y_um: Nonnegative
    width_um: Positive
    height_um: Positive
    orientation: int = Field(default=0, strict=True)
    substrate: Text
    package: Text
    temperature_c: Interval
    required_services: tuple[Identifier, ...]
    ports: tuple[GeometryPort, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def ports_fit(self):
        if type(self.orientation) is not int or self.orientation not in (0, 90, 180, 270):
            raise ValueError("orientation must be an integer quarter-turn")
        if self.temperature_c.minimum < -273.15:
            raise ValueError("temperature envelope below absolute zero")
        if len({p.id for p in self.ports}) != len(self.ports):
            raise ValueError("duplicate placement port")
        for port in self.ports:
            bound = self.width_um if port.side in ("north", "south") else self.height_um
            if port.offset_um > bound:
                raise ValueError("port offset exceeds its footprint edge")
        return self


class Endpoint(Contract):
    placement: Identifier
    port: Identifier


class Route(Contract):
    id: Identifier
    source: Endpoint
    target: Endpoint
    carrier: Carrier
    points: tuple[Point, ...] = Field(min_length=2, max_length=512)
    capacity: Count
    usage: Count
    direction: Literal["directional", "bidirectional"] = "directional"
    interface_id: Identifier | None = None
    crossing_policy: Literal["forbidden", "declared_component"] = "forbidden"
    crossing_component_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def route_shape(self):
        if self.source == self.target:
            raise ValueError("route must connect distinct endpoint ports")
        if self.usage > self.capacity:
            raise ValueError("route usage exceeds declared channel capacity")
        if self.crossing_policy == "forbidden" and self.crossing_component_ids:
            raise ValueError("crossing components cannot be declared under forbidden policy")
        if len(set(self.crossing_component_ids)) != len(self.crossing_component_ids):
            raise ValueError("crossing component IDs must be distinct")
        for a, b in zip(self.points, self.points[1:]):
            if a == b or (a.x_um != b.x_um and a.y_um != b.y_um):
                raise ValueError("routes require nonzero axis-aligned segments")
        for a, b, c in zip(self.points, self.points[1:], self.points[2:]):
            if a.x_um == b.x_um == c.x_um and (b.y_um - a.y_um) * (c.y_um - b.y_um) < 0:
                raise ValueError("route cannot backtrack over its own segment")
            if a.y_um == b.y_um == c.y_um and (b.x_um - a.x_um) * (c.x_um - b.x_um) < 0:
                raise ValueError("route cannot backtrack over its own segment")
        return self


class Geometry(Contract):
    schema_version: Literal["npp-geometry-1"] = "npp-geometry-1"
    routing_model: Literal["explicit_routes", "shared_electrical_bus_v1"] = "explicit_routes"
    regions: tuple[Region, ...] = Field(min_length=1, max_length=64)
    placements: tuple[Placement, ...] = Field(min_length=1, max_length=2048)
    routes: tuple[Route, ...] = Field(max_length=4096)
    assumptions: tuple[Text, ...] = Field(min_length=1)


def _box(placement):
    width, height = placement.width_um, placement.height_um
    if placement.orientation in (90, 270):
        width, height = height, width
    result = (placement.x_um, placement.y_um, placement.x_um + width, placement.y_um + height)
    if not all(math.isfinite(x) for x in result) or result[2] <= result[0] or result[3] <= result[1]:
        raise ArithmeticError("placement coordinates exceed supported precision/range")
    return result


def _port_position(placement, port_name):
    port = next((p for p in placement.ports if p.id == port_name), None)
    if port is None:
        raise ValueError(f"Placement {placement.id} has no port {port_name}")
    w, h = placement.width_um, placement.height_um
    if port.side == "north":
        x, y = port.offset_um, h
    elif port.side == "south":
        x, y = port.offset_um, 0.0
    elif port.side == "east":
        x, y = w, port.offset_um
    else:
        x, y = 0.0, port.offset_um
    if placement.orientation == 90:
        x, y = h - y, x
    elif placement.orientation == 180:
        x, y = w - x, h - y
    elif placement.orientation == 270:
        x, y = y, w - x
    return placement.x_um + x, placement.y_um + y


def _boxes_overlap(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _penetrates(a, b, box):
    if a.y_um == b.y_um:
        return box[1] < a.y_um < box[3] and max(min(a.x_um, b.x_um), box[0]) < min(max(a.x_um, b.x_um), box[2])
    return box[0] < a.x_um < box[2] and max(min(a.y_um, b.y_um), box[1]) < min(max(a.y_um, b.y_um), box[3])


def _intersection(a, b, c, d):
    ah, ch = a.y_um == b.y_um, c.y_um == d.y_um
    if ah == ch:
        same_line = a.y_um == c.y_um if ah else a.x_um == c.x_um
        if not same_line:
            return None
        a1, a2 = sorted((a.x_um, b.x_um) if ah else (a.y_um, b.y_um))
        c1, c2 = sorted((c.x_um, d.x_um) if ah else (c.y_um, d.y_um))
        if max(a1, c1) < min(a2, c2):
            return "overlap"
        return None
    if not ah:
        a, b, c, d = c, d, a, b
    x, y = c.x_um, a.y_um
    if min(a.x_um, b.x_um) <= x <= max(a.x_um, b.x_um) and min(c.y_um, d.y_um) <= y <= max(c.y_um, d.y_um):
        return (x, y)
    return None


def validate_geometry(data):
    """Return strict normalized data and independently derived geometric facts."""
    try:
        geometry = Geometry.model_validate(data) if isinstance(data, Geometry) else Geometry.model_validate_json(canonical_json(data))
        total_segments = sum(len(route.points) - 1 for route in geometry.routes)
        if total_segments > MAX_ROUTE_SEGMENTS or total_segments * len(geometry.placements) > MAX_FOOTPRINT_SEGMENT_WORK:
            raise GeometryLimitError("geometry exceeds bounded work: at most 1024 total segments and 1M footprint-segment comparisons")
        regions = {r.id: r for r in geometry.regions}
        placements = {p.id: p for p in geometry.placements}
        if len(regions) != len(geometry.regions) or len(placements) != len(geometry.placements) or len({r.id for r in geometry.routes}) != len(geometry.routes):
            raise ValueError("duplicate region, placement or route id")
        region_boxes = {}
        for region in geometry.regions:
            box = (region.x_um, region.y_um, region.x_um + region.width_um, region.y_um + region.height_um)
            if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ArithmeticError("region exceeds finite coordinate range")
            if any(_boxes_overlap(box, other) for other in region_boxes.values()):
                raise ValueError("regions overlap in this planar geometry model")
            region_boxes[region.id] = box
        boxes = {}
        for placement in geometry.placements:
            if placement.region not in regions:
                raise ValueError("placement references an unknown region")
            region = regions[placement.region]
            if placement.substrate != region.substrate or placement.package != region.package:
                raise ValueError(f"Placement {placement.id} has incompatible substrate/package")
            if not placement.temperature_c.minimum <= region.temperature_c <= placement.temperature_c.maximum:
                raise ValueError(f"Placement {placement.id} temperature envelope violated")
            if not set(placement.required_services) <= set(region.services):
                raise ValueError(f"Placement {placement.id} requires unavailable services")
            box = _box(placement)
            bounds = region_boxes[placement.region]
            if box[0] < bounds[0] or box[1] < bounds[1] or box[2] > bounds[2] or box[3] > bounds[3]:
                raise ValueError(f"Placement {placement.id} is outside its declared region")
            if any(_boxes_overlap(box, other) for other in boxes.values()):
                raise ValueError(f"Placement {placement.id} overlaps another footprint")
            boxes[placement.id] = box
        derived, endpoint_usage = [], {}
        for route in geometry.routes:
            if route.source.placement not in placements or route.target.placement not in placements:
                raise ValueError("route endpoint references absent placement")
            source, target = placements[route.source.placement], placements[route.target.placement]
            if any(name not in placements or placements[name].kind != "interface" for name in route.crossing_component_ids):
                raise ValueError("every declared crossing must identify a placed interface component")
            actual_start = (route.points[0].x_um, route.points[0].y_um)
            actual_end = (route.points[-1].x_um, route.points[-1].y_um)
            if actual_start != _port_position(source, route.source.port) or actual_end != _port_position(target, route.target.port):
                raise ValueError(f"Route {route.id} endpoints do not meet declared oriented ports")
            if source.region != target.region and route.interface_id is None:
                raise ValueError("inter-region routes require an explicit interface/link identity")
            if source.region == target.region:
                rb = region_boxes[source.region]
                if any(not rb[0] <= p.x_um <= rb[2] or not rb[1] <= p.y_um <= rb[3] for p in route.points):
                    raise ValueError("local route leaves its physical region")
            for a, b in zip(route.points, route.points[1:]):
                if any(_penetrates(a, b, box) and name not in route.crossing_component_ids for name, box in boxes.items()):
                    raise ValueError(f"Route {route.id} passes through a component footprint")
            for endpoint in (route.source, route.target):
                key = (endpoint.placement, endpoint.port)
                endpoint_usage[key] = endpoint_usage.get(key, 0) + route.usage
                port = next(p for p in placements[endpoint.placement].ports if p.id == endpoint.port)
                if endpoint_usage[key] > port.capacity:
                    raise ValueError("connected routes exceed physical port capacity")
            segments = list(zip(route.points, route.points[1:]))
            for index, (a, b) in enumerate(segments):
                for c, d in segments[index + 2:]:
                    if _intersection(a, b, c, d) is not None:
                        raise ValueError("self-intersecting routes require an explicitly decomposed interface graph")
            lengths = [abs(b.x_um - a.x_um) + abs(b.y_um - a.y_um) for a, b in segments]
            length = math.fsum(lengths)
            if not math.isfinite(length):
                raise ArithmeticError("route length exceeds finite range")
            bends = sum((a.x_um == b.x_um) != (b.x_um == c.x_um) for a, b, c in zip(route.points, route.points[1:], route.points[2:]))
            derived.append(dict(id=route.id, length_um=length, bends=bends, crossings=0, capacity=route.capacity,
                                usage=route.usage, carrier=route.carrier, direction=route.direction, source=route.source.model_dump(mode="json"),
                                target=route.target.model_dump(mode="json"), interface_id=route.interface_id))
        crossing_events = []
        used_crossings = {route.id: set() for route in geometry.routes}
        for i, route in enumerate(geometry.routes):
            for j in range(i + 1, len(geometry.routes)):
                other = geometry.routes[j]
                intersections = set()
                for a, b in zip(route.points, route.points[1:]):
                    for c, d in zip(other.points, other.points[1:]):
                        location = _intersection(a, b, c, d)
                        if location == "overlap":
                            raise ValueError("route segments overlap; multiplexing needs one explicit capacity-bearing shared route")
                        if location is None:
                            continue
                        shared_endpoints = set()
                        for x in (route.source, route.target):
                            for y in (other.source, other.target):
                                if x == y:
                                    shared_endpoints.add(_port_position(placements[x.placement], x.port))
                        if location not in shared_endpoints:
                            intersections.add(location)
                if intersections:
                    if route.crossing_policy != "declared_component" or other.crossing_policy != "declared_component":
                        raise ValueError("geometric route crossing requires explicitly declared crossing components")
                    common = set(route.crossing_component_ids) & set(other.crossing_component_ids)
                    if len(common) < len(intersections):
                        raise ValueError("crossing routes lack a shared declared component for every crossing")
                    for component_id, location in zip(sorted(common), sorted(intersections)):
                        if component_id not in placements or placements[component_id].kind != "interface":
                            raise ValueError("crossing component must be a placed interface")
                        box = boxes[component_id]
                        if location != ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2):
                            raise ValueError("declared crossing component must own the actual intersection location")
                        used_crossings[route.id].add(component_id)
                        used_crossings[other.id].add(component_id)
                        crossing_events.append(dict(routes=[route.id, other.id], component_id=component_id, location_um=list(location)))
                    derived[i]["crossings"] += len(intersections)
                    derived[j]["crossings"] += len(intersections)
        for route in geometry.routes:
            if used_crossings[route.id] != set(route.crossing_component_ids):
                raise ValueError("unused crossing declaration cannot bypass footprint collision checks")
        normalized = geometry.model_dump(mode="json")
        if geometry.routing_model == "shared_electrical_bus_v1":
            _validate_bus_topology(normalized, derived)
        return dict(status="model_feasible", diagnostics=[], geometry=normalized,
                    routes=derived, crossings=crossing_events,
                    scope="Planar rectilinear geometry only; route loss/delay and crossing-device physics require registered backends")
    except GeometryLimitError as exc:
        return dict(status="unsupported", diagnostics=[dict(code="geometry_work_limit", message=str(exc))])
    except ArithmeticError as exc:
        return dict(status="numerical_failure", diagnostics=[dict(code="geometry_numeric_range", message=str(exc))])
    except (ValueError, TypeError, KeyError) as exc:
        return dict(status="invalid", diagnostics=[dict(code="invalid_geometry", message=str(exc))])


def grid_geometry(component_ids, edges=(), region_id="chip", footprint_um=(10.0, 10.0), gap_um=10.0):
    """Deterministic illustrative grid for isolated blocks or a forward chain.

    Edges are (source_id,target_id[,carrier]) tuples. Each edge receives distinct
    perimeter ports and a separate routing track; the validator rejects routing
    collisions. Arbitrary graphs need an explicitly supplied geometry if this
    simple constructor cannot route them legally. All footprints are hypotheses.
    """
    if type(component_ids) not in (list, tuple) or not 1 <= len(component_ids) <= 2048 or len(set(component_ids)) != len(component_ids):
        raise ValueError("grid needs 1..2048 unique component identifiers")
    if type(gap_um) not in (int, float) or isinstance(gap_um, bool) or not math.isfinite(gap_um) or gap_um <= 0:
        raise ValueError("grid gap must be finite and positive")
    if len(footprint_um) != 2 or any(type(v) not in (int, float) or isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in footprint_um):
        raise ValueError("grid footprint dimensions must be finite and positive")
    width, height = map(float, footprint_um)
    edge_records = []
    ports = {name: [] for name in component_ids}
    for index, edge in enumerate(edges):
        if len(edge) not in (2, 3) or edge[0] not in ports or edge[1] not in ports or edge[0] == edge[1]:
            raise ValueError("grid edge must join distinct declared components")
        edge_records.append((edge[0], edge[1], edge[2] if len(edge) == 3 else "electrical", index))
        ports[edge[0]].append((f"out{index}", index))
        ports[edge[1]].append((f"in{index}", index))
    placements = []
    positions = {}
    for index, name in enumerate(component_ids):
        x, y = gap_um + index * (width + gap_um), gap_um
        local_ports = []
        for port_index, (port_name, _) in enumerate(ports[name]):
            offset = width * (port_index + 1) / (len(ports[name]) + 1)
            local_ports.append(dict(id=port_name, side="north", offset_um=offset, capacity=1))
            positions[(name, port_name)] = (x + offset, y + height)
        if not local_ports:
            local_ports = [dict(id="unused", side="north", offset_um=width / 2, capacity=1)]
        placements.append(dict(id=name, component_id=name, kind="compute", region=region_id, x_um=x, y_um=y,
                               width_um=width, height_um=height, orientation=0,
                               substrate="illustrative_shared_substrate", package="illustrative_shared_package",
                               temperature_c=dict(minimum=0.0, maximum=100.0), required_services=["power"], ports=local_ports))
    routes = []
    for source, target, carrier, index in edge_records:
        a, b = positions[(source, f"out{index}")], positions[(target, f"in{index}")]
        track_y = gap_um + height + (index + 1) * gap_um
        routes.append(dict(id=f"route{index}", source=dict(placement=source, port=f"out{index}"),
                           target=dict(placement=target, port=f"in{index}"), carrier=carrier,
                           points=[dict(x_um=a[0], y_um=a[1]), dict(x_um=a[0], y_um=track_y),
                                   dict(x_um=b[0], y_um=track_y), dict(x_um=b[0], y_um=b[1])],
                           capacity=1, usage=1))
    data = dict(schema_version="npp-geometry-1", regions=[dict(id=region_id, x_um=0.0, y_um=0.0,
                    width_um=len(component_ids) * (width + gap_um) + gap_um,
                    height_um=height + (len(edge_records) + 3) * gap_um,
                    substrate="illustrative_shared_substrate", package="illustrative_shared_package", temperature_c=25.0,
                    services=["power"])], placements=placements, routes=routes,
                assumptions=["Hypothetical footprints and shared integration region; no fabrication claim",
                             "Geometric route distances require separate technology-specific loss and delay models"])
    checked = validate_geometry(data)
    if checked["status"] != "model_feasible":
        raise ValueError("Simple grid cannot legally route this graph: " + checked["diagnostics"][0]["message"])
    return checked["geometry"]


def bus_geometry(component_ids, region_id="chip", substrate="illustrative_shared_substrate",
                 package="illustrative_shared_package", temperature_c=25.0, services=("power",),
                 footprints=None, kinds=None, gap_um=10.0):
    """A single shared, explicitly bidirectional electrical bus with wire taps.

    Junctions are ordinary electrical interconnect abstractions, not photonic
    crossings or converters. Schedule every transfer on the shared link resource.
    bus_distances includes internal junction traversal for per-length accounting.
    """
    if type(component_ids) not in (list, tuple) or not 1 <= len(component_ids) <= 128 or len(set(component_ids)) != len(component_ids):
        raise ValueError("bus geometry supports 1..128 unique resource placements")
    if type(gap_um) not in (int, float) or isinstance(gap_um, bool) or not math.isfinite(gap_um) or gap_um <= 1:
        raise ValueError("bus gap must be finite and greater than the 1 um junction width")
    footprints, kinds = footprints or {}, kinds or {}
    widths, heights = {}, {}
    for name in component_ids:
        footprint = footprints.get(name, (10.0, 10.0))
        if len(footprint) != 2 or any(type(v) not in (int, float) or isinstance(v, bool) or not math.isfinite(v) or v <= 1 for v in footprint):
            raise ValueError("bus footprints require finite dimensions greater than 1 um")
        widths[name], heights[name] = map(float, footprint)
    bus_y = max(heights.values()) + 2 * gap_um
    placements, routes, junctions = [], [], []
    x = float(gap_um)
    common = dict(region=region_id, orientation=0, substrate=substrate, package=package,
                  temperature_c=dict(minimum=temperature_c, maximum=temperature_c), required_services=list(services))
    for index, name in enumerate(component_ids):
        width, height = widths[name], heights[name]
        center_x = x + width / 2
        placements.append(dict(common, id=name, component_id=name, kind=kinds.get(name, "compute"),
                               x_um=x, y_um=float(gap_um), width_um=width, height_um=height,
                               ports=[dict(id="bus", side="north", offset_um=width / 2, capacity=1)]))
        junction = f"bus_junction_{index}"
        if junction in component_ids:
            raise ValueError("resource identifier collides with generated bus junction")
        placements.append(dict(common, id=junction, component_id=junction, kind="interface",
                               x_um=center_x - 0.5, y_um=bus_y - 0.5, width_um=1.0, height_um=1.0,
                               ports=[dict(id="south", side="south", offset_um=0.5, capacity=1),
                                      dict(id="west", side="west", offset_um=0.5, capacity=1),
                                      dict(id="east", side="east", offset_um=0.5, capacity=1)]))
        junctions.append((junction, center_x))
        routes.append(dict(id=f"bus_tap_{index}", source=dict(placement=name, port="bus"),
                           target=dict(placement=junction, port="south"), carrier="electrical", direction="bidirectional",
                           points=[dict(x_um=center_x, y_um=gap_um + height), dict(x_um=center_x, y_um=bus_y - 0.5)], capacity=1, usage=1))
        x += width + gap_um
    for index, ((left, lx), (right, rx)) in enumerate(zip(junctions, junctions[1:])):
        routes.append(dict(id=f"bus_spine_{index}", source=dict(placement=left, port="east"),
                           target=dict(placement=right, port="west"), carrier="electrical", direction="bidirectional",
                           points=[dict(x_um=lx + 0.5, y_um=bus_y), dict(x_um=rx - 0.5, y_um=bus_y)], capacity=1, usage=1))
    geometry = dict(schema_version="npp-geometry-1", routing_model="shared_electrical_bus_v1", regions=[dict(id=region_id, x_um=0.0, y_um=0.0,
                    width_um=x, height_um=bus_y + gap_um, substrate=substrate, package=package,
                    temperature_c=temperature_c, services=list(services))], placements=placements, routes=routes,
                    assumptions=["Hypothetical electrical shared-bus placement; no photonic cross-domain routing is implied",
                                 "Every transfer must reserve the same shared link capacity",
                                 "Ordinary wire junctions have 1 um footprints; internal path lengths are included by bus_distances",
                                 "Technology-specific per-length loss, energy and delay must be evaluated separately"])
    checked = validate_geometry(geometry)
    if checked["status"] != "model_feasible":
        raise ValueError("Cannot build bounded bus: " + checked["diagnostics"][0]["message"])
    return checked["geometry"]


def _validate_bus_topology(geometry, derived_routes):
    """Validate the only built-in forwarding mechanism accepted by bus_distances.

    Ordinary resource components are leaves. Only unit, centered, unrotated
    electrical wire junctions connect distinct ports; their half-width/height
    traversal is what the distance accumulator explicitly charges.
    """
    if geometry["routing_model"] != "shared_electrical_bus_v1":
        raise ValueError("bus distances require the declared shared_electrical_bus_v1 routing model")
    placements = {p["id"]: p for p in geometry["placements"]}
    junctions = {name for name in placements if name.startswith("bus_junction_")}
    resources = set(placements) - junctions
    if not 1 <= len(resources) <= 128 or len(junctions) != len(resources) or len(geometry["regions"]) != 1:
        raise ValueError("shared bus requires one region and one explicit wire junction per resource")
    expected_junction_ports = {"south": ("south", 0.5), "west": ("west", 0.5), "east": ("east", 0.5)}
    for name, placement in placements.items():
        if placement["orientation"] != 0 or placement["component_id"] != name:
            raise ValueError("shared bus forwarding uses unrotated, explicitly identified placements")
        ports = {port["id"]: port for port in placement["ports"]}
        if any(port["capacity"] != 1 for port in ports.values()):
            raise ValueError("shared_electrical_bus_v1 has one explicitly modeled channel per port")
        if name in junctions:
            if placement["kind"] != "interface" or placement["width_um"] != 1.0 or placement["height_um"] != 1.0 or set(ports) != set(expected_junction_ports):
                raise ValueError("bus forwarding requires unit wire-junction interfaces with south/west/east ports")
            if any((ports[port_id]["side"], ports[port_id]["offset_um"]) != expected for port_id, expected in expected_junction_ports.items()):
                raise ValueError("bus junction ports must be centered for declared internal traversal lengths")
        elif len(ports) != 1 or "bus" not in ports or ports["bus"]["side"] != "north" or ports["bus"]["offset_um"] != placement["width_um"] / 2:
            raise ValueError("resource placements are terminal bus leaves; internal forwarding through compute/storage is not modeled")
    ordered_junctions = sorted(junctions, key=lambda name: placements[name]["x_um"])
    if len({placements[name]["y_um"] for name in junctions}) != 1:
        raise ValueError("shared bus wire junctions must lie on one horizontal spine")
    expected_spines = {frozenset(((a, "east"), (b, "west"))) for a, b in zip(ordered_junctions, ordered_junctions[1:])}
    actual_spines, tapped_resources, tapped_junctions = set(), set(), set()
    for route, derived in zip(geometry["routes"], derived_routes):
        if route["carrier"] != "electrical" or route["direction"] != "bidirectional":
            raise ValueError("bus distances require explicitly bidirectional electrical routes")
        if route["capacity"] != 1 or route["usage"] != 1 or len(route["points"]) != 2 or route["interface_id"] is not None or route["crossing_component_ids"]:
            raise ValueError("shared bus permits only straight single-channel taps/spine, without hidden conversion or crossings")
        endpoints = [(route["source"]["placement"], route["source"]["port"]),
                     (route["target"]["placement"], route["target"]["port"])]
        a, b = endpoints[0][0], endpoints[1][0]
        if a in junctions and b in junctions:
            pair = frozenset(endpoints)
            if pair not in expected_spines or pair in actual_spines:
                raise ValueError("bus spine must connect adjacent east/west junction ports exactly once")
            actual_spines.add(pair)
        elif (a in resources) != (b in resources):
            resource_endpoint, junction_endpoint = endpoints if a in resources else list(reversed(endpoints))
            resource_id, resource_port = resource_endpoint
            junction_id, junction_port = junction_endpoint
            if resource_port != "bus" or junction_port != "south" or resource_id in tapped_resources or junction_id in tapped_junctions:
                raise ValueError("each terminal resource needs exactly one dedicated south-port junction tap")
            if route["points"][0]["x_um"] != route["points"][1]["x_um"]:
                raise ValueError("bus resource taps must be vertical")
            tapped_resources.add(resource_id)
            tapped_junctions.add(junction_id)
        else:
            raise ValueError("resource-to-resource forwarding is not part of the shared bus model")
    if actual_spines != expected_spines or tapped_resources != resources or tapped_junctions != junctions:
        raise ValueError("shared bus must contain the complete dedicated taps and connected adjacent-junction spine")
    return placements, junctions, resources


def bus_distances(geometry):
    """Shortest electrical path distances, including each junction's interior.

    Returns {source_resource: {target_resource: distance_um}}. Only the explicit
    generated bus abstraction is accepted; this cannot turn directed interfaces
    into reversible routes or infer inter-domain conversion availability.
    """
    checked = validate_geometry(geometry)
    if checked["status"] != "model_feasible":
        raise ValueError("Cannot measure invalid bus geometry: " + checked["diagnostics"][0]["message"])
    placements, junctions, resources = _validate_bus_topology(checked["geometry"], checked["routes"])
    adjacency = {name: [] for name in placements}
    for route in checked["routes"]:
        if route["carrier"] != "electrical" or route["direction"] != "bidirectional":
            raise ValueError("bus distances require explicitly bidirectional electrical routes")
        a, b = route["source"]["placement"], route["target"]["placement"]
        distance = route["length_um"] + (0.5 if a in junctions else 0.0) + (0.5 if b in junctions else 0.0)
        if not math.isfinite(distance):
            raise ArithmeticError("bus distance exceeds finite range")
        adjacency[a].append((b, distance))
        adjacency[b].append((a, distance))
    result = {}
    for source in sorted(resources):
        distances = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            distance, current = heapq.heappop(queue)
            if distance != distances[current]:
                continue
            for neighbor, weight in adjacency[current]:
                candidate = distance + weight
                if not math.isfinite(candidate):
                    raise ArithmeticError("bus distance exceeds finite range")
                if candidate < distances.get(neighbor, math.inf):
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        if not resources <= distances.keys():
            raise ValueError("bus resources are disconnected")
        result[source] = {target: distances[target] for target in sorted(resources)}
    return result
