"""Original carrier selection from complete canonical metric model outputs."""
from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np

from rexgraph.cpp_algebraic import _carrier, _key, _output_declaration, _polynomial_axis
from rexgraph.cpp_algebraic_grade import algebraic_grade_observation
from rexgraph.cpp_algebraic_grade_model import _cancel
from rexgraph.cpp_algebraic_metric import restore_algebraic_grade_metric_observation
from rexgraph.cpp_algebraic_query import _closed
from rexgraph.cpp_algebraic_stalk import _implementation, _policy, _same_field
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_state import ModelOutput, freeze_tree, thaw_tree
from rexgraph.source_binding import CppAlgebraicSourceReference, _sha
from rexgraph.value_codec import pack_value, unpack_value

__all__ = ["AlgebraicMetricStalkSelection", "prepare_algebraic_metric_stalks"]
_PROFILE = "algebraic-grade-metric-stalk-v1"
_PARENT = "algebraic-grade-metric-action-v1"


def _digest(source, receipt):
    return sha256(pack_value({"profile": _PROFILE, "source": source.as_record(), "cell": receipt["cell"],
        "origin_model": receipt["origin_model"], "origin_output": receipt["origin_output"]})).hexdigest()


def _vector_parent(output):
    if not isinstance(output, ModelOutput) or output.method != _PARENT:
        raise TypeError("metric stalk selection requires a complete metric model output")
    output.check_state()
    if output.execution_receipt["operation"] not in {"hodge", "coboundary"}:
        raise TypeError("metric scalar outputs have no original cell stalks")
    return thaw_tree(output.execution_receipt["observation"])


def _parent(receipt):
    raw, maximum = receipt["origin"], _key(receipt["origin_max_bytes"])
    if type(raw) is not bytes or len(raw) > maximum:
        raise ValueError("metric stalk origin exceeds its retained byte admission")
    output = unpack_model(unpack_value(raw))
    _vector_parent(output)
    return output


def validate_metric_stalk_output(output):
    """Check the exact selected carrier, result slice and complete metric producer."""
    r = thaw_tree(output.execution_receipt)
    _closed(r, ("profile", "engine", "kind", "output_carrier", "native_shape", "coefficient_basis",
        "origin_model", "origin_output", "origin", "origin_max_bytes", "operation", "cell", "source_slice",
        "execution_limits", "declared_limits", "max_retained_coordinates", "declared_max_retained_coordinates"))
    if output.method != _PROFILE or r["profile"] != _PROFILE or output.arithmetic != "algebraic":
        raise ValueError("unsupported metric stalk output profile")
    if not isinstance(output.source, CppAlgebraicSourceReference):
        raise TypeError("metric stalk output requires its complete algebraic source")
    _sha(r["origin_model"]); _sha(r["origin_output"])
    parent = _parent(r); p = parent.execution_receipt
    observation, detail = p["observation"], p["observation"]["detail"]
    if (not parent.source.matches(output.source) or r["origin_model"] != parent.model_digest
            or r["origin_output"] != parent.coefficient_digest or r["engine"] != thaw_tree(p["engine"])
            or r["operation"] != p["operation"] or r["kind"] != "vector" or r["coefficient_basis"] != "power"
            or r["origin_max_bytes"] > p["origin_max_bytes"]):
        raise ValueError("metric stalk output differs from its complete retained producer")
    declared = _policy(r["declared_limits"], None)
    policy = _policy(r["execution_limits"], None)
    if (declared != thaw_tree(p["execution_limits"]) or any(policy[key] > declared[key] for key in policy)
            or _key(r["declared_max_retained_coordinates"]) != p["max_retained_coordinates"]
            or _key(r["max_retained_coordinates"]) > r["declared_max_retained_coordinates"]):
        raise ValueError("metric stalk selection enlarged its retained admission")
    from rexgraph.cpp_algebraic_metric import validate_algebraic_grade_metric_observation
    maximum = max(output.values.size, sum(len(row) for row in observation["input"]["detail"]["coefficients"]),
                  sum(len(row) for row in detail["coefficients"]))
    validate_algebraic_grade_metric_observation(thaw_tree(observation), execution_limits=policy, max_coordinates=maximum)
    cell = _key(r["cell"])
    if cell >= detail["cell_count"]:
        raise ValueError("metric stalk cell is outside its original result grade")
    begin, end = detail["stalk_offsets"][cell:cell + 2]
    carrier = _carrier(output.source, r["output_carrier"])
    if (carrier != {"grade": detail["grade"], "index": cell, "rank": end - begin,
                   "field_degree": output.source.native.field_degree} or r["source_slice"] != (begin, end)):
        raise ValueError("metric stalk carrier or coordinate slice differs")
    space, grade, shape = _output_declaration(output.source, "vector", carrier)
    if (output.space != space or output.grade != grade or output.axes != (_polynomial_axis(output.source),)
            or output.variance != "coordinate" or r["native_shape"] != shape or output.dependencies
            or tuple(tuple(row) for row in output.values) != tuple(detail["coefficients"][begin:end])
            or output.model_digest != _digest(output.source, r)):
        raise ValueError("metric stalk differs from its exact original result coordinates")


def _restore_metric_stalk(output, snapshot, *, execution_limits=None, max_coordinates=65536,
                          max_retained_coordinates=None, cancellation=None):
    _cancel(cancellation); output.check_state()
    r = output.execution_receipt
    if output.method != _PROFILE:
        raise TypeError("metric replay requires a selected original metric stalk")
    parent = _parent(r)
    _implementation(r["engine"])
    retained = r["max_retained_coordinates"]
    if max_retained_coordinates is not None:
        retained = min(retained, _key(max_retained_coordinates))
    restored = restore_algebraic_grade_metric_observation(snapshot, thaw_tree(parent.execution_receipt["observation"]),
        execution_limits=_policy(r["execution_limits"], execution_limits), max_coordinates=max_coordinates,
        max_retained_coordinates=retained, cancellation=cancellation)
    value = restored[-1].stalk(r["cell"])
    if value.coefficients != tuple(tuple(row) for row in output.values):
        raise ValueError("metric stalk differs from its canonical replayed result")
    _cancel(cancellation)
    return value


@dataclass(frozen=True)
class AlgebraicMetricStalkSelection:
    """Retain actual metric result carriers for repeated original cell selections."""
    output: ModelOutput
    snapshot_owner: object
    values_owner: object
    execution_limits: object
    max_retained_coordinates: int
    max_coordinates: int = 65536
    max_origin_bytes: int | None = None
    _origin_bytes: bytes = field(init=False, repr=False)

    def __post_init__(self):
        from rexgraph_cpp import NativeAlgebraicMetricValues, NativeAlgebraicSnapshot
        observation = _vector_parent(self.output)
        if type(self.snapshot_owner) is not NativeAlgebraicSnapshot or type(self.values_owner) is not NativeAlgebraicMetricValues:
            raise TypeError("metric stalk selection requires owning native result values and snapshot")
        values = self.values_owner
        if not values.snapshot_owner.same_publication(self.snapshot_owner):
            raise ValueError("metric stalk result belongs to another actual publication")
        self.output.source.bind(self.snapshot_owner.archive_owner)
        _implementation(self.output.execution_receipt["engine"])
        policy = _policy(self.output.execution_receipt["execution_limits"], self.execution_limits)
        retained = min(_key(self.max_retained_coordinates), self.output.execution_receipt["max_retained_coordinates"])
        maximum = _key(self.max_coordinates)
        bound = self.output.execution_receipt["origin_max_bytes"]
        if self.max_origin_bytes is not None: bound = min(bound, _key(self.max_origin_bytes))
        from rexgraph.cpp_algebraic_metric import validate_algebraic_grade_metric_observation
        validate_algebraic_grade_metric_observation(observation, execution_limits=policy, max_coordinates=maximum)
        if (values.operation != observation["operation"] or values.field_owner.metric_keys != observation["metric_keys"]
                or values.field_owner.max_retained_coordinates > retained or values.preview(values.domain[0]) != observation["detail"]):
            raise ValueError("metric stalk result differs from its retained metric action or admission")
        actual = algebraic_grade_observation(values.input_owner, availability=observation["availability"], max_coordinates=maximum)
        if not _same_field(observation["input"], actual):
            raise ValueError("metric stalk result input differs from its complete retained lineage")
        origin = pack_value(pack_model(self.output, native=True))
        if len(origin) > bound:
            raise ValueError("metric stalk producer exceeds its retained byte admission")
        object.__setattr__(self, "execution_limits", freeze_tree(policy))
        object.__setattr__(self, "max_retained_coordinates", retained)
        object.__setattr__(self, "max_coordinates", maximum)
        object.__setattr__(self, "max_origin_bytes", bound)
        object.__setattr__(self, "_origin_bytes", origin)

    def native(self, cell, *, cancellation=None):
        """Return an actual vector on the selected original result carrier."""
        _cancel(cancellation); self.output.check_state()
        cell = _key(cell)
        if cell >= self.values_owner.cell_count:
            raise ValueError("metric stalk cell is outside its original result grade")
        value = self.values_owner.stalk(cell)
        _cancel(cancellation)
        return value

    def select(self, cell, *, cancellation=None):
        """Capture exact original stalk coordinates and the complete metric producer."""
        value = self.native(cell, cancellation=cancellation)
        cell = _key(cell)
        reference = self.output.source.bind(self.snapshot_owner.archive_owner)
        carrier = value.preview(0)["source_carrier"]
        space, grade, shape = _output_declaration(reference, "vector", carrier)
        p = self.output.execution_receipt
        receipt = {"profile": _PROFILE, "engine": thaw_tree(p["engine"]), "kind": "vector",
            "output_carrier": carrier, "native_shape": shape, "coefficient_basis": "power",
            "origin_model": self.output.model_digest, "origin_output": self.output.coefficient_digest,
            "origin": self._origin_bytes, "origin_max_bytes": self.max_origin_bytes, "operation": p["operation"],
            "cell": cell, "source_slice": self.values_owner.stalk_offsets[cell:cell + 2],
            "execution_limits": thaw_tree(self.execution_limits), "declared_limits": thaw_tree(p["execution_limits"]),
            "max_retained_coordinates": self.max_retained_coordinates,
            "declared_max_retained_coordinates": p["max_retained_coordinates"]}
        rows = np.array(value.coefficients, dtype=object).reshape(carrier["rank"], reference.native.field_degree)
        output = ModelOutput(rows, space, (_polynomial_axis(reference),), reference, _digest(reference, receipt),
                             "algebraic", _PROFILE, grade=grade, execution_receipt=receipt)
        _cancel(cancellation)
        return output

    def select_many(self, cells, *, max_selections=256, cancellation=None):
        """Select a bounded ordered sequence, including explicit repeated original cells."""
        maximum = _key(max_selections); selected = []
        for cell in cells:
            _cancel(cancellation)
            if len(selected) >= maximum:
                raise ValueError("metric stalk count exceeds its admission")
            cell = _key(cell)
            if cell >= self.values_owner.cell_count:
                raise ValueError("metric stalk cell is outside its original result grade")
            selected.append(cell)
        _cancel(cancellation)
        return tuple(self.select(cell, cancellation=cancellation) for cell in selected)


def prepare_algebraic_metric_stalks(output, snapshot, *, values=None, execution_limits=None,
                                    max_coordinates=65536, max_retained_coordinates=None,
                                    max_origin_bytes=None, cancellation=None):
    """Replay once or verify supplied owning metric results before original stalk selection."""
    _cancel(cancellation)
    observation = _vector_parent(output)
    _implementation(output.execution_receipt["engine"])
    policy = _policy(output.execution_receipt["execution_limits"], execution_limits)
    retained = output.execution_receipt["max_retained_coordinates"]
    if max_retained_coordinates is not None: retained = min(retained, _key(max_retained_coordinates))
    if values is None:
        values = restore_algebraic_grade_metric_observation(snapshot, observation, execution_limits=policy,
            max_coordinates=max_coordinates, max_retained_coordinates=retained, cancellation=cancellation)[-1]
    result = AlgebraicMetricStalkSelection(output, snapshot, values, policy, retained, max_coordinates, max_origin_bytes)
    _cancel(cancellation)
    return result
