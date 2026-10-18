"""Explicit original stalk selection from complete whole grade model outputs."""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from rexgraph.cpp_algebraic import _carrier, _key, _output_declaration, _polynomial_axis
from rexgraph.cpp_algebraic_grade import (
    _matches_publication,
    algebraic_grade_observation,
    restore_algebraic_grade_observation,
    validate_algebraic_grade_observation,
)
from rexgraph.cpp_algebraic_grade_model import _PROFILE, _cancel, _engine
from rexgraph.cpp_algebraic_query import (
    _check_publication,
    _check_retained_snapshot,
    _closed,
    _limits,
    algebraic_snapshot_record,
    read_algebraic_snapshot_record,
)
from rexgraph.model_state import ModelOutput, freeze_tree, thaw_tree
from rexgraph.source_binding import CppAlgebraicSourceReference, _sha
from rexgraph.value_codec import pack_value

__all__ = ["AlgebraicGradeStalkSelection", "prepare_algebraic_grade_stalks", "cpp_algebraic_grade_stalk",
           "algebraic_grade_stalk_record", "read_algebraic_grade_stalk_output", "read_algebraic_grade_stalk_record"]
_STALK = "algebraic-grade-stalk-v1"


def _digest(source, receipt):
    return sha256(pack_value({"profile": _STALK, "source": source.as_record(), "cell": receipt["cell"],
                             "origin_model": receipt["origin_model"], "origin_output": receipt["origin_output"]})).hexdigest()


def _policy(saved, requested):
    limits = _limits(saved, complete=True)
    if requested is not None:
        reader = _limits(requested)
        for key in requested:
            limits[key] = min(limits[key], reader[key])
    return limits


def _implementation(receipt):
    from rexgraph_cpp import build_info
    _engine(receipt)
    current = build_info()
    if any(receipt[key] != current[key] for key in ("engine_version", "engine_digest", "scalar_policy")):
        raise ValueError("stalk selection requires its retained canonical engine implementation")


def validate_stalk_output(output):
    """Check a selected original carrier and its complete grade observation without native imports."""
    r = thaw_tree(output.execution_receipt)
    _closed(r, ("profile", "engine", "kind", "output_carrier", "native_shape", "coefficient_basis",
               "origin_model", "origin_output", "observation", "cell", "source_slice", "execution_limits", "declared_limits"))
    if output.method != _STALK or output.arithmetic != "algebraic" or r["profile"] != _STALK:
        raise ValueError("unsupported original stalk output profile")
    if not isinstance(output.source, CppAlgebraicSourceReference):
        raise TypeError("stalk selection requires its retained algebraic archive source")
    _engine(r["engine"])
    _sha(r["origin_model"]); _sha(r["origin_output"])
    observation = r["observation"]
    limits, declared = _limits(r["execution_limits"], complete=True), _limits(r["declared_limits"], complete=True)
    if any(limits[key] > declared[key] for key in limits):
        raise ValueError("stalk selection enlarged its retained execution admission")
    validate_algebraic_grade_observation(observation, execution_limits=limits,
                                         max_coordinates=sum(len(row) for row in observation["detail"]["coefficients"]))
    if observation["operation"] != "field" or _check_publication(observation["publication"]).native != output.source.native:
        raise ValueError("stalk selection requires its complete original grade and source publication")
    if r["engine"] != observation["engine"] or r["kind"] != "vector" or r["coefficient_basis"] != "power":
        raise ValueError("stalk selection changed its engine, value kind or polynomial basis")
    detail = observation["detail"]
    cell = _key(r["cell"])
    if cell >= detail["cell_count"]:
        raise ValueError("stalk selection cell is outside its original grade")
    begin, end = detail["stalk_offsets"][cell:cell + 2]
    carrier = _carrier(output.source, r["output_carrier"])
    if (carrier != {"grade": detail["grade"], "index": cell, "rank": end - begin,
                    "field_degree": output.source.native.field_degree}
            or r["source_slice"] != (begin, end)):
        raise ValueError("stalk selection differs from its original carrier and coordinate slice")
    space, grade, shape = _output_declaration(output.source, "vector", carrier)
    if (output.space != space or output.grade != grade or r["native_shape"] != shape
            or output.axes != (_polynomial_axis(output.source),) or output.variance != "coordinate"
            or tuple(tuple(row) for row in output.values) != detail["coefficients"][begin:end]):
        raise ValueError("stalk selection differs from its retained exact polynomial coordinates")
    if output.model_digest != _digest(output.source, r):
        raise ValueError("stalk selection operation identity differs")


def _same_field(saved, actual):
    if not _matches_publication(saved["publication"], actual["publication"]):
        return False
    if len(saved["lineage"]) != len(actual["lineage"]):
        return False
    for old, new in zip(saved["lineage"], actual["lineage"], strict=True):
        if {key: v for key, v in old.items() if key != "execution_limits"} != {key: v for key, v in new.items() if key != "execution_limits"}:
            return False
    return ({key: v for key, v in saved["detail"].items() if key != "execution_limits"}
            == {key: v for key, v in actual["detail"].items() if key != "execution_limits"}
            and saved["availability"] == actual["availability"] and saved["sum"] == actual["sum"])


@dataclass(frozen=True)
class AlgebraicGradeStalkSelection:
    """Retain one checked original grade owner for repeated explicit cell selections."""
    output: ModelOutput
    snapshot_owner: object
    tensor_owner: object
    execution_limits: object
    max_coordinates: int = 65536

    def __post_init__(self):
        from rexgraph_cpp import NativeAlgebraicGradeTensor, NativeAlgebraicSnapshot
        if not isinstance(self.output, ModelOutput) or self.output.method != _PROFILE:
            raise TypeError("stalk selection requires a complete whole-grade model output")
        self.output.check_state()
        if type(self.snapshot_owner) is not NativeAlgebraicSnapshot or type(self.tensor_owner) is not NativeAlgebraicGradeTensor:
            raise TypeError("stalk selection requires owning native snapshot and grade tensor pins")
        if not self.tensor_owner.matches_source(self.snapshot_owner):
            raise ValueError("stalk selection tensor belongs to another actual publication")
        self.output.source.bind(self.snapshot_owner.archive_owner)
        _implementation(self.output.execution_receipt["engine"])
        maximum = _key(self.max_coordinates)
        policy = _policy(self.output.execution_receipt["execution_limits"], self.execution_limits)
        saved = thaw_tree(self.output.execution_receipt["observation"])
        validate_algebraic_grade_observation(saved, execution_limits=policy, max_coordinates=maximum)
        actual = algebraic_grade_observation(self.tensor_owner, availability=saved["availability"], max_coordinates=maximum)
        validate_algebraic_grade_observation(actual, execution_limits=policy, max_coordinates=maximum)
        if not _same_field(saved, actual):
            raise ValueError("stalk selection owner differs from its complete retained grade lineage")
        object.__setattr__(self, "execution_limits", freeze_tree(policy))
        object.__setattr__(self, "max_coordinates", maximum)

    def native(self, cell, *, cancellation=None):
        """Return the actual original stalk vector without replaying or flattening the grade again."""
        _cancel(cancellation)
        self.output.check_state()
        cell = _key(cell)
        if cell >= self.tensor_owner.cell_count:
            raise ValueError("stalk selection cell is outside its original grade")
        value = self.tensor_owner.stalk(cell)
        _cancel(cancellation)
        return value

    def select(self, cell, *, cancellation=None):
        """Capture one exact vector output with its original grade and full boundary lineage."""
        value = self.native(cell, cancellation=cancellation)
        cell = _key(cell)
        reference = self.output.source.bind(self.snapshot_owner.archive_owner)
        carrier = value.preview(0)["source_carrier"]
        space, grade, shape = _output_declaration(reference, "vector", carrier)
        offsets = self.tensor_owner.stalk_offsets
        receipt = {"profile": _STALK, "engine": thaw_tree(self.output.execution_receipt["engine"]), "kind": "vector",
                   "output_carrier": carrier, "native_shape": shape, "coefficient_basis": "power",
                   "origin_model": self.output.model_digest, "origin_output": self.output.coefficient_digest,
                   "observation": thaw_tree(self.output.execution_receipt["observation"]), "cell": cell,
                   "source_slice": offsets[cell:cell + 2], "execution_limits": thaw_tree(self.execution_limits),
                   "declared_limits": thaw_tree(self.output.execution_receipt["execution_limits"])}
        rows = np.array(value.coefficients, dtype=object).reshape((carrier["rank"], reference.native.field_degree))
        result = ModelOutput(rows, space, (_polynomial_axis(reference),), reference, _digest(reference, receipt),
                             "algebraic", _STALK, grade=grade, execution_receipt=receipt)
        _cancel(cancellation)
        return result

    def select_many(self, cells, *, max_selections=256, cancellation=None):
        """Select a bounded ordered sequence, including explicit repeated original cells."""
        maximum = _key(max_selections)
        selected = []
        for cell in cells:
            _cancel(cancellation)
            if len(selected) >= maximum:
                raise ValueError("stalk selection count exceeds its admission")
            cell = _key(cell)
            if cell >= self.tensor_owner.cell_count:
                raise ValueError("stalk selection cell is outside its original grade")
            selected.append(cell)
        _cancel(cancellation)
        return tuple(self.select(cell, cancellation=cancellation) for cell in selected)


def prepare_algebraic_grade_stalks(output, snapshot, *, tensor=None, execution_limits=None,
                                  max_coordinates=65536, cancellation=None, metric_values=None,
                                  max_retained_coordinates=None, max_origin_bytes=None):
    """Replay once or check a supplied owning grade field before repeated stalk selection."""
    if isinstance(output, ModelOutput) and output.method == "algebraic-grade-metric-action-v1":
        if tensor is not None:
            raise TypeError("metric outputs require native metric values on their original result carriers")
        from rexgraph.cpp_algebraic_metric_stalk import prepare_algebraic_metric_stalks
        return prepare_algebraic_metric_stalks(output, snapshot, values=metric_values, execution_limits=execution_limits,
            max_coordinates=max_coordinates, max_retained_coordinates=max_retained_coordinates,
            max_origin_bytes=max_origin_bytes, cancellation=cancellation)
    if metric_values is not None or max_retained_coordinates is not None or max_origin_bytes is not None:
        raise TypeError("metric stalk admission requires a complete metric model output")
    if not isinstance(output, ModelOutput) or output.method != _PROFILE:
        raise TypeError("stalk selection requires a complete whole-grade model output")
    _cancel(cancellation)
    output.check_state()
    _implementation(output.execution_receipt["engine"])
    policy = _policy(output.execution_receipt["execution_limits"], execution_limits)
    if tensor is None:
        tensor = restore_algebraic_grade_observation(snapshot, thaw_tree(output.execution_receipt["observation"]),
            execution_limits=policy, max_coordinates=max_coordinates, cancellation=cancellation)
    result = AlgebraicGradeStalkSelection(output, snapshot, tensor, policy, max_coordinates)
    _cancel(cancellation)
    return result


def cpp_algebraic_grade_stalk(output, snapshot, *, cell, execution_limits=None, max_coordinates=65536,
                             cancellation=None, max_retained_coordinates=None, max_origin_bytes=None):
    """Select one original stalk as an exact carrier level model output."""
    return prepare_algebraic_grade_stalks(output, snapshot, execution_limits=execution_limits,
        max_coordinates=max_coordinates, max_retained_coordinates=max_retained_coordinates,
        max_origin_bytes=max_origin_bytes, cancellation=cancellation).select(cell, cancellation=cancellation)


def algebraic_grade_stalk_record(output, snapshot):
    """Retain a selected vector beside its complete source publication and prefix bytes."""
    if not isinstance(output, ModelOutput) or output.method not in {_STALK, "algebraic-grade-metric-stalk-v1"}:
        raise TypeError("stalk record requires an original stalk model output")
    output.check_state()
    if output.method == "algebraic-grade-metric-stalk-v1":
        from rexgraph.cpp_algebraic_metric_stalk import _parent, _restore_metric_stalk
        parent = _parent(output.execution_receipt)
        maximum = max(parent.values.size, sum(len(row) for row in parent.execution_receipt["observation"]["input"]["detail"]["coefficients"]))
        _restore_metric_stalk(output, snapshot, max_coordinates=maximum)
        record = algebraic_snapshot_record(snapshot)
        record.attach_metadata(1, 0, "algebraic_grade_stalk", output)
        return record
    tensor = restore_algebraic_grade_observation(snapshot, thaw_tree(output.execution_receipt["observation"]),
        execution_limits=output.execution_receipt["execution_limits"],
        max_coordinates=sum(len(row) for row in output.execution_receipt["observation"]["detail"]["coefficients"]))
    if tensor.stalk(output.execution_receipt["cell"]).coefficients != tuple(tuple(row) for row in output.values):
        raise ValueError("selected stalk differs from its actual replayed source")
    record = algebraic_snapshot_record(snapshot)
    record.attach_metadata(1, 0, "algebraic_grade_stalk", output)
    return record


def read_algebraic_grade_stalk_output(record, index=0, *, max_archive_bytes=256 << 20, max_history_bytes=256 << 20):
    """Check a portable selected vector and every retained source byte without native imports."""
    index = _key(index)
    output = record.get_metadata(1, index, "algebraic_grade_stalk")
    if not isinstance(output, ModelOutput) or output.method not in {_STALK, "algebraic-grade-metric-stalk-v1"}:
        raise TypeError("selected record has no original stalk model output")
    output.check_state()
    declared = _check_retained_snapshot(record, index, max_archive_bytes, max_history_bytes)
    if output.method == "algebraic-grade-metric-stalk-v1":
        from rexgraph.cpp_algebraic_metric_stalk import _parent
        observation = _parent(output.execution_receipt).execution_receipt["observation"]
    else:
        observation = output.execution_receipt["observation"]
    if not _matches_publication(observation["publication"], declared["publication"]):
        raise ValueError("selected stalk differs from its retained source publication")
    return output


def read_algebraic_grade_stalk_record(record, index=0, *, limits=None, execution_limits=None,
                                     max_coordinates=65536, history_max_bytes=256 << 20, cancellation=None,
                                     max_retained_coordinates=None):
    """Replay a retained grade and return its selected vector with the actual restored snapshot."""
    _cancel(cancellation)
    output = read_algebraic_grade_stalk_output(record, index, max_history_bytes=history_max_bytes,
        max_archive_bytes=(limits or {}).get("max_file_bytes", 256 << 20))
    _implementation(output.execution_receipt["engine"])
    snapshot = read_algebraic_snapshot_record(record, index, limits=limits, history_max_bytes=history_max_bytes)
    if output.method == "algebraic-grade-metric-stalk-v1":
        from rexgraph.cpp_algebraic_metric_stalk import _restore_metric_stalk
        value = _restore_metric_stalk(output, snapshot, execution_limits=execution_limits,
            max_coordinates=max_coordinates, max_retained_coordinates=max_retained_coordinates, cancellation=cancellation)
        return output, snapshot, value
    if max_retained_coordinates is not None:
        raise TypeError("metric stalk admission requires a selected metric model output")
    tensor = restore_algebraic_grade_observation(snapshot, thaw_tree(output.execution_receipt["observation"]),
        execution_limits=_policy(output.execution_receipt["execution_limits"], execution_limits),
        max_coordinates=max_coordinates, cancellation=cancellation)
    value = tensor.stalk(output.execution_receipt["cell"])
    if value.coefficients != tuple(tuple(row) for row in output.values):
        raise ValueError("selected stalk differs from its actual replayed source")
    _cancel(cancellation)
    return output, snapshot, value
