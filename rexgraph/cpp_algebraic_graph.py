"""Explicit exact model observations across named native graph comparisons."""
from collections.abc import Mapping
from hashlib import sha256

import numpy as np

from rexgraph.cpp_algebraic import (
    _binding,
    _carrier,
    _key,
    _output_declaration,
    _output_kind,
    _polynomial_axis,
    _portable_limits,
    _space,
)
from rexgraph.model_state import ModelOutput, thaw_tree
from rexgraph.source_binding import CppAlgebraicSourceReference, _sha, source_from_record
from rexgraph.value_codec import pack_value

__all__ = ["cpp_algebraic_compare"]
_profile = "algebraic-graph-comparison-v1"
_operations = {"scalar": {"apply"}, "carrier": {"apply"}, "measurement": {"apply", "quadrance"}}


def _operation_digest(receipt, target):
    return sha256(pack_value({"profile": _profile, "source": thaw_tree(receipt["source"]),
                             "target": target.as_record(), "kind": receipt["comparison_kind"],
                             "key": receipt["comparison_key"], "operation": receipt["operation"],
                             "origin_model": receipt["origin_model"]})).hexdigest()


def validate_comparison_output(output):
    """Check the retained graph, exact coordinate axes and explicit comparison receipt."""
    r = output.execution_receipt
    names = {"profile", "engine", "graph_digest", "comparison_kind", "comparison_key", "operation",
             "source", "source_carrier", "kind", "output_carrier", "native_shape", "origin_model",
             "origin_output", "input_digest", "coefficient_basis", "limits", "comparison"}
    if not isinstance(r, Mapping) or set(r) != names or r["profile"] != _profile:
        raise ValueError("algebraic comparison output requires its complete execution receipt")
    target = output.source
    source = source_from_record(r["source"], allow_cpp=True)
    if (not isinstance(target, CppAlgebraicSourceReference) or not isinstance(source, CppAlgebraicSourceReference)
            or target.native.graph_digest is None or source.native.graph_digest != target.native.graph_digest
            or r["graph_digest"] != target.native.graph_digest):
        raise ValueError("algebraic comparison sources require the same declared graph")
    kind = r["comparison_kind"]
    if kind not in _operations or r["operation"] not in _operations[kind]:
        raise ValueError("unsupported algebraic model comparison operation")
    key = _key(r["comparison_key"])
    if key == 0:
        raise ValueError("algebraic comparison key must be nonzero")
    for name in ("origin_model", "origin_output", "input_digest"):
        _sha(r[name])
    _portable_limits(r["limits"], graph=True)
    if r["coefficient_basis"] != "power" or output.axes != (_polynomial_axis(target),):
        raise ValueError("algebraic comparison output coefficient basis differs")
    rank, grade, shape = _output_kind(target, r["kind"], r["output_carrier"])
    if rank > r["limits"]["archive.modules.max_rank"] or target.native.field_degree > r["limits"]["archive.field.max_degree"]:
        raise ValueError("algebraic comparison output exceeds its declared dimensions")
    if (output.space != _space(target, r["output_carrier"]) or output.grade != grade
            or tuple(_key(v) for v in r["native_shape"]) != shape or output.variance != "coordinate"):
        raise ValueError("algebraic comparison output carrier differs")
    if kind == "scalar":
        if r["source_carrier"] is not None or r["kind"] != "scalar":
            raise ValueError("scalar comparison requires scalar inputs and outputs")
    else:
        carrier = _carrier(source, r["source_carrier"])
        if carrier["rank"] > r["limits"]["archive.modules.max_rank"]:
            raise ValueError("algebraic comparison input exceeds its declared rank")
        if r["kind"] != ("scalar" if r["operation"] == "quadrance" else "vector"):
            raise ValueError("algebraic comparison output kind differs from its operation")
    comparison = r["comparison"]
    comparison_names = {"kind", "key", "graph_digest", "source_field", "target_field", "source_bundle", "target_bundle",
                        "source_domain", "target_domain", "scalar_key", "carrier_key", "source_archive", "target_archive",
                        "source_record", "target_record", "accession_key", "reconstruction_key", "source_carrier",
                        "target_carrier", "coefficient_count", "coefficients", "coefficients_truncated",
                        "action_shape", "action", "action_truncated"}
    if (not isinstance(comparison, Mapping) or set(comparison) != comparison_names or comparison.get("kind") != kind
            or comparison.get("key") != key or comparison.get("graph_digest") != r["graph_digest"]):
        raise ValueError("algebraic comparison name differs from its graph receipt")
    for name in ("key", "source_field", "target_field", "source_bundle", "target_bundle", "source_domain",
                 "target_domain", "scalar_key", "carrier_key", "source_archive", "target_archive", "source_record",
                 "target_record", "accession_key", "reconstruction_key", "coefficient_count"):
        _key(comparison[name])
    action_shape = tuple(_key(v) for v in comparison["action_shape"])
    if len(action_shape) != 2 or comparison["coefficients"] or comparison["action"]:
        raise ValueError("algebraic model comparison retains its declared zero length preview")
    if kind == "scalar":
        if (action_shape != (target.native.field_degree, source.native.field_degree)
                or comparison["source_carrier"] is not None or comparison["target_carrier"] is not None):
            raise ValueError("algebraic scalar comparison dimensions differ")
    else:
        source_carrier = _carrier(source, comparison["source_carrier"])
        target_carrier = _carrier(target, comparison["target_carrier"])
        if (dict(r["source_carrier"]) != source_carrier or action_shape != (
                target_carrier["rank"] * target.native.field_degree, source_carrier["rank"] * source.native.field_degree)
                or r["kind"] == "vector" and dict(r["output_carrier"]) != target_carrier):
            raise ValueError("algebraic carrier comparison dimensions differ")
    if output.model_digest != _operation_digest(r, target):
        raise ValueError("algebraic comparison operation identity differs")


def cpp_algebraic_compare(output, source_archive, target_archive, *, kind, key,
                          operation="apply", carrier_record=None, carrier_side="source"):
    """Rebind a retained scalar or vector and apply one named graph comparison."""
    binding = _binding()
    if not isinstance(output, ModelOutput) or output.arithmetic != "algebraic":
        raise TypeError("algebraic model comparison requires an exact ModelOutput")
    output.check_state()
    if type(source_archive) is not binding.NativeAlgebraicArchive or type(target_archive) is not binding.NativeAlgebraicArchive:
        raise TypeError("model comparisons require owning algebraic archive views")
    graph = source_archive.graph_owner
    if graph is None or target_archive.graph_owner is not graph:
        raise ValueError("model comparison requires both archives on one actual graph owner")
    ref = output.source.bind(source_archive)
    if kind not in _operations or operation not in _operations[kind]:
        raise ValueError("unsupported algebraic model comparison operation")
    if dict(output.execution_receipt["engine"]) != binding.build_info():
        raise ValueError("model comparison requires the retained native semantic implementation")
    if output.execution_receipt["kind"] == "scalar":
        if kind != "scalar" or carrier_record is not None or carrier_side != "source":
            raise ValueError("scalar model comparison has no module carrier record")
        value = source_archive.scalar(output.values[0])
        carrier = None
    else:
        if kind == "scalar" or output.execution_receipt["kind"] != "vector" or carrier_record is None:
            raise ValueError("vector comparison requires an explicit source carrier record")
        value = source_archive.vector(carrier_record, output.values, carrier_side=carrier_side)
        carrier = value.preview(0)["source_carrier"]
        if output.space != _space(ref, carrier) or output.axes != (_polynomial_axis(ref),):
            raise ValueError("comparison input differs from its declared source carrier")
    result = graph.compare(kind, key, value, target_archive=target_archive.graph_key, operation=operation)
    target = CppAlgebraicSourceReference(target_archive)
    info = result.preview(0)
    space, grade, shape = _output_declaration(target, info["kind"], info["source_carrier"])
    rows = np.array(result.coefficients, dtype=object).reshape((len(space.keys), target.native.field_degree))
    receipt = {"profile": _profile, "engine": binding.build_info(), "graph_digest": target.native.graph_digest,
               "comparison_kind": kind, "comparison_key": _key(key), "operation": operation,
               "source": ref.as_record(), "source_carrier": carrier, "kind": info["kind"],
               "output_carrier": info["source_carrier"], "native_shape": shape,
               "origin_model": output.model_digest, "origin_output": output.coefficient_digest,
               "input_digest": sha256(pack_value({"source": ref.as_record(), "coefficients": value.coefficients})).hexdigest(),
               "coefficient_basis": "power", "limits": graph.limits, "comparison": graph.comparison(kind, key, 0)}
    return ModelOutput(rows, space, (_polynomial_axis(target),), target, _operation_digest(receipt, target),
                       "algebraic", _profile, grade=grade, execution_receipt=receipt)
