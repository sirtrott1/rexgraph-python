"""Durable local model declarations and retained output coordinates."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral
from types import MappingProxyType

import numpy as np

from rexgraph.exact_value import validate_exact_array
from rexgraph.model_integrity import seal_model, check_model, _native_freeze
from rexgraph.tensor_field import FieldSource, TensorField
from rexgraph.type_accession import CoordinateSpace

__all__ = ["ModelInput", "model_input", "ModelState", "ModelOutput", "ModelBatch", "ModelTimeline", "freeze_tree", "thaw_tree"]
_metadata_scalars = frozenset({type(None), bool, int, str, float})
_metadata_cache_limit = 1024


def _primitive_map_key(value):
    if len(value) > 8:
        return None
    result = []
    for key, item in value.items():
        if type(key) not in {str, int} or type(item) not in _metadata_scalars:
            return None
        result.append((type(key), key, type(item), item.hex() if type(item) is float else item))
    return tuple(result)


def freeze_tree(value):
    """Copy supported data and share repeated immutable scalar declarations."""
    if _native_freeze is not None:
        return _native_freeze(value, _metadata_cache_limit)
    return _freeze_tree(value, {})


def _freeze_tree(value, declarations):
    if value is None or isinstance(value, (str, bytes, bool, Fraction)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("model metadata must be finite")
        return value
    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True, order="C")
        if result.dtype.hasobject:
            validate_exact_array(result, context="model exact object tensors")
        elif result.dtype.kind not in "biuf" or not result.dtype.isnative or result.dtype.itemsize not in {1, 2, 4, 8}:
            raise TypeError("unsupported model tensor dtype")
        elif result.dtype.kind == "f" and not np.isfinite(result).all():
            raise ValueError("model tensor must be finite")
        result.flags.writeable = False
        return result
    if isinstance(value, ModelInput):
        value.check_state()
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(k, (str, int)) or isinstance(k, bool) for k in value):
            raise TypeError("model mapping keys must be strings or integers")
        result = {k: _freeze_tree(v, declarations) for k, v in value.items()}
        key = _primitive_map_key(result)
        if key is not None:
            known = declarations.get(key)
            if known is not None:
                return known
        frozen = MappingProxyType(result)
        if key is not None and len(declarations) < _metadata_cache_limit:
            declarations[key] = frozen
        return frozen
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_tree(v, declarations) for v in value)
    raise TypeError(f"unsupported model value {type(value).__name__}")


def thaw_tree(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {k: thaw_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(thaw_tree(v) for v in value)
    return value


def _source(ref):
    if not isinstance(ref, FieldSource):
        raise TypeError("model requires an explicit native source reference")
    ref.check()
    return ref.detached()


def _sha(value, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("expected a SHA256 identity")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("invalid SHA256 identity") from exc


def _digest(value):
    from rexgraph.model_codec import model_payload
    from rexgraph.state import encode_tensors, state_digest
    tensors = model_payload(value)
    # The JSON object prefix decreases, so integer structural codecs do not apply.
    spec = tensors.pop("spec")
    codecs = encode_tensors(tensors)
    tensors["spec"] = spec
    tensors["codec"] = np.frombuffer(json.dumps(codecs, sort_keys=True).encode(), dtype=np.uint8)
    return state_digest(tensors)


@dataclass(frozen=True, eq=False)
class ModelState:
    """Source bound parameters, observations and optional continuation state."""
    adapter: str
    adapter_version: str
    source: FieldSource
    space: CoordinateSpace
    output_axes: tuple[CoordinateSpace, ...]
    configuration: object
    payload: object
    arithmetic: str
    step: int = 0
    parent: str | None = None
    dependencies: tuple[FieldSource, ...] = ()
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.adapter, str) or not self.adapter or not isinstance(self.adapter_version, str) or not self.adapter_version:
            raise ValueError("model adapter and version must be declared")
        if not isinstance(self.space, CoordinateSpace):
            raise TypeError("model requires named primary coordinates")
        axes = tuple(self.output_axes)
        if any(not isinstance(v, CoordinateSpace) for v in axes):
            raise TypeError("model outputs require named axes")
        if len({v.name for v in (self.space, *axes)}) != len(axes) + 1:
            raise ValueError("model axis names must be distinct")
        if self.arithmetic not in {"rational", "algebraic", "approximate"}:
            raise ValueError("model arithmetic must be rational, algebraic or approximate")
        if isinstance(self.step, bool) or not isinstance(self.step, Integral) or self.step < 0:
            raise ValueError("model step must be a nonnegative integer")
        _sha(self.parent, optional=True)
        object.__setattr__(self, "source", _source(self.source))
        object.__setattr__(self, "dependencies", tuple(_source(v) for v in self.dependencies))
        object.__setattr__(self, "configuration", freeze_tree(self.configuration))
        object.__setattr__(self, "payload", freeze_tree(self.payload))
        if not isinstance(self.configuration, Mapping) or not isinstance(self.payload, Mapping):
            raise TypeError("model configuration and payload require mappings")
        object.__setattr__(self, "output_axes", axes)
        object.__setattr__(self, "step", int(self.step))
        if self.arithmetic == "algebraic":
            from rexgraph.cpp_algebraic import validate_algebraic_state
            validate_algebraic_state(self)
        if self.adapter == "cpp_perceptron" and any(
                isinstance(entry, Mapping) and "coefficient_features" in entry
                for entry in self.payload.get("history", ())):
            from rexgraph.cpp_coefficient_features import validate_coefficient_feature_history
            validate_coefficient_feature_history(self)
        if self.adapter == "cpp_binary_gate":
            from rexgraph.cpp_binary_gate import validate_binary_gate_state
            validate_binary_gate_state(self)
        if self.adapter == "cpp_correspondence":
            from rexgraph.cpp_correspondence import validate_correspondence_state
            validate_correspondence_state(self)
        if self.adapter == "cpp_context_catalog":
            from rexgraph.cpp_context_catalog import validate_context_catalog
            validate_context_catalog(self)
        if self.adapter == "cpp_relation_response":
            from rexgraph.cpp_relation_response import validate_relation_response_state
            validate_relation_response_state(self)
        if self.adapter == "cpp_channels":
            from rexgraph.cpp_channels import validate_channel_state
            validate_channel_state(self)
        if self.adapter == "cpp_relational_sections":
            from rexgraph.cpp_relational_sections import validate_relational_sections_state
            validate_relational_sections_state(self)
        if self.adapter == "cpp_stalk_fields":
            from rexgraph.cpp_stalk_fields import validate_stalk_fields_state
            validate_stalk_fields_state(self)
        if self.adapter == "cpp_stalk_boundary":
            from rexgraph.cpp_stalk_boundary import validate_stalk_boundary_state
            validate_stalk_boundary_state(self)
        if self.adapter == "cpp_graded_stalk_transport":
            from rexgraph.cpp_graded_stalk_transport import validate_graded_stalk_transport_state
            validate_graded_stalk_transport_state(self)
        if self.adapter == "cpp_stalk_transport":
            from rexgraph.cpp_stalk_transport import validate_stalk_transport_state
            validate_stalk_transport_state(self)
        if self.adapter == "cpp_stalk_pattern_bank":
            from rexgraph.cpp_stalk_pattern_model import validate_stalk_pattern_bank_state
            validate_stalk_pattern_bank_state(self)
        if self.adapter == "cpp_relational_known_product":
            from rexgraph.cpp_relational_known_product import validate_known_product_state
            validate_known_product_state(self)
        if self.adapter == "cpp_relational_field_program":
            from rexgraph.cpp_relational_field_program import validate_field_program_state
            validate_field_program_state(self)
        if self.adapter == "cpp_conditioned_occurrence_program":
            from rexgraph.cpp_conditioned_occurrence_program import validate_conditioned_occurrence_state
            validate_conditioned_occurrence_state(self)
        if self.adapter == "cpp_conditioned_field_family":
            from rexgraph.cpp_conditioned_field_family import validate_conditioned_family_state
            validate_conditioned_family_state(self)
        if self.adapter == "cpp_conditioned_field_program":
            from rexgraph.cpp_conditioned_field_program import validate_conditioned_field_state
            validate_conditioned_field_state(self)
        if self.adapter == "cpp_relational_literal_finite_field":
            from rexgraph.cpp_relational_literal_finite_field import validate_literal_finite_state
            validate_literal_finite_state(self)
        if self.adapter == "cpp_relational_finite_field":
            from rexgraph.cpp_relational_finite_field import validate_finite_field_state
            validate_finite_field_state(self)
        if self.adapter == "cpp_relational_finite_operator":
            from rexgraph.cpp_relational_finite_operator import validate_finite_operator_state
            validate_finite_operator_state(self)
        if self.adapter == "cpp_stalk_objective":
            from rexgraph.cpp_stalk_objective import validate_stalk_objective_state
            validate_stalk_objective_state(self)
        if self.adapter == "cpp_stalk_sections":
            from rexgraph.cpp_stalk_sections import validate_stalk_sections_state
            validate_stalk_sections_state(self)
        if self.adapter == "cpp_local_sections":
            from rexgraph.cpp_local_sections import validate_local_sections_state
            validate_local_sections_state(self)
        seal_model(self, _digest, "model state changed after construction")

    @property
    def coefficient_digest(self):
        self.check_state()
        return self._digest

    def check_state(self):
        check_model(self, _digest, "model state changed after construction")

    def bind(self, source):
        self.check_state()
        if self.adapter in {"cpp_algebraic_grade", "cpp_algebraic_grade_metric"}:
            from rexgraph.cpp_algebraic_grade_model import bind_grade_model_source
            return bind_grade_model_source(self, source)
        return self.source.bind(source)

    def child(self, *, payload, configuration=None, source=None, space=None, step=None, dependencies=None):
        return ModelState(self.adapter, self.adapter_version, self.source if source is None else source,
            self.space if space is None else space, self.output_axes,
            self.configuration if configuration is None else configuration, payload, self.arithmetic,
            self.step if step is None else step, self.coefficient_digest,
            self.dependencies if dependencies is None else dependencies)


@dataclass(frozen=True, eq=False)
class ModelOutput:
    """Numerical or exact values with their model and source identities."""
    values: object
    space: CoordinateSpace
    axes: tuple[CoordinateSpace, ...]
    source: FieldSource
    model_digest: str
    arithmetic: str
    method: str
    dependencies: tuple[FieldSource, ...] = ()
    grade: int | None = None
    variance: str = "coordinate"
    execution_receipt: object = None
    field_archive: object = None
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        _sha(self.model_digest)
        if self.grade is not None and (isinstance(self.grade, bool) or not isinstance(self.grade, Integral) or self.grade < 0):
            raise ValueError("output grade must be a nonnegative integer")
        if self.variance not in {"coordinate", "chain", "cochain"}:
            raise ValueError("invalid output variance")
        if not isinstance(self.space, CoordinateSpace) or any(not isinstance(a, CoordinateSpace) for a in self.axes):
            raise TypeError("model output requires named coordinates")
        axes = tuple(self.axes)
        if len({a.name for a in (self.space, *axes)}) != len(axes) + 1:
            raise ValueError("output axis names must be distinct")
        if self.arithmetic not in {"rational", "algebraic", "approximate"}:
            raise ValueError("unknown output arithmetic")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("model output requires its executed method")
        packed_integer = (self.arithmetic == "rational" and self.method == "checked-int64-lattice-batch-v1"
                          and isinstance(self.execution_receipt, Mapping)
                          and self.execution_receipt.get("scalar_policy") == "checked-int64")
        packed_boolean = (self.arithmetic == "rational" and self.method == "packed-boolean-gate-v1"
                          and isinstance(self.execution_receipt, Mapping)
                          and self.execution_receipt.get("scalar_policy") == "packed-boolean")
        values = np.asarray(self.values, dtype=object if self.arithmetic in {"rational", "algebraic"} and not (packed_integer or packed_boolean) else None)
        if values.shape != (len(self.space.keys), *(len(a.keys) for a in axes)):
            raise ValueError("output values differ from declared axes")
        if packed_boolean:
            if values.dtype != np.dtype("uint8") or np.any(values > 1):
                raise TypeError("packed Boolean output requires unsigned byte coefficients zero or one")
        elif packed_integer:
            if values.dtype != np.dtype("int64"):
                raise TypeError("packed integer output requires native signed 64 bit coefficients")
        elif self.arithmetic in {"rational", "algebraic"}:
            from rexgraph.graded_metric import _fraction
            values = np.array([_fraction(v) for v in values.flat], dtype=object).reshape(values.shape)
        elif values.dtype.kind != "f":
            raise TypeError("numerical model output requires floating coefficients")
        object.__setattr__(self, "values", freeze_tree(values))
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "source", _source(self.source))
        object.__setattr__(self, "dependencies", tuple(_source(v) for v in self.dependencies))
        if self.field_archive is not None:
            from rexgraph.field_archive import FIELD_ARCHIVE_METHODS, freeze_field_archive
            from rexgraph.source_binding import CppSourceReference
            archive = freeze_field_archive(self.field_archive)
            if (self.arithmetic != "rational" or self.method != FIELD_ARCHIVE_METHODS.get(archive["schema"])
                    or not isinstance(self.source, CppSourceReference)):
                raise ValueError("native field archives require the declared rational program output and source")
            object.__setattr__(self, "field_archive", archive)
        elif self.method in {"exact-available-relational-known-product-family-v1", "exact-relational-field-program-v1", "exact-relational-accession-field-program-v1",
                             "exact-relational-affine-field-program-v1", "exact-relational-affine-evaluation-v1",
                             "exact-relational-comparison-field-program-v1", "exact-available-relational-field-program-v1",
                             "exact-available-relational-local-solve-v1", "exact-available-relational-finite-field-family-v1",
                             "exact-available-relational-finite-field-evaluation-v1",
                             "exact-available-relational-finite-operator-family-v1",
                             "exact-available-relational-finite-operator-evaluation-v1"}:
            raise ValueError("whole-field program output requires its immutable field archive")
        if self.execution_receipt is not None:
            if not isinstance(self.execution_receipt, Mapping):
                raise TypeError("model execution receipt requires a mapping")
            object.__setattr__(self, "execution_receipt", freeze_tree(self.execution_receipt))
        if self.arithmetic == "algebraic":
            from rexgraph.cpp_algebraic import validate_algebraic_output
            validate_algebraic_output(self)
        if isinstance(self.execution_receipt, Mapping) and "coefficient_features" in self.execution_receipt:
            from rexgraph.cpp_coefficient_features import validate_coefficient_feature_output
            validate_coefficient_feature_output(self)
        if self.method == "packed-boolean-gate-v1":
            from rexgraph.cpp_binary_gate import validate_binary_gate_output
            validate_binary_gate_output(self)
        if self.method in {"exact-context-correspondence-v1", "exact-context-correspondence-v2"}:
            from rexgraph.cpp_correspondence import validate_correspondence_output
            validate_correspondence_output(self)
        if self.method == "exact-context-counts-v1":
            from rexgraph.cpp_context_catalog import validate_context_catalog_output
            validate_context_catalog_output(self)
        if self.method == "exact-relational-stalk-green-v1":
            from rexgraph.cpp_stalk_fields import validate_stalk_output
            validate_stalk_output(self)
        if self.method == "exact-relational-channels-v1":
            from rexgraph.cpp_channels import validate_channel_output
            validate_channel_output(self)
        if self.method == "exact-graded-stalk-boundary-v1":
            from rexgraph.cpp_stalk_boundary import validate_stalk_boundary_output
            validate_stalk_boundary_output(self)
        if self.method == "exact-graded-affine-stalk-transport-v1":
            from rexgraph.cpp_graded_stalk_transport import validate_graded_stalk_transport_output
            validate_graded_stalk_transport_output(self)
        if self.method in {"exact-affine-stalk-transport-v1", "exact-tied-stalk-transport-v2", "exact-column-stalk-transport-v3", "exact-tied-column-stalk-transport-v4"}:
            from rexgraph.cpp_stalk_transport import validate_stalk_transport_output
            validate_stalk_transport_output(self)
        if self.method == "exact-conditional-stalk-field-objective-v1":
            from rexgraph.cpp_stalk_objective import validate_stalk_objective_output
            validate_stalk_objective_output(self)
        if self.method == "exact-stalk-parameter-relations-v1":
            from rexgraph.cpp_stalk_parameter_relations import validate_stalk_parameter_relations_output
            validate_stalk_parameter_relations_output(self)
        if self.method == "exact-joint-stalk-pattern-bank-v1":
            from rexgraph.cpp_stalk_pattern_model import validate_stalk_pattern_bank_output
            validate_stalk_pattern_bank_output(self)
        if self.method == "compatible-sections-whole-stalk-fields-v1":
            from rexgraph.cpp_stalk_sections import validate_stalk_sections_output
            validate_stalk_sections_output(self)
        if self.method == "native-local-tensor-sections-v1":
            from rexgraph.cpp_local_sections import validate_local_sections_output
            validate_local_sections_output(self)
        seal_model(self, _digest, "model output changed")

    @property
    def coefficient_digest(self):
        self.check_state()
        return self._digest

    def check_state(self):
        check_model(self, _digest, "model output changed")

    def tensor(self):
        """Expose rational output without changing numerical predictions into exact claims."""
        self.check_state()
        if self.field_archive is not None:
            raise TypeError("whole-field program output requires explicit native node and stalk selection")
        if self.arithmetic == "algebraic":
            raise TypeError("algebraic model output retains polynomial coordinates and its field declaration")
        if self.arithmetic != "rational":
            raise TypeError("numerical model output is not an exact tensor field")
        if self.method in {"exact-context-correspondence-v1", "exact-context-correspondence-v2"}:
            from rexgraph.cpp_correspondence import correspondence_output_field
            return correspondence_output_field(self)
        if self.method == "exact-relational-stalk-green-v1":
            from rexgraph.cpp_stalk_fields import stalk_output_field
            return stalk_output_field(self)
        if self.method == "exact-graded-stalk-boundary-v1":
            from rexgraph.cpp_stalk_boundary import stalk_boundary_stage
            return stalk_boundary_stage(self)
        if self.method == "exact-graded-affine-stalk-transport-v1":
            from rexgraph.cpp_graded_stalk_transport import graded_stalk_transport_field
            return graded_stalk_transport_field(self)
        if self.method in {"exact-affine-stalk-transport-v1", "exact-tied-stalk-transport-v2", "exact-column-stalk-transport-v3", "exact-tied-column-stalk-transport-v4"}:
            from rexgraph.cpp_stalk_transport import stalk_transport_field
            return stalk_transport_field(self)
        if self.method == "compatible-sections-whole-stalk-fields-v1":
            raise TypeError("compatible interpretations require an explicit case with stalk_section_field")
        if self.method == "exact-conditional-stalk-field-objective-v1":
            raise TypeError("conditional objective output retains both families and requires an explicit interpretation")
        if self.method == "exact-stalk-parameter-relations-v1":
            raise TypeError("parameter relation families require an explicit relation with stalk_parameter_relation")
        if self.method == "exact-joint-stalk-pattern-bank-v1":
            raise TypeError("joint field families require stalk_pattern_bank_result")
        if self.method == "native-local-tensor-sections-v1":
            raise TypeError("ragged local interpretations require an explicit case with local_section_fields")
        return TensorField(self.space, self.values, self.axes, self.source, self.grade, self.variance,
            provenance=(self.model_digest, self.coefficient_digest), dependencies=self.dependencies)

    def recorded_binary(self):
        """Represent the recorded finite binary values as rationals, not exact inference."""
        self.check_state()
        if self.arithmetic == "rational":
            return self.tensor()
        if self.arithmetic == "algebraic":
            raise TypeError("algebraic coefficients require an explicitly selected field embedding")
        values = np.array([Fraction.from_float(float(v)) for v in self.values.flat], dtype=object).reshape(self.values.shape)
        marker = hashlib.sha256(b"recorded_binary_model_output_not_exact_training").hexdigest()
        return TensorField(self.space, values, self.axes, self.source, self.grade, self.variance,
            provenance=(self.model_digest, self.coefficient_digest, marker), dependencies=self.dependencies)


@dataclass(frozen=True, eq=False)
class ModelInput:
    """Declared numerical conversion with the original field and coordinate identity."""
    original: object
    values: object
    space: CoordinateSpace
    axes: tuple[CoordinateSpace, ...]
    source: FieldSource
    origin_digest: str
    origin_arithmetic: str
    dependencies: tuple[FieldSource, ...] = ()
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        _sha(self.origin_digest)
        if self.origin_arithmetic not in {"rational", "approximate"}:
            raise ValueError("input requires the original arithmetic contract")
        if not isinstance(self.space, CoordinateSpace) or any(not isinstance(a, CoordinateSpace) for a in self.axes):
            raise TypeError("model input requires named axes")
        axes = tuple(self.axes)
        if len({a.name for a in (self.space, *axes)}) != len(axes) + 1:
            raise ValueError("model input axes must have distinct names")
        original = freeze_tree(np.asarray(self.original, dtype=object if self.origin_arithmetic == "rational" else None))
        values = freeze_tree(np.asarray(self.values))
        expected = (len(self.space.keys), *(len(a.keys) for a in axes))
        if original.shape != expected or values.shape != expected or values.dtype.kind != "f":
            raise ValueError("model input conversion must preserve every named axis")
        if self.origin_arithmetic == "rational":
            from rexgraph.graded_metric import _fraction
            original = freeze_tree(np.array([_fraction(v) for v in original.flat], dtype=object).reshape(expected))
        if np.asarray(original, dtype=values.dtype).tobytes() != values.tobytes():
            raise ValueError("numerical input differs from its declared conversion")
        object.__setattr__(self, "original", original)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "source", _source(self.source))
        object.__setattr__(self, "dependencies", tuple(_source(v) for v in self.dependencies))
        seal_model(self, _digest, "model input changed")

    @property
    def arithmetic(self):
        return "rounded" if self.origin_arithmetic == "rational" else "approximate"

    @property
    def coefficient_digest(self):
        self.check_state()
        return self._digest

    def check_state(self):
        check_model(self, _digest, "model input changed")


def model_input(value, dtype="float64"):
    """Convert a retained field for numerical evaluation without discarding its source."""
    if not isinstance(value, (TensorField, ModelOutput)):
        raise TypeError("numerical model conversion requires TensorField or ModelOutput")
    value.check_state()
    if value.source is None:
        raise ValueError("numerical model input requires a source reference")
    dt = np.dtype(dtype)
    if dt not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
        raise ValueError("model conversion requires float16, float32 or float64")
    arithmetic = "rational" if isinstance(value, TensorField) else value.arithmetic
    if arithmetic == "algebraic":
        raise TypeError("algebraic coefficients require an explicitly selected field embedding")
    return ModelInput(value.values, np.asarray(value.values, dtype=dt), value.space, value.axes,
        value.source, value.coefficient_digest, arithmetic, value.dependencies)


def _encoded_stalk_targets(target, source, space, mask):
    """Validate source coordinates and observed rows for encoded whole stalk targets."""
    keys = {"profile", "source_reference", "space", "samples", "stalks"}
    if (set(target) != keys or target["profile"] != "encoded-original-stalk-targets-v1" or
            target["space"] != (space.name,space.keys) or
            freeze_tree(target["source_reference"]) != freeze_tree(_source(source).as_record()) or
            type(target["samples"]) is not int or target["samples"] <= 0 or
            type(target["stalks"]) is not tuple or not target["stalks"]):
        raise ValueError("bound original encoded stalk targets required")
    expected = np.zeros(len(space.keys),dtype=bool);previous = -1
    for cell, fields in target["stalks"]:
        if type(cell) is not int or not previous < cell < len(space.keys) or type(fields) is not tuple or not fields:
            raise ValueError("ordered distinct observed original stalk cells required")
        previous = cell;expected[cell] = True;sample_before = -1
        for sample, tensor in fields:
            if (type(sample) is not int or not sample_before < sample < target["samples"] or
                    type(tensor) is not bytes or not tensor):
                raise ValueError("ordered original sample indices and immutable stalk tensor bytes required")
            sample_before = sample
    if not np.array_equal(mask,expected):raise ValueError("observed rows differ from encoded original stalk targets")
    return target


@dataclass(frozen=True, eq=False)
class ModelBatch:
    """Observed numerical or encoded stalk targets on declared source coordinates."""
    source: FieldSource
    space: CoordinateSpace
    targets: object
    observed: object
    inputs: object = None
    dependencies: tuple[FieldSource, ...] = ()
    target_input: ModelInput | None = None
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.space, CoordinateSpace):
            raise TypeError("batch requires named coordinates")
        mask = np.asarray(self.observed)
        if mask.dtype != bool or mask.shape != (len(self.space.keys),):
            raise ValueError("observation mask must be boolean and match rows")
        if isinstance(self.targets,Mapping):
            if self.target_input is not None:raise ValueError("encoded stalk targets retain their own source declaration")
            if self.targets.get("profile") == "rexgraph.relational-field-training-targets.v1":
                from rexgraph.cpp_relational_field_learning import validate_field_training_targets
                target = validate_field_training_targets(self.targets,self.source,self.space,mask)
            elif self.targets.get("profile") == "rexgraph.available-known-product-training.v1":
                from rexgraph.cpp_relational_known_product import validate_known_product_targets
                target = validate_known_product_targets(self.targets,self.source,self.space,mask)
            elif self.targets.get("profile") == "rexgraph.available-relational-field-training-targets.v1":
                from rexgraph.cpp_relational_field_evidence_admission import validate_available_field_training_targets
                target = validate_available_field_training_targets(self.targets,self.source,self.space,mask)
            elif self.targets.get("profile") == "rexgraph.conditioned-occurrence-training.v1":
                from rexgraph.cpp_conditioned_occurrence_program import validate_conditioned_occurrence_targets
                target = validate_conditioned_occurrence_targets(self.targets,self.source,self.space,mask)
            elif self.targets.get("profile") == "rexgraph.conditioned-field-training.v1":
                from rexgraph.cpp_conditioned_field_program import validate_conditioned_field_targets
                target = validate_conditioned_field_targets(self.targets,self.source,self.space,mask)
            else:
                target = _encoded_stalk_targets(self.targets,self.source,self.space,mask)
            declared_target = None
        else:
            declared_target = self.targets if isinstance(self.targets, ModelInput) else self.target_input
            target = np.asarray(self.targets.values if isinstance(self.targets, ModelInput) else self.targets)
            if declared_target is not None:
                if not isinstance(declared_target, ModelInput):raise TypeError("target conversion must be ModelInput")
                declared_target.check_state()
                if declared_target.space != self.space or not declared_target.source.matches(self.source):
                    raise ValueError("target conversion source or coordinates differ")
                if target.dtype != declared_target.values.dtype or target.tobytes() != declared_target.values.tobytes():
                    raise ValueError("target values differ from their declared conversion")
            if target.ndim < 1 or target.shape[0] != len(self.space.keys):
                raise ValueError("target rows must match source coordinates")
        object.__setattr__(self,"target_input",declared_target)
        object.__setattr__(self,"targets",freeze_tree(target))
        object.__setattr__(self,"observed",freeze_tree(mask))
        prepared = freeze_tree(self.inputs)
        def validate(v):
            if isinstance(v, ModelInput):
                if v.space != self.space or not v.source.matches(self.source):
                    raise ValueError("batch input source or coordinates differ")
            elif isinstance(v, Mapping):
                for x in v.values(): validate(x)
            elif isinstance(v, tuple):
                for x in v: validate(x)
        validate(prepared)
        object.__setattr__(self, "inputs", prepared)
        object.__setattr__(self, "source", _source(self.source))
        object.__setattr__(self, "dependencies", tuple(_source(v) for v in self.dependencies))
        seal_model(self, _digest, "model batch changed")

    @property
    def coefficient_digest(self):
        self.check_state()
        return self._digest

    def check_state(self):
        check_model(self, _digest, "model batch changed")


@dataclass(frozen=True, eq=False)
class ModelTimeline:
    """Exact observation times and model references beside a verified temporal index."""
    times: tuple[Fraction, ...]
    model_digests: tuple[str, ...]
    source_digests: tuple[str, ...]
    temporal_header: object
    temporal_tensors: object
    time_axis: str = "model_observation"
    time_unit: str = "step"
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        from rexgraph.graded_metric import _fraction
        from rexgraph.temporal_state import TemporalState, verify_temporal_state
        if any(not isinstance(v, str) or not v for v in (self.time_axis, self.time_unit)):
            raise ValueError("model history requires a named time axis and unit")
        times = tuple(_fraction(v) for v in self.times)
        if not times or any(a >= b for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("model history requires strictly increasing exact times")
        if len(times) != len(self.model_digests) or len(times) != len(self.source_digests):
            raise ValueError("history identities and times disagree")
        for digest in (*self.model_digests, *self.source_digests):
            _sha(digest)
        temporal = TemporalState(dict(self.temporal_tensors), dict(self.temporal_header))
        if not verify_temporal_state(temporal) or temporal.header["T"] != len(times):
            raise ValueError("model timeline requires a verified matching TemporalState")
        if temporal.header["times"] != [float(i) for i in range(len(times))]:
            raise ValueError("model temporal index uses ordinal steps, not rounded event times")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "model_digests", tuple(self.model_digests))
        object.__setattr__(self, "source_digests", tuple(self.source_digests))
        object.__setattr__(self, "temporal_header", freeze_tree(self.temporal_header))
        object.__setattr__(self, "temporal_tensors", freeze_tree(self.temporal_tensors))
        seal_model(self, _digest, "model timeline changed")

    @property
    def coefficient_digest(self):
        self.check_state()
        return self._digest

    def check_state(self):
        check_model(self, _digest, "model timeline changed")

    def at(self, time):
        from bisect import bisect_right

        from rexgraph.graded_metric import _fraction
        index = bisect_right(self.times, _fraction(time)) - 1
        if index < 0:
            raise KeyError("no recorded model exists at this time")
        return index

    def temporal_state(self):
        from rexgraph.temporal_state import TemporalState
        self.check_state()
        # TemporalState uses JSON lists while immutable model declarations use tuples.
        def lists(v):
            if isinstance(v, Mapping): return {k: lists(x) for k, x in v.items()}
            if isinstance(v, tuple): return [lists(x) for x in v]
            return v
        return TemporalState(thaw_tree(self.temporal_tensors), lists(self.temporal_header))
