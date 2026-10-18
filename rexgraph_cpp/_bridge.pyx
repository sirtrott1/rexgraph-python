# distutils: language = c++
"""Owning native source, field and prepared execution handles."""
from collections import OrderedDict
from collections.abc import Mapping
from fractions import Fraction
from hashlib import sha256
from numbers import Integral
from threading import RLock

cimport cython

from libc.stddef cimport size_t
from libc.stdint cimport int64_t, uint16_t, uint32_t, uint64_t
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport move

cdef extern from "chain_bridge.hpp" nogil:
    vector[vector[uint64_t]] observe_context_counts "rexgraph::python::observe_context_counts"(const vector[vector[uint64_t]]&,
        const vector[uint32_t]&, const vector[uint32_t]&, uint64_t) except +
    cdef cppclass ChainTerm "rexgraph::python::ChainTerm":
        uint32_t target
        string coefficient
    cdef cppclass CppChainMap "rexgraph::python::ChainMap":
        @staticmethod
        shared_ptr[CppChainMap] prepare(const shared_ptr[CppSource]&, const shared_ptr[CppSource]&,
            const vector[vector[vector[ChainTerm]]]&, uint64_t, const vector[size_t]&) except +
        vector[string] push(uint16_t, const vector[string]&) except +
        vector[int64_t] push_integer(uint16_t, const vector[int64_t]&) except +
    cdef cppclass LiteralTokenSpan "rexgraph::python::LiteralTokenSpan":
        size_t begin, end
    vector[LiteralTokenSpan] cpp_literal_whitespace_spans "rexgraph::python::literal_whitespace_spans"(
        const string&) except +
    cdef cppclass LiteralTokenChart "rexgraph::python::LiteralTokenChart":
        shared_ptr[CppSource] tokens, bytes
        shared_ptr[CppChainMap] map
        vector[vector[vector[ChainTerm]]] family
        vector[LiteralTokenSpan] spans
    LiteralTokenChart cpp_literal_token_chart "rexgraph::python::literal_token_chart"(
        const string&, const vector[string]&, const vector[LiteralTokenSpan]&, uint64_t,
        size_t, const vector[size_t]&) except +
    LiteralTokenChart cpp_literal_whitespace_chart "rexgraph::python::literal_whitespace_chart"(
        const string&, uint64_t, size_t, const vector[size_t]&) except +

cdef extern from "phase_fold_bridge.hpp" nogil:
    cdef cppclass PhaseFoldOperationInfo "rexgraph::python::PhaseFoldOperationInfo":
        uint64_t identity
        bint active, forward
        size_t crease, edge
        vector[unsigned] moving
        vector[string] phase
    cdef cppclass PhaseFoldProposalInfo "rexgraph::python::PhaseFoldProposalInfo":
        uint64_t key
        size_t position
        bint active
        vector[string] phase
    cdef cppclass PhaseFoldInfo "rexgraph::python::PhaseFoldInfo":
        unsigned kind, grade, index
        bint has_cell
        size_t operations, proposals, degree, endpoint_coordinates, observation_coordinates, branch
        vector[size_t] source_sizes, shape
        vector[string] polynomial, involution, endpoint, positions
        vector[PhaseFoldOperationInfo] operation_views
        vector[PhaseFoldProposalInfo] proposal_views
    cdef cppclass CppPhaseFoldArchive "rexgraph::python::PhaseFoldArchive":
        @staticmethod
        vector[size_t] default_limits() except +
        @staticmethod
        shared_ptr[CppPhaseFoldArchive] load(const char*, size_t, const vector[size_t]&) except +
        @staticmethod
        shared_ptr[CppPhaseFoldArchive] from_source(const shared_ptr[CppSource]&, const vector[uint64_t]&, const vector[size_t]&) except +
        string image() except +
        string source_image() except +
        PhaseFoldInfo info(size_t, bint) except +
        shared_ptr[CppPhaseFoldArchive] prefix(size_t) except +
        shared_ptr[CppPhaseFoldArchive] propose(uint64_t, size_t, const vector[string]&, bint) except +
        shared_ptr[CppPhaseFoldArchive] apply(uint64_t) except +
        bint same_carrier(const shared_ptr[CppPhaseFoldArchive]&) except +

cdef extern from "algebraic_bridge.hpp" nogil:
    cdef cppclass AlgebraicCarrierInfo "rexgraph::python::AlgebraicCarrierInfo":
        bint present
        unsigned grade, index
        size_t rank, degree
    cdef cppclass AlgebraicValueInfo "rexgraph::python::AlgebraicValueInfo":
        unsigned kind
        size_t rows, columns, coefficient_count
        AlgebraicCarrierInfo source, target
        vector[vector[string]] coefficients
    cdef cppclass AlgebraicArchiveInfo "rexgraph::python::AlgebraicArchiveInfo":
        string archive, source, certificate_shift
        size_t degree
        unsigned certificate_kind, embedding_kind, chain_policy
        uint64_t certificate_prime, certificate_order
        bint has_involution, has_metric, has_fibers, flat
        vector[string] polynomial, involution, embedding
        vector[size_t] source_sizes
        vector[uint64_t] record_keys, reconstruction_keys
        vector[unsigned] record_kinds
    cdef cppclass CppAlgebraicValue "rexgraph::python::AlgebraicValue":
        AlgebraicValueInfo info(size_t) except +
        shared_ptr[CppAlgebraicValue] operation(unsigned, const shared_ptr[CppAlgebraicValue]&) except +
    cdef cppclass CppAlgebraicArchive "rexgraph::python::AlgebraicArchive":
        @staticmethod
        vector[size_t] default_limits() except +
        @staticmethod
        shared_ptr[CppAlgebraicArchive] load(const char*, size_t, const vector[size_t]&) except +
        @staticmethod
        shared_ptr[CppAlgebraicArchive] from_source(const shared_ptr[CppSource]&, const vector[uint64_t]&, const vector[size_t]&) except +
        AlgebraicArchiveInfo info() except +
        shared_ptr[CppAlgebraicValue] record(uint64_t) except +
        shared_ptr[CppAlgebraicValue] scalar(const vector[string]&) except +
        shared_ptr[CppAlgebraicValue] vector(uint64_t, unsigned, const vector[vector[string]]&) except +
        shared_ptr[CppAlgebraicValue] reconstruction(uint64_t, unsigned, const shared_ptr[CppAlgebraicValue]&) except +
    cdef cppclass AlgebraicGraphInfo "rexgraph::python::AlgebraicGraphInfo":
        string image
        vector[vector[uint64_t]] keys
    cdef cppclass AlgebraicComparisonInfo "rexgraph::python::AlgebraicComparisonInfo":
        uint64_t source_field, target_field, source_bundle, target_bundle, source_domain, target_domain
        uint64_t scalar_key, carrier_key, source_archive, target_archive, source_record, target_record
        uint64_t accession_key, reconstruction_key
        AlgebraicCarrierInfo source, target
        size_t coefficient_count, action_rows, action_columns
        vector[vector[string]] coefficients
        vector[string] action
    cdef cppclass CppAlgebraicGraph "rexgraph::python::AlgebraicGraph":
        @staticmethod
        vector[size_t] default_limits() except +
        @staticmethod
        shared_ptr[CppAlgebraicGraph] load(const char*, size_t, const vector[size_t]&) except +
        AlgebraicGraphInfo info() except +
        vector[uint64_t] archive_refs(uint64_t) except +
        shared_ptr[CppAlgebraicArchive] archive(uint64_t) except +
        AlgebraicComparisonInfo comparison(unsigned, uint64_t, size_t) except +
        shared_ptr[CppAlgebraicValue] compare(unsigned, uint64_t, unsigned,
                                             const shared_ptr[CppAlgebraicValue]&, uint64_t) except +

cdef extern from "bridge.hpp" nogil:
    cdef cppclass SourceInfo "rexgraph::python::SourceInfo":
        uint64_t name_space, sequence, first_sequence
        string archive, endpoint
        vector[vector[string]] coordinates
        vector[uint64_t] carrier_keys, tensor_keys, tensor_carriers
        vector[unsigned] carrier_grades, carrier_kinds, tensor_roles, tensor_has_output
    cdef cppclass FieldInfo "rexgraph::python::FieldInfo":
        uint64_t key, carrier
        vector[string] values
        vector[size_t] support
    cdef cppclass ExecutionResult "rexgraph::python::ExecutionResult":
        vector[string] values
        uint64_t input_carrier, output_carrier, input_field, name_space, sequence
        uint16_t output_grade
        string profile, implementation, error_message, error_resource
        size_t integer_bits, scalar_work, error_required, error_limit
        unsigned error
    cdef cppclass RationalBatchResult "rexgraph::python::RationalBatchResult":
        ExecutionResult status
        vector[uint64_t] input_fields
        size_t cells, samples, workspace_capacity, workspace_slots
    cdef cppclass CppCancellation "rexgraph::python::Cancellation":
        CppCancellation() except +
        void cancel()
        bint cancelled()
    cdef cppclass CppField "rexgraph::python::Field":
        uint64_t key() except +
        uint64_t carrier() except +
        FieldInfo info() except +
    cdef cppclass CppPlan "rexgraph::python::Plan":
        ExecutionResult execute(const shared_ptr[CppField]&, unsigned, uint16_t,
                                const string&, const shared_ptr[CppCancellation]&) except +
        RationalBatchResult execute_batch(const vector[shared_ptr[CppField]]&, unsigned, uint16_t,
                                          const string&, const shared_ptr[CppCancellation]&) except +
    cdef cppclass IntegerLayer "rexgraph::python::IntegerLayer":
        vector[unsigned] channels
        vector[int64_t] gains
        int64_t bias
        unsigned activation
    cdef cppclass TrainingSample "rexgraph::python::TrainingSample":
        uint64_t input, target, event_time, knowledge_time, label_available_time
        unsigned partition
    cdef cppclass TrainingRequest "rexgraph::python::TrainingRequest":
        uint16_t grade
        vector[unsigned] channels
        vector[TrainingSample] samples
        string model
        uint64_t gains, bias, channel_coordinates, bias_coordinates, history_namespace
        uint64_t metadata_carrier, metadata_tensor, time_coordinates, metadata_coordinates
        uint64_t initial_parameter_knowledge_time, publication_cutoff
        size_t metadata_capacity
    cdef cppclass TrainingInfo "rexgraph::python::TrainingInfo":
        string archive, loss
        vector[string] parameters
        uint64_t step, history_namespace
        uint64_t parameter_knowledge_time
        bint has_loss
    cdef cppclass TrainingExecutionInfo "rexgraph::python::TrainingExecutionInfo":
        string profile
        vector[string] response_channels
        vector[vector[size_t]] parameter_coordinates
        size_t capacity, cached_samples, retained_slots, hits, misses
        size_t scalar_work, integer_bits, rational_slots
    cdef cppclass CppIntegerInput "rexgraph::python::IntegerInput":
        vector[int64_t] values() except +
        uint64_t carrier() except +
        uint16_t grade() except +
    cdef cppclass CppIntegerBatch "rexgraph::python::IntegerBatch":
        vector[int64_t] values() except +
        uint64_t carrier() except +
        uint16_t grade() except +
        size_t cells() except +
        size_t samples() except +
    cdef cppclass IntegerBatchBuild "rexgraph::python::IntegerBatchBuild":
        shared_ptr[CppIntegerBatch] batch
        ExecutionResult status
    cdef cppclass IntegerBatchResult "rexgraph::python::IntegerBatchResult":
        ExecutionResult status
        vector[int64_t] values
        size_t cells, samples, workspace_capacity, workspace_slots
    cdef cppclass CppIntegerNetwork "rexgraph::python::IntegerNetwork":
        ExecutionResult forward(const shared_ptr[CppIntegerInput]&, const shared_ptr[CppCancellation]&) except +
        IntegerBatchResult forward_batch(const shared_ptr[CppIntegerBatch]&, const shared_ptr[CppCancellation]&) except +
    cdef cppclass CppTrainer "rexgraph::python::Trainer"
    cdef cppclass TrainingUpdate "rexgraph::python::TrainingUpdate":
        shared_ptr[CppTrainer] trainer
        ExecutionResult status
    cdef cppclass CppTrainer "rexgraph::python::Trainer":
        TrainingInfo info() except +
        uint64_t parameter_knowledge_time() noexcept
        TrainingExecutionInfo training_execution() except +
        TrainingUpdate advance(const string&, uint64_t, uint64_t, size_t, const shared_ptr[CppCancellation]&) except +
        ExecutionResult forward(const shared_ptr[CppField]&, const shared_ptr[CppCancellation]&) except +
        RationalBatchResult forward_batch(const vector[shared_ptr[CppField]]&, const shared_ptr[CppCancellation]&) except +
    cdef cppclass SourceType "rexgraph::python::SourceType":
        string name
        size_t min_arity, max_arity
    cdef cppclass SourceOccurrence "rexgraph::python::SourceOccurrence":
        uint64_t key, lower
        bint exists, negative
        string magnitude
        vector[unsigned] aspects
    cdef cppclass SourceRelation "rexgraph::python::SourceRelation":
        uint16_t grade
        uint64_t key
        unsigned type
        vector[SourceOccurrence] occurrences
    cdef cppclass SourceCarrier "rexgraph::python::SourceCarrier":
        uint64_t key, coordinates
        unsigned kind
        uint16_t grade
        size_t size
    cdef cppclass SourceField "rexgraph::python::SourceField":
        uint64_t key, carrier
        vector[size_t] indices, support
        vector[string] values
        bint full
    cdef cppclass SourceMetric "rexgraph::python::SourceMetric":
        uint64_t key, carrier
        vector[string] diagonal
    cdef cppclass SourceDeclaration "rexgraph::python::SourceDeclaration":
        uint64_t name_space
        uint16_t max_grade, semantic_scale
        vector[string] role_aspects
        vector[SourceType] types
        vector[uint64_t] witnesses
        vector[SourceRelation] relations
        vector[SourceCarrier] carriers
        vector[SourceField] fields
        vector[SourceMetric] metrics
    cdef cppclass SparseFeature "rexgraph::python::SparseFeature":
        unsigned index
        int64_t value
    cdef cppclass PerceptronInfo "rexgraph::python::PerceptronInfo":
        vector[int64_t] weights
        int64_t bias
        size_t mistakes
    cdef cppclass PackedPerceptronInfo "rexgraph::python::PackedPerceptronInfo":
        vector[int64_t] parameters
        size_t mistakes
    cdef cppclass CppPerceptronRows "rexgraph::python::PerceptronRows":
        size_t rows() except +
        size_t entries() except +
        size_t features() except +
    cdef cppclass PerceptronRowsBuild "rexgraph::python::PerceptronRowsBuild":
        shared_ptr[CppPerceptronRows] rows
        ExecutionResult status
    cdef cppclass CppPerceptron "rexgraph::python::Perceptron"
    cdef cppclass PerceptronUpdate "rexgraph::python::PerceptronUpdate":
        shared_ptr[CppPerceptron] model
        ExecutionResult status
    cdef cppclass PerceptronOutput "rexgraph::python::PerceptronOutput":
        vector[int64_t] labels, scores
        ExecutionResult status
        size_t requested_workers, worker_slots, launched_threads, tiles, rows_per_tile
    cdef cppclass CppPerceptron "rexgraph::python::Perceptron":
        CppPerceptron(const vector[int64_t]&, int64_t, const vector[size_t]&) except +
        PerceptronInfo info() except +
        PackedPerceptronInfo packed_info() except +
        PerceptronRowsBuild prepare_rows(const vector[vector[SparseFeature]]&, const shared_ptr[CppCancellation]&) except +
        PerceptronOutput forward_prepared(const shared_ptr[CppPerceptronRows]&, size_t, size_t, const shared_ptr[CppCancellation]&) except +
        PerceptronOutput forward(const vector[vector[SparseFeature]]&, const shared_ptr[CppCancellation]&) except +
        PerceptronOutput forward_cpu(const vector[vector[SparseFeature]]&, size_t, size_t, const shared_ptr[CppCancellation]&) except +
        PerceptronUpdate advance(const vector[vector[SparseFeature]]&, const vector[int64_t]&, int64_t, const shared_ptr[CppCancellation]&) except +
    cdef cppclass CarrierLink "rexgraph::python::CarrierLink":
        uint64_t source, target
        vector[unsigned] indices
    cdef cppclass TransportInfo "rexgraph::python::TransportInfo":
        vector[vector[size_t]] occurrences
    cdef cppclass CppTransport "rexgraph::python::Transport":
        TransportInfo info() except +
        ExecutionResult push(uint64_t, const vector[string]&, const shared_ptr[CppCancellation]&) except +
        ExecutionResult check_tensor(uint64_t, uint64_t, const shared_ptr[CppCancellation]&) except +
    cdef cppclass TransportBuild "rexgraph::python::TransportBuild":
        shared_ptr[CppTransport] transport
        ExecutionResult status
    cdef cppclass CppSource "rexgraph::python::Source":
        @staticmethod
        shared_ptr[CppSource] load(const char*, size_t, size_t, size_t, size_t, size_t) except +
        @staticmethod
        shared_ptr[CppSource] build(const SourceDeclaration&, size_t) except +
        SourceCarrier carrier(uint64_t) except +
        SourceInfo info() except +
        TransportBuild transport(const shared_ptr[CppSource]&, const vector[vector[unsigned]]&,
                                 const vector[CarrierLink]&, const vector[size_t]&, const shared_ptr[CppCancellation]&) except +
        shared_ptr[CppSource] at(uint64_t) except +
        shared_ptr[CppField] field(uint64_t) except +
        shared_ptr[CppPlan] prepare(const vector[uint64_t]&, const vector[uint64_t]&,
                                   unsigned, bint, size_t, const vector[size_t]&) except +
        shared_ptr[CppIntegerInput] integer_input(uint16_t, uint64_t, const vector[int64_t]&) except +
        IntegerBatchBuild integer_batch(uint16_t, uint64_t, vector[int64_t], size_t, const vector[size_t]&) except +
        shared_ptr[CppIntegerNetwork] integer_network(uint16_t, uint64_t, const vector[IntegerLayer]&, const vector[size_t]&) except +
        PerceptronUpdate perceptron(uint64_t, uint64_t, const vector[size_t]&, const shared_ptr[CppCancellation]&) except +
        shared_ptr[CppTrainer] trainer(const shared_ptr[CppPlan]&, const TrainingRequest&, const string&, size_t) except +
    string cpp_engine_digest "rexgraph::python::engine_digest"() except +
    string cpp_engine_version "rexgraph::python::engine_version"() except +
    string cpp_binding_digest "rexgraph::python::binding_digest"() except +


class ExecutionCancelled(RuntimeError):
    """Native execution was cancelled before its result could be returned."""


class ExecutionLimitError(RuntimeError):
    """A declared native execution resource limit was exceeded."""
    def __init__(self, message, resource, required, limit):
        super().__init__(message)
        self.resource, self.required, self.limit = resource, required, limit


_SIZE_MAX = int(<size_t>-1)


def _integer(value, maximum):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("native keys and resource limits require integers")
    value = int(value)
    if not 0 <= value <= maximum:
        raise ValueError("native integer is outside its declared width")
    return value


def _fraction(bytes value):
    parts = value.decode("ascii").split("/")
    return Fraction(int(parts[0], 16), int(parts[1], 16) if len(parts) == 2 else 1)


def _signed(value):
    if isinstance(value, Fraction):
        if value.denominator != 1:
            raise TypeError("integer coefficient requires an integral rational")
        value = value.numerator
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("integer coefficient requires an integer or integral Fraction")
    value = int(value)
    if not -(1 << 63) <= value < 1 << 63:
        raise OverflowError("integer coefficient is outside signed 64 bit storage")
    return value


cdef void _errors(const ExecutionResult& result) except *:
    if result.error == 1:
        raise ExecutionCancelled((<bytes>result.error_message).decode("utf-8"))
    if result.error == 2:
        raise ExecutionLimitError((<bytes>result.error_message).decode("utf-8"),
            (<bytes>result.error_resource).decode("utf-8"), result.error_required, result.error_limit)


cdef object _output(const ExecutionResult& result):
    _errors(result)
    cdef string value
    return {"values": tuple(_fraction(<bytes>value) for value in result.values),
            "grade": int(result.output_grade), "input_carrier": int(result.input_carrier),
            "output_carrier": int(result.output_carrier), "input_field": int(result.input_field),
            "namespace": int(result.name_space), "sequence": int(result.sequence),
            "profile": (<bytes>result.profile).decode("ascii"),
            "implementation": (<bytes>result.implementation).decode("ascii"),
            "integer_bits": int(result.integer_bits), "scalar_work": int(result.scalar_work)}


cdef object _batch_output(const RationalBatchResult& result):
    output = _output(result.status)
    cdef size_t samples = result.samples
    cdef size_t cells = result.cells
    if not samples or len(output["values"]) != cells * samples:
        raise RuntimeError("native rational batch returned an invalid field shape")
    values = output["values"]
    output["values"] = tuple(values[i*samples:(i+1)*samples] for i in range(cells))
    output.update(input_fields=tuple(result.input_fields), batch_size=int(result.samples),
                  workspace_capacity=int(result.workspace_capacity), workspace_slots=int(result.workspace_slots),
                  layout="cell-major-samples-contiguous-v1")
    return output


cdef vector[shared_ptr[CppField]] _batch_fields(object inputs, size_t limit) except *:
    if not 1 <= len(inputs) <= limit:
        raise ValueError("native rational batch sample count exceeds its record bound")
    cdef vector[shared_ptr[CppField]] fields
    cdef NativeField field
    for value in inputs:
        if not isinstance(value, NativeField):
            raise TypeError("native rational batches require stored NativeField inputs")
        field = value
        field._require()
        if fields.size() >= limit:
            raise ValueError("native rational batch sample count exceeds its record bound")
        fields.push_back(field._handle)
    return fields


def build_info():
    return {"engine_version": (<bytes>cpp_engine_version()).decode("ascii"),
            "engine_digest": (<bytes>cpp_engine_digest()).decode("ascii"),
            "binding_digest": (<bytes>cpp_binding_digest()).decode("ascii"),
            "binding_version": "1", "scalar_policy": "gmp-rational-exact"}


_ALGEBRAIC_LIMIT_NAMES = (
    "max_file_bytes", "max_records", "max_rational_coordinates", "max_integer_bytes",
    "field.max_degree", "field.max_input_coefficients", "field.max_integer_bits",
    "field.max_scalar_steps", "field.max_matrix_coordinates", "modules.max_cells",
    "modules.max_rank", "modules.max_retained_coordinates", "modules.max_matrix_coordinates",
    "fibers.max_occurrences", "fibers.max_retained_map_coordinates", "real_root.max_bisections",
    "complex_root.max_refinements", "complex_root.max_rounding_bits", "complex_root.max_trace_coordinates",
    "metric.max_sign_refinements", "metric.max_factor_steps",
)
_ALGEBRAIC_KINDS = ("vector", "covector", "map", "form", "metric", "scalar")
_ALGEBRAIC_GRAPH_LIMIT_NAMES = tuple("archive." + name for name in _ALGEBRAIC_LIMIT_NAMES) + (
    "max_file_bytes", "max_owner_records", "max_edges", "max_rational_coordinates",
    "measurement.max_coefficient_products")
_ALGEBRAIC_GRAPH_TABLES = ("fields", "involutions", "embeddings", "domains", "bundles", "fibers",
                           "archives", "scalars", "carriers", "metric_comparisons", "reconstructions", "measurements")
_ALGEBRAIC_COMPARISONS = ("scalar", "carrier", "measurement", "reconstruction", "domain")


cdef vector[string] _algebraic_coordinates(object row, size_t degree, object bits) except *:
    if isinstance(row, (str, bytes, Mapping)) or len(row) != degree:
        raise ValueError("algebraic value requires complete polynomial coordinates")
    cdef vector[string] block
    for value in row:
        if isinstance(value, bool) or not isinstance(value, (Integral, Fraction)):
            raise TypeError("algebraic coordinates require integers or Fraction values")
        q = Fraction(int(value)) if isinstance(value, Integral) else value
        if max(abs(q.numerator).bit_length(), q.denominator.bit_length()) > bits:
            raise OverflowError("algebraic coordinate exceeds its native integer bit admission")
        block.push_back((format(q.numerator, "x") + "/" + format(q.denominator, "x")).encode("ascii"))
    return block


cdef vector[size_t] _algebraic_limits(object overrides) except *:
    cdef vector[size_t] result = CppAlgebraicArchive.default_limits()
    if result.size() != len(_ALGEBRAIC_LIMIT_NAMES):
        raise RuntimeError("native algebraic admission fields differ")
    if overrides is None:
        return result
    if not isinstance(overrides, Mapping) or set(overrides) - set(_ALGEBRAIC_LIMIT_NAMES):
        raise ValueError("unknown algebraic archive admission fields")
    for key, value in overrides.items():
        result[_ALGEBRAIC_LIMIT_NAMES.index(key)] = _integer(value, _SIZE_MAX)
    return result


cdef object _algebraic_carrier(const AlgebraicCarrierInfo& value):
    if not value.present:
        return None
    return {"grade": int(value.grade), "index": int(value.index),
            "rank": int(value.rank), "field_degree": int(value.degree)}


cdef object _algebraic_value_info(const AlgebraicValueInfo& value):
    cdef vector[string] row
    cdef string q
    return {"kind": _ALGEBRAIC_KINDS[value.kind],
            "shape": () if value.kind == 5 else (int(value.rows), int(value.columns)),
            "source_carrier": _algebraic_carrier(value.source),
            "target_carrier": _algebraic_carrier(value.target),
            "coefficient_count": int(value.coefficient_count),
            "coefficients": tuple(tuple(_fraction(<bytes>q) for q in row) for row in value.coefficients),
            "truncated": value.coefficients.size() < value.coefficient_count}


cdef class NativeAlgebraicArchive:
    """An immutable RFCXA archive with freshly validated field and carrier owners."""
    cdef shared_ptr[CppAlgebraicArchive] _handle
    cdef bint _loading
    cdef object _archive, _source, _manifest, _limits
    cdef object _graph_owner, _history_owner, _history_channel
    cdef uint64_t _graph_key

    def __init__(self, bytes archive, *, limits=None):
        if self._loading or self._handle.get() != NULL:
            raise RuntimeError("native algebraic archive is immutable")
        cdef vector[size_t] bounds = _algebraic_limits(limits)
        cdef size_t size = len(archive)
        cdef const char* data = archive
        if size < 32 or size > bounds[0]:
            raise ValueError("algebraic archive exceeds its byte admission or is truncated")
        self._loading = True
        try:
            with nogil:
                self._handle = CppAlgebraicArchive.load(data, size, bounds)
            self._initialise(bounds)
        finally:
            self._loading = False

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("native algebraic archive is not initialized")

    cdef void _initialise(self, const vector[size_t]& bounds) except *:
        cdef AlgebraicArchiveInfo value
        cdef string q
        with nogil:
            value = self._handle.get().info()
        self._archive = <bytes>value.archive
        self._source = <bytes>value.source
        self._limits = tuple(bounds)
        if value.certificate_kind == 2:
            certificate = {"kind": "cyclotomic", "order": int(value.certificate_order)}
        elif value.certificate_kind <= 1:
            certificate = {"kind": ("irreducible_reduction", "eisenstein_shift")[value.certificate_kind],
                           "prime": int(value.certificate_prime)}
            if value.certificate_kind == 1:
                certificate["shift"] = int((<bytes>value.certificate_shift).decode("ascii"), 16)
        else:
            raise ValueError("unsupported native algebraic field certificate")
        self._manifest = {
            "codec": "rfcxa-v1", "archive_digest": sha256(self._archive).hexdigest(),
            "source_digest": sha256(self._source).hexdigest(), "field_degree": int(value.degree),
            "minimal_polynomial": tuple(_fraction(<bytes>q) for q in value.polynomial),
            "certificate": certificate,
            "involution": tuple(_fraction(<bytes>q) for q in value.involution) if value.has_involution else None,
            "embedding": None if value.embedding_kind == 0 else {
                "kind": ("real", "complex")[value.embedding_kind - 1],
                "region": tuple(_fraction(<bytes>q) for q in value.embedding)},
            "metric_domain": bool(value.has_metric), "source_sizes": tuple(value.source_sizes),
            "chain_policy": ("require_flat", "declared_deformation")[value.chain_policy] if value.has_fibers else None,
            "flat": bool(value.flat) if value.has_fibers else None,
            "records": tuple((int(key), _ALGEBRAIC_KINDS[kind]) for key, kind in
                             zip(value.record_keys, value.record_kinds)),
            "reconstructions": tuple(value.reconstruction_keys),
        }

    @staticmethod
    def default_limits():
        return dict(zip(_ALGEBRAIC_LIMIT_NAMES, CppAlgebraicArchive.default_limits()))

    @classmethod
    def from_history(cls, NativeSource source not None, channel, *, sequence, limits=None):
        """Read one declared algebraic channel from a complete selected native prefix."""
        source._require()
        if isinstance(channel, (str, bytes, Mapping)) or len(channel) != 3:
            raise ValueError("algebraic history requires three explicit channel names")
        names = tuple(_integer(v, (1 << 64) - 1) for v in channel)
        if not all(names):
            raise ValueError("algebraic history channel names must be nonzero")
        cdef uint64_t cutoff = _integer(sequence, (1 << 64) - 1)
        cdef NativeSource selected = source if cutoff == source._identity["sequence"] else source.at(cutoff)
        cdef vector[uint64_t] native_channel = names
        cdef vector[size_t] bounds = _algebraic_limits(limits)
        cdef NativeAlgebraicArchive result = NativeAlgebraicArchive.__new__(NativeAlgebraicArchive)
        with nogil:
            result._handle = CppAlgebraicArchive.from_source(selected._handle, native_channel, bounds)
        result._initialise(bounds)
        result._history_owner, result._history_channel = selected, names
        return result

    @property
    def field_degree(self):
        self._require()
        return self._manifest["field_degree"]

    @property
    def record_count(self):
        self._require()
        return len(self._manifest["records"])

    @property
    def limits(self):
        self._require()
        return dict(zip(_ALGEBRAIC_LIMIT_NAMES, self._limits))

    @property
    def archive(self):
        self._require()
        return self._archive

    @property
    def source_rfcx(self):
        self._require()
        return self._source

    @property
    def manifest(self):
        self._require()
        from copy import deepcopy
        return deepcopy(self._manifest)

    @property
    def graph_owner(self):
        self._require()
        return self._graph_owner

    @property
    def graph_key(self):
        self._require()
        return None if self._graph_owner is None else int(self._graph_key)

    @property
    def history_owner(self):
        self._require()
        return self._history_owner

    @property
    def history_channel(self):
        self._require()
        return self._history_channel

    def preview(self, max_values=256):
        self._require()
        maximum = _integer(max_values, _SIZE_MAX)
        result = self.manifest
        for key in ("minimal_polynomial", "involution", "source_sizes", "records", "reconstructions"):
            values = result[key]
            if values is not None:
                result[key] = values[:maximum]
                result[key + "_count"] = len(values)
                result[key + "_truncated"] = len(values) > maximum
        return result

    def record(self, key):
        self._require()
        cdef uint64_t selected = _integer(key, (1 << 64) - 1)
        cdef NativeAlgebraicValue result = NativeAlgebraicValue.__new__(NativeAlgebraicValue)
        with nogil:
            result._handle = self._handle.get().record(selected)
        result._owner = self
        return result

    def vector(self, carrier_record, coefficients, *, carrier_side="source"):
        """Construct exact polynomial coordinates on a record's declared module carrier."""
        self._require()
        if carrier_side not in {"source", "target"}:
            raise ValueError("algebraic carrier side must be source or target")
        cdef unsigned side = 0 if carrier_side == "source" else 1
        cdef uint64_t key = _integer(carrier_record, (1 << 64) - 1)
        cdef AlgebraicValueInfo info
        cdef shared_ptr[CppAlgebraicValue] prototype
        with nogil:
            prototype = self._handle.get().record(key)
            info = prototype.get().info(0)
        cdef AlgebraicCarrierInfo module = info.source if side == 0 else info.target
        if not module.present:
            raise TypeError("scalar record has no module carrier")
        if isinstance(coefficients, (str, bytes, Mapping)) or len(coefficients) != module.rank:
            raise ValueError("algebraic vector requires one coefficient block per module coordinate")
        cdef vector[vector[string]] rows
        cdef vector[string] block
        for row in coefficients:
            if isinstance(row, (str, bytes, Mapping)) or len(row) != module.degree:
                raise ValueError("algebraic vector requires complete polynomial coordinates")
            block.clear()
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (Integral, Fraction)):
                    raise TypeError("algebraic coordinates require integers or Fraction values")
                q = Fraction(int(value)) if isinstance(value, Integral) else value
                if max(abs(q.numerator).bit_length(), q.denominator.bit_length()) > self._limits[6]:
                    raise OverflowError("algebraic coordinate exceeds its native integer bit admission")
                block.push_back((format(q.numerator, "x") + "/" + format(q.denominator, "x")).encode("ascii"))
            rows.push_back(block)
        cdef NativeAlgebraicValue result = NativeAlgebraicValue.__new__(NativeAlgebraicValue)
        with nogil:
            result._handle = self._handle.get().vector(key, side, rows)
        result._owner = self
        return result

    def scalar(self, coefficients):
        """Construct an exact field element on this archive's actual scalar owner."""
        self._require()
        cdef vector[string] row = _algebraic_coordinates(
            coefficients, self._manifest["field_degree"], self._limits[6])
        cdef NativeAlgebraicValue result = NativeAlgebraicValue.__new__(NativeAlgebraicValue)
        with nogil:
            result._handle = self._handle.get().scalar(row)
        result._owner = self
        return result

    cdef object _reconstruction(self, object key, unsigned operation, NativeAlgebraicValue value):
        self._require(); value._require()
        cdef uint64_t selected = _integer(key, (1 << 64) - 1)
        cdef NativeAlgebraicValue result = NativeAlgebraicValue.__new__(NativeAlgebraicValue)
        with nogil:
            result._handle = self._handle.get().reconstruction(selected, operation, value._handle)
        result._owner = self
        return result

    def accession(self, key, NativeAlgebraicValue value not None):
        return self._reconstruction(key, 0, value)

    def reconstruct(self, key, NativeAlgebraicValue value not None):
        return self._reconstruction(key, 1, value)

    def project(self, key, NativeAlgebraicValue value not None):
        return self._reconstruction(key, 2, value)


cdef class NativeAlgebraicValue:
    """A typed exact scalar, vector, covector, map, form or metric with native owners."""
    cdef shared_ptr[CppAlgebraicValue] _handle
    cdef object _owner

    def __init__(self):
        raise TypeError("native algebraic values are created by an owning archive or operation")

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("native algebraic value is not initialized")

    @property
    def archive_owner(self):
        self._require()
        return self._owner

    @property
    def coefficients(self):
        self._require()
        cdef size_t maximum = _SIZE_MAX
        cdef AlgebraicValueInfo result
        with nogil:
            result = self._handle.get().info(maximum)
        return _algebraic_value_info(result)["coefficients"]

    @property
    def kind(self):
        return self.preview(0)["kind"]

    @property
    def shape(self):
        return self.preview(0)["shape"]

    def preview(self, max_values=256):
        self._require()
        cdef size_t maximum = _integer(max_values, _SIZE_MAX)
        cdef NativeAlgebraicArchive owner = self._owner
        cdef AlgebraicValueInfo result
        with nogil:
            result = self._handle.get().info(maximum)
        out = _algebraic_value_info(result)
        out.update(archive_digest=owner._manifest["archive_digest"],
                   field_degree=owner._manifest["field_degree"])
        if owner._graph_owner is not None:
            out.update(graph_digest=owner._manifest["graph_digest"], archive_key=int(owner._graph_key))
        return out

    cdef object _operation(self, unsigned operation, NativeAlgebraicValue value):
        self._require(); value._require()
        cdef NativeAlgebraicValue result = NativeAlgebraicValue.__new__(NativeAlgebraicValue)
        with nogil:
            result._handle = self._handle.get().operation(operation, value._handle)
        result._owner = self._owner
        return result

    def apply(self, NativeAlgebraicValue value not None):
        return self._operation(0, value)

    def solve(self, NativeAlgebraicValue value not None):
        return self._operation(1, value)

    def evaluate(self, NativeAlgebraicValue value not None):
        return self._operation(2, value)

    def quadrance(self, NativeAlgebraicValue value not None):
        return self._operation(3, value)

    def as_covector(self, NativeAlgebraicValue value not None):
        return self._operation(4, value)


cdef class NativeAlgebraicArchiveGraph:
    """An immutable RFCXG graph retaining shared fields, carriers and certified comparisons."""
    cdef shared_ptr[CppAlgebraicGraph] _handle
    cdef bint _loading
    cdef object _image, _manifest, _limits, _archives

    def __init__(self, bytes image, *, limits=None):
        if self._loading or self._handle.get() != NULL:
            raise RuntimeError("native algebraic graph is immutable")
        cdef vector[size_t] bounds = CppAlgebraicGraph.default_limits()
        if bounds.size() != len(_ALGEBRAIC_GRAPH_LIMIT_NAMES):
            raise RuntimeError("native algebraic graph admission fields differ")
        if limits is not None:
            if not isinstance(limits, Mapping) or set(limits) - set(_ALGEBRAIC_GRAPH_LIMIT_NAMES):
                raise ValueError("unknown algebraic graph admission fields")
            for name, value in limits.items():
                bounds[_ALGEBRAIC_GRAPH_LIMIT_NAMES.index(name)] = _integer(value, _SIZE_MAX)
        cdef const char* data = image
        cdef size_t size = len(image)
        if size < 32 or size > bounds[21]:
            raise ValueError("algebraic graph exceeds its byte admission or is truncated")
        cdef AlgebraicGraphInfo info
        self._loading = True
        try:
            with nogil:
                self._handle = CppAlgebraicGraph.load(data, size, bounds)
                info = self._handle.get().info()
            self._image = <bytes>info.image
            self._limits = tuple(bounds)
            self._archives = {}
            self._manifest = {"codec": "rfcxg-v1", "graph_digest": sha256(self._image).hexdigest(),
                              "tables": {name: tuple(keys) for name, keys in zip(_ALGEBRAIC_GRAPH_TABLES, info.keys)}}
        finally:
            self._loading = False

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("native algebraic graph is not initialized")

    @staticmethod
    def default_limits():
        return dict(zip(_ALGEBRAIC_GRAPH_LIMIT_NAMES, CppAlgebraicGraph.default_limits()))

    @property
    def limits(self):
        self._require()
        return dict(zip(_ALGEBRAIC_GRAPH_LIMIT_NAMES, self._limits))

    @property
    def image(self):
        self._require()
        return self._image

    @property
    def manifest(self):
        self._require()
        from copy import deepcopy
        return deepcopy(self._manifest)

    def preview(self, max_values=256):
        maximum = _integer(max_values, _SIZE_MAX)
        result = self.manifest
        result["tables"] = {name: {"keys": keys[:maximum], "count": len(keys), "truncated": len(keys) > maximum}
                            for name, keys in result["tables"].items()}
        return result

    def archive(self, key):
        """Select an owning archive view without decoding a separate RFCXA image."""
        self._require()
        cdef uint64_t selected = _integer(key, (1 << 64) - 1)
        if selected in self._archives:
            return self._archives[selected]
        cdef NativeAlgebraicArchive result = NativeAlgebraicArchive.__new__(NativeAlgebraicArchive)
        cdef vector[uint64_t] refs
        cdef vector[size_t] bounds = self._limits[:21]
        bounds[0] = self._limits[21]
        bounds[2] = self._limits[24]
        with nogil:
            result._handle = self._handle.get().archive(selected)
            refs = self._handle.get().archive_refs(selected)
        result._initialise(bounds)
        result._graph_owner = self
        result._graph_key = selected
        result._manifest.update(codec="rfcxg-archive-v1", graph_digest=self._manifest["graph_digest"],
                                archive_key=int(selected), bundle_key=int(refs[0]), field_key=int(refs[1]),
                                domain_key=int(refs[2]))
        return self._archives.setdefault(selected, result)

    def comparison(self, kind, key, max_values=256):
        """Read a named complete comparison with a bounded coefficient and rational action preview."""
        self._require()
        if kind not in _ALGEBRAIC_COMPARISONS:
            raise ValueError("unknown algebraic graph comparison kind")
        cdef unsigned selected_kind = _ALGEBRAIC_COMPARISONS.index(kind)
        cdef uint64_t selected = _integer(key, (1 << 64) - 1)
        cdef size_t maximum = _integer(max_values, _SIZE_MAX)
        cdef AlgebraicComparisonInfo info
        cdef vector[string] row
        cdef string q
        with nogil:
            info = self._handle.get().comparison(selected_kind, selected, maximum)
        return {"kind": kind, "key": int(selected), "graph_digest": self._manifest["graph_digest"],
                "source_field": int(info.source_field), "target_field": int(info.target_field),
                "source_bundle": int(info.source_bundle), "target_bundle": int(info.target_bundle),
                "source_domain": int(info.source_domain), "target_domain": int(info.target_domain),
                "scalar_key": int(info.scalar_key), "carrier_key": int(info.carrier_key),
                "source_archive": int(info.source_archive), "target_archive": int(info.target_archive),
                "source_record": int(info.source_record), "target_record": int(info.target_record),
                "accession_key": int(info.accession_key), "reconstruction_key": int(info.reconstruction_key),
                "source_carrier": _algebraic_carrier(info.source), "target_carrier": _algebraic_carrier(info.target),
                "coefficient_count": int(info.coefficient_count),
                "coefficients": tuple(tuple(_fraction(<bytes>q) for q in row) for row in info.coefficients),
                "coefficients_truncated": info.coefficients.size() < info.coefficient_count,
                "action_shape": (int(info.action_rows), int(info.action_columns)),
                "action": tuple(_fraction(<bytes>q) for q in info.action),
                "action_truncated": info.action.size() < info.action_rows * info.action_columns}

    def compare(self, kind, key, NativeAlgebraicValue value not None, *, target_archive, operation="apply"):
        """Apply a named comparison to an actual source owner and select the declared destination."""
        self._require(); value._require()
        if kind not in _ALGEBRAIC_COMPARISONS or kind == "domain":
            raise ValueError("comparison requires an executable scalar, carrier, measurement or reconstruction")
        allowed = {"scalar": {"apply": 0}, "carrier": {"apply": 0},
                   "measurement": {"apply": 0, "quadrance": 3},
                   "reconstruction": {"accession": 0, "reconstruct": 1, "project": 2}}
        if operation not in allowed[kind]:
            raise ValueError("operation is not defined for this graph comparison kind")
        cdef NativeAlgebraicArchive source = value._owner
        if source._graph_owner is not self:
            raise ValueError("comparison input belongs to another algebraic graph owner")
        cdef uint64_t selected = _integer(key, (1 << 64) - 1)
        cdef uint64_t target = _integer(target_archive, (1 << 64) - 1)
        cdef unsigned selected_kind = _ALGEBRAIC_COMPARISONS.index(kind)
        cdef unsigned selected_operation = allowed[kind][operation]
        cdef NativeAlgebraicArchive owner = self.archive(target)
        cdef NativeAlgebraicValue result = NativeAlgebraicValue.__new__(NativeAlgebraicValue)
        with nogil:
            result._handle = self._handle.get().compare(selected_kind, selected, selected_operation, value._handle, target)
        result._owner = owner
        return result



cdef extern from "algebraic_bridge.hpp" nogil:
    cdef cppclass AlgebraicGraphSourceInfo "rexgraph::python::AlgebraicGraphSourceInfo":
        uint64_t bundle
        unsigned kind
        vector[vector[size_t]] cells, occurrences
    cdef cppclass AlgebraicGraphRevisionInfo "rexgraph::python::AlgebraicGraphRevisionInfo":
        string image
        uint64_t name_space, sequence, anchor
        vector[AlgebraicGraphSourceInfo] sources
        vector[vector[uint64_t]] archives
    cdef cppclass CppAlgebraicGraphRevision "rexgraph::python::AlgebraicGraphRevision":
        @staticmethod
        vector[size_t] default_limits() except +
        @staticmethod
        shared_ptr[CppAlgebraicGraphRevision] capture(const shared_ptr[CppSource]&, const vector[uint64_t]&, const vector[size_t]&) except +
        AlgebraicGraphRevisionInfo info() except +
        shared_ptr[CppAlgebraicGraph] graph() except +
        bint same_owner(const shared_ptr[CppAlgebraicGraphRevision]&) except +
    cdef cppclass AlgebraicGraphDatabaseBuild "rexgraph::python::AlgebraicGraphDatabaseBuild":
        shared_ptr[CppAlgebraicGraphDatabase] database
        ExecutionResult status
    cdef cppclass AlgebraicGraphArchiveSelection "rexgraph::python::AlgebraicGraphArchiveSelection":
        shared_ptr[CppAlgebraicSnapshot] snapshot
        ExecutionResult status
    cdef cppclass CppAlgebraicGraphDatabase "rexgraph::python::AlgebraicGraphDatabase":
        @staticmethod
        AlgebraicGraphDatabaseBuild select(const shared_ptr[CppAlgebraicGraphRevision]&, uint64_t, uint64_t, const vector[size_t]&, const shared_ptr[CppCancellation]&) except +
        AlgebraicGraphArchiveSelection archive(uint64_t, uint64_t, uint64_t, const vector[size_t]&, const shared_ptr[CppCancellation]&) except +
        vector[size_t] requirements() except +
    cdef cppclass AlgebraicRecordInfo "rexgraph::python::AlgebraicRecordInfo":
        uint64_t key
        vector[size_t] domain
        AlgebraicValueInfo value
    cdef cppclass AlgebraicOccurrenceInfo "rexgraph::python::AlgebraicOccurrenceInfo":
        uint64_t key
        string coefficient
        bint active
        shared_ptr[CppAlgebraicValue] underlying, weighted, contribution
    cdef cppclass CppAlgebraicRecord "rexgraph::python::AlgebraicRecord":
        AlgebraicRecordInfo info(size_t) except +
        shared_ptr[CppAlgebraicValue] value() except +
        shared_ptr[CppAlgebraicValue] coordinate(const vector[size_t]&) except +
    cdef cppclass CppAlgebraicSnapshot "rexgraph::python::AlgebraicSnapshot":
        @staticmethod
        shared_ptr[CppAlgebraicSnapshot] publish(const shared_ptr[CppAlgebraicArchive]&, uint64_t, uint64_t) except +
        bint same_publication(const shared_ptr[CppAlgebraicSnapshot]&) except +
        vector[uint64_t] graph_origin() except +
        shared_ptr[CppAlgebraicRecord] record(uint64_t) except +
        AlgebraicOccurrenceInfo occurrence(uint16_t, size_t) except +
    cdef cppclass AlgebraicWindowInfo "rexgraph::python::AlgebraicWindowInfo":
        vector[size_t] domain, source_domain, lower, upper, requirements
        size_t supported_count
    cdef cppclass AlgebraicWindowValues "rexgraph::python::AlgebraicWindowValues":
        vector[vector[string]] coefficients
        vector[size_t] source_indices
    cdef cppclass CppAlgebraicWindow "rexgraph::python::AlgebraicWindow":
        @staticmethod
        AlgebraicWindowBuild prepare(const shared_ptr[CppAlgebraicRecord]&, const vector[size_t]&, const vector[size_t]&,
            uint64_t, uint64_t, const vector[size_t]&, const shared_ptr[CppCancellation]&) except +
        AlgebraicWindowInfo info() except +
        AlgebraicWindowValues values(size_t) except +
        size_t source_index(size_t) except +
        vector[size_t] window_index(size_t) except +
        shared_ptr[CppAlgebraicValue] coordinate(size_t) except +
        AlgebraicSum sum(uint64_t, uint64_t, const vector[size_t]&, const shared_ptr[CppCancellation]&) except +
    cdef cppclass AlgebraicWindowBuild "rexgraph::python::AlgebraicWindowBuild":
        shared_ptr[CppAlgebraicWindow] window
        ExecutionResult status
    cdef cppclass AlgebraicSum "rexgraph::python::AlgebraicSum":
        shared_ptr[CppAlgebraicValue] value
        ExecutionResult status
    cdef cppclass AlgebraicGradeInfo "rexgraph::python::AlgebraicGradeInfo":
        uint16_t grade
        bint derived
        size_t cells, coordinates
        vector[uint64_t] keys
        vector[size_t] offsets, requirements
    cdef cppclass AlgebraicGradeValues "rexgraph::python::AlgebraicGradeValues":
        vector[vector[string]] coefficients
        vector[vector[size_t]] coordinates
    cdef cppclass CppAlgebraicGradeTensor "rexgraph::python::AlgebraicGradeTensor":
        @staticmethod
        AlgebraicGradeBuild select(const shared_ptr[CppAlgebraicSnapshot]&, uint16_t, const vector[uint64_t]&,
            uint64_t, uint64_t, const vector[size_t]&, const shared_ptr[CppCancellation]&) except +
        AlgebraicGradeBuild boundary(uint64_t, uint64_t, const vector[size_t]&, const shared_ptr[CppCancellation]&) except +
        AlgebraicGradeInfo info() except +
        AlgebraicGradeValues values(size_t) except +
        vector[size_t] source_coordinate(size_t) except +
        size_t source_index(uint16_t, uint32_t, size_t) except +
        shared_ptr[CppAlgebraicValue] coordinate(size_t) except +
        shared_ptr[CppAlgebraicValue] stalk(uint32_t) except +
        bint same_owner(const shared_ptr[CppAlgebraicGradeTensor]&) except +
        bint matches_source(const shared_ptr[CppAlgebraicSnapshot]&) except +
    cdef cppclass AlgebraicGradeBuild "rexgraph::python::AlgebraicGradeBuild":
        shared_ptr[CppAlgebraicGradeTensor] tensor
        ExecutionResult status
    cdef cppclass CppAlgebraicGradeWindow "rexgraph::python::AlgebraicGradeWindow":
        @staticmethod
        AlgebraicGradeWindowBuild prepare(const shared_ptr[CppAlgebraicGradeTensor]&, const vector[size_t]&,
            const vector[size_t]&, const vector[size_t]&, uint64_t, uint64_t,
            const vector[size_t]&, const shared_ptr[CppCancellation]&) except +
        AlgebraicWindowInfo info() except +
        AlgebraicGradeValues values(size_t) except +
        size_t source_index(size_t) except +
        vector[size_t] source_coordinate(size_t) except +
        vector[size_t] window_index(size_t) except +
        shared_ptr[CppAlgebraicValue] coordinate(size_t) except +
        AlgebraicSum sum(uint64_t, uint64_t, const vector[size_t]&, const shared_ptr[CppCancellation]&) except +
    cdef cppclass AlgebraicGradeWindowBuild "rexgraph::python::AlgebraicGradeWindowBuild":
        shared_ptr[CppAlgebraicGradeWindow] window
        ExecutionResult status
    cdef cppclass CppAlgebraicMetricField "rexgraph::python::AlgebraicMetricField":
        @staticmethod
        AlgebraicMetricBuild prepare(const shared_ptr[CppAlgebraicSnapshot]&, const vector[uint64_t]&,
            size_t, uint64_t, uint64_t, const shared_ptr[CppCancellation]&) except +
        AlgebraicMetricAction apply(const shared_ptr[CppAlgebraicGradeTensor]&, unsigned,
            uint64_t, uint64_t, const shared_ptr[CppCancellation]&) except +
        AlgebraicSum scalar(const shared_ptr[CppAlgebraicGradeTensor]&, const shared_ptr[CppAlgebraicGradeTensor]&,
            unsigned, uint64_t, uint64_t, const shared_ptr[CppCancellation]&) except +
    cdef cppclass CppAlgebraicMetricValues "rexgraph::python::AlgebraicMetricValues":
        AlgebraicGradeInfo info() except +
        AlgebraicGradeValues values(size_t) except +
        shared_ptr[CppAlgebraicValue] stalk(uint32_t) except +
    cdef cppclass AlgebraicMetricBuild "rexgraph::python::AlgebraicMetricBuild":
        shared_ptr[CppAlgebraicMetricField] field
        ExecutionResult status
    cdef cppclass AlgebraicMetricAction "rexgraph::python::AlgebraicMetricAction":
        shared_ptr[CppAlgebraicMetricValues] values
        ExecutionResult status
    cdef cppclass AlgebraicPredicate "rexgraph::python::AlgebraicPredicate":
        vector[size_t] coordinate
        unsigned operation
        shared_ptr[CppAlgebraicValue] literal
    cdef cppclass CppAlgebraicQuery "rexgraph::python::AlgebraicQuery":
        @staticmethod
        AlgebraicQueryBuild prepare(const shared_ptr[CppAlgebraicSnapshot]&, int,
            const shared_ptr[CppAlgebraicValue]&, const shared_ptr[CppAlgebraicValue]&,
            const vector[AlgebraicPredicate]&, uint64_t, uint64_t, const vector[size_t]&, size_t,
            const shared_ptr[CppCancellation]&) except +
        AlgebraicQueryInfo info() except +
        AlgebraicQueryExecution execute(uint64_t, uint64_t, const vector[size_t]&,
            size_t, size_t, size_t, const shared_ptr[CppCancellation]&, const shared_ptr[CppAlgebraicQueryCache]&) except +
    cdef cppclass AlgebraicQueryBuild "rexgraph::python::AlgebraicQueryBuild":
        shared_ptr[CppAlgebraicQuery] query
        ExecutionResult status
    cdef cppclass AlgebraicQueryInfo "rexgraph::python::AlgebraicQueryInfo":
        size_t candidate_count
        vector[size_t] requirements
    cdef cppclass AlgebraicQueryExecution "rexgraph::python::AlgebraicQueryExecution":
        shared_ptr[CppAlgebraicQueryResult] result
        ExecutionResult status
    cdef cppclass AlgebraicQueryResultInfo "rexgraph::python::AlgebraicQueryResultInfo":
        vector[uint64_t] keys
        uint64_t name_space, max_sequence
        bint cache_hit
        vector[size_t] cpu
    cdef cppclass CppAlgebraicQueryResult "rexgraph::python::AlgebraicQueryResult":
        AlgebraicQueryResultInfo info() except +
        shared_ptr[CppAlgebraicRecord] record(size_t) except +
    cdef cppclass CppAlgebraicQueryCache "rexgraph::python::AlgebraicQueryCache":
        CppAlgebraicQueryCache(size_t) except +
        void clear() except +
        vector[size_t] counts() except +

cdef class CancellationToken:
    cdef shared_ptr[CppCancellation] _handle
    def __cinit__(self):
        self._handle = shared_ptr[CppCancellation](new CppCancellation())
    def cancel(self):
        self._handle.get().cancel()
    @property
    def cancelled(self):
        return bool(self._handle.get().cancelled())


cdef class NativeSource:
    cdef shared_ptr[CppSource] _handle
    cdef bint _loading
    cdef size_t _max_bytes
    cdef object _identity, _archive, _carriers, _tensors, _plans, _plan_lock, _models

    def __cinit__(self):
        self._max_bytes = 256 * 1024 * 1024

    def __init__(self, bytes archive, *, max_bytes=256 * 1024 * 1024,
                 max_dimension=1000000, max_records=1000000, max_logical_size=None):
        if self._loading or self._handle.get() != NULL:
            raise RuntimeError("native source publication is immutable")
        cdef size_t limit = _integer(max_bytes, _SIZE_MAX)
        cdef size_t dimension = _integer(max_dimension, _SIZE_MAX)
        cdef size_t records = _integer(max_records, _SIZE_MAX)
        cdef size_t logical = 0 if max_logical_size is None else _integer(max_logical_size, _SIZE_MAX)
        cdef size_t size = len(archive)
        cdef const char* data = archive
        if not size or size > limit:
            raise ValueError("native source archive exceeds its byte limit or is empty")
        self._loading = True
        try:
            with nogil:
                self._handle = CppSource.load(data, size, limit, dimension, records, logical)
        finally:
            self._loading = False
        self._max_bytes = limit
        self._initialise()

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("native source is not initialized")

    cdef void _initialise(self) except *:
        cdef SourceInfo info = self._handle.get().info()
        cdef vector[string] keys
        cdef string key
        self._archive = <bytes>info.archive
        coordinates = tuple(tuple((<bytes>key).decode("ascii") for key in keys)
                            for keys in info.coordinates)
        self._identity = {"namespace": int(info.name_space), "sequence": int(info.sequence),
                          "first_sequence": int(info.first_sequence), "codec": "rfcxs-v1",
                          "archive_digest": sha256(self._archive).hexdigest(),
                          "endpoint_digest": sha256(<bytes>info.endpoint).hexdigest(),
                          "coordinates": coordinates}
        self._carriers = tuple(zip(info.carrier_keys, info.carrier_grades, info.carrier_kinds))
        self._tensors = tuple((key, carrier if present else None, role)
            for key, carrier, role, present in zip(info.tensor_keys, info.tensor_carriers,
                                                  info.tensor_roles, info.tensor_has_output))
        self._plans = OrderedDict()
        self._models = OrderedDict()
        self._plan_lock = RLock()

    @property
    def identity(self):
        self._require()
        return dict(self._identity)

    @property
    def max_bytes(self):
        self._require()
        return int(self._max_bytes)

    @classmethod
    def from_declaration(cls, declaration, *, max_bytes=256 * 1024 * 1024,
                         max_records=1000000, max_slots=16000000, max_integer_bits=1 << 24):
        from ._declaration import normalise
        values = normalise(declaration, max_records, max_slots, max_integer_bits)
        cdef size_t bound = _integer(max_bytes, _SIZE_MAX)
        cdef SourceDeclaration d
        d.name_space = values["namespace"]; d.max_grade = values["max_grade"]; d.semantic_scale = values["semantic_scale"]
        d.witnesses = values["witnesses"]
        d.role_aspects = values["role_aspects"]
        cdef SourceType t
        cdef SourceOccurrence o
        cdef SourceRelation r
        cdef SourceCarrier c
        cdef SourceField f
        cdef SourceMetric m
        for item in values["types"]:
            t.name, t.min_arity, t.max_arity = item
            d.types.push_back(t)
        for item in values["relations"]:
            r.grade, r.key, r.type = item[:3]
            r.occurrences.clear()
            for occurrence in item[3]:
                o.key, o.lower, o.exists, o.negative, o.magnitude, o.aspects = occurrence
                r.occurrences.push_back(o)
            d.relations.push_back(r)
        for item in values["carriers"]:
            c.key, c.kind, c.grade, c.coordinates, c.size = item
            d.carriers.push_back(c)
        for item in values["fields"]:
            f.key, f.carrier = item[:2]
            f.indices.clear(); f.values.clear(); f.support.clear()
            for entry in item[2]:
                f.indices.push_back(entry[0]); f.values.push_back(entry[1])
            f.full = item[3] is None
            if not f.full: f.support = item[3]
            d.fields.push_back(f)
        for item in values["metrics"]:
            m.key, m.carrier, m.diagonal = item
            d.metrics.push_back(m)
        cdef NativeSource result = NativeSource.__new__(NativeSource)
        with nogil:
            result._handle = CppSource.build(d, bound)
        result._max_bytes = bound
        result._initialise()
        return result

    def carrier(self, key):
        self._require()
        cdef uint64_t selected = _integer(key, (1 << 64) - 1)
        cdef SourceCarrier info
        with nogil:
            info = self._handle.get().carrier(selected)
        return {"key": int(info.key), "kind": ("explicit", "source_grade", "cell_stalk")[info.kind],
                "grade": int(info.grade), "coordinates": int(info.coordinates), "size": int(info.size)}

    def transport(self, NativeSource destination not None, cell_maps, carrier_links, *,
                  limits=(4000000, 16000000, 16000000, 64000000, 1 << 24),
                  CancellationToken cancellation=None):
        self._require(); destination._require()
        bounds = tuple(_integer(v, _SIZE_MAX) for v in limits)
        if len(bounds) != 5:
            raise ValueError("transport requires five resource limits")
        records = len(cell_maps) + len(carrier_links)
        for row in cell_maps:
            records += len(row)
        for link in carrier_links:
            if not isinstance(link, Mapping) or set(link) != {"source", "target", "indices"}:
                raise ValueError("transport carrier link requires source, target and indices")
            records += len(link["indices"])
        if records > bounds[0]:
            raise ValueError("transport declaration record bound exceeded")
        maps = tuple(tuple(_integer(v, (1 << 32)-1) for v in row) for row in cell_maps)
        links = tuple({"source": _integer(link["source"], (1 << 64)-1),
                       "target": _integer(link["target"], (1 << 64)-1),
                       "indices": tuple(_integer(v, (1 << 32)-1) for v in link["indices"])}
                      for link in carrier_links)
        cdef vector[vector[unsigned]] ms = maps
        cdef vector[CarrierLink] ls
        cdef CarrierLink item
        for link in links:
            item.source, item.target, item.indices = link["source"], link["target"], link["indices"]
            ls.push_back(item)
        cdef vector[size_t] policy = bounds
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None: token = cancellation._handle
        cdef TransportBuild built
        with nogil:
            built = self._handle.get().transport(destination._handle, ms, ls, policy, token)
        _errors(built.status)
        cdef NativeTransport result = NativeTransport.__new__(NativeTransport)
        result._handle = built.transport
        result._source, result._destination = self, destination
        result._limits = bounds
        return result

    def chain_map(self, NativeSource destination not None, family, *, denominator=0,
                  limits=(1000000, 64000000, 16384)):
        self._require(); destination._require()
        bounds = tuple(_integer(v, _SIZE_MAX) for v in limits)
        if len(bounds) != 3 or not all(bounds):
            raise ValueError("chain map requires three positive resource limits")
        cdef uint64_t n = _integer(denominator, (1 << 64)-1)
        if n.bit_length() > bounds[2]:
            raise ValueError("chain denominator bit bound exceeded")
        cdef vector[vector[vector[ChainTerm]]] maps
        cdef vector[vector[ChainTerm]] row
        cdef vector[ChainTerm] image
        cdef ChainTerm term
        records = len(family)
        if len(family) != len(self._identity["coordinates"]):
            raise ValueError("chain map requires the complete grade family")
        if len(destination._identity["coordinates"]) < len(family):
            raise ValueError("chain target has insufficient grades")
        for k, grade in enumerate(family):
            if len(grade) != len(self._identity["coordinates"][k]):
                raise ValueError("chain map source cell count differs")
            records += len(grade)
            row.clear()
            for cell in grade:
                records += len(cell)
                if records > bounds[0]:
                    raise ValueError("chain map record bound exceeded")
                image.clear()
                for pair in cell:
                    if len(pair) != 2:
                        raise ValueError("chain term requires target and coefficient")
                    term.target = _integer(pair[0], (1 << 32)-1)
                    if term.target >= len(destination._identity["coordinates"][k]):
                        raise ValueError("chain map target cell index differs")
                    q = _chain_scalar(pair[1], bounds[2])
                    term.coefficient = q
                    image.push_back(term)
                row.push_back(image)
            maps.push_back(row)
        cdef vector[size_t] policy = bounds
        cdef NativeChainMap result = NativeChainMap.__new__(NativeChainMap)
        with nogil:
            result._handle = CppChainMap.prepare(self._handle, destination._handle, maps, n, policy)
        result._source, result._destination, result._limits = self, destination, bounds
        result._denominator = int(n)
        return result

    @property
    def archive(self):
        self._require()
        return self._archive

    @property
    def carriers(self):
        self._require()
        return self._carriers

    @property
    def tensors(self):
        self._require()
        return self._tensors

    def at(self, cutoff):
        self._require()
        cdef uint64_t selected = _integer(cutoff, (1 << 64) - 1)
        cdef NativeSource result = NativeSource.__new__(NativeSource)
        with nogil:
            result._handle = self._handle.get().at(selected)
        result._max_bytes = self._max_bytes
        result._initialise()
        return result

    def field(self, key):
        self._require()
        cdef uint64_t selected = _integer(key, (1 << 64) - 1)
        cdef NativeField result = NativeField.__new__(NativeField)
        with nogil:
            result._handle = self._handle.get().field(selected)
        result._owner = self
        return result

    def prepare(self, grade_carriers, *, metric_keys=(), metric_selection="unit",
                profile="finite", dimension_limit=64,
                limits=(4000000, 16000000, 16000000, 64000000, 1 << 24)):
        self._require()
        if metric_selection not in {"unit", "diagonal", "positive"}:
            raise ValueError("unknown native metric selection")
        if profile not in {"finite", "global"}:
            raise ValueError("unknown native field profile")
        carriers = tuple(_integer(v, (1 << 64) - 1) for v in grade_carriers)
        metrics = tuple(_integer(v, (1 << 64) - 1) for v in metric_keys)
        bounds = tuple(_integer(v, _SIZE_MAX) for v in limits)
        dimension_limit = _integer(dimension_limit, _SIZE_MAX)
        cache_key = (carriers, metrics, metric_selection, profile, dimension_limit, bounds)
        cdef vector[uint64_t] cs = carriers
        cdef vector[uint64_t] ms = metrics
        cdef vector[size_t] ls = bounds
        cdef unsigned selection = {"unit": 0, "diagonal": 1, "positive": 2}[metric_selection]
        cdef bint global_profile = profile == "global"
        cdef size_t cap = dimension_limit
        cdef PreparedField result
        with self._plan_lock:
            if cache_key in self._plans:
                self._plans.move_to_end(cache_key)
                return self._plans[cache_key]
            result = PreparedField.__new__(PreparedField)
            with nogil:
                result._handle = self._handle.get().prepare(cs, ms, selection, global_profile, cap, ls)
            result._batch_limit = bounds[0]
            self._plans[cache_key] = result
            while len(self._plans) > 8:
                self._plans.popitem(last=False)
            return result

    def integer_input(self, carrier, values, *, grade=1):
        self._require()
        cdef uint16_t k = _integer(grade, (1 << 16) - 1)
        cdef uint64_t c = _integer(carrier, (1 << 64) - 1)
        cdef vector[int64_t] xs = tuple(_signed(value) for value in values)
        cdef NativeIntegerInput result = NativeIntegerInput.__new__(NativeIntegerInput)
        with nogil:
            result._handle = self._handle.get().integer_input(k, c, xs)
        result._owner = self
        return result

    def integer_batch_input(self, carrier, const int64_t[:, ::1] values not None, *, grade=1,
                            limits=(4000000, 16000000, 16000000, 64000000, 1 << 24)):
        self._require()
        cdef uint16_t k = _integer(grade, (1 << 16)-1)
        cdef uint64_t c = _integer(carrier, (1 << 64)-1)
        bound = tuple(_integer(v, _SIZE_MAX) for v in limits)
        if len(bound) != 5: raise ValueError("integer batch requires five resource limits")
        cdef size_t cells = values.shape[0]
        cdef size_t samples = values.shape[1]
        if not cells or not samples or cells > _SIZE_MAX // samples:
            raise ValueError("integer batch requires nonempty matching axes")
        if cells * samples > min(bound[:3]): raise ValueError("integer batch coefficient bound exceeded")
        cdef vector[size_t] ls = bound
        cdef vector[int64_t] xs
        xs.assign(&values[0, 0], &values[0, 0] + cells * samples)
        cdef IntegerBatchBuild built
        with nogil: built = self._handle.get().integer_batch(k, c, move(xs), samples, ls)
        _errors(built.status)
        cdef NativeIntegerBatch result = NativeIntegerBatch.__new__(NativeIntegerBatch)
        result._handle = built.batch
        result._owner = self
        return result

    def integer_network(self, carrier, layers, *, grade=1,
                        limits=(4000000, 16000000, 16000000, 64000000, 1 << 24)):
        self._require()
        cdef uint16_t k = _integer(grade, (1 << 16) - 1)
        cdef uint64_t c = _integer(carrier, (1 << 64) - 1)
        cdef vector[size_t] bounds = tuple(_integer(value, _SIZE_MAX) for value in limits)
        cdef vector[IntegerLayer] declarations
        cdef IntegerLayer layer
        signatures = []
        layers = tuple(layers)
        if not 1 <= len(layers) <= 64:
            raise ValueError("integer network requires 1 to 64 layers")
        channels = {"identity": 0, "hodge_down": 1, "hodge_up": 2, "hodge": 3}
        activations = {"identity": 0, "relu": 1, "sign": 2}
        for declaration in layers:
            if set(declaration) != {"channels", "gains", "bias", "activation"}:
                raise ValueError("integer layer requires channels, gains, bias and activation")
            kinds = tuple(declaration["channels"])
            if len(kinds) > 64 or any(kind not in channels for kind in kinds):
                raise ValueError("unsupported integer channel or channel count")
            gains = tuple(_signed(value) for value in declaration["gains"])
            bias = _signed(declaration["bias"])
            activation = declaration["activation"]
            if activation not in activations or len(kinds) != len(gains):
                raise ValueError("integer layer activation or channel/gain mismatch")
            layer.channels = tuple(channels[kind] for kind in kinds)
            layer.gains = gains
            layer.bias = bias
            layer.activation = activations[activation]
            declarations.push_back(layer)
            signatures.append((kinds, gains, bias, activation))
        cache_key = ("integer", int(k), int(c), tuple(signatures), tuple(bounds))
        cdef PreparedIntegerNetwork result
        with self._plan_lock:
            if cache_key in self._models:
                self._models.move_to_end(cache_key)
                return self._models[cache_key]
            result = PreparedIntegerNetwork.__new__(PreparedIntegerNetwork)
            with nogil:
                result._handle = self._handle.get().integer_network(k, c, declarations, bounds)
            self._models[cache_key] = result
            while len(self._models) > 8:
                self._models.popitem(last=False)
            return result

    def perceptron(self, weights_key, bias_key, *,
                   limits=(4000000, 16000000, 16000000, 64000000, 1 << 24),
                   CancellationToken cancellation=None):
        """Load integral parameter fields on full explicit feature and scalar carriers."""
        self._require()
        cdef uint64_t w = _integer(weights_key, (1 << 64)-1)
        cdef uint64_t b = _integer(bias_key, (1 << 64)-1)
        bounds = tuple(_integer(v, _SIZE_MAX) for v in limits)
        if len(bounds) != 5:
            raise ValueError("native perceptron requires five resource limits")
        cdef vector[size_t] policy = bounds
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None: token = cancellation._handle
        cdef PerceptronUpdate built
        with nogil: built = self._handle.get().perceptron(w, b, policy, token)
        _errors(built.status)
        cdef NativePerceptron result = NativePerceptron.__new__(NativePerceptron)
        result._handle = built.model
        result._limits = bounds
        return result

    def trainer(self, PreparedField plan not None, declaration, *, bytes archive=b"",
                max_bytes=256 * 1024 * 1024, cache_samples=8):
        self._require()
        if plan._handle.get() == NULL:
            raise RuntimeError("native execution plan is not initialized")
        names = ("gains", "bias", "channel_coordinates", "bias_coordinates", "history_namespace",
                 "metadata_carrier", "metadata_tensor", "time_coordinates", "metadata_coordinates",
                 "initial_parameter_knowledge_time", "publication_cutoff")
        expected = {*names, "grade", "channels", "samples", "model"}
        if set(declaration) not in (expected, expected | {"metadata_capacity"}):
            raise ValueError("unsupported native training declaration")
        if len(archive) > _integer(max_bytes, _SIZE_MAX):
            raise ValueError("native training archive exceeds its byte limit")
        model = declaration["model"]
        if not isinstance(model, str) or not model or len(model.encode("utf-8")) > 256:
            raise ValueError("native training model name requires 1 to 256 UTF8 bytes")
        values = {name: _integer(declaration[name], (1 << 64) - 1) for name in names}
        channel_names = tuple(declaration["channels"])
        kinds = {"identity": 0, "hodge_down": 1, "hodge_up": 2, "hodge": 3}
        if not 1 <= len(channel_names) <= 64 or any(name not in kinds for name in channel_names):
            raise ValueError("unsupported exact gradient channel or count")
        cdef TrainingRequest request
        request.grade = _integer(declaration["grade"], (1 << 16) - 1)
        request.model = model.encode("utf-8")
        request.channels = tuple(kinds[name] for name in channel_names)
        request.gains = values["gains"]; request.bias = values["bias"]
        request.channel_coordinates = values["channel_coordinates"]; request.bias_coordinates = values["bias_coordinates"]
        request.history_namespace = values["history_namespace"]; request.metadata_carrier = values["metadata_carrier"]
        request.metadata_tensor = values["metadata_tensor"]; request.time_coordinates = values["time_coordinates"]
        request.metadata_coordinates = values["metadata_coordinates"]
        request.initial_parameter_knowledge_time = values["initial_parameter_knowledge_time"]
        request.publication_cutoff = values["publication_cutoff"]
        request.metadata_capacity = _integer(declaration.get("metadata_capacity", 4096), 65536)
        if request.metadata_capacity < 8:
            raise ValueError("training checkpoint capacity must be between 8 and 65536 bytes")
        samples = tuple(declaration["samples"])
        if not 1 <= len(samples) <= 1024:
            raise ValueError("exact training requires 1 to 1024 samples")
        signatures = []
        cdef TrainingSample sample
        sample_names = ("input", "target", "event_time", "knowledge_time", "label_available_time")
        for item in samples:
            if set(item) != {*sample_names, "partition"} or item["partition"] not in {"training", "evaluation"}:
                raise ValueError("unsupported native training sample declaration")
            selected = tuple(_integer(item[name], (1 << 64) - 1) for name in sample_names)
            sample.input, sample.target, sample.event_time, sample.knowledge_time, sample.label_available_time = selected
            sample.partition = 0 if item["partition"] == "training" else 1
            request.samples.push_back(sample)
            signatures.append((selected, item["partition"]))
        cdef size_t cache_capacity = min(_integer(cache_samples, 1024), len(samples))
        signature = ("trainer", plan, request.grade, channel_names, model,
                     tuple(values[name] for name in names), tuple(signatures), int(cache_capacity),
                     int(request.metadata_capacity))
        cache_key = (*signature, archive)
        cdef string data
        cdef NativeTrainer result
        with self._plan_lock:
            if cache_key in self._models:
                self._models.move_to_end(cache_key)
                return self._models[cache_key]
            result = NativeTrainer.__new__(NativeTrainer)
            data = archive
            with nogil:
                result._handle = self._handle.get().trainer(plan._handle, request, data, cache_capacity)
            result._source = self._handle
            result._signature = signature
            result._batch_limit = plan._batch_limit
            self._models[cache_key] = result
            while len(self._models) > 8:
                self._models.popitem(last=False)
            return result

    def remember_trainer(self, NativeTrainer trainer not None):
        self._require()
        trainer._require()
        if trainer._source.get() != self._handle.get():
            raise ValueError("trainer belongs to a different native publication")
        cache_key = (*trainer._signature, trainer.snapshot["archive"])
        with self._plan_lock:
            self._models[cache_key] = trainer
            self._models.move_to_end(cache_key)
            while len(self._models) > 8:
                self._models.popitem(last=False)


cdef class NativeField:
    cdef shared_ptr[CppField] _handle
    cdef object _owner

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("native field is not initialized")

    cdef FieldInfo _info(self) except *:
        self._require()
        return self._handle.get().info()

    @property
    def owner(self):
        self._require()
        return self._owner

    @property
    def key(self):
        self._require()
        return int(self._handle.get().key())

    @property
    def carrier(self):
        self._require()
        return int(self._handle.get().carrier())

    @property
    def values(self):
        cdef FieldInfo info = self._info()
        cdef string value
        return tuple(_fraction(<bytes>value) for value in info.values)

    @property
    def support(self):
        return tuple(self._info().support)


cdef class NativeIntegerInput:
    cdef shared_ptr[CppIntegerInput] _handle
    cdef object _owner

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("native integer input is not initialized")

    @property
    def owner(self):
        self._require()
        return self._owner

    @property
    def values(self):
        self._require()
        return tuple(self._handle.get().values())

    @property
    def carrier(self):
        self._require()
        return int(self._handle.get().carrier())

    @property
    def grade(self):
        self._require()
        return int(self._handle.get().grade())


cdef class NativeIntegerBatch:
    cdef shared_ptr[CppIntegerBatch] _handle
    cdef object _owner

    cdef void _require(self) except *:
        if self._handle.get() == NULL: raise RuntimeError("native integer batch is not initialized")

    @property
    def owner(self):
        self._require()
        return self._owner

    @property
    def carrier(self):
        self._require()
        return int(self._handle.get().carrier())

    @property
    def grade(self):
        self._require()
        return int(self._handle.get().grade())

    @property
    def shape(self):
        self._require()
        return (int(self._handle.get().cells()), int(self._handle.get().samples()))

    @property
    def values(self):
        self._require()
        cdef vector[int64_t] values = self._handle.get().values()
        raw = (<char*>values.data())[:values.size() * sizeof(int64_t)]
        return memoryview(raw).cast("q", shape=self.shape)


cdef class PreparedIntegerNetwork:
    cdef shared_ptr[CppIntegerNetwork] _handle

    def forward(self, NativeIntegerInput inputs not None, *, CancellationToken cancellation=None):
        if self._handle.get() == NULL or inputs._handle.get() == NULL:
            raise RuntimeError("native integer execution handles are not initialized")
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None:
            token = cancellation._handle
        cdef ExecutionResult result
        with nogil:
            result = self._handle.get().forward(inputs._handle, token)
        return _output(result)


    def forward_batch(self, NativeIntegerBatch inputs not None, *, CancellationToken cancellation=None):
        if self._handle.get() == NULL or inputs._handle.get() == NULL:
            raise RuntimeError("native integer batch execution handles are not initialized")
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None: token = cancellation._handle
        cdef IntegerBatchResult result
        with nogil: result = self._handle.get().forward_batch(inputs._handle, token)
        _errors(result.status)
        out = _output(result.status)
        raw = (<char*>result.values.data())[:result.values.size() * sizeof(int64_t)]
        out.update(values=memoryview(raw).cast("q", shape=(result.cells, result.samples)),
                   batch_size=int(result.samples), workspace_capacity=int(result.workspace_capacity),
                   workspace_slots=int(result.workspace_slots), layout="cell-major-samples-contiguous-v1")
        return out


cdef class NativeTrainer:
    cdef shared_ptr[CppTrainer] _handle
    cdef shared_ptr[CppSource] _source
    cdef object _signature
    cdef object _snapshot
    cdef size_t _batch_limit

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("native trainer is not initialized")

    @property
    def snapshot(self):
        self._require()
        cdef TrainingInfo info
        cdef string value
        if self._snapshot is None:
            with nogil:
                info = self._handle.get().info()
            self._snapshot = {"archive": <bytes>info.archive,
                              "parameters": tuple(_fraction(<bytes>value) for value in info.parameters),
                              "last_loss": _fraction(<bytes>info.loss) if info.has_loss else None,
                              "step": int(info.step), "history_namespace": int(info.history_namespace)}
        return dict(self._snapshot)

    @property
    def parameter_knowledge_time(self):
        self._require()
        cdef uint64_t time
        with nogil: time = self._handle.get().parameter_knowledge_time()
        return int(time)

    @property
    def training_execution(self):
        """Return physical training cache counters outside the checkpoint payload."""
        self._require()
        cdef TrainingExecutionInfo info
        cdef string value
        cdef vector[size_t] row
        with nogil: info = self._handle.get().training_execution()
        return dict(profile=info.profile.decode("utf8"), capacity=int(info.capacity),
                    response_channels=tuple(value.decode("utf8") for value in info.response_channels),
                    parameter_coordinates=tuple(tuple(row) for row in info.parameter_coordinates),
                    cached_samples=int(info.cached_samples), retained_slots=int(info.retained_slots),
                    hits=int(info.hits), misses=int(info.misses), scalar_work=int(info.scalar_work),
                    integer_bits=int(info.integer_bits), rational_slots=int(info.rational_slots))

    def advance(self, rate, *, event_cutoff, knowledge_cutoff, steps=1,
                CancellationToken cancellation=None):
        self._require()
        if isinstance(rate, bool) or not isinstance(rate, (int, Fraction)):
            raise TypeError("exact training rate requires an integer or Fraction")
        rate = Fraction(rate)
        cdef string coefficient = (format(rate.numerator, "x") + "/" + format(rate.denominator, "x")).encode("ascii")
        cdef uint64_t events = _integer(event_cutoff, (1 << 64) - 1)
        cdef uint64_t knowledge = _integer(knowledge_cutoff, (1 << 64) - 1)
        cdef size_t count = _integer(steps, 1024)
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None:
            token = cancellation._handle
        cdef TrainingUpdate updated
        with nogil:
            updated = self._handle.get().advance(coefficient, events, knowledge, count, token)
        _errors(updated.status)
        cdef NativeTrainer result = NativeTrainer.__new__(NativeTrainer)
        result._handle = updated.trainer
        result._source = self._source
        result._signature = self._signature
        result._batch_limit = self._batch_limit
        return result

    def forward(self, NativeField inputs not None, *, CancellationToken cancellation=None):
        self._require()
        if inputs._handle.get() == NULL:
            raise RuntimeError("native input field is not initialized")
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None:
            token = cancellation._handle
        cdef ExecutionResult result
        with nogil:
            result = self._handle.get().forward(inputs._handle, token)
        return _output(result)

    def forward_batch(self, inputs, *, CancellationToken cancellation=None):
        self._require()
        cdef vector[shared_ptr[CppField]] fields = _batch_fields(inputs, self._batch_limit)
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None: token = cancellation._handle
        cdef RationalBatchResult result
        with nogil: result = self._handle.get().forward_batch(fields, token)
        return _batch_output(result)


def _chain_scalar(value, bits):
    if isinstance(value, bool) or not isinstance(value, (Integral, Fraction)):
        raise TypeError("chain coefficients require integers or Fraction values")
    q = Fraction(int(value)) if isinstance(value, Integral) else value
    if max(abs(q.numerator).bit_length(), q.denominator.bit_length()) > bits:
        raise ValueError("chain coefficient bit bound exceeded")
    return (format(q.numerator, "x")+"/"+format(q.denominator, "x")).encode("ascii")


@cython.final
def fit_context_counts(counts, features, labels, *, maximum=1000000):
    """Return checked integer counts after the ordered observations."""
    cdef vector[vector[uint64_t]] table
    cdef vector[uint64_t] row
    cdef vector[uint32_t] feature_ids, label_ids
    cdef vector[vector[uint64_t]] result
    cdef uint64_t limit = _integer(maximum, (1 << 64)-1)
    for values in counts:
        row.clear()
        for value in values:
            row.push_back(_integer(value, (1 << 64)-1))
        table.push_back(row)
    for value in features:
        feature_ids.push_back(_integer(value, (1 << 32)-1))
    for value in labels:
        label_ids.push_back(_integer(value, (1 << 32)-1))
    with nogil:
        result = observe_context_counts(table, feature_ids, label_ids, limit)
    return tuple(tuple(result[i][j] for j in range(result[i].size())) for i in range(result.size()))


cdef class NativeChainMap:
    cdef shared_ptr[CppChainMap] _handle
    cdef object _source, _destination, _limits, _denominator

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("native chain map is not initialized")

    @property
    def source(self):
        self._require()
        return self._source

    @property
    def destination(self):
        self._require()
        return self._destination

    @property
    def denominator(self):
        self._require()
        return self._denominator

    def push(self, NativeSource source not None, grade, values, *, integer=False):
        self._require()
        if source is not self._source:
            raise ValueError("chain input requires its actual source owner")
        if type(integer) is not bool:
            raise TypeError("integer chain execution must be explicitly boolean")
        cdef uint16_t k = _integer(grade, (1 << 16)-1)
        if k >= len(source._identity["coordinates"]) or len(values) != len(source._identity["coordinates"][k]):
            raise ValueError("chain input differs from its grade coordinates")
        cdef vector[string] qs, result
        cdef vector[int64_t] ns, integers
        cdef string value
        if integer:
            for item in values:
                _chain_scalar(item, self._limits[2])
                ns.push_back(_signed(item))
            with nogil:
                integers = self._handle.get().push_integer(k, ns)
            return tuple(integers)
        for item in values:
            qs.push_back(_chain_scalar(item, self._limits[2]))
        with nogil:
            result = self._handle.get().push(k, qs)
        return tuple(_fraction(<bytes>value) for value in result)


def literal_token_chart(bytes raw, words, spans, *, namespace=920060,
                        max_bytes=256 * 1024 * 1024, limits=(1000000, 64000000, 16384),
                        retain_family=False):
    """Construct owned byte and token interval complexes with a complete native chain map.

    Spans name ordered nonempty byte intervals matching the supplied Unicode words.
    Other source bytes are retained unchanged. Equal adjacent cuts remain distinct
    token vertices. The admitted profile has at most 8192 bytes and 128 words.
    Each archive obeys max_bytes; limits bound chain records, work and integer bits.
    With retain_family, the fourth result is the complete sparse grade family.
    """
    if not raw or len(raw) > 8192 or not 1 <= len(words) <= 128 or len(words) != len(spans):
        raise ValueError("token chart dimensions")
    if type(retain_family) is not bool:
        raise TypeError("retain_family must be boolean")
    cdef uint64_t ns = _integer(namespace, (1 << 64) - 2)
    cdef size_t byte_limit = _integer(max_bytes, _SIZE_MAX)
    bounds = tuple(_integer(v, _SIZE_MAX) for v in limits)
    if not ns or not byte_limit or len(bounds) != 3 or not all(bounds):
        raise ValueError("token chart namespace or resource limits")
    cdef vector[size_t] policy = bounds
    cdef vector[string] encoded
    cdef vector[LiteralTokenSpan] intervals
    cdef LiteralTokenSpan interval
    cdef string raw_bytes = raw
    for word, span in zip(words, spans, strict=True):
        if not isinstance(word, str):
            raise TypeError("literal token word must be Unicode text")
        if len(span) != 2 or any(type(v) is not int for v in span):
            raise ValueError("literal token span requires two integer byte offsets")
        interval.begin = _integer(span[0], len(raw))
        interval.end = _integer(span[1], len(raw))
        intervals.push_back(interval)
        encoded.push_back(word.encode("utf-8"))
    cdef LiteralTokenChart built
    with nogil:
        built = cpp_literal_token_chart(raw_bytes, encoded, intervals, ns, byte_limit, policy)
    return _literal_chart_result(built, byte_limit, bounds, retain_family, False)


cdef object _literal_chart_result(const LiteralTokenChart& built, size_t byte_limit,
                                 object bounds, bint retain_family, bint retain_spans):
    cdef NativeSource tokens = NativeSource.__new__(NativeSource)
    cdef NativeSource octets = NativeSource.__new__(NativeSource)
    tokens._handle, octets._handle = built.tokens, built.bytes
    tokens._max_bytes = octets._max_bytes = byte_limit
    tokens._initialise(); octets._initialise()
    cdef NativeChainMap mapping = NativeChainMap.__new__(NativeChainMap)
    mapping._handle = built.map
    mapping._source, mapping._destination = tokens, octets
    mapping._limits, mapping._denominator = bounds, 0
    result = (tokens, octets, mapping)
    if retain_spans:
        result += (tuple((int(span.begin), int(span.end)) for span in built.spans),)
    if retain_family:
        unit = Fraction(1)
        family = tuple(tuple(tuple((int(term.target), unit) for term in image) for image in grade)
                       for grade in built.family)
        result += (family,)
    return result


def literal_whitespace_spans(bytes raw):
    """Return maximal original intervals separated by bytes 9 through 13 or 32.

    The input has 1 through 8192 bytes and 1 through 128 nonempty intervals.
    Scanning preserves punctuation, case, non ASCII bytes and binary values.
    This function returns intervals without constructing or serializing sources.
    """
    if not raw or len(raw) > 8192:
        raise ValueError("token chart dimensions")
    cdef string raw_bytes = raw
    cdef vector[LiteralTokenSpan] spans
    with nogil:
        spans = cpp_literal_whitespace_spans(raw_bytes)
    return tuple((int(span.begin), int(span.end)) for span in spans)


def literal_whitespace_chart(bytes raw, *, namespace=920060, max_bytes=256 * 1024 * 1024,
                             limits=(1000000, 64000000, 16384), retain_family=False):
    """Return native interval sources, their complete map and ASCII whitespace spans.

    Bytes 9 through 13 and byte 32 separate maximal nonempty intervals. All bytes
    remain in the raw source. Punctuation, case and non ASCII bytes are unchanged.
    The admitted profile has at most 8192 bytes and 128 intervals. Each archive
    obeys max_bytes; limits bound chain records, work and integer bits.
    With retain_family, the fifth result is the complete sparse grade family.
    """
    if not raw or len(raw) > 8192:
        raise ValueError("token chart dimensions")
    if type(retain_family) is not bool:
        raise TypeError("retain_family must be boolean")
    cdef uint64_t ns = _integer(namespace, (1 << 64) - 2)
    cdef size_t byte_limit = _integer(max_bytes, _SIZE_MAX)
    bounds = tuple(_integer(v, _SIZE_MAX) for v in limits)
    if not ns or not byte_limit or len(bounds) != 3 or not all(bounds):
        raise ValueError("token chart namespace or resource limits")
    cdef vector[size_t] policy = bounds
    cdef string raw_bytes = raw
    cdef LiteralTokenChart built
    with nogil:
        built = cpp_literal_whitespace_chart(raw_bytes, ns, byte_limit, policy)
    return _literal_chart_result(built, byte_limit, bounds, retain_family, True)


cdef class NativeTransport:
    cdef shared_ptr[CppTransport] _handle
    cdef object _source, _destination, _limits

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("native transport is not initialized")

    @property
    def occurrence_maps(self):
        self._require()
        cdef TransportInfo info
        with nogil: info = self._handle.get().info()
        return tuple(tuple(row) for row in info.occurrences)

    def push_values(self, carrier, values, *, CancellationToken cancellation=None):
        self._require()
        if len(values) > self._limits[0]:
            raise ValueError("transport value record bound exceeded")
        cdef uint64_t selected = _integer(carrier, (1 << 64)-1)
        cdef vector[string] vs
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (Integral, Fraction)):
                raise TypeError("transport coefficients require integers or Fraction values")
            value = Fraction(int(value)) if isinstance(value, Integral) else Fraction(
                int(value.numerator), int(value.denominator))
            if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > self._limits[4]:
                raise ValueError("transport coefficient bit bound exceeded")
            vs.push_back((format(value.numerator, "x")+"/"+format(value.denominator, "x")).encode("ascii"))
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None: token = cancellation._handle
        cdef ExecutionResult result
        with nogil: result = self._handle.get().push(selected, vs, token)
        return _output(result)

    def check_tensor(self, source_key, destination_key, *, CancellationToken cancellation=None):
        self._require()
        cdef uint64_t a = _integer(source_key, (1 << 64)-1)
        cdef uint64_t b = _integer(destination_key, (1 << 64)-1)
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None: token = cancellation._handle
        cdef ExecutionResult result
        with nogil: result = self._handle.get().check_tensor(a, b, token)
        _errors(result)


cdef vector[vector[SparseFeature]] _sparse_rows(rows, limits) except *:
    if not 1 <= len(rows) <= limits[0]:
        raise ValueError("sparse integer input exceeds its row bound")
    cdef vector[vector[SparseFeature]] result
    cdef vector[SparseFeature] row
    cdef SparseFeature entry
    result.reserve(len(rows))
    nnz = 0
    for values in rows:
        nnz += len(values)
        if nnz > limits[2]:
            raise ValueError("sparse integer input exceeds its entry bound")
        row.clear()
        row.reserve(len(values))
        for pair in values:
            if len(pair) != 2:
                raise ValueError("sparse feature requires index and integer value")
            entry.index = _integer(pair[0], (1 << 32) - 1)
            entry.value = _signed(pair[1])
            row.push_back(entry)
        result.push_back(row)
    return result


@cython.final
cdef class NativePerceptronRows:
    """Owned ordered sparse integer rows for a declared feature count."""
    cdef shared_ptr[CppPerceptronRows] _handle

    def __init__(self):
        raise TypeError("use NativePerceptron.prepare_rows to construct sparse rows")

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("prepared sparse rows are not initialized")

    @property
    def shape(self):
        self._require()
        return (int(self._handle.get().rows()), int(self._handle.get().features()))

    @property
    def entries(self):
        self._require()
        return int(self._handle.get().entries())


cdef class NativePerceptron:
    cdef shared_ptr[CppPerceptron] _handle
    cdef object _limits
    cdef bint _loading

    def __init__(self, weights, bias=0, *, limits=(4000000, 16000000, 16000000, 64000000, 1 << 24)):
        if self._loading or self._handle.get() != NULL:
            raise RuntimeError("native perceptron is immutable")
        bounds = tuple(_integer(v, _SIZE_MAX) for v in limits)
        if len(bounds) != 5 or len(weights) > bounds[0]:
            raise ValueError("native perceptron exceeds its parameter bound")
        cdef vector[int64_t] parameters = tuple(_signed(v) for v in weights)
        cdef int64_t b = _signed(bias)
        cdef vector[size_t] policy = bounds
        self._loading = True
        try:
            with nogil:
                self._handle = shared_ptr[CppPerceptron](new CppPerceptron(parameters, b, policy))
        finally:
            self._loading = False
        self._limits = bounds

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("native perceptron is not initialized")

    @property
    def snapshot(self):
        self._require()
        cdef PerceptronInfo info
        with nogil: info = self._handle.get().info()
        return {"weights": tuple(info.weights), "bias": int(info.bias), "mistakes": int(info.mistakes)}

    @property
    def packed_snapshot(self):
        """Return owned immutable int64 storage containing weights followed by bias."""
        self._require()
        cdef PackedPerceptronInfo info
        with nogil: info = self._handle.get().packed_info()
        raw = (<char*>info.parameters.data())[:info.parameters.size() * sizeof(int64_t)]
        return {"parameters": memoryview(raw).cast("q"), "mistakes": int(info.mistakes)}

    def prepare_rows(self, rows, *, CancellationToken cancellation=None):
        """Copy and validate integer occurrences for repeated inference."""
        self._require()
        cdef vector[vector[SparseFeature]] values = _sparse_rows(rows, self._limits)
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None: token = cancellation._handle
        cdef PerceptronRowsBuild result
        with nogil: result = self._handle.get().prepare_rows(values, token)
        _errors(result.status)
        cdef NativePerceptronRows prepared = NativePerceptronRows.__new__(NativePerceptronRows)
        prepared._handle = result.rows
        return prepared

    def forward(self, rows, *, cpu_workers=None, rows_per_tile=256, CancellationToken cancellation=None):
        self._require()
        cdef size_t workers = 0, tile_rows = 0
        if cpu_workers is not None:
            workers = _integer(cpu_workers, 64)
            tile_rows = _integer(rows_per_tile, 1000000)
            if not workers or not tile_rows:
                raise ValueError("CPU scoring requires positive workers and rows per tile")
        elif rows_per_tile != 256:
            raise ValueError("rows_per_tile requires an explicit cpu_workers count")
        cdef vector[vector[SparseFeature]] values
        cdef NativePerceptronRows prepared = None
        if isinstance(rows, NativePerceptronRows):
            prepared = rows
            prepared._require()
        else:
            values = _sparse_rows(rows, self._limits)
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None: token = cancellation._handle
        cdef PerceptronOutput result
        with nogil:
            if prepared is not None:
                result = self._handle.get().forward_prepared(prepared._handle, workers, tile_rows, token)
            elif workers:
                result = self._handle.get().forward_cpu(values, workers, tile_rows, token)
            else:
                result = self._handle.get().forward(values, token)
        _errors(result.status)
        output = {"values": tuple(result.labels), "scores": tuple(result.scores)}
        if workers:
            output["cpu_execution"] = {
                "profile": "sparse-integer-cpu-tiles-v1", "requested_workers": result.requested_workers,
                "worker_slots": result.worker_slots, "launched_threads": result.launched_threads,
                "tiles": result.tiles, "rows_per_tile": result.rows_per_tile,
            }
        return output

    def advance(self, rows, labels, *, rate=1, CancellationToken cancellation=None):
        self._require()
        if len(labels) != len(rows):
            raise ValueError("integer labels must match sparse feature rows")
        cdef vector[vector[SparseFeature]] values = _sparse_rows(rows, self._limits)
        cdef vector[int64_t] targets = tuple(_signed(v) for v in labels)
        cdef int64_t step = _signed(rate)
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None: token = cancellation._handle
        cdef PerceptronUpdate result
        with nogil: result = self._handle.get().advance(values, targets, step, token)
        _errors(result.status)
        cdef NativePerceptron child = NativePerceptron.__new__(NativePerceptron)
        child._handle = result.model
        child._limits = self._limits
        return child


cdef class PreparedField:
    cdef shared_ptr[CppPlan] _handle
    cdef size_t _batch_limit

    def execute_batch(self, inputs, *, operation="hodge", grade=1,
                      parameter=Fraction(0), CancellationToken cancellation=None):
        if self._handle.get() == NULL:
            raise RuntimeError("native execution plan is not initialized")
        kinds = {"boundary": 0, "adjoint": 1, "hodge_down": 2, "hodge_up": 3, "hodge": 4}
        if operation not in kinds:
            raise ValueError("unsupported native finite batch operation")
        if isinstance(parameter, bool) or not isinstance(parameter, (int, Fraction)):
            raise TypeError("operation parameter requires an exact integer or Fraction")
        parameter = Fraction(parameter)
        cdef string coefficient = (format(parameter.numerator, "x")+"/"+format(parameter.denominator, "x")).encode("ascii")
        cdef vector[shared_ptr[CppField]] fields = _batch_fields(inputs, self._batch_limit)
        cdef unsigned kind = kinds[operation]
        cdef uint16_t selected_grade = _integer(grade, (1 << 16)-1)
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None: token = cancellation._handle
        cdef RationalBatchResult result
        with nogil: result = self._handle.get().execute_batch(fields, kind, selected_grade, coefficient, token)
        return _batch_output(result)

    def execute(self, NativeField field not None, *, operation="hodge", grade=1,
                parameter=Fraction(0), CancellationToken cancellation=None):
        if self._handle.get() == NULL or field._handle.get() == NULL:
            raise RuntimeError("native execution handles are not initialized")
        kinds = {"boundary": 0, "adjoint": 1, "hodge_down": 2, "hodge_up": 3,
                 "hodge": 4, "green": 6, "harmonic": 7}
        if operation not in kinds:
            raise ValueError("unsupported native field operation")
        if isinstance(parameter, bool) or not isinstance(parameter, (int, Fraction)):
            raise TypeError("operation parameter requires an exact integer or Fraction")
        parameter = Fraction(parameter)
        cdef string coefficient = (format(parameter.numerator, "x") + "/" +
                                   format(parameter.denominator, "x")).encode("ascii")
        cdef unsigned kind = kinds[operation]
        cdef uint16_t selected_grade = _integer(grade, (1 << 16) - 1)
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None:
            token = cancellation._handle
        cdef ExecutionResult result
        with nogil:
            result = self._handle.get().execute(field._handle, kind, selected_grade, coefficient, token)
        if result.error == 1:
            raise ExecutionCancelled((<bytes>result.error_message).decode("utf-8"))
        if result.error == 2:
            raise ExecutionLimitError((<bytes>result.error_message).decode("utf-8"),
                (<bytes>result.error_resource).decode("utf-8"), result.error_required, result.error_limit)
        cdef string value
        values = tuple(_fraction(<bytes>value) for value in result.values)
        return {"values": values, "grade": int(result.output_grade),
                "input_carrier": int(result.input_carrier), "output_carrier": int(result.output_carrier),
                "input_field": int(result.input_field), "namespace": int(result.name_space),
                "sequence": int(result.sequence), "profile": (<bytes>result.profile).decode("ascii"),
                "implementation": (<bytes>result.implementation).decode("ascii"),
                "integer_bits": int(result.integer_bits), "scalar_work": int(result.scalar_work)}

_QUERY_LIMIT_NAMES = ("records", "rational_slots", "sparse_entries", "scalar_work", "integer_bits")
_QUERY_LIMIT_DEFAULTS = (4000000, 16000000, 16000000, 64000000, 1 << 24)
_QUERY_CPU_NAMES = ("requested_workers", "worker_slots", "launched_threads", "tiles", "completed_tiles")


cdef vector[size_t] _query_limits(object overrides, object base=_QUERY_LIMIT_DEFAULTS) except *:
    cdef vector[size_t] bounds = base
    if overrides is not None:
        if not isinstance(overrides, Mapping) or set(overrides) - set(_QUERY_LIMIT_NAMES):
            raise ValueError("unknown algebraic execution admission fields")
        for name, value in overrides.items():
            bounds[_QUERY_LIMIT_NAMES.index(name)] = _integer(value, _SIZE_MAX)
    return bounds


def _query_availability(value):
    if isinstance(value, (str, bytes, Mapping)) or len(value) != 2:
        raise ValueError("availability requires a namespace and maximum sequence")
    ns, sequence = (_integer(v, (1 << 64) - 1) for v in value)
    if not ns:
        raise ValueError("availability namespace must be nonzero")
    return ns, sequence


cdef vector[size_t] _query_indices(object values, size_t rank) except *:
    if isinstance(values, (str, bytes, Mapping)) or len(values) != rank:
        raise ValueError("algebraic coordinate rank differs")
    cdef vector[size_t] out
    for value in values:
        if out.size() >= rank:
            raise ValueError("algebraic coordinate rank differs")
        out.push_back(_integer(value, _SIZE_MAX))
    if out.size() != rank:
        raise ValueError("algebraic coordinate rank differs")
    return out


def _query_admit(required, limits):
    names = ("binding records", "reference rational slots", "reference sparse entries",
             "reference scalar work envelope", "exact integer growth bound")
    for value, limit, name in zip(required, limits, names):
        if value > limit:
            raise ExecutionLimitError(f"{name} requires {value}, limit {limit}", name, value, limit)


cdef object _query_value(shared_ptr[CppAlgebraicValue] handle, NativeAlgebraicArchive owner):
    cdef NativeAlgebraicValue value = NativeAlgebraicValue.__new__(NativeAlgebraicValue)
    value._handle = handle
    value._owner = owner
    return value


_GRAPH_HISTORY_LIMIT_NAMES = _ALGEBRAIC_GRAPH_LIMIT_NAMES + (
    "max_snapshot_bytes", "max_source_bindings", "max_archive_bindings",
    "max_source_coordinates", "max_history_revisions")


cdef class NativeAlgebraicGraphRevision:
    """One actual native endpoint and its anchored RFCXH graph owners."""
    cdef shared_ptr[CppAlgebraicGraphRevision] _handle
    cdef object _source, _channel, _limits, _image, _manifest, _graph
    cdef bint _loading

    def __init__(self, NativeSource source not None, channel, *, sequence, limits=None):
        if self._loading or self._handle.get() != NULL:
            raise RuntimeError("native graph revision is immutable")
        source._require()
        if isinstance(channel, (str, bytes, Mapping)) or len(channel) != 3:
            raise ValueError("graph history requires three nonzero channel names")
        names = tuple(_integer(v, (1 << 64) - 1) for v in channel)
        if not all(names):
            raise ValueError("graph history requires three nonzero channel names")
        cdef uint64_t cutoff = _integer(sequence, (1 << 64) - 1)
        cdef NativeSource selected = source
        cdef vector[uint64_t] native_channel = names
        cdef vector[size_t] bounds = CppAlgebraicGraphRevision.default_limits()
        if bounds.size() != len(_GRAPH_HISTORY_LIMIT_NAMES):
            raise RuntimeError("native graph revision admission fields differ")
        if limits is not None:
            if not isinstance(limits, Mapping) or set(limits) - set(_GRAPH_HISTORY_LIMIT_NAMES):
                raise ValueError("unknown graph revision admission fields")
            for name, value in limits.items():
                bounds[_GRAPH_HISTORY_LIMIT_NAMES.index(name)] = _integer(value, _SIZE_MAX)
        cdef AlgebraicGraphRevisionInfo info
        cdef AlgebraicGraphSourceInfo binding
        cdef AlgebraicGraphInfo graph_info
        cdef NativeAlgebraicArchiveGraph graph = NativeAlgebraicArchiveGraph.__new__(NativeAlgebraicArchiveGraph)
        self._loading = True
        try:
            selected = source if cutoff == source.identity["sequence"] else source.at(cutoff)
            with nogil:
                self._handle = CppAlgebraicGraphRevision.capture(selected._handle, native_channel, bounds)
                info = self._handle.get().info()
                graph._handle = self._handle.get().graph()
                graph_info = graph._handle.get().info()
            graph._image = <bytes>graph_info.image
            graph._limits = tuple(bounds)[:26]
            graph._archives = {}
            graph._manifest = {"codec": "rfcxg-v1", "graph_digest": sha256(graph._image).hexdigest(),
                               "tables": {name: tuple(keys) for name, keys in zip(_ALGEBRAIC_GRAPH_TABLES, graph_info.keys)}}
            self._source, self._channel, self._limits = selected, names, tuple(bounds)
            self._image, self._graph = <bytes>info.image, graph
            self._manifest = {"codec": "rfcxh-v1", "namespace": int(info.name_space), "sequence": int(info.sequence),
                "anchor_bundle": int(info.anchor), "snapshot_digest": sha256(self._image).hexdigest(),
                "graph_digest": graph._manifest["graph_digest"], "archives": tuple(tuple(row) for row in info.archives),
                "sources": tuple({"bundle": int(binding.bundle), "kind": ("independent", "anchor_morphism")[binding.kind],
                                  "cells": tuple(tuple(row) for row in binding.cells),
                                  "occurrences": tuple(tuple(row) for row in binding.occurrences)} for binding in info.sources)}
        finally:
            self._loading = False

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("native graph revision is not initialized")

    @staticmethod
    def default_limits():
        return dict(zip(_GRAPH_HISTORY_LIMIT_NAMES, CppAlgebraicGraphRevision.default_limits()))

    @property
    def limits(self):
        self._require()
        return dict(zip(_GRAPH_HISTORY_LIMIT_NAMES, self._limits))

    @property
    def history_owner(self):
        self._require()
        return self._source

    @property
    def history_channel(self):
        self._require()
        return self._channel

    @property
    def image(self):
        self._require()
        return self._image

    @property
    def graph_owner(self):
        self._require()
        return self._graph

    @property
    def manifest(self):
        self._require()
        from copy import deepcopy
        return deepcopy(self._manifest)

    def same_owner(self, NativeAlgebraicGraphRevision other not None):
        self._require()
        other._require()
        return bool(self._handle.get().same_owner(other._handle))

    def preview(self, max_values=256):
        self._require()
        maximum = _integer(max_values, _SIZE_MAX)
        result = self.manifest
        sources = result.pop("sources")
        coordinates = 0
        rows = []
        for binding in sources[:maximum]:
            row = {"bundle": binding["bundle"], "kind": binding["kind"]}
            for name in ("cells", "occurrences"):
                values = binding[name]
                shown = []
                for grade in values[:maximum]:
                    take = min(len(grade), max(0, maximum - coordinates))
                    shown.append(grade[:take])
                    coordinates += take
                row[name] = tuple(shown)
                row[name + "_counts"] = tuple(map(len, values))[:maximum]
            rows.append(row)
        result.update(sources=tuple(rows), source_count=len(sources), archive_count=len(result["archives"]),
                      limits=self.limits)
        result["archives"] = result["archives"][:maximum]
        result["truncated"] = len(sources) > maximum or result["archive_count"] > maximum or sum(
            len(v) for binding in sources for name in ("cells", "occurrences") for v in binding[name]) > maximum
        return result


cdef class NativeAlgebraicGraphDatabase:
    """Stable native archive publications selected from one graph revision."""
    cdef shared_ptr[CppAlgebraicGraphDatabase] _handle
    cdef object _revision, _limits, _requirements
    cdef bint _loading

    def __init__(self, NativeAlgebraicGraphRevision revision not None, *, availability,
                 execution_limits=None, CancellationToken cancellation=None):
        if self._loading or self._handle.get() != NULL:
            raise RuntimeError("native graph database is immutable")
        revision._require()
        cdef vector[size_t] bounds = _query_limits(execution_limits)
        cdef uint64_t ns, cutoff
        ns, cutoff = _query_availability(availability)
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None:
            token = cancellation._handle
        cdef AlgebraicGraphDatabaseBuild built
        cdef vector[size_t] required
        self._loading = True
        try:
            with nogil:
                built = CppAlgebraicGraphDatabase.select(revision._handle, ns, cutoff, bounds, token)
            _errors(built.status)
            self._handle = built.database
            with nogil:
                required = self._handle.get().requirements()
            self._revision, self._limits, self._requirements = revision, tuple(bounds), tuple(required)
        finally:
            self._loading = False

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("native graph database is not initialized")

    @property
    def revision_owner(self):
        self._require()
        return self._revision

    @property
    def requirements(self):
        self._require()
        return dict(zip(_QUERY_LIMIT_NAMES, self._requirements))

    def archive(self, key, *, availability, execution_limits=None, CancellationToken cancellation=None):
        """Recheck cutoff, cancellation and resource limits before returning a native pin."""
        self._require()
        cdef uint64_t selected = _integer(key, (1 << 64) - 1)
        cdef uint64_t ns, cutoff
        ns, cutoff = _query_availability(availability)
        policy = dict(zip(_QUERY_LIMIT_NAMES, self._limits))
        if execution_limits is not None:
            if not isinstance(execution_limits, Mapping) or set(execution_limits) - set(policy):
                raise ValueError("unknown algebraic execution admission fields")
            policy.update({name: min(policy[name], _integer(value, _SIZE_MAX)) for name, value in execution_limits.items()})
        cdef vector[size_t] bounds = _query_limits(policy)
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None:
            token = cancellation._handle
        cdef AlgebraicGraphArchiveSelection built
        cdef NativeAlgebraicSnapshot result = NativeAlgebraicSnapshot.__new__(NativeAlgebraicSnapshot)
        cdef NativeAlgebraicGraphRevision revision = self._revision
        with nogil:
            built = self._handle.get().archive(selected, ns, cutoff, bounds, token)
        _errors(built.status)
        result._handle = built.snapshot
        result._archive = revision._graph.archive(selected)
        result._namespace = revision._manifest["namespace"]
        result._sequence = revision._manifest["sequence"]
        result._graph_revision = revision
        return result

    def preview(self, max_values=256):
        self._require()
        return {"profile": "native-algebraic-graph-database-v1", "revision": self._revision.preview(max_values),
                "requirements": self.requirements, "execution_limits": dict(zip(_QUERY_LIMIT_NAMES, self._limits))}


cdef class NativeAlgebraicSnapshot:
    """An owning archive publication with explicitly declared namespace and sequence labels."""
    cdef shared_ptr[CppAlgebraicSnapshot] _handle
    cdef object _archive, _namespace, _sequence, _graph_revision
    cdef bint _loading

    def __init__(self, NativeAlgebraicArchive archive not None, *, namespace, sequence):
        if self._loading or self._handle.get() != NULL:
            raise RuntimeError("algebraic snapshot publication is immutable")
        archive._require()
        cdef uint64_t ns = _integer(namespace, (1 << 64) - 1)
        cdef uint64_t seq = _integer(sequence, (1 << 64) - 1)
        if archive._history_owner is not None:
            identity = archive._history_owner.identity
            if ns != identity["namespace"] or seq != identity["sequence"]:
                raise ValueError("algebraic publication differs from its selected native history endpoint")
        self._loading = True
        try:
            with nogil:
                self._handle = CppAlgebraicSnapshot.publish(archive._handle, ns, seq)
            self._archive, self._namespace, self._sequence = archive, int(ns), int(seq)
        finally:
            self._loading = False

    @classmethod
    def from_history(cls, NativeSource source not None, channel, *, sequence, limits=None):
        """Publish an algebraic channel at its actual selected native endpoint."""
        archive = NativeAlgebraicArchive.from_history(source, channel, sequence=sequence, limits=limits)
        return cls(archive, namespace=archive.history_owner.identity["namespace"], sequence=sequence)

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("algebraic snapshot is not initialized")

    @property
    def archive_owner(self):
        self._require()
        return self._archive

    @property
    def namespace(self):
        self._require()
        return self._namespace

    @property
    def sequence(self):
        self._require()
        return self._sequence

    @property
    def graph_revision_owner(self):
        self._require()
        return self._graph_revision

    @property
    def graph_origin(self):
        self._require()
        cdef vector[uint64_t] origin
        with nogil:
            origin = self._handle.get().graph_origin()
        if origin.empty():
            return None
        return dict(zip(("archive_key", "bundle_key", "anchor_bundle", "namespace", "sequence"), origin))

    def same_publication(self, NativeAlgebraicSnapshot other not None):
        self._require()
        other._require()
        return bool(self._handle.get().same_publication(other._handle))

    def record(self, key):
        self._require()
        cdef uint64_t selected = _integer(key, (1 << 64) - 1)
        cdef NativeAlgebraicRecord result = NativeAlgebraicRecord.__new__(NativeAlgebraicRecord)
        with nogil:
            result._handle = self._handle.get().record(selected)
        result._snapshot = self
        return result

    def occurrence(self, grade, slot):
        self._require()
        cdef uint16_t k = _integer(grade, (1 << 16) - 1)
        cdef size_t index = _integer(slot, _SIZE_MAX)
        cdef AlgebraicOccurrenceInfo info
        with nogil:
            info = self._handle.get().occurrence(k, index)
        return {"grade": int(k), "slot": int(index), "key": int(info.key), "active": bool(info.active),
                "coefficient": _fraction(<bytes>info.coefficient),
                "underlying": _query_value(info.underlying, self._archive),
                "weighted": _query_value(info.weighted, self._archive),
                "contribution": _query_value(info.contribution, self._archive)}

    def preview(self, max_values=256):
        self._require()
        count = _integer(max_values, _SIZE_MAX)
        details = self._archive.preview(count)
        details.update(profile="algebraic-snapshot-publication-v1", namespace=self._namespace,
                       sequence=self._sequence, native_history=self._archive.history_owner is not None or self._graph_revision is not None)
        if self._graph_revision is not None:
            details["graph_origin"] = self.graph_origin
        if self._archive.history_owner is not None:
            identity = self._archive.history_owner.identity
            details["history"] = {"namespace": identity["namespace"], "sequence": identity["sequence"],
                "first_sequence": identity["first_sequence"], "archive_digest": identity["archive_digest"],
                "endpoint_digest": identity["endpoint_digest"], "channel": self._archive.history_channel}
        return details


cdef class NativeAlgebraicRecord:
    """An aliasing pin to one supported record in an owning algebraic snapshot."""
    cdef shared_ptr[CppAlgebraicRecord] _handle
    cdef object _snapshot

    def __init__(self):
        raise TypeError("algebraic records are obtained from snapshots or query results")

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("algebraic record pin is not initialized")

    @property
    def snapshot_owner(self):
        self._require()
        return self._snapshot

    def preview(self, max_values=256):
        self._require()
        cdef size_t maximum = _integer(max_values, _SIZE_MAX)
        cdef AlgebraicRecordInfo info
        with nogil:
            info = self._handle.get().info(maximum)
        result = _algebraic_value_info(info.value)
        result.update(key=int(info.key), domain=tuple(info.domain), namespace=self._snapshot.namespace,
                      sequence=self._snapshot.sequence)
        return result

    @property
    def key(self):
        return self.preview(0)["key"]

    @property
    def domain(self):
        return self.preview(0)["domain"]

    @property
    def coefficients(self):
        return self.preview(_SIZE_MAX)["coefficients"]

    @property
    def value(self):
        self._require()
        cdef shared_ptr[CppAlgebraicValue] handle
        with nogil:
            handle = self._handle.get().value()
        return _query_value(handle, self._snapshot.archive_owner)

    def coordinate(self, indices):
        self._require()
        cdef vector[size_t] position = _query_indices(indices, len(self.domain))
        cdef shared_ptr[CppAlgebraicValue] handle
        with nogil:
            handle = self._handle.get().coordinate(position)
        return _query_value(handle, self._snapshot.archive_owner)


cdef class NativeAlgebraicTensorWindow:
    """An owning row major coefficient box with supported zeros and exact contraction."""
    cdef shared_ptr[CppAlgebraicWindow] _handle
    cdef object _record, _info, _limits
    cdef bint _loading

    def __init__(self, NativeAlgebraicRecord record not None, lower, upper, *, availability,
                 execution_limits=None, CancellationToken cancellation=None):
        if self._loading or self._handle.get() != NULL:
            raise RuntimeError("algebraic tensor window is immutable")
        record._require()
        if cancellation is not None and cancellation.cancelled:
            raise ExecutionCancelled("exact execution cancelled at a phase boundary")
        cdef vector[size_t] bounds = _query_limits(execution_limits)
        cdef size_t rank = len(record.domain)
        _query_admit((1 + 4 * rank, 0, 0, 0, 0), tuple(bounds))
        cdef vector[size_t] lo = _query_indices(lower, rank)
        cdef vector[size_t] hi = _query_indices(upper, rank)
        cdef uint64_t ns, cutoff
        ns, cutoff = _query_availability(availability)
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None:
            token = cancellation._handle
        cdef AlgebraicWindowBuild built
        cdef AlgebraicWindowInfo info
        self._loading = True
        try:
            with nogil:
                built = CppAlgebraicWindow.prepare(record._handle, lo, hi, ns, cutoff, bounds, token)
            _errors(built.status)
            self._handle = built.window
            with nogil:
                info = self._handle.get().info()
            self._record = record
            self._limits = tuple(bounds)
            self._info = {"domain": tuple(info.domain), "source_domain": tuple(info.source_domain),
                          "lower": tuple(info.lower), "upper": tuple(info.upper),
                          "supported_count": int(info.supported_count), "requirements": tuple(info.requirements)}
        finally:
            self._loading = False

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("algebraic tensor window is not initialized")

    @property
    def record_owner(self):
        self._require()
        return self._record

    @property
    def domain(self):
        self._require()
        return self._info["domain"]

    @property
    def supported_count(self):
        self._require()
        return self._info["supported_count"]

    @property
    def requirements(self):
        self._require()
        return dict(zip(_QUERY_LIMIT_NAMES, self._info["requirements"]))

    @property
    def execution_limits(self):
        self._require()
        return dict(zip(_QUERY_LIMIT_NAMES, self._limits))

    def source_index(self, coordinate):
        self._require()
        cdef size_t position = _integer(coordinate, _SIZE_MAX)
        return int(self._handle.get().source_index(position))

    def window_index(self, coordinate):
        self._require()
        cdef size_t position = _integer(coordinate, _SIZE_MAX)
        cdef vector[size_t] result = self._handle.get().window_index(position)
        return int(result[0]) if result.size() else None

    def coordinate(self, coordinate):
        self._require()
        cdef size_t position = _integer(coordinate, _SIZE_MAX)
        cdef shared_ptr[CppAlgebraicValue] handle
        with nogil:
            handle = self._handle.get().coordinate(position)
        return _query_value(handle, self._record.snapshot_owner.archive_owner)

    def sum(self, *, availability, execution_limits=None, CancellationToken cancellation=None):
        self._require()
        cdef vector[size_t] bounds = _query_limits(execution_limits, self._limits)
        cdef uint64_t ns, cutoff
        ns, cutoff = _query_availability(availability)
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None:
            token = cancellation._handle
        cdef AlgebraicSum result
        with nogil:
            result = self._handle.get().sum(ns, cutoff, bounds, token)
        _errors(result.status)
        if result.value.get() == NULL:
            return None
        return _query_value(result.value, self._record.snapshot_owner.archive_owner)

    def preview(self, max_values=256):
        self._require()
        cdef size_t maximum = _integer(max_values, _SIZE_MAX)
        count = min(maximum, self.supported_count)
        cdef AlgebraicWindowValues values
        cdef vector[string] row
        cdef string q
        with nogil:
            values = self._handle.get().values(maximum)
        result = {key: value for key, value in self._info.items() if key != "requirements"}
        result.update(profile="algebraic-coefficient-window-v1", record_key=self._record.key,
                      namespace=self._record.snapshot_owner.namespace, sequence=self._record.snapshot_owner.sequence,
                      requirements=self.requirements, execution_limits=self.execution_limits,
                      coefficients=tuple(tuple(_fraction(<bytes>q) for q in row) for row in values.coefficients),
                      source_indices=tuple(values.source_indices),
                      truncated=count < self.supported_count)
        return result


cdef object _grade_info(AlgebraicGradeInfo info):
    return {"grade": int(info.grade), "derived": bool(info.derived), "cell_count": int(info.cells),
            "domain": (int(info.coordinates),), "record_keys": tuple(info.keys),
            "stalk_offsets": tuple(info.offsets), "requirements": tuple(info.requirements)}


cdef vector[size_t] _grade_limits(object overrides, object base) except *:
    cdef vector[size_t] result = _query_limits(overrides, base)
    for i in range(len(base)):
        result[i] = min(result[i], base[i])
    return result


cdef class NativeAlgebraicGradeTensor:
    """One exact vector per original cell, with native stalk offsets and boundary lineage."""
    cdef shared_ptr[CppAlgebraicGradeTensor] _handle
    cdef object _snapshot, _input, _info, _limits
    cdef bint _loading

    def __init__(self, NativeAlgebraicSnapshot snapshot not None, grade, record_keys, *, availability,
                 execution_limits=None, CancellationToken cancellation=None):
        cdef vector[size_t] bounds
        cdef vector[uint64_t] keys
        cdef uint16_t k
        cdef uint64_t ns, cutoff
        cdef size_t count
        cdef shared_ptr[CppCancellation] token
        cdef AlgebraicGradeBuild built
        cdef AlgebraicGradeInfo info
        if self._loading or self._handle.get() != NULL:
            raise RuntimeError("algebraic grade tensor is immutable")
        self._loading = True
        try:
            snapshot._require()
            if cancellation is not None and cancellation.cancelled:
                raise ExecutionCancelled("exact execution cancelled at a phase boundary")
            bounds = _query_limits(execution_limits)
            k = _integer(grade, (1 << 16) - 1)
            if isinstance(record_keys, (str, bytes, Mapping)):
                raise TypeError("grade tensor keys require a sized sequence")
            count = len(record_keys)
            _query_admit((2 + 3 * int(count), 0, 0, 0, 0), tuple(bounds))
            for key in record_keys:
                if keys.size() >= count:
                    raise ValueError("grade tensor key count changed")
                keys.push_back(_integer(key, (1 << 64) - 1))
            if keys.size() != count:
                raise ValueError("grade tensor key count changed")
            ns, cutoff = _query_availability(availability)
            if cancellation is not None:
                token = cancellation._handle
            with nogil:
                built = CppAlgebraicGradeTensor.select(snapshot._handle, k, keys, ns, cutoff, bounds, token)
            _errors(built.status)
            with nogil:
                info = built.tensor.get().info()
            self._info, self._limits = _grade_info(info), tuple(bounds)
            self._snapshot, self._input = snapshot, None
            self._handle = built.tensor
        finally:
            self._loading = False

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("algebraic grade tensor is not initialized")

    @property
    def snapshot_owner(self):
        self._require()
        return self._snapshot

    @property
    def input_owner(self):
        self._require()
        return self._input

    @property
    def grade(self):
        self._require()
        return self._info["grade"]

    @property
    def cell_count(self):
        self._require()
        return self._info["cell_count"]

    @property
    def domain(self):
        self._require()
        return self._info["domain"]

    @property
    def record_keys(self):
        self._require()
        return self._info["record_keys"]

    @property
    def stalk_offsets(self):
        self._require()
        return self._info["stalk_offsets"]

    @property
    def derived(self):
        self._require()
        return self._info["derived"]

    @property
    def requirements(self):
        self._require()
        return dict(zip(_QUERY_LIMIT_NAMES, self._info["requirements"]))

    @property
    def execution_limits(self):
        self._require()
        return dict(zip(_QUERY_LIMIT_NAMES, self._limits))

    @property
    def coefficients(self):
        return self.preview(_SIZE_MAX)["coefficients"]

    def same_owner(self, NativeAlgebraicGradeTensor other not None):
        self._require()
        other._require()
        return bool(self._handle.get().same_owner(other._handle))

    def matches_source(self, NativeAlgebraicSnapshot snapshot not None):
        self._require()
        snapshot._require()
        return bool(self._handle.get().matches_source(snapshot._handle))

    def source_coordinate(self, coordinate):
        self._require()
        cdef size_t position = _integer(coordinate, _SIZE_MAX)
        return tuple(self._handle.get().source_coordinate(position))

    def source_index(self, coordinate):
        self._require()
        cdef vector[size_t] parts = _query_indices(coordinate, 3)
        cdef uint16_t k = _integer(parts[0], (1 << 16) - 1)
        cdef uint32_t cell = _integer(parts[1], (1 << 32) - 1)
        return int(self._handle.get().source_index(k, cell, parts[2]))

    def coordinate(self, coordinate):
        self._require()
        cdef size_t position = _integer(coordinate, _SIZE_MAX)
        cdef shared_ptr[CppAlgebraicValue] handle
        with nogil:
            handle = self._handle.get().coordinate(position)
        return _query_value(handle, self._snapshot.archive_owner)

    def stalk(self, cell):
        self._require()
        cdef uint32_t index = _integer(cell, (1 << 32) - 1)
        cdef shared_ptr[CppAlgebraicValue] handle
        with nogil:
            handle = self._handle.get().stalk(index)
        return _query_value(handle, self._snapshot.archive_owner)

    def boundary(self, *, availability, execution_limits=None, CancellationToken cancellation=None):
        self._require()
        cdef vector[size_t] bounds = _grade_limits(execution_limits, self._limits)
        cdef uint64_t ns, cutoff
        ns, cutoff = _query_availability(availability)
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None:
            token = cancellation._handle
        cdef AlgebraicGradeBuild built
        cdef AlgebraicGradeInfo info
        with nogil:
            built = self._handle.get().boundary(ns, cutoff, bounds, token)
        _errors(built.status)
        with nogil:
            info = built.tensor.get().info()
        cdef NativeAlgebraicGradeTensor result = NativeAlgebraicGradeTensor.__new__(NativeAlgebraicGradeTensor)
        result._info, result._limits = _grade_info(info), tuple(bounds)
        result._snapshot, result._input = self._snapshot, self
        result._handle = built.tensor
        return result

    def preview(self, max_values=256):
        self._require()
        cdef size_t maximum = _integer(max_values, _SIZE_MAX)
        cdef AlgebraicGradeValues values
        cdef vector[string] row
        cdef vector[size_t] coordinate
        cdef string q
        with nogil:
            values = self._handle.get().values(maximum)
        result = {key: value for key, value in self._info.items() if key != "requirements"}
        result.update(profile="original-algebraic-grade-field-v1", namespace=self._snapshot.namespace,
                      sequence=self._snapshot.sequence, requirements=self.requirements,
                      execution_limits=self.execution_limits,
                      coefficients=tuple(tuple(_fraction(<bytes>q) for q in row) for row in values.coefficients),
                      source_coordinates=tuple(tuple(coordinate) for coordinate in values.coordinates),
                      truncated=values.coefficients.size() < self.domain[0])
        return result


cdef class NativeAlgebraicMetricField:
    """Explicit certified metrics and occurrence adjoints on one owning publication."""
    cdef shared_ptr[CppAlgebraicMetricField] _handle
    cdef object _snapshot, _keys, _maximum
    cdef bint _loading

    def __init__(self, NativeAlgebraicSnapshot snapshot not None, metric_keys, *, availability,
                 max_retained_coordinates=1000000, CancellationToken cancellation=None):
        cdef vector[uint64_t] keys
        cdef uint64_t ns, cutoff
        cdef size_t maximum, count
        cdef shared_ptr[CppCancellation] token
        cdef AlgebraicMetricBuild built
        if self._loading or self._handle.get() != NULL:
            raise RuntimeError("algebraic metric field is immutable")
        self._loading = True
        try:
            snapshot._require()
            if cancellation is not None and cancellation.cancelled:
                raise ExecutionCancelled("exact execution cancelled at a phase boundary")
            maximum = _integer(max_retained_coordinates, _SIZE_MAX)
            if isinstance(metric_keys, (str, bytes, Mapping)):
                raise TypeError("metric keys require a sized sequence")
            count = len(metric_keys)
            if count != sum(snapshot.archive_owner.manifest["source_sizes"]):
                raise ValueError("metric field requires one metric record per retained cell")
            for key in metric_keys:
                if keys.size() >= count:
                    raise ValueError("metric key count changed")
                keys.push_back(_integer(key, (1 << 64) - 1))
            if keys.size() != count:
                raise ValueError("metric key count changed")
            ns, cutoff = _query_availability(availability)
            if cancellation is not None:
                token = cancellation._handle
            with nogil:
                built = CppAlgebraicMetricField.prepare(snapshot._handle, keys, maximum, ns, cutoff, token)
            _errors(built.status)
            self._snapshot, self._keys, self._maximum = snapshot, tuple(keys), int(maximum)
            self._handle = built.field
        finally:
            self._loading = False

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("algebraic metric field is not initialized")

    @property
    def snapshot_owner(self):
        self._require()
        return self._snapshot

    @property
    def metric_keys(self):
        self._require()
        return self._keys

    @property
    def max_retained_coordinates(self):
        self._require()
        return self._maximum

    def apply(self, NativeAlgebraicGradeTensor tensor not None, operation, *, availability,
              CancellationToken cancellation=None):
        self._require()
        tensor._require()
        cdef unsigned op
        cdef uint64_t ns, cutoff
        cdef shared_ptr[CppCancellation] token
        cdef AlgebraicMetricAction built
        cdef AlgebraicGradeInfo info
        cdef NativeAlgebraicMetricValues result
        if type(operation) is not str or operation not in ("hodge", "coboundary"):
            raise ValueError("whole-grade metric action must be hodge or coboundary")
        op = 0 if operation == "hodge" else 1
        ns, cutoff = _query_availability(availability)
        if cancellation is not None:
            token = cancellation._handle
        with nogil:
            built = self._handle.get().apply(tensor._handle, op, ns, cutoff, token)
        _errors(built.status)
        with nogil:
            info = built.values.get().info()
        result = NativeAlgebraicMetricValues.__new__(NativeAlgebraicMetricValues)
        result._handle = built.values
        result._field, result._input, result._operation = self, tensor, operation
        result._info = _grade_info(info)
        return result

    def inner(self, NativeAlgebraicGradeTensor tensor not None, NativeAlgebraicGradeTensor other not None,
              *, availability, CancellationToken cancellation=None):
        return self._scalar(tensor, other, 0, availability, cancellation)

    def adjacent_quadrance(self, NativeAlgebraicGradeTensor tensor not None, *, availability,
                           CancellationToken cancellation=None):
        return self._scalar(tensor, None, 1, availability, cancellation)

    cdef object _scalar(self, NativeAlgebraicGradeTensor tensor, NativeAlgebraicGradeTensor other,
                        unsigned op, object availability, CancellationToken cancellation):
        self._require()
        tensor._require()
        cdef shared_ptr[CppAlgebraicGradeTensor] right
        cdef shared_ptr[CppCancellation] token
        cdef uint64_t ns, cutoff
        cdef AlgebraicSum built
        if other is not None:
            other._require()
            right = other._handle
        ns, cutoff = _query_availability(availability)
        if cancellation is not None:
            token = cancellation._handle
        with nogil:
            built = self._handle.get().scalar(tensor._handle, right, op, ns, cutoff, token)
        _errors(built.status)
        return _query_value(built.value, self._snapshot.archive_owner)


cdef class NativeAlgebraicMetricValues:
    """Exact metric action vectors on the original result grade and cell carriers."""
    cdef shared_ptr[CppAlgebraicMetricValues] _handle
    cdef object _field, _input, _operation, _info

    def __init__(self):
        raise TypeError("metric values are constructed by an owning metric field")

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("algebraic metric values are not initialized")

    @property
    def field_owner(self):
        self._require()
        return self._field

    @property
    def snapshot_owner(self):
        self._require()
        return self._field.snapshot_owner

    @property
    def input_owner(self):
        self._require()
        return self._input

    @property
    def operation(self):
        self._require()
        return self._operation

    @property
    def grade(self):
        self._require()
        return self._info["grade"]

    @property
    def domain(self):
        self._require()
        return self._info["domain"]

    @property
    def cell_count(self):
        self._require()
        return self._info["cell_count"]

    @property
    def stalk_offsets(self):
        self._require()
        return self._info["stalk_offsets"]

    @property
    def coefficients(self):
        return self.preview(_SIZE_MAX)["coefficients"]

    def stalk(self, cell):
        self._require()
        cdef uint32_t index = _integer(cell, (1 << 32) - 1)
        cdef shared_ptr[CppAlgebraicValue] value
        with nogil:
            value = self._handle.get().stalk(index)
        return _query_value(value, self.snapshot_owner.archive_owner)

    def preview(self, max_values=256):
        self._require()
        cdef size_t maximum = _integer(max_values, _SIZE_MAX)
        cdef AlgebraicGradeValues values
        cdef vector[string] row
        cdef string q
        with nogil:
            values = self._handle.get().values(maximum)
        return {"profile": "original-algebraic-metric-values-v1", "grade": self.grade,
                "cell_count": self.cell_count, "domain": self.domain, "stalk_offsets": self.stalk_offsets,
                "coefficients": tuple(tuple(_fraction(<bytes>q) for q in row) for row in values.coefficients),
                "source_coordinates": tuple(tuple(c) for c in values.coordinates),
                "truncated": len(values.coefficients) != self.domain[0]}


cdef class NativeAlgebraicGradeTensorWindow:
    """An owning shape window over a whole grade with original stalk coordinate witnesses."""
    cdef shared_ptr[CppAlgebraicGradeWindow] _handle
    cdef object _tensor, _info, _limits
    cdef bint _loading

    def __init__(self, NativeAlgebraicGradeTensor tensor not None, shape, lower, upper, *, availability,
                 execution_limits=None, CancellationToken cancellation=None):
        cdef vector[size_t] bounds, extents, lo, hi
        cdef size_t rank
        cdef uint64_t ns, cutoff
        cdef shared_ptr[CppCancellation] token
        cdef AlgebraicGradeWindowBuild built
        cdef AlgebraicWindowInfo info
        if self._loading or self._handle.get() != NULL:
            raise RuntimeError("algebraic grade window is immutable")
        self._loading = True
        try:
            tensor._require()
            if cancellation is not None and cancellation.cancelled:
                raise ExecutionCancelled("exact execution cancelled at a phase boundary")
            bounds = _grade_limits(execution_limits, tensor._limits)
            if isinstance(shape, (str, bytes, Mapping)):
                raise TypeError("algebraic grade shape requires a sized sequence")
            rank = len(shape)
            _query_admit((1 + 4 * int(rank), 0, 0, 0, 0), tuple(bounds))
            extents, lo, hi = _query_indices(shape, rank), _query_indices(lower, rank), _query_indices(upper, rank)
            ns, cutoff = _query_availability(availability)
            if cancellation is not None:
                token = cancellation._handle
            with nogil:
                built = CppAlgebraicGradeWindow.prepare(tensor._handle, extents, lo, hi, ns, cutoff, bounds, token)
            _errors(built.status)
            with nogil:
                info = built.window.get().info()
            self._info = {"domain": tuple(info.domain), "source_domain": tuple(info.source_domain),
                          "lower": tuple(info.lower), "upper": tuple(info.upper),
                          "supported_count": int(info.supported_count), "requirements": tuple(info.requirements)}
            self._tensor, self._limits = tensor, tuple(bounds)
            self._handle = built.window
        finally:
            self._loading = False

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("algebraic grade window is not initialized")

    @property
    def tensor_owner(self):
        self._require()
        return self._tensor

    @property
    def domain(self):
        self._require()
        return self._info["domain"]

    @property
    def supported_count(self):
        self._require()
        return self._info["supported_count"]

    @property
    def requirements(self):
        self._require()
        return dict(zip(_QUERY_LIMIT_NAMES, self._info["requirements"]))

    @property
    def execution_limits(self):
        self._require()
        return dict(zip(_QUERY_LIMIT_NAMES, self._limits))

    def source_index(self, coordinate):
        self._require()
        cdef size_t position = _integer(coordinate, _SIZE_MAX)
        return int(self._handle.get().source_index(position))

    def source_coordinate(self, coordinate):
        self._require()
        cdef size_t position = _integer(coordinate, _SIZE_MAX)
        return tuple(self._handle.get().source_coordinate(position))

    def window_index(self, coordinate):
        self._require()
        cdef size_t position = _integer(coordinate, _SIZE_MAX)
        cdef vector[size_t] result = self._handle.get().window_index(position)
        return int(result[0]) if result.size() else None

    def coordinate(self, coordinate):
        self._require()
        cdef size_t position = _integer(coordinate, _SIZE_MAX)
        cdef shared_ptr[CppAlgebraicValue] handle
        with nogil:
            handle = self._handle.get().coordinate(position)
        return _query_value(handle, self._tensor.snapshot_owner.archive_owner)

    def sum(self, *, availability, execution_limits=None, CancellationToken cancellation=None):
        self._require()
        cdef vector[size_t] bounds = _grade_limits(execution_limits, self._limits)
        cdef uint64_t ns, cutoff
        ns, cutoff = _query_availability(availability)
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None:
            token = cancellation._handle
        cdef AlgebraicSum result
        with nogil:
            result = self._handle.get().sum(ns, cutoff, bounds, token)
        _errors(result.status)
        if result.value.get() == NULL:
            return None
        return _query_value(result.value, self._tensor.snapshot_owner.archive_owner)

    def preview(self, max_values=256):
        self._require()
        cdef size_t maximum = _integer(max_values, _SIZE_MAX)
        cdef AlgebraicGradeValues values
        cdef vector[string] row
        cdef vector[size_t] coordinate
        cdef string q
        with nogil:
            values = self._handle.get().values(maximum)
        result = {key: value for key, value in self._info.items() if key != "requirements"}
        result.update(profile="original-algebraic-grade-window-v1", grade=self._tensor.grade,
                      namespace=self._tensor.snapshot_owner.namespace, sequence=self._tensor.snapshot_owner.sequence,
                      requirements=self.requirements, execution_limits=self.execution_limits,
                      coefficients=tuple(tuple(_fraction(<bytes>q) for q in row) for row in values.coefficients),
                      source_coordinates=tuple(tuple(coordinate) for coordinate in values.coordinates),
                      source_indices=tuple(self.source_index(i) for i in range(values.coefficients.size())),
                      truncated=values.coefficients.size() < self.supported_count)
        return result


cdef class PreparedAlgebraicQuery:
    """An immutable native query over complete exact field equality predicates."""
    cdef shared_ptr[CppAlgebraicQuery] _handle
    cdef object _snapshot, _limits, _request, _requirements, _count, _tile_rows
    cdef bint _loading

    def __init__(self, NativeAlgebraicSnapshot snapshot not None, *, availability, kind=None,
                 NativeAlgebraicValue source_carrier=None, NativeAlgebraicValue target_carrier=None,
                 all_of=(), rows_per_tile=256, execution_limits=None, CancellationToken cancellation=None):
        if self._loading or self._handle.get() != NULL:
            raise RuntimeError("prepared algebraic query is immutable")
        snapshot._require()
        if cancellation is not None and cancellation.cancelled:
            raise ExecutionCancelled("exact execution cancelled at a phase boundary")
        cdef int code = -1
        if kind is not None:
            if kind not in _ALGEBRAIC_KINDS[:5]:
                raise ValueError("unknown algebraic query record kind")
            code = _ALGEBRAIC_KINDS.index(kind)
        cdef vector[size_t] bounds = _query_limits(execution_limits)
        cdef size_t tile_rows = _integer(rows_per_tile, _SIZE_MAX)
        if not tile_rows:
            raise ValueError("algebraic query tile size must be positive")
        cdef uint64_t ns, cutoff
        ns, cutoff = _query_availability(availability)
        cdef NativeAlgebraicArchive archive = snapshot.archive_owner
        records = archive.record_count
        degree = archive.field_degree
        if isinstance(all_of, (str, bytes, Mapping)):
            raise TypeError("algebraic query predicates require a sized sequence")
        clauses = len(all_of)
        required = [6 * records + records * clauses + clauses, clauses * degree, 0,
                    records * clauses * (degree + 2), 0]
        _query_admit(required, tuple(bounds))
        cdef shared_ptr[CppAlgebraicValue] source, target
        if source_carrier is not None:
            source_carrier._require()
            source = source_carrier._handle
        if target_carrier is not None:
            target_carrier._require()
            target = target_carrier._handle
        cdef vector[AlgebraicPredicate] predicates
        cdef AlgebraicPredicate predicate
        cdef NativeAlgebraicValue literal
        retained = []
        for item in all_of:
            if len(retained) >= clauses or not isinstance(item, Mapping) or set(item) != {"coordinate", "operation", "literal"}:
                raise ValueError("invalid algebraic equality predicate declaration")
            if item["operation"] not in ("eq", "ne"):
                raise ValueError("algebraic predicates require eq or ne")
            position = item["coordinate"]
            if isinstance(position, (str, bytes, Mapping)):
                raise TypeError("algebraic predicate coordinates require a sized sequence")
            required[0] += len(position)
            _query_admit(required, tuple(bounds))
            predicate.coordinate = _query_indices(position, len(position))
            predicate.operation = 0 if item["operation"] == "eq" else 1
            if not isinstance(item["literal"], NativeAlgebraicValue):
                raise TypeError("algebraic predicates require owning scalar literals")
            literal = item["literal"]
            literal._require()
            predicate.literal = literal._handle
            predicates.push_back(predicate)
            retained.append({"coordinate": tuple(predicate.coordinate), "operation": item["operation"],
                             "literal": literal})
        if len(retained) != clauses:
            raise ValueError("algebraic predicate count changed during preparation")
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None:
            token = cancellation._handle
        cdef AlgebraicQueryBuild built
        cdef AlgebraicQueryInfo info
        self._loading = True
        try:
            with nogil:
                built = CppAlgebraicQuery.prepare(snapshot._handle, code, source, target, predicates, ns, cutoff,
                                                 bounds, tile_rows, token)
            _errors(built.status)
            self._handle = built.query
            with nogil:
                info = self._handle.get().info()
            self._snapshot, self._limits = snapshot, tuple(bounds)
            self._count, self._requirements, self._tile_rows = int(info.candidate_count), tuple(info.requirements), int(tile_rows)
            self._request = {"kind": kind, "source_carrier": source_carrier, "target_carrier": target_carrier,
                             "all_of": tuple(retained)}
        finally:
            self._loading = False

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("prepared algebraic query is not initialized")

    @property
    def snapshot_owner(self):
        self._require()
        return self._snapshot

    @property
    def candidate_count(self):
        self._require()
        return self._count

    @property
    def requirements(self):
        self._require()
        return dict(zip(_QUERY_LIMIT_NAMES, self._requirements))

    @property
    def execution_limits(self):
        self._require()
        return dict(zip(_QUERY_LIMIT_NAMES, self._limits))

    def execute(self, *, availability, cpu_workers=1, worker_limit=64, tile_limit=1000000,
                execution_limits=None, CancellationToken cancellation=None, NativeAlgebraicQueryCache cache=None):
        self._require()
        cdef uint64_t ns, cutoff
        ns, cutoff = _query_availability(availability)
        cdef size_t workers = _integer(cpu_workers, _SIZE_MAX)
        cdef size_t limit = _integer(worker_limit, _SIZE_MAX)
        cdef size_t tiles = _integer(tile_limit, _SIZE_MAX)
        cdef vector[size_t] bounds = _query_limits(execution_limits, self._limits)
        cdef shared_ptr[CppCancellation] token
        if cancellation is not None:
            token = cancellation._handle
        cdef shared_ptr[CppAlgebraicQueryCache] retained_cache
        if cache is not None:
            cache._require()
            retained_cache = cache._handle
        cdef AlgebraicQueryExecution execution
        with nogil:
            execution = self._handle.get().execute(ns, cutoff, bounds, workers, limit, tiles, token, retained_cache)
        _errors(execution.status)
        cdef NativeAlgebraicQueryResult result = NativeAlgebraicQueryResult.__new__(NativeAlgebraicQueryResult)
        result._handle = execution.result
        result._query = self
        result._execution_limits = tuple(bounds)
        result._cpu_policy = (int(workers), int(limit), int(tiles))
        result._initialise()
        return result

    def preview(self, max_values=256):
        self._require()
        count = _integer(max_values, _SIZE_MAX)
        predicates = tuple({"coordinate": item["coordinate"], "operation": item["operation"],
                            "literal": item["literal"].preview(1)} for item in self._request["all_of"][:count])
        return {"profile": "prepared-algebraic-query-v1", "namespace": self._snapshot.namespace,
                "sequence": self._snapshot.sequence, "candidate_count": self._count, "kind": self._request["kind"],
                "source_carrier": None if self._request["source_carrier"] is None else self._request["source_carrier"].preview(0)["source_carrier"],
                "target_carrier": None if self._request["target_carrier"] is None else self._request["target_carrier"].preview(0)["target_carrier"],
                "all_of": predicates, "predicate_count": len(self._request["all_of"]),
                "truncated": len(predicates) < len(self._request["all_of"]),
                "rows_per_tile": self._tile_rows, "requirements": self.requirements, "execution_limits": self.execution_limits}


cdef class NativeAlgebraicQueryResult:
    """Ordered owning record pins selected by one prepared exact query."""
    cdef shared_ptr[CppAlgebraicQueryResult] _handle
    cdef object _query, _keys, _availability, _cpu, _hit, _execution_limits, _cpu_policy

    def __init__(self):
        raise TypeError("algebraic query results are obtained by execution")

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("algebraic query result is not initialized")

    cdef void _initialise(self) except *:
        cdef AlgebraicQueryResultInfo info
        with nogil:
            info = self._handle.get().info()
        self._keys, self._availability = tuple(info.keys), (int(info.name_space), int(info.max_sequence))
        self._cpu, self._hit = tuple(info.cpu), bool(info.cache_hit)

    @property
    def query_owner(self):
        self._require()
        return self._query

    @property
    def keys(self):
        self._require()
        return self._keys

    @property
    def availability(self):
        self._require()
        return self._availability

    @property
    def cache_hit(self):
        self._require()
        return self._hit

    @property
    def execution_limits(self):
        self._require()
        return dict(zip(_QUERY_LIMIT_NAMES, self._execution_limits))

    @property
    def cpu_policy(self):
        self._require()
        return dict(zip(("workers", "worker_limit", "tile_limit"), self._cpu_policy))

    @property
    def cpu_execution(self):
        self._require()
        return dict(zip(_QUERY_CPU_NAMES, self._cpu))

    def record(self, position):
        self._require()
        cdef size_t index = _integer(position, _SIZE_MAX)
        cdef NativeAlgebraicRecord result = NativeAlgebraicRecord.__new__(NativeAlgebraicRecord)
        with nogil:
            result._handle = self._handle.get().record(index)
        result._snapshot = self._query.snapshot_owner
        return result

    def preview(self, max_values=256):
        self._require()
        maximum = _integer(max_values, _SIZE_MAX)
        return {"profile": "derived-algebraic-query-v1", "namespace": self._query.snapshot_owner.namespace,
                "sequence": self._query.snapshot_owner.sequence, "availability": self._availability,
                "keys": self._keys[:maximum], "record_count": len(self._keys), "truncated": len(self._keys) > maximum,
                "cache_hit": self._hit, "cpu_execution": self.cpu_execution,
                "execution_limits": self.execution_limits, "cpu_policy": self.cpu_policy}


cdef class NativeAlgebraicQueryCache:
    """A synchronized result pin count cache keyed by actual prepared operation ownership."""
    cdef shared_ptr[CppAlgebraicQueryCache] _handle
    cdef object _limit

    def __init__(self, max_records):
        if self._handle.get() != NULL:
            raise RuntimeError("algebraic query cache limit is immutable")
        cdef size_t limit = _integer(max_records, _SIZE_MAX)
        self._handle = shared_ptr[CppAlgebraicQueryCache](new CppAlgebraicQueryCache(limit))
        self._limit = int(limit)

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("algebraic query cache is not initialized")

    def clear(self):
        self._require()
        with nogil:
            self._handle.get().clear()

    @property
    def statistics(self):
        self._require()
        cdef vector[size_t] counts
        with nogil:
            counts = self._handle.get().counts()
        return {"applications": int(counts[0]), "hits": int(counts[1]), "max_records": self._limit}


_PHASE_FOLD_KINDS = ("rational-phase", "algebraic-phase", "rational-fold", "rational-walk", "algebraic-walk")
_PHASE_FOLD_LIMIT_NAMES = tuple("wire." + key for key in _ALGEBRAIC_LIMIT_NAMES) + (
    "history.max_steps", "history.max_coordinate_slots", "history.max_integer_bits",
    "history.max_source_cells", "history.max_source_occurrences", "history.max_interaction_terms",
    "cover.max_vertices", "cover.max_path_steps", "cover.max_integer_bits", "panels.max_cells",
    "panels.max_occurrences", "panels.max_creases", "panels.max_integer_bits", "max_proposals")


cdef vector[size_t] _phase_fold_limits(object values) except *:
    cdef vector[size_t] result = CppPhaseFoldArchive.default_limits()
    if result.size() != len(_PHASE_FOLD_LIMIT_NAMES):
        raise RuntimeError("native phase/fold admission fields differ")
    if values is not None:
        if not isinstance(values, Mapping) or set(values) - set(_PHASE_FOLD_LIMIT_NAMES):
            raise ValueError("unknown phase/fold archive admission fields")
        for key, value in values.items():
            result[_PHASE_FOLD_LIMIT_NAMES.index(key)] = _integer(value, _SIZE_MAX)
    return result


cdef object _phase_fold_info(const PhaseFoldInfo& value):
    cdef string q
    cdef size_t i
    cdef PhaseFoldOperationInfo op
    cdef PhaseFoldProposalInfo proposal
    operations = []
    for i in range(value.operation_views.size()):
        op = value.operation_views[i]
        if value.kind >= 3:
            operations.append({"edge": int(op.edge), "forward": bool(op.forward)})
        else:
            item = {"identity": int(op.identity), "active": bool(op.active),
                    "phase": tuple(_fraction(<bytes>q) for q in op.phase)}
            if value.kind == 2:
                item.update(crease=int(op.crease), moving_panels=tuple(op.moving))
            operations.append(item)
    proposals = []
    for i in range(value.proposal_views.size()):
        proposal = value.proposal_views[i]
        proposals.append({"key": int(proposal.key), "position": int(proposal.position),
                          "active": bool(proposal.active), "phase": tuple(_fraction(<bytes>q) for q in proposal.phase)})
    return {"word_kind": _PHASE_FOLD_KINDS[value.kind], "operation_count": int(value.operations),
            "proposal_count": int(value.proposals), "field_degree": int(value.degree),
            "source_sizes": tuple(value.source_sizes), "cell": {"grade": int(value.grade), "index": int(value.index)} if value.has_cell else None,
            "minimal_polynomial": tuple(_fraction(<bytes>q) for q in value.polynomial),
            "involution": tuple(_fraction(<bytes>q) for q in value.involution),
            "endpoint_shape": tuple(value.shape), "endpoint_coordinate_count": int(value.endpoint_coordinates),
            "observation_coordinate_count": int(value.observation_coordinates),
            "endpoint": tuple(_fraction(<bytes>q) for q in value.endpoint),
            "operations": tuple(operations), "proposals": tuple(proposals),
            "branch_vertex": int(value.branch) if value.kind >= 3 else None,
            "positions": tuple(int((<bytes>q).decode("ascii"), 16) for q in value.positions)}


cdef class NativePhaseFoldArchive:
    """An owning exact operation word with separate prospective replacements."""
    cdef shared_ptr[CppPhaseFoldArchive] _handle
    cdef object _archive, _source, _manifest, _limits, _history_owner, _history_channel
    cdef bint _loading

    def __init__(self, bytes archive, *, limits=None):
        if self._loading or self._handle.get() != NULL:
            raise RuntimeError("phase/fold archive publication is immutable")
        cdef vector[size_t] bounds = _phase_fold_limits(limits)
        cdef size_t size = len(archive)
        cdef const char* data = archive
        if size < 32 or size > bounds[0]:
            raise ValueError("phase/fold archive exceeds its byte admission or is truncated")
        self._loading = True
        try:
            with nogil:
                self._handle = CppPhaseFoldArchive.load(data, size, bounds)
            self._initialise(bounds)
        finally:
            self._loading = False

    cdef void _require(self) except *:
        if self._handle.get() == NULL:
            raise RuntimeError("native phase/fold archive is not initialized")

    cdef void _initialise(self, const vector[size_t]& limits) except *:
        cdef PhaseFoldInfo info
        cdef string image, source
        with nogil:
            info = self._handle.get().info(0, False)
            image = self._handle.get().image()
            source = self._handle.get().source_image()
        self._archive, self._source, self._limits = <bytes>image, <bytes>source, tuple(limits)
        self._manifest = _phase_fold_info(info)
        self._manifest.update(codec="rfcxw-v1", archive_digest=sha256(self._archive).hexdigest(), source_digest=sha256(self._source).hexdigest())
        for key in ("endpoint", "operations", "proposals", "positions"):
            self._manifest.pop(key)

    @staticmethod
    def default_limits():
        return dict(zip(_PHASE_FOLD_LIMIT_NAMES, CppPhaseFoldArchive.default_limits()))

    @classmethod
    def from_history(cls, NativeSource source not None, channel, *, sequence, limits=None):
        """Read a declared RFCXW channel at a retained native publication."""
        source._require()
        if isinstance(channel, (str, bytes, Mapping)) or len(channel) != 3:
            raise ValueError("phase/fold history requires three channel names")
        names = tuple(_integer(v, (1 << 64) - 1) for v in channel)
        if not all(names):
            raise ValueError("phase/fold history channel names must be nonzero")
        cdef uint64_t cutoff = _integer(sequence, (1 << 64) - 1)
        cdef NativeSource selected = source if cutoff == source._identity["sequence"] else source.at(cutoff)
        cdef vector[uint64_t] native_channel = names
        cdef vector[size_t] bounds = _phase_fold_limits(limits)
        cdef NativePhaseFoldArchive result = NativePhaseFoldArchive.__new__(NativePhaseFoldArchive)
        with nogil:
            result._handle = CppPhaseFoldArchive.from_source(selected._handle, native_channel, bounds)
        result._initialise(bounds)
        result._history_owner, result._history_channel = selected, names
        return result

    @property
    def archive(self):
        self._require()
        return self._archive

    @property
    def source_rfcx(self):
        self._require()
        return self._source

    @property
    def manifest(self):
        self._require()
        from copy import deepcopy
        return deepcopy(self._manifest)

    @property
    def limits(self):
        self._require()
        return dict(zip(_PHASE_FOLD_LIMIT_NAMES, self._limits))

    @property
    def history_owner(self):
        self._require()
        return self._history_owner

    @property
    def history_channel(self):
        self._require()
        return self._history_channel

    def preview(self, max_values=256):
        self._require()
        cdef size_t maximum = _integer(max_values, _SIZE_MAX)
        cdef PhaseFoldInfo info
        with nogil:
            info = self._handle.get().info(maximum, True)
        out = _phase_fold_info(info)
        for key in ("minimal_polynomial", "involution"):
            out[key] = out[key][:maximum]
        out.update(archive_digest=self._manifest["archive_digest"], source_digest=self._manifest["source_digest"],
                   native_history=self._history_owner is not None)
        return out

    def observe(self, *, max_coordinates=65536):
        """Return complete exact endpoint coordinates and retained operations."""
        self._require()
        cdef size_t maximum = _integer(max_coordinates, _SIZE_MAX)
        if maximum < self._manifest["observation_coordinate_count"]:
            raise ValueError("phase/fold observation exceeds its complete coordinate admission")
        cdef PhaseFoldInfo info
        with nogil:
            info = self._handle.get().info(maximum, False)
        out = _phase_fold_info(info)
        out.pop("proposals")
        return out

    def observed_prefix(self, count):
        """Own a logical prefix without later operations or prospective replacements."""
        self._require()
        cdef size_t cutoff = _integer(count, _SIZE_MAX)
        cdef NativePhaseFoldArchive result = NativePhaseFoldArchive.__new__(NativePhaseFoldArchive)
        with nogil:
            result._handle = self._handle.get().prefix(cutoff)
        result._initialise(_phase_fold_limits(self.limits))
        return result

    def propose(self, key, position, coordinates, *, active=True):
        """Declare an exact unit phase replacement separately from the original word."""
        self._require()
        if type(active) is not bool:
            raise TypeError("phase/fold activity must be boolean")
        cdef uint64_t name = _integer(key, (1 << 64) - 1)
        cdef size_t index = _integer(position, _SIZE_MAX)
        degree = self._manifest["field_degree"] if self._manifest["word_kind"] == "algebraic-phase" else 2
        bits = min(self.limits["history.max_integer_bits"], self.limits["wire.field.max_integer_bits"])
        cdef vector[string] row = _algebraic_coordinates(coordinates, degree, bits)
        cdef NativePhaseFoldArchive result = NativePhaseFoldArchive.__new__(NativePhaseFoldArchive)
        cdef bint flag = active
        with nogil:
            result._handle = self._handle.get().propose(name, index, row, flag)
        result._initialise(_phase_fold_limits(self.limits))
        return result

    def apply(self, key):
        """Replay a named replacement on the actual carrier and discard prior proposals."""
        self._require()
        cdef uint64_t name = _integer(key, (1 << 64) - 1)
        cdef NativePhaseFoldArchive result = NativePhaseFoldArchive.__new__(NativePhaseFoldArchive)
        with nogil:
            result._handle = self._handle.get().apply(name)
        result._initialise(_phase_fold_limits(self.limits))
        return result

    def same_carrier(self, NativePhaseFoldArchive other not None):
        self._require()
        other._require()
        cdef bint matches
        with nogil:
            matches = self._handle.get().same_carrier(other._handle)
        return bool(matches)
