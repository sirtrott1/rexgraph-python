# distutils: language=c++
"""Exact affine identification of original incidence local transport entries."""
from fractions import Fraction
from collections.abc import Mapping
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from ._stalk_fields import _natural, _scalar, _axes, _values, _split

cdef object _view_matrix(size_t rows,size_t columns,const vector[BoundaryMapEntry]& entries):
    return dict(shape=(int(rows),int(columns)),entries=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in entries))

cdef extern from "stalk_transport_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass BoundaryMapEntry:
        size_t row, column
        string value
    cdef cppclass TransportSlot:
        size_t occurrence, row, column
    cdef cppclass TransportObservation:
        vector[string] input, target
        vector[size_t] observed
    cdef cppclass EncodedColumnField:
        size_t cell
        string tensor
    cdef cppclass ColumnObservation:
        vector[EncodedColumnField] input, target
    cdef cppclass ColumnTensorEntry:
        size_t axis
        string value
    string encode_column_tensor(size_t,const vector[ColumnTensorEntry]&,const vector[size_t]&) except +
    cdef cppclass TransportFitReceipt:
        vector[string] particular, target
        vector[BoundaryMapEntry] kernel
        vector[vector[size_t]] constraints
        vector[BoundaryMapEntry] design
        vector[size_t] pivots, free_columns
        size_t rank, steps, retained
    cdef cppclass TransportResponse:
        vector[vector[string]] boundary, occurrences
        vector[size_t] directions, zero_directions
        size_t steps
    cdef cppclass ContextFrame:
        size_t frame
        vector[size_t] axes
    cdef cppclass ContextEquation:
        string name, expected
        vector[size_t] entries
        vector[string] gains
    cdef cppclass ContextBlock:
        size_t origin, target
        vector[vector[size_t]] cells, occurrences
        vector[size_t] equations, evidence_samples
    cdef cppclass ContextAcquisition:
        vector[size_t] seed_rows
        vector[ContextBlock] blocks
        vector[ContextEquation] equations
        vector[string] rejection_names
        vector[size_t] rejection_counts
        size_t attempts, steps, retained
    cdef cppclass ContextRule:
        size_t origin
        vector[size_t] forced, excluded, open
        cppbool admitted
    cdef cppclass ContextRules:
        ContextAcquisition comparisons
        vector[ContextRule] rules
        vector[size_t] statuses, selected
        size_t steps, retained
    cdef cppclass ContextView:
        size_t frame, dimension
        vector[BoundaryMapEntry] accession, reconstruction
    cdef cppclass ContextViewWitness:
        size_t common_rows, origin_rows, target_rows, columns
        vector[BoundaryMapEntry] common_input, origin_input, target_input
        vector[vector[size_t]] response_shapes, equation_coordinates
    cdef cppclass ContextViewAcquisition:
        ContextAcquisition acquisition
        vector[ContextViewWitness] witnesses
        vector[ContextDiagram] diagrams
        vector[string] diagram_rejection_names
        vector[size_t] diagram_rejection_counts
        size_t diagram_attempts
    cdef cppclass ContextViewRules:
        ContextViewAcquisition views
        vector[ContextRule] rules
        vector[size_t] statuses, ranks, selected
        vector[vector[size_t]] covering_peers
        vector[string] rejection_names
        vector[size_t] rejection_counts
        size_t steps, retained
    cdef cppclass ContextDiagram:
        size_t patch
        vector[size_t] blocks, equations
        vector[vector[size_t]] rows, cells, occurrences, shared_cells
    cdef cppclass StalkTransportFit:
        StalkTransportFit(const string&, const vector[vector[vector[string]]]&, const vector[vector[vector[BoundaryMapEntry]]]&, const vector[TransportSlot]&, const vector[TransportObservation]&, const vector[size_t]&, size_t) except +
        StalkTransportFit(const string&, const vector[vector[vector[string]]]&, const vector[vector[vector[BoundaryMapEntry]]]&, const vector[TransportSlot]&, const vector[TransportObservation]&, const vector[size_t]&, size_t, size_t, const vector[BoundaryMapEntry]&) except +
        TransportFitReceipt receipt() except +
        shared_ptr[StalkTransportFit] append(const vector[TransportObservation]&) except +
        shared_ptr[StalkTransportFit] append_columns(const vector[ColumnObservation]&) except +
        size_t column_observation_count()
        size_t materialized_field_entries()
        vector[size_t] update_stats() except +
        ContextAcquisition contextual_hypotheses(const vector[size_t]&, const vector[vector[ContextFrame]]&) except +
        ContextRules contextual_rules(const vector[size_t]&, const vector[size_t]&, const vector[vector[ContextFrame]]&, size_t) except +
        ContextViewAcquisition contextual_views(const vector[size_t]&, const vector[vector[ContextView]]&) except +
        ContextViewRules contextual_view_rules(const vector[size_t]&, const vector[size_t]&, const vector[vector[ContextView]]&, size_t, cppbool, cppbool) except +
        ContextViewAcquisition contextual_diagrams(const vector[vector[size_t]]&, const vector[vector[ContextView]]&) except +
        size_t template_rank() except +
        vector[vector[size_t]] occurrence_cells(size_t) except +
        vector[string] occurrence_keys(size_t) except +
        vector[string] occurrence_coefficients(size_t) except +
        vector[size_t] occurrence_activity(size_t) except +
        vector[string] parameter_values(const vector[string]&) except +
        vector[string] entry_values(const vector[string]&) except +
        TransportResponse execute(const vector[string]&) except +

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_STALK_TRANSPORT_BINDING_DIGEST

cdef extern from "stalk_pattern_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass PatternFrameLabel:
        size_t frame
        string name
        vector[string] axes
    cdef cppclass PatternReceipt:
        vector[string] weights, occurrence_keys
        vector[BoundaryMapEntry] directions, input_image
        vector[vector[size_t]] coordinates
        vector[PatternFrameLabel] frames
        vector[size_t] carrier_frames, carrier_dimensions
        string source_sha256, evidence_sha256, source_archive
        size_t input_rows, input_columns, parameters, observations, peers, retained, provenance_bytes
    cdef cppclass PatternResponse:
        vector[string] origin, coordinates
        vector[BoundaryMapEntry] directions
        vector[size_t] offsets
        size_t parameters
    cdef cppclass StalkTransportPattern:
        StalkTransportPattern(const StalkTransportFit&, const vector[size_t]&, size_t, const vector[vector[ContextView]]&, size_t, const vector[size_t]&, size_t, const string&, const string&, const vector[PatternFrameLabel]&, cppbool) except +
        StalkTransportPattern(const string&, size_t, const vector[size_t]&, size_t) except +
        string encode() except +
        PatternReceipt receipt() except +
        PatternResponse response(const vector[string]&, bint) except +

cdef extern from "stalk_pattern_bank_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass PatternScopeReceipt:
        string source_archive
        vector[string] keys, types
        size_t observations, retained
    cdef cppclass StalkTransportPatternScope:
        StalkTransportPatternScope(const StalkTransportFit&, const vector[vector[ContextView]]&, size_t, const string&, const string&, const vector[size_t]&, size_t) except +
        StalkTransportPatternScope(const string&, size_t, const vector[size_t]&, size_t) except +
        StalkTransportPatternScope(const StalkTransportPatternScope&, const StalkTransportPatternBank&, size_t) except +
        string encode() except +
        PatternScopeReceipt receipt() except +
        vector[vector[size_t]] coverage(const StalkTransportPatternBank&) except +
    cdef cppclass PatternBankReceipt:
        PatternReceipt common
        vector[vector[BoundaryMapEntry]] images
        vector[size_t] rows, columns
        vector[string] pattern_keys
        cppbool declared_templates
    cdef cppclass PatternBankInput:
        size_t pattern
        vector[string] value
    cdef cppclass PatternSparseEntry:
        PatternSparseEntry() except +
        size_t index
        string value
    cdef cppclass PatternBankSparseInput:
        PatternBankSparseInput() except +
        size_t pattern, frame, size
        vector[PatternSparseEntry] entries
    cdef cppclass PatternBankComparison:
        string name
        size_t left, right, left_frame, right_frame, interface_frame, rows, left_columns, right_columns
        vector[BoundaryMapEntry] left_map, right_map
    cdef cppclass PatternBankResponse:
        PatternResponse field
        vector[vector[string]] coordinates
    cdef cppclass PatternBankCandidate:
        size_t pattern
        cppbool supported
        PatternBankResponse response
    cdef cppclass PatternProgramEdge:
        PatternProgramEdge() except +
        size_t node, occurrence, source_frame, target_frame, rows, columns
        vector[BoundaryMapEntry] map
    cdef cppclass PatternProgramNode:
        PatternProgramNode() except +
        string name
        size_t pattern
        cppbool has_input
        PatternBankSparseInput input
        vector[PatternProgramEdge] edges
    cdef cppclass PatternProgramResult:
        string status
        cppbool has_input, has_response
        size_t size, parameters, response_size
        vector[PatternSparseEntry] input
        vector[BoundaryMapEntry] input_directions, parameter_directions
        PatternBankResponse response
    cdef cppclass PatternBankGluing:
        cppbool consistent, forced
        vector[string] defect_origin, parameter_origin, contradiction, comparison_names
        vector[BoundaryMapEntry] defect_directions, parameter_directions
        size_t parameters, free_parameters
        vector[size_t] offsets, section_cells
        PatternBankResponse boundary_response, section
    cdef cppclass PatternTrainingReceipt:
        vector[ContextRule] rules
        vector[size_t] statuses, ranks, origins, targets
        size_t equations, attempts, steps, retained
    cdef cppclass StalkTransportPatternTraining:
        StalkTransportPatternTraining(const StalkTransportFit&, const vector[size_t]&, const vector[vector[ContextView]]&, size_t, const string&, const string&, const vector[PatternFrameLabel]&, cppbool, cppbool) except +
        PatternTrainingReceipt receipt() except +
    cdef cppclass StalkTransportPatternBank:
        StalkTransportPatternBank(const StalkTransportFit&, const vector[size_t]&, const vector[size_t]&, const vector[vector[ContextView]]&, size_t, const vector[size_t]&, size_t, const string&, const string&, const vector[PatternFrameLabel]&, cppbool, cppbool) except +
        StalkTransportPatternBank(const string&, size_t, const vector[size_t]&, size_t) except +
        StalkTransportPatternBank(const StalkTransportPatternBank&, const vector[size_t]&, const vector[size_t]&, size_t) except +
        StalkTransportPatternBank(const StalkTransportPatternTraining&, const vector[size_t]&, const vector[size_t]&, size_t) except +
        string encode() except +
        PatternBankReceipt receipt() except +
        PatternBankResponse response(const vector[PatternBankInput]&, bint) except +
        PatternBankResponse response_sparse(const vector[PatternBankSparseInput]&, bint) except +
        vector[PatternBankCandidate] response_sparse_candidates(const vector[PatternBankSparseInput]&, bint) except +
        vector[PatternProgramResult] response_sparse_program(const vector[PatternProgramNode]&) except +
        vector[PatternBankResponse] response_sparse_batch(const vector[vector[PatternBankSparseInput]]&, bint) except +
        PatternBankGluing glue(const vector[PatternBankInput]&, const vector[PatternBankComparison]&, bint) except +
        PatternBankGluing glue_sparse(const vector[PatternBankSparseInput]&, const vector[PatternBankComparison]&, bint) except +

cdef extern from "stalk_objective_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass ObjectiveViewTerm:
        size_t occurrence, axis
        string coefficient
    cdef cppclass ObjectiveView:
        string name
        vector[ObjectiveViewTerm] terms
    cdef cppclass ObjectiveField:
        vector[string] origin
        vector[BoundaryMapEntry] directions
    cdef cppclass ObjectiveReceipt:
        vector[string] particular, coordinate_particular, target
        vector[BoundaryMapEntry] kernel, coordinate_kernel, hessian
        vector[size_t] free_columns
        vector[ObjectiveField] fields
        string initial_energy, minimum_energy
        size_t rank, steps, retained
    cdef cppclass StalkFieldObjective:
        StalkFieldObjective(shared_ptr[StalkTransportFit], const vector[vector[string]]&, const vector[ObjectiveView]&, const vector[string]&, const vector[string]&, const string&, bint, bint) except +
        ObjectiveReceipt receipt() except +
        TransportResponse execute(const vector[string]&) except +

def stalk_objective_build_info():
    return dict(stalk_transport_build_info(),profile="owning-stalk-field-objective-v1")

def stalk_parameter_relations_build_info():
    return dict(stalk_transport_build_info(),profile="owning-stalk-parameter-relations-v1")

def stalk_transport_build_info(version=1):
    if type(version) is not int or version not in (1,2,3,4):raise ValueError("supported transport profile version required")
    return dict(engine="rexgraph-cpp",engine_version=RFC_ENGINE_VERSION.decode(),
        engine_headers=RFC_ENGINE_DIGEST.decode(),binding=RFC_STALK_TRANSPORT_BINDING_DIGEST.decode(),
        profile=("owning-affine-stalk-transport-v1","owning-tied-stalk-transport-v2","owning-column-stalk-transport-v3",
            "owning-tied-column-stalk-transport-v4")[version-1])

def encode_stalk_column_tensor(dimension,entries,*,limits=(4096,16000000,64000000,16384)):
    """Encode a complete exact stalk tensor with sparse nonzero coordinates."""
    bounds=tuple(limits)
    if len(bounds)!=4 or any(type(v) is not int or v<=0 for v in bounds):raise ValueError("four positive action admissions required")
    cdef size_t size=_natural(dimension)
    if size>bounds[0]:raise ValueError("stalk tensor dimension exceeds admission")
    cdef vector[ColumnTensorEntry] values
    cdef ColumnTensorEntry entry
    for axis,value in entries:
        if values.size()*2>=bounds[1]:raise ValueError("stalk tensor entries exceed admission")
        entry.axis=_natural(axis);entry.value=_scalar(value,bounds[3]);values.push_back(entry)
    cdef vector[size_t] admitted=bounds
    cdef string result
    with nogil:result=encode_column_tensor(size,values,admitted)
    return <bytes>result

def _whole(rows, axes, bits):
    if len(rows)!=len(axes):raise ValueError("complete original stalk field required")
    out=[]
    for row,names in zip(rows,axes,strict=True):
        if len(row)!=len(names):raise ValueError("complete named stalk vector required")
        out.extend(_scalar(v,bits) for v in row)
    return tuple(out)

cdef class PreparedStalkTransport:
    """Own original source support, exact observations and every unidentified direction."""
    cdef shared_ptr[StalkTransportFit] _handle
    cdef object _axes, _maps, _slots, _bounds, _digest, _lower, _occurrences, _fit, _parameter_keys
    cdef size_t _observation_count
    cdef void _require(self) except *:
        if self._handle.get()==NULL:raise RuntimeError("transport owner is not initialized")
    def __init__(self,archive,axes,fixed_maps,parameter_slots,observations=(),*,limits=(4096,16000000,64000000,16384),max_bytes=256*1024*1024,parameter_space=None,parameter_basis=None,column_fields=False):
        from hashlib import sha256
        if self._handle.get()!=NULL:raise RuntimeError("transport owner is immutable")
        if not isinstance(archive,bytes):raise TypeError("transport requires original native source bytes")
        bounds=tuple(_natural(v) for v in limits);maximum=_natural(max_bytes)
        if len(bounds)!=4 or not all(bounds) or not maximum:raise ValueError("positive action and archive admissions required")
        if len(archive)>maximum:raise ValueError("transport archive exceeds byte admission")
        declared=[];dimension=0
        for grade in axes:
            if len(declared)>=bounds[0]:raise ValueError("grade count exceeds admission")
            family=_axes(grade,bounds[0]);dimension+=sum(map(len,family))
            if dimension>bounds[0]:raise ValueError("whole stalk dimensions exceed admission")
            declared.append(family)
        declared=tuple(declared)
        if len(declared)<2 or any(declared[2:]):raise ValueError("grade-one source with empty higher grades required")
        cdef vector[vector[vector[string]]] names=tuple(tuple(tuple(n.encode() for n in row) for row in grade) for grade in declared)
        cdef vector[vector[vector[BoundaryMapEntry]]] maps
        cdef vector[vector[BoundaryMapEntry]] family_maps
        cdef vector[BoundaryMapEntry] block
        cdef BoundaryMapEntry entry
        retained=[];count=0
        for grade in fixed_maps:
            if maps.size()>=len(declared):raise ValueError("fixed maps exceed original grades")
            family_maps.clear();records=[]
            for entries in grade:
                count+=1
                if count>bounds[1]:raise ValueError("flag declarations exceed admission")
                block.clear();record=[]
                for r,c,v in entries:
                    count+=1
                    if count>bounds[1]:raise ValueError("fixed map entries exceed admission")
                    entry.row=_natural(r);entry.column=_natural(c);entry.value=_scalar(v,bounds[3])
                    block.push_back(entry);record.append((int(entry.row),int(entry.column),Fraction(v)))
                family_maps.push_back(block);records.append(tuple(record))
            maps.push_back(family_maps);retained.append(tuple(records))
        cdef vector[TransportSlot] slots
        cdef TransportSlot slot
        for o,r,c in parameter_slots:
            if slots.size()+1>=bounds[0]:raise ValueError("parameter dimensions exceed admission")
            slot.occurrence=_natural(o);slot.row=_natural(r);slot.column=_natural(c);slots.push_back(slot)
        cdef vector[BoundaryMapEntry] basis
        cdef size_t parameter_count=slots.size()
        basis_record=[]
        if parameter_space is None:
            if parameter_basis is not None:raise ValueError("named parameter space required with a sparse basis")
            parameter_keys=None
        else:
            if (not isinstance(parameter_space,Mapping) or set(parameter_space)!={"name","keys"} or
                    type(parameter_space["name"]) is not str or not parameter_space["name"].strip() or
                    len(parameter_space["name"].encode())>256 or isinstance(parameter_space["keys"],(str,bytes)) or
                    parameter_basis is None):raise ValueError("closed named parameter space and sparse basis required")
            parameter_keys=_axes((parameter_space["keys"],),bounds[0])[0]
            parameter_count=len(parameter_keys)
            if parameter_count+1>bounds[0]:raise ValueError("named parameter dimensions exceed admission")
            for r,c,v in parameter_basis:
                count+=1
                if count>bounds[1]:raise ValueError("parameter basis entries exceed admission")
                entry.row=_natural(r);entry.column=_natural(c);entry.value=_scalar(v,bounds[3])
                basis.push_back(entry);basis_record.append((int(entry.row),int(entry.column),_values((entry.value,))[0]))
        cdef vector[TransportObservation] samples
        cdef TransportObservation sample
        count+=slots.size()*3
        for observation in observations:
            if samples.size()>=bounds[0]:raise ValueError("observation count exceeds admission")
            if set(observation)!={"input","target","observed"}:raise ValueError("closed whole-field observation required")
            field=_whole(observation["input"],declared[1],bounds[3]);target=_whole(observation["target"],declared[0],bounds[3])
            mask=tuple(observation["observed"])
            if len(mask)!=len(declared[0]) or any(type(v) is not bool for v in mask):raise ValueError("boolean mask on original lower stalks required")
            count+=len(field)+len(target)+len(mask)
            if count>bounds[1]:raise ValueError("observations exceed entry admission")
            sample.input=field;sample.target=target;sample.observed=tuple(int(v) for v in mask);samples.push_back(sample)
        cdef string image=archive
        cdef vector[size_t] admitted=bounds
        cdef size_t cap=maximum
        if parameter_keys is None:
            with nogil:self._handle=shared_ptr[StalkTransportFit](new StalkTransportFit(image,names,maps,slots,samples,admitted,cap))
        else:
            with nogil:self._handle=shared_ptr[StalkTransportFit](new StalkTransportFit(image,names,maps,slots,samples,admitted,cap,parameter_count,basis))
        if type(column_fields) is not bool:raise TypeError("boolean column field profile required")
        if column_fields and samples.size():raise ValueError("column field profile requires compact acquisition")
        self._axes=declared;self._maps=tuple(retained)
        self._slots=tuple((int(s.occurrence),int(s.row),int(s.column)) for s in slots)
        self._bounds=bounds;self._digest=sha256(archive).hexdigest();self._parameter_keys=parameter_keys
        self._observation_count=samples.size()
        offsets=[0]
        for row in declared[0]:offsets.append(offsets[-1]+len(row))
        self._lower=tuple(offsets)
        cells=[()];keys=[()];coefficients=[()];activity=[()]
        for k in range(1,len(declared)):
            cells.append(tuple(tuple(row) for row in self._handle.get().occurrence_cells(k)))
            keys.append(tuple(int(v) for v in self._handle.get().occurrence_keys(k)))
            coefficients.append(_values(self._handle.get().occurrence_coefficients(k)))
            activity.append(tuple(bool(v) for v in self._handle.get().occurrence_activity(k)))
        offsets=[0]
        for _,lo in cells[1]:offsets.append(offsets[-1]+len(declared[0][lo]))
        self._occurrences=tuple(offsets)
        cdef TransportFitReceipt fit=self._handle.get().receipt()
        self._fit=dict(stalk_transport_build_info(),archive_digest=self._digest,stalk_axes=self._axes,
            fixed_maps=self._maps,parameter_slots=self._slots,
            occurrence_cells=tuple(cells),occurrence_keys=tuple(keys),
            occurrence_coefficients=tuple(coefficients),occurrence_activity=tuple(activity),
            particular=_values(fit.particular),kernel=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in fit.kernel),
            kernel_shape=(int(parameter_count),len(fit.free_columns)),
            rank=int(fit.rank),pivots=tuple(fit.pivots),free_columns=tuple(fit.free_columns),
            constraints=tuple(tuple(row) for row in fit.constraints),target=_values(fit.target),
            design=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in fit.design),
            preparation_steps=int(fit.steps),retained_entries=int(fit.retained),limits=bounds,
            parameterization="independent-original-flag-entries-v1")
        if parameter_keys is not None:
            self._fit.update(stalk_transport_build_info(2),parameter_space=(parameter_space["name"],parameter_keys),
                parameter_basis=tuple(basis_record),basis_shape=(len(self._slots),int(parameter_count)),
                basis_rank=int(self._handle.get().template_rank()),parameterization="named-original-flag-templates-v2")
        if column_fields:self._fit.update(stalk_transport_build_info(3 if parameter_keys is None else 4))
    @property
    def receipt(self):
        self._require();return dict(self._fit)
    @property
    def execution_stats(self):
        self._require()
        values=tuple(self._handle.get().update_stats())
        return dict(materialized_field_entries=int(self._handle.get().materialized_field_entries()),
            column_observations=int(self._handle.get().column_observation_count()),source_reused=bool(values[0]),reused_observations=values[1],
            assembled_observations=values[2],guard_steps=values[3])
    def append(self,observations):
        """Append original whole fields through the retained source and exact row family."""
        self._require()
        if self._fit["profile"] in {"owning-column-stalk-transport-v3","owning-tied-column-stalk-transport-v4"}:
            raise ValueError("compact column observations required")
        cdef vector[TransportObservation] samples
        cdef TransportObservation sample
        cdef TransportFitReceipt fit
        cdef PreparedStalkTransport child
        count=len(self._slots)*3
        for observation in observations:
            if self._observation_count+samples.size()>=self._bounds[0]:raise ValueError("observation count exceeds admission")
            if not isinstance(observation,Mapping) or set(observation)!={"input","target","observed"}:raise ValueError("closed whole-field observation required")
            field=_whole(observation["input"],self._axes[1],self._bounds[3])
            target=_whole(observation["target"],self._axes[0],self._bounds[3])
            mask=tuple(observation["observed"])
            if len(mask)!=len(self._axes[0]) or any(type(v) is not bool for v in mask):raise ValueError("boolean mask on original lower stalks required")
            count+=len(field)+len(target)+len(mask)
            if count>self._bounds[1]:raise ValueError("observations exceed entry admission")
            sample.input=field;sample.target=target;sample.observed=tuple(int(v) for v in mask);samples.push_back(sample)
        child=PreparedStalkTransport.__new__(PreparedStalkTransport)
        with nogil:child._handle=self._handle.get().append(samples)
        child._axes=self._axes;child._maps=self._maps;child._slots=self._slots;child._bounds=self._bounds
        child._digest=self._digest;child._lower=self._lower;child._occurrences=self._occurrences;child._parameter_keys=self._parameter_keys
        child._observation_count=self._observation_count+samples.size()
        fit=child._handle.get().receipt();child._fit=dict(self._fit)
        child._fit.update(particular=_values(fit.particular),
            kernel=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in fit.kernel),
            kernel_shape=(len(fit.particular),len(fit.free_columns)),rank=int(fit.rank),
            pivots=tuple(fit.pivots),free_columns=tuple(fit.free_columns),
            constraints=tuple(tuple(row) for row in fit.constraints),target=_values(fit.target),
            design=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in fit.design),
            preparation_steps=int(fit.steps),retained_entries=int(fit.retained))
        return child
    def append_columns(self,observations):
        """Acquire exact encoded tensors on original relation columns and observed stalks."""
        self._require()
        if self._fit['profile'] not in {'owning-column-stalk-transport-v3','owning-tied-column-stalk-transport-v4'}:
            raise ValueError("column transport profile required")
        cdef vector[ColumnObservation] samples
        cdef ColumnObservation sample
        cdef EncodedColumnField field
        cdef TransportFitReceipt fit
        cdef PreparedStalkTransport child
        count=0
        for observation in observations:
            if self._observation_count+samples.size()>=self._bounds[0]:raise ValueError("observation count exceeds admission")
            if not isinstance(observation,Mapping) or set(observation)!={'input','target'}:raise ValueError("closed original column observation required")
            sample.input.clear();sample.target.clear()
            for target in (False,True):
                for cell,tensor in observation['target' if target else 'input']:
                    if type(tensor) is not bytes:raise TypeError("immutable encoded stalk tensor bytes required")
                    count+=len(tensor)+2
                    if count>self._bounds[1]:raise ValueError("encoded column observations exceed admission")
                    field.cell=_natural(cell);field.tensor=tensor
                    if target:sample.target.push_back(field)
                    else:sample.input.push_back(field)
            samples.push_back(sample)
        child=PreparedStalkTransport.__new__(PreparedStalkTransport)
        with nogil:child._handle=self._handle.get().append_columns(samples)
        child._axes=self._axes;child._maps=self._maps;child._slots=self._slots;child._bounds=self._bounds
        child._digest=self._digest;child._lower=self._lower;child._occurrences=self._occurrences;child._parameter_keys=self._parameter_keys
        child._observation_count=self._observation_count+samples.size()
        fit=child._handle.get().receipt();child._fit=dict(self._fit)
        child._fit.update(particular=_values(fit.particular),
            kernel=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in fit.kernel),
            kernel_shape=(len(fit.particular),len(fit.free_columns)),rank=int(fit.rank),
            pivots=tuple(fit.pivots),free_columns=tuple(fit.free_columns),
            constraints=tuple(tuple(row) for row in fit.constraints),target=_values(fit.target),
            design=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in fit.design),
            preparation_steps=int(fit.steps),retained_entries=int(fit.retained))
        return child
    def contextual_hypotheses(self,targets,frames):
        """Acquire complete typed star hypotheses through explicit common frames."""
        return self._contextual_hypotheses(targets,frames,None,0)
    def contextual_rules(self,training,targets,frames,*,minimum_peers=2):
        """Learn prospective sharing rules from independent complete field evidence."""
        return self._contextual_hypotheses(targets,frames,training,minimum_peers)
    def _contextual_hypotheses(self,targets,frames,training,minimum_peers):
        self._require()
        cdef vector[size_t] rows
        cdef vector[size_t] train
        cdef size_t peers
        cdef vector[vector[ContextFrame]] atlas
        cdef vector[ContextFrame] grade
        cdef ContextFrame chart
        cdef ContextAcquisition result
        cdef ContextRules learned
        if isinstance(targets,(str,bytes)) or isinstance(frames,(str,bytes)) or len(frames)!=2:
            raise ValueError("target rows and a complete two-grade frame atlas required")
        count=0;names=[];declared=[];target_rows=[]
        for target in targets:
            row=_natural(target);count+=1
            if row>=len(self._axes[1]) or row in target_rows or count>self._bounds[0]:
                raise ValueError("distinct admitted original target rows required")
            target_rows.append(row);rows.push_back(row)
        for k,family in enumerate(frames):
            if isinstance(family,(str,bytes)) or len(family)!=len(self._axes[k]):
                raise ValueError("frame atlas must cover every original stalk")
            grade.clear();records=[]
            for frame in family:
                chart.axes.clear();count+=2
                if frame is None:
                    chart.frame=0;records.append(None)
                else:
                    name,axes=frame
                    if type(name) is not str or not name.strip() or len(name.encode())>256 or isinstance(axes,(str,bytes)):
                        raise ValueError("bounded common frame name and explicit permutation required")
                    count+=len(name.encode());coordinates=[]
                    for axis in axes:
                        count+=1
                        if count>self._bounds[1]:raise ValueError("context frame declaration exceeds entry admission")
                        index=_natural(axis);chart.axes.push_back(index);coordinates.append(index)
                    if name not in names:names.append(name)
                    chart.frame=names.index(name)+1;records.append((name,tuple(coordinates)))
                if count>self._bounds[1]:raise ValueError("context frame declaration exceeds entry admission")
                grade.push_back(chart)
            declared.append(tuple(records));atlas.push_back(grade)
        if training is None:
            with nogil:result=self._handle.get().contextual_hypotheses(rows,atlas)
        else:
            if self._parameter_keys is not None:raise ValueError("context rules require independent original entries")
            if isinstance(training,(str,bytes)):raise TypeError("training rows require an original row sequence")
            peers=_natural(minimum_peers)
            if peers==0:raise ValueError("positive independent peer admission required")
            for row in training:
                index=_natural(row);count+=1
                if index>=len(self._axes[1]) or count>self._bounds[1]:raise ValueError("admitted original training rows required")
                train.push_back(index)
            with nogil:learned=self._handle.get().contextual_rules(train,rows,atlas,peers)
            result=learned.comparisons
        equations=tuple((e.name.decode(),tuple(zip(e.entries,_values(e.gains),strict=True)),Fraction(0)) for e in result.equations)
        output=dict(stalk_transport_build_info(),profile="acquired-context-transport-hypotheses-v1",
            archive_digest=self._digest,frame_atlas=tuple(declared),target_rows=tuple(target_rows),
            seed_rows=tuple(result.seed_rows),relations=equations,
            blocks=tuple(dict(origin=int(b.origin),target=int(b.target),cell_map=tuple(tuple(row) for row in b.cells),
                occurrence_map=tuple(tuple(row) for row in b.occurrences),equations=tuple(b.equations),
                evidence_samples=tuple(b.evidence_samples)) for b in result.blocks),
            rejections=dict(zip((s.decode() for s in result.rejection_names),result.rejection_counts,strict=True)),
            attempts=int(result.attempts),execution_steps=int(result.steps),retained_entries=int(result.retained),
            frame_scope="declared-common-frame-permutations",truth_status="prospective")
        if training is not None:
            output.update(profile="learned-context-transport-rules-v1",training_rows=tuple(train),minimum_peers=int(peers),
                block_statuses=tuple(("forced","excluded","open")[i] for i in learned.statuses),
                rules=tuple(dict(origin=int(r.origin),forced_blocks=tuple(r.forced),excluded_blocks=tuple(r.excluded),
                    open_blocks=tuple(r.open),admitted=bool(r.admitted)) for r in learned.rules),
                selected_block_indices=tuple(learned.selected),
                selected_blocks=tuple((int(result.blocks[i].origin),int(result.blocks[i].target)) for i in learned.selected),
                execution_steps=int(learned.steps),retained_entries=int(learned.retained),
                transfer_scope="conditional-on-observed-context-invariance")
        return output
    def contextual_views(self,targets,frames):
        """Compare whole responses on the exact intersection of declared input images."""
        return self._contextual_views(targets,frames,None)
    def contextual_view_rules(self,training,targets,frames,*,minimum_peers=2,observed_inputs=False,declared_templates=False):
        """Learn covered shared input rules and retain their native evidence owner."""
        if type(observed_inputs) is not bool:raise TypeError("observed input scope must be Boolean")
        if type(declared_templates) is not bool:raise TypeError("declared template scope must be Boolean")
        return self._contextual_views(targets,frames,None,training,minimum_peers,observed_inputs,declared_templates)
    cdef void _pattern_atlas(self,object frames,vector[vector[ContextView]]& atlas,vector[PatternFrameLabel]& labels) except *:
        from ._context_views import declaration
        declared=declaration(self._axes,self._bounds,(),frames)
        cdef vector[ContextView] grade
        cdef ContextView chart
        cdef BoundaryMapEntry entry
        cdef PatternFrameLabel label
        names={}
        for family in declared['frames']:
            grade.clear()
            for frame in family:
                chart.accession.clear();chart.reconstruction.clear();chart.frame=0;chart.dimension=0
                if frame is not None:
                    name,keys,accession,reconstruction=frame
                    if name not in names:
                        names[name]=len(names)+1
                        label.frame=names[name];label.name=name.encode();label.axes=tuple(key.encode() for key in keys);labels.push_back(label)
                    chart.frame=names[name];chart.dimension=len(keys)
                    for i,j,value in accession:
                        entry.row=i;entry.column=j;entry.value=_scalar(value,self._bounds[3]);chart.accession.push_back(entry)
                    for i,j,value in reconstruction:
                        entry.row=i;entry.column=j;entry.value=_scalar(value,self._bounds[3]);chart.reconstruction.push_back(entry)
                grade.push_back(chart)
            atlas.push_back(grade)
    def learned_pattern(self,training,origin,frames,*,model_namespace,coordinate_ids,evidence_digest,minimum_peers=2,observed_inputs=False):
        """Export the complete observable parameter family on independently covered inputs."""
        self._require()
        if self._parameter_keys is not None:raise ValueError("pattern learning requires independent original entries")
        if type(observed_inputs) is not bool:raise TypeError("observed input selection must be boolean")
        cdef cppbool observed=observed_inputs
        cdef vector[size_t] train
        cdef vector[size_t] coordinates=tuple(_natural(x) for x in coordinate_ids)
        cdef size_t row=_natural(origin), peers=_natural(minimum_peers), ns=_natural(model_namespace)
        if not ns or not peers or len(coordinate_ids)!=3 or not all(coordinate_ids):raise ValueError("nonzero model namespace, coordinates and peer admission required")
        if isinstance(training,(str,bytes)):raise TypeError("original training row sequence required")
        seen=set()
        for i in training:
            i=_natural(i)
            if i>=len(self._axes[1]) or i in seen or train.size()>=self._bounds[0]:raise ValueError("distinct admitted original training rows required")
            seen.add(i);train.push_back(i)
        if row not in seen:raise ValueError("pattern origin must be a training row")
        cdef vector[vector[ContextView]] atlas
        cdef vector[PatternFrameLabel] labels
        self._pattern_atlas(frames,atlas,labels)
        if type(evidence_digest) is not str:raise TypeError("external evidence digest must be a hexadecimal string")
        cdef string source=self._digest.encode(), evidence=evidence_digest.encode()
        cdef PreparedStalkTransportPattern result=PreparedStalkTransportPattern.__new__(PreparedStalkTransportPattern)
        with nogil:result._handle=shared_ptr[StalkTransportPattern](new StalkTransportPattern(self._handle.get()[0],train,row,atlas,peers,coordinates,ns,source,evidence,labels,observed))
        result._bounds=self._bounds
        return result
    def observed_pattern_scope(self,frames,*,source_namespace,evidence_digest,max_bytes=256*1024*1024):
        """Retain every observed original star and its exact common input image."""
        self._require()
        cdef size_t ns=_natural(source_namespace), maximum=_natural(max_bytes)
        if not ns or not maximum:raise ValueError("positive scope namespace and byte admission required")
        if type(evidence_digest) is not str:raise TypeError("external evidence digest must be hexadecimal text")
        cdef vector[vector[ContextView]] atlas
        cdef vector[PatternFrameLabel] labels
        self._pattern_atlas(frames,atlas,labels)
        cdef string source=self._digest.encode(), evidence=evidence_digest.encode()
        cdef vector[size_t] bounds=self._bounds
        cdef PreparedStalkTransportPatternScope result=PreparedStalkTransportPatternScope.__new__(PreparedStalkTransportPatternScope)
        with nogil:result._handle=shared_ptr[StalkTransportPatternScope](new StalkTransportPatternScope(self._handle.get()[0],atlas,ns,source,evidence,bounds,maximum))
        return result
    def prepare_pattern_bank(self,training,frames,*,evidence_digest,minimum_peers=2,observed_inputs=False,declared_templates=False):
        """Own the full original comparison proof without scalar equation export."""
        self._require()
        if type(declared_templates) is not bool or type(observed_inputs) is not bool:
            raise TypeError("template and observed input scopes must be Boolean")
        if self._parameter_keys is not None and not declared_templates:raise ValueError("bank learning requires independent original entries")
        if declared_templates and self._parameter_keys is None:raise ValueError("declared bank templates require a named parameter basis")
        cdef vector[size_t] train
        cdef size_t peers=_natural(minimum_peers)
        if not peers:raise ValueError("positive independent peer admission required")
        if isinstance(training,(str,bytes)):raise TypeError("original training row sequence required")
        seen=set()
        for row in training:
            row=_natural(row)
            if row>=len(self._axes[1]) or row in seen or train.size()>=self._bounds[0]:raise ValueError("distinct admitted training rows required")
            seen.add(row);train.push_back(row)
        cdef vector[vector[ContextView]] atlas
        cdef vector[PatternFrameLabel] labels
        self._pattern_atlas(frames,atlas,labels)
        if type(evidence_digest) is not str:raise TypeError("external evidence digest must be hexadecimal text")
        cdef string source=self._digest.encode(), evidence=evidence_digest.encode()
        cdef cppbool observed=observed_inputs,templates=declared_templates
        cdef PreparedStalkTransportPatternTraining result=PreparedStalkTransportPatternTraining.__new__(PreparedStalkTransportPatternTraining)
        with nogil:result._handle=shared_ptr[StalkTransportPatternTraining](new StalkTransportPatternTraining(self._handle.get()[0],train,atlas,peers,source,evidence,labels,observed,templates))
        result._transport=self;result._training=tuple(train)
        result._identity=(self._digest,evidence_digest,minimum_peers,observed_inputs,declared_templates)
        return result
    def learned_pattern_bank(self,training,origins,frames,*,model_namespace,coordinate_ids,evidence_digest,minimum_peers=2,observed_inputs=False,declared_templates=False):
        """Export a retained star union with its complete joint observable weights."""
        self._require()
        if type(declared_templates) is not bool:raise TypeError("declared template scope must be Boolean")
        if self._parameter_keys is not None and not declared_templates:raise ValueError("bank learning requires independent original entries")
        if declared_templates and self._parameter_keys is None:raise ValueError("declared bank templates require a named parameter basis")
        if type(observed_inputs) is not bool:raise TypeError("observed input selection must be boolean")
        cdef cppbool observed=observed_inputs
        if not isinstance(coordinate_ids,Mapping) or set(coordinate_ids)!={'weights','parameters','domains'}:
            raise ValueError("bank weights, parameters and domain identities required")
        cdef vector[size_t] train, selected
        cdef vector[size_t] coordinates=(_natural(coordinate_ids['weights']),_natural(coordinate_ids['parameters']),*tuple(_natural(i) for i in coordinate_ids['domains']))
        cdef size_t peers=_natural(minimum_peers),ns=_natural(model_namespace)
        if not ns or not peers:raise ValueError("nonzero bank namespace and peer admission required")
        seen=set()
        for row in training:
            row=_natural(row)
            if row>=len(self._axes[1]) or row in seen or train.size()>=self._bounds[0]:raise ValueError("distinct admitted training rows required")
            seen.add(row);train.push_back(row)
        previous=-1
        for row in origins:
            row=_natural(row)
            if row not in seen or row<=previous or selected.size()>=self._bounds[0]:raise ValueError("ordered distinct training origins required")
            previous=row;selected.push_back(row)
        if not selected.size() or coordinates.size()!=selected.size()+2:raise ValueError("one domain identity per bank origin required")
        cdef vector[vector[ContextView]] atlas
        cdef vector[PatternFrameLabel] labels
        self._pattern_atlas(frames,atlas,labels)
        if type(evidence_digest) is not str:raise TypeError("external evidence digest must be hexadecimal text")
        cdef string source=self._digest.encode(), evidence=evidence_digest.encode()
        cdef PreparedStalkTransportPatternBank result=PreparedStalkTransportPatternBank.__new__(PreparedStalkTransportPatternBank)
        cdef cppbool templates=declared_templates
        with nogil:result._handle=shared_ptr[StalkTransportPatternBank](new StalkTransportPatternBank(self._handle.get()[0],train,selected,atlas,peers,coordinates,ns,source,evidence,labels,observed,templates))
        result._bounds=self._bounds
        return result
    def contextual_diagrams(self,target_patches,frames):
        """Glue observed whole star comparisons on shared original boundaries."""
        from ._context_views import diagram_declaration
        self._require()
        declared=diagram_declaration(self._axes,self._bounds,target_patches,frames)
        targets=tuple(dict.fromkeys(row for patch in declared['target_patches'] for row in patch))
        return self._contextual_views(targets,declared['frames'],declared['target_patches'])
    def _contextual_views(self,targets,frames,patches,training=None,minimum_peers=0,observed_inputs=False,declared_templates=False):
        self._require()
        from ._context_views import declaration
        declared=declaration(self._axes,self._bounds,targets,frames)
        cdef vector[size_t] rows
        cdef vector[size_t] train
        cdef size_t peers
        cdef cppbool observed=observed_inputs
        cdef cppbool templates=declared_templates
        cdef vector[vector[size_t]] groups
        cdef vector[size_t] patch
        cdef vector[vector[ContextView]] atlas
        cdef vector[ContextView] grade
        cdef ContextView chart
        cdef BoundaryMapEntry entry
        cdef ContextViewAcquisition result
        cdef ContextViewRules learned
        cdef PreparedStalkParameterRelations plan
        cdef ContextAcquisition acquired
        cdef ContextBlock b
        cdef ContextViewWitness w
        cdef ContextDiagram d
        for row in declared['targets']:rows.push_back(row)
        names={}
        for family in declared['frames']:
            grade.clear()
            for frame in family:
                chart.accession.clear();chart.reconstruction.clear()
                chart.frame=0;chart.dimension=0
                if frame is not None:
                    name,keys,accession,reconstruction=frame
                    if name not in names:names[name]=len(names)+1
                    chart.frame=names[name];chart.dimension=len(keys)
                    for i,j,value in accession:
                        entry.row=i;entry.column=j;entry.value=_scalar(value,self._bounds[3]);chart.accession.push_back(entry)
                    for i,j,value in reconstruction:
                        entry.row=i;entry.column=j;entry.value=_scalar(value,self._bounds[3]);chart.reconstruction.push_back(entry)
                grade.push_back(chart)
            atlas.push_back(grade)
        if training is not None:
            if self._parameter_keys is not None and not declared_templates:raise ValueError("context view rules require independent original entries")
            if declared_templates and self._parameter_keys is None:raise ValueError("declared context templates require a named parameter basis")
            if isinstance(training,(str,bytes)):raise TypeError("original training row sequence required")
            peers=_natural(minimum_peers)
            if peers==0:raise ValueError("positive independent peer admission required")
            seen=set()
            for row in training:
                index=_natural(row)
                if index>=len(self._axes[1]) or index in seen or index in declared['targets'] or train.size()>=self._bounds[0]:
                    raise ValueError("distinct admitted original training rows outside the target family required")
                seen.add(index);train.push_back(index)
            with nogil:learned=self._handle.get().contextual_view_rules(train,rows,atlas,peers,observed,templates)
            result=learned.views
        elif patches is None:
            with nogil:result=self._handle.get().contextual_views(rows,atlas)
        else:
            for group in patches:
                patch.clear()
                for row in group:patch.push_back(row)
                groups.push_back(patch)
            with nogil:result=self._handle.get().contextual_diagrams(groups,atlas)
        acquired=result.acquisition
        blocks=[]
        for i in range(acquired.blocks.size()):
            b=acquired.blocks[i];w=result.witnesses[i]
            blocks.append(dict(origin=int(b.origin),target=int(b.target),cell_map=tuple(tuple(row) for row in b.cells),
                occurrence_map=tuple(tuple(row) for row in b.occurrences),equations=tuple(b.equations),
                evidence_samples=tuple(b.evidence_samples),common_input=_view_matrix(w.common_rows,w.columns,w.common_input),
                origin_input=_view_matrix(w.origin_rows,w.columns,w.origin_input),
                target_input=_view_matrix(w.target_rows,w.columns,w.target_input),
                response_shapes=tuple(tuple(row) for row in w.response_shapes),
                equation_coordinates=tuple(tuple(row) for row in w.equation_coordinates)))
        output=dict(stalk_transport_build_info(),profile="acquired-context-transport-views-v1",
            archive_digest=self._digest,frame_atlas=declared['frames'],target_rows=declared['targets'],
            seed_rows=tuple(acquired.seed_rows),relations=tuple((e.name.decode(),tuple(zip(e.entries,_values(e.gains),strict=True)),
                _values((e.expected,))[0]) for e in acquired.equations),blocks=tuple(blocks),
            rejections=dict(zip((s.decode() for s in acquired.rejection_names),acquired.rejection_counts,strict=True)),
            attempts=int(acquired.attempts),execution_steps=int(acquired.steps),retained_entries=int(acquired.retained),
            frame_scope="declared-faithful-views-common-input-intersection",truth_status="prospective")
        if patches is not None:
            output.update(profile="acquired-context-transport-diagrams-v1",target_patches=patches,
                diagrams=tuple(dict(patch=int(d.patch),blocks=tuple(d.blocks),equations=tuple(d.equations),
                    row_map=tuple(tuple(row) for row in d.rows),cell_map=tuple(tuple(row) for row in d.cells),
                    occurrence_map=tuple(tuple(row) for row in d.occurrences),shared_cell_map=tuple(tuple(row) for row in d.shared_cells))
                    for d in result.diagrams),diagram_attempts=int(result.diagram_attempts),
                diagram_rejections=dict(zip((s.decode() for s in result.diagram_rejection_names),result.diagram_rejection_counts,strict=True)),
                gluing_scope="injective-complete-retained-boundary-maps")
        if training is not None:
            output.update(profile="learned-context-transport-view-rules-v1",training_rows=tuple(train),minimum_peers=int(peers),
                block_statuses=tuple(("forced","excluded","open")[i] for i in learned.statuses),
                block_ranks=tuple(None if learned.statuses[i]==1 else int(learned.ranks[i]) for i in range(learned.ranks.size())),
                covering_peer_blocks=tuple(tuple(row) for row in learned.covering_peers),
                rules=tuple(dict(origin=int(r.origin),forced_blocks=tuple(r.forced),excluded_blocks=tuple(r.excluded),
                    open_blocks=tuple(r.open),admitted=bool(r.admitted)) for r in learned.rules),
                selected_block_indices=tuple(learned.selected),
                selected_blocks=tuple((int(acquired.blocks[i].origin),int(acquired.blocks[i].target)) for i in learned.selected),
                rule_rejections=dict(zip((s.decode() for s in learned.rejection_names),learned.rejection_counts,strict=True)),
                execution_steps=int(learned.steps),retained_entries=int(learned.retained),
                transfer_scope="conditional-on-covered-context-invariance")
            if observed_inputs:
                output.update(profile="learned-observed-context-transport-view-rules-v1",
                    input_scope="exact-observed-field-spans",transfer_scope="conditional-on-covered-observed-field-invariance")
            plan=None
            if output['relations']:
                plan=PreparedStalkParameterRelations.__new__(PreparedStalkParameterRelations)
                with nogil:plan._handle=shared_ptr[StalkParameterRelations](new StalkParameterRelations(self._handle,learned))
                plan._record_fit(self,output['relations'])
                plan._fit['native_evidence_reused']=True
            return output,plan
        return output
    def apply(self,field):
        self._require()
        cdef vector[string] values=_whole(field,self._axes[1],self._bounds[3])
        cdef TransportResponse result
        with nogil:result=self._handle.get().execute(values)
        return dict(boundary_family=tuple(_split(_values(row),self._lower) for row in result.boundary),
            occurrence_family=tuple(_split(_values(row),self._occurrences) for row in result.occurrences),
            execution_steps=int(result.steps),
            components=("origin",*("free:"+(str(i) if self._parameter_keys is None else self._parameter_keys[i]) for i in result.directions)),
            zero_components=tuple("free:"+(str(i) if self._parameter_keys is None else self._parameter_keys[i]) for i in result.zero_directions),
            profile="whole-stalk-affine-response-v1" if self._parameter_keys is None else "whole-stalk-tied-response-v2")
    def materialize(self,free_parameters=None):
        """Select every free parameter explicitly before constructing complete flag maps."""
        self._require();count=len(self._fit["free_columns"])
        if free_parameters is None:
            if count:raise ValueError("explicit values for every unidentified parameter required")
            free_parameters=()
        if len(free_parameters)!=count:raise ValueError("complete original free-parameter coordinates required")
        cdef vector[string] free=tuple(_scalar(v,self._bounds[3]) for v in free_parameters)
        cdef vector[string] result, entries
        with nogil:result=self._handle.get().parameter_values(free)
        coefficients=_values(result)
        if self._parameter_keys is None:entry_values=coefficients
        else:
            with nogil:entries=self._handle.get().entry_values(free)
            entry_values=_values(entries)
        family=[list(block) for block in self._maps[1]]
        for (slot,row,col),value in zip(self._slots,entry_values,strict=True):family[slot].append((row,col,value))
        maps=(self._maps[0],tuple(tuple(sorted(block,key=lambda e:(e[0],e[1]))) for block in family),*self._maps[2:])
        materialized=dict(profile="materialized-original-stalk-transports-v1",flag_maps=maps,
            free_parameters=tuple(Fraction(v) for v in free_parameters),parameter_values=coefficients,
            parameter_slots=self._slots)
        if self._parameter_keys is not None:
            materialized.update(profile="materialized-tied-stalk-transports-v2",parameter_space=self._fit["parameter_space"],
                parameter_basis=self._fit["parameter_basis"],entry_values=entry_values)
        return materialized


cdef class PreparedStalkTransportPattern:
    """Own a compact exact relation pattern independently of the training archive."""
    cdef shared_ptr[StalkTransportPattern] _handle
    cdef object _bounds
    cdef void _require(self) except *:
        if self._handle.get()==NULL:raise RuntimeError("pattern owner is not initialized")
    def __init__(self,archive,*,model_namespace,limits=(4096,16000000,64000000,16384),max_bytes=256*1024*1024):
        if self._handle.get()!=NULL:raise RuntimeError("pattern owner is immutable")
        if type(archive) is not bytes:raise TypeError("native pattern bytes required")
        bounds=tuple(_natural(v) for v in limits)
        cdef vector[size_t] admitted=bounds
        cdef size_t ns=_natural(model_namespace),maximum=_natural(max_bytes)
        if len(bounds)!=4 or not all(bounds) or not ns or not maximum or len(archive)>maximum:raise ValueError("positive bounded pattern admission required")
        cdef string image=archive
        with nogil:self._handle=shared_ptr[StalkTransportPattern](new StalkTransportPattern(image,ns,admitted,maximum))
        self._bounds=bounds
    def archive(self):
        """Return canonical source and tensor bytes for cold inference."""
        self._require()
        cdef string result
        with nogil:result=self._handle.get().encode()
        return <bytes>result
    def inspect(self):
        """Read original occurrence identities, parameter coordinates and exact uncertainty."""
        self._require()
        cdef PatternReceipt result
        with nogil:result=self._handle.get().receipt()
        moving={int(e.row) for e in result.directions}
        return dict(profile="exact-context-transport-pattern-v1",weight_origin=_values(result.weights),
            weight_coordinates=tuple(tuple(row) for row in result.coordinates),
            identified=tuple(i not in moving for i in range(result.weights.size())),
            parameter_directions=_view_matrix(result.weights.size(),result.parameters,result.directions),
            input_image=_view_matrix(result.input_rows,result.input_columns,result.input_image),
            occurrence_keys=tuple(int(v) for v in result.occurrence_keys),
            common_frames=tuple((int(f.frame),f.name.decode(),tuple(v.decode() for v in f.axes)) for f in result.frames),
            carrier_frames=tuple(result.carrier_frames),carrier_dimensions=tuple(result.carrier_dimensions),
            pattern_source=<bytes>result.source_archive,source_sha256=result.source_sha256.decode(),
            evidence_sha256=result.evidence_sha256.decode(),observations=int(result.observations),peers=int(result.peers),
            retained_entries=int(result.retained),provenance_bytes=int(result.provenance_bytes),provenance_scope="external-source-and-evidence-references")
    def response(self,common_input,*,shared=True):
        """Return the complete affine response and reject unsupported input directions."""
        self._require()
        if type(shared) is not bool:raise TypeError("shared boundary selection must be boolean")
        if isinstance(common_input,(str,bytes)):raise TypeError("exact common input field required")
        if len(common_input)>self._bounds[0]:raise ValueError("pattern input exceeds dimension admission")
        cdef vector[string] values=tuple(_scalar(v,self._bounds[3]) for v in common_input)
        cdef cppbool combine=shared
        cdef PatternResponse result
        with nogil:result=self._handle.get().response(values,combine)
        return dict(origin=_values(result.origin),directions=_view_matrix(result.origin.size(),result.parameters,result.directions),
            domain_coordinates=_values(result.coordinates),offsets=tuple(result.offsets),identified=result.parameters==0)

cdef object _bank_field(PatternBankResponse value):
    return dict(origin=_values(value.field.origin),
        directions=_view_matrix(value.field.origin.size(),value.field.parameters,value.field.directions),
        domain_coordinates=tuple(_values(v) for v in value.coordinates),offsets=tuple(value.field.offsets),identified=value.field.parameters==0)

cdef object _program_field(PatternProgramResult value,object item):
    return dict(name=item['name'],pattern=item['pattern'],status=value.status.decode(),
        input=dict(profile='complete-sparse-stalk-field-v1',frame=item['input']['frame'] if item['input'] is not None else item['edges'][0]['target_frame'],
            size=int(value.size),entries=tuple((int(e.index),_values((e.value,))[0]) for e in value.input)) if value.has_input else None,
        input_directions=_view_matrix(value.size,value.parameters,value.input_directions) if value.has_input else None,
        parameter_directions=_view_matrix(value.response_size,value.parameters,value.parameter_directions) if value.has_response else None,
        response=_bank_field(value.response) if value.has_response else None)

cdef class PreparedStalkTransportPatternScope:
    """Own original observed column coverage without retaining a training checkpoint."""
    cdef shared_ptr[StalkTransportPatternScope] _handle
    cdef void _require(self) except *:
        if self._handle.get()==NULL:raise RuntimeError("observed scope owner is not initialized")
    def __init__(self,archive,*,source_namespace,limits=(4096,16000000,64000000,16384),max_bytes=256*1024*1024):
        if self._handle.get()!=NULL:raise RuntimeError("observed scope owner is immutable")
        if type(archive) is not bytes:raise TypeError("native observed scope bytes required")
        cdef vector[size_t] bounds=tuple(_natural(v) for v in limits)
        cdef size_t ns=_natural(source_namespace),maximum=_natural(max_bytes)
        if bounds.size()!=4 or not all(bounds) or not ns or not maximum or len(archive)>maximum:
            raise ValueError("positive bounded observed scope admission required")
        cdef string image=archive
        with nogil:self._handle=shared_ptr[StalkTransportPatternScope](new StalkTransportPatternScope(image,ns,bounds,maximum))
    def archive(self):
        """Return canonical native observed source, carrier and input image bytes."""
        self._require()
        cdef string result
        with nogil:result=self._handle.get().encode()
        return <bytes>result
    def inspect(self):
        """Read retained original identities without authenticating external documents."""
        self._require()
        cdef PatternScopeReceipt result
        with nogil:result=self._handle.get().receipt()
        return dict(profile='original-observed-transport-pattern-scope-v1',
            source_keys=tuple(int(v) for v in result.keys),source_types=tuple(v.decode() for v in result.types),
            observations=int(result.observations),retained_entries=int(result.retained),
            observed_source=<bytes>result.source_archive,
            provenance_scope='retained-observed-columns-with-external-source-and-evidence-references')
    def coverage(self,PreparedStalkTransportPatternBank bank not None):
        """Compare complete typed stars and their exact observed input subspaces."""
        self._require();bank._require()
        cdef vector[vector[size_t]] result
        with nogil:result=self._handle.get().coverage(bank._handle.get()[0])
        return tuple(tuple(row) for row in result)
    def for_bank(self,PreparedStalkTransportPatternBank bank not None,*,source_namespace):
        """Keep all observed branches of stored primary types and common input carriers."""
        self._require();bank._require()
        cdef size_t ns=_natural(source_namespace)
        if not ns:raise ValueError("positive local scope namespace required")
        cdef PreparedStalkTransportPatternScope result=PreparedStalkTransportPatternScope.__new__(PreparedStalkTransportPatternScope)
        with nogil:result._handle=shared_ptr[StalkTransportPatternScope](new StalkTransportPatternScope(self._handle.get()[0],bank._handle.get()[0],ns))
        return result

cdef class PreparedStalkTransportPatternTraining:
    """Own exact native rule admissions and compile banks from the retained proof."""
    cdef shared_ptr[StalkTransportPatternTraining] _handle
    cdef PreparedStalkTransport _transport
    cdef object _training, _identity
    cdef _require(self):
        if self._handle.get()==NULL:raise RuntimeError("pattern training owner is not initialized")
    def inspect(self):
        """Return original block admissions without exporting scalar equations."""
        self._require()
        cdef PatternTrainingReceipt result
        with nogil:result=self._handle.get().receipt()
        cdef ContextRule r
        return dict(stalk_transport_build_info(),profile="native-context-pattern-training-v1",
            archive_digest=self._identity[0],evidence_digest=self._identity[1],training_rows=self._training,
            minimum_peers=self._identity[2],observed_inputs=self._identity[3],declared_templates=self._identity[4],
            rules=tuple(dict(origin=int(r.origin),forced_blocks=tuple(r.forced),excluded_blocks=tuple(r.excluded),
                open_blocks=tuple(r.open),admitted=bool(r.admitted)) for r in result.rules),
            block_statuses=tuple(("forced","excluded","open")[i] for i in result.statuses),
            block_ranks=tuple(None if result.statuses[i]==1 else int(result.ranks[i]) for i in range(result.ranks.size())),
            block_rows=tuple(zip(result.origins,result.targets,strict=True)),equation_count=int(result.equations),
            attempts=int(result.attempts),execution_steps=int(result.steps),retained_entries=int(result.retained),
            proof_scope="complete-original-evidence-family",proof_storage="owning-native",truth_status="prospective")
    def compile(self,origins,*,model_namespace,coordinate_ids):
        """Compile admitted original stars without reacquiring their comparisons."""
        self._require()
        if isinstance(origins,(str,bytes)):raise TypeError("ordered original bank origins required")
        if not isinstance(coordinate_ids,Mapping) or set(coordinate_ids)!={'weights','parameters','domains'}:
            raise ValueError("bank weights, parameters and domain identities required")
        cdef vector[size_t] selected
        cdef vector[size_t] coordinates=(_natural(coordinate_ids['weights']),_natural(coordinate_ids['parameters']),*tuple(_natural(i) for i in coordinate_ids['domains']))
        cdef size_t ns=_natural(model_namespace)
        if not ns:raise ValueError("nonzero bank namespace required")
        previous=-1
        for row in origins:
            row=_natural(row)
            if row not in self._training or row<=previous or selected.size()>=self._transport._bounds[0]:
                raise ValueError("ordered distinct training origins required")
            previous=row;selected.push_back(row)
        if not selected.size() or coordinates.size()!=selected.size()+2:raise ValueError("one domain identity per bank origin required")
        cdef PreparedStalkTransportPatternBank result=PreparedStalkTransportPatternBank.__new__(PreparedStalkTransportPatternBank)
        with nogil:result._handle=shared_ptr[StalkTransportPatternBank](new StalkTransportPatternBank(self._handle.get()[0],selected,coordinates,ns))
        result._bounds=self._transport._bounds
        return result

cdef class PreparedStalkTransportPatternBank:
    """Own a retained source union and its joint exact observable parameter family."""
    cdef shared_ptr[StalkTransportPatternBank] _handle
    cdef object _bounds
    cdef void _require(self) except *:
        if self._handle.get()==NULL:raise RuntimeError("model bank owner is not initialized")
    def __init__(self,archive,*,model_namespace,limits=(4096,16000000,64000000,16384),max_bytes=256*1024*1024):
        if self._handle.get()!=NULL:raise RuntimeError("model bank owner is immutable")
        if type(archive) is not bytes:raise TypeError("native model bank bytes required")
        bounds=tuple(_natural(v) for v in limits)
        cdef vector[size_t] admitted=bounds
        cdef size_t ns=_natural(model_namespace),maximum=_natural(max_bytes)
        if len(bounds)!=4 or not all(bounds) or not ns or not maximum or len(archive)>maximum:raise ValueError("positive bounded bank admission required")
        cdef string image=archive
        with nogil:self._handle=shared_ptr[StalkTransportPatternBank](new StalkTransportPatternBank(image,ns,admitted,maximum))
        self._bounds=bounds
    @property
    def limits(self):
        self._require()
        return self._bounds
    def select(self,patterns,*,model_namespace,coordinate_ids):
        """Copy selected stored stars and their joint family into an independent bank."""
        self._require()
        if isinstance(patterns,(str,bytes)):raise TypeError("ordered stored pattern indices required")
        if not isinstance(coordinate_ids,Mapping) or set(coordinate_ids)!={'weights','parameters'}:
            raise ValueError("new weight and parameter coordinate identities required")
        cdef vector[size_t] selected
        cdef vector[size_t] coordinates=(_natural(coordinate_ids['weights']),_natural(coordinate_ids['parameters']))
        cdef size_t ns=_natural(model_namespace),index
        if not ns or not all(coordinates):raise ValueError("nonzero selection namespace and coordinates required")
        for item in patterns:
            index=_natural(item)
            if (selected.size() and index<=selected.back()) or selected.size()>=self._bounds[0]:raise ValueError("ordered distinct admitted stored patterns required")
            selected.push_back(index)
        if not selected.size():raise ValueError("nonempty stored pattern selection required")
        cdef PreparedStalkTransportPatternBank result=PreparedStalkTransportPatternBank.__new__(PreparedStalkTransportPatternBank)
        with nogil:result._handle=shared_ptr[StalkTransportPatternBank](new StalkTransportPatternBank(self._handle.get()[0],selected,coordinates,ns))
        result._bounds=self._bounds
        return result
    def archive(self):
        """Return canonical source and learned tensor bytes."""
        self._require()
        cdef string result
        with nogil:result=self._handle.get().encode()
        return <bytes>result
    def inspect(self):
        """Read source identities, named input images and the complete joint uncertainty."""
        self._require()
        cdef PatternBankReceipt result
        cdef PatternReceipt common
        with nogil:result=self._handle.get().receipt()
        common=result.common
        moving={int(e.row) for e in common.directions}
        return dict(profile="exact-context-transport-pattern-bank-v1",weight_origin=_values(common.weights),
            weight_coordinates=tuple(tuple(row) for row in common.coordinates),identified=tuple(i not in moving for i in range(common.weights.size())),
            parameter_directions=_view_matrix(common.weights.size(),common.parameters,common.directions),
            input_images=tuple(_view_matrix(result.rows[i],result.columns[i],result.images[i]) for i in range(result.rows.size())),
            pattern_keys=tuple(int(v) for v in result.pattern_keys),occurrence_keys=tuple(int(v) for v in common.occurrence_keys),
            common_frames=tuple((int(f.frame),f.name.decode(),tuple(v.decode() for v in f.axes)) for f in common.frames),
            carrier_frames=tuple(common.carrier_frames),carrier_dimensions=tuple(common.carrier_dimensions),
            pattern_source=<bytes>common.source_archive,source_sha256=common.source_sha256.decode(),evidence_sha256=common.evidence_sha256.decode(),
            observations=int(common.observations),peers=int(common.peers),retained_entries=int(common.retained),provenance_bytes=int(common.provenance_bytes),
            provenance_scope="external-source-and-evidence-references",
            parameter_admission="declared-templates" if result.declared_templates else "independent-entries")
    cdef vector[PatternBankInput] _inputs(self,object inputs) except *:
        if not isinstance(inputs,Mapping):raise TypeError("original bank pattern indices mapped to exact input fields required")
        if len(inputs)>self._bounds[0]:raise ValueError("bank input count exceeds admission")
        cdef vector[PatternBankInput] out
        cdef PatternBankInput row
        for key,value in inputs.items():
            row.pattern=_natural(key)
            if isinstance(value,(str,bytes)) or len(value)>self._bounds[0]:raise ValueError("bounded exact common input field required")
            row.value=tuple(_scalar(v,self._bounds[3]) for v in value);out.push_back(row)
        return out
    def response(self,inputs,*,shared=True):
        """Evaluate selected stars with correlated weights on their evidenced input images."""
        self._require()
        if type(shared) is not bool:raise TypeError("shared boundary selection must be boolean")
        cdef vector[PatternBankInput] requests=self._inputs(inputs)
        cdef cppbool combine=shared
        cdef PatternBankResponse result
        with nogil:result=self._handle.get().response(requests,combine)
        return _bank_field(result)
    cdef vector[PatternBankSparseInput] _sparse_inputs(self,object inputs) except *:
        if not isinstance(inputs,Mapping):raise TypeError("original bank pattern indices mapped to complete sparse fields required")
        if len(inputs)>self._bounds[0]:raise ValueError("bank input count exceeds admission")
        cdef vector[PatternBankSparseInput] out
        cdef PatternBankSparseInput row
        cdef PatternSparseEntry entry
        cdef size_t total=0
        for key,value in inputs.items():
            if not isinstance(value,Mapping) or set(value)!={'profile','frame','size','entries'} or value['profile']!='complete-sparse-stalk-field-v1':raise ValueError("closed complete sparse field declaration required")
            row.pattern=_natural(key);row.frame=_natural(value['frame']);row.size=_natural(value['size']);row.entries.clear()
            if row.size>self._bounds[0]:raise ValueError("sparse field exceeds dimension admission")
            for i,v in value['entries']:
                total+=1
                if total>self._bounds[1]:raise ValueError("sparse field coefficients exceed admission")
                entry.index=_natural(i);entry.value=_scalar(v,self._bounds[3]);row.entries.push_back(entry)
            out.push_back(row)
        return out
    def response_sparse(self,inputs,*,shared=True):
        """Evaluate fully supported fields from their declared nonzero coefficients."""
        self._require()
        if type(shared) is not bool:raise TypeError("shared boundary selection must be boolean")
        cdef vector[PatternBankSparseInput] requests=self._sparse_inputs(inputs)
        cdef cppbool combine=shared
        cdef PatternBankResponse result
        with nogil:result=self._handle.get().response_sparse(requests,combine)
        return _bank_field(result)
    def response_sparse_batch(self,batch,*,shared=True):
        """Evaluate a complete sparse field batch under one native action budget."""
        self._require()
        if type(shared) is not bool:raise TypeError("shared boundary selection must be boolean")
        if isinstance(batch,(str,bytes)) or not 0<len(batch)<=self._bounds[0]:raise ValueError("bounded nonempty sparse field batch required")
        cdef vector[vector[PatternBankSparseInput]] requests
        cdef vector[PatternBankSparseInput] row
        cdef PatternBankSparseInput field
        cdef size_t total=0
        for inputs in batch:
            row=self._sparse_inputs(inputs)
            for field in row:
                total+=field.entries.size()
                if total>self._bounds[1]:raise ValueError("sparse batch coefficients exceed admission")
            requests.push_back(row)
        cdef cppbool combine=shared
        cdef vector[PatternBankResponse] results
        with nogil:results=self._handle.get().response_sparse_batch(requests,combine)
        return tuple(_bank_field(value) for value in results)
    def response_sparse_candidates(self,inputs,*,shared=True):
        """Keep each candidate's complete response or explicit outside domain status."""
        self._require()
        if type(shared) is not bool:raise TypeError("shared boundary selection must be boolean")
        cdef vector[PatternBankSparseInput] requests=self._sparse_inputs(inputs)
        cdef cppbool combine=shared
        cdef vector[PatternBankCandidate] results
        with nogil:results=self._handle.get().response_sparse_candidates(requests,combine)
        return tuple(dict(pattern=int(value.pattern),supported=bool(value.supported),
            response=_bank_field(value.response) if value.supported else None) for value in results)
    def response_sparse_program(self,nodes):
        """Compose named stages through complete maps of original occurrence fields."""
        self._require()
        if isinstance(nodes,(str,bytes)) or not 0<len(nodes)<=self._bounds[0]:raise ValueError("bounded nonempty ordered program required")
        cdef vector[PatternProgramNode] declared
        cdef PatternProgramNode node
        cdef PatternProgramEdge edge
        cdef BoundaryMapEntry entry
        cdef vector[PatternBankSparseInput] inputs
        cdef vector[PatternProgramResult] results
        cdef size_t total=0
        for item in nodes:
            if not isinstance(item,Mapping) or set(item)!={'name','pattern','input','edges'}:raise ValueError("closed named program stage required")
            name=item['name']
            if type(name) is not str or not name or len(name.encode())>256:raise ValueError("bounded stage name required")
            node.name=name.encode();node.pattern=_natural(item['pattern']);node.edges.clear()
            node.has_input=item['input'] is not None
            if node.has_input:
                inputs=self._sparse_inputs({item['pattern']:item['input']});node.input=inputs[0]
                total+=node.input.entries.size()
                if total>self._bounds[1]:raise ValueError("program entries exceed admission")
            for e in item['edges']:
                if node.edges.size()>=self._bounds[0] or not isinstance(e,Mapping) or set(e)!={'node','occurrence','source_frame','target_frame','map'}:raise ValueError("closed original occurrence map required")
                edge.node=_natural(e['node']);edge.occurrence=_natural(e['occurrence'])
                edge.source_frame=_natural(e['source_frame']);edge.target_frame=_natural(e['target_frame']);edge.map.clear()
                m=e['map']
                if not isinstance(m,Mapping) or set(m)!={'shape','entries'} or len(m['shape'])!=2:raise ValueError("complete sparse stage map required")
                shape=tuple(_natural(v) for v in m['shape'])
                if max(shape)>self._bounds[0]:raise ValueError("stage map exceeds dimension admission")
                edge.rows=shape[0];edge.columns=shape[1]
                for i,j,v in m['entries']:
                    total+=1
                    if total>self._bounds[1]:raise ValueError("program entries exceed admission")
                    entry.row=_natural(i);entry.column=_natural(j);entry.value=_scalar(v,self._bounds[3]);edge.map.push_back(entry)
                node.edges.push_back(edge)
            declared.push_back(node)
        with nogil:results=self._handle.get().response_sparse_program(declared)
        return tuple(_program_field(results[i],nodes[i]) for i in range(results.size()))
    def glue(self,inputs,comparisons=(),*,shared=True):
        """Glue complete shared cell readings and apply declared interface restrictions."""
        return self._glue(inputs,comparisons,shared,False)
    def glue_sparse(self,inputs,comparisons=(),*,shared=True):
        """Glue complete sparse fields through the declared original interfaces."""
        return self._glue(inputs,comparisons,shared,True)
    cdef object _glue(self,object inputs,object comparisons,object shared,bint sparse):
        self._require()
        if type(shared) is not bool:raise TypeError("shared boundary selection must be boolean")
        cdef vector[PatternBankInput] requests
        cdef vector[PatternBankSparseInput] sparse_requests
        if sparse:sparse_requests=self._sparse_inputs(inputs)
        else:requests=self._inputs(inputs)
        cdef vector[PatternBankComparison] declared
        cdef PatternBankComparison comparison
        cdef BoundaryMapEntry entry
        cdef size_t total=0
        required={'name','left','right','left_frame','right_frame','interface_frame','left_map','right_map'}
        for item in comparisons:
            if declared.size()>=self._bounds[0] or not isinstance(item,Mapping) or set(item)!=required:raise ValueError("complete bounded interface comparison required")
            name=item['name']
            if type(name) is not str or not name or len(name.encode())>256:raise ValueError("bounded comparison name required")
            comparison.name=name.encode();comparison.left=_natural(item['left']);comparison.right=_natural(item['right'])
            comparison.left_frame=_natural(item['left_frame']);comparison.right_frame=_natural(item['right_frame']);comparison.interface_frame=_natural(item['interface_frame'])
            comparison.left_map.clear();comparison.right_map.clear()
            for side in ('left','right'):
                value=item[side+'_map']
                if not isinstance(value,Mapping) or set(value)!={'shape','entries'} or len(value['shape'])!=2:raise ValueError("declared interface matrix shape and entries required")
                shape=tuple(_natural(v) for v in value['shape'])
                if max(shape)>self._bounds[0]:raise ValueError("interface matrix exceeds dimension admission")
                if side=='left':comparison.rows=shape[0];comparison.left_columns=shape[1]
                else:
                    if comparison.rows!=shape[0]:raise ValueError("one common interface carrier required")
                    comparison.right_columns=shape[1]
                for i,j,v in value['entries']:
                    total+=1
                    if total>self._bounds[1]:raise ValueError("interface coefficients exceed admission")
                    entry.row=_natural(i);entry.column=_natural(j);entry.value=_scalar(v,self._bounds[3])
                    if side=='left':comparison.left_map.push_back(entry)
                    else:comparison.right_map.push_back(entry)
            declared.push_back(comparison)
        cdef cppbool combine=shared
        cdef PatternBankGluing result
        with nogil:
            if sparse:result=self._handle.get().glue_sparse(sparse_requests,declared,combine)
            else:result=self._handle.get().glue(requests,declared,combine)
        return dict(consistent=bool(result.consistent),forced=bool(result.forced),
            defect_origin=_values(result.defect_origin),defect_directions=_view_matrix(result.defect_origin.size(),result.parameters,result.defect_directions),
            parameter_origin=_values(result.parameter_origin),parameter_directions=_view_matrix(result.parameters,result.free_parameters,result.parameter_directions),
            contradiction=_values(result.contradiction),comparison_names=tuple(v.decode() for v in result.comparison_names),offsets=tuple(result.offsets),
            section_cells=tuple(result.section_cells),section=_bank_field(result.section) if result.consistent else None,
            boundary_response=_bank_field(result.boundary_response) if result.consistent else None)

cdef class PreparedStalkFieldObjective:
    """Retain an evidence family and its conditional minimum field action family."""
    cdef shared_ptr[StalkFieldObjective] _handle
    cdef PreparedStalkTransport _transport
    cdef object _fit
    def __init__(self,PreparedStalkTransport transport,probes,views,*,weights=None,
                 channel_coefficients=(0,0,0,0),mass=0,c_reading="share",trace_normalized=False):
        if self._handle.get()!=NULL:raise RuntimeError("field objective owner is immutable")
        transport._require()
        bounds=transport._bounds
        if type(trace_normalized) is not bool or c_reading not in ("share","count"):
            raise ValueError("declared boolean normalization and C reading required")
        cdef vector[vector[string]] inputs
        probe_record=[];count=0
        for probe in probes:
            if inputs.size()>=bounds[0]:raise ValueError("probe count exceeds admission")
            field=_whole(probe,transport._axes[1],bounds[3]);count+=len(field)
            if count>bounds[1]:raise ValueError("field objective declarations exceed admission")
            inputs.push_back(field)
            probe_record.append(tuple(tuple(Fraction(v) for v in row) for row in probe))
        cdef vector[ObjectiveView] maps
        cdef ObjectiveView view
        cdef ObjectiveViewTerm term
        view_record=[]
        for name,terms in views:
            if maps.size()>=bounds[0]:raise ValueError("view count exceeds admission")
            if type(name) is not str or not name.strip() or len(name.encode())>256:
                raise ValueError("bounded named original occurrence view required")
            view.name=name.encode();view.terms.clear();record=[]
            for occurrence,axis,value in terms:
                count+=3
                if count>bounds[1]:raise ValueError("field objective declarations exceed admission")
                term.occurrence=_natural(occurrence);term.axis=_natural(axis);term.coefficient=_scalar(value,bounds[3])
                view.terms.push_back(term);record.append((int(term.occurrence),int(term.axis),_values((term.coefficient,))[0]))
            maps.push_back(view);view_record.append((name,tuple(record)))
        if weights is None:weights=(1,)*len(transport._axes[1])
        if len(weights)!=len(transport._axes[1]) or len(channel_coefficients)!=4:
            raise ValueError("original relation weights and four channel coefficients required")
        cdef vector[string] w=tuple(_scalar(v,bounds[3]) for v in weights)
        cdef vector[string] coefficients=tuple(_scalar(v,bounds[3]) for v in channel_coefficients)
        cdef string m=_scalar(mass,bounds[3])
        cdef bint count_reading=c_reading=="count"
        cdef bint normalized=trace_normalized
        with nogil:self._handle=shared_ptr[StalkFieldObjective](new StalkFieldObjective(transport._handle,inputs,maps,w,coefficients,m,count_reading,normalized))
        self._transport=transport
        cdef ObjectiveReceipt result=self._handle.get().receipt()
        cdef ObjectiveField f
        d=len(transport._fit["free_columns"]);n=len(transport._fit["particular"])
        fields=[];i=0
        for f in result.fields:
            fields.append(dict(origin=_values(f.origin),directions=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in f.directions),
                shape=(len(transport._axes[1]),d),probe_index=i//len(view_record),view_name=view_record[i%len(view_record)][0]))
            i+=1
        self._fit=dict(stalk_objective_build_info(),archive_digest=transport._digest,
            evidence_fit=transport.receipt,probes=tuple(probe_record),views=tuple(view_record),weights=_values(w),
            channel_coefficients=_values(coefficients),mass=_values((m,))[0],c_reading=c_reading,trace_normalized=trace_normalized,
            particular=_values(result.particular),kernel=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in result.kernel),
            kernel_shape=(n,len(result.free_columns)),coordinate_particular=_values(result.coordinate_particular),
            coordinate_kernel=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in result.coordinate_kernel),
            coordinate_kernel_shape=(d,len(result.free_columns)),
            hessian=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in result.hessian),hessian_shape=(d,d),
            target=_values(result.target),rank=int(result.rank),free_columns=tuple(result.free_columns),
            fields=tuple(fields),
            initial_energy=_values((result.initial_energy,))[0],minimum_energy=_values((result.minimum_energy,))[0],
            preparation_steps=int(result.steps),retained_entries=int(result.retained),
            interpretation="conditional-field-action-family-v1",field_metric="unit-original-relation-field-v1",
            raw_geometry="weighted-unsigned-boundary-pairing-v1")
    @property
    def receipt(self):
        if self._handle.get()==NULL:raise RuntimeError("field objective owner is not initialized")
        from copy import deepcopy
        return deepcopy(self._fit)
    def apply(self,field):
        if self._handle.get()==NULL:raise RuntimeError("field objective owner is not initialized")
        cdef PreparedStalkTransport transport=self._transport
        cdef vector[string] values=_whole(field,transport._axes[1],transport._bounds[3])
        cdef TransportResponse result
        with nogil:result=self._handle.get().execute(values)
        keys=transport._parameter_keys
        return dict(boundary_family=tuple(_split(_values(row),transport._lower) for row in result.boundary),
            occurrence_family=tuple(_split(_values(row),transport._occurrences) for row in result.occurrences),
            execution_steps=int(result.steps),
            components=("conditional-origin",*("objective-free:"+(str(i) if keys is None else keys[i]) for i in result.directions)),
            zero_components=tuple("objective-free:"+(str(i) if keys is None else keys[i]) for i in result.zero_directions),
            profile="conditional-whole-stalk-field-family-v1")


cdef extern from "stalk_parameter_relations_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass ParameterRelationTerm:
        size_t entry
        string coefficient
    cdef cppclass ParameterRelation:
        string name, expected
        vector[ParameterRelationTerm] terms
    cdef cppclass ParameterRelationsReceipt:
        vector[string] residual_origin
        vector[BoundaryMapEntry] residual_directions
        vector[size_t] statuses
        size_t steps, retained
    cdef cppclass ParameterConditionReceipt:
        bint consistent
        vector[string] particular, coordinate_particular
        vector[BoundaryMapEntry] kernel, coordinate_kernel
        vector[size_t] free_columns
        size_t rank, steps
    cdef cppclass FieldConditionReceipt:
        FieldConditionReceipt() except +
        ParameterConditionReceipt family
        vector[vector[size_t]] constraints
        vector[string] contradiction
        string contradiction_value
        size_t base_rank,additional_rank,status
        cppbool base_consistent
    cdef cppclass StalkParameterRelations:
        StalkParameterRelations(shared_ptr[StalkTransportFit], const vector[ParameterRelation]&, cppbool) except +
        StalkParameterRelations(shared_ptr[StalkTransportFit], const ContextViewRules&) except +
        ParameterRelationsReceipt receipt() except +
        vector[BoundaryMapEntry] sparse_residual() except +
        ParameterConditionReceipt condition(const vector[size_t]&) except +
        FieldConditionReceipt restrict_fields(const vector[size_t]&, const vector[TransportObservation]&) except +
        TransportResponse apply(const vector[size_t]&, const vector[string]&) except +


cdef class PreparedStalkParameterRelations:
    """Retain exact relation residuals over the complete original evidence family."""
    cdef shared_ptr[StalkParameterRelations] _handle
    cdef PreparedStalkTransport _transport
    cdef object _fit, _names
    def __init__(self,PreparedStalkTransport transport,relations,*,allow_constants=False):
        if self._handle.get()!=NULL:raise RuntimeError("parameter relation owner is immutable")
        if transport is None:raise TypeError("owning original transport required")
        if type(allow_constants) is not bool:raise TypeError("constant relation admission must be Boolean")
        cdef cppbool constants=allow_constants
        transport._require();bounds=transport._bounds
        cdef vector[ParameterRelation] maps
        cdef ParameterRelation relation
        cdef ParameterRelationTerm term
        records=[];names=[];count=0
        for name,terms,expected in relations:
            if maps.size()>=bounds[0]:raise ValueError("parameter relation count exceeds admission")
            if type(name) is not str or not name.strip() or len(name.encode())>256 or name in names:
                raise ValueError("distinct bounded parameter relation names required")
            count+=len(name.encode())+2
            relation.name=name.encode();relation.expected=_scalar(expected,bounds[3]);relation.terms.clear();record=[]
            for entry,value in terms:
                count+=2
                if count>bounds[1]:raise ValueError("parameter relation declarations exceed admission")
                term.entry=_natural(entry);term.coefficient=_scalar(value,bounds[3]);relation.terms.push_back(term)
                record.append((int(term.entry),_values((term.coefficient,))[0]))
            maps.push_back(relation);names.append(name)
            records.append((name,tuple(record),_values((relation.expected,))[0]))
        with nogil:self._handle=shared_ptr[StalkParameterRelations](new StalkParameterRelations(transport._handle,maps,constants))
        self._record_fit(transport,tuple(records))
    cdef _record_fit(self,PreparedStalkTransport transport,object records):
        self._transport=transport;self._names=tuple(row[0] for row in records)
        names=self._names
        cdef ParameterRelationsReceipt result=self._handle.get().receipt()
        self._fit=dict(stalk_parameter_relations_build_info(),archive_digest=transport._digest,
            evidence_fit=transport.receipt,relations=tuple(records),residual_origin=_values(result.residual_origin),
            residual_directions=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in result.residual_directions),
            residual_shape=(len(names),len(transport._fit["free_columns"])),
            statuses=tuple(("forced","excluded","open")[i] for i in result.statuses),
            preparation_steps=int(result.steps),retained_entries=int(result.retained),
            interpretation="complete-evidence-parameter-relations-v1",
            evidence_scope="independent-original-entries" if transport._parameter_keys is None else "conditional-on-declared-parameter-basis")
    @property
    def receipt(self):
        if self._handle.get()==NULL:raise RuntimeError("parameter relation owner is not initialized")
        from copy import deepcopy
        return deepcopy(self._fit)
    def sparse_residual(self):
        """Return only nonzero exact residual coordinates in canonical order."""
        if self._handle.get()==NULL:raise RuntimeError("parameter relation owner is not initialized")
        cdef vector[BoundaryMapEntry] result
        with nogil:result=self._handle.get().sparse_residual()
        return tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in result)
    def condition(self,selected=()):
        """Intersect named equations without replacing the original evidence family."""
        if self._handle.get()==NULL:raise RuntimeError("parameter relation owner is not initialized")
        if isinstance(selected,(str,bytes)):raise TypeError("selected relations require a sequence of names")
        cdef vector[size_t] indices
        names=[]
        for name in selected:
            if type(name) is not str or name not in self._names or name in names:
                raise ValueError("distinct selected original parameter relation names required")
            names.append(name);indices.push_back(self._names.index(name))
        cdef ParameterConditionReceipt result
        with nogil:result=self._handle.get().condition(indices)
        n=len(self._transport._fit["particular"]);d=len(self._transport._fit["free_columns"])
        return dict(profile="conditional-parameter-relation-family-v1",selected=tuple(names),consistent=bool(result.consistent),
            particular=_values(result.particular),kernel=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in result.kernel),
            kernel_shape=(n,len(result.free_columns)),coordinate_particular=_values(result.coordinate_particular),
            coordinate_kernel=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in result.coordinate_kernel),
            coordinate_kernel_shape=(d,len(result.free_columns)),free_columns=tuple(result.free_columns),
            rank=int(result.rank),execution_steps=int(result.steps))
    def restrict_fields(self,observations,selected=()):
        """Intersect whole observed fields jointly and retain exact contradiction witnesses."""
        if self._handle.get()==NULL:raise RuntimeError("parameter relation owner is not initialized")
        if isinstance(selected,(str,bytes)):raise TypeError("selected relations require a sequence of names")
        cdef vector[size_t] indices
        names=[]
        for name in selected:
            if type(name) is not str or name not in self._names or name in names:
                raise ValueError("distinct selected original parameter relation names required")
            names.append(name);indices.push_back(self._names.index(name))
        cdef vector[TransportObservation] samples
        cdef TransportObservation sample
        records=[];bounds=self._transport._bounds;count=0;upper=[0]
        for local in self._transport._axes[1]:upper.append(upper[-1]+len(local))
        for observation in observations:
            if samples.size()>=bounds[0]:raise ValueError("field restriction count exceeds admission")
            if not isinstance(observation,Mapping) or set(observation)!={"input","target","observed"}:
                raise ValueError("closed whole-field observation required")
            sample.input=_whole(observation["input"],self._transport._axes[1],bounds[3])
            sample.target=_whole(observation["target"],self._transport._axes[0],bounds[3])
            mask=tuple(observation["observed"])
            if len(mask)!=len(self._transport._axes[0]) or any(type(v) is not bool for v in mask):
                raise ValueError("boolean mask on original lower stalks required")
            count+=sample.input.size()+sample.target.size()+len(mask)
            if count>bounds[1]:raise ValueError("field restriction declarations exceed entry admission")
            sample.observed=tuple(int(v) for v in mask);samples.push_back(sample)
            records.append(dict(input=_split(_values(sample.input),upper),
                target=_split(_values(sample.target),self._transport._lower),observed=mask))
        cdef FieldConditionReceipt receipt
        with nogil:receipt=self._handle.get().restrict_fields(indices,samples)
        cdef ParameterConditionReceipt result=receipt.family
        n=len(self._transport._fit["particular"]);d=len(self._transport._fit["free_columns"])
        family=dict(consistent=bool(result.consistent),particular=_values(result.particular),
            kernel=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in result.kernel),
            kernel_shape=(n,len(result.free_columns)),coordinate_particular=_values(result.coordinate_particular),
            coordinate_kernel=tuple((int(e.row),int(e.column),_values((e.value,))[0]) for e in result.coordinate_kernel),
            coordinate_kernel_shape=(d,len(result.free_columns)),free_columns=tuple(result.free_columns),rank=int(result.rank))
        return dict(profile="conditional-whole-stalk-field-restriction-v1",selected=tuple(names),
            archive_digest=self._transport._digest,observations=tuple(records),family=family,
            status=("forced","excluded","open","inconsistent-base")[receipt.status],
            base_consistent=bool(receipt.base_consistent),base_rank=int(receipt.base_rank),
            additional_rank=int(receipt.additional_rank),constraints=tuple(tuple(row) for row in receipt.constraints),
            contradiction=_values(receipt.contradiction),
            contradiction_value=_values((receipt.contradiction_value,))[0] if receipt.contradiction_value.size() else None,
            execution_steps=int(result.steps))
    def apply(self,field,selected=()):
        """Apply the complete conditional family to a whole original upper field."""
        if self._handle.get()==NULL:raise RuntimeError("parameter relation owner is not initialized")
        if isinstance(selected,(str,bytes)):raise TypeError("selected relations require a sequence of names")
        cdef vector[size_t] indices
        names=[]
        for name in selected:
            if type(name) is not str or name not in self._names or name in names:
                raise ValueError("distinct selected original parameter relation names required")
            names.append(name);indices.push_back(self._names.index(name))
        cdef vector[string] values=_whole(field,self._transport._axes[1],self._transport._bounds[3])
        cdef TransportResponse result
        with nogil:result=self._handle.get().apply(indices,values)
        return dict(boundary_family=tuple(_split(_values(row),self._transport._lower) for row in result.boundary),
            occurrence_family=tuple(_split(_values(row),self._transport._occurrences) for row in result.occurrences),
            execution_steps=int(result.steps),components=("conditional-origin",*("free:"+str(i) for i in result.directions)),
            zero_components=tuple("free:"+str(i) for i in result.zero_directions),selected=tuple(names),
            profile="conditional-whole-stalk-affine-response-v1")
