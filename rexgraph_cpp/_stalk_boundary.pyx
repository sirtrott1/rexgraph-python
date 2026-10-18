# distutils: language=c++
"""Owning exact boundary traces on original graded stalks."""
from fractions import Fraction
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libcpp cimport bool as cppbool
from ._stalk_fields import _natural, _scalar, _axes, _values, _split

cdef extern from "stalk_boundary_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass BoundaryMapEntry:
        size_t row, column
        string value
    cdef cppclass BoundaryTrace:
        vector[vector[string]] fields, occurrences
        vector[vector[vector[string]]] channels
        size_t steps
    cdef cppclass BoundaryInjection:
        vector[string] field, boundary
        size_t steps, columns, occurrences
    cdef cppclass StalkTransportDeclaration:
        StalkTransportDeclaration(const string&, const vector[vector[vector[string]]]&, const vector[vector[vector[BoundaryMapEntry]]]&, const vector[size_t]&, size_t) except +
        vector[vector[size_t]] offsets() except +
        size_t preparation_steps() except +
        size_t retained_entries() except +
        vector[vector[size_t]] occurrence_cells(size_t) except +
        vector[string] occurrence_coefficients(size_t) except +
        vector[string] occurrence_keys(size_t) except +
        vector[size_t] occurrence_activity(size_t) except +
    cdef cppclass StalkBoundaryTower(StalkTransportDeclaration):
        StalkBoundaryTower(const string&, const vector[vector[vector[string]]]&, const vector[vector[vector[BoundaryMapEntry]]]&, const vector[size_t]&, size_t) except +
        BoundaryTrace execute(size_t, size_t, const vector[string]&) except +
        BoundaryInjection inject(size_t, const string&, const vector[size_t]&, const vector[string]&, size_t) except +

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_STALK_BOUNDARY_BINDING_DIGEST

cdef extern from "graded_transport_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass GradedTransportObservation:
        vector[string] input, target
        vector[size_t] observed
    cdef cppclass GradedTransportReceipt:
        vector[BoundaryMapEntry] chain, design, observation_design, kernel
        vector[vector[size_t]] chain_coordinates, observation_coordinates
        vector[string] particular, target, observation_target
        vector[size_t] pivots, free_columns
        size_t chain_rows, design_rows, observation_rows, parameters, rank, chain_rank, observation_rank, steps, entries
    cdef cppclass GradedTransportResponse:
        vector[string] boundary, occurrences
        vector[BoundaryMapEntry] boundary_directions, occurrence_directions
        size_t free_count, steps
    cdef cppclass GradedTransportMaterialization:
        vector[string] parameters, entries, local_values
        size_t steps
    cdef cppclass GradedTransportFitBridge:
        GradedTransportFitBridge(const StalkTransportDeclaration&, size_t, const vector[vector[size_t]]&,
            const vector[GradedTransportObservation]&, cppbool, size_t,
            const vector[BoundaryMapEntry]&, const vector[size_t]&) except +
        shared_ptr[GradedTransportFitBridge] append(const vector[GradedTransportObservation]&) except +
        vector[size_t] update_stats() except +
        GradedTransportReceipt receipt() except +
        GradedTransportResponse execute(const vector[string]&) except +
        GradedTransportMaterialization materialize(const vector[string]&, const vector[string]&) except +

def graded_transport_build_info():
    return dict(stalk_boundary_build_info(), profile="owning-single-grade-chain-transport-v1")

cdef object _graded_entries(vector[BoundaryMapEntry] records):
    return tuple((int(e.row), int(e.column), _values((e.value,))[0]) for e in records)

def _graded_field(axes, field, bits):
    rows=[]
    for i,row in enumerate(field):
        if i>=len(axes):raise ValueError("field exceeds original grade cells")
        values=[]
        for value in row:
            if len(values)>=len(axes[i]):raise ValueError("field exceeds its original named stalk")
            _scalar(value,bits);values.append(Fraction(value))
        if len(values)!=len(axes[i]):raise ValueError("complete original named stalk field required")
        rows.append(tuple(values))
    if len(rows)!=len(axes):raise ValueError("field must cover every original grade cell")
    return tuple(rows)

def stalk_boundary_build_info():
    return dict(engine="rexgraph-cpp",engine_version=RFC_ENGINE_VERSION.decode(),
        engine_headers=RFC_ENGINE_DIGEST.decode(),binding=RFC_STALK_BOUNDARY_BINDING_DIGEST.decode(),
        profile="owning-graded-stalk-boundary-v1")

cdef class PreparedStalkDeclaration:
    """Own original stalk/map declarations without publishing a boundary executor."""
    cdef shared_ptr[StalkTransportDeclaration] _handle
    cdef shared_ptr[StalkBoundaryTower] _boundary
    cdef object _axes, _maps, _bounds, _maximum, _digest, _offsets, _cells, _coefficients, _activity, _occurrence_offsets, _keys
    cdef void _require(self) except *:
        if self._handle.get()==NULL and self._boundary.get()==NULL:
            raise RuntimeError("stalk declaration owner is not initialized")
    cdef StalkTransportDeclaration* _declaration(self) except NULL:
        self._require()
        if self._boundary.get()!=NULL:return self._boundary.get()
        return self._handle.get()
    def __init__(self,archive,axes,flag_maps,*,limits=(4096,16000000,64000000,16384),max_bytes=256*1024*1024):
        self._initialize(archive,axes,flag_maps,limits,max_bytes,False)
    cdef void _initialize(self,archive,axes,flag_maps,limits,max_bytes,cppbool strict) except *:
        from hashlib import sha256
        if self._handle.get()!=NULL or self._boundary.get()!=NULL:raise RuntimeError("stalk declaration owner is immutable")
        if not isinstance(archive,bytes):raise TypeError("stalk boundaries require native source bytes")
        bounds=tuple(_natural(v) for v in limits);maximum=_natural(max_bytes)
        if len(bounds)!=4 or not all(bounds) or not maximum:raise ValueError("positive action and archive bounds required")
        if len(archive)>maximum:raise ValueError("stalk boundary archive exceeds byte admission")
        declared=[];dimension=0
        for grade in axes:
            if len(declared)>=bounds[0]:raise ValueError("grade count exceeds admission")
            family_axes=_axes(grade,bounds[0]);dimension+=sum(map(len,family_axes))
            if dimension>bounds[0]:raise ValueError("all grade stalk dimensions exceed admission")
            declared.append(family_axes)
        declared=tuple(declared)
        if not declared:raise ValueError("complete original grade axes required")
        cdef vector[vector[vector[string]]] names=tuple(tuple(tuple(n.encode() for n in row) for row in grade) for grade in declared)
        cdef vector[vector[vector[BoundaryMapEntry]]] maps
        cdef vector[vector[BoundaryMapEntry]] family
        cdef vector[BoundaryMapEntry] block
        cdef BoundaryMapEntry entry
        retained=[];count=0
        for grade in flag_maps:
            if maps.size()>=len(declared):raise ValueError("flag maps exceed original grades")
            family.clear();records=[]
            for entries in grade:
                count+=1
                if count>bounds[1]:raise ValueError("original flags exceed entry admission")
                block.clear();record=[]
                for item in entries:
                    count+=1
                    if count>bounds[1]:raise ValueError("declared blocks exceed entry admission")
                    if len(item)!=3:raise ValueError("flag entry requires row, column and exact coefficient")
                    entry.row=_natural(item[0]);entry.column=_natural(item[1]);entry.value=_scalar(item[2],bounds[3])
                    block.push_back(entry);record.append((int(entry.row),int(entry.column),Fraction(item[2])))
                family.push_back(block);records.append(tuple(record))
            maps.push_back(family);retained.append(tuple(records))
        cdef string image=archive
        cdef vector[size_t] admitted=bounds
        cdef size_t cap=maximum
        if strict:
            with nogil:self._boundary=shared_ptr[StalkBoundaryTower](new StalkBoundaryTower(image,names,maps,admitted,cap))
        else:
            with nogil:self._handle=shared_ptr[StalkTransportDeclaration](new StalkTransportDeclaration(image,names,maps,admitted,cap))
        self._axes=declared;self._maps=tuple(retained);self._bounds=bounds;self._maximum=maximum
        self._digest=sha256(archive).hexdigest();self._offsets=tuple(tuple(v) for v in self._declaration().offsets())
        cells=[()];coefficients=[()];activity=[()];offsets=[(0,)];keys=[()]
        for k in range(1,len(declared)):
            local=tuple(tuple(v) for v in self._declaration().occurrence_cells(k));cells.append(local)
            coefficients.append(_values(self._declaration().occurrence_coefficients(k)))
            keys.append(tuple(int(v) for v in self._declaration().occurrence_keys(k)))
            activity.append(tuple(bool(v) for v in self._declaration().occurrence_activity(k)))
            positions=[0]
            for _,lo in local:positions.append(positions[-1]+len(declared[k-1][lo]))
            offsets.append(tuple(positions))
        self._cells=tuple(cells);self._coefficients=tuple(coefficients);self._activity=tuple(activity)
        self._occurrence_offsets=tuple(offsets);self._keys=tuple(keys)
    @property
    def receipt(self):
        self._require()
        info = stalk_boundary_build_info()
        if self._boundary.get()==NULL: info['profile']='owning-graded-stalk-declaration-v1'
        return dict(info,archive_digest=self._digest,stalk_axes=self._axes,flag_maps=self._maps,
            occurrence_cells=self._cells,occurrence_coefficients=self._coefficients,occurrence_activity=self._activity,
            occurrence_keys=self._keys,
            dimensions=tuple(v[-1] for v in self._offsets),limits=self._bounds,
            preparation_steps=int(self._declaration().preparation_steps()),retained_entries=int(self._declaration().retained_entries()),
            strict_chain_family=self._boundary.get()!=NULL,
            readout="original-grades-and-intermediate-channels-v1" if self._boundary.get()!=NULL else "original-stalk-declaration-v1")
    def fit_transports(self,grade,parameter_slots,observations=(),*,parameter_space=None,parameter_basis=None):
        self._require()
        return PreparedGradedStalkTransport(self,grade,parameter_slots,observations,
            parameter_space=parameter_space,parameter_basis=parameter_basis)

cdef class PreparedStalkBoundary(PreparedStalkDeclaration):
    """Execute original graded stalk maps after checking every adjacent law."""
    def __init__(self,archive,axes,flag_maps,*,limits=(4096,16000000,64000000,16384),max_bytes=256*1024*1024):
        self._initialize(archive,axes,flag_maps,limits,max_bytes,True)
    cdef void _require_boundary(self) except *:
        if self._boundary.get()==NULL:raise RuntimeError("stalk boundary owner is not initialized")
    def apply(self,grade,field,*,boundary_steps=1):
        self._require_boundary();grade=_natural(grade);steps=_natural(boundary_steps)
        if grade>=len(self._axes) or steps>grade:raise ValueError("boundary steps exceed the original input grade")
        cdef vector[string] values
        cells=0
        for i,row in enumerate(field):
            if i>=len(self._axes[grade]):raise ValueError("field exceeds original grade cells")
            size=0
            for value in row:
                size+=1
                if size>len(self._axes[grade][i]):raise ValueError("field exceeds its named local stalk")
                values.push_back(_scalar(value,self._bounds[3]))
            if size!=len(self._axes[grade][i]):raise ValueError("complete named stalk required")
            cells+=1
        if cells!=len(self._axes[grade]):raise ValueError("field must cover every original cell")
        cdef size_t k=grade,n=steps
        cdef BoundaryTrace result
        with nogil:result=self._boundary.get().execute(k,n,values)
        fields=tuple(_split(_values(v),self._offsets[grade-i]) for i,v in enumerate(result.fields))
        occurrences=tuple(_split(_values(v),self._occurrence_offsets[grade-i]) for i,v in enumerate(result.occurrences))
        channels=tuple(tuple(_split(_values(v),self._offsets[grade-i-2]) for v in group)
            if i+1<steps else () for i,group in enumerate(result.channels))
        return dict(grades=tuple(range(grade,grade-steps-1,-1)),fields=fields,occurrence_fields=occurrences,
            intermediate_channels=channels,execution_steps=int(result.steps),receipt=self.receipt)
    def inject_acquisition(self,grade,archive,mapping,amplitudes):
        self._require_boundary();grade=_natural(grade)
        if grade>=len(self._axes):raise ValueError("injection requires an original grade")
        from hashlib import sha256
        if not isinstance(archive,bytes):raise TypeError("acquisition requires native source bytes")
        if len(archive)>self._maximum:raise ValueError("acquisition archive exceeds byte admission")
        cdef vector[size_t] directions
        for value in mapping:
            if directions.size()>=self._bounds[0]:raise ValueError("injection dimensions exceed admission")
            directions.push_back(_natural(value))
        cdef vector[string] values
        for value in amplitudes:
            if values.size()>=self._bounds[0]:raise ValueError("acquisition columns exceed admission")
            values.push_back(_scalar(value,self._bounds[3]))
        cdef string image=archive
        cdef size_t k=grade,cap=self._maximum
        cdef BoundaryInjection result
        with nogil:result=self._boundary.get().inject(k,image,directions,values,cap)
        return dict(grade=grade,field=_split(_values(result.field),self._offsets[grade]),boundary=_values(result.boundary),
            acquisition_archive_digest=sha256(archive).hexdigest(),columns=int(result.columns),retained_occurrences=int(result.occurrences),
            mapping=tuple(directions),amplitudes=_values(values),execution_steps=int(result.steps),
            profile="declared-graded-stalk-acquisition-v1")


cdef class PreparedGradedStalkTransport:
    """Fit one original transport grade subject to the complete strict chain family."""
    cdef shared_ptr[GradedTransportFitBridge] _handle
    cdef PreparedStalkDeclaration _source
    cdef object _grade, _slots, _samples, _space, _basis, _fit, _named
    cdef size_t _input_entries
    cdef void _require(self) except *:
        if self._handle.get()==NULL:raise RuntimeError("graded transport owner is not initialized")

    def __init__(self,PreparedStalkDeclaration source,grade,parameter_slots,observations=(),*,parameter_space=None,parameter_basis=None):
        from collections.abc import Mapping
        if self._handle.get()!=NULL:raise RuntimeError("graded transport owner is immutable")
        source._require();grade=_natural(grade)
        if not 0<grade<len(source._axes):raise ValueError("positive original learned transport grade required")
        bounds=source._bounds;slots=[];seen=set()
        for row in parameter_slots:
            if len(slots)>=bounds[0]:raise ValueError("parameter support exceeds dimension admission")
            if len(row)!=3:raise ValueError("original occurrence, row and column required")
            item=tuple(_natural(v) for v in row)
            if item in seen:raise ValueError("distinct original parameter support required")
            seen.add(item);slots.append(item)
        slots=tuple(slots);basis=[]
        named=parameter_space is not None
        if named:
            if not isinstance(parameter_space,Mapping) or set(parameter_space)!={'name','keys'}:
                raise ValueError("closed named parameter space required")
            name=parameter_space['name'];keys=tuple(parameter_space['keys'])
            if (not isinstance(name,str) or not name or len(name.encode())>256 or len(keys)>bounds[0] or
                len(set(keys))!=len(keys) or any(not isinstance(n,str) or not n or len(n.encode())>256 for n in keys)):
                raise ValueError("distinct bounded original parameter names required")
            if parameter_basis is None:raise ValueError("named parameters require a complete declared basis")
            space=(name,keys);seen=set()
            for row in parameter_basis:
                if len(basis)>=bounds[1]:raise ValueError("parameter basis exceeds entry admission")
                if len(row)!=3:raise ValueError("parameter basis requires support row, parameter and exact coefficient")
                i,j=_natural(row[0]),_natural(row[1]);_scalar(row[2],bounds[3])
                if i>=len(slots) or j>=len(keys) or (i,j) in seen:raise ValueError("distinct declared parameter basis coordinates required")
                seen.add((i,j));basis.append((i,j,Fraction(row[2])))
        else:
            if parameter_basis is not None:raise ValueError("parameter basis requires a named parameter space")
            space=('original-transport-entries',tuple('entry:'+str(i) for i in range(len(slots))))
        basis=tuple(basis);retained=len(slots)*3+len(basis)*3
        if retained>bounds[1]:raise ValueError("original parameter declarations exceed entry admission")
        self._source=source;self._grade=grade;self._slots=slots;self._space=space;self._basis=basis;self._named=named
        cdef vector[GradedTransportObservation] acquired
        samples,retained=self._acquire(observations,acquired,0,retained)
        cdef vector[vector[size_t]] support=slots
        cdef vector[BoundaryMapEntry] directions
        cdef BoundaryMapEntry entry
        for i,j,value in basis:
            entry.row=i;entry.column=j;entry.value=_scalar(value,bounds[3]);directions.push_back(entry)
        cdef size_t k=grade,n=len(space[1])
        cdef cppbool declared=named
        cdef vector[size_t] admitted=bounds
        cdef StalkTransportDeclaration* source_declaration=source._declaration()
        with nogil:self._handle=shared_ptr[GradedTransportFitBridge](new GradedTransportFitBridge(
            source_declaration[0],k,support,acquired,declared,n,directions,admitted))
        self._source=source;self._grade=grade;self._slots=slots;self._samples=tuple(samples);self._space=space;self._basis=basis;self._named=named;self._input_entries=retained
        self._retain_fit()

    cdef object _acquire(self,observations,vector[GradedTransportObservation]& acquired,size_t count,size_t retained):
        from collections.abc import Mapping
        bounds=self._source._bounds;samples=[]
        cdef GradedTransportObservation observation
        for sample in observations:
            if count+len(samples)>=bounds[0]:raise ValueError("observation count exceeds dimension admission")
            if not isinstance(sample,Mapping) or set(sample)!={'input','target','observed'}:
                raise ValueError("closed whole-field observation required")
            field=_graded_field(self._source._axes[self._grade],sample['input'],bounds[3])
            target=_graded_field(self._source._axes[self._grade-1],sample['target'],bounds[3]);mask=tuple(sample['observed'])
            if len(mask)!=len(target) or any(type(v) is not bool for v in mask):raise ValueError("original lower-stalk Boolean mask required")
            retained+=sum(map(len,field))+sum(map(len,target))+len(mask)
            if retained>bounds[1]:raise ValueError("complete observation fields exceed entry admission")
            observation.input=tuple(_scalar(v,bounds[3]) for row in field for v in row)
            observation.target=tuple(_scalar(v,bounds[3]) for row in target for v in row)
            observation.observed=mask;acquired.push_back(observation)
            samples.append(dict(input=field,target=target,observed=mask))
        return tuple(samples),retained

    cdef void _retain_fit(self) except *:
        n=len(self._space[1])
        cdef GradedTransportReceipt receipt=self._handle.get().receipt()
        self._fit=dict(chain_shape=(int(receipt.chain_rows),n),chain=_graded_entries(receipt.chain),
            chain_coordinates=tuple(tuple(v) for v in receipt.chain_coordinates),
            design_shape=(int(receipt.design_rows),n),design=_graded_entries(receipt.design),target=_values(receipt.target),
            observation_design_shape=(int(receipt.observation_rows),n),observation_design=_graded_entries(receipt.observation_design),
            observation_target=_values(receipt.observation_target),observation_coordinates=tuple(tuple(v) for v in receipt.observation_coordinates),
            particular=_values(receipt.particular),kernel_shape=(n,receipt.free_columns.size()),kernel=_graded_entries(receipt.kernel),
            rank=int(receipt.rank),chain_rank=int(receipt.chain_rank),observation_rank=int(receipt.observation_rank),
            pivots=tuple(receipt.pivots),free_columns=tuple(receipt.free_columns),preparation_steps=int(receipt.steps),retained_entries=int(receipt.entries))

    @property
    def receipt(self):
        self._require()
        return dict(graded_transport_build_info(),source=self._source.receipt,grade=self._grade,parameter_slots=self._slots,
            parameter_space=self._space,parameter_basis=self._basis,named_parameters=self._named,
            observations=tuple(dict(s) for s in self._samples),fit=dict(self._fit),strict_chain_family=True)

    @property
    def execution_stats(self):
        self._require();values=tuple(self._handle.get().update_stats())
        return dict(chain_reused=bool(values[0]),source_reused=bool(values[0]),reused_observations=values[1],
            assembled_observations=values[2],guard_steps=values[3])

    def append(self,observations):
        """Insert new whole fields through the retained original chain and row family."""
        self._require()
        cdef vector[GradedTransportObservation] acquired
        suffix,retained=self._acquire(observations,acquired,len(self._samples),self._input_entries)
        if not suffix:raise ValueError("nonempty original observation suffix required")
        cdef PreparedGradedStalkTransport child=PreparedGradedStalkTransport.__new__(PreparedGradedStalkTransport)
        with nogil:child._handle=self._handle.get().append(acquired)
        child._source=self._source;child._grade=self._grade;child._slots=self._slots;child._samples=self._samples+suffix
        child._space=self._space;child._basis=self._basis;child._named=self._named;child._input_entries=retained
        child._retain_fit();return child

    def apply(self,field):
        self._require();rows=_graded_field(self._source._axes[self._grade],field,self._source._bounds[3])
        cdef vector[string] values=tuple(_scalar(v,self._source._bounds[3]) for row in rows for v in row)
        cdef GradedTransportResponse result
        with nogil:result=self._handle.get().execute(values)
        return dict(grade=self._grade,input=rows,
            boundary=_split(_values(result.boundary),self._source._offsets[self._grade-1]),
            occurrences=_split(_values(result.occurrences),self._source._occurrence_offsets[self._grade]),
            boundary_directions=_graded_entries(result.boundary_directions),occurrence_directions=_graded_entries(result.occurrence_directions),
            free_columns=self._fit['free_columns'],free_count=int(result.free_count),execution_steps=int(result.steps),receipt=self.receipt)

    def materialize(self,*,free_parameters=None):
        self._require();free=self._fit['free_columns']
        if free_parameters is None:
            if free:raise ValueError("explicit values for every original free parameter required")
            free_parameters=()
        free_parameters=tuple(free_parameters)
        if len(free_parameters)!=len(free):raise ValueError("complete original free parameter carrier required")
        cdef vector[string] values=tuple(_scalar(v,self._source._bounds[3]) for v in free_parameters)
        original=[dict(((i,j),q) for i,j,q in block) for block in self._source._maps[self._grade]]
        cdef vector[string] origins=tuple(_scalar(original[p].get((i,j),Fraction(0)),self._source._bounds[3]) for p,i,j in self._slots)
        cdef GradedTransportMaterialization result
        with nogil:result=self._handle.get().materialize(values,origins)
        entries=_values(result.entries);family=[list(grade) for grade in self._source._maps]
        blocks=[dict(((i,j),q) for i,j,q in block) for block in family[self._grade]]
        for (p,i,j),q in zip(self._slots,_values(result.local_values)):blocks[p][(i,j)]=q
        family[self._grade]=[tuple((i,j,q) for (i,j),q in sorted(block.items())) for block in blocks]
        return dict(flag_maps=tuple(tuple(grade) for grade in family),parameter_values=_values(result.parameters),
            entry_values=entries,free_parameters=tuple(Fraction(v) for v in free_parameters),
            strict_chain_family=True,execution_steps=int(result.steps),receipt=self.receipt)
