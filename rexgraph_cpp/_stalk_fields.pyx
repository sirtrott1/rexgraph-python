# distutils: language=c++
"""Exact rational fields on named, ragged relation stalks."""
from fractions import Fraction
from numbers import Integral
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t

cdef extern from "stalk_field_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass StalkMapEntry:
        size_t row, column
        string value
    cdef cppclass StalkFieldResult:
        vector[vector[string]] response, boundary, occurrences
        size_t steps
    cdef cppclass StalkInjectionResult:
        vector[string] field, boundary
        size_t steps, columns, occurrences
    cdef cppclass StalkFieldResponse:
        StalkFieldResponse(const string&, const vector[vector[string]]&, const vector[vector[string]]&,
                           const vector[vector[StalkMapEntry]]&, const string&, const vector[size_t]&, size_t) except +
        vector[size_t] upper_offsets() except +
        vector[size_t] lower_offsets() except +
        vector[size_t] occurrence_lower() except +
        vector[size_t] occurrence_upper() except +
        vector[string] occurrence_coefficients() except +
        vector[size_t] occurrence_activity() except +
        size_t preparation_steps() except +
        size_t retained_entries() except +
        StalkFieldResult apply(const vector[vector[string]]&) except +
        StalkInjectionResult inject(const string&, const vector[size_t]&, const vector[string]&, size_t) except +

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_STALK_BINDING_DIGEST

def stalk_fields_build_info():
    return dict(engine="rexgraph-cpp",engine_version=RFC_ENGINE_VERSION.decode(),
        engine_headers=RFC_ENGINE_DIGEST.decode(),binding=RFC_STALK_BINDING_DIGEST.decode(),
        profile="owning-ragged-stalk-fields-green-v1")

def _natural(value):
    if isinstance(value,bool) or not isinstance(value,Integral) or not 0<=value<(1<<64):
        raise ValueError("stalk index or bound requires an unsigned integer")
    return int(value)

def _scalar(value,bits):
    if isinstance(value,bool) or not isinstance(value,(Integral,Fraction)):
        raise TypeError("stalk fields require integer or Fraction coefficients")
    q=Fraction(int(value)) if isinstance(value,Integral) else Fraction(int(value.numerator),int(value.denominator))
    if max(abs(q.numerator).bit_length(),q.denominator.bit_length())>bits:
        raise ValueError("stalk coefficient exceeds integer bit admission")
    return (format(q.numerator,"x")+"/"+format(q.denominator,"x")).encode()

def _values(encoded):
    out=[]
    for word in encoded:
        parts=word.decode("ascii").split("/")
        out.append(Fraction(int(parts[0],16),int(parts[1],16) if len(parts)==2 else 1))
    return tuple(out)

def _axes(rows,limit):
    result=[];total=0
    for row in rows:
        names=[]
        for name in row:
            if not isinstance(name,str) or not name or len(name.encode())>256:
                raise ValueError("bounded nonempty stalk axis names required")
            total+=1
            if total>limit: raise ValueError("stalk dimensions exceed admission")
            names.append(name)
        if not names or len(set(names))!=len(names):raise ValueError("distinct nonempty stalk axes required")
        result.append(tuple(names))
        if len(result)>limit:raise ValueError("stalk cell count exceeds admission")
    return tuple(result)

def _split(values,offsets):
    return tuple(values[a:b] for a,b in zip(offsets,offsets[1:]))

cdef class PreparedStalkFields:
    """Own exact flag blocks, their source and one reusable Green solve."""
    cdef shared_ptr[StalkFieldResponse] _handle
    cdef object _bounds, _maximum, _archive_digest, _lower, _upper, _maps
    cdef object _lower_offsets, _upper_offsets, _occurrence_offsets, _cells, _parameter, _coefficients, _activity
    cdef void _require(self) except *:
        if self._handle.get()==NULL:raise RuntimeError("stalk field owner is not initialized")
    def __init__(self,archive,lower_axes,upper_axes,flag_maps,*,parameter=Fraction(1),
                 limits=(4096,16000000,64000000,16384),max_bytes=256*1024*1024):
        from hashlib import sha256
        if self._handle.get()!=NULL:raise RuntimeError("stalk field owner is immutable")
        if not isinstance(archive,bytes):raise TypeError("stalk fields require native source bytes")
        bounds=tuple(_natural(i) for i in limits);maximum=_natural(max_bytes)
        if len(bounds)!=4 or not all(bounds) or not maximum:raise ValueError("positive stalk and archive bounds required")
        if len(archive)>maximum:raise ValueError("stalk archive exceeds byte admission")
        lower=_axes(lower_axes,bounds[0]);upper=_axes(upper_axes,bounds[0])
        cdef vector[vector[string]] lo=tuple(tuple(n.encode() for n in r) for r in lower)
        cdef vector[vector[string]] up=tuple(tuple(n.encode() for n in r) for r in upper)
        cdef vector[vector[StalkMapEntry]] maps
        cdef vector[StalkMapEntry] entries
        cdef StalkMapEntry entry
        retained=[];count=0
        for block in flag_maps:
            if maps.size()>=bounds[1]:raise ValueError("flag count exceeds entry admission")
            entries.clear();record=[]
            for item in block:
                count+=1
                if count>bounds[1]:raise ValueError("flag maps exceed entry admission")
                if len(item)!=3:raise ValueError("flag entry requires row, column and exact coefficient")
                entry.row=_natural(item[0]);entry.column=_natural(item[1]);entry.value=_scalar(item[2],bounds[3])
                entries.push_back(entry);record.append((int(entry.row),int(entry.column),Fraction(item[2])))
            maps.push_back(entries);retained.append(tuple(record))
        cdef string image=archive
        cdef string q=_scalar(parameter,bounds[3])
        cdef vector[size_t] admitted=bounds
        cdef size_t cap=maximum
        with nogil:self._handle=shared_ptr[StalkFieldResponse](new StalkFieldResponse(image,lo,up,maps,q,admitted,cap))
        self._bounds=bounds;self._maximum=maximum;self._archive_digest=sha256(archive).hexdigest()
        self._parameter=Fraction(parameter)
        self._lower=lower;self._upper=upper;self._maps=tuple(retained)
        self._lower_offsets=tuple(self._handle.get().lower_offsets());self._upper_offsets=tuple(self._handle.get().upper_offsets())
        lows=tuple(self._handle.get().occurrence_lower());ups=tuple(self._handle.get().occurrence_upper())
        self._cells=tuple(zip(ups,lows));offsets=[0]
        self._coefficients=_values(self._handle.get().occurrence_coefficients())
        self._activity=tuple(bool(v) for v in self._handle.get().occurrence_activity())
        for i in lows:offsets.append(offsets[-1]+len(lower[i]))
        self._occurrence_offsets=tuple(offsets)
    @property
    def receipt(self):
        self._require()
        return dict(stalk_fields_build_info(),archive_digest=self._archive_digest,
            upper_axes=self._upper,lower_axes=self._lower,flag_maps=self._maps,occurrence_cells=self._cells,
            occurrence_coefficients=self._coefficients,occurrence_activity=self._activity,parameter=self._parameter,
            upper_dimension=self._upper_offsets[-1],lower_dimension=self._lower_offsets[-1],
            retained_occurrences=len(self._cells),preparation_steps=int(self._handle.get().preparation_steps()),
            retained_entries=int(self._handle.get().retained_entries()),limits=self._bounds,
            metrics="declared-unit-stalk-metrics",higher_profile="no-nonempty-upper-grade")
    def apply(self,fields):
        self._require()
        cdef vector[vector[string]] encoded
        cdef vector[string] row
        count=0
        for field in fields:
            if encoded.size()>=self._bounds[0]:raise ValueError("field count exceeds admission")
            row.clear();cells=0
            for i,values in enumerate(field):
                if i>=len(self._upper):raise ValueError("field exceeds original upper cells")
                size=0
                for value in values:
                    size+=1;count+=1
                    if size>len(self._upper[i]):raise ValueError("field exceeds named stalk axes")
                    if count>self._bounds[1]:raise ValueError("field coefficients exceed admission")
                    row.push_back(_scalar(value,self._bounds[3]))
                if size!=len(self._upper[i]):raise ValueError("field must cover named stalk axes")
                cells+=1
            if cells!=len(self._upper):raise ValueError("field must cover original upper cells")
            encoded.push_back(row)
        if encoded.empty():raise ValueError("at least one whole stalk field required")
        cdef StalkFieldResult result
        with nogil:result=self._handle.get().apply(encoded)
        return dict(fields=tuple(_split(_values(v),self._upper_offsets) for v in result.response),
            boundaries=tuple(_split(_values(v),self._lower_offsets) for v in result.boundary),
            occurrence_fields=tuple(_split(_values(v),self._occurrence_offsets) for v in result.occurrences),
            execution_steps=int(result.steps),residual_exact_zero=True,receipt=self.receipt)
    def inject_acquisition(self,archive,mapping,amplitudes):
        self._require()
        from hashlib import sha256
        if not isinstance(archive,bytes):raise TypeError("acquisition requires native source bytes")
        if len(archive)>self._maximum:raise ValueError("acquisition archive exceeds byte admission")
        cdef vector[size_t] directions
        for index in mapping:
            if directions.size()>=self._bounds[0]:raise ValueError("injection dimensions exceed admission")
            directions.push_back(_natural(index))
        cdef vector[string] values
        for value in amplitudes:
            if values.size()>=self._bounds[0]:raise ValueError("acquisition columns exceed admission")
            values.push_back(_scalar(value,self._bounds[3]))
        cdef string image=archive
        cdef size_t cap=self._maximum
        cdef StalkInjectionResult result
        with nogil:result=self._handle.get().inject(image,directions,values,cap)
        return dict(field=_split(_values(result.field),self._upper_offsets),boundary=_values(result.boundary),
            acquisition_archive_digest=sha256(archive).hexdigest(),columns=int(result.columns),
            retained_occurrences=int(result.occurrences),mapping=tuple(directions),amplitudes=tuple(_values(values)),
            execution_steps=int(result.steps),profile="declared-acquisition-boundary-injection-v1")
