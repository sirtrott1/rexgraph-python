# distutils: language=c++
"""Owning rational field responses on retained relation stalks."""
from fractions import Fraction
from numbers import Integral
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t

cdef extern from "relational_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass RelationResponseResult:
        vector[string] gram, response, boundary, contributions, membership_contributions, role_contributions
        size_t fields, steps
    cdef cppclass RelationResponse:
        RelationResponse(const string&, const vector[vector[string]]&, const vector[size_t]&,
                         const string&, const vector[size_t]&, size_t, const vector[vector[string]]&) except +
        size_t relations() except +
        size_t lower_dimension() except +
        size_t occurrences() except +
        size_t preparation_steps() except +
        size_t factor_entries() except +
        RelationResponseResult moments(const vector[vector[string]]&) except +

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_RELATION_BINDING_DIGEST

def relation_build_info():
    return {"engine": "rexgraph-cpp", "engine_version": RFC_ENGINE_VERSION.decode(),
            "engine_headers": RFC_ENGINE_DIGEST.decode(), "binding": RFC_RELATION_BINDING_DIGEST.decode(),
            "profile": "owning-rational-relation-stalk-green-v1"}

def _natural(value):
    if isinstance(value,bool) or not isinstance(value,Integral) or not 0<=value<(1<<64):
        raise ValueError("relation index or bound requires an unsigned integer")
    return int(value)

def _scalar(value, integer_bits):
    if isinstance(value,bool) or not isinstance(value,(Integral,Fraction)):
        raise TypeError("relation field requires integer or Fraction values")
    q=Fraction(value)
    if max(abs(q.numerator).bit_length(),q.denominator.bit_length())>integer_bits:
        raise ValueError("relation scalar exceeds its integer bit admission")
    return (format(q.numerator,"x")+"/"+format(q.denominator,"x")).encode()

def _values(encoded):
    result=[]
    for value in encoded:
        text=value.decode("ascii");parts=text.split("/")
        result.append(Fraction(int(parts[0],16),int(parts[1],16) if len(parts)==2 else 1))
    return tuple(result)

cdef class PreparedRelationResponse:
    """Retain a complete source, incidence maps and one native augmented factor."""
    cdef shared_ptr[RelationResponse] _handle
    cdef object _archive_digest
    cdef object _bounds
    cdef object _transports
    cdef void _require(self) except *:
        if self._handle.get() == NULL:raise RuntimeError("relation response is not initialized")
    def __init__(self, archive, lower_axes, occurrence_axes, *, parameter=Fraction(1),
                 limits=(4096,16000000,64000000,1<<24), max_bytes=256*1024*1024,
                 role_transports=None):
        from hashlib import sha256
        if self._handle.get() != NULL:raise RuntimeError("relation response owner is immutable")
        if not isinstance(archive,bytes):raise TypeError("relation response requires encoded source bytes")
        if len(archive)>_natural(max_bytes):raise ValueError("relation archive exceeds byte limit")
        checked_bounds=tuple(_natural(i) for i in limits)
        if len(checked_bounds)!=4 or any(not i for i in checked_bounds):raise ValueError("four positive action limits required")
        cdef vector[vector[string]] axes
        cdef vector[string] row
        axis_count=0
        for names in lower_axes:
            row.clear()
            for name in names:
                if not isinstance(name,str) or not name or len(name.encode("utf-8"))>256:raise ValueError("bounded named stalk axes required")
                axis_count+=1
                if axis_count>checked_bounds[0]:raise ValueError("lower stalk dimension exceeds admission")
                row.push_back(name.encode("utf-8"))
            axes.push_back(row)
            if axes.size()>checked_bounds[0]:raise ValueError("lower cell count exceeds admission")
        cdef vector[size_t] slots
        for slot in occurrence_axes:
            if slots.size()>=checked_bounds[1]:raise ValueError("occurrence count exceeds admission")
            slots.push_back(_natural(slot))
        cdef vector[size_t] bounds=checked_bounds
        cdef vector[vector[string]] gains
        retained=[];names=set()
        if role_transports is not None:
            for entry in role_transports:
                if len(retained)>=checked_bounds[1]:raise ValueError("role gains exceed entry admission")
                if len(entry)!=3 or not isinstance(entry[0],str) or not entry[0] or len(entry[0].encode())>256 or entry[0] in names:
                    raise ValueError("distinct bounded role names and two exact gains required")
                name,membership,role=entry
                row.clear();row.push_back(name.encode());row.push_back(_scalar(membership,checked_bounds[3]));row.push_back(_scalar(role,checked_bounds[3]))
                if membership<0 or role<0:raise ValueError("nonnegative role gains required")
                gains.push_back(row);names.add(name);retained.append((name,Fraction(membership),Fraction(role)))
        cdef string raw=archive
        cdef string parameter_text=_scalar(parameter,checked_bounds[3])
        cdef size_t maximum=_natural(max_bytes)
        with nogil:
            self._handle=shared_ptr[RelationResponse](new RelationResponse(raw,axes,slots,parameter_text,bounds,maximum,gains))
        self._archive_digest=sha256(archive).hexdigest()
        self._bounds=checked_bounds
        self._transports=None if role_transports is None else tuple(retained)
    @property
    def receipt(self):
        self._require()
        result=dict(relation_build_info(),archive_digest=self._archive_digest,
            relations=int(self._handle.get().relations()),lower_dimension=int(self._handle.get().lower_dimension()),
            retained_occurrences=int(self._handle.get().occurrences()),
            preparation_steps=int(self._handle.get().preparation_steps()),factor_entries=int(self._handle.get().factor_entries()),
            metrics="declared-unit-stalk-metrics",higher_profile="no-nonempty-upper-grade",limits=self._bounds)
        if self._transports is not None:
            result.update(transport_profile="declared-role-gains-v2",role_transports=self._transports)
        return result
    def moments(self, fields):
        self._require()
        cdef vector[vector[string]] values
        cdef vector[string] row
        count=0
        for field in fields:
            if values.size()>=self._bounds[0]:raise ValueError("field count exceeds admission")
            row.clear()
            for value in field:
                if row.size()>=self._handle.get().relations():raise ValueError("field exceeds original relation coordinates")
                count+=1
                if count>self._bounds[1]:raise ValueError("field coefficients exceed admission")
                row.push_back(_scalar(value,self._bounds[3]))
            if row.size()!=self._handle.get().relations():raise ValueError("field must cover original relation coordinates")
            values.push_back(row)
        cdef RelationResponseResult result
        with nogil:result=self._handle.get().moments(values)
        count=int(result.fields);gram=_values(result.gram)
        output={"gram":tuple(gram[i*count:(i+1)*count] for i in range(count)),
                "response":_values(result.response),"boundary":_values(result.boundary),
                "contributions":_values(result.contributions),"execution_steps":int(result.steps),
                "receipt":self.receipt,"residual_exact_zero":True}
        if self._transports is not None:
            output.update(membership_contributions=_values(result.membership_contributions),
                          role_contributions=_values(result.role_contributions))
        return output
