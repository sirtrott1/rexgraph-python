# distutils: language=c++
"""Finite compatible sections on complete retained relation carriers."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from ._relation import _natural

cdef extern from "sections_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass SectionChart:
        string name
        size_t relation
        vector[size_t] support
        vector[vector[size_t]] allowed
    cdef cppclass SectionResult:
        vector[vector[size_t]] sections
        vector[size_t] conflict_charts, conflict_cells
        size_t candidate_count, domain_count, admitted_work
        bint satisfiable, subset_minimal, constraint_complete
    cdef cppclass RelationalSections:
        RelationalSections(const string&, const vector[vector[size_t]]&, const vector[SectionChart]&,
                           const vector[size_t]&, size_t, const vector[size_t]&) except +
        SectionResult sections() except +
cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_SECTIONS_BINDING_DIGEST

def sections_build_info():
    return {"engine":"rexgraph-cpp", "engine_version":RFC_ENGINE_VERSION.decode(),
        "engine_headers":RFC_ENGINE_DIGEST.decode(), "binding":RFC_SECTIONS_BINDING_DIGEST.decode(),
        "profile":"owning-finite-relational-sections-v1"}

cdef class PreparedRelationalSections:
    """Pin a native carrier, complete finite domains and closed relation charts."""
    cdef shared_ptr[RelationalSections] _handle
    cdef object _receipt
    cdef bint _loading
    def __init__(self, archive, domains, charts, *, limits=(65536,1000000,64000000), max_bytes=256*1024*1024, enumeration_charts=()):
        if self._loading or self._handle.get()!=NULL:raise RuntimeError("section owner is immutable")
        self._loading=True
        try:self._initialize(archive,domains,charts,limits,max_bytes,enumeration_charts)
        finally:self._loading=False
    cdef void _initialize(self, object archive, object domains, object charts, object limits, object max_bytes, object enumeration_charts) except *:
        from hashlib import sha256
        if not isinstance(archive,bytes):raise TypeError("section carrier requires encoded bytes")
        checked=tuple(_natural(i) for i in limits)
        if len(checked)!=3 or any(not i for i in checked):raise ValueError("three positive section limits required")
        maximum=_natural(max_bytes)
        if not maximum or len(archive)>maximum:raise ValueError("section archive exceeds byte admission")
        cdef vector[vector[size_t]] values
        cdef vector[size_t] row
        cdef vector[SectionChart] declared
        cdef SectionChart chart
        cdef size_t entries=0
        for domain in domains:
            row.clear()
            for value in domain:
                entries+=1
                if entries>checked[1]:raise ValueError("section entries exceed admission")
                row.push_back(_natural(value))
            values.push_back(row)
            entries+=1
            if entries>checked[1]:raise ValueError("section entries exceed admission")
        for item in charts:
            if set(item)!={"name","relation","support","allowed"}:raise ValueError("closed section chart declaration required")
            name=item["name"]
            if not isinstance(name,str) or not name or len(name.encode())>256:raise ValueError("bounded section chart name required")
            chart.name=name.encode();chart.relation=_natural(item["relation"])
            chart.support.clear();chart.allowed.clear()
            entries+=1
            for cell in item["support"]:
                entries+=1
                if entries>checked[1]:raise ValueError("section entries exceed admission")
                chart.support.push_back(_natural(cell))
            for allowed in item["allowed"]:
                entries+=1
                if entries>checked[1]:raise ValueError("section entries exceed admission")
                row.clear()
                for value in allowed:
                    entries+=1
                    if entries>checked[1]:raise ValueError("section entries exceed admission")
                    row.push_back(_natural(value))
                chart.allowed.push_back(row)
            declared.push_back(chart)
            if entries>checked[1]:raise ValueError("section entries exceed admission")
        cdef string raw=archive
        cdef vector[size_t] bounds=checked
        cdef vector[size_t] selectors
        for selector in enumeration_charts:
            if selectors.size()>=checked[1]:raise ValueError("section selectors exceed entry admission")
            selectors.push_back(_natural(selector))
        cdef size_t maximum_bytes=maximum
        with nogil:
            self._handle=shared_ptr[RelationalSections](new RelationalSections(raw,values,declared,bounds,maximum_bytes,selectors))
        self._receipt=dict(sections_build_info(),archive_digest=sha256(archive).hexdigest(),limits=checked,
                           lower_cells=int(values.size()),charts=int(declared.size()),enumeration_charts=tuple(selectors))
    @property
    def receipt(self):
        if self._handle.get()==NULL:raise RuntimeError("section owner is not initialized")
        return dict(self._receipt)
    def sections(self):
        if self._handle.get()==NULL:raise RuntimeError("section owner is not initialized")
        cdef SectionResult result
        with nogil:result=self._handle.get().sections()
        return dict(sections=tuple(tuple(row) for row in result.sections),
            conflict_charts=tuple(result.conflict_charts),conflict_cells=tuple(result.conflict_cells),
            satisfiable=bool(result.satisfiable),subset_minimal=bool(result.subset_minimal),
            constraint_complete=bool(result.constraint_complete),candidate_count=int(result.candidate_count),
            domain_count=int(result.domain_count),admitted_work=int(result.admitted_work),receipt=self.receipt)
