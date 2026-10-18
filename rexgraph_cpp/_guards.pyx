# distutils: language=c++
"""Owning exact guards over complete ordered relational context boundaries."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._relation import _natural

cdef extern from "guard_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass GuardMatch:
        bint matched
        string reason
        vector[vector[size_t]] cells, relations, occurrences
        size_t steps
    cdef cppclass GuardOccurrence:
        size_t slot,lower
        uint64_t key
        bint exists,negative
        string coefficient
        vector[string] roles
    cdef cppclass GuardBoundary:
        string type
        vector[GuardOccurrence] occurrences
    cdef cppclass DictionaryLicense:
        size_t synset
        vector[size_t] left,right
        GuardMatch witness
    cdef cppclass DictionaryPair:
        size_t origin,target
        vector[DictionaryLicense] licenses
    cdef cppclass DictionaryComparison:
        vector[DictionaryPair] pairs
        size_t steps
    cdef cppclass RelationGuards:
        RelationGuards(const string&,const vector[size_t]&,size_t,size_t) except +
        size_t preparation_steps()
        GuardBoundary boundary(size_t) except +
        GuardMatch match(const vector[size_t]&,const vector[size_t]&,const vector[size_t]&,size_t) except +
        GuardMatch encoding_accession(const RelationGuards&,const vector[size_t]&,const vector[size_t]&,size_t) except +
        DictionaryComparison shared_synsets(const vector[vector[size_t]]&,size_t) except +
    cdef cppclass DictionarySection:
        vector[size_t] choices,left,right,fixed
        GuardMatch witness
    cdef cppclass DictionarySections:
        vector[DictionarySection] sections
        size_t steps,domain_count,candidate_count,native_calls,overlap_rejections
    cdef cppclass DictionaryGuards:
        DictionaryGuards(const shared_ptr[RelationGuards]&,const vector[vector[size_t]]&,size_t) except +
        DictionaryComparison comparison() except +
        DictionarySections sections(const vector[size_t]&,const vector[size_t]&,const vector[size_t]&,const vector[size_t]&,size_t,size_t) except +
cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_GUARD_BINDING_DIGEST

cdef extern from "lexical_routes_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass LexicalSpelling:
        LexicalSpelling() except +
        size_t row
        string raw
        vector[size_t] uses, encodings, begins, ends
    cdef cppclass LexicalRoute:
        size_t spelling, owner, entry, sense, synset
        vector[size_t] rows, slots
        vector[uint64_t] occurrence_keys
    cdef cppclass LexicalUse:
        size_t first, last, begin, end
        vector[size_t] routes
    cdef cppclass LexicalLookup:
        vector[LexicalUse] uses
        size_t steps
    cdef cppclass LexicalPattern:
        size_t pattern, synset
    cdef cppclass LexicalRoutes:
        LexicalRoutes(const shared_ptr[RelationGuards]&,const vector[LexicalSpelling]&) except +
        size_t preparation_steps()
        const vector[LexicalRoute]& routes()
        vector[LexicalPattern] bind_patterns(const shared_ptr[RelationGuards]&) except +
        LexicalLookup lookup(const string&,const vector[size_t]&,const vector[size_t]&,size_t,size_t) except +
    cdef cppclass LexicalContextField:
        LexicalLookup lookup
        vector[string] axes, missing
        vector[size_t] entries
        size_t steps
    cdef cppclass LexicalContextFields:
        LexicalContextFields(const shared_ptr[LexicalRoutes]&,const string&,bint,bint,bint,bint,const vector[string]&) except +
        LexicalContextField lookup(const string&,const vector[size_t]&,const vector[size_t]&,const vector[string]&,size_t,size_t) except +

def guards_build_info():
    return {"engine":"rexgraph-cpp", "engine_version":RFC_ENGINE_VERSION.decode(),
        "engine_headers":RFC_ENGINE_DIGEST.decode(), "binding":RFC_GUARD_BINDING_DIGEST.decode(),
        "profile":"owning-ordered-context-guards-v1"}

cdef object _comparison_result(DictionaryComparison result,object receipt):
    return dict(profile="retained-common-synset-v1",execution_steps=int(result.steps),receipt=dict(receipt),
        pairs=tuple(dict(origin=int(p.origin),target=int(p.target),licenses=tuple(dict(synset=int(l.synset),
            source_rows=tuple(l.left),target_rows=tuple(l.right),fixed_lowers=(int(l.synset),),
            cell_map=tuple(tuple(row) for row in l.witness.cells),relation_map=tuple(tuple(row) for row in l.witness.relations),
            occurrence_map=tuple(tuple(row) for row in l.witness.occurrences),execution_steps=int(l.witness.steps),
            strict_morphism=bool(l.witness.matched),chain_map=bool(l.witness.matched),ordered=bool(l.witness.matched))
            for l in p.licenses)) for p in result.pairs))

cdef vector[size_t] _indices(object values,size_t limit) except *:
    cdef vector[size_t] result
    for value in values:
        if result.size()>=limit:raise ValueError("dictionary request exceeds entry admission")
        result.push_back(_natural(value))
    return result

cdef class PreparedRelationGuards:
    """Retain one source endpoint and validate whole context map families."""
    cdef shared_ptr[RelationGuards] _handle
    cdef object _receipt
    cdef bint _loading
    def __init__(self,archive,*,limits=(1000000,64000000,16384),max_bytes=256*1024*1024,max_logical_size=None):
        if self._loading or self._handle.get()!=NULL:raise RuntimeError("guard owner is immutable")
        self._loading=True
        try:self._initialize(archive,limits,max_bytes,max_logical_size)
        finally:self._loading=False
    cdef void _initialize(self,object archive,object limits,object max_bytes,object max_logical_size) except *:
        from hashlib import sha256
        if not isinstance(archive,bytes):raise TypeError("guard source requires encoded bytes")
        checked=tuple(_natural(i) for i in limits)
        if len(checked)!=3 or any(not i for i in checked):raise ValueError("three positive guard limits required")
        maximum=_natural(max_bytes)
        if not maximum or len(archive)>maximum:raise ValueError("guard archive exceeds byte admission")
        cdef string raw=archive
        cdef vector[size_t] bounds=checked
        cdef size_t size=maximum
        cdef size_t logical=0 if max_logical_size is None else _natural(max_logical_size)
        with nogil:self._handle=shared_ptr[RelationGuards](new RelationGuards(raw,bounds,size,logical))
        self._receipt=dict(guards_build_info(),archive_digest=sha256(archive).hexdigest(),limits=checked,
            preparation_steps=self._handle.get().preparation_steps())
        if logical:self._receipt["max_logical_size"]=int(logical)
    @property
    def receipt(self):
        if self._handle.get()==NULL:raise RuntimeError("guard owner is not initialized")
        return dict(self._receipt)
    def match(self,origins,targets,*,fixed_lowers=(),work_limit=None):
        if self._handle.get()==NULL:raise RuntimeError("guard owner is not initialized")
        cdef vector[size_t] a,b,fixed
        for value in origins:
            if a.size()>=self._receipt["limits"][0]:raise ValueError("guard rows exceed entry admission")
            a.push_back(_natural(value))
        for value in targets:
            if b.size()>=self._receipt["limits"][0]:raise ValueError("guard rows exceed entry admission")
            b.push_back(_natural(value))
        for value in fixed_lowers:
            if fixed.size()>=self._receipt["limits"][0]:raise ValueError("guard anchors exceed entry admission")
            fixed.push_back(_natural(value))
        cdef size_t remaining=_natural(self._receipt["limits"][1] if work_limit is None else work_limit)
        cdef GuardMatch result
        with nogil:result=self._handle.get().match(a,b,fixed,remaining)
        return dict(matched=bool(result.matched),reason=result.reason.decode(),
            cell_map=tuple(tuple(row) for row in result.cells),relation_map=tuple(tuple(row) for row in result.relations),
            occurrence_map=tuple(tuple(row) for row in result.occurrences),execution_steps=int(result.steps),
            source_rows=tuple(a),target_rows=tuple(b),fixed_lowers=tuple(fixed),
            strict_morphism=bool(result.matched),chain_map=bool(result.matched),ordered=bool(result.matched),receipt=self.receipt)
    def boundary(self,row):
        if self._handle.get()==NULL:raise RuntimeError("guard owner is not initialized")
        cdef size_t index=_natural(row)
        cdef GuardBoundary result
        with nogil:result=self._handle.get().boundary(index)
        return dict(type=result.type.decode(),occurrences=tuple(dict(slot=int(o.slot),lower=int(o.lower),key=int(o.key),
            exists=bool(o.exists),negative=bool(o.negative),coefficient=o.coefficient.decode(),
            roles=tuple(role.decode() for role in o.roles)) for o in result.occurrences))
    def encoding_accession(self,PreparedRelationGuards target,origins,targets,*,work_limit=None):
        """Map complete closed encoding carriers between two explicit owners.

        The map preserves ordered unit encoding flags and original shared
        boundaries. It is a linear chain accession across named encoding roles,
        rather than a strict morphism of the owners' entire type signatures.
        """
        if self._handle.get()==NULL or target is None or target._handle.get()==NULL:
            raise RuntimeError("encoding accession requires two initialized owners")
        cdef vector[size_t] a=_indices(origins,128),b=_indices(targets,128)
        cdef size_t remaining=_natural(self._receipt["limits"][1] if work_limit is None else work_limit)
        cdef GuardMatch result
        with nogil:result=self._handle.get().encoding_accession(target._handle.get()[0],a,b,remaining)
        return dict(profile="closed-encoding-chain-accession-v1",matched=bool(result.matched),reason=result.reason.decode(),
            cell_map=tuple(tuple(row) for row in result.cells),relation_map=tuple(tuple(row) for row in result.relations),
            occurrence_map=tuple(tuple(row) for row in result.occurrences),execution_steps=int(result.steps),
            source_rows=tuple(a),target_rows=tuple(b),ordered=bool(result.matched),chain_map=bool(result.matched),
            strict_morphism=False,source_receipt=self.receipt,target_receipt=target.receipt)
    def shared_synsets(self,pairs,*,work_limit=None):
        """Compare original record columns through every common retained synset."""
        if self._handle.get()==NULL:raise RuntimeError("guard owner is not initialized")
        cdef vector[vector[size_t]] checked
        cdef vector[size_t] pair
        for values in pairs:
            if checked.size()>=self._receipt["limits"][0]:raise ValueError("dictionary pairs exceed entry admission")
            pair.clear()
            for value in values:
                if pair.size()>=2:raise ValueError("dictionary pair requires two cells")
                pair.push_back(_natural(value))
            checked.push_back(pair)
        cdef size_t remaining=_natural(self._receipt["limits"][1] if work_limit is None else work_limit)
        cdef DictionaryComparison result
        with nogil:result=self._handle.get().shared_synsets(checked,remaining)
        return _comparison_result(result,self.receipt)
    def prepare_dictionary(self,pairs,*,work_limit=None):
        """Prepare one owning catalog of original dictionary route alternatives."""
        if self._handle.get()==NULL:raise RuntimeError("guard owner is not initialized")
        cdef vector[vector[size_t]] checked
        for pair in pairs:
            if checked.size()>=self._receipt["limits"][0]:raise ValueError("dictionary pairs exceed entry admission")
            checked.push_back(_indices(pair,2))
        cdef size_t remaining=_natural(self._receipt["limits"][1] if work_limit is None else work_limit)
        cdef PreparedDictionaryGuards result=PreparedDictionaryGuards.__new__(PreparedDictionaryGuards)
        with nogil:result._handle=shared_ptr[DictionaryGuards](new DictionaryGuards(self._handle,checked,remaining))
        result._receipt=self.receipt
        return result

cdef class PreparedDictionaryGuards:
    """Own retained dictionary route domains and certify their whole union maps."""
    cdef shared_ptr[DictionaryGuards] _handle
    cdef object _receipt
    def __init__(self,*args,**kwargs):
        raise RuntimeError("dictionary catalogs require prepare_dictionary")
    @property
    def comparison(self):
        if self._handle.get()==NULL:raise RuntimeError("dictionary catalog is not initialized")
        cdef DictionaryComparison result
        with nogil:result=self._handle.get().comparison()
        return _comparison_result(result,self._receipt)
    @property
    def receipt(self):
        if self._handle.get()==NULL:raise RuntimeError("dictionary catalog is not initialized")
        return dict(self._receipt)
    def sections(self,indices,*,source_rows=(),target_rows=(),fixed_lowers=(),candidate_limit=4096,work_limit=None):
        """Retain all dictionary choices that glue with one complete seed map."""
        if self._handle.get()==NULL:raise RuntimeError("dictionary catalog is not initialized")
        cdef size_t limit=self._receipt["limits"][0]
        cdef vector[size_t] selected=_indices(indices,limit)
        cdef vector[size_t] origins=_indices(source_rows,limit)
        cdef vector[size_t] targets=_indices(target_rows,limit)
        cdef vector[size_t] fixed=_indices(fixed_lowers,limit)
        cdef size_t candidates=_natural(candidate_limit)
        cdef size_t remaining=_natural(self._receipt["limits"][1] if work_limit is None else work_limit)
        cdef DictionarySections result
        with nogil:result=self._handle.get().sections(selected,origins,targets,fixed,candidates,remaining)
        return dict(profile="retained-dictionary-union-maps-v1",pair_indices=tuple(selected),
            source_rows=tuple(origins),target_rows=tuple(targets),fixed_lowers=tuple(fixed),
            candidate_limit=int(candidates),domain_count=int(result.domain_count),candidate_count=int(result.candidate_count),
            native_calls=int(result.native_calls),overlap_rejections=int(result.overlap_rejections),
            execution_steps=int(result.steps),complete=True,receipt=self.receipt,
            sections=tuple(dict(choices=tuple(s.choices),source_rows=tuple(s.left),target_rows=tuple(s.right),
                fixed_lowers=tuple(s.fixed),cell_map=tuple(tuple(row) for row in s.witness.cells),
                relation_map=tuple(tuple(row) for row in s.witness.relations),
                occurrence_map=tuple(tuple(row) for row in s.witness.occurrences),execution_steps=int(s.witness.steps),
                strict_morphism=bool(s.witness.matched),chain_map=bool(s.witness.matched),ordered=bool(s.witness.matched))
                for s in result.sections))

cdef class PreparedLexicalRoutes:
    """Own exact spelling accessions and all complete original lexical paths.

    Spelling bytes are supplied accessions. The linguistic source adapter
    verifies them against retained byte fields and complete token maps.
    """
    cdef shared_ptr[LexicalRoutes] _handle
    cdef object _receipt
    cdef bint _loading
    def __init__(self, PreparedRelationGuards owner, spellings):
        if self._loading or self._handle.get()!=NULL:raise RuntimeError("lexical catalog is immutable")
        self._loading=True
        try:self._initialize(owner,spellings)
        finally:self._loading=False
    cdef void _initialize(self,PreparedRelationGuards owner,object spellings) except *:
        from hashlib import sha256
        from struct import pack
        if owner is None or owner._handle.get()==NULL:raise RuntimeError("guard owner is not initialized")
        cdef vector[LexicalSpelling] declarations
        cdef LexicalSpelling item
        cdef size_t limit=owner._receipt["limits"][0]
        cdef size_t retained=0, required
        digest=sha256()
        for spelling in spellings:
            if declarations.size()>=limit:raise ValueError("lexical spelling entry admission exceeded")
            if set(spelling)!={"row","raw","uses","encoding_rows","spans"}:
                raise ValueError("closed original lexical spelling accession required")
            raw=spelling["raw"]
            if not isinstance(raw,bytes):raise TypeError("lexical spelling requires bytes")
            if not 1<=len(raw)<=8192:raise ValueError("bounded lexical spelling bytes required")
            item=LexicalSpelling();item.row=_natural(spelling["row"]);item.raw=raw
            item.uses=_indices(spelling["uses"],128);item.encodings=_indices(spelling["encoding_rows"],128)
            for span in spelling["spans"]:
                if item.begins.size()>=128 or len(span)!=2:raise ValueError("bounded lexical spelling intervals required")
                item.begins.push_back(_natural(span[0]));item.ends.push_back(_natural(span[1]))
            required=len(raw)+item.uses.size()+item.encodings.size()+item.begins.size()+item.ends.size()+6
            if required>limit-retained:raise ValueError("lexical spelling entry admission exceeded")
            retained+=required
            digest.update(pack("<QQ",item.row,len(raw)));digest.update(raw)
            for indices in (tuple(item.uses),tuple(item.encodings),tuple(item.begins),tuple(item.ends)):
                digest.update(pack("<Q",len(indices)))
                for index in indices:digest.update(pack("<Q",index))
            declarations.push_back(item)
        with nogil:self._handle=shared_ptr[LexicalRoutes](new LexicalRoutes(owner._handle,declarations))
        self._receipt=dict(owner.receipt,profile="original-lexical-byte-routes-v1",
            spelling_digest=digest.hexdigest(),spelling_count=int(declarations.size()),
            route_count=int(self._handle.get().routes().size()),
            catalog_preparation_steps=int(self._handle.get().preparation_steps()))
    @property
    def receipt(self):
        if self._handle.get()==NULL:raise RuntimeError("lexical catalog is not initialized")
        return dict(self._receipt)
    @property
    def routes(self):
        if self._handle.get()==NULL:raise RuntimeError("lexical catalog is not initialized")
        return tuple(dict(spelling=int(r.spelling),owner=int(r.owner),entry=int(r.entry),sense=int(r.sense),
            synset=int(r.synset),source_rows=tuple(r.rows),occurrence_slots=tuple(r.slots),
            occurrence_keys=tuple(r.occurrence_keys)) for r in self._handle.get().routes())
    def lookup(self,raw,spans,*,candidate_limit=4096,work_limit=None):
        """Match all exact spelling routes at the declared original interval cuts."""
        if self._handle.get()==NULL:raise RuntimeError("lexical catalog is not initialized")
        if not isinstance(raw,bytes):raise TypeError("lexical query requires bytes")
        if not 1<=len(raw)<=8192:raise ValueError("bounded lexical query bytes required")
        cdef vector[size_t] begins,ends
        for span in spans:
            if begins.size()>=128 or len(span)!=2:raise ValueError("bounded lexical query intervals required")
            begins.push_back(_natural(span[0]));ends.push_back(_natural(span[1]))
        cdef string data=raw
        cdef size_t candidates=_natural(candidate_limit)
        cdef size_t work=_natural(self._receipt["limits"][1] if work_limit is None else work_limit)
        cdef LexicalLookup result
        with nogil:result=self._handle.get().lookup(data,begins,ends,candidates,work)
        return dict(profile="original-lexical-byte-use-routes-v1",receipt=self.receipt,
            execution_steps=int(result.steps),complete=True,
            uses=tuple(dict(first=int(u.first),last=int(u.last),span=(int(u.begin),int(u.end)),
                routes=tuple(u.routes)) for u in result.uses))
    def bind_patterns(self,PreparedRelationGuards owner):
        """Verify original primary identities and complete native pattern stars."""
        if self._handle.get()==NULL:raise RuntimeError("lexical catalog is not initialized")
        if owner is None or owner._handle.get()==NULL:raise RuntimeError("pattern guard owner is not initialized")
        cdef vector[LexicalPattern] result
        with nogil:result=self._handle.get().bind_patterns(owner._handle)
        return tuple((int(item.pattern),int(item.synset)) for item in result)


cdef class PreparedLexicalContextFields:
    """Own a lexical catalog and its complete named sparse context frame."""
    cdef shared_ptr[LexicalContextFields] _handle
    cdef object _receipt, _axes
    cdef bint _bound, _loading, _annotations
    def __init__(self, PreparedLexicalRoutes catalog, dictionary_digest, *,
                 lexical_kinds=False, phrases=True, observed_upos=False, frame_axes=None):
        if self._loading or self._handle.get()!=NULL:
            raise RuntimeError("lexical context frame is immutable")
        self._loading=True
        try:self._initialize(catalog,dictionary_digest,lexical_kinds,phrases,observed_upos,frame_axes)
        finally:self._loading=False
    cdef void _initialize(self,PreparedLexicalRoutes catalog,object digest,object kinds,object phrases,object annotations,object axes) except *:
        if catalog is None or catalog._handle.get()==NULL:
            raise RuntimeError("lexical catalog is not initialized")
        if type(digest) is not str or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError("canonical dictionary archive digest required")
        if digest!=catalog._receipt['archive_digest']:
            raise ValueError("context field digest differs from its dictionary owner")
        if type(kinds) is not bool or type(phrases) is not bool or type(annotations) is not bool:
            raise TypeError("boolean lexical field selections required")
        cdef vector[string] frame
        cdef string fingerprint=digest.encode('ascii')
        cdef bint use_kinds=kinds, use_phrases=phrases, use_annotations=annotations, bound=axes is not None
        cdef size_t limit=catalog._receipt['limits'][0], retained=0, required
        if bound:
            for axis in axes:
                if frame.size()>=limit or type(axis) is not str or not axis or len(axis.encode('utf-8'))>256:
                    raise ValueError("bounded named lexical frame directions required")
                encoded=axis.encode('utf-8');required=2+(len(encoded)+7)//8
                if required>limit-retained:
                    raise ValueError("lexical frame entry admission exceeded")
                retained+=required;frame.push_back(encoded)
        with nogil:
            self._handle=shared_ptr[LexicalContextFields](new LexicalContextFields(
                catalog._handle,fingerprint,use_kinds,use_phrases,use_annotations,bound,frame))
        self._bound=bound
        self._annotations=use_annotations
        self._axes=tuple(axis.decode('utf-8') for axis in frame)
        self._receipt=dict(catalog.receipt,profile='original-lexical-context-fields-v1',
            dictionary_digest=digest,lexical_kinds=kinds,phrases=phrases,
            frame_axes=self._axes if bound else None)
        if self._annotations:self._receipt.update(profile='original-relational-past-fields-v1',observed_upos=True)
    @property
    def receipt(self):
        if self._handle.get()==NULL:raise RuntimeError("lexical context frame is not initialized")
        return dict(self._receipt)
    def lookup(self,raw,spans,*,annotations=None,candidate_limit=4096,work_limit=None):
        """Construct token and phrase directions in one native bounded call."""
        if self._handle.get()==NULL:raise RuntimeError("lexical context frame is not initialized")
        if not isinstance(raw,bytes):raise TypeError("original query bytes required")
        if not 1<=len(raw)<=8192:raise ValueError("bounded original query bytes required")
        raw.decode('utf-8')
        cdef vector[size_t] begins,ends
        for span in spans:
            if begins.size()>=128 or len(span)!=2:raise ValueError("bounded original query intervals required")
            start,end=span
            if type(start) is not int or type(end) is not int or not 0<=start<end<=len(raw):
                raise ValueError("original UTF8 query interval required")
            raw[start:end].decode('utf-8')
            begins.push_back(_natural(start));ends.push_back(_natural(end))
        cdef string data=raw
        cdef vector[string] labels
        if self._annotations:
            if not isinstance(annotations,(tuple,list)) or len(annotations)!=begins.size():
                raise ValueError('complete declared original past annotations required')
            for name in annotations:
                if type(name) is not str or not 1<=len(name)<=6 or not name.isascii():
                    raise ValueError('declared original UPOS annotation required')
                labels.push_back(name.encode('ascii'))
        elif annotations is not None:
            raise ValueError('past annotation fields are not declared by this owner')
        cdef size_t candidates=_natural(candidate_limit)
        cdef size_t work=_natural(self._receipt['limits'][1] if work_limit is None else work_limit)
        cdef LexicalContextField result
        with nogil:result=self._handle.get().lookup(data,begins,ends,labels,candidates,work)
        report=dict(profile='original-lexical-context-field-v1',receipt=self.receipt,
            execution_steps=int(result.steps),axes=tuple(axis.decode('utf-8') for axis in result.axes),
            unmapped_axes=tuple(axis.decode('utf-8') for axis in result.missing),
            frame_size=len(self._axes) if self._bound else None,
            entries=tuple((int(i),1) for i in result.entries) if self._bound and result.missing.empty() else None,
            status='unsupported-input' if not result.missing.empty() else 'complete' if self._bound else 'unbound',
            raw=raw,spans=tuple(zip(begins,ends)),
            uses=tuple(dict(first=int(u.first),last=int(u.last),span=(int(u.begin),int(u.end)),
                routes=tuple(u.routes)) for u in result.lookup.uses))
        if self._annotations:report.update(profile='original-relational-past-field-v1',annotations=tuple(annotations))
        return report
