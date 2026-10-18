# distutils: language=c++
"""Owning known products on complete original source input fields."""
from ._field_interface cimport (
    NativeLiteralInput, NativeAccessionInput, NativeInputAccession, PreparedAccessionFieldProgram,
    _bounds, _maximum, _image,
)
from ._field_program import _closed
from ._stalk_fields import _natural

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_KNOWN_TENSOR_GRAPH_BINDING_DIGEST

_LIMITS=(4096,16000000,64000000,16384)

cdef class _Check:
    pass

def known_tensor_graph_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),
            'engine_headers':RFC_ENGINE_DIGEST.decode(),'binding':RFC_KNOWN_TENSOR_GRAPH_BINDING_DIGEST.decode(),
            'profile':'owning-known-source-input-graph-v1','scalar_policy':'gmp-rational-exact'}

cdef _Check _check(value):
    if value is None:return None
    if not callable(value):raise TypeError('native check must be callable')
    cdef _Check state=_Check.__new__(_Check)
    state.callback=value;state.error=None
    return state

cdef int _invoke_check(void* context) noexcept with gil:
    cdef _Check state=<_Check><object>context
    try:state.callback()
    except BaseException as error:
        state.error=error
        return 0
    return 1

cdef GraphCheck _callback(_Check state):
    cdef GraphCheck out
    if state is not None:out.run=_invoke_check;out.context=<void*>state
    return out

cdef void _raise_check(_Check state) except *:
    if state is not None and state.error is not None:raise state.error

cdef GraphDeclaration _declaration(value,vector[size_t] bounds):
    _closed(value,('profile','key','revision','blocks'))
    if value['profile']!='rexgraph.relational-known-tensor-graph.v1':raise ValueError('unknown known tensor graph profile')
    cdef GraphDeclaration out
    cdef GraphBlock block
    cdef size_t count=0
    out.key=_natural(value['key']);out.revision=_natural(value['revision'])
    for row in value['blocks']:
        _closed(row,('cell','source_basis','lifted_basis','left','right','products'))
        grade,cell=row['cell'];block.grade=_natural(grade);block.cell=_natural(cell)
        block.source_basis=_natural(row['source_basis']);block.lifted_basis=_natural(row['lifted_basis'])
        block.left.clear();block.right.clear();block.products.clear()
        count+=1
        if count>bounds[1]:raise ValueError('complete graph declaration admission exceeded')
        for name in row['left']:
            block.left.push_back(_natural(name));count+=1
            if count>bounds[1]:raise ValueError('complete graph declaration admission exceeded')
        for name in row['right']:
            block.right.push_back(_natural(name));count+=1
            if count>bounds[1]:raise ValueError('complete graph declaration admission exceeded')
        for name in row['products']:
            block.products.push_back(_natural(name));count+=1
            if count>bounds[1]:raise ValueError('complete graph declaration admission exceeded')
        out.blocks.push_back(block)
        if out.blocks.size()>bounds[0]:raise ValueError('graph stalk count admission exceeded')
    return out

cdef class NativeKnownTensorGraph:
    cdef shared_ptr[KnownGraphBridge] _pin(self) except *:
        if self._loading or self._handle.get()==NULL:raise RuntimeError('known tensor graph is not initialized')
        return self._handle

    def __init__(self, parent, declaration, *, limits=_LIMITS,max_bytes=256*1024*1024,check=None):
        if self._loading or self._handle.get()!=NULL:raise TypeError('known tensor graph is immutable')
        self._loading=True
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef GraphDeclaration d
        cdef shared_ptr[LiteralInputBridge] literal
        cdef shared_ptr[AccessionInputBridge] input
        cdef NativeLiteralInput literal_parent
        cdef NativeAccessionInput raw_parent
        cdef shared_ptr[KnownGraphBridge] loaded
        cdef _Check state
        cdef GraphCheck callback
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);d=_declaration(declaration,bounds)
            if isinstance(parent,NativeLiteralInput):literal_parent=parent;literal=literal_parent._pin()
            elif isinstance(parent,NativeAccessionInput):raw_parent=parent;input=raw_parent._pin()
            else:raise TypeError('actual native literal or accession input owner required')
            state=_check(check);callback=_callback(state)
            try:
                with nogil:loaded=shared_ptr[KnownGraphBridge](new KnownGraphBridge(literal,input,d,bounds,maximum,callback))
            except BaseException:_raise_check(state);raise
            self._limits=bounds;self._maximum=maximum;self._handle=loaded
        finally:self._loading=False

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS,max_bytes=256*1024*1024,check=None):
        cdef NativeKnownTensorGraph out=cls.__new__(cls)
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef string image
        cdef _Check state
        cdef GraphCheck callback
        cdef shared_ptr[KnownGraphBridge] loaded
        if out._loading or out._handle.get()!=NULL:raise TypeError('native cold owner is immutable')
        out._loading=True
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);image=_image(archive,maximum)
            state=_check(check);callback=_callback(state)
            try:
                with nogil:loaded=KnownGraphBridge.restore(image,bounds,maximum,callback)
            except BaseException:_raise_check(state);raise
            out._limits=bounds;out._maximum=maximum;out._handle=loaded;return out
        finally:out._loading=False

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def archive(self, *, check=None):
        cdef shared_ptr[KnownGraphBridge] pin=self._pin()
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef string image
        try:
            with nogil:image=pin.get().archive(callback)
        except BaseException:_raise_check(state);raise
        return <bytes>image

    def declaration(self, *, max_records=4096):
        cdef shared_ptr[KnownGraphBridge] pin=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef GraphDeclaration d
        with nogil:d=pin.get().declaration(maximum)
        return {'profile':'rexgraph.relational-known-tensor-graph.v1','key':int(d.key),'revision':int(d.revision),
                'blocks':tuple({'cell':(int(b.grade),int(b.cell)),'source_basis':int(b.source_basis),
                    'lifted_basis':int(b.lifted_basis),'left':tuple(b.left),'right':tuple(b.right),'products':tuple(b.products)} for b in d.blocks)}

    def stalks(self, *, max_records=4096):
        cdef shared_ptr[KnownGraphBridge] pin=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef vector[ProgramStalkInfo] values
        with nogil:values=pin.get().stalks(maximum)
        return tuple({'cell':(int(s.grade),int(s.cell)),'basis':int(s.basis),'directions':tuple(s.directions),'metric':<bytes>s.metric} for s in values)

    def lifted_field(self):
        cdef shared_ptr[KnownGraphBridge] pin=self._pin()
        cdef string image
        with nogil:image=pin.get().field()
        return <bytes>image

    def product(self, cell):
        grade,index=cell
        cdef size_t g=_natural(grade),i=_natural(index)
        cdef shared_ptr[KnownGraphBridge] pin=self._pin()
        cdef string image
        with nogil:image=pin.get().product(g,i)
        return <bytes>image

    def matrix(self, kind):
        cdef unsigned int k={'metric':0,'original_metric':1,'projection':2,'zero_section':3}[kind]
        cdef shared_ptr[KnownGraphBridge] pin=self._pin()
        cdef string image
        with nogil:image=pin.get().matrix(k)
        return <bytes>image

    @property
    def retained_entries(self):
        cdef shared_ptr[KnownGraphBridge] pin=self._pin()
        cdef size_t count
        with nogil:count=pin.get().retained_entries()
        return int(count)

    def parent(self):
        cdef shared_ptr[KnownGraphBridge] pin=self._pin()
        cdef NativeAccessionInput out=NativeAccessionInput.__new__(NativeAccessionInput)
        cdef shared_ptr[AccessionInputBridge] loaded
        with nogil:loaded=pin.get().parent()
        out._handle=loaded;return out

    def literal_parent(self):
        cdef shared_ptr[KnownGraphBridge] pin=self._pin()
        cdef shared_ptr[LiteralInputBridge] loaded
        cdef NativeLiteralInput out
        with nogil:loaded=pin.get().literal()
        if loaded.get()==NULL:return None
        out=NativeLiteralInput.__new__(NativeLiteralInput)
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

    def same_owner(self, NativeKnownTensorGraph other not None):
        cdef shared_ptr[KnownGraphBridge] pin=self._pin(),actual=other._pin()
        cdef cppbool same
        with nogil:same=pin.get().same_owner(actual.get()[0])
        return bool(same)

    def require_image(self, field, *, check=None):
        cdef shared_ptr[KnownGraphBridge] pin=self._pin()
        cdef string image=_image(field,self._maximum)
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        try:
            with nogil:pin.get().require_image(image,callback)
        except BaseException:_raise_check(state);raise

    def admit(self, NativeInputAccession target not None, *, check=None):
        cdef shared_ptr[KnownGraphBridge] pin=self._pin()
        cdef shared_ptr[InputAccessionBridge] actual=target._pin()
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef shared_ptr[KnownGraphInputBridge] loaded
        cdef NativeKnownTensorGraphInput out
        try:
            with nogil:loaded=pin.get().admit(actual,callback)
        except BaseException:_raise_check(state);raise
        out=NativeKnownTensorGraphInput.__new__(NativeKnownTensorGraphInput)
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

cdef class NativeKnownTensorGraphInput:
    cdef shared_ptr[KnownGraphInputBridge] _pin(self) except *:
        if self._loading or self._handle.get()==NULL:raise RuntimeError('known graph input is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS,max_bytes=256*1024*1024,check=None):
        cdef NativeKnownTensorGraphInput out=cls.__new__(cls)
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef string image
        cdef _Check state
        cdef GraphCheck callback
        cdef shared_ptr[KnownGraphInputBridge] loaded
        if out._loading or out._handle.get()!=NULL:raise TypeError('native cold owner is immutable')
        out._loading=True
        try:
            bounds=_bounds(limits);maximum=_maximum(max_bytes);image=_image(archive,maximum)
            state=_check(check);callback=_callback(state)
            try:
                with nogil:loaded=KnownGraphInputBridge.restore(image,bounds,maximum,callback)
            except BaseException:_raise_check(state);raise
            out._limits=bounds;out._maximum=maximum;out._handle=loaded;return out
        finally:out._loading=False

    def archive(self, *, check=None):
        cdef shared_ptr[KnownGraphInputBridge] pin=self._pin()
        cdef _Check state=_check(check)
        cdef GraphCheck callback=_callback(state)
        cdef string image
        try:
            with nogil:image=pin.get().archive(callback)
        except BaseException:_raise_check(state);raise
        return <bytes>image

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits),'max_bytes':self._maximum}

    def graph(self):
        cdef shared_ptr[KnownGraphInputBridge] pin=self._pin()
        cdef shared_ptr[KnownGraphBridge] loaded
        cdef NativeKnownTensorGraph out=NativeKnownTensorGraph.__new__(NativeKnownTensorGraph)
        with nogil:loaded=pin.get().graph()
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out

    def input(self):
        cdef shared_ptr[KnownGraphInputBridge] pin=self._pin()
        cdef shared_ptr[AccessionInputBridge] loaded
        cdef NativeAccessionInput out=NativeAccessionInput.__new__(NativeAccessionInput)
        with nogil:loaded=pin.get().input()
        out._handle=loaded;return out

    def program(self):
        cdef shared_ptr[KnownGraphInputBridge] pin=self._pin()
        cdef shared_ptr[FieldProgramBridge] loaded
        cdef PreparedAccessionFieldProgram out=PreparedAccessionFieldProgram.__new__(PreparedAccessionFieldProgram)
        with nogil:loaded=pin.get().program()
        out._limits=self._limits;out._maximum=self._maximum;out._handle=loaded;return out
