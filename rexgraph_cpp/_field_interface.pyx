# distutils: language=c++
"""Actual source use owners, native faithful accessions and compatible readings."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from hashlib import sha256
from ._stalk_fields import _natural, _scalar, _values
from ._field_program import _closed

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_FIELD_INTERFACE_BINDING_DIGEST

_LIMITS = (4096, 16000000, 64000000, 16384)
_STATUSES = ('derived', 'observed', 'prospective')


def field_interface_build_info():
    return {'engine': 'rexgraph-cpp', 'engine_version': RFC_ENGINE_VERSION.decode(),
            'engine_headers': RFC_ENGINE_DIGEST.decode(), 'binding': RFC_FIELD_INTERFACE_BINDING_DIGEST.decode(),
            'profile': 'owning-relational-field-interface-v1', 'scalar_policy': 'gmp-rational-exact'}


cdef vector[size_t] _bounds(value):
    cdef vector[size_t] out
    values = tuple(value)
    if len(values) != 4 or any(not _natural(v) for v in values):
        raise ValueError('four positive native action bounds required')
    for v in values: out.push_back(_natural(v))
    return out


cdef size_t _maximum(value):
    cdef size_t out = _natural(value)
    if not out: raise ValueError('positive native archive byte admission required')
    return out


cdef string _image(value, size_t maximum):
    if type(value) is not bytes: raise TypeError('immutable canonical native bytes required')
    if len(value) > maximum: raise ValueError('native image exceeds byte admission')
    return value


cdef ProgramMatrix _matrix(value, size_t bits):
    _closed(value, ('shape', 'entries'))
    cdef ProgramMatrix out
    cdef ProgramEntry entry
    shape = tuple(value['shape'])
    if len(shape) != 2: raise ValueError('explicit rectangular map shape required')
    out.rows = _natural(shape[0]); out.columns = _natural(shape[1])
    for row, column, coefficient in value['entries']:
        entry.row = _natural(row); entry.column = _natural(column); entry.value = _scalar(coefficient, bits)
        out.entries.push_back(entry)
    return out


cdef ProgramStalk _stalk_declaration(value, size_t bits):
    _closed(value, ('cell', 'basis', 'directions', 'metric'))
    cdef ProgramStalk out
    grade, cell = value['cell']
    out.grade = _natural(grade); out.cell = _natural(cell); out.basis = _natural(value['basis'])
    for key in value['directions']: out.directions.push_back(_natural(key))
    out.metric = _matrix(value['metric'], bits)
    return out


cdef InterfaceAccession _accession_declaration(value, size_t bits):
    _closed(value, ('profile', 'key', 'node', 'grade', 'stalks', 'instances', 'accession', 'reconstruction', 'isometric'))
    if value['profile'] != 'rexgraph.relational-input-accession.v1': raise ValueError('unknown input accession profile')
    if type(value['isometric']) is not bool: raise TypeError('explicit boolean metric contract required')
    _closed(value['instances'], ('cells', 'occurrences'))
    cdef InterfaceAccession out
    out.key = _natural(value['key']); out.node = _natural(value['node']); out.grade = _natural(value['grade'])
    out.isometric = value['isometric']
    for item in value['stalks']: out.stalks.push_back(_stalk_declaration(item, bits))
    out.cells = _indices(value['instances']['cells']); out.occurrences = _indices(value['instances']['occurrences'])
    out.accession = _matrix(value['accession'], bits); out.reconstruction = _matrix(value['reconstruction'], bits)
    return out


cdef vector[vector[size_t]] _indices(value):
    cdef vector[vector[size_t]] out
    cdef vector[size_t] row
    for values in value:
        row.clear()
        for item in values: row.push_back(_natural(item))
        out.push_back(row)
    return out


cdef object _publication(ProgramSourceInfo source):
    return {'namespace': int(source.name_space), 'sequence': int(source.sequence),
            'semantic_scale': int(source.semantic_scale), 'endpoint_digest': sha256(<bytes>source.endpoint).hexdigest(),
            'coordinates': tuple(tuple((<bytes>v).decode() for v in row) for row in source.coordinates)}


cdef object _stalk_info(ProgramStalkInfo value):
    return {'cell': (int(value.grade), int(value.cell)), 'basis': int(value.basis),
            'directions': tuple(int(k) for k in value.directions), 'metric': <bytes>value.metric}


cdef object _input_info(InterfaceInputInfo value):
    return {'key': int(value.key), 'node': int(value.node), 'use': int(value.use), 'grade': int(value.grade),
            'isometric': bool(value.isometric), 'asset': <bytes>value.asset, 'field': <bytes>value.field,
            'represented': <bytes>value.represented, 'accession': <bytes>value.accession,
            'reconstruction': <bytes>value.reconstruction, 'source_metric': <bytes>value.source_metric,
            'target_metric': <bytes>value.target_metric, 'stalks': tuple(_stalk_info(s) for s in value.stalks),
            'instances': {'cells': tuple(tuple(int(i) for i in r) for r in value.cells),
                          'occurrences': tuple(tuple(int(i) for i in r) for r in value.occurrences)}}


cdef InterfaceView _view(value, size_t bits):
    _closed(value, ('profile', 'key', 'basis', 'directions', 'accession', 'reconstruction', 'physical'))
    if value['profile'] != 'rexgraph.relational-field-view.v1': raise ValueError('unknown field view profile')
    cdef InterfaceView out
    out.key = _natural(value['key']); out.basis = _natural(value['basis'])
    for key in value['directions']: out.directions.push_back(_natural(key))
    out.accession = _matrix(value['accession'], bits); out.reconstruction = _matrix(value['reconstruction'], bits)
    out.has_physical = value['physical'] is not None
    if out.has_physical: out.physical = _matrix(value['physical'], bits)
    return out


cdef object _selected(ProgramSelected value):
    out = _stalk_info(value.stalk)
    out.update(field=<bytes>value.field, values=_values(value.values))
    return out




cdef class PreparedAccessionFieldProgram:
    cdef shared_ptr[FieldProgramBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native accession program is not initialized')
        return self._handle

    def __init__(self, program_archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        if self._loading or self._handle.get() != NULL: raise TypeError('native program owners are immutable')
        self._loading = True
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef string image
        cdef shared_ptr[FieldProgramBridge] handle
        try:
            bounds = _bounds(limits); maximum = _maximum(max_bytes); image = _image(program_archive, maximum)
            with nogil: handle = FieldProgramBridge.restore(image, bounds, maximum)
            self._handle = handle; self._limits = bounds; self._maximum = maximum
        finally: self._loading = False

    def archive(self):
        cdef shared_ptr[FieldProgramBridge] handle = self._pin()
        cdef string result
        with nogil: result = handle.get().archive()
        return <bytes>result

    def identity(self):
        cdef shared_ptr[FieldProgramBridge] handle = self._pin()
        cdef uint64_t key, revision
        with nogil: key = handle.get().key(); revision = handle.get().revision()
        return {'key': int(key), 'revision': int(revision)}

    def publication(self):
        cdef shared_ptr[FieldProgramBridge] handle = self._pin()
        cdef ProgramSourceInfo info
        with nogil: info = handle.get().source()
        return _publication(info)

    def execute(self, inputs):
        cdef shared_ptr[FieldProgramBridge] handle = self._pin()
        cdef vector[shared_ptr[AccessionInputBridge]] values
        cdef NativeAccessionInput item
        cdef NativeAccessionFieldResult out = NativeAccessionFieldResult.__new__(NativeAccessionFieldResult)
        for value in inputs:
            if not isinstance(value, NativeAccessionInput): raise TypeError('actual admitted native accession inputs required')
            item = value; values.push_back(item._pin())
            if values.size() > self._limits[1]: raise ValueError('input count exceeds admission')
        with nogil: out._handle = AccessionResultBridge.execute(handle, values, self._limits, self._maximum)
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeInputUseAsset:
    cdef shared_ptr[InputUseAssetBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native input asset is not initialized')
        return self._handle

    def __init__(self, source_archive, declaration, *, limits=_LIMITS, max_bytes=256*1024*1024):
        if self._loading or self._handle.get() != NULL: raise TypeError('native input asset owners are immutable')
        self._loading = True
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef string source, raw, encoding
        cdef uint64_t key, channel
        cdef vector[InterfaceUse] uses
        cdef InterfaceUse use
        cdef vector[uint64_t] pair
        cdef shared_ptr[InputUseAssetBridge] handle
        try:
            _closed(declaration, ('profile', 'key', 'channel', 'encoding', 'bytes', 'uses'))
            if declaration['profile'] != 'rexgraph.relational-input-use-asset.v1': raise ValueError('unknown input asset profile')
            if not isinstance(declaration['encoding'], str): raise TypeError('retained encoding label must be text')
            bounds = _bounds(limits); maximum = _maximum(max_bytes)
            source = _image(source_archive, maximum); raw = _image(declaration['bytes'], maximum)
            encoding = declaration['encoding'].encode(); key = _natural(declaration['key']); channel = _natural(declaration['channel'])
            for value in declaration['uses']:
                _closed(value, ('key', 'cells', 'occurrences', 'spans'))
                use.key = _natural(value['key']); use.cells.clear(); use.occurrences.clear()
                for grade, identity in value['cells']:
                    pair.clear(); pair.push_back(_natural(grade)); pair.push_back(_natural(identity)); use.cells.push_back(pair)
                for grade, identity in value['occurrences']:
                    pair.clear(); pair.push_back(_natural(grade)); pair.push_back(_natural(identity)); use.occurrences.push_back(pair)
                use.spans = _indices(value['spans']); uses.push_back(use)
                if uses.size() > bounds[1]: raise ValueError('input use count exceeds admission')
            with nogil: handle = shared_ptr[InputUseAssetBridge](new InputUseAssetBridge(source, key, channel, encoding, raw, uses, bounds, maximum))
            self._handle = handle; self._limits = bounds; self._maximum = maximum
        finally: self._loading = False

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef NativeInputUseAsset out = cls.__new__(cls)
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string image = _image(archive, maximum)
        with nogil: out._handle = InputUseAssetBridge.restore(image, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[InputUseAssetBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().archive()
        return <bytes>image

    def source_archive(self):
        cdef shared_ptr[InputUseAssetBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().source_archive()
        return <bytes>image

    def inspect(self):
        cdef shared_ptr[InputUseAssetBridge] handle = self._pin()
        cdef InterfaceAssetInfo value
        with nogil: value = handle.get().info()
        return {'key': int(value.key), 'channel': int(value.channel), 'encoding': (<bytes>value.encoding).decode(),
                'byte_count': int(value.byte_count), 'use_count': int(value.use_count), 'source': _publication(value.source)}

    def uses(self, *, max_records=256):
        cdef shared_ptr[InputUseAssetBridge] handle = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef vector[InterfaceUse] values
        with nogil: values = handle.get().uses(maximum)
        return tuple({'key': int(v.key), 'cells': tuple(tuple(int(i) for i in row) for row in v.cells),
                      'occurrences': tuple(tuple(int(i) for i in row) for row in v.occurrences),
                      'spans': tuple(tuple(int(i) for i in row) for row in v.spans)} for v in values)

    def recover(self, use, *, max_bytes=4096):
        cdef shared_ptr[InputUseAssetBridge] handle = self._pin()
        cdef uint64_t key = _natural(use)
        cdef size_t maximum = _natural(max_bytes)
        cdef vector[string] values
        with nogil: values = handle.get().recover(key, maximum)
        return tuple(<bytes>v for v in values)

    def same_owner(self, NativeInputUseAsset other not None):
        cdef shared_ptr[InputUseAssetBridge] left = self._pin(), right = other._pin()
        cdef cppbool result
        with nogil: result = left.get().same_owner(right.get()[0])
        return bool(result)


cdef class NativeInputAccession:
    cdef shared_ptr[InputAccessionBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native input accession is not initialized')
        return self._handle

    def __init__(self, NativeInputUseAsset asset not None, PreparedAccessionFieldProgram program not None,
                 declaration, *, limits=_LIMITS, max_bytes=256*1024*1024):
        if self._loading or self._handle.get() != NULL: raise TypeError('native input accessions are immutable')
        self._loading = True
        cdef shared_ptr[InputUseAssetBridge] original
        cdef shared_ptr[FieldProgramBridge] target
        cdef shared_ptr[InputAccessionBridge] handle
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef InterfaceAccession value
        try:
            bounds = _bounds(limits); maximum = _maximum(max_bytes); original = asset._pin(); target = program._pin()
            value = _accession_declaration(declaration, bounds[3])
            with nogil: handle = shared_ptr[InputAccessionBridge](new InputAccessionBridge(original, target, value, bounds, maximum))
            self._handle = handle; self._limits = bounds; self._maximum = maximum
        finally: self._loading = False

    def admit(self, use, field):
        cdef shared_ptr[InputAccessionBridge] handle = self._pin()
        cdef uint64_t key = _natural(use)
        cdef string image = _image(field, self._maximum)
        cdef NativeAccessionInput out = NativeAccessionInput.__new__(NativeAccessionInput)
        with nogil: out._handle = handle.get().admit(key, image)
        return out


cdef class NativeAccessionInput:
    cdef shared_ptr[AccessionInputBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native accession input is not initialized')
        return self._handle

    def inspect(self):
        cdef shared_ptr[AccessionInputBridge] handle = self._pin()
        cdef InterfaceInputInfo info
        with nogil: info = handle.get().info()
        return _input_info(info)


cdef class NativeLiteralInput:
    cdef shared_ptr[LiteralInputBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native literal input is not initialized')
        return self._handle

    def __init__(self, NativeInputAccession accession not None, declaration, *,
                 limits=_LIMITS, max_bytes=256*1024*1024):
        if self._loading or self._handle.get() != NULL: raise TypeError('native literal input owners are immutable')
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef LiteralDeclaration value
        cdef LiteralCoordinate coordinate
        cdef shared_ptr[InputAccessionBridge] parent = accession._pin()
        cdef shared_ptr[LiteralInputBridge] handle
        kinds = ('byte-equal', 'byte-prefix', 'byte-suffix', 'byte-punctuation')
        _closed(declaration, ('profile', 'key', 'revision', 'use', 'coordinates'))
        if declaration['profile'] != 'rexgraph.relational-literal-input.v1': raise ValueError('unknown literal input profile')
        value.key = _natural(declaration['key']); value.revision = _natural(declaration['revision']); value.use = _natural(declaration['use'])
        for row in declaration['coordinates']:
            _closed(row, ('key', 'cell', 'direction', 'use', 'span_index', 'span', 'kind', 'operand'))
            if len(row['cell']) != 2 or len(row['span']) != 2: raise ValueError('original cell and exact byte span required')
            if row['kind'] not in kinds: raise ValueError('unknown native byte predicate')
            coordinate.key = _natural(row['key']); coordinate.grade = _natural(row['cell'][0]); coordinate.cell = _natural(row['cell'][1])
            coordinate.direction = _natural(row['direction']); coordinate.use = _natural(row['use'])
            coordinate.span_index = _natural(row['span_index']); coordinate.begin = _natural(row['span'][0]); coordinate.end = _natural(row['span'][1])
            coordinate.kind = kinds.index(row['kind']); coordinate.operand = _image(row['operand'], maximum)
            value.coordinates.push_back(coordinate)
            if value.coordinates.size() > bounds[1]: raise ValueError('literal coordinate declaration exceeds admission')
        self._loading = True
        try:
            with nogil: handle = shared_ptr[LiteralInputBridge](new LiteralInputBridge(parent, value, bounds, maximum))
            self._handle = handle; self._limits = bounds; self._maximum = maximum
        finally: self._loading = False

    @classmethod
    def from_archive(cls, image, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef NativeLiteralInput out = cls.__new__(cls)
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string raw = _image(image, maximum)
        with nogil: out._handle = LiteralInputBridge.restore(raw, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[LiteralInputBridge] owner = self._pin()
        cdef string raw
        with nogil: raw = owner.get().archive()
        return <bytes>raw

    def declaration(self, *, max_records=4096):
        cdef shared_ptr[LiteralInputBridge] owner = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef LiteralDeclaration value
        kinds = ('byte-equal', 'byte-prefix', 'byte-suffix', 'byte-punctuation')
        with nogil: value = owner.get().info(maximum)
        return {'profile': 'rexgraph.relational-literal-input.v1', 'key': int(value.key),
                'revision': int(value.revision), 'use': int(value.use),
                'coordinates': tuple({'key': int(c.key), 'cell': (int(c.grade), int(c.cell)),
                    'direction': int(c.direction), 'use': int(c.use), 'span_index': int(c.span_index),
                    'span': (int(c.begin), int(c.end)), 'kind': kinds[c.kind], 'operand': <bytes>c.operand}
                    for c in value.coordinates)}

    def input(self):
        cdef shared_ptr[LiteralInputBridge] owner = self._pin()
        cdef NativeAccessionInput out = NativeAccessionInput.__new__(NativeAccessionInput)
        with nogil: out._handle = owner.get().input()
        return out

    def asset(self):
        cdef shared_ptr[LiteralInputBridge] owner = self._pin()
        cdef NativeInputUseAsset out = NativeInputUseAsset.__new__(NativeInputUseAsset)
        with nogil: out._handle = owner.get().asset()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def program(self):
        cdef shared_ptr[LiteralInputBridge] owner = self._pin()
        cdef PreparedAccessionFieldProgram out = PreparedAccessionFieldProgram.__new__(PreparedAccessionFieldProgram)
        with nogil: out._handle = owner.get().program()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def same_owner(self, NativeLiteralInput other not None):
        cdef shared_ptr[LiteralInputBridge] left = self._pin(), right = other._pin()
        cdef cppbool same
        with nogil: same = left.get().same_owner(right.get()[0])
        return bool(same)


cdef class NativeAccessionFieldResult:
    cdef shared_ptr[AccessionResultBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native accession result is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef NativeAccessionFieldResult out = cls.__new__(cls)
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string image = _image(archive, maximum)
        with nogil: out._handle = AccessionResultBridge.restore(image, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[AccessionResultBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().archive()
        return <bytes>image

    def program(self):
        cdef shared_ptr[AccessionResultBridge] handle = self._pin()
        cdef PreparedAccessionFieldProgram out = PreparedAccessionFieldProgram.__new__(PreparedAccessionFieldProgram)
        with nogil: out._handle = handle.get().result().get().program()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def input_uses(self, *, max_records=256):
        cdef shared_ptr[AccessionResultBridge] handle = self._pin()
        cdef vector[InterfaceInputInfo] values
        cdef size_t maximum = _natural(max_records)
        with nogil: values = handle.get().inputs(maximum)
        return tuple(_input_info(v) for v in values)

    def fields(self, *, max_records=256):
        cdef shared_ptr[AccessionResultBridge] handle = self._pin()
        cdef vector[ProgramFieldInfo] values
        cdef size_t maximum = _natural(max_records)
        with nogil: values = handle.get().result().get().fields(maximum)
        return tuple({'node': int(v.node), 'grade': int(v.grade), 'dimension': int(v.dimension),
                      'supported': int(v.supported), 'nonzeros': int(v.nonzeros),
                      'occurrence_dimension': int(v.occurrence_dimension), 'has_occurrences': bool(v.has_occurrences)} for v in values)

    def field_archive(self, node, *, occurrences=False):
        if type(occurrences) is not bool: raise TypeError('explicit boolean occurrence selection required')
        cdef shared_ptr[AccessionResultBridge] handle = self._pin()
        cdef uint64_t key = _natural(node)
        cdef cppbool flags = occurrences
        cdef string image
        with nogil: image = handle.get().result().get().field(key, flags)
        return <bytes>image

    def select(self, node, cells, *, coordinate_limit=256):
        cdef shared_ptr[AccessionResultBridge] handle = self._pin()
        cdef uint64_t key = _natural(node)
        cdef size_t maximum = _natural(coordinate_limit)
        cdef vector[vector[size_t]] requested = _indices(cells)
        cdef vector[ProgramSelected] values
        with nogil: values = handle.get().result().get().select(key, requested, maximum)
        return tuple(_selected(v) for v in values)

    def interface(self, declaration):
        _closed(declaration, ('profile', 'key', 'branch', 'cells'))
        if declaration['profile'] != 'rexgraph.relational-field-interface.v1': raise ValueError('unknown interface profile')
        cdef shared_ptr[AccessionResultBridge] handle = self._pin()
        cdef InterfacePatch patch
        cdef InterfaceNodeCell item
        cdef NativeFieldInterface out = NativeFieldInterface.__new__(NativeFieldInterface)
        patch.key = _natural(declaration['key']); patch.branch = _natural(declaration['branch'])
        for node, grade, cell in declaration['cells']:
            item.node = _natural(node); item.grade = _natural(grade); item.cell = _natural(cell); patch.cells.push_back(item)
        with nogil: out._handle = handle.get().interface(patch)
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeFieldInterface:
    cdef shared_ptr[FieldPatchBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native field interface is not initialized')
        return self._handle

    def inspect(self):
        cdef shared_ptr[FieldPatchBridge] handle = self._pin()
        cdef InterfacePatch info
        cdef string metric, original
        with nogil: info = handle.get().info(); metric = handle.get().metric(); original = handle.get().original()
        return {'key': int(info.key), 'branch': int(info.branch), 'cells': tuple((int(v.node), int(v.grade), int(v.cell)) for v in info.cells),
                'metric': <bytes>metric, 'original': <bytes>original}

    def same_owner(self, NativeFieldInterface other not None):
        cdef shared_ptr[FieldPatchBridge] left = self._pin(), right = other._pin()
        cdef cppbool result
        with nogil: result = left.get().same_owner(right.get()[0])
        return bool(result)

    def observe(self, declaration):
        cdef shared_ptr[FieldPatchBridge] handle = self._pin()
        cdef InterfaceView view = _view(declaration, self._limits[3])
        cdef NativeFieldReading out = NativeFieldReading.__new__(NativeFieldReading)
        with nogil: out._handle = handle.get().observe(view)
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def admit(self, declaration, field, *, status):
        if status not in _STATUSES: raise ValueError('explicit derived/observed/prospective reading status required')
        cdef shared_ptr[FieldPatchBridge] handle = self._pin()
        cdef InterfaceView view = _view(declaration, self._limits[3])
        cdef string image = _image(field, self._maximum)
        cdef unsigned declared = _STATUSES.index(status)
        cdef NativeFieldReading out = NativeFieldReading.__new__(NativeFieldReading)
        with nogil: out._handle = handle.get().admit(view, image, declared)
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeFieldReading:
    cdef shared_ptr[FieldReadingBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native field reading is not initialized')
        return self._handle

    def inspect(self):
        cdef shared_ptr[FieldReadingBridge] handle = self._pin()
        cdef InterfaceReadingInfo info
        with nogil: info = handle.get().info()
        return {'key': int(info.key), 'basis': int(info.basis), 'directions': tuple(int(k) for k in info.directions),
                'interface': int(info.interface_key), 'branch': int(info.branch), 'status': _STATUSES[info.status],
                'visible': <bytes>info.visible, 'reconstructed': <bytes>info.reconstructed, 'metric': <bytes>info.metric,
                'accession': <bytes>info.accession, 'reconstruction': <bytes>info.reconstruction,
                'physical': <bytes>info.physical if info.has_physical else None}

    def interface(self):
        cdef shared_ptr[FieldReadingBridge] handle = self._pin()
        cdef NativeFieldInterface out = NativeFieldInterface.__new__(NativeFieldInterface)
        with nogil: out._handle = handle.get().interface()
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeFieldSection:
    cdef shared_ptr[FieldSectionBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native field section is not initialized')
        return self._handle

    def __init__(self, readings, *, kind='compatible', limits=_LIMITS, max_bytes=256*1024*1024):
        if self._loading or self._handle.get() != NULL: raise TypeError('native field section owners are immutable')
        self._loading = True
        cdef vector[shared_ptr[FieldReadingBridge]] values
        cdef NativeFieldReading item
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef cppbool contributions
        cdef shared_ptr[FieldSectionBridge] handle
        try:
            if kind not in ('compatible', 'contribution_sum'): raise ValueError('explicit native compatible/contribution_sum section kind required')
            bounds = _bounds(limits); maximum = _maximum(max_bytes); contributions = kind == 'contribution_sum'
            for value in readings:
                if not isinstance(value, NativeFieldReading): raise TypeError('actual native readings required')
                item = value; values.push_back(item._pin())
                if values.size() > bounds[1]: raise ValueError('reading count exceeds admission')
            with nogil: handle = FieldSectionBridge.assemble(values, contributions, bounds, maximum)
            self._handle = handle; self._limits = bounds; self._maximum = maximum
        finally: self._loading = False

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef NativeFieldSection out = cls.__new__(cls)
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string image = _image(archive, maximum)
        with nogil: out._handle = FieldSectionBridge.restore(image, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[FieldSectionBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().archive()
        return <bytes>image

    @property
    def kind(self):
        cdef shared_ptr[FieldSectionBridge] handle = self._pin()
        cdef unsigned value
        with nogil: value = handle.get().kind()
        return ('compatible', 'contribution_sum')[value]

    def readings(self, *, max_records=256):
        cdef shared_ptr[FieldSectionBridge] handle = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef vector[shared_ptr[FieldReadingBridge]] values
        cdef NativeFieldReading out
        with nogil: values = handle.get().readings(maximum)
        result = []
        for value in values:
            out = NativeFieldReading.__new__(NativeFieldReading); out._handle = value
            out._limits = self._limits; out._maximum = self._maximum; result.append(out)
        return tuple(result)

    def parent(self):
        cdef shared_ptr[FieldSectionBridge] handle = self._pin()
        cdef NativeAccessionFieldResult out = NativeAccessionFieldResult.__new__(NativeAccessionFieldResult)
        with nogil: out._handle = handle.get().parent()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def select(self, cells, *, coordinate_limit=256):
        cdef shared_ptr[FieldSectionBridge] handle = self._pin()
        cdef size_t maximum = _natural(coordinate_limit)
        cdef vector[vector[size_t]] requested = _indices(cells)
        cdef vector[ProgramSelected] values
        with nogil: values = handle.get().select(requested, maximum)
        return tuple(_selected(v) for v in values)
