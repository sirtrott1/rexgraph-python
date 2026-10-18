# distutils: language = c++
"""Bounded compiled content checks and compatible model payload encoding."""
from collections.abc import Mapping
from fractions import Fraction
from types import MappingProxyType
from numbers import Integral
import math
import numpy as np

from cpython.buffer cimport Py_buffer, PyObject_GetBuffer, PyBuffer_Release, PyBUF_SIMPLE
from libc.stddef cimport size_t
from libcpp.memory cimport shared_ptr

cdef extern from "model_storage_bridge.hpp":
    cdef cppclass ModelByteImage "rexgraph::python::ModelByteImage":
        ModelByteImage(const unsigned char*, size_t, size_t) except +
        bint matches(const unsigned char*, size_t) noexcept

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_MODEL_STORAGE_BINDING_DIGEST

_ATOMS = {type(None), bool, int, str, bytes}

include "_value_codec.pxi"

class _Unavailable(Exception):
    pass

class StorageTraversalLimit(RecursionError):
    pass

cdef class _ByteImage:
    cdef shared_ptr[ModelByteImage] image

    def __cinit__(self, value, size_t maximum):
        cdef Py_buffer view
        PyObject_GetBuffer(value, &view, PyBUF_SIMPLE)
        try:
            self.image.reset(new ModelByteImage(<const unsigned char*>view.buf, view.len, maximum))
        finally:
            PyBuffer_Release(&view)

    cdef bint matches(self, object value) except -1:
        cdef Py_buffer view
        PyObject_GetBuffer(value, &view, PyBUF_SIMPLE)
        try:
            return self.image.get().matches(<const unsigned char*>view.buf, view.len)
        finally:
            PyBuffer_Release(&view)

cdef tuple _parts(object value):
    cdef object n=value._numerator, d=value._denominator
    return (type(n), int(n), type(d), int(d))

cdef bint _fraction(object value, tuple parts) except -1:
    cdef object n, d
    if type(value) is not Fraction:
        return False
    n=value._numerator; d=value._denominator
    return type(n) is parts[0] and n == parts[1] and type(d) is parts[2] and d == parts[3]

cdef bint _fraction_block(object values, tuple parts) except -1:
    cdef Py_ssize_t i = 0
    cdef object value
    for value in values:
        if i >= len(parts) or not _fraction(value, parts[i]):
            return False
        i += 1
    return i == len(parts)

cdef class _Capture:
    cdef Py_ssize_t nodes, bytes, max_nodes, max_bytes
    cdef dict owners
    cdef object space_type, source_type, input_type, fields

    def __cinit__(self, Py_ssize_t max_nodes, Py_ssize_t max_bytes):
        from rexgraph.model_state import ModelInput
        from rexgraph.tensor_field import FieldSource
        from rexgraph.type_accession import CoordinateSpace
        from rexgraph.model_integrity import _fields
        self.max_nodes = max_nodes; self.max_bytes = max_bytes
        self.owners = {}
        self.space_type = CoordinateSpace; self.source_type = FieldSource
        self.input_type = ModelInput; self.fields = _fields

    cdef void charge(self, Py_ssize_t count=1, Py_ssize_t size=0) except *:
        if count > self.max_nodes-self.nodes or size > self.max_bytes-self.bytes:
            raise _Unavailable
        self.nodes += count; self.bytes += size

    cdef tuple node(self, object value, Py_ssize_t depth=0):
        cdef object kind = type(value), known, result, v, k, raw
        cdef list children
        if depth > 128:
            raise _Unavailable
        self.charge()
        if kind in _ATOMS:
            return (0, value)
        if kind is float:
            return (1, value.hex())
        if kind is Fraction:
            return (2, _parts(value))
        if isinstance(value, np.integer):
            return (3, kind, int(value))
        known = self.owners.get(id(value))
        if known is not None and known[0] is value:
            return known[1]
        if kind is np.ndarray:
            if value.dtype.hasobject:
                if value.size > self.max_nodes-self.nodes:
                    raise _Unavailable
                if all(type(v) is Fraction for v in value.flat):
                    self.charge(value.size)
                    children = []
                    for v in value.flat:
                        children.append(_parts(v))
                    result = (5, value.dtype.str, value.shape, tuple(children))
                else:
                    children = []
                    for v in value.flat:
                        children.append(self.node(v, depth+1))
                    result = (6, value.dtype.str, value.shape, tuple(children))
            else:
                self.charge(1, value.nbytes)
                raw = value if value.flags.c_contiguous else value.tobytes(order='C')
                result = (4, value.dtype.str, value.shape, _ByteImage(raw, self.max_bytes))
        elif kind is dict or kind is MappingProxyType or isinstance(value, Mapping):
            if len(value) > self.max_nodes-self.nodes:
                raise _Unavailable
            children = []
            for k, v in value.items():
                if type(k) not in {str, int}:
                    raise _Unavailable
                children.append((k, self.node(v, depth+1)))
            result = (7, tuple(children))
        elif kind is tuple and all(type(v) in _ATOMS for v in value):
            result = (8, value)
        elif kind is tuple and all(type(v) is Fraction for v in value):
            self.charge(len(value))
            children = []
            for v in value:
                children.append(_parts(v))
            result = (9, tuple(children))
        elif kind is tuple or kind is list:
            if len(value) > self.max_nodes-self.nodes:
                raise _Unavailable
            children = []
            for v in value:
                children.append(self.node(v, depth+1))
            result = (10, tuple(children))
        elif isinstance(value, self.space_type):
            result = (11, self.node(value.name, depth+1), self.node(value.keys, depth+1))
        elif isinstance(value, self.source_type):
            result = (12, kind, self.node(value.as_record(), depth+1))
        elif isinstance(value, self.input_type):
            kind, v = self.fields(value)
            result = (13, kind, self.node(v, depth+1))
        else:
            raise _Unavailable
        self.owners[id(value)] = (value, result)
        return result

cdef bint _match(object value, tuple node, dict visited, object space_type, object fields, bint ordered) except -1:
    cdef int tag = node[0]
    cdef object key, v, k, child, old, kind, raw
    cdef Py_ssize_t i
    cdef bint valid = False
    if tag == 0:
        return type(value) is type(node[1]) and (value is node[1] or value == node[1])
    if tag == 1:
        return type(value) is float and value.hex() == node[1]
    if tag == 2:
        return _fraction(value, node[1])
    if tag == 3:
        return type(value) is node[1] and int(value) == node[2]
    if tag == 8:
        if type(value) is not tuple or len(value) != len(node[1]):
            return False
        if value is node[1]:
            return True
        for i in range(len(value)):
            v = value[i]; old = node[1][i]
            if type(v) is not type(old) or v != old:
                return False
        return True
    key = (id(value), id(node))
    if visited.get(key) is value:
        return True
    if tag == 9:
        valid = type(value) is tuple and len(value) == len(node[1]) and _fraction_block(value, node[1])
    elif tag == 4 or tag == 5 or tag == 6:
        if type(value) is not np.ndarray or value.dtype.str != node[1] or value.shape != node[2]:
            return False
        if tag == 4:
            raw = value if value.flags.c_contiguous else value.tobytes(order='C')
            valid = (<_ByteImage>node[3]).matches(raw)
        elif tag == 5:
            valid = _fraction_block(value.flat, node[3])
        else:
            i = 0
            for v in value.flat:
                if i >= len(node[3]) or not _match(v, node[3][i], visited, space_type, fields, ordered):
                    return False
                i += 1
            valid = i == len(node[3])
    elif tag == 7:
        if (type(value) is not dict and type(value) is not MappingProxyType and not isinstance(value, Mapping)) or len(value) != len(node[1]):
            return False
        if ordered:
            i = 0
            for k in value:
                if i >= len(node[1]):
                    return False
                old, child = node[1][i]
                if type(k) is not type(old) or k != old or not _match(value[k], child, visited, space_type, fields, ordered):
                    return False
                i += 1
        else:
            kinds = {k: type(k) for k in value}
            for old, child in node[1]:
                if kinds.get(old) is not type(old) or not _match(value[old], child, visited, space_type, fields, ordered):
                    return False
        valid = True
    elif tag == 10:
        if (type(value) is not tuple and type(value) is not list) or len(value) != len(node[1]):
            return False
        for i in range(len(value)):
            if not _match(value[i], node[1][i], visited, space_type, fields, ordered):
                return False
        valid = True
    elif tag == 11:
        valid = (isinstance(value, space_type) and _match(value.name, node[1], visited, space_type, fields, ordered)
                 and _match(value.keys, node[2], visited, space_type, fields, ordered))
    elif tag == 12:
        valid = type(value) is node[1] and _match(value.as_record(), node[2], visited, space_type, fields, ordered)
    elif tag == 13:
        kind, v = fields(value)
        valid = kind is node[1] and _match(v, node[2], visited, space_type, fields, ordered)
    if valid:
        visited[key] = value
    return valid

cdef class NativeContentWitness:
    """An owning tensor byte image and type sensitive declaration witness."""
    cdef tuple tree
    cdef object space_type, fields

    def __init__(self):
        raise TypeError('use capture_content')

    def matches(self, value, *, bint ordered=True):
        try:
            return _match(value, self.tree, {}, self.space_type, self.fields, ordered)
        except (AttributeError, KeyError, IndexError, TypeError, ValueError, _Unavailable):
            return False

def capture_content(value, Py_ssize_t max_nodes=262144, Py_ssize_t max_bytes=32*1024*1024):
    """Capture finite supported content without retaining mutable array aliases."""
    cdef _Capture capture
    cdef NativeContentWitness result
    if max_nodes < 0 or max_bytes < 0:
        return None
    try:
        capture = _Capture(max_nodes, max_bytes)
        result = NativeContentWitness.__new__(NativeContentWitness)
        result.tree = capture.node(value)
        result.space_type = capture.space_type; result.fields = capture.fields
        return result
    except _Unavailable:
        return None

cdef class PayloadEncoder:
    """Encode retained values with the existing model checkpoint tags and order."""
    cdef public dict tensors
    cdef public list cpp_sources
    cdef dict rationals
    cdef object input_type, cpp_type, algebraic_type
    cdef Py_ssize_t cache_limit
    cdef Py_ssize_t depth

    def __cinit__(self, Py_ssize_t cache_limit=1024):
        from rexgraph.model_state import ModelInput
        from rexgraph.source_binding import CppSourceReference, CppAlgebraicSourceReference
        self.input_type = ModelInput; self.cpp_type = CppSourceReference; self.algebraic_type = CppAlgebraicSourceReference
        self.tensors = {}; self.cpp_sources = []; self.rationals = {}; self.cache_limit = cache_limit

    cdef object source(self, object value):
        if isinstance(value, self.cpp_type) or isinstance(value, self.algebraic_type):
            self.cpp_sources.append(value)
        return value.as_record()

    def put(self, value):
        return self.encode(value)

    cdef object encode(self, object value):
        if self.depth >= 128:
            raise StorageTraversalLimit('compiled model metadata nesting exceeds admission')
        self.depth += 1
        try:
            return self.encode_value(value)
        finally:
            self.depth -= 1

    cdef object encode_value(self, object value):
        cdef object name, a, stored, parts, declaration, k, v, identity
        cdef list items
        cdef dict owners, declarations
        if isinstance(value, self.input_type):
            return ['input', self.encode(value.original), self.encode(value.values), [value.space.name, list(value.space.keys)],
                    [[a.name, list(a.keys)] for a in value.axes], self.source(value.source),
                    value.origin_digest, value.origin_arithmetic, [self.source(v) for v in value.dependencies]]
        if isinstance(value, np.ndarray):
            name = f'tensor/{len(self.tensors)}'
            a = np.array(value, copy=True, order='C')
            stored = a.astype(np.uint8) if a.dtype == bool else a
            self.tensors[name] = stored.reshape(1) if a.ndim == 0 else stored
            return ['array', name, a.dtype.str, list(a.shape)]
        if value is None: return ['none']
        if isinstance(value, bool): return ['bool', value]
        if isinstance(value, str): return ['str', value]
        if isinstance(value, bytes):
            name = f'tensor/{len(self.tensors)}'
            self.tensors[name] = np.frombuffer(value, dtype=np.uint8).copy()
            return ['bytes', name]
        if isinstance(value, int): return ['int', hex(value)]
        if isinstance(value, Fraction):
            parts = (value._numerator, value._denominator) if type(value) is Fraction else (value.numerator, value.denominator)
            declaration = self.rationals.get(parts)
            if declaration is None:
                declaration = ['rational', hex(parts[0]), hex(parts[1])]
                if len(self.rationals) < self.cache_limit:
                    self.rationals[parts] = declaration
            return declaration
        if isinstance(value, float): return ['float', value.hex()]
        if type(value) is dict or type(value) is MappingProxyType or isinstance(value, Mapping):
            items = []
            for k, v in sorted(value.items(), key=_mapping_order):
                items.append([self.encode(k), self.encode(v)])
            return ['map', items]
        if isinstance(value, (tuple, list)):
            declarations = {}; owners = {}; items = []
            for v in value:
                identity = id(v)
                if type(v) is MappingProxyType and _primitive_map(v):
                    declaration = declarations.get(identity)
                    if declaration is None:
                        declaration = self.encode(v)
                        if len(declarations) < self.cache_limit:
                            owners[identity] = v; declarations[identity] = declaration
                    items.append(declaration)
                else:
                    items.append(self.encode(v))
            return ['tuple', items]
        raise TypeError('unsupported model payload value')

cdef bint _primitive_map(object value) except -1:
    cdef object k, v
    if len(value) > 8:
        return False
    for k, v in value.items():
        if type(k) not in {str, int} or type(v) not in {type(None), bool, int, str, float}:
            return False
    return True

def _mapping_order(item):
    return type(item[0]).__name__, str(item[0])

cdef class _Freezer:
    cdef dict declarations
    cdef object input_type
    cdef Py_ssize_t cache_limit
    cdef Py_ssize_t depth

    def __cinit__(self, Py_ssize_t cache_limit):
        from rexgraph.model_state import ModelInput
        self.input_type = ModelInput; self.cache_limit = cache_limit; self.declarations = {}

    cdef object node(self, object value):
        if self.depth >= 128:
            raise _Unavailable
        self.depth += 1
        try:
            return self.freeze(value)
        finally:
            self.depth -= 1

    cdef object freeze(self, object value):
        cdef object k, v, result, key, known, frozen
        cdef list items, parts
        if value is None or isinstance(value, (str, bytes, bool, Fraction)):
            return value
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, (float, np.floating)):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError('model metadata must be finite')
            return result
        if isinstance(value, np.ndarray):
            result = np.array(value, copy=True, order='C')
            if result.dtype.hasobject:
                for v in result.flat:
                    if isinstance(v, Fraction):
                        continue
                    if isinstance(v, Integral) and not isinstance(v, (bool, np.bool_)):
                        continue
                    raise TypeError('model exact object tensors require integer or Fraction coefficients')
            elif result.dtype.kind not in 'biuf' or not result.dtype.isnative or result.dtype.itemsize not in {1,2,4,8}:
                raise TypeError('unsupported model tensor dtype')
            elif result.dtype.kind == 'f' and not np.isfinite(result).all():
                raise ValueError('model tensor must be finite')
            result.flags.writeable = False
            return result
        if isinstance(value, self.input_type):
            value.check_state()
            return value
        if type(value) is dict or type(value) is MappingProxyType or isinstance(value, Mapping):
            result = {}
            for k, v in value.items():
                if not isinstance(k, (str,int)) or isinstance(k, bool):
                    raise TypeError('model mapping keys must be strings or integers')
                result[k] = self.node(v)
            key = None
            if _primitive_map(result):
                parts = []
                for k, v in result.items():
                    parts.append((type(k), k, type(v), v.hex() if type(v) is float else v))
                key = tuple(parts)
                known = self.declarations.get(key)
                if known is not None:
                    return known
            frozen = MappingProxyType(result)
            if key is not None and len(self.declarations) < self.cache_limit:
                self.declarations[key] = frozen
            return frozen
        if isinstance(value, (tuple,list)):
            items = []
            for v in value:
                items.append(self.node(v))
            return tuple(items)
        raise TypeError(f'unsupported model value {type(value).__name__}')

def freeze_model_tree(value, Py_ssize_t cache_limit=1024):
    """Copy supported tensors and retain bounded shared scalar declarations."""
    cdef _Freezer freezer = _Freezer(cache_limit)
    try:
        return freezer.node(value)
    except _Unavailable:
        from rexgraph.model_state import _freeze_tree
        return _freeze_tree(value, {})

cdef class PayloadDecoder:
    """Decode existing model value tags while retaining exact claimed tensor names."""
    cdef object tensors, used, source, space, input_type

    def __cinit__(self, tensors, used, source, space):
        from rexgraph.model_state import ModelInput
        self.tensors = tensors; self.used = used; self.source = source; self.space = space
        self.input_type = ModelInput

    def take(self, value):
        return self.decode(value)

    cdef object decode(self, object value, Py_ssize_t depth=0):
        cdef object tag, a, dtype, shape, n, key, item, v
        cdef list items
        cdef dict result
        if depth > 128:
            raise StorageTraversalLimit('compiled model value nesting exceeds admission')
        if not isinstance(value, list) or not value:
            raise ValueError('invalid model value')
        tag = value[0]
        if tag == 'input' and len(value) == 9:
            return self.input_type(self.decode(value[1], depth+1), self.decode(value[2], depth+1),
                self.space(value[3]), tuple(self.space(v) for v in value[4]), self.source(value[5]),
                value[6], value[7], tuple(self.source(v) for v in value[8]))
        if (tag == 'array' and len(value) == 4) or (tag == 'bytes' and len(value) == 2):
            self.used.add(value[1]); a = np.asarray(self.tensors[value[1]])
            if tag == 'bytes':
                if a.dtype != np.uint8 or a.ndim != 1:
                    raise ValueError('invalid model bytes')
                return a.tobytes()
            dtype = np.dtype(value[2]); shape = tuple(value[3])
            for n in shape:
                if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                    raise ValueError('invalid model tensor shape')
            if dtype == np.dtype(bool):
                if a.dtype != np.uint8 or not np.isin(a, (0,1)).all():
                    raise ValueError('invalid model boolean tensor')
                a = a.astype(bool)
            elif a.dtype != dtype:
                raise ValueError('model tensor dtype differs')
            return a.reshape(shape)
        if tag == 'none' and len(value) == 1: return None
        if tag == 'bool' and len(value) == 2 and isinstance(value[1], bool): return value[1]
        if tag == 'str' and len(value) == 2 and isinstance(value[1], str): return value[1]
        if tag == 'int' and len(value) == 2: return int(value[1], 16)
        if tag == 'rational' and len(value) == 3: return Fraction(int(value[1],16), int(value[2],16))
        if tag == 'float' and len(value) == 2: return float.fromhex(value[1])
        if tag == 'tuple' and len(value) == 2:
            items = []
            for item in value[1]:
                items.append(self.decode(item, depth+1))
            return tuple(items)
        if tag == 'map' and len(value) == 2:
            result = {}
            for key, item in value[1]:
                key = self.decode(key, depth+1)
                if key in result:
                    raise ValueError('duplicate model mapping key')
                result[key] = self.decode(item, depth+1)
            return result
        raise ValueError('unsupported model value tag')

def model_storage_build_info():
    return {'engine': (<bytes>RFC_ENGINE_VERSION).decode(), 'engine_headers': (<bytes>RFC_ENGINE_DIGEST).decode(),
            'binding': (<bytes>RFC_MODEL_STORAGE_BINDING_DIGEST).decode(), 'profile': 'model-content-storage-v1'}
