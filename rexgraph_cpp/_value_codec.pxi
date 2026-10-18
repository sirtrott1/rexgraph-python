from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_GET_SIZE, PyBytes_FromStringAndSize
from libc.stdint cimport uint64_t
import struct

cdef extern from "model_storage_bridge.hpp":
    cdef cppclass ModelValueWriter "rexgraph::python::ModelValueWriter":
        ModelValueWriter() except +
        void raw(const unsigned char*, size_t) except +
        void byte(unsigned char) except +
        void uint(size_t) except +
        const char* data() noexcept
        size_t size() noexcept

cdef class _ValueEncoder:
    cdef ModelValueWriter writer
    cdef object exact_type, absent, approx_type, time_type, range_type, array_bytes

    def __cinit__(self):
        from rexgraph.exact_array import ExactArray
        from rexgraph.value import Absent, Approx, ExactTime, TimeRange
        from rexgraph.value_codec import _array_bytes
        self.exact_type=ExactArray;self.absent=Absent;self.approx_type=Approx
        self.time_type=ExactTime;self.range_type=TimeRange;self.array_bytes=_array_bytes

    cdef void raw(self, bytes value) except *:
        self.writer.raw(<const unsigned char*>PyBytes_AS_STRING(value),PyBytes_GET_SIZE(value))

    cdef void blob(self, bytes value) except *:
        self.writer.uint(PyBytes_GET_SIZE(value));self.raw(value)

    cdef bytes finish(self):
        return PyBytes_FromStringAndSize(self.writer.data(),self.writer.size())

    cdef void integer(self, object value) except *:
        cdef bytes raw
        value=int(value)
        raw=value.to_bytes(max(1,(value.bit_length()+8)//8),'little',signed=True)
        while len(raw)>1 and ((raw[-1]==0 and raw[-2]<128) or (raw[-1]==255 and raw[-2]>=128)):
            raw=raw[:-1]
        self.blob(raw)

    cdef void node(self, object value, Py_ssize_t depth) except *:
        cdef object n,d,dtype,array,pair,k,v
        cdef list pairs
        cdef _ValueEncoder child
        if depth>64:raise ValueError('native value nesting is too deep')
        if value is self.absent:self.writer.byte(65)
        elif value is None:self.writer.byte(78)
        elif isinstance(value,(bool,np.bool_)):self.writer.byte(84 if value else 70)
        elif isinstance(value,Integral):
            self.writer.byte(73);self.integer(value)
        elif isinstance(value,Fraction):
            self.writer.byte(81);n=value.numerator;d=value.denominator
            if (type(value) is Fraction and type(n) is int and type(d) is int and
                    -(1<<63)<=n<(1<<63) and 0<d<(1<<63) and math.gcd(n,d)==1):
                self.blob(b'RGXA\x01\x00'+struct.pack('<qq',n,d)+b'\x01\x01\x00')
            else:self.blob(self.exact_type.from_values(value).to_bytes())
        elif isinstance(value,self.exact_type):
            self.writer.byte(69);self.blob(value.to_bytes())
        elif isinstance(value,self.approx_type):
            self.writer.byte(80);self.raw(struct.pack('<d',value.value));self.node(value.source,depth+1)
        elif isinstance(value,self.time_type):
            self.writer.byte(90);self.node(value.seconds,depth+1)
        elif isinstance(value,self.range_type):
            self.writer.byte(82);self.node(value.start,depth+1);self.node(value.end,depth+1)
        elif type(value) is float:
            if not np.isfinite(value):raise ValueError('native real values must be finite')
            self.writer.byte(68);self.raw(struct.pack('<d',value))
        elif isinstance(value,str):self.writer.byte(83);self.blob(value.encode('utf-8'))
        elif isinstance(value,bytes):self.writer.byte(66);self.blob(<bytes>value)
        elif isinstance(value,np.generic):self.writer.byte(71);self.node(np.asarray(value),depth+1)
        elif isinstance(value,np.ndarray):
            if value.dtype.hasobject:
                self.writer.byte(79);self.blob(self.exact_type.from_values(value).to_bytes());return
            if value.dtype.fields is not None or value.dtype.kind not in 'biufcSU':
                raise TypeError('unsupported native tensor dtype')
            if value.dtype.kind in 'fc' and not np.isfinite(value).all():
                raise ValueError('native tensors require finite values')
            dtype=value.dtype.newbyteorder('<');array=np.ascontiguousarray(value.astype(dtype,copy=False))
            self.writer.byte(89);self.blob(dtype.str.encode('ascii'));self.writer.uint(value.ndim)
            for n in value.shape:self.writer.uint(n)
            self.blob(self.array_bytes(array))
        elif isinstance(value,(tuple,list)):
            self.writer.byte(76 if isinstance(value,list) else 85);self.writer.uint(len(value))
            for v in value:self.node(v,depth+1)
        elif isinstance(value,dict):
            pairs=[]
            for k,v in value.items():
                child=_ValueEncoder();child.node(k,depth+1);pair=child.finish()
                child=_ValueEncoder();child.node(v,depth+1);pairs.append((pair,child.finish()))
            pairs.sort();self.writer.byte(77);self.writer.uint(len(pairs))
            for k,v in pairs:self.raw(k);self.raw(v)
        else:raise TypeError(f'unsupported native value type {type(value).__name__}')

cdef class _ValueDecoder:
    cdef bytes data
    cdef const unsigned char* raw
    cdef Py_ssize_t pos,size
    cdef object exact_type, absent, approx_type, time_type, range_type, array_bytes, padding_width

    def __cinit__(self, data):
        from rexgraph.exact_array import ExactArray
        from rexgraph.value import Absent, Approx, ExactTime, TimeRange
        from rexgraph.value_codec import _array_bytes, _padding_width
        if not isinstance(data,bytes):raise TypeError('native record must be bytes')
        self.data=<bytes>data;self.raw=<const unsigned char*>PyBytes_AS_STRING(self.data);self.size=len(data)
        self.exact_type=ExactArray;self.absent=Absent;self.approx_type=Approx
        self.time_type=ExactTime;self.range_type=TimeRange
        self.array_bytes=_array_bytes;self.padding_width=_padding_width

    cdef bytes take(self, Py_ssize_t count):
        if count<0 or count>self.size-self.pos:raise ValueError('truncated native record')
        cdef Py_ssize_t start=self.pos
        self.pos+=count;return self.data[start:self.pos]

    cdef size_t uint(self) except *:
        cdef uint64_t value=0
        cdef unsigned int shift=0
        cdef unsigned char byte
        cdef Py_ssize_t i
        for i in range(10):
            if self.pos==self.size:raise ValueError('truncated native record')
            byte=self.raw[self.pos];self.pos+=1
            if shift==63 and byte&127:raise ValueError('noncanonical or oversized unsigned frame')
            value|=(<uint64_t>(byte&127))<<shift
            if byte<128:
                if i and byte==0:raise ValueError('noncanonical or oversized unsigned frame')
                return value
            shift+=7
        raise ValueError('oversized unsigned frame')

    cdef object integer(self):
        cdef bytes raw=self.take(self.uint())
        cdef Py_ssize_t n=len(raw)
        if n==0:raise ValueError('empty integer frame')
        if n>1 and ((raw[n-1]==0 and raw[n-2]<128) or (raw[n-1]==255 and raw[n-2]>=128)):
            raise ValueError('noncanonical integer frame')
        return int.from_bytes(raw,'little',signed=True)

    cdef object node(self, Py_ssize_t depth):
        cdef unsigned char tag
        cdef bytes raw,prior=None
        cdef object value,exact,n,d,dtype,text,array,shape,key,result,count,ndim
        cdef Py_ssize_t start
        if depth>64:raise ValueError('native value nesting is too deep')
        if self.pos==self.size:raise ValueError('truncated native record')
        tag=self.raw[self.pos];self.pos+=1
        if tag==65:return self.absent
        if tag==78:return None
        if tag==84:return True
        if tag==70:return False
        if tag==73:return self.integer()
        if tag in (81,69,79):
            raw=self.take(self.uint())
            if tag==81 and len(raw)==25 and raw[:6]==b'RGXA\x01\x00' and raw[22:]==b'\x01\x01\x00':
                n,d=struct.unpack_from('<qq',raw,6)
                if d<=0 or math.gcd(n,d)!=1:raise ValueError('invalid exact denominator or coefficient')
                return Fraction(n,d)
            exact=self.exact_type.from_bytes(raw)
            if tag==81:
                if exact.shape!=() or not exact.presence.item() or exact.kind.item()!=1:
                    raise ValueError('invalid rational scalar')
                return exact.values().item()
            return exact if tag==69 else exact.values()
        if tag==80:return self.approx_type(struct.unpack('<d',self.take(8))[0],self.node(depth+1))
        if tag==90:
            value=self.node(depth+1)
            if not isinstance(value,Fraction):raise ValueError('timestamp requires a rational carrier')
            return self.time_type(value)
        if tag==82:return self.range_type(self.node(depth+1),self.node(depth+1))
        if tag==68:
            value=struct.unpack('<d',self.take(8))[0]
            if not np.isfinite(value):raise ValueError('native real values must be finite')
            return value
        if tag in (83,66):
            raw=self.take(self.uint());return raw.decode('utf-8') if tag==83 else raw
        if tag==71:
            array=self.node(depth+1)
            if not isinstance(array,np.ndarray) or array.shape!=():raise ValueError('invalid NumPy scalar')
            return array[()]
        if tag==89:
            text=self.take(self.uint()).decode('ascii');dtype=np.dtype(text)
            if (dtype.hasobject or dtype.fields is not None or dtype.kind not in 'biufcSU' or
                    dtype.str!=dtype.newbyteorder('<').str or dtype.str!=text):
                raise ValueError('unsupported or noncanonical native tensor dtype')
            ndim=self.uint()
            if ndim>64:raise ValueError('native tensor has too many dimensions')
            shape=tuple(self.uint() for _ in range(ndim));raw=self.take(self.uint())
            if math.prod(shape)*dtype.itemsize!=len(raw):raise ValueError('native tensor shape does not match its bytes')
            array=np.frombuffer(raw,dtype=dtype).reshape(shape)
            if dtype.kind in 'fc' and not np.isfinite(array).all():raise ValueError('native tensor requires finite values')
            if dtype.kind=='b' and any(byte>1 for byte in raw):raise ValueError('noncanonical boolean tensor')
            if self.padding_width(array) and self.array_bytes(array)!=raw:
                raise ValueError('noncanonical extended-float tensor padding')
            return array
        if tag in (76,85,77):
            count=self.uint()
            if count>(self.size-self.pos)//(2 if tag==77 else 1):raise ValueError('truncated native collection')
            if tag!=77:
                value=[self.node(depth+1) for _ in range(count)];return value if tag==76 else tuple(value)
            result={}
            for _ in range(count):
                start=self.pos;key=self.node(depth+1);raw=self.data[start:self.pos]
                if prior is not None and raw<=prior:raise ValueError('noncanonical native map key order')
                try:
                    if key in result:raise ValueError('duplicate native map key')
                    result[key]=self.node(depth+1)
                except TypeError as exc:raise ValueError('native map key must be hashable') from exc
                prior=raw
            return result
        raise ValueError(f'unknown native value tag {bytes((tag,))!r}')


def pack_native_value(value):
    """Emit canonical RGVL values with the existing typed byte format."""
    cdef _ValueEncoder encoder=_ValueEncoder()
    encoder.raw(b'RGVL\x01');encoder.node(value,0);return encoder.finish()


def unpack_native_value(data):
    """Validate and decode canonical RGVL values without Python cursor loops."""
    cdef _ValueDecoder decoder=_ValueDecoder(data)
    if decoder.take(5)!=b'RGVL\x01':raise ValueError('unsupported native value format')
    value=decoder.node(0)
    if decoder.pos!=decoder.size:raise ValueError('trailing bytes in native record')
    return value
