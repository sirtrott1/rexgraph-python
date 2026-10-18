from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t

cdef extern from "field_program_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass ProgramEntry:
        size_t row, column
        string value
    cdef cppclass ProgramMatrix:
        size_t rows, columns
        vector[ProgramEntry] entries
    cdef cppclass ProgramStalk:
        size_t grade, cell
        uint64_t basis
        vector[uint64_t] directions
        ProgramMatrix metric
    cdef cppclass ProgramSourceInfo:
        uint64_t name_space, sequence
        size_t semantic_scale
        string endpoint, archive
        vector[vector[string]] coordinates
    cdef cppclass ProgramStalkInfo:
        size_t grade, cell
        uint64_t basis
        vector[uint64_t] directions
        string metric
    cdef cppclass ProgramFieldInfo:
        uint64_t node
        size_t grade, dimension, supported, nonzeros, occurrence_dimension
        cppbool has_occurrences
    cdef cppclass ProgramSelected:
        ProgramStalkInfo stalk
        string field
        vector[string] values
    cdef cppclass FieldProgramBridge:
        @staticmethod
        shared_ptr[FieldProgramBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        ProgramSourceInfo source() except +
        uint64_t key() except +
        uint64_t revision() except +
    cdef cppclass FieldProgramResultBridge:
        shared_ptr[FieldProgramBridge] program() except +
        vector[ProgramFieldInfo] fields(size_t) except +
        string field(uint64_t, cppbool) except +
        vector[ProgramSelected] select(uint64_t, const vector[vector[size_t]]&, size_t) except +

cdef extern from "field_interface_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass InterfaceUse:
        uint64_t key
        vector[vector[uint64_t]] cells, occurrences
        vector[vector[size_t]] spans
    cdef cppclass InterfaceAssetInfo:
        uint64_t key, channel
        string encoding
        size_t byte_count, use_count
        ProgramSourceInfo source
    cdef cppclass InterfaceAccession:
        uint64_t key, node
        size_t grade
        vector[ProgramStalk] stalks
        vector[vector[size_t]] cells, occurrences
        ProgramMatrix accession, reconstruction
        cppbool isometric
    cdef cppclass InterfaceInputInfo:
        uint64_t key, node, use
        size_t grade
        cppbool isometric
        string asset, field, represented, accession, reconstruction, source_metric, target_metric
        vector[ProgramStalkInfo] stalks
        vector[vector[size_t]] cells, occurrences
    cdef cppclass InterfaceNodeCell:
        uint64_t node
        size_t grade, cell
    cdef cppclass InterfacePatch:
        uint64_t key, branch
        vector[InterfaceNodeCell] cells
    cdef cppclass InterfaceView:
        uint64_t key, basis
        vector[uint64_t] directions
        ProgramMatrix accession, reconstruction, physical
        cppbool has_physical
    cdef cppclass InterfaceReadingInfo:
        uint64_t key, basis, interface_key, branch
        unsigned status
        vector[uint64_t] directions
        string visible, reconstructed, metric, accession, reconstruction, physical
        cppbool has_physical
    cdef cppclass InputUseAssetBridge:
        InputUseAssetBridge(const string&, uint64_t, uint64_t, const string&, const string&,
            const vector[InterfaceUse]&, const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[InputUseAssetBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        InterfaceAssetInfo info() except +
        string source_archive() except +
        vector[InterfaceUse] uses(size_t) except +
        vector[string] recover(uint64_t, size_t) except +
        cppbool same_owner(const InputUseAssetBridge&) except +
    cdef cppclass AccessionInputBridge:
        InterfaceInputInfo info() except +
    cdef cppclass InputAccessionBridge:
        InputAccessionBridge(shared_ptr[InputUseAssetBridge], shared_ptr[FieldProgramBridge],
            const InterfaceAccession&, const vector[size_t]&, size_t) except +
        shared_ptr[AccessionInputBridge] admit(uint64_t, const string&) except +
    cdef cppclass FieldPatchBridge
    cdef cppclass FieldReadingBridge
    cdef cppclass AccessionResultBridge:
        @staticmethod
        shared_ptr[AccessionResultBridge] execute(shared_ptr[FieldProgramBridge],
            const vector[shared_ptr[AccessionInputBridge]]&, const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[AccessionResultBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        shared_ptr[FieldProgramResultBridge] result() except +
        vector[InterfaceInputInfo] inputs(size_t) except +
        shared_ptr[FieldPatchBridge] interface(const InterfacePatch&) except +
    cdef cppclass FieldPatchBridge:
        InterfacePatch info() except +
        string metric() except +
        string original() except +
        cppbool same_owner(const FieldPatchBridge&) except +
        shared_ptr[FieldReadingBridge] observe(const InterfaceView&) except +
        shared_ptr[FieldReadingBridge] admit(const InterfaceView&, const string&, unsigned) except +
    cdef cppclass FieldReadingBridge:
        InterfaceReadingInfo info() except +
        shared_ptr[FieldPatchBridge] interface() except +
    cdef cppclass FieldSectionBridge:
        @staticmethod
        shared_ptr[FieldSectionBridge] assemble(const vector[shared_ptr[FieldReadingBridge]]&,
            cppbool, const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[FieldSectionBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        unsigned kind() except +
        vector[shared_ptr[FieldReadingBridge]] readings(size_t) except +
        shared_ptr[AccessionResultBridge] parent() except +
        vector[ProgramSelected] select(const vector[vector[size_t]]&, size_t) except +


cdef vector[size_t] _bounds(value)
cdef size_t _maximum(value)
cdef string _image(value, size_t maximum)
cdef ProgramMatrix _matrix(value, size_t bits)
cdef InterfaceAccession _accession_declaration(value, size_t bits)
cdef vector[vector[size_t]] _indices(value)
cdef object _publication(ProgramSourceInfo source)
cdef object _input_info(InterfaceInputInfo value)
cdef object _selected(ProgramSelected value)

cdef extern from "literal_input_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass LiteralCoordinate:
        uint64_t key, direction, use
        size_t grade, cell, span_index, begin, end
        unsigned kind
        string operand
    cdef cppclass LiteralDeclaration:
        uint64_t key, revision, use
        vector[LiteralCoordinate] coordinates
    cdef cppclass LiteralInputBridge:
        LiteralInputBridge(shared_ptr[InputAccessionBridge], const LiteralDeclaration&,
                           const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[LiteralInputBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        LiteralDeclaration info(size_t) except +
        shared_ptr[AccessionInputBridge] input() except +
        shared_ptr[InputUseAssetBridge] asset() except +
        shared_ptr[FieldProgramBridge] program() except +
        cppbool same_owner(const LiteralInputBridge&) except +

cdef class NativeLiteralInput:
    cdef shared_ptr[LiteralInputBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[LiteralInputBridge] _pin(self) except *

cdef class PreparedAccessionFieldProgram:
    cdef shared_ptr[FieldProgramBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[FieldProgramBridge] _pin(self) except *

cdef class NativeInputUseAsset:
    cdef shared_ptr[InputUseAssetBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[InputUseAssetBridge] _pin(self) except *

cdef class NativeInputAccession:
    cdef shared_ptr[InputAccessionBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[InputAccessionBridge] _pin(self) except *

cdef class NativeAccessionInput:
    cdef shared_ptr[AccessionInputBridge] _handle
    cdef shared_ptr[AccessionInputBridge] _pin(self) except *

cdef class NativeAccessionFieldResult:
    cdef shared_ptr[AccessionResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[AccessionResultBridge] _pin(self) except *

cdef class NativeFieldInterface:
    cdef shared_ptr[FieldPatchBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FieldPatchBridge] _pin(self) except *

cdef class NativeFieldReading:
    cdef shared_ptr[FieldReadingBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FieldReadingBridge] _pin(self) except *

cdef class NativeFieldSection:
    cdef shared_ptr[FieldSectionBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[FieldSectionBridge] _pin(self) except *
