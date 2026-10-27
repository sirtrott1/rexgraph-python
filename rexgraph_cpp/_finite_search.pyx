# distutils: language=c++
"""Owning compiled finite queries, support fields and correlated interface images."""
from ._finite_field cimport NativeFiniteSource
from ._field_interface cimport _bounds, _maximum, _indices, _image
from ._field_program import _closed
from ._stalk_fields import _natural

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_FINITE_SEARCH_BINDING_DIGEST

_LIMITS = (4096, 16000000, 64000000, 16384)
_SEARCH = (65536, 4194304, 16777216)
_FAMILY = (65536, 64000000, 16777216, 65536, 65536)
_SOLVE_STATUS = ('satisfiable', 'unsatisfiable', 'budget_exceeded', 'cancelled')


def finite_search_build_info():
    return {'engine': 'rexgraph-cpp', 'engine_version': RFC_ENGINE_VERSION.decode(),
            'engine_headers': RFC_ENGINE_DIGEST.decode(), 'binding': RFC_FINITE_SEARCH_BINDING_DIGEST.decode(),
            'profile': 'raw-owning-relational-finite-search-v2', 'authority': 'raw-none',
            'proof_profile': 'rexgraph.finite-projection-proof.v2',
            'scalar_policy': 'exact-symbolic'}


cdef vector[size_t] _admission(value, size_t count):
    if len(value) != count: raise ValueError('finite execution limit arity differs')
    cdef vector[size_t] out
    for v in value: out.push_back(_natural(v))
    return out


cdef vector[FiniteAssignmentValue] _fixed(value, size_t maximum):
    cdef vector[FiniteAssignmentValue] out
    cdef FiniteAssignmentValue row
    for item in value:
        if len(item) != 3: raise ValueError('fixed finite value requires grade, cell index and symbol')
        if out.size() >= maximum: raise ValueError('fixed finite values exceed admission')
        row.grade = _natural(item[0]); row.cell = _natural(item[1]); row.symbol = _natural(item[2])
        out.push_back(row)
    return out


cdef vector[uint64_t] _symbols(value, size_t maximum):
    cdef vector[uint64_t] out
    for item in value:
        if out.size() >= maximum: raise ValueError('finite symbol list exceeds admission')
        out.push_back(_natural(item))
    return out


cdef shared_ptr[FiniteSearchStopBridge] _stop(value):
    if value is None: return shared_ptr[FiniteSearchStopBridge]()
    if not isinstance(value, FiniteSearchCancellation): raise TypeError('actual finite cancellation owner required')
    return (<FiniteSearchCancellation>value)._pin()


cdef object _receipt(FiniteSearchReceipt w):
    return {'steps': w.steps, 'value_trials': w.value_trials, 'chart_tests': w.chart_tests,
            'tuple_tests': w.tuple_tests, 'coordinate_tests': w.coordinate_tests, 'variables': w.variables,
            'peak_assigned': w.peak_assigned, 'peak_entries': w.peak_entries,
            'domain_reductions': w.domain_reductions, 'search_nodes': w.search_nodes,
            'distinct_groups': w.distinct_groups, 'matching_tests': w.matching_tests}


cdef object _family_receipt(RelationalFiniteFamilyWork w):
    return {'steps': w.steps, 'peak_entries': w.peak_entries, 'candidates': w.candidates,
            'completed_queries': w.completed_queries, 'preparations': w.preparations,
            'source_queries': w.source_queries, 'rows': w.rows, 'witness_values': w.witness_values,
            'reused_preparations': w.reused_preparations, 'prior_preparation_steps': w.prior_preparation_steps,
            'reused_propagations': w.reused_propagations, 'prior_propagation_steps': w.prior_propagation_steps,
            'continued_source_queries': w.continued_source_queries,
            'propagations': w.propagations, 'pruned_values': w.pruned_values, 'native': _receipt(w.native)}


cdef object _pruning_info(FinitePruningInfo info):
    return dict(present=bool(info.present), complete=bool(info.complete),
        status=('complete', 'budget_exceeded', 'cancelled')[info.status],
        limits=tuple(info.limits),
        window=tuple(tuple(row) for row in info.window) if info.present else None,
        unsatisfiable_leaf=info.unsatisfiable_leaf if info.has_unsatisfiable_leaf else None,
        propagations=tuple(dict(leaf=p.leaf,
            status=('fixed_point', 'unsatisfiable', 'budget_exceeded', 'cancelled')[p.status],
            reused=bool(p.reused), has_snapshot=bool(p.has_snapshot), variable_count=p.variables,
            work=_receipt(p.work)) for p in info.propagations))


cdef object _section(FiniteSectionReply result):
    return {'status': _SOLVE_STATUS[result.status],
            'witness': tuple((v.grade, v.cell, v.symbol) for v in result.witness) if result.has_witness else None,
            'work': _receipt(result.work)}


cdef object _projection_info(FiniteProjectionInfo d):
    return dict(key=d.key, revision=d.revision, cells=tuple(tuple(c) for c in d.cells),
        distinct_relations=tuple(tuple(c) for c in d.relations),
        fixed=tuple((r.grade, r.cell, r.symbol) for r in d.fixed),
        window=tuple(tuple(r) for r in d.window) if d.has_window else None)


cdef object _query_rows(vector[RelationalFiniteImageQuery] rows):
    return tuple({'tuple': tuple(r.tuple), 'status': _SOLVE_STATUS[<unsigned>r.status],
                  'source_queries': r.source_queries, 'native_steps': r.native_steps,
                  'decisive_leaf': r.decisive_leaf.value() if r.decisive_leaf.has_value() else None} for r in rows)


cdef NativeFiniteSource _source(shared_ptr[FiniteSourceBridge] handle, vector[size_t] limits, size_t maximum):
    cdef NativeFiniteSource out = NativeFiniteSource.__new__(NativeFiniteSource)
    out._handle = handle; out._limits = limits; out._maximum = maximum
    return out


cdef NativeFiniteFamily _family(shared_ptr[FiniteFamilyBridge] handle, vector[size_t] limits, size_t maximum):
    cdef NativeFiniteFamily out = NativeFiniteFamily.__new__(NativeFiniteFamily)
    out._handle = handle; out._limits = limits; out._maximum = maximum
    return out


cdef NativeFinitePropagation _root(shared_ptr[FinitePropagationBridge] handle, vector[size_t] limits, size_t maximum):
    cdef NativeFinitePropagation out = NativeFinitePropagation.__new__(NativeFinitePropagation)
    out._handle = handle; out._limits = limits; out._maximum = maximum
    return out


cdef NativeFiniteExtension _extension(shared_ptr[FiniteExtensionBridge] handle, vector[size_t] limits, size_t maximum):
    cdef NativeFiniteExtension out = NativeFiniteExtension.__new__(NativeFiniteExtension)
    out._handle = handle; out._limits = limits; out._maximum = maximum
    return out


cdef class FiniteSearchCancellation:
    def __init__(self):
        if self._handle.get() != NULL: raise TypeError('finite cancellation owner is immutable')
        self._handle = shared_ptr[FiniteSearchStopBridge](new FiniteSearchStopBridge())

    cdef shared_ptr[FiniteSearchStopBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('finite cancellation owner is not initialized')
        return self._handle

    def cancel(self):
        cdef shared_ptr[FiniteSearchStopBridge] owner = self._pin()
        with nogil: owner.get().cancel()


cdef class PreparedFiniteSearch:
    cdef shared_ptr[FinitePreparedSearchBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('prepared finite search is not initialized')
        return self._handle

    @classmethod
    def prepare(cls, NativeFiniteSource source not None, distinct_relations=(), *, search_limits=_SEARCH, cancellation=None):
        cdef shared_ptr[FiniteSourceBridge] owner = source._pin()
        cdef vector[vector[size_t]] groups = _indices(distinct_relations)
        cdef vector[size_t] limits = _admission(search_limits, 3)
        cdef shared_ptr[FiniteSearchStopBridge] stop = _stop(cancellation)
        cdef FinitePreparedReply result
        cdef PreparedFiniteSearch prepared
        with nogil: result = FinitePreparedSearchBridge.prepare(owner, groups, limits, stop, source._limits, source._maximum)
        if result.prepared.get() != NULL:
            prepared = cls.__new__(cls); prepared._handle = result.prepared
            prepared._limits = source._limits; prepared._maximum = source._maximum
        else: prepared = None
        return {'status': ('ready', 'budget_exceeded', 'cancelled')[result.status],
                'prepared': prepared, 'work': _receipt(result.work)}

    def source(self):
        cdef shared_ptr[FinitePreparedSearchBridge] owner = self._pin()
        return _source(owner.get().source(), self._limits, self._maximum)

    def same_owner(self, PreparedFiniteSearch other not None):
        return bool(self._pin().get().same_owner(other._pin().get()[0]))

    def inspect(self, *, max_records=4096):
        cdef shared_ptr[FinitePreparedSearchBridge] owner = self._pin()
        cdef vector[vector[size_t]] groups = owner.get().groups(_natural(max_records))
        return {'distinct_relations': tuple(tuple(r) for r in groups),
                'retained_entries': owner.get().retained_entries(), 'preparation_work': _receipt(owner.get().work())}

    def global_section(self, fixed=(), *, search_limits=_SEARCH, cancellation=None):
        cdef shared_ptr[FinitePreparedSearchBridge] owner = self._pin()
        cdef vector[size_t] limits = _admission(search_limits, 3)
        cdef vector[FiniteAssignmentValue] rows = _fixed(fixed, limits[0])
        cdef shared_ptr[FiniteSearchStopBridge] stop = _stop(cancellation)
        cdef FiniteSectionReply result
        with nogil: result = owner.get().solve(rows, limits, stop)
        return _section(result)

    def propagate(self, fixed=(), *, search_limits=_SEARCH, cancellation=None):
        cdef shared_ptr[FinitePreparedSearchBridge] owner = self._pin()
        cdef vector[size_t] limits = _admission(search_limits, 3)
        cdef vector[FiniteAssignmentValue] rows = _fixed(fixed, limits[0])
        cdef shared_ptr[FiniteSearchStopBridge] stop = _stop(cancellation)
        cdef FinitePropagationReply result
        with nogil: result = owner.get().propagate(rows, limits, stop)
        return {'status': ('fixed_point', 'unsatisfiable', 'budget_exceeded', 'cancelled')[result.status],
                'snapshot': _root(result.snapshot, self._limits, self._maximum) if result.snapshot.get() != NULL else None,
                'work': _receipt(result.work)}


cdef class NativeFinitePropagation:
    cdef shared_ptr[FinitePropagationBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('finite propagation is not initialized')
        return self._handle

    def source(self):
        return _source(self._pin().get().source(), self._limits, self._maximum)

    def prepared(self):
        cdef PreparedFiniteSearch out = PreparedFiniteSearch.__new__(PreparedFiniteSearch)
        out._handle = self._pin().get().prepared(); out._limits = self._limits; out._maximum = self._maximum
        return out

    def same_owner(self, NativeFinitePropagation other not None):
        return bool(self._pin().get().same_owner(other._pin().get()[0]))

    def inspect(self, *, max_records=4096):
        cdef shared_ptr[FinitePropagationBridge] owner = self._pin()
        cdef vector[FiniteAssignmentValue] fixed = owner.get().fixed(_natural(max_records))
        return {'variable_count': owner.get().size(), 'fixed': tuple((v.grade, v.cell, v.symbol) for v in fixed),
                'retained_entries': owner.get().retained_entries(), 'propagation_work': _receipt(owner.get().work())}

    def select_domains(self, begin, count, *, max_values=4096):
        cdef shared_ptr[FinitePropagationBridge] owner = self._pin()
        cdef size_t start = _natural(begin), length = _natural(count), maximum = _natural(max_values)
        cdef vector[FiniteDomainRow] rows
        with nogil: rows = owner.get().domains(start, length, maximum)
        return tuple((r.grade, r.cell, tuple(r.values)) for r in rows)

    def continue_global_section(self, additional_fixed=(), *, search_limits=_SEARCH, cancellation=None):
        cdef shared_ptr[FinitePropagationBridge] owner = self._pin()
        cdef vector[size_t] limits = _admission(search_limits, 3)
        cdef vector[FiniteAssignmentValue] rows = _fixed(additional_fixed, limits[0])
        cdef shared_ptr[FiniteSearchStopBridge] stop = _stop(cancellation)
        cdef FiniteSectionReply result
        with nogil: result = owner.get().solve(rows, limits, stop)
        return _section(result)


cdef class NativeFiniteFamily:
    cdef shared_ptr[FiniteFamilyBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('finite family is not initialized')
        return self._handle

    @classmethod
    def projection(cls, source, declaration, *, limits=_LIMITS, max_bytes=256*1024*1024):
        _closed(declaration, ('key', 'revision', 'cells', 'window', 'fixed', 'distinct_relations'))
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef FiniteProjectionInfo d
        d.key = _natural(declaration['key']); d.revision = _natural(declaration['revision'])
        d.cells = _indices(declaration['cells']); d.relations = _indices(declaration['distinct_relations'])
        if d.cells.size() > bounds[0] or d.relations.size() > bounds[0]: raise ValueError('finite projection exceeds admission')
        d.fixed = _fixed(declaration['fixed'], bounds[0]); d.has_window = declaration['window'] is not None
        if d.has_window:
            for row in declaration['window']:
                if d.window.size() >= bounds[0]: raise ValueError('finite window exceeds admission')
                d.window.push_back(_symbols(row, bounds[1]))
        cdef shared_ptr[FiniteSourceBridge] raw
        cdef shared_ptr[FinitePreparedSearchBridge] prepared
        cdef shared_ptr[FinitePropagationBridge] root
        if isinstance(source, NativeFiniteSource): raw = (<NativeFiniteSource>source)._pin()
        elif isinstance(source, PreparedFiniteSearch): prepared = (<PreparedFiniteSearch>source)._pin()
        elif isinstance(source, NativeFinitePropagation): root = (<NativeFinitePropagation>source)._pin()
        else: raise TypeError('actual finite source, prepared search or propagation required')
        cdef shared_ptr[FiniteFamilyBridge] result
        with nogil: result = FiniteFamilyBridge.projection(raw, prepared, root, d, bounds, maximum)
        return _family(result, bounds, maximum)

    @classmethod
    def intersect_images(cls, NativeFiniteFamily left not None, NativeFiniteFamily right not None, comparison, *, limits=_LIMITS, max_bytes=256*1024*1024):
        _closed(comparison, ('key', 'revision', 'right_positions', 'right_symbols'))
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef FiniteComparisonInfo d
        d.key = _natural(comparison['key']); d.revision = _natural(comparison['revision'])
        d.positions = _admission(comparison['right_positions'], len(comparison['right_positions']))
        if d.positions.size() > bounds[0]: raise ValueError('finite comparison exceeds admission')
        for row in comparison['right_symbols']:
            if d.symbols.size() >= bounds[0]: raise ValueError('finite comparison exceeds admission')
            d.symbols.push_back(_symbols(row, bounds[1]))
        cdef shared_ptr[FiniteFamilyBridge] a = left._pin(), b = right._pin(), result
        with nogil: result = FiniteFamilyBridge.intersect(a, b, d, bounds, maximum)
        return _family(result, bounds, maximum)

    def source(self, leaf=0):
        return _source(self._pin().get().source(_natural(leaf)), self._limits, self._maximum)

    def propagation(self, leaf=0):
        cdef shared_ptr[FinitePropagationBridge] result = self._pin().get().propagation(_natural(leaf))
        return _root(result, self._limits, self._maximum) if result.get() != NULL else None

    def parent(self, *, right=False):
        if type(right) is not bool: raise TypeError('finite parent side must be bool')
        return _family(self._pin().get().parent(right), self._limits, self._maximum)

    def same_owner(self, NativeFiniteFamily other not None):
        return bool(self._pin().get().same_owner(other._pin().get()[0]))

    def declaration(self, leaf=0, *, max_records=4096):
        return _projection_info(self._pin().get().declaration(_natural(leaf), _natural(max_records)))

    def comparison(self, *, max_records=4096):
        cdef shared_ptr[FiniteFamilyBridge] owner = self._pin()
        cdef FiniteComparisonInfo d
        if not owner.get().has_comparison(): return None
        d = owner.get().comparison(_natural(max_records))
        return dict(key=d.key, revision=d.revision, right_positions=tuple(d.positions),
                    right_symbols=tuple(tuple(r) for r in d.symbols))

    def source_mapping(self, leaf=0, *, max_records=4096):
        cdef FiniteFamilyInfo value = self._pin().get().source_mapping(_natural(leaf), _natural(max_records))
        return dict(cells=tuple(tuple(r) for r in value.cells), symbols=tuple(tuple(r) for r in value.carriers))

    def inspect(self, *, max_records=4096):
        cdef FiniteFamilyInfo value = self._pin().get().info(_natural(max_records))
        return {'key': value.key, 'revision': value.revision, 'leaf_count': value.leaves,
                'retained_entries': value.retained_entries, 'cells': tuple(tuple(r) for r in value.cells),
                'carriers': tuple(tuple(r) for r in value.carriers), 'window': tuple(tuple(r) for r in value.window)}

    def extend(self, values, *, family_limits=_FAMILY, cancellation=None):
        cdef shared_ptr[FiniteFamilyBridge] owner = self._pin()
        cdef vector[size_t] limits = _admission(family_limits, 5)
        cdef vector[uint64_t] row = _symbols(values, limits[0])
        cdef shared_ptr[FiniteSearchStopBridge] stop = _stop(cancellation)
        cdef FiniteExtensionReply result
        with nogil: result = owner.get().extend(row, limits, stop)
        return {'status': _SOLVE_STATUS[result.status],
                'extension': _extension(result.extension, self._limits, self._maximum) if result.extension.get() != NULL else None,
                'decisive_leaf': result.decisive_leaf if result.has_decisive_leaf else None, 'work': _family_receipt(result.work)}

    def materialize(self, *, family_limits=_FAMILY, cancellation=None):
        cdef shared_ptr[FiniteFamilyBridge] owner = self._pin()
        cdef vector[size_t] limits = _admission(family_limits, 5)
        cdef shared_ptr[FiniteSearchStopBridge] stop = _stop(cancellation)
        cdef FiniteImageReply result
        cdef NativeFiniteImage image
        cdef NativeFiniteImageQueries queries
        with nogil: result = owner.get().materialize(limits, stop)
        if result.image.get() != NULL:
            image = NativeFiniteImage.__new__(NativeFiniteImage); image._handle = result.image
            image._limits = self._limits; image._maximum = self._maximum
        else: image = None
        queries = NativeFiniteImageQueries.__new__(NativeFiniteImageQueries); queries._handle = result.queries
        queries._limits = self._limits; queries._maximum = self._maximum
        return {'status': ('complete', 'budget_exceeded', 'cancelled')[result.status],
                'image': image, 'queries': queries, 'work': _family_receipt(result.work)}


cdef class NativeFiniteExtension:
    cdef shared_ptr[FiniteExtensionBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('finite extension is not initialized')
        return self._handle

    def family(self):
        return _family(self._pin().get().family(), self._limits, self._maximum)

    def inspect(self, *, max_records=4096):
        cdef shared_ptr[FiniteExtensionBridge] owner = self._pin()
        cdef vector[uint64_t] row = owner.get().tuple(_natural(max_records))
        return {'tuple': tuple(row), 'leaf_count': owner.get().leaf_count()}

    def field_size(self, leaf=0):
        return self._pin().get().field_size(_natural(leaf))

    def witness(self, leaf=0, *, begin=0, count=None, max_records=4096):
        cdef shared_ptr[FiniteExtensionBridge] owner = self._pin()
        cdef size_t index = _natural(leaf), start = _natural(begin), maximum = _natural(max_records)
        cdef size_t length = owner.get().field_size(index) if count is None else _natural(count)
        if count is None:
            if start > length: raise IndexError('finite field slice outside coordinates')
            length -= start
        cdef vector[FiniteAssignmentValue] rows
        with nogil: rows = owner.get().witness(index, start, length, maximum)
        return tuple((v.grade, v.cell, v.symbol) for v in rows)


cdef class NativeFiniteImage:
    cdef shared_ptr[FiniteImageBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('finite image is not initialized')
        return self._handle

    def family(self):
        return _family(self._pin().get().family(), self._limits, self._maximum)

    def inspect(self):
        cdef shared_ptr[FiniteImageBridge] owner = self._pin()
        return {'row_count': owner.get().size(), 'query_count': owner.get().query_count()}

    def pruning(self, *, max_values=4096):
        """Inspect bounded canonical enumeration evidence without expanding root fields."""
        cdef shared_ptr[FiniteImageBridge] owner = self._pin()
        cdef size_t maximum = _natural(max_values)
        cdef FinitePruningInfo info
        with nogil: info = owner.get().pruning(maximum)
        return _pruning_info(info)

    def pruning_root(self, index):
        """Retain an actual per-leaf domain owner, or None for a rootless status."""
        cdef shared_ptr[FiniteImageBridge] owner = self._pin()
        cdef size_t selected = _natural(index)
        cdef shared_ptr[FinitePropagationBridge] root
        with nogil: root = owner.get().pruning_root(selected)
        return _root(root, self._limits, self._maximum) if root.get() != NULL else None

    def classify_equalities(self, claims, *, max_checks=64000000):
        """Classify conjunctions over this complete joint image, relative to its declared view."""
        if not isinstance(claims, (tuple, list)):
            raise TypeError('finite claims require a sequence of equality conjunctions')
        cdef size_t maximum = _natural(max_checks)
        if len(claims) > maximum: raise ValueError('finite claims exceed admission')
        cdef vector[vector[FiniteAssignmentValue]] rows
        cdef vector[FiniteAssignmentValue] row
        cdef size_t used = len(claims)
        for claim in claims:
            if not isinstance(claim, (tuple, list)):
                raise TypeError('finite claim requires a sequence of grade/cell/symbol triples')
            if len(claim) > maximum - used: raise ValueError('finite claims exceed admission')
            used += len(claim)
            row = _fixed(claim, len(claim))
            rows.push_back(row)
        cdef shared_ptr[FiniteImageBridge] owner = self._pin()
        cdef vector[FiniteClaimVerdict] result
        with nogil: result = owner.get().classify_equalities(rows, maximum)
        statuses = ('entailed', 'refuted', 'alternatives', 'inconsistent_family', 'unsupported')
        return tuple(dict(status=statuses[v.status], satisfying=v.satisfying, refuting=v.refuting) for v in result)

    def analyze_question(self, answer_cells, observation_cells=(), *, max_checks=64000000, cancellation=None):
        """Analyze exact answers and whether a declared observation determines them.

        Both maps are projections onto actual cells in this complete joint image.
        An obstruction is a pair of rows agreeing on observation cells but
        disagreeing on answer cells. The original whole family remains owned.
        """
        cdef vector[vector[size_t]] q, m
        cdef vector[size_t] row
        for requested, is_question in ((answer_cells, True), (observation_cells, False)):
            if not isinstance(requested, (tuple, list)):
                raise TypeError('question and observation maps require cell sequences')
            if len(requested) > self._limits[0]: raise ValueError('question exceeds coordinate admission')
            for cell in requested:
                if not isinstance(cell, (tuple, list)) or len(cell) != 2:
                    raise ValueError('question cells require grade/index pairs')
                row = _admission(cell, 2)
                if is_question: q.push_back(row)
                else: m.push_back(row)
        cdef shared_ptr[FiniteImageBridge] owner = self._pin()
        cdef size_t maximum = _natural(max_checks)
        cdef shared_ptr[FiniteSearchStopBridge] stop = _stop(cancellation)
        cdef FiniteQuestionInfo result
        with nogil: result = owner.get().analyze_question(q, m, maximum, stop)
        complete = result.status < 3
        return dict(status=('unique', 'alternatives', 'inconsistent_family', 'unsupported', 'budget_exceeded', 'cancelled')[result.status],
            answer=tuple(result.answer) if result.has_answer else None,
            answer_count=result.answers if complete else None,
            observation_count=result.observations if complete else None,
            max_remaining_answers=result.max_answers if complete else None,
            observation_determines_answer=bool(result.max_answers == 1) if complete and result.answers else None,
            obstruction=(result.left, result.right) if result.has_obstruction else None,
            checks=result.checks, retained_values=result.retained_values)

    def proof_archive(self, *, max_bytes=67108864):
        """Serialize the full unprepared projection, witnesses and query ledger natively.

        The packet is compared during fresh replay; it does not grant execution authority.
        Intersections and propagated roots require a separate replay contract.
        """
        cdef shared_ptr[FiniteImageBridge] owner = self._pin()
        cdef size_t maximum = _natural(max_bytes)
        cdef string result
        with nogil: result = owner.get().proof_archive(maximum)
        return <bytes>result

    def row(self, index):
        return _extension(self._pin().get().row(_natural(index)), self._limits, self._maximum)

    def select_queries(self, begin, count, *, max_values=4096):
        cdef shared_ptr[FiniteImageBridge] owner = self._pin()
        cdef size_t start = _natural(begin), length = _natural(count), maximum = _natural(max_values)
        cdef vector[RelationalFiniteImageQuery] rows
        with nogil: rows = owner.get().queries(start, length, maximum)
        return _query_rows(rows)


cdef class NativeFiniteImageQueries:
    cdef shared_ptr[FiniteImageQueriesBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('finite query receipts are not initialized')
        return self._handle

    def proof_archive(self, *, max_bytes=67108864):
        """Retain native query receipts even when image materialization was interrupted."""
        cdef shared_ptr[FiniteImageQueriesBridge] owner = self._pin()
        cdef size_t maximum = _natural(max_bytes)
        cdef string result
        with nogil: result = owner.get().proof_archive(maximum)
        return <bytes>result

    def family(self):
        return _family(self._pin().get().family(), self._limits, self._maximum)

    def pruning(self, *, max_values=4096):
        """Inspect complete or interrupted pruning evidence under an explicit bound."""
        cdef shared_ptr[FiniteImageQueriesBridge] owner = self._pin()
        cdef size_t maximum = _natural(max_values)
        cdef FinitePruningInfo info
        with nogil: info = owner.get().pruning(maximum)
        return _pruning_info(info)

    def pruning_root(self, index):
        """Retain a reached canonical root without granting image completeness."""
        cdef shared_ptr[FiniteImageQueriesBridge] owner = self._pin()
        cdef size_t selected = _natural(index)
        cdef shared_ptr[FinitePropagationBridge] root
        with nogil: root = owner.get().pruning_root(selected)
        return _root(root, self._limits, self._maximum) if root.get() != NULL else None

    def inspect(self):
        return {'query_count': self._pin().get().query_count()}

    def select(self, begin, count, *, max_values=4096):
        cdef shared_ptr[FiniteImageQueriesBridge] owner = self._pin()
        cdef size_t start = _natural(begin), length = _natural(count), maximum = _natural(max_values)
        cdef vector[RelationalFiniteImageQuery] rows
        with nogil: rows = owner.get().queries(start, length, maximum)
        return _query_rows(rows)


cdef FiniteSourceMapInfo _source_map(value, size_t maximum):
    _closed(value, ('cells', 'occurrences'))
    cdef FiniteSourceMapInfo out
    cdef vector[size_t] row
    cdef size_t used = 0
    for name in ('cells', 'occurrences'):
        for values in value[name]:
            used += 1
            if len(values) > maximum or used > maximum - len(values): raise ValueError('finite source map exceeds admission')
            used += len(values)
            row = _admission(values, len(values))
            if name == 'cells': out.cells.push_back(row)
            else: out.occurrences.push_back(row)
    return out


cdef object _source_map_info(FiniteSourceMapInfo value):
    return dict(cells=tuple(tuple(r) for r in value.cells), occurrences=tuple(tuple(r) for r in value.occurrences))


cdef class NativeFiniteGluing:
    cdef shared_ptr[FiniteGluingBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('finite gluing is not initialized')
        return self._handle

    def __init__(self, common_archive, NativeFiniteSource left not None, NativeFiniteSource right not None,
                 NativeFiniteSource global_source not None, source_maps, symbol_maps, *, limits=_LIMITS, max_bytes=256*1024*1024):
        if self._loading or self._handle.get() != NULL: raise TypeError('finite gluing owner is immutable')
        _closed(source_maps, ('key', 'revision', 'common_left', 'common_right', 'left_glued', 'right_glued'))
        _closed(symbol_maps, ('key', 'revision', 'left', 'right'))
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string common = _image(common_archive, maximum)
        cdef FiniteGluingInfo d
        d.key = _natural(source_maps['key']); d.revision = _natural(source_maps['revision'])
        d.symbol_key = _natural(symbol_maps['key']); d.symbol_revision = _natural(symbol_maps['revision'])
        d.common_left = _source_map(source_maps['common_left'], bounds[1])
        d.common_right = _source_map(source_maps['common_right'], bounds[1])
        d.left_glued = _source_map(source_maps['left_glued'], bounds[1])
        d.right_glued = _source_map(source_maps['right_glued'], bounds[1])
        for row in symbol_maps['left']:
            if d.left_symbols.size() >= bounds[0]: raise ValueError('finite symbol map exceeds admission')
            d.left_symbols.push_back(_symbols(row, bounds[1]))
        for row in symbol_maps['right']:
            if d.right_symbols.size() >= bounds[0]: raise ValueError('finite symbol map exceeds admission')
            d.right_symbols.push_back(_symbols(row, bounds[1]))
        cdef shared_ptr[FiniteSourceBridge] a = left._pin(), b = right._pin(), c = global_source._pin()
        self._loading = True
        try:
            with nogil: self._handle = shared_ptr[FiniteGluingBridge](new FiniteGluingBridge(common, a, b, c, d, bounds, maximum))
            self._limits = bounds; self._maximum = maximum
        finally: self._loading = False

    def same_owner(self, NativeFiniteGluing other not None):
        return bool(self._pin().get().same_owner(other._pin().get()[0]))

    def source(self, side):
        if side not in ('left', 'right', 'global'): raise ValueError('finite gluing side must be left, right or global')
        return _source(self._pin().get().source(('left', 'right', 'global').index(side)), self._limits, self._maximum)

    def inspect(self):
        cdef shared_ptr[FiniteGluingBridge] owner = self._pin()
        return {'retained_entries': owner.get().retained_entries(), 'preparation_steps': owner.get().preparation_steps()}

    def maps(self, *, max_records=4096):
        cdef FiniteGluingInfo d = self._pin().get().maps(_natural(max_records))
        return {'source_maps': dict(key=d.key, revision=d.revision,
            common_left=_source_map_info(d.common_left), common_right=_source_map_info(d.common_right),
            left_glued=_source_map_info(d.left_glued), right_glued=_source_map_info(d.right_glued)),
            'symbol_maps': dict(key=d.symbol_key, revision=d.symbol_revision,
                left=tuple(tuple(r) for r in d.left_symbols), right=tuple(tuple(r) for r in d.right_symbols))}

    def glue(self, NativeFiniteExtension left not None, NativeFiniteExtension right not None, *, left_leaf=0, right_leaf=0,
             family_limits=_FAMILY, cancellation=None):
        cdef shared_ptr[FiniteGluingBridge] owner = self._pin()
        cdef shared_ptr[FiniteExtensionBridge] a = left._pin(), b = right._pin()
        cdef size_t ai = _natural(left_leaf), bi = _natural(right_leaf)
        cdef vector[size_t] limits = _admission(family_limits, 5)
        cdef shared_ptr[FiniteSearchStopBridge] stop = _stop(cancellation)
        cdef FiniteGluingReply result
        cdef NativeFiniteGluedSection section
        with nogil: result = owner.get().glue(a, ai, b, bi, limits, stop)
        if result.section.get() != NULL:
            section = NativeFiniteGluedSection.__new__(NativeFiniteGluedSection); section._handle = result.section
            section._limits = self._limits; section._maximum = self._maximum
        else: section = None
        return {'status': ('compatible', 'incompatible', 'refuted_by_global_source', 'budget_exceeded', 'cancelled')[result.status],
                'section': section, 'work': _family_receipt(result.work)}


cdef class NativeFiniteGluedSection:
    cdef shared_ptr[FiniteGluedSectionBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('finite glued section is not initialized')
        return self._handle

    def gluing(self):
        cdef NativeFiniteGluing out = NativeFiniteGluing.__new__(NativeFiniteGluing)
        out._handle = self._pin().get().gluing(); out._limits = self._limits; out._maximum = self._maximum
        return out

    def parent(self, *, right=False):
        if type(right) is not bool: raise TypeError('finite parent side must be bool')
        return _extension(self._pin().get().parent(right), self._limits, self._maximum)

    def inspect(self):
        cdef shared_ptr[FiniteGluedSectionBridge] owner = self._pin()
        return {'variable_count': owner.get().size(), 'left_leaf': owner.get().leaf(False), 'right_leaf': owner.get().leaf(True)}

    def select(self, begin, count, *, max_records=4096):
        cdef shared_ptr[FiniteGluedSectionBridge] owner = self._pin()
        cdef size_t start = _natural(begin), length = _natural(count), maximum = _natural(max_records)
        cdef vector[FiniteAssignmentValue] rows
        with nogil: rows = owner.get().field(start, length, maximum)
        return tuple((v.grade, v.cell, v.symbol) for v in rows)
