"""Canonical native literal inputs and complete fixed profile admission."""
from importlib import import_module

def _native():
    return import_module("rexgraph_cpp")

from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from weakref import WeakKeyDictionary

from rexgraph.model_state import freeze_tree, thaw_tree
from rexgraph.value_codec import pack_value, unpack_value
from rexgraph.cpp_relational_field_evidence_admission import inspect_available_field_learning_model


from rexgraph.conditioned_source import (
    ConditionedLanguageSource, conditioned_language_source, restore_conditioned_language_source,
)
from rexgraph.conditioned_source import PAIR_ORDER

PROFILE = 'rexgraph.conditioned-four-use-field-application.v1'
LIMITS = (4096, 16000000, 64000000, 16384)
MAX_BYTES = 16*1024*1024
POLICY = {'limits': LIMITS, 'max_bytes': MAX_BYTES}
PATTERN_PACKET = pack_value({'profile': PROFILE, 'pairs': PAIR_ORDER,
    'upper_basis': ('head-prefix', 'head-suffix', 'dependent-prefix', 'dependent-suffix'),
    'lower_basis': ('prospective-response-0', 'prospective-response-1'),
    'parameterization': 'shared2x4 dependent-flag map; explicit zero head/context maps'})
PATTERN_SHA256 = sha256(PATTERN_PACKET).hexdigest()
_inputs_lock = RLock()
_admitted_inputs = WeakKeyDictionary()


def matrix(rows, columns, entries=()):
    return {'shape': (rows, columns), 'entries': tuple(entries)}


def identity(size):
    return matrix(size, size, ((i, i, 1) for i in range(size)))


def program_declaration():
    """Declare all original stalks and flags; coefficients execute natively."""
    occurrences = []
    for slot in range(28):
        entries = ()
        occurrences.append({'grade': 1, 'slot': slot, 'role': 'coefficient_independent',
                            'map': matrix(2, 4, entries)})
    return {'profile': 'rexgraph.relational-field-program.v1', 'key': 83001, 'revision': 1,
            'differential': True,
            'stalks': tuple({'cell': (grade, i), 'basis': 84000+grade,
                            'directions': tuple(range(85000+10*grade, 85000+10*grade+width)),
                            'metric': identity(width)}
                           for grade, count, width in ((0, 4, 2), (1, 13, 4)) for i in range(count)),
            'occurrences': tuple(occurrences),
            'nodes': ({'key': 1, 'kind': 'input', 'grade': 1},
                      {'key': 2, 'kind': 'boundary', 'grade': 0, 'input': 1})}


def parameters():
    variables = tuple((5+2*pair, row, column) for pair in range(12) for row in range(2) for column in range(4))
    return {'profile': 'rexgraph.relational-field-parameters.v1', 'key': 86001, 'revision': 1, 'node': 2,
            'variables': variables, 'parameters': tuple(range(87001, 87009)),
            'basis': matrix(96, 8, ((i, 4*row+column, 1) for i, (_, row, column) in enumerate(variables)))}


def _proof_packet(origin):
    info = origin.inspect()
    return {'key': info['key'], 'role': info['role'], 'payload': origin.payload(), 'receipt': origin.receipt()}


def _restore_proofs(rows, context):
    result = []
    for row in rows:
        if set(row) != {'key', 'role', 'payload', 'receipt'}:
            raise ValueError('complete literal evidence origin receipt required')
        origin = context.require(row['key'], row['role'], row['payload'], max_bytes=MAX_BYTES)
        if origin.receipt() != row['receipt']:
            raise ValueError('actual proof source publication differs from saved origin')
        result.append(origin)
    return tuple(result)


def _recheck(proofs, context):
    for origin in proofs:
        context.recheck(origin)


def _accession(key):
    return {'profile': 'rexgraph.relational-input-accession.v1', 'key': 90000+key, 'node': 1, 'grade': 1,
            'stalks': tuple(s for s in program_declaration()['stalks'] if s['cell'][0] == 1),
            'instances': {'cells': (tuple(range(4)), tuple(range(13))), 'occurrences': ((), tuple(range(28)))},
            'accession': identity(52), 'reconstruction': identity(52), 'isometric': True}


def _grammar(source):
    return tuple(thaw_tree(source.declaration["observations"]["predicates"]))


def _literal_declaration(source, key):
    uses = source.parent.parent.declaration['uses']; coordinates = []
    for index, (head, dependent) in enumerate(((0, 1), *PAIR_ORDER)):
        for column, (use_index, predicate_index) in enumerate(((head, 0), (head, 1), (dependent, 0), (dependent, 1))):
            use = uses[use_index]; predicate = _grammar(source)[predicate_index]
            coordinates.append({'key': 1+4*index+column, 'cell': (1, index), 'direction': 85010+column,
                'use': use['key'], 'span_index': 0, 'span': use['spans'][0],
                'kind': predicate['kind'], 'operand': predicate['argument']})
    return {'profile': 'rexgraph.relational-literal-input.v1', 'key': 91000+key, 'revision': 1,
            'use': 88000+key, 'coordinates': tuple(coordinates)}


def _checked_source(value):
    if (not isinstance(value, ConditionedLanguageSource)
            or value.declaration['identity']['profile'] != 'literal-context-v2' or value.contexts or value.symbolic
            or value.declaration['identity']['pattern_sha256'] != PATTERN_SHA256
            or tuple(p['kind'] for p in _grammar(value)) != ('byte-prefix', 'byte-suffix')):
        raise ValueError('first field profile requires frozen literal grammar; additional context/morphology requires a declared extension')
    checked = conditioned_language_source(value.parent, value.declaration['request'],
        max_bytes=value.declaration['policy']['max_bytes'], max_records=value.declaration['policy']['max_records'])
    if checked.archive() != value.archive():
        raise ValueError('source declaration differs from original owners')
    uses = value.parent.parent.declaration['uses']
    if len(uses) != 4 or any(len(u['spans']) != 1 for u in uses):
        raise ValueError('initial execution chart requires one supplied span per original use')
    return value.parent.model_source


def _asset(value, key):
    original = value.parent.model_source; group = value.parent.parent.declaration['relation_key']
    uses = value.parent.parent.declaration['uses']; keys = tuple(u['key'] for u in uses)
    if 88000+key in keys:
        raise ValueError('whole-chart accession use collides with original use')
    rows = tuple({'key': u['key'], 'cells': ((0, u['key']), (1, group)),
                  'occurrences': ((1, i+1),), 'spans': u['spans']} for i, u in enumerate(uses))
    rows += ({'key': 88000+key, 'cells': tuple((0, k) for k in keys)+tuple((1, group+i) for i in range(13)),
              'occurrences': tuple((1, i+1) for i in range(28)), 'spans': tuple(u['spans'][0] for u in uses)},)
    return _native().NativeInputUseAsset(original.archive, {'profile': 'rexgraph.relational-input-use-asset.v1',
        'key': 89000+key, 'channel': 89050, 'encoding': value.parent.parent.declaration['encoding'],
        'bytes': value.parent.parent.declaration['raw'], 'uses': rows}, **POLICY)


def _source_dependencies(value):
    original = value.parent.model_source
    payloads = {'literal': value.parent.asset.archive(), 'structure': original.archive, 'predicates': pack_value(list(_grammar(value)))}
    kinds = {'literal': 'literal', 'structure': 'structure', 'predicates': 'predicate-grammar'}
    rows = value.declaration['availability']['dependencies']
    if len(rows) != 3 or {d['key'] for d in rows} != set(payloads):
        raise ValueError('complete literal, structure and grammar dependencies required')
    result = []
    for name in payloads:
        dep = next(d for d in rows if d['key'] == name); raw = payloads[name]
        if (dep['kind'] != kinds[name] or sha256(raw).hexdigest() != dep['payload_sha256']
                or dep['namespace'] != original.identity['namespace']
                or not original.identity['first_sequence'] <= dep['sequence'] <= original.identity['sequence']):
            raise ValueError('actual input producer publication differs from declared source dependency')
        result.append(('input-asset' if name == 'literal' else 'interface', raw, dep['sequence']))
    return tuple(result)


def _expected_proofs(source, literal, row, key):
    native = _native().field_input_origin_payloads(literal.program(), {k: row[k] for k in ('asset', 'accession', 'use', 'field')}, **POLICY)
    end = source.parent.model_source.identity['sequence']
    result = [(role, native[name], end) for name, role in zip(
        ('asset', 'value', 'accession', 'reconstruction'),
        ('input-asset', 'input-value', 'input-accession', 'reconstruction'), strict=True)]
    result += [('interface', literal.archive(), end), ('interface', source.archive(), end)]
    result += list(_source_dependencies(source))
    return tuple((1000+20*key+i, *values) for i, values in enumerate(result))


@dataclass(frozen=True, eq=False)
class PreparedConditionedInput:
    source: object
    literal: object
    row: object
    proofs: tuple

    def __post_init__(self):
        object.__setattr__(self, 'row', freeze_tree(self.row))
        object.__setattr__(self, 'proofs', tuple(self.proofs))

    def _cached(self):
        with _inputs_lock:
            value = _admitted_inputs.get(self)
        if value is not None and all(a is b for a,b in zip(value[:4], (self.source, self.literal, self.row, self.proofs), strict=True)):
            return value
        return None

    def admit(self, context, program_archive, *, grammar=None):
        cached = self._cached()
        if cached is not None:
            if cached[4] != program_archive or (grammar is not None and cached[5] != tuple(thaw_tree(grammar))):
                raise ValueError('input native program or frozen grammar differs')
            _recheck(self.proofs, context)
            return thaw_tree(self.row)
        original = _checked_source(self.source)
        if grammar is not None and _grammar(self.source) != tuple(thaw_tree(grammar)):
            raise ValueError("input predicate grammar differs from the frozen model")
        if not isinstance(self.literal, _native().NativeLiteralInput) or self.literal.program().archive() != program_archive:
            raise ValueError('actual source-derived literal input for this program required')
        declaration = self.literal.declaration(); key = declaration['key']-91000
        if (not 1 <= key <= 24 or declaration != _literal_declaration(self.source, key)
                or thaw_tree(self.row['accession']) != _accession(key)
                or self.literal.asset().archive() != _asset(self.source, key).archive()):
            raise ValueError('literal source, grammar, pattern or accession differs from the fixed input profile')
        actual = self.literal.input().inspect()
        fixed = _native().NativeInputAccession(self.literal.asset(), self.literal.program(), _accession(key), **POLICY)
        expected_input = fixed.admit(declaration['use'], actual['field']).inspect()
        if actual != expected_input:
            raise ValueError('retained native accession differs from the complete fixed input profile')
        if any(self.row[k] != actual[k] for k in ('asset', 'use', 'field')):
            raise ValueError('caller input field differs from source-derived native observation')
        expected = _expected_proofs(self.source, self.literal, thaw_tree(self.row), key)
        if len(self.proofs) != len(expected):
            raise ValueError('complete original source, grammar and literal proof set required')
        for origin, (proof_key, role, payload, sequence) in zip(self.proofs, expected, strict=True):
            info = origin.inspect()
            if (info['key'] != proof_key or info['role'] != role or origin.payload() != payload
                    or info['namespace'] != original.identity['namespace'] or info['sequence'] != sequence):
                raise ValueError('literal proof differs from its complete original source dependency')
        _recheck(self.proofs, context)
        if dict(self.row['origins']) != dict(zip(('asset', 'value', 'accession', 'reconstruction'), range(1000+20*key, 1004+20*key), strict=True)):
            raise ValueError('generic input origins differ from retained literal proof')
        packet = self.archive()
        with _inputs_lock:
            _admitted_inputs[self] = (self.source, self.literal, self.row, self.proofs, program_archive, _grammar(self.source), packet)
        return thaw_tree(self.row)

    def archive(self):
        cached = self._cached()
        if cached is not None:
            return cached[6]
        packet = pack_value({'profile': PROFILE, 'source': self.source.archive(), 'literal': self.literal.archive(),
                           'row': thaw_tree(self.row), 'proofs': tuple(_proof_packet(p) for p in self.proofs)})
        if len(packet) > MAX_BYTES:
            raise ValueError('conditioned input exceeds declared byte admission')
        return packet

    @classmethod
    def from_archive(cls, packet, context, *, grammar=None):
        if len(packet) > MAX_BYTES:
            raise ValueError('conditioned input exceeds declared byte admission')
        row = unpack_value(packet)
        if set(row) != {'profile', 'source', 'literal', 'row', 'proofs'} or row['profile'] != PROFILE:
            raise ValueError('closed conditioned input packet required')
        source = restore_conditioned_language_source(row['source'], max_bytes=MAX_BYTES)
        literal = _native().NativeLiteralInput.from_archive(row['literal'], **POLICY)
        actual = literal.input().inspect()
        expected = {'asset': actual['asset'], 'accession': row['row']['accession'],
                    'use': actual['use'], 'field': actual['field'], 'origins': row['row']['origins']}
        if row['row'] != expected:
            raise ValueError('input field or source asset differs from recomputed native literal owner')
        proofs = _restore_proofs(row['proofs'], context)
        result = cls(source, literal, freeze_tree(row['row']), proofs)
        result.admit(context, literal.program().archive(), grammar=grammar)
        if result.archive() != packet:
            raise ValueError('conditioned input packet is not canonical')
        return result


def check_conditioned_model(state, source, inputs, context, *, available=None, grammar=None):
    available = inspect_available_field_learning_model(state, source, execution_context=context) if available is None else available
    return check_conditioned_evidence(available, source, inputs, context, grammar=grammar,
                                      program_archive=state.configuration['program_archive'])


def check_conditioned_evidence(available, source, inputs, context, *, grammar, program_archive=None):
    available.admit(context)
    evidence = available.evidence()
    actual_program = evidence.program().archive()
    if program_archive is None:
        program_archive = actual_program
    if actual_program != program_archive:
        raise ValueError('evidence actual program differs from its complete declaration')
    fixed = _native().PreparedAccessionFieldProgram(_native().PreparedFieldProgram(source.archive, program_declaration(), **POLICY).archive(), **POLICY)
    root = evidence
    while root.parent() is not None:
        root = root.parent()
    if (program_archive != fixed.archive()
            or root.archive() != _native().PreparedFieldEvidence(fixed, parameters(), **POLICY).archive()):
        raise ValueError('model differs from fixed zero-origin whole-map pattern and parameter family')
    if evidence.identity()['observation_count'] != len(inputs):
        raise ValueError('complete training literal proof obligations required')
    for index, value in enumerate(inputs):
        value.admit(context, program_archive, grammar=grammar)
        observation = evidence.observation(index)
        actual = value.literal.program().execute((value.literal.input(),)).archive()
        if (observation['key'] != value.literal.declaration()['key']-91000
                or observation['input'] != actual
                or observation['target_asset'] != value.row['asset']
                or observation['target_use'] != value.row['use']
                or observation['observed'] != tuple((0, i) for i in range(4))):
            raise ValueError('stored training input differs from complete source-derived literal execution')
    return available


def prepare_conditioned_input(value, program, *, key, publish):
    if not isinstance(value, ConditionedLanguageSource) or type(key) is not int or not 1 <= key <= 24:
        raise ValueError('bounded actual source-conditioned input required')
    original = _checked_source(value)
    dependencies = _source_dependencies(value)
    base = 1000+20*key
    asset = _asset(value, key); whole_use = 88000+key
    accession = _accession(key)
    literal = _native().NativeLiteralInput(_native().NativeInputAccession(asset, program, accession, **POLICY),
        _literal_declaration(value, key), **POLICY)
    actual = literal.input().inspect()
    row = {'asset': actual['asset'], 'accession': accession, 'use': whole_use, 'field': actual['field']}
    base = 1000+20*key; payloads = _native().field_input_origin_payloads(program, row, **POLICY)
    origin_keys = {}
    proofs = []
    for offset, (name, role) in enumerate(zip(('asset', 'value', 'accession', 'reconstruction'),
                                            ('input-asset', 'input-value', 'input-accession', 'reconstruction'), strict=True)):
        origin_keys[name] = base+offset
        proofs.append(publish(base+offset, original, role, payloads[name]))
    proofs.extend((publish(base+4, original, 'interface', literal.archive()),
                   publish(base+5, original, 'interface', value.archive())))
    for offset, (role, raw, sequence) in enumerate(dependencies, 6):
        proofs.append(publish(base+offset, original, role, raw, cutoff=sequence))
    row['origins'] = origin_keys
    return PreparedConditionedInput(value, literal, freeze_tree(row), tuple(proofs))
