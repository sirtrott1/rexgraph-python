"""Declared native evidence families with complete external source admission."""
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from weakref import WeakKeyDictionary

from rexgraph import cpp_relational_field_evidence_admission as base
from rexgraph.conditioned_field import (
    LIMITS, MAX_BYTES, POLICY, PreparedConditionedInput, check_conditioned_evidence,
    _proof_packet, _restore_proofs, _recheck, _accession,
)
from rexgraph.cpp_conditioned_field_program import _closed, _grammar
from rexgraph.cpp_relational_field_admission import _context
from rexgraph.cpp_relational_field_program import _publication, _space
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelState, ModelOutput, freeze_tree, thaw_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import pack_value, unpack_value

PROFILE = 'rexgraph.conditioned-field-evidence-family.v1'
MEMBER = 'rexgraph.conditioned-field-evidence-member.v1'
QUERY = 'rexgraph.conditioned-field-evidence-family-query.v1'
SCHEMA = 'rexgraph.conditioned-joint-field-family.v1'
METHOD = 'exact-conditioned-joint-field-family-v1'
ADAPTER = 'cpp_conditioned_field_family'
PROVIDER = {'name': 'rexgraph.cpp.conditioned_field_family', 'version': '1'}
_lock = RLock()
_members = WeakKeyDictionary()
_models = WeakKeyDictionary()
_outputs = WeakKeyDictionary()
_provider = None


def _binding():
    module = base._binding()
    if not hasattr(module, 'PreparedFieldFamily'):
        raise ImportError('conditioned families require the native field family provider')
    return module


def _receipts():
    module = _binding()
    return {'family': module.field_family_build_info(), 'literal': module.field_interface_build_info(),
            'evidence': module.field_evidence_admission_build_info()}


def _bytes(value):
    if type(value) is not bytes or not 0 < len(value) <= MAX_BYTES:
        raise ValueError('bounded immutable family packet required')
    return value


@dataclass(frozen=True, eq=False)
class PreparedConditionedFamilyMember:
    source: object
    available: object
    inputs: tuple
    grammar: object
    provenance: bytes = b''

    def __post_init__(self):
        object.__setattr__(self, 'inputs', tuple(self.inputs))
        object.__setattr__(self, 'grammar', freeze_tree(_grammar(self.grammar)))
        if type(self.provenance) is not bytes or len(self.provenance) > MAX_BYTES:
            raise ValueError('bounded passive member provenance required')

    def admit(self, context):
        module = _binding(); context = _context(module, context)
        with _lock: known = _members.get(self)
        if known is None:
            if (not isinstance(self.source, module.NativeSource)
                    or not isinstance(self.available, module.PreparedAvailableFieldEvidence)
                    or any(not isinstance(v, PreparedConditionedInput) for v in self.inputs)
                    or self.available.admission() != POLICY):
                raise TypeError('actual source, available evidence and literal proof owners required')
            reference = CppSourceReference(self.source)
            program = self.available.evidence().program()
            _publication(program, reference)
            check_conditioned_evidence(self.available, self.source, self.inputs, context, grammar=self.grammar)
            dependencies = (reference,)+base._dependencies(module, self.available, LIMITS)
            dependencies += tuple(CppSourceReference(item.source.parent.model_source) for item in self.inputs for _ in item.proofs)
            known = (program.archive(), dependencies)
            with _lock: _members[self] = known
        self.available.admit(context)
        for item in self.inputs: item.admit(context, known[0], grammar=self.grammar)
        return known

    def archive(self):
        packet = pack_value({'profile': MEMBER, 'source': self.source.archive,
            'reference': CppSourceReference(self.source).as_record(), 'available': self.available.archive(),
            'inputs': tuple(item.archive() for item in self.inputs), 'grammar': thaw_tree(self.grammar),
            'provenance': self.provenance})
        return _bytes(packet)

    @classmethod
    def from_archive(cls, packet, context, *, source=None):
        row = unpack_value(_bytes(packet))
        _closed(row, ('profile', 'source', 'reference', 'available', 'inputs', 'grammar', 'provenance'))
        if row['profile'] != MEMBER: raise ValueError('versioned complete family member required')
        module = _binding()
        if source is None: source = module.NativeSource(_bytes(row['source']), max_bytes=MAX_BYTES)
        elif source.archive != row['source']: raise ValueError('family primary source bytes differ')
        source_from_record(row['reference'], allow_cpp=True).bind(source)
        available = module.PreparedAvailableFieldEvidence.from_archive(_bytes(row['available']), context, **POLICY)
        inputs = tuple(PreparedConditionedInput.from_archive(p, context, grammar=row['grammar']) for p in row['inputs'])
        value = cls(source, available, inputs, row['grammar'], row['provenance'])
        value.admit(context)
        if value.archive() != packet: raise ValueError('noncanonical family member packet')
        return value


def conditioned_family_member(source, *, available_evidence, inputs, grammar, execution_context, provenance=b''):
    """Admit a new typed member from native evidence, preserving passive provenance."""
    result = PreparedConditionedFamilyMember(source, available_evidence, inputs, grammar, provenance)
    result.admit(execution_context)
    return result


def conditioned_family_member_from_model(state, source, *, execution_context):
    from rexgraph.cpp_conditioned_field_program import prepare_conditioned_field_model
    _, _, available, inputs = prepare_conditioned_field_model(state, source, execution_context=execution_context)
    return conditioned_family_member(source, available_evidence=available, inputs=inputs,
        grammar=state.configuration['grammar'], execution_context=execution_context,
        provenance=pack_value(pack_model(state)))


def _declaration(value):
    _closed(value, ('profile', 'key', 'revision', 'interfaces'))
    if (value['profile'] != 'rexgraph.relational-field-family.v1'
            or any(type(value[k]) is not int or not 0 < value[k] < 1 << 64 for k in ('key', 'revision'))
            or len(value['interfaces']) > LIMITS[0]):
        raise ValueError('identified bounded native family declaration required')
    for row in value['interfaces']:
        _closed(row, ('key', 'revision', 'left', 'right', 'common', 'common_cutoff', 'common_left', 'common_right', 'left_views', 'right_views'))
        _bytes(row['common'])
        if any(type(row[k]) is not int or not 0 <= row[k] < 1 << 64 for k in ('key', 'revision', 'left', 'right', 'common_cutoff')):
            raise ValueError('exact family interface identities and indices required')
    if len(pack_value(thaw_tree(value))) > MAX_BYTES: raise ValueError('family declaration exceeds admission')
    return freeze_tree(value)


def _interface_proofs(declaration, proofs, context):
    if len(proofs) != len(declaration['interfaces']): raise ValueError('one complete proof pair per declared interface required')
    dependencies = []
    for row, pair in zip(declaration['interfaces'], proofs, strict=True):
        if len(pair) != 2: raise ValueError('interface declaration and common publication proofs required')
        common = _binding().NativeSource(row['common'], max_bytes=MAX_BYTES)
        if common.identity['sequence'] != row['common_cutoff']:
            raise ValueError('family interface requires its explicit common cutoff snapshot')
        for origin, payload in zip(pair, (pack_value(thaw_tree(row)), row['common']), strict=True):
            info = origin.inspect()
            if (info['role'] != 'interface' or origin.payload() != payload
                    or info['namespace'] != common.identity['namespace'] or info['sequence'] != row['common_cutoff']
                    or origin.source_archive() != common.archive):
                raise ValueError('family interface proof differs from its actual common publication')
            context.recheck(origin)
            dependencies.append(CppSourceReference(common))
    return tuple(dependencies)


def validate_conditioned_family_state(state):
    c, p = state.configuration, state.payload
    _closed(c, ('profile', 'execution_provider', 'engine_receipt', 'limits', 'max_bytes', 'member_count'))
    _closed(p, ('members', 'declaration', 'interface_proofs', 'native_family'))
    if (state.adapter != ADAPTER or state.adapter_version != '1' or state.arithmetic != 'rational'
            or c['profile'] != PROFILE or dict(c['execution_provider']) != PROVIDER
            or tuple(c['limits']) != LIMITS or c['max_bytes'] != MAX_BYTES or state.output_axes
            or not isinstance(state.source, CppSourceReference) or state.step != 0 or state.parent is not None
            or type(c['member_count']) is not int or not 0 < c['member_count'] <= LIMITS[0]
            or len(p['members']) != c['member_count']):
        raise ValueError('versioned immutable conditioned native family required')
    _closed(c['engine_receipt'], ('family', 'literal', 'evidence'))
    declaration = _declaration(p['declaration'])
    if state.space != _space(state.source, declaration['key']): raise ValueError('family primary source coordinate order differs')
    for image in p['members']: _bytes(image)
    _bytes(p['native_family'])
    if len(p['interface_proofs']) != len(declaration['interfaces']): raise ValueError('complete interface proof records required')
    for row in p['interface_proofs']:
        if len(row) != 2: raise ValueError('complete interface proof pair required')
        for proof in row:
            _closed(proof, ('key', 'role', 'payload', 'receipt'))
            _bytes(proof['payload']); _bytes(proof['receipt'])


def _admit(value, context):
    source, members, proofs, family = value
    for member in members: member.admit(context)
    for pair in proofs: _recheck(pair, context)
    return value


@dataclass(frozen=True)
class _PreparedFamily:
    source: object
    members: tuple
    proofs: tuple
    family: object
    declaration: object

    def __iter__(self): return iter((self.source, self.members, self.proofs, self.family))


def prepare_conditioned_family_model(state, source, *, execution_context):
    state.check_state(); state.bind(source)
    context = _context(_binding(), execution_context)
    if thaw_tree(state.configuration['engine_receipt']) != _receipts():
        raise ValueError('conditioned family requires its pinned native implementation')
    with _lock: known = _models.get(state)
    if known is not None:
        if known.source is not source: raise ValueError('live family requires its actual admitted primary source owner; restore a cold state explicitly')
        return _admit(known, context).family
    members = tuple(PreparedConditionedFamilyMember.from_archive(image, context, source=source if i == 0 else None)
                    for i, image in enumerate(state.payload['members']))
    proofs = tuple(_restore_proofs(row, context) for row in state.payload['interface_proofs'])
    declaration = state.payload['declaration']
    interface_deps = _interface_proofs(declaration, proofs, context)
    family = _binding().PreparedFieldFamily(thaw_tree(declaration), tuple(m.available.evidence() for m in members), **POLICY)
    if family.archive() != state.payload['native_family']: raise ValueError('family complete native members/interfaces/proof differ')
    deps = tuple(dep for member in members for dep in member.admit(context)[1])+interface_deps
    if len(deps) != len(state.dependencies) or any(not a.matches(b) for a,b in zip(deps, state.dependencies, strict=True)):
        raise ValueError('family complete original member/interface source dependencies differ')
    value = _PreparedFamily(source, members, proofs, family, declaration)
    with _lock: _models[state] = value
    return family


def cpp_conditioned_family_model(source, *, members, declaration, interface_proofs, execution_context, reference=None):
    context = _context(_binding(), execution_context); members = tuple(members)
    if not members or any(not isinstance(m, PreparedConditionedFamilyMember) for m in members):
        raise TypeError('actual admitted conditioned family members required')
    if source.archive != members[0].source.archive: raise ValueError('family primary source must match the complete first member publication')
    declaration = _declaration(declaration); proofs = tuple(tuple(row) for row in interface_proofs)
    deps = tuple(dep for m in members for dep in m.admit(context)[1])+_interface_proofs(declaration, proofs, context)
    family = _binding().PreparedFieldFamily(thaw_tree(declaration), tuple(m.available.evidence() for m in members), **POLICY)
    c = {'profile': PROFILE, 'execution_provider': PROVIDER, 'engine_receipt': _receipts(),
         'limits': LIMITS, 'max_bytes': MAX_BYTES, 'member_count': len(members)}
    p = {'members': tuple(m.archive() for m in members), 'declaration': declaration,
         'interface_proofs': tuple(tuple(_proof_packet(o) for o in row) for row in proofs), 'native_family': family.archive()}
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    state = ModelState(ADAPTER, '1', ref, _space(ref, declaration['key']), (), c, p, 'rational', dependencies=deps)
    _bytes(pack_value(pack_model(state)))
    with _lock: _models[state] = _PreparedFamily(source, members, proofs, family, declaration)
    enable_conditioned_family_provider()
    return state


@dataclass(frozen=True)
class ConditionedFamilyQuery(Mapping):
    declaration: object
    prepared: object
    def __getitem__(self, name): return self.declaration[name]
    def __iter__(self): return iter(self.declaration)
    def __len__(self): return len(self.declaration)


def conditioned_family_query(state, member, prepared):
    state.check_state()
    if type(member) is not int or not 0 <= member < state.configuration['member_count']:
        raise ValueError('original family member index required')
    if not isinstance(prepared, PreparedConditionedInput): raise TypeError('actual prepared literal input required')
    return ConditionedFamilyQuery(freeze_tree({'profile': QUERY, 'model_digest': state.coefficient_digest,
        'source': state.source.as_record(), 'member': member, 'input': prepared.archive()}), prepared)


def _query(state, value, context):
    _closed(value, ('profile', 'model_digest', 'source', 'member', 'input'))
    if (value['profile'] != QUERY or value['model_digest'] != state.coefficient_digest
            or not source_from_record(value['source'], allow_cpp=True).matches(state.source)
            or type(value['member']) is not int or not 0 <= value['member'] < state.configuration['member_count']):
        raise ValueError('query differs from the complete declared family/member')
    with _lock: known = _models[state]
    member = known.members[value['member']]
    item = value.prepared if isinstance(value, ConditionedFamilyQuery) else PreparedConditionedInput.from_archive(value['input'], context, grammar=member.grammar)
    if item.archive() != value['input']: raise ValueError('live family literal proof differs from its saved query')
    item.admit(context, member.admit(context)[0], grammar=member.grammar)
    return known, item


def _execute(family, member, item):
    module = _binding(); program = family.member(member).program()
    if item.literal.program().archive() != program.archive(): raise ValueError('literal full parent program differs from selected family member')
    declaration = item.literal.declaration(); key = declaration['key']-91000
    # This explicitly declared input bundle adopts one checked parent owner.
    accession = module.NativeInputAccession(item.literal.asset(), program, _accession(key), **POLICY)
    literal = module.NativeLiteralInput(accession, declaration, **POLICY)
    execution = program.execute((literal.input(),))
    if execution.archive() != item.literal.program().execute((item.literal.input(),)).archive():
        raise ValueError('adopted family input differs from its complete original literal execution')
    return family.execute(member, execution)


def _output(state, known, item, query, result):
    packet = pack_value({'profile': PROFILE, 'source': known.source.archive, 'model': pack_model(state),
                         'query': thaw_tree(query), 'joint': result.archive()})
    _bytes(packet)
    dependencies = state.dependencies+tuple(CppSourceReference(item.source.parent.model_source) for _ in item.proofs)
    archive = {'schema': SCHEMA, 'data': packet, 'sha256': sha256(packet).hexdigest(),
               'program_sha256': sha256(result.program().archive()).hexdigest(), 'engine_receipt': state.configuration['engine_receipt']}
    output = ModelOutput([], CoordinateSpace('native-conditioned-joint-field-readout', ()), (), state.source,
        state.coefficient_digest, 'rational', METHOD, dependencies=dependencies, field_archive=archive,
        execution_receipt={'profile': METHOD, 'limits': LIMITS, 'max_bytes': MAX_BYTES, 'member': result.member_index,
                           'dependency_count': len(dependencies), 'interpretation': 'complete-native-joint-field-family'})
    with _lock: _outputs[output] = (state, known.source, item, query, result)
    return output


def _infer(state, source, inputs, *, execution_context):
    family = prepare_conditioned_family_model(state, source, execution_context=execution_context)
    known, item = _query(state, inputs, execution_context)
    result = _execute(family, inputs['member'], item)
    return _output(state, known, item, inputs, result)


def read_conditioned_family_result(output, *, execution_context, state=None, limits=None, max_bytes=None):
    if (not isinstance(output, ModelOutput) or output.method != METHOD
            or output.space != CoordinateSpace('native-conditioned-joint-field-readout', ())
            or output.axes or output.values.size or output.grade is not None or output.variance != 'coordinate'):
        raise TypeError('complete conditioned joint native output required')
    output.check_state(); a, r = output.field_archive, output.execution_receipt
    if limits is not None and (len(limits) != 4 or any(type(v) is not int or v < minimum for v, minimum in zip(limits, LIMITS, strict=True))):
        raise ValueError('requested admission does not cover the fixed family profile')
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < len(a['data'])):
        raise ValueError('family output exceeds requested byte admission')
    _closed(r, ('profile', 'limits', 'max_bytes', 'member', 'dependency_count', 'interpretation'))
    if (a['schema'] != SCHEMA or thaw_tree(a['engine_receipt']) != _receipts()
            or r['profile'] != METHOD or tuple(r['limits']) != LIMITS or r['max_bytes'] != MAX_BYTES
            or r['interpretation'] != 'complete-native-joint-field-family' or r['dependency_count'] != len(output.dependencies)):
        raise ValueError('joint output receipt or pinned implementation differs')
    _bytes(a['data'])
    with _lock: known = _outputs.get(output)
    if known is None:
        row = unpack_value(a['data']); _closed(row, ('profile', 'source', 'model', 'query', 'joint'))
        if row['profile'] != PROFILE: raise ValueError('conditioned joint output profile differs')
        model = unpack_model(row['model']); source = _binding().NativeSource(_bytes(row['source']), max_bytes=MAX_BYTES)
        family = prepare_conditioned_family_model(model, source, execution_context=execution_context)
        prepared, item = _query(model, row['query'], execution_context)
        result = _execute(family, row['query']['member'], item)
        if result.archive() != row['joint']: raise ValueError('saved complete native joint fields differ from replay')
        rebuilt = _output(model, prepared, item, row['query'], result)
        if pack_value(pack_model(rebuilt)) != pack_value(pack_model(output)):
            raise ValueError('joint output complete dependencies, source, input or receipt differs')
        known = (model, source, item, row['query'], result)
        with _lock: _outputs[output] = known
    model, source, item, query, result = known
    if state is not None and state.coefficient_digest != model.coefficient_digest:
        raise ValueError('joint output belongs to another complete model')
    prepare_conditioned_family_model(model, source, execution_context=execution_context)
    _query(model, ConditionedFamilyQuery(query, item), execution_context)
    if output.model_digest != model.coefficient_digest or r['member'] != query['member']:
        raise ValueError('joint result model or original member differs')
    return result


def _create(source, *, adapter=ADAPTER, **configuration):
    if adapter != ADAPTER: raise ValueError('conditioned native family adapter required')
    return cpp_conditioned_family_model(source, **configuration)


def enable_conditioned_family_provider():
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(PROVIDER['name'], '1', ((ADAPTER, '1'),), _infer,
                create=_create, prepare=prepare_conditioned_family_model, context_operations=('create', 'prepare', 'infer'))
            register_model_provider(_provider)
    return _provider
