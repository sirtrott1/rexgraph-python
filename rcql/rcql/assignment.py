"""Immutable RCQL worker specializations over exact native assignment checkpoints."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedAssignmentProgram:
    """Retain an actual child owner alongside its portable finite declaration."""
    program: object
    source: object
    state: object
    owner: object
    candidate: int
    binding: object
    parent_program: object
    transformation: object

    def runtime_parameters(self, authority):
        if not callable(authority): raise TypeError('current execution authority resolver required')
        return {'authority': authority, 'owner': lambda: self.owner}

    def inspect(self):
        from rexgraph.model_state import thaw_tree
        return dict(program_digest=self.program.coefficient_digest,
            parent_program_digest=self.parent_program.coefficient_digest,
            transformation_digest=self.transformation.coefficient_digest,
            model_digest=self.state.coefficient_digest, parent_model_digest=self.state.parent,
            candidate=self.candidate, binding=thaw_tree(self.binding),
            native_model_sha256=self.state.payload['sha256'], source=self.state.source.as_record())


def prepare_assignment_program(state, source, *, execution_context, name='assignment_worker'):
    from rexgraph.cpp_assignment import assignment_checkpoint, prepare_assignment_model
    from rexgraph.model_state import freeze_tree

    from . import OutputRef, Program, ProgramInput, ProgramStep, ProgramTransformation, parse
    owner = prepare_assignment_model(state, source, execution_context=execution_context)
    info = owner.inspect(execution_context, max_records=state.configuration['limits'][1])
    if state.step < 1 or info['identified_candidate'] is None or info['remainder_count']:
        raise ValueError('a learned child requires measured native evidence and one covered structural assignment')
    selected = info['identified_candidate']
    actual = owner.candidate(selected, execution_context, max_records=state.configuration['limits'][1])
    checkpoint = assignment_checkpoint(state, source, execution_context=execution_context)
    parent = Program(name, (ProgramStep('execute', parse(
        'FROM $r RETURN MODEL_ASSIGNMENT_EXECUTE($checkpoint,$input,$authority,$owner)')),),
        (ProgramInput('checkpoint', 'Bytes'), ProgramInput('input', 'Bytes'),
         ProgramInput('authority'), ProgramInput('owner')), (('response', OutputRef('execute')),))
    child = parent.specialize({'checkpoint': checkpoint})
    change = ProgramTransformation.program(parent, 'specialize', {'checkpoint': checkpoint})
    change.verify(parent)
    return PreparedAssignmentProgram(child, source, state, owner, selected, freeze_tree(actual['binding']), parent, change)


def restore_assignment_program(program, source, *, execution_context):
    from rexgraph.cpp_assignment import assignment_checkpoint_state, prepare_assignment_model

    from . import Program
    if not isinstance(program, Program): raise TypeError('stored immutable RCQL program required')
    if len(program.captures) != 1 or program.captures[0][0].name != 'checkpoint':
        raise ValueError('assignment child requires one complete native checkpoint capture')
    state = assignment_checkpoint_state(program.captures[0][1])
    prepare_assignment_model(state, source, execution_context=execution_context)
    prepared = prepare_assignment_program(state, source, execution_context=execution_context, name=program.name)
    if prepared.program.coefficient_digest != program.coefficient_digest:
        raise ValueError('stored program differs from its native assignment specialization')
    return prepared


def assignment_execute(source, checkpoint, input_archive, execution_context, owner=None):
    from rexgraph.cpp_assignment import assignment_checkpoint_state, infer_assignment_checkpoint

    from .execution_trace import current_binding, record_method
    from .model_operators import authorize
    state = assignment_checkpoint_state(checkpoint)
    actual = authorize(source, state, current_binding(), execution_context=execution_context)
    output = infer_assignment_checkpoint(checkpoint, actual, input_archive, execution_context=execution_context, owner=owner)
    record_method('conditioned-native-assignment-worker-execution', arithmetic='rational', model_digest=state.coefficient_digest,
                  parent_model_digest=state.parent, native_model_sha256=state.payload['sha256'],
                  candidate=output.execution_receipt['candidate'], binding=output.execution_receipt['binding'])
    return output


def assignment_batch(source, state, observations, execution_context):
    from rexgraph.cpp_assignment import assignment_batch as make_batch

    from .execution_trace import current_binding
    from .model_operators import authorize
    actual = authorize(source, state, current_binding(), execution_context=execution_context)
    return make_batch(state, actual, observations, execution_context=execution_context)


def assignment_field(source, output, node, execution_context, occurrences=False):
    from rexgraph.cpp_assignment import read_assignment_result

    from .model_operators import source_for
    if type(occurrences) is not bool: raise TypeError('occurrence selection requires bool')
    output.source.bind(source_for(source, execution_context=execution_context))
    value = read_assignment_result(output, execution_context=execution_context)
    original = value.original(execution_context)
    return dict(field=value.field_archive(node, execution_context, occurrences=occurrences),
        coordinates=value.inspect(execution_context)['coordinates'],
        program=original.evidence().program().program().publication(),
        source=output.source.as_record(), model_digest=output.model_digest,
        interpretation='conditioned-correlated-native-family')


def install(register):
    for name, fn in (('MODEL_ASSIGNMENT_EXECUTE', assignment_execute),
                     ('MODEL_ASSIGNMENT_BATCH', assignment_batch), ('MODEL_ASSIGNMENT_FIELD', assignment_field)):
        register(name)(fn)


def install_signatures(register):
    from .signatures import OperatorSignature, TypePattern
    from .types import Domain, Effect, Exactness, RCType
    from .types import ValueKind as K
    context = TypePattern('execution_context')
    specs = (
        ('MODEL_ASSIGNMENT_EXECUTE', (TypePattern('checkpoint', literal=bytes), TypePattern('input_archive', literal=bytes),
            context, TypePattern('owner', optional=True)), K.MODEL_OUTPUT),
        ('MODEL_ASSIGNMENT_BATCH', (TypePattern('state', kind=K.MODEL_STATE), TypePattern('observations'), context), K.MODEL_BATCH),
        ('MODEL_ASSIGNMENT_FIELD', (TypePattern('output', kind=K.MODEL_OUTPUT), TypePattern('node', literal=int),
            context, TypePattern('occurrences', literal=bool, optional=True)), K.RECORD),
    )
    for name, inputs, kind in specs:
        register(OperatorSignature(name=name, inputs=inputs,
            result=RCType(kind.value, kind=kind, domain=Domain.RATIONAL, exactness=Exactness.RATIONAL), source_kind=K.REX,
            memoizable=False, effects=frozenset({Effect.READ}), requires=frozenset({'read'}),
            implementation_key='native-assignment-evidence'))
