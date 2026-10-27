"""Typed RCQL operations over complete native relational runtime packets."""


def _runtime(packet, owner=None):
    from rexgraph.relational_runtime import RelationalRuntime
    if owner is None:
        return RelationalRuntime.from_bytes(packet)
    if not isinstance(owner, RelationalRuntime) or type(packet) is not bytes or owner.to_bytes() != packet:
        raise ValueError("prepared relational owner differs from the selected complete packet")
    return owner


def relational_create(source, namespace):
    from rexgraph.relational_runtime import RelationalRuntime
    return RelationalRuntime.create(namespace).to_bytes()


def relational_extend(source, packet, operation, specification, owner=None):
    return _runtime(packet, owner).extend(operation, specification).to_bytes()


def relational_info(source, packet, observation="source", identity=None, owner=None):
    value = _runtime(packet, owner)
    if observation == "source":
        return value.inspect()
    if observation == "function":
        return value.function(identity)
    if observation == "fit":
        return value.fit(identity)
    if observation == "evolution":
        return value.evolution(identity)
    if observation == "selection":
        return value.selection(identity)
    if observation == "layer":
        return value.layer(identity)
    if observation == "support":
        return {"domains": value.support(identity)}
    if observation == "program":
        return value.program(identity)
    if observation == "memberships":
        return {"members": value.memberships(identity)}
    raise ValueError("unsupported relational runtime observation")


def relational_evaluate(source, packet, parameter, identity=None, tensor=False, owner=None):
    return _runtime(packet, owner).evaluate(parameter, identity, tensor=tensor)


def relational_packet(source, index=0):
    from rexgraph.io.relational_record import read_relational_record
    return read_relational_record(source, index).to_bytes()


def relational_field_function(source, envelope, record, polynomial=True):
    from rexgraph.relational_runtime import RelationalFieldFunction
    return RelationalFieldFunction.from_envelope(envelope, record, polynomial=polynomial).to_bytes()


def relational_field_info(source, packet, parameter=None):
    from rexgraph.relational_runtime import RelationalFieldFunction
    value = RelationalFieldFunction.from_bytes(packet)
    info = value.inspect()
    if parameter is not None:
        field = value.evaluate(parameter)
        info.update(parameter=parameter, field=field.inspect(), quadrance=field.quadrance())
    return info


def action_description(source, recipe):
    from rexgraph.action_program import ActionProgram
    return ActionProgram.description(recipe)


def action_prepare(source, source_archive, endpoint, recipe, publication, availability):
    from rexgraph.action_program import ActionProgram
    return ActionProgram.create(source_archive, endpoint, recipe, publication, availability).to_bytes()


def action_evidence(source, envelope, specification):
    from rexgraph.action_program import ActionEvidence
    return ActionEvidence.create(envelope, specification).to_bytes()


def action_execute(source, packet, evidence, availability, input_selector=None, pairs=(), owner=None, lease=None):
    from rexgraph.action_program import ActionEvidence, ActionProgram
    if type(evidence) is bytes:
        evidence = ActionEvidence.from_bytes(evidence)
    if owner is None:
        owner = ActionProgram.from_bytes(packet, availability=availability)
    elif not isinstance(owner, ActionProgram) or owner.to_bytes() != packet:
        raise ValueError("prepared action owner differs from the selected complete packet")
    return owner.execute(evidence, availability, input_selector=input_selector, pairs=pairs, lease=lease).to_bytes()


def action_result(source, packet, evidence, availability):
    from rexgraph.action_program import ActionEvidence, ActionOutput
    if type(evidence) is bytes:
        evidence = ActionEvidence.from_bytes(evidence)
    return ActionOutput.replay(packet, evidence=evidence, availability=availability).inspect()


def action_packet(source, record=None):
    from rexgraph.graph import RexGraph
    from rexgraph.io.action_record import action_packet as read_packet
    selected = source if record is None else record
    if not isinstance(selected, RexGraph):
        raise TypeError("action packet requires a selected RexGraph application record")
    return read_packet(selected)


def install(register):
    for name, fn in (("RELATIONAL_CREATE", relational_create), ("RELATIONAL_EXTEND", relational_extend),
                     ("RELATIONAL_INFO", relational_info), ("RELATIONAL_EVALUATE", relational_evaluate),
                     ("RELATIONAL_PACKET", relational_packet), ("RELATIONAL_FIELD_FUNCTION", relational_field_function),
                     ("RELATIONAL_FIELD_INFO", relational_field_info),
                     ("ACTION_DESCRIPTION", action_description), ("ACTION_PREPARE", action_prepare),
                     ("ACTION_EVIDENCE", action_evidence), ("ACTION_EXECUTE", action_execute),
                     ("ACTION_RESULT", action_result), ("ACTION_PACKET", action_packet)):
        register(name)(fn)


def install_signatures(register):
    from .signatures import OperatorSignature, TypePattern
    from .types import Domain, Effect, Exactness, RCType
    from .types import ValueKind as K
    packet = TypePattern("packet", kind=K.BYTES, literal=bytes)
    owner = TypePattern("owner", optional=True)
    identity = TypePattern("identity", optional=True)
    specs = (
        ("RELATIONAL_CREATE", (TypePattern("namespace", literal=int),), K.BYTES),
        ("RELATIONAL_EXTEND", (packet, TypePattern("operation", literal=str), TypePattern("specification"), owner), K.BYTES),
        ("RELATIONAL_INFO", (packet, TypePattern("observation", literal=str, optional=True), identity, owner), K.RECORD),
        ("RELATIONAL_EVALUATE", (packet, TypePattern("parameter"), identity,
            TypePattern("tensor", literal=bool, optional=True), owner), K.EXACT_RATIONAL),
        ("RELATIONAL_PACKET", (TypePattern("index", literal=int, optional=True),), K.BYTES),
        ("RELATIONAL_FIELD_FUNCTION", (TypePattern("envelope", kind=K.BYTES, literal=bytes), TypePattern("record", literal=int),
            TypePattern("polynomial", literal=bool, optional=True)), K.BYTES),
        ("RELATIONAL_FIELD_INFO", (packet, TypePattern("parameter", optional=True)), K.RECORD),
        ("ACTION_DESCRIPTION", (TypePattern("recipe"),), K.TEXT),
        ("ACTION_PREPARE", (TypePattern("source_archive", kind=K.BYTES, literal=bytes),
            TypePattern("endpoint", kind=K.BYTES, literal=bytes), TypePattern("recipe"),
            TypePattern("publication", literal=int), TypePattern("availability")), K.BYTES),
        ("ACTION_EVIDENCE", (TypePattern("envelope", kind=K.BYTES, literal=bytes), TypePattern("specification")), K.BYTES),
        ("ACTION_EXECUTE", (packet, TypePattern("evidence"), TypePattern("availability"),
            TypePattern("input_selector", optional=True), TypePattern("pairs", optional=True), owner,
            TypePattern("lease", optional=True)), K.BYTES),
        ("ACTION_RESULT", (packet, TypePattern("evidence"), TypePattern("availability")), K.RECORD),
        ("ACTION_PACKET", (TypePattern("record", optional=True),), K.BYTES),
    )
    for name, inputs, kind in specs:
        register(OperatorSignature(name=name, inputs=inputs,
            result=RCType(kind.value, kind=kind, domain=Domain.RATIONAL if kind == K.EXACT_RATIONAL else Domain.METADATA,
                          exactness=Exactness.RATIONAL if kind == K.EXACT_RATIONAL else Exactness.STRUCTURAL),
            source_kind=(K.REX, K.RCDB_STORE) if name.startswith("ACTION_") else K.REX,
            memoizable=False, effects=frozenset({Effect.READ}), requires=frozenset({"read"}),
            implementation_key="native-relational-function-runtime"))
