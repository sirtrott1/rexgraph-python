"""Executable finite layer products through RCQL's existing binding authority."""
from .execution_trace import current_binding, current_evidence, record_method


def _sources(sources):
    if sources is not None:
        return sources
    binding = current_binding()
    if binding is None:
        raise PermissionError("layer execution requires an explicit query binding")
    return {binding.name: binding}


def _reference():
    from rexgraph.tensor_field import FieldSource
    binding = current_binding()
    if binding is None:
        raise PermissionError("layer declaration requires a query binding")
    binding.source.require("read")
    return FieldSource(binding.value, binding.ref.record_id, binding.ref.record_version)


def _authorize(value, sources=None, training=False):
    from .layer_product import _execution_sources
    binding = current_binding()
    if binding is not None:
        binding.source.require("read")
        if training:
            binding.source.require("train")
    bindings = _execution_sources(value, _sources(sources))
    for item in bindings.values():
        item.source.require("read")
        if training:
            item.source.require("train")
    evidence = current_evidence()
    if evidence is not None:
        if binding is not None:
            evidence.validate_binding(binding)
        for item in bindings.values():
            evidence.validate_binding(item)
        evidence.validate_values((value,))
    return bindings


def layer(source, key, space, role="field", program=None, coefficients=(), inputs=()):
    from .layer_product import Layer
    return Layer.affine(key, space, role=role, program=program, coefficients=coefficients, inputs=inputs)


def layer_function(source, key, model, support, function=None, program=0, scale=1, shift=0, role="function"):
    from .layer_product import Layer
    return Layer.function(key, model, support, function=function, program=program, scale=scale, shift=shift, role=role)


def layer_family(source, key, model, family=None, role="context"):
    from .layer_product import Layer
    return Layer.family(key, model, family, role=role)


def layer_interface(source, key, maps):
    from .layer_product import LayerInterface
    return LayerInterface.equality(key, maps)


def layer_product(source, layers, interfaces=(), name="layers", links=(), outputs=(), transitions=()):
    from .layer_product import LayerProduct
    return LayerProduct.create(layers, interfaces, name=name, links=links, outputs=outputs,
                               transitions=transitions, dependencies=(_reference(),))


def layer_modify(source, product, replacements=(), interfaces=None):
    return product.modify(replacements, interfaces)


def layer_solve(source, product, pins=(), arguments=None, sources=None):
    bindings = _authorize(product, sources)
    record_method("retained-exact-layer-product", profile=product.profile)
    return product.solve(pins, arguments, bindings)


def layer_observe(source, solution, action, sources=None):
    from .layer_product import LayerReadout
    _authorize(solution, sources)
    return LayerReadout.create(solution, action)


def layer_project(source, solution, layer, sources=None):
    return layer_observe(source, solution, solution.product.system().selection(layer), sources)


def layer_value(source, readout, sources=None):
    bindings = _authorize(readout, sources)
    return readout.value(bindings)


def layer_compile(source, value, sources=None, parameters=None):
    from .layer_product import LayerOperator
    bindings = _authorize(value, sources)
    record_method("validated-layer-operator", target_evaluated=False)
    return LayerOperator.compile(value, bindings, {} if parameters is None else parameters)


def layer_apply(source, operation, sources=None, parameters=None):
    bindings = _authorize(operation, sources)
    record_method("runtime-layer-operator", profile=operation.product.profile)
    return operation.execute(bindings, {} if parameters is None else parameters)


def layer_info(source, value, sources=None):
    from .layer_product import LayerProduct, LayerSolution, LayerOperator, LayerReadout
    bindings = _authorize(value, sources)
    if isinstance(value, LayerReadout):
        image = value.image(bindings)
        return {"determined": image.determined, "family_dimension": image.family.dimension,
                "digest": value.coefficient_digest}
    if isinstance(value, LayerSolution):
        value.verify(bindings)
        d = value.declaration
        return {"status": d["status"], "profile": value.product.profile,
                "dimension": value.family(bindings).dimension if value.product.profile == "affine" and d["status"] == "compatible" else None,
                "certificate": d["certificate"], "native": None if d["native"] is None else ({"constructors":len(value.native(bindings).declaration["records"])} if value.product.profile == "c_workflows" else value.native(bindings).info())}
    if isinstance(value, (LayerProduct, LayerOperator)):
        product = value if isinstance(value, LayerProduct) else value.product
        return {"profile": product.profile, "layers": tuple((v.key,v.declaration["role"]) for v in product.layers),
                "digest": value.coefficient_digest, "parent": product.declaration["parent"]}
    raise TypeError("layer info requires a product, solution, readout or operator")


def layer_record(source, value):
    from .layer_product import LAYER_VALUES
    if type(value) not in LAYER_VALUES.values():
        raise TypeError("typed layer declaration required")
    return value.to_record()


def layer_read(source, record=None, kind="LayerProduct"):
    from .layer_product import LAYER_VALUES
    if kind not in LAYER_VALUES:
        raise ValueError("unknown layer record kind")
    return LAYER_VALUES[kind].from_record(source if record is None else record)


def function_replay(source,model,expected_build,limits,sources=None):
    bindings=_authorize(model,sources)
    record_method("explicit-native-function-replay",original_implementation=expected_build)
    return model.replay(expected_build,limits,bindings)


def function_import(source, packet, kind="function", limits=()):
    from rexgraph.relational_runtime import RelationalRuntime
    from .native_temporal import TemporalModel
    return TemporalModel.from_relational(RelationalRuntime.from_bytes(packet), kind=kind, limits=limits,
                                         dependencies=(_reference(),))


def function_export(source, model, limits=None, sources=None):
    _authorize(model, sources)
    return model.to_relational(limits=limits).to_bytes()


def function_source(source, name_space=1, limits=()):
    from .native_temporal import TemporalModel
    return TemporalModel.create(name_space, limits, (_reference(),))


def function_declare(source, model, kind, specification, sources=None):
    _authorize(model, sources)
    return model.declare(kind, specification)


def function_program(source, model, opcode, operands=(), specification=None, sources=None):
    return function_declare(source, model, "program", {**({} if specification is None else specification),
                           "opcode": opcode, "operands": operands}, sources)


def function_apply(source, model, program, bindings, sources=None):
    _authorize(model, sources)
    record_method("native-whole-function-application")
    return model.apply(program, bindings)


def function_fit(source, model, function_set, context, candidates, expected, observation_domain=None, sources=None):
    _authorize(model, sources, training=True)
    record_method("native-finite-context-fit", evidence_profile="NamedInputs")
    return model.fit(function_set, context, candidates, expected, observation_domain)


def function_transfer(source, model, family, bindings, sources=None):
    _authorize(model, sources)
    record_method("native-qualified-whole-function-transfer")
    return model.apply(0, bindings, learned_family=family)


def function_joint(source, model, families, transitions=(), sources=None):
    _authorize(model, sources)
    return model.joint(families, transitions)


def function_modify(source, model, program, changes, sources=None):
    _authorize(model, sources)
    return model.revise_program(program, changes)


def function_value(source, model, at, function=None, sources=None):
    _authorize(model, sources)
    value = model.value(at, function)
    if value is None:
        raise ValueError("whole function is undefined at the requested coordinate")
    return value


def function_field(source, model, injection, grade, powers, function=None, sources=None):
    bindings = _authorize(model, sources)
    from .function_embedding import embed_function
    return embed_function(source, model, injection, grade, powers, function, _reference(), bindings)


def function_id(source, model):
    return model.selected


def function_info(source, model, sources=None):
    _authorize(model, sources)
    return model.info()


def function_verify(source, model, sources=None):
    _authorize(model, sources)
    return model.verify()


def function_record(source, model):
    return model.to_record()


def function_read(source, record=None):
    from .native_temporal import TemporalModel
    return TemporalModel.from_record(source if record is None else record)


def install(register):
    from .layer_contracts import ARGUMENTS
    for name in ARGUMENTS:
        register(name)(globals()[name.lower()])
