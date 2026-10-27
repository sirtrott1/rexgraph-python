"""First-class finite layer diagrams and native whole-function declarations."""
from .types import RCType, ValueKind, Exactness

ARGUMENTS = {
    "LAYER": (("key", "space", "role", "program", "coefficients", "inputs"), ("field", None, (), ())),
    "LAYER_FUNCTION": (("key", "model", "support", "function", "program", "scale", "shift", "role"), (None, 0, 1, 0, "function")),
    "LAYER_FAMILY": (("key", "model", "family", "role"), (None, "context")),
    "LAYER_INTERFACE": (("key", "maps"), ()),
    "LAYER_PRODUCT": (("layers", "interfaces", "name", "links", "outputs", "transitions"), ((), "layers", (), (), ())),
    "LAYER_MODIFY": (("product", "replacements", "interfaces"), ((), None)),
    "LAYER_SOLVE": (("product", "pins", "arguments", "sources"), ((), None, None)),
    "LAYER_OBSERVE": (("solution", "action", "sources"), (None,)),
    "LAYER_PROJECT": (("solution", "layer", "sources"), (None,)),
    "LAYER_VALUE": (("readout", "sources"), (None,)),
    "LAYER_COMPILE": (("value", "sources", "parameters"), (None, None)),
    "LAYER_APPLY": (("operation", "sources", "parameters"), (None, None)),
    "LAYER_INFO": (("value", "sources"), (None,)),
    "LAYER_RECORD": (("value",), ()),
    "LAYER_READ": (("record", "kind"), (None, "LayerProduct")),
    "FUNCTION_REPLAY": (("model","expected_build","limits","sources"),(None,)),
    "FUNCTION_IMPORT": (("packet", "kind", "limits"), ("function", ())),
    "FUNCTION_EXPORT": (("model", "limits", "sources"), (None, None)),
    "FUNCTION_SOURCE": (("name_space", "limits"), (1, ())),
    "FUNCTION_DECLARE": (("model", "kind", "specification", "sources"), (None,)),
    "FUNCTION_PROGRAM": (("model", "opcode", "operands", "specification", "sources"), ((), None, None)),
    "FUNCTION_APPLY": (("model", "program", "bindings", "sources"), (None,)),
    "FUNCTION_FIT": (("model", "function_set", "context", "candidates", "expected", "observation_domain", "sources"), (None, None)),
    "FUNCTION_TRANSFER": (("model", "family", "bindings", "sources"), (None,)),
    "FUNCTION_JOINT": (("model", "families", "transitions", "sources"), ((), None)),
    "FUNCTION_MODIFY": (("model", "program", "changes", "sources"), (None,)),
    "FUNCTION_VALUE": (("model", "at", "function", "sources"), (None, None)),
    "FUNCTION_FIELD": (("model", "injection", "grade", "powers", "function", "sources"), (None, None)),
    "FUNCTION_ID": (("model",), ()),
    "FUNCTION_INFO": (("model", "sources"), (None,)),
    "FUNCTION_VERIFY": (("model", "sources"), (None,)),
    "FUNCTION_RECORD": (("model",), ()),
    "FUNCTION_READ": (("record",), (None,)),
}


def literal_type(binding, value):
    from .layer_product import LAYER_VALUES
    from .native_temporal import TemporalModel
    cls = type(value)
    if cls is TemporalModel or cls in LAYER_VALUES.values():
        kind = ValueKind.TEMPORAL_MODEL if cls is TemporalModel else getattr(ValueKind, value._kind.upper())
        return RCType(cls.__name__, kind=kind, exactness=Exactness.STRUCTURAL,
                      declaration_digest=value.coefficient_digest,
                      layer_profile=getattr(value, "profile", getattr(getattr(value, "product", None), "profile", None)))
    return None


def install(register, OperatorSignature, TypePattern):
    v = ValueKind
    layer, model, product, solution = v.LAYER, v.TEMPORAL_MODEL, v.LAYER_PRODUCT, v.LAYER_SOLUTION
    table = {
        "LAYER": ((str, None, str, None, None, None), layer),
        "LAYER_FUNCTION": ((str, model, int, None, int, None, None, str), layer),
        "LAYER_FAMILY": ((str, model, None, str), layer),
        "LAYER_INTERFACE": ((str, None), v.LAYER_INTERFACE),
        "LAYER_PRODUCT": ((None, None, str, None, None, None), product),
        "LAYER_MODIFY": ((product, None, None), product),
        "LAYER_SOLVE": ((product, None, None, None), solution),
        "LAYER_OBSERVE": ((solution, v.COORDINATE_MAP, None), v.LAYER_READOUT),
        "LAYER_PROJECT": ((solution, str, None), v.LAYER_READOUT),
        "LAYER_VALUE": ((v.LAYER_READOUT, None), v.TENSOR_FIELD),
        "LAYER_COMPILE": ((None, None, None), v.LAYER_OPERATOR),
        "LAYER_APPLY": ((v.LAYER_OPERATOR, None, None), v.PROGRAM_RESULT),
        "LAYER_INFO": ((None, None), v.RECORD), "LAYER_RECORD": ((None,), v.REX),
        "LAYER_READ": ((None, str), product),
        "FUNCTION_REPLAY": ((model,None,None,None),model),
        "FUNCTION_IMPORT": ((bytes, str, None), model), "FUNCTION_EXPORT": ((model, None, None), v.BYTES),
        "FUNCTION_SOURCE": ((int, None), model), "FUNCTION_DECLARE": ((model, str, None, None), model),
        "FUNCTION_PROGRAM": ((model, str, None, None, None), model),
        "FUNCTION_APPLY": ((model, int, None, None), model),
        "FUNCTION_FIT": ((model, int, int, None, int, None, None), model),
        "FUNCTION_TRANSFER": ((model, int, None, None), model),
        "FUNCTION_JOINT": ((model, None, None, None), model),
        "FUNCTION_MODIFY": ((model, int, None, None), model),
        "FUNCTION_VALUE": ((model, None, None, None), v.EXACT_RATIONAL),
        "FUNCTION_FIELD": ((model, v.COORDINATE_MAP, int, None, None, None), v.TENSOR_FIELD),
        "FUNCTION_ID": ((model,), v.EXACT_INTEGER), "FUNCTION_INFO": ((model, None), v.RECORD),
        "FUNCTION_VERIFY": ((model, None), v.BOOLEAN), "FUNCTION_RECORD": ((model,), v.REX),
        "FUNCTION_READ": ((None,), model),
    }
    for name, (labels, defaults) in ARGUMENTS.items():
        kinds, output = table[name]
        patterns = tuple(TypePattern(label, kind=kind, optional=i >= len(labels)-len(defaults))
                         if isinstance(kind, ValueKind) else
                         TypePattern(label, literal=kind, optional=i >= len(labels)-len(defaults))
                         for i, (label, kind) in enumerate(zip(labels, kinds, strict=True)))
        result = RCType(output.value, kind=output, exactness=Exactness.STRUCTURAL)
        if name in {"FUNCTION_VALUE", "LAYER_VALUE", "FUNCTION_FIELD"}:
            from .types import Domain
            result = result.with_(exactness=Exactness.RATIONAL, domain=Domain.RATIONAL)
        if name == "FUNCTION_ID":
            from .types import Domain
            result = result.with_(exactness=Exactness.INTEGER, domain=Domain.INTEGER)
        register(OperatorSignature(name, v.REX, patterns, result,
                 "rcql.layer_operators."+name.lower(),
                 requires=frozenset({"read", "train"} if name == "FUNCTION_FIT" else {"read"}), memoizable=True))


def refine(typed, children, context):
    # Planning validates declarations and authority; it never runs a native
    # executor, solves a diagram or executes a target program.
    from .layer_product import LayerOperator, LayerReadout, LayerSolution, LayerProduct
    from .native_temporal import TemporalModel
    from .execution_trace import capture_methods
    values = tuple(context.known_value(c) for c in children)
    result = typed.result
    profile = getattr(typed.args[0], "layer_profile", None) if typed.args else None
    if typed.operator in {"LAYER", "LAYER_FUNCTION", "LAYER_FAMILY"}:
        profile = {"LAYER": "affine", "LAYER_FUNCTION": "functions", "LAYER_FAMILY": "families"}[typed.operator]
    elif typed.operator == "LAYER_PRODUCT":
        declarations = typed.args[0]
        profiles = {getattr(v, "layer_profile", None) for v in declarations} if isinstance(declarations, (tuple,list)) else set()
        profile = next(iter(profiles)) if len(profiles) == 1 else None
    if typed.operator in {"LAYER", "LAYER_FUNCTION", "LAYER_FAMILY", "LAYER_PRODUCT", "LAYER_MODIFY", "LAYER_SOLVE", "LAYER_COMPILE"}:
        result = result.with_(layer_profile=profile)
    if typed.operator == "LAYER_APPLY" and profile == "c_workflows":
        result = RCType("CValue",kind=ValueKind.C_VALUE,exactness=Exactness.STRUCTURAL)
    if typed.operator == "LAYER_APPLY" and profile == "functions":
        result = RCType("TemporalModel", kind=ValueKind.TEMPORAL_MODEL, exactness=Exactness.STRUCTURAL)
    if values and isinstance(values[0], (TemporalModel, LayerOperator, LayerReadout, LayerSolution, LayerProduct)):
        with capture_methods(binding=context.binding):
            # Explicit contributor maps are checked by execution when available.
            # Selected source permission and historical closure are checked here.
            context.binding.source.require("read")
        if typed.operator == "LAYER_APPLY" and isinstance(values[0], LayerOperator) and values[0].product.profile == "affine":
            from .layer_product import _execution_sources
            from .program import Program
            from .program_operators import _program_executor
            sources = values[1] if len(values) > 1 and values[1] is not None else {context.binding.name: context.binding}
            parameters = values[2] if len(values) > 2 and values[2] is not None else {}
            if isinstance(sources, dict) and isinstance(parameters, dict):
                with capture_methods(binding=context.binding):
                    bindings = _execution_sources(values[0].product, sources, values[0].product.system(sources))
                    plan = Program.from_bytes(values[0].declaration["program"]).execute(_program_executor(bindings, parameters), explain=True)
                result = result.with_(program_outputs=plan["output_types"])
        if typed.operator == "LAYER_VALUE" and isinstance(values[0], LayerReadout):
            from .layer_product import _read_map
            from .types import CoordinateDescriptor, ShapeRef, Domain
            action = _read_map(values[0].declaration["map"])
            result = result.with_(coordinates=CoordinateDescriptor(action.codomain.name, action.codomain.keys),
                                  shape=ShapeRef((len(action.codomain.keys),)), domain=Domain.RATIONAL,
                                  exactness=Exactness.RATIONAL)
    if typed.operator == "FUNCTION_FIELD":
        from .function_embedding import validate_embedding
        from .types import CoordinateDescriptor, ShapeRef, Domain, Variance
        injection, grade, powers = values[1:4]
        if injection is not None and grade is not None and powers is not None:
            validate_embedding(context.binding.value, injection, grade, powers)
            result = result.with_(grade=grade, variance=Variance.CHAIN, source=context.binding.ref,
                     coordinates=CoordinateDescriptor(injection.codomain.name, injection.codomain.keys),
                     tensor_axes=(CoordinateDescriptor(powers.name, powers.keys),),
                     shape=ShapeRef((len(injection.codomain.keys),len(powers.keys))), domain=Domain.RATIONAL)
    if typed.operator == "LAYER_READ":
        kind = values[1] if len(values) > 1 and values[1] is not None else "LayerProduct"
        from .layer_product import LAYER_VALUES
        if kind not in LAYER_VALUES:
            raise ValueError("unknown layer record kind")
        result = result.with_(kind=getattr(ValueKind, LAYER_VALUES[kind]._kind.upper()), name=kind)
    return result, ()
