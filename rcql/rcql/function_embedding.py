"""An explicit admitted map from a native polynomial to a physical field."""
from fractions import Fraction


def validate_embedding(source, injection, grade, powers):
    from rexgraph.coordinate_map import CoordinateMap
    from rexgraph.type_accession import CoordinateSpace
    from rexgraph.native_field import NativeFieldCalculus
    if (not isinstance(injection, CoordinateMap) or len(injection.domain.keys) != 1
            or not isinstance(powers, CoordinateSpace) or not 1 <= len(powers.keys) <= 4096
            or powers.keys != tuple(str(i) for i in range(len(powers.keys)))
            or type(grade) is not int or grade < 0):
        raise ValueError("polynomial embedding requires one amplitude, an explicit grade and ordered power coordinates")
    calculus = NativeFieldCalculus.from_rex(source)
    if grade >= len(calculus.complex.spaces) or injection.codomain != calculus.complex.spaces[grade]:
        raise ValueError("polynomial injection must target the actual original cell carrier")


def embed_function(source, model, injection, grade, powers, function, reference, bindings):
    from rexgraph.tensor_field import TensorField, apply_tensor
    from .layer_product import _bound_dependencies
    validate_embedding(source, injection, grade, powers)
    identity = model.selected if function is None else function
    info = model._native.function_info(identity)
    if len(info["pieces"]) != 1:
        raise ValueError("polynomial field embedding requires a single whole-domain piece")
    piece = info["pieces"][0]
    denominator = piece["denominator"]
    if (piece["domain"] != (None,None,False,False) or len(denominator) != 1 or not denominator[0]
            or len(piece["numerator"]) > len(powers.keys)):
        raise ValueError("embedding cannot discard function domains, denominator exclusions or polynomial powers")
    coefficients = tuple(piece["numerator"]) + (Fraction(0),)*(len(powers.keys)-len(piece["numerator"]))
    coefficients = tuple(v/denominator[0] for v in coefficients)
    field = TensorField(injection.domain, [coefficients], (powers,), reference,
                        provenance=(model.coefficient_digest,), dependencies=_bound_dependencies(model, bindings))
    result = apply_tensor(injection, field)
    from dataclasses import replace
    return replace(result, grade=grade, variance="chain")
