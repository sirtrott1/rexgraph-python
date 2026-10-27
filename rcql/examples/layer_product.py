"""Run with an installed RCQL and rexgraph-temporal-bindings SDK build."""
from rcql import Executor, TemporalModel, Layer, LayerProduct, LayerOperator, parse
from rexgraph import RexGraph


def main():
    model = TemporalModel.create(2026).declare('coordinate',
        dict(name='t', clock='event clock', unit='seconds', origin='declared zero'))
    coordinate = model.selected
    model = model.declare('function', dict(coordinate=coordinate,
        pieces=({'numerator': (1, 2, 3)},), name='q'))
    function = model.selected
    model = model.declare('program', {'opcode': 'input'})
    argument = model.selected
    model = model.declare('program', {'opcode': 'derivative', 'operands': (argument,)})
    derivative = model.selected
    model = model.declare('support', dict(coordinate=coordinate, components=((0, 2, True, True),)))
    support = model.selected
    product = LayerProduct.create(tuple(Layer.function(key, model, support, program=derivative, role=role)
        for key, role in (('left', 'convergent'), ('right', 'divergent'))))
    original = LayerOperator.compile(product, {}, {})
    changed = model.revise_program(derivative, {'opcode': 'multiply', 'operands': (argument, argument)})
    revised = product.modify(tuple(Layer.function(layer.key, changed, support, program=changed.selected,
        role=layer.declaration['role']) for layer in product.layers))
    replacement = LayerOperator.compile(revised, {}, {})
    root = RexGraph.from_cells([1, []])
    executor = Executor(sources={'r': root}, params={'old': original, 'new': replacement, 'bindings': {1: function}})
    result = executor.execute(parse('''FROM $r
        LET before=LAYER_APPLY($old,parameters=$bindings)
        LET after=LAYER_APPLY($new,parameters=$bindings)
        RETURN FUNCTION_VALUE(before,2),FUNCTION_VALUE(after,2)'''))
    assert result.values == (14, 289)
    assert original.apply({}, {1: function}).value(2) == 14
    print('original derivative at 2:', result.values[0])
    print('revised square at 2:', result.values[1])
    print('retained prior product:', revised.declaration['parent'] == product.data)


if __name__ == '__main__':
    main()
