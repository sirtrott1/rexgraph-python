"""RCQL creates the declared encoded column transport provider versions."""
import pytest

from rexgraph.io.model_record import model_record
from rexgraph.model_contract import resolve_model_provider
from rexgraph.model_runtime import infer_model
from rexgraph.model_state import thaw_tree
from rexgraph.cpp_stalk_transport import stalk_transport_query
from rexgraph.value_codec import pack_value
from rcql import Executor, parse

from test_column_transport import learn, models
from test_tied_column_transport import acquire, columns


@pytest.mark.parametrize("tied,version", [(False, "3"), (True, "4")])
def test_rcql_creates_encoded_columns_with_the_actual_parameter_contract(tied, version):
    source, _, state = columns() if tied else models()
    config = {key: thaw_tree(state.configuration[key]) for key in
              ("stalk_axes", "fixed_maps", "parameter_slots", "training_rows")}
    config["column_fields"] = True
    if tied:
        config.update(parameter_space=thaw_tree(state.configuration["parameter_space"]),
                      parameter_basis=thaw_tree(state.configuration["parameter_basis"]))
    child, batch = acquire(state, source) if tied else learn(source, state, (((1, 0, 0),),))
    query = stalk_transport_query(state, ((1, 0, 0),))
    result = Executor(sources={"record": model_record(source, state)},
                      params={"config": config, "batch": batch, "query": query}).execute(parse(
        'FROM $record RETURN MODEL_INIT("cpp_stalk_transport",$config),'
        'MODEL_INFER(MODEL_TRAIN(MODEL_INIT("cpp_stalk_transport",$config),$batch),$query)'))
    created = result.values[0]
    assert created.adapter_version == version
    assert created.coefficient_digest == state.coefficient_digest
    provider = resolve_model_provider(created, "train")
    assert provider.version == version
    assert provider.target_profiles == ("encoded-original-stalk-targets-v1",)
    expected = infer_model(child, source, query)
    assert result.values[1].coefficient_digest == expected.coefficient_digest
    assert pack_value(thaw_tree(result.values[1].execution_receipt)) == pack_value(thaw_tree(expected.execution_receipt))


@pytest.mark.parametrize("value", [1, "true"])
def test_rcql_column_profile_requires_a_declared_boolean(value):
    source, _, state = models()
    with pytest.raises(TypeError, match="boolean column field profile"):
        Executor(sources={"record": model_record(source, state)},
                 params={"config": {"column_fields": value}}).execute(parse(
            'FROM $record RETURN MODEL_INIT("cpp_stalk_transport",$config)'))
