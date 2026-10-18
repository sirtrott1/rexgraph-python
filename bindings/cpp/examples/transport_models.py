"""Transport learned integer parameters and resume a saved destination model."""
import sys
from pathlib import Path

from rexgraph_cpp import NativeSource
from source_models import source_declaration

from rexgraph.cpp_perceptron import cpp_perceptron_batch, cpp_perceptron_input, cpp_perceptron_model
from rexgraph.cpp_transport import cpp_transport
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_runtime import infer_model, train_model, transport_model
from rexgraph.source_binding import CppSourceReference


def permute(values, indices):
    result = [None] * len(values)
    for i, j in enumerate(indices): result[j] = values[i]
    return tuple(result)


def main(directory):
    directory.mkdir(parents=True, exist_ok=True)
    declaration = source_declaration()
    source = NativeSource.from_declaration(declaration)
    availability = tuple(dict(event_time=1, knowledge_time=1, label_available_time=1,
                              partition="training") for _ in range(3))
    state = cpp_perceptron_model(source, 11, 4000, 4001, availability,
                                feature_coordinates=900, time_coordinates=999)
    rows = (((0, 2), (1, 1)), ((0, -2), (1, -1)), ((0, 1), (0, 1), (1, 2)))
    observations = cpp_perceptron_batch(state, source, rows, (1, -1, 1), rate=2,
                                        event_cutoff=1, knowledge_cutoff=10)
    trained = train_model(state, source, observations, steps=3)
    maps = ((1, 2, 0), (2, 0, 1))
    feature_map = (1, 0)
    carrier_links = tuple(dict(source=key, target=key+100,
        indices=maps[grade] if kind == 1 else feature_map if key == 40 else (0,))
        for key, grade, kind in source.carriers)
    indices = {v["source"]: v["indices"] for v in carrier_links}
    target_declaration = dict(declaration, namespace=188,
        witnesses=permute((1010, 1011, 1012), maps[0]),
        relations=permute(tuple({**r, "key": r["key"]+1000,
            "occurrences": tuple({**o, "key": o["key"]+1000, "lower": o["lower"]+1000}
                                 for o in r["occurrences"])} for r in declaration["relations"]), maps[1]),
        carriers=tuple({**c, "key": c["key"]+100,
            "coordinates": c["coordinates"]+1000 if c["coordinates"] else 0} for c in declaration["carriers"]))
    fields = []
    for field in declaration["fields"]:
        values = field["entries"]
        if field["key"] == 4000: values = tuple(enumerate(trained.payload["parameters"][:-1]))
        if field["key"] == 4001: values = ((0, trained.payload["parameters"][-1]),)
        fields.append({**field, "key": field["key"]+1000, "carrier": field["carrier"]+100,
                       "entries": tuple((indices[field["carrier"]][i], v) for i, v in values)})
    target_declaration["fields"] = tuple(fields)
    target = NativeSource.from_declaration(target_declaration)
    mapping = cpp_transport(source, target, maps, carrier_links,
        tuple(dict(source=key, target=key+1000) for key, _, _ in source.tensors),
        configuration=dict(carrier=111, weights_key=5000, bias_key=5001,
            availability=permute(availability, maps[1]), feature_coordinates=1900,
            time_coordinates=999, initial_parameter_knowledge_time=10))
    child = transport_model(trained, source, CppSourceReference(target), mapping)
    target_rows = permute(tuple(tuple((feature_map[i], v) for i, v in row) for row in rows), maps[1])
    old_output = infer_model(trained, source, cpp_perceptron_input(trained, source, rows))
    new_output = infer_model(child, target, cpp_perceptron_input(child, target, target_rows))
    assert tuple(new_output.values) == permute(tuple(old_output.values), maps[1])
    (directory/"destination.rfcxs").write_bytes(target.archive)
    path = directory/"transported.rcbd"
    save_model(path, model_record(target, child))
    restored, loaded = read_model_record(load_model(path))
    targets = permute((-1, 1, -1), maps[1])
    def batch(model, data):
        return cpp_perceptron_batch(model, data, target_rows, targets, rate=2,
                                    event_cutoff=1, knowledge_cutoff=11)
    resumed = train_model(restored, loaded, batch(restored, loaded), steps=2)
    expected = train_model(child, target, batch(child, target), steps=2)
    assert resumed.coefficient_digest == expected.coefficient_digest
    save_model(directory/"resumed.rcbd", model_record(loaded, resumed))
    print("transported parameters:", child.payload["parameters"])
    print("resumed parameters:", resumed.payload["parameters"])
    print("cumulative training steps:", resumed.step)


if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: transport_models.py OUTPUT_DIRECTORY")
    main(Path(sys.argv[1]))
