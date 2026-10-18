"""Construct native parameters, train exact models and resume saved model records."""
import sys
from fractions import Fraction
from pathlib import Path

from rexgraph_cpp import NativeSource

from rexgraph.cpp_learning import cpp_trainable_model, cpp_training_batch
from rexgraph.cpp_perceptron import cpp_perceptron_batch, cpp_perceptron_input, cpp_perceptron_model
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_runtime import infer_model, train_model


def source_declaration():
    relations = []
    for i, (left, right) in enumerate(((10, 11), (11, 12), (12, 10))):
        relations.append({"grade": 1, "key": 100+i, "type": 0, "occurrences": (
            {"key": 2*i, "lower": left, "exists": True, "negative": False, "magnitude": 1},
            {"key": 2*i+1, "lower": right, "exists": True, "negative": True, "magnitude": 1})})
    return {
        "profile": "declared-exact-source-v1", "namespace": 88, "semantic_scale": 0, "max_grade": 1,
        "types": ({"name": "declared-pair", "min_arity": 2, "max_arity": 2},), "witnesses": (),
        "relations": relations,
        "carriers": (
            {"key": 10, "kind": "source_grade", "grade": 0, "coordinates": 0, "size": 3},
            {"key": 11, "kind": "source_grade", "grade": 1, "coordinates": 0, "size": 3},
            {"key": 40, "kind": "explicit", "grade": 0, "coordinates": 900, "size": 2},
            {"key": 41, "kind": "explicit", "grade": 0, "coordinates": 901, "size": 1}),
        "fields": (
            {"key": 1001, "carrier": 11, "entries": ((0, Fraction(1, 3)), (2, Fraction(-2, 5))), "support": None},
            {"key": 4000, "carrier": 40, "entries": ((0, 1), (1, -2)), "support": None},
            {"key": 4001, "carrier": 41, "entries": ((0, 3),), "support": None}),
        "metrics": (),
    }


def main(directory):
    directory.mkdir(parents=True, exist_ok=True)
    source = NativeSource.from_declaration(source_declaration())
    (directory / "source.rfcxs").write_bytes(source.archive)
    availability = tuple({"event_time": 1, "knowledge_time": 1, "label_available_time": 1,
                          "partition": "training"} for _ in range(3))
    state = cpp_perceptron_model(source, 11, 4000, 4001, availability,
                                feature_coordinates=900, time_coordinates=999)
    rows = (((0, 2), (1, 1)), ((0, -2), (1, -1)), ((0, 1), (0, 1), (1, 2)))
    batch = cpp_perceptron_batch(state, source, rows, (1, -1, 1), rate=2,
                                event_cutoff=1, knowledge_cutoff=1)
    trained = train_model(state, source, batch, steps=4)
    save_model(directory / "integer.rcbd", model_record(source, trained))
    restored, loaded = read_model_record(load_model(directory / "integer.rcbd"))
    resumed = train_model(restored, loaded, cpp_perceptron_batch(
        restored, loaded, rows, (1, -1, 1), rate=2, event_cutoff=1, knowledge_cutoff=2))
    expected = train_model(trained, source, cpp_perceptron_batch(
        trained, source, rows, (1, -1, 1), rate=2, event_cutoff=1, knowledge_cutoff=2))
    assert resumed.coefficient_digest == expected.coefficient_digest
    output = infer_model(resumed, loaded, cpp_perceptron_input(resumed, loaded, rows))
    print("integer parameters:", resumed.payload["parameters"])
    print("integer labels:", tuple(output.values))
    declaration = {
        "grade": 1, "channels": ("identity", "hodge"), "model": "channel-example",
        "gains": 4000, "bias": 4001, "channel_coordinates": 900, "bias_coordinates": 901,
        "history_namespace": 700, "metadata_carrier": 9900, "metadata_tensor": 9901,
        "time_coordinates": 9902, "metadata_coordinates": 9903,
        "initial_parameter_knowledge_time": 0, "publication_cutoff": 0,
        "samples": ({"input": 1001, "target": 1001, "event_time": 1, "knowledge_time": 1,
                     "label_available_time": 1, "partition": "training"},),
    }
    state = cpp_trainable_model(source, (10, 11), declaration)
    trained = train_model(state, source, cpp_training_batch(
        state, source, rate=Fraction(1, 1000), event_cutoff=1, knowledge_cutoff=1), steps=4)
    save_model(directory / "rational.rcbd", model_record(source, trained))
    restored, loaded = read_model_record(load_model(directory / "rational.rcbd"))
    assert restored.coefficient_digest == trained.coefficient_digest
    print("rational parameters:", restored.payload["parameters"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: source_models.py OUTPUT_DIRECTORY")
    main(Path(sys.argv[1]))
