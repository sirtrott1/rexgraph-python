"""Train a native integer classifier on explicitly linked saved coefficient features."""
import argparse
import json
from fractions import Fraction
from pathlib import Path

from rexgraph.cpp_algebraic_stalk import read_algebraic_grade_stalk_output
from rexgraph.cpp_coefficient_features import (
    cpp_coefficient_perceptron_batch,
    cpp_coefficient_perceptron_input,
)
from rexgraph.io import load, save
from rexgraph.io.model_record import model_record, read_model_record
from rexgraph.model_runtime import infer_model, train_model
from rexgraph.model_state import thaw_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--rate", type=int, default=1)
    parser.add_argument("--event-cutoff", type=int)
    parser.add_argument("--knowledge-cutoff", type=int)
    args = parser.parse_args()
    state, source = read_model_record(load(args.model))
    rows = json.loads(args.features.read_text())
    outputs = tuple(read_algebraic_grade_stalk_output(load(args.features.parent / row["record"])) for row in rows)
    keys = tuple(row["row"] for row in rows)
    features = tuple(tuple((index, coordinate, power, Fraction(scale))
                          for index, coordinate, power, scale in row["features"]) for row in rows)
    if args.epochs:
        if args.event_cutoff is None or args.knowledge_cutoff is None:
            parser.error("training requires explicit event and knowledge cutoffs")
        captured = cpp_coefficient_perceptron_batch(state, source, outputs, tuple(row["label"] for row in rows),
            row_keys=keys, features=features, rate=args.rate, event_cutoff=args.event_cutoff,
            knowledge_cutoff=args.knowledge_cutoff)
        state = train_model(state, source, captured, steps=args.epochs)
    inputs = cpp_coefficient_perceptron_input(state, source, outputs, row_keys=keys, features=features, prepare=True)
    output = infer_model(state, source, inputs)
    record = model_record(source, state)
    record.attach_metadata(1, 0, "coefficient_input", thaw_tree(inputs))
    record.attach_metadata(1, 0, "model_output", output)
    save(args.output, record)
    restored = load(args.output)
    current, owner = read_model_record(restored)
    actual = infer_model(current, owner, restored.get_metadata(1, 0, "coefficient_input"))
    if actual.coefficient_digest != output.coefficient_digest:
        raise RuntimeError("restored coefficient feature inference differs")
    print(json.dumps({"step": state.step, "parameters": [int(v) for v in state.payload["parameters"]],
        "scores": output.execution_receipt["scores"], "output_digest": output.coefficient_digest,
        "record": str(args.output)}))


if __name__ == "__main__":
    main()
