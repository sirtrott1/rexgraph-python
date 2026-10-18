"""Fit and restore a canonical Boolean gate on a saved model's declared rows."""
import argparse
import json
from pathlib import Path

from rexgraph.cpp_binary_gate import (
    cpp_binary_gate_batch,
    cpp_binary_gate_input,
    pack_binary_plane,
)
from rexgraph.io import load, save
from rexgraph.io.model_record import model_record, read_model_record
from rexgraph.model_runtime import infer_model, train_model
from rexgraph.model_state import thaw_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--event-cutoff", type=int)
    parser.add_argument("--knowledge-cutoff", type=int)
    args = parser.parse_args()
    state, source = read_model_record(load(args.model))
    data = json.loads(args.data.read_text())
    if not isinstance(data, dict) or set(data) not in ({"first", "second"}, {"first", "second", "targets"}):
        parser.error("data requires first and second Boolean planes, with optional targets")
    first, second = pack_binary_plane(data["first"]), pack_binary_plane(data["second"])
    if any(len(data[key]) != len(state.space.keys) for key in data):
        parser.error("every plane must match the model's declared row count")
    if "targets" in data:
        if args.event_cutoff is None or args.knowledge_cutoff is None:
            parser.error("fitting requires explicit event and knowledge cutoffs")
        batch = cpp_binary_gate_batch(state, source, first, second, pack_binary_plane(data["targets"]),
            event_cutoff=args.event_cutoff, knowledge_cutoff=args.knowledge_cutoff)
        state = train_model(state, source, batch)
    inputs = cpp_binary_gate_input(state, source, first, second, prepare=True)
    output = infer_model(state, source, inputs)
    record = model_record(source, state)
    record.attach_metadata(1, 0, "binary_input", thaw_tree(inputs))
    record.attach_metadata(1, 0, "model_output", output)
    save(args.output, record)
    restored = load(args.output)
    current, owner = read_model_record(restored)
    actual = infer_model(current, owner, restored.get_metadata(1, 0, "binary_input"))
    if actual.coefficient_digest != output.coefficient_digest:
        raise RuntimeError("restored Boolean gate inference differs")
    print(json.dumps({"step": state.step, "truth_table": state.payload["truth_table"],
        "hamming_error": state.payload["last_hamming_error"], "output_digest": output.coefficient_digest,
        "record": str(args.output)}))


if __name__ == "__main__":
    main()
