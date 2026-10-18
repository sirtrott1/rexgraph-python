"""Declare, save and restore a whole grade action on a selected native graph publication."""
import argparse
import json
from pathlib import Path

from rexgraph_cpp import NativeAlgebraicGraphDatabase, NativeAlgebraicGraphRevision, NativeSource

from rexgraph.cpp_algebraic_grade_model import cpp_algebraic_grade_model
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_runtime import infer_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--channel", type=int, nargs=3, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--archive", type=int, required=True)
    parser.add_argument("--grade", type=int, required=True)
    parser.add_argument("--keys", type=int, nargs="*", required=True)
    parser.add_argument("--boundary-steps", type=int, default=1)
    args = parser.parse_args()
    native = NativeSource(args.history.read_bytes())
    revision = NativeAlgebraicGraphRevision(native, args.channel, sequence=args.sequence)
    availability = (revision.history_owner.identity["namespace"], args.sequence)
    source = NativeAlgebraicGraphDatabase(revision, availability=availability).archive(args.archive, availability=availability)
    state = cpp_algebraic_grade_model(source, grade=args.grade, record_keys=args.keys,
                                    boundary_steps=args.boundary_steps, availability=availability)
    output = infer_model(state, source)
    record = model_record(source, state)
    record.attach_metadata(1, 0, "model_output", output)
    save_model(args.output, record)
    restored_state, restored_source = read_model_record(load_model(args.output))
    restored_output = infer_model(restored_state, restored_source)
    if restored_output.coefficient_digest != output.coefficient_digest:
        raise RuntimeError("restored grade model output differs")
    print(json.dumps({"model_digest": state.coefficient_digest, "output_digest": output.coefficient_digest,
                      "namespace": source.namespace, "sequence": source.sequence,
                      "grade": output.grade, "shape": list(output.values.shape),
                      "coefficient_basis": "power", "record": str(args.output)}))


if __name__ == "__main__":
    main()
