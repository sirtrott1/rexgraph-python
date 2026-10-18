"""Save a fixed metric model and restore exact inference on its graph publication."""
import argparse
import json
from pathlib import Path

from rexgraph_cpp import NativeAlgebraicGraphDatabase, NativeAlgebraicGraphRevision, NativeSource

from rexgraph.cpp_algebraic_grade_metric_model import cpp_algebraic_grade_metric_model
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
    parser.add_argument("--metrics", type=int, nargs="*", required=True)
    parser.add_argument("--boundary-steps", type=int, default=0)
    parser.add_argument("--operation", choices=("hodge", "coboundary", "inner", "adjacent_quadrance"), default="hodge")
    parser.add_argument("--other-grade", type=int)
    parser.add_argument("--other-keys", type=int, nargs="*")
    parser.add_argument("--other-boundary-steps", type=int, default=0)
    args = parser.parse_args()
    native = NativeSource(args.history.read_bytes())
    revision = NativeAlgebraicGraphRevision(native, args.channel, sequence=args.sequence)
    availability = (revision.history_owner.identity["namespace"], args.sequence)
    source = NativeAlgebraicGraphDatabase(revision, availability=availability).archive(args.archive, availability=availability)
    request = {"grade": args.grade, "record_keys": args.keys, "boundary_steps": args.boundary_steps}
    other = None if args.other_grade is None else {
        "grade": args.other_grade, "record_keys": args.other_keys, "boundary_steps": args.other_boundary_steps}
    state = cpp_algebraic_grade_metric_model(source, request=request, metric_keys=args.metrics,
        operation=args.operation, other_request=other, availability=availability)
    output = infer_model(state, source)
    record = model_record(source, state)
    record.attach_metadata(1, 0, "model_output", output)
    save_model(args.output, record)
    restored_state, restored_source = read_model_record(load_model(args.output))
    restored_output = infer_model(restored_state, restored_source)
    if restored_output.coefficient_digest != output.coefficient_digest:
        raise RuntimeError("restored metric model output differs")
    print(json.dumps({"model_digest": state.coefficient_digest, "output_digest": output.coefficient_digest,
        "namespace": source.namespace, "sequence": source.sequence, "operation": args.operation,
        "grade": output.grade, "shape": list(output.values.shape), "coefficient_basis": "power",
        "record": str(args.output)}))


if __name__ == "__main__":
    main()
