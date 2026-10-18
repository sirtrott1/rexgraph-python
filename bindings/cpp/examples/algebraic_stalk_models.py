"""Select an original stalk, apply a carrier model and check its restored exact result."""
import argparse
import json
from pathlib import Path

from rexgraph.cpp_algebraic import cpp_algebraic_input, cpp_algebraic_model
from rexgraph.cpp_algebraic_stalk import (
    algebraic_grade_stalk_record,
    prepare_algebraic_grade_stalks,
    read_algebraic_grade_stalk_record,
)
from rexgraph.io import load, save
from rexgraph.io.model_record import load_model, read_model_record
from rexgraph.model_runtime import infer_model
from rexgraph.model_state import thaw_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cell", type=int, required=True)
    parser.add_argument("--operator", type=int, required=True)
    parser.add_argument("--input-key", type=int, required=True)
    parser.add_argument("--operation", choices=("apply", "solve", "quadrance", "as_covector", "evaluate"), default="apply")
    args = parser.parse_args()
    grade_model, snapshot = read_model_record(load_model(args.model))
    grade_output = infer_model(grade_model, snapshot)
    selection = prepare_algebraic_grade_stalks(grade_output, snapshot).select(args.cell)
    archive = snapshot.archive_owner
    carrier_model = cpp_algebraic_model(archive, operator_key=args.operator, input_key=args.input_key, operation=args.operation)
    captured = cpp_algebraic_input(carrier_model, archive, selection)
    output = infer_model(carrier_model, archive, captured)
    record = algebraic_grade_stalk_record(selection, snapshot)
    record.attach_metadata(1, 0, "carrier_model", carrier_model)
    record.attach_metadata(1, 0, "carrier_input", thaw_tree(captured))
    record.attach_metadata(1, 0, "carrier_output", output)
    save(args.output, record)
    restored = load(args.output)
    _, owner, _ = read_algebraic_grade_stalk_record(restored)
    actual = infer_model(restored.get_metadata(1, 0, "carrier_model"), owner.archive_owner,
                         restored.get_metadata(1, 0, "carrier_input"))
    if actual.coefficient_digest != output.coefficient_digest:
        raise RuntimeError("restored carrier model output differs")
    print(json.dumps({"grade": selection.grade, "cell": args.cell, "operation": args.operation,
                      "selection_digest": selection.coefficient_digest, "output_digest": output.coefficient_digest,
                      "origin_depth": actual.execution_receipt["origin_depth"], "record": str(args.output)}))


if __name__ == "__main__":
    main()
