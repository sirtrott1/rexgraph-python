"""Run a complete original grade metric action and roundtrip its exact source record."""
import argparse
from pathlib import Path

from rexgraph_cpp import NativeAlgebraicGradeTensor, NativeAlgebraicMetricField

from rexgraph.cpp_algebraic_metric import (
    algebraic_grade_metric_record,
    read_algebraic_grade_metric_record,
)
from rexgraph.cpp_algebraic_query import read_algebraic_snapshot_record
from rexgraph.io import load, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grade", type=int, required=True)
    parser.add_argument("--vector-keys", type=int, nargs="+", required=True)
    parser.add_argument("--metric-keys", type=int, nargs="+", required=True)
    parser.add_argument("--operation", choices=("hodge", "coboundary", "inner", "adjacent_quadrance"), default="hodge")
    parser.add_argument("--max-sequence", type=int, required=True)
    parser.add_argument("--max-retained-coordinates", type=int, default=1000000)
    args = parser.parse_args()
    source = read_algebraic_snapshot_record(load(args.source))
    availability = (source.namespace, args.max_sequence)
    field = NativeAlgebraicMetricField(source, args.metric_keys, availability=availability,
                                       max_retained_coordinates=args.max_retained_coordinates)
    tensor = NativeAlgebraicGradeTensor(source, args.grade, args.vector_keys, availability=availability)
    record = algebraic_grade_metric_record(field, tensor, args.operation, availability=availability,
                                           other=tensor if args.operation == "inner" else None)
    save(args.output, record)
    observation, _, (_, _, _, restored) = read_algebraic_grade_metric_record(load(args.output))
    assert restored.coefficients == observation["detail"]["coefficients"]
    print(observation["seal"])


if __name__ == "__main__":
    main()
