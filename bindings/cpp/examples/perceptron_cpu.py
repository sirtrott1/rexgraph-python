"""Measure sparse integer inference with explicit CPU worker counts."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np
from rexgraph_cpp import NativeSource, build_info

from rexgraph.cpp_perceptron import cpp_perceptron_input, cpp_perceptron_model, prepare_perceptron
from rexgraph.model_runtime import infer_model


def measure(features, cells, nonzeros, workers, repeats, seed, rows_per_tile, compare_prepared=False):
    rng = random.Random(seed)
    rows = tuple(tuple((rng.randrange(features), rng.choice((-3, -2, -1, 0, 1, 2, 3)))
                       for _ in range(nonzeros)) for _ in range(cells))
    entries = tuple((i, (i % 13) - 6) for i in range(0, features, max(1, features // 8192)))
    declaration = {
        "profile": "declared-exact-source-v1", "namespace": 500 + seed,
        "semantic_scale": 0, "max_grade": 1,
        "types": ({"name": "cpu-feature-task", "min_arity": 0, "max_arity": 0},),
        "witnesses": (),
        "relations": tuple({"grade": 1, "key": 10000 + i, "type": 0, "occurrences": ()}
                           for i in range(cells)),
        "carriers": (
            {"key": 11, "kind": "source_grade", "grade": 1, "coordinates": 0, "size": cells},
            {"key": 40, "kind": "explicit", "grade": 0, "coordinates": 900, "size": features},
            {"key": 41, "kind": "explicit", "grade": 0, "coordinates": 901, "size": 1}),
        "fields": ({"key": 4000, "carrier": 40, "entries": entries, "support": None},
                   {"key": 4001, "carrier": 41, "entries": ((0, 3),), "support": None}),
        "metrics": (),
    }
    start = time.perf_counter(); source = NativeSource.from_declaration(declaration)
    source_s = time.perf_counter() - start
    availability = tuple({"event_time": 1, "knowledge_time": 1, "label_available_time": 1,
                          "partition": "evaluation"} for _ in rows)
    start = time.perf_counter()
    state = cpp_perceptron_model(source, 11, 4000, 4001, availability,
                                feature_coordinates=900, time_coordinates=999, parameter_storage="packed")
    model_s = time.perf_counter() - start
    native = prepare_perceptron(state, source)
    reference = native.forward(rows)
    weights = dict(entries)
    expected = tuple(3 + sum(weights.get(i, 0) * value for i, value in row) for row in rows)
    assert reference["scores"] == expected
    scores_hash = hashlib.sha256(np.asarray(expected, dtype=np.int64).tobytes()).hexdigest()
    profiles = [(count, prepared) for count in (None, *workers)
                for prepared in ((False, True) if compare_prepared else (False,))]
    rng.shuffle(profiles)
    reports = []
    for count, prepared in profiles:
        kwargs = {} if count is None else {"cpu_workers": count, "rows_per_tile": rows_per_tile}
        start = time.perf_counter()
        inputs = cpp_perceptron_input(state, source, rows, prepare=prepared, **kwargs)
        capture_s = time.perf_counter() - start
        start = time.perf_counter()
        native_rows = native.prepare_rows(rows) if prepared else rows
        prepare_s = time.perf_counter() - start
        native.forward(native_rows, **kwargs); infer_model(state, source, inputs)
        native_times, core_times = [], []
        for _ in range(repeats):
            start = time.perf_counter(); result = native.forward(native_rows, **kwargs)
            native_times.append(time.perf_counter() - start)
            assert result["scores"] == expected and result["values"] == reference["values"]
            start = time.perf_counter(); output = infer_model(state, source, inputs)
            core_times.append(time.perf_counter() - start)
            assert output.execution_receipt["scores"] == expected
            assert tuple(output.values) == reference["values"]
        reports.append({"workers": count, "prepared": prepared, "input_capture_s": capture_s,
                        "native_prepare_s": prepare_s,
                        "native_call_s": statistics.median(native_times),
                        "core_call_s": statistics.median(core_times),
                        "native_samples_s": native_times, "core_samples_s": core_times,
                        "cpu_execution": result.get("cpu_execution")})
    return {"feature_count": features, "parameter_count": features + 1, "cells": cells,
            "nonzeros_per_row": nonzeros, "stored_nonzero_weights": sum(v != 0 for _, v in entries),
            "seed": seed, "repeats": repeats, "rows_per_tile": rows_per_tile,
            "source_s": source_s, "model_s": model_s, "scores_sha256": scores_hash,
            "boundary_source": inspect.getsourcefile(prepare_perceptron),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "engine_receipt": build_info(), "profiles": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=65536)
    parser.add_argument("--cells", type=int, default=8192)
    parser.add_argument("--nonzeros", type=int, default=64)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rows-per-tile", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-prepared", action="store_true")
    args = parser.parse_args()
    if (not 0 < args.features <= 1000000 or not 0 < args.cells <= 1000000
            or not 0 <= args.nonzeros <= 1000000 or args.repeats < 1 or args.seed < 0
            or any(not 0 < w <= 64 for w in args.workers) or not 0 < args.rows_per_tile <= 1000000):
        parser.error("invalid size, worker, repeat, seed or tile count")
    if (args.features + args.cells * args.nonzeros > 16000000
            or (args.cells * args.nonzeros + args.cells) * 8 > 64000000):
        parser.error("feature entries exceed the default native execution bounds")
    result = measure(args.features, args.cells, args.nonzeros, args.workers,
                     args.repeats, args.seed, args.rows_per_tile, args.compare_prepared)
    text = json.dumps(result, indent=2) + "\n"
    if args.output: args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
