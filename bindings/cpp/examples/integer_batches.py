"""Compare complete scalar and packed batch inference on a declared integer model."""
import argparse
import json
import os
import platform
import resource
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from rexgraph_cpp import NativeSource, build_info

from rexgraph.cpp_learning import cpp_integer_batch_input, cpp_integer_input, cpp_integer_model
from rexgraph.model_runtime import infer_model
from rexgraph.type_accession import CoordinateSpace


def ring(cells):
    def occurrence(key, lower, negative):
        return dict(key=key, lower=lower, exists=True, negative=negative, magnitude=1)
    return dict(profile="declared-exact-source-v1", namespace=88, semantic_scale=0, max_grade=1,
        types=(dict(name="edge", min_arity=2, max_arity=2),), witnesses=(),
        relations=tuple(dict(grade=1, key=100000+i, type=0,
            occurrences=(occurrence(2*i, i, False), occurrence(2*i+1, (i+1) % cells, True))) for i in range(cells)),
        carriers=(dict(key=11, kind="source_grade", grade=1, coordinates=0, size=cells),), fields=(), metrics=())


def timed(call):
    start = perf_counter()
    result = call()
    return (perf_counter()-start)*1000, result


def run(cells, batches, repeats):
    setup_ms, source = timed(lambda: NativeSource.from_declaration(ring(cells)))
    layers = (dict(channels=("identity", "hodge_down", "hodge_up", "hodge"),
                   gains=(1, 1, -1, 1), bias=1, activation="relu"),
              dict(channels=("identity", "hodge_down", "hodge_up", "hodge"),
                   gains=(2, -1, 1, 0), bias=0, activation="identity"))
    limits = (4000000, 16000000, 16000000, 2000000000, 1 << 24)
    prepare_ms, state = timed(lambda: cpp_integer_model(source, 11, layers, limits=limits))
    plan = source.integer_network(11, layers, limits=limits)
    cases = []
    for batch_size in batches:
        values = (np.arange(cells*batch_size, dtype=np.int64) % 7-3).reshape(cells, batch_size)
        samples = CoordinateSpace("observations", tuple(f"sample-{i}" for i in range(batch_size)))
        copy_ms, packed = timed(lambda values=values: source.integer_batch_input(11, values, limits=limits))
        inputs = tuple(source.integer_input(11, tuple(values[:, j])) for j in range(batch_size))
        core_inputs = tuple(cpp_integer_input(state, source, values[:, j]) for j in range(batch_size))
        core_batch = cpp_integer_batch_input(state, source, values, samples)
        calls = {
            "native_scalar": lambda inputs=inputs: tuple(plan.forward(value)["values"] for value in inputs),
            "native_batch": lambda packed=packed: plan.forward_batch(packed),
            "core_scalar": lambda core_inputs=core_inputs: tuple(infer_model(state, source, value) for value in core_inputs),
            "core_batch": lambda core_batch=core_batch: infer_model(state, source, core_batch),
        }
        scalar = np.asarray(calls["native_scalar"](), dtype=object).T
        batched = calls["native_batch"]()
        assert np.array_equal(np.asarray(batched["values"]), scalar)
        assert np.array_equal(calls["core_batch"]().values, scalar)
        durations = {name: [] for name in calls}
        for repeat in range(repeats):
            names = tuple(calls) if repeat % 2 == 0 else tuple(reversed(calls))
            for name in names:
                elapsed, result = timed(calls[name]); durations[name].append(elapsed)
                del result
        times = {name+"_ms": median(values) for name, values in durations.items()}
        cases.append(dict(batch_size=batch_size, coefficients=cells*batch_size,
            input_copy_ms=copy_ms, **times, durations_ms=durations,
            native_speedup=times["native_scalar_ms"]/times["native_batch_ms"],
            core_speedup=times["core_scalar_ms"]/times["core_batch_ms"],
            workspace_capacity=batched["workspace_capacity"],
            workspace_bytes=batched["workspace_slots"]*8))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(cells=cells, layers=layers, limits=limits, repeats=repeats,
        parameter_count=sum(len(layer["gains"])+1 for layer in layers),
        source_build_ms=setup_ms, model_prepare_ms=prepare_ms,
        python=platform.python_version(), machine=platform.machine(),
        logical_cpus=os.cpu_count(), affinity_cpus=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        native_threads=1, engine=build_info(),
        process_peak_rss_bytes=peak if platform.system() == "Darwin" else peak*1024,
        profile="serial-checked-int64-lattice-batch-v1", cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=4096)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cells < 2 or args.repeats < 1 or any(b < 1 for b in args.batches):
        parser.error("cells, samples and repeats must be positive; cells must be at least two")
    report = run(args.cells, args.batches, args.repeats)
    text = json.dumps(report, indent=2)+"\n"
    if args.output: args.output.write_text(text)
    print(text, end="")
