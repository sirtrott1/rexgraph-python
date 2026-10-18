"""Compare complete scalar and finite rational batch inference on stored fields."""
import argparse
import json
import os
import platform
import resource
from fractions import Fraction as Q
from pathlib import Path
from statistics import median

import numpy as np
from integer_batches import ring, timed
from rexgraph_cpp import NativeSource, build_info

from rexgraph.cpp_learning import cpp_trainable_model, prepare_learning_model
from rexgraph.cpp_model import cpp_model, cpp_rational_batch_input
from rexgraph.model_runtime import infer_model
from rexgraph.type_accession import CoordinateSpace


def declared_fields(cells, samples, metric):
    declaration = ring(cells)
    declaration["carriers"] = (
        dict(key=10, kind="source_grade", grade=0, coordinates=0, size=cells),
        dict(key=11, kind="source_grade", grade=1, coordinates=0, size=cells),
        dict(key=40, kind="explicit", grade=0, coordinates=900, size=4),
        dict(key=41, kind="explicit", grade=0, coordinates=901, size=1))
    fields = []
    for s in range(samples):
        entries = tuple((i, Q((i+3*s) % 11-5, 7+i % 3)) for i in range(cells) if (i+3*s) % 11 != 5)
        fields.append(dict(key=1000+s, carrier=11, entries=entries, support=None))
    fields += [dict(key=4000, carrier=40, entries=tuple(enumerate((1, -1, 1, 2))), support=None),
               dict(key=4001, carrier=41, entries=((0, Q(1, 3)),), support=None)]
    declaration["fields"] = tuple(fields)
    if metric == "diagonal":
        declaration["metrics"] = tuple(dict(key=2000+k, carrier=10+k,
            diagonal=tuple(Q(2+k)+Q(i % (5+k), 3+k) for i in range(cells))) for k in range(2))
    return declaration


def training_contract():
    return dict(grade=1, channels=("identity", "hodge_down", "hodge_up", "hodge"),
        model="rational-ring", gains=4000, bias=4001, channel_coordinates=900, bias_coordinates=901,
        history_namespace=700, metadata_carrier=9900, metadata_tensor=9901, time_coordinates=9902,
        metadata_coordinates=9903, initial_parameter_knowledge_time=2, publication_cutoff=0,
        samples=(dict(input=1000, target=1000, event_time=3, knowledge_time=4,
                      label_available_time=5, partition="training"),))


def run(cells, batches, repeats, metric, adapters):
    build_ms, source = timed(lambda: NativeSource.from_declaration(declared_fields(cells, max(batches), metric)))
    limits = (4000000, 16000000, 16000000, 64000000, 1 << 24)
    options = dict(metric_selection=metric, metric_keys=(2000, 2001) if metric == "diagonal" else (), limits=limits)
    cases = []
    for adapter in adapters:
        prepare_ms, state = timed(lambda adapter=adapter: cpp_model(source, (10, 11), **options) if adapter == "field"
            else cpp_trainable_model(source, (10, 11), training_contract(), **options))
        plan = source.prepare((10, 11), **options) if adapter == "field" else prepare_learning_model(state, source)
        scalar_call = plan.execute if adapter == "field" else plan.forward
        batch_call = plan.execute_batch if adapter == "field" else plan.forward_batch
        for b in batches:
            keys = tuple(1000+s for s in range(b))
            fields = tuple(source.field(key) for key in keys)
            samples = CoordinateSpace("observations", tuple(f"sample-{s}" for s in range(b)))
            capture_ms, record = timed(lambda state=state, keys=keys, samples=samples:
                cpp_rational_batch_input(state, source, keys, samples))
            core_inputs = fields if adapter == "field" else tuple({"field_key": key} for key in keys)
            calls = {
                "native_scalar": lambda fields=fields, scalar_call=scalar_call: tuple(scalar_call(value)["values"] for value in fields),
                "native_batch": lambda fields=fields, batch_call=batch_call: batch_call(fields),
                "core_scalar": lambda state=state, core_inputs=core_inputs: tuple(infer_model(state, source, value) for value in core_inputs),
                "core_batch": lambda state=state, record=record: infer_model(state, source, record),
            }
            scalar = np.asarray(calls["native_scalar"](), dtype=object).T
            batched = calls["native_batch"]()
            assert np.array_equal(np.asarray(batched["values"], dtype=object), scalar)
            assert np.array_equal(calls["core_batch"]().values, scalar)
            durations = {name: [] for name in calls}
            for repeat in range(repeats):
                order = tuple(calls) if repeat % 2 == 0 else tuple(reversed(calls))
                for name in order:
                    elapsed, result = timed(calls[name]); durations[name].append(elapsed)
                    del result
            times = {name+"_ms": median(values) for name, values in durations.items()}
            cases.append(dict(adapter=adapter, batch_size=b, coefficients=cells*b, input_capture_ms=capture_ms,
                model_prepare_ms=prepare_ms, **times, durations_ms=durations,
                native_speedup=times["native_scalar_ms"]/times["native_batch_ms"],
                core_speedup=times["core_scalar_ms"]/times["core_batch_ms"],
                workspace_capacity=batched["workspace_capacity"], workspace_slots=batched["workspace_slots"],
                admitted_scalar_work=batched["scalar_work"], admitted_integer_bits=batched["integer_bits"],
                parameter_count=5 if adapter == "metric" else 0))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(cells=cells, repeats=repeats, metric=metric, limits=limits, native_source_action_work_limit=4000000,
        source_build_ms=build_ms, python=platform.python_version(), machine=platform.machine(),
        logical_cpus=os.cpu_count(), affinity_cpus=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        native_threads=1, engine=build_info(), process_peak_rss_bytes=peak if platform.system() == "Darwin" else peak*1024,
        profile="serial-gmp-rational-finite-batch-v1", cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=512)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--metric", choices=["unit", "diagonal"], default="diagonal")
    parser.add_argument("--adapters", choices=["field", "metric"], nargs="+", default=["field", "metric"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cells < 2 or args.repeats < 1 or any(b < 1 for b in args.batches):
        parser.error("cells, samples and repeats must be positive; cells must be at least two")
    report = run(args.cells, args.batches, args.repeats, args.metric, args.adapters)
    text = json.dumps(report, indent=2)+"\n"
    if args.output: args.output.write_text(text)
    print(text, end="")
