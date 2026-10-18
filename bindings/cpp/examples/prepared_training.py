"""Compare direct and cached metric pairings for exact training on a fixed source."""
import argparse
import json
import os
import platform
import resource
from fractions import Fraction as Q
from pathlib import Path
from statistics import median

from integer_batches import timed
from rational_batches import declared_fields, training_contract
from rexgraph_cpp import NativeSource, build_info

from rexgraph.cpp_learning import cpp_trainable_model, cpp_training_batch
from rexgraph.model_runtime import infer_model, train_model


def exact(value):
    return format(value.numerator, "x") + "/" + format(value.denominator, "x")


def input_key(sample):
    return 1000+sample if sample < 1000 else 100000+sample


def make_task(cells, metric, training_samples=4, heldout_samples=2):
    declaration = declared_fields(cells, training_samples+heldout_samples, metric)
    declaration["fields"] = tuple({**field, "key": input_key(field["key"]-1000)}
                                  if field["carrier"] == 11 else field
                                  for field in declaration["fields"])
    declaration["fields"] = tuple({**field, "entries": ()} if field["key"] in {4000, 4001}
                                  else field for field in declaration["fields"])
    source = NativeSource.from_declaration(declaration)
    options = dict(metric_selection=metric, metric_keys=(2000, 2001) if metric == "diagonal" else ())
    plan = source.prepare((10, 11), **options)
    diagonal = tuple(Q(3)+Q(i % 6, 4) for i in range(cells)) if metric == "diagonal" else (Q(1),)*cells
    fields = list(declaration["fields"])
    traces = []
    known = (Q(2), Q(-1, 5), Q(0), Q(1, 3), Q(1, 7))
    for sample in range(training_samples+heldout_samples):
        field = source.field(input_key(sample))
        responses = (field.values, *(plan.execute(field, operation=operation)["values"]
                                    for operation in ("hodge_down", "hodge_up", "hodge")))
        values = tuple(known[-1]+sum(gain*response[i] for gain, response in zip(known[:-1], responses, strict=True))
                       for i in range(cells))
        fields.append(dict(key=5000+sample, carrier=11,
                           entries=tuple((i, value) for i, value in enumerate(values) if value), support=None))
        if sample < training_samples:
            traces.append(sum(weight*(1+sum(response[i]**2 for response in responses))
                              for i, weight in enumerate(diagonal)))
    declaration["fields"] = tuple(fields)
    source = NativeSource.from_declaration(declaration)
    contract = training_contract()
    contract["samples"] = tuple(dict(input=input_key(s), target=5000+s, event_time=3+s, knowledge_time=4+s,
                                      label_available_time=5+s, partition="training") for s in range(training_samples))
    return source, options, contract, diagonal, Q(1)/(1+max(traces)), training_samples, heldout_samples


def losses(state, source, diagonal, sample_indices):
    total = Q(0)
    for sample in sample_indices:
        prediction = infer_model(state, source, {"field_key": input_key(sample)}).values
        target = source.field(5000+sample).values
        total += sum(weight*(value-expected)**2 for weight, value, expected in zip(diagonal, prediction, target, strict=True))/2
    return total/len(sample_indices)


def run(cells, metric, steps, repeats, *, training_samples=4, heldout_samples=2, metadata_capacity=None,
        capture="stored"):
    setup_ms, task = timed(lambda: make_task(cells, metric, training_samples, heldout_samples))
    source, options, contract, diagonal, rate, samples, heldout = task
    if metadata_capacity is not None:
        contract["metadata_capacity"] = metadata_capacity
    plan = source.prepare((10, 11), **options)
    scalar = source.trainer(plan, contract, cache_samples=0)
    prepared = source.trainer(plan, contract, cache_samples=samples)
    advance = dict(event_cutoff=max(20,samples+2), knowledge_cutoff=max(20,samples+4))
    cold_direct_ms, one_direct = timed(lambda: scalar.advance(rate, **advance))
    cold_cached_ms, one_cached = timed(lambda: prepared.advance(rate, **advance))
    assert one_direct.snapshot == one_cached.snapshot
    warm = prepared.advance(rate, steps=steps, **advance)
    reference = scalar.advance(rate, steps=steps, **advance)
    assert warm.snapshot == reference.snapshot
    core_scalar = cpp_trainable_model(source, (10, 11), contract, training_cache_samples=0, **options)
    core_cached = cpp_trainable_model(source, (10, 11), contract, training_cache_samples=samples, **options)
    batches, capture_ms = {}, {}
    for mode, state in (("direct", core_scalar), ("cached", core_cached)):
        elapsed, observations = timed(lambda: cpp_training_batch(state, source, rate=rate, capture=capture, **advance))
        batches[mode], capture_ms[mode] = observations, elapsed
    calls = {
        "native_direct": lambda: scalar.advance(rate, steps=steps, **advance),
        "native_cached": lambda: prepared.advance(rate, steps=steps, **advance),
        "core_direct": lambda: train_model(core_scalar, source, batches["direct"], steps=steps),
        "core_cached": lambda: train_model(core_cached, source, batches["cached"], steps=steps),
    }
    assert calls["core_direct"]().payload == calls["core_cached"]().payload
    durations = {name: [] for name in calls}
    final = None
    for repeat in range(repeats):
        order = tuple(calls) if repeat % 2 == 0 else tuple(reversed(calls))
        for name in order:
            elapsed, result = timed(calls[name]); durations[name].append(elapsed)
            if name == "core_cached": final = result
            del result
    medians = {name+"_ms": median(values) for name, values in durations.items()}
    quality = {}
    for name, indices in (("training", range(samples)), ("heldout", range(samples, samples+heldout))):
        before = losses(core_cached, source, diagonal, indices)
        after = losses(final, source, diagonal, indices)
        quality[name] = dict(initial_loss_hex=exact(before), final_loss_hex=exact(after),
                             initial_loss=float(before), final_loss=float(after), ratio=float(after/before))
    assert quality["training"]["ratio"] < 1 and quality["heldout"]["ratio"] < 1
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(cells=cells, metric=metric, training_samples=samples, heldout_samples=heldout,
        steps=steps, repeats=repeats, parameter_count=5, metadata_capacity=contract.get("metadata_capacity",4096),
        training_capture=capture, batch_capture_ms=capture_ms,
        batch_target_coefficients=int(batches["cached"].targets.size),
        parameter_archive_bytes=len(final.payload["archive"]), cutoffs=advance,
        rate_hex=exact(rate), source_task_setup_ms=setup_ms,
        cold_first_direct_ms=cold_direct_ms, cold_first_cached_ms=cold_cached_ms,
        durations_ms=durations, **medians, native_speedup=medians["native_direct_ms"]/medians["native_cached_ms"],
        core_speedup=medians["core_direct_ms"]/medians["core_cached_ms"], quality=quality,
        training_execution=prepared.training_execution, python=platform.python_version(),
        logical_cpus=os.cpu_count(), affinity_cpus=len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
        native_threads=1, engine=build_info(), process_peak_rss_bytes=peak if platform.system()=="Darwin" else peak*1024,
        scope="Complete serial fixed-source metric updates, canonical history publication and binding/Core return; five original scalar parameters; shared finite responses; bounded serial learner")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--metric", choices=["unit","diagonal"], default="diagonal")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--training-samples", type=int, default=4)
    parser.add_argument("--heldout-samples", type=int, default=2)
    parser.add_argument("--metadata-capacity", type=int)
    parser.add_argument("--capture", choices=["stored", "values"], default="stored")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cells < 2 or not 4 <= args.steps <= 1024 or args.repeats < 1:
        parser.error("cells must be at least two, steps from 4 through 1024, repeats positive")
    if not 1 <= args.training_samples <= 1024 or not 1 <= args.heldout_samples <= 1024:
        parser.error("training and held out sample counts must each be from 1 through 1024")
    if args.metadata_capacity is not None and not 8 <= args.metadata_capacity <= 65536:
        parser.error("metadata capacity must be from 8 through 65536 bytes")
    report = run(args.cells, args.metric, args.steps, args.repeats, training_samples=args.training_samples,
                 heldout_samples=args.heldout_samples, metadata_capacity=args.metadata_capacity, capture=args.capture)
    text = json.dumps(report, indent=2)+"\n"
    if args.output: args.output.write_text(text)
    print(text, end="")
