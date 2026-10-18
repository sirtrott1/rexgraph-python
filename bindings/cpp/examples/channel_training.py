"""Train original exact gains with shared finite channel responses."""
import argparse
import json
import os
import resource
from fractions import Fraction as Q
from hashlib import sha256
from random import Random
from statistics import median

from integer_batches import timed
from rational_batches import declared_fields, training_contract
from rexgraph_cpp import NativeSource, build_info

from rexgraph.cpp_learning import cpp_trainable_model, cpp_training_batch
from rexgraph.model_runtime import train_model


def make_task(cells=16, count=64, metric="unit", seed=0, channels=None):
    """Declare independent gain coordinates, exact targets and held out fields."""
    if not 1 <= count <= 64 or cells < 2 or metric not in {"unit", "diagonal"}:
        raise ValueError("finite training requires 1 to 64 channels and at least two cells")
    random = Random(seed)
    selected = list(channels) if channels is not None else [
        ("identity", "hodge_down", "hodge_up", "hodge")[i % 4] for i in range(count)]
    if len(selected) != count:
        raise ValueError("channel count differs from its parameter carrier")
    random.shuffle(selected)
    declaration = declared_fields(cells, 4, metric)
    declaration["namespace"] = 88 + seed
    declaration["carriers"] = tuple({**carrier, "size": count} if carrier["key"] == 40 else carrier
                                    for carrier in declaration["carriers"])
    fields = []
    for field in declaration["fields"]:
        if field["key"] == 4000:
            entries = tuple((i, Q(random.randrange(-3, 4), 53+i % 7)) for i in range(count))
        elif field["key"] == 4001:
            entries = ((0, Q(-1, 17)),)
        else:
            entries = tuple((i, Q(random.randrange(-5, 6), 7+i % 3)) for i in range(cells))
        fields.append({**field, "entries": entries})
    declaration["fields"] = tuple(fields)
    source = NativeSource.from_declaration(declaration)
    options = dict(metric_selection=metric, metric_keys=(2000, 2001) if metric == "diagonal" else ())
    plan = source.prepare((10, 11), **options)
    diagonal = tuple(Q(3)+Q(i % 6, 4) for i in range(cells)) if metric == "diagonal" else (Q(1),)*cells
    traces = []
    responses = []
    for sample in range(4):
        field = source.field(1000+sample)
        actions = {"identity": field.values}
        actions.update((name, plan.execute(field, operation=name)["values"])
                       for name in ("hodge_down", "hodge_up", "hodge"))
        target = tuple(Q(1, 7)+2*actions["identity"][i]+Q(1, 3)*actions["hodge"][i]
                       for i in range(cells))
        fields.append(dict(key=5000+sample, carrier=11, entries=tuple(enumerate(target)), support=None))
        responses.append(tuple(actions[name] for name in selected) + ((Q(1),)*cells,))
        if sample < 2:
            traces.append(sum(weight*sum(column[i]**2 for column in responses[-1])
                              for i, weight in enumerate(diagonal)))
    declaration["fields"] = tuple(fields)
    source = NativeSource.from_declaration(declaration)
    contract = training_contract()
    contract.update(channels=tuple(selected), model="finite-channel-training", metadata_capacity=4096,
                    samples=tuple(dict(input=1000+s, target=5000+s, event_time=3+s, knowledge_time=4+s,
                                       label_available_time=5+s, partition="training") for s in range(2)))
    return source, options, contract, diagonal, Q(1)/(1+max(traces)), responses


def heldout_loss(trainer, source, diagonal):
    """Measure exact mean loss on fields excluded from the training declaration."""
    result = Q(0)
    for sample in (2, 3):
        actual = trainer.forward(source.field(1000+sample))["values"]
        target = source.field(5000+sample).values
        result += sum(weight*(value-expected)**2
                      for weight, value, expected in zip(diagonal, actual, target, strict=True))/2
    return result/2


def run(cells, count, metric, seed, steps, repeats):
    """Compare complete direct and cached native and Core training calls."""
    setup_ms, task = timed(lambda: make_task(cells, count, metric, seed))
    source, options, contract, diagonal, rate, _ = task
    plan = source.prepare((10, 11), **options)
    controls = dict(event_cutoff=20, knowledge_cutoff=20)
    native = {name: source.trainer(plan, contract, cache_samples=capacity)
              for name, capacity in (("direct", 0), ("cached", 2))}
    cold_ms, first = timed(lambda: native["cached"].advance(rate, **controls))
    assert first.snapshot == native["direct"].advance(rate, **controls).snapshot
    cached = native["cached"].advance(rate, steps=steps, **controls)
    direct = native["direct"].advance(rate, steps=steps, **controls)
    assert cached.snapshot == direct.snapshot
    fresh = NativeSource(source.archive)
    restored = fresh.trainer(fresh.prepare((10, 11), **options), contract,
                             archive=cached.snapshot["archive"], cache_samples=2)
    assert restored.advance(rate, **controls).snapshot == direct.advance(rate, **controls).snapshot
    states = {name: cpp_trainable_model(source, (10, 11), contract, training_cache_samples=capacity, **options)
              for name, capacity in (("direct", 0), ("cached", 2))}
    batches = {name: cpp_training_batch(state, source, rate=rate, **controls) for name, state in states.items()}
    calls = {"native_"+name: (lambda name=name: native[name].advance(rate, steps=steps, **controls)) for name in native}
    calls.update({"core_"+name: (lambda name=name: train_model(states[name], source, batches[name], steps=steps)) for name in states})
    assert calls["core_direct"]().payload == calls["core_cached"]().payload
    durations = {name: [] for name in calls}
    for repeat in range(repeats):
        for name in tuple(calls) if repeat % 2 == 0 else tuple(reversed(calls)):
            elapsed, result = timed(calls[name]); durations[name].append(elapsed)
            del result
    before, after = heldout_loss(native["cached"], source, diagonal), heldout_loss(cached, source, diagonal)
    snapshot = cached.snapshot
    parameters = snapshot["parameters"]
    return dict(cells=cells, parameter_count=count+1, metric=metric, seed=seed, steps=steps,
                source_setup_ms=setup_ms, cold_cached_update_ms=cold_ms, durations_ms=durations,
                medians_ms={name: median(values) for name, values in durations.items()},
                heldout_initial=str(before), heldout_final=str(after), heldout_ratio=float(after/before),
                maximum_parameter_bits=max(max(abs(v.numerator).bit_length(), v.denominator.bit_length()) for v in parameters),
                archive_bytes=len(snapshot["archive"]), archive_digest=sha256(snapshot["archive"]).hexdigest(),
                parameter_digest=sha256(repr(parameters).encode("ascii")).hexdigest(),
                execution=cached.training_execution, native_threads=1, cpu_affinity=sorted(os.sched_getaffinity(0)),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, engine=build_info(),
                scope="Bounded original scalar gains on finite channels, complete exact updates and checkpoint return")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--metric", choices=("unit", "diagonal"), default="unit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.channels <= 64 or args.cells < 2 or not 1 <= args.steps < 1024 or args.repeats < 1 or args.seed < 0:
        parser.error("cells, channels, steps, repeats or seed are outside the declared bounds")
    print(json.dumps(run(args.cells, args.channels, args.metric, args.seed, args.steps, args.repeats), indent=2))
