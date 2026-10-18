"""Exact observation tensors on original relation columns and boundary stalks."""
from collections.abc import Mapping
import numpy as np
from rexgraph.model_state import ModelBatch, freeze_tree

_PROFILE = "column-stalk-transport-observation-group-v1"
_PROVIDER = None
_TIED_PROVIDER = None


def _packet(state, sample):
    c = state.configuration
    if not isinstance(sample, Mapping) or set(sample) != {"input", "target"}:
        raise ValueError("closed original column tensor observation required")
    result = {}; count = 0
    for kind, grade in (("input", 1), ("target", 0)):
        records = []; previous = -1
        for cell, tensor in sample[kind]:
            if (type(cell) is not int or not previous < cell < len(c["stalk_axes"][grade]) or
                    type(tensor) is not bytes or not tensor or len(tensor) > c["max_bytes"]):
                raise ValueError("ordered original cells and bounded immutable tensor bytes required")
            previous = cell; count += len(tensor)+2
            if count > c["limits"][1]:raise ValueError("encoded observation exceeds admission")
            if kind == "target" and cell not in c["training_rows"]:
                raise ValueError("observed tensor lies outside original training stalks")
            records.append((cell,tensor))
        result[kind] = tuple(records)
    if not result["target"]:raise ValueError("nonempty observed original stalk tensors required")
    return result


def column_transport_observation(state, input_columns, target_stalks):
    """Encode complete sparse tensors on declared original cells; omitted inputs are zero."""
    from rexgraph.cpp_stalk_transport import validate_stalk_transport_state
    from rexgraph_cpp import encode_stalk_column_tensor
    state.check_state();validate_stalk_transport_state(state)
    if state.adapter_version not in {"3", "4"}:raise ValueError("column transport model required")
    fields = {}
    for kind, grade, values in (("input",1,input_columns),("target",0,target_stalks)):
        if not isinstance(values,Mapping):raise TypeError("original cells mapped to sparse stalk coordinates required")
        records = []
        for cell, entries in sorted(values.items()):
            if type(cell) is not int or not 0 <= cell < len(state.configuration["stalk_axes"][grade]):
                raise ValueError("original stalk cell required")
            tensor = (entries if type(entries) is bytes else encode_stalk_column_tensor(
                len(state.configuration["stalk_axes"][grade][cell]),entries,limits=state.configuration["limits"]))
            records.append((cell,tensor))
        fields[kind] = tuple(records)
    return freeze_tree(_packet(state, fields))


def column_transport_observation_group(state, observations):
    """Bind ordered column observations to one batch without padded scalar targets."""
    from rexgraph.cpp_stalk_transport import validate_stalk_transport_state
    state.check_state();validate_stalk_transport_state(state)
    if state.adapter_version not in {"3", "4"}:raise ValueError("column transport model required")
    packets = []; by_cell = {}; size = 0
    for sample in observations:
        if state.step+len(packets) >= state.configuration["limits"][0]:raise ValueError("observation count exceeds admission")
        packet = _packet(state,sample)
        size += sum(len(tensor)+2 for fields in packet.values() for _,tensor in fields)
        if size > state.configuration["limits"][1]:raise ValueError("column group exceeds entry admission")
        for cell,tensor in packet["target"]:by_cell.setdefault(cell,[]).append((len(packets),tensor))
        packets.append(packet)
    if not packets:raise ValueError("nonempty original column observations required")
    observed = np.zeros(len(state.space.keys),dtype=bool)
    for cell in by_cell:observed[cell]=True
    targets = dict(profile="encoded-original-stalk-targets-v1",source_reference=state.source.as_record(),
        space=(state.space.name,state.space.keys),samples=len(packets),
        stalks=tuple((cell,tuple(by_cell[cell])) for cell in sorted(by_cell)))
    return ModelBatch(state.source,state.space,targets,observed,
        dict(profile=_PROFILE,partition="training",source_reference=state.source.as_record(),observations=tuple(packets)))


def _batch_packets(state,batch):
    if not isinstance(batch,ModelBatch):raise TypeError("original column ModelBatch required")
    batch.check_state();b = batch.inputs;c = state.configuration
    if (not isinstance(batch,ModelBatch) or batch.space != state.space or not batch.source.matches(state.source) or
            batch.dependencies or batch.target_input is not None or not isinstance(b,Mapping) or
            set(b) != {"profile","partition","source_reference","observations"} or
            b["profile"] != _PROFILE or b["partition"] != "training" or
            dict(b["source_reference"]) != state.source.as_record() or not isinstance(batch.targets,Mapping) or
            batch.targets["profile"] != "encoded-original-stalk-targets-v1"):
        raise ValueError("bound original column tensor batch required")
    packets = tuple(_packet(state,o) for o in b["observations"])
    if not packets or len(packets) > c["limits"][0]:raise ValueError("bounded nonempty column group required")
    by_cell = {};size = 0
    for sample,packet in enumerate(packets):
        size += sum(len(tensor)+2 for fields in packet.values() for _,tensor in fields)
        for cell,tensor in packet["target"]:by_cell.setdefault(cell,[]).append((sample,tensor))
    if size > c["limits"][1]:raise ValueError("column group exceeds entry admission")
    expected = tuple((cell,tuple(by_cell[cell])) for cell in sorted(by_cell))
    if batch.targets["samples"] != len(packets) or batch.targets["stalks"] != expected:
        raise ValueError("batch targets differ from complete original tensor observations")
    return packets


def validate_column_history(state):
    history = state.payload["observations"];size = 0;constraints = 0;first = 0;digest = None
    for index,o in enumerate(history):
        if not isinstance(o,Mapping) or set(o) != {"input","target","batch","batch_checkpoint","batch_sample","batch_first"}:
            raise ValueError("complete column observation and ordered original batch witness required")
        packet = _packet(state,{k:o[k] for k in ("input","target")})
        if o["batch_sample"] == 0:
            first = index;digest = o["batch"]
            if not isinstance(o["batch_checkpoint"],Mapping):raise ValueError("first observation must retain its original batch")
        elif o["batch_checkpoint"] is not None:raise ValueError("batch checkpoint belongs only to its first observation")
        if (type(o["batch_sample"]) is not int or type(o["batch_first"]) is not int or
                o["batch_sample"] != index-first or o["batch_first"] != first or o["batch"] != digest or
                not isinstance(digest,str) or len(digest)!=64):
            raise ValueError("column observation batch order or identity differs")
        size += sum(len(tensor)+2 for fields in packet.values() for _,tensor in fields)
        constraints += sum(len(state.configuration["stalk_axes"][0][cell]) for cell,_ in packet["target"])
    return size,constraints


def verify_column_history(state):
    from rexgraph.model_codec import unpack_model
    history = state.payload["observations"];index = 0
    while index < len(history):
        record = history[index];batch = unpack_model(record["batch_checkpoint"])
        if not isinstance(batch,ModelBatch) or batch.coefficient_digest != record["batch"]:
            raise ValueError("column history differs from its original batch digest")
        packets = _batch_packets(state,batch)
        if index+len(packets) > len(history):raise ValueError("incomplete original column batch history")
        for sample,packet in enumerate(packets):
            o = history[index+sample]
            if (o["batch_first"] != index or o["batch_sample"] != sample or o["batch"] != record["batch"] or
                    freeze_tree({k:o[k] for k in ("input","target")}) != freeze_tree(packet)):
                raise ValueError("column tensor history differs from its original batch")
        index += len(packets)


def train_columns(state,source,batch,*,steps):
    from rexgraph.cpp_stalk_transport import _prepare,_remember
    from rexgraph.model_codec import pack_model
    packets = _batch_packets(state,batch)
    if steps != len(packets):raise ValueError("learning steps must equal original column observations")
    parent = _prepare(state,source);checkpoint = pack_model(batch);digest = batch.coefficient_digest
    acquired = tuple(dict(packet,batch=digest,batch_checkpoint=checkpoint if sample==0 else None,
        batch_sample=sample,batch_first=state.step) for sample,packet in enumerate(packets))
    plan = parent.append_columns(packets)
    child = state.child(payload=dict(observations=(*state.payload["observations"],*acquired),fit=plan.receipt),step=state.step+len(packets))
    _remember(child,plan);_prepare(child,source);return child


def enable_column_transport_provider(version="3"):
    """Register exact column acquisition with the existing full affine inference path."""
    from rexgraph.cpp_stalk_transport import _infer,_prepare,cpp_stalk_transport_model
    from rexgraph.model_contract import ModelProvider,register_model_provider
    if version not in {"3", "4"}:raise ValueError("supported column transport provider version required")
    global _PROVIDER, _TIED_PROVIDER
    from rexgraph.cpp_stalk_transport import _lock
    with _lock:
        if _PROVIDER is None:
            def create(source,*,adapter="cpp_stalk_transport",**configuration):
                if adapter != "cpp_stalk_transport":raise ValueError("column transport adapter required")
                if configuration.get("parameter_space") is not None:raise ValueError("column provider version 3 requires independent parameters")
                return cpp_stalk_transport_model(source,column_fields=True,**configuration)
            _PROVIDER=ModelProvider("rexgraph.cpp.stalk_transport","3",(("cpp_stalk_transport","3"),),
                _infer,create=create,prepare=_prepare,train=train_columns,target_profiles=("encoded-original-stalk-targets-v1",))
            register_model_provider(_PROVIDER)
        if _TIED_PROVIDER is None:
            def create_tied(source,*,adapter="cpp_stalk_transport",**configuration):
                if adapter != "cpp_stalk_transport" or configuration.get("parameter_space") is None:
                    raise ValueError("column provider version 4 requires named templates")
                return cpp_stalk_transport_model(source,column_fields=True,**configuration)
            _TIED_PROVIDER=ModelProvider("rexgraph.cpp.stalk_transport","4",(("cpp_stalk_transport","4"),),
                _infer,create=create_tied,prepare=_prepare,train=train_columns,target_profiles=("encoded-original-stalk-targets-v1",))
            register_model_provider(_TIED_PROVIDER)
        return _TIED_PROVIDER if version == "4" else _PROVIDER


def replay_column_model(state,source):
    """Replay original encoded column batches into the current native implementation."""
    from rexgraph.cpp_stalk_transport import cpp_stalk_transport_model,_receipt_matches
    from rexgraph.model_codec import unpack_model
    from rexgraph.model_runtime import train_model
    verify_column_history(state);c = state.configuration
    extra = {k:c[k] for k in ("parameter_space", "parameter_basis")} if state.adapter_version == "4" else {}
    current = cpp_stalk_transport_model(source,c["stalk_axes"],c["fixed_maps"],c["parameter_slots"],c["training_rows"],
        limits=c["limits"],max_bytes=c["max_bytes"],source_accession=c["source_accession"],reference=state.source,column_fields=True,**extra)
    for o in state.payload["observations"]:
        if o["batch_sample"] == 0:
            batch = unpack_model(o["batch_checkpoint"])
            current = train_model(current,source,batch,steps=len(batch.inputs["observations"]))
    changing = {"engine_headers","binding","preparation_steps","retained_entries"}
    if not _receipt_matches({k:v for k,v in state.payload["fit"].items() if k not in changing},
                            {k:v for k,v in current.payload["fit"].items() if k not in changing}):
        raise ValueError("column transport replay changed the complete mathematical family")
    return current
