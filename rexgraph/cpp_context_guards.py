"""Context conditions certified on complete retained boundary subcomplexes."""
from collections import OrderedDict
from collections.abc import Mapping
from itertools import permutations
from threading import RLock

_PROFILE = "ordered-context-boundary-v1"
_DICTIONARY_PROFILE = "ordered-context-dictionary-v1"
_JOINT_DICTIONARY_PROFILE = "ordered-context-dictionary-union-v1"
_DICTIONARY_USE_PROFILE = "ordered-context-dictionary-use-union-v1"
_lock = RLock()
_owners = OrderedDict()


def _occurrences(c, row):
    from rexgraph.cpp_relational_sections import _context
    _, chart = _context(c, row)
    accession = c["field"]["source_accession"]
    qid = accession["relations"][row]["provenance"]["query"]
    return tuple(o for o in accession["occurrences"] if isinstance(o["provenance"], Mapping)
        and o["provenance"].get("query") == qid and not o["role"].startswith("phrase-member:")
        and ("word" in o["provenance"] or "binding" in o["provenance"])), chart


def _context_rows(c, row):
    relations = c["field"]["source_accession"]["relations"]
    qid = relations[row]["provenance"]["query"]
    return (row, *(i for i, r in enumerate(relations) if r["kind"] == "recorded-phrase"
        and r["provenance"]["query"] == qid))


def guard_anchor_cells(c, support, guard):
    """Resolve declared role anchors on their original context boundaries."""
    if (not isinstance(guard, Mapping) or set(guard) != {"profile", "anchors"}
            or guard["profile"] not in {_PROFILE, _DICTIONARY_PROFILE, _JOINT_DICTIONARY_PROFILE, _DICTIONARY_USE_PROFILE}):
        raise ValueError("closed ordered context guard required")
    rows = tuple(dict.fromkeys(row for row, _ in support))
    if guard['profile'] == _DICTIONARY_USE_PROFILE and 'dictionary_links' not in c:
        raise ValueError('dictionary use guard requires a dictionary-bound model')
    if "dictionary_links" in c:
        axes = {c["layers"].index(link[0]) for link in c["dictionary_links"]}
        if (guard["profile"] != _DICTIONARY_USE_PROFILE
                or any(not axes.issubset(a for r, a in support if r == row) for row in rows)):
            raise ValueError("dictionary-bound guards require a union map and every linked layer")
    if len(guard["anchors"]) != len(rows):
        raise ValueError("guard anchors must cover every distinct source context")
    pins = []
    for row, anchors in zip(rows, guard["anchors"], strict=True):
        occurrences, _ = _occurrences(c, row)
        roles = tuple(o["role"] for o in occurrences)
        target_roles = set(role for role in roles if role.startswith("target:"))
        if (not anchors or len(set(anchors)) != len(anchors)
                or any(not isinstance(role, str) or role not in roles for role in anchors)
                or not target_roles or not target_roles.issubset(anchors)):
            raise ValueError("guard must pin every original target role and only retained roles")
        for o in occurrences:
            if o["role"] in anchors:
                pins.append(o["lower"])
    return tuple(dict.fromkeys(pins))


def guarded_annotation_group(state, name, support, anchors, *, dictionary=False, joint_dictionary=False):
    """Declare a typed annotation group with exact context role anchors."""
    if state.adapter != "cpp_relational_sections" or state.adapter_version not in {"2", "3", "4"}:
        raise ValueError("context guards require joint model version 2")
    if type(dictionary) is not bool or type(joint_dictionary) is not bool:
        raise ValueError("dictionary condition must be explicitly boolean")
    support = tuple(tuple(pair) for pair in support)
    guard = dict(profile=(_DICTIONARY_USE_PROFILE if state.adapter_version in {'3', '4'} else _JOINT_DICTIONARY_PROFILE)
        if joint_dictionary else _DICTIONARY_PROFILE if dictionary else _PROFILE,
        anchors=tuple(tuple(row) for row in anchors))
    if name not in state.configuration["compatibility_types"]:
        raise ValueError("guard requires a declared compatibility type")
    if (not support or len(set(support)) != len(support)
            or any(len(pair) != 2 or any(type(i) is not int for i in pair) for pair in support)
            or tuple(a for _, a in support) != state.configuration["compatibility_types"][name]):
        raise ValueError("guard support differs from its declared layer signature")
    guard_anchor_cells(state.configuration, support, guard)
    return dict(type=name, support=support, guard=guard)


def prepare_context_guards(state, source):
    """Check native context occurrences and pin their owning source publication."""
    from rexgraph_cpp import PreparedRelationGuards, guards_build_info
    c = state.configuration
    if dict(c["guard_receipt"]) != guards_build_info():
        raise ValueError("context guards require their retained native implementation")
    key = c["source_archive_digest"], tuple(c["guard_limits"]), c["field"]["max_bytes"]
    with _lock:
        owner = _owners.get(key)
        if owner is None:
            owner = PreparedRelationGuards(source.archive, limits=c["guard_limits"][1:], max_bytes=c["field"]["max_bytes"])
            _owners[key] = owner
            while len(_owners) > 4:
                _owners.popitem(last=False)
        _owners.move_to_end(key)
    accession = c["field"]["source_accession"]
    for row, relation in enumerate(accession["relations"]):
        native = owner.boundary(row)
        if native["type"] != relation["kind"]:
            raise ValueError("context accession differs from its full native occurrence boundary")
        for flag in native["occurrences"]:
            record = accession["occurrences"][flag["slot"]]
            if (record["key"] != flag["key"]
                    or record["lower"] != flag["lower"] or (record["role"],) != flag["roles"]
                    or not flag["exists"] or flag["negative"] or flag["coefficient"] != "1"):
                raise ValueError("context role accession differs from its retained native occurrence")
        if relation["kind"] != "observed-context":
            continue
        records, chart = _occurrences(c, row)
        if tuple(accession["occurrences"][o["slot"]] for o in native["occurrences"]) != records:
            raise ValueError("context accession differs from its full native occurrence boundary")
        target_words = tuple(i for i, (a,b) in enumerate(chart["spans"]) if a < chart["target"][1] and b > chart["target"][0])
        if not target_words:
            raise ValueError("context target must overlap an original word occurrence")
        word_indices = []
        for record in records:
            p = record["provenance"]
            if "word" in p:
                i = p["word"]
                if (type(i) is not int or not 0 <= i < len(chart["words"])
                        or tuple(p["span"]) != tuple(chart["spans"][i])
                        or tuple(accession["lower"][record["lower"]]) != ("literal-utf8", chart["words"][i].encode().hex())):
                    raise ValueError("context literal identity differs from its ordered byte interval")
                role = ("target:"+str(i-target_words[0]) if i in target_words else
                        "left:"+str(target_words[0]-i) if i < target_words[0] else "right:"+str(i-target_words[-1]))
                if record["role"] != role:
                    raise ValueError("context role differs from its original target interval")
                word_indices.append(i)
            else:
                i = p["binding"]
                if type(i) is not int or not 0 <= i < len(chart["bindings"]):
                    raise ValueError("context dictionary binding is outside its literal chart")
                binding = chart["bindings"][i]
                if (tuple(p["span"]) != tuple(binding["span"]) or p["kind"] != binding["kind"]
                        or tuple(accession["lower"][record["lower"]]) !=
                        ("dictionary-record", (binding["entry"], binding["record_kind"], binding["record_ordinal"]))):
                    raise ValueError("context dictionary identity differs from its retained binding")
                first, last = binding["first"], binding["last"]
                if (type(first) is not int or type(last) is not int or not 0 <= first <= last < len(chart["words"])
                        or tuple(binding["span"]) != (chart["spans"][first][0], chart["spans"][last][1])):
                    raise ValueError("dictionary binding differs from its complete literal interval")
                raw = bytes.fromhex(chart["raw_utf8_hex"])[slice(*binding["span"])]
                record_raw = bytes.fromhex(binding["record_utf8_hex"])
                if (binding["kind"] == "exact" and raw != record_raw or
                        binding["kind"] == "casefold" and raw.decode().casefold() != record_raw.decode().casefold() or
                        binding["kind"] not in {"exact", "casefold"}):
                    raise ValueError("dictionary record differs from its declared literal comparison")
        if tuple(word_indices) != tuple(range(len(chart["words"]))):
            raise ValueError("context must retain every original ordered word occurrence")
    return owner


def _dictionary_pairs(c, origins, targets, witness, *, include_unchanged=False):
    """Require actual mapped bindings for each changed literal participant."""
    accession = c["field"]["source_accession"]
    lower = accession["lower"]
    mapping = dict(witness["cell_map"])
    pairs = tuple((a,b) for a,b in witness["cell_map"] if (a != b or include_unchanged) and lower[a][0] == "dictionary-record")
    for origin, target in zip(origins, targets, strict=True):
        old, _ = _occurrences(c, origin); new, _ = _occurrences(c, target)
        old_words = tuple(o for o in old if "word" in o["provenance"])
        new_words = tuple(o for o in new if "word" in o["provenance"])
        old_bindings = tuple(o for o in old if "binding" in o["provenance"])
        new_bindings = tuple(o for o in new if "binding" in o["provenance"])
        for a,b in zip(old_words, new_words, strict=True):
            if a["lower"] == b["lower"]:
                continue
            start,end = a["provenance"]["span"]
            other_start,other_end = b["provenance"]["span"]
            if not any(x["provenance"]["span"][0] <= start and end <= x["provenance"]["span"][1]
                       and y["provenance"]["span"][0] <= other_start and other_end <= y["provenance"]["span"][1]
                       and mapping[x["lower"]] == y["lower"] for x in old_bindings for y in new_bindings):
                return None
    return pairs


def discover_context_charts(state, contexts, owner, *, reserved_work=0, reserved_candidates=0):
    """Match acquired guarded groups and retain their complete native map witnesses."""
    c = state.configuration
    observations = state.payload["compatibility_observations"]
    attempts = 0
    for o in observations:
        size = len(dict.fromkeys(row for row, _ in o["support"]))
        count = 1
        for i in range(size):
            count *= max(0, len(contexts)-i)
        attempts += count
        if attempts > c["guard_limits"][0]:
            raise ValueError("context guard candidate admission exceeded")
    attempts += reserved_candidates
    if attempts > c["guard_limits"][0]:
        raise ValueError("context guard candidate admission exceeded")
    work = owner.receipt["preparation_steps"] + reserved_work
    if work > c["guard_limits"][2]:
        raise ValueError("context guard work admission exceeded")
    matched = OrderedDict(); cache = {}; calls = 0; pending = []; dictionary_pairs = OrderedDict()
    dictionary_required = any(o["guard"]["profile"] in {_DICTIONARY_PROFILE, _JOINT_DICTIONARY_PROFILE, _DICTIONARY_USE_PROFILE} for o in observations)
    joint_required = any(o["guard"]["profile"] in {_JOINT_DICTIONARY_PROFILE, _DICTIONARY_USE_PROFILE} for o in observations)
    for index, observation in enumerate(observations):
        origins = tuple(dict.fromkeys(row for row, _ in observation["support"]))
        pins = guard_anchor_cells(c, observation["support"], observation["guard"])
        for query_indices in permutations(range(len(contexts)), len(origins)):
            targets = tuple(contexts[q] for q in query_indices)
            origin_groups = tuple(_context_rows(c, row) for row in origins)
            target_groups = tuple(_context_rows(c, row) for row in targets)
            if any(len(a) != len(b) for a, b in zip(origin_groups,target_groups,strict=True)):
                continue
            source_rows = tuple(row for group in origin_groups for row in group)
            target_rows = tuple(row for group in target_groups for row in group)
            key = origins, targets, pins
            if key not in cache:
                result = owner.match(source_rows, target_rows, fixed_lowers=pins, work_limit=c["guard_limits"][2]-work)
                work += result["execution_steps"]; calls += 1
                cache[key] = result
            result = cache[key]
            if not result["matched"]:
                continue
            lower = c["field"]["source_accession"]["lower"]
            if any(lower[a][0] != lower[b][0] for a, b in result["cell_map"]):
                continue
            uses = dict(zip(origins, query_indices, strict=True))
            support = tuple((uses[row], a) for row, a in observation["support"])
            item = dict(observation_index=index, witness=result)
            if observation["guard"]["profile"] in {_DICTIONARY_PROFILE, _JOINT_DICTIONARY_PROFILE, _DICTIONARY_USE_PROFILE}:
                pairs = _dictionary_pairs(c, origins, targets, result,
                    include_unchanged=observation["guard"]["profile"] in {_JOINT_DICTIONARY_PROFILE, _DICTIONARY_USE_PROFILE})
                if pairs is None:
                    continue
                for pair in pairs:
                    dictionary_pairs.setdefault(pair, None)
                pending.append(((observation["type"], support), item, pairs))
            else:
                matched.setdefault((observation["type"], support), []).append(item)
    comparison = None; catalog = None; union_calls = 0; union_candidates = 0; union_cache = {}
    if dictionary_pairs:
        if joint_required:
            catalog = owner.prepare_dictionary(tuple(dictionary_pairs), work_limit=c["guard_limits"][2]-work)
            comparison = catalog.comparison
        else:
            comparison = owner.shared_synsets(tuple(dictionary_pairs), work_limit=c["guard_limits"][2]-work)
        work += comparison["execution_steps"]
        dictionary_pairs.update({(p["origin"],p["target"]):p for p in comparison["pairs"]})
    pair_indices = {pair:i for i,pair in enumerate(dictionary_pairs)}
    for key, item, pairs in pending:
        if all(dictionary_pairs[pair]["licenses"] for pair in pairs):
            if observations[item["observation_index"]]["guard"]["profile"] in {_JOINT_DICTIONARY_PROFILE, _DICTIONARY_USE_PROFILE}:
                if catalog is None:
                    catalog = owner.prepare_dictionary((), work_limit=c["guard_limits"][2]-work)
                    comparison = catalog.comparison; work += comparison["execution_steps"]
                witness = item["witness"]
                indices = tuple(pair_indices[pair] for pair in pairs)
                request = indices, witness["source_rows"], witness["target_rows"], witness["fixed_lowers"]
                if request not in union_cache:
                    candidates = c["guard_limits"][0]-attempts-union_candidates
                    if candidates <= 0:
                        raise ValueError("joint dictionary candidate admission exceeded")
                    if observations[item['observation_index']]['guard']['profile'] == _DICTIONARY_USE_PROFILE:
                        from rexgraph.cpp_dictionary_branches import dictionary_use_union
                        union = dictionary_use_union(c,owner,witness,tuple(dictionary_pairs[pair] for pair in pairs),
                            candidate_limit=candidates,work_limit=c['guard_limits'][2]-work)
                    else:
                        union = catalog.sections(indices, source_rows=witness["source_rows"], target_rows=witness["target_rows"],
                            fixed_lowers=witness["fixed_lowers"], candidate_limit=candidates, work_limit=c["guard_limits"][2]-work)
                    work += union["execution_steps"]; union_candidates += union["domain_count"]; union_calls += 1
                    union_cache[request] = union
                union = union_cache[request]
                if not union["sections"]:
                    continue
                item.update(dictionary_union=union, dictionary_union_catalog_pairs=tuple(dictionary_pairs))
            item["dictionary_comparison"] = dict(profile="retained-common-synset-v1",
                pairs=tuple(dictionary_pairs[pair] for pair in pairs), receipt=owner.receipt,
                interpretation="existential-dictionary-support-not-contextual-sense")
            matched.setdefault(key, []).append(item)
    report = dict(profile=_PROFILE, candidate_count=attempts, native_calls=calls, native_work=work,
        matched_groups=len(matched), limits=c["guard_limits"], receipt=owner.receipt)
    if dictionary_required:
        report.update(dictionary_profile=_DICTIONARY_PROFILE, dictionary_calls=int(comparison is not None),
            dictionary_pairs=len(dictionary_pairs), dictionary_comparison=comparison)
    if joint_required:
        report.update(dictionary_union_profile=_DICTIONARY_USE_PROFILE if 'dictionary_links' in c else _JOINT_DICTIONARY_PROFILE,
            dictionary_union_calls=union_calls,
            dictionary_union_candidates=union_candidates, dictionary_union_requests=tuple(union_cache.values()))
    return matched, report


def clear_context_guard_cache():
    """Release prepared native context owners."""
    with _lock:
        _owners.clear()
