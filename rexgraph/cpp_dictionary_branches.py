"""Declared annotation links and original dictionary routes on context uses."""
from collections import OrderedDict
from itertools import product
from threading import RLock

_catalog_lock = RLock()
_catalogs = OrderedDict()


def clear_dictionary_branch_cache():
    """Release prepared owning dictionary catalogs."""
    with _catalog_lock:
        _catalogs.clear()


def validate_dictionary_links(c):
    """Check complete named layer links to actual source synset identities."""
    links = c["dictionary_links"]
    lower = c["field"]["source_accession"]["lower"]
    names = c["layers"]
    if not isinstance(links, tuple) or not links or any(len(link) != 3 for link in links):
        raise ValueError("dictionary links require named layer, value and source synset")
    if len(set(links)) != len(links):
        raise ValueError("dictionary links must be distinct")
    for name, value, synset in links:
        if (name not in names or value not in {v[names.index(name)] for v in c["joint_values"]}
                or type(synset) is not int or not 0 <= synset < len(lower) or lower[synset][0] != "synset"):
            raise ValueError("dictionary link outside declared annotation or source synset")
    for name in dict.fromkeys(link[0] for link in links):
        declared = {v[names.index(name)] for v in c["joint_values"]}
        if declared != {value for n, value, _ in links if n == name}:
            raise ValueError("dictionary links must cover every value of each linked layer")


def branch_accepts(c, values, branch):
    """Require all linked annotation coordinates to admit the chosen synset."""
    return all(any(n == name and value == values[c["layers"].index(name)] and synset == branch[2]
                   for n, value, synset in c["dictionary_links"])
               for name in dict.fromkeys(link[0] for link in c["dictionary_links"]))


def context_dictionary_branches(c, rows, owner):
    """Certify complete original routes for dictionary bindings covering each target."""
    from rexgraph.cpp_context_guards import _occurrences
    bindings = {}; pairs = OrderedDict()
    for row in dict.fromkeys(rows):
        occurrences, chart = _occurrences(c, row)
        start, end = chart["target"]
        entries = []
        for occurrence in occurrences:
            provenance = occurrence["provenance"]
            if "binding" not in provenance:
                continue
            a, b = provenance["span"]
            if a <= start and end <= b:
                record = occurrence["lower"]
                entries.append((provenance["binding"], record, (a, b)))
                pairs.setdefault((record, record), None)
        if not entries:
            raise ValueError("dictionary-bound target has no complete retained dictionary binding")
        bindings[row] = tuple(entries)
    work = owner.receipt["preparation_steps"]
    if work >= c["guard_limits"][2]:
        raise ValueError("dictionary branch work admission exceeded")
    key = owner, tuple(pairs)
    with _catalog_lock:
        catalog = _catalogs.get(key)
        if catalog is None:
            catalog = owner.prepare_dictionary(tuple(pairs), work_limit=c["guard_limits"][2]-work)
            _catalogs[key] = catalog
            while len(_catalogs) > 4:
                _catalogs.popitem(last=False)
        _catalogs.move_to_end(key)
    comparison = catalog.comparison
    work += comparison["execution_steps"]
    licenses = {p["origin"]: p["licenses"] for p in comparison["pairs"]}
    count = sum(len(values) for values in licenses.values())
    if count > c["guard_limits"][0]:
        raise ValueError("dictionary branch candidate admission exceeded")
    branches = {}
    for row, entries in bindings.items():
        choices = OrderedDict()
        for binding, record, span in entries:
            for route in licenses[record]:
                identity = (binding, record, route["synset"], tuple(route["target_rows"]))
                choices.setdefault(identity, dict(identity=identity, span=span))
        if not choices:
            raise ValueError("dictionary-bound target has no certified original route")
        branches[row] = tuple(choices.values())
    return branches, dict(profile="target-dictionary-branches-v1", comparison=comparison,
        native_work=work, candidate_count=count, receipt=owner.receipt,
        contexts=tuple(dict(source_row=row, branches=values) for row, values in branches.items()))


def validate_dictionary_observations(state, branches):
    """Check acquired alternatives against the actual current source routes."""
    c = state.configuration
    for observation in state.payload["dictionary_observations"]:
        if set(observation) != {"source_row", "joint_values", "branches", "batch"}:
            raise ValueError("closed acquired dictionary observation required")
        row, values = observation["source_row"], observation["joint_values"]
        if (type(row) is not int or row not in c["field"]["training_rows"]
                or values not in c["joint_values"] or observation["batch"] not in state.payload["batches"]):
            raise ValueError("dictionary observation outside acquired training annotations")
        label = c["field"]["labels"][c["joint_values"].index(values)]
        if row not in dict(state.payload["fields"][label]):
            raise ValueError("dictionary observation has no acquired annotation field")
        native = tuple(b["identity"] for b in branches[row] if branch_accepts(c, values, b["identity"]))
        retained = observation["branches"]
        if not retained or tuple(retained) != native:
            raise ValueError("acquired dictionary branch differs from its annotation or native source")
    counts = {}
    for o in state.payload["dictionary_observations"]:
        label = c["field"]["labels"][c["joint_values"].index(o["joint_values"])]
        counts[label, o["source_row"]] = counts.get((label, o["source_row"]), 0) + 1
    expected = {(name, row): count for name, entries in state.payload["fields"].items() for row, count in entries}
    if counts != expected:
        raise ValueError("dictionary observations must cover every acquired field count")


def dictionary_use_union(c, owner, seed, pairs, *, candidate_limit, work_limit):
    """Certify one retained union with an independent route choice per original binding."""
    from rexgraph.cpp_context_guards import _occurrences
    if candidate_limit <= 0 or work_limit <= 0:
        raise ValueError('positive dictionary use union candidate and work admissions required')
    lookup = {(p['origin'],p['target']): p for p in pairs}
    scopes = []
    for origin, target in zip(seed['source_rows'],seed['target_rows'],strict=True):
        if c['field']['source_accession']['relations'][origin]['kind'] != 'observed-context':
            continue
        before,_ = _occurrences(c,origin);after,_ = _occurrences(c,target)
        for a,b in zip(before,after,strict=True):
            if 'binding' not in a['provenance']:
                continue
            pair = lookup[a['lower'],b['lower']]
            scopes.append(dict(pair,origin_context=origin,target_context=target,
                origin_binding=a['provenance']['binding'],target_binding=b['provenance']['binding']))
    size = 1
    for scope in scopes:
        size *= len(scope['licenses'])
    if size > candidate_limit:
        raise ValueError('dictionary use union candidate admission exceeded')
    if len(scopes) + sum(len(p['licenses']) for p in scopes) > c['guard_limits'][1]:
        raise ValueError('dictionary use union entry admission exceeded')
    work = 0; native_calls = 0; rejected = 0; sections = []
    for choices in product(*(range(len(p['licenses'])) for p in scopes)):
        cells = dict(seed['cell_map']); inverse_cells = {b:a for a,b in cells.items()}
        rows = dict(zip(seed['source_rows'],seed['target_rows'],strict=True))
        inverse_rows = {b:a for a,b in rows.items()}
        pins = dict.fromkeys(seed['fixed_lowers']); compatible = True
        for scope, choice in zip(scopes,choices,strict=True):
            license = scope['licenses'][choice]
            pins[license['synset']] = None
            for mapping,inverse,entries in ((cells,inverse_cells,license['cell_map']),
                (rows,inverse_rows,license['relation_map'])):
                for a,b in entries:
                    work += 1
                    if work >= work_limit:
                        raise ValueError('dictionary use union work admission exceeded')
                    if a in mapping and mapping[a] != b or b in inverse and inverse[b] != a:
                        compatible = False;break
                    mapping[a]=b;inverse[b]=a
                if not compatible:break
            if not compatible:break
        if not compatible:
            rejected += 1;continue
        source_rows = tuple(sorted(rows));target_rows = tuple(rows[row] for row in source_rows)
        if len(source_rows)+len(cells)+len(pins) > c['guard_limits'][1]:
            raise ValueError('dictionary use union entry admission exceeded')
        result = owner.match(source_rows,target_rows,fixed_lowers=tuple(pins),work_limit=work_limit-work)
        work += result['execution_steps'];native_calls += 1
        if result['matched']:
            sections.append(dict(choices=choices,source_rows=source_rows,target_rows=target_rows,fixed_lowers=tuple(pins),
                **{k:result[k] for k in ('cell_map','relation_map','occurrence_map','execution_steps',
                    'strict_morphism','chain_map','ordered')}))
        else:
            raise RuntimeError('compatible dictionary use maps fail the complete native union check')
    return dict(profile='retained-dictionary-use-union-maps-v1',source_rows=seed['source_rows'],target_rows=seed['target_rows'],
        fixed_lowers=seed['fixed_lowers'],scopes=tuple(scopes),candidate_limit=candidate_limit,domain_count=size,
        candidate_count=size,native_calls=native_calls,overlap_rejections=rejected,execution_steps=work,
        complete=True,receipt=owner.receipt,sections=tuple(sections))


def guarded_branch_tuples(state, witness, support, contexts, branches, *, candidate_limit):
    """Restrict native union choices to acquired source and actual query target routes."""
    observation = state.payload["compatibility_observations"][witness["observation_index"]]
    uses = tuple(dict.fromkeys(q for q, _ in support))
    row_map = dict(witness["witness"]["relation_map"])
    origin_for = {contexts.index(row_map[row]): row for row, _ in observation["support"]}
    acquired = {}
    for q in uses:
        origin = origin_for[q]
        values = {a: v for (row, a), v in zip(observation["support"], observation["values"], strict=True) if row == origin}
        acquired[q] = tuple(b for o in state.payload["dictionary_observations"]
            if o["source_row"] == origin and o["batch"] == observation["batch"]
            and all(o["joint_values"][a] == v for a, v in values.items()) for b in o["branches"])
    allowed = OrderedDict(); candidates = 0
    pairs = witness["dictionary_comparison"]["pairs"]
    if witness['dictionary_union']['profile'] == 'retained-dictionary-use-union-maps-v1':
        pairs = witness['dictionary_union']['scopes']
    for section_index, section in enumerate(witness["dictionary_union"]["sections"]):
        selected = tuple(p["licenses"][choice] for p, choice in zip(pairs, section["choices"], strict=True))
        alternatives = []
        for q in uses:
            compatible = []
            for record in branches[contexts[q]]:
                target = record["identity"]
                if any(p["target_context"] == contexts[q] and p['origin_context'] == origin_for[q]
                        and p['target_binding'] == target[0] and p["target"] == target[1] and route["synset"] == target[2]
                        and tuple(route["target_rows"]) == target[3]
                        and any(origin[0] == p['origin_binding'] and origin[1] == p["origin"] and origin[2] == route["synset"]
                                and origin[3] == tuple(route["source_rows"]) for origin in acquired[q])
                        for p, route in zip(pairs, selected, strict=True)):
                    compatible.append(target)
            alternatives.append(tuple(compatible))
        size = 1
        for values in alternatives:
            size *= len(values)
        candidates += size
        if candidates > candidate_limit:
            raise ValueError("dictionary branch rule candidate admission exceeded")
        for choices in product(*alternatives):
            allowed.setdefault(tuple(choices), []).append(section_index)
    return uses, tuple(allowed), candidates, tuple(tuple(indices) for indices in allowed.values())
