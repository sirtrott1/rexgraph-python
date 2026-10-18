"""Closed declarations of exact accession and reconstruction maps."""
from ._stalk_fields import _scalar, _values


def declaration(stalk_axes, bounds, targets, frames, selected_blocks=()):
    """Validate source coordinates before native reconstruction certification."""
    if isinstance(targets, (str, bytes)) or isinstance(frames, (str, bytes)) or len(frames) != 2:
        raise ValueError("original target rows and complete two-grade view atlas required")
    rows = []; atlas = []; names = {}; count = 0
    for row in targets:
        if type(row) is not int or not 0 <= row < len(stalk_axes[1]) or row in rows or len(rows) >= bounds[0]:
            raise ValueError("distinct admitted original target rows required")
        rows.append(row); count += 1
    for k, family in enumerate(frames):
        if isinstance(family, (str, bytes)) or len(family) != len(stalk_axes[k]):
            raise ValueError("view atlas must cover every original stalk")
        records = []
        for i, frame in enumerate(family):
            count += 5
            if frame is None:
                records.append(None); continue
            name, axes, accession, reconstruction = frame
            if type(name) is not str or not name.strip() or len(name.encode()) > 256 or isinstance(axes, (str, bytes)):
                raise ValueError("bounded common carrier name and complete coordinate keys required")
            keys = []; seen = set()
            for key in axes:
                if type(key) is not str or not key.strip() or len(key.encode()) > 256 or key in seen or len(keys) >= bounds[0]:
                    raise ValueError("distinct bounded common carrier coordinates required")
                seen.add(key); keys.append(key); count += len(key.encode())+1
            keys = tuple(keys)
            if name in names and names[name] != keys:
                raise ValueError("declared common carrier coordinate keys conflict")
            names[name] = keys; count += len(name.encode()); maps = []
            n = len(stalk_axes[k][i])
            for shape, entries in (((len(keys), n), accession), ((n, len(keys)), reconstruction)):
                if isinstance(entries, (str, bytes)):
                    raise TypeError("sparse rational view map entries required")
                support = set(); values = []
                for row, column, value in entries:
                    count += 3
                    if (type(row) is not int or type(column) is not int or not 0 <= row < shape[0] or
                            not 0 <= column < shape[1] or (row, column) in support or count > bounds[1]):
                        raise ValueError("distinct admitted view map coordinates required")
                    support.add((row, column)); values.append((row, column, _values((_scalar(value, bounds[3]),))[0]))
                maps.append(tuple(values))
            if count > bounds[1]:
                raise ValueError("context view declaration exceeds entry admission")
            records.append((name, keys, *maps))
        atlas.append(tuple(records))
    selected = []
    if isinstance(selected_blocks, (str, bytes)):
        raise TypeError("selected view blocks require original row pairs")
    for origin, target in selected_blocks:
        pair = (origin, target); count += 2
        if (type(origin) is not int or type(target) is not int or not 0 <= origin < len(stalk_axes[1]) or
                target not in rows or pair in selected or len(selected) >= bounds[0]):
            raise ValueError("distinct selected original view row pairs required")
        selected.append(pair)
    if count > bounds[1]:
        raise ValueError("context view declaration exceeds entry admission")
    return dict(targets=tuple(rows), frames=tuple(atlas), selected_blocks=tuple(selected))


def diagram_declaration(stalk_axes, bounds, patches, frames, selected_diagrams=()):
    """Validate explicit original patches and complete selected row maps."""
    if isinstance(patches, (str, bytes)) or isinstance(selected_diagrams, (str, bytes)):
        raise TypeError("original diagram patches and selected row maps required")
    groups = []; distinct = set(); targets = []; count = 0
    for patch in patches:
        if isinstance(patch, (str, bytes)):
            raise TypeError("original relation rows required")
        rows = []
        for row in patch:
            count += 1
            if (type(row) is not int or not 0 <= row < len(stalk_axes[1]) or row in rows or
                    len(rows) >= bounds[0] or count > bounds[1]):
                raise ValueError("distinct admitted original diagram rows required")
            rows.append(row)
            if row not in targets:
                targets.append(row)
        signature = tuple(sorted(rows))
        if len(rows) < 2 or signature in distinct or len(groups) >= bounds[0]:
            raise ValueError("distinct original patches with at least two rows required")
        distinct.add(signature); groups.append(tuple(rows))
    declared = declaration(stalk_axes, bounds, targets, frames)
    chosen = []
    for diagram in selected_diagrams:
        if isinstance(diagram, (str, bytes)):
            raise TypeError("complete original diagram row map required")
        pairs = []; origins = set(); images = set()
        for origin, target in diagram:
            count += 2
            if (type(origin) is not int or type(target) is not int or not 0 <= origin < len(stalk_axes[1]) or
                    target not in targets or origin in origins or target in images or count > bounds[1]):
                raise ValueError("injective original diagram row map required")
            origins.add(origin); images.add(target); pairs.append((origin, target))
        signature = tuple(sorted(pairs))
        if tuple(sorted(images)) not in distinct or signature in chosen or len(chosen) >= bounds[0]:
            raise ValueError("distinct selected maps must cover a complete declared patch")
        chosen.append(signature)
    # Account for atlas and patch records together, including repeated target rows.
    atlas_count = sum(5 if frame is None else 5+len(frame[0].encode())+
        sum(len(key.encode())+1 for key in frame[1])+3*(len(frame[2])+len(frame[3]))
        for grade in declared['frames'] for frame in grade)
    if count+atlas_count+len(targets) > bounds[1]:
        raise ValueError("context diagram declaration exceeds entry admission")
    return dict(target_patches=tuple(groups), frames=declared['frames'], selected_diagrams=tuple(chosen))
