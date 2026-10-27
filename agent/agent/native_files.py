"""Load installed Core formats, including ZIP transport of an RCBD directory."""
from pathlib import Path, PurePosixPath


def load_native_file(path, *, format=None):
    from rexgraph.io import load
    path = Path(path)
    selected = format.lower() if isinstance(format, str) else format
    if selected != "rcbd" or path.is_dir():
        return load(path, format=format)
    import shutil
    import stat
    import tempfile
    import zipfile
    from .server.handles import DEFAULT_MAX_FILE
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory(prefix="rexgraph_rcbd_") as temporary:
        members = archive.infolist()
        if sum(member.file_size for member in members) > DEFAULT_MAX_FILE:
            raise ValueError("unpacked RCBD bundle exceeds the file byte limit")
        names = set()
        for member in members:
            name = PurePosixPath(member.filename)
            if (not name.parts or name.is_absolute() or ".." in name.parts
                    or "\\" in member.filename or ":" in name.parts[0]):
                raise ValueError("RCBD archive member must stay inside its bundle")
            mode = member.external_attr >> 16
            if stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ValueError("RCBD archive members must be regular files or directories")
            if str(name) in names:
                raise ValueError("duplicate RCBD archive member")
            names.add(str(name))
        if "MANIFEST.json" in names:
            prefix = None
        else:
            candidates = [name for name in names if name.endswith("/MANIFEST.json")]
            if len(candidates) != 1 or len(PurePosixPath(candidates[0]).parts) != 2:
                raise ValueError("RCBD archive requires one bundle manifest")
            prefix = PurePosixPath(candidates[0]).parts[0]
            if any(PurePosixPath(name).parts[0] != prefix for name in names):
                raise ValueError("RCBD archive contains files outside its bundle")
        root = Path(temporary) / "source.rcbd"
        root.mkdir()
        for member in members:
            parts = PurePosixPath(member.filename).parts
            parts = parts[1:] if prefix is not None else parts
            target = root.joinpath(*parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=64 * 1024)
        return load(root, format="rcbd")
