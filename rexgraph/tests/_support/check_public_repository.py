#!/usr/bin/env python3
"""Check tracked public files and, optionally, reachable Git history."""
from pathlib import Path, PurePosixPath
import argparse
import ast
import io
import json
import os
import re
import subprocess
import tokenize

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = {"internal", "private", "handoff", "handoffs", "coordination",
           "verification", "backups", "archives", "notes"}
GENERATED = {".o", ".obj", ".a", ".lib", ".so", ".dylib", ".dll", ".exe",
             ".pyc", ".log", ".html", ".zip", ".patch", ".pdf", ".ipynb",
             ".pch", ".gch", ".pdb", ".ilk", ".tex", ".bib", ".aux", ".bbl",
             ".blg", ".fls", ".fdb_latexmk", ".tgz", ".7z"}
SOURCE_HTML = {"agent/frontend/index.html", "system/system/frontend/index.html"}
ATTRIBUTION = re.compile(r"(?im)^.*(?:co[- ]authored[- ]by|generated[- ]by).*"
                         r"(?:cod[e]x|chat[g]pt|open[a]i).*$")
WORKSPACE_LINK = re.compile(r"(?:/home/[A-Za-z0-9_.-]+/|\.\./(?:\.\./)*(?:internal|private|handoff)/|"
                            r"documentation/[A-Z][A-Z_]+\.(?:txt|json)|reference[/]software/)")
PROCESS_LABEL = re.compile(r"\b(?:C(?:0[1-9]|1[0-6])|E0?[1-3]|B\d{1,2}-(?:REPAIR|EDIT|GLUE|LEARN|EXEC|MOMENT|FORMAT|RECODE)[A-Za-z0-9.-]*|"
                           r"(?:GLUE|MOMENT|EXEC|RECODE|FORMAT|LEARN|REPAIR|EDIT)-\d+[a-z]?|"
                           r"P\d+(?:\.\d+)+|ADR\s+\d+|[Pp]hase[- ]+\d+|[Bb]atch\s+\d+)\b")
STYLE = re.compile(r"\b(?:leverage[sd]?|leveraging|delve[sd]?|delving|game[- ]chang(?:er|ing)|"
                   r"cutting[- ]edge|world[- ]class|best[- ]in[- ]class|revolutionary|groundbreaking|"
                   r"seamless(?:ly)?|supercharge[sd]?|synerg(?:y|ies)|transformative|"
                   r"next[- ]generation|state[- ]of[- ]the[- ]art|robust(?:ly)?|powerful|"
                   r"unlock(?:s|ed|ing)?|empower(?:s|ed|ing)?|foster(?:s|ed|ing)?)\b", re.I)
DIVIDER = re.compile(r"(?m)^\s*(?:(?://|#|\*)\s*)?[-=_*]{5,}\s*$")


def prose(rel, text):
    path = PurePosixPath(rel)
    if path.suffix in {'.md', '.rst'}:
        return re.sub(r'```[^\n]*\n[\s\S]*?```', '', text)
    if path.suffix in {'.cpp', '.hpp', '.h', '.c', '.comp'}:
        tokens = re.finditer(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/', text)
        return '\n'.join(m[0] for m in tokens if m[0].startswith(('//', '/*')))
    if path.suffix in {'.py', '.pyx', '.pxd'}:
        try:
            comments = [t.string for t in tokenize.generate_tokens(io.StringIO(text).readline)
                        if t.type == tokenize.COMMENT]
            docstrings = []
            if path.suffix == '.py':
                for n in ast.walk(ast.parse(text)):
                    if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
                        first = n.body[0]
                        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                            docstrings.append(first.value.value)
            else:
                docstrings = [t.string for t in tokenize.generate_tokens(io.StringIO(text).readline)
                              if t.type == tokenize.STRING and t.string.startswith((chr(34) * 3, chr(39) * 3))]
            return '\n'.join(comments + docstrings)
        except (SyntaxError, tokenize.TokenError, IndentationError):
            return ''
    if path.name == 'CMakeLists.txt' or path.suffix in {'.cmake', '.sh', '.yml', '.yaml', '.toml'}:
        return '\n'.join(line for line in text.splitlines() if line.lstrip().startswith('#'))
    return ''


def content_reasons(rel, data):
    text = data.decode('utf-8', errors='replace')
    errors = []
    if ATTRIBUTION.search(text):
        errors.append('automated co-author or generation attribution')
    if WORKSPACE_LINK.search(text):
        errors.append('workspace-specific path or document reference')
    public_prose = prose(rel, text)
    for pattern, why in [(DIVIDER, 'decorative section divider')]:
        if pattern.search(public_prose):
            errors.append(why)
    for line in public_prose.splitlines():
        labels = PROCESS_LABEL.findall(line)
        technical = (re.search(r"\bC11\b", line) and re.search(r"compiler|LTO|link time|deployment|wheel", line)) or ("C12" in line and re.search(r"carbon|schema|label", line))
        labels = [label for label in labels if label != 'B1.T' and not re.fullmatch(r'B\d+\.[A-Za-z]', label)]
        if labels and not technical:
            errors.append('implementation tracking label in prose')
        for match in STYLE.finditer(line):
            term = match.group().lower()
            if term == 'foster' and match.group().startswith('Foster'):
                continue
            if term == 'leverage' and not re.search(r"\b(?:we|you|can|could|will|to|and)\s+leverage\b", line, re.I):
                continue
            errors.append('promotional or vague wording')
    return errors


def message_reasons(message):
    errors = content_reasons('README.md', message.encode())
    if not re.fullmatch(r'[a-z0-9._/-]+: [a-z][^\n]+', message.splitlines()[0] if message.splitlines() else ''):
        errors.append('commit subject must use scope and lowercase subject')
    if not re.search(r'\nTests: \d+ total\.\s*$', message):
        errors.append('commit body must end with an exact test-count footer')
    if re.search(r'(?im)^(?:co-authored-by|signed-off-by):', message):
        errors.append('commit trailer is outside the project convention')
    if re.search(r'\b(?:[Rr]elease [ABC]|[CE] workflows|private Python tests)\b', message):
        errors.append('development record reference in commit message')
    return errors


def history_commits(base=None, event=None, event_name=None):
    rewritten_push = False
    if event is not None:
        kind = event_name if event_name is not None else os.environ.get('GITHUB_EVENT_NAME', '')
        if kind in {'schedule', 'workflow_dispatch'}:
            return [subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()]
        candidate = event.get('pull_request', {}).get('base', {}).get('sha') or event.get('before', '')
        if re.fullmatch(r'[0-9a-f]{40}', candidate) and candidate != '0' * 40:
            base = candidate
            rewritten_push = kind == 'push' and event.get('forced') is True
            if 'pull_request' in event:
                base = subprocess.check_output(['git', '-C', str(ROOT), 'merge-base', base, 'HEAD'], text=True).strip()
        else:
            default = event.get('repository', {}).get('default_branch')
            ref = event.get('ref', '')
            if default and ref.startswith(('refs/heads/', 'refs/tags/')) and ref != 'refs/heads/' + default:
                base = subprocess.check_output(['git', '-C', str(ROOT), 'merge-base',
                                                'refs/remotes/origin/' + default, 'HEAD'], text=True).strip()
            else:
                return subprocess.check_output(['git', '-C', str(ROOT), 'rev-list', '--reverse', 'HEAD'], text=True).splitlines()
    if base is not None:
        resolved = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--verify', '--end-of-options',
                                            base + '^{commit}'], text=True).strip()
        ancestor = subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', resolved, 'HEAD'])
        if ancestor.returncode and not rewritten_push:
            raise ValueError('History base must be an ancestor of HEAD')
        return subprocess.check_output(['git', '-C', str(ROOT), 'rev-list', '--reverse', resolved + '..HEAD'], text=True).splitlines()
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-list', '--all'], text=True).splitlines()


def history_errors(commits):
    seen = set()
    with subprocess.Popen(['git', '-C', str(ROOT), 'cat-file', '--batch'],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as reader:
        for version in commits:
            entries = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-rz', version])
            for entry in entries.split(b'\0'):
                if not entry:
                    continue
                metadata, rel = entry.split(b'\t', 1)
                rel = rel.decode()
                if why := reason(rel):
                    yield f'{version}: {rel}: {why}'
                blob = metadata.split()[2]
                key = (rel, blob)
                if key in seen:
                    continue
                seen.add(key)
                reader.stdin.write(blob + b'\n')
                reader.stdin.flush()
                header = reader.stdout.readline().split()
                if len(header) != 3 or header[1] != b'blob':
                    raise ValueError('Expected a Git blob while checking history')
                data = reader.stdout.read(int(header[2]))
                if reader.stdout.read(1) != b'\n':
                    raise ValueError('Incomplete Git blob response')
                for why in content_reasons(rel, data):
                    yield f'{version}: {rel}: {why}'
            metadata, message = subprocess.check_output(['git', '-C', str(ROOT), 'show', '-s',
                '--format=%an%n%ae%n%cn%n%ce%x00%B', version]).decode().split('\0', 1)
            identities = metadata.splitlines()
            names = '\n'.join(identities[i] for i in (0, 2))
            if re.search(r'(?im)^(?:cod[e]x|chat[g]pt|open[a]i)(?:$|[ -](?:agent|bot)\b)', names):
                yield f'{version}: automated attribution in commit metadata'
            for why in message_reasons(message):
                yield f'{version}: {why}'
        reader.stdin.close()
        if reader.wait():
            raise ValueError('Git history reader failed')


def paths(*args):
    data = subprocess.check_output(["git", "-C", str(ROOT), *args])
    return [p for p in data.decode().split("\0") if p]


def reason(rel):
    path = PurePosixPath(rel)
    if path.name in {"AGENTS.md", "CONTEXT.md", "NEXT_STEPS.md", "HANDOFF.md"}:
        return "private development document"
    if any(part in PRIVATE for part in path.parts[:-1]):
        return "private workspace material"
    if any(p.startswith("build") or p in {"CMakeFiles", "__pycache__", ".venv", "Testing", "dist"} for p in path.parts[:-1]):
        return "generated build or tool directory"
    if ((path.suffix in GENERATED and rel not in SOURCE_HTML) or ".so." in path.name
            or path.name.endswith((".tar.gz", ".tar.xz", ".executed.ipynb", ".synctex.gz"))):
        return "generated output or intake archive"
    if path.suffix.startswith(".rfcx") and not rel.startswith(("data/fixtures/", "tests/fixtures/", "tests/storage/fixtures/", "bindings/cpp/tests/fixtures/")):
        return "native output outside the frozen fixture directories"
    if path.name in {"CMakeCache.txt", "compile_commands.json", "install_manifest.txt",
                     "AGENTS.md", "CHECKSUMS.sha256", "FILE_MANIFEST.txt"}:
        return "local build or intake manifest"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="also check every locally reachable commit")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--base', help='with --history, check only commits added after this ancestor')
    scope.add_argument('--event', type=Path,
                       help='with --history, check commits introduced by a GitHub event; scheduled runs check HEAD')
    parser.add_argument("--working-tree", action="store_true",
                        help="also check untracked files admitted by Git ignores")
    args = parser.parse_args()
    if (args.base or args.event) and not args.history:
        parser.error('--base and --event require --history')
    inventories = [("index", paths("ls-files", "-z"))]
    if args.working_tree:
        inventories.append(("working tree", paths("ls-files", "--others", "--exclude-standard", "-z")))
    errors = []
    for version, files in inventories:
        errors.extend(f"{version}: {rel}: {why}" for rel in files if (why := reason(rel)))
        for rel in files:
            path = ROOT / rel
            if path.is_file():
                errors.extend(f'{rel}: {why}' for why in content_reasons(rel, path.read_bytes()))
    commits = []
    if args.history:
        event = json.loads(args.event.read_text()) if args.event else None
        commits = history_commits(base=args.base, event=event)
        errors.extend(history_errors(commits))
    if errors:
        print("\n".join(errors))
        return 1
    historical = len(commits)
    untracked = len(inventories[1][1]) if args.working_tree else 0
    print(f"Public repository check passed: {len(inventories[0][1])} tracked files, "
          f"{untracked} untracked files, {historical} historical commits checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
