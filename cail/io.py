import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: {exc.msg}") from exc


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def digest(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unique_by_id(rows):
    result = {}
    for row in rows:
        key = str(row["id"])
        if key in result:
            raise ValueError(f"重复 id: {key}")
        result[key] = row
    return result


def load_config(path):
    path = Path(path).resolve()
    def read(current):
        child = json.loads(current.read_text(encoding="utf-8-sig"))
        parent = child.pop("extends", None)
        if parent is None:
            return child
        base = read(current.parent / parent)
        for key, value in child.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key].update(value)
            else:
                base[key] = value
        return base
    config = read(path)
    root = path.parent.parent
    for key, value in config.get("paths", {}).items():
        if not PurePosixPath(value).is_absolute() and not PureWindowsPath(value).is_absolute():
            config["paths"][key] = str((root / value).resolve())
    return config
