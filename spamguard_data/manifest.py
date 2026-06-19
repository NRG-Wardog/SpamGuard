from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MANIFEST_PATH = Path("data/dataset_manifest.json")


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"version": 1, "sources": []}


def save_manifest(manifest: dict[str, Any], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def upsert_source(manifest: dict[str, Any], source: dict[str, Any]) -> None:
    sources = manifest.setdefault("sources", [])
    for idx, existing in enumerate(sources):
        if existing.get("source_name") == source.get("source_name"):
            merged = {**existing, **source}
            sources[idx] = merged
            return
    sources.append(source)
