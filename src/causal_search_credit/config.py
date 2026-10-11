from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def environments(config: dict):
    from .retrieval import RetrievalEnvironment
    return {
        item["id"]: RetrievalEnvironment(
            item["id"],
            float(item["temperature"]),
            item.get("backend", "bm25"),
        )
        for item in config["retrieval"]["environments"]
    }
