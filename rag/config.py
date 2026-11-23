"""Load config.yaml and resolve relative paths against the project root."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    with open(path or ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    for key, value in (overrides or {}).items():      # e.g. {"chunking.strategy": "fixed"}
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    for key, value in cfg["paths"].items():
        p = Path(value).expanduser()
        cfg["paths"][key] = str(p if p.is_absolute() else ROOT / p)
    return cfg
