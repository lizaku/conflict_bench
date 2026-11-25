"""Paths, config and JSONL helpers shared by the construction steps."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent          # the conflict_bench folder
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path=None, model=None):
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if model:
        m = cfg.get("model")
        cfg["model"] = dict(m, name=model) if isinstance(m, dict) else model
    return cfg


def model_name(cfg) -> str:
    m = cfg["model"]
    return m if isinstance(m, str) else (m.get("name") or m.get("model_name"))


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def bench_root(cfg) -> Path:
    return resolve(cfg.get("bench_dir", "bench"))


def seed_dir(cfg) -> Path:
    """Model-free step output: bench/_seed/."""
    d = bench_root(cfg) / "_seed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_dir(cfg) -> Path:
    """Everything built for one model: bench/<model>/."""
    d = bench_root(cfg) / model_name(cfg).rstrip("/\\").replace("\\", "/").split("/")[-1]
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def spread(recs, limit, seed=0):
    """About `limit` instances, whole facts at a time (so the paired
    instances of a fact stay together), taken round-robin over relations."""
    import random
    from collections import defaultdict
    if not limit or limit >= len(recs):
        return list(recs)
    groups, by_rel = defaultdict(list), defaultdict(list)
    for r in recs:
        groups[r["group"]].append(r)
    for gid in sorted(groups):
        by_rel[groups[gid][0]["relation"]].append(gid)
    rng = random.Random(seed)
    for v in by_rel.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < limit and any(i < len(v) for v in by_rel.values()):
        for rel in sorted(by_rel):
            if i < len(by_rel[rel]) and len(out) < limit:
                out += groups[by_rel[rel][i]]
        i += 1
    return out


def utf8_stdout():
    """Windows consoles default to a legacy code page; entity names are not
    ASCII, and a progress line must never be what kills a GPU job."""
    import sys
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


class Appender:
    """Append rows one at a time, so an interrupted GPU step resumes."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8", newline="\n")

    def add(self, row):
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def load_model(cfg):
    """The model, with the prompt format of the config applied."""
    from conflict_bench.core import prompts
    from conflict_bench.core.model import ModelWrapper
    prompts.configure(**(cfg.get("prompt") or {}))
    return ModelWrapper.from_config(
        cfg, activation_cache_dir=model_dir(cfg) / "cache" / "activations")
