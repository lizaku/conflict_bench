#!/usr/bin/env python
"""Pilot driver: load the model and data, run every configured method, write
one final table.

    python -m conflict_bench.experiments.run_pilot
    python -m conflict_bench.experiments.run_pilot --limit 200
    python -m conflict_bench.experiments.run_pilot --model google/gemma-2-2b-it

Steps, in order:
    1. model      ModelWrapper from the config's `model:` block
    2. data       load, drop first-token collisions, sample `limit` items
                  evenly over relations, label each by the model's generation
    3. conditions N/S/C/R margins for every item (fills the shared margin table)
    4. detection  every detector, GroupKFold by relation
    5. steering   every steerer, every target x factor
    6. table      one row per method -> final_table.csv

What is run is decided by the config (configs/pilot_five.yaml), not here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from conflict_bench import data as data_mod
from conflict_bench import metrics, run as runner
from conflict_bench.core import prompts
from conflict_bench.core.artifacts import ArtifactStore
from conflict_bench.core.model import ModelWrapper

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "pilot_five.yaml"


def load_data(cfg, model, out):
    """Items and their behavioural labels (1 = the model followed the context)."""
    loader = getattr(data_mod, cfg["dataset"]["loader"])
    items, _ = loader(**cfg["dataset"].get("kwargs", {}))
    if cfg.get("qc", {}).get("drop_first_token_collisions"):
        data_mod.first_token_collisions(items, model.tok)
        items = [it for it in items if not it.meta.get("first_token_collision")]
    items = runner.stratified_limit(items, cfg.get("limit"), cfg.get("seed", 0))

    labels, rows = runner.behavioural_labels(model, items)
    pd.DataFrame(rows).to_csv(out / "labels.csv", index=False, encoding="utf-8")
    return items, labels


def run_steering(cfg, model, items, labels, groups, out, artifacts):
    """Steering records -> per (method, target, factor) summary."""
    records = runner.run_steering(cfg, model, items, labels, groups, out, artifacts)
    with open(out / "steering_records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.__dict__, default=str, ensure_ascii=False) + "\n")
    summary = metrics.steering_summary(records)
    summary.to_csv(out / "steering_summary.csv", index=False)
    return summary


def final_table(det_summ, steer_summary, out):
    """One row per method: detectors by held-out AUROC, steerers at their best
    factor. See metrics.simple_report for the columns."""
    table = metrics.simple_report(det_summ, steer_summary)
    table.to_csv(out / "final_table.csv", index=False)
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--limit", type=int, default=None, help="override `limit`")
    ap.add_argument("--model", default=None, help="override `model.name`")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.model:
        cfg["model"]["name"] = args.model

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactStore(out / "artifacts")
    prompts.configure(**(cfg.get("prompt") or {}))

    # 1. model
    model = ModelWrapper.from_config(cfg, activation_cache_dir=out / "activations",
                                     margin_table_path=out / "margin_table.jsonl")
    print("model:", model.describe())

    # 2. data
    items, labels = load_data(cfg, model, out)
    groups = [it.relation for it in items]
    manifest = {"config": cfg, "model": model.describe(),
                "prompt": prompts.DEFAULT.as_dict(), "n_items": len(items),
                "relations": sorted(set(groups)),
                "context_following_rate": float(np.mean(labels))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                       encoding="utf-8")
    print(f"data: {len(items)} items over {len(set(groups))} relations, "
          f"context-following rate {np.mean(labels):.3f}")

    # 3-5. methods
    runner.run_conditions(cfg, model, items, labels, out)
    det_summ = runner.run_detection(cfg, model, items, labels, groups, out, artifacts)
    steer_summary = run_steering(cfg, model, items, labels, groups, out, artifacts)
    model.acts.flush()
    model.margins.flush()

    # 6. table
    table = final_table(det_summ, steer_summary, out)
    print()
    print(table.to_string(index=False))
    print(f"\nwrote {out / 'final_table.csv'}")


if __name__ == "__main__":
    main()
