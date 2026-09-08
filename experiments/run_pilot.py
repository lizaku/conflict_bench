#!/usr/bin/env python
"""Standalone pilot driver: 5 steering methods + a BOW detection baseline.

Calls the conflict_bench package but owns its own control flow, so it can be
edited and re-run without touching the library:

    python -m conflict_bench.experiments.run_pilot --limit 200
    python -m conflict_bench.experiments.run_pilot --stages detection --force
    python -m conflict_bench.experiments.run_pilot --model google/gemma-2-2b-it

Runs in five checkpointed stages - data, conditions, detection, steering,
report.  Each writes its output and is skipped on a re-run if that output is
already there, so an interrupted pilot resumes instead of re-spending the GPU
hours, and a single stage can be re-run in isolation with --force.

The methods, and the detection twin each one is paired with in the report:

    prompting  prompt_instruct / p_true
    CAD        cad             / context_kl
    AdaCAD     adacad          / context_jsd
    CK-PLUG    ckplug          / confidence_gain
    CAA        caa             / diffmean_proj
    BOW        (detection only) - the text-only null
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from conflict_bench import data as data_mod
from conflict_bench import metrics, run as runner
from conflict_bench.core import margins as margins_mod, prompts
from conflict_bench.core.artifacts import ArtifactStore
from conflict_bench.core.model import ModelWrapper
from conflict_bench.core.types import Condition

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "pilot_five.yaml"

#: report label -> (detection method, steering method)
METHOD_PAIRS = [
    ("prompting", "p_true", "prompt_instruct"),
    ("CAD", "context_kl", "cad"),
    ("AdaCAD", "context_jsd", "adacad"),
    ("CK-PLUG", "confidence_gain", "ckplug"),
    ("CAA", "diffmean_proj", "caa"),
    ("CAA (orthogonalized)", "diffmean_proj", "act_add"),
    ("BOW baseline", "bow", None),
    ("margin (DV floor)", "margin", None),
    ("logit lens (readout control)", "logit_lens", None),
]

STAGES = ("data", "conditions", "detection", "steering", "report")


# --------------------------------------------------------------------- utils
def log(stage, msg):
    print(f"[{stage:<10}] {msg}", flush=True)


def primary_row(det_summ, method):
    """The primary (position, condition) cell for a base method name."""
    rows = [r for r in det_summ.values() if r.get("method") == method]
    if not rows:
        return {}
    for r in rows:
        if r.get("primary"):
            return r
    return rows[0]


# --------------------------------------------------------------------- stages
def stage_data(cfg, model, out):
    """Load, QC, subsample, and label. Checkpoint: labels.csv + items.json."""
    items_path, labels_path = out / "pilot_items.json", out / "labels.csv"
    loader = getattr(data_mod, cfg["dataset"]["loader"])
    loaded = loader(**cfg["dataset"].get("kwargs", {}))
    items, dropped = loaded if isinstance(loaded, tuple) else (loaded, [])

    if items_path.exists() and labels_path.exists():
        keep = set(json.loads(items_path.read_text(encoding="utf-8")))
        items = [it for it in items if it.item_id in keep]
        lab = pd.read_csv(labels_path).set_index("item_id")
        items = [it for it in items if it.item_id in lab.index]
        labels = [int(lab.loc[it.item_id, "label"]) for it in items]
        log("data", f"resumed {len(items)} labelled items from {labels_path.name}")
        return items, labels, dropped

    if cfg.get("qc", {}).get("drop_first_token_collisions"):
        collisions = data_mod.first_token_collisions(items, model.tok)
        items = [it for it in items if not it.meta.get("first_token_collision")]
        dropped += [{"item_id": i, "reasons": ["first_token_collision"]}
                    for i in collisions]
        log("data", f"dropped {len(collisions)} first-token-collision items")

    items = runner.stratified_limit(items, cfg.get("limit"), cfg.get("seed", 0))
    log("data", f"{len(items)} items over "
                f"{len({it.relation for it in items})} relations")

    t0 = time.time()
    labels, rows = runner.behavioural_labels(model, items)
    pd.DataFrame(rows).to_csv(labels_path, index=False, encoding="utf-8")
    items_path.write_text(json.dumps([it.item_id for it in items]),
                          encoding="utf-8")
    (out / "qc_dropped.json").write_text(json.dumps(dropped, indent=2),
                                         encoding="utf-8")
    log("data", f"context-following rate = {np.mean(labels):.3f} "
                f"({int(np.sum(labels))}/{len(labels)}) in "
                f"{time.time() - t0:.0f}s")
    if len(set(labels)) < 2:
        log("data", "WARNING: labels are single-class - every trainable method "
                    "(bow, probes, caa/act_add vectors) will refuse to fit")
    return items, labels, dropped


def stage_conditions(cfg, model, items, labels, out, force):
    path = out / "conditions_summary.csv"
    if path.exists() and not force:
        log("conditions", f"skip (have {path.name})")
        return pd.read_csv(path)
    return runner.run_conditions(cfg, model, items, labels, out)


def stage_detection(cfg, model, items, labels, groups, out, artifacts, force):
    path = out / "detection_summary.csv"
    if path.exists() and not force:
        log("detection", f"skip (have {path.name})")
        df = pd.read_csv(path, index_col=0)
        return {k: v for k, v in df.to_dict("index").items()}
    return runner.run_detection(cfg, model, items, labels, groups, out, artifacts)


def stage_steering(cfg, model, items, labels, groups, out, artifacts, force):
    path = out / "steering_summary.csv"
    if path.exists() and not force:
        log("steering", f"skip (have {path.name})")
        return pd.read_csv(path)
    records = runner.run_steering(cfg, model, items, labels, groups, out,
                                  artifacts)
    if not records:
        return pd.DataFrame()
    with open(out / "steering_records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.__dict__, default=str,
                               ensure_ascii=False) + "\n")
    summ = metrics.steering_summary(records)
    summ.to_csv(path, index=False)
    return summ


# --------------------------------------------------------------------- report
def build_report(det_summ, steer_df, cfg, out, manifest):
    """The pilot's headline table: one row per method, both axes."""
    rows = []
    for label, det_name, steer_name in METHOD_PAIRS:
        d = primary_row(det_summ, det_name) if det_name else {}
        s = (steer_df[steer_df.method == steer_name]
             if steer_name and len(steer_df) else pd.DataFrame())
        best = (s.loc[s.flip_rate_flippable.idxmax()]
                if len(s) and s.flip_rate_flippable.notna().any() else None)
        rows.append({
            "method": label,
            "detector": det_name,
            "auroc": _f(d.get("auroc")),
            "auroc_base_rate": _f(d.get("base_rate_auroc")),
            "auroc_over_readout": _f(d.get("auroc_over_readout")),
            "det_position": d.get("position"),
            "steerer": steer_name or "",
            "best_factor": _f(best["factor"]) if best is not None else None,
            "flip_rate": _f(best["flip_rate"]) if best is not None else None,
            "flip_rate_flippable": (_f(best["flip_rate_flippable"])
                                    if best is not None else None),
            "flip_rate_raw": (_f(best.get("flip_rate_raw"))
                              if best is not None else None),
            "d_margin": _f(best["d_margin"]) if best is not None else None,
            "specific_effect": (_f(best["specific_effect"])
                                if best is not None else None),
            "fluency": _f(best.get("fluency")) if best is not None else None,
        })
    report = pd.DataFrame(rows)
    report.to_csv(out / "pilot_report.csv", index=False)

    lines = ["# Pilot report", "",
             f"- model: `{manifest['model']['model']}` "
             f"({manifest['model']['n_layers']} layers, "
             f"{manifest['model']['dtype']})",
             f"- items: {manifest['n_items']} over "
             f"{len(manifest['relations'])} relations",
             f"- context-following rate: {manifest['context_following_rate']:.3f}",
             f"- detection position (primary): "
             f"{cfg.get('detection_positions', ['-'])[0]}", "",
             "## Headline", "", _as_table(report), "",
             "## Reading it", "",
             "- `auroc` beats `auroc_base_rate` or the detector is reading the",
             "  relation, not the item.",
             "- `auroc` beats the **BOW baseline** row or it is reading the text,",
             "  not the model.",
             "- `auroc_over_readout` > 0 or the probe is decoding an answer the",
             "  model had already committed to.",
             "- `specific_effect` is delta-margin minus the matched control;",
             "  `flip_rate_flippable` conditions on items that could move.",
             "- `flip_rate` vs `flip_rate_raw`: a large gap means the flips are",
             "  an artifact of the passage template, not the intervention.", ""]
    (out / "pilot_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def _as_table(df):
    """Markdown table, or a fixed-width one when `tabulate` is absent."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def _f(v):
    try:
        return None if v is None or (isinstance(v, float) and np.isnan(v)) \
            else round(float(v), 4)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=None, help="override out_dir")
    ap.add_argument("--model", default=None, help="override the model name")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--relations", nargs="*", default=None,
                    help="restrict to these relation files, e.g. P17 P27")
    ap.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    ap.add_argument("--force", action="store_true",
                    help="re-run the requested stages even if outputs exist")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.model:
        cfg["model"] = dict(cfg.get("model") or {},
                            **{"name": args.model}) if isinstance(
            cfg.get("model"), dict) else args.model
    for key, val in (("device", args.device), ("dtype", args.dtype)):
        if val and isinstance(cfg.get("model"), dict):
            cfg["model"][key] = val
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.relations:
        cfg["dataset"].setdefault("kwargs", {})["relations"] = args.relations
    if args.out:
        cfg["out_dir"] = args.out

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False),
                                          encoding="utf-8")
    artifacts = ArtifactStore(out / "artifacts")
    prompts.configure(**(cfg.get("prompt") or {}))

    log("setup", f"out_dir = {out.resolve()}")
    model = ModelWrapper.from_config(
        cfg, activation_cache_dir=out / "activations",
        margin_table_path=out / "margin_table.jsonl")
    log("setup", f"model = {model.describe()}")

    manifest = {"config": cfg, "model": model.describe(),
                "prompt": prompts.DEFAULT.as_dict(),
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    items, labels, dropped = stage_data(cfg, model, out)
    groups = [it.relation for it in items]
    manifest.update(n_items=len(items), n_dropped=len(dropped),
                    relations=sorted(set(groups)),
                    context_following_rate=float(np.mean(labels)))

    if "conditions" in args.stages:
        stage_conditions(cfg, model, items, labels, out, args.force)

    det_summ = {}
    if "detection" in args.stages:
        det_summ = stage_detection(cfg, model, items, labels, groups, out,
                                   artifacts, args.force)

    steer_df = pd.DataFrame()
    if "steering" in args.stages:
        steer_df = stage_steering(cfg, model, items, labels, groups, out,
                                  artifacts, args.force)

    if "report" in args.stages:
        if not det_summ and (out / "detection_summary.csv").exists():
            det_summ = pd.read_csv(out / "detection_summary.csv",
                                   index_col=0).to_dict("index")
        if not len(steer_df) and (out / "steering_summary.csv").exists():
            steer_df = pd.read_csv(out / "steering_summary.csv")
        metrics.triplet_report(det_summ, steer_df if len(steer_df) else None
                               ).to_csv(out / "triplet_report.csv", index=False)
        report = build_report(det_summ, steer_df, cfg, out, manifest)
        print()
        print(report.to_string(index=False))
        print()
        log("report", f"wrote pilot_report.csv / pilot_report.md to {out}")

    model.acts.flush()
    model.margins.flush()
    manifest.update(activations=model.acts.stats(),
                    margins=model.margins.stats(),
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                  default=str), encoding="utf-8")
    log("done", str(out.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
