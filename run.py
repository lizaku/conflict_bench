"""Runner: one config -> all methods -> tidy outputs.

    python -m conflict_bench.run --config conflict_bench/configs/default.yaml

Outputs (out_dir):
  manifest.json             config snapshot, model description, timings, QC
  labels.csv                behavioural label + the generation behind each
  qc_dropped.json           everything the loader/QC threw away, with reasons
  condition_margins.csv     the N/S/C/R design matrix, per item: log-probs and
                            margins under every condition, raw and R-corrected
  conditions_summary.csv    per-condition means / context-win rates, raw and
                            corrected - the "did the manipulation do anything"
                            table, and the check that R is only formatting
  margin_table.jsonl        the cached teacher-forced scores behind both
  detection_records.jsonl   per-item detector scores (with extras)
  steering_records.jsonl    per-item per-factor steering results
  detection_summary.csv     grouped AUROC per (method, position, condition),
                            with the base-rate control AND the logit-lens
                            readout control at the same layer/position
  steering_summary.csv      flip rates / delta-margin / specific effect
  triplet_report.csv        the mandatory joint table
  activations/layer*.npz    residual-stream tensors (shared cache, resumable)
  artifacts/<method>/       fitted parameters: probe coefs, w, CAA vectors

Both axes fit inside a GroupKFold-by-relation split when the method needs
training, so a probe is never scored on a relation it saw and a CAA vector is
never applied to the items it was built from.
"""
import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

from conflict_bench.core.model import ModelWrapper
from conflict_bench.core.types import Condition
from conflict_bench.core.artifacts import ArtifactStore
from conflict_bench.core import prompts
from conflict_bench.core import margins as margins_mod
from conflict_bench.core import positions as positions_mod
from conflict_bench.methods.base import DETECTORS, STEERERS
import conflict_bench.methods.detectors.detectors  # noqa: F401 (register)
import conflict_bench.methods.steerers.steerers    # noqa: F401 (register)
from conflict_bench import data as data_mod
from conflict_bench import metrics


def behavioural_labels(model, items, condition=Condition.CONFLICTING,
                       max_new_tokens=8):
    """Label = 1 if model follows context (closed-world generation check,
    NOT teacher-forcing - the parametric axis must be defined by generation).
    Returns (labels, rows) where rows carry the generation behind each label."""
    labels, rows = [], []
    for it in items:
        prompt = prompts.build_prompt(it, condition)
        gen = model.generate(prompt, max_new_tokens=max_new_tokens)[0]
        g = gen.strip().lower()
        follows = any(a.lower() in g for a in
                      [it.counterfactual_answer] + it.cf_aliases)
        keeps = any(a.lower() in g for a in
                    [it.true_answer] + it.true_aliases)
        labels.append(int(follows))
        rows.append({"item_id": it.item_id, "relation": it.relation,
                     "label": int(follows), "matched_parametric": int(keeps),
                     "generation": gen})
    return labels, rows


def stratified_limit(items, n, seed=0):
    """Take n items spread evenly over relations.

    items[:n] would take whole relations in filename order and leave the run
    with one or two groups - GroupKFold-by-relation then has nothing to split.
    """
    if not n or n >= len(items):
        return items
    rng = np.random.default_rng(seed)
    by_rel = {}
    for it in items:
        by_rel.setdefault(it.relation, []).append(it)
    for v in by_rel.values():
        rng.shuffle(v)
    picked, rels, i = [], sorted(by_rel), 0
    while len(picked) < n:
        pool = by_rel[rels[i % len(rels)]]
        if pool:
            picked.append(pool.pop())
        i += 1
        if all(not by_rel[r] for r in rels):
            break
    return sorted(picked, key=lambda it: (it.relation, it.item_id))


def group_folds(items, labels, groups, n_splits):
    """GroupKFold indices, or a single trivial fold when there is nothing to
    split (one relation) - callers still get (train, test) pairs."""
    n_splits = min(n_splits, len(set(groups)))
    if n_splits < 2:
        idx = np.arange(len(items))
        return [(idx, idx)]
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.arange(len(items)), labels, groups))


def run_conditions(cfg, model, items, labels, out):
    """Score the whole N/S/C/R design and report it, raw and R-corrected.

    This is the design the benchmark claims to run, so it runs as its own
    pass rather than only surviving as a hidden offset inside two methods.
    It is also what populates the shared MarginTable, so every later method
    that wants a margin - the margin detector, every steerer's baseline and
    R offset - reads it from here instead of recomputing it.
    """
    conditions = [margins_mod.as_condition(c)
                  for c in cfg.get("conditions", ["N", "S", "C", "R"])]
    table = model.margins
    if table.reference not in conditions:
        table.correct = False
        print(f"[cond] reference condition {table.reference.value} not in "
              f"conditions - reporting raw margins only, no correction")
    t0 = time.time()
    table.score_all(items, conditions)
    rows = table.rows(items, conditions, labels)
    pd.DataFrame(rows).to_csv(out / "condition_margins.csv", index=False,
                              encoding="utf-8")
    summ = margins_mod.condition_summary(rows, conditions, table.reference)
    summ.to_csv(out / "conditions_summary.csv", index=False)
    table.flush()
    print(f"[cond] scored {len(conditions)} conditions x {len(items)} items "
          f"in {round(time.time() - t0, 1)}s")
    print(summ.to_string(index=False))
    return summ


def detection_jobs(cfg):
    """The (detector, position, condition) grid to run.

    Position and condition are swept, not fixed, because the headline claim -
    that arbitration is *detectable* - depends entirely on where and when you
    read.  A probe at the last prompt token under C, scored against the
    model's own continuation from that same prompt, is mostly decoding an
    answer already committed to; the same probe at end_of_context is making a
    prediction.  Both are run so the difference is a number in the table
    instead of an assumption in the config.
    """
    positions = cfg.get("detection_positions", ["end_of_context", "last"])
    conditions = [margins_mod.as_condition(c) for c in
                  cfg.get("detection_conditions",
                          [cfg.get("detection_condition", "C")])]
    jobs = []
    for name in cfg.get("detectors", []):
        if name not in DETECTORS:
            print(f"[detect] unknown detector '{name}' - skipping")
            continue
        cls = DETECTORS[name]
        mcfg = dict(cfg.get("method_cfg", {}).get(name, {}))
        # a detector with no position has nothing to sweep; one pinned in
        # method_cfg keeps its pin
        pos_list = ([mcfg["position"]] if "position" in mcfg
                    else list(positions) if cls.uses_position else [None])
        for pi, pos in enumerate(pos_list):
            for ci, cond in enumerate(conditions):
                key = name
                if len(pos_list) > 1:
                    key += f"@{pos}"
                if len(conditions) > 1:
                    key += f"|{cond.value}"
                jcfg = dict(mcfg)
                if pos is not None:
                    jcfg["position"] = pos
                jobs.append({"key": key, "method": name, "cls": cls,
                             "cfg": jcfg, "position": pos, "condition": cond,
                             "primary": pi == 0 and ci == 0})
    return jobs


def attach_readout_control(det_summ, readout_method="logit_lens"):
    """Pair every activation-reading detector with the logit-lens readout at
    the same (layer, position, condition).

    `auroc_over_readout` is the number that decides whether a probe result is
    about arbitration or about decoding: it is how much the probe beats the
    model's own unembedding read at the identical site.  Near zero at the last
    prompt token is the expected, uninteresting result - the probe and the
    unembedding are reading the same committed answer.  A positive value at
    end_of_context is the claim worth making.
    """
    readouts = {}
    for row in det_summ.values():
        if row.get("method") == readout_method:
            readouts[(row.get("position"), row.get("condition"),
                      row.get("layer"))] = row.get("auroc")
    if not readouts:
        return det_summ
    for row in det_summ.values():
        if row.get("method") == readout_method or row.get("position") in (None, "None"):
            continue
        ref = readouts.get((row.get("position"), row.get("condition"),
                            row.get("layer")))
        if ref is not None and not np.isnan(ref):
            row["readout_auroc"] = ref
            row["auroc_over_readout"] = row["auroc"] - ref
    return det_summ


def detection_item_mask(cfg, model, items, jobs, out):
    """Items resolvable at EVERY swept position/condition.

    Some items have no verbatim mention of their counterfactual answer in the
    passage, or a subject that does not appear verbatim in the question, so a
    named position does not exist for them.  Two things follow:

      - a single such item must not kill a whole detector row, and
      - AUROC at end_of_context and AUROC at `last` are only comparable if
        they are computed over the SAME items, so the sweep runs on the
        intersection rather than letting each position pick its own subset.

    The whole detection table then sits on one item set, and what was dropped
    is written to detection_dropped.json rather than silently absorbed.
    """
    wanted = {(j["position"], j["condition"]) for j in jobs
              if j["position"] is not None}
    if not wanted:
        return list(range(len(items))), []
    keep, dropped = [], []
    for i, it in enumerate(items):
        reasons = []
        for pos, cond in sorted(wanted, key=lambda x: (str(x[0]), x[1].value)):
            try:
                positions_mod.resolve(model, it, cond, pos)
            except positions_mod.PositionError as e:
                reasons.append(f"{pos}@{cond.value}: {e}")
        (dropped if reasons else keep).append(
            {"item_id": it.item_id, "reasons": reasons} if reasons else i)
    if dropped:
        (out / "detection_dropped.json").write_text(
            json.dumps(dropped, indent=2), encoding="utf-8")
        print(f"[detect] {len(dropped)}/{len(items)} items dropped: a swept "
              f"position does not exist for them (detection_dropped.json)")
    return keep, dropped


def run_detection(cfg, model, items, labels, groups, out, artifacts):
    det_records, det_summ = [], {}
    n_splits = cfg.get("n_splits", 5)
    jobs = detection_jobs(cfg)
    keep, dropped = detection_item_mask(cfg, model, items, jobs, out)
    items = [items[i] for i in keep]
    labels = [labels[i] for i in keep]
    groups = [groups[i] for i in keep]
    if not items:
        print("[detect] no items survive the position sweep - skipping")
        return {}

    for job in jobs:
        key, name, cls, mcfg = job["key"], job["method"], job["cls"], job["cfg"]
        cond = job["condition"]
        t0 = time.time()
        store = artifacts.sub(key.replace("|", "_").replace("@", "_at_"))
        scores = np.full(len(items), np.nan)
        scores_raw = np.full(len(items), np.nan)
        records = []
        try:
            if cls.requires_training:
                for k, (tr, te) in enumerate(
                        group_folds(items, labels, groups, n_splits)):
                    D = cls(model, mcfg)
                    D.fit([items[i] for i in tr], [labels[i] for i in tr])
                    D.save_artifacts(store, tag=f"fold{k}_")
                    for i in te:
                        r = D.score(items[i], cond)
                        r.label = labels[i]
                        r.extras["fold"] = k
                        scores[i] = r.score
                        if r.score_raw is not None:
                            scores_raw[i] = r.score_raw
                        records.append(r)
            else:
                D = cls(model, mcfg)
                for i, (it, y) in enumerate(zip(items, labels)):
                    r = D.score(it, cond)
                    r.label = y
                    scores[i] = r.score
                    if r.score_raw is not None:
                        scores_raw[i] = r.score_raw
                    records.append(r)
                D.save_artifacts(store)
        except NotImplementedError as e:
            print(f"[detect] {key}: not implemented ({e}) - skipping")
            continue
        except positions_mod.PositionError as e:
            print(f"[detect] {key}: position unavailable ({e}) - skipping")
            continue
        except Exception as e:                       # keep the sweep alive
            print(f"[detect] {key}: FAILED ({type(e).__name__}: {e})")
            store.json("error.json", {"error": repr(e)})
            continue

        det_records.extend(records)
        store.jsonl("scores.jsonl", [r.__dict__ for r in records])
        row = metrics.grouped_auroc(scores, labels, groups, n_splits=n_splits)
        row.update(method=name, condition=cond.value,
                   position=(str(job["position"])
                             if job["position"] is not None else None),
                   layer=mcfg.get("layer"), primary=job["primary"],
                   n_items=len(items), n_items_dropped=len(dropped),
                   is_readout_position=positions_mod.is_readout(job["position"])
                   if job["position"] is not None else None)
        if not np.isnan(scores_raw).all():
            # the same detector before its R correction - reported beside the
            # corrected number, never instead of it
            raw = metrics.grouped_auroc(scores_raw, labels, groups,
                                        n_splits=n_splits)
            row.update(auroc_raw=raw["auroc"], auroc_raw_std=raw["auroc_std"],
                       auroc_correction_gain=row["auroc"] - raw["auroc"])
        row["seconds"] = round(time.time() - t0, 1)
        det_summ[key] = row
        print(f"[detect] {key}: auroc={row['auroc']:.3f} "
              f"base_rate={row['base_rate_auroc']:.3f} "
              f"pos={row['position']} cond={row['condition']}")

    attach_readout_control(det_summ)

    if det_records:
        with open(out / "detection_records.jsonl", "w", encoding="utf-8") as f:
            for r in det_records:
                f.write(json.dumps(r.__dict__, default=str,
                                   ensure_ascii=False) + "\n")
    if det_summ:
        df = pd.DataFrame(det_summ).T
        front = [c for c in ("method", "position", "condition", "layer",
                             "auroc", "base_rate_auroc", "readout_auroc",
                             "auroc_over_readout", "is_readout_position")
                 if c in df.columns]
        df = df[front + [c for c in df.columns if c not in front]]
        df.to_csv(out / "detection_summary.csv")
    return det_summ


def run_steering(cfg, model, items, labels, groups, out, artifacts):
    steer_records = []
    factors = cfg.get("factors", [0.5, 1.0, 2.0, 4.0])
    targets = cfg.get("targets", ["use_parametric", "use_context"])
    steer_cond = margins_mod.as_condition(cfg.get("steering_condition", "C"))
    n_splits = cfg.get("n_splits", 5)

    for name in cfg.get("steerers", []):
        if name not in STEERERS:
            print(f"[steer] unknown steerer '{name}' - skipping")
            continue
        t0 = time.time()
        cls = STEERERS[name]
        mcfg = cfg.get("method_cfg", {}).get(name, {})
        store = artifacts.sub(name)
        # prompt_instruct's factor is a discrete variant index, not a dose;
        # let a method declare its own sweep instead of re-running the same
        # two templates once per continuous alpha in the global list.
        method_factors = mcfg.get("factors", factors)
        # Fitted steerers (act_add) get the same fold discipline as probes:
        # vectors built on train relations, applied to held-out relations.
        if cls.requires_training and not mcfg.get("vectors_path"):
            folds = group_folds(items, labels, groups,
                                cfg.get("steering_n_splits", 2))
        else:
            allidx = np.arange(len(items))
            folds = [(allidx, allidx)]

        n_before = len(steer_records)
        failed = None
        for k, (tr, te) in enumerate(folds):
            S = cls(model, mcfg)
            fit_items = [items[i] for i in tr]
            takes_labels = len(inspect.signature(S.fit).parameters) > 1
            try:
                S.fit(fit_items, [labels[i] for i in tr]) if takes_labels \
                    else S.fit(fit_items)
            except NotImplementedError as e:
                failed = f"fit not implemented ({e})"
                break
            except Exception as e:
                failed = f"fit failed ({type(e).__name__}: {e})"
                break
            S.save_artifacts(store, tag=f"fold{k}_" if len(folds) > 1 else "")
            for target in targets:
                for factor in method_factors:
                    for i in te:
                        try:
                            rec = S.steer(items[i], steer_cond, target, factor)
                        except NotImplementedError as e:
                            failed = f"steer not implemented ({e})"
                            break
                        except Exception as e:
                            failed = f"steer failed ({type(e).__name__}: {e})"
                            break
                        rec.extras.setdefault("fold", k)
                        steer_records.append(rec)
                    if failed:
                        break
                if failed:
                    break
            if failed:
                break

        if failed:
            print(f"[steer] {name}: {failed} - skipping")
            store.json("error.json", {"error": failed})
            del steer_records[n_before:]
            continue
        store.jsonl("records.jsonl",
                    [r.__dict__ for r in steer_records[n_before:]])
        print(f"[steer] {name}: {len(steer_records) - n_before} records "
              f"in {round(time.time() - t0, 1)}s")
    return steer_records


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactStore(out / "artifacts")
    manifest = {"config_path": str(cfg_path), "config": cfg,
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    prompts.configure(**(cfg.get("prompt") or {}))
    manifest["prompt"] = prompts.DEFAULT.as_dict()

    model = ModelWrapper.from_config(
        cfg, activation_cache_dir=out / "activations",
        margin_table_path=out / "margin_table.jsonl")
    manifest["model"] = model.describe()
    print(f"[model] {manifest['model']}")

    # ---------------- data + QC ----------------
    loader = getattr(data_mod, cfg["dataset"]["loader"])
    loaded = loader(**cfg["dataset"].get("kwargs", {}))
    items, dropped = loaded if isinstance(loaded, tuple) else (loaded, [])

    qc = cfg.get("qc", {})
    if qc.get("drop_first_token_collisions", False):
        collisions = data_mod.first_token_collisions(items, model.tok)
        keep = [it for it in items if not it.meta.get("first_token_collision")]
        dropped += [{"item_id": i, "reasons": ["first_token_collision"]}
                    for i in collisions]
        print(f"[qc] dropped {len(collisions)} first-token-collision items")
        items = keep
    items = stratified_limit(items, cfg.get("limit"), cfg.get("seed", 0))
    (out / "qc_dropped.json").write_text(json.dumps(dropped, indent=2),
                                         encoding="utf-8")
    manifest["n_items"] = len(items)
    manifest["n_dropped"] = len(dropped)
    manifest["relations"] = sorted({it.relation for it in items})
    print(f"[data] {len(items)} items over {len(manifest['relations'])} relations")

    # ---------------- behavioural labels ----------------
    t0 = time.time()
    labels, label_rows = behavioural_labels(model, items)
    groups = [it.relation for it in items]
    pd.DataFrame(label_rows).to_csv(out / "labels.csv", index=False,
                                    encoding="utf-8")
    manifest["label_seconds"] = round(time.time() - t0, 1)
    manifest["context_following_rate"] = float(np.mean(labels)) if labels else None
    print(f"[labels] context-following rate = {manifest['context_following_rate']:.3f}")

    run_conditions(cfg, model, items, labels, out)
    det_summ = run_detection(cfg, model, items, labels, groups, out, artifacts)
    steer_records = run_steering(cfg, model, items, labels, groups, out, artifacts)

    summ = None
    if steer_records:
        with open(out / "steering_records.jsonl", "w", encoding="utf-8") as f:
            for r in steer_records:
                f.write(json.dumps(r.__dict__, default=str,
                                   ensure_ascii=False) + "\n")
        summ = metrics.steering_summary(steer_records)
        summ.to_csv(out / "steering_summary.csv", index=False)
        print(summ.to_string(index=False))

    metrics.triplet_report(det_summ, summ).to_csv(
        out / "triplet_report.csv", index=False)

    model.acts.flush()
    model.margins.flush()
    manifest["activations"] = model.acts.stats()
    manifest["margins"] = model.margins.stats()
    manifest["conditions"] = [margins_mod.as_condition(c).value
                              for c in cfg.get("conditions", ["N", "S", "C", "R"])]
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                  default=str), encoding="utf-8")
    print(f"[done] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conflict_bench/configs/default.yaml")
    main(ap.parse_args().config)
