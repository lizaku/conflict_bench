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
    have_ref = table.reference in conditions
    # `correct` selects which view is PRIMARY; the other is reported beside it
    # whenever R was scored at all (CLAUDE.md invariant 5).
    table.correct = bool(cfg.get("format_correct", False)) and have_ref
    if not have_ref:
        print(f"[cond] reference condition {table.reference.value} not in "
              f"conditions - raw margins only, no corrected view available")
    elif not table.correct:
        print(f"[cond] R scored and the corrected view reported alongside, "
              f"but RAW margins are primary (format_correct: false)")
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


def method_cfg(cfg, name):
    """A method's config block with the run-level settings folded in.

    `task` and `format_correct` are properties of the RUN, not of a method, so
    the runner injects them rather than making every method reach back into
    the global config (or, worse, default them differently from each other).
    An explicit entry in `method_cfg:` still wins.
    """
    mcfg = dict(cfg.get("method_cfg", {}).get(name, {}))
    mcfg.setdefault("task", cfg.get("detection_task", "conflict"))
    mcfg.setdefault("format_correct", cfg.get("format_correct", False))
    return mcfg


def detection_instances(cfg, items, labels):
    """The (item, condition, label) instances the detection axis is scored on.

    Two tasks, and they are genuinely different questions:

    `conflict` (default) - "Does this passage contradict the model's internal
        knowledge?"  Every item is presented TWICE, once with its supporting
        passage and once with its conflicting one, and the label is which.
        Three things follow, all of them improvements on the arbitration
        framing:
          - the base rate is exactly 0.5 by construction, so the relation
            base-rate control has nothing to exploit;
          - the design is paired - both instances share the item, the
            question, the template and both answer strings - so the per-item
            answer-string constant cancels exactly.  That is what the R
            correction was trying to achieve by subtraction, and it is why R
            is no longer needed here;
          - both halves of a pair carry the same relation, so GroupKFold puts
            them in the same fold and a pair is never split across train/test.

    `arbitration` (legacy) - "Will the model follow the context?"  One
        instance per item under C, labelled by what the model actually
        generated.  Kept so the earlier numbers stay reproducible.

    One caveat worth keeping in view for the conflict task: `bow` stops being
    a null and becomes close to an oracle, because whether a passage
    contradicts is partly decidable from the passage text alone.  GroupKFold
    by relation blunts that but does not remove it, so "beats BOW" is a
    weaker claim here than it is under arbitration.  Read the BOW row as a
    ceiling on the text-only route, not as a floor a probe must clear.
    """
    task = cfg.get("detection_task", "conflict")
    if task == "arbitration":
        cond = margins_mod.as_condition(cfg.get("detection_condition", "C"))
        return task, [(i, cond, int(y)) for i, y in enumerate(labels)]
    if task != "conflict":
        raise ValueError(
            f"detection_task '{task}' is neither 'conflict' nor 'arbitration'")
    pos = margins_mod.as_condition(cfg.get("conflict_condition", "C"))
    neg = margins_mod.as_condition(cfg.get("support_condition", "S"))
    inst = []
    for i in range(len(items)):
        inst.append((i, neg, 0))
        inst.append((i, pos, 1))
    return task, inst


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
    # Under the conflict task the condition is part of the INSTANCE (S and C
    # are the two classes), so there is nothing to sweep and the job carries
    # condition=None, meaning "whatever the instance says". The sweep is
    # arbitration-only, where the condition is fixed and the label is not.
    if cfg.get("detection_task", "conflict") == "conflict":
        conditions = [None]
    else:
        conditions = [margins_mod.as_condition(c) for c in
                      cfg.get("detection_conditions",
                              [cfg.get("detection_condition", "C")])]
    jobs = []
    for name in cfg.get("detectors", []):
        if name not in DETECTORS:
            print(f"[detect] unknown detector '{name}' - skipping")
            continue
        cls = DETECTORS[name]
        mcfg = method_cfg(cfg, name)
        # a detector with no position has nothing to sweep; one pinned in
        # method_cfg keeps its pin
        pos_list = ([mcfg["position"]] if "position" in mcfg
                    else list(positions) if cls.uses_position else [None])
        for pi, pos in enumerate(pos_list):
            for ci, cond in enumerate(conditions):
                key = name
                if len(pos_list) > 1:
                    key += f"@{pos}"
                if cond is not None and len(conditions) > 1:
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


def detection_item_mask(cfg, model, items, jobs, instances, out):
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
    # a job with condition=None (the conflict task) is scored under every
    # instance condition, so the position has to resolve under all of them or
    # the two halves of a pair would be measured over different item sets
    inst_conds = sorted({c for _, c, _ in instances},
                        key=lambda c: c.value)
    wanted = set()
    for j in jobs:
        if j["position"] is None:
            continue
        for c in (inst_conds if j["condition"] is None else [j["condition"]]):
            wanted.add((j["position"], c))
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
    """Score every detector over the run's detection instances.

    The unit is an INSTANCE - an (item, condition) pair with a label - not an
    item, because under the conflict task the same item appears twice, once
    as S and once as C, and the label is which one it was.  Everything
    downstream (folds, AUROC, the base-rate control) works on instances, and
    the relation groups are carried per instance so GroupKFold keeps both
    halves of a pair in the same fold.
    """
    det_records, det_summ = [], {}
    n_splits = cfg.get("n_splits", 5)
    task, instances = detection_instances(cfg, items, labels)
    jobs = detection_jobs(cfg)
    keep, dropped = detection_item_mask(cfg, model, items, jobs, instances, out)
    keep_set = set(keep)
    instances = [(i, c, y) for (i, c, y) in instances if i in keep_set]
    if not instances:
        print("[detect] no instances survive the position sweep - skipping")
        return {}

    inst_items = [items[i] for i, _, _ in instances]
    inst_conds = [c for _, c, _ in instances]
    inst_labels = [y for _, _, y in instances]
    inst_groups = [groups[i] for i, _, _ in instances]
    n = len(instances)
    pos_rate = float(np.mean(inst_labels))
    print(f"[detect] task={task}: {n} instances over {len(items)} items, "
          f"{len(set(inst_groups))} relations, positive rate {pos_rate:.3f}")
    if len(set(inst_labels)) < 2:
        print("[detect] instances are single-class - nothing to score")
        return {}

    for job in jobs:
        key, name, cls, mcfg = job["key"], job["method"], job["cls"], job["cfg"]
        # condition=None means "use the instance's own condition" (conflict
        # task); a fixed condition is the arbitration sweep
        job_cond = job["condition"]
        conds = inst_conds if job_cond is None else [job_cond] * n
        t0 = time.time()
        store = artifacts.sub(key.replace("|", "_").replace("@", "_at_"))
        scores = np.full(n, np.nan)
        scores_raw = np.full(n, np.nan)
        scores_corr = np.full(n, np.nan)
        records = []

        def take(r, i):
            scores[i] = r.score
            if r.score_raw is not None:
                scores_raw[i] = r.score_raw
            if r.score_corrected is not None:
                scores_corr[i] = r.score_corrected
            records.append(r)

        try:
            if cls.requires_training:
                folds = group_folds(inst_items, inst_labels, inst_groups,
                                    n_splits)
                for k, (tr, te) in enumerate(folds):
                    D = cls(model, mcfg)
                    D.fit([inst_items[i] for i in tr],
                          [inst_labels[i] for i in tr],
                          conditions=[conds[i] for i in tr])
                    D.save_artifacts(store, tag=f"fold{k}_")
                    for i in te:
                        r = D.score(inst_items[i], conds[i])
                        r.label = inst_labels[i]
                        r.extras["fold"] = k
                        take(r, i)
            else:
                D = cls(model, mcfg)
                for i in range(n):
                    r = D.score(inst_items[i], conds[i])
                    r.label = inst_labels[i]
                    take(r, i)
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
        row = metrics.grouped_auroc(scores, inst_labels, inst_groups,
                                    n_splits=n_splits)
        row.update(method=name, task=task,
                   condition=("instance" if job_cond is None
                              else job_cond.value),
                   position=(str(job["position"])
                             if job["position"] is not None else None),
                   layer=mcfg.get("layer"), primary=job["primary"],
                   n_items=len(items), n_instances=n,
                   positive_rate=pos_rate, n_items_dropped=len(dropped),
                   is_readout_position=positions_mod.is_readout(job["position"])
                   if job["position"] is not None else None)
        # BOTH views of a format-corrected score, always, never one instead of
        # the other - whichever one `score` mirrors is named by `view`.
        primary_is_corr = bool(mcfg.get("format_correct", False))
        row["view"] = "r_corrected" if primary_is_corr else "raw"
        for label, arr in (("raw", scores_raw), ("corrected", scores_corr)):
            if np.isnan(arr).all():
                continue
            alt = metrics.grouped_auroc(arr, inst_labels, inst_groups,
                                        n_splits=n_splits)
            row[f"auroc_{label}"] = alt["auroc"]
            row[f"auroc_{label}_std"] = alt["auroc_std"]
        if not np.isnan(row.get("auroc_raw", np.nan)) and \
                not np.isnan(row.get("auroc_corrected", np.nan)):
            row["auroc_correction_gain"] = (row["auroc_corrected"]
                                            - row["auroc_raw"])
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
        front = [c for c in ("method", "task", "position", "condition",
                             "layer", "view", "auroc", "base_rate_auroc",
                             "readout_auroc", "auroc_over_readout",
                             "auroc_raw", "auroc_corrected", "accuracy",
                             "majority_accuracy", "n_instances")
                 if c in df.columns]
        df = df[front + [c for c in df.columns if c not in front]]
        df.to_csv(out / "detection_summary.csv", index=False)
    return det_summ


def run_steering(cfg, model, items, labels, groups, out, artifacts):
    """Every steerer x every target x every dose x every condition.

    The condition loop is the point.  The benchmark's steering question is

        Given a CONFLICTING passage, can the intervention make the model
        output the conflicting answer when it otherwise would not?

    and "otherwise would not" needs a measurement, not an assumption.  So each
    steerer runs under C (the conflict it is meant to resolve) and under S
    (the same item, same dose, but a passage that agrees with memory).  The S
    arm is a matched specificity control: a method that shifts the margin just
    as hard when there is nothing to arbitrate is pushing on the passage, not
    resolving a conflict.  `metrics.steering_summary` pairs them by item.

    Both targets always run.  Which one a method "wins" on is a property of
    the report, not of the design - and collapsing to the winner by argmax is
    exactly how a 4-of-6-items flip rate came to outrank a 390-of-394 one, so
    the summary keeps every (target, condition) cell and carries n_flippable
    beside every rate.
    """
    steer_records = []
    factors = cfg.get("factors", [0.5, 1.0, 2.0, 4.0])
    targets = cfg.get("targets", ["use_parametric", "use_context"])
    steer_conds = [margins_mod.as_condition(c) for c in
                   cfg.get("steering_conditions",
                           [cfg.get("steering_condition", "C")])]
    n_splits = cfg.get("n_splits", 5)

    for name in cfg.get("steerers", []):
        if name not in STEERERS:
            print(f"[steer] unknown steerer '{name}' - skipping")
            continue
        t0 = time.time()
        cls = STEERERS[name]
        mcfg = method_cfg(cfg, name)
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
            for steer_cond in steer_conds:
                for target in targets:
                    for factor in method_factors:
                        for i in te:
                            try:
                                rec = S.steer(items[i], steer_cond, target,
                                              factor)
                            except NotImplementedError as e:
                                failed = f"steer not implemented ({e})"
                                break
                            except Exception as e:
                                failed = (f"steer failed "
                                          f"({type(e).__name__}: {e})")
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
                break

        if failed:
            print(f"[steer] {name}: {failed} - skipping")
            store.json("error.json", {"error": failed})
            del steer_records[n_before:]
            continue
        store.jsonl("records.jsonl",
                    [r.__dict__ for r in steer_records[n_before:]])
        print(f"[steer] {name}: {len(steer_records) - n_before} records "
              f"over {len(steer_conds)} conditions x {len(targets)} targets "
              f"x {len(method_factors)} doses in "
              f"{round(time.time() - t0, 1)}s")
    return steer_records


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactStore(out / "artifacts")
    manifest = {"config_path": str(cfg_path), "config": cfg,
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    prompts.configure(**(cfg.get("prompt") or {}))

    model = ModelWrapper.from_config(
        cfg, activation_cache_dir=out / "activations",
        margin_table_path=out / "margin_table.jsonl")
    # AFTER the model: ModelWrapper binds the tokenizer, and only then does
    # `chat: auto` resolve - snapshotting the spec earlier records
    # chat_active: false for a run that in fact used the chat template
    manifest["prompt"] = prompts.DEFAULT.as_dict()
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
