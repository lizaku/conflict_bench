"""Run methods on a built benchmark and evaluate them.

    python -m conflict_bench.harness --model microsoft/Phi-3.5-mini-instruct --limit 100

One straight line: load instances -> pick a subset -> for every method, fit on
the training folds and score / steer the held-out fold -> predictions files ->
evaluate.  The harness, not the method, builds prompts, generates the steered
answer, matches answers and counts forward passes.

`--limit N` takes about N instances, whole facts at a time (so the paired
instances of a fact stay together), spread over relations.  `--recipes`
restricts the taxonomy cells; the config default is the four core ones.

Output: bench/<model>/runs/<name>/
    predictions/<method>.detection.jsonl    one row per instance and task
    predictions/<method>.steering.jsonl     one row per instance, direction, strength
    detection.csv  localization.csv  steering.csv  (+ *_uncertain.csv)  report.md
    artifacts/<method>/                     probe weights, steering vectors
    manifest.json                           config, instance ids, skipped methods
"""
import argparse
import json
import time
import traceback
from collections import Counter

import conflict_bench.methods  # noqa: F401  (registers every method)
from conflict_bench import evaluate
from conflict_bench.api import DETECTORS, STEERERS, PublicInstance
from conflict_bench.construct.common import (load_config, load_model,
                                             model_dir, read_jsonl, spread,
                                             utf8_stdout, write_jsonl)
from conflict_bench.core import prompts
from conflict_bench.core.artifacts import ArtifactStore


# ------------------------------------------------------------------ selection
def select(recs, limit=None, recipes=None, formats=None, seed=0):
    recs = [r for r in recs if r.get("measured")
            and (not recipes or r["recipe"] in recipes)
            and (not formats or r["format"] in formats)]
    return spread(recs, limit, seed)


def labels_for(rec, task) -> dict:
    """What `fit` is allowed to see: labels, never an answer string."""
    t, d, follows = rec["targets"], rec.get("d"), None
    if d and t.get("ctx") and t.get("mem") and t["ctx"] != t["mem"]:
        follows = {(t["ctx"],): "ctx", (t["mem"],): "mem"}.get(tuple(d["matches"]))
    return {"conflict": rec["labels"]["conflict_global" if task == "global"
                                      else "conflict_q"],
            "follows": follows, "recipe": rec["recipe"],
            "chunks": rec["labels"]["chunk_asserts_b"]}


def _spec(entry):
    """'name' or {name: ..., as: ..., **cfg} -> (name, alias, cfg)."""
    if isinstance(entry, str):
        return entry, entry, {}
    cfg = dict(entry)
    name = cfg.pop("name")
    return name, cfg.pop("as", name), cfg


def _folds(cls, model, mcfg, recs, pubs, task, view, log, store, alias):
    """Yield (fitted method, held-out records). Trainable methods are fitted
    once per fold on the other folds; training-free ones are built once."""
    if not cls.requires_training:
        yield cls(model, mcfg), recs
        return
    for k in sorted({r["fold"] for r in recs}):
        train = [(view(pubs[r["instance_id"]]), labels_for(r, task))
                 for r in recs if r["fold"] != k and not r["has_uncertain"]]
        test = [r for r in recs if r["fold"] == k]
        method = cls(model, mcfg)
        try:
            if not train:
                raise ValueError("no training instances outside this fold")
            method.fit(train)
            method.save_artifacts(store.sub(alias), tag=f"{task}_fold{k}_")
        except ValueError as e:
            log(f"{alias} [{task}] fold {k} skipped: {e}")
            continue
        yield method, test


# ------------------------------------------------------------------ detection
def run_detector(model, entry, recs, pubs, tasks, log, store):
    name, alias, mcfg = _spec(entry)
    cls = DETECTORS[name]
    rows = []
    for task in tasks:
        view = (lambda p: p.without_question()) if task == "global" else (lambda p: p)
        for det, test in _folds(cls, model, mcfg, recs, pubs, task, view, log,
                                store, alias):
            for r in test:
                inst = view(pubs[r["instance_id"]])
                n0 = model.forward_passes
                out = det.score(inst, r) if cls.oracle else det.score(inst)
                if out.conflict != out.conflict:          # NaN: not applicable
                    continue
                rows.append({"instance_id": r["instance_id"], "method": alias,
                             "task": task, "score": float(out.conflict),
                             "chunks": out.chunks, "label": out.label,
                             "forward_passes": model.forward_passes - n0,
                             "access": cls.access, "oracle": cls.oracle})
    return alias, rows


# ------------------------------------------------------------------- steering
def apply(model, inst, iv, answers, max_new_tokens, score_margin):
    """Generate under an intervention. -> (x, {answer: log-prob})."""
    prompt = iv.prompt or inst.prompt
    with iv.installed():
        x = model.greedy(prompt, max_new_tokens, iv.logits)
        lp = {a: model.logp_steered(prompt, prompts.continuation(a), iv.logits)
              for a in answers} if score_margin else None
    return x, lp


def run_steerer(model, entry, recs, pubs, scfg, log, store):
    name, alias, mcfg = _spec(entry)
    cls = STEERERS[name]
    strengths = mcfg.get("strengths", cls.strengths)
    max_new = scfg.get("max_new_tokens", 12)
    rows = []
    for st, test in _folds(cls, model, mcfg, recs, pubs, "qr", lambda p: p,
                           log, store, alias):
        for r in test:
            inst = pubs[r["instance_id"]]
            jobs = [(d, None, r["targets"].get(d)) for d in ("ctx", "mem")
                    if d in scfg.get("directions", ["ctx", "mem", "span"])]
            if "span" in scfg.get("directions", ["ctx", "mem", "span"]):
                jobs += [("span", [pid], t)
                         for pid, t in (r["targets"].get("span") or {}).items()]
            for direction, span, target in jobs:
                if target is None:
                    continue
                for s in strengths:
                    n0 = model.forward_passes
                    iv = st.intervene(inst, direction, s, span)
                    if iv is None:
                        break
                    x, lp = apply(model, inst, iv, r["answers"], max_new,
                                  scfg.get("score_margin", True))
                    cost = model.forward_passes - n0
                    ctrl = st.control(inst, direction, s, span)
                    xc = apply(model, inst, ctrl, r["answers"], max_new,
                               False)[0] if ctrl else None
                    rows.append({
                        "instance_id": r["instance_id"], "method": alias,
                        "direction": direction if not span
                        else f"span={''.join(span)}",
                        "strength": s, "target": target, "x": x, "logp": lp,
                        "x_control": xc, "forward_passes": cost,
                        "access": cls.access})
    return alias, rows


# --------------------------------------------------------------------- driver
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model")
    ap.add_argument("--config")
    ap.add_argument("--limit", type=int, help="about this many instances")
    ap.add_argument("--recipes", help="comma-separated; 'all' for every recipe")
    ap.add_argument("--formats", help="short,cloze")
    ap.add_argument("--detectors", help="comma-separated subset; 'none' to skip")
    ap.add_argument("--steerers", help="comma-separated subset; 'none' to skip")
    ap.add_argument("--name", help="run folder name (default: n<limit>)")
    args = ap.parse_args(argv)
    utf8_stdout()
    cfg = load_config(args.config, args.model)
    hcfg = cfg.get("harness", {})

    split = lambda s: None if s in (None, "all") else [x.strip() for x in s.split(",")]
    recipes = split(args.recipes) if args.recipes else hcfg.get("recipes")
    limit = args.limit or hcfg.get("limit")
    base = model_dir(cfg)
    all_recs = read_jsonl(base / "instances.jsonl")
    if not all_recs:
        raise SystemExit(f"no instances under {base} - run "
                         f"`python -m conflict_bench.construct.build` first")
    recs = select(all_recs, limit, recipes, split(args.formats), cfg.get("seed", 0))
    if not recs:
        raise SystemExit("no measured instances match the selection - the "
                         "`measure` step of the build must have run on them")
    out = base / "runs" / (args.name or f"n{limit or 'all'}")
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(out / "artifacts")
    print(f"[harness] {len(recs)} instances from "
          f"{len({r['group'] for r in recs})} facts; recipes "
          f"{dict(Counter(r['recipe'] for r in recs))}")

    def pick(entries, arg):
        if arg == "none":
            return []
        want = split(arg)
        return [e for e in entries if not want or _spec(e)[1] in want]

    model = load_model(cfg)
    pubs = {r["instance_id"]: PublicInstance.from_record(r) for r in recs}
    skipped, timing = [], {}

    def log(msg):
        print(f"[harness] {msg}")
        skipped.append(msg)

    def guarded(kind, fn, entry, *a):
        alias, t0 = _spec(entry)[1], time.time()
        try:
            alias, rows = fn(model, entry, *a)
        except Exception as e:           # one method must not sink the run
            traceback.print_exc()
            return log(f"{alias} failed: {type(e).__name__}: {e}")
        write_jsonl(out / "predictions" / f"{alias}.{kind}.jsonl", rows)
        timing[alias] = round(time.time() - t0, 1)
        print(f"[harness] {kind:9s} {alias:22s} {len(rows):6d} rows "
              f"{timing[alias]:8.1f}s")

    tasks = [t for t in hcfg.get("tasks", ["qr", "global"]) if t in ("qr", "global")]
    for entry in pick(cfg.get("detectors", []), args.detectors):
        guarded("detection", run_detector, entry, recs, pubs, tasks, log, store)
    scfg = cfg.get("steering", {})
    for entry in pick(cfg.get("steerers", []), args.steerers):
        guarded("steering", run_steerer, entry, recs, pubs, scfg, log, store)
    model.acts.flush()

    (out / "manifest.json").write_text(json.dumps({
        "model": model.describe(), "prompt": prompts.DEFAULT.as_dict(),
        "config": cfg, "n_instances": len(recs),
        "recipes": dict(Counter(r["recipe"] for r in recs)),
        "instance_ids": [r["instance_id"] for r in recs],
        "seconds": timing, "skipped": skipped}, indent=2, default=str),
        encoding="utf-8")
    evaluate.run(out, recs)


if __name__ == "__main__":
    main()
