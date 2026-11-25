"""Step 1 - seed facts.  No model.  ConfiQA records -> facts.jsonl + pools.json

Only the *facts and templates* of the seed are used: subject, relation,
question, cloze template, answer and aliases.  Its passages are not read -
every context sentence is later produced by filling a cloze template.

    facts.jsonl   one fact per record
    pools.json    {relation: {qid: {name, aliases}}} - every entity that
                  appears as an answer of the relation (original or
                  counterfactual in ConfiQA); candidate values for `b`
"""
import ast
import json
from collections import Counter

import yaml

from conflict_bench.construct import match
from conflict_bench.construct.common import (ROOT, resolve, seed_dir,
                                             write_jsonl)

RELATIONS = ROOT / "construct" / "relations.yaml"


def load_relations() -> dict:
    return yaml.safe_load(RELATIONS.read_text(encoding="utf-8"))


def cloze_stem(template):
    """The cloze sentence up to its blank, or None when the blank is not at
    the end (868 of 6000 records) - those facts are short-answer only."""
    t = template.strip()
    for blank in ("____.", "____"):
        if t.endswith(blank):
            return t[: -len(blank)].rstrip()
    return None


def assign_folds(counts: dict, n_folds: int) -> dict:
    """relation -> fold.  Largest relation first, each into the lightest fold,
    so folds are balanced in facts and no relation is ever split."""
    load = [0] * n_folds
    out = {}
    for rel, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        k = load.index(min(load))
        out[rel] = k
        load[k] += n
    return out


def fold_map(facts, n_folds: int) -> dict:
    """Balance the folds on the single-valued relations: only their facts are
    ever on an answer path, so only they are instances.  The multi-valued
    relations (fillers, hop 1) are balanced separately."""
    out = {}
    for flag in (True, False):
        out.update(assign_folds(
            Counter(f["relation"] for f in facts if f["single_valued"] == flag),
            n_folds))
    return out


def run(cfg):
    src = resolve(cfg.get("seed_data", "data/ConfiQA_cloze_clustered"))
    files = sorted(src.glob("P*.json"))
    if not files:
        raise FileNotFoundError(f"no P*.json under {src}")
    rels = load_relations()
    facts, rejected, pools = [], [], {}

    def add_to_pool(rel, qid, name, aliases):
        if not match.names(name):
            return
        e = pools.setdefault(rel, {}).setdefault(qid, {"name": name, "aliases": []})
        e["aliases"] = sorted(set(e["aliases"]) | set(aliases or []))

    for f in files:
        rel = f.stem
        meta = rels.get(rel)
        if meta is None:
            raise KeyError(f"{rel} is missing from construct/relations.yaml")
        for i, r in enumerate(json.loads(f.read_text(encoding="utf-8"))):
            fid = f"{rel}:{i}"
            s_qid, _, o_qid = ast.literal_eval(r["orig_path"])[0]
            subject = str(ast.literal_eval(r["orig_path_labeled"])[0][0])
            cf_qid = ast.literal_eval(r["cf_path"])[0][2]
            add_to_pool(rel, o_qid, r["orig_answer"], r.get("orig_alias"))
            add_to_pool(rel, cf_qid, r["cf_answer"], r.get("cf_alias"))
            cloze = r["cloze_template"].strip()
            reason = None
            if not match.names(r["orig_answer"]):
                reason = "answer shorter than 3 characters"
            elif cloze.count("____") != 1 or subject not in cloze:
                reason = "cloze template has no single blank or no subject"
            elif match.has(subject, match.names(r["orig_answer"])):
                reason = "answer is contained in the subject"
            if reason:
                rejected.append({"step": "seed", "fact_id": fid, "reason": reason})
                continue
            facts.append({
                "fact_id": fid, "relation": rel, "relation_label": meta["label"],
                "single_valued": bool(meta.get("single_valued")),
                "subject": subject, "subject_qid": s_qid,
                "gold": r["orig_answer"], "gold_qid": o_qid,
                "gold_aliases": list(r.get("orig_alias") or []),
                "question": r["question"].strip(), "cloze": cloze,
                "cloze_stem": cloze_stem(cloze)})

    folds = fold_map(facts, int(cfg.get("n_folds", 5)))
    for f in facts:
        f["fold"] = folds[f["relation"]]

    out = seed_dir(cfg)
    write_jsonl(out / "facts.jsonl", facts)
    write_jsonl(out / "rejected.jsonl", rejected)
    (out / "pools.json").write_text(json.dumps(pools, ensure_ascii=False),
                                    encoding="utf-8")
    print(f"[seed] {len(facts)} facts, {len(rejected)} rejected, "
          f"{len(pools)} relations -> {out}")
    return facts, pools


def load(cfg):
    """(facts, pools) from disk, running the step if it has not been run."""
    from conflict_bench.construct.common import read_jsonl
    out = seed_dir(cfg)
    if not (out / "facts.jsonl").exists():
        return run(cfg)
    return (read_jsonl(out / "facts.jsonl"),
            json.loads((out / "pools.json").read_text(encoding="utf-8")))
