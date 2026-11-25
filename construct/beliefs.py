"""Step 2 - what does M know?   facts.jsonl + M -> beliefs.jsonl   (GPU)

Closed-book, per fact.  Condition 1 of the conflict definition is
"a = M(q) is stable and confident", so two things are measured:

  stable     the greedy answer to several forms of the question
             (seed question, cloze, two paraphrases) names the same entity
  confident  K sampled answers agree; their entropy and the teacher-forced
             probability of `a` are the two confidence rows of the taxonomy

Every raw signal is stored, so the state can be re-derived under another
definition of "known" without touching the model again.

An answer is identified by *which entity of the relation's answer pool it
names* (construct/match.py).  Answers that name no pool entity are clustered
by their content words; a fact whose stable answer is outside the pool is
marked `known` with `a_qid: null` and is not used to build contexts (we could
not write a sentence that supports or contradicts it reliably).
"""
import math
import re
from collections import Counter

from conflict_bench.core import prompts
from conflict_bench.construct import match
from conflict_bench.construct.common import (Appender, model_dir, read_jsonl)
from conflict_bench.construct.seed import load_relations

ABSTAIN = re.compile(
    r"\b(i (do not|don't|cannot|can't|am not|'m not|couldn't|could not)"
    r"|not sure|no (reliable |specific )?information|not aware|unable to"
    r"|not (publicly )?(known|available)|as an ai|there is no)\b", re.I)

_STOP = set("the a an is was are were of in on at to for and by with it its "
            "this that as be has have had".split())


def entropy_bits(counts) -> float:
    n = sum(counts)
    return -sum(c / n * math.log2(c / n) for c in counts if c) if n else 0.0


def select_facts(facts, limit, seed=0):
    """`limit` facts spread evenly over relations (all of them if no limit)."""
    import random
    if not limit or limit >= len(facts):
        return list(facts)
    by_rel = {}
    for f in facts:
        by_rel.setdefault(f["relation"], []).append(f)
    rng = random.Random(seed)
    for v in by_rel.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < limit:
        for rel in sorted(by_rel):
            if i < len(by_rel[rel]) and len(out) < limit:
                out.append(by_rel[rel][i])
        i += 1
    return out


class Clusterer:
    """Maps a generated answer to a cluster id for one fact."""

    def __init__(self, fact, pool):
        stem_text = " ".join([fact["question"], fact["cloze"], fact["subject"]])
        self.stop = _STOP | set(match.norm(stem_text).split())
        # pool entities that the question itself mentions cannot be told apart
        # from the answer by string matching, so they are not candidates
        self.entities = [(qid, ns) for qid, e in pool.items()
                         for ns in [match.names(e["name"], e["aliases"])]
                         if ns and not match.has(stem_text, ns)]

    def __call__(self, text) -> str:
        if ABSTAIN.search(text):
            return "ABSTAIN"
        t = f" {match.norm(text)} "
        best = (len(t) + 1, 0, None)          # earliest mention, then longest
        for qid, ns in self.entities:
            for n in ns:
                i = t.find(f" {n} ")
                if i >= 0 and (i, -len(n)) < (best[0], best[1]):
                    best = (i, -len(n), qid)
        if best[2]:
            return best[2]
        words = [w for w in t.split() if w not in self.stop][:2]
        return "OTHER:" + " ".join(words)


def state_of(modal, sample_agree, forms_agree, n_forms, th) -> str:
    if modal == "ABSTAIN" or sample_agree < th["unknown_samples"]:
        return "unknown"
    if (sample_agree >= th["known_samples"]
            and forms_agree >= math.ceil(th["known_forms"] * n_forms)):
        return "known"
    return "uncertain"


def elicit(model, fact, pool, rel, bcfg) -> dict:
    cluster = Clusterer(fact, pool)
    max_new = bcfg.get("max_new_tokens", 12)
    stems = [("question", fact["question"])]
    if fact["cloze_stem"]:
        stems.append(("cloze", fact["cloze_stem"]))
    stems += [(f"paraphrase{i + 1}", p.format(s=fact["subject"]))
              for i, p in enumerate(rel.get("paraphrases", []))]

    forms = {}
    for name, stem in stems:
        text = model.greedy(prompts.render(stem)[0], max_new)
        forms[name] = {"stem": stem, "answer": text, "cluster": cluster(text)}

    q_prompt = prompts.render(fact["question"])[0]
    samples = model.sample(q_prompt, bcfg.get("samples", 10), max_new,
                           bcfg.get("temperature", 1.0))
    sc = [cluster(s) for s in samples]
    counts = Counter(sc)
    top = max(counts.values())
    tied = [c for c, n in counts.items() if n == top]
    greedy_q = forms["question"]["cluster"]
    modal = greedy_q if greedy_q in tied else sorted(tied)[0]
    sample_agree = top / len(sc)
    forms_agree = sum(f["cluster"] == modal for f in forms.values())

    in_pool = modal in pool
    a = pool[modal]["name"] if in_pool else None
    lp = lambda ans: model.logp_continuation(q_prompt, prompts.continuation(ans))
    return {
        "fact_id": fact["fact_id"],
        "state": state_of(modal, sample_agree, forms_agree, len(forms), bcfg),
        "a": a, "a_qid": modal if in_pool else None,
        "a_aliases": pool[modal]["aliases"] if in_pool else [],
        "a_equals_gold": bool(in_pool and modal == fact["gold_qid"]),
        "modal_cluster": modal,
        "sample_agree": sample_agree, "forms_agree": forms_agree,
        "n_forms": len(forms),
        "entropy_q": entropy_bits(counts.values()),
        "p_a_q": math.exp(lp(a)) if a else None,
        "p_gold_q": math.exp(lp(fact["gold"])),
        # entities M produced at least once: never offered as a "contradicting" b
        "sampled_qids": sorted({c for c in sc + [f["cluster"] for f in forms.values()]
                                if c in pool}),
        "forms": forms, "samples": samples, "sample_clusters": sc}


def run(cfg, model, facts, pools, limit=None):
    out = model_dir(cfg) / "beliefs.jsonl"
    done = {b["fact_id"] for b in read_jsonl(out)}
    todo = [f for f in select_facts(facts, limit, cfg.get("seed", 0))
            if f["fact_id"] not in done]
    rels = load_relations()
    bcfg = cfg.get("beliefs", {})
    print(f"[beliefs] {len(done)} done, {len(todo)} to elicit -> {out}")
    w = Appender(out)
    for i, f in enumerate(todo):
        w.add(elicit(model, f, pools[f["relation"]], rels[f["relation"]], bcfg))
        if (i + 1) % 50 == 0:
            print(f"[beliefs] {i + 1}/{len(todo)}")
    w.close()
    beliefs = read_jsonl(out)
    print("[beliefs] states:", dict(Counter(b["state"] for b in beliefs)))
    return beliefs
