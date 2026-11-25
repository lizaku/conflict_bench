"""Step 4 - measure.  contexts.jsonl + M -> measured.jsonl   (GPU)

The model's own behaviour on every instance, with no method applied:

    d            = M(q, c), greedy, and which candidate answers it names
    d0           = M(q) for the same question (the closed-book answer)
    entropy_qc   entropy in bits over K sampled answers given q & c
    logp_qc      teacher-forced log-prob of each candidate answer given q & c
    logp_q       the same given q alone

Nothing here is a gate.  Whether M follows the context is the phenomenon, so
no instance is dropped for its `d`.  `d` is what the steering formulas are
conditioned on, and the confidence numbers are the two confidence rows of the
taxonomy measured "given q & c".
"""
from collections import Counter

from conflict_bench.api import PublicInstance
from conflict_bench.core import prompts
from conflict_bench.construct import match
from conflict_bench.construct.beliefs import entropy_bits
from conflict_bench.construct.common import (Appender, model_dir,
                                             read_jsonl, spread)


def measure(model, rec, mcfg) -> dict:
    inst = PublicInstance.from_record(rec)
    answers = rec["answers"]
    max_new = mcfg.get("max_new_tokens", 12)
    p_ctx, p_no = inst.prompt, inst.prompt_no_context

    d = model.greedy(p_ctx, max_new)
    d0 = model.greedy(p_no, max_new)
    out = {"instance_id": rec["instance_id"],
           "d": {"text": d, "matches": match.which(d, answers)},
           "d0": {"text": d0, "matches": match.which(d0, answers)},
           "logp_qc": {}, "logp_q": {}}
    for name in answers:
        cont = prompts.continuation(name)
        out["logp_qc"][name] = model.logp_continuation(p_ctx, cont)
        out["logp_q"][name] = model.logp_continuation(p_no, cont)

    k = mcfg.get("samples", 10)
    if k:
        samples = model.sample(p_ctx, k, max_new, mcfg.get("temperature", 1.0))
        clusters = ["+".join(match.which(s, answers))
                    or "other:" + " ".join(match.norm(s).split()[:3])
                    for s in samples]
        out["entropy_qc"] = entropy_bits(Counter(clusters).values())
        out["samples_qc"] = samples
    return out


def run(cfg, model, limit=None):
    d = model_dir(cfg)
    contexts = read_jsonl(d / "contexts.jsonl")
    path = d / "measured.jsonl"
    done = {m["instance_id"] for m in read_jsonl(path)}
    live = {c["instance_id"] for c in contexts}
    if limit:       # a pilot: whole facts, spread over relations
        contexts = spread(contexts, limit, cfg.get("seed", 0))
    todo = [c for c in contexts if c["instance_id"] not in done]
    print(f"[measure] {len(done & live)} done, {len(todo)} to measure -> {path}")
    w = Appender(path)
    for i, rec in enumerate(todo):
        w.add(measure(model, rec, cfg.get("measure", {})))
        if (i + 1) % 100 == 0:
            print(f"[measure] {i + 1}/{len(todo)}")
    w.close()
