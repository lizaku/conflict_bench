"""End-to-end smoke test on CPU, in a couple of minutes, with no download.

    python -m conflict_bench.tests.smoke [--tokenizer <cached hf id>] [--keep]

Builds a tiny randomly-initialised Llama (real architecture, a real cached
tokenizer with a chat template, 4 layers, hidden 64) and pushes it through
every step: seed -> beliefs -> contexts -> measure -> export -> harness ->
evaluate.  Real code paths, meaningless numbers.

A random model knows nothing, so two things are faked, and only these two:
  * beliefs: after the real elicitation code has run on a few facts, the
    belief file is replaced by synthetic states (known / uncertain / unknown);
  * d: the measured default answer is overwritten to follow the context or
    memory at random, so the behavioural contrast CAA is fitted on exists.

Then it checks the invariants of the construction (see `check_instances`).
"""
import argparse
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from conflict_bench import harness
from conflict_bench.construct import beliefs, contexts, export, measure, seed
from conflict_bench.construct.common import (load_config, load_model, model_dir,
                                             read_jsonl, write_jsonl)


def tiny_model(tokenizer_name, path):
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    import torch
    tok = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4,
                      max_position_embeddings=1024,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    LlamaForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)


def synthetic_beliefs(facts, pools, n, seed_):
    rng = random.Random(seed_)
    out = []
    for f in beliefs.select_facts(facts, n, seed_):
        pool = pools[f["relation"]]
        u = rng.random()
        state = "known" if u < 0.6 else "uncertain" if u < 0.75 else "unknown"
        qid = f["gold_qid"] if rng.random() < 0.85 else rng.choice(sorted(pool))
        has_a = state != "unknown" and qid in pool
        out.append({
            "fact_id": f["fact_id"], "state": state,
            "a": pool[qid]["name"] if has_a else None,
            "a_qid": qid if has_a else None,
            "a_aliases": pool[qid]["aliases"] if has_a else [],
            "a_equals_gold": has_a and qid == f["gold_qid"],
            "sample_agree": {"known": 1.0, "uncertain": 0.5, "unknown": 0.2}[state],
            "forms_agree": 4, "n_forms": 4,
            "entropy_q": {"known": 0.0, "uncertain": 1.5, "unknown": 2.6}[state],
            "p_a_q": 0.9 if has_a else None, "p_gold_q": 0.5,
            "sampled_qids": [qid] if has_a else [], "synthetic": True})
    return out


def check_instances(recs, public):
    """The properties the construction promises."""
    assert recs, "no instances were built"
    allowed = {"instance_id", "fold", "format", "question", "passages"}
    for p in public:
        assert set(p) == allowed, f"public record leaks {set(p) - allowed}"
        assert all(isinstance(c, str) for ps in p["passages"] for c in ps["chunks"])
    by_group = defaultdict(dict)
    for r in recs:
        chunks = [c for p in r["passages"] for c in p["chunks"]]
        lab, path = r["labels"], r["path"]
        # item labels are derived from the per-fact / per-sentence labels
        assert lab["conflict_q"] == int(any(p["label"] == "contradicts" for p in path))
        assert lab["conflict_global"] == int(any(c["label"] == "contradicts" for c in chunks))
        assert lab["conflict_global"] >= lab["conflict_q"]
        assert len(lab["chunk_asserts_b"]) == len(chunks)
        for p in path:
            # the three conditions decide the label, in this order
            want = ("irrelevant" if not p["gives_answer"] else
                    "novel" if not p["known"] else
                    "supports" if p["compatible"] else "contradicts")
            assert p["label"] == want, (r["recipe"], p)
            assert p["known"] == (p["state"] == "known")
        if r["has_uncertain"]:
            assert lab["conflict_q"] == 0     # uncertain is never a conflict
        t = r["targets"]
        if lab["conflict_q"]:
            assert t["mem"] and (t.get("ctx") != t["mem"]), r["recipe"]
        if r["recipe"] in ("sup", "extra-true", "off-path", "unrelated", "2hop-sup-sup"):
            assert t["ctx"] == t["mem"]
        if r["recipe"] in ("off-path", "unrelated"):
            assert (lab["conflict_q"], lab["conflict_global"]) == (0, 1)
        if r["recipe"] == "nov":
            assert r["state"] == "unknown" and not lab["conflict_global"]
        for name in [v for v in (t.get("ctx"), t.get("mem")) if v]:
            assert name in r["answers"], (r["recipe"], name)
        by_group[(r["group"], r["format"])][r["recipe"]] = r
    n_pairs = 0
    for g in by_group.values():
        if "sup" in g and "con" in g:
            n_pairs += 1
            s = [c["text"] for c in g["sup"]["passages"][0]["chunks"]]
            c = [c["text"] for c in g["con"]["passages"][0]["chunks"]]
            # a pair shares the question and differs in exactly one sentence
            assert g["sup"]["question"] == g["con"]["question"]
            assert sum(a != b for a, b in zip(s, c)) == 1 and len(s) == len(c)
    assert n_pairs, "no sup/con pair was built"
    return n_pairs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct",
                    help="any locally cached tokenizer with a chat template")
    ap.add_argument("--facts", type=int, default=400)
    ap.add_argument("--measure", type=int, default=160,
                    help="instances pushed through the model")
    ap.add_argument("--keep", action="store_true", help="keep the temp folder")
    args = ap.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="conflict_bench_smoke_"))
    try:
        tiny = tmp / "tiny-random-llama"
        tiny_model(args.tokenizer, tiny)
        cfg = load_config(model=str(tiny))
        cfg["bench_dir"] = str(tmp / "bench")
        cfg["beliefs"].update(samples=3, max_new_tokens=4)
        cfg["measure"].update(samples=2, max_new_tokens=3)
        cfg["steering"]["max_new_tokens"] = 3
        # one strength per steerer keeps the CPU run short
        cfg["steerers"] = [{"name": s, "strengths": [1.0]} for s in cfg["steerers"]]
        cfg_path = tmp / "smoke.yaml"
        import yaml
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        d = model_dir(cfg)

        facts, pools = seed.run(cfg)

        # model-free: every recipe on the WHOLE seed, with synthetic beliefs
        full, _ = contexts.Builder(facts, synthetic_beliefs(facts, pools, None, 1),
                                   pools, cfg, "synthetic").build()
        from conflict_bench.api import PublicInstance
        n_pairs = check_instances(
            full, [PublicInstance.from_record(r).public_record() for r in full])
        counts = Counter(r["recipe"] for r in full)
        missing = set(contexts.ALL_RECIPES) - set(counts)
        print(f"[smoke] whole seed: {len(full)} instances, {n_pairs} sup/con "
              f"pairs, invariants hold; per recipe {dict(sorted(counts.items()))}")
        assert not missing, f"recipes that built nothing: {missing}"
        assert len({r['instance_id'] for r in full}) == len(full)

        model = load_model(cfg)
        real = beliefs.run(cfg, model, facts, pools, limit=6)   # the real code path
        assert len(real) == 6 and all("forms" in b for b in real)

        write_jsonl(d / "beliefs.jsonl",
                    synthetic_beliefs(facts, pools, args.facts, 0))
        built = contexts.run(cfg, facts, read_jsonl(d / "beliefs.jsonl"), pools,
                             str(tiny))
        measure.run(cfg, model, limit=args.measure)

        rng = random.Random(0)
        measured = read_jsonl(d / "measured.jsonl")
        targets = {r["instance_id"]: r["targets"] for r in built}
        for m in measured:                        # fake a default behaviour
            t = targets[m["instance_id"]]
            pick = t.get(rng.choice(["ctx", "mem"])) or t.get("mem")
            if pick:
                m["d"] = {"text": pick, "matches": [pick]}
        write_jsonl(d / "measured.jsonl", measured)

        recs = export.run(cfg)
        n_pairs = check_instances(recs, read_jsonl(d / "public.jsonl"))
        print(f"[smoke] invariants hold on {len(recs)} instances "
              f"({n_pairs} sup/con pairs)")
        del model

        common = ["--config", str(cfg_path)]
        harness.main(common + ["--limit", "100", "--name", "pilot100"])
        harness.main(common + ["--recipes", "all", "--name", "all_recipes"])
        for run in ("pilot100", "all_recipes"):
            out = d / "runs" / run
            import json
            man = json.loads((out / "manifest.json").read_text())
            print(f"\n[smoke] {run}: {man['n_instances']} instances, "
                  f"recipes {man['recipes']}")
            for s in man["skipped"]:
                print("   skipped:", s)
            for t in ("detection", "localization", "steering",
                      "detection_uncertain", "steering_uncertain"):
                print(f"   {t + '.csv':26s}",
                      "ok" if (out / f"{t}.csv").exists() else "MISSING")
            assert (out / "detection.csv").exists()
            assert (out / "steering.csv").exists()
            failed = [s for s in man["skipped"] if "failed" in s]
            assert not failed, failed
        print("\n[smoke] PASSED")
        if args.keep:
            print("[smoke] kept:", tmp)
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
