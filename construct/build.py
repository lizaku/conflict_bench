"""Build the benchmark for one model - five steps in a straight line.

    python -m conflict_bench.construct.build --model microsoft/Phi-3.5-mini-instruct
    python -m conflict_bench.construct.build --model <hf-id> --limit 300     # a pilot

    seed      ConfiQA records        -> bench/_seed/facts.jsonl      no model
    beliefs   facts + M              -> bench/<model>/beliefs.jsonl  GPU
    contexts  beliefs                -> bench/<model>/contexts.jsonl no model
    measure   contexts + M           -> bench/<model>/measured.jsonl GPU
    export    all of the above       -> instances.jsonl, public.jsonl, card.md

`--limit N` elicits beliefs for N seed facts spread over the relations
instead of all 6000; everything downstream follows from that.  The two GPU
steps append as they go and skip what is already on disk, so an interrupted
build is resumed by running the same command again.  `--steps` runs a subset,
e.g. `--steps contexts,export` after editing a recipe (no GPU needed).
"""
import argparse

from conflict_bench.construct import beliefs, contexts, export, measure, seed
from conflict_bench.construct.common import (load_config, load_model,
                                             model_dir, model_name, read_jsonl,
                                             utf8_stdout)

STEPS = ["seed", "beliefs", "contexts", "measure", "export"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="HF id or local path (default: config)")
    ap.add_argument("--config")
    ap.add_argument("--limit", type=int,
                    help="number of seed facts to elicit (default: all)")
    ap.add_argument("--measure-limit", type=int,
                    help="measure about this many instances, spread over "
                         "relations (default: all)")
    ap.add_argument("--steps", default=",".join(STEPS))
    args = ap.parse_args(argv)
    utf8_stdout()
    cfg = load_config(args.config, args.model)
    steps = [s.strip() for s in args.steps.split(",")]
    limit = args.limit or cfg.get("limit")

    facts, pools = seed.run(cfg) if "seed" in steps else seed.load(cfg)
    model = load_model(cfg) if {"beliefs", "measure"} & set(steps) else None
    if "beliefs" in steps:
        beliefs.run(cfg, model, facts, pools, limit)
    if "contexts" in steps:
        contexts.run(cfg, facts, read_jsonl(model_dir(cfg) / "beliefs.jsonl"),
                     pools, model_name(cfg))
    if "measure" in steps:
        measure.run(cfg, model, args.measure_limit)
    if "export" in steps:
        export.run(cfg)


if __name__ == "__main__":
    main()
