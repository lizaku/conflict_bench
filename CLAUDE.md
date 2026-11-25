# CLAUDE.md — conflict_bench

Working notes for agents. `README.md` is the human-facing description,
`PLAN.md` the design with a worked example, `DECISIONS.md` the record of every
choice made while implementing it. This file is the state of the code and the
invariants that must not be broken.

Last substantive update: 2026-11-25 (benchmark v2: per-model construction).
The previous design (fixed ConfiQA items under N/S/C/R, margin-flip steering)
is the git tag `milestone-1-nscr-pilot`; its CLAUDE.md is in that commit.

## What this is

A benchmark for knowledge conflicts between a model's memory and its context,
with two tasks — detection and steering — and a fixed contract so that method
families (contrastive decoding, representations, components) are written by
different people against the same inputs and outputs.

The authority for every definition is `LLM Conflicts Benchmark Breakdown.docx`.
Notation is its notation: `M`, `q`, `c` (sentences `s1…sn`), `a = M(q)`,
`b` (what `c` asserts), `d = M(q, c)`, `x` (steered output), `π` (direction).

A conflict is **relative to the model**: (1) `a` is stable and confident,
(2) `c` asserts `b`, (3) `a` and `b` cannot both be true. So the dataset is a
pipeline run once per model, not a fixed file.

## Layout

```
construct/          benchmark construction - five steps, one file each
  relations.yaml    per-relation metadata: single_valued, paraphrases, np
  match.py          THE answer matcher (word boundary, min 3 chars)
  seed.py           1. ConfiQA records -> facts.jsonl, pools.json   (no model)
  beliefs.py        2. closed-book elicitation -> beliefs.jsonl     (GPU)
  contexts.py       3. recipes -> contexts.jsonl; label()           (no model)
  measure.py        4. d, d0, confidence given q & c -> measured.jsonl (GPU)
  export.py         5. instances.jsonl, public.jsonl, card.md
  build.py          the straight-line driver / CLI
  common.py         paths, config, jsonl, spread(), load_model()
api.py              PublicInstance, Detection, Intervention, Detector, Steerer
harness.py          fit on train folds, score / steer held-out, write predictions
evaluate.py         predictions + instances -> tables, report.md
methods/            baselines.py (mine), contrastive.py, representations.py
core/               model.py, prompts.py, positions.py, activations.py, artifacts.py
configs/default.yaml   one config for build and harness
tests/smoke.py      end-to-end on CPU with a tiny random model
bench/              OUTPUT, gitignored: bench/_seed/, bench/<model>/
data/               the ConfiQA seed, gitignored
```

## Method inventory

**Detectors** (HIGHER = more conflict, always)

| name | file | notes |
|---|---|---|
| `bow` | baselines | text-only null, trainable |
| `p_true` | baselines | self-report |
| `with_without` | baselines | `a = M(q)` vs `d = M(q, c)` |
| `sampling_entropy` | baselines | entropy of sampled answers |
| `margin` | baselines | **oracle control**: log p(b) − log p(a) |
| `confidence_gain` | contrastive | CK-PLUG's signal, score = −CG |
| `context_jsd` / `context_kl` | contrastive | AdaCAD's / CAD's signal; `localize` = leave-one-sentence-out |
| `linear_probe` | representations | `localize` = same probe at each sentence end |
| `diffmean_proj` | representations | |
| `logit_lens` | representations | **oracle readout control** for probes |

**Steerers**

| name | file | directions | notes |
|---|---|---|---|
| `prompt_instruct` | baselines | ctx, mem, span | strength = variant index; placebo control |
| `delete_context` | baselines | mem, span | input oracle |
| `cad` | contrastive | ctx, mem, span | `(1+a)·z_ctx − a·z_par` |
| `adacad` | contrastive | ctx, mem, span | per-token JSD weight |
| `ckplug` | contrastive | ctx, mem, span | entropy-gated; `(1−2a)` coefficient — see its docstring |
| `caa` | representations | ctx, mem | plain mean-difference |
| `act_add` | representations | ctx, mem | orthogonalized against the "context present" direction |

Not built: the entailment reference baseline, the component family (student
topic), anything fine-tuned.

## Invariants — do not quietly remove these

1. **`label()` in `construct/contexts.py` is the only place a label is
   assigned.** Item labels (`conflict_q`, `conflict_global`,
   `chunk_asserts_b`) are derived in `emit` from per-fact and per-sentence
   labels. `tests/smoke.py::check_instances` asserts this.
2. **A method never sees a label or an answer string.** `score` and
   `intervene` get a `PublicInstance`. `fit` gets `harness.labels_for`, which
   holds labels but no `a` / `b`. The two `oracle = True` controls are the
   only exception and are flagged as such in every table.
3. **The harness generates, not the method.** A steerer returns an
   `Intervention`; `harness.apply` runs `ModelWrapper.greedy`. `d` (in
   `measure.py`) and every `x` come from that same loop, never from HF
   `generate` (which applies the model's generation config).
4. **No model writes or checks a context.** Every sentence is
   `statement(fact, value)`: a seed cloze template with a value filled in.
5. **Fillers are facts M knows, stated with M's answer.** A filler from
   Wikidata could contradict M and create an unlabelled global conflict.
6. **Instances of one fact differ only in the carrier sentence.** Same
   question, same fillers. This is what makes `con` vs `sup` a paired
   contrast. Asserted in the smoke test.
7. **Only single-valued relations carry instances** (`relations.yaml`).
   For a multi-valued relation a swap is not a contradiction.
8. **Folds are whole relations**, for detectors and steerers alike. A CAA
   vector fitted on the items it steers is the same leak as a probe scored on
   its training relation.
9. **Uncertain facts are never conflicts, never training data, and never in a
   main table.** They go to `*_uncertain.csv`, scored with `*_if_known` labels.
10. **Detection AUROC against each negative recipe separately**, not only
    pooled. `con` vs `nov` is the benchmark's point: same text shape, and only
    M's knowledge tells them apart.
11. **`bow` beside every detector**, and the oracle `logit_lens` at the same
    layer and position as every probe.
12. **Every success rate with its denominator** (`n_d_ne_t`), every
    collateral rate with its (`n_d_eq_t`). Both directions are their own rows.
13. **A matched control for every steering effect**: random direction of
    equal norm (activation methods), placebo instruction (prompting), the
    unsteered `d` itself (decoding family).
14. **Prompts come only from `core/prompts.render`.** One exception: `p_true`
    asks a different question, and still goes through `prompts.wrap_chat`.
15. **The chat template is applied in `core/prompts.py`, never in
    `core/model.py`** — positions are resolved from character spans, and a
    template applied later would shift them.
16. **Continuations come from `prompts.continuation()`, never `" " + answer`.**
17. **One tokenisation path: `ModelWrapper.encode`.** It skips the tokenizer's
    BOS when the chat template already wrote one. The old code tokenised the
    templated text with special tokens on, i.e. a double BOS on Llama/Gemma.
18. **Answers are matched only through `construct/match.py`.**

## Conventions

- Recipes are the taxonomy cells; the list is `contexts.ALL_RECIPES` and the
  config's `contexts.recipes`. Core: `sup con irr nov`.
- An instance: `passages[].chunks[]` (each with `text, role, fact_id, asserts,
  label`), `path[]` (one entry per fact on the answer path, with the three
  condition booleans), `answers` ({display name: normalised names} of every
  candidate answer), `targets` ({ctx, mem, span}), `labels`, `features`.
- `group` = the last-hop fact id; all instances of a fact share it and its fold.
- Steering: `success = #{d≠t ∧ x=t} / #{d≠t}`, `collateral = #{d=t ∧ x≠t} /
  #{d=t}`; "x = t" is `match.hit`: x names t and no other candidate.
- `π_mem` target: `a` for known facts, the context's answer for unknown ones.
- Layers accept depth fractions (`0.6`).
- Probe positions are named: `last`, `end_of_stem`, `end_of_context`,
  `chunk:i`. There is no "answer mention" position — that would be a label.
- Cost = forward passes of the network, counted by a hook in `ModelWrapper`.

## Running

```bash
# from the parent of this folder
python -m conflict_bench.construct.build --model microsoft/Phi-3.5-mini-instruct [--limit 300]
python -m conflict_bench.harness --model microsoft/Phi-3.5-mini-instruct --limit 100
python -m conflict_bench.evaluate conflict_bench/bench/<model>/runs/<name>   # re-score only
python -m conflict_bench.tests.smoke
```

`build.py` and `harness.py` are deliberately straight lines. Keep them so.

## Testing without a GPU

The dev machine has no GPU and too little RAM for a real model.
`tests/smoke.py` builds a tiny random Llama with a cached tokenizer and runs
every step. A random model knows nothing, so it fakes exactly two things:
the belief states (after running the real elicitation code on a few facts)
and `d` (so a behavioural contrast exists for CAA). It also builds every
recipe on the whole seed with synthetic beliefs — model-free, about 10 s —
and asserts the invariants. Run it after any change to `construct/`.

The tiny model costs ~40 ms per forward pass on this machine because the
vocabulary is real; keep `--measure` small.

## Known gaps and caveats

- **Nothing has been run on a real model yet.** The thresholds in
  `beliefs:` and the abstention regex are untested against real answers;
  read `card.md` after the first build before trusting any number.
- Greedy decoding and teacher-forcing under an intervention re-run the full
  prompt for every token (no KV cache). Simple and identical for all methods,
  but the steering loop is the slow part of a run.
- Cloze prompts go through the chat template too; an instruct model may
  restate the sentence instead of continuing it, which lowers the
  teacher-forced probability of the bare answer in that format.
- A cached activation is free on a re-run, so `forward_passes` of the probes
  is only right on a cold cache.
- CK-PLUG is reimplemented from the paper's equations, not vendored. AdaCAD's
  push toward memory extrapolates past the paper.
- See `DECISIONS.md` § "Not built yet".
