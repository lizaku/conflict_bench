# conflict_bench

Unified harness for comparing **mechanistic** and **output-based** methods for
detecting and steering context-vs-parametric knowledge arbitration, in the
spirit of AxBench (Wu et al. 2025) but specialized to knowledge conflict and
extended beyond representation methods.

## Two axes (AxBench analogue)

| AxBench | Here |
|---|---|
| C concept detection | **Detection**: score how strongly the model is being pulled toward the contextual (counterfactual) answer |
| S model steering | **Steering**: intervene to push arbitration toward context or memory; measure flip rate + Δ format-corrected margin + fluency |

Shared DV everywhere: the within-item teacher-forced log-prob margin,
`m = logp(true) − logp(cf)` (positive = parametric wins), format-corrected via
the R condition (v3 semantics).

## The N/S/C/R design actually runs

All four conditions are scored for every item, as their own pass, before any
method runs (`core/margins.py`, `run.run_conditions`):

| | passage | what it isolates |
|---|---|---|
| **N** | none | what the model believes unprompted |
| **S** | supporting | passage agrees with the parametric answer |
| **C** | conflicting | the counterfactual passage — the arbitration condition |
| **R** | random | irrelevant passage of the same shape: the **format baseline** |

R exists because a bare margin under C confounds two things — how much the
counterfactual passage moved the model, and how the passage *template* moved
the two answer strings relative to each other before any conflict was
involved. R has the second effect and not the first, so
`corrected(C) = margin(C) − margin(R)`.

**Both numbers are reported everywhere, never one instead of the other.**
`condition_margins.csv` is the per-item design matrix (log-probs and margins
under every condition, raw and corrected); `conditions_summary.csv` is the
per-condition sanity table. Detectors carry `score` and `score_raw`, and
`detection_summary.csv` carries `auroc`, `auroc_raw` and
`auroc_correction_gain`. Steering records carry `margin_before/after`,
`margin_before/after_raw`, `r_offset` and both `flipped` and `flipped_raw`.

This is not decoration. In the CPU smoke run the raw margin correlated
**0.99997** with the R margin — i.e. the raw DV was almost entirely the
per-item answer-string/format constant, and the corrected margin was two
orders of magnitude smaller than the raw one (sd 0.19 vs 21.4). A conclusion
that survives only one of the two columns is a conclusion about the passage
template.

Because the R offset is a per-item constant it cancels out of Δmargin
exactly — so `d_margin` is the same either way, and there is deliberately no
`d_margin_raw` column pretending otherwise. What the correction changes is
which side of zero an item starts and ends on: the **flip rate**. A gap
between `flip_rate` and `flip_rate_raw` means flips are being manufactured
(or hidden) by the template rather than by the intervention.

Config keys: `conditions`, `reference_condition`, `detection_condition`,
`steering_condition`. Drop R from `conditions` and the run proceeds
uncorrected and says so, rather than silently correcting against a condition
it never scored.

One method opts out of the shared correction on purpose: CAD's DV is a
*first-token logit* margin, so the table's full-continuation R offset is the
wrong quantity to subtract, and it computes its own R baseline in its own
units.

## Where the probes read, and the readout control

A probe read at the **last prompt token** and scored against the model's own
greedy continuation *from that same prompt* can succeed for two very different
reasons: the residual stream represents the arbitration, or it already
contains the answer and the probe is decoding it. AUROC alone cannot tell
those apart, and the second is nearly guaranteed to score well. The old
`position: -1` default put every probe squarely in that regime.

Two changes make this measurable rather than assumed.

**1. Named positions, resolved per item** (`core/positions.py`). A raw integer
is not a probe site — token index 137 is a different linguistic location on
every item once passages differ in length. Positions are resolved against the
prompt's block spans via the tokenizer's offset mapping:

| position | where | predictive? |
|---|---|---|
| `end_of_context` | last token of the passage — conflict read, no answer formulated | yes |
| `answer_mention_last` | where the passage states the counterfactual answer | yes |
| `subject_last` | last token of the subject in the question | yes |
| `end_of_stem` / `last` | the model is about to emit its answer | **no — readout** |

`detection_positions` sweeps them; the first entry is the primary cell in
`triplet_report.csv`. `last` stays in the sweep on purpose, so the
readout-vs-upstream gap is a number in the table rather than an assumption.
Rows are flagged `is_readout_position`.

**2. `logit_lens` — a mandatory readout control.** Whatever a probe can read
at (layer, position), the model's own unembedding can read there too. The
`logit_lens` detector *is* that readout: the logit-lens margin between the
counterfactual and true answers' first tokens, at exactly the layer and
position the probe uses. `detection_summary.csv` carries `readout_auroc` and
**`auroc_over_readout`** for every probe — how much it beats the model's own
decoding at the identical site. This is to the decoding confound what the
base-rate AUROC is to the relation confound, and it is reported the same way.

Near-zero `auroc_over_readout` at `last` is the expected, uninteresting
result. A positive `auroc_over_readout` at `end_of_context` is the claim
worth making.

Because AUROC at two positions is only comparable over the same items, the
whole detection table runs on the intersection of items resolvable at every
swept position; what is dropped (an answer with no verbatim mention in its
passage, a subject absent from its question) goes to `detection_dropped.json`
with reasons. A position that does not exist for a condition — `end_of_context`
under N — skips that row with an explicit message instead of failing.

`detection_conditions` additionally allows scoring under a condition other
than the one that produced the labels; labels are behavioural under C, so
`[N, S, C]` asks whether the pre-conflict state already predicts how an item
will be arbitrated. It defaults to `[C]` because that is a different and much
stronger hypothesis, not a drop-in replacement.

## Prompt format

One source of truth, `core/prompts.py`, configured from the `prompt:` block —
never hard-coded in a method, or the four conditions stop being comparable and
the R correction stops correcting for anything:

```
N        ->  {question}
S / C / R ->  Background: {passage}\n\n{question}
```

`style: cloze` switches the stem to the cloze template (`"The head of
government of Monaco is"`) for the 5132/6000 items whose blank is terminal,
falling back to the question form for the rest. A steering instruction
(`prompt_instruct`) is inserted as its own block between background and
question.

## Methods

### Detection
| method | class | access | code |
|---|---|---|---|
| margin baseline | grey-box | logits | this repo |
| CK-PLUG confidence gain | grey-box | logits | https://github.com/byronBBL/CK-PLUG |
| p(True) / verbalized | black-box | prompting | this repo |
| **BOW (TF-IDF)** *(baseline)* | black-box | text only | this repo — the null a probe must beat |
| context shift (KL / JSD) | grey-box | logits | this repo — CAD / AdaCAD detection twins |
| DiffMean projection | white-box | activations | AxBench: https://github.com/stanfordnlp/axbench |
| logit-lens readout **(control)** | white-box | activations | this repo — the probe's mandatory partner |
| linear probe (GroupKFold) | white-box | activations | this repo (v3 port) |
| tuned-lens trajectory features | white-box | activations | https://github.com/AlignmentResearch/tuned-lens |
| semantic entropy | black-box | K samples | https://github.com/jlko/semantic_uncertainty (Nature version) |
| semantic entropy probes | white-box | activations | https://github.com/OATML/semantic-entropy-probes |
| SelfCheckGPT-NLI | black-box | K samples | https://github.com/potsawee/selfcheckgpt (`pip install selfcheckgpt`) |

### Steering
| method | class | code |
|---|---|---|
| prompt instruction (baseline) | black-box | this repo |
| CAA / DiffMean act-add, L17-19, orthogonalized | white-box | https://github.com/nrimsky/CAA — vectors fitted in-run |
| SAE feature steering (optional) | white-box | Llama-Scope SAEs (check availability for Llama-3-8B base vs 3.1) |
| CAD contrastive decoding | grey-box | https://github.com/xhan77/context-aware-decoding |
| **AdaCAD** adaptive contrastive decoding | grey-box | Wang et al. 2024, arXiv:2409.07394 |
| plain CAA (`caa`, no orthogonalization) | white-box | https://github.com/nrimsky/CAA |
| CK-PLUG decoding | grey-box | https://github.com/byronBBL/CK-PLUG |
| CSKS proxy models | grey-box | https://github.com/OliveJuiceLin/CSKS |

### CAA vectors, fitted per run
`act_add` builds its own vectors from the behavioural contrast on the training
relations — no persisted v3 artefact required:

```
v_override[l] = mean(h_C[l] | followed context) − mean(h_C[l] | kept parametric)
v_use[l]      = mean(h_S[l]) − mean(h_N[l])
```

`v_override` is orthogonalized against `v_use` so the intervention is the
arbitration direction and not "a document exists" — the component the v3
postmortem blamed for the wrong-sign result (cos = −0.39 at L25); the realised
cosine is written into every vector's metadata. α is expressed in units of the
layer's mean residual norm (`scale: residual`) so a factor sweep means the same
thing after a model swap. Point `vectors_path` at a saved `*vectors.pt` to
reuse vectors instead of refitting.

## Metrics (non-negotiable set)
0. **Every margin**: reported raw *and* R-corrected, never one alone
   (`core/margins.py`). See the N/S/C/R section above.
1. **Detection**: GroupKFold-by-relation AUROC, always paired with the
   relation-base-rate AUROC control (`metrics.grouped_auroc`); for methods
   with a format-corrected score, with `auroc_raw` beside it; and for every
   activation-reading method, with the `logit_lens` readout control at the
   same layer and position (`auroc_over_readout`). A probe that does not beat
   the readout has not been shown to detect anything.
2. **Steering**: flip rate + Δmargin, always vs matched control
   (norm-matched random direction for act-add; α=0 for decoding methods);
   dose–response sweep over factor → Pareto (flip rate vs fluency),
   the analogue of AxBench Fig. 4.
3. **Triplet report** per method: probe AUROC | causal effect vs control |
   prompting-baseline effect (`metrics.triplet_report`).

Fitted steerers get the same fold discipline as probes: CAA vectors are built
on training relations and applied only to held-out ones, so the steering
numbers are not reporting a vector's fit to the items it steers.

## Datasets (start → later)
1. **`data/ConfiQA_cloze_clustered/`** — primary. 40 relation files
   (`P6.json`, `P17.json`, …), 6000 records, uniform schema:
   `question`, `cloze_template`, `orig_answer`/`cf_answer`,
   `orig_alias`/`cf_alias`, `orig_context`/`cf_context` (the S and C passages),
   `*_context_piece` (the single carrier sentence), `*_path_labeled`
   (subject is parsed from here), `*_statement`.
   Two things the loader supplies that the files do not: the **subject**, and
   the **R passage** — no irrelevant context ships with the data, so each item
   deterministically borrows a paragraph from a different relation. R is the
   format baseline the margin is corrected against, so it has to be a real
   passage of the same shape, not an empty string.
2. ConFiQA-QA raw **with QC filters** (`load_confiqa` logs everything dropped —
   that log feeds the benchmark-audit paper).
3. NQ-Swap (Longpre et al. 2021) — real-QA entity substitution.
4. DynamicQA (static/temporal/disputable conflict partitions) — used by CSKS.
5. CSKS synthetic sensitivity data — for cross-benchmark comparison of
   steering dose–response.

First-token collisions are **recomputed per tokenizer**
(`data.first_token_collisions`) rather than read from a static id list, because
which items collide depends on the model; CAD scores a first-token margin, so
leaving them in hands that method a structurally zero DV.

## Outputs

Nothing is reported as a bare number: every fitted object is written next to
the summary that quotes it.

```
out_dir/
  manifest.json            config snapshot, model description, timings, QC counts
  labels.csv               behavioural label + the generation behind each
  qc_dropped.json          everything dropped, with reasons
  condition_margins.csv    the N/S/C/R design matrix per item, raw + corrected
  conditions_summary.csv   per-condition means / context-win rates, raw + corrected
  margin_table.jsonl       cached teacher-forced scores behind both (resumable)
  detection_records.jsonl  per-item scores with extras (log-probs, samples, fold)
  detection_dropped.json   items with no resolvable probe position, with reasons
  steering_records.jsonl   per-item per-factor margins, generations, controls
  detection_summary.csv    grouped AUROC + base-rate control
  steering_summary.csv     flip rate (all / flippable), Δmargin, specific effect
  triplet_report.csv       the mandatory joint table
  activations/layer{L}.npz residual-stream tensors, shared + resumable
  artifacts/
    linear_probe/fold{k}_coef.npy, fold{k}_meta.json
    diffmean_proj/fold{k}_w.npy
    act_add/fold{k}_vectors.pt, fold{k}_v_override_L{l}.npy, fold{k}_v_use_L{l}.npy
    <method>/scores.jsonl | records.jsonl
```

The activation cache is keyed by `(item, condition, layer, position)` and
shared by `linear_probe`, `diffmean_proj` and `act_add`'s vector fit — so the
residual stream is extracted once, and a re-run against the same `out_dir`
picks it up instead of re-extracting.

## Phasing
- **Phase 1 (runs today)**: margin, confidence_gain, p_true, diffmean_proj,
  linear_probe, semantic_entropy + prompt_instruct, cad, act_add.
- **Phase 2**: semantic_entropy with real NLI clustering (the K samples are
  already persisted, so it can be re-clustered without re-sampling),
  SelfCheckGPT-NLI, CK-PLUG (vendor their decode fn), tuned-lens features.
- **Phase 3**: CSKS (train proxies), ReFT-r1 via pyvene, SAE steering.

## Pilot experiment

Five methods plus the BOW baseline, as a script and as a notebook that call
the *same* stage functions so the two cannot drift:

```bash
python -m conflict_bench.experiments.run_pilot --limit 200
python -m conflict_bench.experiments.run_pilot --stages detection --force
python -m conflict_bench.experiments.run_pilot --model google/gemma-2-2b-it     --relations P17 P27 P36
```

`experiments/pilot_five.ipynb` is the interactive twin, with the dose-response
and flips-vs-fluency Pareto plots.

| | steering | detection twin | why that twin |
|---|---|---|---|
| prompting | `prompt_instruct` | `p_true` | both ask the model verbally |
| CAD | `cad` | `context_kl` | CAD amplifies this shift |
| AdaCAD | `adacad` | `context_jsd` | AdaCAD's adaptive alpha *is* this JSD |
| CK-PLUG | `ckplug` | `confidence_gain` | CK-PLUG's gate *is* this signal |
| CAA | `caa` / `act_add` | `diffmean_proj` | same mean-difference vector |
| **BOW** | — | `bow` | the text-only null |

Five checkpointed stages — `data`, `conditions`, `detection`, `steering`,
`report` — each skipped on a re-run if its output exists, so an interrupted
pilot resumes and a single stage can be re-run with `--force`. Outputs land in
`pilot_report.csv` / `pilot_report.md` on top of the standard run artifacts.

Two notes on the implementations:

- **CAD, AdaCAD and CK-PLUG share one contrastive core** and are scored on the
  full multi-token teacher-forced margin, not a first-token proxy, so they sit
  in the same table as the activation methods. All three reduce *exactly* to
  the unsteered margin at strength 0 (verified), which makes their matched
  control the real unsteered value rather than a separate quantity.
- **CK-PLUG is reimplemented from the mechanism its paper describes**, not
  vendored from `byronBBL/CK-PLUG`. Check it against their code before
  quoting numbers against theirs. Its `factor` is alpha in [0, 1], not a
  dose, so it carries its own factor list.

## Run
```bash
pip install torch transformers scikit-learn pyyaml pandas --break-system-packages
python -m conflict_bench.run --config conflict_bench/configs/default.yaml
```

The model is a config knob — either `model: <hf-id>` or a block with
`name` / `device` / `dtype` / `revision` / `cache_dir`. `device: auto` and
`dtype: auto` resolve to cuda+bf16 where available and cpu+fp32 otherwise, and
layer discovery covers llama / mistral / gemma / qwen / gpt-neox / gpt2, so
swapping the 8B Llama out is one line. Layer indices may be given as depth
fractions (`0.6` = 60% of the way up) so a swap does not silently move the
probe to a different part of a differently-sized network.

`limit:` samples evenly across relations rather than taking the first N —
taking the first N walks whole relation files in filename order and leaves
GroupKFold-by-relation with nothing to split.
