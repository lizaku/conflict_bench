# CLAUDE.md — conflict_bench

Working notes for agents. `README.md` is the human-facing description; this
file is the state of the code, the invariants that must not be broken, and the
reasoning behind the recent changes so they are not undone by accident.

Last substantive update: 2026-09-22.

## What this is

A research harness comparing **mechanistic** (activation-level) against
**output-based** (prompting / decoding / sampling) methods for detecting and
steering context-vs-parametric knowledge arbitration. AxBench-shaped (a
detection axis and a steering axis), specialised to knowledge conflict.

Runs today, end to end. Not a skeleton.

## The two questions

Both axes ask one question each, over the same items under the same
conditions, so that a detector and a steerer are comparable:

**Detection** — *"Does this passage contradict the model's internal
knowledge?"* Every item is scored **twice**: once with its supporting passage
(label 0) and once with its conflicting one (label 1). This is the S-vs-C
contrast, and it is the question CK-PLUG's confidence gain was actually
defined for.

**Steering** — *"Given a conflicting passage, can the intervention make the
model output the conflicting answer when it otherwise would not?"* Every
steerer runs under **both** C and S: C is the measurement, S is the matched
specificity control — same item, same dose, nothing to arbitrate.

Three things follow from the paired S-vs-C design, and they are the reason it
replaced the old framing:

1. The base rate is **exactly 0.5** by construction, so the relation
   base-rate control has nothing left to exploit.
2. Both instances share the item, the question, the template and both answer
   strings, so the per-item answer-string constant **cancels exactly**. That
   is what the R correction was trying to achieve by subtraction — see "The R
   correction" below for why subtraction failed.
3. Both halves of a pair carry the same relation, so GroupKFold keeps them in
   the same fold automatically. A pair is never split across train/test.

The old framing — *"will the model follow the context?"*, scored under C alone
against a behavioural label — is still runnable as `detection_task:
arbitration`, because the two questions are genuinely different and the
earlier numbers should stay reproducible.

## Layout

```
core/
  types.py        Item, Condition (N/S/C/R), ScoredCandidates, Detection/SteeringRecord
  prompts.py      PromptSpec - THE single source of prompt truth; build_with_spans
  positions.py    named probe positions resolved per item via offset mapping
  margins.py      MarginTable - the N/S/C/R teacher-forced margins, shared + cached
  activations.py  ActivationCache - residual stream, shared + on-disk + resumable
  artifacts.py    ArtifactStore - where fitted params and tensors go
  model.py        ModelWrapper - the only thing that touches the HF model
data.py           loaders; first_token_collisions()
metrics.py        grouped_auroc, steering_summary, triplet_report
run.py            the generic runner (5 stages inline)
methods/          base.py registry + detectors/ + steerers/
configs/          default.yaml, pilot_five.yaml
experiments/      run_pilot.py (standalone driver) + pilot_five.ipynb
data/ConfiQA_cloze_clustered/   40 relation files, 6000 records
```

## Method inventory

**Detectors** (score convention: HIGHER = more context-following)

| name | status | notes |
|---|---|---|
| `margin` | ✅ | the DV itself, the floor; reports raw + R-corrected |
| `bow` | ✅ | TF-IDF baseline — the null a probe must beat |
| `p_true` | ✅ | prompting leg |
| `confidence_gain` | ✅ | CK-PLUG's signal |
| `context_kl` / `context_jsd` | ✅ | CAD / AdaCAD detection twins |
| `diffmean_proj` | ✅ | CAA's vector as a detector |
| `linear_probe` | ✅ | logistic on residual stream |
| `logit_lens` | ✅ | **readout control** — not a method, a control |
| `semantic_entropy` | ✅ | alias-clustering surrogate; K samples persisted |
| `selfcheck_nli` | ❌ | raises NotImplementedError |

**Steerers**

| name | status | notes |
|---|---|---|
| `prompt_instruct` | ✅ | mandatory baseline; factor = variant index |
| `cad` | ✅ | `z = (1+a)·z_ctx − a·z_par` |
| `adacad` | ✅ | per-token `a_t = JSD(p_ctx‖p_par)`, log base 2 |
| `ckplug` | ✅ | entropy-gated geometric blend; factor IS alpha ∈ [0,1] |
| `caa` | ✅ | plain mean-difference CAA |
| `act_add` | ✅ | CAA + orthogonalization against v_use |
| `csks` | ❌ | raises; needs trained proxy models |

## Invariants — do not quietly remove these

These are confound controls, not decoration. Each exists because the naive
version of the measurement was wrong.

1. **GroupKFold by relation, for both axes.** Probes AND steering vectors are
   fitted on training relations and applied only to held-out ones. A CAA
   vector fitted on the items it then steers is the same leak as a probe
   scored on its training relation.
2. **Base-rate AUROC beside every detection AUROC.** A predictor that knows
   only the relation's training base rate. Provenance: a 0.822-vs-0.936
   postmortem where the "probe" was largely reading which relation it was.
   Under the conflict task it is 0.5 by construction — keep reporting it
   anyway, because that is the check that the pairing actually held.
3. **BOW beside every detection AUROC.** Text-only null. If a probe cannot
   beat TF-IDF over the same passage and question, it found a property of the
   items, not of the model. Strictly stronger than the base-rate control.
   **Caveat under the conflict task:** "does this passage contradict?" is
   partly decidable from the passage text alone, so BOW stops being a null and
   becomes close to an oracle. GroupKFold by relation blunts that but does not
   remove it. Read the BOW row here as a *ceiling on the text-only route*, not
   as a floor a probe must clear — and say so in any write-up.
4. **`logit_lens` readout control at the probe's own layer and position.**
   Reported as `auroc_over_readout`. Without it, a probe at the answer
   position cannot be distinguished from a decoder.
5. **Raw AND R-corrected numbers, always both.** Never one instead of the
   other. What changed is which is *primary*: raw now is (`format_correct:
   false`), and the corrected view rides alongside in `auroc_corrected` /
   `flip_rate_corrected` / `margin_*_corrected`. See "The R correction" below.
6. **Matched control for every steering effect.** Norm-matched random
   direction for the activation methods; the unsteered distribution for the
   decoding family; a **placebo instruction** — same block, same slot, no
   directional claim — for `prompt_instruct`. `specific_effect` = Δmargin −
   control Δmargin, and it is **NaN when a method declares no control**, never
   a silent zero. (It used to be `.fillna(0)`, which quietly put
   `prompt_instruct`'s bare Δmargin in the same column as CAD's controlled
   one.) `has_control` in the summary says which is which.
7. **The S arm is the specificity control for every steerer.** Each one runs
   under C *and* S at the same dose on the same item;
   `conflict_specific_effect` = effect(C) − effect(S). A method that moves the
   margin just as hard when there is nothing to arbitrate is amplifying the
   passage, not resolving a conflict.
8. **Both targets, always, as their own rows — and never a flip rate without
   its `n_flippable`.** Collapsing a steerer to its best cell across targets by
   argmax is how `use_context` at 4-of-6-flippable-items came to outrank
   `use_parametric` at 390-of-394 in the pilot table, and how a method whose
   context arm was a no-op by construction still got a plausible-looking row.
   Rank on `effect_toward_target` (the controlled Δmargin re-signed so higher
   is better for whichever target the row is about), not on flip rate.
9. **Prompts come only from `core/prompts.py`.** A method that builds its own
   prompt breaks comparability across conditions. One documented exception:
   `p_true` asks a question that is not the item's question, so the shared
   stem does not apply — but it still calls `prompts.wrap_chat()`, so it
   reaches the model in the same format as everything else.
10. **The chat template is applied in `core/prompts.py`, never in
   `core/model.py`.** `core/positions.py` resolves named probe positions by
   mapping block character spans through the offset mapping, so wrapping the
   prompt any later shifts every span and silently points each probe at the
   wrong token. `build_with_spans` returns spans already shifted past the
   template prefix, and falls back to the raw body with a warning if the
   template does not reproduce the body verbatim.
11. **Continuations come from `prompts.continuation()`, never `" " + answer`.**
   In raw mode the answer continues the question and carries its leading
   space; after a chat template's assistant header the turn starts fresh and
   that space is a spurious token. Hard-coding it computes the margin over a
   different string than the prompt implies. `data.first_token_collisions`
   follows the same rule.

## Conventions

- Margin `m = logp(true) − logp(cf)`. **m > 0 parametric wins, m < 0 context
  wins.**
- Detector score convention depends on the task: under `conflict`, HIGHER =
  more contradiction; under `arbitration`, HIGHER = more context-following.
  Most detectors point the same way under both — `margin`, the divergences and
  `logit_lens` all grow when the passage pushes away from the parametric
  answer. Two flip explicitly on `self.task`: **`confidence_gain`** and
  **`p_true`**. Trainable detectors learn the orientation from the labels.
- Every flip rate is reported with the **`n_flippable`** it was computed over,
  and **both targets are always their own rows**. Never collapse a steerer to
  its best cell across targets — see the postmortem below.
- `use_context` success is `m0 > 0 → m1 < 0`; `use_parametric` is the mirror.
  Items already on the target side are not flippable; `flip_rate_flippable`
  conditions on the ones that could move.
- Labels are **behavioural**: the model's generated answer under C matched
  against counterfactual aliases. Never teacher-forced — the parametric axis
  has to be defined by what the model actually says.
- Layer indices accept depth fractions (`0.6` = 60% up) so configs survive a
  model swap.

## The R correction — now OFF by default

`corrected(C) = margin(C) − margin(R)`, where R is a real passage of the same
shape from a different relation, synthesised by the loader. It was meant to
remove the part of the margin that is the passage *template* reweighting the
two answer strings, before any conflict is involved.

**It is no longer the default, and the reason is measured.** R differs from C
on *two* axes at once:

1. the template / answer-string reweighting — what we wanted to remove;
2. **whether the passage is about this item at all** — what we did not.

Under R the model has nothing usable and falls back on memory, so `m_R` is
large and positive, and subtracting it drives `corrected` negative almost
always. Backed out exactly from the 400-item pilot table (three method rows
agree on integer counts):

- `margin(C) − margin(R) < 0` for **394/400 items (98.5%)**
- behavioural context-following rate: **0.470**
- ⇒ only **6 items** were flippable toward context, which is what
  `caa`/`act_add` at "0.667" (4 of 6) and `prompt_instruct` at "0.167" (1 of 6)
  actually were
- ⇒ `margin` as a detector collapsed to **0.472**, i.e. chance, because the
  corrected score is near-constant-sign

The `corr(margin_C, margin_R) = 0.99997` figure quoted here previously came
from the CPU smoke run on a **randomly-initialised** model, where both margins
are pure answer-string constants because there is no knowledge to interfere.
It was never evidence about the real 8B, and it was the stated justification
for the whole correction.

What replaced it: the **paired S-vs-C design**, which removes the same per-item
constant *exactly* rather than estimating and subtracting it, because S and C
share the item, the question, the template and both answer strings.

R is still scored, and the corrected view is still reported in every table
beside the raw one (invariant 5). `format_correct` only decides which of the
two is **primary**, and it defaults to `false`. Set it `true` to reproduce the
old numbers.

The offset is a per-item constant, so **it cancels out of Δmargin exactly**.
There is deliberately no `d_margin_raw` column. What the correction changes is
which side of zero an item sits on, i.e. the flip rate — hence `flip_rate_raw`
vs `flip_rate_corrected`, both reported.

## Probe positions

`position: -1` is the last prompt token — the state that emits the next token.
A probe there, scored against the model's own continuation from the same
prompt, is largely decoding an answer already committed to. A raw integer is
not a fix either: index 137 is a different linguistic location on every item.

So positions are **named and resolved per item** against the prompt's block
spans via the tokenizer's offset mapping (needs a fast tokenizer):
`end_of_context`, `answer_mention_last`, `subject_last`, `end_of_stem`,
`last`. `detection_positions` sweeps them; `last` stays in so the
readout-vs-upstream gap is measured rather than assumed.

Gotcha already hit and fixed: BPE tokens carry their leading space, so
`" Monaco"` starts one char before the span of `"Monaco"` — `_last_token_in`
matches on token *end* inside the span, not strict containment.

The detection table runs on the **intersection** of items resolvable at every
swept position (AUROC at two positions is only comparable over the same
items); drops go to `detection_dropped.json`.

## Context of the recent changes

Four rounds, in order. Knowing why each happened prevents re-breaking them.

**Round 1 — unblocking the run.** Four things blocked any real run: the
loader used guessed JSON field names, `build_prompt` was a placeholder,
`act_add.fit` raised, and the model was semi-hardcoded. Fixed against the real
`ConfiQA_cloze_clustered` schema; CAA vectors are now fitted in-run from the
behavioural contrast; the model is a config block with architecture-agnostic
layer discovery. Also added the shared activation cache and artifact store so
mechanistic methods persist tensors rather than only AUROCs.

Found while doing it: `limit: 200` took whole relation files in filename order
and left GroupKFold with 2 groups (now stratified); the steering loop's
for/else/break aborted a whole method on any exception; `p_true` formatted a
`{subject}` placeholder that the cloze templates do not contain.

**Round 2 — N/S/C/R actually running.** The design was declared but only C was
ever scored, and the R correction survived only as a hidden offset inside two
methods, each recomputing it. Introduced `MarginTable` (shared, cached,
persisted), made the four conditions their own pass, and made every record
carry raw *and* corrected values. `ScoredCandidates` had been dead code; it is
now the table's unit.

Note: CAD at the time used a first-token DV, so it needed its own R baseline
in its own units. Round 4 removed that special case.

**Round 3 — the detection tautology.** Detection was near-circular: probe at
position −1 of the C prompt, predict the model's greedy output from that same
prompt. Added named positions, the position sweep, and the `logit_lens`
readout control. The critique that prompted this was right in direction;
"tautological" overstated it (the label is an 8-token alias match, folds are
grouped, layer 19 is mid-network) — but the fix was needed regardless, and
moving the position alone would not have been enough without the readout
control to prove it.

**Round 4 — the five methods + BOW.** Added `bow`, `context_kl`/`context_jsd`,
`adacad`, `ckplug`, `caa`, and rewrote `cad`. The decoding family now shares
one contrastive core and is scored on the **full multi-token teacher-forced
margin**, not a first-token proxy — previously CAD reported a different DV
from every other method, so its numbers did not belong in the same table. All
three verified to reduce exactly to the unsteered margin at strength 0, which
is why their matched control is the real unsteered value.

**Round 5 — the unified S-vs-C paradigm, and two genuine bugs.**

Triggered by CK-PLUG and `confidence_gain` scoring *below chance*. Reading the
paper (arXiv:2503.15888, verbatim equations) against the code found two real
errors and one framing mistake.

*The `ckplug` blend was wrong.* The code had
`log p = a·log p_ctx + (1−a)·log p_par`. The paper (Eqs. 5, 6, 8) mixes the
parametric log-probs with a **log-ratio**, `q_cont = log p_ctx − log p_par`,
which works out to `a·log p_ctx + (1−2a)·log p_par`. Note the coefficient:
**(1−2a), not (1−a)**. Under the old form `a = 1` gives plain `log p_ctx` —
the *unsteered* model — so CK-PLUG could only interpolate between "do nothing"
and "go fully parametric", and **its entire `use_context` arm was a no-op by
construction**. Under the paper's form `a = 1` gives `log p_ctx − log p_par`, a
real extrapolation away from memory. Also added: the Eq. 9 `V_head` top-k mask
(`select_top`, default **0/off** — a masked answer token makes the
teacher-forced margin `-inf`) and the Appendix-B relative gate
(`gate_epsilon`, default 0.0 = the Eq. 7 `CG < 0` form). `factor` is now a dose
in [0,1] so it sits on the same axis as every other steerer; `factor_is_alpha:
true` restores the old parameterization. Honest caveat kept in the docstring:
unlike `cad`/`adacad`, CK-PLUG does **not** reduce to the unsteered margin
exactly at dose 0 (`a = 0.5` gives `0.5·log p_ctx`, rank-equivalent but not
identical). Watch `gated_token_rate`: a passage that names an answer *sharpens*
the distribution at the answer tokens, so `CG > 0` and the gate skips exactly
the tokens the DV is computed from.

*`confidence_gain` had the paper's sign, applied to the wrong question.*
`CG = H(p|question) − H(p|passage,question)`; CG < 0 = conflict. That is a
**between-condition** contrast — conflicting vs supporting passage. The old
detection axis asked a **within-C** question (who won?), where the direction
inverts: a model that commits to the context is confident about the
counterfactual, so H_ctx is *low* and CG is *positive*. Hence 0.466, i.e.
0.534 the other way up. Now oriented by `self.task`. Entropy moved to log base
2 to match Eqs. 2–3 (irrelevant to AUROC, relevant to the ε thresholds).
Note the paper reports **no AUROC for CG as a classifier anywhere** — it is
validated only as a gate — so there is no published number to match.

*The framing.* The above is what motivated the S-vs-C reformulation: CK-PLUG's
own question is the benchmark's detection question, and running it as
`detection_task: conflict` makes the base rate 0.5, pairs the design, and
retires the R correction (which the same postmortem showed was doing real
damage — see "The R correction"). Steering gained its S arm as a matched
specificity control. `p_true` gained a conflict-task template that asks about
the passage rather than a fixed counterfactual claim, because the old one
returned nearly the same number under S and C and could not have separated
them. `answer_mention_last` became condition-aware — it hard-coded the
counterfactual, so it was unresolvable under S and would have dropped every
item from the new axis.

*The prompting arm.* Two things were costing it, both now fixed in
`core/prompts.py`. The harness fed **raw text to an Instruct model**, which is
not the regime instruction-following was trained into — `chat: auto` now wraps
every prompt in the model's own template. And `prompt_instruct` inserted its
directive **between** the passage and the question, where it reads as another
sentence of the document; `instruction_position: before_context` is now the
default, with `after_context` kept for the old behaviour. It also gained its
placebo control. Both settings hurt `prompt_instruct` most, since it is the
only method whose whole mechanism is instruction-following, but the chat
template also removes a chunk of the formatting noise in `confidence_gain`'s
entropy difference.

Watch for on the next real run: the chat template roughly doubles prompt
length (25 -> 60 tokens on the smoke item), `last` becomes the end of the
assistant header rather than the question mark, and the answer continuation
loses its leading space — so **first-token collisions are recomputed
differently** and the margin is over a different string. The old numbers are
not comparable across this change; `chat: off` reproduces them.

## Running

```bash
python -m conflict_bench.run --config conflict_bench/configs/default.yaml
python -m conflict_bench.experiments.run_pilot --limit 200
python -m conflict_bench.experiments.run_pilot --model google/gemma-2-2b-it
```

`run_pilot.py` is deliberately a straight line: model → data + labels →
conditions → detection → steering → `final_table.csv`
(`metrics.simple_report`). No stage checkpointing, no CLI overrides beyond
`--config/--limit/--model`, no extra report variants — what runs is decided by
the config. Keep it that way. `pilot_five.ipynb` calls the same functions
(`run_pilot.load_data/run_steering/final_table`, `run.run_conditions/
run_detection`) — keep it that way, or they drift. The margin table and the
activation cache are still on disk, so a re-run reuses those.

## Known gaps and caveats

- **CK-PLUG is reimplemented from the paper's equations**, not vendored from
  `byronBBL/CK-PLUG`. Verify against their code before quoting numbers against
  theirs — in particular `_ck_decoding` in their vendored
  `transformers-4.49/.../generation/utils.py`, which was not read when this
  was written (whether they use natural log despite `log2` in the paper,
  whether step 0 is skipped, how the two KV caches are aligned). Same caution for AdaCAD's `use_parametric`
  direction, which extrapolates past a paper posed purely as a
  context-faithfulness method.
- **The decoding family has no `fluency`** — it scores teacher-forced margins
  without generating, so it appears in the dose-response panel but not the
  Pareto. Needs generation through a logits processor.
- `selfcheck_nli` and `csks` still raise.
- `semantic_entropy` clusters by alias match, not NLI. The K samples are
  persisted, so it can be re-clustered without re-sampling.
- Named positions need a **fast tokenizer**; slow tokenizers fall back to `-1`
  with an explicit error.
- **Guards fire loudly on degenerate labels.** If the context-following rate
  is near 0 or 1, every trainable method (bow, probes, CAA vectors) refuses to
  fit with a single-class message. That is correct behaviour, not a bug —
  check `context_following_rate` in `manifest.json` before scaling a run up.

## Testing without a GPU

There is no test suite. The way everything so far was verified: build a tiny
randomly-initialised Llama (real architecture, real tokenizer, ~4 layers,
hidden 64) and run the full pipeline on CPU in seconds. Real code paths, no
downloads, meaningless numbers by design. For the trainable methods, patch
`run.behavioural_labels` to return balanced synthetic labels — otherwise a
random model gives a single-class split and every fit refuses.

Three bugs were caught this way that reading would not have caught: the
BPE-leading-space position bug, colliding labels in the notebook chart, and
`control_margin_after` being written in the *corrected* view while
`margin_before/after` were in the raw one — which silently made
`specific_effect` NaN for every steerer, including the ones that do declare a
control. Worth keeping the habit.

Under `detection_task: conflict` the label is the condition, not the
generation, so the detection axis no longer needs the balanced-synthetic-label
patch. The **trainable steerers** still do (their contrast is behavioural), so
keep patching `run.behavioural_labels` when exercising `caa` / `act_add`.

What the smoke run is good for beyond "it runs": it reproduces the R-correction
pathology in miniature. On the random model the raw margins are large
(~-15.7) so `flip_rate_raw` is 0.000 everywhere, while the corrected margins
are tiny (~-0.05) and noise flips them — `flip_rate_corrected` comes out at
0.04-0.21. That gap *is* the mechanism: the correction manufactures flips.
Check `n_flippable` too — it should be a workable split on both arms (8 vs 16
of 24 on the smoke run), never the 6-vs-394 the corrected view produced.
