# IDEAS.md — audit findings and the fixes to apply

Written 2026-09-09 from an audit of behaviour rather than of claims: findings
were derived from the code, a full CPU pipeline run on a randomly-initialised
4-layer Llama, and descriptive statistics over all 6000 records. README.md and
CLAUDE.md were then read as a cross-check. Where a finding contradicts a
documented decision that is said explicitly, so the disagreement is visible as
a disagreement — see F2 and F8.

Ordering is by *how much a conclusion moves if the fix is not applied*, not by
effort. F1–F3 change published numbers. F4–F5 are silent-corruption bugs.
F6 onward are correctness and hygiene. F12 is documentation that has drifted
from the code.

Each entry: what is wrong, where, the evidence, the fix, and the check that
says the fix landed.

---

## F1. The behavioural label is contaminated — fix the matcher first

**Where:** `run.py:62-66` (`behavioural_labels`), `data.py:116` (the alias
filter that isn't).

**What is wrong.** The label is an unanchored, case-folded substring test of
the alias list against 8 generated tokens:

```python
follows = any(a.lower() in g for a in [it.counterfactual_answer] + it.cf_aliases)
```

Measured over the corpus:

| symptom | rate |
|---|---|
| items carrying a cf alias < 3 chars | 23.5 % (1410/6000) |
| items labelled context-following by **boilerplate alone** (generation contains no answer) | 3.1 % (188/6000) |
| items labelled context-following while emitting a **perfect true answer** | 2.6 % (157/6000) |
| mirror case: `matched_parametric` true while emitting the cf answer | 2.7 % |

The offenders are `IN` (India, 60 items), `CA`, `IT`, `es`, `se`, `ro`, `en`,
`US`. Lowercased, `"IN"` matches the word "in", so those items are label-1
regardless of what the model says. `"en"` matches "when". `"acting"` matches
`"film acting"`.

Both configs set `min_alias_len: 0`, and raising it would not help: the loader
in use only filters `cf_answer`, never the alias list (`data.py:116`). The
alias filter exists only in `load_confiqa`, the unused flat-JSON path
(`data.py:161-166`).

Net label noise is at least ~5 %, asymmetric, and correlated with relation and
entity type — exactly the confound the base-rate and BOW controls exist to
catch, now injected upstream of them. Every AUROC, every fitted probe and
every CAA vector inherits it, and the effects the harness is built to resolve
(`auroc_over_readout`, `auroc_margin_over_base`) live at that same scale.

**Fix.**

1. Replace the substring test with a word-boundary matcher that keeps case for
   short strings:

   ```python
   def _alias_hit(alias, gen):
       a = alias.strip()
       flags = 0 if len(a) < 4 else re.I     # "IN", "en", "US", "G B"
       return re.search(rf"(?<!\w){re.escape(a)}(?!\w)", gen, flags) is not None
   ```

   Short aliases keep their case (real acronym answers are uppercase);
   everything else stays case-insensitive. Boundaries are `\w`-based so
   `"U.K."` and `"Manchester City F.C."` still match.

2. Add `min_alias_len` filtering **of the alias lists** to
   `load_confiqa_cloze_clusters`, defaulting to 3, and drop the 30 items whose
   true and cf alias sets intersect. Set `min_alias_len: 3` in both configs.

3. Use `matched_parametric` instead of discarding it. `behavioural_labels`
   already computes `keeps` at `run.py:65` and throws it away. Return a
   three-way outcome and let the runner drop the ambiguous cells:

   ```
   follows and not keeps  -> label 1
   keeps and not follows  -> label 0
   neither                -> label None   (refusal / off-topic)
   both                   -> label None   (ambiguous)
   ```

   Items with `label is None` are excluded from every fit and every AUROC and
   counted in `manifest.json` as `n_label_abstain` / `n_label_ambiguous`.
   Right now "answered parametrically", "refused" and "said something else"
   are one class, which is not the dichotomy the design claims to measure.

4. Constrain the generation: set `answer_prefix: "\nAnswer:"` in the prompt
   spec, or stop at the first newline, so the match window is the answer slot
   rather than 8 tokens of preamble.

**Check.** Re-run `stage_data`; `labels.csv` gains an `outcome` column;
`n_label_ambiguous` is reported and non-trivial (expect 10–25 % on a real
model); the boilerplate probe above returns 0/6000. Keep that probe as the
first case in the test suite (F11c).

---

## F2. The R-corrected margin is an effect size; stop reading its sign

**Where:** `core/margins.py:109` (`corrected`), `:174-181`
(`context_win_rate_corrected`), `methods/steerers/steerers.py:36,74`
(`_flipped`, `record`), `metrics.py:46` (`steering_summary`).

**This one is a disagreement with a stated decision, not a bug report.**
README.md ("Because the R offset is a per-item constant it cancels out of
Δmargin exactly … What the correction changes is which side of zero an item
starts and ends on: the **flip rate**") and CLAUDE.md both describe this
behaviour exactly and present it as the design. The argument below is that the
reasoning is right and the conclusion drawn from it is backwards.

**What is wrong.** `corrected(C) = m(C) − m(R)` is a good
difference-in-differences: it equals `[m(C) − m(N)] − [m(R) − m(N)]`, the
conflict effect net of the template effect. The problem is what the code then
does with it — takes its **sign** as "which answer the model prefers", and
makes that primary:

- `context_win_rate_corrected` = `(margin_corr < 0).mean()`
- `flipped` = sign change of the corrected margin, and `flip_rate` is the
  headline in `pilot_report.csv`; the behavioural quantity is demoted to
  `flip_rate_raw`.

A DiD ≤ 0 means "the conflicting passage moved the model further toward the
counterfactual than an irrelevant passage did". It does **not** mean the
counterfactual answer wins. Those are different claims, and only the second is
what the behavioural label measures.

**Evidence** — `conditions_summary.csv` from the CPU run (random weights, so
there is no real effect to find):

```
condition  sd_margin  context_win_rate | sd_corr  context_win_rate_corrected
    N        19.07         0.2917      |  0.164          0.4167
    S        19.07         0.2917      |  0.165          0.5000
    C        19.05         0.2917      |  0.151          0.4583
    R        19.06         0.2917      |    —              —
```

The correction removes 99.2 % of the variance. The raw win rate is *identical*
across all four conditions — correct, nothing is happening. The corrected rate
swings 0.42 / 0.50 / 0.46, which is noise crossing zero reported as
arbitration. Downstream, `act_add` reports `flip_rate = 0.25` on
`d_margin = 2.8e-4`.

**The evidence quoted for the design is from a null model.** README.md's
justification is: "In the CPU smoke run the raw margin correlated **0.99997**
with the R margin — i.e. the raw DV was almost entirely the per-item
answer-string/format constant, and the corrected margin was two orders of
magnitude smaller than the raw one (sd 0.19 vs 21.4)." My run reproduces the
shape of this exactly (sd 19.06 raw vs 0.15 corrected).

But that smoke run is a **randomly-initialised** network. A model with no
context sensitivity *must* produce `m(C) ≈ m(R)` — there is no conflict effect
for the C passage to add, so the two conditions differ only by noise, and the
correlation is 1.0 by construction. The statistic cannot distinguish "the raw
DV is dominated by a format constant" from "this particular model has no
context sensitivity at all", and on a random model the second is known to be
true. So 0.99997 is not evidence for the correction; it is the null model
behaving like a null model.

The same number on **Llama-3.1-8B** would be evidence, and it is cheap to get:
`condition_margins.csv` from any real pilot run already contains both columns.
Until that number exists, the decision to promote the corrected margin to
primary rests on nothing. Measure `corr(margin_C, margin_R)` and
`corr(margin_C − margin_R, margin_C − margin_N)` on the real model before
keeping or discarding this design.

**The built-in guard points the wrong way.** `condition_summary`'s docstring
says the correction is suspect if R's win rate is *far* from N's. The real
failure mode is the opposite: R ≈ N is what tells you the subtraction is
removing the parametric prior along with the template. Rewrite that note.

**Fix.**

1. **Flips are defined on the raw margin.** Swap the roles in
   `_MarginMixin.record`: `flipped` ← raw, `flipped_corrected` ← corrected
   (keep both, name both honestly). `steering_summary`'s `flip_rate` becomes
   the raw one; `pilot_report` / `triplet_report` follow.
2. **`flippable` is defined on the raw margin too** (`steerers.py:81-82`) — an
   item is flippable if the model's actual preference is on the wrong side.
3. Rename `context_win_rate_corrected` to `conflict_effect_negative_rate`, or
   drop it. It is not a win rate.
4. Keep the corrected margin as the effect DV for `d_margin`,
   `specific_effect` and dose-response — where the per-item offset cancels
   exactly and the correction costs nothing — and report it with an interval
   (F3), not a sign.
5. Add `margin_R_minus_N` per item to `condition_margins.csv`. That column
   *is* the template effect the correction is meant to remove; if it is near
   zero, R is doing nothing and the correction is cosmetic.

**Check.** `flip_rate` and `flip_rate_raw` become the same number by
construction; the corrected flip rate survives as a clearly-labelled
diagnostic. On the random-model smoke run every flip rate should be ~0.

---

## F3. Headline numbers are maxima over a sweep, with no interval

**Where:** `metrics.py:132,136,138` (`best_flip_rate`,
`best_flip_rate_flippable`, `best_specific_effect`), `metrics.py:114`
(`prompting_baseline_flip`), `experiments/run_pilot.py:163`
(`s.loc[s.flip_rate_flippable.idxmax()]`).

**What is wrong.** Every steering number in every headline table is a maximum
over targets × factors, chosen on the same data it is reported on. With a 4–5
point factor sweep and two targets that is a max over 8–10 draws — a
systematic upward bias, and the ranking between methods is partly a ranking of
how many factors each was given (`ckplug` gets 5, `adacad` gets 3).

There is also **no confidence interval, bootstrap or significance test
anywhere in the repo**. `auroc_std` is the spread of 5 fold estimates over ~8
relations, not a CI. The notebook's "beats BOW / does NOT beat BOW" verdict
(cell 12) is a bare comparison of two point estimates.

**Fix.**

1. **Select the factor on held-out relations.** The fold structure already
   exists: pick the best factor per (method, target) on each fold's training
   relations, report the effect at that factor on its test relations. Nested
   selection, same GroupKFold.
2. If that is too heavy for a pilot, make the **whole dose-response curve** the
   headline and rename the maximum to `best_flip_rate_insample_max` so nobody
   quotes it as an effect.
3. **Add a bootstrap.** Cluster bootstrap over *relations* (not items — items
   within a relation are not independent), 1000 resamples, for detection
   AUROC, `auroc_margin_over_base`, `auroc_over_readout`, `d_margin` and
   `specific_effect`. One helper in `metrics.py`:

   ```python
   def cluster_bootstrap(fn, groups, n=1000, seed=0): ...
   ```

   Every summary table gains `*_lo` / `*_hi` columns.
4. The "beats BOW" verdict becomes a **paired** bootstrap on the AUROC
   difference over the same items, reported as a CI, not a `>`.

**Check.** `detection_summary.csv` and `steering_summary.csv` carry CI
columns; `pilot_report.md`'s reading notes cite intervals; no bare `.max()`
survives in a column whose name does not say `insample_max`.

---

## F4. Probes are fitted on C and scored under whatever condition is swept

**Where:** `run.py:273` calls `D.fit(train_items, train_labels)`;
`detectors.py:179,251,301` declare
`fit(self, items, labels, condition=Condition.CONFLICTING)`.

**What is wrong.** The runner never passes the job's condition, so the default
wins. With `detection_conditions: [N, S, C]`, `bow`, `linear_probe` and
`diffmean_proj` train on C text/activations and are applied to N or S. Silent
train/test distribution mismatch, and the resulting AUROC is uninterpretable.

This matters more than it first looks, because README.md singles out that exact
mode as the interesting one: "labels are behavioural under C, so `[N, S, C]`
asks whether the pre-conflict state already predicts how an item will be
arbitrated. It defaults to `[C]` because that is a different and **much
stronger hypothesis**." The stronger hypothesis is the one the bug breaks —
the first person to flip that config key gets a number that looks like a
result.

**Fix.** Pass the condition through:

```python
D.fit([items[i] for i in tr], [labels[i] for i in tr], condition=cond)
```

and lift `condition` into the `Detector.fit` signature in `methods/base.py` so
the contract is declared, not implied by three subclasses agreeing. Record
`fit_condition` in each method's `fit_meta` and in the `detection_summary`
row, so a mismatch is visible in the output even when it is intended.

**Check.** Run with `detection_conditions: [N, S, C]`; `artifacts/*/meta.json`
shows `fit_condition` tracking the job condition; probe AUROC under N is no
longer silently produced by a C-trained probe.

---

## F5. Caches are keyed by out_dir alone — a model swap reuses stale tensors

**Where:** `core/margins.py` (`MarginTable.__init__` → `load()`),
`core/activations.py` (`ActivationCache.__init__` → `load()`),
`experiments/run_pilot.py` (`stage_data` resuming `labels.csv`).

**What is wrong.** Both caches key on `(item_id, condition[@position])` and
load from `out_dir` on construction. Nothing about the model, dtype or prompt
spec enters the key — README.md states the activation key as
`(item, condition, layer, position)`, which is accurate and is the problem.
`out_dir` comes from the config, so:

```
python -m conflict_bench.experiments.run_pilot                        # Llama
python -m conflict_bench.experiments.run_pilot --model gemma-2-2b-it  # reuses Llama's
```

silently reuses the first model's labels, margins and activations. The same
applies to editing the `prompt:` block, which changes every prompt string
while leaving every cached margin keyed identically.

**This is a documented workflow, not a hypothetical.** README.md's own
quickstart block is:

```bash
python -m conflict_bench.experiments.run_pilot --model google/gemma-2-2b-it \
    --relations P17 P27 P36
```

with no `--out`, run against a config whose `out_dir` is `runs/pilot_five`.
Following the README as written, after any earlier Llama run, produces a Gemma
"result" built from Llama's labels, Llama's margins and Llama's residual
stream. Fix the code, then fix the README example to carry `--out`.

**Fix.**

1. Write `cache_key.json` into `out_dir` on first use holding
   `{model_name, revision, dtype, prompt_spec, dataset_kwargs}`. On load,
   compare; on mismatch **refuse to start**, naming the changed field and
   suggesting a new `out_dir` (or an explicit `--force-cache-reset`).
2. Put a short hash of that blob in the on-disk names
   (`margin_table.<hash>.jsonl`, `activations/<hash>/layer*.npz`) so two
   models can share an out_dir instead of corrupting one.
3. `stage_data` validates the same key before resuming `labels.csv`.

**Check.** Re-running with `--model` changed into an existing out_dir aborts
with a clear message rather than producing a table.

---

## F6. `semantic_entropy` scores the label, not a prediction of it

**Where:** `detectors.py:408` (`score=frac_cf`).

The score is the fraction of K temperature-1 samples containing the
counterfactual answer, matched with the same alias matcher that defines the
label from the greedy sample. That is a Monte-Carlo estimate of the label, not
a detector of it — it will top the detection table and mean nothing. The
entropy the class is named after is computed and discarded.

**Fix.** Two options, and the second is worth more:

- Score `-H` and state that it predicts *unresolved* arbitration, a
  non-directional uncertainty claim with AUROC expected near 0.5 either way; or
- keep `frac_cf`, rename it `sample_agreement_oracle`, and move it out of the
  detector table into **label-quality diagnostics** — it is a direct measure of
  how many items have a label that is an artifact of greedy decoding, which is
  exactly what F1 needs.

**Check.** No detector in `detection_summary.csv` has access to the label's own
generating process. The new diagnostic reports the fraction of items whose
greedy label disagrees with the majority of K samples.

---

## F7. `margin` is framed as a floor but is a ceiling by construction

**Where:** `detectors.py` (`MarginBaseline`), and the comments in
`configs/*.yaml`.

The teacher-forced margin under C is the log-prob twin of the greedy
generation that *defines* the label. "The floor every fancier detector must
beat" is upside down: it is close to an oracle, and a probe failing to beat it
is the expected result, not a finding.

**Fix.** Move `margin` out of the comparison column into its own row labelled
*concurrent readout of the DV*, next to `logit_lens`. The floor methods must
beat is `bow` and `base_rate`; the ceiling is `margin`. Say so in
`pilot_report.md`'s reading notes, and report `auroc_margin_over_bow`
explicitly as the number that matters.

---

## F8. The "matched control" is degenerate for four of six steerers

**Where:** `steerers.py:365` (`control_margin_after_raw=m0` in
`_ContrastiveDecoder.steer`).

**Also a disagreement with a stated decision.** README.md metric #2 says
"always vs matched control (norm-matched random direction for act-add; **α=0
for decoding methods**)", so α=0 is the intended control, not an oversight.

For `cad`, `adacad` and `ckplug` that control is the unsteered margin, so
`d_margin_ctrl ≡ 0` and `specific_effect ≡ d_margin` identically — confirmed in
the smoke run. A control that is arithmetically guaranteed to be zero cannot
discriminate anything; it makes `specific_effect` a second name for
`d_margin` while implying it has been adjusted for something. The α=0
reduction is a valuable *sanity check* (and it passes), but it is not the same
object as act_add's norm-matched random direction, and putting the two in one
column labelled "matched control" makes the mechanistic and decoding arms look
comparable when only one of them has been controlled.

**Fix.** Give the decoding family a real matched control: compute `z_par` from
the **R prompt** rather than the N prompt, at the same strength. That holds the
amount of contrastive machinery fixed and varies only whether the contrast is
informative — the decoding analogue of a norm-matched random direction. Keep
the strength-0 reduction as a separately-named sanity check
(`reduces_at_zero`), which is what it actually is.

**Check.** `d_margin_control` is non-zero and non-constant for the decoding
family; `specific_effect` stops being an alias of `d_margin`.

---

## F9. Dataset QC the loader in use does not apply

**Where:** `data.py:97-146` (`load_confiqa_cloze_clusters`) vs
`data.py:147-186` (`load_confiqa`, unused).

Measured over 6000 records:

| condition | rate |
|---|---|
| subject absent verbatim from the **conflicting** passage | 14.8 % |
| true answer absent from the **supporting** passage (S does not support) | 6.8 % |
| true answer still present in the conflicting passage | 2.0 % |
| cf answer present in the supporting passage | 0.8 % |
| synthesised R passage contains the item's true / cf answer | 0.7 % / 0.6 % |

The `subject_rewritten` filter exists in the unused loader
(`data.py:164-165`) and not in the one that runs.

**Fix.** Port the QC into `load_confiqa_cloze_clusters` as flags, all logged to
`qc_dropped.json` with reasons: `subject_rewritten`,
`unsupported_supporting_passage` (true answer missing from S),
`answer_leak_in_R`. Default them on — S is a named condition of the design, and
1 item in 15 where S does not support is a mislabelled condition, not noise.
Re-draw R when it leaks rather than dropping the item.

Also: `drop_first_token_collisions: true` is still the default in both configs
though nothing but `logit_lens` uses a first-token DV since the decoding family
moved to multi-token teacher forcing. It removes a non-random slice of the
population (the orthographically similar answer pairs) for a defunct reason.
Default it to `false` and let `logit_lens` report its own collision rate.

The docs confirm this is a leftover rather than a live choice: README.md still
justifies the filter with "CAD scores a first-token margin, so leaving them in
hands that method a structurally zero DV", and CLAUDE.md's Round 4 note says
that special case was removed. Both configs' inline comments repeat the stale
justification. Whatever is decided about the flag, the three explanations
should stop contradicting each other (F12).

---

## F10. Smaller correctness items

- **`p_true` builds its own prompt** (`detectors.py:437`), bypassing
  `PromptSpec` and hardcoding `"Background: "`. Breaks the
  single-source-of-prompt rule and desynchronises from the `prompt:` config
  block. Route it through `build_with_spans` with an instruction-style block.
- **The readout control attaches by `row["layer"]`** (`run.py:197,204`), which
  is populated only from `method_cfg`. Omit `layer:` from a probe's config —
  the class default of 19 lives in `__init__` — and `auroc_over_readout`
  silently never attaches. Have methods report their *resolved* layer into the
  summary row (from `fit_meta`), not the config dict, and raise when a
  position-swept probe finds no readout partner, since it is documented as
  mandatory.
- **`mean_adaptive_alpha` / `gated_token_rate` are wrong** — stashed on `self`
  inside `combine()`, which runs twice per `steer()` (true continuation, then
  cf), so the recorded value is only the cf pass's. Return them from `combine`
  or average the two.
- **Nothing seeds torch.** `semantic_entropy` samples at temperature 1.0
  unseeded. Add `set_seed(cfg["seed"])` covering `torch`, `numpy` and `random`
  in `run.main` / `run_pilot.main`, and seed per item for sampling so a resumed
  run reproduces.
- **`act_add` fits at position −1 and applies at every position**
  (`add_direction(..., positions=slice(None))`). Defensible as CAA practice,
  but it is an undeclared mismatch between where the vector is measured and
  where it acts. Make the application span an explicit config key
  (`apply_positions: all | after_context | last`) and record it.
- **`confidence_gain`, `context_kl`, `context_jsd` are non-directional** — they
  measure *that* the passage moved the distribution, not *toward what*. The
  a-priori sign is right (better than flipping post hoc), but an AUROC below
  0.5 should be reported as `|AUROC − 0.5|` with the sign in its own column, or
  it reads as failure when it is evidence of a reversed sign.
- **`run_steering` decides whether to pass labels via
  `len(inspect.signature(S.fit).parameters) > 1`.** Breaks for any steerer whose
  `fit` takes `**kwargs`. Make `fit(self, items, labels=None)` the declared
  contract in `methods/base.py` and always pass labels.
- **Score-array alignment is undefended.** `run_detection` fills `scores[i]`
  per fold, then `metrics.grouped_auroc` independently reconstructs the same
  GroupKFold split. Correct today because both call `min(n_splits, n_groups)`,
  but nothing enforces it. Pass the fold assignment into `grouped_auroc`
  instead of recomputing it.
- **`group_folds` silently returns a single train=test fold** when there is one
  relation (`run.py:105-107`). That is a total leak reported as a normal AUROC.
  Warn loudly and set `n_folds: 0` plus a `note` in the summary row.

---

## F11. Framework work — what to build before adding methods

Method extensibility is genuinely good and should be left alone:
`@register_detector` / `@register_steerer`, a two-method contract,
`requires_training` / `uses_position` / `access` flags the runner honours,
`save_artifacts` for fitted params, per-method error isolation. A new detector
is ~30 lines and inherits the folds, the nulls, the position sweep and the
artifact tree. `_ContrastiveDecoder` reducing three papers to one `combine()`
is the right abstraction.

The gaps are elsewhere.

### 11a. Datasets are second-class

`getattr(data_mod, cfg["dataset"]["loader"])` means every corpus must be a
function inside `data.py`; there is no path to an external module. `Item` is
schema-locked to exactly one true string, one counterfactual string and two
alias lists, so free-form answers, multiple counterfactuals and
numeric/temporal conflicts do not fit — and the margin DV does not survive that
change either. N/S/C/R is hardcoded in three places
(`margins.ALL_CONDITIONS`, `prompts.build_with_spans`, `positions.resolve`),
and every loader must synthesise R itself.

Fix: a `register_dataset` decorator mirroring the method registry, with a
declared `(items, dropped)` return contract; move R synthesis into a shared
`conditions.py` helper so a loader supplies passages and the harness builds the
design; make the condition set data-driven where the enum currently hardcodes
it. Give `Item` an `answers: dict[str, list[str]]` shape
(`{"parametric": [...], "contextual": [...]}`) so a third arbitration target or
a multi-answer item is expressible.

### 11b. Nothing is batched

Every score is one forward pass per (item, candidate), batch size 1. The
redundancy is mechanical and large:

- the decoding family recomputes `z_ctx` and `z_par` for every (target, factor)
  although neither depends on either — roughly 10× waste;
- `confidence_gain`, `context_kl` and `context_jsd` each recompute the same
  N-prompt next-token logits;
- `logp_continuation` runs a separate full forward for the true and cf
  candidates that share a prompt.

For the default 200-item config the steering stage alone is ~50k forward
passes, most of them repeats. Add a `(prompt, continuation) → logits` cache
alongside the existing `MarginTable` (same shape, same lifecycle) and batch the
candidate pair. Expect 5–10× on the pilot with no change to any number.

### 11c. There are no tests

No `pyproject.toml`, no requirements file, no CI; the package imports only
because its parent scratch directory happens to be on `sys.path`. The
tiny-random-Llama CPU run used for this audit takes ~3 minutes and exercises
every real code path. Commit it as `tests/test_smoke.py` with assertions rather
than eyeballs:

- decoding family Δmargin == 0 at strength 0 (currently true: 0.0, 0.0, 4e-8);
- the boilerplate-label probe from F1 returns 0 matches;
- position resolution returns a token whose decoded text ends the intended span
  (the BPE leading-space case);
- a model or prompt change against an existing out_dir raises (F5);
- `flip_rate` on a random model is ~0 after F2.

Add `pyproject.toml` with the real dependencies (`torch`, `transformers`,
`scikit-learn`, `pandas`, `numpy`, `pyyaml`) so the package is installable
rather than path-dependent.

---

## F12. Documentation that has drifted from the code

These cost nothing to fix and are the kind of thing that silently misleads the
next person (or the next run of an agent) into trusting a behaviour that no
longer exists.

- **README.md still describes CAD as opting out of the shared R correction**:
  "One method opts out of the shared correction on purpose: CAD's DV is a
  *first-token logit* margin, so the table's full-continuation R offset is the
  wrong quantity to subtract, and it computes its own R baseline in its own
  units." The code's `ContextAwareDecoding` is a `_ContrastiveDecoder` using
  `_MarginMixin` on the full multi-token margin, and the same README contradicts
  itself further down: "CAD, AdaCAD and CK-PLUG share one contrastive core and
  are scored on the full multi-token teacher-forced margin, not a first-token
  proxy." Delete the first paragraph.
- **The first-token-collision justification is stale in three places** (README,
  both config comments) — see F9.
- **README's method tables mix implemented with planned.** `tuned-lens
  trajectory features`, `semantic entropy probes` and `SAE feature steering`
  appear in the same tables as working methods with no marker, while
  `selfcheck_nli` and `csks` raise `NotImplementedError`. CLAUDE.md's ✅/❌
  table is the right format; port it into README so a reader can tell what runs
  from what is aspirational.
- **README's outputs tree omits `detection_summary.csv`'s controls.** It lists
  "grouped AUROC + base-rate control" but the file also carries
  `readout_auroc`, `auroc_over_readout`, `auroc_raw` and
  `auroc_correction_gain`. Minor, but the readout control is described
  elsewhere as mandatory and should be visible in the artifact listing.
- Once F1–F5 land, **CLAUDE.md's invariant list needs editing, not just
  appending**: invariant 5 ("raw AND R-corrected, always both") is honest but
  currently applies to exactly one detector, and invariant 6 ("matched control
  for every steering effect") is not met for four of six steerers (F8).

---

## Suggested order of work

1. **F1** — label matcher and ambiguous-outcome exclusion. Everything else is
   measured against this label.
2. **F2** — flips on the raw margin. A role swap in one function; changes the
   headline.
3. **F5**, **F4** — silent-corruption bugs, cheap.
4. **F11c** — commit the smoke test to lock in 1–3 before touching anything
   else.
5. **F3** — bootstrap and held-out factor selection. Needed before any number
   leaves the repo.
6. **F6–F10** — correctness cleanup.
7. **F12** — doc repair, once the code it describes has settled.
8. **F11a / F11b** — before the next dataset or the next five methods.

One thing to do *before* any of it, because it is a single number and it
decides F2: run the real model on a pilot and report
`corr(margin_C, margin_R)`. If it is 0.99997 on Llama-3.1-8B as it is on random
weights, the correction is load-bearing and F2 becomes a narrower fix (rename
the win-rate column, move flips to raw, keep everything else). If it is much
lower, the smoke-run statistic was a null-model artifact and the corrected
margin should not be primary anywhere.
