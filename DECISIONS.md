# DECISIONS.md — choices made while implementing the benchmark

Recorded 2026-11-25. The authority is `LLM Conflicts Benchmark Breakdown.docx`
("the breakdown"). Where it decides something, it is followed and the entry
says so. Where it is silent or says two things, a choice was made for now;
each entry names the one place to change it.

Status: **B** = taken from the breakdown · **Y** = your instruction ·
**D** = my decision for now, open to change.

## Definitions and labels

| # | Decision | Status | Where |
|---|---|---|---|
| 1 | A conflict needs all three conditions: `a = M(q)` stable and confident, `c` asserts `b`, `a` and `b` cannot both be true. | B | `construct/contexts.py::label` |
| 2 | **Unknown fact + context that says nothing → `irrelevant`.** The breakdown's flowchart asks "does M know?" first and would give *novel*; its text says "irrelevant means c says nothing about the fact". The text is the definition, so it wins. Swap the first two `if`s of `label()` for the flowchart reading. | D | `label()` |
| 3 | **Uncertain facts are never conflicts** (condition 1 fails, so `label()` gives *novel* or *irrelevant*). They are built through `sup / con / irr`, kept out of every main table, and **reported separately** in `*_uncertain.csv`, where they are scored as if the weak answer counted as knowledge (`label_if_known`, `conflict_q_if_known`). They are never used to train a method. | Y + D | `contexts.py::path_entry`, `evaluate.py` |
| 4 | Labels are per fact and per sentence; the three item labels (conflict for q, any conflict, which sentence asserts `b`) are derived from them, never assigned. | B | `contexts.py::emit` |
| 5 | `known` does not mean correct. `a` is M's own stable answer even when it differs from the seed's; `a_equals_gold` is recorded. | B | `beliefs.py` |

## Knowledge state (step 2)

| # | Decision | Status | Where |
|---|---|---|---|
| 6 | *Stable* = the greedy answer names the same entity on ≥ 75% of the question forms (seed question, cloze, two paraphrases). *Confident* = ≥ 80% of 10 sampled answers agree. *Unknown* = abstains, or no answer reaches 40% of samples. *Uncertain* = everything else. Starting values, to be frozen after the first real run. | D | `configs/default.yaml: beliefs` |
| 7 | An answer is identified by which entity of the relation's answer pool it names (all ConfiQA original and counterfactual answers of that relation, with aliases). | D | `beliefs.py::Clusterer` |
| 8 | A fact whose stable answer is **outside** the pool is `known` with `a_qid: null` and is **not used**: no sentence that supports or contradicts it could be written reliably. Treating it as unknown would mislabel a real conflict as novel. | D | `contexts.py`, reason "stable answer outside the answer pool" |
| 9 | Matching is on word boundaries after normalisation; names shorter than 3 characters are never used (the old matcher let the alias "IN" match "in"). Facts whose answer is shorter than that, or is contained in the subject, are rejected at the seed. | D | `construct/match.py`, `seed.py` |
| 10 | Confidence, entropy: over K sampled answers, in bits. Confidence #2: teacher-forced probability of the answer string. Both given `q` (step 2) and given `q & c` (step 4). | B (rows) / D (K = 10, T = 1) | `beliefs.py`, `measure.py` |
| 11 | The two paraphrases per relation are hand-written templates. | D | `construct/relations.yaml` |

## Contexts (step 3)

| # | Decision | Status | Where |
|---|---|---|---|
| 12 | **No model writes or checks a context.** Every sentence is a seed cloze template with a value filled in. The seed's own passages are not used (the Curie records show why: "part of the Mexico", a swapped nationality adjective instead of the language). | D | `contexts.py::statement` |
| 13 | Fillers are facts **M knows**, stated with **M's answer**, from another subject and another relation. | D | `pick_fillers` |
| 14 | Instances of one fact share the question and the fillers and differ only in the carrier sentence. | D (keeps the paired design) | `one_hop` |
| 15 | **Only single-valued relations carry instances** (17 of 40, 1966 facts). For a multi-valued relation a swapped value is not guaranteed to be incompatible with `a` (condition 3), and negation is dropped for now. The 23 others serve as fillers and as hop 1. When in doubt a relation was marked multi-valued. | D | `relations.yaml: single_valued` |
| 16 | `b` is drawn from the relation's pool, is not `a`, was never produced by M for this question, and shares no name with `a` or the subject. | D | `pick_b` |
| 17 | Conflict form is entity swap only; fact dynamics is not modelled. | Y | — |
| 18 | `nov` states the seed's answer half the time and a swapped one half the time, so "false in the world" cannot separate `con` from `nov`. | D | `one_hop` |
| 19 | Answer path k = 1 and 2. Every k = 2 item is a chain, so "length of answer path" and "reasoning structure" coincide for now. Unordered fact sets and k = 3 are not built (the seed has 934 3-hop chains). | D | `two_hop` |
| 20 | 2-hop questions are the hop-2 question with its subject replaced by a noun phrase for hop 1 ("the founder of Velorix"). | D | `relations.yaml: np` |
| 21 | Invented bridge entities (the Velorix case) are **novel by construction** and are not elicited: no model can know a made-up name. A consistent hallucination about one is not checked. | B (example) / D (no check) | `contexts.py::INVENTED` |
| 22 | Conflict on hop 1 (`2hop-con-sup`) is built only when M knows the hop-2 fact of the entity the context names, so "follow the context" has a defined answer. | D | `two_hop` |
| 23 | Position and repetition: first / middle / last are built; "stated 2–3 times" is **not** (it needs paraphrased statement templates). | D | — |
| 24 | Output format: cloze and short answer. A fact whose cloze blank is not at the end (868 of 6000) is short-answer only. Long-form is not built. | Y | `seed.py::cloze_stem` |
| 25 | Folds are whole relations, balanced on the single-valued ones. A 2-hop instance takes the fold of its last hop; its hop-1 relation may appear on both sides of a split. | D | `seed.py::fold_map` |

## Measurement and the method contract

| # | Decision | Status | Where |
|---|---|---|---|
| 26 | `d = M(q, c)` and every steered `x` come from the same greedy loop in `ModelWrapper.greedy`, not from HF `generate`, which applies whatever the model's generation config ships. | D | `core/model.py` |
| 27 | A method never sees `a`, `b`, `d` or a label in `score` / `intervene`. `fit` gets labels but no answer string. | B ("detectors see only (q, c, M)"; "π is given as input") | `api.py` |
| 28 | A steerer returns an intervention; the harness generates. | D | `api.py::Intervention`, `harness.py::apply` |
| 29 | Two controls are told the answers and are flagged `oracle`: `margin` (log p(b) − log p(a)) and `logit_lens` (the readout control for probes). They are reference rows, not methods. | D | `methods/baselines.py`, `representations.py` |
| 30 | The old relation base-rate control is dropped: on held-out relations it is a constant. `bow` remains as the text-only null. | D | — |
| 31 | The N/S/C/R conditions and the R correction are retired. Irrelevant is a labelled recipe, not an offset to subtract. | D (follows from the breakdown's four labels) | — |
| 32 | Cost is measured, not declared: every forward pass of the network is counted. A cached activation counts as free on a re-run. | B ("comes with its cost") / D | `core/model.py` |

## Evaluation

| # | Decision | Status | Where |
|---|---|---|---|
| 33 | Detection AUROC is reported against all negatives and against each negative recipe separately. | D | `evaluate.py::detection_table` |
| 34 | Global task: the same detector is given the context without the question; trainable detectors are fitted again on those views. | B (input is `(c, M)`) / D | `harness.py::run_detector` |
| 35 | Localization F1 uses a threshold chosen on the other folds. | D | `evaluate.py` |
| 36 | Success and collateral damage exactly as in the breakdown, on generated answers; "x equals t" = x names t and no other candidate answer. | B / D (the matching rule) | `evaluate.py`, `match.hit` |
| 37 | `π_mem` target: `a` for facts M knows, the context's answer for facts it does not. `π_ctx` on two-passage contexts has no target and is not scored; `π_span` is. `π_both` is not built. | B | `contexts.py` (`targets`) |
| 38 | Every strength is its own row. Choosing the strength on training folds and reporting it on held-out ones is **not implemented**. | D | — |
| 39 | Fluency of steered answers is **not measured** yet. | D | — |

## Changes to your methods when porting

| # | Change | Why |
|---|---|---|
| 40 | `p_true` asks about "the answer to this question" instead of naming the subject. | The subject is not part of the public input. |
| 41 | `prompt_instruct` says "the text" instead of "the passage above". | The instruction is placed before the context, so "above" was wrong. |
| 42 | `cad`, `adacad`, `ckplug` are logits functions applied at every decoding step, and are scored on generated answers. | They used to be scored on a teacher-forced margin only, with no generation. |
| 43 | `caa` / `act_add` layers default to depth fractions (0.55, 0.6, 0.65). | Absolute layers 17–19 were chosen for a 32-layer model. |
| 44 | `linear_probe` / `diffmean_proj` are fitted on conflict vs no-conflict instances. The probe position "last mention of the answer" is gone. | Where the answer is in the context is a label. |
| 45 | `semantic_entropy` became `sampling_entropy`, clustering by the first words of the answer. | It used the answer aliases to cluster, which a detector does not have. |
| 46 | `selfcheck_nli` and `csks` (both only ever raised `NotImplementedError`) were removed. | Dead code. |

## Not built yet

The entailment reference baseline (it needs an NLI model, and the choice of
model is yours); repetition 2–3×; k = 3; long-form output; `π_both`;
fact dynamics; numeric deviation and negation; strength selection on training
folds; fluency; natural (non-templated) passages; a second model.
