# PLAN.md — benchmark architecture: I build and evaluate, students plug in methods

Written 2026-11-25. Replaces the plan of 2026-11-20 and the old `examples/` folder.

**Status (2026-11-25): M0–M4 and most of M5–M6 are implemented** and pass the
CPU smoke test; nothing has been run on a real model yet. Paths below are the
implemented ones. Every choice made on the way, including how each open
decision of §11 was settled, is in `DECISIONS.md`.

Sources: `LLM Conflicts Benchmark Breakdown.docx` (definitions, taxonomy, tasks,
formulas — the authority; called "the breakdown" below),
`Context_Parametric_Slides-10.pdf` (student topics), the current harness, and
counts measured on `data/ConfiQA_cloze_clustered/` for this plan.

Notation is the breakdown's: `M` model, `q` question, `c` context made of
chunks `s1…sn`, `a = M(q)`, `b` the answer `c` asserts, `d = M(q, c)`,
`f1…fk` the answer path, `π` the direction, `x` the steered output.

---

## 1. Scope

| Taxonomy row | Now | Values built |
|---|---|---|
| Parametric knowledge state of each fact | **yes** | known · uncertain · unknown |
| Context vs parametric knowledge | **yes** | supports · contradicts · novel · irrelevant |
| Length of answer path | **yes** | k = 1, 2 (3 if the yield allows) |
| Reasoning structure | **yes** | 1 hop · k hops |
| Context self-consistency | **yes** | consistent · one passage agrees with M, one doesn't · passages disagree with M and each other |
| Output format | **yes** | cloze · short answer (long-form later) |
| Confidence (entropy) | **yes** | measured given `q` and given `q & c` |
| Confidence #2 (teacher-forced probability of `a`) | **yes** | measured given `q` and given `q & c` |
| Where the contradiction is found | second pass | on answer path · same entity, off path · unrelated |
| Position and repetition | second pass | first / middle / last; stated 1–3 times |
| Fact dynamics | dropped for now | every item is treated as static |
| Conflict form and strength | dropped for now | entity swap only |

"Second pass" rows cost nothing extra in the design (same builder, one more
recipe each) and two of the three detection tasks need them: the global task
needs off-path and unrelated contradictions, and localization is only
interesting when the position varies. They come after the first student
handoff.

Model: `microsoft/Phi-3.5-mini-instruct` first. Nothing below is specific to it.

## 2. The architecture in one picture

```
            MINE                                STUDENTS                     MINE
 ┌───────────────────────────┐        ┌──────────────────────────┐   ┌───────────────────┐
 │ build  (once per model)   │        │ one family per student   │   │ evaluate          │
 │ seed → beliefs → contexts │──────► │ Detector.score(inst)     │──►│ predictions       │
 │      → measure → export   │ public │ Steerer.intervene(inst,π)│   │  + labels → tables│
 └───────────────────────────┘ view   └──────────────────────────┘   └───────────────────┘
        instances.jsonl                  run only through harness.py
        (labels inside)                  (it builds prompts, generates, counts cost)
```

Three rules make the split work:

1. **A method never sees a label.** `score` and `intervene` receive a
   `PublicInstance`: `(q, c, M)` and nothing else. This is the breakdown's
   "detectors see only (question, context, model)", enforced by the type.
2. **A method never produces the scored output itself.** A detector returns
   numbers; a steerer returns an *intervention* and the harness does the
   generation. So decoding settings, answer matching, cost counting and
   fluency are identical for everyone.
3. **The predictions file is the only thing the evaluator reads.** Evaluation
   can be re-run, and re-defined, without re-running any method.

## 3. Definitions, as code

One fact, one context → one label. This function is the whole labelling logic
and it is the only place labels are assigned.

```python
def label(known: bool, gives_answer: bool, compatible: bool) -> str:
    if not gives_answer: return "irrelevant"    # c says nothing about the fact
    if not known:        return "novel"         # no a, so nothing to contradict
    if compatible:       return "supports"      # a and b can both be true
    return "contradicts"                        # the only conflict
```

- `known` — condition 1: `a = M(q)` is stable under paraphrase and confident.
- `gives_answer` — condition 2: `c` asserts `b` as an answer to `q`.
- `compatible` — condition 3 negated: `a` and `b` can both be true.

All three booleans are stored on every instance, so the label can be
re-derived under another reading without rebuilding.

The label is applied **per fact and per chunk**, which is what the breakdown
asks ("for every fact taken from c categorize it into my 4 labels"). Item
labels are then derived, never assigned by hand:

| Item label | Definition (breakdown) | Derived as |
|---|---|---|
| question-relevant conflict | `c` contradicts `a` for a fact on the answer path of `q` | any path fact is `contradicts` |
| global conflict | `c` contradicts something `M` knows, even a fact `q` does not need | any chunk is `contradicts` |
| localization target | which sentence of `c` asserts `b` | chunks that assert `b` for a path fact |

Knowledge state, from closed-book elicitation (§4.2):

| State | Rule (starting thresholds, frozen after the first run) |
|---|---|
| known | same answer on ≥ 3 of 4 question forms **and** ≥ 80% of 10 samples agree |
| unknown | abstains, or no answer reaches 40% of samples |
| uncertain | everything else; kept, labelled, reported as its own slice |

`known` does not mean correct. If `M` stably gives a wrong answer, that is its
`a`, and a passage stating the Wikidata answer is a conflict for this model.
No ground truth about the world enters any label.

`compatible` for entity swap: `b` is drawn from the answers of the same
relation, is not an alias of `a`, is not among M's sampled answers, and the
relation is single-valued. Multi-valued relations (`award received`, `genre`,
`member of`, `country of citizenship`, `position held`, `field of work`, …)
cannot give a guaranteed contradiction by swapping, and negation is dropped
for now, so they are used as fillers and hop-1 facts only.

## 4. Generation: five steps, five files

Every step reads one file and writes one file. All files are JSONL and
readable by eye. Nothing is dropped silently: each step appends to
`rejected.jsonl` with the step name and a reason.

```
construct/seed.py      data/ConfiQA_cloze_clustered → bench/_seed/facts.jsonl   no model   once
construct/beliefs.py   facts.jsonl + M              → beliefs.jsonl             GPU        per model
construct/contexts.py  beliefs.jsonl + recipes      → contexts.jsonl            no model   per model
construct/measure.py   contexts.jsonl + M           → measured.jsonl            GPU        per model
construct/export.py    all of the above             → instances.jsonl, public.jsonl, card.md
construct/build.py     runs the five in order: python -m conflict_bench.construct.build --model <hf-id>
```

Output goes to `bench/<model>/`. It is gitignored: the folder is shared
directly, or rebuilt. Steps 1, 3 and 5 need no GPU, so contexts can be rebuilt and
inspected on the dev machine from a `beliefs.jsonl` produced elsewhere.

**No language model writes any context, and no NLI model checks one.** Every
chunk is a seed cloze template with a value filled in, so what a chunk asserts
is known by construction. That is the main simplification against the old
plan, and the reason the process can be audited by reading it.

### 4.1 Seed (`seed.py`, reuses `data.load_confiqa_cloze_clusters`)

One `Fact` per ConfiQA record: `fact_id, subject, subject_qid, relation,
question, cloze, gold, gold_aliases, answer_pool`. The seed's passages are not
used; only its facts and templates are.

Measured on the seed (6000 facts, 40 relations, 3645 subjects):

| Property | Count | Used for |
|---|---|---|
| `cloze_template` filled with the answer equals `orig_statement` | 6000 / 6000 | `statement(fact, x)` is one string replace |
| cloze with the blank at the end | 5132 | the cloze output format; the other 868 are short-answer only |
| subject string appears in the question | 5540 | composing 2-hop questions |
| subjects with ≥ 2 relations | 1683 | same entity, off path |
| 2-hop chains (object of one fact is the subject of another) | 2383 | real multi-hop items |
| 3-hop chains | 934 | k = 3 |

`construct/relations.yaml` adds, by hand, once, 40 rows:

```yaml
P36:  {single_valued: true,  subject_type: country, np: "the capital of {s}",
       paraphrases: ["Which city is the capital of {s}?", "Name the capital city of {s}."]}
P166: {single_valued: false, subject_type: person}
```

Also here: the alias matcher (`construct/match.py`), word-boundary and minimum
length 3, returning `a | b | both | neither`. Every later step depends on it,
and the current substring match is known to be wrong (`IDEAS.md` F1).

### 4.2 Beliefs (`beliefs.py`) — what does M know?

Closed-book, per fact, through `core/prompts.py`:

| Signal | How | Feeds |
|---|---|---|
| paraphrase stability | greedy answer to 4 forms: seed question, cloze, 2 paraphrases | condition 1, "stable" |
| entropy given `q` | 10 samples at T = 1, clustered by alias match, entropy in bits | condition 1, "confident"; taxonomy row *Confidence* |
| teacher-forced `p(a | q)` | `model.logp_continuation` on the modal answer | taxonomy row *Confidence #2* |
| abstention | refusal pattern | unknown |

`a` is the modal answer. All raw signals are stored, so "known" can later be
re-defined (self-report, entropy only, log-prob margin) without touching the
model — that is the "how stable is the known label" student topic for free.

### 4.3 Contexts (`contexts.py`) — one recipe per taxonomy cell

A context is a list of chunks. A chunk is one sentence that asserts one value
for one fact:

```
chunk = {text, passage: "A", fact_id, asserts: <value>, label: <4-way, relative to M>}
```

Two chunk sources only:

- **carrier** — `statement(f, x)` for a path fact `f`: `x = a` supports,
  `x = b` contradicts, any value is novel if `f` is unknown.
- **filler** — `statement(g, a_g)` for another subject's fact `g` that **M
  knows**, stated with M's own answer. Fillers are therefore always
  `supports` for their own fact and say nothing about `q`. A filler taken
  from Wikidata instead of from M's beliefs could contradict M by accident
  and create an unlabelled global conflict.

Fillers are sampled once per fact and reused in every instance of that fact,
so instances of the same fact differ **only in the carrier**. This keeps the
paired design of the current harness (same question, same surroundings, base
rate 0.5 for contradicts vs supports).

The recipe list (`contexts.recipes` in the config, one function each in
`construct/contexts.py`) is what gets built. It is short enough to read in
full, and it is the coverage table of the taxonomy:

| Recipe | Needs | Chunks | Path labels |
|---|---|---|---|
| `sup` | known `f` | F · carrier(a) · F · F | supports |
| `con` | known `f` | F · carrier(b) · F · F | **contradicts** |
| `irr` | any `f` | F · F · F · F | irrelevant |
| `nov` | unknown `f` | F · carrier(gold or swapped, 50/50) · F · F | novel |
| `2hop-sup-con` | known `f1`, known `f2`, real chain | carrier(f1, a1) · carrier(f2, b2) · F · F | supports, **contradicts** |
| `2hop-sup-sup` | same | carrier(f1, a1) · carrier(f2, a2) · F · F | supports, supports |
| `2hop-con-sup` | same; `b1` has a known `f2` | carrier(f1, b1) · F · F · F | **contradicts**, irrelevant |
| `2hop-nov-con` | invented bridge + known `f2` | carrier(f1, subject of f2) · carrier(f2, b2) · F · F | novel, **contradicts** |
| `2hop-nov-sup` | same | carrier(f1, …) · carrier(f2, a2) · F · F | novel, supports |
| `two-agree-disagree` | known `f` | A: `sup` passage ‖ B: `con` passage (order counterbalanced) | **contradicts** |
| `two-disagree-disagree` | known `f` | A: carrier(b) ‖ B: carrier(b′) | **contradicts** |
| second pass: `off-path`, `unrelated`, `extra-true`, `pos-first/last`, `rep-2/3` | | | |

Every recipe is emitted in both output formats where the cloze allows it.
Uncertain facts go through `sup / con / irr` unchanged and carry
`state: uncertain`.

`nov` asserts the gold answer half the time and a swapped one half the time,
so "is the claim false in the world" cannot separate `con` from `nov`. Only
M's knowledge separates them, which is the contrast the benchmark is about.

**2-hop questions** are composed mechanically: take the question of `f2` and
replace its subject by the noun phrase of `f1` from `relations.yaml`
("Where was *Marie Curie* born?" → "Where was *the founder of Velorix* born?").
Two bridge sources:

- *real chains* from the seed (2383 available). Hop 1 is known to M.
- *invented bridges*, the breakdown's Velorix case: a made-up entity from a
  fixed name list plus a bridge template per subject type
  (`person: "{X} was founded by {s}"`). Hop 1 is novel by construction and
  the name is checked in step 2 like any fact — it must come out `unknown`.
  This is the only recipe where `π_mem` is not the same as deleting the
  context.

Checks inside this step (strings only): the asserted value is in the carrier;
no alias of `a` appears in a `con` context and no alias of `b` in a `sup` one;
fillers mention neither the subject, `a` nor `b`; `b` satisfies §3.

### 4.4 Measure (`measure.py`) — the model's own behaviour on each instance

Adds to every instance, for this `M`, with no method applied:

- `d = M(q, c)`, greedy, and what it matches (`a | b | both | neither`);
- entropy given `q & c` (10 samples, same clustering as step 2);
- teacher-forced `p(a | q, c)` and `p(b | q, c)`.

None of this is a gate. Whether `M` follows the context is the phenomenon, so
no instance is dropped for its `d`. `d` is needed by the steering formulas
(§7.2) and by the with/without baseline.

### 4.5 Export (`export.py`)

- `public.jsonl` — what methods receive (§6.1). Instance ids are opaque
  hashes, so the id does not reveal the recipe.
- `instances.jsonl` — the same instances with everything else: labels,
  answers, targets, M's behaviour.
- `card.md` — the funnel (seed → state counts → instances per recipe), the
  rejection reasons, and 3 random instances per recipe printed in full for a
  manual audit.

## 5. Instance schema

```json
{
  "instance_id": "7c1e09a4",
  "model": "microsoft/Phi-3.5-mini-instruct",
  "fold": 2,
  "recipe": "con", "format": "short",
  "question": "Where was Marie Curie born?",
  "passages": [{"id": "A", "chunks": [
    {"text": "The capital of Ecuador is Quito.",        "fact_id": "P36:0",  "asserts": "Quito",   "label": "supports"},
    {"text": "Marie Curie was born in Austria.",        "fact_id": "X:0",    "asserts": "Austria", "label": "contradicts"},
    {"text": "The currency of Japan is the yen.",       "fact_id": "P38:7",  "asserts": "yen",     "label": "supports"},
    {"text": "The author of Jujutsu Kaisen is Gege Akutami.", "fact_id": "P50:0", "asserts": "Gege Akutami", "label": "supports"}]}],
  "path": [{"fact_id": "X:0", "state": "known", "a": "Poland", "b": "Austria",
            "known": true, "gives_answer": true, "compatible": false, "label": "contradicts"}],
  "labels": {"conflict_q": 1, "conflict_global": 1, "chunk_asserts_b": [0, 1, 0, 0]},
  "features": {"k": 1, "structure": "1 hop", "self_consistency": "consistent",
               "location": "on answer path", "position": "middle", "repetition": 1},
  "targets": {"ctx": "Austria", "mem": "Poland"},
  "confidence": {"entropy_q": 0.0, "p_a_q": 0.93,
                 "entropy_qc": 0.72, "p_a_qc": 0.11, "p_b_qc": 0.64},
  "d": {"text": "Austria", "matches": "b"}
}
```

`public.jsonl` keeps only `instance_id, model, fold, format, question,
passages[].id, passages[].chunks[].text`.

## 6. The student contract

### 6.1 Input

```python
@dataclass(frozen=True)
class PublicInstance:
    instance_id: str
    question: str | None        # None in the global task: the input is (c, M)
    passages: list[Passage]     # Passage(id, chunks: list[str])
    format: str                 # "cloze" | "short"
    prompt: str                 # built by core/prompts.py, chat template applied
    prompt_no_context: str      # the same question with no context
    chunk_spans: list[tuple]    # character span of every chunk inside `prompt`
```

plus `M`, the existing `ModelWrapper` (generate, log-probs, residual stream,
hooks). A method may call `M` as much as it likes; the harness counts the
forward passes.

Methods that need training get `fit(train)`, where `train` is a list of
`(PublicInstance, labels)` from the training folds only. Folds are assigned
by relation, as now.

### 6.2 Output

```python
class Detector:
    access = "activations"                        # "text" | "logits" | "activations"
    def fit(self, train): ...                     # optional
    def score(self, inst) -> Detection: ...

Detection(conflict: float,                        # higher = more conflict
          chunks: list[float] | None = None,      # one score per chunk, for localization
          label: str | None = None)               # optional 4-way guess

class Steerer:
    strengths = [1.0]                             # the dial; a single value if there is none
    def fit(self, train): ...                     # optional
    def intervene(self, inst, direction, strength, span=None) -> Intervention: ...

Intervention(prompt: str | None = None,           # a rewritten prompt (prompting)
             hooks: list = (),                    # forward hooks (CAA, SpARE, PH3, JUICE, …)
             logits_processor = None)             # per-step logit edit (CAD, AdaCAD, CK-PLUG, …)
```

`direction` is `"ctx" | "mem" | "span"`; `span` names the designated passages
for `π_span`. A steerer never sees `a`, `b` or `d`: "π is given as input".

The harness then, identically for every method:

- detection: calls `score(inst)` for the question-relevant task and
  `score(inst without question)` for the global task;
- steering: installs the intervention, generates greedily with a fixed token
  budget → `x`; also teacher-forces `a` and `b` under the same intervention
  for the dose-response curve;
- writes `predictions/<method>.detection.jsonl` /
  `predictions/<method>.steering.jsonl`, one row per instance (× direction ×
  strength), with the forward-pass count and the access level.

### 6.3 Who writes what

| File | Owner | Content |
|---|---|---|
| `methods/baselines.py` | me | every baseline of the breakdown (§7) + `bow` + delete-context oracle |
| `methods/contrastive.py` | student | detect: JSD, entropy, confidence gain · steer: CAD, AdaCAD, COIECD, CK-PLUG |
| `methods/representations.py` | student | detect: linear probes, SAE features, logit lens · steer: CAA / ActAdd, RepE, SpARE |
| `methods/components.py` | student | detect: activation patching, head attribution · steer: PH3, JUICE, IRCAN |

My own versions of the first two families already exist
(`methods/detectors/detectors.py`, `methods/steerers/steerers.py`); they are
ported to the new interface first and serve as the worked examples students
start from.

## 7. Evaluation (`evaluate.py`)

### 7.1 Detection

| Task | Input → output | Positives | Metric |
|---|---|---|---|
| Question-relevant conflict | `(q, c, M)` → score | `conflict_q = 1` | AUROC |
| Global knowledge conflict | `(c, M)` → score | `conflict_global = 1` | AUROC |
| Localization | `(q, c, M)` → score per sentence | chunks with `chunk_asserts_b = 1` | sentence-level AUROC, top-1 hit, F1 |

- AUROC is reported for `con` against **each** negative recipe separately
  (`sup`, `nov`, `irr`, …) and then pooled. `con` vs `irr` is easy; `con` vs
  `nov` is the real test, and a pooled number hides that.
- F1 uses a threshold fitted on the training folds (`metrics._best_threshold`).
- If a method returns the optional 4-way `label`: confusion matrix and macro-F1.
- Every row is also broken down by taxonomy feature (state, k, self-consistency,
  format) and by terciles of the two confidence measures.
- Every row carries its cost: forward passes per instance and access level.
- Next to each detector: its agreement with the entailment reference, since
  detectors "are compared on how close they get to the reference pipeline, at
  what cost, and on which rows of the taxonomy".

Baselines (mine):

| Baseline | Implementation | Known failure |
|---|---|---|
| with/without context | `a` vs `d` differ | fires on novel items; misses conflicts where `d = a` |
| self-report | ask `M` whether the context contradicts what it knows (existing `p_true`) | — |
| entailment (reference) | get `a = M(q)`, check `c` against `statement(a)` with an NLI model; three outcomes kept apart: contradicts / entails / neutral | cost; a second model; multi-hop and many-claim contexts |
| text only | existing `bow` | must be at chance on `con` vs `nov` |

Paraphrase consistency is not a detector: it measures condition 1 and is used
in step 2.

### 7.2 Steering

Three runs: `M(q) → a`, `M(q, c) → d`, `M(q, c, method, π) → x`. With `t` the
target of `π` on the item:

```
success(π)    = #{ d ≠ t and x = t } / #{ d ≠ t }
collateral(π) = #{ d = t and x ≠ t } / #{ d = t }
```

Both always reported with their denominators. "`x = t`" is decided by
`construct/match.py`: `x` matches `t` and not the other answer.

Targets are computed at build time and stored on the instance:

| Direction | Target |
|---|---|
| `π_ctx` | `b`: every hop is taken from the context where the context asserts it |
| `π_mem` | `a` for facts `M` knows; the context for facts it does not |
| `π_span` | what the designated passages assert |
| `π_both` | optional, not built now |

The main steering set is the breakdown's: instances where `c` asserts `b` and
`b ≠ a`. The same two formulas are also reported on `sup`, `nov` and `irr`
where a target exists, as the check that a method does not change answers
when there is nothing to resolve (today's S arm, generalised). `π_ctx` on the
two-passage recipes has no single target and is not scored; `π_span` is.

Baselines (mine):

| Family | `π_ctx` | `π_mem` | `π_span` |
|---|---|---|---|
| Prompting | "answer only from the context" | "ignore the context" | "use only passage A" |
| Input oracle | — | delete the context | delete the other passages |

On 1-hop items deleting the context is the ceiling for `π_mem`. On
`2hop-nov-*` it fails by construction, which is what makes that recipe the
real `π_mem` test.

Strength is chosen on the training folds and reported on held-out ones.
Secondary, for dose-response plots only: the change in
`log p(a) − log p(b)` under the intervention.

### 7.3 Kept from the current harness

Folds by relation for both tasks; the `bow` row; the base-rate row; the
`logit_lens` readout control at a probe's own layer and position; a matched
control for every steering effect (random direction of equal norm, placebo
instruction, unsteered distribution); prompts only from `core/prompts.py`;
both directions as separate rows with their `n`.

## 8. Marie Curie, through every row

Fact: *Marie Curie — place of birth — Poland*. Birthplace is not one of the 40
seed relations (the seed's Curie records are native language, field of work
and citizenship); it is used here because it is the breakdown's own example.
The builder produces the same rows from the seed record `P103` with the
template "The native language of Marie Curie is ____.". **All numbers in this
section are illustrative, not measured.**

Fillers, each a fact model A knows, stated with A's own answer:
**F1** "The capital of Ecuador is Quito." · **F2** "The currency of Japan is
the yen." · **F3** "The author of Jujutsu Kaisen is Gege Akutami." ·
**F4** "The head of state of Spain is Felipe VI."

### 8.1 Knowledge state (step 2)

| Fact | Question | What the model does closed-book | State | `a` |
|---|---|---|---|---|
| birthplace, model A | Where was Marie Curie born? | 4/4 forms "Poland"; 10/10 samples; entropy 0.0 bits; `p(Poland | q)` = 0.93 | **known** | Poland |
| doctoral advisor, model A | Who was Marie Curie's doctoral advisor? | forms: Lippmann 2, Becquerel 2; samples 5 / 3 / 2; entropy 1.5 bits | **uncertain** | Lippmann (weak) |
| father, model A | Who was Marie Curie's father? | abstains on 2 forms; 7 different names in 10 samples; entropy 2.6 bits | **unknown** | — |
| birthplace, model B | Where was Marie Curie born? | Poland 4/10, France 3/10, Russia 3/10; entropy 1.6 bits | **unknown** | — |

Choosing `b` for model A: *Warsaw* rejected (alias of `a`); *Russian Empire*
rejected (compatible: Warsaw was in the Russian Empire in 1867); **Austria**
accepted.

### 8.2 Context vs parametric knowledge

Question: "Where was Marie Curie born?" unless stated.

| Recipe | Context | known / gives answer / compatible | Label | conflict for q | any conflict |
|---|---|---|---|---|---|
| `sup` | F1 · **Marie Curie was born in Poland.** · F2 · F3 | yes / yes / yes | supports | 0 | 0 |
| `con` | F1 · **Marie Curie was born in Austria.** · F2 · F3 | yes / yes / no | **contradicts** | 1 | 1 |
| `irr` | F1 · F4 · F2 · F3 | yes / no / – | irrelevant | 0 | 0 |
| `nov` (true claim), q: "Who was Marie Curie's father?" | F1 · **The father of Marie Curie is Władysław Skłodowski.** · F2 · F3 | no / yes / – | novel | 0 | 0 |
| `nov` (false claim), same q | F1 · **The father of Marie Curie is Jan Nowak.** · F2 · F3 | no / yes / – | novel | 0 | 0 |
| `irr` on the unknown fact, same q | F1 · F4 · F2 · F3 | no / no / – | irrelevant | 0 | 0 |
| `con` text, **model B** | identical to `con` above | no / yes / – | novel | 0 | 0 |
| `con` on the uncertain fact, q: advisor | F1 · **The doctoral advisor of Marie Curie is Henri Poincaré.** · F2 · F3 | no (uncertain) / yes / – | novel, `state: uncertain` | 0 | 0 |

Row 7 is the reason the dataset is a pipeline: the same text is a conflict for
model A and novel for model B.

### 8.3 Length of answer path and reasoning structure

| Recipe | Question | Context (path chunks) | Path labels | `π_ctx` | `π_mem` | Delete context |
|---|---|---|---|---|---|---|
| k = 1 | Where was Marie Curie born? | Marie Curie was born in Austria. | contradicts | Austria | Poland | Poland |
| `2hop-nov-con` | Where was the founder of Velorix born? | Velorix was founded by Marie Curie. · Marie Curie was born in Austria. | novel, **contradicts** | Austria | Poland | fails: M cannot know the founder |
| `2hop-nov-sup` | same | Velorix was founded by Marie Curie. · Marie Curie was born in Poland. | novel, supports | Poland | Poland | fails |
| `2hop-sup-con` | Where was the spouse of Pierre Curie born? | The spouse of Pierre Curie is Marie Curie. · Marie Curie was born in Austria. | supports, **contradicts** | Austria | Poland | Poland |
| `2hop-sup-sup` | same | The spouse of Pierre Curie is Marie Curie. · Marie Curie was born in Poland. | supports, supports | Poland | Poland | Poland |
| `2hop-con-sup` | same | The spouse of Pierre Curie is Lise Meitner. | **contradicts**, irrelevant | Austria (hop 1 from the context, hop 2 from memory: Meitner was born in Austria) | Poland | Poland |

`2hop-con-sup` needs `b1` (Lise Meitner) to have a hop-2 fact that M knows;
otherwise the `π_ctx` target is undefined and the instance is not built.

### 8.4 Context self-consistency

| Recipe | Context | conflict for q | Localization | `π_span = A` | `π_span = B` | `π_mem` |
|---|---|---|---|---|---|---|
| consistent | the `con` row of 8.2 | 1 | chunk 2 | — | — | Poland |
| one passage agrees with M, one doesn't | **A:** F1 · Marie Curie was born in Poland. · F2 ‖ **B:** F3 · Marie Curie was born in Austria. · F4 | 1 | B, chunk 2 (not A, chunk 2) | Poland | Austria | Poland |
| passages disagree with M and each other | **A:** F1 · Marie Curie was born in Austria. · F2 ‖ **B:** F3 · Marie Curie was born in Hungary. · F4 | 1 | A chunk 2 and B chunk 2 | Austria | Hungary | Poland |

### 8.5 Output format

| Format | What the model sees after the context | Expected continuation |
|---|---|---|
| cloze | Marie Curie was born in | Poland / Austria |
| short answer | Where was Marie Curie born? | Poland / Austria |

### 8.6 Confidence (measured, never edited)

| Measure | Given `q` | Given `q & c` on `sup` | Given `q & c` on `con` |
|---|---|---|---|
| entropy over 10 sampled answers (bits) | 0.0 | 0.0 | 0.72 (Austria 8, Poland 2) |
| teacher-forced `p(a = Poland)` | 0.93 | 0.98 | 0.11 |
| teacher-forced `p(b = Austria)` | 0.001 | 0.0005 | 0.64 |

### 8.7 Second pass: location, position, repetition

| Row | Context | conflict for q | any conflict |
|---|---|---|---|
| on answer path | F1 · Marie Curie was born in Austria. · F2 · F3 | 1 | 1 |
| same entity, off path | F1 · Marie Curie was born in Poland. · F2 · **The spouse of Marie Curie is Albert Einstein.** | 0 | 1 |
| unrelated | F1 · Marie Curie was born in Poland. · F2 · **The capital of Belgium is Paris.** | 0 | 1 |
| extra true statement (control) | F1 · Marie Curie was born in Poland. · F2 · The spouse of Marie Curie is Pierre Curie. | 0 | 0 |
| position first / last | carrier moved to slot 1 / slot 4 of the `con` row | 1 | 1 |
| stated 2 / 3 times | 1 / 2 fillers of the `con` row replaced by paraphrased carriers ("Marie Curie's birthplace is Austria.") | 1 | 1 |

The off-path and unrelated facts must themselves be known to M (spouse =
Pierre Curie, capital of Belgium = Brussels), or the chunk is novel and not a
contradiction.

Dropped for now, so no example: fact dynamics, numeric deviation, negation,
long-form output, `π_both`.

### 8.8 What a method receives and returns on the `con` row

Detector input: the question, the four sentences, `M`. Output, for example:
`Detection(conflict=0.91, chunks=[0.03, 0.88, 0.05, 0.02])`. Scored as a
positive for both AUROC tasks; localization top-1 is a hit (chunk 2).

Steerer input: the same, plus `direction="mem"`, `strength=2.0`. Output: an
`Intervention`. The harness generates `x`. Suppose `d = "Austria"`:

- `π_mem`, `t = Poland`: `d ≠ t`, so the item is in the success denominator;
  `x = "Poland"` is a success.
- `π_ctx`, `t = Austria`: `d = t`, so the item is in the collateral
  denominator; `x = "Poland"` would be collateral damage.

## 9. What happens to the existing code

| Existing | Becomes | Change |
|---|---|---|
| `core/model.py` | kept | add a forward-pass counter; `generate` accepts hooks and a logits processor |
| `core/prompts.py` | kept, single source of prompts | build from an instance (one or several passages) instead of `Item` + `Condition`; return one span per chunk; add a context-only prompt for the global task |
| `core/positions.py` | kept | named positions gain `end_of_chunk[i]` from the chunk spans |
| `core/activations.py`, `core/artifacts.py` | kept | cache key = model + `instance_id` (`IDEAS.md` F5) |
| `data.py` loader | `construct/seed.py` | passages no longer read; R-passage synthesis removed |
| `run.behavioural_labels` | `construct/match.py` + `measure.py` | word-boundary matcher, four outcomes |
| `semantic_entropy` sampling and clustering | `construct/beliefs.py`, `measure.py` | same code, now a measurement, not a detector |
| `core/margins.py` | `measure.py` | teacher-forced scoring kept; the N/S/C/R table and the R correction are retired — irrelevant is a labelled recipe, not an offset |
| `core/types.py` | `api.py` + plain JSON records | `Item`/`Condition` → `PublicInstance`; facts, beliefs and instances are dicts in JSONL |
| `methods/base.py` | `api.py` | the two interfaces of §6.2; registry kept |
| `run.py`, `experiments/run_pilot.py` | `harness.py` | one straight line: load → fit on train folds → score/steer held-out → predictions file |
| `metrics.py` | `evaluate.py` | `grouped_auroc`, `_best_threshold` kept; flip rates replaced by §7.2 |
| `detectors.py`, `steerers.py` | `methods/baselines.py`, `contrastive.py`, `representations.py` | ported; decoding steerers become logits processors, which also closes the "no generation, no fluency" gap |

The old runner stays runnable until the new path produces its first table,
then it is removed and `CLAUDE.md` is rewritten (its conditions, conventions
and R-correction sections describe the retired design).

## 10. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M0 | `match.py`, `relations.yaml`, `seed.py` | `facts.jsonl` for 6000 facts; the §8 tables reproduced by a CPU test from hand-written beliefs |
| M1 | `beliefs.py` on Phi-3.5-mini | known / uncertain / unknown rates per relation; thresholds frozen |
| M2 | `contexts.py` core recipes (`sup con irr nov`), `measure.py`, `export.py` | `card.md` read by hand, 30 instances per recipe |
| M3 | `api.py`, `harness.py`, `evaluate.py`, `baselines.py` | detection and steering tables for the baselines — **student handoff** |
| M4 | existing methods ported as worked examples | contrastive and representation families run end to end |
| M5 | 2-hop and two-passage recipes, `π_span` | §8.3 and §8.4 rows exist for real items |
| M6 | second pass: location, position, repetition | global and localization tasks scored |
| later | dynamics, form and strength, long-form, second model | |

The CPU smoke habit stays: tiny random Llama, with `beliefs.jsonl` written by
hand, since a random model knows nothing and every fact would come out
`unknown`.

## 11. Open decisions

*All eight were settled for now on 2026-11-25; see `DECISIONS.md` (#2, #3,
#19, #15, #12, #25, the `bench/` folder, #6).*

1. **Unknown fact, context says nothing.** The breakdown's flowchart asks
   "does M know?" first and would call this *novel*; its text defines
   *irrelevant* as "c says nothing about the fact". §3 follows the text.
   Swapping the first two lines of `label()` gives the flowchart.
2. **Uncertain facts.** Condition 1 fails, so `label()` returns *novel* for
   them. They keep `state: uncertain` and are reported separately. The
   alternative is to leave them unlabelled.
3. **Answer path vs reasoning structure.** The taxonomy has both rows and the
   breakdown leaves open whether order matters. Every k > 1 item built here is
   a chain, so the two coincide. Unordered sets ("Who was born earlier, Curie
   or Einstein?") would need comparison questions the seed does not have.
4. **Multi-valued relations** are excluded from `con` (§3). That removes the
   largest relations from the conflict cells. Negation would bring them back,
   and it is dropped for now.
5. **Templated sentences vs natural passages.** Contexts are lists of template
   sentences. Natural ConfiQA passages can be added later as one more recipe
   for the 4204 records whose counterfactual passage is an exact answer swap,
   valid only where `a` equals the gold answer.
6. **Folds for 2-hop items.** An instance takes the fold of its last-hop
   relation; the hop-1 relation may then appear on both sides of a split.
7. **Where `bench/<model>/` lives.** It has to be shared with students and
   must not sit under a gitignored folder; it contains text derived from
   ConfiQA, which `.gitignore` currently treats as not redistributable.
8. **Thresholds for known / unknown** (§3) are starting values.
