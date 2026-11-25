# conflict_bench

A benchmark for **knowledge conflicts between a model's memory and its
context**: detecting them, and steering which source the model follows.

A conflict is defined relative to the model `M`. Three conditions must hold:

1. `a = M(q)` is stable and confident — the model knows the fact;
2. the context `c` asserts an answer `b` to `q`;
3. `a` and `b` cannot both be true.

So the benchmark is not a fixed dataset. It is a pipeline that is run once per
model: find out what `M` knows, then build contexts by controlled edits so
that every one lands in a known cell of the taxonomy (supports · contradicts ·
novel · irrelevant; 1 or 2 facts on the answer path; one or two passages;
cloze or short answer). No ground truth about the world is used.

Three parts, with a fixed contract between them:

| Part | Code | Who |
|---|---|---|
| **Construction** — builds every instance for a model | `construct/` | benchmark |
| **Methods** — detectors and steerers, one file per family | `methods/` | students |
| **Evaluation** — runs methods, scores predictions | `harness.py`, `evaluate.py` | benchmark |

A method receives the question, the context sentences and the model. It never
receives a label or an answer. A detector returns a score; a steerer returns
an *intervention* and the harness generates the answer.

`PLAN.md` has the design and a worked example (Marie Curie) for every row of
the taxonomy. `DECISIONS.md` records every choice made while implementing it.
`CLAUDE.md` is the working notes on the code.

## Run

Run from the **parent** of this folder (the code imports itself as
`conflict_bench.*`).

```bash
pip install torch transformers scikit-learn pandas numpy pyyaml

# 1. build the benchmark for a model  ->  conflict_bench/bench/<model>/
python -m conflict_bench.construct.build --model microsoft/Phi-3.5-mini-instruct

# 2. run the methods on ~100 instances and evaluate
python -m conflict_bench.harness --model microsoft/Phi-3.5-mini-instruct --limit 100
```

Sizes:

| Command | What it does |
|---|---|
| `build` | all 6000 seed facts |
| `build --limit 300` | a pilot: 300 seed facts spread over the relations |
| `build --measure-limit 500` | build all contexts, run the model on about 500 of them |
| `build --steps contexts,export` | rebuild contexts after editing a recipe (no GPU) |
| `harness --limit 100` | about 100 instances of the four core recipes |
| `harness --limit 1000 --recipes all` | every recipe that was built |
| `harness --detectors bow,p_true --steerers none` | a subset of methods |

Both GPU steps of the build append as they go; re-running the same command
resumes.

## What is in `bench/<model>/`

| File | Content |
|---|---|
| `beliefs.jsonl` | per seed fact: the answers M gave, its state (known / uncertain / unknown), entropy, probability of `a` |
| `instances.jsonl` | the benchmark: sentences with their labels, answers, targets, and M's own behaviour (`d`, confidence with context) |
| `public.jsonl` | what a method sees: question, sentences, format |
| `card.md` | the funnel, the rejections, and 3 examples per recipe to read by hand |
| `rejected.jsonl` | every fact that produced no instance, with the reason |
| `runs/<name>/` | predictions, `detection.csv`, `localization.csv`, `steering.csv`, `report.md` |

`bench/` is gitignored. Share the folder, or rebuild it.

## Adding a method

```python
# methods/components.py
from conflict_bench.api import Detection, Detector, Intervention, Steerer, \
    register_detector, register_steerer

@register_detector("my_detector")
class MyDetector(Detector):
    access = "activations"                  # "text" | "logits" | "activations"
    def score(self, inst):                  # inst.question, inst.chunks, inst.prompt
        return Detection(conflict=0.7, chunks=[0.1, 0.9, 0.0, 0.1])

@register_steerer("my_steerer")
class MySteerer(Steerer):
    strengths = [1.0, 2.0]
    def intervene(self, inst, direction, strength, span=None):   # "ctx" | "mem" | "span"
        return Intervention(hooks=lambda: self.model.add_direction(...))
```

Import the file in `methods/__init__.py` and list the names under `detectors:`
/ `steerers:` in `configs/default.yaml`. A method that needs training defines
`fit(train)` and sets `requires_training = True`; it is then fitted on the
training folds only (folds are whole relations).

## Test without a GPU

```bash
python -m conflict_bench.tests.smoke
```

A tiny randomly-initialised model through every step, plus a check of the
construction's invariants on the whole seed. Needs one cached tokenizer with a
chat template (`--tokenizer`).

## Data

The ConFiQA-derived seed is **not included**. The build expects one JSON file
per Wikidata relation under `data/ConfiQA_cloze_clustered/` (`seed_data` in
the config).

## History

`git checkout milestone-1-nscr-pilot` is the previous benchmark: fixed ConfiQA
items scored under four passage conditions (N/S/C/R), with margin-flip
steering.
