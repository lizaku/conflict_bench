# conflict_bench

A research harness for studying how a language model arbitrates between what a
prompt tells it and what it already knows. Each item pairs a question with a
passage that has been rewritten to assert a counterfactual answer, and is
scored under four conditions — no passage, a supporting passage, the
conflicting passage, and an irrelevant one that serves as a format baseline.
The dependent variable throughout is the teacher-forced log-probability margin
between the true and counterfactual answers.

On top of that design the harness runs two axes, in the shape of AxBench.

**Detection** asks *"does this passage contradict what the model already
knows?"* Each item is scored twice — once with its supporting passage, once
with its conflicting one — and the label is which of the two it was. Linear
probes and mean-difference directions are compared against prompting,
sampling and decoding-based signals, with a TF-IDF text baseline, a relation
base-rate control and a logit-lens readout control reported beside every
score. Because every item contributes one instance of each class, the base
rate is 0.5 by construction and the design is paired, so the per-item
answer-string constant cancels exactly.

**Steering** asks *"given a conflicting passage, can the intervention make the
model output the conflicting answer when it otherwise would not?"* Activation
addition (CAA) is compared against contrastive decoding methods (CAD, AdaCAD,
CK-PLUG) and an explicit prompt instruction, on a shared margin DV. Each
method runs under both the conflicting and the supporting passage: the
supporting arm is a matched specificity control on the same item, so an
effect that survives the subtraction is resolving a conflict rather than
amplifying whatever the passage said. Both targets — push toward the context,
push toward memory — are always reported as their own rows, each with the
number of items that could actually move.

Both axes fit under GroupKFold by relation, so nothing is scored on a relation
it was fitted on.

Prompts are built in one place (`core/prompts.py`) and, when the tokenizer has
a chat template, wrapped in it — an instruction-tuned model is addressed the
way it was tuned rather than in raw-completion mode. A steering instruction is
placed ahead of the passage. Both are config switches (`prompt.chat`,
`prompt.instruction_position`) rather than code.

## Data

The ConFiQA-derived corpus this runs on is **not included** in the repository.
`data.load_confiqa_cloze_clusters` expects one JSON file per Wikidata relation
under `data/ConfiQA_cloze_clustered/`; point `dataset.kwargs.data_dir` at your
own copy.

## Run

```bash
pip install torch transformers scikit-learn pandas numpy pyyaml
python -m conflict_bench.run --config conflict_bench/configs/default.yaml
python -m conflict_bench.experiments.run_pilot --limit 200
```
