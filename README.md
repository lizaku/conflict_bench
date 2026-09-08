# conflict_bench

A research harness for studying how a language model arbitrates between what a
prompt tells it and what it already knows. Each item pairs a question with a
passage that has been rewritten to assert a counterfactual answer, and is
scored under four conditions — no passage, a supporting passage, the
conflicting passage, and an irrelevant one that serves as a format baseline.
The dependent variable throughout is the teacher-forced log-probability margin
between the true and counterfactual answers.

On top of that design the harness runs two axes, in the shape of AxBench.
**Detection** asks whether it is predictable, from the model's activations or
its outputs, that a given item will end up following the context — comparing
linear probes and mean-difference directions against prompting, sampling and
decoding-based signals, with a TF-IDF text baseline, a relation base-rate
control and a logit-lens readout control reported beside every score.
**Steering** asks whether that arbitration can be pushed either way, comparing
activation addition (CAA) against contrastive decoding methods (CAD, AdaCAD,
CK-PLUG) and an explicit prompt instruction, on a shared margin DV with a
matched control. Both axes fit under GroupKFold by relation, so nothing is
scored on a relation it was fitted on.

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
