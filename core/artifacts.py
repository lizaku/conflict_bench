"""Where a run puts everything that is not a number in a summary table.

Bare summary CSVs are not reproducible: a probe AUROC without the probe's
coefficients, or a steering flip rate without the vector that produced it,
cannot be re-audited later.  Every method writes its fitted parameters,
per-item scores and raw generations under `out_dir/artifacts/<method>/`.

Layout
------
out_dir/
  manifest.json                 config snapshot, model id, item ids, timings
  labels.csv                    behavioural labels + the generation behind each
  detection_records.jsonl       per (item, condition, method)
  steering_records.jsonl        per (item, condition, method, target, factor)
  *_summary.csv, triplet_report.csv
  activations/                  residual-stream tensors, one .npz per layer
  artifacts/
    linear_probe/fold{k}_coef.npy, fold{k}_meta.json
    diffmean_proj/fold{k}_w.npy
    act_add/vectors_fold{k}.pt  {"v_override": {l: [d]}, "v_use": {l: [d]}, ...}
    <method>/generations.jsonl
"""
import json
from pathlib import Path

import numpy as np


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def sub(self, *parts) -> "ArtifactStore":
        return ArtifactStore(self.root.joinpath(*parts))

    def path(self, name) -> Path:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ---------- writers ----------
    def json(self, name, obj):
        self.path(name).write_text(json.dumps(obj, indent=2, default=str),
                                   encoding="utf-8")
        return self.path(name)

    def array(self, name, arr):
        np.save(self.path(name), np.asarray(arr))
        return self.path(name)

    def matrix(self, name, X, index):
        """A [n, d] matrix plus the row identifiers that make it interpretable."""
        np.save(self.path(name + ".npy"), np.asarray(X))
        self.json(name + ".index.json", list(index))
        return self.path(name + ".npy")

    def torch(self, name, obj):
        import torch
        torch.save(obj, self.path(name))
        return self.path(name)

    def jsonl(self, name, rows):
        with open(self.path(name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
        return self.path(name)

    def append_jsonl(self, name, row):
        with open(self.path(name), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    # ---------- readers ----------
    def read_json(self, name, default=None):
        p = self.root / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default
