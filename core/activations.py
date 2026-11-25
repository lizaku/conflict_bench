"""Shared residual-stream cache.

Re-extracting activations dominates runtime, and several methods want the same
tensors (linear_probe, diffmean_proj, logit_lens, the CAA vector fit).  So
extraction happens once, through one cache, and is persisted.

On disk: one `layer{L}.npz` per layer, keyed `"{inst.key}@{position}"`.  The
cache directory lives under the model's own benchmark folder, so a model swap
can never reuse another model's tensors.
"""
from pathlib import Path

import numpy as np


class ActivationCache:
    def __init__(self, model, cache_dir=None, autoflush_every=256):
        self.model = model
        self.dir = Path(cache_dir) if cache_dir else None
        self._mem: dict[int, dict[str, np.ndarray]] = {}
        self._dirty: set[int] = set()
        self._since_flush = 0
        self.autoflush_every = autoflush_every
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(self.dir.glob("layer*.npz")):
                with np.load(f) as z:
                    self._mem.setdefault(int(f.stem[5:]), {}).update(
                        {k: z[k] for k in z.files})

    def flush(self):
        if not self.dir:
            return
        for layer in sorted(self._dirty):
            np.savez_compressed(self.dir / f"layer{layer}.npz",
                                **self._mem[layer])
        self._dirty.clear()
        self._since_flush = 0

    def get_many(self, inst, layers, position="last") -> dict:
        """{layer: vector} for one instance view. `layers` may be depth
        fractions; keys of the result are the values passed in."""
        from conflict_bench.core import positions
        resolved = {l: self.model.resolve_layer(l) for l in layers}
        k = f"{inst.key}@{position}"
        missing = sorted({r for r in resolved.values()
                          if k not in self._mem.get(r, {})})
        if missing:
            idx = positions.resolve(self.model, inst, position)
            acts = self.model.residual_at(inst.prompt, missing, idx)
            for r, v in acts.items():
                self._mem.setdefault(r, {})[k] = np.asarray(v, dtype=np.float32)
                self._dirty.add(r)
            self._since_flush += 1
            if self.autoflush_every and self._since_flush >= self.autoflush_every:
                self.flush()
        return {l: self._mem[r][k] for l, r in resolved.items()}

    def get(self, inst, layer, position="last") -> np.ndarray:
        return self.get_many(inst, [layer], position)[layer]

    def stack(self, insts, layer, position="last") -> np.ndarray:
        return np.stack([self.get(i, layer, position) for i in insts])
