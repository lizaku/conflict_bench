"""Shared residual-stream cache.

Re-extracting activations dominates runtime (the v3 lesson in the probe
docstring), and three methods want the *same* tensors: linear_probe,
diffmean_proj and act_add's vector fit.  So extraction happens once, through
one cache, and the tensors are persisted — both to make a re-run cheap and
because the tensors themselves are a deliverable of the mechanistic arm.

On disk: one `layer{L}.npz` per layer under `out_dir/activations/`, keyed
`"{item_id}|{condition}"`.  A run that is re-launched with the same out_dir
picks the cache up and only extracts what is missing.
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
        self.n_extracted = 0
        self.n_hits = 0
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.load()

    # ---------- persistence ----------
    def load(self):
        if not self.dir:
            return
        for f in sorted(self.dir.glob("layer*.npz")):
            layer = int(f.stem.replace("layer", ""))
            with np.load(f) as z:
                self._mem.setdefault(layer, {}).update(
                    {k: z[k] for k in z.files})

    def flush(self):
        if not self.dir:
            return
        for layer in sorted(self._dirty):
            np.savez_compressed(self.dir / f"layer{layer}.npz", **self._mem[layer])
        self._dirty.clear()
        self._since_flush = 0

    # ---------- access ----------
    @staticmethod
    def key(item, condition) -> str:
        cond = getattr(condition, "value", condition)
        return f"{item.item_id}|{cond}"

    def get(self, item, condition, layer: int, position: int = -1,
            prompt: str = None) -> np.ndarray:
        return self.get_many(item, condition, [layer], position, prompt)[layer]

    def get_many(self, item, condition, layers, position=-1,
                 prompt: str = None) -> dict[int, np.ndarray]:
        # keyed by the position *name*, not the resolved index: a named
        # position lands on a different token index in every item, so caching
        # by index would collide unrelated locations
        k = self.key(item, condition)
        if position != -1:
            k = f"{k}@{position}"
        missing = [l for l in layers if k not in self._mem.get(l, {})]
        if missing:
            from conflict_bench.core.prompts import build_prompt
            from conflict_bench.core import positions as pos_mod
            p = prompt if prompt is not None else build_prompt(item, condition)
            idx = pos_mod.resolve(self.model, item, condition, position, prompt=p)
            acts = self.model.residual_at(p, missing, idx)
            for l, v in acts.items():
                self._mem.setdefault(l, {})[k] = np.asarray(v, dtype=np.float32)
                self._dirty.add(l)
            self.n_extracted += 1
            self._since_flush += 1
            if self.autoflush_every and self._since_flush >= self.autoflush_every:
                self.flush()
        else:
            self.n_hits += 1
        return {l: self._mem[l][k] for l in layers}

    def stack(self, items, condition, layer, position=-1) -> np.ndarray:
        return np.stack([self.get(it, condition, layer, position) for it in items])

    def stats(self):
        return {"extracted": self.n_extracted, "cache_hits": self.n_hits,
                "layers": sorted(self._mem), "dir": str(self.dir)}
