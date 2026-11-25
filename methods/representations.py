"""The representation family: read and write the residual stream.

Detection
  linear_probe    logistic regression on the residual stream
  diffmean_proj   projection on the mean-difference direction
  logit_lens      ORACLE readout control at the probe's layer and position

Steering
  caa             plain mean-difference CAA (Rimsky et al. 2024)
  act_add         CAA orthogonalized against the "a context is present" direction

Layers are depth fractions (0.6 = 60% of the way up) so a config survives a
model swap.  Activations come from the shared on-disk cache (model.acts).
"""
import numpy as np
import torch

from conflict_bench.api import (Detection, Detector, Intervention, Steerer,
                                register_detector, register_steerer)
from conflict_bench.core import prompts
from conflict_bench.core.positions import PositionError


class _Probe(Detector):
    """Shared: where the feature vector comes from."""
    requires_training = True

    def __init__(self, model, cfg=None):
        super().__init__(model, cfg)
        self.layer = self.cfg.get("layer", 0.6)
        # a NAMED position (core/positions.py). `last` is the readout
        # position; `end_of_context` is before the question is read.
        self.position = self.cfg.get("position", "end_of_context")

    def _feat(self, inst, position=None):
        return self.model.acts.get(inst, self.layer, position or self.position)

    def _xy(self, train):
        X = np.stack([self._feat(i) for i, _ in train])
        y = np.asarray([lab["conflict"] for _, lab in train])
        if len(np.unique(y)) < 2:
            raise ValueError(f"{self.name}: training fold is single-class "
                             f"({int(y.sum())}/{len(y)} positive)")
        return X, y

    def _value(self, h) -> float:
        raise NotImplementedError

    def score(self, inst):
        chunks = None
        if self.cfg.get("localize", False):
            # the same readout at the end of every sentence
            chunks = [self._value(self._feat(inst, f"chunk:{i}"))
                      for i in range(len(inst.chunks))]
        return Detection(self._value(self._feat(inst)), chunks)


@register_detector("linear_probe")
class LinearProbe(_Probe):
    """Logistic probe on the residual stream at (layer, position)."""

    def fit(self, train):
        from sklearn.linear_model import LogisticRegression
        X, y = self._xy(train)
        self.clf = LogisticRegression(max_iter=2000, C=self.cfg.get("C", 1.0))
        self.clf.fit(X, y)

    def _value(self, h):
        return float(self.clf.decision_function(h[None])[0])

    def save_artifacts(self, store, tag=""):
        store.array(f"{tag}coef.npy", self.clf.coef_[0])
        store.array(f"{tag}intercept.npy", self.clf.intercept_)


@register_detector("diffmean_proj")
class DiffMeanProjection(_Probe):
    """w = mean(conflict) - mean(no conflict); score = h . w.  Cheaper than
    the probe, and the same kind of vector CAA steers with."""

    def fit(self, train):
        X, y = self._xy(train)
        w = X[y == 1].mean(0) - X[y == 0].mean(0)
        self.w = w / (np.linalg.norm(w) + 1e-12)

    def _value(self, h):
        return float(h @ self.w)

    def save_artifacts(self, store, tag=""):
        store.array(f"{tag}w.npy", self.w)


@register_detector("logit_lens")
class LogitLensReadout(Detector):
    """ORACLE readout control - the partner of every probe.

    Whatever a probe can read at (layer, position), the model's own
    unembedding can read there too.  This is that readout: the logit-lens
    margin between the first tokens of b and a at the probe's layer and
    position.  It is told both answers, so it is a control, not a method.  A
    probe that does not beat it is not detecting anything the model was not
    already about to say.  Keep its layer and position equal to the probe's.
    """
    oracle = True

    def score(self, inst, rec):
        c = rec.get("contrast")
        if not c:
            return Detection(float("nan"))
        h = self.model.acts.get(inst, self.cfg.get("layer", 0.6),
                                self.cfg.get("position", "end_of_context"))
        logp = torch.log_softmax(self.model.unembed(h), -1)
        first = lambda s: self.model.first_token_id(prompts.continuation(s))
        return Detection(float(logp[first(c["b"])] - logp[first(c["a"])]))


# ================================================================= steering
@register_steerer("act_add")
class ActivationAddition(Steerer):
    """CAA-style additive steering, vectors fitted in-run on training folds.

        v_override[l] = mean(h | M followed the context)
                      - mean(h | M kept its memory)      on conflict instances
        v_use[l]      = mean(h | supporting context) - mean(h | no context)

    v_override is the arbitration direction.  v_use is the direction for the
    mere presence of a context, measured where there is no conflict; steering
    along the part of v_override inside it just turns up "there is a document
    here", so v_override is orthogonalized against it (`orthogonalize`).

    The contrast is behavioural (`labels["follows"]`), so a fold where M
    always follows one source cannot be fitted - the method then refuses,
    loudly, and the harness reports it as skipped.

    Strength is in units of the layer's typical residual norm
    (`scale: residual`).  Control: a random direction of equal norm.
    """
    requires_training = True
    strengths = [1.0, 2.0, 4.0]

    def __init__(self, model, cfg=None):
        super().__init__(model, cfg)
        self.layers = [model.resolve_layer(l)
                       for l in self.cfg.get("layers", [0.55, 0.6, 0.65])]
        self.position = self.cfg.get("position", "last")
        self.orthogonalize = self.cfg.get("orthogonalize", True)
        self.vecs = {}
        self._rng = np.random.default_rng(self.cfg.get("seed", 0))

    def fit(self, train):
        acts = self.model.acts
        ctx = [i for i, lab in train if lab["conflict"] and lab["follows"] == "ctx"]
        mem = [i for i, lab in train if lab["conflict"] and lab["follows"] == "mem"]
        if not ctx or not mem:
            raise ValueError(
                f"{self.name}: no behavioural contrast in the training folds "
                f"({len(ctx)} conflict instances followed the context, "
                f"{len(mem)} kept memory)")
        sup = [i for i, lab in train if lab["recipe"] == "sup"]
        if self.orthogonalize and not sup:
            raise ValueError(f"{self.name}: orthogonalize needs `sup` instances")
        self.meta = {"n_followed_context": len(ctx), "n_kept_memory": len(mem)}
        for l in self.layers:
            H_ctx, H_mem = (acts.stack(x, l, self.position) for x in (ctx, mem))
            v = H_ctx.mean(0) - H_mem.mean(0)
            v = v / (np.linalg.norm(v) + 1e-12)
            if self.orthogonalize:
                u = (acts.stack(sup, l, self.position).mean(0)
                     - acts.stack([i.without_context() for i in sup], l,
                                  self.position).mean(0))
                u = u / (np.linalg.norm(u) + 1e-12)
                self.meta[f"cos_override_use_L{l}"] = float(v @ u)
                v = v - (v @ u) * u
                v = v / (np.linalg.norm(v) + 1e-12)
            resid = float(np.linalg.norm(np.concatenate([H_ctx, H_mem]), axis=1).mean())
            step = resid if self.cfg.get("scale", "residual") == "residual" else 1.0
            self.vecs[l] = torch.from_numpy((v * step).astype(np.float32))

    def save_artifacts(self, store, tag=""):
        store.torch(f"{tag}vectors.pt", {"vectors": self.vecs, "meta": self.meta})
        return self.meta

    def _add(self, vecs, alpha):
        return Intervention(
            hooks=lambda: self.model.add_direction(vecs, self.layers, alpha=alpha))

    def intervene(self, inst, direction, strength, span=None):
        if direction == "span":
            return None            # one vector cannot name a passage
        return self._add(self.vecs, strength if direction == "ctx" else -strength)

    def control(self, inst, direction, strength, span=None):
        rand = {}
        for l, v in self.vecs.items():
            r = torch.from_numpy(
                self._rng.standard_normal(v.shape[0]).astype(np.float32))
            rand[l] = r / r.norm() * v.norm()
        return self._add(rand, strength)


@register_steerer("caa")
class SimpleCAA(ActivationAddition):
    """Plain CAA: the mean-difference vector, added, nothing else.  The pair
    caa / act_add isolates what the orthogonalization buys."""

    def __init__(self, model, cfg=None):
        cfg = dict(cfg or {})
        cfg.setdefault("orthogonalize", False)
        super().__init__(model, cfg)
