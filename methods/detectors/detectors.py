"""Detectors: does/will the model follow the context?

Mechanistic:   linear_probe, diffmean_proj  (residual-stream, GroupKFold outside)
Grey-box:      margin, confidence_gain      (log-probs / entropy only)
Black-box:     semantic_entropy, selfcheck_nli, p_true  (samples / verbal)

Score convention: HIGHER = more context-following predicted.

Prompt construction lives in core.prompts (one source of truth for every
method and every condition); `build_prompt` is re-exported here because the
steerers and the runner have always imported it from this module.
"""
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # container smoke tests only

from conflict_bench.core.types import Item, Condition, DetectionRecord
from conflict_bench.core.prompts import build_prompt  # noqa: F401 (re-export)
from conflict_bench.methods.base import Detector, register_detector


# ---------------------------------------------------------------- grey-box
@register_detector("margin")
class MarginBaseline(Detector):
    """Teacher-forced margin - the v3 DV itself, the floor every fancier
    detector must beat.

    Reported twice, because the two numbers answer different questions:
      score_raw = -(margin under C)        what was actually observed
      score     = -(margin under C - margin under R)   the part attributable
                  to the conflict rather than to the passage template

    Sign is negated so that HIGHER = more context-following, the convention
    every detector in this file follows.  Both come out of the shared
    MarginTable, so R is scored once per item for the whole run.
    """
    access = "grey-box"

    def score(self, item, condition):
        table = self.model.margins
        sc = table.scored(item, condition)
        raw = -sc.margin
        extras = {"logp_true": sc.logp_true, "logp_cf": sc.logp_cf,
                  "margin_raw": sc.margin}
        corrected = raw
        if self.cfg.get("format_correct", True):
            ref = table.scored(item, table.reference)
            corrected = -table.corrected(item, condition)
            extras.update(reference_condition=table.reference.value,
                          logp_true_ref=ref.logp_true,
                          logp_cf_ref=ref.logp_cf,
                          margin_ref=ref.margin,
                          margin_corrected=-corrected)
        return DetectionRecord(item.item_id, item.relation, condition,
                               self.name, score=corrected, score_raw=raw,
                               extras=extras)


@register_detector("confidence_gain")
class ConfidenceGain(Detector):
    """CK-PLUG's conflict signal: entropy shift of next-token distribution
    after context insertion (Bi et al. 2025, arXiv:2503.15888).
    Negative gain => conflict => model being pulled by context."""
    access = "grey-box"

    def score(self, item, condition):
        import torch.nn.functional as F
        stem_logits = self.model.next_token_logits(
            build_prompt(item, Condition.NORMAL))
        ctx_logits = self.model.next_token_logits(build_prompt(item, condition))

        def h(z):
            return -(F.softmax(z, -1) * F.log_softmax(z, -1)).sum().item()

        h_stem, h_ctx = h(stem_logits), h(ctx_logits)
        gain = h_stem - h_ctx  # >0: context sharpened the prediction
        # conflict signal: sharpened toward WHAT? combine with margin sign.
        return DetectionRecord(item.item_id, item.relation, condition,
                               self.name, score=-gain,
                               extras={"H_stem": h_stem, "H_ctx": h_ctx})


@register_detector("context_jsd")
class ContextShift(Detector):
    """How far the passage moved the next-token distribution.

        p_ctx = p(. | passage, question)      p_par = p(. | question)
        score = JSD(p_ctx || p_par)   (or KL, per cfg['divergence'])

    This is the detection-side twin of the decoding family: it is literally
    the quantity AdaCAD uses as its per-token adaptive alpha, and the signal
    CAD amplifies with a fixed one.  Registered twice (`context_jsd`,
    `context_kl`) because the two divergences behave differently as
    detectors - JSD is bounded and symmetric, KL is neither.

    Note what this can and cannot see: it measures that the passage changed
    the distribution, not that it changed it toward the counterfactual.  A
    passage that merely confuses the model scores high too.  That is exactly
    why it is reported next to `margin`, which is directional.
    """
    access = "grey-box"

    def score(self, item, condition):
        z_ctx = self.model.next_token_logits(build_prompt(item, condition))
        z_par = self.model.next_token_logits(
            build_prompt(item, Condition.NORMAL))
        p = torch.softmax(z_ctx, -1)
        q = torch.softmax(z_par, -1)
        div = self.cfg.get("divergence", "jsd")
        if div == "kl":
            val = float((p * (torch.log(p.clamp_min(1e-12))
                              - torch.log(q.clamp_min(1e-12)))).sum())
        else:
            m = 0.5 * (p + q)
            logm = torch.log(m.clamp_min(1e-12))
            kl_p = (p * (torch.log(p.clamp_min(1e-12)) - logm)).sum()
            kl_q = (q * (torch.log(q.clamp_min(1e-12)) - logm)).sum()
            val = float(0.5 * (kl_p + kl_q) / float(np.log(2.0)))
        return DetectionRecord(item.item_id, item.relation, condition,
                               self.name, score=val,
                               extras={"divergence": div})


@register_detector("context_kl")
class ContextShiftKL(ContextShift):
    """`context_jsd` with KL instead - see that class."""

    def __init__(self, model, cfg=None):
        cfg = dict(cfg or {})
        cfg.setdefault("divergence", "kl")
        super().__init__(model, cfg)


# ---------------------------------------------------------------- text baseline
@register_detector("bow")
class BagOfWords(Detector):
    """Bag-of-words logistic regression on the prompt text. No model internals.

    The baseline that decides how much of a probe's AUROC is a claim about
    representations at all.  Everything a probe reads is downstream of this
    text; if TF-IDF over the passage and question predicts context-following
    just as well, then what the probe found is a property of the *items* -
    which relations, which entity types, how the passage is phrased - and not
    a mechanism inside the model.

    It is also the natural partner to the relation-base-rate control, which
    only sees the relation id: BOW sees the surface form too, so it is the
    stronger of the two nulls and the one worth beating.

    Trainable, so the runner puts it under the same GroupKFold-by-relation
    split as the probes and it never sees a test relation's vocabulary.
    """
    access = "black-box"
    requires_training = True

    def __init__(self, model, cfg=None):
        super().__init__(model, cfg)
        self.pipe = None
        self.fit_meta = {}

    def _text(self, item, condition):
        """The same text the model is conditioned on - nothing extra.

        cfg['fields'] can narrow it to isolate where the leakage lives:
        'passage' alone, 'question' alone, or both (default).
        """
        fields = self.cfg.get("fields", "both")
        passage = item.passages.get(getattr(condition, "value", condition), "")
        question = item.question or item.cloze_template
        if fields == "passage":
            return passage
        if fields == "question":
            return question
        return passage + "\n" + question

    def fit(self, items, labels, condition=Condition.CONFLICTING):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        y = np.asarray(labels)
        if len(np.unique(y)) < 2:
            raise ValueError(
                f"bow: train fold is single-class ({int(y.sum())}/{len(y)} "
                f"context-followers)")
        vec = TfidfVectorizer(
            lowercase=True,
            ngram_range=tuple(self.cfg.get("ngram_range", (1, 2))),
            min_df=self.cfg.get("min_df", 2),
            max_features=self.cfg.get("max_features", 50000),
            sublinear_tf=True,
            stop_words=self.cfg.get("stop_words") or None)
        clf = LogisticRegression(max_iter=self.cfg.get("max_iter", 2000),
                                 C=self.cfg.get("C", 1.0))
        self.pipe = make_pipeline(vec, clf)
        X = [self._text(it, condition) for it in items]
        self.pipe.fit(X, y)
        self.fit_meta = {
            "fields": self.cfg.get("fields", "both"),
            "n_train": len(items), "train_pos_rate": float(y.mean()),
            "n_features": len(vec.vocabulary_),
            "train_acc": float(self.pipe.score(X, y)),
            "train_relations": sorted({it.relation for it in items})}

    def score(self, item, condition):
        p = self.pipe.predict_proba([self._text(item, condition)])[0, 1]
        return DetectionRecord(item.item_id, item.relation, condition,
                               self.name, score=float(p))

    def save_artifacts(self, store, tag=""):
        if self.pipe is None:
            return {}
        vec, clf = self.pipe.steps[0][1], self.pipe.steps[1][1]
        coef = clf.coef_[0]
        order = np.argsort(coef)
        inv = {i: w for w, i in vec.vocabulary_.items()}
        # the features doing the work: if these read as relation or entity-type
        # markers, the "detector" is a topic classifier
        store.json(f"{tag}top_features.json", {
            "toward_context": [[inv[i], float(coef[i])] for i in order[-40:][::-1]],
            "toward_parametric": [[inv[i], float(coef[i])] for i in order[:40]]})
        store.json(f"{tag}meta.json", self.fit_meta)
        return self.fit_meta


# ---------------------------------------------------------------- mechanistic
@register_detector("linear_probe")
class LinearProbe(Detector):
    """Logistic probe on the residual stream. GroupKFold-by-relation is
    enforced by the runner (fit only sees train folds).  Activations come
    from the shared on-disk cache (model.acts) - re-extraction dominates
    runtime otherwise (known v3 lesson), and the tensors are themselves an
    output of the mechanistic arm."""
    requires_training = True
    uses_position = True

    def __init__(self, model, cfg=None):
        super().__init__(model, cfg)
        self.layer = self.cfg.get("layer", 19)
        # a NAMED position, resolved per item (core/positions.py). The runner
        # sweeps this; "last" is the readout position and is reported as such.
        self.position = self.cfg.get("position", "end_of_context")
        self.clf = None
        self.fit_meta = {}

    def _feat(self, item, condition):
        return self.model.acts.get(item, condition, self.layer, self.position)

    def fit(self, items, labels, condition=Condition.CONFLICTING):
        from sklearn.linear_model import LogisticRegression
        X = np.stack([self._feat(it, condition) for it in items])
        y = np.asarray(labels)
        self.clf = LogisticRegression(max_iter=self.cfg.get("max_iter", 2000),
                                      C=self.cfg.get("C", 1.0))
        self.clf.fit(X, y)
        self.fit_meta = {"layer": self.layer, "position": self.position,
                         "C": self.cfg.get("C", 1.0), "n_train": len(items),
                         "train_pos_rate": float(y.mean()),
                         "train_relations": sorted({it.relation for it in items}),
                         "train_acc": float(self.clf.score(X, y))}

    def score(self, item, condition):
        h = self._feat(item, condition)
        p = self.clf.predict_proba(h[None])[0, 1]
        return DetectionRecord(item.item_id, item.relation, condition,
                               self.name, score=float(p),
                               extras={"layer": self.layer,
                                       "decision": float(
                                           self.clf.decision_function(h[None])[0])})

    def save_artifacts(self, store, tag=""):
        if self.clf is None:
            return {}
        store.array(f"{tag}coef.npy", self.clf.coef_[0])
        store.array(f"{tag}intercept.npy", self.clf.intercept_)
        store.json(f"{tag}meta.json", self.fit_meta)
        return self.fit_meta


@register_detector("diffmean_proj")
class DiffMeanProjection(Detector):
    """AxBench's best detector, transplanted: w = mean(context-followers)
    - mean(parametric-followers); score = h . w.  Cheaper than the probe, and
    the same w is exactly the raw material act_add steers with - which is why
    the vector is written to disk, not just its AUROC."""
    requires_training = True
    uses_position = True

    def __init__(self, model, cfg=None):
        super().__init__(model, cfg)
        self.layer = self.cfg.get("layer", 19)
        self.position = self.cfg.get("position", "end_of_context")
        self.w = None
        self.fit_meta = {}

    def _feat(self, item, condition):
        return self.model.acts.get(item, condition, self.layer, self.position)

    def fit(self, items, labels, condition=Condition.CONFLICTING):
        X = np.stack([self._feat(it, condition) for it in items])
        y = np.asarray(labels)
        if y.sum() == 0 or y.sum() == len(y):
            raise ValueError(
                f"diffmean_proj: train fold has a single class "
                f"({int(y.sum())}/{len(y)} context-followers); no contrast to "
                f"take a difference of")
        mu_ctx, mu_par = X[y == 1].mean(0), X[y == 0].mean(0)
        w = mu_ctx - mu_par
        self.raw_norm = float(np.linalg.norm(w))
        self.w = w / (self.raw_norm + 1e-12)
        self.fit_meta = {"layer": self.layer, "position": self.position,
                         "n_train": len(items), "n_context": int(y.sum()),
                         "raw_norm": self.raw_norm,
                         "train_relations": sorted({it.relation for it in items})}

    def score(self, item, condition):
        h = self._feat(item, condition)
        return DetectionRecord(item.item_id, item.relation, condition,
                               self.name, score=float(h @ self.w),
                               extras={"layer": self.layer})

    def save_artifacts(self, store, tag=""):
        if self.w is None:
            return {}
        store.array(f"{tag}w.npy", self.w)
        store.json(f"{tag}meta.json", self.fit_meta)
        return self.fit_meta


@register_detector("logit_lens")
class LogitLensReadout(Detector):
    """The readout control - the mandatory partner of every probe.

    A probe read at (layer, position) and scored against the model's own
    continuation from that same prompt can succeed for two very different
    reasons: the residual stream represents the arbitration, or the residual
    stream already contains the answer and the probe is decoding it.  AUROC
    alone cannot tell those apart.

    So: whatever a probe can read at (layer, position), the model's own
    unembedding can read there too.  This detector *is* that readout - the
    logit-lens margin between the counterfactual and true answers' first
    tokens at exactly the layer and position the probe uses.  A probe that
    does not beat it is not detecting anything the model was not already
    about to say.

    Report it beside every probe, the way base-rate AUROC is reported beside
    every detection AUROC (`auroc_over_readout` in detection_summary.csv).
    """
    access = "white-box"
    uses_position = True

    def __init__(self, model, cfg=None):
        super().__init__(model, cfg)
        self.layer = self.cfg.get("layer", 19)
        self.position = self.cfg.get("position", "end_of_context")

    def score(self, item, condition):
        h = self.model.acts.get(item, condition, self.layer, self.position)
        z = self.model.unembed(h)
        logp = torch.log_softmax(z, -1)
        t_id = self.model.first_token_id(" " + item.true_answer)
        c_id = self.model.first_token_id(" " + item.counterfactual_answer)
        return DetectionRecord(
            item.item_id, item.relation, condition, self.name,
            score=float(logp[c_id] - logp[t_id]),
            extras={"layer": self.layer, "position": str(self.position),
                    "logp_true_lens": float(logp[t_id]),
                    "logp_cf_lens": float(logp[c_id])})


# ---------------------------------------------------------------- black-box
@register_detector("semantic_entropy")
class SemanticEntropy(Detector):
    """Kuhn/Farquhar-style SE (github.com/jlko/semantic_uncertainty).
    Sample K continuations, cluster by alias-match (cheap surrogate for NLI
    on cloze answers - answers are short entities, so exact/alias matching
    is a defensible clustering here), entropy over clusters.
    High SE under C => arbitration unresolved; direction read off cluster ids.
    All K samples are kept in `extras` so the clustering can be redone with a
    real NLI model without re-sampling."""
    access = "black-box"

    def score(self, item, condition):
        K = self.cfg.get("num_samples", 10)
        temp = self.cfg.get("temperature", 1.0)
        samples = self.model.generate(build_prompt(item, condition),
                                      max_new_tokens=self.cfg.get("max_new_tokens", 8),
                                      temperature=temp, num_samples=K)

        def cluster(s):
            s = s.strip().lower()
            if any(a.lower() in s for a in [item.true_answer] + item.true_aliases):
                return "true"
            if any(a.lower() in s for a in
                   [item.counterfactual_answer] + item.cf_aliases):
                return "cf"
            return "other:" + (s.split()[0] if s.split() else "empty")

        assignments = [cluster(s) for s in samples]
        ids, counts = np.unique(assignments, return_counts=True)
        p = counts / counts.sum()
        H = float(-(p * np.log(p)).sum())
        frac_cf = float(counts[list(ids).index("cf")] / K) if "cf" in ids else 0.0
        return DetectionRecord(item.item_id, item.relation, condition,
                               self.name, score=frac_cf,
                               extras={"entropy": H,
                                       "clusters": dict(zip(ids.tolist(),
                                                            counts.tolist())),
                                       "samples": samples,
                                       "assignments": assignments})


@register_detector("selfcheck_nli")
class SelfCheckNLIDetector(Detector):
    """SelfCheckGPT-NLI (pip install selfcheckgpt; Manakul et al. 2023).
    Consistency of the greedy answer against sampled answers.
    TODO: instantiate SelfCheckNLI lazily; for cloze answers consider the
    cheaper alias-consistency variant above first."""
    access = "black-box"

    def score(self, item, condition):
        raise NotImplementedError(
            "pip install selfcheckgpt --break-system-packages; "
            "wrap SelfCheckNLI.predict over greedy vs sampled continuations")


@register_detector("p_true")
class PTrue(Detector):
    """Verbalized self-report: ask the model whether the passage's claim is
    true / whether it will answer from the passage; score = P('Yes').
    This is the 'prompting baseline' leg of the mandatory triplet."""
    access = "black-box"

    TEMPLATE = ("Background: {passage}\n\nQuestion: is the following claim "
                "consistent with your knowledge: \"{claim}\"?\n"
                "Answer Yes or No.\nAnswer:")

    def score(self, item, condition):
        claim = item.meta.get("cf_statement") or (
            item.cloze_template.replace("____", item.counterfactual_answer))
        prompt = self.TEMPLATE.format(
            passage=item.passages.get(condition.value, ""), claim=claim)
        lp_yes = self.model.logp_continuation(prompt, " Yes")
        lp_no = self.model.logp_continuation(prompt, " No")
        m = max(lp_yes, lp_no)
        p_yes = float(np.exp(lp_yes - m) / (np.exp(lp_yes - m) + np.exp(lp_no - m)))
        return DetectionRecord(item.item_id, item.relation, condition,
                               self.name, score=p_yes,
                               extras={"logp_yes": lp_yes, "logp_no": lp_no,
                                       "claim": claim})
