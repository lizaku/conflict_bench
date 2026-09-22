"""Steerers: push arbitration toward context or toward memory.

Mechanistic:  act_add (CAA/DiffMean, layers 17-19, orthogonalized v_override)
Output-based: prompt_instruct, cad, adacad, ckplug (contrastive decoding),
              csks (external repo, proxy models)

The question every steerer answers:

    Given a CONFLICTING passage, can the intervention make the model output
    the conflicting answer when it otherwise would not?

so each one runs under BOTH conditions the runner gives it:

    C   the measurement - the conflict the method is supposed to resolve
    S   the matched specificity control - the same intervention, the same
        item, the same dose, but a passage that agrees with memory.  A method
        that moves the margin just as hard under S is not resolving a
        conflict, it is just pushing on the passage.

`metrics.steering_summary` pairs the two by item and reports
`conflict_specific_effect = d_margin(C) - d_margin(S)`.  That control is a
matched *condition*, not a synthetic baseline, and it is the reason the S
arm is worth its forward passes.

Margin convention (as everywhere else in the repo):
    m = logp(true) - logp(cf)      m > 0 parametric wins, m < 0 context wins
so a `use_context` success is m0 > 0 -> m1 < 0, and a `use_parametric`
success is m0 < 0 -> m1 > 0.  Items that already sit on the target side are
not flippable; that is recorded per item so the flip rate can be read both
unconditionally and conditioned on flippable items, and `n_flippable` is
carried into every summary row - a flip rate over 6 items and a flip rate
over 394 are not the same measurement and must not share a column unlabelled.

Effect metric everywhere: delta(margin) + flip rate, always against the
method's matched control.  Every steerer declares one; a method that returns
None gets `specific_effect = NaN` rather than having its raw delta quietly
promoted into a controlled number.
"""
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # container smoke tests only

from conflict_bench.core.types import Item, Condition, SteeringRecord
from conflict_bench.core.prompts import build_prompt, continuation
from conflict_bench.methods.base import Steerer, register_steerer


def _raw_margin(model, item, condition, prompt_override=None):
    p = prompt_override if prompt_override is not None else build_prompt(item, condition)
    lp_t = model.logp_continuation(p, continuation(item.true_answer))
    lp_c = model.logp_continuation(p, continuation(item.counterfactual_answer))
    return lp_t - lp_c  # >0 parametric wins


def _flipped(m0, m1, target):
    return (m0 > 0 > m1) if target == "use_context" else (m0 < 0 < m1)


def _flippable(m0, target):
    return (m0 > 0) if target == "use_context" else (m0 < 0)


class _MarginMixin:
    """The two margin views + fluency, shared by every steerer.

    The R correction subtracts the per-item margin under the irrelevant
    passage from both margin_before and margin_after.  It cancels out of
    delta-margin exactly - the offset is a per-item constant - so what it
    changes is which side of zero an item sits on, i.e. the flip rate.

    It is no longer the default, and the reason is measured rather than
    theoretical: R differs from C on two axes at once, the passage template
    AND whether the passage is about this item at all.  Subtracting it removes
    the second along with the first, and on the 400-item pilot that put
    394/400 items on the context side of zero while only 47% of them actually
    behaved that way - which flattened the margin detector to chance and left
    the use_context arm with 6 testable items.  See core/margins.py.

    So: the RAW pair is primary, the corrected pair is computed and recorded
    beside it whenever R was scored, and `format_correct: true` swaps which
    one `margin_before/after` mirrors.  Neither is ever dropped - a flip rate
    that only exists under one of the two views is visible as exactly that.
    """

    def _offset(self, item):
        """The R margin, or 0.0 when R was not part of this run.

        Independent of `format_correct`: whether the correction is *primary*
        and whether it is *available* are different questions, and the reports
        want both views whenever R exists.
        """
        return self.model.margins.offset(item)

    def _has_reference(self, item):
        return self.model.margins.reference_margin(item) is not None

    def raw_margin(self, item, condition, prompt_override=None):
        """What is actually observed - no correction."""
        if prompt_override is None:
            return self.model.margins.margin(item, condition)
        return _raw_margin(self.model, item, condition, prompt_override)

    def margin(self, item, condition, prompt_override=None):
        return self.raw_margin(item, condition, prompt_override) \
            - self._offset(item)

    def record(self, item, condition, target, factor, m0_raw, m1_raw, **kw):
        """Build a SteeringRecord carrying both views and the matched control.

        `control_margin_after_raw=None` means the method declared no control;
        it stays None so metrics reports specific_effect as NaN instead of
        silently promoting the raw delta into a controlled number.
        """
        have_ref = self._has_reference(item)
        off = self._offset(item)
        m0_corr = m0_raw - off if have_ref else None
        m1_corr = m1_raw - off if have_ref else None
        use_corr = bool(self.cfg.get("format_correct", False)) and have_ref
        m0, m1 = (m0_corr, m1_corr) if use_corr else (m0_raw, m1_raw)

        extras = kw.pop("extras", {})
        extras.setdefault("flippable", _flippable(m0, target))
        extras.setdefault("flippable_raw", _flippable(m0_raw, target))
        extras.setdefault("view", "r_corrected" if use_corr else "raw")
        ctrl_raw = kw.pop("control_margin_after_raw", None)
        return SteeringRecord(
            item.item_id, item.relation, condition, self.name, target, factor,
            margin_before=m0, margin_after=m1,
            flipped=_flipped(m0, m1, target),
            margin_before_raw=m0_raw, margin_after_raw=m1_raw,
            flipped_raw=_flipped(m0_raw, m1_raw, target),
            margin_before_corrected=m0_corr, margin_after_corrected=m1_corr,
            flipped_corrected=(None if not have_ref
                               else _flipped(m0_corr, m1_corr, target)),
            r_offset=off if have_ref else None,
            # the control must live in the SAME view as margin_before/after,
            # or specific_effect differences two different quantities
            control_margin_after=(None if ctrl_raw is None
                                  else (ctrl_raw - off if use_corr
                                        else ctrl_raw)),
            control_margin_after_raw=ctrl_raw,
            extras=extras, **kw)

    def fluency(self, prompt, generation):
        """Mean log-prob per token of the steered generation under the
        UNSTEERED model - the capability axis of the AxBench Fig-4 Pareto.
        Higher (closer to 0) is more fluent."""
        if not generation.strip() or not self.cfg.get("measure_fluency", True):
            return None
        return self.model.logp_continuation(prompt, generation,
                                            length_normalize=True)


@register_steerer("act_add")
class ActivationAddition(_MarginMixin, Steerer):
    """CAA-style additive steering (Rimsky et al. 2024; nrimsky/CAA).

    Vectors are computed from scratch during the run, per layer, from the
    behavioural contrast on the training relations:

        v_override[l] = mean(h_C[l] | followed context)
                      - mean(h_C[l] | kept parametric)
        v_use[l]      = mean(h_S[l]) - mean(h_N[l])

    v_override is the arbitration direction; v_use is the direction for the
    mere *presence and use* of a passage, measured where there is no conflict
    (S vs N).  Steering along the part of v_override that lies inside v_use
    would just be turning the volume up on "there is a document here", which
    is what the v3 postmortem traced the wrong-sign result to
    (cos(v_override, v_use) = -0.39 at L25) - so v_override is orthogonalized
    against v_use before use.

    v3 lessons baked in:
      - target mid-network layers (default 17-19), never final layers
      - orthogonalize v_override against v_use before steering
      - control = norm-matched random direction, same layers/positions
      - alpha is expressed in units of the layer's typical residual norm
        (`scale: residual`), so the factor sweep means the same thing after
        a model swap

    Everything fitted is written to disk (`artifacts/act_add/*.pt`): the
    vectors, their norms, the cosines between them, and the fold membership.
    """
    requires_training = True

    def __init__(self, model, cfg=None):
        super().__init__(model, cfg)
        self.layers = [int(l) for l in self.cfg.get("layers", [17, 18, 19])]
        self.position = self.cfg.get("position", -1)
        self.v_override = {}     # {layer: np.ndarray [d]}  unit norm
        self.v_use = {}
        self.scale = {}          # {layer: float} alpha=1 step size
        self.fit_meta = {}
        self._rng = np.random.default_rng(self.cfg.get("seed", 0))

    # ---------- fitting ----------
    def fit(self, items, labels=None):
        path = self.cfg.get("vectors_path")
        if path:
            self._load(path)
            return
        if labels is None:
            raise ValueError("act_add.fit needs behavioural labels to define "
                             "the contrast (runner passes them)")
        y = np.asarray(labels)
        if y.sum() == 0 or y.sum() == len(y):
            raise ValueError(
                f"act_add: training fold is single-class "
                f"({int(y.sum())}/{len(y)} context-followers) - no behavioural "
                f"contrast to build v_override from")

        acts = self.model.acts
        H_C = {l: acts.stack(items, Condition.CONFLICTING, l, self.position)
               for l in self.layers}
        need_use = self.cfg.get("orthogonalize", True)
        H_S = ({l: acts.stack(items, Condition.SUPPORTING, l, self.position)
                for l in self.layers} if need_use else {})
        H_N = ({l: acts.stack(items, Condition.NORMAL, l, self.position)
                for l in self.layers} if need_use else {})

        meta_layers = {}
        for l in self.layers:
            d = H_C[l][y == 1].mean(0) - H_C[l][y == 0].mean(0)
            n = float(np.linalg.norm(d))
            self.v_override[l] = d / (n + 1e-12)
            if need_use:
                du = H_S[l].mean(0) - H_N[l].mean(0)
                self.v_use[l] = du / (np.linalg.norm(du) + 1e-12)
            resid_norm = float(np.linalg.norm(H_C[l], axis=1).mean())
            mode = self.cfg.get("scale", "residual")
            self.scale[l] = {"residual": resid_norm, "unit": 1.0,
                             "raw": n}.get(mode, resid_norm)
            meta_layers[l] = {
                "raw_diff_norm": n, "mean_residual_norm": resid_norm,
                "alpha1_step": self.scale[l],
                "cos_override_use": (float(self.v_override[l] @ self.v_use[l])
                                     if need_use else None)}
        self.fit_meta = {"layers": self.layers, "position": self.position,
                         "scale_mode": self.cfg.get("scale", "residual"),
                         "orthogonalize": need_use, "n_train": len(items),
                         "n_context_followers": int(y.sum()),
                         "train_relations": sorted({it.relation for it in items}),
                         "per_layer": meta_layers}

    def _load(self, path):
        blob = torch.load(path, map_location="cpu")
        self.v_override = {int(l): np.asarray(v) for l, v in blob["v_override"].items()}
        self.v_use = {int(l): np.asarray(v) for l, v in blob.get("v_use", {}).items()}
        self.scale = {int(l): float(s) for l, s in blob.get("scale", {}).items()}
        self.layers = sorted(self.v_override)
        self.scale = self.scale or {l: 1.0 for l in self.layers}
        self.fit_meta = dict(blob.get("meta", {}), loaded_from=str(path))

    def save_artifacts(self, store, tag=""):
        if not self.v_override:
            return {}
        store.torch(f"{tag}vectors.pt",
                    {"v_override": {l: torch.as_tensor(v)
                                    for l, v in self.v_override.items()},
                     "v_use": {l: torch.as_tensor(v)
                               for l, v in self.v_use.items()},
                     "scale": self.scale, "meta": self.fit_meta})
        for l, v in self.v_override.items():
            store.array(f"{tag}v_override_L{l}.npy", v)
        for l, v in self.v_use.items():
            store.array(f"{tag}v_use_L{l}.npy", v)
        store.json(f"{tag}meta.json", self.fit_meta)
        return self.fit_meta

    # ---------- steering ----------
    def _effective_vecs(self, target):
        """{layer: tensor} already signed, orthogonalized and scaled so that
        alpha=1 is one 'typical residual norm' of push."""
        out = {}
        for l in self.layers:
            v = np.array(self.v_override[l], dtype=np.float32)
            if target != "use_context":
                v = -v
            if self.cfg.get("orthogonalize", True) and l in self.v_use:
                u = self.v_use[l] / (np.linalg.norm(self.v_use[l]) + 1e-12)
                v = v - (v @ u) * u
                v = v / (np.linalg.norm(v) + 1e-12)
            out[l] = torch.from_numpy(v * self.scale.get(l, 1.0))
        return out

    def steer(self, item, condition, target, factor):
        prompt = build_prompt(item, condition)
        m0 = self.raw_margin(item, condition)
        vecs = self._effective_vecs(target)
        with self.model.add_direction(vecs, self.layers, alpha=factor):
            m1 = _raw_margin(self.model, item, condition)
            gen = self.model.generate(
                prompt, max_new_tokens=self.cfg.get("max_new_tokens", 8))[0]
        return self.record(
            item, condition, target, factor, m0, m1, generated=gen,
            fluency=self.fluency(prompt, gen),
            control_margin_after_raw=self.control(item, condition, target,
                                                  factor),
            extras={"layers": self.layers,
                    "alpha1_step": {l: self.scale.get(l) for l in self.layers}})

    def control(self, item, condition, target, factor):
        """Norm-matched random direction, same layers, same alpha. Raw -
        `record` applies the same R offset it applies to the real margins."""
        vecs = self._effective_vecs(target)
        rand = {}
        for l, v in vecs.items():
            r = torch.from_numpy(
                self._rng.standard_normal(v.shape[0]).astype(np.float32))
            rand[l] = r / r.norm() * v.norm()
        with self.model.add_direction(rand, self.layers, alpha=factor):
            return _raw_margin(self.model, item, condition)


@register_steerer("caa")
class SimpleCAA(ActivationAddition):
    """Plain CAA: the mean-difference vector, added, and nothing else.

    Identical machinery to `act_add` minus the orthogonalization against
    v_use, so the pair isolates exactly what that correction buys.  This is
    the method as originally described (Rimsky et al. 2024) and the honest
    baseline for the mechanistic arm; `act_add` is the repo's corrected
    variant, and reporting only the corrected one would hide whether the
    correction matters.
    """

    def __init__(self, model, cfg=None):
        cfg = dict(cfg or {})
        cfg.setdefault("orthogonalize", False)
        super().__init__(model, cfg)


@register_steerer("prompt_instruct")
class PromptInstruct(_MarginMixin, Steerer):
    """AxBench's winner and the mandatory baseline: explicit instruction to
    trust (or distrust) the document. Factor is discrete: instruction variants
    of increasing strength (index into TEMPLATES).

    Its matched control is a PLACEBO instruction - same block, same position,
    same rough length, no directional claim.  Without it the comparison is
    "an instruction" against "nothing inserted", which confounds the
    instruction's content with the fact that any text was added before the
    question at all.  This method previously declared no control, and since
    `specific_effect` filled a missing control with zero, its number was a
    bare delta-margin sitting in the same column as CAD's controlled one.
    """
    access = "black-box"

    #: matched for shape and length, directionally inert
    PLACEBO = ("The passage above is provided for reference. Answer the "
               "question that follows.")

    TEMPLATES = {
        "use_context": [
            "Answer strictly according to the passage above, even if it "
            "contradicts what you believe.",
            "The passage above is the authoritative, up-to-date source. Your "
            "prior knowledge may be outdated; answer ONLY from the passage.",
        ],
        "use_parametric": [
            "The passage above may contain errors. Answer from your own "
            "knowledge, ignoring the passage if it conflicts.",
            "Disregard the passage entirely; answer from what you know to be "
            "true.",
        ],
    }

    def steer(self, item, condition, target, factor):
        m0 = self.raw_margin(item, condition)
        variants = self.cfg.get("templates", {}).get(target) \
            or self.TEMPLATES[target]
        idx = min(max(int(factor), 0), len(variants) - 1)
        instr = variants[idx]
        prompt = build_prompt(item, condition, instruction=instr)
        m1 = self.raw_margin(item, condition, prompt_override=prompt)
        gen = self.model.generate(
            prompt, max_new_tokens=self.cfg.get("max_new_tokens", 8))[0]
        return self.record(
            item, condition, target, float(idx), m0, m1, generated=gen,
            fluency=self.fluency(prompt, gen),
            control_margin_after_raw=self.control(item, condition, target, idx),
            extras={"instruction": instr, "variant_index": idx})

    def control(self, item, condition, target, factor):
        """Margin after the placebo instruction - the same edit to the prompt
        with the directional content removed."""
        placebo = self.cfg.get("placebo", self.PLACEBO)
        if not placebo:
            return None
        return self.raw_margin(
            item, condition,
            prompt_override=build_prompt(item, condition, instruction=placebo))


class _ContrastiveDecoder(_MarginMixin, Steerer):
    """Shared core for the decoding family: CAD, AdaCAD, CK-PLUG.

    All three answer the same question - given the next-token distribution
    with the passage, p_ctx, and without it, p_par, how should the two be
    combined? - and differ only in the combination rule.  So the machinery is
    written once here and each method supplies
    `combine(z_ctx, z_par, factor, target) -> z`, operating on the full
    [T, V] teacher-forced logits for the candidate answer.

    Scoring the whole continuation rather than only its first token matters:
    the rest of the harness reports a multi-token teacher-forced margin, and a
    first-token proxy is a different DV that cannot go in the same table.  A
    pleasant consequence of doing it properly is that strength 0 reduces
    exactly to the shared MarginTable value, so this family uses the same R
    correction as everything else and its alpha=0 control is the real
    unsteered margin rather than a separate quantity.
    """
    access = "grey-box"

    def combine(self, z_ctx, z_par, factor, target):
        raise NotImplementedError

    def _logp(self, item, condition, continuation, factor, target):
        z_ctx, ids = self.model.teacher_forced_logits(
            build_prompt(item, condition), continuation)
        z_par, _ = self.model.teacher_forced_logits(
            build_prompt(item, Condition.NORMAL), continuation)
        z = self.combine(z_ctx, z_par, factor, target)
        logp = torch.log_softmax(z, -1)
        return float(logp.gather(-1, ids.unsqueeze(-1)).sum())

    def decoded_margin(self, item, condition, factor, target):
        return (self._logp(item, condition,
                           continuation(item.true_answer), factor, target)
                - self._logp(item, condition,
                             continuation(item.counterfactual_answer),
                             factor, target))

    def steer(self, item, condition, target, factor):
        # strength 0 == the plain teacher-forced margin, so the matched
        # control for this family is the unsteered margin itself
        m0 = self.raw_margin(item, condition)
        m1 = self.decoded_margin(item, condition, factor, target)
        return self.record(
            item, condition, target, factor, m0, m1,
            control_margin_after_raw=m0,
            extras=dict(self.steer_extras(item, condition, factor, target),
                        dv="teacher_forced_margin"))

    def steer_extras(self, item, condition, factor, target):
        return {}


@register_steerer("cad")
class ContextAwareDecoding(_ContrastiveDecoder):
    """CAD (Shi et al., NAACL 2024; github.com/xhan77/context-aware-decoding).

        z = (1 + a) * z_ctx - a * z_par

    A fixed, global amplification of whatever the passage did to the
    distribution.  a > 0 steers toward the context; a < 0 toward memory, which
    is how `use_parametric` is served here.
    """

    def combine(self, z_ctx, z_par, factor, target):
        a = factor if target == "use_context" else -factor
        return (1 + a) * z_ctx - a * z_par

    def steer_extras(self, item, condition, factor, target):
        return {"alpha": factor if target == "use_context" else -factor}


@register_steerer("adacad")
class AdaptiveContextAwareDecoding(_ContrastiveDecoder):
    """AdaCAD (Wang et al. 2024, arXiv:2409.07394).

    CAD applies one alpha to every token, including the ones where context and
    memory do not disagree at all and amplification is pure noise.  AdaCAD
    makes the weight per-token and proportional to how much the two
    distributions actually conflict, measured as Jensen-Shannon divergence:

        a_t = JSD(p_ctx,t || p_par,t)          in [0, 1] with log base 2
        z_t = (1 + s*a_t) * z_ctx,t - s*a_t * z_par,t

    With s = 1 this is AdaCAD as published; `factor` sweeps s so the method
    sits on the same dose-response axis as the others, and s = 0 recovers the
    unsteered distribution.  For `use_parametric` the adaptive weight is
    negated - an extrapolation past the paper, which poses AdaCAD purely as a
    context-faithfulness method, and it is labelled as such in the records.

    The per-item mean of a_t is kept in `extras`; it is the same JSD the
    `context_jsd` detector scores with, so this method's detection and
    steering arms read one quantity.
    """

    @staticmethod
    def _jsd(z_ctx, z_par):
        p, q = torch.softmax(z_ctx, -1), torch.softmax(z_par, -1)
        m = 0.5 * (p + q)
        logm = torch.log(m.clamp_min(1e-12))
        kl_p = (p * (torch.log(p.clamp_min(1e-12)) - logm)).sum(-1)
        kl_q = (q * (torch.log(q.clamp_min(1e-12)) - logm)).sum(-1)
        # log base 2 so the divergence is bounded in [0, 1]
        return (0.5 * (kl_p + kl_q) / float(np.log(2.0))).clamp(0.0, 1.0)

    def combine(self, z_ctx, z_par, factor, target):
        a = self._jsd(z_ctx, z_par).unsqueeze(-1) * factor
        if target != "use_context":
            a = -a
        self._last_alpha = float(a.abs().mean())
        return (1 + a) * z_ctx - a * z_par

    def steer_extras(self, item, condition, factor, target):
        return {"mean_adaptive_alpha": getattr(self, "_last_alpha", None),
                "scale": factor}


@register_steerer("ckplug")
class CKPlug(_ContrastiveDecoder):
    """CK-PLUG (Bi et al. 2025, arXiv:2503.15888; github.com/byronBBL/CK-PLUG).

    Reimplemented from the paper's equations, NOT vendored from their repo -
    check it against their code before quoting numbers against theirs.

    Two parts.

    **1. The gate (Eq. 7).**  Per token the confidence gain is

        CG_t = H(p_par,t) - H(p_ctx,t)        [bits, Eqs. 2-4]

    and only tokens with CG_t below the threshold are touched; everywhere
    else the context distribution passes through untouched.  That gate is
    what separates CK-PLUG from CAD, which reweights every token
    unconditionally.  The main text uses CG < 0; Appendix B uses a stricter
    entropy-relative bound, CG < eps * |H_ctx| with eps in {-1, -2, -3} set
    per model.  `gate_epsilon` selects between them and defaults to 0.0, the
    Eq. 7 form.

    Watch `gated_token_rate` in the records.  This harness scores
    teacher-forced answer tokens inside a prompt whose passage states an
    answer, and a passage that names an answer *sharpens* the distribution
    there - so CG > 0 and the token is skipped.  A near-zero gate rate means
    CK-PLUG is a no-op on exactly the tokens the DV is computed from, which
    is a property of the measurement, not a bug to patch away.

    **2. The blend (Eqs. 5, 6, 8).**  The paper mixes the parametric
    log-probs with a *log-ratio*, not with the context log-probs:

        q_para = log p_par
        q_cont = log p_ctx - log p_par            <- contrastive, Eq. 6
        F      = alpha * q_para + (1 - alpha) * q_cont

    with alpha = 1 fully parametric and alpha = 0 fully context-reliant.
    Writing a = 1 - alpha for context reliance, that is

        log p = a * log p_ctx + (1 - 2a) * log p_par

    Note the coefficient on the parametric term: **(1 - 2a), not (1 - a)**.
    The difference is not cosmetic.  Under the (1 - a) form this class used
    to implement, a = 1 gives plain log p_ctx - the *unsteered* model - so
    the method could only ever interpolate between "do nothing" and "go fully
    parametric", and its entire use_context arm was a no-op by construction.
    Under the paper's form a = 1 gives log p_ctx - log p_par, a genuine
    extrapolation away from memory (CAD with amplification 1).

    **Dose semantics.**  `factor` is a dose in [0, 1] like every other
    steerer here, mapped to a = 0.5 +/- 0.5 * factor so that factor = 0 is
    the neutral midpoint for BOTH targets and factor = 1 is the full push.
    Set `factor_is_alpha: true` to feed the paper's alpha in directly
    instead.  One honest caveat either way: a = 0.5 gives 0.5 * log p_ctx,
    which has the same *ranking* as the unsteered distribution but not the
    same probabilities, so unlike `cad` and `adacad` this method does not
    reduce to the unsteered margin exactly at dose 0.  That is a property of
    CK-PLUG's formula.  The matched control is the true unsteered margin
    regardless (see `_ContrastiveDecoder.steer`), so the comparison stays
    honest; the dose-0 row simply is not identical to it.

    **V_head (Eq. 9).**  The paper restricts the blended scores to the union
    of the two distributions' top-k sets and sends everything else to -inf.
    Implemented, but `select_top` defaults to 0 (off), because this harness
    scores a teacher-forced margin over specific answer strings: any answer
    token falling outside the mask gets log-prob -inf and the margin becomes
    non-finite and non-comparable with every other row in the table.  Turn it
    on only when generating.
    """

    def _alpha(self, factor, target):
        """-> (a, alpha) where a is context reliance and alpha the paper's."""
        f = float(factor)
        if self.cfg.get("factor_is_alpha", False):
            # the repo's original parameterisation: factor IS the
            # context-reliance knob a, so a = 0.5 is neutral, a = 1 full
            # context, a = 0 full parametric (the paper's alpha = 1 - a)
            a = f if target == "use_context" else 1.0 - f
        else:
            half = 0.5 * min(max(f, 0.0), 1.0)
            a = 0.5 + half if target == "use_context" else 0.5 - half
        a = min(max(a, 0.0), 1.0)
        return a, 1.0 - a

    @staticmethod
    def _v_head(q_para, q_cont, k):
        """Eq. 9: the union of the two top-k candidate sets."""
        k = min(int(k), q_para.shape[-1])
        keep = torch.zeros_like(q_para, dtype=torch.bool)
        for q in (q_para, q_cont):
            keep |= q >= q.topk(k, dim=-1).values[..., -1:]
        return keep

    def combine(self, z_ctx, z_par, factor, target):
        a, alpha = self._alpha(factor, target)
        logp_ctx = torch.log_softmax(z_ctx, -1)
        logp_par = torch.log_softmax(z_par, -1)

        q_para = logp_par                       # Eq. 5
        q_cont = logp_ctx - logp_par            # Eq. 6
        blended = alpha * q_para + (1.0 - alpha) * q_cont    # Eq. 8

        k = int(self.cfg.get("select_top", 0) or 0)
        if k > 0:
            blended = blended.masked_fill(
                ~self._v_head(q_para, q_cont, k), float("-inf"))

        # Eq. 7 / Appendix B: gate on the confidence gain, in bits
        ln2 = float(np.log(2.0))
        h_ctx = -(logp_ctx.exp() * logp_ctx).sum(-1) / ln2
        h_par = -(logp_par.exp() * logp_par).sum(-1) / ln2
        cg = h_par - h_ctx
        eps = float(self.cfg.get("gate_epsilon", 0.0))
        gated = cg < (eps * h_ctx.abs())
        self._last_gate_rate = float(gated.float().mean())
        self._last_cg = float(cg.mean())
        self._last_alpha = alpha
        return torch.where(gated.unsqueeze(-1), blended, logp_ctx)

    def steer_extras(self, item, condition, factor, target):
        return {"alpha": getattr(self, "_last_alpha", None),
                "context_reliance": 1.0 - getattr(self, "_last_alpha", 0.0),
                "gate_epsilon": float(self.cfg.get("gate_epsilon", 0.0)),
                "select_top": int(self.cfg.get("select_top", 0) or 0),
                "mean_confidence_gain_bits": getattr(self, "_last_cg", None),
                "gated_token_rate": getattr(self, "_last_gate_rate", None)}


@register_steerer("csks")
class CSKS(Steerer):
    """CSKS (Wang et al., EMNLP 2025; github.com/OliveJuiceLin/CSKS):
    proxy-model logit arithmetic; needs two fine-tuned small proxies
    (context-faithful / parametric-faithful). Heaviest to set up - phase 3.
    Note their proxy pair for Llama needs training on your cloze format."""
    access = "grey-box"

    def steer(self, item, condition, target, factor):
        raise NotImplementedError("clone OliveJuiceLin/CSKS; train proxies")
