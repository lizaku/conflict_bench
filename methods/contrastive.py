"""The contrastive-decoding family: run the model with and without the context.

Detection (how different are the two next-token distributions?)
  confidence_gain   CK-PLUG's signal: the entropy change the context causes
  context_jsd / context_kl   CAD's and AdaCAD's signal

Steering (push further along the difference) - each one is a logits function
applied at every decoding step, so it is scored on generated answers like
every other method:
  cad      z = (1 + a) z_ctx - a z_par
  adacad   per-token a_t = JSD(p_ctx || p_par)
  ckplug   entropy-gated blend of parametric and contrastive log-probs
"""
import numpy as np
import torch

from conflict_bench.api import (Detection, Detector, Intervention, Steerer,
                                register_detector, register_steerer)

LN2 = float(np.log(2.0))


def entropy_bits(z):
    """Shannon entropy of softmax(z) in bits - CK-PLUG Eqs. 2-3 use log2."""
    logp = torch.log_softmax(z, -1)
    return float(-(logp.exp() * logp).sum() / LN2)


def jsd_bits(z_p, z_q):
    """Jensen-Shannon divergence, log base 2, so it is bounded in [0, 1]."""
    p, q = torch.softmax(z_p, -1), torch.softmax(z_q, -1)
    m = 0.5 * (p + q)
    logm = torch.log(m.clamp_min(1e-12))
    kl_p = (p * (torch.log(p.clamp_min(1e-12)) - logm)).sum(-1)
    kl_q = (q * (torch.log(q.clamp_min(1e-12)) - logm)).sum(-1)
    return (0.5 * (kl_p + kl_q) / LN2).clamp(0.0, 1.0)


# ================================================================ detection
@register_detector("confidence_gain")
class ConfidenceGain(Detector):
    """CK-PLUG's confidence gain (Bi et al. 2025, arXiv:2503.15888, Eq. 4):

        CG = H(p(x | question)) - H(p(x | context, question))      [bits]

    CG < 0: the context made the model *less* certain, the paper's signature
    of a conflict; CG > 0: the context agreed with what it believed.  So
    score = -CG.  The paper validates CG only as a gate for its decoder and
    reports no AUROC for it as a classifier - there is no published number to
    match.  Not defined without a question (there is nothing to be certain of).
    """
    access = "logits"

    def score(self, inst):
        if inst.question is None:
            return Detection(float("nan"))
        h_par = entropy_bits(self.model.next_logits(inst.prompt_no_context))
        h_ctx = entropy_bits(self.model.next_logits(inst.prompt))
        cg = h_par - h_ctx
        return Detection(-cg, extras={"H_parametric": h_par, "H_context": h_ctx})


@register_detector("context_jsd")
class ContextShift(Detector):
    """How far the context moved the next-token distribution:

        score = JSD(p(. | context, question) || p(. | question))

    The quantity AdaCAD uses as its adaptive weight and CAD amplifies with a
    fixed one.  It measures that the context changed the distribution, not
    that it changed it away from what the model knows - a novel fact moves it
    too.  With `localize: true`, sentence i is scored by how much of the
    shift disappears when it is left out.
    """
    access = "logits"
    divergence = "jsd"

    def _div(self, z_p, z_q):
        if self.cfg.get("divergence", self.divergence) == "kl":
            p = torch.softmax(z_p, -1)
            return float((p * (torch.log_softmax(z_p, -1)
                               - torch.log_softmax(z_q, -1))).sum())
        return float(jsd_bits(z_p, z_q))

    def score(self, inst):
        if inst.question is None:
            return Detection(float("nan"))
        z_par = self.model.next_logits(inst.prompt_no_context)
        z_ctx = self.model.next_logits(inst.prompt)
        full = self._div(z_ctx, z_par)
        chunks = None
        if self.cfg.get("localize", False):
            chunks = [full - self._div(
                self.model.next_logits(inst.without_chunk(i).prompt), z_par)
                for i in range(len(inst.chunks))]
        return Detection(full, chunks)


@register_detector("context_kl")
class ContextShiftKL(ContextShift):
    """`context_jsd` with KL(p_ctx || p_par) instead: unbounded, asymmetric."""
    divergence = "kl"


# ================================================================= steering
class _ContrastiveDecoder(Steerer):
    """Shared core: at every decoding step get the next-token logits with the
    context (z_ctx) and without it (z_par) and combine them.  Each method
    supplies `combine(z_ctx, z_par, strength, toward_context) -> z`.

    pi_ctx / pi_mem contrast the full prompt with the question alone.
    pi_span contrasts the full context with the context minus the designated
    passages, and amplifies what the designated passages add.
    """
    access = "logits"

    def combine(self, z_ctx, z_par, strength, toward_context):
        raise NotImplementedError

    def intervene(self, inst, direction, strength, span=None):
        if direction == "span":
            if not span:
                return None
            rest = [p for p in inst.passage_ids if p not in span]
            base, toward = inst.only_passages(rest).prompt, True
        else:
            base, toward = inst.prompt_no_context, direction == "ctx"
        model = self.model

        def logits(z_ctx, generated):
            return self.combine(z_ctx, model.next_logits(base, generated),
                                strength, toward)
        return Intervention(logits=logits)


@register_steerer("cad")
class ContextAwareDecoding(_ContrastiveDecoder):
    """CAD (Shi et al., NAACL 2024):  z = (1 + a) z_ctx - a z_par.

    A fixed amplification of whatever the context did to the distribution.
    a > 0 pushes toward the context, a < 0 toward memory.  Strength 0 is the
    unsteered model exactly.
    """
    strengths = [0.5, 1.0, 2.0]

    def combine(self, z_ctx, z_par, strength, toward_context):
        a = strength if toward_context else -strength
        return (1 + a) * z_ctx - a * z_par


@register_steerer("adacad")
class AdaptiveContextAwareDecoding(_ContrastiveDecoder):
    """AdaCAD (Wang et al. 2025, arXiv:2409.07394): CAD with a per-token
    weight proportional to how much the two distributions disagree,

        a_t = s * JSD(p_ctx,t || p_par,t),   z_t = (1 + a_t) z_ctx,t - a_t z_par,t

    s = 1 is AdaCAD as published.  The push toward memory negates a_t - an
    extrapolation past the paper, which is posed purely as a
    context-faithfulness method.
    """
    strengths = [0.5, 1.0, 2.0]

    def combine(self, z_ctx, z_par, strength, toward_context):
        a = float(jsd_bits(z_ctx, z_par)) * strength
        if not toward_context:
            a = -a
        return (1 + a) * z_ctx - a * z_par


@register_steerer("ckplug")
class CKPlug(_ContrastiveDecoder):
    """CK-PLUG (Bi et al. 2025, arXiv:2503.15888).  Reimplemented from the
    paper's equations, NOT vendored from byronBBL/CK-PLUG - check against
    their code before quoting numbers against theirs.

    The gate (Eq. 7).  Per token, CG_t = H(p_par,t) - H(p_ctx,t) in bits.
    Only tokens with CG_t < eps * |H_ctx| are touched (`gate_epsilon`,
    default 0.0 = the main-text CG < 0 form; Appendix B uses -1 / -2 / -3 per
    model).  Everywhere else the context distribution passes through.

    The blend (Eqs. 5, 6, 8).  The paper mixes the parametric log-probs with
    a log-RATIO, q_cont = log p_ctx - log p_par:

        F = alpha * log p_par + (1 - alpha) * q_cont

    With a = 1 - alpha for context reliance this is
    a * log p_ctx + (1 - 2a) * log p_par.  The coefficient is (1 - 2a), not
    (1 - a): under (1 - a), a = 1 is the unsteered model and the push toward
    the context is a no-op by construction.

    `strength` is a dose in [0, 1], mapped to a = 0.5 +/- 0.5 * strength, so
    0 is the neutral midpoint for both directions.  `factor_is_alpha: true`
    feeds the context-reliance knob in directly.  Unlike CAD, dose 0 is not
    the unsteered model exactly: a = 0.5 gives 0.5 * log p_ctx, same ranking,
    different probabilities.

    V_head (Eq. 9): restrict to the union of the two top-k sets
    (`select_top`, default 0 = off; with it on, a masked answer token gets a
    teacher-forced log-prob of -inf).
    """
    strengths = [0.5, 1.0]

    def _a(self, strength, toward_context):
        f = float(strength)
        if self.cfg.get("factor_is_alpha", False):
            a = f if toward_context else 1.0 - f
        else:
            half = 0.5 * min(max(f, 0.0), 1.0)
            a = 0.5 + half if toward_context else 0.5 - half
        return min(max(a, 0.0), 1.0)

    def combine(self, z_ctx, z_par, strength, toward_context):
        alpha = 1.0 - self._a(strength, toward_context)
        logp_ctx = torch.log_softmax(z_ctx, -1)
        logp_par = torch.log_softmax(z_par, -1)
        q_para = logp_par                                     # Eq. 5
        q_cont = logp_ctx - logp_par                          # Eq. 6
        blended = alpha * q_para + (1.0 - alpha) * q_cont     # Eq. 8

        k = int(self.cfg.get("select_top", 0) or 0)
        if k > 0:                                             # Eq. 9
            k = min(k, q_para.shape[-1])
            keep = torch.zeros_like(q_para, dtype=torch.bool)
            for q in (q_para, q_cont):
                keep |= q >= q.topk(k, dim=-1).values[..., -1:]
            blended = blended.masked_fill(~keep, float("-inf"))

        h_ctx = -(logp_ctx.exp() * logp_ctx).sum(-1) / LN2
        h_par = -(logp_par.exp() * logp_par).sum(-1) / LN2
        gated = (h_par - h_ctx) < float(self.cfg.get("gate_epsilon", 0.0)) * h_ctx.abs()
        return blended if bool(gated) else logp_ctx           # Eq. 7
