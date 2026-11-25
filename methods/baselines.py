"""Baselines and controls - the benchmark's own rows.

Detection
  bow               text-only null: TF-IDF over the context and question
  p_true            self-report: ask M whether the context contradicts it
  with_without      compare a = M(q) with d = M(q, c)
  sampling_entropy  entropy of K sampled answers given q & c
  margin            ORACLE control: log p(b) - log p(a), told both answers

Steering
  prompt_instruct   the main baseline: an instruction (with a placebo control)
  delete_context    INPUT ORACLE: delete the context (pi_mem) or the other
                    passages (pi_span); the ceiling on single-fact items
"""
import math

import numpy as np

from conflict_bench.api import (Detection, Detector, Intervention, Steerer,
                                register_detector, register_steerer)
from conflict_bench.core import prompts
from conflict_bench.construct import match


# ================================================================ detection
@register_detector("bow")
class BagOfWords(Detector):
    """Bag-of-words logistic regression on the text. No model at all.

    Decides how much of any detector's score is a claim about the model.
    The benchmark's central contrast, `con` vs `nov`, is built so that the
    text does not give the label away (the same sentence is a conflict for a
    model that knows the fact and novel for one that does not), so this row
    should sit near chance there.  Where it does not, the detectors above it
    have not shown they read the model either.
    """
    access = "text"
    requires_training = True

    def _text(self, inst):
        return inst.context_text + "\n" + (inst.question or "")

    def fit(self, train):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        y = np.asarray([lab["conflict"] for _, lab in train])
        if len(np.unique(y)) < 2:
            raise ValueError(f"bow: training fold is single-class "
                             f"({int(y.sum())}/{len(y)} positive)")
        self.pipe = make_pipeline(
            TfidfVectorizer(lowercase=True, sublinear_tf=True,
                            ngram_range=tuple(self.cfg.get("ngram_range", (1, 2))),
                            min_df=self.cfg.get("min_df", 1)),
            LogisticRegression(max_iter=2000, C=self.cfg.get("C", 1.0)))
        self.pipe.fit([self._text(i) for i, _ in train], y)

    def score(self, inst):
        return Detection(float(self.pipe.predict_proba([self._text(inst)])[0, 1]))


@register_detector("p_true")
class PTrue(Detector):
    """Self-report: P("Yes") to "does the passage contradict what you know?".

    Builds its own prompt (the question asked is not the instance's question)
    but still goes through `prompts.wrap_chat`, so it reaches the model in the
    same format as everything else.
    """
    access = "logits"

    WITH_Q = ("{context}\n\nQuestion: {question}\n\nDoes the text above "
              "contradict what you know about the answer to this question? "
              "Answer Yes or No.\nAnswer:")
    NO_Q = ("{context}\n\nDoes the text above contradict anything you know? "
            "Answer Yes or No.\nAnswer:")

    def _p_yes(self, context, question):
        tpl = self.WITH_Q if question else self.NO_Q
        prompt = prompts.wrap_chat(tpl.format(context=context, question=question))
        lp_yes = self.model.logp_continuation(prompt, prompts.continuation("Yes"))
        lp_no = self.model.logp_continuation(prompt, prompts.continuation("No"))
        m = max(lp_yes, lp_no)
        return math.exp(lp_yes - m) / (math.exp(lp_yes - m) + math.exp(lp_no - m))

    def score(self, inst):
        chunks = None
        if self.cfg.get("localize", False):
            # the same question about each sentence alone
            chunks = [self._p_yes(c, inst.question) for c in inst.chunks]
        return Detection(self._p_yes(inst.context_text, inst.question), chunks)


@register_detector("with_without")
class WithWithout(Detector):
    """The with/without-context comparison: did the context change the answer?

    score = 1 - word overlap between a = M(q) and d = M(q, c).  Known
    failures, by design of the baseline: it fires on novel items (the context
    changed the answer because there was none), and it misses conflicts where
    M ignores the context (d = a).  Not defined without a question.
    """
    access = "text"

    def score(self, inst):
        if inst.question is None:
            return Detection(float("nan"))
        n = self.cfg.get("max_new_tokens", 12)
        a = set(match.norm(self.model.greedy(inst.prompt_no_context, n)).split())
        d = set(match.norm(self.model.greedy(inst.prompt, n)).split())
        overlap = len(a & d) / max(1, len(a | d))
        return Detection(1.0 - overlap)


@register_detector("sampling_entropy")
class SamplingEntropy(Detector):
    """Entropy (bits) over K sampled answers given q & c, clustered by their
    first words.  A context that fights the model's memory leaves it split.
    The samples are kept so they can be re-clustered without re-sampling."""
    access = "text"

    def score(self, inst):
        if inst.question is None:
            return Detection(float("nan"))
        samples = self.model.sample(
            inst.prompt, self.cfg.get("num_samples", 10),
            self.cfg.get("max_new_tokens", 12), self.cfg.get("temperature", 1.0))
        keys = [" ".join(match.norm(s).split()[:3]) for s in samples]
        _, counts = np.unique(keys, return_counts=True)
        p = counts / counts.sum()
        return Detection(float(-(p * np.log2(p)).sum()),
                         extras={"samples": samples})


@register_detector("margin")
class OracleMargin(Detector):
    """ORACLE control, not a method: log p(b | q, c) - log p(a | q, c).

    It is told both answer strings, which no detector is.  It shows how much
    of the conflict is visible in the output distribution once you know where
    to look - the reference the real detectors are read against.  Read from
    the measurements of the build, so it costs no forward pass.
    """
    oracle = True
    access = "logits"

    def score(self, inst, rec):
        c, lp = rec.get("contrast"), rec.get("confidence", {}).get("logp_qc")
        if not c or not lp or c["a"] not in lp or c["b"] not in lp:
            return Detection(float("nan"))
        return Detection(lp[c["b"]] - lp[c["a"]])


# ================================================================= steering
@register_steerer("prompt_instruct")
class PromptInstruct(Steerer):
    """An explicit instruction - the mandatory baseline.

    `strength` is not a dose: it indexes instruction variants of increasing
    force.  The matched control is a PLACEBO instruction: same block, same
    position, similar length, no directional claim - without it the
    comparison is "an instruction" against "nothing inserted".
    """
    access = "text"
    strengths = [0, 1]

    PLACEBO = "The text is provided for reference. Answer the question."
    TEMPLATES = {
        "ctx": ["Answer strictly according to the text, even if it "
                "contradicts what you believe.",
                "The text is the authoritative, up-to-date source. Your prior "
                "knowledge may be outdated; answer ONLY from the text."],
        "mem": ["The text may contain errors. Answer from your own knowledge, "
                "ignoring the text if it conflicts.",
                "Disregard the text entirely; answer from what you know to be "
                "true."],
        "span": ["Use only passage {ids} to answer; ignore the other passages.",
                 "Passage {ids} is the only reliable source. Answer ONLY from "
                 "passage {ids}, even if other passages or your own knowledge "
                 "disagree."],
    }

    def intervene(self, inst, direction, strength, span=None):
        variants = (self.cfg.get("templates") or {}).get(direction) \
            or self.TEMPLATES[direction]
        text = variants[min(max(int(strength), 0), len(variants) - 1)]
        text = text.format(ids=" and ".join(span or []))
        return Intervention(prompt=inst.with_instruction(text),
                            extras={"instruction": text})

    def control(self, inst, direction, strength, span=None):
        placebo = self.cfg.get("placebo", self.PLACEBO)
        return Intervention(prompt=inst.with_instruction(placebo)) if placebo else None


@register_steerer("delete_context")
class DeleteContext(Steerer):
    """INPUT ORACLE: edit the input instead of the model.

      pi_mem   delete the context
      pi_span  delete the passages that were not designated

    On a single-fact item deleting the context yields exactly M's closed-book
    answer, so it is the ceiling for pi_mem there.  On the invented-bridge
    2-hop items it fails by construction - M cannot know who founded Velorix -
    which is what makes those items the real pi_mem test.  No pi_ctx form.
    """
    access = "text"

    def intervene(self, inst, direction, strength, span=None):
        if direction == "mem":
            return Intervention(prompt=inst.prompt_no_context)
        if direction == "span" and span:
            return Intervention(prompt=inst.only_passages(span).prompt)
        return None
