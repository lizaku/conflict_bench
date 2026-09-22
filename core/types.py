"""Core data types for the unified conflict-arbitration benchmark.

Design mirrors the v3 pipeline (context_vs_parametric3.py):
  - within-item paired conditions N / S / C / R
  - candidate answers scored by teacher-forced log-prob
  - continuous margin as the primary DV, format-corrected via R baseline
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Condition(str, Enum):
    NORMAL = "N"        # no passage
    SUPPORTING = "S"    # passage consistent with parametric answer
    CONFLICTING = "C"   # counterfactual passage
    IRRELEVANT = "R"    # random passage (format baseline)


@dataclass
class Item:
    """One knowledge-conflict item (one Wikidata cloze record or one QA pair)."""
    item_id: str
    relation: str                       # e.g. "P17" — grouping key for GroupKFold
    cloze_template: str                 # "The head of government of Monaco is ____."
    subject: str
    true_answer: str
    counterfactual_answer: str
    question: str = ""                  # "Who is the head of government of Monaco?"
    true_aliases: list[str] = field(default_factory=list)
    cf_aliases: list[str] = field(default_factory=list)
    passages: dict[str, str] = field(default_factory=dict)  # keys: "S", "C", "R"
    # R is synthesised by the loader (a passage from an unrelated item) — it is
    # the format baseline the margin is corrected against, so it must exist.
    distractors: dict[str, list[str]] = field(default_factory=dict)  # ds / dx / dw
    meta: dict = field(default_factory=dict)

    @property
    def first_token_collision(self) -> Optional[bool]:
        return self.meta.get("first_token_collision")


@dataclass
class ScoredCandidates:
    """Teacher-forced log-probs for one (item, condition)."""
    item_id: str
    condition: Condition
    logp_true: float
    logp_cf: float
    logp_distractors: dict[str, float] = field(default_factory=dict)

    @property
    def margin(self) -> float:
        """Positive => parametric answer wins; negative => contextual wins."""
        return self.logp_true - self.logp_cf


@dataclass
class DetectionRecord:
    """Output of a Detector for one (item, condition)."""
    item_id: str
    relation: str
    condition: Condition
    method: str
    score: float                 # the PRIMARY score; see the convention below
    # Convention depends on the run's detection_task (methods/base.py):
    #   conflict     HIGHER = the passage contradicts the model's knowledge
    #   arbitration  HIGHER = the model will follow the context
    label: Optional[int] = None  # conflict: 1 = C (conflicting passage)
    #                              arbitration: 1 = followed context behaviourally
    # The two views of the same score, reported side by side. `score` is
    # whichever one the run's `format_correct` selected; the other is kept so
    # a conclusion that only survives one of them is visible as exactly that.
    score_raw: Optional[float] = None        # what was observed
    score_corrected: Optional[float] = None  # after subtracting the R margin
    extras: dict = field(default_factory=dict)


@dataclass
class SteeringRecord:
    """Output of a Steerer for one (item, condition, factor)."""
    item_id: str
    relation: str
    condition: Condition
    method: str
    target: str                  # "use_parametric" | "use_context"
    factor: float
    margin_before: float         # the PRIMARY pair: raw unless the run set
    margin_after: float          # format_correct: true, in which case R is
    #                              subtracted from both
    flipped: bool                # sign flip of the primary margin
    # Both views, always, whichever one `margin_before/after` mirrors:
    margin_before_raw: Optional[float] = None         # what was observed
    margin_after_raw: Optional[float] = None
    flipped_raw: Optional[bool] = None
    margin_before_corrected: Optional[float] = None   # after subtracting R,
    margin_after_corrected: Optional[float] = None    # None when R is absent
    flipped_corrected: Optional[bool] = None
    r_offset: Optional[float] = None            # the margin under R itself
    generated: Optional[str] = None
    fluency: Optional[float] = None      # e.g. perplexity of continuation
    control_margin_after: Optional[float] = None  # the method's matched control:
    control_margin_after_raw: Optional[float] = None  # a norm-matched random
    # direction (activation methods), a placebo instruction of the same shape
    # (prompting), or the unsteered distribution (the decoding family, which
    # reduces to it exactly). None means the method declared no control, and
    # metrics.steering_summary reports specific_effect as NaN rather than
    # silently equating it with the raw delta.
    extras: dict = field(default_factory=dict)
