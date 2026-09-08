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
    score: float                 # higher = predicts context-following (convention!)
    label: Optional[int] = None  # 1 = model followed context behaviourally
    score_raw: Optional[float] = None   # uncorrected counterpart of `score`,
    # for methods whose score is format-corrected against R. `score` is what
    # the summary AUROC uses; both are reported, because a conclusion that
    # only survives one of them is a conclusion about the passage template.
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
    margin_before: float         # format-corrected (R subtracted) unless the
    margin_after: float          # method sets format_correct: false
    flipped: bool                # sign flip of the corrected margin
    margin_before_raw: Optional[float] = None   # what was actually observed,
    margin_after_raw: Optional[float] = None    # before the R subtraction
    flipped_raw: Optional[bool] = None
    r_offset: Optional[float] = None            # the margin under R itself
    generated: Optional[str] = None
    fluency: Optional[float] = None      # e.g. perplexity of continuation
    control_margin_after: Optional[float] = None  # norm-matched random direction
    control_margin_after_raw: Optional[float] = None
    extras: dict = field(default_factory=dict)
