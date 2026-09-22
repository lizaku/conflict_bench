"""Abstract interfaces + registry.

Two axes, mirroring AxBench's C (detection) and S (steering).  Both ask one
question, so that a detector and a steerer can be compared on the same items
under the same conditions:

  C  = "Does this passage contradict the model's internal knowledge?"
       Scored per (item, condition) over the S-vs-C contrast: the same item
       is presented once with a supporting passage and once with a
       conflicting one, and the label is which one it was.  The design is
       paired, so the per-item answer-string constant cancels exactly and the
       base rate is 0.5 by construction.
       Score convention: HIGHER = more conflict.

  S  = "Given a conflicting passage, can the intervention make the model
       output the conflicting answer when it otherwise would not?"
       Every steerer is applied under C (where the conflict is) and under S
       (where there is none), so the S arm is a matched specificity control
       on the same item rather than a separate synthetic baseline.

The legacy arbitration framing - "will the model follow the context?", scored
under C alone against a behavioural label - is still runnable via
`detection_task: arbitration`, because the two questions are genuinely
different and the older numbers should stay reproducible.  Under that task
the score convention is HIGHER = more context-following; a detector whose
orientation differs between the two tasks flips on `self.task` (see
`confidence_gain`, which is the clearest case).

Every method implements one interface so the runner is method-agnostic.

Both interfaces expose `requires_training`.  When it is True the runner fits
the method **inside a GroupKFold-by-relation split** and only ever applies it
to held-out relations - that applies to steering vectors as much as to
probes, because a CAA vector fitted on the items it then steers is the same
leak the 0.822-vs-0.936 postmortem was about.

`save_artifacts(store)` is where a method writes whatever a summary CSV
cannot hold: probe coefficients, steering vectors, per-fold metadata.
"""
from abc import ABC, abstractmethod

from conflict_bench.core.types import (Item, Condition, DetectionRecord,
                                       SteeringRecord)

DETECTORS: dict[str, type] = {}
STEERERS: dict[str, type] = {}


def register_detector(name):
    def deco(cls):
        cls.name = name
        DETECTORS[name] = cls
        return cls
    return deco


def register_steerer(name):
    def deco(cls):
        cls.name = name
        STEERERS[name] = cls
        return cls
    return deco


class Method(ABC):
    requires_training = False
    uses_position = False       # True => the runner sweeps `detection_positions`
    access = "white-box"        # "white-box" | "grey-box" (logits) | "black-box"

    def __init__(self, model, cfg=None):
        self.model = model
        self.cfg = cfg or {}
        # "conflict" (S-vs-C) or "arbitration" (the legacy per-item label).
        # The runner injects it into every method's cfg so a method never has
        # to guess which question it is being asked.
        self.task = self.cfg.get("task", "conflict")

    def save_artifacts(self, store, tag=""):
        """Persist fitted parameters / tensors. `store` is an ArtifactStore
        already scoped to this method; `tag` distinguishes CV folds.
        Training-free methods have nothing to write."""
        return {}


class Detector(Method):
    """Produces a scalar score per (item, condition).

    HIGHER means more conflict under the `conflict` task, more
    context-following under `arbitration`.
    """

    def fit(self, items: list[Item], labels: list[int],
            conditions: list[Condition] = None):
        """Train on labelled instances. Runner guarantees
        GroupKFold-by-relation: fit() only ever sees train folds.

        `conditions[i]` is the condition instance i was presented under, and
        it is NOT constant under the conflict task - the same item appears
        once as S and once as C, and the label is exactly which.  A trainable
        detector that ignores it would read both halves of every pair from the
        same activations and learn nothing.  None means "all CONFLICTING",
        the legacy arbitration case.  No-op for training-free methods.
        """
        pass

    @abstractmethod
    def score(self, item: Item, condition: Condition) -> DetectionRecord:
        ...


class Steerer(Method):
    """Intervenes to push arbitration toward `target`, sweeping `factor`."""

    def fit(self, items: list[Item], labels: list[int] = None):
        """E.g. compute CAA vectors from contrast pairs; train proxy models.
        Gets the behavioural labels because the contrast that defines the
        arbitration direction is behavioural, not textual."""
        pass

    @abstractmethod
    def steer(self, item: Item, condition: Condition, target: str,
              factor: float) -> SteeringRecord:
        """`condition` is C for the real measurement and S for the matched
        specificity control - the same intervention on the same item where
        there is no conflict to resolve.  A steerer must therefore not assume
        C; anything that needs "the answer the passage asserts" should ask
        `core.positions.asserted_answers`."""
        ...

    def control(self, item: Item, condition: Condition, target: str,
                factor: float) -> float | None:
        """Norm-matched random-direction (or otherwise matched) control.
        Return margin_after under the control, or None if not applicable
        (e.g. prompting has no natural random control; report vs. no-op)."""
        return None
