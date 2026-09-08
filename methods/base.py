"""Abstract interfaces + registry.

Two axes, mirroring AxBench's C (detection) and S (steering), but here:
  C  = "will/does the model follow the context over its parametric answer?"
       (per-item score; mechanistic methods read activations,
        output-based methods read samples/logits/verbal reports)
  S  = "can we push arbitration toward context or toward memory?"
       (intervention; mechanistic = activation-level,
        output-based = prompt / decoding / proxy-model level)

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

    def save_artifacts(self, store, tag=""):
        """Persist fitted parameters / tensors. `store` is an ArtifactStore
        already scoped to this method; `tag` distinguishes CV folds.
        Training-free methods have nothing to write."""
        return {}


class Detector(Method):
    """Produces a scalar 'context-following' score per (item, condition)."""

    def fit(self, items: list[Item], labels: list[int]):
        """Train on labelled items (behavioural label: 1 = followed context).
        Runner guarantees GroupKFold-by-relation: fit() only ever sees
        train folds. No-op for training-free methods."""
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
        ...

    def control(self, item: Item, condition: Condition, target: str,
                factor: float) -> float | None:
        """Norm-matched random-direction (or otherwise matched) control.
        Return margin_after under the control, or None if not applicable
        (e.g. prompting has no natural random control; report vs. no-op)."""
        return None
