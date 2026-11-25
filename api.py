"""The contract between the benchmark and a method.

A method gets a `PublicInstance` - the question, the context sentences and
the prompt built from them - and the model.  It never gets a label, `a`, `b`
or `d`.  It returns:

    Detector.score(inst)                        -> Detection
    Steerer.intervene(inst, direction, strength, span) -> Intervention

A detector returns numbers.  A steerer returns an *intervention*, and the
harness does the generation, so decoding settings, answer matching and cost
counting are the same for every method.

Trainable methods get `fit(train)`: a list of `(PublicInstance, labels)` from
the training folds only (folds are by relation).  `labels` is a small dict,
still without any answer string:

    conflict   1 / 0      the label of the task being fitted
    follows    "ctx" | "mem" | None   which source M's default answer d
               followed, when the two targets differ and d names one of them
    recipe     the taxonomy cell the instance was built from
    chunks     [1 / 0]    which sentences assert b

Two controls (`margin`, `logit_lens`) need the answer strings and are marked
`oracle = True`; the harness gives them the full instance record.  They are
reference rows, not methods.
"""
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Callable, Optional

from conflict_bench.core import prompts

DETECTORS: dict[str, type] = {}
STEERERS: dict[str, type] = {}

DIRECTIONS = ("ctx", "mem", "span")


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


# ------------------------------------------------------------------ input
@dataclass
class PublicInstance:
    instance_id: str
    question: Optional[str]          # the stem the answer continues; None in
    #                                  the global task, whose input is (c, M)
    passages: tuple                  # ((passage_id, (sentence, ...)), ...)
    format: str                      # "short" | "cloze"
    fold: int = -1
    key: str = ""                    # cache key of this view

    def __post_init__(self):
        self.key = self.key or self.instance_id

    @classmethod
    def from_record(cls, rec: dict) -> "PublicInstance":
        return cls(rec["instance_id"], rec["question"],
                   tuple((p["id"], tuple(c["text"] if isinstance(c, dict) else c
                                         for c in p["chunks"]))
                         for p in rec["passages"]),
                   rec["format"], rec.get("fold", -1))

    def public_record(self) -> dict:
        return {"instance_id": self.instance_id, "fold": self.fold,
                "format": self.format, "question": self.question,
                "passages": [{"id": pid, "chunks": list(ch)}
                             for pid, ch in self.passages]}

    # ---- what the model sees
    @cached_property
    def _rendered(self):
        return prompts.render(self.question, self.passages)

    @property
    def prompt(self) -> str:
        return self._rendered[0]

    @property
    def spans(self) -> dict:
        """Character spans in `prompt`: context, stem, chunks[i]."""
        return self._rendered[1]

    @property
    def chunks(self) -> list:
        return [c for _, ch in self.passages for c in ch]

    @property
    def passage_ids(self) -> list:
        return [pid for pid, _ in self.passages]

    @property
    def context_text(self) -> str:
        return " ".join(self.chunks)

    @property
    def prompt_no_context(self) -> str:
        return self.without_context().prompt

    def with_instruction(self, text: str) -> str:
        """The prompt with an instruction block added."""
        return prompts.render(self.question, self.passages, instruction=text)[0]

    # ---- derived views (each has its own cache key)
    def _view(self, tag, **changes):
        return replace(self, key=f"{self.key}#{tag}", **changes)

    def without_question(self):
        return self._view("noq", question=None)

    def without_context(self):
        return self._view("noctx", passages=())

    def only_passages(self, ids):
        ids = set(ids)
        return self._view("only" + "".join(sorted(ids)),
                          passages=tuple(p for p in self.passages if p[0] in ids))

    def without_chunk(self, i):
        out, n = [], 0
        for pid, ch in self.passages:
            out.append((pid, tuple(c for j, c in enumerate(ch) if n + j != i)))
            n += len(ch)
        return self._view(f"drop{i}", passages=tuple(out))


# ------------------------------------------------------------------ output
@dataclass
class Detection:
    conflict: float                         # higher = more conflict
    chunks: Optional[list] = None           # one score per sentence (localization)
    label: Optional[str] = None             # optional 4-way guess:
    #                                         supports | contradicts | novel | irrelevant
    extras: dict = field(default_factory=dict)


@dataclass
class Intervention:
    """What a steerer asks the harness to do while it generates.

    prompt   a replacement prompt (prompting, context deletion)
    hooks    zero-argument callable returning a context manager that installs
             forward hooks on the model (activation and component methods)
    logits   `fn(z, generated_ids) -> z`, applied to the next-token logits at
             every decoding step (the contrastive-decoding family)
    """
    prompt: Optional[str] = None
    hooks: Optional[Callable] = None
    logits: Optional[Callable] = None
    extras: dict = field(default_factory=dict)

    def installed(self):
        return self.hooks() if self.hooks else nullcontext()


# ------------------------------------------------------------------ methods
class Method:
    requires_training = False
    oracle = False                  # True: a control that is given the labels
    access = "activations"          # "text" | "logits" | "activations"

    def __init__(self, model, cfg=None):
        self.model = model
        self.cfg = cfg or {}

    def fit(self, train: list):
        """`train` = [(PublicInstance, labels), ...] from the training folds."""

    def save_artifacts(self, store, tag=""):
        """Persist fitted parameters (probe weights, steering vectors)."""
        return {}


class Detector(Method):
    def score(self, inst: PublicInstance) -> Detection:
        raise NotImplementedError


class Steerer(Method):
    #: the dial. A method without one keeps a single value.
    strengths = [1.0]

    def intervene(self, inst: PublicInstance, direction: str, strength: float,
                  span=None) -> Optional[Intervention]:
        """`direction` is "ctx", "mem" or "span"; `span` lists the passage ids
        to follow when it is "span".  Return None if the method does not
        support the direction."""
        raise NotImplementedError

    def control(self, inst, direction, strength, span=None) -> Optional[Intervention]:
        """The matched control: the same kind of edit with the directional
        content removed (a random direction of equal norm, a placebo
        instruction).  None when the unsteered model is the control."""
        return None
