"""Named probe positions, resolved per item.

`position: -1` in a config is not a probe location, it is a *sentence
position*: the last token of the prompt, which is exactly the state the model
uses to emit its next token.  A probe read there, scored against the model's
own greedy continuation from the same prompt, is largely decoding an answer
the model has already committed to - it does not show that arbitration is
represented anywhere upstream.  And a raw integer is not a fix either: token
index 137 is a different linguistic location on every item once passages
differ in length.

So positions are named and resolved per item, against the character spans of
the prompt blocks (core/prompts.PromptSpec.build_with_spans) mapped through
the tokenizer's offset mapping:

    end_of_context      last token of the passage, BEFORE the question is
                        re-stated - the earliest point at which the conflict
                        has been read but no answer has been formulated
    answer_mention_last last token of the counterfactual answer where the
                        passage states it - where the conflicting claim
                        actually enters the residual stream
    subject_last        last token of the subject mention in the question
    end_of_stem         last token of the question (before answer_prefix)
    last                the final prompt token - the readout position, kept
                        so the readout/upstream contrast can be measured
                        rather than assumed

Requires a fast tokenizer (offset mapping). Without one, only `last` and
integer positions resolve.
"""
from conflict_bench.core.types import Condition
from conflict_bench.core import prompts

NAMED = ("last", "end_of_stem", "end_of_context", "answer_mention_last",
         "subject_last", "first")

#: positions at or after which the model's answer is effectively determined,
#: i.e. where a probe is at serious risk of being a readout rather than a
#: detector. Used only for labelling in reports.
READOUT_POSITIONS = ("last", "end_of_stem")


class PositionError(ValueError):
    """A named position that does not exist for this (item, condition)."""


def _last_token_in(offsets, start, end):
    """Index of the last real token *ending* inside (start, end].

    Deliberately not strict containment: BPE tokens carry their leading
    space, so the token for " Monaco" starts one character before the span of
    "Monaco" and containment would reject the very token being asked for.
    A token straddling the far edge is excluded, so the result never reads
    past the span.
    """
    best = None
    for i, (s, e) in enumerate(offsets):
        if e > s and e <= end and e > start:
            best = i
    return best


def _find_span(text, needle, within):
    """Char span of the last occurrence of `needle` inside `within`."""
    if not needle:
        return None
    lo, hi = within
    hay, ndl = text[lo:hi].lower(), needle.lower()
    idx = hay.rfind(ndl)
    if idx < 0:
        return None
    return (lo + idx, lo + idx + len(ndl))


def resolve(model, item, condition, position, prompt=None, spec=None):
    """-> absolute token index for this (item, condition).

    Integers pass through unchanged (still supported, still a bad idea for
    anything but -1).
    """
    if isinstance(position, (int,)) and not isinstance(position, bool):
        return position
    name = str(position)
    if name in ("-1", "last"):
        return -1
    if name not in NAMED:
        raise PositionError(
            f"unknown position '{name}'; expected an int or one of {NAMED}")

    spec = spec or prompts.DEFAULT
    text, spans = spec.build_with_spans(item, condition)
    if prompt is not None and prompt != text:
        raise PositionError("prompt/spec mismatch while resolving a position")

    if not getattr(model.tok, "is_fast", False):
        raise PositionError(
            f"position '{name}' needs a fast tokenizer for offset mapping; "
            f"{model.model_name} has a slow one - use position: -1 (and read "
            f"the caveat in core/positions.py)")
    enc = model.tok(text, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    if name == "first":
        return 0
    if name == "end_of_stem":
        target = spans["stem"]
    elif name == "end_of_context":
        if "background" not in spans:
            raise PositionError(
                f"condition {getattr(condition, 'value', condition)} has no "
                f"passage, so 'end_of_context' does not exist for it")
        target = spans["background"]
    elif name == "answer_mention_last":
        if "background" not in spans:
            raise PositionError(
                "'answer_mention_last' needs a passage to find the answer in")
        target = None
        for cand in [item.counterfactual_answer] + list(item.cf_aliases):
            target = _find_span(text, cand, spans["background"])
            if target:
                break
        if not target:
            raise PositionError(
                f"item {item.item_id}: counterfactual answer "
                f"'{item.counterfactual_answer}' does not occur verbatim in "
                f"its passage")
    elif name == "subject_last":
        target = _find_span(text, item.subject, spans["stem"])
        if not target:
            raise PositionError(
                f"item {item.item_id}: subject '{item.subject}' does not "
                f"occur verbatim in the question")
    idx = _last_token_in(offsets, *target)
    if idx is None:
        raise PositionError(
            f"item {item.item_id}: no token falls inside the '{name}' span")
    return idx


def is_readout(position) -> bool:
    """True for positions at which the model's next token is already decided."""
    if isinstance(position, int) and not isinstance(position, bool):
        return position in (-1,)
    return str(position) in READOUT_POSITIONS
