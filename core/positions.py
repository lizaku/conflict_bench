"""Named token positions, resolved per instance.

A raw integer index is a different linguistic location on every instance once
contexts differ in length, so positions are named and resolved against the
character spans of the prompt blocks (core/prompts.render) through the
tokenizer's offset mapping:

    last             the final prompt token - the readout position
    end_of_stem      last token of the question / cloze stem
    end_of_context   last token of the context, before the question
    chunk:i          last token of the i-th sentence of the context

`last` is kept so the readout-vs-upstream gap can be measured, not assumed:
a probe there is largely decoding an answer the model has committed to.

There is deliberately no "position of the answer mention": where the answer
is in the context is a label, and a method does not get labels.

Requires a fast tokenizer (offset mapping); otherwise only `last` resolves.
"""

READOUT_POSITIONS = ("last", "end_of_stem")


class PositionError(ValueError):
    """A named position that does not exist for this instance."""


def _last_token_in(offsets, start, end):
    """Index of the last real token *ending* inside (start, end].

    Not strict containment: BPE tokens carry their leading space, so the token
    for " Monaco" starts one character before the span of "Monaco".
    """
    best = None
    for i, (s, e) in enumerate(offsets):
        if e > s and start < e <= end:
            best = i
    return best


def resolve(model, inst, position) -> int:
    """-> absolute token index into `model.encode(inst.prompt)`."""
    if isinstance(position, int) and not isinstance(position, bool):
        return position
    name = str(position)
    if name in ("-1", "last"):
        return -1
    spans = inst.spans
    if name == "end_of_stem":
        target = spans.get("stem")
    elif name == "end_of_context":
        target = spans.get("context")
    elif name.startswith("chunk:"):
        chunks = spans.get("chunks", [])
        i = int(name.split(":")[1])
        target = chunks[i] if i < len(chunks) else None
    else:
        raise PositionError(f"unknown position '{name}'")
    if target is None:
        raise PositionError(
            f"instance {inst.instance_id}: position '{name}' does not exist "
            f"(no such block in this prompt)")
    if not getattr(model.tok, "is_fast", False):
        raise PositionError(
            f"position '{name}' needs a fast tokenizer for offset mapping")
    idx = _last_token_in(model.offsets(inst.prompt), *target)
    if idx is None:
        raise PositionError(
            f"instance {inst.instance_id}: no token falls inside '{name}'")
    return idx


def is_readout(position) -> bool:
    return position in (-1, "-1") or str(position) in READOUT_POSITIONS
