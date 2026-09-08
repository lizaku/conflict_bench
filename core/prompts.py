"""Prompt construction — the single source of truth for every method.

Default format (`prompt.style: question`, what the pilot runs use):

    N          ->  "{question}"
    S / C / R  ->  "Background: {passage}\n\n{question}"

A steering instruction (prompt_instruct) is inserted as its own block between
the background and the question.  Continuations are scored teacher-forced as
" {answer}", so a prompt never ends in whitespace.

Everything here is data, not code: change the format in the config
(`prompt:` block) rather than in a method, or the conditions stop being
comparable and the R-based format correction becomes meaningless.
"""
from dataclasses import dataclass, asdict

from conflict_bench.core.types import Item, Condition


@dataclass
class PromptSpec:
    style: str = "question"          # "question" | "cloze"
    background_prefix: str = "Background: "
    separator: str = "\n\n"
    answer_prefix: str = ""          # e.g. "\nAnswer:" — appended after the stem
    strict: bool = True              # raise if a non-N condition has no passage

    def stem(self, item: Item) -> str:
        """The part the answer continues."""
        if self.style == "cloze":
            t = (item.cloze_template or "").strip()
            for blank in ("____.", "____"):
                if t.endswith(blank):
                    return t[: -len(blank)].rstrip()
            # blank is mid-sentence (868/6000 items): no terminal stem exists,
            # fall through to the question form rather than mangle the item.
        return item.question or item.cloze_template

    def build(self, item: Item, condition: Condition, instruction: str = "") -> str:
        return self.build_with_spans(item, condition, instruction)[0]

    def build_with_spans(self, item: Item, condition: Condition,
                         instruction: str = ""):
        """The prompt, plus the character span of each block.

        The spans are what makes a *named* probe position possible: a raw
        integer index is a different linguistic location on every item once
        passages differ in length, so `end_of_context` has to be resolved per
        item rather than configured as a number (see core/positions.py).
        """
        parts = []
        if condition != Condition.NORMAL:
            passage = item.passages.get(condition.value, "")
            if not passage and self.strict:
                raise ValueError(
                    f"item {item.item_id}: no passage for condition "
                    f"{condition.value} — the R baseline and the S/C contrast "
                    f"both need one (see data.load_confiqa_cloze_clusters)")
            if passage:
                parts.append(("background", self.background_prefix + passage))
        if instruction:
            parts.append(("instruction", instruction.strip()))
        parts.append(("stem", self.stem(item)))

        text, spans = "", {}
        for i, (label, chunk) in enumerate(parts):
            if i:
                text += self.separator
            start = len(text)
            text += chunk
            spans[label] = (start, len(text))
        text += self.answer_prefix
        return text, spans

    def as_dict(self):
        return asdict(self)


DEFAULT = PromptSpec()


def configure(**kwargs) -> PromptSpec:
    """Called once by the runner from cfg['prompt']; every method then sees it."""
    global DEFAULT
    DEFAULT = PromptSpec(**{k: v for k, v in kwargs.items() if v is not None})
    return DEFAULT


def build_prompt(item: Item, condition: Condition, instruction: str = "",
                 spec: PromptSpec = None) -> str:
    return (spec or DEFAULT).build(item, condition, instruction)
