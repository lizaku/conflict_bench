"""Prompt construction — the single source of truth for every method.

Default format (`prompt.style: question`, what the pilot runs use):

    N          ->  "{question}"
    S / C / R  ->  "Background: {passage}\n\n{question}"

A steering instruction (prompt_instruct) is inserted as its own block. It goes
**before** the background by default (`instruction_position: before_context`),
because an instruction wedged between a long passage and the question reads as
another sentence of the document rather than as a directive addressed to the
model — which is how every paper in this literature places it (Longpre et al.;
the faithfulness-prompting line; CK-PLUG's own `instr` schema). Set
`instruction_position: after_context` for the older behaviour.

## The chat template

`chat: auto` wraps the whole thing in the model's own chat template when the
tokenizer has one, so an instruction-tuned model is addressed the way it was
tuned rather than in raw-completion mode. This matters most for
`prompt_instruct`, the one method whose entire mechanism is instruction
following, and it also stabilises `confidence_gain`, whose DV is an entropy
difference that raw-completion formatting inflates.

It has to happen HERE rather than inside ModelWrapper. `core/positions.py`
resolves named probe positions by mapping character spans of these blocks
through the tokenizer's offset mapping, so if the template were applied later
every span would silently point at the wrong token. `build_with_spans`
therefore returns the final text and spans already shifted past whatever
prefix the template added, and it falls back to the raw body (with a one-time
warning) if the template does not reproduce the body verbatim.

Continuations are scored teacher-forced via `continuation()`, which is the
other half of the same decision: in raw mode an answer continues the question
and needs its leading space, but after a chat template's assistant header the
turn starts fresh and a leading space is the wrong token. Every call site asks
`continuation()` rather than hard-coding `" " + answer`, or the margin is
computed over a different string than the prompt implies.

Everything here is data, not code: change the format in the config
(`prompt:` block) rather than in a method, or the conditions stop being
comparable across methods.
"""
from dataclasses import dataclass, asdict, field

from conflict_bench.core.types import Item, Condition

#: bound by ModelWrapper once the tokenizer exists; `chat: auto` is inert
#: until then, so a PromptSpec built before the model still works.
_TOKENIZER = None
_WARNED = set()


def bind_tokenizer(tok):
    """Called by ModelWrapper. `configure()` usually runs before the model is
    loaded, so the tokenizer is attached afterwards rather than passed in."""
    global _TOKENIZER
    _TOKENIZER = tok
    return tok


def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[prompts] {msg}")


@dataclass
class PromptSpec:
    style: str = "question"          # "question" | "cloze"
    background_prefix: str = "Background: "
    separator: str = "\n\n"
    answer_prefix: str = ""          # e.g. "\nAnswer:" — appended after the stem
    strict: bool = True              # raise if a non-N condition has no passage
    # "before_context" puts a steering instruction ahead of the passage, which
    # is where the literature puts it; "after_context" is the old behaviour.
    instruction_position: str = "before_context"
    # "auto" uses the model's chat template when the tokenizer has one,
    # "off" never does, "on" requires one and raises if it is missing.
    chat: str = "auto"
    system: str = ""                 # optional system turn, chat mode only

    # ---------------------------------------------------------------- body
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

    # ---------------------------------------------------------------- chat
    def _tokenizer(self):
        if self.chat == "off":
            return None
        tok = _TOKENIZER
        if tok is None or not getattr(tok, "chat_template", None):
            if self.chat == "on":
                raise ValueError(
                    "prompt.chat: on, but the tokenizer has no chat_template "
                    "(bind one via prompts.bind_tokenizer, or use chat: off)")
            return None
        return tok

    def uses_chat(self) -> bool:
        return self._tokenizer() is not None

    def _wrap(self, body: str):
        """-> (text, offset of `body` inside text). (body, 0) when not chatting."""
        tok = self._tokenizer()
        if tok is None:
            return body, 0
        msgs = ([{"role": "system", "content": self.system}] if self.system
                else []) + [{"role": "user", "content": body}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
        except Exception as e:                       # template refused the turn
            _warn_once("apply", f"chat template failed ({type(e).__name__}: "
                                f"{e}) - falling back to raw prompts")
            return body, 0
        idx = text.find(body)
        if idx < 0:
            # the template escaped or reflowed the body, so the block spans no
            # longer describe it and every named position would be wrong
            _warn_once("verbatim",
                       "chat template did not reproduce the prompt body "
                       "verbatim - falling back to raw prompts so probe "
                       "positions stay valid")
            return body, 0
        return text, idx

    def continuation(self, answer: str) -> str:
        """The teacher-forced continuation for an answer string.

        Raw mode: the answer continues the question, so it carries its leading
        space (and BPE tokens carry that space, which is why the position code
        matches on token *end*). Chat mode: the assistant turn has just been
        opened by the generation prompt, so a leading space would be a
        spurious token.
        """
        return answer if self.uses_chat() else " " + answer

    # ---------------------------------------------------------------- build
    def build(self, item: Item, condition: Condition, instruction: str = "") -> str:
        return self.build_with_spans(item, condition, instruction)[0]

    def build_with_spans(self, item: Item, condition: Condition,
                         instruction: str = ""):
        """The prompt, plus the character span of each block.

        The spans are what makes a *named* probe position possible: a raw
        integer index is a different linguistic location on every item once
        passages differ in length, so `end_of_context` has to be resolved per
        item rather than configured as a number (see core/positions.py). They
        are returned in coordinates of the FINAL text, chat wrapper included.
        """
        background = None
        if condition != Condition.NORMAL:
            passage = item.passages.get(condition.value, "")
            if not passage and self.strict:
                raise ValueError(
                    f"item {item.item_id}: no passage for condition "
                    f"{condition.value} — the R baseline and the S/C contrast "
                    f"both need one (see data.load_confiqa_cloze_clusters)")
            if passage:
                background = ("background", self.background_prefix + passage)

        instr = ("instruction", instruction.strip()) if instruction else None
        parts = []
        if instr and self.instruction_position != "after_context":
            parts.append(instr)
        if background:
            parts.append(background)
        if instr and self.instruction_position == "after_context":
            parts.append(instr)
        parts.append(("stem", self.stem(item)))

        body, spans = "", {}
        for i, (label, chunk) in enumerate(parts):
            if i:
                body += self.separator
            start = len(body)
            body += chunk
            spans[label] = (start, len(body))
        body += self.answer_prefix

        text, shift = self._wrap(body)
        if shift:
            spans = {k: (a + shift, b + shift) for k, (a, b) in spans.items()}
        return text, spans

    def as_dict(self):
        d = asdict(self)
        d["chat_active"] = self.uses_chat()
        return d


DEFAULT = PromptSpec()


def configure(**kwargs) -> PromptSpec:
    """Called once by the runner from cfg['prompt']; every method then sees it."""
    global DEFAULT
    DEFAULT = PromptSpec(**{k: v for k, v in kwargs.items() if v is not None})
    return DEFAULT


def build_prompt(item: Item, condition: Condition, instruction: str = "",
                 spec: PromptSpec = None) -> str:
    return (spec or DEFAULT).build(item, condition, instruction)


def wrap_chat(text: str, spec: PromptSpec = None) -> str:
    """Apply the chat template to a prompt built outside PromptSpec.

    `p_true` asks a question that is not the item's question, so it composes
    its own text (the one documented exception to invariant 7). It still has
    to reach the model in the same format as everything else, or its number is
    not comparable with the detectors that go through `build_prompt`.
    """
    return (spec or DEFAULT)._wrap(text)[0]


def continuation(answer: str, spec: PromptSpec = None) -> str:
    """The teacher-forced continuation string — ask this, never `" " + a`."""
    return (spec or DEFAULT).continuation(answer)
