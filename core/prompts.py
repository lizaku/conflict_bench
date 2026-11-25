"""Prompt construction - the single source of truth for every method.

    no context    ->  "{stem}"
    one passage   ->  "Background: {s1} {s2} {s3} {s4}\n\n{stem}"
    two passages  ->  "Passage A: {s1} {s2} {s3}\nPassage B: {s4} {s5} {s6}\n\n{stem}"

`stem` is what the answer continues: the question (format `short`) or the
cloze sentence cut before its blank (format `cloze`).  It is None for the
global detection task, whose input is (c, M) with no question.

A steering instruction (prompt_instruct) is its own block.  It goes before the
context by default, because an instruction wedged between a passage and the
question reads as another sentence of the document.

## The chat template

`chat: auto` wraps the whole thing in the model's own chat template when the
tokenizer has one, so an instruction-tuned model is addressed the way it was
tuned.  It has to happen HERE rather than inside ModelWrapper:
`core/positions.py` resolves named positions by mapping the character spans
returned by `render` through the tokenizer's offset mapping, so a template
applied later would shift every span.  `render` returns spans in coordinates
of the final text, and falls back to the raw body (with a one-time warning) if
the template does not reproduce the body verbatim.

Continuations are scored teacher-forced via `continuation()`: in raw mode the
answer continues the question and carries its leading space; after a chat
template's assistant header the turn starts fresh and that space would be a
spurious token.  Never hard-code `" " + answer`.
"""
from dataclasses import dataclass, asdict

#: bound by ModelWrapper once the tokenizer exists
_TOKENIZER = None
_WARNED = set()


def bind_tokenizer(tok):
    global _TOKENIZER
    _TOKENIZER = tok
    return tok


def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[prompts] {msg}")


@dataclass
class PromptSpec:
    background_prefix: str = "Background: "
    passage_prefix: str = "Passage {id}: "   # used when there are 2+ passages
    separator: str = "\n\n"
    answer_prefix: str = ""
    instruction_position: str = "before_context"   # or "after_context"
    chat: str = "auto"                             # "auto" | "off" | "on"
    system: str = ""

    # ---------------------------------------------------------------- chat
    def _tokenizer(self):
        if self.chat == "off":
            return None
        tok = _TOKENIZER
        if tok is None or not getattr(tok, "chat_template", None):
            if self.chat == "on":
                raise ValueError("prompt.chat: on, but the tokenizer has no "
                                 "chat_template")
            return None
        return tok

    def uses_chat(self) -> bool:
        return self._tokenizer() is not None

    def _wrap(self, body: str):
        """-> (text, offset of `body` inside text)."""
        tok = self._tokenizer()
        if tok is None:
            return body, 0
        msgs = ([{"role": "system", "content": self.system}] if self.system
                else []) + [{"role": "user", "content": body}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
        except Exception as e:
            _warn_once("apply", f"chat template failed ({type(e).__name__}: "
                                f"{e}) - falling back to raw prompts")
            return body, 0
        idx = text.find(body)
        if idx < 0:
            _warn_once("verbatim", "chat template did not reproduce the prompt "
                                   "body verbatim - falling back to raw prompts "
                                   "so positions stay valid")
            return body, 0
        return text, idx

    def continuation(self, answer: str) -> str:
        return answer if self.uses_chat() else " " + answer

    # ---------------------------------------------------------------- build
    def render(self, stem, passages=(), instruction=""):
        """-> (text, spans).

        passages: sequence of (passage_id, [sentence, ...]).
        spans: {"context": (a, b), "stem": (a, b), "instruction": (a, b),
                "chunks": [(a, b), ...]}  - character spans in `text`; a key is
        absent when its block is.  `chunks` runs over all sentences of all
        passages in order.
        """
        passages = [(pid, list(ch)) for pid, ch in passages if ch]
        blocks = []                      # (label, text, [chunk spans in text])
        if passages:
            ctx, chunk_spans = "", []
            for n, (pid, chunks) in enumerate(passages):
                if n:
                    ctx += "\n"
                ctx += (self.background_prefix if len(passages) == 1
                        else self.passage_prefix.format(id=pid))
                for j, ch in enumerate(chunks):
                    if j:
                        ctx += " "
                    chunk_spans.append((len(ctx), len(ctx) + len(ch)))
                    ctx += ch
            blocks.append(("context", ctx, chunk_spans))
        instr = ("instruction", instruction.strip(), []) if instruction else None
        if instr:
            blocks.insert(0 if self.instruction_position != "after_context"
                          else len(blocks), instr)
        if stem:
            blocks.append(("stem", stem, []))

        body, spans = "", {}
        for i, (label, text, chunk_spans) in enumerate(blocks):
            if i:
                body += self.separator
            start = len(body)
            body += text
            spans[label] = (start, len(body))
            if label == "context":
                spans["chunks"] = [(start + a, start + b) for a, b in chunk_spans]
        body += self.answer_prefix

        text, shift = self._wrap(body)
        if shift:
            spans = {k: ([(a + shift, b + shift) for a, b in v] if k == "chunks"
                         else (v[0] + shift, v[1] + shift))
                     for k, v in spans.items()}
        return text, spans

    def as_dict(self):
        d = asdict(self)
        d["chat_active"] = self.uses_chat()
        return d


DEFAULT = PromptSpec()


def configure(**kwargs) -> PromptSpec:
    """Called once from cfg['prompt']; every method then sees it."""
    global DEFAULT
    DEFAULT = PromptSpec(**{k: v for k, v in kwargs.items() if v is not None})
    return DEFAULT


def render(stem, passages=(), instruction=""):
    return DEFAULT.render(stem, passages, instruction)


def wrap_chat(text: str) -> str:
    """Chat-wrap a prompt built outside `render` (only `p_true` does this: it
    asks a question that is not the instance's question)."""
    return DEFAULT._wrap(text)[0]


def continuation(answer: str) -> str:
    """The teacher-forced continuation string - ask this, never `" " + a`."""
    return DEFAULT.continuation(answer)
