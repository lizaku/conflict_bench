"""Single shared model wrapper - the `M` every method receives.

  - teacher-forced log-prob of a continuation
  - residual-stream capture at chosen layers/positions
  - additive residual-stream intervention (CAA / DiffMean steering)
  - greedy decoding with an optional per-step logits function, and sampling
  - a counter of forward passes, which is how the harness reports cost

The model is a config knob, not a constant: `from_config` accepts either
`model: <hf-id>` or a `model:` block with name/device/dtype/revision/etc.
Layer discovery is architecture-agnostic (llama / mistral / gemma / qwen /
gpt-neox / gpt2 all resolve), so swapping the 8B Llama for anything else is
a one-line config change.
"""
from contextlib import contextmanager

import torch

_DTYPES = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
           "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
           "float32": torch.float32, "fp32": torch.float32}


def _resolve_device(device):
    if device not in (None, "auto"):
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype, device):
    if dtype in (None, "auto"):
        # fp16/bf16 on CPU is slow and numerically awkward for log-probs
        return torch.float32 if device == "cpu" else torch.bfloat16
    if isinstance(dtype, torch.dtype):
        return dtype
    return _DTYPES[str(dtype).lower()]


def _find_layers(model):
    """The list of transformer blocks, whatever the architecture calls it."""
    for attr in ("model.layers", "model.decoder.layers", "transformer.h",
                 "gpt_neox.layers", "transformer.blocks", "model.transformer.h"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if hasattr(obj, "__len__") and len(obj):
            return obj
    raise AttributeError(
        "could not locate transformer blocks on " + type(model).__name__ +
        "; add its path to core.model._find_layers")


class ModelWrapper:
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct",
                 device="auto", dtype="auto", cache_dir=None,
                 activation_cache_dir=None, trust_remote_code=False,
                 revision=None, load_kwargs=None, tokenizer_name=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype, self.device)
        kw = dict(torch_dtype=self.dtype, trust_remote_code=trust_remote_code)
        if revision:
            kw["revision"] = revision
        if cache_dir:
            kw["cache_dir"] = cache_dir
        kw.update(load_kwargs or {})
        if self.device.startswith("cuda"):
            kw.setdefault("device_map", self.device)
        tok_kw = {"cache_dir": cache_dir} if cache_dir else {}
        self.tok = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name, trust_remote_code=trust_remote_code,
            **tok_kw)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        # core/prompts applies the chat template itself (it owns the block
        # spans that core/positions maps through the offset mapping, so the
        # wrapping cannot happen later without invalidating every named
        # probe position). It needs the tokenizer to do that.
        from conflict_bench.core import prompts as _prompts
        _prompts.bind_tokenizer(self.tok)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        if "device_map" not in kw:
            self.model.to(self.device)
        self.model.eval()
        self._layers = _find_layers(self.model)
        self.n_layers = len(self._layers)
        self.hidden_size = self.model.config.hidden_size
        from conflict_bench.core.activations import ActivationCache
        self.acts = ActivationCache(self, activation_cache_dir)
        # Every forward pass of the network is counted, whoever asks for it,
        # so each method's cost is measured by the harness instead of declared.
        self.forward_passes = 0
        self.model.register_forward_hook(self._count_forward)
        self._stop_ids = self._find_stop_ids()

    def _count_forward(self, *_):
        self.forward_passes += 1

    def _find_stop_ids(self):
        ids = {self.tok.eos_token_id}
        vocab = self.tok.get_vocab()
        for t in ("<|end|>", "<|eot_id|>", "<end_of_turn>", "<|im_end|>",
                  "<|endoftext|>", "</s>"):
            if t in vocab:
                ids.add(vocab[t])
        return {i for i in ids if i is not None}

    @classmethod
    def from_config(cls, cfg, activation_cache_dir=None):
        """`model:` may be a plain hf id or a block of kwargs."""
        spec = cfg.get("model", "microsoft/Phi-3.5-mini-instruct")
        if isinstance(spec, str):
            spec = {"model_name": spec}
        else:
            spec = dict(spec)
            name = spec.pop("name", None) or spec.pop("model_name", None)
            spec["model_name"] = name
        spec.setdefault("activation_cache_dir", activation_cache_dir)
        return cls(**spec)

    def describe(self):
        return {"model": self.model_name, "device": self.device,
                "dtype": str(self.dtype), "n_layers": self.n_layers,
                "hidden_size": self.hidden_size}

    def resolve_layer(self, layer):
        """Negative / fractional layer indices, so configs survive a model swap.

        0 < layer < 1 is read as a depth fraction: 0.6 -> 60% of the way up.
        """
        if isinstance(layer, float) and 0 < layer < 1:
            return int(round(layer * (self.n_layers - 1)))
        return int(layer) % self.n_layers

    # ---------- tokenisation: the one place a prompt becomes ids ----------
    def _add_specials(self, prompt: str) -> bool:
        """A chat template may already have written BOS into the text; adding
        it again gives the model a prompt it was never trained on."""
        bos = self.tok.bos_token
        return not (bos and prompt.startswith(bos))

    def encode(self, prompt: str):
        return self.tok(prompt, return_tensors="pt",
                        add_special_tokens=self._add_specials(prompt)
                        ).input_ids.to(self.device)

    def offsets(self, prompt: str):
        """Character span of every token of `encode(prompt)`."""
        return self.tok(prompt, return_offsets_mapping=True,
                        add_special_tokens=self._add_specials(prompt)
                        )["offset_mapping"]

    def continuation_ids(self, continuation: str):
        return self.tok(continuation, add_special_tokens=False).input_ids

    # ---------- teacher-forced scoring ----------
    @torch.no_grad()
    def logp_continuation(self, prompt: str, continuation: str,
                          length_normalize: bool = False) -> float:
        """Sum (or mean) log-prob of `continuation` tokens given `prompt`."""
        p_ids = self.encode(prompt)
        c_ids = self.tok(continuation, add_special_tokens=False,
                         return_tensors="pt").input_ids.to(self.device)
        ids = torch.cat([p_ids, c_ids], dim=1)
        logits = self.model(ids).logits[:, :-1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        tgt = ids[:, 1:]
        lp = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        cont_lp = lp[:, p_ids.shape[1] - 1:]  # positions predicting continuation
        total = cont_lp.sum().item()
        return total / c_ids.shape[1] if length_normalize else total

    def first_token_id(self, text: str) -> int:
        return self.tok(text, add_special_tokens=False).input_ids[0]

    @torch.no_grad()
    def teacher_forced_logits(self, prompt: str, continuation: str):
        """-> (logits[T, V], continuation_ids[T]) for the continuation tokens.

        The per-position logits that *predict* each continuation token, which
        is what a contrastive decoder (CAD / AdaCAD / CK-PLUG) needs in order
        to be scored on the same multi-token margin as every other method
        rather than on a first-token proxy.

        Summing log_softmax of these at the continuation ids reproduces
        `logp_continuation` exactly, so a decoding method at strength 0 lands
        on the shared MarginTable value by construction.
        """
        p_ids = self.encode(prompt)
        c_ids = self.tok(continuation, add_special_tokens=False,
                         return_tensors="pt").input_ids.to(self.device)
        ids = torch.cat([p_ids, c_ids], dim=1)
        logits = self.model(ids).logits[0].float()
        start = p_ids.shape[1] - 1
        return logits[start:start + c_ids.shape[1]], c_ids[0]

    # ---------- activation capture ----------
    @torch.no_grad()
    def residual_at(self, prompt: str, layers, position: int = -1):
        """Residual stream (layer output) at `position` for each layer.

        position semantics: -1 is the last prompt token (end of stem, the
        position the answer is predicted from).  Pass an explicit index to
        probe end-of-context or earlier positions.
        """
        acts = {}
        hooks = []

        def make_hook(l):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                acts[l] = h[0, position].detach().float().cpu()
            return hook

        for l in layers:
            hooks.append(self._layers[self.resolve_layer(l)]
                         .register_forward_hook(make_hook(l)))
        try:
            self.model(self.encode(prompt))
        finally:
            for h in hooks:
                h.remove()
        return acts

    @torch.no_grad()
    def residual_norm(self, prompt: str, layers, position: int = -1):
        return {l: float(v.norm())
                for l, v in self.residual_at(prompt, layers, position).items()}

    # ---------- logit lens (the readout control) ----------
    def _final_norm(self):
        for attr in ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm",
                     "model.final_layernorm", "transformer.norm_f"):
            obj = self.model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        return None

    def _unembed_head(self):
        for attr in ("lm_head", "embed_out"):
            head = getattr(self.model, attr, None)
            if head is not None:
                return head
        raise AttributeError("no unembedding head found on this model")

    @torch.no_grad()
    def unembed(self, h, apply_norm: bool = True):
        """Project a residual-stream vector to vocabulary logits.

        This is what makes "is the probe detecting, or just decoding?" a
        measurable question rather than a rhetorical one: whatever a probe can
        read at (layer, position), the model's own unembedding can also read
        there, and the probe has to beat that to be doing anything else.
        """
        if not isinstance(h, torch.Tensor):
            h = torch.as_tensor(h)
        dtype = next(self.model.parameters()).dtype
        h = h.to(self.device, dtype=dtype).reshape(1, -1)
        norm = self._final_norm()
        if apply_norm and norm is not None:
            h = norm(h)
        return self._unembed_head()(h)[0].float()

    # ---------- additive intervention ----------
    @contextmanager
    def add_direction(self, vec, layers, alpha: float = 1.0,
                      positions: slice = slice(None)):
        """Context manager: h <- h + alpha * vec at given layers/positions.

        `vec` is either one tensor applied at every layer, or a
        {layer: tensor} mapping (CAA vectors are layer-specific).
        Use with a norm-matched random `vec` for the control condition.
        Orthogonalization against v_use happens BEFORE calling this
        (see methods/steerers/steerers.py::ActivationAddition._effective_vec).
        """
        dtype = next(self.model.parameters()).dtype

        def prep(v):
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            return v.to(self.device, dtype=dtype)

        vecs = ({l: prep(v) for l, v in vec.items()} if isinstance(vec, dict)
                else {l: prep(vec) for l in layers})
        hooks = []

        def make_hook(v):
            def hook(_m, _i, out):
                if isinstance(out, tuple):
                    h = out[0]
                    h[:, positions] = h[:, positions] + alpha * v
                    return (h,) + out[1:]
                out[:, positions] = out[:, positions] + alpha * v
                return out
            return hook

        for l in layers:
            hooks.append(self._layers[self.resolve_layer(l)]
                         .register_forward_hook(make_hook(vecs[l])))
        try:
            yield
        finally:
            for h in hooks:
                h.remove()

    # ---------- generation ----------
    @torch.no_grad()
    def next_logits(self, prompt: str, generated=()):
        """Next-token logits after `prompt` + the token ids in `generated`."""
        ids = self.encode(prompt)
        if len(generated):
            ids = torch.cat([ids, torch.tensor([list(generated)],
                                               device=self.device)], dim=1)
        return self.model(ids).logits[0, -1].float()

    @torch.no_grad()
    def greedy(self, prompt: str, max_new_tokens=12, logits_fn=None):
        """Greedy decoding, one full forward pass per token.

        Written out instead of calling `model.generate` for two reasons: the
        unsteered answer `d` and every steered answer `x` must come from the
        same decoding loop (HF `generate` silently applies whatever the
        model's generation_config ships - repetition penalties and the like),
        and a decoding-time method plugs in as `logits_fn(z, generated) -> z`.
        Activation hooks are simply left installed around the call.
        """
        out = []
        for _ in range(max_new_tokens):
            z = self.next_logits(prompt, out)
            if logits_fn is not None:
                z = logits_fn(z, out)
            t = int(z.argmax())
            if t in self._stop_ids:
                break
            out.append(t)
        return self.tok.decode(out, skip_special_tokens=True).strip()

    @torch.no_grad()
    def logp_steered(self, prompt: str, continuation: str, logits_fn=None):
        """Teacher-forced log-prob of `continuation` under the same loop as
        `greedy` (so hooks and `logits_fn` apply to it identically)."""
        ids = self.continuation_ids(continuation)
        total = 0.0
        for t in range(len(ids)):
            z = self.next_logits(prompt, ids[:t])
            if logits_fn is not None:
                z = logits_fn(z, ids[:t])
            total += float(torch.log_softmax(z, -1)[ids[t]])
        return total

    @torch.no_grad()
    def sample(self, prompt: str, num_samples=10, max_new_tokens=12,
               temperature=1.0):
        """`num_samples` sampled answers (for answer entropy)."""
        ids = self.encode(prompt)
        out = self.model.generate(
            ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=1.0, top_k=0,
            repetition_penalty=1.0, num_return_sequences=num_samples,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return [self.tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip()
                for o in out]
