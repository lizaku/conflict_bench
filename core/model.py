"""Single shared model wrapper.

Everything every method needs from the model lives here so that mechanistic
and output-based methods run against the *same* forward passes and prompts:
  - teacher-forced log-prob of a candidate continuation (v3 semantics)
  - residual-stream capture at chosen layers/positions
  - additive residual-stream intervention (CAA / DiffMean steering)
  - free generation and multi-sample generation (for SE / SelfCheckGPT)
  - logits with and without context (for CAD / CK-PLUG-style decoding)

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
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct",
                 device="auto", dtype="auto", cache_dir=None,
                 activation_cache_dir=None, margin_table_path=None,
                 reference_condition="R", format_correct=False,
                 trust_remote_code=False,
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
        from conflict_bench.core.margins import MarginTable
        self.acts = ActivationCache(self, activation_cache_dir)
        # the N/S/C/R teacher-forced margins, scored once and shared by every
        # method that wants them (detectors, steerers, the R correction)
        self.margins = MarginTable(self, margin_table_path,
                                   reference=reference_condition,
                                   correct=format_correct)

    @classmethod
    def from_config(cls, cfg, activation_cache_dir=None,
                    margin_table_path=None):
        """`model:` may be a plain hf id or a block of kwargs."""
        spec = cfg.get("model", "meta-llama/Llama-3.1-8B-Instruct")
        if isinstance(spec, str):
            spec = {"model_name": spec}
        else:
            spec = dict(spec)
            name = spec.pop("name", None) or spec.pop("model_name", None)
            spec["model_name"] = name
        spec.setdefault("activation_cache_dir", activation_cache_dir)
        spec.setdefault("margin_table_path", margin_table_path)
        spec.setdefault("reference_condition", cfg.get("reference_condition", "R"))
        spec.setdefault("format_correct", cfg.get("format_correct", False))
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

    # ---------- teacher-forced scoring (matches v3) ----------
    @torch.no_grad()
    def logp_continuation(self, prompt: str, continuation: str,
                          length_normalize: bool = False) -> float:
        """Sum (or mean) log-prob of `continuation` tokens given `prompt`."""
        p_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
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
        p_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
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
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
            self.model(ids)
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
    def generate(self, prompt: str, max_new_tokens=32, temperature=0.0,
                 num_samples=1):
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        ids = enc.input_ids
        do_sample = temperature > 0
        out = self.model.generate(
            ids, attention_mask=enc.get("attention_mask"),
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else None,
            num_return_sequences=num_samples,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return [self.tok.decode(o[ids.shape[1]:], skip_special_tokens=True)
                for o in out]

    # ---------- contrastive logits (CAD / CK-PLUG family) ----------
    @torch.no_grad()
    def next_token_logits(self, prompt: str):
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        return self.model(ids).logits[0, -1].float()
