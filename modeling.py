import os, math, numbers, collections, logging, numpy, torch

import warnings

from io import StringIO

from typing import Callable, Optional, Literal, Union, Tuple, Iterator, Dict, List, Any

def _adamw(model, weight_decay=1e-2, lr=1e-3, betas=(0.9, 0.999)):
    r""" Creates an AdamW optimizer with separate parameter groups for weight decay and no weight decay. """
    def is_weight_decay(name, param):
        return hasattr(param, 'WEIGHT_DECAY') and param.WEIGHT_DECAY
    _with_weight_decay = []
    _with_out_weight_decay = []
    for name, param in model.named_parameters(remove_duplicate=True):
        if not param.requires_grad:
            continue
        if is_weight_decay(name, param):
            _with_weight_decay.append(param)
        else:
            _with_out_weight_decay.append(param)
    AdamW = torch.optim.AdamW(
        [
            {'params': list(_with_weight_decay), 'weight_decay': weight_decay},
            {'params': list(_with_out_weight_decay), 'weight_decay': 0.0}
        ],
        lr=lr,
        betas=betas
    )
    return AdamW

def _cosine_decay_with_warmup(step: int, lr: float, warmup: float, train: int):
    r"""
    Computes learning rate with linear warmup followed by cosine decay.
    """ 
    if step < 0:
        raise ValueError("Step must be non-negative.")
    if lr <= 0:
        raise ValueError("Invalid learning rate.")
    if train < 0:
        raise ValueError("Total steps must be positive.")
    if warmup < 0:
        raise ValueError("Warmup steps must be non-negative.")
    if warmup < 1:
        warmup = train * warmup # % of the total
    warmup = int(warmup)
    if step < warmup:
        return lr * (step + 1) / (warmup + 1)
    if step >= train:
        return 1e-8
    decay = min(max((step - warmup) / (train - warmup), 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return 1e-8 + coeff * (lr - 1e-8)

def rsqrt(x: numpy.ndarray, eps: float = 1e-6) -> numpy.ndarray:
    """
    Reciprocal square‑root: 1/sqrt(x + eps)
    """
    return 1.0 / numpy.sqrt(x + eps)

def rmsnorm(x: Union[torch.Tensor|numpy.ndarray], eps: float = 1e-6) -> Union[torch.Tensor|numpy.ndarray]:
    if isinstance(x, numpy.ndarray):
        return x * rsqrt(numpy.mean(numpy.square(x), axis=-1, keepdims=True), eps)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

def _norm(norm: Union[bool | Literal['rmsnorm', 'layernorm'] | Callable], dim: int, *, eps: float, bias: bool):
    if norm is None or isinstance(norm, bool):
        return torch.nn.LayerNorm(dim, bias=bias, eps=eps) if norm else None
    elif norm == "RMSNorm" or norm == "RMS" or norm == "rmsnorm" or norm == "rms":
        return RMSNorm(dim, bias=bias, eps=eps)
    elif norm == "LayerNorm" or norm == "Layer" or norm == "layernorm" or norm == "layer":
        return  torch.nn.LayerNorm(dim, bias=bias, eps=eps)
    else:
        raise ValueError(f"Unsupported normalization type: {norm}. Use 'RMSNorm', 'LayerNorm', or None.") 

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, bias: bool = False, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(dim))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        x = rmsnorm(x.float(), self.eps).type_as(x)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, bias={self.bias is not None}"

class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def _init_weights(self, module):
        generator = None
        if hasattr(self, "generator"):
            generator = self.generator
        if hasattr(module, 'wte') and isinstance(module.wte, torch.nn.Parameter):
            torch.nn.init.normal_(module.wte, mean = 0.0, std = 0.02, generator = generator)
        if hasattr(module, 'tte') and isinstance(module.tte, torch.nn.Parameter):
            torch.nn.init.normal_(module.tte, mean = 0.0, std = 0.02, generator = generator)
        if hasattr(module, 'wpe') and isinstance(module.wpe, torch.nn.Parameter):
            torch.nn.init.normal_(module.wpe, mean = 0.0, std = 0.02, generator = generator)
        elif isinstance(module, torch.nn.Linear):
            std = 0.02 / math.sqrt(2 * self.cfg.depth) if hasattr(self, 'cfg') and self.cfg.depth is not None and hasattr(module, 'RESIDUAL_SCALE_FLAG') and module.RESIDUAL_SCALE_FLAG else 0.02
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std, generator = generator)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02, generator = generator)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.RMSNorm):
            raise RuntimeError("Not supported.")
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02, generator = generator)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.ConvTranspose2d):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02, generator = generator)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def extra_repr(self) -> str:
        s = ""
        if hasattr(self, 'wte') and isinstance(self.wte, torch.nn.Parameter):
            s += f"(wte): Parameter{tuple(self.wte.shape)}\n"
        if hasattr(self, 'tte') and isinstance(self.wte, torch.nn.Parameter):
            s += f"(tte): Parameter{tuple(self.tte.shape)}\n"
        if hasattr(self, 'wpe') and isinstance(self.wpe, torch.nn.Parameter):
            s += f"(wpe): Parameter{tuple(self.wpe.shape)}\n"
        return s.strip()

class MLP(torch.nn.Module):
    r"""
    Generic MLP block used by most transformer architectures.
    """
    def __init__(
        self,
        *,
        norm: Union[bool | str | Callable],
        in_features: int,
        hidden_features: int,
        out_features: int,
        eps: float,
        bias: bool,
        drop: float
    ):
        super().__init__()
        norm = _norm(norm, in_features, eps=eps, bias=bias)
        if norm:
            self.norm = norm
        self.hidden = torch.nn.Linear(in_features, hidden_features, bias=bias)
        self.hidden.weight.WEIGHT_DECAY = True
        self.act = torch.nn.GELU()
        self.proj = torch.nn.Linear(hidden_features, out_features, bias=bias)
        self.proj.RESIDUAL_SCALE_FLAG = True
        self.proj.weight.WEIGHT_DECAY = True
        self.drop = torch.nn.Dropout(drop) if drop is not None and drop > 0 else torch.nn.Identity()

    def forward(self, x):
        if hasattr(self, "norm"):
            x = self.norm(x)
        x = self.hidden(x)
        if self.act is not None:
            x = self.act(x)
        x = self.proj(x)
        if self.drop is not None:
            x = self.drop(x)
        return x

# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

def _scaled_dot_product_attention(query, key, value, attention_mask=None, dropout=0.0, is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attention_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attention_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attention_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attention_bias.masked_fill_(attention_mask.logical_not(), float("-inf"))
        else:
            attention_bias = attention_mask + attention_bias
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
    attention_weight = query @ key.transpose(-2, -1) * scale_factor
    attention_weight += attention_bias
    attention_weight = torch.softmax(attention_weight, dim=-1)
    attention_weight = torch.dropout(attention_weight, p=dropout, train=True)
    return attention_weight @ value, attention_weight

class MultiHeadSelfAttention(torch.nn.Module):
    r"""
    Multi-purpose multi-head self-attention.
    """
    def __init__(
        self,
        *,
        in_features: int,
        norm: Union[bool | str | Callable],
        eps: float,
        heads: int,
        DoF: int,
        out_features: int,
        bias: bool = True,
        drop: float = .0,
        # post: Union[bool | str | Callable],
        qkv: Union[bool | Literal['qkv']] = "qkv"
    ):
        super().__init__()
        assert in_features > 0 and in_features <= 768
        assert out_features > 0 and out_features <= 768
        assert heads > 0 and heads <= 16
        assert DoF > 0 and DoF <= 256
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.DoF = DoF # dimension of each head
        self.drop = drop
        # norm the input (Optional)
        norm = _norm(norm, in_features, eps=eps, bias=bias)
        if norm:
            self.norm = norm
        # q/k/v
        if qkv is not None and ((isinstance(qkv, bool) and qkv == True) or qkv == "qkv"):
            self.qkv = torch.nn.Linear(in_features, 3 * heads * DoF, bias=bias)
            self.qkv.weight.WEIGHT_DECAY = True
        else:
            self.wq = torch.nn.Linear(in_features, heads * DoF, bias=bias)
            self.wq.weight.WEIGHT_DECAY = True
            self.wk = torch.nn.Linear(in_features, heads * DoF, bias=bias)
            self.wk.weight.WEIGHT_DECAY = True
            self.wv = torch.nn.Linear(in_features, heads * DoF, bias=bias)
            self.wv.weight.WEIGHT_DECAY = True
        # output
        self.wo = torch.nn.Linear(heads * DoF, out_features, bias=bias)
        self.wo.RESIDUAL_SCALE_FLAG = True
        self.wo.weight.WEIGHT_DECAY = True
        # # norm the output (Optional)
        # post = _norm(post, out_features, eps=eps, bias=bias)
        # if post:
        #     self.post = post

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # norm the input (Optional)
        if hasattr(self, "norm"):
            x = self.norm(x)
        # q/k/v projections
        B, seq_len, in_features = x.shape
        assert self.in_features == self.in_features
        if hasattr(self, "qkv"):
            qkv = self.qkv(x)
            q, k, v = qkv.split(self.heads * self.DoF, dim=2)
        else:
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)
        q = q.view(B, seq_len, self.heads, self.DoF).transpose(1, 2)
        k = k.view(B, seq_len, self.heads, self.DoF).transpose(1, 2)
        v = v.view(B, seq_len, self.heads, self.DoF).transpose(1, 2)
        drop = self.drop if self.training else 0
        __DEBUG__ = False
        if __DEBUG__:
            drop = 0 # turn off dropout when debugging scaled_dot_product_attention
        if not self.training or __DEBUG__:
            y, attention = _scaled_dot_product_attention(q, k, v, attention_mask, drop, is_causal=False)
        else:
            y, attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, attention_mask, drop, is_causal=False), None
        """
        if __DEBUG__:
            y0 = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask, drop, is_causal=False)
            assert torch.allclose(y, y0, rtol=0, atol=1e-04)
        """
        # final projection
        y = y.transpose(1, 2).contiguous().view(B, seq_len, self.heads * self.DoF)
        y = self.wo(y)
        y = torch.dropout(y, p=drop, train=self.training)
        # # norm the output (Optional)
        # if hasattr(self, "post"):
        #     y = self.post(y)
        return y, attention

class MultiHeadSelfAttentionBlock(torch.nn.Module):
    r"""
    Multi-head self-attention block mostly used by GPTs but common enough
        to that it can be reused in other architectures...
    """
    def __init__(
        self,
        *,
        in_features: int,
        norm: Union[bool | str | Callable],
        heads: int,
        DoF: int,
        out_features: int,
        multiplier: int,
        bias: bool,
        eps: float,
        drop: float,
        qkv: Union[bool | Literal['qkv']] = "qkv"
    ):
        super().__init__()
        self.att = MultiHeadSelfAttention(
            norm=norm,
            eps=eps,
            in_features=in_features,
            heads=heads,
            DoF=DoF,
            out_features=out_features,
            bias=bias,
            drop=drop,
            qkv=qkv
        )
        self.mlp = MLP(
            norm=norm,
            eps=eps,
            in_features=out_features,
            hidden_features=int(multiplier * out_features),
            out_features=out_features,
            bias=bias,
            drop=drop)

    def forward(self, x, mask: torch.Tensor = None):
        y, a = self.att(x, mask)
        x = x + y
        y = self.mlp(x)
        x = x + y
        return x, a

# https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama4/tokenizer.py#L251

def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int) -> Iterator[str]:
    """
    Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
    consecutive whitespaces or consecutive non-whitespaces.
    """
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    slice_start = 0
    for i in range(len(s)):
        is_now_space = s[i].isspace()
        if current_slice_is_space ^ is_now_space:
            current_slice_len = 1
            current_slice_is_space = is_now_space
        else:
            current_slice_len += 1
            if current_slice_len > max_consecutive_slice_len:
                yield s[slice_start:i]
                slice_start = i
                current_slice_len = 1
    yield s[slice_start:]

from typing import Iterator

def _strip_whitespaces(s: str) -> Iterator[str]:
    """
    Yield non-empty whitespace-separated tokens from the input string.
    """
    if not s:
        return
    length = len(s)
    i = 0
    while i < length:
        # skip any leading whitespace
        while i < length and s[i].isspace():
            i += 1
        if i >= length:
            break
        start = i
        # advance until next whitespace or end
        while i < length and not s[i].isspace():
            i += 1
        # yield the token we found
        yield s[start:i]

# https://github.com/huggingface/transformers/blob/e8a6eb3304033fdd9346fe3b3293309fe50de238/src/transformers/models/bert/tokenization_bert.py#L31

def _load_vocab(
    vocab_file
) -> collections.OrderedDict:
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
    return vocab

# https://github.com/huggingface/transformers/blob/e8a6eb3304033fdd9346fe3b3293309fe50de238/src/transformers/models/bert/tokenization_bert.py#L239

def _save_vocab(vocab: collections.OrderedDict, vocab_file) -> tuple[str]:
    index = 0
    with open(vocab_file, "w", encoding="utf-8") as writer:
        for token, token_index in sorted(vocab.items(), key=lambda kv: kv[1]):
            if index != token_index:
                print(
                    f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                    " Please check that the vocabulary is not corrupted!"
                )
                index = token_index
            writer.write(token + "\n")
            index += 1

# We save the model parameters in a raw binary file using MATLAB MAT4 format in Raw-Major.

def _save_to_mat4(self: torch.nn.Module, ckpt):
    r""" Save model parameters to a raw binary file. (MATLAB MAT4 format in Raw-Major) """
    path = os.path.dirname(os.path.abspath(ckpt))
    os.makedirs(path, exist_ok=True)
    params = self.state_dict();
    with open(ckpt, "wb") as file:
        for k in params:
            t = params[k].detach().float().cpu()
            name = k.encode('utf-8')
            header = torch.zeros(5, dtype=torch.int32)
            header[0] = 10 # type
            if len(t.shape) == 2:
                header[1] = t.size(1) # mrows
                header[2] = t.size(0) # cols
            else:
                header[1] = 1 # mrows
                header[2] = t.numel() # cols
            header[3] = 0 # imagf
            header[4] = len(name) # dwNameLength
            assert header[4] <= 63
            file.write(header.cpu().numpy().tobytes()) # header
            file.write(name)
            file.write(t.numpy().tobytes())

def _load_from_mat4(self: torch.nn.Module, ckpt, errors: Literal["strict"] = "strict"):
    r""" Load model parameters from a raw binary file. (MATLAB MAT4 format in Raw-Major) """
    params = self.state_dict();
    with open(ckpt, "rb") as f:
        for k in params:
            header = numpy.frombuffer(f.read(5 * 4), dtype=numpy.int32)
            assert len(header) == 5
            assert header[0] == 10
            assert header[1] > 0 and header[1] <= 0xFFFF
            assert header[2] > 0 and header[2] <= 0xFFFF
            assert header[3] == 0
            assert header[4] > 0 and header[4] <= 63
            name = f.read(header[4])
            assert len(name) == header[4]
            name = name.decode('utf-8').strip("\n\r \t\0")
            if name != k:
                s = f"Parameter name mismatch '{k}' expected, but got '{name}'"
                warnings.warn(s, RuntimeWarning)
            data = numpy.frombuffer(f.read(header[1] * header[2] * 4), dtype=numpy.float32)
            assert len(data) == header[1] * header[2]
            with torch.no_grad():
                params[k].copy_(torch.tensor(data).view(params[k].shape))