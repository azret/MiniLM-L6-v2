import math, os, sys, time

import numpy, torch

from modeling import _Model, MultiHeadSelfAttention, MLP
from modeling import _save_to_mat4, _load_from_mat4
from modeling import _strip_whitespaces, _save_vocab, _load_vocab

from typing import Callable, Literal, Optional, Tuple, Union, List, Dict

from dataclasses import dataclass

WELL_KNOWN_MODELS = {
    "all-MiniLM-L6-v2": {
        "base": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "multiplier": 4,
        "heads": 12,
        "depth": 6,
        "seq_len": 512,
        "vocab_size": 30522,
        "type_vocab_size": 2,
        "bias": True,
        "eps": 1e-12
        },
    "all-MiniLM-L12-v2": {
        "base": "sentence-transformers/all-MiniLM-L12-v2",
        "dim": 384,
        "multiplier": 4,
        "heads": 12,
        "depth": 12,
        "seq_len": 512,
        "vocab_size": 30522,
        "type_vocab_size": 2,
        "bias": True,
        "eps": 1e-12
        },
}

@dataclass
class BertConfig:
    base: str = ""
    dim: int = 384
    seq_len: int = 512
    vocab_size: int = 30522
    type_vocab_size: int = 2
    multiplier: int = 4
    heads: int = 12
    depth: int = 6
    bias: bool = True
    eps: float = 1e-12
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

class BertEncoding:
    def __init__(self, model: str):
        r""" Tokenizer for Bert-based models. """
        from transformers import BertTokenizer as _BertTokenizer
        self.tokenizer = _BertTokenizer.from_pretrained(WELL_KNOWN_MODELS[model]["base"])
    def __call__(self, text):
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return encoded

class Bert(_Model):
    class BertAttention(torch.nn.Module):
        def __init__(
            self,
            *,
            dim: int,
            heads: int,
            bias: bool,
            eps: float,
            drop: float
        ):
            super().__init__()
            self.attention = MultiHeadSelfAttention(
                norm=None,
                eps=eps,
                in_features=dim,
                heads=heads,
                DoF=dim // heads,
                out_features=dim,
                bias=bias,
                qkv=None,
                drop=drop
            )
            self.norm = torch.nn.LayerNorm(
                dim,
                eps=eps,
                bias=bias)
        def forward(self, x, attention_mask: torch.Tensor = None):
            a, w = self.attention(x, attention_mask)
            y = self.norm(a + x)
            return y, w

    class BertLayer(torch.nn.Module):
        def __init__(
            self,
            *,
            dim: int,
            heads: int,
            multiplier: int,
            bias: bool,
            eps: float,
            drop: float
        ):
            super().__init__()
            self.attention = Bert.BertAttention(
                eps=eps,
                dim=dim,
                heads=heads,
                bias=bias,
                drop=drop
            )
            self.mlp = MLP(
                norm=None,
                eps=eps,
                in_features=dim,
                hidden_features=int(multiplier * dim),
                out_features=dim,
                bias=bias,
                drop=drop)
            self.norm = torch.nn.LayerNorm(
                dim,
                eps=eps,
                bias=bias)
        def forward(self, x, attention_mask: torch.Tensor = None):
            a, w = self.attention(x, attention_mask)
            z = self.mlp(a)
            y = self.norm(z + a)
            return y, w

    def __init__(self, cfg: BertConfig, dropout: float = 0.1):
        super().__init__()
        assert cfg.dim >= 1 and cfg.dim <= 1024
        assert cfg.dim % cfg.heads == 0 and cfg.dim // cfg.heads > 0
        assert cfg.multiplier in [1, 2, 3, 4]
        assert cfg.depth >= 1 and cfg.depth <= 12
        assert dropout >= 0 and dropout <= 1
        self.cfg = cfg
        self.wte = torch.nn.Parameter(torch.empty((cfg.vocab_size, cfg.dim)))
        self.wte.WEIGHT_DECAY = 1
        self.wpe = torch.nn.Parameter(torch.empty((cfg.seq_len, cfg.dim)))
        self.wpe.WEIGHT_DECAY = 1
        self.tte = torch.nn.Parameter(torch.empty((cfg.type_vocab_size, cfg.dim)))
        self.tte.WEIGHT_DECAY = 1
        torch.nn.init.normal_(self.wte, std=0.02)
        torch.nn.init.normal_(self.wpe, std=0.02)
        torch.nn.init.normal_(self.tte, std=0.02)
        self.norm = torch.nn.LayerNorm(cfg.dim, bias=cfg.bias, eps=cfg.eps)
        self.dropout = torch.nn.Dropout(dropout)
        self.encoder = torch.nn.ModuleDict(dict(
            layer = torch.nn.ModuleList(
                [
                    Bert.BertLayer(
                        dim = cfg.dim,
                        eps = cfg.eps,
                        heads = cfg.heads,
                        multiplier = cfg.multiplier,
                        bias = cfg.bias,
                        drop=dropout
                    )
                    for _ in range(cfg.depth)
                ]
            )
        ))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None
    ):
        B, T = input_ids.size()
        assert T <= self.cfg.seq_len
        if token_type_ids is None:
            token_type_ids = torch.zeros((B, T), dtype=torch.long, device=input_ids.device)
        assert token_type_ids.size() == (B, T)
        x = self.wte[input_ids]
        x = x + self.tte[token_type_ids]
        x = x + self.wpe[:T]
        x = self.norm(x)
        x = self.dropout(x)
        if attention_mask is None:
            attention_mask = torch.ones((B, T))
        attention_mask = attention_mask.to(device=x.device)
        assert attention_mask.size() == (B, T)
        attention_mask = attention_mask[:, None, None, :].expand(B, 1, T, T).to(x.dtype)
        inverted_mask = torch.tensor(1.0, dtype=x.dtype, device=x.device) - attention_mask
        attention_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(x.dtype).min)
        for block in self.encoder.layer:
            x, _ = block(x, attention_mask)
        return x

def _adopt_from_hugging_face(dst, src):
    r""" Adopts parameters from HuggingFace. """
    dst_parameters = dst.state_dict()
    src_parameters = src.state_dict()
    for src_k in list(src_parameters.keys()):
        dst_k = None
        if src_k == 'embeddings.word_embeddings.weight':
            dst_k = 'wte'
        elif src_k == 'embeddings.position_embeddings.weight':
            dst_k = 'wpe'
        elif src_k == 'embeddings.token_type_embeddings.weight':
            dst_k = 'tte'
        elif src_k == 'embeddings.LayerNorm.weight':
            dst_k = 'norm.weight'
        elif src_k == 'embeddings.LayerNorm.bias':
            dst_k = 'norm.bias'
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".attention.self.query.weight"):
            dst_k = src_k.replace(".attention.self.query.weight", ".attention.attention.wq.weight")
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".attention.self.query.bias"):
            dst_k = src_k.replace(".attention.self.query.bias", ".attention.attention.wq.bias")
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".attention.self.key.weight"):
            dst_k = src_k.replace(".attention.self.key.weight", ".attention.attention.wk.weight")
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".attention.self.key.bias"):
            dst_k = src_k.replace(".attention.self.key.bias", ".attention.attention.wk.bias")
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".attention.self.value.weight"):
            dst_k = src_k.replace(".attention.self.value.weight", ".attention.attention.wv.weight")
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".attention.self.value.bias"):
            dst_k = src_k.replace(".attention.self.value.bias", ".attention.attention.wv.bias")
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".attention.output.dense.weight"):
            dst_k = src_k.replace(".attention.output.dense.weight", ".attention.attention.wo.weight")
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".attention.output.dense.bias"):
            dst_k = src_k.replace(".attention.output.dense.bias", ".attention.attention.wo.bias")
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".attention.output.LayerNorm.weight"):
            dst_k = src_k.replace(".attention.output.LayerNorm.weight", ".attention.norm.weight")
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".attention.output.LayerNorm.bias"):
            dst_k = src_k.replace(".attention.output.LayerNorm.bias", ".attention.norm.bias")
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".intermediate.dense.weight"):
            dst_k = src_k.replace(".intermediate.dense.weight", ".mlp.hidden.weight")
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".intermediate.dense.bias"):
            dst_k = src_k.replace(".intermediate.dense.bias", ".mlp.hidden.bias")
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".output.dense.weight"):
            dst_k = src_k.replace(".output.dense.weight", ".mlp.proj.weight")
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".output.dense.bias"):
            dst_k = src_k.replace(".output.dense.bias", ".mlp.proj.bias")
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".output.LayerNorm.weight"):
            dst_k = src_k.replace(".output.LayerNorm.weight", ".norm.weight")
        elif src_k.startswith("encoder.layer.") and src_k.endswith(".output.LayerNorm.bias"):
            dst_k = src_k.replace(".output.LayerNorm.bias", ".norm.bias")
        if dst_k is None:
            if src_k == "pooler.dense.weight" or src_k == "pooler.dense.bias":
                continue
            raise ValueError(f"Unrecognized Hugging Face parameter key: {src_k}")
        if dst_k not in dst_parameters:
            raise ValueError(f"Local model missing parameter key: {dst_k}")
        assert src_parameters[src_k].shape == dst_parameters[dst_k].shape, \
            f"Shape mismatch for {src_k} -> {dst_k}: {src_parameters[src_k].shape} != {dst_parameters[dst_k].shape}"
        with torch.no_grad():
            dst_parameters[dst_k].copy_(src_parameters[src_k])

def _download_from_hugging_face(model: str):
    r""" Download pre-trained model from HuggingFace and adopt the parameters. """
    assert model in WELL_KNOWN_MODELS
    print(f"> Downloading pre-trained '{WELL_KNOWN_MODELS[model]}'...", end="")
    try:
        from transformers import BertModel as _BertModel, BertTokenizer as _BertTokenizer
        stdout = sys.stdout
        stderr = sys.stderr
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                hugging_face_model = _BertModel.from_pretrained(WELL_KNOWN_MODELS[model]["base"])
                cfg = BertConfig()
                cfg.dim = hugging_face_model.config.hidden_size
                cfg.multiplier = hugging_face_model.config.intermediate_size // hugging_face_model.config.hidden_size
                assert cfg.multiplier * cfg.dim == hugging_face_model.config.intermediate_size
                cfg.heads = hugging_face_model.config.num_attention_heads
                cfg.depth = hugging_face_model.config.num_hidden_layers
                cfg.seq_len = hugging_face_model.config.max_position_embeddings
                cfg.vocab_size = hugging_face_model.config.vocab_size
                cfg.type_vocab_size = hugging_face_model.config.type_vocab_size
                cfg.bias = True # always true
                cfg.eps = hugging_face_model.config.layer_norm_eps
                assert hugging_face_model.config.hidden_act == "gelu"
                assert hugging_face_model.config.position_embedding_type == "absolute"
                assert hugging_face_model.config.model_type == "bert"
                assert WELL_KNOWN_MODELS[model]["dim"] == cfg.dim
                assert WELL_KNOWN_MODELS[model]["multiplier"] == cfg.multiplier
                assert WELL_KNOWN_MODELS[model]["heads"] == cfg.heads
                assert WELL_KNOWN_MODELS[model]["depth"] == cfg.depth
                assert WELL_KNOWN_MODELS[model]["seq_len"] == cfg.seq_len
                assert WELL_KNOWN_MODELS[model]["vocab_size"] == cfg.vocab_size
                assert WELL_KNOWN_MODELS[model]["type_vocab_size"] == cfg.type_vocab_size
                model = Bert(cfg)
                model.apply(model._init_weights)
                _adopt_from_hugging_face(model, hugging_face_model)
                model.eval()
                return model
            finally:
                sys.stdout = stdout
                sys.stderr = stderr
    finally:
        print("\n", end="")

def _load_model_by_name(model, path, download: bool | None = None):
    from pathlib import Path
    ckpt = os.path.join(path, f"{model}.ckpt")
    if not os.path.exists(ckpt) and download:
        _save_to_mat4(
            _download_from_hugging_face(model),
            ckpt
        )
    if not os.path.exists(ckpt):
        return None, None
    encoding = BertEncoding(model)
    cfg = BertConfig(**WELL_KNOWN_MODELS[model])
    model = Bert(cfg)
    model.apply(model._init_weights)
    _load_from_mat4(
        model,
        ckpt
    )
    model.eval()
    return model, encoding

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "FastAPI is handling this request on Azure right now.",
        "Latency tests should include both warm and cold runs.",
        "Embeddings let us compare text by semantic similarity.",
        "This is a short sentence.",
        "This is a slightly longer sentence that should still fit comfortably under the max token limit.",
        "🚀 Unicode emojis should not break tokenization.",
        "Caffè con panna is delicious.",
        "こんにちは、これは埋め込みのテストです。",
        "    The model should ignore leading and trailing whitespace.    ",
        "HTTP 200 is good; HTTP 404 is still useful for latency baselines.",
        "Azure App Service adds a bit of overhead compared to localhost.",
        "OpenAI’s endpoint is usually between 200ms and 400ms for small payloads.",
        "Please summarize the following paragraph in one sentence.",
        "The capital of France is Paris.",
        "1 2 3 4 5 6 7 8 9 10",
        "Special characters: !@#$%^&*()_+[]{}|;':\",./<>?",
        "Here is some code: `def hello(name): return f\"Hello, {name}\"`.",
        "Large language models can generate and embed text.",
        "Running multiple concurrent requests should hit the worker pool.",
        "This sentence is intentionally verbose so that we can check how the model handles inputs that approach the maximum text length parameter configured on the server.",
        "The weather today is cloudy with a chance of performance regressions.",
        "Please check CPU usage and memory consumption during this batch.",
        "Similar sentence to number 4: embeddings allow us to measure how close two pieces of text are.",
        "Totally unrelated topic: penguins live in the Southern Hemisphere.",
        "Another unrelated topic: GPU utilization on Azure can vary.",
        "lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Multiline\ntext\nshould\nstill\nwork.",
        "I wonder how fast this request will be when four workers are busy.",
        "End of test batch.",
    ]
    import argparse
    print(f"> python = {sys.version}")
    print(f"> numpy = {numpy.version.version}")
    print(f"> torch = {torch.version.__version__}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='all-MiniLM-L6-v2')
    parser.add_argument("--device", type=str, default="auto")
    args, _ = parser.parse_known_args()
    assert args.model in [
        'all-MiniLM-L6-v2',
        'all-MiniLM-L12-v2'
    ]
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.device in ["cuda", "cpu"]
    for k, v in args._get_kwargs():
        print(f"> {k} = {v}")
    print()
    try:
        torch.manual_seed(65537)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(65537)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model, encoding =  _load_model_by_name(
            args.model,
            os.path.dirname(__file__),
            download=True)
        if model is None or encoding is None:
            raise ValueError(f"Model '{args.model}' is not available.")
        model.to(args.device)
        assert model.training is False
        print(f"\x1b[38;2;{114};{204};{232}m")
        print(str(model))
        print(f"\x1b[0m", end="")
        def _calc_num_params(model):
            return sum(t.numel() for t in model.parameters())
        num_params = _calc_num_params(model)
        try:
            print(f"\n> parameters = {num_params:,}", end="")
        except:
            pass
        print(end="\n\n")
        encoded = encoding(sentences)
        with torch.inference_mode():
            token_embeddings = model(
                input_ids = encoded.data['input_ids'], 
                token_type_ids = encoded.data['token_type_ids'],
                attention_mask = encoded.data['attention_mask']
            )
            def mean_pooling(embeddings, attention_mask):
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                input_mask_expanded = input_mask_expanded.to(device=embeddings.device)
                return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            sentence_embeddings0 = mean_pooling(token_embeddings, encoded['attention_mask'])
            sentence_embeddings0 = torch.nn.functional.normalize(sentence_embeddings0, p=2, dim=1)
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer(WELL_KNOWN_MODELS[args.model]["base"])
        sentence_embeddings1 = _SentenceTransformer.encode(sentences)
        assert sentence_embeddings0.shape == sentence_embeddings1.shape
        assert numpy.allclose(
            sentence_embeddings0.cpu().numpy(),
            sentence_embeddings1,
            atol=1e-5
        )
        print("Success...\n")
    except Exception as e:
        print()
        print(f"\x1b[38;2;{243};{0};{0}mERROR:\x1b[0m ", end="")
        print(f"\x1b[38;2;{255};{255};{255}m{e}\x1b[0m")
        print()
        raise e
    finally:
        pass
