import math, os, sys, time

import numpy, torch

from modeling import _Model, MultiHeadSelfAttention, MLP

from typing import Callable, Literal, Optional, Tuple, Union

from dataclasses import dataclass

@dataclass
class Config:
    dim: int = 384
    seq_len: int = 512
    vocab_size: int = 30522
    type_vocab_size: int = 2
    multiplier: int = 4
    heads: int = 12
    depth: int = 6
    bias: bool = True
    eps: float = 1e-12

class BertModel(_Model):
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
            self.attention = BertModel.BertAttention(
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

    def __init__(self, cfg: Config, dropout: float = 0.1):
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
                    BertModel.BertLayer(
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

    @classmethod
    def download(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = "sentence-transformers/all-MiniLM-L6-v2"):
        """ Load model from pre-trained 'sentence-transformers/all-MiniLM-L6-v2'. """
        print("> Downloading pre-trained 'sentence-transformers/all-MiniLM-L6-v2'...", end="")
        try:
            from transformers import BertModel as _BertModel, BertTokenizer
            stdout = sys.stdout
            stderr = sys.stderr
            with open(os.devnull, "w") as devnull:
                sys.stdout = devnull
                sys.stderr = devnull
                try:
                    encoding = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
                    hugging_face_model = _BertModel.from_pretrained(pretrained_model_name_or_path)
                    cfg = Config()
                    cfg.dim = 384
                    cfg.multiplier = 4
                    cfg.heads = 12
                    cfg.depth = 6
                    cfg.seq_len = 512
                    cfg.vocab_size = 30522
                    cfg.type_vocab_size = 2
                    cfg.bias = True
                    cfg.eps = 1e-12
                    model = BertModel(cfg)
                    model.apply(model._init_weights)
                    cls._adopt_from_hugging_face(model, hugging_face_model)
                    model.eval()
                    return model, encoding
                finally:
                    sys.stdout = stdout
                    sys.stderr = stderr
        finally:
            print("\n", end="")

    @staticmethod
    def _adopt_from_hugging_face(dst, src):
        """ Adopt parameters from Hugging Face model to local model. """
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

    @classmethod
    def load(cls, ckpt: Union[str, os.PathLike], pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = "sentence-transformers/all-MiniLM-L6-v2"):
        """ Load model from local checkpoint. """
        print(f"> Loading checkpoint '{ckpt}'...", end="")
        try:
            from transformers import BertTokenizer
            stdout = sys.stdout
            stderr = sys.stderr
            with open(os.devnull, "w") as devnull:
                sys.stdout = devnull
                sys.stderr = devnull
                try:
                    encoding = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
                    cfg = Config()
                    cfg.dim = 384
                    cfg.multiplier = 4
                    cfg.heads = 12
                    cfg.depth = 6
                    cfg.seq_len = 512
                    cfg.vocab_size = 30522
                    cfg.type_vocab_size = 2
                    cfg.bias = True
                    cfg.eps = 1e-12
                    model = BertModel(cfg)
                    model.apply(model._init_weights)
                    model._load(ckpt)
                    model.eval()
                    return model, encoding
                finally:
                    sys.stdout = stdout
                    sys.stderr = stderr
        finally:
            print("\n", end="")

def _model():
    r""" Factory """
    ckpt = os.path.join(os.path.dirname(__file__), "MiniLM-L6-v2.ckpt")
    if not os.path.exists(ckpt):
        model, encoding =  BertModel.download()
        model._save(ckpt)
    model, encoding =  BertModel.load(ckpt)
    return model, encoding

if __name__ == "__main__":
    import argparse
    print(f"> python = {sys.version}")
    print(f"> numpy = {numpy.version.version}")
    print(f"> torch = {torch.version.__version__}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='MiniLM-L6-v2')
    parser.add_argument("--device", type=str, default="auto")
    args, _ = parser.parse_known_args()
    assert args.model in [
        'MiniLM-L6-v2'
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
        model, encoding =  _model()
        model.to(args.device)
        print(f"\x1b[38;2;{114};{204};{232}m", end="")
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
        sentences = [
            "The weather is lovely today.",
            "It's so sunny outside!",
            "He drove to the stadium.",
        ]
        encoded = encoding(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            token_embeddings = model(
                input_ids = encoded.data['input_ids'], 
                token_type_ids = encoded.data['token_type_ids'],
                attention_mask = encoded.data['attention_mask']
            )
            def mean_pooling(token_embeddings, attention_mask):
                token_embeddings = token_embeddings.cpu()
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1)
            sentence_embeddings = mean_pooling(token_embeddings, encoded['attention_mask'])
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            # print(sentence_embeddings.shape)
            # print(sentence_embeddings)
        # from sentence_transformers import SentenceTransformer
        # _SentenceTransformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # sentence_embeddings = _SentenceTransformer.encode(sentences)
        # print(sentence_embeddings.shape)
        # print(sentence_embeddings)
        print()
    except Exception as e:
        print()
        print(f"\x1b[38;2;{243};{0};{0}mERROR:\x1b[0m ", end="")
        print(f"\x1b[38;2;{255};{255};{255}m{e}\x1b[0m")
        print()
        raise e
    finally:
        pass
