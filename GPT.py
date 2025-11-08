# The original Open AI model - https://github.com/openai/gpt-2 here for debugging purposes.

import math, os, sys, time

import numpy, torch, tiktoken

from modeling import _Model, MultiHeadSelfAttentionBlock

from typing import Literal, Optional, Tuple, Union

from dataclasses import dataclass

@dataclass
class Config:
    dim: Optional[int] = 768
    multiplier: Optional[int] = 4
    heads: Optional[int] = 12
    degree_of_freedom: Optional[int] = 64
    depth: Optional[int] = 12
    bias: Optional[bool] = True
    seq_len: Optional[int] = 1024
    out_features: Optional[int] = 50257
    norm: Optional[Literal["layernorm", "rmsnorm"]] = "layernorm"
    eps: float = 1e-05

class GPT(_Model):
    r""" OpenAI GPT-2 """
    def __init__(self, cfg: Config, dropout: float = 0.):
        super().__init__()
        assert cfg.dim >= 1 and cfg.dim <= 1024
        assert cfg.dim % cfg.heads == 0 and cfg.dim // cfg.heads > 0
        assert cfg.multiplier in [1, 2, 3, 4]
        assert cfg.depth >= 1 and cfg.depth <= 12
        assert dropout >= 0 and dropout <= 1
        self.cfg = cfg
        self.wte = torch.nn.Parameter(torch.empty((cfg.out_features, cfg.dim)))
        self.wte.WEIGHT_DECAY = 1
        self.wpe = torch.nn.Parameter(torch.empty((cfg.seq_len, cfg.dim)))
        self.wpe.WEIGHT_DECAY = 1
        torch.nn.init.normal_(self.wte, std=0.02)
        torch.nn.init.normal_(self.wpe, std=0.02)
        self.transformer = torch.nn.ModuleDict(dict(
            drop = torch.nn.Dropout(dropout),
            h = torch.nn.ModuleList(
                [
                    MultiHeadSelfAttentionBlock(
                        in_features = cfg.dim,
                        norm = cfg.norm, # pre-attention normalization
                        eps = cfg.eps,
                        heads = cfg.heads,
                        DoF = cfg.dim // cfg.heads,
                        out_features = cfg.dim,
                        multiplier = cfg.multiplier,
                        bias = cfg.bias,
                        qkv = "qkv", # Hugging Face GPT implementation uses a single qkv projection
                        drop=dropout
                    )
                    for _ in range(cfg.depth)
                ]
            ),
            norm = torch.nn.LayerNorm(cfg.dim, bias=cfg.bias),
        ))
        self.head = torch.nn.Linear(cfg.dim, int(cfg.out_features), bias=False)
        # https://paperswithcode.com/method/weight-tying
        self.head.weight = self.wte

    def forward(self, inputs, targets=None):
        assert self.head.weight.data_ptr() == self.wte.data_ptr()
        B, T = inputs.size()
        assert T <= self.cfg.seq_len
        x = self.wte[inputs] + self.wpe[:T]
        x = self.transformer.drop(x)
        # https://github.com/meta-llama/llama-models/blob/a9c89c471f793423afd4cc3ca8671d6e56fe64cb/models/llama4/model.py#L407
        causal_attn_mask = None
        if T > 1:
            causal_attn_mask = torch.full((T, T), float("-inf"), device=x.device)
            causal_attn_mask = torch.triu(causal_attn_mask, diagonal=1).type_as(x)
            # https://github.com/pytorch/pytorch/issues/100005
            # torch.triu is buggy when the device is mps: filled values are
            # nan instead of 0.
            if causal_attn_mask.device.type == torch.device("mps").type:
                causal_attn_mask = torch.nan_to_num(causal_attn_mask, nan=0.0)
        for block in self.transformer.h:
            x, attention = block(x, causal_attn_mask)
        x = self.transformer.norm(x)
        if targets is not None:
            logits = self.head(x)
            loss = torch.torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.head(x[:, [-1], :]) 
            loss = None
        return logits, loss

def _download_from_hugging_face(out_ckpt):
    print("> Downloading a pre-trained model from: 'https://huggingface.co/openai-community/gpt2'...", end="")
    try:
        encoding = tiktoken.get_encoding("gpt2")
        from transformers import GPT2LMHeadModel
        stdout = sys.stdout
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            try:
                # This is the smallest version of GPT-2, with 124M parameters.
                #   https://huggingface.co/openai-community/gpt2
                hugging_face_model = GPT2LMHeadModel.from_pretrained("GPT2")
                cfg = Config()
                cfg.dim = 768
                cfg.multiplier = 4
                cfg.heads = 12
                cfg.degree_of_freedom = None # Calulated in GPT init based on dim and heads
                cfg.depth = 12
                cfg.seq_len = 1024
                cfg.out_features = 50257
                cfg.bias = True
                cfg.norm = "layernorm"
                cfg.eps = 1e-5
                model = GPT(cfg)
                model.apply(model._init_weights)
                _adopt_from_hugging_face(model, hugging_face_model)
                del hugging_face_model
                if out_ckpt is not None:
                    model.save(out_ckpt)
                model.eval()
                return model, encoding
            finally:
                sys.stdout = stdout
    finally:
        print("\n")

def _adopt_from_hugging_face(model, hugging_face_model):
    r""" adopt model parameters from hugging face impl. """
    hugging_face_parameters = hugging_face_model.state_dict()
    hugging_face_keys = hugging_face_parameters.keys()
    hugging_face_keys = [_hf_k for _hf_k in hugging_face_keys if not _hf_k.endswith('.attn.masked_bias')]
    hugging_face_keys = [_hf_k for _hf_k in hugging_face_keys if not _hf_k.endswith('.attn.bias')]
    parameters = model.state_dict()
    keys = list(parameters.keys())
    assert len(hugging_face_keys) == len(keys), f"Incompatible models: {len(hugging_face_keys)} != {len(keys)}"
    for _hf_k in hugging_face_keys:
        k = _hf_k
        k = k.replace("attn.c_attn", "att.qkv")
        k = k.replace("attn.c_proj", "att.wo")
        k = k.replace("mlp.c_fc", "mlp.hidden")
        k = k.replace("mlp.c_proj", "mlp.proj")
        k = k.replace("ln_f", "norm")
        k = k.replace("lm_head", "head")
        k = k.replace(".ln_1", ".att.pre")
        k = k.replace(".ln_2", ".mlp.norm")
        if _hf_k == 'transformer.wte.weight':
            k = 'wte'
        if _hf_k == 'transformer.wpe.weight':
            k = 'wpe'
        if any(_hf_k.endswith(w) for w in ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']):
            # OpenAI checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
            #   this means that we have to transpose these weights when we import them.
            assert hugging_face_parameters[_hf_k].shape[::-1] == parameters[k].shape
            with torch.no_grad():
                parameters[k].copy_(hugging_face_parameters[_hf_k].t())
        else:
            assert hugging_face_parameters[_hf_k].shape == parameters[k].shape
            with torch.no_grad():
                parameters[k].copy_(hugging_face_parameters[_hf_k])

def _load_from_check_point(device: str, in_ckpt):
    print("> Loading a pre-trained model from: '%s'..." % in_ckpt, end="")
    try:
        encoding = tiktoken.get_encoding("gpt2")
        cfg = Config()
        cfg.dim = 768
        cfg.multiplier = 4
        cfg.heads = 12
        cfg.depth = 12
        cfg.seq_len = 1024
        cfg.out_features = 50257
        cfg.bias = True
        cfg.norm = "layernorm"
        cfg.eps = 1e-5
        model = GPT(cfg)
        model.load(in_ckpt)
        model.to(device = device, dtype=torch.float32)
        model.eval()
        return model, encoding
    finally:
        print("\n")

@torch.inference_mode
def generate(model, encoding, queryString, max_gen_len: int = 256, temperature: float = 0., echo: bool = False):
    if max_gen_len is None or max_gen_len == 0 or max_gen_len >= model.cfg.seq_len:
        max_gen_len = model.cfg.seq_len - 1
    assert temperature is None or (temperature >= 0 and temperature <= 2)
    inputs = encoding.encode(queryString, allowed_special="all")
    if echo:
        queryResult=encoding.decode(inputs)
        yield queryResult
    max_gen_len = max_gen_len - len(inputs)
    inputs = (torch.tensor(inputs, dtype=torch.long)[None, ...])
    for _ in range(max_gen_len):
        model.eval()
        logits, loss = model(inputs)
        logits = logits[:, -1, :]
        if temperature is not None and temperature > 0:
            logits = logits / temperature
        top_k = 140
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        softmax = torch.torch.nn.functional.softmax(logits, dim=-1)
        if temperature is not None and temperature > 0:
            token = torch.multinomial(softmax, num_samples=1)
        else:
            token = torch.argmax(softmax, dim=-1).view(-1, 1)
        token = token.to(device=inputs.device)
        inputs = torch.cat((inputs, token), dim=-1)
        yield token.item()

@torch.inference_mode
def chat(model, encoding, args, temperature=0.7):
    while True:
        try:
            queryString = input(f"\x1b[38;2;{66};{244};{66}m{args.model}\x1b[0m>").strip()
            if queryString == "cls":
                os.system('cls' if os.name == 'nt' else 'clear')
                print("\x1b[H\x1b[2J\x1b[3J", end="")
                continue
            if queryString:
                print(f"\x1b[38;2;{255};{255};{255}m")
                try:
                    output = generate(
                        model=model,
                        encoding=encoding,
                        queryString=queryString,
                        temperature=temperature)
                    cc = 0;
                    toks = []
                    for t in output:
                        toks.append(t)
                        try:
                            s = encoding.decode(toks, errors='strict')
                            toks = []
                        except UnicodeDecodeError:
                            continue
                        if len(s.strip()) == 0 and cc == 0:
                            continue
                        print(s, end="")
                        cc += 1
                    print()
                finally:
                    print("\x1b[0m")
        except KeyboardInterrupt:
            return

if __name__ == "__main__":
    import argparse

    print(f"> python = {sys.version}")
    print(f"> numpy = {numpy.version.version}")
    print(f"> torch = {torch.version.__version__}")

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, default='GPT')
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--optim", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.9, 0.999])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup_steps", type=float, default=0.1)
    parser.add_argument("--train_steps", type=int, default=1000)

    args, _ = parser.parse_known_args()

    assert args.model in [
        'GPT'
    ]

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    assert args.device in ["cuda", "cpu"]

    assert 0 <= args.lr <= 1
    assert 0 <= args.weight_decay <= 1
    assert 0 <= args.dropout < 1
    assert 1 <= args.batch_size <= 1024
    assert 0 <= args.train_steps
    assert 0 <= args.warmup_steps

    for k, v in args._get_kwargs():
        print(f"> {k} = {v}")

    print()

    try:
        torch.manual_seed(65537)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(65537)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        ckpt = os.path.join(os.path.dirname(__file__), "GPT.ckpt")

        if not os.path.exists(ckpt):
           _download_from_hugging_face(out_ckpt=ckpt)

        model, encoding = _load_from_check_point(args.device, in_ckpt=ckpt)

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

        # _finetune(
        #     model,
        #     args=args,
        #     dataset="./data/The Adventures of Tom Sawyer.uint16"
        # )

        chat(
            model,
            encoding,
            args,
            temperature=0. # just a little creativity
        )

        print()

    except Exception as e:
        print()
        print(f"\x1b[38;2;{243};{0};{0}mERROR:\x1b[0m ", end="")
        print(f"\x1b[38;2;{255};{255};{255}m{e}\x1b[0m")
        print()
        raise e
    finally:
        pass