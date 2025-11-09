import math, os, importlib, sys, time

from typing import List, Optional, Union

JWT_SECRET = os.getenv("APP_JWT_SECRET", "{CHANGEME}")
JWT_ALG = os.getenv("APP_JWT_ALG", "HS256")
JWT_ISS = os.getenv("APP_JWT_ISS", "")
JWT_LEEWAY_SECONDS = int(os.getenv("APP_JWT_LEEWAY", "30"))

# Max batch size supported

MAX_BATCH = 128

# Max text length supported

MAX_TEXT = 1024

_ALL_MODELS_ = {
    "MiniLM-L6-v2": {
        "dim": 384,
        "description": "MiniLM-L6-v2 embedding model",
        "url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    }
}

import torch

def _resolve(
    name
) -> torch.nn.Module:
    import hashlib, importlib
    r""" Resolve model by name.
         Dynamically loads the model from '{MODEL}.py' and calls the factory method. """
    def base48(
        n: int, pad: int = 2
    ) -> str:
        digits = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghkjmnpqrstuvwxyz'
        if n == 0:
            return digits[0].rjust(pad, digits[0])
        res = ""
        while n > 0:
            remainder = n % len(digits)
            res = digits[remainder] + res
            n //= len(digits)
        return res.rjust(pad, digits[0])
    def md5(
        s
    ) -> str:
        if sys.version_info >= (3, 9):
            md5_ = hashlib.md5(usedforsecurity=False)
        else:
            md5_ = hashlib.md5()
        md5_.update(s.encode("utf-8"))
        return base48(int.from_bytes(md5_.digest()))
    path = os.path.abspath(name + ".py").lower()
    spec = importlib.util.spec_from_file_location(md5(name), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._model()

for m in _ALL_MODELS_:
    _ALL_MODELS_[m]['model'], _ALL_MODELS_[m]['encoding'] = _resolve(m)

def inference(
    model: str,
    inputs: List[str]
) -> List[List[float]]:
    encoding = _ALL_MODELS_[model]["encoding"]
    model = _ALL_MODELS_[model]["model"]
    inputs = [t.strip() for t in inputs]
    encoded = encoding(
        inputs,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        attention_mask = encoded["attention_mask"]
        token_embeddings = model(
            input_ids=encoded.data["input_ids"],
            token_type_ids=encoded.data["token_type_ids"],
            attention_mask=attention_mask,
        )
        token_embeddings = token_embeddings.cpu()
        attention_mask = attention_mask.cpu()
        # mean pooling over valid tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()
        token_embeddings = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1)
        # L2 normalize each row
        token_embeddings = torch.nn.functional.normalize(
            token_embeddings,
            p=2,
            dim=1
        )
        return token_embeddings.tolist()

import jwt

def verify_jwt_token(token: str) -> dict:
    r"""
    Verify signature, exp (always), and issuer (if set).
    """
    decode_kwargs = {
        "key": JWT_SECRET,
        "algorithms": [JWT_ALG],
        "leeway": JWT_LEEWAY_SECONDS,
        # we do NOT set options={"verify_exp": False} -> so exp is enforced
    }
    if JWT_ISS:
        decode_kwargs["issuer"] = JWT_ISS
    try:
        payload = jwt.decode(token, **decode_kwargs)
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidIssuerError:
        raise HTTPException(status_code=401, detail="Invalid token issuer")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pydantic import BaseModel

app = FastAPI()

bearer_scheme = HTTPBearer(auto_error=False)

def verify_principal(
    bearer: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> dict:
    if bearer is None or bearer.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return verify_jwt_token(bearer.credentials)

r""" /health """

@app.get(
    "/health"
)
def health():
    return {"status": "ok"}

r""" /ready """

@app.get(
    "/ready"
)
def ready():
    for name, meta in _ALL_MODELS_.items():
        if "model" not in meta or "encoding" not in meta:
            raise HTTPException(status_code=503, detail=f"Model '{name}' not ready")
    return {"status": "ready"}

r""" /info """

@app.get("/info")
def info(principal: dict = Security(verify_principal)):
    return {
        "models": {
            name: {
                "dim": meta.get("dim"),
                "description": meta.get("description"),
                "url": meta.get("url"),
            }
            for name, meta in _ALL_MODELS_.items()
        },
    }

r""" /v1/embeddings """

class EmbeddingRequest(BaseModel):
    r""" Request body for embedding generation. """
    model: str
    dim: Optional[int] = 384
    input: Union[str, List[str]] = []
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None

class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]

class EmbeddingResponse(BaseModel):
    model: str
    usage: Optional[int] = None
    object: str = "list"
    data: List[EmbeddingData]

@app.post(
    "/v1/embeddings",
    response_model = EmbeddingResponse,
    response_model_exclude_none = True
)
def embeddings(payload: EmbeddingRequest):
    if payload.model not in _ALL_MODELS_:
        raise HTTPException(
            status_code=400, 
            detail=f"The specified model '{payload.model}' is not supported."
            )
    if payload.encoding_format != "float":
        raise HTTPException(
            status_code=400,
            detail=f"The specified encoding format '{payload.encoding_format}' is not supported.")
    if payload.dim != _ALL_MODELS_[payload.model]["dim"]:
        raise HTTPException(
            status_code=400,
            detail=f"The specified dimension '{payload.dim}' is not supported.")
    if isinstance(payload.input, str):
        inputs = [payload.input]
    else:
        inputs = payload.input
    if len(inputs) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"Too many inputs in one request. Max: {MAX_BATCH}")
    for t in inputs:
        if not isinstance(t, str):
            raise HTTPException(status_code=400, detail="The specified input format is not supported.")
        if len(t) > MAX_TEXT:
            raise HTTPException(status_code=400, detail=f"Input too long. Max: {MAX_TEXT}")
    data: List[EmbeddingData] = []
    embeddings = inference(payload.model, inputs)
    for i, v in enumerate(embeddings):
        data.append(
            EmbeddingData(
                index=i,
                embedding=v,
            )
        )
    return EmbeddingResponse(
        data=data,
        model=payload.model,
    )

if __name__ == "__main__":
    embeddings = inference("MiniLM-L6-v2", 
              ["The quick brown fox",
              "jumped over a lazy dog"]
    )
