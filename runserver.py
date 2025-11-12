import math, os, importlib, sys, time

from typing import Any, List, Optional, Union

# The model being hosted by this node

MODEL = os.getenv("MODEL", "all-MiniLM-L6-v2").strip()
print(f"> MODEL={MODEL}")
DEVICE = os.getenv("DEVICE", "auto").strip().lower()
if DEVICE not in ["auto", "cpu", "cuda"]:
    raise ValueError(f"Invalid environment variable 'DEVICE={DEVICE}'. Must be 'auto', 'cpu' or 'cuda'.")
print(f"> DEVICE={DEVICE}")

JWT_SECRET = os.getenv("APP_JWT_SECRET", "")
if not JWT_SECRET or JWT_SECRET == len(""):
    print("\r\nEnvironment variable 'JWT_SECRET' is not set.")
JWT_ALG = os.getenv("APP_JWT_ALG", "HS256")
JWT_ISS = os.getenv("APP_JWT_ISS", "")
JWT_LEEWAY_SECONDS = int(os.getenv("APP_JWT_LEEWAY", "30"))

# Max batch size supported.
MAX_BATCH = 128
# Max text length supported.
MAX_TEXT = 1024

import torch

def _load_model_by_name(
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
    path = os.path.join(os.path.dirname(__file__), f"{name}.py")
    spec = importlib.util.spec_from_file_location(md5(name), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, encoding = module._load_model_by_name(name)
    model.eval()
    return model, encoding

import asyncio

r""" The loadbalncing here is not state of the art. We could just serv on a single worker since
        the model does not have any state. """

class Worker:
    model: Any
    encoding: Any
    def _inference(
        self,
        inputs: List[str]
    ) -> List[List[float]]:
        if self.model is None or self.encoding is None:
             raise HTTPException(status_code=500, detail="Not available")
        inputs = [t.strip() for t in inputs]
        encoded = self.encoding(
            inputs
        )
        with torch.inference_mode():
            attention_mask = encoded["attention_mask"]
            # forward pass
            token_embeddings = self.model(
                input_ids=encoded.data["input_ids"],
                token_type_ids=encoded.data["token_type_ids"],
                attention_mask=attention_mask,
            )
            token_embeddings = token_embeddings.cpu()
            attention_mask = attention_mask.cpu()
            # mean pooling over non-masked tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()).float()
            token_embeddings = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1)
            # L2 normalize
            token_embeddings = torch.nn.functional.normalize(
                token_embeddings,
                p=2,
                dim=1
            )
            return token_embeddings.tolist()

class WorkerPool:
    def __init__(self):
        self.workers: List[Worker] = []
        self.queue: asyncio.Queue[Worker] = asyncio.Queue()
    async def initialize(self):
        torch.manual_seed(65537)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(65537)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        device = DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            device = "cpu"
        assert device in ["cpu", "cuda"]
        num = torch.cuda.device_count() if device == "cuda" and torch.cuda.is_available() else 4
        self.workers = []
        self.queue = asyncio.Queue()
        for _id in range(num):
            model, encoding = _load_model_by_name(MODEL)
            if model is None or encoding is None:
                print(f"> _load_from_check_point('{MODEL}') failed.", file=sys.stder)
            if device == "cuda":
                model.to(device= f"cuda:{_id}")
                print(f"> device=cuda:{_id}")
            worker = Worker()
            worker.model = model;
            worker.encoding = encoding;
            self.workers.append(worker)
            await self.queue.put(worker)
        print(f"> All {num} workers initialized!")
    async def acquire(self) -> Worker:
        return await self.queue.get()
    async def release(self, worker: Worker):
        await self.queue.put(worker)

import jwt

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pydantic import BaseModel

from contextlib import asynccontextmanager

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model workers...")
    app.state.WorkerPool = WorkerPool()
    await app.state.WorkerPool.initialize()
    worker = await app.state.WorkerPool.acquire()
    try:
        embeddings = worker._inference(inputs=["Warmup inference"])
    finally:
        await app.state.WorkerPool.release(worker)
    print(f"Ready...")
    yield

app = FastAPI(lifespan=lifespan)

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

r""" /v1/embeddings """

class EmbeddingRequest(BaseModel):
    r""" Request body for embedding generation. """
    model: str
    dimensions: int = 384
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
    model_config = {
            "exclude_none": True
        }

@app.post(
    "/v1/embeddings",
    response_model = EmbeddingResponse,
    response_model_exclude_none = True
)
async def embeddings(payload: EmbeddingRequest, principal: dict = Security(verify_principal)):
    if payload.model != MODEL:
        raise HTTPException(
            status_code=400, 
            detail=f"The specified model '{payload.model}' is not supported."
            )
    if payload.encoding_format != "float":
        raise HTTPException(
            status_code=400,
            detail=f"The specified encoding format '{payload.encoding_format}' is not supported.")
    if payload.dimensions != 384:
        raise HTTPException(
            status_code=400,
            detail=f"The specified dimensions '{payload.dimensions}' are not supported.")
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
    worker = await app.state.WorkerPool.acquire()
    try:
        t0 = time.time()
        embeddings = worker._inference(inputs)
    finally:
        await app.state.WorkerPool.release(worker)
    for i, v in enumerate(embeddings):
        data.append(
            EmbeddingData(
                index=i,
                embedding=v,
            )
        )
    print(f"> _inference: {int((time.time() - t0) * 1000)}ms")
    return EmbeddingResponse(
        model=payload.model,
        data=data
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
