import sys, os, json, requests, time, jwt, numpy

BASE_URL = "http://127.0.0.1:8000"
# BASE_URL = "https://api.openai.com"

JWT_SECRET = os.getenv("APP_JWT_SECRET", "")

if not JWT_SECRET or JWT_SECRET == len(""):
    print("Environment variable 'JWT_SECRET' is not set.", file=sys.stderr)

JWT_ALG = os.getenv("APP_JWT_ALG", "HS256")
JWT_ISS = os.getenv("APP_JWT_ISS", "")
JWT_LEEWAY_SECONDS = int(os.getenv("APP_JWT_LEEWAY", "30"))

def issue_token(
    subject: str,
    expires_in_seconds: int = -1,
    extra_claims: dict | None = None,
) -> str:
    now = int(time.time())
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + expires_in_seconds,
    }
    if JWT_ISS:
        payload["iss"] = JWT_ISS
    r"""
    Issue a long-lived JWT (default: 1 year).
    """
    if expires_in_seconds < 0:
        expires_in_seconds = 365 * 24 * 60 * 60 # 1 year
    payload["exp"] = now + expires_in_seconds
    if extra_claims:
        payload.update(extra_claims)
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return token

from concurrent.futures import ThreadPoolExecutor, as_completed

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

def do_one_request(token: str, idx: int) -> float:
    t0 = time.time()
    url = f"{BASE_URL}/v1/embeddings"
    payload = {
        "model": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "input": sentences,
    }
    if BASE_URL == "https://api.openai.com":
        payload["model"] = "text-embedding-3-small"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers)
    data = resp.json()
    elapsed = time.time() - t0
    return (idx, resp.status_code, elapsed, len(data["data"]), data["data"])

def loadtest():
    token = issue_token(subject="client.py")
    if BASE_URL == "https://api.openai.com":
        token = os.getenv("OPENAI_API_KEY", "")
    total_requests = 100 # how many requests in total
    concurrency = 1 # how many to run at the same time
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(do_one_request, token, i)
            for i in range(total_requests)
        ]
        for fut in as_completed(futures):
            idx, status, elapsed, len_, embeddings = fut.result()
            print(f"[{idx:03d}] status={status} latency={elapsed*1000:.2f}ms batch={len_}")

def test():
    token = issue_token(subject="client.py")
    _, _, _, len_, data = do_one_request(token, 0)
    from sentence_transformers import SentenceTransformer
    _SentenceTransformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sentence_embeddings = _SentenceTransformer.encode(sentences)
    assert len_ == len(sentences)
    assert len_ == len(data)
    assert len_ == len(sentence_embeddings)
    for i in range(len_):
        sentence_embedding0 = numpy.array(data[i]["embedding"])
        sentence_embedding1 = sentence_embeddings[i]
        assert sentence_embedding0.shape == sentence_embedding1.shape
        assert numpy.allclose(
            sentence_embedding0,
            sentence_embedding1,
            atol=1e-5
        )
    print("Success!")

if __name__ == "__main__":
    # loadtest()
    test()
