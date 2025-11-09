import os, json, requests, time, jwt

BASE_URL = "http://127.0.0.1:8000"
BASE_URL = "https://minilm-l6-v2-fsdhaggedqfrddhg.eastus-01.azurewebsites.net"

JWT_SECRET = os.getenv("APP_JWT_SECRET", "{CHANGEME}")
JWT_ALG = os.getenv("APP_JWT_ALG", "HS256")
JWT_ISS = os.getenv("APP_JWT_ISS", "")
JWT_LEEWAY_SECONDS = int(os.getenv("APP_JWT_LEEWAY", "30"))

def issue_token(
    subject: str,
    expires_in_seconds: int = 365 * 24 * 60 * 60, # 1 year
    extra_claims: dict | None = None,
) -> str:
    r"""
    Issue a long-lived JWT (default: 1 year).
    """
    now = int(time.time())
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + expires_in_seconds,
    }
    if JWT_ISS:
        payload["iss"] = JWT_ISS
    if extra_claims:
        payload.update(extra_claims)
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return token

from concurrent.futures import ThreadPoolExecutor, as_completed

def do_one_request(token: str, idx: int) -> float:
    t0 = time.time()
    url = f"{BASE_URL}/v1/embeddings"
    payload = {
        "model": "MiniLM-L6-v2",
        "input": "The quick brown fox",
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers)
    data = resp.json()
    elapsed = time.time() - t0
    return (idx, resp.status_code, elapsed)

def loadtest():
    token = issue_token(subject="client.py")
    total_requests = 1000 # how many requests in total
    concurrency = 4 # how many to run at the same time
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(do_one_request, token, i)
            for i in range(total_requests)
        ]
        for fut in as_completed(futures):
            idx, status, elapsed = fut.result()
            print(f"[{idx:03d}] status={status} latency={elapsed*1000:.2f}ms")

if __name__ == "__main__":
    loadtest()
