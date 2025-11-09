import os, json, requests, time, jwt

BASE_URL = "http://127.0.0.1:8000"

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

def main():
    token = issue_token(subject="client.py")
    url = f"{BASE_URL}/v1/embeddings"
    payload = {
        "model": "MiniLM-L6-v2",
        "input": "The quick brown fox"
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers)
    print("status:", resp.status_code)
    try:
        data = resp.json()
        print(json.dumps(data, indent=2))
    except Exception:
        print(resp.text)

if __name__ == "__main__":
    main()
