# bb-embeddings | tests/http_test.py | v1.0.0 | 2026-11-17 | BB
# HTTP checks against a running container (EMBEDDING_SERVICE.md test matrix).
# Standard library only, so it runs on the CI host, outside the image.
#
# Usage: python3 tests/http_test.py <base-url> <token>
# The container must be started with EMBED_SERVICE_TOKEN=<token>.

import base64
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "scene_512x384.jpg")
BOOT_WAIT_S = 240


def call(base, path, body=None, token=None):
    """Returns (status, parsed json or None). Never raises on an HTTP status."""
    headers = {"content-type": "application/json"}
    if token is not None:
        headers["x-embed-token"] = token
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, None


def wait_ready(base):
    deadline = time.monotonic() + BOOT_WAIT_S
    last = "no answer"
    while time.monotonic() < deadline:
        try:
            status, body = call(base, "/healthz")
            last = f"{status} {body}"
            if status == 200 and body and body.get("ok") is True and body.get("model_loaded") is True:
                return True, last
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last = repr(e)
        time.sleep(3)
    return False, last


def main(argv):
    base, token = argv[0].rstrip("/"), argv[1]
    results = []

    def check(label, ok, detail=""):
        results.append(ok)
        print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' - ' + str(detail)) if detail and not ok else ''}")

    ready, last = wait_ready(base)
    check(f"/healthz ready within {BOOT_WAIT_S}s", ready, last)
    if not ready:
        print("http test: FAIL (service never became ready)")
        return 1

    status, body = call(base, "/version")
    check("/version 200 with a torch version", status == 200 and bool(body and body.get("torch")), body)
    print(f"  /version -> {body}")

    with open(FIXTURE, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    good = {"image_b64": b64, "request_id": "ci-1"}

    status, body = call(base, "/embed", good, token)
    emb = (body or {}).get("embedding") or []
    check("/embed 200", status == 200, f"{status} {str(body)[:200]}")
    check("dim 384", len(emb) == 384 and (body or {}).get("dim") == 384, len(emb))
    n = math.sqrt(sum(x * x for x in emb)) if emb else 0.0
    check("norm 1.0 +/- 1e-5", abs(n - 1.0) <= 1e-5, n)
    check("request_id echoed", (body or {}).get("request_id") == "ci-1", body and body.get("request_id"))
    check("model is dinov2_vits14", (body or {}).get("model") == "dinov2_vits14", body and body.get("model"))

    status2, body2 = call(base, "/embed", good, token)
    check("same image twice gives the same vector", status2 == 200 and (body2 or {}).get("embedding") == emb)

    status, _ = call(base, "/embed", good, "wrong-token")
    check("wrong token is 401", status == 401, status)
    status, _ = call(base, "/embed", good, None)
    check("no token is 401", status == 401, status)

    status, _ = call(base, "/embed", {"image_b64": "A" * (256 * 1024 + 4)}, token)
    check("body over 256 KB is 413", status == 413, status)

    status, _ = call(base, "/embed", {"image_b64": "!!!not-base64!!!"}, token)
    check("bad base64 is 400", status == 400, status)

    status, _ = call(base, "/embed", {"image_b64": base64.b64encode(b"not an image").decode()}, token)
    check("valid base64 that is not an image is 400", status == 400, status)

    status, _ = call(base, "/embed", {"request_id": "x"}, token)
    check("missing image_b64 is 422", status == 422, status)

    failed = results.count(False)
    print(f"http test: {'FAIL' if failed else 'PASS'} ({len(results) - failed} pass, {failed} fail)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
