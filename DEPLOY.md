# Deploying bb-embeddings | v1.0 | 2026-11-17 | BB

There is no GitHub source on the Railway service. A merge to `master` deploys
nothing. A deploy is a manual `railway up` of a clean checkout of `origin/master`
to project BB-Production, environment production, service `bb-embeddings`.

## Where the settings live

In the Railway service settings, not in this repo. `railway.json` was removed on
2026-11-17 because Railway stops reading it on 2026-12-01.

| Setting | Value |
|---|---|
| Builder | Dockerfile (`Dockerfile`) |
| Start command | `bash start.sh` |
| Healthcheck path | `/healthz` |
| Healthcheck timeout | 60 seconds |
| Restart policy | on failure, 5 retries |
| Replicas | 1 (US West) |

Proof they are read from the service: deployment `0564f47d` (2026-11-17,
11:35 AM PT / 19:35Z) was built from `fbc6a8d` with `railway.json` deleted. Its
manifest shows every value above and an empty file mapping.

## After a deploy, check

- `/version` shows the expected service and torch versions.
- `/healthz` returns 200 with `model_loaded: true`.
