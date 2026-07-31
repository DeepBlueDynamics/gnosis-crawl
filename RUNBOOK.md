# grubcrawler — build / deploy / run book

Single source of truth for how grubcrawler is built, versioned, deployed, and rolled back. Three deploy targets, one image, one VERSION file.

---

## Deploy targets

| Target               | Image source                                        | Used by                                                |
|----------------------|-----------------------------------------------------|--------------------------------------------------------|
| **Cloud Run prod**   | `gcr.io/gnosis-459403/grubcrawler:<tag>`            | `grub.nuts.services`, `grubcrawler.dev` (Cloud Run service `grubcrawler`) |
| **Docker Hub public**| `deepbluedynamics/grubcrawler:<tag>`                | Self-hosters, agents, third-party Docker pulls         |
| **Local dev**        | local `docker-compose build` OR pull from GCR/DH    | Iteration without paying Cloud Build                   |

All three run the **same image**, listening on `$PORT` (default `6792`).

---

## Versioning

- **Single source of truth**: `VERSION` file at repo root (currently `0.13.1`).
- `app/main.py` reads it at startup; surfaces via `/health` and the FastAPI `/docs` page.
- `Dockerfile` copies `VERSION` into the image, so every container reports exactly what it was built from.
- Bump the version by editing the file. Tag git with `v<version>` on the commit you released. Don't skip the tag — that's how rollback finds it.

```powershell
# Bump
"0.14.0" | Set-Content VERSION
git add VERSION
git commit -m "bump version to 0.14.0"
git tag v0.14.0
git push --tags
```

---

## Build pipelines

Two independent surfaces. The Docker Hub side runs in GitHub Actions; the Cloud Run side runs in Cloud Build with Kaniko cache. Triggers and audit trails are separate by design.

### A. `cloudbuild-deploy.yaml` — Cloud Run (manual)

Fast Kaniko build (cache at `gcr.io/gnosis-459403/grubcrawler/cache/*`) on `E2_HIGHCPU_32`. Pushes to GCR (`:<sha>` and `:latest`), deploys to Cloud Run. **Does not** touch Docker Hub. First build ~25 min, subsequent ~3 min.

### B. `.github/workflows/release.yml` — Docker Hub (auto on tag)

Fires on `git push` of any `v*` tag. Builds the Dockerfile in GitHub Actions runner, pushes to `deepbluedynamics/grubcrawler` with semver + `:latest` tags. Uses GHA cache (`type=gha`) for camoufox layer reuse. Requires repo secrets `DOCKER_USERNAME` and `DOCKER_TOKEN`.

### C. `cloudbuild.yaml` — legacy Docker Hub from Cloud Build

Pre-GHA pipeline that pushes to Docker Hub from Cloud Build. Superseded by GHA. Kept only as fallback if GHA is down. Requires `dockerhub-token` in Secret Manager (which is **not** set up by default).

---

## One-time setup (do once, never again)

### Docker Hub push credentials (GitHub Actions)

`.github/workflows/release.yml` reads `DOCKER_USERNAME` and `DOCKER_TOKEN` from GitHub repo secrets. Set both at https://github.com/DeepBlueDynamics/grubcrawler/settings/secrets/actions:

- `DOCKER_USERNAME` = `deepbluedynamics`
- `DOCKER_TOKEN` = a Docker Hub Personal Access Token from hub.docker.com → Security → "New Access Token", scope **Read/Write/Delete**.

GitHub secrets are write-only — once set, you can't read the value back. To rotate, generate a new PAT and update the `DOCKER_TOKEN` secret value.

---

## Workflows

### 1. Make a code change (dev iteration → Cloud Run only)

```powershell
# ...edit code...
git add . ; git commit -m "fix: whatever"

$sha = (git rev-parse --short HEAD).Trim()
gcloud builds submit --config cloudbuild-deploy.yaml --project gnosis-459403 `
  --region=us-central1 --substitutions "_SHA=$sha"
```
~3 min. Verify:
```powershell
curl https://grub.nuts.services/health
```

### 2. Cut a versioned release (Cloud Run + Docker Hub)

Docker Hub is automated by GH Actions (`.github/workflows/release.yml`) — push a `v*` tag and it builds + pushes. Cloud Run is still manual.

```powershell
# 1. Bump VERSION, commit, tag, push the tag (this triggers GH Actions)
"0.14.0" | Set-Content VERSION
git add VERSION ; git commit -m "release v0.14.0"
git tag v0.14.0
git push origin main --tags

# 2. Deploy Cloud Run via Kaniko (manual)
$sha = (git rev-parse --short HEAD).Trim()
gcloud builds submit --config cloudbuild-deploy.yaml --project gnosis-459403 `
  --region=us-central1 --substitutions "_SHA=$sha"
```
After both finish (~3-5 min each, in parallel):
- `deepbluedynamics/grubcrawler:0.14.0`, `:v0.14.0`, `:latest` — pushed to Docker Hub by GHA
- `gcr.io/gnosis-459403/grubcrawler:<sha>`, `:latest` — in GCR
- Cloud Run service `grubcrawler` serving the new SHA
- `curl https://grub.nuts.services/health` reports `version: "0.14.0"` (from the VERSION file baked into the image)

Watch the GHA run at https://github.com/DeepBlueDynamics/grubcrawler/actions.

### 3. Roll back Cloud Run to a previous build (~30 sec, no rebuild)

Every Kaniko build tags with the git SHA. To roll back without rebuilding:
```powershell
gcloud run deploy grubcrawler --region us-central1 --project gnosis-459403 `
  --image gcr.io/gnosis-459403/grubcrawler:<old-sha-or-version> `
  --port 6792 --allow-unauthenticated
```
Or to roll back to the previous released version:
```powershell
gcloud run deploy grubcrawler --region us-central1 --project gnosis-459403 `
  --image gcr.io/gnosis-459403/grubcrawler:v0.13.1 --port 6792 --allow-unauthenticated
```

### 4. Run locally (no Cloud Build needed)

**Option A — local build (slowest first time, fast after):**
```powershell
./deploy.ps1 -Target local
# → docker-compose up -d. Listens on http://localhost:6792
```

**Option B — pull the latest image from GCR (after `gcloud auth configure-docker gcr.io`):**
```powershell
docker pull gcr.io/gnosis-459403/grubcrawler:latest
docker run -p 6792:6792 -e DISABLE_AUTH=true gcr.io/gnosis-459403/grubcrawler:latest
```

**Option C — pull the public image from Docker Hub:**
```powershell
docker run -p 6792:6792 -e DISABLE_AUTH=true deepbluedynamics/grubcrawler:latest
# Port-remap to taste: -p 8766:6792 maps host 8766 → container 6792
```

### 5. Rebuild Docker Hub image without a release (rare)

If a vuln patch needs to ship to self-hosters without a Cloud Run deploy:
```powershell
gcloud builds submit --config cloudbuild.yaml --project gnosis-459403 `
  --substitutions "_DATE_TAG=$(Get-Date -Format yyyyMMdd-HHmm)"
```
This uses the legacy `cloudbuild.yaml` (Docker Hub only, no Kaniko cache, no Cloud Run touch).

---

## Gotchas

- **Must run on Cloud Run `execution-environment=gen2`, not the gen1 default.** gen1 uses gVisor, a userspace syscall-emulation sandbox; Camoufox's Firefox binary segfaults (`Uncaught signal: 11`) on 100% of launch attempts under gen1 — confirmed 2026-07-31 by diffing identical images across gen1 (crashes) vs gen2 (works) vs local Docker (works, real kernel). `cloudbuild-deploy.yaml`'s deploy step passes `--execution-environment=gen2` for exactly this reason — do not remove it, and pass it on any manual `gcloud run deploy` too.
- **`gcloud run deploy` has not reliably auto-promoted traffic to the new revision on this service** — observed twice (2026-07-31): the revision builds, deploys, and goes Ready, but the old revision keeps serving 100% until traffic is explicitly promoted. `cloudbuild-deploy.yaml` now has a `promote-traffic` step for this; if deploying manually, always follow up with `gcloud run services update-traffic grubcrawler --region us-central1 --project gnosis-459403 --to-latest` and verify with `gcloud run services describe grubcrawler --format="value(status.traffic)"`.
- **`deploy.ps1 -Target cloudrun` deploys to the wrong service.** It hardcodes `$ServiceName = "grub-crawl"` (line 61), which is a stale duplicate Cloud Run service with no domain mapping. Production is `grubcrawler`. Use the direct `gcloud builds submit` flows above instead.
- **Cloud Build user substitutions must start with underscore** — `_SHA` works, `SHORT_SHA` is rejected.
- **Cold Kaniko build is still ~25 min** — only the first one after a deep Dockerfile change (e.g. changing the camoufox version). Cache lives 7 days; just don't let it expire.
- **`E2_HIGHCPU_32` worker, not the default 8**. Camoufox extracts to ~2 GB and OOMs Kaniko's snapshotter on small workers.
- **Cloud Run injects `PORT=8080`**, image's default is `6792`. The CMD honors `$PORT` so both work. Local: 6792. Cloud Run: 8080 internally, but `grub.nuts.services` proxies in front.
- **Auth gate**: `/mcp/*` and `/api/*` require `Authorization: Bearer <token>` on prod; `/`, `/dashboard`, `/login`, `/auth/callback`, `/health` are public. Local has `DISABLE_AUTH=true`.

---

## Authentication

Three token shapes, all issued by `nuts-auth` at `auth.nuts.services`:

| Token              | Shape                  | Lifetime  | Where it comes from                   | How grub validates it                   |
|--------------------|------------------------|-----------|---------------------------------------|----------------------------------------|
| Browser JWT        | `eyJ...` (3-part b64)  | 30 min    | Magic-link email login flow           | `GET auth.nuts.services/api/verify` with `Authorization: Bearer <jwt>`. Returns decoded claims. |
| `ahp_` API token   | `ahp_<40-char-opaque>` | 1 year    | nuts-auth dashboard → "New Token"     | `POST auth.nuts.services/auth` with form `token=<ahp_...>`. Returns a fresh JWT we then decode. |
| Internal HMAC      | `<b64>.<b64>` (2-part) | seconds   | pre-signed query-string URLs only     | Local HMAC verification — never leaves grub-crawl. |

`app/auth.py:AuthClient.validate_token` branches on shape. JWT detection: 2 dots, not prefixed with `ahp_`.

### Browser login flow (homepage / dashboard)

1. User hits `grub.nuts.services/` → topbar shows **Login**.
2. Click → `grub.nuts.services/login` (302) → `auth.nuts.services/login?return_url=https://grub.nuts.services/auth/callback`.
3. User enters email at auth.nuts.services. Submitted → both the magic-link email is sent AND the token-entry form is rendered immediately (one page covers both delivery hint and manual entry).
4. User clicks the email link → `auth.nuts.services/api/auth/token?token=<mail_token>&email=<...>&return_url=https://grub.nuts.services/auth/callback`. Server verifies, mints a JWT, redirects to the `return_url` with `?token=<JWT>`.
5. Browser lands on `grub.nuts.services/auth/callback?token=<JWT>` → tiny page stashes JWT in `localStorage.nuts_session_token` and redirects to `/dashboard`.
6. `/dashboard` reads `localStorage.nuts_session_token`, decodes the JWT payload for email display.

### Cross-origin token passing

`localStorage` is per-origin. To open `auth.nuts.services/dashboard` (token management UI) from `grub.nuts.services/dashboard` without forcing a re-login, the link is built dynamically as `https://auth.nuts.services/dashboard?token=<our-jwt>`. The auth dashboard's JS auto-captures `?token=` into its own (separate-origin) localStorage. Same pattern works for any cross-service link.

### Auth-gate paths

- **Public** (auth bypassed in `app/core/middleware.py`): `/`, `/dashboard`, `/login`, `/auth/callback`, `/health`, `/docs`, `/site`, `/view`, `/download`, `/tools`.
- **Token required**: everything else, especially `/api/*` and `/mcp/*`.
- Token sources accepted: `Authorization: Bearer <token>` header OR `?bearer_token=<token>` query string.

---

## Related services

| Service        | Cloud Run name  | Domain                  | Build pipeline                            |
|----------------|-----------------|-------------------------|-------------------------------------------|
| grubcrawler    | `grubcrawler`   | grub.nuts.services      | this repo's `cloudbuild-{deploy,release}.yaml` |
| nuts-auth      | `gnosis-auth`   | auth.nuts.services      | `nuts-auth/` repo, `gcloud run deploy --source` |
| dbd-site       | `deepblue-site` | deepbluedynamics.com    | `dbd-site/` repo                          |

---

## When something goes wrong

| Symptom                                | Check                                                                                |
|----------------------------------------|--------------------------------------------------------------------------------------|
| Build reports `FAILURE` but service updated | Cloud Build's `images:` validator. We removed it from cloudbuild-deploy.yaml — see also cloudbuild-release.yaml.|
| Kaniko OOM (`exit code 137`)           | Bump `machineType` higher, or `--single-snapshot` flag, or split camoufox into base image. |
| Email magic link not arriving          | `AGENTMAIL_API_KEY` unset on `gnosis-auth`. See `nuts-auth/` runbook.                |
| `/mcp` returns 404                     | Need trailing slash: `/mcp/`. FastAPI mount only matches `/mcp/<anything>`.          |
| `Failed to load tokens.` on auth dashboard | JWT expired (30-min default). Re-login, or bump `ACCESS_TOKEN_EXPIRE_MINUTES`.   |
| Try-it widget on homepage returns 401      | Either the JWT in localStorage is stale (re-login) or grub's `AuthClient.validate_token` is failing to reach `auth.nuts.services/api/verify`. Check nuts-auth logs for the token. |
| Magic-link email never arrives             | `AGENTMAIL_API_KEY` unset OR invalid on `gnosis-auth`. `gcloud logging read 'resource.labels.service_name="gnosis-auth" AND textPayload:"AgentMail"' --freshness=10m`. |
| MCP returns 500 `Task group not initialized` | FastMCP's `streamable_http_app` sub-app didn't get its lifespan started. `app/main.py` wraps `yield` with `async with mcp.session_manager.run():` — must stay nested. |
| Cloud Run uses port 6792 but `PORT=8080` injected | Image's CMD honors `$PORT` (defaults 6792). Cloud Run injects 8080 internally; we still set `--port=6792` on `gcloud run deploy` so the front-end maps the right port. |
| GHA release workflow fails on `docker/login-action` | `DOCKER_USERNAME` / `DOCKER_TOKEN` missing or wrong in repo secrets. See **One-time setup → Docker Hub push credentials (GitHub Actions)**. |
