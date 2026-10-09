# Hosting the play server

How to run the play-vs-model server (`python -m mtg_ml.hosted serve`) for a small
group of invited players on one Hetzner Cloud machine behind Cloudflare Access.

> Status (2026-10-09): **live at https://play.mtg-ml.com behind Cloudflare Access.**
> Stack `mtg-play-1` (Hetzner CAX21, fsn1) runs image `mtg-play:97fa6dc` with the
> pinned checkpoints. Cloudflare: zone `mtg-ml.com`, Zero Trust team
> `broken-mode-8274`, remotely managed tunnel `mtg-play-1`, Access app `mtg-play`
> (One-time PIN only, 24 h session) with the reusable policy `mtg-play allowlist`.
> Players are added or removed with the `play-players` skill
> (`.claude/skills/play-players/SKILL.md`), which keeps the policy and
> `MTG_ALLOWED_EMAILS` in sync. The image tag in service is pinned in
> `/opt/mtg-ml/deploy/.env` (gitignored) on the box.

## Architecture

```
player browser
   │  HTTPS (play.example.com)
   ▼
Cloudflare edge ── Access: email one-time PIN + allow policy (listed emails)
   │  Cf-Access-Jwt-Assertion header on every request
   ▼  (outbound tunnel, initiated from the server; no inbound port open)
Hetzner CAX21 (arm64), fsn1, Ubuntu 24.04         Hetzner firewall: inbound TCP 22 from the admin CIDR only
└─ Docker Compose (deploy/compose.yaml)
   ├─ cloudflared  ── tunnel run (TUNNEL_TOKEN) ──► http://app:8080
   └─ app          mtg-play image, uid 10001, read-only root fs, published on 127.0.0.1:8080 only
         /models        ← /srv/mtg-play/models   (ro, checkpoints pinned in mtg_ml/play_config.toml)
         /data/state    ← /srv/mtg-play/state    (play.sqlite3)
         /data/replays  ← /srv/mtg-play/replays
```

- The **app verifies the Access JWT itself** (issuer `MTG_ACCESS_TEAM_DOMAIN`,
  audience `MTG_ACCESS_AUD`, keys from `/cdn-cgi/access/certs`) and checks the
  email against `MTG_ALLOWED_EMAILS`. Cloudflare Access is the first gate, not the
  only one: a request that bypassed the tunnel would still be refused.
- Only `GET /healthz` is unauthenticated (used by the Docker healthcheck).
- Weights never enter the image or Git. The server refuses to start unless every
  checkpoint in `mtg_ml/play_config.toml` `[models]` is present with the pinned
  SHA-256 and feature set (`python -m mtg_ml.hosted check`).
- Game state survives restarts in SQLite; replays are files.

The interface between app and container (env variables, mounts, health) is
fixed in the hosted-play interface contract; the env variables are listed in
`deploy/app.env.example`.

### Files

| path | what |
|---|---|
| `deploy/Dockerfile` | multi-stage image: Rust + maturin build the native engine; CPU torch, numpy, `hosted` extra; python:3.11-slim runtime, uid 10001 |
| `deploy/compose.yaml` | `app` + `cloudflared` services |
| `deploy/app.env.example`, `deploy/cloudflared.env.example` | placeholders for `/etc/mtg-play/*.env` |
| `deploy/cloud-init.yaml` | server bootstrap: Docker Engine, directories, unattended-upgrades, ufw, sshd hardening |
| `tools/hetzner.py` | `inspect` / `plan` / `apply` / `teardown` of the Hetzner resources |
| `.dockerignore` | keeps `.venv`, `runs/`, checkpoints, `.git`, `.context` out of the build context |

CI job `container` builds the image and smoke-tests it (uid, imports, static
files, start-up with a dev user). It is informational, not a required gate.

## Cost and capacity (measured 2026-10-09)

Read-only `hcloud server-type describe cx33` on 2026-10-09 (UTC):

| | |
|---|---|
| CX33 | 4 vCPU (shared, x86), 8 GB RAM, 80 GB local disk |
| fsn1 (Falkenstein) | available |
| price | EUR 8.49/month net (EUR 10.10 gross), EUR 0.0136/hour net; 20 TB traffic included |
| primary IPv4 | billed separately (not in the server-type price) |
| image | `ubuntu-24.04` (id 161547269), available |

CX33 was sold out in fsn1, nbg1 and hel1 on 2026-10-09, so `mtg-play-1` runs on
**CAX21** (Ampere arm64, 4 vCPU shared, 8 GB RAM, 80 GB disk, EUR 10.49/month net
plus IPv4): `--server-type cax21`. The image builds natively on the box; nothing
in the Dockerfile is x86-specific, but CI builds only the x86 image.

Prices change; `tools/hetzner.py inspect` prints the current figures and `apply`
rechecks them right before creating anything.

## Inputs the operator provides

None of these are in the repository.

| input | where it goes |
|---|---|
| domain / hostname, e.g. `play.example.com` (zone on Cloudflare) | tunnel public hostname, Access application, `MTG_HOSTED_PUBLIC_ORIGIN` |
| Cloudflare Zero Trust team name | `MTG_ACCESS_TEAM_DOMAIN=https://<team>.cloudflareaccess.com` |
| Access application audience (AUD) tag | `MTG_ACCESS_AUD` |
| tunnel token | `TUNNEL_TOKEN` in `/etc/mtg-play/cloudflared.env` |
| allowed emails | Access policy **and** `MTG_ALLOWED_EMAILS` |
| admin CIDR(s) | `tools/hetzner.py --admin-cidr` (SSH source) |
| SSH public key | `tools/hetzner.py --ssh-public-key` |
| Hetzner API token | `HETZNER_TOKEN` in `~/conductor/workspaces/mtg-ml/.env` (or `--credentials`) |
| checkpoints | copied to `/srv/mtg-play/models` |

## 1. Provision (tools/hetzner.py)

The token is read from the dotenv file as data and passed to `hcloud` only as
`HCLOUD_TOKEN` in the child environment; it never appears in arguments, output
or the state file. Every resource gets the labels
`app=mtg-play,managed-by=mtg-ml-hetzner,stack=<stack>`; IDs are saved in
`~/.config/mtg-ml/hetzner-<stack>.json` (mode 0600, outside Git).

```bash
# read-only: token, catalog (type/location/price/image), our labelled resources
.venv/bin/python tools/hetzner.py inspect

# dry run: exactly what apply would create, keep or refuse
.venv/bin/python tools/hetzner.py plan \
    --ssh-public-key ~/.ssh/id_ed25519.pub --admin-cidr 203.0.113.7/32

# creates billed resources: ssh-key mtg-play-1-admin, firewall mtg-play-1-ssh, server mtg-play-1
.venv/bin/python tools/hetzner.py apply \
    --ssh-public-key ~/.ssh/id_ed25519.pub --admin-cidr 203.0.113.7/32 \
    --yes-create-paid-resources
```

Defaults: `--server-type cx33 --location fsn1 --image ubuntu-24.04 --stack mtg-play-1`,
user data `deploy/cloud-init.yaml`. `--admin-cidr` repeats (e.g. one IPv4 and one
IPv6 range); `0.0.0.0/0` and `::/0` are refused unless `--allow-open-ssh`.

Safety rules: a rerun reuses labelled resources and creates nothing twice; a
resource with one of our names but without our labels stops the run (nothing is
touched); an SSH key already registered under another name is refused (Hetzner
allows a key once). If the admin CIDR changes, `apply` replaces the rules of our
own firewall. `apply` prints the server's IPv4/IPv6 at the end.

## 2. Server bootstrap

cloud-init installs Docker Engine and the compose plugin from Docker's apt
repository, enables unattended security upgrades (no automatic reboot), sets
key-only SSH (`PermitRootLogin prohibit-password`, no passwords), enables ufw
(deny inbound, rate-limited 22/tcp) and creates:

| path | owner | mode |
|---|---|---|
| `/srv/mtg-play/models` | root | 0755 (mounted read-only) |
| `/srv/mtg-play/state`, `/srv/mtg-play/replays` | 10001:10001 | 0750 |
| `/srv/mtg-play/backups` | root | 0700 |
| `/etc/mtg-play` | root | 0700 |

Wait for it to finish, then check:

```bash
ssh root@<ipv4> cloud-init status --wait
ssh root@<ipv4> 'docker version && docker compose version && ufw status verbose'
```

Docker's published-port iptables rules bypass ufw; that is harmless here because
the only published port is bound to 127.0.0.1.

## 3. Checkpoints (read-only, SHA-256 checked)

Copy each checkpoint named in `mtg_ml/play_config.toml` `[models]` to
`/srv/mtg-play/models/<name>.pt`, e.g. `r7-lr075/policy.pt` and
`r4-control/policy.pt` (the `source` field says where each lives on h100-private).

```bash
ssh root@<ipv4> 'install -d -m 0755 /srv/mtg-play/models/r7-lr075 /srv/mtg-play/models/r4-control'
scp r7-lr075/policy.pt   root@<ipv4>:/srv/mtg-play/models/r7-lr075/policy.pt
scp r4-control/policy.pt root@<ipv4>:/srv/mtg-play/models/r4-control/policy.pt
ssh root@<ipv4> 'chown -R root:root /srv/mtg-play/models && find /srv/mtg-play/models -type f -exec chmod 0444 {} + \
  && sha256sum /srv/mtg-play/models/r7-lr075/policy.pt /srv/mtg-play/models/r4-control/policy.pt'
```

Compare the printed hashes with the `sha256` fields in `play_config.toml`
(`672aaa0c…68f636e` for r7-lr075/policy, `cb19222d…f85024f72` for
r4-control/policy). The container check below verifies them again.

## 4. Configuration

```bash
ssh root@<ipv4>
git clone https://github.com/<owner>/mtg-ml.git /opt/mtg-ml && cd /opt/mtg-ml && git checkout <release-rev>
install -m 0600 deploy/app.env.example /etc/mtg-play/app.env        # then edit: real values
install -m 0600 deploy/cloudflared.env.example /etc/mtg-play/cloudflared.env
```

Never set `MTG_HOSTED_INSECURE_DEV_USER` on the server. `MTG_HOSTED_GITHUB_FILING`
stays unset unless issue filing is wanted.

## 5. Build, check, start

```bash
cd /opt/mtg-ml
export GIT_REV=$(git rev-parse HEAD) MTG_PLAY_IMAGE=mtg-play:$(git rev-parse --short HEAD)
docker compose -f deploy/compose.yaml build app      # Rust LTO build: several minutes on a CX33
docker compose -f deploy/compose.yaml run --rm --no-deps app python -m mtg_ml.hosted check
docker compose -f deploy/compose.yaml up -d
docker compose -f deploy/compose.yaml ps             # app healthy, cloudflared running
curl -fsS http://127.0.0.1:8080/healthz             # model hashes + runtime
```

Record the image tag in use (e.g. `echo $MTG_PLAY_IMAGE >> /srv/mtg-play/RELEASES`)
and pin it for later commands: `echo MTG_PLAY_IMAGE=$MTG_PLAY_IMAGE > deploy/.env`
(gitignored; compose reads it automatically, otherwise it falls back to `mtg-play:local`).

## 6. Cloudflare tunnel and Access

1. Zero Trust → Networks → Tunnels → create a **cloudflared** tunnel; copy its
   token into `/etc/mtg-play/cloudflared.env`. Public hostname
   `play.example.com` → service `http://app:8080`.
2. Zero Trust → Settings → Authentication → add the **One-time PIN** login method.
3. Access → Applications → add a **self-hosted** application for
   `play.example.com` (whole host, no path), identity provider One-time PIN only,
   session duration e.g. 24 h. Copy the **Application Audience (AUD) tag** into
   `MTG_ACCESS_AUD`.
4. Policy: action **Allow**, include **Emails** = the invited addresses. Put the
   same list in `MTG_ALLOWED_EMAILS`.
5. `docker compose -f deploy/compose.yaml up -d` again after editing env files.

The same steps work through the Cloudflare API: `POST /accounts/{id}/cfd_tunnel`
(`config_src: cloudflare`), `PUT .../cfd_tunnel/{tunnel}/configurations` with the
ingress rule plus a final `http_status:404`, a proxied CNAME
`play` → `<tunnel>.cfargotunnel.com`, `POST /accounts/{id}/access/policies`, then
`POST /accounts/{id}/access/apps` (`type: self_hosted`, `allowed_idps` = the OTP
provider, `auto_redirect_to_identity: true`); its `aud` is `MTG_ACCESS_AUD`.
Fetch the token with `GET .../cfd_tunnel/{tunnel}/token` and pipe it over SSH into
`cloudflared.env`; never put it on a command line or in Git.

## 7. Verification

- `tools/hetzner.py inspect`: one server, firewall and key with our labels.
- From outside: `nc -vz <ipv4> 8080` and `:80`/`:443` fail; SSH works only from the admin CIDR.
- `https://play.example.com` without a session → Access login page; an email not
  on the policy cannot get a PIN through; an allowed email reaches the play page.
- A request to the origin without a valid JWT (e.g. `curl http://127.0.0.1:8080/play/` on the box) is refused by the app.
- `/healthz` lists the pinned model hashes; `docker inspect --format '{{.Config.User}}' mtg-play-app-1` is `10001:10001`.
- Play one game, restart the app (`docker compose restart app`), confirm the game resumes.

## 8. Backups

SQLite online backup (consistent while the app runs) plus the replays:

```bash
ts=$(date -u +%Y%m%dT%H%M%SZ)
sqlite3 /srv/mtg-play/state/play.sqlite3 ".backup '/srv/mtg-play/backups/play-$ts.sqlite3'"
tar -C /srv/mtg-play -czf /srv/mtg-play/backups/replays-$ts.tar.gz replays
install -m 0600 /srv/mtg-play/state/token.key /srv/mtg-play/backups/token-$ts.key
find /srv/mtg-play/backups -type f -mtime +14 -delete
```

Run it daily from root's crontab and copy `/srv/mtg-play/backups` off the box
(e.g. `rsync` to another machine, or a Hetzner Storage Box). Hetzner's own server
backups (+20% of the server price) are an alternative, not configured by default.
`token.key` (created by the app next to the database) derives the game tokens
players hold: without it, restored games cannot be opened again. Keep it with
the database backups and treat it as a secret. Checkpoints are not backed up here: their originals live in `~/mtg-ml-checkpoints`.

## 9. Restore

```bash
docker compose -f deploy/compose.yaml stop app
install -o 10001 -g 10001 -m 0640 /srv/mtg-play/backups/play-<ts>.sqlite3 /srv/mtg-play/state/play.sqlite3
rm -f /srv/mtg-play/state/play.sqlite3-wal /srv/mtg-play/state/play.sqlite3-shm
install -o 10001 -g 10001 -m 0600 /srv/mtg-play/backups/token-<ts>.key /srv/mtg-play/state/token.key
tar -C /srv/mtg-play -xzf /srv/mtg-play/backups/replays-<ts>.tar.gz && chown -R 10001:10001 /srv/mtg-play/replays
docker compose -f deploy/compose.yaml up -d
```

On a fresh server: provision, bootstrap, copy checkpoints and env files, then restore as above.

## 10. Upgrades

```bash
cd /opt/mtg-ml && git fetch && git checkout <new-rev>
export GIT_REV=$(git rev-parse HEAD) MTG_PLAY_IMAGE=mtg-play:$(git rev-parse --short HEAD)
docker compose -f deploy/compose.yaml build app
docker compose -f deploy/compose.yaml run --rm --no-deps app python -m mtg_ml.hosted check
# back up first (section 8), then replace the container and pin the new tag
docker compose -f deploy/compose.yaml up -d app
echo MTG_PLAY_IMAGE=$MTG_PLAY_IMAGE > deploy/.env && echo $MTG_PLAY_IMAGE >> /srv/mtg-play/RELEASES
```

A restart on the same release restores unfinished games from SQLite. A new
release (`GIT_REV` changed), a changed checkpoint or engine ends unfinished
games on purpose: the record stores choices as option indices, which another
rules revision could read differently, so players see "Cannot resume" rather
than a silently different game. Finished games stay reviewable. Deploy when
nobody is mid-game: `sqlite3 /srv/mtg-play/state/play.sqlite3 "SELECT count(*) FROM games WHERE status='active'"`.

New checkpoints: copy them in (section 3), update `play_config.toml` in the
release, run `check`, then restart. Upgrade cloudflared by changing its pinned
tag in `deploy/compose.yaml`. Host packages update unattended; reboot by hand
when `/var/run/reboot-required` exists.

## 11. Rollback

```bash
echo MTG_PLAY_IMAGE=mtg-play:<previous-short-rev> > deploy/.env   # see /srv/mtg-play/RELEASES
docker compose -f deploy/compose.yaml up -d app
```

If the newer release changed the SQLite schema, restore the backup taken before
the upgrade (section 9).

## 12. Teardown (deliberate)

```bash
.venv/bin/python tools/hetzner.py teardown                # dry run: lists what would be deleted
.venv/bin/python tools/hetzner.py teardown --yes-delete   # deletes server, firewall, ssh key
```

Only resources that carry our labels **and** whose IDs are in the state file are
deleted; anything else is listed as skipped. Take a final backup first. Remove
the Cloudflare tunnel, Access application and DNS record by hand.
