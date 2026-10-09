---
name: play-players
description: Add, remove or list the players allowed on the hosted play server https://play.mtg-ml.com (Cloudflare Access policy + app allowlist on mtg-play-1). Use when the user says "add <email>", "invite <name>", "remove <email>" or "who can play" in the context of the play site.
---

# Players on play.mtg-ml.com

Two lists must always match, or a player either can't log in or gets a 403 after logging in:

1. the Cloudflare Access policy **`mtg-play allowlist`** (who can get a login code), and
2. `MTG_ALLOWED_EMAILS` in `/etc/mtg-play/app.env` on `root@178.105.116.184` (who the app accepts).

Fixed IDs: account `1b9d8ad03ee720dd08c48d53be897784`, policy `d345342d-d4ea-42d0-8448-76dd32f0479c`.
Emails are stored lower-cased. Do it directly; no subagent and no confirmation needed for add/remove of the
emails the user named. Never touch other Access apps, policies or the `homeassistant` tunnel.

## 1. Cloudflare policy

Use the Cloudflare plugin MCP (`mcp__plugin_cloudflare_cloudflare__execute`; load it with ToolSearch).
Set `ADD` / `REMOVE` (both empty = list only):

```js
async () => {
  const ADD = ["new.player@example.com"], REMOVE = [];
  const path = `/accounts/${accountId}/access/policies/d345342d-d4ea-42d0-8448-76dd32f0479c`;
  const cur = (await cloudflare.request({method: "GET", path})).result;
  const before = cur.include.map(r => r.email?.email?.toLowerCase()).filter(Boolean);
  const drop = new Set(REMOVE.map(e => e.toLowerCase()));
  const emails = [...new Set([...before, ...ADD.map(e => e.toLowerCase())])].filter(e => !drop.has(e));
  if (!ADD.length && !REMOVE.length) return {emails: before};
  const res = await cloudflare.request({method: "PUT", path, body: {name: cur.name, decision: cur.decision,
    session_duration: cur.session_duration, include: emails.map(e => ({email: {email: e}}))}});
  return {ok: res.success, errors: res.errors, emails: res.result?.include.map(r => r.email?.email)};
}
```

If the MCP is not authorized, stop and tell the user to authorize the Cloudflare plugin via `/mcp`.

## 2. App allowlist (write exactly the list the policy returned)

```bash
ssh root@178.105.116.184 'EMAILS="a@x.com,b@y.com" python3 - <<"EOF"
import os, re
p = "/etc/mtg-play/app.env"
s = open(p).read()
s, n = re.subn(r"(?m)^MTG_ALLOWED_EMAILS=.*$", "MTG_ALLOWED_EMAILS=" + os.environ["EMAILS"], s)
assert n == 1, "MTG_ALLOWED_EMAILS line missing"
open(p, "w").write(s)
EOF
grep ^MTG_ALLOWED_EMAILS= /etc/mtg-play/app.env
cd /opt/mtg-ml && docker compose -f deploy/compose.yaml up -d'
```

`up -d` recreates only `app` (its env changed); open games are restored from SQLite. The image tag comes
from `/opt/mtg-ml/deploy/.env`, so no `export` is needed.

## 3. Verify

```bash
ssh root@178.105.116.184 'cd /opt/mtg-ml && sleep 20 && docker compose -f deploy/compose.yaml ps --format "{{.Service}} {{.Status}}" && curl -fsS http://127.0.0.1:8080/healthz'
```

`app` must be `healthy`. Report the final list. Tell the user that the new player logs in at
https://play.mtg-ml.com with that email and a one-time code. They don't need to register anywhere.
Update the allowlist line in the `play-vs-model-hosting` memory.
