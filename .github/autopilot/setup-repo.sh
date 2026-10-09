#!/usr/bin/env bash
# One-time repo settings the autopilot relies on. Run once by hand: bash .github/autopilot/setup-repo.sh
set -euo pipefail
repo=${1:-fmssn/mtg-ml}
branch=$(gh api "repos/$repo" --jq .default_branch)

# Auto-merge, the "update branch" API, squash merges, tidy branches.
gh api -X PATCH "repos/$repo" \
  -F allow_auto_merge=true -F allow_update_branch=true \
  -F allow_squash_merge=true -F delete_branch_on_merge=true > /dev/null

# Default branch: CI's three checks must pass before anything merges. Not strict (no
# up-to-date requirement), so a merge does not force CI to re-run on every open PR.
gh api -X POST "repos/$repo/rulesets" --input - > /dev/null <<JSON
{
  "name": "autopilot: CI gates merges",
  "target": "branch",
  "enforcement": "active",
  "bypass_actors": [{"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}],
  "conditions": {"ref_name": {"include": ["refs/heads/$branch"], "exclude": []}},
  "rules": [
    {"type": "pull_request", "parameters": {
      "required_approving_review_count": 0, "dismiss_stale_reviews_on_push": false,
      "require_code_owner_review": false, "require_last_push_approval": false,
      "required_review_thread_resolution": false}},
    {"type": "required_status_checks", "parameters": {
      "strict_required_status_checks_policy": false,
      "required_status_checks": [
        {"context": "lint", "integration_id": 15368},
        {"context": "python", "integration_id": 15368},
        {"context": "native", "integration_id": 15368}]}}
  ]
}
JSON

for l in needs-opus needs-human autopilot:opus-used autopilot:round-1 autopilot:round-2; do
  gh label create "$l" --repo "$repo" --color ededed 2> /dev/null || true
done
echo "done: auto-merge on, ruleset on $branch, labels created"
