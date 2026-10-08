# Project agents

Use the selected Codex or Claude model with small task briefs and selective
delegation. The parent owns integration, final checks, commits and PR actions.
These roles support the existing tracker and autopilot, not a new scheduler.

## Routing

| Role | Assign when |
|---|---|
| `mtg-scout` | A bounded repository, dependency or status investigation would crowd the parent context |
| `mtg-engine` | Rules, cards or encoding need paired Python/Rust changes |
| `mtg-rl` | Features, models, PPO, inference or checkpoint behavior changes |
| `mtg-product` | Replay/play UI and live protocol need implementation or browser checks |
| `mtg-evidence` | Research, expert-data pipelines or experiment evidence need investigation |
| `mtg-automation` | CI, autopilot or workspace tooling needs a scoped change |
| `mtg-reviewer` | A substantial or risky diff needs an independent, evidence-backed review |

Canonical instructions live in `.agents/roles/`; read only the assigned role
and its `contract.md`. The adapters in `.codex/agents/` and `.claude/agents/`
contain routing metadata and pointers, not copies of domain instructions.
Keep role bodies below 500 words and descriptions below 35 words. If changing a
role, update both descriptions when its routing changes. No generator is needed.

Do small edits directly. Start one or two children for independent substantial
subtasks, at most three active per coordinator, subject to a lower runtime cap.
Children never delegate. Give each file one writer; pair both engines under one
owner. Serialize ledger, golden-digest and test-registration changes. A reviewer
works against an identified snapshot; coordinate ongoing edits with its parent.

Independent deliverables use separate existing workspaces. In a single
deliverable, children can share a checkout with disjoint file ownership. Only
the user creates local Conductor workspaces; these definitions do not automate
that action. See [Conductor's parallel-work guidance](https://www.conductor.build/docs/concepts/parallel-agents).

## Briefs and examples

Pass objective, completion criteria, workspace/branch/revision, relevant paths,
allowed writes, dependencies, checks, resource limits and artifact destination.
Prefer fresh context to a full conversation fork. With Codex's `fork_turns`
parameter, explicitly pass `"none"`; omitting it can copy the whole conversation.
Return at most 300 words with
outcome, changes, evidence, actual check results, gaps and artifact paths. Put
long logs in `.context/`; preserve essential evidence in the PR or committed
report before the workspace is archived. No persistent per-agent memory.

Example scout assignment:

> Use mtg-scout with a fresh context. In this workspace at the supplied HEAD,
> identify how engine tests run against Python and Rust. Read Makefile,
> tests/conftest.py and tests/test_difftest.py. No tracked writes or remote jobs.
> Return the registration rule, focused parity command and full differential
> command, with file/line references. Stop when those questions are answered.

Example implementation split:

> Parent owns the protocol contract and integration. mtg-product owns only the
> named app files; mtg-engine owns both implementations of the identified rules
> defect and their tests. Agree on observable behavior first. Neither child
> edits the other's files. Parent runs integration checks after both finish.

Example independent review:

> Use mtg-reviewer on the supplied base/head and listed uncommitted paths. Check
> hidden-information boundaries and missing regression coverage. No tracked
> writes. Run only the specified focused checks; record findings and reviewed
> revision, including unrun checks. Do not post GitHub comments or launch agents.

Reuse the locked progress updater located by `docs/context.md` for work owned
by this parent. Include timestamp, workspace, branch, exact write ownership,
dependencies and handoff artifact paths in the task note. Check GitHub for PR
state/head; historical rows and agent IDs do not establish current ownership.
If the external tracker is unavailable, say so and keep a local handoff instead
of creating a competing shared tracker. Resolve unclear overlapping ownership
before writing. A child reports blockers to its parent rather than taking over
another task. GPU/CPU jobs need current load checks and explicit assignments.

## Host configuration and boundaries

Codex uses standalone project TOML agents and
`agents.max_concurrent_threads_per_session = 3`. Agent files omit model and
reasoning overrides. Follow the invoking session's model/effort; if a user-level
subagent default overrides those, explicitly select the parent values when the
spawn interface allows it, otherwise report the mismatch. Project configuration
must be trusted/loaded by the client. See [OpenAI's subagent reference](https://learn.chatgpt.com/docs/agent-configuration/subagents).

Claude adapters use `model: inherit`, no `memory` or preloaded `skills`, and
exclude `Agent`. Project settings limit concurrent Agent children to three and
nesting to one layer. They deny the five generic user-level agents
`arch-code-reviewer`, `unit-test-writer`, `security-auditor`,
`code-quality-pipeline` and `doc-commit-push` here; their global files remain
available elsewhere. See [Claude's definition and settings reference](https://code.claude.com/docs/en/sub-agents).

These are per-session limits, not a global resource scheduler. Claude Workflow
agents and other host features have separate limits; resumed agents or runtime
overrides may also bypass a native cap. The parent still enforces the operating
contract. Read-only roles specify behavior; Bash access and inherited host
permissions mean the role text is not a security sandbox. No permission or
approval defaults are relaxed by this configuration.

After introducing the directories, start a fresh session to verify discovery.
ChatGPT hosted chats or other clients may not read local project adapters. If a
custom role is unavailable, pass its canonical role and contract to the host's
native subagent with the same limits. If delegation itself is unavailable,
execute serially and report that limitation. Do not launch a second provider
just to bypass host limitations.

The game-review skills and Claude Workflow remain the checkpoint-review entry
points with their existing models, evidence and calibration. They are separate
from ordinary development delegation. Run seeded-fault calibration before
claiming recall for a changed reviewer/prompt. These definitions do not launch
reviews, training or deployments by themselves.

## Assessment and validation

The October 8 assessment found good shared rules and workspace isolation but no
project-specific roles. Five generic Claude agents had 1,009 words of routing
descriptions and broad cross-project instructions. Blocking them avoids their
automatic pipelines; it does not by itself prove token savings or that a client
removes their descriptions from context.

At implementation start (2026-10-08, about 16:25 CEST), all six PRs below were
drafts. This is a dated snapshot, not a task board:

| Work | Observed status and dependency |
|---|---|
| [#55](https://github.com/fmssn/mtg-ml/pull/55) | Paired pilot card support; native CI failing; prerequisite to reconstruction |
| [#53](https://github.com/fmssn/mtg-ml/pull/53) | Training optimization; CI running; full performance/strength comparison outstanding |
| [#54](https://github.com/fmssn/mtg-ml/pull/54) | Autopilot capacity; required CI green; existing owner retains controller work |
| [#49](https://github.com/fmssn/mtg-ml/pull/49) | Play UI; required CI green; tracker records further user-requested UI changes |
| [#46](https://github.com/fmssn/mtg-ml/pull/46), [#44](https://github.com/fmssn/mtg-ml/pull/44) | Ledger and research plan; required CI green; coordinate document ownership |

The tracker also records reconstruction, an independent literature review and
r6 training. It still described merged #48/#50 as pending. Verify current state
before assigning work; no existing owner or PR is reassigned by this setup.

Validation procedure:

1. Parse TOML/JSON and Claude frontmatter; check names, corresponding adapters,
   canonical references, description/body limits and inherited-model settings.
2. Check discovery using installed clients, including the disabled Claude names.
3. Smoke a small task without delegation, a scoped scout investigation, paired
   engine ownership and independent review without tracked edits, in both hosts.
4. Compare serial and delegated forms of the same fixed investigation per
   provider. Record revision, machine, elapsed time, model, parent/child usage
   where exposed, repeat reads, correctness and rework. Separate cached tokens
   and unavailable metrics; do not infer child usage from parent totals.
5. Run lint and whitespace checks, open a draft PR and observe its CI. Config-
   only changes do not require local engine/GPU test campaigns.

Measured results are recorded in [agent-validation.md](agent-validation.md).
Parallelism can increase total tokens even when it reduces elapsed time. Use
measurements to adjust delegation; no percentage saving is promised.
