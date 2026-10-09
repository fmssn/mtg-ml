# Naming runs and models

Until r8 every run got an ad-hoc name ("arm B", `r8-jund-blue-ft`, `lr075`, `r4-control`, `s2`) and the campaign directories had their own (`r8b`, `r8d`, `r8e3`, `r8f`). That works for a day and then nobody can say which model is which. This page defines the scheme; [`models.md`](models.md) / [`models.json`](models.json) apply it to every run so far.

## What other teams do (short survey)

- **Weights & Biases** gives every run a unique, immutable id (used in URLs and for resuming) and, separately, a human-readable run name that is not unique and can be edited; grouping and tags carry the rest. ([Runs](https://docs.wandb.ai/guides/runs/), "Find and customize a run's ID or name".)
- **MLflow Model Registry** separates the *registered model* (stable name) from its *versions* (immutable numbers) and adds **aliases**: "a mutable, named reference to a particular version", e.g. `champion`; production is switched by reassigning the alias, not by renaming anything. Tags carry free-form labels. ([Model Registry concepts](https://mlflow.org/docs/latest/ml/model-registry/).)
- **Hugging Face Hub** names a model `owner/short-name`, pins exact states by commit hash or tag (revision), and puts the structured facts (base model, datasets, metrics) in the model card metadata instead of the name. ([Model cards](https://huggingface.co/docs/hub/model-cards).)

Take-away: an identity that never changes, a short name humans can say, and *moving* pointers for roles. Facts go in structured fields, not in an ever longer name.

## The two layers

| layer | what | mutable? | used for |
|---|---|---|---|
| **run ID** | `<yyyymmdd>-<campaign>-<arch>-<scope>-<init>[-<variant>]-s<seed>` | never | run and campaign directories, archive ids, ledger `id` of new runs |
| **handle** | short unique name, `<campaign>-<what>`, e.g. `r7-lr075`, `r8-jund-blue` | never once assigned | what people say; `ft-<handle>` in another run's ID; the `handle` column in the dashboard |
| **role alias** | `best/general`, `best/jund_blue`, `play/jund` ... | moves | "which model should I use/deploy"; points to a handle |

One registry row per run ties them together: run ID, handle, all legacy names. Tools read the structured fields in `models.json`; they never parse IDs.

## Run ID grammar

```
<yyyymmdd>-<campaign>-<arch>-<scope>-<init>[-<variant>]-s<seed>[.x<n>]
```

| field | meaning | vocabulary |
|---|---|---|
| `yyyymmdd` | launch date (Berlin), same day as the first iteration | `20261009` |
| `campaign` | the round of experiments the run belongs to | `r8`, `fs4`, `scr` (throughput screens), `ov` (overnight), `rm`, `s64` |
| `arch` | feature set and width, `fs<N>h<W>` | `fs7h256`, `fs8h256`, `fs1h128`; other trunk choices (attention layer, GRU) are a variant word if they are not the default of that feature set |
| `scope` | what the network trains on | `all` (full matrix), `mu-<a>-<b>` (one matchup, `mu-jund-jund` for the mirror), `dk-<deck>` (one deck against the field), `mix<n>` (a named subset of n matchups, listed in the registry) |
| `init` | where the weights come from | `scratch`, or `ft-<parent handle>` (also for a resume: add variant `continue`) |
| `variant` | optional single word naming the one thing that differs from the control | `belief`, `lr075`, `ctl`, `attn`, `continue` |
| `s<seed>` | the `--seed` flag | `s8`, `s9` |
| `.x<n>` | attempt suffix for a crashed or aborted launch of the same run; the successful one has none | `.x1` |

Deck short names: `jund` (Jund Wildfire), `blue` (Mono Blue Terror), `madness` (Red Madness), `affinity` (Grixis Affinity), `elves`, `tron`. For a matchup, name the decks as `match.MATCHUPS` does (`jund_blue` becomes `mu-jund-blue`); with feature set 7 or later seats swap, so the order carries no meaning.

Rules:

1. Campaign, arch, scope and init describe facts; variant is the single difference under test. Two arms of one question share everything except variant (and seed for replications).
2. Seeds are part of the identity. A replication is a new run with a new seed, not a suffix on the old one.
3. Run directories, campaign directories and archive ids of **new** runs are the run ID. Old directories keep their names (the archive is append-only; running dirs are not renamed) and are mapped in the registry's legacy column.
4. IDs stay lowercase ASCII with hyphens; handles never contain a date.

## Handles

`<campaign>-<what>`, two to four hyphen-separated words, unique across the registry, chosen when the run is launched and never changed. Prefer deck words over cryptic abbreviations (`r8-jund-blue`, not `r8-jb`). Same-question replications append the seed (`r8-jund-blue-s2`); failed attempts append `-x<n>` (`r8-belief-x1`). Because handles contain hyphens, the init field is only unambiguous together with the registry; that is why tools use the JSON fields.

## Role aliases

Roles say what a model is *for*, and move when a better model arrives. Format `<kind>/<target>`:

| role | meaning |
|---|---|
| `best/general` | strongest full-matrix model; default parent for new work |
| `best/<matchup>` | strongest model for one matchup, e.g. `best/jund_blue`, `best/jund_jund` |
| `best/<deck>` | strongest model for one deck against the field |
| `play/<deck>` | the model the play site offers for that deck |

Moving a role is a documented act: change `roles` in `models.json`, the "Current roles" table in `models.md`, append to `role_history`, and say why in the ledger entry or PR that justifies it ("beats X on benchmark, L1 and head to head"). An alias never points to a run that is not in the registry.

## Examples

| run ID | reads as |
|---|---|
| `20261008-r7-fs7h256-all-scratch-lr075-s8` | r7, feature set 7 at width 256, full matrix, from scratch, the lr 7.5e-5 arm, seed 8 (handle `r7-lr075`) |
| `20261009-r8-fs7h256-mu-jund-blue-ft-r7-lr075-s8` | r8 fine-tune of `r7-lr075` on Jund vs Blue only (handle `r8-jund-blue`, formerly "arm C", `r8-jund-blue-ft`) |
| `20261009-r8-fs7h256-mu-jund-jund-ft-r7-lr075-s8` | the same for the mirror (handle `r8-jund-mirror`, "arm D", campaign dir `r8e3`) |
| `20261009-r8-fs7h256-dk-jund-ft-r7-lr075-s8` | Jund against the whole field (handle `r8-jund-pilot`) |
| `20261009-r8-fs8h256-all-scratch-belief-s8.x1` | the first, crashed launch of `r8-belief` |

## Launching a run

1. Pick the run ID and the handle, add the row to `models.json` and `models.md` (status `running`, legacy names empty), and use the run ID as the run directory name and campaign name.
2. When it ends: set the status (`stopped`, `archived`, ...), games, best bench and L1, and the archive path. Archive ids stay `<yyyymmdd>-<name>` where `<name>` is the run ID without its date.
3. Ledger entries use the run ID as `id` and mention the handle once.

## What was not renamed

Existing archives, the running campaign directories under `~/mtg-ml-r8/campaigns/`, the ledger ids and `mtg_ml/play_config.toml` model keys keep their names. The registry's legacy column lists each one, including the old shorthand ("arm C", `lr075`, `r8d`).
Seeds of runs before r7 were not recorded; the registry shows the default `--seed 0` for them (`seed_recorded: false` in the JSON).
