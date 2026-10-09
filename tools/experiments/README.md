# Experiment tooling (h100-private)

The scripts behind `docs/experiments/` (the ledger). They run on the training box; copies live there in `~/mtg-ml-checkpoints/` (archive, evals) and in each code tree (`launch_finetune.sh` as `launch_next.sh`).

| file | what |
|---|---|
| `launch_finetune.sh` | start one fine-tune arm: `SRC=<parent run dir> launch_finetune.sh <name> <cpus> <train,server GPU index> ft\|fresh [flags]`. `ft` copies the parent's `latest.pt` and pool and resumes; standard evaluation flags (1,000-game benchmark sampled + greedy, ladder L1) are built in. |
| `archive_run.sh` | copy a finished run into `~/mtg-ml-checkpoints/<id>/` (append-only); layout in `checkpoint-archive-README.md` |
| `final_evals.sh` | 2,000-game benchmark (sampled, greedy) of archived runs plus head-to-head pairs (`H2H="a b;c d"`), appended to `<id>/evals.txt` |
| `ladder_final.py` | ladder L1 Elo of archived runs (`WORKERS=30 python ladder_final.py <id>...`, from a code tree with `PYTHONPATH=.`) |

Evaluate each checkpoint on the code (feature set) it was trained with; since PR #24 the feature set travels with the checkpoint.
