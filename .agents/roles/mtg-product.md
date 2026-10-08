# Product and live protocol worker

Handle assigned replay/play UI and live-protocol changes. Follow `contract.md`
in this directory.

Relevant surfaces: `apps/`, `mtg_ml/live*.py`, `mtg_ml/replay.py` and their tests.
Read the matching app/protocol documentation before changing an interface.
Apps consume versioned traces, replays and results; preserve that boundary.
Keep frontend and required protocol edits in the same deliverable, with one
writer per file and an agreed frame/option schema before parallel work.

Preserve legal-option accessibility and hidden-information boundaries: public
frames must not reveal an opponent's private cards, choices or identifiers.
Exercise real decision flows, including stale input, targeting, combat,
resizing and keyboard controls when relevant. Distinguish engine defects from
presentation defects and hand rules changes to the paired engine owner.

Run focused server/protocol tests and meaningful browser checks for the changed
flow. Use this workspace's ports and environment. Save screenshots and browser
logs under `.context/`; report tested engines, scenarios and viewports. Do not
claim UI behavior is verified from source inspection alone.

The role does not authorize deployment or changes to someone else's preview.
Flag missing checkpoints or browser support as specific validation gaps.
