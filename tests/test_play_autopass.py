"""Web client auto-pass (apps/play/play.js autoAnswer), run under node.

Playtest 2026-10-09: after activating Krark-Clan Shaman once, the client
passed priority for the player, so the ability could not be stacked."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PLAY_JS = Path(__file__).resolve().parents[1] / "apps" / "play" / "play.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

# Evaluate only the automation block (isReal .. autoAnswer's end) against a stub client state.
RUNNER = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[1], 'utf8');
const a = src.indexOf('const isReal = ');
const b = src.indexOf('// The rest of a gesture the client collected');
const cfg = JSON.parse(process.argv[2]);
const S = {seat: 0, yields: new Set(), holdOnce: false, fullControl: false, plan: null, passMode: null,
  lastMine: -1, lastPriority: -1, reserved: [], payPlan: null, payAssign: null, autoRest: false,
  raw: [{actions: [], state: cfg.state}]};
const PREF = {stops: {me: {}, opp: {}}, autoPayNoAsk: false, autoTarget: false};
const last = () => S.raw.length - 1, opp = () => 1 - S.seat, stateAt = () => cfg.state, manaWaiting = () => 0;
const planAnswer = () => null;
eval(src.slice(a, b) + '; globalThis.autoAnswer = autoAnswer;');
const d = {kind: 'priority', options: cfg.refs.map(r => r.label), refs: cfg.refs};
console.log(JSON.stringify(autoAnswer(d)));
"""

PASS = {"type": "pass", "label": "Pass priority"}
KCS = {"type": "activate", "name": "Krark-Clan Shaman", "label": "Krark-Clan Shaman: 1 damage to each creature without flying"}
MANA = {"type": "mana", "label": "Tap Forest"}
DRAW = {"type": "activate", "name": "Ichor Wellspring", "label": "Ichor Wellspring: draw a card", "ability": "draw a card"}


def answer(refs, **kw):
    top = {"controller": 0, "name": "Krark-Clan Shaman: 1 damage", "card": "Krark-Clan Shaman"}
    state = {"stack": [top], "battlefield": [], "active": 0, "step": "main1", "turn": 1}
    out = subprocess.run(["node", "-e", RUNNER, str(PLAY_JS), json.dumps({"refs": refs, "state": state, **kw})],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def test_own_ability_on_stack_with_another_activation_keeps_the_stop():
    assert answer([PASS, KCS, MANA]) is None


def test_own_spell_on_stack_with_only_pass_or_mana_still_auto_passes():
    assert answer([PASS, MANA]) == 0
    assert answer([MANA, PASS]) == 1


def test_own_spell_on_stack_with_only_a_quiet_ability_still_auto_passes():
    assert answer([PASS, DRAW]) == 0

