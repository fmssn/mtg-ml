# How to add a new deck or cards

The card pool is data: **`mtg_ml/engine/cards.toml`** is the single source of truth, loaded by the Python reference engine (`mtg_ml/engine/cards.py`) and by the Rust port (`native/src/cards.rs`, which receives the file's text from Python at import time). Effects are lists of **ops** that are implemented once per engine. So there are three cases, from cheapest to most expensive:

| the new card needs | you edit | Rust rebuild |
|---|---|---|
| only existing ops, targets, costs and triggers | `cards.toml` (+ oracle snapshot, tests) | no |
| a new reusable op (e.g. "target's controller loses N life") | `cards.toml`, one function in `cards.py`, one enum variant + parser arm + `run_op` arm in Rust | yes |
| a one-off effect (Brainstorm, Ponder, Duress...) | as above, as a `custom` effect | yes |
| a rule the engine does not have (first strike, a new cost type, scry N > 1...) | `game.py` first, then the same change in `state.rs` / `engine.rs` | yes |

In every case the order is the same: **Python reference first, then Rust, then the differential tests must pass.** The Python engine defines what is correct; the Rust engine must match it bit for bit (options in the same order, the same labels, keys and prompts, the same RNG use). [native-engine.md](native-engine.md) explains why that matters and how the comparison works.

- [Worked example: Vapor Snag](#worked-example-vapor-snag)
- [Checklist](#checklist)
- [Reference: cards.toml fields](#reference-cardstoml-fields)
- [Reference: ops](#reference-ops)
- [Writing an op so both engines agree](#writing-an-op-so-both-engines-agree)
- [Adding a whole deck](#adding-a-whole-deck)
- [Bots and the feature vocabulary](#bots-and-the-feature-vocabulary)

## Worked example: Vapor Snag

[Vapor Snag](https://scryfall.com/search?q=%21%22Vapor+Snag%22) ({U}, Instant, Pauper-legal, a Mono Blue staple): *"Return target creature to its owner's hand. Its controller loses 1 life."* The bounce exists (`bounce_target`, used by Steel Sabotage); the life loss does not. This is the real change in this repository (commit "Add Vapor Snag ..."); it is in the pool but in no decklist yet.

**1. Oracle snapshot.** Append the Scryfall fields to `data/oracle_cards.json` (same keys as the other entries):

```bash
curl -s "https://api.scryfall.com/cards/named?exact=Vapor+Snag" | python -c "import json,sys; c=json.load(sys.stdin); print(json.dumps({k: c.get(k) for k in ('name','mana_cost','type_line','oracle_text','power','toughness','keywords','cmc')}, indent=1))"
```

```json
{"name": "Vapor Snag", "mana_cost": "{U}", "type_line": "Instant", "oracle_text": "Return target creature to its owner's hand. Its controller loses 1 life.", "power": null, "toughness": null, "keywords": [], "cmc": 1.0}
```

`tests/test_card_spec.py` checks every `cards.toml` card against this file (cost, types, subtypes, power/toughness; double-faced cards via `faces`).

**2. The spec** in `cards.toml`:

```toml
[[card]]
name = "Vapor Snag"
cost = "{U}"
types = "Instant"
text = "Return target creature to its owner's hand. Its controller loses 1 life."
targets = ["creature"]
effect = [{ op = "lose_life", who = "target_controller", n = 1 }, { op = "bounce_target" }]
```

The life loss is listed first because ops re-read the target and the bounce moves it (see [the rules below](#writing-an-op-so-both-engines-agree)). The result is the same: nothing can happen between the two.

**3. The new op in Python** (`mtg_ml/engine/cards.py`), generic enough to be reused:

```python
def _op_lose_life(g, item, op):
    """who: you | opponent | target_player | target_controller (of the targeted permanent)."""
    who = op["who"]
    if who == "you":
        p = item.controller
    elif who == "opponent":
        p = 1 - item.controller
    else:
        t = g.target(item)
        if t is None:
            return
        p = t[1] if who == "target_player" else t.controller
    g.players[p].life -= op["n"]

OPS = {..., "lose_life": _op_lose_life, ...}
```

At this point the card works in the reference engine. Write its tests (step 5) and run them before touching Rust.

**4. The same op in Rust.** Three places, each next to its siblings:

`native/src/cards.rs`: the variant, its allowed keys and its parser.

```rust
pub enum Who { You, Opponent, TargetPlayer, TargetController }

pub enum Op {
    ...
    LoseLife { who: Who, n: i32 },
}

// parse_ops(): allowed keys ...
"lose_life" => &["op", "who", "n"],
// ... and the parser arm
"lose_life" => Op::LoseLife {
    who: match req_str(t, "who")? {
        "you" => Who::You,
        "opponent" => Who::Opponent,
        "target_player" => Who::TargetPlayer,
        "target_controller" => Who::TargetController,
        w => return Err(format!("lose_life: unknown who {w:?}")),
    },
    n: n()?,
},
```

`native/src/engine.rs`, `Eng::run_op`: the behaviour, mirroring the Python line by line.

```rust
Op::LoseLife { who, n } => {
    let st = self.s();
    let p = match who {
        Who::You => Some(ctl),
        Who::Opponent => Some(1 - ctl),
        Who::TargetPlayer => match st.target(item, 0) {
            Some(Tgt::Player(p)) => Some(p),
            _ => None,
        },
        Who::TargetController => match st.target(item, 0) {
            Some(Tgt::Card(c)) => Some(st.c(c).controller),
            Some(Tgt::Spell(sid)) => Some(st.stack[st.stack_pos(sid).unwrap()].controller),
            _ => None,
        },
    };
    if let Some(p) = p {
        st.players[p as usize].life -= n;
    }
}
```

Rebuild: `cd native && maturin develop --release` (or `maturin build` + `pip install`). An op that exists only in Python makes the native module refuse the spec at import ("unknown op ...: add it to Op (native/src/cards.rs) and Eng::run_op (native/src/engine.rs)"), so a forgotten Rust side cannot go unnoticed.

**5. Per-card rules tests**, in the deck's test file (`tests/test_cards_blue.py`). They use the scenario helpers and run on **both engines automatically** (`tests/conftest.py`):

```python
def test_vapor_snag_bounces_and_its_controller_loses_life():
    g = scenario(p0={"battlefield": ["Gixian Infiltrator"]}, p1={"hand": ["Vapor Snag"], "battlefield": ISLANDS(1)}, active=1)
    choose(g, "Cast Vapor Snag")  # the only target and the only payment are taken by settle()
    resolve_stack(g)
    assert "Gixian Infiltrator" not in bf(g) and names(g.players[0].hand) == ["Gixian Infiltrator"]
    assert g.players[0].life == 19 and g.players[1].life == 20
    assert names(g.players[1].graveyard) == ["Vapor Snag"]
```

plus the cases that make the card interesting: bouncing your own creature costs you the life, and the spell fizzles (no life loss) if the creature is gone. Test the exact option labels where a decision is offered.

```bash
pytest tests/test_cards_blue.py -k vapor      # [python] and [native] variants
```

**6. Differential fuzz.** A card in no decklist is never played by the fuzzer, so swap it into a deck for the run (`SEAT:NAME:COPIES`, replacing the deck's last cards):

```bash
python -m mtg_ml.difftest fuzz --games 400 --jobs 8 --with-card "1:Vapor Snag:4"
# 400/400 games identical in both engines (35s)
```

Check that the card was actually exercised (Vapor Snag was cast 53 times in 60 such games). When the card goes into a real decklist, the normal `pytest tests/test_difftest.py` and `python -m mtg_ml.difftest fuzz` cover it from then on.

**7. Decklists, bots, features** — see [Adding a whole deck](#adding-a-whole-deck) and [Bots and the feature vocabulary](#bots-and-the-feature-vocabulary). For Vapor Snag that would be: add it to `MONO_BLUE_TERROR` (or a sideboard plan), give `BlueBot.cast_score` a rule for it (the bots never cast cards they do not know), and regenerate the golden digests because the decks changed.

## Checklist

For each new card:

- [ ] Oracle entry in `data/oracle_cards.json` (Scryfall fields; `faces` for double-faced cards).
- [ ] `[[card]]` entry in `mtg_ml/engine/cards.toml` (`[[token]]` for tokens it creates, `[[face]]` for a back face). Only existing ops? Then no code and no Rust rebuild.
- [ ] New op / custom effect: Python in `cards.py` (`OPS` / `CUSTOM`) **first**; then Rust: `enum Op` (or `enum Custom`) + allowed keys + parser arm in `native/src/cards.rs`, behaviour in `Eng::run_op` (or `Eng::custom`) in `native/src/engine.rs`. Rebuild the native module.
- [ ] New rule, keyword with rules meaning, target kind, trigger event, cost type or decision kind: change `game.py` and the matching code in `native/src/state.rs` / `engine.rs` (see the table in [Reference: cards.toml fields](#reference-cardstoml-fields) for where each lives). A new decision kind also goes into `PUBLIC_KINDS` (`rl/features.py`) and `is_public` (`native/src/features.rs`).
- [ ] Per-card rules tests in `tests/test_cards_<deck>.py` (scenario helpers; they run on both engines). Cover every option the card offers and its edge cases (fizzling, illegal targets, empty library...).
- [ ] `pytest` passes (both engine variants, `test_card_spec.py`, golden digests).
- [ ] `python -m mtg_ml.difftest fuzz --games 500 --jobs 8 --with-card "SEAT:NAME:4"` reports 0 divergences, and the card was actually played.
- [ ] If it joins a decklist: `decks.py`, golden digests regenerated on purpose (`python -m mtg_ml.trace record --games 210`, say so in the commit), the deck's bot taught to play it, README decklist text.
- [ ] RL: nothing to register (hashed features), but existing checkpoints have never seen the card; plan to fine-tune. From feature set 5 on the card's entity also carries shape tokens derived from its spec (`cards.card_shape`, [features.md](features.md#card-shapes-and-hand-entities-set-5)), so a card that only uses existing ops starts out looking like the cards that work alike. A new spec field goes into `SHAPE_*_FIELDS` (and `card_shape`) or `NON_SHAPE_*_FIELDS`, under the same names in both `cards.py` and `native/src/cards.rs`; an unknown field fails to load in either engine with that hint, and `test_spec_field_sets_identical` / `test_card_shapes_identical` check the engines agree.

## Reference: cards.toml fields

All fields are optional unless marked. Unknown fields are an error in both engines.

| field | meaning |
|---|---|
| `name` (required) | the engine's name for the card (front face; accents folded: "Lorien Revealed") |
| `cost` | mana cost, `"{X}{G}{G}"`; omitted = no cost. Colours are derived from it. Phyrexian symbols (`"{R/P}"`) count as their colour; the card also gets the cast mode "phyrexian": the rest of the cost plus 2 life per symbol (Gut Shot) |
| `types` (required), `subtypes`, `supertypes` | space separated, e.g. `"Artifact Creature"`, `"Zombie Rat"`, `"Basic"` |
| `colors` | override, e.g. `"U"` for a colour indicator; `devoid = true` makes the card colourless |
| `power`, `toughness`, `keywords`, `ward` | `keywords = ["flying"]`; `ward = 2` is ward {2} |
| `text` | display only |
| `targets`, `effect` | an instant / sorcery's targets (target kinds below) and its ops |
| `modes` | modal spells: `[{ name, targets, effect }]`, each cast as its own option |
| `additional_sac` | `"artifact"` or `"artifact_or_creature"`: sacrifice as an additional cost |
| `additional_discard` | `true`: discard a card as an additional cost (the spell remembers whether it was a land) |
| `cost_reduction` | `instants_and_sorceries_in_graveyard`, `artifacts_you_control`, `cards_drawn_this_turn` |
| `flashback`, `escape` + `escape_exile`, `bestow` | alternative costs |
| `madness`, `plot` | madness cost (a discarded card goes to exile, and a trigger lets its owner cast it for this cost at any speed); plot cost (a sorcery-speed special action exiles it; cast free as a sorcery on a later turn, option "Cast X (plotted)") |
| `overload` + `overload_effect` | cast mode "overload": this cost, no targets, `overload_effect` instead of `effect` |
| `alternative_cost`, `flashback_cost` | `{ sacrifice = "mountain", n = N }`: sacrifice lands instead of paying mana (cast mode "alternative" from hand; flashback with only `flashback_cost` is free apart from the sacrifice). `alternative_cost = { reveal_hand = true }`: reveal your hand instead of paying, offered only with no other land card in hand; the hand becomes known to both players (Land Grant) |
| `prototype` + `prototype_face` | cast mode "prototype" for this cost; the permanent (and spell) uses the named `[[face]]` (Boulderbranch Golem) |
| `station` | `{ n = 7, keywords = ["flying"] }`: a Spacecraft, a creature with these keywords at `n` charge counters (an ability with `tap_other = "creature"` and op `station` adds them) |
| `additional_choose_creature` | `true`: like `additional_power`, but read by op `damage_target_from` with `from = "chosen_power"` (no card uses it; Monstrous Emergence uses `additional_power`) |
| `bargain` | `true`: cast mode "bargain", sacrificing an artifact, enchantment or token as an additional cost; an `etb` trigger with `condition = { bargained = true }` only triggers then (Troublemaker Ouphe) |
| `enters_tapped`, `etb_x_counters`, `back` | `back` names a `[[face]]` (transform) |
| `enters_tapped_unless_forests` | `N`: enters tapped unless its controller controls N other Forests (Gingerbread Cabin) |
| `omen` | `true`: the `back` face is an omen; cast mode "omen" casts that face from hand, and on resolution the card is shuffled into its owner's library (countered: graveyard). Sagu Wildling // Roost Seek |
| `collect_evidence` | `N`: cast mode "evidence", the normal cost plus exiling cards with total mana value N or more from your graveyard; an `etb` trigger with `condition = { cast_mode = "evidence" }` only triggers then (Vitu-Ghazi Inspector) |
| `additional_power` | `true`: as an additional cost choose a creature you control or reveal a creature card from hand; `damage_chosen_power` deals its power (Monstrous Emergence) |
| `abilities` | `[{ name (required), cost, tap, sac_self, sac_other, discard_self, discard_other, exile_self, x_target_mv, x_reveal, zone = "battlefield" \| "hand" \| "graveyard", sorcery_speed, mana = ["B", "R"], mana_amount, return_land, once_per_turn, tap_other, targets, effect }]`; `mana` makes it a mana ability; `mana_amount = "elves"`: one unit per Elf on the battlefield, the payment uses one and the rest floats (Priest of Titania); `mana_amount = { n, if_control = [subtypes] }`: `n` units while its controller controls each subtype (Urza's lands; extra units float); a mana ability with a `cost` (exactly `{1}`) is a **filter**, offered only while paying ("Activate X for U") (Barrels of Blasting Jelly); `return_land = "forest"`: return a Forest you control to its owner's hand as a cost (Quirion Ranger); `once_per_turn`; `tap_other = "creature"`: tap another untapped creature you control as a cost (station); `discard_other` / `exile_self`: discard a card / exile this permanent as a cost; `zone = "graveyard"`: activated from its owner's graveyard (with `exile_self`: Bramble Wurm); `x_target_mv = 2`: the cost has {X}{X}, X = the (single) target's mana value, only affordable targets are offered (Gorilla Shaman); `x_reveal = "red"`: choose X, then reveal X red cards from hand as a cost (Martyr of Ashes) |
| `triggers` | `[{ name (required), event, effect, condition, targets, up_to }]`; `targets` are chosen as the trigger goes on the stack, and a trigger without legal targets is removed (603.3d), events: `etb`, `leaves_battlefield`, `to_graveyard_from_battlefield`, `cast`, `you_sacrifice_another`, `your_upkeep`, `you_cast` (another spell its controller casts, from the battlefield), `third_draw` (its owner draws their third card in a turn, from the graveyard); condition: `{ sacrificed_subtype = "Eldrazi" }`, `{ spell = "noncreature" \| "instant_or_sorcery" }` (for `you_cast`), `{ equipped = true, spell = ... }` (for `you_cast`, only while this Equipment is attached: a trigger it grants the equipped creature; keys in alphabetical order, as the Rust TOML table iterates them), `{ bargained = true }`, `{ entered_untapped = true }`, `{ cast_mode = "evidence" }` (for `etb`); event `room`: a `[[dungeon]]` room (below) |
| `[[dungeon]]` | top-level: `name`, `rooms = [{ name, next, targets, effect }]`, first room on top; rooms are the dungeon's triggers. Only the Undercity exists (initiative: `take_initiative`, the upkeep venture, combat damage to the holder takes it) |
| `equipped_power`, `equipped_toughness` | Equipment: what the creature it is attached to gets (Black Mage's Rod: `equipped_power = 1`); `equipped_keywords` the keywords it gets (Whispersilk Cloak: `["unblockable", "shroud"]`). Equip is an ordinary ability (`cost`, `sorcery_speed = true`, `targets = ["creature_you_control"]`, op `attach_source_to_target`); an attached permanent without `bestow` is Equipment |

Target kinds: `creature`, `nonlegendary_creature`, `nonartifact_creature`, `creature_you_control`, `creature_you_dont_control`, `land`, `nonland_permanent`, `permanent`, `artifact`, `noncreature_artifact`, `blue_permanent`, `red_permanent`, `spell`, `blue_spell`, `red_spell`, `instant_spell`, `sorcery_spell`, `artifact_spell`, `player`, `opponent`, `player_with_creature`, `creature_of_target_player` (a creature controlled by the player chosen as the previous target: Searing Blaze), `another_creature` (a creature not already chosen as a target of the same spell: Cast into the Fire), `artifact_or_enchantment_spell`, `artifact_or_enchantment_you_dont_control`, `artifact_or_enchantment`, `noncreature_spell` (a bestowed spell counts: it is an Aura spell; Spell Pierce), `any`. A permanent with `shroud` is never a legal target (`Game.targetable`: candidates and the resolution re-check).

Sacrifice filters (`additional_sac`, `sac_other`, land-sacrifice costs): `artifact`, `artifact_or_creature`, `mountain`, `artifact_enchantment_or_token` (bargain), `land` (Crop Rotation).

Where the vocabulary lives, for when it needs to grow:

| vocabulary | Python | Rust |
|---|---|---|
| target kinds | `Game.target_candidates` / `_perm_matches` / `_spell_matches` | `TK` + `TK_NAMES` (cards.rs), `perm_matches` / `spell_matches` (state.rs) |
| trigger events | the `_emit_*` methods of `Game` | `Event` (cards.rs), `emit_*` (state.rs) |
| cost reductions, conditions | `COST_REDUCTIONS`, `_condition` (cards.py) | `CostRed`, `sacrificed_subtype` (cards.rs), `cost_reduction` / `emit_sacrifice` (state.rs) |
| keywords with rules meaning | `Game.has(...)` call sites (`shroud`: `targetable`; `unblockable`: `_can_block`) | `ENGINE_KEYWORDS` (cards.rs) or `has_known` (a keyword only some card defines: `shroud`, `unblockable`) and the same call sites |

## Reference: ops

Ops run in order. "The target" is target 0 of the spell or ability, re-checked by each op: if it became illegal, the op does nothing.

| op | parameters | effect |
|---|---|---|
| `draw` | `n`, `who = "you"` (default) or `"target_player"`, `n_cast_from_graveyard`, `each_controlling` | the controller draws `n` (or `n_cast_from_graveyard` when the spell was cast from a graveyard); `each_controlling`: instead each player who controls a permanent with that name (Bonder's Ornament) |
| `mill` | `who` = `you` \| `target_player`, `n` | mill `n` |
| `counter_target` | `if_color` | counter the targeted spell (only if it has colour `if_color`, e.g. `"U"`) |
| `counter_target_unless_paid` | `cost` | its controller may pay `cost`; otherwise counter it (Force Spike) |
| `destroy_target` | `if_color`, `mv_is_x` | destroy the targeted permanent (indestructible survives; only if it has colour `if_color` / its mana value equals X) |
| `bounce_target` | | return the targeted permanent to its owner's hand |
| `tap_target` | `skip_untap` | tap it; it skips that many of its controller's untap steps |
| `grant_target` | `keywords` | it gains the keywords until end of turn |
| `create_token` | `token`, `n`, `attach_source` | the controller creates `n` tokens (a `[[token]]` name); `attach_source`: then attach the source Equipment to the token (job select) |
| `attach_source_to_target` | | attach the source Equipment to the targeted creature (equip); nothing happens if the Equipment is itself a creature (CR 301.5c) |
| `gain_life` | `n`, `per_storm`, `n_from = "source_power"` | the controller gains `n` life (or life equal to the source's power: Boulderbranch Golem) |
| `lose_life` | `who` = `you` \| `opponent` \| `target_player` \| `target_controller`, `n` | that player loses `n` life |
| `counter_on_source` | | a +1/+1 counter on the source, if it is still on the battlefield |
| `damage_target` | `n`, `index`, `n_landfall` | the source (the spell itself, or the ability's source) deals `n` damage to target `index` (default 0; creature or player); `n_landfall` instead if a land entered under the controller's control this turn |
| `damage_target_controller` | `n` | the source deals `n` damage to the controller of the targeted permanent (Smash to Smithereens; list it before the op that removes the permanent) |
| `damage_each_opponent` | `n`, `if_discarded_nonland` | the source deals `n` damage to each opponent (only if the card discarded as the additional cost was not a land) |
| `damage_each_creature` | `n` or `x = true`, `without`, `except_subtype`, `whose = "opponent"` | the source deals `n` (or X) damage to each creature (without the keyword; not of the creature type `except_subtype`, changelings included: Fiery Cannonade; only the opponent's) |
| `discard` | `n` | the controller discards `n` cards of their choice |
| `return_to_battlefield` | `tapped` | graveyard trigger: this card returns to the battlefield, if it is still that object in the graveyard |
| `exile_graveyard` | | exile the target player's graveyard |
| `exile_all_graveyards` | | exile both graveyards |
| `exile_target` | | exile the targeted permanent |
| `exile_from_graveyards` | `n` | the controller exiles up to `n` cards from any graveyards, chosen one at a time on resolution (Faerie Macabre; the engine has no graveyard targets) |
| `search_library` | `supertype`, `type`, `subtypes_any`, `colorless`, `dest` = `battlefield` \| `hand`, `tapped`, `reveal`, `what` | search for a matching card (finding nothing is allowed), put it there, shuffle |
| `optional_payment` | `cost`, `prompt`, `then` | the controller may pay; if paid, run the `then` ops |
| `scry` | `n` | scry n: for n = 1 a top / bottom choice, for n > 1 one ORDER decision over (top in order, bottom in order) |
| `untap_target` | | untap the targeted permanent |
| `pump_target` | `n` or `count = "elves"` | +n/+n (or +X/+X, X = Elves on the battlefield on resolution) until end of turn |
| `counters_target` | `n` | `n` +1/+1 counters on the targeted permanent |
| `shuffle_target_into_library` | | its owner shuffles the targeted permanent into their library (Deglamer) |
| `dig` | `n`, `take` or `choose_type = [...]`, `rest` = `graveyard` \| `bottom` | top `n`: the cards of one type (chosen first with `choose_type`) to hand, the rest to the graveyard (all revealed: Winding Way) or the bottom in order (Lead the Stampede) |
| `may_exile_from_graveyard` | `type`, `then` | the controller may exile a card of `type` from their graveyard; if they do, run `then` (Masked Vandal) |
| `damage_chosen_power` | | the source deals damage equal to the power chosen as the `additional_power` cost to target 0 |
| `take_initiative` | | the controller takes the initiative and ventures into Undercity |
| `reveal_to_battlefield` | `n`, `type`, `counters`, `hexproof` | reveal the top `n`, put a `type` card from among them onto the battlefield with `counters` +1/+1 counters (hexproof until your next turn), shuffle (Throne of the Dead Three) |
| `surveil` | `n` (only 1) | surveil 1 |
| `look_top` | `n`, `type` or `permanent = true`, `colorless`, `rest` = `bottom` (default) \| `graveyard`, `what` | look at the top `n`, may put a matching card into your hand (revealed), the rest on the bottom (Ancient Stirrings); `rest = "graveyard"`: all `n` are revealed and the rest go to the graveyard (Malevolent Rumble) |
| `cascade` | | on a `cast` trigger: cascade below the spell's mana value (Maelstrom Colossus) |
| `station` | | charge counters on the source equal to the power of the creature tapped for `tap_other` |
| `damage_target_from` | `from` = `x` \| `chosen_power`, `index` | damage equal to X (Kaervek's Torch), or to the power of the creature chosen for `additional_choose_creature` |
| `return_random_from_graveyard` | `type` | a card of that type at random from your graveyard to your hand (Haunted Fengraf) |
| `return_cards_from_graveyards` | `types`, `n`, `each_type`, `whose` = `you` \| `any` | up to `n` cards of those types to their owners' hands, chosen on resolution (`each_type`: one per type: Call Damage Control) |
| `explore_target` | | the targeted creature explores |
| `shuffle_into_library` | | dies trigger: shuffle this card from the graveyard into its owner's library |
| `custom` | `fn` | `delver_reveal`, `brainstorm`, `ponder`, `deem_inferior`, `opponent_discards_else_draw`, `wildfire`, `duress`, `highway_robbery`, `relic_exile_one` |

## Writing an op so both engines agree

- **Python first.** The reference defines the behaviour; port it statement by statement, in the same order (side effects, zone moves, id allocation, log lines).
- **Decisions.** Every choice goes through `g.ask(player, kind, prompt, options)` (Rust: `self.ask(...)`) with the same options in the same order, the same labels (they may contain object ids such as `#123`), keys (stable, id-free tuples of str / int / None, the action vocabulary) and prompt. Deduplicate the way existing code does (`_dedupe_by_name`, `_dedupe_by_equiv`).
- **Targets.** Read the target with `g.target(item)` inside each op; do what the target needs before moving it (Vapor Snag).
- **Randomness** only through the game RNG (`g.shuffle`); never iterate a Python `set` to decide anything.
- **Live vs last-known objects.** `item.source` is a snapshot for activated abilities and dies triggers, the live card otherwise; use `g.live(item.source)` for "if it is still on the battlefield". Rust has the same distinction (`Src::Live` / `Src::Snap`).
- **Prove it.** Scenario tests on both engines, then the fuzzer with `--with-card`. If they diverge, the fuzzer prints the first differing value and writes a minimized reproducer: `python -m mtg_ml.difftest repro difftest-failure.json`.

## Adding a whole deck

1. Every card of the 75 through the card steps above.
2. `mtg_ml/engine/decks.py`: the list and its 15-card sideboard (`DECKS`, `SIDEBOARDS`); `mtg_ml/engine/sideboard_plans.toml`: one plan row against each existing deck and one for each existing deck against it ([sideboarding.md](sideboarding.md)).
3. Matchups are currently fixed to seat 0 = Jund Wildfire, seat 1 = Mono Blue Terror: `match.DECK_NAMES`, `bots.make_bot(seat)`, `agents.play_game`'s default decks and the RL deck conditioning (`seat:` feature) assume it. A third deck means generalizing those to (deck, seat) pairs first.
4. A scripted bot for the deck (`mtg_ml/bots/<deck>.py`, subclass `Bot`) so there is a baseline opponent.
5. Regenerate the golden digests on purpose (`python -m mtg_ml.trace record --games 210`), run the full test suite and a long differential fuzz (`--games 5000`).

## Bots and the feature vocabulary

- **Features need no registration.** State features and option keys are hashed strings (`encode.state_features`, `rl.features.option_tokens`), so a new card name or a new key shape gets buckets automatically, in both engines (`native/src/features.rs` hashes the same strings). Existing checkpoints still load, but they have never seen the new card: expect to fine-tune, and compare against the bots before and after.
- **New decision kinds** must be added to `PUBLIC_KINDS` if the opponent may see the choice (and to `is_public` in Rust).
- **Bots only play what they know.** `BlueBot.cast_score` and `JundBot.cast_score` return "never" for unknown names, and `card_value` / `CURVE` / `REMOVAL` / `THREATS` tables drive discards, search and mulligans. Add the card there, with a test in `tests/test_bots.py` if the decision matters.

Flashback may additionally specify `flashback_life` (default 0). The life cost
is checked when listing casts and paid with other casting costs, even if the
spell is subsequently countered. New pilot cards do not alter fixed decklists.
