# Play UI: playing against the models in the browser

`apps/play/` is the board client for live games against a checkpoint (or a scripted bot). It replaces clicking option labels in the replay viewer's side panel with what an MTG client does: drag a card to the battlefield, click a target, collect a whole attack or block and confirm it once. The server stays authoritative; the page knows no rules.

```bash
python -m mtg_ml.replay serve --models runs/          # then open http://127.0.0.1:8765/play/
python -m mtg_ml.replay serve --scripted-bot          # no checkpoint: play the decks' scripted bots (dev)
```

`--scripted-bot` marks a development server: the new-game dialog then also offers **dev scenarios** (`mtg_ml/live_dev.py`), games that start in a set position: a crowded Elves board, a trampling Hydra that must be chump-blocked (damage assignment), Tron lands that float mana. Players on a normal server never see them.

The replay viewer (`/`) and its own live mode are unchanged. A finished game is written to the replay directory as before; the game-over dialog links to it.

## Controls

| what | how |
|---|---|
| play a land, cast a spell | drag the glowing card above the hand, or double-click it. A single click opens a menu of its options (modes, flashback, cycling...). |
| cast and target in one go | drop a targeted spell onto the creature or player |
| activate an ability | click the permanent (blue glow) → menu; double-click if it has one ability |
| flashback, plotted cards | click the glowing Grave / Exile counter on your plate |
| choose a target | click a glowing (cyan) permanent, player plate or stack item; the arrow follows the pointer |
| see what a spell will tap | hover or drag a castable card: the lands and sources auto-pay will use light up ("tap", "tap ×2" on a stack), and anything it would sacrifice is marked red |
| attack | click creatures (or drag them forward), **A** all attack, then **Space** "Attack with N" |
| block | drag your creature onto an attacker (or click yours, then theirs); **Space** "Block (N)" |
| undo before confirming | right-click a selected attacker or assigned blocker (right-click an attacker drops all its blockers); **Esc** clears every selection. Nothing is undone once sent: no server-side undo |
| assign combat damage | the panel opens on a legal split; + and − move a point between recipients (the total stays the attacker's power), it says why a split is not allowed |
| pass priority | **Space**: the button says what happens ("Pass → Combat", "Resolve Lightning Bolt", "End turn") |
| pass until the opponent acts | **R**; **Esc** cancels. Space and Enter on a focused button or option press that control, never the global hotkeys |
| full control (never auto-pass) | **F** |
| everything else | **O** or "All options": the plain list of the engine's options, always there |
| inspect a card | hover (preview on the right); right-click pins it |
| skip the opponent's replay | click the board or **Space** |
| flag an opponent move | the ⚑ on its chip or log line (stored locally for now; flag + describe comes with hosted play) |

## Protocol (`mtg_ml/live_proto.py`)

The live server's frames keep the replay format, with two additions.

**`decision.refs`**, one per option, derived from `Option.key` and `Option.value` in the live layer (no engine change; both engines expose card objects in values):

| decision | ref |
|---|---|
| priority | `{type: pass}`, `{type: play_land / cast / plot, name, uid, zone}`, cast also `from`, `mode`, `spell_mode`; `{type: activate / mana, name, oid or uid, ability}` |
| target | `{type: target, oid}` / `{player}` / `{sid}` / `{none: true}` |
| pay_mana | `{type: pay, pool}` or `{type: pay, via: source / filter, oid, name, color}` |
| declare_attacker | `{type: attack, attacker, name}` or `{done: true}` |
| declare_blocker | `{type: block, blocker, attacker, attacker_name}` or `{blocker, none: true}` |
| assign_damage | `{type: damage, split, to: [blocker oids], player}` |
| choose_x | `{type: x, x}` |
| card choices | `{type: <kind>, name, uid}` when the card is in plain view |
| mulligan | `{type: keep}` / `{type: mulligan}` |

Two more fields for the player's own decisions: a `pay_mana` decision carries `auto`, the option the client takes when paying automatically (`live_proto.auto_pay_index`), and each cast / activate / plot ref at priority carries `taps`, the oids of the player's sources that this option plus automatic payment would tap. `taps` comes from simulating the option on a `Game.copy()` with the same `auto_pay_index` (other choices on the way take their first option), so the preview and the real payment cannot drift apart; `tests/test_live_proto.py` checks they are equal in played games on both engines.

**`actions`**: the frame's visible log lines parsed into events (`turn`, `step`, `play`, `cast`, `activate`, `trigger`, `resolve`, `enter`, `leave`, `dies`, `attack` with oids, `block` with pairs, `discard`, `sacrifice`, `mulligan`, `countered`, `game_over`, else `note`).

**No leaks.** Every uid, oid and stack id in a ref is checked against the snapshot the player gets in the same frame; anything not in it (a library card, the model's face-down hand) is dropped and only the name the label already shows stays. The model's frames carry one ref for the chosen option of public kinds (`rl.features.PUBLIC_KINDS`) and `{type: hidden}` otherwise. `actions` are parsed from lines `replay.visible_events` already let through. `tests/test_live_proto.py` plays games on both engines and checks this for every frame.

The engine's per-step decisions stay as they are. The client batches: an attack is a set of creatures, sent as the engine's one-at-a-time `declare_attacker` decisions; blocks are a blocker → attacker map, sent per blocker; mana is paid with the server's `auto` choice (floating mana first, then sources that keep the most colours open; basics before other lands before artifacts before creatures; sacrifices and filters last). Equivalent permanents are deduplicated by the engine, so an attack plan picks the exact creature when offered and an interchangeable one otherwise.

## Design

The research brief behind this (2026-10-08) looked at endstep.cc (the closest analogue: Forge headless on a server, a thin browser view), MTG Arena, MTGO, Forge/XMage, Cockatrice/untap.in, Hearthstone, Runeterra and Slay the Spire. Where a number below comes from a source it is linked; px/ms values are our recommendations, not measurements.

### Principles, in priority order

1. **The prompt is always obvious and the primary button names the consequence.** One button, same place (bottom right), "Pass → Combat", "Resolve Lightning Bolt", "Attack with 3", "No blocks", "Keep 7". Never a bare OK: MTGO's generic OK made people pass through steps; it was renamed to name the next step in Nov 2025 ([MTGO QoL](https://www.mtgo.com/news/qol-nov2025)). Space presses it; it pulses only when the human is asked.
2. **Priority automation is the biggest lever.** Never ask a question with one sensible answer, never skip a moment the player wanted, and show what the bot did before asking. endstep's "pass until action" stops on any opponent cast, activation, trigger or attack ([changelog](https://endstep.cc/changelog)); Arena's auto-pass is hated for skipping windows you wanted ([mtgazone](https://mtgazone.com/playing-a-match/), [smart priority](https://magic.wizards.com/en/news/mtg-arena/announcements-october-27-2025)). Forge had End Turn carry into the next game, so modes reset at game boundaries ([Forge forum](https://slightlymagic.net/forum/viewtopic.php?p=157887)).
3. **The bot is instant, its moves must not be.** Replay them as a queue: spells fly into a spotlight and hold ~1 s, lands and attacks get a beat, the whole turn stays within a few seconds, a click skips. Input for a decision only after the events before it played (Arena's "server ahead of the animations" is the anti-pattern). Hearthstone's lesson: animation must never cost the player time ([Blizzard forum](https://us.forums.blizzard.com/en/hearthstone/t/abuse-of-animations/108732)).
4. **Direct manipulation.** Hover lift ~120-150 ms, preview with full Oracle text, playable glow (never grey out), drag threshold 6-8 px, valid drop zones glow, snap back on an invalid drop, double-click as the alternative (Arena/MTGO convention), popover menus anchored to the card for multi-option cards.
5. **Targeting arrows** end at the nearest card edge (endstep), legal targets glow, players are targeted through their plate, stack items show arrows to their targets.
6. **Mana: auto-pay by default**, manual on demand (settings).
7. **Combat is batched in the client and streamed to the server**: toggle attackers, "All attack"; drag blockers onto attackers with lines; a rough outcome preview (skulls, "→ 11 after combat") in the spirit of Runeterra and Slay the Spire's enemy intents ([StS intents](https://sts2.untapped.gg/guides/how-to-read-enemy-intent)); damage assignment starts from a legal split, editable (endstep removed its confusing Auto-Assign).
8. **Confirm guards that stay armed**: passing with floating mana, "No attacks" while creatures could attack (endstep: the guard now stays armed until used or cancelled).
9. **Board readability**: lands grouped with a count, big P/T badges with damage shown, tapped = rotated, all zone counts visible, a stack column, life with animated change and red at ≤ 5.
10. **Avoid**: bare OK buttons, hidden auto-pass state, undo across priority, animations that block input, dimming the opponent's board while declaring, auto-targeting by default, modals for routine choices, asking for each mana payment.

Other sources: [endstep about](https://endstep.cc/about), [Arena hotkeys](https://draftsim.com/mtg-arena-keyboard-shortcuts/), [Arena patch notes](https://devtrackers.gg/magic-arena/p/657c7027-apr-25-0-14-00-00-patch-notes), [MTGO tips](https://www.mtgo.com/getting-started/getting-started-tips-tricks), [MTGO stops thread](https://steamcommunity.com/app/316010/discussions/0/535152511343701059), [SCG on the MTGO clock](https://articles.starcitygames.com/articles/conquering-the-clock-on-mtgo-and-more-pro-tour-tales/), [Cockatrice vs untap](https://tappedout.net/mtg-forum/online-magic/cockatrice-vs-untapin), [Hearthstone screenshake](https://makegamessa.com/discussion/comment/30301), [card animation timings](https://playbooks.com/skills/dylantarre/animation-principles/cards-containers).

### Auto-pass rules (as built)

A priority decision is answered "pass" by the client unless:

- full control is on (**F**);
- some option is a real play: anything but pass and side-effect mana abilities (Treasure). Only then can any of the rest stop it;
- the opponent's spell or ability is on top of the stack, or the opponent cast, activated, attacked or blocked in the current step since your last decision;
- it is the first priority after blockers were declared (the combat-trick window): always on the opponent's turn, on yours when something blocked, whatever the stop settings say;
- the step has a stop (phase rail; defaults: your main 1 and main 2, the opponent's declare-blockers and end steps), unless "pass until the opponent acts" is running.

Your own spell on top of the stack passes (it resolves unless they respond, and a response stops you).

Batched plans (an attack, blocks, a drop-to-target) answer only decisions of the turn and step they were made in; a plan left over from an earlier combat is dropped, never applied. "Pass until the opponent acts" ends at the first stop.

Input belongs to the decision it was made on: when a new decision differs in kind or turn from the last one, keys and the primary button are locked for 350 ms (the button shows it), and held-key repeats never answer anything. A Space pressed for one decision cannot land on the next after the engine auto-ran ahead.

Double-clicking a permanent whose ability sacrifices, discards or exiles as a cost opens its menu instead of firing it.

Drop and double-click mean "play it": they take only a land play or a normal cast. When a card has anything else (cycling, flashback, an alternative or additional cost, modes) or no plain play, the option menu opens instead.

### Decision kind → interaction

| kind | interaction |
|---|---|
| priority | drag / double-click / menu; primary button passes |
| target | glowing targets, arrow from the spell or ability; "No target" as the button when allowed |
| pay_mana | auto-pay; manual mode: click glowing sources, button "Auto-pay" |
| declare_attacker / declare_blocker | batched, see above |
| assign_damage | overlay with −/+ per recipient, starts from the split that kills most blockers in order |
| choose_card, sacrifice, exile_from_graveyard, ... | card browser overlay (board peekable), text options below the cards |
| yes_no, choose_mode, choose_x, order, order_triggers | option buttons above the primary button |
| mulligan | "Keep 7" as the primary button, "Mulligan to 6" beside it |

### Log and feed

The log is written in Magic's words, one line per action, from the parsed events, the decisions and the difference between two states: "Opponent casts Lightning Bolt → your Sagu Wildling", "Cryptic Serpent deals 6 damage to you", "Your life 18 → 12 (−6)", "Tolarian Terror dies", "You draw Ponder", "Mountain, Ponder are put into the opponent's graveyard from the library". Targets and X join the line of their spell; steps, mana, priority passes and engine tokens (p0/p1, choose_card, (normal), winner=1) never show. Triggers and resolutions are dimmed.

The side-panel feed keeps the opponent's recent actions and everything that hit you (damage, deaths of your creatures, life lost) in order, across your own moves; entries older than the last turn fade. The opponent's spells aimed at you or your permanents get a longer spotlight with "→ your X".

### Board layout

Each battlefield row (creatures in front, lands and other permanents behind) gets its share of the side's height and picks the largest card size that fits in one to three lines, up to a cap, measured after layout. Cards never overlap. Identical permanents stack with a ×N count (lands, tokens, and creatures outside combat declarations; while declaring attackers or blockers every creature is its own card). Opponent actions since your last move are listed in full in the side panel, each with a ⚑; the latest also shows on the opponent's plate.

The page is exactly the viewport (`100dvh`, every grid track `minmax(0, …)`) and re-fits on resize and zoom, down to 1280×720. While the stack is non-empty it has its own lane on the right (the rows make room); more than four items collapse to the top three plus "+N more" (an item that is a legal target is never hidden).

Feedback effects (life change, damage numbers, damage flash, a permanent entering) are recorded when a frame is first shown and kept for 0.45 to 1.4 s across re-renders, resuming their animation instead of restarting.

Combat plays out before the new state lands: each hit lunges its attacker toward the target, the target shakes and a red damage number floats off it; creatures that die turn grey, shrink and fly to their owner's graveyard counter; the result holds 300 ms (600 ms on a lethal blow) before input opens. Freshly declared attackers step forward. When the attack you are declaring (or facing) would bring a player to 0, the plate shows a pulsing LETHAL instead of the life after combat.

Combat damage and life changes are not in the engine's log; the live layer adds `combat: <attacker> deals N damage to <player or blocker>` and `life: pN old -> new` lines (parsed as `hit` and `life` events) to the player's log and the saved replay. Life lines are exact; combat lines use the powers before damage.

### Known gaps

- The tap preview assumes the first choice for anything decided while casting (targets, X, additional costs), so an X spell is previewed at its smallest X. A playtest saw empty previews for instants on the opponent's turn; not reproduced in 112 checked casts against the model on both engines (`tests/test_live_proto.py` covers an opponent-turn instant).
- Combat lines are approximate for first strike, pump effects in the damage step and multi-blocks (life lines are exact).
- No server-side undo, no "always yes/no", no smart stops per card, no manual-mana undo.
- The combat preview ignores first strike and tricks; it is a hint.
- Motion covers zone changes on the board and hand (FLIP); draws, deaths and damage get simple effects, not full flights to the graveyard.
- A board of 25+ different permanents on one side gets small cards (three lines); hover still shows the full card on the right.
- Card art comes from Scryfall on first sight; until it arrives a card shows a text face. When Scryfall is unreachable the client backs off for two minutes (text faces, no waiting before the opponent's spells).
- A game lives only in the server's memory: after a server restart the page says so and offers a new game.
- Phone layout is not done; the target is desktop 1280×800 and up.
