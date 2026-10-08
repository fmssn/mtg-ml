# Play UI: playing against the models in the browser

`apps/play/` is the board client for live games against a checkpoint (or a scripted bot). It replaces clicking option labels in the replay viewer's side panel with what an MTG client does: drag a card to the battlefield, click a target, collect a whole attack or block and confirm it once. The server stays authoritative; the page knows no rules.

```bash
python -m mtg_ml.replay serve --models runs/          # then open http://127.0.0.1:8765/play/
python -m mtg_ml.replay serve --models runs/ --dev    # development: every checkpoint and matchup, scripted bots, scenarios
python -m mtg_ml.replay serve --models runs/ --max-games 20   # hold more games at once (default 8)
```

### What a normal server offers (`mtg_ml/play_config.toml`)

A normal server offers exactly what `mtg_ml/play_config.toml` (or `--play-config FILE`) lists, and refuses any other request:

- `player_decks`: the decks a player may pick. Today these are Jund Wildfire and Mono Blue Terror.
- `opponents`: each has an `id`, a `label`, a `model` (a checkpoint under `--models`), the `deck` it plays and `greedy`. Today there is one: **Delver**, the r4-control model playing Mono Blue Terror.

The new-game screen has two sections, **Your deck** and **Opponent**. An opponent shows only if the server has its checkpoint, and only for decks that have a matchup against it (`matchup_for`). The client sends `{deck, opponent}` and the server resolves the matchup and seat; model names, matchups, scenarios and the scripted bot are not accepted.

**The Delver mirror** is the matchup `blue_mirror` (Mono Blue Terror in both seats; the player takes seat 0). For games 2 and 3 it uses the plan table's mirror row: Gut Shot and Envelop come in, Sleep of the Dead and two Force Spike go out. `blue_mirror` is in `match.EXPLICIT_ONLY`, so code that enumerates the matchups (benchmarks, tools, the difftest token sweep) skips it. Training plays it only when a `--matchup` list names it. A deck with no matchup against any offered opponent would still be listed but greyed out.

`--dev` (`--scripted-bot` still works as an alias) gives a development server: the old form with every checkpoint, every matchup, seat choice, the scripted bots and the **dev scenarios** (`mtg_ml/live_dev.py`). Scenarios are games that start in a set position: a crowded Elves board, a trampling Hydra that must be chump-blocked, Tron lands that float mana, Cleansing Wildfire with a Drossforge Bridge. Players on a normal server never see any of this.

The replay viewer (`/`) and its own live mode are unchanged. A finished game is written to the replay directory as before; the game-over dialog links to it.

## Controls

| what | how |
|---|---|
| play a land, cast a spell | drag the glowing card above the hand, or double-click it. A single click opens a menu of its options (modes, flashback, cycling...). A spell then waits in the **paying step** (below) |
| pay for it | the card waits next to the stack with its cost as pips, nothing sent yet. Click lands (or other sources) to tap them: each fills a pip, a Bridge asks for its colour when it matters. Or **Auto pay** (button, **Space** or **Enter**). Once every pip is filled the spell is cast. **Esc**, right-click or Cancel puts the card back |
| cast and target in one go | drop a targeted spell onto the creature or player |
| activate an ability | click the permanent (blue glow) → menu; double-click if it has one ability |
| flashback, plotted cards | click the glowing Grave / Exile counter on your plate |
| choose a target | click a glowing (cyan) permanent, player plate or stack item; the arrow follows the pointer |
| tap a land for mana | click an untapped land (or other plain mana source) at priority. A source with several colours (Drossforge Bridge) asks which: click a pip or press **W/U/B/R/G/C**. It turns sideways and its mana shows as a pip in the pool on your plate; click it again to untap it |
| see what a spell will tap | hover or drag a castable card: your pool's pips say "used" and only the extra lands auto-pay will tap light up ("tap", "tap ×2" on a stack); anything it would sacrifice is marked red |
| cast at once, auto-paid | Settings → "Auto-pay without asking" (off by default) skips the paying step |
| the opponent's choices | Brainstorm, Ponder, scry, Delver's reveal, mulligans, targets, modes, X, blocks, sacrifices and their sideboarding show in a centred spotlight with the card for a few seconds (Settings: short 1.5 s, normal 2.8 s, long 4.5 s); a click or **Space** continues |
| attack | click creatures (or drag them forward), **A** all attack, then **Space** "Attack with N" |
| block | drag your creature onto an attacker (or click yours, then theirs); **Space** "Block (N)" |
| undo before confirming | right-click a selected attacker or assigned blocker (right-click an attacker drops all its blockers); **Esc** clears every selection. Nothing is undone once sent: no server-side undo |
| assign combat damage | the panel opens on a legal split; + and − move a point between recipients (the total stays the attacker's power), it says why a split is not allowed |
| pass priority | **Space**: the button says what happens ("Pass → Combat", "Resolve Lightning Bolt", "End turn") |
| pass until the opponent acts | **R**; **Esc** cancels. Space and Enter on a focused button or option press that control, never the global hotkeys |
| full control (never auto-pass) | **F** |
| hold priority after your next spell | **H** |
| stop yielding / yield to a source | the **auto-pass** button on an opponent's stack item |
| everything else | **O** or "All options": the plain list of the engine's options, always there |
| inspect a card | hover: a large floating preview on the side away from the pointer; right-click pins it, **Esc** unpins |
| game log | **L** or the Log button: a drawer over the board's right edge, closed by default |
| menu | ☰ Menu: New game, Concede, Bot played wrong…, Bug: engine / UI…, Decklists, Settings, Replay viewer |
| skip the opponent's replay | click the board or **Space** |
| report a bad bot play | the ⚑ after the opponent's last move on their plate (or on a log line), or Menu → Bot played wrong… |
| report a bug | Menu → Bug: engine / UI… |

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
| card choices | `{type: <kind>, name, uid or oid, tapped, verb}` when the card is in plain view (Highway Robbery's "Sacrifice Mountain" carries the Mountain's oid) |
| order | `{type: order, top: [names], bottom: [names]}` |
| mulligan | `{type: keep}` / `{type: mulligan}` |

Decisions about the player's own known cards (scry, surveil, Ponder, explore, Delver) carry `cards`: those names (the player's own knowledge only), with their card text in `cards` data. Two more fields for the player's own decisions: a `pay_mana` decision carries `auto`, the option the client takes when paying automatically (`live_proto.auto_pay_index`), and each cast / activate / plot ref at priority carries `taps`, the oids of the player's sources that this option plus automatic payment would tap. `taps` comes from simulating the option on a `Game.copy()` with the same `auto_pay_index` (other choices on the way take their first option), so the preview and the real payment cannot drift apart; `tests/test_live_proto.py` checks they are equal in played games on both engines.

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
6. **Mana: a deliberate paying step** (MTG Arena): a spell waits until you tap lands or press Auto pay; casting at once is an opt-in setting.
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

Your own spell on top of the stack passes (it resolves unless they respond, and a response stops you), unless you pressed **H** (hold priority once): then you keep priority with your spell on top, to respond to it yourself.

"Something you could answer" means a cast or an ability with timing value. Abilities without it (draw a card, search, cycling, scry, gaining life, making tokens: a Clue, Twisted Landscape, cycling lands) never turn the opponent's actions into stops; they still count for the step stops you set (the opponent's end step is where you crack them). Each opponent stack item has an **auto-pass** button: its source's spells and triggers no longer stop you this game (click again to undo; the pill lists the sources). The middle strip says how many priority passes were answered for you since your last decision.

Batched plans (an attack, blocks, a drop-to-target) answer only decisions of the turn and step they were made in; a plan left over from an earlier combat is dropped, never applied. "Pass until the opponent acts" ends at the first stop.

Input belongs to the decision it was made on: when a new decision differs in kind or turn from the last one, keys and the primary button are locked for 350 ms (the button shows it), and held-key repeats never answer anything. A Space pressed for one decision cannot land on the next after the engine auto-ran ahead.

Double-clicking a permanent whose ability sacrifices, discards or exiles as a cost opens its menu instead of firing it.

Drop and double-click mean "play it": they take only a land play or a normal cast. When a card has anything else (cycling, flashback, an alternative or additional cost, modes) or no plain play, the option menu opens instead.

### Decision kind → interaction

| kind | interaction |
|---|---|
| priority | drag / double-click / menu; primary button passes |
| target | glowing targets, arrow from the spell or ability; "No target" as the button when allowed |
| pay_mana | answered from the paying step's plan; an engine pay step the client did not plan (X spells) waits: click glowing sources, or "Auto-pay the rest" |
| declare_attacker / declare_blocker | batched, see above |
| assign_damage | overlay with −/+ per recipient, starts from the split that kills most blockers in order |
| choose_card, sacrifice, exile_from_graveyard, ... | card browser overlay (board peekable), text options ("Find nothing") below the cards; tapped permanents are tilted and say so, identical names get "permanent 2 of 3" |
| order (Ponder, scry 2+) | drag the cards (or use the arrows) into a top row and, for scry, a bottom row; "Confirm order" sends the matching engine option |
| scry 1, surveil, explore, Delver reveals | the card(s) shown large above the choices (number keys 1-9 pick a choice) |
| order_triggers | identical triggers are ordered for you; otherwise one button per order |
| yes_no, choose_mode, choose_x, order, order_triggers | option buttons above the primary button |
| mulligan | "Keep 7" as the primary button, "Mulligan to 6" beside it |

### Match flow

Games are best of three (`mtg_ml.match`): game 1 with the maindecks, games 2 and 3 sideboarded. The model takes the plan table's plan (`engine/sideboard_plans.toml`); between games the result card lets you take the same table's standard plan for your deck (its swaps are listed) or keep the maindeck, and if you lost you choose to play or draw (the model always plays first after a loss). After the match the button is a rematch: a new match with the same settings. The header shows "Game 2 of 3 · 0–1", each plate "on the play" / "on the draw".

**Concede** (header, with a confirm) ends a game as a loss; **New game** asks before leaving a running game. A game ends with a VICTORY / DEFEAT banner, then the result card: the reason in words, damage dealt and taken, cards played, flags, the match score, the full replay link.

The opening hand opens a mulligan screen: the seven cards large, the land count (amber at 0-1 or 6-7), play or draw, and Keep / Mulligan as two buttons. Click a deck name on a plate for both decklists (maindeck and sideboard).

Endpoints: `POST /api/live/<id>/concede`, `POST /api/live/<id>/next {plan, play}`, `GET /api/live/sideboard?matchup=&seat=`, `GET /api/live/decks?matchup=&seat=`. A full server never drops a game that is being played (touched in the last 10 minutes and not over): it refuses a new game with a message instead.

### Polish

- A short "Your turn" / "Opponent's turn" banner at each turn start (not at instant replay speed).
- Sounds (Web Audio, synthesized, no files): cast, land, attack, hit (louder for more damage), death, your turn, win, lose, your decision. Rate-limited so a big attack is one sound. The 🔊 header button mutes; Settings has a volume slider.
- Colour is never the only cue: playable cards glow white and sit a little higher, targets carry a ◎, attackers a ⚔, blockers a ⛨. Text colours meet WCAG AA on every panel (muted 6.6-7.7:1, dim 4.6-5.3:1).
- The stop bars have a ~22 px hit area around the small bar.
- Keyboard: Tab reaches playable cards, activatable or targetable permanents, plates and stack items (each with an aria label); Enter or Space clicks the focused one, Shift+Enter double-clicks (plays it); focusing a card shows it in the preview. Modals take focus; Esc closes menus and modals.
- More than four attackers draw one bundled arrow with the count; hover an attacker to see its own. Attackers lift only slightly, so they stay inside their row.

### Mana: tapping lands, the pool, paying

The engine has no "tap a land for mana" action outside a payment: its priority options hold mana abilities only for sources that sacrifice themselves (an Eldrazi Spawn), and the golden digests pin the engine's behaviour. So tapping a plain land at priority is a **client-side reservation**, presented the way Arena and MTGO show floating mana:

- Clicking an untapped land (any permanent whose card has a plain `{T}: Add` ability; the server sends its colours as `cards[name].mana`) puts `{oid, colour}` into the client's pool. The land shows sideways with its colour pip, and the pool on your plate shows the pip and when it empties ("empties end of main 1"). Clicking the land again takes it back. Sources that sacrifice themselves keep using the engine's own mana option (from their menu), and that mana floats in the engine's pool for real. A land that also has other abilities (Twisted Landscape) gets "Tap for mana" in its menu.
- Paying (`planPay`): the engine's floating mana goes first. Then your reserved mana: the same source, in the reserved colour if the cost can use it, else its other colour. Then the sources the tap preview showed for this cast (`S.payPlan`), then the usual auto-pay. The preview subtracts the pool: pool pips say "used" and only the extra lands light up.
- The pool empties at the end of the step, like real mana. The land was never tapped, so it simply stays untapped, and a toast says so. While the pool holds mana, auto-pass always stops. Pressing Space first asks "Pass? N mana empties", and the prompt warns that the pool empties when the step ends.

### The paying step (MTG Arena style)

Nothing is paid automatically. Playing a spell or an ability with a mana cost (drop, double-click or its menu) does not send anything: the card goes into a paying panel next to the stack, its hand slot fades, and its cost shows as pips (generic first, as printed).

- **What the client knows before sending.** For each cast or activation, the server simulates it on a copy (`live_proto.tap_preview`, the same run that makes the tap preview) and adds to the option's ref:
  - `cost`: the mana still to pay at the first pay step, e.g. `{1}{R}`;
  - `targets`: the refs of the first target decision before payment;
  - `before_pay`: other decisions on the way. A `choose_x` there means the cost depends on X; such spells skip the panel, and their engine pay steps wait for clicks instead.
- **Target first.** A targeted spell that was not dropped onto a target asks for it in the panel (the legal targets glow, your own permanents too); then the pips.
- **Filling pips.** The engine's floating mana and lands you tapped before casting fill pips first. A click on an untapped source fills its colour's pip, else a generic one. A Drossforge Bridge asks B or R only when the two would fill different pips. A tapped-for-this-payment land shows sideways with its pip; clicking it again takes it back. The lands Auto pay would use for the open pips glow.
- **Commit.** When the last pip is filled (or on Auto pay) the client sends the cast, answers the target decision with the chosen target, then answers each engine pay step from the plan (`planPay`): your picked sources (that source, that colour), then the preview's lands, then the auto-pay order. Decisions on the way that the plan does not cover (a sacrifice for Eviscerator's Insight) stop for you as usual.
- **Cancel** (Esc, right-click, the Cancel button) only clears the client's panel, so no engine undo is ever needed.
- **Opt-in:** Settings → "Auto-pay without asking" casts at once and auto-pays (the old behaviour).

### Post-game review

After a game ends, the result card's **Review game** (or Menu → "Review the finished game") opens the game omnisciently on the same board, read-only. It answers the question "did the bot misplay, or was it flooded or screwed?".

- **Server:** `GET /api/live/<id>/review?token=…` returns the omniscient replay: both hands, libraries, and the model's `policy` and `value` at each of its decisions. It is refused while the game runs and without the game's token. The token is a secret handed to the player only in the answer that starts the game (`live.token`, kept in localStorage per game id). Later views never carry it, so knowing a game id (it appears in public issues) does not open the review.
- **Timeline:** a strip under the header with turn markers and a dot per bot decision. Orange dots are surprising picks (an option under 15 % that the sampling policy chose). A red or green ring marks a big value swing (|Δ| ≥ 0.35 by its next decision). Click the strip to jump. Keys: ←/→ step, Shift+←/→ a turn, [ ] the previous or next bot decision, Home/End, Esc leaves.
- **Decision panel:** for each bot decision, its options ranked by probability (top five with bars, the chosen one marked), and its value before and after (the next decision's estimate, or the result at the end). Values are the network's own estimate of its return, −1 to +1, not a calibrated win probability. Your own decisions show what you chose.
- **Draw quality:** a strip under the timeline, one cell per turn and player aligned with the turns (lands and spells drawn, the land drop; red for a missed drop or 5+ lands in hand; the tooltip has the rest), and the full table behind the Draw quality button or Q. For each player it shows:
  - lands and spells drawn (cards new to the hand, so tutors count too);
  - lands and cards in hand at the start of the turn;
  - lands in play;
  - the land drop;
  - mana spent (pay steps) against mana sources.

  Chips in the strip summarise it: "mulligan to N", "flooded (turn T: 5+ lands in hand)", "screwed (missed N land drops)" (an own turn with no land in hand and no land played, under five lands), and the lands and spells drawn in total.
- **Reports from the review:** "Report: bot played wrong" (on a bot decision) and "Bug: engine / UI" (anywhere) open the same form. They post `{category, review_frame, token, …}`. Since the game is over, the issue holds everything:
  - the bot's hand and the full board;
  - its options with probabilities and its value;
  - the seed and the decision number;
  - `python -m mtg_ml.live_issues rebuild … --choices <the choices before this decision>`, which rebuilds the game up to that decision.
- **Replay viewer:** a link opens the same replay file in the viewer for the full debug view.

### The opponent's choices in the spotlight

The bot's moves still replay in order, but the choices that matter get a centred spotlight with the card's art that stays for `PREF.spotMs` (Settings: short 1.5 s, normal 2.8 s, long 4.5 s) or until a click or Space (which ends only the spotlight, not the whole replay):

- casts with a target ("Opponent casts Counterspell → your Lightning Bolt"), modes, X, Delver's reveal, Ponder's shuffle, blocks (all of a combat's blocks in one), sacrifices, a mulligan with its bottoms, and at the start of games 2 and 3 how many cards they sideboarded;
- their hidden choices as the server describes them (`live_proto.hidden_summary`): counts and places only, never a card the player cannot see. Examples: "Brainstorm: puts 2 cards from hand back on top", "Ponder: puts 3 cards back on top in a chosen order, does not shuffle", "Scry 1: puts a card on the bottom", "searches the library and finds a card". Consecutive choices of one spell become one spotlight.

A plain keep of seven, and casts without a target, keep the short spotlight.

The difference from real floating mana: no mana burn (none exists), and an unused reservation leaves the land untapped instead of tapped. Mana abilities with a cost (filters) are not reservable.

Verified with a Playwright run of the dev scenario "wildfire": it taps Drossforge Bridge for B, casts Cleansing Wildfire with it plus a Mountain onto the player's own tapped Bridge (own lands glow as possible targets while dragging and take the drop), the indestructible Bridge survives, the player fetches a basic in the card browser and draws; then a second Wildfire destroys the opponent's Island.

### Feedback: reports as GitHub issues, and the survey

There are two kinds of report, both visible but small:

- **Bot played wrong**: the ⚑ after the opponent's last move on their plate or on a log line, or Menu → Bot played wrong…. The form picks the bot play (recent public plays, newest first).
- **Bug: engine / UI**: Menu → Bug: engine / UI….

Each form asks what happened (one line), an optional note, and an optional name or nickname. The name is remembered locally, shown publicly, and is never a real name. The request is `POST /api/live/<id>/flag {category: "bot"|"bug", frame?, what, note, pseudonym}`. The submit button stays disabled while the request runs, and after filing the form shows the issue link.

The server files each report as one issue on `fmssn/mtg-ml` (`mtg_ml/live_issues.py`). It never sends a token to the browser.

- **How it files:** with `gh issue create` when `gh` is authenticated, else through the REST API with `MTG_PLAY_GITHUB_TOKEN`. For hosting, use a fine-grained personal access token with **Issues: write** on this repository only. `MTG_PLAY_GITHUB_REPO` overrides the repository, and `--no-issues` turns filing off.
- **Without GitHub access:** the report is saved with the game and the player sees "Saved with the game, not filed".
- **Labels:** `bot-play` for bot reports (created if missing) and `bug` for bugs.
- **Title:** `[bot-play] Jund Wildfire vs Mono Blue Terror, turn 3: <first line of what happened>` (`mulligan` instead of the turn before the game starts).
- **Rate limit:** at most 10 issues per game. Later reports are saved, not filed.
- **No hidden information while the game runs.** The repository is public, so the issue holds only what the flagging player could see:
  - the pseudonym, game id, engine, model, code commit and turn/step;
  - the flagged action as the player saw it;
  - the board from the player's own snapshot: their hand by name, the opponent's hand as a count (plus cards revealed to them), battlefields, graveyards and the stack;
  - the player's description and the recent visible log.
- **After the game ends:** the server adds a comment to each of the game's issues with the seed, matchup, seat, sideboard plan, starting player, engine, the omniscient replay file on the server, and a rebuild command: `python -m mtg_ml.live_issues rebuild --matchup … --seed … --seat … --game … --plan … --engine … --choices …`. The seed and choices reveal the bot's hand and library.

The result card keeps its short survey: how strong the bot played (1-5) and the hardest moment (`POST /api/live/<id>/survey`).

Both are saved in the replay file under `feedback` (format 1). It holds:

- `player`: the pseudonym, or "anonymous";
- `seat`, `seed`, `matchup`, `match_game`;
- `choices`: every decision's chosen index;
- `flags`: each has `category`, the player's `frame`, the replay's `replay_frame`, `action`, `kind`, `turn`, `step`, `what`, `note`, `at` and the `issue` url;
- `survey`.

Feedback given after the game ends rewrites the saved file. Tests use a fake filer (no real issues in CI).

### Help and information

A **?** in the header opens the controls sheet. A first game shows five short tips, once: the hand, the main button, your lands ("click a land to tap it for mana"), the phase rail and the Log button. The sheet can show them again. Each plate counts its open mana (untapped lands, and creatures and artifacts that tap for mana). Hand cards that cost less right now (Tolarian Terror, affinity) carry a "−N cost" badge: the server sends `reductions` ({uid: n}, from the engine's own cost reduction) with each of the player's decisions.

### Log and feed

The log is written in Magic's words, one line per action, from the parsed events, the decisions and the difference between two states: "Opponent casts Lightning Bolt → your Sagu Wildling", "Cryptic Serpent deals 6 damage to you", "Your life 18 → 12 (−6)", "Tolarian Terror dies", "You draw Ponder", "Mountain, Ponder are put into the opponent's graveyard from the library". Targets and X join the line of their spell; steps, mana, priority passes and engine tokens (p0/p1, choose_card, (normal), winner=1) never show. Triggers and resolutions are dimmed.

There is no side panel. The board takes the full width. The card preview floats on hover, the opponent's latest action shows on their plate with its ⚑, and the full log is a drawer (L), closed by default. The opponent's spells aimed at you or your permanents still get a spotlight with "→ your X" while they replay.

### Board layout

Each battlefield row (creatures in front, lands and other permanents behind) gets its share of the side's height and picks the largest card size that fits in one to three lines, up to a cap, measured after layout. Cards never overlap. Identical permanents stack with a ×N count (lands, tokens, and creatures outside combat declarations; while declaring attackers or blockers every creature is its own card). The opponent's latest action shows on their plate (with a ⚑ to report it); the log drawer has the rest.

The page is exactly the viewport (`100dvh`, every grid track `minmax(0, …)`) and re-fits on resize and zoom, down to 1280×720. While the stack is non-empty it has its own lane on the right (the rows make room); more than four items collapse to the top three plus "+N more" (an item that is a legal target is never hidden).

Feedback effects (life change, damage numbers, damage flash, a permanent entering) are recorded when a frame is first shown and kept for 0.45 to 1.4 s across re-renders, resuming their animation instead of restarting.

Combat plays out before the new state lands: each hit lunges its attacker toward the target, the target shakes and a red damage number floats off it; creatures that die turn grey, shrink and fly to their owner's graveyard counter; the result holds 300 ms (600 ms on a lethal blow) before input opens. Freshly declared attackers step forward. When the attack you are declaring (or facing) would bring a player to 0, the plate shows a pulsing LETHAL instead of the life after combat.

Combat damage and life changes are not in the engine's log; the live layer adds `combat: <attacker> deals N damage to <player or blocker>` and `life: pN old -> new` lines (parsed as `hit` and `life` events) to the player's log and the saved replay. Life lines are exact; combat lines use the powers before damage.

### Known gaps

- The tap preview assumes the first choice for anything decided while casting (targets, X, additional costs), so an X spell is previewed at its smallest X. A playtest saw empty previews for instants on the opponent's turn; not reproduced in 112 checked casts against the model on both engines (`tests/test_live_proto.py` covers an opponent-turn instant).
- Combat lines and hit animations use the powers before damage: first strike, pump effects in the damage step and non-default multi-block splits are approximate (life lines are exact).
- No server-side undo or cancel once a cast has started (targets, costs); the engine's sacrifice-before-mana order is a separate engine PR.
- Sideboarding offers the plan table's standard plan or the maindeck, not free card-by-card swaps; game 1 play/draw is a coin flip.
- "Quiet" abilities (no timing value) are recognised from their text (draw, search, cycling, scry, life, tokens); a per-card stop list is the yield button on stack items only.
- Card art comes from Scryfall on first sight; until it arrives a card shows a text face. When Scryfall is unreachable the client backs off for two minutes (text faces, no waiting before the opponent's spells).
- A game lives only in the server's memory: after a server restart the page says so and offers a new game.
- Phone layout is not done; the target is desktop 1280×720 and up.
- The review needs the game to still be in the server's memory (finished games are kept until idle eviction); afterwards the replay viewer still has the file.
- X spells skip the paying panel (their cost is known only after X): their engine pay steps wait for clicks, and they cannot be cancelled once X is chosen. Hybrid or phyrexian pips never appear in the panel (the engine resolves them into plain pips or a cast mode first).
- Tapping a plain land at priority is a client-side reservation (see *Mana*): an unused one leaves the land untapped when the step ends.
