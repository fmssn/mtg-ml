//! Hashed network inputs, bit-identical to mtg_ml/rl/features.py and
//! mtg_ml/encode.py (zlib.crc32 of the same strings), computed without
//! building the strings as Python objects. `py.rs::state_features` keeps the
//! string form for the differential tests.

use std::fmt::Write;

use crc32fast::Hasher;

use crate::cards::{db, type_names, CardDef, Op, SacFilter, TYPE_NAMES};
use crate::mana::{bit, ManaCost, Remaining};
use crate::state::*;

/// Kinds whose chosen option is public when the opponent makes it
/// (rl/features.py `PUBLIC_KINDS`).
pub fn is_public(kind: Kind) -> bool {
    !matches!(kind, Kind::ChooseCard | Kind::Order | Kind::ChooseMode)
}

fn rel(p: u8, viewer: u8) -> &'static str {
    if p == viewer {
        "self"
    } else {
        "opponent"
    }
}

fn bucket(n: i64) -> usize {
    const EDGES: [i64; 8] = [0, 1, 2, 3, 5, 8, 13, 20];
    EDGES.iter().rposition(|e| n >= *e).unwrap_or(0)
}

/// encode.py `COUNT_CAP` and thermometer steps.
const COUNT_CAP: u32 = 8;
const LIFE_STEPS: &[i64] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25];
const TURN_STEPS: &[i64] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20];
const POWER_STEPS: &[i64] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20];
const COUNT_STEPS: &[i64] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const GY_STEPS: &[i64] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15];

/// Receives state features: `direct` ones as they are, `raw` per-object ones
/// counted (encode.py `_counted`: each distinct feature once, then `#2`..`#n`).
pub trait FeatureOut {
    fn direct(&mut self, args: std::fmt::Arguments);
    fn raw(&mut self, args: std::fmt::Arguments);
}

macro_rules! direct {
    ($o:expr, $($arg:tt)*) => { $o.direct(format_args!($($arg)*)) };
}
macro_rules! raw {
    ($o:expr, $($arg:tt)*) => { $o.raw(format_args!($($arg)*)) };
}

fn thermo<O: FeatureOut>(o: &mut O, name: &str, n: i64, steps: &[i64]) {
    for &k in steps {
        if n >= k {
            direct!(o, "{name}>={k}");
        }
    }
}

/// encode.py `FEATURES` / `FEATURE_VERSIONS`: feature-set versions (1: up to
/// 2026-10-06; 2: + readiness, known positions, skip_untap, stack targets, X,
/// option previews; 3: + `opp:deck:`; 4: + combat relations, incoming
/// damage, choose_x previews and pointer, mana colours; 5: + card shapes,
/// hand entities; 6: + simulated option previews, `sim.rs`).
pub const FEATURES: u8 = 7;
pub const FEATURE_VERSIONS: &[u8] = &[1, 2, 3, 4, 5, 6, 7];

pub fn check_features(features: u8) -> Result<u8, String> {
    if FEATURE_VERSIONS.contains(&features) {
        Ok(features)
    } else {
        Err(format!("unknown feature-set version {features} (known: {FEATURE_VERSIONS:?})"))
    }
}

/// `encode.state_features(game, viewer, features)`: the direct features in order, plus
/// the per-object ones, which the sink counts and appends in `finish`.
pub fn state_features_into<O: FeatureOut>(st: &State, viewer: u8, features: u8, o: &mut O) {
    let opp = 1 - viewer;
    direct!(o, "step:{}", st.step_name);
    direct!(o, "active:{}", rel(st.active, viewer));
    direct!(o, "postboard:{}", if st.match_game > 1 { "True" } else { "False" });
    if let Some(deck) = st.args.deck_names[viewer as usize].as_ref().filter(|_| (2..7).contains(&features)) {
        direct!(o, "self:deck:{deck}");
    }
    if let Some(deck) = st.args.deck_names[opp as usize].as_ref().filter(|_| (3..7).contains(&features)) {
        direct!(o, "opp:deck:{deck}");
    }
    if features >= 7 {
        for (zone, cards) in [("registered_main", &st.args.registered_main[viewer as usize]), ("registered_sideboard", &st.args.registered_sideboards[viewer as usize]), ("current_main", &st.args.decks[viewer as usize])] {
            let mut counts = std::collections::BTreeMap::new();
            for name in cards { *counts.entry(name).or_insert(0usize) += 1; }
            for (name, count) in counts { for k in 1..=count { direct!(o, "self:list:{zone}:{name}#{k}"); } }
        }
        if let Some(a) = &st.damage_allocation {
            direct!(o, "damage:recipient:{}", a.recipient);
            direct!(o, "damage:remaining:{}", a.remaining);
            direct!(o, "damage:player:{}", a.player_damage);
            for (i, n) in a.assigned.iter().enumerate() { direct!(o, "damage:assigned:{i}:{n}"); }
            for (i, n) in a.lethal.iter().enumerate() { direct!(o, "damage:lethal:{i}:{n}"); }
        }
    }
    thermo(o, "turn", st.turn as i64, TURN_STEPS);
    if st.active == viewer {
        direct!(o, "land_played:{}", if st.lands_played > 0 { "True" } else { "False" });
    }
    for (side, p) in [("self", viewer), ("opponent", opp)] {
        let pl = &st.players[p as usize];
        thermo(o, &format!("{side}:life"), pl.life as i64, LIFE_STEPS);
        direct!(o, "{side}:library:{}", bucket(pl.library.len() as i64));
        direct!(o, "{side}:mulligans:{}", st.mulligans_taken[p as usize]);
        thermo(o, &format!("{side}:gy_count"), pl.graveyard.len() as i64, GY_STEPS);
        thermo(o, &format!("{side}:exile_count"), pl.exile.len() as i64, COUNT_STEPS);
        for &c in &pl.graveyard {
            raw!(o, "{side}:gy:{}", st.c(c).name());
        }
        for &c in &pl.exile {
            // set 1 predates the plotted mark
            if features >= 2 {
                raw!(o, "{side}:exile:{}", st.c(c).exiled_name());
            } else {
                raw!(o, "{side}:exile:{}", st.c(c).name());
            }
        }
        for (c, n) in &pl.pool {
            direct!(o, "{side}:pool:{}:{n}", color_str(*c));
        }
        for &c in pl.library.iter().filter(|&&c| st.c(c).known_to & pbit(viewer) != 0).take(3) {
            raw!(o, "{side}:known_library:{}", st.c(c).name());
        }
    }
    let me = &st.players[viewer as usize];
    thermo(o, "self:hand_count", me.hand.len() as i64, COUNT_STEPS);
    for &c in &me.hand {
        raw!(o, "self:hand:{}", st.c(c).name());
    }
    let them = &st.players[opp as usize];
    thermo(o, "opponent:hand_count", them.hand.len() as i64, COUNT_STEPS);
    for &c in them.hand.iter().filter(|&&c| st.c(c).known_to & pbit(viewer) != 0) {
        raw!(o, "opponent:hand_known:{}", st.c(c).name());
    }
    let mut power = [0i64; 2]; // [self, opponent]
    for &ci in &st.battlefield {
        let c = st.c(ci);
        let side = rel(c.controller, viewer);
        let types = st.types(c);
        for (i, t) in TYPE_NAMES.iter().enumerate() {
            if types & (1 << i) != 0 {
                raw!(o, "{side}:bf_type:{t}");
                if !c.tapped {
                    raw!(o, "{side}:bf_type:{t}:untapped");
                }
            }
        }
        if st.is_creature(c) {
            power[(c.controller != viewer) as usize] += st.power(c).max(0) as i64;
        }
    }
    thermo(o, "self:power", power[0], POWER_STEPS);
    thermo(o, "opponent:power", power[1], POWER_STEPS);
    thermo(o, "stack_count", st.stack.len() as i64, COUNT_STEPS);
    if features < 2 {
        return;
    }
    board_features(st, viewer, o);
    for (side, p) in [("self", viewer), ("opponent", opp)] {
        let lib = &st.players[p as usize].library;
        for (pos, &c) in lib.iter().enumerate() {
            if st.c(c).known_to & pbit(viewer) == 0 {
                continue;
            }
            if pos < KNOWN_POS_CAP {
                direct!(o, "{side}:known_library:{pos}:{}", st.c(c).name());
            } else if pos == lib.len() - 1 {
                direct!(o, "{side}:known_library:bottom:{}", st.c(c).name());
            }
        }
    }
    if features >= 4 {
        combat_features(st, viewer, o);
        colour_features(st, viewer, o);
    }
}

/// `thermo` for `{side}:{name}` without allocating the name.
fn side_thermo<O: FeatureOut>(o: &mut O, side: &str, name: &str, n: i64, steps: &[i64]) {
    for &k in steps {
        if n >= k {
            direct!(o, "{side}:{name}>={k}");
        }
    }
}

/// encode.py `KNOWN_POS_CAP`.
const KNOWN_POS_CAP: usize = 8;

/// encode.py `_ready`: could this creature attack in its controller's
/// current (if active) or next turn?
fn ready(c: &Card, active: bool) -> bool {
    if active {
        !c.tapped && !c.sick
    } else {
        !(c.tapped && c.skip_untap > 0)
    }
}

/// encode.py `_board_features`: ready power, the part of it the defender's
/// untapped creatures cannot block, potential blockers, lethal flags and
/// untapped mana sources, per side.
fn board_features<O: FeatureOut>(st: &State, viewer: u8, o: &mut O) {
    let d = db();
    let (fly, reach) = (d.kw("flying"), d.kw("reach"));
    let unblockable = d.keyword_names.iter().position(|k| k == "unblockable").map_or(0, |i| 1u32 << i);
    // (controller, untapped, ready-if-its-controller-is-active, ready otherwise, power, keywords)
    let creatures: Vec<(u8, bool, bool, bool, i64, u32)> = st
        .battlefield
        .iter()
        .map(|&ci| st.c(ci))
        .filter(|c| st.is_creature(c))
        .map(|c| (c.controller, !c.tapped, ready(c, true), ready(c, false), st.power(c).max(0) as i64, st.keywords(c)))
        .collect();
    for (side, p) in [("self", viewer), ("opponent", 1 - viewer)] {
        let active = st.active == p;
        let blockers: Vec<u32> = creatures.iter().filter(|c| c.0 != p && c.1).map(|c| c.5).collect();
        let (mut ready_power, mut evasive) = (0i64, 0i64);
        for c in creatures.iter().filter(|c| c.0 == p && if active { c.2 } else { c.3 }) {
            ready_power += c.4;
            let blockable = c.5 & unblockable == 0 && blockers.iter().any(|&b| c.5 & fly == 0 || b & (fly | reach) != 0);
            if !blockable {
                evasive += c.4;
            }
        }
        let life = st.players[(1 - p) as usize].life as i64;
        side_thermo(o, side, "ready_power", ready_power, POWER_STEPS);
        side_thermo(o, side, "ready_evasive_power", evasive, POWER_STEPS);
        side_thermo(o, side, "potential_blockers", creatures.iter().filter(|c| c.0 == p && c.1).count() as i64, COUNT_STEPS);
        if ready_power > 0 && ready_power >= life {
            direct!(o, "{side}:lethal_on_board");
        }
        if evasive > 0 && evasive >= life {
            direct!(o, "{side}:evasive_lethal_on_board");
        }
        side_thermo(o, side, "untapped_mana", st.mana_sources(p, &[]).len() as i64, COUNT_STEPS);
    }
}

/// encode.py `_attackers`: attacking creatures still on the battlefield, in declaration order.
fn attackers(st: &State) -> impl Iterator<Item = &Card> + '_ {
    st.attackers.iter().filter_map(move |&a| st.perm(a)).map(move |ci| st.c(ci))
}

/// encode.py `_blockers_of`: creatures still on the battlefield blocking attacker `aoid`.
fn blockers_of(st: &State, aoid: u32) -> impl Iterator<Item = &Card> + '_ {
    st.blocks.iter().filter(move |(_, a)| *a == aoid).filter_map(move |(b, _)| st.perm(*b)).map(move |ci| st.c(ci))
}

/// encode.py `_unblocked_power`: power of the attackers not blocked, leaving out `skip`.
fn unblocked_power(st: &State, skip: Option<u32>) -> i64 {
    attackers(st).filter(|c| !st.blocked.contains(&c.oid) && Some(c.oid) != skip).map(|c| st.power(c).max(0) as i64).sum()
}

/// encode.py `_combat_features` (set 4).
fn combat_features<O: FeatureOut>(st: &State, viewer: u8, o: &mut O) {
    if attackers(st).next().is_none() {
        return;
    }
    let side = rel(st.active, viewer);
    let total: i64 = attackers(st).map(|c| st.power(c).max(0) as i64).sum();
    let unblocked = unblocked_power(st, None);
    let life = st.players[(1 - st.active) as usize].life as i64;
    side_thermo(o, side, "attacking_power", total, POWER_STEPS);
    side_thermo(o, side, "unblocked_power", unblocked, POWER_STEPS);
    if unblocked > 0 && unblocked >= life {
        direct!(o, "{side}:incoming_lethal");
    }
    side_thermo(o, rel(1 - st.active, viewer), "life_after_unblocked", life - unblocked, LIFE_STEPS);
}

/// encode.py `_mana_colors`: bitmask of the colours (WUBRG) the card's mana ability makes.
pub(crate) fn mana_colors(c: &Card) -> u8 {
    let colors = PREVIEW_COLORS.iter().fold(0u8, |m, &col| m | bit(col));
    c.face().abilities.iter().find_map(|a| a.mana.as_ref()).map_or(0, |v| v.iter().fold(0u8, |m, &col| m | bit(col)) & colors)
}

/// encode.py `_colour_counts`: (permanents of `viewer` making each of
/// WUBRG, bitmask of the colours in the costs of `viewer`'s hand cards).
fn colour_counts(st: &State, viewer: u8) -> ([i64; 5], u8) {
    let mut sources = [0i64; 5];
    for &ci in &st.battlefield {
        let c = st.c(ci);
        if c.controller == viewer {
            let m = mana_colors(c);
            for (i, &col) in PREVIEW_COLORS.iter().enumerate() {
                if m & bit(col) != 0 {
                    sources[i] += 1;
                }
            }
        }
    }
    let mut needs = 0u8;
    for &ci in &st.players[viewer as usize].hand {
        for &(col, n) in &st.c(ci).face().cost.colored {
            if n > 0 && PREVIEW_COLORS.contains(&col) {
                needs |= bit(col);
            }
        }
    }
    (sources, needs)
}

/// encode.py `_colour_features` (set 4).
fn colour_features<O: FeatureOut>(st: &State, viewer: u8, o: &mut O) {
    let (sources, needs) = colour_counts(st, viewer);
    for (i, &col) in PREVIEW_COLORS.iter().enumerate() {
        for &k in COUNT_STEPS {
            if sources[i] >= k {
                direct!(o, "self:sources:{}>={k}", col as char);
            }
        }
    }
    for &col in &PREVIEW_COLORS {
        if needs & bit(col) != 0 {
            direct!(o, "self:hand_needs:{}", col as char);
        }
    }
    for (i, &col) in PREVIEW_COLORS.iter().enumerate() {
        if needs & bit(col) != 0 && sources[i] == 0 {
            direct!(o, "self:hand_missing:{}", col as char);
        }
    }
}

/// encode.py `ENT_STEPS` / `MAX_ENTITIES`.
const ENT_STEPS: &[i64] = &[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15];
pub const MAX_ENTITIES: usize = 64; // legacy versions only; set 7 is ragged

/// Receives entity features: `begin` opens the next entity.
pub trait EntityOut {
    fn begin(&mut self);
    fn tok(&mut self, args: std::fmt::Arguments);
}

macro_rules! tok {
    ($o:expr, $($arg:tt)*) => { $o.tok(format_args!($($arg)*)) };
}

fn ent_thermo<E: EntityOut>(o: &mut E, name: &str, n: i64) {
    for &k in ENT_STEPS {
        if n >= k {
            tok!(o, "{name}>={k}");
        }
    }
}

/// encode.py `_combat_entity` (set 4): attacker and blocker relations.
fn combat_entity<E: EntityOut>(st: &State, c: &Card, o: &mut E) {
    let dt = |x: &Card| st.has(x, "deathtouch");
    if let Some(j) = st.attackers.iter().position(|&a| a == c.oid) {
        tok!(o, "e:attack_slot:{j}");
        tok!(o, "{}", if st.blocked.contains(&c.oid) { "e:blocked" } else { "e:unblocked" });
        ent_thermo(o, "e:blockers", blockers_of(st, c.oid).count() as i64);
        let power: i32 = blockers_of(st, c.oid).map(|b| st.power(b).max(0)).sum();
        ent_thermo(o, "e:block_power", power as i64);
        if dies_to(st, c, power, blockers_of(st, c.oid).any(dt)) {
            tok!(o, "e:block_lethal");
        }
    }
    let a = match st.blocks.iter().find(|(b, _)| *b == c.oid).and_then(|(_, a)| st.perm(*a)) {
        Some(ai) => st.c(ai),
        None => return,
    };
    if let Some(j) = st.attackers.iter().position(|&x| x == a.oid) {
        tok!(o, "e:blocking:slot:{j}");
    }
    tok!(o, "e:blocking:name:{}", a.name());
    ent_thermo(o, "e:blocking:power", st.power(a) as i64);
    ent_thermo(o, "e:blocking:toughness", st.toughness(a) as i64);
    if dies_to(st, a, st.power(c), dt(c)) {
        tok!(o, "e:blocking:kills");
    }
    if dies_to(st, c, st.power(a), dt(a)) {
        tok!(o, "e:blocking:dies");
    }
}

/// `encode.entity_features(game, viewer)`: permanents in battlefield order,
/// then the stack from the top, then (set 5) the viewer's hand. Returns the object id of each entity.
pub fn entity_features_into<E: EntityOut>(st: &State, viewer: u8, features: u8, o: &mut E) -> Vec<u32> {
    let (v2, v5) = (features >= 2, features >= 5);
    let mut ids = Vec::with_capacity(st.battlefield.len() + st.stack.len());
    for &ci in &st.battlefield {
        if features < 7 && ids.len() == MAX_ENTITIES {
            return ids;
        }
        let c = st.c(ci);
        o.begin();
        tok!(o, "e:name:{}", c.name());
        tok!(o, "e:ctrl:{}", rel(c.controller, viewer));
        let types = st.types(c);
        for (i, t) in TYPE_NAMES.iter().enumerate() {
            if types & (1 << i) != 0 {
                tok!(o, "e:type:{t}");
            }
        }
        for k in db().keyword_list(st.keywords(c)) {
            tok!(o, "e:kw:{k}");
        }
        if c.tapped {
            tok!(o, "e:tapped");
        }
        if c.sick {
            tok!(o, "e:sick");
        }
        if st.attackers.contains(&c.oid) {
            tok!(o, "e:attacking");
        }
        if c.is_token {
            tok!(o, "e:token");
        }
        if st.blocks.iter().any(|(b, _)| *b == c.oid) {
            tok!(o, "e:blocking");
        }
        if c.attached_to.is_some() {
            tok!(o, "e:attached");
        }
        if st.is_creature(c) {
            ent_thermo(o, "e:power", st.power(c) as i64);
            ent_thermo(o, "e:toughness", st.toughness(c) as i64);
        }
        ent_thermo(o, "e:damage", c.damage as i64);
        if c.counters > 0 {
            ent_thermo(o, "e:counters", c.counters as i64);
        } else if c.counters < 0 {
            tok!(o, "e:counters:{}", c.counters);
        }
        if v2 {
            if c.skip_untap > 0 {
                tok!(o, "e:skip_untap:{}", c.skip_untap);
            }
            targeted_by(st, viewer, Ref::Perm(c.oid), o);
        }
        if features >= 4 {
            let m = mana_colors(c);
            for &col in &PREVIEW_COLORS {
                if m & bit(col) != 0 {
                    tok!(o, "e:produces:{}", col as char);
                }
            }
            combat_entity(st, c, o);
        }
        if v5 {
            shape(c.face(), o);
        }
        if let Some(a) = st.damage_allocation.as_ref().filter(|_| features >= 7) {
            if c.oid == a.attacker { tok!(o, "e:damage:source"); }
            if let Some(j) = a.blockers.iter().position(|&b| b == c.oid) {
                tok!(o, "e:damage:slot:{j}");
                tok!(o, "e:damage:assigned:{}", a.assigned[j]);
                tok!(o, "e:damage:lethal:{}", a.lethal[j]);
                if j as i32 == a.recipient { tok!(o, "e:damage:recipient"); }
                if (j as i32) < a.recipient { tok!(o, "e:damage:committed"); }
            }
        }
        ids.push(c.oid);
    }
    for (i, it) in st.stack.iter().rev().enumerate() {
        if features < 7 && ids.len() == MAX_ENTITIES {
            break;
        }
        o.begin();
        tok!(o, "e:stack");
        tok!(o, "e:name:{}", it.name);
        tok!(o, "e:ctrl:{}", rel(it.controller, viewer));
        tok!(o, "e:stack_pos:{}", i.min(3));
        tok!(o, "e:stack_kind:{}", it.kind.name());
        if !v2 {
            ids.push(it.sid);
            continue;
        }
        if it.x > 0 {
            ent_thermo(o, "e:x", it.x as i64);
        }
        for &r in it.targets.iter().filter(|&&r| ref_exists(st, r)) {
            let (kind, who) = match r {
                Ref::Player(p) => ("player", p),
                Ref::Perm(oid) => ("perm", st.c(st.perm(oid).unwrap()).controller),
                Ref::Stack(sid) => ("spell", st.stack[st.stack_pos(sid).unwrap()].controller),
            };
            tok!(o, "e:targets:{kind}:{}", rel(who, viewer));
        }
        targeted_by(st, viewer, Ref::Stack(it.sid), o);
        if v5 {
            if it.kind == SKind::Spell {
                shape(st.c(it.card.expect("spell has a card")).face(), o);
            }
            res_ops(it.effect.unwrap_or(&[]), o);
        }
        ids.push(it.sid);
    }
    if v5 {
        hand_entities(st, viewer, features, o, &mut ids);
    }
    ids
}

fn shape<E: EntityOut>(d: &CardDef, o: &mut E) {
    for t in &d.shape {
        tok!(o, "{t}");
    }
}

/// encode.py `op_names(_ops(item.effect))` as `e:res:op:` tokens.
fn res_ops<E: EntityOut>(ops: &[Op], o: &mut E) {
    for op in ops {
        if let Some(n) = op.name() {
            tok!(o, "e:res:op:{n}");
        }
        if let Op::OptionalPayment { then, .. } = op {
            res_ops(then, o);
        }
    }
}

/// encode.py `_hand_entities`: the viewer's own hand cards (set 5).
fn hand_entities<E: EntityOut>(st: &State, viewer: u8, features: u8, o: &mut E, ids: &mut Vec<u32>) {
    let mut castable: Vec<&str> = vec![];
    if let Some(d) = st.decision.as_ref().filter(|d| d.player == viewer && d.kind == Kind::Priority) {
        for opt in &d.options {
            if let [KI::S("cast"), KI::S(name), KI::S("hand"), ..] = opt.key.as_slice() {
                castable.push(name);
            }
        }
    }
    for &ci in &st.players[viewer as usize].hand {
        if features < 7 && ids.len() == MAX_ENTITIES {
            return;
        }
        let c = st.c(ci);
        let f = c.face();
        o.begin();
        tok!(o, "e:zone:hand");
        tok!(o, "e:name:{}", c.name());
        tok!(o, "e:ctrl:self");
        for t in type_names(f.types) {
            tok!(o, "e:type:{t}");
        }
        for k in db().keyword_list(f.keywords) {
            tok!(o, "e:kw:{k}");
        }
        if let (Some(p), Some(t)) = (f.power, f.toughness) {
            ent_thermo(o, "e:power", p as i64);
            ent_thermo(o, "e:toughness", t as i64);
        }
        shape(f, o);
        if castable.contains(&c.name()) {
            tok!(o, "e:castable");
        }
        ids.push(c.oid);
    }
}

fn ref_exists(st: &State, r: Ref) -> bool {
    match r {
        Ref::Player(_) => true,
        Ref::Stack(s) => st.stack_pos(s).is_some(),
        Ref::Perm(o) => st.perm(o).is_some(),
    }
}

/// encode.py `_targeted_by`: who targets this object, and with what.
fn targeted_by<E: EntityOut>(st: &State, viewer: u8, r: Ref, o: &mut E) {
    for it in &st.stack {
        for _ in it.targets.iter().filter(|&&t| t == r) {
            let who = rel(it.controller, viewer);
            tok!(o, "e:targeted_by:{who}");
            tok!(o, "e:targeted_by:{who}:{}", it.name);
        }
    }
}

/// encode.py `option_object_ids`: every `#<digits>` in the label, plus the
/// source of an activated or mana ability; in set 4 also the stack item a
/// choose_x option is for.
pub fn option_object_ids(st: &State, o: &Opt, kind: Kind, features: u8) -> Vec<u32> {
    let mut ids = vec![];
    let b = o.label.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'#' {
            let mut j = i + 1;
            let mut n: u64 = 0;
            while j < b.len() && b[j].is_ascii_digit() {
                n = n * 10 + (b[j] - b'0') as u64;
                j += 1;
            }
            if j > i + 1 {
                ids.push(n as u32);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    match o.value {
        Val::Activate(c, _) | Val::Mana(c, _) => ids.push(st.c(c).oid),
        // Set 5: the hand card (other zones' cards are no entities).
        Val::Cast(c, _, _) | Val::Land(c) | Val::Plot(c) if features >= 5 => ids.push(st.c(c).oid),
        _ => {}
    }
    if features >= 4 && kind == Kind::ChooseX {
        if let Some(it) = st.stack.last() {
            ids.push(it.sid);
        }
    }
    ids
}

/// encode.py `MANA_LEFT_CAP` / `PREVIEW_COLORS`.
const MANA_LEFT_CAP: i32 = 8;
pub(crate) const PREVIEW_COLORS: [u8; 5] = [b'W', b'U', b'B', b'R', b'G'];

/// encode.py `_dies_to`: would `n` more damage destroy creature `c`?
fn dies_to(st: &State, c: &Card, n: i32, deathtouch: bool) -> bool {
    if n <= 0 || st.has(c, "indestructible") {
        return false;
    }
    deathtouch || c.damage + n >= st.toughness(c)
}

fn plus(cost: &ManaCost, generic: i32, color: Option<u8>) -> Remaining {
    let mut c = cost.clone();
    c.generic += generic;
    if let Some(col) = color {
        match c.colored.iter_mut().find(|(k, _)| *k == col) {
            Some(e) => e.1 += 1,
            None => {
                c.colored.push((col, 1));
                c.colored.sort();
            }
        }
    }
    Remaining::of(&c)
}

/// encode.py `_mana_preview`.
fn mana_preview(st: &State, player: u8, cost: &ManaCost, sac: Option<SacFilter>, exclude: &[u32], out: &mut impl FnMut(std::fmt::Arguments)) {
    let pl = &st.players[player as usize];
    let sources = st.mana_sources(player, exclude);
    let avail = pl.pool.iter().map(|(_, n)| n).sum::<i32>() + sources.len() as i32;
    out(format_args!("pv:mana_left_after:{}", (avail - cost.mana_value()).clamp(0, MANA_LEFT_CAP)));
    // Colours nothing can make are never left (skips the feasibility search).
    let mut makes = pl.pool.iter().filter(|(_, n)| *n > 0).fold(0u8, |m, (c, _)| m | bit(*c));
    for &(ci, ai) in sources.iter().chain(st.mana_filters(player, exclude).iter()) {
        makes |= st.c(ci).face().abilities[ai].mana.as_ref().map_or(0, |v| v.iter().fold(0, |m, c| m | bit(*c)));
    }
    for col in PREVIEW_COLORS {
        if makes & bit(col) != 0 && st.cost_feasible(player, &plus(cost, 0, Some(col)), sac, exclude, None, &[]) {
            out(format_args!("pv:colors_left:{}", col as char));
        }
    }
}

/// encode.py `_kills_preview`.
fn kills_preview(st: &State, player: u8, ops: Option<&[Op]>, source: &Card, out: &mut impl FnMut(std::fmt::Arguments)) {
    let mut dmg: Vec<(u32, i32)> = vec![];
    for op in ops.unwrap_or(&[]) {
        if let Op::DamageEachCreature { n, x: false, without, opponent_only, except_subtype } = op {
            for &ci in &st.battlefield {
                let c = st.c(ci);
                if *opponent_only && c.controller == player {
                    continue;
                }
                if except_subtype.as_deref().is_some_and(|s| c.face().has_creature_type(s)) {
                    continue;
                }
                if st.is_creature(c) && !(*without != 0 && st.keywords(c) & without != 0) {
                    match dmg.iter_mut().find(|e| e.0 == c.oid) {
                        Some(e) => e.1 += n,
                        None => dmg.push((c.oid, *n)),
                    }
                }
            }
        }
    }
    if dmg.is_empty() {
        return;
    }
    let deathtouch = st.has(source, "deathtouch");
    let (mut opp, mut me) = (0i64, 0i64);
    for &ci in &st.battlefield {
        let c = st.c(ci);
        if let Some(&(_, n)) = dmg.iter().find(|e| e.0 == c.oid) {
            if dies_to(st, c, n, deathtouch) {
                if c.controller == player {
                    me += 1;
                } else {
                    opp += 1;
                }
            }
        }
    }
    for &k in COUNT_STEPS {
        if opp >= k {
            out(format_args!("pv:kills_opp>={k}"));
        }
    }
    for &k in COUNT_STEPS {
        if me >= k {
            out(format_args!("pv:kills_self>={k}"));
        }
    }
    if opp + me == 0 {
        out(format_args!("pv:kills_none"));
    }
}

/// encode.py `_item_cost`: (cost, sacrifice filter, excluded source) of the
/// stack item whose targets are being chosen.
fn item_cost(st: &State, item: &StackItem, x: Option<i32>) -> (ManaCost, Option<SacFilter>, Vec<u32>) {
    if item.kind == SKind::Spell {
        let ci = item.card.expect("spell has a card");
        let base = st.mode_cost(ci, item.method).expect("cast mode has a cost");
        return (base.with_x(x.unwrap_or(item.x)).reduced(st.cost_reduction(item.controller, ci)), st.c(ci).face().additional_sac, vec![]);
    }
    if let Some(s) = &item.source {
        let src = st.src(s);
        for ab in &src.face().abilities {
            if item.name == format!("{}: {}", src.name(), ab.name) {
                return (ab.cost.clone(), ab.sac_other, if ab.tap { vec![src.oid] } else { vec![] });
            }
        }
    }
    (ManaCost::default(), None, vec![])
}

/// encode.py `_target_preview`.
fn target_preview(st: &State, player: u8, r: Ref, out: &mut impl FnMut(std::fmt::Arguments)) {
    let item = match st.stack.last() {
        Some(it) if !matches!(r, Ref::Stack(_)) => it,
        _ => return,
    };
    let mut target: Option<&Card> = None;
    if let Ref::Perm(oid) = r {
        let c = match st.perm(oid) {
            Some(ci) => st.c(ci),
            None => return,
        };
        let ward = c.face().ward;
        if ward > 0 && c.controller != player {
            out(format_args!("pv:target_ward:{ward}"));
            let (cost, sac, exclude) = item_cost(st, item, None);
            if st.cost_feasible(player, &plus(&cost, ward, None), sac, &exclude, None, &[]) {
                out(format_args!("pv:ward_payable"));
            }
        }
        target = Some(c);
    }
    let source: Option<&Card> = match (item.card, &item.source) {
        (Some(ci), _) if item.kind == SKind::Spell => Some(st.c(ci)),
        (_, Some(s)) => Some(st.src(s)),
        _ => None,
    };
    let deathtouch = source.map(|s| st.has(st.live(s).map(|ci| st.c(ci)).unwrap_or(s), "deathtouch")).unwrap_or(false);
    let landfall = st.players[item.controller as usize].landfall_turn == st.turn;
    for op in item.effect.unwrap_or(&[]) {
        // Only the op for the target being chosen.
        if let Op::DamageTarget { n, index, n_landfall, .. } = op {
            if *index != item.targets.len() {
                continue;
            }
            let n = match n_landfall {
                Some(l) if landfall => *l,
                _ => *n,
            };
            let lethal = match (r, target) {
                (Ref::Player(p), _) => n >= st.players[p as usize].life,
                (_, Some(c)) => st.is_creature(c) && dies_to(st, c, n, deathtouch),
                _ => false,
            };
            if lethal {
                out(format_args!("pv:damage_lethal_to_target"));
            }
        }
    }
}

/// encode.py `_block_preview` (set 4): a declare_blocker option.
fn block_preview(st: &State, player: u8, o: &Opt, out: &mut impl FnMut(std::fmt::Arguments)) {
    let mut skip = None;
    if let Val::Card(ai) = o.value {
        let a = st.c(ai);
        let b = match option_object_ids(st, o, Kind::DeclareBlocker, 1).first().and_then(|&id| st.perm(id)) {
            Some(bi) => st.c(bi),
            None => return,
        };
        if st.blocked.contains(&a.oid) {
            out(format_args!("pv:attacker_already_blocked"));
        }
        let power = blockers_of(st, a.oid).map(|x| st.power(x).max(0)).sum::<i32>() + st.power(b).max(0);
        let dt = blockers_of(st, a.oid).any(|x| st.has(x, "deathtouch")) || st.has(b, "deathtouch");
        if dies_to(st, a, power, dt) {
            out(format_args!("pv:attacker_dies"));
        }
        if dies_to(st, b, st.power(a), st.has(a, "deathtouch")) {
            out(format_args!("pv:blocker_dies"));
        }
        skip = Some(a.oid);
    }
    let left = unblocked_power(st, skip);
    for &k in POWER_STEPS {
        if left >= k {
            out(format_args!("pv:unblocked_damage_left>={k}"));
        }
    }
    if left > 0 && left >= st.players[player as usize].life as i64 {
        out(format_args!("pv:lethal_left"));
    }
}

/// encode.py `_x_preview` (set 4): a choose_x option.
fn x_preview(st: &State, player: u8, x: i32, out: &mut impl FnMut(std::fmt::Arguments)) {
    for &k in COUNT_STEPS {
        if x as i64 >= k {
            out(format_args!("pv:x>={k}"));
        }
    }
    let max = st.decision.as_ref().and_then(|d| d.options.iter().filter_map(|o| if let Val::Int(v) = o.value { Some(v) } else { None }).max());
    if max == Some(x) {
        out(format_args!("pv:x_is_max"));
    }
    let item = match st.stack.last() {
        Some(it) => it,
        None => return,
    };
    let (cost, sac, exclude) = item_cost(st, item, Some(x));
    mana_preview(st, player, &cost, sac, &exclude, out);
    if item.kind == SKind::Spell {
        let face = st.c(item.card.expect("spell has a card")).face();
        if face.etb_x_counters {
            let p = (face.power.unwrap_or(0) + x) as i64;
            for &k in ENT_STEPS {
                if p >= k {
                    out(format_args!("pv:enters_power>={k}"));
                }
            }
        }
    }
}

/// encode.py `_adds_colour_preview` (set 4): playing or fetching a land.
fn adds_colour_preview(st: &State, player: u8, c: &Card, out: &mut impl FnMut(std::fmt::Arguments)) {
    let m = mana_colors(c);
    if m == 0 {
        return;
    }
    let (sources, needs) = colour_counts(st, player);
    for &col in &PREVIEW_COLORS {
        if m & bit(col) != 0 {
            out(format_args!("pv:adds_color:{}", col as char));
        }
    }
    if PREVIEW_COLORS.iter().enumerate().any(|(i, &col)| m & bit(col) != 0 && needs & bit(col) != 0 && sources[i] == 0) {
        out(format_args!("pv:adds_missing_color"));
    }
}

/// encode.py `_pay_preview` (set 4): colours still producible once the rest
/// of the cost is paid after spending this unit.
fn pay_preview(st: &State, player: u8, val: &Val, out: &mut impl FnMut(std::fmt::Arguments)) {
    let (rem, sac, exclude) = match &st.paying {
        Some(p) => p,
        None => return,
    };
    let mut r2 = rem.clone();
    let mut excl = exclude.clone();
    let mut pool2 = None;
    let mut gone = vec![];
    match *val {
        Val::Pool(col) => {
            r2.apply(col);
            let mut pool = st.players[player as usize].pool.clone();
            for e in pool.iter_mut() {
                if e.0 == col {
                    e.1 -= 1;
                }
            }
            pool2 = Some(pool);
        }
        Val::Source(ci, col) => {
            r2.apply(col);
            let c = st.c(ci);
            excl.push(c.oid);
            if c.face().abilities.iter().find(|a| a.mana.is_some()).is_some_and(|a| a.sac_self) {
                gone.push(c.oid);
            }
        }
        _ => return,
    }
    for col in PREVIEW_COLORS {
        let mut r3 = r2.clone();
        match r3.colored.iter_mut().find(|(k, _)| *k == col) {
            Some(e) => e.1 += 1,
            None => r3.colored.push((col, 1)),
        }
        if st.cost_feasible(player, &r3, *sac, &excl, pool2.as_deref(), &gone) {
            out(format_args!("pv:colors_left:{}", col as char));
        }
    }
}

/// encode.py `_preview_v4`: set-4 previews; false for the decision kinds and
/// options it does not cover.
fn preview_v4(st: &State, player: u8, kind: Kind, o: &Opt, out: &mut impl FnMut(std::fmt::Arguments)) -> bool {
    match (kind, &o.value) {
        (Kind::DeclareBlocker, _) => block_preview(st, player, o, out),
        (Kind::ChooseX, Val::Int(x)) => x_preview(st, player, *x, out),
        (Kind::PayMana, v) => pay_preview(st, player, v, out),
        (Kind::Priority, Val::Land(ci)) => adds_colour_preview(st, player, st.c(*ci), out),
        (Kind::ChooseCard, Val::Card(ci)) if matches!(o.key.first(), Some(KI::S("search"))) => adds_colour_preview(st, player, st.c(*ci), out),
        _ => return false,
    }
    true
}

/// encode.py `option_preview`: engine-computed effects of taking an option.
pub fn option_preview(st: &State, player: u8, kind: Kind, o: &Opt, features: u8, out: &mut impl FnMut(std::fmt::Arguments)) {
    if features >= 4 && preview_v4(st, player, kind, o, out) {
        return;
    }
    match (kind, &o.value) {
        (Kind::Target, Val::Ref(r)) => target_preview(st, player, *r, out),
        (Kind::Priority, Val::Activate(ci, ai)) => {
            // A determinized copy re-deals hidden cards under pending options,
            // so an option may name an ability, mode or cost the card no longer has.
            let c = st.c(*ci);
            let ab = match c.face().abilities.get(*ai as usize) {
                Some(ab) => ab,
                None => return,
            };
            kills_preview(st, player, ab.effect.as_deref(), c, out);
            let exclude = if ab.tap { vec![c.oid] } else { vec![] };
            mana_preview(st, player, &ab.cost, ab.sac_other, &exclude, out);
        }
        (Kind::Priority, Val::Cast(ci, method, choice)) => {
            let d = st.c(*ci).face();
            let base = match st.mode_cost(*ci, *method) {
                Some(b) if choice.map_or(true, |i| (i as usize) < d.modes.len()) => b,
                _ => return,
            };
            let effect = match choice {
                _ if *method == Method::Overload => d.overload_effect.as_deref(),
                Some(i) => Some(d.modes[*i as usize].effect.as_slice()),
                None => d.effect.as_deref(),
            };
            kills_preview(st, player, effect, st.c(*ci), out);
            let cost = base.with_x(0).reduced(st.cost_reduction(player, *ci));
            mana_preview(st, player, &cost, d.additional_sac, &[], out);
        }
        _ => {}
    }
}

/// String form of option previews (differential tests).
pub fn option_preview_strings(st: &State, player: u8, i: usize, features: u8) -> Option<Vec<String>> {
    let d = st.decision.as_ref()?;
    let o = d.options.get(i)?;
    let mut v = vec![];
    if features >= 2 {
        option_preview(st, player, d.kind, o, features, &mut |a| v.push(a.to_string()));
    }
    Some(v)
}

/// String form of entities (differential tests).
#[derive(Default)]
pub struct EntityStrings(pub Vec<Vec<String>>);

impl EntityOut for EntityStrings {
    fn begin(&mut self) {
        self.0.push(vec![]);
    }
    fn tok(&mut self, args: std::fmt::Arguments) {
        self.0.last_mut().unwrap().push(args.to_string());
    }
}

/// Hashed entities appended to the state: separator `dim`, then the sorted,
/// deduplicated hashes of the entity's features.
struct EntityHashes<'a> {
    out: &'a mut Vec<u32>,
    start: usize,
    buf: String,
    dim: u32,
}

impl EntityHashes<'_> {
    fn close(&mut self) {
        if self.start < self.out.len() {
            let seg = &mut self.out[self.start..];
            seg.sort_unstable();
            let mut w = 0;
            for r in 0..seg.len() {
                if r == 0 || seg[r] != seg[w - 1] {
                    seg[w] = seg[r];
                    w += 1;
                }
            }
            let keep = self.start + w;
            self.out.truncate(keep);
        }
    }
}

impl EntityOut for EntityHashes<'_> {
    fn begin(&mut self) {
        self.close();
        self.out.push(self.dim);
        self.start = self.out.len();
    }
    fn tok(&mut self, args: std::fmt::Arguments) {
        self.buf.clear();
        let _ = self.buf.write_fmt(args);
        self.out.push(crc32fast::hash(self.buf.as_bytes()) % self.dim);
    }
}

/// String form (differential tests): direct features in order, then each
/// distinct counted feature followed by its `#k` copies, in first-occurrence order.
#[derive(Default)]
pub struct StringOut {
    out: Vec<String>,
    counted: Vec<(String, u32)>,
    index: std::collections::HashMap<String, usize>,
}

impl FeatureOut for StringOut {
    fn direct(&mut self, args: std::fmt::Arguments) {
        self.out.push(args.to_string());
    }
    fn raw(&mut self, args: std::fmt::Arguments) {
        let s = args.to_string();
        if let Some(&i) = self.index.get(&s) {
            self.counted[i].1 += 1;
        } else {
            self.index.insert(s.clone(), self.counted.len());
            self.counted.push((s, 1));
        }
    }
}

impl StringOut {
    pub fn finish(mut self) -> Vec<String> {
        for (s, n) in std::mem::take(&mut self.counted) {
            for k in 2..=n.min(COUNT_CAP) {
                let extra = format!("{s}#{k}");
                if k == 2 {
                    self.out.push(s.clone());
                }
                self.out.push(extra);
            }
            if n < 2 {
                self.out.push(s);
            }
        }
        self.out
    }
}

/// Hashed form: formats each feature into one reusable buffer and keeps only
/// crc32 values. Counted features are keyed by their full crc and extended
/// with `#k` from a saved hasher state, so no strings are allocated.
struct HashOut {
    buf: String,
    out: Vec<u32>,
    dim: u32,
    counted: Vec<(u32, Hasher, u32)>,
}

impl FeatureOut for HashOut {
    fn direct(&mut self, args: std::fmt::Arguments) {
        self.buf.clear();
        let _ = self.buf.write_fmt(args);
        self.out.push(crc32fast::hash(self.buf.as_bytes()) % self.dim);
    }
    fn raw(&mut self, args: std::fmt::Arguments) {
        self.buf.clear();
        let _ = self.buf.write_fmt(args);
        let mut h = Hasher::new();
        h.update(self.buf.as_bytes());
        let full = h.clone().finalize();
        if let Some(e) = self.counted.iter_mut().find(|e| e.0 == full) {
            e.2 += 1;
        } else {
            self.counted.push((full, h, 1));
        }
    }
}

impl HashOut {
    fn finish(mut self) -> Vec<u32> {
        let mut b = String::with_capacity(4);
        for (full, h, n) in std::mem::take(&mut self.counted) {
            self.out.push(full % self.dim);
            for k in 2..=n.min(COUNT_CAP) {
                b.clear();
                let _ = write!(b, "#{k}");
                let mut hk = h.clone();
                hk.update(b.as_bytes());
                self.out.push(hk.finalize() % self.dim);
            }
        }
        self.out
    }
}

fn write_ki(buf: &mut String, k: &KI) {
    match k {
        KI::S(x) => buf.push_str(x),
        KI::I(i) => {
            let _ = write!(buf, "{i}");
        }
        KI::N => buf.push_str("None"),
        KI::T(v) => {
            buf.push('(');
            for (j, x) in v.iter().enumerate() {
                if j > 0 {
                    buf.push_str(", ");
                }
                let _ = write!(buf, "{x}");
            }
            if v.len() == 1 {
                buf.push(',');
            }
            buf.push(')');
        }
    }
}

/// Hashes of `[tag + t for t in option_tokens(kind, key)]`.
pub fn option_token_hashes(kind: Kind, key: &Key, tag: &str, dim: u32, out: &mut Vec<u32>) {
    let mut elems: Vec<String> = Vec::with_capacity(key.len() + 1);
    elems.push(kind.name().to_string());
    for k in key {
        let mut b = String::new();
        write_ki(&mut b, k);
        elems.push(b);
    }
    let mut buf = String::with_capacity(32);
    for (i, e) in elems.iter().enumerate() {
        buf.clear();
        let _ = write!(buf, "{tag}{i}={e}");
        out.push(crc32fast::hash(buf.as_bytes()) % dim);
    }
    let mut h = Hasher::new();
    h.update(tag.as_bytes());
    h.update(elems[0].as_bytes());
    for e in &elems[1..] {
        h.update(b"|");
        h.update(e.as_bytes());
        out.push(h.clone().finalize() % dim);
    }
}

/// `rl.features.featurize(game, player, state_dim, option_dim, features)`.
/// `sims`: each option's hashed simulated previews (set 6, `sim.rs`), which
/// need the game, not only its state.
pub fn featurize(st: &State, player: u8, state_dim: u32, option_dim: u32, features: u8, sims: Option<&[Vec<u32>]>) -> Option<(Vec<u32>, Vec<Vec<u32>>)> {
    let d = st.decision.as_ref()?;
    let mut s = HashOut { buf: String::with_capacity(64), out: Vec::with_capacity(200), dim: state_dim, counted: Vec::with_capacity(96) };
    state_features_into(st, player, features, &mut s);
    if features < 7 { direct!(s, "seat:{player}"); }
    direct!(s, "decision:{}", d.kind.name());
    let mut state = s.finish();
    state.sort_unstable();
    state.dedup();
    let start = state.len();
    let ids = {
        let mut eo = EntityHashes { out: &mut state, start, buf: String::with_capacity(48), dim: state_dim };
        let ids = entity_features_into(st, player, features, &mut eo);
        eo.close();
        ids
    };
    let mut pbuf = String::with_capacity(32);
    let opts = d
        .options
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let mut v = Vec::with_capacity(2 * o.key.len() + 3);
            option_token_hashes(d.kind, &o.key, "", option_dim, &mut v);
            if let Some(s) = sims {
                v.extend_from_slice(&s[i]);
            }
            if features >= 2 {
                option_preview(st, player, d.kind, o, features, &mut |a| {
                    pbuf.clear();
                    let _ = pbuf.write_fmt(a);
                    v.push(crc32fast::hash(pbuf.as_bytes()) % option_dim);
                });
            }
            v.sort_unstable();
            v.dedup();
            let mut ptr: Vec<u32> = option_object_ids(st, o, d.kind, features).iter().filter_map(|id| ids.iter().position(|x| x == id)).map(|k| option_dim + k as u32).collect();
            ptr.sort_unstable();
            ptr.dedup();
            v.extend(ptr);
            v
        })
        .collect();
    Some((state, opts))
}

/// Hashes of `event_tokens(kind, key, mine)` for both players after option
/// `i` of the current decision is taken: (decider's, opponent's).
pub fn event_hashes(st: &State, i: usize, option_dim: u32) -> Option<(Vec<u32>, Vec<u32>)> {
    let d = st.decision.as_ref()?;
    let key = &d.options.get(i)?.key;
    let mut mine = vec![];
    option_token_hashes(d.kind, key, "self>", option_dim, &mut mine);
    let mut theirs = vec![];
    if is_public(d.kind) {
        option_token_hashes(d.kind, key, "opp>", option_dim, &mut theirs);
    } else {
        theirs.push(crc32fast::hash(format!("opp>{}", d.kind.name()).as_bytes()) % option_dim);
    }
    Some((mine, theirs))
}

