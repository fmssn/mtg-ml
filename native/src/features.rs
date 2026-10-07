//! Hashed network inputs, bit-identical to mtg_ml/rl/features.py and
//! mtg_ml/encode.py (zlib.crc32 of the same strings), computed without
//! building the strings as Python objects. `py.rs::state_features` keeps the
//! string form for the differential tests.

use std::fmt::Write;

use crc32fast::Hasher;

use crate::cards::{db, Op, SacFilter, TYPE_NAMES};
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
/// option previews).
pub const FEATURES: u8 = 2;

pub fn check_features(features: u8) -> Result<u8, String> {
    if (1..=FEATURES).contains(&features) {
        Ok(features)
    } else {
        Err(format!("unknown feature-set version {features} (known: 1..={FEATURES})"))
    }
}

/// `encode.state_features(game, viewer, features)`: the direct features in order, plus
/// the per-object ones, which the sink counts and appends in `finish`.
pub fn state_features_into<O: FeatureOut>(st: &State, viewer: u8, features: u8, o: &mut O) {
    let opp = 1 - viewer;
    direct!(o, "step:{}", st.step_name);
    direct!(o, "active:{}", rel(st.active, viewer));
    direct!(o, "postboard:{}", if st.match_game > 1 { "True" } else { "False" });
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
            raw!(o, "{side}:exile:{}", st.c(c).name());
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
            let blockable = blockers.iter().any(|&b| c.5 & fly == 0 || b & (fly | reach) != 0);
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

/// encode.py `ENT_STEPS` / `MAX_ENTITIES`.
const ENT_STEPS: &[i64] = &[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15];
pub const MAX_ENTITIES: usize = 64;

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

/// `encode.entity_features(game, viewer)`: permanents in battlefield order,
/// then the stack from the top. Returns the object id of each entity.
pub fn entity_features_into<E: EntityOut>(st: &State, viewer: u8, features: u8, o: &mut E) -> Vec<u32> {
    let v2 = features >= 2;
    let mut ids = Vec::with_capacity(st.battlefield.len() + st.stack.len());
    for &ci in &st.battlefield {
        if ids.len() == MAX_ENTITIES {
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
        ids.push(c.oid);
    }
    for (i, it) in st.stack.iter().rev().enumerate() {
        if ids.len() == MAX_ENTITIES {
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
        ids.push(it.sid);
    }
    ids
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
/// source of an activated or mana ability.
pub fn option_object_ids(st: &State, o: &Opt) -> Vec<u32> {
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
    if let Val::Activate(c, _) | Val::Mana(c, _) = o.value {
        ids.push(st.c(c).oid);
    }
    ids
}

/// encode.py `MANA_LEFT_CAP` / `PREVIEW_COLORS`.
const MANA_LEFT_CAP: i32 = 8;
const PREVIEW_COLORS: [u8; 5] = [b'W', b'U', b'B', b'R', b'G'];

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
    for &(ci, ai) in &sources {
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
        if let Op::DamageEachCreature { n, without } = op {
            for &ci in &st.battlefield {
                let c = st.c(ci);
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
fn item_cost(st: &State, item: &StackItem) -> (ManaCost, Option<SacFilter>, Vec<u32>) {
    if item.kind == SKind::Spell {
        let ci = item.card.expect("spell has a card");
        let base = st.mode_cost(ci, item.method).expect("cast mode has a cost");
        return (base.with_x(item.x).reduced(st.cost_reduction(item.controller, ci)), st.c(ci).face().additional_sac, vec![]);
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
            let (cost, sac, exclude) = item_cost(st, item);
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
    for op in item.effect.unwrap_or(&[]) {
        if let Op::DamageTarget { n } = op {
            let lethal = match (r, target) {
                (Ref::Player(p), _) => *n >= st.players[p as usize].life,
                (_, Some(c)) => st.is_creature(c) && dies_to(st, c, *n, deathtouch),
                _ => false,
            };
            if lethal {
                out(format_args!("pv:damage_lethal_to_target"));
            }
        }
    }
}

/// encode.py `option_preview`: engine-computed effects of taking an option.
pub fn option_preview(st: &State, player: u8, kind: Kind, val: &Val, out: &mut impl FnMut(std::fmt::Arguments)) {
    match (kind, val) {
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
pub fn option_preview_strings(st: &State, player: u8, i: usize) -> Option<Vec<String>> {
    let d = st.decision.as_ref()?;
    let o = d.options.get(i)?;
    let mut v = vec![];
    option_preview(st, player, d.kind, &o.value, &mut |a| v.push(a.to_string()));
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
pub fn featurize(st: &State, player: u8, state_dim: u32, option_dim: u32, features: u8) -> Option<(Vec<u32>, Vec<Vec<u32>>)> {
    let d = st.decision.as_ref()?;
    let mut s = HashOut { buf: String::with_capacity(64), out: Vec::with_capacity(200), dim: state_dim, counted: Vec::with_capacity(96) };
    state_features_into(st, player, features, &mut s);
    direct!(s, "seat:{player}");
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
        .map(|o| {
            let mut v = Vec::with_capacity(2 * o.key.len() + 3);
            option_token_hashes(d.kind, &o.key, "", option_dim, &mut v);
            if features >= 2 {
                option_preview(st, player, d.kind, &o.value, &mut |a| {
                    pbuf.clear();
                    let _ = pbuf.write_fmt(a);
                    v.push(crc32fast::hash(pbuf.as_bytes()) % option_dim);
                });
            }
            v.sort_unstable();
            v.dedup();
            let mut ptr: Vec<u32> = option_object_ids(st, o).iter().filter_map(|id| ids.iter().position(|x| x == id)).map(|k| option_dim + k as u32).collect();
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

