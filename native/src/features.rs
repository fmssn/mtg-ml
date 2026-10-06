//! Hashed network inputs, bit-identical to mtg_ml/rl/features.py and
//! mtg_ml/encode.py (zlib.crc32 of the same strings), computed without
//! building the strings as Python objects. `py.rs::state_features` keeps the
//! string form for the differential tests.

use std::fmt::Write;

use crc32fast::Hasher;

use crate::cards::{db, TYPE_NAMES};
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

/// `encode.state_features(game, viewer)`: the direct features in order, plus
/// the per-object ones, which the sink counts and appends in `finish`.
pub fn state_features_into<O: FeatureOut>(st: &State, viewer: u8, o: &mut O) {
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
pub fn entity_features_into<E: EntityOut>(st: &State, viewer: u8, o: &mut E) -> Vec<u32> {
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
        ids.push(it.sid);
    }
    ids
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

/// `rl.features.featurize(game, player, state_dim, option_dim)`.
pub fn featurize(st: &State, player: u8, state_dim: u32, option_dim: u32) -> Option<(Vec<u32>, Vec<Vec<u32>>)> {
    let d = st.decision.as_ref()?;
    let mut s = HashOut { buf: String::with_capacity(64), out: Vec::with_capacity(200), dim: state_dim, counted: Vec::with_capacity(96) };
    state_features_into(st, player, &mut s);
    direct!(s, "seat:{player}");
    direct!(s, "decision:{}", d.kind.name());
    let mut state = s.finish();
    state.sort_unstable();
    state.dedup();
    let start = state.len();
    let ids = {
        let mut eo = EntityHashes { out: &mut state, start, buf: String::with_capacity(48), dim: state_dim };
        let ids = entity_features_into(st, player, &mut eo);
        eo.close();
        ids
    };
    let opts = d
        .options
        .iter()
        .map(|o| {
            let mut v = Vec::with_capacity(2 * o.key.len() + 3);
            option_token_hashes(d.kind, &o.key, "", option_dim, &mut v);
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

