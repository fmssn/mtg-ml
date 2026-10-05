//! Hashed network inputs, bit-identical to mtg_ml/rl/features.py and
//! mtg_ml/encode.py (zlib.crc32 of the same strings), computed without
//! building the strings as Python objects. `py.rs::state_features` keeps the
//! string form for the differential tests.

use std::fmt::Write;

use crc32fast::Hasher;

use crate::cards::TYPE_NAMES;
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

/// Formats features into one reusable buffer and collects their hashes.
struct Sink {
    buf: String,
    out: Vec<u32>,
    dim: u32,
}

impl Sink {
    #[inline]
    fn emit(&mut self) {
        self.out.push(crc32fast::hash(self.buf.as_bytes()) % self.dim);
        self.buf.clear();
    }
}

macro_rules! feat {
    ($s:expr, $($arg:tt)*) => {{
        let _ = write!($s.buf, $($arg)*);
        $s.emit();
    }};
}

/// `encode.state_features(game, viewer)` hashed into `dim` buckets (order
/// as in Python; callers sort and dedupe).
fn state_feature_hashes(st: &State, viewer: u8, s: &mut Sink) {
    let opp = 1 - viewer;
    feat!(s, "step:{}", st.step_name);
    feat!(s, "active:{}", rel(st.active, viewer));
    feat!(s, "postboard:{}", if st.match_game > 1 { "True" } else { "False" });
    for (side, p) in [("self", viewer), ("opponent", opp)] {
        let pl = &st.players[p as usize];
        feat!(s, "{side}:life:{}", bucket(pl.life as i64));
        feat!(s, "{side}:library:{}", bucket(pl.library.len() as i64));
        feat!(s, "{side}:mulligans:{}", st.mulligans_taken[p as usize]);
        for &c in &pl.graveyard {
            feat!(s, "{side}:gy:{}", st.c(c).name());
        }
        for (c, n) in &pl.pool {
            feat!(s, "{side}:pool:{}:{n}", color_str(*c));
        }
        for &c in pl.library.iter().filter(|&&c| st.c(c).known_to & pbit(viewer) != 0).take(3) {
            feat!(s, "{side}:known_library:{}", st.c(c).name());
        }
    }
    for &c in &st.players[viewer as usize].hand {
        feat!(s, "self:hand:{}", st.c(c).name());
    }
    let them = &st.players[opp as usize];
    feat!(s, "opponent:hand_count:{}", bucket(them.hand.len() as i64));
    for &c in them.hand.iter().filter(|&&c| st.c(c).known_to & pbit(viewer) != 0) {
        feat!(s, "opponent:hand_known:{}", st.c(c).name());
    }
    for &ci in &st.battlefield {
        let c = st.c(ci);
        let side = rel(c.controller, viewer);
        let name = c.name();
        feat!(s, "{side}:bf:{name}");
        let types = st.types(c);
        for (i, t) in TYPE_NAMES.iter().enumerate() {
            if types & (1 << i) != 0 {
                feat!(s, "{side}:bf_type:{t}");
            }
        }
        if c.tapped {
            feat!(s, "{side}:bf:{name}:tapped");
        }
        if c.sick {
            feat!(s, "{side}:bf:{name}:sick");
        }
        if st.attackers.contains(&c.oid) {
            feat!(s, "{side}:bf:{name}:attacking");
        }
        if st.blocks.iter().any(|(b, _)| *b == c.oid) {
            feat!(s, "{side}:bf:{name}:blocking");
        }
        if st.is_creature(c) {
            feat!(s, "{side}:bf:{name}:pt:{}/{}", st.power(c), st.toughness(c));
        }
        if c.counters != 0 {
            feat!(s, "{side}:bf:{name}:counters:{}", c.counters);
        }
    }
    for (i, it) in st.stack.iter().rev().enumerate() {
        feat!(s, "stack:{}:{}:{}", i.min(3), rel(it.controller, viewer), it.name);
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
    let mut s = Sink { buf: String::with_capacity(64), out: Vec::with_capacity(96), dim: state_dim };
    state_feature_hashes(st, player, &mut s);
    feat!(s, "seat:{player}");
    feat!(s, "decision:{}", d.kind.name());
    let mut state = s.out;
    state.sort_unstable();
    state.dedup();
    let opts = d
        .options
        .iter()
        .map(|o| {
            let mut v = Vec::with_capacity(2 * o.key.len() + 2);
            option_token_hashes(d.kind, &o.key, "", option_dim, &mut v);
            v.sort_unstable();
            v.dedup();
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

