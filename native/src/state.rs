//! Game state and the rules that never ask a player anything (zones,
//! characteristics, state-based actions, targets, cost feasibility).
//! Mirrors the non-generator methods of mtg_ml/engine/game.py one to one;
//! the method names are kept so the two files can be read side by side.

use crate::cards::{db, CardDef, CostRed, DefId, Event, Op, SacFilter, TriggerDef, T_ARTIFACT, T_CREATURE, T_INSTANT, T_LAND, T_SORCERY, TK};
use crate::mana::{bit, can_pay, ManaCost, Remaining};
use crate::rng::PyRandom;

pub type CIdx = u32;

pub const MAX_HAND: usize = 7;

pub const STEPS: [&str; 12] = [
    "untap",
    "upkeep",
    "draw",
    "main1",
    "begin_combat",
    "declare_attackers",
    "declare_blockers",
    "combat_damage",
    "end_combat",
    "main2",
    "end",
    "cleanup",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Zone {
    Library,
    Hand,
    Graveyard,
    Exile,
    Battlefield,
    Stack,
    Gone,
}

impl Zone {
    pub fn name(self) -> &'static str {
        match self {
            Zone::Library => "library",
            Zone::Hand => "hand",
            Zone::Graveyard => "graveyard",
            Zone::Exile => "exile",
            Zone::Battlefield => "battlefield",
            Zone::Stack => "stack",
            Zone::Gone => "gone",
        }
    }
    pub fn parse(s: &str) -> Option<Zone> {
        Some(match s {
            "library" => Zone::Library,
            "hand" => Zone::Hand,
            "graveyard" => Zone::Graveyard,
            "exile" => Zone::Exile,
            "battlefield" => Zone::Battlefield,
            "stack" => Zone::Stack,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TempEffect {
    pub keywords: u32,
    pub power: i32,
    pub toughness: i32,
}

pub const BOTH: u8 = 0b11;

pub fn pbit(p: u8) -> u8 {
    1 << p
}

#[derive(Clone, Debug)]
pub struct Card {
    pub uid: u32,
    pub oid: u32,
    pub def: DefId,
    pub owner: u8,
    pub controller: u8,
    pub zone: Zone,
    pub is_token: bool,
    pub transformed: bool,
    pub tapped: bool,
    pub damage: i32,
    pub deathtouch_damage: bool,
    pub counters: i32,
    pub sick: bool,
    pub attached_to: Option<u32>,
    pub skip_untap: i32,
    pub temp: Vec<TempEffect>,
    /// Bit p set: player p knows this card.
    pub known_to: u8,
}

impl Card {
    pub fn face(&self) -> &'static CardDef {
        let d = db().def(self.def);
        if self.transformed {
            if let Some(b) = d.back {
                return db().def(b);
            }
        }
        d
    }
    pub fn defn(&self) -> &'static CardDef {
        db().def(self.def)
    }
    pub fn name(&self) -> &'static str {
        &self.face().name
    }
    pub fn reset_state(&mut self) {
        self.transformed = false;
        self.tapped = false;
        self.damage = 0;
        self.deathtouch_damage = false;
        self.counters = 0;
        self.sick = false;
        self.attached_to = None;
        self.skip_untap = 0;
        self.temp.clear();
    }
    pub fn repr(&self) -> String {
        format!("{}#{}", self.name(), self.oid)
    }
}

/// A reference to a card object: the live object (Python: the same `Card`
/// instance, following it through zone changes) or a last-known-information
/// copy (`Card.snapshot()`).
#[derive(Clone, Debug)]
pub enum Src {
    Live(CIdx),
    Snap(Box<Card>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ref {
    Player(u8),
    Perm(u32),
    Stack(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SKind {
    Spell,
    Ability,
    Trigger,
}

impl SKind {
    pub fn name(self) -> &'static str {
        match self {
            SKind::Spell => "spell",
            SKind::Ability => "ability",
            SKind::Trigger => "trigger",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Normal,
    Bestow,
    Flashback,
    Escape,
}

impl Method {
    pub fn name(self) -> &'static str {
        match self {
            Method::Normal => "normal",
            Method::Bestow => "bestow",
            Method::Flashback => "flashback",
            Method::Escape => "escape",
        }
    }
}

/// Trigger / stack item payload (Python: the `data` dict). Only the keys a
/// given trigger sets are `Some`.
#[derive(Clone, Debug, Default)]
pub struct Data {
    pub card: Option<CIdx>,
    pub oid: Option<u32>,
    pub sacrificed: Option<Box<Card>>,
    pub spell_sid: Option<u32>,
    pub sid: Option<u32>,
    pub amount: Option<i32>,
    pub source_oid: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct StackItem {
    pub sid: u32,
    pub kind: SKind,
    pub controller: u8,
    pub name: String,
    pub effect: Option<&'static [Op]>,
    pub target_specs: Vec<TK>,
    pub targets: Vec<Ref>,
    pub card: Option<CIdx>,
    pub source: Option<Src>,
    pub method: Method,
    pub cast_from: Zone,
    pub x: i32,
    pub data: Data,
}

#[derive(Clone, Debug)]
pub struct PendingTrigger {
    pub controller: u8,
    pub source: Src,
    pub tdef: &'static TriggerDef,
    pub data: Data,
}

#[derive(Clone, Debug, Default)]
pub struct Player {
    pub idx: u8,
    pub life: i32,
    pub library: Vec<CIdx>,
    pub hand: Vec<CIdx>,
    pub graveyard: Vec<CIdx>,
    pub exile: Vec<CIdx>,
    /// Insertion-ordered (Python dict).
    pub pool: Vec<(u8, i32)>,
    pub drew_from_empty: bool,
    pub cards_drawn_this_turn: i32,
}

impl Player {
    pub fn zone(&self, z: Zone) -> &Vec<CIdx> {
        match z {
            Zone::Library => &self.library,
            Zone::Hand => &self.hand,
            Zone::Graveyard => &self.graveyard,
            Zone::Exile => &self.exile,
            _ => panic!("not a player zone: {z:?}"),
        }
    }
    pub fn zone_mut(&mut self, z: Zone) -> &mut Vec<CIdx> {
        match z {
            Zone::Library => &mut self.library,
            Zone::Hand => &mut self.hand,
            Zone::Graveyard => &mut self.graveyard,
            Zone::Exile => &mut self.exile,
            _ => panic!("not a player zone: {z:?}"),
        }
    }
    pub fn pool_get(&self, c: u8) -> i32 {
        self.pool.iter().find(|(k, _)| *k == c).map(|e| e.1).unwrap_or(0)
    }
    pub fn pool_add(&mut self, c: u8, n: i32) {
        match self.pool.iter_mut().find(|(k, _)| *k == c) {
            Some(e) => e.1 += n,
            None => self.pool.push((c, n)),
        }
    }
    /// `pool[c] -= 1; if pool[c] == 0: del pool[c]`
    pub fn pool_spend(&mut self, c: u8) {
        let pos = self.pool.iter().position(|(k, _)| *k == c).expect("spent mana not in pool");
        self.pool[pos].1 -= 1;
        if self.pool[pos].1 == 0 {
            self.pool.remove(pos);
        }
    }
}

// ---------------------------------------------------------------------------
// Decisions
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Priority,
    ChooseX,
    Target,
    PayMana,
    Sacrifice,
    ExileFromGy,
    YesNo,
    ChooseCard,
    Order,
    OrderTriggers,
    DeclareAttacker,
    DeclareBlocker,
    AssignDamage,
    ChooseMode,
    Mulligan,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Priority => "priority",
            Kind::ChooseX => "choose_x",
            Kind::Target => "target",
            Kind::PayMana => "pay_mana",
            Kind::Sacrifice => "sacrifice",
            Kind::ExileFromGy => "exile_from_graveyard",
            Kind::YesNo => "yes_no",
            Kind::ChooseCard => "choose_card",
            Kind::Order => "order",
            Kind::OrderTriggers => "order_triggers",
            Kind::DeclareAttacker => "declare_attacker",
            Kind::DeclareBlocker => "declare_blocker",
            Kind::AssignDamage => "assign_damage",
            Kind::ChooseMode => "choose_mode",
            Kind::Mulligan => "mulligan",
        }
    }
}

/// One element of an option key. Every string in a key is static: keywords
/// of the key grammar, card / ability / mode / trigger names (from the
/// static card database) or mana colours.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KI {
    S(&'static str),
    I(i64),
    N,
    T(Vec<i64>),
}

pub type Key = Vec<KI>;

#[derive(Clone, Debug)]
pub enum Val {
    None,
    Bool(bool),
    Int(i32),
    Pass,
    Land(CIdx),
    Cast(CIdx, Method, Option<u8>),
    Activate(CIdx, u8),
    Mana(CIdx, u8),
    Card(CIdx),
    Ref(Ref),
    Pool(u8),
    Source(CIdx, u8),
    Trigger(usize),
    Split(Vec<i32>),
    Group(usize),
    Order(Vec<CIdx>),
    Top,
    Bottom,
}

#[derive(Clone, Debug)]
pub struct Opt {
    pub label: String,
    pub key: Key,
    pub value: Val,
}

#[derive(Clone, Debug)]
pub struct Decision {
    pub player: u8,
    pub kind: Kind,
    pub prompt: String,
    pub options: Vec<Opt>,
}

pub fn color_str(c: u8) -> &'static str {
    match c {
        b'W' => "W",
        b'U' => "U",
        b'B' => "B",
        b'R' => "R",
        b'G' => "G",
        b'C' => "C",
        _ => panic!("bad colour"),
    }
}

/// Stops the engine: the game ended (CR 104) or an engine invariant broke.
#[derive(Clone, Debug)]
pub enum Stop {
    GameOver { winner: Option<u8>, reason: &'static str },
    Rules(String),
}

pub type R<T> = Result<T, Stop>;

pub fn rules<T>(msg: impl Into<String>) -> R<T> {
    Err(Stop::Rules(msg.into()))
}

/// Identity of a permanent for deduplication (`Game.equiv_key`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EquivKey {
    Unique(u32),
    State { face: DefId, controller: u8, is_token: bool, tapped: bool, damage: i32, deathtouch_damage: bool, counters: i32, sick: bool, skip_untap: i32, transformed: bool, temp: Vec<TempEffect>, attacking: bool },
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Args {
    pub decks: [Vec<String>; 2],
    pub seed: i128,
    pub starting_player: Option<u8>,
    pub auto_single: bool,
    pub max_turns: i32,
    pub log: bool,
    pub has_setup: bool,
    pub start_step: String,
    pub mulligans: bool,
    pub match_game: i32,
}

pub struct State {
    pub args: Args,
    pub match_game: i32,
    pub rng: PyRandom,
    pub auto_single: bool,
    pub max_turns: i32,
    pub logging: bool,
    pub log: Vec<String>,
    pub actions: Vec<u32>,
    pub next_id: u32,
    pub cards: Vec<Card>,
    pub players: [Player; 2],
    pub battlefield: Vec<CIdx>,
    pub stack: Vec<StackItem>,
    pub pending: Vec<PendingTrigger>,
    pub turn: i32,
    pub step_name: &'static str,
    pub lands_played: i32,
    pub attackers: Vec<u32>,
    pub blocked: Vec<u32>,
    /// blocker oid -> attacker oid, insertion ordered (Python dict).
    pub blocks: Vec<(u32, u32)>,
    pub winner: Option<u8>,
    pub over: bool,
    pub end_reason: &'static str,
    pub decision: Option<Decision>,
    pub active: u8,
    pub starting_player: u8,
    pub skip_first_draw: bool,
    pub mulligan_phase: bool,
    pub mulligans_taken: [i32; 2],
}

/// `str.capitalize()` (first character upper, the rest lower).
pub fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().chain(c.flat_map(|x| x.to_lowercase())).collect(),
    }
}

/// Python `repr()` of a str.
pub fn py_repr_str(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(q);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

impl State {
    pub fn new(args: Args) -> Result<State, String> {
        let dbase = db();
        let mut st = State {
            match_game: args.match_game,
            rng: PyRandom::new(args.seed),
            auto_single: args.auto_single,
            max_turns: args.max_turns,
            logging: args.log,
            log: vec![],
            actions: vec![],
            next_id: 1,
            cards: vec![],
            players: [Player { idx: 0, life: 20, ..Default::default() }, Player { idx: 1, life: 20, ..Default::default() }],
            battlefield: vec![],
            stack: vec![],
            pending: vec![],
            turn: 0,
            step_name: "untap",
            lands_played: 0,
            attackers: vec![],
            blocked: vec![],
            blocks: vec![],
            winner: None,
            over: false,
            end_reason: "",
            decision: None,
            active: 0,
            starting_player: 0,
            skip_first_draw: false,
            mulligan_phase: false,
            mulligans_taken: [0, 0],
            args: args.clone(),
        };
        for p in 0..2u8 {
            for name in &args.decks[p as usize] {
                let def = *dbase.cards.get(name).ok_or_else(|| format!("unknown card {name:?}"))?;
                let c = st.new_card(def, p, Zone::Library, false);
                st.players[p as usize].library.push(c);
            }
        }
        let sp = match args.starting_player {
            Some(s) => s,
            None => st.rng.randbelow(2) as u8,
        };
        st.active = sp;
        st.starting_player = sp;
        if !args.has_setup {
            for p in 0..2 {
                let mut lib = std::mem::take(&mut st.players[p].library);
                st.rng.shuffle(&mut lib);
                st.players[p].library = lib;
            }
            for p in 0..2 {
                st.draw(p, 7, false);
            }
        }
        st.skip_first_draw = !args.has_setup;
        st.mulligan_phase = !args.has_setup && args.mulligans;
        Ok(st)
    }

    #[inline]
    pub fn log_on(&self) -> bool {
        self.logging
    }

    pub fn push_log(&mut self, msg: String) {
        if self.logging {
            self.log.push(msg);
        }
    }

    pub fn new_id(&mut self) -> u32 {
        let i = self.next_id;
        self.next_id += 1;
        i
    }

    pub fn new_card(&mut self, def: DefId, owner: u8, zone: Zone, token: bool) -> CIdx {
        let i = self.new_id();
        let known_to = match zone {
            Zone::Battlefield | Zone::Graveyard | Zone::Exile | Zone::Stack => BOTH,
            _ => 0,
        };
        self.cards.push(Card {
            uid: i,
            oid: i,
            def,
            owner,
            controller: owner,
            zone,
            is_token: token,
            transformed: false,
            tapped: false,
            damage: 0,
            deathtouch_damage: false,
            counters: 0,
            sick: false,
            attached_to: None,
            skip_untap: 0,
            temp: vec![],
            known_to,
        });
        (self.cards.len() - 1) as CIdx
    }

    #[inline]
    pub fn c(&self, i: CIdx) -> &Card {
        &self.cards[i as usize]
    }
    #[inline]
    pub fn cm(&mut self, i: CIdx) -> &mut Card {
        &mut self.cards[i as usize]
    }

    pub fn src<'a>(&'a self, s: &'a Src) -> &'a Card {
        match s {
            Src::Live(i) => self.c(*i),
            Src::Snap(b) => b,
        }
    }

    /// `Game.add_card` for scenario setups.
    pub fn add_card(&mut self, name: &str, player: u8, zone: Zone, tapped: bool, sick: bool, counters: i32) -> Result<CIdx, String> {
        let dbase = db();
        let (def, token) = match dbase.cards.get(name) {
            Some(d) => (*d, false),
            None => (*dbase.tokens.get(name).ok_or_else(|| format!("unknown card {name:?}"))?, true),
        };
        let c = self.new_card(def, player, zone, token);
        if zone == Zone::Battlefield {
            let card = self.cm(c);
            card.tapped = tapped;
            card.sick = sick;
            card.counters = counters;
            self.battlefield.push(c);
        } else {
            if zone == Zone::Hand {
                self.cm(c).known_to = pbit(player);
            }
            self.players[player as usize].zone_mut(zone).push(c);
        }
        Ok(c)
    }

    pub fn empty_pools(&mut self) {
        for p in self.players.iter_mut() {
            p.pool.clear();
        }
    }

    pub fn sorcery_timing(&self, p: u8) -> bool {
        p == self.active && (self.step_name == "main1" || self.step_name == "main2") && self.stack.is_empty()
    }

    // ------------------------------------------------------------------
    // Characteristics
    // ------------------------------------------------------------------

    pub fn perm(&self, oid: u32) -> Option<CIdx> {
        self.battlefield.iter().copied().find(|&c| self.c(c).oid == oid)
    }

    pub fn live(&self, card: &Card) -> Option<CIdx> {
        self.perm(card.oid)
    }

    pub fn stack_pos(&self, sid: u32) -> Option<usize> {
        self.stack.iter().position(|it| it.sid == sid)
    }

    pub fn is_bestowed(&self, c: &Card) -> bool {
        c.zone == Zone::Battlefield && c.attached_to.is_some()
    }

    pub fn types(&self, c: &Card) -> u16 {
        let mut t = c.face().types;
        if self.is_bestowed(c) {
            t &= !T_CREATURE;
        }
        t
    }

    pub fn is_creature(&self, c: &Card) -> bool {
        c.face().types & T_CREATURE != 0 && !self.is_bestowed(c)
    }
    pub fn is_artifact(&self, c: &Card) -> bool {
        c.face().types & T_ARTIFACT != 0
    }
    pub fn is_land(&self, c: &Card) -> bool {
        c.face().types & T_LAND != 0
    }

    fn auras_on(&self, oid: u32) -> impl Iterator<Item = &Card> + '_ {
        self.battlefield.iter().map(move |&a| self.c(a)).filter(move |a| a.attached_to == Some(oid))
    }

    pub fn power(&self, c: &Card) -> i32 {
        c.face().power.unwrap_or(0) + c.counters + c.temp.iter().map(|t| t.power).sum::<i32>() + self.auras_on(c.oid).map(|a| a.counters).sum::<i32>()
    }

    pub fn toughness(&self, c: &Card) -> i32 {
        c.face().toughness.unwrap_or(0) + c.counters + c.temp.iter().map(|t| t.toughness).sum::<i32>() + self.auras_on(c.oid).map(|a| a.counters).sum::<i32>()
    }

    pub fn keywords(&self, c: &Card) -> u32 {
        let mut k = c.face().keywords;
        for t in &c.temp {
            k |= t.keywords;
        }
        if self.auras_on(c.oid).next().is_some() {
            let d = db();
            k |= d.kw("reach") | d.kw("trample");
        }
        k
    }

    pub fn has(&self, c: &Card, kw: &str) -> bool {
        self.keywords(c) & db().kw(kw) != 0
    }

    pub fn referenced_oids(&self) -> Vec<u32> {
        let mut refs = vec![];
        for it in &self.stack {
            for t in &it.targets {
                if let Ref::Perm(o) = t {
                    refs.push(*o);
                }
            }
        }
        for (b, a) in &self.blocks {
            refs.push(*b);
            refs.push(*a);
        }
        for &c in &self.battlefield {
            if let Some(a) = self.c(c).attached_to {
                refs.push(a);
            }
        }
        refs
    }

    pub fn equiv_key(&self, ci: CIdx, refs: &[u32]) -> EquivKey {
        let c = self.c(ci);
        if refs.contains(&c.oid) || c.attached_to.is_some() {
            return EquivKey::Unique(c.oid);
        }
        EquivKey::State {
            face: c.face().id,
            controller: c.controller,
            is_token: c.is_token,
            tapped: c.tapped,
            damage: c.damage,
            deathtouch_damage: c.deathtouch_damage,
            counters: c.counters,
            sick: c.sick,
            skip_untap: c.skip_untap,
            transformed: c.transformed,
            temp: c.temp.clone(),
            attacking: self.attackers.contains(&c.oid),
        }
    }

    pub fn dedupe_by_equiv(&self, cards: impl IntoIterator<Item = CIdx>) -> Vec<CIdx> {
        let refs = self.referenced_oids();
        let mut seen: Vec<EquivKey> = vec![];
        let mut out = vec![];
        for c in cards {
            let k = self.equiv_key(c, &refs);
            if !seen.contains(&k) {
                seen.push(k);
                out.push(c);
            }
        }
        out
    }

    pub fn dedupe_by_name(&self, cards: impl IntoIterator<Item = CIdx>) -> Vec<CIdx> {
        let mut seen: Vec<&'static str> = vec![];
        let mut out = vec![];
        for c in cards {
            let n = self.c(c).name();
            if !seen.contains(&n) {
                seen.push(n);
                out.push(c);
            }
        }
        out
    }

    pub fn hand_card_options(&self, p: u8, verb: &'static str) -> Vec<Opt> {
        let cap = capitalize(verb);
        self.dedupe_by_name(self.players[p as usize].hand.iter().copied())
            .into_iter()
            .map(|c| {
                let n = self.c(c).name();
                Opt { label: format!("{cap} {n}"), key: vec![KI::S(verb), KI::S(n)], value: Val::Card(c) }
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // Zones
    // ------------------------------------------------------------------

    /// `Game._move`. Returns the card (now a new object) or None for a token
    /// that ceased to exist.
    pub fn move_card(&mut self, ci: CIdx, to: Zone, controller: Option<u8>, position: Pos, known_to: Option<u8>, tapped: bool) -> Option<CIdx> {
        let frm = self.c(ci).zone;
        let lki = if frm == Zone::Battlefield { Some(self.c(ci).clone()) } else { None };
        if frm == Zone::Battlefield {
            let pos = self.battlefield.iter().position(|&x| x == ci).expect("card not on battlefield");
            self.battlefield.remove(pos);
            let oid = self.c(ci).oid;
            self.remove_from_combat(oid);
        } else if frm != Zone::Stack {
            let owner = self.c(ci).owner as usize;
            let list = self.players[owner].zone_mut(frm);
            let pos = list.iter().position(|&x| x == ci).expect("card not in its zone");
            list.remove(pos);
        }
        let prev_known = self.c(ci).known_to;
        if self.c(ci).is_token && to != Zone::Battlefield {
            self.cm(ci).zone = Zone::Gone;
            if let Some(l) = lki {
                self.after_leave_battlefield(l, ci, to);
            }
            return None;
        }
        let oid = self.new_id();
        {
            let c = self.cm(ci);
            c.reset_state();
            c.oid = oid;
            c.zone = to;
            c.controller = controller.unwrap_or(c.owner);
        }
        let owner = self.c(ci).owner as usize;
        match to {
            Zone::Battlefield => {
                let c = self.cm(ci);
                c.known_to = BOTH;
                c.sick = true;
                c.tapped = tapped || c.face().enters_tapped;
                self.battlefield.push(ci);
                if self.logging {
                    let c = self.c(ci);
                    let m = format!("enters: {}#{} (p{})", c.name(), c.oid, c.controller);
                    self.log.push(m);
                }
            }
            Zone::Library => {
                self.cm(ci).known_to = known_to.unwrap_or(0);
                let lib = &mut self.players[owner].library;
                match position {
                    Pos::Top => lib.insert(0, ci),
                    Pos::Bottom => lib.push(ci),
                    Pos::At(i) => {
                        let i = i.min(lib.len());
                        lib.insert(i, ci)
                    }
                }
            }
            Zone::Hand => {
                self.cm(ci).known_to = prev_known | pbit(owner as u8) | known_to.unwrap_or(0);
                self.players[owner].hand.push(ci);
            }
            Zone::Graveyard | Zone::Exile => {
                self.cm(ci).known_to = BOTH;
                self.players[owner].zone_mut(to).push(ci);
            }
            Zone::Stack => {
                self.cm(ci).known_to = BOTH;
            }
            Zone::Gone => panic!("cannot move to gone"),
        }
        if let Some(l) = lki {
            self.after_leave_battlefield(l, ci, to);
        }
        Some(ci)
    }

    pub fn mv(&mut self, ci: CIdx, to: Zone) -> Option<CIdx> {
        self.move_card(ci, to, None, Pos::Top, None, false)
    }

    fn after_leave_battlefield(&mut self, lki: Card, new: CIdx, to: Zone) {
        if self.logging {
            let m = format!("leaves: {}#{} (p{}) -> {}", lki.name(), lki.oid, lki.controller, to.name());
            self.log.push(m);
        }
        if to == Zone::Graveyard {
            let face = lki.face();
            let new_oid = self.c(new).oid;
            for t in &face.triggers {
                if t.event == Event::ToGraveyardFromBattlefield {
                    self.pending.push(PendingTrigger {
                        controller: lki.controller,
                        source: Src::Snap(Box::new(lki.clone())),
                        tdef: t,
                        data: Data { card: Some(new), oid: Some(new_oid), ..Default::default() },
                    });
                }
            }
        }
    }

    pub fn draw(&mut self, p: usize, n: i32, count: bool) {
        for _ in 0..n {
            if self.players[p].library.is_empty() {
                self.players[p].drew_from_empty = true;
                self.push_log_lazy(|_| format!("p{p} draws from an empty library"));
                continue;
            }
            let card = self.players[p].library[0];
            self.mv(card, Zone::Hand);
            if count {
                self.players[p].cards_drawn_this_turn += 1;
            }
        }
    }

    #[inline]
    pub fn push_log_lazy(&mut self, f: impl FnOnce(&State) -> String) {
        if self.logging {
            let m = f(self);
            self.log.push(m);
        }
    }

    pub fn mill(&mut self, p: usize, n: i32) {
        for _ in 0..n {
            if self.players[p].library.is_empty() {
                return;
            }
            let c = self.players[p].library[0];
            self.mv(c, Zone::Graveyard);
        }
    }

    pub fn discard(&mut self, ci: CIdx) {
        if self.logging {
            let c = self.c(ci);
            let m = format!("p{} discards {}", c.owner, c.name());
            self.log.push(m);
        }
        self.mv(ci, Zone::Graveyard);
    }

    pub fn shuffle(&mut self, p: usize) {
        let mut lib = std::mem::take(&mut self.players[p].library);
        self.rng.shuffle(&mut lib);
        for &c in &lib {
            self.cards[c as usize].known_to = 0;
        }
        self.players[p].library = lib;
    }

    pub fn sacrifice(&mut self, ci: CIdx) {
        let lki = self.c(ci).clone();
        self.push_log_lazy(|_| format!("p{} sacrifices {}#{}", lki.controller, lki.name(), lki.oid));
        self.mv(ci, Zone::Graveyard);
        self.emit_sacrifice(&lki);
    }

    pub fn destroy(&mut self, ci: CIdx) -> bool {
        if self.has(self.c(ci), "indestructible") {
            return false;
        }
        self.mv(ci, Zone::Graveyard);
        true
    }

    pub fn create_token(&mut self, p: u8, def: DefId) -> CIdx {
        let c = self.new_card(def, p, Zone::Battlefield, true);
        self.cm(c).sick = true;
        self.battlefield.push(c);
        if self.logging {
            let card = self.c(c);
            let m = format!("enters: {}#{} (p{p})", card.name(), card.oid);
            self.log.push(m);
        }
        self.emit_etb(c);
        c
    }

    pub fn put_onto_battlefield(&mut self, ci: CIdx, controller: u8, tapped: bool) -> CIdx {
        let new = self.move_card(ci, Zone::Battlefield, Some(controller), Pos::Top, None, tapped).expect("token cannot be put onto the battlefield from elsewhere");
        self.emit_etb(new);
        new
    }

    pub fn deal_damage(&mut self, source: &Card, target: Ref, amount: i32) {
        if amount <= 0 {
            return;
        }
        let kws = self.keywords(source);
        let d = db();
        match target {
            Ref::Player(i) => self.players[i as usize].life -= amount,
            Ref::Perm(oid) => {
                let c = match self.perm(oid) {
                    Some(c) if self.is_creature(self.c(c)) => c,
                    _ => return,
                };
                self.cm(c).damage += amount;
                if kws & d.kw("deathtouch") != 0 {
                    self.cm(c).deathtouch_damage = true;
                }
            }
            Ref::Stack(_) => panic!("damage to a stack item"),
        }
        if kws & d.kw("lifelink") != 0 {
            self.players[source.controller as usize].life += amount;
        }
    }

    // ------------------------------------------------------------------
    // Events and triggers
    // ------------------------------------------------------------------

    pub fn emit_etb(&mut self, ci: CIdx) {
        let c = self.c(ci);
        let controller = c.controller;
        for t in &c.face().triggers {
            if t.event == Event::Etb {
                self.pending.push(PendingTrigger { controller, source: Src::Live(ci), tdef: t, data: Data::default() });
            }
        }
    }

    pub fn emit_sacrifice(&mut self, lki: &Card) {
        let mut new = vec![];
        for &perm in &self.battlefield {
            let pc = self.c(perm);
            if pc.controller != lki.controller || pc.oid == lki.oid {
                continue;
            }
            for t in &pc.face().triggers {
                if t.event == Event::YouSacrificeAnother && t.sacrificed_subtype.as_ref().map_or(true, |s| lki.face().has_subtype(s)) {
                    new.push(PendingTrigger { controller: pc.controller, source: Src::Live(perm), tdef: t, data: Data { sacrificed: Some(Box::new(lki.clone())), ..Default::default() } });
                }
            }
        }
        self.pending.extend(new);
    }

    pub fn emit_upkeep(&mut self) {
        let mut new = vec![];
        for &perm in &self.battlefield {
            let pc = self.c(perm);
            if pc.controller != self.active {
                continue;
            }
            for t in &pc.face().triggers {
                if t.event == Event::YourUpkeep {
                    new.push(PendingTrigger { controller: pc.controller, source: Src::Live(perm), tdef: t, data: Data::default() });
                }
            }
        }
        self.pending.extend(new);
    }

    pub fn emit_cast(&mut self, sid: u32) {
        let it = &self.stack[self.stack_pos(sid).unwrap()];
        let card = it.card.unwrap();
        let controller = it.controller;
        let mut new = vec![];
        for t in &self.c(card).face().triggers {
            if t.event == Event::Cast {
                new.push(PendingTrigger { controller, source: Src::Live(card), tdef: t, data: Data { spell_sid: Some(sid), ..Default::default() } });
            }
        }
        self.pending.extend(new);
    }

    pub fn emit_targeted(&mut self, sid: u32) {
        let it = &self.stack[self.stack_pos(sid).unwrap()];
        let controller = it.controller;
        let targets = it.targets.clone();
        for r in targets {
            if let Ref::Perm(oid) = r {
                if let Some(c) = self.perm(oid) {
                    let card = self.c(c);
                    let ward = card.face().ward;
                    if ward != 0 && controller != card.controller {
                        let cc = card.controller;
                        self.pending.push(PendingTrigger { controller: cc, source: Src::Live(c), tdef: &db().ward, data: Data { sid: Some(sid), amount: Some(ward), ..Default::default() } });
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // State-based actions
    // ------------------------------------------------------------------

    pub fn sba(&mut self) -> R<bool> {
        let mut changed = false;
        let losers: Vec<u8> = self.players.iter().filter(|p| p.life <= 0 || p.drew_from_empty).map(|p| p.idx).collect();
        if !losers.is_empty() {
            let winner = if losers.len() == 2 { None } else { Some(1 - losers[0]) };
            let reason = if losers.iter().any(|&i| self.players[i as usize].life <= 0) { "life" } else { "decking" };
            return Err(Stop::GameOver { winner, reason });
        }
        for i in 0..self.battlefield.len() {
            let ci = self.battlefield[i];
            if let Some(a) = self.c(ci).attached_to {
                let ok = match self.perm(a) {
                    Some(h) => self.is_creature(self.c(h)),
                    None => false,
                };
                if !ok {
                    self.cm(ci).attached_to = None;
                    changed = true;
                }
            }
        }
        let mut dying = vec![];
        for &ci in &self.battlefield {
            let c = self.c(ci);
            if !self.is_creature(c) {
                continue;
            }
            let t = self.toughness(c);
            if t <= 0 || ((c.damage >= t || (c.deathtouch_damage && c.damage > 0)) && !self.has(c, "indestructible")) {
                dying.push(ci);
            }
        }
        for ci in dying {
            if self.logging {
                let n = self.c(ci).name();
                self.log.push(format!("SBA: {n} dies"));
            }
            self.mv(ci, Zone::Graveyard);
            changed = true;
        }
        Ok(changed)
    }

    // ------------------------------------------------------------------
    // Targets
    // ------------------------------------------------------------------

    pub fn target_candidates(&self, spec: TK, controller: u8, exclude_sid: Option<u32>) -> Vec<Ref> {
        match spec {
            TK::Player => return vec![Ref::Player(controller), Ref::Player(1 - controller)],
            TK::Opponent => return vec![Ref::Player(1 - controller)],
            _ => {}
        }
        if spec.is_spell() {
            return self.stack.iter().filter(|it| it.kind == SKind::Spell && Some(it.sid) != exclude_sid && self.spell_matches(spec, it)).map(|it| Ref::Stack(it.sid)).collect();
        }
        let mut out: Vec<Ref> = self.battlefield.iter().filter(|&&c| self.perm_matches(spec, self.c(c), controller)).map(|&c| Ref::Perm(self.c(c).oid)).collect();
        if spec == TK::Any {
            out.push(Ref::Player(controller));
            out.push(Ref::Player(1 - controller));
        }
        out
    }

    pub fn perm_matches(&self, spec: TK, c: &Card, controller: u8) -> bool {
        match spec {
            TK::Creature | TK::Any => self.is_creature(c),
            TK::NonlegendaryCreature => self.is_creature(c) && !c.face().has_supertype("Legendary"),
            TK::CreatureYouControl => self.is_creature(c) && c.controller == controller,
            TK::Land => self.is_land(c),
            TK::NonlandPermanent => !self.is_land(c),
            TK::NonartifactCreature => self.is_creature(c) && !self.is_artifact(c),
            TK::Artifact => self.is_artifact(c),
            TK::BluePermanent => c.face().colors & crate::cards::color_bit(b'U') != 0,
            TK::RedPermanent => c.face().colors & crate::cards::color_bit(b'R') != 0,
            _ => false,
        }
    }

    pub fn spell_matches(&self, spec: TK, it: &StackItem) -> bool {
        let d = self.c(it.card.unwrap()).face();
        match spec {
            TK::Spell => true,
            TK::BlueSpell => d.colors & crate::cards::color_bit(b'U') != 0,
            TK::RedSpell => d.colors & crate::cards::color_bit(b'R') != 0,
            TK::InstantSpell => d.is_type(T_INSTANT),
            TK::ArtifactSpell => d.is_type(T_ARTIFACT),
            _ => false,
        }
    }

    pub fn target_legal(&self, item: &StackItem, i: usize) -> bool {
        let spec = item.target_specs[i];
        match item.targets[i] {
            Ref::Player(p) => matches!(spec, TK::Player | TK::Any) || (spec == TK::Opponent && p != item.controller),
            Ref::Stack(sid) => match self.stack_pos(sid) {
                Some(pos) => {
                    let it = &self.stack[pos];
                    it.kind == SKind::Spell && spec.is_spell() && self.spell_matches(spec, it)
                }
                None => false,
            },
            Ref::Perm(oid) => match self.perm(oid) {
                Some(c) => self.perm_matches(spec, self.c(c), item.controller),
                None => false,
            },
        }
    }

    /// `Game.target(item, i)`: the still-legal target.
    pub fn target(&self, item: &StackItem, i: usize) -> Option<Tgt> {
        if i >= item.targets.len() || !self.target_legal(item, i) {
            return None;
        }
        Some(match item.targets[i] {
            Ref::Perm(oid) => Tgt::Card(self.perm(oid).unwrap()),
            Ref::Stack(sid) => Tgt::Spell(sid),
            Ref::Player(p) => Tgt::Player(p),
        })
    }

    pub fn describe_ref(&self, r: Ref, viewer: u8) -> (String, Key) {
        let rel = |p: u8| if p == viewer { "self" } else { "opponent" };
        match r {
            Ref::Player(p) => (format!("player {p} ({})", rel(p)), vec![KI::S("player"), KI::S(rel(p))]),
            Ref::Stack(sid) => {
                let it = &self.stack[self.stack_pos(sid).expect("described stack item")];
                (format!("spell {}#{} ({})", it.name, it.sid, rel(it.controller)), vec![KI::S("spell"), KI::S(rel(it.controller)), KI::S(self.static_item_name(it))])
            }
            Ref::Perm(oid) => {
                let c = self.c(self.perm(oid).expect("described permanent"));
                (format!("{}#{} ({})", c.name(), c.oid, rel(c.controller)), vec![KI::S("perm"), KI::S(rel(c.controller)), KI::S(c.name())])
            }
        }
    }

    /// A spell's stack name as a static string (its card name at cast time).
    fn static_item_name(&self, it: &StackItem) -> &'static str {
        let d = db();
        match d.cards.get(&it.name).or_else(|| d.tokens.get(&it.name)) {
            Some(id) => &d.def(*id).name,
            None => d.defs.iter().find(|x| x.name == it.name).map(|x| x.name.as_str()).expect("spell name"),
        }
    }

    // ------------------------------------------------------------------
    // Costs and mana
    // ------------------------------------------------------------------

    /// Untapped mana sources: (card, index of its mana ability).
    pub fn mana_sources(&self, p: u8, exclude: &[u32]) -> Vec<(CIdx, usize)> {
        let mut out = vec![];
        for &ci in &self.battlefield {
            let c = self.c(ci);
            if c.controller != p || exclude.contains(&c.oid) {
                continue;
            }
            for (i, ab) in c.face().abilities.iter().enumerate() {
                if ab.mana.is_none() {
                    continue;
                }
                if ab.tap && (c.tapped || (self.is_creature(c) && c.sick)) {
                    continue;
                }
                out.push((ci, i));
                break;
            }
        }
        out
    }

    pub fn sac_candidates(&self, p: u8, flt: SacFilter, exclude: &[u32]) -> Vec<CIdx> {
        self.battlefield
            .iter()
            .copied()
            .filter(|&ci| {
                let c = self.c(ci);
                if c.controller != p || exclude.contains(&c.oid) {
                    return false;
                }
                match flt {
                    SacFilter::Artifact => self.is_artifact(c),
                    SacFilter::ArtifactOrCreature => self.is_artifact(c) || self.is_creature(c),
                }
            })
            .collect()
    }

    /// `Game._cost_feasible`. `pool`: None = the player's pool.
    pub fn cost_feasible(&self, p: u8, rem: &Remaining, sac_filter: Option<SacFilter>, exclude: &[u32], pool: Option<&[(u8, i32)]>, gone: &[u32]) -> bool {
        let pool = pool.unwrap_or(&self.players[p as usize].pool);
        let mut excl: Vec<u32> = exclude.to_vec();
        excl.extend_from_slice(gone);
        let sources = self.mana_sources(p, &excl);
        let mut base: Vec<u8> = vec![];
        for (c, n) in pool {
            for _ in 0..*n {
                base.push(bit(*c));
            }
        }
        let mut selfsac: Vec<CIdx> = vec![];
        for (ci, ai) in &sources {
            let ab = &self.c(*ci).face().abilities[*ai];
            if ab.sac_self {
                selfsac.push(*ci);
            } else {
                base.push(ab.mana.as_ref().unwrap().iter().fold(0, |m, c| m | bit(*c)));
            }
        }
        match sac_filter {
            None => {
                for &ci in &selfsac {
                    let ab = self.c(ci).face().abilities.iter().find(|a| a.mana.is_some()).unwrap();
                    base.push(ab.mana.as_ref().unwrap().iter().fold(0, |m, c| m | bit(*c)));
                }
                can_pay(rem, &base)
            }
            Some(flt) => {
                let cands: Vec<u32> = self.sac_candidates(p, flt, gone).iter().map(|&c| self.c(c).oid).collect();
                if cands.is_empty() {
                    return false;
                }
                for &ci in &selfsac {
                    let ab = self.c(ci).face().abilities.iter().find(|a| a.mana.is_some()).unwrap();
                    assert!(ab.mana.as_deref() == Some(&[b'C'][..]), "self-sacrificing mana sources must produce {{C}}");
                }
                let mut ordered: Vec<u32> = selfsac.iter().map(|&c| self.c(c).oid).filter(|o| !cands.contains(o)).collect();
                ordered.extend(selfsac.iter().map(|&c| self.c(c).oid).filter(|o| cands.contains(o)));
                for k in 0..=ordered.len() {
                    let used = &ordered[..k];
                    let left = cands.iter().filter(|o| !used.contains(o)).count();
                    if left >= 1 {
                        let mut units = base.clone();
                        units.extend(std::iter::repeat(bit(b'C')).take(k));
                        if can_pay(rem, &units) {
                            return true;
                        }
                    }
                }
                false
            }
        }
    }

    pub fn can_afford(&self, p: u8, cost: &ManaCost) -> bool {
        self.cost_feasible(p, &Remaining::of(cost), None, &[], None, &[])
    }

    pub fn activate_mana_ability(&mut self, ci: CIdx, ai: usize) {
        let ab = &self.c(ci).face().abilities[ai];
        let (tap, sac) = (ab.tap, ab.sac_self);
        if tap {
            self.cm(ci).tapped = true;
        }
        if sac {
            self.sacrifice(ci);
        }
    }

    // ------------------------------------------------------------------
    // Casting checks
    // ------------------------------------------------------------------

    pub fn mode_cost(&self, ci: CIdx, mode: Method) -> Option<&'static ManaCost> {
        let d = self.c(ci).face();
        match mode {
            Method::Normal => Some(&d.cost),
            Method::Bestow => d.bestow.as_ref(),
            Method::Flashback => d.flashback.as_ref(),
            Method::Escape => d.escape.as_ref(),
        }
    }

    pub fn mode_targets(&self, ci: CIdx, mode: Method, choice: Option<u8>) -> Vec<TK> {
        if mode == Method::Bestow {
            return vec![TK::Creature];
        }
        let d = self.c(ci).face();
        match choice {
            Some(i) => d.modes[i as usize].targets.clone(),
            None => d.targets.clone(),
        }
    }

    pub fn cost_reduction(&self, p: u8, ci: CIdx) -> i32 {
        match self.c(ci).face().cost_reduction {
            None => 0,
            Some(CostRed::InstantsAndSorceriesInGraveyard) => {
                self.players[p as usize].graveyard.iter().filter(|&&c| self.c(c).face().types & (T_INSTANT | T_SORCERY) != 0).count() as i32
            }
            Some(CostRed::ArtifactsYouControl) => self.battlefield.iter().filter(|&&c| self.c(c).controller == p && self.is_artifact(self.c(c))).count() as i32,
            Some(CostRed::CardsDrawnThisTurn) => self.players[p as usize].cards_drawn_this_turn,
        }
    }

    pub fn can_cast(&self, p: u8, ci: CIdx, mode: Method, choice: Option<u8>) -> bool {
        let c = self.c(ci);
        let d = c.face();
        let base = match self.mode_cost(ci, mode) {
            Some(b) => b,
            None => return false,
        };
        if matches!(mode, Method::Normal | Method::Bestow) && c.zone != Zone::Hand {
            return false;
        }
        if matches!(mode, Method::Flashback | Method::Escape) && c.zone != Zone::Graveyard {
            return false;
        }
        if d.is_type(T_LAND) {
            return false;
        }
        if !d.is_type(T_INSTANT) && !self.sorcery_timing(p) {
            return false;
        }
        for spec in self.mode_targets(ci, mode, choice) {
            if self.target_candidates(spec, p, None).is_empty() {
                return false;
            }
        }
        if mode == Method::Escape && (self.players[p as usize].graveyard.len() as i32) - 1 < d.escape_exile {
            return false;
        }
        let cost = base.with_x(0).reduced(self.cost_reduction(p, ci));
        self.cost_feasible(p, &Remaining::of(&cost), d.additional_sac, &[], None, &[])
    }

    pub fn can_activate(&self, p: u8, ci: CIdx, ai: usize) -> bool {
        let c = self.c(ci);
        let ab = &c.face().abilities[ai];
        if ab.zone_hand {
            if c.zone != Zone::Hand {
                return false;
            }
        } else if c.zone != Zone::Battlefield || c.controller != p {
            return false;
        }
        if ab.sorcery_speed && !self.sorcery_timing(p) {
            return false;
        }
        if ab.tap && (c.tapped || (self.is_creature(c) && c.sick)) {
            return false;
        }
        if ab.mana.is_some() {
            return true;
        }
        for spec in &ab.targets {
            if self.target_candidates(*spec, p, None).is_empty() {
                return false;
            }
        }
        let exclude: Vec<u32> = if ab.tap { vec![c.oid] } else { vec![] };
        self.cost_feasible(p, &Remaining::of(&ab.cost), ab.sac_other, &exclude, None, &[])
    }

    // ------------------------------------------------------------------
    // Priority options
    // ------------------------------------------------------------------

    pub fn priority_options(&self, p: u8) -> Vec<Opt> {
        let pl = &self.players[p as usize];
        let mut opts = vec![Opt { label: "Pass priority".into(), key: vec![KI::S("pass")], value: Val::Pass }];
        let sorcery_ok = self.sorcery_timing(p);
        if sorcery_ok && self.lands_played < 1 {
            for card in self.dedupe_by_name(pl.hand.iter().copied().filter(|&c| self.c(c).face().is_type(T_LAND))) {
                let n = self.c(card).name();
                opts.push(Opt { label: format!("Play {n}"), key: vec![KI::S("play_land"), KI::S(n)], value: Val::Land(card) });
            }
        }
        let hand = self.dedupe_by_name(pl.hand.iter().copied());
        for &card in &hand {
            let face = self.c(card).face();
            let n = face.name.as_str();
            for (i, sm) in face.modes.iter().enumerate() {
                if self.can_cast(p, card, Method::Normal, Some(i as u8)) {
                    opts.push(Opt {
                        label: format!("Cast {n} ({})", sm.name),
                        key: vec![KI::S("cast"), KI::S(n), KI::S("hand"), KI::S("normal"), KI::S(sm.name.as_str())],
                        value: Val::Cast(card, Method::Normal, Some(i as u8)),
                    });
                }
            }
            if !face.modes.is_empty() {
                continue;
            }
            for mode in [Method::Normal, Method::Bestow] {
                if self.can_cast(p, card, mode, None) {
                    let label = if mode == Method::Normal { format!("Cast {n}") } else { format!("Cast {n} ({})", mode.name()) };
                    opts.push(Opt { label, key: vec![KI::S("cast"), KI::S(n), KI::S("hand"), KI::S(mode.name())], value: Val::Cast(card, mode, None) });
                }
            }
        }
        for card in self.dedupe_by_name(pl.graveyard.iter().copied()) {
            let n = self.c(card).name();
            for mode in [Method::Flashback, Method::Escape] {
                if self.can_cast(p, card, mode, None) {
                    opts.push(Opt {
                        label: format!("Cast {n} ({})", mode.name()),
                        key: vec![KI::S("cast"), KI::S(n), KI::S("graveyard"), KI::S(mode.name())],
                        value: Val::Cast(card, mode, None),
                    });
                }
            }
        }
        for &card in &hand {
            let face = self.c(card).face();
            for (i, ab) in face.abilities.iter().enumerate() {
                if ab.zone_hand && self.can_activate(p, card, i) {
                    opts.push(Opt {
                        label: format!("{}: {}", face.name, ab.name),
                        key: vec![KI::S("activate"), KI::S(face.name.as_str()), KI::S(ab.name.as_str())],
                        value: Val::Activate(card, i as u8),
                    });
                }
            }
        }
        for card in self.dedupe_by_equiv(self.battlefield.iter().copied().filter(|&c| self.c(c).controller == p)) {
            let face = self.c(card).face();
            for (i, ab) in face.abilities.iter().enumerate() {
                if ab.zone_hand {
                    continue;
                }
                if ab.mana.is_some() {
                    if ab.sac_self && self.can_activate(p, card, i) {
                        opts.push(Opt {
                            label: format!("{}: {}", face.name, ab.name),
                            key: vec![KI::S("mana"), KI::S(face.name.as_str()), KI::S(ab.name.as_str())],
                            value: Val::Mana(card, i as u8),
                        });
                    }
                } else if self.can_activate(p, card, i) {
                    opts.push(Opt {
                        label: format!("{}: {}", face.name, ab.name),
                        key: vec![KI::S("activate"), KI::S(face.name.as_str()), KI::S(ab.name.as_str())],
                        value: Val::Activate(card, i as u8),
                    });
                }
            }
        }
        opts
    }

    // ------------------------------------------------------------------
    // Combat helpers
    // ------------------------------------------------------------------

    pub fn clear_combat(&mut self) {
        self.attackers.clear();
        self.blocked.clear();
        self.blocks.clear();
    }

    pub fn remove_from_combat(&mut self, oid: u32) {
        if let Some(pos) = self.attackers.iter().position(|&a| a == oid) {
            self.attackers.remove(pos);
        }
        self.blocks.retain(|&(b, _)| b != oid);
        self.blocks.retain(|&(_, a)| a != oid);
    }

    pub fn can_block(&self, blocker: &Card, attacker: &Card) -> bool {
        !(self.has(attacker, "flying") && !(self.has(blocker, "flying") || self.has(blocker, "reach")))
    }

    pub fn lethal(&self, attacker: &Card, blocker: &Card) -> i32 {
        if self.has(attacker, "deathtouch") {
            return 1;
        }
        (self.toughness(blocker) - blocker.damage).max(0)
    }

    pub fn untap_step(&mut self) {
        let active = self.active;
        for i in 0..self.battlefield.len() {
            let ci = self.battlefield[i];
            let c = self.cm(ci);
            if c.controller != active {
                continue;
            }
            c.sick = false;
            if c.skip_untap > 0 {
                c.skip_untap -= 1;
            } else {
                c.tapped = false;
            }
        }
    }

    pub fn begin_turn(&mut self) {
        self.lands_played = 0;
        for p in self.players.iter_mut() {
            p.cards_drawn_this_turn = 0;
        }
        self.clear_combat();
        if self.logging {
            let m = format!("=== Turn {}: player {} ===", self.turn, self.active);
            self.log.push(m);
        }
    }
}

/// Library position for `move_card`.
#[derive(Clone, Copy, Debug)]
pub enum Pos {
    Top,
    Bottom,
    At(usize),
}

/// A resolved target (`Game.target`).
#[derive(Clone, Copy, Debug)]
pub enum Tgt {
    Card(CIdx),
    Spell(u32),
    Player(u8),
}

/// All tuples of `parts` non-negative ints summing to `total`, in the order
/// of game.py's `_compositions`.
pub fn compositions(total: i32, parts: usize) -> Vec<Vec<i32>> {
    if parts == 1 {
        return vec![vec![total]];
    }
    let mut out = vec![];
    for first in 0..=total {
        for rest in compositions(total - first, parts - 1) {
            let mut v = Vec::with_capacity(parts);
            v.push(first);
            v.extend(rest);
            out.push(v);
        }
    }
    out
}

pub fn is_instant_or_sorcery(d: &CardDef) -> bool {
    d.types & (T_INSTANT | T_SORCERY) != 0
}
