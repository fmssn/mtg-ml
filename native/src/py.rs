//! Python bindings (`mtg_ml_native`). The friendly, Game-compatible API is
//! the pure-Python wrapper `mtg_ml.engine.native.NativeGame`; this module
//! exposes the raw pieces it is built from.

use pyo3::create_exception;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::cards::{self, db, type_names, SacFilter, TK};
use crate::game::{Game, StepError};
use crate::state::*;

create_exception!(mtg_ml_native, NativeRulesError, pyo3::exceptions::PyException);

fn rel(p: u8, viewer: u8) -> &'static str {
    if p == viewer {
        "self"
    } else {
        "opponent"
    }
}

fn key_to_py<'py>(py: Python<'py>, key: &Key) -> Bound<'py, PyTuple> {
    let items: Vec<PyObject> = key
        .iter()
        .map(|k| match k {
            KI::S(s) => s.into_py(py),
            KI::I(i) => i.into_py(py),
            KI::N => py.None(),
            KI::T(v) => PyTuple::new_bound(py, v).into_py(py),
        })
        .collect();
    PyTuple::new_bound(py, items)
}

fn ref_to_py(py: Python<'_>, r: Ref) -> PyObject {
    match r {
        Ref::Player(p) => ("player", p).into_py(py),
        Ref::Perm(o) => ("perm", o).into_py(py),
        Ref::Stack(s) => ("stack", s).into_py(py),
    }
}

fn known_list(k: u8) -> Vec<u8> {
    (0..2).filter(|p| k & (1 << p) != 0).collect()
}

fn card_tuple(py: Python<'_>, c: &Card) -> PyObject {
    let d = db();
    let temp: Vec<PyObject> = c.temp.iter().map(|t| (d.keyword_list(t.keywords), t.power, t.toughness).into_py(py)).collect();
    let fields: Vec<PyObject> = vec![
        c.uid.into_py(py),
        c.oid.into_py(py),
        c.name().into_py(py),
        c.defn().name.as_str().into_py(py),
        c.owner.into_py(py),
        c.controller.into_py(py),
        c.zone.name().into_py(py),
        c.is_token.into_py(py),
        c.transformed.into_py(py),
        c.tapped.into_py(py),
        c.damage.into_py(py),
        c.deathtouch_damage.into_py(py),
        c.counters.into_py(py),
        c.sick.into_py(py),
        c.attached_to.into_py(py),
        c.skip_untap.into_py(py),
        temp.into_py(py),
        known_list(c.known_to).into_py(py),
    ];
    PyTuple::new_bound(py, fields).into_py(py)
}

fn data_list(py: Python<'_>, st: &State, d: &Data) -> PyObject {
    let mut v: Vec<(&str, PyObject)> = vec![];
    if let Some(c) = d.card {
        v.push(("card", card_tuple(py, st.c(c))));
    }
    if let Some(a) = d.amount {
        v.push(("amount", a.into_py(py)));
    }
    if let Some(o) = d.oid {
        v.push(("oid", o.into_py(py)));
    }
    if let Some(s) = &d.sacrificed {
        v.push(("sacrificed", card_tuple(py, s)));
    }
    if let Some(s) = d.sid {
        v.push(("sid", s.into_py(py)));
    }
    if let Some(s) = d.source_oid {
        v.push(("source_oid", s.into_py(py)));
    }
    if let Some(s) = d.spell_sid {
        v.push(("spell_sid", s.into_py(py)));
    }
    v.sort_by(|a, b| a.0.cmp(b.0));
    v.into_py(py)
}

fn sac_filter(s: &str) -> PyResult<SacFilter> {
    match s {
        "artifact" => Ok(SacFilter::Artifact),
        "artifact_or_creature" => Ok(SacFilter::ArtifactOrCreature),
        _ => Err(PyValueError::new_err(format!("unknown sacrifice filter {s:?}"))),
    }
}

#[pyclass(unsendable, module = "mtg_ml_native", name = "Game")]
pub struct PyGame {
    g: Game,
    /// Bumped by every step and every mutation; Python-side caches key on it.
    version: u64,
}

impl PyGame {
    fn st(&self) -> &State {
        self.g.state()
    }
    fn mutate(&mut self) -> &mut State {
        self.version += 1;
        self.g.state_mut()
    }
    fn card(&self, c: CIdx) -> PyResult<&Card> {
        self.st().cards.get(c as usize).ok_or_else(|| PyIndexError::new_err("bad card index"))
    }
    fn step_err(e: StepError) -> PyErr {
        match e {
            StepError::Over => NativeRulesError::new_err("game is over"),
            StepError::Index(i, n) => PyIndexError::new_err(format!("option {i} out of range ({n})")),
            StepError::Rules(m) => NativeRulesError::new_err(m),
        }
    }
    fn decision(&self) -> PyResult<&Decision> {
        self.st().decision.as_ref().ok_or_else(|| NativeRulesError::new_err("no decision pending"))
    }
}

#[pymethods]
impl PyGame {
    #[new]
    #[pyo3(signature = (decks, seed=0, starting_player=None, auto_single=true, max_turns=100, log=false, has_setup=false, start_step="untap".to_string(), mulligans=true, match_game=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        decks: (Vec<String>, Vec<String>),
        seed: i128,
        starting_player: Option<u8>,
        auto_single: bool,
        max_turns: i32,
        log: bool,
        has_setup: bool,
        start_step: String,
        mulligans: bool,
        match_game: i32,
    ) -> PyResult<Self> {
        if !STEPS.contains(&start_step.as_str()) {
            return Err(PyValueError::new_err(format!("{start_step:?} is not in list")));
        }
        let args = Args { decks: [decks.0, decks.1], seed, starting_player, auto_single, max_turns, log, has_setup, start_step, mulligans, match_game };
        let g = Game::new(args).map_err(PyValueError::new_err)?;
        Ok(PyGame { g, version: 0 })
    }

    fn start(&mut self) -> PyResult<()> {
        self.version += 1;
        self.g.start().map_err(NativeRulesError::new_err)
    }

    fn step(&mut self, index: usize) -> PyResult<()> {
        self.version += 1;
        self.g.step(index).map_err(Self::step_err)
    }

    /// Step through a list of option indices (fast fork-by-replay).
    fn replay(&mut self, actions: Vec<usize>) -> PyResult<()> {
        self.version += 1;
        for a in actions {
            self.g.step(a).map_err(Self::step_err)?;
        }
        Ok(())
    }

    // -- scalar state --------------------------------------------------------

    #[getter]
    fn version(&self) -> u64 {
        self.version
    }
    #[getter]
    fn over(&self) -> bool {
        self.st().over
    }
    #[getter]
    fn winner(&self) -> Option<u8> {
        self.st().winner
    }
    #[getter]
    fn end_reason(&self) -> &str {
        self.st().end_reason
    }
    #[getter]
    fn turn(&self) -> i32 {
        self.st().turn
    }
    #[getter]
    fn step_name(&self) -> &str {
        self.st().step_name
    }
    #[getter]
    fn active(&self) -> u8 {
        self.st().active
    }
    #[setter]
    fn set_active(&mut self, v: u8) {
        self.mutate().active = v;
    }
    #[getter]
    fn starting_player(&self) -> u8 {
        self.st().starting_player
    }
    #[getter]
    fn lands_played(&self) -> i32 {
        self.st().lands_played
    }
    #[getter]
    fn match_game(&self) -> i32 {
        self.st().match_game
    }
    #[getter]
    fn mulligans_taken(&self) -> Vec<i32> {
        self.st().mulligans_taken.to_vec()
    }
    #[getter]
    fn actions(&self) -> Vec<u32> {
        self.st().actions.clone()
    }
    #[getter]
    fn n_actions(&self) -> usize {
        self.st().actions.len()
    }
    #[getter]
    fn log(&self) -> Vec<String> {
        self.st().log.clone()
    }
    #[getter]
    fn attackers(&self) -> Vec<u32> {
        self.st().attackers.clone()
    }
    #[getter]
    fn blocked(&self) -> Vec<u32> {
        self.st().blocked.clone()
    }
    #[getter]
    fn blocks(&self) -> Vec<(u32, u32)> {
        self.st().blocks.clone()
    }
    #[getter]
    fn n_pending(&self) -> usize {
        self.st().pending.len()
    }
    #[getter]
    fn broken(&self) -> Option<String> {
        self.g.broken.clone()
    }

    fn player(&self, py: Python<'_>, p: usize) -> PyObject {
        let pl = &self.st().players[p];
        (pl.life, pl.library.clone(), pl.hand.clone(), pl.graveyard.clone(), pl.exile.clone(), pl.pool.iter().map(|(c, n)| (color_str(*c), *n)).collect::<Vec<_>>(), pl.drew_from_empty, pl.cards_drawn_this_turn)
            .into_py(py)
    }

    fn life(&self, p: usize) -> i32 {
        self.st().players[p].life
    }

    fn set_life(&mut self, p: usize, v: i32) {
        self.mutate().players[p].life = v;
    }
    fn set_cards_drawn_this_turn(&mut self, p: usize, v: i32) {
        self.mutate().players[p].cards_drawn_this_turn = v;
    }

    #[getter]
    fn battlefield(&self) -> Vec<CIdx> {
        self.st().battlefield.clone()
    }

    /// (uid, oid, name, defn name, owner, controller, zone, is_token,
    /// transformed, tapped, damage, deathtouch_damage, counters, sick,
    /// attached_to, skip_untap, temp, known_to)
    fn card_info(&self, py: Python<'_>, c: CIdx) -> PyResult<PyObject> {
        Ok(card_tuple(py, self.card(c)?))
    }

    /// Stack items bottom to top: (sid, kind, controller, name, targets, card
    /// index or None, method, cast_from, x, source card tuple or None, data).
    fn stack(&self, py: Python<'_>) -> PyObject {
        let st = self.st();
        st.stack
            .iter()
            .map(|it| {
                let targets: Vec<PyObject> = it.targets.iter().map(|r| ref_to_py(py, *r)).collect();
                let src = it.source.as_ref().map(|s| card_tuple(py, st.src(s)));
                let o: PyObject = (it.sid, it.kind.name(), it.controller, it.name.as_str(), targets, it.card, it.method.name(), it.cast_from.name(), it.x, src, data_list(py, st, &it.data)).into_py(py);
                o
            })
            .collect::<Vec<PyObject>>()
            .into_py(py)
    }

    // -- the current decision ------------------------------------------------

    /// (player, kind, prompt) or None.
    fn decision_head(&self) -> Option<(u8, &'static str, String)> {
        self.st().decision.as_ref().map(|d| (d.player, d.kind.name(), d.prompt.clone()))
    }

    fn option_labels(&self) -> PyResult<Vec<String>> {
        Ok(self.decision()?.options.iter().map(|o| o.label.clone()).collect())
    }

    fn option_keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let d = self.decision()?;
        Ok(PyList::new_bound(py, d.options.iter().map(|o| key_to_py(py, &o.key))))
    }

    fn n_options(&self) -> usize {
        self.st().decision.as_ref().map_or(0, |d| d.options.len())
    }

    /// Engine-internal value of option i, encoded with card indices:
    /// ("card", idx) stands for a Card object.
    fn option_value(&self, py: Python<'_>, i: usize) -> PyResult<PyObject> {
        let d = self.decision()?;
        let v = &d.options.get(i).ok_or_else(|| PyIndexError::new_err("option index"))?.value;
        let card = |c: CIdx| ("card", c).into_py(py);
        Ok(match v {
            Val::None => py.None(),
            Val::Bool(b) => b.into_py(py),
            Val::Int(i) => i.into_py(py),
            Val::Pass => ("pass",).into_py(py),
            Val::Land(c) => ("land", card(*c)).into_py(py),
            Val::Cast(c, m, None) => ("cast", card(*c), m.name()).into_py(py),
            Val::Cast(c, m, Some(i)) => ("cast", card(*c), m.name(), *i).into_py(py),
            Val::Activate(c, i) => ("activate", card(*c), *i).into_py(py),
            Val::Mana(c, i) => ("mana", card(*c), *i).into_py(py),
            Val::Card(c) => card(*c),
            Val::Ref(r) => ref_to_py(py, *r),
            Val::Pool(c) => ("pool", color_str(*c)).into_py(py),
            Val::Source(c, col) => ("source", card(*c), color_str(*col)).into_py(py),
            Val::Trigger(i) => ("trigger", *i).into_py(py),
            Val::Split(v) => PyTuple::new_bound(py, v).into_py(py),
            Val::Group(g) => g.into_py(py),
            Val::Order(v) => PyTuple::new_bound(py, v.iter().map(|&c| card(c))).into_py(py),
            Val::Top => "top".into_py(py),
            Val::Bottom => "bottom".into_py(py),
        })
    }

    // -- rules queries (for bots and tests) ----------------------------------

    fn power(&self, c: CIdx) -> PyResult<i32> {
        Ok(self.st().power(self.card(c)?))
    }
    fn toughness(&self, c: CIdx) -> PyResult<i32> {
        Ok(self.st().toughness(self.card(c)?))
    }
    fn keywords(&self, c: CIdx) -> PyResult<Vec<&'static str>> {
        Ok(db().keyword_list(self.st().keywords(self.card(c)?)))
    }
    fn has(&self, c: CIdx, kw: &str) -> PyResult<bool> {
        let d = db();
        if !d.keyword_names.iter().any(|k| k == kw) {
            return Ok(false);
        }
        Ok(self.st().has(self.card(c)?, kw))
    }
    fn types(&self, c: CIdx) -> PyResult<Vec<&'static str>> {
        Ok(type_names(self.st().types(self.card(c)?)))
    }
    fn is_creature(&self, c: CIdx) -> PyResult<bool> {
        Ok(self.st().is_creature(self.card(c)?))
    }
    fn perm(&self, oid: u32) -> Option<CIdx> {
        self.st().perm(oid)
    }
    fn sorcery_timing(&self, p: u8) -> bool {
        self.st().sorcery_timing(p)
    }
    fn mana_sources(&self, p: u8) -> Vec<(CIdx, usize)> {
        self.st().mana_sources(p, &[])
    }
    fn sac_candidates(&self, p: u8, flt: &str) -> PyResult<Vec<CIdx>> {
        Ok(self.st().sac_candidates(p, sac_filter(flt)?, &[]))
    }
    fn cost_reduction(&self, p: u8, c: CIdx) -> PyResult<i32> {
        self.card(c)?;
        Ok(self.st().cost_reduction(p, c))
    }
    fn lethal(&self, a: CIdx, b: CIdx) -> PyResult<i32> {
        Ok(self.st().lethal(self.card(a)?, self.card(b)?))
    }
    fn target_candidates(&self, py: Python<'_>, kind: &str, controller: u8) -> PyResult<Vec<PyObject>> {
        let k = TK::parse(kind).map_err(PyValueError::new_err)?;
        Ok(self.st().target_candidates(k, controller, None).into_iter().map(|r| ref_to_py(py, r)).collect())
    }

    // -- scenario setup and determinization ----------------------------------

    #[pyo3(signature = (name, player, zone, tapped=false, sick=false, counters=0))]
    fn add_card(&mut self, name: &str, player: u8, zone: &str, tapped: bool, sick: bool, counters: i32) -> PyResult<CIdx> {
        let z = Zone::parse(zone).filter(|z| *z != Zone::Stack).ok_or_else(|| PyValueError::new_err(format!("bad zone {zone:?}")))?;
        self.mutate().add_card(name, player, z, tapped, sick, counters).map_err(PyValueError::new_err)
    }

    // -- direct rules actions (scenario tests) --------------------------------

    /// Mutate one field of a card (scenario tests poke card objects directly).
    fn set_card_field(&mut self, c: CIdx, field: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.card(c)?;
        let card = self.mutate().cm(c);
        match field {
            "tapped" => card.tapped = value.extract()?,
            "transformed" => card.transformed = value.extract()?,
            "sick" => card.sick = value.extract()?,
            "deathtouch_damage" => card.deathtouch_damage = value.extract()?,
            "damage" => card.damage = value.extract()?,
            "counters" => card.counters = value.extract()?,
            "skip_untap" => card.skip_untap = value.extract()?,
            "attached_to" => card.attached_to = value.extract()?,
            "controller" => card.controller = value.extract()?,
            _ => return Err(PyValueError::new_err(format!("card field {field:?} is not settable"))),
        }
        Ok(())
    }

    /// Append an 'until end of turn' effect to a card.
    fn add_temp(&mut self, c: CIdx, keywords: Vec<String>, power: i32, toughness: i32) -> PyResult<()> {
        self.card(c)?;
        let d = db();
        let mut mask = 0;
        for k in &keywords {
            if !d.keyword_names.contains(k) {
                return Err(PyValueError::new_err(format!("unknown keyword {k:?}")));
            }
            mask |= d.kw(k);
        }
        self.mutate().cm(c).temp.push(TempEffect { keywords: mask, power, toughness });
        Ok(())
    }

    fn untap_step(&mut self) {
        self.mutate().untap_step();
    }
    fn destroy(&mut self, c: CIdx) -> PyResult<bool> {
        self.card(c)?;
        Ok(self.mutate().destroy(c))
    }
    fn sacrifice(&mut self, c: CIdx) -> PyResult<()> {
        self.card(c)?;
        self.mutate().sacrifice(c);
        Ok(())
    }
    /// Pending triggers: (controller, source card tuple, trigger name, data).
    fn pending(&self, py: Python<'_>) -> Vec<PyObject> {
        let st = self.st();
        st.pending.iter().map(|t| (t.controller, card_tuple(py, st.src(&t.source)), t.tdef.name.as_str(), data_list(py, st, &t.data)).into_py(py)).collect()
    }
    fn clear_pending(&mut self) {
        self.mutate().pending.clear();
    }

    /// Replace a card's definition (`determinize`): `c.defn = d; c.transformed = False`.
    fn set_card_def(&mut self, c: CIdx, name: &str) -> PyResult<()> {
        self.card(c)?;
        let d = db();
        let def = *d.cards.get(name).ok_or_else(|| PyValueError::new_err(format!("unknown card {name:?}")))?;
        let card = self.mutate().cm(c);
        card.def = def;
        card.transformed = false;
        Ok(())
    }

    /// `random.Random.setstate()` for the engine's RNG (625 words: state + index).
    fn set_rng_state(&mut self, words: Vec<u32>) -> PyResult<()> {
        self.mutate().rng.set_state(&words).map_err(PyValueError::new_err)
    }

    /// `random.Random.getstate()` of the engine's RNG.
    fn rng_state(&self, py: Python<'_>) -> PyObject {
        let words: Vec<u64> = self.st().rng.state().into_iter().map(|w| w as u64).collect();
        (3, PyTuple::new_bound(py, words), py.None()).into_py(py)
    }

    // -- views ---------------------------------------------------------------

    /// `view.observe(game, viewer)`.
    fn observe<'py>(&self, py: Python<'py>, viewer: u8) -> PyResult<Bound<'py, PyDict>> {
        let st = self.st();
        let opp = 1 - viewer;
        let me = &st.players[viewer as usize];
        let them = &st.players[opp as usize];
        let names = |v: &Vec<CIdx>| v.iter().map(|&c| st.c(c).name()).collect::<Vec<_>>();
        let known_library = |p: &Player| p.library.iter().enumerate().filter(|(_, &c)| st.c(c).known_to & pbit(viewer) != 0).map(|(i, &c)| (i, st.c(c).name())).collect::<Vec<_>>();
        let pool = |p: &Player| -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new_bound(py);
            for (c, n) in &p.pool {
                d.set_item(color_str(*c), *n)?;
            }
            Ok(d)
        };
        let o = PyDict::new_bound(py);
        o.set_item("turn", st.turn)?;
        o.set_item("match_game", st.match_game)?;
        o.set_item("active", rel(st.active, viewer))?;
        o.set_item("step", st.step_name)?;
        o.set_item("over", st.over)?;
        o.set_item("winner", st.winner.map(|w| rel(w, viewer)))?;
        let s = PyDict::new_bound(py);
        s.set_item("life", me.life)?;
        s.set_item("hand", names(&me.hand))?;
        s.set_item("library_count", me.library.len())?;
        s.set_item("library_known", known_library(me))?;
        s.set_item("graveyard", names(&me.graveyard))?;
        s.set_item("exile", names(&me.exile))?;
        s.set_item("pool", pool(me)?)?;
        s.set_item("cards_drawn_this_turn", me.cards_drawn_this_turn)?;
        s.set_item("mulligans", st.mulligans_taken[viewer as usize])?;
        o.set_item("self", s)?;
        let t = PyDict::new_bound(py);
        t.set_item("life", them.life)?;
        t.set_item("hand_count", them.hand.len())?;
        t.set_item("hand_known", them.hand.iter().filter(|&&c| st.c(c).known_to & pbit(viewer) != 0).map(|&c| st.c(c).name()).collect::<Vec<_>>())?;
        t.set_item("library_count", them.library.len())?;
        t.set_item("library_known", known_library(them))?;
        t.set_item("graveyard", names(&them.graveyard))?;
        t.set_item("exile", names(&them.exile))?;
        t.set_item("pool", pool(them)?)?;
        t.set_item("mulligans", st.mulligans_taken[opp as usize])?;
        o.set_item("opponent", t)?;
        o.set_item("lands_played", if st.active == viewer { Some(st.lands_played) } else { None })?;
        let bf = PyList::empty_bound(py);
        for &ci in &st.battlefield {
            let c = st.c(ci);
            let p = PyDict::new_bound(py);
            let cr = st.is_creature(c);
            p.set_item("oid", c.oid)?;
            p.set_item("name", c.name())?;
            p.set_item("controller", rel(c.controller, viewer))?;
            p.set_item("token", c.is_token)?;
            p.set_item("types", type_names(st.types(c)))?;
            p.set_item("tapped", c.tapped)?;
            p.set_item("sick", c.sick)?;
            p.set_item("damage", c.damage)?;
            p.set_item("counters", c.counters)?;
            p.set_item("power", if cr { Some(st.power(c)) } else { None })?;
            p.set_item("toughness", if cr { Some(st.toughness(c)) } else { None })?;
            p.set_item("keywords", db().keyword_list(st.keywords(c)))?;
            p.set_item("attached_to", c.attached_to)?;
            p.set_item("attacking", st.attackers.contains(&c.oid))?;
            p.set_item("blocking", st.blocks.iter().find(|(b, _)| *b == c.oid).map(|(_, a)| *a))?;
            p.set_item("skip_untap", c.skip_untap)?;
            bf.append(p)?;
        }
        o.set_item("battlefield", bf)?;
        let stack = PyList::empty_bound(py);
        for it in &st.stack {
            let d = PyDict::new_bound(py);
            d.set_item("sid", it.sid)?;
            d.set_item("name", it.name.as_str())?;
            d.set_item("kind", it.kind.name())?;
            d.set_item("controller", rel(it.controller, viewer))?;
            let targets: Vec<String> = it.targets.iter().filter(|r| ref_exists(st, **r)).map(|r| st.describe_ref(*r, viewer).0).collect();
            d.set_item("targets", targets)?;
            d.set_item("x", it.x)?;
            d.set_item("method", it.method.name())?;
            stack.append(d)?;
        }
        o.set_item("stack", stack)?;
        match &st.decision {
            Some(d) if d.player == viewer => {
                let dd = PyDict::new_bound(py);
                dd.set_item("kind", d.kind.name())?;
                dd.set_item("prompt", d.prompt.as_str())?;
                dd.set_item("options", d.options.iter().map(|o| o.label.as_str()).collect::<Vec<_>>())?;
                dd.set_item("keys", PyList::new_bound(py, d.options.iter().map(|o| key_to_py(py, &o.key))))?;
                o.set_item("decision", dd)?;
            }
            Some(d) => {
                let dd = PyDict::new_bound(py);
                dd.set_item("kind", d.kind.name())?;
                dd.set_item("waiting_for", "opponent")?;
                o.set_item("decision", dd)?;
            }
            None => o.set_item("decision", py.None())?,
        }
        Ok(o)
    }

    /// `encode.state_features(game, viewer)`.
    fn state_features(&self, viewer: u8) -> Vec<String> {
        state_features(self.st(), viewer)
    }

    /// `encode.entity_features(game, viewer)`: (feature lists, {object id: entity index}).
    fn entity_features(&self, viewer: u8) -> (Vec<Vec<String>>, std::collections::HashMap<u32, usize>) {
        let mut o = crate::features::EntityStrings::default();
        let ids = crate::features::entity_features_into(self.st(), viewer, &mut o);
        (o.0, ids.into_iter().enumerate().map(|(k, id)| (id, k)).collect())
    }

    /// `rl.features.featurize(game, player, state_dim, option_dim)`.
    fn featurize(&self, player: u8, state_dim: u32, option_dim: u32) -> PyResult<(Vec<u32>, Vec<Vec<u32>>)> {
        crate::features::featurize(self.st(), player, state_dim, option_dim).ok_or_else(|| NativeRulesError::new_err("no decision pending"))
    }

    /// `featurize` with the options flattened: (state, option lengths, all
    /// option tokens). Cheaper to pack into requests and samples.
    fn featurize_flat(&self, player: u8, state_dim: u32, option_dim: u32) -> PyResult<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        let (state, opts) = crate::features::featurize(self.st(), player, state_dim, option_dim).ok_or_else(|| NativeRulesError::new_err("no decision pending"))?;
        let lens = opts.iter().map(|o| o.len() as u32).collect();
        Ok((state, lens, opts.concat()))
    }

    /// Hashed `event_tokens` of option i for (decider, opponent).
    fn event_hashes(&self, i: usize, option_dim: u32) -> PyResult<(Vec<u32>, Vec<u32>)> {
        crate::features::event_hashes(self.st(), i, option_dim).ok_or_else(|| PyIndexError::new_err("no such option"))
    }

    /// Canonical dump of the full (hidden) state for differential testing.
    fn dump(&self, py: Python<'_>) -> PyResult<PyObject> {
        let st = self.st();
        let cards = |v: &Vec<CIdx>| v.iter().map(|&c| card_tuple(py, st.c(c))).collect::<Vec<_>>();
        let d = PyDict::new_bound(py);
        d.set_item("turn", st.turn)?;
        d.set_item("active", st.active)?;
        d.set_item("step", st.step_name)?;
        d.set_item("lands_played", st.lands_played)?;
        d.set_item("starting_player", st.starting_player)?;
        d.set_item("mulligans_taken", st.mulligans_taken.to_vec())?;
        d.set_item("next_id", st.next_id)?;
        d.set_item("over", st.over)?;
        d.set_item("winner", st.winner)?;
        d.set_item("end_reason", st.end_reason)?;
        d.set_item("match_game", st.match_game)?;
        let players = PyList::empty_bound(py);
        for p in &st.players {
            let pd = PyDict::new_bound(py);
            pd.set_item("life", p.life)?;
            pd.set_item("library", cards(&p.library))?;
            pd.set_item("hand", cards(&p.hand))?;
            pd.set_item("graveyard", cards(&p.graveyard))?;
            pd.set_item("exile", cards(&p.exile))?;
            pd.set_item("pool", p.pool.iter().map(|(c, n)| (color_str(*c), *n)).collect::<Vec<_>>())?;
            pd.set_item("drew_from_empty", p.drew_from_empty)?;
            pd.set_item("cards_drawn_this_turn", p.cards_drawn_this_turn)?;
            players.append(pd)?;
        }
        d.set_item("players", players)?;
        d.set_item("battlefield", cards(&st.battlefield))?;
        let stack = PyList::empty_bound(py);
        for it in &st.stack {
            let sd = PyDict::new_bound(py);
            sd.set_item("sid", it.sid)?;
            sd.set_item("kind", it.kind.name())?;
            sd.set_item("controller", it.controller)?;
            sd.set_item("name", it.name.as_str())?;
            sd.set_item("target_specs", it.target_specs.iter().map(|t| t.name()).collect::<Vec<_>>())?;
            sd.set_item("targets", it.targets.iter().map(|r| ref_to_py(py, *r)).collect::<Vec<_>>())?;
            sd.set_item("card", it.card.map(|c| card_tuple(py, st.c(c))))?;
            sd.set_item("source", it.source.as_ref().map(|s| card_tuple(py, st.src(s))))?;
            sd.set_item("method", it.method.name())?;
            sd.set_item("cast_from", it.cast_from.name())?;
            sd.set_item("x", it.x)?;
            sd.set_item("data", data_list(py, st, &it.data))?;
            stack.append(sd)?;
        }
        d.set_item("stack", stack)?;
        let pending = PyList::empty_bound(py);
        for t in &st.pending {
            pending.append((t.controller, card_tuple(py, st.src(&t.source)), t.tdef.name.as_str(), data_list(py, st, &t.data)))?;
        }
        d.set_item("pending", pending)?;
        d.set_item("attackers", st.attackers.clone())?;
        let mut blocked = st.blocked.clone();
        blocked.sort_unstable();
        d.set_item("blocked", blocked)?;
        d.set_item("blocks", st.blocks.clone())?;
        let rs = st.rng.state();
        d.set_item("rng_index", rs[624])?;
        Ok(d.into_py(py))
    }
}

/// A live view of one card of a game (Python: `NativeCard` subclasses it).
/// Every getter reads the engine's current state, so a view stays valid,
/// and keeps its identity, for the whole game.
#[pyclass(unsendable, subclass, module = "mtg_ml_native", name = "CardView")]
pub struct CardView {
    game: Py<PyGame>,
    idx: CIdx,
}

macro_rules! card_get {
    ($($name:ident : $t:ty => $e:expr;)* @set $($sname:ident = $field:literal),*) => {
        #[pymethods]
        impl CardView {
            #[new]
            fn new(game: Py<PyGame>, idx: CIdx) -> Self {
                CardView { game, idx }
            }
            #[getter]
            fn _idx(&self) -> CIdx {
                self.idx
            }
            #[getter]
            fn _game(&self, py: Python<'_>) -> Py<PyGame> {
                self.game.clone_ref(py)
            }
            $(
                #[getter]
                fn $name(&self, py: Python<'_>) -> $t {
                    let g = self.game.borrow(py);
                    #[allow(unused_variables)]
                    let st = g.st();
                    let c = &st.cards[self.idx as usize];
                    ($e)(c)
                }
            )*
            $(
                #[setter]
                fn $sname(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
                    self.game.borrow_mut(py).set_card_field(self.idx, $field, value)
                }
            )*
        }
    };
}

card_get! {
    uid: u32 => |c: &Card| c.uid;
    oid: u32 => |c: &Card| c.oid;
    name: &'static str => |c: &Card| c.name();
    _defn_name: &'static str => |c: &Card| c.defn().name.as_str();
    owner: u8 => |c: &Card| c.owner;
    controller: u8 => |c: &Card| c.controller;
    zone: &'static str => |c: &Card| c.zone.name();
    is_token: bool => |c: &Card| c.is_token;
    transformed: bool => |c: &Card| c.transformed;
    tapped: bool => |c: &Card| c.tapped;
    damage: i32 => |c: &Card| c.damage;
    deathtouch_damage: bool => |c: &Card| c.deathtouch_damage;
    counters: i32 => |c: &Card| c.counters;
    sick: bool => |c: &Card| c.sick;
    attached_to: Option<u32> => |c: &Card| c.attached_to;
    skip_untap: i32 => |c: &Card| c.skip_untap;
    _known: Vec<u8> => |c: &Card| known_list(c.known_to);
    _temp: Vec<(Vec<&'static str>, i32, i32)> => |c: &Card| c.temp.iter().map(|t| (db().keyword_list(t.keywords), t.power, t.toughness)).collect::<Vec<_>>();
    @set set_tapped = "tapped", set_transformed = "transformed", set_sick = "sick", set_deathtouch_damage = "deathtouch_damage",
    set_damage = "damage", set_counters = "counters", set_skip_untap = "skip_untap", set_attached_to = "attached_to", set_controller = "controller"
}

fn ref_exists(st: &State, r: Ref) -> bool {
    match r {
        Ref::Player(_) => true,
        Ref::Stack(s) => st.stack_pos(s).is_some(),
        Ref::Perm(o) => st.perm(o).is_some(),
    }
}


/// Port of `encode.state_features` (same features in the same order).
pub fn state_features(st: &State, viewer: u8) -> Vec<String> {
    let mut o = crate::features::StringOut::default();
    crate::features::state_features_into(st, viewer, &mut o);
    o.finish()
}

#[pyfunction]
fn load_cards(spec: &str) -> PyResult<()> {
    cards::load(spec).map_err(PyValueError::new_err)
}

#[pyfunction]
fn loaded_spec() -> String {
    db().spec_text.clone()
}

#[pymodule]
fn mtg_ml_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_class::<CardView>()?;
    m.add_function(wrap_pyfunction!(load_cards, m)?)?;
    m.add_function(wrap_pyfunction!(loaded_spec, m)?)?;
    m.add("NativeRulesError", m.py().get_type_bound::<NativeRulesError>())?;
    Ok(())
}
