//! The decision-making half of the rules: everything that is a generator in
//! game.py. It runs on a stackful coroutine (`corosensei`), so `ask()`
//! suspends exactly where Python's `yield` does and the code below reads
//! like the Python it mirrors.
//!
//! Aliasing: the state lives behind a raw pointer shared with the driver
//! (`game.rs`), which reads (and, for determinization, edits) it while the
//! coroutine is suspended. Engine code therefore never holds a `&mut State`
//! across `ask()`: `Eng::s()` hands out a fresh borrow tied to `&mut self`,
//! and `ask()` itself takes `&mut self`, so the borrow checker enforces it.

use corosensei::Yielder;

use crate::cards::{db, type_bit, Custom, Op, SacFilter, SearchFilter, Who, T_CREATURE, T_LAND};
use crate::mana::{ManaCost, Remaining};
use crate::state::*;

pub struct Eng {
    pub s: *mut State,
    pub y: *const Yielder<usize, ()>,
}

fn opt(label: String, key: Key, value: Val) -> Opt {
    Opt { label, key, value }
}

fn s(x: &'static str) -> KI {
    KI::S(x)
}

/// game.py `_filtered`: a filter pays a coloured symbol; its own (generic) cost is added instead.
fn filtered(rem: &Remaining, color: u8, generic: i32) -> Remaining {
    let mut r = rem.clone();
    let pos = r.colored.iter().position(|(k, _)| *k == color).expect("filtered colour");
    r.colored[pos].1 -= 1;
    if r.colored[pos].1 == 0 {
        r.colored.remove(pos);
    }
    r.generic += generic;
    r
}

/// `pool2 = dict(pool); pool2[color] = pool2.get(color, 0) + n`
fn pool_with(pool: &[(u8, i32)], color: u8, n: i32) -> Vec<(u8, i32)> {
    let mut v = pool.to_vec();
    match v.iter_mut().find(|(k, _)| *k == color) {
        Some(e) => e.1 += n,
        None => v.push((color, n)),
    }
    v
}

/// Lower-case card type name (the key of a return_from_graveyard choice).
fn type_lower(name: &'static str) -> &'static str {
    match name {
        "Artifact" => "artifact",
        "Battle" => "battle",
        "Creature" => "creature",
        "Enchantment" => "enchantment",
        "Instant" => "instant",
        "Kindred" => "kindred",
        "Land" => "land",
        "Planeswalker" => "planeswalker",
        _ => "sorcery",
    }
}

impl Eng {
    #[inline(always)]
    pub fn s(&mut self) -> &mut State {
        unsafe { &mut *self.s }
    }

    /// `Game.ask`: returns the chosen option's value.
    pub fn ask(&mut self, player: u8, kind: Kind, prompt: impl FnOnce() -> String, mut options: Vec<Opt>) -> R<Val> {
        if options.is_empty() {
            return rules(format!("decision '{}' ({}) has no options", kind.name(), prompt()));
        }
        let st = self.s();
        if st.auto_single && options.len() == 1 && st.sim_viewer.map_or(true, |v| v == player) {
            if st.logging && kind != Kind::Priority {
                let m = format!("  p{player} {}: {} (only option)", kind.name(), options[0].label);
                st.log.push(m);
            }
            return Ok(options.pop().unwrap().value);
        }
        st.decision = Some(Decision { player, kind, prompt: prompt(), options });
        let idx = unsafe { (*self.y).suspend(()) };
        if idx == ABORT {
            return Err(Stop::Abort);
        }
        let d = self.s().decision.take().expect("resumed without a decision");
        Ok(d.options.into_iter().nth(idx).expect("option index").value)
    }

    fn log(&mut self, f: impl FnOnce(&State) -> String) {
        let st = self.s();
        if st.logging {
            let m = f(st);
            st.log.push(m);
        }
    }

    // ------------------------------------------------------------------
    // Turn structure
    // ------------------------------------------------------------------

    /// `resume`: `Game::copy` finishing a turn from a step-start snapshot.
    pub fn main(&mut self, start_step: &str, resume: Option<(&'static str, bool)>) -> R<()> {
        let mut first = true;
        if let Some((step, skip_draw)) = resume {
            self.run_turn(step, skip_draw)?;
            let st = self.s();
            st.active = 1 - st.active;
            first = false;
        } else if self.s().mulligan_phase {
            self.mulligans()?;
        }
        loop {
            let st = self.s();
            st.turn += 1;
            if st.turn > st.max_turns {
                return Err(Stop::GameOver { winner: None, reason: "turn limit" });
            }
            if !(first && st.args.has_setup) {
                st.begin_turn();
            }
            let start = if first { start_step } else { "untap" };
            let skip_draw = first && st.skip_first_draw;
            self.run_turn(start, skip_draw)?;
            first = false;
            let st = self.s();
            st.active = 1 - st.active;
        }
    }

    fn mulligans(&mut self) -> R<()> {
        self.s().step_name = "mulligan";
        let sp = self.s().starting_player;
        let order = [sp, 1 - sp];
        let mut deciding: Vec<u8> = order.to_vec();
        while !deciding.is_empty() {
            let mut again = vec![];
            for &p in &deciding {
                let n = self.s().mulligans_taken[p as usize];
                let mut options = vec![opt(format!("Keep ({} cards)", 7 - n), vec![s("mulligan"), s("keep")], Val::Bool(false))];
                if n < 6 {
                    options.push(opt(format!("Mulligan (to {})", 6 - n), vec![s("mulligan"), s("mulligan")], Val::Bool(true)));
                }
                if let Val::Bool(true) = self.ask(p, Kind::Mulligan, || format!("Opening hand, {n} mulligan(s) taken: keep {}?", 7 - n), options)? {
                    again.push(p);
                }
            }
            for &p in &again {
                let st = self.s();
                st.mulligans_taken[p as usize] += 1;
                let n = st.mulligans_taken[p as usize];
                st.push_log_lazy(|_| format!("p{p} mulligans ({n})"));
                for c in st.players[p as usize].hand.clone() {
                    st.move_card(c, Zone::Library, None, Pos::Bottom, None, false);
                }
                st.shuffle(p as usize);
                st.draw(p as usize, 7, false);
            }
            deciding = again;
        }
        for p in order {
            let n = self.s().mulligans_taken[p as usize];
            for i in 0..n {
                let options = self.s().hand_card_options(p, "bottom");
                let c = self.ask(p, Kind::ChooseCard, || format!("Mulligan: put a card on the bottom of your library ({}/{n})", i + 1), options)?;
                if let Val::Card(c) = c {
                    self.s().move_card(c, Zone::Library, None, Pos::Bottom, Some(pbit(p)), false);
                }
            }
        }
        Ok(())
    }

    fn run_turn(&mut self, start: &str, skip_draw: bool) -> R<()> {
        let from = STEPS.iter().position(|x| *x == start).ok_or_else(|| Stop::Rules(format!("unknown step {start:?}")))?;
        for &name in &STEPS[from..] {
            if (name == "declare_blockers" || name == "combat_damage") && self.s().attackers.is_empty() {
                continue;
            }
            let st = self.s();
            if st.snapshots {
                st.take_snapshot(name, skip_draw);
            } else if let Some(s) = &st.snap {
                // A copy's inherited snapshot is stale once it leaves the
                // snapshot's own step start (where it resumed).
                if !(s.step == name && s.n_actions == st.actions.len() && s.state.turn == st.turn) {
                    st.snap = None;
                }
            }
            self.s().step_name = name;
            self.log(|_| format!("-- {name}"));
            match name {
                "untap" => self.s().untap_step(),
                "upkeep" => {
                    self.s().emit_upkeep();
                    self.priority_round()?;
                }
                "draw" => {
                    if !skip_draw {
                        let a = self.s().active as usize;
                        self.s().draw(a, 1, true);
                    }
                    self.priority_round()?;
                }
                "declare_attackers" => {
                    self.declare_attackers()?;
                    self.priority_round()?;
                }
                "declare_blockers" => {
                    self.declare_blockers()?;
                    self.priority_round()?;
                }
                "combat_damage" => {
                    self.combat_damage()?;
                    self.priority_round()?;
                }
                "end_combat" => {
                    self.priority_round()?;
                    self.s().clear_combat();
                }
                "cleanup" => self.cleanup_step()?,
                _ => self.priority_round()?,
            }
            self.s().empty_pools();
        }
        Ok(())
    }

    fn cleanup_step(&mut self) -> R<()> {
        loop {
            let p = self.s().active;
            while self.s().players[p as usize].hand.len() > MAX_HAND {
                let n = self.s().players[p as usize].hand.len();
                let options = self.s().hand_card_options(p, "discard");
                if let Val::Card(c) = self.ask(p, Kind::ChooseCard, || format!("Discard to hand size ({n}/{MAX_HAND})"), options)? {
                    self.s().discard(c);
                }
            }
            let st = self.s();
            for i in 0..st.battlefield.len() {
                let ci = st.battlefield[i];
                let c = st.cm(ci);
                c.damage = 0;
                c.deathtouch_damage = false;
                c.temp.clear();
            }
            let changed = st.sba()?;
            if changed || !self.s().pending.is_empty() {
                self.priority_round()?;
                self.s().empty_pools();
                continue;
            }
            return Ok(());
        }
    }

    // ------------------------------------------------------------------
    // Priority
    // ------------------------------------------------------------------

    fn priority_round(&mut self) -> R<()> {
        let mut p = self.s().active;
        let mut passes = 0;
        loop {
            self.sba_and_triggers()?;
            let step = self.s().step_name;
            let act = if let Some(v) = self.s().sim_viewer {
                // game.py: a simulation never lists priority options; the
                // "assume the opponent passes" simulation passes for the
                // other player, any other priority stops it.
                if self.s().sim_assume_pass && p != v {
                    Val::Pass
                } else {
                    self.s().decision = Some(Decision { player: p, kind: Kind::Priority, prompt: format!("Priority ({step})"), options: vec![] });
                    if unsafe { (*self.y).suspend(()) } == ABORT {
                        return Err(Stop::Abort);
                    }
                    return rules("a simulation copy cannot continue past a priority");
                }
            } else {
                let options = self.s().priority_options(p);
                if self.s().auto_pass && self.s().uneventful_priority(p, &options) {
                    Val::Pass
                } else {
                    self.ask(p, Kind::Priority, || format!("Priority ({step})"), options)?
                }
            };
            if let Val::Pass = act {
                passes += 1;
                if passes >= 2 {
                    if self.s().stack.is_empty() {
                        return Ok(());
                    }
                    self.resolve_top()?;
                    passes = 0;
                    p = self.s().active;
                } else {
                    p = 1 - p;
                }
            } else {
                self.take_action(p, act)?;
                passes = 0;
            }
        }
    }

    fn sba_and_triggers(&mut self) -> R<()> {
        loop {
            let changed = self.s().sba()?;
            if !self.s().pending.is_empty() {
                self.put_triggers_on_stack()?;
                continue;
            }
            if !changed {
                return Ok(());
            }
        }
    }

    fn take_action(&mut self, p: u8, act: Val) -> R<()> {
        match act {
            Val::Land(card) => {
                let st = self.s();
                st.lands_played += 1;
                st.push_log_lazy(|s| format!("p{p} plays {}", s.c(card).name()));
                st.put_onto_battlefield(card, p, false);
                Ok(())
            }
            Val::Cast(card, mode, choice) => self.cast(p, card, mode, choice),
            Val::Activate(card, i) => self.activate(p, card, i as usize),
            Val::Plot(card) => self.plot(p, card),
            Val::Mana(card, i) => {
                let st = self.s();
                let color = st.c(card).face().abilities[i as usize].mana.as_ref().unwrap()[0];
                st.activate_mana_ability(card, i as usize);
                st.players[p as usize].pool_add(color, 1);
                Ok(())
            }
            other => rules(format!("unknown action {other:?}")),
        }
    }

    fn put_triggers_on_stack(&mut self) -> R<()> {
        let pend = std::mem::take(&mut self.s().pending);
        let active = self.s().active;
        let (mut first, mut second): (Vec<PendingTrigger>, Vec<PendingTrigger>) = (vec![], vec![]);
        for t in pend {
            if t.controller == active {
                first.push(t);
            } else if t.controller == 1 - active {
                second.push(t);
            }
        }
        for (p, mut mine) in [(active, first), (1 - active, second)] {
            while !mine.is_empty() {
                let st = self.s();
                let mut options = vec![];
                let mut seen: Vec<(&'static str, &'static str)> = vec![];
                for (i, t) in mine.iter().enumerate() {
                    let key = (st.src(&t.source).name(), t.tdef.name.as_str());
                    if seen.contains(&key) {
                        continue;
                    }
                    seen.push(key);
                    options.push(opt(format!("Put on stack: {} - {}", key.0, key.1), vec![s("trigger"), s(key.0), s(key.1)], Val::Trigger(i)));
                }
                let i = match self.ask(p, Kind::OrderTriggers, || "Choose the next trigger to put on the stack (first = resolves last)".to_string(), options)? {
                    Val::Trigger(i) => i,
                    _ => unreachable!(),
                };
                let t = mine.remove(i);
                let st = self.s();
                let sid = st.new_id();
                let name = format!("{}: {}", st.src(&t.source).name(), t.tdef.name);
                let up_to = t.tdef.up_to;
                st.push_log_lazy(|_| format!("trigger -> stack: {name}"));
                st.stack.push(StackItem {
                    sid,
                    kind: SKind::Trigger,
                    controller: t.controller,
                    name: name.clone(),
                    effect: Some(&t.tdef.effect),
                    target_specs: t.tdef.targets.clone(),
                    targets: vec![],
                    card: None,
                    source: Some(t.source),
                    method: Method::Normal,
                    cast_from: Zone::Hand,
                    x: 0,
                    data: t.data,
                });
                if !t.tdef.targets.is_empty() {
                    // 603.3d: a triggered ability without legal targets is removed from the stack
                    // ("up to" targets can always be chosen: none).
                    let st = self.s();
                    if !up_to && t.tdef.targets.iter().any(|&spec| st.target_candidates(spec, p, Some(sid), &[]).is_empty()) {
                        let pos = st.stack_pos(sid).unwrap();
                        let item = st.stack.remove(pos);
                        st.push_log_lazy(|_| format!("{}: no legal targets", item.name));
                        continue;
                    }
                    self.choose_targets(p, sid, None, up_to)?;
                    self.s().emit_targeted(sid);
                }
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Targets, costs and mana
    // ------------------------------------------------------------------

    /// `allowed`: an extra filter on the candidates (targets the cost can be paid for).
    /// `up_to`: "up to one target": choosing no target ends the choice.
    fn choose_targets(&mut self, p: u8, sid: u32, allowed: Option<&dyn Fn(&State, Ref) -> bool>, up_to: bool) -> R<()> {
        let specs = {
            let st = self.s();
            st.stack[st.stack_pos(sid).unwrap()].target_specs.clone()
        };
        for spec in specs {
            let st = self.s();
            let chosen = &st.stack[st.stack_pos(sid).unwrap()].targets;
            let mut cands = st.target_candidates(spec, p, Some(sid), chosen);
            if let Some(f) = allowed {
                cands.retain(|&r| f(st, r));
            }
            let refs = st.referenced_oids();
            let mut seen_perm: Vec<EquivKey> = vec![];
            let mut seen_ref: Vec<Ref> = vec![];
            let mut options = vec![];
            for r in cands {
                match r {
                    Ref::Perm(oid) => {
                        let k = st.equiv_key(st.perm(oid).unwrap(), &refs);
                        if seen_perm.contains(&k) {
                            continue;
                        }
                        seen_perm.push(k);
                    }
                    _ => {
                        if seen_ref.contains(&r) {
                            continue;
                        }
                        seen_ref.push(r);
                    }
                }
                let (label, key) = st.describe_ref(r, p);
                let mut full = vec![s("target"), s(spec.name())];
                full.extend(key);
                options.push(opt(format!("Target {label}"), full, Val::Ref(r)));
            }
            if up_to {
                options.insert(0, opt("No target".into(), vec![s("target"), s(spec.name()), KI::N], Val::None));
            }
            let item_name = st.stack[st.stack_pos(sid).unwrap()].name.clone();
            let r = match self.ask(p, Kind::Target, || format!("Choose target ({}) for {item_name}", spec.name()), options)? {
                Val::Ref(r) => r,
                Val::None => return Ok(()),
                _ => unreachable!(),
            };
            let st = self.s();
            let pos = st.stack_pos(sid).unwrap();
            st.stack[pos].targets.push(r);
        }
        Ok(())
    }

    fn pay_mana(&mut self, p: u8, mut rem: Remaining, sac_filter: Option<SacFilter>, exclude: &[u32], what: &str) -> R<()> {
        while !rem.is_paid() {
            let st = self.s();
            let mut options = vec![];
            let mut pool_sorted = st.players[p as usize].pool.clone();
            pool_sorted.sort();
            for &(color, n) in &pool_sorted {
                if n <= 0 || !rem.useful(color) {
                    continue;
                }
                let mut r2 = rem.clone();
                r2.apply(color);
                let mut pool2 = st.players[p as usize].pool.clone();
                for e in pool2.iter_mut() {
                    if e.0 == color {
                        e.1 -= 1;
                    }
                }
                if st.cost_feasible(p, &r2, sac_filter, exclude, Some(&pool2), &[]) {
                    let c = color_str(color);
                    options.push(opt(format!("Pay with floating {c}"), vec![s("pay"), s("pool"), s(c)], Val::Pool(color)));
                }
            }
            let sources: Vec<CIdx> = st.mana_sources(p, exclude).into_iter().map(|(c, _)| c).collect();
            for card in st.dedupe_by_equiv(sources) {
                let c = st.c(card);
                let ab = &c.face().abilities[st.mana_ability(card)];
                let n = st.mana_amount(card, ab);
                for &color in ab.mana.as_ref().unwrap() {
                    if !rem.useful(color) {
                        continue;
                    }
                    let mut r2 = rem.clone();
                    r2.apply(color);
                    let gone: Vec<u32> = if ab.sac_self { vec![c.oid] } else { vec![] };
                    let mut excl = exclude.to_vec();
                    excl.push(c.oid);
                    // The extra units float.
                    let pool2 = if n > 1 { pool_with(&st.players[p as usize].pool, color, n - 1) } else { vec![] };
                    let pool_arg = if n > 1 { Some(&pool2[..]) } else { None };
                    if st.cost_feasible(p, &r2, sac_filter, &excl, pool_arg, &gone) {
                        let verb = if ab.sac_self { "Sacrifice" } else { "Tap" };
                        let cs = color_str(color);
                        options.push(opt(format!("{verb} {}#{} for {cs}", c.name(), c.oid), vec![s("pay"), s("source"), s(c.name()), s(cs)], Val::Source(card, color)));
                    }
                }
            }
            let filters: Vec<CIdx> = st.mana_filters(p, exclude).into_iter().map(|(c, _)| c).collect();
            for card in st.dedupe_by_equiv(filters) {
                let c = st.c(card);
                let ab = c.face().abilities.iter().find(|a| a.is_filter()).unwrap();
                for &color in ab.mana.as_ref().unwrap() {
                    if rem.colored_get(color) <= 0 {
                        continue;
                    }
                    let r2 = filtered(&rem, color, ab.cost.generic);
                    let mut excl = exclude.to_vec();
                    excl.push(c.oid);
                    if st.cost_feasible(p, &r2, sac_filter, &excl, None, &[]) {
                        let cs = color_str(color);
                        options.push(opt(format!("Activate {}#{} for {cs}", c.name(), c.oid), vec![s("pay"), s("filter"), s(c.name()), s(cs)], Val::Filter(card, color)));
                    }
                }
            }
            let rem_s = rem.to_string();
            let auto = if st.auto_mana { st.auto_pay_index(p, &rem, &options) } else { None };
            let choice = match auto {
                Some(i) => {
                    if st.logging {
                        let m = format!("  p{p} pay_mana: {} (auto)", options[i].label);
                        st.log.push(m);
                    }
                    options.into_iter().nth(i).unwrap().value
                }
                None => {
                    st.paying = Some((rem.clone(), sac_filter, exclude.to_vec()));
                    let v = self.ask(p, Kind::PayMana, || format!("Pay {rem_s} for {what}"), options)?;
                    self.s().paying = None;
                    v
                }
            };
            match choice {
                Val::Pool(c) => {
                    self.s().players[p as usize].pool_spend(c);
                    rem.apply(c);
                }
                Val::Source(card, color) => {
                    let st = self.s();
                    let ai = st.mana_ability(card);
                    let n = st.mana_amount(card, &st.c(card).face().abilities[ai]);
                    st.activate_mana_ability(card, ai);
                    if !rem.apply(color) {
                        return rules("mana unit could not be applied");
                    }
                    if n > 1 {
                        st.players[p as usize].pool_add(color, n - 1);
                    }
                }
                Val::Filter(card, color) => {
                    let st = self.s();
                    let ab = st.c(card).face().abilities.iter().find(|a| a.is_filter()).unwrap();
                    let (tap, once, generic) = (ab.tap, ab.once_per_turn, ab.cost.generic);
                    if tap {
                        st.cm(card).tapped = true;
                    }
                    if once {
                        st.cm(card).mana_used_turn = st.turn;
                    }
                    st.push_log_lazy(|s| format!("p{p} activates {}#{} for {}", s.c(card).name(), s.c(card).oid, color_str(color)));
                    rem = filtered(&rem, color, generic);
                }
                _ => unreachable!(),
            }
        }
        Ok(())
    }

    /// 'You may pay {cost}': returns whether it was paid.
    fn optional_payment(&mut self, p: u8, cost: &ManaCost, prompt: String) -> R<bool> {
        let mut options = vec![opt("Don't pay".into(), vec![s("pay_optional"), s("no")], Val::Bool(false))];
        if self.s().can_afford(p, cost) {
            options.push(opt(format!("Pay {}", cost.to_string()), vec![s("pay_optional"), s("yes")], Val::Bool(true)));
        }
        let pr = prompt.clone();
        let pay = matches!(self.ask(p, Kind::YesNo, move || pr, options)?, Val::Bool(true));
        if pay {
            self.pay_mana(p, Remaining::of(cost), None, &[], &prompt)?;
        }
        Ok(pay)
    }

    /// `Game.choose_discard`: `p` discards a card of their choice from hand (if any).
    fn choose_discard(&mut self, p: u8, what: &str) -> R<Option<CIdx>> {
        if self.s().players[p as usize].hand.is_empty() {
            return Ok(None);
        }
        let options = self.s().hand_card_options(p, "discard");
        let c = match self.ask(p, Kind::ChooseCard, || format!("{what}: discard a card"), options)? {
            Val::Card(c) => c,
            _ => unreachable!(),
        };
        self.s().discard(c);
        Ok(Some(c))
    }

    /// Returns the sacrificed permanent's mana value.
    fn choose_sacrifice(&mut self, p: u8, flt: SacFilter, what: &str) -> R<i32> {
        let st = self.s();
        let cands = st.dedupe_by_equiv(st.sac_candidates(p, flt, &[]));
        let mut options: Vec<Opt> = cands
            .iter()
            .map(|&c| {
                let card = st.c(c);
                opt(format!("Sacrifice {}#{}", card.name(), card.oid), vec![s("sacrifice"), s(card.name())], Val::Card(c))
            })
            .collect();
        // CR 601.2g-h: mana abilities are activated before costs are paid, so
        // an untapped mana source can be tapped for mana and then sacrificed
        // to the same cost; the mana floats (`Game._choose_sacrifice`).
        let tappable: Vec<(CIdx, usize)> = st.mana_sources(p, &[]).into_iter().filter(|&(c, ai)| !st.c(c).face().abilities[ai].sac_self).collect();
        for &c in &cands {
            let Some(&(_, ai)) = tappable.iter().find(|&&(t, _)| t == c) else { continue };
            let card = st.c(c);
            for &color in card.face().abilities[ai].mana.as_ref().unwrap() {
                let cs = color_str(color);
                options.push(opt(
                    format!("Tap {}#{} for {cs}, then sacrifice it", card.name(), card.oid),
                    vec![s("sacrifice"), s(card.name()), s("tap"), s(cs)],
                    Val::Source(c, color),
                ));
            }
        }
        let fname = flt.name().replace('_', " ");
        let article = if fname.starts_with(['a', 'e', 'i', 'o', 'u']) { "an" } else { "a" };
        let c = match self.ask(p, Kind::Sacrifice, || format!("Sacrifice {article} {fname} for {what}"), options)? {
            Val::Card(c) => c,
            Val::Source(c, color) => {
                let st = self.s();
                let ai = st.mana_ability(c);
                let n = st.mana_amount(c, &st.c(c).face().abilities[ai]);
                st.activate_mana_ability(c, ai);
                st.players[p as usize].pool_add(color, n);
                st.push_log_lazy(|s| format!("p{p} taps {}#{} for {}", s.c(c).name(), s.c(c).oid, color_str(color)));
                c
            }
            _ => unreachable!(),
        };
        let st = self.s();
        let mv = st.c(c).defn().mana_value();
        st.sacrifice(c);
        Ok(mv)
    }

    // ------------------------------------------------------------------
    // Casting spells and activating abilities
    // ------------------------------------------------------------------

    fn cast(&mut self, p: u8, card: CIdx, mode: Method, choice: Option<u8>) -> R<()> {
        let st = self.s();
        let d = st.c(card).face();
        let from_zone = st.c(card).zone;
        let face = if mode == Method::Omen { db().def(d.back.unwrap()) } else { d };
        let name = face.name.clone();
        let effect: Option<&'static [Op]> = match choice {
            _ if mode == Method::Overload => d.overload_effect.as_deref(),
            None => face.effect.as_deref(),
            Some(i) => Some(&d.modes[i as usize].effect),
        };
        let target_specs = st.mode_targets(card, mode, choice);
        st.move_card(card, Zone::Stack, Some(p), Pos::Top, None, false);
        if mode == Method::Omen {
            st.cm(card).transformed = true; // on the stack it is the omen face (Roost Seek)
        }
        let name = if mode == Method::Prototype {
            st.cm(card).prototyped = true;
            st.c(card).name().to_string()
        } else {
            name
        };
        let sid = st.c(card).oid;
        st.stack.push(StackItem {
            sid,
            kind: SKind::Spell,
            controller: p,
            name,
            effect,
            target_specs,
            targets: vec![],
            card: Some(card),
            source: None,
            method: mode,
            cast_from: from_zone,
            x: 0,
            data: Data::default(),
        });
        st.push_log_lazy(|s| format!("p{p} casts {} ({}) from {}", s.c(card).name(), mode.name(), from_zone.name()));
        let base = st.mode_cost(card, mode).unwrap().clone();
        let reduction = st.cost_reduction(p, card);
        let cname = st.c(card).name();
        let add_sac = st.mode_additional_sac(card, mode);
        if base.x != 0 {
            let mut options = vec![];
            let mut x = 0;
            while st.cost_feasible(p, &Remaining::of(&base.with_x(x).reduced(reduction)), add_sac, &[], None, &[]) {
                options.push(opt(format!("X={x}"), vec![s("x"), KI::I(x as i64)], Val::Int(x)));
                x += 1;
            }
            let xv = match self.ask(p, Kind::ChooseX, || format!("Choose X for {cname}"), options)? {
                Val::Int(v) => v,
                _ => unreachable!(),
            };
            let st = self.s();
            let pos = st.stack_pos(sid).unwrap();
            st.stack[pos].x = xv;
        }
        self.choose_targets(p, sid, None, false)?;
        let st = self.s();
        let xv = st.stack[st.stack_pos(sid).unwrap()].x;
        let cost = base.with_x(xv).reduced(reduction);
        self.pay_mana(p, Remaining::of(&cost), add_sac, &[], cname)?;
        if mode == Method::Phyrexian {
            let st = self.s();
            st.players[p as usize].life -= d.phyrexian_life;
            let life = d.phyrexian_life;
            st.push_log_lazy(|_| format!("p{p} pays {life} life for {cname}"));
        }
        if let Some(flt) = add_sac {
            let mv = self.choose_sacrifice(p, flt, cname)?;
            if d.additional_sac.is_some() {
                let st = self.s();
                let pos = st.stack_pos(sid).unwrap();
                st.stack[pos].data.sacrificed_mv = Some(mv);
            }
        }
        if d.additional_discard {
            let gone = self.choose_discard(p, cname)?.expect("casting checked the hand");
            let st = self.s();
            let land = st.is_land(st.c(gone));
            let pos = st.stack_pos(sid).unwrap();
            st.stack[pos].data.discarded_land = Some(land);
        }
        if d.additional_power {
            self.choose_power(p, sid, card)?;
        }
        if d.additional_choose_creature {
            let options = self.s().creature_choices(p, card);
            let chosen = match self.ask(p, Kind::ChooseCard, || format!("{cname}: choose a creature you control or reveal a creature card"), options)? {
                Val::Card(c) => c,
                _ => unreachable!(),
            };
            let st = self.s();
            if st.c(chosen).zone == Zone::Hand {
                st.cm(chosen).known_to = BOTH;
            }
            let snap = st.c(chosen).clone();
            let pos = st.stack_pos(sid).unwrap();
            st.stack[pos].data.chosen = Some(Box::new(snap));
        }
        if let Some((flt, n)) = self.s().mode_sac(card, mode) {
            for _ in 0..n {
                self.choose_sacrifice(p, flt, cname)?;
            }
        }
        if mode == Method::Alternative && d.alternative_reveal {
            let st = self.s();
            for i in 0..st.players[p as usize].hand.len() {
                let h = st.players[p as usize].hand[i];
                st.cm(h).known_to = BOTH;
            }
            st.push_log_lazy(|s| {
                let names: Vec<&str> = s.players[p as usize].hand.iter().map(|&h| s.c(h).name()).collect();
                format!("p{p} reveals their hand for {cname}: {}", if names.is_empty() { "empty".to_string() } else { names.join(", ") })
            });
        }
        if mode == Method::Escape {
            self.exile_from_graveyard(p, d.escape_exile, cname)?;
        }
        if mode == Method::Evidence {
            self.collect_evidence(p, d.collect_evidence, cname)?;
        }
        self.s().emit_cast(sid);
        self.s().emit_targeted(sid);
        Ok(())
    }

    /// `Game._choose_power`: Monstrous Emergence's additional cost.
    fn choose_power(&mut self, p: u8, sid: u32, card: CIdx) -> R<()> {
        let st = self.s();
        let mut options = vec![];
        for c in st.power_sources(p, card) {
            let cc = st.c(c);
            if cc.zone == Zone::Battlefield {
                options.push(opt(format!("Choose {}#{}", cc.name(), cc.oid), vec![s("choose_power"), s("battlefield"), s(cc.name())], Val::Card(c)));
            } else {
                options.push(opt(format!("Reveal {}", cc.name()), vec![s("choose_power"), s("hand"), s(cc.name())], Val::Card(c)));
            }
        }
        let iname = st.stack[st.stack_pos(sid).unwrap()].name.clone();
        let c = match self.ask(p, Kind::ChooseCard, || format!("{iname}: choose a creature you control or reveal a creature card"), options)? {
            Val::Card(c) => c,
            _ => unreachable!(),
        };
        let st = self.s();
        let pos = st.stack_pos(sid).unwrap();
        let power = if st.c(c).zone == Zone::Battlefield {
            let pw = st.power(st.c(c));
            st.stack[pos].data.chosen_oid = Some(st.c(c).oid);
            pw
        } else {
            st.cm(c).known_to = BOTH;
            st.c(c).face().power.unwrap_or(0)
        };
        st.stack[pos].data.power = Some(power);
        st.push_log_lazy(|s| format!("p{p} chooses {} (power {power})", s.c(c).name()));
        Ok(())
    }

    /// Return a Forest you control to its owner's hand as a cost (Quirion Ranger).
    fn choose_return_land(&mut self, p: u8, what: &str) -> R<()> {
        let st = self.s();
        let cands = st.dedupe_by_equiv(st.return_land_candidates(p));
        let options = cands
            .iter()
            .map(|&c| {
                let card = st.c(c);
                opt(format!("Return {}#{}", card.name(), card.oid), vec![s("return_land"), s(card.name())], Val::Card(c))
            })
            .collect();
        if let Val::Card(c) = self.ask(p, Kind::Sacrifice, || format!("Return a forest you control to its owner's hand for {what}"), options)? {
            self.s().mv(c, Zone::Hand);
        }
        Ok(())
    }

    /// 725.2: the player takes the initiative (again, if they had it) and ventures into Undercity.
    fn take_initiative(&mut self, p: u8) -> R<()> {
        let st = self.s();
        st.initiative = Some(p);
        st.push_log_lazy(|_| format!("p{p} takes the initiative"));
        self.venture(p)
    }

    /// 701.49: move the venture marker into the next room and trigger it.
    fn venture(&mut self, p: u8) -> R<()> {
        let d = db();
        let dungeon = d.def(d.undercity.expect("no Undercity"));
        let room = self.s().players[p as usize].dungeon_room;
        let nxt: &Vec<usize> = match room {
            Some(r) => &d.room_next[r],
            None => &d.room_next[0],
        };
        let new = if room.is_none() || nxt.is_empty() {
            0
        } else if nxt.len() == 1 {
            nxt[0]
        } else {
            let options = nxt.iter().map(|&r| {
                let n = dungeon.triggers[r].name.as_str();
                opt(format!("Venture into {n}"), vec![s("venture"), s(n)], Val::Name(n))
            }).collect();
            let cur = dungeon.triggers[room.unwrap()].name.as_str();
            match self.ask(p, Kind::ChooseMode, || format!("Venture into Undercity: choose the room after {cur}"), options)? {
                Val::Name(n) => dungeon.triggers.iter().position(|t| t.name == n).unwrap(),
                _ => unreachable!(),
            }
        };
        let st = self.s();
        st.players[p as usize].dungeon_room = Some(new);
        let rn = dungeon.triggers[new].name.as_str();
        st.push_log_lazy(|_| format!("p{p} ventures into {rn}"));
        let src = Src::Snap(Box::new(st.dungeon_card(p)));
        st.pending.push(PendingTrigger { controller: p, source: src, tdef: &dungeon.triggers[new], data: Data::default() });
        Ok(())
    }

    /// Plot (702.170): a special action. Pay the plot cost, exile the card
    /// face up; it can be cast for free as a sorcery on a later turn.
    fn plot(&mut self, p: u8, card: CIdx) -> R<()> {
        let st = self.s();
        let d = st.c(card).face();
        let name = d.name.as_str();
        st.push_log_lazy(|_| format!("p{p} plots {name}"));
        self.pay_mana(p, Remaining::of(d.plot.as_ref().unwrap()), None, &[], &format!("plot {name}"))?;
        let st = self.s();
        let new = st.mv(card, Zone::Exile).unwrap();
        st.cm(new).plotted_turn = st.turn;
        Ok(())
    }

    /// Collect evidence N (701.59): exile cards with total mana value N or
    /// more from your graveyard, one at a time (casting checked the total).
    fn collect_evidence(&mut self, p: u8, n: i32, what: &str) -> R<()> {
        let mut total = 0;
        while total < n {
            let st = self.s();
            let options = st
                .dedupe_by_name(st.players[p as usize].graveyard.iter().copied())
                .into_iter()
                .map(|c| {
                    let nm = st.c(c).name();
                    opt(format!("Exile {nm}"), vec![s("exile_gy"), s(nm)], Val::Card(c))
                })
                .collect();
            let shown = total;
            match self.ask(p, Kind::ExileFromGy, || format!("Collect evidence {n} for {what}: exile a card from your graveyard ({shown}/{n})"), options)? {
                Val::Card(c) => {
                    let st = self.s();
                    total += st.c(c).face().mana_value();
                    st.mv(c, Zone::Exile);
                }
                _ => unreachable!(),
            }
        }
        Ok(())
    }

    fn exile_from_graveyard(&mut self, p: u8, n: i32, what: &str) -> R<()> {
        for i in 0..n {
            let st = self.s();
            let options = st
                .dedupe_by_name(st.players[p as usize].graveyard.iter().copied())
                .into_iter()
                .map(|c| {
                    let nm = st.c(c).name();
                    opt(format!("Exile {nm}"), vec![s("exile_gy"), s(nm)], Val::Card(c))
                })
                .collect();
            if let Val::Card(c) = self.ask(p, Kind::ExileFromGy, || format!("Exile a card from your graveyard for {what} ({}/{n})", i + 1), options)? {
                self.s().mv(c, Zone::Exile);
            }
        }
        Ok(())
    }

    fn activate(&mut self, p: u8, card: CIdx, index: usize) -> R<()> {
        let st = self.s();
        let ab = &st.c(card).face().abilities[index];
        let sid = st.new_id();
        let name = format!("{}: {}", st.c(card).name(), ab.name);
        let oid = st.c(card).oid;
        st.stack.push(StackItem {
            sid,
            kind: SKind::Ability,
            controller: p,
            name: name.clone(),
            effect: ab.effect.as_deref(),
            target_specs: ab.targets.clone(),
            targets: vec![],
            card: None,
            source: Some(Src::Live(card)),
            method: Method::Normal,
            cast_from: Zone::Hand,
            x: 0,
            data: Data { source_oid: Some(oid), ..Default::default() },
        });
        st.push_log_lazy(|_| format!("p{p} activates {name}"));
        if ab.once_per_turn {
            let t = st.turn;
            st.cm(card).once_used_turn = t;
        }
        if ab.x_reveal {
            let n = self.s().red_cards_in_hand(p).len() as i32;
            let options = (0..=n).map(|v| opt(format!("X={v}"), vec![s("x"), KI::I(v as i64)], Val::Int(v))).collect();
            let xv = match self.ask(p, Kind::ChooseX, || format!("Choose X for {name}"), options)? {
                Val::Int(v) => v,
                _ => unreachable!(),
            };
            let st = self.s();
            let pos = st.stack_pos(sid).unwrap();
            st.stack[pos].x = xv;
        }
        let mut cost = ab.cost.clone();
        if ab.x_target_mv != 0 {
            let allowed = move |st: &State, r: Ref| st.x_target_affordable(p, card, index, r);
            self.choose_targets(p, sid, Some(&allowed), false)?;
            let st = self.s();
            let pos = st.stack_pos(sid).unwrap();
            let t = st.stack[pos].targets[0];
            cost = st.x_target_cost(card, index, t);
            let mv = match t {
                Ref::Perm(oid) => st.c(st.perm(oid).unwrap()).face().mana_value(),
                _ => unreachable!(),
            };
            st.stack[pos].x = mv;
        } else {
            self.choose_targets(p, sid, None, false)?;
        }
        let exclude: Vec<u32> = if ab.tap { vec![self.s().c(card).oid] } else { vec![] };
        self.pay_mana(p, Remaining::of(&cost), ab.sac_other, &exclude, &name)?;
        if ab.tap {
            self.s().cm(card).tapped = true;
        }
        if let Some(flt) = ab.sac_other {
            self.choose_sacrifice(p, flt, &name)?;
        }
        if ab.discard_other {
            self.choose_discard(p, &name)?;
        }
        if ab.return_forest {
            self.choose_return_land(p, &name)?;
        }
        if ab.tap_other {
            let st = self.s();
            let cands = st.dedupe_by_equiv(st.tap_other_candidates(p, card));
            let options = cands
                .iter()
                .map(|&c| {
                    let cd = st.c(c);
                    opt(format!("Tap {}#{}", cd.name(), cd.oid), vec![s("tap_cost"), s(cd.name())], Val::Card(c))
                })
                .collect();
            let t = match self.ask(p, Kind::ChooseCard, || format!("Tap another untapped creature you control for {name}"), options)? {
                Val::Card(c) => c,
                _ => unreachable!(),
            };
            let st = self.s();
            st.cm(t).tapped = true;
            let pw = st.power(st.c(t));
            let pos = st.stack_pos(sid).unwrap();
            st.stack[pos].data.tapped_power = Some(pw);
        }
        if ab.x_reveal {
            let st = self.s();
            let xv = st.stack[st.stack_pos(sid).unwrap()].x;
            self.reveal_red(p, xv, &name)?;
        }
        let st = self.s();
        let snap = st.c(card).clone();
        let pos = st.stack_pos(sid).unwrap();
        st.stack[pos].source = Some(Src::Snap(Box::new(snap)));
        if ab.discard_self {
            st.discard(card);
        }
        if ab.sac_self && st.c(card).zone == Zone::Battlefield {
            st.sacrifice(card);
        }
        if ab.exile_self && matches!(st.c(card).zone, Zone::Battlefield | Zone::Graveyard) {
            st.mv(card, Zone::Exile);
        }
        st.emit_targeted(sid);
        Ok(())
    }

    /// Reveal `n` red cards from hand, one at a time (a cost: Martyr of Ashes).
    fn reveal_red(&mut self, p: u8, n: i32, what: &str) -> R<()> {
        let mut revealed: Vec<CIdx> = vec![];
        for i in 0..n {
            let st = self.s();
            let cands = st.dedupe_by_name(st.red_cards_in_hand(p).into_iter().filter(|c| !revealed.contains(c)));
            let options = cands
                .iter()
                .map(|&c| {
                    let nm = st.c(c).name();
                    opt(format!("Reveal {nm}"), vec![s("reveal"), s(nm)], Val::Card(c))
                })
                .collect();
            let c = match self.ask(p, Kind::ChooseCard, || format!("{what}: reveal a red card ({}/{n})", i + 1), options)? {
                Val::Card(c) => c,
                _ => unreachable!(),
            };
            self.s().cm(c).known_to = BOTH;
            revealed.push(c);
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Resolution
    // ------------------------------------------------------------------

    fn resolve_top(&mut self) -> R<()> {
        let item = self.s().stack.last().unwrap().clone();
        self.log(|_| format!("resolve {}", item.name));
        let st = self.s();
        if !item.targets.is_empty() && !(0..item.targets.len()).any(|i| st.target_legal(&item, i)) {
            if !(item.kind == SKind::Spell && item.method == Method::Bestow) {
                st.push_log_lazy(|_| format!("{} fizzles (no legal targets)", item.name));
                st.stack.pop();
                if item.kind == SKind::Spell {
                    self.spell_leaves_stack(&item);
                }
                return Ok(());
            }
        }
        if item.kind == SKind::Spell {
            let card = item.card.unwrap();
            if self.s().c(card).face().is_permanent_card() {
                self.s().stack.pop();
                self.resolve_permanent_spell(&item);
            } else {
                self.run_effect(item.effect, &item)?;
                if let Some(pos) = self.s().stack_pos(item.sid) {
                    self.s().stack.remove(pos);
                    self.spell_leaves_stack_resolved(&item, true);
                }
            }
        } else {
            self.run_effect(item.effect, &item)?;
            if let Some(pos) = self.s().stack_pos(item.sid) {
                self.s().stack.remove(pos);
            }
        }
        Ok(())
    }

    fn spell_leaves_stack(&mut self, item: &StackItem) {
        self.spell_leaves_stack_resolved(item, false)
    }

    /// A resolved omen is shuffled into its owner's library.
    fn spell_leaves_stack_resolved(&mut self, item: &StackItem, resolved: bool) {
        let st = self.s();
        let card = item.card.unwrap();
        if resolved && item.method == Method::Omen {
            st.mv(card, Zone::Library);
            let owner = st.c(card).owner as usize;
            st.shuffle(owner);
            return;
        }
        let dest = if item.method == Method::Flashback { Zone::Exile } else { Zone::Graveyard };
        st.mv(card, dest);
    }

    fn resolve_permanent_spell(&mut self, item: &StackItem) {
        let st = self.s();
        let card = item.card.unwrap();
        let host = if item.method == Method::Bestow {
            match st.target(item, 0) {
                Some(Tgt::Card(h)) => Some(h),
                _ => None,
            }
        } else {
            None
        };
        let new = st.move_card(card, Zone::Battlefield, Some(item.controller), Pos::Top, None, false).unwrap();
        if item.method == Method::Prototype {
            st.cm(new).prototyped = true;
        }
        if st.c(card).face().etb_x_counters {
            st.cm(new).counters = item.x;
        }
        if let Some(h) = host {
            let hoid = st.c(h).oid;
            st.cm(new).attached_to = Some(hoid);
        }
        st.emit_etb(new, Some(item.method));
    }

    pub fn counter(&mut self, sid: u32) {
        let st = self.s();
        if let Some(pos) = st.stack_pos(sid) {
            st.push_log_lazy(|s| format!("{} is countered", s.stack[pos].name));
            let item = st.stack.remove(pos);
            if item.kind == SKind::Spell {
                self.spell_leaves_stack(&item);
            }
        }
    }

    // ------------------------------------------------------------------
    // Library helpers used by card effects
    // ------------------------------------------------------------------

    fn search_library(&mut self, p: u8, filter: &SearchFilter, to_battlefield: bool, what: &str, tapped: bool, reveal: bool) -> R<Option<CIdx>> {
        let st = self.s();
        let lib = st.players[p as usize].library.clone();
        let cands = st.dedupe_by_name(lib.into_iter().filter(|&c| search_matches(filter, st.c(c))));
        let mut options = vec![opt("Find nothing".into(), vec![s("search"), KI::N], Val::None)];
        for c in cands {
            let n = st.c(c).name();
            options.push(opt(format!("Find {n}"), vec![s("search"), s(n)], Val::Card(c)));
        }
        let found = match self.ask(p, Kind::ChooseCard, || format!("Search your library for {what}"), options)? {
            Val::Card(c) => Some(c),
            _ => None,
        };
        let st = self.s();
        if let Some(c) = found {
            if to_battlefield {
                st.put_onto_battlefield(c, p, tapped);
            } else {
                st.move_card(c, Zone::Hand, None, Pos::Top, if reveal { Some(BOTH) } else { None }, false);
            }
        }
        st.shuffle(p as usize);
        Ok(found)
    }

    fn scry(&mut self, p: u8, n: usize) -> R<()> {
        let st = self.s();
        let top: Vec<CIdx> = st.players[p as usize].library.iter().take(n).copied().collect();
        for &c in &top {
            st.cm(c).known_to |= pbit(p);
        }
        if top.len() > 1 {
            return self.scry_many(p, top);
        }
        if top.is_empty() {
            return Ok(());
        }
        let c = top[0];
        let n = st.c(c).name();
        let options = vec![
            opt(format!("Keep {n} on top"), vec![s("scry"), s("top")], Val::Top),
            opt(format!("Put {n} on the bottom"), vec![s("scry"), s("bottom")], Val::Bottom),
        ];
        if let Val::Bottom = self.ask(p, Kind::ChooseMode, || format!("Scry 1: {n}"), options)? {
            let lib = &mut self.s().players[p as usize].library;
            let pos = lib.iter().position(|&x| x == c).unwrap();
            lib.remove(pos);
            lib.push(c);
        }
        Ok(())
    }

    /// game.py `_scry_many`: scry N > 1 as one ORDER decision.
    fn scry_many(&mut self, p: u8, top: Vec<CIdx>) -> R<()> {
        let st = self.s();
        let n = top.len();
        let mut options = vec![];
        let mut seen: Vec<(Vec<&'static str>, Vec<&'static str>)> = vec![];
        for mask in 0u32..(1 << n) {
            let ups: Vec<CIdx> = top.iter().enumerate().filter(|(i, _)| mask >> i & 1 == 0).map(|(_, &c)| c).collect();
            let downs: Vec<CIdx> = top.iter().enumerate().filter(|(i, _)| mask >> i & 1 == 1).map(|(_, &c)| c).collect();
            for pu in permutations(&ups) {
                for pd in permutations(&downs) {
                    let tn: Vec<&'static str> = pu.iter().map(|&c| st.c(c).name()).collect();
                    let bn: Vec<&'static str> = pd.iter().map(|&c| st.c(c).name()).collect();
                    let k = (tn.clone(), bn.clone());
                    if seen.contains(&k) {
                        continue;
                    }
                    seen.push(k);
                    let ts = if tn.is_empty() { "-".to_string() } else { tn.join(", ") };
                    let bs = if bn.is_empty() { "-".to_string() } else { bn.join(", ") };
                    let mut key = vec![s("scry"), s("top")];
                    key.extend(tn.iter().map(|x| s(x)));
                    key.push(s("bottom"));
                    key.extend(bn.iter().map(|x| s(x)));
                    options.push(opt(format!("Scry: top {ts}; bottom {bs}"), key, Val::Scry(pu.clone(), pd)));
                }
            }
        }
        if let Val::Scry(pu, pd) = self.ask(p, Kind::Order, || format!("Scry {n}: top (first = top card) and bottom (last = bottom card)"), options)? {
            let lib = &mut self.s().players[p as usize].library;
            lib.drain(..n);
            for (i, c) in pu.into_iter().enumerate() {
                lib.insert(i, c);
            }
            lib.extend(pd);
        }
        Ok(())
    }

    /// game.py `surveil`: surveil 1.
    fn surveil(&mut self, p: u8) -> R<()> {
        let st = self.s();
        let c = match st.players[p as usize].library.first() {
            Some(&c) => c,
            None => return Ok(()),
        };
        st.cm(c).known_to |= pbit(p);
        let n = st.c(c).name();
        let options = vec![
            opt(format!("Keep {n} on top"), vec![s("surveil"), s("top")], Val::Top),
            opt(format!("Put {n} into your graveyard"), vec![s("surveil"), s("graveyard")], Val::Bottom),
        ];
        if let Val::Bottom = self.ask(p, Kind::ChooseMode, || format!("Surveil 1: {n}"), options)? {
            self.s().mv(c, Zone::Graveyard);
        }
        Ok(())
    }

    /// game.py `dig` (op look_top): look at the top n, maybe take a matching card, the rest to the bottom.
    fn look_top(&mut self, p: u8, n: i32, filter: &SearchFilter, what: &str, name: &str) -> R<()> {
        let st = self.s();
        let top: Vec<CIdx> = st.players[p as usize].library.iter().take(n as usize).copied().collect();
        for &c in &top {
            st.cm(c).known_to |= pbit(p);
        }
        let mut options = vec![opt("Take nothing".into(), vec![s("dig"), KI::N], Val::None)];
        for c in st.dedupe_by_name(top.iter().copied().filter(|&c| search_matches(filter, st.c(c)))) {
            let nm = st.c(c).name();
            options.push(opt(format!("Take {nm}"), vec![s("dig"), s(nm)], Val::Card(c)));
        }
        let found = match self.ask(p, Kind::ChooseCard, || format!("{name}: reveal {what} and put it into your hand"), options)? {
            Val::Card(c) => Some(c),
            _ => None,
        };
        let st = self.s();
        if let Some(c) = found {
            st.move_card(c, Zone::Hand, None, Pos::Top, Some(BOTH), false);
        }
        for c in top {
            if Some(c) != found {
                st.move_card(c, Zone::Library, None, Pos::Bottom, Some(pbit(p)), false);
            }
        }
        Ok(())
    }

    /// game.py `cascade` (702.85).
    fn cascade(&mut self, p: u8, mv: i32) -> R<()> {
        let mut exiled: Vec<CIdx> = vec![];
        let mut hit: Option<CIdx> = None;
        loop {
            let st = self.s();
            let top = match st.players[p as usize].library.first() {
                Some(&c) => c,
                None => break,
            };
            let c = st.mv(top, Zone::Exile).unwrap();
            exiled.push(c);
            if !st.is_land(st.c(c)) && st.c(c).face().mana_value() < mv {
                hit = Some(c);
                break;
            }
        }
        self.log(|st| {
            let names = exiled.iter().map(|&c| py_repr_str(st.c(c).name())).collect::<Vec<_>>().join(", ");
            format!("p{p} cascades into {}, exiling [{names}]", hit.map_or("nothing", |h| st.c(h).name()))
        });
        let mut cast: Option<Option<u8>> = None;
        if let Some(h) = hit {
            let st = self.s();
            let n = st.c(h).name();
            let mut options = vec![opt(format!("Don't cast {n}"), vec![s("cascade"), s("no")], Val::None)];
            let face = st.c(h).face();
            if !face.modes.is_empty() {
                for (i, sm) in face.modes.iter().enumerate() {
                    if st.can_cast(p, h, Method::Cascade, Some(i as u8)) {
                        options.push(opt(format!("Cast {n} ({})", sm.name), vec![s("cascade"), s("cast"), s(sm.name.as_str())], Val::Cast(h, Method::Cascade, Some(i as u8))));
                    }
                }
            } else if st.can_cast(p, h, Method::Cascade, None) {
                options.push(opt(format!("Cast {n}"), vec![s("cascade"), s("cast")], Val::Cast(h, Method::Cascade, None)));
            }
            if let Val::Cast(_, _, ch) = self.ask(p, Kind::YesNo, || format!("Cascade: cast {n} without paying its mana cost?"), options)? {
                cast = Some(ch);
            }
        }
        let st = self.s();
        let mut rest: Vec<CIdx> = exiled.into_iter().filter(|&c| !(cast.is_some() && Some(c) == hit)).collect();
        st.rng.shuffle(&mut rest);
        for c in rest {
            st.move_card(c, Zone::Library, None, Pos::Bottom, None, false);
        }
        if let Some(ch) = cast {
            self.cast(p, hit.unwrap(), Method::Cascade, ch)?;
        }
        Ok(())
    }

    fn explore(&mut self, creature: CIdx) -> R<()> {
        let st = self.s();
        let p = st.c(creature).controller;
        let lib = st.players[p as usize].library.clone();
        if !lib.is_empty() && st.is_land(st.c(lib[0])) {
            st.move_card(lib[0], Zone::Hand, None, Pos::Top, Some(BOTH), false);
            return Ok(());
        }
        if !lib.is_empty() {
            st.cm(lib[0]).known_to = BOTH;
        }
        if let Some(live) = st.live(&st.c(creature).clone()) {
            st.cm(live).counters += 1;
        }
        if !lib.is_empty() {
            let card = lib[0];
            let n = st.c(card).name();
            let options = vec![
                opt(format!("Keep {n} on top"), vec![s("explore"), s("keep")], Val::Bool(false)),
                opt(format!("Put {n} into graveyard"), vec![s("explore"), s("graveyard")], Val::Bool(true)),
            ];
            if let Val::Bool(true) = self.ask(p, Kind::YesNo, || format!("Explore: put {n} into your graveyard?"), options)? {
                self.s().mv(card, Zone::Graveyard);
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Combat
    // ------------------------------------------------------------------

    fn declare_attackers(&mut self) -> R<()> {
        let st = self.s();
        let p = st.active;
        let eligible: Vec<CIdx> = st
            .battlefield
            .iter()
            .copied()
            .filter(|&ci| {
                let c = st.c(ci);
                c.controller == p && st.is_creature(c) && !c.tapped && !c.sick
            })
            .collect();
        let refs = st.referenced_oids();
        let mut groups: Vec<Vec<CIdx>> = vec![];
        let mut index: Vec<EquivKey> = vec![];
        for c in eligible {
            let k = st.equiv_key(c, &refs);
            match index.iter().position(|x| *x == k) {
                Some(i) => groups[i].push(c),
                None => {
                    index.push(k);
                    groups.push(vec![c]);
                }
            }
        }
        // Any group with creatures left, in any order; chosen creatures attack
        // at once (the state shows the pending declaration) and tap when done.
        let mut chosen: Vec<CIdx> = vec![];
        loop {
            let st = self.s();
            let mut options = vec![opt("Done declaring attackers".into(), vec![s("attack"), KI::N], Val::None)];
            let mut subjects = vec![vec![]];
            for gi in 0..groups.len() {
                if let Some(&c) = groups[gi].iter().find(|c| !chosen.contains(c)) {
                    let card = st.c(c);
                    options.push(opt(format!("Attack with {}#{}", card.name(), card.oid), vec![s("attack"), s(card.name())], Val::Group(gi)));
                    subjects.push(vec![card.oid]);
                }
            }
            st.combat_subjects = subjects;
            match self.ask(p, Kind::DeclareAttacker, || "Declare attackers".to_string(), options)? {
                Val::Group(gi) => {
                    let c = *groups[gi].iter().find(|c| !chosen.contains(c)).unwrap();
                    chosen.push(c);
                    let st = self.s();
                    let oid = st.c(c).oid;
                    st.attackers.push(oid);
                }
                _ => break,
            }
        }
        let st = self.s();
        for &c in &chosen {
            st.cm(c).tapped = true;
        }
        if !chosen.is_empty() {
            st.push_log_lazy(|s| format!("p{p} attacks with [{}]", chosen.iter().map(|&c| s.c(c).repr()).collect::<Vec<_>>().join(", ")));
        }
        Ok(())
    }

    fn declare_blockers(&mut self) -> R<()> {
        let st = self.s();
        let d = 1 - st.active;
        let blockers: Vec<CIdx> = st
            .battlefield
            .iter()
            .copied()
            .filter(|&ci| {
                let c = st.c(ci);
                c.controller == d && st.is_creature(c) && !c.tapped
            })
            .collect();
        for (i, &b) in blockers.iter().enumerate() {
            let st = self.s();
            let bc = st.c(b);
            let later = &blockers[i + 1..];
            let attackers: Vec<CIdx> = st.attackers.iter().filter_map(|&a| st.perm(a)).filter(|&a| st.can_block(bc, st.c(a)) && st.menace_ok(a, later)).collect();
            let mut options = vec![opt(format!("{}#{} does not block", bc.name(), bc.oid), vec![s("block"), s(bc.name()), KI::N], Val::None)];
            let mut subjects = vec![vec![bc.oid]];
            let refs = st.referenced_oids();
            let mut seen: Vec<EquivKey> = vec![];
            for a in attackers {
                let k = st.equiv_key(a, &refs);
                if seen.contains(&k) {
                    continue;
                }
                seen.push(k);
                let ac = st.c(a);
                options.push(opt(format!("{}#{} blocks {}#{}", bc.name(), bc.oid, ac.name(), ac.oid), vec![s("block"), s(bc.name()), s(ac.name())], Val::Card(a)));
                subjects.push(vec![bc.oid, ac.oid]);
            }
            let (bn, bo) = (bc.name(), bc.oid);
            st.combat_subjects = subjects;
            if let Val::Card(a) = self.ask(d, Kind::DeclareBlocker, || format!("Block with {bn}#{bo}?"), options)? {
                let st = self.s();
                let aoid = st.c(a).oid;
                let boid = st.c(b).oid;
                match st.blocks.iter_mut().find(|(x, _)| *x == boid) {
                    Some(e) => e.1 = aoid,
                    None => st.blocks.push((boid, aoid)),
                }
                if !st.blocked.contains(&aoid) {
                    st.blocked.push(aoid);
                }
            }
        }
        let st = self.s();
        for aoid in st.attackers.clone() {
            let mine: Vec<u32> = st.blocks.iter().filter(|(_, at)| *at == aoid).map(|(b, _)| *b).collect();
            if let Some(a) = st.perm(aoid) {
                if mine.len() == 1 && st.has(st.c(a), "menace") {
                    // 702.111b: a lone blocker of a menace creature is not a legal block; it is undone.
                    st.blocks.retain(|(b, _)| *b != mine[0]);
                    st.blocked.retain(|x| *x != aoid);
                    st.push_log_lazy(|s| format!("menace: the block of {} by one creature is undone", s.c(a).repr()));
                }
            }
        }
        if !st.blocks.is_empty() {
            st.push_log_lazy(|s| format!("p{d} blocks: {{{}}}", s.blocks.iter().map(|(b, a)| format!("{b}: {a}")).collect::<Vec<_>>().join(", ")));
        }
        Ok(())
    }

    fn combat_damage(&mut self) -> R<()> {
        let mut assignments: Vec<(CIdx, Ref, i32)> = vec![];
        let st = self.s();
        let defender = 1 - st.active;
        for aoid in st.attackers.clone() {
            let st = self.s();
            let a = match st.perm(aoid) {
                Some(a) => a,
                None => continue,
            };
            let pw = st.power(st.c(a));
            if pw <= 0 {
                continue;
            }
            if !st.blocked.contains(&aoid) {
                assignments.push((a, Ref::Player(defender), pw));
                continue;
            }
            let blockers: Vec<CIdx> = st.blocks.iter().filter(|(_, at)| *at == aoid).filter_map(|(b, _)| st.perm(*b)).collect();
            let trample = st.has(st.c(a), "trample");
            if blockers.is_empty() {
                if trample {
                    assignments.push((a, Ref::Player(defender), pw));
                }
                continue;
            }
            if blockers.len() == 1 && !trample {
                assignments.push((a, Ref::Perm(st.c(blockers[0]).oid), pw));
                continue;
            }
            let lethal: Vec<i32> = blockers.iter().map(|&b| st.lethal(st.c(a), st.c(b))).collect();
            let split = if let Some(splits) = damage_splits(pw, &lethal, trample) {
                let mut options = vec![];
                for split in splits {
                    let mut parts: Vec<String> = split.iter().zip(&blockers).map(|(s, &b)| format!("{s} to {}", st.c(b).repr())).collect();
                    if trample { parts.push(format!("{} to player", split.last().unwrap())); }
                    let key = vec![s("damage"), KI::T(split.iter().map(|&v| v as i64).collect())];
                    options.push(opt(parts.join(", "), key, Val::Split(split)));
                }
                let (an, ao, active) = (st.c(a).name(), st.c(a).oid, st.active);
                st.combat_subjects = options.iter().map(|_| std::iter::once(ao).chain(blockers.iter().map(|&b| st.c(b).oid)).collect()).collect();
                match self.ask(active, Kind::AssignDamage, || format!("Assign {pw} damage from {an}#{ao}"), options)? {
                    Val::Split(v) => v, _ => unreachable!(),
                }
            } else {
                self.allocate_damage(a, &blockers, &lethal, pw, trample, defender)?
            };
            let st = self.s();
            for (sv, &b) in split.iter().zip(&blockers) {
                if *sv != 0 {
                    assignments.push((a, Ref::Perm(st.c(b).oid), *sv));
                }
            }
            if trample && *split.last().unwrap() != 0 {
                assignments.push((a, Ref::Player(defender), *split.last().unwrap()));
            }
        }
        let st = self.s();
        for (boid, aoid) in st.blocks.clone() {
            let (b, a) = match (st.perm(boid), st.perm(aoid)) {
                (Some(b), Some(a)) => (b, a),
                _ => continue,
            };
            let pw = st.power(st.c(b));
            if pw > 0 {
                assignments.push((b, Ref::Perm(st.c(a).oid), pw));
            }
        }
        let steal = st.initiative == Some(defender) && assignments.iter().any(|(_, t, n)| *t == Ref::Player(defender) && *n > 0);
        for (src, tgt, n) in assignments {
            let card = st.c(src).clone();
            st.deal_damage(&card, tgt, n);
        }
        if steal {
            // 725.2: combat damage to the player with the initiative takes it.
            let a = st.active;
            let src = Src::Snap(Box::new(st.dungeon_card(a)));
            st.pending.push(PendingTrigger { controller: a, source: src, tdef: &db().take_initiative, data: Data::default() });
        }
        Ok(())
    }

    fn allocate_damage(&mut self, a: CIdx, blockers: &[CIdx], lethal: &[i32], pw: i32, trample: bool, defender: u8) -> R<Vec<i32>> {
        let mut assigned = vec![0; blockers.len()];
        let (mut remaining, mut player_damage) = (pw, if trample { -1 } else { 0 });
        let recipients = (if trample { -1 } else { 0 })..blockers.len() as i32;
        for recipient in recipients {
            let st = self.s();
            let (an, ao, active) = (st.c(a).name(), st.c(a).oid, st.active);
            st.damage_allocation = Some(DamageAllocation { attacker: ao, blockers: blockers.iter().map(|&b| st.c(b).oid).collect(), lethal: lethal.to_vec(), assigned: assigned.clone(), recipient, remaining, defender, player_damage });
            let name = if recipient == -1 { "player" } else { st.c(blockers[recipient as usize]).name() };
            let target = if recipient == -1 { "player".to_string() } else { st.c(blockers[recipient as usize]).repr() };
            let options = damage_amounts(remaining, lethal, recipient, player_damage).map(|n| {
                opt(format!("{n} to {target} from {an}#{ao}"), vec![s("damage_amount"), s(an), s(name), KI::I(recipient as i64), KI::I(n as i64), KI::I(remaining as i64), KI::T(assigned.iter().map(|&v| v as i64).collect()), KI::I(player_damage as i64)], Val::Int(n))
            }).collect();
            let n = match self.ask(active, Kind::AssignDamageAmount, || format!("Assign damage from {an}#{ao} to {target} ({remaining} remaining)"), options)? { Val::Int(n) => n, _ => unreachable!() };
            remaining -= n;
            if recipient == -1 { player_damage = n; } else { assigned[recipient as usize] = n; }
        }
        self.s().damage_allocation = None;
        if trample { assigned.push(player_damage); }
        Ok(assigned)
    }

    // ------------------------------------------------------------------
    // Effects
    // ------------------------------------------------------------------

    fn run_effect(&mut self, ops: Option<&'static [Op]>, item: &StackItem) -> R<()> {
        if let Some(ops) = ops {
            for op in ops {
                self.run_op(op, item)?;
            }
        }
        Ok(())
    }

    /// cards.py `_source`: the object dealing an effect's damage, the spell
    /// itself or the (last known information of the) source of an ability.
    fn source_card(&mut self, item: &StackItem) -> Card {
        let st = self.s();
        match &item.source {
            Some(src) => st.src(src).clone(),
            None => st.c(item.card.unwrap()).clone(),
        }
    }

    fn run_op(&mut self, op: &'static Op, item: &StackItem) -> R<()> {
        let ctl = item.controller;
        match op {
            Op::Draw { n, n_cast_from_graveyard, each_controlling } => {
                let n = if item.cast_from == Zone::Graveyard { n_cast_from_graveyard.unwrap_or(*n) } else { *n };
                let st = self.s();
                match each_controlling {
                    None => st.draw(ctl as usize, n, true),
                    Some(name) => {
                        for q in [ctl, 1 - ctl] {
                            if st.battlefield.iter().any(|&c| st.c(c).controller == q && st.c(c).name() == name.as_str()) {
                                st.draw(q as usize, n, true);
                            }
                        }
                    }
                }
            }
            Op::Mill { target_player, n } => {
                if !target_player {
                    self.s().mill(ctl as usize, *n);
                } else if let Some(Tgt::Player(p)) = self.s().target(item, 0) {
                    self.s().mill(p as usize, *n);
                }
            }
            Op::CounterTarget { if_color } => {
                if let Some(Tgt::Spell(sid)) = self.s().target(item, 0) {
                    let st = self.s();
                    let colors = st.c(st.stack[st.stack_pos(sid).unwrap()].card.unwrap()).face().colors;
                    if *if_color == 0 || colors & if_color != 0 {
                        self.counter(sid);
                    }
                }
            }
            Op::CounterTargetUnlessPaid { cost } => {
                if let Some(Tgt::Spell(sid)) = self.s().target(item, 0) {
                    let st = self.s();
                    let t = &st.stack[st.stack_pos(sid).unwrap()];
                    let (tc, tn) = (t.controller, t.name.clone());
                    let paid = self.optional_payment(tc, cost, format!("{}: pay {} or {tn} is countered", item.name, cost.to_string()))?;
                    if !paid {
                        self.counter(sid);
                    }
                }
            }
            Op::DestroyTarget { if_color, mv_is_x } => {
                if let Some(Tgt::Card(c)) = self.s().target(item, 0) {
                    let st = self.s();
                    let face = st.c(c).face();
                    if (*if_color == 0 || face.colors & if_color != 0) && !(*mv_is_x && face.mana_value() != item.x) {
                        st.destroy(c);
                    }
                }
            }
            Op::BounceTarget => {
                if let Some(Tgt::Card(c)) = self.s().target(item, 0) {
                    self.s().mv(c, Zone::Hand);
                }
            }
            Op::TapTarget { skip_untap } => {
                if let Some(Tgt::Card(c)) = self.s().target(item, 0) {
                    let card = self.s().cm(c);
                    card.tapped = true;
                    card.skip_untap = card.skip_untap.max(*skip_untap);
                }
            }
            Op::GrantTarget { keywords } => {
                if let Some(Tgt::Card(c)) = self.s().target(item, 0) {
                    self.s().cm(c).temp.push(TempEffect { keywords: *keywords, power: 0, toughness: 0 });
                }
            }
            Op::CreateToken { token, n, attach_source } => {
                for _ in 0..*n {
                    let tok = self.s().create_token(ctl, *token);
                    if *attach_source {
                        let src = self.source_card(item);
                        let st = self.s();
                        let eq = st.live(&src);
                        st.attach(eq, Some(tok));
                    }
                }
            }
            Op::AttachSourceToTarget => {
                let src = self.source_card(item);
                let st = self.s();
                if let Some(Tgt::Card(host)) = st.target(item, 0) {
                    let eq = st.live(&src);
                    st.attach(eq, Some(host));
                }
            }
            Op::GainLife { n, per_storm, sacrificed_mv, source_power } => {
                if *sacrificed_mv {
                    let v = item.data.sacrificed_mv.unwrap_or(0);
                    self.s().players[ctl as usize].life += v;
                } else if *source_power {
                    let src = self.source_card(item);
                    let st = self.s();
                    let pw = match st.live(&src) {
                        Some(ci) => st.power(st.c(ci)),
                        None => st.power(&src),
                    };
                    st.players[ctl as usize].life += pw;
                } else {
                    let k = if *per_storm { item.data.storm.unwrap_or(0) } else { 1 };
                    self.s().players[ctl as usize].life += n * k;
                }
            }
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
            Op::CounterOnSource => {
                let src = self.source_card(item);
                let st = self.s();
                if let Some(live) = st.live(&src) {
                    st.cm(live).counters += 1;
                }
            }
            Op::DamageTargetFrom { index, chosen_power } => {
                let i = *index;
                if i < item.targets.len() && self.s().target_legal(item, i) {
                    let src = self.source_card(item);
                    let st = self.s();
                    let amount = if *chosen_power {
                        let ch = item.data.chosen.as_ref().expect("chosen creature");
                        match st.live(ch) {
                            Some(ci) => st.power(st.c(ci)),
                            None => st.power(ch),
                        }
                    } else {
                        item.x
                    };
                    st.deal_damage(&src, item.targets[i], amount);
                }
            }
            Op::DamageTarget { n, index, n_landfall, n_metalcraft } => {
                let i = *index;
                if i < item.targets.len() && self.s().target_legal(item, i) {
                    let src = self.source_card(item);
                    let st = self.s();
                    let mut amount = match n_landfall {
                        Some(l) if st.players[ctl as usize].landfall_turn == st.turn => *l,
                        _ => *n,
                    };
                    if let Some(m) = n_metalcraft {
                        let artifacts = st.battlefield.iter().filter(|&&c| st.c(c).controller == ctl && st.is_artifact(st.c(c))).count();
                        if artifacts >= 3 {
                            amount = *m;
                        }
                    }
                    st.deal_damage(&src, item.targets[i], amount);
                }
            }
            Op::DamageTargetController { n } => {
                if let Some(Tgt::Card(c)) = self.s().target(item, 0) {
                    let src = self.source_card(item);
                    let st = self.s();
                    let r = Ref::Player(st.c(c).controller);
                    st.deal_damage(&src, r, *n);
                }
            }
            Op::DamageEachOpponent { n, if_discarded_nonland } => {
                if *if_discarded_nonland && item.data.discarded_land.unwrap_or(true) {
                    return Ok(());
                }
                let src = self.source_card(item);
                self.s().deal_damage(&src, Ref::Player(1 - ctl), *n);
            }
            Op::DamageEachCreature { n, x, without, opponent_only } => {
                let src = self.source_card(item);
                let amount = if *x { item.x } else { *n };
                let st = self.s();
                for c in st.battlefield.clone() {
                    let card = st.c(c);
                    if !st.is_creature(card) || (*without != 0 && st.keywords(card) & without != 0) {
                        continue;
                    }
                    if *opponent_only && card.controller == ctl {
                        continue;
                    }
                    let oid = card.oid;
                    st.deal_damage(&src, Ref::Perm(oid), amount);
                }
            }
            Op::Discard { n } => {
                for _ in 0..*n {
                    if self.choose_discard(ctl, &item.name)?.is_none() {
                        break;
                    }
                }
            }
            Op::ReturnToBattlefield { tapped } => {
                let st = self.s();
                let c = item.data.card.unwrap();
                if st.c(c).zone == Zone::Graveyard && Some(st.c(c).oid) == item.data.oid {
                    let owner = st.c(c).owner;
                    st.put_onto_battlefield(c, owner, *tapped);
                }
            }
            Op::ExileAllGraveyards => {
                let st = self.s();
                for q in 0..2 {
                    for c in st.players[q].graveyard.clone() {
                        st.mv(c, Zone::Exile);
                    }
                }
            }
            Op::Madness => self.madness(item)?,
            Op::ExileTarget => {
                if let Some(Tgt::Card(c)) = self.s().target(item, 0) {
                    self.s().mv(c, Zone::Exile);
                }
            }
            Op::ExileFromGraveyards { n } => {
                let n = *n;
                for i in 0..n {
                    let st = self.s();
                    let mut options = vec![opt("Exile nothing more".to_string(), vec![s("exile_any_gy"), KI::N], Val::None)];
                    for q in [ctl, 1 - ctl] {
                        let rel = if q == ctl { "self" } else { "opponent" };
                        for c in st.dedupe_by_name(st.players[q as usize].graveyard.iter().copied()) {
                            let nm = st.c(c).name();
                            options.push(opt(format!("Exile {nm} ({rel} graveyard)"), vec![s("exile_any_gy"), s(rel), s(nm)], Val::Card(c)));
                        }
                    }
                    if options.len() == 1 {
                        break;
                    }
                    let name = item.name.clone();
                    match self.ask(ctl, Kind::ExileFromGy, || format!("{name}: exile a card from a graveyard ({}/{n})", i + 1), options)? {
                        Val::Card(c) => {
                            self.s().mv(c, Zone::Exile);
                        }
                        _ => break,
                    }
                }
            }
            Op::ExileGraveyard => {
                if let Some(Tgt::Player(p)) = self.s().target(item, 0) {
                    let st = self.s();
                    for c in st.players[p as usize].graveyard.clone() {
                        st.mv(c, Zone::Exile);
                    }
                }
            }
            Op::SearchLibrary { filter, to_battlefield, tapped, reveal, what } => {
                self.search_library(ctl, filter, *to_battlefield, what, *tapped, *reveal)?;
            }
            Op::OptionalPayment { cost, prompt, then } => {
                if self.optional_payment(ctl, cost, prompt.clone())? {
                    self.run_effect(Some(then), item)?;
                }
            }
            Op::Scry { n } => self.scry(ctl, *n as usize)?,
            Op::Surveil => self.surveil(ctl)?,
            Op::LookTop { filter, n, what } => {
                let name = item.name.clone();
                self.look_top(ctl, *n, filter, what, &name)?;
            }
            Op::Cascade => {
                let mv = self.source_card(item).face().mana_value();
                self.cascade(ctl, mv)?;
            }
            Op::Station => {
                let src = self.source_card(item);
                let st = self.s();
                if let Some(l) = st.live(&src) {
                    st.cm(l).charge += item.data.tapped_power.expect("station cost paid");
                }
            }
            Op::ReturnRandomFromGraveyard { types } => {
                let st = self.s();
                let cands: Vec<CIdx> = st.players[ctl as usize].graveyard.iter().copied().filter(|&c| st.c(c).face().types & types != 0).collect();
                if !cands.is_empty() {
                    let c = cands[st.rng.randbelow(cands.len() as u32) as usize];
                    st.push_log_lazy(|s| format!("p{ctl} returns {} at random", s.c(c).name()));
                    st.mv(c, Zone::Hand);
                }
            }
            Op::ReturnCardsFromGraveyards { types, n, each_type, any } => {
                let owners: Vec<u8> = if *any { vec![ctl, 1 - ctl] } else { vec![ctl] };
                let mut used: Vec<u16> = vec![];
                for _ in 0..*n {
                    let st = self.s();
                    let mut options = vec![opt("Return nothing more".into(), vec![s("return_gy"), KI::N], Val::None)];
                    for &q in &owners {
                        let (rel, whose) = if q == ctl { ("self", "your") } else { ("opponent", "the opponent's") };
                        for &(tb, tname) in types {
                            if *each_type && used.contains(&tb) {
                                continue;
                            }
                            let gy = st.players[q as usize].graveyard.iter().copied().filter(|&c| st.c(c).face().types & tb != 0);
                            for c in st.dedupe_by_name(gy) {
                                let nm = st.c(c).name();
                                let tl = type_lower(tname);
                                options.push(opt(format!("Return {nm} ({tl}) from {whose} graveyard"), vec![s("return_gy"), s(rel), s(tl), s(nm)], Val::Typed(tname, c)));
                            }
                        }
                    }
                    let iname = item.name.clone();
                    match self.ask(ctl, Kind::ChooseCard, || format!("{iname}: return a card from a graveyard to its owner's hand"), options)? {
                        Val::Typed(tname, c) => {
                            used.push(crate::cards::type_bit(tname).unwrap());
                            self.s().mv(c, Zone::Hand);
                        }
                        _ => break,
                    }
                }
            }
            Op::DamageChosenPower => {
                if !item.targets.is_empty() && self.s().target_legal(item, 0) {
                    let src = self.source_card(item);
                    let st = self.s();
                    let live = item.data.chosen_oid.and_then(|o| st.perm(o));
                    let amount = match live {
                        Some(c) => st.power(st.c(c)),
                        None => item.data.power.unwrap_or(0),
                    };
                    st.deal_damage(&src, item.targets[0], amount);
                }
            }
            Op::UntapTarget => {
                if let Some(Tgt::Card(c)) = self.s().target(item, 0) {
                    self.s().cm(c).tapped = false;
                }
            }
            Op::PumpTarget { n, count_elves } => {
                if let Some(Tgt::Card(c)) = self.s().target(item, 0) {
                    let st = self.s();
                    let x = if *count_elves { st.count_elves() } else { *n };
                    st.cm(c).temp.push(TempEffect { keywords: 0, power: x, toughness: x });
                }
            }
            Op::CountersTarget { n } => {
                if let Some(Tgt::Card(c)) = self.s().target(item, 0) {
                    self.s().cm(c).counters += n;
                }
            }
            Op::ShuffleTargetIntoLibrary => {
                if let Some(Tgt::Card(c)) = self.s().target(item, 0) {
                    let st = self.s();
                    let owner = st.c(c).owner as usize;
                    st.mv(c, Zone::Library);
                    st.shuffle(owner);
                }
            }
            Op::Dig { n, take, choose, rest_graveyard } => self.dig(item, *n, take.as_deref(), choose, *rest_graveyard)?,
            Op::MayExileFromGraveyard { types, type_name, then } => {
                let st = self.s();
                let cands = st.dedupe_by_name(st.players[ctl as usize].graveyard.iter().copied().filter(|&c| st.c(c).face().types & types != 0));
                if cands.is_empty() {
                    return Ok(());
                }
                let mut options = vec![opt("Exile nothing".into(), vec![s("exile_gy"), KI::N], Val::None)];
                for c in cands {
                    let nm = st.c(c).name();
                    options.push(opt(format!("Exile {nm}"), vec![s("exile_gy"), s(nm)], Val::Card(c)));
                }
                let lower = type_name.to_lowercase();
                if let Val::Card(c) = self.ask(ctl, Kind::ExileFromGy, || format!("{}: exile a {lower} card from your graveyard?", item.name), options)? {
                    self.s().mv(c, Zone::Exile);
                    self.run_effect(Some(then), item)?;
                }
            }
            Op::TakeInitiative => self.take_initiative(ctl)?,
            Op::Venture => self.venture(ctl)?,
            Op::RevealToBattlefield { n, types, type_name, counters, hexproof } => {
                let st = self.s();
                let top: Vec<CIdx> = st.players[ctl as usize].library.iter().take(*n as usize).copied().collect();
                for &c in &top {
                    st.cm(c).known_to = BOTH;
                }
                st.push_log_lazy(|s| format!("p{ctl} reveals [{}]", top.iter().map(|&c| py_repr_str(s.c(c).name())).collect::<Vec<_>>().join(", ")));
                let cands = st.dedupe_by_name(top.iter().copied().filter(|&c| st.c(c).face().types & types != 0));
                if !cands.is_empty() {
                    let options = cands
                        .iter()
                        .map(|&c| {
                            let nm = st.c(c).name();
                            opt(format!("Put {nm} onto the battlefield"), vec![s("put"), s(nm)], Val::Card(c))
                        })
                        .collect();
                    let lower = type_name.to_lowercase();
                    if let Val::Card(c) = self.ask(ctl, Kind::ChooseCard, || format!("{}: put a {lower} card onto the battlefield", item.name), options)? {
                        let st = self.s();
                        let new = st.put_onto_battlefield(c, ctl, false);
                        st.cm(new).counters += counters;
                        st.cm(new).hexproof = *hexproof;
                    }
                }
                self.s().shuffle(ctl as usize);
            }
            Op::ExploreTarget => {
                if let Some(Tgt::Card(c)) = self.s().target(item, 0) {
                    self.explore(c)?;
                }
            }
            Op::ShuffleIntoLibrary => {
                let st = self.s();
                let c = item.data.card.unwrap();
                if st.c(c).zone == Zone::Graveyard && Some(st.c(c).oid) == item.data.oid {
                    st.mv(c, Zone::Library);
                    let owner = st.c(c).owner as usize;
                    st.shuffle(owner);
                }
            }
            Op::Ward => {
                let st = self.s();
                let sid = item.data.sid.unwrap();
                if let Some(pos) = st.stack_pos(sid) {
                    let amount = item.data.amount.unwrap();
                    let (tc, tn) = (st.stack[pos].controller, st.stack[pos].name.clone());
                    let paid = self.optional_payment(tc, &ManaCost::generic(amount), format!("Ward: pay {{{amount}}} or {tn} is countered"))?;
                    if !paid {
                        self.counter(sid);
                    }
                }
            }
            Op::CountersOnTarget { n, keywords } => {
                if let Some(Tgt::Card(c)) = self.s().target(item, 0) {
                    let card = self.s().cm(c);
                    card.counters += n;
                    card.granted |= keywords;
                }
            }
            Op::AnimateTarget { power, toughness, keywords } => {
                if let Some(Tgt::Card(c)) = self.s().target(item, 0) {
                    let card = self.s().cm(c);
                    card.animated = Some((*power, *toughness));
                    card.granted |= keywords;
                }
            }
            Op::TapOrUntapTarget => {
                if let Some(Tgt::Card(c)) = self.s().target(item, 0) {
                    let r = self.s().c(c).repr();
                    let options = vec![
                        opt("Leave it".into(), vec![s("tap_or_untap"), s("neither")], Val::None),
                        opt(format!("Tap {r}"), vec![s("tap_or_untap"), s("tap")], Val::Bool(true)),
                        opt(format!("Untap {r}"), vec![s("tap_or_untap"), s("untap")], Val::Bool(false)),
                    ];
                    let name = item.name.clone();
                    if let Val::Bool(b) = self.ask(ctl, Kind::ChooseMode, || format!("{name}: tap or untap {r}?"), options)? {
                        self.s().cm(c).tapped = b;
                    }
                }
            }
            Op::ReturnFromGraveyard { types, type_name, n } => {
                for i in 0..*n {
                    let st = self.s();
                    let cands = st.dedupe_by_name(st.players[ctl as usize].graveyard.iter().copied().filter(|&c| st.c(c).face().types & types != 0));
                    if cands.is_empty() {
                        break;
                    }
                    let mut options = vec![opt("Stop".into(), vec![s("return_gy"), KI::N], Val::None)];
                    for c in cands {
                        let nm = st.c(c).name();
                        options.push(opt(format!("Return {nm}"), vec![s("return_gy"), s(nm)], Val::Card(c)));
                    }
                    let name = item.name.clone();
                    match self.ask(ctl, Kind::ChooseCard, || format!("{name}: return a {type_name} card from your graveyard to your hand ({}/{n})", i + 1), options)? {
                        Val::Card(c) => {
                            self.s().move_card(c, Zone::Hand, None, Pos::Top, Some(BOTH), false);
                        }
                        _ => break,
                    }
                }
            }
            Op::OpponentSacrifices { greatest_power_if_evidence } => {
                let st = self.s();
                let opp = 1 - ctl;
                let mut cands: Vec<CIdx> = st.battlefield.iter().copied().filter(|&c| st.c(c).controller == opp && st.is_creature(st.c(c))).collect();
                if cands.is_empty() {
                    return Ok(());
                }
                if *greatest_power_if_evidence && item.method == Method::Evidence {
                    let top = cands.iter().map(|&c| st.power(st.c(c))).max().unwrap();
                    cands.retain(|&c| st.power(st.c(c)) == top);
                }
                let options = st
                    .dedupe_by_equiv(cands)
                    .iter()
                    .map(|&c| {
                        let card = st.c(c);
                        opt(format!("Sacrifice {}#{}", card.name(), card.oid), vec![s("sacrifice"), s(card.name())], Val::Card(c))
                    })
                    .collect();
                let name = item.name.clone();
                if let Val::Card(c) = self.ask(opp, Kind::Sacrifice, || format!("{name}: sacrifice a creature"), options)? {
                    self.s().sacrifice(c);
                }
            }
            Op::Custom(f) => self.custom(*f, item)?,
        }
        Ok(())
    }

    /// cards.py `_op_dig`: top n cards, one card type to hand, the rest to
    /// the graveyard (all revealed) or the bottom (in order).
    fn dig(&mut self, item: &StackItem, n: i32, take: Option<&str>, choose: &'static [String], rest_graveyard: bool) -> R<()> {
        let p = item.controller;
        let typ: &str = match take {
            Some(t) => t,
            None => {
                let options = choose.iter().map(|t| opt(format!("Choose {}", t.to_lowercase()), vec![s("choose_type"), s(t.as_str())], Val::Name(t.as_str()))).collect();
                match self.ask(p, Kind::ChooseMode, || format!("{}: choose a card type", item.name), options)? {
                    Val::Name(t) => t,
                    _ => unreachable!(),
                }
            }
        };
        let tb = type_bit(typ).unwrap();
        let st = self.s();
        let top: Vec<CIdx> = st.players[p as usize].library.iter().take(n as usize).copied().collect();
        for &c in &top {
            if rest_graveyard {
                st.cm(c).known_to = BOTH;
            } else {
                st.cm(c).known_to |= pbit(p);
            }
        }
        let taken: Vec<CIdx> = top.iter().copied().filter(|&c| st.c(c).face().types & tb != 0).collect();
        let shown = if rest_graveyard { &top } else { &taken };
        st.push_log_lazy(|s| format!("p{p} reveals [{}]", shown.iter().map(|&c| py_repr_str(s.c(c).name())).collect::<Vec<_>>().join(", ")));
        for &c in &taken {
            st.move_card(c, Zone::Hand, None, Pos::Top, Some(BOTH), false);
        }
        for &c in &top {
            if taken.contains(&c) {
                continue;
            }
            if rest_graveyard {
                st.mv(c, Zone::Graveyard);
            } else {
                st.move_card(c, Zone::Library, None, Pos::Bottom, Some(pbit(p)), false);
            }
        }
        Ok(())
    }

    /// game.py `_madness_effect`: cast the exiled card for its madness cost,
    /// or put it into the graveyard.
    fn madness(&mut self, item: &StackItem) -> R<()> {
        let st = self.s();
        let card = item.data.card.unwrap();
        let c = st.c(card);
        if c.zone != Zone::Exile || Some(c.oid) != item.data.oid {
            return Ok(());
        }
        let p = c.owner;
        let n = c.name();
        let mut options = vec![opt(format!("Put {n} into your graveyard"), vec![s("madness"), s("graveyard")], Val::Bool(false))];
        if st.can_cast(p, card, Method::Madness, None) {
            options.push(opt(format!("Cast {n} for its madness cost"), vec![s("madness"), s("cast")], Val::Bool(true)));
        }
        let cost = c.face().madness.as_ref().unwrap().to_string();
        if let Val::Bool(true) = self.ask(p, Kind::YesNo, || format!("Madness: cast {n} for {cost}?"), options)? {
            self.cast(p, card, Method::Madness, None)
        } else {
            self.s().mv(card, Zone::Graveyard);
            Ok(())
        }
    }

    fn custom(&mut self, f: Custom, item: &StackItem) -> R<()> {
        let p = item.controller;
        match f {
            Custom::DelverReveal => {
                let st = self.s();
                let top = match st.players[p as usize].library.first() {
                    Some(&c) => c,
                    None => return Ok(()),
                };
                st.cm(top).known_to |= pbit(p);
                let n = st.c(top).name();
                let options = vec![
                    opt("Don't reveal".into(), vec![s("reveal"), s("no")], Val::Bool(false)),
                    opt(format!("Reveal {n}"), vec![s("reveal"), s("yes")], Val::Bool(true)),
                ];
                if let Val::Bool(true) = self.ask(p, Kind::YesNo, || format!("Delver of Secrets: reveal {n}?"), options)? {
                    let src = self.source_card(item);
                    let st = self.s();
                    st.cm(top).known_to = BOTH;
                    let live = st.live(&src);
                    if is_instant_or_sorcery(st.c(top).face()) {
                        if let Some(l) = live {
                            if !st.c(l).transformed {
                                st.cm(l).transformed = true;
                                st.push_log_lazy(|s| {
                                    let c = s.c(l);
                                    format!("{}#{} transforms into {}", c.defn().name, c.oid, c.name())
                                });
                            }
                        }
                    }
                }
            }
            Custom::Brainstorm => {
                self.s().draw(p as usize, 3, true);
                for i in 0..2 {
                    if self.s().players[p as usize].hand.is_empty() {
                        break;
                    }
                    let options = self.s().hand_card_options(p, "put back");
                    if let Val::Card(c) = self.ask(
                        p,
                        Kind::ChooseCard,
                        || format!("Brainstorm: put a card from your hand on top of your library ({}/2, the second one ends on top)", i + 1),
                        options,
                    )? {
                        self.s().move_card(c, Zone::Library, None, Pos::Top, Some(pbit(p)), false);
                    }
                }
            }
            Custom::Ponder => {
                let st = self.s();
                let lib = &st.players[p as usize].library;
                let top: Vec<CIdx> = lib.iter().take(3).copied().collect();
                for &c in &top {
                    st.cm(c).known_to |= pbit(p);
                }
                if top.len() > 1 {
                    let mut options = vec![];
                    let mut seen: Vec<Vec<&'static str>> = vec![];
                    for perm in permutations(&top) {
                        let names: Vec<&'static str> = perm.iter().map(|&c| st.c(c).name()).collect();
                        if seen.contains(&names) {
                            continue;
                        }
                        seen.push(names.clone());
                        let mut key = vec![s("order")];
                        key.extend(names.iter().map(|n| s(n)));
                        options.push(opt(format!("Top to bottom: {}", names.join(", ")), key, Val::Order(perm)));
                    }
                    if let Val::Order(order) = self.ask(p, Kind::Order, || "Ponder: put the cards back in any order".to_string(), options)? {
                        let lib = &mut self.s().players[p as usize].library;
                        lib[..order.len()].copy_from_slice(&order);
                    }
                }
                let options = vec![
                    opt("Don't shuffle".into(), vec![s("shuffle"), s("no")], Val::Bool(false)),
                    opt("Shuffle".into(), vec![s("shuffle"), s("yes")], Val::Bool(true)),
                ];
                if let Val::Bool(true) = self.ask(p, Kind::YesNo, || "Ponder: shuffle your library?".to_string(), options)? {
                    self.s().shuffle(p as usize);
                }
                self.s().draw(p as usize, 1, true);
            }
            Custom::DeemInferior => {
                if let Some(Tgt::Card(t)) = self.s().target(item, 0) {
                    let st = self.s();
                    let owner = st.c(t).owner;
                    let n = st.c(t).name();
                    let mut options = vec![opt(format!("Put {n} on the bottom"), vec![s("deem"), s("bottom")], Val::Bottom)];
                    if st.players[owner as usize].library.len() >= 2 {
                        options.insert(0, opt(format!("Put {n} second from the top"), vec![s("deem"), s("second")], Val::Int(1)));
                    }
                    let pos = match self.ask(owner, Kind::ChooseMode, || format!("Deem Inferior: where does {n} go?"), options)? {
                        Val::Int(i) => Pos::At(i as usize),
                        _ => Pos::Bottom,
                    };
                    self.s().move_card(t, Zone::Library, None, pos, Some(BOTH), false);
                }
            }
            Custom::OpponentDiscardsElseDraw => {
                let opp = 1 - p;
                if self.s().players[opp as usize].hand.is_empty() {
                    self.s().draw(p as usize, 1, true);
                    return Ok(());
                }
                let src_name = self.source_card(item).name();
                let options = self.s().hand_card_options(opp, "discard");
                if let Val::Card(c) = self.ask(opp, Kind::ChooseCard, || format!("{src_name}: discard a card"), options)? {
                    self.s().discard(c);
                }
            }
            Custom::Wildfire => {
                if let Some(Tgt::Card(t)) = self.s().target(item, 0) {
                    let controller = self.s().c(t).controller;
                    self.s().destroy(t);
                    let basic = SearchFilter { supertype: Some("Basic".into()), types: T_LAND, subtypes_any: vec![], colorless: false };
                    self.search_library(controller, &basic, true, "a basic land card", true, false)?;
                }
            }
            Custom::Duress => {
                if let Some(Tgt::Player(v)) = self.s().target(item, 0) {
                    let st = self.s();
                    let hand = st.players[v as usize].hand.clone();
                    for &c in &hand {
                        st.cm(c).known_to = BOTH;
                    }
                    st.push_log_lazy(|s| format!("p{v} reveals [{}]", hand.iter().map(|&c| py_repr_str(s.c(c).name())).collect::<Vec<_>>().join(", ")));
                    let cands = st.dedupe_by_name(hand.iter().copied().filter(|&c| {
                        let f = st.c(c).face();
                        !f.is_type(T_CREATURE) && !f.is_type(T_LAND)
                    }));
                    if cands.is_empty() {
                        return Ok(());
                    }
                    let options = cands
                        .iter()
                        .map(|&c| {
                            let n = st.c(c).name();
                            opt(format!("Discard {n}"), vec![s("duress"), s(n)], Val::Card(c))
                        })
                        .collect();
                    if let Val::Card(c) = self.ask(p, Kind::ChooseCard, || "Duress: choose a card to discard".to_string(), options)? {
                        self.s().discard(c);
                    }
                }
            }
            Custom::HighwayRobbery => {
                let st = self.s();
                let mut options = vec![opt("Neither: draw nothing".into(), vec![s("robbery"), s("none")], Val::None)];
                for c in st.dedupe_by_name(st.players[p as usize].hand.iter().copied()) {
                    let n = st.c(c).name();
                    options.push(opt(format!("Discard {n}"), vec![s("robbery"), s("discard"), s(n)], Val::Robbery(false, c)));
                }
                let lands: Vec<CIdx> = st.battlefield.iter().copied().filter(|&c| st.c(c).controller == p && st.is_land(st.c(c))).collect();
                for c in st.dedupe_by_equiv(lands) {
                    let card = st.c(c);
                    options.push(opt(format!("Sacrifice {}#{}", card.name(), card.oid), vec![s("robbery"), s("sacrifice"), s(card.name())], Val::Robbery(true, c)));
                }
                match self.ask(p, Kind::ChooseCard, || "Highway Robbery: discard a card or sacrifice a land to draw two?".to_string(), options)? {
                    Val::Robbery(sac, c) => {
                        let st = self.s();
                        if sac {
                            st.sacrifice(c);
                        } else {
                            st.discard(c);
                        }
                        st.draw(p as usize, 2, true);
                    }
                    _ => return Ok(()),
                }
            }
            Custom::RelicExileOne => {
                if let Some(Tgt::Player(t)) = self.s().target(item, 0) {
                    let st = self.s();
                    if st.players[t as usize].graveyard.is_empty() {
                        return Ok(());
                    }
                    let options = st
                        .dedupe_by_name(st.players[t as usize].graveyard.iter().copied())
                        .into_iter()
                        .map(|c| {
                            let nm = st.c(c).name();
                            opt(format!("Exile {nm}"), vec![s("exile_gy"), s(nm)], Val::Card(c))
                        })
                        .collect();
                    let name = item.name.clone();
                    if let Val::Card(c) = self.ask(t, Kind::ExileFromGy, || format!("{name}: exile a card from your graveyard"), options)? {
                        self.s().mv(c, Zone::Exile);
                    }
                }
            }
        }
        Ok(())
    }
}

pub fn search_matches(f: &SearchFilter, c: &Card) -> bool {
    let face = c.face();
    if let Some(sup) = &f.supertype {
        if !face.has_supertype(sup) {
            return false;
        }
    }
    if f.types != 0 && face.types & f.types == 0 {
        return false;
    }
    if !f.subtypes_any.is_empty() && !f.subtypes_any.iter().any(|s| face.has_subtype(s)) {
        return false;
    }
    if f.colorless && face.colors != 0 {
        return false;
    }
    true
}

/// itertools.permutations order (lexicographic in positions).
fn permutations(items: &[CIdx]) -> Vec<Vec<CIdx>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = vec![];
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let x = rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, x);
            out.push(p);
        }
    }
    out
}
