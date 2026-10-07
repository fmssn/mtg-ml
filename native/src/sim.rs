//! Simulated option previews (feature set 6), bit-identical to encode.py
//! `sim_previews` / `_simulate`: each option of a decision is applied to a
//! copy of the game, advanced while that needs no hidden information and no
//! choice of the opponent, and the change the decider can observe is
//! featurized as `pv:sim:*` tokens. Reference: docs/features.md.

use std::fmt::{Arguments, Write};

use crate::features::{mana_colors, PREVIEW_COLORS};
use crate::game::{Game, StepError};
use crate::mana::bit;
use crate::state::*;

/// encode.py `SIM_MAX_OPTIONS` / `SIM_MAX_STEPS` / `SIM_TIER_CAP`.
pub const SIM_MAX_OPTIONS: usize = 32;
pub const SIM_MAX_STEPS: usize = 64;
const SIM_TIER_CAP: i64 = 5;
const MANA_LEFT_CAP: i64 = 8;
const COUNT_CAP: usize = 8;
const LIFE_STEPS: &[i64] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25];
const POWER_STEPS: &[i64] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20];
const COUNT_STEPS: &[i64] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

/// encode.py `SIM_FLAGS`.
const SIM_FLAGS: [&str; 6] = [
    "self:lethal_on_board",
    "self:evasive_lethal_on_board",
    "opponent:lethal_on_board",
    "opponent:evasive_lethal_on_board",
    "self:incoming_lethal",
    "opponent:incoming_lethal",
];

/// encode.py `SIM_FORCED_KINDS`: opponent decisions a simulation takes when
/// they have exactly one option (their options are public).
fn forced_kind(k: Kind) -> bool {
    matches!(
        k,
        Kind::Target | Kind::PayMana | Kind::Sacrifice | Kind::ExileFromGy | Kind::OrderTriggers | Kind::DeclareAttacker | Kind::DeclareBlocker | Kind::AssignDamage | Kind::ChooseX
    )
}

/// One permanent: (oid, the viewer's, creature, power, tapped, damage).
type Perm = (u32, bool, bool, i64, bool, i64);

/// encode.py `_sim_summary`.
pub struct Summary {
    perms: Vec<Perm>,
    life: [i64; 2],
    zones: [[i64; 4]; 2],
    stack: i64,
    flags: u8,
    mana: i64,
    colors: u8,
    turn: i32,
    step: &'static str,
}

/// The `SIM_FLAGS` set by `features.rs::board_features` / `combat_features`
/// (encode.py reads them from those strings), computed without formatting
/// their thermometers: the same sums, compared the same way.
fn flags(st: &State, v: u8) -> u8 {
    let d = crate::cards::db();
    let (fly, reach) = (d.kw("flying"), d.kw("reach"));
    let mut out = 0u8;
    for (k, p) in [v, 1 - v].into_iter().enumerate() {
        let active = st.active == p;
        let (mut ready, mut evasive) = (0i64, 0i64);
        for &ci in &st.battlefield {
            let c = st.c(ci);
            if c.controller != p || !st.is_creature(c) {
                continue;
            }
            let can_attack = if active { !c.tapped && !c.sick } else { !(c.tapped && c.skip_untap > 0) };
            if !can_attack {
                continue;
            }
            let pw = st.power(c).max(0) as i64;
            ready += pw;
            let kw = st.keywords(c);
            let blockable = st.battlefield.iter().map(|&b| st.c(b)).any(|b| b.controller != p && !b.tapped && st.is_creature(b) && (kw & fly == 0 || st.keywords(b) & (fly | reach) != 0));
            if !blockable {
                evasive += pw;
            }
        }
        let life = st.players[(1 - p) as usize].life as i64;
        if ready > 0 && ready >= life {
            out |= 1 << (2 * k);
        }
        if evasive > 0 && evasive >= life {
            out |= 1 << (2 * k + 1);
        }
    }
    let attackers: Vec<&Card> = st.attackers.iter().filter_map(|&a| st.perm(a)).map(|ci| st.c(ci)).collect();
    if !attackers.is_empty() {
        let unblocked: i64 = attackers.iter().filter(|c| !st.blocked.contains(&c.oid)).map(|c| st.power(c).max(0) as i64).sum();
        let life = st.players[(1 - st.active) as usize].life as i64;
        if unblocked > 0 && unblocked >= life {
            out |= 1 << if st.active == v { 4 } else { 5 };
        }
    }
    out
}

pub fn summary(st: &State, v: u8) -> Summary {
    let perms = st
        .battlefield
        .iter()
        .map(|&ci| {
            let c = st.c(ci);
            let cr = st.is_creature(c);
            (c.oid, c.controller == v, cr, if cr { st.power(c).max(0) as i64 } else { 0 }, c.tapped, c.damage as i64)
        })
        .collect();
    let sides = [&st.players[v as usize], &st.players[1 - v as usize]];
    let fl = flags(st, v);
    let sources = st.mana_sources(v, &[]);
    let mut colors = 0u8;
    for &(ci, _) in &sources {
        colors |= mana_colors(st.c(ci));
    }
    let pool = &st.players[v as usize].pool;
    for &(col, n) in pool {
        if n > 0 && PREVIEW_COLORS.contains(&col) {
            colors |= bit(col);
        }
    }
    let zones = |p: &Player| [p.hand.len() as i64, p.graveyard.len() as i64, p.exile.len() as i64, p.library.len() as i64];
    Summary {
        perms,
        life: [sides[0].life as i64, sides[1].life as i64],
        zones: [zones(sides[0]), zones(sides[1])],
        stack: st.stack.len() as i64,
        flags: fl,
        mana: sources.len() as i64 + pool.iter().map(|e| e.1 as i64).sum::<i64>(),
        colors,
        turn: st.turn,
        step: st.step_name,
    }
}

/// encode.py `_hidden_touched`: a library shuffled, a card leaving,
/// entering or moving in a library, or a library card `viewer` did not know
/// looked at.
fn hidden_touched(a: &State, b: &State, viewer: u8) -> bool {
    let me = pbit(viewer);
    if a.shuffles != b.shuffles {
        return true;
    }
    for (pa, pb) in a.players.iter().zip(b.players.iter()) {
        if pa.library.len() != pb.library.len() {
            return true;
        }
        for (&x, &y) in pa.library.iter().zip(pb.library.iter()) {
            let (x, y) = (a.c(x), b.c(y));
            if x.oid != y.oid || (x.known_to & me) != (y.known_to & me) {
                return true;
            }
        }
    }
    false
}

fn thermo(out: &mut impl FnMut(Arguments), name: std::fmt::Arguments, n: i64, steps: &[i64]) {
    if n < steps[0] {
        return;
    }
    let mut b = String::with_capacity(48);
    let _ = b.write_fmt(name);
    for &k in steps {
        if n >= k {
            out(format_args!("{b}>={k}"));
        }
    }
}

/// encode.py `_signed`.
fn signed(out: &mut impl FnMut(Arguments), name: std::fmt::Arguments, d: i64, steps: &[i64]) {
    if d == 0 {
        return;
    }
    let mut b = String::with_capacity(48);
    let _ = b.write_fmt(name);
    let (sign, n) = if d > 0 { ('+', d) } else { ('-', -d) };
    for &k in steps {
        if n >= k {
            out(format_args!("{b}{sign}>={k}"));
        }
    }
}

/// Whether a simulation can start: the game has a step-start snapshot to copy
/// from (the Python wrapper makes sure a game that was never copied has one).
pub fn ready(g: &Game) -> bool {
    g.started() && g.state().snap.is_some()
}

/// The result of one simulation (encode.py `_run`).
enum Outcome {
    Hidden,
    Done { stop: &'static str, over: bool, winner: Option<u8>, next: Option<(bool, Kind)>, after: Summary },
}

/// encode.py `_run`: step a copy with option `i`, take the opponent's forced
/// decisions and, with `assume_pass`, pass for it at every priority; None
/// when the game cannot be copied.
fn run(g: &mut Game, player: u8, i: usize, assume_pass: bool) -> Result<Option<Outcome>, StepError> {
    let mut g2 = match g.copy()? {
        Some(x) => x,
        None => return Ok(None),
    };
    g2.state_mut().sim_viewer = Some(player);
    g2.state_mut().sim_assume_pass = assume_pass;
    g2.step(i)?;
    let mut steps = 1;
    let stop = loop {
        if hidden_touched(g.state(), g2.state(), player) {
            return Ok(Some(Outcome::Hidden));
        }
        let st = g2.state();
        if st.over {
            break "game_over";
        }
        let d = st.decision.as_ref().expect("a decision");
        if d.player == player {
            break "own_decision";
        }
        if !forced_kind(d.kind) || d.options.len() != 1 {
            break "opponent_decision";
        }
        if steps >= SIM_MAX_STEPS {
            break "step_cap";
        }
        g2.step(0)?;
        steps += 1;
    };
    let st = g2.state();
    let next = st.decision.as_ref().map(|d| (d.player == player, d.kind));
    Ok(Some(Outcome::Done { stop, over: st.over, winner: st.winner, next, after: summary(st, player) }))
}

/// encode.py `_simulate`: `pv:sim:` (the opponent may respond: its priority
/// stops the simulation), then `pv:simp:` (it passes). The second simulation
/// runs only when the first stopped at the opponent's priority; otherwise
/// the first one's result is repeated under `pv:simp:`.
pub fn simulate(g: &mut Game, player: u8, i: usize, before: &Summary, out: &mut impl FnMut(Arguments)) -> Result<(), StepError> {
    let o1 = match run(g, player, i, false)? {
        Some(o) => o,
        None => {
            out(format_args!("pv:sim:skipped"));
            out(format_args!("pv:simp:skipped"));
            return Ok(());
        }
    };
    delta("pv:sim", &o1, before, player, out);
    if matches!(o1, Outcome::Done { stop: "opponent_decision", next: Some((false, Kind::Priority)), .. }) {
        match run(g, player, i, true)? {
            Some(o2) => delta("pv:simp", &o2, before, player, out),
            None => out(format_args!("pv:simp:skipped")),
        }
    } else {
        delta("pv:simp", &o1, before, player, out);
    }
    Ok(())
}

/// encode.py `_delta`: one simulation's tokens under prefix `pre`.
fn delta(pre: &str, o: &Outcome, before: &Summary, player: u8, out: &mut impl FnMut(Arguments)) {
    let (stop, over, winner, next, after) = match o {
        Outcome::Hidden => {
            out(format_args!("{pre}:stop:hidden_info"));
            return;
        }
        Outcome::Done { stop, over, winner, next, after } => (*stop, *over, *winner, *next, after),
    };
    out(format_args!("{pre}:stop:{stop}"));
    if over {
        match winner {
            None => out(format_args!("{pre}:draw_game")),
            Some(w) if w == player => out(format_args!("{pre}:won")),
            Some(_) => out(format_args!("{pre}:lost")),
        }
    } else if let Some((mine, kind)) = next {
        out(format_args!("{pre}:next:{}:{}", if mine { "self" } else { "opponent" }, kind.name()));
    }
    if after.turn != before.turn {
        out(format_args!("{pre}:new_turn"));
    }
    if after.turn != before.turn || after.step != before.step {
        out(format_args!("{pre}:step:{}", after.step));
    }
    let was = |oid: u32| before.perms.iter().find(|p| p.0 == oid);
    let is_now = |oid: u32| after.perms.iter().any(|p| p.0 == oid);
    for (k, side) in ["self", "opponent"].into_iter().enumerate() {
        let mine = k == 0;
        signed(out, format_args!("{pre}:{side}:life"), after.life[k] - before.life[k], LIFE_STEPS);
        let lost: Vec<&Perm> = before.perms.iter().filter(|p| p.1 == mine && !is_now(p.0)).collect();
        let gained: Vec<&Perm> = after.perms.iter().filter(|p| p.1 == mine && was(p.0).is_none()).collect();
        let kept: Vec<(&Perm, &Perm)> = after.perms.iter().filter(|p| p.1 == mine).filter_map(|p| was(p.0).map(|a| (a, p))).collect();
        let lost_cr: Vec<&&Perm> = lost.iter().filter(|p| p.2).collect();
        let gained_cr: Vec<&&Perm> = gained.iter().filter(|p| p.2).collect();
        thermo(out, format_args!("{pre}:{side}:creatures_lost"), lost_cr.len() as i64, COUNT_STEPS);
        thermo(out, format_args!("{pre}:{side}:power_lost"), lost_cr.iter().map(|p| p.3).sum(), POWER_STEPS);
        // encode.py `_counted` over the tiers, in first-occurrence order
        let mut tiers: Vec<(i64, usize)> = vec![];
        for p in &lost_cr {
            let t = p.3.min(SIM_TIER_CAP);
            match tiers.iter_mut().find(|e| e.0 == t) {
                Some(e) => e.1 += 1,
                None => tiers.push((t, 1)),
            }
        }
        for (t, n) in tiers {
            out(format_args!("{pre}:{side}:lost_power_tier:{t}"));
            for c in 2..=n.min(COUNT_CAP) {
                out(format_args!("{pre}:{side}:lost_power_tier:{t}#{c}"));
            }
        }
        thermo(out, format_args!("{pre}:{side}:creatures_gained"), gained_cr.len() as i64, COUNT_STEPS);
        thermo(out, format_args!("{pre}:{side}:power_gained"), gained_cr.iter().map(|p| p.3).sum(), POWER_STEPS);
        thermo(out, format_args!("{pre}:{side}:perms_lost"), lost.len() as i64, COUNT_STEPS);
        thermo(out, format_args!("{pre}:{side}:perms_gained"), gained.len() as i64, COUNT_STEPS);
        thermo(out, format_args!("{pre}:{side}:tapped"), kept.iter().filter(|(a, b)| b.4 && !a.4).count() as i64, COUNT_STEPS);
        thermo(out, format_args!("{pre}:{side}:untapped"), kept.iter().filter(|(a, b)| a.4 && !b.4).count() as i64, COUNT_STEPS);
        thermo(out, format_args!("{pre}:{side}:damage"), kept.iter().map(|(a, b)| (b.5 - a.5).max(0)).sum(), POWER_STEPS);
        for (z, zone) in ["hand", "graveyard", "exile", "library"].into_iter().enumerate() {
            signed(out, format_args!("{pre}:{side}:{zone}"), after.zones[k][z] - before.zones[k][z], COUNT_STEPS);
        }
    }
    signed(out, format_args!("{pre}:stack"), after.stack - before.stack, COUNT_STEPS);
    thermo(out, format_args!("{pre}:mana_left"), after.mana.min(MANA_LEFT_CAP), COUNT_STEPS);
    for &col in &PREVIEW_COLORS {
        if after.colors & bit(col) != 0 {
            out(format_args!("{pre}:color:{}", col as char));
        }
    }
    for (j, flag) in SIM_FLAGS.iter().enumerate() {
        let (a, b) = (after.flags & (1 << j) != 0, before.flags & (1 << j) != 0);
        if a {
            out(format_args!("{pre}:{flag}"));
            if !b {
                out(format_args!("{pre}:gained:{flag}"));
            }
        } else if b {
            out(format_args!("{pre}:lost:{flag}"));
        }
    }
}

/// encode.py `sim_previews`: `emit(option index, token)` for every option of
/// the current decision of `player`.
pub fn sim_previews(g: &mut Game, player: u8, mut emit: impl FnMut(usize, Arguments)) -> Result<(), StepError> {
    let n = g.state().decision.as_ref().map_or(0, |d| d.options.len());
    if n > SIM_MAX_OPTIONS || !ready(g) {
        for i in 0..n {
            emit(i, format_args!("pv:sim:skipped"));
            emit(i, format_args!("pv:simp:skipped"));
        }
        return Ok(());
    }
    let before = summary(g.state(), player);
    for i in 0..n {
        simulate(g, player, i, &before, &mut |a| emit(i, a))?;
    }
    Ok(())
}

/// encode.py `sim_preview`: the tokens of option `i` alone.
pub fn sim_preview(g: &mut Game, player: u8, i: usize, out: &mut impl FnMut(Arguments)) -> Result<(), StepError> {
    let n = g.state().decision.as_ref().map_or(0, |d| d.options.len());
    if n > SIM_MAX_OPTIONS || !ready(g) {
        out(format_args!("pv:sim:skipped"));
        out(format_args!("pv:simp:skipped"));
        return Ok(());
    }
    let before = summary(g.state(), player);
    simulate(g, player, i, &before, out)
}
