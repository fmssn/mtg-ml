// Play vs model: a board client for mtg_ml.live (design: docs/play-ui.md).
//
// The server holds the game and sends the player's view: frames in the replay
// format, each with a decision whose options carry structured `refs` (what
// card an option casts, which permanent it activates, which creature a
// blocker blocks) and parsed `actions` (the visible log, for animation).
// This page maps those options onto gestures (drag to play, click to target,
// batch attacks and blocks) and answers the engine's one-at-a-time
// decisions in order. It never knows the rules: anything it cannot map stays
// reachable through the "All options" drawer.
'use strict';

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const sleep = ms => new Promise(r => setTimeout(r, ms));
const API = new URL('../api/live/', location.href).href;

const STEPS = ['upkeep', 'draw', 'main1', 'begin_combat', 'declare_attackers', 'declare_blockers', 'combat_damage', 'end_combat', 'main2', 'end'];
const STEP_LABEL = {untap: 'Untap', upkeep: 'Upkeep', draw: 'Draw', main1: 'Main 1', begin_combat: 'Combat', declare_attackers: 'Attack', declare_blockers: 'Block', combat_damage: 'Damage', end_combat: 'End combat', main2: 'Main 2', end: 'End', cleanup: 'Cleanup'};
const NEXT = {untap: 'Upkeep', upkeep: 'Draw', draw: 'Main 1', main1: 'Combat', begin_combat: 'Attacks', declare_attackers: 'Blocks', declare_blockers: 'Damage', combat_damage: 'End of combat', end_combat: 'Main 2', main2: 'End step', end: 'Next turn', cleanup: 'Next turn'};
const SPEEDS = {'0.5': 2, '1': 1, '2': 0.5, instant: 0};

// ---------------------------------------------------------------- preferences
const PREF = Object.assign({
  speed: '1', autoPayNoAsk: false, spotMs: 2800, confirmEmptyAttack: true, sound: true, autoTarget: false,
  stops: {me: {main1: true, main2: true}, opp: {declare_blockers: true, end: true}}, stopsVersion: 2,
}, loadJSON('mtgml-play-pref', {}));
function loadJSON(k, d) { try { return JSON.parse(localStorage.getItem(k) || 'null') ?? d; } catch (e) { return d; } }
function saveJSON(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private window */ } }
const savePref = () => saveJSON('mtgml-play-pref', PREF);
if ((PREF.stopsVersion || 1) < 2) { PREF.stops.opp.declare_blockers = true; PREF.stopsVersion = 2; savePref(); }  // new default: the after-blockers window

// ---------------------------------------------------------------- game state
const S = {
  gid: null, seat: 0, raw: [], cards: {}, meta: null, over: false, replay: null,
  shown: -1, prev: null, stopLog: [], lastAutoPassed: 0, yields: new Set(), holdOnce: false, autoPassed: 0, stats: {dealt: 0, taken: 0, played: 0}, lastCast: {}, seenUids: new Set(), pendingSpells: [], fx: {life: {}, dmg: {}, flash: {}, enter: {}}, queue: [], pumping: false, busy: false, error: null, skip: false,
  plan: null, ui: null, lastMine: -1, passMode: null, fullControl: false, feed: [], logDone: 0,
};
const opp = () => 1 - S.seat;
const last = () => S.raw.length - 1;
const stateAt = i => S.raw[i].state;
const myDecision = () => { const f = S.raw[last()]; return f && f.decision && f.decision.player === S.seat && f.decision.chosen == null ? f.decision : null; };

// ---------------------------------------------------------------- card images (Scryfall; cache shared with the replay viewer)
const IMG_KEY = 'mtgml-scryfall-v1';
let IMG = loadJSON(IMG_KEY, {});
function addSf(c, alias) {
  const url = x => x && (x.normal || x.large);
  if (c.image_uris) { IMG[c.name] = url(c.image_uris); if (alias) IMG[alias] = url(c.image_uris); }
  for (const f of c.card_faces || []) if (f.image_uris) IMG[f.name] = url(f.image_uris);
  if (alias && !IMG[alias] && c.card_faces?.[0]?.image_uris) IMG[alias] = url(c.card_faces[0].image_uris);
}
let imgBusy = null;
// Wait (at most `ms`) until a card's image is known and loaded, so the spotlight shows art, not a text face.
let imgBackoffUntil = 0;
async function ensureImg(name, ms) {
  if (performance.now() < imgBackoffUntil) return;  // offline: text faces, no waiting
  const end = performance.now() + ms;
  if (!(name in IMG)) resolveImages();
  while (!(name in IMG) && performance.now() < end) await sleep(40);
  if (!IMG[name]) return;
  const im = new Image();
  im.src = IMG[name];
  await Promise.race([im.decode().catch(() => {}), sleep(Math.max(0, end - performance.now()))]);
}
function resolveImages() {
  if (imgBusy) return imgBusy;
  imgBusy = fetchImages().finally(() => { imgBusy = null; });
  return imgBusy;
}
async function fetchImages() {
  const want = Object.keys(S.cards).filter(n => !(n in IMG));
  if (!want.length || performance.now() < imgBackoffUntil) return;
  try {
    const normal = want.filter(n => !S.cards[n].token), tokens = want.filter(n => S.cards[n].token);
    for (let i = 0; i < normal.length; i += 75) {
      const r = await fetch('https://api.scryfall.com/cards/collection', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({identifiers: normal.slice(i, i + 75).map(name => ({name}))})});
      for (const c of (await r.json()).data || []) addSf(c);
    }
    for (const n of tokens) {
      const r = await fetch(`https://api.scryfall.com/cards/search?include_extras=true&q=${encodeURIComponent(`!"${n}" t:token`)}`);
      if (r.ok) { const j = await r.json(); if (j.data?.length) addSf(j.data[0], n); }
      await sleep(110);
    }
    for (const n of want) if (!(n in IMG)) {
      const r = await fetch(`https://api.scryfall.com/cards/named?exact=${encodeURIComponent(n)}`);
      if (r.ok) addSf(await r.json(), n); else if (r.status === 404) IMG[n] = null;
      await sleep(110);
    }
    saveJSON(IMG_KEY, IMG);
  } catch (e) { console.warn('scryfall', e); imgBackoffUntil = performance.now() + 120000; }  // retry in two minutes
  refreshImages();
}
// Swap text faces for images in place: a re-render now would break a drag or a double-click.
function refreshImages() {
  for (const el of $$('.card[data-name]')) {
    const url = IMG[el.dataset.name], tf = el.querySelector(':scope > .tface');
    if (url && tf) tf.outerHTML = `<img src="${esc(url)}" alt="${esc(el.dataset.name)}" draggable="false">`;
  }
}

function colorOf(name) {
  const info = S.cards[name]; if (!info) return '';
  const cs = new Set(info.cost.match(/[WUBRG]/g) || []);
  if (cs.size > 1) return 'cM';
  return cs.size ? 'c' + [...cs][0] : '';
}
function textFace(name) {
  const i = S.cards[name] || {types: [], subtypes: [], cost: '', text: ''};
  const tl = i.types.join(' ') + (i.subtypes.length ? ' — ' + i.subtypes.join(' ') : '');
  return `<div class="tface ${colorOf(name)}"><div class="tn"><span>${esc(name)}</span><span>${esc(i.cost.replace(/[{}]/g, ''))}</span></div>
    <div class="tt2">${esc((i.token ? 'Token ' : '') + tl)}</div><div class="tx">${esc(i.text || '')}</div>${i.power != null ? `<div class="tpt">${i.power}/${i.toughness}</div>` : ''}</div>`;
}
function cardHtml(name, o = {}) {
  const url = IMG[name];
  const face = url ? `<img src="${esc(url)}" alt="${esc(name)}" draggable="false">` : textFace(name);
  return `<div class="card ${o.cls || ''}" data-name="${esc(name)}"${o.uid != null ? ` data-uid="${o.uid}"` : ''}${o.attrs || ''}>${face}${o.badges || ''}</div>`;
}
const backHtml = (attrs = '') => `<div class="card back"${attrs}></div>`;
const clean = label => String(label).replace(/#\d+/g, '');

// ---------------------------------------------------------------- server
// The game's token (kept per game id in this browser, never shown or sent
// anywhere else) goes with every request about the game, in the
// X-Game-Token header only (never in a URL): the server refuses a game id
// without it. On the hosted server the account (Cloudflare Access) must own
// the game too.
const gameToken = gid => loadJSON('mtgml-play-tokens', {})[gid] || '';
async function api(path, body) {
  const headers = {};
  if (S.gid) headers['X-Game-Token'] = gameToken(S.gid);
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  // redirect 'manual': an expired Access session answers with a redirect to
  // the login page, which a fetch cannot follow; it shows up as opaqueredirect.
  const init = {headers, redirect: 'manual', credentials: 'same-origin'};
  const r = await fetch(API + path, body === undefined ? init : {...init, method: 'POST', body: JSON.stringify(body)});
  if (r.type === 'opaqueredirect' || r.status === 0) { signedOut(); throw apiError('Your sign-in expired. Reload the page to sign in again.', 401, {auth: true}); }
  let j; try { j = await r.json(); } catch (e) { throw apiError(`server answered ${r.status}`, r.status, {}); }
  if (!r.ok || j.error) {
    const err = apiError(j.error || `server answered ${r.status}`, r.status, j);
    if (err.kind === 'auth') signedOut();
    throw err;
  }
  return j;
}
// An error with what the page needs to react: kind 'auth' (Access session
// gone: reload), 'unrecoverable' (the server cannot rebuild this game),
// 'gone' (unknown or expired game).
function apiError(message, status, j) {
  const e = new Error(message);
  e.status = status;
  e.kind = j.auth ? 'auth' : j.unrecoverable ? 'unrecoverable' : (j.expired || status === 404 || /no such game/i.test(message)) ? 'gone' : '';
  return e;
}
function signedOut() {
  if ($('#signedout')) return;
  const d = document.createElement('div');
  d.id = 'signedout';
  d.className = 'signedout';
  d.setAttribute('role', 'alert');
  d.innerHTML = `<span>Your sign-in expired. Your game is saved on the server.</span><button class="primary" type="button">Reload and sign in</button>`;
  d.querySelector('button').onclick = () => location.reload();
  document.body.appendChild(d);
}

function applyView(view) {
  const L = view.live;
  if (S.gid !== L.id) resetGame(L.id);
  if (L.token) { const t = loadJSON('mtgml-play-tokens', {}); t[L.id] = L.token; for (const k of Object.keys(t).slice(0, -20)) delete t[k]; saveJSON('mtgml-play-tokens', t); }
  Object.assign(S, {seat: L.seat, over: L.over, replay: L.replay, meta: view.meta, matchup: L.matchup || S.matchup});
  Object.assign(S.cards, view.cards);
  const before = S.raw.length;
  S.raw.splice(L.since, Infinity, ...view.frames);
  for (let i = Math.max(before, S.shown + 1); i < S.raw.length; i++) S.queue.push(i);
  try { history.replaceState(null, '', `#g=${encodeURIComponent(S.gid)}`); } catch (e) { /* file: */ }
  renderMeta();
  resolveImages();
}

function resetGame(id) {
  // Every auto-pass mode and plan resets at a game boundary (Forge bug: End Turn carried over).
  Object.assign(S, {gid: id, flags: [], stopLog: [], lastAutoPassed: 0, yields: new Set(), holdOnce: false, autoPassed: 0, stats: {dealt: 0, taken: 0, played: 0}, lastPriority: -1, lastCast: {}, seenUids: new Set(), pendingSpells: [], raw: [], shown: -1, prev: null, fx: {life: {}, dmg: {}, flash: {}, enter: {}}, queue: [], plan: null, ui: null, lastMine: -1, passMode: null, lastOpp: '',
    fullControl: false, feed: [], logDone: 0, error: null, errorKind: '', skip: false, reserved: [], payPlan: null, payAssign: null, paying: null, autoRest: false, justTapped: null, spotDone: new Set()});
  $('#log').innerHTML = '';
  renderFeed();
}

async function send(index) {
  const frame = last();
  S.busy = true; S.error = null; S.errorKind = ''; S.ui = null; closePop(); closeOverlay(); renderDock();
  if (S.shown >= 0) render(S.shown, {noAnim: true});  // drop the highlights of the decision just made
  try {
    const kind = S.raw[frame].decision.kind;
    const v = await api(`${encodeURIComponent(S.gid)}/choose`, {frame, index, since: frame});
    S.lastMine = frame;
    if (kind === 'priority') S.lastPriority = frame;
    applyView(v);
  } catch (e) {
    S.error = e.message; S.errorKind = e.kind || '';
  } finally { S.busy = false; }
}

// Pick an option (from a gesture) and run the game on to the next prompt.
function stamp(plan) {  // scope a plan to the current turn and step
  const s = stateAt(last());
  return plan && {...plan, turn: s.turn, step: s.step};
}
async function act(index, plan, frame, pay) {
  if (S.busy || S.pumping || !myDecision()) return;
  if (frame != null && frame !== last()) return;  // made for a decision that is gone
  if (plan !== undefined) S.plan = stamp(plan);
  if (pay) { S.payAssign = pay.assign; S.payPlan = pay.rest; }
  else if (myDecision().kind === 'priority') { S.payAssign = null; S.payPlan = null; }
  await send(index);
  pump();
}

// Play queued frames, then answer decisions that need no human (auto-pass,
// auto-pay, the rest of a batched attack or block) until one does.
async function pump() {
  if (S.pumping) return;
  S.pumping = true;
  let autos = 0;
  try {
    for (;;) {
      while (S.queue.length) await playFrame(S.queue.shift());
      if (S.error) { enterDecision(); break; }
      const d = myDecision();
      if (S.over || !d) { finishGame(); break; }
      expireReserved();
      let a = null;
      try { a = autoAnswer(d); } catch (e) { console.error(e); a = null; }
      if (a == null || autos > 400) { S.plan = null; S.skip = false; S.passMode = null; enterDecision(true); break; }
      autos++;
      if (myDecision()?.kind === 'priority' && d.refs[a]?.type === 'pass') S.autoPassed++;
      await send(a);
    }
  } finally { S.pumping = false; renderDock(); }
}

// ---------------------------------------------------------------- automation (P2, P6, P7 of the brief)
const isReal = r => r.type !== 'pass' && r.type !== 'mana';
// Abilities with no timing value (draw, search, cycling, scry, life, tokens):
// they never make the opponent's actions a reason to stop. Casts and other
// abilities (pump, untap, damage, counters...) do.
const QUIET_ABILITY = /draw a card|search|cycling|scry|surveil|investigate|gain \d+ life|create .*token|shuffle|look at/i;
const sourceOf = r => r.name || '';
function timely(r, label) {
  if (!isReal(r) || S.yields.has(sourceOf(r))) return false;
  if (r.type === 'activate') return !QUIET_ABILITY.test(r.ability || label);
  return true;
}
const ACTS = new Set(['cast', 'activate', 'plot', 'attack', 'block']);

// Did the opponent act in the current step since my last decision? (cast,
// activate, attack, block). A step or turn change clears it: what they did
// in an earlier step has been seen and answered already.
function oppActedSince(fi) {
  let acted = false;
  for (let i = fi + 1; i <= last(); i++) {
    for (const a of S.raw[i].actions || []) {
      if (a.t === 'step' || a.t === 'turn') acted = false;
      else if (a.p === opp() && ACTS.has(a.t)) acted = true;
    }
  }
  return acted;
}

function autoAnswer(d) {
  const refs = d.refs || [];
  if (S.plan) {
    const a = planAnswer(d);
    if (a != null) return a;
  }
  if (d.kind === 'pay_mana') return S.payAssign || S.autoRest || PREF.autoPayNoAsk ? planPay(d) : null;  // nothing is paid unless the player committed a payment
  if (d.kind === 'order_triggers' && new Set(d.options).size === 1) return 0;  // identical triggers: the order cannot matter
  if (d.kind === 'priority') {
    S.payPlan = null; S.payAssign = null; S.autoRest = false;  // a cast is over
    const pass = refs.findIndex(r => r.type === 'pass');
    if (pass < 0) return null;
    if (S.fullControl) return null;
    if (manaWaiting(stateAt(last()))) return null;  // mana you tapped is waiting: never pass it away silently
    if (!refs.some(isReal)) return pass;  // mana abilities alone never stop
    const s = stateAt(last());
    const top = s.stack[s.stack.length - 1];
    const acted = oppActedSince(S.lastMine);
    const canRespond = refs.some((r, i) => timely(r, d.options[i]));
    if (top && top.controller === S.seat && S.holdOnce) { S.holdOnce = false; return null; }  // hold priority once
    if (top && top.controller !== S.seat) return canRespond && !S.yields.has(top.card || top.name.split(':')[0]) ? null : pass;  // something of theirs to answer
    if (acted && canRespond) return null;  // never pass right after they did something you could answer
    if (top && canRespond) return null;  // my own spell or ability is on the stack and I can add to it (stack a second Krark-Clan Shaman activation): keep the stop
    if (top) return pass;  // my own spell, nothing worth adding: let it resolve (they may still respond)
    const mine = s.active === S.seat;
    if (S.passMode === 'opp') {
      if (mine && s.turn !== S.passTurn) S.passMode = null; else return pass;
    }
    // Combat tricks: the first priority after blockers are declared, in either
    // player's turn, stops whenever an instant-speed play exists (whatever
    // the stop settings say). Later priorities in the same step don't.
    const prevS = S.lastPriority >= 0 ? stateAt(S.lastPriority) : null;
    const firstInStep = !prevS || prevS.turn !== s.turn || prevS.step !== s.step;
    const blocked = s.battlefield.some(c => c.blocking != null);
    if (s.step === 'declare_blockers' && firstInStep && canRespond && s.battlefield.some(c => c.attacking) && (!mine || blocked)) return null;
    if (S.passMode) return null;
    return PREF.stops[mine ? 'me' : 'opp'][s.step] ? null : pass;
  }
  if (d.kind === 'target' && PREF.autoTarget) {
    const real = refs.filter(r => !r.none);
    if (real.length === 1 && refs.length === 1) return 0;
  }
  return null;
}

// The rest of a gesture the client collected (attackers, blockers, a target).
function planAnswer(d) {
  const P = S.plan, refs = d.refs || [];
  const s = stateAt(last());
  // A plan answers only decisions of the turn and step it was made in: an
  // attack or block plan must never reach a later combat (it would answer
  // "no attack" / "no block" for you).
  if (P.turn !== s.turn || P.step !== s.step) { S.plan = null; return null; }
  const nameOf = oid => s.battlefield.find(c => c.oid === oid)?.name;
  if (P.kind === 'attack' && d.kind === 'declare_attacker') {
    let i = refs.findIndex(r => P.want.includes(r.attacker));
    if (i < 0) i = refs.findIndex(r => r.attacker != null && P.want.some(o => nameOf(o) === r.name));
    if (i >= 0) {
      const exact = P.want.indexOf(refs[i].attacker);
      P.want.splice(exact >= 0 ? exact : P.want.findIndex(o => nameOf(o) === refs[i].name), 1);
      return i;
    }
    S.plan = null;
    return refs.findIndex(r => r.done);
  }
  if (P.kind === 'block' && d.kind === 'declare_blocker') {
    const b = refs[0]?.blocker, want = P.map.get(b);
    let i = want == null ? -1 : refs.findIndex(r => r.attacker === want);
    if (i < 0 && want != null) i = refs.findIndex(r => r.attacker != null && r.attacker_name === nameOf(want));
    return i >= 0 ? i : refs.findIndex(r => r.none);
  }
  if (P.kind === 'target' && d.kind === 'target') {
    S.plan = null;
    const t = P.t;
    const i = refs.findIndex(r => (t.oid != null && r.oid === t.oid) || (t.player != null && r.player === t.player) || (t.sid != null && r.sid === t.sid));
    return i >= 0 ? i : null;
  }
  if (d.kind === 'priority' && P.kind !== 'target') S.plan = null;
  return null;
}

// Pay mana like a careful player: floating mana first; then the source that
// keeps the most options open (basic lands, single-colour sources, lands
// before artifacts before creatures); sacrifices and filters last.
function autoPay(d) {
  if (Number.isInteger(d.auto) && d.auto >= 0 && d.auto < d.options.length) return d.auto;  // live_proto.auto_pay_index
  const s = stateAt(last());
  const perm = oid => s.battlefield.find(c => c.oid === oid);
  const count = new Map();
  d.refs.forEach(r => { if (r.oid != null) count.set(r.oid, (count.get(r.oid) || 0) + 1); });
  let best = 0, bestKey = null;
  d.refs.forEach((r, i) => {
    const label = d.options[i], p = r.oid != null ? perm(r.oid) : null;
    const types = p?.types || [];
    const key = [
      r.pool ? 0 : 1,
      /^Sacrifice/.test(label) ? 1 : 0,
      r.via === 'filter' ? 1 : 0,
      types.includes('Creature') ? 2 : types.includes('Land') ? 0 : 1,
      count.get(r.oid) || 1,
      /^(Plains|Island|Swamp|Mountain|Forest|Wastes)$/.test(r.name || '') ? 0 : 1,
      i,
    ];
    if (!bestKey || lexLess(key, bestKey)) { best = i; bestKey = key; }
  });
  return best;
}

// ---- tapping for mana (Arena/MTGO style)
// Clicking an untapped land (or other plain mana source) at priority taps it
// for mana. The engine cannot tap plain lands outside a payment, so the
// client holds that mana as a reservation: the source shows tapped with its
// colour in your mana pool, and the next payment spends it first (the same
// source, that colour if the cost can use it). Like real floating mana it is
// gone when the step ends (the land simply stays untapped). Sources that
// sacrifice themselves (Eldrazi Spawn) use the engine's own mana ability and
// float for real.
const manaInfo = name => { const i = S.cards[name]; return i && i.mana && !i.mana_sac ? i.mana : null; };
const isReserved = oid => S.reserved.some(r => r.oid === oid);
function canTapForMana(c) {
  return c && c.controller === S.seat && !c.tapped && manaInfo(c.name) && !(c.power != null && c.sick) && !isReserved(c.oid);
}
function reserveMana(oid, color) {
  const s = stateAt(last()), c = s.battlefield.find(x => x.oid === oid);
  if (!canTapForMana(c)) return;
  S.reserved.push({oid, name: c.name, color, turn: s.turn, step: s.step});
  S.justTapped = oid;
  sound('tap');
  closePop();
  render(S.shown, {noAnim: true}); renderDock();
}
function unreserve(oid) {
  S.reserved = S.reserved.filter(r => r.oid !== oid);
  render(S.shown, {noAnim: true}); renderDock();
}
function tapForMana(el) {  // a click on one of your mana sources at priority
  const s = stateAt(last());
  const oids = el.dataset.oids.split(' ').map(Number);
  const back = [...oids].reverse().find(isReserved);
  if (back != null) { unreserve(back); return true; }
  const c = oids.map(o => s.battlefield.find(x => x.oid === o)).find(canTapForMana);
  if (!c) return false;
  const colors = manaInfo(c.name);
  if (colors.length === 1) reserveMana(c.oid, colors[0]);
  else colorPicker(el, c, colors);
  return true;
}
const MANA_NAME = {W: 'white', U: 'blue', B: 'black', R: 'red', G: 'green', C: 'colourless'};
function colorPicker(anchor, c, colors) {
  const pop = $('#pop');
  pop.innerHTML = `<div class="ph">Tap ${esc(c.name)} for</div><div class="pips">${colors.map(k => `<button class="pip ${k}" data-mana="${k}" title="${MANA_NAME[k] || k} (${k})">${k}</button>`).join('')}</div>`;
  pop.classList.add('on');
  const r = anchor.getBoundingClientRect(), pw = pop.offsetWidth, ph = pop.offsetHeight;
  let x = r.left + r.width / 2 - pw / 2, y = r.top - ph - 8;
  if (y < 50) y = r.bottom + 8;
  pop.style.left = `${Math.max(8, Math.min(innerWidth - pw - 8, x))}px`; pop.style.top = `${Math.max(8, Math.min(innerHeight - ph - 8, y))}px`;
  pop._source = anchor; pop._plan = undefined; pop._mana = {oid: c.oid, colors};
}
// Reservations last for the step they were made in, like a mana pool.
function expireReserved() {
  if (!S.reserved.length || last() < 0) return;
  const s = stateAt(last());
  const keep = S.reserved.filter(r => r.turn === s.turn && r.step === s.step && s.battlefield.some(c => c.oid === r.oid && c.controller === S.seat && !c.tapped));
  const gone = S.reserved.filter(r => !keep.includes(r));
  S.reserved = keep;
  const lost = gone.filter(r => r.turn !== s.turn || r.step !== s.step);
  if (lost.length) toast(`Unused mana emptied at the end of the step: ${lost.map(r => `${r.color} from ${r.name}`).join(', ')}. The land was not tapped.`);
}
const poolCount = s => Object.values(s.players[S.seat].pool || {}).reduce((a, b) => a + b, 0);
const manaWaiting = s => poolCount(s) + S.reserved.length;
function sameSource(s, a, b) {  // the engine offers one of several interchangeable permanents
  if (a === b) return true;
  const x = s.battlefield.find(c => c.oid === a), y = s.battlefield.find(c => c.oid === b);
  return !!(x && y && x.name === y.name && x.controller === y.controller && !!x.tapped === !!y.tapped && !!x.sick === !!y.sick);
}
// Pay one pip: floating engine mana, then your reserved mana (that source, the
// tapped colour if the cost can use it, else its other colour), then the
// sources the tap preview showed for this cast, then auto-pay.
function planPay(d) {
  const s = stateAt(last()), refs = d.refs;
  const pool = refs.findIndex(r => r.pool != null);
  if (pool >= 0) return pool;
  for (let k = 0; S.payAssign && k < S.payAssign.length; k++) {  // the sources the player picked in the paying step
    const a = S.payAssign[k];
    if (a.pool) continue;
    const ok = r => r.oid != null && r.via !== 'filter' && sameSource(s, r.oid, a.oid);
    let j = refs.findIndex(r => ok(r) && r.color === a.color);
    if (j < 0) j = refs.findIndex(ok);
    if (j >= 0) { S.payAssign.splice(k, 1); S.reserved = S.reserved.filter(x => x.oid !== a.oid); return j; }
  }
  for (let k = 0; k < S.reserved.length; k++) {
    const res = S.reserved[k];
    const ok = r => r.oid != null && r.via !== 'filter' && sameSource(s, r.oid, res.oid);
    let j = refs.findIndex(r => ok(r) && r.color === res.color);
    if (j < 0) j = refs.findIndex(ok);
    if (j >= 0) { S.reserved.splice(k, 1); return j; }
  }
  if (S.payPlan) {
    for (let k = 0; k < S.payPlan.length; k++) {
      const j = refs.findIndex(r => r.oid != null && r.via !== 'filter' && sameSource(s, r.oid, S.payPlan[k]));
      if (j >= 0) { S.payPlan.splice(k, 1); return j; }
    }
  }
  return autoPay(d);
}

// ---- the paying step (MTG Arena style): a spell you play waits, lifted next
// to the stack with its cost as pips, until you pay it: click sources (each
// fills a pip) or Auto pay (Space/Enter). Nothing has been sent yet, so Cancel
// (Esc, right-click) simply puts the card back. On commit the client sends the
// cast, the target and then answers the engine's pay steps with the sources
// you picked (`planPay`); auto pay covers the rest the way the preview showed.
function parseCost(cost) {
  const pips = [];
  for (const m of String(cost).matchAll(/\{([^}]+)\}/g)) {
    const t = m[1];
    if (/^\d+$/.test(t)) for (let k = 0; k < +t; k++) pips.push({sym: '1', generic: true});
    else pips.push({sym: t, generic: false});
  }
  return pips.filter(p => p.generic).concat(pips.filter(p => !p.generic));  // generic first, as printed
}
function beginCast(el, i, plan) {
  const d = myDecision(), r = d?.refs[i];
  if (!r) return;
  closePop();
  const needsPay = r.cost && !(r.before_pay || []).includes('choose_x');
  if (PREF.autoPayNoAsk || !needsPay) {  // a land, a free spell, or the opt-in instant cast
    if (!PREF.autoPayNoAsk && r.cost) return act(i, plan);  // X spells: the engine asks X first, then the pay steps wait for you
    return act(i, plan, undefined, PREF.autoPayNoAsk ? {assign: [], rest: tapsFor([i])?.taps.slice() || null} : undefined);
  }
  const s = stateAt(last());
  const P = {i, name: r.name || el?.dataset.name, uid: el?.dataset.uid != null ? +el.dataset.uid : null, srcOid: el?.dataset.oid != null ? +el.dataset.oid : null,
    verb: r.type === 'activate' ? 'Activating' : 'Casting', cost: r.cost, pips: parseCost(r.cost), plan, targets: null, step: 'pay', taps: r.taps || []};
  const real = (r.targets || []).filter(t => !t.none);
  if (real.length && !(plan && plan.kind === 'target')) {
    P.targets = r.targets;
    P.step = 'target';
    if (PREF.autoTarget && real.length === 1 && r.targets.length === 1) { P.plan = {kind: 'target', t: real[0].player != null ? {player: real[0].player} : real[0].sid != null ? {sid: real[0].sid} : {oid: real[0].oid}}; P.step = 'pay'; }
  }
  P.fi = last();
  S.paying = P;
  // floating mana and lands you tapped before casting fill pips first
  for (const [c, n] of Object.entries(s.players[S.seat].pool || {})) for (let k = 0; k < n; k++) fillPip({pool: c, color: c}, c);
  for (const res of S.reserved) fillPip({oid: res.oid, color: res.color, res: true, name: res.name}, res.color);
  sound('cast');
  refreshPaying();
}
function openPip(color) {  // the pip `color` mana would fill: its own colour first, else generic
  const P = S.paying;
  return P.pips.findIndex(p => !p.by && !p.generic && p.sym === color) >= 0 ? P.pips.findIndex(p => !p.by && !p.generic && p.sym === color)
    : P.pips.findIndex(p => !p.by && p.generic);
}
function fillPip(by, color) {
  const k = openPip(color);
  if (k < 0) return false;
  S.paying.pips[k].by = by;
  return true;
}
const payingHeld = oid => S.paying?.pips.some(p => p.by && p.by.oid === oid);
function payWith(el) {  // a click on a permanent while paying
  const P = S.paying, s = stateAt(last());
  const oids = el.dataset.oids.split(' ').map(Number);
  const held = [...oids].reverse().find(o => payingHeld(o));
  if (held != null) {  // click it again: take it back
    const k = P.pips.findIndex(p => p.by && p.by.oid === held);
    const by = P.pips[k].by;
    P.pips[k].by = null;
    if (by.res) S.reserved = S.reserved.filter(x => x.oid !== held);  // a pre-tapped land untaps too
    return refreshPaying();
  }
  const c = oids.map(o => s.battlefield.find(x => x.oid === o)).find(x => canTapForMana(x) && !payingHeld(x.oid));
  if (!c) { if (oids.some(o => s.battlefield.find(x => x.oid === o)?.controller === S.seat)) toast(`${el.dataset.name} can't make mana right now.`); return; }
  const colors = manaInfo(c.name).filter(k => openPip(k) >= 0);
  if (!colors.length) { toast(`${c.name} can't pay any of the open pips.`); return; }
  // several colours: ask only when the choice matters (they would fill different pips)
  const where = new Set(colors.map(k => S.paying.pips[openPip(k)].generic ? '1' : k));
  const pay = color => { closePop(); fillPip({oid: c.oid, color, name: c.name}, color); S.justTapped = c.oid; sound('tap'); refreshPaying(); };
  if (where.size > 1) { colorPicker(el, c, colors); $('#pop')._mana.cb = (oid, color) => pay(color); return; }
  pay(colors[0]);
}
function chooseCastTarget(t) {  // t: {oid} | {player} | {sid}
  const P = S.paying, s = stateAt(last());
  const r = P.targets.find(x => (t.player != null && x.player === t.player) || (t.sid != null && x.sid === t.sid) || (t.oid != null && x.oid != null && sameSource(s, x.oid, t.oid)));
  if (!r) return false;
  P.plan = {kind: 'target', t: r.player != null ? {player: r.player} : r.sid != null ? {sid: r.sid} : {oid: r.oid}};
  P.step = 'pay';
  refreshPaying();
  return true;
}
const payTargetKeys = () => {
  const P = S.paying, out = new Set();
  if (!P || P.step !== 'target') return out;
  const s = stateAt(last());
  for (const r of P.targets) {
    if (r.player != null) out.add('p' + r.player);
    if (r.sid != null) out.add('s' + r.sid);
    if (r.oid != null) for (const c of s.battlefield) if (sameSource(s, c.oid, r.oid)) out.add('o' + c.oid);
  }
  return out;
};
function payingSuggestion() {  // the extra lands auto pay would tap for the open pips
  const P = S.paying; if (!P) return null;
  const s = stateAt(last()), open = P.pips.filter(p => !p.by).length;
  const held = P.pips.filter(p => p.by && p.by.oid != null).map(p => p.by.oid);
  return {taps: P.taps.filter(o => !held.some(h => sameSource(s, o, h))).slice(0, open), sacs: [], used: []};
}
function refreshPaying() {
  const P = S.paying;
  render(S.shown, {noAnim: true});
  renderDock();
  if (P && P.step === 'pay' && P.pips.every(p => p.by)) { setTimeout(() => { if (S.paying === P) payCommit(); }, 260); }  // paid: it is cast
}
function payCommit() {
  const P = S.paying; if (!P) return;
  if (P.step === 'target') { toast('Choose a target first (or Cancel).'); return; }
  const assign = P.pips.filter(p => p.by).map(p => p.by);
  const sug = payingSuggestion();
  S.paying = null;
  S.autoRest = false;
  return act(P.i, P.plan, undefined, {assign: assign.filter(a => !a.pool).map(a => ({oid: a.oid, color: a.color})), rest: sug ? sug.taps : null});
}
function payCancel(quiet) {
  if (!S.paying) return;
  S.paying = null;
  if (!quiet) toast('Cancelled: the card is back in your hand.');
  render(S.shown, {noAnim: true}); renderDock();
}
function renderPaying() {
  let el = $('#paying');
  const P = S.paying;
  if (!P) { if (el) el.remove(); return; }
  if (!el) { el = document.createElement('div'); el.id = 'paying'; document.body.appendChild(el); }
  const filled = P.pips.filter(p => p.by).length;
  el.innerHTML = `<div class="ph">${esc(P.verb)}</div>${cardHtml(P.name)}
    <div class="pips cost">${P.pips.map(p => `<span class="pip ${p.by ? 'paid ' + esc(p.by.color) : p.generic ? 'gen' : esc(p.sym)}" title="${p.by ? esc(`${p.by.color} from ${p.by.pool ? 'your mana pool' : p.by.name || 'a land'}`) : 'unpaid'}">${esc(p.by ? p.by.color : p.sym)}</span>`).join('')}</div>
    <div class="pstep">${P.step === 'target' ? 'Choose a target (glowing).' : `${filled}/${P.pips.length} paid. Click lands to tap them, or Auto pay (Space).`}</div>
    <div class="pbtns"><button class="btn small" data-pay="cancel" title="Put the card back (Esc, right-click)">Cancel <kbd>Esc</kbd></button></div>`;
}

function lexLess(a, b) {
  for (let j = 0; j < a.length; j++) if (a[j] !== b[j]) return a[j] < b[j];
  return false;
}

// ---------------------------------------------------------------- playback of what happened (P3)
const beatScale = () => SPEEDS[PREF.speed] ?? 1;
async function wait(ms) {
  ms *= beatScale();
  const end = performance.now() + ms;
  while (!S.skip && performance.now() < end) await sleep(Math.min(40, end - performance.now()));
}

async function playFrame(i) {
  const f = S.raw[i];
  const acts = f.actions || [];
  const theirs = acts.filter(a => a.p === opp());
  const notable = acts.some(a => ['cast', 'play', 'activate', 'attack', 'block', 'resolve', 'trigger', 'enter', 'leave', 'dies', 'turn', 'discard', 'hit', 'life'].includes(a.t));
  const isLast = i === last();
  const cast = theirs.find(a => a.t === 'cast' || a.t === 'activate' || a.t === 'plot');
  const td = f.decision && f.decision.player === opp() && f.decision.kind === 'target' ? f.decision : null;
  if (cast && td) S.spotDone.add(i);  // the target joins the cast's spotlight
  const choice = S.spotDone.has(i) ? null : oppChoiceSpot(i);
  if (!notable && !isLast && i !== 0 && !choice && !cast) { appendLog(i); return; }
  if (cast && !S.skip && beatScale() > 0) {
    const name = cast.t === 'activate' ? cast.name.split(':')[0].trim() : cast.name;
    await ensureImg(name, 900);
    const aim = td ? oppAim(td.options[0]) : '';
    const atMe = /^(you|your )/.test(aim);
    const title = cast.t === 'cast' ? `${oppLabel()} casts ${cast.name}` : cast.t === 'plot' ? `${oppLabel()} plots ${cast.name}` : `${oppLabel()} activates ${cast.name}`;
    if (aim) await bigSpot({name, title, lines: [`→ ${aim}`], atMe});  // a target is a choice that matters: the long spotlight
    else { showSpot(name, title, atMe); await wait(950); hideSpot(); }
  }
  if (choice) await bigSpot(choice);
  // Combat and deaths play out on the board as it was, before the new state lands.
  const fought = await playCombat(acts);
  await playDeaths(acts);
  render(i);
  appendLog(i);
  if (acts.some(a => a.t === 'cast' || a.t === 'activate')) sound('cast');
  else if (acts.some(a => a.t === 'play')) sound('land');
  const turnEv = acts.filter(a => a.t === 'turn').at(-1);
  if (turnEv && i > 0) turnBanner(turnEv.p === S.seat);
  if (theirs.some(a => a.t === 'attack')) nudgeAttackers();
  if (fought && !S.skip) await wait(300);  // hold the result a moment before input opens
  if (f.state.players.some(P => P.life <= 0)) await wait(600);  // the lethal blow lands before the result
  if (isLast) return;
  let beat = 0;
  if (theirs.some(a => a.t === 'play')) beat = Math.max(beat, 420);
  if (theirs.some(a => a.t === 'attack' || a.t === 'block')) beat = Math.max(beat, 700);
  if (acts.some(a => a.t === 'resolve' || a.t === 'trigger')) beat = Math.max(beat, 380);
  if (acts.some(a => a.t === 'dies' || a.t === 'leave')) beat = Math.max(beat, 380);
  if (acts.some(a => a.t === 'turn')) beat = Math.max(beat, 300);
  if (beat) await wait(beat);
}

// ---- combat on screen: lunges, impacts, damage numbers, deaths
const sleepFx = ms => wait(ms);
function floatText(el, text, cls) {
  const b = $('#board').getBoundingClientRect(), r = el.getBoundingClientRect();
  const d = document.createElement('div');
  d.className = 'floatnum ' + (cls || '');
  d.textContent = text;
  d.style.left = `${r.left - b.left + r.width / 2}px`;
  d.style.top = `${r.top - b.top + r.height * 0.35}px`;
  $('#board').appendChild(d);
  setTimeout(() => d.remove(), 1100 * Math.max(0.3, beatScale()));
}
async function playCombat(acts) {
  const hits = acts.filter(a => a.t === 'hit');
  if (!hits.length || S.skip || beatScale() === 0) return false;
  const k = beatScale();
  for (const h of hits) {
    const src = permEl(h.oid);
    const tgt = h.p != null ? $(`#plate${h.p === S.seat ? 0 : 1}`) : permEl(h.to_oid);
    if (!src || !tgt) continue;
    const a = src.getBoundingClientRect(), t = tgt.getBoundingClientRect();
    const dx = t.left + t.width / 2 - (a.left + a.width / 2), dy = t.top + t.height / 2 - (a.top + a.height / 2);
    const len = Math.hypot(dx, dy) || 1, ux = dx / len * 34, uy = dy / len * 34;
    src.animate([{translate: '0 0'}, {translate: `${ux}px ${uy}px`, offset: 0.55}, {translate: '0 0'}], {duration: 320 * k, easing: 'cubic-bezier(.3,.7,.3,1)'});
    setTimeout(() => {
      tgt.animate([{translate: '0 0'}, {translate: '-4px 0'}, {translate: '4px 0'}, {translate: '0 0'}], {duration: 120, iterations: 2});
      floatText(tgt, `-${h.n}`, h.p != null ? 'big' : '');
      sound('hit', h.n);
    }, 180 * k);
  }
  await sleepFx(560);
  return true;
}
async function playDeaths(acts) {
  const dead = acts.filter(a => a.t === 'leave' && a.to === 'graveyard').map(a => permEl(a.oid)).filter(Boolean);
  if (!dead.length || S.skip || beatScale() === 0) return;
  const k = beatScale(), b = $('#board').getBoundingClientRect();
  for (const el of dead) {
    const ctl = +el.dataset.ctl, grave = $(`#plate${ctl === S.seat ? 0 : 1} .zone[data-zone="graveyard"]`);
    const r = el.getBoundingClientRect(), g = grave ? grave.getBoundingClientRect() : r;
    const ghost = el.cloneNode(true);
    ghost.classList.add('dying');
    Object.assign(ghost.style, {position: 'absolute', left: `${r.left - b.left}px`, top: `${r.top - b.top}px`, margin: 0, zIndex: 140, pointerEvents: 'none'});
    $('#board').appendChild(ghost);
    el.style.visibility = 'hidden';
    ghost.animate([{filter: 'none', scale: 1, opacity: 1}, {filter: 'grayscale(1) brightness(.7)', scale: 0.85, opacity: 1, offset: 0.45},
      {filter: 'grayscale(1)', scale: 0.25, opacity: 0.2, translate: `${g.left + g.width / 2 - r.left - r.width / 2}px ${g.top + g.height / 2 - r.top - r.height / 2}px`}],
      {duration: 600 * k, easing: 'cubic-bezier(.5,0,.7,1)', fill: 'forwards'});
    setTimeout(() => { ghost.remove(); grave?.animate([{scale: 1}, {scale: 1.25}, {scale: 1}], {duration: 260}); }, 600 * k);
  }
  sound('death');
  await sleepFx(620);
}
// A short banner when a turn starts (skipped at instant replay speed).
function turnBanner(mine) {
  if (beatScale() === 0 || S.skip) return;
  $$('.turnbanner').forEach(x => x.remove());
  const b = document.createElement('div');
  b.className = `turnbanner ${mine ? 'me' : 'opp'}`;
  b.textContent = mine ? 'Your turn' : "Opponent's turn";
  $('#board').appendChild(b);
  if (mine) sound('turn');
  setTimeout(() => b.remove(), 900);
}

// Freshly declared attackers step forward once.
function nudgeAttackers() {
  for (const el of $$('#board .perm.attacking')) el.animate([{translate: '0 0'}, {translate: `0 ${el.closest('#side1') ? 14 : -14}px`}, {translate: '0 0'}], {duration: 380, easing: 'ease-out'});
  sound('attack');
}

function showSpot(name, cap, atMe) {
  const el = $('#spot');
  el.className = '';
  el.innerHTML = cardHtml(name) + `<div class="cap ${atMe ? 'atme' : ''}">${esc(cap)}</div>`;
  el.classList.add('on');
}
function hideSpot() { $('#spot').classList.remove('on', 'big'); }

// ---- the opponent's choices that matter, in a centred spotlight that stays
// PREF.spotMs (Settings) or until a click or Space: Brainstorm and Ponder put
// backs, scry, Delver's reveal, mulligans, modes, X, targets, blocks,
// sacrifices, and their sideboarding before games 2 and 3. Hidden parts stay
// hidden: the server sends only counts and where cards went (live_proto.hidden_summary).
async function bigSpot(sp) {
  if (S.skip || beatScale() === 0 || !(+PREF.spotMs > 0)) return;
  if (sp.name && S.cards[sp.name]) await ensureImg(sp.name, 900);
  const el = $('#spot'), ms = +PREF.spotMs;
  el.className = '';
  el.innerHTML = `${sp.name && S.cards[sp.name] ? cardHtml(sp.name) : backHtml()}<div class="spotbox ${sp.atMe ? 'atme' : ''}"><div class="st">${esc(sp.title)}</div>${(sp.lines || []).map(l => `<div class="sl">${esc(l)}</div>`).join('')}
    <div class="bar"><i style="animation-duration:${ms}ms"></i></div><div class="hint">click or Space to continue</div></div>`;
  el.classList.add('on', 'big');
  if (!pinned) $('#hoverprev').classList.remove('on');
  sound('decide');
  S.spotSkip = false;
  const end = performance.now() + ms;
  while (!S.skip && !S.spotSkip && performance.now() < end) await sleep(40);
  S.spotSkip = false;
  hideSpot();
}
const CHOICE_KINDS = new Set(['yes_no', 'choose_mode', 'choose_x', 'sacrifice', 'exile_from_graveyard', 'mulligan', 'declare_blocker', 'order', 'choose_card', 'hidden', 'target']);
function oppChoiceSpot(i) {
  const f = S.raw[i], d = f.decision;
  if (i === 0 && S.meta?.match?.game_no > 1 && S.meta.match.opp_swaps != null && !S.spotDone.has('sb' + S.gid + S.meta.match.game_no)) {
    S.spotDone.add('sb' + S.gid + S.meta.match.game_no);
    const n = S.meta.match.opp_swaps;
    return {title: `Game ${S.meta.match.game_no}: ${oppLabel()} sideboards`, lines: [n ? `${n} cards in, ${n} out (which ones stays hidden)` : 'no changes: the maindeck again']};
  }
  if (!d || d.player !== opp() || d.chosen == null) return null;
  const r = d.refs?.[0] || {}, kind = r.type === 'hidden' ? 'hidden' : d.kind;
  if (!CHOICE_KINDS.has(kind) && !CHOICE_KINDS.has(d.kind)) return null;
  if (d.kind === 'declare_blocker' || d.kind === 'mulligan' || r.group === 'bottom') return groupSpot(i, d.kind === 'declare_blocker' ? 'declare_blocker' : 'mulligan');
  return groupSpot(i, 'source');
}
function choiceLine(d) {
  const r = d.refs?.[0] || {};
  if (r.type === 'hidden') return r.text;
  let line = clean(d.options[0]);
  if (d.kind === 'choose_x') return `chooses ${line.replace(/^X\s*=?\s*/i, 'X = ')}`;
  if (d.kind === 'yes_no') {
    if (/^Reveal /.test(line)) return `reveals ${line.slice(7)} from the top of the library`;
    if (/^Don't reveal/.test(line)) return 'does not reveal the top card';
    if (line === 'Shuffle') return 'shuffles the library';
    if (line === "Don't shuffle") return 'does not shuffle';
  }
  if (d.kind === 'target') return `→ ${oppAim(d.options[0])}`;
  return line;
}
// Consecutive opponent choices become one spotlight: everything one spell
// or ability decided while it resolved (Brainstorm's put backs, Ponder's
// order and shuffle, a scry), a mulligan and its bottoms, a combat's blocks.
function groupSpot(i, key) {
  const srcOf = j => S.raw[j].state.stack.at(-1)?.card || S.raw[j].state.stack.at(-1)?.name || null;
  const same = j => {
    const d = S.raw[j]?.decision;
    if (!d || d.player !== opp() || d.chosen == null) return false;
    const r = d.refs?.[0] || {};
    if (key === 'mulligan') return d.kind === 'mulligan' || r.group === 'bottom';
    if (key === 'declare_blocker') return d.kind === 'declare_blocker';
    if (d.kind === 'declare_blocker' || d.kind === 'mulligan' || !(r.type === 'hidden' || CHOICE_KINDS.has(d.kind) || d.kind === 'target')) return false;
    return srcOf(j) === srcOf(i) && (srcOf(i) != null || (r.source && r.source === (S.raw[i].decision.refs?.[0] || {}).source));
  };
  const js = [i];
  for (const j of S.queue) { if (j <= js.at(-1)) continue; if (j === js.at(-1) + 1 && same(j)) { js.push(j); S.spotDone.add(j); } else break; }
  const ds = js.map(j => S.raw[j].decision), refs = ds.map(d => d.refs?.[0] || {});
  const top = S.raw[i].state.stack.at(-1), src = top?.card || null;
  if (key === 'declare_blocker') {
    const real = ds.filter(d => !d.refs?.[0]?.none);
    const lines = real.map(d => plainBlock(clean(d.options[0])));
    return {name: real.length ? S.raw[i].state.battlefield.find(c => c.oid === real[0].refs[0].blocker)?.name : null,
      title: `${oppLabel()} ${lines.length ? 'blocks' : 'does not block'}`, lines};
  }
  if (key === 'mulligan') {
    const mull = ds.filter(d => d.kind === 'mulligan').map(d => clean(d.options[0]));
    const bottoms = refs.filter(r => r.group === 'bottom').length;
    const lines = mull.map(t => t.replace(/^Keep \((\d+) cards\)$/, 'keeps $1').replace(/^Mulligan \(to (\d+)\)$/, 'mulligans to $1'));
    if (bottoms) lines.push(`puts ${bottoms} card${bottoms > 1 ? 's' : ''} on the bottom`);
    if (lines.length === 1 && /^keeps 7$/.test(lines[0])) return null;  // a plain keep is not worth a pause
    return {title: `${oppLabel()}'s opening hand`, lines};
  }
  // one line per choice, counting repeated put backs and scry cards
  const lines = [];
  const putBack = refs.filter(r => r.group === 'put_back').length;
  const scry = refs.filter(r => r.group === 'scry'), scryTop = scry.filter(r => /top/.test(r.text)).length;
  let saidPut = false, saidScry = false;
  ds.forEach((d, k) => {
    const r = refs[k];
    if (r.group === 'put_back') { if (!saidPut) lines.push(`puts ${putBack} card${putBack > 1 ? 's' : ''} from hand back on top`); saidPut = true; return; }
    if (r.group === 'scry') {
      if (!saidScry) lines.push(`${r.source || 'Scry'}: ` + [scryTop ? `keeps ${scryTop} on top` : '', scry.length - scryTop ? `puts ${scry.length - scryTop} on the bottom` : ''].filter(Boolean).join(', '));
      saidScry = true; return;
    }
    lines.push(choiceLine(d));
  });
  const r0 = refs[0];
  const name = src || (r0.source && S.cards[r0.source] ? r0.source : null);
  const label = src || (r0.source && !/^Scry|^Mulligan/.test(r0.source) ? r0.source : '');
  return {name, title: `${oppLabel()}${label ? ` · ${label.split(':')[0]}` : ''}`, lines, atMe: lines.some(l => /→ (you|your )/.test(l))};
}
const oppAim = label => plainTarget(label, opp()).replace(/^the opponent$/, 'themselves').replace(/^the opponent's /, 'their own ');  // their choice, in my words
function plainBlock(label) {  // "Insectile Aberration blocks Sagu Wildling" from their side
  return label.replace(/ \((self|opponent)\)/g, '').replace(/^(.+?) blocks (.+)$/, '$1 blocks your $2');
}
const oppLabel = () => S.meta ? 'Opponent' : 'Opponent';

// The opponent's actions as chips (each can be flagged), held until you act.
function mine(label) {  // my own choices: "(opponent)" is theirs, "(self)" is mine
  return label.replace(/player (\d) \((self|opponent)\)/, (_, p) => +p === S.seat ? 'yourself' : 'the opponent')
    .replace(/ \(opponent\)/, ' (theirs)').replace(/ \(self\)/, ' (yours)');
}
function humanize(label) {
  return clean(label)
    .replace(/player (\d) \((self|opponent)\)/, (_, p) => +p === S.seat ? 'you' : 'themselves')
    .replace(/ \(opponent\)/, ' (yours)').replace(/ \(self\)/, ' (theirs)');
}
function pushFeed(text, fi, quiet, kind) {
  const e = {text, fi, quiet, kind, turn: S.shown >= 0 ? stateAt(S.shown).turn : 0};
  S.feed.push(e);
  if (!quiet && kind !== 'impact') { S.lastOpp = text; S.lastOppFi = fi; }
  if (S.feed.length > 14) S.feed.shift();
  // the cast line an opponent's target is later added to updates its feed entry too
  const lc = S.lastCast[opp()];
  if (lc && !kind && lc.text.replace(/^Opponent /, '') === text) lc.feed = e;
  renderFeed();
}
// ---- reports: "Bot played wrong" (a bot play) and "Bug: engine / UI"; the server files them as GitHub issues
const pseudonym = () => loadJSON('mtgml-play-name', '') || '';
function isFlagged(fi) { return (S.flags || []).some(f => f.frame === fi && f.category === 'bot'); }
function botPlays() {  // the bot's recent public plays, newest first
  const out = [];
  for (let i = S.raw.length - 1; i >= 0 && out.length < 12; i--) {
    const d = S.raw[i].decision;
    if (d && d.player === opp() && d.chosen != null && !['pass', 'pay', 'hidden'].includes(d.refs?.[0]?.type) && !d.refs?.[0]?.done) out.push([i, `Turn ${S.raw[i].state.turn}: ${humanize(d.options[d.chosen])}`]);
  }
  return out;
}
function toggleFlag(fi) { openFlagForm('bot', fi); }
function openFlagForm(category, fi, review) {
  if (!S.gid) return;
  const bot = category === 'bot', plays = review != null ? [] : bot ? botPlays() : [];
  if (bot && review == null && !plays.length) { toast('The bot has not made a play yet.'); return; }
  const sel = fi != null && plays.some(([i]) => i === fi) ? fi : plays[0]?.[0];
  const m = $('#modal');
  m.innerHTML = `<div class="mbox flagbox"><h2>${bot ? '⚑ Bot played wrong' : '⚠ Bug: engine / UI'}</h2>
    <div class="note">${S.filing === false ? `Saved with the game for the developers (this server does not publish reports).${review != null ? rvDecisionText(review) : ''}` : review != null ? `From the review (the game is over): the public GitHub issue gets everything, including the bot&#39;s hand, its options with probabilities and value, the seed and a command that rebuilds the game up to this decision.${rvDecisionText(review)}` : bot ? 'A bad decision by the bot. It becomes a public GitHub issue with what you could see (never the bot&#39;s hidden cards).' : 'Wrong rules, an illegal play, or the page misbehaving. It becomes a public GitHub issue with the board as you see it.'}</div>
    <form id="fFlag">${bot && review == null ? `<label>Which play</label><select name="frame">${plays.map(([i, t]) => `<option value="${i}" ${i === sel ? 'selected' : ''}>${esc(t)}</option>`).join('')}</select>` : ''}
      <label>What happened? (one line)</label><input name="what" maxlength="200" required placeholder="${bot ? 'e.g. attacked into my 4/4 with a 2/2' : 'e.g. my creature did not untap'}">
      <label>Anything else (optional)</label><textarea name="note" rows="3" maxlength="2000" placeholder="${bot ? 'What would you have done?' : 'Steps, what you expected'}"></textarea>
      <label>Your name or nickname (optional, shown publicly; never your real name)</label><input name="who" maxlength="40" value="${esc(pseudonym())}">
      <div style="display:flex;gap:8px;margin-top:6px"><button class="primary" type="submit">Send report</button><button class="btn" type="button" id="bFlagCancel">Cancel</button></div>
      <div class="result" id="flagResult"></div></form></div>`;
  m.classList.add('on');
  const f = $('#fFlag');
  f.what.focus();
  $('#bFlagCancel').onclick = () => m.classList.remove('on');
  f.onsubmit = async e => {
    e.preventDefault();
    const btn = f.querySelector('button[type=submit]');
    if (btn.disabled) return;
    btn.disabled = true; btn.textContent = 'Sending…';
    saveJSON('mtgml-play-name', f.who.value.trim());
    const body = {category, what: f.what.value.trim(), note: f.note.value, pseudonym: f.who.value.trim()};
    if (review != null) body.review_frame = review;  // the token goes in the header (api)
    else if (bot) body.frame = +f.frame.value;
    try {
      const out = await api(`${encodeURIComponent(S.gid)}/flag`, body);
      S.flags = out.flags;
      renderFeed();
      $$('#log .flag').forEach(b => b.classList.toggle('on', isFlagged(+b.dataset.flag)));
      $('#flagResult').innerHTML = out.filed ? `Thanks. Filed as <a href="${esc(out.issue)}" target="_blank" rel="noopener">${esc(out.issue.replace('https://github.com/', ''))}</a>.` : esc(out.message);
      btn.textContent = out.filed ? 'Filed' : 'Saved';
      $('#bFlagCancel').textContent = 'Close';
    } catch (err) { $('#flagResult').textContent = err.message; btn.disabled = false; btn.textContent = 'Send report'; }
  };
}
function renderFeed() {
  const la = $('#plate1 .lastact');
  if (la) la.innerHTML = lastActHtml();
}
const lastActHtml = () => S.lastOpp ? `<span>Last:</span> ${esc(S.lastOpp)}${S.lastOppFi != null ? `<button class="flagbot" data-flag="${S.lastOppFi}" title="Bot played wrong: report this play">⚑</button>` : ''}` : '';

// ---------------------------------------------------------------- log
// ---- the game log in Magic's words: one line per action, built from the
// parsed events, the decisions and the difference between two states (draws,
// mills). No engine tokens (p0, choose_card, (normal), winner=1).
const who = p => p === S.seat ? 'You' : 'Opponent';
const whose = p => p === S.seat ? 'your' : "the opponent's";
const cleanName = n => String(n).replace(/#\d+/g, '');
const KEYWORDS = {nonlegendary_creature: 'nonlegendary creature', creature_or_planeswalker: 'creature or planeswalker'};
function plainTarget(label, p) {  // "Target Sagu Wildling#12 (opponent)" from p's side -> "your Sagu Wildling" from mine
  const t = cleanName(label).replace(/^Target /, '');
  const m = /^player (\d)/.exec(t);
  if (m) return +m[1] === S.seat ? 'you' : 'the opponent';
  const rel = /\((self|opponent)\)$/.exec(t);
  const name = t.replace(/ \((self|opponent)\)$/, '').replace(/^spell /, '');
  if (!rel) return name;
  const theirs = (rel[1] === 'self') === (p !== S.seat);  // relative to the chooser
  return `${theirs ? "the opponent's" : 'your'} ${name}`;
}
const LOG_KINDS_SKIP = new Set(['priority', 'pay_mana', 'declare_attacker', 'declare_blocker', 'mulligan', 'order_triggers', 'assign_damage', 'assign_damage_amount', 'sacrifice']);
function logEntry(cls, text, extra = '') { return {cls, text, extra}; }
function frameLog(i) {
  const f = S.raw[i], prevS = i > 0 ? S.raw[i - 1].state : null, out = [];
  const nameOf = o => f.state.battlefield.find(c => c.oid === o)?.name || prevS?.battlefield.find(c => c.oid === o)?.name || 'a creature';
  const acts = f.actions || [];
  const resolvedNames = new Set(acts.filter(a => a.t === 'resolve' || a.t === 'play').map(a => a.name));
  const sacrificed = new Set(acts.filter(a => a.t === 'sacrifice').map(a => a.name));
  // the opponent's previous decision caused this frame's events: its line carries the flag
  const cause = i > 0 && S.raw[i - 1].decision && S.raw[i - 1].decision.player === opp() ? i - 1 : null;
  let flagged = false;
  const flagFor = () => { if (cause == null || flagged) return ''; flagged = true; return `<button class="flag ${isFlagged(cause) ? 'on' : ''}" data-flag="${cause}" data-fi="${cause}" title="Flag this play">⚑</button>`; };
  let lineIdx = 0;
  const lines = f.events || [];
  for (const a of acts) {
    // align echo lines (decisions) with the events: walk the raw lines alongside
    while (lineIdx < lines.length && lines[lineIdx].startsWith('  ')) { echo(lines[lineIdx]); lineIdx++; }
    lineIdx++;
    const p = a.p;
    switch (a.t) {
      case 'turn': out.push(logEntry('th', `Turn ${a.turn} · ${a.p === S.seat ? 'your turn' : "opponent's turn"}`)); break;
      case 'play': if (p === S.seat) S.stats.played++; out.push(logEntry(p === S.seat ? 'me' : 'opp', `${who(p)} ${p === S.seat ? 'play' : 'plays'} ${a.name}`, p !== S.seat ? flagFor() : '')); break;
      case 'cast': {
        const how = a.mode && a.mode !== 'normal' ? ` (${a.mode})` : '', from = a.zone && a.zone !== 'hand' ? ` from ${a.zone === 'graveyard' ? whose(p) + ' graveyard' : a.zone}` : '';
        const e = logEntry(p === S.seat ? 'me' : 'opp', `${who(p)} ${p === S.seat ? 'cast' : 'casts'} ${a.name}${how}${from}`, p !== S.seat ? flagFor() : '');
        e.castBy = p; out.push(e); S.lastCast[p] = e; if (p === S.seat) S.stats.played++; break;
      }
      case 'activate': { const e = logEntry(p === S.seat ? 'me' : 'opp', `${who(p)} ${p === S.seat ? 'activate' : 'activates'} ${a.name}`, p !== S.seat ? flagFor() : ''); out.push(e); S.lastCast[p] = e; break; }
      case 'plot': out.push(logEntry(p === S.seat ? 'me' : 'opp', `${who(p)} ${p === S.seat ? 'plot' : 'plots'} ${a.name}`)); break;
      case 'trigger': out.push(logEntry('dim', `Trigger: ${a.name.replace(/: /, ' — ')}`)); break;
      case 'resolve': if (!/: /.test(a.name)) out.push(logEntry('dim', `${a.name} resolves`)); break;
      case 'countered': out.push(logEntry('', `${a.name} is countered`)); break;
      case 'fizzle': out.push(logEntry('', `${a.name} has no legal target and fizzles`)); break;
      case 'enter': if (!resolvedNames.has(a.name)) out.push(logEntry('dim', `${a.name} enters the battlefield`)); break;
      case 'leave': {
        if (a.to === 'graveyard' && sacrificed.has(a.name)) break;
        const creature = prevS?.battlefield.find(c => c.oid === a.oid)?.power != null;
        const t = a.to === 'graveyard' ? (creature ? `${a.name} dies` : `${a.name} is put into the graveyard`) : a.to === 'exile' ? `${a.name} is exiled` : a.to === 'hand' ? `${a.name} returns to its owner's hand` : `${a.name} is put into the library`;
        out.push(logEntry(a.p === S.seat ? 'bad' : '', t));
        if (a.p === S.seat && creature) pushFeed(`Your ${a.name} ${a.to === 'graveyard' ? 'died' : a.to === 'exile' ? 'was exiled' : 'left the battlefield'}`, cause ?? i, false, 'impact');
        break;
      }
      case 'attack': out.push(logEntry(p === S.seat ? 'me' : 'opp', `${who(p)} ${p === S.seat ? 'attack' : 'attacks'} with ${(a.oids || []).map(nameOf).join(', ')}`, p !== S.seat ? flagFor() : '')); break;
      case 'block': {
        const pairs = (a.pairs || []).map(([b, at]) => `${nameOf(b)} blocks ${nameOf(at)}`);
        if (pairs.length) out.push(logEntry(p === S.seat ? 'me' : 'opp', `${who(p)}: ${pairs.join('; ')}`, p !== S.seat ? flagFor() : ''));
        break;
      }
      case 'hit': {
        const to = a.p != null ? (a.p === S.seat ? 'you' : 'the opponent') : a.to_name;
        out.push(logEntry('combat', `${a.name} deals ${a.n} damage to ${to}`));
        if (a.p === S.seat) pushFeed(`${a.name} hit you for ${a.n}`, cause ?? i, false, 'impact');
        break;
      }
      case 'life': {
        const d = a.new - a.old;
        if (d < 0) S.stats[a.p === S.seat ? 'taken' : 'dealt'] -= d;
        out.push(logEntry(`life ${d < 0 ? 'neg' : 'pos'}`, `${a.p === S.seat ? 'Your' : "Opponent's"} life ${a.old} → ${a.new} (${d > 0 ? '+' : '−'}${Math.abs(d)})`));
        if (a.p === S.seat && d < 0) pushFeed(`You lost ${-d} life (${a.new} left)`, cause ?? i, true, 'impact');
        break;
      }
      case 'discard': out.push(logEntry(p === S.seat ? 'me' : 'opp', `${who(p)} ${p === S.seat ? 'discard' : 'discards'} ${a.name}`)); break;
      case 'sacrifice': out.push(logEntry(p === S.seat ? 'me' : 'opp', `${who(p)} ${p === S.seat ? 'sacrifice' : 'sacrifices'} ${a.name}`)); break;
      case 'mulligan': out.push(logEntry(p === S.seat ? 'me' : 'opp', `${who(p)} ${p === S.seat ? 'mulligan' : 'mulligans'} to ${7 - a.n}`)); break;
      case 'game_over': out.push(logEntry('th', `Game over: ${resultText()}`)); break;
      case 'concede': out.push(logEntry('th', `${who(p)} ${p === S.seat ? 'concede' : 'concedes'}`)); break;
      case 'note': out.push(logEntry('dim', cleanName(a.text).replace(/\bp(\d)\b/g, (_, q) => +q === S.seat ? 'you' : 'the opponent'))); break;
      default: break;  // step, mana, dies (the leave line says it)
    }
  }
  while (lineIdx < lines.length) { if (lines[lineIdx].startsWith('  ')) echo(lines[lineIdx]); lineIdx++; }
  // the opponent's public choices that have no event line of their own
  const d = f.decision;
  if (d && d.player === opp() && d.chosen != null) {
    const r = d.refs?.[0] || {}, label = d.options[d.chosen];
    if (r.type === 'target' && S.lastCast[opp()]) S.lastCast[opp()].text += ` → ${plainTarget(label, opp())}`, S.lastCast[opp()].dirty = true;
    else if (d.kind === 'choose_x' && S.lastCast[opp()]) S.lastCast[opp()].text += ` (${label})`, S.lastCast[opp()].dirty = true;
    else if (d.kind === 'yes_no' && !/shuffle/i.test(label)) out.push(logEntry('opp dim', `Opponent: ${cleanName(label)}`));
  }
  // draws and mills come from the state difference; they go before a new turn's header
  const diff = [];
  const discarded = acts.filter(a => a.t === 'discard').map(a => a.name);
  if (prevS) {
    const had = new Set([...prevS.players.flatMap(P => [...P.hand, ...P.graveyard, ...P.exile]), ...prevS.battlefield].map(c => c.uid));
    const drawn = f.state.players[S.seat].hand.filter(c => !had.has(c.uid) && c.uid >= 0).map(c => c.name);
    if (drawn.length) diff.push(logEntry('me dim', `You draw ${drawn.join(', ')}`));
    f.state.players.forEach((P, q) => {
      const milled = P.graveyard.filter(c => !had.has(c.uid) && !S.seenUids.has(c.uid)).map(c => c.name).filter(n => {
        const k = S.pendingSpells.indexOf(n); if (k >= 0) { S.pendingSpells.splice(k, 1); return false; }
        const j = discarded.indexOf(n); if (j >= 0) { discarded.splice(j, 1); return false; }  // discarded from a hidden hand
        return true;
      });
      if (milled.length) diff.push(logEntry('dim', `${milled.join(', ')} ${milled.length > 1 ? 'are' : 'is'} put into ${whose(q)} graveyard from the library`));
    });
  }
  const th = out.findIndex(e => e.cls === 'th');
  // your own turn starting: the draw belongs after its header; otherwise before it
  out.splice(th < 0 ? out.length : out[th].text.endsWith('your turn') ? th + 1 : th, 0, ...diff);
  for (const a of acts) if (a.t === 'cast') S.pendingSpells.push(a.name);
  for (const P of f.state.players) for (const c of [...P.hand, ...P.graveyard, ...P.exile]) if (c.uid >= 0) S.seenUids.add(c.uid);
  for (const c of f.state.battlefield) S.seenUids.add(c.uid);
  return out;

  function echo(line) {  // a decision line "  p0 kind: label"
    const m = /^ {2}p(\d) (\w+): (.*)$/.exec(line);
    if (!m) return;
    const q = +m[1], kind = m[2], label = m[3].replace(/ \((only option|auto)\)$/, '');
    if (LOG_KINDS_SKIP.has(kind) || q !== S.seat) return;
    if (kind === 'target' && S.lastCast[q]) { S.lastCast[q].text += ` → ${plainTarget(label, q)}`; S.lastCast[q].dirty = true; return; }
    if (kind === 'choose_x' && S.lastCast[q]) { S.lastCast[q].text += ` (${label})`; S.lastCast[q].dirty = true; return; }
    if (kind === 'choose_card' && /^Discard /.test(label)) return;  // the discard event says it
    const t = cleanName(label).replace(/^Bottom /, 'You put on the bottom: ').replace(/^Find nothing$/, 'You find nothing').replace(/^Find /, 'You find ').replace(/^Choose /, 'You choose ');
    out.push(logEntry('me dim', /^You /.test(t) ? t : `You: ${t}`));
  }
}
function appendLog(i) {
  if (i < S.logDone) return;
  S.logDone = i + 1;
  const el = $('#log');
  // entries changed after the fact (a target added to its cast) are re-drawn
  for (const e of Object.values(S.lastCast)) if (e && e.dirty && e.el) { e.el.firstChild.textContent = e.text; e.dirty = false; }
  for (const e of frameLog(i)) {
    const div = document.createElement('div');
    div.className = e.cls;
    div.appendChild(document.createTextNode(e.text));
    if (e.extra) div.insertAdjacentHTML('beforeend', e.extra);
    el.appendChild(div);
    e.el = div;
    if (e.cls.startsWith('opp') && !e.cls.includes('dim')) pushFeed(e.text.replace(/^Opponent /, ''), +(div.querySelector('.flag')?.dataset.flag ?? i), false);
  }
  for (const e of Object.values(S.lastCast)) if (e && e.dirty && e.el) { e.el.firstChild.textContent = e.text; e.dirty = false; if (e.feed) { e.feed.text = e.text.replace(/^Opponent /, ''); renderFeed(); } }
  el.scrollTop = el.scrollHeight;
}

// ---------------------------------------------------------------- highlights: options mapped onto the board
// H.hand: name -> option indices (hand options are per card name), H.gy / H.ex
// likewise for graveyard and exile; H.perm: oid -> indices (abilities);
// H.target: key -> index ('o<oid>', 'p<player>', 's<sid>'); H.pay: oid -> indices.
function highlights(d, s) {
  const H = {hand: new Map(), gy: new Map(), ex: new Map(), perm: new Map(), target: new Map(), pay: new Map(), loose: []};
  if (!d) return H;
  const add = (m, k, i) => { if (!m.has(k)) m.set(k, []); m.get(k).push(i); };
  const twins = oid => {  // permanents the engine treats as interchangeable with `oid`
    const c = s.battlefield.find(x => x.oid === oid); if (!c) return [oid];
    const key = x => [x.name, x.controller, !!x.tapped, !!x.sick, x.damage || 0, x.counters || 0, x.attached_to ?? -1, !!x.attacking, x.blocking ?? -1].join('|');
    return [...new Set(s.battlefield.filter(x => key(x) === key(c) && !s.battlefield.some(y => y.attached_to === x.oid)).map(x => x.oid).concat(oid))];
  };
  d.refs.forEach((r, i) => {
    if (d.kind === 'priority') {
      if (r.type === 'pass') return;
      if (r.zone === 'hand' || (r.from === 'hand' && !r.oid)) add(H.hand, r.name, i);
      else if (r.zone === 'graveyard' || r.from === 'graveyard') add(H.gy, r.name, i);
      else if (r.zone === 'exile' || r.from === 'exile') add(H.ex, r.name, i);
      else if (r.oid != null) for (const o of twins(r.oid)) add(H.perm, o, i);
      else H.loose.push(i);
    } else if (d.kind === 'target') {
      if (r.oid != null) for (const o of twins(r.oid)) { if (!H.target.has('o' + o)) H.target.set('o' + o, i); }
      else if (r.player != null) H.target.set('p' + r.player, i);
      else if (r.sid != null) H.target.set('s' + r.sid, i);
      else H.loose.push(i);
    } else if (d.kind === 'pay_mana') {
      if (r.oid != null) for (const o of twins(r.oid)) add(H.pay, o, i);
      else H.loose.push(i);
    }
  });
  return H;
}

// ---------------------------------------------------------------- render
let H = highlights(null);
// Transient feedback (life deltas, damage numbers, flashes, entries) is
// recorded once, when a frame is first shown, with its start time, and
// re-applied by every render until it expires; animations resume where they
// were (negative animation-delay) instead of restarting or vanishing.
const FX_MS = {life: 1400, dmg: 900, flash: 450, enter: 600};
function recordFx(s, prev) {
  if (!prev) return;
  const now = performance.now();
  s.players.forEach((P, p) => { const d = P.life - prev.players[p].life; if (d) S.fx.life[p] = {n: d, t: now}; });
  const before = new Map(prev.battlefield.map(c => [c.oid, c]));
  for (const c of s.battlefield) {
    const pc = before.get(c.oid);
    if (!pc) S.fx.enter[c.uid] = {t: now};
    else if ((c.damage || 0) > (pc.damage || 0)) { S.fx.dmg[c.oid] = {n: c.damage - (pc.damage || 0), t: now}; S.fx.flash[c.oid] = {t: now}; }
  }
}
function fx(kind, key) {  // the live effect, or null; with `ago` (ms since it started)
  const e = S.fx[kind][key];
  if (!e) return null;
  const ago = performance.now() - e.t;
  if (ago > FX_MS[kind]) { delete S.fx[kind][key]; return null; }
  return {...e, ago};
}
const fxDelay = e => ` style="animation-delay:-${Math.round(e.ago)}ms"`;

function render(i, o = {}) {
  const f = S.raw[i], s = f.state;
  const before = o.noAnim ? null : rects();
  const prev = S.prev;
  if (i !== S.shown) { recordFx(s, prev); S.prev = s; }
  S.shown = i;
  const d = S.ui ? S.ui.d : null;
  H = highlights(d, s);
  for (const p of [0, 1]) { renderPlate(p, s, prev); renderField(p, s, prev); }
  renderHand(s);
  renderMid(s);
  renderStack(s);
  fitRows();
  fitHand();
  if (before) flip(before, prev);
  renderPaying();
  if (S.paying) showTaps(payingSuggestion());
  requestAnimationFrame(drawArrows);
  scheduleFxCleanup();
  markFocusable();
}
let fxTimer = null;
function scheduleFxCleanup() {
  clearTimeout(fxTimer);
  const now = performance.now();
  let next = Infinity;
  for (const kind of Object.keys(S.fx)) {
    for (const [k, e] of Object.entries(S.fx[kind])) {
      const left = e.t + FX_MS[kind] - now;
      if (left < -50) delete S.fx[kind][k]; else next = Math.min(next, left);  // expired: drop (its element may be gone)
    }
  }
  if (next < Infinity) fxTimer = setTimeout(expireFx, Math.max(30, next + 20));
}
// Remove expired effects from the DOM in place: re-rendering the board here
// would drop a hovered card's lift, an open menu's anchor or a drag's target.
function expireFx() {
  for (const p of [0, 1]) if (!fx('life', p)) {
    const el = $(`#plate${p === S.seat ? 0 : 1} .life`);
    if (el) { el.classList.remove('hit', 'heal'); el.querySelector('.delta')?.remove(); }
  }
  for (const el of $$('#board .perm')) {
    const oid = +el.dataset.oid, uid = el.querySelector(':scope > .card')?.dataset.uid;
    if (!fx('dmg', oid)) el.querySelector('.b.dmg')?.remove();
    if (!fx('flash', oid)) el.classList.remove('flash');
    if (uid != null && !fx('enter', +uid)) el.classList.remove('entered');
  }
  scheduleFxCleanup();
}

function rects() {
  const m = new Map();
  for (const el of $$('#board [data-uid]')) { const r = el.getBoundingClientRect(); if (r.width) m.set(el.dataset.uid, r); }
  const oh = $('#ohand').getBoundingClientRect(), pl = $('#plate0').getBoundingClientRect();
  m.set('ohand', oh); m.set('mylib', pl);
  return m;
}
function flip(before, prev) {
  if (beatScale() === 0 && S.queue.length) return;
  const prevUids = prev ? new Set([...prev.battlefield.map(c => String(c.uid)), ...prev.players.flatMap(p => p.hand.map(c => String(c.uid)))]) : new Set();
  for (const el of $$('#board [data-uid]')) {
    const uid = el.dataset.uid;
    let from = before.get(uid);
    if (!from && prev && !prevUids.has(uid)) {
      const inHand = el.closest('#hand');
      const perm = el.closest('.perm');
      if (perm && perm.closest('#side1')) from = before.get('ohand');
      else if (inHand) from = before.get('mylib');
    }
    if (!from) continue;
    const to = el.getBoundingClientRect();
    const dx = from.left + from.width / 2 - (to.left + to.width / 2), dy = from.top + from.height / 2 - (to.top + to.height / 2);
    if (Math.abs(dx) < 2 && Math.abs(dy) < 2) continue;
    el.animate([{translate: `${dx}px ${dy}px`}, {translate: '0 0'}], {duration: 320, easing: 'cubic-bezier(.2,.8,.2,1)'});
  }
}

function renderMeta() {
  const m = S.meta; if (!m) return;
  const me = S.seat, op = opp();
  const mm = m.match;
  const score = mm ? ` · Game ${mm.game_no} of 3 · ${mm.wins[me]}–${mm.wins[op]}` : '';
  $('#meta').innerHTML = `<b>${esc(m.decks[me])}</b> (you) vs <b>${esc(m.decks[op])}</b> (${esc(agentName(m.agents[op]))})${score}`;
  $('#bConcede').disabled = !!S.over;
  $('#bReviewMenu').disabled = !S.over;
}

function renderPlate(p, s, prev) {
  const P = s.players[p], el = $(`#plate${p === S.seat ? 0 : 1}`);
  const isMe = p === S.seat;
  const pv = prev?.players[p];
  const lf = fx('life', p), delta = lf ? lf.n : 0;
  void pv;
  const pool = Object.entries(P.pool || {}).flatMap(([c, n]) => Array(n).fill(`<i class="${esc(c)}">${esc(c)}</i>`)).join('')
    + (isMe ? S.reserved.map(r => `<i class="${esc(r.color)} res" data-res="${r.oid}" title="${esc(r.color)} from ${esc(r.name)} (click the land to untap it)">${esc(r.color)}</i>`).join('') : '');
  const d = S.ui?.d;
  const tgt = H.target.has('p' + p) || payTargetKeys().has('p' + p);
  const preview = isMe ? blockPreviewLife(s) : attackPreviewLife(s);
  const gyCast = !isMe ? '' : (H.gy.size ? 'castable' : ''), exCast = !isMe ? '' : (H.ex.size ? 'castable' : '');
  el.className = `plate ${s.active === p ? 'active' : ''} ${tgt ? 'targetable' : ''}`;
  el.dataset.player = p;
  el.innerHTML = `
    <div class="who"><span class="dot"></span>${isMe ? 'You' : esc(agentName(S.meta?.agents[p]))}</div>
    <div class="deck"><button class="decklink" data-decks="${p}" title="See the decklist">${esc(S.meta?.decks[p] || '')}</button></div>
    <div class="playdraw">${S.meta && S.meta.starting_player === p ? 'on the play' : 'on the draw'}</div>
    <div class="life ${P.life <= 5 ? 'low' : ''} ${delta < 0 ? 'hit' : delta > 0 ? 'heal' : ''}"${lf ? fxDelay(lf) : ''}>${P.life}${delta ? `<span class="delta ${delta < 0 ? 'neg' : 'pos'}"${fxDelay(lf)}>${delta > 0 ? '+' : ''}${delta}</span>` : ''}</div>
    ${preview != null ? (preview <= 0 ? `<div class="lethal" title="This combat would be lethal if nothing changes">LETHAL</div>` : `<div class="preview-life">→ ${preview} after combat</div>`) : ''}
    <div class="zones">
      <div class="zone" title="Cards in hand">Hand <b>${P.hand.length}</b></div>
      <div class="zone" title="Cards in library${P.library_top_known.length ? '; known on top: ' + esc(P.library_top_known.join(', ')) : ''}">Library <b>${P.library}</b></div>
      <div class="zone clickable ${gyCast}" data-zone="graveyard" data-p="${p}" title="Click to see the graveyard">Grave <b>${P.graveyard.length}</b></div>
      <div class="zone clickable ${exCast}" data-zone="exile" data-p="${p}" title="Click to see exiled cards">Exile <b>${P.exile.length}</b></div>
    </div>
    <div class="openmana" title="Untapped mana sources: lands, and creatures and artifacts that tap for mana">${openMana(s, p)} open mana</div>
    ${pool ? `<div class="mana" title="Your mana pool: spent first by your next payment">${pool}<span class="empties">empties end of ${esc((STEP_LABEL[s.step] || s.step).toLowerCase())}</span></div>` : ''}
    ${isMe ? '' : `<div class="lastact" title="The opponent's latest action">${lastActHtml()}</div>`}`;
  void d;
}

// Untapped sources that can make mana now: lands; creatures (not summoning
// sick) and artifacts whose text has "{T}: Add".
function openMana(s, p) {
  return s.battlefield.filter(c => {
    if (c.controller !== p || c.tapped || isReserved(c.oid)) return false;
    if (c.types.includes('Land')) return true;
    const text = S.cards[c.name]?.text || '';
    if (!/\{T\}(, [^:]*)?: Add/.test(text)) return false;
    return !(c.power != null && c.sick);
  }).length;
}

function groupPerms(perms, s) {
  const hosts = new Set(s.battlefield.filter(c => c.attached_to != null).map(c => c.attached_to));
  // While declaring attackers or blockers each creature is its own card.
  const combat = S.ui && (S.ui.kind === 'declare_attacker' || S.ui.kind === 'declare_blocker');
  const out = [], by = new Map();
  for (const c of perms) {
    const creature = c.power != null;
    if ((creature && combat) || c.attacking || c.blocking != null || c.damage || c.counters || hosts.has(c.oid)) { out.push([c]); continue; }
    const k = [c.name, !!c.tapped, !!c.sick, c.power, c.toughness, isReserved(c.oid), payingHeld(c.oid)].join('|');
    if (by.has(k)) by.get(k).push(c); else { const g = [c]; by.set(k, g); out.push(g); }
  }
  return out;
}

function renderField(p, s, prev) {
  const top = p !== S.seat, k = top ? 1 : 0;
  const mine = s.battlefield.filter(c => c.controller === p);
  const live = new Set(s.battlefield.map(c => c.oid));
  const attachedTo = new Map();
  for (const c of s.battlefield) if (c.attached_to != null && live.has(c.attached_to)) {
    if (!attachedTo.has(c.attached_to)) attachedTo.set(c.attached_to, []);
    attachedTo.get(c.attached_to).push(c);
  }
  const free = mine.filter(c => c.attached_to == null || !live.has(c.attached_to));
  const front = free.filter(c => c.power != null);
  const back = free.filter(c => c.power == null);
  back.sort((a, b) => (a.types.includes('Land') ? 1 : 0) - (b.types.includes('Land') ? 1 : 0));
  const prevBy = new Map((prev?.battlefield || []).map(c => [c.oid, c]));
  $(`#crea${k}`).innerHTML = groupPerms(front, s).map(g => permHtml(g, s, prevBy, attachedTo)).join('');
  $(`#lands${k}`).innerHTML = groupPerms(back, s).map(g => permHtml(g, s, prevBy, attachedTo)).join('');
  const oh = $('#ohand');
  if (top) oh.innerHTML = s.players[p].hand.map(c => c.hidden ? backHtml() : cardHtml(c.name, {uid: c.uid})).join('');
}

function permHtml(g, s, prevBy, attachedTo) {
  const c = g[0], n = g.length, oids = g.map(x => x.oid);
  const ui = S.ui, cls = ['perm'];
  const pc = prevBy.get(c.oid);
  const res = S.reserved.filter(r => oids.includes(r.oid)).concat((S.paying?.pips || []).filter(p => p.by && p.by.oid != null && !p.by.res && oids.includes(p.by.oid)).map(p => p.by));
  if (c.tapped || res.length) cls.push('tapped');
  if (res.length) cls.push('reserved');
  if (res.length && oids.includes(S.justTapped)) { cls.push('justtapped'); S.justTapped = null; }
  if (!res.length && !S.paying && ui?.kind === 'priority' && oids.some(o => canTapForMana(s.battlefield.find(x => x.oid === o)))) cls.push('manasrc');
  if (c.attacking) cls.push('attacking');
  if (c.blocking != null) cls.push('blocking');
  if (n > 1) cls.push('grouped');
  const fEnter = fx('enter', c.uid), fFlash = fx('flash', c.oid), fDmg = fx('dmg', c.oid);
  if (fEnter) cls.push('entered');
  if (fFlash) cls.push('flash');
  void pc;
  if (oids.some(o => H.perm.has(o))) cls.push('activatable');
  if (oids.some(o => H.target.has('o' + o)) || (S.paying?.step === 'target' && oids.some(o => payTargetKeys().has('o' + o)))) cls.push('targetable');
  if (S.paying?.step === 'pay' && !res.length && c.controller === S.seat && oids.some(o => canTapForMana(s.battlefield.find(x => x.oid === o)) && !payingHeld(o))) cls.push('paysrc');
  if (oids.some(o => H.pay.has(o))) cls.push('payable');
  let b = '';
  if (ui?.kind === 'declare_attacker' && c.controller === S.seat) {
    if (ui.sel.has(c.oid)) cls.push('attack-sel'); else if (ui.eligible.has(c.oid)) cls.push('can-attack');
  }
  if (ui?.kind === 'declare_blocker') {
    if (c.controller === S.seat && ui.blockers.has(c.oid)) cls.push(ui.blockSel === c.oid ? 'block-sel' : ui.blocks.has(c.oid) ? 'blocked-by' : 'can-block');
    if (c.attacking && c.controller !== S.seat) cls.push('attacker-target');
  }
  const preview = ui?.kind === 'declare_attacker' || ui?.kind === 'declare_blocker' ? combatPreview(s) : {dies: new Set()};
  if (n > 1) b += `<span class="b cnt">×${n}</span>`;
  if (res.length) b += `<span class="b pool">${res.map(r => `<i class="${esc(r.color)}">${esc(r.color)}</i>`).join('')}</span>`;
  if (c.power != null) {
    const base = S.cards[c.name];
    const buff = base && base.power != null && (c.power > base.power || c.toughness > base.toughness);
    b += `<span class="b pt ${c.damage ? 'hurt' : buff ? 'buff' : ''}">${c.power}/${c.toughness - (c.damage || 0)}</span>`;
  }
  if (c.counters) b += `<span class="b ctr">+${c.counters}</span>`;
  if (c.sick && c.power != null) b += `<span class="b sick" title="Summoning sick">zZ</span>`;
  if (c.attacking || (ui?.kind === 'declare_attacker' && ui.sel.has(c.oid))) b += `<span class="b sword" title="Attacking">⚔</span>`;
  else if (c.blocking != null || (ui?.kind === 'declare_blocker' && ui.blocks.has(c.oid))) b += `<span class="b shield" title="Blocking">⛨</span>`;
  if (preview.dies.has(c.oid)) b += `<span class="b skull" title="Dies in this combat (if nothing changes)">☠</span>`;
  if (fDmg) b += `<span class="b dmg"${fxDelay(fDmg)}>-${fDmg.n}</span>`;
  const att = (attachedTo.get(c.oid) || []).map(a => `<div class="attached" data-oid="${a.oid}">${cardHtml(a.name, {uid: a.uid})}</div>`).join('');
  return `<div class="${cls.join(' ')}" data-oid="${c.oid}" data-oids="${oids.join(' ')}" data-name="${esc(c.name)}" data-ctl="${c.controller}">${att}${cardHtml(c.name, {uid: c.uid})}<div class="badges">${b}</div></div>`;
}

function renderHand(s) {
  const P = s.players[S.seat];
  const el = $('#hand');
  const red = myDecision()?.reductions || S.reductions || {};
  S.reductions = red;
  el.innerHTML = P.hand.map(c => {
    const opts = H.hand.get(c.name);
    const n = red[c.uid];
    const badge = n ? `<div class="badges"><span class="b reduced" title="This costs ${n} less right now">−${n} cost</span></div>` : '';
    return cardHtml(c.name, {uid: c.uid, cls: (opts ? 'playable' : '') + (S.paying && S.paying.uid === c.uid ? ' casting' : ''), attrs: ' data-hand="1"', badges: badge});
  }).join('');
}

function fitHand() {
  const el = $('#hand'), cards = [...el.children], n = cards.length;
  if (!n) return;
  const W = el.clientWidth, w = cards[0].offsetWidth;
  const step = n > 1 ? Math.min(w * 0.82, (W - w - 20) / (n - 1)) : 0;
  const total = w + step * (n - 1), x0 = Math.max(10, (W - total) / 2);
  const mid = (n - 1) / 2, rot = Math.min(4, 26 / n);
  cards.forEach((c, i) => {
    c.style.left = `${x0 + i * step}px`;
    c.style.zIndex = 10 + i;
    c.style.setProperty('--rot', `${(i - mid) * rot}deg`);
    c.style.setProperty('--ty', `${Math.pow(Math.abs(i - mid), 2) * 1.6}px`);
  });
}

// Card size per row from the space it has: as big as fits (up to a cap),
// in one to three lines, never overlapping. Rows split the field's height,
// creatures getting the larger share.
const AR = 1.3934, GAP = 8;
function fitRows() {
  const u = Math.min(innerHeight * 0.01, innerWidth * 0.006);
  for (const k of [0, 1]) {
    const field = $('#field' + k), front = $('#crea' + k), back = $('#lands' + k);
    front.style.height = back.style.height = '0px';  // the track decides the field's size, not last layout's rows
    const cs = getComputedStyle(field);
    const H = field.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom) - GAP;
    const W = field.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    const nf = front.children.length, nb = back.children.length;
    const hf = nf ? H * (nb ? 0.6 : 0.8) : H * 0.25, hb = H - hf;
    layoutRow(front, Math.min(W, front.clientWidth), hf, u * 18);
    layoutRow(back, Math.min(W, back.clientWidth), hb, u * 11.5);
  }
}
function layoutRow(row, W, h, maxW) {
  row.style.height = `${Math.max(0, h)}px`;
  const kids = [...row.children];
  if (!kids.length) return;
  const unit = el => el.classList.contains('tapped') ? AR : 1;
  const units = kids.reduce((a, el) => a + unit(el), 0);
  let best = {w: 0, L: 1};
  for (let L = 1; L <= 3; L++) {
    const perLine = units / L + (L > 1 ? 0.6 : 0);
    const wH = (h - (L - 1) * GAP) / L / AR;
    const wW = (W - (Math.ceil(kids.length / L) - 1) * GAP) / perLine;
    const w = Math.min(wH, wW, maxW);
    if (w > best.w * 1.1) best = {w, L};
  }
  let w = Math.max(24, best.w);
  row.classList.toggle('wrap', best.L > 1);
  row.style.setProperty('--rw', `${w}px`);
  // Measured, not guessed; offsets ignore transforms (hover, entry and attack animations).
  const extent = () => {
    let bottom = 0, right = 0;
    for (const el of kids) { bottom = Math.max(bottom, el.offsetTop + el.offsetHeight); right = Math.max(right, el.offsetLeft + el.offsetWidth); }
    return [bottom - row.offsetTop, right - row.offsetLeft];
  };
  for (let i = 0; i < 8 && w > 24; i++) {
    const [eh, ew] = extent();
    if (eh <= h + 1 && ew <= W + 1) break;
    w *= 0.92;
    row.style.setProperty('--rw', `${w}px`);
  }
}

function renderMid(s) {
  const mine = s.active === S.seat;
  $('#turn').innerHTML = `<b class="${mine ? 'me' : 'opp'}">${mine ? 'Your turn' : "Opponent's turn"}</b><span style="color:var(--muted)">Turn ${s.turn} · ${esc(STEP_LABEL[s.step] || s.step)}</span>`;
  const who = mine ? 'me' : 'opp';
  $('#rail').innerHTML = STEPS.map(st => `<div class="step ${st === s.step ? 'cur ' + who : ''}" title="${esc(STEP_LABEL[st])}: toggle where auto-pass stops (gold: your turn, blue: opponent's turn)">
    <span>${esc(STEP_LABEL[st])}</span><span class="stops"><button class="stop me ${PREF.stops.me[st] ? 'on' : ''}" data-stop="me:${st}" title="Stop here on your turn"></button><button class="stop opp ${PREF.stops.opp[st] ? 'on' : ''}" data-stop="opp:${st}" title="Stop here on the opponent's turn"></button></span></div>`).join('');
}

let stackOpen = false;
function renderStack(s) {
  const el = $('#stack');
  $('#board').classList.toggle('has-stack', s.stack.length > 0);
  if (!s.stack.length) { el.innerHTML = ''; stackOpen = false; return; }
  const all = [...s.stack].reverse(), SHOW = 3;
  // Targets below the fold stay reachable: an item that is a legal target is never hidden.
  const items = stackOpen || all.length <= SHOW + 1 ? all : all.filter((it, j) => j < SHOW || H.target.has('s' + it.sid));
  const more = all.length - items.length;
  el.innerHTML = `<div class="sh">Stack · ${all.length} · top first</div>` + items.map((it, j) => {
    const tgt = H.target.has('s' + it.sid);
    const name = it.card || it.name;
    const src = it.card || it.name.split(':')[0];
    const yieldBtn = it.controller !== S.seat ? `<button class="yield ${S.yields.has(src) ? 'on' : ''}" data-yield="${esc(src)}" title="${S.yields.has(src) ? 'Stop again for' : "Don't stop for"} ${esc(src)} this game">${S.yields.has(src) ? 'auto-passing ✓' : 'auto-pass'}</button>` : '';
    return `<div class="sitem ${it.controller === S.seat ? 'c-me' : 'c-opp'} ${j === 0 ? 'top' : ''} ${tgt ? 'targetable' : ''}" data-sid="${it.sid}" data-name="${esc(name)}">${yieldBtn}
      ${cardHtml(name)}<div class="st"><b>${esc(it.name)}</b>${it.controller === S.seat ? 'yours' : 'opponent'}${it.x ? ` · X=${it.x}` : ''}${it.targets.length ? `<div class="tg">→ ${esc(it.targets.map(t => clean(t.replace(/^player (\d)/, (_, p) => +p === S.seat ? 'you' : 'opponent'))).join(', '))}</div>` : ''}</div></div>`;
  }).join('') + (more > 0 ? `<button class="smore" data-cmd="stack-more">+${more} more below</button>` : stackOpen && all.length > SHOW + 1 ? '<button class="smore" data-cmd="stack-more">Show fewer</button>' : '');
}

// ---------------------------------------------------------------- combat helpers
const kw = (c, k) => (c.keywords || []).includes(k);
function canBlock(b, a) {
  if (kw(a, 'unblockable')) return false;
  if (kw(a, 'flying') && !(kw(b, 'flying') || kw(b, 'reach'))) return false;
  return true;
}
// A rough outcome preview (a hint, not the rules): who dies, damage through.
function combatPreview(s) {
  const out = {dies: new Set(), through: 0, to: null};
  const ui = S.ui;
  if (ui?.kind === 'declare_attacker') {
    for (const o of ui.sel) { const c = s.battlefield.find(x => x.oid === o); if (c) out.through += Math.max(0, c.power); }
    out.to = opp();
    return out;
  }
  const attackers = s.battlefield.filter(c => c.attacking);
  if (!attackers.length) return out;
  const blocks = ui?.kind === 'declare_blocker' ? ui.blocks : new Map(s.battlefield.filter(c => c.blocking != null).map(c => [c.oid, c.blocking]));
  for (const a of attackers) {
    const bs = [...blocks].filter(([, at]) => at === a.oid).map(([b]) => s.battlefield.find(x => x.oid === b)).filter(Boolean);
    if (!bs.length) { out.through += Math.max(0, a.power); continue; }
    if (bs.reduce((t, b) => t + Math.max(0, b.power), 0) >= a.toughness - (a.damage || 0) || bs.some(b => kw(b, 'deathtouch') && b.power > 0)) out.dies.add(a.oid);
    let left = a.power;
    for (const b of bs) {
      const need = kw(a, 'deathtouch') ? 1 : b.toughness - (b.damage || 0);
      if (left >= need && left > 0) { out.dies.add(b.oid); left -= need; } else left = 0;
    }
    if (kw(a, 'trample')) out.through += Math.max(0, left);
  }
  out.to = s.active === S.seat ? opp() : S.seat;
  return out;
}
function blockPreviewLife(s) {
  if (S.ui?.kind !== 'declare_blocker') return null;
  const p = combatPreview(s);
  return p.through ? s.players[S.seat].life - p.through : null;
}
function attackPreviewLife(s) {
  const ui = S.ui;
  if (ui?.kind !== 'declare_attacker' || !ui.sel.size) return null;
  const through = combatPreview(s).through;
  return through ? s.players[opp()].life - through : null;
}

// ---------------------------------------------------------------- arrows
function edgePoint(r, toward) {  // where the line from r's centre to `toward` leaves r
  const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
  const dx = toward[0] - cx, dy = toward[1] - cy;
  if (!dx && !dy) return [cx, cy];
  const sx = dx ? (r.width / 2) / Math.abs(dx) : Infinity, sy = dy ? (r.height / 2) / Math.abs(dy) : Infinity;
  const k = Math.min(sx, sy, 1);
  return [cx + dx * k, cy + dy * k];
}
function arrow(fromEl, toElOrPt, cls, marker) {
  const box = $('#board').getBoundingClientRect();
  const a = fromEl.getBoundingClientRect();
  const b = Array.isArray(toElOrPt) ? {left: toElOrPt[0], top: toElOrPt[1], width: 0, height: 0} : toElOrPt.getBoundingClientRect();
  if (!a.width || (!Array.isArray(toElOrPt) && !b.width)) return '';
  const bc = [b.left + b.width / 2, b.top + b.height / 2], ac = [a.left + a.width / 2, a.top + a.height / 2];
  const p1 = edgePoint(a, bc), p2 = Array.isArray(toElOrPt) ? bc : edgePoint(b, ac);
  const [x1, y1] = [p1[0] - box.left, p1[1] - box.top], [x2, y2] = [p2[0] - box.left, p2[1] - box.top];
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, len = Math.hypot(x2 - x1, y2 - y1);
  const nx = -(y2 - y1) / (len || 1), ny = (x2 - x1) / (len || 1), bend = Math.min(60, len * 0.18);
  return `<path class="${cls}" d="M${x1},${y1} Q${mx + nx * bend},${my + ny * bend} ${x2},${y2}" marker-end="url(#${marker})"/>`;
}
const permEl = oid => document.querySelector(`#board .perm[data-oids~="${oid}"]`) || document.querySelector(`#board .attached[data-oid="${oid}"]`);
let pointer = [0, 0];
// What a target is being chosen for: my newest stack item, else the permanent I just used.
function targetSource(s) {
  const top = s.stack[s.stack.length - 1];
  if (top && top.controller === S.seat) return document.querySelector(`#stack .sitem[data-sid="${top.sid}"] .card`);
  return S.srcOid != null ? permEl(S.srcOid) : null;
}
function drawArrows() {
  if (S.shown < 0) return;
  const s = stateAt(S.shown), ui = S.ui;
  let h = '';
  // spells on the stack -> their targets
  for (const it of s.stack) {
    const from = document.querySelector(`#stack .sitem[data-sid="${it.sid}"] .card`);
    if (!from) continue;
    for (const o of it.target_oids || []) { const t = permEl(o); if (t) h += arrow(from, t, 'a-gold', 'mGold'); }
    for (const t of it.targets || []) { const m = /^player (\d)/.exec(t); if (m) h += arrow(from, $(`#plate${+m[1] === S.seat ? 0 : 1}`), 'a-gold', 'mGold'); }
  }
  // combat
  const blocks = ui?.kind === 'declare_blocker' ? ui.blocks : new Map(s.battlefield.filter(c => c.blocking != null).map(c => [c.oid, c.blocking]));
  const blocked = new Set(blocks.values());
  const attackers = ui?.kind === 'declare_attacker' ? [...ui.sel] : s.battlefield.filter(c => c.attacking).map(c => c.oid);
  const free = attackers.filter(o => !blocked.has(o)).map(o => [s.battlefield.find(x => x.oid === o), permEl(o)]).filter(([c, el]) => c && el);
  if (free.length > 4) {
    // many attackers: one bundled arrow (with the count) from the middle of the group; hover one to see its own
    const els = free.map(([, el]) => el.getBoundingClientRect()), box = $('#board').getBoundingClientRect();
    const cx = els.reduce((a, r) => a + r.left + r.width / 2, 0) / els.length, cy = els.reduce((a, r) => a + r.top + r.height / 2, 0) / els.length;
    const plate = $(`#plate${free[0][0].controller === S.seat ? 1 : 0}`);
    h += arrow({getBoundingClientRect: () => ({left: cx - 2, top: cy - 2, width: 4, height: 4})}, plate, 'a-red', 'mRed');
    h += `<text class="a-count" x="${cx - box.left}" y="${cy - box.top - 14}">⚔ ${free.length}</text>`;
    const hov = document.querySelector('#board .perm.hovered');
    for (const [c, el] of free) if (el === hov) h += arrow(el, plate, 'a-red', 'mRed');
  } else {
    for (const [c, el] of free) h += arrow(el, $(`#plate${c.controller === S.seat ? 1 : 0}`), 'a-red', 'mRed');
  }
  for (const [b, a] of blocks) { const be = permEl(b), ae = permEl(a); if (be && ae) h += arrow(be, ae, 'a-blue', 'mBlue'); }
  // a gesture in progress: block drag, or targeting from the source
  if (drag?.started && drag.kind === 'block') h += arrow(drag.el, pointer, 'a-blue', 'mBlue');
  if (ui?.kind === 'target' && !drag) {
    const src = targetSource(s);
    if (src) h += arrow(src, pointer, 'a-cyan', 'mCyan');
  }
  $('#arrowg').innerHTML = h;
}
window.addEventListener('resize', () => { if (S.shown >= 0) { fitRows(); fitHand(); drawArrows(); } });
// Zoom, panel changes and anything else that resizes the board re-fit it too.
let lastBoard = '';
new ResizeObserver(() => {
  const r = $('#board').getBoundingClientRect(), key = `${Math.round(r.width)}x${Math.round(r.height)}`;
  if (key === lastBoard || S.shown < 0) return;
  lastBoard = key; fitRows(); fitHand(); drawArrows();
}).observe($('#board'));

// ---------------------------------------------------------------- the decision in front of the player
// Input is stamped with the decision on screen. A decision that differs in
// kind or turn from the previous one locks keys and the primary button for
// a moment, so a press meant for the last decision (or a held key) cannot
// answer the new one after the engine ran ahead.
const INPUT_LOCK_MS = 350;
let shownKey = '', lockUntil = 0;
const inputLocked = () => performance.now() < lockUntil;
function enterDecision(fresh) {
  const d = myDecision();
  if (d) {
    const st = stateAt(last()), key = `${d.kind}|${st.turn}`;
    if (key !== shownKey) { lockUntil = performance.now() + INPUT_LOCK_MS; setTimeout(renderDock, INPUT_LOCK_MS + 10); }
    shownKey = key;
  }
  if (!d) { S.ui = null; if (S.shown !== last() && S.raw.length) render(last()); renderDock(); return; }
  const fi = last(), s = stateAt(fi);
  S.lastAutoPassed = S.autoPassed; S.autoPassed = 0;
  // what stopped here (for the docs' numbers and tests): step, whose turn, whether a timely play existed
  S.stopLog.push({fi, kind: d.kind, step: s.step, mine: s.active === S.seat, timely: d.kind === 'priority' ? d.refs.some((r, i) => timely(r, d.options[i])) : null, top: s.stack.at(-1)?.controller ?? null});
  const ui = {kind: d.kind, d, fi, sel: new Set(), blocks: new Map(), blockSel: null, armed: null, eligible: new Set(), blockers: new Set(), source: S.ui?.source};
  if (d.kind === 'declare_attacker') {
    const names = new Set(d.refs.filter(r => r.attacker != null).map(r => r.name));
    for (const c of s.battlefield) if (c.controller === S.seat && c.power != null && !c.tapped && !c.sick && !c.attacking && names.has(c.name)) ui.eligible.add(c.oid);
    for (const r of d.refs) if (r.attacker != null) ui.eligible.add(r.attacker);
  }
  if (d.kind === 'declare_blocker') {
    for (const c of s.battlefield) if (c.controller === S.seat && c.power != null && !c.tapped) ui.blockers.add(c.oid);
  }
  if (d.kind === 'assign_damage') ui.dmg = defaultSplit(d, s);
  if (d.kind !== 'target') ui.source = null;
  if (S.paying && (d.kind !== 'priority' || fi !== S.paying.fi)) S.paying = null;
  S.ui = ui;
  render(fi, {noAnim: S.shown === fi});
  renderDock();
  renderOverlay();
  if (fresh) ping();
  if (d.kind === 'priority') coach();
}

function passLabel(s) {
  const top = s.stack[s.stack.length - 1];
  if (top) return `Resolve ${top.name.split(':')[0]}`;  // "Resolve Guttersnipe", not the whole trigger text
  const mine = s.active === S.seat;
  if (mine && s.step === 'main2') return 'End turn';
  if (s.step === 'end' || s.step === 'cleanup') return mine ? "Pass → Opponent's turn" : 'Pass → Your turn';
  return `Pass → ${NEXT[s.step] || 'next step'}`;
}

const HINTS = {
  priority: '<kbd>R</kbd> pass till they act · <kbd>H</kbd> hold · <kbd>F</kbd> full control',
  declare_attacker: 'undo: <kbd>right-click</kbd> · <kbd>Esc</kbd> clears',
  declare_blocker: 'undo: <kbd>right-click</kbd> · <kbd>Esc</kbd> clears',
  target: 'the spell is being cast: pick a target',
  pay_mana: '<kbd>Space</kbd> pays automatically',
};
function modeHint(d, ui) {
  if (!d || !ui || S.busy || S.pumping) return S.queue.length ? '<kbd>click</kbd> or <kbd>Space</kbd> skips the replay' : '';
  const auto = S.lastAutoPassed ? `<span title="Priority passes the client answered for you since your last decision">auto-passed ${S.lastAutoPassed}</span> · ` : '';
  return auto + (HINTS[d.kind] || '<kbd>O</kbd> lists every option');
}

function renderDock() {
  const d = myDecision(), ui = S.ui, P = $('#primary'), pr = $('#prompt'), ch = $('#choices');
  const pill = $('#pill');
  const yields = S.yields.size ? ` · not stopping for ${[...S.yields].join(', ')}` : '';
  pill.className = S.passMode || S.fullControl || S.holdOnce || S.yields.size ? 'on' : '';
  pill.textContent = S.fullControl ? 'Full control: no auto-pass · F' : S.passMode === 'opp' ? 'Passing until the opponent acts · Esc'
    : S.holdOnce ? 'Holding priority after your next spell · H' : yields ? yields.slice(3) : '';
  pill.title = 'Esc ends passing until they act; the auto-pass button on a stack item toggles yielding to that source';
  P.className = 'primary'; P.disabled = true; ch.innerHTML = ''; pr.className = '';
  $('#modehint').innerHTML = modeHint(d, ui);
  $('#bAll').style.visibility = d && ui ? 'visible' : 'hidden';
  if (S.error) {
    const kind = S.errorKind || '';
    const auth = kind === 'auth', lost = kind === 'unrecoverable';
    const gone = lost || kind === 'gone' || /no such game/i.test(S.error), down = /failed to fetch|networkerror|load failed/i.test(S.error);
    const msg = auth ? 'Your sign-in expired. Reload the page to sign in again; the game is saved on the server.'
      : lost ? S.error
      : gone ? (/expired/i.test(S.error) ? S.error : 'This game is no longer on the server (it expired or belongs to another account).')
      : down ? 'Lost the connection to the game server.' : S.error;
    pr.innerHTML = `<span class="k">${auth ? 'Signed out' : lost ? 'Cannot resume' : gone ? 'Game ended' : 'Problem'}</span><span class="err">${esc(msg)}</span>`;
    ch.innerHTML = gone || auth ? '' : `<button class="choice" data-resync="1">Reload the game from the server</button>`;
    if (auth) { P.textContent = 'Reload'; P.dataset.act = 'reload'; }
    else if (gone) { P.textContent = 'New game'; P.dataset.act = 'new'; } else { P.textContent = down ? 'Reconnect' : 'Retry'; P.dataset.act = 'resync'; }
    P.disabled = false;
    return;
  }
  if (S.over) { pr.innerHTML = '<span class="k">Game over</span>' + esc(resultText()); P.textContent = 'New game'; P.disabled = false; P.dataset.act = 'new'; return; }
  if (!d || !ui || S.busy || S.pumping) {
    pr.innerHTML = `<span class="k">${S.queue.length ? 'Opponent acting' : 'Waiting'}</span>${S.queue.length ? 'Click or press Space to skip ahead' : 'The opponent is thinking…'}`;
    P.textContent = S.queue.length ? 'Opponent acting…' : 'Waiting…';
    P.dataset.act = '';
    return;
  }
  pr.className = 'mine';
  if (inputLocked()) P.classList.add('locked');
  const s = stateAt(ui.fi);
  const choiceBtns = (idxs, cls = "") => idxs.map((i, n) => `<button class="choice ${cls}" data-opt="${i}">${n < 9 ? `<kbd>${n + 1}</kbd> ` : ""}${esc(clean(d.options[i]))}</button>`).join("");
  P.dataset.act = 'primary';
  switch (d.kind) {
    case 'priority': {
      if (S.paying) {
        const Pg = S.paying;
        pr.innerHTML = `<span class="k">${esc(Pg.verb)} ${esc(Pg.name)}</span>${Pg.step === 'target' ? 'Choose a target, or Esc to cancel.' : `Pay ${esc(Pg.cost)}: click lands to tap them, or Auto pay. Esc or right-click cancels.`}`;
        P.textContent = 'Auto pay'; P.dataset.act = 'paycommit'; P.disabled = Pg.step === 'target'; P.classList.add('ask');
        break;
      }
      const playable = d.refs.filter(isReal).length;
      const floating = manaWaiting(s);
      pr.innerHTML = `<span class="k">Your priority · ${esc(STEP_LABEL[s.step] || s.step)}</span>${floating ? `<b class="floatwarn">${floating} mana in your pool: it empties when this step ends.</b> ` : ''}${playable ? 'Drag a glowing card to the battlefield, or double-click it. Click a land to tap it for mana.' : 'Nothing to play.'}`;
      if (H.loose.length) ch.innerHTML = choiceBtns(H.loose);
      P.textContent = ui.armed === 'float' ? `Pass? ${floating} mana empties` : passLabel(s);
      P.classList.add(ui.armed ? 'armed' : 'ask');
      P.disabled = false;
      break;
    }
    case 'declare_attacker': {
      const n = ui.sel.size;
      pr.innerHTML = `<span class="k">Declare attackers</span>Click creatures to attack (or drag them forward). ${n ? '' : ui.eligible.size + ' can attack.'}`;
      ch.innerHTML = `<button class="choice sec" data-cmd="all">All attack (A)</button>${n ? '<button class="choice sec" data-cmd="clear">Clear</button>' : ''}`;
      P.textContent = n ? `Attack with ${n}` : ui.armed === 'noattack' ? 'Really skip attacking?' : 'No attacks';
      P.classList.add(n ? 'attack' : ui.armed ? 'armed' : 'ask');
      P.disabled = false;
      break;
    }
    case 'declare_blocker': {
      const n = ui.blocks.size;
      const atk = s.battlefield.filter(c => c.attacking && c.controller !== S.seat);
      const menace = atk.filter(a => kw(a, 'menace') && [...ui.blocks.values()].filter(x => x === a.oid).length === 1);
      pr.innerHTML = `<span class="k">Declare blockers</span>Drag your creature onto an attacker (or click yours, then theirs).${menace.length ? `<br><span class="err">${esc(menace[0].name)} has menace: it needs two blockers.</span>` : ''}`;
      if (n) ch.innerHTML = '<button class="choice sec" data-cmd="clear">Clear blocks</button>';
      P.textContent = n ? `Block (${n})` : 'No blocks';
      P.classList.add(n ? 'block' : 'ask');
      P.disabled = false;
      break;
    }
    case 'target': {
      const none = d.refs.findIndex(r => r.none);
      pr.innerHTML = `<span class="k">Choose a target</span>${esc(clean(d.prompt))}`;
      if (H.loose.length) ch.innerHTML = choiceBtns(H.loose.filter(i => i !== none));
      if (none >= 0) { P.textContent = 'No target'; P.disabled = false; P.dataset.act = 'opt:' + none; } else { P.textContent = 'Choose a target'; }
      break;
    }
    case 'pay_mana': {
      pr.innerHTML = `<span class="k">Pay mana</span>${esc(clean(d.prompt))}. Click a glowing source to pay with it.`;
      if (H.loose.length) ch.innerHTML = choiceBtns(H.loose);
      P.textContent = 'Auto-pay the rest'; P.disabled = false; P.dataset.act = 'autopay'; P.classList.add('ask');
      break;
    }
    case 'mulligan': {
      const keep = d.refs.findIndex(r => r.type === 'keep');
      pr.innerHTML = `<span class="k">Opening hand</span>${esc(d.prompt)}`;
      P.textContent = clean(d.options[keep]).replace(/^Keep \((\d+) cards\)$/, 'Keep $1'); P.disabled = false; P.dataset.act = 'opt:' + keep; P.classList.add('ask');
      ch.innerHTML = d.refs.map((r, i) => r.type === 'mulligan' ? `<button class="choice" data-opt="${i}">${esc(d.options[i].replace(/^Mulligan \(to (\d+)\)$/, 'Mulligan to $1'))}</button>` : '').join('');
      break;
    }
    case 'assign_damage_amount': {
      pr.innerHTML = `<span class="k">Assign combat damage</span>${esc(clean(d.prompt))}`;
      ch.innerHTML = d.refs.map((r, i) => `<button class="choice" data-opt="${i}">${r.amount} damage</button>`).join('');
      P.textContent = 'Choose damage amount'; P.disabled = true;
      break;
    }
    case 'assign_damage': {
      pr.innerHTML = `<span class="k">Assign combat damage</span>${esc(clean(d.prompt))}`;
      const i = splitIndex(d, ui.dmg);
      P.textContent = 'Assign damage'; P.disabled = i < 0; P.dataset.act = 'opt:' + i;
      break;
    }
    default: {
      pr.innerHTML = `<span class="k">${esc(d.kind.replace(/_/g, ' '))}</span>${esc(clean(d.prompt))}`;
      if (!overlayKind(d)) ch.innerHTML = choiceBtns(d.options.map((_, i) => i));
      P.textContent = overlayKind(d) ? 'Choose above' : 'Choose an option';
    }
  }
}
const agentName = a => !a ? 'Opponent' : a.replace(/^model:/, '').replace(/\/(model|latest)\b/, '').replace('scripted-bot', 'Scripted bot');
const REASONS = {life: 'life reached 0', concede: 'conceded', decked: 'drew from an empty library', poison: 'ten poison counters', turns: 'turn limit'};
function reasonText() {
  const m = S.meta, r = m?.end_reason || '', loser = m?.winner == null ? null : 1 - m.winner;
  if (r === 'concede') return loser === S.seat ? 'you conceded' : 'the opponent conceded';
  const why = REASONS[r] || r;
  return loser == null ? why : `${loser === S.seat ? 'your' : "the opponent's"} ${why}`;
}
const resultText = () => { const w = S.meta?.winner; return w == null ? `Draw (${reasonText()})` : w === S.seat ? `You win: ${reasonText()}` : `You lose: ${reasonText()}`; };

// Kinds shown as a card browser: options that name cards.
function overlayKind(d) {
  if (d.kind === 'assign_damage_amount') return false;
  if (d.kind === 'mulligan') return true;
  if (d.kind === 'order' && d.refs.some(r => r.top)) return true;
  if ((d.kind === 'choose_mode' || d.kind === 'yes_no') && d.cards?.length) return true;
  if (['priority', 'declare_attacker', 'declare_blocker', 'target', 'pay_mana'].includes(d.kind)) return false;
  if (d.kind === 'assign_damage') return true;
  return d.refs.filter(r => r.name && S.cards[r.name]).length >= Math.max(1, d.refs.length - 1);
}

function renderOverlay() {
  const d = myDecision(), ui = S.ui, el = $('#overlay');
  if (!d || !ui || !overlayKind(d)) { if (!el.dataset.zone) closeOverlay(); return; }
  delete el.dataset.zone;
  if (d.kind === 'mulligan') {
    const s = stateAt(ui.fi), hand = s.players[S.seat].hand;
    const lands = hand.filter(c => (S.cards[c.name]?.types || []).includes('Land')).length;
    const onPlay = S.meta?.starting_player === S.seat;
    const keep = d.refs.findIndex(r => r.type === 'keep'), mull = d.refs.findIndex(r => r.type === 'mulligan');
    const taken = s.players[S.seat].mulligans || 0;
    el.innerHTML = `<div class="obox mull"><h2>Opening hand${taken ? ` (after ${taken} mulligan${taken > 1 ? 's' : ''})` : ''}</h2>
      <div class="sub">You are <b>${onPlay ? 'on the play' : 'on the draw'}</b> · <span class="landchip ${lands <= 1 || lands >= 6 ? 'warn' : ''}">${lands} land${lands === 1 ? '' : 's'}</span>${taken ? ` · you will put ${taken} card${taken > 1 ? 's' : ''} on the bottom` : ''}</div>
      <div class="grid">${hand.map(c => `<div class="tile">${cardHtml(c.name, {uid: c.uid})}</div>`).join('')}</div>
      <div class="orow"><button class="primary" data-opt="${keep}">Keep ${hand.length - taken}</button>${mull >= 0 ? `<button class="primary alt" data-opt="${mull}">Mulligan to ${hand.length - taken - 1}</button>` : ''}</div></div>`;
    el.classList.add('on'); el.classList.remove('peek');
    return;
  }
  if (d.kind === 'assign_damage') {
    const r = d.refs[0], s = stateAt(ui.fi);
    const names = (r.to || []).map(o => s.battlefield.find(c => c.oid === o));
    const rows = names.map((c, j) => `<div class="dmgrow"><div>${esc(c ? c.name : '?')}<div class="l">${c ? `${c.power}/${c.toughness}, ${c.damage || 0} damage marked` : ''}</div></div>
      <button class="btn" data-dmg="${j}:-1">−</button><div class="n">${ui.dmg[j]}</div><button class="btn" data-dmg="${j}:1">+</button></div>`).join('')
      + (r.player ? `<div class="dmgrow"><div>Defending player<div class="l">trample</div></div><button class="btn" data-dmg="${names.length}:-1">−</button><div class="n">${ui.dmg[names.length]}</div><button class="btn" data-dmg="${names.length}:1">+</button></div>` : '');
    const ok = splitIndex(d, ui.dmg) >= 0;
    const why = ok ? '' : r.player ? 'Each blocker must get lethal damage before any tramples over.' : 'Not a split the rules allow.';
    const total = ui.dmg.reduce((a, b) => a + b, 0);
    el.innerHTML = `<div class="obox"><h2>${esc(clean(d.prompt))}</h2><div class="sub">Starts from a legal split (lethal to each blocker in order). + and − move a point between recipients · Space confirms.</div>${rows}
      <div class="dmgsum">${total} assigned${why ? ` · <span class="err">${esc(why)}</span>` : ''}</div>
      <div class="orow"><button class="primary" ${ok ? '' : 'disabled'} data-opt="${splitIndex(d, ui.dmg)}">${ok ? 'Assign damage' : 'Not legal yet'}</button></div></div>`;
  } else if (d.kind === 'order') {
    renderOrder(d, ui, el);
    return;
  } else if (d.cards?.length && (d.kind === 'choose_mode' || d.kind === 'yes_no')) {
    // a decision about cards you know (scry, surveil, explore, Delver): show them
    el.innerHTML = `<div class="obox"><h2>${esc(plainPrompt(d))}</h2><div class="sub">${d.cards.length > 1 ? 'The cards' : 'The card'} this is about</div>
      <div class="grid">${d.cards.map(n => `<div class="tile">${cardHtml(n)}</div>`).join('')}</div>
      <div class="textopts">${d.options.map((o, i) => `<button class="choice" data-opt="${i}"><kbd>${i + 1}</kbd> ${esc(clean(o))}</button>`).join('')}</div></div>
      <button class="btn peekbtn" data-cmd="peek">Peek at the board</button>`;
  } else {
    const isCard = r => r.name && S.cards[r.name];
    // identical names get their position; tapped permanents say so; "nothing" options go last
    const count = {}, seen = {};
    d.refs.forEach(r => { if (isCard(r)) count[r.name] = (count[r.name] || 0) + 1; });
    const tiles = d.refs.map((r, i) => {
      if (!isCard(r)) return '';
      seen[r.name] = (seen[r.name] || 0) + 1;
      const tag = [r.tapped ? 'tapped' : '', count[r.name] > 1 ? `${r.zone === 'battlefield' ? 'permanent' : 'copy'} ${seen[r.name]} of ${count[r.name]}` : ''].filter(Boolean).join(' · ');
      return `<button class="tile ${r.tapped ? 'is-tapped' : ''}" data-opt="${i}" ${r.oid != null ? `data-hl="${r.oid}"` : ''}>${cardHtml(r.name)}<span>${esc(clean(d.options[i]))}${tag ? `<em>${esc(tag)}</em>` : ''}</span></button>`;
    }).join('');
    const texts = d.refs.map((r, i) => isCard(r) ? '' : `<button class="choice" data-opt="${i}">${esc(clean(d.options[i]))}</button>`).join('');
    el.innerHTML = `<div class="obox"><h2>${esc(plainPrompt(d))}</h2><div class="sub">${esc(KIND_TEXT[d.kind] || 'Choose one')}</div><div class="grid">${tiles}</div>${texts ? `<div class="textopts">${texts}</div>` : ''}</div>
      <button class="btn peekbtn" data-cmd="peek">Peek at the board</button>`;
  }
  el.classList.add('on'); el.classList.remove('peek');
}
const KIND_TEXT = {choose_card: 'Choose a card', sacrifice: 'Choose what to sacrifice', exile_from_graveyard: 'Choose a card to exile from your graveyard', order_triggers: 'Which trigger goes on the stack first (it resolves last)'};
function plainPrompt(d) {
  return clean(d.prompt).replace(/\(([a-z_]+)\)/g, (_, k) => `(${k.replace(/_/g, ' ')})`).replace(/^Choose target \(([^)]*)\) for (.*)$/, 'Choose a target for $2: $1');
}

// ---- library orders (Ponder, scry 2+): drag the cards into the order you want
function renderOrder(d, ui, el) {
  const opts = d.refs.map(r => ({top: r.top || [], bottom: r.bottom || []}));
  const canBottom = opts.some(o => o.bottom.length);
  if (!ui.order) ui.order = {top: [...opts[0].top], bottom: [...opts[0].bottom]};
  const o = ui.order;
  const match = opts.findIndex(x => x.top.join('|') === o.top.join('|') && x.bottom.join('|') === o.bottom.join('|'));
  const tile = (row, n, i) => `<div class="otile" draggable="true" data-row="${row}" data-i="${i}">${cardHtml(n)}<div class="obtns">
      <button data-ord="${row}:${i}:-1" title="Earlier">◀</button>${canBottom ? `<button data-ord="${row}:${i}:x" title="${row === 'top' ? 'To the bottom' : 'To the top'}">${row === 'top' ? '▼' : '▲'}</button>` : ''}<button data-ord="${row}:${i}:1" title="Later">▶</button></div></div>`;
  const rowHtml = (row, label) => `<div class="orowlbl">${label}</div><div class="ordrow" data-row="${row}">${o[row].map((n, i) => tile(row, n, i)).join('') || '<span class="sub">empty</span>'}</div>`;
  el.innerHTML = `<div class="obox order"><h2>${esc(plainPrompt(d))}</h2><div class="sub">Drag the cards (or use the arrows). Leftmost ends on top of your library.</div>
    ${rowHtml('top', 'Top of library: first card is drawn first')}${canBottom ? rowHtml('bottom', 'Bottom of library: last card is the very bottom') : ''}
    <div class="orow"><button class="primary" ${match < 0 ? 'disabled' : ''} data-opt="${match}">${match < 0 ? 'Not an allowed order' : 'Confirm order'}</button></div></div>`;
  el.classList.add('on'); el.classList.remove('peek');
}
function moveOrder(ui, row, i, how) {
  const o = ui.order, [card] = o[row].splice(i, 1);
  if (how === 'x') o[row === 'top' ? 'bottom' : 'top'].push(card);
  else o[row].splice(Math.max(0, Math.min(o[row].length, i + how)), 0, card);
}
let ordDrag = null;
document.addEventListener('dragstart', e => { const t = e.target.closest?.('.otile'); if (t) { ordDrag = {row: t.dataset.row, i: +t.dataset.i}; e.dataTransfer.effectAllowed = 'move'; } });
document.addEventListener('dragover', e => { if (ordDrag && e.target.closest?.('.ordrow')) e.preventDefault(); });
document.addEventListener('drop', e => {
  const rowEl = e.target.closest?.('.ordrow'); if (!ordDrag || !rowEl || !S.ui?.order) return;
  e.preventDefault();
  const o = S.ui.order, [card] = o[ordDrag.row].splice(ordDrag.i, 1), row = rowEl.dataset.row;
  const over = e.target.closest('.otile');
  let at = over ? +over.dataset.i : o[row].length;
  if (over && over.dataset.row === ordDrag.row && ordDrag.row === row && ordDrag.i < at) at -= 0;
  o[row].splice(at, 0, card);
  ordDrag = null;
  renderOrder(S.ui.d, S.ui, $('#overlay'));
});

function closeOverlay() { const el = $('#overlay'); el.classList.remove('on', 'peek'); el.innerHTML = ''; delete el.dataset.zone; }

function openZone(p, zone) {
  const s = stateAt(S.shown), cards = s.players[p][zone];
  const el = $('#overlay');
  const map = p === S.seat ? (zone === 'graveyard' ? H.gy : H.ex) : new Map();
  el.dataset.zone = '1';
  el.innerHTML = `<div class="obox"><h2>${p === S.seat ? 'Your' : "Opponent's"} ${zone} (${cards.length})</h2><div class="sub">${map.size ? 'Glowing cards can be cast or activated: click one.' : 'Newest last.'}</div>
    <div class="grid">${cards.map(c => `<button class="tile" data-zcard="${esc(c.name)}" data-zp="${p}">${cardHtml(c.name, {cls: map.has(c.name) ? 'playable' : ''})}</button>`).join('') || '<span class="sub">Empty</span>'}</div>
    <div class="orow"><button class="btn" data-cmd="close">Close (Esc)</button></div></div>`;
  el.classList.add('on');
  el._map = map;
}

// ---------------------------------------------------------------- damage assignment
function splitIndex(d, split) { return d.refs.findIndex(r => r.split && r.split.length === split.length && r.split.every((v, j) => v === split[j])); }
function defaultSplit(d, s) {
  // Lethal to each blocker in order, the rest to the player (or the last blocker).
  const r0 = d.refs[0], blockers = (r0.to || []).map(o => s.battlefield.find(c => c.oid === o));
  let best = r0.split, bestKey = null;
  for (const r of d.refs) {
    const kills = r.split.reduce((n, v, j) => n + (blockers[j] && v >= blockers[j].toughness - (blockers[j].damage || 0) ? 1 : 0), 0);
    // most blockers killed, then most to the player, then earlier blockers first
    const key = [kills, r.player ? r.split[r.split.length - 1] : 0, ...r.split];
    if (!bestKey || lexLess(bestKey, key)) { best = r.split; bestKey = key; }
  }
  return [...best];
}

// ---------------------------------------------------------------- popover menus
function showMenu(anchor, idxs, title) {
  const d = myDecision(); if (!d) return;
  const pop = $('#pop');
  pop.innerHTML = `<div class="ph">${esc(title)}</div>` + idxs.map(i => `<button data-opt="${i}">${esc(clean(d.options[i]))}</button>`).join('');
  pop.classList.add('on');
  const r = anchor.getBoundingClientRect(), pw = pop.offsetWidth, ph = pop.offsetHeight;
  let x = r.left + r.width / 2 - pw / 2, y = r.top - ph - 8;
  if (y < 50) y = r.bottom + 8;
  x = Math.max(8, Math.min(window.innerWidth - pw - 8, x));
  y = Math.max(8, Math.min(window.innerHeight - ph - 8, y));
  pop.style.left = x + 'px'; pop.style.top = y + 'px';
  pop._source = anchor;
  pop._plan = undefined;  // a caller that drops onto a target sets it after
  pop._mana = null; pop._extra = [];
}
function addMenuItem(label, fn) {
  const pop = $('#pop');
  pop._extra.push(fn);
  pop.insertAdjacentHTML('beforeend', `<button data-extra="${pop._extra.length - 1}">${esc(label)}</button>`);
}
function closePop() { $('#pop').classList.remove('on'); }

// ---------------------------------------------------------------- gestures
const canAct = () => !!(myDecision() && S.ui && !S.busy && !S.pumping);
function handOpts(name) { return H.hand.get(name) || []; }
function permOpts(el) { const oids = el.dataset.oids.split(' ').map(Number); for (const o of oids) if (H.perm.has(o)) return H.perm.get(o); return []; }

// Drop and double-click mean "play it": only a land play or a normal cast.
// Anything else (cycling, flashback, alternative or extra costs, modes) or
// several such options open the menu instead, so nothing surprising happens.
function plainPlay(idxs) {
  const d = myDecision();
  const plain = idxs.filter(i => { const r = d.refs[i]; return r.type === 'play_land' || (r.type === 'cast' && r.mode === 'normal' && !r.spell_mode); });
  return plain.length === 1 && idxs.every(i => ['play_land', 'cast', 'plot', 'activate'].includes(d.refs[i].type)) && !idxs.some(i => i !== plain[0] && d.refs[i].type === 'cast') ? plain[0] : null;
}
function playGesture(el, idxs, plan) {
  if (!idxs.length) return;
  const i = plainPlay(idxs);
  if (i == null) { showMenu(el, idxs, el.dataset.name || 'Choose'); $('#pop')._plan = plan; return; }
  S.srcOid = null;
  beginCast(el, i, plan);
}
function chooseFromCard(el, idxs, plan) {
  if (!idxs.length) return;
  S.srcOid = el.dataset.oid != null ? +el.dataset.oid : null;
  if (idxs.length === 1) beginCast(el, idxs[0], plan);
  else { showMenu(el, idxs, el.dataset.name || 'Choose'); $('#pop')._plan = plan; }
}

function targetAt(x, y) {
  const el = document.elementFromPoint(x, y);
  if (!el) return null;
  const perm = el.closest('#board .perm'), plate = el.closest('.plate');
  if (perm) return {oid: +perm.dataset.oid, el: perm};
  if (plate) return {player: +plate.dataset.player, el: plate};
  return null;
}
// What a targeted spell may target, read from its text (the engine decides after the drop).
function markMayTarget(name) {
  const t = (S.cards[name]?.text || '').toLowerCase();
  const kinds = [];
  for (const k of ['land', 'creature', 'artifact', 'enchantment', 'planeswalker']) if (new RegExp(`target (\\w+ )?${k}|target (\\w+ )?(\\w+ or )?${k}`).test(t)) kinds.push(k[0].toUpperCase() + k.slice(1));
  const any = /target permanent|any target/.test(t);
  const s = stateAt(last());
  for (const el of $$('#board .perm')) {
    const c = s.battlefield.find(x => x.oid === +el.dataset.oid);
    if (c && (any || kinds.some(k => c.types.includes(k)))) el.classList.add('may-target');
  }
}
const mentionsTarget = name => /\btarget\b/i.test(S.cards[name]?.text || '');

let drag = null, suppressClick = false;
document.addEventListener('pointerdown', e => {
  if (e.button !== 0 || !canAct() || S.paying) return;
  const hc = e.target.closest('#hand .card'), pm = e.target.closest('#board .perm');
  const ui = S.ui;
  if (hc && ui.kind === 'priority' && handOpts(hc.dataset.name).length) drag = {kind: 'hand', el: hc, x0: e.clientX, y0: e.clientY};
  else if (pm && ui.kind === 'declare_blocker' && ui.blockers.has(+pm.dataset.oid)) drag = {kind: 'block', el: pm, x0: e.clientX, y0: e.clientY};
  else if (pm && ui.kind === 'declare_attacker' && ui.eligible.has(+pm.dataset.oid)) drag = {kind: 'attack', el: pm, x0: e.clientX, y0: e.clientY};
});
document.addEventListener('pointermove', e => {
  pointer = [e.clientX, e.clientY];
  if (S.ui?.kind === 'target' && !drag) requestAnimationFrame(drawArrows);
  if (!drag) return;
  const dx = e.clientX - drag.x0, dy = e.clientY - drag.y0;
  if (!drag.started) {
    if (Math.hypot(dx, dy) < 7) return;  // below the threshold it is a click
    drag.started = true;
    closePop();
    if (drag.kind === 'hand') startHandDrag(e);
  }
  if (drag.kind === 'hand') moveHandDrag(e);
  if (drag.kind === 'block') { hotTarget(e, el => el.closest('.perm.attacker-target')); drawArrows(); }
  if (drag.kind === 'attack') drag.el.style.translate = `0 ${Math.min(0, dy)}px`;
});
document.addEventListener('pointerup', e => {
  if (!drag) return;
  const dg = drag; drag = null;
  if (!dg.started) return;
  suppressClick = true; setTimeout(() => { suppressClick = false; }, 0);
  if (dg.kind === 'hand') endHandDrag(dg, e);
  if (dg.kind === 'block') {
    clearHot();
    const t = document.elementFromPoint(e.clientX, e.clientY)?.closest('.perm.attacker-target');
    if (t) assignBlock(+dg.el.dataset.oid, +t.dataset.oid);
    drawArrows();
  }
  if (dg.kind === 'attack') {
    dg.el.style.translate = '';
    if (e.clientY - dg.y0 < -25) toggleAttacker(+dg.el.dataset.oid, true);
  }
});

function startHandDrag(e) {
  showTaps(tapsFor(handOpts(drag.el.dataset.name)));
  const g = $('#ghost');
  g.innerHTML = drag.el.outerHTML.replace(/class="card [^"]*"/, 'class="card"');
  g.style.display = 'block';
  g.classList.remove('back');
  drag.el.classList.add('dragging');
  drag.w = drag.el.offsetWidth; drag.h = drag.el.offsetHeight;
  drag.vx = 0; drag.lx = e.clientX;
  $('#field0').classList.add('drop');
  drag.targets = mentionsTarget(drag.el.dataset.name) && plainPlay(handOpts(drag.el.dataset.name)) != null;
  if (drag.targets) markMayTarget(drag.el.dataset.name);
}
function moveHandDrag(e) {
  const g = $('#ghost');
  drag.vx = drag.vx * 0.7 + (e.clientX - drag.lx) * 0.3; drag.lx = e.clientX;
  const tilt = Math.max(-8, Math.min(8, drag.vx * 0.8));
  g.style.transform = `translate(${e.clientX - drag.w / 2}px, ${e.clientY - drag.h * 0.35}px) rotate(${tilt}deg) scale(1.05)`;
  const field = $('#field0'), r = $('#board').getBoundingClientRect();
  const inPlay = e.clientY < r.bottom - r.height * 0.22 && e.clientX < r.right;
  field.classList.toggle('hot', inPlay);
  clearHot();
  if (drag.targets) { const t = targetAt(e.clientX, e.clientY); if (t && (t.el.closest('.plate') || t.el.classList.contains('may-target'))) t.el.classList.add('drop-hot'); }
}
function endHandDrag(dg, e) {
  const g = $('#ghost'), field = $('#field0');
  const r = $('#board').getBoundingClientRect();
  const inPlay = e.clientY < r.bottom - r.height * 0.22 && e.clientX < r.right && e.clientX > r.left;
  field.classList.remove('drop', 'hot');
  clearHot();
  $$('.may-target').forEach(x => x.classList.remove('may-target'));
  showTaps(null);
  const idxs = handOpts(dg.el.dataset.name);
  if (inPlay && idxs.length) {
    g.style.display = 'none';
    dg.el.classList.remove('dragging');
    let plan;
    if (dg.targets) { const t = targetAt(e.clientX, e.clientY); if (t && (t.player != null || t.el.closest('#side1, #side0'))) plan = {kind: 'target', t: t.player != null ? {player: t.player} : {oid: t.oid}}; }
    playGesture(dg.el, idxs, plan);
    return;
  }
  // Snap back to the hand slot.
  const to = dg.el.getBoundingClientRect();
  g.classList.add('back');
  g.style.transform = `translate(${to.left}px, ${to.top}px) rotate(0deg) scale(1)`;
  setTimeout(() => { g.style.display = 'none'; g.classList.remove('back'); dg.el.classList.remove('dragging'); }, 230);
}
function hotTarget(e, pick) { clearHot(); const el = document.elementFromPoint(e.clientX, e.clientY); const t = el && pick(el); if (t) t.classList.add('drop-hot'); }
function clearHot() { $$('.drop-hot').forEach(x => x.classList.remove('drop-hot')); }

function toggleAttacker(oid, on) {
  const ui = S.ui; if (!ui || ui.kind !== 'declare_attacker' || !ui.eligible.has(oid)) return;
  if (on === true) ui.sel.add(oid); else if (ui.sel.has(oid)) ui.sel.delete(oid); else ui.sel.add(oid);
  ui.armed = null;
  render(S.shown, {noAnim: true}); renderDock();
}
function assignBlock(b, a) {
  const ui = S.ui, s = stateAt(ui.fi);
  const bc = s.battlefield.find(c => c.oid === b), ac = s.battlefield.find(c => c.oid === a);
  if (!bc || !ac) return;
  if (!canBlock(bc, ac)) { toast(`${bc.name} can't block ${ac.name}${kw(ac, 'flying') ? ' (flying)' : ''}.`); return; }
  ui.blocks.set(b, a); ui.blockSel = null;
  render(S.shown, {noAnim: true}); renderDock();
}

// Clicks
document.addEventListener('click', e => {
  if (suppressClick) return;
  const t = e.target;
  if ($('#spot').classList.contains('big') && (t.closest('#spot') || t.closest('#board'))) { S.spotSkip = true; return; }
  // skip the bot's replay
  if (S.queue.length && S.pumping && t.closest('#board')) { S.skip = true; return; }
  const mb = t.closest('[data-mana]');
  if (mb && $('#pop')._mana) { const pm = $('#pop')._mana; (pm.cb || reserveMana)(pm.oid, mb.dataset.mana); return; }
  const pb = t.closest('[data-pay]');
  if (pb) { if (pb.dataset.pay === 'auto') payCommit(); else payCancel(); return; }
  if (S.paying && canAct()) {
    const P = S.paying, pm = t.closest('#board .perm'), plate = t.closest('.plate'), si = t.closest('.sitem');
    if (P.step === 'target') {
      const keys = payTargetKeys();
      if (pm) { const o = pm.dataset.oids.split(' ').find(x => keys.has('o' + x)); if (o) { chooseCastTarget({oid: +o}); return; } }
      if (plate && keys.has('p' + plate.dataset.player)) { chooseCastTarget({player: +plate.dataset.player}); return; }
      if (si && keys.has('s' + si.dataset.sid)) { chooseCastTarget({sid: +si.dataset.sid}); return; }
    } else if (pm) { payWith(pm); return; }
    if (t.closest('#hand .card')) { toast('Finish paying (Auto pay) or Cancel first.'); return; }
    return;
  }
  const mi = t.closest('[data-extra]');
  if (mi && $('#pop')._extra) { $('#pop')._extra[+mi.dataset.extra](); return; }
  const flag = t.closest('[data-flag]');
  if (flag) { openFlagForm('bot', +flag.dataset.flag); return; }
  const stop = t.closest('[data-stop]');
  if (stop) { const [who, st] = stop.dataset.stop.split(':'); PREF.stops[who][st] = !PREF.stops[who][st]; savePref(); renderMid(stateAt(S.shown)); return; }
  if (t.closest('[data-resync]')) { resync(); return; }
  const cmd = t.closest('[data-cmd]');
  if (cmd) return command(cmd.dataset.cmd);
  const opt = t.closest('[data-opt]');
  if (opt && canAct()) {
    const i = +opt.dataset.opt; if (i < 0 || inputLocked()) return;
    const pop = t.closest('#pop'), plan = pop?._plan, src = pop?._source;
    closePop(); closeDrawer();
    if (S.ui?.kind === 'priority' && ['cast', 'activate', 'plot'].includes(S.ui.d.refs[i]?.type)) return beginCast(src, i, plan);
    return act(i, plan);
  }
  const zc = t.closest('[data-zcard]');
  if (zc) {
    const idxs = $('#overlay')._map?.get(zc.dataset.zcard) || [];
    if (idxs.length && canAct()) showMenu(zc, idxs, zc.dataset.zcard);
    return;
  }
  const yb = t.closest('[data-yield]');
  if (yb) {
    const n = yb.dataset.yield;
    if (S.yields.has(n)) S.yields.delete(n); else S.yields.add(n);
    toast(S.yields.has(n) ? `Auto-passing ${n}'s spells and triggers this game.` : `Stopping for ${n} again.`);
    render(S.shown, {noAnim: true}); renderDock();
    if (S.yields.has(n) && canAct() && S.ui.kind === 'priority') { const top = stateAt(last()).stack.at(-1); if (top && top.controller !== S.seat && (top.card || top.name.split(':')[0]) === n) return act(S.ui.d.refs.findIndex(r => r.type === 'pass')); }
    return;
  }
  const dl = t.closest('[data-decks]');
  if (dl) { openDecks(); return; }
  const zone = t.closest('.zone[data-zone]');
  if (zone) { openZone(+zone.dataset.p, zone.dataset.zone); return; }
  if (!t.closest('#pop')) closePop();
  if (t.closest('#fFlag')) return;
  if (!canAct()) return;
  const ui = S.ui;
  const hc = t.closest('#hand .card'), pm = t.closest('#board .perm'), plate = t.closest('.plate'), si = t.closest('.sitem');
  if (ui.kind === 'priority') {
    if (hc && handOpts(hc.dataset.name).length) return showMenu(hc, handOpts(hc.dataset.name), hc.dataset.name);
    const tapRes = pm && pm.classList.contains('reserved');
    if (pm && permOpts(pm).length && !tapRes) {
      S.srcOid = +pm.dataset.oid;
      showMenu(pm, permOpts(pm), pm.dataset.name);
      if (pm.classList.contains('manasrc')) addMenuItem('Tap for mana', () => { closePop(); tapForMana(pm); });
      return;
    }
    if (pm && (pm.classList.contains('manasrc') || tapRes)) { tapForMana(pm); return; }
  }
  if (ui.kind === 'target') {
    if (pm) { const o = pm.dataset.oids.split(' ').find(x => H.target.has('o' + x)); if (o) return act(H.target.get('o' + o)); }
    if (plate && H.target.has('p' + plate.dataset.player)) return act(H.target.get('p' + plate.dataset.player));
    if (si && H.target.has('s' + si.dataset.sid)) return act(H.target.get('s' + si.dataset.sid));
  }
  if (ui.kind === 'pay_mana' && pm) { const o = pm.dataset.oids.split(' ').find(x => H.pay.has(+x)); if (o) { const idxs = H.pay.get(+o); return idxs.length === 1 ? act(idxs[0]) : showMenu(pm, idxs, 'Pay with'); } }
  if (ui.kind === 'declare_attacker' && pm) return toggleAttacker(+pm.dataset.oid);
  if (ui.kind === 'declare_blocker' && pm) {
    const oid = +pm.dataset.oid;
    if (ui.blockers.has(oid)) {
      if (ui.blocks.has(oid)) ui.blocks.delete(oid); else ui.blockSel = ui.blockSel === oid ? null : oid;
      render(S.shown, {noAnim: true}); renderDock(); return;
    }
    if (pm.classList.contains('attacker-target') && ui.blockSel != null) return assignBlock(ui.blockSel, oid);
  }
});
document.addEventListener('dblclick', e => {
  if (!canAct() || S.ui.kind !== 'priority') return;
  const hc = e.target.closest('#hand .card'), pm = e.target.closest('#board .perm');
  if (S.paying) return;
  if (pm && (pm.classList.contains('manasrc') || pm.classList.contains('reserved')) && !permOpts(pm).length) {
    if (!pm.classList.contains('reserved')) tapForMana(pm);
    return;
  }
  closePop();
  if (hc && handOpts(hc.dataset.name).length) return playGesture(hc, handOpts(hc.dataset.name));
  if (pm && permOpts(pm).length) {
    const d = myDecision(), idxs = permOpts(pm);
    const risky = idxs.some(i => /sacrific|discard|exile/i.test(d.refs[i].ability || d.options[i]));
    return risky || idxs.length > 1 ? showMenu(pm, idxs, pm.dataset.name) : chooseFromCard(pm, idxs);
  }
});
// Focus: interactive cards can be reached with Tab; focusing one shows it in the preview.
function markFocusable() {
  for (const el of $$('#hand .card.playable, #board .perm.activatable, #board .perm.targetable, #board .perm.can-attack, #board .perm.attack-sel, #board .perm.can-block, #board .perm.blocked-by, #board .perm.attacker-target, #board .perm.payable, .plate.targetable, .sitem')) {
    el.tabIndex = 0;
    const name = el.dataset.name || el.querySelector('[data-name]')?.dataset.name || (el.classList.contains('plate') ? (el.dataset.player == S.seat ? 'You' : 'Opponent') : '');
    el.setAttribute('role', 'button');
    el.setAttribute('aria-label', name + (el.classList.contains('tapped') ? ', tapped' : ''));
  }
}
document.addEventListener('focusin', e => { const c = e.target.closest?.('[data-name]'); if (c && S.cards[c.dataset.name]) showPreview(c.dataset.name); });

document.addEventListener('contextmenu', e => {
  if (S.paying) { e.preventDefault(); payCancel(); return; }
  const ui = S.ui, pm = e.target.closest('#board .perm');
  if (ui && pm && canAct()) {  // undo a selection that has not been sent yet
    const oid = +pm.dataset.oid;
    if (ui.kind === 'declare_attacker' && ui.sel.has(oid)) { e.preventDefault(); return toggleAttacker(oid); }
    if (ui.kind === 'declare_blocker' && (ui.blocks.has(oid) || ui.blockSel === oid)) {
      e.preventDefault(); ui.blocks.delete(oid); ui.blockSel = null; render(S.shown, {noAnim: true}); renderDock(); return;
    }
    if (ui.kind === 'declare_blocker' && [...ui.blocks.values()].includes(oid)) {  // right-click an attacker: drop its blockers
      e.preventDefault(); for (const [b, a] of [...ui.blocks]) if (a === oid) ui.blocks.delete(b); render(S.shown, {noAnim: true}); renderDock(); return;
    }
  }
  const c = e.target.closest('.card[data-name]');
  if (c) { e.preventDefault(); showPreview(c.dataset.name, true); }
});

// Auto-pay preview: the server simulated each cast with the auto-pay choice
// (live_proto.tap_preview); light up those sources while a card is in hand.
function tapsFor(idxs) {
  const d = myDecision();
  if (!d || !S.ui || S.ui.kind !== 'priority') return null;
  for (const i of idxs || []) {
    const r = d.refs[i];
    if (!r?.taps) continue;
    if (!S.reserved.length) return {taps: r.taps, sacs: r.sacs || [], used: []};
    const s = stateAt(last());
    const need = r.taps.length;
    const used = S.reserved.slice(0, need).map(x => x.oid);
    const free = r.taps.filter(o => !used.some(u => sameSource(s, o, u)));
    return {taps: free.slice(0, Math.max(0, need - used.length)), sacs: r.sacs || [], used};
  }
  return null;
}
function showTaps(p) {
  $$('#board .will-tap').forEach(el => { el.classList.remove('will-tap', 'will-sac'); el.querySelectorAll('.b.tap').forEach(b => b.remove()); });
  $$('.mana i.used, #board .perm.uses').forEach(el => el.classList.remove('used', 'uses'));
  if (!p) return;
  for (const o of p.used || []) {
    $(`.mana i[data-res="${o}"]`)?.classList.add('used');
    $$('#board .perm.reserved').forEach(el => { if (el.dataset.oids.split(' ').includes(String(o))) el.classList.add('uses'); });
  }
  const taps = new Set(p.taps), sacs = new Set(p.sacs);
  for (const el of $$('#board .perm')) {
    const oids = el.dataset.oids.split(' ').map(Number);
    const n = oids.filter(o => taps.has(o)).length, m = oids.filter(o => sacs.has(o)).length;
    if (!n && !m) continue;
    el.classList.add('will-tap');
    if (m) el.classList.add('will-sac');
    const label = [n ? (n > 1 ? `tap ×${n}` : 'tap') : '', m ? (m > 1 ? `sacrifice ×${m}` : 'sacrifice') : ''].filter(Boolean).join(' · ');
    el.querySelector('.badges').insertAdjacentHTML('beforeend', `<span class="b tap">${label}</span>`);
  }
}

// Hover: hand lift with neighbours spreading, and the preview panel.
let hoverTimer = null, pinned = false;
document.addEventListener('pointerover', e => {
  const hp = e.target.closest('#board .perm');
  if (hp !== document.querySelector('#board .perm.hovered')) {
    $$('#board .perm.hovered').forEach(x => x.classList.remove('hovered'));
    if (hp) hp.classList.add('hovered');
    if (S.shown >= 0 && stateAt(S.shown).battlefield.filter(c => c.attacking).length > 4) requestAnimationFrame(drawArrows);
  }
  if (!drag) {
    const hcard = e.target.closest('#hand .card'), pcard = e.target.closest('#board .perm.activatable'), mitem = e.target.closest('#pop [data-opt]');
    showTaps(S.paying ? payingSuggestion() : hcard ? tapsFor(handOpts(hcard.dataset.name)) : pcard ? tapsFor(permOpts(pcard)) : mitem ? tapsFor([+mitem.dataset.opt]) : null);
  }
  const c = e.target.closest('.card[data-name], .sitem[data-name], .dline[data-name]');
  const hc = e.target.closest('#hand .card');
  $$('#hand .card').forEach(x => { x.classList.toggle('hover', x === hc && !drag); });
  if (hc && !drag) {
    const cards = $$('#hand .card'), k = cards.indexOf(hc);
    cards.forEach((x, j) => x.style.setProperty('--sx', j === k ? '0px' : `${j < k ? -14 : 14}px`));
  } else if (!e.target.closest('#hand')) $$('#hand .card').forEach(x => x.style.setProperty('--sx', '0px'));
  clearTimeout(hoverTimer);
  if (c && !pinned && !S.paying) hoverTimer = setTimeout(() => showPreview(c.dataset.name), hc ? 160 : 300);
  else if (!c || S.paying) hidePreview();
});
document.addEventListener('pointerleave', () => $$('#hand .card').forEach(x => x.classList.remove('hover')));
document.addEventListener('pointerout', e => {  // leaving a hand card for anything outside the hand drops its lift
  const hc = e.target.closest('#hand .card');
  if (hc && !(e.relatedTarget && e.relatedTarget.closest && e.relatedTarget.closest('#hand .card'))) { hc.classList.remove('hover'); showTaps(null); }
});
function showPreview(name, pin) {
  if (pin) pinned = !pinned;
  const i = S.cards[name], el = $('#hoverprev');
  if (!i || drag || $('#spot').classList.contains('big')) return;
  const tl = i.types.join(' ') + (i.subtypes.length ? ' — ' + i.subtypes.join(' ') : '');
  el.innerHTML = cardHtml(name) + `<div class="ptxt"><b>${esc(name)}</b> ${esc(i.cost)}<div>${esc((i.token ? 'Token ' : '') + tl)}${i.power != null ? ` · ${i.power}/${i.toughness}` : ''}</div><div class="ot">${esc(i.text)}</div>${pinned ? '<div class="st">Pinned (right-click again or Esc to unpin)</div>' : ''}</div>`;
  el.classList.add('on');
  const w = el.offsetWidth, h = el.offsetHeight;
  const left = pointer[0] > innerWidth * 0.6 ? 16 : innerWidth - w - 16;  // the right edge (your lands sit left), unless the pointer is there
  el.style.left = `${left}px`;
  el.style.top = `${Math.max(50, Math.min(innerHeight - h - 150, pointer[1] - h / 2))}px`;  // clear of the dock
}
function hidePreview() { if (!pinned) $('#hoverprev').classList.remove('on'); }

// ---------------------------------------------------------------- commands and keys
async function command(c) {
  const ui = S.ui;
  if (c === 'peek') { $('#overlay').classList.toggle('peek'); setTimeout(() => { if ($('#overlay').classList.contains('peek')) document.addEventListener('click', () => $('#overlay').classList.remove('peek'), {once: true}); }, 0); return; }
  if (c === 'close') { closeOverlay(); return; }
  if (c === 'stack-more') { stackOpen = !stackOpen; render(S.shown, {noAnim: true}); return; }
  if (!ui) return;
  if (c === 'all' && ui.kind === 'declare_attacker') { ui.eligible.forEach(o => ui.sel.add(o)); ui.armed = null; render(S.shown, {noAnim: true}); renderDock(); }
  if (c === 'clear') { ui.sel.clear(); ui.blocks.clear(); ui.blockSel = null; render(S.shown, {noAnim: true}); renderDock(); }
}

function primary() {
  if ($('#spot').classList.contains('big')) { S.spotSkip = true; return; }
  if (S.queue.length && S.pumping) { S.skip = true; return; }
  if (inputLocked()) return;
  const P = $('#primary');
  if (P.disabled) return;
  const a = P.dataset.act || '';
  if (a === 'new') return openNewGame();
  if (a === 'resync') return resync();
  if (a === 'reload') return location.reload();
  if (a.startsWith('opt:')) { const i = +a.slice(4); if (i >= 0) return act(i); return; }
  if (a === 'autopay') { const d = myDecision(); S.autoRest = true; return act(planPay(d)); }
  if (a === 'paycommit') return payCommit();
  if (!canAct()) return;
  const ui = S.ui, d = ui.d, s = stateAt(ui.fi);
  if (ui.kind === 'priority') {
    if (manaWaiting(s) && ui.armed !== 'float') { ui.armed = 'float'; renderDock(); return; }
    return act(d.refs.findIndex(r => r.type === 'pass'));
  }
  if (ui.kind === 'declare_attacker') {
    if (!ui.sel.size) {
      if (PREF.confirmEmptyAttack && ui.eligible.size && ui.armed !== 'noattack') { ui.armed = 'noattack'; renderDock(); return; }
      return act(d.refs.findIndex(r => r.done));
    }
    const want = [...ui.sel];
    S.plan = stamp({kind: 'attack', want});
    const i = planAnswer(d);
    return act(i);
  }
  if (ui.kind === 'declare_blocker') {
    S.plan = stamp({kind: 'block', map: new Map(ui.blocks)});
    return act(planAnswer(d));
  }
}

document.addEventListener('keydown', e => {
  if (S.rv && !$('#modal').classList.contains('on') && !e.target.closest?.('input, select, textarea')) { rvKey(e); return; }
  if (e.key === 'Escape' && $('#modal').classList.contains('on') && S.gid && !$('#fNew')) { $('#modal').classList.remove('on'); return; }
  if (e.key === 'Escape' && $('#pop').classList.contains('on')) { closePop(); return; }
  const pm = $('#pop').classList.contains('on') && $('#pop')._mana;
  if (pm && pm.colors.includes(e.key.toUpperCase()) && !e.ctrlKey && !e.metaKey) { e.preventDefault(); (pm.cb || reserveMana)(pm.oid, e.key.toUpperCase()); return; }
  if (S.paying && !$('#modal').classList.contains('on')) {
    if (e.key === 'Escape') { e.preventDefault(); payCancel(); return; }
    if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) { e.preventDefault(); payCommit(); return; }
  }
  if (e.key === 'Escape' && $('#menu').classList.contains('on')) { $('#menu').classList.remove('on'); return; }
  if (e.key === 'Escape' && pinned) { pinned = false; hidePreview(); return; }
  if (e.key === 'Escape' && $('#logdrawer').classList.contains('on')) { toggleLog(false); return; }
  if (e.target.closest('input, select, textarea') || $('#modal').classList.contains('on')) return;
  const k = e.key;
  if (e.repeat && (k === ' ' || k === 'Enter' || k === 'r' || k === 'R' || k === 'a' || k === 'n')) { e.preventDefault(); return; }  // a held key is not a decision
  if (inputLocked() && [' ', 'Enter', 'r', 'R', 'a', 'A', 'n', 'N'].includes(k)) { e.preventDefault(); return; }
  // A focused card, permanent, plate or stack item: Enter or Space clicks it (Shift+Enter double-clicks: plays it).
  const fc = e.target.closest?.('#hand .card, #board .perm, .plate, .sitem');
  if (fc && fc.matches(':focus-visible') && (k === 'Enter' || k === ' ')) {  // keyboard focus only: a clicked card must not capture Space
    e.preventDefault();
    fc.dispatchEvent(new MouseEvent(e.shiftKey ? 'dblclick' : 'click', {bubbles: true}));
    return;
  }
  // A focused button, link or option keeps Space and Enter (activate it), never the global hotkeys.
  if ((k === ' ' || k === 'Enter') && e.target !== document.body && (e.target.closest('button, a') || (e.target.closest('[role=button], [tabindex]') && e.target.matches(':focus-visible')))) return;
  if (k === ' ') { e.preventDefault(); if ($('#overlay').classList.contains('on') && (S.ui?.kind === 'assign_damage' || S.ui?.kind === 'mulligan')) { const b = $('#overlay .primary'); if (b && !b.disabled) b.click(); return; } primary(); return; }
  if (k === 'Escape') {
    if (drag) { const dg = drag; drag = null; $('#field0').classList.remove('drop', 'hot'); $('#ghost').style.display = 'none'; dg.el.classList.remove('dragging'); dg.el.style.translate = ''; clearHot(); }
    closePop(); closeDrawer();
    if ($('#overlay').dataset.zone) closeOverlay();
    S.passMode = null;
    if (S.ui) { S.ui.armed = null; S.ui.blockSel = null; if (S.ui.kind === 'declare_attacker' || S.ui.kind === 'declare_blocker') { S.ui.sel.clear(); S.ui.blocks.clear(); } render(S.shown, {noAnim: true}); }
    renderDock();
    return;
  }
  if ((k === 'r' || k === 'R') && canAct() && S.ui.kind === 'priority') {
    S.passMode = 'opp'; S.passTurn = stateAt(last()).active === S.seat ? stateAt(last()).turn : -1;
    return act(S.ui.d.refs.findIndex(r => r.type === 'pass'));
  }
  if (/^[1-9]$/.test(k) && canAct() && !inputLocked()) {
    const b = document.querySelector(`#overlay.on [data-opt] kbd, #choices [data-opt] kbd`) ? [...document.querySelectorAll('#overlay.on [data-opt], #choices [data-opt]')].find(x => x.querySelector('kbd')?.textContent === k) : null;
    if (b) { b.click(); return; }
  }
  if (k === 'l' || k === 'L') { toggleLog(); return; }
  if (k === 'h' || k === 'H') { S.holdOnce = !S.holdOnce; toast(S.holdOnce ? 'You keep priority after your next spell (to respond to it yourself).' : 'Hold priority off.'); renderDock(); return; }
  if (k === 'f' || k === 'F') { S.fullControl = !S.fullControl; toast(S.fullControl ? 'Full control: every priority stop is yours.' : 'Auto-pass back on.'); renderDock(); return; }
  if ((k === 'a' || k === 'A') && S.ui?.kind === 'declare_attacker') return command('all');
  if ((k === 'n' || k === 'N') && canAct()) {
    const d = S.ui.d;
    if (S.ui.kind === 'declare_attacker') return act(d.refs.findIndex(r => r.done));
    if (S.ui.kind === 'declare_blocker') { S.plan = stamp({kind: 'block', map: new Map()}); return act(planAnswer(d)); }  // no blocks at all
  }
  if (k === 'o' || k === 'O') { $('#drawer').classList.contains('on') ? closeDrawer() : openDrawer(); }
});
$('#primary').addEventListener('click', e => { e.stopPropagation(); primary(); });
document.addEventListener('mouseup', e => { const b = document.activeElement; if (b && b !== document.body && b.closest('button, [tabindex]') && !b.closest('form, #modal')) b.blur(); });  // mouse clicks don't park focus (Space stays the pass key)
$('#bAll').addEventListener('click', e => { e.stopPropagation(); openDrawer(); });
$('#bNew').addEventListener('click', () => {
  if (S.gid && !S.over && S.raw.length && !confirm('Leave this game? It counts as a loss and cannot be resumed.')) return;
  openNewGame();
});
async function concede() {
  if (!S.gid || S.over || S.busy || S.pumping) return;
  if (!confirm('Concede this game?')) return;
  try { applyView(await api(`${encodeURIComponent(S.gid)}/concede`, {})); pump(); } catch (e) { toast(e.message); }
}
$('#bHelp').addEventListener('click', () => openHelp());
const muteLabel = () => { $('#bMute').textContent = PREF.sound ? '🔊' : '🔇'; $('#bMute').title = PREF.sound ? 'Sound on (click to mute)' : 'Muted (click for sound)'; };
$('#bMute').addEventListener('click', () => { PREF.sound = !PREF.sound; savePref(); muteLabel(); if (PREF.sound) sound('decide'); });
muteLabel();

const HELP = [
  ['Play a land or cast a spell', 'Drag the glowing card up onto the battlefield, or double-click it. One click opens its options (flashback, cycling, modes).'],
  ['Tap a land for mana', 'Click an untapped land at priority. A land with several colours (Drossforge Bridge) asks which: click a pip or press W/U/B/R/G/C. The mana shows in your pool on your plate; click the land again to untap it. Unused mana empties when the step ends.'],
  ['Paying', 'A spell you play waits beside the stack with its cost as pips; nothing is paid yet. Click lands to tap them (each fills a pip) or press Auto pay (Space/Enter). Esc, right-click or Cancel puts the card back. Floating mana and lands you tapped beforehand fill pips first; the lands auto pay would use glow. Settings: "Auto-pay without asking" casts at once.'],
  ['Target', 'Click a glowing (cyan) creature, player plate or stack item. Or drop a targeted spell right onto its target.'],
  ['Attack', 'Click your creatures (or drag them forward), A for all, then Space. Right-click undoes one.'],
  ['Block', 'Drag your creature onto an attacker, or click yours then theirs. Space confirms.'],
  ['Pass', 'Space. The button always says what happens next. R passes until the opponent does something; F turns auto-pass off; H keeps priority after your next spell.'],
  ['Auto-pass stops', 'The small bars under each step in the middle strip: gold for your turn, blue for theirs.'],
  ['Everything else', 'O lists every option the engine offers. Number keys pick listed choices. Esc cancels a menu or selection.'],
  ['Read a card', 'Hover it for a big preview; right-click pins it (Esc unpins). Click a deck name for both decklists.'],
  ['The opponent', 'Their moves replay one by one; click or Space skips. Choices that matter (Brainstorm, Ponder, scry, Delver, mulligans, targets, modes, blocks, sideboarding) stay in a centred spotlight for a few seconds (Settings: short, normal, long); click or Space continues. Hidden cards stay hidden: you see how many and where they went. Their last move shows on their plate; L opens the full log.'],
  ['Report a problem', '⚑ next to their last move, or the Menu: "Bot played wrong" or "Bug: engine / UI". It becomes a public GitHub issue with only what you could see.'],
];
function openHelp() {
  const m = $('#modal');
  m.innerHTML = `<div class="mbox help"><h2>How to play</h2><table>${HELP.map(([a, b]) => `<tr><th>${esc(a)}</th><td>${esc(b)}</td></tr>`).join('')}</table>
    <div style="margin-top:14px;display:flex;gap:8px"><button class="primary" id="bHelpClose">Got it</button><button class="btn" id="bCoachAgain">Show the first-game tips again</button></div></div>`;
  m.classList.add('on');
  $('#bHelpClose').focus();
  $('#bHelpClose').onclick = () => m.classList.remove('on');
  $('#bCoachAgain').onclick = () => { saveJSON('mtgml-play-coached', false); m.classList.remove('on'); coach(true); };
}

// First-game callouts: short tips anchored to the parts of the screen, once.
const COACH = [  // first-game tips
  ['#hand', 'Your hand. Drag a glowing card up to play it, or double-click it.'],
  ['#primary', 'The main button says what happens next. Space presses it.'],
  ['#lands0', 'Your lands. Play a spell and it waits for payment: click lands to tap them, or press Auto pay (Space). Esc cancels.'],
  ['#rail', 'Where auto-pass stops: click the small bars under a step.'],
  ['#bLog', 'The game log (L). The ⚑ next to the opponent\'s last move reports a bad bot play.'],
];
function coach(force) {
  if (!force && loadJSON('mtgml-play-coached', false)) return;
  let k = 0;
  const show = () => {
    $$('.coach').forEach(x => x.remove());
    if (k >= COACH.length) { saveJSON('mtgml-play-coached', true); return; }
    const [sel, text] = COACH[k], at = $(sel)?.getBoundingClientRect();
    if (!at) { k++; return show(); }
    const c = document.createElement('div');
    c.className = 'coach';
    c.innerHTML = `<div>${esc(text)}</div><div class="cbtns"><span>${k + 1}/${COACH.length}</span><button class="btn small" data-coach="skip">Skip tips</button><button class="primary small" data-coach="next">${k + 1 < COACH.length ? 'Next' : 'Got it'}</button></div>`;
    document.body.appendChild(c);
    const w = c.offsetWidth, h = c.offsetHeight;
    let x = at.left + at.width / 2 - w / 2, y = at.top - h - 12;
    if (y < 50) y = at.bottom + 12;
    c.style.left = `${Math.max(8, Math.min(innerWidth - w - 8, x))}px`;
    c.style.top = `${Math.max(8, Math.min(innerHeight - h - 8, y))}px`;
    c.querySelector('[data-coach=next]').onclick = () => { k++; show(); };
    c.querySelector('[data-coach=skip]').onclick = () => { k = COACH.length; show(); };
  };
  show();
}
function toggleLog(on) {
  const d = $('#logdrawer'), open = on ?? !d.classList.contains('on');
  d.classList.toggle('on', open); $('#bLog').classList.toggle('on', open);
  if (open) $('#log').scrollTop = $('#log').scrollHeight;
}
$('#bLog').addEventListener('click', e => { e.stopPropagation(); toggleLog(); });
$('#bLogClose').addEventListener('click', e => { e.stopPropagation(); toggleLog(false); });
$('#bMenu').addEventListener('click', e => { e.stopPropagation(); $('#menu').classList.toggle('on'); });
$('#menu').addEventListener('click', e => {
  const b = e.target.closest('[data-menu]'); if (!b) return;
  $('#menu').classList.remove('on');
  const k = b.dataset.menu;
  if (k === 'viewer') return;
  e.preventDefault();
  if (k === 'new') $('#bNew').click();
  else if (k === 'concede') concede();
  else if (k === 'flag-bot') openFlagForm('bot', S.lastOppFi);
  else if (k === 'flag-bug') openFlagForm('bug');
  else if (k === 'decks') openDecks();
  else if (k === 'review') openReview();
  else if (k === 'settings') openSettings();
});
document.addEventListener('click', e => { if (!e.target.closest('#menu, #bMenu')) $('#menu').classList.remove('on'); }, true);
$('#overlay').addEventListener('click', e => {
  const ob = e.target.closest('[data-ord]');
  if (ob && S.ui?.order) { const [row, i, how] = ob.dataset.ord.split(':'); moveOrder(S.ui, row, +i, how === 'x' ? 'x' : +how); renderOrder(S.ui.d, S.ui, $('#overlay')); }
});
$('#overlay').addEventListener('click', e => { if (e.target.dataset.dmg) { const [j, dv] = e.target.dataset.dmg.split(':').map(Number); moveDamage(S.ui.dmg, j, dv); renderOverlay(); renderDock(); } });
// + takes a point from another recipient (the last one that has some), − gives
// it to the next one, so the split always adds up to the attacker's power.
function moveDamage(dmg, j, dv) {
  const n = dmg.length;
  if (dv > 0) {
    let k = -1;
    for (let i = n - 1; i >= 0; i--) if (i !== j && dmg[i] > 0) { k = i; break; }
    if (k < 0) return;
    dmg[k]--; dmg[j]++;
  } else if (dmg[j] > 0) {
    dmg[j]--; dmg[(j + 1) % n]++;
  }
}

// The fallback: every option of the current decision as a plain list.
function openDrawer() {
  const d = myDecision(); if (!d || !S.ui) return;
  const el = $('#drawer');
  el.innerHTML = `<div class="dh">All options <span style="color:var(--muted);font-weight:400;font-size:12px">${esc(d.kind.replace(/_/g, ' '))}</span><span class="sp"></span><button class="btn small" data-cmd="drawer-close" id="bDrawerClose">Close</button></div>
    <div class="dl">${d.options.map((o, i) => `<button data-opt="${i}">${esc(o)}</button>`).join('')}</div>`;
  el.classList.add('on');
  $('#bDrawerClose').onclick = e => { e.stopPropagation(); closeDrawer(); };
}
function closeDrawer() { $('#drawer').classList.remove('on'); }

function toast(text) {
  const el = document.createElement('div'); el.textContent = text;
  $('#toast').appendChild(el);
  setTimeout(() => el.remove(), 2600);
}
// Short synthesized sounds (Web Audio, no files): each is a few oscillator
// notes with a fast envelope. PREF.sound mutes, PREF.volume scales.
const SOUNDS = {
  cast: [[520, 880, 0.12, 'triangle']], land: [[110, 70, 0.12, 'sine']], attack: [[300, 160, 0.16, 'sawtooth']],
  hit: [[180, 90, 0.10, 'square']], death: [[260, 70, 0.35, 'triangle']], turn: [[660, 660, 0.10, 'sine'], [990, 990, 0.14, 'sine', 0.09]],
  win: [[523, 523, 0.14, 'triangle'], [659, 659, 0.14, 'triangle', 0.12], [784, 784, 0.3, 'triangle', 0.24]],
  lose: [[392, 392, 0.2, 'triangle'], [330, 330, 0.2, 'triangle', 0.18], [262, 262, 0.4, 'triangle', 0.36]], draw: [[440, 440, 0.3, 'triangle']],
  decide: [[880, 1320, 0.18, 'sine']], tap: [[340, 520, 0.07, 'triangle']],
};
const lastSound = {};
function sound(name, amount) {
  if (!PREF.sound || !SOUNDS[name] || beatScale() === 0 && name !== 'decide') return;
  const now = performance.now();
  if (now - (lastSound[name] || 0) < 90) return;  // a 15-creature attack is one sound, not fifteen
  lastSound[name] = now;
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
    const vol = 0.06 * (PREF.volume ?? 0.6) * (name === 'hit' ? Math.min(2, 0.7 + (amount || 1) / 4) : 1);
    for (const [f0, f1, dur, type, delay = 0] of SOUNDS[name]) {
      const t = audio.currentTime + delay, o = audio.createOscillator(), g = audio.createGain();
      o.type = type; o.frequency.setValueAtTime(f0, t); o.frequency.exponentialRampToValueAtTime(Math.max(30, f1), t + dur);
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(vol, t + 0.015); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      o.connect(g).connect(audio.destination); o.start(t); o.stop(t + dur + 0.02);
    }
  } catch (e) { /* no audio */ }
}
let audio = null;
function ping() { sound('decide'); }

// ---------------------------------------------------------------- post-game review
// The finished game, omniscient: the same board read-only with a timeline,
// both hands, the bot's options with probabilities and its value at each of
// its decisions, and a draw-quality strip (was it a misplay, or flood/screw?).
// The server opens it only once the game is over and only with the game's
// token (handed to the player with the new game).
async function openReview() {
  if (!S.gid || !S.over) { toast('The review opens when the game is over.'); return; }
  if (S.paying) payCancel(true);
  let rep;
  try { rep = await api(`${encodeURIComponent(S.gid)}/review`); } catch (e) { toast(e.message); return; }
  Object.assign(S.cards, rep.cards);
  S.rv = {rep, back: {raw: S.raw, shown: S.shown, ui: S.ui, prev: S.prev}, i: 0, drawq: false};
  S.rv.a = rvAnalyse(rep);
  S.raw = rep.frames; S.ui = null; S.prev = null;
  document.body.classList.add('reviewing');
  closePop(); closeDrawer(); toggleLog(false);
  rvBuild();
  const first = S.rv.a.bot[0] ?? 0;
  rvGo(first);
}
function closeReview() {
  if (!S.rv) return;
  const b = S.rv.back;
  S.raw = b.raw; S.ui = b.ui; S.prev = null; S.rv = null;
  document.body.classList.remove('reviewing');
  $$('#revbar, #revpanel, #drawq').forEach(x => x.remove());
  render(b.shown, {noAnim: true}); renderDock();
}
function rvAnalyse(rep) {
  const frames = rep.frames, seat = rep.review.seat, bot = rep.review.bot;
  const isLand = n => (S.cards[n]?.types || []).includes('Land');
  const isSource = c => c.types.includes('Land') || (S.cards[c.name]?.mana && !S.cards[c.name]?.mana_sac && !(c.power != null && c.sick));
  const seen = [new Set(), new Set()], turns = new Map(), turnStart = new Map();
  const row = t => { if (!turns.has(t)) turns.set(t, {t, active: null, p: [0, 1].map(() => ({drawnL: 0, drawnS: 0, handLands: null, hand: null, bfLands: null, played: 0, spent: 0, avail: null}))}); return turns.get(t); };
  frames.forEach((f, i) => {
    const st = f.state, t = st.turn, r = row(t);
    if (!turnStart.has(t)) turnStart.set(t, i);
    if (r.active == null && t > 0) r.active = st.active;
    for (const p of [0, 1]) {
      const P = st.players[p], pr = r.p[p];
      if (pr.handLands == null) {
        pr.handLands = P.hand.filter(c => isLand(c.name)).length; pr.hand = P.hand.length;
        pr.bfLands = st.battlefield.filter(c => c.controller === p && c.types.includes('Land')).length;
        pr.avail = st.battlefield.filter(c => c.controller === p && isSource(c)).length;
      }
      for (const c of P.hand) if (!seen[p].has(c.uid) && t > 0) { if (isLand(c.name)) pr.drawnL++; else pr.drawnS++; }
      for (const c of P.hand) seen[p].add(c.uid);
      for (const z of ['graveyard', 'exile']) for (const c of P[z]) seen[p].add(c.uid);
      for (const c of st.battlefield) if (c.controller === p) seen[p].add(c.uid);
    }
    for (const e of f.events) {
      let m = /^p(\d) plays /.exec(e); if (m) r.p[+m[1]].played++;
      m = /^ {2}p(\d) pay_mana:/.exec(e); if (m) r.p[+m[1]].spent++;
    }
  });
  const rows = [...turns.values()].filter(r => r.t > 0).sort((a, b) => a.t - b.t);
  const last = frames.at(-1).state;
  const flags = [0, 1].map(p => {
    const out = [], mull = last.players[p].mulligans || 0, mine = rows.filter(r => r.active === p);
    if (mull) out.push({cls: 'bad', text: `mulligan to ${7 - mull}`});
    const flood = mine.find(r => r.p[p].handLands >= 5);
    if (flood) out.push({cls: 'bad', text: `flooded (turn ${flood.t}: ${flood.p[p].handLands} lands in hand)`});
    const missed = mine.filter(r => r.p[p].played === 0 && r.p[p].handLands === 0 && r.p[p].bfLands < 5);
    if (missed.length >= 2) out.push({cls: 'bad', text: `screwed (missed ${missed.length} land drops)`});
    else if (missed.length === 1) out.push({cls: '', text: `missed a land drop (turn ${missed[0].t})`});
    const L = rows.reduce((a, r) => a + r.p[p].drawnL, 0), Sp = rows.reduce((a, r) => a + r.p[p].drawnS, 0);
    out.push({cls: '', text: `drew ${L} land${L === 1 ? '' : 's'}, ${Sp} spell${Sp === 1 ? '' : 's'}`});
    return out;
  });
  // the bot's decisions: probabilities, value before -> after, surprises and swings
  const botIdx = [];
  frames.forEach((f, i) => { const d = f.decision; if (d && d.player === bot && d.policy && d.options.length > 1) botIdx.push(i); });
  const allBot = [];
  frames.forEach((f, i) => { const d = f.decision; if (d && d.player === bot && d.value != null) allBot.push(i); });
  const w = rep.meta.winner, final = w == null ? 0 : w === bot ? 1 : -1;
  const info = new Map();
  for (const i of botIdx) {
    const d = frames[i].decision, k = allBot.indexOf(i);
    const v0 = d.value, v1 = k + 1 < allBot.length ? frames[allBot[k + 1]].decision.value : final;
    const pc = d.policy[d.chosen] ?? 0;
    info.set(i, {v0, v1, dv: v1 - v0, pc, surprising: pc < 0.15, swing: Math.abs(v1 - v0) >= 0.35});
  }
  return {rows, flags, turnStart, bot: botIdx, info, seat, botSeat: bot};
}
function rvBuild() {
  const rv = S.rv, n = rv.rep.frames.length, a = rv.a;
  const bar = document.createElement('div');
  bar.id = 'revbar';
  const x = i => `${(i / Math.max(1, n - 1)) * 100}%`;
  const ticks = [...a.turnStart.entries()].filter(([t]) => t > 0).map(([t, i]) => `<span class="tturn" style="left:${x(i)}" data-go="${i}">T${t}</span>`).join('');
  const marks = a.bot.map(i => { const f = a.info.get(i); return `<span class="tdec ${f.surprising ? 'surp' : ''} ${f.swing ? (f.dv < 0 ? 'drop' : 'rise') : ''}" style="left:${x(i)}" data-go="${i}" title="${esc(`turn ${rv.rep.frames[i].state.turn}: ${clean(rv.rep.frames[i].decision.options[rv.rep.frames[i].decision.chosen])}${f.surprising ? ' · surprising (' + Math.round(f.pc * 100) + '%)' : ''}${f.swing ? ' · value ' + (f.dv > 0 ? '+' : '') + f.dv.toFixed(2) : ''}`)}"></span>`; }).join('');
  const meta = rv.rep.meta, w = meta.winner;
  bar.innerHTML = `<div class="rvhead"><b>Review</b><span>${w == null ? 'Draw' : w === S.seat ? 'You won' : 'You lost'} · turn ${meta.turns}</span>
      <button class="btn small" data-rv="prevdec" title="Previous bot decision ([)">◀ bot</button><button class="btn small" data-rv="prev" title="Back (←; Shift+← a turn)">◀</button>
      <button class="btn small" data-rv="next" title="Forward (→; Shift+→ a turn)">▶</button><button class="btn small" data-rv="nextdec" title="Next bot decision (])">bot ▶</button>
      <span class="rvflags">${[0, 1].map(p => `<span class="who">${p === S.seat ? 'You' : 'Bot'}:</span>${a.flags[p].map(f => `<span class="chip ${f.cls}">${esc(f.text)}</span>`).join('')}`).join('')}</span>
      <span class="sp"></span><button class="btn small" data-rv="drawq">Draw quality</button>
      ${rv.rep.review.replay ? `<a class="btn small" href="../#r=${encodeURIComponent(rv.rep.review.replay)}" target="_blank" title="The full debug view of this game">Replay viewer</a>` : ''}
      <button class="btn small" data-rv="exit">Exit (Esc)</button></div>
    <div class="track" id="rvtrack"><div class="line"></div>${ticks}${marks}<span class="cursor" id="rvcur"></span></div>
    <div class="dqstrip">${[S.seat, a.botSeat].map(p => `<div class="dqrow"><span class="dqwho">${p === S.seat ? 'You' : 'Bot'}</span>${dqCells(p, x)}</div>`).join('')}</div>
    <div class="legend"><span class="tdec"></span>bot decision <span class="tdec surp"></span>surprising pick <span class="tdec drop"></span>value drop <span class="tdec rise"></span>value rise · ←/→ step, Shift+←/→ turn, [ ] bot decisions</div>`;
  $('#app').insertBefore(bar, $('#board'));
  const panel = document.createElement('div');
  panel.id = 'revpanel';
  $('#board').appendChild(panel);
  const dq = document.createElement('div');
  dq.id = 'drawq';
  $('#board').appendChild(dq);
  bar.addEventListener('click', e => {
    const g = e.target.closest('[data-go]');
    if (g) return rvGo(+g.dataset.go);
    const b = e.target.closest('[data-rv]');
    if (b) return rvCmd(b.dataset.rv);
    const tr = e.target.closest('#rvtrack');
    if (tr) { const r = tr.getBoundingClientRect(); rvGo(Math.round((e.clientX - r.left) / r.width * (n - 1))); }
  });
  panel.addEventListener('click', e => {
    const b = e.target.closest('[data-rvflag]');
    if (b) openFlagForm(b.dataset.rvflag, null, S.rv.i);
  });
  dq.addEventListener('click', e => { const r = e.target.closest('[data-turn]'); if (r) rvGo(S.rv.a.turnStart.get(+r.dataset.turn)); });
  requestAnimationFrame(() => { fitRows(); fitHand(); });
}
// The draw-quality strip: per turn and player, aligned with the timeline's turns.
function dqCells(p, x) {
  const a = S.rv.a, starts = [...a.turnStart.entries()].filter(([t]) => t > 0).sort((u, v) => u[1] - v[1]), n = S.rv.rep.frames.length;
  return starts.map(([t, i], k) => {
    const r = a.rows.find(z => z.t === t); if (!r) return '';
    const q = r.p[p], own = r.active === p, end = k + 1 < starts.length ? starts[k + 1][1] : n - 1;
    const missed = own && !q.played && !q.handLands && q.bfLands < 5, flood = q.handLands >= 5;
    const drawn = [q.drawnL ? `<b class="ld">${q.drawnL}L</b>` : '', q.drawnS ? `<b class="sp">${q.drawnS}S</b>` : ''].join('') || '–';
    const tip = `Turn ${t} (${own ? 'own turn' : 'their turn'}): drew ${q.drawnL} land(s), ${q.drawnS} spell(s); ${q.handLands} land(s) of ${q.hand} cards in hand; ${q.bfLands} land(s) in play; ${own ? (q.played ? 'land drop made' : 'no land drop') + '; ' : ''}mana spent ${q.spent} of ${q.avail} source(s)`;
    return `<span class="dqc ${own ? 'own' : ''} ${missed || flood ? 'warn' : ''}" style="left:${x(i)};width:calc(${x(end)} - ${x(i)})" title="${esc(tip)}" data-go="${i}">${drawn}${own ? `<i>${q.played ? '✓' : '✗'}</i>` : ''}${flood ? '<i>flood</i>' : ''}</span>`;
  }).join('');
}
function rvCmd(c) {
  const rv = S.rv, n = rv.rep.frames.length, a = rv.a;
  if (c === 'exit') return closeReview();
  if (c === 'prev') return rvGo(rv.i - 1);
  if (c === 'next') return rvGo(rv.i + 1);
  if (c === 'prevdec') { const j = [...a.bot].reverse().find(i => i < rv.i); return rvGo(j ?? rv.i); }
  if (c === 'nextdec') { const j = a.bot.find(i => i > rv.i); return rvGo(j ?? rv.i); }
  if (c === 'prevturn' || c === 'nextturn') {
    const t = rv.rep.frames[rv.i].state.turn, starts = [...a.turnStart.entries()].sort((x, y) => x[1] - y[1]);
    const j = c === 'nextturn' ? starts.find(([tt]) => tt > t)?.[1] : (rv.i > a.turnStart.get(t) ? a.turnStart.get(t) : starts.filter(([tt]) => tt < t).at(-1)?.[1]);
    return rvGo(j ?? (c === 'nextturn' ? n - 1 : 0));
  }
  if (c === 'drawq') { rv.drawq = !rv.drawq; return rvGo(rv.i); }
}
function rvKey(e) {
  const k = e.key;
  const map = {ArrowLeft: e.shiftKey ? 'prevturn' : 'prev', ArrowRight: e.shiftKey ? 'nextturn' : 'next', '[': 'prevdec', ']': 'nextdec', Escape: 'exit', q: 'drawq', Q: 'drawq'};
  if (map[k]) { e.preventDefault(); rvCmd(map[k]); return; }
  if (k === 'Home') { e.preventDefault(); rvGo(0); }
  if (k === 'End') { e.preventDefault(); rvGo(S.rv.rep.frames.length - 1); }
}
function rvGo(i) {
  const rv = S.rv; if (!rv) return;
  const n = rv.rep.frames.length;
  i = Math.max(0, Math.min(n - 1, i));
  rv.i = i;
  render(i, {noAnim: true});
  const cur = $('#rvcur'); if (cur) cur.style.left = `${(i / Math.max(1, n - 1)) * 100}%`;
  rvPanel(); rvDrawq();
}
function rvDecisionText(i) {
  const d = S.rv?.rep.frames[i]?.decision;
  return d ? ` <b>Decision:</b> ${esc(d.kind)}, chose “${esc(clean(d.options[d.chosen]))}”.` : '';
}
const pct = p => `${Math.round(p * 1000) / 10}%`;
function rvPanel() {
  const rv = S.rv, f = rv.rep.frames[rv.i], st = f.state, d = f.decision, a = rv.a, el = $('#revpanel');
  let h = `<div class="rph">Turn ${st.turn} · ${esc(STEP_LABEL[st.step] || st.step || '')} <span class="muted">frame ${rv.i + 1}/${rv.rep.frames.length}</span></div>`;
  if (!d) h += `<div class="muted">The game ended here.</div>`;
  else {
    const isBot = d.player === a.botSeat, info = a.info.get(rv.i);
    h += `<div class="rpk">${isBot ? 'Bot' : 'You'} · ${esc(d.kind.replace(/_/g, ' '))}</div>${d.prompt ? `<div class="muted">${esc(clean(d.prompt))}</div>` : ''}`;
    if (isBot && d.policy) {
      const order = d.options.map((o, k) => k).sort((x, y) => (d.policy[y] ?? 0) - (d.policy[x] ?? 0));
      const shown = order.slice(0, 5);
      if (!shown.includes(d.chosen)) shown.push(d.chosen);
      h += `<div class="probs">${shown.map(k => `<div class="prob ${k === d.chosen ? 'chosen' : ''}"><div class="pl">${k === d.chosen ? '✓ ' : ''}${esc(clean(d.options[k]))}</div><div class="pb"><i style="width:${Math.max(1, (d.policy[k] ?? 0) * 100)}%"></i><span>${pct(d.policy[k] ?? 0)}</span></div></div>`).join('')}${d.options.length > shown.length ? `<div class="muted">+${d.options.length - shown.length} more option${d.options.length - shown.length > 1 ? 's' : ''}</div>` : ''}</div>`;
      if (info) {
        const vb = v => `<span class="vbar"><i style="left:${(v + 1) * 50}%"></i></span>`;
        h += `<div class="vals"><div>Value before ${vb(info.v0)} <b>${info.v0.toFixed(2)}</b></div><div>after ${vb(info.v1)} <b>${info.v1.toFixed(2)}</b> <span class="${info.dv < 0 ? 'neg' : 'pos'}">(${info.dv > 0 ? '+' : ''}${info.dv.toFixed(2)})</span></div>
          <div class="muted">The bot's own estimate of its result (−1 loss … +1 win), at this and at its next decision.</div></div>`;
        if (info.surprising) h += `<div class="badge surp">Surprising: picked a ${pct(info.pc)} option (the bot samples its moves)</div>`;
        if (info.swing) h += `<div class="badge ${info.dv < 0 ? 'drop' : 'rise'}">Big value ${info.dv < 0 ? 'drop' : 'rise'} by its next decision</div>`;
      }
    } else if (isBot) {
      h += `<div class="probs"><div class="prob chosen"><div class="pl">✓ ${esc(clean(d.options[d.chosen]))}</div></div></div><div class="muted">${d.options.length === 1 ? 'Its only option.' : 'No policy recorded for this decision.'}</div>`;
    } else if (d.chosen != null) h += `<div class="probs"><div class="prob chosen mine"><div class="pl">✓ ${esc(clean(d.options[d.chosen]))}</div></div></div>`;
    h += `<div class="rpb">${isBot ? '<button class="btn small" data-rvflag="bot">⚑ Report: bot played wrong</button>' : ''}<button class="btn small" data-rvflag="bug">⚠ Bug: engine / UI</button></div>`;
  }
  el.innerHTML = h;
}
function rvDrawq() {
  const rv = S.rv, el = $('#drawq');
  el.classList.toggle('on', rv.drawq);
  if (!rv.drawq) return;
  const t = rv.rep.frames[rv.i].state.turn, me = S.seat, bot = rv.a.botSeat;
  const cell = (r, p) => { const x = r.p[p]; const own = r.active === p;
    return `<td class="${own ? 'own' : ''}"><span title="drawn this turn: lands / spells">${x.drawnL ? `<b class="ld">${x.drawnL}L</b>` : ''}${x.drawnS ? `<b class="sp">${x.drawnS}S</b>` : ''}${!x.drawnL && !x.drawnS ? '–' : ''}</span></td>
      <td class="${own ? 'own' : ''} ${x.handLands >= 5 ? 'warn' : ''}" title="lands in hand / cards in hand at the start of the turn">${x.handLands}/${x.hand}</td>
      <td class="${own ? 'own' : ''}" title="lands on the battlefield at the start of the turn">${x.bfLands}</td>
      <td class="${own ? 'own' : ''} ${own && !x.played && !x.handLands ? 'warn' : ''}" title="land drop this turn">${own ? (x.played ? '✓' : '✗') : ''}</td>
      <td class="${own ? 'own' : ''}" title="mana spent / mana sources at the start of the turn">${x.spent}/${x.avail}</td>`; };
  el.innerHTML = `<div class="dqh"><b>Draw quality</b><span class="muted">L lands · S spells drawn; hand lands/cards, lands in play, land drop, mana spent/sources (own turns in bold)</span></div>
    <table><thead><tr><th></th><th colspan="5">You</th><th colspan="5">Bot</th></tr><tr><th>T</th>${'<th>drew</th><th>hand</th><th>play</th><th>drop</th><th>mana</th>'.repeat(2)}</tr></thead>
    <tbody>${rv.a.rows.map(r => `<tr data-turn="${r.t}" class="${r.t === t ? 'cur' : ''}"><td>${r.t}</td>${cell(r, me)}${cell(r, bot)}</tr>`).join('')}</tbody></table>`;
  el.querySelector('tr.cur')?.scrollIntoView({block: 'nearest'});
}

// ---------------------------------------------------------------- game start / end
function finishGame() {
  if (S.paying) payCancel(true);  // a game that ended (a concession, a timeout) leaves no half-played spell
  S.ui = null;
  if (S.raw.length) render(last(), {noAnim: true});
  renderDock();
  renderMeta();
  if (!S.over || S.resultShown === S.gid) return;
  S.resultShown = S.gid;
  const w = S.meta?.winner, win = w === S.seat;
  const banner = document.createElement('div');
  banner.className = `banner ${w == null ? '' : win ? 'win' : 'loss'}`;
  banner.textContent = w == null ? 'DRAW' : win ? 'VICTORY' : 'DEFEAT';
  $('#board').appendChild(banner);
  sound(w == null ? 'draw' : win ? 'win' : 'lose');
  setTimeout(() => { banner.remove(); showResult(); }, beatScale() === 0 ? 300 : 1700);
}

// Both decklists (public in Pauper): click a deck name on a plate.
async function openDecks() {
  if (!S.matchup) return;
  let d;
  try { d = await api(`decks?matchup=${encodeURIComponent(S.matchup)}&seat=${S.seat}`); } catch (e) { toast(e.message); return; }
  const group = (list, types) => {
    const by = {};
    for (const [n, k] of Object.entries(list)) { const t = types[n] || S.cards[n]?.types || []; const g = t.includes('Creature') ? 'Creatures' : t.includes('Land') ? 'Lands' : 'Spells'; (by[g] = by[g] || []).push([n, k]); }
    return Object.entries(by).map(([g, xs]) => `<div class="dgrp"><h4>${g} (${xs.reduce((a, [, k]) => a + k, 0)})</h4>${xs.map(([n, k]) => `<div class="dline" data-name="${esc(n)}"><span>${k}</span> ${esc(n)}</div>`).join('')}</div>`).join('');
  };
  const col = (x, label) => `<div class="dcol"><h3>${label}: ${esc(x.title)}</h3>${group(x.main, x.types || {})}<div class="dgrp side"><h4>Sideboard (15)</h4>${Object.entries(x.side).map(([n, k]) => `<div class="dline" data-name="${esc(n)}"><span>${k}</span> ${esc(n)}</div>`).join('')}</div></div>`;
  const m = $('#modal');
  m.innerHTML = `<div class="mbox decks"><h2>Decklists</h2><div class="dcols">${col(d.mine, 'You')}${col(d.theirs, 'Opponent')}</div>
    <div class="note">Games 2 and 3 are sideboarded: the opponent plays its standard plan for this matchup.</div>
    <div style="margin-top:12px"><button class="primary" id="bDecksClose">Close</button></div></div>`;
  m.classList.add('on');
  $('#bDecksClose').focus();
  $('#bDecksClose').onclick = () => m.classList.remove('on');
}

async function showResult() {
  const m = $('#modal'), meta = S.meta, mm = meta?.match, w = meta?.winner, win = w === S.seat;
  const me = S.seat, op = opp();
  const matchOver = !mm || mm.over;
  let sb = null;
  if (mm && !mm.over) { try { sb = await api(`sideboard?matchup=${encodeURIComponent(S.matchup || '')}&seat=${me}`); } catch (e) { sb = null; } }
  const swaps = sb && Object.keys(sb.in).length ? `<div class="sbplan"><div><b>In</b> ${Object.entries(sb.in).map(([n, k]) => `${k}× ${esc(n)}`).join(', ')}</div><div><b>Out</b> ${Object.entries(sb.out).map(([n, k]) => `${k}× ${esc(n)}`).join(', ')}</div></div>` : '';
  const flags = (S.flags || []).length;
  m.innerHTML = `<div class="mbox result"><div class="note">${mm ? `Game ${mm.game_no} of 3 · ` : ''}turn ${meta?.turns}</div>
    <div class="big ${w == null ? '' : win ? 'win' : 'loss'}">${w == null ? 'Draw' : win ? 'You win' : 'You lose'}</div>
    <div class="note">${esc(reasonText())}</div>
    <div class="stats"><div><b>${S.stats.dealt}</b><span>damage dealt</span></div><div><b>${S.stats.taken}</b><span>damage taken</span></div><div><b>${S.stats.played}</b><span>cards played</span></div><div><b>${flags}</b><span>plays flagged</span></div></div>
    ${mm ? `<div class="matchline">Match: you ${mm.wins[me]} – ${mm.wins[op]} ${esc(agentName(meta.agents[op]))}${matchOver ? (mm.wins[me] > mm.wins[op] ? ' · you win the match' : mm.wins[me] < mm.wins[op] ? ' · they win the match' : '') : ''}</div>` : ''}
    ${!matchOver ? `<form id="fNext" class="nextgame">
        <label>Sideboard</label><span class="seg"><label><input type="radio" name="plan" value="standard" checked>standard plan</label><label><input type="radio" name="plan" value="maindeck">keep the maindeck</label></span>
        ${swaps ? `<div class="full">${swaps}</div>` : ''}
        ${mm.you_choose_play ? `<label>You lost, so you choose</label><span class="seg"><label><input type="radio" name="play" value="1" checked>play first</label><label><input type="radio" name="play" value="0">draw first</label></span>` : `<div class="full note">They lost the last game, so they play first.</div>`}
      </form>` : ''}
    <form id="fSurvey" class="survey"><div class="ph">How was it? (optional, saved with the game)</div>
      <label>The bot played</label><span class="seg">${[1, 2, 3, 4, 5].map(n => `<label><input type="radio" name="strength" value="${n}">${['very weak', 'weak', 'okay', 'strong', 'very strong'][n - 1]}</label>`).join('')}</span>
      <label>Hardest moment</label><input name="hardest" maxlength="1000" placeholder="e.g. their turn-6 Counterspell">
      <span></span><button class="btn small" type="submit">Send feedback</button></form>
    <div class="rbtns">${S.replay ? `<a class="btn" href="../#r=${encodeURIComponent(S.replay)}" target="_blank">Full replay (both hands)</a>` : ''}
      <button class="btn" id="bLook">Look at the board</button><button class="btn" id="bReview" title="Both hands, the bot's options with probabilities and its value, draw quality">Review game</button><button class="btn" id="bChange">Change opponent or decks</button>
      <button class="primary" id="bAgain">${matchOver ? 'Rematch' : `Next game (${mm.game_no + 1} of 3)`}</button></div></div>`;
  m.classList.add('on');
  $('#bAgain').focus();
  $('#bAgain').onclick = async () => {
    const f = $('#fNext');
    const req = f ? {plan: f.plan.value, play: f.play ? f.play.value === '1' : true} : {};
    $('#bAgain').disabled = true;
    try { const v = await api(`${encodeURIComponent(S.gid)}/next`, req); m.classList.remove('on'); S.gid = null; applyView(v); pump(); }
    catch (e) { toast(e.message); $('#bAgain').disabled = false; }
  };
  $('#bChange').onclick = () => openNewGame();
  $('#fSurvey').onsubmit = async e => {
    e.preventDefault();
    const f = e.target, st = f.strength.value;
    try { await api(`${encodeURIComponent(S.gid)}/survey`, {strength: st ? +st : null, hardest: f.hardest.value, pseudonym: pseudonym()}); toast('Thanks for the feedback.'); f.querySelector('button').disabled = true; }
    catch (err) { toast(err.message); }
  };
  $('#bLook').onclick = () => m.classList.remove('on');
  $('#bReview').onclick = () => { m.classList.remove('on'); openReview(); };
}

async function openNewGame() {
  const m = $('#modal');
  m.innerHTML = `<div class="mbox"><h2>Play against a model</h2><div class="note">Loading…</div></div>`;
  m.classList.add('on');
  let opt;
  try { opt = await api('options'); } catch (e) {
    m.innerHTML = `<div class="mbox"><h2>Live play is off</h2><div class="err">${esc(e.message)}</div><p class="note">Start the server with <code>python -m mtg_ml.replay serve --models DIR</code> (add <code>--dev</code> for scenarios and the scripted bot).</p></div>`;
    return;
  }
  S.filing = opt.filing;
  if (opt.mode === 'play') return newGamePlay(m, opt);
  const last = loadJSON('mtgml-play-last', {});
  const mus = Object.entries(opt.matchups);
  m.innerHTML = `<div class="mbox"><h2>Play against a model</h2><form id="fNew">
    ${Object.keys(opt.scenarios || {}).length ? `<label>Dev scenario</label><select name="scenario"><option value="">none (a normal game)</option>${Object.entries(opt.scenarios).map(([k, t]) => `<option value="${esc(k)}">${esc(t)}</option>`).join('')}</select>` : ''}
    <label>Opponent</label><select name="model">${opt.models.map(n => `<option ${n === last.model ? 'selected' : ''}>${esc(n)}</option>`).join('')}</select>
    <label>Matchup</label><select name="matchup">${mus.map(([k, v]) => `<option value="${esc(k)}" ${k === (last.matchup || 'jund_blue') ? 'selected' : ''}>${esc(v[0])} vs ${esc(v[1])}</option>`).join('')}</select>
    <label>You play</label><span class="seg" id="segSeat"></span>
    <label></label><label class="note"><input type="checkbox" name="greedy" ${last.greedy ? 'checked' : ''}> greedy model (always its most likely move)</label>
    <div class="full" style="display:flex;gap:10px;align-items:center;margin-top:8px"><button class="primary" type="submit">Start game</button>${S.gid && !S.over ? '<button class="btn" type="button" id="bCancel">Back to the game</button>' : ''}<span class="err" id="newErr"></span></div>
    </form><p class="note" style="margin-top:14px">Space passes or confirms · drag cards to play them (or double-click) · R passes until the opponent acts · F full control · O all options</p></div>`;
  const f = $('#fNew');
  const seats = () => { const v = opt.matchups[f.matchup.value]; $('#segSeat').innerHTML = v.map((d, i) => `<label><input type="radio" name="seat" value="${i}" ${i === (last.seat || 0) ? 'checked' : ''}>${esc(d)}</label>`).join(''); };
  seats(); f.matchup.onchange = seats;
  if ($('#bCancel')) $('#bCancel').onclick = () => m.classList.remove('on');
  if (!opt.models.length) $('#newErr').textContent = 'No checkpoints found in --models.';
  f.onsubmit = async e => {
    e.preventDefault();
    const req = {model: f.model.value, matchup: f.matchup.value, seat: +(f.seat.value || 0), greedy: f.greedy.checked};
    S.matchup = req.matchup;
    if (f.scenario && f.scenario.value) req.scenario = f.scenario.value;
    saveJSON('mtgml-play-last', req);
    const b = f.querySelector('button[type=submit]'); b.disabled = true; b.textContent = 'Starting…';
    try {
      const v = await api('new', req);
      m.classList.remove('on');
      S.gid = null;
      applyView(v);
      pump();
    } catch (err) { $('#newErr').textContent = err.message; b.disabled = false; b.textContent = 'Start game'; }
  };
}

// The normal new-game screen: the decks and opponents this server offers (mtg_ml/play_config.toml).
function newGamePlay(m, opt) {
  const last = loadJSON('mtgml-play-pick', {});
  let deck = opt.player_decks.some(d => d.deck === last.deck && d.opponents.length) ? last.deck : opt.player_decks.find(d => d.opponents.length)?.deck;
  let foe = last.opponent;
  const draw = () => {
    const mine = opt.player_decks.find(d => d.deck === deck);
    if (!mine || !mine.opponents.includes(foe)) foe = mine?.opponents[0];
    m.innerHTML = `<div class="mbox newgame"><h2>New game</h2>
      <div class="ngsec"><h3>Your deck</h3><div class="ngopts">${opt.player_decks.map(d => `<button type="button" class="ngopt ${d.deck === deck ? 'sel' : ''}" data-deck="${esc(d.deck)}" ${d.opponents.length ? '' : 'disabled'}><b>${esc(d.title)}</b><span>${d.opponents.length ? 'Best of three, sideboarded games 2 and 3' : 'No opponent for this deck yet'}</span></button>`).join('')}</div></div>
      <div class="ngsec"><h3>Opponent</h3><div class="ngopts">${opt.opponents.map(o => `<button type="button" class="ngopt ${o.id === foe ? 'sel' : ''}" data-opp="${esc(o.id)}" ${mine?.opponents.includes(o.id) ? '' : 'disabled'}><b>${esc(o.label)}</b><span>${esc(o.note || `The model playing ${o.deck_title}`)}</span></button>`).join('') || '<div class="note">No opponent is available on this server.</div>'}</div></div>
      <div style="display:flex;gap:10px;align-items:center"><button class="primary" id="bStart" ${deck && foe ? '' : 'disabled'}>Start game</button>${S.gid && !S.over ? '<button class="btn" type="button" id="bCancel">Back to the game</button>' : ''}<span class="err" id="newErr"></span></div>
      <p class="note" style="margin-top:14px">Space passes or confirms · drag cards to play them (or double-click) · click a land to tap it for mana · ? shows every control</p></div>`;
    m.querySelectorAll('[data-deck]').forEach(b => b.onclick = () => { deck = b.dataset.deck; draw(); });
    m.querySelectorAll('[data-opp]').forEach(b => b.onclick = () => { foe = b.dataset.opp; draw(); });
    if ($('#bCancel')) $('#bCancel').onclick = () => m.classList.remove('on');
    const b = $('#bStart');
    b.focus();
    b.onclick = async () => {
      b.disabled = true; b.textContent = 'Starting…';
      saveJSON('mtgml-play-pick', {deck, opponent: foe});
      try { const v = await api('new', {deck, opponent: foe}); m.classList.remove('on'); S.gid = null; applyView(v); pump(); }
      catch (err) { $('#newErr').textContent = err.message; b.disabled = false; b.textContent = 'Start game'; }
    };
  };
  draw();
}

function openSettings() {
  const m = $('#modal');
  setTimeout(() => m.querySelector('input, button')?.focus(), 0);
  m.innerHTML = `<div class="mbox"><h2>Settings</h2><form id="fSet">
    <label>Opponent replay speed</label><span class="seg">${Object.keys(SPEEDS).map(k => `<label><input type="radio" name="speed" value="${k}" ${PREF.speed === k ? 'checked' : ''}>${k === 'instant' ? 'instant' : k + '×'}</label>`).join('')}</span>
    <label>Paying</label><label class="note"><input type="checkbox" name="autoPayNoAsk" ${PREF.autoPayNoAsk ? 'checked' : ''}> Auto-pay without asking (cast at once; default: a paying step)</label>
    <label>Opponent's choices</label><span class="seg">${[[1500, 'short'], [2800, 'normal'], [4500, 'long']].map(([v, l]) => `<label><input type="radio" name="spotMs" value="${v}" ${+PREF.spotMs === v ? 'checked' : ''}>${l}</label>`).join('')}</span>
    <label>Attacks</label><label class="note"><input type="checkbox" name="confirmEmptyAttack" ${PREF.confirmEmptyAttack ? 'checked' : ''}> confirm "No attacks" when creatures could attack</label>
    <label>Targets</label><label class="note"><input type="checkbox" name="autoTarget" ${PREF.autoTarget ? 'checked' : ''}> pick the only legal target automatically</label>
    <label>Sound</label><span><label class="note"><input type="checkbox" name="sound" ${PREF.sound ? 'checked' : ''}> sounds on</label> <input type="range" name="volume" min="0" max="1" step="0.1" value="${PREF.volume ?? 0.6}" title="Volume"></span>
    <label>Priority</label><label class="note"><input type="checkbox" name="full" ${S.fullControl ? 'checked' : ''}> full control (never auto-pass; F)</label>
    <div class="full note">Auto-pass stops: click the small bars under each step in the phase rail (gold: your turn, blue: the opponent's).</div>
    <div class="full" style="margin-top:8px"><button class="primary" type="submit">Done</button></div></form></div>`;
  m.classList.add('on');
  const f = $('#fSet');
  f.onsubmit = e => {
    e.preventDefault();
    PREF.speed = f.speed.value; PREF.autoPayNoAsk = f.autoPayNoAsk.checked; PREF.spotMs = +f.spotMs.value; PREF.confirmEmptyAttack = f.confirmEmptyAttack.checked;
    PREF.autoTarget = f.autoTarget.checked; PREF.sound = f.sound.checked; PREF.volume = +f.volume.value; S.fullControl = f.full.checked;
    savePref(); m.classList.remove('on'); renderDock();
  };
}

async function resync() {
  if (!S.gid) return openNewGame();
  const gid = S.gid;
  try {
    const v = await api(encodeURIComponent(gid) + '?since=0');
    S.gid = null; applyView(v);
    S.queue = []; S.logDone = 0; $('#log').innerHTML = '';
    for (let i = 0; i < S.raw.length; i++) appendLog(i);
    S.lastMine = S.raw.length - 1;
    render(last(), {noAnim: true});
    pump();
  } catch (e) { S.error = e.message; S.errorKind = e.kind || ''; renderDock(); }
}

(async function init() {
  const h = new URLSearchParams(location.hash.slice(1));
  api('options').then(o => { S.filing = o.filing; }).catch(() => {});  // whether reports are published
  if (h.get("g") && !gameToken(h.get("g"))) toast("That game was started in another browser: its key is not here.");
  else if (h.get("g")) {
    S.gid = h.get("g");
    try { await resync(); if (!S.error) return; } catch (e) { /* fall through */ }
    if (S.errorKind === 'auth') return;  // the sign-in banner says what to do
    if (S.errorKind === 'unrecoverable') { renderDock(); return; }  // say so; "New game" starts another
    S.error = null; S.errorKind = ''; S.gid = null; toast("Your last game is no longer on the server (it expired or the server lost it).");
  }
  openNewGame();
})();

// For tests and debugging: the state and a way to read the current decision.
window.PLAY = {
  S, myDecision, act, rvGo,
  info: () => ({ready: canAct() && !S.queue.length && !inputLocked(), over: S.over, busy: S.busy || S.pumping, error: S.error, frame: last(), kind: S.ui?.kind || null,
    options: myDecision()?.options || [], overlay: $('#overlay').classList.contains('on'), turn: S.shown >= 0 ? stateAt(S.shown).turn : 0}),
};
