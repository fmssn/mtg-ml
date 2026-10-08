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
  speed: '1', manualPay: false, confirmEmptyAttack: true, sound: true, autoTarget: false,
  stops: {me: {main1: true, main2: true}, opp: {declare_blockers: true, end: true}}, stopsVersion: 2,
}, loadJSON('mtgml-play-pref', {}));
function loadJSON(k, d) { try { return JSON.parse(localStorage.getItem(k) || 'null') ?? d; } catch (e) { return d; } }
function saveJSON(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private window */ } }
const savePref = () => saveJSON('mtgml-play-pref', PREF);
if ((PREF.stopsVersion || 1) < 2) { PREF.stops.opp.declare_blockers = true; PREF.stopsVersion = 2; savePref(); }  // new default: the after-blockers window

// ---------------------------------------------------------------- game state
const S = {
  gid: null, seat: 0, raw: [], cards: {}, meta: null, over: false, replay: null,
  shown: -1, prev: null, fx: {life: {}, dmg: {}, flash: {}, enter: {}}, queue: [], pumping: false, busy: false, error: null, skip: false,
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
async function api(path, body) {
  const r = await fetch(API + path, body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  let j; try { j = await r.json(); } catch (e) { throw new Error(`server answered ${r.status}`); }
  if (!r.ok || j.error) throw new Error(j.error || `server answered ${r.status}`);
  return j;
}

function applyView(view) {
  const L = view.live;
  if (S.gid !== L.id) resetGame(L.id);
  Object.assign(S, {seat: L.seat, over: L.over, replay: L.replay, meta: view.meta});
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
  Object.assign(S, {gid: id, lastPriority: -1, raw: [], shown: -1, prev: null, fx: {life: {}, dmg: {}, flash: {}, enter: {}}, queue: [], plan: null, ui: null, lastMine: -1, passMode: null, lastOpp: '',
    fullControl: false, feed: [], logDone: 0, error: null, skip: false});
  $('#log').innerHTML = '';
  renderFeed();
}

async function send(index) {
  const frame = last();
  S.busy = true; S.error = null; S.ui = null; closePop(); closeOverlay(); renderDock();
  if (S.shown >= 0) render(S.shown, {noAnim: true});  // drop the highlights of the decision just made
  try {
    const kind = S.raw[frame].decision.kind;
    const v = await api(`${encodeURIComponent(S.gid)}/choose`, {frame, index, since: frame});
    S.lastMine = frame;
    if (kind === 'priority') S.lastPriority = frame;
    applyView(v);
  } catch (e) {
    S.error = e.message;
  } finally { S.busy = false; }
}

// Pick an option (from a gesture) and run the game on to the next prompt.
function stamp(plan) {  // scope a plan to the current turn and step
  const s = stateAt(last());
  return plan && {...plan, turn: s.turn, step: s.step};
}
async function act(index, plan, frame) {
  if (S.busy || S.pumping || !myDecision()) return;
  if (frame != null && frame !== last()) return;  // made for a decision that is gone
  if (plan !== undefined) S.plan = stamp(plan);
  S.feed = [];
  renderFeed();
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
      let a = null;
      try { a = autoAnswer(d); } catch (e) { console.error(e); a = null; }
      if (a == null || autos > 400) { S.plan = null; S.skip = false; S.passMode = null; enterDecision(true); break; }
      autos++;
      await send(a);
    }
  } finally { S.pumping = false; renderDock(); }
}

// ---------------------------------------------------------------- automation (P2, P6, P7 of the brief)
const isReal = r => r.type !== 'pass' && r.type !== 'mana';
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
  if (d.kind === 'pay_mana' && !PREF.manualPay) return autoPay(d);
  if (d.kind === 'priority') {
    const pass = refs.findIndex(r => r.type === 'pass');
    if (pass < 0) return null;
    if (S.fullControl) return null;
    if (!refs.some(isReal)) return pass;  // mana abilities alone never stop
    const s = stateAt(last());
    const top = s.stack[s.stack.length - 1];
    const acted = oppActedSince(S.lastMine);
    if (top && top.controller !== S.seat) return null;  // something of theirs to answer
    if (acted) return null;  // never pass right after they did something
    if (top) return pass;  // my own spell: let it resolve (they may still respond)
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
    if (s.step === 'declare_blockers' && firstInStep && s.battlefield.some(c => c.attacking) && (!mine || blocked)) return null;
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
    const i = refs.findIndex(r => (t.oid != null && r.oid === t.oid) || (t.player != null && r.player === t.player));
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
  const prevD = i > 0 ? S.raw[i - 1].decision : null;
  if (prevD && prevD.player === opp()) feedFromDecision(prevD, i - 1);
  for (const a of acts) if (a.p === opp() && ['discard', 'mulligan', 'sacrifice'].includes(a.t)) pushFeed(quietText(a), i, true);
  const notable = acts.some(a => ['cast', 'play', 'activate', 'attack', 'block', 'resolve', 'trigger', 'enter', 'leave', 'dies', 'turn', 'discard', 'hit', 'life'].includes(a.t));
  const isLast = i === last();
  if (!notable && !isLast && i !== 0) { appendLog(i); return; }
  const cast = theirs.find(a => a.t === 'cast' || a.t === 'activate' || a.t === 'plot');
  if (cast && !S.skip && beatScale() > 0) {
    const name = cast.t === 'activate' ? cast.name.split(':')[0].trim() : cast.name;
    await ensureImg(name, 900);
    showSpot(name, cast.t === 'cast' ? `${oppLabel()} casts ${cast.name}` : cast.t === 'plot' ? `${oppLabel()} plots ${cast.name}` : `${oppLabel()} activates ${cast.name}`);
    await wait(950);
    hideSpot();
  }
  // Combat and deaths play out on the board as it was, before the new state lands.
  const fought = await playCombat(acts);
  await playDeaths(acts);
  render(i);
  appendLog(i);
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
// Freshly declared attackers step forward once.
function nudgeAttackers() {
  for (const el of $$('#board .perm.attacking')) el.animate([{translate: '0 0'}, {translate: `0 ${el.closest('#side1') ? 14 : -14}px`}, {translate: '0 0'}], {duration: 380, easing: 'ease-out'});
  sound('attack');
}

function showSpot(name, cap) {
  const el = $('#spot');
  el.innerHTML = cardHtml(name) + `<div class="cap">${esc(cap)}</div>`;
  el.classList.add('on');
}
function hideSpot() { $('#spot').classList.remove('on'); }
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
function feedFromDecision(d, fi) {
  const r = d.refs?.[0] || {};
  if (['pass', 'pay', 'hidden', 'keep'].includes(r.type) || d.kind === 'pay_mana') return;
  if (r.type === 'attack' && r.done) return;
  pushFeed(humanize(d.options[d.chosen ?? 0]), fi, false);
}
function quietText(a) {
  if (a.t === 'discard') return `discards ${a.name}`;
  if (a.t === 'mulligan') return `mulligans (${a.n})`;
  if (a.t === 'sacrifice') return `sacrifices ${a.name}`;
  return a.t;
}
function pushFeed(text, fi, quiet) {
  S.feed.push({text, fi, quiet});
  if (!quiet) S.lastOpp = text;
  if (S.feed.length > 12) S.feed.shift();
  renderFeed();
}
const FLAGS = () => loadJSON('mtgml-play-flags', []);
function isFlagged(fi) { return FLAGS().some(f => f.game === S.gid && f.frame === fi); }
function toggleFlag(fi, text) {
  // Stub for the hosted-play plan (flag + describe): stored locally only.
  let flags = FLAGS();
  if (flags.some(f => f.game === S.gid && f.frame === fi)) flags = flags.filter(f => !(f.game === S.gid && f.frame === fi));
  else { flags.push({game: S.gid, frame: fi, action: text, at: Date.now()}); toast('Flagged. Describing what went wrong comes later.'); }
  saveJSON('mtgml-play-flags', flags);
  renderFeed();
  $$(`#log .flag[data-fi="${fi}"]`).forEach(b => b.classList.toggle('on', isFlagged(fi)));
}
function renderFeed() {
  $('#feed').innerHTML = S.feed.length ? [...S.feed].reverse().map(c => `<div class="fchip ${c.quiet ? 'quiet' : ''}"><span class="txt">${esc(c.text)}</span><button class="flag ${isFlagged(c.fi) ? 'on' : ''}" data-flag="${c.fi}" title="Flag this play as odd">⚑</button></div>`).join('')
    : '<div class="none">Nothing yet</div>';
  const la = $('#plate1 .lastact');
  if (la) la.innerHTML = S.lastOpp ? `<span>Last:</span> ${esc(S.lastOpp)}` : '';
}

// ---------------------------------------------------------------- log
function appendLog(i) {
  if (i < S.logDone) return;
  S.logDone = i + 1;
  const f = S.raw[i], el = $('#log'), out = [];
  for (const line of f.events || []) {
    if (/^-- /.test(line) || / \((only option|auto)\)$/.test(line)) continue;
    if (/^ {2}p\d (priority|pay_mana|declare_attacker|mulligan|order_triggers):/.test(line)) continue;  // the action has a line of its own
    const m = /^=== Turn (\d+): player (\d) ===$/.exec(line);
    if (m) { out.push(`<div class="th">Turn ${m[1]} · ${+m[2] === S.seat ? 'your turn' : "opponent's turn"}</div>`); continue; }
    const hm = /^combat: (.+)#\d+ deals (\d+) damage to (?:p(\d)|(.+)#\d+)$/.exec(line);
    if (hm) {
      const to = hm[3] != null ? (+hm[3] === S.seat ? 'you' : 'the opponent') : hm[4];
      out.push(`<div class="combat">${esc(`${hm[1]} deals ${hm[2]} damage to ${to}`)}</div>`);
      continue;
    }
    const lm = /^life: p(\d) (-?\d+) -> (-?\d+)$/.exec(line);
    if (lm) {
      const d = +lm[3] - +lm[2];
      out.push(`<div class="life ${d < 0 ? 'neg' : 'pos'}">${esc(`${+lm[1] === S.seat ? 'Your' : "Opponent's"} life ${lm[2]} → ${lm[3]} (${d > 0 ? '+' : '−'}${Math.abs(d)})`)}</div>`);
      continue;
    }
    const bm = /^p(\d) blocks: \{(.*)\}$/.exec(line);
    if (bm) {
      const name = o => f.state.battlefield.find(c => c.oid === o)?.name || S.prev?.battlefield.find(c => c.oid === o)?.name || '?';
      const pairs = [...bm[2].matchAll(/(\d+): (\d+)/g)].map(m => `${name(+m[1])} blocks ${name(+m[2])}`);
      out.push(`<div class="${+bm[1] === S.seat ? 'me' : 'opp'}">${esc((+bm[1] === S.seat ? 'You: ' : 'Opp: ') + pairs.join(', '))}</div>`);
      continue;
    }
    const pm = /^\s*p(\d) /.exec(line);
    const who = pm ? (+pm[1] === S.seat ? 'me' : 'opp') : '';
    const text = (who === 'me' ? mine(clean(line.trim())) : clean(line.trim())).replace(/^p(\d) /, (_, p) => +p === S.seat ? 'You: ' : 'Opp: ')
      .replace(/^(\w+): (Pass priority|Done declaring attackers)$/, '$1: $2');
    out.push(`<div class="${who} ${/^\s/.test(line) ? 'sub' : ''}">${esc(text)}</div>`);
  }
  const d = f.decision;
  if (d && d.player === opp() && d.chosen != null && !['pass', 'pay', 'hidden'].includes(d.refs?.[0]?.type) && !d.refs?.[0]?.done) {
    out.push(`<div class="opp sub">${esc('Opp: ' + humanize(d.options[d.chosen]))}<button class="flag ${isFlagged(i) ? 'on' : ''}" data-flag="${i}" data-fi="${i}" title="Flag this play">⚑</button></div>`);
  }
  if (out.length) { el.insertAdjacentHTML('beforeend', out.join('')); el.scrollTop = el.scrollHeight; }
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
  requestAnimationFrame(drawArrows);
  scheduleFxCleanup();
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
  $('#meta').innerHTML = `<b>${esc(m.decks[me])}</b> (you) vs <b>${esc(m.decks[op])}</b> (${esc(agentName(m.agents[op]))}) · ${esc(m.engine)} engine`;
}

function renderPlate(p, s, prev) {
  const P = s.players[p], el = $(`#plate${p === S.seat ? 0 : 1}`);
  const isMe = p === S.seat;
  const pv = prev?.players[p];
  const lf = fx('life', p), delta = lf ? lf.n : 0;
  void pv;
  const pool = Object.entries(P.pool || {}).flatMap(([c, n]) => Array(n).fill(`<i class="${esc(c)}">${esc(c)}</i>`)).join('');
  const d = S.ui?.d;
  const tgt = H.target.has('p' + p);
  const preview = isMe ? blockPreviewLife(s) : attackPreviewLife(s);
  const gyCast = !isMe ? '' : (H.gy.size ? 'castable' : ''), exCast = !isMe ? '' : (H.ex.size ? 'castable' : '');
  el.className = `plate ${s.active === p ? 'active' : ''} ${tgt ? 'targetable' : ''}`;
  el.dataset.player = p;
  el.innerHTML = `
    <div class="who"><span class="dot"></span>${isMe ? 'You' : esc(agentName(S.meta?.agents[p]))}</div>
    <div class="deck">${esc(S.meta?.decks[p] || '')}</div>
    <div class="life ${P.life <= 5 ? 'low' : ''} ${delta < 0 ? 'hit' : delta > 0 ? 'heal' : ''}"${lf ? fxDelay(lf) : ''}>${P.life}${delta ? `<span class="delta ${delta < 0 ? 'neg' : 'pos'}"${fxDelay(lf)}>${delta > 0 ? '+' : ''}${delta}</span>` : ''}</div>
    ${preview != null ? (preview <= 0 ? `<div class="lethal" title="This combat would be lethal if nothing changes">LETHAL</div>` : `<div class="preview-life">→ ${preview} after combat</div>`) : ''}
    <div class="zones">
      <div class="zone" title="Cards in hand">Hand <b>${P.hand.length}</b></div>
      <div class="zone" title="Cards in library${P.library_top_known.length ? '; known on top: ' + esc(P.library_top_known.join(', ')) : ''}">Library <b>${P.library}</b></div>
      <div class="zone clickable ${gyCast}" data-zone="graveyard" data-p="${p}" title="Click to see the graveyard">Grave <b>${P.graveyard.length}</b></div>
      <div class="zone clickable ${exCast}" data-zone="exile" data-p="${p}" title="Click to see exiled cards">Exile <b>${P.exile.length}</b></div>
    </div>
    ${pool ? `<div class="mana" title="Floating mana">${pool}</div>` : ''}
    ${isMe ? '' : `<div class="lastact" title="The opponent's latest action">${S.lastOpp ? `<span>Last:</span> ${esc(S.lastOpp)}` : ''}</div>`}`;
  void d;
}

function groupPerms(perms, s) {
  const hosts = new Set(s.battlefield.filter(c => c.attached_to != null).map(c => c.attached_to));
  // While declaring attackers or blockers each creature is its own card.
  const combat = S.ui && (S.ui.kind === 'declare_attacker' || S.ui.kind === 'declare_blocker');
  const out = [], by = new Map();
  for (const c of perms) {
    const creature = c.power != null;
    if ((creature && combat) || c.attacking || c.blocking != null || c.damage || c.counters || hosts.has(c.oid)) { out.push([c]); continue; }
    const k = [c.name, !!c.tapped, !!c.sick, c.power, c.toughness].join('|');
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
  if (c.tapped) cls.push('tapped');
  if (c.attacking) cls.push('attacking');
  if (c.blocking != null) cls.push('blocking');
  if (n > 1) cls.push('grouped');
  const fEnter = fx('enter', c.uid), fFlash = fx('flash', c.oid), fDmg = fx('dmg', c.oid);
  if (fEnter) cls.push('entered');
  if (fFlash) cls.push('flash');
  void pc;
  if (oids.some(o => H.perm.has(o))) cls.push('activatable');
  if (oids.some(o => H.target.has('o' + o))) cls.push('targetable');
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
  el.innerHTML = P.hand.map(c => {
    const opts = H.hand.get(c.name);
    return cardHtml(c.name, {uid: c.uid, cls: opts ? 'playable' : '', attrs: ' data-hand="1"'});
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
    return `<div class="sitem ${it.controller === S.seat ? 'c-me' : 'c-opp'} ${j === 0 ? 'top' : ''} ${tgt ? 'targetable' : ''}" data-sid="${it.sid}" data-name="${esc(name)}">
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
  for (const o of attackers) {
    if (blocked.has(o)) continue;
    const c = s.battlefield.find(x => x.oid === o), el = permEl(o);
    if (!c || !el) continue;
    h += arrow(el, $(`#plate${c.controller === S.seat ? 1 : 0}`), 'a-red', 'mRed');
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
  S.ui = ui;
  render(fi, {noAnim: S.shown === fi});
  renderDock();
  renderOverlay();
  if (fresh) ping();
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
  priority: '<kbd>R</kbd> pass till they act · <kbd>F</kbd> full control',
  declare_attacker: 'undo: <kbd>right-click</kbd> · <kbd>Esc</kbd> clears',
  declare_blocker: 'undo: <kbd>right-click</kbd> · <kbd>Esc</kbd> clears',
  target: 'the spell is being cast: pick a target',
  pay_mana: '<kbd>Space</kbd> pays automatically',
};
function modeHint(d, ui) {
  if (!d || !ui || S.busy || S.pumping) return S.queue.length ? '<kbd>click</kbd> or <kbd>Space</kbd> skips the replay' : '';
  return HINTS[d.kind] || '<kbd>O</kbd> lists every option';
}

function renderDock() {
  const d = myDecision(), ui = S.ui, P = $('#primary'), pr = $('#prompt'), ch = $('#choices');
  const pill = $('#pill');
  pill.className = S.passMode || S.fullControl ? 'on' : '';
  pill.textContent = S.fullControl ? 'Full control: no auto-pass · F' : S.passMode === 'opp' ? 'Passing until the opponent acts · Esc' : '';
  P.className = 'primary'; P.disabled = true; ch.innerHTML = ''; pr.className = '';
  $('#modehint').innerHTML = modeHint(d, ui);
  $('#bAll').style.visibility = d && ui ? 'visible' : 'hidden';
  if (S.error) {
    const gone = /no such game/i.test(S.error), down = /failed to fetch|networkerror|load failed/i.test(S.error);
    const msg = gone ? 'This game is no longer on the server (the server restarted or the game expired).' : down ? 'Lost the connection to the game server.' : S.error;
    pr.innerHTML = `<span class="k">${gone ? 'Game ended' : 'Problem'}</span><span class="err">${esc(msg)}</span>`;
    ch.innerHTML = gone ? '' : `<button class="choice" data-resync="1">Reload the game from the server</button>`;
    if (gone) { P.textContent = 'New game'; P.dataset.act = 'new'; } else { P.textContent = down ? 'Reconnect' : 'Retry'; P.dataset.act = 'resync'; }
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
  const choiceBtns = (idxs, cls = '') => idxs.map(i => `<button class="choice ${cls}" data-opt="${i}">${esc(clean(d.options[i]))}</button>`).join('');
  P.dataset.act = 'primary';
  switch (d.kind) {
    case 'priority': {
      const playable = d.refs.filter(isReal).length;
      const floating = Object.values(s.players[S.seat].pool || {}).reduce((a, b) => a + b, 0);
      pr.innerHTML = `<span class="k">Your priority · ${esc(STEP_LABEL[s.step] || s.step)}</span>${playable ? 'Drag a glowing card to the battlefield, or double-click it.' : 'Nothing to play.'}`;
      if (H.loose.length) ch.innerHTML = choiceBtns(H.loose);
      P.textContent = ui.armed === 'float' ? `Pass with ${floating} mana floating?` : passLabel(s);
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
      pr.innerHTML = `<span class="k">Pay mana</span>${esc(clean(d.prompt))}. Click a glowing source.`;
      if (H.loose.length) ch.innerHTML = choiceBtns(H.loose);
      P.textContent = 'Auto-pay'; P.disabled = false; P.dataset.act = 'autopay'; P.classList.add('ask');
      break;
    }
    case 'mulligan': {
      const keep = d.refs.findIndex(r => r.type === 'keep');
      pr.innerHTML = `<span class="k">Opening hand</span>${esc(d.prompt)}`;
      P.textContent = clean(d.options[keep]).replace(/^Keep \((\d+) cards\)$/, 'Keep $1'); P.disabled = false; P.dataset.act = 'opt:' + keep; P.classList.add('ask');
      ch.innerHTML = d.refs.map((r, i) => r.type === 'mulligan' ? `<button class="choice" data-opt="${i}">${esc(d.options[i].replace(/^Mulligan \(to (\d+)\)$/, 'Mulligan to $1'))}</button>` : '').join('');
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
const resultText = () => { const w = S.meta?.winner; return w == null ? `Draw (${S.meta?.end_reason || ''})` : w === S.seat ? `You win (${S.meta.end_reason})` : `You lose (${S.meta.end_reason})`; };

// Kinds shown as a card browser: options that name cards.
function overlayKind(d) {
  if (['priority', 'declare_attacker', 'declare_blocker', 'target', 'pay_mana', 'mulligan'].includes(d.kind)) return false;
  if (d.kind === 'assign_damage') return true;
  return d.refs.filter(r => r.name && S.cards[r.name]).length >= Math.max(1, d.refs.length - 1);
}

function renderOverlay() {
  const d = myDecision(), ui = S.ui, el = $('#overlay');
  if (!d || !ui || !overlayKind(d)) { if (!el.dataset.zone) closeOverlay(); return; }
  delete el.dataset.zone;
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
  } else {
    const isCard = r => r.name && S.cards[r.name];
    const tiles = d.refs.map((r, i) => isCard(r) ? `<button class="tile" data-opt="${i}">${cardHtml(r.name)}<span>${esc(clean(d.options[i]))}</span></button>` : '').join('');
    const texts = d.refs.map((r, i) => isCard(r) ? '' : `<button class="choice" data-opt="${i}">${esc(clean(d.options[i]))}</button>`).join('');
    el.innerHTML = `<div class="obox"><h2>${esc(clean(d.prompt))}</h2><div class="sub">${esc(d.kind.replace(/_/g, ' '))} · click one</div><div class="grid">${tiles}</div>${texts ? `<div class="textopts">${texts}</div>` : ''}</div>
      <button class="btn peekbtn" data-cmd="peek">Peek at the board</button>`;
  }
  el.classList.add('on'); el.classList.remove('peek');
}
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
  act(i, plan);
}
function chooseFromCard(el, idxs, plan) {
  if (!idxs.length) return;
  S.srcOid = el.dataset.oid != null ? +el.dataset.oid : null;
  if (idxs.length === 1) act(idxs[0], plan);
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
const mentionsTarget = name => /\btarget\b/i.test(S.cards[name]?.text || '');

let drag = null, suppressClick = false;
document.addEventListener('pointerdown', e => {
  if (e.button !== 0 || !canAct()) return;
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
  if (drag.targets) { const t = targetAt(e.clientX, e.clientY); if (t && t.el.closest('#side1, .plate')) t.el.classList.add('drop-hot'); }
}
function endHandDrag(dg, e) {
  const g = $('#ghost'), field = $('#field0');
  const r = $('#board').getBoundingClientRect();
  const inPlay = e.clientY < r.bottom - r.height * 0.22 && e.clientX < r.right && e.clientX > r.left;
  field.classList.remove('drop', 'hot');
  clearHot();
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
  // skip the bot's replay
  if (S.queue.length && S.pumping && t.closest('#board')) { S.skip = true; return; }
  const flag = t.closest('[data-flag]');
  if (flag) { toggleFlag(+flag.dataset.flag, flag.parentElement.textContent.replace('⚑', '').trim()); return; }
  const stop = t.closest('[data-stop]');
  if (stop) { const [who, st] = stop.dataset.stop.split(':'); PREF.stops[who][st] = !PREF.stops[who][st]; savePref(); renderMid(stateAt(S.shown)); return; }
  if (t.closest('[data-resync]')) { resync(); return; }
  const cmd = t.closest('[data-cmd]');
  if (cmd) return command(cmd.dataset.cmd);
  const opt = t.closest('[data-opt]');
  if (opt && canAct()) {
    const i = +opt.dataset.opt; if (i < 0 || inputLocked()) return;
    const plan = t.closest('#pop')?._plan;
    closePop(); closeDrawer();
    return act(i, plan);
  }
  const zc = t.closest('[data-zcard]');
  if (zc) {
    const idxs = $('#overlay')._map?.get(zc.dataset.zcard) || [];
    if (idxs.length && canAct()) showMenu(zc, idxs, zc.dataset.zcard);
    return;
  }
  const zone = t.closest('.zone[data-zone]');
  if (zone) { openZone(+zone.dataset.p, zone.dataset.zone); return; }
  if (!t.closest('#pop')) closePop();
  if (!canAct()) return;
  const ui = S.ui;
  const hc = t.closest('#hand .card'), pm = t.closest('#board .perm'), plate = t.closest('.plate'), si = t.closest('.sitem');
  if (ui.kind === 'priority') {
    if (hc && handOpts(hc.dataset.name).length) return showMenu(hc, handOpts(hc.dataset.name), hc.dataset.name);
    if (pm && permOpts(pm).length) { S.srcOid = +pm.dataset.oid; return showMenu(pm, permOpts(pm), pm.dataset.name); }
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
  closePop();
  if (hc && handOpts(hc.dataset.name).length) return playGesture(hc, handOpts(hc.dataset.name));
  if (pm && permOpts(pm).length) {
    const d = myDecision(), idxs = permOpts(pm);
    const risky = idxs.some(i => /sacrific|discard|exile/i.test(d.refs[i].ability || d.options[i]));
    return risky || idxs.length > 1 ? showMenu(pm, idxs, pm.dataset.name) : chooseFromCard(pm, idxs);
  }
});
document.addEventListener('contextmenu', e => {
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
  for (const i of idxs || []) if (d.refs[i]?.taps) return {taps: d.refs[i].taps, sacs: d.refs[i].sacs || []};
  return null;
}
function showTaps(p) {
  $$('#board .will-tap').forEach(el => { el.classList.remove('will-tap', 'will-sac'); el.querySelectorAll('.b.tap').forEach(b => b.remove()); });
  if (!p) return;
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
  if (!drag) {
    const hcard = e.target.closest('#hand .card'), pcard = e.target.closest('#board .perm.activatable'), mitem = e.target.closest('#pop [data-opt]');
    showTaps(hcard ? tapsFor(handOpts(hcard.dataset.name)) : pcard ? tapsFor(permOpts(pcard)) : mitem ? tapsFor([+mitem.dataset.opt]) : null);
  }
  const c = e.target.closest('.card[data-name], .sitem[data-name]');
  const hc = e.target.closest('#hand .card');
  $$('#hand .card').forEach(x => { x.classList.toggle('hover', x === hc && !drag); });
  if (hc && !drag) {
    const cards = $$('#hand .card'), k = cards.indexOf(hc);
    cards.forEach((x, j) => x.style.setProperty('--sx', j === k ? '0px' : `${j < k ? -14 : 14}px`));
  } else if (!e.target.closest('#hand')) $$('#hand .card').forEach(x => x.style.setProperty('--sx', '0px'));
  clearTimeout(hoverTimer);
  if (c && !pinned) hoverTimer = setTimeout(() => showPreview(c.dataset.name), hc ? 120 : 250);
});
document.addEventListener('pointerleave', () => $$('#hand .card').forEach(x => x.classList.remove('hover')));
document.addEventListener('pointerout', e => {  // leaving a hand card for anything outside the hand drops its lift
  const hc = e.target.closest('#hand .card');
  if (hc && !(e.relatedTarget && e.relatedTarget.closest && e.relatedTarget.closest('#hand .card'))) { hc.classList.remove('hover'); showTaps(null); }
});
function showPreview(name, pin) {
  if (pin) pinned = !pinned;
  const i = S.cards[name]; if (!i) return;
  const tl = i.types.join(' ') + (i.subtypes.length ? ' — ' + i.subtypes.join(' ') : '');
  $('#preview').innerHTML = cardHtml(name) + `<div class="ptxt"><b>${esc(name)}</b> ${esc(i.cost)}<div>${esc((i.token ? 'Token ' : '') + tl)}${i.power != null ? ` · ${i.power}/${i.toughness}` : ''}</div><div class="ot">${esc(i.text)}</div>${pinned ? '<div class="st">Pinned (right-click again to unpin)</div>' : ''}</div>`;
}

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
  if (S.queue.length && S.pumping) { S.skip = true; return; }
  if (inputLocked()) return;
  const P = $('#primary');
  if (P.disabled) return;
  const a = P.dataset.act || '';
  if (a === 'new') return openNewGame();
  if (a === 'resync') return resync();
  if (a.startsWith('opt:')) { const i = +a.slice(4); if (i >= 0) return act(i); return; }
  if (a === 'autopay') { const d = myDecision(); return act(autoPay(d)); }
  if (!canAct()) return;
  const ui = S.ui, d = ui.d, s = stateAt(ui.fi);
  if (ui.kind === 'priority') {
    const floating = Object.values(s.players[S.seat].pool || {}).reduce((x, y) => x + y, 0);
    if (floating && ui.armed !== 'float') { ui.armed = 'float'; renderDock(); return; }
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
  if (e.target.closest('input, select, textarea') || $('#modal').classList.contains('on')) return;
  const k = e.key;
  if (e.repeat && (k === ' ' || k === 'Enter' || k === 'r' || k === 'R' || k === 'a' || k === 'n')) { e.preventDefault(); return; }  // a held key is not a decision
  if (inputLocked() && [' ', 'Enter', 'r', 'R', 'a', 'A', 'n', 'N'].includes(k)) { e.preventDefault(); return; }
  // A focused button, link or option keeps Space and Enter (activate it), never the global hotkeys.
  if ((k === ' ' || k === 'Enter') && e.target !== document.body && e.target.closest('button, a, [role=button], [tabindex]')) return;
  if (k === ' ') { e.preventDefault(); if ($('#overlay').classList.contains('on') && S.ui?.kind === 'assign_damage') { const b = $('#overlay .primary'); if (b && !b.disabled) b.click(); return; } primary(); return; }
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
document.addEventListener('mouseup', e => { const b = e.target.closest('button'); if (b && e.detail > 0) b.blur(); });  // mouse clicks don't park focus
$('#bAll').addEventListener('click', e => { e.stopPropagation(); openDrawer(); });
$('#bNew').addEventListener('click', () => openNewGame());
$('#bSettings').addEventListener('click', () => openSettings());
$('#bLog').addEventListener('click', () => { const l = $('#log'); l.classList.toggle('hide'); $('#bLog').textContent = l.classList.contains('hide') ? 'Show' : 'Hide'; });
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
function sound() { /* the sound set is added with the polish pass */ }
let audio = null;
function ping() {
  if (!PREF.sound) return;
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
    const o = audio.createOscillator(), g = audio.createGain();
    o.type = 'sine'; o.frequency.setValueAtTime(880, audio.currentTime); o.frequency.exponentialRampToValueAtTime(1320, audio.currentTime + 0.09);
    g.gain.setValueAtTime(0.0001, audio.currentTime); g.gain.exponentialRampToValueAtTime(0.05, audio.currentTime + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + 0.22);
    o.connect(g).connect(audio.destination); o.start(); o.stop(audio.currentTime + 0.25);
  } catch (e) { /* no audio */ }
}

// ---------------------------------------------------------------- game start / end
function finishGame() {
  S.ui = null;
  if (S.raw.length) render(last(), {noAnim: true});
  renderDock();
  if (!S.over) return;
  const w = S.meta?.winner, win = w === S.seat;
  const m = $('#modal');
  m.innerHTML = `<div class="mbox"><div class="note">Game over · turn ${S.meta?.turns}</div><div class="big ${w == null ? '' : win ? 'win' : 'loss'}">${w == null ? 'Draw' : win ? 'You win' : 'You lose'}</div>
    <div class="note">${esc(S.meta?.end_reason || '')}</div>
    <div style="display:flex;gap:8px;margin-top:16px;flex-wrap:wrap">${S.replay ? `<a class="btn" href="../#r=${encodeURIComponent(S.replay)}" target="_blank">Watch the full replay (both hands)</a>` : ''}
    <button class="btn" id="bLook">Look at the board</button><button class="primary" id="bAgain">New game</button></div></div>`;
  m.classList.add('on');
  $('#bAgain').onclick = () => openNewGame();
  $('#bLook').onclick = () => m.classList.remove('on');
}

async function openNewGame() {
  const m = $('#modal');
  m.innerHTML = `<div class="mbox"><h2>Play against a model</h2><div class="note">Loading…</div></div>`;
  m.classList.add('on');
  let opt;
  try { opt = await api('options'); } catch (e) {
    m.innerHTML = `<div class="mbox"><h2>Live play is off</h2><div class="err">${esc(e.message)}</div><p class="note">Start the server with <code>python -m mtg_ml.replay serve --models DIR</code> (or <code>--scripted-bot</code>).</p></div>`;
    return;
  }
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

function openSettings() {
  const m = $('#modal');
  m.innerHTML = `<div class="mbox"><h2>Settings</h2><form id="fSet">
    <label>Opponent replay speed</label><span class="seg">${Object.keys(SPEEDS).map(k => `<label><input type="radio" name="speed" value="${k}" ${PREF.speed === k ? 'checked' : ''}>${k === 'instant' ? 'instant' : k + '×'}</label>`).join('')}</span>
    <label>Mana</label><label class="note"><input type="checkbox" name="manualPay" ${PREF.manualPay ? 'checked' : ''}> pay mana by hand (default: auto-pay)</label>
    <label>Attacks</label><label class="note"><input type="checkbox" name="confirmEmptyAttack" ${PREF.confirmEmptyAttack ? 'checked' : ''}> confirm "No attacks" when creatures could attack</label>
    <label>Targets</label><label class="note"><input type="checkbox" name="autoTarget" ${PREF.autoTarget ? 'checked' : ''}> pick the only legal target automatically</label>
    <label>Sound</label><label class="note"><input type="checkbox" name="sound" ${PREF.sound ? 'checked' : ''}> ping when it is your decision</label>
    <label>Priority</label><label class="note"><input type="checkbox" name="full" ${S.fullControl ? 'checked' : ''}> full control (never auto-pass; F)</label>
    <div class="full note">Auto-pass stops: click the small bars under each step in the phase rail (gold: your turn, blue: the opponent's).</div>
    <div class="full" style="margin-top:8px"><button class="primary" type="submit">Done</button></div></form></div>`;
  m.classList.add('on');
  const f = $('#fSet');
  f.onsubmit = e => {
    e.preventDefault();
    PREF.speed = f.speed.value; PREF.manualPay = f.manualPay.checked; PREF.confirmEmptyAttack = f.confirmEmptyAttack.checked;
    PREF.autoTarget = f.autoTarget.checked; PREF.sound = f.sound.checked; S.fullControl = f.full.checked;
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
  } catch (e) { S.error = e.message; renderDock(); }
}

(async function init() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("g")) { S.gid = h.get("g"); try { await resync(); if (!S.error) return; } catch (e) { /* fall through */ } S.error = null; S.gid = null; toast("Your last game is no longer on the server (it was restarted or the game expired)."); }
  openNewGame();
})();

// For tests and debugging: the state and a way to read the current decision.
window.PLAY = {
  S, myDecision, act,
  info: () => ({ready: canAct() && !S.queue.length && !inputLocked(), over: S.over, busy: S.busy || S.pumping, error: S.error, frame: last(), kind: S.ui?.kind || null,
    options: myDecision()?.options || [], overlay: $('#overlay').classList.contains('on'), turn: S.shown >= 0 ? stateAt(S.shown).turn : 0}),
};
