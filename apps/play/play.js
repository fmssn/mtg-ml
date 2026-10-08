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
  stops: {me: {main1: true, main2: true}, opp: {end: true}},
}, loadJSON('mtgml-play-pref', {}));
function loadJSON(k, d) { try { return JSON.parse(localStorage.getItem(k) || 'null') ?? d; } catch (e) { return d; } }
function saveJSON(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private window */ } }
const savePref = () => saveJSON('mtgml-play-pref', PREF);

// ---------------------------------------------------------------- game state
const S = {
  gid: null, seat: 0, raw: [], cards: {}, meta: null, over: false, replay: null,
  shown: -1, prev: null, queue: [], pumping: false, busy: false, error: null, skip: false,
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
let imgBusy = false;
async function resolveImages() {
  if (imgBusy) return;
  const want = Object.keys(S.cards).filter(n => !(n in IMG));
  if (!want.length) return;
  imgBusy = true;
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
  } catch (e) { console.warn('scryfall', e); } finally { imgBusy = false; }
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
  Object.assign(S, {gid: id, raw: [], shown: -1, prev: null, queue: [], plan: null, ui: null, lastMine: -1, passMode: null,
    fullControl: false, feed: [], logDone: 0, error: null, skip: false});
  $('#log').innerHTML = '';
  $('#feed').innerHTML = '';
}

async function send(index) {
  const frame = last();
  S.busy = true; S.error = null; S.ui = null; closePop(); closeOverlay(); renderDock();
  try {
    const v = await api(`${encodeURIComponent(S.gid)}/choose`, {frame, index, since: frame});
    S.lastMine = frame;
    applyView(v);
  } catch (e) {
    S.error = e.message;
  } finally { S.busy = false; }
}

// Pick an option (from a gesture) and run the game on to the next prompt.
async function act(index, plan) {
  if (S.busy || S.pumping || !myDecision()) return;
  if (plan !== undefined) S.plan = plan;
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
      if (a == null || autos > 400) { S.plan = null; S.skip = false; enterDecision(true); break; }
      autos++;
      await send(a);
    }
  } finally { S.pumping = false; renderDock(); }
}

// ---------------------------------------------------------------- automation (P2, P6, P7 of the brief)
const isReal = r => r.type !== 'pass' && r.type !== 'mana';
const ACTS = new Set(['cast', 'activate', 'plot', 'attack', 'block']);

function oppActedSince(fi) {
  for (let i = fi + 1; i <= last(); i++) {
    for (const a of S.raw[i].actions || []) if (a.p === opp() && ACTS.has(a.t)) return true;
    const d = S.raw[i].decision;
    if (d && d.player === opp() && d.chosen != null && ['cast', 'activate', 'plot'].includes(d.refs?.[0]?.type)) return true;
  }
  return false;
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
  appendLog(i);
  const acts = f.actions || [];
  const theirs = acts.filter(a => a.p === opp());
  const prevD = i > 0 ? S.raw[i - 1].decision : null;
  if (prevD && prevD.player === opp()) feedFromDecision(prevD, i - 1);
  for (const a of acts) if (a.p === opp() && ['discard', 'mulligan', 'sacrifice'].includes(a.t)) pushFeed(quietText(a), i, true);
  const notable = acts.some(a => ['cast', 'play', 'activate', 'attack', 'block', 'resolve', 'trigger', 'enter', 'leave', 'dies', 'turn', 'discard'].includes(a.t));
  const isLast = i === last();
  if (!notable && !isLast && i !== 0) return;
  const cast = theirs.find(a => a.t === 'cast' || a.t === 'activate' || a.t === 'plot');
  if (cast && !S.skip && beatScale() > 0) {
    const name = cast.t === 'activate' ? cast.name.split(':')[0].trim() : cast.name;
    showSpot(name, cast.t === 'cast' ? `${oppLabel()} casts ${cast.name}` : cast.t === 'plot' ? `${oppLabel()} plots ${cast.name}` : `${oppLabel()} activates ${cast.name}`);
    await wait(950);
    hideSpot();
  }
  render(i);
  if (isLast) return;
  let beat = 0;
  if (theirs.some(a => a.t === 'play')) beat = Math.max(beat, 420);
  if (theirs.some(a => a.t === 'attack' || a.t === 'block')) beat = Math.max(beat, 700);
  if (acts.some(a => a.t === 'resolve' || a.t === 'trigger')) beat = Math.max(beat, 380);
  if (acts.some(a => a.t === 'dies' || a.t === 'leave')) beat = Math.max(beat, 380);
  if (acts.some(a => a.t === 'turn')) beat = Math.max(beat, 300);
  if (beat) await wait(beat);
}

function showSpot(name, cap) {
  const el = $('#spot');
  el.innerHTML = cardHtml(name) + `<div class="cap">${esc(cap)}</div>`;
  el.classList.add('on');
}
function hideSpot() { $('#spot').classList.remove('on'); }
const oppLabel = () => S.meta ? 'Opponent' : 'Opponent';

// The opponent's actions as chips (each can be flagged), held until you act.
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
  if (S.feed.length > 6) S.feed.shift();
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
  $('#feed').innerHTML = S.feed.slice(-4).map(c => `<span class="fchip ${c.quiet ? 'quiet' : ''}" title="${esc(c.text)}"><span class="txt">Opp: ${esc(c.text)}</span><button class="flag ${isFlagged(c.fi) ? 'on' : ''}" data-flag="${c.fi}" title="Flag this play as odd">⚑</button></span>`).join('');
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
    const pm = /^\s*p(\d) /.exec(line);
    const who = pm ? (+pm[1] === S.seat ? 'me' : 'opp') : '';
    const text = clean(line.trim()).replace(/^p(\d) /, (_, p) => +p === S.seat ? 'You: ' : 'Opp: ')
      .replace(/^(\w+): (Pass priority|Done declaring attackers)$/, '$1: $2');
    out.push(`<div class="${who} ${/^\s/.test(line) ? 'sub' : ''}">${esc(text)}</div>`);
  }
  const d = f.decision;
  if (d && d.player === opp() && d.chosen != null && !['pass', 'pay', 'hidden'].includes(d.refs?.[0]?.type)) {
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
function render(i, o = {}) {
  const f = S.raw[i], s = f.state;
  const before = o.noAnim ? null : rects();
  const prev = S.prev;
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
  S.prev = s;
  requestAnimationFrame(drawArrows);
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
  const delta = pv ? P.life - pv.life : 0;
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
    <div class="life ${P.life <= 5 ? 'low' : ''} ${delta < 0 ? 'hit' : delta > 0 ? 'heal' : ''}">${P.life}${delta ? `<span class="delta ${delta < 0 ? 'neg' : 'pos'}">${delta > 0 ? '+' : ''}${delta}</span>` : ''}</div>
    ${preview != null ? `<div class="preview-life">→ ${preview} after combat</div>` : ''}
    <div class="zones">
      <div class="zone" title="Cards in hand">Hand <b>${P.hand.length}</b></div>
      <div class="zone" title="Cards in library${P.library_top_known.length ? '; known on top: ' + esc(P.library_top_known.join(', ')) : ''}">Library <b>${P.library}</b></div>
      <div class="zone clickable ${gyCast}" data-zone="graveyard" data-p="${p}" title="Click to see the graveyard">Grave <b>${P.graveyard.length}</b></div>
      <div class="zone clickable ${exCast}" data-zone="exile" data-p="${p}" title="Click to see exiled cards">Exile <b>${P.exile.length}</b></div>
    </div>
    ${pool ? `<div class="mana" title="Floating mana">${pool}</div>` : ''}`;
  void d;
}

function groupPerms(perms, s) {
  const hosts = new Set(s.battlefield.filter(c => c.attached_to != null).map(c => c.attached_to));
  const out = [], by = new Map();
  for (const c of perms) {
    const creature = c.power != null;
    if (creature || c.attacking || c.blocking != null || c.damage || c.counters || hosts.has(c.oid) || !(c.types.includes('Land') || c.token)) { out.push([c]); continue; }
    const k = [c.name, !!c.tapped].join('|');
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
  if (!pc && prevBy.size) cls.push('entered');
  if (pc && (c.damage || 0) > (pc.damage || 0)) cls.push('flash');
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
  if (pc && (c.damage || 0) > (pc.damage || 0)) b += `<span class="b dmg">-${c.damage - (pc.damage || 0)}</span>`;
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

function fitRows() {
  for (const row of $$('.row')) {
    row.style.setProperty('--ov', '0px');
    const kids = [...row.children];
    if (kids.length < 2) continue;
    const avail = row.clientWidth, need = row.scrollWidth;
    if (need > avail) {
      const w = kids.reduce((a, k) => a + k.offsetWidth, 0) / kids.length;
      const ov = Math.min(w * 0.7, (need - avail) / (kids.length - 1) + 2);
      row.style.setProperty('--ov', `${ov}px`);
    }
  }
}

function renderMid(s) {
  const mine = s.active === S.seat;
  $('#turn').innerHTML = `<b class="${mine ? 'me' : 'opp'}">${mine ? 'Your turn' : "Opponent's turn"}</b><span style="color:var(--muted)">Turn ${s.turn} · ${esc(STEP_LABEL[s.step] || s.step)}</span>`;
  const who = mine ? 'me' : 'opp';
  $('#rail').innerHTML = STEPS.map(st => `<div class="step ${st === s.step ? 'cur ' + who : ''}" title="${esc(STEP_LABEL[st])}: toggle where auto-pass stops (gold: your turn, blue: opponent's turn)">
    <span>${esc(STEP_LABEL[st])}</span><span class="stops"><button class="stop me ${PREF.stops.me[st] ? 'on' : ''}" data-stop="me:${st}" title="Stop here on your turn"></button><button class="stop opp ${PREF.stops.opp[st] ? 'on' : ''}" data-stop="opp:${st}" title="Stop here on the opponent's turn"></button></span></div>`).join('');
}

function renderStack(s) {
  const el = $('#stack');
  if (!s.stack.length) { el.innerHTML = ''; return; }
  const items = [...s.stack].reverse();
  el.innerHTML = `<div class="sh">Stack · top first</div>` + items.map((it, j) => {
    const tgt = H.target.has('s' + it.sid);
    const name = it.card || it.name;
    return `<div class="sitem ${it.controller === S.seat ? 'c-me' : 'c-opp'} ${j === 0 ? 'top' : ''} ${tgt ? 'targetable' : ''}" data-sid="${it.sid}" data-name="${esc(name)}">
      ${cardHtml(name)}<div class="st"><b>${esc(it.name)}</b>${it.controller === S.seat ? 'yours' : 'opponent'}${it.x ? ` · X=${it.x}` : ''}${it.targets.length ? `<div class="tg">→ ${esc(it.targets.map(t => clean(t.replace(/^player (\d)/, (_, p) => +p === S.seat ? 'you' : 'opponent'))).join(', '))}</div>` : ''}</div></div>`;
  }).join('');
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

// ---------------------------------------------------------------- the decision in front of the player
function enterDecision(fresh) {
  const d = myDecision();
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
  if (top) return `Resolve ${top.name}`;
  const mine = s.active === S.seat;
  if (mine && s.step === 'main2') return 'End turn';
  if (s.step === 'end' || s.step === 'cleanup') return mine ? "Pass → Opponent's turn" : 'Pass → Your turn';
  return `Pass → ${NEXT[s.step] || 'next step'}`;
}

function renderDock() {
  const d = myDecision(), ui = S.ui, P = $('#primary'), pr = $('#prompt'), ch = $('#choices');
  const pill = $('#pill');
  pill.className = S.passMode || S.fullControl ? 'on' : '';
  pill.textContent = S.fullControl ? 'Full control: no auto-pass · F' : S.passMode === 'opp' ? 'Passing until the opponent acts · Esc' : '';
  P.className = 'primary'; P.disabled = true; ch.innerHTML = ''; pr.className = '';
  $('#bAll').style.visibility = d && ui ? 'visible' : 'hidden';
  if (S.error) {
    pr.innerHTML = `<span class="k">Problem</span><span class="err">${esc(S.error)}</span>`;
    ch.innerHTML = `<button class="choice" data-resync="1">Reload the game from the server</button>`;
    P.textContent = 'Retry'; P.disabled = false; P.dataset.act = 'resync';
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
    el.innerHTML = `<div class="obox"><h2>${esc(clean(d.prompt))}</h2><div class="sub">Starts from a legal split; adjust with − and +. Space confirms.</div>${rows}
      <div class="orow"><button class="primary" ${ok ? '' : 'disabled'} data-opt="${splitIndex(d, ui.dmg)}">${ok ? 'Assign damage' : 'Not a legal split'}</button></div></div>`;
  } else {
    const tiles = d.refs.map((r, i) => r.name && S.cards[r.name]
      ? `<button class="tile" data-opt="${i}">${cardHtml(r.name)}<span>${esc(clean(d.options[i]))}</span></button>`
      : `<button class="choice" data-opt="${i}">${esc(clean(d.options[i]))}</button>`).join('');
    el.innerHTML = `<div class="obox"><h2>${esc(clean(d.prompt))}</h2><div class="sub">${esc(d.kind.replace(/_/g, ' '))} · click one</div><div class="grid">${tiles}</div></div>
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
    const key = [kills, r.player ? r.split[r.split.length - 1] : 0];
    if (!bestKey || key[0] > bestKey[0] || (key[0] === bestKey[0] && key[1] > bestKey[1])) { best = r.split; bestKey = key; }
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
}
function closePop() { $('#pop').classList.remove('on'); }

// ---------------------------------------------------------------- gestures
const canAct = () => !!(myDecision() && S.ui && !S.busy && !S.pumping);
function handOpts(name) { return H.hand.get(name) || []; }
function permOpts(el) { const oids = el.dataset.oids.split(' ').map(Number); for (const o of oids) if (H.perm.has(o)) return H.perm.get(o); return []; }

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
  const g = $('#ghost');
  g.innerHTML = drag.el.outerHTML.replace(/class="card [^"]*"/, 'class="card"');
  g.style.display = 'block';
  g.classList.remove('back');
  drag.el.classList.add('dragging');
  drag.w = drag.el.offsetWidth; drag.h = drag.el.offsetHeight;
  drag.vx = 0; drag.lx = e.clientX;
  $('#field0').classList.add('drop');
  drag.targets = mentionsTarget(drag.el.dataset.name) && handOpts(drag.el.dataset.name).length === 1;
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
  const idxs = handOpts(dg.el.dataset.name);
  if (inPlay && idxs.length) {
    g.style.display = 'none';
    dg.el.classList.remove('dragging');
    let plan;
    if (dg.targets) { const t = targetAt(e.clientX, e.clientY); if (t && (t.player != null || t.el.closest('#side1, #side0'))) plan = {kind: 'target', t: t.player != null ? {player: t.player} : {oid: t.oid}}; }
    chooseFromCard(dg.el, idxs, plan);
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
    const i = +opt.dataset.opt; if (i < 0) return;
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
  if (hc && handOpts(hc.dataset.name).length) return chooseFromCard(hc, handOpts(hc.dataset.name));
  if (pm && permOpts(pm).length) return chooseFromCard(pm, permOpts(pm));
});
document.addEventListener('contextmenu', e => { const c = e.target.closest('.card[data-name]'); if (c) { e.preventDefault(); showPreview(c.dataset.name, true); } });

// Hover: hand lift with neighbours spreading, and the preview panel.
let hoverTimer = null, pinned = false;
document.addEventListener('pointerover', e => {
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
  if (!ui) return;
  if (c === 'all' && ui.kind === 'declare_attacker') { ui.eligible.forEach(o => ui.sel.add(o)); ui.armed = null; render(S.shown, {noAnim: true}); renderDock(); }
  if (c === 'clear') { ui.sel.clear(); ui.blocks.clear(); ui.blockSel = null; render(S.shown, {noAnim: true}); renderDock(); }
}

function primary() {
  if (S.queue.length && S.pumping) { S.skip = true; return; }
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
    S.plan = {kind: 'attack', want};
    const i = planAnswer(d);
    return act(i);
  }
  if (ui.kind === 'declare_blocker') {
    S.plan = {kind: 'block', map: new Map(ui.blocks)};
    return act(planAnswer(d));
  }
}

document.addEventListener('keydown', e => {
  if (e.target.closest('input, select, textarea') || $('#modal').classList.contains('on')) return;
  const k = e.key;
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
  if ((k === 'Enter' || k === 'r' || k === 'R') && canAct() && S.ui.kind === 'priority') {
    S.passMode = 'opp'; S.passTurn = stateAt(last()).active === S.seat ? stateAt(last()).turn : -1;
    return act(S.ui.d.refs.findIndex(r => r.type === 'pass'));
  }
  if (k === 'f' || k === 'F') { S.fullControl = !S.fullControl; toast(S.fullControl ? 'Full control: every priority stop is yours.' : 'Auto-pass back on.'); renderDock(); return; }
  if ((k === 'a' || k === 'A') && S.ui?.kind === 'declare_attacker') return command('all');
  if ((k === 'n' || k === 'N') && canAct() && ['declare_attacker', 'declare_blocker'].includes(S.ui.kind)) { const d = S.ui.d; return act(d.refs.findIndex(r => r.done || (r.none && S.ui.kind === 'declare_blocker'))); }
  if (k === 'o' || k === 'O') { $('#drawer').classList.contains('on') ? closeDrawer() : openDrawer(); }
});
$('#primary').addEventListener('click', e => { e.stopPropagation(); primary(); });
$('#bAll').addEventListener('click', e => { e.stopPropagation(); openDrawer(); });
$('#bNew').addEventListener('click', () => openNewGame());
$('#bSettings').addEventListener('click', () => openSettings());
$('#bLog').addEventListener('click', () => { const l = $('#log'); l.classList.toggle('hide'); $('#bLog').textContent = l.classList.contains('hide') ? 'Show' : 'Hide'; });
$('#overlay').addEventListener('click', e => { if (e.target.dataset.dmg) { const [j, dv] = e.target.dataset.dmg.split(':').map(Number); const ui = S.ui; ui.dmg[j] = Math.max(0, ui.dmg[j] + dv); renderOverlay(); renderDock(); } });

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
  if (h.get('g')) { S.gid = h.get('g'); try { await resync(); if (!S.error) return; } catch (e) { /* fall through */ } S.error = null; S.gid = null; }
  openNewGame();
})();

// For tests and debugging: the state and a way to read the current decision.
window.PLAY = {
  S, myDecision, act,
  info: () => ({ready: canAct() && !S.queue.length, over: S.over, busy: S.busy || S.pumping, error: S.error, frame: last(), kind: S.ui?.kind || null,
    options: myDecision()?.options || [], overlay: $('#overlay').classList.contains('on'), turn: S.shown >= 0 ? stateAt(S.shown).turn : 0}),
};
