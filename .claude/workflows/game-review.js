export const meta = {
  name: 'game-review',
  description: 'Review recorded games with one Opus agent per game, cluster setup flaws across games, verify each against the code',
  whenToUse: 'After `python -m mtg_ml.review review ... --backend prompt` wrote <dir>/<stem>.prompt.md files (see the game-review skill)',
  phases: [
    { title: 'Review', detail: 'one reviewer per game' },
    { title: 'Cluster', detail: 'group findings across games into candidate flaws' },
    { title: 'Verify', detail: 'try to refute each candidate against the code and the game files' },
  ],
}

// args: {dir, code, games: [stem, ...], previous?: path to an earlier flaws.json}
const DIR = args.dir
const CODE = args.code
const games = args.games

phase('Review')
const reviewed = await pipeline(games, (g) => agent(
  `You are acting as an LLM game reviewer. Read the file ${DIR}/${g}.prompt.md in full (read it in chunks if needed; read ALL of it). The part in <system>...</system> is your instructions; the rest is the game to review. Do not read any other files, do not run code, do not look at the .json game file or other reviews. Follow the instructions exactly and write ONLY the JSON object they ask for to ${DIR}/${g}.reply.txt using the Write tool. Then return the number of findings and the path you wrote.`,
  { label: `review ${g.slice(-6)}`, phase: 'Review', schema: { type: 'object', properties: { findings: { type: 'integer' }, wrote: { type: 'string' } }, required: ['findings', 'wrote'] } }))
const ok = reviewed.filter(Boolean).length
log(`${ok}/${games.length} games reviewed`)
if (ok < games.length) log(`missing reviews: ${games.filter((g, i) => !reviewed[i]).join(', ')}`)

phase('Cluster')
const prev = args.previous
  ? `\n\nAn earlier review of an older checkpoint produced ${args.previous} (JSON: flaws with ids and verdicts). For each flaw you report, set "previous" to the matching earlier id, or "new". Also list the earlier confirmed or partly-confirmed ids that no longer appear in "gone".`
  : ''
const CLUSTER_SCHEMA = { type: 'object', properties: {
  flaws: { type: 'array', items: { type: 'object', properties: {
    id: { type: 'string' }, cause: { type: 'string' }, title: { type: 'string' }, description: { type: 'string' },
    games: { type: 'integer' }, previous: { type: 'string' },
    examples: { type: 'array', items: { type: 'object', properties: { game: { type: 'string' }, decision: { type: 'string' }, note: { type: 'string' } }, required: ['game', 'decision', 'note'] } },
  }, required: ['id', 'cause', 'title', 'description', 'games', 'examples'] } },
  gone: { type: 'array', items: { type: 'string' } },
}, required: ['flaws'] }
const clusters = await agent(
  `${games.length} game reviews are in ${DIR}/*.reply.txt (one JSON object per game: summary + findings; the seed is in the file name). Read ALL of them. Group every finding whose cause is engine_bug, masking_gap, observation_gap or architecture into distinct candidate setup flaws: one flaw = one underlying problem, however many games show it. Also add a flaw for any undertraining / reward_shaping / sampling_noise pattern that recurs in 4 or more games and plausibly has a setup cause (say which). Merge duplicates aggressively, keep distinct problems apart. For each flaw give a short kebab-case id, the cause, a title, a precise description of the claimed problem, the number of games it appears in, and up to 4 examples (seed, decision id like d123, one-line note). Order by number of games.${prev}`,
  { phase: 'Cluster', schema: CLUSTER_SCHEMA })
const flaws = (clusters && clusters.flaws) || []
log(`${flaws.length} candidate flaws`)

phase('Verify')
const VERDICT = { type: 'object', properties: {
  verdict: { type: 'string', enum: ['confirmed', 'refuted', 'partly'] },
  evidence: { type: 'string' }, root_cause: { type: 'string' }, fix: { type: 'string' }, files: { type: 'array', items: { type: 'string' } },
}, required: ['verdict', 'evidence', 'root_cause', 'fix', 'files'] }
const verified = await pipeline(flaws, (f) => agent(
  `Verify or refute one claimed flaw in an RL setup for Magic: The Gathering (Pauper). Try hard to REFUTE it; only confirm what the code and the game files prove.

Claimed flaw (${f.cause}): ${f.title}
${f.description}
Examples: ${JSON.stringify(f.examples)}

Material (read-only; do not edit anything):
- The code the games were played with: ${CODE} (Python reference engine mtg_ml/engine/game.py and cards.py, card definitions mtg_ml/engine/cards.toml, the policy's features mtg_ml/encode.py and mtg_ml/rl/features.py, the network mtg_ml/rl/model.py, docs/features.md; the Rust port in native/src mirrors them).
- Game files: ${DIR}/<stem>.json for the seeds above (replay format: frames[i] = {state, events, decision{player, kind, prompt, options, chosen, policy, value}}; i is the decision id dN). The rendered transcript is the matching .prompt.md.
- You may run Python from ${CODE} (cd there; python -m ... uses that code) to replay or inspect: mtg_ml.review.verify.prefixes(rep, [d]) yields the rebuilt game at decision d; mtg_ml.encode.state_features / entity_features and mtg_ml.rl.features.featurize show what the policy sees; mtg_ml.review.verify.compare runs paired rollouts for "option B was better" claims.

Decide: confirmed (the flaw is real as described), partly (real but different from the description: say how), or refuted (the engine/feature behaviour is correct or the claim misreads the game). For engine claims, cite the Magic comprehensive rules and the engine code path. For feature/architecture claims, show the features at an example decision. Give the root cause, a concrete fix (which file/function, and whether it needs both engines and a feature-set bump), and the files involved.`,
  { label: `verify ${f.id}`, phase: 'Verify', schema: VERDICT }).then(v => ({ ...f, check: v })))

return { reviewed: ok, flaws: verified.filter(Boolean), gone: (clusters && clusters.gone) || [] }
