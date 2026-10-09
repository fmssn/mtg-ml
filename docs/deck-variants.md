# Deck variants

A **variant** is one registered 75 (60 main + 15 sideboard) of a supported archetype (`decks.DECKS`). The belief head samples the registered 75 once per match and uses its archetype and composition as supervised targets; sideboarding between games changes only the partition of those 75 cards.

| what | where |
|---|---|
| frozen manifest (generated, do not edit) | `mtg_ml/engine/variants.toml` |
| loader, validation, API | `mtg_ml/engine/variants.py` |
| builder | `tools/build_variants.py` (`--check` reports whether the manifest is current) |
| tests | `tests/test_variants.py` |

## API

```python
from mtg_ml.engine.variants import VARIANTS, Variant, variants, sample_variant, plan_for_variant, sampling_prior

v = sample_variant("jund_wildfire", rng, split="train")   # weighted by v.weight; one rng.random() call
plan = plan_for_variant(v, "mono_blue_terror")            # SideboardPlan subclass, relative to v.main
plan.apply()       # the sideboarded 60
plan.side_after()  # the 15 left over; together always v.composition
sampling_prior("test")  # {archetype: {variant_id: probability}}
```

`Variant` fields: `id` (`"<archetype>:<main build>+<side build>"`, the stock deck is `"<archetype>:stock"`), `archetype`, `main`, `sideboard`, `source`, `weight`, `provenance`, `constructed`, `split`, `stock`, `plans` (stored plan adjustments). `registered_75` is the identity key (sorted main, sorted sideboard); `split_key` hashes the 75's composition.

## Source

The local Pauper-Research snapshot (`/Users/fabsi/repos/mtg-ml-orchestration/pauper-research`, commit `592796a`, 2026-10-05), dashboard `reports/dashboard/index.html` (generated 2026-10-04), period 90 days (2026-07-05 to 2026-10-02), all events ("combined"), section **Builds and decisions**. These are the numbers `decks.py` already cites (Jund sideboard "Stock list" 124 of 428 lists, and so on); the snapshot calls Jund Wildfire "Jund Midrange". `prototypes/deck_decisions/data.js` is the older prototype of that section and has no named builds, so it is not used. The exact snapshot commit, file hash and window are in the manifest's `[snapshot]` table.

The dashboard reports main-deck builds and sideboard builds **independently** (each a typical list with its list count and share). No joint main+side list is available, so every variant combines one main build with one sideboard build: `constructed = true`, `weight = main share x side share` (independence assumption). Each variant's `provenance` records both builds, their list counts and shares, and the archetype's list count.

## Rules

- **Names:** only the aliases in `ALIASES` (`Lórien Revealed` -> `Lorien Revealed`); the snapshot already folds Snow-Covered basics into basics.
- **Unsupported cards:** a build with any card missing from `cards.CARDS` is rejected, with its missing cards, and so is every combination with it. Nothing is substituted or dropped.
- **Legality:** 60 main, 15 side, at most 4 copies of a non-basic across the 75 (`sideboard.check_deck`). Combinations that break it are rejected with the reason.
- **Duplicates:** identical registered 75s merge (weights add).
- **Stock decks** are variants (`<archetype>:stock`, `stock = true`). When the stock 75 equals a snapshot combination, it takes that weight (Red Madness, Grixis Affinity, Elves). Otherwise (Jund and Mono Blue Terror use MTGGoldfish maindecks, Tron uses the archetype's aggregate list) it gets the largest accepted weight of its archetype (`weight_basis = "policy:max"`).
- **Splits** (`assign_splits`): per archetype, the distinct 75 compositions other than the stock one are ordered by their SHA-256 key. With n of them, the first `max(1, round(0.15 n))` go to **test** (n >= 2), the next as many to **dev** (n >= 3), the rest to **train**. Stock is always train, and so is any variant with the stock 75. Duplicates share a key and therefore a split. The order depends only on card identity, not on names or weights; adding variants can move existing ones, so experiments should cite the manifest's commit.
- **Sampling prior:** `sampling_prior(split)` normalises the weights within each archetype and split. These are the per-archetype proportions only; how often each archetype is drawn is up to the caller.

## Sideboard plans

`plan_for_variant(variant, opponent)` reuses the stock row of `sideboard_plans.toml` when `validate_plan` accepts it against the variant's lists. Otherwise the manifest stores an adjusted plan for that matchup (`[variant.plans.<opponent>]`, with a `why` that lists every adjustment). The builder adjusts the stock row deterministically:

1. Each `in` card is taken up to what the variant's sideboard has. Any shortfall comes from `PLAN_EQUIVALENTS` (Pyroblast and Red Elemental Blast, Hydroblast and Blue Elemental Blast, Steel Sabotage for Annul).
2. Each `out` card is taken up to what the variant's maindeck has. Any shortfall comes from the variant's slot replacements: nonland cards it plays more copies of than the stock maindeck, most extra copies first.
3. Both sides are trimmed from the end to the same total.

Every plan for every variant and opponent is validated at import: 60 / 15, at most 4 copies, and the 75 unchanged. Of the 71 x 6 variant/opponent pairs, 282 use an adjusted plan. Of those, 113 keep as many swaps as the stock row, 133 lose one and 36 lose two to four. Sideboard cards that only a variant has (Ancient Grudge, Go for the Throat, Steel Sabotage outside the Annul slot, ...) never come in under these plans. A learned or expert plan should replace them.

## Current manifest (rebuilt 2026-10-09 with the cards of #75)

71 variants. Spell Pierce, Prophetic Prism, Malevolent Rumble and Fiery
Cannonade (#75) brought in the most-played Mono Blue Terror mains and most Tron
builds (54 -> 71).

| archetype | train | dev | test |
|---|---|---|---|
| jund_wildfire | 21 | 4 | 4 |
| mono_blue_terror | 9 | 2 | 2 |
| red_madness | 2 | 0 | 1 |
| grixis_affinity | 8 | 1 | 1 |
| elves | 2 | 1 | 1 |
| tron | 8 | 2 | 2 |

Sideboard plans: 282 of the 426 variant/opponent pairs use an adjusted plan
(113 keep every stock swap, 133 lose one, 36 lose two to four).

Still rejected (build share; full list in `[[rejected]]` of the manifest):

- **Jund:** main "Nyxborn Hydra build, 3 Fanatical Offering" (17.5%): Snuff Out. Two pairings with 5 Nihil Spellbomb across the 75.
- **Mono Blue Terror:** "14 Island build" (14%): Deprive, Preordain. "Boomerang build" (11%): Boomerang. Side "No Blue Elemental Blast build" (16.5%): Spreading Seas.
- **Red Madness:** main "2 Faithless Looting build" (19%): Melded Moxite.
- **Grixis Affinity:** "Cryogen Relic build" (13%): Cryogen Relic. "2 Drossforge Bridge build" (7%): Chromatic Star.
- **Elves:** side "No Spinewoods Paladin build" (30%): Lignify, Primordial Pachyderm, Rooftop Percher, Scattershot Archer. "Negate build" (13%): Hallow, Negate, Prohibit. Both Birchlore Rangers mains (7.5%, 7.3%) and "Nylea's Disciple build" (7%): Birchlore Rangers, Elvish Vanguard, Jaspera Sentinel, Vines of Vastwood, Wellwisher, Distant Melody, Salt Road Packbeast, Nylea's Disciple, Mwonvuli Acid-Moss.
- **Tron:** side "Breath Weapon build" (23%): Earth Rift. "No Candy Trail build" (19%): Rooftop Percher. Nyxborn Hydra builds (9%, 8%): Firebolt, Wooded Ridgeline. Coalition Honor Guard builds (8% each). One pairing with 5 Breath Weapon.

Next cards by unlocked share: the Elves sideboard set (30%), Earth Rift, Melded Moxite, Rooftop Percher, Snuff Out, Spreading Seas. After adding cards, rerun `tools/build_variants.py`.

## Decisions (2026-10-09, project owner)

- The per-archetype split, the stock-deck weight rule (`policy:max`) and the
  rule-adjusted sideboard plans are accepted as they are. The plans that lose
  two to four swaps (22 at first, 36 after the rebuild) are a known weakness, to be replaced by researched or learned
  plans later.
- Cards missing from the most-played builds are implemented rather than the
  builds dropped: Spell Pierce, Prophetic Prism, Malevolent Rumble and Fiery
  Cannonade first (#75); the manifest was rebuilt with them.
