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

Every plan for every variant and opponent is validated at import: 60 / 15, at most 4 copies, and the 75 unchanged. Of the 54 x 6 variant/opponent pairs, 198 use an adjusted plan. Of those, 84 keep as many swaps as the stock row, 92 lose one and 22 lose two to four. Sideboard cards that only a variant has (Ancient Grudge, Go for the Throat, Steel Sabotage outside the Annul slot, ...) never come in under these plans. A learned or expert plan should replace them.

## Current manifest (built 2026-10-09)

| archetype | train | dev | test | accepted main x side builds |
|---|---|---|---|---|
| jund_wildfire | 21 | 4 | 4 | 5 x 6 (2 combinations illegal), + stock |
| mono_blue_terror | 3 | 1 | 1 | 1 x 4, + stock |
| red_madness | 2 | 0 | 1 | 3 x 1 (stock is one of them) |
| grixis_affinity | 8 | 1 | 1 | 2 x 5 (stock is one of them) |
| elves | 2 | 1 | 1 | 4 x 1 (stock is one of them) |
| tron | 2 | 0 | 1 | 1 x 2, + stock |

Rejected (play share in brackets; full list with counts in `[[rejected]]` of the manifest):

- **Jund:** main "Nyxborn Hydra build, 3 Fanatical Offering" (17.5%), Snuff Out. "No Krark-Clan Shaman build" with the "Stock list" or "Pyroblast build 2" sideboard: 5 Nihil Spellbomb across the 75.
- **Mono Blue Terror:** main "Stock list" (47%) and "Murmuring Mystic build" (18%), Spell Pierce. "14 Island build", Preordain and Deprive. "Boomerang build", Boomerang and Spell Pierce. Side "No Blue Elemental Blast build", Spreading Seas.
- **Red Madness:** main "2 Faithless Looting build" (19%), Melded Moxite.
- **Grixis Affinity:** main "Cryogen Relic build", Cryogen Relic. "2 Drossforge Bridge build", Chromatic Star.
- **Elves:** both Birchlore Rangers builds (Birchlore Rangers, Elvish Vanguard, Jaspera Sentinel, Vines of Vastwood, Wellwisher, Distant Melody, Salt Road Packbeast). Side "No Spinewoods Paladin build" (30%), Primordial Pachyderm, Lignify, Rooftop Percher, Scattershot Archer. "Negate build", Negate, Prohibit, Hallow. "Nylea's Disciple build", Nylea's Disciple, Mwonvuli Acid-Moss, Scattershot Archer, Rooftop Percher.
- **Tron:** main "Stock list" (40%) and "No Generous Ent build", Prophetic Prism. "No Candy Trail build", Rooftop Percher. Three Nyxborn Hydra builds, Malevolent Rumble (also Prophetic Prism, Firebolt, Wooded Ridgeline). Side "Stock list" (31%), Fiery Cannonade. "Breath Weapon build", Earth Rift. Both Coalition Honor Guard builds, Coalition Honor Guard.

Implementing Spell Pierce, Prophetic Prism, Malevolent Rumble, Fiery Cannonade and Melded Moxite (docs/adding-cards.md) would unlock most of the missing play share. After adding cards, rerun `tools/build_variants.py`.

## Decisions (2026-10-09, project owner)

- The per-archetype split, the stock-deck weight rule (`policy:max`) and the
  rule-adjusted sideboard plans are accepted as they are. The 22 plans that lose
  two to four swaps are a known weakness, to be replaced by researched or learned
  plans later.
- Cards missing from the most-played builds are implemented rather than the
  builds dropped: Spell Pierce, Prophetic Prism, Malevolent Rumble and Fiery
  Cannonade first (separate cards PR); the manifest is rebuilt when they land.
