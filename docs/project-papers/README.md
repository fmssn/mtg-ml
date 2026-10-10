# mtg-ml companion papers

Two standalone English papers, each five A4 pages including figures and references:

- [General edition](general.md): *Teaching a Computer to Play Magic: Inside the mtg-ml Project*.
- [Technical edition](technical.md): *mtg-ml: Engineering a Self-Play Learning System for Pauper Magic*.

The account is frozen to **10 October 2026**, repository revision
`18a3efa3e891b2b4ecddb51db22b615189a71ef5`. It describes recorded results, not
newly executed training or evaluation. [Source notes](source-notes.md) map claims
to evidence and explain checkpoint identities and statistical limits.

## Rebuild

From the repository root, using this workspace's existing environment:

```bash
uv pip install --python .venv/bin/python -r docs/project-papers/requirements.txt
.venv/bin/python docs/project-papers/build.py
```

The builder produces [the general PDF](../../output/pdf/mtg-ml-general.pdf) and
[the technical PDF](../../output/pdf/mtg-ml-technical.pdf). It also regenerates
the three figures in [figures/](figures/) as SVG and PDF, using ReportLab's vector
graphics. It needs no network, trained checkpoint, native extension, GPU, or
system fonts. Standard PDF fonts are used. The renderer uses invariant metadata
so rebuilding with the same dependencies produces identical PDF bytes.

Edit the Markdown to change prose; it is the only prose input. The supported
subset is headings, paragraphs, emphasis, links, inline code, the two-column
configuration table, and standalone figure images. Explicit `<!-- pagebreak -->`
markers preserve the five editorial sections. The builder fails if either PDF
has a different page count; revise the prose or layout rather than removing the
check. Figure captions and drawing code live in `build.py`; chart values live
in `matrix-data.json` and must retain their provenance when updated.

## Verification

```bash
make lint
git diff --check
mkdir -p .context/paper-qa
pdftoppm -scale-to 1400 -png output/pdf/mtg-ml-general.pdf .context/paper-qa/general
pdftoppm -scale-to 1400 -png output/pdf/mtg-ml-technical.pdf .context/paper-qa/technical
```

Poppler supplies `pdftoppm` and `pdfinfo`. Inspect **every page**, including
captions, diagrams, tables, references, and footers. The builder's page-count
check is not a substitute for visual review. PDF text extraction and annotation
inspection can additionally verify content coverage and clickable references.
The page PNGs are review intermediates and stay in `.context/`.

No engine behavior, feature contracts, or experiment data are changed by these
papers. Repository-wide engine tests are not required for this documentation
and PDF-rendering addition; lint, rebuild, evidence checks, link checks, and
visual inspection are the relevant acceptance checks.
