# CHARMM developer feature-usage survey

Tooling to find out which CHARMM features/modules developers actually use,
so the team can decide what to keep, modernize, document, or retire.

The survey unit is a **high-level feature** (a human-facing grouping of the
low-level compile-time `KEY_*` keywords), not an individual keyword.

## Files

| File | What it is |
|------|------------|
| `create_charmm_poll.gs` | Google Apps Script that builds the Google Form automatically. **Start here.** |
| `poll_questions.md` | The human-readable question list the script is built from. Edit this and the `.gs` together if you want to change wording or coverage. |
| `feature_map.md` | The full mapping: 38 high-level features, each with its one-line description, the `KEY_*` keywords it covers, and the `doc/*.info` topic(s) it comes from. |
| `keywords_canonical.txt` | The 147 canonical keywords (from `tool/cmake/prefx_keywords.cmake`). |
| `keywords_used.txt` | The 262 `KEY_*` tokens actually referenced in `source/`, with counts. |
| `keywords_junk.md` | The diff (used-but-not-canonical, 115 tokens) classified as junk / legitimate-undeclared / unsure. The clear junk has already been fixed (see the cleanup commit); this file documents what was found and what is left for review. |

## Building the Google Form

1. Open <https://script.google.com> and create a new project.
2. Paste the contents of `create_charmm_poll.gs` over the placeholder code.
3. Run the `createCharmmPoll` function and authorize it (it only needs
   permission to create a Form in your own Drive).
4. The execution log prints two links: the **published URL** to share with
   developers, and the **edit URL** for you. The Form also appears in your
   Drive.

Each survey section is rendered as one multiple-choice grid (features = rows,
usage scale = columns) so a developer can sweep a whole topic quickly. The
usage scale is: *use it / maintain it / don't use it / didn't know it existed*.

## Keyword-cleanup follow-ups (not yet done)

The clearly-bogus keywords (mangled `KEY_IF/ELSE/ENDIF/IFN` directives and the
`CADPACK`/`EXAND`/`LJMPE` typos) were removed in the cleanup commit. Still open
for a human decision in `keywords_junk.md`:

- **Dead `_old`/scaffold branches** — `KEY_MNDO97_old`, `KEY_OLD_IMCUBES`,
  `KEY_lsecd`: never defined, wrap stale code; delete after author confirms the
  bodies are not wanted.
- **Developer sentinels** — `KEY_UNUSED`, `KEY_NOTDEF`, `KEY_JUNK`, `KEY_BROKEN`,
  `KEY_GERHARD`, etc.: intentional "comment this out" idiom; consider replacing
  with `#if 0`.
- **Undeclared-but-real features** (~95) — legacy parallel/platform/QM tokens and
  genuine modules (ABPO, TMD, DIMS, EDS, MSCALE, RISM, APBS, ...) that the modern
  CMake build never defines: decide per item whether to re-add to the canonical
  list or formally retire.
- **ABPO wiring** — `--with-abpo` adds the `ENSEMBLE` keyword but not `ABPO`,
  while the source gates on `KEY_ABPO`; ABPO code is therefore never compiled.
  Needs a real fix + validation (tracked separately).
