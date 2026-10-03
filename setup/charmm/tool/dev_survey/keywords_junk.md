# Keyword diff: used-in-source but NOT in the canonical cmake list

Canonical list: 147 tokens (tool/cmake/prefx_keywords.cmake, see
`keywords_canonical.txt`).
Unique tokens referenced in `source/`: 262 (see `keywords_used.txt`).
Tokens used but not canonical: **115**.

Each is classified as one of:

- **(a) JUNK / mangling artifact** — a preprocessor mistake or stale token that
  should be removed or corrected. Confident.
- **(b) LEGITIMATE-but-undeclared** — a real feature/build/platform/QM keyword
  that the modern CMake build simply never defines, so the guarded code is
  dormant. NOT a typo. (Most of these are legacy parallel-variant, platform, or
  QM-interface keywords; the branches are effectively always-off in a normal
  build but are intentional, not mistakes.)
- **(c) UNSURE** — needs a human eyeball.

**DO NOT EDIT** anything based on this file alone; it is for hand review.

---

## (a) JUNK / mangling artifacts — confident

### A1. `KEY_IF`, `KEY_ELSE`, `KEY_ENDIF`, `KEY_IFN` — mangled `##IF/##ELSE/##ENDIF` directives

These are the smoking gun: an old prefx-style `##IF ... ##ELSE ... ##ENDIF`
block was run through an automated `.src -> .F90` converter that turned the bare
directive keywords into literal `#if KEY_<DIRECTIVE>==1` wrappers. Since no
`KEY_IF` / `KEY_ELSE` / `KEY_ENDIF` is ever defined, every one of these is an
always-false branch wrapping a blank or commented-out line. Pure dead cruft.

`KEY_IF==1` (active `#if`):
- source/misc/smbp.F90:1211 — `#if KEY_IF==1 || KEY_QCHEM==1 || KEY_G09==1`
- source/misc/ssbp.F90:58 — `#if KEY_IF==1 || KEY_SAVEFCM==1`
- source/misc/pbeq.F90:21562 — `#if KEY_IF==1 || KEY_PBEQ==1`
- source/io/mainio.F90:621 — `#if KEY_IF==1 || KEY_PARALLEL==1`
- source/pert/tsmh_ltm.F90:36, :55, :64 — `#if KEY_IF==1 || KEY_CMAP==1`

`KEY_IFN==1`:
- source/ltm/aaa_sanity_checks.F90:96 — `#if KEY_IFN==1 || KEY_PARALLEL==1`

`KEY_ELSE==1`:
- source/misc/pbeq.F90:12461 — `#if KEY_ELSE==1`
- source/pert/tsmh_ltm.F90:40 — `#if KEY_ELSE==1`

`KEY_ENDIF==1`:
- source/misc/smbp.F90:1216
- source/misc/ssbp.F90:62
- source/misc/pbeq.F90:12465, :22762
- source/ltm/aaa_sanity_checks.F90:101
- source/nbonds/ace.F90:79
- source/io/mainio.F90:625
- source/pert/tsmh_ltm.F90:44, :59, :68

**Proposed fix:** For the bare `KEY_ELSE==1` / `KEY_ENDIF==1` blocks: delete the
whole inert `#if ... #endif` (they wrap nothing but blanks/comments). For the
`KEY_IF==1 || KEY_REAL==1` forms: drop the leading `KEY_IF==1 || ` so the guard
becomes the intended `#if KEY_QCHEM==1 || KEY_G09==1` etc. — but verify the body
isn't already fully commented out (in tsmh_ltm.F90 / mainio.F90 / smbp.F90 /
ssbp.F90 the bodies ARE commented out, so the entire block can just be deleted).
Hand-review each; mechanical but needs eyes.

### A2. `KEY_EXAND` — typo of `EXPAND`, only in #else/#endif comment
- source/nbonds/nbonda7.inc:26 — `#else /* KEY_EXAND && KEY_PERT */`
- source/nbonds/nbonda7.inc:36 — `#endif /* KEY_EXAND && KEY_PERT */`

Appears ONLY inside the trailing C-comment on `#else`/`#endif`, never in an
active condition. The matching `#if` (check earlier in the file) uses the
correct `EXPAND`. Harmless misspelling of `EXPAND` in a comment.
**Proposed fix:** correct the comment text `EXAND` -> `EXPAND` (cosmetic).

### A3. `KEY_CADPACK` — typo of canonical `CADPAC`
- source/charmm/iniall.F90:314 — `#if KEY_CADPACK==1` guarding `call cadpac_init`

The CADPAC QM interface keyword is `CADPAC` (used 47x elsewhere). `CADPACK`
(with trailing K) is a one-off misspelling, so `cadpac_init` would never be
called even in a CADPAC build. Latent bug.
**Proposed fix:** `KEY_CADPACK` -> `KEY_CADPAC`.

### A4. `KEY_MNDO97_old` — stale/renamed token
- source/gukint/gukini.F90:592 — `#if KEY_MNDO97_old==1 /*nosquantm*/`

Never defined (canonical is `MNDO97`). Dead branch left over from a rename.
**Proposed fix:** delete the dead `_old` branch, or restore intent if the code
inside is still wanted under `MNDO97`. Needs author intent — but the token
itself is junk.

### A5. `KEY_OLD_IMCUBES` — stale variant of `IMCUBES`
- source/image/nbndgcm.F90:971 — `#if KEY_OLD_IMCUBES==1`

Never defined. Dead "old code path" branch superseded by `IMCUBES`.
**Proposed fix:** delete dead branch.

### A6. `KEY_LJMPE` — typo of `LJPME`, only in #endif comment
- source/nbonds/helpme_wrapper.F90:541 — `#endif /* KEY_LJMPE */`

Comment-only misspelling of canonical `LJPME` on a closing `#endif`.
**Proposed fix:** correct comment `LJMPE` -> `LJPME` (cosmetic).

### A7. `KEY_lsecd` — lowercase one-off (never defined)
- source/nbonds/enbondg.F90:1479 — `#if KEY_lsecd==1 /*lsecd2*/`
- source/nbonds/enbondg.F90:1498 — `#if KEY_lsecd==1 /*lsecd2*/`

Lowercase token; CHARMM keywords are uppercase and `lsecd` is in no list.
Always-off dead branch (developer scaffolding).
**Proposed fix:** delete dead branch (confirm with author it is not a
half-finished second-derivative path).

### A8. Developer dead-code-disabling sentinels (intentional, but never legal keywords)

The following are deliberately-undefined sentinels used as a "comment out this
block" idiom; the trailing comments (e.g. `/*..._unused*/`, `/*no_path_to_here*/`)
confirm intent. They are NOT preprocessor mistakes, but they ARE junk in the
sense that they pollute the KEY_ namespace and should arguably be replaced with
`#if 0`. Listing them so the reviewer can decide; LOW priority.

- `KEY_UNUSED` (12x) — e.g. source/misc/mmfp.F90:2418, source/misc/nmr.F90:759,
  source/energy/inertia.F90:6, source/mmff/assignpar.F90:896,
  source/pert/tsme.F90:1520,:1611, source/rxncor/travel2.F90:2725,
  source/rxncor/path.F90:224,:303, source/quantum/qmjunc.F90:100 (+others)
- `KEY_NOTDEF` (8x) — source/energy/ecmap.F90:994,1057,1217,1313,1321,1401,1497,1505
- `KEY_JUNK` (2x) — source/dynamc/dynutil.F90:1676,:1693
- `KEY_BROKEN` (2x) — source/zerom/zerom2.F90:1642,:1732
- `KEY_NOTINEWMOD` (1x) — source/nbonds/ewaldf.F90:1
- `KEY_NOSKULL` (1x, `==0`) — source/machdep/machutil.F90:280
- `KEY_SMDbad` (1x) — source/rxncor/rxndef.F90:756 (also lowercase tail)
- `KEY_GERHARD` (1x) — source/misc/freene_calc.F90:2482 (developer name marker)
- `KEY_SGGRID` (1x, commented `!#if`) — source/dynamc/sglds.F90:1500

**Proposed fix:** replace each with `#if 0 /* dead: ... */` (or delete) to free
the KEY_ namespace. Not urgent. Hand-review; keep conservative.

---

## (b) LEGITIMATE-but-undeclared — real keywords the modern build never defines

These are genuine historical feature / build-variant / platform / QM-interface
keywords. The CMake build simply does not emit `-D` for them, so their `#if`
branches are dormant. They are NOT typos and should NOT be blindly deleted — but
they are candidates for either (i) re-adding to the canonical list if the feature
is still wanted, or (ii) a separate dead-code-removal pass if the feature is
truly retired. Grouped by kind:

**Parallel / decomposition variants (legacy MPI build matrix):**
PARASCAL (468), SPACDEC (273), MTS (353), PARINFNTY (23), PARASCC (11),
PARCMD (6), ALLMPI (6), CMPI (101), CMPI is paired with PARALLEL, SYNCHRON (28),
ASYNC_PME (14), MPIFFTZ (2), GENCOMM is canonical(skip), NO_BYCC (11),
NO_BYCU (2), NOPARASWAP (2), NONETWORK (3), IMPI (2), PRLLOUT (2).
  (These were historically added via `-a` keyword sets or alternate parallel
   builds; in the modern default build they are off. MTS specifically is still
   recognized by CMakeLists.txt:1448 as an `-a MTS` add-on incompatible with
   DOMDEC — so MTS is a real, intentionally-add-only keyword.)

**Precision / integer build variants:**
SINGLE (186), INTEGER8 (234), RMSDDBL (8).
  (single-precision / 64-bit-integer builds; controlled by other CMake paths,
   not the keyword list.)

**Platform / OS / compiler tokens (legacy, set by old install.com, not CMake):**
EM64T (1), OSX (21), WIN32 (13), WIN64 (9), OS2 (1), XT4 (1), G95 (5),
GFORTRAN (1), GHO (1).
  (Old platform selectors. Modern CMake detects platform differently. Mostly
   dead but historically legitimate.)

**QM/MM interfaces not wired into the modern CMake keyword list:**
CADPAC (47), GAMESSUK (176), GAUSSIAN (4), QUANTA (3), CHARMMRATE (4),
DFTBPLUS (11), DFTBMKL is canonical(skip), GHO (1).
  (CADPAC, GAMESS-UK, Gaussian, QUANTA, CHARMM-RATE, DFTB+ QM interfaces. These
   are real features whose enabling flags were never ported to the new
   configure FEATURES dict. Strong candidates for "re-add to canonical or
   formally retire" review.)

**External-library / math-backend selectors:**
EISPACK (5), INTELMKL (4), MKLLIB (1), LBMASSV (4), CRAY_1DFFT (2),
ANNLIB (4), APBS (4), RISM (18), GRAPE is canonical(skip).
  (Alternate eigensolver / FFT / Poisson-Boltzmann backends. APBS and RISM are
   real solvation features; EISPACK/LBMASSV/CRAY_1DFFT are legacy math backends.)

**Real feature modules missing from the canonical list (highest-value finds):**
- ABPO (22) — adaptively biased path optimization (configure has `abpo`
  feature, but it maps to the ENSEMBLE keyword, not an ABPO keyword; source uses
  `KEY_ABPO`). **Mismatch worth reviewing.**
- EABF is canonical(skip).
- ACTBOND (42) — active-bond / reactive bookkeeping.
- ADUMBRXNCOR (15) — ADUMB x RXNCOR coupling (composite of two canonical keys).
- DOMDEC_MMFF (15) — DOMDEC + MMFF combination.
- DIMS (49) — dynamic importance sampling MD.
- EDS (21) — enveloping distribution sampling.
- TMD (47) — targeted MD (note: distinct from canonical TAMD).
- ZEROM (25) — zero-order / ZeroM module.
- MODELLER (49) — MODELLER interface.
- MSCALE (22) — MSCALE multiscale driver.
- POLAR (10) — polarizable model branch.
- CSA (7), DISTENE (5), MCMA (5), SAMC (21) — conformational/MC sampling
  variants.
- CORSOL (6) — correlated-solvent.
- CVELOCI (11), IPRESS (11) — constant-velocity / pressure variants.
- HFB is canonical(skip).
- MIDSINR (90) — image/nonbond inner-loop variant (heavily used!).
- MOBHY (8), PINS (9), PNM (15), VIBPARA (29), TIMER_DEBUG (9),
  DBGTIMER (4), DEBUGNMD (17), DEBUGREPD (26), GBMVDEBUG (24), MTPL_DEBUG (17),
  REPDEB (30) — debug/timer instrumentation toggles.
- NOMISC (118), NOCONVERT (16), NOST2 (59) — feature-exclusion toggles.
- LICENSE (15) — license-gate branch.
- AVIMODS (6), FASTENBFS8 (4), FEWMFC (3), FEWSB (3), LNTAB1 (2),
  CKSHKTOL (1), DISTENE (5) — nonbond/Ewald/shake micro-optimizations.
- DRUDE (1) — Drude polarizable (likely subsumed by other build paths now).
- CGENFF (1, in dimens_ltm.F90 as dimension selector) — note CGENFF *is*
  canonical; this single extra ref is a dimension-sizing branch.

**Vendor / IDE / misc markers (almost certainly removable, but verify):**
VSCODE (4), NAMD (1), DISCOVER (1), INSIGHT (1), YAMMP (1), AMBER (2),
SGGRID (already in A8), FILEINPUT (1), FILEOUTPUT (1), CONFIGURE is
canonical(skip).
  - `KEY_VSCODE` (source/nbonds/enbips.F90:4757,:4763) — an editor-name guard;
    almost certainly developer scaffolding. Borderline JUNK — flagged here as
    UNSURE-leaning-junk; **recommend review/delete.**
  - `KEY_NAMD` (source/ltm/consta_ltm.F90:69), `KEY_DISCOVER`
    (consta_ltm.F90:66), `KEY_AMBER` (consta_ltm.F90:63),
    `KEY_INSIGHT`/`KEY_XT4` (source/charmm/iniall.F90:626,:638),
    `KEY_YAMMP` (source/ltm/dimens_ltm.F90:155) — legacy interop / unit-system
    and array-sizing selectors for other MD packages. Dormant; historically
    legitimate.
  - `KEY_FILEINPUT`/`KEY_FILEOUTPUT` (source/machdep/machio.F90:123,:135) —
    legacy I/O-redirection build switches.

`NODISPLAY` (11): used in source but in the cmake list it is only a REMOVE_ITEM
*target* and is never APPENDed -> never defined. So `#if KEY_NODISPLAY==1`
branches are always-off. The paired `NOGRAPHICS`/`XDISPLAY` ARE handled. This is
a build-logic gap, not a source typo. **Review:** either add NODISPLAY to the
base list (so the no-graphics path can be selected) or retire it.

`RESIZE` (370): heavily used dynamic-array-resize machinery; never in the
keyword list (it is unconditionally-on in practice via other means). Treat as
LEGITIMATE-but-undeclared; do not touch — see existing "resize heap corruption"
ASan workstream in project memory.

`DEBUG` (210): generic `#if KEY_DEBUG==1` developer-debug toggle, never defined
by CMake. Legitimate developer switch (enable via `-a DEBUG`); not junk.

---

## (c) UNSURE — needs a human

- `KEY_VSCODE` — see above; leans JUNK but listed UNSURE pending author confirm.
- `KEY_ABPO` vs the `abpo` configure feature mapping to `ENSEMBLE` — possible
  wiring bug (feature flag never defines the keyword the source checks).
- `KEY_GHO` (source/gukint/gukini.F90:967) — GHO QM/MM boundary; single ref,
  may be intentionally gated by another keyword.
- `KEY_PM1` appears canonical(NOT-lite list) yet is also used with a
  `/*PM1orPM6*/` comment (source/energy/polar.F90:1022) — confirm PM1 is the
  intended spelling and not meant to be a broader semiempirical toggle.
- The full A8 sentinel set: decide policy (`#if 0` vs delete) project-wide.
