// COSMO Lab (EPFL) interview — MMML terminal deck
// Compile: typst compile --root .. presentations/cosmo_epfl_mmml.typ
//   (from presentations/)  OR: typst compile presentations/cosmo_epfl_mmml.typ

#import "@preview/polylux:0.4.0": *
#import "@preview/sicons:16.0.0": sicon

// ── Palette (GitHub dark + git semantics) ───────────────────────────────────
#let bg       = rgb("#0d1117")
#let surface  = rgb("#161b22")
#let surface2 = rgb("#21262d")
#let border   = rgb("#30363d")
#let fg       = rgb("#c9d1d9")
#let muted    = rgb("#8b949e")
#let prompt   = rgb("#3fb950")
#let path     = rgb("#58a6ff")
#let cmd      = rgb("#f0f6fc")
#let comment  = rgb("#8b949e")
#let keyword  = rgb("#ff7b72")
#let string   = rgb("#a5d6ff")
#let git-add  = rgb("#3fb950")
#let git-pr   = rgb("#8957e5")
#let git-issue= rgb("#3fb950")
#let git-branch= rgb("#d29922")
#let git-merge= rgb("#a371f7")
#let ok       = rgb("#3fb950")
#let phase    = rgb("#79c0ff")
#let dot-red   = rgb("#ff5f57")
#let dot-yell  = rgb("#febc2e")
#let dot-green = rgb("#28c840")

#let mono = "DejaVu Sans Mono"
#let sans = "DejaVu Sans"
#let icons = "icons"

#set page(
  paper: "presentation-16-9",
  margin: (x: 0.9cm, y: 0.65cm),
  fill: bg,
  footer: context {
    set text(font: mono, size: 9pt, fill: muted)
    align(right)[#toolbox.slide-number / #toolbox.last-slide-number]
  },
)
#set text(font: mono, size: 11pt, fill: fg)

// ── Icons (simple-icons via sicons + local jax.svg) ─────────────────────────
#let ico(slug, size: 12pt, color: "default") = {
  sicon(slug: slug, size: size, icon-color: color)
}

#let ico-file(file, size: 13pt) = {
  image(icons + "/" + file, width: size, height: size)
}

#let ico-label(slug, label, size: 12pt, color: "default") = {
  grid(
    columns: (auto, auto),
    column-gutter: 4pt,
    align: horizon,
    ico(slug, size: size, color: color),
    text(size: 10pt, fill: muted)[#label],
  )
}

#let stack-icons = align(center)[
  #grid(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    column-gutter: 14pt,
    align: horizon,
    ico("github", color: "ffffff"),
    ico("python", color: "3776AB"),
    ico-file("jax.svg"),
    ico("fortran", color: "734F96"),
    ico("nvidia", color: "76B900"),
    ico("pytorch", color: "EE4C2C"),
    ico("git", color: "F05032"),
  )
]

#let term-title(label) = {
  grid(
    columns: (auto, 1fr),
    column-gutter: 8pt,
    align: horizon,
    text(size: 9pt)[
      #text(fill: dot-red)[●] #text(fill: dot-yell)[●] #text(fill: dot-green)[●]
    ],
    align(right)[
      #text(size: 9pt, fill: muted)[#label]
    ],
  )
  v(0.12em)
  line(length: 100%, stroke: 0.5pt + border)
  v(0.3em)
}

#let term(body) = block(
  width: 100%,
  fill: surface,
  stroke: 0.75pt + border,
  radius: 6pt,
  inset: (x: 12pt, y: 8pt),
)[
  #term-title("mmml — zsh")
  #body
]

#let prompt-char = text(fill: prompt)[\$ ]
#let prompt-line(body) = {
  prompt-char
  body
  v(0.08em)
}

#let out-line(body) = {
  text(fill: muted, size: 11pt)[#body]
  v(0.06em)
}

#let cmnt-prefix = "// "
#let cmnt(body) = {
  text(fill: comment, size: 10.5pt)[#cmnt-prefix#body]
  v(0.06em)
}

#let gh-link(url, label) = link(url)[#text(fill: path, size: 11pt)[#label]]

#let link-row(slug, url, label, color: "ffffff", file: none) = {
  grid(
    columns: (auto, auto),
    column-gutter: 0.35em,
    align: horizon,
    if file != none {
      ico-file(file, size: 11pt)
    } else {
      ico(slug, size: 11pt, color: color)
    },
    gh-link(url, label),
  )
  v(0.1em)
}

#let gh-repo(url, label: "EricBoittier/mmml") = link-row(
  "git", url, label, color: "F05032",
)

#let gh-pr(num, url, additions: none, state: "open") = {
  block(
    inset: (x: 8pt, y: 5pt),
    radius: 4pt,
    fill: surface2,
    stroke: (left: 2.5pt + git-pr, rest: 0.5pt + border),
  )[
    #set text(fill: fg)
    #ico("git", size: 11pt, color: "F05032")
    #h(0.2em)
    #text(fill: git-pr, weight: "bold", size: 11pt)[PR]
    #h(0.25em)
    #link(url)[
      #text(fill: path, size: 11pt)[#("metatensor/metatomic#" + str(num))]
    ]
    #if additions != none [
      #h(0.35em)
      #text(fill: git-add, size: 11pt)[+#str(additions)]
    ]
    #h(0.25em)
    #text(size: 10pt, fill: git-pr)[● #state]
  ]
}

#let gh-issue(num, url) = {
  block(
    inset: (x: 8pt, y: 5pt),
    radius: 4pt,
    fill: surface2,
    stroke: (left: 2.5pt + git-issue, rest: 0.5pt + border),
  )[
    #set text(fill: fg)
    #ico("github", size: 11pt, color: "ffffff")
    #h(0.2em)
    #text(fill: git-issue, weight: "bold", size: 11pt)[issue]
    #h(0.25em)
    #link(url)[#text(fill: path, size: 11pt)[#str(num)]]
  ]
}

#let ci-step(name, status) = {
  grid(
    columns: (auto, auto, 1fr, auto),
    column-gutter: 5pt,
    align: horizon,
    ico("githubactions", size: 10pt, color: "ffffff"),
    text(fill: ok)[✓],
    text(size: 10.5pt)[#name],
    text(size: 10pt, fill: ok)[#status],
  )
  v(0.05em)
}

#let slide-h1(body) = {
  text(font: sans, size: 18pt, weight: "bold", fill: cmd)[#body]
  v(0.2em)
}

#let phase-tag(slug, label, color: "default", file: none) = {
  grid(
    columns: (auto, auto),
    column-gutter: 5pt,
    align: horizon,
    if file != none { ico-file(file) } else { ico(slug, size: 12pt, color: color) },
    text(fill: phase, weight: "bold", size: 11pt)[#label],
  )
  v(0.08em)
}

#let cmd-line(kw, rest) = {
  text(fill: keyword, size: 11pt)[#kw]
  text(fill: cmd, size: 11pt)[#rest]
  v(0.04em)
}

// ── Slide 1: Title · links · contributors ─────────────────────────────────────

#slide[
  #grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    term[
      #align(center)[
        #text(font: sans, size: 24pt, weight: "bold", fill: cmd)[MMML]
        #v(0.12em)
        #stack-icons
        #v(0.12em)
        #text(font: sans, size: 9.5pt, fill: phase)[
          end-to-end MLIP toolkit
        ]
      ]
      #v(0.1em)
      #cmnt[data · train · MD · spectra · deploy · 31+ CLI commands]
      #prompt-line[#text(fill: keyword)[mmml --help]]
      #cmnt[COSMO Lab · EPFL]
    ],
    term[
      #link-row("git", "https://github.com/EricBoittier/mmml", "EricBoittier/mmml", color: "F05032")
      #link-row("readthedocs", "https://mmml.readthedocs.io/en/latest/", "docs", color: "8CA0AF")
      #link-row("github", "https://github.com/EricBoittier/mmml_tutorial", "EricBoittier/mmml_tutorial", color: "ffffff")
      #link-row("githubactions", "https://github.com/EricBoittier/mmml/actions", "CI", color: "ffffff")
      #v(0.08em)
      #cmnt[
        contributions from Sena Aydin, Khamlek Chaton, and everyone who has built MMML
      ]
    ],
  )
]

// ── Slide 2: Full MLIP workflow ───────────────────────────────────────────────

#slide[
  #slide-h1[MLIP workflow — generate · train · simulate · analyze]

  #grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    term[
      #phase-tag("nvidia", "GPU reference data", color: "76B900")
      #cmd-line("mmml pyscf-dft", "") #cmd-line("mmml pyscf-mp2", "")
      #cmd-line("mmml pyscf-evaluate", "") #cmd-line("mmml normal-mode-sample", "")
      #cmd-line("mmml xml2npz", "") #cmd-line("mmml fix-and-split", "")
      #cmd-line("mmml validate", "") #cmd-line("mmml active-learning", "")
      #v(0.06em)
      #phase-tag("python", "train & evaluate", color: "3776AB")
      #cmd-line("mmml physnet-train", "") #cmd-line("mmml train-joint", "")
      #cmd-line("mmml ef-train", "") #cmd-line("mmml physnet-evaluate", "")
    ],
    term[
      #phase-tag("fortran", "MD & hybrid MM/ML", color: "734F96")
      #ico-file("jax.svg", size: 11pt)
      #cmd-line("mmml md-system", "") #cmd-line("mmml physnet-md", "")
      #cmd-line("mmml ef-md", "") #cmnt[CHARMM MLpot · JAX-MD · λ-dynamics TI]
      #v(0.06em)
      #phase-tag("python", "spectra & conformers", color: "3776AB")
      #cmd-line("mmml-spectra-md", "") #cmnt[IR · VCD · Raman]
      #cmd-line("mmml orca-server", "") #cmnt[ExtOpt GOAT]
      #cmd-line("mmml gui", "") #cmd-line("mmml lambda-mbar", "")
      #v(0.06em)
      #text(size: 9pt, fill: muted, style: "italic")[
        Hybrid CHARMM integration is one deployment path — not the whole repo.
      ]
    ],
  )
]

// ── Slide 3: Engineering · MetaTensor / ORCA ──────────────────────────────────

#slide[
  #slide-h1[Atomistic ML software — MMML today, Metatomic next?]

  #grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    term[
      #phase-tag("githubactions", "CI/CD · tests", color: "ffffff")
      #ci-step("python -m build", "pass")
      #ci-step("pytest tests/", "pass")
      #ci-step("mkdocs build --strict", "pass")
      #out-line[
        #text(fill: git-add)[65+]
        #text(fill: muted)[ unit tests · ]
        #text(fill: path)[pycharmm]
        #text(fill: muted)[ | ]
        #text(fill: path)[gpu]
        #text(fill: muted)[ | ]
        #text(fill: path)[mlpot]
      ]
      #cmnt[Python 3.13 · uv · JAX JIT · Pydantic · NPZ schema · Ruff · mypy]
    ],
    term[
      #cmnt[metatensor/metatomic · ORCA ExtOpt]
      #grid(
        columns: (1fr, 1fr),
        gutter: 6pt,
        gh-issue(228, "https://github.com/metatensor/metatomic/issues/228"),
        gh-pr(262, "https://github.com/metatensor/metatomic/pull/262", additions: 1328),
      )
      #v(0.06em)
      #grid(
        columns: (auto, 1fr),
        column-gutter: 6pt,
        align: horizon,
        ico("pytorch", size: 11pt, color: "EE4C2C"),
        out-line[PR 262: metatomic-orca-server / client],
      )
      #grid(
        columns: (auto, 1fr),
        column-gutter: 6pt,
        align: horizon,
        ico-file("jax.svg", size: 11pt),
        out-line[MMML: mmml orca-server · GPU batch · FastAPI],
      )
      #v(0.06em)
      #block(
        inset: (x: 6pt, y: 4pt),
        radius: 4pt,
        fill: surface2,
        stroke: (left: 2.5pt + git-merge, rest: 0.5pt + border),
      )[
        #text(size: 10pt, fill: fg)[
          Same ExtOpt protocol in JAX (MMML) and metatomic-torch → COSMO
        ]
      ]
    ],
  )
]
