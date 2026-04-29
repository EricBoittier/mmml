// Methanol dimer: lambda TI + 2D cutoff scan — figure summary
// Compile:  typst compile --root /path/to/mmml /path/to/mmml/reports/meoh_dimer_lambda_ti_and_scan.typ

#set document(title: [Methanol dimer: \u{03bb} TI and 2D cutoff scan])
#set page(margin: 2cm, numbering: "1")
#set text(size: 11pt)
#set heading(numbering: "1.")

#let ti = json("../artifacts/meoh_dimer_lambda_ti/lambda_ti_summary.json")
#let scan = json("../artifacts/meoh_dimer_2d_cutoff_scan/scan_2d_summary.json")

#let img(path, w: 100%) = image(path, width: w)

#align(center)[
  #text(size: 17pt, weight: "bold")[Methanol dimer analysis report]
  #v(0.5em)
  #text(size: 12pt, style: "italic")[MMML \u{03bb}-dynamics TI trajectories and 2D cutoff scans]
  #v(0.75em)
  #text(size: 10pt, fill: gray.darken(20%))[
    Source: `artifacts/meoh_dimer_lambda_ti/` and `artifacts/meoh_dimer_2d_cutoff_scan/` \
    Generated: #datetime.today().display()
  ]
]

#v(1em)
#outline(depth: 2, indent: auto)
#pagebreak()

= Lambda dynamics and thermodynamic integration

The trajectory workflow (`scripts/meoh_dimer_lambda_ti.py`) runs NVE sampling at fixed \u{03bb} windows on the methanol dimer with the MMML hybrid calculator. Per-window samples estimate $chevron.l partial U \/ partial lambda chevron.r$ (via the inter-monomer energy difference between \u{03bb}-on and \u{03bb}-off calculators at the same geometry). Thermodynamic integration (TI) integrates these means over \u{03bb}. Multistate Bennett acceptance ratio (MBAR) recomputes full hybrid energies on stored snapshots for a coupled free-energy estimate and uncertainty.

== Geometry (from run summary)

- Initial COM separation: #ti.geometry.at("initial_com_separation_A")~\u{00C5}
- Seed COM separation: #ti.geometry.at("seed_com_separation_A")~\u{00C5}

== Free-energy summary (this artifact run)

#align(center)[
  #table(
    columns: (1fr, 1fr, 1fr),
    inset: 8pt,
    stroke: 0.5pt + gray,
    [*Quantity*], [*Value*], [*Note*],
    [TI $Delta F_"couple"$], [#ti.at("delta_F_couple_eV")~eV \ (#ti.at("delta_F_couple_kcal_mol")~kcal/mol)], [Binding path $integral_0^1 chevron.l partial U \/ partial lambda chevron.r thin d lambda$],
    [TI $Delta F_"diss"$], [#ti.at("delta_F_diss_eV")~eV], [Negative of coupling path],
    [MBAR $Delta F_"couple"$], [
      #ti.mbar.at("Delta_F_couple_eV") $plus.minus$ #ti.mbar.at("dDelta_F_couple_eV")~eV \
      (#ti.mbar.at("Delta_F_couple_kcal_mol") $plus.minus$ #ti.mbar.at("dDelta_F_couple_kcal_mol")~kcal/mol)
    ], [From `pymbar`; uncertainties in last column],
  )
]

#figure(
  img("../artifacts/meoh_dimer_lambda_ti/ti_components_per_window.png", w: 92%),
  caption: [Mean $partial U \/ partial lambda$ (eV) per \u{03bb} window with SEM and window standard deviation.],
)

#figure(
  img("../artifacts/meoh_dimer_lambda_ti/ti_repeat_components_per_window.png", w: 92%),
  caption: [Repeat-level means (gray) and window mean (red): sampling spread across repeats pooled into MBAR.],
)

#figure(
  img("../artifacts/meoh_dimer_lambda_ti/mbar_per_window_diagnostics.png", w: 92%),
  caption: [MBAR diagnostics: raw snapshot counts $N_k$, effective samples after decorrelation $N_k^"eff"$, and statistical inefficiency $g_k$ per window.],
)

#pagebreak()

= Two-dimensional cutoff scan (distance $times$ \u{03bb})

The scan script (`scripts/scan_meoh_dimer_2d_cutoffs.py`) grids COM distance and alchemical \u{03bb}, reporting MMML energy decomposition and force forms $-partial E \/ partial d$. Heatmaps use robust percentile color limits; vertical slices mark the complementary handoff interval between ML taper and MM switch.

== Scan parameters (from `scan_2d_summary.json`)

- $d in [#scan.args.at("dmin"), #scan.args.at("dmax")]$~\u{00C5}, #scan.args.at("n_dist") distance points
- ML cutoff width: #scan.args.at("ml_cutoff")~\u{00C5}, MM switch end: #scan.args.at("mm_switch_on")~\u{00C5}, MM outer scale: #scan.args.at("mm_cutoff")~\u{00C5}
- Handoff interval: [#scan.args.at("cutoff_region_start_A"), #scan.args.at("cutoff_region_end_A")]~\u{00C5}

== Total and interaction energies along distance

#figure(
  img("../artifacts/meoh_dimer_2d_cutoff_scan/total_energy_vs_distance.png", w: 88%),
  caption: [Total hybrid energy vs.\ COM distance at representative \u{03bb} values.],
)

#figure(
  img("../artifacts/meoh_dimer_2d_cutoff_scan/interaction_energy_vs_distance.png", w: 88%),
  caption: [Interaction energy ($E_"tot" - E_"internal"$) vs.\ distance.],
)

== Two-dimensional energy maps

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(img("../artifacts/meoh_dimer_2d_cutoff_scan/E_total_2d.png", w: 100%), caption: [$E_"tot"(d, lambda)$]),
  figure(img("../artifacts/meoh_dimer_2d_cutoff_scan/E_interaction_2d.png", w: 100%), caption: [Interaction $E_"tot" - E_"internal"$]),
)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(img("../artifacts/meoh_dimer_2d_cutoff_scan/E_internal_2d.png", w: 100%), caption: [Internal ML monomer energy]),
  figure(img("../artifacts/meoh_dimer_2d_cutoff_scan/E_ml2b_2d.png", w: 100%), caption: [ML two-body (dimer) contribution]),
)

#figure(
  img("../artifacts/meoh_dimer_2d_cutoff_scan/E_mm_2d.png", w: 55%),
  caption: [MM inter-monomer contribution (neighbor list excludes intra-monomer pairs).],
)

== Force forms $-partial E \/ partial d$

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(img("../artifacts/meoh_dimer_2d_cutoff_scan/F_total_2d.png", w: 100%), caption: [$-partial E_"tot" \/ partial d$]),
  figure(img("../artifacts/meoh_dimer_2d_cutoff_scan/F_internal_2d.png", w: 100%), caption: [$-partial E_"internal" \/ partial d$]),
)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(img("../artifacts/meoh_dimer_2d_cutoff_scan/F_ml2b_2d.png", w: 100%), caption: [$-partial E_"ml2b" \/ partial d$]),
  figure(img("../artifacts/meoh_dimer_2d_cutoff_scan/F_mm_2d.png", w: 100%), caption: [$-partial E_"mm" \/ partial d$]),
)

== \u{03bb}-effect maps ($lambda_max - lambda_min$)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(img("../artifacts/meoh_dimer_2d_cutoff_scan/DeltaLambda_E_total.png", w: 100%), caption: [$Delta_lambda E_"tot"$ along $d$]),
  figure(img("../artifacts/meoh_dimer_2d_cutoff_scan/DeltaLambda_E_ml2b.png", w: 100%), caption: [$Delta_lambda E_"ml2b"$]),
)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(img("../artifacts/meoh_dimer_2d_cutoff_scan/DeltaLambda_E_mm.png", w: 100%), caption: [$Delta_lambda E_"mm"$]),
  figure(img("../artifacts/meoh_dimer_2d_cutoff_scan/DeltaLambda_F_total.png", w: 100%), caption: [$Delta_lambda F_"tot"$]),
)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(img("../artifacts/meoh_dimer_2d_cutoff_scan/DeltaLambda_F_ml2b.png", w: 100%), caption: [$Delta_lambda F_"ml2b"$]),
  figure(img("../artifacts/meoh_dimer_2d_cutoff_scan/DeltaLambda_F_mm.png", w: 100%), caption: [$Delta_lambda F_"mm"$]),
)

== Cutoff-region slices

#figure(
  img("../artifacts/meoh_dimer_2d_cutoff_scan/cutoff_component_slices.png", w: 92%),
  caption: [Energy components vs.\ distance at three \u{03bb} values; vertical lines mark handoff start/end.],
)

#figure(
  img("../artifacts/meoh_dimer_2d_cutoff_scan/cutoff_forceform_slices.png", w: 92%),
  caption: [Force forms vs.\ distance with the same cutoff markers.],
)

= Data files

- TI: `artifacts/meoh_dimer_lambda_ti/lambda_ti_summary.json`, trajectories under `artifacts/meoh_dimer_lambda_ti/trajectories/`
- Scan: `artifacts/meoh_dimer_2d_cutoff_scan/scan_2d_summary.json`, `scan_2d_components.csv`
