/**
 * create_charmm_poll.gs
 * ─────────────────────────────────────────────────────────────────────
 * Google Apps Script that builds the "CHARMM Feature Usage Survey"
 * Google Form automatically.
 *
 * HOW TO RUN
 *   1. Go to https://script.google.com  ->  New project.
 *   2. Delete the placeholder code, paste this whole file in.
 *   3. Click  Run  (the createCharmmPoll function).
 *   4. Authorize the script when prompted (it needs permission to create
 *      a Form in your Drive).  This is your own account; nothing is shared
 *      until you choose to.
 *   5. The Execution log prints the live form URL (to share with
 *      developers) and the edit URL (for you).  The Form is also dropped
 *      in the root of your Google Drive.
 *
 * Re-running creates a brand new Form each time.
 *
 * Each section is rendered as a single multiple-choice GRID: features are
 * the rows, the usage scale is the columns, so a developer can sweep
 * through a whole topic in a few clicks instead of answering dozens of
 * separate questions.
 * ─────────────────────────────────────────────────────────────────────
 */

// The usage scale, used as the columns of every grid.
var SCALE = [
  'I actively USE it',
  'I develop / MAINTAIN it',
  'I DO NOT use it',
  "Didn't know it existed",
];

// Survey content: one object per section, each with grid rows.
// Row label format: "Human name (KEY_* keywords)".
var SECTIONS = [
  {
    title: 'QM/MM',
    help: 'Quantum-mechanical / molecular-mechanical interfaces and boundaries.',
    rows: [
      'SCC-DFTB semiempirical QM (SCCDFTB)',
      'GAMESS / GAMESS-UK ab initio QM (GAMESS)',
      'Q-Chem ab initio QM (QCHEM)',
      'Gaussian 09 QM/MM (G09)',
      'MNDO97 semiempirical QM (MNDO97)',
      'Built-in QUANTUM / SQuanTM / TURBOMOLE / semiempirical (QUANTUM, SQUANTM, QTURBO, QMMMSEMI)',
      'QM/MM solvent-boundary potentials GSBP / SMBP (GSBP, SMBP)',
    ],
  },
  {
    title: 'Free energy & enhanced sampling',
    help: 'Alchemical free energy, biasing, replica and path methods.',
    rows: [
      'Free-energy perturbation: PERT / TSM (PERT, TSM, CHEMPERT)',
      'Adaptive umbrella & reaction-coordinate biasing (ADUMB, RXNCOR, RXNCONS)',
      'Replica & path methods (REPLICA, RPATH, PATHINT, CPATH, TPS)',
      'Distributed replica exchange (REPDSTR, GENCOMM)',
      'Ensemble / ABPO methods (ENSEMBLE)',
      'The string method (STRINGM)',
      'Multicanonical / Tsallis sampling (MULTCAN, TSALLIS)',
      'Accelerated / biased sampling: GAMUS, eABF, AFM (GAMUS, EABF, DENBIAS, HQBM, AFM, AXD)',
      'Steered / targeted MD (SMD)',
      'Monte Carlo & grand-canonical MC (MC, MEHMC, GCMC, GENETIC)',
    ],
  },
  {
    title: 'Electrostatics & implicit solvent',
    help: 'Continuum and implicit-solvent electrostatics models.',
    rows: [
      'Poisson-Boltzmann electrostatics (PBEQ)',
      'Generalized-Born implicit solvent: GB/GBSW/GBMV (GBFIXAT, GBINLINE, HDGBVDW, DHDGB)',
      'Analytic implicit solvent: ACE / FACTS / SASA / SCP-ISM / ASP (ACE, FACTS, SASAE, SCPISM, ASPENER, ASPMEMB)',
      'Shell / radial solvation models (SHELL, RDFSOL)',
      'RISM integral-equation solvent (RISM)',
    ],
  },
  {
    title: 'Force fields & energy terms',
    help: 'Non-default force fields and specialized energy terms.',
    rows: [
      'Alternate force fields: MMFF / CFF / OPLS / CGenFF (MMFF, CFF, OPLS, CGENFF)',
      'Polarizable / advanced electrostatics: CHEQ / FlucQ / PIPF / multipoles (CHEQ, FLUCQ, PIPF, MTPL)',
      'Constant-pH MD (PHMD)',
      'Special bonded terms: VALBOND / CMAP / lone pairs / MMPT (VALBOND, CMAP, LONEPAIR, MMPT, MSMMPT)',
      'VdW long-range & soft-core: LRVDW / WCA / soft-core / IPS / LJ-PME (LRVDW, WCA, SOFTVDW, NBIPS, LJPME)',
      'Fast Ewald / PME / FMM backends (FASTEW, FMA, COLFFT)',
      'Data-driven & NMR-restraint energy terms: GNN / LarmorD / RDC / NOE / SSNMR / PMF (GNN, LARMORD, RDC, PNOE, SSNMR, EPMF, ESTATS)',
    ],
  },
  {
    title: 'Restraints, geometry & manipulation',
    help: 'Restraints, structure comparison, docking and density fitting.',
    rows: [
      'Geometric / consensus restraints (HMCOM, RGYCONS, DMCONS, CONSHELIX)',
      'Structure overlap / shape / PRIMO (OVERLAP, SHAPES, PRIMO, PRIMSH)',
      'Docking & grid potentials (DOCK, GRID, FFTDOCK)',
      '4D sampling & energy lookup (FOURD, LOOKUP)',
      'EMAP density-map fitting (EMAP)',
    ],
  },
  {
    title: 'Dynamics, modes & fitting',
    help: 'Integrators, normal modes, and parameter fitting.',
    rows: [
      'Integrator & dynamics variants: VV2 / SGLD / PBC / fast-SHAKE / TNPACK / MRMD (DYNVV2, OLDDYN, SGLD, PBOUND, FSSHK, TNPACK, MRMD)',
      'Normal modes & vibrational analysis (DIMB, MOLVIB)',
      'Charge / parameter fitting (FITCHG, FLEXPARM)',
      'Reaction-path search: TRAVEL / RMD (TRAVEL, RMD)',
    ],
  },
  {
    title: 'GPU & acceleration',
    help: 'GPU engines and accelerator backends.',
    rows: [
      'OpenMM offload (+ OpenMM-Torch) (OPENMM, OMMTORCH)',
      'BLaDE GPU engine (BLADE)',
      'GPU/accelerator backends: CUDA / OpenCL / Metal / GRAPE (CUDA, OPENCL, METAL, GRAPE)',
      'Domain decomposition, incl. GPU (DOMDEC, DOMDEC_GPU)',
    ],
  },
  {
    title: 'Build / parallel / platform (build maintainers)',
    help: 'Answer only if you build or package CHARMM yourself; otherwise feel free to skip this section.',
    rows: [
      'MPI parallel build (MPI, PARALLEL, PARAFULL)',
      'FFT / math backend: MKL vs FFTW (MKL, FFTW)',
      'Interactive graphics / no-graphics (XDISPLAY, NOGRAPHICS)',
      'Platform / packaging switches (STATIC, EXPAND, LONGLINE, NIH)',
    ],
  },
];

// Free-text prompts shown at the end.
var CLOSING = [
  'Which CHARMM feature do you rely on most that you would be unhappy to lose?',
  'Are there features above you believe should be retired or merged?',
  'Anything else we should know about how you build or use CHARMM?',
];

function createCharmmPoll() {
  var form = FormApp.create('CHARMM Feature Usage Survey');

  form.setDescription(
    'For each CHARMM feature below, tell us your relationship to it. This ' +
    'helps the development team decide what to keep, modernize, document, or ' +
    'retire.\n\n' +
    '"Develop / maintain" means you touch its source code; "use" means you ' +
    'run it in your science. Pick the single best answer per feature; leave a ' +
    'feature blank if it does not apply. The keyword(s) in parentheses are the ' +
    'compile-time names, for reference.');

  // Collect email + a little context up front.
  form.setCollectEmail(true);
  form.setProgressBar(true);

  form.addTextItem()
      .setTitle('Name and group / institution (optional)');

  form.addMultipleChoiceItem()
      .setTitle('Your primary role with CHARMM')
      .setChoiceValues([
        'Developer (I write/maintain CHARMM source)',
        'Power user (I script and run advanced features)',
        'User (I run standard simulations)',
        'Other',
      ]);

  // One grid per section.
  SECTIONS.forEach(function (section) {
    form.addSectionHeaderItem()
        .setTitle(section.title)
        .setHelpText(section.help);

    form.addGridItem()
        .setTitle(section.title + ': your usage of each feature')
        .setRows(section.rows)
        .setColumns(SCALE);
  });

  // Closing free-text questions.
  form.addPageBreakItem().setTitle('A few open questions');
  CLOSING.forEach(function (q) {
    form.addParagraphTextItem().setTitle(q);
  });

  Logger.log('CHARMM Feature Usage Survey created.');
  Logger.log('Share this link with developers: ' + form.getPublishedUrl());
  Logger.log('Edit the form here:             ' + form.getEditUrl());
}
