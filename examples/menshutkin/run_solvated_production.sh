#!/usr/bin/env bash
# Solvated PMF, production settings. THE entry point for a new solvent.
#
#   GPU=1 SOLVENT=methanol EMB=mechanical    bash examples/menshutkin/run_solvated_production.sh
#   GPU=0 SOLVENT=methanol EMB=electrostatic bash examples/menshutkin/run_solvated_production.sh
#
# Env:
#   SOLVENT   water | methanol | acetonitrile | benzene | cyclohexane   (default water)
#   EMB       mechanical-fluct | mechanical | electrostatic   (default fluct)
#             mechanical-fluct : model charges q(R) recomputed every step, no
#                                dq/dR force term. Stable, and the forces ARE
#                                the exact gradient of a well-defined energy.
#             mechanical       : charges frozen at the reactant geometry. The
#                                control that isolates what the response is worth.
#             electrostatic    : q(R) WITH dq/dR forces. Fully self-consistent
#                                and physically right, but the charge-response
#                                feedback ran away (q(Cl) -0.80 -> -1.03 in
#                                50 fs). Needs --freeze-charge-forces, which is
#                                an approximation -- so it is NOT the default.
#   XI_MAX    last window. 1.6 = Turan's exact comparison range (default);
#             5.6 extends through the contact ion pair to the SSIP  (default 1.6)
#   FINE_TO   where the 0.1 A spacing stops and 0.25 A takes over. Defaults to
#             XI_MAX, i.e. fine everywhere. For the extended range set this to
#             1.6 so the reaction region keeps 0.1 A resolution and the long
#             dissociation tail is sampled coarsely -- XI_MAX=5.6 FINE_TO=1.6
#             gives the 46-window ladder the production runs use. Leaving it
#             unset with XI_MAX=5.6 would ask for 0.1 A spacing all the way out
#             and build 70+ windows.                            (default XI_MAX)
#   TAG       output directory suffix naming the physics variant: empty for the
#             baseline, `dqdr`, `pol`, `both`. Output lands in
#             artifacts/menshutkin/pmf_full_<solvent>[_<tag>]/<solvent>, which is
#             the layout every analysis script globs for.              (default "")
#   PROD_PS   production ps per window                                  (default 2.0)
#   GPU       CUDA device                                               (default 1)
#
# ---------------------------------------------------------------------------
# WHY THE SETTINGS ARE WHAT THEY ARE. Every one of these was paid for.
#
# --channel-k 50 --channel-tol 0.60 --channel-cn-tol 0.80
#   Tolerances are set from the LITERATURE, not from our own runs. Truong et al.
#   (J. Chem. Phys. 107, 1881) measure water elongating the C-N bond by 0.42 A
#   and shortening C-Cl by 0.30 A at the transition state. A tight bound on
#   r(C-N) would suppress exactly that -- the largest solvent effect on the
#   geometry and the thing this campaign measures. Our own earlier estimate of
#   the shift (+0.106 A) could not be used: it came from runs already clamping
#   at 0.10 A, so it was the suppressed value, and sizing the restraint from it
#   was circular.
#   The SUM tolerance can stay tighter (0.60) because +0.42 and -0.30 nearly
#   cancel: the solvated sum moves only ~0.12 A.
#
# --dt-fs 1.0
#   MEASURED by NVE drift from the equilibrated box: +0.018 meV/atom/ps at
#   1.0 fs against -0.006 at 0.25, both far inside the 0.5 acceptance. The
#   harmonic 20-steps-per-O-H-period rule would cap dt at 0.50; the integrator
#   tolerates more. Geometry at 1 fs matches 0.25 fs to 0.013 A in <xi>, well
#   under the 0.06-0.11 A within-window thermal spread.
#
# --prod-ps 10
#   1 ps gave overlap 0.000 between two windows and a 100 %-one-sided
#   split-half drift. Turan used 50 ps/window; 10 is the compromise that fits
#   the GPU budget.
#
# (superseded note) --channel-tol 0.20
#   The reaction-channel restraint, and the reason this campaign works at all.
#   xi = r(C-Cl) - r(C-N) fixes only the DIFFERENCE of two distances; the sum is
#   free, and on a fitted potential the dissociated branch is downhill. Without
#   any bound the methyl left both partners (r(C-Cl) 3.2-5.3 AND r(C-N) 3.1-3.7
#   at once) while reporting a perfect reaction coordinate, and the reported
#   "barrier" was the cost of tearing the methyl off.
#   A CONSTANT bound is not enough either: it is a box, and the path is a line
#   inside it. Two production runs died mid-flight (mechanical at xi = +0.70,
#   electrostatic at +0.60) while sitting inside the min(r) <= 2.25 A wall the
#   whole time, 0.9 A off the path in the sum, off the fitted manifold.
#   tol = 0.20 A because the MEASURED spread of the sum at fixed xi in the
#   training set is 0.09-0.14 A. At the 0.35 A this started with, the restraint
#   permitted systematic drift, and since xi pins only the difference that drift
#   migrated into the bond: r(C-Cl) came out 1.90-1.96 A at the reactant end
#   against a training 1.804 +- 0.022 and an experimental CH3Cl bond of 1.785.
#
# --bond-r-max 2.25
#   Emergency catch: at least one bond always formed. Training max is 2.18 A
#   inside Turan's range and only 1.57 A beyond it, so this is safe everywhere.
#   With the channel restraint on it should essentially never fire -- if it is
#   in contact more than a few percent, the channel is not doing its job.
#
# --sum-max 0 (off) but --angle-min-deg 130 (ON for the Turan range)
#   The SUM wall is range-specific and stays off: past the ion pair the training
#   sum grows to 8.67 A at xi = +5.5, so a constant bound would forbid the
#   separation being measured. The channel restraint handles that direction.
#
#   The ANGLE is different, and turning it off was a mistake. With fluctuating
#   charges the chloride becomes strongly negative (q -> -0.86) and the solvent
#   prefers to solvate it side-on: a run without this wall sampled N-C-Cl at
#   87-93 deg across the whole reactant side -- perpendicular approach, not an
#   Sn2 at all -- while the bond lengths looked perfectly healthy, so nothing
#   else flagged it. The channel restraint could not catch it either, because it
#   constrains distances and this is an angular escape.
#   Training sits at 169-175 deg over xi in [-1.3, +1.6], so 130 is a generous
#   floor that only rejects the pathological cases.
#
#   IT MUST STAY ON FOR THE SSIP EXTENSION TOO. This file previously said to set
#   ANGLE_MIN=0 past the ion pair, on the grounds that "training falls to ~85 deg
#   once Cl is far away". That figure was read off the WRONG FILE. The two
#   training sources disagree, and the disagreement is real:
#
#     examples/m/scan_nh3_ch3cl.npz     (relaxed scan = THE REACTION PATH)
#         xi +1.5..+2.0  mean 174.7 deg   xi +4.0..+4.5  mean 177.0, min 175.3
#     examples/m/nh3_ch3cl_filtered.npz (NMS + ORIENTATIONAL sampling)
#         xi +1.5..+2.0  mean  90.0 deg   xi +4.0..+4.5  mean  83.1, min   3.8
#
#   The second file deliberately moves the chloride around the methylammonium at
#   fixed separation. Those are legitimate TRAINING configurations; they are not
#   the path. On the path the angle is 175-177 deg at xi = +4, so a 130 deg wall
#   NEVER BINDS and forbids nothing.
#
#   Running the extension with ANGLE_MIN=0 (2026-08-01) produced exactly the
#   failure this wall exists to prevent: at xi = +1.60 the main run held
#   mean 141.9 deg while the extension sat at mean 65.6 deg (range 57-75) --
#   side-on, not an Sn2. Three of six windows then went non-finite, and the
#   three that survived were worse: clean <xi>, sd, minr and 0.000 restraint
#   contact, all for the wrong geometry.
#
#   It is also required for STITCHING. fig_pmf.py joins the extension to the
#   main run through their overlap at xi +1.2..+1.6. Different walls there means
#   different Hamiltonians, which is not an overlap at all -- the offset between
#   the two profiles would be undetermined.
#
# --lr-solver ewald
#   The reaction creates a +/-1 ion pair; a truncated Coulomb under-stabilises
#   it. Includes the reciprocal-space cross term between ML and MM charges.
#
# --equilibrate-box-ps 100
#   Cached per (solvent, box, seed) and reused by every later run, so the cost
#   is paid once. A freshly packed box is at the right density but has no liquid
#   structure: it drops ~850 eV in its first heat stage.
#
# --save-traj full
#   Needed for solvent structure (RDFs, coordination numbers). Solute-only
#   trajectories cannot answer those at all.
# ---------------------------------------------------------------------------
set -u
cd /mmhome/andreychev/mmml/mmml
export MENSH_CKPT="${MENSH_CKPT:-/mmhome/andreychev/mmml/mmml/ckpts/menshutkin_longrange/longrange-c2398efc-a6f3-478f-8a3b-af6c66cda0fc/epoch-1436}"
source examples/menshutkin/_env.sh
export MMML_EXAMPLE_DEVICE=gpu
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTHONUNBUFFERED=1

SOLVENT="${SOLVENT:-water}"
EMB="${EMB:-mechanical-fluct}"

# Output directory. The convention every analysis script globs for is
#
#     artifacts/menshutkin/pmf_full_<solvent>[_<tag>]/<solvent>
#
# so TAG names the physics variant: empty for the baseline, `dqdr`, `pol`,
# `both`. This used to be built as pmf_${EMB}_${SOLVENT}, which produced
# `pmf_electrostatic_water` -- a name nothing reads -- so every launch appended a
# second --output-dir on the command line to correct it. argparse takes the last
# occurrence, so it worked, but the effective configuration was then invisible
# from either this file or the invocation, and `--fine-to` was being silently
# overridden the same way (5.6 in the script, 1.6 on the command line).
# Set TAG instead of overriding --output-dir.
TAG="${TAG:-}"
OUT="/mmhome/andreychev/mmml/mmml/artifacts/menshutkin/pmf_full_${SOLVENT}${TAG:+_${TAG}}"

EXTRA=()
# FREEZE_CHARGE_FORCES is now OPT-IN, and defaults OFF.
#
# This used to be added automatically whenever EMB=electrostatic. That set
# charge_gradient=False, which is exactly what mechanical-fluct does -- so every
# "electrostatic" run ever launched through this script had mechanical-fluct
# forces and never tested dq/dR at all.
#
# Dropping dq/dR is not free: measured against central finite differences of the
# same energy, it changes the force by 101 % and is 21-139 % of the retained
# force across xi = -1.3..+2.25. The forces are then not the gradient of the
# energy. The historical justification -- that the charge response ran away --
# came from a probe seeded at the wrong xi; re-tested from a properly
# equilibrated frame, undamped dq/dR survives (diag/probe_dqdr.py).
#
# If it does turn out to need taming, damp rather than delete: pass
# --charge-gradient-scale LAM for a fractional response.
[[ "${FREEZE_CHARGE_FORCES:-0}" == "1" ]] && EXTRA+=(--freeze-charge-forces)

exec /mmhome/andreychev/mmml/mmml/.venv/bin/python -u \
  examples/menshutkin/07_solvated_pmf.py \
  --solvent "${SOLVENT}" --embedding "${EMB}" \
  --xi-min -1.3 --xi-max "${XI_MAX:-1.6}" --fine 0.1 --fine-to "${FINE_TO:-${XI_MAX:-1.6}}" --coarse 0.25 \
  --lr-solver ewald \
  --bond-r-max 2.25 --sum-min 0.0 --sum-max 0 \
  --angle-min-deg "${ANGLE_MIN:-130}" \
  --channel-k 50.0 --channel-tol 0.60 --channel-cn-tol 0.80 \
  --equilibrate-box-ps 100 --heat-stages 5 --ramp-stages 10 \
  --dt-fs "${DT:-1.0}" \
  --equil-ps 1.0 --prod-ps "${PROD_PS:-10.0}" --record-every 20 \
  --save-traj full --traj-stride 25 \
  --output-dir "${OUT}/${SOLVENT}" "${EXTRA[@]}" "$@"
