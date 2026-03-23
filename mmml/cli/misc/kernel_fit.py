"""CLI: fit kernel (distance matrix -> charge positions) and write CHARMM kernel files + H5."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Fit kernel ridge: distance matrix -> (AQ,BQ,CQ). Write CHARMM kernel files and optional H5."
    )
    parser.add_argument("h5", type=Path, help="charmm_ml_comparison.h5 or similar")
    parser.add_argument("--out-dir", "-o", type=Path, default=Path("kernel_out"),
                        help="Directory for x_fit.txt, coefs*.txt")
    parser.add_argument("--natmk", type=int, default=None,
                        help="Number of atoms for distance matrix (default: from H5)")
    parser.add_argument("--out-h5", type=Path, default=None,
                        help="If set, evaluate and write H5 for GUI")
    parser.add_argument("--out-mdcm", type=Path, default=None,
                        help="Write .mdcm file (default: out_dir/RESIDUE.mdcm)")
    parser.add_argument("--out-kmdcm", type=Path, default=None,
                        help="Write .kmdcm kernel file (default: out_dir/RESIDUE.kmdcm)")
    parser.add_argument("--residue-name", default="MEOH",
                        help="Residue name for mdcm header and default filenames")
    parser.add_argument("--nkfr", type=int, default=None,
                        help="NKFR for kmdcm (default: number of frames)")
    parser.add_argument("--optimize", action="store_true",
                        help="Optimize (AQ,BQ,CQ) per frame before fitting")
    parser.add_argument("--train-frames", type=str, default=None,
                        help="Comma-separated frame indices for training (default: all)")
    parser.add_argument("--lam", type=float, default=1e-6, help="Kernel ridge regularization")
    parser.add_argument("--sigma", type=float, default=1.0, help="RBF kernel width")
    parser.add_argument("--base-name", default="x_fit", help="Prefix for x_fit/coefs filenames")
    args = parser.parse_args()

    train_indices = None
    if args.train_frames:
        train_indices = [int(x) for x in args.train_frames.split(",")]

    from mmml.interfaces.dcmInterface.kernel_pipeline import run_kernel_fit_pipeline

    result = run_kernel_fit_pipeline(
        h5_path=args.h5,
        out_dir=args.out_dir,
        natmk=args.natmk if args.natmk is not None else None,
        out_h5=args.out_h5,
        out_mdcm=args.out_mdcm,
        out_kmdcm=args.out_kmdcm,
        optimize_positions=args.optimize,
        train_frame_indices=train_indices,
        lam=args.lam,
        sigma=args.sigma,
        base_name=args.base_name,
        residue_name=args.residue_name,
        nkfr=args.nkfr,
    )

    # Print fit errors (AQ, BQ, CQ)
    fm = result["fit_metrics"]
    print("\n" + "=" * 60)
    print("FIT ERRORS (kernel: distance matrix -> AQ,BQ,CQ)")
    print("=" * 60)
    print(f"  RMSE (Å):     {fm['rmse']:.6f}")
    print(f"  MAE (Å):      {fm['mae']:.6f}")
    print(f"  R²:           {fm['r2']:.4f}")
    if fm.get("rmse_per_output") is not None:
        rpo = fm["rmse_per_output"]
        if len(rpo) <= 12:
            print(f"  RMSE/output:  {rpo}")
        else:
            print(f"  RMSE/output:  [first 6] {rpo[:6].round(6)} ... (n={len(rpo)})")

    if "esp_metrics" in result:
        em = result["esp_metrics"]
        print("\n" + "=" * 60)
        print("ESP ERRORS (kernel vs reference)")
        print("=" * 60)
        print(f"  RMSE (Hartree/e):  {em['rmse']:.6f}")
        print(f"  MAE (Hartree/e):   {em['mae']:.6f}")
        print(f"  R²:                {em['r2']:.4f}")
    print()

    print(f"Wrote kernel files to {args.out_dir}:")
    for p in result["paths"]:
        print(f"  {p}")
    if "out_mdcm_path" in result:
        print(f"Wrote .mdcm: {result['out_mdcm_path']}")
    if "out_kmdcm_path" in result:
        print(f"Wrote .kmdcm: {result['out_kmdcm_path']}")
    if "out_h5_path" in result:
        print(f"Wrote H5 for GUI: {result['out_h5_path']}")
        print("  (esp_rmse_kernel, esp_mae_kernel, esp_r2_kernel in H5)")
    return 0


if __name__ == "__main__":
    exit(main())
