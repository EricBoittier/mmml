import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_select as M
U = "/mmhome/boittier/home/metatomic-runs/training/omol_universal/mlmm_ft"
files = sorted(glob.glob(f"{U}/map_full_ready/*.aselmdb"))[int(sys.argv[1])::int(sys.argv[2])]
for f in files:
    b = os.path.basename(f).replace(".aselmdb", "")
    sel = f"{U}/select_full/{b}_sel.npz"
    if os.path.exists(sel): continue
    n, ns, dt = M.map_select_file(os.path.realpath(f), f"/tmp/{b}_map_dup.npz", sel)
    os.remove(f"/tmp/{b}_map_dup.npz"); print(f"{b}: {n} structures, {ns} selected, {dt:.0f}s", flush=True)
