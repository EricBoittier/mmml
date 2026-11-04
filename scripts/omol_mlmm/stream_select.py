"""Stream the OMol25 'train' tarball once and map every LMDB member that is not mapped yet, without writing it to NFS.
Each member goes to node-local scratch, is mapped by a worker pool (map_chemspace.map_file), then deleted.
Members already being mapped from the NFS copy (listed in --skip-dir) or with an existing output are skipped.
    cat part0..part7 | python stream_map.py <out_dir> <scratch_dir> <skip_dir> <workers>"""
import os, sys, tarfile, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_select as M

def work(path, out):
    sel = out.replace("/map_full/", "/select_full/").replace(".npz", "_sel.npz"); n, ns, dt = M.map_select_file(path, out, sel)
    os.remove(path); lk = path + "-lock"
    if os.path.exists(lk): os.remove(lk)
    return os.path.basename(path), n, dt

if __name__ == "__main__":
    out, scratch, skipd, W = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    os.makedirs(out, exist_ok=True); os.makedirs(scratch, exist_ok=True)
    skip = set(os.listdir(skipd)) if os.path.isdir(skipd) else set()
    t0 = time.time(); pend = []; nbytes = 0
    with ProcessPoolExecutor(W) as pool, tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz") as tf:
        for m in tf:
            name = os.path.basename(m.name)
            if not m.isfile() or not name.endswith(".aselmdb"):
                continue
            o = os.path.join(out, name.replace(".aselmdb", ".npz"))
            if name in skip or os.path.exists(o):
                print(f"{time.time() - t0:7.0f}s skip {name}", flush=True); continue
            # back-pressure: keep at most W files waiting on local disk
            while sum(not f.done() for f in pend) >= W:
                time.sleep(2)
            dst = os.path.join(scratch, name)
            with tf.extractfile(m) as src, open(dst, "wb") as fh:
                while True:
                    b = src.read(64 << 20)
                    if not b: break
                    fh.write(b)
            nbytes += m.size; pend.append(pool.submit(work, dst, o))
            print(f"{time.time() - t0:7.0f}s extracted {name} {m.size / 1e9:.1f} GB ({nbytes / 1e9 / max(time.time() - t0, 1) * 1e3:.0f} MB/s avg)", flush=True)
            for f in [f for f in pend if f.done()]:
                nm, n, dt = f.result(); print(f"          mapped {nm}: {n} structures in {dt:.0f}s", flush=True); pend.remove(f)
        for f in pend:
            nm, n, dt = f.result(); print(f"          mapped {nm}: {n} structures in {dt:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
