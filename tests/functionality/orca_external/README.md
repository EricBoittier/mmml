# ORCA + MMML external tool (smoke workflow)

Manual verification for `mmml orca-server` / `mmml orca-client` with ORCA 6.
Do not run in agent sessions; use your GPU node with ORCA + MMML venv.

## 1. Start the MMML server (terminal 1)

```bash
cd ~/mmml
source .venv/bin/activate

export MMML_CHECKPOINT=~/mmml/mmml/models/physnetjax/defaults/hf_json/test-f41c04c0-62e3-4785-9018-351ffdc161c4_epoch-251_portable.json

mmml orca-server --checkpoint "$MMML_CHECKPOINT" --warmup -b 127.0.0.1:8888
```

For GOAT / multi-worker ORCA runs, enable GPU micro-batching (default: up to 16
requests, 10 ms collect window). ORCA temp files are read immediately on each
client request, before batching waits, so parallel GOAT workers are safe.

```bash
mmml orca-server --checkpoint "$MMML_CHECKPOINT" --warmup \
  -b 127.0.0.1:8888 --batch-size 16 --batch-wait-ms 10
```

Use `--batch-size 1` to disable batching (strictly serial requests).

Leave this running. First request after `--warmup` should be fast (JAX already loaded).

## 2. ORCA client wrapper

ORCA `ProgExt` must be a single executable path that exists on disk.

**Bundled wrapper** (after `git pull`): `examples/orca/mmml-orca-client` — resolves
`~/mmml/.venv/bin/mmml orca-client` from the repo root. Make it executable once:

```bash
chmod +x ~/mmml/examples/orca/mmml-orca-client
~/mmml/examples/orca/mmml-orca-client -h
```

Point `ProgExt` at the **absolute** path to that script (see `water_opt.inp`).

**Optional** `~/bin` symlink if you prefer a short path:

```bash
mkdir -p ~/bin
ln -sf ~/mmml/examples/orca/mmml-orca-client ~/bin/mmml-orca-client
```

**Alternative:** `ProgExt` can be the venv entry point directly:

```text
ProgExt "/home/boittier/mmml/.venv/bin/mmml-orca-client"
```

(after `uv sync` in `~/mmml`). Or set `export EXTOPTEXE=...` and omit `ProgExt`.

## 3. Example files (`examples/orca/water_opt/`)

**`water.xyz`** (Angstrom):

```text
3
water
O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200
```

**`water_opt.inp`**:

```text
! ExtOpt Opt
%maxcore 2000

%method
  ProgExt "/home/boittier/mmml/examples/orca/mmml-orca-client"
  Ext_Params "-b 127.0.0.1:8888"
end

* xyzfile 0 1 water.xyz
```

**Important:** ORCA only calls `ProgExt` when `! ExtOpt` is on the simple-input line
(`! ExtOpt Opt`, `! ExtOpt GOAT`, etc.). `%method ProgExt` alone is ignored and ORCA
will run its default electronic-structure method instead.

Replace `/home/boittier` with your username/path.

## 4. Run ORCA (terminal 2)

```bash
cd ~/mmml/examples/orca/water_opt   # or any directory containing water.xyz
orca water_opt.inp
```

ORCA will write `water_EXT.extinp.tmp`, `water_EXT.xyz`, call the client, read `water_EXT.engrad`, and optimize.

## 5. Smoke test without ORCA

Mimics one external-tool call:

```bash
cd ~/mmml/examples/orca/water_opt

cat > water_EXT.extinp.tmp <<'EOF'
water_EXT.xyz
0
1
1
1
EOF

cat > water_EXT.xyz <<'EOF'
3
water
O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200
EOF

mmml orca-client -b 127.0.0.1:8888 --checkpoint "$MMML_CHECKPOINT" water_EXT.extinp.tmp
ls -l water_EXT.engrad
head water_EXT.engrad
```

Pass: `water_EXT.engrad` exists, contains atom count `3`, energy in Eh, and 9 gradient values.

## Checkpoint notes

- The bundled `neutral_best_forces` JSON is a **PhysNet EF** model (`natoms: 34` in config). It is trained for small neutral organics, not water — energies are for **smoke testing the plumbing**, not publication-quality water PES.
- For production ORCA workflows, use a checkpoint trained on your chemistry (joint `.pkl` + `model_config.pkl`, or portable JSON/Orbax from your training run).
- Joint PhysNet+DCMNet pickles support `natoms` padding up to the trained limit.

## ORCA input variants

Geometry optimization (`examples/orca/water_opt/water_opt.inp`), or single-point gradient:

```text
! ExtOpt EnGrad
%method
  ProgExt "/home/boittier/mmml/examples/orca/mmml-orca-client"
  Ext_Params "-b 127.0.0.1:8888"
end
* xyzfile 0 1 water.xyz
```

### GOAT conformer search

See `examples/orca/goat_ethanol/` for a global optimization example using
[ORCA GOAT](https://www.faccts.de/docs/orca/6.0/manual/contents/typical/GOAT.html)
with MMML as the external PES. Start the server first, then:

```bash
cd ~/mmml/examples/orca/goat_ethanol
orca ethanol_goat.inp
```

GOAT performs many gradient calls — keep `mmml-orca-server` warm and use `!PAL` /
`%GOAT NWORKERS` to parallelize workers.

Standalone (no server; slow — reloads JAX each call):

```text
%method
  ProgExt "/home/boittier/mmml/.venv/bin/mmml"
  Ext_Params "orca-external --checkpoint /path/to/epoch.pkl water_EXT.extinp.tmp"
end
```

(`ProgExt` + `Ext_Params` forwarding is awkward for standalone; prefer server + client.)

## Unit tests (local)

```bash
pytest tests/unit/test_orca_external.py tests/unit/test_orca_external_server.py tests/unit/test_orca_external_cli.py tests/unit/test_orca_external_batching.py -q
```
