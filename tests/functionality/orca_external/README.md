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

Leave this running. First request after `--warmup` should be fast (JAX already loaded).

## 2. ORCA client wrapper

ORCA `ProgExt` must be a single executable. Point it at a small wrapper:

```bash
cat > ~/bin/mmml-orca-client <<'EOF'
#!/bin/bash
exec "$HOME/mmml/.venv/bin/mmml" orca-client "$@"
EOF
chmod +x ~/bin/mmml-orca-client
```

Adjust `$HOME/mmml/.venv` if your venv lives elsewhere.

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
! Opt
%maxcore 2000

%method
  ProgExt "/home/boittier/bin/mmml-orca-client"
  Ext_Params "-b 127.0.0.1:8888"
end

* xyzfile 0 1 water.xyz
```

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

Geometry optimization (above), or single-point gradient:

```text
! EnGrad
%method
  ProgExt "/home/boittier/bin/mmml-orca-client"
  Ext_Params "-b 127.0.0.1:8888"
end
* xyzfile 0 1 water.xyz
```

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
pytest tests/unit/test_orca_external.py tests/unit/test_orca_external_server.py tests/unit/test_orca_external_cli.py -q
```
