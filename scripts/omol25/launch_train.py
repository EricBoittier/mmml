#!/usr/bin/env python3
"""Turn configs/*.yaml into the exact train_so3lr_spooky_extxyz.py command; print it, or run it with --exec.

  python -I launch_train.py ../configs/pilot.yaml          # print command
  python -I launch_train.py ../configs/pilot.yaml --exec   # run it (on the GPU node / inside sbatch)
"""
import os
import shlex
import sys

import yaml


def build(cfg):
    argv = [cfg["python"], "-P", cfg["trainer"]]
    for k, v in cfg["trainer_args"].items():
        if v is True:
            argv.append(f"--{k}")
        elif v is False or v is None:
            continue
        elif isinstance(v, (list, tuple)):
            argv += [f"--{k}", *map(str, v)]
        else:
            argv += [f"--{k}", str(v)]
    env = {"PYTHONPATH": cfg["pythonpath"], **{k: str(v) for k, v in (cfg.get("env") or {}).items()}}
    return argv, env


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    with open(sys.argv[1]) as fh:
        cfg = yaml.safe_load(fh)
    argv, env = build(cfg)
    print(" ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()), " ".join(shlex.quote(a) for a in argv))
    if "--exec" in sys.argv[2:]:
        os.makedirs(cfg["trainer_args"]["workdir"], exist_ok=True)
        os.execve(argv[0], argv, {**os.environ, **env})


if __name__ == "__main__":
    main()
