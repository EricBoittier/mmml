# `mmml pyscf-mp2`

GPU MP2.


## Usage

```bash
mmml pyscf-mp2 --help
```

## Options

```text
usage: mmml pyscf-mp2 [-h] --mol MOL [--output OUTPUT] [--basis BASIS]
                      [--spin SPIN] [--charge CHARGE] [--energy] [--gradient]
                      [--gradient-fd] [--fd-step ANG] [--log_file LOG_FILE]

GPU-accelerated MP2 calculations

options:
  -h, --help           show this help message and exit
  --mol MOL            Molecule (xyz string or file)
  --output OUTPUT      Output base path (.npz and .h5)
  --basis BASIS        Basis set
  --spin SPIN
  --charge CHARGE
  --energy             Compute MP2 energy
  --gradient           Compute MP2 gradient
  --gradient-fd        Use central finite differences for MP2 gradient (Issue
                       #13)
  --fd-step ANG        Finite-difference step in Angstrom (default: 1e-3)
  --log_file LOG_FILE
```


---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
