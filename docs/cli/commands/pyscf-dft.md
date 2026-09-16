# `mmml pyscf-dft`

GPU DFT (energy, gradient, hessian, …).


## Usage

```bash
mmml pyscf-dft --help
```

## Options

```text
usage: mmml pyscf-dft [-h] --mol MOL [--output OUTPUT] [--log_file LOG_FILE]
                      [--monomer_a MONOMER_A] [--monomer_b MONOMER_B]
                      [--basis BASIS] [--xc XC] [--spin SPIN] [--charge CHARGE]
                      [--energy] [--optimize] [--gradient] [--hessian]
                      [--harmonic] [--thermo] [--interaction] [--dens_esp]
                      [--ir] [--shielding] [--polarizability] [--ir-efield]
                      [--efield-points EFIELD_POINTS]
                      [--efield-fd-axis EFIELD_FD_AXIS] [--efield-scf]
                      [--efield-scf-no-forces]
                      [--efield-dipole-unit EFIELD_DIPOLE_UNIT]
                      [--no-efield-include-nuclear-energy]
                      [--save_option SAVE_OPTION]

Scientific model:
  --basis BASIS
  --xc XC
  --spin SPIN
  --charge CHARGE
  --energy
  --ir-efield           IR + Hessian pipeline in a uniform E-field; scan fields
                        from --efield-points
  --efield-points EFIELD_POINTS
                        Semicolon-separated Ex,Ey,Ez in a.u., e.g.
                        '0,0,0;0,0,0.001;0,0,-0.001'
  --efield-fd-axis EFIELD_FD_AXIS
                        Cartesian axis (0=x,1=y,2=z) for finite-difference dμ/dE
                        from the scan
  --efield-scf          SCF only in uniform E-field: energy, dipole, forces (use
                        --efield-points); no IR/Hessian
  --efield-scf-no-forces
                        With --efield-scf, skip nuclear gradient (energy +
                        dipole only)
  --efield-dipole-unit EFIELD_DIPOLE_UNIT
                        Dipole unit for --efield-scf (e.g. DEBYE, AU)
  --no-efield-include-nuclear-energy
                        After SCF in a uniform field, omit nuclear-field energy
                        (use mf.kernel energy only).

Output & artifacts:
  --output OUTPUT
  --log_file LOG_FILE
  --save_option SAVE_OPTION

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --mol MOL
  --monomer_a MONOMER_A
  --monomer_b MONOMER_B
  --optimize
  --gradient
  --hessian
  --harmonic
  --thermo
  --interaction
  --dens_esp
  --ir
  --shielding
  --polarizability
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
