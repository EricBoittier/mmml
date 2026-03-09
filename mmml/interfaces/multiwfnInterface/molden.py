from pyscf.tools.molden import orbital_coeff, header, order_ao_index

# Write Molden file
with open('output.molden', 'w') as f:
    header(mol, f)
    orbital_coeff(mol, f, atoms.calc.mf.mo_coeff[0], ignore_h=False)