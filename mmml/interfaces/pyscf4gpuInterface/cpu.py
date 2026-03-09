
import numpy
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha, Bohr, Debye
# from pyscf.prop.polarizability.uhf import polarizability, Polarizability
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
import jsonpickle

class parameters():
    # holds the calculation mode and user-chosen attributes of post-HF objects
    def __init__(self):
        self.mode = 'hf'
    def show(self):
        print('------------------------')
        print('calculation-specific parameters set by the user')
        print('------------------------')
        for v in vars(self):
            print('{}:  {}'.format(v,vars(self)[v]))
        print('\n\n')
                                                         


def todict(x):
    return jsonpickle.encode(x, unpicklable=False)


def init_geo(mf, atoms):
    # convert ASE structural information to PySCF information
    if atoms.pbc.any():
        cell = mf.cell.copy()
        cell.atom = atoms_from_ase(atoms)
        cell.a = atoms.cell.copy()
        cell.build()
        mf.reset(cell=cell.copy())
    else:
        mol = mf.mol.copy()
        mol.atom = atoms_from_ase(atoms)
        mol.build()
        mf.reset(mol=mol.copy())

class PYSCF(Calculator):
    # PySCF ASE calculator
    # by Jakob Kraus
    # units:  ase         -> units [eV,Angstroem,eV/Angstroem,e*A,A**3]
    #         pyscf       -> units [Ha,Bohr,Ha/Bohr,Debye,Bohr**3]

    implemented_properties = ['energy','forces','dipole',
                              # 'polarizability'
                             ]
    
    
    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='PySCF', atoms=None, directory='.', **kwargs):
        # constructor
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, directory, **kwargs)
        self.initialize(**kwargs)


    def initialize(self, mf=None, p=None):
        # attach the mf object to the calculator
        # add the todict functionality to enable ASE trajectories:
        # https://github.com/pyscf/pyscf/issues/624
        self.mf = mf
        self.p = p
        self.mf.todict = lambda: todict(self.mf)
        self.p.todict = lambda: todict(self.p)

    def set(self, **kwargs):
        # allow for a calculator reset
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()
    
    def get_polarizability(self,atoms=None):
        return self.get_property('polarizability',atoms)
    
    def calculate(self,atoms=None,properties=['energy'],system_changes=all_changes):
        
        Calculator.calculate(self,atoms=atoms,properties=properties,system_changes=system_changes)
        
        # update your mf object with new structural information
        init_geo(self.mf, atoms)

        # further update your mf object for post-HF methods
        if hasattr(self.mf,'_scf'):
            self.mf._scf.kernel()
            self.mf.__init__(self.mf._scf)
            for v in vars(self.p):
                if v != 'mode':
                    setattr(self.mf,v,vars(self.p)[v])
        self.mf.kernel()
        e = self.mf.e_tot
        
        if self.p.mode.lower() == 'ccsd(t)':
            e += self.mf.ccsd_t()
        
        self.results['energy'] = e * Ha
        
        if 'forces' in properties:
            gf = self.mf.nuc_grad_method()
            gf.verbose = self.mf.verbose
            if self.p.mode.lower() == 'dft':
                gf.grid_response = True
            forces = -1. * gf.kernel() * (Ha / Bohr)
            totalforces = []
            totalforces.extend(forces)
            totalforces = numpy.array(totalforces)
            self.results['forces'] = totalforces
        
        if hasattr(self.mf,'_scf'):
            self.results['dipole'] = self.mf._scf.dip_moment(verbose=self.mf._scf.verbose) * Debye
            # self.results['polarizability'] = Polarizability(self.mf._scf).polarizability() * (Bohr**3)
        else:
            self.results['dipole'] = self.mf.dip_moment(verbose=self.mf.verbose) * Debye
            # self.results['polarizability'] = Polarizability(self.mf).polarizability() * (Bohr**3)


# In[35]:




# In[37]:


def set_up_pyscf_cpu_calc(atoms, 
                          sym=False, 
                          index = 0,
                         basis='cc-pvdz',
                         spin=0,
                         charge=0):
    atoms = res_atoms.copy()
    mol = pyscf.M(atom=atoms_from_ase(atoms),
                  basis=basis,
                  spin=spin,charge=charge
                 )
    mf = mol.UHF()
    mf.verbose = 3
    # mf.kernel() 
    
    mf = [mf,mf.MP2(),mf.CISD(),mf.CCSD(),mf.CCSD()][index]
    p = parameters()
    p.mode = ['hf','mp2','cisd','ccsd','ccsd(t)'][index]
    p.verbose = 5
    p.show()
    if sym:
        mf.mol.symmetry = True
        mf.mol.symmetry = sym
    return PYSCF(mf=mf,p=p)
