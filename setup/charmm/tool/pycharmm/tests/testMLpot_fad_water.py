# Test Script to import PhysNet as energy function in CHARMM via PyCHARMM

# Basics
import os
import numpy as np

# PyCHARMM
from pycharmm import *

# Asparagus
from asparagus import Asparagus


# Step 0: Load parameter files
#-----------------------------------------------------------

toppar_dir = 'toppar'
data_dir = 'data'

chm_settings.set_bomb_level(-1)
chm_read.rtf('toppar/top_all36_prot.rtf')
chm_read.rtf('toppar/top_all36_cgenff.rtf', append = True)
chm_read.rtf('toppar/probes.rtf', append = True)
chm_read.prm('toppar/par_all36m_prot.prm', flex = True)
chm_read.prm('toppar/par_all36_cgenff.prm', append = True, flex = True)
chm_read.prm('toppar/probes.prm', append = True, flex = True)

# Water
charmm_script(
    "stream {:s}/toppar_water_ions.str".format(toppar_dir))

chm_settings.set_bomb_level(0)
chm_settings.set_warn_level(-1)

# Step 1: Generate zero-bond Formic Acid Dimer in CHARMM with water
#-------------------------------------------------------------------

Nfad = 10
Ntip3 = 729

# Generate new segment of FAD and water molecules
falab = 'FORH'
chm_read.sequence_string('{0:s} {0:s}'.format(falab))
gen.new_segment(
    seg_name='FAD',
    setup_ic=True)

if Ntip3:
    chm_read.sequence_string('TIP3 '*Ntip3)
    gen.new_segment(
        seg_name='TIP3',
        setup_ic=True)

# ... or read from the later generated psf file
# chm_read.psf_card("fad_water.psf")

# Read coordinates
chm_read.coor_card(os.path.join(data_dir, "fad_water.cor"))

# Step 2: Set CHARMM Properties
#-----------------------------------------------------------

# Non-bonding parameter
dict_nbonds = {
    'atom': True,
    'vdw': True,
    'vswitch': True,
    'cutnb': 14,
    'ctofnb': 12,
    'ctonnb': 10,
    'cutim': 14,
    'lrc': True,
    'inbfrq': -1,
    'imgfrq': -1
    }

nbond = NonBondedScript(**dict_nbonds)
nbond.run()

# Energy
energy.show()

# PBC box
stats = coor.stat()
size = ( stats['xmax'] - stats['xmin']\
       + stats['ymax'] - stats['ymin']\
       + stats['zmax'] - stats['zmin'] ) / 3
# Size with density of 0.99 (if all water residues are included)
#size = 28.036

offset = size/2.
xyz = coor.get_positions()
xyz += size/2.

crystal.define_cubic(length=size)
crystal.build(cutoff=dict_nbonds['cutim'])

image.setup_segment(offset,offset,offset,'FAD')
image.setup_residue(offset,offset,offset,'TIP3')

# H-bonds constraint
#shake.on(bonh=True, tol=1e-7)
charmm_script('shake bonh para sele resname TIP3 end')

# Write pdb and psf files
chm_write.coor_pdb(
    os.path.join(data_dir, "fad_water.pdb"),
    title="Formic Acid Dimer with Water")
chm_write.psf_card(
    os.path.join(data_dir, "fad_water.psf"),
    title="Formic Acid Dimer with Water")

# Energy
energy.show()

# Step 2: Define PhysNet energy function
#-----------------------------------------------------------

# Load Asparagus model
ml_model = Asparagus(config='data/fad_model/fad_physnet.json')

# Get atomic number from ASE atoms object
ml_Z = [6, 1, 8, 8, 1, 6, 1, 8, 8, 1]

# Prepare PhysNet input parameter
ml_selection = SelectAtoms(seg_id='FAD')

# Initialize PhysNet calculator
calc = MLpot(
    ml_model,
    ml_Z,
    ml_selection,
    ml_charge=0,
    ml_fq=True,
)

# Custom energy
energy.show()

# Step 3: Minimization
#-----------------------------------------------------------

chm_settings.set_bomb_level(-2)

# Fix FAD
cons_fix.setup(SelectAtoms(seg_id='FAD'))

# Optimization with PhysNet parameter
minimize.run_sd(**{
    'nstep': 100,
    'tolenr': 1e-5,
    'tolgrd': 1e-5})

# Unfix FAD
cons_fix.turn_off()

# Optimization with PhysNet parameter
minimize.run_abnr(**{
    'nstep': 100,
    'tolenr': 1e-5,
    'tolgrd': 1e-5})

chm_settings.set_bomb_level(0)

# Step 4: Heating - CHARMM, PhysNet
#-----------------------------------------------------------

if True:

    timestep = 0.0005   # 0.5 fs

    res_file = CharmmFile(
        file_name='heat.res', file_unit=2, formatted=True, read_only=False)
    dcd_file = CharmmFile(
        file_name='heat.dcd', file_unit=1, formatted=False, read_only=False)

    # Run some dynamics
    dynamics_dict = {
        'leap': False,
        'verlet': True,
        'cpt': False,
        'new': False,
        'langevin': False,
        'timestep': timestep,
        'start': True,
        'nstep': 1.*1./timestep,
        'nsavc': 0.01*1./timestep,
        'nsavv': 0,
        'inbfrq':-1,
        'ihbfrq': 50,
        'ilbfrq': 50,
        'imgfrq': 50,
        'ixtfrq': 1000,
        'iunrea':-1,
        'iunwri': res_file.file_unit,
        'iuncrd': dcd_file.file_unit,
        'nsavl':  0,  # frequency for saving lambda values in lamda-dynamics
        'iunldm':-1,
        'ilap': -1,
        'ilaf': -1,
        'nprint': 100, # Frequency to write to output
        'iprfrq': 500, # Frequency to calculate averages
        'isvfrq': 1000, # Frequency to save restart file
        'ntrfrq': 1000,
        'ihtfrq': 200,
        'ieqfrq': 1000,
        'firstt': 100,
        'finalt': 350,
        'tbath': 350,
        'iasors': 0,
        'iasvel': 1,
        'ichecw': 0,
        'iscale': 0,  # scale velocities on a restart
        'scale': 1,  # scaling factor for velocity scaling
        'echeck':-1}

    dyn_heat = DynamicsScript(**dynamics_dict)
    dyn_heat.run()

    res_file.close()
    dcd_file.close()

# Step 5: NVE - CHARMM, PhysNet
#-----------------------------------------------------------

if True:

    timestep = 0.0002   # 0.2 fs

    str_file = CharmmFile(
        file_name='heat.res', file_unit=3, formatted=True, read_only=False)
    res_file = CharmmFile(
        file_name='nve.res', file_unit=2, formatted=True, read_only=False)
    dcd_file = CharmmFile(
        file_name='nve.dcd', file_unit=1, formatted=False, read_only=False)

    # Run some dynamics
    dynamics_dict = {
        'leap': False,
        'verlet': True,
        'cpt': False,
        'new': False,
        'langevin': False,
        'timestep': timestep,
        'start': False,
        'restart': True,
        'nstep': 1*1./timestep,
        'nsavc': 0.01*1./timestep,
        'nsavv': 0,
        'inbfrq':-1,
        'ihbfrq': 50,
        'ilbfrq': 50,
        'imgfrq': 50,
        'ixtfrq': 1000,
        'iunrea': str_file.file_unit,
        'iunwri': res_file.file_unit,
        'iuncrd': dcd_file.file_unit,
        'nsavl':  0,  # frequency for saving lambda values in lamda-dynamics
        'iunldm':-1,
        'ilap': -1,
        'ilaf': -1,
        'nprint': 100, # Frequency to write to output
        'iprfrq': 500, # Frequency to calculate averages
        'isvfrq': 1000, # Frequency to save restart file
        'ntrfrq': 0,
        'ihtfrq': 0,
        'ieqfrq': 0,
        'firstt': 350,
        'finalt': 350,
        'tbath': 350,
        'iasors': 0,
        'iasvel': 1,
        'ichecw': 0,
        'iscale': 0,  # scale velocities on a restart
        'scale': 1,  # scaling factor for velocity scaling
        'echeck':-1}

    dyn_nve = DynamicsScript(**dynamics_dict)
    dyn_nve.run()

    str_file.close()
    res_file.close()
    dcd_file.close()

# Step 6: Equilibration - CHARMM, PhysNet
#-----------------------------------------------------------

if True:

    timestep = 0.0002   # 0.2 fs

    pmass = int(np.sum(select.get_property('mass'))/50.0)
    tmass = int(pmass*10)

    str_file = CharmmFile(
        file_name='heat.res', file_unit=3, formatted=True, read_only=False)
    res_file = CharmmFile(
        file_name='equi.res', file_unit=2, formatted=True, read_only=False)
    dcd_file = CharmmFile(
        file_name='equi.dcd', file_unit=1, formatted=False, read_only=False)

    # Run some dynamics
    dynamics_dict = {
        'leap': True,
        'verlet': False,
        'cpt': True,
        'new': False,
        'langevin': False,
        'timestep': timestep,
        'start': False,
        'restart': True,
        'nstep': 1*1./timestep,
        'nsavc': 0.01*1./timestep,
        'nsavv': 0,
        'inbfrq':-1,
        'ihbfrq': 50,
        'ilbfrq': 50,
        'imgfrq': 50,
        'ixtfrq': 1000,
        'iunrea': str_file.file_unit,
        'iunwri': res_file.file_unit,
        'iuncrd': dcd_file.file_unit,
        'nsavl':  0,  # frequency for saving lambda values in lamda-dynamics
        'iunldm':-1,
        'ilap': -1,
        'ilaf': -1,
        'nprint': 100, # Frequency to write to output
        'iprfrq': 500, # Frequency to calculate averages
        'isvfrq': 1000, # Frequency to save restart file
        'ntrfrq': 1000,
        'ihtfrq': 200,
        'ieqfrq': 0,
        'firstt': 350,
        'finalt': 350,
        'tbath': 350,
        'pint pconst pref': 1,
        'pgamma': 5,
        'pmass': pmass,
        'hoover reft': 350,
        'tmass': tmass,
        'iasors': 0,
        'iasvel': 1,
        'ichecw': 0,
        'iscale': 0,  # scale velocities on a restart
        'scale': 1,  # scaling factor for velocity scaling
        'echeck':-1}

    dyn_equi = DynamicsScript(**dynamics_dict)
    dyn_equi.run()

    str_file.close()
    res_file.close()
    dcd_file.close()


# Step 7: Production - CHARMM, PhysNet
#-----------------------------------------------------------

if True:

    timestep = 0.0002   # 0.2 fs

    pmass = int(np.sum(select.get_property('mass'))/50.0)
    tmass = int(pmass*10)

    for ii in range(0, 2):

        if ii==0:

            str_file = CharmmFile(
                file_name='equi.res',
                file_unit=3, formatted=True, read_only=False)
            res_file = CharmmFile(
                file_name='dyna.{:d}.res'.format(ii),
                file_unit=2, formatted=True, read_only=False)
            dcd_file = CharmmFile(
                file_name='dyna.{:d}.dcd'.format(ii),
                file_unit=1, formatted=False, read_only=False)

        else:

            str_file = CharmmFile(
                file_name='dyna.{:d}.res'.format(ii - 1),
                file_unit=3, formatted=True, read_only=False)
            res_file = CharmmFile(
                file_name='dyna.{:d}.res'.format(ii),
                file_unit=2, formatted=True, read_only=False)
            dcd_file = CharmmFile(
                file_name='dyna.{:d}.dcd'.format(ii),
                file_unit=1, formatted=False, read_only=False)

        # Run some dynamics
        dynamics_dict = {
            'leap': True,
            'verlet': False,
            'cpt': True,
            'new': False,
            'langevin': False,
            'timestep': timestep,
            'start': False,
            'restart': True,
            'nstep': 1*1./timestep,
            'nsavc': 0.01*1./timestep,
            'nsavv': 0,
            'inbfrq':-1,
            'ihbfrq': 50,
            'ilbfrq': 50,
            'imgfrq': 50,
            'ixtfrq': 1000,
            'iunrea': str_file.file_unit,
            'iunwri': res_file.file_unit,
            'iuncrd': dcd_file.file_unit,
            'nsavl':  0,  # frequency for saving lambda values in lamda-dynamics
            'iunldm':-1,
            'ilap': -1,
            'ilaf': -1,
            'nprint': 100, # Frequency to write to output
            'iprfrq': 500, # Frequency to calculate averages
            'isvfrq': 1000, # Frequency to save restart file
            'ntrfrq': 1000,
            'ihtfrq': 0,
            'ieqfrq': 0,
            'firstt': 350,
            'finalt': 350,
            'tbath': 350,
            'pint pconst pref': 1,
            'pgamma': 5,
            'pmass': pmass,
            'hoover reft': 350,
            'tmass': tmass,
            'iasors': 0,
            'iasvel': 1,
            'ichecw': 0,
            'iscale': 0,  # scale velocities on a restart
            'scale': 1,  # scaling factor for velocity scaling
            'echeck':-1}

        dyn_prod = DynamicsScript(**dynamics_dict)
        dyn_prod.run()

        str_file.close()
        res_file.close()
        dcd_file.close()
