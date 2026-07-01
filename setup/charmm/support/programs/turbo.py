#!/usr/bin/env python

import os
import sys
import shutil
import time

# the variable jobtype is used by the script to determine which TURBOMOLE executable to use
# ridft indicates that the Resolution of Identity DFT code should be used
# ricc2 indicates that the Resolution of Identity CC2 code should be used
# other methods are not currently implemented in this script
# note that this only determines the TURBOMOLE executable. The appropriate data files 
# for each job type must be created before CHARMM/TURBOMOLE is executed

# for parallel jobs, if mpi is true the MPI executablse are used
# otherwise, the SMP executables are used

mpi=True

# the nprocs variable determines how many processors to execute TURBOMOLE with.
# On some PBS queuing systems, this can be determined from an environment variable,
# otherwise it should be set by the user here
# note that using the maximum number of processors on a note is not always optimal

#nprocs=int(os.environ['PBS_NNODES'])
nprocs=12

# set the environment variable OMP_NUM_THREADS to nprocs for SMP jobs
#os.environ['OMP_NUM_THREADS']=str(nprocs)

# the paths of the turbomole executables should be specified here

mpiexec='/home/rowley_group/programs/TURBOMOLE/mpirun_scripts/em64t-unknown-linux-gnu_mpi/PMPI/bin/mpiexec'
turbomole='/home/rowley_group/programs/TURBOMOLE/bin/x86_64-unknown-linux-gnu'

ridft_serial=turbomole+'/ridft'
rdgrad_serial=turbomole+'/rdgrad'
dscf_serial=turbomole+'/dscf'
grad_serial=turbomole+'/grad'
ricc2_serial=turbomole+'/ricc2'
escf_serial=turbomole+'/escf'
egrad_serial=turbomole+'/egrad'

ridft_smp=turbomole+'_smp/ridft_smp'
rdgrad_smp=turbomole+'_smp/rdgrad_smp'
dscf_smp=turbomole+'_smp/dscf_smp'
grad_smp=turbomole+'_smp/grad_smp'
ricc2_smp=turbomole+'_smp/ricc2_omp'
escf_smp=turbomole+'_smp/escf_smp'
egrad_smp=turbomole+'_smp/egrad_smp'

ridft_mpi=turbomole+'_mpi/ridft_mpi'
rdgrad_mpi=turbomole+'_mpi/rdgrad_mpi'
dscf_mpi=turbomole+'_mpi/dscf_mpi'
grad_mpi=turbomole+'_mpi/grad_mpi'
ricc2_mpi=turbomole+'_mpi/ricc2_mpi'
escf_mpi=turbomole+'_mpi/escf'
egrad_mpi=turbomole+'_mpi/egrad'


# tmpdir should point to a local directory where TURBOMOLE can be executed
# temporary and output files will be written here
# on some systems, this location is defined in the environment variable TMPDIR
# otherwise, it should be set by the user here

#tmpdir=os.environ['TMPDIR']
tmpdir='/tmp/' 

# two directories must be provided by CHARMM as command line arguments in execution of this script
# the datadir variable corresponds to the path containing the TURBOMOLE input files:
# control, basis, auxbasis, mos or alpha, beta (optional: mos.bin, alpha.bin, beta.bin)
# the turbopwd variable correponds to the path where CHARMM with write the coordinates of the QM system (coord file)
# and the coordinates/magnitudes of the MM point charges (if present) in the file point_charges
# this script will copy the output of the TURBOMOLE calculation (gradient, pc_gradient) to this directory 
# so that it can be read by CHARMM

datadir=sys.argv[1]
turbopwd=sys.argv[2]

if(datadir[-1]!='/'):
    datadir=datadir + '/'

if(turbopwd[-1]!='/'):
    turbopwd=turbopwd + '/'

if(datadir[0]!='/'):
    datadir=os.getcwd() + '/' + datadir

if(datadir[-1]!='/'):
    datadir=datadir + '/'

if(turbopwd[0]!='/'):
    turbopwd=os.getcwd() + '/' + turbopwd
            
# remove existing output files from the turbopwd directory

if(os.path.exists(turbopwd + '/gradient')):
    os.unlink(turbopwd + '/gradient')

if(os.path.exists(turbopwd + '/pc_gradient')):
    os.unlink(turbopwd + '/pc_gradient')

if(os.path.exists(turbopwd + '/energy')):
    os.unlink(turbopwd + '/energy')

if(os.path.exists(turbopwd + '/sing_a')):
    os.unlink(turbopwd + '/sing_a')

if(os.path.exists(tmpdir + '/sing_a')):
    os.unlink(tmpdir + '/sing_a')

# if this is an FEP job, a 3rd command line argument is provided
# that indicates which state is being calculated (0 or 1)

if(len(sys.argv)==4):
    pert=True
    state=int(sys.argv[3])
else:
    pert=False

# see if scratch directories exist
# if scratch directories do not exist, copy them into place

if(pert):
    # if this is an FEP job, the temporary and data directories are updated to point to subdirectories
    # state_0 or state_1
    tmpdir=tmpdir + '/state_' + str(state)
    datadir=datadir + '/state_' + str(state)

if(os.path.exists(tmpdir)==False):
    # the temporary directory does not exist
    # create it and copy the swap files to it
    os.mkdir(tmpdir)
    if(os.path.exists(datadir + '/mos')):
        shutil.copyfile(datadir + '/mos', tmpdir + '/mos')
        
    if(os.path.exists(datadir + '/mos.bin')):
        shutil.copyfile(datadir + '/mos.bin', tmpdir + '/mos.bin')

    if(os.path.exists(datadir + '/alpha.bin')):
        shutil.copyfile(datadir + '/alpha.bin', tmpdir + '/alpha.bin')

    if(os.path.exists(datadir + '/beta.bin')):
        shutil.copyfile(datadir + '/beta.bin', tmpdir + '/beta.bin')

    if(os.path.exists(datadir + '/alpha')):
        shutil.copyfile(datadir + '/alpha', tmpdir + '/alpha')

    if(os.path.exists(datadir + '/beta')):
        shutil.copyfile(datadir + '/beta', tmpdir + '/beta')

    if(os.path.exists(datadir + '/auxbasis')):
        shutil.copyfile(datadir + '/auxbasis', tmpdir + '/auxbasis')

    if(os.path.exists(datadir + '/basis')):
        shutil.copyfile(datadir + '/basis', tmpdir + '/basis')

    if(os.path.exists(datadir + '/control')):
        shutil.copyfile(datadir + '/control', tmpdir + '/control')
else:
    # the temporary directory exists
    # copy the input files to it as needed
    if(os.path.exists(tmpdir + '/mos')==False and os.path.exists(datadir + '/mos')==True):
        shutil.copyfile(datadir + '/mos', tmpdir + '/mos')

    if(os.path.exists(tmpdir + '/mos.bin')==False and os.path.exists(datadir + '/mos.bin')==True):
        shutil.copyfile(datadir + '/mos.bin', tmpdir + '/mos.bin')

    if(os.path.exists(tmpdir + '/alpha.bin')==False and os.path.exists(datadir + '/alpha.bin')==True):
        shutil.copyfile(datadir + '/alpha.bin', tmpdir + '/alpha.bin')

    if(os.path.exists(tmpdir + '/beta.bin')==False and os.path.exists(datadir + '/beta.bin')==True):
        shutil.copyfile(datadir + '/beta.bin', tmpdir + '/beta.bin')

    if(os.path.exists(tmpdir + '/alpha')==False and os.path.exists(datadir + '/alpha')==True):
        shutil.copyfile(datadir + '/alpha', tmpdir + '/alpha')

    if(os.path.exists(tmpdir + '/beta')==False and os.path.exists(datadir + '/beta')==True):
        shutil.copyfile(datadir + '/beta', tmpdir + '/beta')

    if(os.path.exists(tmpdir + '/basis')==False and os.path.exists(datadir + '/basis')==True):
        shutil.copyfile(datadir + '/basis', tmpdir + '/basis')

    if(os.path.exists(tmpdir + '/auxbasis')==False and os.path.exists(datadir + '/auxbasis')==True):
        shutil.copyfile(datadir + '/auxbasis', tmpdir + '/auxbasis')

    if(os.path.exists(tmpdir + '/control')==False and os.path.exists(datadir + '/control')==True):
        shutil.copyfile(datadir + '/control', tmpdir + '/control')

# copy the coord file written by CHARMM to the temporary directory

shutil.copyfile(turbopwd + '/coord', tmpdir + '/coord')

controlfh=open(tmpdir + '/control', 'r')
controllines=controlfh.readlines()
controlfh.close()

escf_on=False
egrad_on=False
jobtype='dft'
for l in controllines:
    if(l[0:10]=='$scfinstab'):
        if(l[-5:-1]=='rpas' or l[-5:-1]=='ciss'):
            escf_on=True
    if(l[0:6]=='$exopt'):
        egrad_on=True
    if(l[0:4]=='$rij'):
        jobtype='ridft'
    if(l[0:6]=='$ricc2'):
        jobtype='ricc2'

# if present, copy the point_charges file written
# by CHARMM to the temporary directory

if(os.path.exists(turbopwd + '/point_charges')):
    shutil.copyfile(turbopwd + '/point_charges', tmpdir + '/point_charges')

os.chdir(tmpdir)

if(os.path.exists('gradient')):
    os.unlink('gradient')

if(os.path.exists('energy')):
    os.unlink('energy')

if(os.path.exists('pc_gradient')):
    os.unlink('pc_gradient')

if(os.path.exists('escf.log')):
    os.unlink('escf.log')

if(jobtype=='dft'):
    if(nprocs==1):
        os.system(dscf_serial + ' >& dscf.log')
        if(egrad_on):
            os.system(egrad_serial + ' >& escf.log')
        else:
            os.system(grad_serial + ' >& grad.log')

        if(escf_on and not egrad_on):
            os.system(escf_serial + ' >& escf.log')
    else:
        if(mpi):
#	mpi executeables of escf and egrad have not implemented yet.
            os.system(mpiexec + ' -n ' + str(nprocs) + ' ' + dscf_mpi + ' >& dscf.log')
            if(egrad_on):
                os.system(egrad_smp + ' -smpcpus  '+str(nprocs)+ ' >& escf.log')
            else:
                os.system(mpiexec + ' -n ' + str(nprocs) + ' ' + grad_mpi + ' >& grad.log')
            if(escf_on and not egrad_on):
                os.system(escf_smp + '  -smpcpus '+str(nprocs)+ ' >& escf.log')
        else:
            os.system(dscf_smp + ' -smpcpus '+str(nprocs)+ ' >& dscf.log')
            if(egrad_on):
                os.system(egrad_smp + ' -smpcpus  '+str(nprocs)+ ' >& escf.log')
            else:
                os.system(grad_smp + ' -smpcpus '+str(nprocs)+ ' >& grad.log')
            if(escf_on and not egrad_on):
                os.system(escf_smp + '  -smpcpus '+str(nprocs)+ ' >& escf.log')

elif(jobtype=='ridft'):
    if(nprocs==1):
        os.system(ridft_serial + ' >& ridft.log')
        if(egrad_on):
            os.system(egrad_serial + ' >& escf.log')
        else:
            os.system(rdgrad_serial + ' >& rdgrad.log')

        if(escf_on and not egrad_on):
            os.system(escf_serial + ' >& escf.log')
    else:
        if(mpi):
#       mpi executeables of escf and egrad have not implemented yet.
            os.system(mpiexec + ' -n ' + str(nprocs) + ' ' + ridft_mpi + ' >& ridft.log')
            if(egrad_on):
                os.system(egrad_smp + ' -smpcpus  '+str(nprocs)+ ' >& escf.log')
            else:
                os.system(mpiexec + ' -n ' + str(nprocs) + ' ' + rdgrad_mpi + ' >& rdgrad.log')
            if(escf_on and not egrad_on):
                os.system(escf_smp + '  -smpcpus '+str(nprocs)+ ' >& escf.log')
        else:
            os.system(ridft_smp + ' -smpcpus '+str(nprocs)+ ' >& ridft.log')
            if(egrad_on):
                os.system(egrad_smp + ' -smpcpus  '+str(nprocs)+ ' >& escf.log')
            else:
                os.system(rdgrad_smp + ' -smpcpus '+str(nprocs)+ ' >& rdgrad.log')
            if(escf_on and not egrad_on):
                os.system(escf_smp + '  -smpcpus '+str(nprocs)+ ' >& escf.log')

elif(jobtype=='ricc2'):
    if(nprocs==1):
        os.system(dscf_exe + ' >& dscf.log')
        os.system(ricc2_exe + ' >& ricc2.log')
    else:
        if(mpi):
            os.system(mpiexec + ' -n ' + str(nprocs) + ' ' + dscf_mpi + ' >& dscf.log')
            os.system(mpiexec + ' -n ' + str(nprocs) + ' ' + ricc2_mpi + ' >& ricc2.log')

        else:
            os.environ['OMP_NUM_THREADS']=str(nprocs)
            os.system(dscf_smp + ' -smpcpus '+str(nprocs)+ ' >& dscf.log')
            os.system(ricc2_smp + ' >& ricc2.log')
else:
    print jobtype + ' not supported'
    sys.exit()

# if we are calculating the excited states properties, read the absorption energies (nm), rotatory strength, 
# and oscillator strength (in the length representation) from the output file.


if(escf_on):
    fh=open(tmpdir + '/escf.log', 'r')
    lines=fh.readlines()
    fh.close()

    energy_nm=[]
    oscillator_strength=[]
    rotatory_strength=[]
    state=[]
            
    for l in lines:
        if(l[-21:]=='singlet a excitation\n'):
            f=l.split()
            state.append(f[0])
            add2oscillator=True
        if(l[0:24]==' Excitation energy / nm:'):
            f=l.split()
            energy_nm.append(float(f[-1]))
        if(l[0:26]=='    length representation:'):
            f=l.split()
            if(add2oscillator):
                oscillator_strength.append(float(f[-1]))
                add2oscillator=False
        if(l[0:35]=='    length rep. / 10^(-40)erg*cm^3:'):
            f=l.split()
            rotatory_strength.append(float(f[-1]))


    escf_log=turbopwd + 'escf.txt'
    if('ESCF_LOG' in os.environ):
        escf_log=os.environ['ESCF_LOG']
    fh=open(escf_log, 'a')
    if (os.path.getsize(escf_log)==0):
        fh.write("# ex_state     E_ex(nm)   R(10^(-40)erg*cm^3)   Oscillator_strength\n")

    for i in range(0, len(energy_nm)):
        fh.write("%7s     %10.10f     %10.10f       %10.10f \n"%(state[i],energy_nm[i],rotatory_strength[i],oscillator_strength[i]))

    fh.write('\n')
    fh.close()

# copies the TURBOMOLE generated output files from the temporary
# directory to the working directory

if(os.path.exists('pc_gradient')):
    shutil.copyfile('pc_gradient', turbopwd + '/pc_gradient')

fh=open('gradient', 'r')
lines=fh.readlines()
fh.close()

fh=open(turbopwd + '/gradient', 'w')
for l in lines:
    fh.write(l.replace('ex. state energy', 'SCF Energy'))
fh.close()

shutil.copyfile('energy', turbopwd + '/energy')
os.chdir(turbopwd)

