
# OpenMM MSES plugin

This is an OpenMM plugin that accelerates the comformational sampling of intrinsically disordered proteins.

The plugin only support the Reference, CPU and CUDA platform, and its further optimization and development are also in progress.

# Descriptions on files

1. CMakeLists.txt 

This is a CMake compile file and is used to create the OpenMMMSES shared libraries.

2. wrappers

This folder includes the C/Fortran API.

3. openmmapi

This folder include the API between OpenMM and MSES plugin platforms.

4. platforms

Currently, the Reference, CPU and CUDA platform can work, where Reference
is the same as the CPU platform.

5. implementations

* step 1: Methodology in the MSES

The total energy of systems is divided into three parts,

ETOT(AT,CG) = E(AT) + E(CG) + EMSES(AT,CG),

where E(AT) and E(CG) are the total energy of AT and CG system, respectively,
in which there are no bonded interactions between them and 
they used the same force field expression. EMSES(AT,CG) is the coupling 
term between the AT and CG system, it is expected that the faster conformational
sampling of CG system with less freedoms can help the AT system to capture 
more conformational transitions, thus accelerating the sampling of atomistic 
structures.

For the E(AT) and E(CG), the customNonbondedForce was used to calculate these two
parts because there are no bonded interactions between the AT and CG system. 
The nonbonded interactions include the vdW and electrostatic pairwise interactions,
so the delta function was used in the nonbonded energies, to determine the charges
and epsilons. If the two atoms in one pair are from the same system, then the 
charge will be q1 x q2, otherwise be zero.

For the EMSES(AT,CG), the OpenMM plugin was used to calcaulte this coupling energy 
and force. Essentially, the EMSES term is the sum of many four-body interaction that 
has the following interactions.

EMSES(AT,CG) = sum(Eterm), Eterm is a four-body interaction.

AT1 AT2 CG1 CG2 kc fmax dcut

dATCG = ABS(|pos(AT1)-pos(AT2)| - |pos(CG1)-pos(CG2)|)

Eterm = 0.5 x kc x dATCG x dATCG (if dATCG < dcut)

Eterm = kc x (A + B/dATCG + fmax x dATCG) (if dATCG >= dcut)

where parameters A and B will be determined by the continuous function for the dcut.

Please read [the MSES paper][https://pubs.acs.org/doi/abs/10.1021/ct500031v]
for details.

* step 2: implement the MSES coupling OpenMM plugin

The MSES plugin is based on the [OpenMM example plugin][OpenMMExamplePlugin],
because that example is a two-body interaction, some difference can be found in
this plugin. Please diff these documents for the comparisons.
Personlly, the OpenMM example plugin provides a simple but unfamilar at first glance
feeling because it used many unfamilar classes, which will cost you much time 
and effort if you want to know the details. However, if your energy interactions 
are not complicated, then building an OpenMM plguin is quite easy, you just compare 
the related documents and find out where you should add or delete.

* step 3: implement this plugin into the CHARMM program

Before putting it into the CHARMM program, the Fortran wrapper should be built,
please see the wrapper document, which has the C/Fortran API.
The CHARMM has a script input, and the reference for the writing an OpenMM plugin
is limited, so the following is described carefully.

First, the input data from MSES should be imported,

""

open read card unit 10 name contact.list

MSES DIST FORC 1 IUNLIST 10 RSWI 2.0 FMAX 0.5 

close unit 10                                         

Here, the contacts between AT and CG are put into the contact.list file,
and the parameters (FORC RSWI and FMAX) are assigned.
For the CHARMM, we hope that CHARMM main function will read the MSES command, 
and it call the ommmses setup subroutine, and set up the input data.

So, the CHARMM main subroutine will be modified to call MSES Fortran module, so that
the input data can be imported into the MSES module.

""

Second, make sure the omm command can call the MSES module

""

omm platform reference

or omm platform cpu

or omm platform cuda

After reading the omm command, it calls the ommcommand setup subroutine, 
and the qmsesomm == qmses if MSES is ready, meaning MSES plugin should be
involved.

""

Third, import the CHARMM MSES input data into the OpenMM MSES plugin

""

energy omm

The energy will be calcuated by using the OpenMM program, before running the
OpenMM MSES plugin, the input data should be imported into the OpenMM program from 
the CHARMM program.
charmm main function will read the energy command, because 
the omm has been called and the ommmain subroutine will be 
called, before running the omm calculations, the charmm data
should be imported into the openmm by the subroutine importpsf,
also the setupmses subroutine will be also called if qmsesomm is true.
Thus, the mses plugin will be involved.

""

Fourth, setup necessary Fortran modules for CHARMM and OpenMM and 
setup the CMakeList.txt file in the main directory

""

please look at the openmm\_api.F90 file in the openmm directory and wrapper directory
in the mses plugin, and CMakeList.txt file in the main directory.

""

please look at the CHARMM code for detailed modifications using the following command.

grep -r "AN EXAMPLE OF OPENMM PLUGIN" "source directory".

# Tutorials

Please look at [these examples for MSES calculations][MSESGitHub].

# Credits

This plugin is currently maintained by [Jianhan Chen group][JianhanChenGroup],
which is based on the [OpenMM example plugin][OpenMMExamplePlugin].

This work was supported by National Science Foundation (MCB 1817332).


[JianhanChenGroup]: https://people.chem.umass.edu/jchenlab/
[OpenMMExamplePlugin]: https://github.com/peastman/openmmexampleplugin
[MSESGitHub]: https://github.com/XipingGong/msestutorial 


