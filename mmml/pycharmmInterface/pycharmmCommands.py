import pycharmm


header = """bomlev -2
prnlev 3
wrnlev 1

!#########################################
! Tasks
!#########################################

! 0:    Do it, or
! Else: Do not, there is no try!
set mini 0
set heat 0
set equi 0
set ndcd 1
! Start Production at dcd number n
set ndcd 0

!#########################################
! Setup System
!#########################################

! Read topology and parameter files
!stream toppar_complex.str

open unit 1 card read name lig.top
read rtf card unit 1
close unit 1

open unit 1 form read name lig.par
read param card unit 1
close unit 1

! File name
set name dclm

OPEN UNIT 1 READ FORM NAME init.pdb
READ SEQU PDB UNIT 1
CLOSE UNIT 1
GENERATE DCM FIRST NONE LAST NONE SETUP 

OPEN UNIT 1 READ FORM NAME init.pdb
READ COOR PDB UNIT 1
CLOSE UNIT 1"""

pbcs = """!#########################################
! Setup PBC (Periodic Boundary Condition)
!#########################################

coor stat sele all end

calc xdim = int ( ( ?xmax - ?xmin + 0.0 ) ) + 1
calc ydim = int ( ( ?ymax - ?ymin + 0.0 ) ) + 1
calc zdim = int ( ( ?zmax - ?zmin + 0.0 ) ) + 1

set bsiz = 0

if @xdim .gt. @bsiz then
   set bsiz = @xdim
endif
if @ydim .gt. @bsiz then
   set bsiz = @ydim
endif
if @zdim .gt. @bsiz then
   set bsiz = @zdim
endif

open read unit 10 card name crystal_image.str
crystal defi cubic @bsiz @bsiz @bsiz 90. 90. 90.
crystal build cutoff 14.0 nope 0
image byres xcen 0.0 ycen 0.0 zcen 0.0 sele all end"""

nbonds = """!#########################################
! Bonded/Non-bonded Options & Constraints
!#########################################

! Non-bonding parameters
# nbonds atom ewald pmewald kappa 0.43  -
#   fftx 32 ffty 32 fftz 32 order 4 -
#   cutnb 14.0  ctofnb 12.0 ctonnb 10.0 -
#   lrc vdw vswitch -
#   inbfrq -1 imgfrq -1

! Constrain all X-H bonds
!shake bonh para sele all end
"""

cons = """!#########################################
! Setup PBC (Periodic Boundary Condition)
!#########################################

coor stat sele all end

calc xdim = int ( ( ?xmax - ?xmin + 0.0 ) ) + 1
calc ydim = int ( ( ?ymax - ?ymin + 0.0 ) ) + 1
calc zdim = int ( ( ?zmax - ?zmin + 0.0 ) ) + 1

set bsiz = 0

if @xdim .gt. @bsiz then
   set bsiz = @xdim
endif
if @ydim .gt. @bsiz then
   set bsiz = @ydim
endif
if @zdim .gt. @bsiz then
   set bsiz = @zdim
endif

open read unit 10 card name crystal_image.str
crystal defi cubic @bsiz @bsiz @bsiz 90. 90. 90.
crystal build cutoff 14.0 nope 0
image byres xcen 0.0 ycen 0.0 zcen 0.0 sele all end"""

mini = """!#########################################
! Minimization
!#########################################

mini sd nstep 1000 nprint 100

open write unit 10 card name mini.pdb
write coor unit 10 pdb

"""

heat = """!#########################################
! Heating - NVT
!#########################################

scalar mass stat
calc pmass = int ( ?stot  /  50.0 )
calc tmass = @pmass * 10

calc tmin = 3000 * 0.2 

open write unit 31 card name heat.res       ! Restart file
open write unit 32 file name heat.dcd       ! Coordinates file

dyna leap verlet start -
   timestp 0.001 nstep 5000 -
   firstt @tmin finalt 300 tbath 300 -
   ihtfrq 1000 teminc 5 ieqfrq 0 -
   iasors 1 iasvel 1 iscvel 0 ichecw 0 -
   nprint 1000 nsavc 1000 ntrfrq 200 -
   echeck 100.0   -
   iunrea -1 iunwri 31 iuncrd 32 iunvel -1

open unit 1 write card name heat.crd
write coor card unit 1
close unit 1

open write unit 10 card name heat.pdb
write coor unit 10 pdb

"""

equi = """!#########################################
! Equilibration - NpT
!#########################################

open read  unit 30 card name heat.res      ! Restart file
open write unit 31 card name equi.res      ! Restart file
open write unit 32 file name equi.dcd      ! Coordinates file

dyna restart leap cpt nstep 1000 timestp 0.0002 -
  nprint 1000 nsavc 1000 ntrfrq 200 -
  iprfrq 500 inbfrq 10 imgfrq 50 ixtfrq 1000 -
  ihtfrq 0 ieqfrq 0 -
  pint pconst pref 1 pgamma 5 pmass @pmass -
   iseed  {iseed} -
  hoover reft 300 tmass @tmass firstt 300 -
  iunrea 30 iunwri 31 iuncrd 32 iunvel -1


open unit 1 write card name equi.crd
write coor card unit 1
close unit 1

open write unit 10 card name equi.pdb
write coor unit 10 pdb

close unit 30
close unit 31
close unit 32
"""

dyna = """!#########################################
! Production - NpT
!#########################################

set ndcd {NDCD}

if @ndcd .eq. 0 then
  set m @ndcd
  open read unit 33 card name equi.res        ! Restart file
  open write unit 34 card name dyna.@ndcd.res ! Restart file
  open write unit 35 file name dyna.@ndcd.dcd ! Coordinates file
else
  calc m @ndcd-1
  open read unit 33 card name dyna.@m.res
  open write unit 34 card name dyna.@ndcd.res
  open write unit 35 file name dyna.@ndcd.dcd
endif

dyna restart leap res nstep 10000 timestp 0.0002 -
  nprint 100 nsavc 10 ntrfrq 200 -
  iprfrq 1000 inbfrq -1 imgfrq 50 ixtfrq 1000 -
  ihtfrq 0 ieqfrq 0 -
  cpt pint pconst pref 1 pgamma 0 pmass @pmass -
  hoover reft 300 tmass @tmass -
   iseed  {iseed} -
  IUNREA 33 IUNWRI 34 IUNCRD 35 IUNVEL -1
  
open unit 1 write card name dyna.@ndcd.crd
write coor card unit 1
close unit 1

open write unit 10 card name dyna.@ndcd.pdb
write coor unit 10 pdb

close unit 33
close unit 34
close unit 35

"""

write_system_psf = """write psf card name psf/system.psf
* My PSF file
*
"""


pbcs = """!#########################################
! Setup PBC (Periodic Boundary Condition)
!#########################################

coor stat sele all end

calc xdim = int ( ( ?xmax - ?xmin + 0.0 ) ) + 1
calc ydim = int ( ( ?ymax - ?ymin + 0.0 ) ) + 1
calc zdim = int ( ( ?zmax - ?zmin + 0.0 ) ) + 1

set bsiz = 0

if @xdim .gt. @bsiz then
   set bsiz = @xdim
endif
if @ydim .gt. @bsiz then
   set bsiz = @ydim
endif
if @zdim .gt. @bsiz then
   set bsiz = @zdim
endif

open read unit 10 card name crystal_image.str
crystal defi cubic @bsiz @bsiz @bsiz 90. 90. 90.
crystal build cutoff 14.0 nope 0
image byres xcen 0.0 ycen 0.0 zcen 0.0 sele all end"""

nbonds = """!#########################################
! Bonded/Non-bonded Options & Constraints
!#########################################

! Non-bonding parameters
# nbonds atom ewald pmewald kappa 0.43  -
#   fftx 32 ffty 32 fftz 32 order 4 -
#   cutnb 14.0  ctofnb 12.0 ctonnb 10.0 -
#   lrc vdw vswitch -
#   inbfrq -1 imgfrq -1

! Constrain all X-H bonds
!shake bonh para sele all end
"""

cons = """!#########################################
! Setup PBC (Periodic Boundary Condition)
!#########################################

coor stat sele all end

calc xdim = int ( ( ?xmax - ?xmin + 0.0 ) ) + 1
calc ydim = int ( ( ?ymax - ?ymin + 0.0 ) ) + 1
calc zdim = int ( ( ?zmax - ?zmin + 0.0 ) ) + 1

set bsiz = 0

if @xdim .gt. @bsiz then
   set bsiz = @xdim
endif
if @ydim .gt. @bsiz then
   set bsiz = @ydim
endif
if @zdim .gt. @bsiz then
   set bsiz = @zdim
endif

open read unit 10 card name crystal_image.str
crystal defi cubic @bsiz @bsiz @bsiz 90. 90. 90.
crystal build cutoff 14.0 nope 0
image byres xcen 0.0 ycen 0.0 zcen 0.0 sele all end"""

mini = """!#########################################
! Minimization {iseed} {NDCD}
!#########################################

mini sd nstep 1000 nprint 100

open write unit 10 card name pdb/mini.pdb
write coor unit 10 pdb

"""

pbcset = """ SET BOXTYPE  = RECT
 SET XTLTYPE  = CUBIC
 SET A = {SIDELENGTH}
 SET B = {SIDELENGTH}
 SET C = {SIDELENGTH}
 SET ALPHA = 90.0
 SET BETA  = 90.0
 SET GAMMA = 90.0
 SET IMPATCH = NO
 SET FFTX  = 40
 SET FFTY  = 40
 SET FFTZ  = 40
 SET XCEN  = 0
 SET YCEN  = 0
 SET ZCEN  = 0
"""

# heat = """!#########################################
# ! Heating - NVT {NDCD}
# !#########################################

# scalar mass stat
# calc pmass = int ( ?stot  /  50.0 )
# calc tmass = @pmass * 10

# calc tmin = 300 * 0.2 

# open write unit 31 card name heat.res       ! Restart file
# open write unit 32 file name heat.dcd       ! Coordinates file

# dyna leap verlet start -
#    timestp 0.0002 nstep 50000 -
#    firstt @tmin finalt 300 tbath 300 -
#    ihtfrq 1000 teminc 5 ieqfrq 0 -
#    iasors 1 iasvel 1 iscvel 0 ichecw 0 -
#    nprint 1000 nsavc 1000 ntrfrq 200 -
#    iseed  {iseed} -
#    echeck 100.0   -
#    iunrea -1 iunwri 31 iuncrd 32 iunvel -1

# open unit 1 write card name heat.crd
# write coor card unit 1
# close unit 1

# open write unit 10 card name heat.pdb
# write coor unit 10 pdb

# """

equi = """!#########################################
! Equilibration - NpT {NDCD}
!#########################################

open read  unit 30 card name heat.res      ! Restart file
open write unit 31 card name equi.res      ! Restart file
open write unit 32 file name equi.dcd      ! Coordinates file

dyna restart leap cpt nstep 100000 timestp 0.0002 -
  nprint 1000 nsavc 1000 ntrfrq 200 -
  iprfrq 500 inbfrq 10 imgfrq 50 ixtfrq 1000 -
  ihtfrq 0 ieqfrq 0 -
  pint pconst pref 1 pgamma 5 pmass @pmass -
   iseed  {iseed} -
  hoover reft 300 tmass @tmass firstt 300 -
  iunrea 30 iunwri 31 iuncrd 32 iunvel -1


open unit 1 write card name equi.crd
write coor card unit 1
close unit 1

open write unit 10 card name equi.pdb
write coor unit 10 pdb

close unit 30
close unit 31
close unit 32
"""

dyna = """!#########################################
! Production - NpT
!#########################################

set ndcd {NDCD}

if @ndcd .eq. 0 then
  set m @ndcd
  open read unit 33 card name equi.res        ! Restart file
  open write unit 34 card name dyna.@ndcd.res ! Restart file
  open write unit 35 file name dyna.@ndcd.dcd ! Coordinates file
else
  calc m @ndcd-1
  open read unit 33 card name dyna.@m.res
  open write unit 34 card name dyna.@ndcd.res
  open write unit 35 file name dyna.@ndcd.dcd
endif

dyna restart leap res nstep 10000 timestp 0.0002 -
  nprint 100 nsavc 10 ntrfrq 200 -
  iprfrq 1000 inbfrq -1 imgfrq 50 ixtfrq 1000 -
  ihtfrq 0 ieqfrq 0 -
  cpt pint pconst pref 1 pgamma 0 pmass @pmass -
  hoover reft 300 tmass @tmass -
   iseed  {iseed} -
  IUNREA 33 IUNWRI 34 IUNCRD 35 IUNVEL -1
  
open unit 1 write card name dyna.@ndcd.crd
write coor card unit 1
close unit 1

open write unit 10 card name dyna.@ndcd.pdb
write coor unit 10 pdb

close unit 33
close unit 34
close unit 35

"""

def CLEAR_CHARMM():
    s = """DELETE ATOM SELE ALL END"""
    pycharmm.lingo.charmm_script(s)
    s = """DELETE PSF SELE ALL END"""
    pycharmm.lingo.charmm_script(s)
