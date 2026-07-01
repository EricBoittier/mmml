# hier fuegen wir den zmatrix-input ein, mit einfrieren von
#  koordinaten das wird dann mode 9


#      PROGRAM DYLCAO
#  QC: 
       SUBROUTINE DYLCAO
#     ================
#
#     Copyright 1991 by Peter Blaudeck, Dirk Porezag 
#
# *********************************************************************
#
#     PROGRAM CHARACTERISTICS 
#     -----------------------
#
# DYLCAO calculates the dynamics of various systems within a 
# two-centre SETB formalism 
#
# *********************************************************************
#
# maxima definitions
#
implicit REAL*8 (A-H,O-Z)
include 'maxima.h'
#
# QC: change izp here to izp2 to avoid conflict with charmm.
integer izp2(NNDIM), lmax(MAXTYP)
integer dim(MAXTYP,MAXTYP), conat(NNDIM+1)
integer numint(MAXTYP,MAXTYP),nlat
real*8 x(3*NNDIM), xparam(3*NNDIM), xold(3*NNDIM), gr(3*NNDIM), norm
real*8 skhtab(10,MAXTAB,MAXTYP,MAXTYP), skstab(10,MAXTAB,MAXTYP,MAXTYP)
real*8 skself(3,MAXTYP), sr(MAXTYP,MAXTYP), nel, convec(3,NNDIM)
real*8 coeff(6,MAXINT,MAXTYP,MAXTYP),xr(2,MAXINT,MAXTYP,MAXTYP)
real*8 efkt(3,MAXTYP,MAXTYP),cutoff(MAXTYP,MAXTYP)
real*8 xm(MAXTYP), xmrc(MAXTYP), rmax(MAXTYP,MAXTYP)
# QC: delete qmat, defined in SCCOPT.
real*8 ar(10,MAXTYP,MAXTYP), slkcutoff
real*8 espin(MAXTYP)
character*65 infile, outfile, name, tmpchr
character*60 smode(9)
character*70 atnames
character*2  atyp(MAXTYP)
logical mdtemp, mdfree, stdc, cgrmin, atomic, evector, scfhelp
logical period, converge, mdcorr, dorelax, doecalc
logical chrr,constr
logical lbfgs,lzmat,lgrad
external skhpar, skspar
#DISPERSION
#integer subsys(NNDIM)
#real*8 pol(MAXTYP,4),ion(MAXTYP,4),Edis
logical dispers
common /disper/ Edis,dispers
#common /disper/ subsys, pol, ion, dispers, Edis
integer writeout
common /eglext/ writeout
#END DISPERSION
common /sktab/ skhtab,skstab,skself,sr,dim
common /spin/ espin
common /spltab/ coeff,xr,efkt,cutoff,numint
common /concgr/ convec, constr, conat
#
###### 
# These data are read in by gettab. 
# Every pair of atoms requires a table containing the Slater-Koster 
# data. The first line consists of the distance stepwidth and the
# number of distance values. If the file describes homonuclear
# interactions, an additional second line contains the energies
# of the valence d, p and s electrons (diagonal elements of the
# hamiltonian). 
# All other lines contain 10 hamiltonian and 10 overlap numbers
# in the order dds ddp ddd pds pdp pps ppp sds sps sss.
# The first of these lines (r = 0) contains (in this order):
# element 01:    mass of the atom (0 for heteroatomic files)
#         02-09: polynomial coefficients for repulsive part
#         10:    range of repulsive part
#         11:    cutoff radius for next neighbour determination
# All energies in H = Hartrees = 4.358d-11 g*cm**2/s**2
# All masses in nucleon masses = 1.6603d-24 g
#
common /lmax/ lmax
#QC: CHANGE THE COMMON BLOCK NAME, SLOPPY CODING!
common /izpc/ izp2, nel, nbeweg
common /rdaten/ ar
common /box/ boxsiz(3,3), xinvbox(3,3), xnullvec(3), period, nlat(3)
common /machine/ racc
common /mcharge/ qzero(MAXTYP), uhubb(MAXTYP)

########################################
# QC: CHARMM passover variables.
INTEGER NSCCTC,NPTC,ICHSCC,IZP
REAL*8 SCCZIN
CHARACTER*10 SCCATM
# QC: May 2001, increase the size of scczin etc. just for 
# replica
COMMON /SCCINF/ NSCCTC,ICHSCC,SCCZIN(NNDIM*3),izp(NNDIM*3),SCCATM(NNDIM*3)
INTEGER MXCHMPT
parameter(MXCHMPT=25120)
real*8  xE(3,MXCHMPT), ZE(MXCHMPT)
integer nE
character*2 EXT
integer mysccid
COMMON/EXTCHR/ xE, ZE, nE, EXT,mysccid
# QC: The parameter common block to be transferred over to eglcao. 
# QC: We take TELEC,SCFTOL,MXITSCF,LPRVEC FROM CHARMM, QMAT from here.
INTEGER MXITSCF,LPRVEC
COMMON/SCCOPT/QMAT(NNDIM), TELEC,SCFTOL,MXITSCF,LPRVEC
# QC: control para input unit
INTEGER IUNIT
# QC: We need the crd from CHARMM BE CAREFUL ABOUT THE DEFINITION@@@
# The following is for large and IMAGE
# in principle it's O.K. because we don't really use CHRMX in the
# calculations
INTEGER MAXAIM
parameter(MAXAIM=2*60120)
REAL*8 CHRMX,CHRMY,CHRMZ,CHRMWMAIN
COMMON /COORD/ CHRMX(MAXAIM),CHRMY(MAXAIM),CHRMZ(MAXAIM),CHRMWMAIN(MAXAIM)

########################################

data smode /'MD relaxation with heat reservoir   ',
            'free MD without external constraints',
            'steepest descent relaxation         ',
            'conjugate gradient relaxation       ',
            'constraint conjugate gradient relaxation',
            'Mulliken charge and atomic energy calculation',
            'Option number 6 + prints out the eigenvectors',
            'BFGS relaxation in internal coordinates',
	    'Option number 6 + Z-matrix input'/
#
if (mysccid ==0) print *,'** dftb (version 26.11.1998) **'
#
# Get the machine accuracy
#
racc = 1.0
while ((1.0+racc) > 1.0) racc = racc*0.5
racc = 2*racc 
#
# Enter relaxation mode and fmax and scftol. Valid relaxation modes are
# 1 ... MD with scaling of velocities according to temperature
# 2 ... MD without scaling of velocities according to temperature
# 3 ... Steepest descent (velocities are set to zero after each step)
# 4 ... Conjugate gradient relaxation
# 5 ... Constraint Conjugate gradient relaxation
# 6 ... Mulliken analysis and atomic energy calculation
# 7 ... Option number 6 + prints out the eigenvectors
# 8 ... BFGS relaxation in internal coordinates
# 9 ... Option number 6 + Z-matrix input
# If the total force acting on each atom in the structure is smaller 
# than fmax, the conjugate gradient or steepest descent routine is 
# converged and the program terminates.
# scftol is the convergence criterion for the charge SCF convergence.
#
#
mdtemp = .false. ; mdfree = .false. ; stdc = .false. ; cgrmin = .false.
atomic = .false. ; evector = .false.; constr =.false.; lbfgs = .false.
lzmat = .false. ; lgrad = .true. 
if (mysccid ==0) print *,'enter mode, fmax, scf, scftol, read charge,dispers,EXT'

# QC: IF CHARMM is set, we read the SCCTB stuff from unit 4

# QC: Although EXT is already set to "CH".
# read *,mode,fmax,scfhelp,scftol,chrr,dispers,EXT
if (EXT=='CH') {
# QC: Force certain mode in CHARMM/SCCTB.
  iunit=4
  scfhelp= .true.
  chrr= .false.
  mode= 3
  open(iunit,file='sccdftb.dat',status='old'); rewind iunit
#  read (iunit,*) dispers,writeout
writeout = 0
                 }
else {
  iunit=5
  read (iunit,*) mode,fmax,scfhelp,scftol,chrr,dispers,EXT
}


if(scfhelp) scftol=abs(scftol)
else scftol=-abs(scftol)
if (mode == 1) mdtemp = .true.
else if (mode == 2) mdfree = .true.
else if (mode == 3) stdc = .true. 
else if (mode == 4) cgrmin = .true.
else if (mode == 5) {
 constr=.true.
 cgrmin=.true.
 }
else if (mode == 6) {
 atomic = .true.
 stdc   = .true.
 }
else if (mode == 7) {
 atomic  = .true.
 stdc    = .true.
 evector = .true.
 }
else if (mode == 8) lbfgs = .true.
else if (mode == 9) {
 lbfgs = .true.
 lzmat = .true.
 }
else {
  print *,'invalid relaxation mode' 
  goto 9 
}
if (mysccid ==0) write(*,'(1x,2a)') 'relaxation mode: ',smode(mode)
#
# Read in name of structure file
#
# print *,'enter filename for input structure'
#
#read *,infile
# 
# All lengths in a.u. = 5.292d-9 cm
#
# print *,'infile :',infile
# QC: Skip this for CHARMM
if (EXT=='CH') {
 nn=NSCCTC
 if (mysccid ==0) print *,'skip struc. file,atoms from charmm:',nn,nE
               }
else { 

open (1,file=infile,status='old'); rewind 1

if (mysccid ==0) write(*,*) 'Input net charge on the system'
read(iunit,*) nel
if (mysccid ==0) write(*,*) 'Number of electrons set to ',nel
if (mysccid ==0) write(*,*) 'Number of moveable atoms'
read(iunit,*) nbeweg
if (mysccid ==0) write(*,*) 'Number of moveable atoms set to ',nbeweg
if(constr) {
 write(*,*) 'How many atoms to be constraint:'
 read(iunit,*) conat(1)
 write(*,*) 'Enter ',conat(1),' times: atom-number constraint-vector (x y z) :'
 do i=1,conat(1) {
  read(iunit,*) conat(i+1),(convec(j,i),j=1,3)
  norm=convec(1,i)*convec(1,i)+convec(2,i)*convec(2,i)+convec(3,i)*convec(3,i)
  norm=sqrt(norm)
  convec(1,i)=convec(1,i)/norm
  convec(2,i)=convec(2,i)/norm
  convec(3,i)=convec(3,i)/norm
 }
}

#
# Read in coordinates and box parameters
#
if(lzmat) {
# call zinput(1,nn,period,izp,xparam,atyp)
}
else {
 call input(1,nn,period,izp,x,boxsiz,atnames)
}
if (nn > NNDIM) {
  print *,'dftb: nn > ', NNDIM
  goto 9
}

close (1)
# QC: SKIP reading end here.
     }

if(nbeweg==0){
nbeweg=nn
}

n3 = 3*nn
nbew3 = 3*nbeweg

#
# calculate Inverse of Boxmatrix
#
#if(period) {
# call inversebox
#}

# QC: moved up.
# close (1)

#
# Check if ntype in limits of MAXTYP 
#
# QC: also copies izp to izp2.
ntype = 1
do n = 1,nn {
  if (izp(n) > ntype) ntype = izp(n)
  izp2(n) = izp(n)
}
if (ntype > MAXTYP) {
  print *,'dftb: ntype >', MAXTYP
  goto 9
}
#
# Read name of output file
#
if (mysccid ==0) write (*,*) 'enter filename for output structure'
# QC: skip 
#read *,outfile

#
# Read in whether atom shall have a virtual atom at the same place,
# i.e. wether you want an additional s (or p) orbital or not
#

n3 = 3*nn
nbew3 = 3*nbeweg

# Hier weitermachen 

#
# Read in maximal angular momentum for each type
# 1,2,3  for  s,p,d, respectively
#
#if (mysccid ==0) write (*,*) 'enter ',ntype,' * lmax'
# QC:
#read *,(lmax(i),i = 1,ntype)
#read (iunit,*) (lmax(i),i = 1,ntype)
#
# Read in Slater-Koster-table 
#
call gettab(ntype)
#
# nel is the number of net electrons
 do i = 1,nn {
   nel = nel+qzero(izp(i))
 }
# QC: Consider the CHARGE here!
   nel = nel - dble(ICHSCC)
if (mysccid ==0) write(*,*) 'Total charge of SCCDFTB molecule:',ICHSCC
if (mysccid ==0) write(*,*) 'Total number of electrons:', nel
#
# set charges
if(scfhelp) {
if(!chrr) {
 do i = 1,nn {
   qmat(i) = qzero(izp(i))
 }
}
else {
 open(37,file="CHR.DAT",status="unknown")
 do i=1,5 {
  read(37,*) tmpchr
 }
 do i=1,nn {
  read(37,*) j,qmat(i)
 }
 close(37)
}
}
#DISPERSION
if(dispers) {
# open(54,file="DISPERSION.DATA",status="unknown") 
# do i=1,ntype {
#   read(54,*) (pol(i,j),j=1,4),(ion(i,j),j=1,4)
# } 
#       maxsyst=0
# do i=1,nn 
#   read(54,*) subsys(i)
#   if ( subsys(i).ge.maxsyst) maxsyst=subsys(i)

 if (EXT=='CH') {
# We need to assign coordinate first for CHARMM
   do i=1,nn  {
    x(3*i-2)=CHRMX(i)/0.52917
    x(3*i-1)=CHRMY(i)/0.52917
    x(3*i  )=CHRMZ(i)/0.52917
              }
# call dispersionread(nn,ntype,izp,x)
  }
  else {
#call dispersionread(nn,ntype,izp,x)
# if (period) {
# write(*,*) 'careful with dispersion!'
# write(*,*) 'implemented only for summation of neighbor'
# write(*,*) 'neighbor cells'
# }
 }
 
}
#END DISPERSION
#CHARMM (QC: we skip here, since passed over with COMMON)
#if (EXT=='CH') { 
# open(98,file='CHARMM.DAT')
# read(98,*) nE
# do i=1,nE {
#   read(98,*) (xE(j,i),j=1,3),ZE(i)
#   do j=1,3 {
#    xE(j,i)=xE(j,i)/0.529177
#   }
# }      
#}
if (EXT=='EF') {
# same variables a for extcharges!
 open(98,file='EXTFIELD.DAT')
 read(98,*) nE
 do i=1,nE {
   read(98,*) xE(1,i),ZE(i)
    xE(1,i)=xE(1,i)/0.529177
 }  
}
#END CHARMM

#
# Get and check masses, data for repulsive part and next neighbours 
#
do i = 1,ntype {
  ar(1,i,i)=skhtab(1,1,i,i)
  xm(i) = ar(1,i,i)
  if (! cgrmin) {
    if (xm(i) < 0.1) {
      write (*,*) 'mass of sort ',i,' is too small: ',xm(i)
      goto 9
    }
    else xmrc(i) = 1.0/(1822.887428*xm(i))
  }
  do j = 1,ntype {
    do k = 2,10 {
      ar(k,i,j) = skhtab(k,1,i,j)
    }
    rmax(i,j) = skstab(1,1,i,j)
  }
}
#
# QC: We return for CHARMM (no period)
if (EXT=='CH'){
   close (4) 
   return
              }

# Check boxsize according to the cutoff in SLK-data:
#    the shortest distance of two vertices has to be twice the shortest
#    cutoff
#

if (period) { 
  slkcutoff=0.0d0
  do i = 1,ntype {
    do j = 1,ntype {
     yhlp=sr(i,j)*dim(i,j)+0.3
     if(yhlp>slkcutoff) slkcutoff=yhlp
   }
  }
  call gamma_summind(slkcutoff,boxsiz,nlat)
  write(*,*) 'Number of lattice sums for Gammapoint matrix:', nlat(1), nlat(2), nlat(3)
}
# Initialize random generator
# Set up initial velocities and old coordinates
# Take care: frozen coordinates must have xold(i) = x(i)
#
inr1 = 1802; inr2 = 9373; call rmarin(inr1,inr2)
do i = 1,n3 {
  xold(i) = x(i)
}
if (mdtemp) {
  do i = 1,nbew3 {
    xold(i) = x(i) - 0.1*(ranmar()-0.5)
  }
  
}
#
# BFGS optimization in internal coordinates starts here
# 
if (lbfgs) {
#  call bfgs(n3,telec,x,xparam,scftol,atomic,e,eel,gr,niter_
#               ,evector,qmat,fmax,infile,izp,lzmat)
#  converge = .true.
  write(*,*) 'bfgs not activated'
  goto 8
}
#
# Setup for relaxation loop
#
icycle = 0; maxrun = 0; irun = 1; converge = .false.
e = 0.0; eel = 0.0; dxsmx = 0.0; fsmx = 0.0; 
#
# Start relaxation loop 
# Multiple input lines are only allowed if (mdtemp), they do not make
# sense in any other case. There would be a problem with xold if the
# time stepwidth is changed since xold is with respect to the old time
# stepwidth. However, since vnorm is always called in the first step,
# xold will be set to a reasonable value. 
#
repeat {

  if (irun > maxrun) {
    icycle = icycle+1
    if ((irun > 1) && (! mdtemp) && (! mdfree)) {   
#
# Close cgrad files and leave
#
      icgr = 1
      if (cgrmin) call cgrad(icgr,nbew3,fmax,e,x,gr)
      goto 9
    }
    irun = 1
#
# Read in deltat, tatom, telec, wvscale, maxrun
# deltat is the time stepwidth in 2.4189d-17 s. tatom is the temperature
# of the atomar system, telec the the temperature of the electronic
# system, both in K. wvscale is the probability to rescale the
# velocities after each step according to the temperature. It may be
# seen as a simple coupling parameter between the system and a
# heat reservoir. maxrun is maximum number of steps to perform
# in that cycle.
#
    dorelax = .true. ; doecalc = .true.
    write (*,*) 'enter deltat, tatom, telec, wvscale, maxrun'
    read (*,*,end = 9) deltat, tatom, telec, wvscale, maxrun
    if(atomic) {
     deltat = 0.0
     maxrun = 0.0
     }
    delth2 = deltat**2
    if(telec > 5.0 && maxrun>1) {
      write(*,*) 'Warning: Relaxation with finite electron temperature may not'
      write(*,*) '         lead to the right result!'
    }
#
# Check if deltat is greater than zero 
#
    if (deltat < 1.0d-10) {
      dorelax = .false. ;
      if (maxrun > 0) {
        maxrun = 0
        print *,'maxrun has been set to 0'
      }
    }
    else rcdt = 1.0/deltat
    if (maxrun < 0) doecalc = .false.

    if (mdtemp || mdfree) mdcorr = .true. ; else mdcorr = .false.
    open(95,file='ENERGY.TMP',form='formatted',status='unknown')
    close(95)
    open(95,file='ENERGY.TMP',form='formatted',status='unknown')
    write (*,12)
    write (95,12)
    close(95)
  }
#
# Done with setup
#
  if (dorelax && (! cgrmin)) {
#
# Mass centre --> 0,0,0
#  
#   if (! constr ){
    if (EXT=='NO') {
    if (deltat > 1.0d-10) call xnorm(nn,x,xold,xm,izp)
    }
#   }
#
# All atoms back into the supercell
#
    if (period) {
      
      do i = 1,nn {
        call coordback(x(3*i-2),x(3*i-1),x(3*i))
      }
    }
    
    
  }
#
# Set total impulse = 0, scale velocities according to temperature
# Velocities must always be scaled in the first step
#
  if (dorelax && mdtemp) {
    if (irun == 1) call vnorm(nbeweg,x,xold,tatom,deltat,xm,izp) 
    else if (ranmar() < wvscale) {
     mdcorr = .true.
     call vnorm(nbeweg,x,xold,tatom,deltat,xm,izp) 
    }
  }
#
# If (! doecalc), skip energy evaluation and geometry update
#
  if (! doecalc) goto 7
#
# Do the work (total energy, gradient)
#
  call eglcao(n3,telec,x,scftol,atomic,e,eel,gr,niter,evector,qmat,lgrad )
     

#
# Output of necessary imnformation (forces, masses, etc.) for vibrational modes
#
#
 call outforces(nbeweg,izp,gr,xm)

#
# Calculate fsmx, the maximal force acting on a single atom
#
  fsmx = 0.0
  do i = 1,nbeweg {
    ftmp = 0.0
    do j = -2,0 {
      ftmp = ftmp + gr(3*i+j)**2
    }
    ftmp = sqrt(ftmp)
    fsmx = max(fsmx,ftmp)
  }
#
# Check if conjugate gradient or stdc method is converged
# Setup icgr
# If (converge && cgrmin), delete file cgrad 
#
  if (stdc || cgrmin) {
    if (fsmx < fmax) converge = .true.
  }
  icgr = 0
  if (converge) icgr = 2
  if (converge && cgrmin) call cgrad(icgr,nbew3,fmax,e,x,gr)
#
# If (converge || (! dorelax)), skip geometry update
#
  if (converge || (! dorelax)) goto 7
#
# Geometry update (conjugate gradient, stdc or verlet)
# Hint: When using the Verlet algorithm and velocities at the same 
# time, we have a problem. The time dependence of x can be written as
#
# (1)  x(t+dt) = x(t) + v(t)*dt + 0.5*dt*dt*f(t)/m
#
# When we use Verlet, we approximate v(t) pretty reasonable as
#
# (2)  v(t) = (x(t+dt)-x(t-dt))/(2*dt)
# 
# Substituting (2) in (1) leads to the Verlet formula
# 
# (3)  x(t+dt) = 2*x(t) - x(t-dt) + dt*dt*f(t)/m
#
# This geometry update is due to v(t) and f(t). That means, we have the
# coordinates for x(t+dt), but not the velocities. But we desperately 
# need these velocities to determine the temperature of the system.
# Thus we have to make a compromise: use v(t+dt) = (x(t+dt) - x(t))/dt 
# to determine the temperature and to rescale the velocities.
# There is one more thing to take care of: There are a lot of cases when
# we exactly know v(t): either in the very first step of the MD simulation
# or after we did a rescaling of the velocities according to the current
# temperature. In either case, we have determined the velocity of the 
# atoms using the formula v(t) = (x(t)-x(t-dt))/dt, which is in
# contrast to (2). However, after calling eglcao we know f(t) and
# are able to correct for this error by resetting x(t-dt) according to
#
# (4)  x(t-dt) = x(t-dt) + 0.5*f(t)/m
#
# This is what is done in the next loop. It is executed if (irun == 1) 
# or if vscale has been called.
# Since the gradient of the frozen coordinates has already been set to
# zero, they will not be changed.
#
  if (mdcorr) {
    mdcorr = .false.
    do i = 1,nbew3 {
     xold(i) = xold(i) + 0.5*delth2*gr(i)*xmrc(izp((i+2)/3))
    }
  }
#
# Now start with the actual geometry update
# dxsmx is the maximal displacement of a single atom
#
  dxsmx = 0.0
  if (cgrmin || stdc) {
    if (cgrmin) call cgrad(icgr,nbew3,fmax,e,x,gr)
    else {
      do i = 1,nbew3 {
      x(i) = xold(i) - delth2*gr(i)*xmrc(izp((i+2)/3))
      }
    }
    do i = 1,nbeweg {
      dxtmp = 0.0
      do j = -2,0 {
        dxtmp = dxtmp + (x(3*i+j)-xold(3*i+j))**2
        xold(3*i+j) = x(3*i+j)
      }
    dxtmp = sqrt(dxtmp)
    dxsmx = max(dxsmx,dxtmp)
    }
  }
  else if (mdtemp || mdfree) { 
    do i = 1,nbew3 { 
      xv = 2*x(i) - xold(i) - delth2*gr(i)*xmrc(izp((i+2)/3)) 
      xold(i) = x(i)
      x(i) = xv
    }
    do i = 1,nbeweg {
      dxtmp = 0.0
      do j = -2,0 {
        dxtmp = dxtmp + (x(3*i+j)-xold(3*i+j))**2
      }
      dxtmp = sqrt(dxtmp)
      dxsmx = max(dxsmx,dxtmp)
    }
  }
#
# create neighbour table, get number of atoms without
# neighbours
#
7 nfree = 0
  
#
# normalize coordinates if one cycle is complete
#
  if (converge || (irun >= maxrun)) {
    if (dorelax && (!cgrmin) ) {
#     if (! constr ){
       if (EXT=='NO') {
      call xnorm(nn,x,xold,xm,izp)
      }
#     }
    }
  }
# calculate atomization energy
  geseatom=0.0
  do i=1,nn {
    geseatom = geseatom+espin(izp(i))
  }
#
#
# output to stdout
#
  open(95,file='ENERGY.TMP',form='formatted',status='unknown')
  write (*,13) irun,niter,nfree,e,eel,(e-geseatom)*627.5095,Edis,dxsmx,fsmx
  write (95,13) irun,niter,nfree,e,eel,Edis,dxsmx,fsmx
  close(95)
#
# output to structure outputfile
#
8 continue  
  write (name,'(a,i3.3)') outfile(:index(outfile,' ')-1), icycle
# QC: disable zoutput
#  if(lzmat) {
#   call zoutput(name,izp,xparam,atyp)
#  }
#  else {
   call outcoord(name, nn, period, ntype, izp, _
     x, boxsiz, atnames)
#  }
#

  irun = irun+1
#
  if (converge) goto 9
}
#
# end of relaxation loop
#
9 continue
  
#
# format specifications
#
12 format(/2x,4x,'iter  free',6x,'e(total)',5x,'e(bandstr)', _
           5x,'e(binding)',6x,'deltax(max)',9x,'f(max)'/,1x,95('='))
13 format(1x,i5,1x,'/',1x,i2,1x,i3,3(1x,f14.6),1x,f14.6,2(1x,f15.6))
print *,' '
print *,'***** end of dftb *****'

#QC:
return
end


