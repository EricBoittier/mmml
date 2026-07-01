! MOBHY : Classical MD with Mobile Hydrogens
! 
! Ref: Lazaridis & Hummer, JCIM 57:2833 (2017)
! 
! Work in progress. For any issue contact tlazaridis@ccny.cuny.edu
! 
! 29feb16: parse which H3O are protonated
!          place H of s.ch using IC BUILD code, print deltae0 
! 1mar16: option to accept when DE<CHOP and mini afterwards
! 15mar16: Diff Ebb for HSD,HSE
! 31mar: Arg
! 10apr: Fhop() Bhop() in EBBR.prm
! 16apr: use IMOVE even without mini
! 26may: fix error at the end of mobhy, fix TIP3 double-switching, move XSAVE allocations to mobhy
! 20jun: fix Arg switching
! 18jul: OH
! 25jul: Update all HB arrays upon switching (mol. or H). Correct placeHonOH. 
! 4aug: SELE for VOLT
! 29nov: variable charge for OH
! 22dec: keyword for order of hop attempts on H3O
! 3mar17: CG instead of numbers for H3O
! 26nov22: XSAVE the dummy (in case O of Asp,Glu have switched, to prevent shake errors)

module mobhy_mod
  use chm_kinds
  use dimens_fcm
  implicit none
  character(len=*),private,parameter :: file_name ="mobhy.F90"

! #if KEY_MOBHY==1 /*main_mobhy*/      Might use later

integer ihopfr,Ntitr,Nh3o,Nh3otot,Nohtot,iflip,iseg,Noh
integer, parameter :: MxTitr=30, MxH3O=1000
integer Pstatus(MxTitr),PstatusH3O(MxH3O),PstatusOH(MxH3O)
integer TitrsitesH3O(MxH3O),TitrsitesOH(MxH3O),Titrsites(MxTitr)
integer iresh3o(MxH3O),iresoh(MxH3O)
logical lmobhy,lhopmin,lhopmin2,ldebug,Lmaft,Largswitch
logical Lbasic,Lacidic,LWdiss,LH3Oback
real(chm_real) :: Ebbreak(10),Fhop(10),Bhop(10),Emobhy,Volt,tempr
character(len=3) :: nstpm
INTEGER VALENCE
real(chm_real) CONC,RVAL,GKAPPA,ACONS,WIDTH

integer,allocatable,dimension(:) :: findires,findidum,iacchb
integer,allocatable,dimension(:) :: Findtitr,FindtitrH3O,FindtitrOH
integer,allocatable,dimension(:) :: VSLCT
logical,allocatable,dimension(:) :: LabileH,Pacc   
integer,allocatable,dimension(:,:) :: atypflip
real(chm_real),allocatable,dimension(:,:) :: chrgflip
real(chm_real),allocatable,dimension(:) :: XSAVE,YSAVE,ZSAVE
logical Lvolt,msetup

contains

subroutine mobhy  !----------------------------------------
! Setup mobile proton simulation
! Syntax: MOBHY IHOPfr int NTITr num NH3O nh3o EBBRf file (segid resid p-status)num 
! p-status: 0 if P, 1 if U (2 if U2 and so on for multiple unprotonated forms)
! For His, 1 corresponds to HSD and 2 to HSE
! nh3o: how many actual hydroniums in the initial state
! Specify H.B. criteria with Hbonds command.

use psf
use comand
use string
use number
use consta
use memory
use stream, only: OUTU, PRNLEV 
use code
use select
use rtf, only: natct,atct

integer i,j,k,ires,jres,ititr,icounter,jcounter,ihydcounter,iohcounter
character(len=80) :: readline
character(len=30) :: EBBRfile
character(len=8) :: segd,resd,pst,xres,aname
character(len=*),parameter :: routine_name="mobhy"
real(chm_real) :: tempx,tempy,tempz,COH2
logical LH3Ogiven,LOHgiven

if (.not. msetup) then
call chmalloc(file_name,routine_name,'findires',maxa,intg=findires)
call chmalloc(file_name,routine_name,'findidum',maxa,intg=findidum)
call chmalloc(file_name,routine_name,'VSLCT',maxa,intg=VSLCT)
call chmalloc(file_name,routine_name,'iacchb',maxa,intg=iacchb)
call chmalloc(file_name,routine_name,'Findtitr',maxres,intg=Findtitr)
call chmalloc(file_name,routine_name,'FindtitrH3O',maxres,intg=FindtitrH3O)
call chmalloc(file_name,routine_name,'FindtitrOH',maxres,intg=FindtitrOH)
call chmalloc(file_name,routine_name,'LabileH',maxa,log=LabileH)
call chmalloc(file_name,routine_name,'Pacc',maxa,log=Pacc)
call chmalloc(file_name,routine_name,'atypflip',maxa,4,intg=atypflip)
call chmalloc(file_name,routine_name,'chrgflip',maxa,4,crl=chrgflip)
call chmalloc(file_name,routine_name,'XSAVE',maxa,crl=XSAVE)
call chmalloc(file_name,routine_name,'YSAVE',maxa,crl=YSAVE)
call chmalloc(file_name,routine_name,'ZSAVE',maxa,crl=ZSAVE)
endif

!do i=1,natct
!write (6,*) atct(i)
!enddo

lmobhy=.true.
Findtitr=0
FindtitrH3O=0
FindtitrOH=0

Ihopfr= gtrmi(comlyn,comlen,'IHOP',10)
Ntitr= gtrmi(comlyn,comlen,'NTIT',0)
LBASIC=.false.
LACIDIC=.false.
LACIDIC= indxa(comlyn,comlen,'ACID') > 0  ! This way you can enforce A or B regardless of nh3o,noh
LBASIC= indxa(comlyn,comlen,'BASI') > 0
Nh3o= gtrmi(comlyn,comlen,'NH3O',0) 
Noh= gtrmi(comlyn,comlen,'NOH',0) 
if (noh.gt.0) LBASIC=.true.
if (nh3o.gt.0) LACIDIC=.true.
EBBRfile= gtrma30(comlyn,comlen,'EBBR')      ! file name must be <= 30 characters
! convert to lowercase
call cnvtlc(EBBRfile,LEN(EBBRfile))
Ldebug= indxa(comlyn,comlen,'DEBU') > 0
LWdiss= indxa(comlyn,comlen,'WDIS') > 0    ! allow water autoionization or not
Lhopmin= indxa(comlyn,comlen,'MINI') > 0
Lmaft= indxa(comlyn,comlen,'MAFT') > 0
LH3Oback= indxa(comlyn,comlen,'HBAC') > 0
LH3Ogiven= indxa(comlyn,comlen,'H3OS') > 0
LOHgiven= indxa(comlyn,comlen,'OHS') > 0
if (Lhopmin .or. Lmaft) then
  Lhopmin2= indxa(comlyn,comlen,'MIN2') > 0
  nstpm=gtrma(comlyn,comlen,'NSTE')
  if (nstpm.eq.'   ') nstpm='50 '
endif
VOLT = GTRMF(COMLYN,COMLEN,'VOLT',ZERO)
COH2=-1.0                                    ! charge on OH2 of OH
COH2 = GTRMF(COMLYN,COMLEN,'COH2',COH2)
IF (abs(VOLT).GT.ZERO) LVOLT=.TRUE.
IF (LVOLT) THEN
  TEMPR= GTRMF(COMLYN,COMLEN,'TEMP',ROOMT)
  CONC= GTRMF(COMLYN,COMLEN,'CONC',ONE)
  VALENCE= GTRMI(COMLYN,COMLEN,'VALE',1)
  RVAL=FLOAT(VALENCE)
  GKAPPA= 5.622667*RVAL*SQRT(CONC/TEMPR)     !1/debye length
  WIDTH= GTRMF(COMLYN,COMLEN,'WIDT',26.0d0)
  ACONS= VOLT/(2.+40.*GKAPPA*WIDTH)
  VSLCT=1     ! By default all selected
  CALL SELCTA(COMLYN,COMLEN,VSLCT,(/ZERO/),(/ZERO/),(/ZERO/),(/ZERO/),.false.)
ENDIF

! process titratable residues
! For now only ASP,GLU,HIS,ARG (and H3O) allowed. Arg preliminary, not tested.
! For His, P=doubly protonated, U1=HSD, U2=HSE

do ititr=1,Ntitr
  segd=NEXTA4(COMLYN,COMLEN)
  resd=NEXTA4(COMLYN,COMLEN)
  pst=NEXTA4(COMLYN,COMLEN)    
  if (pst.eq.'P   ') then
    Pstatus(ititr)=0
  else if ((pst.eq.'U   ') .or. (pst.eq.'U1  ')) then
    Pstatus(ititr)=1
  else if (pst.eq.'U2  ') then
    Pstatus(ititr)=2
  else if (pst.eq.'U3  ') then
    Pstatus(ititr)=3
  else
    CALL WRNDIE(-5,'<MOBHY>','Unrecognized Protonation Status, &
      &only P,U(=U1),U2,U3 are possible')
  endif 
 
  !Find resnum.   
  iseg=0
  do j=1,nseg
    if (segid(j).eq.segd) iseg=j
  enddo
  if (iseg.eq.0) CALL WRNDIE(-5,'<MOBHY>','Segment not found')
  do k=nictot(iseg)+1,nictot(iseg+1)
    if (resid(k).eq.resd) then
      ires=k
      exit
    endif
  enddo
 
  Titrsites(ititr)=ires             !or -ires if Nt or Ct. Later
  Findtitr(ires)=ititr
enddo ! ititr

! Process ires of H3O, if given (default: the first Nh3o are protonated)
if (LH3Ogiven) then
  do i=1,NH3O
    iresh3o(i)=nexti(COMLYN,COMLEN)
  enddo
endif
if (LOHgiven) then
  do i=1,NOH
    iresoh(i)=nexti(COMLYN,COMLEN)
  enddo
endif

CALL XTRANE(COMLYN,COMLEN,'MOBHY')
IF (COMLEN > 0) CALL DIEWRN(0)

! Initialize the "flip arrays" with the current atom type & charge
! (Some waste of space, only titratables needed)  

atypflip(1:Natom,1)=iac(1:Natom)
atypflip(1:Natom,2)=iac(1:Natom)
atypflip(1:Natom,3)=iac(1:Natom)
atypflip(1:Natom,4)=iac(1:Natom)
chrgflip(1:Natom,1)=cg(1:Natom)
chrgflip(1:Natom,2)=cg(1:Natom)
chrgflip(1:Natom,3)=cg(1:Natom)
chrgflip(1:Natom,4)=cg(1:Natom)
LabileH=.false.
Pacc=.false.
findidum=0       ! findidum(iacc): the atom # of the DUM/H bonded to a potential acceptor

! Fill the "flip" arrays with atom types & charges of the alternative forms
! of the titratable residues. 
! At the same time make list of labile H and potential acceptors
! Here everything is still fully protonated
! The first entry is the atom type or charge of the current (protonated) form
! A search is made for the atom type numbers in the deprotonated form
! The charge is assigned. If atom types or partial charges change in the future,
! this code should also be updated.

icounter=0   ! for labile H
jcounter=0   ! for Pacceptors
do ititr=1,Ntitr
  ires = Titrsites(ititr)
  !if (ires < 0) then    ! termini. See if Nt or Ct      .Later
  !  ires=-res
  !  Lct=.true.
  !  do i=1,nseg
  !  if (nictot(i)+1 == ires) Lct=.false.
  !  enddo
  !  if (Lct) then
  !
  !  else
  !
  !  endif
  !else                 ! side chains
  xres=res(ires)
  restype: select case(xres)
  case ('ASP') restype   ! 1 is protonated, 2 deprotonated
    do k=ibase(ires)+1,ibase(ires+1)
      aname=atype(k)
      atmtyp1: select CASE(aname) 
      CASE ('CB') atmtyp1
        if (ATCT(IAC(k)).ne.'CT2') then 
            CALL WRNDIE(-5,'<MOBHY>',' All titratables must be protonated in the PSF!')
        endif
        atypflip(k,1)=IAC(k)
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'CT2A') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=-0.28
      CASE ('CG') atmtyp1
        atypflip(k,1)=IAC(k)
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'CC') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=0.62
      CASE ('OD1') atmtyp1
        atypflip(k,1)=IAC(k)
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'OC') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=-0.76
        if (Pstatus(ititr) == 1) then
         jcounter=jcounter+1
         Pacc(k)=.true.
        endif
      CASE ('OD2') atmtyp1
        atypflip(k,1)=IAC(k)
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'OC') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=-0.76
        if (Pstatus(ititr) == 1) then
         jcounter=jcounter+1
         Pacc(k)=.true.
        endif
        findidum(k)=k+1     ! assuming HD2 is next atom. 
      CASE ('HD2') atmtyp1
        atypflip(k,1)=IAC(k)
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'DUM') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=0.00
        if (Pstatus(ititr) == 0) then
         icounter=icounter+1
         Labileh(k)=.true.
        endif
      end select atmtyp1
      if (Pstatus(ititr) == 1) then
        iac(k)=atypflip(k,2)
        cg(k)=chrgflip(k,2)
      endif
!      write (6,*) xres,aname,atypflip(k,1),atypflip(k,2)
    enddo
  case ('GLU') restype
    do k=ibase(ires)+1,ibase(ires+1)
      aname=atype(k)
      atmtyp2: select CASE(aname) 
      CASE ('CG') atmtyp2
        chrgflip(k,1)=-0.21
        chrgflip(k,2)=-0.28
      CASE ('CD') atmtyp2
        if (ATCT(IAC(k)).ne.'CD') then
            CALL WRNDIE(-5,'<MOBHY>',' All titratables must be protonated in the PSF!')
        endif
        atypflip(k,1)=IAC(k)
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'CC') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=0.62
      CASE ('OE1') atmtyp2
        atypflip(k,1)=IAC(k)
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'OC') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=-0.76
        if (Pstatus(ititr) == 1) then
         jcounter=jcounter+1
         Pacc(k)=.true.
        endif
      CASE ('OE2') atmtyp2
        atypflip(k,1)=IAC(k)
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'OC') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=-0.76
        if (Pstatus(ititr) == 1) then
         jcounter=jcounter+1
         Pacc(k)=.true.
        endif
        findidum(k)=k+1     ! assuming HE2 is next atom
      CASE ('HE2') atmtyp2
        atypflip(k,1)=IAC(k)
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'DUM') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=0.00
        if (Pstatus(ititr) == 0) then
         icounter=icounter+1
         Labileh(k)=.true.
        endif
      end select atmtyp2
      if (Pstatus(ititr) == 1) then
        iac(k)=atypflip(k,2)
        cg(k)=chrgflip(k,2)
      endif
!      write (6,*) xres,aname,atypflip(k,1),atypflip(k,2)
    enddo
  case ('HSP') restype       !3 His forms: Pstatus 0:HSP, 1:HSD, 2:HSE     
    do k=ibase(ires)+1,ibase(ires+1)
      aname=atype(k)
      atmtyp3: select CASE(aname) 
      CASE ('CB') atmtyp3
        if (ATCT(IAC(k)).ne.'CT2A') then
            CALL WRNDIE(-5,'<MOBHY>',' All titratables must be protonated in the PSF!')
        endif
        atypflip(k,1)=IAC(k)                  
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'CT2') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=-0.09
        atypflip(k,3)=i
        chrgflip(k,3)=-0.08
      CASE ('CG') atmtyp3
        chrgflip(k,1)=CG(k)
        chrgflip(k,2)=-0.05
        chrgflip(k,3)=0.22
      CASE ('CD2') atmtyp3
        chrgflip(k,1)=CG(k)
        chrgflip(k,2)=0.22
        chrgflip(k,3)=-0.05
      CASE ('CE1') atmtyp3
        chrgflip(k,1)=CG(k)
        chrgflip(k,2)=0.25
        chrgflip(k,3)=0.25
      CASE ('ND1') atmtyp3
        atypflip(k,1)=IAC(k)
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'NR1') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=-0.36
        do i=1,natct
          if (atct(i).eq.'NR2') exit
        enddo
        atypflip(k,3)=i
        chrgflip(k,3)=-0.70
        if (Pstatus(ititr) == 2) then  
         jcounter=jcounter+1
         Pacc(k)=.true.
        endif
        findidum(k)=k+1     ! assuming HD1 is next atom
      CASE ('NE2') atmtyp3
        atypflip(k,1)=IAC(k)
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'NR2') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=-0.70
        do i=1,natct
          if (atct(i).eq.'NR1') exit
        enddo
        atypflip(k,3)=i
        chrgflip(k,3)=-0.36
        if (Pstatus(ititr) == 1) then  
         jcounter=jcounter+1
         Pacc(k)=.true.
        endif
        findidum(k)=k+1     ! assuming HE2 is next atom
      CASE ('HD2') atmtyp3
        atypflip(k,1)=IAC(k)
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'HR3') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=0.10
        atypflip(k,3)=i
        chrgflip(k,3)=0.09
      CASE ('HE1') atmtyp3
        atypflip(k,1)=IAC(k)
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'HR1') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=0.13
        atypflip(k,3)=i
        chrgflip(k,3)=0.13
      CASE ('HD1') atmtyp3
        atypflip(k,1)=IAC(k)
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'H') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=0.32
        do i=1,natct
          if (atct(i).eq.'DUM') exit
        enddo
        atypflip(k,3)=i
        chrgflip(k,3)=0.00
        if (Pstatus(ititr) == 0) then
         icounter=icounter+1
         Labileh(k)=.true. 
        endif
      CASE ('HE2') atmtyp3
        atypflip(k,1)=IAC(k)
        chrgflip(k,1)=CG(k)
        do i=1,natct
          if (atct(i).eq.'DUM') exit
        enddo
        atypflip(k,2)=i
        chrgflip(k,2)=0.00
        do i=1,natct
          if (atct(i).eq.'H') exit
        enddo
        atypflip(k,3)=i
        chrgflip(k,3)=0.32
        if (Pstatus(ititr) == 0) then
         icounter=icounter+1
         Labileh(k)=.true. 
        endif
      end select atmtyp3
      if (Pstatus(ititr) == 1) then
        iac(k)=atypflip(k,2)
        cg(k)=chrgflip(k,2)
      elseif (Pstatus(ititr) == 2) then
        iac(k)=atypflip(k,3)
        cg(k)=chrgflip(k,3)
      endif
    enddo
!  case ('ARG') restype       !4 Arg forms: Pstatus 0:ARG, 1:RN1, 2:RN2, 3:RN3
!                             !              iflip  1      2      3      4
!    do k=ibase(ires)+1,ibase(ires+1)
!      aname=atype(k)
!      atmtyp4: select CASE(aname) 
!      CASE ('CD') atmtyp4
!        chrgflip(k,1)=0.20
!        chrgflip(k,2)=0.06
!        chrgflip(k,3)=-0.11
!        chrgflip(k,4)=-0.11
!      CASE ('NE') atmtyp4
!        atypflip(k,1)=73  
!        chrgflip(k,1)=-0.70
!        atypflip(k,2)=507
!        chrgflip(k,2)=-0.86
!        atypflip(k,3)=508
!        chrgflip(k,3)=-0.54
!        atypflip(k,4)=508
!        chrgflip(k,4)=-0.54
!        if (Pstatus(ititr) == 1) then  
!         jcounter=jcounter+1
!         Pacc(k)=.true.
!        endif
!        findidum(k)=k+1     ! assuming HE is next atom
!      CASE ('HE') atmtyp4
!        atypflip(k,1)= 32
!        chrgflip(k,1)=0.44
!        atypflip(k,2)= 497
!        chrgflip(k,2)=0.0
!        atypflip(k,3)= 503
!        chrgflip(k,3)=0.36
!        atypflip(k,4)= 503
!        chrgflip(k,4)=0.44
!        if (Pstatus(ititr) == 0) then
!         icounter=icounter+1
!         Labileh(k)=.true. 
!        endif
!      CASE ('CZ') atmtyp4
!        atypflip(k,1)=46
!        chrgflip(k,1)=0.64
!        atypflip(k,2)=505
!        chrgflip(k,2)=0.66
!        atypflip(k,3)=505
!        chrgflip(k,3)=0.59
!        atypflip(k,4)=505
!        chrgflip(k,4)=0.59
!      CASE ('NH1') atmtyp4
!        atypflip(k,1)=73  
!        chrgflip(k,1)=-0.80
!        atypflip(k,2)=509
!        chrgflip(k,2)=-0.60
!        atypflip(k,3)=507
!        chrgflip(k,3)=-0.91
!        atypflip(k,4)=507
!        chrgflip(k,4)=-0.95
!        if (Pstatus(ititr) == 2 .or. Pstatus(ititr) == 3) then  
!         jcounter=jcounter+1
!         Pacc(k)=.true.
!        endif
!        if (Pstatus(ititr) == 2) then         !you must update this dyn'ly
!         findidum(k)=k+2  
!        elseif (Pstatus(ititr) == 3) then
!         findidum(k)=k+1 
!        endif
!        write (outu,*) 'findidum of ',k,findidum(k)
!      CASE ('HH11') atmtyp4
!        atypflip(k,1)= 32
!        chrgflip(k,1)=0.46
!        atypflip(k,2)= 504
!        chrgflip(k,2)=0.29
!        atypflip(k,3)= 502
!        chrgflip(k,3)=0.37
!        atypflip(k,4)= 497
!        chrgflip(k,4)=0.0
!        if (Pstatus(ititr) == 0) then
!         icounter=icounter+1
!         Labileh(k)=.true. 
!        endif
!      CASE ('HH12') atmtyp4
!        atypflip(k,1)= 32
!        chrgflip(k,1)=0.46
!        atypflip(k,2)= 504
!        chrgflip(k,2)=0.29
!        atypflip(k,3)= 497
!        chrgflip(k,3)=0.0
!        atypflip(k,4)= 502
!        chrgflip(k,4)=0.33
!        if (Pstatus(ititr) == 0) then
!         icounter=icounter+1
!         Labileh(k)=.true. 
!        endif
!      CASE ('NH2') atmtyp4
!        atypflip(k,1)=73  
!        chrgflip(k,1)=-0.80
!        atypflip(k,2)=509
!        chrgflip(k,2)=-0.60
!        atypflip(k,3)=509
!        chrgflip(k,3)=-0.60
!        atypflip(k,4)=509
!        chrgflip(k,4)=-0.60
!      CASE ('HH21') atmtyp4
!        atypflip(k,1)= 32
!        chrgflip(k,1)=0.46
!        atypflip(k,2)= 504
!        chrgflip(k,2)=0.29
!        atypflip(k,3)= 504
!        chrgflip(k,3)=0.33
!        atypflip(k,4)= 504
!        chrgflip(k,4)=0.33
!        if (Pstatus(ititr) == 0) then
!         icounter=icounter+1
!         Labileh(k)=.true. 
!        endif
!      CASE ('HH22') atmtyp4
!        atypflip(k,1)= 32
!        chrgflip(k,1)=0.46
!        atypflip(k,2)= 504
!        chrgflip(k,2)=0.29
!        atypflip(k,3)= 504
!        chrgflip(k,3)=0.33
!        atypflip(k,4)= 504
!        chrgflip(k,4)=0.33
!        if (Pstatus(ititr) == 0) then
!         icounter=icounter+1
!         Labileh(k)=.true. 
!        endif
!      end select atmtyp4
!      if (Pstatus(ititr) == 1) then
!        iac(k)=atypflip(k,2)
!        cg(k)=chrgflip(k,2)
!      elseif (Pstatus(ititr) == 2) then
!        iac(k)=atypflip(k,3)
!        cg(k)=chrgflip(k,3)
!      elseif (Pstatus(ititr) == 3) then
!        iac(k)=atypflip(k,4)
!        cg(k)=chrgflip(k,4)
!      endif
!    enddo
  case default restype   
    CALL WRNDIE(-5,'<MOBHY>','Titratable residue not supported!')
end select restype
enddo   ! do ititr

!For H3O select all protons. 
!In the psf there should be at least as many H3O as titratable s.chains
!Deprotonate all except the number Nh3o specified by the user 
!The user can specify which H3O are protonated. If not, the first N3O
!are assumed protonated

PstatusH3O=1   ! 1 deprotonated, 0 protonated
PstatusOH=0    

ihydcounter=0
iohcounter=0
do ires=1,Nres
  do k=ibase(ires)+1,ibase(ires+1)
    findires(k)=ires
  enddo
  xres=res(ires)
  wattype: select case (xres)
     case ('H3O') wattype 
       !create flip arrays,select all H+ as labile,only first Nh3o are protonated
       ihydcounter=ihydcounter+1
       TitrsitesH3O(ihydcounter)=ires
       FindtitrH3O(ires)=ihydcounter
       if (LH3Ogiven) then
         do i=1,NH3O
            if (ires.eq.iresh3o(i)) PstatusH3O(ihydcounter)=0 
         enddo
       else
         if (ihydcounter.le.Nh3o) PstatusH3O(ihydcounter)=0
       endif
       do k=ibase(ires)+1,ibase(ires+1)
         aname=atype(k)
         watmtyp: select CASE(aname) 
         CASE ('H3') watmtyp       !the DUM is placed on H3. If needed,switch names
           atypflip(k,1)=IAC(k)
           chrgflip(k,1)=CG(k)       
           do i=1,natct
              if (atct(i).eq.'DUM') exit
           enddo
           atypflip(k,2)=i
           chrgflip(k,2)=0.00
           if (PstatusH3O(ihydcounter)==0) then
            icounter=icounter+1
            Labileh(k)=.true.
           endif
         CASE ('H1') watmtyp     
           atypflip(k,1)=IAC(k)
           chrgflip(k,1)=CG(k)        
           do i=1,natct
              if (atct(i).eq.'HT') exit
           enddo
           atypflip(k,2)=i
           chrgflip(k,2)=0.417     !TIP3P
           if (PstatusH3O(ihydcounter)==0) then
            icounter=icounter+1
            Labileh(k)=.true. 
           endif
         CASE ('H2') watmtyp     
           atypflip(k,1)=IAC(k)
           chrgflip(k,1)=CG(k)    
           do i=1,natct
              if (atct(i).eq.'HT') exit
           enddo
           atypflip(k,2)=i
           chrgflip(k,2)=0.417
           if (PstatusH3O(ihydcounter)==0) then
            icounter=icounter+1
            Labileh(k)=.true. 
           endif
         CASE ('OH2') watmtyp     
           chrgflip(k,1)=CG(k)        
           chrgflip(k,2)=-0.834
           if (PstatusH3O(ihydcounter)==1) then
            jcounter=jcounter+1
            Pacc(k)=.true.
           endif
           findidum(k)=k+3     ! assuming order in the psf is: H1,H2,H3 
         end select watmtyp
!         write (6,*) xres,aname,atypflip(k,1),atypflip(k,2)
       enddo   ! k 
!     case ('HOH') wattype     ! in non-basic solution, it should behave just like TIP3
!       iohcounter=iohcounter+1
!       TitrsitesOH(iohcounter)=ires
!       FindtitrOH(ires)=iohcounter
!       if (LOHgiven) then
!         do i=1,NOH
!            if (ires.eq.iresoh(i)) PstatusOH(iohcounter)=1 
!         enddo
!       else
!         if (iohcounter.le.Noh) PstatusOH(iohcounter)=1
!       endif
!       do k=ibase(ires)+1,ibase(ires+1)
!         aname=atype(k)
!         watmtp: select CASE(aname) 
!         CASE ('H2') watmtp       !the DUM is placed on H2. If needed,switch names
!           atypflip(k,1)=1
!           chrgflip(k,1)=0.417
!           atypflip(k,2)=497
!           chrgflip(k,2)=0.00
!           if (PstatusOH(iohcounter)==0 .and. LBASIC) then
!            icounter=icounter+1
!            Labileh(k)=.true.
!           endif
!         CASE ('H1') watmtp     
!           atypflip(k,1)=1
!           chrgflip(k,1)=0.417
!           atypflip(k,2)=2
!           chrgflip(k,2)=0.3                      ! subject to revision
!           if (PstatusOH(iohcounter)==0 .and. LBASIC) then
!            icounter=icounter+1
!            Labileh(k)=.true. 
!           endif
!         CASE ('OH2') watmtp     
!           atypflip(k,1)=3
!           chrgflip(k,1)=-0.834
!           atypflip(k,2)=4
!           chrgflip(k,2)=COH2                     ! subject to revision
!           if (PstatusOH(iohcounter)==1 .or. LACIDIC) then
!            jcounter=jcounter+1
!            Pacc(k)=.true.
!           endif
!           findidum(k)=k+2     ! assuming order H1,H2 in the psf
!         end select watmtp
!       enddo   ! k 
     case ('TIP3') wattype
       !For all waters just select O as p.acceptor
       do k=ibase(ires)+1,ibase(ires+1)
         aname=atype(k)
         if (LBASIC.and.(aname.ne.'OH2')) then
             icounter=icounter+1
             Labileh(k)=.true. 
         endif
         if (LACIDIC.and.(aname.eq.'OH2')) then
             jcounter=jcounter+1
             Pacc(k)=.true.
         endif
       enddo
     end select wattype
enddo  ! do ires

Nh3otot=ihydcounter
Nohtot=iohcounter

if (prnlev.ge.2) write(outu,10) icounter,jcounter   
10 format("MOBHY>Numbers of labile protons and potential acceptors",2I7)     
!write (6,*) 'Findires:'
!write (6,'(15I5)') 'Findires',findires(1:natom)

! All titratables and H3O start fully protonated in the psf
! (Later check, and if not, apply protonation patches?)
! Here set the IAC,CG arrays to the correct form 
! First, for the titratable sidechains                       Must update LabileH !
do ititr=1,Ntitr
  ires = Titrsites(ititr)
  if (Pstatus(ititr)==0) cycle
  if (Pstatus(ititr)==1) then
    iflip=2
  else 
    iflip=3
  endif
  do k=ibase(ires)+1,ibase(ires+1)
   iac(k)= atypflip(k,iflip)          
   cg(k)= chrgflip(k,iflip)
  enddo
enddo
! Then, for the hydroniums. The first NH3O are protonated.
do ititr=1,Nh3otot
  if (PstatusH3O(ititr)==0) cycle
  ires = TitrsitesH3O(ititr)
  do k=ibase(ires)+1,ibase(ires+1)
   iac(k)= atypflip(k,2)          
   cg(k)= chrgflip(k,2)
  enddo
enddo
! And for OH
do ititr=1,Nohtot
  if (PstatusOH(ititr)==0) cycle
  ires = TitrsitesOH(ititr)
  do k=ibase(ires)+1,ibase(ires+1)
   iac(k)= atypflip(k,2)          
   cg(k)= chrgflip(k,2)
  enddo
enddo

!Must call CODES to update the internal energy parameter arrays
CALL CODES(ICB,ICT,ICP,ICI,NATOM,IMOVE,IAC,NBOND,IB,JB,NTHETA,IT,JT,KT, &
NPHI,IP,JP,KP,LP,NIMPHI,IM,JM,KM,LM,QDRUDE,NBDRUDE, & 
#if KEY_CMAP==1
ICCT,NCRTERM,I1CT,J1CT,K1CT,L1CT,I2CT,J2CT,K2CT,L2CT, &
#endif
.true.,.FALSE.)

! Read Bond breaking Energies from param file in format
! H3O value  (->Ebbreak(1))
! Asp value  (->Ebbreak(2))
! Glu value  (->Ebbreak(3))
! HSD value  (->Ebbreak(4))    (the energy of removing HD1)   switch HSD,HSE
! HSE value  (->Ebbreak(5))    (the energy of removing HE2)
! AR1 value  (->Ebbreak(6))    (the energy of removing HE)
! AR2 value  (->Ebbreak(7))    (the energy of removing HH12)
! AR3 value  (->Ebbreak(8))    (the energy of removing HH11)
! H2O value  (->Ebbreak(9))

open (unit=93,file=EBBRfile)   !or call OPNGLU? This is easier
if (msetup) rewind (unit=93)
115     read (93,'(a80)',end=116) readline
if (readline(1:1).eq."!") goto 115
read (unit=readline,fmt='(a3,3F8.1)') xres,tempx,tempy,tempz
ebbrf: select case(xres)
  case ('H3O')
   ebbreak(1)=tempx
   Fhop(1)=tempy
   Bhop(1)=tempz
  case ('ASP')
   ebbreak(2)=tempx
   Fhop(2)=tempy
   Bhop(2)=tempz
  case ('GLU')
   ebbreak(3)=tempx
   Fhop(3)=tempy
   Bhop(3)=tempz
  case ('HSD')
   ebbreak(4)=tempx
   Fhop(4)=tempy
   Bhop(4)=tempz
  case ('HSE')
   ebbreak(5)=tempx
   Fhop(5)=tempy
   Bhop(5)=tempz
  case ('AR1')
   ebbreak(6)=tempx
   Fhop(6)=tempy
   Bhop(6)=tempz
  case ('AR2')
   ebbreak(7)=tempx
   Fhop(7)=tempy
   Bhop(7)=tempz
  case ('AR3')
   ebbreak(8)=tempx
   Fhop(8)=tempy
   Bhop(8)=tempz
  case ('H2O')
   ebbreak(9)=tempx
   Fhop(9)=tempy
   Bhop(9)=tempz
end select ebbrf
goto 115
116     continue
if (prnlev.ge.2) then
write (outu,16) 'Ebb values for H3O,Asp,Glu,Hsd,Hse,Ar*,H2O',(Ebbreak(i),i=1,9) 
write (outu,16) 'Fhop values for H3O,Asp,Glu,Hsd,Hse,Ar*,H2O',(Fhop(i),i=1,9) 
write (outu,16) 'Bhop values for H3O,Asp,Glu,Hsd,Hse,Ar*,H2O',(Bhop(i),i=1,9) 
16 format (A45,/9F8.2)
endif

! Energy related to H-donor, H-acceptor bonds. Set reference value here
Emobhy=ZERO
msetup=.true.

end subroutine mobhy        ! ---------------------------


subroutine hopattempt(ihyd,idono,iacce,nxtistart,accept,beta,XOLD,YOLD,ZOLD,VX,VY,VZ)  
! Called from DCNTRL. Attempts to move a proton along an h.bond
! ihyd: atom number of hopping hydrogen
! idono: atom number of donor                  (barely used)
! iacce:     "          h acceptor
! idum: atom number of dummy where the H will go
! accept: logical returned
! beta: 1/kT

use psf   
use coord
use deriv
use number 
use consta
use dimens_fcm
use clcg_mod, only: random,oldrandom
use bases_fcm
use stream, only: OUTU, PRNLEV 
use energym, only: energy,eprop,eterm
use abnerm, only: abner
use code 
use memory
use reawri, only: NSAVC,ISEED
use inbnd, only: NNNB            !only for debugging
use string, only: STRLNG
use chutil, only: ATOMID
use intcor_module, only: icr_struct
use intcor2, only: BILDC   
use hbondm, only: IHB,JHB,KHB,nhb

integer :: i,j,k,ihyd,idono,iacce,idum,ires,jres,iires,iaccehh22
integer :: iother,jresother,kother,iflip,jflip,ifirst,iresother
integer :: nsavcsave,igrp,icount,comlenm,nxtistart
integer,save :: iseed1
logical :: accept,Lswitch
character(len=8) :: xires,xjres,name,nameh,named
character(len=8) :: sid,rid,ren,ac
character(len=80) :: comlynm
character(len=*),parameter :: routine_name="hopattempt"
real(chm_real) :: tempx,tempy,tempz,EBBdon,EBBacc,beta,chop
real(chm_real) :: Eold,Enew,DeltaE,DeltaE0,Eold0,Enew0,tkelv,dist1,dist2,dist,angle
! The bond lengths and angles below must be the same as in the parameter file!!
real(chm_real), parameter :: RBTIP=0.9572, RBH3O=0.9517, ABTIP=1.8242, ABH3O=1.9268  !in rad
real(chm_real) :: XDON(40),YDON(40),ZDON(40),XACC(40),YACC(40),ZACC(40)
real(chm_real) :: XOLD(:),YOLD(:),ZOLD(:)
real(chm_real) :: VX(:),VY(:),VZ(:)

data iseed1 /272834949/

!if (Ldebug) then
!write(6,*) '*Coordinates entering'
!write(6,*) '*'
!do iires=1,nres
!do i=ibase(iires)+1,ibase(iires+1)
  !write (6,'(2I7,4F10.5)') i,iac(i),cg(i),X(i),Y(i),Z(i)
!  call ATOMID(I,SID,RID,REN,AC)
!  write (6,'(2I5,1X,A4,1X,A4,3F10.5,1X,A4,1X,A4,F10.5)') i,iires, & 
!     res(iires),atype(i),X(i),Y(i),Z(i),sid,rid,Wmain(i)
!enddo
!enddo
!write (6,*) "Velocities entering hopattempt"
!do i=1,natom
!write (6,'(2I7,4F10.5)') i,iac(i),cg(i),VX(i),VY(i),VZ(i)
!enddo
!if (prnlev.ge.2) write(6,55) EPROP(3)     ! later outu
!55 format("HOPATT>Energy before Hopatt",F15.5)
!write (6,'(7F11.2)') Eterm(1:7),Eterm(16:17),Eterm(19:20),Eterm(46)
!call update("",0,x,y,z,wmain,.false., &
!          .true.,.true.,.false.,.true.,0,0,0,0,0,0,0)
!CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)     
!if (prnlev.ge.2) write(6,56) EPROP(3)     ! later outu
!56 format("HOPATT>Energy before switches",F15.5)
!write (6,'(7F11.2)') Eterm(1:7),Eterm(16:17),Eterm(19:20),Eterm(46)
!write (6,*) 'NNNB',NNNB
!write (6,*) 'BIMAG%NIMNB'
!write (6,'(I14)') BIMAG%NIMNB
!endif

ires=findires(ihyd)
jres=findires(iacce)
xires=res(ires)
xjres=res(jres)
name=atype(iacce)
nameh=atype(ihyd)
if (prnlev.ge.2) write(outu,48) nxtistart-1,ihyd,iacce,ires,jres
48 format("MOBHY>Hop. entered at step",I8," ihyd=",I6," iacce=",I6," ires=",I6," jres=",I6)
! These are for criteria development. Remove later
!dist=(X(idono)-X(iacce))**2 + (Y(idono)-Y(iacce))**2 + (Z(idono)-Z(iacce))**2
!dist1=(X(idono)-X(ihyd))**2 + (Y(idono)-Y(ihyd))**2 + (Z(idono)-Z(ihyd))**2
!dist2=(X(ihyd)-X(iacce))**2 + (Y(ihyd)-Y(iacce))**2 + (Z(ihyd)-Z(iacce))**2
!angle=(X(idono)-X(ihyd))*(X(ihyd)-X(iacce))+ &
! (Y(idono)-Y(ihyd))*(Y(ihyd)-Y(iacce))+(Z(ihyd)-Z(iacce))*(Z(idono)-Z(ihyd))
!angle=angle/sqrt(dist1*dist2)
!write (6,*) 'dist1,dist2,angle',dist1,dist2,angle
!angle=acos(angle)*180.0/3.14159

! Beginning of switches ----------------------

! If donor is Arg HH21 or HH22, switch the two NH2 groups
! But, if I do the attempts in sequence, a switch may have
! already happened for HH21 (Largswitch). So don't switch for HH22, just change ihyd
! but also change its h.b. partner to what it was originally

if (xires.eq.'ARG'.and. (nameh.eq.'HH21'.or.nameh.eq.'HH22')) then
  ! assume the order of atoms in the psf is as in the RTF file
  ! i.e. : NH1,HH11,HH12,NH2,HH21,HH22
  if (nameh.eq.'HH22') then
    if (Largswitch) then
      iacce=iacchb(ihyd)
      if ((Pacc(iacce) .eqv. .false.).and.(prnlev.ge.2)) then
        write (outu,*) 'Original HB acceptor cannot accept H'
        return
      endif
      ihyd=ihyd-3
      if (prnlev.ge.2) write(outu,39) ihyd,iacce
      39 format("HOPATT>Arg HH22, New ihyd=",I7,' Going back to iacce ',I7)
    else
      ifirst=ihyd-2
      do k=ifirst,ifirst+2
        iother=k-3
        tempx=X(k)
        tempy=Y(k)
        tempz=Z(k)
        X(k)=X(iother)
        Y(k)=Y(iother)
        Z(k)=Z(iother)
        X(iother)=tempx
        Y(iother)=tempy
        Z(iother)=tempz
        tempx=XOLD(k)
        tempy=YOLD(k)
        tempz=ZOLD(k)
        XOLD(k)=XOLD(iother)
        YOLD(k)=YOLD(iother)
        ZOLD(k)=ZOLD(iother)
        XOLD(iother)=tempx
        YOLD(iother)=tempy
        ZOLD(iother)=tempz
        tempx=VX(k)
        tempy=VY(k)
        tempz=VZ(k)
        VX(k)=VX(iother)
        VY(k)=VY(iother)
        VZ(k)=VZ(iother)
        VX(iother)=tempx
        VY(iother)=tempy
        VZ(iother)=tempz
      enddo
      ihyd=ihyd-3
      if (prnlev.ge.2) write(outu,41) ires,ihyd    
      41 format("HOPATT>Switching NH2 groups of Arg",I7," New ihyd=",I7)
    endif
  else
    ifirst=ihyd-1
    do k=ifirst,ifirst+2
      iother=k-3
      tempx=X(k)
      tempy=Y(k)
      tempz=Z(k)
      X(k)=X(iother)
      Y(k)=Y(iother)
      Z(k)=Z(iother)
      X(iother)=tempx
      Y(iother)=tempy
      Z(iother)=tempz
      tempx=XOLD(k)
      tempy=YOLD(k)
      tempz=ZOLD(k)
      XOLD(k)=XOLD(iother)
      YOLD(k)=YOLD(iother)
      ZOLD(k)=ZOLD(iother)
      XOLD(iother)=tempx
      YOLD(iother)=tempy
      ZOLD(iother)=tempz
      tempx=VX(k)
      tempy=VY(k)
      tempz=VZ(k)
      VX(k)=VX(iother)
      VY(k)=VY(iother)
      VZ(k)=VZ(iother)
      VX(iother)=tempx
      VY(iother)=tempy
      VZ(iother)=tempz
    enddo
    ihyd=ihyd-3
    if (prnlev.ge.2) write(outu,43) ires,ihyd 
    43 format("HOPATT>Switching NH2 groups of Arg",I7," New ihyd=",I7)
    Largswitch=.true.
  endif
endif

jres=findires(iacce)
xjres=res(jres)
name=atype(iacce)
nameh=atype(ihyd)

! If acceptor is the wrong O of ASP (OD1) or GLU (OE1), 
! switch coordinates and velocities with OE2,OD1. Also set iacce=iother.
! XOLD must be switched also. It contains either the old coordinates (ORIG verlet)
! or the velocities one half step ahead (Leapfrog)
! (actually, VX may not need to be switched, but to be safe...)

Lswitch=.false.
if (xjres.eq.'ASP'.and.name.eq.'OD1') then
 do k=ibase(jres)+1,ibase(jres+1)
   if (atype(k).eq.'OD2') iother=k
 enddo
 Lswitch=.true.
endif
if (xjres.eq.'GLU'.and.name.eq.'OE1') then
 do k=ibase(jres)+1,ibase(jres+1)
   if (atype(k).eq.'OE2') iother=k
 enddo
 Lswitch=.true.
endif

if (LSWITCH) then
  tempx=X(iacce)
  tempy=Y(iacce)
  tempz=Z(iacce)
  X(iacce)=X(iother)
  Y(iacce)=Y(iother)
  Z(iacce)=Z(iother)
  X(iother)=tempx
  Y(iother)=tempy
  Z(iother)=tempz
  tempx=XOLD(iacce)
  tempy=YOLD(iacce)
  tempz=ZOLD(iacce)
  XOLD(iacce)=XOLD(iother)
  YOLD(iacce)=YOLD(iother)
  ZOLD(iacce)=ZOLD(iother)
  XOLD(iother)=tempx
  YOLD(iother)=tempy
  ZOLD(iother)=tempz
  tempx=VX(iacce)
  tempy=VY(iacce)
  tempz=VZ(iacce)
  VX(iacce)=VX(iother)
  VY(iacce)=VY(iother)
  VZ(iacce)=VZ(iother)
  VX(iother)=tempx
  VY(iother)=tempy
  VZ(iother)=tempz
  iacce=iother    
  if (prnlev.ge.2) write(outu,20) xjres,jres    
  20 format("HOPATT>Switching oxygens of ",A5,I7)
endif

! If acceptor is TIP3, switch coor/vel with a random deprotonated H3O
! Also if it is protonated HOH

if ((xjres.eq.'TIP3').or.((xjres.eq.'HOH').and. & 
                           (PstatusOH(FindtitrOH(jres)).eq.0))) then
  Lswitch=.false.
  do j=1,5*Nh3otot  
    tempx=Nh3otot*oldrandom(iseed1)
    i=int(tempx)+1
    if (PstatusH3O(i)==0) cycle
    jresother=TitrsitesH3O(i)
    k=ibase(jresother)+1   ! assuming OH2 is the first atom
    Lswitch=.true.
    exit
  enddo 
  if (.not.Lswitch) CALL WRNDIE(-5,'<Hopattempt>','Not enough H3O in the system')
! Found a suitable H3O. Now switch coor/vel of OH2,H1,H2 
  do k=ibase(jres)+1,ibase(jres+1)
    kother=ibase(jresother)+1 + (k-ibase(jres)-1)
    tempx=X(k)
    tempy=Y(k)
    tempz=Z(k)
    X(k)=X(kother)
    Y(k)=Y(kother)
    Z(k)=Z(kother)
    X(kother)=tempx
    Y(kother)=tempy
    Z(kother)=tempz
    tempx=XOLD(k)
    tempy=YOLD(k)
    tempz=ZOLD(k)
    XOLD(k)=XOLD(kother)
    YOLD(k)=YOLD(kother)
    ZOLD(k)=ZOLD(kother)
    XOLD(kother)=tempx
    YOLD(kother)=tempy
    ZOLD(kother)=tempz
    tempx=VX(k)
    tempy=VY(k)
    tempz=VZ(k)
    VX(k)=VX(kother)
    VY(k)=VY(kother)
    VZ(k)=VZ(kother)
    VX(kother)=tempx
    VY(kother)=tempy
    VZ(kother)=tempz
  enddo
  call place4hyd(X(kother-2),X(kother-1),X(kother),X(kother+1),tempx, &
     Y(kother-2),Y(kother-1),Y(kother),Y(kother+1),tempy, &
     Z(kother-2),Z(kother-1),Z(kother),Z(kother+1),tempz, &
     RBH3O,ABH3O)
  XOLD(kother+1)=ZERO
  YOLD(kother+1)=ZERO
  ZOLD(kother+1)=ZERO
  iacce=ibase(jresother)+1    ! assuming OH2 is the first atom
  if (prnlev.ge.2) write(outu,30) jres,jresother   
  30 format("HOPATT>Switching TIP3 or HOH",I7,"  with H3O",I7)
! Find if these TIP3,H3O participate in other h.bonds and update HB arrays
  do k=1,nhb
    if (JHB(k).eq.ibase(jres)+1) then
       JHB(k)=ibase(jresother)+1
    elseif (JHB(k).eq.ibase(jresother)+1) then
       JHB(k)=ibase(jres)+1
    endif
    if (IHB(k).eq.ibase(jres)+1) then
       IHB(k)=ibase(jresother)+1
    elseif (IHB(k).eq.ibase(jresother)+1) then
       IHB(k)=ibase(jres)+1
    endif 
    if (KHB(k).eq.ibase(jres)+2) then
       KHB(k)=ibase(jresother)+2
    elseif (KHB(k).eq.ibase(jres)+3) then
       KHB(k)=ibase(jresother)+3
    elseif (KHB(k).eq.ibase(jresother)+2) then
       KHB(k)=ibase(jres)+2
    elseif (KHB(k).eq.ibase(jresother)+3) then
       KHB(k)=ibase(jres)+3
    endif 
  enddo
endif

! If donor is TIP3 (basic sol), switch coor/vel with a protonated OH

if (xires.eq.'TIP3') then
  Lswitch=.false.
  do j=1,5*Nohtot  
    tempx=Nohtot*oldrandom(iseed1)
    i=int(tempx)+1
    if (Ldebug.and.(PRNLEV.ge.2)) write (outu,*) 'iseed1,random,i',iseed1,tempx,i
    if (PstatusOH(i)==1) cycle
    iresother=TitrsitesOH(i)
    k=ibase(iresother)+1   ! assuming OH2 is the first atom
    Lswitch=.true.
    exit
  enddo 
  if (.not.Lswitch) CALL WRNDIE(-5,'<Hopattempt>','Not enough HOH in the system')
! Found a suitable HOH. Now switch coor/vel 
  do k=ibase(ires)+1,ibase(ires+1)
    kother=ibase(iresother)+1 + (k-ibase(ires)-1)
    tempx=X(k)
    tempy=Y(k)
    tempz=Z(k)
    X(k)=X(kother)
    Y(k)=Y(kother)
    Z(k)=Z(kother)
    X(kother)=tempx
    Y(kother)=tempy
    Z(kother)=tempz
    tempx=XOLD(k)
    tempy=YOLD(k)
    tempz=ZOLD(k)
    XOLD(k)=XOLD(kother)
    YOLD(k)=YOLD(kother)
    ZOLD(k)=ZOLD(kother)
    XOLD(kother)=tempx
    YOLD(kother)=tempy
    ZOLD(kother)=tempz
    tempx=VX(k)
    tempy=VY(k)
    tempz=VZ(k)
    VX(k)=VX(kother)
    VY(k)=VY(kother)
    VZ(k)=VZ(kother)
    VX(kother)=tempx
    VY(kother)=tempy
    VZ(kother)=tempz
  enddo
  if (nameh.eq.'H1') then
    ihyd=ibase(iresother)+2    
  else
    ihyd=ibase(iresother)+3    
  endif
  xires='HOH'
  if (prnlev.ge.2) write(outu,31) ires,iresother   
  31 format("HOPATT>Switching TIP3",I7,"  with HOH",I7)
! Find if these TIP3,HOH participate in other h.bonds and update HB arrays
  do k=1,nhb
    if (JHB(k).eq.ibase(ires)+1) then
       JHB(k)=ibase(iresother)+1
    elseif (JHB(k).eq.ibase(iresother)+1) then
       JHB(k)=ibase(ires)+1
    endif
    if (IHB(k).eq.ibase(ires)+1) then
       IHB(k)=ibase(iresother)+1
    elseif (IHB(k).eq.ibase(iresother)+1) then
       IHB(k)=ibase(ires)+1
    endif 
    if (KHB(k).eq.ibase(ires)+2) then
       KHB(k)=ibase(iresother)+2
    elseif (KHB(k).eq.ibase(ires)+3) then
       KHB(k)=ibase(iresother)+3
    elseif (KHB(k).eq.ibase(iresother)+2) then
       KHB(k)=ibase(ires)+2
    elseif (KHB(k).eq.ibase(iresother)+3) then
       KHB(k)=ibase(ires)+3
    endif 
  enddo
  ires=iresother
endif

!if (Ldebug) then
  !write(6,*) 'Coordinates after TIP3 switch'
  !do i=1,natom
  !write (6,'(2I7,4F10.5)') i,iac(i),cg(i),X(i),Y(i),Z(i)
  !enddo
  ! LDYNA can be 0
!  call update("",0,x,y,z,wmain,.false., &
!            .true.,.true.,.false.,.true.,0,0,0,0,0,0,0)
!  CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)     
!  if (prnlev.ge.2) write(6,557) EPROP(3)     ! later outu
!  557 format("HOPATT>Energy after TIP3 switches",F15.5)
!  write (6,'(7F11.2)') Eterm(1:7),Eterm(16:17),Eterm(19:20),Eterm(46)
! 1:7 : bond,angle,UB,dihe,impr,vdw,elec  16,17: IMvdw,elec 19,20,46:Ew self,ext,excl
!endif 

! If donor is H3O, make sure the donated H is H3. Otherwise switch

if (xires.eq.'H3O'.and.nameh.ne.'H3') then
  ! H3 is assumed to be the last atom in the residue
  iother=ibase(ires+1)
  tempx=X(ihyd)
  tempy=Y(ihyd)
  tempz=Z(ihyd)
  X(ihyd)=X(iother)
  Y(ihyd)=Y(iother)
  Z(ihyd)=Z(iother)
  X(iother)=tempx
  Y(iother)=tempy
  Z(iother)=tempz
  tempx=XOLD(ihyd)
  tempy=YOLD(ihyd)
  tempz=ZOLD(ihyd)
  XOLD(ihyd)=XOLD(iother)
  YOLD(ihyd)=YOLD(iother)
  ZOLD(ihyd)=ZOLD(iother)
  XOLD(iother)=tempx
  YOLD(iother)=tempy
  ZOLD(iother)=tempz
  tempx=VX(ihyd)
  tempy=VY(ihyd)
  tempz=VZ(ihyd)
  VX(ihyd)=VX(iother)
  VY(ihyd)=VY(iother)
  VZ(ihyd)=VZ(iother)
  VX(iother)=tempx
  VY(iother)=tempy
  VZ(iother)=tempz
  do k=1,nhb
    if (KHB(k).eq.ihyd) then
       KHB(k)=iother
    elseif (KHB(k).eq.iother) then
       KHB(k)=ihyd
    endif 
  enddo
  ihyd=iother    
  if (prnlev.ge.2) write(outu,40) ires,ihyd   
  40 format("HOPATT>Switching hydrogens of H3O",I7," New ihyd=",I7)
endif

! If donor is HOH, make sure the donated H is H2. Otherwise switch
! (any switch with TIP3 has been done above)

if (xires.eq.'HOH'.and.nameh.ne.'H2') then
  ! H2 is assumed to be the last atom in the residue
  iother=ibase(ires+1)
  tempx=X(ihyd)
  tempy=Y(ihyd)
  tempz=Z(ihyd)
  X(ihyd)=X(iother)
  Y(ihyd)=Y(iother)
  Z(ihyd)=Z(iother)
  X(iother)=tempx
  Y(iother)=tempy
  Z(iother)=tempz
  tempx=XOLD(ihyd)
  tempy=YOLD(ihyd)
  tempz=ZOLD(ihyd)
  XOLD(ihyd)=XOLD(iother)
  YOLD(ihyd)=YOLD(iother)
  ZOLD(ihyd)=ZOLD(iother)
  XOLD(iother)=tempx
  YOLD(iother)=tempy
  ZOLD(iother)=tempz
  tempx=VX(ihyd)
  tempy=VY(ihyd)
  tempz=VZ(ihyd)
  VX(ihyd)=VX(iother)
  VY(ihyd)=VY(iother)
  VZ(ihyd)=VZ(iother)
  VX(iother)=tempx
  VY(iother)=tempy
  VZ(iother)=tempz
  do k=1,nhb
    if (KHB(k).eq.ihyd) then
       KHB(k)=iother
    elseif (KHB(k).eq.iother) then
       KHB(k)=ihyd
    endif 
  enddo
  ihyd=iother    
  if (prnlev.ge.2) write(outu,42) ires,ihyd   
  42 format("HOPATT>Switching hydrogens of HOH",I7," New ihyd=",I7)
endif

!do i=1,natom
!write (6,'(2I7,4F10.5)') i,iac(i),cg(i),X(i),Y(i),Z(i)
!enddo

! End of switches ----------------------------------------------------------

! this is a check
dist=(X(ihyd)-X(iacce))**2 + (Y(ihyd)-Y(iacce))**2 + (Z(ihyd)-Z(iacce))**2
if ((dist.gt.16.0).and.(PRNLEV.ge.2)) write(outu,*) & 
                  'Warning: dist^2 between H-acceptor seems too large'

ires=findires(ihyd)
jres=findires(iacce)
if (Ldebug.and.(PRNLEV.ge.2)) write (outu,*) 'after switches ires,jres',ires,jres
xires=res(ires)
xjres=res(jres)
name=atype(iacce)
nameh=atype(ihyd)
chop=ZERO
select case(xires)
  case ('H3O') 
    Ebbdon=Ebbreak(1)
    select case(xjres)
      case ('H3O') 
        chop=Fhop(1)
      case ('ASP') 
        chop=Bhop(2)
      case ('GLU') 
        chop=Bhop(3)
      case ('HSP') 
        if (name.eq.'ND1') chop=Bhop(4)   
        if (name.eq.'NE2') chop=Bhop(5)   
      case ('ARG') 
        if (name.eq.'NE') then
           chop=Bhop(6)  
        else if (name.eq.'NH1') then
          if (findidum(iacce).eq.iacce+2) then
            chop=Bhop(7)  
          else
            chop=Bhop(8)  
          endif
        endif
    end select 
  case ('HOH') 
    Ebbdon=Ebbreak(9)
    select case(xjres)
      case ('H3O','HOH') 
        chop=Fhop(9)
      case ('ASP') 
        chop=Bhop(2)
      case ('GLU') 
        chop=Bhop(3)
      case ('HSP') 
        if (name.eq.'ND1') chop=Bhop(4)   
        if (name.eq.'NE2') chop=Bhop(5)   
      case ('ARG') 
        if (name.eq.'NE') then
           chop=Bhop(6)  
        else if (name.eq.'NH1') then
          if (findidum(iacce).eq.iacce+2) then
            chop=Bhop(7)  
          else
            chop=Bhop(8)  
          endif
        endif
    end select 
  case ('ASP') 
    Ebbdon=Ebbreak(2)
    chop=Fhop(2)
  case ('GLU') 
    Ebbdon=Ebbreak(3)
    chop=Fhop(3)
  case ('HSP') 
    if (nameh.eq.'HD1') then
      Ebbdon=Ebbreak(4)
      chop=Fhop(4)    
    else if (nameh.eq.'HE2') then
      Ebbdon=Ebbreak(5)
      chop=Fhop(5)    
    endif 
  case ('ARG') 
    if (nameh.eq.'HE') then
      Ebbdon=Ebbreak(6)
      chop=Fhop(6)   
    else if (nameh.eq.'HH12') then
      Ebbdon=Ebbreak(7)
      chop=Fhop(7)   
    else if (nameh.eq.'HH11') then
      Ebbdon=Ebbreak(8)
      chop=Fhop(8)   
    endif
end select 

select case(xjres)
  case ('H3O')
   Ebbacc=Ebbreak(1)
  case ('ASP')
   Ebbacc=Ebbreak(2)
  case ('GLU')
   Ebbacc=Ebbreak(3)
  case ('HSP')
   if (name.eq.'ND1') Ebbacc=Ebbreak(4)
   if (name.eq.'NE2') Ebbacc=Ebbreak(5)
  case ('ARG')
   if (name.eq.'NE') Ebbacc=Ebbreak(6)
   if (name.eq.'NH1') then
     if (findidum(iacce).eq.iacce+2) then
        Ebbacc=Ebbreak(7)
     else
        Ebbacc=Ebbreak(8)
     endif
   endif
  case ('HOH') 
    Ebbacc=Ebbreak(9)
end select

! Fix all residues except the ones involved. Then recalculate energy
! But if there are other fixed atoms in the system? They will be released
! Later: save imove and imoveg and restore at the end

! Removing this I got problems,not always but s'times
!call update("",0,x,y,z,wmain,.false., &
!          .true.,.true.,.false.,.true.,0,0,0,0,0,0,0)
!CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)     
!if (prnlev.ge.2) write(6,57) EPROP(3)     ! later outu
!57 format("HOPATT>Energy after switches",F15.5)
!write (6,'(7F11.2)') Eterm(1:7),Eterm(16:17),Eterm(19:20),Eterm(46)

!Eold=EPROP(3)
!Eold0=Eold

!if (Ldebug) then
!write(6,*) '*Coordinates after switches'
!write(6,*) '*'
!do iires=1,nres
!do i=ibase(iires)+1,ibase(iires+1)
  !write (6,'(2I7,4F10.5)') i,iac(i),cg(i),X(i),Y(i),Z(i)
!  call ATOMID(I,SID,RID,REN,AC)
!  write (6,'(2I5,1X,A4,1X,A4,3F10.5,1X,A4,1X,A4,F10.5)') i,iires, & 
!     res(iires),atype(i),X(i),Y(i),Z(i),sid,rid,Wmain(i)
!enddo
!enddo
!endif

! Save coordinates in case of rejection 
do k=ibase(ires)+1,ibase(ires+1)
   XSAVE(k)=X(k)
   YSAVE(k)=Y(k)
   ZSAVE(k)=Z(k)
enddo
do k=ibase(jres)+1,ibase(jres+1)
   XSAVE(k)=X(k)
   YSAVE(k)=Y(k)
   ZSAVE(k)=Z(k)
enddo

! Record old energy

! Fix non-participating atoms
  imove=1
  do k=ibase(ires)+1,ibase(ires+1)
    select case(atype(k))
      case ('N','HN','CA','HA','C','O','CB','HB1','HB2')
        cycle
      case default
        imove(k)=0
    end select
  enddo
  do k=ibase(jres)+1,ibase(jres+1)
    select case(atype(k))
      case ('N','HN','CA','HA','C','O','CB','HB1','HB2')
        cycle
      case default
        imove(k)=0
    end select
  enddo

  imoveg=1           ! because NBONDMA checks this first. 
  do igrp=1,ngrp
    do i=IGPBS(igrp)+1,IGPBS(igrp+1)
      if (imove(i)==0) imoveg(igrp)=0
    enddo
  enddo
  
  MUSTUP=.true.        !logical to update CODES, checked in update
  call update("",0,x,y,z,wmain,.false., &
            .true.,.true.,.false.,.true.,0,0,0,0,0,0,0)

! Fill up XDON,XACC. Moved here in case of image recentering upon update
icount=0
do k=ibase(ires)+1,ibase(ires+1)
   icount=icount+1
   XDON(icount)=X(k)
   YDON(icount)=Y(k)
   ZDON(icount)=Z(k)
enddo
icount=0
do k=ibase(jres)+1,ibase(jres+1)
   icount=icount+1
   XACC(icount)=X(k)
   YACC(icount)=Y(k)
   ZACC(icount)=Z(k)
enddo

  CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)     
  Eold0=Eprop(3)
  Eold=Eold0
  if (prnlev.ge.2) write(outu,59) Eold           
  59 format("HOPATT>Energy after fixing, before hop. Eold",F15.5)
  if (prnlev.ge.2) write (outu,'(7F11.2)') Eterm(1:7),Eterm(16:17), & 
           Eterm(19:20),Eterm(46)

if (lhopmin2) then
  comlynm='NSTEP '//NSTPM
  comlenm=STRLNG(comlynm)
    nsavcsave=NSAVC
    NSAVC=0               ! otherwise abner will screw up my dynamics dcd
    call abner(comlynm,comlenm)
    NSAVC=nsavcsave
    !write(6,*) 'Coordinates after first minimization'
    !do i=1,natom
    !write (6,'(2I7,4F10.5)') i,iac(i),cg(i),X(i),Y(i),Z(i)
    !enddo
  Eold=Eprop(3)
  if (prnlev.ge.2) write(outu,61) Eold        
  61 format("HOPATT>Energy after fixing & mini, before hop. Eold",F15.5)
!  write (6,'(7F11.2)') Eterm(1:7),Eterm(16:17),Eterm(19:20),Eterm(46)
endif  

! Do the Hop -----------------------------------------------

iflip=2
jflip=2
if ((xires.eq.'HSP').and.(nameh.eq.'HD1')) iflip=3
if ((xires.eq.'ARG').and.(nameh.eq.'HH12')) iflip=3
if ((xires.eq.'ARG').and.(nameh.eq.'HH11')) iflip=4
if ((xjres.eq.'HSP').and.(name.eq.'ND1')) jflip=3
if ((xjres.eq.'ARG').and.(name.eq.'NH1')) then
 if (findidum(iacce).eq.iacce+1) jflip=4    
 if (findidum(iacce).eq.iacce+2) jflip=3    
endif

! change flip arrays of donor 
icount=0
do k=ibase(ires)+1,ibase(ires+1)
   icount=icount+1
   iac(k)= atypflip(k,iflip)          
   cg(k)= chrgflip(k,iflip)
enddo
! change flip arrays of acceptor
icount=0
do k=ibase(jres)+1,ibase(jres+1)
   icount=icount+1
   iac(k)= atypflip(k,1)
   cg(k)= chrgflip(k,1)
enddo

! place the moving H in the best possible position

idum=findidum(iacce)
named=atype(idum)
if (Ldebug.and.(PRNLEV.ge.2)) then
  write (6,*) 'idum, atom number of dummy',idum
  write (6,*) 'Old coordinates of dummy',X(idum),Y(idum),Z(idum)
  write (6,*) idum-2,X(idum-2),Y(idum-2),Z(idum-2)
  write (6,*) idum-1,X(idum-1),Y(idum-1),Z(idum-1)
endif
! But, if you use shake for the dummy, its current position is better
! than the position of H in the donor
!X(idum)=X(ihyd)
!Y(idum)=Y(ihyd)
!Z(idum)=Z(ihyd)
!if (Ldebug) write (6,*) 'New coordinates of dummy',X(idum),Y(idum),Z(idum)

! The NONBOND energy is calculated based on IAC and CG, so ok. (in the slow routines!) 
! But for bonding terms need to update codes arrays
! For now call codes, later perhaps you can change only the relevant entries,
! exclude CMAP etc (But it might be good if you have fixed atoms)

!adjust ideally bond lengths and angles of the donor and acceptor
!Assuming a certain order of the atoms in the residue!
!Right now only for water. I would have to work it out for other residues

if (xires.eq.'H3O') then
  !donor adopts a TIP3 geometry
  call fixbondandangle(XDON(1),XDON(2),XDON(3),YDON(1),YDON(2),YDON(3), &
         ZDON(1),ZDON(2),ZDON(3),RBTIP,ABTIP)
  !Pass the new coordinates to the X,Y,Z arrays
  icount=0
  do k=ibase(ires)+1,ibase(ires+1)
     icount=icount+1
     X(k)=XDON(icount)
     Y(k)=YDON(icount)
     Z(k)=ZDON(icount)
  enddo
endif
! if (xires.eq.'HOH'), should I fix the bond length?

if (xjres.eq.'H3O') then
  !The acceptor adopts a H3O geometry
  call fixbondandangle(XACC(1),XACC(2),XACC(3),YACC(1),YACC(2),YACC(3), &
         ZACC(1),ZACC(2),ZACC(3),RBH3O,ABH3O)
  !And the H must be placed in the right geometry
  call place4hyd(XACC(1),XACC(2),XACC(3),XACC(4),XACC(5), &
       YACC(1),YACC(2),YACC(3),YACC(4),YACC(5), &
       ZACC(1),ZACC(2),ZACC(3),ZACC(4),ZACC(5),RBH3O,ABH3O)
  !This routine returns two solutions, at positions 4 and 5.
  !Select the one that is closer to the old H position
  dist1=(XACC(4)-X(ihyd))**2+(YACC(4)-Y(ihyd))**2+(ZACC(4)-Z(ihyd))**2
  dist2=(XACC(5)-X(ihyd))**2+(YACC(5)-Y(ihyd))**2+(ZACC(5)-Z(ihyd))**2
  !Pass the new coordinates to the X,Y,Z arrays
  icount=0
  do k=ibase(jres)+1,ibase(jres+1)
     icount=icount+1
     X(k)=XACC(icount)
     Y(k)=YACC(icount)
     Z(k)=ZACC(icount)
     if (icount.eq.4) then
       if (dist1.lt.dist2) then
       X(k)=XACC(icount)
       Y(k)=YACC(icount)
       Z(k)=ZACC(icount)
       else
       X(k)=XACC(icount+1)
       Y(k)=YACC(icount+1)
       Z(k)=ZACC(icount+1)
       endif
     endif
  enddo
else if (xjres.eq.'HOH') then
  !call fixbondandangle(XACC(1),XACC(2),XACC(3),YACC(1),YACC(2),YACC(3), &
  !       ZACC(1),ZACC(2),ZACC(3),RBH3O,ABH3O)
  !And the H must be placed in the right geometry
  if (LDEBUG.and.(PRNLEV.ge.2)) then
  write (6,*) 'ACC1',XACC(1),YACC(1),ZACC(1)
  write (6,*) 'ACC2',XACC(2),YACC(2),ZACC(2)
  write (6,*) 'ihyd',X(ihyd),Y(ihyd),Z(ihyd)
  endif
  call placeHonOH(XACC(1),XACC(2),X(ihyd),XACC(4),XACC(5), &
       YACC(1),YACC(2),Y(ihyd),YACC(4),YACC(5), &
       ZACC(1),ZACC(2),Z(ihyd),ZACC(4),ZACC(5),RBTIP,ABTIP)
  !This routine returns two solutions, at positions 4 and 5.
  !Select the one that is closer to the old H position
  dist1=(XACC(4)-X(ihyd))**2+(YACC(4)-Y(ihyd))**2+(ZACC(4)-Z(ihyd))**2
  dist2=(XACC(5)-X(ihyd))**2+(YACC(5)-Y(ihyd))**2+(ZACC(5)-Z(ihyd))**2
  !Pass the new coordinates to the X,Y,Z arrays
  k=ibase(jres+1)     !The H2 of the acceptor
  if (dist1.lt.dist2) then
       X(k)=XACC(4)
       Y(k)=YACC(4)
       Z(k)=ZACC(4)
  else
       X(k)=XACC(5)
       Y(k)=YACC(5)
       Z(k)=ZACC(5)
  endif
  if (LDEBUG.and.(PRNLEV.ge.2)) write (6,*) 'New coor of H, k=',k,X(k),Y(k),Z(k)
else
  !for sidechain acceptors: just place H in the right place using IC BUILD code
  !The IC tables should be filled with the parameters of the protonated res
  !for now ignore changes in geometry
  X(idum)=ANUM
  Y(idum)=ANUM
  Z(idum)=ANUM
  if (icr_struct%lenic.eq.0) CALL WRNDIE(-5,'<Hopattempt>','IC table not read')
! Arg gives me trouble with the original IC table
! No, the problem is with HH11, which is built ignoring the position of HH12
! You must also initialize HH12
  if ((xjres.eq.'ARG').and.(named.eq.'HH11')) then
    X(idum+1)=ANUM
    Y(idum+1)=ANUM
    Z(idum+1)=ANUM
  endif
  CALL BILDC(1,icr_struct%lenic,X,Y,Z, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR, &
               NATOM)
! Update the position of the dummy in case O of Asp,Glu have switched
  XSAVE(idum)=X(idum)
  YSAVE(idum)=Y(idum)
  ZSAVE(idum)=Z(idum)
endif 

! Evaluate energy of product state -----------------------------

MUSTUP=.true.        !logical to update CODES, checked in update
call update("",0,x,y,z,wmain,.false., &
          .true.,.true.,.false.,.true.,0,0,0,0,0,0,0)
CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)     
Enew0=EPROP(3)

!if (Ldebug) then
!  write(6,*) '*Coordinates before minimization'
!  write(6,*) '*'
!  do iires=1,nres
!    do i=ibase(iires)+1,ibase(iires+1)
      !write (6,'(2I7,4F10.5)') i,iac(i),cg(i),X(i),Y(i),Z(i)
!      call ATOMID(I,SID,RID,REN,AC)
!      write (6,'(2I5,1X,A4,1X,A4,3F10.5,1X,A4,1X,A4,F10.5)') i,iires, & 
!         res(iires),atype(i),X(i),Y(i),Z(i),sid,rid,Wmain(i)
!    enddo
!  enddo
!endif

if (Lhopmin) then
  nsavcsave=NSAVC
  NSAVC=0               ! otherwise abner will screw up my dynamics dcd
  call abner(comlynm,comlenm)
  NSAVC=nsavcsave
!  if (Ldebug) then
!    write(6,*) '*Coordinates after minimization'
!    write(6,*) '*'
!    do iires=1,nres
!    do i=ibase(iires)+1,ibase(iires+1)
      !write (6,'(2I7,4F10.5)') i,iac(i),cg(i),X(i),Y(i),Z(i)
!      call ATOMID(I,SID,RID,REN,AC)
!      write (6,'(2I5,1X,A4,1X,A4,3F10.5,1X,A4,1X,A4,F10.5)') i,iires, & 
!         res(iires),atype(i),X(i),Y(i),Z(i),sid,rid,Wmain(i)
!    enddo
!    enddo
!  endif
endif

Enew=EPROP(3)
if (prnlev.ge.2) write(outu,63) Enew          
63 format("HOPATT>Energy after hop. Enew",F15.5)
if (prnlev.ge.2) write (outu,'(7F11.2)') Eterm(1:7),Eterm(16:17), &
    Eterm(19:20),Eterm(46)

DeltaE= Enew-Eold+Ebbdon-Ebbacc
if (prnlev.ge.2) then
  write(outu,70) xires,nameh,xjres,named,DeltaE,dist!   ,angle
endif
70 format("HOPATT>DELTAE,r^2 ",4A8,3F13.5)

!Apply Metropolis-like Criterion.

DeltaE=DeltaE-Chop

if (DeltaE .le. zero) then
  accept=.true.
else 
  tempx=oldrandom(iseed1)
  accept= (Exp(-beta*DeltaE) .ge. tempx)
endif

if (accept) then
  if (prnlev.ge.2) write (outu,112) 'Hop accepted at step',nxtistart-1, &
       ' Donor',ires,nameh,' Accep',jres,name,named,'DE ',Deltae+Chop
  112 format (A20,I8,A6,I5,1X,A4,A6,I5,1X,A4,1X,A4,A3,F13.5)
  if (Lmaft .and. .not. Lhopmin) then
    comlynm='NSTEP '//NSTPM
    comlenm=STRLNG(comlynm)
!    imove=1
!    do k=ibase(ires)+1,ibase(ires+1)
!      select case(atype(k))
!        case ('N','HN','CA','HA','C','O','CB','HB1','HB2')
!          cycle
!        case default
!          imove(k)=0
!      end select
!    enddo
!    do k=ibase(jres)+1,ibase(jres+1)
!      select case(atype(k))
!        case ('N','HN','CA','HA','C','O','CB','HB1','HB2')
!          cycle
!        case default
!          imove(k)=0
!      end select
!    enddo
!    imoveg=1           ! because NBONDMA checks this first. 
!    do igrp=1,ngrp
!      do i=IGPBS(igrp)+1,IGPBS(igrp+1)
!        if (imove(i)==0) imoveg(igrp)=0
!      enddo
!    enddo
!    
!    MUSTUP=.true.        !logical to update CODES, checked in update
!    call update("",0,x,y,z,wmain,.false., &
!              .true.,.true.,.false.,.true.,0,0,0,0,0,0,0)
    nsavcsave=NSAVC
    NSAVC=0               ! otherwise abner will screw up my dynamics dcd
    call abner(comlynm,comlenm)
    NSAVC=nsavcsave
    !Eold=Eprop(3)
    !if (prnlev.ge.2) write(6,61) Eold            ! later 6->outu
    !61 format("HOPATT>Energy after fixing & perhaps mini, before hop. Eold",F15.5)
    !write (6,'(7F11.2)') Eterm(1:7),Eterm(16:17),Eterm(19:20),Eterm(46)
!    imove=0
!    imoveg=0         
!    MUSTUP=.true.        !logical to update CODES, checked in update
!    call update("",0,x,y,z,wmain,.false., &
!            .true.,.true.,.false.,.true.,0,0,0,0,0,0,0)
  endif
  ! update LabileH , Pacc , Pstatus , PstatusH3O 
  ! for the donor 
  select case(xires)
   case ('ASP') 
     do k=ibase(ires)+1,ibase(ires+1)
      if (atype(k).eq.'HD2') LabileH(k)=.false.     !I'm changing IAC, not atype
      if (atype(k).eq.'OD1') Pacc(k)=.true.
      if (atype(k).eq.'OD2') Pacc(k)=.true.
     enddo
     Pstatus(Findtitr(ires))=1
   case ('GLU') 
     do k=ibase(ires)+1,ibase(ires+1)
      if (atype(k).eq.'HE2') LabileH(k)=.false.     
      if (atype(k).eq.'OE1') Pacc(k)=.true.
      if (atype(k).eq.'OE2') Pacc(k)=.true.
     enddo
     Pstatus(Findtitr(ires))=1
   case ('HSP') 
     do k=ibase(ires)+1,ibase(ires+1)
      if (atype(k).eq.'HD1') LabileH(k)=.false.     
      if (atype(k).eq.'HE2') LabileH(k)=.false.     
      if (atype(k).eq.'ND1'.and.(iflip.eq.3)) Pacc(k)=.true.
      if (atype(k).eq.'NE2'.and.(iflip.eq.2)) Pacc(k)=.true.
     enddo
     if (iflip.eq.2) then
       Pstatus(Findtitr(ires))=1
     else
       Pstatus(Findtitr(ires))=2
     endif
   case ('ARG') 
     do k=ibase(ires)+1,ibase(ires+1)
      if (atype(k).eq.'HE') LabileH(k)=.false.     
      if (atype(k).eq.'HH11') LabileH(k)=.false.     
      if (atype(k).eq.'HH12') LabileH(k)=.false.     
      if (atype(k).eq.'HH21') LabileH(k)=.false.     
      if (atype(k).eq.'HH22') LabileH(k)=.false.     
      if (atype(k).eq.'NE'.and.(iflip.eq.2)) Pacc(k)=.true.
      if (atype(k).eq.'NH1'.and.((iflip.eq.3).or.(iflip.eq.4))) then
        Pacc(k)=.true.
        if (iflip.eq.3) findidum(k)=k+2
        if (iflip.eq.4) findidum(k)=k+1
      endif
     enddo
     if (iflip.eq.2) then
       Pstatus(Findtitr(ires))=1
     elseif (iflip.eq.3) then
       Pstatus(Findtitr(ires))=2
     else
       Pstatus(Findtitr(ires))=3
     endif
   case ('H3O') 
     do k=ibase(ires)+1,ibase(ires+1)
      if (atype(k).eq.'H1') LabileH(k)=.false.     
      if (atype(k).eq.'H2') LabileH(k)=.false.     
      if (atype(k).eq.'H3') LabileH(k)=.false.     
      if (atype(k).eq.'OH2') Pacc(k)=.true.
     enddo
     PstatusH3O(FindtitrH3O(ires))=1
   case ('HOH') 
     do k=ibase(ires)+1,ibase(ires+1)
      if (atype(k).eq.'H1') LabileH(k)=.false.     
      if (atype(k).eq.'H2') LabileH(k)=.false.     
      if (atype(k).eq.'OH2') Pacc(k)=.true.
     enddo
     PstatusOH(FindtitrOH(ires))=1
  end select
  ! for the acceptor 
  select case(xjres)
   case ('ASP') 
     do k=ibase(jres)+1,ibase(jres+1)
      if (atype(k).eq.'HD2') LabileH(k)=.true. 
      if (atype(k).eq.'OD1') Pacc(k)=.false.
      if (atype(k).eq.'OD2') Pacc(k)=.false.
     enddo
     Pstatus(Findtitr(jres))=0
   case ('GLU') 
     do k=ibase(jres)+1,ibase(jres+1)
      if (atype(k).eq.'HE2') LabileH(k)=.true.
      if (atype(k).eq.'OE1') Pacc(k)=.false.
      if (atype(k).eq.'OE2') Pacc(k)=.false.
     enddo
     Pstatus(Findtitr(jres))=0
   case ('HSP') 
     do k=ibase(jres)+1,ibase(jres+1)
      if (atype(k).eq.'HD1') LabileH(k)=.true.     
      if (atype(k).eq.'HE2') LabileH(k)=.true.     
      if (atype(k).eq.'ND1') Pacc(k)=.false.
      if (atype(k).eq.'NE2') Pacc(k)=.false.
     enddo
     Pstatus(Findtitr(jres))=0
   case ('ARG') 
     do k=ibase(jres)+1,ibase(jres+1)
      if (atype(k).eq.'HE') LabileH(k)=.true.     
      if (atype(k).eq.'HH11') LabileH(k)=.true.     
      if (atype(k).eq.'HH12') LabileH(k)=.true.     
      if (atype(k).eq.'HH21') LabileH(k)=.true.     
      if (atype(k).eq.'HH22') LabileH(k)=.true.     
      if (atype(k).eq.'NE') Pacc(k)=.false.
      if (atype(k).eq.'NH1') Pacc(k)=.false.
     enddo
     Pstatus(Findtitr(jres))=0
   case ('H3O') 
     do k=ibase(jres)+1,ibase(jres+1)
      if (atype(k).eq.'H1') LabileH(k)=.true.
      if (atype(k).eq.'H2') LabileH(k)=.true.     
      if (atype(k).eq.'H3') LabileH(k)=.true.     
      if (atype(k).eq.'OH2') Pacc(k)=.false.
     enddo
     PstatusH3O(FindtitrH3O(jres))=0
   case ('HOH') 
     do k=ibase(jres)+1,ibase(jres+1)
      if (atype(k).eq.'H1') LabileH(k)=.true.
      if (atype(k).eq.'H2') LabileH(k)=.true.     
      if (atype(k).eq.'OH2') Pacc(k)=.false.
     enddo
     PstatusOH(FindtitrOH(jres))=0
  end select
  ! and change coordinates and velocities of the hopping hydrogen  
!  VX(idum)=VX(ihyd)
!  VY(idum)=VY(ihyd)
!  VZ(idum)=VZ(ihyd)
!  XOLD(idum)=XOLD(ihyd)
!  YOLD(idum)=YOLD(ihyd)
!  ZOLD(idum)=ZOLD(ihyd)
  VX(idum)=ZERO
  VY(idum)=ZERO
  VZ(idum)=ZERO
  XOLD(idum)=ZERO                ! this will be a problem in ORIG verlet !
  YOLD(idum)=ZERO
  ZOLD(idum)=ZERO
  XOLD(ihyd)=ZERO                      
  YOLD(ihyd)=ZERO
  ZOLD(ihyd)=ZERO
  VX(ihyd)=ZERO
  VY(ihyd)=ZERO
  VZ(ihyd)=ZERO
  Emobhy=Emobhy+Ebbdon-Ebbacc
else
  ! If reject, reverse the changes 
  ! reprotonate donor 
  do k=ibase(ires)+1,ibase(ires+1)
   iac(k)= atypflip(k,1)          
   cg(k)= chrgflip(k,1)
   X(k)=XSAVE(k)
   Y(k)=YSAVE(k)
   Z(k)=ZSAVE(k)
  enddo
  ! deprotonate acceptor
  do k=ibase(jres)+1,ibase(jres+1)
   iac(k)= atypflip(k,jflip)  
   cg(k)= chrgflip(k,jflip)
   X(k)=XSAVE(k)
   Y(k)=YSAVE(k)
   Z(k)=ZSAVE(k)
  enddo
endif

! Release all atoms
  imove=0
  imoveg=0         
  MUSTUP=.true.        !logical to update CODES, checked in update
  call update("",0,x,y,z,wmain,.false., &
          .true.,.true.,.false.,.true.,0,0,0,0,0,0,0)

!  CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
!  if (prnlev.ge.2) write(6,60) Eprop(3)           
!  60 format("HOPATT>New Total Energy after release",F15.5)
!  write (6,'(7F11.2)') Eterm(1:7),Eterm(16:17),Eterm(19:20),Eterm(46)
!if (Ldebug) then
!write (6,*) "Coordinates leaving hopattempt"
!do i=1,natom
!write (6,'(2I7,4F10.5)') i,iac(i),cg(i),X(i),Y(i),Z(i)
!enddo
!write (6,*) "Velocities leaving hopattempt"
!do i=1,natom
!write (6,'(2I7,4F10.5)') i,iac(i),cg(i),VX(i),VY(i),VZ(i)
!enddo
!endif

end subroutine hopattempt        ! ---------------------------

subroutine fixbondandangle(x1,x2,x3,y1,y2,y3,z1,z2,z3,RB,THETA)
! Given r1,r2,r3 find new r2 and r3 so that the angle 2-1-3 will be THETA
! and the bonds 1-2 and 1-3 will be RB
! We want the angle to increase or decrease equally from the two sides.
! Break up the problem in two, using rM, the position of the midpoint
! between 2 and 3
real(chm_real) :: x1,y1,z1,x2,y2,z2,x3,y3,z3,xm,ym,zm,R1M
real(chm_real) :: RB,THETA,a,b,c,d,e,f,g,h,i,j,k,l,m,n,p,q,r,dis
!find midpoint
xm=(x2+x3)/2.
ym=(y2+y3)/2.
zm=(z2+z3)/2.
R1M=(xm-x1)**2+(ym-y1)**2+(zm-z1)**2
R1M=sqrt(R1M)
!write (6,*) 'x,y,zM,R1M',xm,ym,zm,R1M
!find equation ax+by+cz+1=0 of plane through the 3 points
call plane(x1,y1,z1,x2,y2,z2,x3,y3,z3,a,b,c)
!write (6,*) 'a,b,c',a,b,c
! find new r2 and r3 (they are the two solutions of a quadratic eq.)
E=RB*R1M*cos(THETA/2.0)
!write (6,*) 'E',E
L=xm-x1
M=ym-y1
N=zm-z1
!write (6,*) 'L,M,N',L,M,N
G=L-N*a/c
H=M-N*b/c
!write (6,*) 'G,H',G,H
I=-b*E/c/H
J=a/c-b*G/c/H
P=1+(G/H)**2+J**2
Q=2*E*G/H**2 +2.*I*J
R=(E/H)**2+I**2-RB**2
!write (6,*) 'P,Q,R',P,Q,R
dis=Q**2-4.0*P*R
X2=(Q+SQRT(dis))/2./P
X3=(Q-SQRT(dis))/2./P
Y2=(E-G*X2)/H
Y3=(E-G*X3)/H
Z2=I-J*X2
Z3=I-J*X3
x2=X2+x1
x3=X3+x1
y2=Y2+y1
y3=Y3+y1
z2=Z2+z1
z3=Z3+z1
!write (6,*) 'x2,y2,z2',x2,y2,z2
!write (6,*) 'x3,y3,z3',x3,y3,z3
! tests
!R1=sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
!R2=sqrt((x3-x1)**2+(y3-y1)**2+(z3-z1)**2)
!angle=(x2-x1)*(x3-x1)+(y2-y1)*(y3-y1)+(z2-z1)*(z3-z1)
!angle=angle/R1/R2
!angle=acos(angle)
!write (6,*) 'Bond length 1-2',R1
!write (6,*) 'Bond length 1-3',R2
!write (6,*) 'Angle 2-1-3',angle*180/3.14159
end subroutine fixbondandangle


subroutine place4hyd(x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,z1,z2,z3,z4,z5,RB,THETA) 
! Given r1,r2,r3 place a third H so that the bond 1-4 will be RB and the angles 
! 2-1-4 and 3-1-4 will be THETA
! There are two solutions, returned in r4,r5. 
! Choose the one that is closest to previous H position
real(chm_real) :: x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,x5,y5,z5
real(chm_real) :: RB,THETA,a,b,c,d,e,f,g,h,i,k,l,p,q,r,dis
a=x2-x1
b=y2-y1
c=z2-z1
d=x3-x1
e=y3-y1
f=z3-z1
g=a-c*d/f
h=b-c*e/f
i=RB**2*cos(theta)*(1.-c/f)
k=RB**2*cos(theta)/f-e*i/f/h
l=d/f-e*g/f/h
p=1+(g/h)**2+l**2
q=2.*i*g/h**2 + 2.*k*l
r=(i/h)**2+k**2-RB**2
dis=Q**2-4.0*P*R
x4=(Q+SQRT(dis))/2./P
x5=(Q-SQRT(dis))/2./P
y4=i/h-g/h*x4
y5=i/h-g/h*x5
z4=k-l*x4
z5=k-l*x5
x4=x4+x1
x5=x5+x1
y4=y4+y1
y5=y5+y1
z4=z4+z1
z5=z5+z1
end subroutine place4hyd

subroutine placeHonOH(x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,z1,z2,z3,z4,z5,RB,THETA) 
! Given r1,r2,r3 find a new position for 3 so that the bond 1-3 will be RB,the angle 
! 2-1-3 will be THETA and 3 will be on the plane of current r1,r2,r3
! There are two solutions, returned in r4,r5. 
! Choose the one that is closest to previous H position
real(chm_real) :: x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,x5,y5,z5
real(chm_real) :: RB,THETA,a,b,c,d,e,f,g,h,i,k,l,p,q,r,dis,bb,aa,dd,ee,ff,gg,hh,v,w,t
call plane(x1,y1,z1,x2,y2,z2,x3,y3,z3,a,b,c)
f=x2-x1
g=y2-y1
h=z2-z1
bb=rb**2 * cos(theta)
aa=-1.0-a*x1-b*y1-c*z1
dd=f-a*h/c
ee=g-b*h/c
ff=bb-aa*h/c
gg=(aa-b*FF/EE)/c
hh=(b*DD/EE-a)/c
v=1.0+DD**2/EE**2+HH**2
w=2.0*GG*HH-2.0*FF*DD/EE**2
t=FF**2/EE**2+GG**2-RB**2
dis=w**2-4.0*v*t
X4=(-w+SQRT(dis))/2./v
X5=(-w-SQRT(dis))/2./v
Y4=(ff-dd*X4)/EE
Y5=(ff-dd*X5)/EE
Z4=(AA-a*X4-b*Y4)/c
Z5=(AA-a*X5-b*Y5)/c
x4=X4+x1
x5=X5+x1
y4=Y4+y1
y5=Y5+y1
z4=Z4+z1
z5=Z5+z1
end subroutine placeHonOH

subroutine plane(x1,y1,z1,x2,y2,z2,x3,y3,z3,a,b,c)
! Find equation of a plane through 3 points, ax+by+cz+1=0
real(chm_real) :: x1,y1,z1,x2,y2,z2,x3,y3,z3,a,b,c,d,d1,d2,d3
d=x1*y2*z3+y1*z2*x3+z1*x2*y3-y1*x2*z3-x1*z2*y3-z1*y2*x3
d1=y2*z3+y1*z2+z1*y3-y1*z3-z2*y3-z1*y2
d2=x1*z3+z2*x3+z1*x2-x2*z3-x1*z2-z1*x3
d3=x1*y2+y1*x3+x2*y3-y1*x2-x1*y3-y2*x3
a=-d1/d
b=-d2/d
c=-d3/d
end subroutine plane

SUBROUTINE ENMOBHY(EU,X,Y,Z,DX,DY,DZ,QECONT,ECONT,NATOMX)
  !     Started life as USERE
  !     EU : Emobhy + Voltage energy
  !     X,Y,Z - Coordinates for energy evaluation
  !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
  !     QECONT - Flag for analysis (0-no analysis,>0=analysis)
  !     ECONT(NATOMX) - Analysis array to be filled if QECONT>0.
  !     NATOMX - Number of atoms
  !
  use chm_kinds
  use psf
  implicit none
  real(chm_real) EU,EVOLT,tmp,tmpe,psi,dpsi
  INTEGER NATOMX
  real(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
  real(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
  LOGICAL QECONT
  real(chm_real) ECONT(NATOMX)
  !
  INTEGER I
  !
  EU=Emobhy                      
  IF (LVOLT) THEN
    EVOLT=0.d0
    do i=1,natomx
      IF (VSLCT(i).ne.1) cycle
      IF (Z(I).LE.-WIDTH/2) THEN
         TMP=Z(I)+WIDTH/2
         TMPE=EXP(GKAPPA*TMP)
         PSI=ACONS*TMPE*23.06
         DPSI=PSI*GKAPPA
      ELSE IF (Z(I).GE.WIDTH/2) THEN
         TMP=Z(I)-WIDTH/2
         TMPE=EXP(-GKAPPA*TMP)
         PSI=(VOLT-ACONS*TMPE)*23.06
         DPSI=GKAPPA*ACONS*TMPE*23.06
      ELSE
         TMP=Z(I)+WIDTH/2
         PSI=ACONS*(40.*GKAPPA*TMP+1.0)*23.06
         DPSI=GKAPPA*ACONS*40.0*23.06
      ENDIF
      EVOLT=EVOLT+PSI*CG(I)
      DZ(I)=DZ(I)+DPSI*CG(I)
    enddo
    EU=EU+EVOLT
  endif
  
  IF(QECONT) THEN
     DO I=1,NATOMX
        ECONT(I)=0.0
     ENDDO
  ENDIF
  RETURN
END SUBROUTINE ENMOBHY

!Is there a better way to do this?  GTRMA is only 8-characters, too few for filenames
  CHARACTER(len=30) FUNCTION GTRMA30(ST,STLEN,KEYWRD)
    !-----------------------------------------------------------------------
    !
    use string
    CHARACTER(len=*) ST
    INTEGER STLEN
    CHARACTER(len=*) KEYWRD
    INTEGER WDLEN
    !
    GTRMA30='        '
    IF(STLEN > 0) THEN
       WDLEN=LEN(GTRMA30)
       CALL GTRMWA(ST,STLEN,KEYWRD,LEN(KEYWRD),SWDTCH,WDLEN,SWDLEN)
       IF (SWDLEN /= 0) GTRMA30=SWDTCH(1:SWDLEN)
    ENDIF
    !
    RETURN
  END FUNCTION GTRMA30


! #endif /*(main_mobhy)*/

end module mobhy_mod
