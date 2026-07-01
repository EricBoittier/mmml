module phmd
#if KEY_PHMD==1 /*phmd_outer*/
  use chm_kinds
  use dimens_fcm
  !  PHMD control parameters
  !  
  real(chm_real),allocatable,dimension(:) :: vpH_Theta,dph_theta, pH_Theta,dpH_Avg, ThetaOld, &
       ParA, ParB, ParK, barr, force, cons, &
       dphold,vphold, &
       residuePKA !!! OMM interface
  real(chm_real),allocatable,dimension(:,:) :: QState1, QState2, &
       VState1, VState2, ParMod, sp_par

  integer,allocatable,dimension(:) :: phmd_sel, sp_grp, GrpList,TitrRes,TitrResN
   
  integer,allocatable,dimension(:,:) :: qcouple

  INTEGER QFile_PHMD,PFile_PHMD, &
       TFile_PHMD,NTitr,NPrint_PHMD, &
       nsolute,nphavg,ncouple,iphfrq

  INTEGER, save :: iupdate,irunph,nsolp 

  LOGICAL QPHMD,PrLam,PrDeriv,QSetLam,QPHIMG,QPHNBD,QPHTRJ,QHYBR,SkpHdr,PrHdr

  REAL(chm_real) :: pH_PHMD,QMass_PHMD,Temp_PHMD,Barrier,BARTAU, &
       BigK,MASS1_PHMD,MASS2_PHMD,MASS3_PHMD, &
       Nose(3),NoseOld(3),NoseV(3),NoseQ(3),KAYTEE,EKINMAX, &
       phbeta, kinold, pkatemp, newph, oldph , &
       RC3,IEPSIN

contains

  subroutine phmd_iniall()
    qphmd=.false.
    return
  end subroutine phmd_iniall

!********************************************************
! Continuous constant pH molecular dynamics
! Author: Michael S. Lee, 2003 
!         Jana Khandogin, 2004 
!*********************************************************
SUBROUTINE StartPHMD(COMLYN, COMLEN)
  !*********************************************************** 
  ! read input parameters
  ! Temp_PHMD is overwritten later by FINALt (simulation temp) 
  ! if they are different
  !***********************************************************

  use gbsw,only: qgbsw, ngbsel  
  use gbmv,only: qgbmv, nmvsel  
  use chm_kinds
  use chm_types
  use stream
  use string
  use dimens_fcm
  use psf
  use number
  use consta
  use coord
  use param
  use inbnd
  use image, only: natim
  use memory
#if KEY_PARALLEL==1
  use parallel   
#endif
  use select

  implicit none
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN,I,J,K,MaxTitr
  real(chm_real) R
  !real(chm_real) NEWPH,OLDPH,PKATEMP  

! JAW; print out lambda values for the last step (MMTSB interfacing)
  IF (INDXA(COMLYN,COMLEN,'LAMBP').GT.0) THEN
     CALL PrintLAMB()
     RETURN
  ENDIF

! J. Wallace - read new pH, old pH, and temperatures for pH exchange to change ParK 
   IF (INDXA(COMLYN,COMLEN,'RESPH').GT.0) THEN
     NEWPH      = GTRMF(COMLYN,COMLEN, 'NEWPH', SEVEN)
     OLDPH      = GTRMF(COMLYN,COMLEN, 'OLDPH', SEVEN)
     PKATEMP       = GTRMF(COMLYN,COMLEN, 'PKATEMP', THRHUN)
     SkpHdr        = (INDXA(COMLYN,COMLEN,'SKPH').GT.0)
     IF (PRNLEV .GE. 5) THEN
#if KEY_PARALLEL==1
       if (mynod == 0) then
          WRITE(OUTU,*)
          WRITE(OUTU,100) ' pH BIAS RESET'
          WRITE(OUTU,100) '--------------------'
          WRITE(OUTU,'(a,f8.2)') ' NEWPH>',NEWPH
          WRITE(OUTU,'(a,f8.2)') ' OLDPH>',OLDPH
          WRITE(OUTU,'(a,f8.2)') ' PKATEMP> ',PKATEMP
          WRITE(OUTU,100) '--------------------'
      endif
#else /**/
      WRITE(OUTU,*)
      WRITE(OUTU,100) ' pH BIAS RESET'
      WRITE(OUTU,100) '--------------------'
      WRITE(OUTU,'(a,f8.2)') ' NEWPH>',NEWPH
      WRITE(OUTU,'(a,f8.2)') ' OLDPH>',OLDPH
      WRITE(OUTU,'(a,f8.2)') ' PKATEMP> ',PKATEMP
      WRITE(OUTU,100) '--------------------'
#endif 
     ENDIF
     CALL ResetParK()
     RETURN
  ENDIF

  iupdate = 0
  irunph = 0
  ncouple = 0
  qphimg = .false.
  qphnbd = .false.

  IF (PRNLEV .GE. 5) THEN
     WRITE(OUTU,*) 
     WRITE(OUTU,100) 'Continuous constant pH molecular dynamics'
     WRITE(OUTU,*)
  endif
  MaxTitr = NRES*2

  ! allow hybrid solvent phmd
  Nsolute = natom
  if (Qgbmv) Nsolute = nmvsel  
  if (Qgbsw) Nsolute = ngbsel  

  IF (PRNLEV .GE. 5) THEN
     if (nsolute .lt. natom) WRITE(OUTU,100) ' PHMD> Hybrid solvent pH md'
  endif

  ! Reset PHMD parameters
  IF (INDXA(COMLYN,COMLEN,'RESE').GT.0) THEN
     IF (QPHMD) THEN
        QPHMD = .FALSE.
        call chmdealloc('phmd.src','StartPHMD','PHMD_sel',NAtom,intg=PHMD_sel)
        call chmdealloc('phmd.src','StartPHMD','QState1',2,NAtom,crl=QState1)
        call chmdealloc('phmd.src','StartPHMD','QState2',2,NAtom,crl=QState2)
        call chmdealloc('phmd.src','StartPHMD','VState1',2,NAtom,crl=VState1)
        call chmdealloc('phmd.src','StartPHMD','VState2',2,NAtom,crl=VState2)
        call chmdealloc('phmd.src','StartPHMD','dpH_Theta',MaxTitr,crl=dpH_Theta)
        call chmdealloc('phmd.src','StartPHMD','dpH_Avg',MaxTitr,crl=dpH_Avg)
        call chmdealloc('phmd.src','StartPHMD','vpH_Theta',MaxTitr,crl=vpH_Theta)
        call chmdealloc('phmd.src','StartPHMD','pH_Theta',MaxTitr,crl=pH_Theta)
        call chmdealloc('phmd.src','StartPHMD','ThetaOld',MaxTitr,crl=ThetaOld)
        call chmdealloc('phmd.src','StartPHMD','ParA',MaxTitr,crl=ParA)
        call chmdealloc('phmd.src','StartPHMD','ParB',MaxTitr,crl=ParB)
        call chmdealloc('phmd.src','StartPHMD','ParMod',6,MaxTitr,crl=ParMod)
        call chmdealloc('phmd.src','StartPHMD','ParK',MaxTitr,crl=ParK)
        call chmdealloc('phmd.src','StartPHMD','SP_PAR',2,MaxTitr,crl=SP_PAR)
        call chmdealloc('phmd.src','StartPHMD','Barr',MaxTitr,crl=Barr)
        call chmdealloc('phmd.src','StartPHMD','Force',MaxTitr,crl=Force)
        call chmdealloc('phmd.src','StartPHMD','Cons',MaxTitr,crl=Cons)
        call chmdealloc('phmd.src','StartPHMD','SP_GRP',MaxTitr,intg=SP_GRP)
        call chmdealloc('phmd.src','StartPHMD','TitrRes',MaxTitr*2,intg=TitrRes)
        call chmdealloc('phmd.src','StartPHMD','TitrResN',NATOM,intg=TitrResN)
        call chmdealloc('phmd.src','StartPHMD','GrpList',NAtom,intg=GrpList)
        call chmdealloc('phmd.src','StartPHMD','dphold',MaxTitr,crl=dphold)
        call chmdealloc('phmd.src','StartPHMD','vphold',MaxTitr,crl=vphold)
        call chmdealloc('phmd.src','StartPHMD','qcouple',2,MaxTitr,intg=qcouple)
        call chmdealloc('phmd.src','StartPHMD','residuePKA',MaxTitr,crl=residuePKA) !!! OMM interface

        IF (PRNLEV .GE. 5) WRITE(OUTU,100)  &
             'All setup for PHMD is cleared'
     ELSE
        IF (PRNLEV .GE. 5) WRITE(OUTU,100)  &
             'Nothing is setup for PHMD'
     ENDIF
     RETURN
  ENDIF ! RES

100 FORMAT(' PHMD> ',a)

  IF (.NOT. QPHMD) QPHMD = .TRUE.
  QSetLam = .FALSE.
  PrLam = .FALSE.
  PrHdr = .FALSE.
  SkpHdr = .FALSE.
  PrDeriv = .FALSE.

  !  Read in parameters
  QFile_PHMD   = GTRMI(COMLYN,COMLEN, 'QFIL',1)
  PFile_PHMD   = GTRMI(COMLYN,COMLEN, 'PAR',2)
  TFile_PHMD   = GTRMI(COMLYN,COMLEN, 'WRI',2)
  QPHTRJ       = ( INDXA(COMLYN,COMLEN,'PHTR') > 0 )
  PH_PHMD      = GTRMF(COMLYN,COMLEN, 'PH',SEVEN)
  QMass_PHMD   = GTRMF(COMLYN,COMLEN, 'MASS',TEN)
  Mass1_PHMD   = GTRMF(COMLYN,COMLEN, 'MA1',THREE)
  Mass2_PHMD   = GTRMF(COMLYN,COMLEN, 'MA2',FIVE)
  Mass3_PHMD   = GTRMF(COMLYN,COMLEN, 'MA3',SEVEN)
  Temp_PHMD    = GTRMF(COMLYN,COMLEN, 'TEMP',ROOMT)
  Barrier      = GTRMF(COMLYN,COMLEN, 'BARR',TWO)
  phbeta       = gtrmf(comlyn,comlen, 'BETA',zero)
  iphfrq       = gtrmi(comlyn,comlen, 'PHFR', 1)
  nphavg       = gtrmi(comlyn,comlen, 'PHAV', 1)
  Ncouple      = GTRMI(COMLYN,COMLEN, 'QCOU',0)
 
  IF(QGBSW.or.QGBMV)THEN
    IF(QGRF)THEN
        CALL WRNDIE(-5,'<PHMD>', &
             'Reaction feild not compatible with GB')
    ENDIF
  ENDIF


  IF(nphavg .gt. iphfrq)then 
     iphfrq = nphavg
     write(OUTU,'(a)') ' PHMD> PHFRQ RESET --- PHAVG greater than PHFRQ'
  ENDIF

  ! Barrier for tautomer interconversion
  BarTau = 2.5
  BarTau       = GTRMF(COMLYN,COMLEN, 'BARTAU',BarTau)
  ! Histidine exclusion barrier
  BigK         = GTRMF(COMLYN,COMLEN, 'BIGK',FIFTY)

  NPrint_PHMD  = GTRMI(COMLYN,COMLEN, 'NPRI',100)
  PrLam        = (INDXA(COMLYN,COMLEN,'LAM').GT.0)
  PrHdr         = (INDXA(COMLYN,COMLEN,'PRTH').GT.0)
  SkpHdr        = (INDXA(COMLYN,COMLEN,'SKPH').GT.0)
  PrDeriv      = (INDXA(COMLYN,COMLEN,'DERI').GT.0)

!J. Wallace - some added output info 
107  format(a,f8.3)
  if(prnlev>2) then
#if KEY_PARALLEL==1
     if(mynod .eq. 0)then
         if(phbeta .gt. zero)then
           write(outu,107) ' PHMD> Langevin thermostat requested (BETA) =', phbeta
         endif
         write(OUTU,'(a,I4)') ' PHMD> Lambda update frequency (PHFRQ) =',iphfrq
         write(OUTU,'(a,I4)') ' PHMD> Force averaging length  (PHAVG) =',nphavg
     endif
#else
     if(phbeta .gt. zero)then
        write(outu,107) ' PHMD> Langevin thermostat requested (BETA) =', phbeta
     endif
     write(OUTU,'(a,I4)') ' PHMD> Lambda update frequency (PHFRQ) =',iphfrq
     write(OUTU,'(a,I4)') ' PHMD> Force averaging length  (PHAVG) =',nphavg
#endif /* parallel */
     if(QPHTRJ)then
#if KEY_PARALLEL==1 
     if(mynod .eq. 0)then
        write(OUTU,'(a)') ' PHMD> -----------Extended CpHMD trajectory output-------------'
        write(OUTU,'(a)') ' PHMD> !WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING!'
        write(OUTU,'(a)') ' PHMD> !!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!'
        write(OUTU,'(a)') ' PHMD> !!!!!!!!!!!!!!!!!WARNING!!!!!!!!WARNING!!!!!!!!!!!!!!!!!'
        write(OUTU,'(a)') ' PHMD> !!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!!'
        write(OUTU,'(a)') ' PHMD> --------------------------------------------------------'
        write(OUTU,'(a)') ' PHMD>          You will have problems with visualization.     '
        write(OUTU,'(a)') ' PHMD> --------------------------------------------------------'
        write(OUTU,'(a)') ' PHMD> !!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!!' 
        write(OUTU,'(a)') ' PHMD> !!!!!!!!!!!!!!!!!WARNING!!!!!!!!WARNING!!!!!!!!!!!!!!!!!'
        write(OUTU,'(a)') ' PHMD> !!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!'
        write(OUTU,'(a)') ' PHMD> !WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING!'
        write(OUTU,'(a)') ' PHMD> -----------Extended CpHMD trajectory output-------------'
     endif
#else /* parallel */
        write(OUTU,'(a)') ' PHMD> -----------Extended CpHMD trajectory output-------------'
        write(OUTU,'(a)') ' PHMD> !WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING!'
        write(OUTU,'(a)') ' PHMD> !!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!'
        write(OUTU,'(a)') ' PHMD> !!!!!!!!!!!!!!!!!WARNING!!!!!!!!WARNING!!!!!!!!!!!!!!!!!'
        write(OUTU,'(a)') ' PHMD> !!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!!'
        write(OUTU,'(a)') ' PHMD> --------------------------------------------------------'
        write(OUTU,'(a)') ' PHMD>          You will have problems with visualization.     '
        write(OUTU,'(a)') ' PHMD> --------------------------------------------------------'
        write(OUTU,'(a)') ' PHMD> !!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!!' 
        write(OUTU,'(a)') ' PHMD> !!!!!!!!!!!!!!!!!WARNING!!!!!!!!WARNING!!!!!!!!!!!!!!!!!'
        write(OUTU,'(a)') ' PHMD> !!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!'
        write(OUTU,'(a)') ' PHMD> !WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING!'
        write(OUTU,'(a)') ' PHMD> -----------Extended CpHMD trajectory output-------------'
#endif /* parallel */
     endif
  endif

!JAW

  call chmalloc('phmd.src','StartPHMD','PHMD_sel',NAtom,intg=PHMD_sel)
  CALL SELCTA(COMLYN,COMLEN,PHMD_SEL,X,Y,Z,WMAIN,.TRUE.)

  ! Allocate heap arrays for PHMD 
  call chmalloc('phmd.src','StartPHMD','QState1',2,NAtom,crl=QState1)
  call chmalloc('phmd.src','StartPHMD','QState2',2,NAtom,crl=QState2)
  call chmalloc('phmd.src','StartPHMD','VState1',2,NAtom,crl=VState1)
  call chmalloc('phmd.src','StartPHMD','VState2',2,NAtom,crl=VState2)
  call chmalloc('phmd.src','StartPHMD','dpH_Theta',MaxTitr,crl=dpH_Theta)
  call chmalloc('phmd.src','StartPHMD','dpH_Avg',MaxTitr,crl=dpH_Avg)
  call chmalloc('phmd.src','StartPHMD','vpH_Theta',MaxTitr,crl=vpH_Theta)
  call chmalloc('phmd.src','StartPHMD','pH_Theta',MaxTitr,crl=pH_Theta)
  call chmalloc('phmd.src','StartPHMD','ThetaOld',MaxTitr,crl=ThetaOld)
  call chmalloc('phmd.src','StartPHMD','ParA',MaxTitr,crl=ParA)
  call chmalloc('phmd.src','StartPHMD','ParB',MaxTitr,crl=ParB)
  call chmalloc('phmd.src','StartPHMD','ParMod',6,MaxTitr,crl=ParMod)
  call chmalloc('phmd.src','StartPHMD','ParK',MaxTitr,crl=ParK)
  call chmalloc('phmd.src','StartPHMD','SP_PAR',2,MaxTitr,crl=SP_PAR)
  call chmalloc('phmd.src','StartPHMD','Barr',MaxTitr,crl=Barr)
  call chmalloc('phmd.src','StartPHMD','Force',MaxTitr,crl=Force)
  call chmalloc('phmd.src','StartPHMD','Cons',MaxTitr,crl=Cons)
  call chmalloc('phmd.src','StartPHMD','SP_GRP',MaxTitr,intg=SP_GRP)
  call chmalloc('phmd.src','StartPHMD','TitrRes',MaxTitr*2,intg=TitrRes)
  call chmalloc('phmd.src','StartPHMD','TitrResN',NATOM,intg=TitrResN)
  call chmalloc('phmd.src','StartPHMD','GrpList',NAtom,intg=GrpList)
  call chmalloc('phmd.src','StartPHMD','dphold',MaxTitr,crl=dphold)
  call chmalloc('phmd.src','StartPHMD','vphold',MaxTitr,crl=vphold)
  call chmalloc('phmd.src','StartPHMD','qcouple',2,MaxTitr,intg=qcouple)
  call chmalloc('phmd.src','StartPHMD','residuePKA',MaxTitr,crl=residuePKA) !!! OMM interface

! J. Wallace - read in coupled residue info
  if(ncouple .gt. 0)then
#if KEY_PARALLEL==1
    IF(mynod == 0)then
       write(OUTU,'(a)') ' PHMD> Charge Coupling Requested'
    endif
     do i=1,ncouple
        qcouple(1,i)      = GTRMI(COMLYN,COMLEN, 'RESI',0)
        qcouple(2,i)      = GTRMI(COMLYN,COMLEN, 'RESC',0)
        if(mynod == 0)then
           write(OUTU,'(a5,i8,a15,a5,i8)') &
                ' RESI', qcouple(1,i), &
                ' COUPLED TO', ' RESC', qcouple(2,i)
        endif
     enddo

#else /* parallel */
     write(OUTU,'(a)') ' PHMD> Charge Coupling Requested'
     do i=1,ncouple
        qcouple(1,i)      = GTRMI(COMLYN,COMLEN, 'RESI',0)
        qcouple(2,i)      = GTRMI(COMLYN,COMLEN, 'RESC',0)
        write(OUTU,'(a5,i8,a15,a5,i8)') &
             ' RESI', qcouple(1,i), &
             ' COUPLED TO', ' RESC', qcouple(2,i)
     enddo

#endif /* parallel */
endif

  CALL LoadPHMD()

  RETURN
END SUBROUTINE StartPHMD

!-----------------------------------------------------------------------
SUBROUTINE PrintLAMB()

  use stream
#if KEY_PARALLEL==1
  use parallel
#endif 

   INTEGER I

#if KEY_PARALLEL==1
   if(mynod == 0) then
      WRITE(OUTU,'(a11)',advance='no') 'LAST LAMB '
      DO I=1,NTitr
         WRITE(OUTU,'(F5.2)',advance='no') DSIN(ph_theta(i))**2
      ENDDO
    endif
#else /**/
    WRITE(OUTU,'(a11)',advance='no') 'LAST LAMB '
    DO I=1,NTitr
       WRITE(OUTU,'(F5.2)',advance='no') DSIN(ph_theta(i))**2
    ENDDO
#endif 

  RETURN
END SUBROUTINE PrintLamb
! -------------------------------------------------------
SUBROUTINE ResetParK()

  use number
  use consta
  use stream

      INTEGER I

      DO I=1,NTitr
         ParK(I)=ParK(I)+LOG(TEN)*KBOLTZ*PKATEMP*(OLDPH-NEWPH)
      ENDDO

      RETURN

END SUBROUTINE ResetParK
!-----------------------------------------------------------------------
Subroutine DoPHMDTest(comlyn, comlen)

  use chm_kinds
  use stream
  use string
  use dimens_fcm
  use psf
  use number
  use coord
  use param
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN,I
  real(chm_real) STEP,SET,frc,POS

  ! Read in parameters

  I            = GTRMI(COMLYN,COMLEN, 'NUM',1)
  STEP         = GTRMF(COMLYN,COMLEN, 'STEP',ZERO)
  SET          = GTRMF(COMLYN,COMLEN, 'SET',ZERO)
  frc          = GTRMF(COMLYN,COMLEN, 'FORC',-HUNDRD)
  POS          = GTRMF(COMLYN,COMLEN, 'POS',ZERO)

  QSetLam = .TRUE.
  IF (frc .LE. -100.0D0) THEN
     CALL TestPHMD(pH_Theta, QState1, &
          QState2, GrpList, &
          SP_GRP, I, STEP, SET)
  ELSE
     CALL SetConsPHMD(Force, Cons, frc, POS, I)
  ENDIF

  RETURN
END Subroutine DoPHMDTest
!---------------------------------------------------------------------

SUBROUTINE LoadPHMD()
  !*********************************************************
  ! QState1(1,1),QState1(2,1): charge for prot tautomers 
  ! QState2(1,1),QState2(2,2): charge for deprot tautomers 
  ! SP_GRP(NTitr) : tautomer number
  ! SP_PAR(1,NTitr) : ParA for conversion
  ! SP_PAR(2,NTitr) : ParB for conversion
  ! GrpList(NATOM): titratable group no.
  !*********************************************************

  use chm_kinds
  use stream
  use dimens_fcm
  use consta
  use psf
  use number
  use coord
  use param
  use param_store, only: set_param
  use reawri
  use image, only: natim
  use ctitla
  use string

  implicit none

  INTEGER, parameter :: MaxTypes=100, MaxCharges=100
  CHARACTER(len=6) RES_NAME(MaxTypes)
  CHARACTER(len=4) ATOM_TYPE(MaxTypes,MaxCharges)
  character(len=10) fmt00
  character(len=20) fmt120
  real(chm_real) CH1(MaxTypes,MaxCharges),CH2(MaxTypes,MaxCharges)
  real(chm_real) RAD1(MaxTypes,MaxCharges),RAD2(MaxTypes,MaxCharges)
  real(chm_real) MODEL_PKA(MAXTYPES),PARA1(MAXTYPES),PARB1(MAXTYPES), &
       PARA2(MAXTYPES),PARB2(MAXTYPES),PAR(6,MAXTYPES), &
       Bar(MAXTYPES)
  INTEGER NUMCH(MaxTypes)
  INTEGER i,j,k,IUNIT,NGT,IRES,NUMFOUND,Ntauto
  real(chm_real) MPKA,MCalc,PA,PB, X1,X2,Qunprot,Qprot

  LOGICAL EOF,MATCH,IfASP,IfGLU,IfLAU,IfHSP,IfTauto,IfCt,IfSOD,IfCLA,IfTIPP,IfTIPU
  INTEGER LLEN
  integer, parameter :: MAXLIN=150
  CHARACTER(len=150) LINE
  EOF=.FALSE.
  Ntauto=0

  ! Step 1: Read in parameter file
  ! *******************************

  ! Read title

  IUNIT = PFile_PHMD

  CALL RDTITL(TITLEB,NTITLB,IUNIT,0)

  ! Read group params in param file 
  NGT = 0
50 CONTINUE

  IF (PRNLEV.GT.5) THEN
     CALL RDCMND(LINE,MAXLIN,LLEN,IUNIT,EOF,.TRUE., &
          .TRUE.,'PHMD>  ')
  ELSE
     CALL RDCMND(LINE,MAXLIN,LLEN,IUNIT,EOF,.TRUE., &
          .FALSE.,'PHMD>  ')    
  ENDIF

  IF (EOF) GOTO 200
  IF(LLEN.EQ.0) GOTO 50 ! go to next group

  CALL TRIME(LINE,LLEN)
  NGT = NGT + 1 ! numbering of groups in the param file

  RES_NAME(NGT)  =  NEXTA6(LINE,LLEN)

  IF (RES_NAME(NGT).EQ.'END') THEN 
     NGT=NGT-1
     GOTO 200 
  ENDIF

  MODEL_PKA(NGT) =  NEXTF(LINE,LLEN)
  PARA1(NGT)     =  NEXTF(LINE,LLEN)
  PARB1(NGT)     =  NEXTF(LINE,LLEN)

  Bar(NGT)      = NEXTF(LINE,LLEN)
  IF (RES_NAME(NGT).EQ.'HSE' .OR. RES_NAME(NGT).EQ.'ASP2' &
       .OR. RES_NAME(NGT).EQ.'GLU2' .OR.  &
            RES_NAME(NGT).EQ.'LAU2' .OR.  &
       (RES_NAME(NGT)(1:2) .EQ. 'CT' .AND.  &
       RES_NAME(NGT)(6:6) .EQ. '2')) THEN
     PARA2(NGT)     = NEXTF(LINE,LLEN) ! tautomer params
     PARB2(NGT)     = NEXTF(LINE,LLEN)

     ! Read one more line of model potential parameters
     IF (PRNLEV .GT. 5) THEN
        CALL RDCMND(LINE,MAXLIN,LLEN,IUNIT,EOF,.TRUE., &
             .TRUE.,'PHMD>  ')
     ELSE
        CALL RDCMND(LINE,MAXLIN,LLEN,IUNIT,EOF,.TRUE., &
             .FALSE.,'PHMD>  ')        
     ENDIF

     DO I=1,6
        PAR(I,NGT) = NEXTF(LINE,LLEN)
     ENDDO
  ELSE
     DO I=1,6
        PAR(I,NGT) = ZERO
     ENDDO
  ENDIF

  ! Read Qstate (prot and unprot charges)
  NCH = 0 ! numbering of atoms in the param file
75 CONTINUE 
  NCH = NCH + 1
  IF (PRNLEV.GT.5) THEN
     CALL RDCMND(LINE,MAXLIN,LLEN,IUNIT,EOF,.TRUE., &
          .TRUE.,'PHMD>    ')
  ELSE
     CALL RDCMND(LINE,MAXLIN,LLEN,IUNIT,EOF,.TRUE., &
          .FALSE.,'PHMD>    ')
  ENDIF

  IF (EOF) THEN 
     NUMCH(NGT)=NCH-1
     GOTO 200
  ENDIF
  IF (LLEN.EQ.0) THEN
     CALL TRIME(LINE,LLEN)
     NUMCH(NGT)=NCH-1 ! total number of atoms in a group
     GOTO 50
  ENDIF

  ATOM_TYPE(NGT,NCH) =  NEXTA4(LINE,LLEN)
  IF (ATOM_TYPE(NGT,NCH).EQ.'END') THEN
     NUMCH(NGT)=NCH-1
     GOTO 200 
  ENDIF
  CH1(NGT,NCH)       =  NEXTF(LINE,LLEN) ! Qstate1(prot)
  CH2(NGT,NCH)       =  NEXTF(LINE,LLEN) ! Qstate2(unprot)

  ! Read Vstate of titratable protons (prot 1 and unprot 0)
  ! This will be used to compute VDW related terms
  IF (LLEN.GT.0) THEN
     RAD1(NGT,NCH) = NEXTF(LINE,LLEN)
     RAD2(NGT,NCH) = NEXTF(LINE,LLEN)
  ELSE
     RAD1(NGT,NCH) = ZERO 
     RAD2(NGT,NCH) = ZERO 
  ENDIF

  GOTO 75 ! loop the param file

200 CONTINUE

  ! Step 2: Assign QState,Par to atoms
  ! ***********************************
  ! First set QState1/2 <- Charge
  ! 
  QState1(1:2,1:NAtom) = ZERO
  QState2(1:2,1:NAtom) = ZERO
  VState1(1:2,1:NAtom) = ZERO  ! vdw state: 1 for protonated H
  VState2(1:2,1:NAtom) = ZERO

  DO I = 1, NAtom
     GrpList(I) = 0         ! titr group number
     QState1(1,I) = CG(I)
     QState1(2,I) = CG(I)
     QState2(1,I) = CG(I)
     QState2(2,I) = CG(I)
  ENDDO

  NTitr = 0
  DO IRES = 1, NRes ! input residues
     DO J = 1, NGT  !  param file residues
        MATCH=(Res(IRES) .EQ. RES_NAME(J))
        IfHSP = (Res(IRES).EQ. 'HSP') 
        IfASP = (Res(IRES).EQ. 'ASP') 
        IfGLU = (Res(IRES).EQ. 'GLU') 
        IfLAU = (Res(IRES).EQ. 'LAU')
        IfSOD = (Res(IRES).EQ. 'SOD')
        IfCLA = (Res(IRES).EQ. 'CLA')
        IfTIPP = (Res(IRES).EQ. 'TIPP')
        IfTIPU = (Res(IRES).EQ. 'TIPU')

        IF (IfHSP) THEN
           IF (RES_NAME(J).EQ.'HSD' .OR. RES_NAME(J).EQ.'HSE') &
                MATCH = .TRUE.
           IF (RES_NAME(J).EQ.'HSD1' .OR. RES_NAME(J).EQ.'HSE1') &
                MATCH = .TRUE.
        ENDIF
        IF (IfASP) THEN
           IF (Res(IRES)(1:3) .EQ. RES_NAME(J)(1:3)) &
                MATCH = .TRUE.
           IF (RES_NAME(J).EQ.'ASP1' .OR. RES_NAME(J).EQ.'ASP2') &
                MATCH = .TRUE.
        ENDIF
        IF (IfGLU) THEN
           IF (RES_NAME(J).EQ.'GLU1' .OR. RES_NAME(J).EQ.'GLU2') &
                MATCH = .TRUE.
        ENDIF
        IF (IfLAU) THEN
           IF (RES_NAME(J).EQ.'LAU1' .OR. RES_NAME(J).EQ.'LAU2') &
                MATCH = .TRUE.
            if(MATCH)THEN
            endif
        ENDIF

        IF ((RES_NAME(J)(1:2) .EQ. 'CT').OR. &
             (RES_NAME(J)(1:2) .EQ. 'NT')) THEN
           MATCH = (RES_NAME(J)(3:5) .EQ. RES(IRES)(1:3))
        ENDIF

        IfCt = .FALSE.
        IF (RES_NAME(J)(1:2) .EQ. 'CT') IfCt = .TRUE.

        IF (MATCH) THEN
           NUMFOUND = 0
           DO I = IBASE(IRES)+1, IBASE(IRES+1) ! loop over atoms
              IF (PHMD_SEL(I) .NE. 0) THEN
                 DO K = 1,NUMCH(J)
                    IF (EQSTWC(ATOM_TYPE(J,K),4,ATYPE(I),4)) THEN
                       NUMFOUND = NUMFOUND + 1 ! atom match
                       TitrResN(I) = ires 
                    ENDIF
                 ENDDO
              ENDIF
           ENDDO

           IF (NUMFOUND .LT. NUMCH(J)) THEN
              IF (PRNLEV .GT. 5)  &
                   WRITE(OUTU,'(a,1X,a,1X,a,1X,I4,1X,I4)') &
                   'insufficient atom match', &
                   RES(IRes), RES_NAME(J),NUMFOUND,NUMCH(J)
           ELSE  ! all atoms match
              NTitr = NTitr + 1
              TitrRes(nTitr) = ires

              IF (PH_PHMD .LE. -98) PH_PHMD = MODEL_PKA(J)

              ParK(NTITR) = LOG(10.0D0) * KBOLTZ * Temp_PHMD * &
                   (MODEL_PKA(J) - pH_PHMD)

              residuePKA(NTITR) = MODEL_PKA(J)  !!! OMM interface

              ParA(NTITR) = PARA1(J)
              ParB(NTITR) = PARB1(J)
              dpH_Theta(NTITR) = ZERO
              dpH_Avg(NTITR) = ZERO
              vpH_Theta(NTITR) = ZERO
              vphold(ntitr) = zero
              dphold(ntitr) = zero
              Force(NTitr) = ZERO
              Cons(NTitr) = ZERO
              kinold = zero
              pH_Theta(NTitr) = PI/TWO/TWO
              Barr(NTitr) = Bar(J)
              DO i=1,6
                 ParMod(i,NTitr) = ZERO
              ENDDO

              IF (Model_PKA(J) .LT. pH_PHMD) THEN 
                 IF (PRNLEV .GE. 5) WRITE(OUTU,101) &
                      ' PHMD> Grp #',NTitr,':',RES_NAME(J),'(', &
                      RES(IRES),IRES,')','pKa=',MODEL_PKA(J), &
                      'Unprot'

              ELSE IF (Model_PKA(J) .GT. pH_PHMD) THEN 
                 IF (PRNLEV .GE. 5) WRITE(OUTU,101) &
                      ' PHMD> Grp #',NTitr,':',RES_NAME(J),'(', &
                      RES(IRES),IRES,')','pKa=',MODEL_PKA(J), &
                      'Prot'

              ELSE          
                 IF (PRNLEV .GE. 5) WRITE(OUTU,101) &
                      ' PHMD> Grp #',NTitr,':',RES_NAME(J),'(', &
                      RES(IRES),IRES,')','pKa=',MODEL_PKA(J), &
                      'Half prot'
              ENDIF

              ! initialize the tautomer interconversion variable x
              SP_GRP(NTitr) = 0
              SP_PAR(1,NTitr) = ZERO 
              SP_PAR(2,NTitr) = ZERO 
              IfTauto = .FALSE.
              IF (J .GT. 1) THEN
                 IF (IfHSP .AND. RES_NAME(J) .EQ. 'HSE') THEN
                    IF (RES_NAME(J-1) .EQ. 'HSD') THEN
                       IfTauto = .TRUE.
                       SP_GRP(NTitr) = 2
                       SP_GRP(NTitr-1) = 1
                    ELSE
                       CALL WRNDIE(-3,' PHMD>','Titrate HSE only')
                    ENDIF
                 ELSEIF (IfHSP .AND. RES_NAME(J) .EQ. 'HSD') THEN
                    IF (RES_NAME(J+1) .NE. 'HSE')  &
                         CALL WRNDIE(-3,' PHMD>','Titrate HSD only')
                 ENDIF
                 ! old HSP model
                 IF (IfHSP .AND. RES_NAME(J) .EQ. 'HSE1') THEN
                    IF (RES_NAME(J-1) .EQ. 'HSD1') THEN
                       SP_GRP(NTitr) = -1
                       SP_GRP(NTitr-1) = 0
                    ELSE
                       CALL WRNDIE(-3,' PHMD>','Titrate HSE1 only')
                    ENDIF
                 ELSEIF (IfHSP .AND. RES_NAME(J) .EQ. 'HSD1') THEN
                    IF (RES_NAME(J+1) .NE. 'HSE1')  &
                         CALL WRNDIE(-3,' PHMD>','Titrate HSD1 only')
                 ENDIF

                 IF (IfASP .AND. RES_NAME(J) .EQ. 'ASP2') THEN
                    IF (RES_NAME(J-1).EQ. 'ASP1') THEN
                       IfTauto = .TRUE.
                       SP_GRP(NTitr) = 4
                       SP_GRP(NTitr-1) = 3
                    ELSE
                       CALL WRNDIE(-3,'<PHMD>','Titrate ASP2 only')
                    ENDIF
                 ELSEIF (IfASP .AND. RES_NAME(J) .EQ. 'ASP1') THEN
                    IF (RES_NAME(J+1) .NE. 'ASP2')  &
                         CALL WRNDIE(-3,'<PHMD>','Titrate ASP1 only')
                 ENDIF

                 IF (IfGLU .AND. RES_NAME(J) .EQ. 'GLU2') THEN 
                    IF (RES_NAME(J-1).EQ. 'GLU1') THEN
                       IfTauto = .TRUE.
                       SP_GRP(NTitr) = 4
                       SP_GRP(NTitr-1) = 3
                    ELSE
                       CALL WRNDIE(-3,'<PHMD>','Titrate GLU2 only')
                    ENDIF
                 ELSEIF (IfGLU .AND. RES_NAME(J) .EQ. 'GLU1') THEN
                    IF (RES_NAME(J+1) .NE. 'GLU2')  &
                         CALL WRNDIE(-3,'<PHMD>','Titrate GLU1 only')
                 ENDIF

                 IF (IfLAU .AND. RES_NAME(J) .EQ. 'LAU2') THEN
                    IF (RES_NAME(J-1).EQ. 'LAU1') THEN
                       IfTauto = .TRUE.
                       SP_GRP(NTitr) = 4
                       SP_GRP(NTitr-1) = 3
                    ELSE
                       CALL WRNDIE(-3,'<PHMD>','Titrate LAU2 only')
                    ENDIF
                 ELSEIF (IfLAU .AND. RES_NAME(J) .EQ. 'LAU1') THEN
                    IF (RES_NAME(J+1) .NE. 'LAU2')  &
                         CALL WRNDIE(-3,'<PHMD>','Titrate LAU1 only')
                 ENDIF

                 IF (IfCt .AND. RES_NAME(J)(6:6) .EQ. '2') THEN
                    IF (RES_NAME(J-1)(6:6).EQ. '1') THEN
                       IfTauto = .TRUE.
                       SP_GRP(NTitr) = 4
                       SP_GRP(NTitr-1) = 3
                    ELSE
                       CALL WRNDIE(-3,'<PHMD>','Titrate CT2 only')
                    ENDIF
                 ELSEIF (IfCt .AND. RES_NAME(J)(6:6) .EQ. '1') THEN
                    IF (RES_NAME(J+1)(6:6) .NE. '2')  &
                         CALL WRNDIE(-3,'<PHMD>','Titrate CT1 only')
                 ENDIF

                 IF (IfTauto) THEN
                    SP_PAR(1,NTitr) = PARA2(J) ! A_DE
                    SP_PAR(2,NTitr) = PARB2(J) ! B_DE
                    DO I = 1,6
                       ParMod(I,NTitr) = PAR(I,J)
                    ENDDO
                    NTauto = NTauto + 1
                 ENDIF

                 IF (IfSOD .AND. RES_NAME(J) .EQ. 'SOD') THEN
                    SP_GRP(NTitr) = -2
                 ENDIF
                 IF (IfCLA .AND. RES_NAME(J) .EQ. 'CLA') THEN
                    SP_GRP(NTitr) = -2
                 ENDIF
                 IF (IfTIPP .AND. RES_NAME(J) .EQ. 'TIPP') THEN
                    SP_GRP(NTitr) = -2
                 ENDIF
                 IF (IfTIPU .AND. RES_NAME(J) .EQ. 'TIPU') THEN
                    SP_GRP(NTitr) = -2
                 ENDIF

               ELSE

                 IF (IfSOD .AND. RES_NAME(J) .EQ. 'SOD') THEN
                    SP_GRP(NTitr) = -2
                 ENDIF
                 IF (IfCLA .AND. RES_NAME(J) .EQ. 'CLA') THEN
                    SP_GRP(NTitr) = -2
                 ENDIF
                 IF (IfTIPP .AND. RES_NAME(J) .EQ. 'TIPP') THEN
                    SP_GRP(NTitr) = -2
                 ENDIF
                 IF (IfTIPU .AND. RES_NAME(J) .EQ. 'TIPU') THEN
                    SP_GRP(NTitr) = -2
                 ENDIF

              ENDIF

              IF (Bar(J) .LE. 0.01D0) THEN
                 IF (IfTauto) THEN
                    Barr(NTitr) = BARTAU 
                 ELSE 
                    Barr(NTitr) = Barrier
                 ENDIF
              ENDIF

              ! Assign  titration group number, charge and vdw states
              ! assign only once for HSP atoms
              DO I = IBASE(IRES)+1, IBASE(IRES+1)
                 DO K = 1,NUMCH(J) 
                    IF (EQSTWC(ATOM_TYPE(J,K),4,ATYPE(I),4)) THEN
                       IF (SP_GRP(NTitr).EQ.2 .OR.  &
                            SP_GRP(NTitr).EQ.4) THEN
                          QState1(2,I) = CH1(J,K)
                          QState2(2,I) = CH2(J,K) 
                       ELSE
                          QState1(1,I) = CH1(J,K)
                          QState2(1,I) = CH2(J,K)

                          QState1(2,I) = QState1(1,I)
                          QState2(2,I) = QState2(1,I)
                          GrpList(I) = NTitr 
                       ENDIF

                       IF (RAD1(J,K) .GT. ZERO) THEN
                          IF (PRNLEV .GE. 5) WRITE(OUTU, &
                               '(a,i5,a,a5,a,a5,i10,a,1x,a4)') &
                               ' PHMD> Grp #',NTitr,':',RES_NAME(J), &
                               '(',RES(IRES),IRES,')',ATYPE(I)
                          IF (SP_GRP(NTitr) .EQ. 2 .OR. &
                               SP_GRP(NTitr).EQ.4) THEN
                             VState1(2,I) = RAD1(J,K)
                             VState2(2,I) = RAD2(J,K)
                          ELSE
                             VState1(1,I) = RAD1(J,K)
                             VState2(1,I) = RAD2(J,K)
                          ENDIF
                       ENDIF
                    ENDIF ! atom match
                 ENDDO      ! k
              ENDDO         ! i

           ENDIF            ! numfound > numch
        ENDIF               ! match
     ENDDO                  ! j
  ENDDO                     ! ires

  IF (PRNLEV .GE. 5) THEN
     WRITE(OUTU,*) 
     WRITE(OUTU,'(a,F7.2)') ' PHMD> simulation pH = ',PH_PHMD
     WRITE(OUTU,'(a,I4)')    ' PHMD> titr grps     = ',NTitr
  endif

101 format(a,i5,a,a5,a,a5,i10,a,1x,a,f6.2,1x,a)

  CALL set_param('NUMPHMD',NTitr)

  !     Assign initial charges for titratable groups
  DO I=1,NAtom
     J = GrpList(I)
     IF (J .ne. 0) THEN
        X1 = DSIN(pH_Theta(J))**2
        IF (SP_GRP(J) .LE. 0) THEN
           X2 = ONE
        ELSE
           X2 = DSIN(pH_Theta(J+1))**2
        ENDIF
        Qprot = X2*QState1(1,I) + (ONE-X2)*QState1(2,I)
        Qunprot = X2*QState2(1,I) + (ONE-X2)*QState2(2,I)
        CG(I) = (ONE-X1)* Qprot + X1 * Qunprot
     ENDIF
  ENDDO

  DO I=1,NTitr 
     IF (PRNLEV .GT. 5) WRITE(OUTU,'(a,i4,2f6.2)')  &
          ' PHMD> lambda,barrier= ',I, DSIN(pH_Theta(NTitr))**2,barr(i)
  ENDDO

  IF (PrLam .and. prnlev .ge. 0 .and.  PrHdr) THEN
     ! Print RES,TITR,PKA INFO
     write(fmt00,'("(a,",i4,"i5)")') ntitr
     write(fmt120,'("(a, ",i4,"(f8.3))")') ntitr
     WRITE(TFile_PHMD,fmt00) '# ititr ',(i,i=1,NTitr)
     WRITE(TFile_PHMD,fmt00) '#  ires ',(TitrRes(i),i=1,NTitr)
     WRITE(TFile_PHMD,fmt00) '# itauto',(SP_GRP(i),i=1,NTitr)
     WRITE(TFile_PHMD,fmt120) '# ParK',(ParK(i),i=1,NTitr)
  ENDIF

  RETURN
END SUBROUTINE LoadPHMD
!-----------------------------------------------------------------------
SUBROUTINE UpdatePHMD(DoTheta, DoVTheta, IStp, CGonly)

  use clcg_mod,only:random !JAW for Langevin
  use chm_kinds
  use stream
  use dimens_fcm
  use consta
  use psf
#if KEY_PARALLEL==1
  use parallel  
#endif
  use number
  use coord
  use param
  use reawri
  use energym
  use rndnum
  use euler
  use exfunc
  implicit none

  INTEGER DoTheta,DoVTheta,cgonly
  INTEGER IStp
  INTEGER I,J,K,L,resi,atom,jcouple,rescouple
  integer ig
!  INTEGER, parameter :: MAXTITR=1000
  real(chm_real) SingleKin
  real(chm_real) X1,X2,KINOLD,Qunprot,Qprot
  real(chm_real) rfd, a, b, frand, pis ! variables for Langevin dynamics
  real(chm_real) NOSENEW(3)
!  real(chm_real) THETANEW(MAXTITR)
  ! Make this an automatic array (cb3)
  real(chm_real) THETANEW(NTitr)
  character(len=10) fmt00
  character(len=20) fmt120
  character(len=20) fmt100, fmt110
  character(len=25) fmt115
  IF (DoTheta .eq. 1 .AND. (.NOT. QSetLam)) THEN
    if(cgonly .eq. 0)then ! J. Wallace - regular dynamics call -> update theta value 
    iupdate = iupdate + 1
    ! J. Wallace - force averaging
    if(nphavg .gt. 1)then
       if (mod(iupdate,nphavg) .eq. 0) then
          do i=1,ntitr
             dph_avg(i) = dph_avg(i) + dph_theta(i)
             dph_theta(i) = dph_avg(i) / nphavg
             dph_avg(i) = ZERO
          enddo
       else
          do i=1,ntitr 
             dph_avg(i) = dph_avg(i) + dph_theta(i)
          enddo
       endif
    endif

       if(mod(iupdate,iphfrq) .eq. 0)then
          if(phbeta .eq. zero) then  ! J. Wallace - Nose` lambda thermostat
          ! Update Nose and charge variables

          DO J=1,100
             X1 = (NOSEQ(2)*NOSEV(2)**2 - KAYTEE)/NOSEQ(3)
             NOSENEW(3) = (TWO*NOSE(3)-NOSEOLD(3)+DELTA**2*X1)
             X2 = NOSENEW(3)-NOSEOLD(3)
           
             X1 = (NOSEQ(1)*NOSEV(1)**2 - KAYTEE)/NOSEQ(2)
             NOSENEW(2) = (TWO*NOSE(2)-NOSEOLD(2)*(ONE-PT25*X2) &
                  +DELTA**2*X1)/(ONE+PT25*X2)
             X2 = NOSENEW(2)-NOSEOLD(2)
             NOSEV(2) = X2/(TWO*DELTA)
           
             X1 = (EPROP(PHKin) - EKINMAX)/NOSEQ(1)
             NOSENEW(1) = (TWO*NOSE(1)-NOSEOLD(1)*(ONE-PT25*X2) &
                  +DELTA**2*X1)/(ONE+PT25*X2)
             X2 = NOSENEW(1)-NOSEOLD(1)
             NOSEV(1) = X2/(TWO*DELTA)
           
             KINOLD = EPROP(PHKin)
             EPROP(PHKin) = ZERO

             DO I=1,NTitr
                THETANEW(I) = (TWO*PH_THETA(I)-THETAOLD(I)*(ONE-PT25*X2) &
                     -DELTA**2*dpH_Theta(I)/QMASS_PHMD)/(ONE+PT25*X2)
              
                vpH_THETA(I) = (THETANEW(I)-THETAOLD(I))/(TWO*DELTA)
                EPROP(PHKin)=EPROP(PHKin) &
                     + HALF*QMASS_PHMD*vpH_Theta(I)**2
             ENDDO
             IF (ABS(KINOLD-EPROP(PHKin)).LT.1D-12) GOTO 50
          ENDDO

50      CONTINUE

             ! J. Wallace - coupled residue titration
             IF(ncouple .gt. 0)then
                do I=1,ntitr
                   IF(SP_GRP(I) .EQ. -2)THEN
                       do k=1,ncouple
                          if(qcouple(2,k) .EQ. TitrRes(I))then
                             rescouple=qcouple(1,k)
                          endif
                       enddo
                       do k=1,ntitr
                          if(rescouple .EQ. TitrRes(k))then
                             jcouple = k
                             exit
                          endif
                       enddo
                       THETANEW(I) = THETANEW(jcouple)
                   ENDIF
                 enddo
              ENDIF 


          DO I=1,NTitr
             THETAOLD(I) = pH_THETA(I) 
             pH_THETA(I) = THETANEW(I) 
          ENDDO

          DO I=1,3
             NOSEOLD(I) = NOSE(I) 
             NOSE(I) = NOSENEW(I) 
          ENDDO

       else  ! J. Wallace - phbeta > 0 ==> Langevin lambda thermostat
          PIS=PI
          IG = 1
          KBT=KBOLTZ*Temp_PHMD
          KINOLD = EPROP(PHKin)
          EPROP(PHKin) = ZERO
          if (phbeta .lt. 100.0D0) then ! regular langevin
            GAM=TIMFAC*PHBETA*DELTA
            RFD=SQRT(TWO*QMASS_PHMD*GAM*KBT)/DELTA
            DO I=1,NTitr
                 A=RFD*SQRT(MINTWO*LOG(RANDOM(IG)))
                 B=TWO*PIS*RANDOM(IG)
                 FRAND=A*DCOS(B)
                 dpH_THETA(I)=dpH_THETA(I)+FRAND
                 vpH_THETA(I)=VPHOLD(I)-(DELTA/TWO)*(dPH_THETA(I)/QMASS_PHMD) &  
                      -(DELTA/TWO)*(dPHOLD(I)/QMASS_PHMD)
           
                 vpH_THETA(I)=(ONE-GAM)*vpH_THETA(I)
                 THETANEW(I) = pH_THETA(I)+vpH_THETA(I)*DELTA &
                     -dPH_THETA(I)*DELTA*DELTA/(TWO*QMASS_PHMD)

                 ! J. Wallace - coupled residue titration
                 IF(ncouple .gt. 0)then
                    IF(SP_GRP(I) .EQ. -2)THEN
                        do k=1,ncouple
                           if(qcouple(2,k) .EQ. TitrRes(I))then
                              rescouple=qcouple(1,k)
                           endif
                        enddo
                        do k=1,ntitr
                           if(rescouple .EQ. TitrRes(k))then
                              jcouple = k
                           exit
                           endif
                        enddo
                       THETANEW(I) = THETANEW(jcouple)
                     ENDIF
                 ENDIF 

                 EPROP(PHKin)=EPROP(PHKin) &
                     + HALF*QMASS_PHMD*vpH_Theta(I)**2

              ENDDO
          else ! J. Wallace - phbeta very big -> diffusive dynamics 
               ! (Gunsteren and Berendsen 1982, Molec. Phys. 45, 637)
            RFD = SQRT(TWO*KBT*DELTA/PHBETA) 
            DO I=1,NTitr
                A=RFD*SQRT(MINTWO*LOG(RANDOM(IG)))
                B=TWO*PIS*RANDOM(IG)
                FRAND=A*DCOS(B)
                THETANEW(I) = PH_THETA(I)-(DELTA/(PHBETA))*DPH_THETA(I) &
                              - (DELTA/(TWO*PHBETA))*(DPH_THETA(I)-DPHOLD(I))+FRAND 


                  ! J. Wallace - coupled residue titration
                  IF(ncouple .gt. 0)then
                     IF(SP_GRP(I) .EQ. -2)THEN
                         do k=1,ncouple
                            if(qcouple(2,k) .EQ. TitrRes(I))then
                               rescouple=qcouple(1,k)
                            endif
                         enddo
                         do k=1,ntitr
                            if(rescouple .EQ. TitrRes(k))then
                               jcouple = k
                              exit
                            endif
                         enddo
                         THETANEW(I) = THETANEW(jcouple)
                     ENDIF
                 ENDIF
 
                VPH_THETA(I) = (THETANEW(I) - PH_THETA(I)) / DELTA 
                EPROP(PHKin)=EPROP(PHKin) &
                     + HALF*QMASS_PHMD*vpH_Theta(I)**2
            ENDDO
          endif
          DO I=1,NTitr
             DPHOLD(I)=dPH_THETA(I)
             VPHOLD(I)=vpH_THETA(I)
             THETAOLD(I) = pH_THETA(I)
             pH_THETA(I) = THETANEW(I)
          ENDDO
       endif  ! Nose' or Langevin
     endif ! iupdate 
   endif ! cgonly 
#if KEY_PARALLEL==1
   call psnd8(pH_Theta,NTitr)
#endif /* parallel */
   DO I=1,NAtom
      J = GrpList(I)
      IF (J .ne. 0) THEN
         X1 = DSIN(pH_Theta(J))**2
         IF (SP_GRP(J) .LE. 0) THEN
            X2 = ONE
         ELSE
            X2 = DSIN(pH_Theta(J+1))**2
         ENDIF
         Qprot = X2*QState1(1,I) + (ONE-X2)*QState1(2,I)
         Qunprot = X2*QState2(1,I) + (ONE-X2)*QState2(2,I)
         CG(I) = (ONE-X1)*Qprot + X1 * Qunprot
      ENDIF
   ENDDO

   cgtot=zero
   do i=1,natom
      cgtot=cgtot + cg(i)
   enddo

ENDIF ! Dotheta

if(cgonly .eq. 1)then
! Update charges
#if KEY_PARALLEL==1
   call psnd8(pH_Theta,NTitr)
#endif /* parallel */
   DO I=1,NAtom
      J = GrpList(I)
      IF (J .ne. 0) THEN
         X1 = DSIN(pH_Theta(J))**2
         IF (SP_GRP(J) .LE. 0) THEN
            X2 = ONE
         ELSE
            X2 = DSIN(pH_Theta(J+1))**2
         ENDIF
         Qprot = X2*QState1(1,I) + (ONE-X2)*QState1(2,I)
         Qunprot = X2*QState2(1,I) + (ONE-X2)*QState2(2,I)
         CG(I) = (ONE-X1)*Qprot + X1 * Qunprot
      ENDIF
   ENDDO

   cgtot=zero
   do i=1,natom
      cgtot=cgtot + cg(i)
   enddo
endif

IF(CGonly .eq. 0)THEN
  ! J. Wallace - print header
  IF (PrLam .and. prnlev .ge. 0 .and. ISTp .eq. 1 .and. (DoTheta .eq. 1) .and. (.not. SkpHdr)) THEN
     ! Print RES,TITR,PKA INFO
     write(fmt00,'("(a,",i4,"i5)")') ntitr
     write(fmt120,'("(a, ",i4,"(f8.3))")') ntitr
     WRITE(TFile_PHMD,fmt00) '# ititr ',(i,i=1,NTitr)
     WRITE(TFile_PHMD,fmt00) '#  ires ',(TitrRes(i),i=1,NTitr)
     WRITE(TFile_PHMD,fmt00) '# itauto',(SP_GRP(i),i=1,NTitr)
     WRITE(TFile_PHMD,fmt120) '# ParK',(ParK(i),i=1,NTitr)
  ENDIF

  IF ((MOD(IStp,NPrint_PHMD) .EQ. 0).AND.(DoTheta .eq. 1)) THEN
     ! Make correct format statements
     write(fmt100,'("(i8, ",i4,"(f5.2))")') ntitr
     write(fmt110,'("(i8, ",i4,"(f10.4))")') ntitr
     write(fmt115,'("(i8, ",i4,"(f10.4,f10.4))")') ntitr
     IF (PrDeriv .AND. PRNLEV .GT. 2) THEN
        WRITE(TFile_PHMD,fmt115) IStp, (pH_Theta(i), &
             dpH_Theta(i),I=1,NTitr)
     ELSEIF (PrLam .AND. PRNLEV .GT. 2) THEN
        WRITE(TFile_PHMD,fmt100) IStp,(DSIN(pH_Theta(i))**2,I=1,NTitr)
     ENDIF
  ENDIF
ENDIF

  RETURN
END SUBROUTINE UpdatePHMD


!-----------------------------------------------------------------------
Subroutine SetConsPHMD(Force,Cons,ForceValue,ConsValue,Num)

  use chm_kinds
  use stream
  use dimens_fcm
  use psf
  use number
  use coord
  use param
  implicit none

  real(chm_real) Force(*),Cons(*),ForceValue,ConsValue 
  INTEGER Num

  IF (PRNLEV .GT. 5) WRITE(outu,1011)  &
       ForceValue,ConsValue,Num

1011 FORMAT('Applying a constraint of ',F10.5, &
       'at ',F10.5,'on theta #',I6)

  Force(Num) = ForceValue 
  Cons(Num) = ConsValue 

  RETURN
END Subroutine SetConsPHMD


!-----------------------------------------------------------------------
Subroutine TestPHMD(pH_Theta,QState1,QState2,GrpList, &
     SP_GRP,ThetaNum,StepSize,SetValue)

  use chm_kinds
  use stream
  use dimens_fcm
  use psf
  use number
  use consta
  use coord
  use param
  implicit none

  real(chm_real) pH_Theta(*),QState1(2,*),QState2(2,*)
  INTEGER GrpList(*),SP_GRP(*)
  INTEGER I,J,K,ThetaNum,ThetaNumNew,dJ
  real(chm_real) StepSize,SetValue,X1,X2,Qunprot,Qprot

  QSetLam = .TRUE.

  IF (ABS(SetValue-1.57D0) .LE. 0.1D0) THEN
     SetValue = PI/TWO
  ELSEIF (ABS(SetValue-0.78D0) .LE. 0.1D0) THEN
     SetValue = PI/TWO/TWO
  ENDIF

  pH_Theta(ThetaNum) = SetValue

  IF (PRNLEV .GE. 5)  WRITE(OUTU,'(a,I3,a,F8.3,a,F8.3)')  &
       ' PHMD > set Theta ',ThetaNum,' to ', &
       SetValue, ' or lambda to ', DSIN(SetValue)**2

  IF (DABS(StepSize) .GT. ZERO) THEN
     if (prnlev .ge. 5)  WRITE(OUTU,'(a,I3,a,F8.3,a,F8.3)')  &
          'Shifting Theta ',ThetaNum,' by ', &
          StepSize, ' or lambda to ', DSIN(StepSize)**2
     pH_Theta(ThetaNum) = pH_Theta(ThetaNum) + StepSize
  ENDIF

  DO K=1,NAtom
     J = GrpList(K)
     dJ = ThetaNum - J
     ThetaNumNew = ThetaNum
     IF (J .NE. 0) THEN
        IF (SP_GRP(J) .GT. 0) THEN
           dJ = ThetaNum - J
           IF (dJ .EQ.1) ThetaNumNew = ThetaNum -1
        ENDIF

        IF (J .EQ. ThetaNumNew) THEN
           X1 = DSIN(pH_Theta(J))**2
           IF (SP_GRP(J) .LE. 0) THEN
              X2 = ONE
           ELSE
              X2 = DSIN(pH_Theta(J+1))**2
           ENDIF
           Qprot = X2*QState1(1,K) + (ONE-X2)*QState1(2,K)
           Qunprot = X2*QState2(1,K) + (ONE-X2)*QState2(2,K)
           CG(K) = (ONE-X1)*Qprot + X1 * Qunprot
        ENDIF
     ENDIF
  ENDDO
  RETURN
END Subroutine TestPHMD


!-----------------------------------------------------------------------
SUBROUTINE RunPHMD(pH_Energy, &
     JNbl,INbl,Inbl14,INblo14, &
     CCNBA,CCNBB,IACNB,NITCC2,LOWTP, &
     DX,DY,DZ, &
     natim, ntrans,imtrns,iminv,NOROT, &
     IMATPT,IMATTR,IMJNBL,IMINBL, &
     NIMNBSL,IMJNBSL,IMINBSL)
  !***************************************************************
  ! Calculates energy and derivatives for PHMD
  ! Charges on titratable group are functions of lambda
  ! ParA(NTitr) : First parameter in the 1-d model potential
  ! ParB(NTitr) : Second parameter in the 1-d model potential
  ! R1,R2,R3,R4,R5,R6 are the parameters in the 2-d model potential
  ! R1-R4 are direct copies of ParMod(1,ititr),...,ParMod(4,ititr)
  ! R5,R6 are obtained from ParMod(i,ititr)
  ! Notice the negative sign in front of R4 in the model potential
  ! QState1 : Charges of system in state 1 
  ! QState2 : Charges of system in state 2 
  ! pH_Energy : Energy of pH constraints
  ! VDW_PHMD : Correction for the VDW energy
  ! pH_Theta : States of titratable groups
  ! dpH_Theta : Derivatives
  ! JNbl,etc. : Excluded and non-bonded list
  ! CCNB(A/B) : VDW energy parameters
  ! IACNB,NITCC2,LOWTP : VDW index
  ! DX,DY,DZ: force on spatial coordinates
  ! ISOLP(NSOLUTE*NSOLUTE,3): temporary array to save atom inices (real and otherwise)
  !                           for image list pre-processing
  !***************************************************************
  use gbsw, only: ngbsel,qgbsw,rborn,epsw,epsp,kappa
  use gbmv, only: nmvsel,qgbmv,kappa_gb
  use gb_common, only: alph_gb,eps_gb

  ! PME_PHMD
  use erfcd_mod,only: erfcd                  ! real space contribution 
  use pme_module,only: QPME,do_pme_phmd      ! k-sapce contribution
  use ewald_1m,only: erfmod,pmekappa=>kappa  ! kappa control the convergency of pme
  ! END PME_PHMD                             ! self & kspace -- Y Huang 2016


  use chm_kinds
  use stream
  use dimens_fcm
  use param
  use psf
  use number
  use energym
  use coord
  use coordc
  use consta
  use inbnd
  use memory
  use prssre,only:getvol
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none
  INTEGER iHTaut,iGTaut
  ! GB
  INTEGER JNbl(:), INbl(:), Inbl14(:), INblo14(:)
  INTEGER IACNB(:),NITCC2,LOWTP(:)
  INTEGER i,j,k,l,PS,ii,ctr,k2,IVect,IACI,rescouple,jcouple
  INTEGER ITemp, NPr, jpr,G,H
  INTEGER aa,bb,jj
  ! image atoms
  INTEGER itrans,jtrans,istrt,iend,ipt,itemp0,cc
  INTEGER ntrans,iminv(:),imatpt(:),imattr(:),ImJNBL(:),ImINBL(:)
  INTEGER natim, NImNBSL,ImJNBSL(:),ImINBSL(:)
! PME
! ewald elec exclusion -- Y Huang 2016
  INTEGER NXI, NXIMAX   

  real(chm_real) RE,RG,RI,RJ,R1,R2,R3,R4,R5,R6,RR
  real(chm_real) Factor,Salt,FGB,C3,C4,X1,X2
  real(chm_real) XDIFF,YDIFF,ZDIFF,Rx,Ry,Rz,Impulse
  real(chm_real) pH_Energy,A,Q,D,C1,C2,S,Q1
  real(chm_real) CCNBA(:),CCNBB(:)
  real(chm_real) TR2,TR6,CA,ENN,ENEVDW,TFVDW,RADA,RADB, &
       RADAB,RADH,RADG
  real(chm_real) LambdaH, LambdaG, Lambda, LambdaH2, LambdaG2
  real(chm_real) Qunprot,Qprot,QG,QH,QXH,QXG,Fact,XH,XG
  real(chm_real) Umod,UpH,Ubarr,Uconv,dUmod,dUpH,dUconv,dUbarr
  real(chm_real) dXH,dXG,FactH,FactG
  real(chm_real) VDW_PHMD
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) tmpx,tmpy,tmpz
! PME
  real(chm_real) erfcx,drfc
  real(chm_real) efac   
  real(chm_real) pmevolume, pmenetch 
! erfcx: real space of ewald erfc(beta*r)/r (beta decides convergency of ewald)
! drfc: first derivative of erfcx (useless here) -- Y Huang 2016
! efac: ewald ele exclusion -- Y Huang 2016
! pmevolume: volume of the periodic system
! pmenetch: net charge of the system
! ENDPME  

  ! GB
  real(chm_real)  C2OfNb, C2OnNb, Rul3, Rul12, RijL, RijU, FSw, DFSw
  real(chm_real)  tau,effsalt,scrfac,kappagb,eps_w,eps_p,pmebeta
  real(chm_real)  cga,cgb,rBorna,rBornb
  real(chm_real)  xa,ya,za,xb,yb,zb
  real(chm_real)  dxab,dyab,dzab,tx,ty,tz
  real(chm_real)  r2ab,r2born,expfac,rGB,rGB2,rab
  real(chm_real)  GBdxaa,GBdyaa,GBdzaa
  ! RXNF
  real(chm_real) rc,rc3,iepsi
  ! image atoms
  real(chm_real)  imtrns(:)
  real(chm_real)  alpha(Nsolute)
  integer, dimension(:,:), allocatable, save :: isolp

  LOGICAL LHTitr, LGTitr, LHTauto, LGTauto, LParMod
  LOGICAL LOuter, CSWIT, CSHIFT, CSHFT, CTRUNC, DSWIT
 ! image atoms
  LOGICAL NOROT
 ! hybrid 

  integer nstrt,nskip

 IF(QGRF)THEN
   rc = 1 / ctofnb
   rc3 = rc / (ctofnb*ctofnb)
   iepsi = 1 / eps
 ENDIF

  qhybr = .false.

! PME     Y Huang 2016
 IF(QPME .and. (.not. QGBSW) .and. (.not. QGBMV))THEN
    pmebeta = pmekappa*ccelec/sqrt(pi)  
! pmekappa is (1/r) cutoff of pme -- Yandong 07/22/2015
    CALL GETVOL(pmevolume)     
! calculate volume for net charge correction
 ELSE
    pmebeta = ZERO
 ENDIF


  IF (QGBSW) THEN
      qhybr = (natom .ne. ngbsel)
      kappagb=kappa
      eps_w = epsw
      eps_p = epsp
      Alpha(1:NSolute)=rborn(1:NSolute)
  ELSE IF (QGBMV) THEN
      qhybr = ( natom .ne. nmvsel )
      kappagb=kappa_gb
      eps_w = eps_gb
      eps_p = ONE
      Alpha(1:NSolute) = alph_gb(1:NSolute)
  ENDIF

  ! J. Wallace
  if((QGBSW.or.QGBMV).and.QHYBR)THEN
      if ( .not. allocated(isolp) ) then
         call  chmalloc('phmd.src','RunPHMD','ISOLP',Nsolute*Nsolute,3,intg=ISOLP)
      else
         if ( Nsolute*Nsolute >= size(isolp) ) then
            call chmdealloc('phmd.src','RunPHMD','ISOLP',Nsolute*Nsolute,3,intg=ISOLP)
            call chmalloc('phmd.src','RunPHMD','ISOLP',Nsolute*Nsolute,3,intg=ISOLP)
         endif
      endif
  ENDIF

  ! Compute energy associated with pH Function
  !********************************************
  ! initialize

  pH_Energy = ZERO
  VDW_PHMD = ZERO
  dPH_Theta(1:NTitr) = ZERO

  ! Model,PH and Barrier potentials
  ! Note: input A is now negative (not -A anymore)
  ! J. Wallace - For parallel, phmd only calculate pH energy and forces for node 0

  model_ene: DO I=1,NTitr !JAW
#if KEY_PARALLEL==1
     IF (mynod > 0) then
        exit model_ene
     ENDIF
#endif /* parallel */

     IF (ParMod(1,i).NE. ZERO .AND. ParMod(2,i).NE. ZERO) THEN
        LParMod = .TRUE.
     ENDIF

     IF (SP_GRP(I) .EQ. 0) THEN ! single site
        Lambda = DSIN(pH_Theta(I))**2
        Ubarr = FOUR*Barr(i)*(Lambda-HALF)**2
        dUbarr = FOUR*TWO*Barr(i)*(Lambda-HALF)
        UpH = ParK(I)*Lambda
        dUpH = ParK(i)
        Umod = ParA(i)*(Lambda-ParB(i))**2
        dUmod = TWO*ParA(i)*(Lambda-ParB(i))
        dpH_Theta(I) = dpH_theta(I) -dUmod - dUbarr + dUpH
     ELSEIF (SP_GRP(I) .EQ. -1) THEN ! old model: histidine exclusion 
        X2 = ONE
        X1 = DSIN(pH_Theta(I-1))**2
        Umod = ParA(i)*(Lambda-ParB(i))**2 + BigK*Lambda*X1
        dUmod = TWO*ParA(i)*(Lambda-ParB(i))
        dpH_Theta(I) = dpH_THETA(I) -dUmod - dUbarr + BigK*X1 !+ dUpH
        dpH_Theta(I-1) = dpH_Theta(I-1) + BigK*Lambda
     ELSEIF (SP_GRP(I) .EQ. 1 .OR. SP_GRP(I) .EQ. 3) THEN
        Umod = zero
        Ubarr = zero
        UpH = zero
     ELSEIF (SP_GRP(I) .EQ. 2) THEN   ! double site pKa1 .ne. pka2
        Lambda = DSIN(pH_Theta(I-1))**2 
        X1 = DSIN(pH_Theta(I))**2
        X2 = ONE - X1
        R2 = -TWO*ParA(i)*ParB(i)
        R1 = -TWO*ParA(i-1)*ParB(i-1) -R2
        R3 = -TWO*SP_PAR(1,i)*SP_PAR(2,i) -R1
        ! cb3 - I DONT UNDERSTAND THE DIFFERENCES HERE WHAT ARE THEY?
        Ubarr = FOUR*Barr(i)*(X1-HALF)**2 + FOUR*Barr(I-1)*(Lambda-HALF)**2     
        UpH = Lambda*(ParK(I-1)*X1 + ParK(I)*X2) ! Park(I-1) for x=1 and Park(I) for x=0 -- Wei Chen
        Umod = SP_PAR(1,i)*Lambda**2*X1**2 +R1*Lambda*X1 &
             + R2*Lambda + R3*Lambda**2*X1 +ParA(i)*Lambda**2

        ! dU/dLamb 
        !Barr
        dUbarr = TWO*FOUR*Barr(I-1)*(Lambda-HALF)              
        !pH 
        dUpH = ParK(I-1)*X1 + ParK(I)*X2 ! Park(I-1) for x=1 and Park(I) for x=0 -- Wei Chen
        !mod 
        dUmod = TWO*SP_PAR(1,i)*Lambda*X1**2 +R1*X1 &
             + R2 + TWO*R3*Lambda*X1 +TWO*ParA(i)*Lambda

        dpH_Theta(I-1) = dph_THETA(I-1)-dUmod - dUbarr + dUpH

        ! dU/dx
        !Barr 
        dUbarr = TWO*FOUR*Barr(I)*(X1-HALF)
        !pH
        dUpH = (ParK(I-1)-ParK(I))*Lambda ! Park(I-1) for x=1 and Park(I) for x=0 -- Wei Chen
        !mod
        dUmod = TWO*SP_PAR(1,i)*Lambda**2*X1 +R1*Lambda &
             + R3*Lambda**2

        dpH_Theta(I)= dph_theta(I) - dUmod - dUbarr + dUpH

     ELSEIF (SP_GRP(I) .EQ. 4) THEN  ! double site pKa1 .eq. pka2  
        Lambda = DSIN(pH_Theta(I-1))**2 
        X1 = DSIN(pH_Theta(I))**2
        X2 = ONE -X1

        IF (LParMod) THEN
           R1 = ParMod(1,I)
           R2 = ParMod(2,I)
           R3 = ParMod(3,I)
           R4 = ParMod(4,I)
           R5 = ParMod(5,I) -R1*R4**2
           R6 = -TWO*ParMod(5,I)*ParMod(6,I) -R2*R4**2
        ELSE
           RR = ONE - TWO*SP_PAR(2,i)
           R1 = (ParA(i-1) - ParA(i))/RR
           R2 = TWO*(ParA(i)*ParB(i)-ParA(i-1)*ParB(i-1))/RR
           R3 = SP_PAR(1,i)
           R4 = SP_PAR(2,i)
           R5 = ParA(i) - R1*R4**2
           R6 = -TWO*ParA(i)*ParB(i)-R2*R4**2
        ENDIF

        Ubarr = FOUR*Barr(i)*(X1-HALF)**2 + FOUR*Barr(I-1)*(Lambda-HALF)**2     
        UpH = Lambda*ParK(i)
        Umod = (R1*Lambda**2 + R2*Lambda + R3)*(X1-R4)**2 &
             + R5*Lambda**2 + R6*Lambda

        !dU/dLamb
        !Barr
        dUbarr = FOUR*TWO*Barr(i-1)*(Lambda-HALF)
        !pH
        dUpH = Park(i-1)
        !mod
        dUmod = (TWO*R1*Lambda + R2)*(X1-R4)**2 + TWO*R5*Lambda + R6 
        dpH_Theta(I-1) = dpH_theta(I-1) -dUmod - dUbarr + dUpH
        
        !dU/dx
        !Barr
        dUbarr = FOUR*TWO*Barr(i)*(X1-HALF)
        !pH
        dUpH = zero
        !mod
        dUmod = TWO*(R1*Lambda**2 + R2*Lambda + R3)*(X1-R4)
        dpH_Theta(I) = dph_theta(I) - dUmod - dUbarr + dUpH
!     ELSEIF(SP_GRP(I) .EQ. -2) THEN !SOD, CLA, TIPP, TIPU  ! Comment out coions, Y Huang 2016
!         Lambda = DSIN(pH_Theta(I))**2
!         UpH = zero
!         dUpH = zero
!         Ubarr = FOUR*Barr(i)*(Lambda-HALF)**2
!         dUbarr = FOUR*TWO*Barr(i)*(Lambda-HALF)
!         Umod = ParA(i)*(Lambda-ParB(i))**2
!         dUmod = TWO*ParA(i)*(Lambda-ParB(i))
!         dpH_Theta(i) = dpH_theta(i) -dUmod -dUbarr + dUpH
     ENDIF

     pH_Energy = pH_Energy - Umod - Ubarr + UpH
     
     IF (Force(i) .gt. ZERO) THEN
        dpH_Theta(I) = dpH_Theta(I) + Force(i)  &
             * (pH_Theta(i) - Cons(i))
     ENDIF
   ENDDO model_ene ! NTitr

  IF (QSetLam) dpH_Theta(1:Ntitr) = zero

  !  Compute the following energy derivatives wrt lambda
  !  1. Coulomb energy and electrostatic solvation energy
  !  2. VDW energy
  !*******************************************************

  ITemp = 0
 
  IF(QGBSW.or.QGBMV)THEN
     tau   = ( 1/eps_p-1/eps_w ) * ccelec
  ENDIF    

  ! Set flags for electrostatics options -- Wei Chen, 07/06/2013
  CSWIT  = LCONS .AND. .NOT. LSHFT .AND. .NOT. LFSWT .AND. .NOT. LTRUNC .AND. .NOT. LEGROM    ! cdie switch
  CSHIFT = LCONS .AND. LSHFT .AND. LFSWT .AND. .NOT. LTRUNC .AND. .NOT. LEGROM                ! cdie force shift
  CSHFT  = LCONS .AND. LSHFT .AND. .NOT. LFSWT .AND. .NOT. LTRUNC .AND. .NOT. LEGROM          ! cdie shift
  CTRUNC = LCONS .AND. .NOT. LSHFT .AND. .NOT. LFSWT .AND. LTRUNC .AND. .NOT. LEGROM          ! cdie truncate
  !For GB, only switch is allowed
  IF((QGBSW .or. QGBMV) .and. .not. CSWIT )THEN
     CSWIT = .TRUE.
     CSHIFT = .FALSE.
     CSHFT = .FALSE.
     CTRUNC = .FALSE.
  ENDIF

  ! Set flags for van der Waals options -- Wei Chen, 07/31/2013
  !DSWIT = .NOT. LVFSWT .AND. .NOT. LVSHFT .AND. .NOT. LVGROM .AND. .NOT. LVTRUNC              ! vdw distance switch
  DSWIT = .FALSE.      ! turn off switch for VDW for lambda to avoid SHAKE errors

  ! 1.1 excluded atom list
  ! **********************

  nstrt=1
  nskip=1
#if KEY_PARALLEL==1
  nstrt=mynodp
  nskip=numnod
#endif 

  IF (QGBSW .OR. QGBMV) THEN
  do aa=nstrt,nsolute,nskip
     If (aa .ne. 1) ITemp = INblo14(aa - 1)
     NPr = INblo14(aa) - ITemp

     cga=cg(aa)
     H = GrpList(aa)

     if(cga.ne.ZERO) then
        xa=x(aa)
        ya=y(aa)
        za=z(aa)
        rBorna=Alpha(aa) 

        do jpr = 1, NPr
           k = Inbl14(Itemp+jpr)
           bb = Abs(k)
           G = GrpList(bb) 

           IF ((H .gt. 0).OR.(G .gt. 0)) THEN 
              cgb= cg(bb)
              if(cgb.ne.ZERO) then

                 ! Setup some variables
                 IF (H.GT.0) THEN
                    IF (SP_GRP(H) .LE. 0) THEN
                       X2 = ONE
                    ELSE
                       X2 = DSIN(pH_Theta(H+1))**2
                       Lambda = DSIN(pH_Theta(H))**2
                       Qunprot = Lambda*(QState2(1,aa)-QState2(2,aa))
                       Qprot = (1-Lambda)*(QState1(1,aa)-QState1(2,aa))
                       QXH = cgb*(Qunprot+Qprot)
                    ENDIF
                    Qunprot = X2*QState2(1,aa) +(1-X2)*QState2(2,aa)
                    Qprot = X2*QState1(1,aa) + (1-X2)*QState1(2,aa)
                    QH = cgb*(Qunprot - Qprot)
                 ENDIF

                 IF (G.GT.0) THEN
                    IF (SP_GRP(G) .LE. 0) THEN
                       X2 = ONE
                    ELSE
                       X2 = DSIN(pH_Theta(G+1))**2
                       Lambda = DSIN(pH_Theta(G))**2
                       Qunprot = Lambda*(QState2(1,bb)-QState2(2,bb))
                       Qprot =(1-Lambda)*(QState1(1,bb)-QState1(2,bb))
                       QXG = cga*(Qunprot + Qprot)
                    ENDIF
                    Qunprot = X2*QState2(1,bb) + (1-X2)*QState2(2,bb)
                    Qprot = X2*QState1(1,bb) + (1-X2)*QState1(2,bb)
                    QG = cga*(Qunprot - Qprot)
                 ENDIF

                 xb=x(bb)
                 yb=y(bb)
                 zb=z(bb)
                 rBornb=Alpha(bb)
                 dxab=xb-xa
                 dyab=yb-ya
                 dzab=zb-za
                 r2ab=dxab*dxab+dyab*dyab+dzab*dzab
                 r2born=rBorna*rBornb

                 IF (k .GT. 0) THEN
                       expfac=exp(-r2ab/(4.0D0*r2born))
                       rGB2=r2ab+r2born*expfac
                       rGB=sqrt(rGB2)

                       IF (kappagb.GT.ZERO) THEN
                          scrfac=1.00/kappagb
                          effsalt=exp(-rGB*scrfac)/eps_w
                          tau=( 1/eps_p-effsalt ) * ccelec
                       ENDIF

                       C2 = -tau/rGB

                       IF (G.GT.0) THEN

                          dpH_Theta(G) = dpH_Theta(G) + C2*QG

                          IF (SP_GRP(G) .GT. 0) THEN
                             dpH_Theta(G+1) = dpH_Theta(G+1) + C2*QXG
                          ENDIF
                       ENDIF
                       IF (H.GT.0) THEN

                          dpH_Theta(H) = dpH_Theta(H) + C2*QH

                          IF (SP_GRP(H) .GT. 0) THEN
                             dpH_Theta(H+1) = dpH_Theta(H+1) + C2*QXH
                          ENDIF
                       ENDIF
                 ELSE       ! k<=0
                    R1 = SQRT(ONE/r2ab) * CCELEC
                    R1 = R1 * (E14Fac - ONE)

                    IF (G.GT.0) THEN
                       dpH_Theta(G) = dpH_Theta(G) + R1*QG

                       IF (SP_GRP(G) .GT. 0) THEN
                          dpH_Theta(G+1) = dpH_Theta(G+1)+R1*QXG
                       ENDIF
                    ENDIF
                    IF (H.GT.0) THEN
                       dpH_Theta(H) = dpH_Theta(H) + R1*QH

                       IF (SP_GRP(H) .GT. 0) THEN
                          dpH_Theta(H+1) = dpH_Theta(H+1)+R1*QXH
                       ENDIF
                    ENDIF
                 ENDIF      ! k>0
              ENDIF         ! cgb/=0
           ENDIF          ! titratable group
        ENDDO ! jpr
     ENDIF ! cga/=0
  ENDDO ! aa
  ENDIF ! GBSW or GBMV 


  ! 1.2 nonbonded atom list
  ! ***********************
  C2OfNB = CtOfNB * CtOfNB
  FSw    = ONE
  DFSw   = ZERO
     IF (CSWIT .or. DSWIT) THEN
        C2OnNb = CtOnNb * CtOnNb
        IF (CtOfNb .GT. CtOnNb) THEN
           Rul3 = ONE / (C2OfNb - C2OnNb)**3
           Rul12= 12.0D0 * Rul3
        ENDIF
     ENDIF

  ! 1.2.1 primary atoms
  ! *******************
!  IF (QGBSW .or. QGBMV .or. QGRF)THEN ! We can truncate electrostatics with GB and Reaction Feild 
! Comment out -- Wei Chen, 07/05/2013

  DO aa=1,nsolute-1
     cga=cg(aa)
     H = GrpList(aa)

     IF (cga.NE.ZERO) THEN  ! VDW even if charge is zero 
        xa=x(aa)
        ya=y(aa)
        za=z(aa)
        rBorna=Alpha(aa)
 
#if KEY_IMCUBES==1
        IF (lbycbim) THEN
           Itemp=INBL(aa+natom)
           Npr=INbl(aa)-ITEMP
        ELSE
#endif 
           IF(aa.GT.1) THEN
              Itemp=INBL(aa-1)
              Npr=INBL(aa)-ITEMP
           ELSE
              Npr=INBL(aa)
              ITEMP=0
           ENDIF
#if KEY_IMCUBES==1
        ENDIF
#endif 
        IACI = IACNB(aa)
        DO jpr = 1, NPr
           k  = JNbl(ITemp+jpr)
           bb = Abs(k)
           if (bb .le. nsolute) then ! J. Wallace - hybrid solvent electrostatic: skip water
           cgb= cg(bb)

           G = GrpList(bb)

           IF ((H .GT. 0).OR.(G .GT. 0)) THEN
              xb=x(bb)
              yb=y(bb)
              zb=z(bb)
              dxab=xb-xa
              dyab=yb-ya
              dzab=zb-za
              r2ab=dxab*dxab+dyab*dyab+dzab*dzab

              IF (r2ab .LT. C2OfNb) THEN

              IF (H .GT. 0) THEN
                 IF (SP_GRP(H) .LE. 0) THEN
                    X2 = ONE
                 ELSE
                    X2 = DSIN(pH_Theta(H+1))**2
                    Lambda = DSIN(pH_Theta(H))**2

                    Qunprot = Lambda*(QState2(1,aa)-QState2(2,aa))
                    Qprot = (1-Lambda)*(QState1(1,aa)-QState1(2,aa))
                    QXH = cgb*(Qunprot+Qprot)
                 ENDIF
                 Qunprot = X2*QState2(1,aa) +(1-X2)*QState2(2,aa)
                 Qprot = X2*QState1(1,aa) + (1-X2)*QState1(2,aa)
                 QH = cgb*(Qunprot - Qprot)

              ENDIF

              IF (G .GT. 0) THEN
                 IF (SP_GRP(G) .LE. 0) THEN
                    X2 = ONE
                 ELSE
                    X2 = DSIN(pH_Theta(G+1))**2
                    Lambda = DSIN(pH_Theta(G))**2
                    Qunprot = Lambda*(QState2(1,bb)-QState2(2,bb))
                    Qprot =(1-Lambda)*(QState1(1,bb)-QState1(2,bb))

                    QXG = cga*(Qunprot + Qprot)
                 ENDIF
                 Qunprot = X2*QState2(1,bb) + (1-X2)*QState2(2,bb)
                 Qprot = X2*QState1(1,bb) + (1-X2)*QState1(2,bb)
                 QG = cga*(Qunprot - Qprot)
              ENDIF

              IF (cgb.NE.ZERO) THEN
                    FSw = ONE
                    DFSw= ZERO
                    IF (CSWIT) THEN
                       LOuter = (r2ab .GT. C2OnNb)

                       IF (LOuter) THEN
                          RijL = C2OnNb - r2ab
                          RijU = C2OfNb - r2ab
                          FSw  = RijU * RijU * (RijU-3.0D0*RijL) * Rul3
                          DfSw = RijL * RijU * Rul12 / FSw ! Why is there FSw in the denominator? -- Wei Chen, 7/9/2013
                       ENDIF
                    ENDIF

                    C2 = ZERO

                    IF (QGBSW .or. QGBMV) THEN
                       rBornb=Alpha(bb)
                       r2born=rBorna*rBornb

                       expfac=exp(-r2ab/(4.0D0*r2born))
                       rGB2=r2ab+r2born*expfac
                       rGB=sqrt(rGB2)

                       IF (kappagb.GT.ZERO) THEN
                          effsalt=exp(-rGB*scrfac)/eps_w
                          tau=( 1/eps_p-effsalt ) * ccelec
                       ENDIF

                       C2 = -FSw*tau/rGB
                    ENDIF

                    IF (QETERM(ELEC)) THEN 
                       IF(QGRF)THEN
                          R1 = iepsi*CCELEC*(1/SQRT(r2ab)+(HALF*RFCON*r2ab*rc3) &
                                                   -((ONE+HALF*RFCON)*rc))
! PME
                       ELSEIF (QPME .and. (.not. QGBSW) .and. (.not. QGBMV)) THEN   
                                         ! real space of pme -- Y Huang 2016
                          rab = sqrt(r2ab)
                          call erfcd(rab,pmekappa,erfcx,drfc,erfmod)
                          R1 = ccelec*erfcx/rab
! ENDPME
                       ELSEIF(CSHIFT)THEN ! force-shifted--Wei Chen, 07/05/2013
                          rab=SQRT(r2ab)
                          R1 = 1/EPS*CCELEC/rab*(1.0 - 2.0*rab/CtOfNb + r2ab/C2OfNb)
                       ELSEIF(CSHFT)THEN ! shift -- Wei Chen, 07/06/2013
                          R1 = 1/EPS*CCELEC/SQRT(r2ab)*(1.0 - r2ab/C2OfNb)**2
                       ELSEIF(CSWIT)THEN ! switch -- Wei Chen, 07/16/2013
                          R1 = 1/EPS*FSw*CCELEC/SQRT(r2ab)
                       ELSE
                          R1 = 1/EPS*CCELEC/SQRT(r2ab)
                       ENDIF
                    ELSE 
                       R1 = 0
                    ENDIF

                    IF (G.GT.0) THEN
                       dpH_Theta(G) = dpH_Theta(G) + QG*(R1+C2)

                       IF (SP_GRP(G) .GT. 0) THEN
                          dpH_Theta(G+1) = dpH_Theta(G+1) +  &
                               QXG*(R1+C2)
                       ENDIF
                    ENDIF

                    IF (H.GT.0) THEN
                       dpH_Theta(H) = dpH_Theta(H) + QH*(R1+C2)
                       IF (SP_GRP(H) .GT. 0) THEN
                          dpH_Theta(H+1) = dpH_Theta(H+1) +  &
                               QXH*(R1+C2)
                       ENDIF
                    ENDIF

                    ENDIF ! cgb/=0
                 ENDIF ! titratable group
              ENDIF ! bb <= nsolute
           ENDIF         ! r2ab
        ENDDO ! jpr
     ENDIF ! cga/=0
  ENDDO ! aa
! ENDIF ! Comment out -- Wei Chen, 07/05/2013

IF ((QGBSW.or.QGBMV).and.QHYBR)THEN ! J. Wallace - hybrid solvent-solute sorting array 
! pre-process images and save the solute indices
! update if either nbonbs or images have been updated
! --------------------------------------------------- 
IF(QPHIMG.or.QPHNBD.or.(IRUNPH .eq. 0))THEN ! update of image, else use old lists
if(irunph .ne. 0)then
#if KEY_PARALLEL==1
  if(mynod .eq. 0)then
     if(prnlev>4) WRITE(OUTU,*) 'PHMD: Image list updated'
  endif
#else /**/
  if(prnlev>4) WRITE(OUTU,*) 'PHMD: Image list updated'
#endif 
endif

  IF (natim.GT.natom) THEN
     nsolp = 0
     itemp0=natom+1
     DO itrans=1,ntrans
        jtrans=iminv(itrans)
        istrt=itemp0
        iend=imatpt(itrans)
        ipt=(itrans-1)*12
        itemp0=iend+1
        DO aa=istrt,iend
           cc=imattr(aa)
           cga=cg(aa)
           H = GrpList(cc)
           IF (cga.NE.ZERO) THEN  
#if KEY_IMCUBES==1
           IF (lbycbim) THEN
               ITEMP=ImINBL(aa+natim)
               Npr=ImINbl(aa)-ITEMP
           ELSE
#endif /*           */
           IF (aa.GT.1) THEN
               ITEMP=ImINBL(aa-1)
               Npr=ImINBL(aa)-ITEMP
           ELSE
               Npr=ImINBL(aa)
               ITEMP=0
           ENDIF
#if KEY_IMCUBES==1
           ENDIF
#endif 
           IACI = IACNB(aa)
           DO jpr = 1, NPr
              k  = ImJNBL(ITemp+jpr)
              bb = Abs(k)
              G = GrpList(bb)
              cgb=cg(bb)
              if(cgb.ne.zero)then
                 if((cc .le. nsolute).and.(bb .le. nsolute))then
                    if((H .ne. 0) .or. (G .ne. 0))then
                       nsolp = nsolp + 1      
                       isolp(nsolp,1)=cc
                       isolp(nsolp,2)=bb
                       isolp(nsolp,3)=aa
                    endif ! H & G
                 endif ! nsolute
              endif ! charge bb
            ENDDO !jpr
          endif ! charge cc
        ENDDO  !aa
     ENDDO !ntrans

  IF (NImNBSL.gt.0) THEN
     itemp0=natom+1
     DO itrans=1,ntrans
        jtrans=iminv(itrans)
        istrt=itemp0
        iend=imatpt(itrans)
        ipt=(itrans-1)*12
        itemp0=iend+1
        DO aa=istrt,iend
           cga=cg(aa)
           cc=imattr(aa)  
           H = GrpList(cc)
           IF(cga.ne.zero)then
#if KEY_IMCUBES==1
           IF (lbycbim) THEN
              ITEMP=ImINBSL(aa+natim) ! natim for image nonbond list
           ELSE
#endif 
           IF (aa.GT.1) THEN
              ITEMP=ImINBSL(aa-1)
           ELSE
              ITEMP=0
           ENDIF
#if KEY_IMCUBES==1
           ENDIF
#endif 
            Npr = ImINBSL(aa) - ITEMP
            IACI = IACNB(aa)
            DO jpr = 1, NPr
               k  = ImJNBSL(ITemp+jpr)
               bb = Abs(k)   
               cgb=cg(bb)
               if(cgb.ne.zero)then
                  if((cc .le. nsolute).or.(bb .le. nsolute))then
                     if((H .ne. 0) .or. (G .ne. 0))then
                        nsolp = nsolp + 1
                        isolp(nsolp,1)=cc
                        isolp(nsolp,2)=bb
                        isolp(nsolp,3)=aa
                    endif ! H & G
                 endif ! nsolute
              endif ! charge bb
            ENDDO ! jpr
           endif ! charge cc
        ENDDO ! aa
     ENDDO ! ntrans
    ENDIF ! self-images
  ENDIF ! images
ENDIF ! update images

  ! 1.2.2 image atoms
  ! *****************
  IF (natim.GT.natom) THEN
     DO i=1,nsolp
        cc =isolp(i,1)
        bb =isolp(i,2)
        aa =isolp(i,3)
        H = GrpList(cc)
        G = GrpList(bb)

        cga = cg(aa)
        cgb = cg(bb)
        IF (H .GT. 0) THEN
           IF (SP_GRP(H) .LE. 0) THEN
              X2 = ONE
           ELSE
              X2 = DSIN(pH_Theta(H+1))**2
              Lambda = DSIN(pH_Theta(H))**2
              Qunprot = Lambda*(QState2(1,cc)-QState2(2,cc))
              Qprot = (1-Lambda)*(QState1(1,cc)-QState1(2,cc))
              QXH = cgb*(Qunprot+Qprot)
           ENDIF
           Qunprot = X2*QState2(1,cc) +(1-X2)*QState2(2,cc)
           Qprot = X2*QState1(1,cc) + (1-X2)*QState1(2,cc)
           QH = cgb*(Qunprot - Qprot)
        ENDIF
        IF (G .GT. 0) THEN
           IF (SP_GRP(G) .LE. 0) THEN
              X2 = ONE
           ELSE
              X2 = DSIN(pH_Theta(G+1))**2
              Lambda = DSIN(pH_Theta(G))**2
              Qunprot = Lambda*(QState2(1,bb)-QState2(2,bb))
              Qprot =(1-Lambda)*(QState1(1,bb)-QState1(2,bb))
              QXG = cga*(Qunprot + Qprot)
           ENDIF
           Qunprot = X2*QState2(1,bb) + (1-X2)*QState2(2,bb)
           Qprot = X2*QState1(1,bb) + (1-X2)*QState1(2,bb)
           QG = cga*(Qunprot - Qprot)
        ENDIF 
        xa=x(aa)
        ya=y(aa)
        za=z(aa)
        xb=x(bb)
        yb=y(bb)
        zb=z(bb)
        dxab=xb-xa
        dyab=yb-ya
        dzab=zb-za
        r2ab=dxab*dxab+dyab*dyab+dzab*dzab
        IF (r2ab .LT. C2OfNb) THEN
          FSw = ONE
          DFSw= ZERO
          IF (CSWIT) THEN
            LOuter = (r2ab .GT. C2OnNb)
            IF (LOuter) THEN
              RijL = C2OnNb - r2ab
              RijU = C2OfNb - r2ab
              FSw  = RijU * RijU * (RijU-3.0D0*RijL) * Rul3
              DfSw = RijL * RijU * Rul12 / FSw
            ENDIF ! outer
          ENDIF
          IF (QGBSW.or.QGBMV) THEN
             rBorna=Alpha(cc)
             rBornb=Alpha(bb)
             r2born=rBorna*rBornb
             expfac=exp(-r2ab/(4.0D0*r2born))
             rGB2=r2ab+r2born*expfac
             rGB=sqrt(rGB2)
             IF (kappagb .GT. ZERO) THEN
                effsalt=exp(-rGB*scrfac)/eps_w
                tau=( 1/eps_p-effsalt ) * ccelec
             ENDIF
             C2 = -FSw*tau/rGB
          ELSE
             C2 = ZERO
          ENDIF ! GB
         R1 = FSw*CCELEC/SQRT(r2ab)
         IF (G.GT.0) THEN
            dpH_Theta(G) = dpH_Theta(G) + QG*(R1+C2)
            IF (SP_GRP(G) .GT. 0) THEN
               dpH_Theta(G+1) = dpH_Theta(G+1) + &
               QXG*(R1+C2)
            ENDIF
         ENDIF
   
         IF (H.GT.0) THEN
            dpH_Theta(H) = dpH_Theta(H) + QH*(R1+C2)
            IF (SP_GRP(H) .GT. 0) THEN
               dpH_Theta(H+1) = dpH_Theta(H+1) + &
               QXH*(R1+C2)
            ENDIF
         ENDIF
       ENDIF ! r2ab
     ENDDO ! nsolp
  ENDIF ! images
ELSE ! J. Wallace - explicit solvent 
! 1.2.2 primary-image
! *******************
      IF (natim.GT.natom) THEN
      itemp0=natom+1
      DO itrans=1,ntrans
         jtrans=iminv(itrans)
         istrt=itemp0
         iend=imatpt(itrans)
         ipt=(itrans-1)*12
         itemp0=iend+1

      DO aa=istrt,iend
         cc=imattr(aa)          ! primary atom index
         cga=cg(aa)
         H = GrpList(cc)

         IF (cga.NE.ZERO) THEN
            xa=x(aa)
            ya=y(aa)
            za=z(aa)
#if KEY_IMCUBES==1
         IF (lbycbim) THEN
            ITEMP=ImINBL(aa+natim)
            Npr=ImINbl(aa)-ITEMP
         ELSE
#endif
            IF (aa.GT.1) THEN
                ITEMP=ImINBL(aa-1)
                Npr=ImINBL(aa)-ITEMP
            ELSE
                Npr=ImINBL(aa)
                ITEMP=0
            ENDIF
#if KEY_IMCUBES==1
        ENDIF
#endif
           IACI = IACNB(aa)
            DO jpr = 1, NPr
               k  = ImJNBL(ITemp+jpr)
               bb = Abs(k)
               G =  GrpList(bb)

               IF ((H .gt. 0).OR.(G .gt. 0)) THEN
                  cgb= cg(bb)
                  xb=x(bb)
                  yb=y(bb)
                  zb=z(bb)
                  dxab=xb-xa
                  dyab=yb-ya
                  dzab=zb-za
                  r2ab=dxab*dxab+dyab*dyab+dzab*dzab
                  IF (r2ab .LT. C2OfNb) THEN
                     IF (H .GT. 0) THEN
                        IF (SP_GRP(H) .LE. 0) THEN
                          X2 = ONE
                        ELSE
                          X2 = DSIN(pH_Theta(H+1))**2
                          Lambda = DSIN(pH_Theta(H))**2
                          Qunprot = Lambda*(QState2(1,cc)-QState2(2,cc))
                          Qprot = (1-Lambda)*(QState1(1,cc)-QState1(2,cc))
                          QXH = cgb*(Qunprot+Qprot)
                        ENDIF
                        Qunprot = X2*QState2(1,cc) +(1-X2)*QState2(2,cc)
                        Qprot = X2*QState1(1,cc) + (1-X2)*QState1(2,cc)
                        QH = cgb*(Qunprot - Qprot)
                     ENDIF

                  IF (G .GT. 0) THEN
                     IF (SP_GRP(G) .LE. 0) THEN
                       X2 = ONE
                     ELSE
                        X2 = DSIN(pH_Theta(G+1))**2
                        Lambda = DSIN(pH_Theta(G))**2
                        Qunprot = Lambda*(QState2(1,bb)-QState2(2,bb))
                        Qprot =(1-Lambda)*(QState1(1,bb)-QState1(2,bb))

                        QXG = cga*(Qunprot + Qprot)
                     ENDIF
                     Qunprot = X2*QState2(1,bb) + (1-X2)*QState2(2,bb)
                     Qprot = X2*QState1(1,bb) + (1-X2)*QState1(2,bb)
                     QG = cga*(Qunprot - Qprot)
                  ENDIF

               IF (cgb .NE. ZERO) THEN
                     FSw = ONE
                     DFSw= ZERO
                     IF (CSWIT) THEN
                        LOuter = (r2ab .GT. C2OnNb)

                        IF (LOuter) THEN
                           RijL = C2OnNb - r2ab
                           RijU = C2OfNb - r2ab
                           FSw  = RijU * RijU * (RijU-3.0D0*RijL) * Rul3
                           DfSw = RijL * RijU * Rul12 / FSw
                        ENDIF
                     ENDIF
         
                     C2 = ZERO
                    
                     IF(QGRF)THEN
                        R1 = iepsi*CCELEC*(1/SQRT(r2ab)+(HALF*RFCON*r2ab*rc3)- &
                                                  ((ONE+HALF*RFCON)*rc))
! PME
                        ELSEIF (QPME .and. (.not. QGBSW) .and. (.not. QGBMV)) THEN    
                                        ! real space pme -- Y Huang 2016
                        rab = sqrt(r2ab)
                        call erfcd(rab,pmekappa,erfcx,drfc,erfmod)
                        R1 = ccelec*erfcx/rab
! ENDPME
                     ELSEIF(CSHIFT)THEN ! force-shifted--Wei Chen, 07/05/2013
                        rab=SQRT(r2ab)
                        R1 = 1/EPS*CCELEC/rab*(1.0 - 2.0*rab/CtOfNb + r2ab/C2OfNb)
                     ELSEIF(CSHFT)THEN ! shift -- Wei Chen, 07/06/2013
                        R1 = 1/EPS*CCELEC/SQRT(r2ab)*(1.0 - r2ab/C2OfNb)**2
                     ELSEIF(CSWIT)THEN ! switch -- Wei Chen, 07/16/2013
                        R1 = 1/EPS*FSw*CCELEC/SQRT(r2ab)
                     ELSE
                        R1 = 1/EPS*CCELEC/SQRT(r2ab)                        
                     ENDIF

                     IF (G.GT.0) THEN
                        dpH_Theta(G) = dpH_Theta(G) + QG*(R1+C2)
                        IF (SP_GRP(G) .GT. 0) THEN
                           dpH_Theta(G+1) = dpH_Theta(G+1) + &
                               QXG*(R1+C2)
                        ENDIF
                     ENDIF

                     IF (H.GT.0) THEN
                        dpH_Theta(H) = dpH_Theta(H) + QH*(R1+C2)
                        IF (SP_GRP(H) .GT. 0) THEN
                           dpH_Theta(H+1) = dpH_Theta(H+1) + &
                             QXH*(R1+C2)
                        ENDIF
                     ENDIF
             
               ENDIF ! cgb/=0
               ENDIF ! titratable group
              ENDIF ! r2ab
            ENDDO ! jpr
         ENDIF ! cga/=0
      ENDDO ! image atom aa
      ENDDO ! itrans
! 1.2.3 interactions with self images
! **********************************

      IF (NImNBSL.gt.0) THEN
      itemp0=natom+1
      DO itrans=1,ntrans
         jtrans=iminv(itrans)
         istrt=itemp0
         iend=imatpt(itrans)
         ipt=(itrans-1)*12
         itemp0=iend+1

      DO aa=istrt,iend
         cc=imattr(aa)          ! primary atom index

         cga=cg(aa)
         H = GrpList(cc)

         IF (cga.NE.ZERO) THEN
            xa=x(aa)
            ya=y(aa)
            za=z(aa)
            rBorna=Alpha(cc)
#if KEY_IMCUBES==1
            IF (lbycbim) THEN
               ITEMP=ImINBSL(aa+natim) ! natim for image nonbond list
               Npr=ImINbl(aa)-ITEMP
            ELSE
#endif
               IF (aa.GT.1) THEN
                  ITEMP=ImINBSL(aa-1)
                  Npr=ImINBSL(aa)-ITEMP
               ELSE
                  Npr=ImINBSL(aa)
                  ITEMP=0
               ENDIF
#if KEY_IMCUBES==1
           ENDIF
#endif
            IACI = IACNB(aa)
            DO jpr = 1, NPr
               k  = ImJNBSL(ITemp+jpr)
               bb = Abs(k)
               G = GrpList(bb)

               IF ((H .gt. 0).OR.(G .gt. 0)) THEN
                  cgb=cg(bb)
                  xb=x(bb)
                  yb=y(bb)
                  zb=z(bb)
                  dxab=xb-xa
                  dyab=yb-ya
                  dzab=zb-za
                  r2ab=dxab*dxab+dyab*dyab+dzab*dzab
                  IF (r2ab .LT. C2OfNb) THEN
                     IF (H .GT. 0) THEN
                        IF (SP_GRP(H) .LE. 0) THEN
                           X2 = ONE
                        ELSE
                           X2 = DSIN(pH_Theta(H+1))**2
                           Lambda = DSIN(pH_Theta(H))**2
                           Qunprot = Lambda*(QState2(1,cc)-QState2(2,cc))
                           Qprot = (1-Lambda)*(QState1(1,cc)-QState1(2,cc))
                           QXH = cgb*(Qunprot+Qprot)
                        ENDIF
                        Qunprot = X2*QState2(1,cc) +(1-X2)*QState2(2,cc)
                        Qprot = X2*QState1(1,cc) + (1-X2)*QState1(2,cc)
                        QH = cgb*(Qunprot - Qprot)
                     ENDIF

                  IF (G .GT. 0) THEN
                     IF (SP_GRP(G) .LE. 0) THEN
                        X2 = ONE
                     ELSE
                        X2 = DSIN(pH_Theta(G+1))**2
                        Lambda = DSIN(pH_Theta(G))**2
                        Qunprot = Lambda*(QState2(1,bb)-QState2(2,bb))
                        Qprot =(1-Lambda)*(QState1(1,bb)-QState1(2,bb))
                        QXG = cga*(Qunprot + Qprot)
                     ENDIF
                     Qunprot = X2*QState2(1,bb) + (1-X2)*QState2(2,bb)
                     Qprot = X2*QState1(1,bb) + (1-X2)*QState1(2,bb)
                     QG = cga*(Qunprot - Qprot)
                  ENDIF

               IF (cgb .NE. ZERO) THEN
                     FSw = ONE
                     DFSw= ZERO
                     IF (CSWIT) THEN
                        LOuter = (r2ab .GT. C2OnNb)

                        IF (LOuter) THEN
                           RijL = C2OnNb - r2ab
                           RijU = C2OfNb - r2ab
                           FSw  = RijU * RijU * (RijU-3.0D0*RijL) * Rul3
                           DfSw = RijL * RijU * Rul12 / FSw
                        ENDIF
                     ENDIF

                     C2 = ZERO
               
                     IF(QGRF)THEN ! J. Wallace
                       R1 = iepsi*CCELEC*(1/SQRT(r2ab)+(HALF*RFCON*r2ab*rc3)-& 
                                                 ((ONE+HALF*RFCON)*rc))
! PME
                      ELSEIF (QPME .and. (.not. QGBSW) .and. (.not. QGBMV)) THEN
                                       ! Y Huang 2016
                        rab = sqrt(r2ab)
                        call erfcd(rab,pmekappa,erfcx,drfc,erfmod)
                        R1 = ccelec*erfcx/rab
! ENDPME                     
                     ELSEIF(CSHIFT)THEN ! force-shifted--Wei Chen, 07/05/2013
                       rab=SQRT(r2ab)
                       R1 = 1/EPS*CCELEC/rab*(1.0 - 2.0*rab/CtOfNb + r2ab/C2OfNb)
                     ELSEIF(CSHFT)THEN ! shift -- Wei Chen, 07/06/2013
                       R1 = 1/EPS*CCELEC/SQRT(r2ab)*(1.0 - r2ab/C2OfNb)**2
                     ELSEIF(CSWIT)THEN ! switch -- Wei Chen, 07/16/2013
                       R1 = 1/EPS*FSw*CCELEC/SQRT(r2ab)
                     ELSE
                       R1 = 1/EPS*CCELEC/SQRT(r2ab) ! Coulomb energy
                     ENDIF

                     IF (G.GT.0) THEN
                        dpH_Theta(G) = dpH_Theta(G)+HALF*QG*(R1+C2)
                        IF (SP_GRP(G) .GT. 0) THEN
                          dpH_Theta(G+1) = dpH_Theta(G+1) + &
                               HALF*QXG*(R1+C2)
                        ENDIF
                     ENDIF

                     IF (H.GT.0) THEN
                        dpH_Theta(H) = dpH_Theta(H)+HALF*QH*(R1+C2)
                        IF (SP_GRP(H) .GT. 0) THEN
                           dpH_Theta(H+1) = dpH_Theta(H+1) + &
                               HALF*QXH*(R1+C2)
                        ENDIF
                     ENDIF
                
               ENDIF ! cgb/=0
               ENDIF ! titratable group
              ENDIF         ! r2ab
           ENDDO ! jpr
         ENDIF ! cga/=0
      ENDDO ! image atom aa
      ENDDO ! itrans
      ENDIF ! NImNBSL
      ENDIF ! natim >natom

ENDIF


! 1.3.1 self GB energy
! ******************
  IF (QGBSW .or. QGBMV) THEN
     nstrt=1
     nskip=1

#if KEY_PARALLEL==1
     nstrt=mynodp
     nskip=numnod
#endif 

     DO aa=nstrt,nsolute,nskip
        cga=cg(aa)
        G = GrpList(aa)

        IF ((G .GT. 0)) THEN
           IF (cga.NE.ZERO) then
              rBorna=Alpha(aa)
              IF (kappagb.GT.ZERO) THEN
                 effsalt=exp(-rBorna*scrfac)/eps_w
                 tau=( 1/eps_p-effsalt ) * ccelec
              ENDIF

              IF (SP_GRP(G) .LE. 0) THEN
                 X2 = ONE
              ELSE
                 X2 = DSIN(pH_Theta(G+1))**2
                 Lambda = DSIN(pH_Theta(G))**2

                 Qunprot = Lambda*(QState2(1,aa)-QState2(2,aa))
                 Qprot =(1-Lambda)*(QState1(1,aa)-QState1(2,aa))
                 QXG = cga*(Qunprot + Qprot)
                 dpH_Theta(G+1) = dpH_Theta(G+1) - QXG*tau/rBorna
              ENDIF
              Qunprot = X2*QState2(1,aa) + (1-X2)*QState2(2,aa)
              Qprot = X2*QState1(1,aa) + (1-X2)*QState1(2,aa)
              QG = cga*(Qunprot - Qprot)
              dpH_Theta(G) = dpH_Theta(G) - QG*tau/rBorna
           ENDIF
        ENDIF                  ! titratable
    ENDDO !aa
  ENDIF

! PME
! 1.3.2 PME self-energy and net charge correction
! ********************************
IF (QPME .and. (.not. QGBSW) .and. (.not. QGBMV)) THEN
     nstrt=1
     nskip=1
!     IF(NETQ .NE. ZERO)            ! check if the system is always neutral
        pmenetch = 0.0
        DO aa=nstrt,nsolute,nskip
           pmenetch = pmenetch+cg(aa) 
        ENDDO                      ! calculate net charge
!        pmenetch = ABS(pmenetch)   ! absolute value is required, or unnessary?
!     ENDIF  
#if KEY_PARALLEL==1
     nstrt=mynodp
     nskip=numnod
#endif
     DO aa=nstrt,nsolute,nskip
        cga=cg(aa)
        G = GrpList(aa)

        IF ((G .GT. 0)) THEN    ! if titratable
           IF (SP_GRP(G) .LE. 0) THEN ! if tautomeric 
              X2 = ONE                        ! non-tautomeric state
           ELSE
              X2 = DSIN(pH_Theta(G+1))**2     ! x --> G+1
              Lambda = DSIN(pH_Theta(G))**2   ! lambda --> G
              Qunprot = Lambda*(QState2(1,aa)-QState2(2,aa))          ! first derivative to x
              Qprot =(1-Lambda)*(QState1(1,aa)-QState1(2,aa))         ! first derivative to x
              QXG =  (Qunprot + Qprot)                                ! dq/dx
              dpH_Theta(G+1) = dpH_Theta(G+1) - QXG*pmebeta*2*cga     ! -pmebeta*(dq**2/dx)
!             IF(NETQ .NE. ZERO) THEN                              ! check if the system is always neutral
                 dpH_Theta(G+1) = dpH_Theta(G+1) - QXG*ccelec*pi*pmenetch/(pmekappa*pmekappa*pmevolume)
!             ENDIF
           ENDIF
           Qunprot = X2*QState2(1,aa) + (1-X2)*QState2(2,aa)          ! first derivative to lamb
           Qprot = X2*QState1(1,aa) + (1-X2)*QState1(2,aa)            ! first derivative to lamb
           QG = (Qunprot - Qprot)                               ! dq/dlamb
           dpH_Theta(G) = dpH_Theta(G) - QG*pmebeta*2*cga             ! -pmebeta*(dq**2/dlamb)
!          IF(NETQ .NE. ZERO) THEN                              ! check if the system is always neutral
              dpH_Theta(G) = dpH_Theta(G) - QG*ccelec*pi*pmenetch/(pmekappa*pmekappa*pmevolume)
!          ENDIF
        ENDIF ! titratable
    ENDDO !aa
ENDIF     
! Excluding intra-solute electrostatic interaction from pme.
! Erfc part without bonded atom pairs have been calculated earlier.
! Adding the two items with k-space item below form the complete pme.
! ENDPME



! 1.4 PME kspace energy (Y Huang 2016)
! *********************
IF (QPME .and. (.not. QGBSW) .and. (.not. QGBMV)) THEN
    call do_pme_phmd(dpH_Theta,pH_Theta,NTitr,Grplist,SP_GRP, &
                     natom,x,y,z,cg,Qstate1,Qstate2)
ENDIF


! PME
! 1.5 PME ewald electrostatic exclusion energy
!**********************************************
IF (QPME .and. (.not. QGBSW) .and. (.not. QGBMV)) THEN
     nstrt=1
     nskip=1
#if KEY_PARALLEL==1
     nstrt=mynodp
     nskip=numnod
#endif
     DO aa=nstrt,natom,nskip
       IF(aa > 1)THEN
          NXI=INblo14(aa-1)+1
       ELSE
          NXI=1
       ENDIF
       NXIMAX=Inblo14(aa)

       cga = cg(aa)
       H = GrpList(aa)

       DO K=NXI,NXIMAX
          bb=Inbl14(K)
          IF(bb > 0) THEN
          
          cgb = cg(bb)
          G = GrpList(bb) 
          
          IF ((H .GT. 0).OR.(G .GT. 0)) THEN

              IF (H .GT. 0) THEN
                 IF (SP_GRP(H) .LE. 0) THEN
                    X2 = ONE
                 ELSE
                    X2 = DSIN(pH_Theta(H+1))**2
                    Lambda = DSIN(pH_Theta(H))**2
                    Qunprot = Lambda*(QState2(1,aa)-QState2(2,aa))
                    Qprot = (1-Lambda)*(QState1(1,aa)-QState1(2,aa))
                    QXH = cgb*(Qunprot+Qprot)
                 ENDIF
                 Qunprot = X2*QState2(1,aa) +(1-X2)*QState2(2,aa)
                 Qprot = X2*QState1(1,aa) + (1-X2)*QState1(2,aa)
                 QH = cgb*(Qunprot - Qprot)

              ENDIF

              IF (G .GT. 0) THEN
                 IF (SP_GRP(G) .LE. 0) THEN
                    X2 = ONE
                 ELSE
                    X2 = DSIN(pH_Theta(G+1))**2
                    Lambda = DSIN(pH_Theta(G))**2
                    Qunprot = Lambda*(QState2(1,bb)-QState2(2,bb))
                    Qprot =(1-Lambda)*(QState1(1,bb)-QState1(2,bb))
                    QXG = cga*(Qunprot + Qprot)
                 ENDIF
                 Qunprot = X2*QState2(1,bb) + (1-X2)*QState2(2,bb)
                 Qprot = X2*QState1(1,bb) + (1-X2)*QState1(2,bb)
                 QG = cga*(Qunprot - Qprot)
              ENDIF



              dxab=x(bb)-x(aa)
              dyab=y(bb)-y(aa)
              dzab=z(bb)-z(aa)
              r2ab=MAX(RSMALL,dxab*dxab+dyab*dyab+dzab*dzab)
              rab=SQRT(r2ab)
             
#ifdef __PGI
              efac = erf(rab * pmekappa)
#else
              EFAC=DERF(rab*pmekappa)
#endif
              R1=-ccelec*EFAC/rab   
              ! Ecorr_exc=-0.5*qi*qj*erf(rij*pmekappa)/rij
              ! where i and j are 1-2/3/4 bonded

              IF (G.GT.0) THEN
                 dpH_Theta(G) = dpH_Theta(G) + QG*R1

                 IF (SP_GRP(G) .GT. 0) THEN
                    dpH_Theta(G+1) = dpH_Theta(G+1) + QXG*R1
                 ENDIF
              ENDIF

              IF (H.GT.0) THEN
                 dpH_Theta(H) = dpH_Theta(H) + QH*R1
                 
                 IF (SP_GRP(H) .GT. 0) THEN
                    dpH_Theta(H+1) = dpH_Theta(H+1) + QXH*R1
                 ENDIF
              ENDIF
          ENDIF  ! atom aa/bb titratable
          ENDIF  ! bb bonded with aa
       ENDDO  ! atom bb
    ENDDO  ! atom aa
ENDIF ! QPME
! END PME_PHMD -- Y Huang 2016



  ! 2. VDW energy
  ! 2.1 primary atoms
  ! *****************
  IF (QETERM(VDW)) THEN
     DO aa=1,nsolute-1
        H = GrpList(aa)

        xa=x(aa)
        ya=y(aa)
        za=z(aa)

        tmpx=ZERO
        tmpy=ZERO
        tmpz=ZERO

#if KEY_IMCUBES==1
        IF (lbycbim) THEN
           Itemp=INBL(aa+natom)
        ELSE
#endif 
           IF(aa.GT.1) THEN
              Itemp=INBL(aa-1)
           ELSE
              ITEMP=0
           ENDIF
#if KEY_IMCUBES==1
        ENDIF
#endif 
        Npr = INBL(aa) - ITEMP
        IACI = IACNB(aa)
        Do jpr = 1, NPr
           k  = JNbl(ITemp+jpr)
           bb = Abs(k)
           if ( bb .gt. nsolute ) then ! J. Wallace 
              G = 0                     
           else                         
              G = GrpList(bb)           
           endif 

           IF ((H .GT. 0).OR.(G .gt. 0)) THEN

              ! Set up some variables

              LHTitr = .FALSE.
              LHTauto = .FALSE.
              XH = ONE
              iHTaut = 0
              IF (H .GT. 0) THEN
                 RADH = VState1(1,aa)
                 LambdaH = DSIN(pH_Theta(H))**2
                 LambdaH2 = ONE - LambdaH 
                 FactH = LambdaH2 
                 IF (SP_GRP(H).EQ.1 .OR. SP_GRP(H).EQ.3) THEN 
                    LHTauto = .TRUE.
                    X2 = DSIN(pH_Theta(H+1))**2
                    RADH = RADH + VState1(2,aa)
                    IF (VState1(1,aa) .GT. ZERO) THEN
                       iHTaut = 1
                       XH = X2
                    ELSEIF (VState1(2,aa) .GT. ZERO) THEN
                       iHTaut = -1
                       XH = ONE - X2
                    ENDIF
                    IF (SP_GRP(H).EQ.1) THEN
                       FactH = ONE - LambdaH*XH
                       dXH = -iHTaut*LambdaH 
                    ELSEIF (SP_GRP(H).EQ.3) THEN
                       FactH = LambdaH2*XH 
                       dXH = iHTaut*LambdaH2
                    ENDIF
                 ENDIF
                 IF (RADH .GT. ZERO) LHTitr = .TRUE.
              ENDIF

              LGTitr = .FALSE.
              LGTauto = .FALSE.
              XG = ONE
              iGTaut = 0
              IF (G .GT. 0) THEN
                 RADG = VState1(1,bb)
                 LambdaG = DSIN(pH_Theta(G))**2
                 LambdaG2 = ONE - LambdaG
                 FactG = LambdaG2
                 IF (SP_GRP(G).EQ.1 .OR. SP_GRP(G).EQ.3) THEN
                    LGTauto = .TRUE.
                    RADG =  RADG + VState1(2,bb)
                    X2 = DSIN(pH_Theta(G+1))**2

                    IF (VState1(1,bb) .GT. ZERO) THEN
                       iGTaut = 1
                       XG = X2
                    ELSEIF (VState1(2,bb) .GT. ZERO) THEN
                       iGTaut = -1
                       XG = ONE - X2
                    ENDIF
                    IF (SP_GRP(G).EQ.1) THEN
                       FactG = ONE - LambdaG*XG 
                       dXG = -iGTaut*LambdaG
                    ELSEIF (SP_GRP(G).EQ.3) THEN
                       FactG = LambdaG2*XG 
                       dXG = iGTaut*LambdaG2
                    ENDIF
                 ENDIF
                 IF (RADG .GT. ZERO) LGTitr = .TRUE.
              ENDIF

              IF (LHTitr .OR. LGTitr) THEN
                 xb=x(bb)
                 yb=y(bb)
                 zb=z(bb)

                 dxab=xb-xa
                 dyab=yb-ya
                 dzab=zb-za
                 r2ab=dxab*dxab+dyab*dyab+dzab*dzab

                 IF (r2ab .LT. C2OfNb) THEN
                    FSw = ONE
                    DFSw= ZERO
                    IF (DSWIT) THEN
                       LOuter = (r2ab .GT. C2OnNb)

                       IF (LOuter) THEN
                          RijL = C2OnNb - r2ab
                          RijU = C2OfNb - r2ab
                          FSw  = RijU * RijU * (RijU-3.0D0*RijL) * Rul3
                          DfSw = RijL * RijU * Rul12 / FSw
                       EnDIF
                    ENDIF

                    ! Compute VDW energy U(aa,bb)
                    IVECT=LOWTP(MAX(IACNB(bb),IACI))+IACNB(bb)+IACI
                    IF (k .LT. 0) IVECT = IVECT + NITCC2

                    TR2 = 1/r2ab
                    TR6 = TR2 * TR2 * TR2
                    CA = CCNBA(IVECT)*TR6*TR6 ! Repulsive term

                    IF (LOUTER) THEN
                       ENEVDW = CA-CCNBB(IVECT)*TR6 ! Attractive term
                       ENN = ENEVDW*FSW
                       TFVDW = ENEVDW*DFSW-SIX*TR2*(ENN+CA*FSW)
                    ELSE
                       ENN = CA-CCNBB(IVECT)*TR6
                       TFVDW = MINSIX*TR2*(ENN+CA)
                    ENDIF


                    ! Compute lambda energy and force for two and one titrating H's

                    Fact = ONE
                    IF (LHTitr .AND. LGTitr) THEN
                       IF (H.NE.G) THEN
                          Fact = FactH*FactG

                          dpH_Theta(H) = dpH_Theta(H) - XH*FactG*ENN
                          dpH_Theta(G) = dpH_Theta(G) - XG*FactH*ENN

                          IF (LHTauto) THEN
                             dpH_Theta(H+1)=dpH_Theta(H+1) + dXH*FactG*ENN
                          ENDIF
                          IF (LGTauto) THEN
                             dpH_Theta(G+1)=dpH_Theta(G+1) + dXG*FactH*ENN
                          ENDIF

                       ELSEIF (SP_GRP(H).EQ.1) THEN 
                          Fact = ONE - LambdaH
                          dpH_Theta(H) = dpH_Theta(H) -ENN
                       ENDIF
                    ELSEIF (LHTitr) THEN
                       Fact = FactH 
                       dpH_Theta(H) = dpH_Theta(H) - XH*ENN
                       IF (LHTauto) THEN 
                          dpH_Theta(H+1) = dpH_Theta(H+1) + dXH*ENN
                       ENDIF
                    ELSEIF (LGTitr) THEN
                       Fact = FactG 
                       dpH_Theta(G) = dpH_Theta(G) - XG*ENN
                       IF (LGTauto) THEN 
                          dpH_Theta(G+1) = dpH_Theta(G+1) + dXG*ENN
                       ENDIF
                    ENDIF

                    Fact = Fact-ONE

                    VDW_PHMD = VDW_PHMD + Fact*ENN

                    ! force on spatial coordinates
                    ! ****************************
                    tx = Fact*TFVDW*dxab
                    ty = Fact*TFVDW*dyab
                    tz = Fact*TFVDW*dzab

                    DX(bb) = DX(bb) + tx
                    DY(bb) = DY(bb) + ty
                    DZ(bb) = DZ(bb) + tz

                    tmpx = tmpx + tx
                    tmpy = tmpy + ty
                    tmpz = tmpz + tz

                 ENDIF              ! One is titratable H
              ENDIF               ! R2AB
           ENDIF                  ! titratable group
        ENDDO                     ! JPR

        DX(aa) = DX(aa) - tmpx
        DY(aa) = DY(aa) - tmpy
        DZ(aa) = DZ(aa) - tmpz
     ENDDO                     ! aa

     ! 2.  VDW energy
     ! 2.2 image atoms
     ! ****************
     IF (natim .GT. natom) THEN
        itemp0=natom+1
        DO itrans=1,ntrans
           jtrans=iminv(itrans)
           istrt=itemp0
           iend=imatpt(itrans)
           ipt=(itrans-1)*12
           itemp0=iend+1

           DO aa=istrt,iend
              cc=imattr(aa) ! primary atom index
              if ( cc .gt. nsolute ) then  ! J. Wallace
                  H = 0                    
              else                         
                  H = GrpList(cc)          
              endif                        

              xa=x(aa)
              ya=y(aa)
              za=z(aa)

              tmpx=ZERO
              tmpy=ZERO
              tmpz=ZERO

#if KEY_IMCUBES==1
              IF (lbycbim) THEN
                 ITEMP=ImINBL(aa+natim) ! natim for image nonbond list
              ELSE
#endif 
                 IF(aa.GT.1) THEN
                    ITEMP=ImINBL(aa-1)
                 ELSE
                    ITEMP=0
                 ENDIF
#if KEY_IMCUBES==1
              endif
#endif 
              Npr = ImINBL(aa) - ITEMP
              IACI = IACNB(aa)
              Do jpr = 1, NPr
                 k = ImJNBL(ITemp+jpr)
                 bb = Abs(k)

                 ! J. Wallace - hybrid solvent vdw: skip water-water
                 if ((cc .gt. nsolute) .and. (bb .gt. nsolute)) cycle
                 if ( bb .gt. nsolute ) then 
                    G = 0                    
                 else                        
                    G = GrpList(bb)          
                 endif 

                 G = GrpList(bb)

                 IF ((H .GT. 0).OR.(G .GT. 0)) THEN
                    LHTitr = .FALSE.
                    LHTauto = .FALSE.
                    XH = ONE
                    iHTaut = 0
                    IF (H .GT. 0) THEN
                       RADH = VState1(1,cc)
                       LambdaH = DSIN(pH_Theta(H))**2
                       LambdaH2 = ONE - LambdaH
                       FactH = LambdaH2
                       IF (SP_GRP(H).EQ.1 .OR. SP_GRP(H).EQ.3) THEN
                          LHTauto = .TRUE.
                          X2 = DSIN(pH_Theta(H+1))**2
                          RADH = RADH + VState1(2,cc)
                          IF (VState1(1,cc) .GT. ZERO) THEN
                             iHTaut = 1
                             XH = X2
                          ELSEIF (VState1(2,cc) .GT. ZERO) THEN
                             iHTaut = -1
                             XH = ONE - X2
                          ENDIF
                          IF (SP_GRP(H).EQ.1) THEN
                             FactH = ONE - LambdaH*XH
                             dXH = -iHTaut*LambdaH
                          ELSEIF (SP_GRP(H).EQ.3) THEN
                             FactH = LambdaH2*XH
                             dXH = iHTaut*LambdaH2
                          ENDIF
                       ENDIF
                       IF (RADH .GT. ZERO) LHTitr = .TRUE.
                    ENDIF

                    LGTitr = .FALSE.
                    LGTauto = .FALSE.
                    XG = ONE
                    iGTaut = 0
                    IF (G .GT. 0) THEN
                       RADG = VState1(1,bb)
                       LambdaG = DSIN(pH_Theta(G))**2
                       LambdaG2 = ONE - LambdaG
                       FactG = LambdaG2
                       IF (SP_GRP(G).EQ.1 .OR. SP_GRP(G).EQ.3) THEN
                          LGTauto = .TRUE.
                          RADG =  RADG + VState1(2,bb)
                          X2 = DSIN(pH_Theta(G+1))**2

                          IF (VState1(1,bb) .GT. ZERO) THEN
                             iGTaut = 1
                             XG = X2
                          ELSEIF (VState1(2,bb) .GT. ZERO) THEN
                             iGTaut = -1
                             XG = ONE - X2
                          ENDIF
                          IF (SP_GRP(G).EQ.1) THEN
                             FactG = ONE - LambdaG*XG
                             dXG = -iGTaut*LambdaG
                          ELSEIF (SP_GRP(G).EQ.3) THEN
                             FactG = LambdaG2*XG
                             dXG = iGTaut*LambdaG2
                          ENDIF
                       ENDIF
                       IF (RADG .GT. ZERO) LGTitr = .TRUE.
                    ENDIF

                    IF (LHTitr .OR. LGTitr) THEN

                       xb=x(bb)
                       yb=y(bb)
                       zb=z(bb)

                       dxab=xb-xa
                       dyab=yb-ya
                       dzab=zb-za
                       r2ab=dxab*dxab+dyab*dyab+dzab*dzab

                       IF (r2ab .Lt. C2OfNb) THEN
                          FSw = ONE
                          DFSw= ZERO
                          If (DSWIT) THEN
                             LOuter = (r2ab .GT. C2OnNb)

                             IF (LOuter) THEN
                                RijL = C2OnNb - r2ab
                                RijU = C2OfNb - r2ab
                                FSw  = RijU * RijU * (RijU-3.0D0*RijL) * Rul3
                                DfSw = RijL * RijU * Rul12 / FSw
                             ENDIF
                          ENDIF

                          ! Compute VDW energy U(aa,bb)

                          IVECT=LOWTP(MAX(IACNB(bb),IACI))+IACNB(bb)+IACI
                          IF (k .LT. 0) IVECT = IVECT + NITCC2

                          TR2 = 1/r2ab
                          TR6 = TR2 * TR2 * TR2
                          CA = CCNBA(IVECT)*TR6*TR6 ! Repulsive term

                          IF (LOUTER) THEN
                             ENEVDW = CA-CCNBB(IVECT)*TR6 ! Attractive term
                             ENN = ENEVDW*FSW
                             TFVDW = ENEVDW*DFSW-SIX*TR2*(ENN+CA*FSW)
                          ELSE
                             ENN = CA-CCNBB(IVECT)*TR6
                             TFVDW = MINSIX*TR2*(ENN+CA)
                          ENDIF


                          ! Compute lambda energy and force for two and one titrating H's

                          Fact = ONE
                          IF (LHTitr .AND. LGTitr) THEN
                             IF (H.NE.G) THEN
                                Fact = FactH*FactG

                                dpH_Theta(H) = dpH_Theta(H) - XH*FactG*ENN
                                dpH_Theta(G) = dpH_Theta(G) - XG*FactH*ENN

                                IF (LHTauto) THEN
                                   dpH_Theta(H+1)=dpH_Theta(H+1) + dXH*FactG*ENN
                                ENDIF
                                IF (LGTauto) THEN
                                   dpH_Theta(G+1)=dpH_Theta(G+1) + dXG*FactH*ENN
                                ENDIF

                             ELSEIF (SP_GRP(H).EQ.1) THEN
                                Fact = ONE - LambdaH
                                dpH_Theta(H) = dpH_Theta(H) -ENN
                             ENDIF
                          ELSEIF (LHTitr) THEN
                             Fact = FactH
                             dpH_Theta(H) = dpH_Theta(H) - XH*ENN
                             IF (LHTauto) THEN
                                dpH_Theta(H+1) = dpH_Theta(H+1) + dXH*ENN
                             ENDIF
                          ELSEIF (LGTitr) THEN
                             Fact = FactG
                             dpH_Theta(G) = dpH_Theta(G) - XG*ENN
                             IF (LGTauto) THEN
                                dpH_Theta(G+1) = dpH_Theta(G+1) + dXG*ENN
                             ENDIF
                          ENDIF

                          Fact = Fact-ONE

                          VDW_PHMD = VDW_PHMD + Fact*ENN

                          ! force on spatial coordinates
                          ! ****************************
                          tx = Fact*TFVDW*dxab
                          ty = Fact*TFVDW*dyab
                          tz = Fact*TFVDW*dzab

                          DX(bb) = DX(bb) + tx
                          DY(bb) = DY(bb) + ty
                          DZ(bb) = DZ(bb) + tz

                          IF (NOROT) THEN
                             tmpx = tmpx + tx
                             tmpy = tmpy + ty
                             tmpz = tmpz + tz
                          ELSE
                             tmpx = tmpx - tx*imtrns(ipt+1)  &
                                  - ty*imtrns(ipt+4) &
                                  - tz*imtrns(ipt+7)

                             tmpy = tmpy - tx*imtrns(ipt+2) &
                                  - ty*imtrns(ipt+5) &
                                  - tz*imtrns(ipt+8)

                             tmpz = tmpz - tx*imtrns(ipt+3) &
                                  - ty*imtrns(ipt+6) &
                                  - tz*imtrns(ipt+9)

                          ENDIF ! NOROT
                       ENDIF      ! One is titratable hydrogen
                    ENDIF       ! R2AB
                 ENDIF          ! titratable group
              ENDDO             ! JPR

              DX(cc) = DX(cc) - tmpx
              DY(cc) = DY(cc) - tmpy
              DZ(cc) = DZ(cc) - tmpz

           ENDDO             ! aa image atoms
        ENDDO             ! itrans

        ! 2.  VDW energy
        ! 2.3 interaction with self images
        ! ********************************
        IF (NImNBSL.gt.0) THEN
           itemp0=natom+1

           DO itrans=1,ntrans
              jtrans=iminv(itrans)
              istrt=itemp0
              iend=imatpt(itrans)
              ipt=(itrans-1)*12
              itemp0=iend+1


              DO aa=istrt,iend
                 cc=imattr(aa) ! primary atom index
                 if ( cc .gt. Nsolute ) then ! J. Wallace
                    H = 0                    
                 else                        
                    H = GrpList(cc)          
                 endif                       

                 xa=x(aa)
                 ya=y(aa)
                 za=z(aa)

                 tmpx=ZERO
                 tmpy=ZERO
                 tmpz=ZERO

#if KEY_IMCUBES==1
                 IF (lbycbim) THEN
                    ITEMP=ImINBSL(aa+natim) ! natim for image nonbond list
                 ELSE
#endif 
                    IF (aa.GT.1) THEN
                       ITEMP=ImINBSL(aa-1)
                    ELSE
                       ITEMP=0
                    ENDIF
#if KEY_IMCUBES==1
                 ENDIF
#endif 

                 Npr = ImINBSL(aa) - ITEMP
                 IACI = IACNB(aa)
                 Do jpr = 1, NPr
                    k = ImJNBSL(ITemp+jpr)
                    bb = Abs(k)
                    ! J. Wallace - hybrid solvent vdw: skip water-water
                    if ((cc .gt. nsolute) .and. (bb .gt. nsolute)) cycle
                    if ( bb .gt. Nsolute ) then 
                       G = 0                    
                    else                        
                       G =GrpList(bb)           
                    endif                       

                    IF ((H .GT. 0).OR.(G .GT. 0)) THEN
                       LHTitr = .FALSE.
                       LHTauto = .FALSE.
                       XH = ONE
                       iHTaut = 0
                       IF (H .GT. 0) THEN
                          RADH = VState1(1,cc)
                          LambdaH = DSIN(pH_Theta(H))**2
                          LambdaH2 = ONE - LambdaH
                          FactH = LambdaH2
                          IF (SP_GRP(H).EQ.1 .OR. SP_GRP(H).EQ.3) THEN
                             LHTauto = .TRUE.
                             X2 = DSIN(pH_Theta(H+1))**2
                             RADH = RADH + VState1(2,cc)
                             IF (VState1(1,cc) .GT. ZERO) THEN
                                iHTaut = 1
                                XH = X2
                             ELSEIF (VState1(2,cc) .GT. ZERO) THEN
                                iHTaut = -1
                                XH = ONE - X2
                             ENDIF
                             IF (SP_GRP(H).EQ.1) THEN
                                FactH = ONE - LambdaH*XH
                                dXH = -iHTaut*LambdaH
                             ELSEIF (SP_GRP(H).EQ.3) THEN
                                FactH = LambdaH2*XH
                                dXH = iHTaut*LambdaH2
                             ENDIF
                          ENDIF
                          IF (RADH .GT. ZERO) LHTitr = .TRUE.
                       ENDIF

                       LGTitr = .FALSE.
                       LGTauto = .FALSE.
                       XG = ONE
                       iGTaut = 0
                       IF (G .GT. 0) THEN
                          RADG = VState1(1,bb)
                          LambdaG = DSIN(pH_Theta(G))**2
                          LambdaG2 = ONE - LambdaG
                          FactG = LambdaG2
                          IF (SP_GRP(G).EQ.1 .OR. SP_GRP(G).EQ.3) THEN
                             LGTauto = .TRUE.
                             RADG =  RADG + VState1(2,bb)
                             X2 = DSIN(pH_Theta(G+1))**2

                             IF (VState1(1,bb) .GT. ZERO) THEN
                                iGTaut = 1
                                XG = X2
                             ELSEIF (VState1(2,bb) .GT. ZERO) THEN
                                iGTaut = -1
                                XG = ONE - X2
                             ENDIF
                             IF (SP_GRP(G).EQ.1) THEN
                                FactG = ONE - LambdaG*XG
                                dXG = -iGTaut*LambdaG
                             ELSEIF (SP_GRP(G).EQ.3) THEN
                                FactG = LambdaG2*XG
                                dXG = iGTaut*LambdaG2
                             ENDIF
                          ENDIF
                          IF (RADG .GT. ZERO) LGTitr = .TRUE.
                       ENDIF

                       IF (LHTitr .OR. LGTitr) THEN

                          xb=x(bb)
                          yb=y(bb)
                          zb=z(bb)

                          dxab=xb-xa
                          dyab=yb-ya
                          dzab=zb-za
                          r2ab=dxab*dxab+dyab*dyab+dzab*dzab

                          IF (r2ab .Lt. C2OfNb) THEN
                             FSw = ONE
                             DFSw= ZERO
                             If (DSWIT) THEN
                                LOuter = (r2ab .GT. C2OnNb)

                                IF (LOuter) THEN
                                   RijL = C2OnNb - r2ab
                                   RijU = C2OfNb - r2ab
                                   FSw  = RijU * RijU * (RijU-3.0D0*RijL) * Rul3
                                   DfSw = RijL * RijU * Rul12 / FSw
                                ENDIF
                             ENDIF

                             ! Compute VDW energy U(aa,bb)

                             IVECT=LOWTP(MAX(IACNB(bb),IACI))+IACNB(bb)+IACI
                             IF (k .LT. 0) IVECT = IVECT + NITCC2

                             TR2 = 1/r2ab
                             TR6 = TR2 * TR2 * TR2
                             CA = CCNBA(IVECT)*TR6*TR6 ! Repulsive term

                             IF (LOUTER) THEN
                                ENEVDW = CA-CCNBB(IVECT)*TR6 ! Attractive term
                                ENN = ENEVDW*FSW
                                TFVDW = ENEVDW*DFSW-SIX*TR2*(ENN+CA*FSW)
                             ELSE
                                ENN = CA-CCNBB(IVECT)*TR6
                                TFVDW = MINSIX*TR2*(ENN+CA)
                             ENDIF


                             ! Compute lambda energy and force for two and one titrating H's

                             Fact = ONE
                             IF (LHTitr .AND. LGTitr) THEN
                                IF (H.NE.G) THEN
                                   Fact = FactH*FactG

                                   dpH_Theta(H) = dpH_Theta(H) - XH*FactG*ENN
                                   dpH_Theta(G) = dpH_Theta(G) - XG*FactH*ENN

                                   IF (LHTauto) THEN
                                      dpH_Theta(H+1)=dpH_Theta(H+1) + dXH*FactG*ENN
                                   ENDIF
                                   IF (LGTauto) THEN
                                      dpH_Theta(G+1)=dpH_Theta(G+1) + dXG*FactH*ENN
                                   ENDIF

                                ELSEIF (SP_GRP(H).EQ.1) THEN
                                   Fact = ONE - LambdaH
                                   dpH_Theta(H) = dpH_Theta(H) -ENN
                                ENDIF
                             ELSEIF (LHTitr) THEN
                                Fact = FactH
                                dpH_Theta(H) = dpH_Theta(H) - XH*ENN
                                IF (LHTauto) THEN
                                   dpH_Theta(H+1) = dpH_Theta(H+1) + dXH*ENN
                                ENDIF
                             ELSEIF (LGTitr) THEN
                                Fact = FactG
                                dpH_Theta(G) = dpH_Theta(G) - XG*ENN
                                IF (LGTauto) THEN
                                   dpH_Theta(G+1) = dpH_Theta(G+1) + dXG*ENN
                                ENDIF
                             ENDIF

                             Fact = Fact-ONE

                             VDW_PHMD = VDW_PHMD + Fact*ENN

                             ! force on spatial coordinates
                             ! ****************************
                             tx = Fact*TFVDW*dxab
                             ty = Fact*TFVDW*dyab
                             tz = Fact*TFVDW*dzab

                             DX(bb) = DX(bb) + tx
                             DY(bb) = DY(bb) + ty
                             DZ(bb) = DZ(bb) + tz

                             IF (NOROT) THEN
                                tmpx = tmpx + tx
                                tmpy = tmpy + ty
                                tmpz = tmpz + tz
                             ELSE
                                tmpx = tmpx - tx*imtrns(ipt+1)  &
                                     - ty*imtrns(ipt+4) &
                                     - tz*imtrns(ipt+7)

                                tmpy = tmpy - tx*imtrns(ipt+2) &
                                     - ty*imtrns(ipt+5) &
                                     - tz*imtrns(ipt+8)

                                tmpz = tmpz - tx*imtrns(ipt+3) &
                                     - ty*imtrns(ipt+6) &
                                     - tz*imtrns(ipt+9)

                             ENDIF ! NOROT
                          ENDIF      ! One is titratable hydrogen
                       ENDIF       ! R2AB
                    ENDIF          ! titratable group
                 ENDDO             ! JPR

                 DX(cc) = DX(cc) - tmpx
                 DY(cc) = DY(cc) - tmpy
                 DZ(cc) = DZ(cc) - tmpz

              ENDDO             ! aa image atoms
           ENDDO             ! itrans
        ENDIF             ! NImNBSL > 0

     ENDIF             ! natim > natom

     ! Add to pH_Energy
     eterm(vdw) = eterm(vdw) + vdw_phmd ! J. Wallace - add modifed vdW energy to vdw term

  ENDIF ! VDW

  ! Put dpH_theta from coupled residue into primary residue
  IF (.FALSE.) THEN ! Not add force on co-ion/titrwat, Wei Chen, 05/19/2014
  IF (ncouple .gt. 0)THEN
     DO I=1,NTitr
        IF(SP_GRP(I) .EQ. -2)THEN
           do j=1,ncouple
              if(qcouple(2,j) .EQ. TitrRes(I))then
                 rescouple=qcouple(1,j)
              endif
           enddo
           do j=1,ntitr
             if(rescouple .EQ. TitrRes(j))then
                jcouple = j
                exit
             endif
           enddo
        dpH_Theta(jcouple) = dpH_Theta(jcouple) + dpH_Theta(I)
        dpH_Theta(I) = ZERO
        ENDIF
    ENDDO
  ENDIF
  ENDIF

  DO I = 1, NTitr 
     dpH_Theta(I) = dpH_Theta(I)*DSIN(TWO*pH_Theta(I))
  ENDDO

#if KEY_PARALLEL==1
  CALL GCOMB(dpH_Theta,NTitr)
#endif 

 irunph = irunph +1

  RETURN
END SUBROUTINE RunPHMD

!---------------------------------------------------------------------

SUBROUTINE PHMDREAD(U)
  !************************************************
  ! Read PHMD restart file: Nose variables, thetas
  !************************************************
  use chm_kinds
  use dimens_fcm
  use stream
  use number
  use psf
#if KEY_PARALLEL==1
  use parallel  
#endif
  use energym
  implicit none
  INTEGER I,J,IGRP,ISub, U
  real(chm_real) TMP1,TMP2,TMP3,TMP4,TMP5,TMP6
  real(chm_real) X1,X2,QState1D,QState1E,QState2D,QState2E, &
       Qunprot,Qprot

  CHARACTER(len=80) LINE
 
  !description line
  READ(U,'(/A)',END=9) LINE

   !if Nose phmd read nose variables
   DO I=1,3
      READ(U,'(3D22.15)') tmp1,tmp2,tmp3
      NOSEOLD(I)=tmp1
      NOSE(I)=tmp2
      NOSEV(I)=tmp3
    ENDDO
    READ(U,'(/A)',END=9) LINE

    EPROP(PHKin) = ZERO

     do i=1,ntitr
        READ(U,'(6D22.15)') TMP1,TMP2,TMP3,TMP4,TMP5,TMP6
        ThetaOld(I) = tmp1
        pH_theta(I) = tmp2
        vpH_Theta(I) = tmp3
        IF(PHBETA .gt. ZERO)THEN
           VPHOLD(I) = tmp4
           dPHOLD(I) = tmp5
        ENDIF
        KINOLD = tmp6
        EPROP(PHKin) = EPROP(PHKin) + HALF*QMASS_PHMD*TMP3*TMP3
     enddo

9 CONTINUE 
  RETURN
END SUBROUTINE PHMDREAD

!---------------------------------------------------------------------
SUBROUTINE PHMDWRIT(U)
  !******************************
  ! Write PHMD restart file
  !******************************

  use chm_kinds
  use number
  use dimens_fcm
  use stream
  use psf
  implicit none
  INTEGER I,U

 ! J. Wallace

 WRITE(U,'(/A)')' !PHMD Nose variables (OLD NEW VELOCITY)'
 WRITE(U,'(3D22.15)')(NOSEOLD(I),NOSE(I),NOSEV(I),I=1,3)
 WRITE(U,'(/A)')' !PHMD Thetas (OLDtheta NEWtheta OLDvelocity NEWvelocity OLDforce KINOLD)'
 IF(PHBETA.eq.ZERO)THEN
     DO I=1,NTitr
         WRITE(U,'(6D22.15)') ThetaOld(I),pH_Theta(I),vpH_Theta(I),ZERO,ZERO,ZERO
     ENDDO
 ELSE
    do i=1,ntitr
       WRITE(U,'(6D22.15)') ThetaOld(I),pH_Theta(I),vpH_Theta(I),VPHOLD(I),dPHOLD(I),KINOLD
    enddo
 ENDIF
  
  RETURN
END SUBROUTINE PHMDWRIT

!---------------------------------------------------------------------
SUBROUTINE PHMDCWRI(U)
  !*******************************************************
  ! Write the PHMD thetas,charges, and proton occupancy
  ! to the trajectory file on unit U
  ! ------------------------------------------------------
  ! NOTES: 
  ! 1. PHMDCWRI is called by /dynamc/cvio.src
  ! 2. additional output to traj is controlled by PHMD
  !    keyword PHTRAJ
  ! 3. Additional output will break visualization software
  !    but, if you know what you are doing it can be 
  !    reformatted.
  !*******************************************************
  ! hocc(natom) - titratable hydrogen occupancy

  use chm_kinds
  use dimens_fcm
  use stream
  use psf
  use number
  implicit none
  INTEGER I,U,H
  real(chm_real) x2,lambdah,lambdah2,tmponoff
  real(chm_real) hocc(natom)
  logical first_h
 
  !write thetas
#if KEY_SINGLE==1
  WRITE(U) PH_THETA
#else /**/
  WRITE(U) (SNGL(pH_Theta(I)),I=1,NTitr)
#endif 
  !write charges
#if KEY_SINGLE==1
  WRITE(U) CG
#else /**/
  WRITE(U) (SNGL(CG(I)),I=1,NATOM)
#endif 

  first_h = .true.

  ! J. Wallace - calculate h-occupancy
  do i=1,natom
     tmponoff = ZERO
     h = grplist(i)
     if(h .gt. 0)then ! titratable
        if(vstate1(1,i) .ne. vstate1(2,i))then ! titratable proton
           if(sp_grp(h) .eq. 1)then ! double-site n
              lambdah = DSIN(pH_Theta(h))**2
              x2 = DSIN(pH_Theta(h+1))**2
              if (first_h) then ! first-proton
                 tmponoff = (ONE-lambdah)*(ONE-x2) 
                 first_h = .false.
              else ! second-proton
                 tmponoff = (ONE-lambdah)*x2
                 first_h = .true.
              endif
           elseif(sp_grp(h) .eq. 3)then ! double-site n
              lambdah = DSIN(pH_Theta(h))**2
              x2 = DSIN(pH_Theta(h+1))**2
              if (first_h) then ! first-proton
                 tmponoff = -(ONE-(ONE-lambdah)*(ONE-x2))
                 first_h = .false.
              else ! second-proton
                 tmponoff = -(ONE-(ONE-lambdah)*x2)
                 first_h = .true.
              endif
           else ! single-site
               lambdah = DSIN(pH_Theta(h))**2
               tmponoff = (ONE-lambdah)
           endif ! double-site
        endif ! titratable proton
     endif ! titratable
     hocc(i) = tmponoff
  enddo ! natom
  
 !write h-occupancy
#if KEY_SINGLE==1
  WRITE(U) hocc
#else /**/
  WRITE(U) (SNGL(hocc(i)),i=1,natom)
#endif /* */

 RETURN
END SUBROUTINE PHMDCWRI
!---------------------------------------------------------------------
SUBROUTINE PHMDCREA(U)
  !*****************************************************
  ! Reads PHMD thetas from the trajectory file on unit U
  !*****************************************************

  use chm_kinds
  use dimens_fcm
  use number
  use stream
  use psf
  implicit none
  INTEGER I,J,ISub,IGRP, U
  real(chm_real) TEMP(NTitr),TMP1,X1,X2,Qunprot,Qprot

  IF (IOLEV .GT. 0) READ(U) TEMP

  DO I=1,NTitr
     pH_Theta(i) = temp(i)
  ENDDO
 
  Call UpdatePHMD(1,1,0,1)

  RETURN
END SUBROUTINE PHMDCREA
!---------------------------------------------------------------------
SUBROUTINE PHMDUPD
  !***********************************************************
  !     Update PHMD parameters on parallel nodes after reading
  !     a restart file
  !***********************************************************

  use chm_kinds
  use dimens_fcm
  use psf
  use parallel,only:psnd4,psnd8
  implicit none
#if KEY_PARALLEL==1
  IF (QPHMD) THEN
     CALL PSND8(CG,NATOM)
     CALL PSND8(ThetaOld,NTitr)
     CALL PSND8(pH_Theta,NTitr)
     CALL PSND4m(NOSE,3)
     CALL PSND4m(NOSEOLD,3)
  ENDIF
#endif 
  RETURN
END SUBROUTINE PHMDUPD
!---------------------------------------------------------------------

SUBROUTINE PHMDZERO
  !***********************************
  ! Start PHMD dynamics 
  ! Overwite Temp_PHMD with FINATt
  !***********************************

  use chm_kinds
  use dimens_fcm
  use stream
  use number
  use consta
  use psf
  use reawri
  use energym
  implicit none

  INTEGER I
  real(chm_real) TMP,R1,R2,DELT
  IF (FINALT .NE. Temp_PHMD) THEN 
     Temp_PHMD = FINALT
     IF (PRNLEV .GE. 5) WRITE(OUTU, '(a,f8.1)')  &
          ' PHMD> Titration temp. changed to', &
          Temp_PHMD
  ENDIF
  EKINMAX = HALF * KBOLTZ * Temp_PHMD * NTitr 
  KAYTEE  = HALF * KBOLTZ * Temp_PHMD
  R1 = DSQRT(TWO * KAYTEE / QMass_PHMD)
  DELT    = TIMEST/TIMFAC

  NOSEQ(1)  = QMass_PHMD * MASS1_PHMD
  NOSEQ(2)  = QMass_PHMD * MASS2_PHMD
  NOSEQ(3)  = QMass_PHMD * MASS3_PHMD

  DO I=1,3
     NOSEV(I) = ZERO
     NOSE(I) = ONE
     NOSEOLD(I) = ONE
  ENDDO

  EPROP(PHKin) = ZERO
  IF (QSetLam) THEN
     DO I=1,NTitr
        TMP = ZERO
        vpH_Theta(I) = tmp
        TMP = pH_Theta(I)
        ThetaOld(I) = tmp
     ENDDO
  ELSE
     DO I=1,NTitr
        IF(PHBETA .gt. ZERO)THEN
          CALL GAUSSI(ZERO,R1,R2,ISEED,1)
          VPHOLD(I) = r2
        ENDIF
        CALL GAUSSI(ZERO,R1,R2,ISEED,1)
        vpH_Theta(I) = r2
        TMP = pH_Theta(I)
        TMP = TMP - DELT * R2
        ThetaOld(I) = tmp
        EPROP(PHKin) = EPROP(PHKin) + HALF*QMASS_PHMD*R2*R2
     ENDDO
  ENDIF
  RETURN
END SUBROUTINE PHMDZERO

#endif /*       (phmd_outer)*/
end module phmd
