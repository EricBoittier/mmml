module dcntrl_mod
  implicit none

contains

  SUBROUTINE DYNOPT(COMLYN, COMLEN &
#if KEY_LIBRARY == 1
       , options &
       , in_vx, in_vy, in_vz &
       , out_vx, out_vy, out_vz &
#endif /* KEY_LIBRARY */
)
  !4/4/90 - switched shkapr from 2Xn to nX2. FORTRAN matrices
  ! go collumn-wise - Steve F.
  !
  !     Process the DYNAMICS options.
  !

  use new_timer,only:timer_start,timer_stop, T_dcntrl
#if KEY_CHEQ==1
  use cheq,only: qcg,qnoco, cgmodel,  cgeq, qcginv, qcginvf,      &
       qcgwater,iocg,qpolar1,ipolar1,checketa,checkqnorm,qpartbin, &
       allocate_cheq
#endif
#if KEY_RMD==1
  use cross, only: allrmd,freermd,NCRUN
#endif
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use memory
  use number
#if KEY_DHDGB==1
!AP/MF
  use dhdgbc
  use dhdgb,only:totals
#endif
  !
#if KEY_ACE==1
  use ace_module,only:lace,bsolv
#endif
  use bases_fcm
  use contrl
  use coordc
  use cvio
  use fourdm
  use image
  use inbnd
  use psf
  use shake
  use stream
  use string
  use euler
  use nose_mod
  use pert
  use parallel
  use tbmts
#if KEY_TSM==1
  use tsms_mod
  use tsmh
#endif
#if KEY_PIPF==1
  use pipfm
#endif
  use phmd
#if KEY_FLUCQ==1
  use flucq
#endif
#if KEY_TMD==1
  use tmd
#endif
#if KEY_GRAPE==1
  use grape
#endif
#if KEY_DIMS==1
  use dims, only: closenmfile, freedims
#endif
#if KEY_DOMDEC==1
  use domdec,only:domdec_com
  use domdec_common,only:q_domdec,natoml,q_split
  use domdec_d2d_comm,only:copy_to_all
  use domdec_dr_common,only:copy_to_recip
  use coord,only:x,y,z
#endif

#if KEY_LIBRARY == 1
  use, intrinsic :: iso_c_binding, only: c_int, c_double
  use api_types, only: dynamics_settings
#endif

  implicit none
  ! . Passed variables.
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  ! . Local variables.
#if KEY_DHDGB==1
!AP/MF
  REAL(chm_real) VK_DHDGB(TOTALS)
#endif
  real(chm_real),allocatable,dimension(:) :: XOLD
  real(chm_real),allocatable,dimension(:) :: YOLD
  real(chm_real),allocatable,dimension(:) :: ZOLD
  real(chm_real),allocatable,dimension(:) :: XNEW
  real(chm_real),allocatable,dimension(:) :: YNEW
  real(chm_real),allocatable,dimension(:) :: ZNEW
#if KEY_DHDGB==1
!AP/MF
  real(chm_real),allocatable,dimension(:) :: SDEFOLD
  real(chm_real),allocatable,dimension(:) :: SDEFNEW
  real(chm_real),allocatable,dimension(:) :: VS_DHDGB
#endif
  real(chm_real),allocatable,dimension(:) :: VX
  real(chm_real),allocatable,dimension(:) :: VY
  real(chm_real),allocatable,dimension(:) :: VZ
  real(chm_real),allocatable,dimension(:) :: IGAMMA
#if KEY_CHEQ==1
  real(chm_real),allocatable,dimension(:) :: CGNEW
  real(chm_real),allocatable,dimension(:) :: CGOLD
  real(chm_real),allocatable,dimension(:) :: VCG
  integer,allocatable,dimension(:) :: FREECG
  INTEGER   J
  LOGICAL   QCHEQRDPRM, QCHEQNORM
#endif
!  logical lqstprt,qstprt
  ! PJ 06/2005
#if KEY_PIPF==1
  real(chm_real),allocatable,dimension(:,:) :: UINDN
  real(chm_real),allocatable,dimension(:,:) :: UINDO
  real(chm_real),allocatable,dimension(:,:) :: VUIND
  INTEGER   NUIMG1,NESAV,N3
#endif
  LOGICAL   ORIG,VVERL
#if KEY_DYNVV2==1
  LOGICAL   VVERL2
#endif
  real(chm_real),allocatable,dimension(:) :: IRFT
  real(chm_real),allocatable,dimension(:) :: IRFD
  INTEGER I,IRFTlen

#if KEY_LIBRARY == 1
  ! args
  type(dynamics_settings), optional, intent(in) :: options
  real(c_double), dimension(:), optional :: &
    in_vx, in_vy, in_vz, &
    out_vx, out_vy, out_vz
#endif /* KEY_LIBRARY */

  !     begin

#if KEY_ENSEMBLE==1
  CALL ENSCHK
#endif
  !
  DYNAMQ=.TRUE.

  ! . Branch on the option.
  IF(INDXA(COMLYN, COMLEN, 'FORM')  >  0) THEN
     CALL DYNFOR
     RETURN
  ELSE IF(INDXA(COMLYN, COMLEN, 'UNFO')  >  0) THEN
     CALL DYNUFO
     RETURN
  ENDIF
  !
  ! Parse the method option
#if KEY_LIBRARY == 0
  ILANG = 0
  IREST = 0
#endif

  IF (INDXA(COMLYN, COMLEN, 'STRT')  >  0) IREST = 0
  IF (INDXA(COMLYN, COMLEN, 'STAR')  >  0) IREST = 0
  IF (INDXA(COMLYN, COMLEN, 'REST')  >  0) IREST = 1
#if KEY_TPS==1
  IF (INDXA(COMLYN, COMLEN, 'RTRJ')  >  0) IREST = 2
#endif
  IF (INDXA(COMLYN, COMLEN, 'VERL')  >  0) ILANG = 0
#if KEY_FOURD==1
  IF (INDXA(COMLYN, COMLEN, 'VER4')  >  0) THEN
     if(.not.allocated(fdim)) then
       call allocate_fourd_ltm()
     endif
     ILANG = 0
     DIM4  = .TRUE.
     REST4 = IREST
     IF (INDXA(COMLYN, COMLEN, 'STR4')  >  0) REST4 = 0
  ENDIF
#endif
  IF (INDXA(COMLYN, COMLEN, 'LANG')  >  0) ILANG = 1
  VVERL=.FALSE.
  IF (INDXA(COMLYN, COMLEN, 'VVER')  >  0) VVERL = .TRUE.
#if KEY_DYNVV2==1 /*dynvv2_select*/
  ! BEGIN DYNA VV2 (G. Lamoureux)
  VVERL2 = .FALSE.
  IF (INDXA(COMLYN, COMLEN, 'VV2')  >  0) THEN
     VVERL = .TRUE.
     VVERL2 = .TRUE.
  ENDIF
  ! END DYNA VV2 (G. Lamoureux)
#else /* (dynvv2_select)*/
  IF (INDXA(COMLYN, COMLEN, 'VV2')  >  0) THEN
     CALL WRNDIE(0,'<DYNOPT>','DYNA VV2 not compiled')
     RETURN
  ENDIF
#endif /* (dynvv2_select)*/
  ! which integrator to use?
  !
  ! Switch to leapfrog integrator as default. BRB - July,1995
  ORIG = .FALSE.
  !
  IF (INDXA(COMLYN, COMLEN, 'ORIG')  >  0) ORIG = .TRUE.
  IF (INDXA(COMLYN, COMLEN, 'LEAP')  >  0) ORIG = .FALSE.
  IF (INDXA(COMLYN, COMLEN, 'NEW')  >  0) ORIG = .FALSE.
  IF (INDXA(COMLYN, COMLEN, 'CPT')  >  0) ORIG = .FALSE.

#if KEY_MTS==1
  ! A. Sandu .......... LN specific input ......
  QLNX = .FALSE.
  QIMPULSE = .FALSE.
  IF (INDXA(COMLYN, COMLEN, 'LNX')  >  0) THEN
     QLNX  = .TRUE.
     ILANG =  1
  END IF
  IF (INDXA(COMLYN, COMLEN, 'IMPU')  >  0) THEN
     QIMPULSE = .TRUE.
  END IF
  ! A. Sandu ..........
  !
  IF (QTBMTS .AND. (.NOT. VVERL) .AND. (.NOT. QLNX)) VVERL = .TRUE.
#endif

  !CC parse NOSE option here to properly set flags - BRB
  !CC QNOSP is already set by the NOSE command.
  QNOSE = INDXA(COMLYN,COMLEN,'NOSE')  >  0
  !
  !ds...B981203.ds2 DYNAMLN calling conditional fix
  !ds      IF (VVERL .OR. QNOSE .OR. QNOSP) ORIG = .TRUE.
#if KEY_MTS==0
  IF (VVERL .OR. QNOSE .OR. QNOSP) ORIG = .TRUE.
#else /**/
  IF (VVERL .OR. QLNX .OR. QNOSE .OR. QNOSP) ORIG = .TRUE.
#endif
#if KEY_PARALLEL==1
  QVSEND=INDXA(COMLYN,COMLEN,'VSEN') > 0
#endif
  ! GPU needs to know few things in advance
#if KEY_GRAPE==1
  lgrape=INDX(COMLYN,COMLEN,'GRAP',4) > 0
#endif
#if KEY_CHEQ==1
  !-------------- Charge Dynamics? -----------------------------
  ! work needed for partial CG fixing.
  QCG= (INDXA(COMLYN, COMLEN, 'CHEQ') >  0) .or. QCG
  IF (QCG) THEN
     if(.not.allocated(qpartbin)) then
        call wrndie(-1,'<dcntrl>','CHEQ not set-up')
     elseif(natim>natom) then
        call allocate_cheq(natim,ngrp)
     endif
     CALL CHECKETA(QCHEQRDPRM)
     IF (.not.QCHEQRDPRM) THEN
        if(prnlev > 1)write(outu,'(/a,l1,/a)') &
             '<DCNTRL> CHEQ DYNAMICS HAS BEEN REQUESTED: CHEQ ' &
             ,QCG, &
             '         CORRECT PARAMETERS HAVE NOT BEEN READ'
        CALL WRNDIE(-1,'<DCNTRL>', &
             'CHEQ PARAMETERS HAVE NOT BEEN READ')
     ENDIF
     !     -----  SINCE USER WANTS CHEQ, MAKE SURE PARMS ARE CORRECT
     CALL CHECKQNORM(QCHEQNORM)
     IF (.not.QCHEQNORM) THEN
        if(prnlev > 1)write(outu,'(2a)') &
             'CHEQ DYNAMICS HAS BEEN REQUESTED BUT', &
             ' CORRECT NORMALIZATION SPECIFICATIONS LACKING'
        CALL WRNDIE(-1,'<DCNTRL>', &
             'CHEQ NORMALIZATION ASSIGNMENT INCORRECT')
     ENDIF
     !     ----   SINCE USER WANTS CHEQ, MAKE SURE NORMALIZATION IS SET UP AND CORRECT
     QNOCO=(INDXA(COMLYN, COMLEN, 'NOCO') >  0)
     CGMODEL=GTRMI(COMLYN,COMLEN,'CGMD',CGMODEL)
     CGEQ=GTRMI(COMLYN,COMLEN,'CGEQ',0)
     QCGINV = INDXA(COMLYN, COMLEN, 'CGIN')  >  0
     QCGINVF = INDXA(COMLYN, COMLEN, 'CGFC')  >  0
     QCGWATER=INDXA(COMLYN, COMLEN, 'WATE') > 0
     IF (QCG) IOCG=0
     QPOLAR1=INDXA(COMLYN, COMLEN, 'QPOL')  >  0
     IPOLAR1=GTRMI(COMLYN,COMLEN,'IPOL',IPOLAR1)
     IF (PRNLEV > 0) THEN
        IF (QCGINV) &
             WRITE(OUTU,'(a)')'<DYNA>: CG INVERSION REQUESTED'
        IF (QCGINVF) &
             WRITE(OUTU,'(a)')'<DYNA>: CG FORCE REQUESTED'
        IF (CGMODEL /= 0)WRITE(OUTU,'(a,i6)') &
             '<DYNA>: CGMODEL SET TO',CGMODEL
        IF (QPOLAR1) WRITE(OUTU,'(2A,I5)') &
             'Single Molecule Polarization dynamics requested for ' &
             ,'Molecule #',IPOLAR1
        IF (QNOCO) WRITE(OUTU,'(A)') &
             'Charge dynamics only, coordinates being fixed.'
        IF (QCGWATER) WRITE(OUTU,'(A)') &
             'Contribution of Hardness Matrix to Force set to zero'
     ENDIF
  ENDIF
#endif
  !
  !
  ! Allocate temporary storage space.
  !
#if KEY_RMD==1
  IF (NCRUN > 0) CALL ALLRMD(NATOM)
#endif
  !
#if KEY_TNPACK==1
  QEULER=.FALSE.
  IF(INDXA(COMLYN, COMLEN, 'EULE')  >  0) THEN
     ORIG = .FALSE.
     ILANG=1
     QEULER=.TRUE.
     call chmalloc('dcntrl.src','DYNOPT','KEHARM',NATOM,crl=KEHARM)
     call chmalloc('dcntrl.src','DYNOPT','RXHARM',NATOM,crl=RXHARM)
     call chmalloc('dcntrl.src','DYNOPT','RYHARM',NATOM,crl=RYHARM)
     call chmalloc('dcntrl.src','DYNOPT','RZHARM',NATOM,crl=RZHARM)
     call chmalloc('dcntrl.src','DYNOPT','IHHARM',NATOM,intg=IHHARM)

     IHHARM(1:NATOM)=1
     LIEFF = GTRMF(COMLYN,COMLEN,'LIEF',LIEFF)
     QESTRT=.TRUE.
  ENDIF
#endif
  IF(INDXA(COMLYN, COMLEN, 'NOLA')  >  0) ILANG = 0
  !
  call chmalloc('dcntrl.src','DYNOPT','XOLD',NATOM,crl=XOLD)
  call chmalloc('dcntrl.src','DYNOPT','YOLD',NATOM,crl=YOLD)
  call chmalloc('dcntrl.src','DYNOPT','ZOLD',NATOM,crl=ZOLD)
  call chmalloc('dcntrl.src','DYNOPT','XNEW',NATOM,crl=XNEW)
  call chmalloc('dcntrl.src','DYNOPT','YNEW',NATOM,crl=YNEW)
  call chmalloc('dcntrl.src','DYNOPT','ZNEW',NATOM,crl=ZNEW)
#if KEY_DHDGB==1
!AP/MF
  call chmalloc('dcntrl.src','DYNOPT','SDEFOLD',TOTALS,crl=SDEFOLD)
  call chmalloc('dcntrl.src','DYNOPT','SDEFNEW',TOTALS,crl=SDEFNEW)
  call chmalloc('dcntrl.src','DYNOPT','VS_DHDGB',TOTALS,crl=VS_DHDGB)
#endif
  call chmalloc('dcntrl.src','DYNOPT','VX',NATOM,crl=VX)
  call chmalloc('dcntrl.src','DYNOPT','VY',NATOM,crl=VY)
  call chmalloc('dcntrl.src','DYNOPT','VZ',NATOM,crl=VZ)

  XOLD=ZERO
  YOLD=ZERO
  ZOLD=ZERO
  XNEW=ZERO
  YNEW=ZERO
  ZNEW=ZERO
#if KEY_DHDGB==1
!AP/MF
  SDEFOLD=ZERO
  SDEFNEW=ZERO
  VS_DHDGB=ZERO
#endif
  VX=ZERO
  VY=ZERO
  VZ=ZERO
#if KEY_CHEQ==1
  call chmalloc('dcntrl.src','DYNOPT','CGNEW',NATOM,crl=CGNEW)
  call chmalloc('dcntrl.src','DYNOPT','CGOLD',NATOM,crl=CGOLD)
  call chmalloc('dcntrl.src','DYNOPT','VCG',NATOM,crl=VCG)
  J= NATOM
  IF (QCG) THEN
     !  assign space for free cg list
     !  this is for future development. No use at the moment
     IF (IOCG > 0) THEN
        J=NATOM
     ELSE
        J=1
     ENDIF
  ENDIF
  call chmalloc('dcntrl.src','DYNOPT','FREECG',J,intg=FREECG)
#endif
  ! PJ 06/2005
#if KEY_PIPF==1
  IF (QPIPF) THEN
     IF (QPFDYN) THEN
        NUIMG1 = MAXAIM
        IF (QUEANG .AND. (NPFIM  >  NPFPR)) THEN
           NESAV = MAXAIM
        ELSE
           NESAV = 1
        ENDIF
        call chmalloc('dcntrl.src','DYNOPT','UIND', 3,NUIMG1,crl=UIND)
        call chmalloc('dcntrl.src','DYNOPT','UINDN',3,NATOM ,crl=UINDN)
        call chmalloc('dcntrl.src','DYNOPT','UINDO',3,NATOM ,crl=UINDO)
        call chmalloc('dcntrl.src','DYNOPT','DUIND',3,NUIMG1,crl=DUIND)
        call chmalloc('dcntrl.src','DYNOPT','VUIND',3,NATOM ,crl=VUIND)
        call chmalloc('dcntrl.src','DYNOPT','IESAV',3,NESAV ,crl=IESAV)
     ENDIF


     !
     ! ASSIGN POINT OF INDUCED DIPOLE IN COMMON BLOCK, SO THE PIPF
     ! ENERGY ROUTINE ARE ABLE TO USE THEM IN THE DYNAMICAL
     ! DIPOLE METHOD
     IF(PFMODE  ==  1) THEN
        N3 = MAXAIM*3
        QFSTDP = .TRUE.
        call chmalloc('dcntrl.src','DYNOPT','INDP',3,MAXAIM,crl=INDP)
     ENDIF
  ENDIF
#endif
  !.  assign velocity with comp.
#if KEY_DHDGB==1
!AP/MF
  VS_DHDGB(1:TOTALS)=SCOMP(1:TOTALS)
#endif

#if KEY_LIBRARY == 1
  if (present(in_vx)) then
     VX(1:natom)=in_vx(1:natom)
     VY(1:natom)=in_vy(1:natom)
     VZ(1:natom)=in_vz(1:natom)
  else
#endif /* KEY_LIBRARY */
     VX(1:natom)=XCOMP(1:natom)
     VY(1:natom)=YCOMP(1:natom)
     VZ(1:natom)=ZCOMP(1:natom)
#if KEY_LIBRARY == 1
  end if
#endif

#if KEY_OLDDYN==0
  IF(ORIG .AND. .NOT.QNOSE .AND. .NOT.QNOSP .AND. .NOT.VVERL) THEN
     CALL WRNDIE(-1,'<DCNTRL>','OLD code is not compiled.')
     ORIG=.FALSE.
  ENDIF
#endif
  IF(ORIG) THEN
     call chmalloc('dcntrl.src','DYNOPT','IGAMMA',NATOM,crl=IGAMMA)
     IRFTlen=2*((NATOM*3+1)/2)
     call chmalloc('dcntrl.src','DYNOPT','IRFT',IRFTlen,crl=IRFT)
     call chmalloc('dcntrl.src','DYNOPT','IRFD',NATOM,crl=IRFD)
  ELSE
     call chmalloc('dcntrl.src','DYNOPT','IGAMMA',4*NATOM,crl=IGAMMA)
  ENDIF

  IF (WRNLEV  >  5) WRITE(OUTU,851) IREST,(1-ILANG),ILANG,NCONST
851 FORMAT(/10X,'CALLING DCNTRL TO DO DYNAMICS' &
       /10X,'  IREST = ',I2,'  IVERL = ',I2,'  ILANG = ',I2, &
       ' NSHAKE = ',I2)

  ! Call the main parsing routine
  !
  call timer_start(T_dcntrl)

  CALL DCNTRL(COMLYN,COMLEN,XOLD,YOLD,ZOLD,XNEW,YNEW,ZNEW,VX,VY,VZ,WCOMP, &
#if KEY_CHEQ==1
       CGNEW,CGOLD,VCG,FREECG, &
#endif
#if KEY_PIPF==1
       UINDN,UINDO,VUIND, &
#endif
       BNBND, BIMAG, &
       IGAMMA,IRFT,IRFD,ORIG,VVERL &
#if KEY_DYNVV2==1
       ,VVERL2    &
#endif
#if KEY_TSM==1
       ,REACLS,PRODLS,PIGGLS,BACKLS &
#endif
#if KEY_TNPACK==1
       ,QEULER,QESTRT,QEHARM,KEHARM,RXHARM,RYHARM,RZHARM,LIEFF &
#endif
#if KEY_ACE==1
       ,LACE,BSOLV        &
#endif
#if KEY_DHDGB==1
!AP/MF
       ,SDEFOLD,SDEFNEW,VS_DHDGB,VK_DHDGB  &
#endif
#if KEY_LIBRARY == 1
       , options &
#endif
       )

  call timer_stop(T_dcntrl)

  ! Save final velocities in the comparison coordinates
#if KEY_LIBRARY == 1
  if (present(out_vx)) then
     out_vx(1:natom) = vx(1:natom)
     out_vy(1:natom) = vy(1:natom)
     out_vz(1:natom) = vz(1:natom)
  else
#endif /* KEY_LIBRARY */
     xcomp(1:natom) = vx(1:natom)
     ycomp(1:natom) = vy(1:natom)
     zcomp(1:natom) = vz(1:natom)
#if KEY_LIBRARY == 1
  end if
#endif /* KEY_LIBRARY */

#if KEY_DHDGB==1
!AP/MF
  SCOMP(1:TOTALS)=VS_DHDGB(1:TOTALS)
#endif
  !
  ! Free all temporary space
  call chmdealloc('dcntrl.src','DYNOPT','XOLD',NATOM,crl=XOLD)
  call chmdealloc('dcntrl.src','DYNOPT','YOLD',NATOM,crl=YOLD)
  call chmdealloc('dcntrl.src','DYNOPT','ZOLD',NATOM,crl=ZOLD)
  call chmdealloc('dcntrl.src','DYNOPT','XNEW',NATOM,crl=XNEW)
  call chmdealloc('dcntrl.src','DYNOPT','YNEW',NATOM,crl=YNEW)
  call chmdealloc('dcntrl.src','DYNOPT','ZNEW',NATOM,crl=ZNEW)
#if KEY_DHDGB==1
!AP/MF
  call chmdealloc('dcntrl.src','DYNOPT','SDEFNEW',TOTALS,crl=SDEFNEW)
  call chmdealloc('dcntrl.src','DYNOPT','SDEFOLD',TOTALS,crl=SDEFOLD)
  call chmdealloc('dcntrl.src','DYNOPT','VS_DHDGB',TOTALS,crl=VS_DHDGB)
#endif
  call chmdealloc('dcntrl.src','DYNOPT','VX',NATOM,crl=VX)
  call chmdealloc('dcntrl.src','DYNOPT','VY',NATOM,crl=VY)
  call chmdealloc('dcntrl.src','DYNOPT','VZ',NATOM,crl=VZ)

#if KEY_DOMDEC==1
  ! Copy coordinates to all nodes
  if (q_domdec) then
     call copy_to_all(x, y, z)
  endif
#endif

#if KEY_DOMDEC==1
  if (q_split) then
     call copy_to_recip(x, y, z)
  endif
#endif

#if KEY_CHEQ==1
  IF (QCG) THEN
     call chmdealloc('dcntrl.src','DYNOPT','CGNEW',NATOM,crl=CGNEW)
     call chmdealloc('dcntrl.src','DYNOPT','CGOLD',NATOM,crl=CGOLD)
     call chmdealloc('dcntrl.src','DYNOPT','VCG',NATOM,crl=VCG)
     call chmdealloc('dcntrl.src','DYNOPT','FREECG',J,intg=FREECG)
  ENDIF
#endif
  ! PJ 06/2005
#if KEY_PIPF==1
  IF (QPIPF) THEN
     IF (QPFDYN) THEN
        NUIMG1 = MAXAIM
        IF (QUEANG .AND. (NPFIM  >  NPFPR)) THEN
           NESAV = MAXAIM
        ELSE
           NESAV = 1
        ENDIF
        call chmdealloc('dcntrl.src','DYNOPT','UIND', 3,NUIMG1,crl=UIND)
        call chmdealloc('dcntrl.src','DYNOPT','UINDN',3,NATOM ,crl=UINDN)
        call chmdealloc('dcntrl.src','DYNOPT','UINDO',3,NATOM ,crl=UINDO)
        call chmdealloc('dcntrl.src','DYNOPT','DUIND',3,NUIMG1,crl=DUIND)
        call chmdealloc('dcntrl.src','DYNOPT','VUIND',3,NATOM ,crl=VUIND)
        call chmdealloc('dcntrl.src','DYNOPT','IESAV',3,NESAV ,crl=IESAV)
     ENDIF
  ENDIF
#endif
  !
  IF(ORIG) THEN
     call chmdealloc('dcntrl.src','DYNOPT','IGAMMA',NATOM,crl=IGAMMA)
     call chmdealloc('dcntrl.src','DYNOPT','IRFT',IRFTlen,crl=IRFT)
     call chmdealloc('dcntrl.src','DYNOPT','IRFD',NATOM,crl=IRFD)
  ELSE
     call chmdealloc('dcntrl.src','DYNOPT','IGAMMA',4*NATOM,crl=IGAMMA)
  ENDIF
  !
#if KEY_TNPACK==1
  IF(QEULER) THEN
     ! remove Euler flag and release space.
     QEULER=.FALSE.
     call chmdealloc('dcntrl.src','DYNOPT','KEHARM',NATOM,crl=KEHARM)
     call chmdealloc('dcntrl.src','DYNOPT','RXHARM',NATOM,crl=RXHARM)
     call chmdealloc('dcntrl.src','DYNOPT','RYHARM',NATOM,crl=RYHARM)
     call chmdealloc('dcntrl.src','DYNOPT','RZHARM',NATOM,crl=RZHARM)
     call chmdealloc('dcntrl.src','DYNOPT','IHHARM',NATOM,intg=IHHARM)
  ENDIF
#endif
  !
  !
  ! BEGIN TPCONTROL (G. Lamoureux)
  !     IF(QNOSE.OR.QNOSP) THEN
  !     For TPCONTROL, this has to be done explicitly with "TPCONTROL OFF"
  IF((QNOSE.OR.QNOSP) .AND. .NOT.QTPCON) THEN
     ! END TPCONTROL (G. Lamoureux)
     ! remove Nose flag and release space.
     QNOSE=.FALSE.
     QNOSP=.FALSE.
     call chmdealloc('dcntrl.src','DYNOPT','INLCKP',NATOM,intg=INLCKP)
  ENDIF
  !
#if KEY_MTS==1
  IF (QTBMTS) THEN
     ! remove all MTS spaces
     CALL DFININ3
     IF(PRNLEV >= 2) WRITE(OUTU,910)
  ENDIF
910 FORMAT(/,3X,'MTS> MTS spaces are cleared')
#endif
#if KEY_TMD==1
  !     Remove TMD temporary space
  call freetmd
#endif
#if KEY_RMD==1
  IF (NCRUN > 0) CALL FREERMD
#endif
#if KEY_DIMS==1
  !     Remove DIMS temporary space
  CALL CLOSENMFILE
  CALL FREEDIMS(2)
#endif
#if KEY_CHEQ==1
  !    ------------Reset to defaults except QCG -----------------
  QNOCO    = .FALSE.
  CGMODEL = 1
  CGEQ     = 0
  QCGINV   = .FALSE.
  QCGINVf  = .FALSE.
  QCGWATER = .FALSE.
  QPOLAR1  = .FALSE.
  IPOLAR1  = -1
#endif
  !
  RETURN
END SUBROUTINE DYNOPT

subroutine dyn_parse_iseed(comlyn, comlen, &
     irest, ilang, iseed)
  use chm_kinds, only: chm_int4
  use rndnum, only: &
       nrand, rngseeds, irndsd, &
       qoldrandom, qbrokenclcg, rngchoice
  use clcg_mod, only: clcginit, rngmodseeds
  use string, only: indx, gtrmim

  use parallel, only: mynod
  
#if KEY_ABPO == 1
  use abpo_ltm, only: q_abpo
  use ensemble, only: old_mynod
#endif

#if KEY_REPDSTR==1
    use repdstr, only: qrexchg, irepdstr
#endif
  
  implicit none

  ! input params
  character(len=*) :: comlyn
  integer :: comlen, irest, ilang, iseed

  ! local vars
  integer :: lrngseeds(nrand), i
  logical :: qpresent
  integer(chm_int4) :: count4, rate4, maxcount4

  qpresent = .false.
  lrngseeds = 0
  
  IF(IREST == 1.AND.ILANG == 1.AND.               & !lni_080627
       INDX(COMLYN,COMLEN,'ISEE',4) > 0)  &
       CALL WRNDIE(-1,'<DCNTRL>', &
       'REUSE OF RANDOM SEED IS NOT RECOMMENDED')
  !!      ISEED=GTRMI(COMLYN,COMLEN,'ISEE',ISEED)
  !      if (.not.qoldrng) then     !yw 05-Aug-2008
  !         CALL CLCGINIT(ISEED)
  !         ISEED=1
  !      endif
  !
  ! call chmalloc('dcntrl.src','DCNTRL','lrngseeds',Nrand,intg=lrngseeds)
  lrngseeds(1:nrand)=rngseeds(1:nrand)
  if(qoldrandom.or.qbrokenclcg)lrngseeds(1:nrand)=iseed
  call gtrmim(nrand,comlyn,comlen,'ISEE',lrngseeds,rngseeds,qpresent)
#if KEY_REPDSTR==1
  ! wxw: set different iseeds for different replicas
  ! DRR: Since each replica now has their own COMM_CHARMM, do
  !      not set seed according to mynodg since this will lead
  !      to results that vary by processor count. Replace mynodg
  !      with irepdstr to be at least consistent with previous results
  !      with 1 thread per replica.
  IF(QREXCHG) THEN
     DO I=1,NRAND
        RNGSEEDS(I)=RNGSEEDS(I)+31415926*(IREPDSTR*IREPDSTR+I)
     ENDDO
  ENDIF
#endif
  if(.not. (qoldrandom.or.qbrokenclcg) )then
     ! For STARTs we generate new random seeds from system clock
     ! LNilsson October 2010
     if( irest == 0 .and. .not. qpresent)then
        call system_clock(count4,rate4,maxcount4)
#if KEY_ABPO==1
        if (q_abpo) then
           rngseeds(1:nrand) = count4 + 50000*old_mynod
        else
#endif /* KEY_ABPO */
           rngseeds(1:nrand) = count4 + 50000*mynod
#if KEY_ABPO==1
        end if
#endif
        if(rngchoice == 1)then
           irndsd = 1380662
           call clcginit(irndsd)
        elseif(rngchoice == 2)then
           call random_seed(put=rngseeds)
        endif
     endif
  endif
  call rngmodseeds(qpresent,iseed)
  ! call chmdealloc('dcntrl.src','DCNTRL','lrngseeds',Nrand,intg=lrngseeds)
end subroutine dyn_parse_iseed

#if KEY_LIBRARY == 1
subroutine dyn_init_rng(irest, ilang)
  use chm_kinds, only: chm_int4
  use rndnum, only: nrand, rngseeds, qapi_seed_set
  use parallel, only: mynod
  
#if KEY_ABPO == 1
  use ensemble, only: old_mynod
#endif

#if KEY_REPDSTR==1
    use repdstr, only: qrexchg, irepdstr
#endif
  
  implicit none

  ! input params
  integer :: irest, ilang

  ! local vars
  integer :: i
  integer(chm_int4) :: count4, rate4, maxcount4

  if (irest == 1 .and. ilang == 1 .and. qapi_seed_set) then
     call wrndie(-1, '<DCNTRL>', &
          'REUSE OF RANDOM SEED IS NOT RECOMMENDED')
  end if
    
#if KEY_REPDSTR==1
  ! wxw: set different iseeds for different replicas
  ! DRR: Since each replica now has their own COMM_CHARMM, do
  !      not set seed according to mynodg since this will lead
  !      to results that vary by processor count. Replace mynodg
  !      with irepdstr to be at least consistent with previous results
  !      with 1 thread per replica.
  if(qrexchg) then
     do i = 1, nrand
        rngseeds(i) = rngseeds(i) + 31415926 * (irepdstr * irepdstr + i)
     end do
  end if
#endif
  
  ! For STARTs we generate new random seeds from system clock
  ! LNilsson October 2010
  if (irest == 0 .and. .not. qapi_seed_set) then
     call system_clock(count4, rate4, maxcount4)
#if KEY_ABPO==1
     if (q_abpo) then
        rngseeds(1:nrand) = count4 + 50000 * old_mynod
     else
#endif /* KEY_ABPO */
        
        rngseeds(1:nrand) = count4 + 50000 * mynod
        
#if KEY_ABPO==1
     end if
#endif
  end if
  
  call random_seed(put = rngseeds)
  qapi_seed_set = .false.
end subroutine dyn_init_rng
#endif

SUBROUTINE DCNTRL(COMLYN,COMLEN,XOLD,YOLD,ZOLD, &
     XNEW,YNEW,ZNEW,VX,VY,VZ,VK, &
#if KEY_CHEQ==1
     CGNEW,CGOLD,VCG,FREECG,               &
#endif
#if KEY_PIPF==1
     UINDN,UINDO,VUIND,                    &
#endif
     BNBND,BIMAG,GAMMA,RFT,RFD,ORIG,VVERL  &
#if KEY_DYNVV2==1
     ,VVERL2                               &
#endif
#if KEY_TSM==1
     ,REACLS,PRODLS,PIGGLS,BACKLS          &
#endif
#if KEY_TNPACK==1
     ,QEULER,QESTRT,QEHARM,KEHARM          &
#endif
#if KEY_TNPACK==1
     ,RXHARM,RYHARM,RZHARM,LIEFF           &
#endif
#if KEY_ACE==1
     ,QACE,BSARR                           &
#endif
#if KEY_DHDGB==1
     ,SDEFOLD,SDEFNEW,VS_DHDGB,VK_DHDGB    &
#endif
#if KEY_LIBRARY
     , options &
#endif /* KEY_LIBRARY */
     )
  !
  !     DCNTRL DISPATCHES THE DYNAMICS LOOPS.
  !
  !     The arguements are:
  !
  !             COMLYN,COMLEN - the command line and its length.  This
  !                     command line is parsed for the various mode
  !                     and parameter settings.
  !             X,Y,Z - The coordinates.
  !             VX,VY,VZ - The velocities in dynamics.
  !             BNBND    - The nonbond data structure base
  !             BIMAG    - The images data structure base
  !
  !     The parameter, code, and internal coordinate lists are all
  !     obtained from common blocks.
  !
  !     much of the command code for updating hydrogen bond and non-bonded
  !     lists is here as well as some of the i/o routines.
  !
  !     CONTROL FLOW PUT INTO FLECS 1/8/81 DJS
  !

#if KEY_LIBRARY == 1
  use, intrinsic :: iso_c_binding, only: c_int, c_double
  use api_types, only: dynamics_settings
  use api_custom, only: custom_dynam_call, custom_dynam_is_set
#endif

  use pme_module,only:qpme
  use new_timer,only:timer_stpstrt,timer_start,timer_stop,  &
       T_dcntrl,T_10extra
#if KEY_CHEQ==1
  use cheq,only: qcg,massq,qfrq,noseflag,tcg, qnoco,pmassq,   &
#endif
#if KEY_CHEQ==1
       cgeq,cheqbasetup,massqd,tcgdef,   &
#endif
#if KEY_CHEQ==1
       DCH,SUMDCH,DDCH
#endif
  use cons_rmsd,only:qrmsd,qrmsddyn,urmsd,irmsdcll
#if KEY_REPDSTR==1
  use repdstrmod
#endif
#if KEY_FLUCQ==1
  use flucqm, only: fqdpar
#endif

#if KEY_RMD==1
  use cross,only:SAVEDSEED,XTIME,QXTIME
#endif
#if KEY_TSALLIS==1
  use tsallis_module, only: qtsall, qttsall, qtalpha, tsalpha, &
       tsq, tsbeta, tsemin
#endif
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
#if KEY_SCCDFTB==1
  use blockscc_fcm
#endif
#if KEY_GRAPE==1
    use grape,only:lfmm
#endif
#if KEY_SCCDFTB==1
  use sccdftb
#endif
  !
  use cnst_fcm
  use coord
  use contrl
  use consta
  use cveloci_mod
  use cvio
  use deriv
  use dynio
  use energym
  use eutil
  use fourdm
  use hbondm
  use image
  use inbnd
  use mmfp
  use param
  use psf
  use pert
  use pert_mod
  use code
  use rndnum
  use reawri
  use sbound
  use shake
  use stream
  use string
  use timerm
  use lonepr
  use nose_mod
  use parallel
#if KEY_RXNCONS==1
  use rxncons
#endif
  use shapes
  use tbmts
  use ctitla
  use repdstr    ! mh050712
#if KEY_DHDGB==1
!AP/MF
  use dhdgb,only:sdef,samass,qfhdgb,totals
  use derivdhdgb
#endif
#if KEY_TSM==1
  !!tsm mod djt/shf 4/4/90
  use machdep
  use tsms_mod
  use icfix
  use icpert
#endif /*  TSM*/
#if KEY_DIMS==1
  use dims
#endif
#if KEY_PIPF==1
  use pipfm
#endif
  use block_fcm
#if KEY_BLOCK==1
!ldm
  use lambdam, only : qldm, qlmc, qmld, rstp, ilaldm, &
       iunldm, itempld, ntempld, ilapot, ilaf, iresd, &
       bixlam, bldold, bivlam, &
       print_gbias, &
       nbiasv, &
       msld_ndegf, &
       ldm_init_qmcfr, &
       ilaldm, gammatheta, tbld, thetabib, thetam, igammald, qthetadm, &
       biblam,bimlam, &
       nsavl, &
       qmcfr, wangf, wangfi, mcpro, mccount, &
       msld_assignvelocities

#endif
#if KEY_FLUCQ==1
  use flucq
#endif
  use phmd
#if KEY_ENSEMBLE==1
  use ensemble
#endif
#if KEY_MULTICOM==1
  use multicom_aux
#endif
#if KEY_TMD==1
  use tmd
#endif
! #if KEY_MOBHY==1
  use clcg_mod,only:clcginit,rngmodseeds,oldrandom
  use mobhy_mod
! #else
!   use clcg_mod,only:clcginit,rngmodseeds
! #endif

#if KEY_SGLD==1
  use sgld,only: SGTYPE,SGAVG0,SGAVG1,SGAVP0,SGAVP1,SGEXP, &
#endif
#if KEY_SGLD==1
       TSGAVG,TSGAVP,TEMPSG,SGFT,SGFF,SGFD,TSGSET,SGSIZE, &    
#endif
#if KEY_SGLD==1
       QSGLD,QSGMD,QSGBZ,QSGCOM,&           
#endif
#if KEY_SGLD==1
       ISGSTA,ISGEND,psgld,sgfree
#endif
  use cstran_mod,only:mkfrat
  use dims
#if KEY_FOURD==1
  use dyn4,only: dynamc4
#endif
  use dynvv2
#if KEY_TPS==1
  use tps,only:tpmain,tpsrst,tpswri,tpcini,ntpath, &
       tps_parse,tpinit,tpsacc,tpfree, tpsstp, &
       nstsav,ibasin
#endif
! for TR-BOMD.
#if KEY_SQUANTM==1
  use squantm, only : LTRBOMD,q_apply_tr_bomd,i_md_step,i_md_step_old
#endif
  use consph, only: dophmc,phrdrstrt,phwrirstrt,write_ph_state,tstate,phrsvrunum
#if KEY_OPENMM==1
  use omm_glblopts, only : qtor_repex, torsion_lambda
  use omm_dynopts, only : omm_dynopts_t, omm_parse_options, omm_report_options
  use omm_ctrl, only : omm_requested
  use omm_main, only : omm_dynamics, omm_change_lambda
#endif
#if KEY_BLADE==1
  use blade_ctrl_module, only : blade_requested
  use blade_dynopts, only : blade_dynopts_t, blade_parse_options, blade_report_options
  use blade_main, only : blade_dynamics
#endif
#if KEY_DOMDEC==1
  use domdec_common,only:natoml,atoml,q_domdec, q_split
  use domdec_d2d_comm,only:copy_to_all
  use domdec_dr_common,only:q_direct_node, q_recip_node, start_split_direct_recip, &
       stop_split_direct_recip
  use domdec_d2r_comm,only:send_stop_recip
  use domdec,only:build_groupl
  use ewald,only:parse_ewald,lewald
  use energy_util,only:energy_recip
#endif
  use prssre
#if KEY_ABPO==1
  use abpo_ltm
#endif
#if KEY_STRINGM==1 /*  VO stringm : when using replica exchange need to keep restart files open */
  use sm_config, only : repl_x_on, repl_x_freq, smcv_on
  use ftsm_var, only : ftsm_on
#endif
  use param_store, only: set_param
  use dynutil, only: assvel
  use cstuff, only: fsystem
  implicit none

#if KEY_LIBRARY == 1
  ! args
  type(dynamics_settings), optional, intent(in) :: options
#endif /* KEY_LIBRARY */

  real(chm_real),allocatable,dimension(:) :: XNSE,YNSE,ZNSE
#if KEY_MTS==1
  real(chm_real),allocatable,dimension(:) :: XMI,YMI,ZMI
  real(chm_real),allocatable,dimension(:) :: XMM,YMM,ZMM
#endif
  integer,allocatable,dimension(:) :: ISKP
  real(chm_real),allocatable,dimension(:) :: ISKPR
  integer,allocatable,dimension(:) :: FREEAT
  real(chm_real),allocatable,dimension(:) :: IXAVE
  real(chm_real),allocatable,dimension(:) :: IYAVE
  real(chm_real),allocatable,dimension(:) :: IZAVE
#if KEY_CHEQ==1
  real(chm_real),allocatable,dimension(:) :: ICGAVE
#endif

#if KEY_TMD==1
  integer,allocatable,dimension(:) :: iscr1
#endif
#if KEY_TMD==1
  real(chm_real),allocatable,dimension(:) :: iscr2
#endif
#if KEY_DHDGB==1
!AP/MF
  real(chm_real),allocatable,dimension(:) :: SDEFLD
#endif
  character(len=*) COMLYN
  INTEGER COMLEN
  real(chm_real) :: XOLD(:), YOLD(:), ZOLD(:)
  real(chm_real) :: XNEW(:), YNEW(:), ZNEW(:)
  real(chm_real) :: VX(:), VY(:), VZ(:), VK(:)
#if KEY_DHDGB==1
!AP/MF
  REAL(chm_real) :: VS_DHDGB(:)
  REAL(chm_real) :: SDEFOLD(:),SDEFNEW(:),VK_DHDGB(:)
#endif
  type(nonbondDataStructure) BNBND
  type(imageDataStructure) BIMAG

  real(chm_real) GAMMA(:),RFD(*)
  real(chm_real) RFT(*)
  LOGICAL ORIG,VVERL
#if KEY_DYNVV2==1
  LOGICAL VVERL2,QVVHOLO
#endif
#if KEY_TSM==1
  INTEGER REACLS(*),PRODLS(*),PIGGLS(*),BACKLS(*)
#endif
#if KEY_TNPACK==1
  LOGICAL QEULER,QEHARM,QESTRT
  real(chm_real) KEHARM(*),RXHARM(*),RYHARM(*),RZHARM(*),LIEFF
#endif
#if KEY_ACE==1
  LOGICAL QACE
#endif
#if KEY_ACE==1
  real(chm_real)  BSARR(*)
#endif
  !
#if KEY_CHEQ==1
  real(chm_real) CGNEW(*),CGOLD(*),VCG(*),KECGTMP,KECGC,VCGSCA
  real(chm_real) VEL,SD
  INTEGER FREECG(*)
#endif
  integer :: i1, j1, k
  ! PJ 06/2005
#if KEY_PIPF==1
  real(chm_real) UINDN(3,*),UINDO(3,*),VUIND(3,*)
#endif
#if KEY_TMD==1
  integer :: icnt, new_inrt = -999
#endif
  !
  ! XKONG add
  !
#if KEY_DIMS==1
  real(chm_real) ZCM,VXCM,VYCM,VZCM,AXCM,AYCM,AZCM
#else /**/
  real(chm_real) XCM,YCM,ZCM,VXCM,VYCM,VZCM,AXCM,AYCM,AZCM
#endif
  real(chm_real) SCALED,HEAT,HEAT2,TEMNEW,JHTEMP
  real(chm_real) SEED
  integer,allocatable,dimension(:) :: lrngseeds
  logical qpresent
  INTEGER NPRIV,NPRIVOLD,NDEGF,IGVOPT,NFREAT
  INTEGER NDEGFI,IDYNPR,ISTART,ISTOP,NCYCLE,IPSTOP,NXTISTART
  INTEGER I,J, IDEGF, ISTPSA, IS, IPT, ITRANS, LDYNA, BTMI
  real(chm_real) TBATH,RBUF,DELTEM,AVETEM
  INTEGER PRLOLD
  INTEGER NAVER
  INTEGER PBMLEV,IJ,ITM
  LOGICAL QAVER,QLANG,QBAKUP,RUNOK,ERR,QSTPRT,DOIT,DONE
  LOGICAL QBROADV,QBROADC
  LOGICAL QNORT(6),QALWRT(6), QKUHEAD
  INTEGER NATMX
  !      INTEGER XNSE,YNSE,ZNSE
  INTEGER ATFRST,ATLAST,JHLAST
#if KEY_MTS==1
  !      INTEGER XMM,YMM,ZMM,XMI,YMI,ZMI
#endif

#if KEY_FOURD==1 /*4ddecl*/
  ! 4D variables:
  INTEGER NDEG4,ICH4
  INTEGER IHT4,IEQ4
  real(chm_real)  FSTT4,FNLT4,TIN4,E4FILL,TWH4,TWL4
  ! 4D temporary arrays
  real(chm_real),allocatable,dimension(:) :: FDNEW
  real(chm_real),allocatable,dimension(:) :: FDOLD
  real(chm_real),allocatable,dimension(:) :: FDAVE
  real(chm_real),allocatable,dimension(:) :: VFD
#endif /* (4ddecl)*/
  !
  ! DEFAULT VALUES FOR THE VARIOUS VARIABLES
  !
  INTEGER ISCALE,IHTFRQ
  INTEGER IASVEL,NTRFRQ,IEQFRQ,ISVFRQ
  real(chm_real)  SCALE
  INTEGER ISCVEL,IASORS,ICHECW,IPRFRQ
#if KEY_FSSHK==1
  integer   ndegrmv
#endif
#if KEY_FSSHK==1
  real(chm_real)   rndegrmv
#endif
#if KEY_TPS==1 /*tps_variables*/
  integer npsavc, npsavv,INBCUT
  integer CREPDSTR
  LOGICAL QTPS, qtpsrs
#endif /*  (tps_variables)*/
  !
  integer(chm_int4) :: count4, rate4, maxcount4
  LOGICAL QRSTN
#if KEY_DHDGB==1
!AP/MF
  LOGICAL QRSTN_FHDGB
#endif
  !
  !
  !     AKMAST is the AKMA units time step for dynamics.  TIMFAC is the
  !     conversion factor from AKMA time units to picoseconds.
  !
  real(chm_real) AKMAST
  !
  ! -- mikem (handle premature termination of dynamics) ->
#if KEY_QUANTA==1
  LOGICAL TERMINATEDYN
#endif
  real(chm_real) PHVAL, PHTEMP
  integer :: NPHFREQ, NMCSTEP, PHUNUM, IDIDPHREX, IUNPHR, IUNPHW
  integer, dimension(nres) :: ophstate
#if KEY_STRINGM==1 || KEY_QCHEM==1 /*  VO stringm : when using many replicas with intensive i/o rst  */
             ! files get corrupted often if calculation runs out of time :
             ! prevent restart file corruption during crashes
#if KEY_UNIX==1 || KEY_GNU==1
  character(len=200) md_step_count
  character(len=200), save :: restart_file_name='default.rst'
  character(len=2*len(restart_file_name)+len('mv -f  .part >/dev/null &  ')) :: syscmd ! exact length of move command
  character(len=12), save :: restart_form='FORMATTED', restart_acc='SEQUENTIAL'
  logical :: openunit
  integer, parameter :: max_recl = 1000000
  integer, save :: restart_recl=max_recl
  integer :: stat ! for system calls
#endif
#endif /* VO */
  !
  ! <- mikem --
  character(len=10) COMLYN2      ! TPCONTROL (G. Lamoureux)
  INTEGER COMLEN2           ! TPCONTROL (G. Lamoureux)
#if KEY_DOMDEC==1
  logical qewald, qnoewa
#endif
! #if KEY_MOBHY==1
  integer iacce,idono,ihyd,i2,iinc,j2,jinc
  integer,save :: iseed1
  logical lhbonded,accept,HBdone(MAXHB),JUSTHOPPED(NRES)
  real(chm_real) :: beta,tempx
  character(len=8) :: xres,xjres
  data iseed1 /272834949/
! #endif
  integer ia
  logical :: want_openmm
  logical :: want_blade
  integer :: ntrfrq_cyc

#if KEY_OPENMM==1
  type(omm_dynopts_t) :: omm_opt
#endif
#if KEY_BLADE==1
  type(blade_dynopts_t) :: blade_opt
#endif

#if KEY_LIBRARY == 1
  real(chm_real) :: custom_ret_val
#endif

  DATA ISCALE,IHTFRQ/0,0/
  DATA IASVEL,NTRFRQ,ISVFRQ,IEQFRQ/1,0,0,0/
  DATA SCALE/1.0D0/
  DATA ISCVEL,IASORS,ICHECW,IPRFRQ/0,0,1,1000/
  DATA QAVER/.FALSE./

#if KEY_LIBRARY == 1
  if (present(options)) then
     ieqfrq = options%ieqfrq
     ntrfrq = options%ntrfrq
     ichecw = options%ichecw
     iasors = options%iasors
     iasvel = options%iasvel
     iscale = options%iscale
     iscvel = options%iscvel
     isvfrq = options%isvfrq
     iprfrq = options%iprfrq
     ihtfrq = options%ihtfrq
  end if
#endif

  QCNSTP = INDXA(COMLYN,COMLEN,'PCON')  >  0
# if KEY_REPDSTR==1
  call setup_pv_correction( qcnstp )
# endif
  ! domdec was here

!!$#if KEY_DOMDEC==1
!!$#if KEY_FSSHK==1
!!$     if (q_domdec .and. qshake .and. qfshake .and. .not.shakeon) then
!!$        ! Re-initialize shake if domdec was not ON when shake was first initialized
!!$        call fsrscshk()  ! De-allocate first
!!$        call fsshkini(iconb_save, renwat_save)
!!$     endif
!!$#endif
!!$#endif

#if KEY_BLOCK==1 /*ldm*/
  ! check the integration method for lambda-dynamics, h-MC/MD
  IF( (QLMC .or. RSTP .or. ILALDM) .and. &
       (ORIG &
#if KEY_MTS==1
       .or. QLNX  &
#endif
#if KEY_FOURD==1
       .or. DIM4  &
#endif
       ) ) THEN
     IF(IUNLDM < 0) CALL WRNDIE(-3,'<DCNTRL>', &
          'USE LEAP FROG (keyword:LEAP) FOR MC/MD,L-DYNAMICS')
  ENDIF
#endif /*  BLOCK*/
  !
  call chmalloc('dcntrl.src','DCNTRL','ISKP',NATOM,intg=ISKP)
  call chmalloc('dcntrl.src','DCNTRL','ISKPR',NATOM,crl=ISKPR)
  !
  QRSTN=.FALSE.
  !
#if KEY_DHDGB==1
!AP/MF
  QRSTN_FHDGB=.FALSE.
#endif
  EOLD = ZERO
  JHSTRT=0
  NPRIV=0
  NPRIVOLD=0
  NAVER=0
  PRLOLD=PRNLEV
  IF(PRNLEV == 5) PRNLEV=4
#if KEY_PARALLEL==1
  plnod0=prnlev
  IF(MYNOD /= 0) plnod0=0
  call gbor(plnod0,1)
#endif
  !

  ! these variables must be initialized
  ! for input scripts that run dyna with
  ! 0 nstep or mpi deadlocks will occur
  jhlast = 0
  qbroadc = .false.
  qbroadv = .false.

#if KEY_DOMDEC==1
  if (q_domdec) then
     atfrst = 1
     atlast = natoml
  else
#endif
#if KEY_PARALLEL==1
#if KEY_SPACDEC==1
     ATFRST=1
     ATLAST=NATOM
#else /**/
     ATFRST=1+IPARPT(MYNOD)
     ATLAST=IPARPT(MYNODP)
#endif
#else /**/
     ATFRST=1
     ATLAST=NATOM
#endif
#if KEY_DOMDEC==1
  endif
#endif
  !
#if KEY_GRAPE==1
  if(lfmm)then
     ATFRST=1
     ATLAST=NATOM
  endif
#endif

  IF(QRMSD)THEN
     QRMSDDYN = (INDXA(COMLYN,COMLEN,'RMSD') > 0)
     URMSD  = GTRMI (COMLYN,COMLEN,'URMS',-1)
     IRMSDCLL = 0
     if(QRMSDDYN)then
        if(prnlev > 2) then
           write(outu,'(a)') ' Time-dependent RMSD restraint is active'
           if(URMSD > 0)then
              write(outu,'(a,i3)') ' output will be written to unit ',URMSD
           endif
        endif
     endif
  ENDIF

  QDYNCALL = .FALSE.

  !
  ! Parse commands for openmm runs
  want_openmm = .false.
#if KEY_OPENMM==1
  want_openmm = omm_requested(COMLYN, COMLEN, 'DCNTRL')
 ! if (want_openmm) then
!     omm_opt = omm_parse_options(comlyn, comlen, 'DCNTRL')
 ! endif
#endif
  want_blade = .false.
#if KEY_BLADE==1
  want_blade = blade_requested(COMLYN, COMLEN, 'DCNTRL')
#endif
  !
  !     read the i/o units for dynamics.
  !
  ! iuncrd 11 iunvel 0 iunwri 10 iunrea 10 kunit 0
#if KEY_LIBRARY == 1
  IUNCRD = GTRMI(COMLYN, COMLEN, 'IUNC', iuncrd)
  IUNREA = GTRMI(COMLYN, COMLEN, 'IUNR', iunrea)
  IUNVEL = GTRMI(COMLYN, COMLEN, 'IUNV', iunvel)
  IUNWRI = GTRMI(COMLYN, COMLEN, 'IUNW', iunwri)
  KUNIT = GTRMI(COMLYN, COMLEN, 'KUNI', kunit)
#else /* KEY_LIBRARY */
  IUNCRD=GTRMI(COMLYN,COMLEN,'IUNC',-1)
  IUNREA=GTRMI(COMLYN,COMLEN,'IUNR',-1)
  IUNVEL=GTRMI(COMLYN,COMLEN,'IUNV',-1)
  IUNWRI=GTRMI(COMLYN,COMLEN,'IUNW',-1)
  KUNIT=GTRMI(COMLYN,COMLEN,'KUNI',-1)
#endif /* KEY_LIBRARY */

  IUNQMC=GTRMI(COMLYN,COMLEN,'IUNQ',-1)
  IUNXYZ=GTRMI(COMLYN,COMLEN,'IUNX',-1)
  QKUHEAD=(KUNIT  >  0)
  IUNOS=GTRMI(COMLYN,COMLEN,'IUNO',-1)

#if KEY_DHDGB==1
!AP/MF
  IUDHDGB=GTRMI(COMLYN,COMLEN,'IUDHDGB',-1)
#endif
#if KEY_BLOCK==1 /*block_clean*/
  IBLCKFEP = GTRMI(COMLYN,COMLEN,'IBLC',IBLCKFEP)
  NBLCKFEP = GTRMI(COMLYN,COMLEN,'NBLC',0)
  ! ldm
  IUNLDM=GTRMI(COMLYN,COMLEN,'IUNL',IUNLDM)
  ITEMPLD=GTRMI(COMLYN,COMLEN,'ILAT',0)
  NTEMPLD=GTRMI(COMLYN,COMLEN,'NLAT',0)
  ILAPOT=GTRMI(COMLYN,COMLEN,'ILAP',ILAPOT)
  ILAF=GTRMI(COMLYN,COMLEN,'ILAF',ILAF)
  IRESD=GTRMI(COMLYN,COMLEN,'IRES',IRESD)
  ! end ldm
#endif /*   (block_clean)*/
  ! whether to do backups of the restart file
  !     if qbakup is true then restart wridyn will backup restart file
  !     as filename- before writing new one
  QBAKUP=(INDX(COMLYN,COMLEN,'BACK',4) /= 0)
  IF (QBAKUP) PBMLEV = GTRMI(COMLYN,COMLEN,'BACK',0)
  IF(PRNLEV >= 2) WRITE(OUTU,5) IUNREA,IUNWRI,IUNOS, &
       IUNCRD,IUNVEL,KUNIT
5 FORMAT('  IUNREA =',I3,7X,'  IUNWRI =',I3,7X,'   IUNOS =',I3,/, &
       '  IUNCRD =',I3,7X,'  IUNVEL =',I3,7X,'   KUNIT =',I3,7X)
  !
#if KEY_DHDGB==1
!AP/MF
  IF (QFHDGB) THEN
  IF(PRNLEV >= 2) WRITE(OUTU,101) IUDHDGB,SAMASS
101 FORMAT (' IUDHDGB =',I3,' SAMASS=',F8.3/)
  ENDIF
#endif
#if KEY_BLOCK==1 /*ldm*/
      IF(qmld.or.QLDM.OR.QLMC) THEN
         IF(PRNLEV >= 2) WRITE(OUTU,501) IUNLDM
501      FORMAT('  IUNLDM =',I3)
         IF(IUNLDM < 0) CALL WRNDIE(-1,'<DCNTRL>', &
              'OUTPUT UNIT FOR LAMBDA WAS NOT SET')
      ENDIF
#endif
#if KEY_FOURD==1
  IF(DIM4) THEN
     ! Allocate space for temporary arrays
     call chmalloc('dcntrl.src','DCNTRL','FDNEW',NATOM,crl=FDNEW)
     call chmalloc('dcntrl.src','DCNTRL','FDOLD',NATOM,crl=FDOLD)
     call chmalloc('dcntrl.src','DCNTRL','FDAVE',NATOM,crl=FDAVE)
     call chmalloc('dcntrl.src','DCNTRL','VFD',NATOM,crl=VFD)
  ENDIF
#endif
  !
#if KEY_FLUCQ==1
  ! For FlucQ, zero charge velocities at start of dynamics
  IF (QFLUC) CALL FQZERO
#endif
  ! JG 5/02
  RCENT = .FALSE.
  IF (INDXA(COMLYN, COMLEN, 'CENT')  >  0) RCENT = .TRUE.
  NCRES = GTRMI (COMLYN,COMLEN,'NCRES',99999)
  IF (RCENT) THEN
     CALL CRDMOV(NATOM,X,Y,Z,XOLD,YOLD,ZOLD,NRES,IBASE)
  ENDIF
  ! JG
  LDYNA=0
  IF (IREST == 1) THEN
     !
     !     To restart dynamics, read save file and output some of the old
     !     parameters before checking the input for new specificatons.
     !
     IF(ORIG) THEN
        LDYNA=-1
     ELSE
        LDYNA=1
     ENDIF
#if KEY_DYNVV2==1
     QVVHOLO=VVERL2.AND.QHOLO
#endif

     CALL READYN(IUNREA,NATOM,X,Y,Z,XOLD,YOLD,ZOLD,VX,VY,VZ, &
#if KEY_CHEQ==1
          CG,CGOLD,VCG,QCG,                        &
#endif
#if KEY_PIPF==1
          UIND,UINDO,VUIND,QPFDYN,                 &
#endif
#if KEY_PIPF==1
          NPFBATHS,PFNHSBATH,PFNHSOBATH,           &
#endif
#if KEY_DYNVV2==1
          QVVHOLO,XNEW,YNEW,ZNEW,                  &
#endif
          NPRIV,JHSTRT,NDEGF,NSTEP,NSAVC,NSAVV,SEED, &
          AVETEM,ISTPSA,LDYNA &
#if KEY_BLOCK==1
          ,QLMC,QLDM,NBLOCK,BIXLAM,BLDOLD,BIVLAM,NSAVL &  /*ldm*/
#endif
#if KEY_FOURD==1
          ,VFD,FDOLD                 &
#endif
#if KEY_SCCDFTB==1
          ,qlamda,qpkac,qsccres,icntdyn,iavti,dvdl,dvdlav,dtmp1 &
#endif
#if KEY_DHDGB==1
          ,QFHDGB,TOTALS,SDEF,SDEFOLD,VS_DHDGB        &
#endif
          )
     NPRIVOLD=NPRIV
     IF(JHSTRT == 0) THEN
        ! do a 2-step start from this restart file
        IGVOPT=2
     ELSE
        ! do a 3-step start from this restart file
        IGVOPT=3
        !              note: the VK array is not saved, so we zap it here
#if KEY_DHDGB==1
!AP/MF
       IF (QFHDGB) THEN
          DO I=1,TOTALS
             VK_DHDGB(I)=ZERO
          ENDDO
       ENDIF
#endif
        DO I=ATFRST,ATLAST
           !RCZ980922 - FMARK replaced with 0 to avoid differences with templates
#if KEY_GRAPE==1
           if(lfmm)then
              if(fmmcpu(i)/=1)cycle
           endif
#endif
#if KEY_SPACDEC==1
           IF(ICPUMAP(I) == MYNOD) THEN
#endif
              VK(I)=ZERO     ! FMARK
#if KEY_SPACDEC==1
           ENDIF
#endif
        ENDDO
     ENDIF
     !
     ! I guess this can be ignored in the new RANDOM code /MH09/
     ISEED=INT(SEED)
     if (.not.qoldrng) then     !yw 05-Aug-2008
        CALL CLCGINIT(ISEED)
        ISEED=1
     endif
     IF(PRNLEV >= 2) WRITE(OUTU,15) NSTEP,JHSTRT
15   FORMAT(' NSTEP  =',I6,'  JHSTRT =',I6)
     JHTEMP=AVETEM*JHSTRT
  ENDIF

  !
#if KEY_TSM==1
  IF(QTSM.AND.PIGSET) THEN
     CALL PIGCVSET(X,Y,Z)
     CALL PIGCVSET(XOLD,YOLD,ZOLD)
  ENDIF
#endif /*  TSM*/
  !
  ! Initialize the HBOND and NONBOND structure and the code-lists
  call timer_stpstrt(T_dcntrl,T_10extra)
  ! if (.not. want_blade) & ! BLaDE doesn't use update -- but this command which wastes a bunch of time also reads nonbonded options
  CALL UPDATE(COMLYN,COMLEN,X,Y,Z,WMAIN,.TRUE.,.TRUE.,.TRUE., &
       .TRUE.,.TRUE.,LDYNA,XOLD,YOLD,ZOLD,VX,VY,VZ)
  call timer_stpstrt(T_10extra,T_dcntrl)
  IF(ORIG) THEN
     LDYNA=-1
  ELSE
     LDYNA=1
  ENDIF
  IF (IREST == 1 .AND. PRNLEV >= 3) CALL PRNHBD(OUTU)
  !
  NSTEP=GTRMI(COMLYN,COMLEN,'NSTE',NSTEP)
  NPRINT=GTRMI(COMLYN,COMLEN,'NPRI',NPRINT)
  !
#if KEY_MNDO97==1
  ! namkh 09/12/03 for MNDO97 printing setup
  ISTOPQM= NSTEP
#endif
#if KEY_SQUANTM==1
  if(LTRBOMD) then           ! for TR-BOMD
     q_apply_tr_bomd=.true.  ! .false. !
     i_md_step=0             ! initialize
     i_md_step_old=-1        ! 1-step behind

     !        call something to read restart information.
  end if
#endif
  !
  !     Set up the free atom array needed for dynamics and the value of
  !     nfreat needed for abner.
  !
  call chmalloc('dcntrl.src','DCNTRL','FREEAT',NATOM,intg=FREEAT)
  CALL MKFRAT(IMOVE,NATOM,FREEAT,NFREAT)
  !
  !
  !     Get the input data specific to dynamics, and then set up and scale
  !     the velocities etc.
  !
  DELTA=TIMEST/TIMFAC
  AKMAST=GTRMF(COMLYN,COMLEN,'AKMA',ZERO)
  IASORS=GTRMI(COMLYN,COMLEN,'IASO',IASORS)
  ICHECW=GTRMI(COMLYN,COMLEN,'ICHE',ICHECW)
  IEQFRQ=GTRMI(COMLYN,COMLEN,'IEQF',IEQFRQ)
  ILBFRQ=GTRMI(COMLYN,COMLEN,'ILBF',ILBFRQ)
  IMGFRQ=GTRMI(COMLYN,COMLEN,'IMGF',IMGFRQ)
#if KEY_ENSEMBLE==1
  IF (JREX) THEN
     FINALT=GTRMF(COMLYN,COMLEN,'FINA',FINALT)
     FINALT=ENSMYT
  ELSE
     FINALT=GTRMF(COMLYN,COMLEN,'FINA',FINALT)
  ENDIF
#else /**/
  FINALT=GTRMF(COMLYN,COMLEN,'FINA',FINALT)
#endif
#if KEY_REPDSTR==1
  temnew=finalt
#endif
  FIRSTT=GTRMF(COMLYN,COMLEN,'FIRS',FIRSTT)
  IASVEL=GTRMI(COMLYN,COMLEN,'IASV',IASVEL)
  IHTFRQ=GTRMI(COMLYN,COMLEN,'IHTF',IHTFRQ)
  ISCALE=GTRMI(COMLYN,COMLEN,'ISCA',ISCALE)
  ISCVEL=GTRMI(COMLYN,COMLEN,'ISCV',ISCVEL)

#if KEY_LIBRARY == 0
  call dyn_parse_iseed(comlyn, comlen, irest, ilang, iseed)
#else
  ! TODO: see if user has set rngseeds with api fun call
  !       then we could clear the setting if necessary
  call dyn_init_rng(irest, ilang)
#endif /* KEY_LIBRARY */

  IPRFRQ=GTRMI(COMLYN,COMLEN,'IPRF',IPRFRQ)
  NSAVC=GTRMI(COMLYN,COMLEN,'NSAVC',NSAVC)
#if KEY_BLOCK==1
  NSAVL=GTRMI(COMLYN,COMLEN,'NSAVL',NSAVL)    /*lsm*/
#endif
  NSAVV=GTRMI(COMLYN,COMLEN,'NSAVV',NSAVV)
  NSAVQ=GTRMI(COMLYN,COMLEN,'NSAVQ',NSAVQ)
  NSAVX=GTRMI(COMLYN,COMLEN,'NSAVX',NSAVX)
  MXYZ=GTRMI(COMLYN,COMLEN,'MXYZ',-1)
  NSNOS=GTRMI(COMLYN,COMLEN,'NSNOS',NSNOS)
  NPRIV=GTRMI(COMLYN,COMLEN,'NPRE',NPRIV)
  SCALE=GTRMF(COMLYN,COMLEN,'SCAL',SCALE)
  SCALED=SCALE
  TEMINC=GTRMF(COMLYN,COMLEN,'TEMI',TEMINC)
  TIMEST=GTRMF(COMLYN,COMLEN,'TIME',ZERO)
  NTRFRQ=GTRMI(COMLYN,COMLEN,'NTRF',NTRFRQ)
  ISVFRQ=GTRMI(COMLYN,COMLEN,'ISVF',0)
  TSTRUC=GTRMF(COMLYN,COMLEN,'TSTR',TSTRUC)
  TWINDH=GTRMF(COMLYN,COMLEN,'TWINDH',TWINDH)
  IF(TWINDH < 0.0) TWINDH=-TWINDH
  TWINDL=GTRMF(COMLYN,COMLEN,'TWINDL',TWINDL)
  IF(TWINDL > 0.0) TWINDL=-TWINDL
  ECHECK=GTRMF(COMLYN,COMLEN,'ECHE',ECHECK)
  NDEGFI=GTRMI(COMLYN,COMLEN,'NDEG',-1)
  ! stops rotation/translation on a segment basis
  QSEGSRT = (INDXA(COMLYN,COMLEN,'SEGS') > 0)
  IF(QSHAKE) SHKTOL=GTRMF(COMLYN,COMLEN,'TOL',SHKTOL)
#if KEY_FLUCQ==1
  IF (QFLUC) THEN
     ! Set FlucQ timestep to be equal to standard dynamics timestep
     CALL FQSETT(TIMEST)
     ! Parse FlucQ dynamics options for thermostatting etc.
     CALL FQDPAR(COMLYN,COMLEN)
  ENDIF
#endif

! phmd depends on either gbsv or gbsw
#if KEY_PHMD==1
  ! Setup velocity and Nose for PHMD
  IF (QPHMD) CALL PHMDZERO
#endif

#if KEY_SGLD==1 /*sgld_parse*/
  !WXW  SGLD flag
  QSGLD = QSGLD.OR.(INDXA(COMLYN,COMLEN,'SGLD')  >  0)
  QSGMD = QSGMD.OR.(INDXA(COMLYN,COMLEN,'SGMD')  >  0)
#if KEY_REPDSTR==1 /*rexsgld*/
  if(qrxsgld)then
     IF(ILANG == 0)QSGMD=.TRUE.
     IF(ILANG /= 0)QSGLD=.TRUE.
  endif
#endif /* (rexsgld)*/
  IF(VVERL2)THEN
    IF(QSGLD.AND..NOT.QNHLANGEVIN) THEN
      QSGMD=.TRUE.
      QSGLD=.FALSE.
    ENDIF
    IF(QSGMD.AND.QNHLANGEVIN) THEN
      QSGLD=.TRUE.
      QSGMD=.FALSE.
    ENDIF
  ELSE
    IF(QSGLD.AND.ILANG == 0) THEN
      QSGMD=.TRUE.
      QSGLD=.FALSE.
    ENDIF
    IF(QSGMD.AND.ILANG /= 0) THEN
      QSGLD=.TRUE.
      QSGMD=.FALSE.
    ENDIF
  ENDIF
  IF(QSGLD.OR.QSGMD) THEN
     IF(QSGMD)SGFT=TWO*PTONE
     IF(QSGLD)SGFT=ONE
     ISGSTA=GTRMI(COMLYN,COMLEN,'ISGSTA',1)
     ISGEND=GTRMI(COMLYN,COMLEN,'ISGEND',NATOM)
     SGTYPE=GTRMI(COMLYN,COMLEN,'SGTYPE',1)
     SGSIZE=GTRMF(COMLYN,COMLEN,'SGSIZE',THREE)
     TEMPSG=GTRMF(COMLYN,COMLEN,'TEMPSG',ZERO)
     TSGAVG=GTRMF(COMLYN,COMLEN,'TSGAVG',TWO*PTONE)
     TSGAVP=GTRMF(COMLYN,COMLEN,'TSGAVP',TEN*TSGAVG)
     SGFT=GTRMF(COMLYN,COMLEN,'SGFT',SGFT)
     SGFF=GTRMF(COMLYN,COMLEN,'SGFF',ZERO)
     SGFD=GTRMF(COMLYN,COMLEN,'SGFD',ZERO)
#if KEY_REPDSTR==1 /*rexsgld*/
     if (qrxsgld) then
        ! Set temperature from REPD input.
        if(sgft <= zero) sgft = my_sg_ft()
        call set_my_sg_ft( sgft )
        TSGSET = my_replica_temperature()
        if (qrepmaster.and.prnlev.gt.2) write(outu,'(a,f10.2,f10.4)') 'RXSG> Setting sgft, sgff from replica: ', sgft,sgff
     endif
#endif /* (rexsgld)*/
     IF(ISGSTA<0)ISGSTA=0
     IF(ISGEND<ISGSTA)ISGEND=NATOM
     QSGCOM = INDXA(COMLYN,COMLEN,'SGCOM')  >  0
     QSGBZ=INDXA(COMLYN,COMLEN,'SGBZ')  >  0
     !WXW Processing SGLD variables
     IF(PRNLEV >= 2) WRITE(OUTU,950)
     SGAVG1=TIMEST/TSGAVG
     SGAVG0=ONE-SGAVG1
     SGAVP1=TIMEST/TSGAVP
     SGAVP0=ONE-SGAVP1
     SGEXP=ZERO
     IF(PRNLEV >= 2) WRITE(OUTU,960)ISGSTA,ISGEND, &
          INT(1.0D0/SGAVG1+HALF),TSGAVG,TSGAVP
950  FORMAT(/' Parameters for SGMD/SGLD simulation:')
960  FORMAT('  Guiding range from',I5,'  to ',I5 / &
          '  Averaging over ',I10,' steps'/ &
          '  Local Average time is: ',F8.4,' ps'/ &
          '  Convergency control average time is: ',F8.4,' ps')
     IF(QSGBZ.AND.PRNLEV >= 2)THEN
        WRITE(OUTU,975)
    ENDIF
    IF(PRNLEV >= 2)WRITE(OUTU,980)SGFT
    IF(SGFF*SGFF>RSMALL.AND.PRNLEV >= 2)WRITE(OUTU,981)SGFF
    IF(SGFD*SGFD>RSMALL.AND.PRNLEV >= 2)WRITE(OUTU,982)SGFD
    IF(PRNLEV >= 2)WRITE(OUTU,984)SGTYPE,SGSIZE
    IF(PRNLEV >= 2 .AND. SGTYPE>3 )THEN
      IF(QSGCOM)THEN
        WRITE(OUTU,985)
      ELSE
        WRITE(OUTU,986)
      ENDIF
    ENDIF
    IF(PRNLEV >= 2)  &
    WRITE(OUTU,983)QSGLD,QSGMD,QSGBZ
    !IF(PRNLEV >= 2)WRITE(OUTU,989)
975  FORMAT('  SGBZ is set to adjust SGFT/SGFF')
979  FORMAT('  SGFT is set to ',F8.4,' and let SGFFI changing')
980  FORMAT('  Momentum guiding factor SGFT is set to:',F8.4)
981  FORMAT('  Force guiding factor SGFF is set to:',F8.4 )
982  FORMAT('  SGLD-GLE momentum guiding factor SGFD is set to:',F8.4)
983  FORMAT('  QSGLD,QSGMD,QSGBZ:'/10L4)
984  FORMAT('  SGTYPE,SGSIZE:',I4,F6.2)
985  FORMAT('  Net Guiding forces in spatial maps are kept ')
986  FORMAT('  Net Guiding forces in spatial maps are removed ')
989  FORMAT(' ............Begin SGMD/SGLD simulation......'/)
  ENDIF
#endif /* (sgld_parse)*/
  !
#if KEY_CHEQ==1
  IF (QCG) THEN
     MASSQ=GTRMF(COMLYN,COMLEN,'QMAS',MASSQD)
     ! input QMAS is in (ps/e)*(ps/e) kcal/mol, need to convert it to AKM time unit
     MASSQ=MASSQ/(TIMFAC*TIMFAC)
     QFRQ=GTRMI(COMLYN,COMLEN,'QFRQ',1000000)
     if(prnlev > 2) write(OUTU,*) ' QFRQ = ', QFRQ
     NOSEFLAG=GTRMI(COMLYN,COMLEN,'FQINT',5)
     IF (NOSEFLAG == 1) THEN
        IF (CHEQBASETUP) THEN
           if(prnlev > 2) then
              write(OUTU,*) ' Charge Hoover Baths Are Set Up Correctly'
           endif
        ELSE
           CALL WRNDIE(-1,'<DCNTRL>', &
                'CHECK "FQBA" SYNTAX for THERMOSTATS CHANGE')
        ENDIF
     ENDIF
     if(prnlev > 1)then
        write(OUTU,*) 'FQ integrator              =', NOSEFLAG
        WRITE(OUTU,*)'QMASS Converted to AKM unit: ', &
             'QMASS = ',MASSQ
     endif
     TCG  =GTRMF(COMLYN,COMLEN,'TCG',TCGDEF)
     IF ((CGEQ > 0).AND.(TCG <= ZERO)) THEN
        if(prnlev > 1)WRITE(OUTU,*) &
             'TCG NEED TO BE GREATER THAN ZERO, Reset to: ' &
             ,TCGDEF
        TCG=TCGDEF
     ENDIF
     ! convert it to kcal/mol
     KECGC =TCG*NATOM*0.0059607 ! 3*0.0019869 ! 2 times the kinetic energy
     ! if NOCO set FIRSTT AND FINALT to 0
     IF (QNOCO) THEN
        FIRSTT=ZERO
        FINALT=ZERO
     ENDIF
  ENDIF
#endif
  IF(INDXA(COMLYN, COMLEN, 'TSAL')  >  0) THEN
#if KEY_TSALLIS==1 /*tsallis_parse*/
     !       ARD 01-06-12
     !       Parse Tsallis options
     !       Modified November 2007, H Kamberaj
     QTSALL = .TRUE.
     QTTSALL= .FALSE.
     QTALPHA= .FALSE.
     TSEMIN = GTRMF(COMLYN,COMLEN,'EMIN', ZERO)
     TSQ    = ONE - GTRMF(COMLYN,COMLEN,'QTSA', ONE)
     IF (TSQ  ==  ZERO) THEN
        CALL WRNDIE(-2,'<DCNTRL>','QTSAllis = 1 --> Boltzmann')
        QTSALL = .FALSE.
     ENDIF
#else /* (tsallis_parse)*/
     CALL WRNDIE(-2,'<DCNTRL>','TSALLIS DYNAMICS CODE NOT COMPILED')
#endif /* (tsallis_parse)*/
  ELSEIF (INDXA(COMLYN,COMLEN,'TTSA')  >  0) THEN
#if KEY_TSALLIS==1 /*ttsallis_parse*/
     IF (.NOT. QTTSALL) THEN
        CALL WRNDIE(-2,'<DCNTRL>', &
             'DIHEDRAL ANGLE TO BE SCALED SHOULD BE SELECTED')
        QTTSALL =.TRUE.
     ENDIF
     QTSALL  = .FALSE.
     QTALPHA = .FALSE.
     TSQ = ONE - GTRMF(COMLYN, COMLEN, 'QTSA', ONE)
     TSEMIN = GTRMF(COMLYN,COMLEN,'EMIN', ZERO)
     IF (TSQ  ==  ZERO) THEN
        CALL WRNDIE(-2,'<DCNTRL>', 'QTSALLIS=1-->BOLTZMANN')
        QTTSALL=.FALSE.
     ENDIF
  ELSEIF (INDXA(COMLYN,COMLEN,'POTS')  >  0) THEN
     QTALPHA = .TRUE.
     QTTSALL = .FALSE.
     QTSALL  = .FALSE.
     TSALPHA = GTRMF(COMLYN,COMLEN,'TSAL',ZERO)
#else /* (ttsallis_parse)*/
     CALL WRNDIE(-2,'<DCNTRL>','TSALLIS DYNAMICS CODE NOT COMPILED')
#endif /* (ttsallis_parse)*/
  ELSE
#if KEY_TSALLIS==1 /*tttsallis_parse*/
     QTALPHA = .FALSE.
     QTTSALL = .FALSE.
     QTSALL  = .FALSE.
#endif /* (tttsallis_parse)*/
  ENDIF
#if KEY_TPS==1 /*tps_parse*/
  !     ARD and M. Hagan for TPS
  !     Parse special transition path sampling keywords
  QTPS = (INDXA(COMLYN, COMLEN, 'PATH')  >  0)
  IF (QTPS) THEN
     call tps_parse(comlyn,comlen)
  ENDIF
  INBCUT = GTRMI(COMLYN,COMLEN,'INBC',0)
#endif /*  (tps_parse)*/

  !
#if KEY_FOURD==1 /*4dparse*/
  !     The fourth-D step at which to begin back projection
  !      and the force constant in 4-d along with 4-D set-up.
  DO I = 1,LENENT
     DIM4ON(I) = 0   ! turn off all 4d energy terms.
  ENDDO
  IF (DIM4) THEN
     FCOUNT = 0
     FNLT4=GTRMF(COMLYN,COMLEN,'FNLT4',FINALT)
     FSTT4=GTRMF(COMLYN,COMLEN,'FSTT4',FIRSTT)
     TIN4=GTRMF(COMLYN,COMLEN,'TIN4',TEMINC)
     IHT4=GTRMI(COMLYN,COMLEN,'IHT4',IHTFRQ)
     IEQ4=GTRMI(COMLYN,COMLEN,'IEQ4',IEQFRQ)
     ICH4=GTRMI(COMLYN,COMLEN,'ICH4',ICHECW)
     TWH4=GTRMF(COMLYN,COMLEN,'TWH4',TWINDH)
     IF(TWH4 < 0.0) TWH4=-TWH4
     TWL4=GTRMF(COMLYN,COMLEN,'TWL4',TWINDL)
     IF(TWL4 > 0.0) TWL4=-TWL4
     K4DI   = GTRMF(COMLYN,COMLEN,'K4DI',K4DI)
     INC4D  = NSTEP+1
     INC4D  = GTRMI(COMLYN,COMLEN,'INC4',INC4D)
     DEC4D  = NSTEP+1
     DEC4D  = GTRMI(COMLYN,COMLEN,'DEC4',DEC4D)
     MULTK4 = GTRMF(COMLYN,COMLEN,'MULT',MULTK4)
     E4FILL = GTRMF(COMLYN,COMLEN,'E4FI',ZERO)
     DO I = 1,LENENT
        DIM4ON(I) = 1     ! turn on all 4d energy terms.
     ENDDO
     !                 turn off selected terms
     IF (INDXA(COMLYN, COMLEN, 'SKBO')  >  0) DIM4ON(BOND)=0
     IF (INDXA(COMLYN, COMLEN, 'SKAN')  >  0) DIM4ON(ANGLE)=0
     IF (INDXA(COMLYN, COMLEN, 'SKDI')  >  0) DIM4ON(DIHE)=0
     IF (INDXA(COMLYN, COMLEN, 'SKVD')  >  0) DIM4ON(VDW)=0
     IF (INDXA(COMLYN, COMLEN, 'SKEL')  >  0) DIM4ON(ELEC)=0
     IF (INDXA(COMLYN, COMLEN, 'SKCO')  >  0) DIM4ON(CDIHE)=0
     IF (INDXA(COMLYN, COMLEN, 'FIL4')  >  0) CALL FILL4(X,Y,Z, &
          NATOM,E4FILL)
     !
  ENDIF
#endif /* (4dparse)*/
  !
  IF(TIMEST /= 0.0 .AND. AKMAST == 0.0) THEN
     DELTA=TIMEST/TIMFAC
  ELSE IF(TIMEST == 0.0 .AND. AKMAST /= 0.0) THEN
     DELTA=AKMAST
     TIMEST=TIMFAC*DELTA
  ELSE IF(TIMEST /= 0.0 .AND. AKMAST /= 0.0) THEN
     CALL WRNDIE(-2,'<DCNTRL>','Conflicting time step specified')
  ENDIF
  TIMEST=DELTA*TIMFAC
  IF(DELTA <= ZERO) THEN
     CALL WRNDIE(-3,'<DCNTRL>','Zero time step specified')
  ENDIF
  !
  ! Parse the stop trans/rotation options
  QNORT(1) = INDXA(COMLYN,COMLEN,'NOXT')  >  0
  QNORT(2) = INDXA(COMLYN,COMLEN,'NOYT')  >  0
  QNORT(3) = INDXA(COMLYN,COMLEN,'NOZT')  >  0
  QNORT(4) = INDXA(COMLYN,COMLEN,'NOXR')  >  0
  QNORT(5) = INDXA(COMLYN,COMLEN,'NOYR')  >  0
  QNORT(6) = INDXA(COMLYN,COMLEN,'NOZR')  >  0

#if KEY_TMD==1
  if (qtmd) then
    ! inrt may also be set by tmdini
    ! default inrt is 1000 in case no one sets it
    ! -999 is the "inrt not found" indicator
    new_inrt = gtrmi(comlyn, comlen, 'INRT', -999)
    if (new_inrt == -999 .and. inrt == -999) then
      inrt = 1000
    else if (new_inrt .ne. -999 .and. inrt == -999) then
      inrt = new_inrt
    else if (new_inrt .ne. -999 .and. inrt .ne. -999 .and. inrt .ne. new_inrt) then
      call wrndie(-2, '<DCNTRL>', &
        'conflicting inrt set in both tmdinit and dyna commands')
      inrt = new_inrt
    end if
  end if ! qtmd
#endif /* KEY_TMD */

#if KEY_TSM==1
  ! Code Thermodynamic Simulation Method
  IF (QTSM) THEN
     IF(NPUMB /= 0) THEN
        CALL PUMBPER(NPUMB,PUMBDH,VPUMB,VPUMBS,ERR)
        IF(ERR) CALL DIEWRN(0)
     ENDIF
     IF(PIGSET) CALL PIGMSET(AMASS,.TRUE.)
  ENDIF
#endif /*  TSM*/
  !
  !-----------------------------------------------------------------------
  ! Parse constant pH options
  !
  ! Note: right now, this only works with the verlet algorithm
  IF(INDXA(COMLYN,COMLEN,'PHCO') > 0) THEN
    NPHFREQ=GTRMI(COMLYN,COMLEN,'PHEXCF',-1)
    NMCSTEP=GTRMI(COMLYN,COMLEN,'PHMCTR',-1)
    PHUNUM=GTRMI(COMLYN,COMLEN,'PHUNUM',-1)
    PHVAL=GTRMF(COMLYN,COMLEN,'PHVA',-9999.0)
    PHTEMP=GTRMF(COMLYN,COMLEN,'PHTEMP',-ONE)
    IUNPHW=GTRMI(COMLYN,COMLEN,'IUNPHW',-1)
    IUNPHR=GTRMI(COMLYN,COMLEN,'IUNPHR',-1)
    PHRSVRUNUM=GTRMI(COMLYN,COMLEN,'IPHRSV',-1)

    !! Fix for Ana, allow negative pH values.
    IF(NPHFREQ < 0 .or. NMCSTEP <= 0) THEN
       CALL WRNDIE(-2,'<DCNTRL>', &
                   'PHVAL, PHEXCF, and PHMCTR NOT SPECIFIED CORRECTLY!')
    ENDIF
    IF(IUNPHW.GT.0) THEN
       IF(PRNLEV.GT.3) WRITE(OUTU,*) 'DCNTRL> Restart file for constant pH will be written to unit ', IUNPHW
    ELSE
       IF(PRNLEV.GT.3) WRITE(OUTU,*) 'DCNTRL> Constant pH restart file will not be written'
    ENDIF
    IF(IUNPHR > 0) CALL PHRDRSTRT(IUNPHR)
  ELSE
    PHVAL=-9999.0
    NPHFREQ=0
    NMCSTEP=0
  ENDIF

  !
  !-----------------------------------------------------------------------
  ! Parse constant temperature options.
  !
  ! The following options are for Langevin dynamics
  ! Note: Changed the default of TBATH from 0.0 to FINALT - brb 11/22/95
#if KEY_ENSEMBLE==1
  IF (JREX) THEN
     TBATH=ENSMYT
  ELSE
#endif /* KEY_ENSEMBLE */

#if KEY_LIBRARY == 1
     if (present(options)) then
        tbath = options%tbath
     else
#endif /* KEY_LIBRARY */

     TBATH=GTRMF(COMLYN,COMLEN,'TBAT',FINALT)

#if KEY_LIBRARY == 1
  end if
#endif /* KEY_LIBRARY */

#if KEY_ENSEMBLE==1
  ENDIF
#endif

  KBT=TBATH*KBOLTZ
  RBUF=GTRMF(COMLYN,COMLEN,'RBUF',ZERO)
#if KEY_ACE==1
  QLBACE=(INDXA(COMLYN,COMLEN,'QLBA')  >  0)
#endif
  !
  !
  ! Parse the commands specific to the Berendsen constant temperature method
  QCNSTT = INDXA(COMLYN,COMLEN,'TCON')  >  0
  IF(QCNSTT) THEN
     TCOUPL = GTRMF(COMLYN,COMLEN,'TCOU',ZERO)
     IF(TCOUPL <= ZERO) THEN
        TCOUPL=RBIG
        QCNSTTX=.TRUE.
     ELSE
        QCNSTTX=.FALSE.
     ENDIF
     ! Note: Changed the default of TREF from ROOMT to FINALT - brb 11/22/95
#if KEY_ENSEMBLE==1
     IF (JREX) THEN
        TREF=ENSMYT
     ELSE
        TREF   = GTRMF(COMLYN,COMLEN,'TREF',FINALT)
     ENDIF
#else /**/
     TREF   = GTRMF(COMLYN,COMLEN,'TREF',FINALT)
#endif
  ELSE
     QCNSTTX=.FALSE.
  ENDIF
  !
  ! Use a thermal piston for constant temperature (Hoover Method)
  QNPT = (INDXA(COMLYN,COMLEN,'HOOV')  >  0)
  IF(QNPT) THEN
     TMASS = GTRMF(COMLYN,COMLEN,'TMAS',HUNDRD)
     ! Note: Changed the default of REFT from ROOMT to FINALT - brb 11/5/96
#if KEY_ENSEMBLE==1
     IF (JREX) THEN
        REFT=ENSMYT
     ELSE
        REFT   = GTRMF(COMLYN,COMLEN,'REFT',FINALT)
     ENDIF
#else /**/
     REFT = GTRMF(COMLYN,COMLEN,'REFT',FINALT)
#endif
     ! Zero the velocity and coordinate of the thermal piston
     IF(IREST  ==  0) THEN
        PNH = ZERO
        PNHV = ZERO
        PNHF = ZERO
     ENDIF
     ! *Note: the thermal piston is integrated using a velocity verlet algorithm
  ENDIF

#if KEY_BLADE==1
  if (want_blade) ilang=1
  if (want_blade .and. NTRFRQ>0) then
     call wrndie(0,'<DCNTRL>', &
          'NTRFRQ should be 0 with BLaDE, otherwise the system may run hot.')
     ! The kinetic energy distribution produced by the BLaDE g-BAOAB integrator
     ! is cooler than would be expected from a Boltzmann distribution, depending
     ! on the timestep. NTRFRQ does not know this, and will equilibrate the
     ! velocities to a standard Boltzmann distribution, and the excess kinetic
     ! energy supplied by NTRFRQ will make the system run hot. For normal
     ! systems, (mostly water, SHAKE and SETTLE on, 2.0 fs timestep, T=300 K),
     ! the effect is only about 1 degree Kelvin.
  endif
#endif
  !-----------------------------------------------------------------------
  ! Parse constant pressure options.
  !
  ! Parse the commands specific to the constant pressure
!  QCNSTP = INDXA(COMLYN,COMLEN,'PCON')  >  0  ! APH: This is moved in the beginning of the
!                                                     subroutine so that recip nodes have it too
  IF(QCNSTP) CALL PRSET(COMLYN,COMLEN,TBATH)
  !
  !-----------------------------------------------------------------------
  ! The following options are for Langevin dynamics (TS, 970725)
  SXREF=GTRMF(COMLYN,COMLEN,'XBUF',SSXREF)
  SYREF=GTRMF(COMLYN,COMLEN,'YBUF',SSYREF)
  SZREF=GTRMF(COMLYN,COMLEN,'ZBUF',SSZREF)
  !
  !-----------------------------------------------------------------------
  ! Nose-Hoover Dynamics
  !
  IF(QNOSE.AND.QNOSP) CALL WRNDIE(-3,'<DCNTRL>', &
       'Calling two different nose methods at the same time.')
  !
  IF(QNOSE.OR.QNOSP) THEN
     QRSTN = INDXA(COMLYN,COMLEN,'RSTN')  >  0
  ENDIF
  !
  IF(IREST == 0 .OR. QRSTN) THEN
     DO ITM=1,MAXNOS
        SN11(ITM) = 0.0
        SN12(ITM) = 0.0
        SN13(ITM) = 0.0
        SNH(ITM) = 0.0
        SNHV(ITM) = 0.0
        SNHF(ITM) = 0.0
     ENDDO
     ! BEGIN TPCONTROL (G. Lamoureux)
     IF(QNHCH) THEN
        DO ITM=1,MAXNOS
           DO I=1,NHCH_MAXM
              NHCH_ETA(ITM,I) = ZERO
              NHCH_ZETA(ITM,I) = ZERO
              NHCH_G(ITM,I) = ZERO
           ENDDO
        ENDDO
     ENDIF
     IF(QPCON) THEN
        DO I=1,3
           DO J=1,3
              ETA_CP(J,I) = ZERO
              ZETA_CP(J,I) = ZERO
              G_CP(J,I) = ZERO
           ENDDO
        ENDDO
        COMLYN2 = 'INIT'
        COMLEN2 = 4
        CALL GETPRS(COMLYN2,COMLEN2)
     ENDIF
     ! END TPCONTROL (G. Lamoureux)
  ENDIF
  !
  IF(QNOSE) THEN
     NOBL=1
     ! Note: Changed the default of RTMPR from ROOMT to FINALT - brb 11/22/95
#if KEY_ENSEMBLE==1
     IF (JREX) THEN
        RTMPR(1) = ENSMYT
     ELSE
        RTMPR(1) = GTRMF(COMLYN,COMLEN,'TREF',FINALT)
     ENDIF
#else /**/
     RTMPR(1) = GTRMF(COMLYN,COMLEN,'TREF',FINALT)
#endif
     SQM(1)   = GTRMF(COMLYN,COMLEN,'QREF',ZERO)
     NSCYC   = GTRMI(COMLYN,COMLEN,'NCYC',0)
     !
     call chmalloc('dcntrl.src','DCNTRL','INLCKP',NATOM,intg=INLCKP)
     CALL NOSINIT(NATOM,1,INLCKP)
     !
     IF(SQM(1) == ZERO) SQM(1)=RBIG
     IF(NSCYC == 0) NSCYC=10
  ENDIF
  !
  IF(QNOSP.AND.(.NOT.QNOSE)) QNOSE=.TRUE.
  !CC      IF(QNOSE) ORIG=.TRUE. ! Can't work - BRB
  !
  EPROP(VOLUME) =  GTRMF(COMLYN,COMLEN,'VOLU',ZERO)
  !
  QRSTN=.FALSE.
  !
#if KEY_MTS==1
  IF (VVERL .AND. (.NOT. QTBMTS)) THEN
     DO ITM=1,NATOM
        IMTS(ITM)=-1
     ENDDO
  ENDIF
#endif
  !
  IDYNPR=0
  !
  QAVER=(INDXA(COMLYN,COMLEN,'AVER') > 0)
  IF(QAVER) THEN
     call chmalloc('dcntrl.src','DCNTRL','IXAVE',NATOM,crl=IXAVE)
     call chmalloc('dcntrl.src','DCNTRL','IYAVE',NATOM,crl=IYAVE)
     call chmalloc('dcntrl.src','DCNTRL','IZAVE',NATOM,crl=IZAVE)
#if KEY_CHEQ==1
     IF(QCG) call chmalloc('dcntrl.src','DCNTRL','ICGAVE',NATOM,crl=ICGAVE)
#endif
  ENDIF
  !
#if KEY_PERT==1
  ! PERTDF wipes out the rest of the command line if the PUNIT
  !  option is used.
  IF(QPERT) CALL PERTDF
#endif
  !
  !
  ! set-up-dynamics
  !
  ! Check the heating counter to be sure we are not restarting past
  ! the heating phase.  also be sure we are not incrementing in the
  ! wrong direction.
  !
  IF (IHTFRQ > 0) THEN
     IF (SIGN(ONE,TEMINC) /= SIGN(ONE,FINALT-FIRSTT)) TEMINC=-TEMINC
     IF(TEMINC /= 0.0) THEN
        I=(FINALT-FIRSTT)/TEMINC+1
     ELSE
        I=0
     ENDIF
     I=I*IHTFRQ
     IF(I <= NPRIV) IHTFRQ=0
  ENDIF
  !
  ! Set ncycle so that all the different cycles will work correctly.
  ! check iprfrq and nprint so that divide by zeroes won't occur.
  !
  I=0
  ntrfrq_cyc = NTRFRQ
  if (want_openmm) ntrfrq_cyc = 0  ! use OpenMM CMMotionRemover
  call fincyc(ncycle,ihtfrq,ieqfrq,ntrfrq_cyc,isvfrq,inbfrq,ihbfrq, &
       ilbfrq,imgfrq, &
#if KEY_REPDSTR==1
       irexfq, &
#endif
#if KEY_REPDSTR==0
       i, &
#endif
       nphfreq, &
       nstep &
#if KEY_TMD==1
       ,inrt  &
#endif
       )
  if (.not. want_openmm) NTRFRQ = ntrfrq_cyc

  if(iprfrq > nstep) then
     iprfrq=nstep
  else if(iprfrq > 0) then
     iprfrq=(iprfrq/ncycle)*ncycle
     if(iprfrq <= 0) iprfrq=ncycle
  else if(ieqfrq > ihtfrq.and.ihtfrq > 0) then
     iprfrq=ihtfrq
  else if(ieqfrq > 0) then
     iprfrq=ieqfrq
  else
     iprfrq=nstep
  endif

#if KEY_FOURD==1 /*4dset*/
  !     Set for 4-D:
  IF(DIM4) THEN
     IF ((IHT4 > 0).AND.(MOD(IHT4,NCYCLE) /= 0)) THEN
        IF (IHT4 < NCYCLE) IHT4=NCYCLE
        IF (IHT4 > NCYCLE) IHT4=NINT(DBLE(IHT4/NCYCLE))*NCYCLE
     ENDIF
     IF ((IEQ4 > 0).AND.(MOD(IEQ4,NCYCLE) /= 0)) THEN
        IF (IEQ4 < NCYCLE) IEQ4=NCYCLE
        IF (IEQ4 > NCYCLE) IEQ4=NINT(DBLE(IEQ4/NCYCLE))*NCYCLE
     ENDIF
     IF (MOD(INC4D,NCYCLE) /= 0) &
          INC4D=NINT(DBLE(INC4D/NCYCLE))*NCYCLE
     IF (MOD(DEC4D,NCYCLE) /= 0) &
          DEC4D=NINT(DBLE(DEC4D/NCYCLE))*NCYCLE
  ENDIF
#endif /* (4dset)*/
  !
  IF(NPRINT <= 0 .OR. NPRINT > IPRFRQ) NPRINT=IPRFRQ
  !
  ! Print out the state of the simulation
  !
  IF(ILANG == 1 .AND. PRNLEV >= 2) THEN
     WRITE(OUTU,'(A)') ' DCNTRL> Langevin integration requested.'
#if KEY_ACE==1
     IF (QLBACE .AND. .NOT.QACE) THEN
        CALL WRNDIE(1,'<DCNTRL>', &
             'QLBAce option requires ACE! QLBA turned off.')
        QLBACE=.FALSE.
     ENDIF
     IF (QLBACE .AND. PRNLEV >= 2) THEN
        WRITE(OUTU,'(A)') &
             ' DCNTRL> Using Born-solvation radius dependent Fbeta.'
     ENDIF
#endif
  ENDIF
  !
  IF(QCNSTT .AND. PRNLEV >= 2 .and. .not. want_openmm) THEN
     IF(QCNSTTX) THEN
        WRITE(OUTU,'(A)') ' DCNTRL> Constant temperature requested.'
        WRITE(OUTU,'(2(8X,A,F12.7,A,/))') &
             ' Reference temperature         = ',TREF,' K.', &
             ' Constraint method.'
     ELSE
        WRITE(OUTU,'(A)') ' DCNTRL> Constant temperature requested.'
        WRITE(OUTU,'(2(8X,A,F12.7,A,/))') &
             ' Reference temperature         = ',TREF,' K.', &
             ' Temperature coupling constant = ',TCOUPL,' ps.'
     ENDIF
  ENDIF
  !
#if KEY_TMD==1
  !
  IF(QTMD) THEN
     !
     ! Initial start situation, read in the target structure.
     !
     ! MSF tmd \./
     ! Added this part to Properly Orient the target structures at the
     ! very beginning.  This fitting is done whether or not a restart is used,
     ! due to the fact that one can now restart from a non-TMD restart file.
     call chmalloc('dcntrl.src','DCNTRL','iscr1',natom+natom,intg=iscr1)
     call chmalloc('dcntrl.src','DCNTRL','iscr2',natom,crl=iscr2)
     call stoprt_tmd(x,y,z,natom,amass, &
          ixtar,iytar,iztar, &
          natom,amass,iscr1, &
          iscr2,istmd)
     ! Orient the second structure, if there is one.
     IF(QZETA) THEN
        call stoprt_tmd(x,y,z,natom,amass, &
             ixtar2,iytar2,iztar2, &
             natom,amass,iscr1, &
             iscr2,istmd)
     ENDIF
     call chmdealloc('dcntrl.src','DCNTRL','iscr1',natom+natom,intg=iscr1)
     call chmdealloc('dcntrl.src','DCNTRL','iscr2',natom,crl=iscr2)

     if(irest  ==  0) then

        ! Compute the original rms distance for the TMD.
        !
        IF(.NOT.QZETA) THEN
           call inidis(ixtar,iytar,iztar,natom,amass,istmd)
        ELSE
           call inizet(ixtar,iytar,iztar,iXTAR2,iYTAR2,iZTAR2, &
                amass,istmd,jSTmd)
        ENDIF
        !
        ! Restart situation, read in the original distance rho0 from
        ! restart target file and the last saved target coordinate.
        !
     elseif(irest  ==  1) then
        ! Compute the original rms distance for the TMD.
        !
        IF(.NOT.QZETA) THEN
           call inidis(ixtar,iytar,iztar,natom,amass,istmd)
        ELSE
           call inizet(ixtar,iytar,iztar,iXTAR2,iYTAR2,iZTAR2, &
                amass,istmd,JStmd)
        ENDIF
     endif
  ENDIF
  !
  IF(QTMD .AND. PRNLEV >= 2) THEN
     call tmdprint(outu)
  ENDIF
#endif
  !
#if KEY_REPDSTR==1
  if (qrexchg) then
    ! Set temperature from REPD input.
    ! This will correctly set replica temperature even in N dimensions
    FINALT   = my_replica_temperature()
    if (qrepmaster.and.prnlev.gt.2) write(outu,'(a,f10.4)') 'REPD> Setting temperature from REPD input: ', FINALT
    FIRSTT   = FINALT
    TBATH    = FINALT
    RTMPR(1) = FINALT
    REFT     = FINALT
  endif
  if (qphrex) then
    PHVAL = my_replica_ph()
    if (qrepmaster.and.prnlev.gt.2) write(outu,'(a,f10.4)') 'REPD> Setting pH from REPD input: ', PHVAL
  endif

  if (ndims.lt.2) then
#   if KEY_BLOCK==1
    if (qrexchgl) then
      if (qfastrepdstr) then
        call wrndie(-5,'<DCNTRL>','Fast biasing potential replica exchange not supported without repdstr_2')
      else
        call print_gbias(nblock, irepdstr) ! ldm
      endif
    endif
#   endif /* KEY_BLOCK */
  endif
#endif /* KEY_REPDSTR */
  !
#if KEY_OPENMM==1
  ! Moved here to be compatible with replica exchange via repdstr
  if (want_openmm) then
#if KEY_REPDSTR==1
     omm_opt = omm_parse_options(comlyn, comlen, temnew, &
          (qrexchg.and.qfastrepdstr.and..not.qtor_repex), 'DCNTRL' )
#else
     omm_opt = omm_parse_options(comlyn, comlen, temnew, &
          .false., 'DCNTRL' )
#endif

     call omm_report_options(omm_opt, 'DCNTRL')
  endif
#endif
#if KEY_BLADE==1
  if (want_blade) then
     blade_opt = blade_parse_options(comlyn, comlen, temnew, .false., 'DCNTRL')
     call blade_report_options(blade_opt, 'DCNTRL')
  endif
#endif
  !
  IF (PRNLEV >= 2) THEN
     !
     IF (QNOSE.AND.(.NOT.QNOSP)) THEN
        WRITE(OUTU,'(1X,A)') 'DCNTRL> Nose-Hoover dynamcs requested.'
        WRITE(OUTU,'(2(8X,A,F12.7,A,/),(8X,A,I5,/))') &
             ' Reference temperature         = ',RTMPR(1),' K.', &
             ' Temperature coupling constant = ',SQM(1),'Kcal sec**2', &
             ' Cycles of convergence         = ',NSCYC
     ENDIF
     !
     WRITE(OUTU,40) NSTEP,NSAVC,NSAVV,    ISCALE,ISCVEL,IASORS, &
          IASVEL,ICHECW,NTRFRQ, IHTFRQ,IEQFRQ,NPRINT,INBFRQ, &
          IHBFRQ,IPRFRQ,ILBFRQ,IMGFRQ,ISVFRQ,NCYCLE, &
          NSNOS
40   FORMAT('   NSTEP =',I9,1X,'   NSAVC =',I9,1X,'   NSAVV =',I9/ &
          '  ISCALE =',I9,1X,'  ISCVEL =',I9,1X,'  IASORS =',I9/ &
          '  IASVEL =',I9,1X,'  ICHECW =',I9,1X,'  NTRFRQ =',I9/ &
          '  IHTFRQ =',I9,1X,'  IEQFRQ =',I9,1X,'  NPRINT =',I9/ &
          '  INBFRQ =',I9,1X,'  IHBFRQ =',I9,1X,'  IPRFRQ =',I9/ &
          '  ILBFRQ =',I9,1X,'  IMGFRQ =',I9,1X,/ &
          '  ISVFRQ =',I9,1X,'  NCYCLE =',I9,1X,'   NSNOS =',I9)
#if KEY_BLOCK==1 /*ldm*/
     IF(QLDM) WRITE(OUTU,403)NSAVL, NBIASV
403     FORMAT('  NSAVL  =',I6,4x,'  NBIASV =',I6)
#endif
     WRITE(OUTU,45) FIRSTT,TEMINC,TSTRUC,FINALT,TWINDH,TWINDL, &
          DELTA,DELTA*TIMFAC
45   FORMAT('  FIRSTT =',F10.3,'  TEMINC =',F10.3,'  TSTRUC =',F10.3/ &
          '  FINALT =',F10.3,'  TWINDH =',F10.3,'  TWINDL =',F10.3/ &
          /'  TIME STEP = ',1PG12.5,' AKMA      ',1PG12.5,' PS'/)
     WRITE(OUTU,'(''  RANDOM NUM. GEN. SEED(S) ='',8(1X,I0))') &
          (RNGSEEDS(I),I=1,NRAND)
#if KEY_FOURD==1 /*4dprint*/
     IF(DIM4) THEN
        WRITE(OUTU,46) FSTT4,TIN4,INC4D
        WRITE(OUTU,47) FNLT4,IHT4,DEC4D
        WRITE(OUTU,49) K4DI,IEQ4,MULTK4
        WRITE(OUTU,51) E4FILL
     ENDIF
46   FORMAT('  FSTT4  =',F10.3,'  TIN4   =',F10.3,'  INC4D  =',I6,4X)
47   FORMAT('  FNLT4  =',F10.3,'  IHT4   =',I6,4X,'  DEC4D  =',I6,4X)
49   FORMAT('  K4DI   =',F10.3,'  IEQ4   =',I6,4X,'  MULTK4 =',F10.3)
51   FORMAT('  E4FILL =',F10.3)
#endif /* (4dprint)*/
     IF(QSHAKE) WRITE(OUTU,50) SHKTOL
50   FORMAT(/,10X,' SHAKE TOLERANCE = ',E15.5)
  ENDIF
  !
  IF(INBFRQ >= 50) THEN
     CALL WRNDIE(1,'<DCNTRL>', &
          'Nonbond bond update frequency may be too large.')
  ENDIF
  IF(IHBFRQ >= 50 .AND. CUTHB > 1.0) THEN
     CALL WRNDIE(1,'<DCNTRL>', &
          'Hydrogen bond update frequency may be too large.')
  ENDIF
  !
  IF(ISCVEL /= 0 .AND. IDGFEX > 0) THEN
     CALL WRNDIE(1,'<DCNTRL>', &
          'SHAKE overspecified. Dont scale by atoms. Disabled.')
     ISCVEL=0
  ENDIF
  !
  IF(NTRFRQ <= 0 .AND. QPME .AND. ILANG /= 1) THEN
     CALL WRNDIE(-1,'<DCNTRL>', &
          'A non-zero NTRFRQ should be used with PME Ewald')
     ISCVEL=0
  ENDIF
  !
  ! Compute which global degrees of freedom are removed.
  ! Don't remove net translation if some atoms are fixed, or if
  ! Langevin dynamics is used.
  DO I=1,6
     QALWRT(I)=.TRUE.
  ENDDO
  !
  IF(NATOM > NFREAT) THEN
     CONTINUE
  ELSE IF(QCNSTR) THEN
     CONTINUE
  ELSE IF(ILANG == 1) THEN
     CONTINUE
     ! Check the image transformation.
  ELSE IF(NTRANS > 0) THEN
     QALWRT(1:6)=.FALSE.
     DO ITRANS=1,NTRANS
        IPT=(ITRANS-1)*12
        IF(ABS(IMTRNS(IPT+10)) > TENM5) THEN
           !                 x translation
           QALWRT(5)=.TRUE.
           QALWRT(6)=.TRUE.
        ENDIF
        IF(ABS(IMTRNS(IPT+11)) > TENM5) THEN
           !                 y translation
           QALWRT(4)=.TRUE.
           QALWRT(6)=.TRUE.
        ENDIF
        IF(ABS(IMTRNS(IPT+12)) > TENM5) THEN
           !                 z translation
           QALWRT(4)=.TRUE.
           QALWRT(5)=.TRUE.
        ENDIF
        IF(.NOT.NOROT) THEN
           !                 check the rotation matrix
           XCM=ABS(IMTRNS(IPT+1)-ONE)
           YCM=ABS(IMTRNS(IPT+5)-ONE)
           ZCM=ABS(IMTRNS(IPT+9)-ONE)
           IF(XCM < TENM5.AND.YCM < TENM5.AND.ZCM < TENM5) THEN
              !                    no rotation
              CONTINUE
           ELSE IF(XCM < TENM5) THEN
              !                    x rotation
              QALWRT(2)=.TRUE.
              QALWRT(3)=.TRUE.
              QALWRT(5)=.TRUE.
              QALWRT(6)=.TRUE.
           ELSE IF(YCM < TENM5) THEN
              !                    y rotation
              QALWRT(1)=.TRUE.
              QALWRT(3)=.TRUE.
              QALWRT(4)=.TRUE.
              QALWRT(6)=.TRUE.
           ELSE IF(ZCM < TENM5) THEN
              !                    z rotation
              QALWRT(1)=.TRUE.
              QALWRT(2)=.TRUE.
              QALWRT(4)=.TRUE.
              QALWRT(5)=.TRUE.
           ELSE
              !                 no simple rotation
              DO I=1,6
                 QALWRT(I)=.TRUE.
              ENDDO
           ENDIF
        ENDIF
     ENDDO
  ELSE
     ! Free vacuum dynamics (remove all 6)
     DO I=1,6
        QALWRT(I)=.FALSE.
     ENDDO
  ENDIF
  ! Override default global motion removal if specified
  DO I=1,6
     IF(QNORT(I)) QALWRT(I)=.FALSE.
  ENDDO
  !
  !
  IF(PRNLEV > 5) WRITE(OUTU,386) QALWRT
386 FORMAT(' NTRFrq flags: Is net (XT,YT,ZT,XR,YR,ZR)', &
       ' an allowed motion?:',6L2)
  !
  QSTPRT=.FALSE.
  !
  ! compute-number-of-degrees-of-freedom
  IF(NDEGFI > 0) THEN
     NDEGF=NDEGFI
  ELSE
     !
     ! not input. must compute NDEGF
     NDEGF=0
     !
     IF(QNOSE) THEN
        DO I=1,NOBL
           NDGN(I)=0
        ENDDO
     ENDIF
     !
#if KEY_FOURD==1
     ! 4D NDEG of Freedom is NDEG4
     NDEG4=0
     IF(DIM4) THEN
        DO I=1,NATOM
           IF (IMOVE4(I) == 0) NDEG4=NDEG4+1
        ENDDO
     ENDIF
#endif
     !
     ! tsm mod djt/shf 4/4/90
     ! reason for mod:  If during perturbation we do not
     ! want to count the degrees of freedom due to the reactant or product
     ! atoms for calculation of the kinetic energy.
     ! next 1 line contains original code commented out.
     !                  IF (IMOVE(I) == 0) NDEGF=NDEGF+3
     ! next 7 lines contain modified code
#if KEY_TSM==1
     IF(QTSM) THEN
        DO I=1,NATOM
           IF (IMOVE(I) == 0.AND.BACKLS(I) == 0) NDEGF=NDEGF+3
        ENDDO
     ELSE
#endif /*  TSM*/
        DO I=1,NATOM
           IF(IMOVE(I) == 0) THEN
              IF(QNOSE) THEN
                 J=INLCKP(I)
                 NDGN(J)=NDGN(J)+3
              ENDIF
              NDEGF=NDEGF+3
           ENDIF
        ENDDO
#if KEY_TSM==1
     ENDIF
#endif /*  TSM*/
     !
! increase for block lambda values
#if KEY_BLOCK==1
     if (qmld) call msld_ndegf(nblock,ndegf)  /*ldm*/
#endif
     ! decrease by active SHAKE constraints
#if KEY_FSSHK==1
     ndegrmv=0
#endif
     IF(QSHAKE) THEN
        DO I=1,NCONST
#if KEY_TSM==1
           !tsm mod shf/djt 4/4/90
           IF(QTSM) THEN
              IF(IMOVE(SHKAPR(1,I)) == 0.OR.IMOVE(SHKAPR(2,I)) == 0) &
                   THEN
                 IF(BACKLS(SHKAPR(1,I)) == 0.OR. &
                      BACKLS(SHKAPR(2,I)) == 0) then
#if KEY_FSSHK==1
                    NDEGrmv=NDEGrmv+1
#endif
#if KEY_FSSHK==0
                    NDEGF=NDEGF-1
#endif
                 endif
              ENDIF
           ELSE
#endif /*  TSM*/
              IF(IMOVE(SHKAPR(1,I)) == 0 .OR. &
                   IMOVE(SHKAPR(2,I)) == 0) THEN
                 !yw+2                     NDEGF=NDEGF-1
#if KEY_FSSHK==0
                 NDEGF=NDEGF-1
#endif
#if KEY_FSSHK==1
                 ndegrmv=ndegrmv+1
#endif
                 IF(QNOSE) THEN
                    IJ=SHKAPR(1,I)
                    J=INLCKP(IJ)
                    NDGN(J)=NDGN(J)-1
                 ENDIF
              ENDIF
#if KEY_TSM==1
           ENDIF
#endif /*  TSM*/
        ENDDO
#if KEY_FSSHK==1
        !...##IF PARALLEL
        !            if(qfshake)then
        !               rndegrmv=ndegrmv
        !               call gcomb(rndegrmv,1)
        !               ndegrmv=rndegrmv+.4
        !            endif
        !...##ENDIF
        ndegf=ndegf-ndegrmv
#endif
        !
        IF(IDGFEX > 0) THEN
           IF(WRNLEV >= 2) WRITE(OUTU,48) IDGFEX
48         FORMAT( &
                ' DCNTRL: SHAKE constraints are probably', &
                ' overspecified.',/, &
                '         The number of degrees of', &
                ' freedom are reduced by:',I6,/, &
                '         NDEGF should be specified', &
                ' if it is not correct.')
           CALL WRNDIE(0,'<DCNTRL>', &
                'Overspecified SHAKE constraints.')
           NDEGF=NDEGF-IDGFEX
        ENDIF
        !
#if KEY_FSSHK==1
     else
#if KEY_PARALLEL==1
        rndegrmv=ndegrmv
        call gcomb(rndegrmv,1)
        ndegrmv=rndegrmv+.4
#endif
        ndegf=ndegf-ndegrmv
#endif
     ENDIF
     !
#if KEY_TSM==1
     !!DJT mod 02/88********************************************************
     ! decrease by number of internal coordinate constraints
     NDEGF=NDEGF-NICF
#endif /*  TSM*/
     !
#if KEY_NOST2==0 /*ndegf_st2*/
     ! decrease by active ST2 constraints where atoms are not fixed.
     DO I=1,NGRP
        IF (IGPTYP(I) == 3) THEN
           IS=IGPBS(I)+1
           IF (IMOVE(IS) == 0.OR.IMOVE(IS+1) == 0) NDEGF=NDEGF-1
           IF (IMOVE(IS) == 0.OR.IMOVE(IS+2) == 0) NDEGF=NDEGF-1
           IF (IMOVE(IS+1) == 0.OR.IMOVE(IS+2) == 0) NDEGF=NDEGF-1
        ENDIF
     ENDDO
#endif /* (ndegf_st2)*/
     !
     !CC##IF LONEPAIR
     !CCC currently not needed because lonepairs have IMOVE(I)=-1
     !CC         NDEGF=NDEGF-NUMLP*3
     !CC##ENDIF
     !
#if KEY_SHAPES==1
     ! Remove 3*N-6 degrees of freedom for each rigid shape
     !  (Assume that no shape is linear)
     DO I=1,NUMSHP
        IF(SHPTYP(I) == 'RIGI') THEN
           IF(SHPNAT(I) > 2) NDEGF=NDEGF+6-3*SHPNAT(I)
        ENDIF
     ENDDO
#endif
     !
#if KEY_RXNCONS==1
     ! remove reaction coordinate constraint  (jc070701)
     IF(LRXCNS) THEN
        NDEGF=NDEGF-NRXNCS
     ENDIF
#endif
     !
     ! Now modify NDEGF based on global considerations
     NDEGF=NDEGF-6

#if KEY_DYNVV2==1
     ! Why???; put back in for VV2! (E. Harder 2005)
     IF(VVERL2) NDEGF=NDEGF+6
     ! CM degrees of freedom removed by damping function (E. Harder 2005)
     IF (IABST  >  0) NDEGF=NDEGF-3

     IF (VVERL2) THEN
        call vv2_alloc(NATOM)
     ENDIF
#endif

#if KEY_TMD==1
     ! For tmd remove one extra degree of freedom for the holonomic constraint
     if(qtmd)then
        NDEGF=NDEGF-1
     endif
#endif
     DO I=1,6
        IF(QALWRT(I)) NDEGF=NDEGF+1
     ENDDO
#if KEY_DHDGB==1
!AP/MF
     IF(QFHDGB) THEN
        NDEGF=NDEGF+TOTALS
     ENDIF
#endif
     !
     IF(NDEGF <= 0) NDEGF=1
  ENDIF
  !
  ! - Nose
  IF(QNOSE) THEN
     NDGN(1)=NDEGF
     DO I=2,NOBL
        NDGN(1)=NDGN(1)-NDGN(I)
     ENDDO
#if KEY_DHDGB==1
!AP/MF
   IF (QFHDGB) THEN
      DO I=1,6
         IF (.NOT. QALWRT(I)) THEN
             NDGN(1)=NDGN(1)+1
             NDGN(2)=NDGN(2)-1
         ENDIF
      ENDDO
      IF (NDGN(1) .NE. 10) THEN
          CALL WRNDIE(-5,'<DCNTRL>', &
        'DHDGB::Selection for the first block should be emprty')
      ENDIF
   ENDIF
#endif
     ! BEGIN TPCONTROL (G. Lamoureux)
     IF (.NOT.QTPCON) THEN
        ! END TPCONTROL (G. Lamoureux)
        DO I=1,NOBL
           IF(NDGN(I) <= 0) NDGN(I)=1
        ENDDO
        ! BEGIN TPCONTROL (G. Lamoureux)
     ELSE
        !           The barostat is an additional degree of freedom (G. Lamoureux)
        IF (QPCON .AND. TI_CP > 0) THEN
           IF (QPCONFULL) THEN
              NDGN(TI_CP) = NDGN(TI_CP) + XDIM
           ELSE
              NDGN(TI_CP) = NDGN(TI_CP) + 1
           ENDIF
        ENDIF
        !           Correction for an absolute thermostat (G. Lamoureux)
        IF (IABST  >  0) THEN
           IF (NDGN(IABST)  /=  0) THEN
              CALL WRNDIE(-5,'<DCNTRL>', &
                   'Selection for the absolute thermostat should be empty')
           ENDIF
           NDGN(IABST) = NDGN(IABST) + 3
           NDEGF = NDEGF + 3
        ENDIF
     ENDIF
     ! END TPCONTROL (G. Lamoureux)
  ENDIF
  !
  IF(PRNLEV >= 2) THEN
     IF(QNOSE) THEN
        DO I=1,NOBL
           IJ=ANINT(NDGN(I))
           WRITE(OUTU,555) I,IJ
        ENDDO
     ENDIF
     WRITE(OUTU,55) NDEGF
55   FORMAT(' NUMBER OF DEGREES OF FREEDOM = ',I6)
555  FORMAT(' BLOCK ',I3, ' # OF DEGREES OF FREEDOM = ',I6)
  ENDIF
  !
  IF(QNOSE.AND.(IUNOS >= 0)) THEN
     IF(PRNLEV >= 2) WRITE(IUNOS,91) NSTEP,NDEGF,NSNOS
91   FORMAT(3X,3I16)
     CALL GFLUSH(IUNOS)
  ENDIF
#if KEY_DHDGB==1
!AP/MF
  IF (QFHDGB .AND. (IUDHDGB .GE. 0)) THEN
      WRITE (IUDHDGB,102) NSTEP,NDEGF,TOTALS
102  FORMAT(3X,3I16)
    CALL GFLUSH(IUDHDGB)
  ENDIF
#endif
  !
  !
  IF(ORIG) THEN
     NATMX=NATOM
     call chmalloc('dcntrl.src','DCNTRL','XNSE',NATMX,crl=XNSE)
     call chmalloc('dcntrl.src','DCNTRL','YNSE',NATMX,crl=YNSE)
     call chmalloc('dcntrl.src','DCNTRL','ZNSE',NATMX,crl=ZNSE)
#if KEY_DHDGB==1
!AP/MF
    IF (QFHDGB) THEN
     call chmalloc('dcntrl.src','DCNTRL','SDEFLD',TOTALS,crl=SDEFLD)
    ENDIF
#endif
#if KEY_MTS==1
     NATMX=1
     IF (QTBMTS) NATMX = NATOM
     call chmalloc('dcntrl.src','DCNTRL','XMI',NATMX,crl=XMI)
     call chmalloc('dcntrl.src','DCNTRL','YMI',NATMX,crl=YMI)
     call chmalloc('dcntrl.src','DCNTRL','ZMI',NATMX,crl=ZMI)
     call chmalloc('dcntrl.src','DCNTRL','XMM',NATMX,crl=XMM)
     call chmalloc('dcntrl.src','DCNTRL','YMM',NATMX,crl=YMM)
     call chmalloc('dcntrl.src','DCNTRL','ZMM',NATMX,crl=ZMM)
#endif
  ENDIF

#if KEY_DOMDEC==1
# if KEY_REPDSTR==1
  nxtistart = 0
  q_repd_domdec_purerecip = .false.
  if (q_domdec.and.qrepdstr) then
    ! DOMDEC with REPDSTR. To ensure all processes are present for the
    ! exchange calculation we need to direct them towards the exchange
    ! subroutine - see labels 750 and 760.
    call start_split_direct_recip()
    if ((.not.q_direct_node).and.q_recip_node) then
      ! Pure reciprocal node in repdstr still need to perform the exchange
      ! calculation, so wait down by the exchange calculation loop.
      q_repd_domdec_purerecip = .true.
      ! Set istart to 1 so recip nodes actually enter the dynamics loop
      istart = 1
      goto 750
    endif
  endif
# endif /* KEY_REPDSTR */
  if (q_domdec) then
     call start_split_direct_recip()
     if (.not.q_direct_node .and. q_recip_node) then
        ! Pure reciprocal nodes go wait in a loop in energy_recip
        call timer_stop(T_dcntrl)
        call energy_recip(x, y, z, dx, dy, dz, .false.)
        call timer_start(T_dcntrl)
        call stop_split_direct_recip()
        return
     endif
  endif
#endif /* KEY_DOMDEC */

  !
  !=======================================================================
  !
  ! Scale velocities - scale factor has been input
  IF(IREST == 1) THEN
     IF(ISCALE /= 0) THEN
        CALL SCAVEL(SCALED,X,Y,Z,VX,VY,VZ,zero, &
             AMASS,NDEGF,0,IGVOPT,NATOM,IMOVE, &
             zero,0,.false.,0)
        QSTPRT=.TRUE.
     ENDIF
     !
     ! Dynamics start.
     ! Assign velocities and coordinates to integration arrays.
     !
  ELSE
     !
     IGVOPT=2
     !
#if KEY_OLDDYN==1
     IF (ORIG) THEN
        XOLD(1:natom)=X(1:natom)
        YOLD(1:natom)=Y(1:natom)
        ZOLD(1:natom)=Z(1:natom)
#if KEY_CHEQ==1
        IF (QCG) THEN
           CGOLD(1:natom)=CG(1:natom)
           CGNEW(1:natom)=CG(1:natom)
           VCG(1:natom)=ZERO
        ENDIF
#endif
#if KEY_PIPF==1
        IF (QPIPF .AND. QPFDYN) THEN
           UINDO( 1:3, 1:natom)=ZERO
           UINDN( 1:3, 1:natom)=ZERO
           VUIND( 1:3, 1:natom)=ZERO
           UIND( 1:3, 1:natom)=ZERO
           DUIND( 1:3, 1:natom)=ZERO
        ENDIF
#endif
        DO I=1,NATOM
           IF(IMOVE(I) /= 0) THEN
              XNEW(I)=X(I)
              YNEW(I)=Y(I)
              ZNEW(I)=Z(I)
              VX(I)=ZERO
              VY(I)=ZERO
              VZ(I)=ZERO
           ENDIF
        ENDDO
     ENDIF
#endif
#if KEY_DHDGB==1
!AP/MF
        IF(QFHDGB) THEN
           DO I=1,TOTALS
              SDEFOLD(I)=SDEF(I)
          ENDDO
        ENDIF
#endif
     !
     IF(IASVEL /= 0) THEN
        IF(TSTRUC /= FMARK) THEN
           TEMNEW=MAX(TWO*FIRSTT-TSTRUC,ZERO)
        ELSE
           TEMNEW=1.25*FIRSTT
        ENDIF
        !
#if KEY_CHEQ==1
        ! Zero CG temperature
        IF (QCG) THEN
           DO I=1,NATOM
              VCG(I)=ZERO
           ENDDO
        ENDIF
#endif
        ! PJ 06/2005
#if KEY_PIPF==1
        ! Zero dipole temperature
        IF (QPIPF .AND. QPFDYN) THEN

           DO I = 1, NATOM
              DO J = 1, 3
                 VUIND(J,I) = ZERO
              ENDDO
           ENDDO
           !
           ! ONE CAN USE ZERO ORDER INDUCED DIPOLE AS INITIAL VALUES (NUFRS>1),
           ! OTHERWISE USE ZEROES (NUFRS=1)
           !
           IF (NUFRS  >  1) THEN
              QDYFST = .TRUE.
           ENDIF

        ENDIF
#endif

        !eh050711
        !     Don't assign velocities to Drude's! (E. Harder 2005)
        !sb this should just work equally for dual Langevin / drudes
        !   so keeping for dynamc/dual langevin case
        IF (NDRUDE > 0) THEN
           !        Put back the Drude masses on their heavy atoms.  (The motion of
           !        the Drude particles will be solved iteratively, by calling
           !        subroutine "OptimizeDrude" before every call to "ENERGY".)
           !        The IMOVE variable of the Drude particles is set to -1 so that
           !        the integration code will ignore them.
           DO I=1,NATOM
              IF (ISDRUDE(I) .AND. AMASS(I) > ZERO) THEN
                 AMASS(I-1) = AMASS(I-1) + AMASS(I)
              ENDIF
              IF (ISDRUDE(I) .AND. IMOVE(I) == -1) IMOVE(I) = 263 ! eh050802
              IF (ISDRUDE(I) .AND. IMOVE(I) == 0) IMOVE(I) = -1
           ENDDO
        ENDIF
        !eh050711----

        CALL ASSVEL(TEMNEW,X,Y,Z,VX,VY,VZ, &
             AMASS,ISEED,IASVEL,IGVOPT,NATOM,IMOVE &
#if KEY_TSM==1
             ,BACKLS &
#endif
#if KEY_DHDGB==1
             ,QFHDGB,VS_DHDGB,SAMASS,TOTALS  &
#endif
             )

        !eh050802 Reset MASS/IMOVE array
        IF (NDRUDE > 0) THEN
           DO I=1,NATOM
              IF (ISDRUDE(I) .AND. AMASS(I) > 0) THEN
                 AMASS(I-1) = AMASS(I-1) - AMASS(I)
              ENDIF
              IF (ISDRUDE(I) .AND. IMOVE(I) == -1) IMOVE(I) = ZERO
              IF (ISDRUDE(I) .AND. IMOVE(I) == 263) IMOVE(I) = -1
           ENDDO
        ENDIF
        !eh050711 ----------------

#if KEY_DYNVV2==1 /*dynvv2_setup*/
        ! BEGIN DYNA VV2 (G. Lamoureux)
        ! sb: keep this specifically for VV2
        IF (VVERL2 .and. QNOSE .AND. NDRUDE > 0) THEN
#if KEY_DOMDEC==1
           if (q_domdec) CALL WRNDIE(-5,'<DCNTRL>','NOT YET IMPLEMENTED FOR DOMDEC')
#endif
           DO I = ATFRST,ATLAST
#if KEY_GRAPE==1
              if(lfmm)then
                 if(fmmcpu(i)/=1)cycle
              endif
#endif
#if KEY_SPACDEC==1
              IF(ICPUMAP(I) == MYNOD)THEN
#endif
                 ITM = INLCKP(I)
                 if (ISDRUDE(I) .and. RTMPR(ITM) < 100.0) then
                    IF(PRNLEV >= 7) WRITE(OUTU,*) 'Drude particle',I, &
                         ' is assigned the velocity of atom',I-1
                    VX(I) = VX(I-1)
                    VY(I) = VY(I-1)
                    VZ(I) = VZ(I-1)
                 endif
#if KEY_SPACDEC==1
              ENDIF
#endif
           ENDDO
#if KEY_PARALLEL==1 && KEY_SPACDEC==0
           CALL VDGBR(VX,VY,VZ,0)
#endif
#if KEY_SPACDEC==1
           CALL SPACSR(VX,VY,VZ)
#endif
        ENDIF
        ! END DYNA VV2 (G. Lamoureux)
#endif /* (dynvv2_setup)*/
        ! sb correct drude velos; will break old integ and old VVER;
        ! VV2 taken care of above
        if ((ndrude > 0) &
#if KEY_DYNVV2==1
           .and. (.not. vverl2)) then
#else
           ) then
#endif
           ! sb: use this point to warn if langevin not set up
           ! presently only dual langevin supported for drudes in dynamc
           if (ilang == 0) then
              write(outu,*)
              write(outu,*) '******** WARNING ***************'
              write(outu,*) '* You are running a Drude'
              write(outu,*) '* system without Langevin setup '
              write(outu,*) '********************************'
              write(outu,*)
           endif
           DO IA = ATFRST,ATLAST
#if KEY_GRAPE==1
              if(lfmm)then
                 if(fmmcpu(ia)/=1)cycle
              endif
#endif
#if KEY_DOMDEC==1
              if (q_domdec) then
                 i = atoml(ia)
              else
#endif
                 i = ia
#if KEY_DOMDEC==1
              endif
#endif
#if KEY_SPACDEC==1
              IF(ICPUMAP(I) == MYNOD)THEN
#endif
                 if (ISDRUDE(I)) then
                    IF(PRNLEV >= 7) WRITE(OUTU,*) 'Drude particle',I, &
                         ' is assigned the velocity of atom',I-1
                    VX(I) = VX(I-1)
                    VY(I) = VY(I-1)
                    VZ(I) = VZ(I-1)
                 endif
#if KEY_SPACDEC==1
              ENDIF
#endif
           ENDDO
#if KEY_DOMDEC==1
           if (q_domdec) then
              call copy_to_all(vx, vy, vz)
           else
#endif
#if KEY_PARALLEL==1 && KEY_SPACDEC==0
              CALL VDGBR(VX,VY,VZ,0)
#endif
#if KEY_SPACDEC==1
              CALL SPACSR(VX,VY,VZ)
#endif
#if KEY_DOMDEC==1
           endif
#endif
        endif
        QSTPRT=.TRUE.
     ELSE
        IF(PRNLEV >= 2) WRITE(OUTU,60)
60      FORMAT(' NOTE: The comparison coordinate values will', &
             ' be used for the initial velocities (AKMA units).')
        IF(NTRFRQ /= 0) QSTPRT=.TRUE.
        !
     ENDIF
  ENDIF
#if KEY_DYNVV2==1 /*dynvv2_setup2*/
  ! BEGIN DYNA VV2 (G. Lamoureux)
  IF (QNOSE) THEN
     DO J = 1,NOBL
        if (QRELT(J)) then
           RELT_VX(J) = ZERO
           RELT_VY(J) = ZERO
           RELT_VZ(J) = ZERO
           RELT_M(J) = ZERO
        endif
     ENDDO
     ABST_VX = ZERO
     ABST_VY = ZERO
     ABST_VZ = ZERO
     ABST_M = ZERO
     do ia=atfrst,atlast
#if KEY_GRAPE==1
        if(lfmm)then
           if(fmmcpu(ia)/=1)cycle
        endif
#endif
#if KEY_DOMDEC==1
        if (q_domdec) then
           i = atoml(ia)
        else
#endif
           i = ia
#if KEY_DOMDEC==1
        endif
#endif
#if KEY_SPACDEC==1
        IF(ICPUMAP(I) == MYNOD)THEN
#endif
           J = INLCKP(I)
           if (QRELT(J)) then
              RELT_VX(J) = RELT_VX(J) + AMASS(I)*VX(I)
              RELT_VY(J) = RELT_VY(J) + AMASS(I)*VY(I)
              RELT_VZ(J) = RELT_VZ(J) + AMASS(I)*VZ(I)
              RELT_M(J) = RELT_M(J) + AMASS(I)
           endif
#if KEY_SPACDEC==1
        ENDIF
#endif
     ENDDO
#if KEY_PARALLEL==1
     CALL GCOMB(RELT_M,NOBL)
#endif
#if KEY_PARALLEL==1
     CALL GCOMB(RELT_VX,NOBL)
#endif
#if KEY_PARALLEL==1
     CALL GCOMB(RELT_VY,NOBL)
#endif
#if KEY_PARALLEL==1
     CALL GCOMB(RELT_VZ,NOBL)
#endif
     DO J = 1,NOBL
        if (QRELT(J)) then
           RELT_VX(J) = RELT_VX(J) / RELT_M(J)
           RELT_VY(J) = RELT_VY(J) / RELT_M(J)
           RELT_VZ(J) = RELT_VZ(J) / RELT_M(J)
           IF (IABST > 0) THEN
              ABST_VX = ABST_VX + RELT_M(J)*RELT_VX(J)
              ABST_VY = ABST_VY + RELT_M(J)*RELT_VY(J)
              ABST_VZ = ABST_VZ + RELT_M(J)*RELT_VZ(J)
              ABST_M = ABST_M + RELT_M(J)
           ENDIF
        endif
     ENDDO
     IF (ABST_M > ZERO) THEN
        ABST_VX = ABST_VX / ABST_M
        ABST_VY = ABST_VY / ABST_M
        ABST_VZ = ABST_VZ / ABST_M
        !           Remove center of mass velocity
        !$$$            IF(PRNLEV >= 3) WRITE(OUTU,*)
        !$$$     $           'Removing center of mass velocity...'
        !$$$            DO I = ATFRST,ATLAST
        !$$$               VX(I) = VX(I) - ABST_VX
        !$$$               VY(I) = VY(I) - ABST_VY
        !$$$               VZ(I) = VZ(I) - ABST_VZ
        !$$$            ENDDO
        !$$$         ELSE
        !$$$C           Remove center of mass velocity anyway...
        !$$$            DO I = ATFRST,ATLAST
        !$$$               ABST_VX = ABST_VX + AMASS(I)*VX(I)
        !$$$               ABST_VY = ABST_VY + AMASS(I)*VY(I)
        !$$$               ABST_VZ = ABST_VZ + AMASS(I)*VZ(I)
        !$$$               ABST_M = ABST_M + AMASS(I)
        !$$$            ENDDO
        !$$$            ABST_VX = ABST_VX / ABST_M
        !$$$            ABST_VY = ABST_VY / ABST_M
        !$$$            ABST_VZ = ABST_VZ / ABST_M
        !$$$            IF(PRNLEV >= 3) WRITE(OUTU,*)
        !$$$     $           'Removing center of mass velocity...'
        !$$$            DO I = ATFRST,ATLAST
        !$$$               VX(I) = VX(I) - ABST_VX
        !$$$               VY(I) = VY(I) - ABST_VY
        !$$$               VZ(I) = VZ(I) - ABST_VZ
        !$$$            ENDDO
     ENDIF
  ENDIF
  ! END DYNA VV2 (G. Lamoureux)
#endif /* (dynvv2_setup2)*/

  IF(QSTPRT) THEN
     CALL STOPRT(X,Y,Z,VX,VY,VZ,AMASS,IGVOPT,NATOM,IMOVE,QALWRT &
#if KEY_TSM==1
          ,BACKLS &
#endif
          )
     QSTPRT=.FALSE.
  ENDIF

  !
  !=======================================================================
  !
  !     At this point all of the input line should be parsed so anything
  !     remaining must be an error.  To avoid expensive mistakes we test
  !     and bomb out.
  !
  CALL XTRANE(COMLYN,COMLEN,'DCNTRL')
  IF (COMLEN > 0) CALL DIEWRN(-2)
  !
#if KEY_BLOCK==1 /*block_2*/
  IF (QPRNTV)  THEN
     IF(IOLEV > 0) THEN
        CALL WRTITL(TITLEA,NTITLA,NBLCKFEP,0)
        WRITE(NBLCKFEP,'(5(I6,1X))') NSTEP,IBLCKFEP,NDEGF,0,1
        ! NSTEP,IBLCKFEP,NDEGF,NPUMB,LPOWER: PNUMB=0, LPOWER=1 (Limitation)
     ENDIF
  ENDIF
#endif /* (block_2)*/

#if KEY_TSM==1
  ! tsm mod shf/djt 4/4/90
  IF(QTSM) THEN
     IF(SAVEP .AND. IOLEV > 0) THEN
        CALL WRTITL(TITLEA,NTITLA,PUNITX,0)
        WRITE(PUNITX,'(5(I6,1X))') NSTEP,PERFRQ,NDEGF,NPUMB,LPOWER
     ENDIF
  ENDIF
  IF (SLOWST) THEN
     ASLOW = ZERO
     LAMBDA = LMFROM
     DLMBDA = (LMTO - LMFROM)/NSTEP
     IF(PIGSET) CALL PIGMSET(AMASS,.FALSE.)
  ENDIF
  IF(NICP > 0 .AND. IUNICP > 0 .AND. IOLEV > 0) &
       WRITE(IUNICP,61) NICP,ICPINC,NDEGF,DELTA
61 FORMAT(3I6,F12.6)
#endif /*  TSM*/
  !
  !=======================================================================
  !=======================================================================
#if KEY_TPS==1 /*tps_loop_top*/
  !     ARD and M. Hagan for TPS
  !     TPS initializations
  QTPSRS = .FALSE.
  IF (INBCUT  >  0) CALL TPCINI()
  IF (QTPS) THEN
     NPSAVC = NSAVC
     NPSAVV = NSAVV
     CALL TPINIT(ISEED,NATOM,X,Y,Z,VX,VY,VZ,NSTEP,IREST,NSAVC, &
          IUNCRD,NSAVV,IUNVEL, &
          DELTA,TIMEST)
  ENDIF

  !     TPS main loop
819 IF (QTPS) THEN

     CALL LNGFIL(ILANG,IPSTOP,GAMMA,TBATH,DELTA,RBUF, &
          SXREF,SYREF,SZREF,X,Y,Z &
#if KEY_ACE==1
          ,CHRAD,BSARR,QLBACE         &
#endif
          )

     CALL TPMAIN(NSTEP,JHSTRT,IGVOPT,X,Y,Z,VX,VY, &
          VZ,NATOM,ISEED,FINALT,QALWRT, &
#if KEY_TSM==1
          BACKLS,                                   &
#endif
          DELTA,GAMMA,NDEGF,ILANG)

     CALL UPDATE(COMLYN,COMLEN,X,Y,Z,WMAIN,.FALSE.,.FALSE.,.TRUE., &
          .FALSE.,.TRUE.,1,XOLD,YOLD,ZOLD,VX,VY,VZ)

  ENDIF
#endif /*  (tps_loop_top)*/
  !
  ISTART=1
#if KEY_BLOCK==1 /*ldm*/
  if(prnlev >= 5)then
     IF(NTEMPLD > 0 .AND. IOLEV > 0) THEN
        WRITE (ITEMPLD,'(A)') &
             'Step     T(total)     T(atoms)     T(lambda)'
     ENDIF
  endif
#endif /*  BLOCK*/

#if KEY_SCCDFTB==1
#if KEY_PARALLEL==1
  IF (MYNOD == 0) THEN
#endif

     if(istart == 1.and.qsccb.and.(qlamda.or.qpkac)) then
        nsccl=int((nstep-sccpass)/sccstep)
        if(sccpass >= nstep) nsccl=0

        if((irest == 1).and.qsccres) then
           nsccl=nsccl+icntdyn
        else
           icntdyn=0
           iavti=0
           dtmp1=0.0d0
           dvdl=0.0d0
           dvdlav=0.0d0
        endif

     endif
# if KEY_PARALLEL==1
  ENDIF
# endif
#endif

#if KEY_BLOCK==1 /*ldm*/
  !ss for SS
  IF(QMCFR) call ldm_init_qmcfr
  !ss
#endif

#if KEY_SGLD==1
  !     Allocate and initialize SGLD arrays
#if KEY_DYNVV2==1 /*dynvv2*/
  IF(VVERL2) THEN
    TSGSET=RTMPR(1)
    I=NDGN(1)
    DO J = 2,NOBL
      IF(NDGN(J)>I)THEN
        TSGSET=RTMPR(J)
        I=J
      ENDIF
    ENDDO
  ELSE
#endif
    IF(QSGLD)TSGSET=TBATH
    IF(QSGMD)THEN
      IF(QNPT)TSGSET=REFT
      IF(QCNSTT)TSGSET=TREF
    ENDIF
#if KEY_DYNVV2==1 /*dynvv2*/
    ENDIF
#endif
  IF(QSGLD.OR.QSGMD)CALL PSGLD(NATOM,TIMEST,X,Y,Z,VX,VY,VZ)
#endif
! #if KEY_MOBHY==1
  IF (LMOBHY .and. (IHOPFR>0)) THEN
    if (NCYCLE.gt.IHOPFR) then
      NCYCLE=IHOPFR   ! check mod(IHOPFR,NYCLE) if NYCLE<IHOPFR ?
      IF (PRNLEV >= 2) write (OUTU,*) 'MOBHY> NCYCLE reduced to match IHOPFR'
    endif
    beta=ONE/(FINALT*KBOLTZ)
  endif
! #endif

  !=======================================================================
  !
  !     start the main loop over ncycle.
  !
#if KEY_TMD==1
  ! Zero the counter for stopping the rotation in TMD method.
  if(qtmd) then
     icnt=0
  endif
#endif
#if KEY_REPDSTR==1
#if KEY_DOMDEC==1
  ! When repdstr + domdec active, pure recip nodes will jump down here
750 continue
#endif
#endif
  dynloop: do while(istart <= nstep)
#    if KEY_REPDSTR==1
#    if KEY_DOMDEC==1
     ! For repdstr + domdec, send pure recip nodes to right before exchange calc
     if ( q_repd_domdec_purerecip ) goto 760
#    endif
#    endif
     call chklim
     if(atlim) then
        if(wrnlev >= 2) write(outu,"(/'  DCNTRL:   EXITING DUE TO ',A,' DEADLINE'/)") limtyp
        !  clean-up-and-exit
        call clean_up_and_exit
        return
     endif

     istop=min(istart+ncycle-1,nstep)
     done=(istop == nstep)
     ipstop=istart-1

#if KEY_RMD==1
     ! Surface crossing : revise time if crossing has occured
     if(qxtime) then
        iseed=savedseed
        istart=istart-xtime
        istop=istop-xtime
        qxtime=.false.
     endif
#endif

     !
     ! update-the-langevin-dynamics-arrays
     !     Update Langevin dynamics
     IF ( (ORIG.AND.(.NOT.VVERL)) &
#if KEY_MTS==1
          .OR. QLNX  &
#endif
          ) THEN

#if KEY_OLDDYN==1
        IF (ILANG == 1 .and. .not. want_openmm) THEN
           IF(IPSTOP == 0) THEN
              CALL LNGFILV(GAMMA,RFD,TBATH,DELTA,RBUF,SXREF,SYREF, &
                   SZREF,X,Y,Z &
# if KEY_ACE==1
                   ,CHRAD,BSARR,QLBACE         &
# endif
                   )
           ELSE
              IF (ILBFRQ /= 0) THEN
                 IF (MOD(IPSTOP,ILBFRQ) == 0) THEN
# if KEY_ACE==1
                    IF (QLBACE .AND. (PRNLEV >= 2)) &
                         WRITE(OUTU,'(A,A,I12)') &
                         'DCNTRL: Langevin parameter ', &
                         'update at step ',IPSTOP
# endif
                    CALL LNGFILV(GAMMA, &
                         RFD,TBATH,DELTA,RBUF,SXREF,SYREF,SZREF,X,Y,Z &
# if KEY_ACE==1
                         ,CHRAD,BSARR,QLBACE             &
# endif
                         )
                 ENDIF
              ENDIF
           ENDIF
        ENDIF
#else /**/
        CALL WRNDIE(-3,'<DCNTRL>','Old code not compiled.')
#endif
     ELSE IF(.NOT. VVERL) THEN
        QLANG=(ILANG == 1)
        IF(QLANG.AND.ILBFRQ /= 0) QLANG=(MOD(IPSTOP,ILBFRQ) == 0)
        IF(IPSTOP == 0) QLANG=.TRUE.
        IF(QLANG .and. .not. want_openmm .and. .not. want_blade) THEN
#if KEY_ACE==1
           IF (QLBACE .AND. (PRNLEV >= 2) .AND. (IPSTOP /= 0)) &
                WRITE(OUTU,'(A,A,I12)') 'DCNTRL: Langevin ', &
                'parameter update at step ',IPSTOP
#endif
           !sb (note: lngfil2hb and lngfil could easily be merged)
           if (ndrude > 0) then
              CALL LNGFIL2HB(ILANG,IPSTOP,GAMMA,TBATH,DELTA,RBUF, &
                   SXREF,SYREF,SZREF,X,Y,Z&
#if KEY_ACE==1
              ,CHRAD,BSARR,QLBACE             &
#endif
              )
           else
              CALL LNGFIL(ILANG,IPSTOP,GAMMA,TBATH,DELTA,RBUF, &
                   SXREF,SYREF,SZREF,X,Y,Z &
#if KEY_ACE==1
              ,CHRAD,BSARR,QLBACE             &
#endif
              )
           endif
#if KEY_BLOCK==1 /*ldm*/
           if (qmld) then
              IF(TBLD /= 0.0) THEN
                 CALL LNGFIL2THETA(ILALDM,IPSTOP,GAMMATHETA &
                      ,TBLD,DELTA,THETABIB,THETAM)
              ELSE
                 CALL LNGFIL2THETA(ILALDM,IPSTOP,GAMMATHETA &
                      ,TBATH,DELTA,THETABIB,THETAM)
              ENDIF
           else
              IF(ILALDM) THEN
                 IF(QTHETADM) THEN   !-> yw070702
                    IF(TBLD /= 0.0) THEN
                       CALL LNGFIL2THETA(ILALDM,IPSTOP,GAMMATHETA &
                            ,TBLD,DELTA,THETABIB,THETAM)
                    ELSE
                       CALL LNGFIL2THETA(ILALDM,IPSTOP,GAMMATHETA &
                            ,TBATH,DELTA,THETABIB,THETAM)
                    ENDIF
                 ELSE                !<- if(qthetadm) wy070702
                    IF(TBLD /= 0.0) THEN
                       CALL LNGFIL2(ILALDM,IPSTOP,IGAMMALD &
                            ,TBLD,DELTA,BIBLAM,NBLOCK,BIMLAM)
                    ELSE
                       CALL LNGFIL2(ILALDM,IPSTOP,IGAMMALD &
                            ,TBATH,DELTA,BIBLAM,NBLOCK,BIMLAM)
                    ENDIF
                 ENDIF               ! if(qthetadm) wy070702
              ENDIF
           endif
#endif /*  BLOCK*/
        ENDIF
     ENDIF
     !
     !=======================================================================
     !
     !=======================================================================
     ! Last modified November 2007, H Kamberaj
#if KEY_TSALLIS==1 /*tsallis_beta*/
     !        Set the temperature factor for Tsallis dynamics
     !         IF (QTSALL) TSBETA = ONE / (KBOLTZ * TEMNEW)
     if (qtsall .or. qttsall) then
        tsbeta = one / (kboltz * finalt)
        IF ( PRNLEV  >  6 ) THEN
#if KEY_PARALLEL==1
           IF ( MYNOD  ==  0 )  THEN
#endif
              write(outu,'(A8,F12.5)') "Temp=", finalt
              write(outu,'(A8,F12.5)') "tsbeta=",tsbeta
#if KEY_PARALLEL==1
           ENDIF
#endif
        ENDIF
     endif
#endif /* (tsallis_beta)*/
     !=======================================================================
     !
     ! -- mikem ->
#if KEY_QUANTA==1
     TERMINATEDYN=.FALSE.
#endif

#if KEY_LIBRARY == 1
     if (custom_dynam_is_set()) then
        custom_ret_val =  custom_dynam_call(istart, natom, &
             vx, vy, vz, &
             xnew, ynew, znew, &
             xold, yold, zold)
     end if
#endif

     ! call-dynamc-and-take-care-of-equilibration
     !
     !ds...B981203.ds2 DYNAMLN calling conditional fix
     !ds         IF(ORIG) THEN
     IF( (ORIG) &
#if KEY_MTS==1
          .AND.(.NOT.QLNX) &
#endif
          ) THEN
#if KEY_OLDDYN==1
        IF(VVERL) THEN
#endif
#if KEY_DYNVV2==1 /*dynvv2_call*/
           ! BEGIN DYNA VV2 (G. Lamoureux)
           IF(VVERL2) THEN
#if KEY_BLOCK==1
              IF (QLDM) CALL WRNDIE(-3,'<DYNAMVV2>','LDM not implemented')     /*ldm*/
#endif
              call timer_stop(T_dcntrl)
              CALL DYNAMVV2(VX,VY,VZ,VK,XNEW,YNEW,ZNEW, &
                   XOLD,YOLD,ZOLD, &
                   AMASS,NPRIV,NDEGF,IGVOPT, &
                   IMOVE,ISKPR,FREEAT,NFREAT,NATOM, &
                   BNBND,BIMAG, &
                   ISTART,ISTOP,IPRFRQ,IDYNPR, &
                   JHTEMP,GAMMA,RFD,RFT,QAVER,NAVER, &
                   IXAVE,IYAVE,IZAVE &
                   ,QKUHEAD &
#if KEY_MTS==1
                   ,XMI,YMI,ZMI,XMM,YMM,ZMM &
#endif
                   ,XNSE,YNSE,ZNSE &
                   )
              call timer_start(T_dcntrl)
           ELSE
              ! END DYNA VV2 (G. Lamoureux)
#endif /* (dynvv2_call)*/
              !
              call timer_stop(T_dcntrl)
              CALL DYNAMVV(VX,VY,VZ,VK,XNEW,YNEW,ZNEW, &
                   XOLD,YOLD,ZOLD, &
#if KEY_DHDGB==1
!AP/MF
                   VS_DHDGB,SDEFNEW,SDEFOLD, &
                   SDEFLD,VK_DHDGB, &
#endif
#if KEY_CHEQ==1
                   CG,CGNEW,CGOLD,VCG,pmassq,            &
#endif
#if KEY_PIPF==1
                   UINDN,UINDO,VUIND,IUMAS, &
#endif
                   AMASS,NPRIV,NDEGF,IGVOPT, &
                   IMOVE,ISKP,FREEAT,NFREAT,NATOM, &
                   BNBND,BIMAG, &
                   ISTART,ISTOP,IPRFRQ,IDYNPR, &
                   JHTEMP,GAMMA,RFD,RFT,QAVER,NAVER, &
                   IXAVE,IYAVE,IZAVE &
#if KEY_CHEQ==1
                   ,ICGAVE                               &
#endif
                   ,QKUHEAD &
#if KEY_MTS==1
                   ,XMI,YMI,ZMI,XMM,YMM,ZMM &
#endif
                   ,XNSE,YNSE,ZNSE &
                   )
              call timer_start(T_dcntrl)
              ! BEGIN DYNA VV2 (G. Lamoureux)
#if KEY_DYNVV2==1
           ENDIF
#endif
           ! END DYNA VV2 (G. Lamoureux)
#if KEY_OLDDYN==1
        ELSE
           call timer_stop(T_dcntrl)

           CALL DYNAMCV(VX,VY,VZ,VK,XNEW,YNEW,ZNEW, &
                XOLD,YOLD,ZOLD, &
                AMASS,NPRIV,NDEGF,IGVOPT, &
                IMOVE,ISKP,FREEAT,NFREAT,NATOM, &
                BNBND,BIMAG,ISTART,ISTOP,IPRFRQ,IDYNPR, &
                JHTEMP,GAMMA,RFD,RFT,QAVER,NAVER, &
                IXAVE,IYAVE,IZAVE &
                ,NDRUDE,ISDRUDE &
                ,QKUHEAD &
#if KEY_TSM==1
                ,REACLS,PRODLS,PIGGLS,BACKLS &
#endif
                ,XNSE,YNSE,ZNSE &
                )
           call timer_start(T_dcntrl)
        ENDIF  ! IF VVERL
#endif /*  OLDDYN*/
        !
#if KEY_MTS==1
     ELSEIF (QLNX) THEN  ! IF ORIG
        ! A. Sandu. Call to DYNAMLN (the LN integrator)
        call timer_stop(T_dcntrl)

        CALL DYNAMLN(VX,VY,VZ,VK,XNEW,YNEW,ZNEW, &
             XOLD,YOLD,ZOLD, &
             AMASS,NPRIV,NDEGF,IGVOPT, &
             IMOVE,ISKP,FREEAT,NFREAT,NATOM, &
             BNBND,BIMAG, &
             ISTART,ISTOP,IPRFRQ,IDYNPR, &
             JHTEMP,GAMMA,RFD,RFT,QAVER,NAVER, &
             IXAVE,IYAVE,IZAVE &
             ,XMI,YMI,ZMI &
             ,XMM,YMM,ZMM &
             !     &            ,XML,YML,ZML
             ,XNSE,YNSE,ZNSE &
             )
        call timer_start(T_dcntrl)
#endif /*  MTS (DYNAMLN)*/
        !
     ELSE IF (want_openmm) THEN
        call timer_stop(T_dcntrl)
        ! Zero eneergy terms as we enter dynamics
        eprop = zero
        eterm = zero
        if (IPSTOP == 0) then
           ! XXX set scale factor for XOLD
           gamma = zero
           gamma(3*natom+1:4*natom) = half / delta
        endif
#if KEY_OPENMM==1
        omm_opt%temperatureReference = finalt
        call omm_dynamics(omm_opt, VX, VY, VZ, XOLD, YOLD, ZOLD, &
              JHTEMP, GAMMA, NDEGF, IGVOPT, NPRIV, ISTART, ISTOP, &
              IPRFRQ, ISVFRQ, NTRFRQ, RUNOK)
#endif
        call timer_start(T_dcntrl)
     ELSE IF (want_blade) THEN
        call timer_stop(T_dcntrl)
        ! Zero eneergy terms as we enter dynamics
        eprop = zero
        eterm = zero
        if (IPSTOP == 0) then
           ! XXX set scale factor for XOLD
           gamma = zero
           gamma(3*natom+1:4*natom) = half / delta
        endif
#if KEY_BLADE==1
        call blade_dynamics(blade_opt, VX, VY, VZ, XOLD, YOLD, ZOLD, &
           jhtemp, gamma, ndegf, igvopt, npriv, ISTART, ISTOP, &
           iprfrq, isvfrq)
#endif
        call timer_start(T_dcntrl)
     ELSE  ! IF ORIG
        !
#if KEY_FOURD==1 /*4ddyna*/
        IF(DIM4) THEN
           call timer_stop(T_dcntrl)

           CALL DYNAMC4(VX,VY,VZ,VK,XNEW,YNEW,ZNEW,XOLD, &
                YOLD,ZOLD, &
                AMASS, FDNEW,FDOLD,FDAVE,VFD, &
                NPRIV,NPRIVOLD,NDEGF,NDEG4, &
                IGVOPT,IMOVE,ISKP,FREEAT,NFREAT, &
                NATOM,BNBND,BIMAG,ISTART,ISTOP,IPRFRQ,IDYNPR, &
                JHTEMP,GAMMA,QAVER,NAVER, &
                IXAVE,IYAVE,IZAVE, &
                QKUHEAD,RUNOK, &
                IEQ4,IHT4,TIN4,FSTT4,FNLT4,TWH4,TWL4,ICH4, &
                IASORS,ISCVEL,IASVEL)
           call timer_start(T_dcntrl)
        ELSE
#endif /* (4ddyna)*/
#if KEY_CHEQ==1
           IF(NPRIV == 0) THEN
              DO I =1,NATOM
                 VCG(I)=ZERO
                 CGOLD(I)=CG(I)
              ENDDO
           ENDIF
#endif

           !           dynamc uses child timers of T_dcntrl; so control of T_dcntrl
           !           is inside dynamc.
         CALL DYNAMC(VX,VY,VZ,VK,XNEW,YNEW,ZNEW,XOLD, &
                YOLD,ZOLD, &
#if KEY_CHEQ==1
                CG,CGNEW,CGOLD,VCG,pmassq,        &
#endif
#if KEY_PIPF==1
                UINDN,UINDO,VUIND,IUMAS,          &
#endif
                AMASS,NPRIV,NPRIVOLD,NDEGF,IGVOPT, &
                IMOVE,ISKP,FREEAT,NFREAT,NATOM, &
                BNBND,BIMAG,ISTART,ISTOP,IPRFRQ,IDYNPR, &
                JHTEMP,GAMMA,QAVER,NAVER, &
                IXAVE,IYAVE,IZAVE, &
#if KEY_CHEQ==1
                ICGAVE,                          &
#endif
                QKUHEAD,RUNOK &
#if KEY_TSM==1
                ,REACLS,PRODLS,PIGGLS,BACKLS &
#endif
#if KEY_BLOCK==1
                ,QPRNTV                   &
#endif
#if KEY_TNPACK==1
                ,QEULER,QESTRT,QEHARM,KEHARM &
#endif
#if KEY_TNPACK==1
                ,RXHARM,RYHARM,RZHARM,LIEFF &
#endif
#if KEY_TSALLIS==1
                ,QTSALL,TSEMIN,TSQ,TSBETA &
#endif
#if KEY_TPS==1
                ,QTPS,INBCUT &
#endif
               )

#if KEY_TMD==1
           if(qtmd.and.(.not.qzeta).and.(tmdrhof <= tmdfrms))then
              nstep=istart
              write(outu,'(" STOPPED DYNAMICS")')
           endif
#endif
#if KEY_DIMS==1
           IF (DDONE) THEN
              NSTEP=ISTART
              WRITE(OUTU,'("DIMS> Dims is done, halting ... ")')
           ENDIF
#endif
#if KEY_FOURD==1 /*4dendif*/
        ENDIF
#endif /* (4dendif)*/
        !
        !=======================================================================
        IF(.NOT.RUNOK) THEN
           !             OK, so it didn't work.  Free space and exit gracefully.
           call free_dcntrl_space
           PRNLEV=PRLOLD
           RETURN
#if KEY_TPS==1
        ELSE IF (INBCUT  >  0) THEN
           !            Commitment to a basin.
           !            Terminate dynamics simulation and return basin in INBA.
           CALL set_param('INBA',IBASIN)
           IF (ABS(IBASIN)  >=  INBCUT) THEN
              call free_dcntrl_space
              PRNLEV=PRLOLD
              RETURN
           ENDIF
#endif
        ENDIF
        !
     ENDIF  ! IF ORIG
     !
#if KEY_PARALLEL==1
     !..MFC This has to be disabled for parallel since not all nodes
     !      will have the same PRNLEV, and a run will hang while
     !      nodes with PRNLEV > 4 do an energy and wait in a collect
     !      that the other nodes never participate in.
     IDYNPR=1
#else /**/
     IF(PRNLEV <= 5) IDYNPR=1
#endif
     AVETEM=JHTEMP/JHSTRT
     JHLAST=JHSTRT
#if KEY_PARALLEL==1
     QBROADV=.FALSE.
     QBROADC=.FALSE.
#endif
     !
     ! Main conditional for type of dynamics control for heating and equil.
     !
#if KEY_CHEQ==1
     IF (QCG.AND.CGEQ >= 0) THEN
        IF (CGEQ == 0) THEN
           !C Zero VCG for CGEQ=0
           !             IF (PRNLEV >= 2)
           !     $          WRITE(OUTU,*)'VCG ZEROED at STEP ',NPRIV
           !             DO I=1,NATOM
           !               VCG(I)=ZERO
           !             ENDDO
        ELSE IF (CGEQ > 1) THEN
           ! Simple scaling for CGEQ > 1
           KECGTMP=ZERO
           DO I=1,NATOM
              KECGTMP=KECGTMP+MASSQ*VCG(I)*VCG(I) ! 2 times kinetic energy
           ENDDO
           IF (KECGTMP > KECGC) THEN
              VCGSCA=SQRT(KECGC/KECGTMP)
              IF (PRNLEV >= 2) THEN
                 WRITE(OUTU,*)'VCG RESCALED at STEP: ',NPRIV
                 WRITE(OUTU,*)'SCALED TMP          : ',TCG
                 WRITE(OUTU,*)'SCALE FACTOR IS     : ',VCGSCA
              ENDIF
              DO I=1,NATOM
                 VCG(I)=VCG(I)*VCGSCA
              ENDDO
           ENDIF
        ELSE IF (CGEQ == 1) THEN
           ! VCG reassigned according to gaussian distribution if CGEQ=1
           !             KECGTMP=ZERO
           !             DO I=1,NATOM
           !               KECGTMP=KECGTMP+MASSQ*VCG(I)*VCG(I) ! 2 times kinetic energy
           !             ENDDO
           !             IF (KECGTMP > KECGC) THEN
           !               IF (PRNLEV > 2) THEN
           !                 WRITE(OUTU,*)'VCG REASSIGNED at STEP: ',NPRIV
           !                 WRITE(OUTU,*)'CG TMP IS             : ',TCG
           !               ENDIF
           !               SD=TCG*KBOLTZ/MASSQ
           !               SD=SQRT(SD)
           !               DO I=1,NATOM
           !                 CALL GAUSSI(ZERO,SD,VEL,ISEED,1)
           !                 VCG(I)=VEL
           !               ENDDO
           !               IF (QPOLAR1) THEN
           !                 DO I=1,MOLNT
           !                    IF (.NOT.(I == IPOLAR1)) THEN
           !                    I1=MOLPTR(I)
           !                    J1=MOLPTR(I+1)-1
           !                    DO J=I1,J1
           !                      VCG(MOLBL(J))=ZERO
           !                    ENDDO
           !                    ENDIF
           !                  ENDDO
           !                ELSE IF (.NOT.QMOLC) THEN      ! global charge normalization
           !                  KECGTMP=ZERO
           !                  DO I=1,NATOM
           !                     KECGTMP=KECGTMP+VCG(I)
           !                  ENDDO
           !                  IF (KECGTMP /= 0.0) THEN
           !                    KECGTMP=KECGTMP/NATOM
           !                    DO I=1,NATOM
           !                      VCG(I)=VCG(I)-KECGTMP
           !                    ENDDO
           !                  ENDIF
           !                ELSE ! charges are constrained to individual molecules
           !                  DO I=1,MOLNT
           !                    LAMDAQ(I)=ZERO
           !                  ENDDO
           !                  DO I=1,NATOM
           !                    K=MOLT(I)
           !                    LAMDAQ(K)=LAMDAQ(K)+VCG(I)
           !                  ENDDO
           !                  DO I=1,MOLNT
           !                    I1=MOLPTR(I)
           !                    J1=MOLPTR(I+1)
           !                    LAMDAQ(I)=LAMDAQ(I)/(J1-I1)
           !                  ENDDO
           !                  DO I=1,NATOM
           !                    K=MOLT(I)
           !                    VCG(I)=VCG(I)-LAMDAQ(K)
           !                  ENDDO
           !                ENDIF
           !              ENDIF     ! KECGTMP > KECGC
        ENDIF       ! cgeq
     ENDIF         ! qcg
#endif


     ! If constant temperature run, then disable usual heating and equil.
     IF(QCNSTT) THEN
        CONTINUE
        ! If langevin dynamics, then disable usual heating and equilibration.
     ELSE IF(ILANG == 1) THEN
        CONTINUE
        ! Nose-Hoover dynamics, then disable usual heating and equilibration
     ELSE IF(QNOSE) THEN
#if KEY_CONSHELIX==1
        LQSTPRT=QSTPRT
#endif
        CONTINUE
        !
        ! Heating is done here until the final temperature should have
        ! been reached, (FINALT-FIRSTT)/TEMINC cycles, then control is
        ! switched to equilibration.
     ELSE IF(IHTFRQ /= 0) THEN
        IF(MOD(NPRIV,IHTFRQ) == 0) THEN
           IF(TEMINC /= 0) THEN
              I=(FINALT-FIRSTT)/TEMINC
           ELSE
              I=0
           ENDIF
           J=NPRIV/IHTFRQ
           TEMNEW=FIRSTT+J*TEMINC
           IF(J > I) THEN
              TEMNEW=FINALT
              IHTFRQ=0
              IF(PRNLEV >= 2) WRITE(OUTU,2030) NPRIV,FINALT
2030          FORMAT(' ** INFO ** DYNAMICS CONTROL CHANGED FROM', &
                   ' HEATING TO EQUILIBRATION AT STEP',I6/ &
                   ' EQUILIBRATION TEMPERATURE =',F10.2)
           ENDIF
           IF(TEMNEW < 0.0) TEMNEW=0.0
           !
           IF(IASORS /= 0) THEN
              IF(TEMNEW > AVETEM) TEMNEW=TEMNEW+(TEMNEW-AVETEM)*0.5
              !
#if KEY_BLOCK==1
                  if (qmld) call msld_assignvelocities(temnew,nblock,&
                             bivlam,iseed,iasvel)
#endif
              CALL ASSVEL(TEMNEW,X,Y,Z,VX,VY,VZ, &
                   AMASS,ISEED,IASVEL,IGVOPT,NATOM,IMOVE &
#if KEY_TSM==1
                   ,BACKLS &
#endif
#if KEY_DHDGB==1
!AP/MF
                   , QFHDGB, VS_DHDGB, SAMASS, TOTALS &
#endif

                   )
           ELSE
              !
              !C This is the correct expression to give proper equipartition,
              !C but we keep the old one for historical reasons - BRB 4/15/92
              !C      HEAT=SQRT(MAX(TWO*TEMNEW/AVETEM-ONE,ZERO))
              HEAT2=TEMNEW/AVETEM
              HEAT=ZERO
              IF(HEAT2 > ZERO) HEAT=SQRT(HEAT2)
              !
#if KEY_PARALLEL==1
#if KEY_SPACDEC==1 /*spacdec*/
              IF(.NOT.QBROADV) THEN
                 CALL SPACBR(VX,NATOM,ICPUMAP)
                 CALL SPACBR(VY,NATOM,ICPUMAP)
                 CALL SPACBR(VZ,NATOM,ICPUMAP)
              ENDIF
              CALL SPACBR(VK,NATOM,ICPUMAP)
#else /* (spacdec)*/
#if KEY_DOMDEC==1 /*domdec*/
              if (q_domdec) then
                 if (.not.qbroadv) call copy_to_all(vx, vy, vz)
                 call copy_to_all(vk)
              else
#endif /* (domdec)*/
                 IF(.NOT.QBROADV) CALL VDGBR(VX,VY,VZ,1)
                 CALL VDGBRE(VK,IPARPT)
#if KEY_DOMDEC==1
              endif
#endif
#endif /* (spacdec)*/
              QBROADV=.TRUE.
#endif
              CALL SCAVEL(HEAT,X,Y,Z,VX,VY,VZ,VK, &
                   AMASS,NDEGF,ISCVEL,IGVOPT,NATOM,IMOVE, &
                   AVETEM,JHSTRT,QSHAKE,IDGF2)
           ENDIF
           QSTPRT=.TRUE.
           JHSTRT=0
        ENDIF
        !
        ! Equilibiration is done here.  the velocities are overscaled
        ! by a factor of two to give a correct reequilibration of
        ! harmonic systems when both momentum and pe modes are considered.
     ELSE IF(IEQFRQ > 0) THEN
        IF(MOD(NPRIV,IEQFRQ) == 0) THEN
           DELTEM=AVETEM-FINALT
           IF (ICHECW == 0.OR.DELTEM <= TWINDL.OR.DELTEM >= TWINDH) &
                THEN
              IF(IASORS /= 0) THEN
                 TEMNEW=FINALT
                 !
#if KEY_BLOCK==1
                     if (qmld) call msld_assignvelocities(temnew,nblock,&
                                bivlam,iseed,iasvel)
#endif
                 CALL ASSVEL(TEMNEW,X,Y,Z,VX,VY,VZ, &
                      AMASS,ISEED,IASVEL,IGVOPT,NATOM,IMOVE &
#if KEY_TSM==1
                      ,BACKLS &
#endif
#if KEY_DHDGB==1
!AP/MF
                      , QFHDGB, VS_DHDGB, SAMASS, TOTALS &
#endif
                      )
              ELSE
                 HEAT2=TWO*FINALT/AVETEM-ONE
                 HEAT=ZERO
                 IF(HEAT2 > ZERO) HEAT=SQRT(HEAT2)
#if KEY_PARALLEL==1
#if KEY_SPACDEC==1 /*spacdec*/
                 IF(.NOT.QBROADV) THEN
                    CALL SPACBR(VX,NATOM,ICPUMAP)
                    CALL SPACBR(VY,NATOM,ICPUMAP)
                    CALL SPACBR(VZ,NATOM,ICPUMAP)
                 ENDIF
                 CALL SPACBR(VK,NATOM,ICPUMAP)
#else /* (spacdec)*/
#if KEY_DOMDEC==1 /*domdec*/
                 if (q_domdec) then
                    if (.not.qbroadv) call copy_to_all(vx, vy, vz)
                    call copy_to_all(vk)
                 else
#endif /* (domdec)*/
                    IF(.NOT.QBROADV) CALL VDGBR(VX,VY,VZ,1)
                    CALL VDGBRE(VK,IPARPT)
#if KEY_DOMDEC==1
                 endif
#endif
#endif /* (spacdec)*/
                 QBROADV=.TRUE.
#endif
                 CALL SCAVEL(HEAT,X,Y,Z,VX,VY,VZ,VK, &
                      AMASS,NDEGF,ISCVEL,IGVOPT,NATOM,IMOVE, &
                      AVETEM,JHSTRT,QSHAKE,IDGF2)
              ENDIF
              QSTPRT=.TRUE.
           ENDIF
           JHSTRT=0
        ENDIF
     ENDIF
     !
     ! Stop rotation ( if necessary ).  rotation is also stopped whenever
     ! scaling or assignment of velocities is done so do not repeat it.
     !
#if KEY_CONSHELIX==1
         LQSTPRT=QSTPRT
#endif
     IF(NTRFRQ /= 0) THEN
        IF(MOD(NPRIV,NTRFRQ) == 0) QSTPRT=.TRUE.
     ENDIF
     !
     ! Remove net translation/rotation for selected global motion.
     IF(QSTPRT) THEN
#if KEY_PARALLEL==1
#if KEY_SPACDEC==1
        IF(.NOT.QBROADV) THEN
           CALL SPACBR(VX,NATOM,ICPUMAP)
           CALL SPACBR(VY,NATOM,ICPUMAP)
           CALL SPACBR(VZ,NATOM,ICPUMAP)
        ENDIF
#else /**/
#if KEY_DOMDEC==1
        if (q_domdec) then
           if (.not.qbroadv) call copy_to_all(vx, vy, vz)
        else
#endif
           IF(.NOT.QBROADV) CALL VDGBR(VX,VY,VZ,1)
#if KEY_DOMDEC==1
        endif
#endif
#endif
        QBROADV=.TRUE.
#endif
        CALL STOPRT(X,Y,Z,VX,VY,VZ,AMASS,IGVOPT,NATOM,IMOVE,QALWRT &
#if KEY_TSM==1
             ,BACKLS &
#endif
             )
        !mw...reset Nose heat bath energy to zero, 12-Feb-96
        ! BEGIN TPCONTROL (G. Lamoureux)
        !           IF(QNOSE.OR.QNOSP) THEN
        !           Don't reset the thermostat variables for TPCONTROL
        IF((QNOSE.OR.QNOSP) .AND. .NOT.QTPCON) THEN
           ! END TPCONTROL (G. Lamoureux)
           DO ITM=1,MAXNOS
              SN11(ITM) = 0.0
              SN12(ITM) = 0.0
              SN13(ITM) = 0.0
              SNH(ITM) = 0.0
              SNHV(ITM) = 0.0
              SNHF(ITM) = 0.0
           ENDDO
        ENDIF
        !mw...
        JHSTRT=0
        QSTPRT=.FALSE.
     ENDIF
     !
#if KEY_TMD==1
     ! Stopping the rotation in TMD method (every INRT steps).
     IF(QTMD) THEN
        !
        !Stop the rotation using the existing CHARMM code.
        icnt=icnt+ncycle
!!LNI Bugfix, August 2016
        if(inrt .le. 0) CALL WRNDIE(-4,'<DCNTRL>', &
                'Frequency for stopping rotation in TMD is zero or negative.')
        if(mod(ncycle,inrt)  /=  0) CALL WRNDIE(-4,'<DCNTRL>', &
                'Frequency for stopping rotation in TMD is wrong, check INRT.')
        !
        if(icnt  ==  inrt) then
           call chmalloc('dcntrl.src','DCNTRL','iscr1',natom+natom,intg=iscr1)
           call chmalloc('dcntrl.src','DCNTRL','iscr2',natom,crl=iscr2)
           call stoprt_tmd(x,y,z,natom,amass,ixtar,iytar,iztar, &
                natom,amass,iscr1,iscr2,istmd)
           ! MSF
           !     fit the 2nd target to the current structure
           !     Only change xtar-->xtar2, ytar-->ytar2, ztar-->ztar2
           IF(QZETA) THEN
              call stoprt_tmd(x,y,z,natom,amass,ixtar2,iytar2,iztar2, &
                   natom,amass,iscr1,iscr2,istmd)
           ENDIF
           call chmdealloc('dcntrl.src','DCNTRL','iscr1',natom+natom,intg=iscr1)
           call chmdealloc('dcntrl.src','DCNTRL','iscr2',natom,crl=iscr2)
           !
           icnt=0
           if(prnlev >= 5) write(outu,'(A,i5,A)') &
                'DYNAMC: ROTATION WAS STOPPED FOR TMD EACH  ', &
                inrt,' STEPS'
           !
           ! Meanwhile print out the current true RMS distance.
           ! MSF
           IF(.not.QZETA) THEN
              if(prnlev >= 5) write(outu,'(A,f10.5)') &
                   'DYNAMC: THE CURRENT TOTAL RMS DISTANCE ', &
                   tmdrhof
           ELSE
              if(prnlev >= 5) write(outu,'(A,f10.5)') &
                   'DYNAMC: CURRENT ZETA VALUE for TMD CONSTRAINT ', &
                   tmdrhof
           ENDIF
        endif
        !
     ENDIF
#endif
     NXTISTART = ISTART + NCYCLE
     IDIDPHREX = 0

     !
     !     Replica exchange call
#if KEY_REPDSTR==1
#    if KEY_DOMDEC==1
760  continue
     if (q_repd_domdec_purerecip) then
       ! Pure recip nodes will wait inside energy_recip before replica exchange.
       ! When direct nodes enter exchange_calc they will tell pure recip nodes
       ! to exit energy_recip so they can join the direct nodes in
       ! exchange_calc.
       call timer_stop(T_dcntrl)
       call energy_recip(x, y, z, dx, dy, dz, .false.)
       call timer_start(T_dcntrl)
       call stop_split_direct_recip()
       if (nxtistart > nstep) then
         ! Finished! Leave now.
         return
       endif
       ! If here, not yet finished. Head down to replica exchange calc.
     endif
#    endif /* KEY_DOMDEC */
     if (q_replica_exchange) then
       if (phval.ge.-9000.0) then
         ! NOTE: Ok to reference mynod here since REPDSTR requires PARALLEL
         if (mynod.eq.0) then 
           do btmi=1,nres
             ophstate(btmi)=tstate(btmi)
           enddo
         endif 
       endif
       q_repd_want_blade=want_blade
       call exchange_calc(x,y,z,wmain,vx,vy,vz,xold,yold,zold,eprop(epot),eprop(temps),&
                          temnew,nxtistart,jhstrt,iseed,iasvel,igvopt &
#                         if KEY_TSM==1
                          ,backls &
#                         endif
                          ,ididphrex)
       if (repd_update_temps) then
         finalt=tempcurrent
         firstt=tempcurrent
         tbath=tempcurrent
         rtmpr(1)=tempcurrent
         reft=tempcurrent
       endif
     endif ! q_replica_exchange
#    if KEY_DOMDEC==1
     q_repd_domdec_purerecip=(q_domdec.and.qrepdstr.and.(.not.q_direct_node).and.q_recip_node)
     if (q_repd_domdec_purerecip) then
       ! Send pure recip node back up to before exchange calc.
       goto 760
     endif
#    endif /* KEY_DOMDEC */
#endif /* KEY_REPDSTR */

     !    Constant pH call
     !================================================================
     if(ididphrex == 0.and.phval >= -9000.0) then
        if(mod(nxtistart-1,nphfreq) == 0) &
           call dophmc(nmcstep,phval,igvopt,iasvel,iseed,x,y,z,dx,dy,dz,phunum,phtemp,nxtistart)
           CALL WRITE_PH_STATE(PHUNUM,NXTISTART,OPHSTATE)
     else if(phval >= -9000.0) then
#if KEY_PARALLEL==1
        if(mynod == 0.and.mod(nxtistart-1,nphfreq) == 0) then
#endif
           if(prnlev >= 3) write(outu,'(a,i8)') 'dcntrl> skipping constant ph mc on step ', nxtistart
           call flush(outu)
           call write_ph_state(phunum,nxtistart,ophstate)
#if KEY_PARALLEL==1
        endif
#endif
     endif
!#if KEY_MOBHY==1
!Themis Lazaridis. MOBHY proton hop attempt.
IF (LMOBHY .and. (IHOPFR>0)) THEN
IF ((MOD(NXTISTART-1,IHOPFR) == 0) .and. (NXTISTART.lt.NSTEP)) THEN
! To make it work for the old Verlet you need to also change the OLD coordinates,
! in addition to the current coordinates
! IF (ORIG) CALL WRNDIE(-5,'<DCNTRL>','ORIG integrator not supported in MOBHY')
! Need to get fresh list of h bonds. Normal update will be done in next cycle
  IHBFRQ=1
  call update("",0,x,y,z,wmain,.false.,.false.,.false.,.true.,.false.,0, &
            [zero],[zero],[zero],[zero],[zero],[zero])
  if (Ldebug) then
    write (6,*) 'Step,NHB',Nxtistart-1,NHB
    do k=1,nhb
    write (6,*) k,KHB(k),JHB(k)
    enddo
  endif
  JUSTHOPPED=.false.
! Save the acceptors because Arg needs it. Go backwards so that you save
! the first hb, if there are more than one
  do k=nhb,1,-1
  i=KHB(k)
  j=JHB(k)
  IACCHB(i)=j
  HBdone(k)=.false.
  enddo
  accept=.false.
! The order of attempts should be random. For now, either go forward or backward
  tempx=oldrandom(iseed1)
  if (tempx.lt.0.5) then
    j1=1
    j2=Nres
    jinc=1
  else
    j1=Nres
    j2=1
    jinc=-1
  endif
  outer: do j=j1,j2,jinc
    if (JUSTHOPPED(j) .eqv. .true.) cycle
    xres=res(j)
! For BASIC solutions TIP3 can donate
    if (LBASIC.and.(xres.eq.'TIP3')) goto 301
    if ((Findtitr(j).eq.0).and.(FindtitrH3O(j).eq.0).and.(FindtitrOH(j).eq.0)) cycle
! Go backwards, so that in H3O H3 will be tried first.
! But in Arg you want to go forward
! BUT: the order of attempts introduces bias. For example, in His
! it will affect the tautomer ratio. For H3O it will bias toward back transfers
301 if (xres.eq.'ARG') then
      Largswitch= .false.
      i1=ibase(j)+1
      i2=ibase(j+1)
      iinc=1
    elseif ((xres.eq.'H3O').and.(LH3Oback)) then
      i1=ibase(j+1)
      i2=ibase(j)+1
      iinc=-1
    else
      tempx=oldrandom(iseed1)
      if (tempx.lt.0.5) then
      i1=ibase(j)+1
      i2=ibase(j+1)
      iinc=1
      else
      i1=ibase(j+1)
      i2=ibase(j)+1
      iinc=-1
      endif
!    else
!      i1=ibase(j+1)
!      i2=ibase(j)+1
!      iinc=-1
    endif
    inner: do i=i1,i2,iinc
      if (Labileh(i)) then
        ! check if this H is h.bonded to a potential acceptor. Use list of H.Bonds
        lhbonded=.false.
        do k=1,nhb
          if (HBdone(k)) cycle
          if (KHB(k) == i) then
            ! Find its hb partner. Is it a potential acceptor?
            ! Note: there could be more than 1 hb. We are using the first
            iacce=JHB(k)
            if (JUSTHOPPED(findires(iacce)) .eqv. .true.) cycle
            xjres=res(findires(iacce))
            if (((xres.eq.'TIP3').and.((xjres.eq.'TIP3').or.(xjres.eq.'H3O'))) &
              .and. .not. LWDISS) cycle
            if (((xres.eq.'TIP3').and.((xjres.eq.'HOH'))).and. &
              (PstatusOH(FindtitrOH(findires(iacce))).eq.0).and. .not. LWDISS) cycle
            if ((xres.eq.'HOH').and.(PstatusOH(FindtitrOH(findires(i))).eq.0) &
              .and. (xjres.ne.'HOH') .and. .not. LWDISS) cycle
            if ((xres.eq.'HOH').and.(PstatusOH(FindtitrOH(findires(i))).eq.0) &
              .and. ((xjres.eq.'HOH').and.(PstatusOH(FindtitrOH(findires(iacce))).eq.0)) &
              .and. .not. LWDISS) cycle
            if (Pacc(iacce) .eqv. .true.) then
              lhbonded=.true.
              idono=IHB(k)
              HBdone(k)=.true.
              exit
            else
              exit
            endif
          endif
        enddo

        !Do I attempt hop now, or make a list and then select from the list (all or one?)
        !Attempt hop: I:labile H, idono,iacce: p.acceptor. Returns accept (true or false)
        !Now I attempt for every labileH/Pacc pair

        if (lhbonded) then
          ihyd=i
          call hopattempt(ihyd,idono,iacce,nxtistart,accept,beta, &
                       XOLD,YOLD,ZOLD,VX,VY,VZ)
        endif
!       if (accept) exit outer  ! Avoid consecutive transfers
        if (accept) then
           JUSTHOPPED(findires(iacce))=.true.
           JUSTHOPPED(j)=.true.
           exit inner  ! Allow multiple residues hopping but avoid c.t.
        endif
      endif !if Labileh
    enddo inner  ! do i
  enddo outer  ! do j
  IHBFRQ=0
endif
endif  ! If LMOBHY
!#endif /*KEY_MOBHY*/
     !
     !================================================================
     ! Finally take care of the restart files.
     !================================================================
     if(iunwri > 0) then
        doit=done
        if(isvfrq /= 0) then
           if(mod(npriv,isvfrq) == 0) doit=.true.
        endif
#if KEY_TPS==1                          /*tps_restart*/
        if (qtps) then
           call tpsrst(doit,x,y,z,vx,vy,vz,natom,done, &
                isvfrq)
        endif
#endif                                  /*  (tps_restart)*/
        if(doit) then
           seed=iseed
           istpsa=0
#if KEY_PARALLEL==1                     /* parallel_1 */
# if KEY_SPACDEC==1                     /*spacdec*/
           if(.not.qbroadv) then
              call spacbr(vx,natom,icpumap)
              call spacbr(vy,natom,icpumap)
              call spacbr(vz,natom,icpumap)
           endif
           if(.not.qbroadc) then
              call spacbr(xold,natom,icpumap)
              call spacbr(yold,natom,icpumap)
              call spacbr(zold,natom,icpumap)
              call spacbr(x,natom,icpumap)
              call spacbr(y,natom,icpumap)
              call spacbr(z,natom,icpumap)
           endif
# else                                   /* (spacdec)*/
#  if KEY_DOMDEC==1
           if (q_domdec) then
              if (.not.qbroadv) call copy_to_all(vx, vy, vz)
              if (.not.qbroadc) call copy_to_all(xold, yold, zold)
              if (.not.qbroadc) call copy_to_all(x, y, z)
           else
#  endif
              if(.not.qbroadv) call vdgbr(vx,vy,vz,1)
              if(.not.qbroadc) call vdgbr(xold,yold,zold,1)
#  if KEY_DOMDEC==1
           endif
#  endif
# endif                                  /* (spacdec)*/
           qbroadv=.true.
           qbroadc=.true.
# if KEY_CHEQ==1
           if(qcg) call vdgbr(vcg,cgold,cg,1)
# endif
#endif                                  /* parallel_1 */
     !
#if KEY_STRINGM==1                      /*  VO begin : write to temporary file, then move to .rst */
# if KEY_UNIX==1 || KEY_GNU==1
     ! adapted from FTSM code
           if ( (ftsm_on.or.smcv_on) .and. (iolev > 0  &
#  if KEY_MULTICOM==1
    &                 .or. ME_LOCAL == 0 &
#  endif
    &              ) .and. iunwri > 0 .and. qbakup) then ! VO : qbakup appears declared but not used; I will use it for its intender purpose
!
            inquire(unit=iunwri,  opened=openunit)
            if (openunit) then ! obtain file properties
             inquire(unit=iunwri, name=restart_file_name, access=restart_acc, form=restart_form,&
     &       recl=restart_recl )
             close(iunwri)
            endif
            if(trim(restart_acc) /= 'direct') restart_recl=max_recl ! maximum recl
            open(unit=iunwri, file=trim(restart_file_name)//'.part', form=restart_form, access=restart_acc,&
     &      status='unknown', recl=restart_recl)
           endif
     !  VO call wridyn : will write to new file with '.part' extension :
# endif
#endif                                  /* VO string */
           call wridyn(iunwri,natom,x,y,z,xold,yold,zold,vx,vy,vz, &
#if KEY_CHEQ==1
                cg,cgold,vcg,qcg,                 &
#endif
#if KEY_PIPF==1
                uind,uindo,vuind,qpfdyn,          &
#endif
#if KEY_PIPF==1
                npfbaths,pfnhsbath,pfnhsobath,    &
#endif
#if KEY_DYNVV2==1
                vverl2.and.qholo,xnew,ynew,znew,  &
#endif
                npriv,jhstrt,ndegf,nstep, &
                nsavc,nsavv,seed,avetem,istpsa,ldyna &
#if KEY_BLOCK==1
                ,qlmc,qldm, nblock, bixlam,bldold,bivlam,nsavl &  /*ldm*/
#endif
#if KEY_FOURD==1
                ,vfd,fdold                        &
#endif
#if KEY_SCCDFTB==1
                ,qlamda,qpkac,icntdyn,iavti,dvdl,dvdlav,dtmp1 &
#endif
#if KEY_DHDGB==1
                ,QFHDGB,TOTALS,SDEF,SDEFOLD,VS_DHDGB &
#endif
                )

#if KEY_STRINGM==1 || KEY_QCHEM==1 /*  VO stringm : when using many replicas with intensive i/o rst  */
           IF(QBAKUP) THEN
             INQUIRE(UNIT=iunwri, NAME=restart_file_name, ACCESS=restart_acc, FORM=restart_form,&
     &       RECL=restart_recl )
             close(iunwri)
             write(md_step_count,'(i100)') MDSTEP-1
             md_step_count=ADJUSTL(md_step_count)
             if(trim(restart_acc).ne.'DIRECT') restart_recl=max_recl !  maximum recl
!            write(*,*)'name = ',trim(restart_file_name)//'.'//trim(md_step_count),' test'
             call system('cp '//trim(restart_file_name)//' '//trim(restart_file_name)//'.'//trim(md_step_count))
!            !open(UNIT=iunwri, FILE=trim(restart_file_name)//'.'//md_step_count, FORM=restart_form, ACCESS=restart_acc,&
             open(UNIT=iunwri, FILE=trim(restart_file_name), FORM=restart_form, ACCESS=restart_acc,&
     &       STATUS='UNKNOWN', RECL=restart_recl)
           ENDIF
#endif /* KEY_STRINGM || KEY_QCHEM */

#if KEY_STRINGM==1                      /*  VO : copy .part file to restart file */
# if KEY_UNIX==1 || KEY_GNU==1
           if ( (ftsm_on.or.smcv_on) .and. (iolev  >  0  &
#  if KEY_MULTICOM==1
    &                 .or. ME_LOCAL == 0 &
#  endif
    &              ) .and. iunwri > 0 .and. qbakup) then
            INQUIRE(UNIT=iunwri, OPENED=openunit)
            if (openunit) close(iunwri) ! no need to reopen b/c io params saved;
            syscmd='mv -f '//trim(restart_file_name)//'.part '//trim(restart_file_name)//' >/dev/null &'//char(0) ! include c string terminator
            i=min(len(syscmd), len_trim(syscmd)+1)
            stat=fsystem(syscmd,i) ! cstuff.c function
!            stat=0
!            call system(syscmd,stat) ! system function; may not be portable between compilers/environments
            if (stat  <  0) then
             if (prnlev >= 3) then
              call wrndie(0,'<DCNTRL>', 'FSYSTEM returned '//itoa(stat))
             endif
            endif
     ! if replica exchange is being used with string, need to open file because restart file names need to be exchanged
     ! with successful replica exchange attempts ; CHARMM does not store restart file names, so we cheat by getting them from
     ! the OS using inquire
            if (repl_x_on.and.repl_x_freq > 0) &
     &      open(UNIT=iunwri, FILE=trim(restart_file_name), FORM=restart_form, ACCESS=restart_acc, &
     &      STATUS='UNKNOWN', RECL=restart_recl)
           endif
# endif
#endif                                  /* VO stringm */
     !
           ! BIOVIA Code Start : Bug fix
           if(done) call vclose(iunwri,'KEEP',err)
           ! BIOVIA Code End

           if(phval >= -9000.0 .and. iunphw .gt. 0) then
              if(prnlev >= 5) write(outu,'(a)') 'DCNTRL> Writing restart file for constant pH.'
              call phwrirstrt(iunphw)
           endif
        endif
     endif

     !=======================================================================
#if KEY_QUANTA==1
     if(terminatedyn) then
        call free_dcntrl_space
        return
     endif
#endif

     istart=istart+ncycle
#if KEY_PERT==1
     if(qpert) then
        if(ptermn) then
           ! terminate dynamics if finished with pert in the auto mode.
           if(nstep > istart-1) nstep=istart-1
        endif
     endif
#endif

     !
     !
  enddo dynloop
  !
  ! End of main loop for dynamics.
  !
  !=======================================================================

#if KEY_DYNVV2==1
  IF (VVERL2) THEN
     call vv2_dealloc()
  ENDIF
#endif

#if KEY_SCCDFTB==1
#if KEY_PARALLEL==1
  IF (MYNOD == 0) THEN
#endif

     if(qsccb.and.(qlamda.or.qpkac)) then
        if(nsccl <= 0) then
           goto 765
        endif
        dvdlav=dvdl/nsccl
        dvdlav=dvdl/icntdyn
     endif

765  continue
#if KEY_PARALLEL==1
  ENDIF
#endif

#endif

#if KEY_TPS==1 /*tps_accept*/
  IF (QTPS) THEN
     !       ARD for TPS, May 2003
     CALL TPSACC(iseed,NATOM,X,Y,Z,VX,VY,VZ,NSTEP,IREST,TIMEST)
     !       Subtract 1 from TPSSTP because it was increased at the end of TPSACC

     CALL TPSWRI(TPSSTP-1, &
          NATOM,FREEAT,NFREAT,NDEGF, &
          DELTA,TITLEA,NTITLA,IUNCRD,IUNVEL, &
          X,Y,Z)
     IF (TPSSTP  <=  NTPATH) GOTO 819
     !       If we're here, the TPS loop is done.
     !       Jump back to write last restart file.
     QTPSRS = .TRUE.
     !MH05: This is not very nice Fortran to jump in the middle of the DO
     !      loop from outside of the loop. I placed the calls here from the
     !      main dynamics loop. WARNING: Not very much tested!!! But it
     !      compiles with the new compiler
     !
     !C        GOTO 820
     CALL TPSRST(DOIT,X,Y,Z,VX,VY,VZ,NATOM,DONE, &
          ISVFRQ)
     IF(DOIT)THEN
        SEED=ISEED
        ISTPSA=0
        IF(QBAKUP) CALL FILBAK(IUNWRI,PBMLEV,ERR)
        CALL WRIDYN(IUNWRI,NATOM,X,Y,Z,XOLD,YOLD,ZOLD,VX,VY,VZ, &
#if KEY_CHEQ==1
             CG,CGOLD,VCG,QCG,                 &
#endif
                                ! PJ 06/2005
#if KEY_PIPF==1
             UIND,UINDO,VUIND,QPFDYN,          &
#endif
#if KEY_PIPF==1
             NPFBATHS,PFNHSBATH,PFNHSOBATH,    &
#endif
#if KEY_DYNVV2==1
             VVERL2.AND.QHOLO,XNEW,YNEW,ZNEW,  &
#endif
             NPRIV,JHSTRT,NDEGF,NSTEP, &
             NSAVC,NSAVV,SEED,AVETEM,ISTPSA,LDYNA &
#if KEY_BLOCK==1
             ,QLMC,QLDM, NBLOCK, BIXLAM,BLDOLD,BIVLAM,NSAVL &  /*ldm*/
#endif
#if KEY_FOURD==1
             ,VFD,FDOLD                        &
#endif
#if KEY_SCCDFTB==1
             ,qlamda,qpkac,icntdyn,iavti,dvdl,dvdlav,dtmp1 &
#endif
             )
        IF(DONE) CALL VCLOSE(IUNWRI,'KEEP',ERR)
     ENDIF

     !       Free the storage arrays.
821  CALL TPFREE(NATOM, &
          NSTEP,IUNCRD,IUNVEL)
     !       Restore some variable values just in case.
     NSTEP = NSTSAV
     NSAVC = NPSAVC
     NSAVV = NPSAVV
  ENDIF
#endif /*  (tps_accept)*/
  ! End of main loop for tps
  !=======================================================================
  !
#if KEY_TSM==1
  ! tsm mod shf/djt 4/4/90
  IF (SLOWST) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,G17.5,A,A,F8.2,A)') &
          'SLOW GROWTH DELTA A ',ASLOW,' KCAL/MOL', &
          ' TEMPERATURE: ',SLTEMP,' K'
     IF(PRNLEV >= 2) WRITE(OUTU,'(A,F15.12)')'FINAL LAMBDA: ',LAMBDA
  ENDIF
#endif /*  TSM*/
  call clean_up_and_exit
  return


  !============================= Contained Subroutines =====================================
contains
  subroutine clean_up_and_exit
#if KEY_SQUANTM==1
    if(ltrbomd) then           ! for tr-bomd
       q_apply_tr_bomd=.false. !
    end if
#endif
  !
  !     end of the ncycle loop, clean up and exit.
  !
  ! process the VK array to yeild average atomic temperatures in WCOMP
    IF(JHLAST > 0) THEN
       DO I=ATFRST,ATLAST
#if KEY_GRAPE==1
        if(lfmm)then
           if(fmmcpu(i)/=1)cycle
        endif
#endif
#if KEY_SPACDEC==1
          if(icpumap(i) == mynod)then
#endif
             if(imove(i) == 0) then
                idegf=6
                if(qshake) idegf=idgf2(i)
                if(idegf > 0) then
                   vk(i)=two*vk(i)/(kboltz*idegf*jhlast)
                else
                   vk(i)=zero
                endif
             else
                vk(i)=zero
             endif
#if KEY_SPACDEC==1
          endif
#endif
       enddo
#if KEY_PARALLEL==1
# if KEY_SPACDEC==1
       call spacbr(vk,natom,icpumap)
# else /**/
#  if KEY_DOMDEC==1
       if (q_domdec) then
          call copy_to_all(vk)
       else
#  endif
          call vdgbre(vk,iparpt)
#  if KEY_DOMDEC==1
       endif
#  endif
# endif
#endif
    endif
  !
#if KEY_PERT==1
  ! do final free energy analysis and print result.
# if KEY_BLOCK==1
    if(.not.QLDM)then              /*ldm  Cc New PBLOCK  Bypass when PBLOCK*/
# endif
       if(qpert) call pertan(.true.)
# if KEY_BLOCK==1
    endif/*ldm  Cc New PBLOCK*/
# endif
#endif
  !
#if KEY_ENSEMBLE==1
    if (jrex) call enstat
#endif

    if(qcnstp) call xtlmsr(xucell)
    ! Igor Vorobyov update ?XTLA ?XTLB ?XTLC and so on for vv2 algorithm
    if (qnose .and. qpcon) call xtlmsr(xucell)

#if KEY_PARALLEL==1
# if KEY_SPACDEC==1
    if(.not.qbroadc)then
       call spacbr(xold,natom,icpumap)
       call spacbr(yold,natom,icpumap)
       call spacbr(zold,natom,icpumap)
    endif
# else /**/
#  if KEY_DOMDEC==1
    if (q_domdec) then
       if (.not.qbroadc) call copy_to_all(xold, yold, zold)
    else
#  endif
       if(.not.qbroadc) call vdgbr(xold,yold,zold,1)
#  if KEY_DOMDEC==1
    endif
#  endif
# endif
    qbroadc=.true.
#endif
  ! copy final coordinates into XYZ (only for old integrator or nose)
    if(orig) then
       x(1:natom)=xold(1:natom)
       y(1:natom)=yold(1:natom)
       z(1:natom)=zold(1:natom)
    endif

#if KEY_DHDGB==1
!AP/MF
  IF(ORIG) THEN
     IF (QFHDGB) THEN
         DO I=1,TOTALS
            SDEF(I)=SDEFOLD(I)
         ENDDO
     ENDIF
  ENDIF
#endif
  !
#if KEY_FOURD==1
  ! Save 4D velocity in FDCOMP array
    if(dim4) fdcomp(1:natom)=vfd(1:natom)
#endif
  !
  ! compute and print center of mass velocity...
#if KEY_PARALLEL==1
# if KEY_SPACDEC==1
    if(.not.qbroadv) then
       call spacbr(vx,natom,icpumap)
       call spacbr(vy,natom,icpumap)
       call spacbr(vz,natom,icpumap)
    endif
# else /**/
#  if KEY_DOMDEC==1
    if (q_domdec) then
       if (.not.qbroadv) call copy_to_all(vx, vy, vz)
    else
#  endif
       if(.not.qbroadv) call vdgbr(vx,vy,vz,1)
#  if KEY_DOMDEC==1
    endif
#  endif
# endif
    qbroadv=.true.
#endif
    call cenmss(x,y,z,vx,vy,vz,amass,xcm,ycm,zcm, &
         vxcm,vycm,vzcm,axcm,aycm,azcm,natom,imove,qalwrt &
#if KEY_TSM==1
         ,backls &
#endif
         )

  ! Close and reopen the output file so that the output will be saved
  ! if the system crashes.
  !
    call saveit(outu)

#if KEY_TSM==1
    if(qtsm.and.pigset) then
       call pigcvset(xold,yold,zold)
       call pigmrset(amass)
    endif
#endif /*  TSM*/

    call free_dcntrl_space
#if KEY_DOMDEC==1
    if (q_domdec) then
       if (q_split) call send_stop_recip()
       call stop_split_direct_recip()
    endif
#endif
  end subroutine clean_up_and_exit

  subroutine free_dcntrl_space

#if KEY_FOURD==1
    IF(DIM4) THEN
       ! Allocate space for temporary arrays
       call chmdealloc('dcntrl.src','DCNTRL','FDNEW',NATOM,crl=FDNEW)
       call chmdealloc('dcntrl.src','DCNTRL','FDOLD',NATOM,crl=FDOLD)
       call chmdealloc('dcntrl.src','DCNTRL','FDAVE',NATOM,crl=FDAVE)
       call chmdealloc('dcntrl.src','DCNTRL','VFD',NATOM,crl=VFD)
    ENDIF
#endif

    IF(QAVER) THEN
       call chmdealloc('dcntrl.src','DCNTRL','IXAVE',NATOM,crl=IXAVE)
       call chmdealloc('dcntrl.src','DCNTRL','IYAVE',NATOM,crl=IYAVE)
       call chmdealloc('dcntrl.src','DCNTRL','IZAVE',NATOM,crl=IZAVE)
#if KEY_CHEQ==1
       IF(QCG) call chmdealloc('dcntrl.src','DCNTRL','ICGAVE',NATOM,crl=ICGAVE)
#endif
    ENDIF
    !=======================================================================
    ! Free space for nose and MTS method by masa
    IF(ORIG) THEN
#if KEY_MTS==1
       NATMX = 1
       IF (QTBMTS) NATMX = NATOM
       call chmdealloc('dcntrl.src','DCNTRL','XMI',NATMX,crl=XMI)
       call chmdealloc('dcntrl.src','DCNTRL','YMI',NATMX,crl=YMI)
       call chmdealloc('dcntrl.src','DCNTRL','ZMI',NATMX,crl=ZMI)
       call chmdealloc('dcntrl.src','DCNTRL','XMM',NATMX,crl=XMM)
       call chmdealloc('dcntrl.src','DCNTRL','YMM',NATMX,crl=YMM)
       call chmdealloc('dcntrl.src','DCNTRL','ZMM',NATMX,crl=ZMM)
#endif
       NATMX=NATOM
       call chmdealloc('dcntrl.src','DCNTRL','XNSE',NATMX,crl=XNSE)
       call chmdealloc('dcntrl.src','DCNTRL','YNSE',NATMX,crl=YNSE)
       call chmdealloc('dcntrl.src','DCNTRL','ZNSE',NATMX,crl=ZNSE)
    ENDIF
    !=======================================================================
    !
#if KEY_SGLD==1
    !WXW Free space for SGLD
    IF(QSGLD.OR.QSGMD) THEN
       CALL SGFREE
    ENDIF
#endif
    !=======================================================================
#if KEY_PARALLEL==1
    plnod0=prnlev
    IF(MYNOD /= 0) plnod0=0
    call gbor(plnod0,1)
#endif
    PRNLEV=PRLOLD

    return
  end subroutine free_dcntrl_space

END SUBROUTINE DCNTRL

#if KEY_BLOCK==1 /*block*/
SUBROUTINE SUM_POTENTIAL(V_TSM)

  use chm_kinds
  use dimens_fcm
  use coordc
  use block_fcm
  implicit none

  integer i
  real(chm_real) V_TSM(*)

  !  DEFINITION
  ! V_TSM(1) VPRTTR: REACTANT PERTURBATION POTENTIAL ENERGY ALL INTERACTIONS
  ! V_TSM(2) VPRTTP: PRODUCT  PERTURBATION POTENTIAL ENERGY ALL INTERACTIONS
  ! V_TSM(3) VPRTNR: REACTANT PERTURBATION POTENTIAL ENERGY VDW + ELEC
  ! V_TSM(4) VPRTNP: PRODUCT  PERTURBATION POTENTIAL ENERGY VDW + ELEC
  ! V_TSM(5) VPRTVR: REACTANT PERTURBATION POTENTIAL ENERGY VDW ONLY
  ! V_TSM(6) VPRTVP: PRODUCT  PERTURBATION POTENTIAL ENERGY VDW ONLY
  ! V_TSM(7) VPRTER: REACTANT PERTURBATION POTENTIAL ENERGY ELECTROSTATIC ONLY
  ! V_TSM(8) VPRTEP: PRODUCT  PERTURBATION POTENTIAL ENERGY ELECTROSTATIC ONLY

  ! initialized
  DO i = 1, 8
     V_TSM(i) = 0.0
  ENDDO
  V_TSM(1) = VBBOND(2) + VBANG(2) &
       + VBIMPR(2) + VBTORS(2) &
       + VBELEC(2) + VBVDW(2) &
#if KEY_CMAP==1
       + VBCMAP(2) &
#endif
       + VBGENB(2)

  V_TSM(2) = VBBOND(3) + VBANG(3) &
       + VBIMPR(3) + VBTORS(3) &
       + VBELEC(3) + VBVDW(3) &
#if KEY_CMAP==1
       + VBCMAP(3) &
#endif
       + VBGENB(3)

  V_TSM(3) = VBELEC(2) + VBVDW(2)
  V_TSM(4) = VBELEC(3) + VBVDW(3)
  V_TSM(5) = VBVDW(2)
  V_TSM(6) = VBVDW(3)
  V_TSM(7) = VBELEC(2)
  V_TSM(8) = VBELEC(3)

  RETURN
END SUBROUTINE SUM_POTENTIAL
#endif /*    (block)*/

end module dcntrl_mod
