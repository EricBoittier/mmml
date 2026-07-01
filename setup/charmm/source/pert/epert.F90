#if KEY_PERT==1
SUBROUTINE EPERT(X, Y, Z, DX, DY, DZ, QECONT, ECONT, &
     NDD1, DD1, QSECD, ICALL)
  !-----------------------------------------------------------------------
  !       CALCULATES THE ENERGY AND FORCES FOR A STRUCTURE.
  !     The total energy and individual energy contributions are
  !     returned in the ENERGY.FCM common block.
  !
  !      X,Y,Z         - Coordinates
  !      DX,DY,DZ      - Forces returned
  !      NDD1            The dimension of the second derivative matrix.
  !      DD1           - Second derivative arrays
  !      QSECD         - Second derivative flags
  !      ICALL         - ECALLS increment
  !
  !     By Bernard R. Brooks (and others)
  !
  !     Jay L. Banks added the MMFF interface on 19 October 1995
  !
  !     Stefan Boresch, Martin Leitgeb May 2003:
  !     added (optional) lambda-dependent MMFP-term
  !
  use ewald,only:lewald,ewvirial,kappa,kspace,ewldex
#if KEY_MSCALE==1
  use mscalemod, only: qmscale,emscale                          
#endif
#if KEY_RMD==1
  use cross, only: ecross,NCRUN                                 
#endif
#if KEY_MRMD==1
  use mrmd_fcm,only: emrmd,mrmd_active
#endif
#if KEY_FLUCQ==1
  use flucqm,only:fqcfor                                        
#endif
  use memory
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use bases_fcm
  use cnst_fcm
#if KEY_HMCOM==1
  use cstran_mod,only:nhmcm,inhmcm,phmcm,nhmcmr,khmcmr,hmcmr,ihmcm1, & 
#endif
#if KEY_HMCOM==1
       qhmcm,hmcmx,hmcmy,hmcmz,khmcm,rhmcm,ihmcm,lhmcmm,ihmcm2         
#endif
  use code

  use dcm_fcm,only:DC_CHLAMBDA,QDCM,ADD_DCM_RECIP_F

  use drude
  use ecntrl
  use eintern
  use eintern_fast
  use enbond_mod
  use energym
  use epert_mod
  use euler
  use eutil
  use exelecm
  use fast
  use fourdm
  use hbondm
  use image
  use image_routines_module, only: transi, transo
  use eimg
  use inbnd
  use mmfp
  use nbips
#if KEY_LOOKUP==1
  use LOOKUP,only:elookup,qlookup,nwwo,iwwo,iwoonbl,jwoonbl,ivunbl,jvunbl,& 
                  iuunbl,juunbl,iwwenr,ewweel,ewwenb,qvu,quu    
#endif
#if KEY_BLOCK==1
  use block_ltm,only:qhybh
#endif 

#if KEY_NBIPS==1
  use aips_module,only: eipssys,aipsfin,aipspbc                  
#endif
  use noem
  use param
  use pathm
  use pbeq   ! NKB, gsbp with pert
  use pbound
#if KEY_POLAR==1
  use polarm, only: qpolar, polar1  
#endif
#if KEY_PRIMSH==1
  use primsh, only: qshel, pshel    
#endif
#if KEY_LJPME==1
  use pmeutil, only: qljpme
#endif
  use psf
  use pull_mod, only: epull
  use quantm
  use sbound
  use shapes
  use stream
  use surface
  use ssbpm, only: ssbp1,qssbp,ipa,rmax
  use timerm
  use tsms_mod
  use tbmts
  use nbthole
  use vector
  use pert
  use pert_mod
#if KEY_FLUCQ==1
  use flucq   
#endif
#if KEY_PARALLEL==1
  use parallel     
#endif
#if KEY_MMFF==1
  use ffieldm
  use mmffm
  use escalar_mm
#endif 
  use chutil,only:atomid
#if KEY_QCHEM==1 || KEY_QTURBO==1
  use gukini_mod,only: gukene  
#endif
  use gamess_fcm,only: qmused_qchem,qmused_turbo
  use machutil,only:die,wrttim
#if KEY_PARALLEL==1
  use machutil,only:eclock     
#endif
  use usermod,only: usere,usracm
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec
#endif 
  use prssre
  ! QC: 11/17 Add DFTB PERT based on CHARMM
  ! though add qmused_sccdftb
#if KEY_SCCDFTB==1
  use blockscc_fcm   !Xiya
  use gamess_fcm, only: QGMREM,QMUSED,QMUSED_SCCDFTB,IGMSEL  !Xiya
  use sccdftb 
  use sccdftbsrc 
#endif
  ! QC: 11/17 Done
  !
  implicit none
  character(len=9) :: file = "epert.src"
  character(len=9) :: routine = "epert"
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)
  LOGICAL QECONT
  real(chm_real) ECONT(*)
  INTEGER NDD1, ICALL
  real(chm_real) DD1(*)
  LOGICAL QSECD
  !
  integer,allocatable,dimension(:) :: IUPT
  integer,allocatable,dimension(:) :: JUPT
  !
  INTEGER I,ISTEP,J,N,II !QC: Add J, N, II 
  INTEGER NDDX
  INTEGER ATFRST,ATLAST,TINDEX,IPAT,ITEMP,PFASTER
  CHARACTER(len=8) SIDI,RIDI,RENI,ACI
  real(chm_real)  SFACT,EVMAX,EVAL
  real(chm_real)  EVDWT,ELECT,LRCS,S,TMPA,TMPB
  LOGICAL QSECDL
  LOGICAL QFARRAY
  real(chm_real) EANIS                            ! (E. Harder)
  !
#if KEY_SCCDFTB==1
  real(chm_real) ECRAP               /*qc_010110*/
  real(chm_real) ESCCTB              /*QC: 11/17*/
#endif
#if KEY_PARALLEL==1
  INTEGER,PARAMETER :: NALLWK=3*LENENT+2*LENENV+15
  real(chm_real) ALLWRK(NALLWK)
  real(chm_real) TIMMER
  INTEGER IPT,MSCINIT
#endif /*  PARALLEL*/
#if KEY_MMFF==1
  INTEGER DERIVS
#endif 
#if KEY_CHEMPERT==1
  real(chm_real) ligeex,ligksu,ligsel,ligqco,liguti
#endif 
#if KEY_LOOKUP==1
  real(chm_real) ENBW,EELW
  logical LOOKUPOLD
#endif

  real(chm_real),allocatable,dimension(:) :: ppcgtmp,cgligtmp
  real(chm_real),dimension(2) :: tmparray
! QC: 11/17 Add DFTB PERT based on Xiya
#if KEY_SCCDFTB==1
  real(chm_real),allocatable,dimension(:) :: cgtmp
#endif
! QC: 11/17 Done
#if KEY_NBIPS==1
  real(chm_real)  EIPSVDW,EIPSELE,ENBAIPS,EELAIPS       
#endif
! DCM Module
  real(chm_real) TETPRT(LENENT)

  !sblrc
#if KEY_LRVDW==1
  real(chm_real) CT3,LRCN,PVLRCM,PVLRCL 
#endif
  !
  IF (TIMER  >  0) CALL WRTTIM('Time into energy:')
  !
  ! . Increment the energy calls counter (used for updating).
  ECALLS = ECALLS + ICALL
  !
  !-------------------------------------------------------------------
  ! Some simple compatibility tests
#if KEY_MTS==1
  IF (QTBMTS) CALL WRNDIE(-3,'<EPERT>', &
       'PERT and MTS are not compatible')
#endif 
  ! -------------------------------------------------------------------
  ! next 4 lines commented out by NKBEXTE to make extended electrostatics
  ! compatible with pert
  !...##IFN NOMISC
  !      IF (QEXTND .AND. QETERM(EXTNDE)) CALL WRNDIE(-3,'<EPERT>',
  !     &           'PERT and Extended Electrostatics are not compatible')
  !...##ENDIF
  !--------------------------------------------------------------------
#if KEY_QUANTUM==1
  IF (NATQM > 0) CALL WRNDIE(-3,'<EPERT>', &
       'PERT and QM/MM are not compatible')
#endif 
#if KEY_SHAPES==1 /*shaptst*/
  IF((NUMSHP > 0) .AND. QETERM(SHAP)) CALL WRNDIE(-3,'<EPERT>', &
       'PERT and Shape restraints are not compatible')
#endif /* (shaptst)*/
#if KEY_REPLICA==1
#if KEY_RPATH==1
  IF(QPATH) CALL WRNDIE(-3,'<EPERT>', &
       'PERT and Replica/Path restraints are not compatible')
#endif 
#endif 
#if KEY_FOURD==1 /*4defour*/
  IF (DIM4) CALL WRNDIE(-3,'<EPERT>', &
       'PERT and Four-dimensional method are not compatible')
#endif /* (4defour)*/
#if KEY_TSM==1
  IF (QTSM) CALL WRNDIE(-3,'<EPERT>', &
       'PERT and TSM are not compatible')
#endif /* TSM*/
#if KEY_NOST2==0
  IF (NST2 > 0) CALL WRNDIE(-3,'<EPERT>', &
       'PERT and ST2 water model are not compatible')
#endif 
#if KEY_MMFF==1
  if(FFIELD == MMFF .and. QSECD) CALL WRNDIE(-3,'<EPERT>', &
       'PERT and MMFF with Hessian are not compatible')
#endif 
  !pssp Trap unappropriate nonbonded options for PSSP
  !     not sure whether this catches all...
  IF(QPSSP) THEN
     IF(QSECD) CALL WRNDIE(-1,'<EPERT>', &
          'No second derivates with PSSP')
     IF(.NOT.LCONS) CALL WRNDIE(-3,'<EPERT>', &
          'Only CDIE with PSSP')
     IF(LVSHFT) CALL WRNDIE(-3,'<EPERT>', &
          'PSSP and VSHFT are not compatible')
     IF(LVFSWT) CALL WRNDIE(-3,'<EPERT>', &
          'PSSP and VFSWIT are not compatible')
     IF(LSHFT.AND.LFSWT) CALL WRNDIE(-3,'<EPERT>', &
          'INAPPROPRIATE NONBONDED OPTION FOR PERT/PSSP')
     IF(LGROUP.AND.LEWALD) CALL WRNDIE(-3,'<EPERT>', &
          'PSSP is not compatible with GROUP/EWALD')
  ENDIF
  !-------------------------------------------------------------------
#if KEY_LOOKUP==1
  IF(QLOOKUP)THEN 
  !.ab.Not compatible. Contact A.Blondel for development.  LNI: REINSTATE CHECK!!!
#if KEY_BLOCK==1
    IF (QHYBH) CALL WRNDIE(-5,'<EPERT>', &
        'HYBH and LOOKUP incompatible, see code.')
#endif 
  !.ab.
  ! Trap inappropriate LOOKUP settings  
    IF(QUU .OR. QVU) CALL WRNDIE(-4,'<EPERT>', &
          'LOOKUP MUST use NOUU and NOUV for use with PERT')
  ENDIF
#endif
  !-------------------------------------------------------------------
  !
#if KEY_PARALLEL==1
  CALL PSYNC()
  TIMMER = ECLOCK()
#endif 
#if KEY_PARALLEL==1 /*paramain*/
  ! Define the atom bounds for this processor.
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=1+IPARPT(MYNOD)
  ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=NATOM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
#else /* (paramain)*/
  ATFRST=1
  ATLAST=NATOM
#endif /* (paramain)*/
  !
  ! . Zero the energy terms calculated in this routine.
  DO I = 1,LENENT
     ETERM(I) = ZERO
     ETPRTM(I)= ZERO
     ETPRTL(I)= ZERO
     ETPRTD(I)= ZERO
  ENDDO
  DO I = 1,LENENV
     EPRESS(I)= ZERO
     EVPRTM(I)= ZERO
     EVPRTL(I)= ZERO
     EVPRTD(I)= ZERO
  ENDDO
  EVDWT=ZERO
  ELECT=ZERO
  !
  ! . Zero out the force arrays
  DO I=1,NATOM
     DX(I)=ZERO
     DY(I)=ZERO
     DZ(I)=ZERO
  ENDDO
  !
  ! . zero out the pssp specific stuff
  ECODLM=ZERO
  ELJDLM=ZERO
  ECODLL=ZERO
  ELJDLL=ZERO
  !
  ! Zero the energy contribution array.
  IF(QECONT) THEN
     DO I=1,NATOM
        ECONT(I)=ZERO
     ENDDO
  ENDIF
  !
  IF(QSECD) THEN
     IF(NDD1 <= 0) CALL DIE
     call chmalloc('epert.src','EPERT','IUPT',NDD1,intg=IUPT)
     call chmalloc('epert.src','EPERT','JUPT',NDD1,intg=JUPT)
     call FILUPT(IUPT,NDD1)
     call FILUPT(JUPT,NDD1)
     NDDX=(NDD1*(NDD1+1))/2
  ENDIF
  !
  ! . Construct coordinates for all image atoms.
  ! . Do it here so user energy routines will work with images.
  !
  IF(NTRANS > 0) THEN
#if KEY_PBOUND==1
     if(.not.qboun) then 
#endif
#if KEY_DOMDEC==1
        if (.not.q_domdec) then  
#endif
           ! . Construct the coordinates.
           CALL TRANSO(X,Y,Z,DX,DY,DZ,.TRUE.,QECONT,ECONT,NATOM,NTRANS, &
                IMTRNS,BIMAG%IMATPT,BIMAG%IMATTR, &
                NOROT,NATIM &
#if KEY_FLUCQ==1
                ,QFLUC,CG,FQCFOR      & 
#endif
                )
#if KEY_DOMDEC==1
        endif  
#endif
        ! . Calculate the volume of the system.
        CALL GETVOL(EPROP(VOLUME))
#if KEY_PBOUND==1
     endif  
#endif
  ENDIF
  !
  ! . Use the FAST option where possible.
  PFASTER=FASTER
  IF (FASTER >= 0) THEN
    ! Force slow routines if RSCA is set for lambda0 perturbed atoms
    IF(ANY(PPRSCLF(1:NATOM)/= ONE))FASTER= -1
    CALL FASTST(X,Y,Z,DX,DY,DZ,QSECD)
  ENDIF
  !
  !=======================================================================
  ! Compute lambda value
  !
  ! Read PERT command if at beginning, or
  ! read next PERT commands if finished with current set.
  IF(IPNCAL >= IPSTP .AND. .NOT.PTERMN) CALL PERTAN(.TRUE.)
  !
  IPNCAL = IPNCAL + ICALL
  ISTEP=IPNCAL
  ! Windowing method or before istart
  LAMDA=LAMDAP
  IF(QPWIND) GOTO 350
  IF(ISTEP <= IPSTRT) GOTO 350
  IF(ISTEP < IPSTP) THEN
     ! Slow growth method in the middle (scale)
     LAMDA=LAMDAI+LAMDEL*(ISTEP-IPSTRT)
  ELSE
     ! Slow growth after istop (do nothing)
     LAMDA=LAMDAF
  ENDIF
350 CONTINUE
  ! limit ranges on the lambda value
  IF(LAMDA > ONE) LAMDA=ONE
  IF(LAMDA < ZERO) LAMDA=ZERO
  LAMDAM=ONE-LAMDA
  !
  !=======================================================================
  ! Compute (lambda=0) (old) energy and force terms
  !
  QSECDL=QSECD.AND.(LAMDAM > ZERO)
  QFARRAY=.FALSE.  ! the fast vectors should have lambda=1 arrays

  PERTLAM=LAMDAM  !Cc New PBLOCK

  !
  !-------------------------------------------------------------------
  !
  ! This is more difficult so we better follow the normal MSCALE
  ! procedure here, by 'skipe all' on main.
  ! Later we may do something more easy on users....
  ! If we do MSCALE then we don't do any energy here
  ! Actually if we are working with the selection we want these
  ! energy terms. At the end we do overwrite of the selected
!!!  for now this is commented out !!
!!!  #if KEY_MSCALE==1
!!!    IF(.NOT.QMSCALE)THEN           
!!!  #endif
     !
     ! . Bond terms.
#if KEY_MMFF==1
     DERIVS=1
     IF (QSECDL) DERIVS=3
#endif 
     IF(NBONDP > 0.AND.QETPRT(BOND)) THEN
        CALL EBONDC(ETPRTM(BOND),PPIB,PPJB, &
             PPICB,NBONDP,CBC,CBB, &
             DX,DY,DZ,X,Y,Z, &
             DD1,IUPT,QSECDL,NATOMT)
        IF (TIMER > 1) CALL WRTTIM('Bond energy times:')
     ENDIF
     !
     !br...060710 anisotropy for drudes by Ed Harder (Sept 2005)
#if KEY_PARALLEL==1
     IF (MYNOD == 0) THEN      
#endif
        CALL EANISOTROPY(EANIS,X,Y,Z,DX,DY,DZ,NATOM)
        ETPRTM(BOND)=ETPRTM(BOND)+EANIS
#if KEY_PARALLEL==1
     ENDIF                     
#endif
     !
     !-------------------------------------------------------------------
     ! . Angle terms.
     IF(NTHETP > 0.AND.QETPRT(ANGLE)) THEN
        CALL EANGLC(ETPRTM(ANGLE),PPIT,PPJT,PPKT, &
             PPICT,NTHETP, &
             DX,DY,DZ,X,Y,Z,DD1,IUPT,QSECDL, &
             IAC,NATOMT,QFARRAY, &
             (/0/),ZERO,ZERO,ZERO)
        IF (TIMER > 1) CALL WRTTIM('Angle energy times:')
     ENDIF
     !
     !-------------------------------------------------------------------
#if KEY_MMFF==1
     IF (FFIELD == MMFF) THEN
        !
        IF(NTHETP > 0.AND.QETPRT(OOPL)) &
                                !     Out-off-plane energy (MMFF)
             CALL EOOPL(ETPRTM(OOPL),PPIT,PPJT,PPKT, &
                  PPLT,PPICOOP,nthetp, &
                  OoplFC, &
                  X,Y,Z,DX,DY,DZ,LTSD,DERIVS, &
                  QECONT,ECONT,0, (/ 0 /))
        !
        IF(NTHETP > 0.AND.QETPRT(STRB)) &
                                !     Strech-Bend coupling energy (MMFF)
             CALL ESTRBND(ETPRTM(STRB),PPIT, &
                  PPJT,PPKT, &
                  PPICT,nthetp,CTB, &
                  PPIB,PPJB, &
                  PPICB,CBB,PPSTRBL, &
                  PPICSTBN,STBNP, &
                  X,Y,Z,DX,DY,DZ,LTSD,DERIVS, &
                  QECONT,ECONT,0, (/ 0 /))
     ELSE
#endif 
        !-------------------------------------------------------------------
        ! . Urey-Bradley terms.
        IF(NTHETP > 0.AND.QETPRT(UREYB)) THEN
           CALL EBONDC(ETPRTM(UREYB),PPIT, &
                PPKT,PPICT,NTHETP, &
                CTUC,CTUB, &
                DX,DY,DZ,X,Y,Z,DD1,IUPT,QSECDL,NATOMT)
           IF (TIMER > 1) CALL WRTTIM('Urey-Bradley energy times:')
        ENDIF
        !-------------------------------------------------------------------
        ! . Improper dihedral terms.
        IF(NIMPHP > 0.AND.QETPRT(IMDIHE)) THEN
           CALL EIMPHIC(ETPRTM(IMDIHE),PPIM,PPJM,PPKM, &
                PPLM,PPICI,NIMPHP, &
                DX,DY,DZ,X,Y,Z,DD1,IUPT,QSECDL, &
                NATOMT,QFARRAY)
           IF (TIMER > 1) CALL WRTTIM('Improper dihedral energy times:')
        ENDIF

#if KEY_CMAP==1
        !-------------------------------------------------------------------
        ! . Dihedral Cross-terms
        IF(NCRTP > 0.AND.QETPRT(CMAP)) THEN
           CALL ECMAPC(ETPRTM(CMAP),PPI1CT,PPJ1CT,PPK1CT, &
                PPL1CT,PPI2CT,PPJ2CT,PPK2CT,PPL2CT,PPICCT,NCRTP, &
                DX,DY,DZ,X,Y,Z,DD1,IUPT,QSECDL,NATOMT,QFARRAY)
           IF (TIMER > 1) &
                CALL WRTTIM('Dihedral cross-term energy times:')
        ENDIF
        !
#endif 

#if KEY_MMFF==1
     ENDIF    !FFIELD == MMFF
#endif /*  MMFF*/
     !
     !-------------------------------------------------------------------
     ! . Proper dihedral terms.
     IF(NPHIP > 0.AND.QETPRT(DIHE)) THEN
        CALL EPHIC(ETPRTM(DIHE),PPIP,PPJP,PPKP,PPLP,PPICP,NPHIP, &
             DX,DY,DZ,X,Y,Z,DD1,IUPT,QSECDL,NATOMT, &
             .FALSE.,QFARRAY, &
             (/0/),(/0/),(/0/),ZERO,ZERO,ZERO,ZERO,ZERO)
        IF (TIMER > 1) CALL WRTTIM('Proper dihedral energy times:')
     ENDIF
     !
     !-------------------------------------------------------------------
     ! QC: 11/17 Add DFTB PERT based on Xiya
#if KEY_SCCDFTB==1
     ! . SCCDFTB/MM terms Xiya
     IF (NSCCTC >0 .AND. QETERM(QMEL) .AND. (.NOT. QMMQM)) THEN
       IF (QSCCPERT) THEN    !restore control parameters for lambda=0 if two QM regions are different
          nel=nela
          do n=1,nndim
             izp2(n)=izp2a(n)
          enddo
          TELEC=TELEC1
          SCFTOL=SCFTOL1
          ICHSCC=ICHSCC1
          nscctc=nscctc1
          do ii=1,nscctc1
             scctyp(ii)=scctyp1(ii)
             scczin(ii)=scczin1(ii)
             sccatm(ii)=sccatm1(ii)
          enddo

          do ii=1,nscctc1
             iqmlst(ii,1)=iqmlst1(ii,1)
          enddo
          igmsel(1:natom)=igmsel1(1:natom)
          do ii=1,natom
          enddo 

          !MG_UW1211: for spin-polarization get the right number of electrons
          lcolspin=lcolspin1
          if (lcolspin) then
             nelup=nelupa
             neldown=neldowna
          endif
          CG(1:natom) = charge1(1:natom)
          PPCG(1:natom) = charge1(1:natom)
       ENDIF

       !bookkeeping of charges (globally stored at ltm/sccdftbsrc_ltm.src)
       if (lldep) then
         do i=1,3*nscctc
           ql(i)=qla(i) 
         enddo
       else
         do i=1,nscctc
           qmat(i)=qmata(i)
         enddo
       endif
       if (lcolspin) then
         do i=1,3*nscctc
           qlup(i)=qlupa(i)
           qldown(i)=qldowna(i)
         enddo
       endif

       IF (.NOT. QSCCPERT) THEN
          !copy ppcg (cg for lambda=0) to cg and keep a copy of cg
          call chmalloc(file,routine,"cgtmp",natom,crl=cgtmp)
          cgtmp(1:natom) = CG(1:natom)
          CG(1:natom) = PPCG(1:natom)
       ENDIF
    
       CALL MKMMLST   !pass current CG of MM atoms to ZPTC, and also take care of charge redistribution if link atom is used

       !Zero out the CG array if Mulliken has been called
       !We need to do this for both state lambda=0 and lambda=1
       IF (LMULIK) THEN
           DO J=1,NSCCRP
              DO I=1,NSCCTC
                 CG(IQMLST(I,J))=ZERO
              ENDDO
           ENDDO
       ENDIF
#if KEY_PBEQ==1
       IF (.NOT. QPBEQ ) THEN
#if KEY_GSBP==1 || KEY_SMBP==1  
           !     if SCC-GSBP is needed, skip the SCC calculations here
           !     and instead do them inside GSBP0 - because we need the GSBP
           !     related quantities to be computed first
           !     JZ_UW12: Added SMBP
           IF (.NOT.(QPBEQ .AND. (QETERM(GSBP) .OR. QETERM(SMBP)) .AND. &
              (QGSBP .OR. QSMBP))) THEN
#endif
#endif
              CALL SCCTBENE(ETPRTM(QMEL),X,Y,Z,DX,DY,DZ,NATOM,.true.)

              !bookkeeping of charges (globally stored at ltm/sccdftbsrc_ltm.src)
              if (lldep) then
                do i=1,3*nscctc
                  qla(i)=ql(i)
                enddo
              else
                do i=1,nscctc
                  qmata(i)=qmat(i)
                enddo
              endif
              if (lcolspin) then
                do i=1,3*nscctc
                  qlupa(i)=qlup(i)
                  qldowna(i)=qldown(i)
                enddo
              endif
#if KEY_PBEQ==1
#if KEY_GSBP==1 || KEY_SMBP==1  
           ENDIF
#endif
       ENDIF
#endif
       IF(.NOT. QSCCPERT) THEN
         ! restore CG
         PPCG(1:natom) = CG(1:natom)
         CG(1:natom) = cgtmp(1:natom)
         call chmdealloc(file,routine,"cgtmp",natom,crl=cgtmp)
       ENDIF
     ENDIF

     !-------------------------------------------------------------------
#endif
     ! QC: 11/17 Done
     !-------------------------------------------------------------------
     ! . Non-bond terms.
     IF (QPSSP) THEN
        !     Since this is the reactant part of the interactions, we
        !     force calling of slow, special purpose energy routines
        TQPSSP=.TRUE.
        LAPSSP=LAMDA
     ENDIF
     IF (NATOM  >  0) THEN
     ! QC: 11/17 Add DFTB PERT based on Xiya
#if KEY_SCCDFTB==1
        IF(QMMQM) THEN              !Xiya
          QGMREM = .FALSE.
          QMUSED = .FALSE.
!         QC: 11/17 - we should manipulate QMUSED_SCCDFTB as well
          QMUSED_SCCDFTB = .FALSE. 
          CALL NBONDS(X,Y,Z,BNBNDR,BIMAG)
        ELSE IF (QSCCPERT) THEN 
          QGMREM = .TRUE.
          QMUSED = .TRUE.
!         QC: 11/17 - we should manipulate QMUSED_SCCDFTB as well
          QMUSED_SCCDFTB = .TRUE.
          CALL NBONDS(X,Y,Z,BNBNDR,BIMAG)
        ENDIF
#endif
     ! QC: 11/17 Done

        if(qdcm)then
         CALL DC_CHLAMBDA(0,NATOMP)
        endif
#if KEY_NBIPS==1 /*nbips*/
        ! See if IPS algorithm is in use. if so, calculate IPS of excluded atoms.
        !
        IF(QIPS) THEN
           !WXW  IPS for excluded pairs
           !  3D periodic IPS calculation
           CALL  EIPSPERT(EIPSVDW,EIPSELE,ATFRST,ATLAST,NATOM,NATC, &
                 QETPRT(VDW),QETPRT(ELEC),LVIPS,LEIPS,LCONS,LDBEXP, &
                 BNBNDR%INB14,BNBNDR%IBLO14, &
                 PPCG,RSCLF,CNBA,CNBB,PPIAC,ITC,MAXCN,EPS,E14FAC, &
                 EPROP(VOLUME),RCGSUM,RCG2SUM,RA2SUM,RC2SUM,RAIJSUM,RCIJSUM,  &
                 DX,DY,DZ,X,Y,Z,QECONT,ECONT,DD1,IUPT,QSECDL,.TRUE.)
        ENDIF
#endif /* (nbips)*/

        CALL ENBOND(ETPRTM(VDW),ETPRTM(ELEC),BNBNDR, &
             1,NATOMP,PPCG,PPRSCLF, &
             NGRPP,PPIGPBS,PPIGPTP, &
             PPIAC,PPIACNB, &
             DX,DY,DZ,X,Y,Z,QECONT,ECONT,QETPRT(EWEXCL), &
             ETPRTM(EWEXCL), &
             DD1,IUPT,QSECDL,QETPRT(VDW),QETPRT(ELEC), &
#if KEY_FLUCQ==1
             QFLUC,FQCFOR,                              & 
#endif
             QETPRT(ST2),NST2P,ETPRTM(ST2),QEXTND &
#if KEY_WCA==1
             ,LSOFTCORE0, SCCUTR0,PPWCA   & 
#endif
#if KEY_LJPME==1
             ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
             )

#if KEY_NBIPS==1 /*nbips*/
     ! See if IPS algorithm is in use. if so, calculate IPS of excluded atoms.
     !
     IF(QIPS) THEN
      ETPRTM(VDW)  =  ETPRTM(VDW) + EIPSVDW
      ETPRTM(ELEC) =  ETPRTM(ELEC) + EIPSELE
     ENDIF  
#endif /* (nbips)*/
        IF (TIMER > 1) CALL WRTTIM('Non-bond energy times:')
     ENDIF
     ! ------------------------------------------------------------------
     ! NKBEXTE, addition of extended electrostatics call for lambda=0
#if KEY_NOMISC==0
     ! . Extended electrostatics terms.
#if KEY_MTS==1
     IF(ENE3) THEN                                          
#endif
        IF(QEXTND .AND. QETPRT(EXTNDE)) THEN
           !         WRITE(OUTU,*) 'exelec being called'
           CALL EXELEC(ETPRTM(EXTNDE), NATOM, X, Y, Z, DX, DY, DZ, &
                QXGRAD, WRNMXD, ATSX, ATSY, ATSZ, &
                ATPOT0,ATFX0 ,ATFY0 , &
                ATFZ0 ,ATGXX0,ATGYY0, &
                ATGZZ0,ATGXY0,ATGYZ0, &
                ATGZX0, OUTU)
           IF(TIMER > 1) CALL WRTTIM('Extended electrostatics times:')
        ENDIF
#if KEY_MTS==1
     ENDIF                                                  
#endif
#endif 
     ! BEGIN Thole (G. Lamoureux)
        IF (QDRUDE .AND. QETPRT(ELEC)) THEN
           !         IF (QDRUDE .AND. QTHOLE .AND. QETPRT(ELEC)) THEN
           !yd...061223 c34a1 fix
           !yd         CALL WRNDIE(-3,'<EPERT>','Call to THOLE_NBX not tested.')
125        FORMAT(' EPERT: Calling routine ',A)
           IF (PRNLEV > 6) WRITE(OUTU,125) 'THOLE_NBX'
           CALL THOLE_NBX(ISDRUDE,PPALPHA,PPTHOLE, &
                PPCG, &
                ETPRTM(ELEC),BNBNDR%INB14,BNBNDR%IBLO14, &
                X,Y,Z,DX,DY,DZ,DD1,IUPT,QSECDL,NATOMP)
           IF (TIMER > 1) CALL WRTTIM('Dipole-dipole energy times:')
            IF (PRNLEV.GT.6) WRITE(OUTU,125) 'THOLE_NB'
            CALL THOLE_NB(ISDRUDE,PPALPHA,PPTHOLE,PPCG,&
                1, PPNBTHOLP, &
                PPNBTHOL1, PPNBTHOL2, PPNBTHOL3, &
                NBTHOLXIJ, &
                ETPRTM(ELEC),X,Y,Z,DX,DY,DZ)
            IF (TIMER.GT.1) CALL WRTTIM('Thole-shielding energy times:')

!     Hyperpolarizability
#if KEY_PARALLEL==1
      IF (MYNOD == 0) THEN      
#endif
            IF(QHYPER) THEN
            CALL EHYPER(NATOM,ISDRUDE,HORDER,KHYPER,RHYPER,ETPRTM(BOND), &
                        X,Y,Z,DX,DY,DZ)
            ENDIF
#if KEY_PARALLEL==1
      ENDIF                     
#endif
        ENDIF
     ! END Thole (G. Lamoureux)
     !
     !-------------------------------------------------------------------
     ! . Image nonbond terms.
     IF(NTRANS > 0) THEN
#if KEY_PBOUND==1 /*pbound*/
        if(.not.qboun) then
#endif /*     (pbound)*/
#if KEY_LOOKUP==1 
!turn off lookup for this call
           LOOKUPOLD=QLOOKUP
           QLOOKUP=.FALSE.
#endif

           if(qdcm)then
            CALL DC_CHLAMBDA(0,NATIM)
           endif

           CALL EIMNBD(ETPRTM(IMVDW),ETPRTM(IMELEC),BNBNDR,BIMAG,BIMAGR, &
                1,NATIM,PPCG, &
                PPRSCLF,NIMGRP,PPIGPBS, &
                PPIGPTP,PPIAC, &
                PPIACNB, &
                DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
                QETPRT(EWEXCL),ETPRTM(EWEXCL),QETPRT(IMVDW), &
                QETPRT(IMELEC), &
#if KEY_FLUCQ==1
                QFLUC,FQCFOR,            & 
#endif
                QETPRT(IMST2),NST2P,ETPRTM(IMST2) &
#if KEY_WCA==1
                ,PPWCA                   & 
#endif
#if KEY_LJPME==1
                ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
                )
           IF (TIMER > 1) CALL WRTTIM('Non-bond energy times:')
#if KEY_LOOKUP==1
           QLOOKUP=LOOKUPOLD
#endif
#if KEY_PBOUND==1 /*pbound*/
        endif
#endif /*     (pbound)*/
     ENDIF
     IF (QPSSP) THEN
        TQPSSP=.FALSE.
        !     ESSNBG and ESSNBA internally always use the E??DLL variable,
        !     where ?? is CO or LJ. This is the reactant part,
        !     so we store it in E??DLM.
        ECODLM=ECODLL
        ELJDLM=ELJDLL
        ECODLL=ZERO
        ELJDLL=ZERO
     ENDIF
     !
     !-----------------------------------------------------------------------
     ! . Dihedral restraints.
     IF((NCSPHP > 0) .AND. QETPRT(CDIHE)) THEN
#if KEY_DOMDEC==1
        IF(q_domdec)  CALL WRNDIE(-5,'<EPERT>','NOT READY FOR DOMDEC')
#endif /**/
        CALL EPHI(ETPRTM(CDIHE),PRICS, &
             PRJCS,PRKCS, &
             PRLCS,PRICCS,NCSPHP, &
             PRCCSC,PRCCSD, &
             PRCCSB,PRCCSCOS, &
             PRCCSSIN,DX,DY,DZ,X,Y,Z, &
             .TRUE.,PRCCSW, &
             QECONT,ECONT,0,(/0/),DD1,IUPT,QSECDL &
             )


        IF(TIMER > 1) CALL WRTTIM('Dihedral constraint energy times:')
     ENDIF
     !
     !-----------------------------------------------------------------------
#if KEY_NOMISC==0 /*miscl0*/
     ! . NOE constraints.
#if KEY_PARALLEL==1
     IF(MYNOD == INODE(8)) THEN
#endif 
        IF((NENUMP > 0) .AND. QETPRT(NOE)) THEN
           CALL NOECNS(ETPRTM(NOE),DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
                NENUMP,NESCAP,PRNEIPT, &
                PRNEJPT,PRNEINM, &
                PRNEJNM,PRNELIS, &
                PRNEEXP,PRNERMN, &
                PRNEKMN,PRNERMX, &
                PRNEKMX,PRNEFMX, &
                PRNETCN,PRNEAVE, &
                PRNEMIN,DD1,IUPT,QSECDL &
                                !JH (soft asymptote)
                ,PRNERSW,PRNESEX &
                ,PRNERAM &
#if KEY_PNOE==1
                ,IsPNOE, C0X, C0Y, C0Z   & 
#endif
#if KEY_PNOE==1
                ,MVPNOE,OC0X,OC0Y,OC0Z   & 
#endif
#if KEY_PNOE==1
                ,TC0X,TC0Y,TC0Z          & 
#endif
#if KEY_PNOE==1
                ,NMPNOE, IMPNOE          & 
#endif
                )
           IF (TIMER > 1) CALL WRTTIM('NOE constraint energy times:')
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif 
#endif /* (miscl0)*/
     !-----------------------------------------------------------------------
     ! . Harmonic restraint terms.
     IF(QCNSRP.AND.QETPRT(CHARM)) THEN
       CALL ECNSTR(ETPRTM(CHARM),QCNSRP,PRREFX, &
             PRREFY,PRREFZ, &
             PRKCNST,NATOMP,PRKCEXP, &
             PRXHSCAL,PRYHSCAL, &
             PRZHSCAL,-1,NUMHSETP, &
             PRTYPEH,PRIHSET, &
             PRQHNORT,PRQHNOTR, &
             X,Y,Z,DX,DY,DZ,QECONT,ECONT, &
             DD1,IUPT,QSECDL &
             ,(/0/),(/0,0/),(/0,0/),(/0,0/) &
             )

        IF(TIMER > 1) CALL WRTTIM('Harmonic constraint energy times:')
     ENDIF

     !-------------------------------------------------------------------

#if KEY_QCHEM==1 || KEY_QTURBO==1
     if(qmused_qchem.or.qmused_turbo) then
     !  Ab initio energy from Q-Chem
     !      IF(QQMPERT) THEN
     QMSTATE=0
     IF (QETPRT(QMEL)) CALL GUKENE(ETPRTM(QMEL),X,Y,Z,DX,DY,DZ, &
          PPCG,PPAMASS, &
          PPIAC,NATOM,NDD1,DD1,QSECD, &
          IUPT,JUPT)

     !      write(*,*)'E(state0)=',ETPRTM(QMEL)
     !      ENDIF
     endif
#endif
     !sblrc Add support for simple LJ long range corrections; modelled after
     !     energy.src
#if KEY_LRVDW==1
     !-------------------------------------------------------------------
     ! . LJ long range correction
     !   LongRange vdw contributions (beyond CUTNB)
     IF (LLRVDW) THEN
#if KEY_PARALLEL==1
        IF(MYNOD == 0) THEN                                 
#endif
           CT3=ONE/(CTOFNB*CTOFNB*CTOFNB)
           LRCN = (LRCBP + LRCAP*CT3*CT3*THIRD) * LRVDW_CONST*CT3
           ETPRTM(ELRC) = LRCN/EPROP(VOLUME)
           LRCN = (LRCBP + TWO*LRCAP*CT3*CT3*THIRD)*TWO * &
                LRVDW_CONST*CT3
           PVLRCM = LRCN/EPROP(VOLUME)
#if KEY_PARALLEL==1
        ENDIF                                               
#endif
     ELSE IF (LLRVDW_MS) THEN                               ! LRC using Shirts's algorithm
#if KEY_PARALLEL==1
       IF(MYNOD == 0) THEN                                 
#endif
          ETPRTM(ELRC) = lrvdw_const_ms_m/EPROP(VOLUME)
          PVLRCM = lrvdw_const_ms_m/EPROP(VOLUME)
#if KEY_PARALLEL==1
       ENDIF                                               
#endif
     ELSE
        ETPRTM(ELRC) = ZERO
        PVLRCM=ZERO
     ENDIF
#endif 
     !
     !-------------------------------------------------------------------
     ! . The total internal virial.
     CALL VIRAL(EPPRTM(VIRI),EVPRTM(VIXX:VIZZ),1,NATOMT,X,Y,Z,DX,DY,DZ)
     !sblrc
#if KEY_LRVDW==1
     !   ADD in long range correction to diagonal virial elements

     IF (LLRVDW .OR. LLRVDW_MS) THEN
        EVPRTM(VIXX) = EVPRTM(VIXX)+ PVLRCM
        EVPRTM(VIYY) = EVPRTM(VIYY)+ PVLRCM
        EVPRTM(VIZZ) = EVPRTM(VIZZ)+ PVLRCM
     ENDIF
#endif 
#if KEY_NBIPS==1 /*nbips_press*/
  IF(QIPS)THEN
     ! Calculate grid based anisotropic IPS interaction
     IF(QAIPS)THEN
     ! wxw: not working yet
         call wrndie(0,'<epert>', &
          ' IPS/DFFT not working yet!  using 3D IPS with: PXYZ.')
       IF(QIPSFIN)THEN
           CALL AIPSFIN(ENBAIPS,EELAIPS,ATFRST,ATLAST,NATOMP, &
                QETERM(VDW),QETERM(ELEC),LVIPS,LEIPS, &
                EPS,PPCG,X,Y,Z,DX,DY,DZ)
        ELSE
           CALL AIPSPBC(ENBAIPS,EELAIPS,ATFRST,ATLAST,NATOMP, &
                LEWALD,QETERM(VDW),QETERM(ELEC),LVIPS,LEIPS, &
                EPS,PPCG,X,Y,Z,DX,DY,DZ)
        ENDIF
        IF(TIMER.GT.1) CALL WRTTIM('AIPS times:')
        ETPRTM(VDW)=ETPRTM(VDW)+ENBAIPS
        ETPRTM(ELEC)=ETPRTM(ELEC)+EELAIPS
     ENDIF
     CALL ADDVEC(EVPRTM(VIXX:VIZZ),PIPSVIR,EVPRTM(VIXX:VIZZ),9)
     EPPRTM(VIRI) = EPPRTM(VIRI) + &
          (PIPSVIR(1)+PIPSVIR(5)+PIPSVIR(9))/THREE

  ENDIF
#endif /*  (nbips_press)*/
     !sb    Note that energy also has an nbips block at this point; i.e.
     !sb    PERT apparently does not fully support IPS at this point!!!
     !
     !-------------------------------------------------------------------
     !  K-space sum forces include both internal and external virial components.
     !  Reciprocal space part of ewald sum calculation
#if KEY_LJPME==1
      if(qljpme) CALL WRNDIE(-5,'<EPERT>', 'LJ-PME has not been implemented for PERT yet')
#endif
     IF(LEWALD) THEN
      IF(QDCM) THEN ! For DCM Module
        ETPRTM(EWKSUM)=ETERM(EWKSUM)
        ETPRTM(EWSELF)=ETERM(EWSELF)
        ETPRTM(EWQCOR)=ETERM(EWQCOR)
        ETPRTM(EWUTIL)=ETERM(EWUTIL)
        ETPRTM(EWEXCL)=ETERM(EWEXCL)
        ETERM(EWEXCL)=0.D0
        CALL ADD_DCM_RECIP_F(EWVIRIAL,X,Y,Z)
        CALL ADDVEC(EVPRTM(VIXX),EWVIRIAL,EVPRTM(VIXX),9)
        EPPRTM(VIRI) = EPPRTM(VIRI) + &
             (EWVIRIAL(1)+EWVIRIAL(5)+EWVIRIAL(9))/THREE
      ELSE
        call chmalloc(file,routine,"ppcgtmp",natom,crl=ppcgtmp)
        ppcgtmp(1:natom) = PPCG(1:natom)
        CALL KSPACE(ETPRTM(EWKSUM),ETPRTM(EWSELF),ETPRTM(EWQCOR), &
             ETPRTM(EWUTIL),QETPRT(EWKSUM),QETPRT(EWSELF), &
             QETPRT(EWQCOR),QETPRT(EWUTIL), &
             X,Y,Z,DX,DY,DZ,NATOM,PPCGtmp,CGTOTP &
#if KEY_LJPME==1
             ,ljc6 &
#endif
#if KEY_FLUCQ==1
             ,QFLUC,FQCFOR                  & 
#endif
             )
        call chmdealloc(file,routine,"ppcgtmp",natom,crl=ppcgtmp)
        ! Correct the pressure due to the k-space sum.
        CALL ADDVEC(EVPRTM(VIXX),EWVIRIAL,EVPRTM(VIXX),9)
        EPPRTM(VIRI) = EPPRTM(VIRI) + &
             (EWVIRIAL(1)+EWVIRIAL(5)+EWVIRIAL(9))/THREE
      ENDIF ! FOR DCM MODULE
     ENDIF
     !
     !-----------------------------------------------------------------------
     ! . Harmonic restraint terms.
     IF(QCNSRP.AND.QETPRT(CHARM)) THEN
        CALL ECNSTR(ETPRTM(CHARM),QCNSRP,PRREFX, &
             PRREFY,PRREFZ, &
             PRKCNST,NATOMP,PRKCEXP, &
             PRXHSCAL,PRYHSCAL, &
             PRZHSCAL,1,NUMHSETP, &
             PRTYPEH,PRIHSET, &
             PRQHNORT,PRQHNOTR, &
             X,Y,Z,DX,DY,DZ,QECONT,ECONT, &
             DD1,IUPT,QSECDL &
             ,(/0/),(/0,0/),(/0,0/),(/0,0/) &
             )

        IF(TIMER > 1) CALL WRTTIM('Harmonic constraint energy times:')
     ENDIF
     !
     !-----------------------------------------------------------------------
     !
     !-----------------------------------------------------------------------
     ! . PCA Harmonic restraint terms.
     IF(QCNSRP.AND.QETPRT(PCHARM).and.numpcap>0) THEN
        CALL ECNSTR(ETPRTM(PCHARM),QCNSRP,PRREFX, &
             PRREFY,PRREFZ, &
             PRKCNST,NATOMP,PRKCEXP, &
             PRXHSCAL,PRYHSCAL, &
             PRZHSCAL,2,NUMHSETP, &
             PRTYPEH,PRIHSET, &
             PRQHNORT,PRQHNOTR, &
             X,Y,Z,DX,DY,DZ,QECONT,ECONT, &
             DD1,IUPT,QSECDL &
             ,numpcap,prpcax,prpcay,prpcaz &
             )

        IF(TIMER > 1) CALL WRTTIM('Harmonic constraint energy times:')
     ENDIF
     !
     !-----------------------------------------------------------------------
#if KEY_NOMISC==0
#if KEY_PBEQ==1
     !     Solvent boundary potential of D.Beglov and B.Roux
     IF (QSSBP .AND. QETPRT(SSBP) ) THEN
        ! SSBP LONG RANGE CORRECTION
        CALL ATOMID(IPA,SIDI,RIDI,RENI,ACI)
        LRCS=0.0
        IPAT=ITC(IAC(IPA))
        S=RMAX
        IF (CTOFNB  <=  RMAX) S=CTOFNB                  !yd061223
        S=1.0/(S*S*S)
        DO I=1, NIPERT
           ITEMP=ITC(IAC(I))
           IF (ITEMP  <=  IPAT) THEN
              TINDEX=IPAT*(IPAT-1)/2 + ITEMP
           ELSE
              TINDEX=ITEMP*(ITEMP-1)/2 +IPAT
           ENDIF
           TMPB=CNBA(TINDEX)*CNBA(TINDEX)*CNBA(TINDEX)
           TMPA=CNBB(TINDEX)*TMPB*S
           LRCS=LRCS+TMPA*(TMPB*S*S/3.0-2.0)
        ENDDO
        EPLRCTMP=LRCS
        CALL SSBP1(NATOM,X,Y,Z,ETPRTM(SSBP), &
             PPCG,DX,DY,DZ)
        IF (TIMER > 1) &
             CALL WRTTIM('Solvent-Boundary-Potential energy times:')
     ENDIF
     ! NKB, gsbp with pert
     !     Generalized Solvent boundary potential (GSBP) by W. Im and B.Roux
     IF (QPBEQ .AND. QETPRT(GSBP) .AND. QGSBP ) THEN
     ! QC: 11/17 Change for DFTB PERT based on Xiya
     !  CALL GSBP0(NATOM,X,Y,Z,PPCG, &
     !       ETPRTM(GSBP),DX,DY,DZ,ecalls,.false.&
!#if KEY_SCCDFTB==1
!           ,.FALSE.,ECRAP &       /*qc_010110*/
!#endif
!           )
#if KEY_SCCDFTB==1
        ! SCCDFTB/GSBP Xiya
        !copy ppcg (cg for lambda=0) to cg and keep a copy of cg
        call chmalloc(file,routine,"cgtmp",natom,crl=cgtmp)
        cgtmp(1:natom) = CG(1:natom)
        CG(1:natom) = PPCG(1:natom)

        write(*,*)"QC> EPERT,checking QMMQM",QMMQM
!       write(*,*)"QC> Checking charge"
!       write(*,'(1x,6F10.6)') (CG(I),I=1,NATOM)
#endif
        IF(QMMQM) THEN
           CALL GSBP0(NATOM,X,Y,Z,CG,ETPRTM(GSBP),DX,DY,DZ, &     !Should be CG instead of PPCG
                ecalls,.false. &
#if KEY_SCCDFTB==1
                ,.FALSE.,ECRAP &   
#endif
                )
        ELSE
           CALL GSBP0(NATOM,X,Y,Z,CG,ETPRTM(GSBP),DX,DY,DZ, &     !Should be CG instead of PPCG
                ecalls,.false. &
#if KEY_SCCDFTB==1
                ,QETERM(QMEL),ETPRTM(QMEL) & 
#endif
                )
        ENDIF
#if KEY_SCCDFTB==1
        IF (NSCCTC >0 .AND. QETERM(QMEL)) THEN
           !bookkeeping of charges (globally stored at ltm/sccdftbsrc_ltm.src)
          if (lldep) then
            do i=1,3*nscctc
              qla(i)=ql(i)
            enddo
          else
            do i=1,nscctc
              qmata(i)=qmat(i)
            enddo
          endif
          if (lcolspin) then
            do i=1,3*nscctc
              qlupa(i)=qlup(i)
              qldowna(i)=qldown(i)
            enddo
          endif
        ENDIF
        ! restore CG
        CG(1:natom) = cgtmp(1:natom)
        call chmdealloc(file,routine,"cgtmp",natom,crl=cgtmp)
#endif /* KEY_SCCDFTB moved here */

     ! QC: 11/17 Done
        IF (TIMER > 1) &
             CALL WRTTIM('GSBP energy times:')
     ENDIF
#endif 
#endif 
     !-----------------------------------------------------------------------
     !ML-----------------------------------------------------------------
     IF(QMMFPE) THEN
        !     added MMFP-TERM (use only if MMFP is specified in PERT command.
        !     Auxillay logical QZEGEO is needed to handle cases where
        !     GEO is turned off (GEO RESET) during an alchemical mutation
        IF(QZEGEO.AND.QETPRT(GEO)) THEN
           CALL GEO2(ETPRTM(GEO),NATOM,X,Y,Z,DX,DY,DZ, &
                PLSGEO, PNGEO,PNTGEO, &
                PIGEO, PJGEO,AMASS, &
                PXRGEO,PYRGEO,PZRGEO,PTRGEO, &
                PXDGEO,PYDGEO,PZDGEO,PDRGEO, &
                PDTGEO,PFCGEO,PP1GEO,PP2GEO, &
                PP3GEO,PIUGEO)
           IF(TIMER > 1) &
                CALL WRTTIM('Mean-Field-Potential energy times:')
        ENDIF
     ENDIF
!!!  for now this is commented out !!
!!!  #if KEY_MSCALE==1
!!!    ENDIF
!!!  #endif

  !-------------------------------------------------------------------
  !
  !
  ! end of lambda=0 energy terms
  !===================================================================
  !
  FASTER=PFASTER
#if KEY_MSCALE==1
  !     MSCINIT=-1 for lambda 0, MSCINIT=-2 for lambda 1
  MSCINIT=-1
  IF(QMSCALE) &
       CALL EMSCALE(MSCINIT,NATOM,LENENT,X,Y,Z,DX,DY,DZ, &
       ETPRTM,EPRESS,EPROP,QSECD,DD1,.TRUE.)
#endif 
  !
  EPPRTM(EPOT)=ZERO
  DO I = 1,LENENT
     EPPRTM(EPOT) = EPPRTM(EPOT) + ETPRTM(I)
  ENDDO
  !
  IF(PRNLEV > 6) THEN
     WRITE (OUTU,'(A,I8,A)') &
          ' PERTURBATION> lambda=0 ', IPNTOT,'  steps:'
     CALL PRINTE(OUTU, EPPRTM, ETPRTM, 'LAM0', 'ENR', .TRUE., &
          IPNTOT, ZERO, ZERO, .TRUE.)
  ENDIF
  !
  ! SAVE (1-LAMBDA) RESULTS
  !
  pertdx(1:natomt) = DX(1:NATOMT)
  pertdy(1:natomt) = DY(1:NATOMT)
  pertdz(1:natomt) = DZ(1:NATOMT)
  DX(1:NATOMT)=zero
  DY(1:NATOMT)=zero
  DZ(1:NATOMT)=zero
  !
  IF(QECONT) THEN
     DO I=1,NATIM
        ECONT(I)=-ECONT(I)
     ENDDO
  ENDIF
  !
  QSECDL=QSECD.AND.(LAMDA > ZERO)
#if KEY_MMFF==1
  DERIVS=1
  IF (QSECDL) DERIVS=3
#endif 
  IF(QSECDL) THEN
     SFACT=LAMDAM/LAMDA
     DO I=1,NDDX
        DD1(I)=DD1(I)*SFACT
     ENDDO
  ENDIF
  !
  !=======================================================================
  ! Compute (lambda=1) (new) energy and force terms
  !
  ! . Use the FAST option where possible.
  PFASTER=FASTER
  IF (FASTER >= 0) THEN
    ! Force slow routines if RSCA is set for lambda0 perturbed atoms
    IF(ANY(RSCLF(1:NATOM)/= ONE))FASTER= -1
    CALL FASTST(X,Y,Z,DX,DY,DZ,QSECD)
  ENDIF
  QFARRAY=.TRUE.

  PERTLAM=LAMDAM  !Cc New PBLOCK

!!!  for now this is commented out !!
!!!  #if KEY_MSCALE==1
!!!    IF(.NOT.QMSCALE)THEN           
!!!  #endif
     !
     !-------------------------------------------------------------------
     ! . Bond terms.
     IF(NBOND > 0.AND.QETERM(BOND)) THEN
        CALL EBONDC(ETPRTL(BOND),IB,JB,ICB,NBOND,CBC,CBB, &
             DX,DY,DZ,X,Y,Z,DD1,IUPT,QSECDL,NATOMT)
        IF (TIMER > 1) CALL WRTTIM('Bond energy times:')
     ENDIF
     !
     !br...060710 anisotropy for drudes by Ed Harder (Sept 2005)
#if KEY_PARALLEL==1
     IF (MYNOD == 0) THEN          
#endif
        CALL EANISOTROPY(EANIS,X,Y,Z,DX,DY,DZ,NATOM)
        ETPRTL(BOND)=ETPRTL(BOND)+EANIS
#if KEY_PARALLEL==1
     ENDIF                         
#endif
     !
     !-------------------------------------------------------------------
     ! . Angle terms.
     IF(NTHETA > 0.AND.QETERM(ANGLE)) THEN
        CALL EANGLC(ETPRTL(ANGLE),IT,JT,KT,ICT,NTHETA,DX,DY,DZ, &
             X,Y,Z,DD1,IUPT,QSECDL,IAC,NATOMT,QFARRAY, &
             (/0/),ZERO,ZERO,ZERO)
        IF (TIMER > 1) CALL WRTTIM('Angle energy times:')
     ENDIF
     !
     !-------------------------------------------------------------------
#if KEY_MMFF==1
     IF (FFIELD == MMFF) THEN
        IF(NTHETA > 0.AND.QETERM(OOPL))       & !  Out-off-plane energy (MMFF)
                                !
             CALL EOOPL(ETPRTL(OOPL),IT,JT,KT,LTHETA,icoop, &
                  ntheta,OoplFC, &
                  X,Y,Z,DX,DY,DZ,LTSD,DERIVS, &
                  QECONT,ECONT,0, (/ 0 /))
        !
        IF(NTHETA > 0.AND.QETERM(STRB))       & !  Strech-Bend coupling energy (MMFF)
                                !
             CALL ESTRBND(ETPRTL(STRB),IT,JT,KT,ICT,ntheta, &
                  CTB, &
                  IB,JB,ICB,CBB,StrbList,ICSTBN,STBNP, &
                  X,Y,Z,DX,DY,DZ,LTSD,DERIVS, &
                  QECONT,ECONT,0, (/ 0 /))
     ELSE
#endif 
        !-------------------------------------------------------------------
        ! . Urey-Bradley terms.
        IF(NTHETA > 0.AND.QETERM(UREYB)) THEN
           CALL EBONDC(ETPRTL(UREYB),IT,KT,ICT,NTHETA,CTUC,CTUB,DX,DY,DZ, &
                X,Y,Z,DD1,IUPT,QSECDL,NATOMT)
           IF (TIMER > 1) CALL WRTTIM('Urey-Bradley energy times:')
        ENDIF
        !
        !-------------------------------------------------------------------
        ! . Improper dihedral terms.
        IF(NIMPHI > 0.AND.QETERM(IMDIHE)) THEN
           CALL EIMPHIC(ETPRTL(IMDIHE),IM,JM,KM,LM,ICI,NIMPHI, &
                DX,DY,DZ,X,Y,Z,DD1,IUPT,QSECDL, &
                NATOMT,QFARRAY)
           IF (TIMER > 1) CALL WRTTIM('Improper dihedral energy times:')
        ENDIF
        !

#if KEY_CMAP==1
        !-------------------------------------------------------------------
        ! . Dihedral Cross-terms
        IF(NCRTERM > 0.AND.QETERM(CMAP)) THEN
           CALL ECMAPC(ETPRTL(CMAP),I1CT,J1CT,K1CT,L1CT, &
                I2CT,J2CT,K2CT,L2CT,ICCT,NCRTERM, &
                DX,DY,DZ,X,Y,Z,DD1,IUPT,QSECDL, &
                NATOMT,QFARRAY)
           IF (TIMER > 1) &
                CALL WRTTIM('Dihedral cross-term energy times:')
        ENDIF
        !
#endif 

#if KEY_MMFF==1
     ENDIF  !(FFIELD == MMFF)
#endif /*  MMFF*/
     !
     !-------------------------------------------------------------------
     ! . Proper dihedral terms.
     IF(NPHI > 0.AND.QETERM(DIHE)) THEN
        CALL EPHIC(ETPRTL(DIHE),IP,JP,KP,LP,ICP,NPHI, &
             DX,DY,DZ,X,Y,Z,DD1,IUPT,QSECDL,NATOMT, &
             .FALSE.,QFARRAY, &
             (/0/),(/0/),(/0/),ZERO,ZERO,ZERO,ZERO,ZERO)
        IF (TIMER > 1) CALL WRTTIM('Proper dihedral energy times:')
     ENDIF
     !
     !-------------------------------------------------------------------
     ! QC: 11/17 Add DFTB PERT based on Xiya
#if KEY_SCCDFTB==1
     ! . SCCDFTB/MM terms Xiya
     IF (NSCCTC >0 .AND. QETERM(QMEL)) THEN
       !if mm to qm perturbation, initialize the mulliken charges
       IF (QMMQM .AND. FIRSTE) THEN
         if (lldep) then
           do i=1,3*nscctc
             qlb(i)=ql(i)
           enddo
         else
           do i=1,nscctc
           qmatb(i)=qmat(i)
           enddo
         endif
         if (lcolspin) then
           do i=1,3*nscctc
             qlupb(i)=qlup(i)
             qldownb(i)=qldown(i)
           enddo
         endif
         FIRSTE = .FALSE.
       ENDIF

       IF (QSCCPERT) THEN    !restore control parameters for lambda=1 if two QM regions are different
          nel=nelb
          do n=1,nndim
             izp2(n)=izp2b(n)
          enddo
          TELEC=TELEC2
          SCFTOL=SCFTOL2
          ICHSCC=ICHSCC2
          nscctc=nscctc2
          do ii=1,nscctc2
             scctyp(ii)=scctyp2(ii)
             scczin(ii)=scczin2(ii)
             sccatm(ii)=sccatm2(ii)
          enddo

           do ii=1,nscctc2
              iqmlst(ii,1)=iqmlst2(ii,1)
           enddo
          igmsel(1:natom)=igmsel2(1:natom)

          !MG_UW1211: for spin-polarization get the right number of electrons
          lcolspin=lcolspin2
          if (lcolspin) then
             nelup=nelupb
             neldown=neldownb
           endif
           CG(1:natom) = charge2(1:natom)
       ENDIF

       !bookkeeping of charges (globally stored at ltm/sccdftbsrc_ltm.src)
       if (lldep) then
         do i=1,3*nscctc
           ql(i)=qlb(i)   
         enddo
       else
         do i=1,nscctc
           qmat(i)=qmatb(i)
         enddo
       endif
       if (lcolspin) then
         do i=1,3*nscctc
           qlup(i)=qlupb(i)
           qldown(i)=qldownb(i)
         enddo
       endif

       CALL MKMMLST   !pass current CG of MM atoms to ZPTC, and also take care of charge redistribution if link atom is used

       !Zero out the CG array if Mulliken has been called
       !We need to do this for both state lambda=0 and lambda=1
       IF (LMULIK) THEN
           DO J=1,NSCCRP
              DO I=1,NSCCTC
                 CG(IQMLST(I,J))=ZERO
              ENDDO
           ENDDO
       ENDIF
#if KEY_PBEQ==1
       IF (.NOT. QPBEQ ) THEN
#if KEY_GSBP==1 || KEY_SMBP==1 
           !     if SCC-GSBP is needed, skip the SCC calculations here
           !     and instead do them inside GSBP0 - because we need the GSBP
           !     related quantities to be computed first
           !     JZ_UW12: Added SMBP
           IF (.NOT.(QPBEQ .AND. (QETERM(GSBP) .OR. QETERM(SMBP)) .AND. &
              (QGSBP .OR. QSMBP))) THEN
#endif
#endif
              CALL SCCTBENE(ETPRTL(QMEL),X,Y,Z,DX,DY,DZ,NATOM,.true.)
  
              !bookkeeping of charges (globally stored at ltm/sccdftbsrc_ltm.src)
              if (lldep) then
                do i=1,3*nscctc
                  qlb(i)=ql(i)
                enddo
              else
                do i=1,nscctc
                  qmatb(i)=qmat(i)
                enddo
              endif
              if (lcolspin) then
                do i=1,3*nscctc
                  qlupb(i)=qlup(i)
                  qldownb(i)=qldown(i)
                enddo
              endif
#if KEY_PBEQ==1
#if KEY_GSBP==1 || KEY_SMBP==1 
           ENDIF
#endif
       ENDIF
#endif
     ENDIF
     !-------------------------------------------------------------------
#endif
     ! QC: 11/17 Done
     !-------------------------------------------------------------------
     IF (QPSSP) THEN
        !     Since this is the product part of the interactions, we
        !     force calling of slow, special purpose energy routines
        TQPSSP=.TRUE.
        LAPSSP=LAMDAM
     ENDIF
     ! . Non-bond terms.

     if(qdcm)then
      CALL DC_CHLAMBDA(1,NATOM)
     endif

     IF (NATOM  >  0) THEN
     ! QC: 11/17 for DFTB PERT based on Xiya
#if KEY_SCCDFTB==1
        IF(QMMQM .OR. QSCCPERT) THEN           !Xiya
          QGMREM = .TRUE.
          QMUSED = .TRUE.
          ! QC: 11/17 We should manipulate QMUSED_SCCDFTB as well
          QMUSED_SCCDFTB = .TRUE. 
          CALL NBONDS(X,Y,Z,BNBNDP,BIMAG)
        ENDIF
#endif
#if KEY_NBIPS==1 /*nbips*/
        ! See if IPS algorithm is in use. if so, calculate IPS of excluded atoms.
        !
        IF(QIPS) THEN
           !WXW  IPS for excluded pairs
           !  3D periodic IPS calculation
           CALL  EIPSPERT(EIPSVDW,EIPSELE,ATFRST,ATLAST,NATOM,NATC, &
                QETERM(VDW),QETERM(ELEC),LVIPS,LEIPS,LCONS,LDBEXP, &
                BNBNDP%INB14,BNBNDP%IBLO14, &
                CG,RSCLF,CNBA,CNBB,IAC,ITC,MAXCN,EPS,E14FAC, &
                EPROP(VOLUME),PCGSUM,PCG2SUM,PA2SUM,PC2SUM,PAIJSUM,PCIJSUM,  &
                DX,DY,DZ,X,Y,Z,QECONT,ECONT,DD1,IUPT,QSECDL,.TRUE.)
        ENDIF
#endif /* (nbips)*/
     ! QC: 11/17 Done
        CALL ENBOND(ETPRTL(VDW),ETPRTL(ELEC),BNBNDP, &
             1,NATOM,CG,RSCLF,NGRP,IGPBS,IGPTYP,IAC,IACNB, &
             DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
             QETERM(EWEXCL),ETPRTL(EWEXCL), &
             DD1,IUPT,QSECDL,QETERM(VDW),QETERM(ELEC), &
#if KEY_FLUCQ==1
             QFLUC,FQCFOR,             & 
#endif
             QETERM(ST2),NST2,ETPRTL(ST2),QEXTND &
#if KEY_WCA==1
             ,LSOFTCORE1, SCCUTR1,WCA   & 
#endif
#if KEY_LJPME==1
             ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
             )
#if KEY_NBIPS==1 /*nbips*/
     ! See if IPS algorithm is in use. if so, calculate IPS of excluded atoms.
     !
     IF(QIPS) THEN
      ETPRTL(VDW)  =  ETPRTL(VDW) + EIPSVDW
      ETPRTL(ELEC) =  ETPRTL(ELEC) + EIPSELE
     ENDIF
#endif /* (nbips)*/
#if KEY_CHEMPERT==1
        !sbcp For EWALD and chem pert we need a special wild hack; a direct call to
        !     the exclusion routine to correct for the EWEXCL of the ligand.
        !     This is only needed in the product part, and we include it as
        !     part of the non-bonded terms for accounting purposes ...
        !     Trying to make sure that this is only called if qchemp=.true. and
        !     the normal calling prerequisites for Ewald exclusion are met, i.e.,
        !     we don't want to call this without Ewald being requested!
        if (qchemp.and.lewald.and.qeterm(elec).and. &
             (nnb14 > 0).and.qeterm(ewexcl)) then
           ligeex=zero
           call chmalloc(file,routine,"cgligtmp",natomt,crl=cgligtmp)
           cgligtmp(1:natomt) = cglig(1:natomt)
           call ewldex(ligeex,natom, &
                bnbnd%iblo14,bnbnd%inb14, &
                nnb14,cgligtmp,ctonnb,ctofnb,eps, &
                dx,dy,dz, &
                x,y,z,.false.,tmparray &
#if KEY_FLUCQ==1
                ,qfluc,fqcfor   & 
#endif
                )
           call chmdealloc(file,routine,"cgligtmp",natomt,crl=cgligtmp)
           etprtl(ewexcl)=etprtl(ewexcl)+ligeex
        endif
#endif 
        IF (TIMER > 1) CALL WRTTIM('Non-bond energy times:')
     ENDIF
     ! -----------------------------------------------------------------
     ! NKBEXTE, addition of extended electrostatics call for lambda=1
#if KEY_NOMISC==0
     ! . Extended electrostatics terms.
#if KEY_MTS==1
     IF(ENE3) THEN                                          
#endif
        IF(QEXTND .AND. QETERM(EXTNDE)) THEN
           !         WRITE(OUTU,*) 'EXELEC being called'
           CALL EXELEC(ETPRTL(EXTNDE), NATOM, X, Y, Z, DX, DY, DZ, &
                QXGRAD, WRNMXD, ATSX, ATSY, ATSZ, &
                atpot,ATFX ,ATFY, &
                ATFZ, ATGXX,ATGYY, &
                ATGZZ,ATGXY,ATGYZ, &
                ATGZX, OUTU)
           IF(TIMER > 1) CALL WRTTIM('Extended electrostatics times:')
        ENDIF
#if KEY_MTS==1
     ENDIF                                                  
#endif
#endif 
     ! BEGIN Thole (G. Lamoureux)
        IF (QDRUDE .AND. QETPRT(ELEC)) THEN
           !         IF (QDRUDE .AND. QTHOLE .AND. QETPRT(ELEC)) THEN
           !yd...061223 c34a1 fix
           !yd         CALL WRNDIE(-3,'<EPERT>','Call to THOLE_NBX not tested.')
           IF (PRNLEV > 6) WRITE(OUTU,125) 'THOLE_NBX'
           CALL THOLE_NBX(ISDRUDE,ALPHADP,THOLEI,CG, &
                ETPRTL(ELEC),BNBNDP%INB14,BNBNDP%IBLO14, &
                X,Y,Z,DX,DY,DZ,DD1,IUPT,QSECDL,NATOM)
           IF (TIMER > 1) CALL WRTTIM('Dipole-dipole energy times:')
            IF (PRNLEV.GT.6) WRITE(OUTU,125) 'THOLE_NB'
            CALL THOLE_NB(ISDRUDE,ALPHADP,THOLEI,CG, &
                1, NBTHOLP, &
                NBTHOL1, NBTHOL2, NBTHOL3, &
                NBTHOLXIJ, &
                ETPRTL(ELEC),X,Y,Z,DX,DY,DZ)
            IF (TIMER.GT.1) CALL WRTTIM('Thole-shielding energy times:')

!     Hyperpolarizability
#if KEY_PARALLEL==1
      IF (MYNOD == 0) THEN      
#endif
            IF(QHYPER) THEN
            CALL EHYPER(NATOM,ISDRUDE,HORDER,KHYPER,RHYPER,ETPRTL(BOND), &
                        X,Y,Z,DX,DY,DZ)
            ENDIF
#if KEY_PARALLEL==1
      ENDIF                     
#endif
        ENDIF
     ! END Thole (G. Lamoureux)
     !
     !-------------------------------------------------------------------
     ! . Image nonbond terms.
     IF(NTRANS > 0) THEN
#if KEY_PBOUND==1 /*pbound*/
        if(.not.qboun) then
#endif /*     (pbound)*/
#if KEY_LOOKUP==1 
!turn off lookup for this call
           LOOKUPOLD=QLOOKUP
           QLOOKUP=.FALSE.
#endif

           if(qdcm)then
            CALL DC_CHLAMBDA(1,NATIM)
           endif

           CALL EIMNBD(ETPRTL(IMVDW),ETPRTL(IMELEC),BNBNDP,BIMAG,BIMAGP, &
                1,NATIM,CG,RSCLF,NIMGRP,IGPBS,IGPTYP,IAC,IACNB, &
                DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
                QETERM(EWEXCL),ETPRTL(EWEXCL), &
                QETERM(IMVDW),QETERM(IMELEC), &
#if KEY_FLUCQ==1
                QFLUC,FQCFOR,         & 
#endif
                QETERM(IMST2),NST2,ETPRTL(IMST2) &
#if KEY_WCA==1
                ,WCA                   & 
#endif
#if KEY_LJPME==1
                ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
                )
           IF (TIMER > 1) CALL WRTTIM('Non-bond energy times:')
#if KEY_LOOKUP==1
           QLOOKUP=LOOKUPOLD
#endif
#if KEY_PBOUND==1 /*pbound*/
        endif
#endif /*     (pbound)*/
     ENDIF
     IF (QPSSP) THEN
        TQPSSP=.FALSE.
        !     The routines ESSNBA and ESSNBG worked with E??DLL, where ??
        !     stands for CO and LJ. This is the correct variable here,
        !     since this is the product contribution.
        !     However, we need to take a sign into account...
        ECODLL=-ECODLL
        ELJDLL=-ELJDLL
     ENDIF
     !
     !-------------------------------------------------------------------
     ! . Dihedral constraints.
     IF((NCSPHI > 0) .AND. QETERM(CDIHE)) THEN
#if KEY_DOMDEC==1
        if(q_domdec) CALL WRNDIE(-5,'<EPERT>',&
             'HARMONIC RESTRAINTS NOT READY FOR DOMDEC')
#endif /**/
        CALL EPHI(ETPRTL(CDIHE),ICS,JCS,KCS,LCS,ICCS,NCSPHI, &
             CCSC,CCSD,CCSB,CCSCOS,CCSSIN,DX,DY,DZ,X,Y,Z, &
             .TRUE.,CCSW,QECONT,ECONT,0,(/0/),DD1,IUPT,QSECDL &
             )
        IF(TIMER > 1) CALL WRTTIM('Dihedral constraint energy times:')
     ENDIF
     !
     !-------------------------------------------------------------------
#if KEY_NOMISC==0
     ! . NOE constraints.
     IF((NOENUM > 0) .AND. QETERM(NOE)) THEN
        CALL NOECNS(ETPRTL(NOE),DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
             NOENUM,NOESCA,NOEIPT,NOEJPT,NOEINM,NOEJNM, &
             NOELIS,NOEEXP,NOERMN,NOEKMN,NOERMX,NOEKMX, &
             NOEFMX,NOETCN,NOEAVE,NOEMIN,DD1,IUPT,QSECDL &
                                !JH (soft asymptote)
             ,NOERSW,NOESEX,NOERAM &
#if KEY_PNOE==1
             ,IsPNOE, C0X, C0Y, C0Z   & 
#endif
#if KEY_PNOE==1
             ,MVPNOE,OC0X,OC0Y,OC0Z   & 
#endif
#if KEY_PNOE==1
             ,TC0X,TC0Y,TC0Z          & 
#endif
#if KEY_PNOE==1
             ,NMPNOE, IMPNOE          & 
#endif
             )
        IF (TIMER > 1) CALL WRTTIM('NOE constraint energy times:')
     ENDIF
#endif 
     !-------------------------------------------------------------------
     ! . Harmonic constraint terms.
     IF(QCNSTR.AND.QETERM(CHARM)) THEN
        CALL ECNSTR(ETPRTL(CHARM),QCNSTR,REFX,REFY,REFZ,KCNSTR,NATOM, &
             KCEXPN,XHSCALE,YHSCALE,ZHSCALE,-1, &
             NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
             X,Y,Z,DX,DY,DZ,QECONT,ECONT, &
             DD1,IUPT,QSECDL &
             ,(/0/),(/0,0/),(/0,0/),(/0,0/) &
             )

        IF(TIMER > 1) CALL WRTTIM('Harmonic constraint energy times:')
     ENDIF

     !-------------------------------------------------------------------

#if KEY_QCHEM==1 || KEY_QTURBO==1
     if(qmused_qchem.or.qmused_turbo) then
     !  Ab initio energy from GAMESS (UK version, US version, Q-chem)
     !      IF(QQMPERT) THEN
     QMSTATE=1
     !      write(*,*)'E(state0)=',ETPRTM(QMEL)
     IF (QETPRT(QMEL)) CALL GUKENE(ETPRTL(QMEL),X,Y,Z,DX,DY,DZ,CG, &
          AMASS,IAC,NATOM,NDD1,DD1,QSECD, &
          IUPT,JUPT)

     !      write(*,*)'E(state1)=',ETPRTL(QMEL)
     !      ENDIF
     endif
#endif
     !
     !sblrc Add support for simple LJ long range corrections; modelled after
     !     energy.src
     !-------------------------------------------------------------------
     ! . LJ long range correction
#if KEY_LRVDW==1
     !   LongRange vdw contributions (beyond CUTNB)
     IF (LLRVDW) THEN
#if KEY_PARALLEL==1
        IF(MYNOD == 0) THEN                                 
#endif
           CT3=ONE/(CTOFNB*CTOFNB*CTOFNB)
           LRCN = (LRCB + LRCA*CT3*CT3*THIRD) *LRVDW_CONST*CT3
           ETPRTL(ELRC) = LRCN/EPROP(VOLUME)
           LRCN = (LRCB + TWO*LRCA*CT3*CT3*THIRD)*TWO *LRVDW_CONST*CT3
           PVLRCL = LRCN/EPROP(VOLUME)
#if KEY_PARALLEL==1
        ENDIF                                               
#endif
     ELSE IF (LLRVDW_MS) THEN                               ! LRC using Shirts's algorithm
#if KEY_PARALLEL==1
       IF(MYNOD == 0) THEN                                 
#endif
          ETPRTL(ELRC) = lrvdw_const_ms_l/EPROP(VOLUME)
          PVLRCL = lrvdw_const_ms_l/EPROP(VOLUME)
#if KEY_PARALLEL==1
       ENDIF                                               
#endif
     ELSE
        ETPRTL(ELRC) = ZERO
        PVLRCL=ZERO
     ENDIF
#endif 
     !
     !-------------------------------------------------------------------
     ! . The total internal virial.
     CALL VIRAL(EPPRTL(VIRI),EVPRTL(VIXX:VIZZ),1,NATOMT,X,Y,Z,DX,DY,DZ)
     !
#if KEY_LRVDW==1
     !   ADD in long range correction to diagonal virial elements

     IF (LLRVDW .OR. LLRVDW_MS) THEN
        EVPRTL(VIXX) = EVPRTL(VIXX)+ PVLRCL
        EVPRTL(VIYY) = EVPRTL(VIYY)+ PVLRCL
        EVPRTL(VIZZ) = EVPRTL(VIZZ)+ PVLRCL
     ENDIF
#endif 
#if KEY_NBIPS==1 /*nbips_press*/
  IF(QIPS)THEN
     ! Calculate grid based anisotropic IPS interaction
     IF(QAIPS)THEN
        IF(QIPSFIN)THEN
           CALL AIPSFIN(ENBAIPS,EELAIPS,1,NATOM,NATOM, &
                QETERM(VDW),QETERM(ELEC),LVIPS,LEIPS, &
                EPS,CG,X,Y,Z,DX,DY,DZ)
        ELSE
           CALL AIPSPBC(ENBAIPS,EELAIPS,ATFRST,ATLAST,NATOM, &
                LEWALD,QETERM(VDW),QETERM(ELEC),LVIPS,LEIPS, &
                EPS,CG,X,Y,Z,DX,DY,DZ)
        ENDIF
        IF(TIMER.GT.1) CALL WRTTIM('AIPS times:')
        ETPRTL(VDW)=ETPRTL(VDW)+ENBAIPS
        ETPRTL(ELEC)=ETPRTL(ELEC)+EELAIPS
     ENDIF
     CALL ADDVEC(EVPRTL(VIXX:VIZZ),PIPSVIR,EVPRTL(VIXX:VIZZ),9)
     EPPRTL(VIRI) = EPPRTL(VIRI) + &
          (PIPSVIR(1)+PIPSVIR(5)+PIPSVIR(9))/THREE
  ENDIF
#endif /*  (nbips_press)*/
     !sb    Note that energy also has an nbips block at this point; i.e.
     !sb    PERT apparently does not fully support IPS at this point!!!
     !
     !-------------------------------------------------------------------
     !  K-space sum forces include both internal and external virial components.
     !  Reciprocal space part of ewald sum calculation
#if KEY_LJPME==1
      if(qljpme) CALL WRNDIE(-5,'<EPERT>', 'LJ-PME has not been implemented for PERT yet')
#endif
     IF(LEWALD.AND.LELEC.AND.QETERM(ELEC)) THEN
      IF(QDCM) THEN ! For DCM Module
        ETPRTL(EWKSUM)=ETERM(EWKSUM)
        ETPRTL(EWSELF)=ETERM(EWSELF)
        ETPRTL(EWQCOR)=ETERM(EWQCOR)
        ETPRTL(EWUTIL)=ETERM(EWUTIL)
        ETPRTL(EWEXCL)=ETERM(EWEXCL)
        ETERM(EWEXCL)=0.D0
        CALL ADD_DCM_RECIP_F(EWVIRIAL,X,Y,Z)
        CALL ADDVEC(EVPRTL(VIXX),EWVIRIAL,EVPRTL(VIXX),9)
        EPPRTL(VIRI) = EPPRTL(VIRI) + &
             (EWVIRIAL(1)+EWVIRIAL(5)+EWVIRIAL(9))/THREE
      ELSE 
        CALL KSPACE(ETPRTL(EWKSUM),ETPRTL(EWSELF),ETPRTL(EWQCOR), &
             ETPRTL(EWUTIL),QETERM(EWKSUM),QETERM(EWSELF), &
             QETERM(EWQCOR),QETERM(EWUTIL), &
             X,Y,Z,DX,DY,DZ,NATOM,CG,CGTOT &
#if KEY_LJPME==1
             ,ljc6 &
#endif
#if KEY_FLUCQ==1
             ,QFLUC,FQCFOR                  & 
#endif
             )
#if KEY_CHEMPERT==1
        !sbcp if chem pert is wanted, we need to add the ksum contribution of
        !     the ligand at this point
        !sb??? does this play well with CPT ???
        if (qchemp) then
           !     aux. k-space energy terms
           ligksu=zero
           ligsel=zero
           ligqco=zero
           liguti=zero
           !     the correction
           call chmalloc(file,routine,"cgligtmp",natomt,crl=cgligtmp)
           cgligtmp(1:natomt) = cglig(1:natomt)
#if KEY_LJPME==1
       if(qljpme) CALL WRNDIE(-5,'<EPERT>', 'LJ-PME has not been implemented for PERT yet')
#endif
           call kspace(ligksu,ligsel,ligqco,liguti, &
                qeterm(ewksum),qeterm(ewself), &
                qeterm(ewqcor),qeterm(ewutil), &
                x,y,z,dx,dy,dz, &
                natom,cgligtmp,cgligt &
#if KEY_LJPME==1
                ,ljc6 &
#endif
#if KEY_FLUCQ==1
                ,qfluc,fqcfor  & 
#endif
                )
           call chmdealloc(file,routine,"cgligtmp",natomt,crl=cgligtmp)
           !     add aux. energy terms to normal k-sum
           etprtl(ewksum) = etprtl(ewksum) + ligksu
           etprtl(ewself) = etprtl(ewself) + ligsel
           etprtl(ewqcor) = etprtl(ewqcor) + ligqco
           etprtl(ewutil) = etprtl(ewutil) + liguti
        endif
#endif 
        ! Correct the pressure due to the k-space sum.
        CALL ADDVEC(EVPRTL(VIXX),EWVIRIAL,EVPRTL(VIXX),9)
        EPPRTL(VIRI) = EPPRTL(VIRI) + &
             (EWVIRIAL(1)+EWVIRIAL(5)+EWVIRIAL(9))/THREE
      ENDIF ! IF NOT QDCM (DCM MODULE)
     ENDIF
     !
     !-------------------------------------------------------------------
     ! . Harmonic constraint terms.
     IF(QCNSTR.AND.QETERM(CHARM)) THEN
        CALL ECNSTR(ETPRTL(CHARM),QCNSTR,REFX,REFY,REFZ,KCNSTR,NATOM, &
             KCEXPN,XHSCALE,YHSCALE,ZHSCALE,1, &
             NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
             X,Y,Z,DX,DY,DZ,QECONT,ECONT, &
             DD1,IUPT,QSECDL &
             ,(/0/),(/0,0/),(/0,0/),(/0,0/) &
             )

        IF(TIMER > 1) CALL WRTTIM('Harmonic constraint energy times:')
     ENDIF
     !
     !-------------------------------------------------------------------
     !
     !-------------------------------------------------------------------
     ! . PCA Harmonic constraint terms.
     IF(QCNSTR.AND.QETERM(PCHARM) .and. numpca>0) THEN
        CALL ECNSTR(ETPRTL(PCHARM),QCNSTR,REFX,REFY,REFZ,KCNSTR,NATOM, &
             KCEXPN,XHSCALE,YHSCALE,ZHSCALE,2, &
             NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
             X,Y,Z,DX,DY,DZ,QECONT,ECONT, &
             DD1,IUPT,QSECDL &
             ,numpca,pcax,pcay,pcaz &
             )

        IF(TIMER > 1) CALL WRTTIM('Harmonic constraint energy times:')
     ENDIF
     !
     !-------------------------------------------------------------------
#if KEY_NOMISC==0
#if KEY_PBEQ==1
     !     Solvent boundary potential of D.Beglov and B.Roux
     IF (QSSBP .AND. QETERM(SSBP) ) THEN
        CALL SSBP1(NATOM,X,Y,Z,ETPRTL(SSBP),CG,DX,DY,DZ)
        IF (TIMER > 1) &
             CALL WRTTIM('Solvent-Boundary-Potential energy times:')
     ENDIF
     ! Generalized Solvent boundary potential of W. Im and B. Roux
     IF (QPBEQ .AND. QETPRT(GSBP) .AND. QGSBP ) THEN
     ! QC: 11/17 for DFTB PERT based on Xiya
!       CALL GSBP0(NATOM,X,Y,Z,CG,ETPRTL(GSBP),DX,DY,DZ, &
!                      ecalls,.false.&
!#if KEY_SCCDFTB==1
!                 ,.FALSE.,ECRAP &       /*qc_010110*/
!#endif
!           )
        ! SCCDFTB/GSBP Xiya
        CALL GSBP0(NATOM,X,Y,Z,CG,ETPRTL(GSBP),DX,DY,DZ, &
             ecalls,.false. &
#if KEY_SCCDFTB==1 
           ,QETERM(QMEL),ETPRTL(QMEL) &
#endif
           )
#if KEY_SCCDFTB==1 
       IF (NSCCTC >0 .AND. QETERM(QMEL)) THEN
          !bookkeeping of charges (globally stored at ltm/sccdftbsrc_ltm.src)
          if (lldep) then
            do i=1,3*nscctc
              qlb(i)=ql(i)
            enddo
          else
            do i=1,nscctc
              qmatb(i)=qmat(i)
            enddo
          endif
          if (lcolspin) then
            do i=1,3*nscctc
              qlupb(i)=qlup(i)
              qldownb(i)=qldown(i)
            enddo
          endif
       ENDIF
#endif
     ! QC: 11/17 Done

        IF (TIMER > 1) &
             CALL WRTTIM('GSBP energy times:')
     ENDIF
#endif 
#endif 
     !-------------------------------------------------------------------
     !ML-----------------------------------------------------------------
     IF(QMMFPE) THEN
        !     added MMFP-TERM (use only if MMFP is specified in PERT command
        IF(QGEO.AND.QETERM(GEO)) THEN
           CALL GEO2(ETPRTL(GEO),NATOM,X,Y,Z,DX,DY,DZ, &
                LSTGEO,NGEO,NTGEO, &
                IGEO,JGEO,AMASS, &
                XRGEO,YRGEO,ZRGEO,TRGEO, &
                XDGEO,YDGEO,ZDGEO,DRGEO, &
                DTGEO,FCGEO,P1GEO,P2GEO, &
                P3GEO,IUGEO)
           IF(TIMER > 1) &
                CALL WRTTIM('Mean-Field-Potential energy times:')
        ENDIF
     ENDIF
!!! for now this is commented out
!!! #if KEY_MSCALE==1
!!!   ENDIF
!!! #endif
  !
  !ML-----------------------------------------------------------------
  !-------------------------------------------------------------------
  ! end of lambda=1 energy terms
  !=======================================================================
  !
  FASTER=PFASTER
   
#if KEY_MSCALE==1
  !     MSCINIT=-1 for lambda 0, MSCINIT=-2 for lambda 1
  MSCINIT=-2
  IF(QMSCALE) &
       CALL EMSCALE(MSCINIT,NATOM,LENENT,X,Y,Z,DX,DY,DZ, &
       ETPRTL,EPRESS,EPROP,QSECD,DD1,.TRUE.)
#endif 
  !
  EPPRTL(EPOT)=ZERO
  DO I = 1,LENENT
     EPPRTL(EPOT) = EPPRTL(EPOT) + ETPRTL(I)
  ENDDO
  !
  IF(QSECDL) THEN
     SFACT=LAMDA
     DO I=1,NDDX
        DD1(I)=DD1(I)*SFACT
     ENDDO
  ENDIF
  !
  IF(PRNLEV > 6) THEN
     WRITE(OUTU,'(A,I8,A)') &
          ' PERTURBATION> lambda=1 ', IPNTOT,'  steps:'
     CALL PRINTE(OUTU, EPPRTL, ETPRTL, 'LAM1', 'ENR', .TRUE., &
          IPNTOT, ZERO, ZERO, .TRUE.)
  ENDIF
  !     print debugging info if wanted
  IF(PRNLEV > 6) THEN
     WRITE(OUTU,'(A,F10.5)') ' PSSP> Lambda= ', LAMDA
     WRITE(OUTU,'(A,F10.5,A,F10.5)') ' PSSP> ELJDLL= ', ELJDLL, &
          ' ECODLL= ', ECODLL
     WRITE(OUTU,'(A,F10.5,A,F10.5)') ' PSSP> ELJDLM= ', ELJDLM, &
          ' ECODLM= ', ECODLM
  ENDIF
  !=======================================================================
  ! Compute ediff and accumulate results
  !
  QACCUM=.FALSE.
  IF(IPNCAL <= IPSTRT) GOTO 380
  IF(IPNCAL > IPSTP) GOTO 380
  IF(ICALL == 0) GOTO 380
  QACCUM=.TRUE.
  !
  !sb   Because of the constraint correction, we cannot accumulate
  !     here.  The following accumulation stuff is removed from
  !     here and put in a subroutine EPSUM at the end of this file.
  !
380 CONTINUE
  !
  !=======================================================================
  ! Compute proper linear combination of e and f
  !
  SFACT=-LAMDAM
  CALL SUBVEC(DX,PERTDX,PERTDX,NATOMT)
  CALL ADDCTV(DX,PERTDX,NATOMT,SFACT)
  CALL SUBVEC(DY,PERTDY,PERTDY,NATOMT)
  CALL ADDCTV(DY,PERTDY,NATOMT,SFACT)
  CALL SUBVEC(DZ,PERTDZ,PERTDZ,NATOMT)
  CALL ADDCTV(DZ,PERTDZ,NATOMT,SFACT)
  !
  ! Do energy averaging
  EPROP(EPOT) = ZERO
  DO I = 1,LENENT
     ETERM(I) = ETPRTL(I)*LAMDA + ETPRTM(I)*LAMDAM
     EPROP(EPOT) = EPROP(EPOT) + ETERM(I)
  ENDDO
  IF(QDCM) THEN
    TETPRT(EWKSUM)=ETERM(EWKSUM) ! DCM WORKAROUND: BACK UP EWALD ENERGY TERMS
    TETPRT(EWSELF)=ETERM(EWSELF)
    TETPRT(EWQCOR)=ETERM(EWQCOR)
    TETPRT(EWUTIL)=ETERM(EWUTIL)
    TETPRT(EWEXCL)=ETERM(EWEXCL)
  ENDIF
  !
  CALL VIRAL(EPPRTD(VIRI),EVPRTD(VIXX:VIZZ),1,NATOMT,X,Y,Z,DX,DY,DZ)
  !
  IF(PRNLEV > 6) THEN
     WRITE (OUTU,'(A,F10.5,A)') &
          ' PERTURBATION> lambda= ', LAMDA,'  ave:'
     CALL PRINTE(OUTU, EPROP, ETERM, 'LAMV', 'ENR', .TRUE., &
          IPNTOT, ZERO, ZERO, .TRUE.)
  ENDIF
  !======================================================================
  ! Compute constant energy and force terms (independent of lambda).
  !
  IF (FASTER >= 0) CALL FASTST(X,Y,Z,DX,DY,DZ,QSECD)
#if KEY_PARALLEL==1
  TMERI(1)=TMERI(1)+ECLOCK()-TIMMER
#endif 
  !-------------------------------------------------------------------
#if KEY_PARALLEL==1
  IF(MYNOD == 0) THEN
#endif 
     IF(QETERM(USER)) THEN
        CALL USERE(ETERM(USER),X,Y,Z,DX,DY,DZ,.FALSE.,[zero],NATOM)
        IF (TIMER > 1) CALL WRTTIM('User energy times:')
     ENDIF

#if KEY_RMD==1
     IF((NCRUN > 0).AND.QETERM(CROS)) THEN
        CALL ECROSS(ETERM(CROS),X,Y,Z,DX,DY,DZ,NATOM)
     ENDIF
#endif 

#if KEY_MRMD==1
     IF(mrmd_active.AND.QETERM(MRMD)) THEN
        CALL EMRMD(ETERM(MRMD),X,Y,Z,DX,DY,DZ,NATOM)
     ENDIF
#endif

#if KEY_PARALLEL==1
  ENDIF
#endif 
  !
  !-------------------------------------------------------------------
  ! . PULLing forces, 23-JUL-96 LN
#if KEY_PARALLEL==1
  IF(MYNOD == INODE(2)) THEN
#endif 
     IF (QETERM(PULL)) THEN
        CALL EPULL(ETERM(PULL),DX,DY,DZ,X,Y,Z)
        IF (TIMER > 1) CALL WRTTIM('PULL energy times:')
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif 
  !
  !-------------------------------------------------------------------
  ! . Non-bond terms.
  !     Environment terms; soft core terms won't be used here.
  !     the following ought to be redundant, but to be on the
  !     safe side...
  IF (QPSSP) TQPSSP=.FALSE.
  !sbcp call as usual
  IF (NATOM  >  0) THEN
#if KEY_CHEMPERT==1
     if (.not.qchemp) then
#endif 

        if(qdcm) then
         CALL DC_CHLAMBDA(1,NATOM)  ! MEANS FINAL STATE MUST HAVE ZEROED DCM CHARGES EVERYWHERE!!
        endif

        CALL ENBOND(EVDWT,ELECT,BNBND, &
             1,NATOM,CG,RSCLF,NGRP,IGPBS,IGPTYP,IAC,IACNB, &
             DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),.FALSE.,ZERO, &
             DD1,IUPT,QSECD,QETERM(VDW),QETERM(ELEC), &
#if KEY_FLUCQ==1
             QFLUC,FQCFOR,   & 
#endif
             QETERM(ST2),NST2,ETERM(ST2),QEXTND &
#if KEY_WCA==1
             ,.FALSE.,ONE,WCA  & 
#endif
#if KEY_LJPME==1
             ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
             )
#if KEY_NBIPS==1 /*nbips*/
        ! See if IPS algorithm is in use. if so, calculate IPS of excluded atoms.
        !
        IF(QIPS) THEN
         !  3D periodic IPS calculation
         CALL  EIPSPERT(EIPSVDW,EIPSELE,ATFRST,ATLAST,NATOM,NATC, &
                QETERM(VDW),QETERM(ELEC),LVIPS,LEIPS,LCONS,LDBEXP, &
                BNBND%INB14,BNBND%IBLO14, &
                CG,RSCLF,CNBA,CNBB,IAC,ITC,MAXCN,EPS,E14FAC, &
                EPROP(VOLUME),CGSUM,CG2SUM,A2SUM,C2SUM,AIJSUM,CIJSUM,  &
                DX,DY,DZ,X,Y,Z,QECONT,ECONT,DD1,IUPT,QSECDL,.FALSE.)
         EVDWT  =  EVDWT + EIPSVDW
         ELECT =  ELECT + EIPSELE
        ENDIF
#endif /* (nbips)*/
#if KEY_CHEMPERT==1

        if(qdcm)then ! DCM MODULE
          ETERM(EWKSUM)=TETPRT(EWKSUM) ! DCM WORKAROUND: BACK UP EWALD ENERGY TERMS
          ETERM(EWSELF)=TETPRT(EWSELF)
          ETERM(EWQCOR)=TETPRT(EWQCOR)
          ETERM(EWUTIL)=TETPRT(EWUTIL)
          ETERM(EWEXCL)=TETPRT(EWEXCL)
        endif
     else
        !sbcp    modified CHEM PERT call; need to use lambda=0 psf

        if(qdcm)then
         CALL DC_CHLAMBDA(0,NATOMP)
        endif

        CALL ENBOND(EVDWT,ELECT,BNBND, &
             1,NATOMP,PPCG,PPRSCLF, &
             NGRPP,PPIGPBS,PPIGPTP, &
             PPIAC,PPIACNB, &
             DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),.FALSE.,ZERO, &
             DD1,IUPT,QSECD,QETERM(VDW),QETERM(ELEC), &
#if KEY_FLUCQ==1
             QFLUC,FQCFOR,   & 
#endif
             QETERM(ST2),NST2,ETERM(ST2),QEXTND &
#if KEY_WCA==1
             ,.FALSE.,ONE,PPWCA  & 
#endif
#if KEY_LJPME==1
             ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
             )
#if KEY_NBIPS==1 /*nbips*/
        ! See if IPS algorithm is in use. if so, calculate IPS of excluded atoms.
        !
        IF(QIPS) THEN
         !  3D periodic IPS calculation
         CALL  EIPSPERT(EIPSVDW,EIPSELE,ATFRST,ATLAST,NATOMP,NATC, &
                QETERM(VDW),QETERM(ELEC),LVIPS,LEIPS,LCONS,LDBEXP, &
                BNBND%INB14,BNBND%IBLO14, &
                PPCG,PPRSCLF,CNBA,CNBB,PPIAC,ITC,MAXCN,EPS,E14FAC, &
                EPROP(VOLUME),CGSUM,CG2SUM,A2SUM,C2SUM,AIJSUM,CIJSUM,  &
                DX,DY,DZ,X,Y,Z,QECONT,ECONT,DD1,IUPT,QSECDL,.FALSE.)
         EVDWT  =  EVDWT + EIPSVDW
         ELECT =  ELECT + EIPSELE
        ENDIF
#endif /* (nbips)*/

        if(qdcm)then ! DCM MODULE
          ETERM(EWKSUM)=TETPRT(EWKSUM) ! DCM WORKAROUND: BACK UP EWALD ENERGY TERMS
          ETERM(EWSELF)=TETPRT(EWSELF)
          ETERM(EWQCOR)=TETPRT(EWQCOR)
          ETERM(EWUTIL)=TETPRT(EWUTIL)
          ETERM(EWEXCL)=TETPRT(EWEXCL)
        endif
     endif
#endif 
     ETERM(VDW)=ETERM(VDW)+EVDWT
     ETERM(ELEC)=ETERM(ELEC)+ELECT
     IF (TIMER > 1) CALL WRTTIM('Non-bond energy times:')
  ENDIF
  !
  !-------------------------------------------------------------------
  ! . Hydrogen-bond terms.
#if KEY_PARALLEL==1
  IF(MYNOD == INODE(6)) THEN
#endif 
     IF(NHB > 0.AND.QETERM(HBOND)) THEN
        CALL EHBND(ETERM(HBOND), ECALLS, X, Y, Z, DX, DY, DZ, &
             .FALSE.,0, DD1, IUPT, QSECD)
        IF (TIMER > 1) CALL WRTTIM('Hydrogen-bond energy times:')
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif 
  !
  !-------------------------------------------------------------------
  ! . Image terms.
  IF(NTRANS > 0) THEN
#if KEY_PBOUND==1 /*pbound*/
     if(.not.qboun) then
#endif /*     (pbound)*/
        CALL EIMAGE(X,Y,Z,DX,DY,DZ,.FALSE.,(/ZERO/),QSECD)
        IF (TIMER > 1) CALL WRTTIM('Image energy times:')
        !
        ! . Image nonbond terms.
#if KEY_CHEMPERT==1
        if (.not.qchemp) then
           !sbcp    make normal call
#endif 

           if(qdcm)then
            CALL DC_CHLAMBDA(1,NATIM)
           endif

           CALL EIMNBD(EVDWT,ELECT,BNBND,BIMAG,BIMAG, &
                1,NATIM,CG,RSCLF,NIMGRP,IGPBS,IGPTYP,IAC,IACNB, &
                DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),.FALSE.,ZERO,QETERM(IMVDW), &
                QETERM(IMELEC), &
#if KEY_FLUCQ==1
                QFLUC,FQCFOR,   & 
#endif
                QETERM(IMST2),NST2,ETERM(IMST2) &
#if KEY_WCA==1
                ,WCA            & 
#endif
#if KEY_LJPME==1
                ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
                )
#if KEY_CHEMPERT==1
           if(qdcm)then ! DCM MODULE
            ETERM(EWKSUM)=TETPRT(EWKSUM) ! RESTORE EWALD ENERGY TERMS
            ETERM(EWSELF)=TETPRT(EWSELF)
            ETERM(EWQCOR)=TETPRT(EWQCOR)
            ETERM(EWUTIL)=TETPRT(EWUTIL)
            ETERM(EWEXCL)=TETPRT(EWEXCL)
            CALL DC_CHLAMBDA(0,NATIM)
           endif
        else
           !sb   modified CHEM PERT call,
           !     need lambda=0 psf for constant term!

           if(qdcm)then
            CALL DC_CHLAMBDA(1,NATIM)
           endif

           CALL EIMNBD(EVDWT,ELECT,BNBND,BIMAG,BIMAG, &
                1,NATIM,PPCG, &
                PPRSCLF,NIMGRP,PPIGPBS, &
                PPIGPTP,PPIAC, &
                PPIACNB, &
                DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),.FALSE.,ZERO,QETERM(IMVDW), &
                QETERM(IMELEC), &
#if KEY_FLUCQ==1
                QFLUC,FQCFOR,   & 
#endif
                QETERM(IMST2),NST2,ETERM(IMST2) &
#if KEY_WCA==1
                ,WCA            & 
#endif
#if KEY_LJPME==1
                ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
                )
           if(qdcm)then ! DCM MODULE
            ETERM(EWKSUM)=TETPRT(EWKSUM) ! RESTORE ENERGY TERMS
            ETERM(EWSELF)=TETPRT(EWSELF)
            ETERM(EWQCOR)=TETPRT(EWQCOR)
            ETERM(EWUTIL)=TETPRT(EWUTIL)
            ETERM(EWEXCL)=TETPRT(EWEXCL)
            CALL DC_CHLAMBDA(0,NATIM)
           endif
        endif
#endif 
        ETERM(IMVDW)=ETERM(IMVDW)+EVDWT
        ETERM(IMELEC)=ETERM(IMELEC)+ELECT
        IF (TIMER > 1) CALL WRTTIM('Non-bond energy times:')
#if KEY_PBOUND==1 /*pbound*/
     endif
#endif /*     (pbound)*/
  ENDIF
!================================LKUP-BEG
#if KEY_LOOKUP==1
  !-----------------------------------------------------------------------
  !LNI -  non-bond lookups
  IF(QLOOKUP)THEN
     CALL ELOOKUP(ENBW,EELW, NWWO,&
          IWWO,IWOONBL,JWOONBL, &
          NATOM,IVUNBL,JVUNBL,IUUNBL,JUUNBL)
     IF(IWWENR.GE.0)THEN
        EWWEEL=EELW
        EWWENB=ENBW
     ELSE
        EELW=EWWEEL
        ENBW=EWWENB
     ENDIF
     ETERM(VDW)=ETERM(VDW)+ENBW
     ETERM(ELEC)=ETERM(ELEC)+EELW
     IF(TIMER.GT.1) CALL WRTTIM('Lookup energy times:')

    ! Turn off energy calculation flag if necessary
    IF(QLOOKUP .AND. IWWENR.EQ.1) IWWENR= -1
  ENDIF
#endif 


!================================LKUP-END

  !
  !-------------------------------------------------------------------
  ! . IC constraints.
  IF(LCIC.AND.QETERM(CINTCR)) THEN
     CALL EICCON(ETERM(CINTCR),DX,DY,DZ,X,Y,Z,.FALSE., (/ ZERO /), &
          CCBIC,CCTIC,CCPIC,CCIIC, &
          DD1,IUPT,QSECD,KBEXPN,LUPPER)
     IF (TIMER > 1) CALL WRTTIM('IC constraint energy times:')
  ENDIF
  !
  !-------------------------------------------------------------------
#if KEY_ASPENER==1
  IF(SOLVENT_ENTERED.AND.QETERM(ASP)) THEN
     CALL ASPENR(ETERM(ASP),X,Y,Z,DX,DY,DZ,.FALSE.,0)
     IF (TIMER > 1) CALL WRTTIM('APS surface energy times:')
  ENDIF
#endif /*  ASPENER*/
  !
  !-------------------------------------------------------------------
  ! . The total internal virial.
  CALL VIRAL(EPROP(VIRI),EPRESS(VIXX:VIZZ),1,NATOMT,X,Y,Z,DX,DY,DZ)
  DO I = 1,LENENV
     EPRESS(I) = EPRESS(I) - EVPRTD(I) + &
          EVPRTL(I)*LAMDA + EVPRTM(I)*LAMDAM
     EVPRTD(I) = EVPRTL(I) - EVPRTM(I)
  ENDDO
  EPROP(VIRI) = EPROP(VIRI) - EPPRTD(VIRI) + &
       EPPRTL(VIRI)*LAMDA + EPPRTM(VIRI)*LAMDAM
  EPPRTD(VIRI) = EPPRTL(VIRI) - EPPRTM(VIRI)
  !sblrc
#if KEY_LRVDW==1
  IF (LLRVDW .OR. LLRVDW_MS) THEN
     PVLRC=LAMDAM*PVLRCM+LAMDA*PVLRCL
  ELSE
     PVLRC=ZERO
  ENDIF
#endif 
  !
  !=======================================================================
  ! . External (or boundary) terms.
  !
  !-------------------------------------------------------------------
#if KEY_TNPACK==1
  ! . Second harmonic restraint term (for implicit Euler integration).
  IF(QEHARM.AND.QETERM(EHARM)) THEN
     CALL ECNSTR(ETERM(EHARM),QEHARM,RXHARM,RYHARM, &
          RZHARM,KEHARM,NATOM, (/ 2 /), &
          (/ ONE /), (/ ONE /), (/ ONE /), 1, 1, (/ 0 /), IHHARM, (/ .false. /), (/ .false. /), &
          X,Y,Z,DX,DY,DZ,.FALSE., (/ ZERO /), DD1, &
          IUPT,QSECD &
          ,(/0/),(/0,0/),(/0,0/),(/0,0/) &
          )

     IF(TIMER > 1) CALL WRTTIM('Harmonic restraint energy times:')
  ENDIF
#endif 
  !
  !-------------------------------------------------------------------
#if KEY_NOMISC==0 /*nomisca*/
  ! . Quartic droplet constraints.
#if KEY_PARALLEL==1
  IF(MYNOD == INODE(12)) THEN
#endif 
     IF(QQCNST.AND.QETERM(CQRT)) THEN
        CALL EQUART(ETERM(CQRT),X,Y,Z,NATOM,LQMASS,AMASS,DX,DY,DZ, &
             .FALSE., (/ ZERO /), KQCNST,KQEXPN,QSECD)
        IF(TIMER > 1) CALL WRTTIM('Droplet constraint energy times:')
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif 
#if KEY_HMCOM==1
  !
  !-------------------------------------------------------------------
  ! . Center of masses harmonic constraint
#if KEY_PARALLEL==1
  IF(MYNOD == INODE(12)) THEN
#endif 
     IF (QHMCM.AND.QETERM(HMCM)) THEN
        CALL ECMCNS(ETERM(HMCM),HMCMX,HMCMY,HMCMZ,KHMCM,RHMCM, &
             NATOM,NHMCM,IHMCM,INHMCM,PHMCM, &
             X,Y,Z,DX,DY,DZ,LHMCMM,AMASS, &
             NHMCMR,KHMCMR,HMCMR,IHMCM1,IHMCM2)
        IF(TIMER > 1) &
             CALL WRTTIM('Center of mass harmonic constraint times:')
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif 

#endif 
  !
  !-------------------------------------------------------------------
  ! . Stochastic boundary terms.
#if KEY_PARALLEL==1
  IF(MYNOD == INODE(15)) THEN
#endif 
     IF(QBOUND.AND.QETERM(SBNDRY)) THEN
        CALL BNDRY(ETERM(SBNDRY),X,Y,Z,DX,DY,DZ,.FALSE.,0, &
             DD1,IUPT,QSECD)
        IF(TIMER > 1) CALL WRTTIM('Solvent boundary energy times:')
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif 
  !
  !-------------------------------------------------------------------
  !ML   added MMFP-TERM at lambda=1, lambda=0 (optional)
  !ML   MMFP as static potential during PERT still default
#if KEY_PARALLEL==1
  IF(MYNOD == INODE(16)) THEN
#endif 
     IF(.NOT.QMMFPE) THEN
        IF(QGEO.AND.QETERM(GEO)) THEN
           CALL GEO2(ETERM(GEO),NATOM,X,Y,Z,DX,DY,DZ, &
                LSTGEO,NGEO,NTGEO, &
                IGEO,JGEO,AMASS, &
                XRGEO,YRGEO,ZRGEO,TRGEO, &
                XDGEO,YDGEO,ZDGEO,DRGEO, &
                DTGEO,FCGEO,P1GEO,P2GEO, &
                P3GEO,IUGEO)
           IF(TIMER > 1) &
                CALL WRTTIM('Mean-Field-Potential energy times:')
        ENDIF
     ENDIF
     !-----------------------------------------------------------------------
#if KEY_PRIMSH==1
     IF (QSHEL.AND. QETERM(SHEL))THEN
        CALL PSHEL(VDWR,IAC,ITC,X,Y,Z, &
             ETERM(SHEL),DX,DY,DZ,NWAT,NPATM,TOTV)
     ENDIF
#endif 
     !
     !-----------------------------------------------------------------------
     IF(QMDIP.AND.QETERM(MDIP)) THEN
        CALL MDIP2(ETERM(MDIP),NATOM,X,Y,Z,DX,DY,DZ,CG, &
             LSTMDIP,NMDIP,NTMDIP, &
             XRMDIP,YRMDIP,ZRMDIP, &
             XDMDIP,YDMDIP,ZDMDIP, &
             FCMDIP,D0MDIP,PWMDIP)
        IF(TIMER > 1) &
             CALL WRTTIM('Mean-Field-Potential energy times:')
     ENDIF
     !
     !-----------------------------------------------------------------------
#if KEY_POLAR==1
     !BR...08-JUL-96, Stillinger-David Polarization Model for Water
     IF (QPOLAR .AND. QETERM(POLAR)) THEN
        CALL POLAR1(ETERM(POLAR),NATOM,X,Y,Z,DX,DY,DZ, &
             NATC,IAC,ITC,CNBA,CNBB,CG)
        IF (TIMER > 1) &
             CALL WRTTIM('Polarization model energy times:')
     ENDIF
#endif 
#if KEY_PARALLEL==1
  ENDIF
#endif 
#endif /* (nomisca)*/
  !-------------------------------------------------------------------
  !
  ! . Backtransform image forces
  IF(NTRANS > 0) THEN
#if KEY_PBOUND==1 /*pbound*/
     if(.not.qboun) then
#endif /*     (pbound)*/
        CALL TRANSI(X,Y,Z,DX,DY,DZ,.FALSE., &
             (/ zero /), &
             NATOM,NTRANS, &
             IMTRNS,BIMAG%IMATPT,BIMAG%IMATTR, &
             NOROT,NATIM,IMINV,IMFORC,IMTORQ &
#if KEY_FLUCQ==1
             ,QFLUC,FQCFOR            & 
#endif
             )
#if KEY_PBOUND==1 /*pbound*/
     endif
#endif /*     (pbound)*/
  ENDIF
  !
  !=======================================================================
  ! . Finish up.
  IF (TIMER  ==  1) CALL WRTTIM('Total energy times:')
  !
  !-------------------------------------------------------------------
#if KEY_PARALLEL==1 /*paramain*/
  ! Process code for parallelization
  TMERI(2) = TMERI(2) + ECLOCK()-TIMMER
  TIMMER = ECLOCK()
  CALL PSYNC()
  TMERI(3) = TMERI(3) + ECLOCK()-TIMMER
  TIMMER = ECLOCK()
  !
  IMERI = IMERI + 1
  !
  ! Combine forces.
  CALL VDGSUM(DX,DY,DZ,0)
  CALL VDGSUM(PERTDX,PERTDY,PERTDZ,0)
  !
  IF(QECONT) THEN
     CALL GCOMB(ECONT,NATIM)
  ENDIF
  !
  TMERI(4)=TMERI(4)+ECLOCK()-TIMMER
  CALL PSYNC()
  TIMMER = ECLOCK()
  !
  ! NOTES:
  !     Distributed vector broadcast of forces done in calling routines.
  !     Combine Hessian explicitly in calling routines if needed.
  !
#endif /* (paramain)*/
  !
  !-------------------------------------------------------------------
  ! . Remove forces on fixed atoms
  DO I=1,NATOM
     IF(IMOVE(I) > 0) THEN
        DX(I)=ZERO
        DY(I)=ZERO
        DZ(I)=ZERO
     ENDIF
  ENDDO
  !-------------------------------------------------------------------
  ! . Final virial computation calculation.
  CALL VIRTOT(EPRESS(VIXX:VIZZ),EPRESS(VEXX:VEZZ),NATOM,X,Y,Z,DX,DY,DZ)
  IF(NTRANS > 0 .AND. XTLTYP /= ' ') &
       CALL LATTFD(XTLTYP,XTLABC,EPRESS(VEXX:VEZZ),DXTL,XTLREF)
  !
  !-------------------------------------------------------------------
#if KEY_PARALLEL==1 /*paraend*/
  ! Process more code for parallelization
  TMERI(2) = TMERI(2) + ECLOCK()-TIMMER
  TIMMER = ECLOCK()
  !
  ! Pack assorted results into a single array for efficiency
  ! in using the global sum operation.
  !
  !      VIRI    - internal virial
  !      VIRE    - external virial
  !      VIRKE   - virial energy ( <f|r> )
  !
  IPT=1
  ALLWRK(IPT)=EPROP(VIRI)
  IPT=IPT+1
  ALLWRK(IPT)=EPROP(VIRE)
  IPT=IPT+1
  ALLWRK(IPT)=EPROP(VIRKE)
  IPT=IPT+1
  ALLWRK(IPT)=EPPRTM(EPOT)
  IPT=IPT+1
  ALLWRK(IPT)=EPPRTL(EPOT)
  DO I=1,LENENT
     IPT=IPT+1
     ALLWRK(IPT)=ETERM(I)
  ENDDO
  DO I=1,LENENT
     IPT=IPT+1
     ALLWRK(IPT)=ETPRTM(I)
  ENDDO
  DO I=1,LENENT
     IPT=IPT+1
     ALLWRK(IPT)=ETPRTL(I)
  ENDDO
  DO I=1,LENENV
     IPT=IPT+1
     ALLWRK(IPT)=EPRESS(I)
  ENDDO
  DO I=1,LENENV
     IPT=IPT+1
     ALLWRK(IPT)=EVPRTD(I)
  ENDDO
  DO I=1,6
     IPT=IPT+1
     ALLWRK(IPT)=DXTL(I)
  ENDDO
  IPT=IPT+1
  ALLWRK(IPT)=ECODLM
  IPT=IPT+1
  ALLWRK(IPT)=ELJDLM
  IPT=IPT+1
  ALLWRK(IPT)=ECODLL
  IPT=IPT+1
  ALLWRK(IPT)=ELJDLL
  !
  CALL GCOMB(ALLWRK,IPT)
  !
  IPT=1
  EPROP(VIRI)=ALLWRK(IPT)
  IPT=IPT+1
  EPROP(VIRE)=ALLWRK(IPT)
  IPT=IPT+1
  EPROP(VIRKE)=ALLWRK(IPT)
  IPT=IPT+1
  EPPRTM(EPOT)=ALLWRK(IPT)
  IPT=IPT+1
  EPPRTL(EPOT)=ALLWRK(IPT)
  DO I=1,LENENT
     IPT=IPT+1
     ETERM(I)=ALLWRK(IPT)
  ENDDO
  DO I=1,LENENT
     IPT=IPT+1
     ETPRTM(I)=ALLWRK(IPT)
  ENDDO
  DO I=1,LENENT
     IPT=IPT+1
     ETPRTL(I)=ALLWRK(IPT)
  ENDDO
  DO I=1,LENENV
     IPT=IPT+1
     EPRESS(I)=ALLWRK(IPT)
  ENDDO
  DO I=1,LENENV
     IPT=IPT+1
     EVPRTD(I)=ALLWRK(IPT)
  ENDDO
  DO I=1,6
     IPT=IPT+1
     DXTL(I)=ALLWRK(IPT)
  ENDDO
  IPT=IPT+1
  ECODLM=ALLWRK(IPT)
  IPT=IPT+1
  ELJDLM=ALLWRK(IPT)
  IPT=IPT+1
  ECODLL=ALLWRK(IPT)
  IPT=IPT+1
  ELJDLL=ALLWRK(IPT)
  !
  ! timing stuff
  TMERI(4)=TMERI(4)+ECLOCK()-TIMMER
#endif /* (paraend)*/
! user stuff was missing
  CALL USRACM(NATOM,X,Y,Z,DX,DY,DZ,.FALSE.,[zero],icall)
!
  !-------------------------------------------------------------------
  IF (TIMER  ==  1) CALL WRTTIM('Total energy times:')
  ! . Add all the terms to get the total energy.
  EPROP(EPOT) = ZERO
  DO I = 1,LENENT
     EPROP(EPOT) = EPROP(EPOT) + ETERM(I)
  ENDDO
  !
  ! . Free stack space for the second derivatives.
  IF(QSECD) then
     call chmdealloc('epert.src','EPERT','IUPT',NDD1,intg=IUPT)
     call chmdealloc('epert.src','EPERT','JUPT',NDD1,intg=JUPT)
  endif
  !
  !-------------------------------------------------------------------
  IF(PRNLEV > 6) THEN
     WRITE(OUTU,'(A,F10.5,A)') &
          ' PERTURBATION> lambda= ', LAMDA,'  final:'
     CALL PRINTE(OUTU, EPROP, ETERM, 'LAMF', 'ENR', .TRUE., &
          IPNTOT, ZERO, ZERO, .TRUE.)
  ENDIF

  RETURN
END SUBROUTINE EPERT

SUBROUTINE EWORKT(VX,VY,VZ,DX,DY,DZ)
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use number
  use bases_fcm
  use pert
  implicit none
  !
  real(chm_real) VX(*),VY(*),VZ(*),DX(*),DY(*),DZ(*)
  !
  !
  CALL EWORKT2(VX,VY,VZ,DX,DY,DZ,PERTDX,PERTDY,PERTDZ)
  !
  RETURN
END SUBROUTINE EWORKT
!
SUBROUTINE EWORKT2(VX,VY,VZ,DX,DY,DZ,DXPERT,DYPERT,DZPERT)
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use number
  use energym
  use epert_mod
  use pert
  use psf
  use reawri
  use parallel
  implicit none
  !
  real(chm_real) VX(*),VY(*),VZ(*),DX(*),DY(*),DZ(*)
  real(chm_real) DXPERT(*),DYPERT(*),DZPERT(*)
  !
  !
  INTEGER I,i0,i1
  real(chm_real) DOT
  !
  ! Do free energy accumulation
  IF(QPERT) THEN
     IF(QACCUM) THEN
        !
        DOT=ZERO
        i0=1
        i1=natom
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
        I0 = 1+IPARPT(MYNOD)
        I1 = IPARPT(MYNODP)
#endif 
#endif 
        do i=i0,i1
#if KEY_PARALLEL==1
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
#endif 
              DOT=DOT+DXPERT(I)*VX(I)+DYPERT(I)*VY(I)+DZPERT(I)*VZ(I)
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
#if KEY_PARALLEL==1
        CALL GCOMB(DOT,1)
#endif 
        EPPRTD(TOTKE)=DOT*DELTA
        !
        EPPRTD(TOTE)=EPPRTD(TOTKE)+EPPRTD(EPOT)
        !
        DO I=1,LENENP
           EPPRTA(I) = EPPRTA(I) + EPPRTD(I)
           EPPRTF(I) = EPPRTF(I) + EPPRTD(I)**2
        ENDDO
        !
     ENDIF
  ENDIF
  !
  RETURN
END SUBROUTINE EWORKT2

SUBROUTINE EPCONS(DELTA2)
  !
  !     Stefan Boresch, April 1995
  !     calculate the constraint correction to a free energy
  !     Lagrangian multipliers from last SHAKE iteration have to
  !     be in PSHAUX(1,LL)!
  !
  use chm_kinds
  use dimens_fcm
  use bases_fcm
  use pert
  !
  !
  implicit none
  real(chm_real) DELTA2
  !
  CALL EPCONS2(DELTA2,PERTCONST,PERTSHAUX)
  !
  RETURN
END SUBROUTINE EPCONS
!
SUBROUTINE EPCONS2(DELTA2,PCONSTR,PSHAUX)
  !
  !     Stefan Boresch, April 1995
  !     calculate the constraint correction to a free energy
  !     Lagrangian multipliers from last SHAKE iteration have to
  !     be in PSHAUX(1,LL)!
  !
  use chm_kinds
  use dimens_fcm
  use number
  use shake
  use pshake
  use energym
  use epert_mod
  implicit none
  !
  real(chm_real) DELTA2
  INTEGER PCONSTR(*)
  real(chm_real) PSHAUX(2,*)
  !
  !
  real(chm_real) CCONS
  INTEGER LL

  DO LL=1,NCONST
     IF (PCONSTR(LL) > 0) THEN
        CCONS=PSHAUX(1,LL)*SQRT(CONSTR(LL))* &
             PSHAUX(2,LL)
        ETPRTM(USER)=ETPRTM(USER)+CCONS/DELTA2
     ENDIF
  ENDDO
  !     this is idiotic, but that way in 'epsum' etprtd should have the
  !     right value...
  ETPRTL(USER)=TWO*ETPRTM(USER)
  !SBp

  RETURN
END SUBROUTINE EPCONS2

SUBROUTINE EPSUM
  !
  !sb   Do some of the accumulation done originally in epert.  Do
  !     it here so that we can include constraint correction...
  !
  use chm_kinds
  use dimens_fcm
  !
  use bases_fcm
  use pert
  implicit none
  !
  CALL EPSUM2(PERTEPC,PERTEP1,PERTEP2)
  !
  RETURN
END SUBROUTINE EPSUM
!
SUBROUTINE EPSUM2(EPCONT,EPCON1,EPCON2)
  !
  !sb   Do some of the accumulation done originally in epert.  Do
  !     it here so that we can include constraint correction...
  !
  use chm_kinds
  use dimens_fcm
  use number
  use energym
  use econtmod
  use epert_mod
  use euler
  use fast
  use hbondm
  use image
  use inbnd
  use noem
  use psf
  use stream
  use timerm
  use pert
  use pshake
#if KEY_PARALLEL==1
  use parallel, only : mynod,numnod
#endif
  !
  !
  implicit none
  real(chm_real) EPCONT(*),EPCON1(*),EPCON2(*)
  !
  INTEGER I
  real(chm_real) EVAL,EVMAX,ETPRWF,ETPRWB

  ! accumulate separation shifted potential (not strictly necessary, but
  ! useful for debugging purposes
  IF (QPSSP) THEN
  ! SB, May 2018 -- this has been taken care of earlier, don't do it
  ! twice, otherwise results wrong!!!    
!#if KEY_PARALLEL==1
!     ! It should be broadcasted for those two terms. Otherwise, wrong (only right with 1 core).
!     if(numnod>1) then
!        call gcomb(ELJDLL,1)
!        call gcomb(ECODLL,1)
!     end if
!#endif
     EPSSLJ=EPSSLJ+ELJDLL*LAMDA+ELJDLM*LAMDAM
     EPSSCO=EPSSCO+ECODLL*LAMDA+ECODLM*LAMDAM
  ENDIF
  ! Get epdiff (only used selected energy terms).
  EPPRTD(EPOT) = ZERO
  !     we now need to integrate the add. dU/dL contribs
  !     into the standard PERT way of doing things...
  !     This requires breaking up of the original loop
  DO I=1,LENENT
     IF(QETPRT(I)) THEN
        ETPRTD(I)=ETPRTL(I)-ETPRTM(I)
     ENDIF
  ENDDO
  ETPRTD(VDW)=ETPRTD(VDW)+ELJDLL*LAMDA+ELJDLM*LAMDAM
  ETPRTD(ELEC)=ETPRTD(ELEC)+ECODLL*LAMDA+ECODLM*LAMDAM
  !     now we can average and compute fluctuations as usual
  DO I=1,LENENT
     IF(QETPRT(I)) THEN
        EPPRTD(EPOT) = EPPRTD(EPOT) + ETPRTD(I)
        ETPRTA(I) = ETPRTA(I) + ETPRTD(I)
        ETPRTF(I) = ETPRTF(I) + ETPRTD(I)**2
     ENDIF
  ENDDO
  !
  IF(QECONT) THEN
     DO I=1,NATIM
        EPCON1(I)=EPCON1(I)+ECONT(I)
        EPCON2(I)=EPCON2(I)+ECONT(I)**2
     ENDDO
  ENDIF
  !
  IF(PRNLEV > 6) THEN
     WRITE(OUTU,'(A,F10.5,A)') &
          ' PERTURBATION> lambda= ', LAMDA,'  accum:'
     CALL PRINTE(OUTU, EPPRTD, ETPRTD, 'LDIF', 'ENR', .TRUE., &
          IPNTOT, ZERO, ZERO, .TRUE.)
  ENDIF
  !
  !br...Output file for WHAM analysis (see also the WHAM command)
  IF(IWHAM > 0)THEN
     WRITE(IWHAM,*)  LAMDA, EPPRTD(EPOT)
  ENDIF
  ! Windowing method (TP)
  ! Modified to account for the forward and backward perturbations (Roux, July 1995).
  ! The simulation is run at E_lamda = lamda*E_0 + (1-lamda)*E_1
  ! the forward  is dG_f = -kBT*ln < exp(-[E_lamdaf-E_lamda]/kbt) >_lamda
  ! the backward is dG_b = -kBT*ln < exp(-[E_lamdai-E_lamda]/kbt) >_lamda
  ! the total pertubation is dG_tot =  dG_f - dG_b
  ! (note:  the energy terms corresponding to E_0 have an M
  !             energy terms corresponding to E_1 have an L )

  ETPRWF = (LAMDAF-LAMDA)*EPPRTD(EPOT)
  ETPRWB = (LAMDAI-LAMDA)*EPPRTD(EPOT)

  ! Check to save reference energy
  IF(IPNTOT == 0)THEN
     EPREF=EPPRTD(EPOT)
     EPREFF=ETPRWF
     EPREFB=ETPRWB
  ENDIF
  !
  ! Forward pert (note that the LAMDEL has been removed from DELDRT in pert.src)
  ETPRWF=ETPRWF-EPREFF
  EVAL=EXP(ETPRWF*DELDRT)
  EPTOT1F=EPTOT1F + EVAL
  EPTOT2F=EPTOT2F + EVAL*EVAL
  !
  ! Backward pert (note that the LAMDEL has been removed from DELDRT in pert.src)
  ETPRWB=ETPRWB-EPREFB
  EVAL=EXP(ETPRWB*DELDRT)
  EPTOT1B=EPTOT1B + EVAL
  EPTOT2B=EPTOT2B + EVAL*EVAL
  !
  ! Slow growth and TI methods
  EPDIFF=EPPRTD(EPOT)-EPREF
  !br...
  EPTOT3=EPTOT3+EPDIFF
  EPTOT4=EPTOT4+EPDIFF*EPDIFF
  !
  ! STEP COUNTER
  IPNTOT=IPNTOT+1
  !
  ! SSBP LRC
  EPSSBPLRC=EPSSBPLRC+EPLRCTMP
  EPSSBPLRCSQ=EPSSBPLRCSQ+EPLRCTMP*EPLRCTMP
  RETURN
END SUBROUTINE EPSUM2
!
!----------------------------------------------------------------------
!
!     all the PSSP specific energy routines -- need to comment this
!
SUBROUTINE ESSNBA(ENB,EEL,IFRSTA,NATOM,JNB,INBLO,CG,RSCLF,CNBA, &
     CNBB,MAXROW,IAC,ITC,NATC,IOFF,LELEC,LVDW,LCONS,LSHFT,LVSHFT, &
     LFSWT,LVFSWT,LDBEXP, &
#if KEY_NBIPS==1
     LVIPS,LEIPS,                                          & 
#endif
     DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
     QECONTX,ECONTX,DD1,IUPT,QSECD)
  !
  !----------------------------------------------------------------------
  !     THIS is a simplified version of the slow EVDW routine to compute
  !     nonbonded interactions between in PERT/PSSP between reactant/product
  !     and the environment.  It also does the PSSP specific derivatives
  !     These are stored in ECODLL and ELJDLL, routine EPERT is responsible
  !     for multiplying it appropriately with LAMDA and LAMDAM and to
  !     take the sign into account.
  !     restrictions: only CDIE; only supports:
  !     shif vswit
  !     fswi vswit
  !     ewal vswit
  !
  !     ENB    - vdw energy returned
  !     EEL    - electrostatic energy returned
  !     IFRSTA - first atom to look at in INBLO
  !     NATOM  - last atom to look at in INBLO (number of atoms)
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     RSCLF  - Radius Scaling Factor (ETPRTL(ELEC),BNBNDP,
  !     &               1,NATOM,NATOM)
  !     CNBA   - vdw well distance squared (MAXROW*2)
  !     CNBB   - vdw well depths (MAXROW*2)
  !     MAXROW  - offset for 1-4 interaction in CNBA and CNBB
  !     LELEC,,,,,,,, - logical flags used in BNBND.FCM
  !     ITC(IAC(J))  - lookup pointers into CNBA and CNBB  (NATOM)
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     DD1,QSECD - second derivative specifiers
  !
  !     By Bernard R. Brooks   5/7/81
  !
  !     Include BLOCK energy partition
  !     By Youngdo Won         12/15/90
  !
  !     Implement IPS and Double exponential potential
  !     By Xiongwu Wu         4/15/2023
  !----------------------------------------------------------------------
  use ewald,only: lewald,kappa,erfmod
  use erfcd_mod,only: erfcd
  use chm_kinds
  use dimens_fcm
  use number
  use econtmod
  use euler
  use fourdm
  use stream
  use consta
#if KEY_NBIPS==1
  use nbips      
#endif
  use pert
  ! for Ewald support
  !c
#if KEY_BLOCK==1
  use block_fcm
  use sftc
#endif 
  use machutil,only:die
  implicit none
  !
  real(chm_real) ENB,EEL
  INTEGER IFRSTA, NATOM
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*),RSCLF(*),CNBA(*),CNBB(*)
  INTEGER MAXROW
  INTEGER IAC(*),ITC(*)
  INTEGER NATC,IOFF(*)
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) CTONNB,CTOFNB,EPS,E14FAC
  LOGICAL QECONTX
  real(chm_real) ECONTX(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
  !
#if KEY_BLOCK==1
  INTEGER IBL,JBL
  real(chm_real)  COEF
#endif 
  !c
  real(chm_real)  COEFV, COEFE,DFV,DFE

  real(chm_real) DFRS,E14F,ERFCX,DRFC,ERFC2,E14M1
  !
  real(chm_real) ETEMP1,ETEMP2,ENBPR,EELPR
  real(chm_real) C2OFNB,C4ROF,C2ROF2,CHROF2,C2ONNB
  real(chm_real) RUL3,RUL12,SGSHSQ,ASH6,BSH6
  real(chm_real) CGF,CGT,RS,CGIJ
  real(chm_real) FDXI,FDYI,FDZI,CRXI,CRYI,CRZI,CGT2,DXI,DYI,DZI
  real(chm_real) S,SIG2,R2,SIG6,SIG12,RIJL,RIJU,FUNCT,DFN
  real(chm_real) EN,DEN,DF,DDF,G2,R1,G1,G3,DXIT,DYIT,DZIT
  real(chm_real) AXX,AYY,AZZ,AXY,AXZ,AYZ
  real(chm_real) EADD,EADDR,ON3,ON6,ONOFF2,OFF3,OFF4,OFF5,OFF6, &
       RMIN2,RMIN6, &
       R3,R4,R6,R8,DENOM,ACOEF,BCOEF,CCOEF,DCOEF,TWOA, &
       TWOC,FOURD,COVER3,DOVER5,CONST
  real(chm_real) RECOF3,RECOF6,OFDIF3,OFDIF6,ONOFF3,ONOFF6, &
       R5,CR6,CR12,RJUNK3,RJUNK6,RECOF2,MIN2OF
  INTEGER I,NB,IADD,ITEMP,J,NPR,I1,IACI,JPR,J1,IC,II,JJ,KK
  LOGICAL QAFIRST
  !
  LOGICAL RSHIFT,RSHFT,RSWIT,RFSWIT,CSHFT,CSWIT,CFSWIT,CSHIFT
  LOGICAL LSECD,ELECFG
  LOGICAL VDWFG,DSWIT,DSHFT,DFSWIT,ELCFG,LUSED
  !
  real(chm_real) R2DLAM,DPDL,ALR2MH,FSH,DFSH,MICD1
  real(chm_real) TSSP01,TSSP02,TSSP03,TSSP04,TSSP05,TSSP06, &
       TSSP07,TSSP08, &
       TSSP09,TSSP10,TSSP11,TSSP12,SQR2DL
#if KEY_NBIPS==1 /*nbips_comm*/
       !WXW Long range potential using isotropic periodic sum (IPS)
       LOGICAL LEIPS,LVIPS,REIPS,CEIPS,DVIPS,DOIPS
       real(chm_real) RIPS2RV,RIPSRE,RIPS2RE
       real(chm_real) U1,U2,U4,U6R,U12R
       real(chm_real) PE,PVC,PVA,DPE,DPVC,DPVA,DDPE,DDPVC,DDPVA
       real(chm_real) ENBA,ENBC
#endif /*   (nbips_comm)*/
   LOGICAL LDBEXP,DBEXP, DBEXPIPS
   real(chm_real) AIPSVAL(0:5),AIPSVCL(0:5),BIPSVAL(5),BIPSVCL(5)
   real(chm_real) ASC,EXPLA,EXPLC,AIJ,CIJ,DFA,DFC,LAPSSP2,FLAPSSP
   real(chm_real) SIG,AKR,EXPAR,CKR,EXPCR,AKRC,EXPARC,CKRC,EXPCRC
       !
  LSECD=QSECD
  E14M1 = E14FAC-ONE
  ELECFG=LELEC.AND.(EPS /= ZERO)
  ! Set flags for electrostatics options (8 possibilities)
   IF(.NOT.ELECFG .or. lewald) THEN
     RSHFT= .FALSE.
     RSWIT= .FALSE.
     CSWIT= .FALSE.
     CSHFT= .FALSE.
     RSHIFT=.FALSE.
     RFSWIT=.FALSE.
     CSHIFT=.FALSE.
     CFSWIT=.FALSE.
  ELSE
     RSHIFT= .NOT.LCONS .AND.      LSHFT .AND.      LFSWT
     RFSWIT= .NOT.LCONS .AND. .NOT.LSHFT .AND.      LFSWT
     RSHFT = .NOT.LCONS .AND.      LSHFT .AND. .NOT.LFSWT
     RSWIT = .NOT.LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT
     CSHIFT=      LCONS .AND.      LSHFT .AND.      LFSWT
     CFSWIT=      LCONS .AND. .NOT.LSHFT .AND.      LFSWT
     CSHFT =      LCONS .AND.      LSHFT .AND. .NOT.LFSWT
     CSWIT =      LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT
  ENDIF
#if KEY_NBIPS==1 /*nbips_comm*/
  IF(leips) THEN
   RSHFT= .FALSE.
   RSWIT= .FALSE.
   CSWIT= .FALSE.
   CSHFT= .FALSE.
   RSHIFT=.FALSE.
   RFSWIT=.FALSE.
   CSHIFT=.FALSE.
   CFSWIT=.FALSE.
   !
   RIPS2RE=ONE/(RIPS2+LAPSSP*ALAMBD)
   RIPSRE=SQRT(RIPS2RE)
   ENDIF
#endif /*nbips_comm*/
 
  !
  ELCFG=.FALSE.
  QAFIRST=.TRUE.
  !
  ! Set flags for van der Waals options
  VDWFG=LVDW
  IF(VDWFG) THEN
    DBEXP = LDBEXP
#if KEY_NBIPS==1
    DBEXP = DBEXP .and. .not. LVIPS
    DVIPS  = LVIPS   .and. .not. LDBEXP                             
    DBEXPIPS = LDBEXP .and. LVIPS
#endif
    IF(DBEXP.OR.DBEXPIPS)THEN
      LAPSSP2=LAPSSP**(LAORDER-1)
      FLAPSSP=LAORDER*LAPSSP2
      EXPLA=EXP(-LAPSSP2*LAPSSP*AEXPPOT)
      EXPLC=EXP(-LAPSSP2*LAPSSP*CEXPPOT)
      AIPSVAL(0)=AIPSVA(0)
      AIPSVAL(1)=AIPSVA(1)*EXPLA
      AIPSVAL(2)=AIPSVA(2)
      AIPSVAL(3)=AIPSVA(3)
      AIPSVAL(4)=AIPSVA(4)
      AIPSVAL(5)=AIPSVA(5)
      AIPSVCL(0)=AIPSVC(0)
      AIPSVCL(1)=AIPSVC(1)*EXPLC
      AIPSVCL(2)=AIPSVC(2)
      AIPSVCL(3)=AIPSVC(3)
      AIPSVCL(4)=AIPSVC(4)
      AIPSVCL(5)=AIPSVC(5)
     ! derivative parameters
      BIPSVAL(1)=-AEXPPOT*AIPSVAL(1)
      BIPSVCL(1)=-CEXPPOT*AIPSVCL(1)
    ELSE IF(DVIPS)THEN 
      RIPS2RV=ONE/(RIPS2+LAPSSP*DLAMBD)
    ENDIF
     DFSWIT = LVFSWT.and..not.lvips.and..not.dbexp
     DSHFT = .NOT.LVFSWT .AND.      LVSHFT.and..not.lvips.and..not.dbexp
     DSWIT = .NOT.LVFSWT .AND. .NOT.LVSHFT.and..not.lvips.and..not.dbexp
  ELSE
     DFSWIT = .FALSE.
     DSWIT  = .FALSE.
     DSHFT  = .FALSE.
  ENDIF
  !
  IF (.NOT.(ELECFG.OR.VDWFG).OR.CSHIFT) RETURN
  !
  NB=0
  !
  C2ONNB=CTONNB*CTONNB
  C2OFNB=CTOFNB*CTOFNB

#if KEY_BLOCK==1
  if(QBPSSP) then
     ALAMBD=BALAMBD   !Cc for ELEC
     DLAMBD=BDLAMBD   !Cc for VDW
  endif
#endif 
  !
  IF (CFSWIT) THEN
     !       force-based electrostatic switching coeffs
     IF(CTONNB  <  CTOFNB) THEN
        !mic
#if KEY_BLOCK==1
        if(.not.QBPSSP) then
#endif 

           TSSP01 = ONE/FIVE/(C2OFNB-C2ONNB)**3
           TSSP02 = -EIGHT*(C2OFNB-FIVE*C2ONNB-FOUR*ALAMBD*LAPSSP)* &
                (C2OFNB+ALAMBD*LAPSSP)**1.5
           TSSP03 = -FIVE*C2OFNB**3+FIFTN*C2OFNB**2*C2ONNB+TWENTY* &
                ALAMBD*LAPSSP*C2OFNB*(THREE*C2ONNB+TWO*ALAMBD* &
                LAPSSP)+ &
                EIGHT*ALAMBD**2*LAPSSP**2*(FIVE*C2ONNB+ &
                FOUR*ALAMBD*LAPSSP)
           TSSP04 = TEN*C2OFNB*(THREE*C2ONNB+TWO*ALAMBD*LAPSSP)+FOUR* &
                ALAMBD*LAPSSP*(FIVE*C2ONNB+FOUR*ALAMBD*LAPSSP)
           TSSP05 = -FIVE*(C2OFNB+C2ONNB)-FOUR*ALAMBD*LAPSSP
           TSSP06 = EIGHT*((-FIVE*C2OFNB+C2ONNB-FOUR*ALAMBD*LAPSSP)* &
                (C2ONNB+ALAMBD*LAPSSP)**1.5+(C2OFNB+ALAMBD* &
                LAPSSP)**1.5*(FIVE*C2ONNB-C2OFNB+FOUR*ALAMBD*LAPSSP) &
                )/(FIVE*(CTOFNB-CTONNB)**3*(CTOFNB+CTONNB)**3)
           TSSP07 = ALAMBD/TWO/(C2OFNB-C2ONNB)**3
           TSSP08 = EIGHT*SQRT(C2OFNB+ALAMBD*LAPSSP)*(THREE*C2ONNB+C2OFNB &
                +FOUR*ALAMBD*LAPSSP)
           TSSP09 = C2OFNB**3-THREE*C2OFNB**2*C2ONNB+TWELVE*ALAMBD* &
                C2OFNB*LAPSSP*(C2ONNB+TWO*ALAMBD*LAPSSP)+EIGHT* &
                ALAMBD**2*LAPSSP**2*(THREE*C2ONNB+FOUR*ALAMBD*LAPSSP)
           TSSP10 = TWO*NINE*C2OFNB*(C2ONNB+TWO*ALAMBD*LAPSSP)+TWELVE* &
                ALAMBD*LAPSSP*(THREE*C2ONNB+FOUR*ALAMBD*LAPSSP)
           TSSP11 = NINE*(C2ONNB+C2OFNB)+TWELVE*ALAMBD*LAPSSP
           TSSP12 = EIGHT/(SQRT(C2ONNB+ALAMBD*LAPSSP)*(THREE*C2OFNB+FOUR* &
                ALAMBD*LAPSSP+C2ONNB)+SQRT(C2OFNB+ALAMBD*LAPSSP)* &
                (THREE*C2ONNB+C2OFNB+FOUR*ALAMBD*LAPSSP))

#if KEY_BLOCK==1
        endif
#endif 
        !
        ONOFF2 = C2ONNB*C2OFNB
        ON3    = C2ONNB*CTONNB
        OFF3   = C2OFNB*CTOFNB
        OFF4   = C2OFNB*C2OFNB
        OFF5   = OFF3*C2OFNB
        DENOM  = ONE/(C2OFNB-C2ONNB)**3
        EADD   = (ONOFF2*(CTOFNB-CTONNB)-(OFF5-ON3*C2ONNB)/FIVE)* &
             EIGHT*DENOM
        ACOEF  = OFF4*(C2OFNB-THREE*C2ONNB)*DENOM
        BCOEF  = SIX*ONOFF2*DENOM
        COVER3 = -(C2ONNB+C2OFNB)*DENOM
        CCOEF  = THREE*COVER3
        DCOEF  = TWO*DENOM
        TWOA   = TWO*ACOEF
        TWOC   = TWO*CCOEF
        FOURD  = FOUR*DCOEF
        DOVER5 = DCOEF/FIVE
        CONST  = BCOEF*CTOFNB-ACOEF/CTOFNB+COVER3*OFF3+DOVER5*OFF5
     ELSE
        EADD  = -ONE/CTOFNB
        EADDR = -ONE/C2OFNB
     ENDIF
     !
  ELSE IF (LSHFT) THEN
     !     SHIFTED DIELECTRIC COEFFICIENTS
     RECOF2 = ONE/C2OFNB
     MIN2OF = MINTWO/CTOFNB
     C4ROF=ONE/(C2OFNB*C2OFNB)
     C2ROF2=MINTWO/C2OFNB
     CHROF2=-HALF/C2OFNB
  ELSE
     !     SWITCHING ELECTROSTATIC OPTIONS
     IF (CTOFNB > CTONNB) THEN
        RUL3=ONE/(C2OFNB-C2ONNB)**3
        RUL12=RUL3*TWELVE
     ENDIF
  ENDIF
  !
  IF (DSWIT) THEN
     !     VAN DER WAAL DISTANCE SWITCHING COEFFICIENTS
     IF (CTOFNB > CTONNB) THEN
        RUL3=ONE/(C2OFNB-C2ONNB)**3
        RUL12=RUL3*TWELVE
     ENDIF
  ELSE
     !IF(VDWFG) CALL DIE
  ENDIF
  !
  ITEMP=0
  ENB=0.0
  EEL=0.0
  IF(ELECFG) CGF=CCELEC/EPS
  !
  !     Initialize the code look up offsets
  !
  J=0
  DO I=1,NATC
     IOFF(I)=J
     J=J+I
  ENDDO
  !
  !     DO VDW AND ELECTROSTATIC TERMS AS REQUESTED
  !
  !=======================================================================
  !   Main loop begin
  !=======================================================================
  DO I=IFRSTA,NATOM
     !
     NPR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
     IF(NPR > 0) THEN
        ETEMP1=0.0
        ETEMP2=0.0
        I1=ITC(IAC(I))
        IACI=IOFF(I1)
        !
        IF (ELECFG) THEN
           CGT=CGF*CG(I)
           ELCFG=(CGT /= 0.0)
        ELSE
           CGT=ZERO
        ENDIF
        !
        !     USE FDXI,FDYI,FDZI FOR ITH COMPONENT OF FORCE VECTORS
        !     USE CRXI,CRYI,CRZI FOR ITH COMPONENT OF THE COORDINATES
        !
        FDXI=DX(I)
        FDYI=DY(I)
        FDZI=DZ(I)
        CRXI=X(I)
        CRYI=Y(I)
        CRZI=Z(I)
        !
        DO JPR=1,NPR
           !
           NB=NB+1
           IF (JNB(NB) < 0) THEN
              CGT2=CGT*E14FAC
              E14F = E14M1
              J=-JNB(NB)
              J1=ITC(IAC(J))
              IF (I1 < J1) THEN
                 IC=IOFF(J1)+I1+MAXROW
              ELSE
                 IC=IACI+J1+MAXROW
              ENDIF
              DOIPS=.FALSE.
           ELSE
              CGT2=CGT
              E14F=ZERO
              J=JNB(NB)
              J1=ITC(IAC(J))
              IF (I1 < J1) THEN
                 IC=IOFF(J1)+I1
              ELSE
                 IC=IACI+J1
              ENDIF
              DOIPS=LVIPS.OR.LEIPS
           ENDIF

           !c
#if KEY_BLOCK==1
           if(QBPSSP) then !qbpssp
              IBL = IBLCKP(I)
              JBL = IBLCKP(J)
              IF (JBL  <  IBL) THEN
                 KK=IBL
                 IBL=JBL
                 JBL=KK
              ENDIF
              KK=IBL+JBL*(JBL-1)/2
              COEF = BLCOEP(KK)
              LAPSSP=1.0D0-COEF
              COEFV = BLCOEV(KK)
              COEFE = BLCOEE(KK)
           endif  !qbpssp
#endif 


           !
           DXI=CRXI-X(J)
           DYI=CRYI-Y(J)
           DZI=CRZI-Z(J)
           S=DXI*DXI+DYI*DYI+DZI*DZI
           R2=1.0/S
           R1 = SQRT(R2)
           !
           !     TO COMPUTE VDW INTERACTION FOR THIS PAIR
           !
           ! LUSED = vdw calculation done
           LUSED = .FALSE.

#if KEY_BLOCK==1
           if(QBPSSP) then !qbpssp
              LAPSSP=1.0D0-COEFV
              !c Reset for safety
              ENBPR=0
           endif  !qbpssp
#endif 


           !
           IF (DSWIT) THEN
              !     VAN DER WAAL DISTANCE SWITCHING FUNCTION
              !
              IF (S < C2OFNB) THEN
                 R2DLAM = ONE/(S+LAPSSP*DLAMBD)
                 SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2DLAM
                 SIG6=SIG2*SIG2*SIG2
                 SIG12=SIG6*SIG6
                 !
                 IF (S > C2ONNB) THEN
                    !
                    RIJL=C2ONNB-S
                    RIJU=C2OFNB-S
                    FUNCT=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                    DFN=RIJL*RIJU*RUL12
                    EN=CNBB(IC)*(SIG12-SIG6-SIG6)
                    ENBPR=(FUNCT*EN)
                    DEN=CNBB(IC)*TWELVE*(SIG6-SIG12)*R2DLAM
                    DF=DFN*EN+FUNCT*DEN
                    ELJDLL=ELJDLL-FUNCT*(THREE*DLAMBD*(EN+ &
                         CNBB(IC)*SIG12)*R2DLAM)
                    !
                 ELSE
                    !
                    ENBPR=(CNBB(IC)*(SIG12-SIG6-SIG6))
                    DF=CNBB(IC)*R2DLAM*12.0*(SIG6-SIG12)
                    ELJDLL=ELJDLL-THREE*DLAMBD*(ENBPR+ &
                         CNBB(IC)*SIG12)*R2DLAM
                 ENDIF
                 !
                 LUSED=.TRUE.
                 !
              ENDIF
              !
            ELSE IF (DVIPS) THEN
               !     IPS VAN DER WAAL 
               !
               IF (DOIPS.AND.S < C2OFNB) THEN
                  R2DLAM = ONE/(S+LAPSSP*DLAMBD)
               SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*RIPS2RV
               SIG6=SIG2*SIG2*SIG2
               SIG12=SIG6*SIG6
               AIJ=CNBB(IC)*SIG12
               CIJ=TWO*CNBB(IC)*SIG6
               U2=RIPS2RV/R2DLAM
               U4=U2*U2
               U6R=ONE/U4/U2
               U12R=U6R*U6R
               !  L-J r6 term
               !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
               !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
               !
               PVC=U6R+AIPSVC(0)+U2*(AIPSVC(1)+U2*(AIPSVC(2)+U2*(AIPSVC(3) &
                    +U2*(AIPSVC(4)+U4*(AIPSVC(5)+U4*AIPSVC(6))))))
               DPVC=-SIX*U6R+U2*(BIPSVC(1)+U2*(BIPSVC(2)+U2*(BIPSVC(3) &
                    +U2*(BIPSVC(4)+U4*(BIPSVC(5)+U4*BIPSVC(6))))))
               !  L-J r12 term 
               !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
               !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
               !
               PVA=U12R+AIPSVA(0)+U2*(AIPSVA(1)+U2*(AIPSVA(2)+U2*(AIPSVA(3) &
                    +U4*(AIPSVA(4)+U4*(AIPSVA(5)+U4*AIPSVA(6))))))
               DPVA=-TWELVE*U12R+U2*(BIPSVA(1)+U2*(BIPSVA(2)+U2*(BIPSVA(3) &
                    +U4*(BIPSVA(4)+U4*(BIPSVA(5)+U4*BIPSVA(6))))))
               ENBPR=AIJ*(PVA-PIPSVAC)-CIJ*(PVC-PIPSVCC)
               DF=(AIJ*DPVA-CIJ*DPVC)*R2DLAM
               ! DE/DL=DE/DrL*DrL/DL+DE/DrcL*DrcL/DL=DLAMDA/(2rL^2)*DE/DrL+DLAMDA/(2cL^2)*DE/DcL
               ! DF=DE/Dr/r=DE/DrL/(rL^2)
               ELJDLL=ELJDLL+HALF*DLAMBD*((ONE-U2)*DF-(TWELVE*AIJ*PVA-SIX*CIJ*PVC)*RIPS2RV)  
               ELSE
                 ENBPR=ZERO
                 DF=ZERO
               ENDIF
                  !
                  LUSED=.TRUE.
                  !
               !
               ELSE IF (DBEXP) THEN
                  !     double exponential VAN DER WAAL 
                  !
                     SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)/S
                     SIG=SQRT(SIG2)
                     IF(S < C2OFNB)THEN
                       AKR=AIPSVAL(0)/SIG
                       !AKRC=AIPSVA(0)*RIPS/SIG
                       EXPAR=EXP(-AKR)
                       ENBA=AIPSVAL(1)*EXPAR
                       ! force 
                       DFA=BIPSVAL(1)*EXPAR/SIG
                       CKR=AIPSVCL(0)/SIG
                       !CKRC=AIPSVC(0)*RIPS/SIG
                       EXPCR=EXP(-CKR)
                       ENBC=AIPSVCL(1)*EXPCR
                      ! energy
                       ENBPR=CNBB(IC)*(ENBA-ENBC)
                       ! force
                       DFC=BIPSVCL(1)*EXPCR/SIG
                       DF=CNBB(IC)*(DFA-DFC)/S
                       ! De/Dl=DF*r*(rm-r)/(1-l)
                       ELJDLL=ELJDLL-CNBB(IC)*FLAPSSP*(AEXPPOT*ENBA-CEXPPOT*ENBC)
                     ELSE
                        ENBPR=ZERO
                        DF=ZERO
                     ENDIF
                     !
                     LUSED=.TRUE.
                     !
                  !
               ELSE IF (DBEXPIPS) THEN
               !     double exponential VAN DER WAAL 
               !
                  SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)/S
                  SIG=SQRT(SIG2)
                  IF(DOIPS.AND.SRIPS*SIG>ONE)THEN
                    AKR=AIPSVAL(0)/SIG
                    !AKRC=AIPSVA(0)*RIPS/SIG
                    EXPAR=EXP(-AKR)
                    ! energy
                    ENBA=AIPSVAL(1)*(EXPAR+AIPSVAL(4)/EXPAR+AIPSVAL(5))
                    ! force 
                    DFA=BIPSVAL(1)*(EXPAR-AIPSVAL(4)/EXPAR)/SIG
                    CKR=AIPSVCL(0)/SIG
                    !CKRC=AIPSVC(0)*RIPS/SIG
                    EXPCR=EXP(-CKR)
                    ! energy
                    ENBC=AIPSVCL(1)*(EXPCR+AIPSVCL(4)/EXPCR+AIPSVCL(5))
                    ENBPR=CNBB(IC)*(ENBA-ENBC)
                    ! force
                    DFC=BIPSVCL(1)*(EXPCR-AIPSVCL(4)/EXPCR)/SIG
                    DF=CNBB(IC)*(DFA-DFC)/S
                    ! De/Dl=DF*r*(rc-r)/(1-l)+E*a*(rc/rm-1)
                     ELJDLL=ELJDLL-CNBB(IC)*FLAPSSP*(AEXPPOT*ENBA-CEXPPOT*ENBC)
                    !ELJDLL=ELJDLL-CNBB(IC)*(AEXPPOT*((ONE-ONE/SIG)*ENBA+  &
                    !    TWO*(ONE/SIG-SRIPS)*(AIPSVAL(4)/EXPAR+AIPSVAL(5)))   &
                    !    -CEXPPOT*((ONE-ONE/SIG)*ENBC+  &
                    !    TWO*(ONE/SIG-SRIPS)*(AIPSVCL(4)/EXPCR+AIPSVCL(5))))
                  ELSE
                        ENBPR=ZERO
                        DF=ZERO
                  ENDIF
                  !
                  LUSED=.TRUE.
                  !
               !
               ENDIF

           !c
           DFV=DF


           !
           !     do electrostatics
           !
           EELPR=0.0

           !c
#if KEY_BLOCK==1
           if(QBPSSP) then !qbpssp
              LAPSSP=ONE-COEFE
           endif  !qbpssp
#endif 

           !
           IF (ELCFG) THEN
              IF (CG(J) /= 0.0) THEN
                 IF(S < C2OFNB) THEN
                    IF (.NOT.LUSED) THEN
                       DF=0.0
                       DDF=0.0
                       LUSED=.TRUE.
                       ENBPR=0.0
                    ENDIF
                    !
                    IF (CSHFT) THEN
                       !
                       ALR2MH=ONE/SQRT(ALAMBD*LAPSSP+S)
                       G1=CGT2*CG(J)*ALR2MH
                       FSH=(ONE-S/C2OFNB)
                       DFSH=FSH*TWO/R1*C2ROF2
                       FSH=FSH*FSH
                       EELPR=G1*FSH
                       MICD1 = -EELPR*ALR2MH*ALR2MH+G1*R1*DFSH
                       DF=DF+MICD1
                       ECODLL=ECODLL-G1*(ALR2MH*ALR2MH*HALF*ALAMBD)*FSH
                       !
                    ELSE IF (CFSWIT) THEN

#if KEY_BLOCK==1
                       if(QBPSSP) then

                          TSSP01 = ONE/FIVE/(C2OFNB-C2ONNB)**3
                          TSSP02 = -EIGHT*(C2OFNB-FIVE*C2ONNB-FOUR*ALAMBD*LAPSSP)* &
                               (C2OFNB+ALAMBD*LAPSSP)**1.5
                          TSSP03 = -FIVE*C2OFNB**3+FIFTN*C2OFNB**2*C2ONNB+TWENTY* &
                               ALAMBD*LAPSSP*C2OFNB*(THREE*C2ONNB+TWO*ALAMBD* &
                               LAPSSP)+ &
                               EIGHT*ALAMBD**2*LAPSSP**2*(FIVE*C2ONNB+ &
                               FOUR*ALAMBD*LAPSSP)
                          TSSP04 = TEN*C2OFNB*(THREE*C2ONNB+TWO*ALAMBD*LAPSSP)+FOUR* &
                               ALAMBD*LAPSSP*(FIVE*C2ONNB+FOUR*ALAMBD*LAPSSP)
                          TSSP05 = -FIVE*(C2OFNB+C2ONNB)-FOUR*ALAMBD*LAPSSP
                          TSSP06 = EIGHT*((-FIVE*C2OFNB+C2ONNB-FOUR*ALAMBD*LAPSSP)* &
                               (C2ONNB+ALAMBD*LAPSSP)**1.5+(C2OFNB+ALAMBD* &
                               LAPSSP)**1.5*(FIVE*C2ONNB-C2OFNB+FOUR*ALAMBD*LAPSSP) &
                               )/(FIVE*(CTOFNB-CTONNB)**3*(CTOFNB+CTONNB)**3)
                          TSSP07 = ALAMBD/TWO/(C2OFNB-C2ONNB)**3
                          TSSP08 = EIGHT*SQRT(C2OFNB+ALAMBD*LAPSSP)*(THREE*C2ONNB+C2OFNB &
                               +FOUR*ALAMBD*LAPSSP)
                          TSSP09 = C2OFNB**3-THREE*C2OFNB**2*C2ONNB+TWELVE*ALAMBD* &
                               C2OFNB*LAPSSP*(C2ONNB+TWO*ALAMBD*LAPSSP)+EIGHT* &
                               ALAMBD**2*LAPSSP**2*(THREE*C2ONNB+FOUR*ALAMBD*LAPSSP)
                          TSSP10 = TWO*NINE*C2OFNB*(C2ONNB+TWO*ALAMBD*LAPSSP)+TWELVE* &
                               ALAMBD*LAPSSP*(THREE*C2ONNB+FOUR*ALAMBD*LAPSSP)
                          TSSP11 = NINE*(C2ONNB+C2OFNB)+TWELVE*ALAMBD*LAPSSP
                          TSSP12 = EIGHT/(SQRT(C2ONNB+ALAMBD*LAPSSP)*(THREE*C2OFNB+FOUR* &
                               ALAMBD*LAPSSP+C2ONNB)+SQRT(C2OFNB+ALAMBD*LAPSSP)* &
                               (THREE*C2ONNB+C2OFNB+FOUR*ALAMBD*LAPSSP))

                       endif
#endif 

                       !                   cdie force-based switching
                       !      WRITE(OUTU,*) 'MICDBG> IN ESSNBA 9'
                       G1 = CGT2*CG(J)
                       R2DLAM = ONE/(S+lapssp*ALAMBD)
                       SQR2DL = SQRT(R2DLAM)
                       !
                       IF (S  >  C2ONNB) THEN
                          !mic
                          DF = DF+(-FIVE * G1 * TSSP01 * (C2OFNB-S)**2 &
                               * (C2OFNB - THREE*C2ONNB + TWO*S) ) &
                               * R2DLAM * SQR2DL
                          EELPR = G1 * TSSP01 * ( TSSP02 - SQR2DL*( &
                               TSSP03 + S*(TSSP04+TSSP05*S+TWO*S*S)))
                          ECODLL = ECODLL + TSSP07 * G1 * (TSSP08 - &
                               R2DLAM * SQR2DL * (TSSP09 + S*( &
                               TSSP10 + TSSP11*S -2*S*S)))
                       ELSE
                          DF = DF - G1 * R2DLAM * SQR2DL
                          EELPR = G1 * (TSSP06 + SQR2DL)
                          ECODLL = ECODLL + HALF * ALAMBD * G1 * &
                               (TSSP12 - R2DLAM * SQR2DL)
                          !
                       ENDIF
                     ELSE if (lewald) then
                        R2DLAM = ONE/(S+lapssp*ALAMBD)
                        SQR2DL = SQRT(R2DLAM)
                        RS=ONE/R1
                        CALL ERFCD(RS,KAPPA,ERFCX,DRFC,ERFMOD)
                        ERFC2=ERFCX + E14F
                        CGIJ = CGT*CG(J)
                        EELPR=CGIJ*ERFC2*sqr2dl
                        DFRS=CGIJ*DRFC*R1*sqr2dl + EELPR*r2dlam
                        DF=DF - DFRS
                        ECODLL=ECODLL-cgij*alambd*erfc2*r2dlam*sqr2dl/two
                     ELSE if (leips) then
                      IF(DOIPS)THEN
                        R2DLAM = ONE/(S+lapssp*ALAMBD)
                        U2=RIPS2RE/R2DLAM
                        IF(LCONS)THEN
                           CGIJ = CGT2*CG(J)*RIPSRE
                           U1=SQRT(U2)
                           PE=ONE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                           +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
                           DPE=-ONE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                           +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                           EELPR=CGIJ*(PE-PIPSEC)
                           DFRS=CGIJ*DPE*R2DLAM
                           ECODLL=ECODLL+HALF*ALAMBD*CGIJ*((ONE-U2)*DPE*R2+PE*RIPS2RE)
                        ELSE
                           CGIJ = CGT2*CG(J)*RIPS2RE
                           PE=ONE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                           +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
                           DPE=-TWO/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                           +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                           EELPR=CGIJ*(PE-PIPSEC)
                           DFRS=CGIJ*DPE*R2DLAM
                           ECODLL=ECODLL+HALF*ALAMBD*CGIJ*((ONE-U2)*DPE*R2+TWO*PE*RIPS2RE)
                        ENDIF
                        DF=DF + DFRS
                      ELSE
                        EELPR=ZERO
                      ENDIF
                     else
                       !     we should have trapped this earlier, but at least we are
                       !     on the safe side...
                       CALL WRNDIE(-5,'<ESSNBA>', &
                            'UNSUPPORTED ELEC OPTION WITH PSSP')

                    ENDIF
                 ENDIF
              ENDIF
           ENDIF
           !

           !c  get ELEC force
           DFE=DF-DFV

           IF(LUSED) THEN

#if KEY_BLOCK==1
              IF(QBPSSP) THEN
                 DF=DFV*COEFV+DFE*COEFE
                 ENBPR=ENBPR*COEFV
                 EELPR=EELPR*COEFE
              ENDIF
#endif 


              !
              DXIT=DXI*DF
              DYIT=DYI*DF
              DZIT=DZI*DF
              FDXI=FDXI+DXIT
              FDYI=FDYI+DYIT
              FDZI=FDZI+DZIT
              DX(J)=DX(J)-DXIT
              DY(J)=DY(J)-DYIT
              DZ(J)=DZ(J)-DZIT
              !
              ETEMP1=ETEMP1+ENBPR
              ETEMP2=ETEMP2+EELPR
              !
           ENDIF ! (LUSED)
        ENDDO  ! JPR
        !
        !     RESTORE ITH COMPONENT OF FORCE IN THE ARRAY
        !
        DX(I)=FDXI
        DY(I)=FDYI
        DZ(I)=FDZI
        !
        ENB=ENB+ETEMP1
        EEL=EEL+ETEMP2
        !
     ENDIF  ! (NPR > 0)
  ENDDO  ! I
  !=======================================================================
  !   Main loop end
  !=======================================================================
  !=======================================================================
  !
  RETURN
END SUBROUTINE ESSNBA
!
SUBROUTINE ESSNBG(EEL,ENB,NATOM,NST2,JNBG,INBLOG,CG,RSCLF, &
     DX,DY,DZ,NGRP,IGPBS,IGPTYP,INB14,IBLO14, &
     LELEC,LVDW,LCONS,LSHFT,LFSWT,LDBEXP, &
#if KEY_NBIPS==1
          LVIPS,LEIPS,                                          & 
#endif
     CNBA,CNBB,MAXCNX,ITC,IAC,NATC,IOFF,X,Y,Z, &
     CTONNB,CTOFNB,EPS,E14FAC,WRNMXD,QECONTX,ECONTX, &
     XCENT,YCENT,ZCENT,QCENT,DD1,IUPT,QSECD)

  !
  !     THIS is a simplified version of the slow EGROUP routine to compute
  !     nonbonded interactions between in PERT/PSSP between reactant/product
  !     and the environment.  It also does the PSSP specific derivatives
  !     These are stored in ECODLL and ELJDLL, routine EPERT is responsible
  !     for multiplying it appropriately with LAMDA and LAMDAM and to
  !     take the sign into account.
  !     restrictions: only CDIE; only supports:
  !     swit vswit
  !
  !     EEL    - electrostatic energy returned
  !     ENB    - van der Waals energy returned
  !     NATOM  - number of atoms
  !     NST2   - number of ST2 waters
  !     JNBG   - group pair list  (INBLOG(NGRP))
  !     INBLOG - pointers into group pair list  (NGRP)
  !     CG     - charges (NATOM)
  !     RSCLF  - Radius Scaling Factor (NATOM)
  !     DX,DY,DZ - force arrays
  !     NGRP   - number of groups total
  !     IGPBS  - base array for groups  (NGRP+1)
  !     IGPTYP - group type array (0-no charges,1-neutral,2-charged,3-ST2)
  !     INB14  - exclusion pair list for atoms  (INB14(NATOM))
  !     IBLO14 - pointer into INB14 (NATOM)
  !     LELEC,,,,,,LFSWT - flags from BNBND.FCM
  !     CNBA,CNBB,MAXCNX,ITC,IAC,NATC,IOFF -vdw parameters and data arrays.
  !     X,Y,Z  - coordinates
  !     CTONNB,CTOFNB - group switching funtion specifiers
  !     EPS    - dielectric constant
  !     E14FAC - Electrostatics scale factor for 1-4 interactions.
  !     WRNMXD - update warning move distance for extended electrostatics
  !     QECONTX - Analysis flag for energy partition.
  !     ECONTX  - Energy partition array
  !     XCENT,YCENT,ZCENT - center for each group
  !     QCENT  - Total change for each group
  !     DD1,IUPT,QSECD - Second derivative arrays and flag.
  !
  use chm_kinds
  use number
  use consta
  use dimens_fcm
  use stream
#if KEY_NBIPS==1
  use nbips      
#endif
  use pert
  !c
#if KEY_BLOCK==1
  use block_fcm
  use sftc
#endif 
  !
  implicit none


  !c
#if KEY_BLOCK==1
  INTEGER IBL,JBL,KK
  real(chm_real)  COEF
#endif 


  !
  real(chm_real) EEL,ENB
  INTEGER NATOM,NST2,INBLOG(*)
  INTEGER JNBG(*)
  real(chm_real) CG(*),RSCLF(*)
  real(chm_real) DX(*),DY(*),DZ(*)
  INTEGER NGRP
  INTEGER IGPBS(*),IGPTYP(*),INB14(*),IBLO14(*)
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LFSWT
  INTEGER MAXCNX,NATC
  real(chm_real) CNBA(*),CNBB(*)
  INTEGER ITC(*),IAC(*)
  INTEGER IOFF(NATC)
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) CTONNB,CTOFNB,EPS,E14FAC,WRNMXD
  LOGICAL QECONTX
  real(chm_real) ECONTX(*)
  real(chm_real) XCENT(*),YCENT(*),ZCENT(*),QCENT(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
  !
  ! local
  real(chm_real) ETEMP1,ETEMP2,EELPR,ENBPR,ESWADD
  real(chm_real) C2ONNB,C2OFNB,RUL3,RUL12
  real(chm_real) SIG2,SIG6,SIG12
  INTEGER IRS,IS,IQ,I,JRS,JS,JQ,J,II,JJ,IX,JX,IADD
  INTEGER NAT,NB,ITEMP,NPR,JRSPR,NI,NJ
  INTEGER LEX14,NXI,NXIMAX,JSX,INBX,I1,J1,IC
  real(chm_real) DXI,DYI,DZI,DXIT,DYIT,DZIT,DXJT,DYJT,DZJT, &
       DXIC,DYIC,DZIC
  real(chm_real) CGF,FUNCT,CGT,CGP,S,R2,DF,DFN,DFI,DFJ
  real(chm_real) AXX,AXY,AXZ,AYY,AYZ,AZZ,DDF,DDFN
  real(chm_real) SCENT,RIJL,RIJU,DFVDW,ADLFAC
  LOGICAL LCSW,LVGRP,LSWITR,LEXCL,LST2,LSWIT
  LOGICAL SKIP
  real(chm_real) RS,ERFCX,DRFC,E14M1,CGIJ
#if KEY_NBIPS==1 /*nbips_comm*/
  !WXW Long range potential using isotropic periodic sum (IPS)
  LOGICAL LEIPS,LVIPS,REIPS,CEIPS,DVIPS,DOIPS
  real(chm_real) RIPS2RV,RIPSRE,RIPS2RE
  real(chm_real) U1,U2,U4,U6R,U12R
  real(chm_real) PE,PVC,PVA,DPE,DPVC,DPVA,DDPE,DDPVC,DDPVA
  real(chm_real) ENBA,ENBC
#endif /*   (nbips_comm)*/
  LOGICAL LDBEXP,DBEXP, DBEXPIPS
  real(chm_real) AIPSVAL(0:5),AIPSVCL(0:5),BIPSVAL(5),BIPSVCL(5)
  real(chm_real) ASC,EXPLA,EXPLC,AIJ,CIJ,DFA,DFC,LAPSSP2,FLAPSSP
  real(chm_real) SIG,AKR,EXPAR,CKR,EXPCR,AKRC,EXPARC,CKRC,EXPCRC
  !
!
  ! Set option flags
  !

#if KEY_BLOCK==1
  if(QBPSSP) then
     ALAMBD=BALAMBD
     DLAMBD=BDLAMBD
  endif
#endif 


  LVGRP=LVDW
  IF (.NOT.(LELEC.OR.LVDW)) RETURN
  IF (EPS == ZERO .AND. .NOT.LVDW) RETURN
  IF (.NOT.LELEC .OR. EPS == ZERO) THEN
     LCSW=.FALSE.
  ELSE
     LCSW=.TRUE.
  ENDIF
  LSWIT=.TRUE.
  !
  IF(LSHFT.OR.(.NOT.LCONS)) THEN
#if KEY_NBIPS==1 /*nbips_comm*/
   IF(.NOT.LCONS.AND..NOT.LEIPS) THEN
#endif /*nbips_comm*/
      CALL WRNDIE(-4,'<EGROUP>','Invalid nonbond options')
     RETURN
#if KEY_NBIPS==1 /*nbips_comm*/
   ENDIF
#endif /*nbips_comm*/
ENDIF
  !
#if KEY_NBIPS==1 /*nbips_comm*/
  IF(leips) THEN
   LSHFT= .FALSE.
   LCSW = .FALSE.
   LSWIT= .FALSE.
   !
   RIPS2RE=ONE/(RIPS2+LAPSSP*ALAMBD)
   RIPSRE=SQRT(RIPS2RE)
   ENDIF
#endif /*nbips_comm*/
  !
IF(LVGRP) THEN
   DBEXP = LDBEXP
#if KEY_NBIPS==1
   DBEXP = DBEXP .and. .not. LVIPS
   DVIPS  = LVIPS   .and. .not. LDBEXP                             
   DBEXPIPS = LDBEXP .and. LVIPS
#endif
   IF(DBEXP.OR.DBEXPIPS)THEN
      LAPSSP2=LAPSSP**(LAORDER-1)
      FLAPSSP=LAORDER*LAPSSP2
      EXPLA=EXP(-LAPSSP2*LAPSSP*AEXPPOT)
      EXPLC=EXP(-LAPSSP2*LAPSSP*CEXPPOT)
      AIPSVAL(0)=AIPSVA(0)
      AIPSVAL(1)=AIPSVA(1)*EXPLA
      AIPSVAL(2)=AIPSVA(2)
      AIPSVAL(3)=AIPSVA(3)
      AIPSVAL(4)=AIPSVA(4)
      AIPSVAL(5)=AIPSVA(5)
      AIPSVCL(0)=AIPSVC(0)
      AIPSVCL(1)=AIPSVC(1)*EXPLC
      AIPSVCL(2)=AIPSVC(2)
      AIPSVCL(3)=AIPSVC(3)
      AIPSVCL(4)=AIPSVC(4)
      AIPSVCL(5)=AIPSVC(5)
     ! derivative parameters
      BIPSVAL(1)=-AEXPPOT*AIPSVAL(1)
      BIPSVCL(1)=-CEXPPOT*AIPSVCL(1)
   ELSE IF(DVIPS)THEN 
     RIPS2RV=ONE/(RIPS2+LAPSSP*DLAMBD)
   ENDIF
 ENDIF
 !
 IF(LVGRP) THEN
     J=0
     DO I=1,NATC
        IOFF(I)=J
        J=J+I
     ENDDO
     ENB=ZERO
  ENDIF
  ENBPR=ZERO
  ADLFAC=ZERO
  !
  E14M1 = E14FAC-ONE
  C2ONNB=CTONNB*CTONNB
  C2OFNB=CTOFNB*CTOFNB
  IF (CTOFNB > CTONNB) THEN
     RUL3=ONE/(C2OFNB-C2ONNB)**3
     RUL12=RUL3*TWELVE
  ENDIF
  !
  CGF=ZERO
  IF (LELEC .AND. EPS /= 0.0) CGF=CCELEC/EPS
  !
  ! Find group centers (no mass weighting)
  !
  DO IRS=1,NGRP
     IS=IGPBS(IRS)+1
     IQ=IGPBS(IRS+1)
     NAT=IQ-IS+1
     DXI=ZERO
     DYI=ZERO
     DZI=ZERO
     DO I=IS,IQ
        DXI=DXI+X(I)
        DYI=DYI+Y(I)
        DZI=DZI+Z(I)
     ENDDO
     XCENT(IRS)=DXI/NAT
     YCENT(IRS)=DYI/NAT
     ZCENT(IRS)=DZI/NAT
     IF (LFSWT) THEN
        DXI=ZERO
        DO I=IS,IQ
           DXI=DXI+CG(I)
        ENDDO
        QCENT(IRS)=ZERO
        IF(LCSW) THEN
#if KEY_BLOCK==1
           if(.not.qbpssp) then
#endif 
              QCENT(IRS)=DXI*SQRT(CGF/SQRT(C2OFNB+ALAMBD*LAPSSP))
              ADLFAC=HALF*ALAMBD/(C2OFNB+ALAMBD*LAPSSP)
#if KEY_BLOCK==1
           endif
#endif 

        ENDIF
     ENDIF
  ENDDO
  !
  EEL=ZERO
  ESWADD=ZERO
  !
  !=======================================================================
  !   Main loop begin
  !=======================================================================
  !     loop over pairs of groups in groups list
  !
  NB=0
  ITEMP=0
  DO IRS=1,NGRP
     NPR=INBLOG(IRS)-ITEMP
     ITEMP=INBLOG(IRS)
     IS=IGPBS(IRS)+1
     IQ=IGPBS(IRS+1)
     NI=IQ-IS+1
     !
     DO JRSPR=1,NPR
        NB=NB+1
        JRS=JNBG(NB)
        LEXCL=(JRS < 0)
        JRS=IABS(JRS)
        JS=IGPBS(JRS)+1
        JQ=IGPBS(JRS+1)
        NJ=JQ-JS+1
        ETEMP1=ZERO
        ETEMP2=ZERO
        FUNCT=ONE
        !
        !     PROCESS THIS GROUP PAIR
        !     do the exclusions only in the short range (ignore distances)
        IF (LEXCL) THEN
           !
           !     check atom exclusion list for exclusions
           IF(IS > 1) THEN
              NXIMAX=IBLO14(IS-1)
           ELSE
              NXIMAX=0
           ENDIF
           DO I=IS,IQ
              NXI=NXIMAX+1
              NXIMAX=IBLO14(I)
              CGT=CGF*CG(I)
              IF (IS == JS) THEN
                 JSX=I+1
              ELSE
                 JSX=JS
              ENDIF
              DO J=JSX,JQ
                 !
                 ! See if pair is in excluded list
                 !
#if KEY_BLOCK==1
                 if(qbpssp) then
                    IBL = IBLCKP(I)
                    JBL = IBLCKP(J)
                    IF (JBL  <  IBL) THEN
                       KK=IBL
                       IBL=JBL
                       JBL=KK
                    ENDIF
                    KK=IBL+JBL*(JBL-1)/2
                    COEF = BLCOEP(KK)
                    LAPSSP=1.0D0-COEF
                 endif
#endif 


                 CGP = CGT*CG(J)
                 INBX=IABS(INB14(NXI))
                 DO WHILE(NXI <= NXIMAX .AND. J > INBX)
                    NXI=NXI+1
                    INBX=IABS(INB14(NXI))
                 ENDDO
                 IF(NXI > NXIMAX) THEN
                    LEX14=0
                 ELSE IF(J == INB14(NXI)) THEN
                    LEX14=-1
                 ELSE
                    IF(J == INBX) THEN
                       LEX14=MAXCNX
                       CGP = CGP*E14FAC
                    ELSE
                       LEX14=0
                    ENDIF
                 ENDIF
                 IF(LEX14 >= 0) THEN
                    DXI=X(I)-X(J)
                    DYI=Y(I)-Y(J)
                    DZI=Z(I)-Z(J)
                    S=DXI*DXI+DYI*DYI+DZI*DZI
                    R2=ONE/(S+ALAMBD*LAPSSP)
                    !
                    IF(LCSW) THEN
                       EELPR=CGP*SQRT(R2)
                       DF=-R2*EELPR
                       ECODLL=ECODLL+DF*ALAMBD*HALF
                    ELSE IF(LEIPS)THEN
                      IF(LEX14==0.AND.S<RIPS2)THEN
                        U2=RIPS2RE/R2
                        IF(LCONS)THEN
                           CGIJ = CGP*RIPSRE
                           U1=SQRT(U2)
                           PE=ONE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                           +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
                           DPE=-ONE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                           +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                           EELPR=CGIJ*(PE-PIPSEC)
                           DF=CGIJ*DPE*R2
                           ECODLL=ECODLL+HALF*ALAMBD*CGIJ*((ONE-U2)*DPE*R2+PE*RIPS2RE)
                        ELSE
                           CGIJ = CGP*RIPS2RE
                           PE=ONE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                           +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
                           DPE=-TWO/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                           +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                           EELPR=CGIJ*(PE-PIPSEC)
                           DF=CGIJ*DPE*R2
                           ECODLL=ECODLL+HALF*ALAMBD*CGIJ*((ONE-U2)*DPE*R2+TWO*PE*RIPS2RE)
                        ENDIF
                      ELSE
                        EELPR=ZERO
                        DF=ZERO
                      ENDIF 
                    ELSE
                        EELPR=ZERO
                        DF=ZERO
                    ENDIF
                    !
                    IF(LVGRP) THEN
                       R2=ONE/(S+DLAMBD*LAPSSP)
                       I1=ITC(IAC(I))
                       J1=ITC(IAC(J))
                       IC=IOFF(MAX(I1,J1))+MIN(I1,J1)+LEX14
                       IF (DVIPS) THEN
                        !     IPS VAN DER WAAL 
                        !
                        IF(LEX14==0.AND.S<RIPS2)THEN
                        SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*RIPS2RV
                        SIG6=SIG2*SIG2*SIG2
                        SIG12=SIG6*SIG6
                        AIJ=CNBB(IC)*SIG12
                        CIJ=TWO*CNBB(IC)*SIG6
                        U2=RIPS2RV/R2
                        U4=U2*U2
                        U6R=ONE/U4/U2
                        U12R=U6R*U6R
                        !  L-J r6 term
                        !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
                        !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
                        !
                        PVC=U6R+AIPSVC(0)+U2*(AIPSVC(1)+U2*(AIPSVC(2)+U2*(AIPSVC(3) &
                             +U2*(AIPSVC(4)+U4*(AIPSVC(5)+U4*AIPSVC(6))))))
                        DPVC=-SIX*U6R+U2*(BIPSVC(1)+U2*(BIPSVC(2)+U2*(BIPSVC(3) &
                             +U2*(BIPSVC(4)+U4*(BIPSVC(5)+U4*BIPSVC(6))))))
                        !  L-J r12 term 
                        !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
                        !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
                        !
                        PVA=U12R+AIPSVA(0)+U2*(AIPSVA(1)+U2*(AIPSVA(2)+U2*(AIPSVA(3) &
                             +U4*(AIPSVA(4)+U4*(AIPSVA(5)+U4*AIPSVA(6))))))
                        DPVA=-TWELVE*U12R+U2*(BIPSVA(1)+U2*(BIPSVA(2)+U2*(BIPSVA(3) &
                             +U4*(BIPSVA(4)+U4*(BIPSVA(5)+U4*BIPSVA(6))))))
                        ENBPR=AIJ*(PVA-PIPSVAC)-CIJ*(PVC-PIPSVCC)
                        DFVDW=(AIJ*DPVA-CIJ*DPVC)*R2
                        DF=DF+DFVDW
                        ! DE/DL=DE/DrL*DrL/DL+DE/DrcL*DrcL/DL=DLAMDA/(2rL^2)*DE/DrL+DLAMDA/(2cL^2)*DE/DcL
                        ! DF=DE/Dr/r=DE/DrL/(rL^2)
                        ELJDLL=ELJDLL+HALF*DLAMBD*((ONE-U2)*DFVDW-(TWELVE*AIJ*PVA-SIX*CIJ*PVC)*RIPS2RV)  
                     ELSE
                          ENBPR=ZERO
                        ENDIF
                        !
                     ELSE IF (DBEXP) THEN
                           !     double exponential VAN DER WAAL 
                           !
                        IF(S<C2OFNB)THEN
                           SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)/S
                              SIG=SQRT(SIG2)
                                AKR=AIPSVAL(0)/SIG
                                !AKRC=AIPSVA(0)*RIPS/SIG
                                EXPAR=EXP(-AKR)
                                ENBA=AIPSVAL(1)*EXPAR
                                CKR=AIPSVCL(0)/SIG
                                !CKRC=AIPSVC(0)*RIPS/SIG
                                EXPCR=EXP(-CKR)
                                ENBC=AIPSVCL(1)*EXPCR
                                ! energy
                                ENBPR=CNBB(IC)*(ENBA-ENBC)
                                ! force
                                DFVDW=(BIPSVAL(1)*EXPAR-BIPSVCL(1)*EXPCR)/SIG
                                DF=DF+CNBB(IC)*DFVDW/S
                                ! De/Dl=DF*r*(rm-r)/(1-l)
                                ELJDLL=ELJDLL-CNBB(IC)*FLAPSSP*(AEXPPOT*ENBA-CEXPPOT*ENBC)
                                !ELJDLL=ELJDLL+CNBB(IC)*(AEXPPOT*AIPSVAL(1)*EXPAR   &
                                !        -CEXPPOT*AIPSVCL(1)*EXPCR)*(ONE-ONE/SIG)
                           !
                        ELSE
                           ENBPR=ZERO
                           DF=ZERO
                        ENDIF
                     ELSE IF (DBEXPIPS) THEN
                        !     double exponential VAN DER WAAL 
                        !
                           SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)/S
                           SIG=SQRT(SIG2)
                           IF(LEX14==0.AND.SRIPS*SIG>ONE)THEN
                             AKR=AIPSVAL(0)/SIG
                             !AKRC=AIPSVA(0)*RIPS/SIG
                             EXPAR=EXP(-AKR)
                             ! energy
                             ENBA=AIPSVAL(1)*(EXPAR+AIPSVAL(4)/EXPAR+AIPSVAL(5))
                             ! force 
                             DFA=BIPSVAL(1)*(EXPAR-AIPSVAL(4)/EXPAR)/SIG
                             CKR=AIPSVCL(0)/SIG
                             !CKRC=AIPSVC(0)*RIPS/SIG
                             EXPCR=EXP(-CKR)
                             ! energy
                             ENBC=AIPSVCL(1)*(EXPCR+AIPSVCL(4)/EXPCR+AIPSVCL(5))
                             ENBPR=CNBB(IC)*(ENBA-ENBC)
                             ! force
                             DFC=BIPSVCL(1)*(EXPCR-AIPSVCL(4)/EXPCR)/SIG
                             DFVDW=CNBB(IC)*(DFA-DFC)/S
                             !
                             DF=DF+DFVDW
                             ! De/Dl=DF*r*(rc-r)/(1-l)+E*a*(rc/rm-1)
                             ELJDLL=ELJDLL-CNBB(IC)*FLAPSSP*(AEXPPOT*ENBA-CEXPPOT*ENBC)
                             !ELJDLL=ELJDLL+CNBB(IC)*(AEXPPOT*((ONE-ONE/SIG)*ENBA+  &
                             !    TWO*(ONE/SIG-SRIPS)*(AIPSVAL(4)/EXPAR+AIPSVAL(5)))   &
                             !    -CEXPPOT*((ONE-ONE/SIG)*ENBC+  &
                             !    TWO*(ONE/SIG-SRIPS)*(AIPSVCL(4)/EXPCR+AIPSVCL(5))))
                           ELSE
                                 ENBPR=ZERO
                           ENDIF
                           !
                        ELSE
                           SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2
                           SIG6=SIG2*SIG2*SIG2
                           SIG12=SIG6*SIG6
                           DFVDW=CNBB(IC)*TWELVE*R2*(SIG6-SIG12)
                           DF=DF+DFVDW
                           ENBPR=CNBB(IC)*(SIG12-SIG6-SIG6)
                           ELJDLL=ELJDLL+HALF*DLAMBD*DFVDW
                        ENDIF
                     ELSE
                           ENBPR=ZERO
                     ENDIF
                     !

#if KEY_BLOCK==1
                    IF(qbpssp) THEN
                       DF=DF*COEF
                       EELPR=EELPR*COEF
                       ENBPR=ENBPR*COEF
                    ENDIF
#endif 


                    DXIT=DXI*DF
                    DYIT=DYI*DF
                    DZIT=DZI*DF
                    !
                    DX(I)=DX(I)+DXIT
                    DY(I)=DY(I)+DYIT
                    DZ(I)=DZ(I)+DZIT
                    DX(J)=DX(J)-DXIT
                    DY(J)=DY(J)-DYIT
                    DZ(J)=DZ(J)-DZIT
                    !
                    ETEMP1=ETEMP1+ENBPR
                    ETEMP2=ETEMP2+EELPR
                    !
                 ENDIF
              ENDDO
           ENDDO
           !
        ELSE
#if KEY_NBIPS==1
         IF(QIPS)THEN
            SCENT=ZERO
         ELSE
#endif 
     !
           ! Do switching function interaction
           !
           DXIC=XCENT(IRS)-XCENT(JRS)
           DYIC=YCENT(IRS)-YCENT(JRS)
           DZIC=ZCENT(IRS)-ZCENT(JRS)

#if KEY_BLOCK==1
           IF(qbpssp) THEN
              IBL = IBLCKP(IS)
              JBL = IBLCKP(JS)
              KK=MAX(IBL,JBL)
              KK=KK*(KK-1)/2+MIN(IBL,JBL)
              COEF = BLCOEP(KK)
              LAPSSP=1.0D0-COEF
           ENDIF
#endif 


           !
           SCENT=DXIC*DXIC+DYIC*DYIC+DZIC*DZIC
#if KEY_NBIPS==1
         ENDIF  
#endif
     !
           IF(SCENT < C2OFNB) THEN
              !
              LSWITR=(SCENT > C2ONNB)
              IF(LSWITR) THEN
                 RIJL=C2ONNB-SCENT
                 RIJU=C2OFNB-SCENT
                 FUNCT=RIJU*RIJU*(RIJU-3*RIJL)*RUL3
                 DFN=RIJL*RIJU*RUL12
              ELSE
                 FUNCT=ONE
              ENDIF
              IF(LFSWT) THEN
                 ESWADD=-QCENT(IRS)*QCENT(JRS)
#if KEY_BLOCK==1
                 IF(qbpssp) THEN
                    ETEMP2=ETEMP2*COEF
                 ENDIF
#endif 

                 ECODLL=ECODLL-ADLFAC*ESWADD*FUNCT
              ENDIF
              !
              DO I=IS,IQ
                 CGT=CGF*CG(I)
                 DO J=JS,JQ
                    DXI=X(I)-X(J)
                    DYI=Y(I)-Y(J)
                    DZI=Z(I)-Z(J)

#if KEY_BLOCK==1
                    IF(qbpssp) THEN
                       IBL = IBLCKP(I)
                       JBL = IBLCKP(J)
                       KK=MAX(IBL,JBL)
                       KK=KK*(KK-1)/2+MIN(IBL,JBL)
                       COEF = BLCOEP(KK)
                       LAPSSP=1.0D0-COEF
                    ENDIF
#endif 


                    S=DXI*DXI+DYI*DYI+DZI*DZI
                    R2=ONE/(S+ALAMBD*LAPSSP)
                    IF(LCSW) THEN
                       EELPR=CGT*CG(J)*SQRT(R2)
                       DF=-R2*EELPR
                       ECODLL=ECODLL+DF*FUNCT*ALAMBD*HALF
                    ELSE IF(LEIPS) THEN
                      IF(S<RIPS2)THEN
                        U2=RIPS2RE/R2
                        IF(LCONS)THEN
                           CGIJ = CGT*CG(J)*RIPSRE
                           U1=SQRT(U2)
                           PE=ONE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                           +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
                           DPE=-ONE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                           +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                           EELPR=CGIJ*(PE-PIPSEC)
                           DF=CGIJ*DPE*R2
                           ECODLL=ECODLL+HALF*ALAMBD*CGIJ*((ONE-U2)*DPE*R2+PE*RIPS2RE)
                        ELSE
                           CGIJ = CGT*CG(J)*RIPS2RE
                           PE=ONE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                           +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
                           DPE=-TWO/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                           +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
                           EELPR=CGIJ*(PE-PIPSEC)
                           DF=CGIJ*DPE*R2
                           ECODLL=ECODLL+HALF*ALAMBD*CGIJ*((ONE-U2)*DPE*R2+TWO*PE*RIPS2RE)
                        ENDIF
                      ELSE
                        EELPR=ZERO
                        DF=ZERO
                      ENDIF 
                    ELSE
                       EELPR=ZERO
                       DF=ZERO
                    ENDIF
                    !
                    IF(LVGRP) THEN
                       R2=ONE/(S+DLAMBD*LAPSSP)
                       I1=ITC(IAC(I))
                       J1=ITC(IAC(J))
                       IC=IOFF(MAX(I1,J1))+MIN(I1,J1)
                     IF (DVIPS) THEN
                        !     IPS VAN DER WAAL 
                        !
                        IF(S<RIPS2)THEN
                        SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*RIPS2RV
                        SIG6=SIG2*SIG2*SIG2
                        SIG12=SIG6*SIG6
                        AIJ=CNBB(IC)*SIG12
                        CIJ=TWO*CNBB(IC)*SIG6
                        U2=RIPS2RV/R2
                        U4=U2*U2
                        U6R=ONE/U4/U2
                        U12R=U6R*U6R
                        !  L-J r6 term
                        !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
                        !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
                        !
                        PVC=U6R+AIPSVC(0)+U2*(AIPSVC(1)+U2*(AIPSVC(2)+U2*(AIPSVC(3) &
                             +U2*(AIPSVC(4)+U4*(AIPSVC(5)+U4*AIPSVC(6))))))
                        DPVC=-SIX*U6R+U2*(BIPSVC(1)+U2*(BIPSVC(2)+U2*(BIPSVC(3) &
                             +U2*(BIPSVC(4)+U4*(BIPSVC(5)+U4*BIPSVC(6))))))
                        !  L-J r12 term 
                        !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
                        !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
                        !
                        PVA=U12R+AIPSVA(0)+U2*(AIPSVA(1)+U2*(AIPSVA(2)+U2*(AIPSVA(3) &
                             +U4*(AIPSVA(4)+U4*(AIPSVA(5)+U4*AIPSVA(6))))))
                        DPVA=-TWELVE*U12R+U2*(BIPSVA(1)+U2*(BIPSVA(2)+U2*(BIPSVA(3) &
                             +U4*(BIPSVA(4)+U4*(BIPSVA(5)+U4*BIPSVA(6))))))
                        ENBPR=AIJ*(PVA-PIPSVAC)-CIJ*(PVC-PIPSVCC)
                        DFVDW=(AIJ*DPVA-CIJ*DPVC)*R2
                        DF=DF+DFVDW
                        ! DE/DL=DE/DrL*DrL/DL+DE/DrcL*DrcL/DL=DLAMDA/(2rL^2)*DE/DrL+DLAMDA/(2cL^2)*DE/DcL
                        ! DF=DE/Dr/r=DE/DrL/(rL^2)
                        ELJDLL=ELJDLL+HALF*DLAMBD*((ONE-U2)*DFVDW-(TWELVE*AIJ*PVA-SIX*CIJ*PVC)*RIPS2RV)  
                     ELSE
                          ENBPR=ZERO
                        ENDIF
                        !
                     ELSE IF (DBEXP) THEN
                           !     double exponential VAN DER WAAL 
                           !
                        IF(S<C2OFNB)THEN
                           SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)/S
                              SIG=SQRT(SIG2)
                                AKR=AIPSVAL(0)/SIG
                                !AKRC=AIPSVA(0)*RIPS/SIG
                                EXPAR=EXP(-AKR)
                                ENBA=AIPSVAL(1)*EXPAR
                                CKR=AIPSVCL(0)/SIG
                                !CKRC=AIPSVC(0)*RIPS/SIG
                                EXPCR=EXP(-CKR)
                                ENBC=AIPSVCL(1)*EXPCR
                                ! energy
                                ENBPR=CNBB(IC)*(ENBA-ENBC)
                                ! force
                                DFVDW=(BIPSVAL(1)*EXPAR-BIPSVCL(1)*EXPCR)/SIG
                                DF=DF+CNBB(IC)*DFVDW/S
                                ! De/Dl=DF*r*(rm-r)/(1-l)
                                ELJDLL=ELJDLL-CNBB(IC)*FLAPSSP*(AEXPPOT*ENBA-CEXPPOT*ENBC)
                                !ELJDLL=ELJDLL+CNBB(IC)*(AEXPPOT*AIPSVAL(1)*EXPAR   &
                                !        -CEXPPOT*AIPSVCL(1)*EXPCR)*(ONE-ONE/SIG)
                           !
                        ELSE
                           ENBPR=ZERO
                           DF=ZERO
                        ENDIF
                     ELSE IF (DBEXPIPS) THEN
                        !     double exponential VAN DER WAAL 
                        !
                           SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)/S
                           SIG=SQRT(SIG2)
                           IF(SRIPS*SIG>ONE)THEN
                             AKR=AIPSVAL(0)/SIG
                             !AKRC=AIPSVA(0)*RIPS/SIG
                             EXPAR=EXP(-AKR)
                             ! energy
                             ENBA=AIPSVAL(1)*(EXPAR+AIPSVAL(4)/EXPAR+AIPSVAL(5))
                             ! force 
                             DFA=BIPSVAL(1)*(EXPAR-AIPSVAL(4)/EXPAR)/SIG
                             CKR=AIPSVCL(0)/SIG
                             !CKRC=AIPSVC(0)*RIPS/SIG
                             EXPCR=EXP(-CKR)
                             ! energy
                             ENBC=AIPSVCL(1)*(EXPCR+AIPSVCL(4)/EXPCR+AIPSVCL(5))
                             ENBPR=CNBB(IC)*(ENBA-ENBC)
                             ! force
                             DFC=BIPSVCL(1)*(EXPCR-AIPSVCL(4)/EXPCR)/SIG
                             DFVDW=CNBB(IC)*(DFA-DFC)/S
                             !
                             DF=DF+DFVDW
                             ! De/Dl=DF*r*(rc-r)/(1-l)+E*a*(rc/rm-1)
                             ELJDLL=ELJDLL-CNBB(IC)*FLAPSSP*(AEXPPOT*ENBA-CEXPPOT*ENBC)
                             !ELJDLL=ELJDLL+CNBB(IC)*(AEXPPOT*((ONE-ONE/SIG)*ENBA+  &
                             !    TWO*(ONE/SIG-SRIPS)*(AIPSVAL(4)/EXPAR+AIPSVAL(5)))   &
                             !    -CEXPPOT*((ONE-ONE/SIG)*ENBC+  &
                             !    TWO*(ONE/SIG-SRIPS)*(AIPSVCL(4)/EXPCR+AIPSVCL(5))))
                           ELSE
                                 ENBPR=ZERO
                           ENDIF
                           !
                        ELSE
                          SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*R2
                          SIG6=SIG2*SIG2*SIG2
                          SIG12=SIG6*SIG6
                          DFVDW=CNBB(IC)*TWELVE*R2*(SIG6-SIG12)
                          DF=DF+DFVDW
                          ENBPR=CNBB(IC)*(SIG12-SIG6-SIG6)
                          ELJDLL=ELJDLL+HALF*DLAMBD*DFVDW
                        ENDIF
                  ENDIF
                  !
                    DF=DF*FUNCT

#if KEY_BLOCK==1
                    IF(qbpssp) THEN
                       EELPR=EELPR*COEF
                       ENBPR=ENBPR*COEF
                       DF=DF*COEF
                    ENDIF
#endif 


                    !
                    DXIT=DXI*DF
                    DYIT=DYI*DF
                    DZIT=DZI*DF
                    DX(I)=DX(I)+DXIT
                    DY(I)=DY(I)+DYIT
                    DZ(I)=DZ(I)+DZIT
                    DX(J)=DX(J)-DXIT
                    DY(J)=DY(J)-DYIT
                    DZ(J)=DZ(J)-DZIT
                    !
                    ETEMP1=ETEMP1+ENBPR
                    ETEMP2=ETEMP2+EELPR
                    !
                 ENDDO
              ENDDO
              !
              ETEMP2=ETEMP2+ESWADD
              !
              IF(LSWITR) THEN
                 ! Backtransform center of geometry forces
                 IF(LSWIT) THEN
                    DFN=DFN*(ETEMP1+ETEMP2)
                    DFI=DFN/NI
                    DFJ=DFN/NJ
                    !
                    DXIT=DXIC*DFI
                    DYIT=DYIC*DFI
                    DZIT=DZIC*DFI
                    DXJT=DXIC*DFJ
                    DYJT=DYIC*DFJ
                    DZJT=DZIC*DFJ
                    DO I=IS,IQ
                       DX(I)=DX(I)+DXIT
                       DY(I)=DY(I)+DYIT
                       DZ(I)=DZ(I)+DZIT
                    ENDDO
                    DO J=JS,JQ
                       DX(J)=DX(J)-DXJT
                       DY(J)=DY(J)-DYJT
                       DZ(J)=DZ(J)-DZJT
                    ENDDO
                 ENDIF
                 !
                 ETEMP1=ETEMP1*FUNCT
                 ETEMP2=ETEMP2*FUNCT
              ENDIF  ! (LSWITR)
           ENDIF  ! (SCENT < C2OFNB)
        ENDIF  ! (LEXCL)
        !
        EEL=EEL+ETEMP2
        ENB=ENB+ETEMP1
        !
     ENDDO  ! JRSPR
  ENDDO  ! IRS
  return
end SUBROUTINE ESSNBG


  SUBROUTINE EIPSPERT(ENB,EEL,IFRSTA,ILASTA,NATOM,NATC, &
       LVDW,LELEC,LVIPS,LEIPS,LCONS,LDBEXP,INB,IBLO,  &
       CG,RSCLF,CNBA,CNBB,IAC,ITC,MAXROW, &
       EPS,E14FAC,VBOX,CGS,CG2S,AIIS,CIIS,AIJS,CIJS, &
       DX,DY,DZ,X,Y,Z,QECONTX,ECONTX,DD1,IUPT,LSECD,LPERT)
    !
    !   3D IPS interaction between excluded atom pairs
    !   This routine must be called first to update IPS parameters when needed
    !    and to inital electrostatic and vdw energies
    !
    !
    ! input/output
    use dimens_fcm
    use consta
    use number
    use stream
    use nbips
    use image
    use parallel
    use sftc          !Cc New PBLOCK
#if KEY_PERT==1
  use pert
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
#endif
    real(chm_real) ENB, EEL
    INTEGER IFRSTA,ILASTA,NATOM, NATC
    INTEGER IBLO(*),INB(*),ICALL
    real(chm_real) CG(*),RSCLF(*),CNBA(*), CNBB(*)
    INTEGER IAC(*), ITC(*),MAXROW
    real(chm_real)  EPS, E14FAC,VBOX,FIPS
    real(chm_real)  CGS,CG2S,AIIS,CIIS,AIJS,CIJS
    real(chm_real) DX(*),DY(*),DZ(*)
    real(chm_real) X(*),Y(*),Z(*)
    LOGICAL QECONTX,LPERT
    real(chm_real) ECONTX(*)
    LOGICAL LVDW,LELEC,LVIPS,LEIPS,LCONS,LDBEXP
    LOGICAL LLJIPSX,LELECX,LDEIPSX
    real(chm_real)  DD1(*)
    INTEGER IUPT(*)
    LOGICAL LSECD
     ! local
    INTEGER I,J,K,IC,ICX,ITI,ITJ,II,JJ,IADD
    INTEGER  ILAST, IFIRST,LJROW,IPI
    real(chm_real) ESCALE,VSCALE
    real(chm_real) DXI, DYI, DZI,DF,DFX,DFY,DFZ
    real(chm_real) ENBIJ,EELIJ,ECONTI
    real(chm_real)  XI,YI,ZI,XIJ,YIJ,ZIJ,R2
    real(chm_real)  CGF,CGI,CGIJ,AIJ,CIJ,AEXIJ,CEXIJ,SIG2,SIG6,SIG12
    real(chm_real)  U1,U2,U4,U6R,U12R
    real(chm_real) PE,PVC,PVA,DPE,DPVC,DPVA,DDPE,DDPVC,DDPVA
    real(chm_real) AXX,AYY,AZZ,AXY,AXZ,AYZ,DDF
    real(chm_real)  SIG,AKR,CKR,EXPAR,EXPCR,ENBA,ENBC,DFA,DFC,DFIJ
    real(chm_real)  SIGEX,EXPAREX,EXPCREX,SRIPS2
#if KEY_PERT==1
  !     REACTANT AND PRODUCT PART CALCULATED BY
  !     BY SEPARATE ROUTINE (SEE EPERT.SRC)
  !c
#if KEY_BLOCK==1
  IF (LPERT.AND.(TQPSSP.or.QBPSSP)) THEN
#else /**/
  IF (LPERT.AND.TQPSSP) THEN
#endif 
    CALL  ESSIPSPERT(ENB,EEL,IFRSTA,ILASTA,NATOM,NATC,  &
          LVDW,LELEC,LVIPS,LEIPS,LCONS,LDBEXP,INB,IBLO, &
          CG,RSCLF,CNBA,CNBB,IAC,ITC,MAXROW,&
          EPS,E14FAC,VBOX,CGS,CG2S,AIIS,CIIS,AIJS,CIJS,  &
          DX,DY,DZ,X,Y,Z,QECONTX,ECONTX,DD1,IUPT,LSECD,LPERT)
  !
    RETURN
  ENDIF
#endif /* KEY_PERT */
    ! begin
    LLJIPSX=LVDW.AND.LVIPS.AND..NOT.LDBEXP
    LDEIPSX=LVDW.AND.LVIPS.AND.LDBEXP
    LELECX=LELEC.AND.LEIPS
    CGF=CCELEC/EPS
    SRIPS2=SRIPS*SRIPS
    QIPSUPD=MOD(NIPSCNT,NIPSFRQ) == 0
    NIPSCNT=NIPSCNT+ICALL
    !
    ! check to see if IPS parameters need to be updated

    IF (QIPSINI )THEN
       QIPSFIN=QIPSFIN.OR.NTRANS == 0
       QIPSINI=.FALSE.
    ENDIF
    !
    IF(QIPSFIN)THEN
      EIPSSNB=ZERO
      EIPSSEL=ZERO
      VIRIPS=ZERO
      FIPS=ONE
    ELSE
      FIPS=FOUR*PI*RIPS2*RIPS/THREE/VBOX
#if KEY_PARALLEL==1
       IF(MYNOD == 0)THEN                
#endif
          IF(QAIPS)THEN
            IF(LDEIPSX)THEN 
               call wrndie(-2,'<AIPS>', &                                      
               ' IPS/DFFT not implemented for DE IPS (add "PXYZ")!')  
            ELSE
                ! Only the vdw 12th-term is considered here
             EIPSSNB=FIPS*AIJS*PIPSVAC*RIPS12R
             EIPSSEL=ZERO
            ENDIF
          ELSE
            IF(LDEIPSX)THEN 
               EIPSSNB=(AIJS/VBOX)*(PIPSVAC-PIPSVCC)
            ELSE IF(LVIPS)THEN
              EIPSSNB=FIPS*(AIJS*(PIPSVAC)*RIPS6R &
                  -CIJS*(PIPSVCC))*RIPS6R
            ENDIF
            EIPSSEL=FIPS*CGF*CGS*CGS*PIPSEC*RIPSR/TWO
          ENDIF
          IF(.NOT.LEIPS)EIPSSEL=ZERO
          IF(.NOT.LVIPS)EIPSSNB=ZERO
#if KEY_PARALLEL==1
       ELSE
          EIPSSNB=ZERO
          EIPSSEL=ZERO
       ENDIF
#endif 
       VIRIPS=(EIPSSNB+EIPSSEL)
    ENDIF
    ! apply option for netcharge boundary contribution
    IF(.NOT.QIPSNETCG)THEN
      !  ignoring net charge boudnary ele ips energy
      EIPSSEL=ZERO
    ENDIF 
    !
    ENB=EIPSSNB
    EEL=EIPSSEL
#if KEY_PARALLEL==1
    IF(MYNOD == 0)THEN                
#endif

         ! add self-term
         IF(LVDW)THEN
            IF(LDEIPSX)THEN 
               ENB=EIPSSNB+HALF*AIIS*(PIPSVA0-PIPSVC0)
            ELSE IF(LVIPS)THEN
               ENB=EIPSSNB+HALF*(AIIS*PIPSVA0*rips6r-CIIS*PIPSVC0)*rips6r
            ENDIF
         ENDIF
         IF(LELECX)THEN
            IF(LCONS)THEN 
               EEL=EIPSSEL+HALF*CG2S*PIPSE0*CGF*RIPSR
            ELSE
               EEL=EIPSSEL+HALF*CG2S*PIPSE0*CGF*RIPS2R
            ENDIF
         ENDIF
#if KEY_PARALLEL==1
    ENDIF
#endif 
    DO I=1,9
       PIPSVIR(I)=ZERO
    ENDDO
    PIPSVIR(1)=VIRIPS
    PIPSVIR(5)=VIRIPS
    PIPSVIR(9)=VIRIPS
#if KEY_DOMDEC==1
    if (q_domdec) RETURN
#endif
!IF(QPSSP)RETURN
    !  Evenly split long rang correction to each atom
    !=======================================================================
    !
    !=======================================================================
    !   Main loop begin
    !=======================================================================
    ILAST = 0
    EELIJ=ZERO
    ENBIJ=ZERO
    IF(IFRSTA > 1)ILAST=IBLO(IFRSTA-1)
    DO I=IFRSTA,ILASTA
       ! check to see if this is a pert atom
       ipi=PERTIP(I)
       !        define index list J
       IFIRST = ILAST+1
       ILAST=IBLO(I)
       IF((LPERT.AND.ipi>0).OR.((.NOT.LPERT).AND.ipi==0))THEN
        IF(LELECX)THEN
          IF(LCONS)THEN
            CGI=CG(I)*CGF*RIPSR
          ELSE
            CGI=CG(I)*CGF*RIPS2R
          ENDIF
            CGIJ=CG(I)*CGI
            ! charge self iteraction.  Already calculated
            EELIJ=HALF*CGIJ*PIPSE0
            !EEL=EEL+EELIJ
          IF(QECONTX) THEN
            ECONTX(I)=ECONTX(I)+EELIJ
          ENDIF
       ENDIF 
       IF(LVIPS)THEN
          ITI=ITC(IAC(I))
          IC=ITI*(ITI-1)/2+ITI
          IF(LDEIPSX)THEN 
             AIJ=RSCLF(I)*RSCLF(I)*CNBB(IC)
             ENBIJ=HALF*AIJ*(PIPSVA0-PIPSVC0)
          ELSE
            SIG2=RSCLF(I)*RSCLF(I)*CNBA(IC)*RIPS2R
            SIG6=SIG2*SIG2*SIG2
            SIG12=SIG6*SIG6
            AIJ=CNBB(IC)*SIG12
            CIJ=TWO*CNBB(IC)*SIG6
            ! Atom i long range referece and self interaction. Already calculated
            ENBIJ=HALF*(AIJ*PIPSVA0-CIJ*PIPSVC0)
          ENDIF
          !ENB=ENB+ENBIJ
          IF(QECONTX) THEN
            ECONTX(I)=ECONTX(I)+ENBIJ
          ENDIF
        ENDIF
       ENDIF
       !
       DXI=DX(I)
       DYI=DY(I)
       DZI=DZ(I)
       XI=X(I)
       YI=Y(I)
       ZI=Z(I)
       loop100: DO K=IFIRST,ILAST
          J = INB(K)
          ! skip 1-4 pairs while will be calculated in nonbond routines
          IF(J < 0)THEN
            J=-J
            ESCALE=E14FAC
            VSCALE=ONE
            LJROW=MAXROW
          ELSE
            ESCALE=ZERO
            VSCALE=ZERO
            LJROW=0
          ENDIF
          IF(LPERT)THEN
            IF(ipi+PERTIP(J)==0)CYCLE
          ELSE
            IF(ipi+pertip(j).NE.0)CYCLE
          ENDIF
          XIJ=XI-X(J)
          YIJ=YI-Y(J)
          ZIJ=ZI-Z(J)
          R2=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
          U2=R2*RIPS2R
          IF(LELECX)THEN
            !  Electrostatic IPS
            !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
            !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
            ! 
            CGIJ=CGI*CG(J)
            IF(LCONS)THEN
              U1=SQRT(U2)
              PE=ESCALE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                 +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
              DPE=-ESCALE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                 +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
            ELSE
               PE=ESCALE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                  +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
               DPE=-ESCALE/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                  +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
            ENDIF
            EELIJ=CGIJ*(PE-PIPSEC)
            EEL=EEL+EELIJ
            DF=-CGIJ*DPE/R2
            !debugwxw
            !if(j<5)write(*,*)"DEBUGWXW1: ",j,escale,cgij,u1,pe,dpe,eelij,df,eel
            IF(LSECD) THEN
               IF(LCONS)THEN
                  DDPE=TWO*ESCALE/U1+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                     +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
               ELSE
                  DDPE=SIX*ESCALE/U2+U2*(BBIPSE(1)+U2*(BBIPSE(2)+U2*(BBIPSE(3) &
                     +U2*(BBIPSE(4)+U2*(BBIPSE(5)+U2*BBIPSE(6))))))
               ENDIF
               DDF=CGIJ*DDPE/R2/R2
            ENDIF
          ELSE
            DF=ZERO
            DDF=ZERO
          ENDIF
          IF(LLJIPSX)THEN
             ITJ=ITC(IAC(J))
             IF(ITI > ITJ)THEN
                IC=ITI*(ITI-1)/2+ITJ
             ELSE
                IC=ITJ*(ITJ-1)/2+ITI
             ENDIF

             SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*RIPS2R
             SIG6=SIG2*SIG2*SIG2
             SIG12=SIG6*SIG6
             AIJ=CNBB(IC)*SIG12
             CIJ=TWO*CNBB(IC)*SIG6
             ICX=IC+LJROW
             SIG2=RSCLF(I)*RSCLF(J)*CNBA(ICX)*RIPS2R
             SIG6=SIG2*SIG2*SIG2
             SIG12=SIG6*SIG6
             AEXIJ=CNBB(ICX)*SIG12
             CEXIJ=TWO*CNBB(ICX)*SIG6
             ! debuging domdec
             !  AIJ=AEXIJ
             !  CIJ=CEXIJ
               AEXIJ=VSCALE*AEXIJ
               CEXIJ=VSCALE*CEXIJ
             !
             U4=U2*U2
             U6R=ONE/U4/U2
             U12R=U6R*U6R
             !  L-J r6 term
             !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
             !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
             !
             PVC=CEXIJ*U6R+CIJ*(AIPSVC(0)+U2*(AIPSVC(1) &
                  +U2*(AIPSVC(2)+U2*(AIPSVC(3)+U2*(AIPSVC(4) &
                  +U4*(AIPSVC(5)+U4*AIPSVC(6))))))-PIPSVCC)
             DPVC=-SIX*CEXIJ*U6R+CIJ*(U2*(BIPSVC(1) &
                  +U2*(BIPSVC(2)+U2*(BIPSVC(3)+U2*(BIPSVC(4) &
                  +U4*(BIPSVC(5)+U4*BIPSVC(6)))))))
             !  L-J r12 term 
             !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
             !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
             !
             PVA=AEXIJ*U12R+AIJ*(AIPSVA(0)+U2*(AIPSVA(1) &
                  +U2*(AIPSVA(2)+U2*(AIPSVA(3)+U4*(AIPSVA(4) &
                  +U4*(AIPSVA(5)+U4*AIPSVA(6))))))-PIPSVAC)
             DPVA=-TWELVE*AEXIJ*U12R+AIJ*(U2*(BIPSVA(1) &
                  +U2*(BIPSVA(2)+U2*(BIPSVA(3)+U4*(BIPSVA(4) &
                  +U4*(BIPSVA(5)+U4*BIPSVA(6)))))))
             ENBIJ=PVA-PVC
             ENB=ENB+ENBIJ
             DF=DF-(DPVA-DPVC)/R2
             IF(LSECD) THEN
              DDPVC=42.0D0*CEXIJ*U6R+CIJ*U2*(BBIPSVC(1)+U2*(BBIPSVC(2)+U2*(BBIPSVC(3) &
                    +U2*(BBIPSVC(4)+U4*(BBIPSVC(5)+U4*BBIPSVC(6))))))
              DDPVA=156.0D0*AEXIJ*U12R+AIJ*U2*(BBIPSVA(1)+U2*(BBIPSVA(2)+U2*(BBIPSVA(3) &
                    +U4*(BBIPSVA(4)+U4*(BBIPSVA(5)+U4*BBIPSVA(6))))))
              DDF=DDF+(DDPVA-DDPVC)/R2/R2
             ENDIF
          ELSE IF(LDEIPSX)THEN
              ITJ=ITC(IAC(J))
              IF(ITI > ITJ)THEN
                 IC=ITI*(ITI-1)/2+ITJ
              ELSE
                 IC=ITJ*(ITJ-1)/2+ITI
              ENDIF
              SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)
              IF(SIG2>RSMALL.AND.R2<SRIPS2*SIG2)THEN
                SIG=SQRT(SIG2/R2)
                AIJ=CNBB(IC)
                ICX=IC+LJROW
                SIGEX=SQRT(RSCLF(I)*RSCLF(J)*CNBA(ICX)/R2)
                AEXIJ=VSCALE*CNBB(ICX)
                AKR=AIPSVA(0)/SIG
                EXPAR=EXP(-AKR)
                EXPAREX=EXP(-AIPSVA(0)/SIGEX)
                ! energy
                ENBA=AEXIJ*AIPSVA(1)*EXPAREX+AIJ*AIPSVA(1)*(AIPSVA(4)/EXPAR+AIPSVA(5))
                ! force 
                DFA=AEXIJ*BIPSVA(1)*EXPAREX/SIGEX-AIJ*BIPSVA(1)*(AIPSVA(4)/EXPAR/SIG)
                CKR=AIPSVC(0)/SIG
                !CKRC=AIPSVC(0)*RIPS/SIG
                EXPCR=EXP(-CKR)
                EXPCREX=EXP(-AIPSVC(0)/SIGEX)
                ! energy
                ENBC=AEXIJ*AIPSVC(1)*EXPCREX+AIJ*AIPSVC(1)*(AIPSVC(4)/EXPCR+AIPSVC(5))
                ENBIJ=ENBA-ENBC
                ! force
                DFC=AEXIJ*BIPSVC(1)*EXPCREX/SIGEX-AIJ*BIPSVC(1)*(AIPSVC(4)/EXPCR/SIG)
                DFIJ=(DFA-DFC)/R2
                ENB=ENB+ENBIJ
                DF=DF-DFIJ
              ENDIF 
          ENDIF
          ! Calculate total forces
          DFX=DF*XIJ
          DFY=DF*YIJ
          DFZ=DF*ZIJ
          DXI=DXI-DFX
          DYI=DYI-DFY
          DZI=DZI-DFZ
          DX(J)=DX(J)+DFX
          DY(J)=DY(J)+DFY
          DZ(J)=DZ(J)+DFZ
           !
          IF(QECONTX) THEN
             ECONTI=HALF*(EELIJ+ENBIJ)
             ECONTX(I)=ECONTX(I)+ECONTI
             ECONTX(J)=ECONTX(J)+ECONTI
          ENDIF
          IF(LSECD) THEN
            !
               !
               DDF=DDF+DF/R2
               !
               !     NOW UPDATE DERIVATIVE MATRICIES
               !
               AXX=XIJ*XIJ*DDF-DF
               AYY=YIJ*YIJ*DDF-DF
               AZZ=ZIJ*ZIJ*DDF-DF
               AXY=XIJ*YIJ*DDF
               AXZ=XIJ*ZIJ*DDF
               AYZ=YIJ*ZIJ*DDF
               II=3*I-2
               JJ=3*J-2
               !
               IADD=IUPT(II)+II
               DD1(IADD)=DD1(IADD)+AXX
               IADD=IUPT(II+1)+II+1
               DD1(IADD)=DD1(IADD)+AYY
               IADD=IUPT(II+2)+II+2
               DD1(IADD)=DD1(IADD)+AZZ
               IADD=IUPT(II)+II+1
               DD1(IADD)=DD1(IADD)+AXY
               IADD=IUPT(II)+II+2
               DD1(IADD)=DD1(IADD)+AXZ
               IADD=IUPT(II+1)+II+2
               DD1(IADD)=DD1(IADD)+AYZ
               !
               IADD=IUPT(JJ)+JJ
               DD1(IADD)=DD1(IADD)+AXX
               IADD=IUPT(JJ+1)+JJ+1
               DD1(IADD)=DD1(IADD)+AYY
               IADD=IUPT(JJ+2)+JJ+2
               DD1(IADD)=DD1(IADD)+AZZ
               IADD=IUPT(JJ)+JJ+1
               DD1(IADD)=DD1(IADD)+AXY
               IADD=IUPT(JJ)+JJ+2
               DD1(IADD)=DD1(IADD)+AXZ
               IADD=IUPT(JJ+1)+JJ+2
               DD1(IADD)=DD1(IADD)+AYZ
               !
               IF (JJ < II) THEN
                  IADD=IUPT(JJ)+II
                  DD1(IADD)=DD1(IADD)-AXX
                  IADD=IUPT(JJ+1)+II+1
                  DD1(IADD)=DD1(IADD)-AYY
                  IADD=IUPT(JJ+2)+II+2
                  DD1(IADD)=DD1(IADD)-AZZ
                  IADD=IUPT(JJ)+II+1
                  DD1(IADD)=DD1(IADD)-AXY
                  IADD=IUPT(JJ+1)+II
                  DD1(IADD)=DD1(IADD)-AXY
                  IADD=IUPT(JJ)+II+2
                  DD1(IADD)=DD1(IADD)-AXZ
                  IADD=IUPT(JJ+2)+II
                  DD1(IADD)=DD1(IADD)-AXZ
                  IADD=IUPT(JJ+1)+II+2
                  DD1(IADD)=DD1(IADD)-AYZ
                  IADD=IUPT(JJ+2)+II+1
                  DD1(IADD)=DD1(IADD)-AYZ
               ELSE
                  IADD=IUPT(II)+JJ
                  DD1(IADD)=DD1(IADD)-AXX
                  IADD=IUPT(II+1)+JJ+1
                  DD1(IADD)=DD1(IADD)-AYY
                  IADD=IUPT(II+2)+JJ+2
                  DD1(IADD)=DD1(IADD)-AZZ
                  IADD=IUPT(II+1)+JJ
                  DD1(IADD)=DD1(IADD)-AXY
                  IADD=IUPT(II)+JJ+1
                  DD1(IADD)=DD1(IADD)-AXY
                  IADD=IUPT(II+2)+JJ
                  DD1(IADD)=DD1(IADD)-AXZ
                  IADD=IUPT(II)+JJ+2
                  DD1(IADD)=DD1(IADD)-AXZ
                  IADD=IUPT(II+2)+JJ+1
                  DD1(IADD)=DD1(IADD)-AYZ
                  IADD=IUPT(II+1)+JJ+2
                  DD1(IADD)=DD1(IADD)-AYZ
               ENDIF
               !
            !
         ENDIF  ! (LSECD)
       enddo loop100
       DX(I) =  DXI
       DY(I) =  DYI
       DZ(I) =  DZI
    ENDDO
    !=======================================================================
    !   Main loop end
    !=======================================================================
    RETURN
  END SUBROUTINE EIPSPERT

  SUBROUTINE ESSIPSPERT(ENB,EEL,IFRSTA,ILASTA,NATOM,NATC, &
       LVDW,LELEC,LVIPS,LEIPS,LCONS,LDBEXP,INB,IBLO,  &
       CG,RSCLF,CNBA,CNBB,IAC,ITC,MAXROW, &
       EPS,E14FAC,VBOX,CGS,CG2S,AIIS,CIIS,AIJS,CIJS, &
       DX,DY,DZ,X,Y,Z,QECONTX,ECONTX,DD1,IUPT,LSECD,LPERT)
    !
    !   3D IPS interaction between excluded atom pairs
    !   This routine must be called first to update IPS parameters when needed
    !    and to inital electrostatic and vdw energies
    !
    !
    ! input/output
    use dimens_fcm
    use consta
    use number
    use stream
    use nbips
    use image
    use parallel
#if KEY_PERT == 1
    use pert
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
#endif
    implicit none
    real(chm_real) ENB, EEL
    INTEGER IFRSTA,ILASTA,NATOM, NATC
    INTEGER IBLO(*),INB(*),ICALL
    real(chm_real) CG(*),RSCLF(*),CNBA(*), CNBB(*)
    INTEGER IAC(*), ITC(*),MAXROW
    real(chm_real)  EPS, E14FAC,VBOX,FIPS
    real(chm_real)  CGS,CG2S,AIIS,CIIS,AIJS,CIJS
    real(chm_real) DX(*),DY(*),DZ(*)
    real(chm_real) X(*),Y(*),Z(*)
    LOGICAL QECONTX,LPERT
    real(chm_real) ECONTX(*)
    LOGICAL LVDW,LELEC,LVIPS,LEIPS,LCONS,LDBEXP
    LOGICAL LLJIPSX,LELECX,LDEIPSX
    real(chm_real)  DD1(*)
    INTEGER IUPT(*)
    LOGICAL LSECD
     ! local
    INTEGER I,J,K,IC,ICX,ITI,ITJ,II,JJ,IADD
    INTEGER  ILAST, IFIRST,LJROW,IPI
    real(chm_real) ESCALE,VSCALE
    real(chm_real) DXI, DYI, DZI,DF,DFX,DFY,DFZ
    real(chm_real) ENBIJ,EELIJ,ECONTI
    real(chm_real)  XI,YI,ZI,XIJ,YIJ,ZIJ,R2
    real(chm_real)  CGF,CGI,CGIJ,AIJ,CIJ,AEXIJ,CEXIJ,SIG2,SIG6,SIG12
    real(chm_real)  U1,U2,U4,U6R,U12R
    real(chm_real) PE,PVC,PVA,DPE,DPVC,DPVA,DDPE,DDPVC,DDPVA
    real(chm_real) AXX,AYY,AZZ,AXY,AXZ,AYZ,DDF
    real(chm_real)  SIG,AKR,CKR,EXPAR,EXPCR,ENBA,ENBC,DFA,DFC,DFIJ
    real(chm_real)  SIGEX,EXPAREX,EXPCREX,SRIPS2
       
    real(chm_real) PIPSE0L,PIPSVA0L,PIPSVC0L,PIPSVACL,PIPSVCCL
    real(chm_real) PCODLL0,PVADLL0,PVCDLL0
    real(chm_real) AIPSVAL(0:5),AIPSVCL(0:5),BIPSVAL(5),BIPSVCL(5)
    real(chm_real) RIPS2RV,RIPS6RV,RIPSRE,RIPS2RE,R2DLAM
    real(chm_real) ASC,EXPLA,EXPLC,LAPSSP2,FLAPSSP
! begin
    LLJIPSX=LVDW.AND.LVIPS.AND..NOT.LDBEXP
    LDEIPSX=LVDW.AND.LVIPS.AND.LDBEXP
    LELECX=LELEC.AND.LEIPS
    CGF=CCELEC/EPS
    SRIPS2=SRIPS*SRIPS
    QIPSUPD=MOD(NIPSCNT,NIPSFRQ) == 0
    NIPSCNT=NIPSCNT+ICALL
    !
    IF(leips) THEN
      RIPS2RE=ONE/(RIPS2+LAPSSP*ALAMBD)
      RIPSRE=SQRT(RIPS2RE)
      U2=LAPSSP*ALAMBD*RIPS2RE
      PE=AIPSE(0)+U2*(AIPSE(1)+  &
         U2*(AIPSE(2)+U2*(AIPSE(3)+U2*(AIPSE(4)+U2*AIPSE(5)))))
      DPE=(BIPSE(1)+  &
         U2*(BIPSE(2)+U2*(BIPSE(3)+U2*(BIPSE(4)+U2*BIPSE(5)))))
      EELIJ=PE
      PIPSE0L=PE-PIPSEC
      IF(LCONS)THEN
         PCODLL0=HALF*ALAMBD*((ONE-U2)*DPE*RIPS2RE-EELIJ*RIPS2RE)
      ELSE
         PCODLL0=HALF*ALAMBD*((ONE-U2)*DPE*RIPS2RE-TWO*EELIJ*RIPS2RE)
      ENDIF
    ENDIF
    IF(LDBEXP)THEN
      LAPSSP2=LAPSSP**(LAORDER-1)
      FLAPSSP=LAORDER*LAPSSP2
      EXPLA=EXP(-LAPSSP2*LAPSSP*AEXPPOT)
      EXPLC=EXP(-LAPSSP2*LAPSSP*CEXPPOT)
      AIPSVAL(0)=AIPSVA(0)
      AIPSVAL(1)=AIPSVA(1)*EXPLA
      AIPSVAL(2)=AIPSVA(2)
      AIPSVAL(3)=AIPSVA(3)
      AIPSVAL(4)=AIPSVA(4)
      AIPSVAL(5)=AIPSVA(5)
      AIPSVCL(0)=AIPSVC(0)
      AIPSVCL(1)=AIPSVC(1)*EXPLC
      AIPSVCL(2)=AIPSVC(2)
      AIPSVCL(3)=AIPSVC(3)
      AIPSVCL(4)=AIPSVC(4)
      AIPSVCL(5)=AIPSVC(5)
     ! derivative parameters
      BIPSVAL(1)=-AEXPPOT*AIPSVAL(1)
      BIPSVCL(1)=-CEXPPOT*AIPSVCL(1)
      PIPSVA0L=PIPSVA0*EXPLA
      PIPSVC0L=PIPSVC0*EXPLC
      PIPSVACL=PIPSVAC*EXPLA
      PIPSVCCL=PIPSVCC*EXPLC
    ELSE 
      RIPS2RV=ONE/(RIPS2+LAPSSP*DLAMBD)
      RIPS6RV=RIPS2RV*RIPS2RV*RIPS2RV
      R2=LAPSSP*DLAMBD
      R2DLAM = ONE/(R2+LAPSSP*DLAMBD)
      U2=R2*RIPS2RV
      PVC=AIPSVC(0)+U2*(AIPSVC(1) &
                  +U2*(AIPSVC(2)+U2*(AIPSVC(3)+U2*(AIPSVC(4) &
                  +U4*(AIPSVC(5)+U4*AIPSVC(6))))))
      DPVC=(BIPSVC(1) &
                  +U2*(BIPSVC(2)+U2*(BIPSVC(3)+U2*(BIPSVC(4) &
                  +U4*(BIPSVC(5)+U4*BIPSVC(6))))))
      !  L-J r12 term 
      PVA=AIPSVA(0)+U2*(AIPSVA(1) &
                  +U2*(AIPSVA(2)+U2*(AIPSVA(3)+U4*(AIPSVA(4) &
                  +U4*(AIPSVA(5)+U4*AIPSVA(6))))))
      DPVA=(BIPSVA(1) &
                  +U2*(BIPSVA(2)+U2*(BIPSVA(3)+U4*(BIPSVA(4) &
                  +U4*(BIPSVA(5)+U4*BIPSVA(6))))))
      PIPSVA0L=PVA-PIPSVAC
      PIPSVC0L=PVC-PIPSVCC
      PVADLL0=HALF*DLAMBD*((ONE-U2)*DPVA*RIPS2RV-TWELVE*PVA*RIPS2RV)  
      PVCDLL0=HALF*DLAMBD*((ONE-U2)*DPVC*RIPS2RV-SIX*PVC*RIPS2RV) 
    ENDIF
    ! check to see if IPS parameters need to be updated

    IF (QIPSINI )THEN
       QIPSFIN=QIPSFIN.OR.NTRANS == 0
       QIPSINI=.FALSE.
    ENDIF
    !
    IF(QIPSFIN)THEN
      EIPSSNB=ZERO
      EIPSSEL=ZERO
      VIRIPS=ZERO
      FIPS=ONE
    ELSE
      FIPS=FOUR*PI*RIPS2*RIPS/THREE/VBOX
#if KEY_PARALLEL==1
       IF(MYNOD == 0)THEN                
#endif
          IF(QAIPS)THEN
            IF(LDEIPSX)THEN 
               call wrndie(-2,'<AIPS>', &                                      
               ' IPS/DFFT not implemented for DE IPS (add "PXYZ")!')  
            ELSE
                ! Only the vdw 12th-term is considered here
             EIPSSNB=FIPS*AIJS*PIPSVAC*RIPS6RV*RIPS6RV
             EIPSSEL=ZERO
            ENDIF
          ELSE
            IF(LDEIPSX)THEN 
               ! DBEXPIPS boundary energy is independent of lamda
               EIPSSNB=(AIJS/VBOX)*(PIPSVACL-PIPSVCCL)
               ELJDLL=ELJDLL-(AIJS/VBOX)*FLAPSSP*(AEXPPOT*PIPSVACL-CEXPPOT*PIPSVCCL)
            ELSE IF(LVIPS)THEN
               ! L-J ips boundary energy depends on lamda
              EIPSSNB=FIPS*(AIJS*(PIPSVAC)*RIPS6RV &
                  -CIJS*(PIPSVCC))*RIPS6RV
            ENDIF
            ! electrostatic boundary energy depends on lamda
            EIPSSEL=FIPS*CGF*CGS*CGS*PIPSEC*RIPSRE/TWO
          ENDIF
          IF(.NOT.LEIPS)EIPSSEL=ZERO
          IF(.NOT.LVIPS)EIPSSNB=ZERO
#if KEY_PARALLEL==1
       ELSE
          EIPSSNB=ZERO
          EIPSSEL=ZERO
       ENDIF
#endif 
       VIRIPS=(EIPSSNB+EIPSSEL)
    ENDIF
    ! apply option for netcharge boundary contribution
    IF(.NOT.QIPSNETCG)THEN
      !  ignoring net charge boudnary ele ips energy
      EIPSSEL=ZERO
    ENDIF 
    !
    ENB=EIPSSNB
    EEL=EIPSSEL
#if KEY_PARALLEL==1
    IF(MYNOD == 0)THEN                
#endif

      ! add self-term
      IF(LVDW)THEN
         IF(LDEIPSX)THEN 
            ENB=EIPSSNB+HALF*AIIS*(PIPSVA0L-PIPSVC0L)
            ELJDLL=ELJDLL-HALF*AIIS*FLAPSSP*(AEXPPOT*PIPSVA0L-CEXPPOT*PIPSVC0L)
         ELSE IF(LVIPS)THEN
            ENB=EIPSSNB+HALF*(AIIS*PIPSVA0L*rips6rv-CIIS*PIPSVC0L)*rips6rv
            ELJDLL=ELJDLL+HALF*(AIIS*PVADLL0*RIPS6RV-CIIS*PVCDLL0)*RIPS6RV
         ENDIF
      ENDIF
      IF(LELECX)THEN
         IF(LCONS)THEN 
            EEL=EIPSSEL+HALF*CG2S*PIPSE0L*CGF*RIPSRE
            ECODLL=ECODLL+HALF*CG2S*CGF*RIPSRE*PCODLL0
         ELSE
            EEL=EIPSSEL+HALF*CG2S*PIPSE0L*CGF*RIPS2RE
            ECODLL=ECODLL+HALF*CG2S*CGF*RIPS2RE*PCODLL0
         ENDIF
      ENDIF
#if KEY_PARALLEL==1
    ENDIF
#endif 
    DO I=1,9
       PIPSVIR(I)=ZERO
    ENDDO
    PIPSVIR(1)=VIRIPS
    PIPSVIR(5)=VIRIPS
    PIPSVIR(9)=VIRIPS
#if KEY_DOMDEC==1
    if (q_domdec) RETURN
#endif
!IF(QPSSP)RETURN
    !  Evenly split long rang correction to each atom
    !=======================================================================
    !
    !=======================================================================
    !   Main loop begin
    !=======================================================================
    ILAST = 0
    EELIJ=ZERO
    ENBIJ=ZERO
    IF(IFRSTA > 1)ILAST=IBLO(IFRSTA-1)
    DO I=IFRSTA,ILASTA
       ! check to see if this is a pert atom
       ipi=PERTIP(I)
       !        define index list J
       IFIRST = ILAST+1
       ILAST=IBLO(I)
       IF(ipi>0)THEN
         IF(LELECX)THEN
          IF(LCONS)THEN
            CGI=CG(I)*CGF*RIPSRE
          ELSE
            CGI=CG(I)*CGF*RIPS2RE
          ENDIF
          CGIJ=CGI*CG(I)
          EELIJ=HALF*CGIJ*PIPSE0L
          !EEL=EEL+EELIJ
          !ECODLL=ECODLL+HALF*CGIJ*PCODLL0
          IF(QECONTX) THEN
            ECONTX(I)=ECONTX(I)+EELIJ
          ENDIF
        ENDIF 
        IF(LVIPS)THEN
          ITI=ITC(IAC(I))
            IC=ITI*(ITI-1)/2+ITI
            IF(LDEIPSX)THEN 
               AIJ=RSCLF(I)*RSCLF(I)*CNBB(IC)
               ENBIJ=HALF*AIJ*PIPSVA0L
               !ENB=ENB+ENBIJ
               !ELJDLL=ELJDLL+HALF*AIJ*PVADLL0
            ELSE
              SIG2=RSCLF(I)*RSCLF(I)*CNBA(IC)*RIPS2RE
              SIG6=SIG2*SIG2*SIG2
              SIG12=SIG6*SIG6
              AIJ=CNBB(IC)*SIG12
              CIJ=TWO*CNBB(IC)*SIG6
              ! Atom i long range referece and self interaction. Already calculated
              ENBIJ=HALF*(AIJ*PIPSVA0L-CIJ*PIPSVC0L)
              !ENB=ENB+ENBIJ
              !ELJDLL=ELJDLL+HALF*(AIJ*PVADLL0*RIPS6RV-CIJ*PVCDLL0)*RIPS6RV
            ENDIF
          IF(QECONTX) THEN
            ECONTX(I)=ECONTX(I)+ENBIJ
          ENDIF
        ENDIF
       ENDIF /* ipi */
       !
       DXI=DX(I)
       DYI=DY(I)
       DZI=DZ(I)
       XI=X(I)
       YI=Y(I)
       ZI=Z(I)
       loop100: DO K=IFIRST,ILAST
         J = INB(K)
         ! skip 1-4 pairs while will be calculated in nonbond routines
         IF(J < 0)THEN
            J=-J
            ESCALE=E14FAC
            VSCALE=ONE
            LJROW=MAXROW
         ELSE
            ESCALE=ZERO
            VSCALE=ZERO
            LJROW=0
         ENDIF
         IF(ipi+PERTIP(J)==0)CYCLE
          XIJ=XI-X(J)
          YIJ=YI-Y(J)
          ZIJ=ZI-Z(J)
          R2=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
          IF(LELECX)THEN
           !  Electrostatic IPS
           !   etr1=1/r+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*(a5+a6*r2)))))
           !   detr1/dr*r=-1/r+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r2*(d5+d6*r2)))))
           ! 
           R2DLAM = ONE/(R2+lapssp*ALAMBD)
           U2=RIPS2RE/R2DLAM
           CGIJ=CGI*CG(J)
           IF(LCONS)THEN
             U1=SQRT(U2)
             PE=ESCALE/U1+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
             DPE=-ESCALE/U1+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
             EELIJ=CGIJ*(PE-PIPSEC)
             EEL=EEL+EELIJ
             DF=-CGIJ*DPE*R2DLAM
             ECODLL=ECODLL+HALF*ALAMBD*CGIJ*((ONE-U2)*DPE*R2DLAM+PE*RIPS2RE)
           ELSE
             PE=ESCALE/U2+U2*(AIPSE(1)+U2*(AIPSE(2)+U2*(AIPSE(3) &
                +U2*(AIPSE(4)+U2*(AIPSE(5)+U2*AIPSE(6))))))
             DPE=-ESCALE/U2+U2*(BIPSE(1)+U2*(BIPSE(2)+U2*(BIPSE(3) &
                +U2*(BIPSE(4)+U2*(BIPSE(5)+U2*BIPSE(6))))))
             EELIJ=CGIJ*(PE-PIPSEC)
             EEL=EEL+EELIJ
             DF=-CGIJ*DPE*R2DLAM
             ECODLL=ECODLL+HALF*ALAMBD*CGIJ*((ONE-U2)*DPE*R2DLAM+TWO*PE*RIPS2RE)
           ENDIF
          !debugwxw
          !if(j<5)write(*,*)"DEBUGWXW1: ",j,escale,cgij,u1,pe,dpe,eelij,df,eel
         ELSE
          DF=ZERO
         ENDIF
         IF(LLJIPSX)THEN
           ITJ=ITC(IAC(J))
           IF(ITI > ITJ)THEN
              IC=ITI*(ITI-1)/2+ITJ
           ELSE
              IC=ITJ*(ITJ-1)/2+ITI
           ENDIF

           R2DLAM = ONE/(R2+LAPSSP*DLAMBD)
           SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)*RIPS2RV
           SIG6=SIG2*SIG2*SIG2
           SIG12=SIG6*SIG6
           AIJ=CNBB(IC)*SIG12
           CIJ=TWO*CNBB(IC)*SIG6
           ICX=IC+LJROW
           SIG2=RSCLF(I)*RSCLF(J)*CNBA(ICX)*RIPS2RV
           SIG6=SIG2*SIG2*SIG2
           SIG12=SIG6*SIG6
           AEXIJ=CNBB(ICX)*SIG12
           CEXIJ=TWO*CNBB(ICX)*SIG6
           ! debuging domdec
           !  AIJ=AEXIJ
           !  CIJ=CEXIJ
           AEXIJ=VSCALE*AEXIJ
           CEXIJ=VSCALE*CEXIJ
           !
           U2=RIPS2RV/R2DLAM
           U4=U2*U2
           U6R=ONE/U4/U2
           U12R=U6R*U6R
           !  L-J r6 term
           !   etr6=1/r6+a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r4*(a5+a6*r4)))))
           !   detr6/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r2*(d4+r4*(d5+d6*r4)))))
           !
           PVC=CEXIJ*U6R+CIJ*(AIPSVC(0)+U2*(AIPSVC(1) &
                +U2*(AIPSVC(2)+U2*(AIPSVC(3)+U2*(AIPSVC(4) &
                +U4*(AIPSVC(5)+U4*AIPSVC(6)))))))
           DPVC=-SIX*CEXIJ*U6R+CIJ*(U2*(BIPSVC(1) &
                +U2*(BIPSVC(2)+U2*(BIPSVC(3)+U2*(BIPSVC(4) &
                +U4*(BIPSVC(5)+U4*BIPSVC(6)))))))
           !  L-J r12 term 
           !   etr12=1/r12+a0+r2*(a1+r2*(a2+r2*(a3+r4*(a4+r4*(a5+a6*r4)))))
           !   detr12/dr*r1=-6/r6+r2*(d1+r2*(d2+r2*(d3+r4*(d4+r4*(d5+d6*r4)))))
           !
           PVA=AEXIJ*U12R+AIJ*(AIPSVA(0)+U2*(AIPSVA(1) &
                +U2*(AIPSVA(2)+U2*(AIPSVA(3)+U4*(AIPSVA(4) &
                +U4*(AIPSVA(5)+U4*AIPSVA(6)))))))
           DPVA=-TWELVE*AEXIJ*U12R+AIJ*(U2*(BIPSVA(1) &
                +U2*(BIPSVA(2)+U2*(BIPSVA(3)+U4*(BIPSVA(4) &
                +U4*(BIPSVA(5)+U4*BIPSVA(6)))))))
           ENBIJ=PVA-AIJ*PIPSVAC-PVC+CIJ*PIPSVCC
           ENB=ENB+ENBIJ
           DFIJ=-(DPVA-DPVC)*R2DLAM
           DF=DF+DFIJ
           ELJDLL=ELJDLL-HALF*DLAMBD*((ONE-U2)*DFIJ+(TWELVE*PVA-SIX*PVC)*RIPS2RV)  
          ELSE IF(LDEIPSX)THEN
            ITJ=ITC(IAC(J))
            IF(ITI > ITJ)THEN
               IC=ITI*(ITI-1)/2+ITJ
            ELSE
               IC=ITJ*(ITJ-1)/2+ITI
            ENDIF
             SIG2=RSCLF(I)*RSCLF(J)*CNBA(IC)
             IF(SIG2>RSMALL.AND.R2<SRIPS2*SIG2)THEN
               SIG=SQRT(SIG2/R2)
               AIJ=CNBB(IC)
               ICX=IC+LJROW
               SIGEX=SQRT(RSCLF(I)*RSCLF(J)*CNBA(ICX)/R2)
               AEXIJ=VSCALE*CNBB(ICX)
               AKR=AIPSVAL(0)/SIG
               EXPAR=EXP(-AKR)
               EXPAREX=EXP(-AIPSVAL(0)/SIGEX)
               ! energy
               ENBA=AEXIJ*AIPSVAL(1)*EXPAREX+AIJ*AIPSVAL(1)*(AIPSVAL(4)/EXPAR+AIPSVAL(5))
               ! force 
               DFA=AEXIJ*BIPSVAL(1)*EXPAREX/SIGEX-AIJ*BIPSVAL(1)*(AIPSVAL(4)/EXPAR/SIG)
               CKR=AIPSVCL(0)/SIG
               !CKRC=AIPSVC(0)*RIPS/SIG
               EXPCR=EXP(-CKR)
               EXPCREX=EXP(-AIPSVCL(0)/SIGEX)
               ! energy
               ENBC=AEXIJ*AIPSVCL(1)*EXPCREX+AIJ*AIPSVCL(1)*(AIPSVCL(4)/EXPCR+AIPSVCL(5))
               ENBIJ=ENBA-ENBC
               ! force
               DFC=AEXIJ*BIPSVCL(1)*EXPCREX/SIGEX-AIJ*BIPSVCL(1)*(AIPSVCL(4)/EXPCR/SIG)
               DFIJ=-(DFA-DFC)/R2
                  ! De/Dl=DF*r*(rc-r)/(1-l)+E*a*(rc/rm-1)
               ELJDLL=ELJDLL-FLAPSSP*(AEXPPOT*ENBA-CEXPPOT*ENBC)
               ENB=ENB+ENBIJ
               DF=DF+DFIJ
             ENDIF 
          ENDIF
          ! Calculate total forces
          DFX=DF*XIJ
          DFY=DF*YIJ
          DFZ=DF*ZIJ
          DXI=DXI-DFX
          DYI=DYI-DFY
          DZI=DZI-DFZ
          DX(J)=DX(J)+DFX
          DY(J)=DY(J)+DFY
          DZ(J)=DZ(J)+DFZ
           !
       enddo loop100
       DX(I) =  DXI
       DY(I) =  DYI
       DZ(I) =  DZI
    ENDDO
    !=======================================================================
    !   Main loop end
    !=======================================================================
    RETURN
  END SUBROUTINE ESSIPSPERT
  
  SUBROUTINE IPSTERMP(NATOM,NATC,CG, &
   CNBA,CNBB,IAC,ITC,CGS,CG2S,AIIS,CIIS,AIJS,CIJS,  &
                     CGP,CG2P,AIIP,CIIP,AIJP,CIJP)
!-----------------------------------------------------------------------
!    Calculate boundary energies for 3D IPS systems
!
use dimens_fcm
use number
use consta
use stream
use image
use memory
use nbips
#if KEY_PERT==1
  use pert
#endif
#if KEY_PARALLEL==1 /*pll*/
use parallel
#endif /* (pll)*/
implicit none
INTEGER NATOM,NATC
real(chm_real) CG(*),CNBA(*), CNBB(*)
INTEGER IAC(*), ITC(*)
INTEGER I,J,ITI,ITJ,ITMAX,IC,IPI
real(chm_real)  CGI,CGIJ,AII,CII,AIJ,CIJ
real(chm_real)  SIG,SIG2,SIG6,SIG12
real(chm_real)  CGS,CG2S,AIIS,CIIS,AIJS,CIJS
real(chm_real)  CGP,CG2P,AIIP,CIIP,AIJP,CIJP
!
CGS=ZERO
CG2S=ZERO
CIIS=ZERO
AIIS=ZERO
CIJS=ZERO
AIJS=ZERO
CGP=ZERO
CG2P=ZERO
CIIP=ZERO
AIIP=ZERO
CIJP=ZERO
AIJP=ZERO
DO I=1,NATOM
   ITI=ITC(IAC(I))
   IC=ITI*(ITI-1)/2+ITI
   SIG2=CNBA(IC)
   CGI=CG(I)
   SIG2=CNBA(IC)
   IF(QIPSEXP)THEN
      SIG=SQRT(SIG2)
      ! make SRIPS accomandate the largest sigma
      IF(SIG2>RSMALL.AND.SRIPS>RIPS/SIG)SRIPS=RIPS/SIG
      AII=CNBB(IC)
      CII=AII
      AIJ=CNBB(IC)*SIG*SIG2
      CIJ=AIJ
   ELSE
     SIG6=SIG2*SIG2*SIG2
     SIG12=SIG6*SIG6
     AII=CNBB(IC)*SIG12
     CII=TWO*CNBB(IC)*SIG6
     AIJ=AII
     CIJ=CII
   ENDIF
   IPI=PERTIP(I)
   IF(IPI>0)THEN
     CGP=CGP+CGI
     CG2P=CG2P+CGI*CGI
     CIIP=CIIP+CII
     AIIP=AIIP+AII
     CIJP=CIJP+HALF*CIJ
     AIJP=AIJP+HALF*AIJ
   ELSE
     CGS=CGS+CGI
     CG2S=CG2S+CGI*CGI
     CIIS=CIIS+CII
     AIIS=AIIS+AII
     CIJS=CIJS+HALF*CIJ
     AIJS=AIJS+HALF*AIJ
   ENDIF
   DO J=I+1, NATOM
      CGIJ=CGI*CG(J)
      ITJ=ITC(IAC(J))
      IF(ITI > ITJ)THEN
         IC=ITI*(ITI-1)/2+ITJ
      ELSE
         IC=ITJ*(ITJ-1)/2+ITI
      ENDIF
      SIG2=CNBA(IC)
      IF(QIPSEXP)THEN
         SIG=SQRT(SIG2)
         IF(SIG2>RSMALL.AND.SRIPS>RIPS/SIG)SRIPS=RIPS/SIG
         AIJ=CNBB(IC)*SIG*SIG2
         CIJ=AIJ
      ELSE
         SIG6=SIG2*SIG2*SIG2
         SIG12=SIG6*SIG6
         AIJ=CNBB(IC)*SIG12
         CIJ=TWO*CNBB(IC)*SIG6
      ENDIF
      IF(IPI+PERTIP(J)==0)THEN
        CIJS=CIJS+CIJ
        AIJS=AIJS+AIJ
      ELSE
        CIJP=CIJP+CIJ
        AIJP=AIJP+AIJ
      ENDIF
   ENDDO
ENDDO
IF(QIPSEXP)THEN
   !AIIS=FOUR*PI*SRIPS**3*AIIS/THREE
   AIJS=FOUR*PI*SRIPS**3*AIJS/THREE
   AIJP=FOUR*PI*SRIPS**3*AIJP/THREE
   CIJS=AIJS
   CIJP=AIJP
ENDIF
RETURN 
END SUBROUTINE IPSTERMP


#endif 
SUBROUTINE NULL_EP
  RETURN
END SUBROUTINE NULL_EP

