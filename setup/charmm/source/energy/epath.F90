MODULE EPATHMOD
  use chm_kinds
  use dimens_fcm
  implicit none
  character(len=*),private,parameter :: file_name   ="epath.src"

  REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:),SAVE :: REPEI
  REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:),SAVE :: REPESQI
  REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:),SAVE :: REPFI
  REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:),SAVE :: REPFSQI
  REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:),SAVE :: REPELAi
  REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:),SAVE :: REPFLAi
  REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:),SAVE :: REPLENG
  REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:),SAVE :: &
       REFPX,REFPY,REFPZ,REFPDX,REFPDY,REFPDZ
  REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:),SAVE :: &
       REFAX,REFAY,REFAZ,REFBX,REFBY,REFBZ
  integer,allocatable,dimension(:),save::arepdmap,prepdmap

#if KEY_RPATH==1
  !  pderiv.fcm -  projected Forces common block.
  !  PJDX,PJDY,PJDZ - Force components
  !
  real(chm_real),allocatable,dimension(:),save :: &
       PJDX,PJDY,PJDZ,PSDX,PSDY,PSDZ,PTANX,PTANY,PTANZ
#endif 

#if KEY_REPDSTR==1
  !  I{X,Y,Z}P{L,R}   Neighboring arrays
  real(chm_real),allocatable,dimension(:),save :: IXPL, IYPL, IZPL
  real(chm_real),allocatable,dimension(:),save :: IXPR, IYPR, IZPR

  REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:,:),SAVE ::  &
       XPEERS,YPEERS,ZPEERS,XPEERC,YPEERC,ZPEERC
  REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:),SAVE ::  &
       APEERS
#endif 

  !=========================================================================
CONTAINS
  !=========================================================================

#if KEY_RPATH==0 /*rpath1*/
  SUBROUTINE EPATH(EPTHR,EPTHA,QEPTHR,QEPTHA, &
       DX,DY,DZ,X,Y,Z,QSECD,ICALL)

    real(chm_real) EPTHR,EPTHA
    LOGICAL QEPTHR,QEPTHA
    real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
    LOGICAL QSECD
    INTEGER ICALL

    return
  end SUBROUTINE EPATH
#else /* (rpath1)*/

  subroutine allocate_epath()
    use memory
    character(len=*),parameter :: routine_name="allocate_epath"
    call chmalloc(file_name,routine_name,'PJDX  ',maxaim,crl=PJDX )
    call chmalloc(file_name,routine_name,'PJDY  ',maxaim,crl=PJDY )
    call chmalloc(file_name,routine_name,'PJDZ  ',maxaim,crl=PJDZ )
    call chmalloc(file_name,routine_name,'PSDX  ',maxaim,crl=PSDX )
    call chmalloc(file_name,routine_name,'PSDY  ',maxaim,crl=PSDY )
    call chmalloc(file_name,routine_name,'PSDZ  ',maxaim,crl=PSDZ )
    call chmalloc(file_name,routine_name,'PTANX ',maxaim,crl=PTANX)
    call chmalloc(file_name,routine_name,'PTANY ',maxaim,crl=PTANY)
    call chmalloc(file_name,routine_name,'PTANZ ',maxaim,crl=PTANZ)


    return
  end subroutine allocate_epath

  SUBROUTINE EPATH(EPTHR,EPTHA,QEPTHR,QEPTHA, &
       DX,DY,DZ,X,Y,Z,QSECD,ICALL)

    use number
    use cnst_fcm

#if KEY_REPLICA==1 /*replica0*/

    use replica_mod
    use memory
    use pathm
    use psf
    use stream
    use econtmod           ! jwchu
    use parallel
    use repdstr
#endif /* (replica0)*/
    real(chm_real) EPTHR,EPTHA
    LOGICAL QEPTHR,QEPTHA
    real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
    LOGICAL QSECD
    INTEGER ICALL


#if KEY_REPLICA==1 /*replica1*/
    !
    ! Add NEB facility by jwchu Jun 02
    ! TWO arrays are added for NEB
    ! RPENR  - Energy of each replica, from ECONT
    ! PATANG - Tangent vector of replicated atoms of each image
    LOGICAL ERR,QPRINT
    INTEGER,ALLOCATABLE,DIMENSION(:) :: ISTREP,ICNREP
    real(chm_real)  WTOT,KRMSX,KANGX,KMNRMSX,KMXRMSX,EPTHN
    INTEGER,ALLOCATABLE,DIMENSION(:,:) :: ATOMPR
    real(chm_real),ALLOCATABLE,DIMENSION(:,:) :: DRA,DRB,rb,ra
    INTEGER NREP,ICIMG                  ! jwchu
    INTEGER I,J,K
    REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:) :: WEIGHT,BFARY,ARMS,BRMS
    REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:) :: DRMS,FRMS

    REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:) :: DRTEMP,RPENR,PATANG
    REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:) :: RPLENG,PATHDL,PATHDL1
    REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:) :: TFCE,PFCE,RPMF,EPMF
    REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:) :: FANG,ACST
    REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:,:) :: XTAN,YTAN,ZTAN
    REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:,:) :: XM1,YM1,ZM1,XP1,YP1,ZP1
    real(chm_real) EPKINE,EPLENG,EPHP

    EPTHR=ZERO
    EPTHA=ZERO
    QPRINT=(PRNLEV.GT.5)
    !
    !C      IF(QPRINT) WRITE(OUTU,45) NREPL
    !C  45  FORMAT(' Entering EPATH:  nrepl=',i5)
    !
#if KEY_REPDSTR==1
    IF(.NOT.QREPDSTR) THEN                      
#endif
       IF(.NOT.QREP) RETURN
       IF(.NOT.QPATH) RETURN
#if KEY_REPDSTR==1
    ELSE
       GOTO 999
    ENDIF
#endif 

    IF(QSECD) THEN
       CALL WRNDIE(-3,'<EPATH>','No second deriv. for path energy')
    ENDIF
    IF(NSUB.GT.2) THEN
       CALL WRNDIE(-3,'<EPATH>', &
            'Only one replica subsection allowed for EPATH')
       RETURN
    ENDIF
    IF(NSUB.LT.2) THEN
       CALL WRNDIE(-3,'<EPATH>', &
            'Replica subsection required for EPATH')
       RETURN
    ENDIF
    IF(NREPL.LT.3) THEN
       IF(.NOT.QPTAU) THEN
          CALL WRNDIE(-3,'<EPATH>', &
               'At least two replicates required for EPATH')
          RETURN
       ELSE ! QPTAU
          IF(NPCALL.EQ.1 .and. prnlev >= 2)THEN
             WRITE(OUTU,25) '    QPTAU with 1 replica only '
             WRITE(OUTU,26) NREPL
          ENDIF
       ENDIF ! QPTAU
    ENDIF

25  FORMAT(A)
26  FORMAT(I5)

999 CONTINUE
    !
    !     There is some confusion here:
    !     NREPL is one more then number of replicas
    !     Perhaps normally because we must delete the original
    !     in REPDSTR we don't have the originals...
    !     also in case of cyclic path we still have the same
    !     numbers fo replicas....
#if KEY_REPDSTR==1
    IF(QREPDSTR)NREPL=NREPDSTR+1                  
#endif

    NREP=NREPL-1

    call chmalloc('epath.src','EPATH','ISTREP',NREPL,intg=ISTREP)
    call chmalloc('epath.src','EPATH','ICNREP',NREPL,intg=ICNREP)
    call chmalloc('epath.src','EPATH','ARMS',NREPL,crl=ARMS)
    call chmalloc('epath.src','EPATH','BRMS',NREPL,crl=BRMS)
    call chmalloc('epath.src','EPATH','DRMS',NREPL,crl=DRMS)
    call chmalloc('epath.src','EPATH','FRMS',NREPL,crl=FRMS)
    call chmalloc('epath.src','EPATH','RPMF',NREPL,crl=RPMF)
    call chmalloc('epath.src','EPATH','EPMF',NREPL,crl=EPMF)
    call chmalloc('epath.src','EPATH','FANG',NREPL,crl=FANG)
    call chmalloc('epath.src','EPATH','ACST',NREPL,crl=ACST)
    call chmalloc('epath.src','EPATH','WEIGHT',NATOM,crl=WEIGHT)

    ARMS(1:NREPL) = ZERO
    BRMS(1:NREPL) = ZERO
    DRMS(1:NREPL) = ZERO
    FRMS(1:NREPL) = ZERO
    RPMF(1:NREPL) = ZERO
    EPMF(1:NREPL) = ZERO
    FANG(1:NREPL) = ZERO
    ACST(1:NREPL) = ZERO

    CALL EPATHC(ISTREP,ICNREP,WEIGHT,NATREP,WTOT, &
         QPWEIG,QPWCOM, &
#if KEY_COMP2==1
         QPWCM2,  & 
#endif
         QPMASS,QPRINT,ERR)

    IF(ERR) GOTO 100

    call chmalloc('epath.src','EPATH','ATOMPR',3,NATREP,intg=ATOMPR)
    call chmalloc('epath.src','EPATH','BFARY',3*NATREP,crl=BFARY)
    call chmalloc('epath.src','EPATH','RA',3,NATREP,crl=RA)
    call chmalloc('epath.src','EPATH','RB',3,NATREP,crl=RB)
    call chmalloc('epath.src','EPATH','DRA',3,NATREP,crl=DRA)
    call chmalloc('epath.src','EPATH','DRB',3,NATREP,crl=DRB)
    call chmalloc('epath.src','EPATH','DRTEMP',3*NATREP,crl=DRTEMP)
    call chmalloc('epath.src','EPATH','PATANG',3*NREPL*NATREP,crl=PATANG)
    call chmalloc('epath.src','EPATH','PATHDL',3*NREPL*NATREP,crl=PATHDL)
    IF (QPTAU) THEN
       call chmalloc('epath.src','EPATH','PATHDL1',3*NREPL*NATREP,crl=PATHDL1)
    ENDIF

    !-----------------------------------------------------------------------
    ! Process off path optimization
    IF(QPROPT) THEN
       IF(QEPTHR) THEN
          IF(QPATHFLG) THEN
             QPATHFLG=.FALSE.
             !C               write(50+mynodg,*)'EPATH>before refrms()'
             CALL REFRMS(BFARY,NREP, &
                  ISTREP,ARMS,BRMS, &
                  DRMS,FRMS,WEIGHT,WTOT, &
                  QPRINT,QPNOTR,QPNORT,QPCYCL,ATOMPR, &
                  RA,RB,DRA,DRB, &
                  DRTEMP,NATOM)
          ENDIF
          !
 !NOT sure about this??? MH-JUL10:  We don't want this in REPDSTR case!
#if KEY_REPDSTR==1
          IF(.NOT.QREPDSTR)THEN  
#endif
#if KEY_PARALLEL==1
             CALL VDGBR(DX,DY,DZ,0) 
#endif
#if KEY_REPDSTR==1
          ENDIF                  
#endif

          CALL GETDIST(EPTHR,EPTHA,DX,DY,DZ,X,Y,Z,NREP,KRMS, &
               KMXRMS,RMXRMS,XPREF,YPREF,ZPREF, &
               NATREP,ISTREP,ARMS,BRMS, &
               DRMS,FRMS, &
               WEIGHT,WTOT,QPRINT, &
               QPNOTR,QPNORT,QPCYCL,ATOMPR, &
               RA,RB,DRA,DRB,DRTEMP, &
               EVWID,NATOM)

          CALL PATHDYNA(DX,DY,DZ,NREP,QPRINT,QPCYCL,WEIGHT, &
               WTOT,ISTREP,BFARY,QPNOTR,QPNORT, &
               ATOMPR,RA,RB,DRA,DRB, &
               DRTEMP,NATOM)

          CALL EPATHO(EPTHR,EPTHA,DX,DY,DZ,X,Y,Z,NREP,KRMS, &
               KMXRMS,RMXRMS,XPREF,YPREF,ZPREF, &
               NATREP,ISTREP,ARMS,BRMS, &
               DRMS,FRMS, &
               WEIGHT,WTOT,QPRINT, &
               QPNOTR,QPNORT,QPCYCL,ATOMPR, &
               RA,RB,DRA,DRB,DRTEMP, &
               EVWID,NATOM)

       ENDIF
       !
       !-----------------------------------------------------------------------
       !     Process cartesian NEB method
    ELSE IF(QNEB) THEN
       CALL EPATHN(EPTHN,DX,DY,DZ,X,Y,Z,NREP,KNEB, &
            KMNRMS,RMNRMS,KMXRMS,RMXRMS, &
            NATREP,ISTREP,ARMS,FRMS, &
            WEIGHT,WTOT,QPRINT, &
            QPNOTR,QPNORT,QPCYCL,ATOMPR,EVWID, &
            RA,RB,DRA)
       EPTHR=EPTHR+EPTHN
       !-----------------------------------------------------------------------
       ! Process rms bestfit NEB method
    ELSE IF(QPNEB.OR.QPTAU) THEN
       call chmalloc('epath.src','EPATH','RPENR',NREPL,crl=RPENR)
       call chmalloc('epath.src','EPATH','RPLENG',NREPL,crl=RPLENG)
       call chmalloc('epath.src','EPATH','TFCE',NREPL,crl=TFCE)
       call chmalloc('epath.src','EPATH','PFCE',NREPL,crl=PFCE)
       RPENR(1:NREPL) = ZERO
       RPLENG(1:NREPL) = ZERO
       TFCE(1:NREPL) = ZERO
       PFCE(1:NREPL) = ZERO

       IF(.NOT.QPTAU) THEN

          CALL PATHANAL(DX,DY,DZ,X,Y,Z,REFPX,REFPY,REFPZ, &
               NREPL,NREP,NATREP,ISTREP,RPENR, &
               RPLENG,WEIGHT,WTOT,QPRINT,QPNOTR, &
               QPNORT,QPCYCL,QRFIX,QPETAN,ATOMPR,RA, &
               RB,DRA,DRB,DRTEMP,PATANG,PATHDL,EVWID,ICIMG, &
               REPEi,REPESQi,REPFi,REPFSQi, &
               REPELAi,REPFLAi,REPLENG,TFCE,PFCE,NPCALL,QPPMF)
          ! Calculate the tangent vector of each image if NEB
          !
          CALL NUDGEF(DX,DY,DZ,X,Y,Z,NREPL,NREP,NATREP,ISTREP, &
               QPRINT,QPCYCL,DRTEMP,PATANG, &
               PJDX,PJDY,PJDZ,PSDX,PSDY,PSDZ, &
               EVWID,QPCIMG,ICIMG,TFCE,PTANX,PTANY,PTANZ)

          IF(QPROPT1) THEN
             KRMSX=ZERO
             IF(QEPTHR) KRMSX=KRMS
             CALL EPATHO1(EPTHR,DX,DY,DZ,X,Y,Z,NREP,KRMSX,KMXRMS, &
                  RMXRMS,REFPX,REFPY,REFPZ, &
                  NATREP,ISTREP,ARMS, &
                  BRMS,DRMS,FRMS,RPMF,EPMF,WEIGHT,WTOT,QPRINT, &
                  QPNOTR,QPNORT,QPCYCL,ATOMPR, &
                  RA,RB,DRA,DRB,DRTEMP, &
                  EVWID,DRATIO,QPRPMF)
          ELSE                 ! QPROPT
             KRMSX=ZERO
             IF(QEPTHR) KRMSX=KRMS
             CALL EPATHR(EPTHR,DX,DY,DZ,X,Y,Z,NREPL,NREP,KRMSX, &
                  KMNRMS,RMNRMS,KMXRMS,RMXRMS, &
                  NATREP,ISTREP,ARMS,FRMS, &
                  WEIGHT,WTOT,QPRINT, &
                  QPNOTR,QPNORT,QPCYCL,ATOMPR, &
                  RA,RB,DRA,DRB,DRTEMP, &
                  PATANG,EVWID,QPCIMG,ICIMG,QPNEB, &
                  PSDX,PSDY,PSDZ)

             IF(QEPTHA) THEN
                CALL EPATHA(EPTHA,DX,DY,DZ,X,Y,Z,NREP,KANG,PCOSMX, &
                     NATREP,ISTREP,ARMS,FANG, &
                     ACST,WEIGHT,WTOT,QPRINT, &
                     QPNOTR,QPNORT,QPCYCL,ATOMPR, &
                     RA,RB,DRA,DRB, &
                     DRTEMP,EVWID,QPNEB,PJDX,PJDY,PJDZ)
             ENDIF
          ENDIF                ! QPROPT

       ELSE                    ! QPTAU

          KRMSX=KRMS

          CALL EPATAU(EPTHR,DX,DY,DZ,X,Y,Z,REFX,REFY,REFZ, &
               REFPX,REFPY,REFPZ,REFAX,REFAY,REFAZ,REFBX,REFBY,REFBZ, &
               NREPL,NREP,KRMSX,KMXRMS,RMXRMS,NATREP,ISTREP, &
               RPENR,ARMS,BRMS, &
               DRMS,FRMS,RPMF,EPMF,WEIGHT,WTOT, &
               QPRINT,QPNOTR,QPNORT,QPCYCL, &
               ATOMPR,RA,RB,DRA,DRB, &
               DRTEMP,PATANG,PATHDL,PATHDL1, &
               EVWID,REPEi,REPESQi,REPFi, &
               REPFSQi,REPELAi,REPFLAi, &
               TFCE,PFCE,NPCALL,QPPMF,QPRPMF,ICIMG)

       ENDIF                   ! QPTAU

       call chmdealloc('epath.src','EPATH','PFCE',NREPL,crl=PFCE)
       call chmdealloc('epath.src','EPATH','TFCE',NREPL,crl=TFCE)
       call chmdealloc('epath.src','EPATH','RPLENG',NREPL,crl=RPLENG)
       call chmdealloc('epath.src','EPATH','RPENR',NREPL,crl=RPENR)

       !-----------------------------------------------------------------------
       ! Process ordinary replica path restraints
    ELSE

       KRMSX=ZERO
       KMNRMSX=ZERO
       KMXRMSX=ZERO
       
       IF (MYNOD == 0 .or. QREPDSTR) THEN
          IF(QEPTHR) THEN
             KRMSX=KRMS
             KMNRMSX=KMNRMS
             KMXRMSX=KMXRMS
          ENDIF
          CALL EPATHR(EPTHR,DX,DY,DZ,X,Y,Z,nrepl,NREP,KRMSX, &
               KMNRMS,RMNRMS,KMXRMS,RMXRMS, &
               NATREP,ISTREP,ARMS,FRMS, &
               WEIGHT,WTOT,QPRINT, &
               QPNOTR,QPNORT,QPCYCL,ATOMPR, &
               RA,RB,DRA,DRB,DRTEMP, &
               PATANG,EVWID,QPCIMG,ICIMG,QPNEB,PSDX,PSDY,PSDZ)

          IF(QEPTHA) THEN
             CALL EPATHA(EPTHA,DX,DY,DZ,X,Y,Z,NREP,KANG,PCOSMX, &
                  NATREP,ISTREP,ARMS,FANG, &
                  ACST,WEIGHT,WTOT,QPRINT, &
                  QPNOTR,QPNORT,QPCYCL,ATOMPR, &
                  RA,RB,DRA,DRB,DRTEMP, &
                  EVWID,QPNEB,PJDX,PJDY,PJDZ)
          ENDIF

          IF(QPKINE.OR.QPLENG.OR.QPHP)THEN
             call chmalloc('epath.src','EPATH','XTAN',NREPL,NATREP,crl=XTAN)
             call chmalloc('epath.src','EPATH','YTAN',NREPL,NATREP,crl=YTAN)
             call chmalloc('epath.src','EPATH','ZTAN',NREPL,NATREP,crl=ZTAN)
             call chmalloc('epath.src','EPATH','XM1' ,NREPL,NATREP,crl=XM1)
             call chmalloc('epath.src','EPATH','YM1' ,NREPL,NATREP,crl=YM1)
             call chmalloc('epath.src','EPATH','ZM1' ,NREPL,NATREP,crl=ZM1)
             call chmalloc('epath.src','EPATH','XP1' ,NREPL,NATREP,crl=XP1)
             call chmalloc('epath.src','EPATH','YP1' ,NREPL,NATREP,crl=YP1)
             call chmalloc('epath.src','EPATH','ZP1' ,NREPL,NATREP,crl=ZP1)

             IF(QPKINE.OR.QPLENG)THEN
                CALL EPATH_KINE(QPKINE,QPLENG,EPKINE,EPLENG, &
                     KPKINE,KPLENG,PLFIX,QPTEMP, &
                     QEPTHR,QEPTHA,QISOKN,QWETHM,QPKNUDG, &
                     PTEMP,DX,DY,DZ,X,Y,Z,REFX,REFY,REFZ, &
                     NREPL,NREP,NATREP,ISTREP, &
                     ARMS,BRMS,FRMS,DRMS,PMF,PATHWORK, &
                     WEIGHT,WTOT,QPRINT,QPNOTR,QPNORT,QPCYCL, &
                     ATOMPR,RA,RB,EVWID,XTAN,YTAN,ZTAN, &
                     XM1,YM1,ZM1,XP1,YP1,ZP1,RPMF,EPMF)

                EPTHA=EPTHA+EPKINE
                EPTHR=EPTHR+EPLENG
             ENDIF

             IF(QPHP)THEN
                CALL EPATH_HYPE(QPHP,EPHP,KPHP,RPHP,QEPTHR,QEPTHA, &
                     DX,DY,DZ,X,Y,Z,NREPL,NREP,NATREP, &
                     ISTREP,ARMS,BRMS,WEIGHT,WTOT,QPRINT, &
                     QPNOTR,QPNORT,QPCYCL, &
                     ATOMPR,RA,RB,DRA,DRB,EVWID, &
                     XTAN,YTAN,ZTAN,XM1,YM1,ZM1,XP1,YP1,ZP1)
                EPTHR=EPTHR+EPHP
             ENDIF

             call chmdealloc('epath.src','EPATH','ZP1' ,NREPL,NATREP,crl=ZP1)
             call chmdealloc('epath.src','EPATH','YP1' ,NREPL,NATREP,crl=YP1)
             call chmdealloc('epath.src','EPATH','XP1' ,NREPL,NATREP,crl=XP1)
             call chmdealloc('epath.src','EPATH','ZM1' ,NREPL,NATREP,crl=ZM1)
             call chmdealloc('epath.src','EPATH','YM1' ,NREPL,NATREP,crl=YM1)
             call chmdealloc('epath.src','EPATH','XM1' ,NREPL,NATREP,crl=XM1)
             call chmdealloc('epath.src','EPATH','ZTAN',NREPL,NATREP,crl=ZTAN)
             call chmdealloc('epath.src','EPATH','YTAN',NREPL,NATREP,crl=YTAN)
             call chmdealloc('epath.src','EPATH','XTAN',NREPL,NATREP,crl=XTAN)
          ENDIF

          CALL EPATHS(NREP,ARMS,FRMS,FANG,ACST,QPCYCL,QPRINT)

       ENDIF
    ENDIF

    !
    !-----------------------------------------------------------------------
    IF (MYNOD == 0 .or. QREPDSTR) THEN
       CALL EPATHS(NREP,ARMS,FRMS,FANG,ACST,QPCYCL,QPRINT)
    ENDIF

    IF (QPTAU) THEN
       call chmdealloc('epath.src','EPATH','PATHDL1',3*NREPL*NATREP,crl=PATHDL1)
    ENDIF
    call chmdealloc('epath.src','EPATH','PATHDL',3*NREPL*NATREP,crl=PATHDL)
    call chmdealloc('epath.src','EPATH','PATANG',3*NREPL*NATREP,crl=PATANG)
    call chmdealloc('epath.src','EPATH','DRTEMP',3*NATREP,crl=DRTEMP)
    call chmdealloc('epath.src','EPATH','DRB',3,NATREP,crl=DRB)
    call chmdealloc('epath.src','EPATH','DRA',3,NATREP,crl=DRA)
    call chmdealloc('epath.src','EPATH','RB',3,NATREP,crl=RB)
    call chmdealloc('epath.src','EPATH','RA',3,NATREP,crl=RA)
    call chmdealloc('epath.src','EPATH','BFARY',3*NATREP,crl=BFARY)
    call chmdealloc('epath.src','EPATH','ATOMPR',3,NATREP,intg=ATOMPR)

100 CONTINUE

    call chmdealloc('epath.src','EPATH','WEIGHT',NATOM,crl=WEIGHT)
    call chmdealloc('epath.src','EPATH','ACST',NREPL,crl=ACST)
    call chmdealloc('epath.src','EPATH','FANG',NREPL,crl=FANG)
    call chmdealloc('epath.src','EPATH','EPMF',NREPL,crl=EPMF)
    call chmdealloc('epath.src','EPATH','RPMF',NREPL,crl=RPMF)
    call chmdealloc('epath.src','EPATH','FRMS',NREPL,crl=FRMS)
    call chmdealloc('epath.src','EPATH','DRMS',NREPL,crl=DRMS)
    call chmdealloc('epath.src','EPATH','BRMS',NREPL,crl=BRMS)
    call chmdealloc('epath.src','EPATH','ARMS',NREPL,crl=ARMS)
    call chmdealloc('epath.src','EPATH','ICNREP',NREPL,intg=ICNREP)
    call chmdealloc('epath.src','EPATH','ISTREP',NREPL,intg=ISTREP)

#endif /* (replica1)  IF REPLICA*/
    RETURN
  END SUBROUTINE EPATH

#if KEY_REPLICA==0 /*replica2*/
  SUBROUTINE PATHPS(COMLYN,COMLEN)
    INTEGER COMLEN
    CHARACTER(len=*) COMLYN
    CALL WRNDIE(-1,'<REPLICA>','REPLICA code not compiled.')
    RETURN
  END SUBROUTINE PATHPS
#else /* (replica2)*/

  SUBROUTINE PATHPS(COMLYN,COMLEN)
    !
    ! Parse the path options
    !
    use number
    use replica_mod
    use pathm
    use stream
    use string
    use memory
    use psf
    use coordc
    use coord    ! jwchu
    use cnst_fcm     ! jwchu
    use econtmod    ! jwchu
    use fast     ! jwchu
    use machdep  ! jwchu
    use parallel
    use repdstr

    INTEGER COMLEN
    CHARACTER(len=*) COMLYN
    !
    ! local
    LOGICAL QCOMP
    INTEGER K,I,J,IPMF
    real(chm_real) ETOT

    INTEGER NREP

    IF(INDXA(COMLYN,COMLEN,'OFF').GT.0) THEN
       QPATH=.FALSE.
       QPATHFLG=.TRUE.
       IF(.NOT.QPA_alloc) THEN
          call chmdealloc('epath.src','PATHPS','REPFLAi',NREPL,crl=REPFLAi)
          call chmdealloc('epath.src','PATHPS','REPELAi',NREPL,crl=REPELAi)
          call chmdealloc('epath.src','PATHPS','REPFSQi',NREPL,crl=REPFSQi)
          call chmdealloc('epath.src','PATHPS','REPFi',NREPL,crl=REPFi)
          call chmdealloc('epath.src','PATHPS','REPESQi',NREPL,crl=REPESQi)
          call chmdealloc('epath.src','PATHPS','REPEi',NREPL,crl=REPEi)

          call chmdealloc('epath.src','PATHPS','REFPDZ',NATOM,crl=REFPDZ)
          call chmdealloc('epath.src','PATHPS','REFPDY',NATOM,crl=REFPDY)
          call chmdealloc('epath.src','PATHPS','REFPDX',NATOM,crl=REFPDX)
          call chmdealloc('epath.src','PATHPS','REFPZ',NATOM,crl=REFPZ)
          call chmdealloc('epath.src','PATHPS','REFPY',NATOM,crl=REFPY)
          call chmdealloc('epath.src','PATHPS','REFPX',NATOM,crl=REFPX)
       ENDIF
       QPA_alloc=.TRUE.  ! jwchu
       NPCALL=0

#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          call chmdealloc('epath.src','PATHPS','XPEERS',NATOM,5,crl=XPEERS)
          call chmdealloc('epath.src','PATHPS','YPEERS',NATOM,5,crl=YPEERS)
          call chmdealloc('epath.src','PATHPS','ZPEERS',NATOM,5,crl=ZPEERS)
          call chmdealloc('epath.src','PATHPS','XPEERC',NATOM,5,crl=XPEERC)
          call chmdealloc('epath.src','PATHPS','YPEERC',NATOM,5,crl=YPEERC)
          call chmdealloc('epath.src','PATHPS','ZPEERC',NATOM,5,crl=ZPEERC)
          call chmdealloc('epath.src','PATHPS','APEERS',5,crl=APEERS)          
       ENDIF
#endif /*       */
       RETURN
    ENDIF

    ! If pathps is called from CHARMM via rpath command and its
    ! not to call rpath off, allocate arrays needed. Check and see if
    ! already allocated. cb3
    if(.not.allocated(PJDX)) call allocate_epath
    if(.not.allocated(xpref)) call allocate_path
    !
    !
    !
    IF ((.NOT.QREP) .AND. (.NOT.QREPDSTR)) THEN
          CALL WRNDIE(-1,'<PATHPS>', &
               'The REPLICA/PATH method needs replicas')
          RETURN
       ENDIF
       !
       !--------------------------------------------------------------------
       ! Analyze and print the current statistics (if any)
       IF(INDXA(COMLYN,COMLEN,'ANAL').GT.0) THEN

          IF(NPCALL.LE.0) THEN
             CALL WRNDIE(-1,'<PATHPS>','there is no data.')
          ENDIF

          IF(QPNEB) THEN
             CALL PATHSTAT(REPEi,REPESQi,REPFi, &
                  REPFSQi,REPELAi,REPFLAi,REPLENG, &
                  NPCALL,QPCYCL)
          ENDIF

          IF(QPROPT) THEN
             IPMF=GTRMI(COMLYN,COMLEN,'IPMF',-1)
             CALL PATHSTAT2(PMF,FLUC,NPCALL,QPCYCL,IPMF)
          ENDIF

          IF(INDXA(COMLYN,COMLEN,'RESE').GT.0) THEN
             PMF(1:NREPL) = ZERO
             FLUC(1:NREPL) = ZERO
             NPCALL=0
             REPEi(1:NREPL) = ZERO
             REPESQi(1:NREPL) = ZERO
             REPFi(1:NREPL) = ZERO
             REPFSQi(1:NREPL) = ZERO
             REPELAi(1:NREPL) = ZERO
             REPFLAi(1:NREPL) = ZERO
          ENDIF

          RETURN
       ENDIF

       ! Copy Path in COMP into PREF
       XPREF(1:NATOM) = XCOMP(1:NATOM)
       YPREF(1:NATOM) = YCOMP(1:NATOM)
       ZPREF(1:NATOM) = ZCOMP(1:NATOM)

       PMF(1:NREPL) = ZERO
       FLUC(1:NREPL) = ZERO
       !
       !--------------------------------------------------------------------
       ! Parse general terms
       !
       QPATH=.TRUE.
       !CC Set for NEB below, others don't need ECONT
       KRMS=GTRMF(COMLYN,COMLEN,'KRMS',ZERO)
       KMXRMS=GTRMF(COMLYN,COMLEN,'KMAX',ZERO)
       RMXRMS=GTRMF(COMLYN,COMLEN,'RMAX',ZERO)
       KMNRMS=GTRMF(COMLYN,COMLEN,'KMIN',ZERO)
       RMNRMS=GTRMF(COMLYN,COMLEN,'RMIN',ZERO)
       KANG=GTRMF(COMLYN,COMLEN,'KANG',ZERO)
       EVWID=GTRMF(COMLYN,COMLEN,'EVWI',ZERO)
       PCOSMX=GTRMF(COMLYN,COMLEN,'COSM',ONE)
       QPMASS=(INDXA(COMLYN,COMLEN,'MASS').GT.0)
       QPWEIG=(INDXA(COMLYN,COMLEN,'WEIG').GT.0)
       QPWCOM=(INDXA(COMLYN,COMLEN,'WCOM').GT.0)
#if KEY_COMP2==1
       QPWCM2=(INDXA(COMLYN,COMLEN,'WCM2').GT.0)
#endif 
       QPCYCL=(INDXA(COMLYN,COMLEN,'CYCL').GT.0)
       QPNORT=.NOT.(INDXA(COMLYN,COMLEN,'ROTA').GT.0)
       QPNOTR=.NOT.(INDXA(COMLYN,COMLEN,'TRAN').GT.0)
       QPNORT=(INDXA(COMLYN,COMLEN,'NORO').GT.0)
       QPNOTR=(INDXA(COMLYN,COMLEN,'NOTR').GT.0)

       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,25)    ' '
          WRITE(OUTU,25)    'The REPLICA/PATH method is initiated.'
       ENDIF
       !
       !--------------------------------------------------------------------
       ! Parse path kinetic energy and path length terms
       !
       QPKINE=(INDXA(COMLYN,COMLEN,'PKIN').GT.0)
       IF(QPKINE)THEN
          KPKINE=GTRMF(COMLYN,COMLEN,'KPKI',ONE)
          QISOKN=(INDXA(COMLYN,COMLEN,'ISOK').GT.0)
          QWETHM=(INDXA(COMLYN,COMLEN,'WETH').GT.0)
          IF(QISOKN.AND.QWETHM)THEN
             CALL WRNDIE(-2,'<EPATH>', &
                  'Isokinetic and work-energy theorem options cannot coexist')
          ENDIF
          QPTEMP=(INDXA(COMLYN,COMLEN,'PTEM').GT.0)
          IF(QPTEMP)PTEMP=GTRMF(COMLYN,COMLEN,'TEMP',ZERO)

          QPKNUDG=(INDXA(COMLYN,COMLEN,'PKNU').GT.0)
       ENDIF

       QPLENG=(INDXA(COMLYN,COMLEN,'PLEN').GT.0)
       IF(QPLENG)THEN
          PLFIX =GTRMF(COMLYN,COMLEN,'LFIX',ZERO)
          KPLENG=GTRMF(COMLYN,COMLEN,'KPLE',HUNDRD)
       ENDIF
       !
       !--------------------------------------------------------------------
       ! Parse hyperplane restraint
       QPHP=(INDXA(COMLYN,COMLEN,'HYPE').GT.0)
       IF(QPHP)THEN
          NREP=NREPL-1
          IF(NREP.NE.3)CALL WRNDIE(-2,'<PATHPS>', &
               'Exactly three replicas are required for hype')
          KPHP=GTRMF(COMLYN,COMLEN,'KPHP',ZERO)
          RPHP=GTRMF(COMLYN,COMLEN,'RPHP',ZERO)

          if (prnlev >= 2) then
             WRITE(OUTU,25)'Hyperplane restraint will be used'
             WRITE(OUTU,26)' KPHP= ',KPHP
             WRITE(OUTU,26)' RPHP= ',RPHP
          endif
       ENDIF

       !
       !--------------------------------------------------------------------
       ! Process cartesian nudged elastic band parse and init
       QNEB=(INDXA(COMLYN,COMLEN,'NEBA').GT.0)
       IF(QNEB) THEN
          KNEB=GTRMF(COMLYN,COMLEN,'KNEB',ZERO)
          IF(PRNLEV.GE.2) THEN
             WRITE(OUTU,25) '    Cartesian NEB optimization will be done.'
             WRITE(OUTU,26)  '    KNEB   =',KNEB
          ENDIF
       ENDIF
       !--------------------------------------------------------------------
       ! Process rms bestfit nudged elastic band parse and init
       QPNEB=(INDXA(COMLYN,COMLEN,'NEBF').GT.0)    ! jwchu
       QPTAU=(INDXA(COMLYN,COMLEN,'PTAU').GT.0)    ! jwchu
       IF(QPNEB.OR.QPTAU) THEN
          QPETAN=(INDXA(COMLYN,COMLEN,'ETAN').GT.0)   ! jwchu
          QPCIMG=(INDXA(COMLYN,COMLEN,'CIMG').GT.0)   ! jwchu
          QRFIX=(INDXA(COMLYN,COMLEN,'RFIX').GT.0)    ! jwchu
          QANAL=(INDXA(COMLYN,COMLEN,'ANAL').GT.0)    ! jwchu
          QPROPT1=(INDXA(COMLYN,COMLEN,'OPTI').GT.0)  ! jwchu
          DRATIO=GTRMF(COMLYN,COMLEN,'DRAT',HALF)     ! jwchu
          QCOMP=(INDXA(COMLYN,COMLEN,'COMP').GT.0)    ! jwchu
          QPPMF=(INDXA(COMLYN,COMLEN,'PPMF').GT.0)    ! jwchu
          QPRPMF=(INDXA(COMLYN,COMLEN,'PRPM').GT.0)   ! jwchu
          FASTER=-1
          LFAST =-1
          QECONT=.TRUE.
          QFASTNB = .FALSE.
          ! Setting qecont true here is equivalent to setting up
          ! energy analysis, need to make sure econt_ltm arrays are
          ! allocated. cb3
          if(.not.allocated(anslct)) call allocate_econt
       ENDIF

       IF(PRNLEV.GE.2) THEN
          IF(QPNEB.OR.QPTAU) THEN
             WRITE(OUTU,'(A,/,A)') &
                  ' PATHPS> Energy partition analysis will performed.', &
                  '         Fast routines disabled.'
             IF(QPNEB)  WRITE(OUTU,'(A)') &
                  '         NEB will be used.'
             IF(QPETAN) WRITE(OUTU,'(A,/,A)')  &
                  '         Energy based tangent estimation of path.', &
                  '         Apply restrain force affects the energy.'
             IF(QPCIMG) WRITE(OUTU,'(A,/,A)') &
                  '         Climbing Image method will be used.', &
                  '         Only the highest energy image will climb.'
             IF(QRFIX)  WRITE(OUTU,'(A)') &
                  '         Path statistics will be done w.r.t a reference path.'
             IF(QCOMP) THEN
                WRITE(OUTU,'(A)') &
                     '         ref coords will be read from comp set.'
             ELSE
                WRITE(OUTU,'(A)') &
                     '         ref coords will be read from main set.'
             ENDIF
             IF(QPROPT) WRITE(OUTU,'(A)') &
                  '         Reference Path restraint turned on.'
             IF(QPROPT) WRITE(OUTU,'(A,F14.6)') &
                  '         DRAT   =',DRATIO
             IF(QPPMF)  WRITE(OUTU,'(A,A)') &
                  '         ENERGY and PMF of each replica will ', &
                  '  be printed at each rpat hcall.'
             IF(QPTAU)  WRITE(OUTU,'(A)') &
                  '         Sampling the path in the off-path dir is on.'
          ENDIF
       ENDIF

       IF(QPROPT) THEN
          IF(QPROPT1.AND.QPNEB) THEN
             CALL WRNDIE(-2,'<EPATH>', &
                  'NEB and resd path optimization connot coexist')
             RETURN
          ENDIF
          IF(DRATIO.GT.ONE.OR.DRATIO.LT.ZERO) THEN
             CALL WRNDIE(-2,'<EPATH>', &
                  'DRATIO has to be in between 0 and 1')
             RETURN
          ENDIF
       ENDIF

       IF(QPTAU.AND.QPNEB) THEN
          CALL WRNDIE(-2,'<EPATH>', &
               'NEB and pathtau connot coexist')
          RETURN
       ENDIF
       !--------------------------------------------------------------------
       ! Process reference path optimization parse and init
       !
       QPROPT=(INDXA(COMLYN,COMLEN,'OPTI').GT.0)
       QCURVC=(INDXA(COMLYN,COMLEN,'CURVC').GT.0)
       QNOCURVC=(INDXA(COMLYN,COMLEN,'NOCURV').GT.0)

       IF(PRNLEV.GE.2) THEN
          IF(QPROPT) THEN
             WRITE(OUTU,25) '    Reference Path optimization will be done.'
          ENDIF
          IF(QCURVC) THEN
             WRITE(OUTU,25) '    Curvature Correction will be performed.'
          ENDIF
       ENDIF
       !
       !--------------------------------------------------------------------
       IF(QPROPT1 .OR. QPNEB .OR. QPTAU) THEN
          IF(QPA_alloc) THEN
             call chmalloc('epath.src','PATHPS','REFPX',NATOM,crl=REFPX)
             call chmalloc('epath.src','PATHPS','REFPY',NATOM,crl=REFPY)
             call chmalloc('epath.src','PATHPS','REFPZ',NATOM,crl=REFPZ)
             call chmalloc('epath.src','PATHPS','REFPDX',NATOM,crl=REFPDX)
             call chmalloc('epath.src','PATHPS','REFPDY',NATOM,crl=REFPDY)
             call chmalloc('epath.src','PATHPS','REFPDZ',NATOM,crl=REFPDZ)

             call chmalloc('epath.src','PATHPS','REPEi',NREPL,crl=REPEi)
             call chmalloc('epath.src','PATHPS','REPESQi',NREPL,crl=REPESQi)
             call chmalloc('epath.src','PATHPS','REPFi',NREPL,crl=REPFi)
             call chmalloc('epath.src','PATHPS','REPFSQi',NREPL,crl=REPFSQi)
             call chmalloc('epath.src','PATHPS','REPELAi',NREPL,crl=REPELAi)
             call chmalloc('epath.src','PATHPS','REPFLAi',NREPL,crl=REPFLAi)
             call chmalloc('epath.src','PATHPS','REPLENG',NREPL,crl=REPLENG)

             IF(QCOMP) THEN
                REFPX(1:NATOM) = XCOMP(1:NATOM)
                REFPY(1:NATOM) = YCOMP(1:NATOM)
                REFPZ(1:NATOM) = ZCOMP(1:NATOM)
             ELSE
                REFPX(1:NATOM) = X(1:NATOM)
                REFPY(1:NATOM) = Y(1:NATOM)
                REFPZ(1:NATOM) = Z(1:NATOM)
             ENDIF
          ENDIF
          IF(QPTAU) THEN
             call chmalloc('epath.src','PATHPS','REFAX',NATOM,crl=REFAX)
             call chmalloc('epath.src','PATHPS','REFAY',NATOM,crl=REFAY)
             call chmalloc('epath.src','PATHPS','REFAZ',NATOM,crl=REFAZ)
             call chmalloc('epath.src','PATHPS','REFBX',NATOM,crl=REFBX)
             call chmalloc('epath.src','PATHPS','REFBY',NATOM,crl=REFBY)
             call chmalloc('epath.src','PATHPS','REFBZ',NATOM,crl=REFBZ)

             !  in QPTAU, the main coord set always corresponds to the active coords
             REFPX(1:NATOM) = X(1:NATOM)
             REFPY(1:NATOM) = Y(1:NATOM)
             REFPZ(1:NATOM) = Z(1:NATOM)
             !  in QPTAU, the comp coord set always corresponds the reference A
             REFAX(1:NATOM) = XCOMP(1:NATOM)
             REFAY(1:NATOM) = YCOMP(1:NATOM)
             REFAZ(1:NATOM) = ZCOMP(1:NATOM)
             !  in QPTAU, the ref  coord set always corresponds the reference B
             REFBX(1:NATOM) = REFX(1:NATOM)
             REFBY(1:NATOM) = REFY(1:NATOM)
             REFBZ(1:NATOM) = REFZ(1:NATOM)
          ENDIF                  ! QPTAU
          QPA_alloc=.FALSE.

          IF(QANAL) THEN
             IF(NPCALL.LE.0) THEN
                CALL WRNDIE(-1,'<PATHPS>','there is no data.')
             ENDIF
             CALL PATHSTAT(REPEi,REPESQi,REPFi, &
                  REPFSQi,REPELAi,REPFLAi, &
                  REPLENG,NPCALL,QPCYCL)

             IF(QPTAU) THEN
                call chmdealloc('epath.src','PATHPS','REFBZ',NATOM,crl=REFBZ)
                call chmdealloc('epath.src','PATHPS','REFBY',NATOM,crl=REFBY)
                call chmdealloc('epath.src','PATHPS','REFBX',NATOM,crl=REFBX)
                call chmdealloc('epath.src','PATHPS','REFAZ',NATOM,crl=REFAZ)
                call chmdealloc('epath.src','PATHPS','REFAY',NATOM,crl=REFAY)
                call chmdealloc('epath.src','PATHPS','REFAX',NATOM,crl=REFAX)
             ENDIF               ! QPTAU

             call chmdealloc('epath.src','PATHPS','REPLENG',NREPL,crl=REPLENG)
             call chmdealloc('epath.src','PATHPS','REPFLAi',NREPL,crl=REPFLAi)
             call chmdealloc('epath.src','PATHPS','REPELAi',NREPL,crl=REPELAi)
             call chmdealloc('epath.src','PATHPS','REPFSQi',NREPL,crl=REPFSQi)
             call chmdealloc('epath.src','PATHPS','REPFi',NREPL,crl=REPFi)
             call chmdealloc('epath.src','PATHPS','REPESQi',NREPL,crl=REPESQi)
             call chmdealloc('epath.src','PATHPS','REPEi',NREPL,crl=REPEi)

             call chmdealloc('epath.src','PATHPS','REFPDZ',NATOM,crl=REFPDZ)
             call chmdealloc('epath.src','PATHPS','REFPDY',NATOM,crl=REFPDY)
             call chmdealloc('epath.src','PATHPS','REFPDX',NATOM,crl=REFPDX)
             call chmdealloc('epath.src','PATHPS','REFPZ',NATOM,crl=REFPZ)
             call chmdealloc('epath.src','PATHPS','REFPY',NATOM,crl=REFPY)
             call chmdealloc('epath.src','PATHPS','REFPX',NATOM,crl=REFPX)

             QPA_alloc=.TRUE.
             NPCALL=0
             RETURN
          ENDIF

       ENDIF
       !--------------------------------------------------------------------
       !
       !     Allocate the space here for the neighboring sets of coordinates
       !     Allocate for previous and next.
       !     This is needed only when QREPDSTR is active
       !
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          call chmalloc('epath.src','PATHPS','XPEERS',NATOM,5,crl=XPEERS)
          call chmalloc('epath.src','PATHPS','YPEERS',NATOM,5,crl=YPEERS)
          call chmalloc('epath.src','PATHPS','ZPEERS',NATOM,5,crl=ZPEERS)
          call chmalloc('epath.src','PATHPS','XPEERC',NATOM,5,crl=XPEERC)
          call chmalloc('epath.src','PATHPS','YPEERC',NATOM,5,crl=YPEERC)
          call chmalloc('epath.src','PATHPS','ZPEERC',NATOM,5,crl=ZPEERC)
          call chmalloc('epath.src','PATHPS','APEERS',5,crl=APEERS)
       ENDIF
#endif 
       !
       !--------------------------------------------------------------------
       ! Print out the results of parsing
       !
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,26)    '    KRMS   =',KRMS
          IF (KMNRMS.GT.ZERO) THEN
             WRITE(OUTU,26) '    KMINrms=',KMNRMS
             WRITE(OUTU,26) '    RMINrms=',RMNRMS
          ENDIF
          IF (KMXRMS.GT.ZERO) THEN
             WRITE(OUTU,26) '    KMAXrms=',KMXRMS
             WRITE(OUTU,26) '    RMAXrms=',RMXRMS
          ENDIF
          WRITE(OUTU,26)    '    KANGle =',KANG
          WRITE(OUTU,26)    '    COSMax =',PCOSMX
          WRITE(OUTU,26)    '    EVWIdth=',EVWID
          IF(QPMASS) THEN
             WRITE(OUTU,25) '    Mass weighting will be used.'
          ELSE
             WRITE(OUTU,25) '    Mass weighting will not be used.'
          ENDIF
          IF(QPWEIG) THEN
             WRITE(OUTU,25) '    The weighting array will be used.'
          ELSE
             WRITE(OUTU,25) '    The weighting array will not be used.'
          ENDIF
          IF(QPNORT) THEN
             WRITE(OUTU,25) '    No rotational best fit will be used.'
          ELSE
             WRITE(OUTU,25) '    A rotational best fit will be used.'
          ENDIF
          IF(QPNOTR) THEN
             WRITE(OUTU,25) '    No translational best fit will be used.'
          ELSE
             WRITE(OUTU,25) '    A translational best fit will be used.'
          ENDIF
          IF(QPCYCL) THEN
             WRITE(OUTU,25) '    A cyclic path will be generated.'
          ELSE
             WRITE(OUTU,25) '    A non-cyclic path will be generated.'
          ENDIF
          WRITE(OUTU,25)    ' '

          IF(QPKINE)THEN
             WRITE(OUTU,25) '    Kinetic energy potential will be added'

             IF(QPTEMP)THEN
                WRITE(OUTU,25) '    Temperature will be used to determine '
                WRITE(OUTU,25) '    the force constants '
                WRITE(OUTU,26) '    the temperature is: ',PTEMP
             ELSE
                WRITE(OUTU,26) '    KPKINE =',KPKINE
             ENDIF

             IF(QISOKN)THEN
                WRITE(OUTU,25) '    Kinetic energy will be maintained '
                WRITE(OUTU,25) '    at contant values along the path '
             ENDIF

             IF(QWETHM)THEN
                WRITE(OUTU,25)'    Work-Energy Theorem will be used to '
                WRITE(OUTU,25)'    determine the kinetic energies '
             ENDIF

             IF(QPKNUDG)THEN
                WRITE(OUTU,25)'    Kinetic energy forces will be nudged '
             ENDIF
          ENDIF !QPKINE

          WRITE(OUTU,25) '                                       '

          IF(QPLENG)THEN
             WRITE(OUTU,25)'    Path length potential will be added'
             WRITE(OUTU,26)'    KPLENG =',KPLENG
             IF(PLFIX.GT.RSMALL)THEN
                WRITE(OUTU,26)'    Path length will be restrained at: ', &
                     PLFIX
             ELSE
                CALL WRNDIE(-2,'<EPATHPS>', &
                     'targeted path length negative')
             ENDIF
          ENDIF
       ENDIF
25     FORMAT(A)
26     FORMAT(A,F14.6)
       
       RETURN
  END SUBROUTINE PATHPS

  SUBROUTINE PATHSTAT(REPEi,REPESQi,REPFi,REPFSQi, &
       REPELAi,REPFLAi,REPLENG,NPCALL,QPCYCL)
    !
    ! output path statictics
    !
    use number
    use replica_mod
    use stream
    use parallel

    real(chm_real) REPEi(*),REPESQi(*),REPFi(*),TMP
    real(chm_real) REPFSQi(*),REPELAi(*),REPFLAi(*),REPLENG(*)
    INTEGER I,NREP,NPCALL
    LOGICAL QPCYCL

    NREP=NREPL-1
    IF(QPCYCL) NREP=NREPL

    TMP=ONE/DBLE(NPCALL)

#if KEY_PARALLEL==1 /*pll*/
    IF(MYNOD.EQ.0)THEN
       WRITE(OUTU,102) NPCALL
102    FORMAT(' number of calls to epath: ',I5)
       WRITE(OUTU,101)
101    FORMAT(' I=','          AVG Energy','       AVG Energy SQ', &
            '         LAST ENERGY','        AVG PMF', &
            '     AVG PMF SQ','      LAST PMF ')
       DO I=1,NREP
          WRITE(OUTU,100) I,REPEi(I)*TMP,REPESQi(I)*TMP,REPELAi(I), &
               REPFi(I)*TMP,REPFSQi(I)*TMP,REPFLAi(I)
       ENDDO
100    FORMAT(I3,1X,3F20.6,3F15.6)
       WRITE(OUTU,112)
112    FORMAT(' I=',' off-path force','  tangent force', &
            '    Real-space length')
       DO I=1,NREP
          WRITE(OUTU,103) I,REPLENG(I)
       ENDDO
103    FORMAT(I3,1X,1F15.6)

    ENDIF
#else /* (pll)*/
    WRITE(OUTU,102) NPCALL
102 FORMAT(' number of calls to epath: ',I5)
    WRITE(OUTU,101)
101 FORMAT(' I=','          AVG Energy','       AVG Energy SQ', &
         '         LAST ENERGY','        AVG PMF', &
         '     AVG PMF SQ','      LAST PMF ')
    DO I=1,NREP
       WRITE(OUTU,100) I,REPEi(I)*TMP,REPESQi(I)*TMP,REPELAi(I), &
            REPFi(I)*TMP,REPFSQi(I)*TMP,REPFLAi(I)
    ENDDO
100 FORMAT(I3,1X,3F20.6,3F15.6)
    WRITE(OUTU,112)
112 FORMAT(' I=',' off-path force','  tangent force', &
         '    Real-space length')
    DO I=1,NREP
       WRITE(OUTU,103) I,REPLENG(I)
    ENDDO
103 FORMAT(I3,1X,1F15.6)

#endif /* (pll)*/

    RETURN
  END SUBROUTINE PATHSTAT

  SUBROUTINE PATHSTAT2(PMF,FLUC,NPCALL,QPCYCL,IPMF)
    !
    ! output path statictics for EPATHO
    !
#if KEY_REPDSTR==1
    use repdstr, only:psetloc,psetglob       
#endif
    use number
    use replica_mod
    use stream
    use gamess_fcm
    use parallel
    use repdstr

    real(chm_real) PMF(*),FLUC(*)
    INTEGER NPCALL,IPMF
    LOGICAL QPCYCL

    INTEGER I,NREP
    INTEGER NREPM                                            ! lw050714
    real(chm_real) ETOT,WORKTOT

    NREPM=NREPL-1
    !hlw_080705 IF(QPCYCL) NREPM=NREPL
    IF(QPCYCL) NREPM=NREPL-2

    IF(IPMF.GT.0) open(IPMF,file='pmf.dat',status='unknown')
#if KEY_REPDSTR==1
    IF(QREPDSTR)THEN
       !C         write(50+irepdstr,'(a,2e20.10,3f15.5)')'pmf0=',(pmf(i),i=1,5)
       CALL PSETGLOB
       CALL GCOMB(FLUC,NREPM)
       CALL GCOMB(PMF,NREPM)
       CALL PSETLOC
    ENDIF
    !C      write(50+irepdstr,'(a,2e20.10,3f15.5)')'pmf=',(pmf(i),i=1,5)
#endif 
    ETOT=ZERO
    WORKTOT=ZERO
    DO I=1,NREPM

       PMF(I) = PMF(I) / NPCALL
       FLUC(I) = (FLUC(I) / NPCALL) - (PMF(I) * PMF(I))

       IF(FLUC(I).GT.ZERO) THEN
          FLUC(I) = SQRT(FLUC(I))
       ELSE
          FLUC(I)=ZERO
       ENDIF

       ETOT = ETOT + PMF(I)*HALF

       IF(PRNLEV.GT.2)WRITE(OUTU, 12) I,PMF(I),           & ! hlw_080705
            FLUC(I),WORKTOT, &
            ETOT
       IF(IPMF.GT.0) THEN
          WRITE(IPMF,11)WORKTOT
          WRITE(IPMF,11)ETOT
       ENDIF

       WORKTOT = WORKTOT + PMF(I)
       ETOT = ETOT + PMF(I)*HALF
    ENDDO

    IF(IPMF.GT.0) CLOSE(IPMF)

    RETURN

10  FORMAT('Step',I3,'  Work :',E20.6,'  Fluc:',E20.6,'  WorkTOT :', &
         E20.6, ' ETOT: ',E20.6)
12  FORMAT('Step',I3,'  Work :',F12.6,'  Fluc:',F12.6,'  WorkTOT :', &
         F12.6, ' ETOT: ',F12.6)
11  FORMAT(F12.6)

    RETURN
  END SUBROUTINE PATHSTAT2

  SUBROUTINE EPATHC(ISTREP,ICNREP,WEIGHT,NATREP, &
       WTOT,QPWEIG,QPWCOM, &
#if KEY_COMP2==1
       QPWCM2,  & 
#endif
       QPMASS,QPRINT,ERR)
    !
    ! Check the replica and path arrays.  Generate pointer and
    ! weight arrays.       BRB - 3/25/94
    !
    use number
    use coord
    use coordc
    use replica_mod
    use neb ! jwchuneb
    use psf
    use stream
    use parallel
    use repdstr

    INTEGER ISTREP(:),ICNREP(:),NATREP
    real(chm_real) WEIGHT(:),WTOT
    LOGICAL QPWEIG,QPMASS,QPRINT,ERR
    LOGICAL QPWCOM
#if KEY_COMP2==1
    LOGICAL QPWCM2
#endif 

    INTEGER I,J,LAST,ILAST,NREPM,INOMV,K,IPT
    LOGICAL QNEGW

    DO I=1,NREPL
       ISTREP(I)=-1
       ICNREP(I)=0
    ENDDO
    !
    ! Check to see if the replicas are normal:
    !                 all have same number of atoms
    !                 all are contiguous
    !
    ERR=.TRUE.
    LAST=0
    ILAST=0
    NATREP=0
    !
    !     Check for distributed replica
    !
#if KEY_REPDSTR==1
    IF(.NOT.QREPDSTR) THEN                  
#endif
       DO I=1,NATOM
          J=REPNOA(I)-1
          IF(J.GT.0) THEN
             IF(ICNREP(J).EQ.0) THEN
                ISTREP(J)=I-1
             ELSE
                IF(J+1.NE.REPNOA(I-1)) THEN
                   CALL WRNDIE(-2,'<EPATH>','Replica are not contiguous')
                   RETURN
                ENDIF
             ENDIF
             ICNREP(J)=ICNREP(J)+1
          ENDIF
       ENDDO

       NREPM=NREPL-1

       IF(QPRINT) WRITE(OUTU,57) NREPM
57     FORMAT('  in EPATHC: Number of replicas in use is:',I5)

       NATREP=ICNREP(1)
       DO I=1,NREPM
          IF(ISTREP(I).EQ.-1) THEN
             CALL WRNDIE(-2,'<EPATH>','Some replica has no atoms')
             RETURN
          ENDIF
          IF(ICNREP(I).NE.NATREP) THEN
             CALL WRNDIE(-2,'<EPATH>', &
                  'All replicas must have same number of atoms')
             RETURN
          ENDIF
       ENDDO
#if KEY_REPDSTR==1
    ENDIF                                 
#endif

#if KEY_REPDSTR==1
    !     For distributed replica
    !     NOTE: we don't have selection on repd command yet
    !
    IF(QREPDSTR) THEN
       NATREP=NATOM
       IF(NATREPCMD.NE.-1)NATREP=NATREPCMD
       NREPL=NREPDSTR
       NREPM=NREPL-1
       DO I=1,NREPL
          ISTREP(I)=0
          ICNREP(I)=NATOM  ! ?? not sure
       ENDDO
    ENDIF
#endif 

    IF(QPRINT) WRITE(OUTU,54) NATREP
54  FORMAT('  In EPATHC: Number of replicated atoms is:',I5)

    !CC fill the NEB block CCCCCCCCCC
    DO I=1,100
       IREPS(I)=0
       IPVAR(I)=0
       NPVAR(I)=0
    ENDDO

    IF(NREPL.GT.1000)CALL WRNDIE(-5,'<EPATHC>','Too many replicas')

    DO I=1,NREPL
       IREPS(I)=ISTREP(I)
    ENDDO

    NPATOMS=NATREP

    DO I=1,NREPM  ! changed from NREPL
       INOMV=0
       DO J=1,ISTREP(I)
          IF(IMOVE(J).NE.0) INOMV=INOMV+1
       ENDDO
       IPVAR(I)=IREPS(I)-INOMV
       NPVAR(I)=NPATOMS
       ! FIXME we access IMOVE(0) when I == NREPL
       DO J=1,NPATOMS
          IPT=ISTREP(I)+J
          IF(IMOVE(IPT).NE.0) NPVAR(I)=NPVAR(I)-1
       ENDDO
    ENDDO
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !
    ! Fill the weight array
    WTOT=ZERO
    QNEGW=.FALSE.
    DO I=1,NATREP

       !C      write(6,55) I,istrep(i)
       !C  55  format('  in EPATHC: starting pointer is:',2i5)

       J=I+ISTREP(1)
       WEIGHT(I)=ONE
       IF(QPMASS) WEIGHT(I)=AMASS(J)
       IF(QPWEIG) THEN
          IF(QPWCOM)THEN
             WEIGHT(I)=WEIGHT(I)*WCOMP(J)
#if KEY_COMP2==1
          ELSEIF(QPWCM2)THEN
             WEIGHT(I)=WEIGHT(I)*WCOMP2(J)
#endif 
          ELSE
             WEIGHT(I)=WEIGHT(I)*WMAIN(J)
          ENDIF
       ENDIF
       WTOT=WTOT+WEIGHT(I)
       IF(WEIGHT(I).LT.ZERO) QNEGW=.TRUE.
    ENDDO

    IF(QPRINT) WRITE(OUTU,56) WTOT
56  FORMAT('  In EPATHC: The total weight is:',F14.5)

    IF(QNEGW) THEN
       CALL WRNDIE(-2,'<EPATH>', &
            'Some atom(s) have a negative weight')
    ENDIF
    IF(WTOT.LE.ZERO) THEN
       CALL WRNDIE(-2,'<EPATH>', &
            'Path weighting array has a nonpositive sum')
    ENDIF

    ERR=.FALSE.
    RETURN
  END SUBROUTINE EPATHC

  SUBROUTINE EPATHR(EPTHR,DX,DY,DZ,X,Y,Z,NREPL,NREP,KRMS, &
       KMNRMS,RMNRMS,KMXRMS,RMXRMS, &
       NATREP,ISTREP,ARMS,FRMS,WEIGHT,WTOT,QPRINT, &
       LNOTRN,LNOROT,LCYCL,ATOMPR,RA,RB,DRA,DRB,DRTEMP, &
       PATANG,EVWID,QPCIMG,ICIMG,QPNEB, &
       PSDX,PSDY,PSDZ)
    !
    !  Calculate the rms path restraint energy. - BRB  3/25/94
    !
    ! Erms = sum  {  0.5* Krms * ( rms  - <rms> )**2 }      I=1,NREP-1
    !           I                     I
    !
    !    where rms  is the weighted rms deviation between replica I and I+1
    !             I
    !
    ! Erms = sum  { 0.5* Kmaxrms * ( rms  - Rmaxrms )**2 }  (rms>Rmaxrms)
    !           I                        I
    !
    ! Erms = sum { 0.5* Kminrms * (Rminrms * ( Rminrms/rms - 1))**2 } (rms<Rminrms)
    !           I                                         I
    !
    ! If QPNEB (NEB with best fit)
    ! the spring force is added to PSDX, PSDY, PSDZ
    !
#if KEY_REPDSTR==1
    use repdstr                            
#endif

    use number
    use stream
    use parallel
    use repdstr
    use memory
    use psf  ! for writing NATOM

    real(chm_real) EPTHR,KRMS,KMNRMS,RMNRMS,KMXRMS,RMXRMS
    real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
    real(chm_real) PSDX(*),PSDY(*),PSDZ(*)
    INTEGER NREPL,NREP,NATREP,ISTREP(*)
    real(chm_real) ARMS(*),FRMS(*),WEIGHT(*),WTOT
    LOGICAL QPRINT,LNOTRN,LNOROT,LCYCL,QPCIMG,QPNEB,QRPATH
    INTEGER ATOMPR(2,NATREP)
    real(chm_real) RA(3,*),RB(3,*),DRA(3,*),DRB(3,*),DRTEMP(*)
    real(chm_real) PATANG(3,NREPL,NATREP)  ! jwchu
    real(chm_real) EVWID

    INTEGER NREPM,LPEER,RPEER,ITYPE,LPEER2,RPEER2,KIDX
    INTEGER IREP,JREP,KREP,I,J,IPT,JPT,ICIMG
    real(chm_real) RMST,ATMP,R2,RNRM,ERMS,TMP,AX,AY,AZ,DERMS
    LOGICAL LPRINT,QFORCA,QFORCB
    real(chm_real) DEVA(3,3)
    REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:) :: DXT,DYT,DZT

    IF(LCYCL) THEN
       NREPM=NREP
    ELSE
       NREPM=NREP-1
    ENDIF

    ! we should get rid of this, but now needed for lcycl
#if KEY_REPDSTR==1
    !!!if(qrepdstr)nrepm=nrep-1 
#endif

    LPRINT = QPRINT .AND. (PRNLEV.GT.7)

    RMST=ZERO

    QRPATH=.TRUE.
    IF(QPNEB)QRPATH=.FALSE.

#if KEY_REPDSTR==1
    IF(.NOT.QREPDSTR) THEN                      
#endif
       DO JREP=1,NREPM
          KREP=JREP+1
          IF(KREP.GT.NREP) KREP=KREP-NREP

          IPT=ISTREP(JREP)
          JPT=ISTREP(KREP)
          DO I=1,NATREP
             IPT=IPT+1
             JPT=JPT+1
             ATOMPR(1,I)=IPT
             ATOMPR(2,I)=JPT
          ENDDO

          CALL ECBSTF(X,Y,Z,X,Y,Z,ATOMPR,NATREP, &
               LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,ATMP,RA,RB,DEVA,EVWID)

          ATMP=ATMP/WTOT
          IF(ATMP.LT.ZERO) ATMP=ZERO
          ATMP=SQRT(ATMP)
          ARMS(JREP)=ATMP
          RMST=RMST+ATMP
       ENDDO
       IF(PRNLEV.GE.7) THEN
          !c         write(OUTU,*)'EPATHR-4>nrepm,atmp=',nrepm,(arms(i),i=1,NREPM)
       ENDIF
#if KEY_REPDSTR==1
    ENDIF                                      
#endif
    !
    !  this is for forces! OLD stuff commented out for now!
!!!    !
!!!    !     For distributed replica we need to get the coordinates
!!!    !
!!!    !     NOTE 1:
!!!    !     NATREP usually equals NATOM. In case of different number
!!!    !     of atoms on each replica we need to specify NATREP
!!!    !     on the command line. Currently this works only for systems
!!!    !     which start with the same atoms from 1,NATREP. For example
!!!    !     protein can be solvated by different number of water molecules.
!!!    !     In the PSF protein must be the first and then the rest of
!!!    !     the system. syntax: REPDstr NREP <int> NATRep <int> ...
!!!    !     default NATRep = -1 ?? With this we can see if it was
!!!    !     specified or not, eg:
!!!    !     natrep=natom
!!!    !     if(natrepcmd.ne.-1)natrep=natrepcmd
!!!    !     the rest of the program stays as is!
!!!    !
!!!    !     NOTE 2: IREPDSTR starts from 0!
!!!    !
!!!    !     EXTEND 1: Generalize this code for parallel/parallel
!!!    !               cases. If everything is in IREPDSTR,NREPDSTR
!!!    !               then the code is already OK. The communication
!!!    !               has to be modified by:
!!!    !               IREPDSTR*NUMNOD,NREPDSTR*NUMNOD from
!!!    !               IREPDSTR,NREPDSTR (LPEER,RPEER)
!!!    !               check the above!!!
!!!    !
!!!    IF(QREPDSTR) THEN
!!!       LPEER=IREPDSTR-1
!!!       RPEER=IREPDSTR+1
!!!       JREP=IREPDSTR+1
!!!       IF(LPEER.LT.0)LPEER=NREPM
!!!       IF(RPEER.GT.NREPM)RPEER=0
!!!       !
!!!       !     Swap the data between the neighbors. This is a double
!!!       !     direction swap since some machines can do it as fast as
!!!       !     a single one. And we might need the other neighbor for angles
!!!       !     Currently for angles there is a problem, because we need I, I+1, I+2.
!!!       !     Maybe we can shift it so it would give the same result also for:
!!!       !     I-1, I, I+1
!!!       !     Or maybe we call this routine once again with exchanged parameters:
!!!       !     previous XPR becomes XPL.... Or write another routine, which goes in
!!!       !     one direction only (maybe GRECSENR() ???)
!!!       !
!!!       CALL SWAPD(NATREP,X,Y,Z,IXPL,IYPL,IZPL, &
!!!            IXPR,IYPR,IZPR,LPEER,RPEER)
!!!
!!!       IPT=0
!!!       JPT=0
!!!       DO I=1,NATREP
!!!          IPT=IPT+1
!!!          JPT=JPT+1
!!!          ATOMPR(1,I)=IPT
!!!          ATOMPR(2,I)=JPT
!!!       ENDDO
!!!
!!!       CALL ECBSTF(X,Y,Z,IXPR,IYPR,IZPR, &
!!!            ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT,0,.FALSE.,WEIGHT, &
!!!            WTOT,ATMP,RA,RB,DEVA,EVWID)
!!!
!!!       ATMP=ATMP/WTOT
!!!       IF(ATMP.LT.ZERO) ATMP=ZERO
!!!       ATMP=SQRT(ATMP)
!!!       ARMS(JREP)=ATMP
!!!       !
!!!       !     In order to sum up we need to reassign processes
!!!       !
!!!       !     Is this OK for parallel/parallel repdstr:
!!!       !     This code is not parallel so it should be executed on mynod==0
!!!       !     zero the vectors ARMS, FRMS for all CPUs and then the stuff below is OK
!!!       !     if (mynod==0) call swapd is perfectly OK!!!! This is what we want,
!!!       !     then we need to broadcast anything ???
!!!       !
!!!       CALL PSETGLOB
!!!       CALL GCOMB(ARMS,NREP)
!!!       CALL PSETLOC
!!!
!!!       DO I=1,NREPM
!!!          RMST=RMST+ARMS(I)
!!!       ENDDO
!!!    ENDIF

#if KEY_REPDSTR==1 /*repd2*/
    !
    !     For distributed replica we need to get the coordinates
    !
    !     NOTE 1:
    !     NATREP usually equals NATOM. In case of different number
    !     of atoms on each replica we need to specify NATREP
    !     on the command line. Currently this works only for systems
    !     which start with the same atoms from 1,NATREP. For example
    !     protein can be solvated by different number of water molecules.
    !     In the PSF protein must be the first and then the rest of
    !     the system. syntax: REPDstr NREP <int> NATRep <int> ...
    !     default NATRep = -1 ?? With this we can see if it was
    !     specified or not, eg:
    !     natrep=natom
    !     if(natrepcmd.ne.-1)natrep=natrepcmd
    !     the rest of the program stays as is!
    !
    !     NOTE 2: IREPDSTR starts from 0!
    !
    IF(QREPDSTR) THEN
       !
       !     This is not yet extended for parallel/parallel
       !     This should work for cyclic case, too:
       LPEER=MOD(NREPDSTR+IREPDSTR-1,NREPDSTR)
       RPEER=MOD(NREPDSTR+IREPDSTR+1,NREPDSTR)
       LPEER2=MOD(NREPDSTR+IREPDSTR-2,NREPDSTR)
       RPEER2=MOD(NREPDSTR+IREPDSTR+2,NREPDSTR)
       JREP=IREPDSTR+1
       !
       !     Swap the data between the neighbors. This is a double
       !     direction swap since some machines can do it as fast as
       !     a single one. And we might need the other neighbor for angles...
       !
       !     X(*)       - current replica
       !     XPEER(*,1) - current - 2
       !     XPEER(*,2) - current - 1
       !     XPEER(*,3) - current  = X(*)
       !     XPEER(*,4) - current + 1
       !     XPEER(*,5) - current + 2
       !
       !C         write(50+irepdstr,*)
       !C     $        'EPATHR>before swapd:irepdstr,lp,rp,lp2,rp2=',
       !C     $        irepdstr,lpeer,rpeer,lpeer2,rpeer2
       !
       DO I=1,NATREP
          XPEERS(I,3)=X(I)
          YPEERS(I,3)=Y(I)
          ZPEERS(I,3)=Z(I)
       ENDDO

       DO I=1,5
          APEERS(I)=ZERO
       ENDDO
       ! In case of parallel/parallel we need to protect this communication
       IF (MYNOD == 0) THEN
       !write(outu,'(a,10i4)')' EPATHR>before swapd:me,np,meg,npg,irepdstr,lp,rp,lp2,rp2=', &
       !     mynod,numnod,mynodg,numnodg,irepdstr,lpeer,rpeer,lpeer2,rpeer2
       !
       !     Get the neighbor's coordinates. In order to do all the forces for
       !     current replica we need coordinates from both: i-1, and i+1
       !    SWAPD() is already made for parallel/parallel:
       !    so no extra scale of lpeer,rpeer needed...
       CALL SWAPD(NATREP,X,Y,Z,XPEERS(1,2),YPEERS(1,2),ZPEERS(1,2), &
            XPEERS(1,4),YPEERS(1,4),ZPEERS(1,4),LPEER,RPEER)
       !
       !     This is really needed just for angle terms but we can do it here
       !     or later move it down to epatha() routine
       !
       CALL SWAPD(NATREP,X,Y,Z,XPEERS(1,1),YPEERS(1,1),ZPEERS(1,1), &
            XPEERS(1,5),YPEERS(1,5),ZPEERS(1,5),LPEER2,RPEER2)
       !
       IPT=0
       JPT=0
       DO I=1,NATREP
          IPT=IPT+1
          JPT=JPT+1
          ATOMPR(1,I)=IPT
          ATOMPR(2,I)=JPT
       ENDDO

       CALL ECBSTF(X,Y,Z,XPEERS(1,4),YPEERS(1,4),ZPEERS(1,4), &
            ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
            WTOT,ATMP,RA,RB,DEVA,EVWID)

       ATMP=ATMP/WTOT
       IF(ATMP.LT.ZERO) ATMP=ZERO
       ATMP=SQRT(ATMP)
       ARMS(JREP)=ATMP
       !write(outu,*)'EPATHR>after swapd...'
       ENDIF   ! mynod==0
       !
       ! this routine may not work in parallel/parallel (test later)
       ! just exit this routine in the case of parallel/parallel REPDSTR
       if(numnod > 1) return
       !
       !     In order to sum up we need to reassign processes
       !
       !     Is this OK for parallel/parallel repdstr:
       !     This code is not parallel so it should be executed on mynod==0
       !     zero the vectors ARMS, FRMS for all CPUs and then the stuff below is OK
       !     if (mynod==0) call swapd is perfectly OK!!!! This is what we want,
       !     then we need to broadcast anything ???
       !
       CALL PSETGLOB
       CALL GCOMB(ARMS,NREP)
       CALL PSETLOC

       DO I=1,NREPM
          RMST=RMST+ARMS(I)
       ENDDO
    ENDIF

#endif /* (repd2)*/

    RMST=RMST/NREPM
    !write(50+mynodg,'(a,4i4,f15.10)') &
    !     'EPATHR-5>me,meg,numnod,nrepm,rmst=',mynod,mynodg,numnod,nrepm,rmst
    !
    RNRM=-ONE
    RNRM=RNRM/NREPM
    ERMS=ZERO


#if KEY_REPDSTR==1
    call chmalloc('epath.src','EPATHR','DXT',NATOM,crl=DXT)
    call chmalloc('epath.src','EPATHR','DYT',NATOM,crl=DYT)
    call chmalloc('epath.src','EPATHR','DZT',NATOM,crl=DZT)
    DXT(1:NATOM)=ZERO
    DYT(1:NATOM)=ZERO
    DZT(1:NATOM)=ZERO
#endif 
    !=====================================================================
    !
    ! in the case of REPDSTR we need to protect more code here for parallel/parallel
    ! do it later.... MH, July 5, 2010
    !
    loopnrepm: DO JREP=1,NREPM
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          !C          IF(IREPDSTR.NE.(JREP-1))GOTO 888
          KIDX=IDXREP(JREP,NREPM,lcycl)
          IF(KIDX.LT.0) GOTO 888
       ELSE
          KIDX=0
       ENDIF
#endif 
       KREP=JREP+1
       IF(KREP.GT.NREP) KREP=KREP-NREP
       IF(ARMS(JREP).GT.TENM5) THEN  ! don't process very short vectors
          IF((.NOT.QPCIMG).OR. &
               (JREP.NE.ICIMG-1.AND.JREP.NE.ICIMG)) THEN
#if KEY_REPDSTR==1
             IF(KIDX.EQ.0) THEN       
#endif
                ERMS=ERMS+KRMS*(ARMS(JREP)-RMST)**2
#if KEY_REPDSTR==1
             ENDIF                    
#endif

             ATMP=ZERO
             DO IREP=1,NREPM
                TMP=RNRM
                IF(IREP.EQ.JREP) TMP=TMP+ONE
                ATMP=ATMP+KRMS*TMP*(ARMS(IREP)-RMST)
             ENDDO
             !
             ! Process extra restraint if path step is too long
             IF(ARMS(JREP).GT.RMXRMS) THEN
                IF(KMXRMS.GT.ZERO) THEN
#if KEY_REPDSTR==1
                   IF(KIDX.EQ.0)THEN                      
#endif
                      ERMS=ERMS+KMXRMS*(ARMS(JREP)-RMXRMS)**2
#if KEY_REPDSTR==1
                   ENDIF                                  
#endif
                   ATMP=ATMP+(ARMS(JREP)-RMXRMS)*KMXRMS
                ENDIF
             ENDIF
             !
             ! Process extra restraint if path step is too short
             IF(ARMS(JREP).LT.RMNRMS) THEN
                IF(KMNRMS.GT.ZERO) THEN
                   TMP=RMNRMS*(RMNRMS/ARMS(JREP)-ONE)
#if KEY_REPDSTR==1
                   IF(KIDX.EQ.0)THEN                      
#endif
                      ERMS=ERMS+KMNRMS*TMP**2
#if KEY_REPDSTR==1
                   ENDIF                                  
#endif
                   ATMP=ATMP-TMP*KMNRMS*(RMNRMS/ARMS(JREP))**2
                ENDIF
             ENDIF

             DERMS=ATMP/ARMS(JREP)  ! (dE/RdR)

#if KEY_REPDSTR==1
             IF(KIDX.EQ.0)THEN             
#endif
                FRMS(JREP)=FRMS(JREP)+ATMP
                FRMS(KREP)=FRMS(KREP)-ATMP
#if KEY_REPDSTR==1
             ENDIF                         
#endif

             IF(QPRINT) WRITE(OUTU,57) JREP,ARMS(JREP),ATMP
57           FORMAT('  In EPATHR: rms between replicas:',I5,F14.5, &
                  '  Force=',F14.5)

#if KEY_REPDSTR==1
             IF(QREPDSTR)THEN
                IPT=0
                JPT=0
             ELSE
#endif 
                IPT=ISTREP(JREP)
                JPT=ISTREP(KREP)
#if KEY_REPDSTR==1
             ENDIF                     
#endif
             DO I=1,NATREP
                IPT=IPT+1
                JPT=JPT+1
                ATOMPR(1,I)=IPT
                ATOMPR(2,I)=JPT
             ENDDO

             IF (QREPDSTR) THEN
!!!                CALL ECBSTF(X,Y,Z,IXPR,IYPR,IZPR, &
!!!                     ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, (/ZERO/), .FALSE.,WEIGHT, &
!!!                     WTOT,ATMP,RA,RB,DEVA,EVWID)
#if KEY_REPDSTR==1
                CALL ECBSTF(XPEERS(:,KIDX+2),YPEERS(:,KIDX+2),ZPEERS(:,KIDX+2), &
                     XPEERS(:,KIDX+3),YPEERS(:,KIDX+3),ZPEERS(:,KIDX+3), &
                     ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
                     WTOT,APEERS(KIDX+2),RA,RB,DEVA,EVWID)
#endif 
             ELSE
                CALL ECBSTF(X,Y,Z,X,Y,Z,ATOMPR,NATREP, &
                     LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
                     WTOT,ATMP,RA,RB,DEVA,EVWID)
             ENDIF
             !
             ! Decide which subroutine to call base on QPNEB
             !
             IF(QPNEB) THEN
                CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
                     .TRUE.,PSDX,PSDY,PSDZ,.TRUE.,PSDX,PSDY,PSDZ, &
                     DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA,QRPATH)
                !     CALL ECFORCNEB(ATOMPR,JREP,KREP,NATREP,
                !     &                   LNOROT,LNOTRN,LPRINT,
                !     &       .TRUE.,PSDX,PSDY,PSDZ,.TRUE.,PSDX,PSDY,PSDZ,
                !     &              DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,
                !     &              PATANG,DRTEMP,DEVA,NREPL)
                !     CALL ECFORCNEB(ATOMPR,JREP,KREP,NATREP,
                !     &                   LNOROT,LNOTRN,LPRINT,
                !     &       .TRUE.,DX,DY,DZ,.TRUE.,DX,DY,DZ,
                !     &              DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,
                !     &              PATANG,DRTEMP,DEVA,NREPL)
             ELSE
                !     CALL ECFORCNEB(ATOMPR,JREP,KREP,NATREP,
                !     &                   LNOROT,LNOTRN,LPRINT,
                !     &       .TRUE.,DX,DY,DZ,.TRUE.,DX,DY,DZ,
                !     &              DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,
                !     &              PATANG,DRTEMP,DEVA,NREPL)
                QFORCA=.TRUE.
                QFORCB=.TRUE.
                !     Note for REPDSTR:
                !     We need to send DRB to next replica, but not update
                !     DX,DY,DZ with it. QFORCB controls just update of DX,DY,DZ and not
                !     calculation of DRB, so it is OK to just put it to .FALSE.
                !
#if KEY_REPDSTR==1
                IF(QREPDSTR)THEN
                   CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
                        QFORCA,DXT,DYT,DZT,QFORCB,DXT,DYT,DZT, &
                        DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA,QRPATH)
                   IF(KIDX.EQ.0)THEN
                      DO I=1,NATREP
                         DX(I)=DX(I)+DRB(1,I)
                         DY(I)=DY(I)+DRB(2,I)
                         DZ(I)=DZ(I)+DRB(3,I)
                      ENDDO
                   ENDIF
                   IF(KIDX.EQ.1)THEN
                      DO I=1,NATREP
                         DX(I)=DX(I)+DRA(1,I)
                         DY(I)=DY(I)+DRA(2,I)
                         DZ(I)=DZ(I)+DRA(3,I)
                      ENDDO
                   ENDIF
                ELSE
#endif 
                CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
                     QFORCA,DX,DY,DZ,QFORCB,DX,DY,DZ, &
                     DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA,QRPATH)
#if KEY_REPDSTR==1
             ENDIF                                             
#endif
          ENDIF

       ENDIF                     ! ARMS
    ENDIF                      ! QPCIMG

!!!    if(.not.qrepdstr)write(50,*)'EPATHR:DX>jrep,krep,natom=', &
!!!         jrep,krep,natom
!!!    if(.not.qrepdstr)write(50,'(6f12.7)')(dx(i),i=1,natom)
!!!    if(.not.qrepdstr)write(50,*)'EPATHR:RA>jrep,krep,natrep=', &
!!!         jrep,krep,natrep
!!!    if(.not.qrepdstr)write(50,'(6f12.7)')(ra(1,i),i=1,natrep)
!!!    if(.not.qrepdstr)write(50,*)'EPATHR:RB>jrep,krep,natrep=', &
!!!         jrep,krep,natrep
!!!    if(.not.qrepdstr)write(50,'(6f12.7)')(rb(1,i),i=1,natrep)
!!!    if(.not.qrepdstr)write(50,*)'EPATHR:DRA>jrep,krep,natrep=', &
!!!         jrep,krep,natrep
!!!    if(.not.qrepdstr)write(50,'(6f12.7)')(dra(1,i),i=1,natrep)
!!!    if(.not.qrepdstr)write(50,*)'EPATHR:DRB>jrep,krep,natrep=', &
!!!         jrep,krep,natrep
!!!    if(.not.qrepdstr)write(50,'(6f12.7)')(drb(1,i),i=1,natrep)
!!!    
!!!    if(qrepdstr)write(50+irepdstr,*)'EPATHR>kidx=',kidx
!!!    if(qrepdstr)write(50+irepdstr,*)'EPATHR:DX>jrep,krep,natom=', &
!!!         jrep,krep,natom
!!!    if(qrepdstr)write(50+irepdstr,'(6f12.7)')(dx(i),i=1,natom)
!!!    if(qrepdstr)write(50+irepdstr,*)'EPATHR:RA>jrep,krep,natrep=', &
!!!         jrep,krep,natrep
!!!    if(qrepdstr)write(50+irepdstr,'(6f12.7)')(ra(1,i),i=1,natrep)
!!!    if(qrepdstr)write(50+irepdstr,*)'EPATHR:RB>jrep,krep,natrep=', &
!!!         jrep,krep,natrep
!!!    if(qrepdstr)write(50+irepdstr,'(6f12.7)')(rb(1,i),i=1,natrep)
!!!    if(qrepdstr)write(50+irepdstr,*)'EPATHR:DRA>jrep,krep,natrep=', &
!!!         jrep,krep,natrep
!!!    if(qrepdstr)write(50+irepdstr,'(6f12.7)')(dra(1,i),i=1,natrep)
!!!    if(qrepdstr)write(50+irepdstr,*)'EPATHR:DRB>jrep,krep,natrep=', &
!!!         jrep,krep,natrep
!!!    if(qrepdstr)write(50+irepdstr,'(6f12.7)')(drb(1,i),i=1,natrep)
!!!    if(qrepdstr)write(50+irepdstr,*)'EPATHR:DXT>jrep,krep,natrep=', &
!!!         jrep,krep,natrep
!!!    if(qrepdstr)write(50+irepdstr,'(6f12.7)')(dxt(i),i=1,natrep)
888 CONTINUE
    ENDDO loopnrepm
    EPTHR=ERMS*HALF
    !
#if KEY_REPDSTR==1
    !C      if(qrepdstr)write(50+irepdstr,*)               
#endif
#if KEY_REPDSTR==1
    !C     $     'End of EPATHR>apeers=',(apeers(i),i=1,5) 
#endif
    !
    !     Need to send/receive DRB to/from the next/previous process
    !
!!!old    IF(QREPDSTR) CALL SENDFOR(IREPDSTR,IREPDSTR+1,IREPDSTR-1, &
!!!old         NREP,NATREP,DRB,DX,DY,DZ,.TRUE.)
    
    !
    !      write(90+irepdstr+1,'(3f25.10)')(dx(j),dy(j),dz(j),j=1,natom)
    !
    !C      stop
    !
    !     In order to summ up we need to reassign processes
#if KEY_REPDSTR==1
    IF(QREPDSTR)THEN
       CALL PSETGLOB
       CALL GCOMB(FRMS,NREP)
       CALL PSETLOC
    ENDIF
    !C      if(.not.qrepdstr)write(50,'(a,5i5)')
    !C     $     'EPATHR:FRMS_GCOMB>nrep,natrep=',nrep,natrep
    !C      if(.not.qrepdstr)write(50,'(6f12.7)')(frms(i),i=1,nrep)
    !C      if(qrepdstr)write(50+irepdstr,'(a,5i5)')
    !C     $     'EPATHR:FRMS_GCOMB>nrep,natrep=',nrep,natrep
    !C      if(qrepdstr)write(50+irepdstr,'(6f12.7)')(frms(i),i=1,nrep)
    
    call chmdealloc('epath.src','EPATHR','DZT',NATOM,crl=DZT)
    call chmdealloc('epath.src','EPATHR','DYT',NATOM,crl=DYT)
    call chmdealloc('epath.src','EPATHR','DXT',NATOM,crl=DXT)
#endif 
    
    !
    !      write(50+mynodg,'(3f25.10)')(dx(i),dy(i),dz(i),i=1,natom)
    !      write(60+mynodg,'(3f25.10)')((dra(i,j),i=1,3),j=1,natom)
    !      write(70+mynodg,'(3f25.10)')((drb(i,j),i=1,3),j=1,natom)
    !      write(80+mynodg,'(3f25.10)')((ra(i,j),i=1,3),j=1,natom)
    !      write(90+mynodg,'(3f25.10)')((rb(i,j),i=1,3),j=1,natom)
    !      stop
    !
    RETURN
  END SUBROUTINE EPATHR

#if KEY_REPDSTR==1
  INTEGER FUNCTION IDXREP(JREP,NREPM,lcycl)
    !
    !     This function returns the positive value if part of the
    !     original JREP=1,NREPM has to be executed, or -1 when not
    !     This one is for EPATHR
    !
    use parallel
    use repdstr

    INTEGER JREP,NREPM
    logical lcycl

    IDXREP=-1
    if(mynod /= 0) return  ! for parallel/parallel
    IF (LCYCL) THEN
       IF(MOD(JREP,NREPM) == IREPDSTR) IDXREP=0
    ELSE
       IF(JREP == IREPDSTR) IDXREP=0
    ENDIF
    IF(JREP == IREPDSTR+1) IDXREP=1

    RETURN
  END FUNCTION IDXREP

  INTEGER FUNCTION JDXREP(JREP,NREPM)
    !
    !     This function returns the positive value if part of the
    !     original JREP=1,NREPM has to be executed, or -1 when not
    !     This one is for EPATHA
    !
    use repdstr

    INTEGER JREP,NREPM

    JDXREP=-1
    IF(JREP.EQ.IREPDSTR-1) JDXREP=0
    IF(JREP.EQ.IREPDSTR)   JDXREP=1
    IF(JREP.EQ.IREPDSTR+1) JDXREP=2

    RETURN
  END FUNCTION JDXREP

#endif 

!!!  SUBROUTINE SENDFOR(IREP,NEXT,PREV,NREP,NAT,DRB,DX,DY,DZ,Q)
!!!    !
!!!    !     This routine performs communication needed to
!!!    !     sum up the forces between two neighboring replicas
!!!
!!!    use memory
!!!    use parallel
!!!
!!!    INTEGER IREP,NREP,NAT,NEXT,PREV
!!!    REAL(CHM_REAL), ALLOCATABLE,DIMENSION(:,:) :: RDRB
!!!    REAL(CHM_REAL) :: DRB(3,*),DX(*),DY(*),DZ(*)
!!!    LOGICAL Q
!!!
!!!    INTEGER I,J,LNEXT,LPREV,NAT3
!!!    !
!!!    !     0 >= IREP < NREP
!!!    !     0 >= NEXT,PREV < NREP
!!!    !
!!!    LNEXT=NEXT
!!!    LPREV=PREV
!!!    IF(LNEXT.GE.NREP)LNEXT=-1  ! this should depend on LCYCL
!!!    IF(LPREV.LT.0)LPREV=-1     ! this should also depend on LCYCL
!!!    NAT3=3*NAT
!!!
!!!    call chmalloc('epath.src','SENDFOR','RDRB',3,NAT,crl=RDRB)
!!!    CALL SWAP1D(NAT3,RDRB,DRB,LPREV*NUMNOD,LNEXT*NUMNOD)
!!!
!!!    IF ((LPREV.GE.0).AND.Q) THEN      ! LCYCL??
!!!       DO I=1,NAT
!!!          DX(I)=DX(I)+RDRB(1,I)
!!!          DY(I)=DY(I)+RDRB(2,I)
!!!          DZ(I)=DZ(I)+RDRB(3,I)
!!!       ENDDO
!!!    ENDIF
!!!    call chmdealloc('epath.src','SENDFOR','RDRB',3,NAT,crl=RDRB)
!!!    !      write(20+irep+1,*)'RDRB,meg,irep,ln,lp=',mynodg,irep,lnext,lprev
!!!    !      write(20+irep+1,'(3f25.10)')((rdrb(i,j),i=1,3),j=1,nat)
!!!    !
!!!    RETURN
!!!  END SUBROUTINE SENDFOR
!!!
!!!  SUBROUTINE Cswap1d(me,meg,irepdstr,x,y,z,xn,yn,zn,xnn,ynn,znn)
!!!
!!!    implicit none
!!!    integer me,meg,irepdstr,i
!!!    real(chm_real) x(*),xn(*),xnn(*)
!!!    real(chm_real) y(*),yn(*),ynn(*)
!!!    real(chm_real) z(*),zn(*),znn(*)
!!!
!!!    return
!!!  end SUBROUTINE CSWAP1D

  SUBROUTINE EPATHA(EPTHA,DX,DY,DZ,X,Y,Z,NREP,KANG,PCOSMX, &
       NATREP,ISTREP,ARMS,FANG,ACST,WEIGHT,WTOT,QPRINT, &
       LNOTRN,LNOROT,LCYCL,ATOMPR,RA,RB,DRA,DRB,DRTEMP, &
       EVWID,QPNEB,PJDX,PJDY,PJDZ)
    !
    !       Rmin=( A**2 + B**2 + 2*A*B*cosmax )**0.5
    !
    !   Eang = sum  { 0.5* Kang * ( Rmin - C )**2 }  (cos(theta)<cosmax)
    !             I
    !
    !        =  0.0   when cos(theta) >= cosmax                  I=1,NREP-2
    !
    !      where cos(theta) is determined from the dotproduct of weighted deviation
    !      vectors between replicas I and I+1 and between I+1 and I+2.
    !
#if KEY_REPDSTR==1
    use repdstr
#endif

    use number
    use stream
    use consta
    use memory
    use parallel
    use repdstr

    real(chm_real) EPTHA,KANG,PCOSMX
    real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
    real(chm_real) PJDX(*),PJDY(*),PJDZ(*)
    INTEGER NREP,NATREP,ISTREP(*)
    real(chm_real) ARMS(*),FANG(*),ACST(*),WEIGHT(*),WTOT
    LOGICAL QPRINT,LNOTRN,LNOROT,LCYCL,QPNEB,QRPATH
    INTEGER ATOMPR(2,NATREP)
    real(chm_real) RA(3,*),RB(3,*),DRA(3,*),DRB(3,*),DRTEMP(*)
    real(chm_real) EVWID
    INTEGER NEXT,NEXTNEXT,PREV,PREVPREV

    real(chm_real)  EANG, ATMP, DXIR, DXJR, DYIR, DTXI, DYJR, DZIR
    real(chm_real)  DTXJ, DTYI, DZJR, DTYJ, DTZI, DTZJ
    real(chm_real)  AX, AY, AZ, DFX, DFY, DGX, DFZ, DGY, DGZ, CST,  &
         R2, AT
    real(chm_real)  RMIN, DERMS, DECST, U(3,3),PREVCST,PREVDECST
    INTEGER NREPM, JREP, KREP, LREP, I, IPT, JPT, KPT, J, KIDX
    LOGICAL LPRINT,QUPF,QDRA,QDRB
    real(chm_real) DEVA(3,3)
    REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:) :: DXT,DYT,DZT

    QUPF=.TRUE.
    LPRINT = QPRINT .AND. (PRNLEV.GT.7)
    IF(LCYCL) THEN
       NREPM=NREP
    ELSE
       NREPM=NREP-2
    ENDIF

    ! this might be a bug:
    ! always the same numbers for cyclic in repdstr
#if KEY_REPDSTR==1
    if (qrepdstr) nrepm=nrep-2           
#endif
    EANG=ZERO

    QRPATH=.TRUE.
    IF(QPNEB)QRPATH=.FALSE.

!!! old force comunication stuff
!!!    IF(QREPDSTR)THEN
!!!       !
!!!       !     Parallelization of the original loop has no direct analogy between
!!!       !     the JREP and MYNOD, because this would require extra
!!!       !     communication. Some of the stuff here is calculated on MYNOD for
!!!       !     JREP+1 (Case C below). So I decided to make new code here and
!!!       !     protect it properly.
!!!       !
!!!       !     This code can be implemented in two ways. The way it is now
!!!       !     (with the communication of the forces) or with the communication
!!!       !     of just coordnate sets. Each replica (except endpoints) need
!!!       !     2 neighbors back and 2 in front for this to work. The problem
!!!       !     in this case would be double calculations, for a little gain
!!!       !     in communication (~20%)
!!!       !
!!!       !  ***   SPECIAL NOTE ***
!!!       !
!!!       !     FANG() array which is used for printouts is not OK in this
!!!       !     version. To be fixed, we need some additional communication
!!!       !     This can be added later if needed
!!!       !     In order to debug it leave the write statements in this routine!
!!!       !
!!!       !     Also QEPTHA seems to be .TRUE. even when kang = 0.0 ??  Be
!!!       !     carefull!
!!!       !------------------------------------------
!!!       !     Make this also for EPATHR():  !!!!
!!!       !     FIX!!
!!!       !     in EPATHR() replace SWAPD() and make it only for IXPL, not IXPR
!!!       !------------------------------------------
!!!       !
!!!       !     First get the coordinates from next, and next to next:
!!!       !
!!!       !c      write(80+irepdstr+1,*)'EPATHA>input forces: irepdstr=',irepdstr
!!!       !c      write(80+irepdstr+1,'(3f25.10)')(dx(j),dy(j),dz(j),j=1,natrep)
!!!       !
!!!       NEXT=IREPDSTR+1
!!!       NEXTNEXT=NEXT+1
!!!       PREV=IREPDSTR-1
!!!       PREVPREV=PREV-1
!!!       IF(NEXT.GE.NREPDSTR)NEXT=-1
!!!       IF(NEXTNEXT.GE.NREPDSTR)NEXTNEXT=-1
!!!       IF(PREV.LT.0)PREV=-1
!!!       IF(PREVPREV.LT.0)PREVPREV=-1
!!!       !
!!!       !     The above needs to be adjusted when LCYCL is true
!!!       !
!!!       CALL SWAP1D(NATREP,IXPL,X,NEXT,PREV)
!!!       CALL SWAP1D(NATREP,IYPL,Y,NEXT,PREV)
!!!       CALL SWAP1D(NATREP,IZPL,Z,NEXT,PREV)
!!!       CALL SWAP1D(NATREP,IXPR,X,NEXTNEXT,PREVPREV)
!!!       CALL SWAP1D(NATREP,IYPR,Y,NEXTNEXT,PREVPREV)
!!!       CALL SWAP1D(NATREP,IZPR,Z,NEXTNEXT,PREVPREV)
!!!
!!!       JREP=IREPDSTR+1
!!!       KREP=IREPDSTR+2
!!!       LREP=IREPDSTR+3
!!!       IF(KREP.GT.NREP) KREP=KREP-NREP
!!!       IF(LREP.GT.NREP) LREP=LREP-NREP
!!!       !
!!!       !CCwhy?         IF(ARMS(JREP).GT.TENM5 .AND. ARMS(KREP).GT.TENM5) THEN
!!!       !
!!!       !     In case of REPDSTR we select all the atoms, but this could
!!!       !     be a place to implement selections
!!!       !
!!!       IPT=0
!!!       JPT=0
!!!       DO I=1,NATREP
!!!          IPT=IPT+1
!!!          JPT=JPT+1
!!!          ATOMPR(1,I)=IPT
!!!          ATOMPR(2,I)=JPT
!!!       ENDDO
!!!       !
!!!       !---------------------------------
!!!       !     Case A: current vs current+2
!!!       ATMP=ZERO
!!!       !cc         IF(NEXTNEXT.NE.-1) THEN
!!!       !
!!!       CALL ECBSTF(X,Y,Z,IXPR,IYPR,IZPR, &
!!!            ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT,0,.FALSE.,WEIGHT, &
!!!            WTOT,ATMP,RA,RB,DEVA,EVWID)
!!!
!!!       ATMP=ATMP/WTOT
!!!       IF(ATMP.LT.ZERO) ATMP=ZERO
!!!       !
!!!       !cc         ENDIF
!!!       !
!!!       !  law of cosines
!!!       CST=ONE
!!!       IF(ARMS(JREP).GT.TENM5 .AND. ARMS(KREP).GT.TENM5) THEN
!!!          CST=(ATMP - ARMS(JREP)**2 - ARMS(KREP)**2)/ &
!!!               (TWO*ARMS(JREP)*ARMS(KREP))
!!!       ENDIF
!!!       ACST(KREP)=CST
!!!
!!!       IF(QPRINT) THEN
!!!          AT=ACOS(CST)
!!!          WRITE(OUTU,451) KREP,AT*RADDEG
!!!451       FORMAT('  In EPATHA: Angle at center:',I5,F14.5)
!!!       ENDIF
!!!
!!!       DERMS=ZERO
!!!       PREVDECST=ZERO
!!!       PREVCST=ZERO
!!!       IF(CST.LT.PCOSMX) THEN
!!!
!!!          EANG=EANG+HALF*KANG*(CST-PCOSMX)**2
!!!          DECST=KANG*(CST-PCOSMX)
!!!
!!!          ATMP=SQRT(ATMP)
!!!          IF(ARMS(JREP).GT.TENM5 .AND. ARMS(KREP).GT.TENM5) THEN
!!!             DERMS=DECST/(ARMS(JREP)*ARMS(KREP))
!!!          ENDIF
!!!
!!!          IF(IREPDSTR.GE.(NREPDSTR-2))THEN
!!!             EANG=ZERO
!!!          ELSE
!!!             FANG(JREP)=FANG(JREP)+ATMP*DERMS
!!!             FANG(LREP)=FANG(LREP)-ATMP*DERMS
!!!          ENDIF
!!!
!!!       ENDIF
!!!       !
!!!       !         write(*,'(a,i2,6f7.3)')
!!!       !     $    'EPATHA-fang1>meg,a*d,fang=',mynodg,atmp*derms,(fang(i),i=1,5)
!!!       !
!!!       !     Because we do the last part of the original loop on the
!!!       !     next processor we need CST and DECST on that processor
!!!       !     Go, send it there!
!!!       !
!!!       CALL SWAP1D(1,PREVCST,CST,PREV,NEXT)
!!!       CALL SWAP1D(1,PREVDECST,DECST,PREV,NEXT)
!!!       !
!!!       !     Cleanup in case we dont have data!
!!!       DRA(1:3,1:NATREP)=zero
!!!       DRB(1:3,1:NATREP)=zero
!!!       IF(NEXTNEXT.EQ.-1)THEN
!!!          DERMS=ZERO
!!!          RB(1:3,1:NATREP)=zero
!!!       ENDIF
!!!
!!!       !     DX,DY,DZ has to get DRB before update, hence the second .FALSE.
!!!       CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
!!!            .TRUE.,DX,DY,DZ,.FALSE.,DX,DY,DZ, &
!!!            DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA, &
!!!            QRPATH)
!!!       !
!!!       !     Need to send/receive DRB to/from the next/previous process
!!!       !
!!!       !         write(70+jrep,*)'EPATHA-CaseA>Before sendfor:'
!!!       !         write(70+jrep,'(3f25.10)')(dx(j),dy(j),dz(j),j=1,natrep)
!!!       !
!!!       CALL SENDFOR(IREPDSTR,NEXTNEXT,PREVPREV,NREP,NATREP, &
!!!            DRB,DX,DY,DZ,qupf)
!!!       ! A:
!!!       !C         write(70+jrep,*)'EPATHA-CaseA>After sendfor:'
!!!       !C         write(70+jrep,'(3f25.10)')(dx(j),dy(j),dz(j),j=1,natrep)
!!!       !CC
!!!       !C         write(10+jrep,*)'EPATHA-CaseA>meg,jrep,krep,lrep=',
!!!       !C     $        mynodg,jrep,krep,lrep
!!!       !C         write(10+jrep,*)'meg,j,l,derms,cst,atmp,arms(j),arms(l)='
!!!       !C         write(10+jrep,*)
!!!       !C     $        mynodg,jrep,lrep,derms,cst,atmp,arms(jrep),arms(lrep)
!!!       !C         write(10+jrep,*)'RA,meg,jrep=',mynodg,jrep
!!!       !C         write(10+jrep,'(3f25.10)')((ra(i,j),i=1,3),j=1,natrep)
!!!       !C         write(10+jrep,*)'RB,meg,lrep=',mynodg,lrep
!!!       !C         write(10+jrep,'(3f25.10)')((rb(i,j),i=1,3),j=1,natrep)
!!!       !C         write(10+jrep,*)'DRA,meg,jrep=',mynodg,jrep
!!!       !C         write(10+jrep,'(3f25.10)')((dra(i,j),i=1,3),j=1,natrep)
!!!       !C         write(10+jrep,*)'DRB,meg,lrep=',mynodg,lrep
!!!       !C         write(10+jrep,'(3f25.10)')((drb(i,j),i=1,3),j=1,natrep)
!!!       !
!!!       !     End of Case A
!!!       !-------------------------------------------------
!!!       !     Case B: current+1 vs current  (note the order!)
!!!       !
!!!       ATMP=ZERO
!!!       !cc         IF(NEXT.NE.-1) THEN
!!!       !
!!!       CALL ECBSTF(IXPL,IYPL,IZPL,X,Y,Z, &
!!!            ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT,0,.FALSE.,WEIGHT, &
!!!            WTOT,ATMP,RA,RB,DEVA,EVWID)
!!!
!!!       ATMP=ATMP/WTOT
!!!       IF(ATMP.LT.ZERO) ATMP=ZERO
!!!       !
!!!       !cc         ENDIF
!!!       !
!!!       DERMS=ZERO
!!!       IF((JREP.GT.0).AND.(KREP.GT.0)) &
!!!            DERMS=DECST*(MINONE/(ARMS(JREP)*ARMS(KREP)) - CST/ATMP)
!!!
!!!       IF(IREPDSTR.LT.(NREPDSTR-2))THEN
!!!          FANG(KREP)=FANG(KREP)+ATMP*DERMS
!!!          FANG(JREP)=FANG(JREP)-ATMP*DERMS
!!!       ENDIF
!!!       !
!!!       !         write(*,'(a,i2,6f7.3)')
!!!       !     $    'EPATHA-fang2>meg,a*d,fang=',mynodg,atmp*derms,(fang(i),i=1,5)
!!!       !
!!!       QDRB=.TRUE.
!!!       !
!!!       !     The replicas at the end get the data from other replicas
!!!       !     via communication SENDFOR()
!!!       !
!!!       IF(IREPDSTR.GE.(NREPDSTR-2))QDRB=.FALSE.
!!!       CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
!!!            .FALSE.,DX,DY,DZ,QDRB,DX,DY,DZ, &
!!!            DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA, &
!!!            QRPATH)
!!!       !
!!!       !     Need to send/receive DRA to/from the next/previous process
!!!       !
!!!       !C         write(70+jrep,*)'EPATHA-CaseB>Before sendfor:'
!!!       !C         write(70+jrep,'(3f25.10)')(dx(j),dy(j),dz(j),j=1,natrep)
!!!       !
!!!       !        No need to update if the last replica:
!!!       IF(IREPDSTR.GE.(NREPDSTR-1))QUPF=.FALSE.
!!!       CALL SENDFOR(IREPDSTR,NEXT,PREV, &
!!!            NREP,NATREP,DRA,DX,DY,DZ,QUPF)
!!!       QUPF=.TRUE.
!!!       ! B:
!!!       !C         write(70+jrep,*)'EPATHA-CaseB>After sendfor:'
!!!       !C         write(70+jrep,'(3f25.10)')(dx(j),dy(j),dz(j),j=1,natrep)
!!!       !CC
!!!       !C         write(10+jrep,*)'EPATHA-CaseB>meg,jrep,krep,lrep=',
!!!       !C     $        mynodg,jrep,krep,lrep
!!!       !C         write(10+jrep,*)'meg,k,j,derms,cst,atmp,arms(k),arms(j)='
!!!       !C         write(10+jrep,*)
!!!       !C     $       mynodg,krep,jrep,derms,cst,atmp,arms(krep),arms(jrep)
!!!       !C         write(10+jrep,*)'RA,meg,krep=',mynodg,krep
!!!       !C         write(10+jrep,'(3f25.10)')((ra(i,j),i=1,3),j=1,natrep)
!!!       !C         write(10+jrep,*)'RB,meg,jrep=',mynodg,jrep
!!!       !C         write(10+jrep,'(3f25.10)')((rb(i,j),i=1,3),j=1,natrep)
!!!       !C         write(10+jrep,*)'DRA,meg,krep=',mynodg,krep
!!!       !C         write(10+jrep,'(3f25.10)')((dra(i,j),i=1,3),j=1,natrep)
!!!       !C         write(10+jrep,*)'DRB,meg,jrep=',mynodg,jrep
!!!       !C         write(10+jrep,'(3f25.10)')((drb(i,j),i=1,3),j=1,natrep)
!!!       !
!!!       !     End of Case B
!!!       !-------------------------------------------------
!!!       !     Case C: current vs current+1  (note the order!)
!!!       !
!!!       ATMP=ZERO
!!!       !cc         IF(NEXT.NE.-1) THEN
!!!       !
!!!       CALL ECBSTF(X,Y,Z,IXPL,IYPL,IZPL, &
!!!            ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT,0,.FALSE.,WEIGHT, &
!!!            WTOT,ATMP,RA,RB,DEVA,EVWID)
!!!
!!!       ATMP=ATMP/WTOT
!!!       IF(ATMP.LT.ZERO) ATMP=ZERO
!!!       !
!!!       !cc         ENDIF
!!!       !
!!!       ! orig:         DERMS=DECST*(MINONE/(ARMS(JREP)*ARMS(KREP)) - CST/ATMP)
!!!       DERMS=ZERO
!!!       IF((JREP.GT.0).AND.(KREP.GT.0).and.(next.ne.-1)) &
!!!            DERMS=PREVDECST &
!!!            *(MINONE/(ARMS(JREP-1)*ARMS(KREP-1))-PREVCST/ATMP)
!!!
!!!       FANG(KREP)=FANG(KREP)+ATMP*DERMS
!!!       FANG(JREP)=FANG(JREP)-ATMP*DERMS
!!!
!!!       CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
!!!            .TRUE.,DX,DY,DZ,.FALSE.,DX,DY,DZ, &
!!!            DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA, &
!!!            QRPATH)
!!!       !
!!!       !     Need to send/receive DRB to/from the next/previous process
!!!       !
!!!       CALL SENDFOR(IREPDSTR,NEXT,PREV, &
!!!            NREP,NATREP,DRB,DX,DY,DZ,qupf)
!!!
!!!       !     End of Case C
!!!       !-------------------------------------------------
!!!       !
!!!       !CCwhy?         ENDIF
!!!       !
!!!       !C         RETURN
!!!    ELSE


!!!! debug printing
!!!       if(.not.qrepdstr)write(50,*)'EPATHA:DX0>nrep,natom=', &
!!!            nrep,nrep*natrep
!!!       if(.not.qrepdstr)write(50,'(6f12.7)')(dx(i),i=1,nrep*natrep)
!!!
!!!       if(qrepdstr)write(50+irepdstr,*)'EPATHA:DX0>nrep,natom=', &
!!!            nrep,natrep
!!!       if(qrepdstr)write(50+irepdstr,'(6f12.7)')(dx(i),i=1,natrep)


#if KEY_REPDSTR==1
       call chmalloc('epath.src','EPATHA','DXT',NATREP,crl=DXT)
       call chmalloc('epath.src','EPATHA','DYT',NATREP,crl=DYT)
       call chmalloc('epath.src','EPATHA','DZT',NATREP,crl=DZT)
       DXT(1:NATREP)=ZERO
       DYT(1:NATREP)=ZERO
       DZT(1:NATREP)=ZERO
#endif 
       ! this may not work in parallel/parallel
       ! just exit this routine in the case of REPDSTR
#if KEY_REPDSTR==1
       if(qrepdstr.and.(numnod > 1)) return             
#endif
       !
       DO JREP=1,NREPM
#if KEY_REPDSTR==1
          IF(QREPDSTR)THEN
             KIDX=JDXREP(JREP,NREPM)
             IF(KIDX.LT.0) GOTO 888
          ELSE
             KIDX=2
          ENDIF
#endif 
          KREP=JREP+1
          IF(KREP.GT.NREP) KREP=KREP-NREP
          LREP=JREP+2
          IF(LREP.GT.NREP) LREP=LREP-NREP
          !
          !     Add here for REPDSTR
          !
          IF(ARMS(JREP).GT.TENM5 .AND. ARMS(KREP).GT.TENM5) THEN

#if KEY_REPDSTR==1
             IF(QREPDSTR)THEN
                IPT=0
                JPT=0
             ELSE
#endif 
                IPT=ISTREP(JREP)
                JPT=ISTREP(LREP)
#if KEY_REPDSTR==1
             ENDIF              
#endif

             DO I=1,NATREP
                IPT=IPT+1
                JPT=JPT+1
                ATOMPR(1,I)=IPT
                ATOMPR(2,I)=JPT
             ENDDO
#if KEY_REPDSTR==1
             !     CaseA (jrep,lrep)
             IF(QREPDSTR) THEN
                CALL ECBSTF(XPEERS(:,KIDX+1),YPEERS(:,KIDX+1),ZPEERS(:,KIDX+1), &
                     XPEERS(:,KIDX+3),YPEERS(:,KIDX+3),ZPEERS(:,KIDX+3), &
                     ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
                     WTOT,ATMP,RA,RB,DEVA,EVWID)
             ELSE
#endif 

                CALL ECBSTF(X,Y,Z,X,Y,Z,ATOMPR,NATREP, &
                     LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
                     WTOT,ATMP,RA,RB,DEVA,EVWID)

#if KEY_REPDSTR==1
             ENDIF                
#endif

             ATMP=ATMP/WTOT
             IF(ATMP.LT.ZERO) ATMP=ZERO
             !  law of cosines
             CST=(ATMP - ARMS(JREP)**2 - ARMS(KREP)**2)/ &
                  (TWO*ARMS(JREP)*ARMS(KREP))
             ACST(KREP)=CST

             IF(QPRINT) THEN
                AT=ACOS(CST)
                WRITE(OUTU,45) KREP,AT*RADDEG
45              FORMAT('  In EPATHA: Angle at center:',I5,F14.5)
             ENDIF

             IF(CST.LT.PCOSMX) THEN

#if KEY_REPDSTR==1
                IF(KIDX.EQ.2)THEN                      
#endif
                   EANG=EANG+HALF*KANG*(CST-PCOSMX)**2
#if KEY_REPDSTR==1
                ENDIF                                  
#endif
                DECST=KANG*(CST-PCOSMX)

                ATMP=SQRT(ATMP)
                DERMS=DECST/(ARMS(JREP)*ARMS(KREP))

#if KEY_REPDSTR==1
                IF(KIDX.EQ.2)THEN                      
#endif
                   FANG(JREP)=FANG(JREP)+ATMP*DERMS
                   FANG(LREP)=FANG(LREP)-ATMP*DERMS
#if KEY_REPDSTR==1
                ENDIF                                  
#endif

#if KEY_REPDSTR==1
                IF(QREPDSTR)THEN
                   CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
                        .TRUE.,DXT,DYT,DZT,.TRUE.,DXT,DYT,DZT, &
                        DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA, &
                        QRPATH)
                   IF(KIDX.EQ.0)THEN
                      DO I=1,NATREP
                         DX(I)=DX(I)+DRB(1,I)
                         DY(I)=DY(I)+DRB(2,I)
                         DZ(I)=DZ(I)+DRB(3,I)
                      ENDDO
                   ENDIF
                   IF(KIDX.EQ.2)THEN
                      DO I=1,NATREP
                         DX(I)=DX(I)+DRA(1,I)
                         DY(I)=DY(I)+DRA(2,I)
                         DZ(I)=DZ(I)+DRA(3,I)
                      ENDDO
                   ENDIF
                ELSE
#endif 

                   CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
                        .TRUE.,DX,DY,DZ,.TRUE.,DX,DY,DZ, &
                        DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA, &
                        QRPATH)
#if KEY_REPDSTR==1
                ENDIF                 
#endif
                IF(QPNEB) THEN
                   CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
                        .TRUE.,PJDX,PJDY,PJDZ,.TRUE.,PJDX,PJDY,PJDZ, &
                        DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA, &
                        QRPATH)
                ENDIF

!!! debug printing...
!!!                if(.not.qrepdstr)write(50,'(a,4i5,3f15.7)') &
!!!                     'EPATHA:E>jrep,krep,lrep,natom,atmp,cst,eang,=', &
!!!                     jrep,krep,lrep,nrep*natrep,atmp,cst,eang
!!!                if(.not.qrepdstr)write(50,'(a,6i5)') &
!!!                     'EPATHA:ACST>jrep,krep,lrep,nrep,natrep,natom=', &
!!!                     jrep,krep,lrep,nrep,natrep,nrep*natrep
!!!                if(.not.qrepdstr)write(50,'(6f12.5)')(acst(i),i=1,nrep)
!!!                if(.not.qrepdstr)write(50,'(a,6i5)') &
!!!                     'EPATHA:FANG>jrep,krep,lrep,nrep,natrep,natom=', &
!!!                     jrep,krep,lrep,nrep,natrep,nrep*natrep
!!!                if(.not.qrepdstr)write(50,'(6f12.7)')(fang(i),i=1,nrep)
!!!                if(.not.qrepdstr)write(50,*) &
!!!                     'EPATHA:DX(jrep,lrep)>jrep,krep,lrep,natom=', &
!!!                     jrep,krep,lrep,nrep*natrep
!!!                if(.not.qrepdstr)write(50,'(6f12.7)')(dx(i),i=1,nrep*natrep)
!!!                if(.not.qrepdstr)write(50,*)'EPATHA:RA>jrep,krep,lrep,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(.not.qrepdstr)write(50,'(6f12.7)')(ra(1,i),i=1,natrep)
!!!                if(.not.qrepdstr)write(50,*)'EPATHA:RB>jrep,krep,lrep,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(.not.qrepdstr)write(50,'(6f12.7)')(rb(1,i),i=1,natrep)
!!!                if(.not.qrepdstr)write(50,*)'EPATHA:DRA>jrep,krep,lrep,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(.not.qrepdstr)write(50,'(6f12.7)')(dra(1,i),i=1,natrep)
!!!                if(.not.qrepdstr)write(50,*)'EPATHA:DRB>jrep,krep,lrep,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(.not.qrepdstr)write(50,'(6f12.7)')(drb(1,i),i=1,natrep)
!!!
!!!                if(qrepdstr)write(50+irepdstr,'(a,4i5,3f15.7)') &
!!!                     'EPATHA:E>jrep,krep,lrep,natrep,atmp,cst,eang,=', &
!!!                     jrep,krep,lrep,natrep,atmp,cst,eang
!!!                if(qrepdstr)write(50+irepdstr,'(a,5i5)') &
!!!                     'EPATHA:ACST>jrep,krep,lrep,nrep,natrep=', &
!!!                     jrep,krep,lrep,nrep,natrep
!!!                if(qrepdstr)write(50+irepdstr,'(6f12.7)')(acst(i),i=1,nrep)
!!!                if(qrepdstr)write(50+irepdstr,'(a,5i5)') &
!!!                     'EPATHA:FANG>jrep,krep,lrep,nrep,natrep=', &
!!!                     jrep,krep,lrep,nrep,natrep
!!!                if(qrepdstr)write(50+irepdstr,'(6f12.7)')(fang(i),i=1,nrep)
!!!                if(qrepdstr)write(50+irepdstr,*)'EPATHA>kidx=',kidx
!!!                if(qrepdstr)write(50+irepdstr,*)'EPATHA:DX>jr,kr,lr,natom=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(qrepdstr)write(50+irepdstr,'(6f12.7)')(dx(i),i=1,natrep)
!!!                if(qrepdstr)write(50+irepdstr,*)'EPATHA:RA>jr,kr,lr,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(qrepdstr)write(50+irepdstr,'(6f12.7)')(ra(1,i),i=1,natrep)
!!!                if(qrepdstr)write(50+irepdstr,*)'EPATHA:RB>jr,kr,lr,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(qrepdstr)write(50+irepdstr,'(6f12.7)')(rb(1,i),i=1,natrep)
!!!                if(qrepdstr)write(50+irepdstr,*)'EPATHA:DRA>jr,kr,lr,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(qrepdstr)write(50+irepdstr,'(6f12.7)')(dra(1,i),i=1,natrep)
!!!                if(qrepdstr)write(50+irepdstr,*)'EPATHA:DRB>jr,kr,lr,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(qrepdstr)write(50+irepdstr,'(6f12.7)')(drb(1,i),i=1,natrep)
!!!                if(qrepdstr)write(50+irepdstr,*)'EPATHA:DXT>jr,kr,lr,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(qrepdstr)write(50+irepdstr,'(6f12.7)')(dxt(i),i=1,natrep)

#if KEY_REPDSTR==1
                IF(QREPDSTR)THEN
                   IPT=0
                   JPT=0
                ELSE
#endif 
                   IPT=ISTREP(KREP)
                   JPT=ISTREP(JREP)
#if KEY_REPDSTR==1
                ENDIF                   
#endif

                DO I=1,NATREP
                   IPT=IPT+1
                   JPT=JPT+1
                   ATOMPR(1,I)=IPT
                   ATOMPR(2,I)=JPT
                ENDDO
#if KEY_REPDSTR==1
                !     CaseB (krep,jrep)
                IF(QREPDSTR)THEN
                   CALL ECBSTF(XPEERS(:,KIDX+2),YPEERS(:,KIDX+2),ZPEERS(:,KIDX+2) &
                        ,XPEERS(:,KIDX+1),YPEERS(:,KIDX+1),ZPEERS(:,KIDX+1) &
                        ,ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
                        WTOT,ATMP,RA,RB,DEVA,EVWID)
                ELSE
#endif 

                   CALL ECBSTF(X,Y,Z,X,Y,Z,ATOMPR,NATREP, &
                        LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
                        WTOT,ATMP,RA,RB,DEVA,EVWID)
#if KEY_REPDSTR==1
                ENDIF                   
#endif

                ATMP=ATMP/WTOT
                DERMS=DECST*(MINONE/(ARMS(JREP)*ARMS(KREP)) - CST/ATMP)

#if KEY_REPDSTR==1
                IF(KIDX.EQ.2)THEN                      
#endif
                   FANG(KREP)=FANG(KREP)+ATMP*DERMS
                   FANG(JREP)=FANG(JREP)-ATMP*DERMS
#if KEY_REPDSTR==1
                ENDIF                                  
#endif

#if KEY_REPDSTR==1
                IF(QREPDSTR)THEN
                   CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
                        .TRUE.,DXT,DYT,DZT,.TRUE.,DXT,DYT,DZT, &
                        DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA, &
                        QRPATH)
                   IF(KIDX.EQ.1)THEN
                      DO I=1,NATREP
                         DX(I)=DX(I)+DRA(1,I)
                         DY(I)=DY(I)+DRA(2,I)
                         DZ(I)=DZ(I)+DRA(3,I)
                      ENDDO
                   ENDIF
                   IF(KIDX.EQ.2)THEN
                      DO I=1,NATREP
                         DX(I)=DX(I)+DRB(1,I)
                         DY(I)=DY(I)+DRB(2,I)
                         DZ(I)=DZ(I)+DRB(3,I)
                      ENDDO
                   ENDIF
                ELSE
#endif 
                   CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
                        .TRUE.,DX,DY,DZ,.TRUE.,DX,DY,DZ, &
                        DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA, &
                        QRPATH)
#if KEY_REPDSTR==1
                ENDIF                    
#endif
                IF(QPNEB) THEN
                   CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
                        .TRUE.,PJDX,PJDY,PJDZ,.TRUE.,PJDX,PJDY,PJDZ, &
                        DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA, &
                        QRPATH)
                ENDIF

!!! debug printout
!!!                if(.not.qrepdstr)write(50,*) &
!!!                     'EPATHA:DX(krep,jrep)>jrep,krep,lrep,natom=', &
!!!                     jrep,krep,lrep,nrep*natrep
!!!                if(.not.qrepdstr)write(50,'(6f12.7)')(dx(i),i=1,nrep*natrep)
!!!                if(.not.qrepdstr)write(50,*)'EPATHA:RA>jrep,krep,lrep,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(.not.qrepdstr)write(50,'(6f12.7)')(ra(1,i),i=1,natrep)
!!!                if(.not.qrepdstr)write(50,*)'EPATHA:RB>jrep,krep,lrep,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(.not.qrepdstr)write(50,'(6f12.7)')(rb(1,i),i=1,natrep)
!!!                if(.not.qrepdstr)write(50,*)'EPATHA:DRA>jrep,krep,lrep,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(.not.qrepdstr)write(50,'(6f12.7)')(dra(1,i),i=1,natrep)
!!!                if(.not.qrepdstr)write(50,*)'EPATHA:DRB>jrep,krep,lrep,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(.not.qrepdstr)write(50,'(6f12.7)')(drb(1,i),i=1,natrep)
!!!
!!!                if(qrepdstr)write(50+irepdstr,*)'EPATHA>kidx=',kidx
!!!                if(qrepdstr)write(50+irepdstr,*)'EPATHA:DX>jr,kr,lr,natom=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(qrepdstr)write(50+irepdstr,'(6f12.7)')(dx(i),i=1,natrep)
!!!                if(qrepdstr)write(50+irepdstr,*)'EPATHA:RA>jr,kr,lr,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(qrepdstr)write(50+irepdstr,'(6f12.7)')(ra(1,i),i=1,natrep)
!!!                if(qrepdstr)write(50+irepdstr,*)'EPATHA:RB>jr,kr,lr,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(qrepdstr)write(50+irepdstr,'(6f12.7)')(rb(1,i),i=1,natrep)
!!!                if(qrepdstr)write(50+irepdstr,*)'EPATHA:DRA>jr,kr,lr,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(qrepdstr)write(50+irepdstr,'(6f12.7)')(dra(1,i),i=1,natrep)
!!!                if(qrepdstr)write(50+irepdstr,*)'EPATHA:DRB>jr,kr,lr,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(qrepdstr)write(50+irepdstr,'(6f12.7)')(drb(1,i),i=1,natrep)
!!!                if(qrepdstr)write(50+irepdstr,*)'EPATHA:DXT>jr,kr,lr,natrep=', &
!!!                     jrep,krep,lrep,natrep
!!!                if(qrepdstr)write(50+irepdstr,'(6f12.7)')(dxt(i),i=1,natrep)

#if KEY_REPDSTR==1
                IF(QREPDSTR)THEN
                   IPT=0
                   JPT=0
                ELSE
#endif 
                   IPT=ISTREP(KREP)
                   JPT=ISTREP(LREP)
#if KEY_REPDSTR==1
                ENDIF                          
#endif

                DO I=1,NATREP
                   IPT=IPT+1
                   JPT=JPT+1
                   ATOMPR(1,I)=IPT
                   ATOMPR(2,I)=JPT
                ENDDO
#if KEY_REPDSTR==1
                !     CaseC(krep,lrep)
                IF(QREPDSTR)THEN
                   CALL ECBSTF(XPEERS(:,KIDX+2),YPEERS(:,KIDX+2),ZPEERS(:,KIDX+2) &
                        ,XPEERS(:,KIDX+3),YPEERS(:,KIDX+3),ZPEERS(:,KIDX+3) &
                        ,ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
                        WTOT,ATMP,RA,RB,DEVA,EVWID)
                ELSE
#endif 

                   CALL ECBSTF(X,Y,Z,X,Y,Z,ATOMPR,NATREP, &
                        LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
                        WTOT,ATMP,RA,RB,DEVA,EVWID)
#if KEY_REPDSTR==1
                ENDIF                        
#endif

                ATMP=ATMP/WTOT
                DERMS=DECST*(MINONE/(ARMS(JREP)*ARMS(KREP)) - CST/ATMP)

#if KEY_REPDSTR==1
                IF(KIDX.EQ.2)THEN                      
#endif
                   FANG(KREP)=FANG(KREP)+ATMP*DERMS
                   FANG(LREP)=FANG(LREP)-ATMP*DERMS
#if KEY_REPDSTR==1
                ENDIF                                  
#endif

#if KEY_REPDSTR==1
                IF(QREPDSTR)THEN
                   CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
                        .TRUE.,DXT,DYT,DZT,.TRUE.,DXT,DYT,DZT, &
                        DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA, &
                        QRPATH)
                   IF(KIDX.EQ.0)THEN
                      DO I=1,NATREP
                         DX(I)=DX(I)+DRB(1,I)
                         DY(I)=DY(I)+DRB(2,I)
                         DZ(I)=DZ(I)+DRB(3,I)
                      ENDDO
                   ENDIF
                   IF(KIDX.EQ.1)THEN
                      DO I=1,NATREP
                         DX(I)=DX(I)+DRA(1,I)
                         DY(I)=DY(I)+DRA(2,I)
                         DZ(I)=DZ(I)+DRA(3,I)
                      ENDDO
                   ENDIF
                ELSE
#endif 

                   CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
                        .TRUE.,DX,DY,DZ,.TRUE.,DX,DY,DZ, &
                        DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA, &
                        QRPATH)
#if KEY_REPDSTR==1
                ENDIF                   
#endif
                IF(QPNEB) THEN
                   CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
                        .TRUE.,PJDX,PJDY,PJDZ,.TRUE.,PJDX,PJDY,PJDZ, &
                        DERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA, &
                        QRPATH)
                ENDIF

             ENDIF
          ENDIF
!!! debug printout
!!!          if(.not.qrepdstr)write(50,*) &
!!!               'EPATHA:DX(krep,lrep)>jrep,krep,lrep,natom=', &
!!!               jrep,krep,lrep,nrep*natrep
!!!          if(.not.qrepdstr)write(50,'(6f12.7)')(dx(i),i=1,nrep*natrep)
!!!          if(.not.qrepdstr)write(50,*)'EPATHA:RA>jrep,krep,lrep,natrep=', &
!!!               jrep,krep,lrep,natrep
!!!          if(.not.qrepdstr)write(50,'(6f12.7)')(ra(1,i),i=1,natrep)
!!!          if(.not.qrepdstr)write(50,*)'EPATHA:RB>jrep,krep,lrep,natrep=', &
!!!               jrep,krep,lrep,natrep
!!!          if(.not.qrepdstr)write(50,'(6f12.7)')(rb(1,i),i=1,natrep)
!!!          if(.not.qrepdstr)write(50,*)'EPATHA:DRA>jrep,krep,lrep,natrep=', &
!!!               jrep,krep,lrep,natrep
!!!          if(.not.qrepdstr)write(50,'(6f12.7)')(dra(1,i),i=1,natrep)
!!!          if(.not.qrepdstr)write(50,*)'EPATHA:DRB>jrep,krep,lrep,natrep=', &
!!!               jrep,krep,lrep,natrep
!!!          if(.not.qrepdstr)write(50,'(6f12.7)')(drb(1,i),i=1,natrep)
!!!
!!!          if(qrepdstr)write(50+irepdstr,*)'EPATHA>kidx=',kidx
!!!          if(qrepdstr)write(50+irepdstr,*)'EPATHA:DX>jr,kr,lr,natom=', &
!!!               jrep,krep,lrep,natrep
!!!          if(qrepdstr)write(50+irepdstr,'(6f12.7)')(dx(i),i=1,natrep)
!!!          if(qrepdstr)write(50+irepdstr,*)'EPATHA:RA>jr,kr,lr,natrep=', &
!!!               jrep,krep,lrep,natrep
!!!          if(qrepdstr)write(50+irepdstr,'(6f12.7)')(ra(1,i),i=1,natrep)
!!!          if(qrepdstr)write(50+irepdstr,*)'EPATHA:RB>jr,kr,lr,natrep=', &
!!!               jrep,krep,lrep,natrep
!!!          if(qrepdstr)write(50+irepdstr,'(6f12.7)')(rb(1,i),i=1,natrep)
!!!          if(qrepdstr)write(50+irepdstr,*)'EPATHA:DRA>jr,kr,lr,natrep=', &
!!!               jrep,krep,lrep,natrep
!!!          if(qrepdstr)write(50+irepdstr,'(6f12.7)')(dra(1,i),i=1,natrep)
!!!          if(qrepdstr)write(50+irepdstr,*)'EPATHA:DRB>jr,kr,lr,natrep=', &
!!!               jrep,krep,lrep,natrep
!!!          if(qrepdstr)write(50+irepdstr,'(6f12.7)')(drb(1,i),i=1,natrep)
!!!          if(qrepdstr)write(50+irepdstr,*)'EPATHA:DXT>jr,kr,lr,natrep=', &
!!!               jrep,krep,lrep,natrep
!!!          if(qrepdstr)write(50+irepdstr,'(6f12.7)')(dxt(i),i=1,natrep)

888       CONTINUE
       ENDDO

#if KEY_REPDSTR==1
       call chmdealloc('epath.src','EPATHA','DZT',NATREP,crl=DZT)
       call chmdealloc('epath.src','EPATHA','DYT',NATREP,crl=DYT)
       call chmdealloc('epath.src','EPATHA','DXT',NATREP,crl=DXT)
#endif 

    EPTHA=EANG

#if KEY_REPDSTR==1
    IF(QREPDSTR)THEN
       CALL PSETGLOB
       CALL GCOMB(FANG,NREP)
       CALL PSETLOC
    ENDIF
    !C      if(.not.qrepdstr)write(50,'(a,5i5)')
    !C     $     'EPATHA:FANG_GCOMB>nrep,natrep=',nrep,natrep
    !C      if(.not.qrepdstr)write(50,'(6f12.7)')(fang(i),i=1,nrep)
    !C      if(qrepdstr)write(50+irepdstr,'(a,5i5)')
    !C     $     'EPATHA:FANG_GCOMB>nrep,natrep=',nrep,natrep
    !C      if(qrepdstr)write(50+irepdstr,'(6f12.7)')(fang(i),i=1,nrep)
#endif 

    RETURN
  END SUBROUTINE EPATHA

  SUBROUTINE EPATHS(NREP,ARMS,FRMS,FANG,ACST,LCYCL,QPRINT)
    !
    ! Print a summary of forces along the pathway.
    !
    use number
    use stream
    use consta

    INTEGER NREP
    real(chm_real) ARMS(*),FRMS(*),FANG(*),ACST(*)
    LOGICAL LCYCL,QPRINT

    INTEGER IREP

    IF(.NOT.QPRINT) RETURN

    WRITE(OUTU,35) 'Summary of forces on path'
35  FORMAT(/,A)
    DO IREP=1,NREP
       WRITE(OUTU,45) IREP,FRMS(IREP),FANG(IREP),ACST(IREP)
45     FORMAT(' Step',I5,' Frad=',F15.6,' Fang=',F15.6,' cosT=',F15.6)
    ENDDO
    WRITE(OUTU,35)

    RETURN
  END SUBROUTINE EPATHS

  SUBROUTINE EPATHO(EPATHR,EPATHA,DX,DY,DZ,X,Y,Z,NREP,KRMS, &
       KMXRMS,RMXRMS,XPREF,YPREF,ZPREF, &
       NATREP,ISTREP,ARMS,BRMS,DRMS,FRMS,WEIGHT,WTOT, &
       QPRINT,LNOTRN,LNOROT,LCYCL,ATOMPR,RA,RB,DRA, &
       DRB,DRTEMP,EVWID,NATOM)

    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !CCC  Compute RMS path restraints neede for RPATH optimizations
    !CCC  H. Lee Woodcock 6/02
    !
    !     Points : i, j, k
    !
    !     Erms = 1/2 * Krms (r_ji - r_jk)**2              I=1,NREP-2
    !
    !     r_ji is the weighted rmsd of the main replica j and reference replica i
    !     r_jk is the weighted rmsd of the main replica j and reference replica k
    !
    !     Forces on r_j due to the reference replicas are computed and applied
    !     to keep the rmsd values of r_ji and r_jk as close as possible
    !
    !     If r_j moves to quickly from R_j then an additional restraint term
    !     is added to prevent point j from unphysical movement
    !
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

    use number
    use stream
    use parallel
    use replica_mod
    use repdstr

    !CCCCCCCCCCCCCCCCCCCCC Variable Declaration CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

    !CCC    Passed Variables   CCCC

    real(chm_real) EPATHR
    real(chm_real) EPATHA
    real(chm_real) DX(*),DY(*),DZ(*)
    real(chm_real) X(*),Y(*),Z(*)
    INTEGER NREP
    real(chm_real) KRMS
    real(chm_real) KMXRMS
    real(chm_real) RMXRMS
    INTEGER NATREP
    INTEGER ISTREP(*)
    real(chm_real) ARMS(*)
    real(chm_real) BRMS(*),DRMS(*)
    real(chm_real) FRMS(*)
    real(chm_real) WEIGHT(*)
    real(chm_real) WTOT
    LOGICAL QPRINT
    LOGICAL LNOTRN
    LOGICAL LNOROT
    LOGICAL LCYCL
    INTEGER ATOMPR(2,NATREP)
    real(chm_real) RA(*),RB(*),DRA(*),DRB(*),DRTEMP(*)
    real(chm_real) EVWID
    INTEGER NATOM
    real(chm_real) XPREF(*),YPREF(*),ZPREF(*)

    !CCC  Local Variables   CCCC

    INTEGER NREPM,LPEER,LPEER2,RPEER,RPEER2
    INTEGER IREP,JREP,KREP,LREP,IPT,JPT,KPT,I,J,K
    real(chm_real)  RMST,R2,RNRM,ERMS,ERMAX,TMP,AX,AY,AZ
    real(chm_real)  ATMP,BTMP,CTMP,DTMP,ETMP ! Temp variables
    real(chm_real) ADERMS,BDERMS,DDERMS      ! Derivative variables
    LOGICAL LPRINT
    real(chm_real) DEVA(3,3)
    !
    !-----------------------------------------------------------------
#if KEY_REPDSTR==1
    !     We need to communicate MAIN and COMP set for EPATHO.
    !
    !     NREPM for parallel communication is always NREP-1, regardless of
    !     LCYCL. Later in this routine NREPM depends on LCYCL
    NREPM=NREP-1
    IF(QREPDSTR) THEN
       !
       !     This is not yet extended for parallel/parallel
       !     This should work for cyclic case, too:
       LPEER=MOD(NREPDSTR+IREPDSTR-1,NREPDSTR)
       RPEER=MOD(NREPDSTR+IREPDSTR+1,NREPDSTR)
       LPEER2=MOD(NREPDSTR+IREPDSTR-2,NREPDSTR)
       RPEER2=MOD(NREPDSTR+IREPDSTR+2,NREPDSTR)
       !C         JREP=IREPDSTR+1
       !
       !     Swap the data between the neighbors. This is a double
       !     direction swap since some machines can do it as fast as
       !     a single one. And we might need the other neighbor for angles
       !     Currently for angles there is a problem, because we need I, I+1, I+2.
       !     Maybe we can shift it so it would give the same result also for:
       !     I-1, I, I+1
       !     Or maybe we call this routine once again with exchanged parameters:
       !     previous XPR becomes XPL.... Or write another routine, which goes in
       !     one direction only (maybe GRECSENR() ???)
       !
       !     X(*)       - current replica
       !     XPEER(*,1) - current - 2
       !     XPEER(*,2) - current - 1
       !     XPEER(*,3) - current  = X(*)
       !     XPEER(*,4) - current + 1
       !     XPEER(*,5) - current + 2
       !
       !C         write(50+irepdstr,*)
       !C     $        'EPATHR>before swapd:irepdstr,lp,rp,lp2,rp2=',
       !C     $        irepdstr,lpeer,rpeer,lpeer2,rpeer2
       !
       DO I=1,NATREP
          XPEERS(I,3)=X(I)
          YPEERS(I,3)=Y(I)
          ZPEERS(I,3)=Z(I)
          XPEERC(I,3)=XPREF(I)
          YPEERC(I,3)=YPREF(I)
          ZPEERC(I,3)=ZPREF(I)
       ENDDO

       DO I=1,5
          APEERS(I)=ZERO
       ENDDO
       !
       !     Get the neighbor's coordinates. In order to do all the forces for
       !     current replica we need coordinates from both: i-1, and i+1
       !
       !     MARK: I think we dont need any swapd here...
       !           Put the [XYZ]PEERC out of the simulation loop in the
       !           calling routine! No communication needed during
       !           simulation!!!!??
       !
       CALL SWAPD(NATREP,X,Y,Z,XPEERS(1,2),YPEERS(1,2),ZPEERS(1,2), &
            XPEERS(1,4),YPEERS(1,4),ZPEERS(1,4),LPEER,RPEER)

       CALL SWAPD(NATREP,X,Y,Z,XPEERS(1,1),YPEERS(1,1),ZPEERS(1,1), &
            XPEERS(1,5),YPEERS(1,5),ZPEERS(1,5),LPEER2,RPEER2)

       CALL SWAPD(NATREP,XPREF,YPREF,ZPREF, &
            XPEERC(1,2),YPEERC(1,2),ZPEERC(1,2), &
            XPEERC(1,4),YPEERC(1,4),ZPEERC(1,4),LPEER,RPEER)
       CALL SWAPD(NATREP,XPREF,YPREF,ZPREF, &
            XPEERC(1,1),YPEERC(1,1),ZPEERC(1,1), &
            XPEERC(1,5),YPEERC(1,5),ZPEERC(1,5),LPEER2,RPEER2)
       !
       !C         write(50+irepdstr,*)'EPATHR>after swapd...'
    ENDIF

#endif 
    !
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !
    IF(LCYCL) THEN
       NREPM=NREP
    ELSE
       NREPM=NREP-2
    ENDIF
    LPRINT = QPRINT .AND. (PRNLEV.GT.7)

    ERMS=ZERO                  ! Sets initial E_rms = 0.0
    ERMAX=ZERO
    !
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !CCC  Loop to compute best fit of adjacent points i,j,k and compute forces CCC
    !CCC  of the points                                                        CCC
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

    DO JREP=1,NREPM

#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          !CC MARK: this is not working:  IF((JREP-1).NE.IREPDSTR)GOTO 888
          IF(JREP.NE.IREPDSTR)GOTO 888
       ENDIF
#endif 

       KREP=JREP+1
       IF(KREP.GT.NREP) KREP=KREP-NREP
       LREP=JREP+2
       IF(LREP.GT.NREP) LREP=LREP-NREP

       !CCC  Setup pointers for points j and i
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          IPT=0
          JPT=0
       ELSE
#endif 
          JPT=ISTREP(KREP)
          IPT=ISTREP(JREP)
#if KEY_REPDSTR==1
       ENDIF                    
#endif
       DO I=1,NATREP
          JPT=JPT+1
          IPT=IPT+1
          ATOMPR(1,I)=JPT
          ATOMPR(2,I)=IPT
       ENDDO

       !CCC  Call the best-fit command using replicas j and i
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          CALL ECBSTF(X,Y,Z,XPEERC(:,2),YPEERC(:,2),ZPEERC(:,2), &
               ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,ATMP,RA,RB,DEVA,EVWID)
       ELSE
#endif 

          CALL ECBSTF(X,Y,Z,XPREF,YPREF,ZPREF,ATOMPR,NATREP, &
               LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,ATMP,RA,RB,DEVA,EVWID)
#if KEY_REPDSTR==1
       ENDIF           
#endif

       ATMP=ATMP/WTOT
       IF(ATMP.LT.ZERO) ATMP=ZERO
       ATMP=SQRT(ATMP)
       !CC MARK: this might be a problem since we are using
       !CC       different indexing then non REPDSTR code
       !CC       check with printouts!!!
       ARMS(JREP)=ATMP

       !CCC  Setup pointers for point k, point j is already setup
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          KPT=0
       ELSE
#endif 
          KPT=ISTREP(LREP)
#if KEY_REPDSTR==1
       ENDIF               
#endif

       DO I=1,NATREP
          KPT=KPT+1
          ATOMPR(2,I)=KPT
       ENDDO

       !CCC  Call the best-fit command using replicas j and k
#if KEY_REPDSTR==1
       IF(QREPDSTR) THEN
          CALL ECBSTF(X,Y,Z,XPEERC(:,4),YPEERC(:,4),ZPEERC(:,4), &
               ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,BTMP,RA,RB,DEVA,EVWID)
       ELSE
#endif 
          CALL ECBSTF(X,Y,Z,XPREF,YPREF,ZPREF,ATOMPR,NATREP, &
               LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,BTMP,RA,RB,DEVA,EVWID)
#if KEY_REPDSTR==1
       ENDIF              
#endif
       BTMP=BTMP/WTOT
       IF(BTMP.LT.ZERO) BTMP=ZERO
       BTMP=SQRT(BTMP)
       BRMS(JREP)=BTMP

       !CCC  Compute the RMS Energy

       CTMP=ATMP-BTMP         ! RMSD between r_ji and r_jk

       IF(CTMP**2.LT.ATMP*BTMP*100) THEN

          ERMS=ERMS+KRMS*(ARMS(JREP) - BRMS(JREP))**2

          !CCC  Take the derivative of E w.r.t. r_ji and r_jk, (dE/rdr)

          ADERMS=KRMS*((ATMP-BTMP)/ATMP)
          BDERMS=KRMS*((BTMP-ATMP)/BTMP)

          !CCC  Get force for the r_jk pair of replicas
          !CC
          !CC  MARK: Do we really dont need to do anything here???
          !CC
          CALL ECFORC(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
               .TRUE.,DX,DY,DZ,.FALSE., (/ ZERO /), (/ ZERO /), (/ ZERO /), &
               BDERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA)

          !CCC  Redo the first Best-Fit procedure
          !CCC  Setup pointers for points j and i
#if KEY_REPDSTR==1
          IF(QREPDSTR)THEN
             IPT=0
          ELSE
#endif 
             IPT=ISTREP(JREP)
#if KEY_REPDSTR==1
          ENDIF               
#endif

          DO I=1,NATREP
             IPT=IPT+1
             ATOMPR(2,I)=IPT
          ENDDO

          !CCC  Call the best-fit command using replicas j and i
#if KEY_REPDSTR==1
          IF(QREPDSTR)THEN
             CALL ECBSTF(X,Y,Z,XPEERC(:,2),YPEERC(:,2),ZPEERC(:,2), &
                  ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
                  WTOT,ATMP,RA,RB,DEVA,EVWID)
          ELSE
#endif 
             CALL ECBSTF(X,Y,Z,XPREF,YPREF,ZPREF,ATOMPR,NATREP, &
                  LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
                  WTOT,ATMP,RA,RB,DEVA,EVWID)
#if KEY_REPDSTR==1
          ENDIF               
#endif

          ATMP=ATMP/WTOT
          IF(ATMP.LT.ZERO) ATMP=ZERO
          ATMP=SQRT(ATMP)
          !
          !     MARK:   Possible problem with JREP in ARMS()??
          !
          ARMS(JREP)=ATMP

          !CCC  Get the force for the r_ji pair of replicas

          CALL ECFORC(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
               .TRUE.,DX,DY,DZ,.FALSE., (/ ZERO /), (/ ZERO /), (/ ZERO /), &
               ADERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA)

       ENDIF

       !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
       !CCC  Add restraint to prevent too big of a step along the hyper-plane  CCCC
       !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

       !CCC  Add restraint to best-fit r_j to R_j(Ref)
       !CCC  Reuse the IPT and JPT pointers
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          IPT=0
       ELSE
#endif 
          IPT=ISTREP(KREP)
#if KEY_REPDSTR==1
       ENDIF               
#endif

       DO I=1,NATREP
          IPT=IPT+1
          ATOMPR(2,I)=IPT
       ENDDO

       !CCC  Call the best-fit command using replicas j and j(Ref)
#if KEY_REPDSTR==1
       !CC
       !CC   MARK: We probaly dont need to change this here for REPDSTR
       IF(QREPDSTR)THEN
          CALL ECBSTF(X,Y,Z,XPEERC(:,3),YPEERC(:,3),ZPEERC(:,3), &
               ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,DTMP,RA,RB,DEVA,EVWID)
       ELSE
#endif 
          CALL ECBSTF(X,Y,Z,XPREF,YPREF,ZPREF,ATOMPR,NATREP, &
               LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,DTMP,RA,RB,DEVA,EVWID)
#if KEY_REPDSTR==1
       ENDIF      
#endif

       !CC   MARK: JREP problem in DRMS()????
       DTMP=DTMP/WTOT
       IF(DTMP.LT.ZERO) DTMP=ZERO
       DTMP=SQRT(DTMP)
       DRMS(JREP)=DTMP

       !CCC------------------------------------------------------

       IF(DRMS(JREP).GT.RMXRMS) THEN

          !CCC  Compute the derivative : (dE/rdr)

          IF(RMXRMS.GT.ZERO) THEN
             DDERMS=KMXRMS*(DRMS(JREP) - RMXRMS)/DRMS(JREP)
          ELSE
             DDERMS=KMXRMS
          ENDIF

          !CCC  Compute forces on r_j due to R_j

          CALL ECFORC(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
               .TRUE.,DX,DY,DZ,.FALSE., (/ ZERO /), (/ ZERO /), (/ ZERO /), &
               DDERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA)

          ERMAX=ERMAX+(KRMS*(DRMS(JREP)-RMXRMS)**2)

       ELSE

          !           WRITE(OUTU,25) '  No need to best fit point to Ref Path'

       ENDIF

       !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

       IF(QPRINT) WRITE(OUTU,57) JREP-1,JREP,ATMP,JREP,JREP+1,BTMP, &
            CTMP,DTMP
57     FORMAT('In EPATHO: RMS of Replicas:',I3,' -',I2,' RMSd =',F9.5, &
            I3,' -',I2,' RMSd =',F9.5, &
            ' Plane Dev. =',F10.5, &
            ' Curr-Ref RMSd =',F9.5)

       !     End of main loop: JREP=1,NREPM
888    CONTINUE
    ENDDO

    EPATHR=HALF*ERMS
    EPATHA=HALF*ERMAX

    IF(LPRINT) THEN

       WRITE(OUTU,80) EPATHR
       WRITE(OUTU,81) EPATHA

    ENDIF

80  FORMAT(' Out-of-plane RMS energy is : ',F14.5)
81  FORMAT(' In-of-plane RMS energy is : ',F14.5)

58  FORMAT('  In EPATHO: rms between current and reference', &
         ' replicas:',I3,F10.5)

25  FORMAT(A)

    RETURN
  END SUBROUTINE EPATHO

  SUBROUTINE EPATHO1(EPATHR,DX,DY,DZ,X,Y,Z,NREP,KRMS,KMXRMS,RMXRMS, &
       REFPX,REFPY,REFPZ, &
       NATREP,ISTREP,ARMS,BRMS,DRMS,FRMS, &
       RPMF,EPMF,WEIGHT,WTOT, &
       QPRINT,LNOTRN,LNOROT,LCYCL,ATOMPR,RA,RB,DRA, &
       DRB,DRTEMP,EVWID,DRATIO,QPRPMF)

    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !CCC  Compute RMS path restraints neede for RPATH optimizations
    !CCC  H. Lee Woodcock 6/02
    !     Jhih-Wei Chu    5/03
    !
    !     Points : j-1, j, j+1
    !
    !     Erms = 1/2 * Krms ((1-dratio)*r(j,j-1)-dratio*r(j,j+1))**2              J=2,NREP-1
    !
    !     r(j,j-1) is the weighted rmsd of the main replica j and reference replica j-1
    !     r(j,j+1) is the weighted rmsd of the main replica j and reference replica j+1
    !
    !     Forces on the j-th replica due to the reference replicas are computed and applied
    !     to keep r(j,j-1)/r(j,j+1)=(1-dratio)/dratio
    !
    !     If the j-th replica moves to far from its original position then
    !     an additional restraint term
    !     is added to prevent point j from unphysical movement
    !
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

    use number
    use stream
    use parallel

    !CCCCCCCCCCCCCCCCCCCCC Variable Declaration CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

    !CCC    Passed Variables   CCCC

    LOGICAL QPRINT,LNOTRN,LNOROT,LCYCL,QPRPMF
    INTEGER NREP
    INTEGER NATREP
    INTEGER ISTREP(*)
    INTEGER ATOMPR(2,NATREP)
    real(chm_real) EPATHR
    real(chm_real) DX(*),DY(*),DZ(*)
    real(chm_real) X(*),Y(*),Z(*),REFPX(*),REFPY(*),REFPZ(*)
    real(chm_real) KRMS,KMXRMS,RMXRMS,WTOT,DRATIO,EVWID
    real(chm_real) ARMS(*),BRMS(*),DRMS(*),FRMS(*)
    real(chm_real) RPMF(*),EPMF(*),WEIGHT(*)
    real(chm_real) RA(*),RB(*),DRA(*),DRB(*),DRTEMP(*)

    !CCC  Local Variables   CCCC

    INTEGER NREPM
    INTEGER IREP,JREP,KREP,LREP,IPT,JPT,KPT,I,J,K
    INTEGER JP1,JM1,IBEG,IEND
    real(chm_real)  RMST,R2,RNRM,ERMS,TMP,AX,AY,AZ
    real(chm_real)  ATMP,BTMP,CTMP,DTMP ! Temp variables
    real(chm_real) ADERMS,BDERMS,DDERMS ! Derivative variables
    LOGICAL LPRINT,QRPATH
    real(chm_real) DEVA(3,3)

    !CCC  Settings

    IF(LCYCL) THEN
       NREPM=NREP
       IBEG=1
       IEND=NREPM
    ELSE
       NREPM=NREP-1
       IBEG=2
       IEND=NREPM
    ENDIF

    LPRINT = QPRINT .AND. (PRNLEV.GT.7)

    QRPATH=.TRUE.

    ERMS=ZERO                  ! Sets initial E_rms = 0.0

    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !CCC  Loop to compute best fit of adjacent points i,j,k and compute forces CCC
    !CCC  of the points                                                        CCC
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

    DO JREP=IBEG,IEND
       JM1=JREP-1
       IF(JM1.LT.1) JM1=NREP
       JP1=JREP+1
       IF(JP1.GT.NREP) JP1=JP1-NREP

       !CCC  Setup pointers for points j and i

       IPT=ISTREP(JREP)
       JPT=ISTREP(JM1)
       DO I=1,NATREP
          IPT=IPT+1
          JPT=JPT+1
          ATOMPR(1,I)=IPT
          ATOMPR(2,I)=JPT
       ENDDO

       !CCC  Call the best-fit command using replicas j and i

       CALL ECBSTF(X,Y,Z,REFPX,REFPY,REFPZ,ATOMPR,NATREP, &
            LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
            WTOT,ATMP,RA,RB,DEVA,EVWID)

       ATMP=ATMP/WTOT
       IF(ATMP.LT.ZERO) ATMP=ZERO
       ATMP=SQRT(ATMP)
       ARMS(JREP)=ATMP

       !CCC  Setup pointers for point k, point j is already setup

       KPT=ISTREP(JP1)
       DO I=1,NATREP
          KPT=KPT+1
          ATOMPR(2,I)=KPT
       ENDDO

       !CCC  Call the best-fit command using replicas j and k

       CALL ECBSTF(X,Y,Z,REFPX,REFPY,REFPZ,ATOMPR,NATREP, &
            LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
            WTOT,BTMP,RA,RB,DEVA,EVWID)

       BTMP=BTMP/WTOT
       IF(BTMP.LT.ZERO) BTMP=ZERO
       BTMP=SQRT(BTMP)
       BRMS(JREP)=BTMP

       !CCC  Compute the RMS Energy

       CTMP=(ONE-DRATIO)*ARMS(JREP)-DRATIO*BRMS(JREP)

       RPMF(JREP)=-KRMS*CTMP*(ARMS(JREP)+BRMS(JREP))
       EPMF(JREP)=HALF*KRMS*CTMP**2

       ERMS=ERMS+KRMS*CTMP**2

       !CCC  Take the derivative of E (dE/rdr)

       ADERMS=KRMS*CTMP/ARMS(JREP)*(ONE-DRATIO)
       BDERMS=-KRMS*CTMP/BRMS(JREP)*DRATIO

       IF(QPRINT) WRITE(OUTU,57) JREP,ATMP,BTMP,CTMP, &
            ADERMS,BDERMS
57     FORMAT('  In EPATHO1: rms between replicas:',I5,3F14.5, &
            '  Force=',2F14.5)

       !CCC  Get force for the r_jk pair of replicas

       CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
            .TRUE.,DX,DY,DZ,.FALSE., (/ ZERO /), (/ ZERO /), (/ ZERO /), &
            BDERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA, &
            QRPATH)

       !CCC  Redo the first Best-Fit procedure
       !CCC  Setup pointers for points j and i

       JPT=ISTREP(JM1)
       DO I=1,NATREP
          JPT=JPT+1
          ATOMPR(2,I)=JPT
       ENDDO

       !CCC  Call the best-fit command using replicas j and i

       CALL ECBSTF(X,Y,Z,REFPX,REFPY,REFPZ,ATOMPR,NATREP, &
            LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
            WTOT,ATMP,RA,RB,DEVA,EVWID)

       !CCC  Get the force for the r_ji pair of replicas

       CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
            .TRUE.,DX,DY,DZ,.FALSE., (/ ZERO /), (/ ZERO /), (/ ZERO /), &
            ADERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA, &
            QRPATH)

       !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
       !CCC  Add restraint to prevent too big of a step along the hyper-plane  CCCC
       !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

       !CCC  Add restraint to best-fit r_j to R_j(Ref)
       !CCC  Reuse the IPT and JPT pointers

       IPT=ISTREP(JREP)
       DO I=1,NATREP
          IPT=IPT+1
          ATOMPR(2,I)=IPT
       ENDDO

       !CCC  Call the best-fit command using replicas j and j(Ref)

       CALL ECBSTF(X,Y,Z,REFPX,REFPY,REFPZ,ATOMPR,NATREP, &
            LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
            WTOT,DTMP,RA,RB,DEVA,EVWID)

       DTMP=DTMP/WTOT
       IF(DTMP.LT.ZERO) DTMP=ZERO
       DTMP=SQRT(DTMP)
       DRMS(JREP)=DTMP

       IF(DRMS(JREP).GT.RMXRMS) THEN

          !CCC  Compute the derivative : (dE/rdr)

          IF(RMXRMS.GT.ZERO) THEN
             DDERMS=KRMS*(DRMS(JREP) - RMXRMS)/DRMS(JREP)
             !CCC  Compute forces on r_j due to R_j

             CALL ECFORCA(ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, &
                  .TRUE.,DX,DY,DZ,.FALSE., (/ ZERO /), (/ ZERO /), (/ ZERO /), &
                  DDERMS,WEIGHT,WTOT,RA,RB,DRA,DRB,DRTEMP,DEVA, &
                  QRPATH)

             ERMS=ERMS+KRMS*(DRMS(JREP)-RMXRMS)**2
             EPMF(JREP)=EPMF(JREP)+HALF*KRMS*(DRMS(JREP)-RMXRMS)**2

          ENDIF

       ENDIF

       !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

    ENDDO

    EPATHR=HALF*ERMS

#if KEY_PARALLEL==1
    IF(MYNOD.EQ.0.AND.QPRPMF)THEN
       WRITE(OUTU,131)
131    FORMAT(' I=','Restraint  Energy',' Restraint  PMF', &
            '            ARMS','            BRMS','            DRMS')
       DO I=1,NREP
          WRITE(OUTU,130) I,EPMF(I),RPMF(I),ARMS(I),BRMS(I),DRMS(I)
       ENDDO
130    FORMAT(I3,1X,5F16.6)
    ENDIF
#else /**/
    IF(QPRPMF) THEN
       WRITE(OUTU,131)
131    FORMAT(' I=','   Restraint  Energy','      Restraint  PMF', &
            '            ARMS','            BRMS','            DRMS')
       DO I=1,NREP
          WRITE(OUTU,130) I,EPMF(I),RPMF(I),ARMS(I),BRMS(I),DRMS(I)
       ENDDO
130    FORMAT(I3,1X,5F20.6)
    ENDIF
#endif 

25  FORMAT(A)

    RETURN
  END SUBROUTINE EPATHO1

  SUBROUTINE REFRMS(BFARY,NREP, &
       ISTREP,ARMS,BRMS,DRMS,FRMS,WEIGHT,WTOT, &
       QPRINT,LNOTRN,LNOROT,LCYCL,ATOMPR,RA,RB,DRA, &
       DRB,DRTEMP,NATOM)

    use number
    use stream
    use pathm
    use parallel
    use repdstr

    !CCC  Passed Variables   CCCC

    real(chm_real) BFARY(3,*)
    INTEGER NREP
    INTEGER ISTREP(*)
    real(chm_real) ARMS(*)
    real(chm_real) BRMS(*),DRMS(*)
    real(chm_real) FRMS(*)
    real(chm_real) WEIGHT(*)
    real(chm_real) WTOT
    LOGICAL QPRINT
    LOGICAL LNOTRN
    LOGICAL LNOROT
    LOGICAL LCYCL
    INTEGER ATOMPR(2,*)
    real(chm_real) RA(3,*),RB(3,*)
    real(chm_real) DRA(*),DRB(*),DRTEMP(*)
    INTEGER NATOM

    !CCC  Local Variables   CCCC

    INTEGER NREPM,LPEER,RPEER,IFLG,JFLG
    INTEGER IREP,JREP,KREP,LREP,IPT,JPT,KPT,I,J,K,L
    real(chm_real)  RMST,R2,RNRM,ERMS,ERMAX,TMP,AX,AY,AZ
    real(chm_real)  FACT                                          ! lw050714
    real(chm_real)  ATMP,BTMP,CTMP,DTMP,ETMP ! Temp variables
    real(chm_real) ADERMS,BDERMS,DDERMS ! Derivative variables
    LOGICAL LPRINT
    real(chm_real) DEVA(3,3)

    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !     Check for cyclic path
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

    !lw      IF(LCYCL) THEN
    !lw         NREPM=NREP
    !lw      ELSE
    !lw         NREPM=NREP-2
    !lw      ENDIF
    LPRINT = QPRINT .AND. (PRNLEV.GT.7)

    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
#if KEY_REPDSTR==1
    IF(QREPDSTR)THEN
       !     For EPATHO we need to communicate COMP set only.
       !
       !     This is not yet extended for parallel/parallel
       !     This should work for cyclic case, too:
       LPEER=MOD(NREP+IREPDSTR-1,NREP)
       RPEER=MOD(NREP+IREPDSTR+1,NREP)

       !CC         write(50+mynodg,*)'EPATH>in refrms()'
       DO I=1,NATREP
          XPEERC(I,3)=XPREF(I)
          YPEERC(I,3)=YPREF(I)
          ZPEERC(I,3)=ZPREF(I)
       ENDDO
       !CC
       !C         DO I=1,5
       !C            APEERS(I)=ZERO
       !C         ENDDO
       !
       !     Get the neighbor's reference coordinates. In order to do all the
       !     forces for current replica we need coordinates from both: i-1, and
       !     i+1
       !
       !     MARK: I think all we need to communicate for EPATHO() is this
       !           one time call. This routine gets called only once per simulation!!
       !
       CALL SWAPD(NATREP,XPREF,YPREF,ZPREF, &
            XPEERC(1,2),YPEERC(1,2),ZPEERC(1,2), &
            XPEERC(1,4),YPEERC(1,4),ZPEERC(1,4),LPEER,RPEER)

    ENDIF
#endif 

    DO KREP=1,NREP
       !        KREP=JREP+1
       !        IF(KREP.GT.NREP) KREP=KREP-NREP
       !        LREP=JREP+2
       !        IF(LREP.GT.NREP) LREP=LREP-NREP
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          IF((KREP-1).NE.IREPDSTR)GOTO 888
       ENDIF
#endif 

       FACT=HALF
       JREP=KREP-1
       IFLG=0
       JFLG=0
       IF (JREP.LT.1) THEN
          IF(LCYCL) THEN
             JREP=NREP
          ELSE
             JREP=KREP
             FACT=ONE
             IFLG=1
          ENDIF
       ENDIF

       LREP=KREP+1
       IF (LREP.GT.NREP) THEN
          IF(LCYCL) THEN
             LREP=1
          ELSE
             LREP=NREP
             FACT=ONE
             JFLG=-1
          ENDIF
       ENDIF

       !CCC  Setup pointers for points j and i
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          JPT=0
          IPT=0
       ELSE
#endif 
          JPT=ISTREP(KREP)
          IPT=ISTREP(JREP)
#if KEY_REPDSTR==1
       ENDIF               
#endif

       DO I=1,NATREP
          JPT=JPT+1
          IPT=IPT+1
          ATOMPR(1,I)=JPT
          ATOMPR(2,I)=IPT
       ENDDO

       !CCC  Call the best-fit command using replicas j and i
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          CALL ECBSTF(XPREF,YPREF,ZPREF, &
               XPEERC(:,2+IFLG),YPEERC(:,2+IFLG),ZPEERC(:,2+IFLG), &
               ATOMPR,NATREP, &
               LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,ATMP,RA,RB,DEVA,EVWID)
       ELSE
#endif 
          CALL ECBSTF(XPREF,YPREF,ZPREF,XPREF,YPREF,ZPREF, &
               ATOMPR,NATREP, &
               LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,ATMP,RA,RB,DEVA,EVWID)
#if KEY_REPDSTR==1
       ENDIF                      
#endif

       ATMP=SQRT(ATMP/WTOT)
       BFREFJI(KREP)=ATMP

       DO J=1,NATREP
          DO K=1,3
             BFARY(K,J)=ZERO
             DO L=1,3
                BFARY(K,J)=BFARY(K,J)-DEVA(K,L)*RB(L,J)
             ENDDO
          ENDDO
       ENDDO

       !CCC  Setup pointers for point k, point j is already setup
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          KPT=0
       ELSE
#endif 
          KPT=ISTREP(LREP)
#if KEY_REPDSTR==1
       ENDIF                    
#endif

       DO I=1,NATREP
          KPT=KPT+1
          ATOMPR(2,I)=KPT
       ENDDO
       !
       !CCC  Call the best-fit command using replicas j and k
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          CALL ECBSTF(XPREF,YPREF,ZPREF, &
               XPEERC(:,4+JFLG),YPEERC(:,4+JFLG),ZPEERC(:,4+JFLG), &
               ATOMPR,NATREP, &
               LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,BTMP,RA,RB,DEVA,EVWID)
       ELSE
#endif 
          CALL ECBSTF(XPREF,YPREF,ZPREF,XPREF,YPREF,ZPREF, &
               ATOMPR,NATREP, &
               LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,BTMP,RA,RB,DEVA,EVWID)
#if KEY_REPDSTR==1
       ENDIF                 
#endif
       BTMP=SQRT(BTMP/WTOT)
       BFREFJK(KREP)=BTMP
       CTMP=ZERO
       !C         write(50+irepdstr,*)'REFRMS>krep,btmp,wtot=',krep,btmp,wtot

       DO J=1,NATREP
          DO K=1,3
             DO L=1,3
                BFARY(K,J)=BFARY(K,J)+DEVA(K,L)*RB(L,J)
             ENDDO
             CTMP=CTMP+WEIGHT(J)*(BFARY(K,J))**2
          ENDDO
       ENDDO
       !C         write(50+irepdstr,*)'REFRMS>krep,ctmp,wtot=',krep,ctmp,wtot

       BFREFIK(KREP)=SQRT(CTMP/WTOT)

       !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          JPT=0
       ELSE
#endif 
          JPT=ISTREP(KREP)
#if KEY_REPDSTR==1
       ENDIF             
#endif
       DO J=1,NATREP
          JPT=JPT+1
          XPDIR(JPT) = FACT*BFARY(1,J)
          YPDIR(JPT) = FACT*BFARY(2,J)
          ZPDIR(JPT) = FACT*BFARY(3,J)
       ENDDO

!!!       write(50+irepdstr,'(a,12f12.5)')'BFARY=',(bfary(1,j),j=1,12)
!!!       write(50+irepdstr,*)'FACT=',FACT
!!!       write(50+irepdstr,'(a,i5,12f12.5)')'REFRMS>krep,xpdir=', &
!!!            krep,(xpdir(j),j=1,12)
!!!       write(50+irepdstr,'(a,i5,12f12.5)')'REFRMS>krep,ypdir=', &
!!!            krep,(ypdir(j),j=1,12)
!!!       write(50+irepdstr,'(a,i5,12f12.5)')'REFRMS>krep,zpdir=', &
!!!            krep,(zpdir(j),j=1,12)


11     FORMAT('TEST : ',F15.6)
50     FORMAT('Best-Fit Array:  ', F15.6)

       !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

!!!       write(50+irepdstr,*)'REFRMS>krep,bfrefik=',krep,bfrefik(krep)
!!!       write(50+irepdstr,*)'REFRMS>krep,bfrefjk=',krep,bfrefjk(krep)
!!!       write(50+irepdstr,*)'REFRMS>krep,bfrefji=',krep,bfrefji(krep)


888    CONTINUE
    ENDDO

    RETURN
  END SUBROUTINE REFRMS

  SUBROUTINE PATHDYNA(DX,DY,DZ,NREP,QPRINT,LCYCL,WEIGHT,WTOT, &
       ISTREP,BFARY,LNOTRN,LNOROT,ATOMPR,RA,RB,DRA, &
       DRB,DRTEMP,NATOM)
    use number
    use stream
    use parallel
    use pathm
    use coord
    use replica_mod
    use repdstr

    !CCC  Passed Variables   CCCC

    real(chm_real) DX(*),DY(*),DZ(*)
    INTEGER NREP
    LOGICAL QPRINT
    LOGICAL LCYCL
    real(chm_real) WEIGHT(*),WTOT
    INTEGER NATOM
    INTEGER ISTREP(*)
    real(chm_real) BFARY(3,*)
    LOGICAL LNOTRN
    LOGICAL LNOROT
    INTEGER ATOMPR(2,*)
    real(chm_real) RA(3,*),RB(3,*)
    real(chm_real) DRA(*),DRB(*),DRTEMP(*)

    !CCC  Local Variables   CCCC

    INTEGER NREPM
    INTEGER JREP,KREP,LREP,IPT,I,J,K,L
    real(chm_real)  RMST,R2,RNRM,ERMS,ERMAX,TMP,AX,AY,AZ
    real(chm_real)  ATMP,BTMP,CTMP,DTMP,ETMP ! Temp variables
    LOGICAL LPRINT
    real(chm_real) RDX,RDY,RDZ
    real(chm_real) DFORD,DBACK,DTOT(100)
    real(chm_real) DCORR(100),REFANG(100),CURANG(100)              ! lw050714
    real(chm_real) DEVA(3,3)
    real(chm_real) JI,JK

    IF(LCYCL) THEN
       NREPM=NREP
    ELSE
       NREPM=NREP-1
    ENDIF
    LPRINT = QPRINT .AND. (PRNLEV.GT.7)

    NPCALL = NPCALL + 1

    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !CCC  Compute Pointers, Work, and Work**2. Sum these and Determine the   CCCCC
    !CCC  Potential of Mean Force (PMF) of the Path                          CCCCC
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

    PATHWORK(1:NREP)=zero

    DO JREP=1,NREPM
       KREP=JREP+1
#if KEY_REPDSTR==1
       !CC
       !CC   MARK: Watch out here!!! this code is funny!
       !CC         jrep=1,nrep-1, but everything in the loop uses
       !CC         krep, which is jrep+1, so it could be without krep
       !           if loop would be from 2,nrep ????
       !           For now we use KREP to choose executing replica/node
       !
       IF(QREPDSTR)THEN
          IF((KREP-1).NE.IREPDSTR)GOTO 888
       ENDIF
#endif 
       IF(KREP.GT.NREP) KREP=KREP-NREP
       !lw         LREP=JREP+2
       !lw         IF(LREP.GT.NREP) LREP=LREP-NREP

       !     -----------------------------------------------------
       !     CALL THE BEST-FIT COMMAND USING REPLICAS J AND J(REF)
       !     -----------------------------------------------------
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          IPT=0
       ELSE
#endif 
          IPT=ISTREP(KREP)
#if KEY_REPDSTR==1
       ENDIF                          
#endif

       DO I=1,NATREP
          IPT=IPT+1
          ATOMPR(1,I)=IPT
          ATOMPR(2,I)=IPT
       ENDDO
       !     MARK:   We dont need anything here for REPDSTR???
       CALL ECBSTF(X,Y,Z,XPREF,YPREF,ZPREF,ATOMPR,NATREP, &
            LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
            WTOT,ETMP,RA,RB,DEVA,EVWID)

       !-------------------------------------------------------------------------

       PATHWORK(KREP)=ZERO
       XPROJV=ZERO
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          IPT=0
       ELSE
#endif 
          IPT=ISTREP(KREP)
#if KEY_REPDSTR==1
       ENDIF                         
#endif

       IF(QCURVC) THEN
          IF (KREP.EQ.2) THEN
             DBACK=(BFIJ(KREP)**2 - BFJJ(KREP)**2)
          ELSE
             DBACK=(BFIJ(KREP)**2 - BFJJ(KREP)**2)/(2*BFREFJI(KREP))
          ENDIF

          IF (KREP.EQ.NREP) THEN
             DFORD=(BFKJ(KREP)**2 - BFJJ(KREP)**2)
          ELSE
             DFORD=(BFKJ(KREP)**2 - BFJJ(KREP)**2)/(2*BFREFJK(KREP))
          ENDIF

          DTOT(KREP)=TWO*(DFORD+DBACK)/(BFREFJK(KREP)+BFREFJI(KREP))

          DCORR(KREP)=0.5*((BFIJ(KREP)-BFREFJI(KREP))+ &
               (BFKJ(KREP)-BFREFJK(KREP)))
       ENDIF

       ! Use Law of cosines to compute angles
       !
       DO I=1,NATREP
          IPT=IPT+1
          IF(WEIGHT(I).GT.ZERO) THEN

             RDX = DX(IPT)*DEVA(1,1)+DY(IPT)*DEVA(2,1)+DZ(IPT)*DEVA(3,1)
             RDY = DX(IPT)*DEVA(1,2)+DY(IPT)*DEVA(2,2)+DZ(IPT)*DEVA(3,2)
             RDZ = DX(IPT)*DEVA(1,3)+DY(IPT)*DEVA(2,3)+DZ(IPT)*DEVA(3,3)

             !     ------------------------
             !     CALCULATE THE PROJECTION
             !     ------------------------

             XPROJV=XPDIR(IPT)*RDX+YPDIR(IPT)* &
                  RDY+ZPDIR(IPT)*RDZ

             !     -------------------------------------------------------
             !     COMPUTE PATHWORK WITH AND WITHOUT CURVATURE CORRECTIONS
             !     -------------------------------------------------------

             IF(QCURVC) THEN
                PATHWORK(KREP) = PATHWORK(KREP) + (XPROJV*DTOT(KREP)) &
                     +(XPROJV*DCORR(KREP))
             ELSEIF(QNOCURVC) THEN
                PATHWORK(KREP) = PATHWORK(KREP) + XPROJV
             ELSE
                PATHWORK(KREP) = PATHWORK(KREP) + (XPROJV* &
                     ((BFREFJK(KREP)+BFREFJI(KREP))/ &
                     BFREFIK(KREP)))
             ENDIF

          ENDIF
       ENDDO
       !C         write(50+irepdstr,'(a,2i4,4f12.5,e20.10)')
       !C     $        'jrep,krep,rdx,rdy,rdz,xprojv,work=',
       !C     $        jrep,krep,rdx,rdy,rdz,xprojv,pathwork(krep)

888    CONTINUE
    ENDDO

11  FORMAT('TEST : ',F15.6)
    !CC
    !CC   MARK: here we need to skew the storage of PMF data!
    !CC         FLUC and PMF are summed when printed at the end of
    !CC         simulation!! No communication, wow.
    !CC
#if KEY_REPDSTR==1
    IF(QREPDSTR)THEN
       !        irepdstr=0 has no data !!!
       IF(IREPDSTR.NE.0)THEN
          KREP=IREPDSTR+1
          PMF(KREP-1)=PMF(KREP-1)+PATHWORK(KREP)
          FLUC(KREP-1)=FLUC(KREP-1)+(PATHWORK(KREP)*PATHWORK(KREP))
       ENDIF
    ELSE
#endif 
       DO JREP=1,NREPM
          KREP=JREP+1
          IF(KREP.GT.NREP) KREP = KREP - NREP

          PMF(JREP) = PMF(JREP) + PATHWORK(KREP)
          FLUC(JREP) = FLUC(JREP) + (PATHWORK(KREP) * PATHWORK(KREP))
       ENDDO
#if KEY_REPDSTR==1
    ENDIF                    
#endif

    RETURN
  END SUBROUTINE PATHDYNA

  SUBROUTINE GETDIST(EPATHR,EPATHA,DX,DY,DZ,X,Y,Z,NREP,KRMS, &
       KMXRMS,RMXRMS,XPREF,YPREF,ZPREF, &
       NATREP,ISTREP,ARMS,BRMS,DRMS,FRMS,WEIGHT,WTOT, &
       QPRINT,LNOTRN,LNOROT,LCYCL,ATOMPR,RA,RB,DRA, &
       DRB,DRTEMP,EVWID,NATOM)

    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !CCC  COMPUTE RMS DIFFERENCES FOR REFERENCE PATH
    !CCC  H. Lee Woodcock 12/03
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    use number
    use stream
    use parallel
    use replica_mod
    use repdstr

    !     --------------------
    !     VARIABLE DECLARATION
    !     --------------------

    !     ----------------
    !     PASSED VARIABLES
    !     ----------------

    real(chm_real) EPATHR
    real(chm_real) EPATHA
    real(chm_real) DX(*),DY(*),DZ(*)
    real(chm_real) X(*),Y(*),Z(*)
    INTEGER NREP
    real(chm_real) KRMS
    real(chm_real) KMXRMS
    real(chm_real) RMXRMS
    INTEGER NATREP
    INTEGER ISTREP(*)
    real(chm_real) ARMS(*)
    real(chm_real) BRMS(*),DRMS(*)
    real(chm_real) FRMS(*)
    real(chm_real) WEIGHT(*)
    real(chm_real) WTOT
    LOGICAL QPRINT
    LOGICAL LNOTRN
    LOGICAL LNOROT
    LOGICAL LCYCL
    INTEGER ATOMPR(2,NATREP)
    real(chm_real) RA(*),RB(*),DRA(*),DRB(*),DRTEMP(*)
    real(chm_real) EVWID
    INTEGER NATOM
    real(chm_real) XPREF(*),YPREF(*),ZPREF(*)

    !     ---------------
    !     LOCAL VARIABLES
    !     ---------------

    INTEGER NREPM
    INTEGER IREP,JREP,KREP,LREP,IPT,JPT,KPT,I,J,K
    real(chm_real)  RMST,R2,RNRM,ERMS,ERMAX,TMP,AX,AY,AZ
    real(chm_real)  ATMP,BTMP,CTMP,DTMP,ETMP ! Temp variables
    real(chm_real) ADERMS,BDERMS,DDERMS      ! Derivative variables
    LOGICAL LPRINT
    real(chm_real) DEVA(3,3)

    !-----------------------------------------------------------------------------

    !lw      IF(LCYCL) THEN
    !lw         NREPM=NREP
    !lw      ELSE
    !lw         NREPM=NREP-2
    !lw      ENDIF
    LPRINT = QPRINT .AND. (PRNLEV.GT.7)

    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !CCC  LOOP TO COMPUTE BEST FIT OF ADJACENT POINTS I,J,K AND COMPUTE FORCES CCC
    !CCC  OF THE POINTS                                                        CCC
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !
    DO KREP=1,NREP
       !        KREP=JREP+1
       !        IF(KREP.GT.NREP) KREP=KREP-NREP
       !        LREP=JREP+2
       !        IF(LREP.GT.NREP) LREP=LREP-NREP
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          IF((KREP-1).NE.IREPDSTR)GOTO 888
       ENDIF
#endif 
       JREP=KREP-1
       IF (JREP.LE.1) THEN
          IF(LCYCL) THEN
             JREP=NREP
          ELSE
             JREP=KREP
          ENDIF
       ENDIF

       LREP=KREP+1
       IF (LREP.GT.NREP) THEN
          IF(LCYCL) THEN
             LREP=1
          ELSE
             LREP=NREP
          ENDIF
       ENDIF

       !     SETUP POINTERS FOR POINTS J AND I
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          IPT=0
          JPT=0
       ELSE
#endif 
          JPT=ISTREP(KREP)
          IPT=ISTREP(JREP)
#if KEY_REPDSTR==1
       ENDIF                
#endif

       DO I=1,NATREP
          JPT=JPT+1
          IPT=IPT+1
          ATOMPR(1,I)=JPT
          ATOMPR(2,I)=IPT
       ENDDO

       !     CALL THE BEST-FIT COMMAND USING REPLICAS J AND I
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          CALL ECBSTF(X,Y,Z,XPEERC(:,2),YPEERC(:,2),ZPEERC(:,2), &
               ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,ATMP,RA,RB,DEVA,EVWID)
       ELSE
#endif 
          CALL ECBSTF(X,Y,Z,XPREF,YPREF,ZPREF,ATOMPR,NATREP, &
               LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,ATMP,RA,RB,DEVA,EVWID)
#if KEY_REPDSTR==1
       ENDIF                             
#endif

       ATMP=ATMP/WTOT
       IF(ATMP.LT.ZERO) ATMP=ZERO
       ATMP=SQRT(ATMP)
       ARMS(KREP)=ATMP
       BFIJ(KREP)=ATMP

       !     SETUP POINTERS FOR POINT K, POINT J IS ALREADY SETUP
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          KPT=0
       ELSE
#endif 
          KPT=ISTREP(LREP)
#if KEY_REPDSTR==1
       ENDIF                   
#endif

       DO I=1,NATREP
          KPT=KPT+1
          ATOMPR(2,I)=KPT
       ENDDO

       !     CALL THE BEST-FIT COMMAND USING REPLICAS J AND K
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          CALL ECBSTF(X,Y,Z,XPEERC(:,4),YPEERC(:,4),ZPEERC(:,4), &
               ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,BTMP,RA,RB,DEVA,EVWID)
       ELSE
#endif 
          CALL ECBSTF(X,Y,Z,XPREF,YPREF,ZPREF,ATOMPR,NATREP, &
               LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,BTMP,RA,RB,DEVA,EVWID)
#if KEY_REPDSTR==1
       ENDIF                              
#endif

       BTMP=BTMP/WTOT
       IF(BTMP.LT.ZERO) BTMP=ZERO
       BTMP=SQRT(BTMP)
       BRMS(KREP)=BTMP
       BFKJ(KREP)=BTMP

       !CCC  ADD RESTRAINT TO BEST-FIT R_J TO R_J(REF)
       !CCC  REUSE THE IPT AND JPT POINTERS
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          IPT=0
       ELSE
#endif 
          IPT=ISTREP(KREP)
#if KEY_REPDSTR==1
       ENDIF                              
#endif

       DO I=1,NATREP
          IPT=IPT+1
          ATOMPR(2,I)=IPT
       ENDDO

       !CCC  CALL THE BEST-FIT COMMAND USING REPLICAS J AND J(REF)
       !CC Nothing to be done here for REPDSTR!
       CALL ECBSTF(X,Y,Z,XPREF,YPREF,ZPREF,ATOMPR,NATREP, &
            LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
            WTOT,DTMP,RA,RB,DEVA,EVWID)

       DTMP=DTMP/WTOT
       IF(DTMP.LT.ZERO) DTMP=ZERO
       DTMP=SQRT(DTMP)
       DRMS(KREP)=DTMP
       BFJJ(KREP)=DTMP
888    CONTINUE
    ENDDO

25  FORMAT(A)

    RETURN
  END SUBROUTINE GETDIST

  SUBROUTINE PATHANAL(DX,DY,DZ,X,Y,Z, &
       REFPX,REFPY,REFPZ, &
       NREPL,NREP,NATREP,ISTREP,RPENR, &
       RPLENG,WEIGHT,WTOT,QPRINT,LNOTRN, &
       LNOROT,LCYCL,LRFIX,LETAN,ATOMPR,RA, &
       RB,DRA,DRB,DRTEMP,PATANG,PATHDL,EVWID,ICIMG, &
       REPEi,REPESQi,REPFi,REPFSQi,REPELAi, &
       REPFLAi,REPLENG,TFCE,PFCE, &
       NPCALL,QPPMF)
    !     This subroutine calculates
    ! 0. The energy of each replica
    ! 1. The unit tangent vector of each image
    ! 2. The length along the tangent direction each image is supposed to go
    ! 3. (F dot tang)*length
    !
    use number
    use psf
    use stream
    use parallel
    INTEGER NREPL,NREP,NATREP,ISTREP(*),ATOMPR(2,NATREP),ICIMG
    INTEGER NPCALL,ATFRST,ATLAST
    real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
    real(chm_real) REFPX(*),REFPY(*),REFPZ(*),RPENR(*),RPLENG(*)
    real(chm_real) WEIGHT(*),WTOT,RA(*),RB(*),DRA(*),DRB(*),DRTEMP(*)
    real(chm_real) PATANG(3,NREPL,NATREP),PATHDL(3,NREPL,NATREP)
    real(chm_real) TFCE(*),PFCE(*)
    real(chm_real) EVWID
    real(chm_real) REPEi(*),REPESQi(*),REPFi(*),REPFSQi(*)
    real(chm_real) REPELAi(*),REPFLAi(*),REPLENG(*)
    LOGICAL QPRINT,LNOTRN,LNOROT,LCYCL,LRFIX,LETAN,QPPMF
    !
    ! local variables
    !
    INTEGER NREPM
    INTEGER IREP,JREP,KREP,I,IPT,JPT
    real(chm_real) TEMP, TEMP1, TEMP2, TEMP3, TEMP4
    LOGICAL LPRINT
    real(chm_real) DEVA(3,3)

    ! At the first call, make all REP*s zero
    IF(NPCALL.EQ.0) THEN
       REPEi  (1:NREPL)=zero
       REPESQi(1:NREPL)=zero
       REPFi  (1:NREPL)=zero
       REPFSQi(1:NREPL)=zero
       REPELAi(1:NREPL)=zero
       REPFLAi(1:NREPL)=zero
       REPLENG(1:NREPL)=zero
    ENDIF
    ! Update the reference path if not fixed
    IF(.NOT.LRFIX) THEN
       refpx(1:natom) = x(1:natom)
       refpy(1:natom) = y(1:natom)
       refpz(1:natom) = z(1:natom)
    ENDIF

    LPRINT = QPRINT .AND. (PRNLEV.GT.7)
    !
    ! Calculate energy of each replica
    !
    CALL REPENR(NREP,NATREP,ISTREP,RPENR, &
         QPRINT,ICIMG) ! jwchu
    !
    ! Calculate Path tangent of each image
    !
    CALL PATHTANG(X,Y,Z,REFPX,REFPY,REFPZ,NREPL,NREP, &
         RPLENG,NATREP,ISTREP,RPENR,WEIGHT,WTOT, &
         LPRINT,LNOTRN,LNOROT,LCYCL,ATOMPR, &
         RA,RB,DRA,DRB,DRTEMP,PATANG,EVWID,LETAN)

    CALL PATHREPDL(X,Y,Z,REFPX,REFPY,REFPZ,NREPL,NREP, &
         NATREP,ISTREP,RPENR,WEIGHT,WTOT, &
         QPRINT,LNOTRN,LNOROT,LCYCL,ATOMPR, &
         RA,RB,DRA,DRB,DRTEMP,PATHDL,EVWID,LETAN)

    ! DL dot tangent

    DO JREP=1,NREP
       TEMP3=ZERO
       DO I=1,NATREP
          TEMP3=TEMP3+PATANG(1,JREP,I)*PATHDL(1,JREP,I)+ &
               PATANG(2,JREP,I)*PATHDL(2,JREP,I)+ &
               PATANG(3,JREP,I)*PATHDL(3,JREP,I)
       ENDDO ! NATREP
       RPLENG(JREP)=TEMP3
    ENDDO
    !
    ! Calculate F dot Tangent
    !
    ! Define the atom bounds for this processor.
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1 /*parfmain*/
    ATFRST=1+IPARPT(MYNOD)
    ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
    ATFRST=1
    ATLAST=NATOM
#endif /* (parfmain)*/
#else /**/
    ATFRST=1
    ATLAST=NATOM
#endif 

    DO JREP=1,NREP
       IPT=ISTREP(JREP)
       TEMP=ZERO
       TEMP1=ZERO
       TEMP3=ZERO
       TEMP4=ZERO
       DO I=1,NATREP
          IPT=IPT+1
          IF(IPT.GE.ATFRST.AND.IPT.LE.ATLAST) THEN
             TEMP=TEMP+DX(IPT)*PATANG(1,JREP,I)+ &
                  DY(IPT)*PATANG(2,JREP,I)+ &
                  DZ(IPT)*PATANG(3,JREP,I)
             !           TEMP1=TEMP1+DX(IPT)*DX(IPT)+
             !     &                 DY(IPT)*DY(IPT)+
             !     &                 DZ(IPT)*DZ(IPT)
             !           IF(WEIGHT(I).GT.TENM8) THEN
             !            TEMP3=TEMP3+DX(IPT)*PATANG(1,JREP,I)/WEIGHT(I)+
             !     &                  DY(IPT)*PATANG(2,JREP,I)/WEIGHT(I)+
             !     &                  DZ(IPT)*PATANG(3,JREP,I)/WEIGHT(I)
             !            TEMP4=TEMP4+DX(IPT)*DX(IPT)/WEIGHT(I)**2+
             !     &                  DY(IPT)*DY(IPT)/WEIGHT(I)**2+
             !     &                  DZ(IPT)*DZ(IPT)/WEIGHT(I)**2
             !           ENDIF ! WEIGHT
          ENDIF ! ATFRST
       ENDDO ! NATREP
       TFCE(JREP)=TEMP
    ENDDO ! JREP
    !
    ! Make sure each node has the correct TFCE array
    !
#if KEY_PARALLEL==1
    IF(NUMNOD.GT.1) THEN
       CALL GCOMB(TFCE,NREPL)
    ENDIF
#endif 
    !
    ! Fill the statistical arrays
    !
    TEMP2=ONE/SQRT(DBLE(NATREP))

    DO JREP=1,NREP
       REPEi(JREP)=REPEi(JREP)+RPENR(JREP)
       REPESQi(JREP)=REPESQi(JREP)+RPENR(JREP)**2
       REPFi(JREP)=REPFi(JREP)+PFCE(JREP)
       REPFSQi(JREP)=REPFSQi(JREP)+PFCE(JREP)**2
       REPELAi(JREP)=RPENR(JREP)
       REPFLAi(JREP)=PFCE(JREP)
       REPLENG(JREP)=RPLENG(JREP)
       PFCE(JREP)=TFCE(JREP)*RPLENG(JREP)
    ENDDO

    NPCALL=NPCALL+1

#if KEY_PARALLEL==1
    IF(MYNOD.EQ.0.AND.QPPMF)THEN
       WRITE(OUTU,131)
131    FORMAT(' I=','              Energy','                 PMF')
       DO I=1,NREP
          WRITE(OUTU,130) I,REPELAi(I),TFCE(I),RPLENG(I),PFCE(I)
       ENDDO
130    FORMAT(I3,1X,4F20.6)
    ENDIF
#else /**/
    IF(QPPMF) THEN
       WRITE(OUTU,131)
131    FORMAT(' I=','              Energy','                 PMF')
       DO I=1,NREP
          WRITE(OUTU,130) I,REPELAi(I),TFCE(I),RPLENG(I),PFCE(I)
       ENDDO
130    FORMAT(I3,1X,4F20.6)
    ENDIF
#endif 

    RETURN
  END SUBROUTINE PATHANAL

  SUBROUTINE PATHTANG(X,Y,Z,REFPX,REFPY,REFPZ,NREPL,NREP, &
       RPLENG,NATREP,ISTREP,RPENR,WEIGHT,WTOT, &
       QPRINT,LNOTRN,LNOROT,LCYCL,ATOMPR, &
       RA,RB,DRA,DRB,DRTEMP,PATANG,EVWID,LETAN)
    !
    !  Calculate the tangent of each image - jwchu  6/22
    !
    !  Ti=(Ri+1-Ri-1) if 1<i<NREP
    !  Ti=(R2-R1) if i=1
    !  Ti=(Rn-Rn-1) if i=NREP
    !  IF QPETAN
    !
    use number
    use psf
    use stream

    real(chm_real) X(*),Y(*),Z(*),REFPX(*),REFPY(*),REFPZ(*)
    INTEGER NREP,NATREP,ISTREP(*),NREPL
    real(chm_real) RPENR(*),RPLENG(*),WEIGHT(*),WTOT
    LOGICAL QPRINT,LNOTRN,LNOROT,LCYCL,LETAN
    INTEGER ATOMPR(2,NATREP)
    real(chm_real) RA(3,NATREP),RB(3,NATREP)
    real(chm_real) DRA(*),DRB(*),DRTEMP(3,NATREP)
    real(chm_real) PATANG(3,NREPL,NATREP),ROTRB,WP1,WM1  ! jwchu
    real(chm_real) EVWID

    INTEGER NREPM
    INTEGER IREP,JREP,KREP,I,IPT,JPT,J,K,KM1   ! jwchu
    real(chm_real) RMST,ATMP,R2,RNRM,ERMS,TMP,AX,AY,AZ,DERMS,SUM
    real(chm_real) SUMIIP1,SUMIIM1
    LOGICAL LPRINT
    real(chm_real) DEVA(3,3)

    DO I=1,3                                   !+jc050711
       DO J=1,3
          IF(I.EQ.J) THEN
             DEVA(I,J)=ONE
          ELSE
             DEVA(I,J)=ZERO
          ENDIF
       ENDDO
    ENDDO                                      !-jc050711

    DO I=1,3
       DO J=1,NREPL
          DO K=1,NATREP
             PATANG(I,J,K)=ZERO
          ENDDO
       ENDDO
    ENDDO

    IF(LCYCL) THEN
       NREPM=NREP
    ELSE
       NREPM=NREP-1
    ENDIF
    LPRINT = QPRINT .AND. (PRNLEV.GT.7)

    RMST=ZERO

    DO JREP=1,NREP
       KREP=JREP+1
       IF(JREP.EQ.NREP) THEN
          IF(LCYCL) THEN
             KREP=KREP-NREP
          ELSE
             KREP=NREP
          ENDIF
       ENDIF
       KM1=JREP-1
       IF(JREP.EQ.1) THEN
          IF(LCYCL) THEN
             KM1=NREP
          ELSE
             KM1=1
          ENDIF
       ENDIF
       WP1=HALF
       WM1=HALF
       IF(JREP.EQ.1.OR.JREP.EQ.NREP) GOTO 111
       !-------------------------------------------------------------------------
       !Energy based tangent estimation
       !-------------------------------------------------------------------------
       IF(LETAN) THEN

          IF(RPENR(JREP).GE.RPENR(KM1).AND.RPENR(JREP).LT.RPENR(KREP)) THEN
             WP1=ONE
             WM1=ZERO
             GOTO 111
          ENDIF

          IF(RPENR(JREP).LT.RPENR(KM1).AND.RPENR(JREP).GE.RPENR(KREP)) THEN
             WP1=ZERO
             WM1=ONE
             GOTO 111
          ENDIF

          IF(RPENR(KREP).GT.RPENR(KM1)) THEN
             WP1=MAX(DABS(RPENR(KREP)-RPENR(JREP)), &
                  DABS(RPENR(KM1)-RPENR(JREP)))
             WM1=MIN(DABS(RPENR(KREP)-RPENR(JREP)), &
                  DABS(RPENR(KM1)-RPENR(JREP)))
             SUM=ONE/(WP1+WM1)
             WP1=WP1*SUM
             WM1=WM1*SUM
             GOTO 111
          ENDIF

          IF(RPENR(KREP).LT.RPENR(KM1)) THEN
             WP1=MIN(DABS(RPENR(KREP)-RPENR(JREP)), &
                  DABS(RPENR(KM1)-RPENR(JREP)))
             WM1=MAX(DABS(RPENR(KREP)-RPENR(JREP)), &
                  DABS(RPENR(KM1)-RPENR(JREP)))
             SUM=ONE/(WP1+WM1)
             WP1=WP1*SUM
             WM1=WM1*SUM
             GOTO 111
          ENDIF

       ENDIF ! QPETAN
111    CONTINUE
       IF(QPRINT) THEN
          WRITE(OUTU,88) JREP,WP1,WM1
       ENDIF

88     FORMAT('At Replica=',I3,1X,'Weight for Ri+1-Ri =',F12.4, &
            1X,'Weight for Ri-Ri-1 =',F12.4)

       IPT=ISTREP(JREP)
       JPT=ISTREP(KREP)
       DO I=1,NATREP
          IPT=IPT+1
          JPT=JPT+1
          ATOMPR(1,I)=IPT
          ATOMPR(2,I)=JPT
       ENDDO

       CALL ECBSTF(X,Y,Z,REFPX,REFPY,REFPZ,ATOMPR,NATREP, &
            LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
            WTOT,ATMP,RA,RB,DEVA,EVWID)

       SUM=ZERO
       DO K=1,NATREP
          IF(WEIGHT(K).GT.TENM8) THEN
             DO I=1,3
                ROTRB=ZERO
                DO J=1,3
                   ROTRB=ROTRB+DEVA(I,J)*RB(J,K)
                ENDDO
                SUM=SUM+(ROTRB-RA(I,K))**2
             ENDDO
          ENDIF
       ENDDO

       SUMIIP1=SUM

       IF(SUM.LT.TENM8) THEN
          SUM=ZERO
       ELSE
          SUM=ONE/SQRT(SUM)
       ENDIF

       DO K=1,NATREP
          IF(WEIGHT(K).GT.TENM8) THEN
             DO I=1,3
                ROTRB=ZERO
                DO J=1,3
                   ROTRB=ROTRB+DEVA(I,J)*RB(J,K)
                ENDDO
                PATANG(I,JREP,K)=(ROTRB-RA(I,K))*WP1*WEIGHT(K) ! jwchu 12162002
                !            PATANG(I,JREP,K)=(ROTRB-RA(I,K))*SUM*WP1
                !            DRTEMP(I,K)=ROTRB-RA(I,K)
             ENDDO
          ENDIF
       ENDDO

       IPT=ISTREP(JREP)
       JPT=ISTREP(KM1)
       DO I=1,NATREP
          IPT=IPT+1
          JPT=JPT+1
          ATOMPR(1,I)=IPT
          ATOMPR(2,I)=JPT
       ENDDO

       CALL ECBSTF(X,Y,Z,REFPX,REFPY,REFPZ,ATOMPR,NATREP, &
            LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
            WTOT,ATMP,RA,RB,DEVA,EVWID)

       SUM=ZERO
       DO K=1,NATREP
          IF(WEIGHT(K).GT.TENM8) THEN
             DO I=1,3
                ROTRB=ZERO
                DO J=1,3
                   ROTRB=ROTRB+DEVA(I,J)*RB(J,K)
                ENDDO
                SUM=SUM+(RA(I,K)-ROTRB)**2
             ENDDO
          ENDIF
       ENDDO

       SUMIIM1=SUM

       IF(SUM.LT.TENM8) THEN
          SUM=ZERO
       ELSE
          SUM=ONE/SQRT(SUM)
       ENDIF

       DO K=1,NATREP
          IF(WEIGHT(K).GT.TENM8) THEN
             DO I=1,3
                ROTRB=ZERO
                DO J=1,3
                   ROTRB=ROTRB+DEVA(I,J)*RB(J,K)
                ENDDO
                PATANG(I,JREP,K)=PATANG(I,JREP,K)+ &
                     (RA(I,K)-ROTRB)*WM1*WEIGHT(K)
             ENDDO
          ELSE
             DO I=1,3
                PATANG(I,JREP,K)=ZERO
             ENDDO
          ENDIF
       ENDDO

       RPLENG(JREP)=WP1*SQRT(SUMIIP1)+WM1*SQRT(SUMIIM1)

    ENDDO
    !
    ! Normalize
    DO JREP=1,NREP
       SUM=ZERO
       DO J=1,NATREP
          DO I=1,3
             SUM=SUM+PATANG(I,JREP,J)**2
          ENDDO
       ENDDO
       IF(SUM.LT.TENM5) THEN
          WRITE(OUTU,88) JREP,WP1,WM1
          !C         write(outu,*) jrep, (X(J),
          !C     &        Y(J),Z(J),j=i,natrep)
          CALL WRNDIE(-2,'<EPATHTANG>','zero tangent length of rep')
       ENDIF
       SUM=SQRT(SUM)
       SUM=ONE/SUM
       DO J=1,NATREP
          DO I=1,3
             PATANG(I,JREP,J)=PATANG(I,JREP,J)*SUM
          ENDDO
       ENDDO
    ENDDO

    IF(QPRINT) THEN
       DO J=1,NREP
          DO K=1,NATREP
             WRITE(OUTU,58) (PATANG(I,J,K),I=1,3)
          ENDDO
       ENDDO
    ENDIF

58  FORMAT('PATANG:',3F14.5)
78  FORMAT(I5)

    RETURN
  END SUBROUTINE PATHTANG

  SUBROUTINE REPENR(NREP,NATREP,ISTREP,RPENR,QPRINT,ICIMG) ! jwhu
    !
    !  Calculate the energy of each replica - jwchu  6/02
    !  RPENR(I)=sigma_j ECONT(J) J: atom j of replica i
    !
    use number
    use stream
    use econtmod   ! jwchu
    use neb   ! jwchuneb

    INTEGER NATREP,ISTREP(*),ICIMG
    real(chm_real) RPENR(*)          !jwchu
    LOGICAL QPRINT
    real(chm_real) E,EMAX            ! jwchu

    INTEGER NREP
    INTEGER JREP,IPT,I       ! jwchu

    ICIMG=1
    EMAX=-10.D10
    DO JREP=1,NREP
       E=ZERO
       IPT=ISTREP(JREP)
       DO I=1,NATREP
          IPT=IPT+1
          E=E+ECONT(IPT)
       ENDDO
       IF(E.GT.EMAX) THEN
          !
          ! because electrostat is not count if fixed
          !
          IF(JREP.NE.1.AND.JREP.NE.NREP) THEN
             EMAX=E
             ICIMG=JREP
          ENDIF
       ENDIF
       RPENR(JREP)=E
    ENDDO

    DO JREP=1,NREP
       EREPS(JREP)=RPENR(JREP)
       !       WRITE(OUTU,*) EREPS(JREP),JREP
    ENDDO

    ICLREP=ICIMG

    IF(QPRINT) THEN
       WRITE(OUTU,59) (JREP,RPENR(JREP),JREP=1,NREP)
       WRITE(OUTU,60) ICIMG,EMAX
59     FORMAT('  In REPENR: energy of each replica:',I5,1F14.5)
60     FORMAT('  In REPENR: The highest energy image:', &
            I5,1X,'E=',1F14.5)
    ENDIF

    RETURN
  END SUBROUTINE REPENR

  SUBROUTINE NUDGEF(DX,DY,DZ,X,Y,Z,NREPL,NREP,NATREP, &
       ISTREP,QPRINT,QPCYCL,DRTEMP,PATANG, &
       PJDX,PJDY,PJDZ,PSDX,PSDY,PSDZ,EVWID,QPCIMG, &
       ICIMG,TFCE,PTANX,PTANY,PTANZ)
    !
    !    Nudge forces parallel to path from (DX,DY,DZ)
    !
    !    F=F-(F*tang)tang
    !    TFCE(NREPL) - Tangent direction force of each replica
    !           I                     I
    use number
    use stream
    use psf
    use parallel

    real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
    real(chm_real) PJDX(*),PJDY(*),PJDZ(*),PSDX(*),PSDY(*),PSDZ(*)
    real(chm_real) PTANX(*),PTANY(*),PTANZ(*)
    INTEGER NREP,NATREP,ISTREP(*),NREPL,ICIMG,IBREP,ISREP
    LOGICAL QPRINT,QPCIMG,QPCYCL
    real(chm_real) DRTEMP(*),PATANG(3,NREPL,NATREP)
    real(chm_real) EVWID,SUM,TFCE(NREPL),TEMP ! jwchu

    INTEGER IREP,JREP,KREP,I,IPT,J
    INTEGER ATFRST,ATLAST

    PSDX(1:NATOM)=zero
    PSDY(1:NATOM)=zero
    PSDZ(1:NATOM)=zero

    pjdx(1:natom) = dx(1:natom)
    pjdy(1:natom) = dy(1:natom)
    pjdz(1:natom) = dz(1:natom)

    !CC transfer pathtang to ptanx, ptany, ptanz
    PTANX(1:NATOM)=zero
    PTANY(1:NATOM)=zero
    PTANZ(1:NATOM)=zero
    DO JREP=1,NREP
       IPT=ISTREP(JREP)
       DO J=1,NATREP
          IPT=IPT+1
          PTANX(IPT)=PATANG(1,JREP,J)
          PTANY(IPT)=PATANG(2,JREP,J)
          PTANZ(IPT)=PATANG(3,JREP,J)
       ENDDO
    ENDDO
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

    ! Define the atom bounds for this processor.
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1 /*parfmain*/
    ATFRST=1+IPARPT(MYNOD)
    ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
    ATFRST=1
    ATLAST=NATOM
#endif /* (parfmain)*/
#else /**/
    ATFRST=1
    ATLAST=NATOM
#endif 
    !
    ! For each replica, Take tangent force out
    !
    IBREP=2
    ISREP=NREP-1
    IF(QPCYCL) IBREP=1
    IF(QPCYCL) ISREP=NREP
    DO JREP=IBREP,ISREP
       IPT=ISTREP(JREP)
       DO I=1,NATREP
          IPT=IPT+1
          IF(IPT.GE.ATFRST.AND.IPT.LE.ATLAST) THEN
             IF(QPCIMG) THEN
                IF(JREP.EQ.ICIMG) THEN
                   PJDX(IPT)=PJDX(IPT)-TWO*TFCE(JREP)*PATANG(1,JREP,I)
                   PJDY(IPT)=PJDY(IPT)-TWO*TFCE(JREP)*PATANG(2,JREP,I)
                   PJDZ(IPT)=PJDZ(IPT)-TWO*TFCE(JREP)*PATANG(3,JREP,I)
                   !jc050711       DX(IPT)=DX(IPT)-TWO*TFCE(JREP)*PATANG(1,JREP,I)
                   !jc050711       DY(IPT)=DY(IPT)-TWO*TFCE(JREP)*PATANG(2,JREP,I)
                   !jc050711       DZ(IPT)=DZ(IPT)-TWO*TFCE(JREP)*PATANG(3,JREP,I)
                ELSE
                   PJDX(IPT)=PJDX(IPT)-TFCE(JREP)*PATANG(1,JREP,I)
                   PJDY(IPT)=PJDY(IPT)-TFCE(JREP)*PATANG(2,JREP,I)
                   PJDZ(IPT)=PJDZ(IPT)-TFCE(JREP)*PATANG(3,JREP,I)
                   !               DX(IPT)=DX(IPT)-TFCE(JREP)*PATANG(1,JREP,I)
                   !               DY(IPT)=DY(IPT)-TFCE(JREP)*PATANG(2,JREP,I)
                   !               DZ(IPT)=DZ(IPT)-TFCE(JREP)*PATANG(3,JREP,I)
                ENDIF ! ICIMG
             ELSE
                PJDX(IPT)=PJDX(IPT)-TFCE(JREP)*PATANG(1,JREP,I)
                PJDY(IPT)=PJDY(IPT)-TFCE(JREP)*PATANG(2,JREP,I)
                PJDZ(IPT)=PJDZ(IPT)-TFCE(JREP)*PATANG(3,JREP,I)
                !             DX(IPT)=DX(IPT)-TFCE(JREP)*PATANG(1,JREP,I)
                !             DY(IPT)=DY(IPT)-TFCE(JREP)*PATANG(2,JREP,I)
                !             DZ(IPT)=DZ(IPT)-TFCE(JREP)*PATANG(3,JREP,I)
             ENDIF ! QPCIMG
          ENDIF ! ATFRST
       ENDDO ! NATREP
    ENDDO
78  FORMAT(I5)

    RETURN
  END SUBROUTINE NUDGEF

  SUBROUTINE PATHREPDL(X,Y,Z,REFPX,REFPY,REFPZ,NREPL,NREP, &
       NATREP,ISTREP,RPENR,WEIGHT,WTOT, &
       QPRINT,LNOTRN,LNOROT,LCYCL,ATOMPR, &
       RA,RB,DRA,DRB,DRTEMP,PATHDL,EVWID,LETAN)
    !
    !  Calculate the tangent of each image - jwchu  6/22
    !
    !  Ti=(Ri+1-Ri-1) if 1<i<NREP
    !  Ti=(R2-R1) if i=1
    !  Ti=(Rn-Rn-1) if i=NREP
    !  IF QPETAN
    !
    use number
    use psf
    use stream

    real(chm_real) X(*),Y(*),Z(*),REFPX(*),REFPY(*),REFPZ(*)
    INTEGER NREP,NATREP,ISTREP(*),NREPL
    real(chm_real) RPENR(*),WEIGHT(*),WTOT
    LOGICAL QPRINT,LNOTRN,LNOROT,LCYCL,LETAN
    INTEGER ATOMPR(2,NATREP)
    real(chm_real) RA(3,NATREP),RB(3,NATREP)
    real(chm_real) DRA(*),DRB(*),DRTEMP(3,NATREP)
    real(chm_real) PATHDL(3,NREPL,NATREP),ROTRB,WP1,WM1  ! jwchu
    real(chm_real) EVWID

    INTEGER NREPM
    INTEGER IREP,JREP,KREP,I,IPT,JPT,J,K,KM1   ! jwchu
    real(chm_real) RMST,ATMP,R2,RNRM,ERMS,TMP,AX,AY,AZ,DERMS,SUM
    real(chm_real) FACT
    LOGICAL LPRINT
    real(chm_real) DEVA(3,3)

    DO I=1,3                                    !+jc050711
       DO J=1,3
          IF(I.EQ.J) THEN
             DEVA(I,J)=ONE
          ELSE
             DEVA(I,J)=ZERO
          ENDIF
       ENDDO
    ENDDO                                       !-jc050711

    DO I=1,3
       DO J=1,NREPL
          DO K=1,NATREP
             PATHDL(I,J,K)=ZERO
          ENDDO
       ENDDO
    ENDDO

    IF(LCYCL) THEN
       NREPM=NREP
    ELSE
       NREPM=NREP-1
    ENDIF
    LPRINT = QPRINT .AND. (PRNLEV.GT.7)

    RMST=ZERO

    DO JREP=1,NREP
       KREP=JREP+1
       IF(JREP.EQ.NREP) THEN
          IF(LCYCL) THEN
             KREP=KREP-NREP
          ELSE
             KREP=NREP
          ENDIF
       ENDIF
       KM1=JREP-1
       IF(JREP.EQ.1) THEN
          IF(LCYCL) THEN
             KM1=NREP
          ELSE
             KM1=1
          ENDIF
       ENDIF
       WP1=HALF
       WM1=HALF
       IF(JREP.EQ.1.OR.JREP.EQ.NREP) GOTO 111
       !-------------------------------------------------------------------------
       !Energy based tangent estimation
       !-------------------------------------------------------------------------
       IF(LETAN) THEN

          IF(RPENR(JREP).GE.RPENR(KM1).AND.RPENR(JREP).LT.RPENR(KREP)) THEN
             WP1=ONE
             WM1=ZERO
             GOTO 111
          ENDIF

          IF(RPENR(JREP).LT.RPENR(KM1).AND.RPENR(JREP).GE.RPENR(KREP)) THEN
             WP1=ZERO
             WM1=ONE
             GOTO 111
          ENDIF

          IF(RPENR(KREP).GT.RPENR(KM1)) THEN
             WP1=MAX(DABS(RPENR(KREP)-RPENR(JREP)), &
                  DABS(RPENR(KM1)-RPENR(JREP)))
             WM1=MIN(DABS(RPENR(KREP)-RPENR(JREP)), &
                  DABS(RPENR(KM1)-RPENR(JREP)))
             SUM=ONE/(WP1+WM1)
             WP1=WP1*SUM
             WM1=WM1*SUM
             GOTO 111
          ENDIF

          IF(RPENR(KREP).LT.RPENR(KM1)) THEN
             WP1=MIN(DABS(RPENR(KREP)-RPENR(JREP)), &
                  DABS(RPENR(KM1)-RPENR(JREP)))
             WM1=MAX(DABS(RPENR(KREP)-RPENR(JREP)), &
                  DABS(RPENR(KM1)-RPENR(JREP)))
             SUM=ONE/(WP1+WM1)
             WP1=WP1*SUM
             WM1=WM1*SUM
             GOTO 111
          ENDIF

       ENDIF ! QPETAN
111    CONTINUE
       IF(QPRINT) THEN
          WRITE(OUTU,88) JREP,WP1,WM1
       ENDIF

88     FORMAT('At Replica=',I3,1X,'Weight for Ri+1-Ri =',F12.4, &
            1X,'Weight for Ri-Ri-1 =',F12.4)

       IPT=ISTREP(JREP)
       JPT=ISTREP(KREP)
       DO I=1,NATREP
          IPT=IPT+1
          JPT=JPT+1
          ATOMPR(1,I)=IPT
          ATOMPR(2,I)=JPT
       ENDDO

       CALL ECBSTF(X,Y,Z,REFPX,REFPY,REFPZ,ATOMPR,NATREP, &
            LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
            WTOT,ATMP,RA,RB,DEVA,EVWID)

       DO K=1,NATREP
          IF(WEIGHT(K).GT.TENM8) THEN
             DO I=1,3
                ROTRB=ZERO
                DO J=1,3
                   ROTRB=ROTRB+DEVA(I,J)*RB(J,K)
                ENDDO
                PATHDL(I,JREP,K)=(ROTRB-RA(I,K))*WP1
             ENDDO
          ENDIF
       ENDDO

       IPT=ISTREP(JREP)
       JPT=ISTREP(KM1)
       DO I=1,NATREP
          IPT=IPT+1
          JPT=JPT+1
          ATOMPR(1,I)=IPT
          ATOMPR(2,I)=JPT
       ENDDO

       CALL ECBSTF(X,Y,Z,REFPX,REFPY,REFPZ,ATOMPR,NATREP, &
            LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
            WTOT,ATMP,RA,RB,DEVA,EVWID)

       DO K=1,NATREP
          IF(WEIGHT(K).GT.TENM8) THEN
             DO I=1,3
                ROTRB=ZERO
                DO J=1,3
                   ROTRB=ROTRB+DEVA(I,J)*RB(J,K)
                ENDDO
                PATHDL(I,JREP,K)=PATHDL(I,JREP,K)+ &
                     (RA(I,K)-ROTRB)*WM1
             ENDDO
          ELSE
             DO I=1,3
                PATHDL(I,JREP,K)=ZERO
             ENDDO
          ENDIF
       ENDDO

    ENDDO

    RETURN
  END SUBROUTINE PATHREPDL

  SUBROUTINE EPATAU(EPATHR,DX,DY,DZ,X,Y,Z,REFX,REFY,REFZ, &
       REFPX,REFPY,REFPZ,REFAX,REFAY,REFAZ, &
       REFBX,REFBY,REFBZ, &
       NREPL,NREP,KRMS,KMXRMS,RMXRMS,NATREP, &
       ISTREP,RPENR,ARMS,BRMS,DRMS,FRMS,RPMF,EPMF, &
       WEIGHT,WTOT,QPRINT,LNOTRN,LNOROT,LCYCL, &
       ATOMPR,RA,RB,DRA,DRB,DRTEMP, &
       PATANG,PATHDL,PATHDL1, &
       EVWID,REPEi,REPESQi,REPFi,REPFSQi,REPELAi, &
       REPFLAi,TFCE,PFCE,NPCALL, &
       QPPMF,QPRPMF,ICIMG)

    use number
    use stream
    use parallel
    use psf

    !CCC  Variables        CCCC

    LOGICAL QPRINT,LNOTRN,LNOROT,LCYCL,QPPMF,QPRPMF
    INTEGER NREPL,NREP,NATREP
    INTEGER ISTREP(*),ATFRST,ATLAST
    INTEGER ATOMPR(2,NATREP)
    INTEGER NPCALL,ICIMG
    real(chm_real) EPATHR
    real(chm_real) DX(*),DY(*),DZ(*),REFX(*),REFY(*),REFZ(*)
    real(chm_real) X(*),Y(*),Z(*),REFPX(*),REFPY(*),REFPZ(*)
    real(chm_real) REFAX(*),REFAY(*),REFAZ(*), &
         REFBX(*),REFBY(*),REFBZ(*)
    real(chm_real) KRMS,KMXRMS,RMXRMS,WTOT,EVWID
    real(chm_real) ARMS(*),BRMS(*),DRMS(*),FRMS(*)
    real(chm_real) RPMF(*),EPMF(*),WEIGHT(*)
    real(chm_real) REPEi(*),REPESQi(*),REPFi(*),REPFSQi(*),REPELAi(*)
    real(chm_real) REPFLAi(*),TFCE(*),PFCE(*),RPENR(*)
    real(chm_real) RA(3,NATREP),RB(3,NATREP)
    real(chm_real) DRA(3,NATREP),DRB(3,NATREP),DRTEMP(3,NATREP)
    real(chm_real) PATHDL(3,NREPL,NATREP),PATANG(3,NREPL,NATREP)
    real(chm_real) PATHDL1(3,NREPL,NATREP)

    !CCC  Local Variables   CCCC

    INTEGER NREPM
    INTEGER IREP,JREP,KREP,LREP,IPT,JPT,KPT,I,J,K
    INTEGER JP1,JM1,IBEG,IEND
    real(chm_real)  RMST,R2,RNRM,ERMS,TMP,AX,AY,AZ,TEMP2
    real(chm_real)  ATMP,BTMP,CTMP,DTMP ! Temp variables
    real(chm_real) ADERMS,BDERMS,DDERMS ! Derivative variables
    real(chm_real) FACT, ROTRB, WM1, WP1,SUM,TEMP  ! Derivative variables
    LOGICAL LPRINT
    real(chm_real) DEVA(3,3),DEVA1(3,3)

    DO I=1,3                                !+jc050711
       DO J=1,3
          IF(I.EQ.J) THEN
             DEVA(I,J)=ONE
             DEVA1(I,J)=ONE
          ELSE
             DEVA(I,J)=ZERO
             DEVA1(I,J)=ZERO
          ENDIF
       ENDDO
    ENDDO                                   !-jc050711

    WM1=HALF
    WP1=HALF
    !
    ! Calculate energy of each replica
    !
    CALL REPENR(NREP,NATREP,ISTREP,RPENR, &
         QPRINT,ICIMG) ! jwchu

    !
    ! Define the atom bounds for this processor.
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1 /*parfmain*/
    ATFRST=1+IPARPT(MYNOD)
    ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
    ATFRST=1
    ATLAST=NATOM
#endif /* (parfmain)*/
#else /**/
    ATFRST=1
    ATLAST=NATOM
#endif 

    LPRINT = QPRINT .AND. (PRNLEV.GT.7)

    ERMS=ZERO

    DO I=1,3
       DO J=1,NREPL
          DO K=1,NATREP
             PATANG(I,J,K)=ZERO
             PATHDL(I,J,K)=ZERO
          ENDDO
       ENDDO
    ENDDO

    DO JREP=1,NREP

       IPT=ISTREP(JREP)
       JPT=ISTREP(JREP)

       DO I=1,NATREP
          IPT=IPT+1
          JPT=JPT+1
          ATOMPR(1,I)=IPT
          ATOMPR(2,I)=JPT
       ENDDO
       ! test prints
       !         write(outu,25) ' current '
       !         IPT=ISTREP(JREP)
       !         DO I=1,NATREP
       !            IPT=IPT+1
       !            write(outu,111) ipt,x(ipt),y(ipt),z(ipt)
       !         ENDDO
       !         write(outu,25) ' refp '
       !         IPT=ISTREP(JREP)
       !         DO I=1,NATREP
       !            IPT=IPT+1
       !            write(outu,111) ipt,refpx(ipt),refpy(ipt),refpz(ipt)
       !         ENDDO
       !         write(outu,25) ' refa '
       !         IPT=ISTREP(JREP)
       !         DO I=1,NATREP
       !            IPT=IPT+1
       !            write(outu,111) ipt,refax(ipt),refay(ipt),refaz(ipt)
       !         ENDDO
       !         write(outu,25) ' refb '
       !         IPT=ISTREP(JREP)
       !         DO I=1,NATREP
       !            IPT=IPT+1
       !            write(outu,111) ipt,refbx(ipt),refby(ipt),refbz(ipt)
       !         ENDDO
111    format(I5,3F16.5)

       ! calculate R-R0

       CALL ECBSTF(X,Y,Z,REFPX,REFPY,REFPZ,ATOMPR,NATREP, &
            LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
            WTOT,DTMP,RA,RB,DEVA,EVWID)

       DTMP=DTMP/WTOT
       IF(DTMP.LT.ZERO) DTMP=ZERO
       DTMP=SQRT(DTMP)
       DRMS(JREP)=DTMP

       DO K=1,NATREP
          IF(WEIGHT(K).GT.TENM8) THEN
             DO I=1,3
                ROTRB=ZERO
                DO J=1,3
                   ROTRB=ROTRB+DEVA(I,J)*RB(J,K)
                ENDDO
                PATHDL(I,JREP,K)=RA(I,K)-ROTRB
             ENDDO
          ENDIF
       ENDDO

       !    R-R0
       !         write(outu,25) ' ra '
       !         DO I=1,NATREP
       !            write(outu,111) i,ra(1,i),ra(2,i),ra(3,i)
       !         ENDDO
       !         write(outu,25) ' rb '
       !         DO I=1,NATREP
       !            write(outu,111) i,rb(1,i),rb(2,i),rb(3,i)
       !         ENDDO

       !    R-R0

       DO K=1,NATREP
          DO I=1,3
             DRB(I,K)=RB(I,K)
             DO J=1,3
                DRB(I,K)=DRB(I,K)-DEVA(J,I)*RA(J,K)
             ENDDO
          ENDDO
       ENDDO

       ! test print
       !         write(outu,25) 'R-R0'
       !         DO K=1,NATREP
       !            write(outu,111) K,DRB(1,K),DRB(2,K),DRB(3,K)
       !         ENDDO
       !         write(outu,25) 'DEVA'
       !         DO K=1,3
       !            write(outu,112) DEVA(K,1),DEVA(K,2),DEVA(K,3)
       !         ENDDO

112    format(3F16.6)

       !    RA-R0

       IPT=ISTREP(JREP)
       JPT=ISTREP(JREP)

       DO I=1,NATREP
          IPT=IPT+1
          JPT=JPT+1
          ATOMPR(1,I)=IPT
          ATOMPR(2,I)=JPT
       ENDDO

       CALL ECBSTF(REFPX,REFPY,REFPZ,REFAX,REFAY,REFAZ,ATOMPR,NATREP, &
            LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
            WTOT,ATMP,RA,RB,DEVA1,EVWID)

       !         write(outu,25) ' refp '
       !         IPT=ISTREP(JREP)
       !         DO I=1,NATREP
       !            IPT=IPT+1
       !            write(outu,111) ipt,refpx(ipt),refpy(ipt),refpz(ipt)
       !         ENDDO
       !         write(outu,25) ' refa '
       !         IPT=ISTREP(JREP)
       !         DO I=1,NATREP
       !            IPT=IPT+1
       !            write(outu,111) ipt,refax(ipt),refay(ipt),refaz(ipt)
       !         ENDDO

       ! PATANG = RA*WM1

       DO K=1,NATREP
          IF(WEIGHT(K).GT.TENM8) THEN
             DO I=1,3
                ROTRB=ZERO
                DO J=1,3
                   ROTRB=ROTRB+DEVA1(I,J)*RB(J,K)
                ENDDO
                PATANG(I,JREP,K)=ROTRB*WM1
             ENDDO
          ENDIF
       ENDDO

       !    (R0-RA) - (R0-R) = R-RA

       DO K=1,NATREP
          DO I=1,3
             DRA(I,K)=RA(I,K)
             DO J=1,3
                DRA(I,K)=DRA(I,K)-DEVA1(I,J)*RB(J,K)
             ENDDO
             DRA(I,K)=DRA(I,K)-DRB(I,K)
          ENDDO
       ENDDO

       ATMP=ZERO
       DO K=1,NATREP
          TMP=ZERO
          DO I=1,3
             TMP=TMP+DRA(I,K)**2
          ENDDO
          ATMP=ATMP+TMP*WEIGHT(K)
       ENDDO

       ATMP=ATMP/WTOT
       IF(ATMP.LT.ZERO) ATMP=ZERO
       ATMP=SQRT(ATMP)
       ARMS(JREP)=ATMP

       !     R-RB

       IPT=ISTREP(JREP)
       JPT=ISTREP(JREP)

       DO I=1,NATREP
          IPT=IPT+1
          JPT=JPT+1
          ATOMPR(1,I)=IPT
          ATOMPR(2,I)=JPT
       ENDDO

       CALL ECBSTF(REFPX,REFPY,REFPZ,REFBX,REFBY,REFBZ,ATOMPR,NATREP, &
            LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
            WTOT,ATMP,RA,RB,DEVA1,EVWID)

       ! PATANG = RB*WP1-RA*WM1

       DO K=1,NATREP
          IF(WEIGHT(K).GT.TENM8) THEN
             DO I=1,3
                ROTRB=ZERO
                DO J=1,3
                   ROTRB=ROTRB+DEVA1(I,J)*RB(J,K)
                ENDDO
                PATANG(I,JREP,K)=ROTRB*WP1-PATANG(I,JREP,K)
             ENDDO
          ENDIF
       ENDDO

       !    (R0-RB) - (R0-R) = R-RB

       DO K=1,NATREP
          DO I=1,3
             DRA(I,K)=RA(I,K)
             DO J=1,3
                DRA(I,K)=DRA(I,K)-DEVA1(I,J)*RB(J,K)
             ENDDO
             DRA(I,K)=DRA(I,K)-DRB(I,K)
          ENDDO
       ENDDO

       BTMP=ZERO
       DO K=1,NATREP
          TMP=ZERO
          DO I=1,3
             TMP=TMP+DRA(I,K)**2
          ENDDO
          BTMP=BTMP+TMP*WEIGHT(K)
       ENDDO

       BTMP=BTMP/WTOT
       IF(BTMP.LT.ZERO) BTMP=ZERO
       BTMP=SQRT(BTMP)
       BRMS(JREP)=BTMP

       ! rotate patang

       DO K=1,NATREP
          DO I=1,3
             ROTRB=ZERO
             DO J=1,3
                ROTRB=ROTRB+DEVA(I,J)*PATANG(J,JREP,K)
             ENDDO
             RA(I,K)=ROTRB
          ENDDO
       ENDDO

       ! PATANG=WEIGHT*(RB*WP1-RA*WM1)

       DO K=1,NATREP
          DO I=1,3
             PATANG(I,JREP,K)=RA(I,K)*WEIGHT(K)
             PATHDL1(I,JREP,K)=RA(I,K)
          ENDDO
       ENDDO

    ENDDO ! JREP

    !
    ! Normalize PATANG
    !
    DO JREP=1,NREP
       SUM=ZERO
       DO J=1,NATREP
          DO I=1,3
             SUM=SUM+PATANG(I,JREP,J)**2
          ENDDO
       ENDDO
       IF(SUM.LT.TENM5) THEN
          WRITE(OUTU,88) JREP,WP1,WM1
          write(outu,*) jrep, (X(J), &
               Y(J),Z(J),j=i,natrep)
          CALL WRNDIE(-2,'<EPTAU>','zero tangent length of rep')
       ENDIF
       SUM=SQRT(SUM)
       SUM=ONE/SUM
       DO J=1,NATREP
          DO I=1,3
             PATANG(I,JREP,J)=PATANG(I,JREP,J)*SUM
          ENDDO
       ENDDO
    ENDDO

88  FORMAT('At Replica=',I3,1X,'Weight for Ri+1-Ri =',F12.4, &
         1X,'Weight for Ri-Ri-1 =',F12.4)

    DO JREP=1,NREP
       ! PATHDL DOT PATANG

       TMP=ZERO
       TEMP=ZERO
       DO K=1,NATREP
          DO I=1,3
             TMP=TMP+PATANG(I,JREP,K)*PATHDL(I,JREP,K)
             TEMP=TEMP+PATANG(I,JREP,K)*PATHDL1(I,JREP,K)
          ENDDO
       ENDDO

       ERMS=ERMS+KRMS*TMP**2
       EPMF(JREP)=HALF*KRMS*TMP**2
       FRMS(JREP)=KRMS*TMP
       RPMF(JREP)=TEMP
       ! TFCE
       IPT=ISTREP(JREP)
       TFCE(JREP)=ZERO
       DO K=1,NATREP
          IPT=IPT+1
          IF(WEIGHT(K).GT.TENM8) THEN
             IF(IPT.GE.ATFRST.AND.IPT.LE.ATLAST) THEN
                TFCE(JREP)=TFCE(JREP)+PATANG(1,JREP,K)*DX(IPT)
                TFCE(JREP)=TFCE(JREP)+PATANG(2,JREP,K)*DY(IPT)
                TFCE(JREP)=TFCE(JREP)+PATANG(3,JREP,K)*DZ(IPT)
             ENDIF
          ENDIF
       ENDDO

       ! ALLPY FORCES
       IPT=ISTREP(JREP)

       DO K=1,NATREP
          IPT=IPT+1
          IF(IPT.GE.ATFRST.AND.IPT.LE.ATLAST) THEN
             DX(IPT)=DX(IPT)+FRMS(JREP)*PATANG(1,JREP,K)
             DY(IPT)=DY(IPT)+FRMS(JREP)*PATANG(2,JREP,K)
             DZ(IPT)=DZ(IPT)+FRMS(JREP)*PATANG(3,JREP,K)
          ENDIF
       ENDDO

    ENDDO ! JREP

    EPATHR=ERMS*HALF

    !
    ! Make sure each node has the correct TFCE array
    !
#if KEY_PARALLEL==1
    IF(NUMNOD.GT.1) THEN
       CALL GCOMB(TFCE,NREPL)
    ENDIF
#endif 
    !
    ! Fill the statistical arrays
    !
    TEMP2=ONE/SQRT(DBLE(NATREP))

    DO JREP=1,NREP
       REPEi(JREP)=REPEi(JREP)+RPENR(JREP)
       REPESQi(JREP)=REPESQi(JREP)+RPENR(JREP)**2
       REPFi(JREP)=REPFi(JREP)+TFCE(JREP)
       REPFSQi(JREP)=REPFSQi(JREP)+TFCE(JREP)**2
       REPELAi(JREP)=RPENR(JREP)
       REPFLAi(JREP)=TFCE(JREP)
    ENDDO

    NPCALL=NPCALL+1

#if KEY_PARALLEL==1
    IF(MYNOD.EQ.0.AND.QPRPMF)THEN
       WRITE(OUTU,131)
131    FORMAT(' I=','Restraint  Energy',' Restraint  PMF', &
            '            ARMS','            BRMS','            DRMS')
       DO I=1,NREP
          WRITE(OUTU,130) I,EPMF(I),FRMS(I),ARMS(I),BRMS(I),DRMS(I)
       ENDDO
130    FORMAT(I3,1X,5F16.6)
    ENDIF
    IF(MYNOD.EQ.0.AND.QPPMF)THEN
       WRITE(OUTU,133)
133    FORMAT(' I=','REPLICA  Energy',' REPLICA  PMF')
       DO I=1,NREP
          WRITE(OUTU,134) I,RPENR(I),TFCE(I),RPMF(I),TFCE(I)*RPMF(I)
       ENDDO
134    FORMAT(I3,1X,4F16.6)
    ENDIF
#else /**/
    IF(QPRPMF) THEN
       WRITE(OUTU,131)
131    FORMAT(' I=','   Restraint  Energy','      Restraint  PMF', &
            '            ARMS','            BRMS','            DRMS')
       DO I=1,NREP
          WRITE(OUTU,130) I,EPMF(I),FRMS(I),ARMS(I),BRMS(I),DRMS(I)
       ENDDO
130    FORMAT(I3,1X,5F16.6)
    ENDIF
    IF(QPPMF)THEN
       WRITE(OUTU,133)
133    FORMAT(' I=','REPLICA  Energy',' REPLICA  PMF')
       DO I=1,NREP
          WRITE(OUTU,134) I,RPENR(I),TFCE(I),RPMF(I),TFCE(I)*RPMF(I)
       ENDDO
134    FORMAT(I3,1X,4F16.6)
    ENDIF
#endif 

25  FORMAT(A)

    RETURN
  END SUBROUTINE EPATAU

  SUBROUTINE NEBTAN(TX, TY, TZ, X, Y, Z, ATOMPR, NATREP)
    !
    ! T[XYZ] -> tangents for image ITAN in X, Y, Z directions
    ! [XYZ]  -> complete coordinates array
    ! ATOMPR -> Double Atom pairs index vector.  Points to images itan-1, itan, itan+1
    ! NATREP -> Number of atoms in each image (replica).
    !
    ! itan is the index of image (replica) for which we obtain the tangent.
    ! it was used to set ATOMPR.
    !
    ! Calculate the tangents corresponding to the nudged elastic band
    ! method at point ITAN.  T[XYZ] contain the tangent vector.  Element
    ! ITAN to ITAN+NATREP of this vector is updated using the images
    ! pointed to by ATOMPR in the coordinates arrays [XYZ].
    !
    ! The simplest definition of tangent itan is the following pseudocode:
    ! dr(:,:) = r(:,:,i3+1) - r(:,:,i3-1)
    ! nrm = dqrt(sum(dr(:,:)**2))
    ! tr(:,:,itan) = dr(:,:) / nrm
    !
    use vector

    real(chm_real) TX(*),TY(*),TZ(*),X(*),Y(*),Z(*)
    INTEGER NATREP,ATOMPR(3,NATREP)

    real(chm_real) t3(3,NATREP)
    INTEGER K,KA,KC

    DO K=1,NATREP
       KA=ATOMPR(1,K)
       KC=ATOMPR(3,K)
       T3(1,K)=X(KC)-X(KA)
       T3(2,K)=Y(KC)-Y(KA)
       T3(3,K)=Z(KC)-Z(KA)
    ENDDO
    ! Call CHARMM''s vector normalization (util/vector.src)
    CALL NORMALL(T3(1,1),3*NATREP)
    DO K=1,NATREP
       TX(K)=T3(1,K)
       TY(K)=T3(2,K)
       TZ(K)=T3(3,K)
    end do

    RETURN
  END SUBROUTINE NEBTAN
  ! NOTE: Last check performed August 18.  Plain rereading.

  SUBROUTINE NEBPFC(PX,PY,PZ,TX,TY,TZ,DX,DY,DZ,ATOMPR,NATREP)
    !
    ! P[XYZ]  -> vectors of length NATREP with the projections at image jrep
    ! T[XYZ]  -> vectors of length NATREP with the tangents of the path at jrep
    ! D[XYZ]  -> the complete potential force vectors
    ! ATOMPR  -> the triple atom pointer list (jrep-1, jrep, jrep+1) (project to jrep)
    ! NATREP  -> the number of atoms per image
    !
    ! Project the potential gradients perpendicular to the tangents vector.
    ! NOTE: Assumes that ATOMPR array values are continuous (in DP evaluation)
    !
    use vector

    real(chm_real) PX(*),PY(*),PZ(*),TX(*),TY(*),TZ(*), &
         DX(*),DY(*),DZ(*)
    INTEGER NATREP,ATOMPR(3,NATREP)

    INTEGER K,KB
    real(chm_real) DP
    !
    !X The location in the full atom vectors were the current replica starts
    !X
    KB=ATOMPR(2,1)
    !
    !X DP is the dot product of the forces and the tangents vector.
    !X
    DP=   DOTVEC(TX,DX(KB),NATREP)
    DP=DP+DOTVEC(TY,DY(KB),NATREP)
    DP=DP+DOTVEC(TZ,DZ(KB),NATREP)
    !X
    !X P[XYZ] has the forces minus the projection along the tangents.
    !X
    DO K=1,NATREP
       KB=ATOMPR(2,K)
       PX(K)=DX(KB)-DP*TX(K)
       PY(K)=DY(KB)-DP*TY(K)
       PZ(K)=DZ(KB)-DP*TZ(K)
    ENDDO

    RETURN
  END SUBROUTINE NEBPFC
  !X NOTE: Last visual inspection (code read) was Aug 18, 2001.

  SUBROUTINE NEBSFC(SX,SY,SZ,TX,TY,TZ,X,Y,Z,ATOMPR,NATREP,KNEB)
    !
    ! S[XYZ]  -> vectors of length NATREP with the projections at image jrep
    ! T[XYZ]  -> vectors of length NATREP with the tangents at image jrep
    ! [XYZ]   -> the complete coordinates vectors
    ! ATOMPR  -> the triple atom pointer list (jrep-1, jrep, jrep+1)
    ! NATREP  -> the number of atoms per image
    ! KNEB    -> the NEB force coupling constant
    !
    ! Project the spring gradients along the current path and update energy.
    ! NOTE: Assumes the ATOMPR array values are continuous (in DP evaluation)
    !
    use vector
    use number

    real(chm_real) SX(*),SY(*),SZ(*),TX(*),TY(*),TZ(*), &
         X(*),Y(*),Z(*),KNEB
    INTEGER NATREP,ATOMPR(3,NATREP)

    INTEGER K,KA,KB,KC
    real(chm_real) R12(3),DP

    DO K=1,NATREP
       KA=ATOMPR(1,K)
       KB=ATOMPR(2,K)
       KC=ATOMPR(3,K)
       !X Calculate the spring forces and put them at S[XYZ] temporarily.
       R12(1)=X(KA)+X(KC)-TWO*X(KB)
       R12(2)=Y(KA)+Y(KC)-TWO*Y(KB)
       R12(3)=Z(KA)+Z(KC)-TWO*Z(KB)
       SX(K)=KNEB*R12(1)
       SY(K)=KNEB*R12(2)
       SZ(K)=KNEB*R12(3)
    ENDDO
    DP=DOTVEC(TX,SX,NATREP)+DOTVEC(TY,SY,NATREP)+DOTVEC(TZ,SZ,NATREP)
    DO K=1,NATREP
       SX(K)=-DP*TX(K)
       SY(K)=-DP*TY(K)
       SZ(K)=-DP*TZ(K)
    ENDDO

    RETURN
  END SUBROUTINE NEBSFC

  SUBROUTINE NEBEN(X,Y,Z,ATOMPR,NATREP,KNEB,JREP,NREPM,EPTHN)
    !
    ! The NEB energy.
    !
    ! In the limit of no _nudging_ it is just the elastic band which
    ! can also be implemented with the rms path restaints.  Nudging
    ! performs projections on the forces and thus fails TEST FIRST.
    !
    ! E = sum { 0.5 * Kneb * <rms>**2 },
    !
    use number
    use vector

    real(chm_real) X(*),Y(*),Z(*)
    INTEGER NATREP,JREP,ATOMPR(3,NATREP),NREPM
    real(chm_real) KNEB,EPTHN

    INTEGER K,KA,KB,KC
    real(chm_real) R(3),RAVER

    DO K=1,NATREP
       KA=ATOMPR(1,K)
       KB=ATOMPR(2,K)
       KC=ATOMPR(3,K)
       R(1)=X(KA)-X(KB)
       R(2)=Y(KA)-Y(KB)
       R(3)=Z(KA)-Z(KB)
       EPTHN=EPTHN+HALF*DOTVEC(R,R,3)*KNEB
       IF (JREP.EQ.NREPM) THEN
          R(1)=X(KC)-X(KB)
          R(2)=Y(KC)-Y(KB)
          R(3)=Z(KC)-Z(KB)
          EPTHN=EPTHN+HALF*DOTVEC(R,R,3)*KNEB
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE NEBEN

  SUBROUTINE EPATHN(EPTHN,DX,DY,DZ,X,Y,Z,NREP,KNEB, &
       KMNRMS,RMNRMS,KMXRMS,RMXRMS, &
       NATREP,ISTREP,ARMS,FRMS,WEIGHT,WTOT,QPRINT, &
       LNOTRN,LNOROT,LCYCL,ATOMPR, &
       EVWID,TXYZ,PXYZ,SXYZ)
    !
    ! The NEB path energy.  - plm 7/25/2001
    !
    ! This is a hack of the rms path restraint energy of Bernie.
    !
    ! In the limit of no _nudging_ it is just the elastic band which
    ! can also be implemented with the rms path restaints.  Nudging
    ! performs projections on the forces and thus fails TEST FIRST.
    ! We add the somewhat random object function:
    ! E = sum { 0.5 * Kneb * (rms-<rms>)**2 }
    !
    ! The forces are projected in two ways:
    ! First the elastic band forces (above term) are projected _along_
    ! the current path.
    ! Second the potential forces (all the preexisting forces) are
    ! projected _perpendicular_ to the path.  This requirement places
    ! constraints as to were the NEB path call is performed in energy.src.
    ! Also note that the projections act as external forces, so, again,
    ! the NEB path call must be made after the virial call in energy.src.
    !
    ! In this alpha version we _neglect_ image alignment!!!!!!!!!!!!!!
    !
    ! TODO:
    ! *) do not process short vectors
    ! *) bestfit align the images along the path
    ! *) MASS weight
    ! *) ANEBA
    ! *) Temperature: this could enter using S=Integral[ Exp(b*U(l)) dl ]
    !
    use number
    use stream

    real(chm_real) EPTHN,KNEB,KMNRMS,RMNRMS,KMXRMS,RMXRMS
    real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
    INTEGER NREP,NATREP,ISTREP(*)
    real(chm_real) ARMS(*),FRMS(*),WEIGHT(*),WTOT
    LOGICAL QPRINT,LNOTRN,LNOROT,LCYCL
    INTEGER ATOMPR(3,NATREP)
    real(chm_real) EVWID
    real(chm_real) TXYZ(NATREP,3)
    real(chm_real) PXYZ(NATREP,3)
    real(chm_real) SXYZ(NATREP,3)

    INTEGER NREPM
    INTEGER IREP,JREP,KREP,I,IPT,JPT,KPT
    real(chm_real) RMST,ATMP,R2,RNRM,ERMS,TMP,AX,AY,AZ,DERMS
    LOGICAL LPRINT
    real(chm_real) DEVA(3,3)
    INTEGER K,KB
    !
    ! If we have a cylic path update the number of replicas.
    !
    IF(LCYCL) THEN
       NREPM=NREP
    ELSE
       NREPM=NREP-1
    ENDIF
    LPRINT = QPRINT .AND. (PRNLEV.GT.7)

    RMST=ZERO
    !
    ! Initialize EPTHN
    !
    EPTHN=ZERO
    !
    ! NEB method. Loop over all the intermediate images.
    !
    DO JREP=2,NREPM
       IREP=JREP-1
       KREP=JREP+1
       IF(KREP.GT.NREP) KREP=KREP-NREP
       !
       ! Prepare the tangents vectors.
       !
       IPT=ISTREP(IREP)
       JPT=ISTREP(JREP)
       KPT=ISTREP(KREP)
       DO I=1,NATREP
          IPT=IPT+1
          JPT=JPT+1
          KPT=KPT+1
          ATOMPR(1,I)=IPT
          ATOMPR(2,I)=JPT
          ATOMPR(3,I)=KPT
       ENDDO
       !
       ! Obtain the tangent at image JREP in vectors T[XYZ]
       !
       CALL NEBTAN(TXYZ(1,1),TXYZ(1,2),TXYZ(1,3),X,Y,Z,ATOMPR,NATREP)
       !
       ! Now project the gradients of the potentials
       !
       CALL NEBPFC(PXYZ(1,1),PXYZ(1,2),PXYZ(1,3), &
            TXYZ(1,1),TXYZ(1,2),TXYZ(1,3), &
            DX,DY,DZ,ATOMPR,NATREP)
       !
       ! Project the spring gradients
       !
       CALL NEBSFC(SXYZ(1,1),SXYZ(1,2),SXYZ(1,3), &
            TXYZ(1,1),TXYZ(1,2),TXYZ(1,3), &
            X,Y,Z,ATOMPR,NATREP,KNEB)
       !
       ! Place the gradients into their proper position.  Kiss your gradients goodby!
       !
       DO K=1,NATREP
          KB=ATOMPR(2,K)
          DX(KB)=PXYZ(K,1)+SXYZ(K,1)
          DY(KB)=PXYZ(K,2)+SXYZ(K,2)
          DZ(KB)=PXYZ(K,3)+SXYZ(K,3)
       ENDDO
       !
       ! Update the energy function.  This energy does not pass TEST FIRST
       !
       CALL NEBEN(X,Y,Z,ATOMPR,NATREP,KNEB,JREP,NREPM,EPTHN)
       !
       ! End of the NEB loop.
       !
    ENDDO

    EPTHN=EPTHN/NATREP
    RETURN
  END SUBROUTINE EPATHN

  SUBROUTINE EPATH_HYPE(QPHP,EPHP,KPHP,RPHP,QEPTHR,QEPTHA, &
       DX,DY,DZ,X,Y,Z,NREPL,NREP,NATREP, &
       ISTREP,ARMS,BRMS,WEIGHT,WTOT,QPRINT, &
       LNOTRN,LNOROT,LCYCL, &
       ATOMPR,RA,RB,DRA,DRB,EVWID, &
       XTAN,YTAN,ZTAN,XM1,YM1,ZM1,XP1,YP1,ZP1)
    use number
    use consta
    use stream
    use parallel
    use repdstr
    use psf  ! for writing NATOM

    LOGICAL QPHP
    real(chm_real) EPHP,KPHP,RPHP
    LOGICAL QEPTHR,QEPTHA
    real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
    INTEGER NREPL,NREP,NATREP,ISTREP(*)
    real(chm_real) ARMS(*),BRMS(*),WEIGHT(*),WTOT
    LOGICAL QPRINT,LNOTRN,LNOROT,LCYCL
    INTEGER ATOMPR(2,NATREP)
    real(chm_real) RA(3,*),RB(3,*),DRA(3,*),DRB(3,*),EVWID

    real(chm_real)  XTAN(NREPL,*),YTAN(NREPL,*),ZTAN(NREPL,*)
    real(chm_real)  XM1(NREPL,*), YM1(NREPL,*), ZM1(NREPL,*)
    real(chm_real)  XP1(NREPL,*), YP1(NREPL,*), ZP1(NREPL,*)

    ! local variables

    LOGICAL LPRINT
    INTEGER NALL
    INTEGER IREP,JREP,KREP,IPT,JPT
    INTEGER I,J
    real(chm_real) ATMP,RMST,ROTRB,ROTRA,DEVA(3,3)
    real(chm_real) TEMP,FACT,SUM,DL,DLEFT,DRIGHT,REFX,REFY,REFZ

    LPRINT = QPRINT .AND. (PRNLEV.GT.7)

    NALL=NREPL*NATREP
    XTAN(1:NREPL,1:NATREP)=zero
    YTAN(1:NREPL,1:NATREP)=zero
    ZTAN(1:NREPL,1:NATREP)=zero
    XM1(1:NREPL,1:NATREP) =zero
    YM1(1:NREPL,1:NATREP) =zero
    ZM1(1:NREPL,1:NATREP) =zero
    XP1(1:NREPL,1:NATREP) =zero
    YP1(1:NREPL,1:NATREP) =zero
    ZP1(1:NREPL,1:NATREP) =zero

    IREP=1
    JREP=2
    KREP=3

    IPT=ISTREP(JREP)
    JPT=ISTREP(IREP)
    DO I=1,NATREP
       IPT=IPT+1
       JPT=JPT+1
       ATOMPR(1,I)=IPT
       ATOMPR(2,I)=JPT
    ENDDO
    CALL ECBSTF(X,Y,Z,X,Y,Z,ATOMPR,NATREP, &
         LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
         WTOT,ATMP,RA,RB,DEVA,EVWID)

    TEMP=ONE/WTOT
    ATMP=ATMP*TEMP
    ATMP=SQRT(ATMP)
    ARMS(IREP)=ATMP
    RMST=RMST+ATMP

    DO I=1,NATREP
       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(1,J)*RB(J,I)
       ENDDO
       XM1(JREP,I)=(RA(1,I)-ROTRB)
       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(2,J)*RB(J,I)
       ENDDO
       YM1(JREP,I)=(RA(2,I)-ROTRB)
       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(3,J)*RB(J,I)
       ENDDO
       ZM1(JREP,I)=(RA(3,I)-ROTRB)
    ENDDO

    IPT=ISTREP(JREP)
    JPT=ISTREP(KREP)
    DO I=1,NATREP
       IPT=IPT+1
       JPT=JPT+1
       ATOMPR(1,I)=IPT
       ATOMPR(2,I)=JPT
    ENDDO
    CALL ECBSTF(X,Y,Z,X,Y,Z,ATOMPR,NATREP, &
         LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
         WTOT,ATMP,RA,RB,DEVA,EVWID)

    ATMP=ATMP*TEMP
    ATMP=SQRT(ATMP)
    ARMS(JREP)=ATMP
    RMST=RMST+ATMP

    DO I=1,NATREP
       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(1,J)*RB(J,I)
       ENDDO
       XP1(JREP,I)=(ROTRB-RA(1,I))
       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(2,J)*RB(J,I)
       ENDDO
       YP1(JREP,I)=(ROTRB-RA(2,I))
       ROTRB=ZERO
       DO J=1,3
          ROTRB=ROTRB+DEVA(3,J)*RB(J,I)
       ENDDO
       ZP1(JREP,I)=(ROTRB-RA(3,I))
    ENDDO

    SUM=ZERO
    DO I=1,NATREP
       FACT=WEIGHT(I)*TEMP
       REFX=XM1(JREP,I)+XP1(JREP,I)
       REFY=YM1(JREP,I)+YP1(JREP,I)
       REFZ=ZM1(JREP,I)+ZP1(JREP,I)

       XTAN(JREP,I)=REFX*FACT
       YTAN(JREP,I)=REFY*FACT
       ZTAN(JREP,I)=REFZ*FACT

       SUM=SUM+REFX*XTAN(JREP,I)+ &
            REFY*YTAN(JREP,I)+ &
            REFZ*ZTAN(JREP,I)
    ENDDO

    DL=SQRT(SUM)

    SUM=ONE/DL

    DO I=1,NATREP
       XTAN(JREP,I)=XTAN(JREP,I)*SUM
       YTAN(JREP,I)=YTAN(JREP,I)*SUM
       ZTAN(JREP,I)=ZTAN(JREP,I)*SUM
    ENDDO

    DLEFT =ZERO
    DRIGHT=ZERO
    DO I=1,NATREP
       DLEFT=DLEFT+  XM1(JREP,I)*XTAN(JREP,I)+ &
            YM1(JREP,I)*YTAN(JREP,I)+ &
            ZM1(JREP,I)*ZTAN(JREP,I)
       DRIGHT=DRIGHT+XP1(JREP,I)*XTAN(JREP,I)+ &
            YP1(JREP,I)*YTAN(JREP,I)+ &
            ZP1(JREP,I)*ZTAN(JREP,I)
    ENDDO

    FACT=(DLEFT-DRIGHT)-RPHP
    EPHP=HALF*KPHP*FACT*FACT

    FACT=TWO*FACT*KPHP

    JPT=ISTREP(JREP)
    DO I=1,NATREP
       JPT=JPT+1
       DX(JPT)=DX(JPT)+FACT*XTAN(JREP,I)
       DY(JPT)=DY(JPT)+FACT*YTAN(JREP,I)
       DZ(JPT)=DZ(JPT)+FACT*ZTAN(JREP,I)
    ENDDO

    RETURN
  END SUBROUTINE EPATH_HYPE

  SUBROUTINE EPATH_KINE(QPKINE,QPLENG,EPKINE,EPLENG,KPKINE,KPLENG, &
       PLFIX,QPTEMP,QEPTHR,QEPTHA, &
       QISOKN,QWETHM,QPKNUDG, &
       PTEMP,DX,DY,DZ,X,Y,Z,REFX,REFY,REFZ, &
       NREPL,NREP,NATREP,ISTREP, &
       ARMS,DLTAN,FRMS,PKE,PKEK,EPKE,WEIGHT,WTOT, &
       QPRINT,LNOTRN,LNOROT,LCYCL,ATOMPR,RA,RB, &
       EVWID,XTAN,YTAN,ZTAN, &
       XM1,YM1,ZM1,XP1,YP1,ZP1,OFPF,FDCR)
    use number
    use consta
    use stream
    use parallel
    use repdstr
    use psf  ! for writing NATOM
    LOGICAL QPKINE,QPLENG,QPTEMP,QEPTHR,QEPTHA,QISOKN,QWETHM
    LOGICAL QPKNUDG
    real(chm_real)  EPKINE,KPKINE,EPLENG,KPLENG,PLENG,PLFIX,PTEMP
    real(chm_real)  DX(*),DY(*),DZ(*),X(*),Y(*),Z(*), &
         REFX(*),REFY(*),REFZ(*)
    INTEGER NREPL,NREP,NATREP,ISTREP(*)
    real(chm_real)  ARMS(*),FRMS(*),DLTAN(*),WEIGHT(*),WTOT
    real(chm_real)  PKE(*),PKEK(*),EPKE(*)
    LOGICAL QPRINT,LNOTRN,LNOROT,LCYCL
    INTEGER ATOMPR(2,NATREP)
    real(chm_real) RA(3,*),RB(3,*)
    real(chm_real) DEVA1(3,3),DEVA2(3,3),EVWID
    LOGICAL LPRINT
    real(chm_real) XTAN(NREPL,*),YTAN(NREPL,*),ZTAN(NREPL,*),TEMPL
    real(chm_real)  XM1(NREPL,*), YM1(NREPL,*), ZM1(NREPL,*)
    real(chm_real)  XP1(NREPL,*), YP1(NREPL,*), ZP1(NREPL,*)
    real(chm_real)  OFPF(*),FDCR(*)

    INTEGER NREPM,IREP,JREP,KREP,IPT,JPT,KPT
    real(chm_real) ATMP,RMST,DEVA(3,3),FACT,ROTRB,ROTRA,TEMP,SUM,ETARG
    real(chm_real) DLL,DLR,FACT1,FACT2
    real(chm_real) TEMPX,TEMPY,TEMPZ,SUM1,KBT,REPKE,KKE
    INTEGER I,J,II,IBEGIN,ISTOP,NALL,NDGC
    real(chm_real) TNM6,TEMP1,PTMP
    integer kidx

    IF((.NOT.QEPTHR).AND.(.NOT.QEPTHA))RETURN

    LPRINT = QPRINT .AND. (PRNLEV.GT.7)

    NALL=NREPL*NATREP
    XTAN(1:NREPL,1:NATREP)=zero
    YTAN(1:NREPL,1:NATREP)=zero
    ZTAN(1:NREPL,1:NATREP)=zero
    EPKE(1:NREPL)=zero

    ! Here we could use XCOORD from rxcons_4, but those arrays are local
    ! there so we just do it here the RPATH way. But parallel/parallel!
    ! Do it analogous to the epathr()
    ! Also the code is tested only with the following options:
    ! rpath krms 0.0 kang 0.0 mass pkin kpki 0.1 [cycl]
    ! but coded for everything!

    ! Calculate the tangent vector and average path length if necessary

    IF(LCYCL) THEN
       IBEGIN=1
       ISTOP =NREP
    ELSE
       IBEGIN=1
       ISTOP =NREP-1
    ENDIF

    TEMP=ONE/DBLE(ISTOP-IBEGIN+1)

    ! printouts:
    !write(outu,*)'X>ibegin,istop=',ibegin,istop
    !write(outu,'(4f12.5)')x(1:natom)
    !if(qrepdstr) then
    !   write(outu,*)'Xpeers:'
    !   write(outu,'(4f12.5)')xpeers(1:natom,4)
    !endif

    RMST=ZERO
    DO JREP=IBEGIN,ISTOP
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          KIDX=IDXREP(JREP,ISTOP,lcycl)
          IF(KIDX < 0) CYCLE
       ELSE
          KIDX=0
       ENDIF
#endif 
       KREP=JREP+1
       IF(KREP.GT.NREP) KREP=KREP-NREP
#if KEY_REPDSTR==1
       IF(QREPDSTR) THEN
          IPT=0
          JPT=0
       ELSE
#endif 
          IPT=ISTREP(JREP)
          JPT=ISTREP(KREP)
#if KEY_REPDSTR==1
       ENDIF                 
#endif
       DO I=1,NATREP
          IPT=IPT+1
          JPT=JPT+1
          ATOMPR(1,I)=IPT
          ATOMPR(2,I)=JPT
       ENDDO
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          !write(outu,*)'eptah_kine>jrep,krep,ibegin,istop=',jrep,krep,ibegin,istop
          CALL ECBSTF(xpeers(1,kidx+2),ypeers(1,kidx+2),zpeers(1,kidx+2), &
               xpeers(1,kidx+3),ypeers(1,kidx+3),zpeers(1,kidx+3), &
               ATOMPR,NATREP,LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,ATMP,RA,RB,DEVA,EVWID)
       ELSE
#endif 
          CALL ECBSTF(X,Y,Z,X,Y,Z,ATOMPR,NATREP, &
               LNOROT,LNOTRN,LPRINT, (/ ZERO /), .FALSE.,WEIGHT, &
               WTOT,ATMP,RA,RB,DEVA,EVWID)
#if KEY_REPDSTR==1
       ENDIF                 
#endif

       ATMP=ATMP/WTOT
       ATMP=SQRT(ATMP)
#if KEY_REPDSTR==1
       IF(.not.(QREPDSTR.and.(KIDX /= 1))) THEN   
#endif
       ARMS(JREP)=ATMP
       RMST=RMST+ATMP
#if KEY_REPDSTR==1
       ENDIF                                      
#endif

       DO I=1,NATREP
          ROTRB=ZERO
          ROTRA=ZERO
          DO J=1,3
             ROTRB=ROTRB+DEVA(1,J)*RB(J,I)
             ROTRA=ROTRA+DEVA(J,1)*RA(J,I)
          ENDDO
          XP1(JREP,I)=(ROTRB-RA(1,I))
          XM1(KREP,I)=(RB(1,I)-ROTRA)
          ROTRB=ZERO
          ROTRA=ZERO
          DO J=1,3
             ROTRB=ROTRB+DEVA(2,J)*RB(J,I)
             ROTRA=ROTRA+DEVA(J,2)*RA(J,I)
          ENDDO
          YP1(JREP,I)=(ROTRB-RA(2,I))
          YM1(KREP,I)=(RB(2,I)-ROTRA)

          ROTRB=ZERO
          ROTRA=ZERO
          DO J=1,3
             ROTRB=ROTRB+DEVA(3,J)*RB(J,I)
             ROTRA=ROTRA+DEVA(J,3)*RA(J,I)
          ENDDO
          ZP1(JREP,I)=(ROTRB-RA(3,I))
          ZM1(KREP,I)=(RB(3,I)-ROTRA)
       ENDDO
    ENDDO

    RMST=RMST*TEMP

    !write(outu,*)'epath_kine-before-gcomb>ibegin,istop=',ibegin,istop
    !write(outu,*)'epath_kine-before-gcomb>mynod,numnod,irepdstr=',mynod,numnod,irepdstr

#if KEY_REPDSTR==1
    IF(QREPDSTR)THEN
       CALL PSETGLOB
       CALL GCOMB(ARMS,ISTOP)
       CALL GCOMB(RMST,1)
       CALL PSETLOC
    ENDIF
#endif 

    !write(outu,*)'epath_kine>rmst=',rmst
    !write(outu,*)'epath_kine-before-qwethm>ibegin,istop=',ibegin,istop

    IF(QWETHM)THEN
       DO JREP=IBEGIN,ISTOP
#if KEY_REPDSTR==1
          IF(QREPDSTR.and.(IDXREP(JREP,ISTOP,lcycl) <= 0)) CYCLE   
#endif
          DO I=1,NATREP
             XTAN(JREP,I)=XM1(JREP,I)+XP1(JREP,I)
             YTAN(JREP,I)=YM1(JREP,I)+YP1(JREP,I)
             ZTAN(JREP,I)=ZM1(JREP,I)+ZP1(JREP,I)
          ENDDO
       ENDDO
    ENDIF


    ! END for the calculation the tangent vector and average path length if necessary
    ! Calculate F dot Tan if work-energy theorem is used for determing the force constants
    ! of kinetic energy potential

    NDGC=0
    DO I=1,NATREP
       IF(WEIGHT(I) > RSMALL)NDGC=NDGC+1
    ENDDO

    IF(QWETHM.OR.QPKNUDG)THEN
       IBEGIN=2
       ISTOP=NREP-1
       IF(LCYCL)THEN
          IBEGIN=1
          ISTOP=NREP
       ENDIF

       DO JREP=IBEGIN,ISTOP
#if KEY_REPDSTR==1
          IF(QREPDSTR.and.(IDXREP(JREP,ISTOP,lcycl) <= 0)) CYCLE   
#endif
          SUM=ZERO
          SUM1=ZERO
          DO I=1,NATREP
             TEMPX=XTAN(JREP,I)
             TEMPY=YTAN(JREP,I)
             TEMPZ=ZTAN(JREP,I)
             SUM=SUM+(TEMPX*TEMPX+TEMPY*TEMPY+TEMPZ*TEMPZ) &
                  *(WEIGHT(I)/WTOT*DBLE(NATREP))**2
             SUM1=SUM1+(TEMPX*TEMPX+TEMPY*TEMPY+TEMPZ*TEMPZ) &
                  *WEIGHT(I)/WTOT*DBLE(NATREP)
          ENDDO
          SUM=SQRT(SUM)*HALF
          SUM1=SQRT(SUM1)*HALF
          DLTAN(JREP)=SUM1*SUM1/SUM
       ENDDO

       TEMP=ONE/WTOT
       DO JREP=IBEGIN,ISTOP
#if KEY_REPDSTR==1
          IF(QREPDSTR.and.(IDXREP(JREP,ISTOP,lcycl) <= 0)) CYCLE   
#endif
          DO I=1,NATREP
             FACT=TEMP*WEIGHT(I)
             XTAN(JREP,I)=XTAN(JREP,I)*FACT
             YTAN(JREP,I)=YTAN(JREP,I)*FACT
             ZTAN(JREP,I)=ZTAN(JREP,I)*FACT
          ENDDO
       ENDDO

       CALL NORM_PATH(NREPL,NATREP,XTAN,YTAN,ZTAN,FRMS,lcycl)
       FRMS(1:NREPL)=zero

       DO JREP=IBEGIN,ISTOP
#if KEY_REPDSTR==1
          IF(QREPDSTR.and.(IDXREP(JREP,ISTOP,lcycl) <= 0)) CYCLE   
#endif
          SUM=ZERO
          SUM1=ZERO
          JPT=ISTREP(JREP)
          DO I=1,NATREP
             JPT=JPT+1
             SUM=SUM+DX(JPT)*XTAN(JREP,I)+ &
                  DY(JPT)*YTAN(JREP,I)+ &
                  DZ(JPT)*ZTAN(JREP,I)
             SUM1=SUM1+DX(JPT)*DX(JPT)+ &
                  DY(JPT)*DY(JPT)+ &
                  DZ(JPT)*DZ(JPT)
          ENDDO
          FRMS(JREP)=SUM
          OFPF(JREP)=SQRT(SUM1-SUM*SUM)
       ENDDO
#if KEY_REPDSTR==1
       IF(QREPDSTR)THEN
          CALL PSETGLOB
          CALL GCOMB(DLTAN,ISTOP)
          CALL GCOMB(FRMS,ISTOP)
          CALL GCOMB(OFPF,ISTOP)
          CALL PSETLOC
       ENDIF
#endif 

    ENDIF !QWETHM
    ! END Calculate F dot Tan if work-energy theorem is used for determing
    ! the force constants of kinetic energy potential
    !
    ! Get the unit (k=1) kinetic energy and path length potentials
    !
    IF(LCYCL)THEN
       IBEGIN=1
       ISTOP =NREP
    ELSE
       IBEGIN=1
       ISTOP =NREP-1
    ENDIF

    DO JREP=IBEGIN,ISTOP
#if KEY_REPDSTR==1
       IF(QREPDSTR.and.(IDXREP(JREP,ISTOP,lcycl) <= 0)) CYCLE   
#endif
       ! krep is not needed here !!! Otherwise the above would fail!
       KREP=JREP+1
       IF(KREP > NREP) KREP=KREP-NREP

       SUM=ZERO
       DO I=1,NATREP
          SUM=SUM+XP1(JREP,I)*XP1(JREP,I)*WEIGHT(I)+ &
               YP1(JREP,I)*YP1(JREP,I)*WEIGHT(I)+ &
               ZP1(JREP,I)*ZP1(JREP,I)*WEIGHT(I)
       ENDDO

       EPKE(JREP)=SUM*HALF
    ENDDO ! JREP

#if KEY_REPDSTR==1
    IF(QREPDSTR)THEN
       CALL PSETGLOB
       CALL GCOMB(EPKE,ISTOP)
       CALL PSETLOC
       ENDIF
#endif 

    !write(outu,'(a,5f15.8)')'epke=',epke(1:istop)

    !write(50+mynodg,'(a,10l4)')'qpkine,qpleng,qptemp,qepthr,qeptha,qisokn,qwethm,qpknudg=', &
    !     qpkine,qpleng,qptemp,qepthr,qeptha,qisokn,qwethm,qpknudg
    !write(50+mynodg,*)'DX>ibegin,istop=',ibegin,istop
    !write(50+mynodg,'(4f12.5)')dx(1:natom)

    !write(50+mynodg,*)'XM1>natrep,ibegin,istop=',natrep,ibegin,istop
    !do i = 1, istop
    !   write(50+mynodg,'(4f12.5)')(xm1(i,j),j=1,natrep)
    !enddo

    !write(50+mynodg,*)'XP1>natrep,ibegin,istop=',natrep,ibegin,istop
    !do i = 1, istop
    !   write(50+mynodg,'(4f12.5)')(xp1(i,j),j=1,natrep)
    !enddo

    TNM6=THREE*DBLE(NDGC)-SIX

    IF(QPTEMP)THEN
       KPKINE=TNM6*KBOLTZ*PTEMP/(RMST*RMST*WTOT)
    ELSE
       PTMP=RMST*RMST*WTOT*KPKINE/(TNM6*KBOLTZ)
    ENDIF

    !
    ! Determining the force constants of kinetic energy potential
    !
    IF(LCYCL) THEN
       IBEGIN=2
       ISTOP =NREP
    ELSE
       IBEGIN=2
       ISTOP =NREP-1
    ENDIF

    ! REPDSTR: because of PKE(JREP-1) below all replicas execute this if block
    !          also then there is no need for communication for pkek and pke
    IF(QPKINE)THEN
       IF(QWETHM)THEN
          PKE(1)=RMST*RMST*WTOT*KPKINE
          PKEK(1)=PKE(1)/EPKE(1)
          DO JREP=IBEGIN,ISTOP
             SUM=FRMS(JREP)*DLTAN(JREP)
             PKE(JREP)=PKE(JREP-1)-SUM
             IF(PKE(JREP) < ZERO)PKE(JREP)=ZERO
             !            IF(PKE(JREP).GT.PKE(1))PKE(JREP)=PKE(1)

             PKEK(JREP)=PKE(JREP)/EPKE(JREP)

          ENDDO
       ELSE
          PKE(1)=RMST*RMST*WTOT*KPKINE
          DO JREP=IBEGIN,ISTOP
             PKEK(JREP)=KPKINE
             PKE(JREP)=PKE(1)
          ENDDO
       ENDIF
    ENDIF
    !write(outu,'(a,5f15.8)')'pke=',pke(1:istop)
    !write(outu,'(a,5f15.8)')'pkek=',pkek(1:istop)
    !
    ! put together energy and gradients
    !
    EPKINE=ZERO
    EPLENG=ZERO

    IF(QPLENG)THEN
       DO JREP=IBEGIN,ISTOP
#if KEY_REPDSTR==1
          IF(QREPDSTR.and.(IDXREP(JREP,ISTOP,lcycl) <= 0)) CYCLE   
#endif
          DLL=ARMS(JREP-1)-PLFIX
          DLR=ARMS(JREP)-PLFIX
          EPLENG=EPLENG+HALF*DLL*DLL*KPLENG
          JPT=ISTREP(JREP)
#if KEY_REPDSTR==1
          IF(QREPDSTR) JPT=0                   
#endif
          TEMP=ONE/WTOT/ARMS(JREP-1)
          TEMP1=ONE/WTOT/ARMS(JREP)
          DO I=1,NATREP
             FACT=TEMP*DLL*KPLENG*WEIGHT(I)
             JPT=JPT+1
             DX(JPT)=DX(JPT)+XM1(JREP,I)*FACT
             DY(JPT)=DY(JPT)+YM1(JREP,I)*FACT
             DZ(JPT)=DZ(JPT)+ZM1(JREP,I)*FACT

             FACT=TEMP1*DLR*KPLENG*WEIGHT(I)
             DX(JPT)=DX(JPT)-XP1(JREP,I)*FACT
             DY(JPT)=DY(JPT)-YP1(JREP,I)*FACT
             DZ(JPT)=DZ(JPT)-ZP1(JREP,I)*FACT
          ENDDO
       ENDDO
       DLR=ARMS(ISTOP)-PLFIX
       EPLENG=EPLENG+HALF*DLR*DLR*KPLENG
    ENDIF

    ! in the case of rpedstr we only execute this on node 0:
#if KEY_REPDSTR==1
    IF(QREPDSTR.and.(mynod /= 0)) GOTO 111                         
#endif
    IF(QPKINE)THEN
       DO JREP=IBEGIN,ISTOP
#if KEY_REPDSTR==1
          IF(QREPDSTR.and.(IDXREP(JREP,ISTOP,lcycl) <= 0)) CYCLE   
#endif
          DO I=1,NATREP
             FACT1=PKEK(JREP-1)*WEIGHT(I)
             FACT2=PKEK(JREP)*WEIGHT(I)
             XM1(JREP,I)=XM1(JREP,I)*FACT1-XP1(JREP,I)*FACT2
             YM1(JREP,I)=YM1(JREP,I)*FACT1-YP1(JREP,I)*FACT2
             ZM1(JREP,I)=ZM1(JREP,I)*FACT1-ZP1(JREP,I)*FACT2
          ENDDO
       ENDDO

       IF(QPKNUDG)THEN
          XP1(1:NREPL,1:NATREP)=zero
          YP1(1:NREPL,1:NATREP)=zero
          ZP1(1:NREPL,1:NATREP)=zero

          DO JREP=IBEGIN,ISTOP
#if KEY_REPDSTR==1
             IF(QREPDSTR.and.(IDXREP(JREP,ISTOP,lcycl) <= 0)) CYCLE   
#endif
             JPT=ISTREP(JREP)
#if KEY_REPDSTR==1
             IF(QREPDSTR)JPT=0                                        
#endif
             SUM=ZERO
             SUM1=ZERO
             DO I=1,NATREP
                JPT=JPT+1
                XP1(JREP,I)=DX(JPT)-FRMS(JREP)*XTAN(JREP,I)
                YP1(JREP,I)=DY(JPT)-FRMS(JREP)*ZTAN(JREP,I)
                ZP1(JREP,I)=DZ(JPT)-FRMS(JREP)*YTAN(JREP,I)
                SUM=SUM+XP1(JREP,I)*XP1(JREP,I)+ &
                     YP1(JREP,I)*YP1(JREP,I)+ &
                     ZP1(JREP,I)*ZP1(JREP,I)

                SUM1=SUM1+XM1(JREP,I)*XM1(JREP,I)+ &
                     YM1(JREP,I)*YM1(JREP,I)+ &
                     ZM1(JREP,I)*ZM1(JREP,I)
             ENDDO
             OFPF(JREP)=SQRT(SUM)
             FDCR(JREP)=SQRT(SUM1)
          ENDDO
          DO JREP=IBEGIN,ISTOP
#if KEY_REPDSTR==1
             IF(QREPDSTR.and.(IDXREP(JREP,ISTOP,lcycl) <= 0)) CYCLE   
#endif
             IF(OFPF(JREP).GT.PT0001.AND.FDCR(JREP).GT.PT0001)THEN
                SUM=ZERO
                DO I=1,NATREP
                   SUM=SUM+XM1(JREP,I)*XP1(JREP,I)+ &
                        YM1(JREP,I)*YP1(JREP,I)+ &
                        ZM1(JREP,I)*ZP1(JREP,I)
                ENDDO
                IF(SUM.LT.ZERO)THEN
                   !
                   ! determine scalar factor
                   !
                   SUM=ABS(SUM)/FDCR(JREP)
                   IF(SUM.GE.FDCR(JREP))THEN
                      FACT=ZERO
                   ELSE
                      FACT=FDCR(JREP)-SUM
                      FACT=FACT/FDCR(JREP)
                   ENDIF

                   DO I=1,NATREP
                      XM1(JREP,I)=XM1(JREP,I)*FACT
                      YM1(JREP,I)=YM1(JREP,I)*FACT
                      ZM1(JREP,I)=ZM1(JREP,I)*FACT
                   ENDDO

                ENDIF !(SUM.LT.ZERO)
             ENDIF ! OFPF
          ENDDO !JREP=IBEGIN,ISTOP
       ENDIF !QPKNUDG

       DO JREP=IBEGIN,ISTOP
#if KEY_REPDSTR==1
          IF(QREPDSTR.and.(IDXREP(JREP,ISTOP,lcycl) <= 0)) CYCLE   
#endif
          FACT=PKEK(JREP)*EPKE(JREP)
          EPKINE=EPKINE+FACT

          JPT=ISTREP(JREP)
#if KEY_REPDSTR==1
          IF(QREPDSTR)JPT=0                                        
#endif
          DO I=1,NATREP
             JPT=JPT+1
             DX(JPT)=DX(JPT)+XM1(JREP,I)
             DY(JPT)=DY(JPT)+YM1(JREP,I)
             DZ(JPT)=DZ(JPT)+ZM1(JREP,I)
          ENDDO
       ENDDO
    ENDIF !QPKINE
111 continue      ! for parallel/parallel ignore on mynod /= 0

    !write(outu,*)'DX>ibegin,istop=',ibegin,istop
    !write(outu,'(4f12.5)')dx(1:natom)

    !write(50+mynodg,'(a,i0)')'weight>natrep=',natrep
    !write(50+mynodg,'(4f12.5)')weight(1:natrep)

    !write(50+mynodg,*)'DX>ibegin,istop=',ibegin,istop
    !write(50+mynodg,'(4f12.5)')dx(1:natom)

    ! this one is just to save my time on bc -l
    !if(numnod.gt.1)call gcomb(dx,natom)
    !write(50+mynodg,*)'DX-sum>ibegin,istop=',ibegin,istop
    !write(50+mynodg,'(4f12.5)')dx(1:natom)


    !write(50+mynodg,*)'XM1>natrep,ibegin,istop=',natrep,ibegin,istop
    !do i = 1, istop
    !   write(50+mynodg,'(4f12.5)')(xm1(i,j),j=1,natrep)
    !enddo

    !write(50+mynodg,*)'XP1>natrep,ibegin,istop=',natrep,ibegin,istop
    !do i = 1, istop
    !   write(50+mynodg,'(4f12.5)')(xp1(i,j),j=1,natrep)
    !enddo

10  FORMAT(A,A,A)
20  FORMAT(I5,2F16.6)

    RETURN
  END SUBROUTINE EPATH_KINE
#endif /* (replica2)  IFN REPLICA*/
#endif /*  (rpath1)   RPATH*/


#if KEY_REPLICA==1
  SUBROUTINE NORM_PATH(NREP,NRXATM,XVEC,YVEC,ZVEC,D_TAN,lcycl)
    !
    use chm_kinds
    use number
    use stream
#if KEY_REPDSTR==1
    use repdstr       
#endif
    implicit none

    INTEGER NREP,NRXATM

    REAL(chm_real) XVEC(NREP,*),YVEC(NREP,*),ZVEC(NREP,*)
    REAL(chm_real) D_TAN(*)

    logical lcycl

    INTEGER JREP,I
    REAL(chm_real) SUM

    DO JREP=1,NREP
#if KEY_REPDSTR==1
       IF(QREPDSTR.and.(IDXREP(JREP,NREP,lcycl) <= 0)) CYCLE   
#endif
       SUM=ZERO

       DO I=1,NRXATM
          SUM=SUM+XVEC(JREP,I)*XVEC(JREP,I)+ &
               YVEC(JREP,I)*YVEC(JREP,I)+ &
               ZVEC(JREP,I)*ZVEC(JREP,I)
       ENDDO

       IF(SUM.GT.TENM8)THEN
          D_TAN(JREP)=SUM
          SUM=ONE/SQRT(SUM)
       ELSE
          SUM=ZERO
          D_TAN(JREP)=SUM
       ENDIF

       DO I=1,NRXATM
          XVEC(JREP,I)=XVEC(JREP,I)*SUM
          YVEC(JREP,I)=YVEC(JREP,I)*SUM
          ZVEC(JREP,I)=ZVEC(JREP,I)*SUM
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE NORM_PATH
#endif 

END MODULE EPATHMOD

