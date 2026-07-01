module nmdimb_module

contains

#if KEY_DIMB==1 /*DIMB*/
  SUBROUTINE NMDIMB(X,Y,Z,NAT3,BNBND,BIMAG,LNOMA,AMASS,DDS,DDSCR, &
       PARDDV,DDV,DDM,PARDDF,DDF,PARDDE,DDEV,DD1BLK, &
       DD1BLL,NADD,LRAISE,DD1CMP,INBCMP,JNBCMP, &
       NPAR,ATMPAR,ATMPAS,BLATOM,PARDIM,NFREG,NFRET, &
       PARFRQ,CUTF1,ITMX,TOLDIM,IUNMOD,IUNRMD, &
       LBIG,LSCI,ATMPAD,SAVF,NBOND,IB,JB,DDVALM)
    !-----------------------------------------------------------------------
    !     01-Jul-1992 David Perahia, Liliane Mouawad
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     This is the main routine for the mixed-basis diagonalization.
    !     See: L.Mouawad and D.Perahia, Biopolymers (1993), 33, 599,
    !     and: D.Perahia and L.Mouawad, Comput. Chem. (1995), 19, 241.
    !     The method iteratively solves the diagonalization of the
    !     Hessian matrix. To save memory space, it uses a compressed
    !     form of the Hessian, which only contains the non-zero elements.
    !     In the diagonalization process, approximate eigenvectors are
    !     mixed with Cartesian coordinates to form a reduced basis. The
    !     Hessian is then diagonalized in the reduced basis. By iterating
    !     over different sets of Cartesian coordinates the method ultimately
    !     converges to the exact eigenvalues and eigenvectors (up to the
    !     requested accuracy).
    !     If no existing basis set is read, an initial basis will be created
    !     which consists of the low-frequency eigenvectors of diagonal blocks
    !     of the Hessian.
    !-----------------------------------------------------------------------
  use chm_kinds
  use chm_types
  use stream
  use dimens_fcm
  use number
  use consta
  use fast
  use deriv
  use energym
  use eutil
  use dimb
  use ctitla
  use memory
  use vibio
  use dimbsubs,only:corarr,inipaf,parint,parlis,rbdg,partds,partic,partid, &
    partit,ipart,selnmd,rnmtst
  use dimbutils,only:adzer,adzerd,adjnme,trrot,cletr
    implicit none
    ! Passed variables
    real(chm_real),allocatable,dimension(:,:) :: TRAROT,ddvbas,ddval
    real(chm_real),allocatable,dimension(:) :: SCIFV1,SCIFV2,SCIFV3,SCIFV4,SCIFV6, &
         DDV2,DDSS,DD5, &
         DRATQ,ERATQ,E2RATQ,BDRATQ

    integer,allocatable,dimension(:) :: INRATQ,iupd,ATMCOR,sublis
    integer,allocatable,dimension(:,:) :: ATMPAF
    INTEGER NAT3,NADD,NPAR,NFREG,NFRET,BLATOM
    INTEGER ATMPAR(2,*),ATMPAS(2,*),ATMPAD(2,*)
    !!      INTEGER BNBND(*),BIMAG(*)
    type(nonbondDataStructure) BNBND
    type(imageDataStructure) BIMAG

    INTEGER INBCMP(*),JNBCMP(*),PARDIM
    INTEGER ITMX,IUNMOD,IUNRMD,SAVF
    INTEGER NBOND,IB(*),JB(*)
    real(chm_real) :: X(:), Y(:), Z(:), AMASS(*),DDSCR(*)
    real(chm_real) DDV(NAT3,*),PARDDV(PARDIM,*),DDM(*),DDS(*)
    real(chm_real) DDF(*),PARDDF(*),DDEV(*),PARDDE(*)
    real(chm_real) :: DD1BLK(*),DD1BLL(*), DD1CMP(:)
    real(chm_real) TOLDIM,DDVALM
    real(chm_real) PARFRQ,CUTF1
    LOGICAL LNOMA,LRAISE,LSCI,LBIG
    ! Local variables
    INTEGER NATOM,NATP,NDIM,I,J,II,OLDFAS,OLDPRN
    INTEGER NPARC,NPARD,NPARS,NFCUT1,NFREG2,NFREG6
    INTEGER IH1,IH2,IH3,IH4,IH5,IH6,IH7,IH8
    INTEGER IS1,IS2,IS3,IS4,JSPACE,JSP
    INTEGER ISTRT,ISTOP,IPA1,IPA2,IRESF
    INTEGER INIDS
    INTEGER NFRRES
    INTEGER LENCM,NTR,NFRE,NFC,N1,N2,NFCUT,NSUBP
    INTEGER I620,I640,I660,I700,I720,I760,I800,I840,I880,I920
    real(chm_real) CVGMX,TOLER
    LOGICAL LCARD,LAPPE,LPURG,LWDINI,QCALC,QMASWT,QMIX,QDIAG

    ! Begin
    QCALC=.TRUE.
    LWDINI=.FALSE.
    INIDS=0
    IS3=0
    IS4=0
    LPURG=.TRUE.
    ITER=0
    NADD=0
    NFSAV=0
    TOLER=TENM5
    QDIAG=.TRUE.
    CVGMX=HUNDRD
    QMIX=.FALSE.
    NATOM=NAT3/3
    NFREG6=(NFREG-6)/NPAR
    NFREG2=NFREG/2
    NFRRES=(NFREG+6)/2
    IF(NFREG > PARDIM) CALL WRNDIE(-3,'<NMDIMB>', &
         'NFREG IS LARGER THAN PARDIM*3')
    ! TO ALLOCATE-SPACE-FOR-TRANSROT-VECTORS
    call chmalloc('nmdimb.src','NMDIMB','TRAROT',NAT3,6,crl=TRAROT)

    !-----------------------------------------------------------------------
    ! TO ALLOCATE-SPACE-FOR-DIAGONALIZATION
    call chmalloc('nmdimb.src','NMDIMB','DDV2',(PARDIM+3)*(pardim+3),crl=DDV2)
    JSPACE= (PARDIM+4)*8
    JSP   = (PARDIM+3)*(PARDIM+4)/2
    call chmalloc('nmdimb.src','NMDIMB','DDSS',JSPACE,crl=DDSS)
    call chmalloc('nmdimb.src','NMDIMB','DD5',JSP,crl=DD5)


    !C-----------------------------------------------------------------------
    !C TO ALLOCATE-SPACE-FOR-REDUCED-BASIS
    IF(LBIG) THEN
       call chmalloc('nmdimb.src','NMDIMB','DDVBAS',NAT3,1,crl=DDVBAS)
    ELSE
       call chmalloc('nmdimb.src','NMDIMB','DDVBAS',nat3,NFREG,crl=DDVBAS)
    ENDIF

    !C-----------------------------------------------------------------------
    !C TO ALLOCATE-SPACE-FOR-OTHER-ARRAYS
    call chmalloc('nmdimb.src','NMDIMB','IUPD',PARDIM+3,intg=IUPD)

    ! Space allocation for working arrays of EISPACK
    ! diagonalization subroutines
    IF(LSCI) THEN
       !C TO ALLOCATE-SPACE-FOR-LSCI
       call chmalloc('nmdimb.src','NMDIMB','SCIFV1',PARDIM+3,crl=SCIFV1)
       call chmalloc('nmdimb.src','NMDIMB','SCIFV2',PARDIM+3,crl=SCIFV2)
       call chmalloc('nmdimb.src','NMDIMB','SCIFV3',PARDIM+3,crl=SCIFV3)
       call chmalloc('nmdimb.src','NMDIMB','SCIFV4',PARDIM+3,crl=SCIFV4)
       call chmalloc('nmdimb.src','NMDIMB','SCIFV6',PARDIM+3,crl=SCIFV6)

       call chmalloc('nmdimb.src','NMDIMB','DRATQ',PARDIM+3,crl=DRATQ)
       call chmalloc('nmdimb.src','NMDIMB','ERATQ',PARDIM+3,crl=ERATQ)
       call chmalloc('nmdimb.src','NMDIMB','E2RATQ',PARDIM+3,crl=E2RATQ)
       call chmalloc('nmdimb.src','NMDIMB','BDRATQ',PARDIM+3,crl=BDRATQ)
       call chmalloc('nmdimb.src','NMDIMB','INRATQ',PARDIM+3,intg=INRATQ)
    else
       !C-----------------------------------------------------------------------
       !C TO ALLOCATE-DUMMY-SPACE-FOR-LSCI
       call chmalloc('nmdimb.src','NMDIMB','SCIFV1',2,crl=SCIFV1)
       call chmalloc('nmdimb.src','NMDIMB','SCIFV2',2,crl=SCIFV2)
       call chmalloc('nmdimb.src','NMDIMB','SCIFV3',2,crl=SCIFV3)
       call chmalloc('nmdimb.src','NMDIMB','SCIFV4',2,crl=SCIFV4)
       call chmalloc('nmdimb.src','NMDIMB','SCIFV6',2,crl=SCIFV6)

       call chmalloc('nmdimb.src','NMDIMB','DRATQ',2,crl=DRATQ)
       call chmalloc('nmdimb.src','NMDIMB','ERATQ',2,crl=ERATQ)
       call chmalloc('nmdimb.src','NMDIMB','E2RATQ',2,crl=E2RATQ)
       call chmalloc('nmdimb.src','NMDIMB','BDRATQ',2,crl=BDRATQ)
       call chmalloc('nmdimb.src','NMDIMB','INRATQ',2,intg=INRATQ)
    ENDIF

    QMASWT=(.NOT.LNOMA)

    IF(.NOT. QDISK) THEN
       LENCM=INBCMP(NATOM-1)*9+NATOM*6
       DO I=1,LENCM
          DD1CMP(I)=0.0
       ENDDO
       OLDFAS=LFAST
       QCMPCT=.TRUE.
       LFAST = -1
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1,NAT3,DD1CMP)
       LFAST=OLDFAS
       QCMPCT=.FALSE.
       !
       ! Mass weight DD1CMP matrix
       !
       CALL MASSDD(DD1CMP,DDM,INBCMP,JNBCMP,NATOM)

    ELSE
       CALL WRNDIE(-3,'<NMDIMB>','QDISK OPTION NOT SUPPORTED YET')
    ENDIF ! (.NOT. QDISK)

    !
    ! Fill DDV with six translation-rotation vectors
    !
    CALL TRROT(X,Y,Z,DDV,NAT3,1,DDM)
    trarot(1:nat3,1:6) = ddv(1:nat3,1:6)
    NTR=6
    OLDPRN=PRNLEV
    PRNLEV=1
    CALL ORTHNM(1,6,NTR,TRAROT,NAT3,.FALSE.,TOLER)
    PRNLEV=OLDPRN

    IF(IUNRMD  <  0) THEN
       !
       ! If no previous basis is read
       !

       IF(PRNLEV >= 2) WRITE(OUTU,502) NPAR
502    FORMAT(/' NMDIMB: Calculating initial basis from block ', &
            'diagonals'/' NMDIMB: The number of blocks is ',I5/)

       NFRET = 6
       DO I=1,NPAR
          IS1=ATMPAR(1,I)
          IS2=ATMPAR(2,I)
          NDIM=(IS2-IS1+1)*3
          NFRE=NDIM
          IF(NFRE > NFREG6) NFRE=NFREG6
          IF(NFREG6 == 0) NFRE=1
          CALL FILUPT(IUPD,NDIM)
          CALL MAKDDU(DD1BLK,DD1CMP,INBCMP,JNBCMP,IUPD, &
               IS1,IS2,NATOM)
          IF(PRNLEV >= 9) CALL PRINTE(OUTU,EPROP,ETERM,'VIBR', &
               'ENR',.TRUE.,1,ZERO,ZERO, .TRUE.)

          ! Generate the lower section of the matrix and diagonalize
#if KEY_EISPACK==1
          IF(LSCI) THEN
             CALL GENLSC(DD1BLK,DD1BLL,NDIM)
             CALL DIASCI(NDIM,DD1BLL,PARDDE,PARDDV, &
                  scifv1,scifv2)
          ELSE
#endif 
             IH1=1
             NATP=NDIM+1
             IH2=IH1+NATP
             IH3=IH2+NATP
             IH4=IH3+NATP
             IH5=IH4+NATP
             IH6=IH5+NATP
             IH7=IH6+NATP
             IH8=IH7+NATP
             CALL DIAGQ(NDIM,NFRE,DD1BLK,PARDDV,DDS(IH2),DDS(IH3), &
                  DDS(IH4),DDS(IH5),DDS,DDS(IH6),DDS(IH7),DDS(IH8),NADD)
#if KEY_EISPACK==1
          ENDIF
#endif 

          ! Put the PARDDV vectors into DDV and replace the elements which do
          ! not belong to the considered partitioned region by zeros.
          CALL ADJNME(DDV,PARDDV,NAT3,NDIM,NFRE,NFRET,IS1,IS2)

          IF(LSCI) THEN
             DO J=1,NFRE
                PARDDF(J)=CNVFRQ*SQRT(ABS(PARDDE(J)))
                IF(PARDDE(J)  <  0.0) PARDDF(J)=-PARDDF(J)
             ENDDO
          ELSE
             DO J=1,NFRE
                PARDDE(J)=DDS(J)
                PARDDF(J)=CNVFRQ*SQRT(ABS(PARDDE(J)))
                IF(PARDDE(J)  <  0.0) PARDDF(J)=-PARDDF(J)
             ENDDO
          ENDIF

          IF(PRNLEV >= 2) THEN
             WRITE(OUTU,512) I
             WRITE(OUTU,514)
             WRITE(OUTU,516) (J,PARDDF(J),J=1,NFRE)
          ENDIF
          NFRET=NFRET+NFRE
          IF(NFRET  >=  NFREG) GOTO 10
       ENDDO  ! DO I=1,NPAR

512    FORMAT(/' NMDIMB: Diagonalization of part',I5,' completed')
514    FORMAT(' NMDIMB: Frequencies'/)
516    FORMAT(5(I4,F12.6))

10     CONTINUE
       !
       ! Orthonormalize the eigenvectors
       !
       OLDPRN=PRNLEV
       PRNLEV=1
       CALL ORTHNM(1,NFRET,NFRET,DDV,NAT3,LPURG,TOLER)
       PRNLEV=OLDPRN
       !
       ! Do reduced basis diagonalization using the DDV vectors
       ! and get eigenvectors of zero iteration
       !
       IF(PRNLEV >= 2) THEN
          WRITE(OUTU,521) ITER
          WRITE(OUTU,523) NFRET
       ENDIF
521    FORMAT(/' NMDIMB: Iteration number = ',I5)
523    FORMAT(' NMDIMB: Dimension of the reduced basis set = ',I5)

       IF(LBIG) THEN
          IF(PRNLEV >= 2) WRITE(OUTU,585) NFRET,IUNMOD
585       FORMAT(' NMDIMB: ',I5,' modes are saved in unit',I5)
525       FORMAT(' NMDIMB: ',I5,' basis vectors are saved in unit',I5)
          REWIND (UNIT=IUNMOD)
          LCARD=.FALSE.
          CALL WRTNMD(LCARD,1,NFRET,NAT3,DDV,DDSCR,DDEV,IUNMOD,AMASS)
          CALL SAVEIT(IUNMOD)
       ELSE
          ddvbas(1:nat3,1:nfret)=ddv(1:nat3,1:nfret)
       ENDIF

       CALL RBDG(X,Y,Z,NAT3,NDIM,NFRET,DDV,DDF,DDEV, &
            DDSCR,DD5,DDSS,DDV2,NADD, &
            INBCMP,JNBCMP,DDVBAS,DD1CMP,QMIX,0,0,IS3,IS4, &
            CUTF1,NFCUT1,NFREG,IUPD,DD1BLL,scifv1, &
            scifv2,scifv3,scifv4,scifv6, &
            DRATQ ,ERATQ ,E2RATQ, &
            BDRATQ,INRATQ,LSCI,LBIG,IUNMOD)
       !
       ! DO-THE-DIAGONALISATIONS-WITH-RESIDUALS
       !
       call sub620

       ! SAVE-MODES
       call sub700

       IF(ITER == ITMX) THEN
          CALL CLEANHP
          RETURN
       ENDIF

    ELSE
       !
       ! Read in existing basis
       !
       IF(PRNLEV >= 2) THEN
          WRITE(OUTU,531)
531       FORMAT(/' NMDIMB: Calculations restarted')
       ENDIF

       ! READ-MODES
       ISTRT=1
       ISTOP=99999999
       LCARD=.FALSE.
       LAPPE=.FALSE.
       CALL RDNMD(LCARD,NFRET,NFREG,NAT3,NDIM, &
            DDV,DDSCR,DDF,DDEV, &
            IUNRMD,LAPPE,ISTRT,ISTOP)
       NFRET=NDIM
       IF(NFRET > NFREG) THEN
          NFRET=NFREG
          CALL WRNDIE(-1,'<NMDIMB>', &
               'Not enough space to hold the basis. Increase NMODes')
       ENDIF

       ! PRINT-MODES
       IF(PRNLEV >= 2) THEN
          WRITE(OUTU,533) NFRET,IUNRMD
          WRITE(OUTU,514)
          WRITE(OUTU,516) (J,DDF(J),J=1,NFRET)
       ENDIF
533    FORMAT(/' NMDIMB: ',I5,' restart modes read from unit ',I5)

       NFRRES=NFRET

    ENDIF ! IF(IUNRMD < 0)

    !
    ! -------------------------------------------------
    ! Here starts the mixed-basis diagonalization part.
    ! -------------------------------------------------
    !

    !
    ! Check cut-off frequency
    !

    CALL SELNMD(DDF,NFRET,CUTF1,NFCUT1)
    ! TEST-NFCUT1
    IF(IUNRMD < 0) THEN
       IF(NFCUT1*2-6 > NFREG) THEN
          IF(PRNLEV >= 2) WRITE(OUTU,537) DDF(NFRRES)
          NFCUT1=NFRRES
          CUTF1=DDF(NFRRES)
       ENDIF
    ELSE
       CUTF1=DDF(NFRRES)
    ENDIF
537 FORMAT(/' NMDIMB: Too many vectors for the given cutoff frequency' &
         /'         Cutoff frequency is decreased to',F9.3)

    !
    ! Compute the new partioning of the molecule
    !
    CALL PARTIC(NAT3,NFREG,NFCUT1,NPARMX,NPARC,ATMPAR,NFRRES, &
         PARDIM)
    NPARS=NPARC
    DO I=1,NPARC
       ATMPAS(1,I)=ATMPAR(1,I)
       ATMPAS(2,I)=ATMPAR(2,I)
    ENDDO

    IF(QDW) THEN
       IF(IPAR1 == 0.OR.IPAR2.EQ.0) LWDINI=.TRUE.
       IF(IPAR1 >= IPAR2) LWDINI=.TRUE.
       IF(IABS(IPAR1) > NPARC*2) LWDINI=.TRUE.
       IF(IABS(IPAR2) > NPARC*2) LWDINI=.TRUE.
       IF(ITER == 0) LWDINI=.TRUE.
    ENDIF
    ITMX=ITMX+ITER
    IF(PRNLEV >= 2) THEN
       WRITE(OUTU,543) ITER,ITMX
       IF(QDW) WRITE(OUTU,545) IPAR1,IPAR2
    ENDIF
543 FORMAT(/' NMDIMB: Previous iteration number = ',I8/ &
         ' NMDIMB: Iteration number to reach = ',I8)
545 FORMAT(' NMDIMB: Previous sub-blocks = ',I5,2X,I5)
    !
    IF(SAVF <= 0) SAVF=NPARC
    IF(PRNLEV >= 2) WRITE(OUTU,547) SAVF
547 FORMAT(' NMDIMB: Eigenvectors will be saved every',I5, &
         ' iterations')
    !
    ! If double windowing is defined, the original block sizes are divided
    ! in two.
    !
    IF(QDW) THEN
       NSUBP=1
       CALL PARTID(NPARC,ATMPAR,NPARD,ATMPAD,NPARMX)
       call chmalloc('nmdimb.src','NMDIMB','ATMPAF',NPARD,NPARD,intg=ATMPAF)
       call chmalloc('nmdimb.src','NMDIMB','ATMCOR',NATOM,intg=ATMCOR)

       call chmalloc('nmdimb.src','NMDIMB','DDVAL',NPARD,NPARD,crl=DDVAL)
       CALL CORARR(ATMPAD,NPARD,ATMCOR,NATOM)
       CALL PARLIS(ATMCOR,ATMPAF,INBCMP,JNBCMP,NPARD, &
            NSUBP,NATOM,X,Y,Z,NBOND,IB,JB,DD1CMP,DDVAL,DDVALM)
       call chmalloc('nmdimb.src','NMDIMB','SUBLIS',NSUBP*2,intg=SUBLIS)
       CALL PARINT(ATMPAF,NPARD,SUBLIS,NSUBP)
       CALL INIPAF(ATMPAF,NPARD)
       !
       ! Find out with which block to continue (double window method only)
       !
       IPA1=IPAR1
       IPA2=IPAR2
       IRESF=0
       IF(LWDINI) THEN
          ITER=0
          LWDINI=.FALSE.
          GOTO 500
       ENDIF
       DO II=1,NSUBP
          CALL IPART(SUBLIS,II,IPAR1,IPAR2,ATMPAF, &
               NPARD,QCALC)
          IF((IPAR1 == IPA1).AND.(IPAR2.EQ.IPA2)) GOTO 500
       ENDDO
    ENDIF  ! (QDW)
500 CONTINUE

    !
    ! Main loop.
    !
    DO WHILE((CVGMX > TOLDIM).AND.(ITER < ITMX))

       IF(.NOT.QDW) THEN
          ITER=ITER+1
          IF(PRNLEV >= 2) WRITE(OUTU,553) ITER
553       FORMAT(/' NMDIMB: Iteration number = ',I8)
          IF(INIDS == 0) THEN
             INIDS=1
          ELSE
             INIDS=0
          ENDIF
          CALL PARTDS(NAT3,NPARC,ATMPAR,NPARS,ATMPAS,INIDS,NPARMX, &
               DDF,NFREG,CUTF1,PARDIM,NFCUT1)
          ! DO-THE-DIAGONALISATIONS
          call sub640

          QDIAG=.FALSE.

          ! DO-THE-DIAGONALISATIONS-WITH-RESIDUALS
          call sub620

          QDIAG=.TRUE.

          ! SAVE-MODES
          call sub700

          !
       ELSE   ! IF (.NOT.QDW)
          DO II=1,NSUBP
             CALL IPART(SUBLIS,II,IPAR1,IPAR2,ATMPAF, &
                  NPARD,QCALC)
             IF(QCALC) THEN
                IRESF=IRESF+1
                ITER=ITER+1
                IF(PRNLEV >= 2) WRITE(OUTU,553) ITER
                ! DO-THE-DWIN-DIAGONALISATIONS
                call sub660

             ENDIF
             IF((IRESF == SAVF).OR.(ITER.EQ.ITMX)) THEN
                IRESF=0
                QDIAG=.FALSE.

                ! DO-THE-DIAGONALISATIONS-WITH-RESIDUALS
                call sub620

                QDIAG=.TRUE.
                IF((CVGMX <= TOLDIM).OR.(ITER == ITMX)) GOTO 600

                ! SAVE-MODES
                call sub700

             ENDIF

          ENDDO   ! II=1,NSUBP
       ENDIF      ! IF (.NOT.QDW)
    ENDDO         ! WHILE((CVGMX > TOLDIM).AND.(ITER < ITMX))

600 CONTINUE
    !
    ! SAVE-MODES
    call sub700


    CALL CLEANHP
    RETURN

    !-----------------------------------------------------------------------
    ! INTERNAL PROCEDURES
    !-----------------------------------------------------------------------
    ! TO DO-THE-DIAGONALISATIONS-WITH-RESIDUALS

  contains
    subroutine sub620
      IF(IUNRMD < 0) THEN
         CALL SELNMD(DDF,NFRET,CUTF1,NFC)
         N1=NFCUT1
         N2=(NFRET+6)/2
         NFCUT=MAX(N1,N2)
         IF(NFCUT*2-6  >  NFREG) THEN
            NFCUT=(NFREG+6)/2
            CUTF1=DDF(NFCUT)
            IF(PRNLEV >= 2) THEN
               WRITE(OUTU,562) ITER
               WRITE(OUTU,564) CUTF1
            ENDIF
         ENDIF
      ELSE
         NFCUT=NFRET
         NFC=NFRET
      ENDIF
562   FORMAT(/' NMDIMB: Not enough space to hold the residual vectors'/ &
           '         into DDV array during iteration ',I5)
564   FORMAT('         Cutoff frequency is changed to ',F9.3)

      !
      ! do reduced diagonalization with preceding eigenvectors plus
      ! residual vectors
      !
      ISTRT=1
      ISTOP=NFCUT
      CALL CLETR(DDV,TRAROT,NAT3,ISTRT,ISTOP,NFCUT,DDEV,DDF)
      CALL RNMTST(DDV,DDVBAS,NAT3,DDSCR,DD1CMP,INBCMP,JNBCMP, &
           7,NFCUT,CVGMX,NFCUT,NFC,QDIAG,LBIG,IUNMOD)
      NFSAV=NFCUT
      IF(QDIAG) THEN
         NFRET=NFCUT*2-6
         IF(PRNLEV >= 2) WRITE(OUTU,566) NFRET
566      FORMAT(/' NMDIMB: Diagonalization with residual vectors. '/ &
              '          Dimension of the reduced basis set'/ &
              '             before orthonormalization = ',I5)
         NFCUT=NFRET
         OLDPRN=PRNLEV
         PRNLEV=1
         CALL ORTHNM(1,NFRET,NFCUT,DDV,NAT3,LPURG,TOLER)
         PRNLEV=OLDPRN
         NFRET=NFCUT
         IF(PRNLEV >= 2) WRITE(OUTU,568) NFRET
568      FORMAT('             after orthonormalization  = ',I5)

         IF(LBIG) THEN
            IF(PRNLEV >= 2) WRITE(OUTU,570) NFCUT,IUNMOD
570         FORMAT(' NMDIMB: ',I5,' basis vectors are saved in unit',I5)
            REWIND (UNIT=IUNMOD)
            LCARD=.FALSE.
            CALL WRTNMD(LCARD,1,NFCUT,NAT3,DDV,DDSCR,DDEV,IUNMOD,AMASS)
            CALL SAVEIT(IUNMOD)
         ELSE
            ddvbas(1:nat3,1:nfcut)=ddv(1:nat3,1:nfcut)
         ENDIF

         QMIX=.FALSE.
         CALL RBDG(X,Y,Z,NAT3,NDIM,NFRET,DDV,DDF,DDEV, &
              DDSCR,DD5,DDSS,DDV2,NADD, &
              INBCMP,JNBCMP,DDVBAS,DD1CMP,QMIX,0,0,IS3,IS4, &
              CUTF1,NFCUT1,NFREG,IUPD,DD1BLL,scifv1, &
              scifv2,scifv3,scifv4,scifv6, &
              DRATQ,ERATQ,E2RATQ, &
              BDRATQ,INRATQ,LSCI,LBIG,IUNMOD)
         CALL SELNMD(DDF,NFRET,CUTF1,NFCUT1)
      ENDIF
      return
    end subroutine sub620
    !
    !-----------------------------------------------------------------------
    ! TO DO-THE-DIAGONALISATIONS

    subroutine sub640
      DO I=1,NPARC
         NFCUT1=NFRRES
         IS1=ATMPAR(1,I)
         IS2=ATMPAR(2,I)
         NDIM=(IS2-IS1+1)*3
         IF(PRNLEV >= 2) WRITE(OUTU,573) I,IS1,IS2
573      FORMAT(/' NMDIMB: Mixed diagonalization, part ',I5/ &
              ' NMDIMB: Block limits: ',I5,2X,I5)
         IF(NDIM+NFCUT1 > PARDIM) CALL WRNDIE(-3,'<NMDIMB>', &
              'Error in dimension of block')

         NFRET=NFCUT1
         IF(NFRET > NFREG) NFRET=NFREG
         CALL CLETR(DDV,TRAROT,NAT3,1,NFCUT1,NFCUT,DDEV,DDF)
         NFCUT1=NFCUT
         CALL ADZER(DDV,1,NFCUT1,NAT3,IS1,IS2)
         NFSAV=NFCUT1
         OLDPRN=PRNLEV
         PRNLEV=1
         CALL ORTHNM(1,NFCUT1,NFCUT,DDV,NAT3,LPURG,TOLER)
         PRNLEV=OLDPRN
         ddvbas(1:nat3,1:nfcut)=ddv(1:nat3,1:nfcut)
         NFRET=NDIM+NFCUT
         QMIX=.TRUE.
         CALL RBDG(X,Y,Z,NAT3,NDIM,NFRET,DDV,DDF,DDEV, &
              DDSCR,DD5,DDSS,DDV2,NADD, &
              INBCMP,JNBCMP,DDVBAS,DD1CMP,QMIX,IS1,IS2,IS3,IS4, &
              CUTF1,NFCUT,NFREG,IUPD,DD1BLL,scifv1, &
              scifv2,scifv3,scifv4,scifv6, &
              DRATQ,ERATQ,E2RATQ, &
              BDRATQ,INRATQ,LSCI,LBIG,IUNMOD)
         QMIX=.FALSE.
         IF(NFCUT > NFRRES) NFCUT=NFRRES
         NFCUT1=NFCUT
         NFRET=NFCUT
      ENDDO   ! I=1,NPARC

      return
    end subroutine sub640

    !
    !-----------------------------------------------------------------------
    ! TO DO-THE-DWIN-DIAGONALISATIONS

    subroutine sub660
      !
      ! Store the DDV vectors into DDVBAS
      !
      NFCUT1=NFRRES
      IS1=ATMPAD(1,IPAR1)
      IS2=ATMPAD(2,IPAR1)
      IS3=ATMPAD(1,IPAR2)
      IS4=ATMPAD(2,IPAR2)
      NDIM=(IS2-IS1+IS4-IS3+2)*3
      IF(PRNLEV >= 2) WRITE(OUTU,577) IPAR1,IPAR2,IS1,IS2,IS3,IS4
577   FORMAT(/' NMDIMB: Mixed double window diagonalization, parts ', &
           2I5/ &
           ' NMDIMB: Block limits: ',I5,2X,I5,4X,I5,2X,I5)
      IF(NDIM+NFCUT1 > PARDIM) CALL WRNDIE(-3,'<NMDIMB>', &
           'Error in dimension of block')

      NFRET=NFCUT1
      IF(NFRET > NFREG) NFRET=NFREG
      !
      ! Prepare the DDV vectors consisting of 6 translations-rotations
      ! + eigenvectors from 7 to NFCUT1 + cartesian displacements vectors
      ! spanning the atoms from IS1 to IS2
      !
      CALL CLETR(DDV,TRAROT,NAT3,1,NFCUT1,NFCUT,DDEV,DDF)
      NFCUT1=NFCUT
      NFSAV=NFCUT1
      CALL ADZERD(DDV,1,NFCUT1,NAT3,IS1,IS2,IS3,IS4)
      OLDPRN=PRNLEV
      PRNLEV=1
      CALL ORTHNM(1,NFCUT1,NFCUT,DDV,NAT3,LPURG,TOLER)
      PRNLEV=OLDPRN
      ddvbas(1:nat3,1:nfcut)=ddv(1:nat3,1:nfcut)

      NFRET=NDIM+NFCUT
      QMIX=.TRUE.
      CALL RBDG(X,Y,Z,NAT3,NDIM,NFRET,DDV,DDF,DDEV, &
           DDSCR,DD5,DDSS,DDV2,NADD, &
           INBCMP,JNBCMP,DDVBAS,DD1CMP,QMIX,IS1,IS2,IS3,IS4, &
           CUTF1,NFCUT,NFREG,IUPD,DD1BLL,scifv1, &
           scifv2,scifv3,scifv4,scifv6, &
           DRATQ,ERATQ,E2RATQ, &
           BDRATQ,INRATQ,LSCI,LBIG,IUNMOD)
      QMIX=.FALSE.
      !
      IF(NFCUT > NFRRES) NFCUT=NFRRES
      NFCUT1=NFCUT
      NFRET=NFCUT

      return
    end subroutine sub660
    !
    !-----------------------------------------------------------------------
    ! TO SAVE-MODES

    subroutine sub700
      IF(PRNLEV >= 2) WRITE(OUTU,583) IUNMOD
583   FORMAT(/' NMDIMB: Saving the eigenvalues and eigenvectors to unit' &
           ,I4)
      REWIND (UNIT=IUNMOD)
      ISTRT=1
      ISTOP=NFSAV
      LCARD=.FALSE.
      IF(PRNLEV >= 2) WRITE(OUTU,585) NFSAV,IUNMOD
      CALL WRTNMD(LCARD,ISTRT,ISTOP,NAT3,DDV,DDSCR,DDEV,IUNMOD, &
           AMASS)
      CALL SAVEIT(IUNMOD)
585   FORMAT(' NMDIMB: ',I5,' modes are saved in unit',I5)
      return
    end subroutine sub700

    SUBROUTINE CLEANHP
      !-----------------------------------------------------------------------
      !     03-Dec-1994 Herman van Vlijmen
      !
      !     This routine cleans up the space allocated for the
      !     DIMB routines.
      !
      !-----------------------------------------------------------------------

      ! Begin
      NATOM=NAT3/3

      ! CLEAN-UP-SPACE-FOR-DIAGONALIZATION
      call chmdealloc('nmdimb.src','NMDIMB','DDV2',(PARDIM+3)*(pardim+3),crl=DDV2)
      call chmdealloc('nmdimb.src','CLEANHP','DDSS',JSPACE,crl=DDSS)
      call chmdealloc('nmdimb.src','CLEANHP','DD5', JSP   ,crl=DD5)

      ! CLEAN-UP-SPACE-FOR-REDUCED-BASIS
      IF(LBIG) THEN
         call chmdealloc('nmdimb.src','CLEANHP','DDVBAS',NAT3,1,crl=DDVBAS)
      ELSE
         call chmdealloc('nmdimb.src','CLEANHP','DDVBAS',nat3,NFREG,crl=DDVBAS)
      ENDIF

      ! CLEAN-UP-SPACE-FOR-TRANSROT-VECTORS
      call chmdealloc('nmdimb.src','CLEANHP','TRAROT',NAT3,6,crl=TRAROT)

      ! CLEAN-UP-SPACE-FOR-OTHER-ARRAYS
      call chmdealloc('nmdimb.src','CLEANHP','IUPD',PARDIM+3,intg=IUPD)

      IF(QDW) THEN
         call chmdealloc('nmdimb.src','CLEANHP','ATMPAF',NPARD,NPARD,intg=ATMPAF)
         call chmdealloc('nmdimb.src','CLEANHP','ATMCOR',NATOM,intg=ATMCOR)
         call chmdealloc('nmdimb.src','CLEANHP','SUBLIS',NSUBP*2,intg=SUBLIS)
         call chmdealloc('nmdimb.src','CLEANHP','DDVAL',NPARD,NPARD,crl=DDVAL)
      ENDIF

      IF(LSCI) THEN
         ! CLEAN-UP-SPACE-FOR-LSCI
         call chmdealloc('nmdimb.src','CLEANHP','SCIFV1',PARDIM+3,crl=SCIFV1)
         call chmdealloc('nmdimb.src','CLEANHP','SCIFV2',PARDIM+3,crl=SCIFV2)
         call chmdealloc('nmdimb.src','CLEANHP','SCIFV3',PARDIM+3,crl=SCIFV3)
         call chmdealloc('nmdimb.src','CLEANHP','SCIFV4',PARDIM+3,crl=SCIFV4)
         call chmdealloc('nmdimb.src','CLEANHP','SCIFV6',PARDIM+3,crl=SCIFV6)

         call chmdealloc('nmdimb.src','CLEANHP','DRATQ',PARDIM+3, crl=DRATQ)
         call chmdealloc('nmdimb.src','CLEANHP','ERATQ',PARDIM+3, crl=ERATQ)
         call chmdealloc('nmdimb.src','CLEANHP','E2RATQ',PARDIM+3,crl=E2RATQ)
         call chmdealloc('nmdimb.src','CLEANHP','BDRATQ',PARDIM+3,crl=BDRATQ)
         call chmdealloc('nmdimb.src','CLEANHP','INRATQ',PARDIM+3,intg=INRATQ)
      ELSE
         ! CLEAN-UP-DUMMY-SPACE-FOR-LSCI
         call chmdealloc('nmdimb.src','CLEANHP','SCIFV1',2,crl=SCIFV1)
         call chmdealloc('nmdimb.src','CLEANHP','SCIFV2',2,crl=SCIFV2)
         call chmdealloc('nmdimb.src','CLEANHP','SCIFV3',2,crl=SCIFV3)
         call chmdealloc('nmdimb.src','CLEANHP','SCIFV4',2,crl=SCIFV4)
         call chmdealloc('nmdimb.src','CLEANHP','SCIFV6',2,crl=SCIFV6)

         call chmdealloc('nmdimb.src','CLEANHP','DRATQ',2, crl=DRATQ)
         call chmdealloc('nmdimb.src','CLEANHP','ERATQ',2, crl=ERATQ)
         call chmdealloc('nmdimb.src','CLEANHP','E2RATQ',2,crl=E2RATQ)
         call chmdealloc('nmdimb.src','CLEANHP','BDRATQ',2,crl=BDRATQ)
         call chmdealloc('nmdimb.src','CLEANHP','INRATQ',2,intg=INRATQ)
      ENDIF

      RETURN
    END SUBROUTINE CLEANHP


  end SUBROUTINE NMDIMB



#endif /* (DIMB)*/


  SUBROUTINE NMDIMB_NULL
    RETURN
  END SUBROUTINE NMDIMB_NULL

end module nmdimb_module

