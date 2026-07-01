module redbas_m
  implicit none

contains

SUBROUTINE REDBAS(COMLYN,COMLEN,NFREQ,NNMDS,NAT3,DDV,DDSCR,DDF, &
     DDEV,DDM,XNEW,YNEW,ZNEW, &
     LENIC,IAR,JAR,KAR,LAR,TAR)
  !
  ! This routine generates a reduced basis and adds the new vectors
  ! to the set of existing vectors.
  !
  ! Bernard R. Brooks      9-Nov-1984
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use coord
  use memory
  use select
  use stream
  use string
  use vector
  use vibsub

  implicit none

  character(len=*) COMLYN
  INTEGER COMLEN,NFREQ,NNMDS,NAT3
  real(chm_real) DDV(NAT3,NNMDS),DDSCR(*),DDF(*),DDEV(*),DDM(*)
  real(chm_real) XNEW(*),YNEW(*),ZNEW(*)
  INTEGER LENIC
  INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
  LOGICAL TAR(*)

  real(chm_real),allocatable,dimension(:,:) :: DDTROT
  integer,allocatable,dimension(:) :: ISLCT
  CHARACTER(len=4) :: WRD
  INTEGER NATIC,IATIC(4)
  INTEGER LFIRST,NTROT,IIC,I,IMODE,IS,IQ,ITROT
  real(chm_real) Q
  LOGICAL OK,LORTH,REMTR
  real(chm_real) TOL
  DATA TOL/1.0E-10/

  LORTH=.NOT.(INDXA(COMLYN,COMLEN,'NOOR').GT.0)
  WRD=NEXTA4(COMLYN,COMLEN)

  !=======================================================================
  IF(WRD.EQ.'IC') THEN
     ! Process IC basis
     LFIRST=0
     IF(INDXA(COMLYN,COMLEN,'FIRS').GT.0) LFIRST=-1
     IF(INDXA(COMLYN,COMLEN,'SECO').GT.0) LFIRST=1
     WRD=NEXTA4(COMLYN,COMLEN)

     DO IIC=1,LENIC
        IF(WRD.EQ.'BOND') THEN
           IF(LFIRST.EQ.0) THEN
              CALL WRNDIE(0,'<REDBAS>','ILLEGAL BASIS OPTION')
              RETURN
           ENDIF
           NATIC=2
           IF(LFIRST.LT.0) THEN
              ! Do first bond basis
              !
              IATIC(1)=IAR(IIC)
              IF(TAR(IIC)) THEN
                 IATIC(2)=KAR(IIC)
              ELSE
                 IATIC(2)=JAR(IIC)
              ENDIF
           ELSE
              ! Do second bond basis
              !
              IATIC(1)=KAR(IIC)
              IATIC(2)=LAR(IIC)
           ENDIF
        ELSE IF(WRD.EQ.'ANGL') THEN
           IF(LFIRST.EQ.0) THEN
              CALL WRNDIE(0,'<REDBAS>','ILLEGAL BASIS OPTION')
              RETURN
           ENDIF
           NATIC=3
           IF(LFIRST.LT.0) THEN
              ! Do first angle basis
              !
              IATIC(1)=IAR(IIC)
              IF(TAR(IIC)) THEN
                 IATIC(2)=KAR(IIC)
                 IATIC(3)=JAR(IIC)
              ELSE
                 IATIC(2)=JAR(IIC)
                 IATIC(3)=KAR(IIC)
              ENDIF
           ELSE
              ! Do second angle basis
              !
              IATIC(1)=JAR(IIC)
              IATIC(2)=KAR(IIC)
              IATIC(3)=LAR(IIC)
           ENDIF
        ELSE IF(WRD.EQ.'DIHE') THEN
           ! Do dihedral basis
           !
           NATIC=4
           IATIC(1)=IAR(IIC)
           IATIC(2)=JAR(IIC)
           IATIC(3)=KAR(IIC)
           IATIC(4)=LAR(IIC)
        ELSE
           CALL WRNDIE(0,'<REDBAS>','ILLEGAL BASIS OPTION')
           RETURN
        ENDIF
        OK=.TRUE.
        DO I=1,NATIC
           IF(IATIC(I).LE.0) OK=.FALSE.
        ENDDO

        IF(OK) THEN
           IF(NFREQ.EQ.NNMDS) THEN
              ! too many vectors
              CALL WRNDIE(-1,'<REDBAS>', &
                   'TOO MANY VECTORS, SOME IGNORED')
              RETURN
           ENDIF

           CALL INTEMD(NATIC,IATIC,X,Y,Z,XNEW,YNEW,ZNEW,NAT3, &
                DDSCR,DDM,.FALSE.,OK)
           IF(OK) THEN
              Q=1.0
              IF(LORTH) THEN
                 DO I=1,NFREQ
                    CALL ORTHOG(DDSCR,DDV(1,I),NAT3)
                 ENDDO
                 CALL DOTPR(DDSCR,DDSCR,NAT3,Q)
              ENDIF
           ELSE
              Q=0.0
           ENDIF
           IF(Q.GT.TOL) THEN
              NFREQ=NFREQ+1
              IF(PRNLEV.GE.2) THEN
                 WRITE(OUTU,21) NFREQ,Q,(IATIC(I),I=1,NATIC)
              ENDIF
21            FORMAT(' Basis vector',I4,' added. Q=',F14.10, &
                   '  Atoms:',4I4)
              DDF(NFREQ)=0.0
              DDEV(NFREQ)=Q
              ddv(1:nat3,nfreq) = DDSCR(1:NAT3)
              CALL NORMALL(DDV(1,NFREQ),NAT3)
           ELSE
              IF(WRNLEV.GE.2) THEN
                 WRITE(OUTU,22) Q,(IATIC(I),I=1,NATIC)
              ENDIF
22            FORMAT(' WARNING: Basis vector rejected. Q=',D10.4, &
                   '  Atoms:',4I4)
           ENDIF
        ENDIF
     ENDDO
     !=======================================================================
  ELSE IF(WRD.EQ.'TR') THEN
     ! Process trans-rot basis
     REMTR=(INDXA(COMLYN,COMLEN,'BALA').GT.0)
     IF(NFREQ+6.GT.NNMDS) THEN
        ! too many vectors
        CALL WRNDIE(-1,'<REDBAS>','TOO MANY VECTORS, SOME IGNORED')
        RETURN
     ENDIF

     call chmalloc('redbas.src','REDBAS','DDTROT',NAT3,6,crl=DDTROT)
     call chmalloc('redbas.src','REDBAS','ISLCT',NATOM,intg=ISLCT)
     IF(REMTR) THEN
        ISLCT(1:NATOM)=1
        CALL GETTR(X,Y,Z,ISLCT,NAT3,DDSCR,DDM,NTROT,DDTROT, &
             .FALSE.,AMASS)
     ENDIF
     CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
     IS=NFREQ+1
     CALL GETTR(X,Y,Z,ISLCT,NAT3,DDSCR,DDM,IQ,DDV(1,IS), &
          .FALSE.,AMASS)
     call chmdealloc('redbas.src','REDBAS','ISLCT',NATOM,intg=ISLCT)
     IQ=NFREQ+IQ
     DO IMODE=IS,IQ
        IF(REMTR) THEN
           DO ITROT=1,NTROT
              CALL ORTHOG(DDV(1,IMODE),DDTROT(1,ITROT),NAT3)
           ENDDO
        ENDIF
        IF(LORTH) THEN
           DO I=1,NFREQ
              CALL ORTHOG(DDV(1,IMODE),DDV(1,I),NAT3)
           ENDDO
        ENDIF
        CALL DOTPR(DDV(1,IMODE),DDV(1,IMODE),NAT3,Q)
        IF(Q.GT.TOL) THEN
           NFREQ=NFREQ+1
           IF(PRNLEV.GE.2) WRITE(OUTU,31) NFREQ,Q
31         FORMAT(' Basis vector',I4,' added. Q=',F14.10)
           DDF(NFREQ)=0.0
           DDEV(NFREQ)=Q
           IF(NFREQ.NE.IMODE) THEN
              ddv(1:nat3,nfreq) = DDV(1:nat3,IMODE)
           ENDIF
           CALL NORMALL(DDV(1,NFREQ),NAT3)
        ELSE
           IF(WRNLEV.GE.2) WRITE(OUTU,32) Q
32         FORMAT(' WARNING: Basis vector rejected. Q=',D10.4)
        ENDIF
     ENDDO
     call chmdealloc('redbas.src','REDBAS','DDTROT',NAT3,6,crl=DDTROT)
     !=======================================================================
  ELSE
     CALL WRNDIE(0,'<REDBAS>','ILLEGAL BASIS OPTION')
     RETURN
  ENDIF

  RETURN
END SUBROUTINE REDBAS

SUBROUTINE RBDIAG(X,Y,Z,NAT3,BNBND,BIMAG, &
     IUNBAS,IUNTRN,NDIM,NFREQ,AMASS,DDV,DDM,DDF,DDEV,DDSCR, &
     DD1,DD5,DDS,DDV2,LBIG,NADD,LFIX,IMOVE, &
     LFINIT,STEP,LNOVEC,LDSCF, &
     NSAVDD1,QREST) ! JZ_UW12
  !
  ! This routine does a reduced basis diagonalization and
  ! saves the eigenvectors in the original basis. The basis functions
  ! must be on a file pointed to by IUNBAS.
  !
  ! Bernard R. Brooks      10-Nov-1984
  !
  ! QC_UW_06 adds LFINIT,STEP,LNOVEC in the calling argement list
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use deriv
  use consta
  use ctitla
  use energym
  use parallel,only: gcomb
  use stream
  use timerm
#if KEY_DIMB==1
  use dimb  
#endif
  use vector
  use vibcom
  use vibio, only: wrtnmd
  use memory
  use parallel,only: psnd8
#if KEY_DIMB==1
  use dimbutils,only:rleg2      
#endif
  use machutil,only:timrb,timre
  implicit none

  INTEGER NAT3,NFREQ,NADD,IUNBAS,IUNTRN,NDIM
  type(nonbondDataStructure) BNBND
  type(imageDataStructure) BIMAG
  INTEGER IMOVE(*)
  real(chm_real) :: X(:), Y(:), Z(:)
  real(chm_real) AMASS(*)
  real(chm_real) DDV(NAT3,*),DDM(*),DDF(*),DDEV(*),DDSCR(NAT3)
  real(chm_real) DD1(:)
  real(chm_real) DD5(*),DDS(*),DDV2(*)
  LOGICAL LBIG,LFIX
  ! QC_UW_06
  LOGICAL LFINIT,LNOVEC,LDSCF
  real(chm_real)  STEP
  INTEGER NSAVDD1,NPT ! JZ_UW12
  LOGICAL QREST       ! JZ_UW12

  INTEGER ICNTRL(20)
  CHARACTER(len=4) :: HDR
  INTEGER NATOM,NATP,N6,I,IPT,J
  INTEGER IH1,IH2,IH3,IH4,IH5,IH6,IH7,IH8
  INTEGER IATOM,JATOM,K
  real(chm_real) VAL
  ! QC_UW_06
  real(chm_real),allocatable,dimension(:) :: XREF,YREF,ZREF
  real(chm_real),allocatable,dimension(:) :: DXF,DYF,DZF

  NATOM=NAT3/3

  IF(TIMER.GT.0) CALL TIMRB
  IF(PRNLEV.GE.2) WRITE(OUTU,33)
33 FORMAT(' REDUCE: Generating the second derivative matrix')

#if KEY_DIMB==1
  IF(QCMPCT) THEN
     N6=NATOM*6+LENCMP*9
  ELSE
#endif /*  DIMB*/
     N6=(NAT3*(NAT3+1))/2
#if KEY_DIMB==1
  ENDIF  
#endif

  ! QC: ADDS FINITE DIFFERENCE OPTION HERE
  IF (LFINIT) THEN
     ! ============================================================
     CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
     call chmalloc('redbas.src','RBDIAG','XREF',NATOM,crl=XREF)
     call chmalloc('redbas.src','RBDIAG','YREF',NATOM,crl=YREF)
     call chmalloc('redbas.src','RBDIAG','ZREF',NATOM,crl=ZREF)
     call chmalloc('redbas.src','RBDIAG','DXF',NATOM,crl=DXF)
     call chmalloc('redbas.src','RBDIAG','DYF',NATOM,crl=DYF)
     call chmalloc('redbas.src','RBDIAG','DZF',NATOM,crl=DZF)

     ! QC: IF WE FIX ANYTHING, WE DONOT GENERATE DD1 AT ALL,
     ! RATHER, WE GENERATE DD5 DIRECTLY!
     IF (LFIX) THEN
        WRITE(OUTU,*) "REDIAG> SOME ATOMS ARE FIXED IN SPACE"
        WRITE(OUTU,*) "REDIAG> CONSTRUCT SMALL HESSIAN DIRECTLY "
        WRITE(OUTU,*) "REDIAG> USING NUMERICAL DERIVATIVES "
        CALL GENSD3(NATOM,NDIM,IMOVE,X,Y,Z,XREF,YREF, &
             ZREF,DD5,DXF,DYF,DZF,STEP,NSAVDD1,QREST)

        call chmdealloc('redbas.src','RBDIAG','DZF',NATOM,crl=DZF)
        call chmdealloc('redbas.src','RBDIAG','DYF',NATOM,crl=DYF)
        call chmdealloc('redbas.src','RBDIAG','DXF',NATOM,crl=DXF)
        call chmdealloc('redbas.src','RBDIAG','ZREF',NATOM,crl=ZREF)
        call chmdealloc('redbas.src','RBDIAG','YREF',NATOM,crl=YREF)
        call chmdealloc('redbas.src','RBDIAG','XREF',NATOM,crl=XREF)

        ! WE MASS-WEIGHT IT HERE, ALSO DO RX PROJECTION
        ! JZ_UW12: For checking
        CALL GENIPT(NDIM,NDIM/3,3,NDIM/3,3,NPT)

        CALL RAISE2(DD5,DDM,NATOM,IMOVE,.TRUE.)

        ! THEN JUMP TO DIAGONALIZATION.

        GO TO 666

     ELSE
#if KEY_DIMB==1
        IF(QCMPCT) THEN
           CALL GENSD4(NATOM,X,Y,Z,XREF,YREF, &
                ZREF,DD1,DXF,DYF,DZF, &
                STEP,PINBCM,PJNBCM)

        ELSE
#endif /*  DIMB*/
           CALL GENSD2(NAT3,X,Y,Z,XREF,YREF,ZREF, &
                DD1,DXF,DYF,DZF,STEP,LDSCF,NSAVDD1,QREST)
#if KEY_DIMB==1
        ENDIF  
#endif

        call chmdealloc('redbas.src','RBDIAG','DZF',NATOM,crl=DZF)
        call chmdealloc('redbas.src','RBDIAG','DYF',NATOM,crl=DYF)
        call chmdealloc('redbas.src','RBDIAG','DXF',NATOM,crl=DXF)
        call chmdealloc('redbas.src','RBDIAG','ZREF',NATOM,crl=ZREF)
        call chmdealloc('redbas.src','RBDIAG','YREF',NATOM,crl=YREF)
        call chmdealloc('redbas.src','RBDIAG','XREF',NATOM,crl=XREF)
     ENDIF !LFIX

     ! ============================================================
  ELSE

     DD1(1:N6)=0.0

     CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1,NAT3,DD1)
  ENDIF
  ! QC_UW_06 Done

#if KEY_PARALLEL==1
  CALL VDGBR(DX,DY,DZ,1)
  CALL GCOMB(DD1,N6)
#endif 

  IF(PRNLEV.GE.2) THEN
     CALL PRINTE(OUTU, EPROP, ETERM, 'VIBR', 'ENR', .TRUE., &
          1, ZERO, ZERO, .TRUE.)
  ENDIF

  IF(TIMER.GT.0) THEN
     IF(PRNLEV.GE.2) WRITE(OUTU,39)
39   FORMAT('      Timing for this step:')
     CALL TIMRE
     CALL TIMRB
  ENDIF
  IF(PRNLEV.GE.2) WRITE(OUTU,34)
34 FORMAT(' REDUCE: Generating the reduced basis force', &
       ' constant matrix')

#if KEY_DIMB==1
  IF(QCMPCT) THEN
     CALL MASSDD(DD1,DDM,PINBCM,PJNBCM,NATOM)
  ELSE
#endif /*  DIMB*/

     CALL RAISE(.FALSE.,.FALSE.,NAT3,NATOM,.FALSE.,0,DD1,DDM, &
          0,0,0,0,0,0,0,.FALSE.,.TRUE.)
#if KEY_DIMB==1
  ENDIF  
#endif

  IF(LBIG) THEN
     !
     ! PROCESS THE MATRIX WITH VECTORS NOT IN MEMORY!
     !
     IPT=0
     DO I=1,NDIM
        IF(IOLEV.GT.0) THEN
           REWIND (UNIT=IUNBAS)
           READ(IUNBAS) HDR,ICNTRL
           CALL RDTITL(TITLEB,NTITLB,IUNBAS,-1)
           READ(IUNBAS)
           READ(IUNBAS)

           DO J=1,I
              READ(IUNBAS) DDSCR
           ENDDO
        ENDIF
#if KEY_PARALLEL==1
        CALL PSND8(DDSCR,NAT3)
#endif 

#if KEY_DIMB==1
        IF(QCMPCT) THEN
           CALL RLEG2(DDSCR,DDV,NATOM,DD1,PINBCM,PJNBCM,1)
        ELSE
#endif /*  DIMB*/
           CALL RALEG2(DDSCR,DDV,NATOM,DD1)
#if KEY_DIMB==1
        ENDIF  
#endif
        DO J=I,NDIM
           IPT=IPT+1
           IF(J.NE.I) THEN
              IF(IOLEV.GT.0) READ(IUNBAS) DDSCR
#if KEY_PARALLEL==1
              CALL PSND8(DDSCR,NAT3)
#endif 
           ENDIF
           CALL DOTPR(DDSCR,DDV,NAT3,DD5(IPT))
           IF(PRNLEV.GE.2) WRITE(OUTU,234) I,J,DD5(IPT)
234        FORMAT(2I5,F20.10)
        ENDDO
     ENDDO

  ELSE
     !
     ! PROCESS THE MATRIX WITH VECTORS IN MEMORY
     !
     IF(.NOT.LFIX) THEN
        IPT=0
        DO I=1,NDIM
#if KEY_DIMB==1
           IF(QCMPCT) THEN
              CALL RLEG2(DDV(1,I),DDSCR,NATOM,DD1,PINBCM,PJNBCM,1)
           ELSE
#endif /*  DIMB*/
              CALL RALEG2(DDV(1,I),DDSCR,NATOM,DD1)
#if KEY_DIMB==1
           ENDIF  
#endif

           DO J=I,NDIM
              IPT=IPT+1
              CALL DOTPR(DDSCR,DDV(1,J),NAT3,DD5(IPT))
              ! IF(PRNLEV.GE.2) WRITE(10,234) I,J,DD5(IPT)
           ENDDO
        ENDDO
     ELSE
        IPT=0
        DO IATOM=1,NATOM
           IF(IMOVE(IATOM).EQ.0) THEN
              DO J=1,3
                 DDV(1:nat3,1)=zero
                 DDV((IATOM-1)*3+J,1)=ONE

#if KEY_DIMB==1
                 IF(QCMPCT) THEN
                    CALL RLEG2(DDV(1,1),DDSCR,NATOM,DD1,PINBCM,PJNBCM,1)
                 ELSE
#endif /*  DIMB*/
                    CALL RALEG2(DDV(1,1),DDSCR,NATOM,DD1)
#if KEY_DIMB==1
                 ENDIF  
#endif

                 DO K=J,3
                    IPT=IPT+1
                    DDV(1:nat3,1)=zero
                    DDV((IATOM-1)*3+K,1)=ONE
                    CALL DOTPR(DDSCR,DDV(1,1),NAT3,DD5(IPT))
                 ENDDO

                 DO JATOM=IATOM+1,NATOM
                    IF (IMOVE(JATOM).EQ.0) THEN
                       DO K=1,3
                          IPT=IPT+1
                          DDV(1:nat3,1)=zero
                          DDV((JATOM-1)*3+K,1)=ONE
                          CALL DOTPR(DDSCR,DDV(1,1),NAT3,DD5(IPT))
                       ENDDO
                    ENDIF
                 ENDDO
              ENDDO
           ENDIF
        ENDDO
     ENDIF
  ENDIF

  IF(TIMER.GT.0) THEN
     IF(PRNLEV.GE.2) WRITE(OUTU,39)
     CALL TIMRE
     CALL TIMRB
  ENDIF

  ! QC_UW_06
666 CONTINUE

  IF(PRNLEV.GE.2) WRITE(OUTU,35)
35 FORMAT(' REDUCE: Diagonalizing the reduced force', &
       ' constant matrix')

  IH1=1
  NATP=NDIM+1
  IH2=IH1+NATP
  IH3=IH2+NATP
  IH4=IH3+NATP
  IH5=IH4+NATP
  IH6=IH5+NATP
  IH7=IH6+NATP
  IH8=IH7+NATP
  CALL DIAGQ(NDIM,NFREQ,DD5,DDV2,DDS(IH2),DDS(IH3), &
       DDS(IH4),DDS(IH5),DDS,DDS(IH6),DDS(IH7),DDS(IH8),NADD)

  DO I=1,NFREQ
     DDEV(I)=DDS(I)
     IF(PRNLEV.GE.2) WRITE(OUTU,234) I,I,DDS(I)
     DDF(I)=CNVFRQ*SQRT(ABS(DDEV(I)))
     IF(DDEV(I).LT.0.0) DDF(I)=-DDF(I)
  ENDDO

  IF(TIMER.GT.0) THEN
     IF(PRNLEV.GE.2) WRITE(OUTU,39)
     CALL TIMRE
     CALL TIMRB
  ENDIF

  ! QC: UW_06
  ! Return if we don't want to print out eigenvectors
  ! But move the frequency print out to here

  IF(PRNLEV.GE.2) THEN
     WRITE(OUTU,625)
625  FORMAT(/15X,'FREQUENCIES'/)
     CALL FREQPR(DDF,NFREQ,OUTU)
  ENDIF

  IF (LNOVEC) RETURN

  IF(IUNTRN.GT.0) THEN

     ! SAVE TRANSFORMATION MATRIX FOR EXTERNAL BACKTRANSFORMATION
     IF(PRNLEV.GE.2) WRITE(OUTU,46) IUNTRN
46   FORMAT(' REDUCE: Saving the transformation matrix to unit',I4)
     CALL WRTNMD(.FALSE.,1,NDIM,NDIM,DDV2,DDSCR,DDEV,IUNTRN,AMASS)
  ENDIF

  IF(.NOT.LBIG) THEN
     IF(PRNLEV.GE.2) WRITE(OUTU,36)
36   FORMAT(' REDUCE: Backtransforming to the original basis')

     IF(.NOT.LFIX) THEN
        IF(IOLEV.GT.0) THEN
           IF (reallow) THEN       
              REWIND (UNIT=IUNBAS)
           ENDIF                   
           READ(IUNBAS) HDR,ICNTRL
           CALL RDTITL(TITLEB,NTITLB,IUNBAS,-1)
           READ(IUNBAS)
           READ(IUNBAS)
        ENDIF

        CALL RBDIA2(NAT3,NFREQ,NDIM,DDV,DDV2,DDSCR,IUNBAS)

     ELSE
        DO I=1,NFREQ
           DDV(1:nat3,I)=zero
        ENDDO
        IPT=0
        DO IATOM=1,NATOM
           IF(IMOVE(IATOM).EQ.0) THEN
              DO J=1,3
                 IPT=IPT+1
                 DDSCR(1:NAT3)=zero
                 DDSCR((IATOM-1)*3+J)=ONE
                 DO K=1,NFREQ
                    VAL=DDV2((K-1)*NDIM+IPT)
                    CALL ADDCTV(DDV(1,K),DDSCR,NAT3,VAL)
                 ENDDO
              ENDDO
           ENDIF
        ENDDO
     ENDIF
  ENDIF

  IF(TIMER.GT.0) THEN
     IF(PRNLEV.GE.2) WRITE(OUTU,39)
     CALL TIMRE
     CALL TIMRB
  ENDIF
  IF(PRNLEV.GE.2) WRITE(OUTU,37)
37 FORMAT(' REDUCE: Calculation completed')

  IF(LBIG) NFREQ=0
  RETURN
END SUBROUTINE RBDIAG

SUBROUTINE RBDIA2(NAT3,NF,NDIM,DDV,DDV2,DDSCR,IUNIT)
  !
  ! This routine back tranforms the eigenvectors from the reduced
  ! basis to the original basis. Extra copying is here due to
  ! duplicate memory usage and space saving tricks.
  !
  ! Bernard R. Brooks       10-Nov-1984
  !
  use chm_kinds
  use stream
  use number
  use vector
  use parallel,only: psnd8
  implicit none

  INTEGER NAT3,NF,NDIM,IUNIT
  real(chm_real) DDV(NAT3,NF),DDV2(NDIM,NF),DDSCR(NAT3)

  INTEGER I,IDIM
  real(chm_real) VAL

  DO I=1,NF
     DDV(1:nat3,I)=zero
  ENDDO

  DO IDIM=1,NDIM
     IF(IOLEV.GT.0) READ(IUNIT) DDSCR
#if KEY_PARALLEL==1
     CALL PSND8(DDSCR,NAT3)
#endif 
     DO I=1,NF
        VAL=DDV2(IDIM,I)
        CALL ADDCTV(DDV(1,I),DDSCR,NAT3,VAL)
     ENDDO
  ENDDO

  RETURN
END SUBROUTINE RBDIA2

end module redbas_m

