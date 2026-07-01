module nmrm
  private
  public :: nmr, sel1at
contains
#if KEY_NOMISC==1 /*nmrmain*/
  SUBROUTINE NMR
    CALL WRNDIE(-1,'<CHARMM>','NMR code is not compiled.')
    RETURN
  END SUBROUTINE NMR
  
  SUBROUTINE SEL1AT
    RETURN
  END SUBROUTINE SEL1AT
#else /* (nmrmain)*/
  SUBROUTINE NMR
     !-----------------------------------------------------------------------
     use chm_kinds
     use dimens_fcm
     use comand
     use psf
     use stream
     use string
     use coord
     use coordc
     use memory

     implicit none
     !
     !-----------------------------------------------------------------------
     ! NMR VARIABLES
     !
     ! Chemical Shift Anisotropy Tensor
     real(chm_real),allocatable,dimension(:,:) :: ZMAT
     integer,allocatable,dimension(:,:) :: IZMAT,ICSA,IDQS,IRTIMS
     real(chm_real),allocatable,dimension(:,:) :: SIGMA
     real(chm_real),allocatable, dimension (:,:,:) :: ZCSA
     real(chm_real),allocatable,dimension(:) :: GYRMAG,DSIGMA
     real(chm_real4),allocatable,dimension(:) :: R4TEMP
     integer,allocatable,dimension(:) :: ITS
     real(chm_real),allocatable,dimension(:,:,:) :: RELAX
     integer,allocatable,dimension(:) :: ISLCT1,ISLCT2,ISLCT3

     !      INTEGER      ICSA,CSA,SIGMA,ZCSA
     !
     ! Deuterium Quadrupol Splitting
     INTEGER      DQS!,IDQS
     !
     ! Longitudinal and Tranverse Relaxation Times T1, T2, NOE, ROE, etc...
     !      INTEGER      IRTIMS,GYRMAG,DSIGMA,IRELAX
     !
     ! Temporary arrays to read trajectory files
     !      INTEGER      ITS,R4TEMP
     !
     !-----------------------------------------------------------------------
     !
     !
     !      INTEGER      ISLCT1,ISLCT2,ISLCT3
     !      INTEGER      ZMAT,IZMAT
     INTEGER      NMRMX

     ! Allocate space

     !  Pick a compromise on number of "atoms" potentially built to be
     !  Let user ask for more than NATOM if needed. LNI
     !
     NMRMX=GTRMI(COMLYN,COMLEN,'MAXA',NATOM)
     call chmalloc('nmr.src','NMR','ISLCT1',NMRMX,intg=ISLCT1)
     call chmalloc('nmr.src','NMR','ISLCT2',NMRMX,intg=ISLCT2)
     call chmalloc('nmr.src','NMR','ISLCT3',NMRMX,intg=ISLCT3)
     call chmalloc('nmr.src','NMR','ZMAT',3,NMRMX,crl=ZMAT)
     call chmalloc('nmr.src','NMR','IZMAT',4,NMRMX,intg=IZMAT)
     call chmalloc('nmr.src','NMR','ICSA',3,NMRMX,intg=ICSA)
     call chmalloc('nmr.src','NMR','SIGMA',3,NMRMX,crl=SIGMA)
     call chmalloc('nmr.src','NMR','ZCSA',2,2,NMRMX,crl=ZCSA)
     call chmalloc('nmr.src','NMR','IDQS',2,NMRMX*NATOM/2,intg=IDQS)
     call chmalloc('nmr.src','NMR','IRTIMS',2,NMRMX,intg=IRTIMS)
     call chmalloc('nmr.src','NMR','GYRMAG',NMRMX,crl=GYRMAG)
     call chmalloc('nmr.src','NMR','DSIGMA',NMRMX,crl=DSIGMA)
     call chmalloc('nmr.src','NMR','R4TEMP',NATOM,cr4=R4TEMP)
     call chmalloc('nmr.src','NMR','ITS',2*NMRMX,intg=ITS)
     call chmalloc('nmr.src','NMR','RELAX',9,2,NMRMX,crl=RELAX)
     call NMR1(NATOM,NRES,NSEG,NATOMT,NREST,NSEGT,IBASE,NICTOT,ATYPE,RESID, &
          RES,SEGID,X,Y,Z,WMAIN,AMASS,XCOMP,YCOMP,ZCOMP,WCOMP,ISLCT1, &
          ISLCT2,ISLCT3,ZMAT,IZMAT,ICSA,SIGMA,ZCSA, &
          IDQS,IRTIMS,GYRMAG,DSIGMA,R4TEMP,ITS,NMRMX,RELAX)
     !Free space
     call chmdealloc('nmr.src','NMR','ISLCT1',NMRMX,intg=ISLCT1)
     call chmdealloc('nmr.src','NMR','ISLCT2',NMRMX,intg=ISLCT2)
     call chmdealloc('nmr.src','NMR','ISLCT3',NMRMX,intg=ISLCT3)
     call chmdealloc('nmr.src','NMR','ZMAT',3,NMRMX,crl=ZMAT)
     call chmdealloc('nmr.src','NMR','IZMAT',4,NMRMX,intg=IZMAT)
     call chmdealloc('nmr.src','NMR','ICSA',3,NMRMX,intg=ICSA)
     call chmdealloc('nmr.src','NMR','SIGMA',3,NMRMX,crl=SIGMA)
     call chmdealloc('nmr.src','NMR','ZCSA',2,2,NMRMX,crl=ZCSA)
     call chmdealloc('nmr.src','NMR','IDQS',2,NMRMX*NATOM/2,intg=IDQS)
     call chmdealloc('nmr.src','NMR','IRTIMS',2,NMRMX,intg=IRTIMS)
     call chmdealloc('nmr.src','NMR','GYRMAG',NMRMX,crl=GYRMAG)
     call chmdealloc('nmr.src','NMR','DSIGMA',NMRMX,crl=DSIGMA)
     call chmdealloc('nmr.src','NMR','R4TEMP',NATOM,cr4=R4TEMP)
     call chmdealloc('nmr.src','NMR','ITS',2*NMRMX,intg=ITS)
     call chmdealloc('nmr.src','NMR','RELAX',9,2,NMRMX,crl=RELAX)
     RETURN
  END SUBROUTINE NMR

  SUBROUTINE NMR1(NATOM,NRES,NSEG,NATOMT,NREST,NSEGT, &
       IBASE,NICTOT,ATYPE,RESID,RES,SEGID, &
       X,Y,Z,WMAIN,AMASS,XCOMP,YCOMP,ZCOMP,WCOMP, &
       ISLCT1,ISLCT2,ISLCT3, &
       ZMAT,IZMAT, &
       ICSA,SIGMA,ZCSA, &
       IDQS, &
       IRTIMS,GYRMAG,DSIGMA, &
       R4TEMP,ITS,NMRMX,RELAX)
    !-----------------------------------------------------------------------
    !
    !  NMR ANALYSIS MAIN SUBROUTINE
    !
    use chm_kinds
    use dimens_fcm
    use number
    use exfunc
    use string
    use stream
    use comand
    use memory
    use ctitla
    use chutil,only:getres,atomid
    use coorio_mod,only:cwrite
    use cvio,only:trjspc

    implicit none
    !-----------------------------------------------------------------------
    ! General PSF and COOR information
    real(chm_real4),allocatable,dimension(:,:) :: XTSERI,YTSERI,ZTSERI
    real(chm_real),allocatable,dimension(:) :: CORR12,WORK1,WORK2,WORK3,WORK4
    real(chm_real),allocatable,dimension(:) :: CSA,CSA1,CSA2, &
         CSAPAR,CSAPA1,CSAPA2,CSAPER,CSAPE1,CSAPE2
    real(chm_real),allocatable,dimension(:) :: DQS,DQS1,DQS2,BMASS
    integer,allocatable,dimension(:,:) :: ATOMPR
    INTEGER       NATOM,NRES,NSEG,NATOMT,NREST,NSEGT
    INTEGER       IBASE(*),NICTOT(*)
    CHARACTER(len=*) ATYPE(*),RESID(*),RES(*),SEGID(*)
    real(chm_real)        X(*),Y(*),Z(*),WMAIN(*),AMASS(*)
    real(chm_real)        XCOMP(*),YCOMP(*),ZCOMP(*),WCOMP(*)

    ! Construction of atoms
    INTEGER       ISLCT1(*),ISLCT2(*),ISLCT3(*)
    real(chm_real)        ZMAT(3,*)
    INTEGER       NZMAT,IZMAT(4,*),NMRMX
    !
    ! Chemical Shift Anisotropy Tensor
    INTEGER       NCSA ,ICSA(3,*)
    !      INTEGER       CSAPAR,CSAPA1,CSAPA2
    !      INTEGER       CSAPER,CSAPE1,CSAPE2
    real(chm_real)        SIGMA(3,*),ZCSA(2,2,*)
    !
    ! Deuterium Quadrupol Splitting
    INTEGER       NDQS,IDQS(2,*)!,DQS,DQS1,DQS2
    !
    ! Transverse and Longitudinal Relaxation Times  T1 and T2
    ! and Cross-Relaxation Rates NOE and ROE
    real(chm_real)        RTUMBL,HFIELD,TMAX
    INTEGER       NRTIMS(2),IRTIMS(2,*)
    real(chm_real)        GYRMAG(*),GYRMAG2,DSIGMA(*),DSIGMA2
    real(chm_real)        CTRTIM
    real(chm_real)        RELAX(9,2,*)
    !
    ! Temporary arrays for trajectory files and time series
    REAL(CHM_REAL4)        R4TEMP(*)
    INTEGER       NTS,ITS(*)
    INTEGER       NBFRAM  !,XTSERI,YTSERI,ZTSERI
    !      INTEGER       CORR12,WORK1,WORK2,WORK3,WORK4
    !-----------------------------------------------------------------------
    !  Miscelaneous Local variables
    INTEGER       NATOMX,NRESX,NSEGX
    INTEGER       IOMODE,ICNTRL(20)
    INTEGER       NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP
    CHARACTER(len=4) WRD
    LOGICAL       DONE,EOF,LUSED,OK,ORIENT,LMASS,LWMAIN
    !      INTEGER       BMASS, ATOMPR
    INTEGER       I,I1,I2,I3,IATOM,IRES,J,K
    INTEGER       IWRIT,ILIST,IMF, IMFDAT, ISDLIST, NAVE
    CHARACTER(len=8) ATNAM,SGID,RID,REN,AC
    real(chm_real)        S11,S22,S33,THE1,THE2,PHI1,PHI2
    real(chm_real)        CTDI2,DIST2,DSIGMA3
    LOGICAL       QCORRE,QTSERI,QPROP,QUVECT,QVERB,CSAR,QSAVE,QOF
    LOGICAL       QPRSTAT
    !
    OK=.TRUE.
    LUSED=.FALSE.
    DONE=.FALSE.
    EOF=.FALSE.
    QCORRE = .FALSE.
    QTSERI = .FALSE.
    QPROP  = .FALSE.
    QUVECT = .FALSE.
    QVERB  = .FALSE.
    ! Averaging of relaxation parameters
    QSAVE  = .FALSE.
    QPRSTAT = .FALSE.
    NAVE =   0
    DO I = 1, NMRMX
       DO J=1,9
          RELAX(J,1,I) = 0.0
          RELAX(J,2,I) = 0.0
       ENDDO
    ENDDO

    ! rotational tumbling time in PS
    RTUMBL=ZERO
    ! magnetic field in TESLA (note that 11.74 T yields 500 MHz for H)
    HFIELD=11.74D0
    CTRTIM=2.5
    TMAX=ZERO
    !
    IF(PRNLEV.GE.3) THEN
       WRITE(OUTU,100)
       WRITE(OUTU,100) 'NMR ANALYSIS SUBROUTINE'
    ENDIF
100 FORMAT(6X,2A)

    !-----------------------------------------------------------------------
1000 CALL XTRANE(COMLYN,COMLEN,'NMR ')
    CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE., &
         '   NMR> ')
    IF(EOF)THEN
       CALL PPSTRM(OK)
       IF(.NOT.OK)  RETURN
    ENDIF
    CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
    IF(LUSED) GOTO 1000

    IWRIT=GTRMI(COMLYN,COMLEN,'IWRI',OUTU)
    IWRIT=GTRMI(COMLYN,COMLEN,'UNIT',IWRIT)
    WRD='    '
    WRD=NEXTA4(COMLYN,COMLEN)
    IF(WRD.EQ.'    ') GOTO 1000

    !.......................................................................

    IF(WRD.EQ.'BUIL')THEN
       !        -------------

       NZMAT=NZMAT+1
       ATNAM=NEXTA8(COMLYN,COMLEN)
       ZMAT(1,NZMAT)=GTRMF(COMLYN,COMLEN,'DIST',ONE)
       ZMAT(2,NZMAT)=GTRMF(COMLYN,COMLEN,'THET',ONE)
       ZMAT(3,NZMAT)=GTRMF(COMLYN,COMLEN,'DIHE',ZERO)
       IZMAT(4,NZMAT)=NATOM+NZMAT
       CALL SEL1AT(COMLYN,COMLEN,IATOM,ISLCT1, &
            NATOMT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID,X,Y,Z,WMAIN)
       IZMAT(3,NZMAT)=IATOM
       CALL SEL1AT(COMLYN,COMLEN,IATOM,ISLCT1, &
            NATOMT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID,X,Y,Z,WMAIN)
       IZMAT(2,NZMAT)=IATOM
       CALL SEL1AT(COMLYN,COMLEN,IATOM,ISLCT1, &
            NATOMT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID,X,Y,Z,WMAIN)
       IZMAT(1,NZMAT)=IATOM
       IRES=GETRES(IZMAT(3,NZMAT),IBASE,NRES)
       CALL ATOMID(IZMAT(3,NZMAT),SGID,RID,REN,AC)
       IF(NATOM+NZMAT.GT.NMRMX) THEN
          CALL WRNDIE(-3,'<NMR1>', &
               'Insufficient allocation - increase NMR MAXA n')
       ENDIF
       NATOMT=NATOM+NZMAT
       NREST=NRES+NZMAT
       NSEGT=NSEG+1
       IBASE(NREST+1)=IBASE(NRES+1)+NZMAT
       NICTOT(NSEGT+1)=NICTOT(NSEG+1)+NZMAT
       ATYPE(IZMAT(4,NZMAT))=ATNAM
       RESID(NREST)=RID
       RES(NREST)=REN
       SEGID(NSEGT)='BUIL'
       IF(PRNLEV.GT.3) WRITE(OUTU,204) 'BUILD',NZMAT,' ZMATRIX ', &
            IZMAT(1,NZMAT), IZMAT(2,NZMAT), ZMAT(1,NZMAT), &
            IZMAT(3,NZMAT), ZMAT(2,NZMAT), IZMAT(4,NZMAT), ZMAT(3,NZMAT)
204    FORMAT(6X,A,I4,A,2I4,F9.3,I4,F9.3,I4,F9.3)
       IATOM=IZMAT(4,NZMAT)
       IF(PRNLEV.GT.3) WRITE(OUTU,100) 'CONSTRUCTED ATOMIC COORDINATES'
       CALL ZCONSTR(.FALSE.,NZMAT,NZMAT,IZMAT,ZMAT,X,Y,Z)
       IF(PRNLEV.GT.3) WRITE(OUTU,203) IATOM,IRES,REN(1:idleng), &
            ATYPE(IATOM)(1:idleng),X(IATOM),Y(IATOM),Z(IATOM), &
            SGID(1:idleng),RID(1:idleng)
203    FORMAT(2I5,2(1X,A),3F10.5,2(1X,A))


       !.......................................................................

    ELSEIF(WRD.EQ.'CSA ')THEN
       !            -------------

       IF(PRNLEV.GT.3) WRITE(OUTU,100) &
            'NMR PROPERTY:  CHEMICAL SHIFT ANISOTROPY'
       S11=GTRMF(COMLYN,COMLEN,'S11 ',ZERO)
       S22=GTRMF(COMLYN,COMLEN,'S22 ',ZERO)
       S33=GTRMF(COMLYN,COMLEN,'S33 ',ZERO)
       THE1=GTRMF(COMLYN,COMLEN,'THE1',ZERO)
       PHI1=GTRMF(COMLYN,COMLEN,'PHI1',ZERO)
       THE2=GTRMF(COMLYN,COMLEN,'THE2',ZERO)
       PHI2=GTRMF(COMLYN,COMLEN,'PHI2',ZERO)
       CALL SEL1AT(COMLYN,COMLEN,I1,ISLCT1, &
            NATOMT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID,X,Y,Z,WMAIN)
       CALL SEL1AT(COMLYN,COMLEN,I2,ISLCT1, &
            NATOMT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID,X,Y,Z,WMAIN)
       CALL SEL1AT(COMLYN,COMLEN,I3,ISLCT1, &
            NATOMT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID,X,Y,Z,WMAIN)
       IF((I1.NE.0).AND.(I2.NE.0).AND.(I3.NE.0))THEN
          NCSA=NCSA+1
          ICSA(1,NCSA)=I1
          ICSA(2,NCSA)=I2
          ICSA(3,NCSA)=I3
          SIGMA(1,NCSA)=S11
          SIGMA(2,NCSA)=S22
          SIGMA(3,NCSA)=S33
          ZCSA(1,1,NCSA)=THE1
          ZCSA(1,2,NCSA)=PHI1
          ZCSA(2,1,NCSA)=THE2
          ZCSA(2,2,NCSA)=PHI2
          IF(PRNLEV.GT.3) THEN
             WRITE(OUTU,'(1X,3F8.3)')  (SIGMA(I,NCSA),I=1,3)
             WRITE(OUTU,'(1X,3I8)')    (ICSA(I,NCSA),I=1,3)
             WRITE(OUTU,'(1X,3F8.3)')  (ZCSA(1,I,NCSA),I=1,2)
             WRITE(OUTU,'(1X,3F8.3)')  (ZCSA(2,I,NCSA),I=1,2)
          ENDIF
       ENDIF

       !.......................................................................

    ELSEIF(WRD.EQ.'DQS ')THEN
       !            -------------

       IF(PRNLEV.GT.3) WRITE(OUTU,100) &
            'NMR PROPERTY:  DEUTERIUM QUADRUPOLE SPLITTING'
       CTDI2=1.35
       CTDI2=GTRMF(COMLYN,COMLEN,'CTDI',CTDI2)
       CTDI2=CTDI2**2
       CALL SEL123(COMLYN,COMLEN,2,ISLCT1,ISLCT2,ISLCT3, &
            NATOMT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID,X,Y,Z,WMAIN)
       DO I1=1,NATOMT
          IF(ISLCT1(I1).NE.0)THEN
             DO I2=1,NATOMT
                IF(ISLCT2(I2).NE.0)THEN
                   DIST2=(X(I1)-X(I2))**2+(Y(I1)-Y(I2))**2+(Z(I1)-Z(I2))**2
                   IF(DIST2.LT.CTDI2)THEN
                      NDQS=NDQS+1
                      IDQS(1,NDQS)=I1   !NMR NUCLEUS
                      IDQS(2,NDQS)=I2
                      IF(PRNLEV.GT.3) WRITE(OUTU,'(1X,3I8)') &
                           NDQS,IDQS(1,NDQS),IDQS(2,NDQS)
                   ENDIF
                ENDIF
             ENDDO
          ENDIF
       ENDDO

       !.......................................................................

    ELSEIF(WRD.EQ.'SET ')THEN
       !            -------------

       IF(PRNLEV.GT.3) WRITE(OUTU,100) 'NMR SET:  '
       RTUMBL=GTRMF(COMLYN,COMLEN,'RTUM',RTUMBL)
       CTRTIM=GTRMF(COMLYN,COMLEN,'CUT ',CTRTIM)
       HFIELD=GTRMF(COMLYN,COMLEN,'HFIE',HFIELD)
       TMAX=GTRMF(COMLYN,COMLEN,'TMAX',TMAX)
       DSIGMA3= GTRMF(COMLYN,COMLEN,'DSIG',ZERO)
       IF(PRNLEV.GT.3) WRITE(OUTU,'(6X,2(A,4X,F12.3))') &
            'HFIELD [TESLA] = ',HFIELD, &
            'RTUMBL [PSEC]  = ',RTUMBL
       GYRMAG2=GTRMF(COMLYN,COMLEN,'GAMM',ZERO)
       CALL SEL123(COMLYN,COMLEN,1,ISLCT1,ISLCT2,ISLCT3, &
            NATOMT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID,X,Y,Z,WMAIN)
       DO I=1,NATOMT
          IF(ISLCT1(I).NE.0)THEN
             GYRMAG(I)=GYRMAG2
          ENDIF
       ENDDO

       !.......................................................................

    ELSEIF(WRD.EQ.'RTIM')THEN
       !            -------------
       !
       CSAR = INDXA(COMLYN, COMLEN, 'CSAR') .GT. 0
       IF(PRNLEV.GT.3) THEN
          WRITE(OUTU,100)
          IF(CSAR)THEN
             WRITE(OUTU,100) 'NMR PROPERTY:  CSA RELAXATION TIMES'
             DSIGMA2=GTRMF(COMLYN,COMLEN,'DSIG',ZERO)
          ELSE
             WRITE(OUTU,100) 'NMR PROPERTY:  DIPOLE-DIPOLE RELAXATION TIMES'
          ENDIF
       ENDIF
       RTUMBL=GTRMF(COMLYN,COMLEN,'RTUM',RTUMBL)
       HFIELD=GTRMF(COMLYN,COMLEN,'HFIE',HFIELD)
       IF(PRNLEV.GT.3) THEN
          WRITE(OUTU,'(6X,2(A,4X,F12.3,1X))') 'HFIELD [TESLA] = ',HFIELD, &
               'RTUMBL [PSEC]  = ',RTUMBL
          WRITE(OUTU,100)
          WRITE(OUTU,100) 'SPIN LIST:'
       ENDIF
       CALL SEL123(COMLYN,COMLEN,2,ISLCT1,ISLCT2,ISLCT3, &
            NATOMT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID,X,Y,Z,WMAIN)
       CALL SETITS(NATOMT,ISLCT1,HFIELD,RTUMBL,GYRMAG, &
            .FALSE.,ZERO,DSIGMA, &
            IRTIMS,1,NRTIMS(1),ITS,NTS,OUTU,PRNLEV)
       IF(PRNLEV.GT.3) THEN
          WRITE(OUTU,204) 'NRTIMS(1) =',NRTIMS(1)
          WRITE(OUTU,100)
       ENDIF
       CALL SETITS(NATOMT,ISLCT2,HFIELD,RTUMBL,GYRMAG, &
            CSAR,DSIGMA2,DSIGMA, &
            IRTIMS,2,NRTIMS(2),ITS,NTS,OUTU,PRNLEV)
       IF(PRNLEV.GT.3) THEN
          WRITE(OUTU,204) 'NRTIMS(2) =',NRTIMS(2)
          WRITE(OUTU,100)
       ENDIF
       !.......................................................................

    ELSEIF(WRD.EQ.'DYNA')THEN
       !            -------------
       !
       IF(PRNLEV.GT.3) THEN
          WRITE(OUTU,100)
          WRITE(OUTU,100) 'NMR DYNAMICS PROPERTIES'
          WRITE(OUTU,100)
       ENDIF
       CTRTIM = GTRMF(COMLYN,COMLEN,'CUT ',CTRTIM)
       RTUMBL = GTRMF(COMLYN,COMLEN,'RTUM',RTUMBL)
       HFIELD = GTRMF(COMLYN,COMLEN,'HFIE',HFIELD)
       TMAX   = GTRMF(COMLYN,COMLEN,'TMAX',TMAX)
       TMAX   = GTRMF(COMLYN,COMLEN,'TMAX',TMAX)
       DSIGMA3= GTRMF(COMLYN,COMLEN,'DSIG',ZERO)
       !
       ILIST = GTRMI(COMLYN,COMLEN,'ILIS',-1)
       ISDLIST = GTRMI(COMLYN,COMLEN,'ISDL',-1)
       IMF = GTRMI(COMLYN,COMLEN,'MODF',-1)
       IMFDAT = GTRMI(COMLYN,COMLEN,'MFDA',-1)
       QSAVE=INDXA(COMLYN,COMLEN,'SAVE').GT.0
       QPRSTAT=INDXA(COMLYN,COMLEN,'WRSTAT').GT.0
       !
       ! Allocate memory
       call chmalloc('nmr.src','NMR1','CSA',NCSA+1,crl=CSA)
       call chmalloc('nmr.src','NMR1','CSA1',NCSA+1,crl=CSA1)
       call chmalloc('nmr.src','NMR1','CSA2',NCSA+1,crl=CSA2)
       call chmalloc('nmr.src','NMR1','CSAPAR',NCSA+1,crl=CSAPAR)
       call chmalloc('nmr.src','NMR1','CSAPA1',NCSA+1,crl=CSAPA1)
       call chmalloc('nmr.src','NMR1','CSAPA2',NCSA+1,crl=CSAPA2)
       call chmalloc('nmr.src','NMR1','CSAPER',NCSA+1,crl=CSAPER)
       call chmalloc('nmr.src','NMR1','CSAPE1',NCSA+1,crl=CSAPE1)
       call chmalloc('nmr.src','NMR1','CSAPE2',NCSA+1,crl=CSAPE2)
       call chmalloc('nmr.src','NMR1','DQS',NDQS+1,crl=DQS)
       call chmalloc('nmr.src','NMR1','DQS1',NDQS+1,crl=DQS1)
       call chmalloc('nmr.src','NMR1','DQS2',NDQS+1,crl=DQS2)
       ORIENT = INDXA(COMLYN, COMLEN, 'ORIE') .GT. 0
       IF(ORIENT)THEN
          LMASS  = INDXA(COMLYN, COMLEN, 'MASS') .GT. 0
          LWMAIN = INDXA(COMLYN, COMLEN, 'WMAI') .GT. 0
          call chmalloc('nmr.src','NMR1','BMASS',NATOMT,crl=BMASS)
          call chmalloc('nmr.src','NMR1','ATOMPR',2,NATOMT,intg=ATOMPR)
          QOF = INDXA(COMLYN, COMLEN, 'NOCOMP') .GT. 0
          CALL SEL123(COMLYN,COMLEN,2,ISLCT1,ISLCT2,ISLCT3, &
               NATOMT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID,X,Y,Z,WMAIN)
          IF(PRNLEV.GE.2) THEN
             IF(QOF)THEN
                WRITE(OUTU,100) &
                     '* ALL COORDINATE FRAMES WILL BE REORIENTED WITH ', &
                     ' RESPECT TO FIRST USED TRAJECTORY FRAME *'
             ELSE
                WRITE(OUTU,100) &
                     '* ALL COORDINATE FRAMES WILL BE REORIENTED WITH ', &
                     ' RESPECT TO THE COMPARISON SET *'
             ENDIF
          ENDIF
          IF(LMASS)THEN
             IF(PRNLEV.GT.3) WRITE(OUTU,100)'  MASS WEIGHTING WILL BE USED '
          ENDIF
          IF(LWMAIN)THEN
             IF(PRNLEV.GT.3) WRITE(OUTU,100)'  WMAIN WEIGHTING WILL BE USED '
          ENDIF
       ELSE
          !        BMASS=1 ! SHOULD NOT BE NEEDED??? LNI
          !        ATOMPR=1
          CALL SEL123(COMLYN,COMLEN,1,ISLCT2,ISLCT2,ISLCT3, &
               NATOMT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID,X,Y,Z,WMAIN)
       ENDIF
       !
       ! Storage for time series
       ! NB: If user does not specify correct values allocation may be wrong,
       ! but not too small. Last frame has to be specified though. LN.
       CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP)
       IF(NSTOP.LE. 0) CALL WRNDIE(-2,'<NMRDYN>', 'STOP not specified')
       NBFRAM=1+(NSTOP-NBEGIN)/NSKIP
       IF(PRNLEV.GT.3) THEN
          WRITE(OUTU,'(1X,A,3I9)') &
               'NBEGIN, NSTOP, NSKIP = ',NBEGIN,NSTOP,NSKIP
          WRITE(OUTU,'(1X,A,I9)') &
               'NUMBER OF FRAMES          = ',NBFRAM
          WRITE(OUTU,'(1X,A,I9)') &
               'NUMBER OF TIME SERIES     = ',NTS
          WRITE(OUTU,'(1X,A,I9)') &
               'MEMORY STORAGE REQUIREMENTS = ',3*NBFRAM*NTS
       ENDIF
       IF(NRTIMS(1).GT.0)THEN
          ! Store the XYZ time-series
          call chmalloc('nmr.src','NMR1','XTSERI',(NBFRAM+1),NTS,cr4=XTSERI)
          call chmalloc('nmr.src','NMR1','YTSERI',(NBFRAM+1),NTS,cr4=YTSERI)
          call chmalloc('nmr.src','NMR1','ZTSERI',(NBFRAM+1),NTS,cr4=ZTSERI)
          call chmalloc('nmr.src','NMR1','CORR12',NBFRAM+1,crl=CORR12)
          call chmalloc('nmr.src','NMR1','WORK1',NBFRAM+1,crl=WORK1)
          call chmalloc('nmr.src','NMR1','WORK2',NBFRAM+1,crl=WORK2)
          call chmalloc('nmr.src','NMR1','WORK3',NBFRAM+1,crl=WORK3)
          call chmalloc('nmr.src','NMR1','WORK4',NBFRAM+1,crl=WORK4)
       ENDIF
       !
       ! Print out specifications
       QCORRE = INDXA(COMLYN, COMLEN, 'C(T)') .GT. 0
       QTSERI = INDXA(COMLYN, COMLEN, 'R(T)') .GT. 0
       QPROP  = INDXA(COMLYN, COMLEN, 'PROP') .GT. 0
       QUVECT = INDXA(COMLYN, COMLEN, 'UVEC') .GT. 0
       QVERB  = INDXA(COMLYN, COMLEN, 'VERB') .GT. 0

       call NMRDYN(NATOM,NRES,NSEG,NATOMT,NREST,NSEGT,IBASE,NICTOT,ATYPE, &
            RESID,RES,SEGID,X,Y,Z,WMAIN,XCOMP,YCOMP,ZCOMP,WCOMP,ORIENT,LMASS, &
            LWMAIN,AMASS,BMASS,ATOMPR,ISLCT1,NZMAT,ZMAT,IZMAT,ISLCT2,NCSA, &
            ICSA,SIGMA,ZCSA,CSA,CSA1,CSA2,CSAPAR,CSAPA1, &
            CSAPA2,CSAPER,CSAPE1,CSAPE2,NDQS,DQS,DQS1,DQS2, &
            IDQS,NRTIMS,IRTIMS,CTRTIM,NTS,ITS,NBFRAM,XTSERI,YTSERI, &
            ZTSERI,GYRMAG,DSIGMA,RTUMBL,HFIELD,TMAX,CORR12,WORK1, &
            WORK2,WORK3,WORK4,NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP,R4TEMP, &
            ICNTRL,ISLCT3,QCORRE,QTSERI,QPROP,QUVECT,QVERB,PRNLEV,IWRIT,IOLEV, &
            ILIST,ISDLIST,IMF,IMFDAT,DSIGMA3,RELAX,NAVE,QOF,QSAVE,QPRSTAT)
       !
       call chmdealloc('nmr.src','NMR1','CSA',NCSA+1,crl=CSA)
       call chmdealloc('nmr.src','NMR1','CSA1',NCSA+1,crl=CSA1)
       call chmdealloc('nmr.src','NMR1','CSA2',NCSA+1,crl=CSA2)
       call chmdealloc('nmr.src','NMR1','CSAPAR',NCSA+1,crl=CSAPAR)
       call chmdealloc('nmr.src','NMR1','CSAPA1',NCSA+1,crl=CSAPA1)
       call chmdealloc('nmr.src','NMR1','CSAPA2',NCSA+1,crl=CSAPA2)
       call chmdealloc('nmr.src','NMR1','CSAPER',NCSA+1,crl=CSAPER)
       call chmdealloc('nmr.src','NMR1','CSAPE1',NCSA+1,crl=CSAPE1)
       call chmdealloc('nmr.src','NMR1','CSAPE2',NCSA+1,crl=CSAPE2)
       call chmdealloc('nmr.src','NMR1','DQS',NDQS+1,crl=DQS)
       call chmdealloc('nmr.src','NMR1','DQS1',NDQS+1,crl=DQS1)
       call chmdealloc('nmr.src','NMR1','DQS2',NDQS+1,crl=DQS2)
       IF(ORIENT)THEN
          call chmdealloc('nmr.src','NMR1','BMASS',NATOMT,crl=BMASS)
          call chmdealloc('nmr.src','NMR1','ATOMPR',2,NATOMT,intg=ATOMPR)
       ENDIF
       IF(NRTIMS(1).GT.0)THEN
          ! Free time series memory
          call chmdealloc('nmr.src','NMR1','XTSERI',(NBFRAM+1),NTS,cr4=XTSERI)
          call chmdealloc('nmr.src','NMR1','YTSERI',(NBFRAM+1),NTS,cr4=YTSERI)
          call chmdealloc('nmr.src','NMR1','ZTSERI',(NBFRAM+1),NTS,cr4=ZTSERI)
          call chmdealloc('nmr.src','NMR1','CORR12',NBFRAM+1,crl=CORR12)
          call chmdealloc('nmr.src','NMR1','WORK1',NBFRAM+1,crl=WORK1)
          call chmdealloc('nmr.src','NMR1','WORK2',NBFRAM+1,crl=WORK2)
          call chmdealloc('nmr.src','NMR1','WORK3',NBFRAM+1,crl=WORK3)
          call chmdealloc('nmr.src','NMR1','WORK4',NBFRAM+1,crl=WORK4)
       ENDIF

       !.......................................................................

    ELSEIF(WRD.EQ.'WRIT')THEN
       !            -------------

       WRD='    '
       WRD=NEXTA4(COMLYN,COMLEN)
       CALL ZCONSTR(.FALSE.,1,NZMAT,IZMAT,ZMAT,X,Y,Z)

       IF(WRD.EQ.'LIST')THEN
          !        -------------
          CALL WRNMR (NZMAT,ZMAT,IZMAT, &
               NCSA,ICSA,SIGMA,ZCSA,NDQS,IDQS, &
               NRTIMS,IRTIMS,NTS,ITS,IWRIT)

       ELSEIF(WRD.EQ.'COOR')THEN
          !            -------------
          CALL SEL123(COMLYN,COMLEN,1,ISLCT1,ISLCT2,ISLCT3, &
               NATOMT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID,X,Y,Z,WMAIN)
          IOMODE=2
          CALL CWRITE(IWRIT,TITLEA,NTITLA,ICNTRL,X,Y,Z,WMAIN, &
               RES,ATYPE,IBASE,NREST,NATOMT,ISLCT1,IOMODE,0,0,.FALSE.)

       ELSEIF(WRD.EQ.'CSA ')THEN
          !            -------------
          IF(NCSA.GT.0)THEN
             call chmalloc('nmr.src','NMR1','CSA',NCSA+1,crl=CSA)
             call chmalloc('nmr.src','NMR1','CSAPAR',NCSA+1,crl=CSAPAR)
             call chmalloc('nmr.src','NMR1','CSAPER',NCSA+1,crl=CSAPER)
             QPROP  = .TRUE.
             QUVECT = INDXA(COMLYN, COMLEN, 'UVEC') .GT. 0
             call CSAF(X,Y,Z,NCSA,ICSA,SIGMA,ZCSA,CSA,CSAPAR,CSAPER, &
                  PRNLEV,IWRIT,QPROP,QUVECT)
             call chmdealloc('nmr.src','NMR1','CSA',NCSA+1,crl=CSA)
             call chmdealloc('nmr.src','NMR1','CSAPAR',NCSA+1,crl=CSAPAR)
             call chmdealloc('nmr.src','NMR1','CSAPER',NCSA+1,crl=CSAPER)
          ENDIF

       ELSEIF(WRD.EQ.'DQS ')THEN
          !            -------------
          IF(NDQS.GT.0)THEN
             call chmalloc('nmr.src','NMR1','DQS',NDQS+1,crl=DQS)
             QPROP  = .TRUE.
             QUVECT = INDXA(COMLYN, COMLEN, 'UVEC') .GT. 0
             call DQSF(X,Y,Z,NDQS,IDQS,DQS,PRNLEV,IWRIT,QPROP,QUVECT)
             call chmdealloc('nmr.src','NMR1','DQS',NDQS+1,crl=DQS)
          ENDIF

       ELSEIF(WRD.EQ.'RTIM')THEN
          !            -------------
          RTUMBL=GTRMF(COMLYN,COMLEN,'RTUM',RTUMBL)
          CTRTIM=GTRMF(COMLYN,COMLEN,'CUT ',CTRTIM)
          HFIELD=GTRMF(COMLYN,COMLEN,'HFIE',HFIELD)
          IF(NRTIMS(1).GT.0)THEN
             CALL RARTIMS(X,Y,Z,NRTIMS,IRTIMS,GYRMAG,DSIGMA,CSAR, &
                  RTUMBL,HFIELD,CTRTIM,IWRIT,PRNLEV)
          ENDIF

       ENDIF

       !.......................................................................
    ELSEIF(WRD.EQ.'RESE')THEN
       !            -------------

       IF(PRNLEV.GT.3) WRITE(OUTU,100) 'NMR RESET TO ZERO'
       NCSA=0
       NDQS=0
       NRTIMS(1)=0
       NRTIMS(2)=0
       NTS=0
       IF(.NOT.(INDXA(COMLYN, COMLEN, 'KEEP') .GT. 0))THEN
          NZMAT=0
          NATOMT=NATOM
          NREST=NRES
          NSEGT=NSEG
       ELSE
          IF(PRNLEV.GT.3) WRITE(OUTU,100) 'Zmatrix build list is kept'
       ENDIF

       !.......................................................................
    ELSEIF(WRD.EQ.'END ')THEN
       !            -------------
       NCSA=0
       NDQS=0
       NRTIMS(1)=0
       NRTIMS(2)=0
       NTS=0
       NZMAT=0
       NATOMT=NATOM
       NREST=NRES
       NSEGT=NSEG
       DONE=.TRUE.

       !.......................................................................
    ELSE
       !
       CALL WRNDIE(-1,'<NMR>','NOT A COMMAND:  '//WRD)
       DONE=.TRUE.

    ENDIF
    !.......................................................................

    IF(DONE) RETURN
    GOTO 1000

  END SUBROUTINE NMR1

  SUBROUTINE SEL1AT(COMLYN,COMLEN,IATOM,ISLCT1, &
       NATOMT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID,X,Y,Z,WMAIN)
    !-----------------------------------------------------------------------
    !     This subroutine is used to do one single atom selection
    use chm_kinds
    use select
    implicit none
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    INTEGER IATOM,ISLCT1(*)
    INTEGER NATOMT,IBASE(*),NICTOT(*),NSEGT
    CHARACTER(len=*) RESID(*),RES(*),SEGID(*)
    real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
    !
    ! Local variables
    INTEGER I, IMODE, ICOUNT

    IMODE=0 !IMPLIES DEFAULT = ALL ATOMS SELECTED
    CALL SELRPN(COMLYN,COMLEN,ISLCT1,NATOMT,1,IMODE, &
         .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEGT, &
         .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
    IF(IMODE.NE.0)THEN
       CALL WRNDIE(0,'<SEL1AT>','ATOM SELECTION PARSING ERROR')
    ENDIF

    ICOUNT=0
    DO I=1,NATOMT
       IF(ISLCT1(I).EQ.1)THEN
          IATOM=I
          ICOUNT=ICOUNT+1
       ENDIF
    ENDDO

    IF(ICOUNT.EQ.0)THEN
       CALL WRNDIE(-1,'<SEL1AT>','NO ATOM SELECTED')
    ELSEIF(ICOUNT.GT.1)THEN
       CALL WRNDIE(-1,'<SEL1AT>','MUTIPLE ATOMS SELECTED')
    ENDIF

    RETURN
  END SUBROUTINE SEL1AT
  !
  SUBROUTINE SEL123(COMLYN,COMLEN,NSLCT,ISLCT1,ISLCT2,ISLCT3, &
       NATOMT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID,X,Y,Z,WMAIN)
    !-----------------------------------------------------------------------
    !     This subroutine is used to do one or two or three atoms selections
    use chm_kinds
    use select
    implicit none
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    INTEGER NSLCT,ISLCT1(*),ISLCT2(*),ISLCT3(*)
    INTEGER NATOMT,IBASE(*),NICTOT(*),NSEGT
    CHARACTER(len=*) RESID(*),RES(*),SEGID(*)
    real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
    !
    ! Local variables
    INTEGER IMODE

    IF(NSLCT.GE.1)THEN
       IMODE=0 !IMPLIES DEFAULT = ALL ATOMS SELECTED
       CALL SELRPN(COMLYN,COMLEN,ISLCT1,NATOMT,1,IMODE, &
            .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEGT, &
            .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0)THEN
          CALL WRNDIE(0,'<NMR>','ATOM SELECTION PARSING ERROR')
       ENDIF
    ENDIF

    IF(NSLCT.GE.2)THEN
       IMODE=0
       CALL SELRPN(COMLYN,COMLEN,ISLCT2,NATOMT,1,IMODE, &
            .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEGT, &
            .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0)THEN
          CALL WRNDIE(0,'<NMR>','ATOM SELECTION PARSING ERROR')
       ENDIF
    ENDIF

    IF(NSLCT.GE.3)THEN
       IMODE=0
       CALL SELRPN(COMLYN,COMLEN,ISLCT3,NATOMT,1,IMODE, &
            .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEGT, &
            .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0)THEN
          CALL WRNDIE(0,'<NMR>','ATOM SELECTION PARSING ERROR')
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE SEL123
#if KEY_UNUSED==1 /*fndatm_unused*/

  SUBROUTINE FNDATM(NATOMT,I,J,ISLCT,X,Y,Z,CTDI2)
    !-----------------------------------------------------------------------
    use chm_kinds
    implicit none
    INTEGER NATOMT,ISLCT(*),I,J
    real(chm_real) X(*),Y(*),Z(*),CTDI2,DIST2
    !
    !     write(6,*) 'FNDATM'
    DO J=1,NATOMT
       IF(ISLCT(J).NE.0)THEN
          DIST2=(X(I)-X(J))**2+(Y(I)-Y(J))**2+(Z(I)-Z(J))**2
          IF(DIST2.LT.CTDI2)THEN
             !     write(6,*) 'select',i,j,'dist=',sqrt(dist2)
             ISLCT(J)=0
             RETURN
          ENDIF
       ENDIF
    ENDDO
    !
    J=0
    RETURN
  END SUBROUTINE FNDATM
#endif /* (fndatm_unused)*/
  SUBROUTINE SETITS(NATOM,ISLCT,HFIELD,RTUMBL,GYRMAG,CSAR,DSIGMA2, &
       DSIGMA,LIST,ILIST,NLIST,ITS,NTS,OUTU,PRNLEV)
    !-----------------------------------------------------------------------
    !     Sets up the Index for the NMR Time-Series (ITS)
    !     The gyromagnetic factors are stored in GYRMAG.
    !     The chemical shift anistropy relaxation, the tensor is stored in the
    !     array DSIGMA and the pointer in LIST is negative.
    !
    use chm_kinds
    use consta
    use exfunc
    use number
    use chutil,only:atomid
    use string

    implicit none
    INTEGER NATOM,ISLCT(*)
    real(chm_real) HFIELD,RTUMBL,GYRMAG(*)
    LOGICAL CSAR
    real(chm_real) DSIGMA2,DSIGMA(*)
    INTEGER LIST(2,*),ILIST,NLIST,ITS(*),NTS
    INTEGER OUTU,PRNLEV
    ! Local variables
    INTEGER I,J,IATOM,JATOM
    CHARACTER(len=8) SGID,RID,REN,AT
    real(chm_real) OMEGA
    real(chm_real), PARAMETER :: PSEC=1.0D-12
    integer tklen

    tklen=4
    if (qxform()) tklen=8

    DO IATOM=1,NATOM
       IF(ISLCT(IATOM).EQ.1)THEN
          CALL ATOMID(IATOM,SGID,RID,REN,AT)
          IF(CSAR)THEN
             GYRMAG(IATOM)=ZERO
             DSIGMA(IATOM)=DSIGMA2 !contains the CSA tensor in ppm
             IF(PRNLEV.GE.2)THEN
                WRITE(OUTU,100) AT(1:tklen),REN(1:tklen), &
                     SGID(1:tklen),RID(1:tklen), &
                     'DSIGMA [ppm]=',DSIGMA(IATOM)
             ENDIF
          ELSE
             GYRMAG(IATOM)=GGAMMA(AT)
             IF(PRNLEV.GE.2)THEN
                OMEGA=GYRMAG(IATOM)*HFIELD*PSEC
                WRITE(OUTU,100) AT(1:tklen),REN(1:tklen), &
                     SGID(1:tklen),RID(1:tklen), &
                     'GAMMA=',GYRMAG(IATOM), &
                     'OMEGA [MHz]=',1.0D6*OMEGA/(TWO*PI)
100             FORMAT(6X,4(A,1X),2X,A,D12.4,2(1X,A,F8.3))
             ENDIF
          ENDIF
          IF(CSAR)THEN
             JATOM=-IATOM
          ELSE
             JATOM=IATOM
          ENDIF
          ! Search to see if this atom is already in the list to avoid duplications
          DO J=1,NLIST
             IF(LIST(ILIST,J).EQ.JATOM)THEN
                GOTO 10
             ENDIF
          ENDDO
          NLIST=NLIST+1
          LIST(ILIST,NLIST)=JATOM
          CALL GETITS(IATOM,I,ITS,NTS)
       ENDIF
10     CONTINUE
    ENDDO

    RETURN
  END SUBROUTINE SETITS
  !
  SUBROUTINE GETITS(IATOM,I,ITS,NTS)
    !-----------------------------------------------------------------------
    !     Sets up the Index for the NMR Time-Series (ITS)
    !     NTS = number of time series to store
    !     ITS = pointer for atoms in time series
    use chm_kinds
    implicit none
    INTEGER IATOM,I,ITS(*),NTS

    DO I=1,NTS
       IF(IATOM.EQ.ITS(I)) RETURN
    ENDDO
    NTS=NTS+1
    ITS(NTS)=IATOM
    I=NTS
    RETURN
  END SUBROUTINE GETITS

  FUNCTION GGAMMA(ATYPE) result(ggamma_rslt)
    !-----------------------------------------------------------------------
    !     Returns the gyromagnetic ratio of a nucleus (identified from
    !     the ATYPE array).  Constants taken from table 2.1 in
    !     "NMR of Proteins and Nucleic Acid" by K. Wuthrich.
    !     Units are in SI [RADIAN/(TESLA*SEC)]
    !
    use chm_kinds
    use number
    use stream
    implicit none
    real(chm_real) :: ggamma_rslt
    CHARACTER(len=*) ATYPE
    !
    IF(ATYPE(1:1).EQ.'H')THEN
       GGAMMA_RSLT=26.75D07
    ELSEIF(ATYPE(1:1).EQ.'C')THEN
       GGAMMA_RSLT=6.73D07
    ELSEIF(ATYPE(1:1).EQ.'N')THEN
       GGAMMA_RSLT=-2.71D07
    ELSEIF(ATYPE(1:1).EQ.'P')THEN
       GGAMMA_RSLT=10.83D07
    ELSE
       IF(PRNLEV.GT.3) WRITE(OUTU,'(A)') &
            'WARNING, UNKNOWN GYROMAGNETIC RATIO FOR ',ATYPE
       GGAMMA_RSLT=ZERO
    ENDIF

    RETURN
  END FUNCTION GGAMMA
  !
  SUBROUTINE DQSF(X,Y,Z,NDQS,IDQS,DQS,PRNLEV,IWRIT,QPROP,QUVECT)
    !-----------------------------------------------------------------------
    !  Deuterium Quadrupol Splitting Function, dqs = (3*Z**2-1)/2.0
    !  the NDQS particles are listed in the vectors idqs(1,*) and idqs(2,*)
    use chm_kinds
    use vector
    use number
    use chutil,only:atomid
    use string

    implicit none
    real(chm_real) X(*),Y(*),Z(*)
    INTEGER NDQS,IDQS(2,*)
    real(chm_real) DQS(*)
    INTEGER PRNLEV,IWRIT
    LOGICAL QPROP,QUVECT

    !  local variables
    real(chm_real) DQS1,DQS2
    INTEGER ICOUNT
    real(chm_real) TEMPO
    real(chm_real) UV(3)
    CHARACTER(len=8) SGID,RID,REN,AC1,AC2
    INTEGER I,IC,ID
    integer tklen

    tklen=4
    if (qxform()) tklen=8

    ICOUNT=0
    IF(QPROP) THEN
       IF(PRNLEV.GT.3) WRITE(IWRIT,*)
    ENDIF

    DQS1=ZERO
    DQS2=ZERO

    DO I=1,NDQS
       ID=IDQS(1,I)
       IC=IDQS(2,I)
       !
       !  Principal axis
       UV(1)=X(ID)-X(IC)
       UV(2)=Y(ID)-Y(IC)
       UV(3)=Z(ID)-Z(IC)
       CALL NORMALL(UV,3)


       DQS(I)=(THREE*UV(3)**2-ONE)*HALF
       DQS1=DQS1+DQS(I)
       DQS2=DQS2+DQS(I)**2

       IF(QPROP) THEN
          CALL ATOMID(IDQS(1,I),SGID,RID,REN,AC1)
          CALL ATOMID(IDQS(2,I),SGID,RID,REN,AC2)
          IF(PRNLEV.GT.3) WRITE(IWRIT,100) '    DQS> ', &
               I,DQS(I),AC1(1:tklen),AC2(1:tklen), &
               REN(1:tklen),RID(1:tklen),SGID(1:tklen)
100       FORMAT(A,I3,5X,F12.5,5X,5(1X,A))
       ENDIF

       IF(QUVECT)THEN
          ICOUNT=ICOUNT+1
          IF(PRNLEV.GT.3) THEN
             WRITE(IWRIT,100) '            ASSOCIATED UNIT VECTOR:'
             WRITE(IWRIT,101) ICOUNT,I,'DQS ','UVECT',UV(1),UV(2),UV(3)
101          FORMAT(2I5,2(1X,A),3F10.5,2(1X,A),2F10.5)
          ENDIF
       ENDIF

    ENDDO

    ! Write out global averages
    IF((NDQS.GT.1).AND.QPROP)THEN
       DQS1=DQS1/NDQS
       DQS2=DQS2/NDQS
       DQS2=SQRT(DQS2-DQS1**2)
       IF(PRNLEV.GT.3) THEN
          WRITE(IWRIT,100)
          WRITE(IWRIT,100) ' DQS AV> ',NDQS,DQS1
          WRITE(IWRIT,100) ' DQS FL> ',NDQS,DQS2
       ENDIF
    ENDIF

    RETURN
  END SUBROUTINE DQSF
  !
  SUBROUTINE CSAF(X,Y,Z,NCSA,ICSA,SIGMA,ZCSA,CSA,CSAPAR,CSAPER, &
       PRNLEV,IWRIT,QPROP,QUVECT)
    !------------------------------------------------------------------------
    !  Chemical Shift Anisotropy
    !  Construct the principal axis from a zmatrix
    !      1         u
    !       \       /          theta 2-3-u   (theta=0 gives u along 2-3)
    !        2 --- 3           phi   1-2-3-u (phi=0 gives a cis)
    !  "u" is the end of the unit vector indicating a principal axis
    !
    !  the 3-particles of j are listed in the vector icsa(i,jcsa)
    !  the i-th principle axis are in sigma()
    !  the i-th vector zmatrix theta are stored in zcsa(i,1,jcsa)
    !                          phi   are stored in zcsa(i,2,jcsa)
    !  the instantaneous chemical shift anisotropy value is in CSA

    use chm_kinds
    use consta
    use vector
    use number
    use chutil,only:atomid
    use string

    implicit none
    real(chm_real)        X(*),Y(*),Z(*)
    INTEGER       NCSA,ICSA(3,*)
    real(chm_real)        SIGMA(3,*),ZCSA(2,2,*),CSA(*),CSAPAR(*), &
         CSAPER(*)
    INTEGER       PRNLEV,IWRIT
    LOGICAL       QPROP,QUVECT

    !  local variables
    integer       tklen
    real(chm_real)        CSA1,CSAPA1,CSAPE1
    real(chm_real)        CSA2,CSAPA2,CSAPE2
    CHARACTER(len=8)   SGID,RID,REN,AC
    real(chm_real)        UV(3,3)
    real(chm_real)        V12(3),V32(3),V34(3),VPL(3),THETA,BOND,PHI
    INTEGER       I,IZ1,IZ2,IZ3,ICOUNT,JCSA
    CHARACTER(len=4) :: UVECT(3)=(/'UV1 ','UV2 ','UV3 '/)

    tklen=4
    if (qxform()) tklen=8
    ICOUNT=0

    IF(QPROP) THEN
       IF(PRNLEV.GT.3) WRITE(IWRIT,'(18X,3(A,6X))') &
            'CSA   ', 'CSAPAR', 'CSAPER'
    ENDIF

    CSA1   = ZERO
    CSAPA1 = ZERO
    CSAPE1 = ZERO

    CSA2   = ZERO
    CSAPA2 = ZERO
    CSAPE2 = ZERO

    DO JCSA=1,NCSA
       IZ1=ICSA(1,JCSA)
       IZ2=ICSA(2,JCSA)
       IZ3=ICSA(3,JCSA)   !THIS ONE IS THE NUCLEI OBSERVED IN NMR

       DO I=1,2
          THETA=ZCSA(I,1,JCSA)*DEGRAD
          PHI=  ZCSA(I,2,JCSA)*DEGRAD

          !     make the unit vector R2-R3
          V32(1)=X(IZ2)-X(IZ3)
          V32(2)=Y(IZ2)-Y(IZ3)
          V32(3)=Z(IZ2)-Z(IZ3)
          CALL NORMALL(V32,3)
          V34(1)=V32(1)
          V34(2)=V32(2)
          V34(3)=V32(3)

          !     make the unit vector R2-R1
          V12(1)=X(IZ2)-X(IZ1)
          V12(2)=Y(IZ2)-Y(IZ1)
          V12(3)=Z(IZ2)-Z(IZ1)
          CALL NORMALL(V12,3)

          !     Make the vector normal to the plane Vpl = (V12 x V32)
          !     check if co-linear with the 2-3 bond
          IF(ABS(ZCSA(I,1,JCSA)-180.0).GE.RSMALL)THEN
             CALL CROSS3(V12,V32,VPL)
             CALL NORMALL(VPL,3)
             CALL ROTATX(V34,VPL,V34,-THETA)
             CALL ROTATX(V34,V32,V34,PHI)
          ELSE
             V34(1)=-V34(1)
             V34(2)=-V34(2)
             V34(3)=-V34(3)
          ENDIF

          UV(1,I)=V34(1)
          UV(2,I)=V34(2)
          UV(3,I)=V34(3)

       ENDDO

       ! Make the third axis from   1 x 2 = 3
       CALL CROSS3(UV(1,1),UV(1,2),UV(1,3))

       DO I=1,3
          CALL NORMALL(UV(1,I),3)
       ENDDO

       CSA(JCSA)=ZERO
       CSAPAR(JCSA)=ZERO
       CSAPER(JCSA)=ZERO
       DO I=1,3
          CSAPAR(JCSA)=CSAPAR(JCSA)+SIGMA(I,JCSA)*UV(3,I)**2
          CSAPER(JCSA)=CSAPER(JCSA)+ &
               SIGMA(I,JCSA)*HALF*(UV(1,I)**2+UV(2,I)**2)
          CSA(JCSA)=CSA(JCSA)+ &
               SIGMA(I,JCSA)*(UV(3,I)**2-HALF*(UV(1,I)**2+UV(2,I)**2))
       ENDDO

       !  Get global averages
       CSA1   = CSA1+CSA(JCSA)
       CSAPA1 = CSAPA1+CSAPAR(JCSA)
       CSAPE1 = CSAPE1+CSAPER(JCSA)

       CSA2   = CSA2+CSA(JCSA)**2
       CSAPA2 = CSAPA2+CSAPAR(JCSA)**2
       CSAPE2 = CSAPE2+CSAPER(JCSA)**2

       IF(QPROP)THEN
          CALL ATOMID(IZ3,SGID,RID,REN,AC)
          IF(PRNLEV.GT.3) WRITE(IWRIT,100) &
               '    CSA> ',JCSA,CSA(JCSA),CSAPAR(JCSA),CSAPER(JCSA), &
               AC(1:tklen),REN(1:tklen),RID(1:tklen),SGID(1:tklen)
100       FORMAT(A,I3,5X,3F12.5,5X,5(1X,A))
       ENDIF

       IF(QUVECT)THEN
          IF(PRNLEV.GT.3) WRITE(IWRIT,100) &
               '            ASSOCIATED UNIT VECTORS:'
          DO I=1,3
             ICOUNT=ICOUNT+1
             IF(PRNLEV.GT.3) WRITE(IWRIT,101) ICOUNT,JCSA,'CSA ',UVECT(I), &
                  UV(1,I),UV(2,I),UV(3,I),'NMR','1',SIGMA(I,JCSA)
101          FORMAT(2I5,2(1X,A),3F10.5,2(1X,A),2F10.5)
          ENDDO
       ENDIF

    ENDDO

    ! Write out the global averages
    IF((NCSA.GT.1).AND.QPROP)THEN
       CSA1   = CSA1/NCSA
       CSAPA1 = CSAPA1/NCSA
       CSAPE1 = CSAPE1/NCSA
       CSA2   = CSA2/NCSA
       CSAPA2 = CSAPA2/NCSA
       CSAPE2 = CSAPE2/NCSA
       CSA2   = SQRT(CSA2-CSA1**2)
       CSAPA2 = SQRT(CSAPA2-CSAPA1**2)
       CSAPE2 = SQRT(CSAPE2-CSAPE1**2)
       IF(PRNLEV.GT.3) THEN
          WRITE(IWRIT,100)
          WRITE(IWRIT,100) ' CSA AV> ',NCSA,CSA1,CSAPA1,CSAPE1
          WRITE(IWRIT,100) ' CSA FL> ',NCSA,CSA2,CSAPA2,CSAPE2
       ENDIF
    ENDIF

    RETURN
  END SUBROUTINE CSAF
  !
  SUBROUTINE ADDNMR(NCOORD,NPROP,PROP,PROP1,PROP2)
    !-----------------------------------------------------------------------
    use chm_kinds
    use number
    implicit none
    INTEGER NCOORD,NPROP
    real(chm_real) PROP(*),PROP1(*),PROP2(*)
    INTEGER I
    ! Initialize counters and zero all variables
    IF(NCOORD.EQ.0)THEN
       DO I=1,NPROP
          PROP1(I)=ZERO
          PROP2(I)=ZERO
       ENDDO
    ELSE
       DO I=1,NPROP
          PROP1(I)=PROP1(I)+PROP(I)
          PROP2(I)=PROP2(I)+PROP(I)**2
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE ADDNMR
  !
  SUBROUTINE AVENMR(NCOORD,NPROP,PROP1,PROP2,PROPT1,PROPT2)
    !-----------------------------------------------------------------------
    use chm_kinds
    use number
    implicit none
    INTEGER NCOORD,NPROP
    real(chm_real) PROP1(*),PROP2(*),PROPT1,PROPT2
    INTEGER I
    ! Initialize counters and zero all variables
    IF(NCOORD.EQ.0)THEN
       DO I=1,NPROP
          PROP1(I)=ZERO
          PROP2(I)=ZERO
       ENDDO
    ELSE
       PROPT1 = ZERO
       PROPT2 = ZERO
       DO I=1,NPROP
          PROP1(I)=PROP1(I)/NCOORD
          PROP2(I)=PROP2(I)/NCOORD
          PROPT1 = PROPT1 + PROP1(I)
          PROPT2 = PROPT2 + PROP2(I)
          PROP2(I)=SQRT(PROP2(I)-PROP1(I)**2)
       ENDDO
    ENDIF

    PROPT1 = PROPT1/NPROP
    PROPT2 = PROPT2/NPROP
    PROPT2 = SQRT(PROPT2 - PROPT1**2)

    RETURN
  END SUBROUTINE AVENMR
  !
  SUBROUTINE NMRDYN(NATOM,NRES,NSEG,NATOMT,NREST,NSEGT, &
       IBASE,NICTOT,ATYPE,RESID,RES,SEGID, &
       X,Y,Z,WMAIN,XCOMP,YCOMP,ZCOMP,WCOMP, &
       ORIENT,LMASS,LWMAIN,AMASS,BMASS,ATOMPR,ISLCT1, &
       NZMAT,ZMAT,IZMAT,ISLCT2, &
       NCSA,ICSA,SIGMA,ZCSA, &
       CSA,CSA1,CSA2, &
       CSAPAR,CSAPA1,CSAPA2, &
       CSAPER,CSAPE1,CSAPE2, &
       NDQS,DQS,DQS1,DQS2,IDQS, &
       NRTIMS,IRTIMS,CTRTIM, &
       NTS,ITS,NBFRAM,XTSERI,YTSERI,ZTSERI, &
       GYRMAG,DSIGMA,RTUMBL,HFIELD,TMAX, &
       CORR12,WORK1,WORK2,WORK3,WORK4, &
       NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP,R4TEMP,ICNTRL,ISLCT3, &
       QCORRE,QTSERI,QPROP,QUVECT,QVERB, &
       PRNLEV,IWRIT,IOLEV,ILIST,ISDLIST,IMF,IMFDAT,DSIGMA3, &
       RELAX,NAVE,QOF,QSAVE,QPRSTAT)
    !-----------------------------------------------------------------------
    use chm_kinds
    use consta
    use exfunc
    use number
    use ctitla
    use corsubs,only:pkmass,rotls1
    use chutil,only:atomid
    use coorio_mod,only:cwrite
    use cvio
    use string

    implicit none

    ! General PSF and COOR information
    INTEGER NATOM,NRES,NSEG,NATOMT,NREST,NSEGT
    INTEGER IBASE(*),NICTOT(*)
    CHARACTER(len=*) ATYPE(*),RESID(*),RES(*),SEGID(*)
    real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
    real(chm_real) XCOMP(*),YCOMP(*),ZCOMP(*),WCOMP(*)

    ! Re-orient the coordinate option
    LOGICAL ORIENT,LMASS,LWMAIN
    real(chm_real) AMASS(*),BMASS(*)
    INTEGER NPR,ATOMPR(2,*),ISLCT1(*)

    ! Construction of atoms and printing coordinates
    real(chm_real) ZMAT(3,*)
    INTEGER NZMAT,IZMAT(4,*)
    INTEGER ISLCT2(*)
    !
    ! Chemical Shift Anisotropy Tensor
    INTEGER NCSA,ICSA(3,*)
    real(chm_real) SIGMA(3,*),ZCSA(2,2,*)
    real(chm_real) CSA(*),CSA1(*),CSA2(*)
    real(chm_real) CSAPAR(*),CSAPA1(*),CSAPA2(*)
    real(chm_real) CSAPER(*),CSAPE1(*),CSAPE2(*)
    !
    ! Deuterium Quadrupol Splitting
    INTEGER NDQS,IDQS(2,*)
    real(chm_real) DQS(*),DQS1(*),DQS2(*)
    !
    ! Time series storage in real(chm_real4)
    INTEGER NTS,ITS(*),NBFRAM
    REAL(CHM_REAL4) XTSERI(NBFRAM,NTS)
    REAL(CHM_REAL4) YTSERI(NBFRAM,NTS)
    REAL(CHM_REAL4) ZTSERI(NBFRAM,NTS)
    real(chm_real) GYRMAG(*),DSIGMA(*),RTUMBL,HFIELD,TMAX
    real(chm_real) CORR12(*),WORK1(*),WORK2(*),WORK3(*),WORK4(*)

    ! Transverse and Longitudinal Relaxation Times  T1 and T2 and
    ! Cross-Relaxation Rates NOE and ROE
    INTEGER NRTIMS(2),IRTIMS(2,*)
    real(chm_real)  CTRTIM
    real(chm_real)  RELAX(9,2,*)
    INTEGER NAVE
    LOGICAL QOF,QSAVE, QPRSTAT

    ! Print out specifications
    LOGICAL QCORRE,QTSERI,QPROP,QUVECT,QVERB
    INTEGER PRNLEV,IWRIT,ILIST,ISDLIST,IMF,IMFDAT,IOLEV
    real(chm_real) DSIGMA3,DR1,DR2,DNOE
    !
    ! Local variables
    integer tklen
    INTEGER NMAX
    real(chm_real) PLAT12
    INTEGER IOMODE,NCOORD
    LOGICAL DONE,LPRINT,LNOROT,CSAR
    real(chm_real) DIST
    CHARACTER(len=8) SGID1,RID1,REN1,AT1
    CHARACTER(len=8) SGID2,RID2,REN2,AT2
    real(chm_real) SD12(5)
    real(chm_real) RT1,RT1TOT
    real(chm_real) RT2,RT2TOT,OMEGA1
    real(chm_real), PARAMETER :: PSEC=1.0D-12,PPM=1.0D-6
    real(chm_real) RNOE,RROE,RR6
    INTEGER I,IATOM,I1,I2,IAT1,IAT2,JCSA,JDQS,ITS1,ITS2
    real(chm_real) CSA1T, CSA2T
    real(chm_real) CSPA1T, CSPA2T, CSPE1T, CSPE2T
    real(chm_real) DQS1T, DQS2T, TAUE,TMAX1, SCONF
    !
    ! General stuff needed for READCV
    INTEGER NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP
    REAL(CHM_REAL4) R4TEMP(*)
    INTEGER ICNTRL(20),ISLCT3(*)
    INTEGER NFREAT,IUNIT,IFILE,ISTEP,ISTATS,NDEGF,NSAVV
    real(chm_real)  DELTA,DELTA2
    CHARACTER(len=4) :: HDR1='COOR',HDR2='CORD'

    tklen=4
    if (qxform()) tklen=8
    !
    IF(QPRSTAT.AND.PRNLEV.GT.3)THEN
       !
       ! write relaxation statistics and return
       WRITE(IWRIT,'(/3X,A,I8,A)') &
            'Relaxation averages computed using',NAVE,' windows.'
       IF(NAVE.LE.1)  &
            CALL WRNDIE(-2,'<NMRDYN>','Too few data sets for averaging')

       IF(ILIST.GT.0)   WRITE(ILIST,200)   'RESI AVG:'
       IF(ISDLIST.GT.0) WRITE(ISDLIST,200) 'RESI SD: '
       DO I1=1,NRTIMS(1)
          ! Skip all-zero entries (likely to be PRO or N-term residue, lacking N-H)
          IF(RELAX(9,1,I1).GT.0.0)THEN
             IAT1=IRTIMS(1,I1)
             IAT2=IRTIMS(2,1)
             CALL ATOMID(IAT1,SGID1,RID1,REN1,AT1)
             CALL ATOMID(IAT2,SGID2,RID2,REN2,AT2)
             WRITE(ILIST,205) RID1(1:TKLEN), &
                  (RELAX(I,1,I1)/NAVE,I=1,9), &
                  REN1(1:TKLEN),AT1(1:TKLEN),AT2(1:TKLEN) 
             IF(ISDLIST.GT.0)  WRITE(ISDLIST,205) RID1(1:TKLEN), &
                  (SQRT( (NAVE*RELAX(I,2,I1)-(RELAX(I,1,I1)**2)) / &
                  (NAVE*(NAVE-1)) ),I=1,9), &
                  REN1(1:TKLEN),AT1(1:TKLEN),AT2(1:TKLEN) 
          ENDIF
          DO I=1,9
             RELAX(I,1,I1)=0.0
             RELAX(I,2,I1)=0.0
          ENDDO
       ENDDO
       NAVE=0
       RETURN
    ENDIF
    IF(.NOT. QSAVE) NAVE=0
    IF(QSAVE.AND.NAVE.EQ.0)THEN
       ! zero accumulators, just to be on safe side
       DO I1=1,NRTIMS(1)
          DO I=1,9
             RELAX(I,1,I1)=0.0
             RELAX(I,2,I1)=0.0
          ENDDO
       ENDDO
    ENDIF
    !
    IF(QVERB)THEN
       QTSERI=.TRUE.
       QCORRE=.TRUE.
       QUVECT=.TRUE.
       QPROP =.TRUE.
    ENDIF
    !
    !     Setup the orient option assuming that the reference structure is
    !     in the COMP set
    !
    IF(ORIENT)THEN
       LPRINT=QVERB.AND.(PRNLEV.GT.3)
       LNOROT=.FALSE.
       NPR=0
       DO I=1,NATOM
          IF(ISLCT1(I).EQ.1)THEN
             NPR=NPR+1
             ATOMPR(1,NPR)=I
             ATOMPR(2,NPR)=I
          ENDIF
       ENDDO
       CALL PKMASS(AMASS,AMASS,BMASS,ATOMPR,NPR,LMASS,LWMAIN,WMAIN)
    ENDIF

    ! Initialize average properties
    NCOORD=0
    CSA1T = ZERO
    CSA2T = ZERO
    !yw...Initialization of CSAPA/CSAPE added 02-Aug-95
    IF(NCSA.GT.0) THEN
       CALL ADDNMR(NCOORD,NCSA,CSA,CSA1,CSA2)
       CALL ADDNMR(NCOORD,NCSA,CSAPAR,CSAPA1,CSAPA2)
       CALL ADDNMR(NCOORD,NCSA,CSAPER,CSAPE1,CSAPE2)
    ENDIF
    !yw...
    IF(NDQS.GT.0) CALL ADDNMR(NCOORD,NDQS,DQS,DQS1,DQS2)
    !
    ! Main loop for reading trajectory file
    IUNIT=FIRSTU
    NFREAT=NATOM
    ISTATS=1
    DONE=.FALSE.

    !.......................................................................

1000 CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
         (/ ZERO /), .FALSE., &  
#endif
         R4TEMP,NATOM,ISLCT3,NFREAT, &
         FIRSTU,NUNIT,IUNIT,IFILE, &
         ISTEP,ISTATS,NDEGF,DELTA2, &
         NBEGIN,NSTOP,NSKIP,NSAVV,HDR1,HDR2, &
         TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
    DELTA=TIMFAC*DELTA2*NSKIP
    NCOORD=NCOORD+1
    DONE=(ISTATS.LT.0).OR.((NCOORD.EQ.NBFRAM).AND.NTS.GT.0)
    IF(QVERB.OR.QPROP)THEN
       IF(PRNLEV.GT.3) THEN
          WRITE(IWRIT,*)
          WRITE(IWRIT,'(1X,A,F8.3,1X,A,I8,1X,A,I8)') &
               'TIME = ',ISTEP*DELTA2*TIMFAC, &
               'NCOORD = ',NCOORD, &
               'STEP = ',ISTEP
       ENDIF
    ENDIF
    IF(NCOORD.GT.NBFRAM)THEN
       CALL WRNDIE(-4,'NMRDYN>  ','NCOORD LARGER THAN NBFRAM')
    ENDIF
    !
    IF(ORIENT)THEN
       IF(QOF .AND. NCOORD.EQ.1)THEN
          ! Use this coordinate set as reference
          DO I=1,NATOM
             XCOMP(I)=X(I)
             YCOMP(I)=Y(I)
             ZCOMP(I)=Z(I)
          ENDDO
       ENDIF
       ! Re-orient the main coordinate set with respect to selected atoms
       ! in the comp
       CALL ROTLS1(XCOMP,YCOMP,ZCOMP,X,Y,Z,NATOM, &
            ATOMPR,NPR,BMASS,LPRINT,LNOROT)
    ENDIF

    ! Construct the dummy atoms
    CALL ZCONSTR(.FALSE.,1,NZMAT,IZMAT,ZMAT,X,Y,Z)
    !
    ! Store TIME SERIES for selected atoms
    DO I=1,NTS
       IATOM=ITS(I)
       XTSERI(NCOORD,I)=X(IATOM)
       YTSERI(NCOORD,I)=Y(IATOM)
       ZTSERI(NCOORD,I)=Z(IATOM)
    ENDDO
    !
    ! Print out coordinates of selected atoms (only used for debugging)
    IF(PRNLEV.GT.10)THEN
       IOMODE=2
       CALL CWRITE(IWRIT,TITLEA,NTITLA,ICNTRL,X,Y,Z,WMAIN, &
            RES,ATYPE,IBASE,NREST,NATOMT,ISLCT2,IOMODE,0,0,.FALSE.)
    ENDIF
    !
    ! Average Hamiltonian properties
    IF(NCSA.GT.0)THEN
       CALL CSAF(X,Y,Z,NCSA,ICSA,SIGMA,ZCSA,CSA,CSAPAR,CSAPER, &
            PRNLEV,IWRIT,QPROP,QUVECT)
       CALL ADDNMR(NCOORD,NCSA,CSA,CSA1,CSA2)
       CALL ADDNMR(NCOORD,NCSA,CSAPAR,CSAPA1,CSAPA2)
       CALL ADDNMR(NCOORD,NCSA,CSAPER,CSAPE1,CSAPE2)
    ENDIF
    IF(NDQS.GT.0)THEN
       CALL DQSF(X,Y,Z,NDQS,IDQS,DQS,PRNLEV,IWRIT,QPROP,QUVECT)
       CALL ADDNMR(NCOORD,NDQS,DQS,DQS1,DQS2)
    ENDIF

    IF(.NOT.DONE) GOTO 1000
    IF(PRNLEV.GT.3) WRITE(IWRIT,*) &
         'A TOTAL OF ',NCOORD,' FRAMES WILL BE USED'
    !
    !.......................................................................
    !
    ! Relaxation times properties, compute correlation functions
    IF(NRTIMS(1).GT.0)THEN
       IF(PRNLEV.GT.3) THEN
          WRITE(IWRIT,*)
          WRITE(IWRIT,100) 'DIPOLE-DIPOLE RELAXATION RATES:'
100       FORMAT(6X,A)
       ENDIF
       NMAX=INT(TMAX/DELTA+1)
       IF(NMAX.GT.NCOORD)THEN
          CALL WRNDIE(-1,'NMRDYN>  ','TMAX LARGER THAN TRAJECTORY')
          NMAX=NCOORD-1
       ENDIF
       IF(PRNLEV.GT.3) THEN
          WRITE(IWRIT,103) 'HFIELD [TESLA] = ',HFIELD, &
               'RTUMBL [PSEC]  = ',RTUMBL
103       FORMAT(6X,2(A,4X,F12.3,1X))
          IF(ILIST.GT.0 .AND. .NOT. QPRSTAT) &
               WRITE(ILIST,200) 'RESI     '
200       FORMAT(A,' R1          R2         NOE         ROE', &
               '      R2/R1        <S2>        Sconf   ', &
               '    TAUE   TMXE', &
               '    RESN AT1 AT2')
       ENDIF
       !
       IF(QSAVE)THEN
          NAVE=NAVE+1
          IF(PRNLEV.GE.3) WRITE(IWRIT,'(/3X,A,I5)')   &
               'Accumulating relaxation statistics. Window #',NAVE
       ENDIF
       !
       DO I1=1,NRTIMS(1)
          IAT1=IRTIMS(1,I1)
          CALL GETITS(IAT1,ITS1,ITS,NTS)
          RT1TOT=ZERO
          RT2TOT=ZERO
          DO I2=1,NRTIMS(2)
             IAT2=IRTIMS(2,I2)
             CSAR=(IAT2.LE.0)
             IF(CSAR)THEN
                IAT2=-IAT2
             ENDIF
             CALL GETITS(IAT2,ITS2,ITS,NTS)
             DIST=SQRT((XTSERI(1,ITS1)-XTSERI(1,ITS2))**2+ &
                  (YTSERI(1,ITS1)-YTSERI(1,ITS2))**2+ &
                  (ZTSERI(1,ITS1)-ZTSERI(1,ITS2))**2)
             IF(CSAR.AND.(DIST.GT.ONE)) GOTO 51
             IF((DIST.LE.CTRTIM).AND.(IAT1.NE.IAT2))THEN
                CALL ATOMID(IAT1,SGID1,RID1,REN1,AT1)
                CALL ATOMID(IAT2,SGID2,RID2,REN2,AT2)
                IF(PRNLEV.GT.3)THEN
                   IF(CSAR)THEN
                      WRITE(IWRIT,101) 'CSAR AXIS:  ', &
                           AT1(1:tklen),REN1(1:tklen),RID1(1:tklen),SGID1(1:tklen), &
                           AT2(1:tklen),REN2(1:tklen),RID2(1:tklen),SGID2(1:tklen)
                   ELSE
                      WRITE(IWRIT,101) 'SPIN PAIR:  ', &
                           AT1(1:tklen),REN1(1:tklen),RID1(1:tklen),SGID1(1:tklen), &
                           AT2(1:tklen),REN2(1:tklen),RID2(1:tklen),SGID2(1:tklen)
                      ! 101 FORMAT(6X,A,2(4(A,1X),2X))
101                   FORMAT(/,6X,A,2(4(A,1X),'  -  '))
                   ENDIF
                ENDIF

                CALL NMRCRR(NBFRAM,NTS,ITS1,ITS2,XTSERI,YTSERI,ZTSERI,NCOORD, &
                     DELTA,NMAX,CORR12,PLAT12,WORK1,WORK2,WORK3,WORK4, &
                     DIST,RR6,IWRIT,QCORRE,QTSERI,QVERB,TAUE,TMAX1,SCONF)
                CALL NMRSD(GYRMAG(IAT1),GYRMAG(IAT2),HFIELD,CORR12,PLAT12, &
                     DELTA,NMAX,RTUMBL,SD12,WORK1,QVERB,IWRIT)
                CALL NMRRAT(GYRMAG(IAT1),GYRMAG(IAT2),CSAR,DSIGMA(IAT2),HFIELD, &
                     RR6,SD12,RT1,RT2,RNOE,RROE,IWRIT)

                ! Add possibility of CSA relaxation contribution  
                !      RT1A=RT1
                !      RT2A=RT2  
                IF(DSIGMA3.NE.ZERO)THEN
                   OMEGA1=HFIELD*GYRMAG(IAT1)
                   RT1=RT1 + &
                        (2*((PPM*DSIGMA3*OMEGA1)**2)/15)*SD12(2)*PSEC/RR6
                   RT2=RT2 + &
                        (2*((PPM*DSIGMA3*OMEGA1)**2)/15)*(2*SD12(1)/3+SD12(2)/2)*PSEC/RR6
                ENDIF
                IF(ILIST.GT.0 .AND. PRNLEV .GT. 3)THEN
                   WRITE(ILIST,205) RID1(1:TKLEN),RT1,RT2,RNOE,RROE, &
                        RT2/RT1,PLAT12/CORR12(1),SCONF,TAUE,TMAX1, &
                        REN1(1:TKLEN),AT1(1:TKLEN),AT2(1:TKLEN) 
205                FORMAT(A,7G12.4,F9.2,F8.1,1X,A,1X,A,1X,A)
                ENDIF
                IF(QSAVE) THEN
                   ! Accumulate statistics
                   ! NB! If there is more than one spin-pair on an I1 atom t
                   !     the summation will be over all pairs,
                   !     which may not be correct for all these properties. 
                   !     This is mainly intended for 
                   !     peptide backbone amide (N-H) groups, where that situation 
                   !     does not arise.
                   !  NO ERROR CHECKING!!
                   !  
                   RELAX(1,1,I1)=RELAX(1,1,I1) + RT1
                   RELAX(2,1,I1)=RELAX(2,1,I1) + RT2
                   RELAX(3,1,I1)=RELAX(3,1,I1) + RNOE
                   RELAX(4,1,I1)=RELAX(4,1,I1) + RROE
                   RELAX(5,1,I1)=RELAX(5,1,I1) + RT2/RT1
                   RELAX(6,1,I1)=RELAX(6,1,I1) + PLAT12/CORR12(1)
                   RELAX(7,1,I1)=RELAX(7,1,I1) + SCONF
                   RELAX(8,1,I1)=RELAX(8,1,I1) + TAUE
                   RELAX(9,1,I1)=RELAX(9,1,I1) + TMAX1

                   RELAX(1,2,I1)=RELAX(1,2,I1) + RT1**2
                   RELAX(2,2,I1)=RELAX(2,2,I1) + RT2**2
                   RELAX(3,2,I1)=RELAX(3,2,I1) + RNOE**2
                   RELAX(4,2,I1)=RELAX(4,2,I1) + RROE**2
                   RELAX(5,2,I1)=RELAX(5,2,I1) + (RT2/RT1)**2
                   RELAX(6,2,I1)=RELAX(6,2,I1) + (PLAT12/CORR12(1))**2
                   RELAX(7,2,I1)=RELAX(7,2,I1) + SCONF**2
                   RELAX(8,2,I1)=RELAX(8,2,I1) + TAUE**2
                   RELAX(9,2,I1)=RELAX(9,2,I1) + TMAX1**2
                ENDIF
                IF(IMF.GT.0)THEN
                   ! Output for Art Palmer's ModelFree analysis program,
                   !  MFPAR & MFDATA files
                   ! For now we assume N-H vectors are used
                   IF(PRNLEV.GT.3)THEN
                      WRITE(IMF,210) RID1,RID1,'N15',GYRMAG(IAT1)*1.0E-7,DIST, &
                           DSIGMA3,AT1(1:TKLEN),AT2(1:TKLEN)
210                   FORMAT('spin ',A, &
                           /'constants ',A,2X,A,3F12.3, &
                           /'vector ',A,2X,A,/)
                   ENDIF
                   !
                   ! Fake error estimates for now; 
                   ! may want to use real statistics if obtained above   
                   DR1=ABS(0.05*RT1)
                   DR2=ABS(0.05*RT2)
                   DNOE=ABS(0.05*RNOE)
                   IF(PRNLEV.GT.3)THEN  
                      WRITE(IMFDAT,215) RID1(1:TKLEN),HFIELD,RT1,DR1, &
                           HFIELD,RT2,DR2,HFIELD,RNOE,DNOE
215                   FORMAT('spin ',A, &
                           /'R1  ',3F12.3,' 1', &
                           /'R2  ',3F12.3,' 1', &
                           /'NOE ',3F12.3,' 1',/)
                   ENDIF
                ENDIF
                RT1TOT=RT1TOT+RT1
                RT2TOT=RT2TOT+RT2
             ENDIF
51           CONTINUE
          ENDDO
          IF(PRNLEV.GT.3)THEN
             IF((RT1TOT.NE.ZERO).OR.(RT2TOT.NE.ZERO))THEN
                WRITE(IWRIT,104) 'TOTAL [1/SEC]:  ', &
                     '1/T1 = ', RT1TOT,'1/T2 = ',RT2TOT, &
                     AT1(1:tklen),REN1(1:tklen),RID1(1:tklen),SGID1(1:tklen)
104             FORMAT(6X,A,2(A,F10.6,1X),3X,4(A,1X))
             ENDIF
          ENDIF
       ENDDO
    ENDIF
    !
    ! Complete Average Hamiltonian properties
    IF(NCSA.GT.0)THEN
       CALL AVENMR(NCOORD,NCSA,CSA1,CSA2,CSA1T,CSA2T)
       CALL AVENMR(NCOORD,NCSA,CSAPA1,CSAPA2,CSPA1T,CSPA2T)
       CALL AVENMR(NCOORD,NCSA,CSAPE1,CSAPE2,CSPE1T,CSPE2T)
       IF(PRNLEV.GT.3) THEN
          WRITE(IWRIT,*)
          WRITE(IWRIT,'(6x,3a)') '|------- CSA   -----', &
               '|------ CSAPAR -----','|------ CSAPER -----|'
       ENDIF
       DO JCSA=1,NCSA
          CALL ATOMID(ICSA(3,JCSA),SGID1,RID1,REN1,AT1)
          IF(PRNLEV.GT.3) WRITE(IWRIT,102) JCSA, CSA1(JCSA),CSA2(JCSA), &
               CSAPA1(JCSA),CSAPA2(JCSA), CSAPE1(JCSA),CSAPE2(JCSA), &
               AT1(1:tklen),REN1(1:tklen),RID1(1:tklen),SGID1(1:tklen)
102       FORMAT(1X,I3,2X,3(2F9.3,2x),2X,5(1X,A))
       ENDDO
       IF(NCSA.GT.1 .AND. PRNLEV.GT.3) THEN
          WRITE(IWRIT,'(1X,A,3(2F9.3,2x))') 'AVE CSA> ',CSA1T,CSA2T, &
               CSPA1T,CSPA2T, CSPE1T,CSPE2T
       ENDIF
    ENDIF

    IF(NDQS.GT.0)THEN
       CALL AVENMR(NCOORD,NDQS,DQS1,DQS2,DQS1T,DQS2T)
       IF(PRNLEV.GT.3) THEN
          WRITE(IWRIT,*)
          WRITE(IWRIT,'(6x,3a)') '|------- DQS -------|'
       ENDIF
       DO JDQS=1,NDQS
          CALL ATOMID(IDQS(1,JDQS),SGID1,RID1,REN1,AT1)
          CALL ATOMID(IDQS(2,JDQS),SGID2,RID2,REN2,AT2)
          IF(PRNLEV.GT.3) WRITE(IWRIT,105) JDQS,DQS1(JDQS),DQS2(JDQS), &
               AT1(1:tklen),AT2(1:tklen),REN1(1:tklen), &
               RID1(1:tklen),SGID1(1:tklen)
105       FORMAT(1X,I3,2X,(2F9.3,2x),2X,5(1X,A))
       ENDDO
       IF(NDQS.GT.1)THEN
          IF(PRNLEV.GT.3) WRITE(IWRIT,'(1X,A,3(2F9.3,2x))') &
               'AVE DQS> ',DQS1T,DQS2T
       ENDIF
    ENDIF

    RETURN
  END SUBROUTINE NMRDYN
  !
  SUBROUTINE RARTIMS(X,Y,Z,NRTIMS,IRTIMS,GYRMAG,DSIGMA,CSAR, &
       RTUMBL,HFIELD,CTRTIM,IWRIT,PRNLEV)
    !-----------------------------------------------------------------------
    ! Relaxation Times in 1/SEC in the rigid tumbling molecule approximation
    use chm_kinds
    use exfunc
    use number
    use chutil,only:atomid
    use string

    implicit none
    real(chm_real) X(*),Y(*),Z(*)
    INTEGER NRTIMS(2),IRTIMS(2,*)
    real(chm_real) GYRMAG(*),DSIGMA(*),RTUMBL,HFIELD,CTRTIM
    INTEGER IWRIT,PRNLEV
    !
    ! Local variables
    INTEGER IAT1,IAT2
    INTEGER NMAX
    real(chm_real) CORR12(1),PLAT12,EXPCOS(1)
    real(chm_real) DIST,DELTA
    CHARACTER(len=8) SGID1,RID1,REN1,AT1
    CHARACTER(len=8) SGID2,RID2,REN2,AT2
    real(chm_real) SD12(5)
    real(chm_real) RT1,RT1TOT
    real(chm_real) RT2,RT2TOT
    real(chm_real) RNOE,RROE,RR6
    INTEGER I1,I2
    LOGICAL QVERB,CSAR
    integer tklen

    tklen=4
    if (qxform()) tklen=8
    QVERB=.FALSE.
    !
    IF(PRNLEV.GT.3) WRITE(IWRIT,100) 'SPIN RELAXATION RATES', &
         'RIGID STRUCTURE APPROXIMATION'
100 FORMAT(6X,A)

    ! Relaxation times properties, compute correlation functions
    IF(NRTIMS(1).GT.0)THEN
       NMAX=1
       DELTA=ZERO
       IF(PRNLEV.GT.3) WRITE(IWRIT,102) 'HFIELD [TESLA] = ',HFIELD, &
            'RTUMBL [PSEC]  = ',RTUMBL
102    FORMAT(6X,4(A,2X,F12.3,1X))
       DO I1=1,NRTIMS(1)
          IAT1=IRTIMS(1,I1)
          RT1TOT=ZERO
          RT2TOT=ZERO
          DO I2=1,NRTIMS(2)
             IAT2=IRTIMS(2,I2)
             CSAR=(IAT2.LE.0)
             IF(CSAR)THEN
                IAT2=-IAT2
             ENDIF
             DIST=SQRT((X(IAT1)-X(IAT2))**2+ &
                  (Y(IAT1)-Y(IAT2))**2+ &
                  (Z(IAT1)-Z(IAT2))**2)
             IF(CSAR.AND.(DIST.GT.ONE)) GOTO 11
             IF((DIST.LE.CTRTIM).AND.(IAT1.NE.IAT2))THEN
                CALL ATOMID(IAT1,SGID1,RID1,REN1,AT1)
                CALL ATOMID(IAT2,SGID2,RID2,REN2,AT2)
                IF(PRNLEV.GT.3)THEN
                   IF(CSAR)THEN
                      WRITE(IWRIT,101) 'CSAR AXIS:  ', &
                           AT1(1:tklen),REN1(1:tklen),RID1(1:tklen),SGID1(1:tklen), &
                           AT2(1:tklen),REN2(1:tklen),RID2(1:tklen),SGID2(1:tklen)
                   ELSE
                      WRITE(IWRIT,101) 'SPIN PAIR:  ', &
                           AT1(1:tklen),REN1(1:tklen),RID1(1:tklen),SGID1(1:tklen), &
                           AT2(1:tklen),REN2(1:tklen),RID2(1:tklen),SGID2(1:tklen)
101                   FORMAT(/,6X,A,2(4(A,1X),'  -  '))
                   ENDIF
                ENDIF

                CORR12(1)=ONE/DIST**6
                PLAT12=CORR12(1)
                IF(PRNLEV.GT.3) WRITE(IWRIT,102) '<R>  =',DIST, &
                     'C(0) =',CORR12(1), &
                     'PLAT =',PLAT12, &
                     '<S2> =',PLAT12/CORR12(1)

                CALL NMRSD(GYRMAG(IAT1),GYRMAG(IAT2),HFIELD,CORR12,PLAT12, &
                     DELTA,NMAX,RTUMBL,SD12,EXPCOS,QVERB,IWRIT)
                CALL NMRRAT(GYRMAG(IAT1),GYRMAG(IAT2),CSAR,DSIGMA(IAT2),HFIELD, &
                     RR6,SD12,RT1,RT2,RNOE,RROE,IWRIT)
                RT1TOT=RT1TOT+RT1
                RT2TOT=RT2TOT+RT2
             ENDIF
11           CONTINUE
          ENDDO
          IF(PRNLEV.GT.3)THEN
             IF((RT1TOT.NE.ZERO).OR.(RT2TOT.NE.ZERO))THEN
                WRITE(IWRIT,103) 'TOTAL [1/SEC]:  ', &
                     '1/T1 = ', RT1TOT,'1/T2 = ',RT2TOT, &
                     AT1(1:tklen),REN1(1:tklen),RID1(1:tklen),SGID1(1:tklen)
103             FORMAT(6X,A,2(A,F10.6,1X),3X,4(A,1X))
             ENDIF
          ENDIF
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE RARTIMS
  !
  SUBROUTINE WRNMR (NZMAT,ZMAT,IZMAT, &
       NCSA,ICSA,SIGMA,ZCSA,NDQS,IDQS, &
       NRTIMS,IRTIMS,NTS,ITS,IWRIT)
    !-----------------------------------------------------------------------
    use chm_kinds
    use stream
    use chutil,only:atomid
    implicit none
    ! Write out all relevant NMR information lists
    INTEGER NZMAT
    real(chm_real) ZMAT(3,*)
    INTEGER IZMAT(4,*)
    INTEGER NCSA,ICSA(3,*)
    real(chm_real) SIGMA(3,*),ZCSA(2,2,*)
    INTEGER NDQS,IDQS(2,*)
    INTEGER NRTIMS(2),IRTIMS(2,*)
    INTEGER NTS,ITS(*)
    INTEGER IWRIT
    CHARACTER(len=8) SEGID,RESID,RESNAM,ATYPE
    INTEGER I,J,N
    !
    ! lni - I don't see why we need different treatments here?
    !      IF(IWRIT.EQ.OUTU .AND. PRNLEV.LT.3) RETURN
    !      IF(IWRIT.NE.OUTU .AND. IOLEV.LT.0) RETURN
    IF(PRNLEV.LT.3) RETURN
    !
    IF(NZMAT.NE.0)THEN
       WRITE(IWRIT,*) 'BUILD','  NZMAT ',NZMAT
       DO N=1,NZMAT
          CALL ATOMID(IZMAT(4,N),SEGID,RESID,RESNAM,ATYPE)
          WRITE(IWRIT,*) ATYPE(1:idleng),RESNAM(1:idleng), &
               RESID(1:idleng),SEGID(1:idleng), &
               (IZMAT(I,N),I=1,4),(ZMAT(I,N),I=1,3)
       ENDDO
    ENDIF

    IF(NCSA.NE.0)THEN
       WRITE(IWRIT,*) 'CSA ','  NCSA  ',NCSA
       DO N=1,NCSA
          WRITE(IWRIT,*) (ICSA(I,N),I=1,3),((ZCSA(I,J,N),I=1,2),J=1,2), &
               (SIGMA(I,N),I=1,3)
       ENDDO
    ENDIF

    IF(NDQS.NE.0)THEN
       WRITE(IWRIT,*) 'DQS ','  NDQS  ',NDQS
       WRITE(IWRIT,*) ((IDQS(I,N),I=1,2),N=1,NDQS)
    ENDIF

    IF(NRTIMS(1).NE.0)THEN
       WRITE(IWRIT,*) 'RTIMS ','  N1  ',NRTIMS(1),' N2 ',NRTIMS(2)
       DO I=1,2
          WRITE(IWRIT,*) (IRTIMS(I,N), N=1,NRTIMS(I))
       ENDDO
    ENDIF

    IF(NTS.NE.0)THEN
       WRITE(IWRIT,*) 'TSER ','  NTS  ',NTS
       WRITE(IWRIT,*) (ITS(I),I=1,NTS)
    ENDIF

    WRITE(IWRIT,*) 'END'

    RETURN
  END SUBROUTINE WRNMR
  !
  SUBROUTINE NMRCRR(NBFRAM,NTS,I1,I2,XTSERI,YTSERI,ZTSERI,NCOORD, &
       DELTA,NMAX,CORR12,PLAT12,DXT12,DYT12,DZT12,DRT12, &
       DIST,RR6,IWRIT,QCORRE,QTSERI,QVERB,TAUE,TMAX1,SCONF)
    !-----------------------------------------------------------------------
    !  This subroutine compute the relevant correlation factor
    !  and averages for a given spin pair (k,l) from a molecular
    !  dynamics trajectory stored in (x,y,z)
    !
    !  Meaning of the passing variables:
    !  nbfram          dimension for number of coordinate frames
    !  nts             total nmber of time-series
    !  I1, I2          the two spins involved
    !  delta           time interval
    !  XTSERI(,)       cartesian coordinates during the trajectory
    !  YTSERI(,)         (single precision)
    !  ZTSERI(,)
    !  ncoord          number of actual coordinate frames
    !  nmax            maximum time to compute the correlation function
    !  corr12()        <P2(cos[0;t])/R12(0)**3 R12(t)**3>, notice that
    !                     this convention differs by a factor of 4*PI
    !                     with R.M.Levy et al.,JACS 103, 5998 (1981),
    !                     the factor is cancelled out later and has been
    !                     omited in NMRCRR.
    !  plat12          plateau value of the correlation function corr12(i)
    !  DXT12           work areas
    !
    use chm_kinds
    use consta
    use number
    use stream
    implicit none
    INTEGER NBFRAM,NTS,I1,I2
    REAL(CHM_REAL4) XTSERI(NBFRAM,NTS)
    REAL(CHM_REAL4) YTSERI(NBFRAM,NTS)
    REAL(CHM_REAL4) ZTSERI(NBFRAM,NTS)
    INTEGER NCOORD
    real(chm_real) DELTA
    INTEGER NMAX
    real(chm_real) CORR12(*)
    real(chm_real) PLAT12
    real(chm_real) DXT12(*),DYT12(*),DZT12(*),DRT12(*)
    real(chm_real) DIST,TAUE,TMAX1,SCONF,RR6
    INTEGER IWRIT
    LOGICAL QCORRE,QTSERI,QVERB
    !
    !  Local variables
    INTEGER I,J, NMAX1
    real(chm_real) COSIJ,P2IJ
    COMPLEX*16 EXPIP,AVY22,AVY21,AVY20
    real(chm_real) SINT,COST,SINP,COSP,AMPL,ASD
    LOGICAL QPRINT
    !
    QPRINT=.TRUE.
    IF(IWRIT.EQ.OUTU .AND. PRNLEV.LT.3) QPRINT=.FALSE.
    IF(IWRIT.NE.OUTU .AND. IOLEV.LT.0) QPRINT=.FALSE.
    !
    IF(QVERB .AND. QPRINT)THEN
       WRITE(IWRIT,*)
       WRITE(IWRIT,100) 'TRAJECTORY:'
100    FORMAT(6X,A)
       WRITE(IWRIT,'(2X,8A)') '    Time    ', &
            '    X1(t)   ', '    Y1(t)   ', '    Z1(t)   ', &
            '    X2(t)   ', '    Y2(t)   ', '    Z2(t)'
       DO I=1,NCOORD
          WRITE(IWRIT,101) (I-1)*DELTA, &
               XTSERI(I,I1),YTSERI(I,I1),ZTSERI(I,I1), &
               XTSERI(I,I2),YTSERI(I,I2),ZTSERI(I,I2)
101       FORMAT(F10.3,8F12.4)
       ENDDO
    ENDIF
    !
    !  First calculate time series:
    DO I=1,NCOORD
       DXT12(I)=XTSERI(I,I1)-XTSERI(I,I2)
       DYT12(I)=YTSERI(I,I1)-YTSERI(I,I2)
       DZT12(I)=ZTSERI(I,I1)-ZTSERI(I,I2)
       DRT12(I)=SQRT(DXT12(I)**2+DYT12(I)**2+DZT12(I)**2)
       DXT12(I)=DXT12(I)/DRT12(I)
       DYT12(I)=DYT12(I)/DRT12(I)
       DZT12(I)=DZT12(I)/DRT12(I)
    ENDDO

    IF(QTSERI .AND. QPRINT)THEN
       WRITE(IWRIT,*)
       WRITE(IWRIT,100)       'TIME SERIES:'
       WRITE(IWRIT,'(2X,8A)') '    Time    ','    DR12(t) ', &
            '    DX12(t) ', '    DY12(t) ','    DZ12(t)'
       DO I=1,NCOORD
          WRITE(IWRIT,101) (I-1)*DELTA,DRT12(I),DXT12(I),DYT12(I),DZT12(I)
       ENDDO
    ENDIF

    !  Calculate the correlation function (using the simple sum method):
    DO J=0,NMAX
       CORR12(J+1)=ZERO
       DO I=1,NCOORD-J
          COSIJ=DXT12(I)*DXT12(I+J)+ &
               DYT12(I)*DYT12(I+J)+DZT12(I)*DZT12(I+J)
          P2IJ=(THREE*COSIJ**2-ONE)*HALF
          CORR12(J+1)=CORR12(J+1)+P2IJ/(DRT12(I)*DRT12(I+J))**3
       ENDDO
       CORR12(J+1)=CORR12(J+1)/(NCOORD-J)
    ENDDO

    IF(QCORRE .AND. QPRINT)THEN
       WRITE(IWRIT,*)
       WRITE(IWRIT,100) 'CORRELATION FUNCTION:'
       WRITE(IWRIT,'(6X,2A)') '    Time    ','     C(t)'
       DO I=1,NMAX
          WRITE(IWRIT,101) (I-1)*DELTA,CORR12(I)
       ENDDO
    ENDIF

    !  Calculate the plateau value of the correlation function
    !  plateau = 3/4 <Y2/R**3>**2 + 3 <Y1/R**3>**2 + 1/4 <Y0/R*3>**2

    AVY22=ZERO
    AVY21=ZERO
    AVY20=ZERO
    DIST=ZERO
    RR6=ZERO
    DO I=1,NCOORD
       COST=DZT12(i)
       SINT=SQRT(ONE-DZT12(I)**2)
       SINP=DYT12(I)/SINT
       COSP=DXT12(I)/SINT
#if KEY_GNU==1
       EXPIP=COSP+cmplx(ZERO,ONE,chm_cmpx)*SINP
#else /**/
       EXPIP=COSP+(ZERO,ONE)*SINP
#endif 
       AVY22=AVY22+(SINT*EXPIP)**2/DRT12(I)**3
       AVY21=AVY21+SINT*COST*EXPIP/DRT12(I)**3
       AVY20=AVY20+(THREE*COST**2-ONE)/DRT12(I)**3
       DIST=DIST+DRT12(I)
       RR6=RR6+ONE/DRT12(I)**6
    ENDDO
    AVY22=AVY22/NCOORD
    AVY21=AVY21/NCOORD
    AVY20=AVY20/NCOORD
    PLAT12=PT75*ABS(AVY22)**2+ &
         THREE*ABS(AVY21)**2+ &
         PT25*ABS(AVY20)**2
    DIST=DIST/NCOORD
    RR6=RR6/NCOORD 
    !  Find point of first crossing of plateau value, to be used as maxtime for computing tau-eff
    NMAX1=NMAX
    DO I=0,NMAX-1
       IF(CORR12(I+1) - PLAT12 .LE. RSMALL)THEN
          NMAX1=I
          GOTO 31
       ENDIF
    ENDDO
31  TMAX1=(NMAX1-1)*DELTA
    !  Get tau-eff from integration of correlation function
    CALL TRAPEZ(NMAX1,CORR12,DELTA,TAUE)   
    TAUE=(TAUE-PLAT12*TMAX1)/(CORR12(1)-PLAT12)
    ! Entropy estimate diffusion-in-a-cone (Yang&Kay,JMB263,p369 (1996) "model 3")
    ! neglecting alternative Sconf values for S2 < 1/64, and using approximation A=-0.11
    ! as suggested by Yang&Kay.
    !
    SCONF=PLAT12/CORR12(1)
    SCONF=KBOLTZ*(-0.11 + LOG(PI *(THREE-  &
         SQRT(ONE+EIGHT*SQRT(SCONF)))))

    IF(QPRINT) WRITE(IWRIT,102) '<R>  =',DIST, &
         'C(0) =',CORR12(1), &
         'PLAT =',PLAT12, &
         '<S2> =',PLAT12/CORR12(1), &
         'TMXE =',TMAX1,  &
         'TAUE =',TAUE, &
         'SCONF=',SCONF
102 FORMAT(6X,(A,2X,F12.4,1X))
    RETURN
  END SUBROUTINE NMRCRR
  !
  SUBROUTINE NMRSD(GAMMA1,GAMMA2,HFIELD,CORR12,PLAT12,DELTA, &
       NMAX,RTUMBL,SD12,EXPCOS,QVERB,IWRIT)
    !-----------------------------------------------------------------------
    ! Compute the Spectral Density SD12() from the correlation function
    ! CORR12().  In the NMR analysis facility, the spectral densities are
    ! defined as
    !
    !                  +inf
    !                  /
    !       J(W) =     \  COS(W*t) C(t) Dt    =   J(|W|)
    !                  /
    !                  0
    !
    ! following the convention of R.M. Levy et al., JACS 103, 5998 (1981),
    ! or E.T. Olejniczak et al., JACS 106, 1923 (1983).  Notice that this
    ! convention differs from other notations such as in "Principles of
    ! Nuclear Magnetic  Resonance in One and Two Dimensions"
    ! by R.R. Ernst, G. Bodenhausen and A. Wokaun, Oxford 1987, where
    ! there is a factor of 2 to account for an integral from -inf to +inf
    ! (see section 2.3).
    !
    ! The frequencies W (omega) are:
    !
    ! Spectral densities:  J(0)    J(W1)   J(W2)   J(W1-W2)      J(W1+W2)
    ! frequencies       :    0     omega1  omega2  omega1-omega2 omega1+omega2
    ! variable storage  :  SD12(1) SD12(2) SD12(3) SD12(4)       SD12(5)
    !
    ! SDP12        spectral density, plateau part
    ! SDF12        spectral density, fast part
    !
    use chm_kinds
    use number
    use consta
    use stream
    implicit none
    real(chm_real) GAMMA1,GAMMA2,HFIELD,CORR12(*),PLAT12,DELTA
    INTEGER NMAX
    real(chm_real) RTUMBL,SD12(5),EXPCOS(*)
    INTEGER IWRIT
    LOGICAL QVERB
    ! Local variables
    INTEGER I,J
    real(chm_real) SDP12(5),SDF12(5)
    real(chm_real) OMEGA1,OMEGA2
    real(chm_real) OMEGA(5)
    real(chm_real) PL12
    real(chm_real), PARAMETER :: PSEC=1.0D-12
    LOGICAL QPRINT
    !
    QPRINT=.TRUE.
    IF(IWRIT.EQ.OUTU .AND. PRNLEV.LT.3) QPRINT=.FALSE.
    IF(IWRIT.NE.OUTU .AND. IOLEV.LT.0) QPRINT=.FALSE.
    !
    OMEGA1=GAMMA1*HFIELD*PSEC
    OMEGA2=GAMMA2*HFIELD*PSEC

    OMEGA(1)=ZERO
    OMEGA(2)=ABS(OMEGA1)
    OMEGA(3)=ABS(OMEGA2)
    OMEGA(4)=ABS(OMEGA1-OMEGA2)
    OMEGA(5)=ABS(OMEGA1+OMEGA2)
    IF(QPRINT) WRITE(IWRIT,100) 'FREQUENCIES  [MHz] = ', &
         (1.0D6*OMEGA(I)/(TWO*PI),I=1,5)
100 FORMAT(6X,A,6(F10.4,2X))

    IF(QVERB .AND. QPRINT)THEN
       WRITE(IWRIT,*)
       WRITE(IWRIT,100) 'Integrand functions:'
    ENDIF

    DO J=1,5
       IF(QVERB .AND. QPRINT)THEN
          WRITE(IWRIT,'(6X,A,D12.4)') 'OMEGA(J):  ',OMEGA(J)
          WRITE(IWRIT,'(6X,3A)') '    Time    ',' C(t)-C_plat', &
               '    SDF12   '
       ENDIF
       !  Make the COS(omega*t)*EXP(-t/RTUMBL) function in psec

       IF(RTUMBL .LE. ZERO)THEN
          !  Make no assumption about overall tumbling
          DO I=1,NMAX
             EXPCOS(I)=COS(OMEGA(J)*DELTA*(I-1))
          ENDDO
          SDP12(J)=ZERO
       ELSE
          DO I=1,NMAX
             EXPCOS(I)=COS(OMEGA(J)*DELTA*(I-1))*EXP(-(DELTA*(I-1))/RTUMBL)
          ENDDO
          !  the plateau part:
          SDP12(J)=PLAT12*RTUMBL/(ONE+(OMEGA(J)*RTUMBL)**2)
       ENDIF
       !  the fast relaxation part:
       PL12=PLAT12
       IF(RTUMBL .LE. ZERO) PL12=ZERO
       SDF12(J)=-0.5*EXPCOS(1)*(CORR12(1)-PL12)*DELTA &
            -0.5*EXPCOS(NMAX)*(CORR12(NMAX)-PL12)*DELTA
       DO I=1,NMAX
          SDF12(J)=SDF12(J)+EXPCOS(I)*(CORR12(I)-PL12)*DELTA
          IF(QVERB .AND. QPRINT)THEN
             WRITE(IWRIT,'(6X,8F12.4)') (I-1)*DELTA, &
                  EXPCOS(I)*(CORR12(I)-PL12),SDF12(J)
          ENDIF
       ENDDO
       SD12(J)=SDP12(J)+SDF12(J)
    ENDDO
    IF(QPRINT) THEN
       WRITE(IWRIT,100) 'Fast part          = ',(SDF12(J),J=1,5)
       WRITE(IWRIT,100) 'Plateau part       = ',(SDP12(J),J=1,5)
       WRITE(IWRIT,100) 'SPECTRAL DENSITIES = ',(SD12(J),J=1,5)
    ENDIF
    !
    RETURN
  END SUBROUTINE NMRSD
  !
  SUBROUTINE NMRRAT(GAMMA1,GAMMA2,CSAR,DSIGMA,HFIELD,RR6, &
       SD12,RT1,RT2,RNOE,RROE,IWRIT)
    !-----------------------------------------------------------------------
    ! Calculate the dipole-dipole relaxation rate from the spectral densities
    ! the chemical shift anisotropy relaxation is included (CSAR)
    !
    ! Spectral densities:  J(0)    J(W1)   J(W2)   J(W1-W2)      J(W1+W2)
    ! frequencies       :    0     omega1  omega2  omega1-omega2 omega1+omega2
    ! variable storage  :  SD12(1) SD12(2) SD12(3) SD12(4)       SD12(5)
    !
    ! Many papers report different formulas for T1, T2, T1R, NOE and ROE.
    ! See for instance Levy and M. Karplus p. 445 "Trajcetory studies of NMR
    ! relaxation in flexible molecules, Chap 18, p. 445, American Chemical
    ! Society 1983. See also, I. Solomon, Phys. Rev. 99, 599 (1955)  and
    ! the references given in SUBROUTINE NMRSD.
    !
    ! The rates are converted to [1/SEC] by the factor FACT12
    !
    use chm_kinds
    use number
    use consta
    use stream
    implicit none
    real(chm_real) GAMMA1,GAMMA2,DSIGMA,HFIELD,RR6
    LOGICAL CSAR
    real(chm_real) SD12(*)
    real(chm_real) RT1,RT2,RNOE,RROE
    INTEGER IWRIT
    real(chm_real), PARAMETER :: PLANCK=6.62618D-34, ANGS=1.0D-10
    real(chm_real), PARAMETER :: PSEC=1.0D-12, MU0=1.0D-07,PPM=1.0D-6
    real(chm_real) FACT12, OMEGA1

    ! Chemical shift anistropy relaxation
    IF(CSAR)THEN
       OMEGA1=GAMMA1*HFIELD ! frequency in SEC

       ! 1/T1 =  (2/15)*(CSA*W1)**2*J(W1), from Goldman's book on NMR
       RT1=(2*((PPM*DSIGMA*OMEGA1)**2)/15)*SD12(2)*PSEC/RR6

       ! 1/T2 =  (2/15)*(CSA*W1)**2*((2/3)*J(0)+(1/2)*J(W1))
       RT2= &
            (2*((PPM*DSIGMA*OMEGA1)**2)/15)*(2*SD12(1)/3+SD12(2)/2)*PSEC/RR6

       !     RT1R=ZERO

       RNOE=ZERO

       RROE=ZERO

       IF(IWRIT.EQ.OUTU .AND. PRNLEV.LT.3) RETURN
       IF(IWRIT.NE.OUTU .AND. IOLEV.LE.0) RETURN
       !
       WRITE(IWRIT,100) 'CSA RATES [1/SEC]:  ', &
            '1/T1 = ', RT1,'1/T2 = ',RT2

       RETURN
    ENDIF

    ! Dipole-dipole relaxation
    FACT12=(MU0*(PLANCK/ANGS**3)/(TWO*PI)*GAMMA1*GAMMA2)**2 *PSEC/TEN

    ! 1/T1 = FACT12*(3*J(W1)+J(W1-W2)+6*J(W1+W2))
    !
    RT1=(3*SD12(2)+SD12(4)+6*SD12(5))

    ! 1/T2 = FACT12*(4*J(0)+3*J(W1)+6*J(W2)+J(W1-W2)+6*J(W1+W2))/2
    !
    RT2=FACT12* &
         (4*SD12(1)+3*SD12(2)+6*SD12(3)+SD12(4)+6*SD12(5))*HALF

    ! 1/T1R = FACT12*(3*J(0)+5*J(W1)+2*J(W1+W2))  ???Check the T1 in rotating frame
    !
    !     RT1R=

    ! NOE = FACT12*(-J(W1-W2)+6*J(W1+W2))
    !
    RNOE= ONE + (GAMMA2/GAMMA1)*(-SD12(4)+6*SD12(5))/RT1
    RT1=FACT12*RT1

    ! ROE = FACT12*(3*J(W1)+2*J(W1-W2))
    !
    RROE=FACT12*(3*SD12(2)+2*SD12(4))

    IF(IWRIT.EQ.OUTU .AND. PRNLEV.LT.3) RETURN
    IF(IWRIT.NE.OUTU .AND. IOLEV.LE.0) RETURN
    !
    WRITE(IWRIT,100) 'DIPOLE-DIPOLE RATES [1/SEC]:  ', &
         '1/T1 = ', RT1,'1/T2 = ',RT2, &
         'SS_NOE  = ',RNOE,'ROE  = ',RROE
100 FORMAT(6X,A,6(A,F10.5,1X))

    RETURN
  END SUBROUTINE NMRRAT
  
  SUBROUTINE TRAPEZ(NP,Y,DELTA,S)   
    !
    ! Compute integral S of curve Y, with NP points spaced DELTA. Extended trapezoidal rule.
    ! Basic routine without extrapolations
    ! Lennart Nilsson, November 2003
    !
    use chm_kinds
    implicit none
    INTEGER NP
    real(chm_real) Y(NP),DELTA,S
    !
    INTEGER I
    !
    S=0.5D0*(Y(1)+Y(NP))
    DO I=2,NP-1
       S=S+Y(I)
    ENDDO
    S=S*DELTA
    RETURN
  END SUBROUTINE TRAPEZ
#endif /* (nmrmain)*/
end module nmrm


