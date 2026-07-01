#if KEY_RISM==0 /*rism_main*/

SUBROUTINE NULL_IREAD
  RETURN
end SUBROUTINE NULL_IREAD

#else /* (rism_main)*/

SUBROUTINE IREAD
  !-------------------------------------------------------------------
  !     This subroutine controls the input-output of all quantities
  !     that can be read: the TITLE, the COORDINATES, the ZMATRIX,
  !     the PARAMETERS, the STRUCTURE and the various DISTRIBUTION
  !     functions DC(R),DG(R),CS(R),US(R),G(R),CH(K)
  !
  use dimens_fcm
  use comand
  use ctitla
  use exfunc
  use stream
  use string
  !
  use rism
  use rism_control
  use fft
  use distri
  use struc
  use chm_kinds
  implicit none
  !
  CHARACTER(len=4) JOB
  INTEGER OUTUN
  INTEGER IOFF,IPAIR,NPAIR,IUA,IUNIT,IU
  LOGICAL QEND
  CHARACTER(len=4) WRD4

  !     Local Heap management
  !!      INTEGER IOFF1,IOFF2

  OUTUN=-ABS(OUTU)
  IF(CHECQUE(COMLYN,'SHOW')) OUTUN=ABS(OUTU)
  IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',ISTRM)
  WRD4=NEXTA4(COMLYN,COMLEN)

  !     read the title
  !     --------------
  IF(.NOT.CHECQUE(COMLYN,'NOTITL'))THEN
     CALL RDTITL(TITLEA,NTITLA,IUNIT,0)
  ENDIF

  !     Read the coordinates
  !     --------------------
  IF(WRD4.EQ.'COOR')THEN
     IF(CHECQUE(COMLYN,'SOLVENT')) THEN
        CALL RDCOOR(IUNIT,ATNAM,RESNAM,X,Y,Z,SEGMID,NSITV,OUTUN)
     ELSEIF(CHECQUE(COMLYN,'SOLUTE'))THEN
        IU=1
        IF(DOTRIM(COMLYN)) IU=NEXTI(COMLYN,COMLEN)
        IOFF=DSITV+(IU-1)*DSITU+1
        CALL RDCOOR(IUNIT,ATNAM(IOFF),RESNAM(IOFF),X(IOFF), &
             Y(IOFF),Z(IOFF),SEGMID(IOFF),NSITU(IU),OUTUN)
     ELSE
        CALL RDCOOR(IUNIT,ATNAM,RESNAM,X,Y,Z,SEGMID,NSITV,OUTUN)
     ENDIF


     !     Read the coordinates of the changed structure
     !     ---------------------------------------------
  ELSEIF(WRD4.EQ.'2COO')THEN
     IF(CHECQUE(COMLYN,'SOLVENT')) THEN
        CALL RDCOOR(IUNIT,ATNAM,RESNAM,X2,Y2,Z2,SEGMID,NSITV,OUTUN)
     ELSEIF(CHECQUE(COMLYN,'SOLUTE'))THEN
        IU=1
        IF(DOTRIM(COMLYN)) IU=NEXTI(COMLYN,COMLEN)
        IOFF=DSITV+(IU-1)*DSITU+1
        CALL RDCOOR(IUNIT,ATNAM(IOFF),RESNAM(IOFF),X2(IOFF), &
             Y2(IOFF),Z2(IOFF),SEGMID(IOFF),NSITU(IU),OUTUN)
     ELSE
        CALL RDCOOR(IUNIT,ATNAM,RESNAM,X2,Y2,Z2,SEGMID,NSITV,OUTUN)
     ENDIF

     !     Read the Z matrix
     !     -----------------
  ELSEIF(WRD4.EQ.'ZMAT')THEN
     IF(CHECQUE(COMLYN,'SOLVENT')) THEN
        CALL RDZMAT(COMLYN,COMLEN,NSITV,ATNAM, &
             IZMAT,ZMAT,IUNIT,OUTUN)
        CALL ZCONSTR(.TRUE.,1,NSITV,IZMAT,ZMAT,X,Y,Z)
     ELSEIF(CHECQUE(COMLYN,'SOLUTE'))THEN
        IU=1
        IF(DOTRIM(COMLYN)) IU=NEXTI(COMLYN,COMLEN)
        IOFF=DSITV+(IU-1)*DSITU+1
        CALL RDZMAT(COMLYN,COMLEN,NSITU(IU),ATNAM(IOFF), &
             IZMAT(1,1,IU+1),ZMAT(1,1,IU+1),IUNIT,OUTUN)
        CALL ZCONSTR(.TRUE.,1,NSITU(IU),IZMAT(1,1,IU+1), &
             ZMAT(1,1,IU+1),X(IOFF),Y(IOFF),Z(IOFF))
     ELSE
        CALL RDZMAT(COMLYN,COMLEN,NSITV,ATNAM, &
             IZMAT,ZMAT,IUNIT,OUTUN)
        CALL ZCONSTR(.TRUE.,1,NSITV,IZMAT,ZMAT,X,Y,Z)
     ENDIF


     !     Read the second Z matrix
     !     -----------------
  ELSEIF(WRD4.EQ.'2ZMA')THEN
     IF(CHECQUE(COMLYN,'SOLVENT')) THEN
        CALL RDZMAT(COMLYN,COMLEN,NSITV,ATNAM, &
             IZMAT,ZMAT,IUNIT,OUTUN)
        CALL ZCONSTR(.TRUE.,1,NSITV,IZMAT,ZMAT,X2,Y2,Z2)
     ELSEIF(CHECQUE(COMLYN,'SOLUTE'))THEN
        IU=1
        IF(DOTRIM(COMLYN)) IU=NEXTI(COMLYN,COMLEN)
        IOFF=DSITV+(IU-1)*DSITU+1
        CALL RDZMAT(COMLYN,COMLEN,NSITU(IU),ATNAM(IOFF), &
             IZMAT(1,1,IU+1),ZMAT(1,1,IU+1),IUNIT,OUTUN)
        CALL ZCONSTR(.TRUE.,1,NSITU(IU),IZMAT(1,1,IU+1), &
             ZMAT(1,1,IU+1),X2(IOFF),Y2(IOFF),Z2(IOFF))
     ELSE
        CALL RDZMAT(COMLYN,COMLEN,NSITV,ATNAM, &
             IZMAT,ZMAT,IUNIT,OUTUN)
        CALL ZCONSTR(.TRUE.,1,NSITV,IZMAT,ZMAT,X2,Y2,Z2)
     ENDIF


     !     Read the parameters
     !     -------------------
  ELSEIF(WRD4.EQ.'PARA')THEN
     CALL RDPARA(IUNIT,OUTUN)

     !     Read the structure
     !     ------------------
  ELSEIF(WRD4.EQ.'STRU')THEN
     CALL RDSTRUC(COMLYN,COMLEN,IUNIT,OUTUN)

  ELSE

     !     For distribution functions only
     IPAIR=GTRMI(COMLYN,COMLEN,'PAIR',1)
     !...  FIRS is the default for JOB
     JOB='FIRS'
     IF(CHECQUE(COMLYN,'LAST')) JOB='LAST'
     CALL GTOFFST(COMLYN,COMLEN,IOFF,NPAIR,IUA)

     !     Read the solvent-solvent derivative
     !     -----------------------------------
     IF(WRD4.EQ.'DC(R')THEN
        CALL RDDAT(IUNIT,IPDCSR(1,NPRVV*(IUA-1)+1),SW,JOB,QEND)

        !     Read the solvent-solvent derivative
        !     -----------------------------------
     ELSEIF(WRD4.EQ.'DG(R')THEN
        CALL RDDAT(IUNIT,IPDGR(1,NPRVV*(IUA-1)+1),SW,JOB,QEND)

        !     Read the short range direct correlation function
        !     ------------------------------------------------
     ELSEIF(WRD4.EQ.'CS(R')THEN
        CALL RDDAT(IUNIT,IPCSR(1,IOFF),SW,JOB,QEND)

        !     Read the short range potential us(r)
        !     ------------------------------------
     ELSEIF(WRD4.EQ.'US(R')THEN
        CALL RDDAT(IUNIT,IPUSR(1,IOFF),SW,JOB,QEND)

        !     Read the 2 particles correlation function
        !     ------------------------------------------
     ELSEIF(WRD4.EQ.'G(R)')THEN
        CALL RDDAT(IUNIT,IPGR(1,IOFF),SW,JOB,QEND)

        !     Read the solvent-solvent suceptibility
        !     -------------------------------------
     ELSEIF(WRD4.EQ.'CH(K')THEN
        CALL RDDAT(IUNIT,IPXVVK,SW,JOB,QEND) !ONLY 'VV'

     ENDIF
  ENDIF
  !
  RETURN
END SUBROUTINE IREAD

SUBROUTINE IWRIT
  !-----------------------------------------------------------------------
  !     This subroutine controls the input-output of all quantities
  !     that are to be written: the TITLE, the COORDINATES, the ZMATRIX,
  !     the PARAMETERS, the STRUCTURE and the various DISTRIBUTION
  !     functions DC(R),DG(R),CS(R),US(R),G(R),CH(K)
  use dimens_fcm
  use comand
  use ctitla
  use exfunc
  use stream 
  use string
  !
  use rism 
  use rism_control
  use fft
  use distri
  use struc
  use chm_kinds
  implicit none
  !
  CHARACTER(len=4) WRD4
  LOGICAL PLT2
  real(chm_real) RFIRST,RLAST
  INTEGER IOFF,NPAIR,IUNIT,IU,IUA

  !     Local Heap management
  !!      INTEGER IOFF1,IOFF2

  IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
  WRD4=NEXTA4(COMLYN,COMLEN)
  !
  !     write the title
  !     --------------
  IF( (WRD4.EQ.'TITL').OR. &
       (.NOT. ((IUNIT.EQ.6).OR.(CHECQUE(COMLYN,'APPE')))) )THEN
     CALL RDTITL(TITLEA,NTITLA,ISTRM,0)
     CALL WRTITL(TITLEA,NTITLA,IUNIT,0)
  ENDIF

  !     Write the coordinates
  !     ---------------------
  IF(WRD4.EQ.'COOR')THEN
     IF(CHECQUE(COMLYN,'SOLVENT')) THEN
        CALL WRCOOR(IUNIT,ATNAM,RESNAM,X,Y,Z,SEGMID,NSITV,IUNIT)
     ELSEIF(CHECQUE(COMLYN,'SOLUTE'))THEN
        IU=1
        IF(DOTRIM(COMLYN)) IU=NEXTI(COMLYN,COMLEN)
        IOFF=DSITV+(IU-1)*DSITU+1
        CALL WRCOOR(IUNIT,ATNAM(IOFF),RESNAM(IOFF),X(IOFF), &
             Y(IOFF),Z(IOFF),SEGMID(IOFF),NSITU(IU),IUNIT)
     ELSE
        CALL WRCOOR(IUNIT,ATNAM,RESNAM,X,Y,Z,SEGMID,NSITV,IUNIT)
     ENDIF

  ELSE
     !     For all distribution functions
     CALL GTOFFST(COMLYN,COMLEN,IOFF,NPAIR,IUA)
     PLT2=CHECQUE(COMLYN,'PLT2')

     IF(WRD4.EQ.'DG(R')THEN
        IF(PLT2)THEN
           CALL PLTDAT(COMLYN,COMLEN,IUNIT,R,IPDGR(1,NPRVV*(IUA-1)+1), &
                NPVV,PRNAM)
        ELSE
           CALL WRTDAT(IUNIT,IPDGR(1,NPRVV*(IUA-1)+1),NPOINT,NPVV,SW)
        ENDIF

     ELSEIF(WRD4.EQ.'DC(R')THEN
        IF(PLT2)THEN
           CALL PLTDAT(COMLYN,COMLEN,IUNIT,R,IPDCSR(1,NPRVV*(IUA-1)+1), &
                NPVV,PRNAM)
        ELSE
           CALL WRTDAT(IUNIT,IPDCSR(1,NPRVV*(IUA-1)+1),NPOINT,NPVV,SW)
        ENDIF

     ELSEIF(WRD4.EQ.'CS(R')THEN
        IF(PLT2)THEN
           CALL PLTDAT(COMLYN,COMLEN,IUNIT,R,IPCSR(1,IOFF), &
                NPAIR,PRNAM(IOFF))
        ELSE
           CALL WRTDAT(IUNIT,IPCSR(1,IOFF),NPOINT,NPAIR,SW)
        ENDIF

     ELSEIF(WRD4.EQ.'US(R')THEN
        IF(PLT2)THEN
           CALL PLTDAT(COMLYN,COMLEN,IUNIT,R,IPUSR(1,IOFF), &
                NPAIR,PRNAM(IOFF))
        ELSE
           CALL WRTDAT(IUNIT,IPUSR(1,IOFF),NPOINT,NPAIR,SW)
        ENDIF

     ELSEIF(WRD4.EQ.'CS(K')THEN
        IF(PLT2)THEN
           CALL PLTDAT(COMLYN,COMLEN,IUNIT,RK,IPCSK(1,IOFF), &
                NPAIR,PRNAM(IOFF))
        ELSE
           CALL WRTDAT(IUNIT,IPCSK(1,IOFF),NPOINT,NPAIR,SW)
        ENDIF

     ELSEIF(WRD4.EQ.'G(R)')THEN
        IF(PLT2)THEN
           CALL PLTDAT(COMLYN,COMLEN,IUNIT,R,IPGR(1,IOFF), &
                NPAIR,PRNAM(IOFF))
        ELSE
           CALL WRTDAT(IUNIT,IPGR(1,IOFF),NPOINT,NPAIR,SW)
        ENDIF

     ELSEIF(WRD4.EQ.'CH(K')THEN
        IF(PLT2)THEN
           CALL PLTDAT(COMLYN,COMLEN,IUNIT,RK,IPXVVK &
                ,NPAIR,PRNAM)
        ELSE
           CALL WRTDAT(IUNIT,IPXVVK,NPOINT,NPV,SW)
        ENDIF

     ELSEIF(WRD4.EQ.'R(I)')THEN
        CALL WRTDAT(IUNIT,R,NPOINT,1,SW)

     ELSEIF(WRD4.EQ.'RK(I')THEN
        CALL WRTDAT(IUNIT,RK,NPOINT,1,SW)

     ENDIF
  ENDIF

  RETURN
END SUBROUTINE IWRIT

SUBROUTINE GTOFFST(COMLYN,COMLEN,IOFF,NPAIR,IUA)
  !-----------------------------------------------------------------------
  !     This subroutine finds what distribution function is asked
  use string
  use rism
  !
  use distri
  use chm_kinds
  implicit none
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN,IOFF,NPAIR,IUA
  !
  INTEGER IUB,IUAA
  !
  NPAIR=NPVV
  IOFF=1

  IF(CHECQUE(COMLYN,'VV'))THEN
     NPAIR=NPVV
     IOFF=1

  ELSEIF(CHECQUE(COMLYN,'UV'))THEN
     IUA=GTRMI(COMLYN,COMLEN,'SOLU',1)
     IOFF=PUV(IUA)
     NPAIR=NPUV(IUA)

  ELSEIF(CHECQUE(COMLYN,'UU'))THEN
     IUA=GTRMI(COMLYN,COMLEN,'SOLU',1)
     IUB=GTRMI(COMLYN,COMLEN,'SOLU',1)
     IF(IUA.LT.IUB)THEN
        IOFF=IUA
        IUA=IUB
        IUB=IOFF
     ENDIF
     IUAA=(IUA*(IUA-1))/2+IUB
     IOFF=PUU(IUAA)
     NPAIR=NPUU(IUAA)
     !
     ! For dgr and dcsr
     !
  ELSE
     IUA=GTRMI(COMLYN,COMLEN,'SOLU',1)
  ENDIF

  RETURN
END SUBROUTINE GTOFFST

SUBROUTINE WRTDAT(IUNIT,RDATA,NPOINT,NPAIR,SW)
  !-----------------------------------------------------------------------
  !     This subroutine writes the data into a file
  use rism
  use stream
  use chm_kinds
  implicit none
  real(chm_real) RDATA(DVECT,*),SW(*)
  INTEGER IUNIT,NPOINT,NPAIR
  !
  INTEGER I,IP,IR

  WRITE(IUNIT,*) NPOINT,NPAIR
  WRITE(IUNIT,*) (SW(I),I=1,4)
  DO IP=1,NPAIR
     WRITE(IUNIT,*) (RDATA(IR,IP),IR=1,NPOINT)
  ENDDO
  WRITE(OUTU,100) NPAIR,NPOINT,IUNIT,(SW(I),I=1,4)
100 FORMAT(6X,I4,' pairs of ',I4,' points were written to unit',I4,/, &
       6X,' switch =',4F9.4,/)

  RETURN
END SUBROUTINE WRTDAT

SUBROUTINE PLTDAT(COMLYN,COMLEN,IUNIT,AXIS,RDATA,NPAIR,PRNAM)
  !-----------------------------------------------------------------------
  !     This subroutine writes the various data in formatted form and
  !     as a function of distance
  use string
  use rism
  use stream
  use fft
  use chm_kinds
  implicit none
  real(chm_real)        AXIS(*),RDATA(DVECT,*)
  CHARACTER(len=*) COMLYN
  CHARACTER(len=13) PRNAM(*)
  INTEGER       COMLEN,IUNIT,NPAIR
  !
  !     local
  real(chm_real) R1,R2
  INTEGER NCOUNT,PLIST(DPAIR),IFRMAT,IPAIR,I,NP,IR
  CHARACTER(len=80) FRMAT,FRMAT2

  !     get the first pair
  IPAIR=GTRMI(COMLYN,COMLEN,'PAIR',1)
  NCOUNT=1
  PLIST(NCOUNT)=IPAIR
  !
1000 CONTINUE
  IPAIR=GTRMI(COMLYN,COMLEN,'PAIR',-1)
  IF(IPAIR.NE.-1)THEN
     NCOUNT=NCOUNT+1
     PLIST(NCOUNT)=IPAIR
     GOTO 1000
  ENDIF

  IF(CHECQUE(COMLYN,'ALL'))THEN
     NCOUNT=NPAIR
     DO I=1,NPAIR
        PLIST(I)=I
     ENDDO
  ENDIF

  CALL GETBETW(COMLYN,'FORMAT(',')',FRMAT)
  IF(DOTRIM(FRMAT))THEN
     IFRMAT=LEN(FRMAT)
     CALL TRIMA(FRMAT,IFRMAT)
     FRMAT2='('//FRMAT(1:IFRMAT)//')'
     FRMAT=FRMAT2
  ELSE
     FRMAT='(1X,25F10.3)'
  ENDIF

  !     get the distances
  R1=GTRMF(COMLYN,COMLEN,'FROM',R(NFIRST))
  R2=GTRMF(COMLYN,COMLEN,'THRU',R(NPOINT))

  NP=0
  DO IR=NFIRST,NPOINT
     IF((AXIS(IR).GE.R1).AND.(AXIS(IR).LE.R2))THEN
        NP=NP+1
        WRITE(IUNIT,FRMAT) AXIS(IR),(RDATA(IR,PLIST(I)),I=1,NCOUNT)
     ENDIF
  ENDDO

  WRITE(OUTU,101) (I,PRNAM(PLIST(I)),I=1,NCOUNT)
101 FORMAT(6X,30(I4,' Pair:  ',A13,', '))
  WRITE(OUTU,102) NP,IUNIT
102 FORMAT(6X,I4,' points were written to unit',I4,/)

  RETURN
END SUBROUTINE PLTDAT

SUBROUTINE RDDAT(IUNIT,RDATA,SW,JOB,QEND)
  !-----------------------------------------------------------------------
  !     This subroutine reads the various data from a file
  use rism
  use stream
  use chm_kinds
  implicit none
  real(chm_real) RDATA(DVECT,*),SW(*)
  CHARACTER(len=4) JOB
  LOGICAL QEND
  !
  !     local
  INTEGER IUNIT,NPOINT,NPAIR,I,IP,IR

  IF(JOB.EQ.'FIRS')THEN
     READ(IUNIT,*) NPOINT,NPAIR
     READ(IUNIT,*) (SW(I),I=1,4)
     DO IP=1,NPAIR
        READ(IUNIT,*,END=2000) (RDATA(IR,IP),IR=1,NPOINT)
     ENDDO
     WRITE(OUTU,105) NPAIR,NPOINT,IUNIT,(SW(I),I=1,4)
  ELSEIF(JOB.EQ.'LAST')THEN
1000 CONTINUE
     READ(IUNIT,*,END=2000) NPOINT,NPAIR
     READ(IUNIT,*) (SW(I),I=1,4)
     DO IP=1,NPAIR
        READ(IUNIT,*) (RDATA(IR,IP),IR=1,NPOINT)
     ENDDO
     WRITE(OUTU,105) NPAIR,NPOINT,IUNIT,(SW(I),I=1,4)
     GOTO 1000
  ENDIF
  RETURN
105 FORMAT(6X,I4,' pairs of ',I4,' points were read from unit',I4,/, &
       6X,' switch =',4F9.4,/)

2000 CONTINUE
  QEND=.TRUE.
  WRITE(OUTU,'(A)') ' RISM: RDDAT> End of File Encountered'
  !
  RETURN
END SUBROUTINE RDDAT

SUBROUTINE RDCOOR(IUNIT,ATNAM,RESNAM,X,Y,Z,SEGID,NSITE,OUTUN)
  !-----------------------------------------------------------------------
  !     Read the coodinates and the atom names in CHARMM format
  use stream
  use rism
  use chm_kinds
  implicit none
  real(chm_real) X(*),Y(*),Z(*)
  CHARACTER(len=*) ATNAM(*),RESNAM(*),SEGID(*)
  INTEGER IUNIT,NSITE,OUTUN
  !
  !     local
  INTEGER I,IAT,IRES
  CHARACTER(len=6) ATNA2,RESNA2,SEGI2
  LOGICAL TEST

  !     read the number of site
  READ(IUNIT,'(I5)') NSITE

  !     read the coordinates
  DO I=1,NSITE
     READ(IUNIT,115) IAT,IRES,RESNA2,ATNA2,X(I),Y(I),Z(I),SEGI2
     TEST=(RESNAM(I).EQ.RESNA2).AND.(ATNAM(I).EQ.ATNA2) &
          .AND.(SEGID(I).EQ.SEGI2)
     IF(.NOT.TEST) THEN
        !
        ! turn on the printing flag
        !
        OUTUN=OUTU 
        WRITE(OUTU,'(A)') ' RISM: RDCOOR> ERROR: Sequence Mismatch'
     ENDIF
     IF(OUTUN.GT.0) THEN
        WRITE(OUTUN,115) IAT,IRES,RESNA2,ATNA2,X(I),Y(I),Z(I),SEGI2
     ENDIF
  ENDDO
  !
  IF(OUTUN.GT.0) WRITE(OUTUN,*)
  !
115 FORMAT(I5,I5,1X,A4,1X,A4,3F10.5,1X,A4)
  RETURN
END SUBROUTINE RDCOOR

SUBROUTINE WRCOOR(IUNIT,ATNAM,RESNAM,X,Y,Z,SEGID,NSITE,OUTUN)
  !-----------------------------------------------------------------------
  !     Write the coodinates and the atom names in CHARMM format
  use stream
  use rism
  use chm_kinds
  implicit none
  real(chm_real) X(*),Y(*),Z(*)
  CHARACTER(len=*) ATNAM(*),RESNAM(*),SEGID(*)
  INTEGER IUNIT,NSITE,OUTUN
  !
  !     local
  INTEGER IRES,I

  !     write the number of site
  WRITE(IUNIT,'(I5)') NSITE

  !     write the coordinates
  IRES=1
  DO I=1,NSITE
     WRITE(IUNIT,115) I,IRES,RESNAM(I),ATNAM(I),X(I),Y(I),Z(I), &
          SEGID(I)
     IF((I.LT.NSITE).AND. &
          (RESNAM(I+1).NE.RESNAM(I))) IRES=IRES+1
  ENDDO
  !
  IF(IUNIT.EQ.OUTU) WRITE(IUNIT,*)
  !
115 FORMAT(I5,I5,1X,A4,1X,A4,3F10.5,1X,A4)
  RETURN
END SUBROUTINE WRCOOR

SUBROUTINE RDPARA(IUNIT,OUTUN)
  !-----------------------------------------------------------------------
  !     This subroutine reads the vdW epsilon and sigma parameters
  !     for the various atoms . It also reads the coulomb charges.
  !     The input line should look like:
  ! 
  !     NBOND
  !     ATOM-NAME  EPSILON      RMIN/2     CHARGE
  !     ............................................
  !     END
  !     NBFIX
  !     ATOM-NAME  ATOM-NAME    EPSILON    RMIN
  !     END
  !
  !     The epsilon values can be negative or positive because they are
  !     converted to their absolute value. Notice that the NBFIX 
  !     specification is for RMIN and NOT RMIN/2.
  !
  use dimens_fcm
  use comand
  use exfunc
  use number
  use stream
  use string
  use rism   
  use struc
  use chm_kinds
  implicit none
  INTEGER   IUNIT,OUTUN
  !
  !     Local variables
  !     ---------------
  real(chm_real) EPS1(DBASE),SIG1(DBASE)
  real(chm_real) EPSIJ,SIGIJ
  real(chm_real) CONVRT1,CONVRT2
  !
  CHARACTER(len=6) WRD1,WRD2
  CHARACTER(len=25) WRD25
  INTEGER IRULE/1/,NPBASE
  INTEGER I,J
  LOGICAL EOF
  !
  !     set the standard combination rule as a default
  !    
  EOF = .FALSE.
  !
  !     CONVRT1 does RMIN/2 --> SIGMA
  !     CONVRT2 is used for NBFIX
  !
  CONVRT1=TWO**(ONE/SIX)/TWO
  CONVRT2=TWO**(ONE/SIX)

1000 CONTINUE
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.TRUE., &
       'PARAM> ')
  IF(EOF) GOTO 9000
  IF(COMLEN.LE.0) GOTO 1000
  IF(CHECQUE(COMLYN,'COMBI'))THEN
     IF(CHECQUE(COMLYN,'STANDARD'))THEN
        !
        !     This is the default value
        !
        IRULE=1
     ELSEIF(CHECQUE(COMLYN,'JORGE'))THEN
        IRULE=2
     ENDIF

  ELSEIF(CHECQUE(COMLYN,'NBOND'))THEN
     NBASE=0
     !
2000 CONTINUE
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.TRUE., &
          'NBOND> ')
     IF(EOF) GOTO 9000
     IF(COMLEN.LE.0) GOTO 2000
     IF(CHECQUE(COMLYN,'END'))THEN
        !
        !     Generate all the non-bonded parameters
        !
        NPBASE=(NBASE*(NBASE+1))/2
        DO I=1,NBASE
           DO J=1,I
              INBASE(I,J)=((I-1)*I)/2+J
              INBASE(J,I)=INBASE(I,J)
              CALL COMBINE(IRULE,EPSIL(INBASE(I,J)),EPS1(I),EPS1(J), &
                   SIG(INBASE(I,J)),SIG1(I),SIG1(J))
           ENDDO
        ENDDO
        GOTO 1000
     ELSE
        NBASE=NBASE+1
        IF(NBASE.GT.DBASE)THEN
           OUTUN=ABS(OUTUN)
           WRITE(OUTUN,115) NBASE
           CALL WRNDIE(-5,'<RDPARA>','JOB ABORTED')
           RETURN
        ENDIF

        BASE(NBASE)=NEXTA6(COMLYN,COMLEN)
        EPS1(NBASE)=NEXTF(COMLYN,COMLEN)
        EPS1(NBASE)=ABS(EPS1(NBASE))
        SIG1(NBASE)=NEXTF(COMLYN,COMLEN)
        SIG1(NBASE)=SIG1(NBASE)/CONVRT1 
        CHARG(NBASE)=NEXTF(COMLYN,COMLEN)

        GOTO 2000
     ENDIF

  ELSEIF(CHECQUE(COMLYN,'NBFIX'))THEN
3000 CONTINUE
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.TRUE., &
          'NBFIX>')
     IF(EOF) GOTO 9000
     IF(COMLEN.LE.0) GOTO 3000
     IF(CHECQUE(COMLYN,'END'))THEN
        GOTO 1000
     ELSE
        WRD1=NEXTA6(COMLYN,COMLEN)
        WRD2=NEXTA6(COMLYN,COMLEN)
        CALL FNBASE(BASE,NBASE,WRD1,I)
        CALL FNBASE(BASE,NBASE,WRD2,J)
        IF((I.GT.0).AND.(I.GT.0))THEN
           EPSIJ=NEXTF(COMLYN,COMLEN)
           SIGIJ=NEXTF(COMLYN,COMLEN)
           SIGIJ=SIGIJ/CONVRT2
           EPSIL(INBASE(I,J))=ABS(EPSIJ)
           SIG(INBASE(I,J))=SIGIJ
        ENDIF
        GOTO 3000
     ENDIF

  ELSEIF(CHECQUE(COMLYN,'END'))THEN
     WRITE(OUTU,125) NBASE
     WRITE(OUTU,*)
     IF(OUTUN.GT.0)THEN
        WRITE(OUTUN,135)
        IF(IRULE.EQ.1) WRITE(OUTUN,145) 
        IF(IRULE.EQ.2) WRITE(OUTUN,155)
        WRITE(OUTUN,165)
        DO I=1,NBASE
           DO J=I,NBASE
              WRITE(OUTUN,175) BASE(I),BASE(J),EPSIL(INBASE(I,J)), &
                   SIG(INBASE(I,J)),CHARG(I),CHARG(J)
           ENDDO
        ENDDO
        WRITE(OUTUN,*)
     ENDIF
     RETURN
  ENDIF

  GOTO 1000
  !
9000 CONTINUE
  WRITE(OUTUN,'(A)') ' RISM:RDPARA> ERROR: End-of-File reached.'
  RETURN
  !
115 FORMAT(' * RDPARA * error, nbase=',I6,' greater than limit ',/)
125 FORMAT(1X,I3,' atom types have been read')
135 FORMAT(/,' Non-Bonded parameter set')
145 FORMAT(1X,'standard combination rule:  ', &
       'SIG(ij)=(SIG(i)+SIG(j))/2',/,29X, &
       'EPSIL(ij)=SQRT(EPSIL(i)*EPSIL(j))')
155 FORMAT(1X,'Jorgensen Combination Rule:  ', &
       'SIG(ij)=SQRT(SIG(i)*SIG(j))',/,29X, &
       'EPSIL(ij)=SQRT(EPSIL(i)*EPSIL(j))')
165 FORMAT(/,1X, &
       ' atom  pair    eps(i,j)  sig(i,j)    Q(i)      Q(j)',//)
175 FORMAT(1X,A,1X,A,4F10.4)
  !
END SUBROUTINE RDPARA

SUBROUTINE COMBINE(IRULE,EPS12,EPS1,EPS2,SIG12,SIG1,SIG2)
  !-----------------------------------------------------------------------
  !     This subroutine applies a combination rule to generate the
  !     LJ6-12 parameters.  The standard rule=1 and  Jorgensen rule=2.
  use number
  use chm_kinds
  implicit none
  real(chm_real) EPS12,EPS1,EPS2,SIG12,SIG1,SIG2
  INTEGER IRULE

  IF(IRULE.EQ.1)THEN
     !     Standard rule
     EPS12=SQRT(EPS1*EPS2)
     SIG12=HALF*(SIG1+SIG2)
  ELSEIF(IRULE.EQ.2)THEN
     !     Jorgensen rule
     EPS12=SQRT(EPS1*EPS2)
     SIG12=SQRT(SIG1*SIG2)
  ENDIF

  RETURN
END SUBROUTINE COMBINE

SUBROUTINE FNBASE(BASE,NBASE,WRD,I)
  !-----------------------------------------------------------------------
  !     Checks if the atom WRD has been previously mentioned in the
  !     parameter list
  use stream
  use chm_kinds
  implicit none
  CHARACTER(len=*) BASE(*)
  CHARACTER(len=6) WRD
  INTEGER I,NBASE

  DO I=1,NBASE
     IF(WRD.EQ.BASE(I)) RETURN
  ENDDO
  WRITE(OUTU,105) WRD
105 FORMAT(' * FNBASE * error atom ',A6,' not found')
  I=-1
  RETURN
END SUBROUTINE FNBASE

SUBROUTINE RDSTRUC(COMLYN,COMLEN,IUNIT,OUTUN)
  !-----------------------------------------------------------------------
  !     This subroutine reads in the structure and generates 
  !     all the pointers. It can also read a ZMATRIX, if it has been 
  !     appended after the structure description
  use dimens_fcm
  use exfunc
  use rism
  use distri
  use struc
  use chm_kinds
  use string
  implicit none
  CHARACTER(len=*) COMLYN
  INTEGER       COMLEN,IUNIT,OUTUN
  !
  !     Local variables
  INTEGER NTYPV,NTYPU(DU)
  INTEGER IOFF,IU
  CHARACTER(len=1) WRD
  LOGICAL EOF
  !
  EOF = .FALSE.

1000 CONTINUE
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.TRUE., &
       'STRUC> ')
  IF(EOF) GOTO 9000
  IF(COMLEN.LE.0) GOTO 1000

2000 CONTINUE
  IF(CHECQUE(COMLYN,'SOLVENT'))THEN
     !
     ! Set default names for the solvent
     !
     SEGMID(1)='SOLV'
     RESNAM(1)='SOLV'
     CALL RDSTRU2(COMLYN,COMLEN,IUNIT,OUTUN,NBASE,BASE, &
          ATNAM,RESNAM,SEGMID,NSITV,DSITV,NTYPV,ITYPE,ICHEM, &
          X,Y,Z)
     CALL SEGMENT(NSITV,SEGMID,ASEGM,NSEGV,ISEGM)
2010 CONTINUE
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.TRUE., &
          'STRUC> ')
     IF(EOF) GOTO 9000
     IF(COMLEN.LE.0) GOTO 2010
     IF(CHECQUE(COMLYN,'ZMAT'))THEN
        CALL RDZMAT(COMLYN,COMLEN,NSITV,ATNAM, &
             IZMAT,ZMAT,IUNIT,OUTUN)
        CALL ZCONSTR(.TRUE.,1,NSITV,IZMAT,ZMAT,X,Y,Z)
     ELSE
        GOTO 2000
     ENDIF
  ENDIF

  IF(CHECQUE(COMLYN,'SOLUTE'))THEN
     IU=1
     IF(DOTRIM(COMLYN)) IU=NEXTI(COMLYN,COMLEN)
     IOFF=DSITV+(IU-1)*DSITU+1
     WRITE(WRD,'(I1)') IU
     !
     !     Set default names for the solute
     !
     SEGMID(IOFF)='SOL'//WRD
     RESNAM(IOFF)='SOL'//WRD
     CALL RDSTRU2(COMLYN,COMLEN,IUNIT,OUTUN,NBASE,BASE, &
          ATNAM(IOFF),RESNAM(IOFF),SEGMID(IOFF), &
          NSITU(IU),DSITU,NTYPU(IU),ITYPE(IOFF),ICHEM(IOFF), &
          X(IOFF),Y(IOFF),Z(IOFF))
2030 CONTINUE
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.TRUE., &
          'STRUC> ')
     IF(EOF) GOTO 9000
     IF(COMLEN.LE.0) GOTO 2030
     IF(CHECQUE(COMLYN,'ZMAT'))THEN
        CALL RDZMAT(COMLYN,COMLEN,NSITU(IU),ATNAM(IOFF), &
             IZMAT(1,1,IU+1),ZMAT(1,1,IU+1),IUNIT,OUTUN)
        CALL ZCONSTR(.TRUE.,1,NSITU(IU),IZMAT(1,1,IU+1), &
             ZMAT(1,1,IU+1),X(IOFF),Y(IOFF),Z(IOFF))
     ELSEIF(CHECQUE(COMLYN,'2ZMA'))THEN
        CALL RDZMAT(COMLYN,COMLEN,NSITU(IU),ATNAM(IOFF), &
             IZMAT(1,1,IU+1),ZMAT(1,1,IU+1),IUNIT,OUTUN)
        CALL ZCONSTR(.TRUE.,1,NSITU(IU),IZMAT(1,1,IU+1), &
             ZMAT(1,1,IU+1),X2(IOFF),Y2(IOFF),Z2(IOFF))
     ELSE
        GOTO 2000
     ENDIF
  ENDIF

  IF(CHECQUE(COMLYN,'END'))THEN
     CALL GENERZ(OUTUN,NTYPV,NTYPU)
     RETURN
  ENDIF

  GOTO 1000
  !     
9000 CONTINUE
  WRITE(OUTUN,'(A)') ' RISM:RDSTRUC> ERROR: End-of-File reached.'
  RETURN
END SUBROUTINE RDSTRUC

SUBROUTINE RDSTRU2(COMLYN,COMLEN,IUNIT,OUTUN,NBASE,BASE, &
     ATNAM,RESNAM,SEGID,NSITE,NMAX,NTYPE,ITYPE,ICHEM, &
     X,Y,Z)
  !-----------------------------------------------------------------------
  !     This subroutine reads the structure file to pass on the
  !     non-bonded parameters eps, sig.
  !     It can also read the atomic coordinates, if they are appended
  !     after the [SEGID name] and [RESNAM name] definitions.
  use dimens_fcm
  use exfunc
  use rism
  use chm_kinds
  use string
  implicit none
  !
  !     Command parser and input unit
  CHARACTER(len=*) COMLYN
  INTEGER       COMLEN,IUNIT,OUTUN
  !
  !     Pointer array for the parameters table
  INTEGER NBASE
  CHARACTER(len=*) BASE(*)
  CHARACTER(len=*) ATNAM(DSITE),RESNAM(DSITE),SEGID(DSITE)
  INTEGER NSITE,NMAX,NTYPE,ITYPE(DSITE),ICHEM(DSITE)
  real(chm_real) X(*),Y(*),Z(*)
  !
  !     Local variables
  !     ---------------
  CHARACTER(len=8) WRD
  CHARACTER(len=20) CHN
  CHARACTER(len=25) WRD25
  INTEGER I,WDLEN
  LOGICAL EOF
  !
  EOF = .FALSE.

1000 CONTINUE
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.TRUE., &
       'STRUC> ')
  IF(EOF) GOTO 9000
  IF(COMLEN.LE.0) GOTO 1000
  NSITE=GTRMI(COMLYN,COMLEN,'NSITE',0)
  IF(NSITE.GT.NMAX)THEN
     OUTUN=ABS(OUTUN)
     WRITE(OUTUN,115) NSITE,NMAX
     CALL WRNDIE(-5,'<RDSTRUC>','JOB ABORTED')
     RETURN
  ENDIF
115 FORMAT(' * RDSTRUC * error, nsite=',I2, &
       ' greater than limit ',I2,/)

  NTYPE=GTRMI(COMLYN,COMLEN,'NTYPE',NSITE)
  !
  loop200:DO I=1,NSITE
2000 CONTINUE
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.TRUE., &
          'STRUC> ')
     IF(EOF) GOTO 9000
     IF(COMLEN.LE.0) GOTO 2000
     CALL GTRMWA(COMLYN,COMLEN,'SEGI',4,WRD,8,WDLEN)
     IF(WDLEN.GT.0 .AND. DOTRIM(WRD))THEN
        SEGID(I)=WRD
     ELSE
        !     Keep the previous segid
        SEGID(I)=SEGID(MAX(I-1,1)) 
     ENDIF
     CALL GTRMWA(COMLYN,COMLEN,'RESN',4,WRD,8,WDLEN)
     IF(WDLEN.GT.0 .AND. DOTRIM(WRD))THEN
        RESNAM(I)=WRD
     ELSE
        !     Keep the previous resnam
        RESNAM(I)=RESNAM(MAX(I-1,1)) 
     ENDIF
     ATNAM(I)=NEXTA8(COMLYN,COMLEN)
     !     Find the atom type
     WRD=NEXTA8(COMLYN,COMLEN)
     CALL FNBASE(BASE,NBASE,WRD,ICHEM(I))
     ITYPE(I)=GTRMI(COMLYN,COMLEN,'TYPE',I)
     !
     !     If after the SEGID and RESNAM definitions there exist the
     !     atomic coordinates, store them in the coordinate arrays
     IF(DOTRIM(COMLYN))THEN
        X(I)=NEXTF(COMLYN,COMLEN)
        Y(I)=NEXTF(COMLYN,COMLEN)
        Z(I)=NEXTF(COMLYN,COMLEN)
     ENDIF
  enddo loop200
  !
  RETURN
  !
9000 CONTINUE
  WRITE(OUTUN,'(A)') ' RISM:RDSTRU2> ERROR: End-of-File reached.'
  RETURN
END SUBROUTINE RDSTRU2

SUBROUTINE GENERZ(OUTUN,NTYPV,NTYPU)
  !-----------------------------------------------------------------------
  !     After the parameters and the structure information have been read,
  !     this subroutine generates the various pointers and calculates
  !     the vdw coefficients A,B and the coulombic coefficient C.
  use number
  use rism
  use distri
  use struc
  use chm_kinds
  implicit none
  INTEGER OUTUN,NTYPV,NTYPU(*)
  !
  !     Local variables
  real(chm_real) SIGIJ,EPSIJ,QI,QJ,QIJ
  INTEGER I,J,IOFF,IU,JU

  !     setup the fixed pointers (determined by the rism.f90 file)
  !     pvv=1 by default
  !
  PUV(1)=1+DPRVV
  PUU(1)=PUV(1)+DPRUV
  PUV(2)=PUU(1)+DPRUU
  PUU(2)=PUV(2)+DPRUV
  PUU(3)=PUU(2)+DPRUU

  !     SOLVENT
  !     -------
  IF(NSITV.GT.0) THEN
     IF(OUTUN.GT.0) WRITE(OUTUN,115)
     !
     !     Intermolecular
     NPVV=(NTYPV*(NTYPV+1))/2
     !
     !     Intramolecular
     NPV=(NSITV*(NSITV+1))/2
     DO I=1,NSITV
        DO J=1,I
           INTV(I,J)=((I-1)*I)/2+J
           INTV(J,I)=INTV(I,J)
           INTVV(I,J)=((ITYPE(I)-1)*(ITYPE(I)))/2+ITYPE(J)
           INTVV(J,I)=INTVV(I,J)
           PRNAM(INTVV(I,J))=ATNAM(J)//' '//ATNAM(I)
           EPSIJ=EPSIL(INBASE(ICHEM(I),ICHEM(J)))
           SIGIJ=SIG(INBASE(ICHEM(I),ICHEM(J)))
           QI=CHARG(ICHEM(I))
           QJ=CHARG(ICHEM(J))
           QIJ=QI*QJ
           IOFF=INTVV(I,J)
           A(IOFF)=FOUR*EPSIJ*SIGIJ**12
           B(IOFF)=-FOUR*EPSIJ*SIGIJ**6
           C(IOFF)=COEFF*QIJ
           IF(OUTUN.GT.0) WRITE(OUTUN,125) PRNAM(IOFF), &
                EPSIJ,SIGIJ,QJ,QI,INTVV(I,J)
        ENDDO
     ENDDO
  ENDIF
  !
115 FORMAT(/,' Non-Bonded parameters',/ &
       ' atom  pair    eps(i,j)  sig(i,j)    Q(i)      Q(j)    index', &
       //,' solvent-solvent:')
125 FORMAT(1X,A,4F10.4,I5)

  !     SOLUTE #1
  !     ---------
  IF(NSITU(1).GT.0)THEN
     IF(OUTUN.GT.0) WRITE(OUTUN,'(/,A)') ' solute(1)-solvent:'
     !
     !     Intermolecular U1-V
     NPUV(1)=NTYPU(1)*NTYPV
     !     Intramolecular
     NPU(1)=(NSITU(1)*(NSITU(1)+1))/2
     DO I=1,NSITU(1)
        DO J=1,NSITV
           IU=I+DSITV
           INTUV(I,J,1)=NTYPV*(ITYPE(IU)-1)+ITYPE(J)
           IOFF=PUV(1)-1+INTUV(I,J,1)
           PRNAM(IOFF)=ATNAM(IU)//' '//ATNAM(J)
           EPSIJ=EPSIL(INBASE(ICHEM(IU),ICHEM(J)))
           SIGIJ=SIG(INBASE(ICHEM(IU),ICHEM(J)))
           QI=CHARG(ICHEM(IU))
           QJ=CHARG(ICHEM(J))
           QIJ=QI*QJ
           A(IOFF)=4.0*EPSIJ*SIGIJ**12
           B(IOFF)=-4.0*EPSIJ*SIGIJ**6
           C(IOFF)=COEFF*QIJ
           IF(OUTUN.GT.0) WRITE(OUTUN,125) PRNAM(IOFF), &
                EPSIJ,SIGIJ,QI,QJ,INTUV(I,J,1)
        ENDDO
     ENDDO

     IF(OUTUN.GT.0) WRITE(OUTUN,'(/,A)') ' solute(1)-solute(1):'
     !
     !     U1-U1
     NPUU(1)=(NTYPU(1)*(NTYPU(1)+1))/2
     DO I=1,NSITU(1)
        DO J=1,I
           INTU(I,J,1)=((I-1)*I)/2+J
           INTU(J,I,1)=INTU(I,J,1)
           IU=I+DSITV
           JU=J+DSITV
           INTUU(I,J,1)=((ITYPE(IU)-1)*(ITYPE(IU)))/2+ITYPE(JU)
           INTUU(J,I,1)=INTUU(I,J,1)
           IOFF=PUU(1)-1+INTUU(I,J,1)
           PRNAM(IOFF)=ATNAM(JU)//' '//ATNAM(IU)
           EPSIJ=EPSIL(INBASE(ICHEM(IU),ICHEM(JU)))
           SIGIJ=SIG(INBASE(ICHEM(IU),ICHEM(JU)))
           QI=CHARG(ICHEM(IU))
           QJ=CHARG(ICHEM(JU))
           QIJ=QI*QJ
           A(IOFF)=FOUR*EPSIJ*SIGIJ**12
           B(IOFF)=-FOUR*EPSIJ*SIGIJ**6
           C(IOFF)=COEFF*QIJ
           IF(OUTUN.GT.0) WRITE(OUTUN,125) PRNAM(IOFF),EPSIJ, &
                SIGIJ,QJ,QI,INTUU(I,J,1)
        ENDDO
     ENDDO
  ENDIF

  !     solute #2
  !     ---------
  IF(NSITU(2).GT.0)THEN
     IF(OUTUN.GT.0) WRITE(OUTUN,'(/,A)') ' solute(2)-solvent:'
     !
     !     Intermolecular U2-V
     NPUV(2)=NTYPU(2)*NTYPV
     !
     !     Intramolecular
     NPU(2)=(NSITU(2)*(NSITU(2)+1))/2
     DO I=1,NSITU(2)
        DO J=1,NSITV
           IU=I+DSITV+DSITU
           INTUV(I,J,2)=NTYPV*(ITYPE(IU)-1)+ITYPE(J)
           IOFF=PUV(2)-1+INTUV(I,J,2)
           PRNAM(IOFF)=ATNAM(IU)//' '//ATNAM(J)
           EPSIJ=EPSIL(INBASE(ICHEM(IU),ICHEM(J)))
           SIGIJ=SIG(INBASE(ICHEM(IU),ICHEM(J)))
           QI=CHARG(ICHEM(IU))
           QJ=CHARG(ICHEM(J))
           QIJ=QI*QJ
           A(IOFF)=FOUR*EPSIJ*SIGIJ**12
           B(IOFF)=-FOUR*EPSIJ*SIGIJ**6
           C(IOFF)=COEFF*QIJ
           IF(OUTUN.GT.0) WRITE(OUTUN,125) PRNAM(IOFF),EPSIJ, &
                SIGIJ,QI,QJ,INTUV(I,J,2)
        ENDDO
     ENDDO

     IF(OUTUN.GT.0) WRITE(OUTUN,'(/,A)') ' solute(2)-solute(1):'
     !
     !     Intermolecular U2-U1
     NPUU(2)=NTYPU(1)*NTYPU(2)
     DO I=1,NSITU(2)
        DO J=1,NSITU(1)
           IU=I+DSITV+DSITU
           JU=J+DSITV
           INTUU(I,J,2)=NTYPU(2)*(ITYPE(IU)-1)+ITYPE(JU)
           IOFF=PUU(2)-1+INTUU(I,J,2)
           PRNAM(IOFF)=ATNAM(IU)//' '//ATNAM(JU)
           EPSIJ=EPSIL(INBASE(ICHEM(IU),ICHEM(JU)))
           SIGIJ=SIG(INBASE(ICHEM(IU),ICHEM(JU)))
           QI=CHARG(ICHEM(IU))
           QJ=CHARG(ICHEM(JU))
           QIJ=QI*QJ
           A(IOFF)=FOUR*EPSIJ*SIGIJ**12
           B(IOFF)=-FOUR*EPSIJ*SIGIJ**6
           C(IOFF)=COEFF*QIJ
           IF(OUTUN.GT.0) WRITE(OUTUN,125) PRNAM(IOFF),EPSIJ, &
                SIGIJ,QI,QJ,INTUU(I,J,2)
        ENDDO
     ENDDO

     IF(OUTUN.GT.0) WRITE(OUTUN,'(/,A)') ' SOLUTE(2)-SOLUTE(2):'
     !
     !     INTERMOLECULAR U2-U2
     NPUU(3)=(NTYPU(2)*(NTYPU(2)+1))/2
     DO I=1,NSITU(2)
        DO J=1,I
           INTU(I,J,2)=((I-1)*I)/2+J
           INTU(J,I,2)=INTU(I,J,2)
           IU=I+DSITV+DSITU
           JU=J+DSITV+DSITU
           INTUU(I,J,3)=((ITYPE(IU)-1)*(ITYPE(IU)))/2+ITYPE(JU)
           INTUU(J,I,3)=INTUU(I,J,3)
           IOFF=PUU(3)-1+INTUU(I,J,3)
           PRNAM(IOFF)=ATNAM(JU)//' '//ATNAM(IU)
           EPSIJ=EPSIL(INBASE(ICHEM(IU),ICHEM(JU)))
           SIGIJ=SIG(INBASE(ICHEM(IU),ICHEM(JU)))
           QI=CHARG(ICHEM(IU))
           QJ=CHARG(ICHEM(JU))
           QIJ=QI*QJ
           A(IOFF)=FOUR*EPSIJ*SIGIJ**12
           B(IOFF)=-FOUR*EPSIJ*SIGIJ**6
           C(IOFF)=COEFF*QIJ
           IF(OUTUN.GT.0) WRITE(OUTUN,125) PRNAM(IOFF),EPSIJ, &
                SIGIJ,QJ,QI,INTUU(I,J,3)
        ENDDO
     ENDDO
  ENDIF

  IF(OUTUN.GT.0) WRITE(OUTUN,*)
  !
  RETURN
END SUBROUTINE GENERZ

SUBROUTINE SEGMENT(NSITE,SEGID,ASEGM,NSEGM,ISEGM)
  !-----------------------------------------------------------------------
  !     This subroutine makes the list of all the segments in the system.
  use chm_kinds
  implicit none
  INTEGER NSITE,NSEGM,ISEGM(*)
  CHARACTER(len=*) SEGID(*),ASEGM(*)
  !
  ! local
  INTEGER ISTART,I

  NSEGM=1
  ISTART=1
  ASEGM(1)=SEGID(1)

1000 CONTINUE
  IF(ISTART.LE.NSITE)THEN
     ASEGM(NSEGM)=SEGID(ISTART)
     ISEGM(NSEGM)=0
     DO I=ISTART,NSITE
        IF(ASEGM(NSEGM).EQ.SEGID(I))THEN
           ISEGM(NSEGM)=ISEGM(NSEGM)+1
        ELSE
           NSEGM=NSEGM+1
           ISTART=I
           GOTO 1000
        ENDIF
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE SEGMENT

SUBROUTINE RDZMAT(COMLYN,COMLEN,NATOM,ATNAM, &
     IZMAT,ZMAT,IUNIT,OUTUN)
  !-----------------------------------------------------------------------
  !     read the ZMATRIX
  use dimens_fcm
  use exfunc
  use rism
  use chm_kinds
  use string
  implicit none
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN,NATOM,IZMAT(4,*),IUNIT,OUTUN
  real(chm_real)    ZMAT(3,*)
  CHARACTER(len=*) ATNAM(DSITE)
  !
  ! local
  CHARACTER(len=25) WRD25
  CHARACTER(len=6) WRD1
  INTEGER I
  LOGICAL EOF
  !
  EOF = .FALSE.
  !
  IF(NATOM.EQ.0) RETURN
  !
  !     FIRST LINE <AT1>
1020 CONTINUE
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.TRUE., &
       'ZMATR> ')
  IF(EOF) GOTO 9000
  IF(COMLEN.LE.0) GOTO 1020
  WRD1=NEXTA6(COMLYN,COMLEN)
  CALL FNBASE(ATNAM,NATOM,WRD1,IZMAT(1,1))
  IF(NATOM.EQ.1) RETURN
  !
  !     second line <at2>  <at1>  bond
1040 CONTINUE
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.TRUE., &
       'ZMATR> ')
  IF(EOF) GOTO 9000
  IF(COMLEN.LE.0) GOTO 1040
  WRD1=NEXTA6(COMLYN,COMLEN)
  CALL FNBASE(ATNAM,NATOM,WRD1,IZMAT(2,2))
  WRD1=NEXTA6(COMLYN,COMLEN)
  CALL FNBASE(ATNAM,NATOM,WRD1,IZMAT(1,2))
  ZMAT(1,2)=NEXTF(COMLYN,COMLEN)
  IF(NATOM.EQ.2) RETURN
  !
  !     third line <at3>  <at2>  bond  <at1> theta
1060 CONTINUE
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.TRUE., &
       'ZMATR> ')
  IF(EOF) GOTO 9000
  IF(COMLEN.LE.0) GOTO 1060
  WRD1=NEXTA6(COMLYN,COMLEN)
  CALL FNBASE(ATNAM,NATOM,WRD1,IZMAT(3,3))
  WRD1=NEXTA6(COMLYN,COMLEN)
  CALL FNBASE(ATNAM,NATOM,WRD1,IZMAT(2,3))
  ZMAT(1,3)=NEXTF(COMLYN,COMLEN)
  WRD1=NEXTA6(COMLYN,COMLEN)
  CALL FNBASE(ATNAM,NATOM,WRD1,IZMAT(1,3))
  ZMAT(2,3)=NEXTF(COMLYN,COMLEN)
  IF(NATOM.EQ.3) RETURN
  !
  !     rest of the Zmatrix <at4> <at3> bond <at2> theta <at1> phi
  DO I=4,NATOM
1080 CONTINUE
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.TRUE., &
          'ZMATR> ')
     IF(EOF) GOTO 9000
     IF(COMLEN.LE.0) GOTO 1080
     WRD1=NEXTA6(COMLYN,COMLEN)
     CALL FNBASE(ATNAM,NATOM,WRD1,IZMAT(4,I))
     WRD1=NEXTA6(COMLYN,COMLEN)
     CALL FNBASE(ATNAM,NATOM,WRD1,IZMAT(3,I))
     ZMAT(1,I)=NEXTF(COMLYN,COMLEN)
     WRD1=NEXTA6(COMLYN,COMLEN)
     CALL FNBASE(ATNAM,NATOM,WRD1,IZMAT(2,I))
     ZMAT(2,I)=NEXTF(COMLYN,COMLEN)
     WRD1=NEXTA6(COMLYN,COMLEN)
     CALL FNBASE(ATNAM,NATOM,WRD1,IZMAT(1,I))
     ZMAT(3,I)=NEXTF(COMLYN,COMLEN)
  ENDDO
  RETURN
  !     
9000 CONTINUE
  WRITE(OUTUN,'(A)') ' RISM:RDSTRU2> ERROR: End-of-File reached.'
  RETURN
END SUBROUTINE RDZMAT

#endif /* (rism_main)*/

