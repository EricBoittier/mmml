module dimbsubs

contains

#if KEY_DIMB==1 /*dimb_main*/
  SUBROUTINE CORARR(ATMPAD,NPARD,ATMCOR,NATOM)
    !-----------------------------------------------------------------------
    !     01-Mar-1994 David Perahia
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     This subroutine defines a correspondence array
    !     ATMCOR between atom numbers and indices of
    !     the blocks to which they belong. 
    !-----------------------------------------------------------------------
  use chm_kinds
    implicit none
    ! Passed variables
    INTEGER ATMPAD(2,*),NPARD,ATMCOR(*),NATOM
    ! Local variables
    INTEGER I,IPAR1,IS1,IS2
    !
    ! Begin
    DO IPAR1=1,NPARD
       IS1=ATMPAD(1,IPAR1)
       IS2=ATMPAD(2,IPAR1)
       DO I=IS1,IS2
          ATMCOR(I)=IPAR1
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE CORARR


  SUBROUTINE INIPAF(ATMPAF,NPARD)
    !-----------------------------------------------------------------------
    !     01-Mar-1994 David Perahia
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     Resets upper diagonal values of ATMPAF to zero.
    !-----------------------------------------------------------------------
  use chm_kinds
    implicit none
    ! Passed variables
    INTEGER NPARD,ATMPAF(NPARD,NPARD)
    ! Local variables
    INTEGER I,J

    ! Begin
    DO I=1,NPARD-1
       DO J=I+1,NPARD
          ATMPAF(I,J)=0
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE INIPAF

  !
  SUBROUTINE IPART(SUBLIS,II,IPAR1,IPAR2,ATMPAF,NPARD,QCALC)
    !-----------------------------------------------------------------------
    !     01-Mar-1994 David Perahia
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     Increases upper diagonal value of ATMPAF corresponding to
    !     sub-blocks IPAR and IPAR2 by one. Next, it checks whether
    !     it is equal to the corresponding lower diagonal value.
    !     If it is, QCALC is set to TRUE, and ATMPAF upper diagonal
    !     is reset to zero. Since the size of the lower diagonal
    !     value of ATMPAF is inversely related to the interaction
    !     strength of the corresponding sub-blocks, this system ensures
    !     that strongly interacting sub-blocks are used most in the 
    !     mixed-basis calculations.
    !-----------------------------------------------------------------------
  use chm_kinds
    implicit none
    ! Passed variables
    INTEGER II,IPAR1,IPAR2,SUBLIS(2,*)
    INTEGER NPARD,ATMPAF(NPARD,NPARD)
    LOGICAL QCALC

    ! Begin
    IPAR1=SUBLIS(1,II)
    IPAR2=SUBLIS(2,II)
    ATMPAF(IPAR1,IPAR2)=ATMPAF(IPAR1,IPAR2)+1
    IF(ATMPAF(IPAR1,IPAR2) /= ATMPAF(IPAR2,IPAR1)) THEN
       QCALC=.FALSE.
    ELSE
       QCALC=.TRUE.
       ATMPAF(IPAR1,IPAR2)=0
    ENDIF

    RETURN
  END SUBROUTINE IPART

  !
  SUBROUTINE PARINT(ATMPAF,NPARD,SUBLIS,NSUBP)
    !-----------------------------------------------------------------------
    !     01-Mar-1994 David Perahia
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     Provides the list  of interacting sub-blocks
    !     NSUBP = input == number of interacting sub-blocks
    !     SUBLIS contains the numbering of interacting sub-blocks
    !     ordered in such a way that  a given sub-block is
    !     distributed uniformly.
    !-----------------------------------------------------------------------
  use chm_kinds
    implicit none
    ! Passed variables
    INTEGER NPARD,ATMPAF(NPARD,NPARD)
    INTEGER SUBLIS(2,*),NSUBP
    ! Local variables
    INTEGER IPAR1,IPAR2,ISUBL,IPASSA,NPASSA,IFPAR,IDIF,IPAO

    ! Begin
    ISUBL=0
    IPASSA=1
    NPASSA=2
    IFPAR=1
    IDIF=1
    IPAO=1

100 IPAR1=IFPAR
    IPAR2=IPAR1+IDIF
    IF(IPAR2 <= NPARD) THEN
       IF(ATMPAF(IPAR1,IPAR2) >= 1) THEN
          ISUBL=ISUBL+1
          SUBLIS(1,ISUBL)=IPAR1
          SUBLIS(2,ISUBL)=IPAR2
       ENDIF
       IF((IPAR1 == 1).AND.(IPAR2.EQ.NPARD)) GOTO 200
       IFPAR=IPAR2+1
       GOTO 100
    ENDIF

    IPASSA=IPASSA+1
    IF(IPASSA <= NPASSA) THEN
       IPAO=IPAO+1
       IFPAR=IPAO
       GOTO 100
    ENDIF

    IDIF=IDIF+1
    IPASSA=1
    NPASSA=NPASSA+1
    IFPAR=1
    IPAO=1
    GOTO 100

200 CONTINUE

    RETURN
  END SUBROUTINE PARINT

  !
  SUBROUTINE PARLIS(ATMCOR,ATMPAF,INBCMP,JNBCMP,NPARD,NSUBP,NATOM, &
       X,Y,Z,NBOND,IB,JB,DD1,DDVAL,DDVALM)
    !-----------------------------------------------------------------------
    !     01-Mar-1994 David Perahia
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     This routine calculates the interacting blocks
    !     ATMPAF(I,J)=1 => interactions between blocks I and J >  DDVALM
    !     ATMPAF(I,J)=0 => interactions between blocks I and J <= DDVALM
    !     DDVALM (default=0.0) is the STREngth parameter from the input file
    !-----------------------------------------------------------------------
  use chm_kinds
  use stream
  use number
    implicit none
    ! Passed variables
    INTEGER INBCMP(*),JNBCMP(*),NPARD,NSUBP
    INTEGER ATMPAF(NPARD,NPARD),ATMCOR(*),NATOM
    INTEGER NBOND,IB(*),JB(*)
    real(chm_real) X(*),Y(*),Z(*),DD1(*),DDVAL(NPARD,NPARD),DDVALM
    ! Local variables
    INTEGER I,J,II,JJ,IJADD,JPR,MM,IL,IH,NBJ,NBR
    real(chm_real) VMAX,DDVALN

    ! Begin
    VMAX=RSMALL
    atmpaf = 0
    ddval = zero
    IJADD=0
    NBJ=0
    DO II=1,NATOM-1
       NBR=INBCMP(II)
       IJADD=IJADD+6
       IF(NBR > NBJ) THEN
          DO JPR=NBJ+1,NBR
             JJ=JNBCMP(JPR)
             DDVAL(ATMCOR(II),ATMCOR(JJ))=DDVAL(ATMCOR(II),ATMCOR(JJ))+ &
                  ABS(DD1(IJADD+1))+ABS(DD1(IJADD+2))+ABS(DD1(IJADD+3))+ &
                  ABS(DD1(IJADD+4))+ABS(DD1(IJADD+5))+ABS(DD1(IJADD+6))+ &
                  ABS(DD1(IJADD+7))+ABS(DD1(IJADD+8))+ABS(DD1(IJADD+9))
             IJADD=IJADD+9
          ENDDO
          NBJ=NBR
       ENDIF
    ENDDO
    !
    ! Select sub-block pairs having coupling strengths
    ! larger than DDVALM
    !
    DO I=1,NPARD-1
       DO J=I+1,NPARD
          IF(DDVAL(I,J) > DDVALM) ATMPAF(I,J)=1
       ENDDO
    ENDDO
    !
    ! Find the maximum of DDVAL
    !
    DO I=1,NPARD-1
       DO J=I+1,NPARD
          IF(DDVAL(I,J) > VMAX) VMAX=DDVAL(I,J)
       ENDDO
    ENDDO
    !
    ! Fill ATMPAF matrix with measures of interaction strengths: a value of 1
    ! indicates large second derivatives in that sub-block, a value of 9999 
    ! indicates very small second derivatives.
    !
    DO I=1,NPARD-1
       DO J=I+1,NPARD
          IF(DDVAL(I,J) <= PTONE) THEN
             ATMPAF(J,I)=9999.0
          ELSE
             ATMPAF(J,I)=VMAX/DDVAL(I,J)+0.5
          ENDIF
       ENDDO
    ENDDO
    !
    ! Finds sub-block pairs linked by a chemical bond
    ! For these sub-blocks, ATMPAF is automatically set to 1.
    !
    DO MM=1,NBOND
       II=IB(MM)
       JJ=JB(MM)
       IF(II < JJ) THEN
          IL=II
          IH=JJ
       ELSE
          IL=JJ
          IH=II
       ENDIF
       IF(ATMPAF(ATMCOR(IL),ATMCOR(IH)) == 0) &
            ATMPAF(ATMCOR(IL),ATMCOR(IH))=1
       ATMPAF(ATMCOR(IH),ATMCOR(IL))=1
    ENDDO

    DDVALN=VMAX/TEN

    IF(PRNLEV >= 2) WRITE(OUTU,105)
105 FORMAT(/' PARLIS: The frequently called interacting ', &
         'sub-blocks are:'/ &
         5X,'Sub-block pair  sumABS(DD1)')

    NSUBP=0
    DO I=1,NPARD-1
       DO J=I+1,NPARD
          IF(ATMPAF(I,J) /= 0) THEN
             NSUBP=NSUBP+1
             IF (DDVAL(I,J) > DDVALN) WRITE(OUTU,107) I,J,DDVAL(I,J)
          ENDIF
       ENDDO
    ENDDO
107 FORMAT(5X,'(',I5,',',I5,')',1X,F12.2)

    IF(PRNLEV >= 2) WRITE(OUTU,113) NSUBP
113 FORMAT(' PARLIS: There are ',I5,' interacting sub-blocks')

    RETURN
  END SUBROUTINE PARLIS

  !
  SUBROUTINE PARTDS(NAT3,NPARC,ATMPAR,NPARS,ATMPAS,INIDS,NPARMX, &
       DDF,NFREG,CUTF1,PARDIM,NFCUT1)
    !-----------------------------------------------------------------------
    !     03-Mar-1993 David Perahia
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     Displace the partition toward the right
    !     ATMPAR(1,*),ATMPAR(2,*): First and last atom numberings
    !                              corresponding to a partitioned block
    !     INIDS=0:  Restore the original partitioning
    !     INIDS=1:  Partitioning by displacing the window to the right
    !               by half the width of the windows.
    !     This is done in single window mode only.
    !-----------------------------------------------------------------------
  use chm_kinds
  use stream
    implicit none
    ! Passed variables
    INTEGER NAT3,NPARMX,PARDIM,INIDS,NFREG,NFCUT1
    INTEGER ATMPAR(2,NPARMX),NPARC
    INTEGER ATMPAS(2,NPARMX),NPARS
    real(chm_real) CUTF1,DDF(*)
    ! Local variables
    INTEGER I,J,ID1,ID2,IP
    INTEGER MXV,PARDIMC,ATDIFF
    LOGICAL QACC

    ! Begin
    IF(INIDS == 0) THEN
       NPARC=NPARS
       DO I=1,NPARC
          ATMPAR(1,I)=ATMPAS(1,I)
          ATMPAR(2,I)=ATMPAS(2,I)
       ENDDO
    ELSEIF(INIDS == 1) THEN
       NPARC=NPARS-1
       ID1=(ATMPAR(2,1)-ATMPAR(1,1)+1)/2
       IP=ID1
       DO I=1,NPARC
          ATMPAR(1,I)=IP+1
          ID2=(ATMPAR(2,I+1)-ATMPAR(1,I+1)+1)/2
          ATMPAR(2,I)=IP+ID1+ID2
          IP=IP+ID1+ID2
          ID1=ID2
       ENDDO
       IF(PRNLEV >= 2) THEN
          WRITE(OUTU,105)
          WRITE(OUTU,107) ((ATMPAR(J,I),J=1,2),I=1,NPARC)
       ENDIF
105    FORMAT(/' PARTDS: The atom numberings corresponding to', &
            ' partitioned blocks are:')
107    FORMAT(8X,I5,1X,I5,10X,I5,1X,I5,10X,I5,1X,I5)

       !      ELSEIF(INIDS == 2) THEN
       !
       !      NPARC=NPARC-1
       !      NMODI=NFREG-NAT3/NPARC
       !      IF(DDF(NMODI) >= CUTF1)
       !     1  CALL PARTII(NAT3,NPARMX,NPARC,ATMPAR,PARDIM,QACC,MXV)
       !      IF(.NOT.QACC .OR. DDF(NMODI) < CUTF1) THEN
       !         NPARC=NPARC+2
       !         CALL PARTII(NAT3,NPARMX,NPARC,ATMPAR,PARDIM,QACC,MXV)
       !      ENDIF
       !      NFCUT1=NFREG-3*MXV

    ENDIF

    RETURN
  END SUBROUTINE PARTDS

  !
  SUBROUTINE PARTIC(NAT3,NFREG,NFCUT,NPARMX,NPARC,ATMPAR, &
       NFRRES,PARDIM)
    !-----------------------------------------------------------------------
    !     12-Feb-1993 David Perahia
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     Partition of the molecule into parts for the cartesian
    !     part of DIMB
    !     NPARMX: Maximum number of parts
    !     ATMPAR(1,*),ATMPAR(2,*): First and last atom numberings
    !                              corresponding to a partitioned block
    !-----------------------------------------------------------------------
  use chm_kinds
  use stream
    implicit none
    ! Passed variables
    INTEGER NAT3,NFREG,NFCUT,NPARMX,NPARC
    INTEGER ATMPAR(2,NPARMX)
    INTEGER PARDIM,NFRRES
    ! Local variables
    INTEGER I,J,NATOM,NATC,MXV,NT,NTD,INC,IATM,IAD,IBAS
    real(chm_real) RNATC,RNPARC,RNPAR2

    ! Begin
    NATOM=NAT3/3
    RNATC=(PARDIM-NFRRES)/3.
    NATC=(PARDIM-NFRRES)/3
    RNPARC=NATOM/RNATC
    NPARC=NATOM/NATC 
    RNPAR2=NPARC

    IF(RNPARC > RNPAR2) NPARC=NPARC+1
    IATM=NATOM/NPARC
    IF((NATOM-IATM*NPARC) > 0) NPARC=NPARC+1
    IF(NPARC <= 1) CALL WRNDIE(-3,'<PARTIC>', &
         'Molecule divided in one part, use DIAG ')

    IATM=NATOM/NPARC
    NT=IATM*NPARC
    NTD=NATOM-NT
    IBAS=0
    IAD=0
    DO I=1,NPARC
       IAD=IAD+1
       INC=0
       IF(IAD <= NTD) INC=1
       ATMPAR(1,I)=IBAS+1
       ATMPAR(2,I)=IBAS+IATM+INC
       IBAS=IBAS+IATM+INC
    ENDDO
    IF(ATMPAR(2,NPARC) < NATOM) ATMPAR(2,NPARC)=NATOM
    IF(NPARC > NPARMX) CALL WRNDIE(-3,'<PARTIC>', &
         'The number of parts exceeded NPARMX')
    IF(PRNLEV >= 2) THEN
       WRITE(OUTU,105)
       WRITE(OUTU,107) ((ATMPAR(J,I),J=1,2),I=1,NPARC)
    ENDIF
105 FORMAT(/' PARTIC: The atom numberings corresponding to', &
         ' partitioned blocks are:')
107 FORMAT(8X,I5,1X,I5,10X,I5,1X,I5,10X,I5,1X,I5)

    !
    ! Find the maximum block
    !
    MXV=0
    DO I=1,NPARC
       IF((ATMPAR(2,I)-ATMPAR(1,I)+1) > MXV) &
            MXV=(ATMPAR(2,I)-ATMPAR(1,I)+1)
    ENDDO
    IF(PRNLEV >= 2) WRITE(OUTU,113) MXV
113 FORMAT(' PARTIC: The largest diagonal block spans ',I5,' atoms')

    RETURN
  END SUBROUTINE PARTIC

  !
  SUBROUTINE PARTID(NPARC,ATMPAR,NPARD,ATMPAD,NPARMX)
    !-----------------------------------------------------------------------
    !     17-Feb-1994 David Perahia
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     Doubles the partition 
    !     ATMPAR(1,*),ATMPAR(2,*): First and last atom numberings
    !                              corresponding to a partitioned block
    !     ATMPAD(1,*),ATMPAD(2,*): First and last atom numberings
    !                              corresponding to doubled partitions
    !     This routine is called only in the double window mode.
    !-----------------------------------------------------------------------
  use chm_kinds
  use stream
    implicit none
    ! Passed variables
    INTEGER NPARMX
    INTEGER ATMPAR(2,NPARMX),NPARC
    INTEGER ATMPAD(2,NPARMX),NPARD
    ! Local variables
    INTEGER I,J

    ! Begin
    NPARD=NPARC*2
    J=1
    DO I=1,NPARC
       ATMPAD(1,J)=ATMPAR(1,I)
       ATMPAD(2,J)=(ATMPAR(2,I)-ATMPAR(1,I)+1)/2+ATMPAR(1,I)-1
       ATMPAD(1,J+1)=ATMPAD(2,J)+1
       ATMPAD(2,J+1)=ATMPAR(2,I)
       J=J+2
       IF(J+1 > NPARMX) CALL WRNDIE(-3,'<PARTID>', &
            'The number of parts exceeded NPARMX')
    ENDDO

    IF(PRNLEV >= 2) THEN
       WRITE(OUTU,105)
       WRITE(OUTU,107) ((ATMPAD(J,I),J=1,2),I=1,NPARD)
    ENDIF
105 FORMAT(/' PARTID: The atom numberings corresponding to', &
         ' subblocks are:')
107 FORMAT(8X,I5,1X,I5,10X,I5,1X,I5,10X,I5,1X,I5)

    RETURN
  END SUBROUTINE PARTID

  !
  SUBROUTINE PARTIT(NAT3,NFREG,NPARMX,NPAR,ATMPAR,PARDIM)
    !-----------------------------------------------------------------------
    !     12-Feb-1993 David Perahia
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     Partition the molecule into as equally sized as possible parts
    !     PARDIM: Maximum allowed matrix dimension. This equals 3*PARD, where
    !             PARD is defined in the input file
    !     NPARMX: Maximum number of allowed parts (in dimb.f90)
    !     ATMPAR(1,*),ATMPAR(2,*): First and last atom numberings
    !                              corresponding to a partitioned block
    !-----------------------------------------------------------------------
  use chm_kinds
  use stream
    implicit none
    ! Passed variables
    INTEGER NAT3,NFREG,NPARMX,NPAR,ATMPAR(2,NPARMX),PARDIM
    ! Local variables
    INTEGER I,J,IATM,NATOM,NT,NTD,IBAS,IADD,INC,MXV
    real(chm_real) RNPAR,R2NPAR

    ! Begin
    NATOM=NAT3/3
    RNPAR=NAT3
    RNPAR=RNPAR/PARDIM
    NPAR=NAT3/PARDIM
    R2NPAR=NPAR
    IF(RNPAR > R2NPAR) NPAR=NPAR+1
    IF(NPAR <= 1) CALL WRNDIE(-3,'<PARTIT>', &
         ' MOLECULE DIVIDED IN ONE PART, USE DIAG INSTEAD OF DIMB')
    IATM=NATOM/NPAR
    NT=IATM*NPAR
    NTD=NATOM-NT
    IBAS=0
    IADD=0
    INC=0
    DO I=1,NPAR
       IADD=IADD+1
       IF(IADD <= NTD) THEN
          INC=1
       ELSE 
          INC=0
       ENDIF
       ATMPAR(1,I)=IBAS+1
       ATMPAR(2,I)=IBAS+IATM+INC
       IBAS=IBAS+IATM+INC
    ENDDO

    IF(ATMPAR(2,NPAR) < NATOM) ATMPAR(2,NPAR)=NATOM
    IF(NPAR > NPARMX) CALL WRNDIE(-3,'<PARTIT>', &
         'NUMBER OF BLOCKS EXCEEDED MAXIMUM ALLOWED NUMBER ')
    WRITE(OUTU,101)
    WRITE(OUTU,103) ((ATMPAR(J,I),J=1,2),I=1,NPAR)
101 FORMAT(/' PARTIT: The atom numberings corresponding to ', &
         'partitioned blocks are:')
103 FORMAT(8X,I5,1X,I5,10X,I5,1X,I5,10X,I5,1X,I5)
    !
    ! Find the maximum block
    !
    MXV=0
    DO I=1,NPAR 
       IF((ATMPAR(2,I)-ATMPAR(1,I)+1) > MXV) &
            MXV=(ATMPAR(2,I)-ATMPAR(1,I)+1)
    ENDDO
    WRITE(OUTU,107) MXV
107 FORMAT(' PARTIT: The largest diagonal block spans ',I5,' atoms')

    RETURN
  END SUBROUTINE PARTIT

  !
  SUBROUTINE RBD2(NAT3,NF,NDIM,DDV,DDV2,DDSCR,DDVBAS,IS1,QMIX,LBIG, &
       IUNMOD)
    !-----------------------------------------------------------------------
    !     10-Nov-1984 Bernard R. Brooks (original RBDIA2 routine)
    !     01-Mar-1994 David Perahia (adapted from RBDIA2 routine)
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     This routine back tranforms the eigenvectors from the reduced
    !     basis to the original basis. Extra copying is here due to
    !     duplicate memory usage and space saving tricks.
    !-----------------------------------------------------------------------
  use chm_kinds
  use number
  use ctitla
  use vector
    implicit none
    ! Passed variables
    INTEGER NAT3,NF,NDIM,IS1,IUNMOD
    real(chm_real) DDV(NAT3,*),DDV2(NDIM,*),DDSCR(NAT3)
    real(chm_real) DDVBAS(NAT3,*)
    LOGICAL QMIX,LBIG
    ! Local variables
    INTEGER I,J,IDIM,IS,ICNTRL(20)
    real(chm_real) VAL
    CHARACTER(len=4) HDR

    ! Begin
    ddv(1:nat3,1:nf)=zero

    IF(LBIG) THEN
       REWIND (UNIT=IUNMOD)
       READ(IUNMOD) HDR,ICNTRL
       CALL RDTITL(TITLEB,NTITLB,IUNMOD,-1)
       READ(IUNMOD)
       READ(IUNMOD)
    ENDIF

    IF(QMIX) THEN
       IS=(IS1-1)*3
       DO IDIM=1,NDIM
          IF(IDIM > NF) THEN
             IS=IS+1
             ddscr(1:nat3)=zero
             DDSCR(IS)=ONE
          ELSE
             IF(LBIG) THEN
                READ(IUNMOD) DDSCR
             ELSE
                ddscr(1:nat3)=ddvbas(1:nat3,idim)
             ENDIF
          ENDIF
          DO I=1,NF
             VAL=DDV2(IDIM,I)
             CALL ADDCTV(DDV(1,I),DDSCR,NAT3,VAL)
          ENDDO
       ENDDO
    ELSE
       DO IDIM=1,NDIM
          IF(LBIG) THEN
             READ(IUNMOD) DDSCR
          ELSE
             ddscr(1:nat3)=ddvbas(1:nat3,idim)
          ENDIF
          DO I=1,NF
             VAL=DDV2(IDIM,I)
             CALL ADDCTV(DDV(1,I),DDSCR,NAT3,VAL)
          ENDDO
       ENDDO
    ENDIF  ! IF(QMIX)

    RETURN
  END SUBROUTINE RBD2

  !
  SUBROUTINE RBDD(NAT3,NF,NDIM,DDV,DDV2,DDSCR,DDVBAS,IS1,IS2,IS3, &
       IS4,QMIX,LBIG,IUNMOD)
    !-----------------------------------------------------------------------
    !     10-Nov-1984 Bernard R. Brooks (original RBDIA2 routine)
    !     01-Mar-1994 David Perahia (adapted from RBDIA2 routine)
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     This routine back tranforms the eigenvectors from the reduced
    !     basis to the original basis. Extra copying is here due to
    !     duplicate memory usage and space saving tricks.
    !     Double window version.
    !-----------------------------------------------------------------------
  use chm_kinds
  use number
  use ctitla
  use vector
    implicit none
    ! Passed variables
    INTEGER NAT3,NF,NDIM,IS1,IS2,IS3,IS4,IUNMOD
    real(chm_real) DDV(NAT3,*),DDV2(NDIM,*),DDSCR(NAT3)
    real(chm_real) DDVBAS(NAT3,*)
    LOGICAL QMIX,LBIG
    ! Local variables
    INTEGER I,J,IDIM,IS,ISF,IS2F,ICNTRL(20)
    real(chm_real) VAL
    CHARACTER(len=4) HDR

    ! Begin
    ddv(1:nat3,1:nf)=zero

    IF(LBIG) THEN
       REWIND (UNIT=IUNMOD)
       READ(IUNMOD) HDR,ICNTRL
       CALL RDTITL(TITLEB,NTITLB,IUNMOD,-1)
       READ(IUNMOD)
       READ(IUNMOD)
    ENDIF

    IF(QMIX) THEN
       IS=(IS1-1)*3
       ISF=(IS3-1)*3
       IS2F=IS2*3
       DO IDIM=1,NDIM
          IF(IDIM > NF) THEN
             IF(IS < IS2F) THEN
                IS=IS+1
                ddscr(1:nat3)=zero
                DDSCR(IS)=ONE
             ELSE
                ISF=ISF+1
                ddscr(1:nat3)=zero
                DDSCR(ISF)=ONE
             ENDIF
          ELSE
             IF(LBIG) THEN
                READ(IUNMOD) DDSCR
             ELSE
                ddscr(1:nat3)=ddvbas(1:nat3,idim)
             ENDIF
          ENDIF
          DO I=1,NF
             VAL=DDV2(IDIM,I)
             CALL ADDCTV(DDV(1,I),DDSCR,NAT3,VAL)
          ENDDO
       ENDDO
    ELSE
       DO IDIM=1,NDIM
          IF(LBIG) THEN
             READ(IUNMOD) DDSCR
          ELSE
             ddscr(1:nat3)=ddvbas(1:nat3,idim)
          ENDIF
          DO I=1,NF
             VAL=DDV2(IDIM,I)
             CALL ADDCTV(DDV(1,I),DDSCR,NAT3,VAL)
          ENDDO
       ENDDO
    ENDIF  ! IF(QMIX)

    RETURN
  END SUBROUTINE RBDD

  !
  SUBROUTINE RBDG(X,Y,Z,NAT3, &
       NDIM,NFRET,DDV,DDF,DDEV,DDSCR, &
       DD5,DDS,DDV2,NADD,INBCMP,JNBCMP,DDVBAS, &
       DD1CMP,QMIX,IS1,IS2,IS3,IS4,CUTF1,NFCUT1,NFREG,IUPD,DD1BLL, &
       FV1,FV2,FV3,FV4,FV6,DRATQ,ERATQ,E2RATQ,BDRATQ,INRATQ, &
       LSCI,LBIG,IUNMOD)
    !-----------------------------------------------------------------------
    !     10-Nov-1984 Bernard R. Brooks (original RBDIAG routine)
    !     01-Mar-1993 David Perahia (adapted from RBDIAG routine)
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     This routine does a reduced basis diagonalization and
    !     saves the eigenvectors in the original basis.
    !     This routine was augmented for doing mixed diagonalizations
    !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use number
  use vector
  use deriv
  use consta
  use ctitla
  use energym
  use eutil
  use stream
  use timerm
  use dimb
  use dimbutils,only:rleg2
    implicit none
    ! Passed variables
    INTEGER NAT3,NDIM,NFRET,NADD
    INTEGER INBCMP(*),JNBCMP(*),IUPD(*)
    INTEGER IS1,IS2,IS3,IS4,NFCUT1,NFREG
    INTEGER INRATQ(*),IUNMOD
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) DDV(NAT3,*),DDF(*),DDEV(*),DDSCR(*)
    real(chm_real) DDVBAS(NAT3,*)
    real(chm_real) DD5(*),DDS(*),DDV2(*)
    real(chm_real) DD1CMP(*),DD1BLL(*)
    real(chm_real) FV1(*),FV2(*),FV3(*),FV4(*),FV6(*)
    real(chm_real) DRATQ(*),ERATQ(*),E2RATQ(*),BDRATQ(*)
    real(chm_real) CUTF1
    LOGICAL QMIX,LSCI,LBIG
    ! Local variables
    INTEGER NATOM,NATP,NFR,NDM2
    INTEGER I,J,IPT,IPU,IS,JS,ISF,ISS,JSS
    INTEGER IH1,IH2,IH3,IH4,IH5,IH6,IH7,IH8

    ! Begin
    NATOM=NAT3/3
    NFR=NFRET-NDIM

    IF(QMIX) THEN

       !
       ! Set up the Hessian DD5 in the mixed reduced basis. Single windowing
       !
       IF(.NOT.QDW) THEN
          IPT=0
          IS=(IS1-1)*3
          DO I=1,NFR
             CALL RLEG2(DDV(1,I),DDSCR,NATOM,DD1CMP,INBCMP,JNBCMP,1)
             JS=IS
             DO J=I,NFRET
                IPT=IPT+1
                IF(J > NFR) THEN
                   JS=JS+1
                   DD5(IPT)=DDSCR(JS)
                ELSE 
                   CALL DOTPR(DDSCR,DDV(1,J),NAT3,DD5(IPT))
                ENDIF
             ENDDO
          ENDDO

          CALL FILUPT(IUPD,NDIM)
          IPT=IPT+1
          CALL MAKDDU(DD5(IPT),DD1CMP,INBCMP,JNBCMP,IUPD, &
               IS1,IS2,NATOM)

       ELSE  ! IF(.NOT.QDW)
          !
          ! Set up the Hessian DD5 in the mixed reduced basis. Double windowing
          !
          IPT=0
          IS=(IS1-1)*3
          ISF=IS2*3
          ISS=(IS3-1)*3
          NDM2=(IS4-IS3+1)*3
          DO I=1,NFR
             CALL RLEG2(DDV(1,I),DDSCR,NATOM,DD1CMP,INBCMP,JNBCMP,1)
             JS=IS
             JSS=ISS
             DO J=I,NFRET
                IPT=IPT+1
                IF(J > NFR) THEN
                   JS=JS+1
                   IF(JS <= ISF) THEN
                      DD5(IPT)=DDSCR(JS)
                   ELSE
                      JSS=JSS+1
                      DD5(IPT)=DDSCR(JSS)
                   ENDIF
                ELSE
                   CALL DOTPR(DDSCR,DDV(1,J),NAT3,DD5(IPT))
                ENDIF
             ENDDO
          ENDDO

          CALL FILUPT(IUPD,NDIM)
          IPU=IPT+1
          CALL MAKDWU(DD5(IPU),DD1CMP,INBCMP,JNBCMP,IUPD, &
               IS1,IS2,IS3,IS4,NATOM)
          CALL FILUPT(IUPD,NDM2)
          IPU=IPT+1+NDIM*(NDIM+1)/2-NDM2*(NDM2+1)/2 
          CALL MAKDDU(DD5(IPU),DD1CMP,INBCMP,JNBCMP,IUPD, &
               IS3,IS4,NATOM)
       ENDIF  ! IF(.NOT.QDW)

    ELSE      ! IF(QMIX)
       !
       ! Set up the Hessian DD5 in the reduced basis. No mixing.
       !
       IPT=0
       DO I=1,NFRET
          CALL RLEG2(DDV(1,I),DDSCR,NATOM,DD1CMP,INBCMP,JNBCMP,1)
          DO J=I,NFRET
             IPT=IPT+1
             CALL DOTPR(DDSCR,DDV(1,J),NAT3,DD5(IPT))
          ENDDO
       ENDDO

    ENDIF      ! IF(QMIX)

    !
    ! Generate the lower section of the matrix and diagonalize
    !
#if KEY_EISPACK==1
    IF(LSCI) THEN
       CALL GENLSC(DD5,DD1BLL,NFRET)
       IF(QMIX) THEN
          CALL DIASCR(NFRET,NFR,DD1BLL,DDEV,DDV2,DRATQ,ERATQ, &
               E2RATQ,BDRATQ,INRATQ,FV1,FV2,FV3,FV4,FV6)
       ELSE
          CALL DIASCI(NFRET,DD1BLL,DDEV,DDV2,FV1,FV2)
       ENDIF
    ELSE
#endif 
       IH1=1
       NATP=NFRET+1
       IH2=IH1+NATP
       IH3=IH2+NATP
       IH4=IH3+NATP
       IH5=IH4+NATP
       IH6=IH5+NATP
       IH7=IH6+NATP
       IH8=IH7+NATP

       IF(QMIX) THEN
          CALL DIAGQ(NFRET,NFR,DD5,DDV2,DDS(IH2),DDS(IH3),DDS(IH4), &
               DDS(IH5),DDS,DDS(IH6),DDS(IH7),DDS(IH8),NADD)
       ELSE
          CALL DIAGQ(NFRET,NFRET,DD5,DDV2,DDS(IH2),DDS(IH3),DDS(IH4), &
               DDS(IH5),DDS,DDS(IH6),DDS(IH7),DDS(IH8),NADD)
       ENDIF

#if KEY_EISPACK==1
    ENDIF   ! IF(LSCI)
#endif 
    NFCUT1=NFRET
    IF(NFRET > NFREG) NFCUT1=NFREG

    IF(.NOT.LSCI) THEN
       DO I=1,NFCUT1
          DDEV(I)=DDS(I)
       ENDDO
    ENDIF

    DO I=1,NFCUT1
       DDF(I)=CNVFRQ*SQRT(ABS(DDEV(I)))
       IF(DDEV(I) < 0.0) DDF(I)=-DDF(I)
    ENDDO
    !
    ! Backtransform to cartesian basis
    !
    IF(QMIX) THEN
       IF(.NOT.QDW) THEN
          CALL RBD2(NAT3,NFR,NFRET,DDV,DDV2,DDSCR,DDVBAS,IS1,QMIX, &
               LBIG,IUNMOD)
       ELSE
          CALL RBDD(NAT3,NFR,NFRET,DDV,DDV2,DDSCR,DDVBAS, &
               IS1,IS2,IS3,IS4,QMIX,LBIG,IUNMOD)
       ENDIF
    ELSE 
       CALL RBD2(NAT3,NFRET,NFRET,DDV,DDV2,DDSCR,DDVBAS,IS1,QMIX, &
            LBIG,IUNMOD)
    ENDIF

    IF((PRNLEV >= 2).AND.(NFSAV > 0)) THEN
       WRITE(OUTU,625)
       WRITE(OUTU,628) (I,DDF(I),I=1,NFSAV)
    ENDIF
625 FORMAT(/' RBDG: Frequencies'/)
628 FORMAT(5(I4,F12.6))
    !
    ! Check cut-off frequency
    !
    IF(NFRET > NFREG) NFRET=NFREG 
    CALL SELNMD(DDF,NFRET,CUTF1,NFCUT1)

    RETURN
  END SUBROUTINE RBDG

  !
  SUBROUTINE RNMTST(V,W,NAT3,DDSCR,DD1CMP,INBCMP,JNBCMP,ISTRT,ISTOP, &
       CMAX,NFIN,NFC,QDIAG,LBIG,IUNMOD)
    !-----------------------------------------------------------------------
    !     01-Mar-1993 David Perahia
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     Residual vectors are added into DDV which
    !     correspond to A*V - (lambda)*I*V (Lanczos type vectors).
    !     (See Brooks and Karplus, (1985), PNAS 82, 4995)
    !     In addition, the magnitude of the residual vector, and the
    !     convergence of the eigenvalues and eigenvectors are calculated.
    !-----------------------------------------------------------------------
  use chm_kinds
  use stream
  use number
  use consta
  use vector
  use dimbutils,only:rleg2
    implicit none
    ! Passed variables
    INTEGER INBCMP(*),JNBCMP(*),NAT3,ISTRT,ISTOP,NFIN,NFC,IUNMOD
    real(chm_real) DD1CMP(*),DDSCR(*)
    real(chm_real) V(NAT3,*),W(NAT3,*),CMAX
    LOGICAL QDIAG,LBIG
    ! Local variables
    INTEGER I,NATOM,IANM,NFR
    real(chm_real) CL,LMAX,FRQ,C,VMAX,CN,CV,CC,CW

    ! Begin
    IF(PRNLEV >= 2) WRITE(OUTU,103)
103 FORMAT(/' RNMTST: Convergence:'/)
    VMAX=RSMALL
    CMAX=RSMALL
    LMAX=RSMALL
    NATOM=NAT3/3
    IANM=NFIN
    NFR=ISTOP-ISTRT+1
    IF(.NOT.LBIG) THEN
       CALL RLEG2(V(1,ISTRT),W(1,ISTRT),NATOM,DD1CMP,INBCMP,JNBCMP,NFR)
       DO I=ISTRT,ISTOP
          CALL DOTPR(V(1,I),W(1,I),NAT3,C)
          CN=-C
          IANM=IANM+1
          IF(QDIAG) THEN
             v(1:nat3,ianm) = w(1:nat3,i)
             CALL ADDCTV(V(1,IANM),V(1,I),NAT3,CN)
          ELSE
             ddscr(1:nat3) = w(1:nat3,i)
             CALL ADDCTV(DDSCR,V(1,I),NAT3,CN)
          ENDIF
          IF(I <= NFC) THEN
             IF(QDIAG) THEN
                CALL DOTPR(V(1,IANM),V(1,IANM),NAT3,CC)
             ELSE
                CALL DOTPR(DDSCR,DDSCR,NAT3,CC)
             ENDIF
             CC=SQRT(CC)
             CALL DOTPR(W(1,I),W(1,I),NAT3,CW)
             CW=SQRT(CW)
             CL=ABS(CW-C)
             CALL NORMALL(W(1,I),NAT3)
             CALL DOTPR(V(1,I),W(1,I),NAT3,CV)
             FRQ=CNVFRQ*SQRT(ABS(C))
             IF(C < ZERO) FRQ=-FRQ
             CV=ONE-ABS(CV)
             IF(PRNLEV >= 2) WRITE(OUTU,105) I,FRQ,CC,CL,CV
             IF(CC > VMAX) VMAX=CC
             IF(CV > CMAX) CMAX=CV
             IF(CL > LMAX) LMAX=CL
          ENDIF
          IF(QDIAG) CALL NORMALL(V(1,IANM),NAT3)
       ENDDO  !  DO I=ISTRT,ISTOP
    ELSE     !  IF(.NOT.LBIG)
       DO I=ISTRT,ISTOP
          CALL RLEG2(V(1,I),W(1,1),NATOM,DD1CMP,INBCMP,JNBCMP,1)
          CALL DOTPR(V(1,I),W(1,1),NAT3,C)
          CN=-C
          IANM=IANM+1
          IF(QDIAG) THEN
             v(1:nat3,ianm) = w(1:nat3,1)
             CALL ADDCTV(V(1,IANM),V(1,I),NAT3,CN)
          ELSE
             ddscr(1:nat3) = w(1:nat3,1)
             CALL ADDCTV(DDSCR,V(1,I),NAT3,CN)
          ENDIF
          IF(I <= NFC) THEN
             IF(QDIAG) THEN
                CALL DOTPR(V(1,IANM),V(1,IANM),NAT3,CC)
             ELSE
                CALL DOTPR(DDSCR,DDSCR,NAT3,CC)
             ENDIF
             CC=SQRT(CC)
             CALL DOTPR(W(1,1),W(1,1),NAT3,CW)
             CW=SQRT(CW)
             CL=ABS(CW-C)
             CALL NORMALL(W(1,1),NAT3)
             CALL DOTPR(V(1,I),W(1,1),NAT3,CV)
             FRQ=CNVFRQ*SQRT(ABS(C))
             IF(C < ZERO) FRQ=-FRQ
             CV=ONE-ABS(CV)
             IF(PRNLEV >= 2) WRITE(OUTU,105) I,FRQ,CC,CL,CV
             IF(CC > VMAX) VMAX=CC
             IF(CV > CMAX) CMAX=CV
             IF(CL > LMAX) LMAX=CL
          ENDIF
          IF(QDIAG) CALL NORMALL(V(1,IANM),NAT3)
       ENDDO
    ENDIF     !  IF(.NOT.LBIG)

105 FORMAT(1X,I5,' Freq= ',F9.4,'  Res-disp= ',F9.5,'  Eval= ',F10.7, &
         '  Evec= ',F10.7)

    IF(PRNLEV >= 2) WRITE(OUTU,107) VMAX,LMAX,CMAX
107 FORMAT(/' RNMTST: Maxima of convergence:'/ &
         13X,' Residual displacement =',F15.7/ &
         13X,' Eigenvalue =           ',F15.7/ &
         13X,' Eigenvector =          ',F15.7)

    RETURN
  END SUBROUTINE RNMTST

  !
  SUBROUTINE SELNMD(DDF,NFRET,CUTF1,NFCUT1)
    !-----------------------------------------------------------------------
    !     01-Feb-1993 David Perahia
    !     15-Dec-1994 Herman van Vlijmen
    !
    !     Determine the numbering of the eigenvector
    !     whose frequency is less than CUTF1
    !-----------------------------------------------------------------------
  use chm_kinds
    implicit none
    ! Passed variables
    real(chm_real) DDF(*),CUTF1
    INTEGER NFRET,NFCUT1
    ! Local variables
    INTEGER I
    !
    DO I=1,NFRET 
       IF(DDF(I) > CUTF1) THEN
          NFCUT1=I-1
          RETURN
       ENDIF
    ENDDO
    NFCUT1=NFRET
    RETURN
  END SUBROUTINE SELNMD

  SUBROUTINE NBLIST(X,Y,Z,NATOM,CTOFNB,INBCMP,JNBCMP,MNBCMP,LENCMP)
    !---------------------------------------------------------------------
    !     15-Jun-1992 David Perahia
    !     16-Dec-1994 Herman van Vlijmen
    !
    !     This routine calculates pair list interaction without any
    !     exclusions. INBCMP and JNBCMP : pair list indices.
    !     At this time, atoms that are farther apart than CTOFNB will
    !     not have entries in the pair list. This may cause errors
    !     in systems where artificial constraints are imposed on atoms
    !     that are far apart.
    !---------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
    implicit none
    ! Passed variables
    real(chm_real) X(*),Y(*),Z(*),CTOFNB,CTFNB2,DIST
    INTEGER NATOM,MNBCMP,INBCMP(*),JNBCMP(*),LENCMP
    ! Local variables
    INTEGER I,J,NBI

    ! Begin
    CTFNB2=CTOFNB*CTOFNB
    NBI=0

    DO I=1,NATOM-1
       DO J=I+1,NATOM
          DIST=(X(I)-X(J))*(X(I)-X(J)) &
               +(Y(I)-Y(J))*(Y(I)-Y(J)) &
               +(Z(I)-Z(J))*(Z(I)-Z(J))
          IF(DIST <= CTFNB2) THEN
             NBI=NBI+1
             JNBCMP(NBI)=J
          ENDIF
       ENDDO
       INBCMP(I)=NBI
       IF(NBI+(NATOM-I-1) > MNBCMP*NATOM) CALL WRNDIE(-3,'<NBLIST>', &
            'ALLOCATED SPACE FOR PAIR LIST OVERFLOW')
    ENDDO
    LENCMP=INBCMP(NATOM-1)
    RETURN
  END SUBROUTINE NBLIST


#endif /* (dimb_main)*/
  subroutine dimbsubs_dummy
    return
  end subroutine dimbsubs_dummy

end module dimbsubs

