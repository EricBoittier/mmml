MODULE OLAPMOD

  use chm_kinds
  use memory
  implicit none

  integer,allocatable,dimension(:),save,public :: IOLAP

CONTAINS
#if KEY_OVERLAP==0 /*main_olap*/
  SUBROUTINE OLAPCMD(COMLYN,COMLEN)
    CHARACTER(len=*) COMLYN
    INTEGER   COMLEN
    CALL WRNDIE(-1,'<OLAP>','OLAP code not compiled.')
    RETURN
  END SUBROUTINE OLAPCMD
#else /* (main_olap)*/
  SUBROUTINE OLAPCMD(COMLYN,COMLEN)
    !----------------------------------------------------------------------
    !
    ! Subroutines for OverLAP method
    !
    ! Written in April 2001 by Milan Hodoscek
    ! Revised in December 2001 by Srdjan Pokorni
    !
    !     This code is licensed under GPL (http://www.gnu/org/copyleft/gpl.html)
    !     and it also follows CHARMM licence rules (current and any future ones)
    !
    use dimens_fcm
    use number
    use olap
    use psf
    use coord
    use select
    use stream
    use string
    implicit none
    !
    CHARACTER(len=*) COMLYN
    INTEGER   COMLEN
    !
    LOGICAL QLPRN,QLAPOF,QQ
    INTEGER ISYST,I
    real(chm_real) SYSWT
    integer,allocatable,dimension(:) :: islct
  !
  !     If this is called the first time then turn on the method
  !     Else we read the selections for each subsystem
  !
    IF(QOLAP)THEN
       QLPRN=INDXA(COMLYN,COMLEN,'PRIN').GT.0
       QLAPOF=INDXA(COMLYN,COMLEN,'OFF').GT.0
       QQ=INDXA(COMLYN,COMLEN,'DEBU').GT.0
       IF(QQ) THEN
          QDOLAP=.TRUE.
          RETURN
       ENDIF
       QQ=INDXA(COMLYN,COMLEN,'NODE').GT.0
       IF(QQ) THEN
          QDOLAP=.FALSE.
          RETURN
       ENDIF
       IF(QLPRN)THEN
          CALL PRNLIST(NATOM,NSYST,NOLAP,LOLAP)
       ELSEIF(QLAPOF)THEN
          QOLAP=.FALSE.
          call chmdealloc("olap.src","olapcmd","iolap",natom,intg=iolap)
          wmain(1:natom) = CESP(1:NATOM)
          IF(PRNLEV.GE.2) WRITE(OUTU,'(A)') &
               'OLAP> ESP charges copied back to WMAIN.'
       ELSE
          call chmalloc("olap.src","olapcmd","islct",natom,intg=islct)
          CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
          ISYST=GTRMI(COMLYN,COMLEN,'SYST',0)
          SYSWT=GTRMF(COMLYN,COMLEN,'WEIG',ONE)
          !
          IF(ISYST.GT.0)THEN
             IF(ISYST.GT.NSYST)CALL WRNDIE(-5,'<OLAP>', &
                  'Subsystem number too big.')
             SYSW(ISYST)=SYSWT
          ELSE
             CALL WRNDIE(-5,'<OLAP>','Wrong subsystem')
          ENDIF
          !
          CALL FILLAP(NATOM,NSYST,ISYST,ISLCT, &
               NOLAP,LOLAP)
          !
          call chmdealloc("olap.src","olapcmd","islct",natom,intg=islct)
       ENDIF
    ELSE
       QOLAP=.TRUE.
       call allocate_olap
       NSYST=GTRMI(COMLYN,COMLEN,'NUMB',0)
       SUPERW=GTRMF(COMLYN,COMLEN,'WEIG',ONE)
       OLAPG=GTRMF(COMLYN,COMLEN,'GAMM',ONE)
       WOLAPG=GTRMF(COMLYN,COMLEN,'WEPO',ONE)
       VOLWF=GTRMF(COMLYN,COMLEN,'VOLW',ZERO)
       CHAWF=GTRMF(COMLYN,COMLEN,'CHAW',ZERO)
       ESPWF=GTRMF(COMLYN,COMLEN,'ESPW',ZERO)
       IF(NSYST.EQ.0)CALL WRNDIE(-5,'<OLAP>','Wrong init.')
       IF((VOLWF.LT.ZERO).OR.(CHAWF.LT.ZERO).OR.(ESPWF.LT.ZERO)) &
            CALL WRNDIE(-5,'<OLAP>','Invalid weighting factors!')
       IF((VOLWF+CHAWF+ESPWF).EQ.ZERO)CALL WRNDIE(-5,'<OLAP>', &
            'No weighting factors specified!')
       IF(PRNLEV.GE.2) WRITE(OUTU,'(A,I5)') &
            'OLAP> Number of subsystems =', nsyst
       LOLAP=NATOM
       IF(NATOM.GT.MAXA-1)CALL WRNDIE(-5,'<OLAP>','Too many atoms')
       call chmalloc("olap.src","olapcmd","iolap",lolap+1,intg=iolap)
       CALL FILLAP0(NATOM,NOLAP,LOLAP)
       cesp(1:natom) = WMAIN(1:NATOM)
       IF(PRNLEV.GE.2) WRITE(OUTU,'(A)') &
            'OLAP> ESP charges copied from WMAIN.'
       IF(PRNLEV.GE.2) WRITE(OUTU,'(A)') &
            'OLAP> use of scalar wmain is now safe.'
    ENDIF
  !
    RETURN
  END SUBROUTINE OLAPCMD
!
  SUBROUTINE PRNLIST(NATOM,NSYST,NOLAP,LOLAP)
    !----------------------------------------------------------------------
    !
    !     This routine prints NOLAP and IOLAP arrays
    !
    use stream
    implicit none
    !
    INTEGER NATOM,NSYST,NOLAP(*),LOLAP,I,J
    !
    DO I=1,NATOM
       IF(PRNLEV.GE.2) WRITE(OUTU,'(2I5)')I,NOLAP(I)
       IF(PRNLEV.GE.2) WRITE(OUTU,'(20I4)') &
            I,(IOLAP(J),J=NOLAP(I),NOLAP(I+1)-1)
    ENDDO
    IF(PRNLEV.GE.2) WRITE(OUTU,'(2I5)')NATOM+1,NOLAP(NATOM+1)
    !
    RETURN
  END SUBROUTINE PRNLIST
  !
  SUBROUTINE FILLAP0(NATOM,NOLAP,LOLAP)
    !----------------------------------------------------------------------
    !
    implicit none
    !
    INTEGER NATOM,NOLAP(*),LOLAP,I
    !
    DO I=1,NATOM+1
       NOLAP(I)=I
       IOLAP(I)=-1
    ENDDO
    !
    RETURN
  END SUBROUTINE FILLAP0
  !
  SUBROUTINE FILLAP1(NATOM,NOLAP,ISEL,LOLAP)
    !----------------------------------------------------------------------
    !
    implicit none
    !
    INTEGER NATOM,NOLAP(*),ISEL(*),LOLAP
    !
    INTEGER I,IPT,NPREV
    !
    ! NOTE:::
    !           **** THERE IS A BUG HERE !!!! (see olap.inp)
    !
    IPT=0
    NPREV=NOLAP(1)
    DO I=2,NATOM+1
       IF(IOLAP(NPREV).NE.-1)IPT=IPT+ISEL(I-1)
       NPREV=NOLAP(I)
       NOLAP(I)=NOLAP(I)+IPT
    ENDDO
    LOLAP=LOLAP+IPT
    !
    RETURN
  END SUBROUTINE FILLAP1
  !
  SUBROUTINE FILLAP2(NATOM,NOLAP,NOLAPT,IOLAPT,ISEL,ISYST)
    !----------------------------------------------------------------------
    !
    implicit none
    !
    INTEGER NATOM,NOLAP(*),NOLAPT(*),IOLAPT(*),ISEL(*),ISYST
    !
    INTEGER I,N
    !
    DO I=1,NATOM
       N=NOLAPT(I+1)-NOLAPT(I)
       IOLAP(NOLAP(I):nolap(i)-1+n) = IOLAPT(NOLAPT(I):NOLAPT(I)+n-1)
       IF(ISEL(I).GT.0)IOLAP(NOLAP(I+1)-1)=ISYST
    ENDDO
    !
    RETURN
  END SUBROUTINE FILLAP2
  !
  SUBROUTINE FILLAP(NATOM,NSYST,ISYST,ISEL,NOLAP,LOLAP)
    !----------------------------------------------------------------------
    !
    !     This routine fills NOLAP and IOLAP arrays
    !
    !     NOLAP(NATOM)    contains pointers into IOLAP.
    !     IOLAP           contains the subsystem numbers
    !                     in which an atom may find itself
    !
    use exfunc
    !
    implicit none
    !
    INTEGER NATOM,NSYST,ISYST,ISEL(*)
    INTEGER NOLAP(*),LOLAP
    !
    INTEGER I,LOLAPT
    integer,allocatable,dimension(:) :: iolapt,nolapt
    !
    !     Get the space for temporary copy and copy
    LOLAPT=LOLAP
    call chmalloc("olap.src","fillap","iolapt",lolapt,intg=iolapt)
    call chmalloc("olap.src","fillap","nolapt",natom+1,intg=nolapt)
    iolapt(1:lolap) = IOLAP(1:LOLAP)
    nolapt(1:natom+1) = NOLAP(1:NATOM+1)
    !
    !     Redefine the NOLAP array
    !
    CALL FILLAP1(NATOM,NOLAP,ISEL,LOLAP)
    !
    !     Expand the IOLAP array, and fill it with the new information
    !
    call chmdealloc("olap.src","fillap","iolap",lolapt,intg=iolap)
    call chmalloc("olap.src","fillap","iolap",lolap,intg=iolap)
    CALL FILLAP2(NATOM,NOLAP,NOLAPT,IOLAPT,ISEL,ISYST)
    !
    !     Clean the HEAP
    call chmdealloc("olap.src","fillap","iolapt",lolapt,intg=iolapt)
    call chmdealloc("olap.src","fillap","nolapt",natom+1,intg=nolapt)
    !
    RETURN
  END SUBROUTINE FILLAP
  !
#endif /* (main_olap)*/
!
END MODULE OLAPMOD

