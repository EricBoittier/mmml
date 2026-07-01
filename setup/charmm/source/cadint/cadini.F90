module cadpac_mod
  use chm_kinds
  use dimens_fcm
  use cadpac_data

  implicit none
  
  integer cadpac_dummy_var

contains

  !CHARMM Element source/cadint/cadini.src $Revision: 1.2 $
#if KEY_CADPAC==0 /*cadpac*/
  SUBROUTINE CADINI(COMLYN,COMLEN)
    CHARACTER(len=*) COMLYN
    INTEGER   COMLEN

    CALL WRNDIE(-1,'<CADINI>','CADPAC code not compiled.')
    return
  end SUBROUTINE CADINI
#else /* (cadpac)*/

  subroutine cadpac_init()
    use gamess_fcm
    ncadpc=0
    nqqchg=0
    qgmrem=.false.
    return
  end subroutine cadpac_init

  SUBROUTINE CADINI(COMLYN,COMLEN)
    !-----------------------------------------------------------------------
    !     This is the interface for running CADPAC with CHARMM for
    !     QM/MM calculations
    !
    !     It closely follows the interface developed for GAMESS by
    !     Milan Hodoscek.
    !
    !     Paul D. Lyne September 1995
    !
    !     Code added for dynamic memory allocation on GNU systems
    !     Ben Webb January 2000

  use code
  use coord
  use param
  use psf
  use stream
  use energym
  use gamess_fcm
  use parallel
  use select
  use string
  use memory

    CHARACTER(len=*) COMLYN
    INTEGER   COMLEN

    LOGICAL   QQINP
    integer,allocatable,dimension(:) :: ISLCT

    qmused = .true.      ! safe here since next are the allocations, ...
    call allocate_gamess ! try to reduce from MAXA to NATOM

    call chmalloc('cadini.src','CADINI','ISLCT',NATOM,intg=ISLCT)
    CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
    QGMREM=(INDXA(COMLYN,COMLEN,'REMO') > 0)
#if KEY_PARALLEL==1
    IF(MYNOD == 0)THEN
#endif
       IF(QGMREM) THEN
          WRITE(OUTU,22) &
               'REMOve: Classical energies within QM atoms are removed.'
       ELSE
          WRITE(OUTU,22) &
               'No REMOve: Classical energies within QM atoms are retained.'
       ENDIF
#if KEY_PARALLEL==1
    ENDIF
#endif
    QGMEXG=(INDXA(COMLYN,COMLEN,'EXGR') > 0)
#if KEY_PARALLEL==1
    IF(MYNOD == 0)THEN
#endif
       IF(QGMEXG) THEN
          WRITE(OUTU,22) &
               'EXGRoup: QM/MM Electrostatics for link host groups removed.'
       ELSE
          WRITE(OUTU,22) &
               'No EXGRoup: QM/MM Elec. for link atom host only is removed.'
       ENDIF
#if KEY_PARALLEL==1
    ENDIF
#endif
    QQINP=(INDXA(COMLYN,COMLEN,'QINP') > 0)
#if KEY_PARALLEL==1
    IF(MYNOD == 0)THEN
#endif
       IF(QQINP) THEN
          WRITE(OUTU,22) &
               'QINP: Charges will input for QM atoms.'
       ELSE
          WRITE(OUTU,22) &
               'No QINP: Charges will be based on atomic numbers.'
       ENDIF
#if KEY_PARALLEL==1
    ENDIF
#endif
22  FORMAT('CADINT> ',A)

    MUSTUP=.TRUE.

    CALL COPSEL(ISLCT,QQINP)
    CALL CH2CAD
    CALL CHMQUA

    !     This initialize cadpac data and performs one gradient calculation
    CALL CADPAC

    call chmdealloc('cadini.src','CADINI','ISLCT',NATOM,intg=ISLCT)
    COMLEN = 0
    ISTRM = 5

    RETURN
  END SUBROUTINE CADINI

  SUBROUTINE CH2CAD
    !-----------------------------------------------------------------------
    !     Define CHARMM atoms as point charges and copy to COMMON/LATTIC/
    !
    !     Paul Lyne September 1995

  use coord
  use psf
  use stream
  use consta
  use gamess_fcm

    INTEGER I

    !     All atoms which are not quantum contribute to electrostatic
    !     interaction in the QM part. See MJF reference:
    !     J. Comp. Chem., Vol. 11, No. 6, 700-733 (1990)

    NLAT=0
    DO I = 1,NATOM
       IF(IGMSEL(I) == 0)THEN
          NLAT = NLAT + 1
          CLAT(1,NLAT)=X(I)/BOHRR
          CLAT(2,NLAT)=Y(I)/BOHRR
          CLAT(3,NLAT)=Z(I)/BOHRR
          ZLAT(NLAT)=CG(I)
          CHALAT=CHALAT+ZLAT(NLAT)
          XALAT=XALAT+ZLAT(NLAT)*CLAT(1,NLAT)
          YALAT=YALAT+ZLAT(NLAT)*CLAT(2,NLAT)
          ZALAT=ZALAT+ZLAT(NLAT)*CLAT(3,NLAT)
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE CH2CAD

  SUBROUTINE CHMQUA
    !----------------------------------------------------------------------
    !     Find the atoms defined as QM atoms and get them ready
    !     for CADPAC. The QM atoms are stored in an array held
    !     in COMMON/COORDN/
    !
    !     Paul D Lyne, September 1995
    !
  use number
  use coord
  use param
  use psf
  use stream
  use gamess_fcm
  use rtf, only: atct
  use linkatom, only: findel

  implicit none

    character(len=10) :: AATOM(MAXCEN)
    REAL(CHM_REAL) :: AZNUC(MAXCEN), CORD(3,MAXCEN)
    INTEGER NAT

    INTEGER I,N,NSLCT,NATMM,NATQM,NATLNK
    character(len=6) :: ELE
    LOGICAL QPRT

    QPRT=.TRUE.
    N=0
    DO I = 1,NATOM
       IF ((IGMSEL(I) == 1).OR.(IGMSEL(I).EQ.2)) THEN
          N = N + 1
          CORD(1,N)=X(I)
          CORD(2,N)=Y(I)
          CORD(3,N)=Z(I)
          AATOM(N)=ATYPE(I)
          IF (ATYPE(I)(1:3) == 'QQH') AATOM(N)=' H'

          CALL FINDEL(ATCT(IAC(I)),AMASS(I),I,ELE,AZNUC(N),QPRT)

          CADATM(N)=AATOM(N)
          CADZIN(N)=AZNUC(N)
          CADCRD(1,N)=CORD(1,N)
          CADCRD(2,N)=CORD(2,N)
          CADCRD(3,N)=CORD(3,N)
       ENDIF
    ENDDO

    CADATM(N+1)='END'
    NAT=N
    NATREL=NAT

    IF (NAT  <=  0) CALL WRNDIE(0,'<CHMQUA>', &
         'No quantum mechanical atoms selected.')
    NATMM = NATOM -N
    NATQM = N
    NCADPC = N

    NSLCT = 0
    DO I = 1,NATOM
       IF (IGMSEL(I) == 2) NSLCT = NSLCT + 1
    ENDDO
    NATLNK = NSLCT

    !     Write out atomic information

    IF (PRNLEV > 2) WRITE (OUTU,'(/,1x,A,/)') &
         ' CADDFN> Some atoms will be treated quantum mechanically.'
    IF (PRNLEV > 2) WRITE (OUTU,'(4(8X,A,I5,/),/)') &
         ' The number of quantum mechanical atoms   = ',NATQM, &
         ' The number of molecular mechanical atoms = ',NATMM, &
         ' The number of MM atoms excluded from QM  = ',NATMM-NLAT, &
         ' Of which the number of QM/MM link atoms  = ',NATLNK
    RETURN
  END SUBROUTINE CHMQUA

  SUBROUTINE CHMINP(CDNAME,CDCHRG,CDCOOR,NUQ)
    !--------------------------------------------------------------
    !     Return unique atom centre information to CADPAC
    !     Paul D Lyne, Harvard University, 1995
    !
  use consta

    character(len=4) :: CDNAME
    REAL(CHM_REAL) :: CDCHRG(MAXCEN), CDCOOR(3,MAXCEN)
    INTEGER(chm_int4) NUQ

    INTEGER I
    REAL(CHM_REAL) :: CFR

    CFR=1.0D0/BOHRR

    CDNAME=CADATM(NUQ)
    CDCHRG(NUQ)=CADZIN(NUQ)
    DO I=1,3
       CDCOOR(I,NUQ)=CADCRD(I,NUQ)*CFR
    ENDDO

    RETURN
  END SUBROUTINE CHMINP

  SUBROUTINE CADENE(CTOT,CX,CY,CZ,DX,DY,DZ,NATM)
    !-----------------------------------------------------------------------
    !
    !     Get energy and forces from CADPAC
    !
    !     Paul D. Lyne, September 1995
    !
  use consta
  use gamess_fcm
  use number
  use stream
  use coord
  use parallel
  use memory

    INTEGER(chm_int4) MAXQ,MMAXQ,MAXDIS
    COMMON/MAXLEN/MAXQ,MMAXQ,MAXDIS

    real(chm_real),allocatable,dimension(:) :: QWRK

    REAL(CHM_REAL) :: CTOT,CX(*),CY(*),CZ(*),DX(*),DY(*),DZ(*)
    INTEGER NATM

    INTEGER ICHARM

    REAL(CHM_REAL) :: E,EG
    COMMON /FUNCT/ E,EG(MAXN3)

    REAL(CHM_REAL) :: ZAN,C
    INTEGER(chm_int4) NAT,ICH,MUL,NUM,NNP,NE,NA,NB
    COMMON/INFOA/NAT,ICH,MUL,NUM,NNP,NE,NA,NB,ZAN(MAXCEN),C(3,MAXCEN)

    INTEGER I,N,IPT

    !     Are there any QM atoms?
    IF(NCADPC == 0) RETURN

    ! Update coordinates
    N=0
    DO I = 1,NATM
       IF ((IGMSEL(I) == 1).OR.(IGMSEL(I).EQ.2)) THEN
          N = N + 1
          C(1,N)=CX(I)/BOHRR
          C(2,N)=CY(I)/BOHRR
          C(3,N)=CZ(I)/BOHRR
       ENDIF
    ENDDO

    CALL CH2CAD

    IF(NLAT /= 0) THEN
       DO ICHARM=1,NLAT
          DXELMM(ICHARM)=ZERO
          DYELMM(ICHARM)=ZERO
          DZELMM(ICHARM)=ZERO
       ENDDO
    ENDIF

    ! mkg 2010
    call chmalloc('cadini.src','CADENE','QWRK',MAXQ,crl=QWRK)
    CALL OPTDRV(QWRK,QWRK,MAXQ)
    call chmdealloc('cadini.src','CADENE','QWRK',MAXQ,crl=QWRK)

#if KEY_PARALLEL==1
    IF(MYNOD == 0)THEN
       CTOT=(E)*TOKCAL
    ELSE
       CTOT=ZERO
    ENDIF
#else /**/
    CTOT=(E)*TOKCAL
#endif

    !     QM atoms

#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif
       N = 0
       DO I = 1,NATM
          IF ((IGMSEL(I) == 1).OR.(IGMSEL(I).EQ.2)) THEN
             N = N + 1
             IPT=3*(N-1)+1
             DX(I) = DX(I) + EG(IPT)*TOKCAL/BOHRR
             DY(I) = DY(I) + EG(IPT+1)*TOKCAL/BOHRR
             DZ(I) = DZ(I) + EG(IPT+2)*TOKCAL/BOHRR

             IF(PRNLEV > 6) THEN
                WRITE(OUTU,'(2I5,3F14.6)') I,N, EG(IPT)*TOKCAL/BOHRR     &
                     , EG(IPT+1)*TOKCAL/BOHRR &
                     , EG(IPT+2)*TOKCAL/BOHRR
             ENDIF

          ENDIF
       ENDDO

       !     MM atoms, without igmsel(i)=-1 !!

       N = 0
       DO I = 1,NATM
          IF (IGMSEL(I)  ==  0) THEN
             N = N + 1
             DX(I) = DX(I) - DXELMM(N)*TOKCAL/BOHRR &
                  + DXREP(N)*TOKCAL/BOHRR
             DY(I) = DY(I) - DYELMM(N)*TOKCAL/BOHRR &
                  + DYREP(N)*TOKCAL/BOHRR
             DZ(I) = DZ(I) - DZELMM(N)*TOKCAL/BOHRR &
                  + DZREP(N)*TOKCAL/BOHRR

             IF(PRNLEV > 6) THEN
                WRITE(OUTU,'(2I5,3F14.6,3F14.9)') I,N &
                     , - DXELMM(N)*TOKCAL/BOHRR &
                     , - DYELMM(N)*TOKCAL/BOHRR &
                     , - DZELMM(N)*TOKCAL/BOHRR &
                     , DXREP(N)*TOKCAL/BOHRR &
                     , DYREP(N)*TOKCAL/BOHRR &
                     , DZREP(N)*TOKCAL/BOHRR
             ENDIF

          ENDIF
       ENDDO
#if KEY_PARALLEL==1
    ENDIF
#endif
    !
    !
    RETURN
  END SUBROUTINE CADENE

  SUBROUTINE COPSEL(ISLCT,QQINP)
    !-----------------------------------------------------------------------
    !     Copies selection vector to common block
    !     so it may be used by GAMESS interface
    !     Call this routine only once and retain definition
    !     of QM, MM, and link atoms throughout the claculation.
    !     We call this from GAMINI which is called from charmm/charmm.src
    !
    !     IGMSEL(I) = 2  Link atom
    !     IGMSEL(I) = 1  QM atom
    !     IGMSEL(I) = 0  MM atom
    !     IGMSEL(I) = -1 MM atom to be excluded from QM/MM interaction
    !
    !     MM atom in position close to link atom is excluded from interaction
    !     of external charges to QM region. Instead of this atom is already
    !     a link atom so no need for two atoms in one place!
    !

  use coord
  use gamess_fcm
  use stream
  use psf
  use number
  use chutil

    INTEGER ISLCT(*)
    LOGICAL QQINP
    INTEGER I,J,I1,I2,N,LN,IS,IQ
    character(len=8) SID, RID, REN, AC
    LOGICAL LNFLAG

    DO I=1, NATOM
       IGMSEL(I)=ISLCT(I)
       IF (ATYPE(I)(1:2) == 'QQ') IGMSEL(I)=2
    ENDDO

    !     Check if link atom is connected to any of its neighbors. If
    !     yes then that atom will not be included in QM/MM interaction.
    !     This is sometimes necessary to prevent oposite charge collision,
    !     since QM cannot prevent this to happen.

    DO I=1,NBOND
       I1=IB(I)
       I2=JB(I)
       IF (IGMSEL(I1) == 2) THEN
          !           Don't change QM atoms
          IF(QGMEXG) THEN
             !              remove the entire group
             J=GETRES(I2,IGPBS,NGRP)
             IS=IGPBS(J)+1
             IQ=IGPBS(J+1)
             DO J=IS,IQ
                IF(IGMSEL(J) == 0) IGMSEL(J)=5
             ENDDO
          ELSE
             !              remove the link host atom
             IF(IGMSEL(I2) == 0) IGMSEL(I2)=5
          ENDIF
       ENDIF
       IF (IGMSEL(I2) == 2) THEN
          IF(QGMEXG) THEN
             !              remove the entire group
             J=GETRES(I1,IGPBS,NGRP)
             IS=IGPBS(J)+1
             IQ=IGPBS(J+1)
             DO J=IS,IQ
                IF(IGMSEL(J) == 0) IGMSEL(J)=5
             ENDDO
          ELSE
             !              remove the link host atom
             IF(IGMSEL(I1) == 0) IGMSEL(I1)=5
          ENDIF
       ENDIF
    ENDDO

    IF(PRNLEV >= 2) THEN
       WRITE(OUTU,118)
       WRITE(OUTU,120)  &
            'Classical atoms excluded from the QM calculation'
    ENDIF
118 FORMAT('------------------------------------------------')
120 FORMAT('CADINT: ',A,':')
122 FORMAT(10X,I5,4(1X,A))
124 FORMAT(10X,'NONE.')
    N=0
    DO I=1,NATOM
       IF(IGMSEL(I) == -1) THEN
          CALL ATOMID(I,SID,RID,REN,AC)
          IF(PRNLEV >= 2) WRITE(OUTU,122) I, &
               SID(1:idleng),RID(1:idleng),REN(1:idleng),AC(1:idleng)
          N=N+1
       ENDIF
    ENDDO
    IF(PRNLEV >= 2) THEN
       IF(N == 0) WRITE(OUTU,124)
       WRITE(OUTU,120) 'Quantum mechanical atoms'
    ENDIF
    N=0
    DO I=1,NATOM
       IF(IGMSEL(I) == 1) THEN
          CALL ATOMID(I,SID,RID,REN,AC)
          IF(PRNLEV >= 2) WRITE(OUTU,122) I, &
               SID(1:idleng),RID(1:idleng),REN(1:idleng),AC(1:idleng)
          N=N+1
       ENDIF
    ENDDO
    IF(PRNLEV >= 2) THEN
       IF(N == 0) WRITE(OUTU,124)
       WRITE(OUTU,120) 'Quantum mechanical link atoms'
    ENDIF
    N=0
    DO I=1,NATOM
       IF(IGMSEL(I) == 2) THEN
          CALL ATOMID(I,SID,RID,REN,AC)
          IF(PRNLEV >= 2) WRITE(OUTU,122) I, &
               SID(1:idleng),RID(1:idleng),REN(1:idleng),AC(1:idleng)
          N=N+1
       ENDIF
    ENDDO
    IF(PRNLEV >= 2) THEN
       IF(N == 0) WRITE(OUTU,124)
       WRITE(OUTU,118)
    ENDIF
    !
    !     Alow for partial charges on any QM atom
    !
    N=0
    LN=0
    DO I = 1,NATOM
       IF (IGMSEL(I)  >=  1) THEN
          N = N + 1
          !
          !     Non integer charges for link atoms can be specified separately
          !     Also allow to change them subsequently with SCALar command
          !     using QINP keyword in GAMEss command.
          !
          LNFLAG=.FALSE.
          IF (ATYPE(I)(1:2) == 'QQ' .AND. NQQCHG /= 0) THEN
             LN=LN+1
             IF(QQCHG(LN) > -NINE99) THEN
                FQQCHG(N) = QQCHG(LN)
             ELSE
                FQQCHG(N) = -THOSND
             ENDIF
             !
             !     Don't have any more link atoms, put NQQCHG to 0
             !     for possible subsequent changes with SCALar command
             !     or on restarts.
             !
             IF(LN == NQQCHG) NQQCHG=0
             LNFLAG=.TRUE.
          ENDIF
          !
          !
          !     First check the flag QINP for all atoms.
          !
          IF(QQINP) THEN
             !
             !     All QM charges (accept ones specified in ADDLink
             !     are taken from PSF
             !
             IF(.NOT.LNFLAG) FQQCHG(N)=CG(I)
          ELSE
             FQQCHG(N)=-THOSND
          ENDIF
          !
          !          IF(PRNLEV >= 2) WRITE(OUTU,'(A,2I5,A,F15.5)')
          !    $          'CADINT: ATOM(',I,N,') has QNUC: ',FQQCHG(N)
       ENDIF
    ENDDO
    !
    ! Zero charges on quantum atoms to remove from MM term.
    IF(QGMREM) THEN
       DO I=1, NATOM
          IF((IGMSEL(I) == 1).EQV.(IGMSEL(I).EQ.2)) CG(I)=ZERO
       ENDDO
    ENDIF
    !
    RETURN
  END SUBROUTINE COPSEL

#endif /* (cadpac)*/
end module cadpac_mod
