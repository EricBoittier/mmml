#if KEY_MMFF==1
! =====================================================================
! LOGICAL FUNCTION ARBOND(I,J) : TRUE if i-j bond in an aromatic ring
! RETURN .TRUE. IF THE BOND BETWEEN ATOMS I AND J LIES WITHIN
! AN AROMATIC RING, OTHERWISE RETURN .FALSE.
! =====================================================================
LOGICAL FUNCTION ARBOND(I,J)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use mmffm
  implicit none
  integer i, j
  ! T.A. Halgren change (B980629.wy)
  integer n
  ! end T.A. Halgren change
  !
  ARBOND=.FALSE.
  IF(.NOT.IFAROM(I)) RETURN
  IF(.NOT.IFAROM(J)) RETURN
  ! T.A. HAlgren change (B980629.wy)
  !      IF(INRING(I).EQ.INRING(J)) THEN
  !   OK, I AND J ARE EACH "AROMATIC" ATOMS AND LIE IN THE SAME
  !   (COMPOSITE) RING
  !             ARBOND=.TRUE.
  !      ENDIF
  ARBOND=.TRUE.
  DO N=1,N_ARBOND
     IF(I.EQ.AR_BOND(1,N).AND.J.EQ.AR_BOND(2,N)) RETURN
     IF(I.EQ.AR_BOND(2,N).AND.J.EQ.AR_BOND(1,N)) RETURN
  ENDDO
  ARBOND=.FALSE.
  ! end T.A. Halgren change
  RETURN
END FUNCTION ARBOND

! ===============================================================
! SUBROUTINE DEFANG : GETS A MISSING ANGLE PARAMETERS
!   CALLED BY SUB KTHETAM TO GET A MISSING BOND ANGLE FORCE CONSTANT
!   (ACON) AND IDEAL ANGLE (AZERO).  MCLASS = 0 SIGNIFIES A "NORMAL"
!   BOND ANGLE.  MCLASS = 3 INDICATE AN ANGLE IN A 3-MEMBERED RING.
!   MCLASS = 4 INDICATES AN ANGLE IN A 4-MEMBERED RING.  IAMMFF, IBmmff, AND
!   IC ARE THE THREE ATOMS WHICH COMPRISE THE IAMMFF-IBmmff-IC ANGLE
!
! ===============================================================
!
SUBROUTINE DEFANG(IAMMFF,IBmmff,IC,MCLASS,MCAB,MCBC,ACON,AZERO, &
     GOTPAR)
  !      SUBROUTINE DEFANG(IAMMFF,IBmmff,IC,KPT,ACON,AZERO,J,GOTPAR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  Jay Banks 25 Oct 95: changed argument IA to IAmmff, to avoid conflict
  !  with variable in common (mmff.f90).
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use mmffm
  use psf
  use rtf, only: atnumt
  use stream
  implicit none
  !
  integer iammff, IBmmff, ic, mclass, mcab, mcbc
  integer ita, itb, itc
  integer ian, ibn, icn
  real(chm_real)  acon, azero
  !
  LOGICAL GOTPAR
  !   SET GOTPAR TO FALSE ON ENTRY
  GOTPAR=.FALSE.
  !   IF AZERO WAS NONZERO, ASSUME THIS VALUE TO BE APPROPRIATE FOR THE
  !   INTERACTION OF INTEREST AND BRANCH TO CALCULATE THE FORCE CONSTANT
  IF(AZERO.GT.0.) GOTO 100
  !
  !   ASSIGN IDEAL ANGLE FROM HYBRIDIZATION
  !
  IF(AZERO.LE.0.) THEN
     !     COMMON/MMFFPROP/MSPEC(KVP0),MCOORD(KVP0),MVALMIN(KVP0),
     !    . MVALMAX(KVP0),MPILP(KVP0),MLTBND(KVP0),MAROM(KVP0),
     !    . MLINBA(KVP0),MSP2SGL(KVP0)
     ITB=MTYPE(IBmmff)
     AZERO=120.
     IF(MCOORD(ITB).EQ.4) AZERO=109.45
     IF(MCOORD(ITB).EQ.3) AZERO=120.
     IF(MCOORD(ITB).EQ.3.AND.MVALMIN(ITB).EQ.3) THEN
        IF(MLTBND(ITB).EQ.0) THEN
           IF(AtNumT(ITB).EQ.7) THEN
              AZERO=107.
           ELSE
              AZERO=92.
           ENDIF
        ENDIF
     ENDIF
     IF(MCOORD(ITB).EQ.2) THEN
        AZERO=120.
        IF(AtNumT(ITB).EQ.8) AZERO=105.
        IF(AtNumT(ITB).GT.10) AZERO=95.
        IF(MLINBA(ITB).GT.0) AZERO=180.
     ENDIF
     IF(MCLASS.EQ.3.OR.MCLASS.EQ.5.OR.MCLASS.EQ.6) AZERO=60.
     IF(MCLASS.EQ.4.OR.MCLASS.EQ.7.OR.MCLASS.EQ.8) AZERO=90.
  ENDIF
  !
  !   JUST RETURN IF AZERO IS STILL NOT ASSIGNED
  !
  IF(AZERO.LE.0.) RETURN
100 CONTINUE
  !
  !   ASSIGN FORCE CONSTANT FROM EMPIRICAL RELATIONSHIP
  !
  ITA=MTYPE(IAMMFF)      ! atomic types
  ITB=MTYPE(IBmmff)
  ITC=MTYPE(IC)
  IAN=ATNUM(IAMMFF)      ! atomic numbers
  IBN=ATNUM(IBmmff)
  ICN=ATNUM(IC)
  CALL EMPFC(IAN,IBN,ICN,ITA,ITB,ITC,AZERO,ACON,MCLASS,MCAB,MCBC)
  !      CALL KANEMP(IAMMFF,IBmmff,IC,AZERO,ACON) ! REPLACED BY CALL TO EMPFC
  GOTPAR=.TRUE.
  !
  RETURN
END SUBROUTINE DEFANG

! ===============================================================
! SUBROUTINE DEFBND : Assign default bond length and force constant
! ===============================================================
SUBROUTINE DEFBND(I,K,MCLASS,FCON,BLZERO,GOTPAR)
  !      SUBROUTINE DEFBND(I,K,FCON,BLZERO,GOTPAR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  use stream
  use mmffm
  implicit none
  !
  integer i, ibtype, ihyb, mclass
  integer k, khyb, ii, kk
  real(chm_real)  blzero, fcon
  !
  LOGICAL GOTPAR,STRDEF
  !      NATOM     (IN)   NUMBER OF SUBSTRATE ATOMS
  !   INFORMATION STORED ON A PER-ATOM BASIS:
  !      X,Y,Z     (IN)   (X(IA),Y(IA),Z(IA)) IS POSN OF ATOM IA
  !      ATNUM  (IN)   ATOMIC NUMBERS
  !      HYBRID  (IN)   HYBRIDIZATION SYMBOL ('SP3','SP2','SP','S', OR 'P')
  !      IFAROM  (IN)   LOGICAL; .TRUE. IF AN ATOM IS AN AROMATIC RING
  !      AtName    (IN)   ATOM NAME
  !      NRES    (IN)   RESIDUE NAME
  !      KRES    (IN)   RESIDUE TYPE
  !      ITAB(*,I) (IN) SPECIFIES ATOMS ATTACHED TO ATOM I; FIRST OCCURENCE
  !                     OF ITAB=-1 INDICATES END OF LIST - I.E., IF ITAB(1,I)=2,
  !                     ITAB(2,I)=5, ITAB(3,I)=-1, THEN I IS ATTACHED TO 2 OTHER
  !                     ATOMS, NUMBERS 1 AND 5.
  !      LOCAT(I) (IN)  VALUE UNSPECIFIED
  !      NENTRY   (IN)  SHOULD BE SAME AS NATOM; REDUNDANT INFORMATION
  !
  !   BOND INFORMATION:
  !     NBOND    (IN)  NUMBER OF BONDS IN THE SUBSTRATE OR INHIBITOR
  !    LOGICAL UNITS:
  !    IN        (IN)  TTY OR SCRIPT-FILE (FOR005)
  !    IO        (IN)  TTY OR LOG-FILE OUTPUT (FOR006)
  !    IZ        (IN)  HARD-COPY OUTPUT IN INTERACTIVE RUNS; DOES
  !                    NOT EXIST (IZ=0) IN BACKGROUND RUNS
  !    IX        (IN)  SCRIPT FILE BEING FORMED, IF IN "ECHO" MODE
  !    IOZ       (IN)  = IZ (INTERACTIVE RUNS) OR =IO (BACKGROUND RUNS);
  !                    USED TO ENABLE "IN" INPUT TO BE ECHOED TO IO
  !                    (IN BACKGROUND RUNS) OR IZ (INTERACTIVE RUNS),
  !                    AS APPROPRIATE
  !   SET GOTPAR TO FALSE ON ENTRY
  GOTPAR=.FALSE.
  !   PREPARE FOR CALL TO STRDEF TO GET THE DEFAULT VALUES
  IHYB=3
  II=MTYPE(I)
  KK=MTYPE(K)
  IF(MLTBND(II).EQ.1.OR.MLTBND(II).EQ.2) IHYB=2
  IF(MLTBND(II).EQ.3) IHYB=1
  KHYB=3
  IF(MLTBND(KK).EQ.1.OR.MLTBND(KK).EQ.2) KHYB=2
  IF(MLTBND(KK).EQ.3) KHYB=1
  IBTYPE=IBORDR(I,K)
  IF(MLTBND(II).EQ.1.AND.MLTBND(KK).EQ.1) IBTYPE=4
  IF(MLTBND(II).EQ.1.AND.MLTBND(KK).EQ.2) IBTYPE=5
  IF(MLTBND(II).EQ.2.AND.MLTBND(KK).EQ.1) IBTYPE=5
  IF(MAROM(II).EQ.1.AND.MAROM(KK).EQ.1.AND.MCLASS.EQ.0) THEN
     IF(MPILP(II).EQ.0.AND.MPILP(KK).EQ.0) THEN
        IBTYPE=4
     ELSE
        !
        !   TAKE ONLY PARTIAL-RESONANCE SHORTENING
        !
        IBTYPE=5
     ENDIF
  ENDIF
  !D     WRITE(OUTU,'(A,6I5)') ' I, K, IHYB, KHYB, MCLASS, IBTYPE',
  !D    .              I,K,IHYB,KHYB,MCLASS,IBTYPE
  BLZERO=0.
  GOTPAR=STRDEF(ATNUM(I),ATNUM(K),IHYB,KHYB,IBTYPE, &
       FCON,BLZERO,GOTPAR)
  !D     WRITE(OUTU,*) '        FCON, BLZERO',FCON,BLZERO
  !     IF(GOTPAR) WRITE(OUTU,*) ' GOTPAR = .TRUE.'
  !
  RETURN
END SUBROUTINE DEFBND

! ===============================================================
! SUBROUTINE DEFCHG : Assign default partial atomic charges
! CALLED BY SUB GENCHG IN "dimens" AND IN MOLEDIT "CHARGES"
! CHG IS THE ATOM I - ATOM J BOND-CHARGE INCREMENT
! THE SIGN CONVENTION IS THAT THE ATOM OF HIGHER MMFF ATOM
! TYPE (MTYPE(I) OR MTYPE(J)) HAS A PARTIAL CHARGE OF CHG
! ELECTRONS; THE ATOM OF LOWER ATOM TYPE HAS A CHARGE OF -CHG.
! ===============================================================
!
SUBROUTINE DEFCHG(I,J,CHG,GOTPAR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  use stream
  use string
  use mmffm
  implicit none
  !
  integer i, ia1, ia2, it1, it2
  integer j
  real(chm_real)  a, chg, dchg, en1, en2, factor
  real(chm_real)  EN(53)
  !
  LOGICAL GOTPAR
  !
  !   DATA STATEMENTS
  DATA EN/2.1,0.,0.,0.,2.0,2.5,3.0,3.5,4.0,0.,0.,0.,0., &
                                !              H  HE LI BE  B   C   N   O   F  NE NA MG AL
       1.8,2.1,2.5,3.0,17*0.,2.8,17*0.,2.5/
  !              SI  P   S   CL        BR        I
  !      NATOM     (IN)   NUMBER OF SUBSTRATE ATOMS
  !   INFORMATION STORED ON A PER-ATOM BASIS:
  !      X,Y,Z     (IN)   (X(IA),Y(IA),Z(IA)) IS POSN OF ATOM IA
  !      ATNUM  (IN)   ATOMIC NUMBERS
  !      MTYPE   (IN)   MMFF ATOM TYPES
  !      SYMB    (IN)   SYMBOLIC MMFF ATOM TYPES (CHARACTER*4)
  !      HYBRID  (IN)   HYBRIDIZATION SYMBOL ('SP3','SP2','SP','S', OR 'P')
  !      IFAROM  (IN)   LOGICAL; .TRUE. IF AN ATOM IN AN AROMATIC RING
  !      AtName    (IN)   ATOM NAME
  !      NRES    (IN)   RESIDUE NAME
  !      KRES    (IN)   RESIDUE TYPE
  !      ITAB(*,I) (IN) SPECIFIES ATOMS ATTACHED TO ATOM I; FIRST OCCURENCE
  !                     OF ITAB=-1 INDICATES END OF LIST - I.E., IF ITAB(1,I)=2,
  !                     ITAB(2,I)=5, ITAB(3,I)=-1, THEN I IS ATTACHED TO 2 OTHER
  !                     ATOMS, NUMBERS 1 AND 5.
  !      LOCAT(I) (IN)  VALUE UNSPECIFIED
  !      NENTRY   (IN)  SHOULD BE SAME AS NATOM; REDUNDANT INFORMATION
  !
  !   SET GOTPAR TO FALSE ON ENTRY
  GOTPAR=.FALSE.
  !
  !   SEE IF STORED "PARTIAL BOND CHARGE INCREMENTS" CAN BE USED TO
  !   ASSIGN THE DEFAULT VALUE FOR THE BOND-CHARGE INCREMENT
  !
  IT1=MTYPE(I)
  IT2=MTYPE(J)
  IF(LPBCI(IT1).AND.LPBCI(IT2)) THEN
     !
     !   USE THE STORED "PARTIAL BOND CHARGE INCREMENTS"
     !
     CHG=PBCI(IT2)-PBCI(IT1)
     GOTPAR=.TRUE.
     if(prnlev.gt.2) WRITE(OUTU,310) &
          QNAME(I),QNAME(J),IT1,IT2,CHG,QNAME(J),QNAME(I)
310  FORMAT(/' NO CHARGE INCREMENT FOUND FOR THE BOND BETWEEN', &
          ' ATOMS ',A,' AND ',A/' (NOTE: THESE ATOMS WERE ASSIGNED MMFF' &
          ,' ATOM TYPES OF ',I3,' AND ',I3,', RESPECTIVELY)'// &
          ' PARTIAL BCIs WILL BE USED TO ASSIGN A CHARGE OF',F6.2, &
          ' ELECTRONS TO ATOM '/A,' AND A CHARGE OF THE SAME MAGNITUDE', &
          ' BUT OPPOSITE SIGN TO ATOM ',A/)
     !        WRITE(6,*) ' GOTPAR = ',GOTPAR
     RETURN
  ELSE
     !
     !   USE THE DIFERENCE IN STORED PAULING ELECTRONEGATIVITIES INSTEAD;
     !   TEST TO MAKE SURE THESE ATOMS ARE ONES COVERED
     !
     IA1=ATNUM(I)
     IA2=ATNUM(J)
     EN1=EN(IA1)
     IF(EN1.LE.0.) THEN
        IF(WRNLEV.GT.-5) WRITE(SCRTCH,'(A,I5,A)') &
             'MISSING PARAMETERS FOR ATOMIC NUMBER:',IA1,QNAME(I)
        CALL WRNDIE(-5,'<DEFCHG>',SCRTCH(:52))
     ENDIF
     EN2=EN(IA2)
     IF(EN2.LE.0) THEN
        IF(WRNLEV.GT.-5) WRITE(SCRTCH,'(A,I5,A)') &
             'MISSING PARAMETERS FOR ATOMIC NUMBER:',IA2,QNAME(J)
        CALL WRNDIE(-5,'<DEFCHG>',SCRTCH(:52))
     ENDIF
     FACTOR=0.25
     CHG=-FACTOR*(EN2-EN1)
     if(prnlev.gt.2) WRITE(OUTU,300) &
          QNAME(I),QNAME(J),IT1,IT2,DCHG,QNAME(J),QNAME(I)
300  FORMAT(/' NO CHARGE INCREMENT FOUND FOR THE BOND BETWEEN', &
          ' ATOMS ',A,' AND ',A/' (NOTE: THESE ATOMS WERE ASSIGNED MMFF' &
          ,' ATOM TYPES OF ',I3,' AND ',I3,', RESPECTIVELY)'// &
          ' THE DEFAULT PROCEDURE WILL ASSIGN A CHARGE OF',F6.2, &
          ' ELECTRONS TO ATOM ',A/' AND A CHARGE OF THE SAME MAGNITUDE', &
          ' BUT OPPOSITE SIGN TO ATOM ',A/)
  ENDIF
  !
  DCHG=0 ! RCZ940202 : what should be this value ?
  !
  CHG=DCHG
  A=CHG
  GOTPAR=.TRUE.
  !
  RETURN
END SUBROUTINE DEFCHG

! ===============================================================
! SUBROUTINE DEFOOP : Assign default values for out-plane angles
! CALLED BY SUB THETA TO OBTAIN A MISSING OUT-OF-PLANE
! BENDING CONSTANT.  ID IS THE OUT-OF-PLANE ATOM; IT IS
! CONNECTED TO CENTRAL ATOM IBmmff IN ANGLE IAMMFF-IBmmff-IC
! ONLY CERTAIN TYPES OF SP2-HYBRIDIZED ATOMS IBmmff ARE ALLOWED
! TO HAVE OOP BENDING.
! COPBK IS THE WILSON OOP BENDING CONSTANT
! ===============================================================
!
SUBROUTINE DEFOOP(IAMMFF,IBmmff,IC,ID,COPBK,GOTPAR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  16 Oct 95 Jay Banks: changed variable name "IB" to "IBmmff" to avoid
  !                       conflict with CHARMM name
  !  Jay Banks 25 Oct 95: changed argument IA to IAmmff, to avoid conflict
  !  with variable in common (mmff.f90).
  !
  use chm_kinds
  use dimens_fcm
  use stream
  implicit none
  !
  integer iammff, ibmmff, ic, id
  real(chm_real)  copbk
  !
  LOGICAL GOTPAR
  !
  !   SET GOTPAR TO FALSE ON ENTRY
  GOTPAR=.FALSE.
  !   THE FOLLOWING COMMENTED-OUT CODE SHOWS THE MESSAGE SUB KTHETAM
  !   WILL PRINT IF GOTPAR=.FALSE. ON RETURN;  IN THIS CASE, NO
  !   OUT-OF-PLANE BENDING IS ASSOCIATED WITH ATOM ID
  !   ISSUE WARNING
  !
  !         WRITE(OUTU,8500) MTYPE(IBmmff),MTYPE(ID),IAMMFF,IBmmff,IC
  !         IF(IZ.NE.0) WRITE(IZ,8500) MTYPE(IBmmff),MTYPE(ID),IAMMFF,IBmmff,IC
  ! 8500    FORMAT(/'* * * * * ERROR - CONSTANTS FOR OUT-OF-PLANE TYPE  0',
  !     .    2I3,' MUST BE READ IN'/1H ,20X,'USED IN ANGLE NUMBER',3I3)
  !
  RETURN
END SUBROUTINE DEFOOP

! ======================================================================
! SUBROUTINE DEFSTBN : ASSIGNS DEFAULT VALUES FOR MISSING STRETCH-BEND
! FORCE CONSTANTS
! ======================================================================
!
SUBROUTINE DEFSTBN(IAN,JAN,KAN,FIJK,FKJI,GOTPAR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  Developed at Merck and Co., Inc. by T. Halgren.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  !#INCLUDE '~/charmm_fcm/code.f90'
  !#INCLUDE '~/charmm_fcm/consta.f90'
  !#INCLUDE '~/charmm_fcm/exfunc.f90'
  use param
  use mmffm
  !#INCLUDE '~/charmm_fcm/psf.f90'
  use stream
  !
  implicit none
  !      COMMON/DEFSB/DEFSBK(0:4,1:4,0:4) ! in mmff.f90
  LOGICAL GOTPAR
  integer ian, jan, kan, ir, jr, kr
  real(chm_real) fijk, fkji
  ! DEFINE SOME ATOM NAME CONSTANTS.
  INTEGER HELIUM,NEON,ARGON,KRYPTN,XENON,RADON
  PARAMETER (HELIUM = 2)
  PARAMETER (NEON = 10)
  PARAMETER (ARGON = 18)
  PARAMETER (KRYPTN = 36)
  PARAMETER (XENON = 54)
  PARAMETER (RADON = 86)
  !
  !    ian    atomic number for atom i
  !    jan    atomic number for atom j
  !    kan    atomic number for atom k
  !    fijk   st-bn force constant, in mdynes/angstrom, for stretching
  !           the i-j bond while bending the i-j-k angle
  !    fkji   st-bn force constant, in mdynes/angstrom, for stretching
  !           the k-j bond while bending the i-j-k angle
  !
  !   ASSIGN ROW POSITION IN PERIODIC TABLE
  !
  IF(IAN.LE.HELIUM) THEN
     IR=0
  ELSE IF(IAN.LE.NEON) THEN
     IR=1
  ELSE IF(IAN.LE.ARGON) THEN
     IR=2
  ELSE IF(IAN.LE.KRYPTN) THEN
     IR=3
  ELSE IF(IAN.LE.XENON) THEN
     IR=4
  ELSE IF(IAN.LE.RADON) THEN
     IR=5
  ENDIF
  !
  IF(JAN.LE.HELIUM) THEN
     JR=0
  ELSE IF(JAN.LE.NEON) THEN
     JR=1
  ELSE IF(JAN.LE.ARGON) THEN
     JR=2
  ELSE IF(JAN.LE.KRYPTN) THEN
     JR=3
  ELSE IF(JAN.LE.XENON) THEN
     JR=4
  ELSE IF(JAN.LE.RADON) THEN
     JR=5
  ENDIF
  !
  IF(KAN.LE.HELIUM) THEN
     KR=0
  ELSE IF(KAN.LE.NEON) THEN
     KR=1
  ELSE IF(KAN.LE.ARGON) THEN
     KR=2
  ELSE IF(KAN.LE.KRYPTN) THEN
     KR=3
  ELSE IF(KAN.LE.XENON) THEN
     KR=4
  ELSE IF(KAN.LE.RADON) THEN
     KR=5
  ENDIF
  !
  !   TEST LIMITS
  !
  GOTPAR=.FALSE.
  FIJK=0.
  FKJI=0.
  IF(JR.EQ.0.OR.JR.GT.4) THEN
     if(wrnlev.ge.2) WRITE(OUTU,'(/A,I3,A/A)') &
          'DEFSTBN> *** Invalid atomic number of', JAN, &
          ' for central atom in a stretch-bend interaction ***', &
          'DEFSTBN> *** Interaction ignored '
     !         IF(IZ.NE.0) WRITE(IZ,'(/A,I3,A/A)')
     !     .   ' *** Invalid atomic number of', JAN,' for central atom '//
     !     .   'in a stretch-bend interaction ***',
     !     .   ' *** Interaction ignored '
     RETURN
  ENDIF
  !
  IF(IR.GT.4) THEN
     if(wrnlev .ge. 2) WRITE(OUTU,'(/A,I3,A/A)') &
          'DEFSTBN> *** Invalid atomic number of', IAN, &
          ' for a wing atom in a stretch-bend interaction ***', &
          'DEFSTBN> *** Interaction ignored '
     !          IF(IZ.NE.0) WRITE(IZ,'(/A,I3,A/A)')
     !     .   ' *** Invalid atomic number of', IAN,' for a wing atom '//
     !     .   'in a stretch-bend interaction ***',
     !     .   ' *** Interaction ignored '
     RETURN
  ENDIF
  !
  IF(KR.GT.4) THEN
     if(wrnlev .ge. 2) WRITE(OUTU,'(/A,I3,A/A)') &
          'DEFSTBN> *** Invalid atomic number of', KAN, &
          ' for a wing atom in a stretch-bend interaction ***', &
          ' *** Interaction ignored '
     !         IF(IZ.NE.0) WRITE(IZ,'(/A,I3,A/A)')
     !     .   ' *** Invalid atomic number of', KAN,' for a wing atom '//
     !     .   'in a stretch-bend interaction ***',
     !     .   ' *** Interaction ignored '
     RETURN
  ENDIF
  !
  !   FIND DEFAULT PARAMETER FOR THIS INTERACTION TYPE
  !
  FIJK=DEFSBK(IR,JR,KR)
  FKJI=DEFSBK(KR,JR,IR)
  !D     WRITE(OUTU,'(A,3I5,2F8.2)') ' IR,JR,KR,FIJK,FKJI ',IR,JR,KR,FIJK,FKJI
  GOTPAR=.TRUE.
  !
  RETURN
END SUBROUTINE DEFSTBN

! ===============================================================
! SUBROUTINE DEFTOR : Assign zero default values for dihedrals
! CALLED BY KOMEGA TO OBTAIN MISSING V1 - V3 CONSTANTS
! (VAL1 - VAL3) FOR TORSION ANGLE i-j-k-l
! THIS APPROACH IS BASED ON, BUT SOMEWHAT MORE SPECIFIC THAN, THE APPROA
! TAKEN BY Rappe and Casewit in UFF (JACS, 1992, 114, 10024-10035)
!
! ===============================================================
!
SUBROUTINE DEFTOR(I,J,K,L,VAL1,VAL2,VAL3,MCLASS,GOTPAR)
  !      SUBROUTINE DEFTOR(IA,IB,IC,ID,VAL1,VAL2,VAL3,MODE,GOTPAR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 27-28 Nov 95: Changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  !  WRITE statements in error conditions.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use psf
  use rtf, only: atnumt
  use stream
  implicit none
  !
  integer i, j, k, l, mclass
  integer ii, jj, kk, ll, jan, kan, jjan, kkan, n, ibjk
  integer mcjj, mckk, mvjj, mvkk, njk
  real(chm_real)  val1, val2, val3
  real(chm_real) v2jj, v2kk, v3jj, v3kk
  real(chm_real) pibojk, vrjj, vrkk
  logical jrow(5), krow(5), arbond
  !
  LOGICAL GOTPAR
  ! DEFINE SOME ATOM NAME CONSTANTS.
  INTEGER HELIUM,NEON,ARGON,KRYPTN,XENON,RADON
  PARAMETER (HELIUM = 2)
  PARAMETER (NEON = 10)
  PARAMETER (ARGON = 18)
  PARAMETER (KRYPTN = 36)
  PARAMETER (XENON = 54)
  PARAMETER (RADON = 86)
  !      LOGICAL JROW1,JROW2,JROW3,JROW4,JROW5
  !      LOGICAL KROW1,KROW2,KROW3,KROW4,KROW5
  !      ATNUM  ATOMIC NUMBERS
  !      MTYPE  MMFF ATOM TYPES
  !   SET GOTPAR TO TRUE ON ENTRY
  GOTPAR=.TRUE.
  !
  II=MTYPE(I)
  JJ=MTYPE(J)
  KK=MTYPE(K)
  LL=MTYPE(L)
  JAN=ATNUM(J)
  KAN=ATNUM(K)
  JJAN=AtNumT(JJ)
  KKAN=AtNumT(KK)
  IF(JAN.NE.JJAN) THEN
     if(wrnlev.ge.2) WRITE(OUTU,'(A,1X,A,A,I4,A)') &
          ' ***** INCONSISTENCY IN RETRIEVED ATOMIC NUMBERS FOR ATOM', &
          QNAME(J),' OF MMFF TYPE',JJ,' *****'
     call wrndie(-5,'<deftor>','inconsistent atomic numbers')
  ELSE IF(KAN.NE.KKAN) THEN
     if(wrnlev.gt.-5) WRITE(OUTU,'(A,1X,A,A,I4,A)') &
          ' ***** INCONSISTENCY IN RETRIEVED ATOMIC NUMBERS FOR ATOM', &
          QNAME(K),' OF MMFF TYPE',KK,' *****'
     call wrndie(-5,'<deftor>','inconsistent atomic numbers')
  ENDIF
  !
  !   ASSIGN ROW POSITION IN PERIODIC TABLE
  !
  DO N=1,5
     JROW(N)=.FALSE.
     KROW(N)=.FALSE.
  ENDDO
  IF(JAN.LE.NEON) THEN
     JROW(1)=.TRUE.
     V3JJ=2.0
     V2JJ=2.0
     IF(JAN.EQ.6) THEN
        V3JJ=2.12
        V2JJ=2.0
     ELSE IF(JAN.EQ.7) THEN
        V3JJ=1.5
        V2JJ=2.0
     ELSE IF(JAN.EQ.8) THEN
        V2JJ=2.
        V3JJ=0.2
     ENDIF
  ELSE IF(JAN.LE.ARGON) THEN
     JROW(2)=.TRUE.
     IF(JAN.EQ.14) V3JJ=1.22
     IF(JAN.EQ.15) V3JJ=2.40
     IF(JAN.EQ.16) V3JJ=0.48
     V2JJ=1.25
  ELSE IF(JAN.LE.KRYPTN) THEN
     JROW(3)=.TRUE.
     IF(JAN.EQ.32) V3JJ=0.70
     IF(JAN.EQ.33) V3JJ=1.5
     IF(JAN.EQ.34) V3JJ=0.34
     V2JJ=0.7
  ELSE IF(JAN.LE.XENON) THEN
     JROW(4)=.TRUE.
     IF(JAN.EQ.50) V3JJ=0.20
     IF(JAN.EQ.51) V3JJ=1.1
     IF(JAN.EQ.52) V3JJ=0.3
     V2JJ=0.2
  ELSE IF(JAN.LE.RADON) THEN
     JROW(5)=.TRUE.
     V3JJ=0.5
     V2JJ=0.1
  ENDIF
  !
  IF(KAN.LE.NEON) THEN
     KROW(1)=.TRUE.
     V3KK=2.0
     V2KK=2.0
     IF(KAN.EQ.6) THEN
        V3KK=2.12
        V2KK=2.0
     ELSE IF(KAN.EQ.7) THEN
        V3KK=1.5
        V2KK=2.
     ELSE IF(KAN.EQ.8) THEN
        V2KK=2.
        V3KK=0.2
     ENDIF
  ELSE IF(KAN.LE.ARGON) THEN
     KROW(2)=.TRUE.
     IF(KAN.EQ.14) V3KK=1.22
     IF(KAN.EQ.15) V3KK=2.40
     IF(KAN.EQ.16) V3KK=0.48
     V2KK=1.25
  ELSE IF(KAN.LE.KRYPTN) THEN
     KROW(3)=.TRUE.
     IF(KAN.EQ.32) V3KK=0.70
     IF(KAN.EQ.33) V3KK=1.5
     IF(KAN.EQ.34) V3KK=0.34
     V2KK=0.7
  ELSE IF(KAN.LE.XENON) THEN
     KROW(4)=.TRUE.
     IF(KAN.EQ.50) V3KK=0.20
     IF(KAN.EQ.51) V3KK=1.1
     IF(KAN.EQ.52) V3KK=0.3
     V2KK=0.2
  ELSE IF(KAN.LE.RADON) THEN
     KROW(5)=.TRUE.
     V3KK=0.3
     V2KK=0.1
  ENDIF
  !
  IBJK=IBORDR(J,K)
  MCJJ=MCOORD(JJ)
  MCKK=MCOORD(KK)
  MVJJ=MVALMIN(JJ)
  MVKK=MVALMIN(KK)
  NJK=(MCJJ-1)*(MCKK-1)
  VAL1=0.
  VAL2=0.
  VAL3=0.
  IF(NJK.EQ.0) THEN
     !
     !   JUST RETURN - NO TORSION COMPONENTS
     !
     RETURN
  ENDIF
  IF(MLINBA(JJ).EQ.1.OR.MLINBA(KK).EQ.1) THEN
     !
     !   JUST RETURN - ONE OR MORE "LINEAR BOND ANGLES" IS INVOLVED
     !
     RETURN
  ENDIF
  !
  !   PROCESS FIRST AROMATIC AND FULL DOUBLE BONDS
  !
  IF(ARBOND(J,K)) THEN
     !
     !    AN AROMATIC PARTIAL DOUBLE BOND
     !
     IF(MPILP(JJ).EQ.0.AND.MPILP(KK).EQ.0) THEN
        PIBOJK=0.5
     ELSE
        !
        !   THE BOND WON'T BE FULLY DELOCALIZED IF J OR K IS THE HETERO-
        !   ATOM IN A 5-MEMBERED AROMATIC RING BEARING A PI LONME PAIR
        !
        PIBOJK=0.3
     ENDIF
     VAL2=6.*SQRT(V2JJ*V2KK)*PIBOJK
     !         IF(MVALMAX(JJ).EQ.4.AND.MVALMAX(KK).NE.4) VAL2=0.5*VAL2
     RETURN
  ENDIF
  !
  IF(IBJK.EQ.2) THEN
     IF(MVALMIN(JJ).EQ.MVALMAX(JJ).AND.MLTBND(JJ).EQ.2) THEN
        IF(MVALMIN(KK).EQ.MVALMAX(KK).AND.MLTBND(KK).EQ.2) THEN
           !
           !   A FULL DOUBLE BOND BETWEEN J AMD K
           !
           PIBOJK=1.
           VAL2=6.*SQRT(V2JJ*V2KK)*PIBOJK
           RETURN
        ENDIF
     ENDIF
     !
     !   NOT A FULL DOUBLE BOND BUT STILL A VERY STRONG RESONANT
     !   INTERACTION - TREAT AS NEARLY HALF A PI BOND
     !
     PIBOJK=0.4
     VAL2=6.*SQRT(V2JJ*V2KK)*PIBOJK
     RETURN
  ENDIF
  !
  IF(MCJJ.EQ.4.AND.MCKK.EQ.4) THEN
     !
     !   BOTH CENTRAL ATOMS ARE TETRACOORDINATE - ASSIGN SATURATED
     !   SINGLE-BOND VALUES
     !
     VAL3=SQRT(V3JJ*V3KK)/NJK
     RETURN
  ENDIF
  IF(MCJJ.EQ.4) THEN
     !
     !   THE FIRST CENTRAL ATOM IS TETRACOORDINATE
     !
     IF(MCKK.EQ.3.AND.(MVKK.EQ.4.OR.MLTBND(KK).GT.0)) THEN
        !
        !   THE SECOND CENTRAL ATOM IS TRICOORDINATE BUT TETRAVALENT -
        !   ASSIGN ZERO TORSION FOR A SP3-SP2 ROTOR, SO JUST RETURN
        !
        RETURN
     ELSE IF(MCKK.EQ.2.AND.(MVKK.EQ.3.OR.MLTBND(KK).GT.0)) THEN
        !
        !   THE SECOND CENTRAL ATOM IS DICOORDINATE BUT TRIVALENT -
        !   ASSIGN ZERO TORSION FOR A SP3-SP2 ROTOR, SO JUST RETURN
        !
        RETURN
     ELSE
        !
        !   TREAT LIKE A SATURATED BOND WITH A TOTAL 3-FOLD POTENTIAL
        !   OF 2 KCAL/MOL
        !
        VAL3=SQRT(V3JJ*V3KK)/NJK
        RETURN
     ENDIF
  ENDIF
  IF(MCKK.EQ.4) THEN
     !
     !   THE SECOND CENTRAL ATOM IS TETRACOORDINATE
     !
     IF(MCJJ.EQ.3.AND.(MVJJ.EQ.4.OR.MLTBND(JJ).GT.0)) THEN
        !
        !   THE FIRST CENTRAL ATOM IS TRICOORDINATE BUT TETRAVALENT -
        !   ASSIGN ZERO TORSION FOR A SP3-SP2 ROTOR, SO JUST RETURN
        !
        RETURN
     ELSE IF(MCJJ.EQ.2.AND.(MVJJ.EQ.3.OR.MLTBND(JJ).GT.0)) THEN
        !
        !   THE FIRST CENTRAL ATOM IS DICOORDINATE BUT TRIVALENT -
        !   ASSIGN ZERO TORSION FOR A SP3-SP2 ROTOR, SO JUST RETURN
        !
        RETURN
     ELSE
        !
        !   TREAT LIKE A SATURATED BOND WITH A TOTAL 3-FOLD POTENTIAL
        !   OF 2 KCAL/MOL
        !
        VAL3=SQRT(V3JJ*V3KK)/NJK
        RETURN
     ENDIF
  ENDIF
  !
  !   DONE WITH CASES IN WHICH EITHER OR BOTH CENTRAL ATOMS ARE
  !   TETRACOORDINATE
  !
  IF(MLTBND(JJ).GT.0.AND.MLTBND(KK).GT.0 &
       .OR. MLTBND(JJ).GT.0.AND.MPILP(KK).EQ.1 &
       .OR.MPILP(JJ).EQ.1.AND.MLTBND(KK).GT.0) THEN
     IF(MPILP(JJ).EQ.1.AND.MPILP(KK).EQ.1) THEN
        !
        !   THIS IS NOT REALLY A RESONANT INTERACTION, SINCE BOTH ATOMS
        !   HAVE PI LONE PAIRS
        !
        PIBOJK=0.
     ELSE IF(MPILP(JJ).EQ.1.AND.MLTBND(KK).GT.0) THEN
        IF(MLTBND(JJ).EQ.1) THEN
           !
           !   ATOM J IS CAPABLE OF STRONG RESONANT INTERACTIONS (EX.: AMIDE
           !   NITROGENS)
           !
           PIBOJK=0.5
        ELSE
           !
           !   THIS IS A PI-LONE-PAIR--DOUBLE BOND INTERACTION
           !
           IF(JROW(1).AND.KROW(1)) THEN
              PIBOJK=0.3
           ELSE
              PIBOJK=0.15
           ENDIF
        ENDIF
     ELSE IF(MPILP(KK).EQ.1.AND.MLTBND(JJ).GT.0) THEN
        IF(MLTBND(KK).EQ.1) THEN
           !
           !   ATOM K IS CAPABLE OF STRONG RESONANT INTERACTIONS (EX.: AMIDE
           !   NITROGENS)
           !
           PIBOJK=0.5
        ELSE
           !
           !   THIS IS A PI-LONE-PAIR--DOUBLE BOND INTERACTION
           !
           IF(JROW(1).AND.KROW(1)) THEN
              PIBOJK=0.3
           ELSE
              PIBOJK=0.15
           ENDIF
        ENDIF
     ELSE IF((MLTBND(JJ).EQ.1.OR.MLTBND(KK).EQ.1).AND. &
          (JAN.NE.6.OR.KAN.NE.6)) THEN
        !
        !   THIS MULTIPLE-BOND--MULTIPLE-BOND INTERACTION IS PROPABLY
        !   A DELOCALIZED PARTIAL DOUBLE BOND OF ORDER 1/2 OR 1/3 IN
        !   A RESONANT SYSTEM
        !
        PIBOJK=0.4
     ELSE
        !
        !   THIS IS A NORMAL MULTIPLE-BOND--MULTIPLE-BOND INTERACTION, AS IN
        !   BUTATIENE
        !
        PIBOJK=0.15
     ENDIF
     !
     !   ASSIGN THE TWO-FOLD TORSION CONSTANTS FOR THIS RESONANT INTERACTION
     !
     VAL2=6.*SQRT(V2JJ*V2KK)*PIBOJK
     RETURN
  ENDIF
  !
  !   IF HERE, BOTH ATOMS MUST BE SATURATED
  !
  IF((JAN.EQ.8.OR.JAN.EQ.16.OR.JAN.EQ.33.OR.JAN.EQ.51).AND. &
       (KAN.EQ.8.OR.KAN.EQ.16.OR.KAN.EQ.33.OR.KAN.EQ.51)) THEN
     !
     !   BOTH ATOMS ARE IN OXYGEN'S COLUMN
     VRJJ=8.0
     IF(JAN.EQ.8) VRJJ=2.0
     VRKK=8.0
     IF(KAN.EQ.8) VRKK=2.
     VAL2=-SQRT(VRJJ*VRKK)
     RETURN
  ELSE
     !
     !   ONE ATOM HAS AT MOST ONE LONE PAIR; TREAT AS A THREE-FOLD
     !   POTENTIAL BETWEEN SATURATED ATOMS
     VAL3=SQRT(V3JJ*V3KK)/NJK
     RETURN
  ENDIF
  !
  !   IF HERE, AN ERROR ...
  !
#if KEY_UNUSED==1 /*no_path_to_here*/
  if(wrnlev.ge.2) then
     WRITE(OUTU,'(A,4I5)') &
          ' *** TORSION DEFAULT NOT ASSIGNED FOR ATOMS ',I,J,K,L
     WRITE(OUTU,'(A,4I5)') ' MMFF TYPES: ',II,JJ,KK,LL
  endif
#endif /* (no_path_to_here)*/
  !
  RETURN
END SUBROUTINE DEFTOR

! ======================================================================
! SUBROUTINE EMPFC: ASSIGNS BENDING CONSTANTS
! FROM AN EMPIRICAL FORMULA ACCORDING TO THE ATOM TYPES
! COMPRISING THE ANGLE AND THE REFERENCE VALUE FOR THE ANGLE
! ======================================================================
SUBROUTINE EMPFC(IAMMFF,IBmmff,IC,ITA,ITB,ITCM,ANAT,ACON,MCLASS, &
     MCAB,MCBC)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  16 Oct 95 Jay Banks: changed variable name "IB" to "IBmmff" to avoid
  !                       conflict with CHARMM name
  !
  !  Jay Banks 25 Oct 95: changed argument IA to IAmmff, to avoid conflict
  !  with variable in common (mmff.f90).
  !
  !  Jay Banks 22 Nov 95: added IF (PRNLEV.GE.2) THEN around two WRITE
  !  statements that didn't have it.
  !
  !  Jay Banks 27 Nov 95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  !  WRITE statements followed by CALL WRNDIE.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  use code
  use consta
  use exfunc
  use mmffm
  use param
  use psf
  use stream
  implicit none
  !
  integer i, iammff, IBmmff, ic, mel
  integer MCAB, MCBC, MCLASS, ita, itb, itcm, itt, j
  logical newpar, full
  real(chm_real)  acon, alpha, anat, beta, cpbmmff
  real(chm_real)  gamma, orabc, r, rab, rbc, scale
  real(chm_real)  zpa, zpc
  real(chm_real)  ZP(12),CP(12)
  !
  INTEGER ELNO(12)
  DATA ELNO/1,5,6,7,8,9,14,15,16,17,35,53/
  DATA ZP/1.395,0.,2.494,2.711,3.045,2.847,2.350,2.350,2.980,2.909, &
       3.017,3.086/
  DATA CP/0.,0.704,1.016,1.113,1.337,0.,0.811,1.068,1.249,1.078, &
       0.,0./
  ! H  1.395 0.000
  ! B  0.    0.704
  ! C  2.494 1.017 ! AVERAGE OF THIS AND C4 "TETRAHED-C" VALUE USED
  ! N  2.711 1.113
  ! O  3.045 1.337
  ! F  2.847 0.
  ! SI 2.350 0.811  ! Z=2.350 ASSIGNED AS EQUAL TO THAT FOR P; NO DATA
  ! P  2.350 1.068
  ! S  2.980 1.249
  ! CL 2.909 1.078
  ! BR 3.017 0.
  ! I  3.086 0.
  ! AS 0.    0.825
  ! C4 0.    1.015  ! SPECIAL VALUE FOR TETRAHEDRAL CARBON
  !
  !     GET IDEAL BOND LENGTHS FROM TABLE
  !
  RAB=0.
  RBC=0.
  !
  CALL BONDCON('PUT',ITA,ITB,MCAB,itt)
  CALL BIFIND(itt,KCB,NCB,J,NCBT,MAXCB,NEWPAR,FULL,2)
  IF(.NOT.NEWPAR) THEN
     RAB=BondEq(j)
  ENDIF
  !
  CALL BONDCON('PUT',ITCM,ITB,MCBC,itt)
  CALL BIFIND(itt,KCB,NCB,J,NCBT,MAXCB,NEWPAR,FULL,2)
  IF(.NOT.NEWPAR) THEN
     RBC=BondEq(j)
  ENDIF
  !D     WRITE(6,'(9I4,4F8.3)') IAMMFF,IBmmff,IC,ITA,ITB,ITCM,MCLASS,
  !D    . MCAB,MCBC,ANAT,ACON,RAB,RBC
  IF(RAB.EQ.0..OR.RBC.EQ.0.) THEN
     IF (WRNLEV.GE.2) THEN
        WRITE(OUTU,60) RAB,RBC
        WRITE(OUTU,'(A,9I4)') ' IAMMFF,IBmmff,IC,ITA,ITB,ITCM,'// &
             'MCLASS,MCAB,MCBC =',IAMMFF,IBmmff,IC,ITA,ITB,ITCM, &
             MCLASS,MCAB,MCBC
        !        IF(IZ.NE.0) WRITE(IZ,60) RAB,RBC
60      FORMAT(' empfc> Invalid RAB or RBC: ',2F8.3)
     ENDIF
     !        IF(IZ.NE.0) WRITE(IZ,'(A,9I4)') 
     !     .  ' IAMMFF,IBmmff,IC,ITA,ITB,ITCM,MCLASS,MCAB,MCBC =',
     !     .  IAMMFF,IBmmff,IC,ITA,ITB,ITCM,MCLASS,MCAB,MCBC
     ACON=0.
     RETURN
  ENDIF
  !
  !     IDEAL BOND ANGLE IS PROVIDED AT INPUT IN ANAT
  !
  !   GET THE APPROPRIATE ZP AND CP CONSTANTS
  !
  ZPA=0.
  cpbmmff=0.
  ZPC=0.
  MEL=12
  DO I=1,MEL
     !       IF(ATNUM(IAMMFF).EQ.ELNO(I)) ZPA=ZP(I)
     !       IF(ATNUM(IBmmff).EQ.ELNO(I)) cpbmmff=CP(I)
     !       IF(ATNUM(IC).EQ.ELNO(I)) ZPC=ZP(I)
     IF(IAMMFF.EQ.ELNO(I)) ZPA=ZP(I)
     IF(IBmmff.EQ.ELNO(I)) cpbmmff=CP(I)
     IF(IC.EQ.ELNO(I)) ZPC=ZP(I)
  enddo
  IF(ZPA.EQ.0..OR.cpbmmff.EQ.0..OR.ZPC.EQ.0.) then
     if(wrnlev.ge.2) THEN
        write(outu,'(a,2i8)') ' empfc> iammff,AtNum(iammff)=', &
             iammff,AtNum(iammff)
        write(outu,'(a,2i8)') ' empfc> ibmmff,AtNum(ibmmff)=', &
             ibmmff,AtNum(ibmmff)
        write(outu,'(a,2i8)') ' empfc> ic,AtNum(ic)=',ic,AtNum(ic) 
        ! CORRECTED BY JLB 29 SEP 95
        WRITE(OUTU,'(A,3F8.2)') ' empfc> ZPA, CPB, ZPC =  ', &
             ZPA,cpbmmff,ZPC
        if(ZPA.EQ.0.) ZPA=1.
        if(cpbmmff.EQ.0.) cpbmmff=1.
        if(ZPC.EQ.0.) ZPC=1.
        WRITE(OUTU,'(A)') ' empfc> reset values....'
        WRITE(OUTU,'(A,3F8.2)') ' empfc> ZPA, CPB, ZPC =  ', &
             ZPA,cpbmmff,ZPC
     endif
     call wrndie(0,'<empfc>','Invalid ZPA, CPB, or ZPC') 
  ENDIF
  !
  !   CALCULATE EMPIRICAL VALUE FOR THE ANGLE-BENDING FORCE CONSTANT
  !
  SCALE=1.75
  ALPHA=2.
  BETA=2.
  GAMMA=1.
  if((RAB*RBC).EQ.0.) CALL WRNDIE(-5,'<empfc>',' bad parameters')
  R=(RAB-RBC)**2/(RAB+RBC)**2
  ORABC=ANAT*DEGRAD
  ACON=SCALE*cpbmmff*ZPA*ZPC*ORABC**(-BETA)*EXP(-ALPHA*R)* &
       (RAB+RBC)**(-GAMMA)
  !     WRITE(6,*) IAMMFF,IBmmff,IC,ANAT,ACON
  !
  !   SCALE THE EMPIRICAL FORCE CONSTANT IF THIS ANGLE IS CONTAINED
  !   IN A SMALL RING
  !
  IF(MCLASS.EQ.3.OR.MCLASS.EQ.5.OR.MCLASS.EQ.6) THEN
     !
     !   THREE MEMBERED RING
     !
     ACON=ACON*0.05
  ELSE IF(MCLASS.EQ.4.OR.MCLASS.EQ.7.OR.MCLASS.EQ.8) THEN
     !
     !   FOUR MEMBERED RING
     !
     ACON=ACON*0.85
  ENDIF
  !D     WRITE(6,'(8F10.3)') RAB,RBC,ZPA,CPB,ZPC,ANAT,ACON
  !
  RETURN
END SUBROUTINE EMPFC

! ===========================================================
! SUBROUTINE GENCHGM: ASSIGN PARTIAL ATOMIC CHARGES
! SO AS TO REPRESENT THE ELECTROSTATIC POTENTIAL.
! INITIALIZE THE PARTIAL CHARGE ARRAY TO THE FORMAL CHARGE ASSIGNED
! TO EACH ATOM (USUALLY ZERO)
! ===========================================================
!
SUBROUTINE GENCHGM(IB,JB,FORMLQ, &
     NBOND,NATOM,ATNUMX)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  ! Jay Banks 25 Oct 95: changed argument name ATNUM to ATNUMX to avoid
  ! conflict with name in common (mmff.f90).
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use param
  use stream
  use mmffm
  implicit none
  !
  integer IB(*),JB(*)
  real(chm_real) FORMLQ(*)
  integer NBOND
  integer NATOM
  integer atnumx(*) ! atomic numbers
  !
  integer i1, i2, ia1, ia2, it1, it12, i
  integer it2, mclass, n, nb
  real(chm_real)  chg, ebci
  !
  LOGICAL NEWPAR,FULL,sp2sgl
  LOGICAL GOTPAR
  !
  !   COPY THE INPUT "PARTIAL ATOMIC CHARGES" TO THE FORMAL-CHARGE
  !   ARRAY
  DO I=1,NATOM
     FORMLQ(I)=PARTLQ(I)
  ENDDO
  !
  !   NOW ASSIGN THE PARTIAL ATOMIC CHARGES BY INSERTING BOND CHARGES
  !   SPECIFIED TO REPRESENT DEVIATIONS FROM FORMAL CHARGES
  loop700:DO nb=1,NBOND
     I1=IB(nb)
     I2=JB(nb)
     !   GET THE ATOMIC NUMBERS
     IA1=ATNUMX(I1)
     IA2=ATNUMX(I2)
     !   GET MMFF ATOM TYPES
     IT1=MTYPE(I1)
     IT2=MTYPE(I2)
     IF(IT1.EQ.IT2) cycle loop700
     MCLASS=0
     IF(SP2SGL(I1,I2)) MCLASS=1
     CALL BONDCON('PUT',IT1,IT2,MCLASS,IT12)
     !   SEE IF BOND CHARGES ARE IN TABLE
     !C##IF DEBUG
     !C    if(prnlev.gt.6) then
     !C      write(outu,'(a,3i10)') ' genchgm> IT12,ncq,ncqt=',IT12,ncq,ncqt
     !C        do n=1,ncq
     !C           write(outu,'(a,i5,i10)') ' genchgm> n,ICHGNDX(n)=',
     !C     &                                         n,ICHGNDX(n)
     !C        enddo
     !C      endif
     !C##ENDIF
     CALL BIFIND(IT12,ICHGNDX,NCQ,N,NCQT,MAXCB,NEWPAR,FULL,2)
     IF(.NOT.NEWPAR) THEN
        !   YES. HERE IT IS. THE STORED VALUE IS IN ELECTRONS
        !   AND REPRESENTS THE DELTA CHARGE FOR THE ATOM OF HIGHER
        !   MTYPE (-CHG IS THAT FOR THE ATOM OF LOWER MTYPE)
        CHG=BCI(N)
        IF(IT2.LT.IT1) CHG=-CHG
        GOTO 600
     ELSE
        !   NOPE. NOT THERE.  GENERATE A DEFAULT VALUE
        CALL DEFCHG(I1,I2,CHG,GOTPAR)
        IF(.NOT.FULL.AND.GOTPAR) THEN
           !   PUT INTO DATA ARRAY IN CASE THIS BOND TYPE RECURS.
           NCQT=NCQT+1
           BCI(NCQT)=CHG
           IF(IT2.LT.IT1) BCI(NCQT)=-BCI(NCQT)
           ICHGNDX(NCQT)=IT12
        ENDIF
     ENDIF
600  CONTINUE
     !
     !   NOW INSERT CHARGE INCREMENTS, CORRECTED BY PARTIAL-FORMAL-CHARGE
     !   ADJUSTMENTS IF NECESSARY
     !
     PARTLQ(I1)=PARTLQ(I1)-CHG
     PARTLQ(I2)=PARTLQ(I2)+CHG
     !
     IF(FORMLQ(I1).NE.0.0) THEN
        IF(LPBCI(IT1)) THEN
           PARTLQ(I1)=PARTLQ(I1)-FCADJ(IT1)*FORMLQ(I1)
           PARTLQ(I2)=PARTLQ(I2)+FCADJ(IT1)*FORMLQ(I1)
        ENDIF
     ENDIF
     IF(FORMLQ(I2).NE.0.0) THEN
        IF(LPBCI(IT2)) THEN
           PARTLQ(I2)=PARTLQ(I2)-FCADJ(IT2)*FORMLQ(I2)
           PARTLQ(I1)=PARTLQ(I1)+FCADJ(IT2)*FORMLQ(I2)
        ENDIF
     ENDIF
  enddo loop700
  !   DEBUG
  !    DO 200 I=1,20
  !     WRITE(OUTU,*) ' GENCHG: I,PARTLQ ',I,PARTLQ(I)
  !     IF(IZ.NE.0) WRITE(IZ,*) ' GENCHG: I,PARTLQ ',I,PARTLQ(I)
  ! 200 CONTINUE
  !
  RETURN
END SUBROUTINE GENCHGM

! ====================================================================
! INTEGER FUNCTION IBORDR : RETURN THE BOND ORDER BETWEEN ATOMS I AND J
! ====================================================================
INTEGER FUNCTION IBORDR(I,J)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use mmffm
  implicit none
  !
  integer i, j
  !
  integer nb
  !
  IBORDR=0
  DO nb=1,NBOND
     IF(IB(nb).EQ.I.AND.JB(nb).EQ.J .OR. &
          IB(nb).EQ.J.AND.JB(nb).EQ.I) THEN
        IBORDR=BondType(nb)
        RETURN
     ENDIF
  ENDDO
  !
  RETURN
END FUNCTION IBORDR

! ======================================================================
! SUBROUTINE KBONDM : ASSIGNS PARAMETERS TO BONDS
! FOR THE MMFF FORCE FIELD
! ACCORDING TO THE TYPE NUMBERS OF THE ATOMS FORMING THEM.
! ======================================================================
!
SUBROUTINE KBONDM(INIT)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 22-NOV-95: Added IF (PRNLEV.GE.2) to WRITE statement.
  !
  !  Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  !  WRITE statement followed by CALL WRNDIE.  Also nested IF
  !  (PRNLEV.GE.2) inside IF (SP2SGL(I,K)) rather than combining with
  !  .AND., so that MCLASS=1 will be executed even if nothing is printed.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  use code
  use consta
  !...##INCLUDE '~/charmm_fcm/debug.f90'
  use exfunc
  use mmffm
  use param
  use psf
  use stream
  use string
  implicit none
  !
  integer i, init, itt
  integer j, k
  !     integer iti, itk !?RCZ940209 - unused
  integer mclass
  integer nb
  real(chm_real)  blzero, fcon
  !
  LOGICAL GOTPAR,NEWPAR,FULL,SP2SGL
  !
  !...##IF DEBUG
  !      call getenv('KBONDM_DEBUG',SCRTCH)
  !      DEBUG=SCRTCH.ne.' '
  !...##ENDIF
  !
  !  ASSIGN PARAMETERS TO BONDS.  L IS THE INDEX TO THE IR-ARRAY, NBOND IS
  !  THE INDEX TO BK, BL, DIP ARRAYS, NH IS AN INDEX FOR EV. DEP.
  !
  loop3700:do nb=1,NBOND
     i=IB(nb)
     k=JB(nb)
     !
     !...##IF DEBUG
     !      if(DEBUG) then
     !        write(OUTU,'(a,4i5)') 'KBONDM_DEBUG> nb,i,k,BondType=',
     !     &                                       nb,i,k,BondType(nb)
     !      endif
     !...##ENDIF
     !
     !   FORM THE PACKED INDEX
     !
     MCLASS=0
     IF(SP2SGL(I,K)) THEN
        IF (prnlev.ge.2) THEN
           WRITE(OUTU,'(A,2(A,1X))') ' Csp2-Csp2 SINGLE BOND ', QNAME(I), &
                QNAME(K)
        ENDIF
        !       IF(IZ.NE.0)
        !    .  WRITE(IZ,'(A,2(A,1X))') ' Csp2-Csp2 SINGLE BOND ', QNAME(I),
        !    .  QNAME(K)
        MCLASS=1
     ENDIF
     MDBOND(nb)=MCLASS
     CALL BONDCON('PUT',MTYPE(I),MTYPE(K),MCLASS,itt)
     CALL BIFIND(itt,KCB,NCB,J,NCBT,MAXCB,NEWPAR,FULL,2)
     IF(.NOT.NEWPAR) THEN
        icb(nb)=j
        !D        WRITE(OUTU,'(A,3I5,I10,2F8.3)') ' MTYPE I,K; nb; J; BK; BL:',
        !D    .   MTYPE(I),MTYPE(K),nb,J,BondFC(ICB(nb)),BondEq(ICB(nb))
        cycle loop3700
     ELSE IF(FULL) THEN
        IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I6,A/A)') &
             ' ***** Too many bond-stretch parameters generated '// &
             '... max = ',MAXCB,' *****',' *** Execution terminated ****'
        CALL WRNDIE(-5,'<kbondm>','Too many bond-stretch'// &
             ' parameters')
     ELSE
        !  IF THIS POINT IS REACHED NO CONSTANT CAN BE FOUND.  CALL
        !  DEFAULT-PARAMETERS ROUTINE
        CALL DEFBND(I,K,MCLASS,FCON,BLZERO,GOTPAR)
        IF(GOTPAR) THEN
           !    PUT PARAMETERS INTO DATA STRUCTURE
           !    ALSO STORE IN DATA BASE IN CASE ANOTHER INTERACTION OF THIS
           !    TYPE IS ENCOUNTERED
           NCBT=NCBT+1
           KCB(NCBT)=itt
           BondFC(NCBT)=FCON*MDAKCAL
           BondEq(NCBT)=BLZERO
           icb(nb)=NCBT
        ELSE
           !   ISSUE WARNING
           !
           !         ITI=MIN(MTYPE(I),MTYPE(K)) !? RCZ940209 - unused ?
           !         ITK=MAX(MTYPE(I),MTYPE(K)) !? RCZ940209 - unused ?
           if(wrnlev.ge.2) WRITE(OUTU,3400) &
                QNAME(I),QNAME(K),MTYPE(I),MTYPE(K)
           !     IF(IZ.NE.0) WRITE(IZ,3400) QNAME(I),QNAME(K),MTYPE(I),MTYPE(K)
3400       FORMAT(/' *** FATAL ERROR ERROR - CONSTANTS FOR BOND ', &
                A,1X,A,' (TYPE',I3,' - ',I3,') NOT FOUND')
           !   MARK INIT=2 TO INDICATE FATAL ERROR
           INIT=2
        ENDIF
     ENDIF
  enddo loop3700
  !
  RETURN
END SUBROUTINE KBONDM

!======================================================================
! SUBROUTINE KOMEGAM : ASSIGNS TORSIONAL CONSTANTS V1,V2 & V3 (TorFC)
!  THIS SUBROUTINE ASSIGNS TORSIONAL CONSTANTS V1,V2 & V3 TO THE ANGLES
!  ACCORDING TO THEIR TYPE, KV.  BECAUSE A MOLECULE CAN CONTAIN MANY
!  TORSIONAL ANGLES AND SINCE MANY OF THESE HAVE THE SAME CONSTANT, THE
!  PROGRAM STORES ONLY ONE SET OF CONSTANTS FOR EACH ANGLE TYPE PRESENT
!  IN THE MOLECULE.  THEY ARE STORED AS TCON1, TCON2 & TCON3.
!======================================================================
!
SUBROUTINE KOMEGAM(INIT)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  16 Oct 95 Jay Banks: changed variable name "IB" to "IBmmff" to avoid
  !                       conflict with CHARMM name
  !
  !  Jay Banks 25 Oct 95: changed variable IA to IAmmff, to avoid conflict
  !  with variable in common (mmff.f90).
  !
  !  Jay Banks 22 NOV 95: added IF (PRNLEV.GE.2) to WRITE statements.
  !
  !  Jay Banks 27 Nov 95: removed IF (PRNLEV.GE.2) from internal WRITE;
  !  changed to IF (WRNLEV.GE.2) for WRITEs followed by CALL WRNDIE.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  use code
  use consta
  use exfunc
  use mmffm
  use param
  use psf
  use stream
  use string
  implicit none
  !
  integer i, iammff, IBmmff, ic, id
  integer ina, inb1, inc, ind, init, iold, ita
  integer itb, itcmmff, itd, itry, j
  integer(chm_int8) :: k, k0
  integer l, last, maxtry
  parameter (MAXTRY=5)
  ! (jlb 02 oct 95) NUMBER OF "REAL" ATOM TYPES
  integer MX
  parameter (MX=MAXDEFI/2)
  integer mclass
  real(chm_real)  val1, val2, val3
  !
  integer ncbut, ncprop
  COMMON/CYCLC/NCBUT,NCPROP
  LOGICAL GOTPAR,NEWPAR,FULL
  LOGICAL SP2SGL,ARBOND
  !
  !  ASSIGN TORSIONAL CONSTANTS.  IF AN ANGLE TYPE IS PRESENT, PLACE THE
  !  V1,V2 & V3 INTO THE TCON ARRAYS (THIS WAY ONLY THOSE CONSTANTS USED
  !  IN THE CALCULATIONS ARE STORED).
  !
  IF(NPHI.LE.0) RETURN
  IF(NPHI.GT.MAXP) THEN
     IF(WRNLEV.GT.-5) WRITE(SCRTCH,'(A,2(1X,I5))') &
          'To many torsions:NPHI,MAXP=',NPHI,MAXP
     CALL WRNDIE(-5,'<komegam>',SCRTCH(:40))
     RETURN
  ENDIF
  !...##IF DEBUG
  !      call getenv('DEBUG_KOMEGAM',SCRTCH)
  !      DEBUG=SCRTCH.ne.' '
  !...##ENDIF
  IOLD=0
  loop5800:DO I=1,NPHI
     IAMMFF=IP(I)
     IBmmff=JP(I)
     IC=KP(I)
     ID=LP(I)
     !      WRITE(OUTU,*) IAMMFF,IBmmff,IC,ID,I
     !     *** MCLASS = 0 IS A FLAG MEANING NORMAL TORSIONAL CONSTANTS ***
     MCLASS=0
     MDOMEG(I)=0
     IF(NCBUT.EQ.0) GOTO 2800
     !     *** 4-MEMBERED RING ? ***
     IF ((ITAB(6,IAMMFF).LE.1).OR.(ITAB(6,ID).LE.1)) GOTO 2800
     IF(CONN12(IAMMFF,IC).EQ.1) GOTO 2800
     IF(CONN12(IBmmff,ID).EQ.1) GOTO 2800
     IF(CONN12(IAMMFF,ID).EQ.1) THEN
        !
        !         FOUR-MEMBERED RING DETECTED
        !     *** MCLASS = 4 IS A FLAG DIRECTING USE OF THE SPECIAL TORSION
        !         CONSTANTS FOR 4-MEMBERED RINGS ***
        !       WRITE(OUTU,*) ' MCLASS=4',IAMMFF,IBmmff,IC,ID
        MCLASS=4
        MDOMEG(I)=4
        GOTO 3700
     ENDIF
2800 CONTINUE
     !
     ! * IF A PAIR OF ATOMS ARE SP2 CARBONS, THEN CHECK TO SEE
     !   IF THE BOND CONNECTING THESE ATOMS IS SINGLE.  IF SO, SET
     !   MCLASS=1 OR MCLASS = 2  AS A FLAG TO USE SPECIAL TORSIONAL CONSTANTS
     !   SO THAT SYSTEMS LIKE C=C-C=C ARE NOT TREATED LIKE =C=C=C=C= **
     !   (PARTS TAKEN FROM 'B A K M O D', BY W. CLARK STILL)
     !
     IF(ARBOND(IBmmff,IC)) GOTO 3700
     IF(IBORDR(IBmmff,IC).NE.1) GOTO 3700
     IF(SP2SGL(IBmmff,IC)) THEN
        MCLASS=1
        MDOMEG(I)=1
        !D     WRITE(OUTU,*) ' CENTRAL SP2-SP2 SINGLE BOND ',IAMMFF,IBmmff,IC,ID
        GOTO 3700
     ENDIF
     IF(SP2SGL(IAMMFF,IBmmff).OR.SP2SGL(IC,ID)) THEN
        MCLASS=2
        MDOMEG(I)=2
        !D     WRITE(OUTU,*) ' WING SP2-SP2 SINGLE BOND ',IAMMFF,IBmmff,IC,ID
        GOTO 3700
     ENDIF
3700 CONTINUE
     IOLD=IAMMFF
     ITRY=1
     ITA=MTYPE(IAMMFF)
     ITB=MTYPE(IBmmff)
     itcmmff=MTYPE(IC)
     ITD=MTYPE(ID)
     !
     !     *** FOR LINEAR ANGLES SKIP CALC FOR TORSIONAL ENERGY ***
     !
     IF(MLINBA(ITB).GT.0.OR.MLINBA(ITCmmff).GT.0) THEN
        !      IF(ITB.EQ.4.OR.ITCmmff.EQ.4.OR.ITB.EQ.42.OR.
        !     .                ITCmmff.EQ.42) THEN
        icp(i)=0
        GOTO 5600
     ENDIF
     !
     IF(MCLASS.EQ.0.AND.CONN13(IAMMFF,ID).EQ.1 &
          .AND.CONN12(IBmmff,ID).EQ.0. &
          .AND.CONN12(IAMMFF,IC).EQ.0) THEN
        IF(ITA.EQ.1.OR.ITB.EQ.1.OR.itcmmff.EQ.1.OR.ITD.EQ.1) THEN
           !
           !  SATURATED FIVE-MEMBERED RING DETECTED
           !  *** MCLASS = 5 IS A FLAG DIRECTING USE OF THE SPECIAL TORSION
           !  CONSTANTS FOR 5-MEMBERED RINGS ***
           !           WRITE(OUTU,'(A,4I4,4X,4I4)') ' MCLASS=5',IAMMFF,IBmmff,
           !     .     IC,ID,ITA,ITB,itcmmff,ITD
           MCLASS=5
           MDOMEG(I)=5
        ENDIF
     ENDIF
     LAST=0
     !   FIRST TRY TO FIND CONSTANT DIRECTLY
     INA=ITA
     inb1=ITB
     INC=itcmmff
     IND=ITD
     ! (jlb 02 oct 95) CHECK FOR EXCEEDING "REAL" NUMBER OF ATOM TYPES; FALL
     ! TO NEXT LEVEL IF IT DOES
     if (INA.GE.MX .OR. INB1.GE.MX .OR. INC.GE.MX .OR. IND.GE.MX) &
          GOTO 3000
     GOTO 3100
     !   NOT FOUND, SO APPLY ANY ATOM-TYPE EQUIVALENCES
3000 ITRY=ITRY+1
     INA=LEQUIV(ITA,ITRY,3)
     inb1=LEQUIV(ITB,ITRY,1)
     INC=LEQUIV(itcmmff,ITRY,1)
     IND=LEQUIV(ITD,ITRY,4)
     ! (jlb 02 oct 95) IF STILL EXCEED "REAL" NUMBER OF ATOM TYPES AT THIS
     ! LEVEL, DIE
     if (INA.GE.MX .OR. INB1.GE.MX .OR. INC.GE.MX .OR. IND.GE.MX) then
        IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I4/5(A,I4))') &
             'Equivalent atom type equals or exceeds ',MX, &
             'ITRY=',ITRY,',INA=',INA,',INB=',INB1,',INC=',INC,',IND=',IND
        CALL WRNDIE(-5,'<komegam>','Equivalent atom type exceeds'// &
             ' maximum')
     endif
3100 CONTINUE
     !   COMPUTE PACKED INDEX FOR THIS DIHEDRAL ANGLE TYPE
     CALL OMEGCON('PUT',INA,inb1,INC,IND,MCLASS,K)
     !D     WRITE(OUTU,'(4I5,A,I15)') INA,inb1,INC,IND,'  K =',K
     !   RECORD THE PRIMARY VALUE FOR THE PACKED INDEX
     IF(ITRY.EQ.1) K0=K
     !   IF NOT THE FIRST TRY, MAKE SURE THAT THE PACKED INDEX
     !   BEING SOUGHT IS A NEW ONE
     IF(ITRY.GT.1.AND.K.EQ.LAST) GOTO 5200
     LAST=K
     CALL BIFIND8(K,KCP,NCP,L,NCPT,MAXCP,NEWPAR,FULL,2)
     IF(.NOT.NEWPAR) THEN
        icp(i)=l   ! Pointer for torsion parameters
        !D       WRITE(OUTU,*) ' NEW ',IAMMFF,IBmmff,IC,ID,V1(L),V2(L),V3(L)
        GOTO 5600
     ELSE
        GOTO 5200
     ENDIF
5200 CONTINUE
     IF(ITRY.LT.MAXTRY) GOTO 3000
     !
     !  IF THIS POINT IS REACHED NO TORSIONAL CONSTANT IS FOUND.
     !  CALL DEFAULT-PARAMETERS ROUTINE
     CALL DEFTOR(IAMMFF,IBmmff,IC,ID,VAL1,VAL2,VAL3,MCLASS,GOTPAR)
     IF(GOTPAR) THEN
        !    PUT PARAMETERS INTO DATA STRUCTURE
        !    ALSO STORE IN DATA BASE IN CASE ANOTHER INTERACTION OF THIS
        !    TYPE IS ENCOUNTERED
        IF(NCPT.GE.MAXCP) THEN
           IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I6,A/A)') &
                ' ***** Too many dihedral-angle parameters generated '// &
                '... max = ',MAXCP,' *****',' *** Execution terminated ****'
           CALL WRNDIE(-5,'<komegam>','Too many dihedral-angle'// &
                ' parameters')
        ENDIF
        NCPT=NCPT+1
        KCP(NCPT)=K0
        if(val1.ne.0. .or. val2.ne.0. .or. val3.ne.0.) then
           CPC(3 * NCPT - 2) = VAL1 * 0.5
           CPC(3 * NCPT - 1) = VAL2 * 0.5
           CPC(3 * NCPT) = VAL3 * 0.5
           icp(i)=NCPT
           !
           !-------------------------------------------------------
           ! if everything is OK this should allow to use
           ! charmm EPHI routine to calculate torsion energy
           !
           j=3*(NCPT-1)
           TorPr(j+1)=-1   ! = CPD in CHARMM (periodicity)
           TorPr(j+2)=-2   ! - indicates that there are more terms
           TorPr(j+3)=3    ! compare with parmio.src
           !                       IF (KCP(I).EQ.KCP(I+1)) CPD(I) = - ABS(CPD(I))
           TorPh(j+1)=0.   ! = CPB in CHARMM (phase)
           TorPh(j+2)=PI
           TorPh(j+3)=0.
           !-------------------------------------------------------
           !
        else
           icp(i)=0
        endif
     ELSE
        !   USE ZERO VALUES FOR MISSING TORSION PARAMETERS
        icp(i)=0
        !
        !   ISSUE WARNING
        !
        if(wrnlev.ge.2) WRITE(OUTU,5300) &
             QNAME(IAMMFF),QNAME(IBmmff),QNAME(IC),QNAME(ID), &
             ITA,ITB,itcmmff,ITD
        !     IF(IZ.NE.0)
        !    .  WRITE(IZ,5300) QNAME(IAMMFF),QNAME(IBmmff),QNAME(IC),
        !    .   QNAME(ID),ITA,ITB,itcmmff,ITD
5300    FORMAT (/' *** TORSION CONSTANTS MISSING FOR ANGLE ', &
             4(1X,A),' (TYPE ',4I3,')'/ &
             '     WARNING - ZERO VALUES WILL BE ASSUMED')
        !   SET NONFATAL-MISSING-PARAMETERS FLAG
        INIT=1
        !
        !    ALSO STORE IN DATA BASE IN CASE ANOTHER PARAMETER OF THIS TYPE
        !    IS ENCOUNTERED - IF THERE IS ROOM
        !
        IF(NCPT.LT.MAXCP) THEN
           NCPT=NCPT+1
           KCP(NCPT)=K0
           CPC(3 * NCPT - 2) = 0.
           CPC(3 * NCPT - 1) = 0.
           CPC(3 * NCPT) = 0.
        ENDIF
     ENDIF
5600 CONTINUE
     !...##IF DEBUG
     !      if(DEBUG) then
     !        write(OUTU,'(a,8i5)')
     !     &     ' DEBUG_KOMEGAM> I,IAMMFF,IBmmff,IC,ID,MDOMEG(I),ICP(I)=',
     !     &                         I,IAMMFF,IBmmff,IC,ID,MDOMEG(I),ICP(I)
     !        if(ITRY.ne.1) write(OUTU,'(a,i5)') ' DEBUG_KOMEGAM> ITRY=',ITRY
     !        if(MCLASS.ne.MDOMEG(I))
     !     &   write(OUTU,'(a,2i5)') ' DEBUG_KOMEGAM> MCLASS,MDOMEG(I)=',
     !     &                                          MCLASS,MDOMEG(I)
     !      endif
     !...##ENDIF
  enddo loop5800
  !
  RETURN
END SUBROUTINE KOMEGAM

! ======================================================================
! SUBROUTINE KTHETAM : ASSIGNS BENDING CONSTANTS (REGULAR & OUT-PL-BEND)
! ACCORDING TO THE ATOM TYPES COMPRISING THE ANGLE.
! ======================================================================
!
SUBROUTINE KTHETAM(AtNum,MTYPE, &
     NBOND,IB,JB,MDBOND,ITAB, &
     NCT,NCTT,NTHETA,IT,JT,KT,LTHETA, &
     MDTHET,ICT,KCT,AnglEq,AnglFC, &
     NCOOP,NCOOPT,ICOOP,KCOOP,OoplFC, &
     NCSB,NCSBT,ICSTBN,KCSTBN,StrbList,STBNP, &
     INIT,MDSTBN)
  !23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   16 Oct 95 Jay Banks: For consistency with CHARMM variable names,
  !                        changed IB to IBmmff, IBOND, JBOND to IB, JB,
  !                        ITHETA, JTHETA, KTHETA to IT, JT, KT, and
  !                        dimension MAXTH to MAXT
  !
  !  Jay Banks 25 Oct 95: changed variable IA to IAmmff, to avoid conflict
  !  with variable in common (mmff.f90).
  !
  !  Jay Banks 22 Nov 95: added IF (PRNLEV.GE.2) to WRITE statements.  In
  !  another WRITE, changed (OUTU,*) to explicit format.
  !
  !  Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  !  WRITE statements followed by CALL WRNDIE.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  !  01 Dec 95 Jay Banks: added code after CALL STBNCON, to store MCLASS
  !  in the MDSTBN array (common block PSFI_MM, in mmff.f90, but PASSED to
  !  this routine, as are MDBOND, MDTHET, etc.).  Added MDSTBN to argument
  !  list.
  !
  use chm_kinds
  !      integer MXCMSZ
  !      parameter (MXCMSZ=64)
  !...##INCLUDE '~/charmm_fcm/debug.f90'
  use dimens_fcm
  !#INCLUDE '~/charmm_fcm/code.f90'
  use consta
  use exfunc
  !#INCLUDE '~/charmm_fcm/param.f90'
  !#INCLUDE '~/charmm_fcm/psf.f90'
  use stream
  use string
  !
  implicit none
  integer MX
  parameter(MX=MAXDEFI/2)
  integer AtNum(*),MTYPE(*)
  integer NBOND,IB(*),JB(*),MDBOND(*),ITAB(6,*)
  integer NCT,NCTT,NTHETA,IT(*),JT(*),KT(*),LTHETA(*)
  integer MDTHET(*),ICT(*),KCT(*)
  integer INIT
  real(chm_real)  AnglEq(*),AnglFC(*),OoplFC(*),STBNP(2,MAXCT)
  integer NCOOP,NCOOPT,ICOOP(*),KCOOP(*)
  integer NCSB,NCSBT,ICSTBN(*),KCSTBN(*),StrbList(2,*)
  INTEGER MDSTBN(*)
  !
  integer iammff, iaa, ibmmff, ibb
  integer ic, icc, id, ie, ina, inbmmff, inc, ind
  integer itmmff, it0, ita, itb, itcmmff, itd
  integer itry, j, k, kpsb
  integer ksb, kpmmff
  integer last, m1, maxtry
  integer mcij, mcijk, mcjk, mclass, mcin
  integer nb, nb1, nb2, nth
  real(chm_real)  azero, aconi, copbk, fabc, fcba
  !
  integer ncbut,ncprop
  COMMON/CYCLC/NCBUT,NCPROP
  LOGICAL GOTPAR,NEWPAR,FULL
  !
  integer NonFatal
  parameter(NonFatal=1)
  !
  !...##IF DEBUG
  !      call getenv('DEBUG_KTHETAM',SCRTCH)
  !      DEBUG=SCRTCH.ne.' '
  !...##ENDIF
  !
  !  ASSIGN CONSTANTS TO TH (FOR ANGLE IAMMFF-IBmmff-IC)

  IF (NTHETA.EQ.0) RETURN
  !
  !   INITIALIZE COUNTER FOR STRETCH-BEND PARAMETERS GENERATED USING
  !   DEFAULT ROUTINE DEFSTBN.  COUNTER IS USED TO LOAD GENERATED
  !   PARAMETERS INTO THE "LINEAR" PORTION OF THE STBN PARAMETERS ARRAY,
  !   BUT IS NOT PROVIDED IN SUBSEQUENT CALLS TO BIFIND.  THUS,
  !   MISSING STRETCH-BEND PARAMETERS ARE FRESHLY GENERATED BUT ARE
  !   NOT SEARCHED AGAINST IN FUTURE PARAMETER REQUESTS ... THERE MIGHT
  !   BE TOO MANY FOR THE "LINEAR SEARCH" TO BE MORE EFFICIENT THAN SIMPLY
  !   RECALCULATING.
  !
  KPSB=NCSB
  loop950: DO nth=1,NTHETA
     !     CALL UNPACK(nglList(,I),ID,IAMMFF,IBmmff,IC)
     ID=LTHETA(nth)
     IAMMFF=IT(nth)
     IBmmff=JT(nth)
     IC=KT(nth)
     !      WRITE(OUTU,5600) nth,IAMMFF,IBmmff,IC,ID
     ! 5600 FORMAT(I6,24X,4I6)
     !
     !  FIND THE NUMBER OF THE BOND CONSTANT FOR IAMMFF-IBmmff FROM
     !  bond list (CALC IN  KBONDM)
     !  FIND BOND CONSTANT NUMBER FOR IBmmff-IC
     !
     nb1=0
     nb2=0
     nb=0
     do while ((nb1.eq.0 .or. nb2.eq.0) .and. nb.lt.NBOND)
        nb=nb+1
        if(IB(nb).eq.IBmmff .and. JB(nb).eq.IAMMFF .or. &
             IB(nb).eq.IAMMFF .and. JB(nb).eq.IBmmff) nb1=nb
        if(IB(nb).eq.IBmmff .and. JB(nb).eq.IC .or. &
             IB(nb).eq.IC .and. JB(nb).eq.IBmmff) nb2=nb
     enddo
     if(nb1.eq.0) then
        SCRTCH=' bond '//qname(iammff)//'--'//qname(IBmmff)// &
             ' not found'
        CALL WRNDIE(-5,'<kthetam>',SCRTCH(:38))
     endif
     if(nb2.eq.0) then
        SCRTCH=' bond '//qname(ic)//'--'//qname(IBmmff)//' not found'
        CALL WRNDIE(-5,'<kthetam>',SCRTCH(:38))
     endif
     StrbList(1,nth)=NB1
     StrbList(2,nth)=NB2
     MCIJ=MDBOND(NB1)
     MCJK=MDBOND(NB2)
     !
     !   ASSIGN MCLASS NOW IF EITHER NB1 OR NB2 ARE CSP2-CSP2 SINGLE BONDS
     !
     MCLASS=MDBOND(NB1)+MDBOND(NB2)
     IF(NCBUT.EQ.0.AND.NCPROP.EQ.0) GOTO 6100
     !  DETERMINE WHETHER ANGLE IAMMFF-IBmmff-IC BELONGS 4-MEMBER OR 3-MEMBER RING
     !   MCLASS=4 FOR 4-MEMBERED RING, MCLASS=3 FOR 3-MEMBERED RING
     MCIN=MCLASS
     IF(CONN12(IAMMFF,IC).EQ.1) THEN
        MCLASS=3
        IF(MCIN.EQ.1) MCLASS=5
        IF(MCIN.EQ.2) MCLASS=6
        GOTO 6100
     ENDIF
     kpmmff=ITAB(6,IC)        ! number of atoms bonded to IC
     loop6000: Do M1=1,kpmmff
        IE=ITAB(M1,IC)
        IF(IE.EQ.IBmmff) cycle loop6000
        IF(CONN12(IAMMFF,IE).EQ.1) THEN
           MCLASS=4
           IF(MCIN.EQ.1) MCLASS=7
           IF(MCIN.EQ.2) MCLASS=8
           cycle loop6000
        ENDIF
     enddo loop6000
6100 CONTINUE
     !D     WRITE(6,'(A,6I4)') ' IAMMFF,IBmmff,IC,NB1,NB2,MCLASS =',
     !D    . IAMMFF,IBmmff,IC,NB1,NB2,MCLASS
     MDTHET(nth)=MCLASS
     IAA=MTYPE(IAMMFF)
     IBB=MTYPE(IBmmff)
     ICC=MTYPE(IC)
     ITRY=1
     MAXTRY=5
     LAST=0
     !   FIRST TRY TO FIND CONSTANT DIRECTLY
     INA=IAA
     inbmmff=IBB
     INC=ICC
     ! (jlb 11 oct 95) CHECK FOR EXCEEDING "REAL" NUMBER OF ATOM TYPES; FALL
     ! TO NEXT LEVEL IF IT DOES
     if (INA.GE.MX.OR.INBmmff.GE.MX.OR.INC.GE.MX) &
          GOTO 6200
     GOTO 6300
     !   NOT FOUND, SO APPLY ANY ATOM-TYPE EQUIVALENCES
6200 CONTINUE
     ITRY=ITRY+1
     !      write(outu,'(a,6i5)') ' ang:',iammff,ibmmff,ic,iaa,ibb,icc
     INA=LEQUIV(IAA,ITRY,2)
     inbmmff=LEQUIV(IBB,ITRY,1)
     INC=LEQUIV(ICC,ITRY,2)
     ! (jlb 11 oct 95) IF STILL EXCEED "REAL" NUMBER OF ATOM TYPES AT THIS
     ! LEVEL, DIE
     if (INA.GE.MX.OR.INBmmff.GE.MX.OR.INC.GE.MX) then
        IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I4/4(A,I4))') &
             'ANGLE: Equivalent atom type equals or exceeds ',MX, &
             'ITRY=',ITRY,',INA=',INA,',INB=',INBmmff, &
             ',INC=',INC
        CALL WRNDIE(-5,'<kthetam>','Equivalent atom type exceeds'// &
             ' maximum for angle')
     endif
6300 CONTINUE
     !
     !   FORM THE PACKED, CANONICALLY ORDERED INDEX
     !
     CALL THETCON('PUT',INA,inbmmff,INC,MCLASS,itmmff)
     !   RECORD THE PRIMARY VALUE OF THE PACKED INDEX
     IF(ITRY.EQ.1) IT0=itmmff
     !   IF THIS IS NOT THE FIRST TRY, MAKE SURE THE PACKED INDEX
     !   TO BE SOUGHT IS A NEW ONE
     IF(ITRY.GT.1.AND.itmmff.EQ.LAST) GOTO 7800
     LAST=itmmff
     !  FIND THE ANGLE IN THE KCT LIST
     CALL BIFIND(itmmff,KCT,NCT,J,NCTT,MAXCT,NEWPAR,FULL,2)
     IF(NEWPAR) GOTO 7800
     ict(nth)=j
     IF(AnglFC(j).LE.0.) THEN
        !
        !   CALCULATE AnglFC FROM  EMPIRICAL FIT TO SPECTROSCOPICALLY
        !   OBSERVED ANGLE-BENDING FORCE CONSTANTS
        !
        AZERO=AnglEq(j)/degrad
        CALL DEFANG(IAMMFF,IBmmff,IC,MCLASS,MCIJ,MCJK,ACONI,AZERO, &
             GOTPAR)
        IF(ACONI.GT.0.) THEN
           !
           !   ALSO STORE IN THE DATA BASE
           !
           NCTT=NCTT+1
           AnglFC(NCTT)=ACONI*MDAKCAL
           AnglEq(NCTT)=AZERO*degrad
           KCT(NCTT)=IT0
           ict(nth)=NCTT
        ENDIF
     ENDIF
     GOTO 8100
7800 CONTINUE
     !  TRY AGAIN IF EQUIVALENCIES HAVE NOT YET BEEN EXHAUSTED
     IF(ITRY.LT.MAXTRY) GOTO 6200
     !  IF THIS POINT IS REACHED, NO ANGLE CONSTANTS ARE FOUND
     !  TIME TO CALL THE DEFAULT-PARAMETERS ROUTINE
     AZERO=0.
     CALL DEFANG(IAMMFF,IBmmff,IC,MCLASS,MCIJ,MCJK,ACONI,AZERO,GOTPAR)
     IF(GOTPAR) THEN
        !    PUT PARAMETERS INTO DATA STRUCTURE
        !    ALSO STORE IN DATA BASE IN CASE ANOTHER INTERACTION OF THIS
        !    TYPE IS ENCOUNTERED
        IF(FULL) THEN
           IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I6,A/A)') &
                ' ***** Too many angle-bending parameters generated '// &
                '... max = ',MAXCT,' *****',' ***** Execution terminated'// &
                ' *****'
           CALL WRNDIE(-5,'<kthetam>','Too many angle-bending'// &
                ' parameters')
        ENDIF
        NCTT=NCTT+1
        AnglFC(NCTT)=ACONI*MDAKCAL
        AnglEq(NCTT)=AZERO*degrad
        KCT(NCTT)=IT0
        ict(nth)=NCTT
        GOTO 8100
     ELSE
        !
        !   FATAL ERROR - NOT EVEN DEFAULT PARAMETERS ARE AVAILABLE
        !   FOR THIS ANGLE TYPE!!!
        !
        if(wrnlev.ge.2) WRITE(OUTU,7920) QNAME(IAMMFF),QNAME(IBmmff), &
             QNAME(IC),MTYPE(IAMMFF),MTYPE(IBmmff),MTYPE(IC),MCLASS
        !        IF(IZ.NE.0)
        !     .  WRITE(IZ,7920) QNAME(IAMMFF),QNAME(IBmmff),QNAME(IC),
        !     .   MTYPE(IAMMFF),MTYPE(IBmmff),MTYPE(IC),MCLASS
7920    FORMAT(/' ***** CONSTANTS NOT FOUND FOR ANGLE ', &
             3(A,1X),' *****' &
             /'       (TYPES',3I4,' WITH ANGLE CLASS =',I2,')'/ &
             ' ***** Execution Ending *****')
        CALL WRNDIE(-5,'<kthetam>','No angle-bending'// &
             ' parameter can be assigned')
     ENDIF
8100 CONTINUE
     !D     WRITE(6,'(7I5,2F10.5)') IAMMFF,IBmmff,IC,IAA,IBB,ICC,MDTHET(nth),
     !D    . AnglFC(nth),ANAT(nth)
     !
     !  CHECK FOR OUT-PL-BEND, IF PRESENT SEARCH FOR PARAMETERS
     !
     IF(ID.LE.0) GOTO 8600
     ITA=MTYPE(IAMMFF)
     ITB=MTYPE(IBmmff)
     itcmmff=MTYPE(IC)
     ITD=MTYPE(ID)
     !
     !   FIRST TRY TO FIND THE CONSTANT DIRECTLY
     ITRY=1
     MAXTRY=5
     INA=ITA
     inbmmff=ITB
     INC=itcmmff
     IND=ITD
     ! (jlb 02 oct 95) CHECK FOR EXCEEDING "REAL" NUMBER OF ATOM TYPES; FALL
     ! TO NEXT LEVEL IF IT DOES
     if (INA.GE.MX.OR.INBmmff.GE.MX.OR.INC.GE.MX.OR.IND.GE.MX) &
          GOTO 8200
     GOTO 8300
     !  NOT FOUND, SO APPLY ATOM-TYPE EQUIVALENCES
8200 CONTINUE
     ITRY=ITRY+1
     !      write(outu,'(a,8i5)') ' oop:',iammff,ibmmff,ic,id,ita,itb,itcmmff,itd
     INA=LEQUIV(ITA,ITRY,2)
     inbmmff=LEQUIV(ITB,ITRY,1)
     INC=LEQUIV(itcmmff,ITRY,2)
     IND=LEQUIV(ITD,ITRY,2)
     ! (jlb 02 oct 95) IF STILL EXCEED "REAL" NUMBER OF ATOM TYPES AT THIS
     ! LEVEL, DIE
     if (INA.GE.MX.OR.INBmmff.GE.MX.OR.INC.GE.MX.OR.IND.GE.MX) then
        IF (WRNLEV.GE.2) WRITE(OUTU,'(A,I4/5(A,I4))') &
             'OOP: Equivalent atom type equals or exceeds ',MX, &
             'ITRY=',ITRY,',INA=',INA,',INB=',INBmmff, &
             ',INC=',INC,',IND=',IND
        CALL WRNDIE(-5,'<kthetam>','Equivalent atom type exceeds'// &
             ' maximum for oop')
     endif
8300 CONTINUE
     CALL OOPCON('PUT',INA,inbmmff,INC,IND,itmmff)
     !   STORE PRIMARY VALUE FOR THE PACKED INDEX
     !      WRITE(6,*) itmmff,INA,inbmmff,INC,IND,ITRY
     IF(ITRY.EQ.1) IT0=itmmff
     CALL BIFIND(itmmff,KCOOP,NCOOP,J,NCOOPT,MAXCT,NEWPAR,FULL,2)
     !      WRITE(6,*) ' NEWPAR = ',NEWPAR
     IF(FULL) THEN
        if(wrnlev.ge.2) WRITE(OUTU,'(A,I6,A/A)') &
             ' ***** Too many out-of-plane bending parameters '// &
             '... max = ',MAXCT,' *****',' ***** Execution terminated'// &
             ' *****'
        CALL WRNDIE(-5,'<kthetam>','Too many out-of-plane '// &
             'parameters')
     ENDIF
     IF(NEWPAR) GOTO 8400
     !      WRITE(6,*) ' INA,inbmmff,INC,IND,i',INA,inb1,INC,IND,i
     !
     !   store the out-of-plane force constant
     !
     icoop(nth)=j
     !
     GOTO 8600
8400 CONTINUE
     !     WRITE(6,*) ' NOT FOUND, NCOOPT=',NCOOPT
     !   IF THIS POINT IS REACHED NO O-P-B CONSTANTS ARE FOUND
     !   TRY AGAIN IF EQUIVALENCIES HAVE NOT YET BEEN EXHAUSTED
     IF(ITRY.LT.MAXTRY) GOTO 8200
     !  TIME TO CALL THE DEFAULT-PARAMETERS ROUTINE
     CALL DEFOOP(IAMMFF,IBmmff,IC,ID,COPBK,GOTPAR)
     NCOOPT=NCOOPT+1
     KCOOP(NCOOPT)=IT0
     IF(GOTPAR) THEN
        OoplFC(NCOOPT)=COPBK*MDAKCAL ! STORE OOPL PARAMETER IN THE DATA
        icoop(nth)=NCOOPT
     ELSE
        IF(INIT.LE.NonFatal) INIT=NonFatal
        !                            ! STORE NULL PARAMETER IN THE DATA BASE
        OoplFC(NCOOPT)=0.    ! IN CASE ANOTHER INTERACTION
        icoop(nth)=0         ! OF THIS TYPE IS ENCOUNTERED
        if(wrnlev.ge.2) then
           WRITE(OUTU,8500) &
                MTYPE(IBmmff),MTYPE(ID),QNAME(IAMMFF),QNAME(IBmmff), &
                QNAME(IC)
8500       FORMAT(/' <kthetam> CONSTANTS MISSING FOR OOPLANE TYPES', &
                2I5,' USED IN ANGLE ',3(A,1X)/'     WARNING - THIS', &
                ' INTERACTION WILL BE NEGLECTED')
           WRITE(OUTU,'(4I5,5X,4I5)') &
                ITA,ITB,itcmmff,ITD,INA,inbmmff,INC,IND
        endif
     ENDIF
8600 CONTINUE
     !
     !  SET UP THE STRETCH-BEND INTERACTIONS
     !
     !   FIND THE LOCATION OF THE STRETCH-BEND CONSTANTS, IF THEY ARE KNOWN
     !
     MCIJ=MDBOND(NB1)
     MCJK=MDBOND(NB2)
     MCIJK=MDTHET(nth)
     MCLASS=0
     ITA=MTYPE(IAMMFF)
     ITB=MTYPE(IBmmff)
     itcmmff=MTYPE(IC)
     !...##IF DEBUG
     !      if(DEBUG) write(OUTU,'(a,3i4,4i3,i10)') 'DEBUG_KTHETAM> b'//
     !     & ' ITA,ITB,itcmmff,MCLASS,MCIJ,MCJK,MCIJK,KSB =',
     !     &   ITA,ITB,itcmmff,MCLASS,MCIJ,MCJK,MCIJK,KSB
     !...##ENDIF
     CALL STBNCON('PUT',ITA,ITB,itcmmff,MCLASS,MCIJ,MCJK,MCIJK,KSB)
     MDSTBN(NTH)=MCLASS
     !...##IF DEBUG
     !      if(DEBUG) write(OUTU,'(a,3i4,4i3,i10)') 'DEBUG_KTHETAM> a'//
     !     & ' ITA,ITB,itcmmff,MCLASS,MCIJ,MCJK,MCIJK,KSB =',
     !     &   ITA,ITB,itcmmff,MCLASS,MCIJ,MCJK,MCIJK,KSB
     !...##ENDIF
     !D      WRITE(6,'(A,2I3,I8,7I3)')
     !D    . ' NB1,NB2,KSB,MCLASS,INA,INBmmff,INC,MCIJ,MCJK,MCIJK =',
     !D    . NB1,NB2,KSB,MCLASS,INA,INBmmff,INC,MCIJ,MCJK,MCIJK
     !      SUBROUTINE STBNCON(PUTGET,I,J,K,MCLASS,MCIJ,MCJK,MCIJK,IJKM)
     CALL BIFIND(KSB,KCSTBN,NCSB,K,NCSBT,MAXCT,NEWPAR,FULL,2)
     !      WRITE(6,'(I5)') K
     IF(.NOT.NEWPAR) THEN
        IF(ITA.LT.itcmmff) THEN
           !           STBNFC(1,nth)=STBNP(1,K)*MDAKCAL
           !           STBNFC(2,nth)=STBNP(2,K)*MDAKCAL
           icstbn(nth)=k
        ELSE IF(ITA.GT.itcmmff) THEN
           !           STBNFC(1,nth)=STBNP(2,K)*MDAKCAL
           !           STBNFC(2,nth)=STBNP(1,K)*MDAKCAL
           icstbn(nth)=-k
        ELSE IF(ITA.EQ.itcmmff) THEN
           IF(MCLASS.EQ.0.OR.MCLASS.EQ.3.OR.MCLASS.EQ.4.OR. &
                MCLASS.EQ.5) THEN
              !              STBNFC(1,nth)=STBNP(1,K)*MDAKCAL
              !              STBNFC(2,nth)=STBNP(2,K)*MDAKCAL
              icstbn(nth)=k
           ELSE IF(MCLASS.EQ.1.AND.MDBOND(NB1).EQ.1) THEN
              !              STBNFC(1,nth)=STBNP(1,K)*MDAKCAL
              !              STBNFC(2,nth)=STBNP(2,K)*MDAKCAL
              icstbn(nth)=k
           ELSE IF(MCLASS.EQ.1.AND.MDBOND(NB2).EQ.1) THEN
              !              STBNFC(1,nth)=STBNP(2,K)*MDAKCAL
              !              STBNFC(2,nth)=STBNP(1,K)*MDAKCAL
              icstbn(nth)=-k
           ELSE IF(MCLASS.EQ.2.AND.MDBOND(NB1).EQ.1) THEN
              !              STBNFC(1,nth)=STBNP(2,K)*MDAKCAL
              !              STBNFC(2,nth)=STBNP(1,K)*MDAKCAL
              icstbn(nth)=-k
           ELSE IF(MCLASS.EQ.2.AND.MDBOND(NB2).EQ.1) THEN
              !              STBNFC(1,nth)=STBNP(1,K)*MDAKCAL
              !              STBNFC(2,nth)=STBNP(2,K)*MDAKCAL
              icstbn(nth)=k
           ELSE
              if(wrnlev.ge.2) then
                 WRITE(OUTU,'(A)') &
                      ' ***** APPARENT PROBLEM IN KTHETAM *****'
                 WRITE(OUTU,'(A,4I4)') ' MCLASS, MCIJ, MCJK, MCIJK =', &
                      MCLASS, MCIJ, MCJK, MCIJK
              endif
           ENDIF
        ENDIF
        !D        WRITE(6,'(A,2F10.3)')
        !D    .   ' STBNFC1, STBNFC2 = ',STBNFC(1,I),STBNFC(2,I)
     ELSE
        !
        !   TIME TO CALL THE DEFAULT-PARAMETERS ROUTINE
        !
        CALL DEFSTBN(AtNum(IAMMFF),AtNum(IBmmff),AtNum(IC),FABC,FCBA, &
             GOTPAR)
        !
        !         STBNFC(1,nth)=FABC*MDAKCAL
        !         STBNFC(2,nth)=FCBA*MDAKCAL
        IF(GOTPAR) THEN
           !
           !   CHECK ON WHETHER THERE IS SPACE TO STORE ANOTHER PARAMETER
           !
           IF(KPSB.GE.MAXCT) THEN
              if(wrnlev.ge.2) WRITE(OUTU,'(A,I6,A/A)') &
                   ' ***** Too many stretch-bend parameters generated '// &
                   '... max = ',MAXCT,' *****',' ***** Execution terminated' &
                   //' *****'
              CALL WRNDIE(-5,'<kthetam>','Too many stretch-bend'// &
                   ' parameters')
           ENDIF
           KPSB=KPSB+1
           KCSTBN(KPSB)=KSB
           STBNP(1,KPSB)=FABC*MDAKCAL
           STBNP(2,KPSB)=FCBA*MDAKCAL
           icstbn(nth)= kpsb
        ELSE
           icstbn(nth)=0
        ENDIF
        GOTO 9400
     ENDIF
9400 CONTINUE
     !D     WRITE(6,'(A,10I5)') ' IAMMFF,IBmmff,IC,ITA,ITB,itcmmff,NB1,NB2',
     !D    . IAMMFF,IBmmff,IC,MTYPE(IAMMFF),MTYPE(IBmmff),MTYPE(IC),NB1,NB2
     !D     WRITE(6,'(A,2I5,2F10.3)') ' StrbList, STBNFC',
     !D    & StrbList(1,nth),StrbList(2,nth),STBNFC(1,I),STBNFC(2,I)
  enddo loop950
  !
  RETURN
END SUBROUTINE KTHETAM

! ======================================================================
! SUBROUTINE KVDW(ERROR) : CHECK FOR ANY UNDEFINED TYPES
! PRESENT IN MOLECULE (RSTAR(MTYPE(I),MTYPE(I)).EQ.0)
! ======================================================================
!
SUBROUTINE KVDW(ERROR)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use psf
  use param
  use stream
  implicit none
  !
  integer i, ERROR
  !
  !23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
  DO I=1,NATOM
     IF(RSTAR(MTYPE(I)*(MTYPE(I)+1)/2).EQ.0.) then
        if(wrnlev.ge.2) WRITE(OUTU,'(A,A,I2,A)') &
             'KVDW> WARNING - ATOM ',QNAME(I),MTYPE(I), &
             ' HAS ZERO VDW RADIUS'
        if(ESTAR(MTYPE(I)*(MTYPE(I)+1)/2).ne.0.) then
           ERROR=2 ! SET FATAL-MISSING-PARAMETERS FLAG
        else
           RSTAR(MTYPE(I)*(MTYPE(I)+1)/2)=1
           if(WRNLEV.ge.2) write(OUTU,'(A,A)') &
                'KVDW> AND ZERO WELL DEPTH AS WELL;', &
                ' VDW INTERACTION FOR THIS ATOM WILL BE IGNORED'
        endif
     endif
  ENDDO
  !
  RETURN
END SUBROUTINE KVDW

! ======================================================================
! INTEGER FUNCTION LEQUIV : GETS THE EQUIVALENT FOR MMFF ATOM TYPE
! "ITYPE" AT EQUIVALENCY LEVEL "LEVEL" USING DEFAULT PROTOCOL "ID"
! ======================================================================
INTEGER FUNCTION LEQUIV(ITYPE,LEVEL,ID)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  use chm_kinds
  use dimens_fcm
  use mmffm
  use stream
  use string
  implicit none

  integer, intent(in) :: itype  ! MMFF atom type
  integer, intent(in) :: level  ! EQUIVALENCY LEVEL
  integer, intent(in) :: id     ! PROTOCOL ID

  integer l, mxid, mxlevel
  PARAMETER (MXLEVEL = 5, MXID =4 )
  integer NL(MXLEVEL, MXID)

  DATA NL/1, 2, 2, 2, 2, &
          1, 2, 3, 4, 5, &
          1, 2, 3, 5, 5, &
          1, 2, 5, 3, 5/

  SCRTCH=' '
  IF (LEVEL.LT.0.OR.LEVEL.GT.MXLEVEL) THEN
     WRITE(SCRTCH,'(A,I5)') 'ILLEGAL LEVEL     =',LEVEL
     CALL WRNDIE(-2,'<LEQUIV>',SCRTCH(:25))
  ENDIF

  IF (ID.LE.0.OR.ID.GT.MXID) THEN
     WRITE(SCRTCH,'(A,I5)') 'ILLEGAL ID        =',ID
     CALL WRNDIE(-2,'<LEQUIV>',SCRTCH(:25))
  ENDIF

  IF ((.not. allocated(mdef)) .or. &
      ITYPE .lt. lbound(mdef, 2) .or. ITYPE .gt. ubound(mdef, 2)) THEN
     WRITE(SCRTCH,'(A,I5)') 'ILLEGAL ATOM TYPE =',ITYPE
     CALL WRNDIE(-2,'<LEQUIV>',SCRTCH(:25))
  ENDIF

  if (LEVEL .eq. 0) then
     LEQUIV = ITYPE
  elseif (SCRTCH .ne. ' ') then
     LEQUIV = ITYPE
     CALL WRNDIE(-5,'<LEQUIV>','ILLEGAL LEVEL,ID or ITYPE')
  else
     L = NL(LEVEL, ID)
     LEQUIV = MDEF(L, ITYPE)
  endif

  RETURN
END FUNCTION LEQUIV

! ===============================================================
! LOGICAL FUNCTION SP2SGL : Check for formally single bond
! between sp2-hybridized atoms
! ===============================================================
LOGICAL FUNCTION SP2SGL(I,J)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  use mmffm
  use exfunc
  use param
  use psf
  implicit none
  !
  integer i, ii, jj, j
  ! T.A. Halgren change (B980629.wy)
  integer n
  ! end T.A. Halgren change
  !
  SP2SGL=.FALSE.
  IF(IBORDR(I,J).NE.1) RETURN
  II=MTYPE(I)
  JJ=MTYPE(J)
  IF(MSP2SGL(II).GT.0.AND.MSP2SGL(JJ).GT.0) THEN
     IF(IFAROM(I).AND.IFAROM(J)) THEN
        ! T.A. HAlgren change (B980629.wy)
        !           IF(INRING(I).EQ.INRING(J)) RETURN
        DO N=1,N_ARBOND
           IF(I.EQ.AR_BOND(1,N).AND.J.EQ.AR_BOND(2,N)) RETURN
           IF(I.EQ.AR_BOND(2,N).AND.J.EQ.AR_BOND(1,N)) RETURN
        ENDDO
        ! end T.A. Halgren change
     ENDIF
     !D       WRITE(IO,'(A,4I5)') ' SP2-SP2 SINGLE BOND',I,J,II,JJ
     !D       IF(IZ.NE.0)
     !D    .  WRITE(IZ,'(A,4I5)') ' SP2-SP2 SINGLE BOND',I,J,II,JJ
     SP2SGL=.TRUE.
  ENDIF
  !
  RETURN
END FUNCTION SP2SGL

! =====================================================================
! LOGICAL FUNCTION STRDEF : assigns default stretching parameters
! =====================================================================
LOGICAL FUNCTION STRDEF ( ATNOI, ATNOJ, HYBRDI, HYBRDJ, &
     BTYPIJ, STRCON, PDIST, SUCCES)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 22 Nov 95: changed WRITE(OUTU,*) statement to explicit
  !  format.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  use stream
  use mmffm
  implicit none
  !      COMMON/BONDSTR/BL1(maxcb),BK1(maxcb),IA(2,maxcb),NBNDCON
  !       real(chm_real) BL1,BK1
  !       INTEGER IA,NBNDCON,I
  !
  INTEGER    ATNOI, ATNOJ, HYBRDI, HYBRDJ, BTYPIJ
  integer ia1, ia2, ijk, i
  real(chm_real) STRCON, PDIST
  integer itemp, maxano
  real(chm_real)  aij, bij, radi, radj, sens, shrink, power
  !----------------------------------------------------------------------
  ! STRDEF.FOR
  ! J. D. ANDOSE, 6/18/85
  ! This routine assigns default stretching parameters for a molecular
  ! mechanics force field.  It utilizes a modified form of the
  ! Shomaker-Stevenson equation for assigning a preferred bond length,
  ! and utilizes a method developed by Badger (see below) to calculate
  ! the force constant.
  ! Reference:  see Chapter 7 of "The Nature of the Chemical Bond", by
  !       L. Pauling, third ed., Cornell University Press, 1960.
  ! CALLING SEQUENCE:
  !       LOGICAL   STRDEF, SUCCES
  !       SUCCES = STRDEF ( ATNOI, ATNOJ, HYBRDI, HYBRDJ, BTYPIJ,
  !                          STRCON, PDIST, SUCCES)
  !   OR,
  !       LOGICAL         SUCCES
  !       CALL STRDEF ( ATNOI, ATNOJ, HYBRDI, HYBRDJ, BTYPIJ,
  !                     STRCON, PDIST, SUCCES)
  ! ARGUMENT DEFINITIONS:
  !       ATNOI   INTEGER     Atomic number of atom I           (Input)
  !       ATNOJ   INTEGER     Atomic number of atom J           (Input)
  !       HYBRDI  INTEGER     Hybridization of atom I           (Input)
  !       HYBRDJ  INTEGER     Hybridization of atom J           (Input)
  !       BTYPIJ  INTEGER     Bond type for bond I to J         (Input)
  !       STRCON  REAL        Stretching force constant for
  !                              bond I-J, in mdyne/Angstrom   (Returned)
  !       PDIST   REAL        Preferred distance               (Returned)
  !       SUCCES LOGICAL     Success indicator                (Returned)
  ! WHERE:
  !       HYBRDI,HYBRDJ - integer value representing hybridization, where
  !              1 = sp, 2 = sp2, and 3 = sp3.  NOTE:  for Hydrogen atoms,
  !              the value specified is not important, but must still
  !              be one of the values 1, 2, or 3.
  !       BTYPIJ - integer value representing bond type between atoms
  !              I and J, where 1 = single bond, 2 = double bond,
  !              3 = triple bond, and 4 = aromatic bond.
  !       SUCCES - returns .TRUE. if successful, .FALSE. if unable
  !              to assign default parameters.
  ! NOTICE:  This routine does NO input/output and therefore prints no
  !          error messages.  It is the CALLing routines responsibility
  !          to check the return value of the routine or the SUCCES
  !          parameter and take appropriate action.
  !----------------------------------------------------------------------
  ! MAXANO : Maximum atomic number handled by this routine.
  PARAMETER (MAXANO = 54)
  ! COVRAD: table of covalent radii, stored by atomic number.  A value of
  !         0.0 (zero) means that this routine cannot handle an atom
  !         with that atomic number.
  real(chm_real) COVRAD(MAXANO)
  ! ELNEG: Table of electronegativities, stored by atomic number.  Again,
  !       a value of 0.0 (zero) means that this routine cannot handle an
  !       atom with that atomic number.
  real(chm_real) ELNEG(MAXANO)
  !
  ! Note:  values of COVRAD have been taken, where available, from R. Bloom
  ! and A. Haaland, J. Molec. Struc. 1985, 128, 21-27.  Those for
  ! electronegetivity and Pauling-scale values defined by Allred, A. L. and
  ! Rochow, E. G., J. Inorg. Nucl. Chem., 1958, 5, 264-258.  Other values
  ! for COVRAD and ELNEG were taken from "TABLE OF PERIODIC PROPERTIES OF
  ! THE ELEMENTS", E. H. Sargent & Co., brochure S-1880 6, copyright 1962
  ! by Dyna-Slide Co.
  !
  !
  ! DEFINE SOME ATOM NAME CONSTANTS.
  !
  INTEGER HYDGEN, HELIUM, CARBON, NEON, ARGON, KRYPTN, XENON, RADON
  PARAMETER (HYDGEN = 1)
  PARAMETER (HELIUM = 2)
  PARAMETER (CARBON = 6)
  PARAMETER (NEON = 10)
  PARAMETER (ARGON = 18)
  PARAMETER (KRYPTN = 36)
  PARAMETER (XENON = 54)
  PARAMETER (RADON = 86)
  LOGICAL SUCCES
  INTEGER  ANI, ANJ
  !======================================================================
  DATA COVRAD /  0.33,                                        0.0, &
       1.34, 0.90,    0.81, 0.77, 0.73, 0.72, 0.74, 0.0, &
       1.54, 1.30,    1.22, 1.15, 1.09, 1.03, 1.01, 0.0, &
       1.96, 1.74, &
       1.44, 1.36, 0.00, 0.00, 0.00, &
       0.00, 0.00, 0.00, 1.38, 1.31, &
       1.19, 1.20, 1.20, 1.16, 1.15, 0.0, &
       2.11, 1.92, &
       1.62, 1.48, 0.00, 0.00, 0.00, &
       0.00, 0.00, 0.00, 1.53, 1.48, &
       1.46, 1.40, 1.41, 1.35, 1.33, 0.0/
  !
  !  ALLRED-ROCHOW ELECTRONEGATIVITIES
  !
  DATA ELNEG / 2.20,                                          0.0, &
       0.97, 1.47,      2.01, 2.5, 3.07, 3.5, 4.10, 0.0, &
       1.01, 1.23,      1.47, 1.74, 2.06, 2.44, 2.83, 0.0, &
       0.91, 1.04, &
       1.3, 1.5, 1.6, 1.6, 1.5, &
       1.8, 1.8, 1.8, 1.9, 1.6, &
       1.82, 2.02, 2.20, 2.48, 2.74, 0.0, &
       0.89, 0.99, &
       1.3, 1.4, 1.6, 1.8, 1.9, &
       2.2, 2.2, 2.2, 1.9, 1.7, &
       1.49, 1.72, 1.82, 2.01, 2.21, 0.0/
  ANI = abs(ATNOI) ! to accomodate negative atomic numbers for
  ANJ = abs(ATNOJ) ! dummy atoms
  SUCCES = .TRUE.
  STRDEF  = .TRUE.
  ! Determine whether or not routine can handle input.
  IF (ANI .GT. MAXANO .OR. &
       ANJ .GT. MAXANO .OR. &
       HYBRDI .LT. 1 .OR. HYBRDI .GT. 3 .OR. &
       HYBRDJ .LT. 1 .OR. HYBRDJ .GT. 3 .OR. &
       BTYPIJ .LT. 1 .OR. BTYPIJ .GT. 5 .OR. &
       COVRAD(ANI) .EQ. 0.0 .OR. &
       COVRAD(ANJ) .EQ. 0.0               ) THEN
     SUCCES = .FALSE.
     STRDEF  = .FALSE.
     RETURN
  ENDIF
  ! Assign preferred distance if undefined on input
  IF(PDIST.le.0) then
     !       First, assign covalent radii corrected by bond type and
     !       hybridization.
     RADI = COVRAD(ANI)
     RADJ = COVRAD(ANJ)
     !       Don't want to correct covalent radius if either atom is Hydrogen.
     IF (ANI .EQ. HYDGEN .OR. ANJ .EQ. HYDGEN) THEN
        RADI = COVRAD(ANI)
        RADJ = COVRAD(ANJ)
     ELSE IF ( BTYPIJ .EQ. 5) THEN
        RADI = RADI - 0.035
        RADJ = RADJ - 0.035
     ELSE IF ( BTYPIJ .EQ. 4) THEN
        RADI = RADI - 0.070
        RADJ = RADJ - 0.070
     ELSE IF (BTYPIJ .EQ. 3) THEN
        RADI = RADI - 0.17
        RADJ = RADJ - 0.17
     ELSE IF (BTYPIJ .EQ. 2) THEN
        RADI = RADI - 0.10
        RADJ = RADJ - 0.10
     ELSE IF (BTYPIJ .EQ. 1) THEN
        IF (HYBRDI .EQ. 1) RADI = RADI - 0.08
        IF (HYBRDJ .EQ. 1) RADJ = RADJ - 0.08
        IF (HYBRDI .EQ. 2) RADI = RADI - 0.035
        IF (HYBRDJ .EQ. 2) RADJ = RADJ - 0.035
     ENDIF

     !       Assign the shrinkage factor.  This takes into account the
     !       expansion of the bond length as a result of other strain
     !       factors in the force field.  This shrink factor has been determined
     !       by comparing actual averaged bond lengths for MP2/6-31G* optimized
     !       structures to the reference bond lengths needed to best reproduce
     !       the observed geometries.
     SHRINK = 0.008
     IF ( ANI .EQ. HYDGEN .OR. ANJ .EQ. HYDGEN) THEN
        SHRINK = 0.0
     ELSE IF ( ANI .GT. NEON .OR. ANJ .GT. NEON) THEN
        SHRINK = 0.0
     ENDIF
     !       Ready, now, to calculate the preferred distance.
     !      SENS and POWER are constants defined in R.Blom and A. Haaland,
     !      J. Molec. Struc, 1985, 128, 21-27.
     SENS=0.08
     IF(ANI.EQ.HYDGEN.OR.ANJ.EQ.HYDGEN) SENS=0.05
     POWER=1.4
     PDIST = RADI + RADJ - &
          SENS*ABS(ELNEG(ANI)-ELNEG(ANJ))**POWER &
          - SHRINK
  endif ! IF(PDIST.le.0) then
  ! -----------------------------------------------------------------
  ! Assign stretching constant.  Where possible, use an inverse 6th-power
  ! relationship.  If requisite data is not available, use relationship
  ! developed by Badger: see R. M. Badger, J. Chem. Phys., 2, 128 (1934);
  ! J. Chem. Phys., 3, 710 (1935).  Note, however, that the parameters
  ! are those described in: D. L. Herschbach and V. W. Laurie, J. Chem.
  ! Phys. 1961, 35, 458-463.
  !
  !   FIND THE RELEVANT SINGLE-BOND ENTRY
  !
  IA1=MIN(ANI,ANJ)
  IA2=MAX(ANI,ANJ)
  !D     WRITE(OUTU,'(A,3I5)') ' NBNDCON, IA1, IA2:',NBNDCON,IA1,IA2
  DO I=1,NBNDCON
     IJK=I
     IF(IA1.EQ.IA(1,I).AND.IA2.EQ.IA(2,I)) then
        !
        !   CALCULATE FORCE CONSTANT USING INVERSE SIXTH POWER RELATIONSHIP
        !
        STRCON=BK1(IJK)*(BL1(IJK)/PDIST)**6
        !D            WRITE(6,'(A,3I5,4F10.3)')
        !D    .       ' IA1, IA2, IJK, BK1, BL1, PDIST, STRCON:',
        !D    .       IA1, IA2, IJK, BK1(IJK), BL1(IJK), PDIST, STRCON
        RETURN
     endif
  ENDDO
  !
  !   IF HERE, REQUIRED SINGLE-BOND INFO NOT FOUND
  !
  if(wrnlev.ge.2) WRITE(OUTU,'(A,2I6)') &
       'STRDEF> SINGLE BOND CONSTANTS NOT FOUND FOR IA1, IA2 =', &
       IA1, IA2
  IF ( ANJ .LT. ANI) THEN
     ITEMP = ANI
     ANI = ANJ
     ANJ = ITEMP
  ENDIF
  !
  IF ( ANI .LE. HELIUM .AND. ANJ .LE. HELIUM) THEN
     AIJ = 1.26
     BIJ = 0.025
  ELSE IF ( ANI .LE. HELIUM .AND. ANJ .LE. NEON) THEN
     AIJ = 1.66
     BIJ = 0.30
  ELSE IF ( ANI .LE. HELIUM .AND. ANJ .LE. ARGON) THEN
     AIJ = 1.84
     BIJ = 0.38
  ELSE IF ( ANI .LE. HELIUM .AND. ANJ .LE. KRYPTN) THEN
     AIJ = 1.98
     BIJ = 0.49
  ELSE IF ( ANI .LE. HELIUM .AND. ANJ .LE. XENON) THEN
     AIJ = 2.03
     BIJ = 0.51
  ELSE IF ( ANI .LE. HELIUM .AND. ANJ .LE. RADON) THEN
     AIJ = 2.03
     BIJ = 0.25
  ELSE IF ( ANI .LE. NEON .AND. ANJ .LE. NEON) THEN
     AIJ = 1.91
     BIJ = 0.68
  ELSE IF ( ANI .LE. NEON .AND. ANJ .LE. ARGON) THEN
     AIJ = 2.28
     BIJ = 0.74
  ELSE IF ( ANI .LE. NEON .AND. ANJ .LE. KRYPTN) THEN
     AIJ = 2.35
     BIJ = 0.85
  ELSE IF ( ANI .LE. NEON .AND. ANJ .LE. XENON) THEN
     AIJ = 2.33
     BIJ = 0.68
  ELSE IF ( ANI .LE. NEON .AND. ANJ .LE. RADON) THEN
     AIJ = 2.50
     BIJ = 0.97
  ELSE IF ( ANI .LE. ARGON .AND. ANJ .LE. ARGON) THEN
     AIJ = 2.41
     BIJ = 1.18
  ELSE IF ( ANI .LE. ARGON .AND. ANJ .LE. KRYPTN) THEN
     AIJ = 2.52
     BIJ = 1.02
  ELSE IF ( ANI .LE. ARGON .AND. ANJ .LE. XENON) THEN
     AIJ = 2.61
     BIJ = 1.28
  ELSE IF ( ANI .LE. ARGON .AND. ANJ .LE. RADON) THEN
     AIJ = 2.60
     BIJ = 0.84
  ELSE IF ( ANI .LE. KRYPTN .AND. ANJ .LE. KRYPTN) THEN
     AIJ = 2.58
     BIJ = 1.41
  ELSE IF ( ANI .LE. KRYPTN .AND. ANJ .LE. XENON) THEN
     AIJ = 2.66
     BIJ = 0.86
  ELSE IF ( ANI .LE. KRYPTN .AND. ANJ .LE. RADON) THEN
     AIJ = 2.75
     BIJ = 1.14
  ELSE IF ( ANI .LE. XENON .AND. ANJ .LE. XENON) THEN
     AIJ = 2.85
     BIJ = 1.62
  ELSE IF ( ANI .LE. XENON .AND. ANJ .LE. RADON) THEN
     AIJ = 2.76
     BIJ = 1.25
  ELSE
     AIJ = 3.15
     BIJ = 1.80
  ENDIF
  STRCON=((AIJ-BIJ)/(PDIST-BIJ))**3
  !D      WRITE(OUTU,'(A,2I5,4F10.3)')
  !D    . ' ANI, ANJ, AIJ, BIJ, PDIST, STRCON:',
  !D    . IA1, IA2, AIJ, BIJ, PDIST, STRCON
  ! DONE.
  !
  RETURN
END FUNCTION STRDEF

! ====================================================================
! SUBROUTINE TRSYMB : TRANSLATE THE SYMBOLIC INTO THE NUMERICAL TYPE
! TRANSLATE THE SYMBOLIC MMFF ATOM TYPE INTO THE NUMERICAL TYPE
! ====================================================================
SUBROUTINE TRSYMB(IATOM)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use param
  use psf
  use rtf, only: ctype, intdef
  use string
  implicit none
  !
  integer i, iatom
  !
  DO I=1,MAXATC
     IF(CTYPE(I)(1:4).EQ.SYMB(IATOM)) THEN
        MTYPE(IATOM)=INTDEF(I)
        !D       WRITE(6,10) QNAME(IATOM),SYMB(IATOM),INTDEF(I)
        !D       WRITE(12,10) QNAME(IATOM),SYMB(IATOM),INTDEF(I)
10      FORMAT(' ATOM',1X,A,'   SYMBOL: ',A,'   TYPE:',I3)
        RETURN
     ENDIF
  enddo
  !  IF STILL HERE, THE SYMBOL COULD NOT BE TRANSLATED -
  !  WRITE ERROR MESSAGE
  !     WRITE(OUTU,200) QNAME(IATOM),SYMB(IATOM)
  SCRTCH='ATOM-TYPE SYMBOL COULD NOT BE TRANSLATED: ATOM NAME>'// &
       QNAME(IATOM)//'< SYMBOL >'//SYMB(IATOM)//'<'
  !     IF(IZ.NE.0) WRITE(IZ,200) QNAME(IATOM),SYMB(IATOM)
  ! 200 FORMAT(/' ***** ERROR - AN ASSIGNED ATOM-TYPE SYMBOL',
  !    . ' COULD NOT BE TRANSLATED'/8X,'ATOM NAME: ',A,'  SYMBOL: ',A)
  call wrndie(-5,'<trsymb>',SCRTCH(:LEN_TRIM(SCRTCH)))
  !
  return
end SUBROUTINE TRSYMB

#endif 
SUBROUTINE NULL_asnpar
  RETURN
END SUBROUTINE NULL_asnpar

