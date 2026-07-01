module mcmvdihe
#if KEY_MC==1
  use chm_types
  use dimens_fcm
  implicit none

contains

  SUBROUTINE MVDIHE(COMLYN,COMLEN,NMVATM, &
       IPIVTP,IMVNGP,IBLSTP,MDXP,MBONDT,QBND, &
       ARMLIM,ARMMAX,ANISO,RMDX,X,Y,Z,WMAIN)
    !
    !       Determines move specific information for a simple
    !       dihedral rotation.
    !
    !       Aaron R. Dinner
    !
    use psf
    use selctam
    use memory
    use mcmvutil
    use select
    use string
    !
    !       Passed Variables
    !
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    type(iptr_ptr) :: IMVNGP, IBLSTP, IPIVTP
    INTEGER NMVATM, MBONDT
    type(chm_ptr) :: MDXP
    real(chm_real)  RMDX, ARMMAX, X(*), Y(*), Z(*), WMAIN(*)
    LOGICAL ARMLIM, ANISO, QBND(MBONDT)
    !
    !       Local Variables
    !
    integer,allocatable,dimension(:) :: JFP, KFP
    INTEGER I
    LOGICAL FEWER

    !       Fill in logicals for which bonded energy terms must be calculated.
    CALL FILLQB(QBND,.FALSE.,.FALSE.,.TRUE.,.FALSE.,.FALSE.)

    !       Generate atom lists that keep track of which internal coordinates
    !       are affected by moving a given atom.
    !       Note that the bond list is also used as a connectivity table.
    CALL MKBNDT()

    !       Get the two SELE...END statements that define the
    !       center atoms of the rotatable dihedrals.
    !       Note: Default behavior is different from SELCTD
    call chmalloc('mvdihe.src','MVDIHE','JFP',NATOM,intg=JFP)
    call chmalloc('mvdihe.src','MVDIHE','KFP',NATOM,intg=KFP)
    JFP(1:NATOM) = 0
    KFP(1:NATOM) = 0
    CALL SELCTA(COMLYN,COMLEN,JFP,X,Y,Z,WMAIN,.TRUE.)
    CALL SELCTA(COMLYN,COMLEN,KFP,X,Y,Z,WMAIN,.TRUE.)

    !       See if the selections are interchangeable
    FEWER = (GTRMI(COMLYN,COMLEN,'FEWE',0) .GT. 0)

    !       Count the number of elements that will be in the list.
    !       If there are no move locations, no point in the move, so return.
    CALL GTDIHE(NMVATM,IPIVTP,IMVNGP,IBLSTP,MDXP, &
         RMDX, JFP, KFP, &
         IABNDP, IAPHIP, &
         NATOM,NBOND,IB,JB,NPHI,JP,KP,FEWER,0)
    IF (NMVATM > 0) THEN

       !       Allocate the space
       allocate(IPIVTP%A(NMVATM))
       allocate(IMVNGP%A(NMVATM))
       allocate(IBLSTP%A(NMVATM))
       call chmalloc('mvdihe.src','MVDIHE','MDXP',NMVATM,crlp=MDXP%A)

       !       Fill the lists.
       CALL GTDIHE(NMVATM,IPIVTP,IMVNGP,IBLSTP,MDXP, &
            RMDX, JFP, KFP, &
            IABNDP, IAPHIP, &
            NATOM,NBOND,IB,JB,NPHI,JP,KP,FEWER,1)

       !       Assign some more global parameters
       ARMLIM = .TRUE.
       ARMMAX = 180.0

       !       Free the space used to hold bonded term to atom information.
       CALL FREBND()
    ENDIF
    call chmdealloc('mvdihe.src','MVDIHE','KFP',NATOM,intg=KFP)
    call chmdealloc('mvdihe.src','MVDIHE','JFP',NATOM,intg=JFP)

    RETURN
  END SUBROUTINE MVDIHE
  
  SUBROUTINE GTDIHE(NDIHE,IPIVTP,IMVNGP,IBLSTP,MDXP, &
       RMDX,JF,KF,IABND,IAPHI, &
       NATOM,NBOND,IB,JB,NPHI,JP,KP,FEWER,MODE)
    !
    !       Counts and assigns move instances of individual dihedral
    !       rotations given the appropriate atom selections.
    !
    !       Aaron R. Dinner
    !
    use memory
    !
    !       Passed Variables
    !
    INTEGER NDIHE
    type(iptr_ptr) :: IMVNGP, IBLSTP, IPIVTP
    type(chm_ptr) :: MDXP
    INTEGER NATOM, NBOND, NPHI, MODE
    INTEGER JF(*), KF(*)
    type(chm_iptr) :: IABND(:), IAPHI(:)
    INTEGER IB(*), JB(*), JP(*), KP(*)
    real(chm_real)  RMDX
    LOGICAL FEWER
    !
    !       Local Variables
    !
    INTEGER I, IPHI, NMV1, NMV2
    integer,allocatable,dimension(:) :: IMVATP, IMAP, NMAP, IUNIQP
    LOGICAL LUNIQ

    NDIHE = 0
    call chmalloc('mvdihe.src','GTDIHE','IMVATP',NATOM,intg=IMVATP)
    call chmalloc('mvdihe.src','GTDIHE','IMAP',NATOM,intg=IMAP)
    call chmalloc('mvdihe.src','GTDIHE','NMAP',NATOM,intg=NMAP)

    !       Keep track of how many times each bond is counted
    !       to make sure each instance is unique
    call chmalloc('mvdihe.src','GTDIHE','IUNIQP',NBOND,intg=IUNIQP)
    IUNIQP(1:NBOND) = 0

    IF (FEWER) THEN
       !         Treat the two selections as interchangeable and
       !         keep only the direction that moves fewer atoms
       DO IPHI = 1, NPHI
          IF (((JF(JP(IPHI)).EQ.1).AND.(KF(KP(IPHI)).EQ.1.)).OR. &
               ((KF(JP(IPHI)).EQ.1).AND.(JF(KP(IPHI)).EQ.1.))) THEN
             CALL UNQPHI(LUNIQ, IUNIQP, IABND(JP(IPHI))%A, &
                  IB,JB,JP(IPHI),KP(IPHI),FEWER)
             IF (LUNIQ) THEN
                CALL TORSMV(IMVATP, NMV1,JP(IPHI),KP(IPHI), &
                     IABND,NATOM,IB,JB, IMAP, NMAP)
                IF (NMV1 .GT. 0) THEN
                   NDIHE = NDIHE + 1
                   IF (MODE .GT. 0) THEN
                      CALL TORSMV(IMVATP, NMV2,KP(IPHI),JP(IPHI), &
                           IABND,NATOM,IB,JB, IMAP, NMAP)
                      IF (NMV2 .LE. NMV1) THEN
                         CALL ADDPHI(NDIHE,KP(IPHI),JP(IPHI),JP,KP, &
                              IMVATP, NATOM,IAPHI, &
                              IPIVTP%A, IMVNGP%A, &
                              IBLSTP%A, MDXP%A, &
                              RMDX)
                      ELSE
                         CALL TORSMV(IMVATP, NMV1,JP(IPHI),KP(IPHI), &
                              IABND,NATOM,IB,JB, IMAP, NMAP)
                         CALL ADDPHI(NDIHE,JP(IPHI),KP(IPHI),JP,KP, &
                              IMVATP, NATOM,IAPHI, &
                              IPIVTP%A, IMVNGP%A, &
                              IBLSTP%A, MDXP%A, RMDX)
                      ENDIF
                   ENDIF
                ENDIF
             ENDIF
          ENDIF
       ENDDO
    ELSE
       !         Retain the directionality of the selection
       DO IPHI = 1, NPHI
          IF ((JF(JP(IPHI)).EQ.1).AND.(KF(KP(IPHI)).EQ.1.)) THEN
             CALL UNQPHI(LUNIQ, IUNIQP, IABND(JP(IPHI))%A, &
                  IB,JB,JP(IPHI),KP(IPHI),FEWER)
             IF (LUNIQ) THEN
                CALL TORSMV(IMVATP, NMV1,JP(IPHI),KP(IPHI), &
                     IABND,NATOM,IB,JB, IMAP, NMAP)
                IMVATP(JP(IPHI)) = 0
                IF (NMV1 .GT. 0) THEN
                   NDIHE = NDIHE + 1
                   IF (MODE .GT. 0) THEN
                      CALL ADDPHI(NDIHE,JP(IPHI),KP(IPHI),JP,KP, &
                           IMVATP, NATOM, IAPHI, IPIVTP%A, &
                           IMVNGP%A, IBLSTP%A, MDXP%A, &
                           RMDX)
                   ENDIF
                ENDIF
             ENDIF
          ENDIF
          IF ((KF(JP(IPHI)).EQ.1).AND.(JF(KP(IPHI)).EQ.1.)) THEN
             CALL UNQPHI(LUNIQ, IUNIQP, IABND(KP(IPHI))%A, &
                  IB,JB,KP(IPHI),JP(IPHI),FEWER)
             IF (LUNIQ) THEN
                CALL TORSMV(IMVATP, NMV2,KP(IPHI),JP(IPHI), &
                     IABND,NATOM,IB,JB, IMAP, NMAP)
                IMVATP(JP(IPHI)) = 0
                IF (NMV2 .GT. 0) THEN
                   NDIHE = NDIHE + 1
                   IF (MODE .GT. 0) THEN
                      CALL ADDPHI(NDIHE,KP(IPHI),JP(IPHI),KP,JP, &
                           IMVATP, NATOM, IAPHI, IPIVTP%A, &
                           IMVNGP%A, IBLSTP%A, MDXP%A, &
                           RMDX)
                   ENDIF
                ENDIF
             ENDIF
          ENDIF
       ENDDO
    ENDIF

    call chmdealloc('mvdihe.src','GTDIHE','IUNIQP',NBOND,intg=IUNIQP)

    call chmdealloc('mvdihe.src','GTDIHE','NMAP',NATOM,intg=NMAP)
    call chmdealloc('mvdihe.src','GTDIHE','IMAP',NATOM,intg=IMAP)
    call chmdealloc('mvdihe.src','GTDIHE','IMVATP',NATOM,intg=IMVATP)

    RETURN
  END SUBROUTINE GTDIHE
  
  SUBROUTINE UNQPHI(LUNIQ,IUNIQ,IABNDJ,IB,JB,JPADD,KPADD,FEWER)
    !
    !       Determines if a dihedral is unique for MC move set
    !
    !       IABNDJ is the list of bonds involving JPADD
    !
    !       Aaron R. Dinner
    !
    use mcmvutil, only: notatm
    INTEGER JPADD,KPADD
    INTEGER IB(*), JB(*), IABNDJ(:), IUNIQ(*)
    LOGICAL FEWER, LUNIQ
    !
    INTEGER I, J, K, NB

    NB = IABNDJ(1) + 1

    DO I = 2, NB
       J = IABNDJ(I)
       K = NOTATM(IB(J),JB(J),JPADD)
       IF (K .EQ. KPADD) THEN
          IF (IUNIQ(J).EQ.0) THEN
             IUNIQ(J) = SIGN(1,JPADD-KPADD)
             LUNIQ = .TRUE.
             RETURN
          ELSE IF ((.NOT. FEWER).AND.(IUNIQ(J).LT.2).AND. &
               (IUNIQ(J).NE.SIGN(1,JPADD-KPADD))) THEN
             IUNIQ(J) = 2
             LUNIQ = .TRUE.
             RETURN
          ELSE
             LUNIQ = .FALSE.
             RETURN
          ENDIF
       ENDIF
    ENDDO

    !       If this point is reached, there is a problem
    CALL WRNDIE (-5, '<UNQPHI>', &
         'INTERNAL ERROR:   COULD NOT FIND BOND')

    RETURN
  END SUBROUTINE UNQPHI

  SUBROUTINE TORSMV(IMVATM,NMV,JPADD,KPADD,IABND,NATOM, &
       IB,JB,IMAP,NMAP)
    !
    !       Tag moving atoms on a torsion move.
    !
    !       NMV is the number of moving atoms.  If the dihedral is found
    !       to be non-rotatable, due to a loop, the routine returns immediately
    !       with NMV = -1.
    !
    !       Aaron R. Dinner
    !
    use mcmvutil, only: mvatm, notatm
    INTEGER JPADD, KPADD, NATOM, NMV
    INTEGER IB(*), JB(*)
    INTEGER IMVATM(*), IMAP(*), NMAP(*)
    type(chm_iptr) :: IABND(:)
    !
    INTEGER I, J, NB

    IMVATM(1:NATOM) = 0
    !       Search along each bond off of the third atom
    NB = IABND(KPADD)%A(1) + 1
    DO I = 2, NB
       J = IABND(KPADD)%A(I)
       J = NOTATM(IB(J),JB(J),KPADD)
       IF (J .NE. JPADD) THEN
          CALL MVATM(IMVATM,NATOM,J,KPADD,IABND,IB,JB,IMAP,NMAP)

          IF (IMVATM(JPADD) .NE. 0) THEN
             NMV = -1
             RETURN
          ENDIF

       ENDIF
    ENDDO

    NMV = 0
    DO I = 1, NATOM
       IF (IMVATM(I) .GT. 0) NMV = NMV + 1
    ENDDO

    RETURN
  END SUBROUTINE TORSMV

  SUBROUTINE ADDPHI(IND,JPADD,KPADD,JP,KP,IMVATM,NATOM, &
       IAPHI,IPIVT,IMVNG,IBLST,RMDXAR,RMDX)
    !
    !       Adds a dihedral move to the lists
    !
    !       Aaron R. Dinner
    !
    use memory
    use mcmvutil, only: imvlst
    INTEGER IND, JPADD, KPADD, NATOM
    INTEGER JP(*), KP(*)
    INTEGER IMVATM(*)
    type(chm_iptr) :: IAPHI(:), IMVNG(:), IBLST(:), IPIVT(:)
    real(chm_real)  RMDXAR(:), RMDX

    !       Run through the list to store the moving atoms
    CALL IMVLST(IMVNG(IND), NATOM, IMVATM)

    !       Save the pivot atoms
    call chmalloc('mvdihe.src','ADDPHI','IPIVT(IND)',2,intgp=IPIVT(IND)%A)
    IPIVT(IND)%A(1) = JPADD
    IPIVT(IND)%A(2) = KPADD

    !       Generate the bonded information
    CALL PHILST(IBLST(IND),JPADD,KPADD,JP,KP,IAPHI)

    !       Assign some limits
    RMDXAR(IND) = RMDX

    RETURN
  END SUBROUTINE ADDPHI

  SUBROUTINE PHILST(IBLSTP,IA2,IA3,JP,KP,IAPHI)
    !
    !       Finds all the dihedral terms affected by a dihedral rotation.
    !       Generates the bonded term lists for that rotation.
    !       Assumes only dihedral terms are affected!
    !
    !       Aaron R. Dinner
    !
    use memory
    !
    !       Passed Variables
    !
    type(chm_iptr) :: IBLSTP, IAPHI(:)
    INTEGER JP(*), KP(*), IA2, IA3
    !
    !       Local Variables
    !
    INTEGER I, J, M, N
    integer,allocatable,dimension(:) :: TEMPP

    N = IAPHI(IA2)%A(1)
    call chmalloc('mvdihe.src','PHILST','TEMPP',N,intg=TEMPP)
    N = N + 1
    M = 0
    DO I = 2, N
       J = IAPHI(IA2)%A(I)
       IF (((JP(J).EQ.IA2).AND.(KP(J).EQ.IA3)).OR. &
            ((JP(J).EQ.IA3).AND.(KP(J).EQ.IA2))) THEN
          M = M + 1
          TEMPP(M) = J
       ENDIF
    ENDDO

    call chmalloc('mvdihe.src','PHILST','IBLSTP',M+1,intgp=IBLSTP%A)
    IBLSTP%A(1) = M + 1
    DO I = 1, M
       IBLSTP%A(I+1) = TEMPP(I)
    ENDDO
    call chmdealloc('mvdihe.src','PHILST','TEMPP',N-1,intg=TEMPP)
    RETURN
  END SUBROUTINE PHILST

#endif 
end module mcmvdihe

