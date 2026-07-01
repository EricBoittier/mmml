module mcmvcrot
#if KEY_MC==1
  use chm_types
  use dimens_fcm
  implicit none


contains

  SUBROUTINE MVCROT(COMLYN,COMLEN,NMVATM, &
       IPIVTP,IMVNGP,IBLSTP,MDXP,MBONDT,QBND, &
       ARMLIM,ARMMAX,ANISO,RMDX,X,Y,Z,WMAIN)
    !
    !       Determines move specific information for a concerted
    !       dihedral rotation.
    !
    !       Aaron R. Dinner
    !
#if KEY_PATHINT==1
    use mpathint, only: ibpimc, jbpimc  
#endif
    use psf
    use selctam
    use memory
    use mcmvutil
    use select
    use string
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
    INTEGER I, J
    INTEGER NNROT
    !       Maximum number of non-rotatable bond types in a single call.
    integer,parameter :: MNROT = 10
    type(chm_iptr) :: INROTIP(MNROT), INROTJP(MNROT)
    integer,allocatable,dimension(:) :: IBACKP, IPATHP, NPATHP
    type(chm_iptr) :: TEMPP
    LOGICAL LSELE, LPIMC

    LPIMC = (INDXA(COMLYN, COMLEN, 'PIMC') .GT. 0)

    !       Generate atom lists that keep track of which internal coordinates
    !       are affected by moving a given atom.
    !       Note that the bond list is also used as a connectivity table.
    CALL MKBNDT()

    !       The first SELE...END statement defines the backbone along which
    !       the rotatable dihedrals lie.
    call chmalloc('mvcrot.src','MVCROT','IBACKP',NATOM,intg=IBACKP)
    IBACKP(1:NATOM) = 0
    CALL SELCTA(COMLYN,COMLEN,IBACKP,X,Y,Z,WMAIN,.TRUE.)

    !       All following SELE...END statments define non-rotatable bonds
    !       which should be skipped.
    NNROT = 0
10  NNROT = NNROT + 1
    call chmalloc('mvcrot.src','MVCROT','INROTIP(NNROT)',NATOM,intgp=INROTIP(NNROT)%A)
    call chmalloc('mvcrot.src','MVCROT','INROTJP(NNROT)',NATOM,intgp=INROTJP(NNROT)%A)
    INROTIP(NNROT)%A(1:NATOM) = 0
    INROTJP(NNROT)%A(1:NATOM) = 0
    CALL SELCTA(COMLYN,COMLEN,INROTIP(NNROT)%A, &
         X,Y,Z,WMAIN,.TRUE.)
    LSELE = CKSELE(INROTIP(NNROT)%A,NATOM)
    IF (LSELE) THEN
       CALL SELCTA(COMLYN,COMLEN,INROTJP(NNROT)%A, &
            X,Y,Z,WMAIN,.TRUE.)
       LSELE = CKSELE(INROTJP(NNROT)%A,NATOM)
       IF (LSELE) GOTO 10
    ENDIF
    !       end of loop
    !       The last pair of INROTxP are empty, free them and decrement NNROT
    call chmdealloc('mvcrot.src','MVCROT','INROTJP(NNROT)',NATOM,intgp=INROTJP(NNROT)%A)
    call chmdealloc('mvcrot.src','MVCROT','INROTIP(NNROT)',NATOM,intgp=INROTIP(NNROT)%A)
    NNROT = NNROT - 1

    !       Count the number of elements that will be in the list.
    !       If there are no move locations, no point in the move, so return.
    call chmalloc('mvcrot.src','MVCROT','IPATHP',NATOM,intg=IPATHP)
    call chmalloc('mvcrot.src','MVCROT','NPATHP',NATOM,intg=NPATHP)
#if KEY_PATHINT==1
    IF (LPIMC) THEN
       CALL GTCROT(NMVATM,IPIVTP,IMVNGP,IBLSTP,MDXP, &
            RMDX, IPATHP, NPATHP, &
            IBACKP, NNROT,INROTIP,INROTJP, &
            IAPIMP, IAPHIP, NATOM, &
            IBPIMC,JBPIMC,JP,KP,0, &
            .FALSE.)
    ELSE
#endif 
       CALL GTCROT(NMVATM,IPIVTP,IMVNGP,IBLSTP,MDXP, &
            RMDX, IPATHP, NPATHP, &
            IBACKP, NNROT,INROTIP,INROTJP, &
            IABNDP, IAPHIP, NATOM,IB,JB,JP,KP,0, &
            .FALSE.)
#if KEY_PATHINT==1
    ENDIF 
#endif
    IF (NMVATM .EQ. 0) RETURN

    !       Allocate the space
    allocate(IPIVTP%A(NMVATM))
    allocate(IMVNGP%A(NMVATM))
    allocate(IBLSTP%A(NMVATM))
    call chmalloc('mvcrot.src','MVCROT','MDXP',NMVATM,crlp=MDXP%A)

    !       Fill the lists.
#if KEY_PATHINT==1
    IF (LPIMC) THEN
       CALL GTCROT(NMVATM,IPIVTP,IMVNGP,IBLSTP,MDXP, &
            RMDX, IPATHP, NPATHP, &
            IBACKP, NNROT,INROTIP,INROTJP, &
            IAPIMP, IAPHIP, NATOM, &
            IBPIMC,JBPIMC,JP,KP,1, &
            .FALSE.)
       !         Do the bonded information here.
       CALL FILLQB(QBND,.TRUE.,.TRUE.,.TRUE.,.TRUE.,.FALSE.)
       DO I = 1, NMVATM
          CALL PIMCBD(TEMPP, IPIVTP%A(I)%A)
          IBLSTP%A(I)%A => TEMPP%A
       ENDDO
    ELSE
#endif 
       CALL GTCROT(NMVATM,IPIVTP,IMVNGP,IBLSTP,MDXP, &
            RMDX, IPATHP, NPATHP, &
            IBACKP, NNROT,INROTIP,INROTJP, &
            IABNDP, IAPHIP, NATOM,IB,JB,JP,KP,1, &
            .TRUE.)
       CALL FILLQB(QBND,.FALSE.,.FALSE.,.TRUE.,.FALSE.,.FALSE.)
#if KEY_PATHINT==1
    ENDIF 
#endif

    !       Set some more global parameters
    ARMLIM = .TRUE.
    ARMMAX = 180.0

    !       Free arrays used for depth-first search in gtcrot
    call chmdealloc('mvcrot.src','MVCROT','NPATHP',NATOM,intg=NPATHP)
    call chmdealloc('mvcrot.src','MVCROT','IPATHP',NATOM,intg=IPATHP)

    !       Free the space used to hold bonded term to atom information.
    CALL FREBND()

    !       Free non-rotatable bond selections
    DO I = 1, NNROT
       call chmdealloc('mvcrot.src','MVCROT','INROTJP(I)',NATOM,intgp=INROTJP(I)%A)
       call chmdealloc('mvcrot.src','MVCROT','INROTIP(I)',NATOM,intgp=INROTIP(I)%A)
    ENDDO
    !       Free backbone selection
    call chmdealloc('mvcrot.src','MVCROT','IBACKP',NATOM,intg=IBACKP)

    RETURN
  END SUBROUTINE MVCROT

  SUBROUTINE GTCROT(NMVATM,IPIVTP,IMVNGP,IBLSTP,MDXP, &
       RMDX,IPATH,NPATH, &
       IBACK,NNROT,INROTIP,INROTJP, &
       IABND,IAPHI,NATOM,IB,JB,JP,KP,MODE,LBND)
    !
    !       Counts and assigns move instances given the appropriate
    !       number of atom selections.
    !
    !       Works by a depth first search given the connectivity list.
    !
    !       Note that there is no screening for symmetric moves, these
    !       could be eliminated by specifying that added atoms had to be
    !       increasing in index.
    !
    !       Aaron R. Dinner
    !
    use memory
    use mcmvdihe, only: philst
    use mcmvutil

    !       Passed Variables
    !
    INTEGER NMVATM
    type(iptr_ptr) :: IMVNGP, IBLSTP, IPIVTP
    type(chm_ptr) :: MDXP
    INTEGER NATOM, MODE
    INTEGER IPATH(*), NPATH(*)
    INTEGER IBACK(*), NNROT
    type(chm_iptr) :: INROTIP(:), INROTJP(:)
    type(chm_iptr) :: IABND(:), IAPHI(:)
    INTEGER IB(*), JB(*), JP(*), KP(*)
    real(chm_real)  RMDX
    LOGICAL LBND
    !
    !       Local Variables
    !
    INTEGER I, J, K, IM, IATOM, ILAST
    INTEGER NRBND, IBOND(2,7)
    INTEGER NB
    type(chm_iptr) :: TEMPP, LISTP
    integer,allocatable,dimension(:) :: IMVATP, IMAP, NMAP

    logical :: is_rotatable

    NMVATM = 0
    IF (MODE .EQ. 1) THEN
       call chmalloc('mvcrot.src','GTCROT','IMVATP',NATOM,intg=IMVATP)
       call chmalloc('mvcrot.src','GTCROT','IMAP',NATOM,intg=IMAP)
       call chmalloc('mvcrot.src','GTCROT','NMAP',NATOM,intg=NMAP)
    ENDIF
    !
    !       IM is the present atom in the path.
    !       IPATH is the list of atoms the path traverses.
    !       NPATH is the number of bonds already tried for that atom.
    !
    DO IATOM = 1, NATOM
       NRBND = 0
       IF (IBACK(IATOM) .GT. 0) THEN
          IM = 1
          IBACK(IATOM) = 2
          IPATH(IM) = IATOM
          NPATH(IM) = 0

10        NPATH(IM) = NPATH(IM) + 1
          !           Try the next move for the current atom.
          IF (NPATH(IM) .GT. IABND(IPATH(IM))%A(1)) GOTO 100
          I = IABND(IPATH(IM))%A(NPATH(IM) + 1)
          I = NOTATM(IB(I),JB(I),IPATH(IM))
          IF (IBACK(I) .NE. 1) GOTO 10

          !           Check if the bond is rotatable.
          is_rotatable = rotabl(ipath(im), i, nnrot, inrotip, inrotjp)
          if (is_rotatable) then
             NRBND = NRBND + 1
             IBOND(1,NRBND) = IPATH(IM)
             IBOND(2,NRBND) = I
          ELSE IF ((MOD(NRBND,2) .EQ. 0).AND.(NRBND .NE. 2)) THEN
             !             The root solving algorithm requires that bond pairs
             !             3-4, 5-6 each be connected (they are indexed one higher here!).
             !             Also need to go back if NRBND is 0
             GOTO 10
          ENDIF

          !           Mark the site
          IBACK(I) = 2
          IM = IM + 1
          IPATH(IM) = I
          NPATH(IM) = 0

          IF (NRBND .EQ. 7) THEN
             IF (MODE .EQ. 0) THEN
                NMVATM = NMVATM + 1
             ELSE IF (MODE .EQ. 1) THEN
                NMVATM = NMVATM + 1
                call chmalloc('mvcrot.src','GTCROT','TEMPP',14,intgp=TEMPP%A)

                J = 0
                DO K = 1, 7
                   J = J + 1
                   TEMPP%A(J) = IBOND(1,K)
                   J = J + 1
                   TEMPP%A(J) = IBOND(2,K)
                ENDDO
                IPIVTP%A(NMVATM)%A => TEMPP%A

                !               Generate the bonded information.
                IF (LBND) THEN
                   DO K = 1, 7
                      CALL PHILST(TEMPP,IBOND(1,K),IBOND(2,K),JP,KP, &
                           IAPHI)
                      CALL IBLADD(IBLSTP%A, NMVATM, K, TEMPP)
                   ENDDO
                ENDIF

                !               Search along each bond off of the second atom on
                !               the bond.  Stop at the second atom of the next bond.
                !               Note that we only need to do the first 6 bonds since the
                !               atoms in front of the last dihedral are outside the window.
                DO K = 1, 6
                   IMVATP(1:NATOM) = 0
                   !                 Mark the stopping point
                   IMVATP(IBOND(2,K+1)) = 1
                   NB = IABND(IBOND(2,K))%A(1) + 1
                   DO I = 2, NB
                      J = IABND(IBOND(2,K))%A(I)
                      J = NOTATM(IB(J),JB(J),IBOND(2,K))
                      IF (J .NE. IBOND(1,K) .AND. &
                           J .NE. IBOND(2,K+1)) THEN
                         CALL MVATM(IMVATP, NATOM,J,IBOND(2,K), &
                              IABND,IB,JB, IMAP, NMAP)
                      ENDIF
                   ENDDO

                   !                 Run through the list to store the moving atoms
                   !                 make this so that it adds to end of previous list
                   CALL IMVLST(TEMPP, NATOM, IMVATP)
                   CALL IMVADD(LISTP, K, TEMPP)
                   IMVNGP%A(NMVATM)%A => LISTP%A

                ENDDO

                !               Assign some limits
                MDXP%A(NMVATM) = RMDX

             ELSE
                CALL WRNDIE (-2, '<GTDIHE>', 'UNKNOWN MODE')
             ENDIF
             IBACK(IPATH(IM)) = 1
             IM = IM - 1
             !             The last bond will always be rotatable since it will be
             !             the one to take NRBND from 6 to 7.
             NRBND = NRBND - 1
          ENDIF

          GOTO 10

100       CONTINUE

          !           Have tried all possible moves for this atom --- move back.
          NPATH(IM) = 0
          IBACK(IPATH(IM)) = 1
          ILAST = IPATH(IM)
          IM = IM - 1
          IF (IM .GT. 0) THEN
             is_rotatable = ROTABL(IPATH(IM),ILAST,NNROT,INROTIP,INROTJP)
             if (is_rotatable) nrbnd = nrbnd - 1
             GOTO 10
          ENDIF

       ENDIF
    ENDDO

    IF (MODE .EQ. 1) THEN
       call chmdealloc('mvcrot.src','GTCROT','NMAP',NATOM,intg=NMAP)
       call chmdealloc('mvcrot.src','GTCROT','IMAP',NATOM,intg=IMAP)
       call chmdealloc('mvcrot.src','GTCROT','IMVATP',NATOM,intg=IMVATP)
    ENDIF

    RETURN
  END SUBROUTINE GTCROT

  SUBROUTINE PIMCBD(TEMPP, IPIVOT)
    !
    !       Gets the bonded term information for a path integral
    !       concerted rotation move.
    !
    !       Aaron R. Dinner 00-11-28
    !
    use memory
    use mcmvrtrn, only: assibl
    use mcmvutil
    type(chm_iptr) :: TEMPP
    INTEGER IPIVOT(*)
    INTEGER IA, I, J, NE
    INTEGER NBOLD, NBNEW, NTOLD, NTNEW
    INTEGER NPOLD, NPNEW, NIOLD, NINEW
    integer,pointer,dimension(:) :: IBOLD, ITOLD, IPOLD, IIOLD
    type(chm_iptr) :: IBNEW, ITNEW, IPNEW, IINEW

    !       Bonds
    IA = IPIVOT(2)
    IBOLD => IABNDP(IA)%A
    NBOLD = IBOLD(1) + 1
    DO I = 4, 12, 2
       IA = IPIVOT(I)
       CALL UNQBND(IBNEW, NBNEW, IBOLD, IABNDP(IA)%A)
       IF (I > 4) call chmdealloc('mvcrot.src','PIMCBD','IBOLD',NBOLD,intgp=IBOLD)
       IBOLD => IBNEW%A
       NBOLD = NBNEW
    ENDDO

    !       Angles
    IA = IPIVOT(2)
    ITOLD => IATHTP(IA)%A
    NTOLD = ITOLD(1) + 1
    DO I = 4, 12, 2
       IA = IPIVOT(I)
       CALL UNQBND(ITNEW, NTNEW, ITOLD, IATHTP(IA)%A)
       IF (I > 4) call chmdealloc('mvcrot.src','PIMCBD','ITOLD',NTOLD,intgp=ITOLD)
       ITOLD => ITNEW%A
       NTOLD = NTNEW
    ENDDO

    !       Dihedrals
    IA = IPIVOT(2)
    IPOLD => IAPHIP(IA)%A
    NPOLD = IPOLD(1) + 1
    DO I = 4, 12, 2
       IA = IPIVOT(I)
       CALL UNQBND(IPNEW, NPNEW, IPOLD, IAPHIP(IA)%A)
       IF (I > 4) call chmdealloc('mvcrot.src','PIMCBD','IPOLD',NPOLD,intgp=IPOLD)
       IPOLD => IPNEW%A
       NPOLD = NPNEW
    ENDDO

    !       Impropers
    IA = IPIVOT(2)
    IIOLD => IAIMPP(IA)%A
    NIOLD = IIOLD(1) + 1
    DO I = 4, 12, 2
       IA = IPIVOT(I)
       CALL UNQBND(IINEW, NINEW, IIOLD, IAIMPP(IA)%A)
       IF (I > 4) call chmdealloc('mvcrot.src','PIMCBD','IIOLD',NIOLD,intgp=IIOLD)
       IIOLD => IINEW%A
       NIOLD = NINEW
    ENDDO

    NE = NBOLD + NTOLD + NPOLD + NIOLD
    call chmalloc('mvcrot.src','PIMCBD','TEMPP',NE,intgp=TEMPP%A)

    NE = 1
    CALL ASSIBL(NE, NBOLD, IBOLD, TEMPP%A)
    NE = NE + 1
    CALL ASSIBL(NE, NTOLD, ITOLD, TEMPP%A)
    NE = NE + 1
    CALL ASSIBL(NE, NPOLD, IPOLD, TEMPP%A)
    NE = NE + 1
    CALL ASSIBL(NE, NIOLD, IIOLD, TEMPP%A)

    call chmdealloc('mvcrot.src','PIMCBD','IBOLD',NBOLD,intgp=IBOLD)
    call chmdealloc('mvcrot.src','PIMCBD','ITOLD',NTOLD,intgp=ITOLD)
    call chmdealloc('mvcrot.src','PIMCBD','IPOLD',NPOLD,intgp=IPOLD)
    call chmdealloc('mvcrot.src','PIMCBD','IIOLD',NIOLD,intgp=IIOLD)

    RETURN
  END SUBROUTINE PIMCBD

  SUBROUTINE UNQBND(IRETP,NB,IAB1,IAB2)
    !
    !       Gets the unique bonded terms from two bonded term lists.
    !
    !       Aaron R. Dinner 00-11-28
    !
    use memory
    type(chm_iptr) :: IRETP
    INTEGER NB, IAB1(:), IAB2(:)
    !
    INTEGER I, N1, N2, ICURR1, ICURR2
    integer,allocatable,dimension(:) :: ITEMPP

    N1 = IAB1(1) + 1
    N2 = IAB2(1) + 1
    ICURR1 = 2
    ICURR2 = 2

    call chmalloc('mvcrot.src','UNQBND','ITEMPP',N1+N2-2,intg=ITEMPP)
    NB = 0

10  IF ((ICURR1.LE.N1).AND.(ICURR2.LE.N2)) THEN
       IF (IAB1(ICURR1) .LE. IAB2(ICURR2)) THEN
          NB = NB + 1
          ITEMPP(NB) = IAB1(ICURR1)
          IF (IAB1(ICURR1) .EQ. IAB2(ICURR2)) ICURR2 = ICURR2 + 1
          ICURR1 = ICURR1 + 1
       ELSE
          NB = NB + 1
          ITEMPP(NB) = IAB2(ICURR2)
          ICURR2 = ICURR2 + 1
       ENDIF
       GOTO 10
    ENDIF

    DO I = ICURR1, N1
       NB = NB + 1
       ITEMPP(NB) = IAB1(I)
    ENDDO

    DO I = ICURR2, N2
       NB = NB + 1
       ITEMPP(NB) = IAB2(I)
    ENDDO

    call chmalloc('mvcrot.src','UNQBND','IRETP',NB+1,intgp=IRETP%A)
    IRETP%A(1) = NB
    DO I = 1, NB
       IRETP%A(I+1) = ITEMPP(I)
    ENDDO
    NB = NB + 1
    call chmdealloc('mvcrot.src','UNQBND','ITEMPP',N1+N2-2,intg=ITEMPP)

    RETURN
  END SUBROUTINE UNQBND

  LOGICAL FUNCTION CKSELE(INROT,NATOM)
    !
    !       Checks if all atoms in a selection are selected, indicating
    !       whether there was no SELE...END statment.
    !
    !       True if there was a selection, false if there was not.
    !
    !       Aaron R. Dinner
    !
    INTEGER INROT(*), NATOM
    !
    INTEGER I, N
    !
    N = 0
    DO I = 1, NATOM
       IF (INROT(I) .EQ. 1) N = N + 1
    ENDDO
    CKSELE = N .LT. NATOM
    RETURN
  END FUNCTION CKSELE

  LOGICAL FUNCTION ROTABL(IA,JA,NNROT,INROTIP,INROTJP)
    !
    !       True if the user wants the bond to be rotatable,
    !       false otherwise.
    !
    !       Note that there is not checking for whether the it is actually
    !       rotatable according to the PSF.
    !
    !       Aaron R. Dinner
    !
    use stream

    INTEGER NNROT
    INTEGER IA, JA
    type(chm_iptr) :: INROTIP(NNROT), INROTJP(NNROT)
    !
    INTEGER I
    !
    !       Run through the non-rotable list and check each
    !       backwards and forwards.
    !
    DO I = 1, NNROT
       IF ((INROTIP(I)%A(IA) == 1 .AND. INROTJP(I)%A(JA) == 1) .OR. &
            (INROTIP(I)%A(JA) == 1 .AND. INROTJP(I)%A(IA) == 1)) THEN
          ROTABL = .FALSE.
          RETURN
       ENDIF
    ENDDO

    !       If we got to here, the user wants to let the bond rotate.
    ROTABL = .TRUE.

    RETURN
  END FUNCTION ROTABL

  SUBROUTINE IBLADD(IBLST,IDX,ICALL,IAPNDP)
    !
    !       Adds the list that stores the changing bonded terms
    !       to end of previous list.
    !
    !       Aaron R. Dinner
    !
    use memory
    type(chm_iptr) :: IBLST(:), IAPNDP
    INTEGER IDX, ICALL

    INTEGER NGOLD, NGNEW, N, I, J
    integer,pointer,dimension(:) :: TEMPP, NEWP

    IF (ICALL .EQ. 1) THEN
       IBLST(IDX)%A => IAPNDP%A
       RETURN
    ENDIF

    TEMPP => IBLST(IDX)%A
    NGOLD = TEMPP(1)
    NGNEW = IAPNDP%A(1)
    N = NGOLD + NGNEW - 1

    call chmalloc('mvcrot.src','IBLADD','NEWP',N,intgp=NEWP)
    J = 1
    DO I = 2, NGOLD
       J = J + 1
       NEWP(J) = TEMPP(I)
    ENDDO
    DO I = 2, NGNEW
       J = J + 1
       NEWP(J) = IAPNDP%A(I)
    ENDDO
    NEWP(1) = J

    IBLST(IDX)%A => NEWP
    call chmdealloc('mvcrot.src','IBLADD','TEMPP',NGOLD,intgp=TEMPP)
    call chmdealloc('mvcrot.src','IBLADD','IAPNDP',NGNEW,intgp=IAPNDP%A)

    RETURN
  END SUBROUTINE IBLADD

  SUBROUTINE MKCROT(DP,QMOVE,ISEED,IDX,IPIVT,IMVNG,RMDX,NLIMIT, &
       X,Y,Z)
    !
    !       Makes a concerted dihedral rotation.
    !
    !       Aaron R. Dinner
    !
    use clcg_mod,only:random
    use consta
    use number
    use corsubs,only:fndu
    use mcmvrtrn, only: applyu
    use mcmvutil, only: r2indx

    INTEGER ISEED, IDX, NLIMIT
    type(chm_iptr) :: IPIVT(:), IMVNG(:)
    real(chm_real) :: RMDX(:), DP
    real(chm_real)  X(*), Y(*), Z(*)
    LOGICAL QMOVE
    !
    INTEGER I, J, K, L, NN, NG, IS, IAF, IAL
    INTEGER IATM(0:6,2)
    INTEGER JMAX, MXROOT, NP, NPI, NPS, NPIS
    INTEGER NRTMAX, NRET
    real(chm_real)  BL(0:6), TMAT(3,3,0:5), PV(0:5,3)
    real(chm_real)  Q1(3), Q2(3), Q0(3)
    real(chm_real)  S1(3), U1(3), S2(3), U2(3)
    real(chm_real)  TOL, ETOL, DTOL, R
    real(chm_real)  RMAT0(3,3), CP0, SP0
    real(chm_real)  PNEW(5), POLD(5)
    real(chm_real)  U(9), RN(3)
    real(chm_real)  P1, P1MIN, P1MAX, DP2
    LOGICAL LOK, LERROR
    PARAMETER (MXROOT = 40, JMAX = 60)
    PARAMETER (TOL = 1.0E-10, ETOL = 1.0E-17, DTOL = 1.0E-14)
    PARAMETER (NPS = 3600, NPIS = 10, NRTMAX = 6)
    INTEGER IR(MXROOT), N1, N2, NP1, IROOT
    INTEGER I2(MXROOT)
    real(chm_real)  XR(MXROOT), RJAC(MXROOT), RJTOT
    real(chm_real)  X2(MXROOT)

    real(chm_real)  F5(4), COSP(4,5), SINP(4,5), DPSAVE, DPTOT
    LOGICAL LF(4), LB(4), LRIG12

    !       It is argued in Hoffman and Knapp (1996) that the maximum
    !       number of roots for the peptide case is 16 for each position of
    !       the driver dihedral.  So, MXROOT = 32 is probably OK, but I put
    !       it to 40 for safety.

    !       Get some parameter matrices and the contraint vectors.
    !       Everything is indexed from 0 for keeping with Go and Scheraga (1970)
    J = 0
    DO I = 0, 6
       J = J + 1
       IATM(I,1) = IPIVT(IDX)%A(J)
       J = J + 1
       IATM(I,2) = IPIVT(IDX)%A(J)
    ENDDO

    CALL CRFIXD(TMAT,S1,U1,Q0,Q1,Q2,IATM,X,Y,Z,PV,BL)

    !       If bonds 1 and 2 are not connected Q0 depends on P1 (not fixed)
    LRIG12 = (IATM(1,2) .NE. IATM(2,1))

    !       Get the old dihedrals.
    CALL GTPOLD(POLD,IATM,X,Y,Z)

    !       Initialize the PHI1 search range
    IF (NLIMIT .GT. 0) THEN
       P1 = POLD(1)
       IF (IATM(0,2).NE.IATM(1,1)) P1 = P1 - PI
       DP = RMDX(IDX)*PI/180.0
       P1MIN = P1 - DP
       P1MAX = P1 + DP
    ELSE
       DP = PI
       P1MIN = -PI
       P1MAX =  PI
    ENDIF
    !       NP and NPI are guesses for good subdivision ranges
    NP  = INT(DP*NPS/PI)
    NPI = NPIS
    LERROR = .FALSE.
    NRET = 0

    !       Find all the available solutions with the
    !       driver dihedral in its original position.

10  CALL CRROOT(XR,IR,N1,P1MIN,P1MAX,NP,NPI,TOL,ETOL,DTOL,JMAX, &
         TMAT,S1,U1,Q0,Q1,Q2,5,PV,LRIG12,LERROR)
    IF (LERROR .OR. (N1 .EQ. 0) .OR. &
         ((NLIMIT.EQ.0).AND.(MOD(N1,2).EQ.1))) THEN
       NRET = NRET + 1
       NP = 2*NP
       NPI = 2*NPI
       LERROR = .FALSE.
       IF (NRET .GT. NRTMAX) THEN
          CALL WRNDIE (3,'<MKCROT>', &
               'CANNOT FIND ALL ROOTS, RETURNING')
          RETURN
       ENDIF
       GOTO 10
    ENDIF

    !       Rotate the two constraining vectors (SV and UV) around
    !       the driver dihedral.  These should be in the frame of bond 0,
    !       so we need apply only a simple yz rotation matrix.
    DPSAVE = (2.0*RANDOM(ISEED) - 1.0)*RMDX(IDX)
    DP = DPSAVE*PI/180.0
    CP0 = COS(DP)
    SP0 = SIN(DP)
    !       This is actually a rotation of -DP.
    CALL GTRMAT(RMAT0,CP0,SP0)
    CALL APMATX(S2(1),S2(2),S2(3),RMAT0,S1(1),S1(2),S1(3))
    CALL APMATX(U2(1),U2(2),U2(3),RMAT0,U1(1),U1(2),U1(3))

    !       Find all the available solutions with the new constraints.
    !       Use the same NP and NPI as above since the function should be similar.
    NP1 = N1 + 1
20  CALL CRROOT(X2,I2,N2,P1MIN,P1MAX,NP,NPI,TOL,ETOL,DTOL,JMAX, &
         TMAT,S2,U2,Q0,Q1,Q2,5,PV,LRIG12,LERROR)
    IF (LERROR.OR.((NLIMIT.EQ.0).AND.(MOD(N2,2).EQ.1))) THEN
       NRET = NRET + 1
       NP = 2*NP
       NPI = 2*NPI
       LERROR = .FALSE.
       IF (NRET .GT. NRTMAX) THEN
          CALL WRNDIE (3,'<MKCROT>', &
               'CANNOT FIND ALL ROOTS, RETURNING')
          RETURN
       ENDIF
       GOTO 10
    ENDIF

    DO I = 1, N2
       NP1 = N1 + I
       XR(NP1) = X2(I)
       IR(NP1) = I2(I)
    ENDDO
    N2 = N1 + N2

    !       ARD and Jie Hu 05-06-28
    !       Screen roots to eliminate those with too large a change
129 CALL GTPNEW(PNEW,IATM,XR(I),IR(I),TMAT,S1,U1,Q0,Q1,Q2,PV,LRIG12)
    DO J = 2, NLIMIT
       !         Determine DP = PNEW - POLD
       DP2 = PNEW(J) - POLD(J)
       !         If ABS(DP) > RMDX then eliminate root (adjust XR, IR, N1, N2)
       IF(ABS(DP2) .GT. RMDX(IDX)) THEN
          DO K = I + 1, N2
             XR(K - 1) = XR(K)
             IR(K - 1) = IR(K)
          ENDDO
          IF (I .LE. N1) N1 = N1 - 1
          N2 = N2 - 1
          I = I - 1
          GOTO 130
       ENDIF
    ENDDO
130 I = I + 1
    IF (I .LE. N2) GOTO 129

    IF (N1 .EQ. 0 .AND. N2 .EQ. 0) THEN
       QMOVE = .FALSE.
       DP = DPSAVE
       RETURN
    ENDIF

    !       Calculate the Jacobians of all the roots.  Keep track of their
    !       total as we go along.  Be careful about 0 position!
    RJTOT = ZERO
    DO I = 1, N2
       IF (I .LE. N1) THEN
          RJAC(I) = CRJACO(XR(I),IR(I),S1,U1,TMAT,Q0,Q1,Q2,PV,BL, &
               LRIG12)
       ELSE
          RJAC(I) = CRJACO(XR(I),IR(I),S2,U2,TMAT,Q0,Q1,Q2,PV,BL, &
               LRIG12)
       ENDIF
       RJTOT = RJTOT + RJAC(I)
    ENDDO

    !       Choose a random number between 0 and the Jacobian total.
    !       Find the root to which this choice corresponds.
    R = RJTOT*RANDOM(ISEED)
    IROOT = R2INDX(R,RJAC,N2)

    !       Get the new dihedrals.
    IF (IROOT .LE. N1) THEN
       DP = ZERO
       CALL GTPNEW(PNEW,IATM,XR(IROOT),IR(IROOT),TMAT,S1,U1,Q0,Q1,Q2, &
            PV,LRIG12)
       !         Note that it is possible to pick the original state --- this is
       !         necessary for detailed balance --- so check if further energy
       !         calculations will be needed.
       DPTOT = ZERO
       DO I = 1, 5
          DPTOT = DPTOT + ABS(POLD(I)-PNEW(I))
       ENDDO
       IF (DPTOT .LT. RSMALL) THEN
          QMOVE = .FALSE.
          DP = DPSAVE
          RETURN
       ENDIF
    ELSE
       CALL GTPNEW(PNEW,IATM,XR(IROOT),IR(IROOT),TMAT,S2,U2,Q0,Q1,Q2, &
            PV,LRIG12)
    ENDIF

    !       Apply the rotations starting with the driver angle.
    DO I = 0, 5
       !         DP is set up for 0 case above.
       IF (I .GT. 0) DP = POLD(I) - PNEW(I)
       DP = DP*180.0/PI
       RN(1) = X(IATM(I,2)) - X(IATM(I,1))
       RN(2) = Y(IATM(I,2)) - Y(IATM(I,1))
       RN(3) = Z(IATM(I,2)) - Z(IATM(I,1))
       CALL FNDU(U,RN,DP,LOK)
       NG = IMVNG(IDX)%A(1)
       IF (I .GT. 0) THEN
          IS = IMVNG(IDX)%A(I+1) + 2
       ELSE
          IS = NG + 2
       ENDIF
       DO J = 2+I, NG
          NN = IMVNG(IDX)%A(J)
          DO K = IS, NN, 2
             IAF = IMVNG(IDX)%A(K - 1)
             IAL = IMVNG(IDX)%A(K)
             DO L = IAF, IAL
                CALL APPLYU(U,X,Y,Z,L, &
                     X(IATM(I,2)),Y(IATM(I,2)),Z(IATM(I,2)))
             ENDDO
          ENDDO
          IS = NN + 2
       ENDDO
    ENDDO

    !       Return DP to zero dihedral for ARM or DOMC
    DP = DPSAVE

    RETURN
  END SUBROUTINE MKCROT

  FUNCTION ANGLEF(CP,SP) result(anglef_1)
    !
    !       Gets the angle from the sine and cosine
    !
    !       Aaron R. Dinner
    !
    use number
    use consta

    real(chm_real) :: anglef_1
    real(chm_real) :: CP, SP
    !       Check for rounding error
    IF (ABS(CP) .GE. ONE) CP = SIGN(SQRT(ONE-SP*SP),CP)
    ANGLEF_1 = ACOS(CP)
    IF (ANGLEF_1 .GT. ZERO) THEN
       IF (SP .LT. ZERO) ANGLEF_1 = -ANGLEF_1
    ELSE
       IF (SP .GT. ZERO) THEN
          ANGLEF_1 = PI + ANGLEF_1
       ELSE
          ANGLEF_1 = PI - ANGLEF_1
       ENDIF
    ENDIF
    RETURN
  END FUNCTION ANGLEF

  SUBROUTINE GTPNEW(PNEW,IATM,P1,IB,TMAT,SV,UV,Q0,Q1,Q2,PV,LRIG12)
    !
    !       Gets the new dihedral values given the sines and cosines
    !
    !       Aaron R. Dinner
    !
    use consta
    INTEGER IB, IATM(0:6,2)
    real(chm_real)  P1, SV(3), UV(3), PNEW(5), TMAT(3,3,0:5)
    real(chm_real)  Q0(3), Q1(3), Q2(3), PV(0:5,3)
    LOGICAL LRIG12
    !
    INTEGER I
    real(chm_real)  F5(4), COSP(4,5), SINP(4,5), RMAT(3,3,5)
    LOGICAL LF(4), LB(4)

    DO I = 1, 4
       LB(I) = .FALSE.
    ENDDO
    LB(IB) = .TRUE.

    CALL CRFUNC(F5,LF,COSP,SINP,5,P1,TMAT,SV,UV,Q0,Q1,Q2,LB, &
         PV,LRIG12)
    CALL GTRMAT(RMAT(1,1,1),COSP(1, 1),SINP(1, 1))
    CALL GTRMAT(RMAT(1,1,2),COSP(IB,2),SINP(IB,2))
    CALL GTRMAT(RMAT(1,1,3),COSP(IB,3),SINP(IB,3))
    CALL GTRMAT(RMAT(1,1,4),COSP(IB,4),SINP(IB,4))
    !       Get Phi5
    CALL CRPHI5(COSP(IB,5),SINP(IB,5),UV,TMAT,RMAT)
    CALL GTRMAT(RMAT(1,1,5),COSP(IB,5),SINP(IB,5))

    !
    !       Need to subtract pi to bonds following non-rotatable ones
    !
    PNEW(1) = P1
    IF (IATM(0,2).NE.IATM(1,1)) PNEW(1) = PNEW(1) - PI
    DO I = 2, 5
       PNEW(I) = ANGLEF(COSP(IB,I),SINP(IB,I))
       IF (IATM(I-1,2).NE.IATM(I,1)) PNEW(I) = PNEW(I) - PI
    ENDDO
    RETURN
  END SUBROUTINE GTPNEW

  SUBROUTINE GTPOLD(POLD,IATM,X,Y,Z)
    !
    !       Gets the old dihedral values.
    !
    !       Aaron R. Dinner
    !
    INTEGER IATM(0:6,2)
    real(chm_real)  POLD(5), X(*), Y(*), Z(*)
    !
    INTEGER I, I1, I2

    DO I = 1, 5
       IF (IATM(I-1,2) .NE. IATM(I,1)) THEN
          I1 = IATM(I-1,2)
       ELSE
          I1 = IATM(I-1,1)
       ENDIF
       IF (IATM(I+1,1) .NE. IATM(I,2)) THEN
          I2 = IATM(I+1,1)
       ELSE
          I2 = IATM(I+1,2)
       ENDIF
       POLD(I) = DIHEF(I1,IATM(I,1),IATM(I,2),I2,X,Y,Z)
    ENDDO
    RETURN
  END SUBROUTINE GTPOLD

  SUBROUTINE CRROOT(XR,IBR,NR,P1MIN,P1MAX,NP,NPI,TOL, &
       ETOL,DTOL,JMAX, &
       TMAT,SV,UV,Q0,Q1,Q2,IFUNC,PV,LRIG12,LERROR)
    !
    !       Bracket the roots of the F5 equation
    !
    !       Need to deal with end points of the branches so don't miss
    !       roots there.
    !
    !       Aaron R. Dinner
    !
    use consta
    !       Passed
    INTEGER NP, NPI, NR
    INTEGER IBR(*), JMAX, IFUNC
    real(chm_real)  XR(*), TOL, DTOL, ETOL
    real(chm_real)  P1MIN, P1MAX
    real(chm_real)  TMAT(3,3,0:5)
    real(chm_real)  Q1(3), Q2(3), Q0(3), PV(0:5,3)
    real(chm_real)  SV(3), UV(3)
    LOGICAL LERROR, LRIG12
    !
    INTEGER IP1, IB
    real(chm_real) P1, DP1, F5(4), FP(4), B1
    real(chm_real) PP, P11(4), P12(4)
    real(chm_real) COSP(4,5), SINP(4,5)
    LOGICAL LF(4), LP(4), LB(4)

    NR = 0
    DP1 = (P1MAX-P1MIN)/FLOAT(NP)

    !       Initialization
    P1 = P1MIN
    LB(1:4) = .TRUE.
    DO IB = 1, 4
       CALL CRFUNC(F5,LF,COSP,SINP,IFUNC,P1,TMAT,SV,UV,Q0,Q1,Q2,LB, &
            PV,LRIG12)
       FP(IB) = F5(IB)
       LP(IB) = LF(IB)
       IF (LF(IB)) P11(IB) = P1
    ENDDO

    DO IP1 = 1, NP
       PP = P1
       P1 = P1 + DP1
       CALL CRFUNC(F5,LF,COSP,SINP,IFUNC,P1,TMAT,SV,UV,Q0,Q1,Q2,LB, &
            PV,LRIG12)

       !         For each branch
       DO IB = 1, 4

          !           Find the endpoints of the function in this subrange
          IF (LF(IB) .AND. .NOT. LP(IB)) THEN
             CALL FNDEND(B1,IB,P1,PP,ETOL,JMAX, &
                  TMAT,SV,UV,Q0,Q1,Q2,IFUNC,PV,LRIG12)
             P11(IB) = B1
          ELSE IF (LP(IB) .AND. .NOT. LF(IB)) THEN
             CALL FNDEND(B1,IB,P1,PP,ETOL,JMAX, &
                  TMAT,SV,UV,Q0,Q1,Q2,IFUNC,PV,LRIG12)
             P12(IB) = B1
             !             Search interval P11 to P12 for roots.
             CALL GTROOT(XR,IBR,NR,IB,P11(IB),P12(IB),NPI,TOL,DTOL, &
                  JMAX,TMAT,SV,UV,Q0,Q1,Q2,IFUNC,PV,LRIG12, &
                  LERROR)
             IF (LERROR) RETURN
          ENDIF

          FP(IB) = F5(IB)
          LP(IB) = LF(IB)
       ENDDO
    ENDDO

    DO IB = 1, 4
       !         If necessary, get the last interval
       IF (LF(IB)) THEN
          P12(IB) = P1
          !           Search interval P11 to P12 for roots.
          CALL GTROOT(XR,IBR,NR,IB,P11(IB),P12(IB),NPI,TOL,DTOL, &
               JMAX,TMAT,SV,UV,Q0,Q1,Q2,IFUNC,PV,LRIG12, &
               LERROR)
          IF (LERROR) RETURN
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE CRROOT

  SUBROUTINE GTROOT(XR,IBR,NR,IB,P1,P2,NP,TOL,DTOL,JMAX, &
       TMAT,SV,UV,Q0,Q1,Q2,IFUNC,PV,LRIG12, &
       LERROR)
    !
    !       Search an interval for a root.
    !
    !       Aaron R. Dinner
    !
    INTEGER IB, NP, IBR(*), NR, IFUNC, JMAX
    real(chm_real) P1, P2, TOL, XR(*), DTOL
    real(chm_real) TMAT(3,3,0:5),Q1(3),Q2(3),Q0(3),SV(3),UV(3), &
         PV(0:5,3)
    LOGICAL LERROR, LRIG12

    INTEGER I, J, NS, NROOT
    real(chm_real)  XROOT(2), P, PP, DP
    LOGICAL LB(4)

    DO I = 1, 4
       LB(I) = .FALSE.
    ENDDO
    LB(IB) = .TRUE.

    NS = NR + 1
    DP = (P2 - P1)/NP
    P  = P1
    DO I = 1, NP
       PP = P
       P  = P + DP
       IF (P .GT. P2) P = P2
       CALL RTSRCH(XROOT(1),XROOT(2),NROOT,PP,P,TOL,DTOL,JMAX,IB, &
            TMAT,SV,UV,Q0,Q1,Q2,IFUNC,LB,PV,LRIG12,LERROR)
       IF (LERROR) RETURN
       DO J = 1, NROOT
          NR = NR + 1
          XR(NR)  = XROOT(J)
          IBR(NR) = IB
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE GTROOT


  SUBROUTINE FNDEND(A,IB,X1,X2,ETOL,JMAX, &
       TMAT,SV,UV,Q0,Q1,Q2,IFUNC,PV,LRIG12)
    !
    !       Finds the end of the interval on which the function is valued.
    !
    !       Aaron R. Dinner
    !
    INTEGER IB, JMAX, IFUNC
    real(chm_real) X1, X2, A, ETOL
    real(chm_real)TMAT(3,3,0:5),Q1(3),Q2(3),Q0(3),SV(3),UV(3), &
         PV(0:5,3)
    LOGICAL LF(4), LRIG12

    INTEGER J
    real(chm_real) DX, F5(4), XMID
    real(chm_real) COSP(4,5), SINP(4,5)
    LOGICAL LB(4)

    DO J = 1, 4
       LB(J) = .FALSE.
    ENDDO
    LB(IB) = .TRUE.

    CALL CRFUNC(F5,LF,COSP,SINP,IFUNC,X1,TMAT,SV,UV,Q0,Q1,Q2,LB, &
         PV,LRIG12)

    IF (LF(IB)) THEN
       A = X1
       DX = X2 - X1
    ELSE
       A = X2
       DX = X1 - X2
    ENDIF

    DO J = 1, JMAX
       DX = DX*0.5
       XMID = A + DX
       CALL CRFUNC(F5,LF,COSP,SINP,IFUNC,XMID, &
            TMAT,SV,UV,Q0,Q1,Q2,LB,PV,LRIG12)
       IF (LF(IB)) A = XMID
       IF (ABS(DX) .LT. ETOL) RETURN
    ENDDO

    CALL WRNDIE (2, '<FNDEND>', &
         'EXCEEDED MAXIMUM NUMBER OF ITERATIONS')

    RETURN
  END SUBROUTINE FNDEND

  SUBROUTINE RTSRCH(XROOT1,XROOT2,NROOT,X1,X2,TOL,DTOL,JMAX, &
       IB,TMAT,SV,UV,Q0,Q1,Q2,IFUNC,LB,PV,LRIG12, &
       LERROR)
    !
    !       Essentially the secant method with ZBRENT root polishing
    !
    !       Aaron R. Dinner
    !
    INTEGER IB, IFUNC, JMAX, NROOT
    real(chm_real)  XROOT1, XROOT2, X1, X2, TOL, DTOL
    real(chm_real)TMAT(3,3,0:5),Q1(3),Q2(3),Q0(3),SV(3),UV(3), &
         PV(0:5,3)
    LOGICAL LERROR, LB(4), LRIG12
    !
    INTEGER J
    real(chm_real) F5(4), FL, F, XL, SWAP, DX
    real(chm_real) COSP(4,5), SINP(4,5)
    LOGICAL L1(4), L2(4)

    NROOT = 0

    CALL CRFUNC(F5,L1,COSP,SINP,IFUNC,X1,TMAT,SV,UV,Q0,Q1,Q2,LB, &
         PV,LRIG12)
    FL = F5(IB)
    CALL CRFUNC(F5,L2,COSP,SINP,IFUNC,X2,TMAT,SV,UV,Q0,Q1,Q2,LB, &
         PV,LRIG12)
    F  = F5(IB)

    IF (.NOT. L1(IB) .OR. .NOT. L2(IB)) THEN
       !         We have two (or more) subranges caused by too coarse an initial scan
       LERROR = .TRUE.
       RETURN
    ENDIF

    IF (F.NE.SIGN(F,FL)) THEN
       XROOT1 = ZBRENT(X1,X2,TOL,DTOL,IB,LB,TMAT,SV,UV,Q0,Q1,Q2, &
            IFUNC,PV,LRIG12)
       NROOT = 1
       RETURN
    ENDIF

    !       Search down the side of the interval with the secant algorithm
    XROOT1 = X1 + TOL
    CALL CRFUNC(F5,L2,COSP,SINP,IFUNC,XROOT1,TMAT,SV,UV,Q0,Q1,Q2,LB, &
         PV,LRIG12)
    F  = F5(IB)

    IF (ABS(F) .LT. ABS(FL)) THEN
       XL = X1
    ELSE
       RETURN
    ENDIF

    DO J = 1, JMAX

       IF (F .EQ. FL) THEN
          !           The secant is parallel to the axis!
          IF (ABS(XL-XROOT1).LT.TOL) THEN
             !             Double root or near miss
             IF (ABS(F).LT.DTOL) THEN
                XROOT2 = XL
                NROOT  = 2
             ENDIF
             RETURN
             !           ELSE
             !             In principle one could scan this interval for roots, but
             !             the chances are pretty slim.
          ENDIF
          !           Secant method breakdown
          RETURN
       ENDIF

       DX = (XL - XROOT1)*F/(F - FL)
       XL = XROOT1
       FL = F
       XROOT1 = XROOT1 + DX
       !         If we are mapped out of the allowed interval, forget it.
       IF (XROOT1 .LT. X1 .OR. XROOT1 .GT. X2) RETURN
       CALL CRFUNC(F5,L1,COSP,SINP,IFUNC,XROOT1, &
            TMAT,SV,UV,Q0,Q1,Q2,LB,PV,LRIG12)

       IF (.NOT. L1(IB)) THEN
          LERROR = .TRUE.
          RETURN
       ENDIF

       F = F5(IB)
       IF (F.NE.SIGN(F,FL)) THEN
          XROOT2 = ZBRENT(X2,XROOT1,TOL,DTOL,IB,LB, &
               TMAT,SV,UV,Q0,Q1,Q2,IFUNC,PV,LRIG12)
          XROOT1 = ZBRENT(X1,XROOT1,TOL,DTOL,IB,LB, &
               TMAT,SV,UV,Q0,Q1,Q2,IFUNC,PV,LRIG12)
          NROOT = 2
          RETURN
       ELSEIF (ABS(DX) .LT.  TOL) THEN
          IF (ABS(FL) .LT. DTOL .AND. ABS(F) .LT. DTOL) THEN
             XROOT2 = XL
             NROOT = 2
          ENDIF
          RETURN
       ENDIF

    ENDDO

    RETURN
  END SUBROUTINE RTSRCH

  FUNCTION ZBRENT(X1,X2,TOL,DTOL,IB,LB,TMAT,SV,UV, &
       Q0,Q1,Q2, &
       IFUNC,PV,LRIG12) result(zbrent_1)
    !
    !       Essentially the Numerical Recipes ZBRENT routine
    !       See Press, 1986.
    !
    !       Aaron R. Dinner
    !
    use stream
    real(chm_real) :: zbrent_1
    INTEGER IB, IFUNC
    real(chm_real) X1, X2, TOL, DTOL
    real(chm_real) TMAT(3,3,0:5)
    real(chm_real) Q1(3), Q2(3), Q0(3)
    real(chm_real) SV(3), UV(3), PV(0:5,3)
    LOGICAL LB(4), LRIG12
    !
    INTEGER ITMAX, ITER, j
    real(chm_real) F5(4)
    real(chm_real) A, B, C, D, E, P, Q, R, S
    real(chm_real) FA, FB, FC, XM
    real(chm_real) EPS, TOL1
    real(chm_real) COSP(4,5), SINP(4,5)
    LOGICAL LF(4)
    PARAMETER (ITMAX = 100, EPS = 3.8E-8)

    A = X1
    B = X2
    CALL CRFUNC(F5,LF,COSP,SINP,IFUNC,A,TMAT,SV,UV,Q0,Q1,Q2,LB, &
         PV,LRIG12)
    FA = F5(IB)
    CALL CRFUNC(F5,LF,COSP,SINP,IFUNC,B,TMAT,SV,UV,Q0,Q1,Q2,LB, &
         PV,LRIG12)
    FB = F5(IB)

    IF (FB*FA .GT. 0.0) THEN
       !         Check for rounding error problems
       IF (ABS(FA).LE.DTOL) THEN
          ZBRENT_1 = A
       ELSE IF (ABS(FB).LE.DTOL) THEN
          ZBRENT_1 = B
       ELSE
          WRITE (OUTU,'(4f22.16)') A, B, FA, FB
          CALL WRNDIE (-5, '<ZBRENT>', &
               'INTERNAL ERROR:   ROOT NOT BRACKETED')
       ENDIF
    ENDIF

    FC = FB

    DO ITER = 1, ITMAX

       IF (FB*FC .GT. 0.0) THEN
          C = A
          FC = FA
          D = B - A
          E = D
       ENDIF

       IF (ABS(FC) .LT. ABS(FB)) THEN
          A = B
          B = C
          C = A
          FA = FB
          FB = FC
          FC = FA
       ENDIF

       TOL1 = 2.0*EPS*ABS(B) + 0.5*TOL
       XM = 0.5*(C - B)

       IF (ABS(XM) .LE. TOL1 .OR. FB .EQ. 0.0) THEN
          ZBRENT_1 = B
          RETURN
       ENDIF

       IF (ABS(E) .GE. TOL1 .AND. ABS(FA) .GT. ABS(FB)) THEN
          S = FB/FA

          IF (A .EQ. C) THEN
             P = 2.0*XM*S
             Q = 1.0 - S
          ELSE
             Q = FA/FC
             R = FB/FC
             P = S*(2.0*XM*Q*(Q-R)-(B-A)*(R-1.0))
             Q = (Q - 1.0)*(R - 1.0)*(S - 1.0)
          ENDIF

          IF (P .GT. 0.0) Q = -Q
          P = ABS(P)

          IF (2.0*P .LT. MIN(3.0*XM*Q-ABS(TOL1*Q),ABS(E*Q))) THEN
             E = D
             D = P/Q
          ELSE
             D = XM
             E = D
          ENDIF

       ELSE
          D = XM
          E = D
       ENDIF

       A = B
       FA = FB
       IF (ABS(D) .GT. TOL1) THEN
          B = B + D
       ELSE
          B = B + SIGN(TOL1,XM)
       ENDIF
       CALL CRFUNC(F5,LF,COSP,SINP,IFUNC,B,TMAT,SV,UV,Q0,Q1,Q2,LB, &
            PV,LRIG12)
       FB = F5(IB)
    ENDDO

    IF (FB*FA .GT. 0.0) CALL WRNDIE (-1, '<ZBRENT>', &
         'INTERNAL ERROR: EXCEEDED MAXIMUM NUMBER OF ITERATIONS')

    ZBRENT_1 = B
    RETURN
  END FUNCTION ZBRENT

  SUBROUTINE CRFUNC(FX,LF,COSP,SINP,IFUNC,P1,TMAT, &
       SV,UV,Q0,Q1,Q2,LB,PV,LRIG12)
    !
    !       A calling routine to pick the right concerted rotation function.
    !       Right now, only the seven dihedral rotation is implemented.  Ones
    !       for the tails that change only four or five are possible.
    !
    !       Aaron R. Dinner
    !
    INTEGER IFUNC
    real(chm_real) P1
    real(chm_real) TMAT(3,3,0:5)
    real(chm_real) Q0(3), Q1(3), Q2(3), PV(0:5,3)
    real(chm_real) SV(3), UV(3), FX(4)
    real(chm_real) COSP(4,5), SINP(4,5)
    LOGICAL LF(4), LB(4), LRIG12

    IF (IFUNC .EQ. 5) THEN
       CALL CRF5(FX,LF,COSP,SINP,P1,TMAT,SV,UV,Q0,Q1,Q2,LB,PV,LRIG12)
    ELSE
       CALL WRNDIE (-5, '<CRFUNC>', 'UNKNOWN FUNCTION')
       STOP
    ENDIF

    RETURN
  END SUBROUTINE CRFUNC


  SUBROUTINE CRF5(F5,LF,COSP,SINP,P1,TMAT, &
       SV,UV,Q0,Q1,Q2,LB,PV,LRIG12)
    !
    !       Returns up to 4 values of F5 given a value of phi1
    !
    !       Note that all degrees are handled in radians
    !
    !       Aaron R. Dinner
    !
    use consta
    real(chm_real) F5(4)
    real(chm_real) COSP(4,5), SINP(4,5), P1
    real(chm_real) TMAT(3,3,0:5), Q0(3), Q1(3), Q2(3), PV(0:5,3)
    real(chm_real) SV(3), UV(3)
    LOGICAL LF(4), LB(4), LRIG12

    !       Local
    INTEGER I, IB
    real(chm_real) RMAT1(3,3), RMAT2(3,3), RMAT3(3,3), RMAT4(3,3)
    real(chm_real) Q3(3), RV(3), TV(3)
    real(chm_real) R2, Q12, Q22
    real(chm_real) A, B, W, DTS, A2B2
    real(chm_real) V1(3), V2(3)

    DO I = 1, 4
       LF(I) = .FALSE.
    ENDDO

    COSP(1,1) = COS(P1)
    SINP(1,1) = SIN(P1)

    !       Get T1, T0, R1 to get r (GS21)
    CALL GTRMAT(RMAT1,COSP(1,1),SINP(1,1))

    !       If bonds 1 and 2 are not connected Q0 depends on P1
    IF (LRIG12) THEN
       CALL APMATX(V1(1),V1(2),V1(3),RMAT1, &
            PV(1,1),PV(1,2),PV(1,3))
       CALL APMATX(Q0(1),Q0(2),Q0(3),TMAT(1,1,0), &
            V1(1),V1(2),V1(3))
       DO I = 1, 3
          Q0(I) = Q0(I) + PV(0,I)
       ENDDO
    ENDIF

    !       Invert T1, T0, R1 (same as transposition)
    CALL TRNPOS(TMAT(1,1,0))
    CALL TRNPOS(TMAT(1,1,1))
    CALL TRNPOS(RMAT1)

    !       Apply the inverted matrices to s-q0 to get r (GS21)
    DO I = 1, 3
       V1(I) = SV(I) - Q0(I)
    ENDDO
    CALL APMATX(V2(1),V2(2),V2(3),TMAT(1,1,0),V1(1),V1(2),V1(3))
    CALL APMATX(V1(1),V1(2),V1(3),RMAT1,V2(1),V2(2),V2(3))
    CALL APMATX(RV(1),RV(2),RV(3),TMAT(1,1,1),V1(1),V1(2),V1(3))

    !       Return T1, T0, R1 to their original forms
    CALL TRNPOS(TMAT(1,1,0))
    CALL TRNPOS(TMAT(1,1,1))
    CALL TRNPOS(RMAT1)

    !       Calculate some necessary lengths
    R2  = DOT(RV(1),RV(2),RV(3),RV(1),RV(2),RV(3))
    Q12 = DOT(Q1(1),Q1(2),Q1(3),Q1(1),Q1(2),Q1(3))
    Q22 = DOT(Q2(1),Q2(2),Q2(3),Q2(1),Q2(2),Q2(3))

    W = 0.5*(R2 + Q12 - Q22 - 2.0*Q1(1)*RV(1))
    A = RV(2)*Q1(2) + RV(3)*Q1(3)
    B = RV(3)*Q1(2) - RV(2)*Q1(3)
    A2B2 = A*A + B*B
    DTS = (A2B2 - W*W)
    !       If DTS less than 0, then the function does not exist
    IF (DTS .LT. 0) RETURN
    DTS = SQRT(DTS)

    COSP(1,2) = (A*W + B*DTS)/A2B2
    SINP(1,2) = (B*W - A*DTS)/A2B2
    COSP(3,2) = (A*W - B*DTS)/A2B2
    SINP(3,2) = (B*W + A*DTS)/A2B2

    !       Each phi2 can have two phi4 (with corresponding phi3)
    COSP(2,2) = COSP(1,2)
    SINP(2,2) = SINP(1,2)
    COSP(4,2) = COSP(3,2)
    SINP(4,2) = SINP(3,2)

    !       x component of q3 is independent of phi2
    Q3(1) = RV(1) - Q1(1)

    !       For each branch
    DO IB = 1, 4
       IF (LB(IB)) THEN

          !           y and z components of q3 depend on phi2
          Q3(2) = RV(2)*COSP(IB,2) + RV(3)*SINP(IB,2) - Q1(2)
          Q3(3) = RV(3)*COSP(IB,2) - RV(2)*SINP(IB,2) - Q1(3)

          CALL TRNPOS(TMAT(1,1,2))
          CALL APMATX(TV(1),TV(2),TV(3),TMAT(1,1,2),Q3(1),Q3(2),Q3(3))
          CALL TRNPOS(TMAT(1,1,2))

          W = TV(1) - TMAT(1,1,3)*Q2(1)
          A = TMAT(1,2,3)*Q2(2) + TMAT(1,3,3)*Q2(3)
          B = TMAT(1,3,3)*Q2(2) - TMAT(1,2,3)*Q2(3)
          A2B2 = A*A + B*B
          DTS = (A2B2 - W*W)
          IF (DTS .GE. 0) THEN
             DTS = SQRT(DTS)

             !             This plus-minus sign indpendent of that for phi2
             IF (MOD(IB,2) .EQ. 1) THEN
                COSP(IB,4) = (A*W + B*DTS)/A2B2
                SINP(IB,4) = (B*W - A*DTS)/A2B2
             ELSE
                COSP(IB,4) = (A*W - B*DTS)/A2B2
                SINP(IB,4) = (B*W + A*DTS)/A2B2
             ENDIF

             CALL GTRMAT(RMAT4,COSP(IB,4),SINP(IB,4))
             CALL APMATX(V1(1),V1(2),V1(3),RMAT4,Q2(1),Q2(2),Q2(3))
             CALL APMATX(V2(1),V2(2),V2(3),TMAT(1,1,3), &
                  V1(1),V1(2),V1(3))

             A2B2 = TV(2)*TV(2) + TV(3)*TV(3)
             COSP(IB,3) = (V2(2)*TV(2) + V2(3)*TV(3))/A2B2
             SINP(IB,3) = (V2(2)*TV(3) - V2(3)*TV(2))/A2B2

             CALL GTRMAT(RMAT2,COSP(IB,2),SINP(IB,2))
             CALL GTRMAT(RMAT3,COSP(IB,3),SINP(IB,3))

             V1(1) = 1.0
             V1(2) = 0.0
             V1(3) = 0.0
             CALL APMATX(V2(1),V2(2),V2(3),TMAT(1,1,4), &
                  V1(1),V1(2),V1(3))
             CALL APMATX(V1(1),V1(2),V1(3),RMAT4,V2(1),V2(2),V2(3))
             CALL APMATX(V2(1),V2(2),V2(3),TMAT(1,1,3), &
                  V1(1),V1(2),V1(3))
             CALL APMATX(V1(1),V1(2),V1(3),RMAT3,V2(1),V2(2),V2(3))
             CALL APMATX(V2(1),V2(2),V2(3),TMAT(1,1,2), &
                  V1(1),V1(2),V1(3))
             CALL APMATX(V1(1),V1(2),V1(3),RMAT2,V2(1),V2(2),V2(3))
             CALL APMATX(V2(1),V2(2),V2(3),TMAT(1,1,1), &
                  V1(1),V1(2),V1(3))
             CALL APMATX(V1(1),V1(2),V1(3),RMAT1,V2(1),V2(2),V2(3))
             CALL APMATX(V2(1),V2(2),V2(3),TMAT(1,1,0), &
                  V1(1),V1(2),V1(3))

             !             Get F5 from GS20
             LF(IB) = .TRUE.
             F5(IB) = DOT(UV(1),UV(2),UV(3),V2(1),V2(2),V2(3)) &
                  - TMAT(1,1,5)
          ENDIF
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE CRF5

  SUBROUTINE GTTLAB(T,I01,I02,I11,I12,X,Y,Z,BL0,BL1)
    !
    !       Get the transformation matrix that takes the frame
    !       of bond 1 into the laboratory frame.
    !
    !       Aaron R. Dinner
    !
    !       Passed
    INTEGER I11, I12, I01, I02
    real(chm_real) X(*), Y(*), Z(*)
    real(chm_real) T(3,3), BL0, BL1
    !       Local
    INTEGER IA1, IA2
    real(chm_real) DX, DY, DZ, ST, CT
    real(chm_real) U0X, U0Y, U0Z, RPL

    !       If statement is to make sure we have three, rather than two,
    !       points defining the 0 coordinate system xy-plane.
    !       Need to cross x0 with p0 in non-bonded case!
    IF (I02 .eq. I11) THEN
       IA1 = I11
       IA2 = I12
       RPL = BL1
       CALL GTCTST(CT,ST,I01,I02,I11,I12,X,Y,Z,BL0,BL1)
    ELSE
       IA1 = I01
       IA2 = I11
       RPL = RDIST(X(IA1),Y(IA1),Z(IA1),X(IA2),Y(IA2),Z(IA2))
       CALL GTCTST(CT,ST,I01,I11,I01,I02,X,Y,Z,BL0,RPL)
    ENDIF

    DX = X(I02) - X(I01)
    DY = Y(I02) - Y(I01)
    DZ = Z(I02) - Z(I01)

    !       Eqn A.3
    T(1,1) = DX/BL0
    T(2,1) = DY/BL0
    T(3,1) = DZ/BL0

    DX = X(IA2) - X(IA1)
    DY = Y(IA2) - Y(IA1)
    DZ = Z(IA2) - Z(IA1)
    U0X = DX/RPL
    U0Y = DY/RPL
    U0Z = DZ/RPL

    CALL VPROD(T(1,3),T(2,3),T(3,3), &
         T(1,1),T(2,1),T(3,1),U0X,U0Y,U0Z)
    !       Eqn A.5
    T(1,3) = T(1,3)/ST
    T(2,3) = T(2,3)/ST
    T(3,3) = T(3,3)/ST
    !       Eqn A.4
    CALL VPROD(T(1,2),T(2,2),T(3,2), &
         T(1,3),T(2,3),T(3,3), &
         T(1,1),T(2,1),T(3,1))

    RETURN
  END SUBROUTINE GTTLAB

  FUNCTION DOT(X1,Y1,Z1,X2,Y2,Z2) result(dot_result)
    !
    !       Dot product of two vectors.
    !
    !       Aaron R. Dinner
    !
    real(chm_real) :: dot_result
    real(chm_real) X1,Y1,Z1,X2,Y2,Z2
    DOT_RESULT = X1*X2 + Y1*Y2 + Z1*Z2
    RETURN
  END FUNCTION DOT

  SUBROUTINE VPROD(X0,Y0,Z0,X1,Y1,Z1,X2,Y2,Z2)
    !
    !       Cross product of two vectors.
    !
    !       Aaron R. Dinner
    !
    real(chm_real) X0,Y0,Z0
    real(chm_real) X1,Y1,Z1,X2,Y2,Z2
    X0 = Y1*Z2 - Y2*Z1
    Y0 = Z1*X2 - Z2*X1
    Z0 = X1*Y2 - X2*Y1
    RETURN
  END SUBROUTINE VPROD

  SUBROUTINE GTRMAT(R,CP,SP)
    !
    !       The rotation matrix that transforms frame of bond i+1
    !       to frame of bond i
    !
    !       Aaron R. Dinner
    !
    real(chm_real) R(3,3), CP, SP

    R(1,1) =  1.0D0
    R(1,2) =  0.0D0
    R(1,3) =  0.0D0
    R(2,1) =  0.0D0
    R(2,2) =  CP
    R(2,3) = -SP
    R(3,1) =  0.0D0
    R(3,2) =  SP
    R(3,3) =  CP

    RETURN
  END SUBROUTINE GTRMAT

  SUBROUTINE GTTMAT(T,CT,ST)
    !
    !       The rotation matrix that transforms frame of bond i+1
    !       to frame of bond i
    !
    !       Aaron R. Dinner
    !
    real(chm_real) T(3,3), CT, ST

    T(1,1) =  CT
    T(1,2) = -ST
    T(1,3) =  0.0D0
    T(2,1) =  ST
    T(2,2) =  CT
    T(2,3) =  0.0D0
    T(3,1) =  0.0D0
    T(3,2) =  0.0D0
    T(3,3) =  1.0D0

    RETURN
  END SUBROUTINE GTTMAT

  FUNCTION RDIST(X0,Y0,Z0,X1,Y1,Z1) result(rdist_result)
    !
    !       Distance between two points
    !
    !       Aaron R. Dinner
    !
    real(chm_real) :: rdist_result
    !       Passed
    real(chm_real),intent(in) :: X0, Y0, Z0, X1, Y1, Z1
    !       Local
    real(chm_real) DX, DY, DZ

    DX = X1 - X0
    DY = Y1 - Y0
    DZ = Z1 - Z0
    RDIST_result = SQRT(DX*DX + DY*DY + DZ*DZ)
    RETURN
  END FUNCTION RDIST

  SUBROUTINE GTCTST(COST,SINT,I11,I12,I21,I22,X,Y,Z,L1,L2)
    !
    !       Find the cos and sin of the supplement of the
    !       angle formed by atoms I1, I2, and I3.
    !
    !       Aaron R. Dinner
    !

    INTEGER I11, I12, I21, I22
    real(chm_real)  X(*), Y(*), Z(*)
    real(chm_real)  COST, SINT, L1, L2
    !       Local
    real(chm_real) DX1, DY1, DZ1
    real(chm_real) DX2, DY2, DZ2
    real(chm_real) DP

    DX1 = X(I12) - X(I11)
    DY1 = Y(I12) - Y(I11)
    DZ1 = Z(I12) - Z(I11)
    DX2 = X(I22) - X(I21)
    DY2 = Y(I22) - Y(I21)
    DZ2 = Z(I22) - Z(I21)
    DP  = DOT(DX1,DY1,DZ1,DX2,DY2,DZ2)
    COST  = DP/(L1*L2)
    SINT = SQRT(1.0D0 - COST*COST)
    RETURN
  END SUBROUTINE GTCTST

  SUBROUTINE TRNPOS(T)
    !
    !       Transpose a 3x3 matrix
    !
    !       Aaron R. Dinner
    !
    real(chm_real) T(3,3)
    !
    INTEGER I, J
    real(chm_real) R

    DO I = 1, 3
       DO J = I+1, 3
          R = T(J,I)
          T(J,I) = T(I,J)
          T(I,J) = R
       ENDDO
    ENDDO
    RETURN
  END SUBROUTINE TRNPOS

  SUBROUTINE APMATX(RX,RY,RZ,T,X1,Y1,Z1)
    !
    !       Apply T to vector xyz
    !
    !       Aaron R. Dinner
    !
    real(chm_real) X1, Y1, Z1
    real(chm_real) RX, RY, RZ, T(3,3)

    RX = T(1,1)*X1 + T(1,2)*Y1 + T(1,3)*Z1
    RY = T(2,1)*X1 + T(2,2)*Y1 + T(2,3)*Z1
    RZ = T(3,1)*X1 + T(3,2)*Y1 + T(3,3)*Z1

    RETURN
  END SUBROUTINE APMATX

  SUBROUTINE MLTMAT(R,T1,T2)
    !
    !       Multiply two matrices
    !
    !       Aaron R. Dinner
    !
    real(chm_real) R(3,3), T1(3,3), T2(3,3)
    !       Local
    INTEGER I, J

    DO I = 1, 3
       DO J = 1, 3
          R(I,J) = T1(I,1)*T2(1,J) + &
               T1(I,2)*T2(2,J) + &
               T1(I,3)*T2(3,J)
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE MLTMAT

  SUBROUTINE GTPV(P,IND,I11,I12,I21,X,Y,Z)
    !
    !       Find the difference in origins for local coordinate systems
    !
    !       Aaron R. Dinner
    !
    INTEGER I11, I12, I21, IND
    real(chm_real)  P(0:5,3), X(*), Y(*), Z(*)

    real(chm_real)  CT, ST, R1, R2

    R1 = RDIST(X(I11),Y(I11),Z(I11),X(I12),Y(I12),Z(I12))

    IF (I21 .NE. I12) THEN
       R2 = RDIST(X(I11),Y(I11),Z(I11),X(I21),Y(I21),Z(I21))
       CALL GTCTST(CT,ST,I11,I12,I11,I21,X,Y,Z,R1,R2)
       P(IND,1) = CT*R2
       P(IND,2) = ST*R2
    ELSE
       P(IND,1) = R1
       P(IND,2) = 0.0
    ENDIF

    P(IND,3) = 0.0

    RETURN
  END SUBROUTINE GTPV

  SUBROUTINE CRFIXD(TMAT,SV,UV,Q0,Q1,Q2, &
       IATM,X,Y,Z,PV,BL)
    !
    !       Get all the fixed quantities of the dihedral root search
    !
    !       Aaron R. Dinner
    !
    INTEGER IATM(0:6,2)
    real(chm_real) X(*), Y(*), Z(*)
    real(chm_real) TMAT(3,3,0:5)
    real(chm_real) SV(3), UV(3), Q0(3), Q1(3), Q2(3)

    INTEGER I, IP1
    real(chm_real) BL(0:6), T0LAB(3,3), PV(0:5,3)

    DO I = 0, 5
       CALL GTPV(PV,I,IATM(I,1),IATM(I,2),IATM(I+1,1), &
            X,Y,Z)
    ENDDO

    DO I = 0, 6
       BL(I) = RDIST(X(IATM(I,1)),Y(IATM(I,1)),Z(IATM(I,1)), &
            X(IATM(I,2)),Y(IATM(I,2)),Z(IATM(I,2)))
    ENDDO

    !       Angles
    DO I = 0, 5
       IP1 = I + 1
       CALL GTTANG(TMAT(1,1,I),IATM(I,1),IATM(I,2), &
            IATM(IP1,1),IATM(IP1,2),X,Y,Z,BL(I),BL(IP1))
    ENDDO

    !       Transformation matrix from frame of bond 1 to lab
    CALL GTTLAB(T0LAB,IATM(0,1),IATM(0,2),IATM(1,1), &
         IATM(1,2),X,Y,Z,BL(0),BL(1))
    CALL TRNPOS(T0LAB)

    !       s vector (use q1 as scratch space)
    Q1(1) = X(IATM(6,1)) - X(IATM(0,1))
    Q1(2) = Y(IATM(6,1)) - Y(IATM(0,1))
    Q1(3) = Z(IATM(6,1)) - Z(IATM(0,1))
    CALL APMATX(SV(1),SV(2),SV(3),T0LAB,Q1(1),Q1(2),Q1(3))

    !       u vector (use q1 as scratch space)
    Q1(1) = X(IATM(6,2)) - X(IATM(6,1))
    Q1(2) = Y(IATM(6,2)) - Y(IATM(6,1))
    Q1(3) = Z(IATM(6,2)) - Z(IATM(6,1))
    CALL APMATX(UV(1),UV(2),UV(3),T0LAB,Q1(1),Q1(2),Q1(3))
    DO I = 1, 3
       UV(I) = UV(I)/BL(6)
    ENDDO

    !       Get q0
    CALL APMATX(Q0(1),Q0(2),Q0(3),TMAT(1,1,0), &
         PV(1,1),PV(1,2),PV(1,3))
    DO I = 1, 3
       Q0(I) = Q0(I) + PV(0,I)
    ENDDO

    !       Get q1
    CALL APMATX(Q1(1),Q1(2),Q1(3),TMAT(1,1,2), &
         PV(3,1),PV(3,2),PV(3,3))
    DO I = 1, 3
       Q1(I) = Q1(I) + PV(2,I)
    ENDDO

    !       Get q2
    CALL APMATX(Q2(1),Q2(2),Q2(3),TMAT(1,1,4), &
         PV(5,1),PV(5,2),PV(5,3))
    DO I = 1, 3
       Q2(I) = Q2(I) + PV(4,I)
    ENDDO

    RETURN
  END SUBROUTINE CRFIXD

  SUBROUTINE GTTANG(TMAT,I11,I12,I21,I22,X,Y,Z,BL1,BL2)
    !
    !       Find the cos and sin of the angle between the x-axis of
    !       local coordinate system 1 and the projection of the x-axis
    !       of local coordinate system 2 on the xy-plane of system 1.
    !
    !       Aaron R. Dinner
    !
    INTEGER I11, I12, I21, I22
    real(chm_real)  X(*), Y(*), Z(*), TMAT(3,3)
    real(chm_real)  BL1, BL2
    !       Local
    real(chm_real) COST, SINT, DX, DY, DZ, RL, TLAB(3,3), TTMP(3,3)

    IF (I12 .eq. I21) THEN
       CALL GTCTST(COST,SINT,I11,I12,I21,I22,X,Y,Z,BL1,BL2)
       CALL GTTMAT(TMAT,COST,SINT)
    ELSE
       !         Transformation matrix from frame of bond 1 to lab
       CALL GTTLAB(TLAB,I11,I12,I21,I22,X,Y,Z,BL1,BL2)
       CALL TRNPOS(TLAB)

       DX = X(I22) - X(I21)
       DY = Y(I22) - Y(I21)
       DZ = Z(I22) - Z(I21)
       TTMP(1,1) = DX/BL2
       TTMP(2,1) = DY/BL2
       TTMP(3,1) = DZ/BL2

       DX = X(I21) - X(I12)
       DY = Y(I21) - Y(I12)
       DZ = Z(I21) - Z(I12)
       RL = SQRT(DOT(DX,DY,DZ,DX,DY,DZ))
       DX = DX/RL
       DY = DY/RL
       DZ = DZ/RL

       CALL VPROD(TTMP(1,3),TTMP(2,3),TTMP(3,3), &
            TTMP(1,1),TTMP(2,1),TTMP(3,1),DX,DY,DZ)
       RL = SQRT(DOT(TTMP(1,3),TTMP(2,3),TTMP(3,3), &
            TTMP(1,3),TTMP(2,3),TTMP(3,3)))
       TTMP(1,3) = TTMP(1,3)/RL
       TTMP(2,3) = TTMP(2,3)/RL
       TTMP(3,3) = TTMP(3,3)/RL
       CALL VPROD(TTMP(1,2),TTMP(2,2),TTMP(3,2), &
            TTMP(1,3),TTMP(2,3),TTMP(3,3), &
            TTMP(1,1),TTMP(2,1),TTMP(3,1))
       CALL MLTMAT(TMAT,TLAB,TTMP)
    ENDIF
    RETURN
  END SUBROUTINE GTTANG

  SUBROUTINE CRPHI5(CP,SP,UV,TMAT,RMAT)
    !
    !       Get the fifth dihedral given the other four of the concerted rotation
    !
    !       Aaron R. Dinner
    !
    real(chm_real) CP, SP
    real(chm_real) UV(3), TMAT(3,3,0:5), RMAT(3,3,5)
    !
    real(chm_real) V1(3), V2(3), D

    CALL TRNPOS(TMAT(1,1,0))
    CALL TRNPOS(TMAT(1,1,1))
    CALL TRNPOS(TMAT(1,1,2))
    CALL TRNPOS(TMAT(1,1,3))
    CALL TRNPOS(TMAT(1,1,4))
    CALL TRNPOS(RMAT(1,1,1))
    CALL TRNPOS(RMAT(1,1,2))
    CALL TRNPOS(RMAT(1,1,3))
    CALL TRNPOS(RMAT(1,1,4))

    CALL APMATX(V2(1),V2(2),V2(3),TMAT(1,1,0),UV(1),UV(2),UV(3))
    CALL APMATX(V1(1),V1(2),V1(3),RMAT(1,1,1),V2(1),V2(2),V2(3))
    CALL APMATX(V2(1),V2(2),V2(3),TMAT(1,1,1),V1(1),V1(2),V1(3))
    CALL APMATX(V1(1),V1(2),V1(3),RMAT(1,1,2),V2(1),V2(2),V2(3))
    CALL APMATX(V2(1),V2(2),V2(3),TMAT(1,1,2),V1(1),V1(2),V1(3))
    CALL APMATX(V1(1),V1(2),V1(3),RMAT(1,1,3),V2(1),V2(2),V2(3))
    CALL APMATX(V2(1),V2(2),V2(3),TMAT(1,1,3),V1(1),V1(2),V1(3))
    CALL APMATX(V1(1),V1(2),V1(3),RMAT(1,1,4),V2(1),V2(2),V2(3))
    CALL APMATX(V2(1),V2(2),V2(3),TMAT(1,1,4),V1(1),V1(2),V1(3))

    D = TMAT(2,1,5)*TMAT(2,1,5) + TMAT(3,1,5)*TMAT(3,1,5)
    CP = (TMAT(2,1,5)*V2(2) + TMAT(3,1,5)*V2(3))/D
    SP = (TMAT(2,1,5)*V2(3) - TMAT(3,1,5)*V2(2))/D

    CALL TRNPOS(TMAT(1,1,0))
    CALL TRNPOS(TMAT(1,1,1))
    CALL TRNPOS(TMAT(1,1,2))
    CALL TRNPOS(TMAT(1,1,3))
    CALL TRNPOS(TMAT(1,1,4))
    CALL TRNPOS(RMAT(1,1,1))
    CALL TRNPOS(RMAT(1,1,2))
    CALL TRNPOS(RMAT(1,1,3))
    CALL TRNPOS(RMAT(1,1,4))

    RETURN
  END SUBROUTINE CRPHI5

  FUNCTION CRJACO(P1,IB,SV,UV,TMAT,Q0,Q1,Q2,PV,BL,LRIG12) &
       result(crjaco_result)
    !
    !       Calculate the Jacobian associated with a concerted dihedral
    !       rotation.  The coordinate transformation is to UV, SV, and
    !       gamma_6 (a Eulerian angle) held constant.
    !
    !       See Dodd, Boone, and Theodorou (1993) Appendix B
    !
    !       Aaron R. Dinner
    !
    real(chm_real) :: crjaco_result
    INTEGER IB
    real(chm_real)  P1
    real(chm_real)  SV(3), UV(3), PV(0:5,3)
    real(chm_real)  BL(0:5), Q0(3), Q1(3), Q2(3), TMAT(3,3,0:5)
    LOGICAL LRIG12
    !
    INTEGER INDX(5), I, J, IM1
    real(chm_real) U(5,3), E1(3), RNEW(5,3), A(3)
    real(chm_real) RMAT(3,3,5), RM(3,3,5), SM(3,3)
    real(chm_real) F5(4), COSP(4,5), SINP(4,5)
    real(chm_real) D, ADDV(3), B(5,5)
    LOGICAL LF(4), LB(4)

    DO I = 1, 4
       LB(I) = .FALSE.
    ENDDO
    LB(IB) = .TRUE.

    E1(1) = 1.0
    E1(2) = 0.0
    E1(3) = 0.0

    CALL CRFUNC(F5,LF,COSP,SINP,5,P1,TMAT,SV,UV,Q0,Q1,Q2,LB, &
         PV,LRIG12)

    CALL GTRMAT(RMAT(1,1,1),COSP(1, 1),SINP(1, 1))
    CALL GTRMAT(RMAT(1,1,2),COSP(IB,2),SINP(IB,2))
    CALL GTRMAT(RMAT(1,1,3),COSP(IB,3),SINP(IB,3))
    CALL GTRMAT(RMAT(1,1,4),COSP(IB,4),SINP(IB,4))

    !       Get Phi5
    CALL CRPHI5(COSP(IB,5),SINP(IB,5),UV,TMAT,RMAT)
    CALL GTRMAT(RMAT(1,1,5),COSP(IB,5),SINP(IB,5))

    !       Calculate the unit vectors in the directions of bonds
    !       1 to 5 (UV is 6) in the local coordinate frame of bond 0.

    CALL MLTMAT(RM(1,1,1),TMAT(1,1,0),RMAT(1,1,1))
    CALL APMATX(U(1,1),U(1,2),U(1,3),RM(1,1,1),E1(1),E1(2),E1(3))

    DO I = 2, 5
       IM1 = I - 1
       CALL MLTMAT(SM,TMAT(1,1,IM1),RMAT(1,1,I))
       CALL MLTMAT(RM(1,1,I),RM(1,1,IM1),SM)
       CALL APMATX(U(I,1),U(I,2),U(I,3),RM(1,1,I),E1(1),E1(2),E1(3))
    ENDDO

    !       Using the results of above calculate the positions of
    !       the second atom in bonds 1 to 5 (SV is 6) in the local
    !       coordinate frame of bond 0.

    DO J = 1, 3
       ADDV(J) = PV(0,J)
       RNEW(1,J) = BL(1)*U(1,J) + ADDV(J)
    ENDDO

    DO I = 2, 5
       IM1 = I - 1
       CALL APMATX(A(1),A(2),A(3),RM(1,1,IM1), &
            PV(IM1,1),PV(IM1,2),PV(IM1,3))
       DO J = 1, 3
          ADDV(J) = ADDV(J) + A(J)
          RNEW(I,J) = BL(I)*U(I,J) + ADDV(J)
       ENDDO
    ENDDO

    !       Load up the 5x5 matrix that will be used for the determinant
    !       calculation.  Note that it differs from (DBT B.11) to allow
    !       for a separation between bonds 5 and 6.

    DO I = 1, 5
       DO J = 1, 3
          A(J) = SV(J) - RNEW(I,J)
       ENDDO
       CALL VPROD(B(1,I),B(2,I),B(3,I), &
            U(I,1),U(I,2),U(I,3),A(1),A(2),A(3))
       CALL VPROD(A(1),A(2),A(3), &
            U(I,1),U(I,2),U(I,3),UV(1),UV(2),UV(3))
       B(4,I) = A(1)
       B(5,I) = A(2)
    ENDDO

    !       Calculate the determinant of the matrix and return it in CRJACO.
    !       Straight from Numerical Recipes section 2.5 (1989)
    CALL MCDCMP(B,5,5,INDX,D)
    DO I = 1, 5
       D = D*B(I,I)
    ENDDO

    !       Check for linear dependence in UV_x and UV_y.
    !       Replace UV_y with UV_z if so.  Necessary to re-calculate
    !       entire B matrix because MCDCMP destroys it.
    IF (D .EQ. 0) THEN
       DO I = 1, 5
          DO J = 1, 3
             A(J) = SV(J) - RNEW(I,J)
          ENDDO
          CALL VPROD(B(1,I),B(2,I),B(3,I), &
               U(I,1),U(I,2),U(I,3),A(1),A(2),A(3))
          CALL VPROD(A(1),A(2),A(3), &
               U(I,1),U(I,2),U(I,3),UV(1),UV(2),UV(3))
          B(4,I) = A(1)
          B(5,I) = A(3)
       ENDDO
    ENDIF

    CRJACO_RESULT = ABS(1.0/D)

    RETURN
  END FUNCTION CRJACO

  FUNCTION DIHEF(I,J,K,L,X,Y,Z) result(dihef_result)
    !
    !       Calculates a dihedral angle.
    !       Based on the CHARMM IC routine.
    !
    !       Aaron R. Dinner
    !
    use number
    real(chm_real) :: dihef_result
    !
    INTEGER I,J,K,L
    real(chm_real) X(*),Y(*),Z(*)
    !
    real(chm_real) XI,YI,ZI,XJ,YJ,ZJ,XK,YK,ZK,XL,YL,ZL
    real(chm_real) FX,FY,FZ,HX,HY,HZ
    real(chm_real) GX,GY,GZ
    real(chm_real) CST
    real(chm_real) AX,AY,AZ,BX,BY,BZ,RA2,RB2,RA,RB,RAR,RBR, &
         AXR,AYR,AZR
    real(chm_real) BXR,BYR,BZR,CX,CY,CZ

    !
    XI=X(I)
    YI=Y(I)
    ZI=Z(I)

    XJ=X(J)
    YJ=Y(J)
    ZJ=Z(J)

    XK=X(K)
    YK=Y(K)
    ZK=Z(K)

    XL=X(L)
    YL=Y(L)
    ZL=Z(L)
    !
    FX=XI-XJ
    FY=YI-YJ
    FZ=ZI-ZJ
    !
    HX=XL-XK
    HY=YL-YK
    HZ=ZL-ZK
    !
    GX=XJ-XK
    GY=YJ-YK
    GZ=ZJ-ZK
    !
    AX=FY*GZ-FZ*GY
    AY=FZ*GX-FX*GZ
    AZ=FX*GY-FY*GX
    BX=HY*GZ-HZ*GY
    BY=HZ*GX-HX*GZ
    BZ=HX*GY-HY*GX
    !
    RA2=AX*AX+AY*AY+AZ*AZ
    RB2=BX*BX+BY*BY+BZ*BZ
    RA=SQRT(RA2)
    RB=SQRT(RB2)
    RAR=1.0/RA
    RBR=1.0/RB

    AXR=AX*RAR
    AYR=AY*RAR
    AZR=AZ*RAR
    BXR=BX*RBR
    BYR=BY*RBR
    BZR=BZ*RBR

    CST=AXR*BXR+AYR*BYR+AZR*BZR
    IF(ABS(CST).GE.1.0) CST=SIGN(ONE,CST)
    DIHEF_RESULT=ACOS(CST)
    CX=AYR*BZR-AZR*BYR
    CY=AZR*BXR-AXR*BZR
    CZ=AXR*BYR-AYR*BXR
    IF(GX*CX+GY*CY+GZ*CZ.GT.0.0) DIHEF_RESULT=-DIHEF_RESULT
    !
    !       This definition was for Dodd Boone and Theodorou / Flory
    !       DIHEF_RESULT = - PI + DIHEF_RESULT
    !
    RETURN
  END FUNCTION DIHEF

#endif 
end module mcmvcrot

