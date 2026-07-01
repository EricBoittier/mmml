module mcmvrtrn
#if KEY_MC==1
  use chm_types
  use dimens_fcm
  implicit none

contains

      SUBROUTINE MVRTRN(COMLYN,COMLEN,LCART,MVTYPE,NMVATM,IPIVTP,IMVNGP, &
                        IBLSTP,MDXP,MBONDT,QBND,ARMLIM,ARMMAX,ANISO, &
                        RMDX,X,Y,Z,WMAIN)
!
!       Determines move specific information for rigid body moves.
!       These include translations, rotations, volume scaling for
!       constant pressure simulations, and particle insertion/deletion
!       for grand canonical simulations.
!
!       MVTYPE          Description of move
!       ------          -----------------------------------
!            1          Translations
!            2          Rotations
!            7          Volume scaling move
!            8          Grand canonical move.
!
!       MODE            Description of rigid body
!       ------          -----------------------------------
!            1          By residue
!            2          By the entire selection
!            3          By heavy atom
!            4          By atom
!
!       Aaron R. Dinner
!
  use psf
  use selctam
  use select
  use string
  use memory
  use corsubs,only:fndu
  use chutil,only:hydrog
  use mcmvutil
!
!       Passed Variables
!
        CHARACTER(len=*) :: COMLYN
        INTEGER COMLEN
        type(iptr_ptr) :: IMVNGP, IBLSTP, IPIVTP
        INTEGER MVTYPE, NMVATM, MBONDT
        type(chm_ptr) :: MDXP
        real(chm_real)  RMDX, ARMMAX, X(*), Y(*), Z(*), WMAIN(*)
        LOGICAL ARMLIM, ANISO, QBND(MBONDT), LCART
!
!       Local Variables
!
        integer,allocatable,dimension(:) :: IFP, IHP
        INTEGER I, J, IADD, IA1, IA2
        type(chm_iptr) :: TEMPP, LISTP
        INTEGER  NORIGB, NORIGT
        INTEGER NSLCT, SLMODE, MODE, NMV, NR, NT, IRP
        LOGICAL COFMAS, LVOL, FEWER
        CHARACTER(len=4) :: WRD

!       Flag to determine if it is a volume move
        LVOL = MVTYPE .EQ. 7

!       Determine the type of rigid body
        IF (LCART) THEN
          MODE = 4
        ELSE IF (MVTYPE .EQ. 8) THEN
          MODE = 1
        ELSE
          WRD = NEXTA4(COMLYN,COMLEN)
          IF (WRD .EQ. 'BYRE') THEN
            MODE = 1
          ELSE IF (WRD .EQ. 'BYAL') THEN
            MODE = 2
          ELSE IF (WRD .EQ. 'BYHE') THEN
            MODE = 3
          ELSE IF (WRD .EQ. 'BYAT') THEN
            MODE = 4
            IF (MVTYPE .EQ. 2) CALL WRNDIE(-2,'<MVRTRN>', &
              'ROTATIONS OF INDIVIDUAL ATOMS')
          ELSE
            CALL WRNDIE(-3,'<MVRTRN>', &
              'UNKNOWN RIGID MOVE SELECTION SPECIFIER')
          ENDIF
        ENDIF

!       Fill in logicals for which bonded energy terms must be calculated.
        IF (MODE .EQ. 3 .OR. MODE .EQ. 4 .AND. .NOT. LVOL) THEN
          CALL FILLQB(QBND,.TRUE.,.TRUE.,.TRUE.,.TRUE.,(MODE.EQ.4))
        ELSE
          CALL FILLQB(QBND,.FALSE.,.FALSE.,.FALSE.,.FALSE.,.FALSE.)
          IBLSTP%A => null()
        ENDIF
!       If necessary build a connectivity table
        IF (MODE .EQ. 3 .OR. MODE .EQ. 4) THEN
          CALL MKBNDT()
        ENDIF

!       Get the SELE...END statement that defines the moveable atoms.
        call chmalloc('mvrtrn.src','MVRTRN','IFP',NATOM,intg=IFP)
        IFP(1:NATOM) = 0
        CALL SELCTA(COMLYN,COMLEN,IFP,X,Y,Z,WMAIN,.TRUE.)

!       ARD 05-06-28
!       Set default differently than in MVDIHE for backward compatibility
        FEWER = (GTRMI(COMLYN,COMLEN,'FEWE',1) .GT. 0)

!       A scratch array for BYHEavy
        IF (MODE .EQ. 3) THEN
          call chmalloc('mvrtrn.src','MVRTRN','IHP',NATOM,intg=IHP)
          IHP(1:NATOM) = 0
        ENDIF

!       Count the number of elements that will be in the list.
!       If there are no move locations, no point in the move, so return.
        NMVATM = 0
        DO I = 1, NATOM
          IF (IFP(I) == 1) THEN
            IF (MODE .EQ. 3) THEN
              IF (HYDROG(I)) THEN
                IFP(I) = 0
              ELSE IF (MVTYPE .EQ. 2) THEN
                TEMPP%A => IABNDP(I)%A
                CALL GTHEFL(NMV, I, TEMPP%A, IHP, 0)
                IF (NMV .GT. 1 .OR. .NOT. FEWER) NMVATM = NMVATM + 1
              ELSE
                NMVATM = NMVATM + 1
              ENDIF
            ELSE IF (MODE .EQ. 1 .AND. LVOL) THEN
!             Make sure we only scale a residue once in volume moves
              CALL GTRSFL(TEMPP,I,IA1,IA2,0)
              IFP(IA1) = 1
              DO J = IA1+1, IA2
                IFP(J) = 0
              ENDDO
              NMVATM = NMVATM + 1
            ELSE
              NMVATM = NMVATM + 1
            ENDIF
          ENDIF
        ENDDO
        IF (NMVATM .EQ. 0) GOTO 100

!       If it is BYALL or a volume scaling move, the moving atom list is
!       the same for all move instances.  Handle these cases here.
        COFMAS = .FALSE.
        IF (MODE .EQ. 2 .OR. LVOL) THEN
          IF (MVTYPE .EQ. 1) THEN
            CALL IMVLST(LISTP, NATOM, IFP)
            NMVATM = 1
          ELSE IF (MVTYPE .EQ. 2) THEN
            CALL IMVLST(LISTP, NATOM, IFP)
!           Call SELRPN directly to get alternative default behavior:
!           none rather than all selected when no SELE...END statement.
            SLMODE = 0
            IFP(1:NATOM) = 0
            CALL SELRPN(COMLYN,COMLEN,IFP,NATOM,0,SLMODE,.FALSE., &
                        1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
                        .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)

!           If there are no selected atoms in the second set, make the
!           rotations around the center of mass.
            NMVATM = 0
            DO I = 1, NATOM
              IF (IFP(I) == 1) NMVATM = NMVATM + 1
            ENDDO
            COFMAS = NMVATM .EQ. 0
            IF (COFMAS) NMVATM = 1

          ELSE IF (LVOL) THEN
!           Volume move
            IF (MODE .EQ. 2 .OR. MODE .EQ. 4) THEN
!             Byatom --- save in a single list
              CALL IMVLST(LISTP, NATOM, IFP)
            ELSE
!             Determine and store the rigid groups
              NMVATM = 0
              DO I = 1, NATOM
                IF (IFP(I) == 1) THEN
                  NMVATM = NMVATM + 1
                  IF (MODE .EQ. 3) THEN
                    TEMPP%A => IABNDP(I)%A
                    CALL GTHEFL(NMV, I, TEMPP%A, IHP, 1)
                    CALL IMVLST(TEMPP, NATOM, IHP)
                  ELSE
                    CALL GTRSFL(TEMPP, I, IA1, IA2, 1)
                  ENDIF
                  CALL IMVADD(LISTP, NMVATM, TEMPP)
                  IF (MODE .EQ. 3) THEN
!                   Clear scratch array for next time
                    TEMPP%A => IABNDP(I)%A
                    CALL GTHEFL(NMV, I, TEMPP%A, IHP, 0)
                  ENDIF
                ENDIF
              ENDDO
            ENDIF
            NMVATM = 1
          ENDIF
        ENDIF

!       Allocate the space
        allocate(IMVNGP%A(NMVATM))
        IF (MVTYPE == 2) allocate(IPIVTP%A(NMVATM))
        IF ((MODE .EQ. 3 .OR. MODE .EQ. 4) .AND. .NOT. LVOL) &
            allocate(IBLSTP%A(NMVATM))
!       If the move is anisotropic, allocate space for a 3x3 matrix.
        IF (ANISO) THEN
          IF (MVTYPE.NE.1) CALL WRNDIE(-3,'<MVRTRN>', &
            'ANISotropic only meaningful for translations')
          call chmalloc('mvrtrn.src','MVRTRN','MDXP',9*NMVATM,crlp=MDXP%A)
        ELSE
          call chmalloc('mvrtrn.src','MVRTRN','MDXP',NMVATM,crlp=MDXP%A)
        ENDIF

        NMVATM = 0
        DO I = 1, NATOM
          IF (IFP(I) == 1 .OR. COFMAS .OR. LVOL) THEN
            NMVATM = NMVATM + 1

!           By residues (no bonded terms checked)
            IF (MODE .EQ. 1 .AND. .NOT. LVOL) THEN
              CALL GTRSFL(TEMPP, I, IA1, IA2, 1)
              IMVNGP%A(NMVATM)%A => TEMPP%A

!           By the entire selection (no bonded terms checked)
            ELSE IF (MODE .EQ. 2 .OR. LVOL) THEN
!             Note that all the instances can point to the same list
              IMVNGP%A(NMVATM)%A => LISTP%A

!           By heavy atom
            ELSE IF (MODE .EQ. 3) THEN
              TEMPP%A => IABNDP(I)%A
!             Count the number of moving atoms
              CALL GTHEFL(NMV, I, TEMPP%A, IHP, 0)
              IF (FEWER .AND. MVTYPE .EQ. 2 .AND. NMV .EQ. 1) THEN
                NMVATM = NMVATM - 1
                GOTO 120
              ENDIF
!             Now that it is OK, fill the list
              CALL GTHEFL(NMV, I, TEMPP%A, IHP, 1)
              CALL IMVLST(LISTP, NATOM, IHP)
              IMVNGP%A(NMVATM)%A => LISTP%A
!             Get the bond list
!             Create an IABND list without the H terms and swap it in
              CALL IREMOH(NORIGB, I, TEMPP%A, IHP, 0)
              TEMPP%A => IATHTP(I)%A
              CALL IREMOH(NORIGT, I, TEMPP%A, IHP, 1)
              CALL GNBNDL(I, NMVATM, IBLSTP%A, QBND)
!             Restore theta list
              TEMPP%A(1) = NORIGT
              TEMPP%A => IABNDP(I)%A
!             Restore bond list
              TEMPP%A(1) = NORIGB
!             Empty ihp stack of moving atoms
              CALL GTHEFL(NMV, I, TEMPP%A, IHP, 0)

!           By atom
            ELSE IF (MODE .EQ. 4) THEN
              call chmalloc('mvrtrn.src','MVRTRN','TEMPP',4,intgp=TEMPP%A)
              TEMPP%A(1) = 2
              TEMPP%A(2) = 4
              TEMPP%A(3) = I
              TEMPP%A(4) = I
              IMVNGP%A(NMVATM)%A => TEMPP%A
!             Bonded term information
              CALL GNBNDL(I, NMVATM, IBLSTP%A, QBND)

            ENDIF

            IF (ANISO) THEN
              CALL FLANIS(RMDX, MDXP%A, NMVATM)
            ELSE
              MDXP%A(NMVATM) = RMDX
            ENDIF

!           Movetype specific information
            IF (MVTYPE .EQ. 2) THEN
              call chmalloc('mvrtrn.src','MVRTRN','TEMPP',1,intgp=TEMPP%A)
              IF (COFMAS) THEN
                TEMPP%A(1) = -1
              ELSE
                TEMPP%A(1) = I
              ENDIF
              IPIVTP%A(NMVATM)%A => TEMPP%A
            ENDIF

!           For volume moves and whole body rigid translations and
!           center-of-mass rotations, only need one pass through loop.
            IF (((MODE.EQ.2).AND.(MVTYPE.EQ.1)).OR.COFMAS.OR.LVOL) &
              GOTO 100

          ENDIF
120       CONTINUE
        ENDDO

!       Movetype specific information
100     IF (MVTYPE .EQ. 2) THEN
          ARMLIM = .TRUE.
          ARMMAX = 180.0
        ELSE
          ARMLIM = .FALSE.
        ENDIF
!       Save the type of scaling (byatom or group) in IPIVTP
        IF (LVOL) THEN
          allocate(IPIVTP%A(1))
          call chmalloc('mvrtrn.src','MVRTRN','IPIVTP',1,intgp=IPIVTP%A(1)%A)
          IPIVTP%A(1)%A(1) = MODE
        ENDIF

!       Free the selection array
        IF (MODE .EQ. 3 .OR. MODE .EQ. 4) CALL FREBND()
        IF (MODE .EQ. 3) THEN
          call chmdealloc('mvrtrn.src','MVRTRN','IHP',NATOM,intg=IHP)
        ENDIF
        call chmdealloc('mvrtrn.src','MVRTRN','IFP',NATOM,intg=IFP)

        RETURN
      END SUBROUTINE MVRTRN

      SUBROUTINE GTRSFL(ILISTP, IATOM, IARESF, IARESL, MODE)
!
!       Does a binary search on the IBASE array to find the first
!       and last atom numbers of a residue containing IATOM
!
!       Aaron R. Dinner
!
  use memory
  use psf
!
!       Passed Variables
!
        type(chm_iptr) :: ILISTP
        INTEGER IATOM, IARESF, IARESL, MODE
!
!       Local Variables
!
        INTEGER IRESF, IRESL, MID

        IRESF = 1
        IRESL = NRES

1       IF (IRESF .LE. IRESL) THEN
          MID = (IRESL + IRESF)/2
          IF ((IBASE(MID) .LT. IATOM) .AND. &
              (IBASE(MID+1) .GE. IATOM)) THEN
            IARESF = IBASE(MID) + 1
            IARESL = IBASE(MID + 1)

            IF (MODE .EQ. 1) THEN
              call chmalloc('mvrtrn.src','GTRSFL','ILISTP',4,intgp=ILISTP%A)
              ILISTP%A(1) = 2
              ILISTP%A(2) = 4
              ILISTP%A(3) = IARESF
              ILISTP%A(4) = IARESL
            ENDIF

            RETURN
          ELSE IF (IBASE(MID+1) .LT. IATOM) THEN
            IRESF = MID + 1
          ELSE
            IRESL = MID - 1
          ENDIF
          GOTO 1
        ENDIF
!
!       Reaching this point indicates that the atom number is
!       outside the list -- something is drastically wrong.
!
        CALL WRNDIE(-5,'<GTRSFL>', &
          'RESIDUE BOUNDS COULD NOT BE FOUND')

        RETURN
      END SUBROUTINE GTRSFL

      SUBROUTINE GTHEFL(NMV, IATOM, IABND, IMVATM, MODE)
!
!       Finds the attached hydrogens to a heavy atom.
!
!       Aaron R. Dinner
!
  use psf
  use chutil,only:hydrog
  use mcmvutil, only: notatm
!
        INTEGER IATOM, IABND(:), IMVATM(:), MODE, NMV
!
        INTEGER I, J, K, N

        NMV = 1

!       Mark the heavy atom and attached hydrogens with 1s
        N = IABND(1) + 1
        DO I = 2, N
          J = IABND(I)
          K = NOTATM(IB(J),JB(J),IATOM)
          IF (HYDROG(K)) THEN
            IMVATM(K) = MODE
            NMV = NMV + 1
          ENDIF
        ENDDO
        IMVATM(IATOM) = MODE

        ! caller should do this explicitly
        ! IF (MODE .GT. 0) CALL IMVLST(ILISTP,NATOM,IMVATM)

        RETURN
      END SUBROUTINE GTHEFL

      SUBROUTINE IREMOH(NORIG, IATOM, IABND, IMVATM, MODE)
!
!       Swaps all the bonded terms with H to the end and
!       decreases the number so they are not seen.
!
!       The original number of terms is returned in NORIG.
!
!       MODE = 0 --> bonds
!       MODE = 1 --> angles
!
!       Aaron R. Dinner
!
  use psf

        INTEGER NORIG, IATOM, IABND(*), IMVATM(*), MODE
!
        INTEGER N, NL, I, J, K, ITMP
!
        INTEGER NOTATM

        NORIG = IABND(1)
        N = NORIG + 1
        NL = N
        I  = 2
10      J = IABND(I)
          IF (MODE .EQ. 0) THEN
            K = IMVATM(IB(J))*IMVATM(JB(J))
          ELSE IF (MODE .EQ. 1) THEN
            K = IMVATM(IT(J))*IMVATM(JT(J))*IMVATM(KT(J))
          ELSE
            CALL WRNDIE(-3,'<IREMOH>','Internal error')
          ENDIF
          IF (K .EQ. 1) THEN
            ITMP = IABND(NL)
            IABND(NL) = IABND(I)
            IABND(I) = ITMP
            NL = NL - 1
          ENDIF
          I = I + 1
        IF (I .LE. NL) GOTO 10
        IABND(1) = NL - 1

        RETURN
      END SUBROUTINE IREMOH

      SUBROUTINE GNBNDL(IATOM, IDX, IBLSTP, QBND)
!
!       Generates the bonded term lists.
!
!       Aaron R. Dinner
!
  use memory
#if KEY_PATHINT==1
  use mpathint, only: qpint  
#endif
  use mcmvutil
!       Passed Variables
!
        type(chm_iptr) :: IBLSTP(:)
        INTEGER IATOM, IDX
        LOGICAL QBND(:)
!
!       Local Variables
!
        INTEGER NB, NT, NI, NP, NE, NPI, i

        NB  = NBTF(IABNDP(IATOM)%A, QBND(1))
        NT  = NBTF(IATHTP(IATOM)%A, QBND(2))
        NP  = NBTF(IAPHIP(IATOM)%A, QBND(3))
        NI  = NBTF(IAIMPP(IATOM)%A, QBND(4))
        NPI = 0
#if KEY_PATHINT==1
        IF (QPINT) NPI = NBTF(IAPIMP(IATOM)%A, QBND(5))  
#endif

        NE = NB + NT + NP + NI + NPI
        IF (NE .EQ. 0) THEN
          IBLSTP(IDX)%A => null()
          RETURN
        ELSE
          call chmalloc('mvrtrn.src','GNBNDL','IBLSTP(IDX)',NE,intgp=IBLSTP(IDX)%A)
        ENDIF

        NE = 0

        IF (QBND(1)) THEN
          NE = NE + 1
          CALL ASSIBL(NE,NB, IABNDP(IATOM)%A, IBLSTP(IDX)%A)
        ENDIF
        IF (QBND(2)) THEN
          NE = NE + 1
          CALL ASSIBL(NE,NT, IATHTP(IATOM)%A, IBLSTP(IDX)%A)
        ENDIF
        IF (QBND(3)) THEN
          NE = NE + 1
          CALL ASSIBL(NE,NP, IAPHIP(IATOM)%A, IBLSTP(IDX)%A)
        ENDIF
        IF (QBND(4)) THEN
          NE = NE + 1
          CALL ASSIBL(NE,NI, IAIMPP(IATOM)%A, IBLSTP(IDX)%A)
        ENDIF
        IF (QBND(5)) THEN
          NE = NE + 1
          CALL ASSIBL(NE,NPI, IAPIMP(IATOM)%A, IBLSTP(IDX)%A)
        ENDIF

        RETURN
      END SUBROUTINE GNBNDL

      INTEGER FUNCTION NBTF(IBND,QBND)
!
!       Returns the number of storage places to keep track of
!       bonded terms in IBLSTP.
!
!       Aaron R. Dinner
!
      implicit none
!
        INTEGER IBND(:)
        LOGICAL QBND

        IF (QBND) THEN
          NBTF = IBND(1) + 1
        ELSE
          NBTF = 0
        ENDIF
        RETURN
      END FUNCTION NBTF

      SUBROUTINE ASSIBL(NE,NTERM,ITERM,IBLST)
!
!       Assigns a bonded term to a list
!
!       Aaron R. Dinner
!
      implicit none
        INTEGER NE, NTERM, ITERM(:), IBLST(*)
        INTEGER I, IS

        IS = NE
        DO I = 2, NTERM
          NE = NE + 1
          IBLST(NE) = ITERM(I)
        ENDDO
        IBLST(IS) = NE
        RETURN
      END SUBROUTINE ASSIBL

      SUBROUTINE RIGTRN(X, Y, Z, IAF, IAL, DX, DY, DZ)
!
!       Make a rigid translation to the atoms between IAF and IAL.
!
!       Aaron R. Dinner
!
!       Passed Variables
!
        INTEGER IAF, IAL
        real(chm_real) DX, DY, DZ, X(*), Y(*), Z(*)
!
!       Local Variables
!
        INTEGER I

        DO I = IAF, IAL
          X(I) = X(I) + DX
          Y(I) = Y(I) + DY
          Z(I) = Z(I) + DZ
        ENDDO
        RETURN
      END SUBROUTINE RIGTRN

      SUBROUTINE TRNALL(X, Y, Z, IMVNG, DX, DY, DZ)
!
!       Make a rigid translation to all the atoms in a
!       move instance.
!
!       Aaron R. Dinner
!
!       Passed Variables
!
        INTEGER IMVNG(:)
        real(chm_real) DX, DY, DZ, X(*), Y(*), Z(*)
!
!       Local Variables
!
        INTEGER I, J, K, L, NG, NN, IAF, IAL

        NG = IMVNG(1)
        J = NG + 2
        DO I = 2, NG
          NN = IMVNG(I)
          DO K = J, NN, 2
            IAF = IMVNG(K-1)
            IAL = IMVNG(K)
            CALL RIGTRN(X, Y, Z, IAF, IAL, DX, DY, DZ)
          ENDDO
          J = NN + 2
        ENDDO
        RETURN
      END SUBROUTINE TRNALL

      SUBROUTINE APPLYU(U,X,Y,Z,IATOM,XCEN,YCEN,ZCEN)
!
!       Apply the rotation matrix to the coordinates of atom IATOM around
!       (XCEN YCEN ZCEN).
!
!       Aaron R. Dinner
!
!       Passed Variables
!
        INTEGER IATOM
        real(chm_real) X(*), Y(*), Z(*)
        real(chm_real) U(9), XCEN, YCEN, ZCEN
!
!       Local Variables
!
        INTEGER I
        real(chm_real) XV, YV, ZV

        XV = X(IATOM) - XCEN
        YV = Y(IATOM) - YCEN
        ZV = Z(IATOM) - ZCEN
        X(IATOM) = XCEN + U(1)*XV+U(2)*YV+U(3)*ZV
        Y(IATOM) = YCEN + U(4)*XV+U(5)*YV+U(6)*ZV
        Z(IATOM) = ZCEN + U(7)*XV+U(8)*YV+U(9)*ZV

        RETURN
      END SUBROUTINE APPLYU

      SUBROUTINE MKRROT(DX,X,Y,Z,IMVNG,IPVT,RMDX,ISEED)
!
!       Make a rigid rotations to the atoms between IAF and IAL.
!
!       Aaron R. Dinner
!
  use clcg_mod,only:random
  use consta
  use number
  use psf
  use corsubs,only:fndu
  use mcmvutil, only: rosphr

        INTEGER IPVT, ISEED, IMVNG(:)
        real(chm_real)  DX, X(*), Y(*), Z(*), RMDX

        INTEGER I, J, K, L, IAF, IAL, NG, NN
        real(chm_real)  RN(3), U(9), XP, YP, ZP, MT
        LOGICAL LOK

        CALL ROSPHR(ISEED,RN)
        DX = (2.0*RANDOM(ISEED) - 1.0)*RMDX
        CALL FNDU(U,RN,DX,LOK)
        IF (.NOT. LOK) CALL WRNDIE(-2,'<MKRROT>','INTERNAL ERROR')

        IF (IPVT .GT. 0) THEN
          XP = X(IPVT)
          YP = Y(IPVT)
          ZP = Z(IPVT)
        ELSE
          CALL MVGCOM(XP,YP,ZP,IMVNG,X,Y,Z)
        ENDIF

        NG = IMVNG(1)
        J = NG + 2
        DO I = 2, NG
          NN = IMVNG(I)
          DO K = J, NN, 2
            IAF = IMVNG(K-1)
            IAL = IMVNG(K)
            DO L = IAF, IAL
              CALL APPLYU(U,X,Y,Z,L,XP,YP,ZP)
            ENDDO
          ENDDO
          J = NN + 2
        ENDDO

        RETURN
      END SUBROUTINE MKRROT

      SUBROUTINE MVGCOM(XP,YP,ZP,IMVNG,X,Y,Z)
!
!       Gets center of mass for moving atoms.
!       Moved from mvgcmc.src to take outside GCMC preprocessor directives.
!
!       Aaron R. Dinner
!
  use number
  use psf
          INTEGER IMVNG(*)
          real(chm_real)  XP, YP, ZP
          real(chm_real)  X(*), Y(*), Z(*)
!
          real(chm_real)  MT
          INTEGER I, J, K, L, NN, NG, IAF, IAL

          XP = ZERO
          YP = ZERO
          ZP = ZERO
          MT = ZERO
          NG = IMVNG(1)
          J = NG + 2
          DO I = 2, NG
            NN = IMVNG(I)
            DO K = J, NN, 2
              IAF = IMVNG(K - 1)
              IAL = IMVNG(K)
!             Get center of mass
              DO L=IAF, IAL
                XP = XP + AMASS(L)*X(L)
                YP = YP + AMASS(L)*Y(L)
                ZP = ZP + AMASS(L)*Z(L)
                MT = MT + AMASS(L)
              ENDDO
            ENDDO
            J = NN + 2
          ENDDO
          XP = XP / MT
          YP = YP / MT
          ZP = ZP / MT

        RETURN
      END SUBROUTINE MVGCOM

      SUBROUTINE MKVOLU(NS,RS,MODE,IMVNG,NTRANS,IMTRNS,IMXCEN,IMYCEN, &
                        IMZCEN,XDIM,XUCELL,XTLABC,X,Y,Z)
!
!       Change the system volume for constant pressure simulations.
!
!       Aaron R. Dinner
!
  use number
  use psf
        INTEGER IMVNG(:), NTRANS, XDIM, NS, MODE
        real(chm_real)  IMXCEN, IMYCEN, IMZCEN, IMTRNS(*)
        real(chm_real)  RS, XUCELL(*), XTLABC(*)
        real(chm_real)  X(*), Y(*), Z(*)
!
        INTEGER I, J, K, L, NA, NG, NN, IAF, IAL, IPT
        real(chm_real)  XP, YP, ZP, XP2, YP2, ZP2, DXP, DYP, DZP, MT

!       Translate the coordinates to be scaled such that the
!       image center is at the origin
        CALL TRNALL(X, Y, Z, IMVNG, -IMXCEN, -IMYCEN, -IMZCEN)

!       Counter for number of scaled bodies
!       If first call, scale image transformation
        IF (NS .EQ. 0) CALL SCLIMG(RS,NTRANS,IMTRNS,IMXCEN,IMYCEN, &
                                   IMZCEN,XDIM,XUCELL,XTLABC)

!       Scale primary coordinates

!       Do atom-based scaling
        IF (MODE.EQ.4) THEN

          NA = IMVNG(2)
          DO I = 4, NA, 2
            IAF = IMVNG(I - 1)
            IAL = IMVNG(I)
            DO J=IAF, IAL
              NS = NS + 1
              X(J) = X(J)*RS
              Y(J) = Y(J)*RS
              Z(J) = Z(J)*RS
            ENDDO
          ENDDO

        ELSE IF (MODE.LE.3) THEN

!         Do residue or heavy atom based scaling
          NG = IMVNG(1)
          J = NG + 2
          DO I = 2, NG
            NS = NS + 1
            NN = IMVNG(I)
            XP = ZERO
            YP = ZERO
            ZP = ZERO
            MT = ZERO
            DO K = J, NN, 2
              IAF = IMVNG(K - 1)
              IAL = IMVNG(K)
!             Get center of mass
              DO L=IAF, IAL
                XP = XP + AMASS(L)*X(L)
                YP = YP + AMASS(L)*Y(L)
                ZP = ZP + AMASS(L)*Z(L)
                MT = MT + AMASS(L)
              ENDDO
            ENDDO
            XP = XP / MT
            YP = YP / MT
            ZP = ZP / MT
!           Get scaled center of mass
            XP2= XP * RS
            YP2= YP * RS
            ZP2= ZP * RS
!           Get translation vector
            DXP = XP2 - XP
            DYP = YP2 - YP
            DZP = ZP2 - ZP
!           Translate residue
            DO K = J, NN, 2
              IAF = IMVNG(K - 1)
              IAL = IMVNG(K)
              DO L=IAF, IAL
                X(L) = X(L) + DXP
                Y(L) = Y(L) + DYP
                Z(L) = Z(L) + DZP
              ENDDO
            ENDDO
            J = NN + 2
          ENDDO
        ENDIF

!       Translate the coordinates back
        CALL TRNALL(X, Y, Z, IMVNG, IMXCEN, IMYCEN, IMZCEN)

        RETURN
      END SUBROUTINE MKVOLU

      SUBROUTINE SCLIMG(RS,NTRANS,IMTRNS,IMXCEN,IMYCEN,IMZCEN,XDIM, &
                        XUCELL,XTLABC)
!
!       Scale the image transformations
!
        INTEGER NTRANS, XDIM
        real(chm_real)  IMXCEN, IMYCEN, IMZCEN, IMTRNS(*)
        real(chm_real)  RS, XUCELL(*), XTLABC(*)
!
        INTEGER I, IPT

        DO I = 1, NTRANS
          IPT = (I - 1)*12
          IMTRNS(IPT+10) = IMTRNS(IPT+10)*RS
          IMTRNS(IPT+11) = IMTRNS(IPT+11)*RS
          IMTRNS(IPT+12) = IMTRNS(IPT+12)*RS
        ENDDO

!       Update crystal data for volume calculation for virial
        IF (XDIM .GT. 0) THEN
          XUCELL(1) = XUCELL(1)*RS
          XUCELL(2) = XUCELL(2)*RS
          XUCELL(3) = XUCELL(3)*RS
          DO I = 1, 6
            XTLABC(I) = XTLABC(I)*RS
          ENDDO
        ENDIF
        RETURN
      END SUBROUTINE SCLIMG

#endif 
end module mcmvrtrn

