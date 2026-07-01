module mcmvutil
#if KEY_MC==1
  use chm_types
  implicit none

  type(chm_iptr),allocatable,dimension(:) :: IABNDP, IATHTP, IAPHIP, IAIMPP, IAPIMP

contains

  SUBROUTINE MKBNDT()
    !
    !       Allocates and fills NATOM long lists with pointers to lists
    !       of bonds, angles, torsions that are affected by that atom.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use dimens_fcm
    use psf
    use memory
    use mc
#if KEY_PATHINT==1
    use mpathint   
#endif
    implicit none
    !
    !       Local Variables
    !
    integer,pointer,dimension(:) :: IAP
    INTEGER I
#if KEY_PATHINT==1
    INTEGER NBPIMC                
#endif

    ! TODO check for failure
    allocate(IABNDP(NATOM))
    allocate(IATHTP(NATOM))
    allocate(IAPHIP(NATOM))
    allocate(IAIMPP(NATOM))
#if KEY_PATHINT==1
    allocate(IAPIMP(NATOM))  
#endif
    !
    !       Use the first element of each list hanging off each atom to
    !       hold how many bonds, angles, and torsions for that atom.
    !
    DO I = 1, NATOM
       call chmalloc('movead.src','MKBNDT','IAP',MCMBND,intgp=IAP)
       IABNDP(I)%A => IAP
       IAP(1) = 0

       call chmalloc('movead.src','MKBNDT','IAP',MCMTHT,intgp=IAP)
       IATHTP(I)%A => IAP
       IAP(1) = 0

       call chmalloc('movead.src','MKBNDT','IAP',MCMPHI,intgp=IAP)
       IAPHIP(I)%A => IAP
       IAP(1) = 0

       call chmalloc('movead.src','MKBNDT','IAP',MCMIMP,intgp=IAP)
       IAIMPP(I)%A => IAP
       IAP(1) = 0
#if KEY_PATHINT==1
       !         ARD 00-11-28
       IF (QPINT) THEN
          call chmalloc('movead.src','MKBNDT','IAP',NBEADS,intgp=IAP)
          IAPIMP(I)%A => IAP
          IAP(1) = 0
       ENDIF
#endif 
    ENDDO
    !
    !       Bonds.
    !
    DO I = 1, NBOND
       IAP => IABNDP(IB(I))%A
       CALL ADDICT(I, IAP, MCMBND)
       IAP => IABNDP(JB(I))%A
       CALL ADDICT(I, IAP, MCMBND)
    ENDDO
    !
    !       Angles.
    !
    DO I = 1, NTHETA
       IAP => IATHTP(IT(I))%A
       CALL ADDICT(I, IAP, MCMTHT)
       IAP => IATHTP(JT(I))%A
       CALL ADDICT(I, IAP, MCMTHT)
       IAP => IATHTP(KT(I))%A
       CALL ADDICT(I, IAP, MCMTHT)
    ENDDO
    !
    !       Dihedrals.
    !
    DO I = 1, NPHI
       IAP => IAPHIP(IP(I))%A
       CALL ADDICT(I, IAP, MCMPHI)
       IAP => IAPHIP(JP(I))%A
       CALL ADDICT(I, IAP, MCMPHI)
       IAP => IAPHIP(KP(I))%A
       CALL ADDICT(I, IAP, MCMPHI)
       IAP => IAPHIP(LP(I))%A
       CALL ADDICT(I, IAP, MCMPHI)
    ENDDO
    !
    !       Impropers.
    !
    DO I = 1, NIMPHI
       IAP => IAIMPP(IM(I))%A
       CALL ADDICT(I, IAP, MCMIMP)
       IAP => IAIMPP(JM(I))%A
       CALL ADDICT(I, IAP, MCMIMP)
       IAP => IAIMPP(KM(I))%A
       CALL ADDICT(I, IAP, MCMIMP)
       IAP => IAIMPP(LM(I))%A
       CALL ADDICT(I, IAP, MCMIMP)
    ENDDO

#if KEY_PATHINT==1
    !       Path integral springs
    IF (QPINT) THEN
       ! TODO relocate inside mpathint
       IF (.not. allocated(IBPIMC) .OR. .not. allocated(JBPIMC)) &
            CALL WRNDIE(-5, '<MKBNDT>', 'MC keyword necessary in PINT')
       NBPIMC = NBEADS*NPIAT
       DO I = 1, NBPIMC
          IAP => IAPIMP(IBPIMC(I))%A
          CALL ADDICT(I, IAP, NBEADS)
          IAP => IAPIMP(JBPIMC(I))%A
          CALL ADDICT(I, IAP, NBEADS)
       ENDDO
    ENDIF
#endif 

    RETURN
  END SUBROUTINE MKBNDT

  SUBROUTINE ADDICT(IIC,ILIST,NMAX)
    !
    !       Puts IIC in the appropriate spot in an IC list and increments
    !       the spot counting how many list members.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    ! TEST
    use mc
    implicit none
    INTEGER IIC, ILIST(:), NMAX
    INTEGER N
    ILIST(1) = ILIST(1) + 1
    N = ILIST(1) + 1
    IF (N .GT. NMAX) CALL WRNDIE (-5, '<ADDICT>', &
         'Exceeded maximum number of bonded terms.')
    ILIST(N) = IIC
    RETURN
  END SUBROUTINE ADDICT

  SUBROUTINE FREBND()
    !
    !       Frees bookkeeping arrays for bonded terms to atoms.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use dimens_fcm
    use psf
    use memory
    use mc
#if KEY_PATHINT==1
    use mpathint      
#endif
    implicit none

    integer,pointer,dimension(:) :: IAP
    INTEGER I

    DO I = 1, NATOM
       IAP => IABNDP(I)%A
       call chmdealloc('movead.src','FREBND','IAP',MCMBND,intgp=IAP)
       IAP => IATHTP(I)%A
       call chmdealloc('movead.src','FREBND','IAP',MCMTHT,intgp=IAP)
       IAP => IAPHIP(I)%A
       call chmdealloc('movead.src','FREBND','IAP',MCMPHI,intgp=IAP)
       IAP => IAIMPP(I)%A
       call chmdealloc('movead.src','FREBND','IAP',MCMIMP,intgp=IAP)
#if KEY_PATHINT==1
       IF (QPINT) THEN
          IAP => IAPIMP(I)%A
          call chmdealloc('movead.src','FREBND','IAP',NBEADS,intgp=IAP)
       ENDIF
#endif 
    ENDDO
    deallocate(IABNDP)
    deallocate(IATHTP)
    deallocate(IAPHIP)
    deallocate(IAIMPP)
#if KEY_PATHINT==1
    deallocate(IAPIMP)  
#endif

    RETURN
  END SUBROUTINE FREBND

  SUBROUTINE FREBLS(LISTP, MBONDT, QBND)
    !
    !       Frees a bonded term list for a single move instance.
    !
    !       Aaron R. Dinner
    !
    use chm_types
    use memory
    implicit none

    type(chm_iptr) :: LISTP
    INTEGER MBONDT
    LOGICAL QBND(MBONDT)
    !
    INTEGER I, NB
    !
    NB = 0
    DO I = 1, MBONDT
       IF (QBND(I)) THEN
          NB = NB + 1
          NB = LISTP%A(NB)
       ENDIF
    ENDDO
    call chmdealloc('moveio.src','FREBLS','LISTP',NB,intgp=LISTP%A)

    RETURN
  END SUBROUTINE FREBLS

  SUBROUTINE MVATM(IMVATM,NATOM,IATOM,IAPREV,IABND, &
       IB,JB,IPATH,NPATH)
    !
    !       Tags the atoms that move with IATOM.
    !       Works by a depth first search given the connectivity list.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    implicit none
    !
    !       Passed Variables
    !
    INTEGER NATOM, IATOM, IAPREV
    type(chm_iptr) :: IABND(:)
    INTEGER IMVATM(*), IPATH(*), NPATH(*)
    INTEGER IB(*), JB(*)
    !
    !       Local Variables
    !
    INTEGER I, N, J, IM
    !
    !       Mark IAPREV to prevent searching that direction.
    !
    IMVATM(IAPREV) = 1
    !
    !       IM is the present atom in the path.
    !       IPATH is the list of atoms the path traverses.
    !       NPATH is the number of bonds already tried for that atom.
    !
    IM = 1
    IPATH(IM) = IATOM
    NPATH(IM) = 0
    IMVATM(IATOM) = 1

10  NPATH(IM) = NPATH(IM) + 1
    !
    !       Try the next move for the current atom.
    !       Check that there is another bond to move across.
    !            IABND contains pointers to lists where the first
    !            element is the number of bonds and the subsequent ones
    !            are the bond indices themselves.
    !       Check that we haven't been there before.
    !
    IF (NPATH(IM) .GT. IABND(IPATH(IM))%A(1)) GOTO 100
    I = IABND(IPATH(IM))%A(NPATH(IM) + 1)
    I = NOTATM(IB(I),JB(I),IPATH(IM))
    IF (IMVATM(I) .GT. 0) GOTO 10

    IM = IM + 1
    IPATH(IM) = I
    NPATH(IM) = 0
    IMVATM(I) = 1

    GOTO 10

100 CONTINUE
    !
    !       Have tried all possible moves for this atom --- move back.
    !
    NPATH(IM) = 0
    IM = IM - 1

    IF (IM .GT. 0) GOTO 10
    !
    !       Unmark IAPREV
    !
    IMVATM(IAPREV) = 0


    RETURN
  END SUBROUTINE MVATM

  SUBROUTINE TAGATM(IMVATM,IMVNG,LVAL,IVAL)
    !
    !       Tags moving atoms.
    !       If with LVAL is true,  tagged with IVAL.
    !       If with LVAL is false, tagged with MC group number in IMVNG.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    INTEGER IMVATM(:), IMVNG(:), IVAL
    LOGICAL LVAL
    !
    INTEGER I, J, K, L, IAF, IAL, NG, NN, ITAG

    ITAG = IVAL

    NG = IMVNG(1)
    J = NG + 2
    DO I = 2, NG
       IF (.NOT. LVAL) ITAG = I - 1
       NN = IMVNG(I)
       DO K = J, NN, 2
          IAF = IMVNG(K-1)
          IAL = IMVNG(K)
          DO L = IAF, IAL
             IMVATM(L) = ITAG
          ENDDO
       ENDDO
       J = NN + 2
    ENDDO

    RETURN
  END SUBROUTINE TAGATM

  SUBROUTINE IMVADD(ILISTP,ICALL,IAPNDP)
    !
    !       Adds the list that stores the moving atoms to end of previous list.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    use memory
    implicit none

    INTEGER ICALL
    type(chm_iptr) :: ILISTP, IAPNDP

    INTEGER NGOLD, NPOLD, NGNEW, NPNEW, N, I, J
    integer,pointer,dimension(:) :: ITEMPP, INEWP

    IF (ICALL .EQ. 1) THEN
       ILISTP%A => IAPNDP%A
       RETURN
    ENDIF

    ITEMPP => ILISTP%A
    NGOLD = ITEMPP(1)
    NPOLD = ITEMPP(NGOLD)
    NGNEW = IAPNDP%A(1)
    NPNEW = IAPNDP%A(NGNEW)
    N = NPOLD + NPNEW - 1

    call chmalloc('moveln.src','IMVADD','INEWP',N,intgp=INEWP)
    J = 1
    DO I = 2, NGOLD
       J = J + 1
       INEWP(J) = ITEMPP(I) + NGNEW - 1
    ENDDO
    DO I = 2, NGNEW
       J = J + 1
       INEWP(J) = IAPNDP%A(I) + NPOLD - 1
    ENDDO
    INEWP(1) = J
    NGOLD = NGOLD + 1
    NGNEW = NGNEW + 1
    DO I = NGOLD, NPOLD
       J = J + 1
       INEWP(J) = ITEMPP(I)
    ENDDO
    DO I = NGNEW, NPNEW
       J = J + 1
       INEWP(J) = IAPNDP%A(I)
    ENDDO

    ILISTP%A => INEWP
    call chmdealloc('moveln.src','IMVADD','ITEMPP',NPOLD,intgp=ITEMPP)
    call chmdealloc('moveln.src','IMVADD','IAPNDP',NPNEW,intgp=IAPNDP%A)

    RETURN
  END SUBROUTINE IMVADD

  SUBROUTINE IMVLST(ILISTP,NATOM,IMVATM)
    !
    !       Tabulate the moving atoms.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    use memory
    implicit none
    !
    !       Passed Variables
    !
    type(chm_iptr) :: ILISTP
    INTEGER NATOM, IMVATM(*)
    !
    !       Local Variables
    !
    INTEGER I, J, N, IPREV, NPAIR
    !
    !       Run through the list to count how many pairs to allocate space.
    !
    NPAIR = 0
    IPREV = 0
    DO I = 1, NATOM
       IF ((IPREV .EQ. 0) .AND. (IMVATM(I) .GE. 1)) NPAIR = NPAIR + 1
       IPREV = IMVATM(I)
    ENDDO
    !
    !       Fill the list.
    !
    N = 2*NPAIR + 2
    call chmalloc('movead.src','IMVLST','ILISTP',N,intgp=ILISTP%A)
    ILISTP%A(1) = 2
    ILISTP%A(2) = N
    J = 2
    IPREV = 0
    DO I = 1, NATOM
       IF ((IPREV .EQ. 0) .AND. (IMVATM(I) .GE. 1)) THEN
          J = J + 1
          ILISTP%A(J) = I
       ELSE IF ((IPREV .GE. 1) .AND. (IMVATM(I) .EQ. 0)) THEN
          J = J + 1
          ILISTP%A(J) = I - 1
       ENDIF
       IPREV = IMVATM(I)
    ENDDO
    IF (IMVATM(NATOM) .GE. 1) THEN
       J = J + 1
       ILISTP%A(J) = NATOM
    ENDIF
    RETURN
  END SUBROUTINE IMVLST

  INTEGER FUNCTION R2INDX(R,RJAC,N)
    !
    !       Finds the window in which R falls.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    INTEGER N
    real(chm_real)  R, RJAC(N)
    !
    INTEGER I
    real(chm_real)  RJTOT

    RJTOT = 0.0
    DO I = 1, N
       RJTOT = RJTOT + RJAC(I)
       IF (R .LT. RJTOT) THEN
          R2INDX = I
          RETURN
       ENDIF
    ENDDO
    !       Pick up any rounding error
    R2INDX = N
    RETURN
  END FUNCTION R2INDX

  SUBROUTINE ROSPHR(ISEED,V)
    !
    !       Returns a random vector on a unit sphere.
    !
    !       Aaron R. Dinner
    !
    use clcg_mod,only:random
    use chm_kinds
    use number
    implicit none
    INTEGER ISEED
    real(chm_real) V(3)

100 V(1) = TWO*RANDOM(ISEED) - ONE
    V(2) = TWO*RANDOM(ISEED) - ONE
    V(3) = V(1)*V(1) + V(2)*V(2)
    IF (V(3).GT.ONE) GOTO 100

    V(1) = TWO*V(1)*SQRT(ONE - V(3))
    V(2) = TWO*V(2)*SQRT(ONE - V(3))
    V(3) = ONE - TWO*V(3)

    RETURN
  END SUBROUTINE ROSPHR

  SUBROUTINE CNTALL(NCNT,X,Y,Z,XCENT,YCENT,ZCENT,QCENT)
    !
    !       Get X, Y, Z, Q center of all groups.
    !
    !       Basically the same as the loops in ENBONDG, ENBFSG,
    !       and ENBFVG except that CG vs CGX is inside loop and
    !       the X, Y, Z calculations are in CNTONE.
    !
    !       Aaron  R. Dinner
    !
    use ewald_1m,only:lewald
    use chm_kinds
    use dimens_fcm
    use consta
    use fast
    use inbnd
    use number
    use psf
    implicit none
    INTEGER NCNT
    real(chm_real)  X(*), Y(*), Z(*), QCENT(*)
    real(chm_real)  XCENT(*), YCENT(*), ZCENT(*)

    INTEGER IRS, IS, IQ, I
    real(chm_real)  CGF, C2OFNB, DXI
    LOGICAL LCSW, LRSW

    CALL STLCNT(LCSW,LRSW,LELEC,EPS,LEWALD,LCONS)

    C2OFNB = CTOFNB*CTOFNB
    CGF = ZERO
    IF (LELEC .AND. EPS.NE.0.0) CGF = CCELEC/EPS
    DO IRS = 1, NCNT
       CALL CNTONE(XCENT,YCENT,ZCENT,IRS,IGPBS,X,Y,Z)
    ENDDO

    IF (LFSWT) THEN
       DO IRS = 1, NCNT
          IS = IGPBS(IRS)+1
          IQ = IGPBS(IRS+1)
          DXI=ZERO
          DO I = IS, IQ
             DXI=DXI+CG(I)
          ENDDO
          QCENT(IRS)=ZERO
          IF (LCSW) QCENT(IRS)=DXI*SQRT(CGF/CTOFNB)
          IF (LRSW) QCENT(IRS)=DXI*SQRT(CGF/C2OFNB)
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE CNTALL

  SUBROUTINE CNTONE(XCENT,YCENT,ZCENT,IRS,IGPBS,X,Y,Z)
    !
    !       Gets one group center.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use number
    implicit none

    INTEGER IRS, IGPBS(*)
    real(chm_real)  XCENT(*), YCENT(*), ZCENT(*)
    real(chm_real)  X(*), Y(*), Z(*)

    INTEGER I, IS, IQ, NAT
    real(chm_real)  DXI, DYI, DZI
    IS=IGPBS(IRS)+1
    IQ=IGPBS(IRS+1)
    NAT=IQ-IS+1
    DXI=ZERO
    DYI=ZERO
    DZI=ZERO

    DO I = IS, IQ
       DXI=DXI+X(I)
       DYI=DYI+Y(I)
       DZI=DZI+Z(I)
    ENDDO
    XCENT(IRS)=DXI/NAT
    YCENT(IRS)=DYI/NAT
    ZCENT(IRS)=DZI/NAT

    RETURN
  END SUBROUTINE CNTONE

  SUBROUTINE STLCNT(LCSW,LRSW,LELEC,EPS,LEWALD,LCONS)
    !
    !       Sets flags for QCENT calculation.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use number
    implicit none
    real(chm_real)  EPS
    LOGICAL LCSW, LRSW, LELEC, LEWALD, LCONS

    IF (.NOT. LELEC .OR. EPS .EQ. ZERO) THEN
       LCSW=.FALSE.
       LRSW=.FALSE.
    ELSE IF (LEWALD) THEN
       LCSW=.FALSE.
       LRSW=.FALSE.
    ELSE IF (LCONS) THEN
       LCSW=.TRUE.
       LRSW=.FALSE.
    ELSE
       LCSW=.FALSE.
       LRSW=.TRUE.
    ENDIF

    RETURN
  END SUBROUTINE STLCNT

  INTEGER FUNCTION NOTATM(I, J, K)
    !
    !       Returns the atom in the bond that is not K.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    INTEGER I, J, K
    IF (I .NE. K) THEN
       NOTATM = I
    ELSE
       NOTATM = J
    ENDIF
    RETURN
  END FUNCTION NOTATM

  SUBROUTINE PADLAB(MVLAB,ILEN)
    !
    !       Pads the 4 character labels with blanks.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    INTEGER ILEN
    CHARACTER(len=4) MVLAB
    INTEGER I
    ILEN = ILEN + 1
    DO I = ILEN, 4
       MVLAB(I:I) = ' '
    ENDDO
    RETURN
  END SUBROUTINE PADLAB

  SUBROUTINE FILLQB(QBND,L1,L2,L3,L4,L5)
    !
    !       Fills the IQBND array.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use dimens_fcm
#if KEY_PATHINT==1
    use mpathint      
#endif
    implicit none

    LOGICAL QBND(5), L1, L2, L3, L4, L5

    QBND(1) = L1
    QBND(2) = L2
    QBND(3) = L3
    QBND(4) = L4
#if KEY_PATHINT==1
    QBND(5) = L5 .AND. QPINT
#else /**/
    QBND(5) = L5
#endif 
    RETURN
  END SUBROUTINE FILLQB

  SUBROUTINE FLANIS(R,RMX,I)
    !
    !       Fill the anisotropic move limit matrix with isotropic values
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use number
    implicit none
    INTEGER I
    real(chm_real)  R, RMX(:)

    INTEGER IADD

    IADD = (I - 1)*9
    RMX(IADD+1) = R
    RMX(IADD+2) = ZERO
    RMX(IADD+3) = ZERO
    RMX(IADD+4) = ZERO
    RMX(IADD+5) = R
    RMX(IADD+6) = ZERO
    RMX(IADD+7) = ZERO
    RMX(IADD+8) = ZERO
    RMX(IADD+9) = R

    RETURN
  END SUBROUTINE FLANIS

  INTEGER FUNCTION MATCHL(NMVTYP,MVLABL,ML)
    !
    !       Finds the index with label ML
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    !
    INTEGER NMVTYP
    CHARACTER(len=4) MVLABL(*), ML
    !
    DO MATCHL = 1, NMVTYP
       IF (ML .EQ. MVLABL(MATCHL)) RETURN
    ENDDO
    MATCHL = 0
    RETURN
  END FUNCTION MATCHL

#endif 
end module mcmvutil

