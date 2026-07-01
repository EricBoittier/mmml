module mcmvad
#if KEY_MC==1
contains

  SUBROUTINE MOVEAD(COMLYN,COMLEN)
    !
    !       Top parsing routine for adding a move to the Monte Carlo
    !       move set.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use dimens_fcm
    use coord
    use gcmc
    use mc
    use stream
    use string
    use mcmvcrot, only: mvcrot
    use mcmvdihe, only: mvdihe
    use mcmvhmc, only: hmcpar, mvhmc
    use mcmvrtrn, only: mvrtrn
    use mcmvutil, only: padlab
#if KEY_SAMC==1
    use number
    use samc, only: spgetparamsbygroup,weparr,meparr,neparr,lspgroup
#endif
    use param_store, only: set_param

    implicit none
    !
    !       Passed Variables
    !
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    !
    !       Local Variables
    !
    INTEGER I, ICDLEN
    real(chm_real)  RMDX
    CHARACTER(len=80) MVCODE
    LOGICAL LCART
    !
    !       This is the NMVTYPth move
    !       Initialize IBLSTP to zero as a flag
    !
    NMVTYP = NMVTYP + 1
    IBLSTP(NMVTYP)%A => null()
    !
    !       Determine the type of move
    !
    CALL GTRMWD(COMLYN,COMLEN,'MVTP',4,MVCODE,80,ICDLEN)
    IF (ICDLEN .EQ. 0) &
         CALL WRNDIE (-5, '<MOVEAD>', 'NO MOVE TYPE SPECIFIED')

    !       If there is one, get the label for this move
    CALL GTRMWD(COMLYN,COMLEN,'LABEL',5,MVLABL(NMVTYP),4,ICDLEN)
    !       Fill the label with blanks so tables do not swim around later
    CALL PADLAB(MVLABL(NMVTYP),ICDLEN)

    CALL MRDPAR(COMLYN,COMLEN,WEIGHT(NMVTYP),RMDX, &
         ARMP(NMVTYP),ARMA(NMVTYP),ARMB(NMVTYP), &
         DOMCF(NMVTYP),ANISO(NMVTYP),NLIMIT(NMVTYP), &
         TFACT(NMVTYP),.TRUE.)
    CALL MINPAR(COMLYN,COMLEN,MCMINN(NMVTYP),MCMTYP(NMVTYP), &
         RMCSTP(NMVTYP),RMCMNF(NMVTYP),RMCMNG(NMVTYP), &
         RMCMNS(NMVTYP),.TRUE.)
    !
    !       Assign a unique integer code for this move type and then
    !       call the appropriate bookkeeping routine.
    !
    IF (MVCODE(1:4) .EQ. 'RTRN' .OR. MVCODE(1:4) .EQ. 'CART') THEN
       !         For backwards compatability
       LCART = (MVCODE(1:4) .EQ. 'CART')

       MVTYPE(NMVTYP) = 1
       CALL MVRTRN(COMLYN,COMLEN,LCART,MVTYPE(NMVTYP),NMVATM(NMVTYP), &
            IPIVTP(NMVTYP),IMVNGP(NMVTYP),IBLSTP(NMVTYP), &
            MDXP(NMVTYP),MBONDT,QBND(:,NMVTYP),ARMLIM(NMVTYP), &
            ARMMAX(NMVTYP),ANISO(NMVTYP),RMDX,X,Y,Z,WMAIN)
    ELSE IF (MVCODE(1:4) .EQ. 'RROT') THEN
       MVTYPE(NMVTYP) = 2
       CALL MVRTRN(COMLYN,COMLEN,.FALSE.,MVTYPE(NMVTYP), &
            NMVATM(NMVTYP),IPIVTP(NMVTYP),IMVNGP(NMVTYP), &
            IBLSTP(NMVTYP),MDXP(NMVTYP),MBONDT,QBND(:,NMVTYP), &
            ARMLIM(NMVTYP),ARMMAX(NMVTYP),ANISO(NMVTYP),RMDX, &
            X,Y,Z,WMAIN)
       !       CART now treated as RTRN
    ELSE IF (MVCODE(1:4) .EQ. 'TORS') THEN
       MVTYPE(NMVTYP) = 4
       CALL MVDIHE(COMLYN,COMLEN,NMVATM(NMVTYP), &
            IPIVTP(NMVTYP),IMVNGP(NMVTYP),IBLSTP(NMVTYP), &
            MDXP(NMVTYP),MBONDT,QBND(:,NMVTYP),ARMLIM(NMVTYP), &
            ARMMAX(NMVTYP),ANISO(NMVTYP),RMDX,X,Y,Z,WMAIN)
    ELSE IF (MVCODE(1:4) .EQ. 'CROT') THEN
       MVTYPE(NMVTYP) = 5
       CALL MVCROT(COMLYN,COMLEN,NMVATM(NMVTYP), &
            IPIVTP(NMVTYP),IMVNGP(NMVTYP),IBLSTP(NMVTYP), &
            MDXP(NMVTYP),MBONDT,QBND(:,NMVTYP),ARMLIM(NMVTYP), &
            ARMMAX(NMVTYP),ANISO(NMVTYP),RMDX,X,Y,Z,WMAIN)
    ELSE IF (MVCODE(1:4) .EQ. 'HMC ') THEN
       MVTYPE(NMVTYP) = 6
       IF (MCMINN(NMVTYP).GT.0) CALL WRNDIE(-2,'<MOVEAD>', &
            'MINI NOT ALLOWED WITH HMC')
       CALL HMCPAR(COMLYN,COMLEN,RMDX,MCMINN(NMVTYP),.TRUE. &
#if KEY_MEHMC==1
            ,MCMTYP(NMVTYP),RMCSTP(NMVTYP),RMCMNF(NMVTYP)  & 
#endif
            )
       MCMINN(NMVTYP) = -MCMINN(NMVTYP)
       CALL MVHMC(COMLYN,COMLEN,NMVATM(NMVTYP),IMVNGP(NMVTYP), &
            MDXP(NMVTYP),MBONDT,QBND(:,NMVTYP),ARMLIM(NMVTYP), &
            ARMMAX(NMVTYP),ANISO(NMVTYP),RMDX,X,Y,Z,WMAIN)
    ELSE IF (MVCODE(1:4) .EQ. 'VOLU') THEN
       MVTYPE(NMVTYP) = 7
       CALL MVRTRN(COMLYN,COMLEN,.FALSE.,MVTYPE(NMVTYP), &
            NMVATM(NMVTYP),IPIVTP(NMVTYP),IMVNGP(NMVTYP), &
            IBLSTP(NMVTYP),MDXP(NMVTYP),MBONDT,QBND(:,NMVTYP), &
            ARMLIM(NMVTYP),ARMMAX(NMVTYP),ANISO(NMVTYP), &
            RMDX,X,Y,Z,WMAIN)
    ELSE IF (MVCODE(1:4) .EQ. 'GCMC') THEN
#if KEY_GCMC==1
       MVTYPE(NMVTYP) = 8
       CALL MVRTRN(COMLYN,COMLEN,.FALSE.,MVTYPE(NMVTYP), &
            NMVATM(NMVTYP),IPIVTP(NMVTYP),IMVNGP(NMVTYP), &
            IBLSTP(NMVTYP),MDXP(NMVTYP),MBONDT,QBND(:,NMVTYP), &
            ARMLIM(NMVTYP),ARMMAX(NMVTYP),ANISO(NMVTYP), &
            RMDX,X,Y,Z,WMAIN)
#else /**/
       CALL WRNDIE(-2,'<MOVEAD>','GCMC code not compiled')
#endif 
    ELSE
       CALL WRNDIE(-2,'<MOVEAD>','UNKNOWN MOVE TYPE')
    ENDIF
    !
    !       Some information printing here, especially if the move
    !       was deleted.
    !
    IF (PRNLEV .GE. 2) THEN
       WRITE (OUTU,'(A,1X,I3,1X,A,1X,I6)') &
            ' MOVEAD> Number of move instances for move index', &
            NMVTYP, ' = ', NMVATM(NMVTYP)
    ENDIF
    CALL set_param('NMVI', NMVATM(NMVTYP))
    !
    !       If there are no appropriate atoms, forget the move.
    !
    IF (NMVATM(NMVTYP) .EQ. 0) THEN
       CALL WRNDIE(-1,'<MOVEAD>','NO MOVE INSTANCES---NOT COUNTED')
       NMVTYP = NMVTYP - 1
    ENDIF

    !       Set up arrays for move linking.
    !       Each move group starts active and unlinked.
    NACMVG = NACMVG + 1
    IACMVG(NACMVG) = NMVTYP
    NXTMVG(NMVTYP) = 0
    ILNMVP(NMVTYP) = IMVNGP(NMVTYP)
    ILNBDP(NMVTYP) = IBLSTP(NMVTYP)
    DO I = 1, MBONDT
       QLNBND(I,NMVTYP) = QBND(I,NMVTYP)
    ENDDO

#if KEY_SAMC==1 /* (samc_read_params) */
    IF (NMVTYP .EQ. 1) THEN
        WEPARR = ZERO
        MEPARR = 0
        NEPARR = 0
        LSPGROUP = .FALSE.
    ENDIF
    CALL SPGETPARAMSBYGROUP(COMLYN,COMLEN,NMVTYP,MVTYPE(NMVTYP),MVCODE(1:4))
#endif /* (samc_read_params) */

    RETURN
  END SUBROUTINE MOVEAD

  SUBROUTINE MRDPAR(COMLYN,COMLEN,WEIGHT,RMDX, &
       ARMP,ARMA,ARMB,DOMCF,ANISO,NLIMIT,TFACT, &
       LINIT)

    !
    !       Reads the parameters associated with an MC move group.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use number
    use string

    implicit none

    CHARACTER(len=*) COMLYN
    INTEGER COMLEN, NLIMIT
    real(chm_real)  RMDX, WEIGHT, ARMP, ARMA, ARMB, DOMCF, TFACT
    LOGICAL ANISO, LINIT
    !
    INTEGER IA

    !       If necessary, initialize the parameters.
    !       Otherwise leave unchanged.
    IA       =  0
    IF (LINIT) THEN
       RMDX   =  ONE
       WEIGHT =  ONE
       NLIMIT =  1
       ARMP   = -ONE
       ARMA   =  0.4
       ARMB   =  0.4
       DOMCF  = -ONE
       TFACT  =  ONE
    ELSE
       !         Flag for MOVE EDIT
       RMDX     = -ONE
       !         To keep anisotropy the same
       IF (ANISO) IA = 1
    ENDIF

    RMDX   = GTRMF(COMLYN,COMLEN,'DMAX', RMDX)
    WEIGHT = GTRMF(COMLYN,COMLEN,'WEIG', WEIGHT)

    !       Number of restricted rotatable dihedral past driver in CROT
    NLIMIT = GTRMI(COMLYN,COMLEN,'NLIM', NLIMIT)
    IF (NLIMIT .GT. 5) THEN
       CALL WRNDIE (3, '<MRDPAR>', 'SETTING NLIMIT to 5')
       NLIMIT = 5
    ENDIF

    !
    !       Constants for ARM and DOMC move size optimization
    !       Probably ZERO is NOT the best default.
    !
    ARMP   = GTRMF(COMLYN,COMLEN,'ARMP', ARMP)
    ARMA   = GTRMF(COMLYN,COMLEN,'ARMA', ARMA)
    ARMB   = GTRMF(COMLYN,COMLEN,'ARMB', ARMB)
    DOMCF  = GTRMF(COMLYN,COMLEN,'DOMC', DOMCF)

    ANISO  = GTRMI(COMLYN,COMLEN,'ANIS', IA) .GT. 0

    !       Allow moves to have different temperature factors
    TFACT  = GTRMF(COMLYN,COMLEN,'TFAC', TFACT)

    RETURN
  END SUBROUTINE MRDPAR


  SUBROUTINE MOVEED(COMLYN,COMLEN)
    !
    !       Edit the parameters for a move.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use mc
    use memory
    use exfunc
#if KEY_MEHMC==1
    use mcmvhmc, only: gtbpvc, hmcpar
#else /**/
    use mcmvhmc, only: hmcpar
#endif 
    use mcmvutil
    use string

    implicit none
    !
    !       Passed Variables
    !
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN

    INTEGER IMVTYP, I, J, K, L, IADD, ICDLEN
    real(chm_real),pointer,dimension(:) :: ITEMPP
    real(chm_real)  RMDX, RMAT(3,3)
    LOGICAL ANOLD
    CHARACTER(len=4) ML

    CALL GTRMWD(COMLYN,COMLEN,'LABEL',5,ML,4,ICDLEN)
    IF (ICDLEN .GT. 0) THEN
       CALL PADLAB(ML,ICDLEN)
       IMVTYP = MATCHL(NMVTYP,MVLABL,ML)
    ELSE
       IMVTYP = GTRMI(COMLYN,COMLEN,'MVIN', 0)
    ENDIF
    IF (IMVTYP .EQ. 0) THEN
       CALL WRNDIE (-5, '<MOVEED>', 'NO MOVE INDEX SPECIFIED')
    ELSE IF (IMVTYP .GT. NMVTYP) THEN
       CALL WRNDIE (-5, '<MOVEED>', 'MOVE INDEX > NUMBER OF MOVES')
    ENDIF

    ANOLD = ANISO(IMVTYP)
    CALL MRDPAR(COMLYN,COMLEN,WEIGHT(IMVTYP),RMDX, &
         ARMP(IMVTYP),ARMA(IMVTYP),ARMB(IMVTYP), &
         DOMCF(IMVTYP),ANISO(IMVTYP),NLIMIT(IMVTYP), &
         TFACT(IMVTYP),.FALSE.)
    CALL MINPAR(COMLYN,COMLEN,MCMINN(NMVTYP),MCMTYP(NMVTYP), &
         RMCSTP(NMVTYP),RMCMNF(NMVTYP),RMCMNG(NMVTYP), &
         RMCMNS(NMVTYP),.FALSE.)
    IF (MVTYPE(IMVTYP) .EQ. 6) THEN
       IF (MCMINN(NMVTYP).GT.0) CALL WRNDIE (-2, '<MOVEED>', &
            'MINI NOT ALLOWED WITH HMC')
       CALL HMCPAR(COMLYN,COMLEN,RMDX,MCMINN(NMVTYP),.FALSE. &
#if KEY_MEHMC==1
            ,MCMTYP(NMVTYP),RMCSTP(NMVTYP),RMCMNF(NMVTYP)  & 
#endif
            )
       MCMINN(NMVTYP) = -MCMINN(NMVTYP)
    ENDIF

    IF (RMDX .GT. 0) THEN

       IF (.NOT. ANOLD .AND. ANISO(IMVTYP)) THEN
          call chmdealloc('movead.src','MOVEED','MDXP(IMVTYP)', &
               NMVATM(IMVTYP),crlp=MDXP(IMVTYP)%A)
          call chmalloc('movead.src','MOVEED','MDXP(IMVTYP)', &
               9*NMVATM(IMVTYP),crlp=MDXP(IMVTYP)%A)
       ELSE  IF (ANOLD .AND. .NOT. ANISO(IMVTYP)) THEN
          call chmdealloc('movead.src','MOVEED','MDXP(IMVTYP)', &
               9*NMVATM(IMVTYP),crlp=MDXP(IMVTYP)%A)
          call chmalloc('movead.src','MOVEED','MDXP(IMVTYP)', &
               NMVATM(IMVTYP),crlp=MDXP(IMVTYP)%A)
       ENDIF

       IF (ANISO(IMVTYP)) THEN
          DO I = 1, NMVATM(IMVTYP)
             CALL FLANIS(RMDX, MDXP(IMVTYP)%A, I)
          ENDDO
       ELSE
          DO I = 1, NMVATM(IMVTYP)
             MDXP(IMVTYP)%A(I) = RMDX
          ENDDO
       ENDIF

    ELSE

       IF (.NOT. ANOLD .AND. ANISO(IMVTYP)) THEN
          call chmalloc('movead.src','MOVEED','ITEMPP',9*NMVATM(IMVTYP),crlp=ITEMPP)
          DO I = 1, NMVATM(IMVTYP)
             RMDX = MDXP(IMVTYP)%A(I)
             CALL FLANIS(RMDX, ITEMPP, I)
          ENDDO
          call chmdealloc('movead.src','MOVEED','MDXP(IMVTYP)', &
               NMVATM(IMVTYP),crlp=MDXP(IMVTYP)%A)
          MDXP(IMVTYP)%A => ITEMPP
       ELSE IF (ANOLD .AND. .NOT. ANISO(IMVTYP)) THEN
          call chmalloc('movead.src','MOVEED','ITEMPP',NMVATM(IMVTYP),crlp=ITEMPP)
          DO I = 1, NMVATM(IMVTYP)
             IADD = (I - 1)*9
             L = 0
             DO J = 1, 3
                DO K = 1, 3
                   L = L + 1
                   RMAT(J,K) = MDXP(IMVTYP)%A(IADD+L)
                ENDDO
             ENDDO
             CALL DETM33(RMAT,RMDX)
             RMDX = ABS(RMDX)**(1.0/3.0)
             ITEMPP(I) = RMDX
          ENDDO
          call chmdealloc('movead.src','MOVEED','MDXP(IMVTYP)', &
               9*NMVATM(IMVTYP),crlp=MDXP(IMVTYP)%A)
          MDXP(IMVTYP)%A => ITEMPP
       ENDIF

    ENDIF

    RETURN
  END SUBROUTINE MOVEED

  SUBROUTINE MOVEDL(COMLYN,COMLEN)
    !
    !       Deletes a move group from the MC move set.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use stream
    use exfunc
    use mc
    use mcmoveio, only: fremvs
    use mcmvutil, only: matchl, padlab
    use string

    implicit none
    !
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    !
    INTEGER I, J, K, IM, ICDLEN
    CHARACTER(len=4) ML

    CALL GTRMWD(COMLYN,COMLEN,'LABEL',5,ML,4,ICDLEN)
    IF (ICDLEN .GT. 0) THEN
       CALL PADLAB(ML,ICDLEN)
       IM = MATCHL(NMVTYP,MVLABL,ML)
    ELSE
       IM = GTRMI(COMLYN,COMLEN,'MVIN', 0)
    ENDIF
    IF (IM .EQ. 0) THEN
       CALL WRNDIE (-5, '<MOVEDL>', 'NO MOVE INDEX SPECIFIED')
    ELSE IF (IM .GT. NMVTYP) THEN
       CALL WRNDIE (-5, '<MOVEDL>', 'MOVE INDEX > NUMBER OF MOVES')
    ENDIF

    !       Check for linking
    IF (NXTMVG(IM) .NE. 0) CALL WRNDIE (-5, '<MOVEDL>', &
         'CANNOT DELETE LINKED MOVE')
    DO I = 1, NMVTYP
       IF (NXTMVG(I) .EQ. IM) CALL WRNDIE (-5, '<MOVEDL>', &
            'CANNOT DELETE LINKED MOVE')
    ENDDO

    !       Free the space
    CALL FREMVS(IMVNGP(IM),IBLSTP(IM),IPIVTP(IM),ILNMVP(IM), &
         ILNBDP(IM),MDXP(IM),NMVATM(IM),MVTYPE(IM), &
         MBONDT,QBND(1,IM),QLNBND(1,IM),ANISO(IM))

    !       Shift all the other moves down
    I = IM
    IM = IM + 1
    DO J = IM, NMVTYP
       MVLABL(I) = MVLABL(J)
       WEIGHT(I) = WEIGHT(J)
       ARMP(I)   = ARMP(J)
       ARMA(I)   = ARMA(J)
       ARMB(I)   = ARMB(J)
       ARMLIM(I) = ARMLIM(J)
       ARMMAX(I) = ARMMAX(J)
       DOMCF(I)  = DOMCF(J)
       ANISO(I)  = ANISO(J)
       NLIMIT(I) = NLIMIT(J)
       TFACT(I)  = TFACT(J)
       MCMINN(I) = MCMINN(J)
       RMCSTP(I) = RMCSTP(J)
       RMCMNF(I) = RMCMNF(J)
       RMCMNG(I) = RMCMNG(J)
       RMCMNS(I) = RMCMNS(J)
       MVTYPE(I) = MVTYPE(J)
       NMVATM(I) = NMVATM(J)
       IPIVTP(I) = IPIVTP(J)
       IMVNGP(I) = IMVNGP(J)
       ILNMVP(I) = ILNMVP(J)
       ILNBDP(I) = ILNBDP(J)
       IBLSTP(I) = IBLSTP(J)
       MDXP(I)   = MDXP(J)
       DO K = 1, MBONDT
          QBND(K,I) = QBND(K,J)
          QLNBND(K,I) = QLNBND(K,J)
       ENDDO
       I = J
    ENDDO

    NMVTYP = NMVTYP - 1

    DO I = 1, NMVTYP
       IF (NXTMVG(I).GT.IM) NXTMVG(I) = NXTMVG(I) - 1
    ENDDO
    DO I = 1, NACMVG
       IF (IACMVG(I).EQ.IM) THEN
          IACMVG(I) = IACMVG(NACMVG)
       ENDIF
    ENDDO
    NACMVG = NACMVG - 1
    DO I = 1, NACMVG
       IF (IACMVG(I).GT.IM) IACMVG(I) = IACMVG(I) - 1
    ENDDO

    RETURN
  END SUBROUTINE MOVEDL

  SUBROUTINE MINPAR(COMLYN,COMLEN,NSTEPS,MCMTYP,STEP, &
       TOLFUN,TOLGRD,TOLSTP,LINIT)
    !
    !       Reads the parameters associated with minimization of
    !       a move instance.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use exfunc
    use number
    use contrl
    use string

    implicit none

    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    INTEGER NSTEPS, MCMTYP
    real(chm_real)  STEP, TOLFUN, TOLGRD, TOLSTP
    !
    INTEGER ICDLEN
    CHARACTER(len=4) MINOPT
    LOGICAL LINIT

    IF (LINIT) THEN
       MCMTYP = 0
       NSTEPS = 0
       STEP   = PT02
       TOLFUN = ZERO
       TOLGRD = ZERO
       TOLSTP = ZERO
    ENDIF

    !       If specified, figure out which algorithm
    CALL GTRMWD(COMLYN,COMLEN,'MINI',4,MINOPT,4,ICDLEN)
    IF (ICDLEN .GT. 0) THEN
       IF (MINOPT .EQ. 'SD  ') THEN
          MCMTYP = 0
       ELSE IF (MINOPT .EQ. 'CONJ') THEN
          MCMTYP = 1
       ELSE
          CALL WRNDIE (-5, '<MINPAR>', 'UNKNOWN MCM ALGORITHM')
       ENDIF
    ENDIF

    NSTEPS = GTRMI(COMLYN,COMLEN,'NSTE', NSTEPS)

    STEP   = GTRMF(COMLYN,COMLEN,'STEP',STEP)
    TOLFUN = GTRMF(COMLYN,COMLEN,'TOLE',TOLFUN)
    TOLGRD = GTRMF(COMLYN,COMLEN,'TOLG',TOLGRD)
    TOLSTP = GTRMF(COMLYN,COMLEN,'TOLS',TOLSTP)

    !       These go through contrl.f90 directly to minimization
    INBFRQ = GTRMI(COMLYN,COMLEN,'INBF',INBFRQ)
    NPRINT = GTRMI(COMLYN,COMLEN,'NPRI',NPRINT)

    RETURN
  END SUBROUTINE MINPAR

  LOGICAL FUNCTION SINGLE(IMVNG)
    !
    !       .TRUE. if the move type is a single atom move.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    INTEGER IMVNG(*)
    SINGLE = (IMVNG(1).EQ.2 .AND. IMVNG(2).EQ.4 .AND. &
         IMVNG(3).EQ.IMVNG(4))
    RETURN
  END FUNCTION SINGLE

  SUBROUTINE IINIT(IARRY,N,I)
    !
    !       Assigns the value I to all N elements of the array IARRY
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    INTEGER IARRY(*), N, I
    INTEGER J
    DO J = 1, N
       IARRY(J) = I
    ENDDO
    RETURN
  END SUBROUTINE IINIT

#endif 
end module mcmvad

