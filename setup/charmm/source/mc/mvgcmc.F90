module mcmvgcmc
#if KEY_MC==1 /*mc*/
  use chm_types
  use dimens_fcm
  implicit none

  integer,allocatable,dimension(:) :: GRDBLK, NGCBLK, LSTGRD, LSTIND

contains
#if KEY_GCMC==1 /*gcmc*/
  SUBROUTINE MKGCMC(GCMCON,LCB,ISEED,IMVNG,NGCIN,NGCTRY,GCCUT2, &
       XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
       ISCVTY,NGCPCP,X,Y,Z, CUTNB2,QGCGRD,QGCSPH, &
       GRDBLK,NGRDBK,NGRDCV,LSTGRD,GRDSIZ, &
       NGRDX,NGRDY,NGRDZ,NGCBLK)
    !
    !       Applies a grand canonical move.
    !
    !       Aaron R. Dinner, Hyung-June Woo and Benoit Roux
    !
    use memory
    use number
    use clcg_mod,only: random
    use mcmvrtrn

    implicit none

    INTEGER ISEED, IMVNG(:), ISCVTY(:), NGCPCP(:)
    INTEGER NGCTRY, NGCIN
    integer,dimension(:) :: LSTGRD, NGCBLK, GRDBLK
    INTEGER NGRDCV, NGRDX, NGRDY, NGRDZ, NGRDBK
    real(chm_real)  GCCUT2, GRDSIZ, CUTNB2
    real(chm_real)  XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX
    real(chm_real)  X(*), Y(*), Z(*)
    LOGICAL GCMCON(:), QGCGRD,QGCSPH,LCB

    integer,allocatable,dimension(:) :: ICVTYP
    real(chm_real),allocatable,dimension(:) :: XNP, YNP, ZNP
    INTEGER I, IDX, NCAVTY
    real(chm_real)  DXGC, DYGC, DZGC
    real(chm_real)  XN, YN, ZN
    real(chm_real)  XCM, YCM, ZCM
    real(chm_real)  DP, R
    LOGICAL QINS

    IF (NGCTRY > 0) THEN
       call chmalloc('mvgcmc.src','MKGCMC','ICVTYP',NGCTRY,intg=ICVTYP)
       call chmalloc('mvgcmc.src','MKGCMC','XNP',NGCTRY,crl=XNP)
       call chmalloc('mvgcmc.src','MKGCMC','YNP',NGCTRY,crl=YNP)
       call chmalloc('mvgcmc.src','MKGCMC','ZNP',NGCTRY,crl=ZNP)
    ENDIF

    IF (GCMCON(FRSIND(IMVNG))) THEN
       !         For deletion, turn off GCMCON first!
       CALL TOGLGC(GCMCON,IMVNG)
       IF (NGCTRY .GT. 0) then
          CALL CVTDEL(ISEED,LCB, ISCVTY, NGCPCP, &
               ICVTYP, NGCTRY,GCCUT2, &
               XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
               NGCIN,GCMCON, XNP, YNP, ZNP, &
               X,Y,Z,QGCSPH, GRDBLK, NGRDBK)
       ENDIF
    ELSE
       !         It is an insertion
       QINS = .FALSE.
       IF (QGCGRD .AND. NGRDCV.GT.0) THEN
          !           Grid-based cavity bias insertion
          CALL CVTINS(XN,YN,ZN,ISEED,NGRDCV, LSTGRD, &
               NGRDX,NGRDY,XGCMIN,YGCMIN,ZGCMIN,GRDSIZ)
          QINS = .TRUE.
       ELSE IF (NGCTRY .GT. 0) THEN
          !           Continuous space cavity bias insertion
          CALL GNGCPT(NCAVTY, ICVTYP, ISEED,NGCTRY,GCCUT2, &
               XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
               NGCIN, ISCVTY, NGCPCP, GCMCON, &
               XNP, YNP, ZNP, X,Y,Z,QGCSPH, &
               GRDBLK, NGRDBK)
          !           If no points in cavity, insert randomly
          LCB = NCAVTY .GT. 0
          IF (LCB) THEN
             QINS = .TRUE.
             IF (NCAVTY .EQ. 1) THEN
                I = 1
             ELSE
                I = DBLE(NCAVTY)*RANDOM(ISEED) + ONE
             ENDIF
             I = ICVTYP(I)
             XN = XNP(I)
             YN = YNP(I)
             ZN = ZNP(I)
          ENDIF
       ENDIF

       !         If we have not yet succeeded in inserting, do it randomly
       IF (.NOT. QINS) THEN
          DXGC = XGCMAX - XGCMIN
          DYGC = YGCMAX - YGCMIN
          DZGC = ZGCMAX - ZGCMIN
123       XN = DXGC*RANDOM(ISEED) + XGCMIN
          YN = DYGC*RANDOM(ISEED) + YGCMIN
          ZN = DZGC*RANDOM(ISEED) + ZGCMIN
          !           Pick a new point if outside the sphere allowed
          IF (QGCSPH .AND. RSPOUT(XN,YN,ZN,XGCMIN,YGCMIN,ZGCMIN, &
               DXGC,DYGC,DZGC)) GOTO 123
       ENDIF

       !         Need to change to center of mass
       CALL MVGCOM(XCM,YCM,ZCM,IMVNG,X,Y,Z)
       XN = XN - XCM
       YN = YN - YCM
       ZN = ZN - ZCM
       CALL TRNALL(X,Y,Z,IMVNG,XN,YN,ZN)
       CALL MKRROT(DP,X,Y,Z,IMVNG,-1,ONE8TY,ISEED)

       !         For insertion, turn on the molecule after calculating Pc(N)
       CALL TOGLGC(GCMCON,IMVNG)

    ENDIF

    IF (NGCTRY > 0) THEN
       call chmdealloc('mvgcmc.src','MKGCMC','ZNP',NGCTRY,crl=ZNP)
       call chmdealloc('mvgcmc.src','MKGCMC','YNP',NGCTRY,crl=YNP)
       call chmdealloc('mvgcmc.src','MKGCMC','XNP',NGCTRY,crl=XNP)
       call chmdealloc('mvgcmc.src','MKGCMC','ICVTYP',NGCTRY,intg=ICVTYP)
    ENDIF

    RETURN
  END SUBROUTINE MKGCMC

  SUBROUTINE CVTDEL(ISEED,LCB,ISCVTY,NGCPCP,ICVTYP,NGCTRY, &
       GCCUT2,XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX, &
       ZGCMAX,NGCIN,GCMCON,XN,YN,ZN,X,Y,Z,QGCSPH, &
       GRDBLK,NGRDBK)
    !
    !       Determines the probability that a particle is deleted
    !       by cavity bias as opposed to randomly.
    !
    use number
    use clcg_mod,only: random
    !
    INTEGER ISEED, ICVTYP(*)
    INTEGER ISCVTY(:), NGCPCP(:), NGCTRY, NGCIN
    INTEGER NGRDBK, GRDBLK(*)
    real(chm_real)  GCCUT2, XN(*), YN(*), ZN(*), X(*), Y(*), Z(*)
    real(chm_real)  XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX
    LOGICAL LCB, GCMCON(:), QGCSPH

    INTEGER N, NCAVTY
    real(chm_real)  R

    !       NGCPCP is indexed one higher so zero does not break.
    IF (NGCPCP(NGCIN) .EQ. 0) THEN
       N = NGCIN + 1
       !         If there is no value for ISCVTY at NGCIN-1, use that at NGCIN.
       !         If the latter is not yet defined, compute it directly.
       IF (NGCPCP(N) .EQ. 0) THEN
          CALL GNGCPT(NCAVTY,ICVTYP,ISEED,NGCTRY,GCCUT2, &
               XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
               NGCIN,ISCVTY,NGCPCP,GCMCON,XN,YN,ZN,X,Y,Z, &
               QGCSPH,GRDBLK,NGRDBK)
       ENDIF
       ISCVTY(NGCIN) = ISCVTY(N)
       NGCPCP(NGCIN) = NGCPCP(N)
    ENDIF

    !       Probability of a cavity when NGCIN-1 particles present.
    R = DBLE(ISCVTY(NGCIN))/(NGCTRY*NGCPCP(NGCIN))
    !       Probability of not finding a cavity in NGCTRY tries.
    !       Grid based case is handled in GCBIAS
    R = (ONE-R)**NGCTRY
    !       Probability of     finding a cavity in NGCTRY tries.
    LCB = RANDOM(ISEED) .GE. R
    RETURN
  END SUBROUTINE CVTDEL

  SUBROUTINE GCSTAT(NMVTYP,MVTYPE,NGCTRY,QGCGRD,NIDXGC,NGRDCV, &
       NOCVTY,NGCPCP,ISTEP,NPR,NPSUM,GCSUM,GC2SUM, &
       NGCIN)
    !
    !       Update GCMC statistics.
    !
    use stream

    implicit none
    !
    INTEGER NMVTYP,MVTYPE(*),NGCTRY
    INTEGER NIDXGC(*),NGRDCV
    type(chm_iptr) :: NOCVTY(:), NGCPCP(:)
    INTEGER IMVTYP,ISTEP,NPR,NPSUM(*),NGCIN(*)
    real(chm_real)  GCSUM(*),GC2SUM(*)
    LOGICAL QGCGRD
    !
    INTEGER I, N
    real(chm_real)  A1,A2,SD

    DO I = 1, NMVTYP
       IF (MVTYPE(I).EQ.8) THEN

          IF (NGCTRY .GT. 0 .OR. QGCGRD) THEN
             !             No-cavity statistics for cavity-bias insertion
             CALL NOCVST(NGCIN(I),NGRDCV, NOCVTY(I)%A, &
                  NGCPCP(I)%A)
          ENDIF

          !           General statistics
          NPSUM(I)  = NPSUM(I)  + NGCIN(I)                 ! yd050626
          GCSUM(I)  = GCSUM(I)  + DBLE(NGCIN(I))           ! yd050626
          GC2SUM(I) = GC2SUM(I) + DBLE(NGCIN(I)*NGCIN(I))  ! yd050626

          IF ((PRNLEV.GE.1) .AND. (MOD(ISTEP,NPR).EQ.0)) THEN
             A1 = DBLE(NPSUM(I)) /NPR
             A2 = GCSUM(I) /ISTEP
             SD = GC2SUM(I)/ISTEP
             WRITE (OUTU, 115)
             WRITE (OUTU, 125)
             WRITE (OUTU, 120) ISTEP/NPR,I,NGCIN(I),A1,A2, &
                  SQRT(SD - A2*A2)
             WRITE (OUTU, 125)
115          FORMAT ('MC GCMC:',3X,'Eval#',6X,'INDEX',8X,'NGCIN',8X, &
                  'NGCMC',9X,'NAVE',9X,'NSTD')
120          FORMAT ('MC GCMC>',I6,I13,I13,F13.5,F13.5,F13.5)

125          FORMAT ('--------',8X,2X,'---------',4X,'---------',4X, &
                  '---------',4X,'---------',4X,'---------')

             NPSUM(I) = 0
          ENDIF

       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE GCSTAT

  SUBROUTINE NOCVST(NGC,NGRDCV,NOCVTY,NGCPCP)
    !
    !       Update statistics necessary for running with cavity bias.
    !
    !
    INTEGER NGC,NGRDCV, NOCVTY(:), NGCPCP(:)
    !
    INTEGER N

    N = NGC + 1
    IF (NGRDCV .EQ. 0) NOCVTY(N) = NOCVTY(N) + 1
    NGCPCP(N) = NGCPCP(N) + 1

    RETURN
  END SUBROUTINE NOCVST

  SUBROUTINE CVTINS(XN,YN,ZN,ISEED,NGRDCV,LSTGRD, &
       NGRDX,NGRDY,XGCMIN,YGCMIN,ZGCMIN,GRDSIZ)
    !
    !       Inserts to a randomly chosen grid-based cavity.
    !
    use clcg_mod,only: random

    !
    INTEGER ISEED, NGRDCV, LSTGRD(:), NGRDX, NGRDY
    real(chm_real)  XN, YN, ZN, XGCMIN, YGCMIN, ZGCMIN, GRDSIZ
    !
    INTEGER IX, IY, IZ, IDX
    real(chm_real)  DX, DY, DZ

    IDX = INT(RANDOM(ISEED)*REAL(NGRDCV)) + 1
    CALL GETXYZ(IX,IY,IZ,LSTGRD(IDX),NGRDX,NGRDY)
    DX = GRDSIZ*(IX-1+RANDOM(ISEED))
    DY = GRDSIZ*(IY-1+RANDOM(ISEED))
    DZ = GRDSIZ*(IZ-1+RANDOM(ISEED))
    XN = XGCMIN+DX
    YN = YGCMIN+DY
    ZN = ZGCMIN+DZ
    RETURN
  END SUBROUTINE CVTINS

  SUBROUTINE GETXYZ(IX,IY,IZ,IND,NX,NY)
    !
    !       Convert from lattice to Cartesian grid coordintes.
    !
    !       Aaron R. Dinner
    !
    INTEGER IX,IY,IZ,IND,NX,NY
    !
    INTEGER I1,IA,NA

    I1 = IND - 1
    NA = NX*NY
    IZ = INT((I1)/NA)
    IA = IZ*NA
    IY = INT((I1 - IA)/NX)
    IX = INT( I1 - IA - IY*NX) + 1
    IY = IY + 1
    IZ = IZ + 1

    RETURN
  END SUBROUTINE GETXYZ


  SUBROUTINE GNGCPT(NCAVTY,ICAVTY,ISEED,NGCTRY,GCCUT2, &
       XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
       NGC,ISCVTY,NGCPCP,GCMCON,XN,YN,ZN,X,Y,Z, &
       QGCSPH,GRDBLK,NGRDBK)
    !
    !       Finds NGCTRY random points
    !
    !       Aaron R. Dinner
    !
    use image
    use number
    use psf
    use clcg_mod,only: random
    !
    INTEGER ISEED, NGCTRY, NCAVTY, ICAVTY(*), NGC
    INTEGER NGCPCP(:), ISCVTY(:), GRDBLK(*), NGRDBK
    real(chm_real)  GCCUT2
    real(chm_real)  XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX
    real(chm_real)  XN(*), YN(*), ZN(*)
    real(chm_real)  X(*), Y(*), Z(*)
    LOGICAL GCMCON(:)
    LOGICAL QGCSPH
    !
    INTEGER I, J, K, N
    real(chm_real)  DXGC, DYGC, DZGC, D2
    LOGICAL LCAVTY

    !       Get NGCTRY random points
    DXGC = XGCMAX - XGCMIN
    DYGC = YGCMAX - YGCMIN
    DZGC = ZGCMAX - ZGCMIN

    NCAVTY = 0
    DO I = 1, NGCTRY
131    XN(I) = DXGC*RANDOM(ISEED) + XGCMIN
       YN(I) = DYGC*RANDOM(ISEED) + YGCMIN
       ZN(I) = DZGC*RANDOM(ISEED) + ZGCMIN
       IF (QGCSPH) THEN
          IF (RSPOUT(XN(I),YN(I),ZN(I),XGCMIN,YGCMIN,ZGCMIN, &
               DXGC,DYGC,DZGC)) GOTO 131
       ENDIF

       !         Check if it is a cavity
       LCAVTY = .TRUE.
       DO K = 1, NGRDBK
          J=GRDBLK(K)
          IF (GCMCON(J)) THEN
             D2 = (X(J)-XN(I))**2 + (Y(J)-YN(I))**2 + (Z(J)-ZN(I))**2
             IF (D2 .LT. GCCUT2) THEN
                LCAVTY = .FALSE.
                GOTO 10
             ENDIF
          ENDIF
       ENDDO
10     IF (LCAVTY) THEN
          NCAVTY = NCAVTY + 1
          ICAVTY(NCAVTY) = I
       ENDIF
    ENDDO

    !       Update probability of cavity for N particles
    N = NGC + 1
    ISCVTY(N) = ISCVTY(N) + NCAVTY
    NGCPCP(N) = NGCPCP(N) + 1

    RETURN
  END SUBROUTINE GNGCPT

  SUBROUTINE GCADNB(MCBLO,IMVATM,IMVNG,CUTNB2,GCMCON,X,Y,Z &
#if KEY_PERT==1
       ,QPERT,IPERT,LENV  & 
#endif
       )
    !
    !       Adds atoms to the symmetric non-bonded list.
    !
    !       Aaron R. Dinner
    !

    use image
    use number
    use psf
    use mcmvrtrn, only: mvgcom
    use mcmvutil, only: tagatm

    type(chm_iptr) :: MCBLO(:)
    INTEGER IMVNG(:), IMVATM(:)
    real(chm_real)  X(*), Y(*), Z(*), CUTNB2
    LOGICAL GCMCON(:)
#if KEY_PERT==1
    LOGICAL LENV, QPERT 
#endif
#if KEY_PERT==1
    INTEGER IPERT(:)    
#endif
    !
    INTEGER I, J, K, IAF, IAL, ITP, N, M, NG, N2
    real(chm_real)  XN, YN, ZN, D2

    !       Assumes GCMC group is relatively small
    CALL MVGCOM(XN,YN,ZN,IMVNG,X,Y,Z)
    CALL TAGATM(IMVATM,IMVNG,.TRUE.,1)

    NG = IMVNG(1)
    N2 = IMVNG(NG)
    NG = NG + 2

    DO J = 1, NATOM
       IF (GCMCON(J) .AND. (IMVATM(J) == 0)) THEN
            !         LENV logic to treat the environment and reactant/product
            !         nonbond lists correctly.  GCMC is assumed to be used only
            !         on the enviroment, never on the product or reactant
            !         Y. Deng March, 2006; incorporated and modified by ARD June, 2006
#if KEY_PERT==1
          IF (QPERT) THEN
             IF ((IPERT(J) /= 0) .EQV. LENV) CYCLE
          ENDIF
#endif 
          D2 = (X(J)-XN)**2 + (Y(J)-YN)**2 + (Z(J)-ZN)**2
          IF (D2 .LT. CUTNB2) THEN

             DO K = NG, N2, 2
                IAF = IMVNG(K-1)
                IAL = IMVNG(K)
                DO I=IAF,IAL

                   M = MCBLO(J)%A(1)
                   N = MCBLO(J)%A(2)
                   N = N + 1
                   IF (N.GT.M) CALL GTMRNB(M,MCBLO(J))
                   MCBLO(J)%A(N) = I
                   MCBLO(J)%A(2) = N

                   M = MCBLO(I)%A(1)
                   N = MCBLO(I)%A(2)
                   N = N + 1
                   IF (N.GT.M) CALL GTMRNB(M,MCBLO(I))
                   MCBLO(I)%A(N) = J
                   MCBLO(I)%A(2) = N

                ENDDO
             ENDDO
          ENDIF
       ENDIF
    ENDDO

    CALL TAGATM(IMVATM,IMVNG,.TRUE.,0)

    RETURN
  END SUBROUTINE GCADNB

  SUBROUTINE GTMRNB(NOLD,OLDP)
    !
    !       Increase the size of a symmetric non-bonded list array.
    !
    !       Aaron R. Dinner
    !
    use chm_types
    use memory
    implicit none
    !
    INTEGER NOLD
    type(chm_iptr) :: OLDP
    integer,pointer,dimension(:) :: NEWP
    !
    INTEGER I
    integer,parameter :: NBBUFF = 100

    call chmalloc('mvgcmc.src','GTMRNB','NEWP',NOLD+NBBUFF,intgp=NEWP)
    NEWP(1) = NOLD + NBBUFF
    NEWP(2:NOLD) = OLDP%A(2:NOLD)
    call chmdealloc('mvgcmc.src','GTMRNB','OLDP',NOLD,intgp=OLDP%A)
    OLDP%A => NEWP

    RETURN
  END SUBROUTINE GTMRNB

  SUBROUTINE GCADIM(MCIML,IMVATM,IMVNG,IMATTR,MCATT,CUTNB2,GCMCON, &
       NATOM,NATIM,X,Y,Z &
#if KEY_PERT==1
       ,QPERT,IPERT,LENV  & 
#endif
       )
    !
    !       Get the new coordinates of the image atoms after a move.
    !
    !       This routine assumes IMALL and thus only checks inserted
    !       images with existing primaries to avoid double-counting.
    !
    !       Aaron R. Dinner
    !
    use mcmvutil, only: tagatm

    type(chm_iptr) :: MCIML(:), MCATT(:)
    INTEGER IMVNG(:), IMATTR(:)
    INTEGER NATOM, IMVATM(:), NATIM
    real(chm_real)  X(*), Y(*), Z(*), CUTNB2
    LOGICAL GCMCON(:)
#if KEY_PERT==1
    LOGICAL LENV, QPERT 
#endif
#if KEY_PERT==1
    INTEGER IPERT(:)    
#endif

    INTEGER I, J, K, L, N, NN, P, M, I1, I2, NG
    INTEGER N2
    real(chm_real)  D2, XN, YN, ZN

    !       Tag inserted primary.
    CALL TAGATM(IMVATM,IMVNG,.TRUE.,1)

    !       Inserted image with existing primary
    NG = IMVNG(1)
    N2 = IMVNG(NG)
    NG = NG + 2
    DO I = NG, N2, 2
       I1 = IMVNG(I-1)
       I2 = IMVNG(I)
       DO J = I1, I2
          NN = MCATT(J)%A(2)
          DO K = 3, NN
             L =  MCATT(J)%A(K)
             !
             DO P = 1, NATOM
                IF (GCMCON(P) .AND. (IMVATM(P) == 0)) THEN
#if KEY_PERT==1
                   IF (QPERT) THEN
                     IF ((IPERT(P) /= 0) .EQV. LENV) CYCLE
                   ENDIF
#endif 
                   D2 = (X(L)-X(P))**2+(Y(L)-Y(P))**2+(Z(L)-Z(P))**2
                   IF (D2 .LT. CUTNB2) THEN

                      M = MCIML(P)%A(1)
                      N = MCIML(P)%A(2)
                      N = N + 1
                      IF (N.GT.M) CALL GTMRNB(M,MCIML(P))
                      MCIML(P)%A(N) = L
                      MCIML(P)%A(2) = N

                      M = MCIML(L)%A(1)
                      N = MCIML(L)%A(2)
                      N = N + 1
                      IF (N.GT.M) CALL GTMRNB(M,MCIML(L))
                      MCIML(L)%A(N) = P
                      MCIML(L)%A(2) = N

                   ENDIF
                ENDIF
             ENDDO
             !
          ENDDO
       ENDDO
    ENDDO

    !       Untag inserted primary.
    CALL TAGATM(IMVATM,IMVNG,.TRUE.,0)

    RETURN
  END SUBROUTINE GCADIM

  SUBROUTINE GCRMNB(MCBLO,IMVNG)
    !
    !       Remove non-active atoms from non-bonded list after deletion.
    !
    !       1-4 will break this routine!
    !
    !       Aaron R. Dinner
    !
    use gcmc

    type(chm_iptr) :: MCBLO(:)
    INTEGER IMVNG(:)

    INTEGER I, J, K, N, IAF, IAL
    integer,pointer,dimension(:) :: ITP

    IAF = IMVNG(3)
    IAL = IMVNG(4)
    DO I=IAF,IAL
       IF (.NOT. GCMCON(I)) THEN
          ITP => MCBLO(I)%A
          N = ITP(2)
          DO 334 K = 3, N
             J = ITP(K)
             CALL GCUPNB(MCBLO(J)%A, J)
334       ENDDO
          ITP(2) = 2
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE GCRMNB

  SUBROUTINE GCRMIM(MCIML,IMVNG,MCATT,GCMCON)
    !
    !       Remove non-active atoms from non-bonded list after deletion.
    !
    !       1-4 will break this routine!
    !
    !       Aaron R. Dinner
    !

    type(chm_iptr) :: MCIML(:), MCATT(:)
    INTEGER IMVNG(:)
    LOGICAL GCMCON(:)

    integer,pointer,dimension(:) :: ITP
    INTEGER I, J, K, N, IAF, IAL
    INTEGER NG, N2, I1, I2, M, P, L, NN
    !

    ! Primary
    IAF = IMVNG(3)
    IAL = IMVNG(4)
    DO I=IAF,IAL
       IF (.NOT. GCMCON(I)) THEN
          ITP => MCIML(I)%A
          N = ITP(2)
          DO K = 3, N
             J = ITP(K)
             CALL GCUPNB(MCIML(J)%A, J)
          ENDDO
          ITP(2) = 2
       ENDIF
    ENDDO

    ! Image
    NG = IMVNG(1)
    N2 = IMVNG(NG)
    NG = NG + 2
    DO I = NG, N2, 2
       I1 = IMVNG(I-1)
       I2 = IMVNG(I)
       DO J = I1, I2
          NN = MCATT(J)%A(2)
          DO K = 3, NN
             L = MCATT(J)%A(K)
             !
             IF (.NOT. GCMCON(L)) THEN
                ITP => MCIML(L)%A
                N = ITP(2)
                DO P = 3, N
                   M = ITP(P)
                   CALL GCUPNB(MCIML(M)%A, M)
                ENDDO
                ITP(2) = 2
             ENDIF
             !
          ENDDO
       ENDDO
    ENDDO
    RETURN
  END SUBROUTINE GCRMIM

  SUBROUTINE GCUPNB(ISYMNB,IATOM)
    !
    !       Remove non-active atoms from updated non-bonded list.
    !
    !       Aaron R. Dinner
    !
    use gcmc

    INTEGER ISYMNB(:), IATOM
    !
    INTEGER I, N

    IF (.NOT. GCMCON(IATOM)) THEN
       ISYMNB(2) = 2
       RETURN
    ENDIF

    N = ISYMNB(2)
    I = 3
10  IF (I .LE. N) THEN
       IF (.NOT. GCMCON(ABS(ISYMNB(I)))) THEN
          ISYMNB(I) = ISYMNB(N)
          N = N - 1
       ELSE
          I = I + 1
       ENDIF
       GOTO 10
    ENDIF
    ISYMNB(2) = N

    RETURN
  END SUBROUTINE GCUPNB

  INTEGER FUNCTION FRSIND(IMVNG)
    !
    !       Return the first atom index of a list of moving atoms.
    !

    INTEGER IMVNG(:)
    !
    INTEGER NG

    NG = IMVNG(1)
    FRSIND = IMVNG(NG + 1)

    RETURN
  END FUNCTION FRSIND

  SUBROUTINE DELMVG(NGRDCV,LSTGRD,LSTIND,NGCBLK,GRDSIZ,NRGRID, &
       NGRDX,NGRDY,NGRDZ,XGCMIN,YGCMIN,ZGCMIN,IMVNG, &
       CRDOLD)
    !
    !       Delete one particle and update the LSTGRD array.
    !
    use gcmc

    !       Passed variables
    !
    INTEGER NGRDCV, NGCBLK(*), IMVNG(:), LSTGRD(*),LSTIND(*)
    INTEGER NRGRID,NGRDX,NGRDY,NGRDZ
    real(chm_real)  XGCMIN,YGCMIN,ZGCMIN,GRDSIZ
    real(chm_real)  CRDOLD(*)
    !
    !       Local variables
    !
    INTEGER I,J,K,N,NG,IAF,IAL
    real(chm_real)  XDEL,YDEL,ZDEL

    NG = IMVNG(1)
    N  = IMVNG(NG)
    NG = NG + 2
    J = 0
    DO K = NG, N, 2
       IAF = IMVNG(K-1)
       IAL = IMVNG(K)
       DO I = IAF, IAL
          J = J + 1
          XDEL = CRDOLD(J)
          J = J + 1
          YDEL = CRDOLD(J)
          J = J + 1
          ZDEL = CRDOLD(J)
          IF (GCBLKR(I)) THEN
             CALL GRDDEL(NGRDCV,LSTGRD,LSTIND,NGCBLK,GRDSIZ,NRGRID, &
                  NGRDX,NGRDY,NGRDZ,XGCMIN,YGCMIN,ZGCMIN, &
                  XDEL,YDEL,ZDEL)
          ENDIF
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE DELMVG


  SUBROUTINE GRDDEL(NGRDCV,LSTGRD,LSTIND,NGCBLK,GRDSIZ,NRGRID, &
       NGRDX,NGRDY,NGRDZ,XGCMIN,YGCMIN,ZGCMIN, &
       XDEL,YDEL,ZDEL)
    !
    !       Delete one particle and update the lstgrd array.
    !
    !
    !       Passed variables
    !
    INTEGER NGRDCV, NGCBLK(*),LSTGRD(*),LSTIND(*)
    INTEGER NRGRID,NGRDX,NGRDY,NGRDZ
    real(chm_real)  XGCMIN,YGCMIN,ZGCMIN,GRDSIZ,XDEL,YDEL,ZDEL
    !
    !       Local variables
    !
    INTEGER NX,NY,NZ,NGX,NGY,NGZ,NDIST,NGRID2,IND

    NGRID2 = NRGRID*NRGRID

    !       Lattice coordinate of the chosen molecule
    NGX=INT((XDEL-XGCMIN)/GRDSIZ+0.01)+1
    NGY=INT((YDEL-YGCMIN)/GRDSIZ+0.01)+1
    NGZ=INT((ZDEL-ZGCMIN)/GRDSIZ+0.01)+1

    !       Go over the cube around the center with length 2*nrgrid
    DO NX=-NRGRID+NGX,NRGRID+NGX
       DO NY=-NRGRID+NGY,NRGRID+NGY
          DO NZ=-NRGRID+NGZ,NRGRID+NGZ

             IF (INGRID(NX,NY,NZ,NGRDX,NGRDY,NGRDZ)) THEN
                NDIST = (NX-NGX)**2+(NY-NGY)**2+(NZ-NGZ)**2
                IF (NDIST.LT.NGRID2) THEN
                   IND = (NZ-1)*NGRDX*NGRDY+(NY-1)*NGRDX+NX
                   IF (NGCBLK(IND) .GT. 0) THEN
                      NGCBLK(IND) = NGCBLK(IND) - 1
                      IF (NGCBLK(IND) .EQ. 0) THEN
                         NGRDCV = NGRDCV + 1
                         LSTGRD(NGRDCV) = IND
                         LSTIND(IND) = NGRDCV
                      ENDIF
                   ELSE IF (NGCBLK(IND).EQ.0) THEN
                      CALL WRNDIE(-5,'<GRDDEL>','Grid error')
                   ENDIF
                ENDIF
             ENDIF
          ENDDO
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE GRDDEL

  SUBROUTINE INSMVG(NGRDCV,NGCBLK,LSTGRD,LSTIND,IMVNG, &
       XGCMIN,YGCMIN,ZGCMIN,GRDSIZ,NRGRID, &
       NGRDX,NGRDY,NGRDZ,X,Y,Z)
    !
    !       Loops over the moving atoms and calls GRDINS for all blockers.
    !
    use gcmc
    !
    !       Passed variables
    !
    INTEGER IMVNG(:), NGRDCV
    INTEGER NGCBLK(*),LSTGRD(*),LSTIND(*)
    INTEGER NRGRID,NGRDX,NGRDY,NGRDZ
    real(chm_real)  XGCMIN,YGCMIN,ZGCMIN,GRDSIZ
    real(chm_real)  X(*),Y(*),Z(*)
    !
    !       Local variables
    !
    INTEGER I,K,N,NG,IAF,IAL,NB

    NG = IMVNG(1)
    N  = IMVNG(NG)
    NG = NG + 2
    DO K = NG, N, 2
       IAF = IMVNG(K-1)
       IAL = IMVNG(K)
       DO I = IAF, IAL
          IF (GCBLKR(I)) THEN
             CALL GRDINS(NGRDCV,NGCBLK,LSTGRD,LSTIND, &
                  XGCMIN,YGCMIN,ZGCMIN,GRDSIZ,NRGRID, &
                  NGRDX,NGRDY,NGRDZ,X(I),Y(I),Z(I))
          ENDIF
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE INSMVG

  SUBROUTINE GRDINS(NGRDCV,NGCBLK,LSTGRD,LSTIND, &
       XGCMIN,YGCMIN,ZGCMIN,GRDSIZ,NRGRID, &
       NGRDX,NGRDY,NGRDZ,XCM,YCM,ZCM)
    !
    !       Inserts one particle in the grid and updates the lstgrid array
    !
    !
    !       Passed variables
    !
    INTEGER NGRDCV, NGCBLK(*)
    INTEGER LSTIND(*), LSTGRD(*)
    INTEGER NRGRID,NGRDX,NGRDY,NGRDZ
    real(chm_real)  XGCMIN,YGCMIN,ZGCMIN,GRDSIZ
    real(chm_real)  XCM,YCM,ZCM
    !
    !       Local variables
    !
    INTEGER NGX,NGY,NGZ,NX,NY,NZ,NDIST,NGRID2,N
    INTEGER LGX,LGY,LGZ,NLST,LGX2,LGY2,LGZ2,NLAST,IFRD

    NGRID2 = NRGRID*NRGRID

    !       Update the spin variables
    NGX=INT((XCM-XGCMIN)/GRDSIZ+0.01)+1
    NGY=INT((YCM-YGCMIN)/GRDSIZ+0.01)+1
    NGZ=INT((ZCM-ZGCMIN)/GRDSIZ+0.01)+1

    DO LGX=-NRGRID+NGX,NRGRID+NGX
       DO LGY=-NRGRID+NGY,NRGRID+NGY
          DO LGZ=-NRGRID+NGZ,NRGRID+NGZ
             IF (INGRID(LGX,LGY,LGZ,NGRDX,NGRDY,NGRDZ)) THEN
                NDIST=(LGX-NGX)**2+(LGY-NGY)**2+(LGZ-NGZ)**2
                IF (NDIST.LT.NGRID2) THEN
                   IFRD=(LGZ-1)*NGRDX*NGRDY+(LGY-1)*NGRDX+LGX
                   IF (NGCBLK(IFRD).EQ.0) THEN
                      !                   If the site was empty, remove it from the cavity list
                      LSTIND(LSTGRD(NGRDCV)) = LSTIND(IFRD)
                      LSTGRD(LSTIND(IFRD)) = LSTGRD(NGRDCV)
                      NGRDCV = NGRDCV - 1
                      NGCBLK(IFRD) = NGCBLK(IFRD) + 1
                   ELSE IF (NGCBLK(IFRD).GE.0) THEN
                      !                   Otherwise, simply increment
                      NGCBLK(IFRD) = NGCBLK(IFRD) + 1
                   ENDIF
                ENDIF
             ENDIF
          ENDDO
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE GRDINS

  SUBROUTINE NOGCMV(NGCBLK,LSTGRD,LSTIND,NGRDCV, &
       IMVNG,IDX,NGRDX,NGRDY,NGRDZ, &
       NRGRID,GRDSIZ,XGCMIN,YGCMIN,ZGCMIN,CRDOLD,X,Y,Z)
    !
    !       Updates NGCBLK after a non-GCMC move.
    !
    !
    !       Passed variables
    !
    type(chm_iptr) :: IMVNG(:)
    INTEGER IDX
    INTEGER NGCBLK(*),LSTGRD(*),LSTIND(*),NGRDCV
    INTEGER NGRDX,NGRDY,NGRDZ,NRGRID
    real(chm_real)  GRDSIZ,XGCMIN,YGCMIN,ZGCMIN
    real(chm_real)  X(*), Y(*), Z(*), CRDOLD(*)

    !       Remove the blocking atoms from their old places.
    CALL DELMVG(NGRDCV,LSTGRD,LSTIND,NGCBLK,GRDSIZ,NRGRID, &
         NGRDX,NGRDY,NGRDZ,XGCMIN,YGCMIN,ZGCMIN, &
         IMVNG(IDX)%A, CRDOLD)

    !       Now insert the blocking atoms in their new places.
    CALL INSMVG(NGRDCV,NGCBLK,LSTGRD,LSTIND, IMVNG(IDX)%A, &
         XGCMIN,YGCMIN,ZGCMIN,GRDSIZ,NRGRID, &
         NGRDX,NGRDY,NGRDZ,X,Y,Z)

    RETURN
  END SUBROUTINE NOGCMV

  LOGICAL FUNCTION INGRID(NX,NY,NZ,NGRDX,NGRDY,NGRDZ)
    !
    !       Check if point NX, NY, NZ is in the grid bounds.
    !

    INTEGER NX,NY,NZ,NGRDX,NGRDY,NGRDZ
    !
    INGRID = NX.GE.1 .AND. NX.LE.NGRDX .AND. &
         NY.GE.1 .AND. NY.LE.NGRDY .AND. &
         NZ.GE.1 .AND. NZ.LE.NGRDZ
    RETURN
  END FUNCTION INGRID

  LOGICAL FUNCTION ISPOUT(NX,NY,NZ,NGRDX,NGRDY,NGRDZ)
    !
    !       Check if point NX, NY, NZ is in the GC insertion sphere.
    !

    INTEGER NX,NY,NZ,NGRDX,NGRDY,NGRDZ
    !
    INTEGER NDIST

    NDIST=(NX-NGRDX/2)**2+(NY-NGRDY/2)**2 +(NZ-NGRDZ/2)**2
    ISPOUT = NDIST.GT.NGRDX*NGRDX/4
    RETURN
  END FUNCTION ISPOUT

  LOGICAL FUNCTION RSPOUT(XN,YN,ZN,XGC,YGC,ZGC,DX,DY,DZ)
    !
    !       Check if point XN, YN, ZN is in the GC insertion sphere.
    !

    real(chm_real) XN,YN,ZN,XGC,YGC,ZGC,DX,DY,DZ
    !
    real(chm_real) D

    D=(XN-XGC-0.5*DX)**2+(YN-YGC-0.5*DY)**2+(ZN-ZGC-0.5*DZ)**2
    RSPOUT = D.GT.DX*DX/4.0
    RETURN
  END FUNCTION RSPOUT

  SUBROUTINE TOGLGC(GCMCON,IMVNG)

    use clcg_mod,only:random
    INTEGER IMVNG(:)
    LOGICAL GCMCON(:)
    !
    INTEGER I, K, N, NG, IAF, IAL

    NG = IMVNG(1)
    N =  IMVNG(NG)
    NG = NG + 2
    DO K = NG, N, 2
       IAF = IMVNG(K-1)
       IAL = IMVNG(K)
       DO I = IAF, IAL
          GCMCON(I) = .NOT. GCMCON(I)
       ENDDO
    ENDDO
    RETURN
  END SUBROUTINE TOGLGC

  FUNCTION GCBIAS(ISEED,GCMCBF,IMVNG,IDX, &
       LGCOLD,LGCCB,QGCGRD, &
       NGCTRY,NCFB,PCFB,NGCIN,NOCVTY,NGCPCP, &
       ISCVTY,NGCBLK,NGRDCV,NGDTOT,NRGRID,GRDSIZ, &
       NGRDX,NGRDY,NGRDZ,XGCMIN,YGCMIN,ZGCMIN, &
       LSTGRD,LSTIND,CRDOLD,X,Y,Z) result(gcbias_result)
    !
    !       Determines the appropriate bias corrections.
    !
    !       Aaron R. Dinner
    !
    use clcg_mod,only:random
    use number
    !
    real(chm_real) :: gcbias_result
    type(chm_iptr) :: IMVNG(:)
    INTEGER ISEED, IDX, NGCTRY, NCFB
    INTEGER NGCIN, NRGRID, NGRDX, NGRDY, NGRDZ, NGRDCV, NGDTOT
    INTEGER NOCVTY(:), NGCPCP(:), ISCVTY(:)
    integer,dimension(:) :: NGCBLK, LSTGRD, LSTIND
    real(chm_real)  GCMCBF, XGCMIN, YGCMIN, ZGCMIN, GRDSIZ, PCFB
    real(chm_real)  X(*), Y(*), Z(*), CRDOLD(*)
    LOGICAL LGCOLD, QGCGRD, LGCCB
    !
    INTEGER NOCV
    real(chm_real)  RS, XDEL, YDEL, ZDEL

    IF (LGCOLD) THEN

       !         Deletion
       GCBIAS_RESULT = GCMCBF - LOG(DBLE(NGCIN))

       IF (QGCGRD) THEN

          !           Update the grid list to get current cavities
          CALL DELMVG(NGRDCV, LSTGRD, LSTIND, NGCBLK, &
               GRDSIZ,NRGRID,NGRDX,NGRDY,NGRDZ,XGCMIN,YGCMIN, &
               ZGCMIN, IMVNG(IDX)%A, CRDOLD)

          NOCV = NOCVTY(NGCIN)
          IF (NOCV.GT.0) THEN
             !             "No-cavity" statistics available
             RS = DBLE(NOCV) / NGCPCP(NGCIN)
             LGCCB = RANDOM(ISEED).GE.RS
          ELSE
             LGCCB=.TRUE.
          ENDIF
          IF (LGCCB) THEN
             !             NGRDCV has been updted in DELMVG
             RS = DBLE(NGRDCV)/NGDTOT
             GCBIAS_RESULT = GCBIAS_RESULT + LOG(RS)
          ENDIF

       ELSE IF (NGCTRY .GT. 0) THEN
          !           Simple cavity-bias -> use average P_N statistics
          IF (LGCCB) THEN
             RS = DBLE(ISCVTY(NGCIN)) / &
                  DBLE(NGCPCP(NGCIN)*NGCTRY)
             GCBIAS_RESULT = GCBIAS_RESULT + LOG(RS)
          ENDIF
       ENDIF

       !         Orientational bias
       IF (NCFB .GT. 1) THEN
          RS = ONE/(PCFB*NCFB)
          GCBIAS_RESULT = GCBIAS_RESULT + LOG(RS)
       ENDIF

    ELSE

       !         Insertion
       GCBIAS_RESULT = -GCMCBF + LOG(DBLE(NGCIN+1))

       IF (QGCGRD) THEN
          IF (NGRDCV.GT.0) THEN
             RS = DBLE(NGRDCV)/NGDTOT
             GCBIAS_RESULT = GCBIAS_RESULT - LOG(RS)
          ENDIF
       ELSE IF (NGCTRY .GT. 0) THEN
          IF (LGCCB) THEN
             RS = DBLE(ISCVTY(NGCIN+1)) / &
                  DBLE(NGCPCP(NGCIN+1)*NGCTRY)
             GCBIAS_RESULT = GCBIAS_RESULT - LOG(RS)
          ENDIF
       ENDIF

       !         Orientational bias
       IF (NCFB .GT. 1) THEN
          RS = 1.0/(PCFB*DBLE(NCFB))
          GCBIAS_RESULT = GCBIAS_RESULT - LOG(RS)
       ENDIF

    ENDIF

    RETURN
  END FUNCTION GCBIAS

  SUBROUTINE ACGCMC(MCBLOP,IMVNG,NTRANS,MCIMLP,MCATTP,IDX,GCMCON, &
       LGCOLD,NMVATM,NIDXGC,IDXGC,QGCGRD,NGRDX,NGRDY, &
       NGRDZ,NGCBLK,LSTGRD,LSTIND,NGRDCV,NRGRID,GRDSIZ, &
       XGCMIN,YGCMIN,ZGCMIN,NGCIN,X,Y,Z &
#if KEY_PERT==1
       ,MCBLORP,MCBLOPP,QPERT            & 
#endif
       )
    !
    !       Accept a grand canonical move.
    !
    !
    type(chm_iptr) :: IMVNG(:)
    type(chm_iptr),dimension(:) :: MCBLOP, MCIMLP, MCATTP
    INTEGER NTRANS, IDX
    INTEGER NMVATM, NIDXGC, IDXGC(*), NGRDX, NGRDY, NGRDZ, NGRDCV
    INTEGER NRGRID, NGCIN
    integer,allocatable,dimension(:) :: NGCBLK, LSTGRD, LSTIND
#if KEY_PERT==1
    type(chm_iptr),dimension(:) :: MCBLORP, MCBLOPP  
#endif
    real(chm_real)  XGCMIN, YGCMIN, ZGCMIN, GRDSIZ, X(*), Y(*), Z(*)
    LOGICAL GCMCON(:), LGCOLD, QGCGRD
#if KEY_PERT==1
    LOGICAL QPERT                               
#endif
    !
    INTEGER I, K, IDFRD, NGX, NGY, NGZ, SOLD, ISWP

    IF (LGCOLD) THEN
       !         It is a deletion.

       !         Swap in last
       !         Find one to swap (need to fix this code)
       DO K = 1, NIDXGC
          IF (IDXGC(K) .EQ. IDX) ISWP = K
       ENDDO
       IDXGC(ISWP) =   IDXGC(NIDXGC)
       IDXGC(NIDXGC) = IDX
       NIDXGC = NIDXGC - 1
       CALL GCRMNB(MCBLOP, IMVNG(IDX)%A)
#if KEY_PERT==1
       IF (QPERT) THEN
          CALL GCRMNB(MCBLORP, IMVNG(IDX)%A)
          CALL GCRMNB(MCBLOPP, IMVNG(IDX)%A)
       ENDIF
#endif 
       IF (NTRANS .GT. 0) CALL GCRMIM(MCIMLP, IMVNG(IDX)%A, &
            MCATTP, GCMCON)

       !         No need to update grid since it is done in GCBIAS

       !         Deletions are only allowed within the volume.
       !         Update number active in the volume accordingly.
       NGCIN = NGCIN - 1

    ELSE
       !         It is an insertion.

       !         Find the ghost to swap
       NIDXGC = NIDXGC + 1
       DO K = NMVATM,NIDXGC,-1
          IF (IDXGC(K) .EQ. IDX) ISWP = K
       ENDDO

       !         Swap with the last
       IDXGC(ISWP)   = IDXGC(NIDXGC)
       IDXGC(NIDXGC) = IDX

       !         Update the grid
       IF (QGCGRD) THEN
          CALL INSMVG(NGRDCV, NGCBLK, LSTGRD, LSTIND, &
               IMVNG(IDX)%A, XGCMIN,YGCMIN,ZGCMIN,GRDSIZ, &
               NRGRID,NGRDX,NGRDY,NGRDZ,X,Y,Z)
       ENDIF

       !         Insertions are only allowed within the volume.
       !         Update number active in the volume accordingly.
       NGCIN = NGCIN + 1

    ENDIF

    RETURN
  END SUBROUTINE ACGCMC

  SUBROUTINE RJGCMC(GCMCON,IDX,IMVNG,ILNMV,LGCOLD,MCBLOP, &
       NTRANS,MCIMLP,MCATTP,QGCGRD,NGRDX,NGRDY,NGRDZ, &
       NGCBLK,LSTGRD,LSTIND,NGRDCV,NRGRID,GRDSIZ, &
       XGCMIN,YGCMIN,ZGCMIN,X,Y,Z &
#if KEY_PERT==1
       ,MCBLORP,MCBLOPP,QPERT               & 
#endif
       )
    !
    !       Reject a grand canonical move.
    !
    type(chm_iptr) :: IMVNG(:), ILNMV(:)
    INTEGER IDX, NTRANS
    type(chm_iptr),dimension(:) :: MCBLOP, MCIMLP, MCATTP
    INTEGER NGRDX, NGRDY, NGRDZ, NGRDCV, NRGRID
    integer,allocatable,dimension(:) :: NGCBLK, LSTGRD, LSTIND
#if KEY_PERT==1
    type(chm_iptr),dimension(:) :: MCBLORP, MCBLOPP  
#endif
    real(chm_real)  XGCMIN, YGCMIN, ZGCMIN, GRDSIZ, X(*), Y(*), Z(*)
    LOGICAL GCMCON(:), LGCOLD, QGCGRD
#if KEY_PERT==1
    LOGICAL QPERT                                    
#endif
    !
    INTEGER I, J, K, N, NG, IAF, IAL

    CALL TOGLGC(GCMCON, IMVNG(IDX)%A)
    !
    IF (LGCOLD) THEN
       !         Update the grid to undo deletion in GCBIAS
       IF (QGCGRD) THEN
          CALL INSMVG(NGRDCV, NGCBLK, LSTGRD, LSTIND, &
               IMVNG(IDX)%A, XGCMIN,YGCMIN,ZGCMIN,GRDSIZ, &
               NRGRID,NGRDX,NGRDY,NGRDZ,X,Y,Z)
       ENDIF
    ELSE
       CALL GCRMNB(MCBLOP, ILNMV(IDX)%A)
#if KEY_PERT==1
       IF (QPERT) THEN
          CALL GCRMNB(MCBLORP, ILNMV(IDX)%A)
          CALL GCRMNB(MCBLOPP, ILNMV(IDX)%A)
       ENDIF
#endif 
       IF (NTRANS .GT. 0) THEN
          CALL GCRMIM(MCIMLP, ILNMV(IDX)%A, MCATTP, &
               GCMCON)
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE RJGCMC

  SUBROUTINE GCUPIM(GCMCON,MCATT,NATOM)
    !
    !       Update the GC flags after an image update.
    !
    !       Aaron R. Dinner
    !
    use chm_types

    implicit none

    type(chm_iptr) :: MCATT(:)
    INTEGER NATOM
    LOGICAL GCMCON(:)

    INTEGER J, K, L, NN

    DO J = 1, NATOM
       NN = MCATT(J)%A(2)
       DO K = 3, NN
          L =  MCATT(J)%A(K)
          GCMCON(L) = GCMCON(J)
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE GCUPIM

  FUNCTION DENS2B(QGCSPH,QGCGRD, &
       NGRDX,NGRDY,NGRDZ,GRDSIZ, &
       XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
       TKELV,MUEX,DENS,RADIUS) result(dens2b_result)
    !
    !       Calculate the GCMC B from the density.
    !
    use number
    use consta

    real(chm_real) :: dens2b_result
    INTEGER NGRDX, NGRDY, NGRDZ
    real(chm_real)  XGCMIN, YGCMIN, ZGCMIN, XGCMAX, YGCMAX, ZGCMAX
    real(chm_real)  TKELV, MUEX, DENS, GRDSIZ
    LOGICAL QGCSPH, QGCGRD
    !
    INTEGER NGDTOT, I, J, K
    real(chm_real)  RADIUS, VOL, BAR

    !       Calculate the insertion volume and then B
    IF (QGCSPH) THEN
       IF (QGCGRD) THEN
          NGDTOT = 0
          DO I = 1, NGRDX
             DO J = 1, NGRDY
                DO K = 1, NGRDZ
                   IF (.NOT. ISPOUT(I,J,K,NGRDX,NGRDY,NGRDZ)) &
                        NGDTOT = NGDTOT + 1
                ENDDO
             ENDDO
          ENDDO
          VOL=NGDTOT*GRDSIZ*GRDSIZ*GRDSIZ
       ELSE
          VOL=(FOUR/THREE)*PI*RADIUS**3
       ENDIF
    ELSE
       VOL=(XGCMAX-XGCMIN)*(YGCMAX-YGCMIN)*(ZGCMAX-ZGCMIN)
    ENDIF
    BAR=VOL*DENS
    DENS2B_RESULT=MUEX/(KBOLTZ*TKELV) + LOG(BAR)

    RETURN
  END FUNCTION DENS2B

  SUBROUTINE GCMCIN(LGCMC,NMVTYP,MVTYPE,QGCGRD,GCMCON,CUTNB2,CUTNB, &
       XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX,NATOM, &
       NATIM,NTRANS,LIMALL,IMATTR,NPSUM,GCSUM,GC2SUM, &
       NGCTRY, GCBLKR, NGRDBK, &
       NGRDX,NGRDY,NGRDZ)
    !
    !       Initializations and checks for GC in MCLOOP.
    !
    use memory
    use number
    !
    INTEGER NMVTYP, MVTYPE(*), NPSUM(*)
    INTEGER NTRANS, IMATTR(*), NATOM, NATIM
    INTEGER NGCTRY, NGRDBK
    INTEGER NGRDX,NGRDY,NGRDZ
    real(chm_real)  GCSUM(*),GC2SUM(*)
    real(chm_real)  XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX
    real(chm_real)  CUTNB2,CUTNB
    LOGICAL LGCMC, GCMCON(:), LIMALL, QGCGRD, GCBLKR(:)
    !
    INTEGER I, NGRNO
    !
    DO I = 1, NMVTYP
       IF (MVTYPE(I).EQ.8) THEN
          LGCMC = .TRUE.
       ENDIF
       NPSUM(I)  = 0
       GCSUM(I)  = ZERO
       GC2SUM(I) = ZERO
    ENDDO
    IF (.NOT. LGCMC) RETURN
    !
    IF (XGCMAX.LE.XGCMIN.OR.YGCMAX.LE.YGCMIN.OR.ZGCMAX.LE.ZGCMIN) &
         CALL WRNDIE(-2,'<MCLOOP>','VOLUME FOR INSERTION IS ZERO')
    IF (NTRANS .GT. 0 .AND. .NOT. LIMALL) &
         CALL WRNDIE(-1,'<MCLOOP>','IMALL recommended for GCMC')

    CUTNB2 = CUTNB*CUTNB
    !       Copy GCMCON for primary to that for image
    DO I = NATOM+1,NATIM
       GCMCON(I) = GCMCON(IMATTR(I))
    ENDDO

    !       If necessary, setup the grid
    IF (QGCGRD .OR. NGCTRY.GT.0) THEN
       !         Count how many blockers.
       NGRDBK = 0
       DO I = 1, NATOM
          IF (GCBLKR(I)) NGRDBK = NGRDBK + 1
       ENDDO
       !         Allocate space to be filled in MCUPDT by GCMCUP.
       call chmalloc('mvgcmc.src','GCMCIN','GRDBLK',NGRDBK,intg=GRDBLK)
    ENDIF
    IF (QGCGRD) THEN
       NGRNO = NGRDX*NGRDY*NGRDZ
       !         Allocate space to be filled in MCUPDT by GCMCUP.
       call chmalloc('mvgcmc.src','GCMCIN','NGCBLK',NGRNO,intg=NGCBLK)
       call chmalloc('mvgcmc.src','GCMCIN','LSTGRD',NGRNO,intg=LSTGRD)
       call chmalloc('mvgcmc.src','GCMCIN','LSTIND',NGRNO,intg=LSTIND)
    ENDIF

    RETURN
  END SUBROUTINE GCMCIN

  SUBROUTINE GCMCUP(NMVTYP,MVTYPE,QGCGRD,NRGRID,GCMCON, &
       XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX, &
       NATOM,NATIM,NTRANS,LIMALL,IMATTR,NPSUM,GCSUM, &
       GC2SUM,NGCIN,NIDXGC,QGCSPH,IDXGCP,IMVNGP,NGCTRY, &
       GCBLKR,NGCBLK,NGRDBK,LSTGRD,LSTIND,NGDTOT, &
       NGRDCV,GRDBLK,GRDSIZ,NGRDX,NGRDY,NGRDZ,X,Y,Z)
    !
    !       Update quantities calculated on the fly in GCMC after image updates.
    !

    use number
    !
    INTEGER NGCIN(*), NIDXGC(*)
    type(chm_iptr) :: IDXGCP(:)
    type(iptr_ptr) :: IMVNGP(:)
    INTEGER NMVTYP, MVTYPE(*), NPSUM(*)
    INTEGER NRGRID, NTRANS, IMATTR(*), NATOM, NATIM
    INTEGER NGCTRY, NGRDBK, NGDTOT
    integer,dimension(:) :: NGCBLK, LSTGRD, LSTIND, GRDBLK
    INTEGER NGRDCV, NGRDX,NGRDY,NGRDZ
    real(chm_real)  X(*),Y(*),Z(*)
    real(chm_real)  GCSUM(*),GC2SUM(*)
    real(chm_real)  XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX
    real(chm_real)  GRDSIZ
    LOGICAL LGCMC, GCMCON(:), LIMALL, QGCSPH, QGCGRD
    LOGICAL GCBLKR(:)
    !
    INTEGER I
    !
    DO I = 1, NMVTYP
       IF (MVTYPE(I).EQ.8) THEN
          NGCIN(I) = NININI(NIDXGC(I),QGCSPH, IDXGCP(I)%A, &
               IMVNGP(I)%A, XGCMIN,YGCMIN,ZGCMIN, &
               XGCMAX,YGCMAX,ZGCMAX,X,Y,Z)
       ELSE
          NGCIN(I) = 0
       ENDIF
    ENDDO

    IF (QGCGRD .OR. NGCTRY.GT.0) THEN
       CALL GENBLK(GRDBLK, NGRDBK,GCBLKR,NATOM)
    ENDIF
    IF (QGCGRD) THEN
       CALL GENGRD(NGCBLK, LSTGRD, LSTIND, NGDTOT, &
            NGRDCV,GCMCON, GRDBLK, NGRDBK,QGCSPH,NRGRID, &
            GRDSIZ,NGRDX,NGRDY,NGRDZ,XGCMIN,YGCMIN,ZGCMIN, &
            X,Y,Z)
    ENDIF

    RETURN
  END SUBROUTINE GCMCUP

  SUBROUTINE GENBLK(GRDBLK,NGRDBK,GCBLKR,NATOM)
    !
    !       Setup the grid for grand canonical simulations.
    !
    INTEGER NGRDBK, GRDBLK(*), NATOM
    LOGICAL GCBLKR(:)

    INTEGER I
    NGRDBK = 0
    DO I=1,NATOM

       IF (GCBLKR(I)) THEN
          NGRDBK = NGRDBK + 1
          GRDBLK(NGRDBK) = I
       ENDIF

    ENDDO

    IF (NGRDBK.EQ.0) THEN
       CALL WRNDIE(-2,'<GENBLK>','GRID/CBIAS BUT NO GCBLOCKERS')
    ENDIF

    RETURN
  END SUBROUTINE GENBLK

  SUBROUTINE GCTEST(LIMVGC,LGCOLD,LGCIN,GCMCON,MVTYPE,IMVNGP, &
       IGCMVG,IDX,QGCSPH,XGCMIN,YGCMIN,ZGCMIN, &
       XGCMAX,YGCMAX,ZGCMAX,X,Y,Z)
    !
    !       Check if a move is a GCMC one.  If it is, determine
    !       whether it is an insertion or deletion.  Otherwise,
    !       check if it is in the volume.
    !
    type(chm_iptr) :: IMVNGP(:)
    INTEGER MVTYPE, IGCMVG, IDX
    real(chm_real)  X(*),Y(*),Z(*)
    real(chm_real)  XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX
    LOGICAL LIMVGC,LGCOLD,LGCIN,GCMCON(:),QGCSPH
    !
    INTEGER I

    !       Linking will break this code I think
    LIMVGC = MVTYPE .EQ. 8
    IF (LIMVGC) LGCOLD = GCMCON(FRSIND(IMVNGP(IDX)%A))

    IF ((IGCMVG .GT. 0) .OR. (LIMVGC .AND. LGCOLD)) THEN
       LGCIN  = GCINF(QGCSPH, IMVNGP(IDX)%A, XGCMIN,YGCMIN,ZGCMIN, &
            XGCMAX,YGCMAX,ZGCMAX,X,Y,Z)
    ENDIF

    RETURN
  END SUBROUTINE GCTEST

  INTEGER FUNCTION NININI(NGC,QGCSPH,IDXGC,IMVNGP,XGCMIN,YGCMIN, &
       ZGCMIN,XGCMAX,YGCMAX,ZGCMAX,X,Y,Z)
    !
    !       Compute total number of GC molecules in the volume.
    !
    type(chm_iptr) :: IMVNGP(:)
    INTEGER NGC, IDXGC(*)
    real(chm_real)  X(*),Y(*),Z(*)
    real(chm_real)  XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX
    LOGICAL QGCSPH
    !
    INTEGER I,IDX

    NININI = 0
    DO I = 1, NGC
       IDX = IDXGC(I)
       IF (GCINF(QGCSPH, IMVNGP(IDX)%A, XGCMIN,YGCMIN,ZGCMIN, &
            XGCMAX,YGCMAX,ZGCMAX,X,Y,Z)) NININI = NININI + 1
    ENDDO

    RETURN
  END FUNCTION NININI

  LOGICAL FUNCTION GCINF(QGCSPH,IMVNG,XGCMIN,YGCMIN,ZGCMIN, &
       XGCMAX,YGCMAX,ZGCMAX,X,Y,Z)
    !
    !       Check if a GC molecule is in the volume.
    !
    use mcmvrtrn, only: mvgcom

    INTEGER IMVNG(:)
    real(chm_real)  X(*),Y(*),Z(*)
    real(chm_real)  XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX
    LOGICAL QGCSPH
    !
    real(chm_real)  XCM,YCM,ZCM,DX,DY,DZ

    CALL MVGCOM(XCM,YCM,ZCM,IMVNG,X,Y,Z)
    IF (QGCSPH) THEN
       DX = XGCMAX - XGCMIN
       DY = YGCMAX - YGCMIN
       DZ = ZGCMAX - ZGCMIN
       GCINF = .NOT.RSPOUT(XCM,YCM,ZCM,XGCMIN,YGCMIN,ZGCMIN,DX,DY,DZ)
    ELSE
       GCINF = XCM.GT.XGCMIN .AND. XCM.LT.XGCMAX .AND. &
            YCM.GT.YGCMIN .AND. YCM.LT.YGCMAX .AND.  &
            ZCM.GT.ZGCMIN .AND. ZCM.LT.ZGCMAX
    ENDIF

    RETURN
  END FUNCTION GCINF

  SUBROUTINE NOGCIN(NGCIN,LGCIN,GCMCON,IMVNGP,IDX,QGCSPH, &
       XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX,X,Y,Z)
    !
    !       Update number of GC molecules in the volume for a non-GC move.
    !
    !       Moves that translate more than one GC molecule will break this
    !       routine.
    !
    !       If very long simulations are run, then NGCIN should be periodically
    !       updated with NININI.
    !
    type(chm_iptr) :: IMVNGP(:)
    INTEGER NGCIN, IDX
    real(chm_real)  X(*),Y(*),Z(*)
    real(chm_real)  XGCMIN,YGCMIN,ZGCMIN,XGCMAX,YGCMAX,ZGCMAX
    LOGICAL QGCSPH,LGCIN,GCMCON(:)
    !
    LOGICAL LINNEW

    !       If the user is wasting time moving inactive atoms, return
    !       without taking any action
    IF (.NOT. GCMCON(FRSIND(IMVNGP(IDX)%A))) RETURN

    !       Get new occupancy
    LINNEW = GCINF(QGCSPH, IMVNGP(IDX)%A, XGCMIN,YGCMIN,ZGCMIN, &
         XGCMAX,YGCMAX,ZGCMAX,X,Y,Z)

    !       If LGCIN is not equal to LINNEW, update NGCIN
    IF (LGCIN .AND. .NOT. LINNEW) THEN
       NGCIN = NGCIN - 1
    ELSEIF (LINNEW .AND. .NOT. LGCIN) THEN
       NGCIN = NGCIN + 1
    ENDIF

    RETURN
  END SUBROUTINE NOGCIN

  SUBROUTINE GENGRD(NGCBLK,LSTGRD,LSTIND,NGDTOT,NGRDCV,GCMCON, &
       GRDBLK,NGRDBK,QGCSPH,NRGRID,GRDSIZ,NGRDX,NGRDY, &
       NGRDZ,XGCMIN,YGCMIN,ZGCMIN,X,Y,Z)
    !
    !       Setup the grid for grand canonical simulations.
    !
    INTEGER NGCBLK(*), LSTGRD(*), LSTIND(*), NGDTOT
    INTEGER NRGRID, NGRDBK, NGRDCV, GRDBLK(*)
    INTEGER NGRDX, NGRDY, NGRDZ
    real(chm_real)  X(*), Y(*), Z(*), XGCMIN, YGCMIN, ZGCMIN, GRDSIZ
    LOGICAL GCMCON(:), QGCSPH
    !
    INTEGER J, NGRNO, IDX, NGRID2, NDIST
    INTEGER NGX, NGY, NGZ, NGX2, NGY2, NGZ2
    INTEGER SOLD
    !
    !       Initialize with si=0
    NGRNO = NGRDX*NGRDY*NGRDZ
    DO J=1,NGRNO
       NGCBLK(J) = 0
    ENDDO
    !       Mark points outside the allowed sphere with si=-1.
    NGDTOT = 0
    IF (QGCSPH) THEN
       DO NGX=1,NGRDX
          DO NGY=1,NGRDY
             DO NGZ=1,NGRDZ
                IF (ISPOUT(NGX,NGY,NGZ,NGRDX,NGRDY,NGRDZ)) THEN
                   IDX=(NGZ-1)*NGRDY*NGRDX+(NGY-1)*NGRDX+NGX
                   NGCBLK(IDX) = -1
                ELSE
                   NGDTOT = NGDTOT + 1
                ENDIF
             ENDDO
          ENDDO
       ENDDO
    ELSE
       NGDTOT = NGRDX*NGRDY*NGRDZ
    ENDIF

    NGRID2 = NRGRID*NRGRID
    !       Increase si for sites occupied
    DO J = 1, NGRDBK

       IDX = GRDBLK(J)

       IF (GCMCON(IDX)) THEN

          !           Get the atom position in lattice units.
          !           Note that this code is somewhat inconsistent
          !           with the later use of the center-of-mass for
          !           Moving groups.
          NGX=INT((X(IDX)-XGCMIN)/GRDSIZ+0.01)+1
          NGY=INT((Y(IDX)-YGCMIN)/GRDSIZ+0.01)+1
          NGZ=INT((Z(IDX)-ZGCMIN)/GRDSIZ+0.01)+1

          !           Mark sites within the atom "volume"
          DO NGX2=NGX-NRGRID,NGX+NRGRID
             DO NGY2=NGY-NRGRID,NGY+NRGRID
                DO NGZ2=NGZ-NRGRID,NGZ+NRGRID
                   IF (INGRID(NGX2,NGY2,NGZ2,NGRDX,NGRDY,NGRDZ)) THEN
                      IDX=(NGZ2-1)*NGRDY*NGRDX+(NGY2-1)*NGRDX+NGX2
                      SOLD=NGCBLK(IDX)
                      IF(SOLD.GE.0) THEN
                         NDIST=(NGX-NGX2)**2+(NGY-NGY2)**2 +(NGZ-NGZ2)**2
                         IF (NDIST.LT.NGRID2) SOLD=SOLD+1
                         NGCBLK(IDX) = SOLD
                      ENDIF
                   ENDIF
                ENDDO
             ENDDO
          ENDDO
       ENDIF
    ENDDO

    !       Fill in the lstgrid array with sites where si=0
    NGRDCV=0
    DO NGX=1,NGRDX
       DO NGY=1,NGRDY
          DO NGZ=1,NGRDZ
             IDX=(NGZ-1)*NGRDY*NGRDX+(NGY-1)*NGRDX+NGX
             SOLD=NGCBLK(IDX)
             IF (SOLD.EQ.0) THEN
                NGRDCV=NGRDCV+1
                LSTGRD(NGRDCV) = IDX
                LSTIND(IDX) = NGRDCV
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    DO J=NGRDCV+1,NGRNO
       LSTGRD(J) = 0
    ENDDO

    RETURN
  END SUBROUTINE GENGRD
#endif /* (gcmc)*/
#endif /* (mc)*/
end module mcmvgcmc

