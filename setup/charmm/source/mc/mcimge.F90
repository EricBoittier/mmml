module mcimg
#if KEY_MC==1
  use chm_types
  use dimens_fcm
  implicit none

contains

  SUBROUTINE MCIMGE(EIMV,EIME,IMVATM,IMVGRP,LFIRST,BIMAG, &
       IMBLOX,IMJNBX, &
       IBLOGX,IJNBGX,BNBND,NATIM,MCIMLP,MCIMGP, &
       IMATPT,NTRANS,MCATT,LGROUP,IAMC,NAMC, &
       IGMC,NGMC,IMLIM,MCGTTP,X,Y,Z,LGCMC &
#if KEY_PERT==1
       ,MCIMLRP,MCIMLPP,IMLIMR,IMLIMP, &
       BNBNDR,BNBNDP,BIMAGR,BIMAGP &
#endif /* KEY_PERT */
       )
    !
    !       Computes the image energy of the moving atoms in MC.
    !
    !       Works similarly to MCNBND in that it sets up a miniature non-bonded
    !       list which it then passes to EIMNBD.
    !
    !       Aaron R. Dinner
    !
    use number
    use energym
    use eimg
    use econtmod
    use deriv
    use param
#if KEY_PERT==1
    use pert
    use epert_mod
#endif /* KEY_PERT */
    use psf
    use fast
    use memory
#if KEY_FLUCQ==1
    use flucqm,only:fqcfor
    use flucq
#endif /* KEY_FLUCQ */

    implicit none

    ! arguments
    INTEGER IMVATM(:), IMVGRP(:)
    INTEGER NATIM
    type(imageDataStructure) BIMAG
    INTEGER IMBLOX(:), IMJNBX(:)
    INTEGER IBLOGX(:), IJNBGX(:)
    type(nonbondDataStructure) BNBND

    type(chm_iptr),dimension(:) :: MCIMLP, MCIMGP, MCATT, MCGTTP
    INTEGER IMATPT(:)
    INTEGER NTRANS, IAMC, NAMC, IGMC, NGMC, IMLIM(:)
    real(chm_real) :: EIMV, EIME
    real(chm_real) :: X(:), Y(:), Z(:)
    LOGICAL LGROUP, LFIRST, LGCMC

    ! local vars
    integer :: NIMGRP
    INTEGER N, I, ITR, NIMTMP, IFRST
    integer,allocatable,dimension(:) :: ICURRP
    real(chm_real)  EMCV, EMCE

#if KEY_PERT==1
    ! arguments
    type(chm_iptr),dimension(:) :: MCIMLRP, MCIMLPP
    INTEGER IMLIMR(:), IMLIMP(:)
    type(nonbondDataStructure) BNBNDR
    type(nonbondDataStructure) BNBNDP
    type(imageDataStructure) BIMAGR
    type(imageDataStructure) BIMAGP

    ! local vars
    INTEGER NR, NP
    integer,allocatable,dimension(:) :: ICURRPR, ICURRPP
    real(chm_real) EIMVR,EIMER,EIMVP,EIMEP,EMCVP,EMCVR,EMCER,EMCEP
#endif

    ! Setup the miniature non-bonded lists.
    ! Note that the exclusion list is not setup so that bonds between
    ! primaries and images will break MC.

    IF (LFIRST) THEN

       IF (LGROUP) THEN

          !           Get space for a scratch array to be used below and initialize it
          call chmalloc('mcimge.src','MCIMGE','ICURRP',NGMC-IGMC+1,intg=ICURRP)
          !           Moving primaries with still images
          CALL PRIMMV(BIMAG%IMBLOG,BIMAG%IMJNBG, IMVGRP, MCIMGP, &
               IGMC, NGMC, ICURRP)
          N = BIMAG%IMBLOG(NGMC)
          !           Moving images with still primaries
          DO ITR = 1, NTRANS
             I = 2*ITR
             CALL IMAGMV(IMLIM(I-1),IMLIM(I),BIMAG%IMBLOG,BIMAG%IMJNBG,IBLOGX, &
                  IJNBGX, IMVGRP, ICURRP, IMATPT(ITR), &
                  MCGTTP, N,IGMC,NGMC,IGPBS,.TRUE., &
                  MCIMGP, LGCMC)
          ENDDO
          call chmdealloc('mcimge.src','MCIMGE','ICURRP',NGMC-IGMC+1,intg=ICURRP)

       ELSE

          !           Same structure as group calls
          call chmalloc('mcimge.src','MCIMGE','ICURRP',NAMC-IAMC+1,intg=ICURRP)
          CALL PRIMMV(BIMAG%IMBLO,BIMAG%IMJNB,IMVATM, MCIMLP, IAMC,NAMC, &
               ICURRP)
#if KEY_PERT==1
          IF (QPERT) THEN
             call chmalloc('mcimge.src','MCIMGE','ICURRPR',NAMC-IAMC+1,intg=ICURRPR)
             call chmalloc('mcimge.src','MCIMGE','ICURRPP',NAMC-IAMC+1,intg=ICURRPP)
             CALL PRIMMV(BIMAGR%IMBLO,BIMAGR%IMJNB,IMVATM, MCIMLRP, IAMC,NAMC, &
                  ICURRPR)
             CALL PRIMMV(BIMAGP%IMBLO,BIMAGP%IMJNB,IMVATM, MCIMLPP, IAMC,NAMC, &
                  ICURRPP)
          ENDIF
#endif 

          N = BIMAG%IMBLO(NAMC)
#if KEY_PERT==1
          IF (QPERT) THEN
             NR = BIMAGR%IMBLO(NAMC)
             NP = BIMAGP%IMBLO(NAMC)
          ENDIF
#endif 
          DO ITR = 1, NTRANS
             I = 2*ITR
             CALL IMAGMV(IMLIM(I-1),IMLIM(I),BIMAG%IMBLO,BIMAG%IMJNB,IMBLOX,IMJNBX, &
                  IMVATM, ICURRP, IMATPT(ITR), MCATT, N, &
                  IAMC,NAMC,IGPBS,.FALSE., &
                  MCIMLP, LGCMC)
#if KEY_PERT==1
             IF (QPERT) THEN
                CALL IMAGMV(IMLIMR(I-1),IMLIMR(I),BIMAGR%IMBLO,BIMAGR%IMJNB,IMBLOX, &
                     IMJNBX, &
                     IMVATM, ICURRPR, IMATPT(ITR), MCATT, NR, &
                     IAMC,NAMC,IGPBS,.FALSE., &
                     MCIMLRP, LGCMC)
                CALL IMAGMV(IMLIMP(I-1),IMLIMP(I),BIMAGP%IMBLO,BIMAGP%IMJNB,IMBLOX, &
                     IMJNBX, &
                     IMVATM, ICURRPP, IMATPT(ITR), MCATT, NP, &
                     IAMC,NAMC,IGPBS,.FALSE., &
                     MCIMLPP, LGCMC)
             ENDIF
#endif 
          ENDDO
#if KEY_PERT==1
          IF (QPERT) THEN
             call chmdealloc('mcimge.src','MCIMGE','ICURRPP',NAMC-IAMC+1,intg=ICURRPP)
             call chmdealloc('mcimge.src','MCIMGE','ICURRPR',NAMC-IAMC+1,intg=ICURRPR)
          ENDIF
#endif 
          call chmdealloc('mcimge.src','MCIMGE','ICURRP',NAMC-IAMC+1,intg=ICURRP)

       ENDIF ! LGROUP

    ENDIF ! LFIRST

    !       Get moving primary contribution

    IF (LGROUP) THEN
       IFRST = IGMC
    ELSE
       IFRST = IAMC
    ENDIF

    CALL EIMNBD(EIMV,EIME,BNBND,BIMAG,BIMAG,IFRST,NAMC,CG,RSCLF, &
         NGMC,IGPBS,IGPTYP,IAC,IACNB,DX,DY,DZ,X,Y,Z,.FALSE., &
         ECONT,.FALSE.,ZERO,QETERM(IMVDW),QETERM(IMELEC), &
#if KEY_FLUCQ==1
         QFLUC,FQCFOR,         & 
#endif
         QETERM(IMST2),NST2,ETERM(IMST2) &
#if KEY_WCA==1
         ,WCA                   & 
#endif
#if KEY_LJPME==1
         ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
         )

#if KEY_PERT==1
    IF (QPERT) THEN
       CALL EIMNBD(EIMVR,EIMER,BNBNDR,BIMAG,BIMAGR,IFRST,NAMC, &
            PPCG,PPRSCLF, &
            NGMC,PPIGPBS,PPIGPTP, &
            PPIAC,PPIACNB, &
            DX,DY,DZ,X,Y,Z,.FALSE., &
            ECONT,.FALSE.,ZERO,QETPRT(IMVDW),QETPRT(IMELEC), &
#if KEY_FLUCQ==1
            QFLUC,FQCFOR,         & 
#endif
            QETPRT(IMST2),NST2,ETPRTM(IMST2) &
#if KEY_WCA==1
            ,PPWCA                                & 
#endif
#if KEY_LJPME==1
            ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
            )

       CALL EIMNBD(EIMVP,EIMEP,BNBNDP,BIMAG,BIMAGP,IFRST,NAMC,CG,RSCLF, &
            NGMC,IGPBS,IGPTYP,IAC,IACNB,DX,DY,DZ,X,Y,Z,.FALSE., &
            ECONT,.FALSE.,ZERO,QETERM(IMVDW),QETERM(IMELEC), &
#if KEY_FLUCQ==1
            QFLUC,FQCFOR,         & 
#endif
            QETERM(IMST2),NST2,ETPRTL(IMST2) &
#if KEY_WCA==1
            ,WCA                   & 
#endif
#if KEY_LJPME==1
           ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
            )
       EIMV = EIMV + EIMVR*LAMDAM + EIMVP*LAMDA
       EIME = EIME + EIMER*LAMDAM + EIMEP*LAMDA
    ENDIF
#endif 
    !       Get moving image contribution
    !       Send each transformation (close in number) separately to EIMNBD

    DO ITR = 1, NTRANS
       I = 2*ITR

       !         Only call EIMNBD if there are moving atoms in transformation ITR
       IF (IMLIM(I-1) .LE. IMLIM(I)) THEN
          CALL EIMNBD(EMCV,EMCE,BNBND,BIMAG,BIMAG, &
               IMLIM(I-1),IMLIM(I),CG,RSCLF,IMLIM(I),IGPBS, &
               IGPTYP,IAC,IACNB,DX,DY,DZ,X,Y,Z,.FALSE.,ECONT, &
               .FALSE.,ZERO,QETERM(IMVDW),QETERM(IMELEC), &
#if KEY_FLUCQ==1
               QFLUC,FQCFOR,         & 
#endif
               QETERM(IMST2),NST2,ETERM(IMST2) &
#if KEY_WCA==1
               ,WCA                   & 
#endif
#if KEY_LJPME==1
               ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
               )
          EIMV = EIMV + EMCV
          EIME = EIME + EMCE
#if KEY_PERT==1
          IF (QPERT) THEN
             CALL EIMNBD(EMCVR,EMCER,BNBNDR,BIMAG,BIMAGR, &
                  IMLIM(I-1),IMLIM(I),PPCG, &
                  PPRSCLF,IMLIM(I), &
                  PPIGPBS, &
                  PPIGPTP,PPIAC, &
                  PPIACNB, &
                  DX,DY,DZ,X,Y,Z,.FALSE.,ECONT, &
                  .FALSE.,ZERO,QETPRT(IMVDW),QETPRT(IMELEC), &
#if KEY_FLUCQ==1
                  QFLUC,FQCFOR,   & 
#endif
                  QETPRT(IMST2),NST2,ETPRTM(IMST2) &
#if KEY_WCA==1
                  ,PPWCA                      & 
#endif
#if KEY_LJPME==1
                  ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
                  )
             CALL EIMNBD(EMCVP,EMCEP,BNBNDP,BIMAG,BIMAGP, &
                  IMLIM(I-1),IMLIM(I),CG,RSCLF,IMLIM(I),IGPBS, &
                  IGPTYP,IAC,IACNB,DX,DY,DZ,X,Y,Z,.FALSE.,ECONT, &
                  .FALSE.,ZERO,QETERM(IMVDW),QETERM(IMELEC), &
#if KEY_FLUCQ==1
                  QFLUC,FQCFOR,   & 
#endif
                  QETERM(IMST2),NST2,ETPRTL(IMST2) &
#if KEY_WCA==1
                  ,WCA                   & 
#endif
#if KEY_LJPME==1
                  ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
                  )
             EIMV = EIMV + EMCVR*LAMDAM + EMCVP*LAMDA
             EIME = EIME + EMCER*LAMDAM + EMCEP*LAMDA
          ENDIF
#endif 
       ENDIF

    ENDDO

    RETURN
  END SUBROUTINE MCIMGE

  SUBROUTINE PRIMMV(IMBLOG,IMJNBG,IMVGR,MCIMGB,IGMC,NGMC,ICURR)
    !
    !       Sets up the image MC restricted non-bonded list if
    !       group energy calculations are required.
    !
    !       Aaron R. Dinner
    !
    INTEGER IMVGR(*),ICURR(*)
    INTEGER IGMC, NGMC
    INTEGER IMBLOG(:),IMJNBG(:)
    type(chm_iptr) :: MCIMGB(:)
    !
    INTEGER I, J, N, NB, IGMCM1

    IGMCM1 = IGMC - 1
    N = 0
    IF (IGMC .GT. 1) IMBLOG(IGMCM1) = N
    DO I = IGMC, NGMC
       IF (IMVGR(I) .GT. 0) THEN
          NB = MCIMGB(I)%A(2)
          DO J = 3, NB
             N = N + 1
             IMJNBG(N) = MCIMGB(I)%A(J)
          ENDDO
       ENDIF
       IMBLOG(I)=N
       !         Initialize ICURR for subsequent image -> primary calls
       ICURR(I-IGMCM1) = 3
    ENDDO

    RETURN
  END SUBROUTINE PRIMMV

  SUBROUTINE IMAGMV(ILO,IHI,IMBLO,IMJNB,IMBLOX,IMJNBX, &
       IMVATM,ICURR,ILIM,MCATT,N,IAMC,NAMC, &
       IGPBS,LGROUP,MCIML,LGCMC)
    !
    !       Sets up the moving image MC restricted non-bonded list
    !       for the next transformation if atom based energy calculations
    !       are required.
    !
    !       Note that ICURR and N must be initialized outside the routine.
    !
    !       Aaron R. Dinner
    !

    INTEGER ILIM, N, IAMC, NAMC, ILO, IHI
    type(chm_iptr) :: MCATT(:)
    INTEGER ICURR(*), IMVATM(:), IGPBS(*)
    INTEGER IMBLOX(:), IMJNBX(:), IMBLO(:), IMJNB(:)
    type(chm_iptr) :: MCIML(:)
    LOGICAL LGROUP, LGCMC

    INTEGER I, J, K, M, IC, J1, NIM, IAMCM1, JLIM
    INTEGER IPL, IFIRST, ILAST
    LOGICAL LF

    LF = .TRUE.

    IAMCM1 = IAMC - 1

    DO I = IAMC, NAMC
       IF (IMVATM(I) .GT. 0) THEN
          NIM = MCATT(I)%A(2)
          IC = I - IAMCM1
          !           Get the next image atom for this primary
          IF (ICURR(IC) .LE. NIM) THEN
             J = MCATT(I)%A(ICURR(IC))

             !             If a group list call, convert to atom number to check
             !             whether J is in the transformation of interest.
             IF (LGROUP) THEN
                JLIM = IGPBS(J)
             ELSE
                JLIM = J
             ENDIF

             IF (JLIM .LE. ILIM) THEN

                !               Fill in missing values to IMBLO
                J1 = J - 1
                IF (LF) THEN
                   LF = .FALSE.
                   IMBLO(J1) = N
                   ILO = J
                ELSE
                   DO K = IPL, J1
                      IMBLO(K) = N
                   ENDDO
                ENDIF

                ICURR(IC) = ICURR(IC) + 1
#if KEY_GCMC==1
                IF (LGCMC) THEN

                   M = MCIML(J)%A(2)
                   DO K = 3, M
                      N = N + 1
                      IMJNB(N) = MCIML(J)%A(K)
                   ENDDO

                ELSE
#endif 
                   IFIRST = IMBLOX(J1) + 1
                   ILAST  = IMBLOX(J)
                   !                 Get its non-bonded pairs
                   DO K = IFIRST, ILAST
                      IF (IMVATM(IMJNBX(K)).EQ.0) THEN
                         N = N + 1
                         IMJNB(N) = IMJNBX(K)
                      ENDIF
                   ENDDO
#if KEY_GCMC==1
                ENDIF
#endif 
                IMBLO(J) = N
                IPL = J + 1
             ENDIF
          ENDIF
       ENDIF
    ENDDO

    IF (LF) THEN
       !         No image atoms for this transformation --- make ILO gt IHI
       ILO =  0
       IHI = -1
    ELSE
       IHI = IPL - 1
    ENDIF

    RETURN
  END SUBROUTINE IMAGMV

  SUBROUTINE GTSIM(NTRANS,IMATPT,IMATTR,MCATTP, &
       IFIRST,NATOM)
    !
    !       Creates a primary->image list from the image->primary list
    !
    !       Aaron R. Dinner
    !
    use memory

    INTEGER NTRANS,IFIRST,NATOM
    INTEGER IMATPT(:), IMATTR(:)
    type(iptr_ptr) :: MCATTP

    INTEGER I, N, J
    integer,pointer,dimension(:) :: TEMPP

    IF (associated(MCATTP%A)) THEN
       DO I = 1, NATOM
          TEMPP => MCATTP%A(I)%A
          N = TEMPP(1)
          call chmdealloc('mcimge.src','GTSIM','TEMPP',N,intgp=TEMPP)
       ENDDO
    ELSE
       allocate(MCATTP%A(NATOM))
    ENDIF

    CALL CTMCIM(IMATPT, IMATTR, MCATTP%A, NTRANS, &
         IFIRST,NATOM)
    CALL FLMCIM(IMATPT, IMATTR, MCATTP%A, &
         NTRANS, IFIRST, NATOM)

    RETURN
  END SUBROUTINE GTSIM

  SUBROUTINE CTMCIM(IMATPT,IMATTR,MCATT,NTRANS,IFIRST,NATOM)
    !
    !       Count the number of image atoms for each primary
    !
    !       Aaron R. Dinner
    !
    INTEGER IMATPT(:), IMATTR(:)
    type(chm_iptr) :: MCATT(:)
    INTEGER IFIRST, NTRANS, NATOM

    INTEGER I, ITRANS, ISTRT, IEND

    DO I = 1, NATOM
       MCATT(I)%LEN = 0
    ENDDO

    ISTRT = IFIRST
    DO ITRANS=1,NTRANS
       IEND=IMATPT(ITRANS)
       DO I=ISTRT,IEND
          MCATT(IMATTR(I))%LEN = MCATT(IMATTR(I))%LEN + 1
       ENDDO
       ISTRT=IEND+1
    ENDDO

    RETURN
  END SUBROUTINE CTMCIM

  SUBROUTINE FLMCIM(IMATPT,IMATTR,MCATT, &
       NTRANS,IFIRST,NATOM)
    !
    !       Fill the primary->image list
    !
    !       Aaron R. Dinner
    !
    use memory
    INTEGER IMATPT(:), IMATTR(:)
    type(chm_iptr) :: MCATT(:)
    INTEGER NTRANS, IFIRST, NATOM

    integer,dimension(NATOM) :: MCTOT
    integer,pointer,dimension(:) :: TEMPP
    INTEGER I, J, K, ISTRT, IEND, N
    integer,parameter :: IMBUFF = 5

    DO I = 1, NATOM
       N = MCATT(I)%LEN + 2 + IMBUFF
       call chmalloc('mcimge.src','FLMCIM','TEMPP',N,intgp=TEMPP)
       TEMPP(1) = N
       TEMPP(2) = MCATT(I)%LEN + 2
       MCATT(I)%A => TEMPP
       MCTOT(I) = 2
    ENDDO

    ISTRT=IFIRST
    DO I=1,NTRANS
       IEND=IMATPT(I)
       DO J=ISTRT,IEND
          K = IMATTR(J)
          MCTOT(K) =  MCTOT(K) + 1
          MCATT(K)%A(MCTOT(K)) = J
       ENDDO
       ISTRT = IEND + 1
    ENDDO

    RETURN
  END SUBROUTINE FLMCIM

  SUBROUTINE MCITRN(X,Y,Z,IMVNG,MCATT,NTRANS,IMTRNS,NOROT,IMATPT &
#if KEY_GCMC==1
       ,GCMCON      & 
#endif
       )
    !
    !       Get the new coordinates of the image atoms after a move
    !
    !       Aaron R. Dinner
    !
    INTEGER NTRANS
    INTEGER IMVNG(:), IMATPT(:)
    type(chm_iptr) :: MCATT(:)
    real(chm_real)  X(*), Y(*), Z(*)
    real(chm_real)  IMTRNS(*)
    LOGICAL NOROT
#if KEY_GCMC==1
    LOGICAL GCMCON(:) 
#endif

    INTEGER I, J, K, L, N, NN, ITRANS, IPT
    INTEGER I1, I2, NG

    NG = IMVNG(1)
    N = IMVNG(NG)
    NG = NG + 2
    DO I = NG, N, 2
       I1 = IMVNG(I-1)
       I2 = IMVNG(I)
       DO J = I1, I2
          NN = MCATT(J)%A(2)
          ITRANS = 0
          DO K = 3, NN
             L =  MCATT(J)%A(K)
             ITRANS = IWTRNS(NTRANS,IMATPT,ITRANS,L)
             IPT = (ITRANS - 1)*12
             IF (NOROT) THEN
                X(L) = X(J) + IMTRNS(IPT+10)
                Y(L) = Y(J) + IMTRNS(IPT+11)
                Z(L) = Z(J) + IMTRNS(IPT+12)
             ELSE
                X(L)=X(J)*IMTRNS(IPT+1)+Y(J)*IMTRNS(IPT+2)+ &
                     Z(J)*IMTRNS(IPT+3)+IMTRNS(IPT+10)
                Y(L)=X(J)*IMTRNS(IPT+4)+Y(J)*IMTRNS(IPT+5)+ &
                     Z(J)*IMTRNS(IPT+6)+IMTRNS(IPT+11)
                Z(L)=X(J)*IMTRNS(IPT+7)+Y(J)*IMTRNS(IPT+8)+ &
                     Z(J)*IMTRNS(IPT+9)+IMTRNS(IPT+12)

             ENDIF
#if KEY_GCMC==1
             GCMCON(L) = GCMCON(J) 
#endif
          ENDDO
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE MCITRN

  INTEGER FUNCTION IWTRNS(NTRANS,IMATPT,IPREV,ICURR)
    !
    !       Returns the index of the image transformation
    !       that includes atom ICURR
    !
    !       Aaron R. Dinner
    !
    !       Passed variables
    INTEGER NTRANS, IMATPT(*), IPREV, ICURR
    !       Local variables
    INTEGER I

    I = IPREV + 1
    DO IWTRNS = I, NTRANS
       IF (ICURR .LE. IMATPT(IWTRNS)) RETURN
    ENDDO

    CALL WRNDIE(-5,'<IWTRNS>','INTERNAL IMAGE ERROR')

  END FUNCTION IWTRNS

  SUBROUTINE GTSIMG(IMATTR,MCGTTP,IA2G,IGPBS,NGRP,NIMGRP)
    !
    !       Creates a primary->image group list from the
    !       image->primary atom list and IGPBS
    !
    !       Aaron R. Dinner
    !

    use memory
    INTEGER NGRP, NIMGRP
    INTEGER IA2G(*), IGPBS(*)
    INTEGER IMATTR(:)
    type(iptr_ptr) :: MCGTTP

    integer,dimension(NGRP) :: MCTOTP
    integer,pointer,dimension(:) :: TEMPP
    INTEGER I, N

    IF (associated(MCGTTP%A)) THEN
       DO I = 1, NGRP
          TEMPP => MCGTTP%A(I)%A
          N = TEMPP(1)
          call chmdealloc('mcimge.src','GTSIMG','TEMPP',N,intgp=TEMPP)
       ENDDO
    ELSE
       allocate(MCGTTP%A(NGRP))
    ENDIF

    CALL CGMCIM(MCGTTP%A, MCTOTP, IMATTR, IA2G, &
         IGPBS,NGRP,NIMGRP,.FALSE.)
    CALL CGMCIM(MCGTTP%A, MCTOTP, IMATTR, IA2G, &
         IGPBS,NGRP,NIMGRP,.TRUE.)

    RETURN
  END SUBROUTINE GTSIMG

  SUBROUTINE CGMCIM(MCGTT,MCTOT,IMATTR,IA2G,IGPBS,NGRP,NIMGRP,LFILL)
    !
    !       If LFILL is false, count the number of image groups for each
    !       primary group
    !
    !       If LFILL is  true, fills the primary->image group lists
    !
    !       Aaron R. Dinner
    !
    use memory
    type(chm_iptr) :: MCGTT(:)
    INTEGER IMATTR(:), IA2G(*), MCTOT(:)
    INTEGER IGPBS(*), NGRP, NIMGRP
    LOGICAL LFILL
    !
    integer,pointer,dimension(:) :: TEMPP
    INTEGER I, N, IGPRIM, ISTRT

    IF (LFILL) THEN
       DO I = 1, NGRP
          N = MCGTT(I)%LEN + 2
          call chmalloc('mcimge.src','CGMCIM','TEMPP',N,intgp=TEMPP)
          TEMPP(1) = N
          TEMPP(2) = MCGTT(I)%LEN + 2
          MCGTT(I)%A => TEMPP
          MCTOT(I) = 2
       ENDDO
    ELSE
       DO I = 1, NGRP
          MCGTT(I)%LEN = 0
       ENDDO
    ENDIF

    ISTRT = NGRP + 1
    DO I = ISTRT, NIMGRP
       IGPRIM = IA2G(IMATTR(IGPBS(I)+1))
       IF (LFILL) THEN
          MCTOT(IGPRIM) = MCTOT(IGPRIM) + 1
          MCGTT(IGPRIM)%A(MCTOT(IGPRIM)) = I
       ELSE
          MCGTT(IGPRIM)%LEN = MCGTT(IGPRIM)%LEN + 1
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE CGMCIM

  SUBROUTINE MCICNT(XCENT,YCENT,ZCENT,IMVNG,MCGTT,IGPBS,NTRANS, &
       NOROT,IMATPT,IMTRNS,X,Y,Z)
    !
    !       Update image coordinates of group centers
    !
    !       Aaron R. Dinner
    !
    INTEGER IMVNG(:), IMATPT(:), IGPBS(*), NTRANS
    type(chm_iptr) :: MCGTT(:)
    real(chm_real)  XCENT(:), YCENT(:), ZCENT(:)
    real(chm_real)  X(*), Y(*), Z(*), IMTRNS(*)
    LOGICAL NOROT
    !
    INTEGER I, I1, I2, J, K, L, NG, N, ITRANS, IPT, NN

    NG = IMVNG(1)
    N = IMVNG(NG)
    NG = NG + 2
    DO I = NG, N, 2
       I1 = IMVNG(I-1)
       I2 = IMVNG(I)
       DO J = I1, I2
          NN = MCGTT(J)%A(2)
          ITRANS = 0
          DO K = 3, NN
             !             L is the image group and J is the primary group
             L = MCGTT(J)%A(K)
             !             Get the transformation index from first atom
             ITRANS = IWTRNS(NTRANS,IMATPT,ITRANS,IGPBS(L)+1)
             IPT = (ITRANS - 1)*12
             IF (NOROT) THEN
                XCENT(L) = XCENT(J) + IMTRNS(IPT+10)
                YCENT(L) = YCENT(J) + IMTRNS(IPT+11)
                ZCENT(L) = ZCENT(J) + IMTRNS(IPT+12)
             ELSE
                X(L)=XCENT(J)*IMTRNS(IPT+1)+ &
                     YCENT(J)*IMTRNS(IPT+2)+ &
                     ZCENT(J)*IMTRNS(IPT+3)+IMTRNS(IPT+10)
                Y(L)=XCENT(J)*IMTRNS(IPT+4)+ &
                     YCENT(J)*IMTRNS(IPT+5)+ &
                     ZCENT(J)*IMTRNS(IPT+6)+IMTRNS(IPT+11)
                Z(L)=XCENT(J)*IMTRNS(IPT+7)+ &
                     YCENT(J)*IMTRNS(IPT+8)+ &
                     ZCENT(J)*IMTRNS(IPT+9)+IMTRNS(IPT+12)
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    RETURN
  END SUBROUTINE MCICNT

#endif 
end module mcimg

