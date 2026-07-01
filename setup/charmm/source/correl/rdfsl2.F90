module rdfsol_subs
  use chm_kinds
  use dimens_fcm
  implicit none

contains


#if KEY_RDFSOL==1 /*rdfsol*/
  !
  !     This file contains only the distance distance calculating routines
  !     since they are rather lengthy...
  !
  SUBROUTINE GOODIS(NAT,NATIM,X,Y,Z,RHI,RHINV,X0,Y0,Z0, &
       NCUBX,NCUBY,NCUBZ,NCUBE,IMATTR,NBIN,DR, &
       NSOLV,NSOLVI,SOLVL,SOLVSL,DIP,REVRES,GXX, &
       QRDF,QDDIP,QQDIP,QHD,QUSER, &
       LSTM0,LSTM1,LSTM2,LSTN0,LSTN1, &
       LSTM0M,LSTM1M,LSTM2M,LSTN0M,LSTN1M, &
       MSOLVL,DOIMAGES,QAWAT,QPREC,NN, &
       QMINI,MINI,LNMINI,LNMIN2)
    !
    !     calculates distances with the cubing algorithm and fills
    !     the GOO, GOH and GHH arrays accordingly
    !     adapted from shell.src which itself bases on code by Mike Crowley
    !
    !     assumptions: both sets are the same, so if we calculate
    !                  dip-dip(XDIP=2) or chardge-dip(XDIP=3) we have
    !                  water-water, else we wouldn't be here
    !
    !     possible speedups: don't disentangle the intertwined linked lists
    !                        since we don't need to sort the lists
    !                        -> work with them directly
    !                       &: work with integer-arithmetics in decisions
    !
    !     Tibor Rudas Dec 2002 - Jun 2003
    !
    !     Aug. 2005: switch from XDIP to logicals and allow for simultaneous
    !                calculation of the functions
    !

  use new_timer,only:timer_start,timer_stop,T_setgrd,T_bldlst 
  use stream

    INTEGER NAT,NATIM,NBIN
    INTEGER NSOLV,NSOLVI,SOLVL(*),REVRES(*),SOLVSL(*)
    real(chm_real)  X(*),Y(*),Z(*),RHI,RHINV
    real(chm_real)  DR,X0,Y0,Z0
    INTEGER NCUBX,NCUBY,NCUBZ,NCUBE,IMATTR(*)
    INTEGER LSTM0(*),LSTM1(*),LSTM2(*),LSTN0(*),LSTN1(*)
    INTEGER LSTM0M(*),LSTM1M(*),LSTM2M(*)
    INTEGER LSTN0M(*),LSTN1M(*),MSOLVL(*),NN
    real(chm_real) GXX(8,*),DIP(4,*)
    LOGICAL QRDF,QDDIP,QQDIP,QHD,QUSER
    LOGICAL DOIMAGES,QAWAT,QPREC
    !
    !     new variables for distance-matrix (minimum image approach)
    LOGICAL QMINI
    INTEGER LNMINI,LNMIN2
    real(chm_real4) MINI(*)
    !
    INTEGER I,J,K,N,N2,N11,N22,M,M2,IM,TMP,NLO,NHI,NLO2,NHI2
    INTEGER NIX,NIX2,NBRN,NBRNIM,BIN
    INTEGER NCUBXY,NTIM,MX,MY,MZ,NBR27,MADD(27)
    real(chm_real)  RHI2,XX,YY,ZZ,DDX,DDY,DDZ,XTMP
    LOGICAL N1IM,N2IM
    !
    !
    !----------------------------------------------------------------------
    !     Executable code begins here
    !
    call timer_start(T_setgrd)
    !
    !---- Compute frequently used intermediate scalar results -------------
    !
    RHI2 = RHI  * RHI
    NCUBXY = NCUBX * NCUBY
    !
    !---- Prepare MADD array-----------------------------------------------
    !
    !     The MADD array is defined so that for a given cube M,
    !     the 27-cube neighborhood consists of the cubes M+MADD(1..27).
    !
    DO MX = -1, 1
       DO MY = -1, 1
          DO MZ = -1, 1
             NBR27 = (MX+1)*9 + (MY+1)*3 + (MZ+1) + 1
             MADD(NBR27) = MX + NCUBX * (MY + NCUBY * MZ)
          ENDDO
       ENDDO
    ENDDO
    !
    !     Remove duplicate offsets from MADD, or we will get duplicate
    !     atoms and therefore array-bound problems down the line.  Any
    !     element that is set to NCUBE is effectively deleted since later
    !     M+NCUBE is out of the range 1..NCUBE.
    !
    !     Crude method OK since max number of operations is 13 * 27 = 351
    !
    !
    DO NBR27 = 1, 27
       DO TMP = 1, NBR27-1
          IF (MADD(TMP) == MADD(NBR27)) MADD(NBR27) = NCUBE + 1
       END DO
    END DO
    !
    !---- Prepare a list of particles for each cube -----------------------
    !
    !     First, temporarily set up LSTN1 such that for a given atom N,
    !     LSTN1(N) is the index of the cube it's in.  This will be
    !     discarded after the next step.
    !
    DO N = 1, NSOLV
       K=SOLVL(N)
       LSTN1(N) = ( INT((Z(K)-Z0)*RHINV)  * NCUBY &
            + INT((Y(K)-Y0)*RHINV)) * NCUBX &
            + INT((X(K)-X0)*RHINV) + 1
    ENDDO
    !     images:
    IF(DOIMAGES) THEN
       DO N = (NSOLV+1), NSOLVI
          K=SOLVL(N)
          LSTN1M(N) = ( INT((Z(K)-Z0)*RHINV)  * NCUBY &
               + INT((Y(K)-Y0)*RHINV)) * NCUBX &
               + INT((X(K)-X0)*RHINV) + 1
       ENDDO
    ENDIF
    !
    !     Invert LSTN1:  Set up LSTM0/LSTN0 as intertwined linked lists
    !     such that for a given cube M, you can recursively read off the
    !     atoms that it contains.  This will be discarded after the next
    !     step.
    !
    !     Vectorizable.
    DO M = 1, NCUBE
       LSTM0(M) = 0
       LSTM0M(M) = 0
    END DO
    !
    DO N = 1, NSOLV
       M = LSTN1(N)
       LSTN0(N) = LSTM0(M)
       LSTM0(M) = N
    ENDDO
    !     images
    IF(DOIMAGES) THEN
       DO N =  (NSOLV+1), NSOLVI
          M = LSTN1M(N)
          LSTN0M(N) = LSTM0M(M)
          LSTM0M(M) = N
       END DO
    ENDIF
    !
    !     we set up the LSTN1's as lists of ATOMNUMBERS of the solvent
    !     primary/image atoms in the given cube M
    !     we need begin and end pointers for the list since we don't use _all_
    !     atoms but only those in solute/solvent. so some cubes may not contain
    !     _any_ atoms of interest. thus LISTM1U/V(M-1) may sometimes be = 0 which
    !     is _not_ the beginning of atoms in cube M.
    !     so: LSTM1/M...(M)  is the start pointer and
    !         LSTM2/M...(M)  is the last element in the list
    !
    !     BEWARE: up to now the numbers in the LST.N.. lists were only
    !             pointers to the solute/solvent/... lists
    !             -> convert to real atom numbers now !
    !
    I  = 0
    IM = 0
    DO M = 1, NCUBE
       !
       !        .... primary .....
       N = LSTM0(M)
       LSTM1(M) = 0
       LSTM2(M) = 0
       DO WHILE (N /= 0)
          I = I + 1
          !           set begin pointer if this is the first element
          IF(LSTM1(M) <= 0) LSTM1(M)=I
          LSTN1(I) = SOLVL(N)
          N = LSTN0(N)
       ENDDO
       IF(LSTM1(M) > 0) LSTM2(M) = I
       !
       !        .... images .....
       IF(DOIMAGES) THEN
          N = LSTM0M(M)
          LSTM1M(M) = 0
          LSTM2M(M) = 0
          DO WHILE (N /= 0)
             IM = IM + 1
             !              set begin pointer if this is the first element
             IF(LSTM1M(M) <= 0) LSTM1M(M)=IM
             LSTN1M(IM) = SOLVL(N)
             N = LSTN0M(N)
          ENDDO
          IF(LSTM1M(M) > 0) LSTM2M(M) = IM
       ENDIF
       !
    ENDDO
    !
    !---- AND FINALLY: check all pairs ------------------------------------
    !
    !     Now we loop over all cubes M.
    !     If M contains atoms -> do this cube.
    !     set up a list of atoms in the  surrounding 27 cubes.
    !     then check all pairs.
    !
    !---- Start of major loop over cubes M --------------------------------
    !
    call timer_stop(T_setgrd)
    call timer_start(T_bldlst)
    !
    DO M = 1,NCUBE
       !
       !------- Determine range of atoms LSTN1(NLO..NHI) in center cube M ----
       !
       !        only do this cube if it contains PRIMARY solute atoms
       IF(LSTM1(M) > 0) THEN
          !
          NLO=LSTM1(M)
          NHI=LSTM2(M)
          !
          !---------- Set LSTN0 equal to an array of neighbor atoms -------------
          !
          !           LSTN0 will be an array of the indices of atoms which are in
          !           the cube M and its neighbors.  In the outer loop, M2 takes on
          !           up to 27 indices of cubes adjacent to M.  For each M2,
          !           LSTN1(NLO2..NHI2) is the set of atoms in M2.  We traverse
          !           that list backwards so we end up with little runs of
          !           ascending order.
          !   
          !           Only build the list of solvent molecules since we only
          !           compute solute-solvent interactions
          !   
          NBRN = 0
          NBRNIM = 0
          DO NBR27 = 1, 27
             !
             !              Propose an M2, and if it's in range, then...
             M2 = M + MADD(NBR27)
             IF (M2 >= 1 .AND. M2 <= NCUBE) THEN ! ----------
                !
                !                 ... PRIMARY ...
                !                 Set NLO2..NHI2 equal to range of indices into LSTN1
                NLO2 = LSTM1(M2)
                NHI2 = LSTM2(M2)
                !
                !                 Loop over those indices, filling LSTN0 with atom numbers
                IF(NLO2 > 0) THEN
                   DO NIX2 = NHI2, NLO2, -1
                      NBRN = NBRN + 1
                      LSTN0(NBRN) = LSTN1(NIX2)
                   END DO
                ENDIF
                !
                !                 ... IMAGES ...
                IF(DOIMAGES) THEN
                   NLO2 = LSTM1M(M2)
                   NHI2 = LSTM2M(M2)
                   IF(NLO2 > 0) THEN
                      DO NIX2 = NHI2, NLO2, -1
                         NBRNIM = NBRNIM + 1
                         LSTN0M(NBRNIM) = LSTN1M(NIX2)
                      END DO
                   ENDIF
                ENDIF
                !
             END IF           ! ----------
          END DO
          !
          !
          !
          !---------- Core: calculate distances & fill GOO etc ------------------
          !
          DO NIX = NLO, NHI
             !
             !              Record often-used parameters for solute atom N
             N = LSTN1(NIX)
             !
             XX = X(N)
             YY = Y(N)
             ZZ = Z(N)
             !
             !              Start this run of N2's (i.e. atoms in the surrounding cube)
             !------------- primary -------------------------------------------------
             DO I = 1, NBRN
                N2 = LSTN0(I)
                IF((N < N2).AND. &
                     (REVRES(N) /= REVRES(N2))) THEN
                   DDX = X(N2) - XX
                   DDY = Y(N2) - YY
                   DDZ = Z(N2) - ZZ
                   XTMP=DDX*DDX+DDY*DDY+DDZ*DDZ
                   IF(XTMP <= RHI2) THEN
                      XTMP=SQRT(XTMP)
                      CALL FILGOO(N,N2,N,N2,XTMP,DDX,DDY,DDZ,NBIN, &
                           DR,QAWAT,GXX,QRDF,QDDIP,QQDIP,QHD,QUSER, &
                           X,Y,Z,2,DIP,QMINI,MINI,LNMINI,LNMIN2)
                   ENDIF
                ENDIF
             END DO
             !------------- End primary --------------------------------------------
             !------------- image --------------------------------------------------
             IF(DOIMAGES)THEN
                DO I = 1, NBRNIM
                   N2 = LSTN0M(I)
                   J=IMATTR(N2)
                   IF(REVRES(N) /= REVRES(J)) THEN
                      DDX = X(N2) - XX
                      DDY = Y(N2) - YY
                      DDZ = Z(N2) - ZZ
                      XTMP=DDX*DDX+DDY*DDY+DDZ*DDZ
                      IF(XTMP <= RHI2) THEN
                         XTMP=SQRT(XTMP)
                         !                          use NN to indicate if we count a prim-img pair
                         !                          once or twice, depending on whether redundand
                         !                           images were generated
                         CALL FILGOO(N,N2,N,J,XTMP,DDX,DDY,DDZ,NBIN, &
                              DR,QAWAT,GXX,QRDF,QDDIP,QQDIP,QHD,QUSER, &
                              X,Y,Z,NN,DIP,QMINI,MINI,LNMINI,LNMIN2)
                      ENDIF
                   ENDIF
                END DO
                !              end if(doimages)
             ENDIF
             !------------- End image ----------------------------------------------
             !
             !           end loop over PRIMARY solute atoms (NIX)
          ENDDO
          !        end if(lstm1u(m) > 0) <- check if central cube contains atoms
       ENDIF
       !
       !     end loop over all cubes (M)
    ENDDO
    !     
    !---- End of major loop over cubes M ----------------------------------
    !
    !
    call timer_stop(t_bldlst)   
    !
    !---- we are done... --------------------------------------------------
    !     
    RETURN
  END SUBROUTINE GOODIS
  !
  !----------------------------------------------------------------------
  !
  !
  SUBROUTINE GXYDIS(NAT,NATIM,X,Y,Z,RHI,RHIINV,X0,Y0,Z0, &
       NCUBX,NCUBY,NCUBZ,NCUBE,IMATTR,NBIN,DR, &
       NSHSLU,NSHLUI,SHSLU,SOLUSL, &
       NSHSLV,NSHLVI,SHSLV,SOLVSL,DIP, &
       REVRES,GXX,QRDF,QDDIP,QQDIP,QHD,QUSER, &
       LSTM0U,LSTM1U,LSTM2U,LSTN0U,LSTN1U, &
       LSTM0V,LSTM1V,LSTM2V,LSTN0V,LSTN1V, &
       LSTM0MU,LSTM1MU,LSTM2MU, &
       LSTM0MV,LSTM1MV,LSTM2MV, &
       LSTN0MU,LSTN1MU, &
       LSTN0MV,LSTN1MV, &
       DOIMAGES,QAWAT,QBWAT,QPREC,NN, &
       QMINI,MINI,LNMINI,LNMIN2)
    !
    !     Similar to GOODIS but uses two separate sets (U/V) and only checks
    !     pairs U-V and V-U
    !
    !     Tibor Rudas Jan 2003 - Jun 2003
    !
    !     Sep. 2005: switch from XDIP to logicals to allow simultaneous
    !                calculation of functions
    !

  use new_timer,only:timer_start,timer_stop,t_setgrd,T_bldlst 
  use stream

    INTEGER NAT,NATIM
    real(chm_real)  X(*),Y(*),Z(*),RHI,RHIINV
    real(chm_real)  X0,Y0,Z0,DR
    INTEGER NCUBX,NCUBY,NCUBZ,NCUBE
    INTEGER IMATTR(*),NBIN
    INTEGER NSHSLU,NSHLUI,SHSLU(*),SOLUSL(*)
    INTEGER NSHSLV,NSHLVI,SHSLV(*),SOLVSL(*)
    INTEGER REVRES(*)
    real(chm_real) GXX(8,*),DIP(4,*)
    INTEGER LSTM0U(*),LSTM1U(*),LSTM2U(*),LSTN0U(*),LSTN1U(*)
    INTEGER LSTM0V(*),LSTM1V(*),LSTM2V(*),LSTN0V(*),LSTN1V(*)
    INTEGER LSTM0MU(*),LSTM1MU(*),LSTM2MU(*)
    INTEGER LSTM0MV(*),LSTM1MV(*),LSTM2MV(*)
    INTEGER LSTN0MU(*),LSTN1MU(*)
    INTEGER LSTN0MV(*),LSTN1MV(*),NN
    LOGICAL QRDF,QDDIP,QQDIP,QHD,QUSER
    LOGICAL DOIMAGES,QAWAT,QBWAT,QPREC
    !
    !     new variables for distance-matrix (minimum image approach)
    LOGICAL QMINI
    INTEGER LNMINI,LNMIN2
    real(chm_real4) MINI(*)
    !
    !
    INTEGER I,J,K,L,N,N2,M,M2,IMU,IMV,TMP,NLO,NHI,NLO2,NHI2
    INTEGER NIX,NIX2,NBRN,NBRNIM,NIMSLV,NIMSLU
    INTEGER NCUBXY,NTIM,MX,MY,MZ,NBR27,MADD(27)
    real(chm_real)  RHI2,XX,YY,ZZ,DDX,DDY,DDZ,XTMP
    LOGICAL QIM1,QIM2
    !
    !
    !----------------------------------------------------------------------
    !     Executable code begins here
    !
    call timer_start(T_setgrd)
    !
    !---- Compute frequently used intermediate scalar results -------------
    !
    RHI2 = RHI * RHI
    NCUBXY = NCUBX * NCUBY
    !
    !---- Prepare MADD array-----------------------------------------------
    !
    !     The MADD array is defined so that for a given cube M,
    !     the 27-cube neighborhood consists of the cubes M+MADD(1..27).
    !
    DO MX = -1, 1
       DO MY = -1, 1
          DO MZ = -1, 1
             NBR27 = (MX+1)*9 + (MY+1)*3 + (MZ+1) + 1
             MADD(NBR27) = MX + NCUBX * (MY + NCUBY * MZ)
          ENDDO
       ENDDO
    ENDDO
    !
    !     Remove duplicate offsets from MADD, or we will get duplicate
    !     atoms and therefore array-bound problems down the line.  Any
    !     element that is set to NCUBE is effectively deleted since later
    !     M+NCUBE is out of the range 1..NCUBE.
    !
    !     Crude method OK since max number of operations is 13 * 27 = 351
    !
    !
    DO NBR27 = 1, 27
       DO TMP = 1, NBR27-1
          IF (MADD(TMP) == MADD(NBR27)) MADD(NBR27) = NCUBE + 1
       END DO
    END DO
    !
    !---- Prepare a list of particles for each cube -----------------------
    !
    !     First, temporarily set up LSTN1 such that for a given atom N,
    !     LSTN1(N) is the index of the cube it's in.  This will be
    !     discarded after the next step.
    !
    !     We need to do this for solute and solvent separately so that
    !     we then can pick only cubes containing solute atoms and put
    !     solvent molecules from this + surounding cubes into the right
    !     shell
    !
    !     solute
    DO N = 1, NSHSLU
       K=SHSLU(N)
       LSTN1U(N) = ( INT((Z(K)-Z0)*RHIINV)  * NCUBY &
            + INT((Y(K)-Y0)*RHIINV)) * NCUBX &
            + INT((X(K)-X0)*RHIINV) + 1
    ENDDO
    !     solvent
    DO N = 1, NSHSLV
       K=SHSLV(N)
       LSTN1V(N) = ( INT((Z(K)-Z0)*RHIINV)  * NCUBY &
            + INT((Y(K)-Y0)*RHIINV)) * NCUBX &
            + INT((X(K)-X0)*RHIINV) + 1
    ENDDO
    !     images:
    IF(DOIMAGES) THEN
       !        solute
       DO N = (NSHSLU+1), NSHLUI
          K=SHSLU(N)
          LSTN1MU(N) = ( INT((Z(K)-Z0)*RHIINV)  * NCUBY &
               + INT((Y(K)-Y0)*RHIINV)) * NCUBX &
               + INT((X(K)-X0)*RHIINV) + 1
       ENDDO
       !        solvent
       DO N = (NSHSLV+1), NSHLVI
          K=SHSLV(N)
          LSTN1MV(N) = ( INT((Z(K)-Z0)*RHIINV)  * NCUBY &
               + INT((Y(K)-Y0)*RHIINV)) * NCUBX &
               + INT((X(K)-X0)*RHIINV) + 1
       ENDDO
    ENDIF
    !
    !     Invert LSTN1:  Set up LSTM0/LSTN0 as intertwined linked lists
    !     such that for a given cube M, you can recursively read off the
    !     atoms that it contains.  This will be discarded after the next
    !     step.
    !
    !     Vectorizable.
    DO M = 1, NCUBE
       LSTM0U(M) = 0
       LSTM0V(M) = 0
       LSTM0MU(M) = 0
       LSTM0MV(M) = 0
    END DO
    !
    !     solute
    DO N = 1, NSHSLU
       M = LSTN1U(N)
       LSTN0U(N) = LSTM0U(M)
       LSTM0U(M) = N
    ENDDO
    !     solvent
    DO N = 1, NSHSLV
       M = LSTN1V(N)
       LSTN0V(N) = LSTM0V(M)
       LSTM0V(M) = N
    ENDDO
    !     images
    IF(DOIMAGES) THEN
       !        solute
       DO N =  (NSHSLU+1), NSHLUI
          M = LSTN1MU(N)
          LSTN0MU(N) = LSTM0MU(M)
          LSTM0MU(M) = N
       END DO
       !        solvent
       DO N =  (NSHSLV+1), NSHLVI
          M = LSTN1MV(N)
          LSTN0MV(N) = LSTM0MV(M)
          LSTM0MV(M) = N
       END DO
    ENDIF
    !
    !     we set up the LSTN1's as lists of ATOMNUMBERS of the solute/solvent
    !     primary/image atoms in the given cube M
    !     we need begin and end pointers for the list since we don't use _all_
    !     atoms but only those in solute/solvent. so some cubes may not contain
    !     _any_ atoms of interest. thus LISTM1U/V(M-1) may sometimes be = 0 which
    !     is _not_ the beginning of atoms in cube M.
    !     so: LSTM1U/V/M...(M)  is the start pointer and
    !         LSTM2U/V/M...(M)  is the last element in the list
    !
    !     BEWARE: up to now the numbers in the LST.N.. lists were only
    !             pointers to the solute/solvent/... lists
    !             -> convert to real atom numbers now !
    !
    I  = 0
    J  = 0
    IMU = 0
    IMV = 0
    DO M = 1, NCUBE
       !
       !        .... primary .....
       !        solute
       N = LSTM0U(M)
       LSTM1U(M) = 0
       LSTM2U(M) = 0
       DO WHILE (N /= 0)
          I = I + 1
          !           set begin pointer if this is the first element
          IF(LSTM1U(M) <= 0) LSTM1U(M)=I
          LSTN1U(I) = SHSLU(N)
          N = LSTN0U(N)
       ENDDO
       IF(LSTM1U(M) > 0) LSTM2U(M) = I
       !        solvent
       N = LSTM0V(M)
       LSTM1V(M) = 0
       LSTM2V(M) = 0
       DO WHILE (N /= 0)
          J = J + 1
          !           set begin pointer if this is the first element
          IF(LSTM1V(M) <= 0) LSTM1V(M)=J
          LSTN1V(J) = SHSLV(N)
          N = LSTN0V(N)
       ENDDO
       IF(LSTM1V(M) > 0) LSTM2V(M) = J
       !
       !        .... images .....
       IF(DOIMAGES) THEN
          !           .... solute ....
          N = LSTM0MU(M)
          LSTM1MU(M) = 0
          LSTM2MU(M) = 0
          DO WHILE (N /= 0)
             IMU = IMU + 1
             !              set begin pointer if this is the first element
             IF(LSTM1MU(M) <= 0) LSTM1MU(M)=IMU
             LSTN1MU(IMU) = SHSLU(N)
             N = LSTN0MU(N)
          ENDDO
          IF(LSTM1MU(M) > 0) LSTM2MU(M) = IMU
          !           .... solvent ....
          N = LSTM0MV(M)
          LSTM1MV(M) = 0
          LSTM2MV(M) = 0
          DO WHILE (N /= 0)
             IMV = IMV + 1
             !              set begin pointer if this is the first element
             IF(LSTM1MV(M) <= 0) LSTM1MV(M)=IMV
             LSTN1MV(IMV) = SHSLV(N)
             N = LSTN0MV(N)
          ENDDO
          IF(LSTM1MV(M) > 0) LSTM2MV(M) = IMV
       ENDIF
       !
    ENDDO
    !
    !---- AND FINALLY: check all pairs ------------------------------------
    !
    !     Now we loop over all cubes M.
    !     If M contains PRIMARY SOLUTE atoms -> do this cube.
    !     set up two lists of PRIMARY/IMAGE SOLVENT atoms in the
    !     surrounding 27 cubes. then check all pairs with PRIMARY
    !     SOLVENT.
    !     If we are doing images -> check paris with IMAGE SOLVENT
    !     and if the cube contains IMAGE SOLUTE check with both
    !     types of SOLVENT
    !
    !
    !---- Start of major loop over cubes M --------------------------------
    !
    call timer_stop(T_setgrd)
    call timer_start(T_bldlst)
    !
    DO M = 1,NCUBE
       !
       !------- Determine range of atoms LSTN1(NLO..NHI) in center cube M ----
       !
       !        only do this cube if it contains PRIMARY solute atoms
       IF(LSTM1U(M) > 0) THEN
          !
          NLO=LSTM1U(M)
          NHI = LSTM2U(M)
          !
          !---------- Set LSTN0 equal to an array of neighbor atoms -------------
          !
          !           LSTN0 will be an array of the indices of atoms which are in
          !           the cube M and its neighbors.  In the outer loop, M2 takes on
          !           up to 27 indices of cubes adjacent to M.  For each M2,
          !           LSTN1(NLO2..NHI2) is the set of atoms in M2.  We traverse
          !           that list backwards so we end up with little runs of
          !           ascending order.
          !   
          !           Only build the list of solvent molecules since we only
          !           compute solute-solvent interactions
          !   
          NBRN = 0
          NBRNIM = 0
          DO NBR27 = 1, 27
             !
             !              Propose an M2, and if it's in range, then...
             M2 = M + MADD(NBR27)
             IF (M2 >= 1 .AND. M2 <= NCUBE) THEN ! ----------
                !
                !                 ... PRIMARY ...
                !                 Set NLO2..NHI2 equal to range of indices into LSTN1
                NLO2 = LSTM1V(M2)
                NHI2 = LSTM2V(M2)
                !
                !                 Loop over those indices, filling LSTN0 with atom numbers
                IF(NLO2 > 0) THEN
                   DO NIX2 = NHI2, NLO2, -1
                      NBRN = NBRN + 1
                      LSTN0V(NBRN) = LSTN1V(NIX2)
                   END DO
                ENDIF
                !
                !                 ... IMAGES ...
                IF(DOIMAGES) THEN
                   NLO2 = LSTM1MV(M2)
                   NHI2 = LSTM2MV(M2)
                   IF(NLO2 > 0) THEN
                      DO NIX2 = NHI2, NLO2, -1
                         NBRNIM = NBRNIM + 1
                         LSTN0MV(NBRNIM) = LSTN1MV(NIX2)
                      END DO
                   ENDIF
                ENDIF
                !
             END IF           ! ----------
          END DO
          !
          !
          !
          !---------- For every PRIMARY solute atom in M test all prim/img solvent
          !
          !           GO: loop over all PRIMARY SOLUTE atoms in the center cube
          !
          DO NIX = NLO, NHI
             !
             !              Record often-used parameters for solute atom N
             N = LSTN1U(NIX)
             XX = X(N)
             YY = Y(N)
             ZZ = Z(N)
             !
             !              Start this run of N2's (i.e. SOLVENT atoms)
             !------------- primary solvent ----------------------------------------
             DO I = 1, NBRN
                N2 = LSTN0V(I)
                IF(REVRES(N) /= REVRES(N2)) THEN
                   !
                   DDX = X(N2) - XX
                   DDY = Y(N2) - YY
                   DDZ = Z(N2) - ZZ
                   !
                   XTMP=DDX*DDX+DDY*DDY+DDZ*DDZ
                   IF(XTMP <= RHI2) THEN
                      XTMP=SQRT(XTMP)
                      CALL FILGXY(N,N2,N,N2,XTMP,DDX,DDY,DDZ,NBIN, &
                           DR,QAWAT,QBWAT, &
                           GXX,QRDF,QDDIP,QQDIP,QHD,QUSER,X,Y,Z, &
                           DIP,QMINI,MINI,LNMINI,LNMIN2)
                   ENDIF
                ENDIF
             END DO
             !
             !------------- End primary solvent ------------------------------------
             !------------- image solvent ------------------------------------------
             IF(DOIMAGES)THEN
                DO I = 1, NBRNIM
                   !
                   N2 = LSTN0MV(I)
                   J=IMATTR(N2)
                   IF(REVRES(N) /= REVRES(J)) THEN
                      DDX = X(N2) - XX
                      DDY = Y(N2) - YY
                      DDZ = Z(N2) - ZZ
                      XTMP=DDX*DDX+DDY*DDY+DDZ*DDZ
                      IF(XTMP <= RHI2) THEN
                         XTMP=SQRT(XTMP)
                         CALL FILGXY(N,N2,N,J,XTMP,DDX,DDY,DDZ,NBIN, &
                              DR,QAWAT,QBWAT, &
                              GXX,QRDF,QDDIP,QQDIP,QHD,QUSER,X,Y,Z, &
                              DIP,QMINI,MINI,LNMINI,LNMIN2)
                      ENDIF
                   ENDIF
                END DO
                !              end if(doimages)
             ENDIF
             !------------- End image solvent --------------------------------------
             !
             !           end loop over PRIMARY solute atoms (NIX)
          ENDDO
          !        end if(lstm1u(m) > 0) <- check if central cube contains solute
       ENDIF
       !
       !------- loop over all IMAGE solute (if wanted) -----------------------
       IF(DOIMAGES) THEN
          IF(LSTM1MU(M) > 0) THEN
             !
             NLO=LSTM1MU(M)
             NHI=LSTM2MU(M)
             !
             !------------- Set LSTN0 equal to an array of neighbor atoms ----------
             !
             NBRN = 0
             DO NBR27 = 1, 27
                M2 = M + MADD(NBR27)
                IF (M2 >= 1 .AND. M2 <= NCUBE) THEN ! ----------
                   !                    ... PRIMARY ...
                   NLO2 = LSTM1V(M2)
                   NHI2 = LSTM2V(M2)
                   IF(NLO2 > 0) THEN
                      DO NIX2 = NHI2, NLO2, -1
                         NBRN = NBRN + 1
                         LSTN0V(NBRN) = LSTN1V(NIX2)
                      END DO
                   ENDIF
                END IF        ! ----------
             END DO
             !
             !------------- For every IMAGE solute atom in M test all primary solvent
             !
             DO NIX = NLO, NHI
                N = LSTN1MU(NIX)
                K=IMATTR(N)
                XX = X(N)
                YY = Y(N)
                ZZ = Z(N)
                !
                !---------------- primary solvent -------------------------------------
                DO I = 1, NBRN
                   N2 = LSTN0V(I)
                   IF(REVRES(K) /= REVRES(N2)) THEN
                      DDX = X(N2) - XX
                      DDY = Y(N2) - YY
                      DDZ = Z(N2) - ZZ
                      !
                      XTMP=DDX*DDX+DDY*DDY+DDZ*DDZ
                      IF(XTMP <= RHI2) THEN
                         XTMP=SQRT(XTMP)
                         CALL FILGXY(N,N2,K,N2,XTMP,DDX,DDY,DDZ,NBIN, &
                              DR,QAWAT,QBWAT, &
                              GXX,QRDF,QDDIP,QQDIP,QHD,QUSER,X,Y,Z, &
                              DIP,QMINI,MINI,LNMINI,LNMIN2)
                      ENDIF
                   ENDIF
                END DO
                !---------------- End primary solvent ---------------------------------
                !              end loop over IMAGE solute
             ENDDO
             !           end if(LSTM1MU(M) > 0) <- check if this cube has IMG solute
          ENDIF
          !        end if(doimages)
       ENDIF
       !
       !     end loop over all cubes (M)
    ENDDO
    !     
    !---- End of major loop over cubes M ----------------------------------
    !
    call timer_stop(T_bldlst)
    !
    !---- we are done... --------------------------------------------------
    !     
    RETURN
  END SUBROUTINE GXYDIS
  !
  !----------------------------------------------------------------------
  !
  !
  SUBROUTINE GPYDIS(NAT,NATIM,X,Y,Z,RMAX, &
       XP,YP,ZP,QP,XDP,YDP,ZDP,LDP, &
       IMATTR,NBIN,DR, &
       NSET,NSETI,SET,SETSL,DIP, &
       REVRES,GXX,QRDF,QDDIP,QQDIP,QHD,QUSER, &
       RESL,NRESL, &
       DOIMAGES,QWAT,QPREC, &
       QMINI,MINI,LNMINI,LNMIN2)
    !
    !     Evaluates all distances of a point (X/Y/ZP) with a set
    !
    !     Tibor Rudas Jan 2003 - Jun 2003
    !
  use stream

    INTEGER NAT,NATIM
    real(chm_real)  X(*),Y(*),Z(*),RMAX
    real(chm_real)  XP,YP,ZP,QP,XDP,YDP,ZDP,LDP
    INTEGER IMATTR(*),NBIN
    real(chm_real) DR,GXX(8,*),DIP(4,*)
    INTEGER NSET,NSETI,SET(*),SETSL(*),REVRES(*)
    INTEGER RESL(*),NRESL
    LOGICAL QRDF,QDDIP,QQDIP,QHD,QUSER
    LOGICAL DOIMAGES,QWAT,QPREC
    !
    !     new variables for distance-matrix (minimum image approach)
    LOGICAL QMINI
    INTEGER LNMINI,LNMIN2
    real(chm_real4) MINI(*)
    !
    !
    INTEGER I,J,K,L,N
    real(chm_real)  RMAX2,DDX,DDY,DDZ,XTMP
    real(chm_real)  DDX2,DDY2,DDZ2
    !
    !
    !----------------------------------------------------------------------
    !     Executable code begins here
    !
    RMAX2=RMAX*RMAX
    !
    !     check all primary atoms (I think for point-set brute force
    !     is quickest... ???)
    !
    DO I=1,NSET
       N=SET(I)
       !        check if the residue of this atom should be omtitted
       !        (if it is in RESL -> omit)
       DO J=1,NRESL
          IF(RESL(J) == REVRES(N)) GOTO 100
       ENDDO
       DDX=X(N)-XP
       DDX2=DDX*DDX
       IF(DDX2 <= RMAX2) THEN
          DDY=Y(N)-YP
          DDY2=DDY*DDY
          IF(DDY2 <= RMAX2) THEN
             DDZ=Z(N)-ZP
             DDZ2=DDZ*DDZ
             IF(DDZ2 <= RMAX2) THEN
                XTMP=DDX2+DDY2+DDZ2
                IF(XTMP <= RMAX2) THEN
                   XTMP=SQRT(XTMP)
                   CALL FILGPY(N,N,XTMP,DDX,DDY,DDZ, &
                        XP,YP,ZP,QP,XDP,YDP,ZDP,LDP, &
                        NBIN,DR,QWAT, &
                        GXX,QRDF,QDDIP,QQDIP,QHD,QUSER,X,Y,Z, &
                        DIP,QMINI,MINI,LNMINI,LNMIN2)
                ENDIF
             ENDIF
          ENDIF
       ENDIF
100    CONTINUE
    ENDDO
    !
    !     check all image atoms (if requested)
    !
    IF(DOIMAGES) THEN
       DO I=(NSET+1),NSETI
          N=SET(I)
          K=IMATTR(N)
          DO J=1,NRESL
             IF(RESL(J) == REVRES(K)) GOTO 200
          ENDDO
          DDX=X(N)-XP
          DDX2=DDX*DDX
          IF(DDX2 <= RMAX2) THEN
             DDY=Y(N)-YP
             DDY2=DDY*DDY
             IF(DDY2 <= RMAX2) THEN
                DDZ=Z(N)-ZP
                DDZ2=DDZ*DDZ
                IF(DDZ2 <= RMAX2) THEN
                   XTMP=DDX2+DDY2+DDZ2
                   IF(XTMP <= RMAX2) THEN
                      XTMP=SQRT(XTMP)
                      CALL FILGPY(N,K,XTMP,DDX,DDY,DDZ, &
                           XP,YP,ZP,QP,XDP,YDP,ZDP,LDP, &
                           NBIN,DR,QWAT, &
                           GXX,QRDF,QDDIP,QQDIP,QHD,QUSER,X,Y,Z, &
                           DIP,QMINI,MINI,LNMINI,LNMIN2)
                   ENDIF
                ENDIF
             ENDIF
          ENDIF
200       CONTINUE
       ENDDO
    ENDIF
    !
    RETURN
  END SUBROUTINE GPYDIS
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE FILGOO(N1,N2,P1,P2,R,DDX,DDY,DDZ,NBIN,DR,QAWAT, &
       GXX,QRDF,QDDIP,QQDIP,QHD,QUSER,X,Y,Z,NN, &
       DIP,QMINI,MINI,LNMINI,LNMIN2)
    !
    !     decide what to put into the bin if GOODIS finds a pair N1-N2 worth
    !     to analyze. R is their distance, DDX/Y/Z the single distance
    !     components. NBIN is how many bins of spacing DR there are. XDIP=1
    !     mean RDF, =2 dipole-dipole correlation and =3 charge-dipole. QAWAT
    !     is true if N1/N2 are the oxygens of tip3. GOO, GOH and GHH are the
    !     bins to be filled .
    !     X,Y and Z are the coordinates
    !     N1 and N2 are the atom numbers and NN is 2 if we find this pair
    !     only once
    !
    !     13. 3. 2003: change: put calculation of bin + property into
    !                  unified subroutines which are called from all
    !                  distance + fill - parts
    !
    !     25. 3. 2003: P1,P2 are the primary atom numbers no matter if N1 or N2
    !                  are primary or image
    !
    !     Tibor Rudas Jan 2003 - Jun 2003
    !
    !     28. 7. 2003: change binning to have bin nr I which prints out at a
    !                  distance of I*DR to contain pairs from I-(DR/2) to I+(DR/2)
    !                  tr
    !
    !     29. 8. 2005: use logicals for desired function to allow simultaneous
    !                  calculation
    !
  use stream
  use number
  use psf

    INTEGER N1,N2,P1,P2,NBIN,NN
    real(chm_real) DDX,DDY,DDZ,R,DR,GXX(8,*)
    real(chm_real) X(*),Y(*),Z(*),DIP(4,*)
    LOGICAL QAWAT,QRDF,QDDIP,QQDIP,QHD,QUSER
    !
    !     new variables for distance-matrix (minimum image approach)
    LOGICAL QMINI
    INTEGER LNMINI,LNMIN2
    real(chm_real4) MINI(*)
    !
    INTEGER I,J,NB,NB2,A,B,AA,BB
    real(chm_real) DX,DY,DZ,R2,XTMP
    real(chm_real) X1D,Y1D,Z1D,L1D
    real(chm_real) X2D,Y2D,Z2D,L2D
    !
    !     the fill-functions
    !
    IF(.NOT.QMINI) THEN
       !
       !        switch to solana behaviour
       !tr         NB=INT(R/DR)+1
       NB=INT(R/DR + HALF)
       !
       IF(QRDF) THEN
          !           ie RDF
          IF((NB <= NBIN).AND.(NB > 0)) THEN
             GXX(1,NB)=GXX(1,NB)+RDFRDF()
             IF(NN == 2) GXX(1,NB)=GXX(1,NB)+RDFRDF()
          ENDIF
          IF(QAWAT) THEN
             A=N1
             B=N2
             DO J=1,2
                DO I=1,2
                   !                    O-H
                   DX=X(A)-X(B+I)
                   DY=Y(A)-Y(B+I)
                   DZ=Z(A)-Z(B+I)
                   R2=(DX*DX)+(DY*DY)+(DZ*DZ)
                   R2=SQRT(R2)
                   NB2=INT(R2/DR + HALF)
                   IF((NB2 <= NBIN).AND.(NB2 > 0)) THEN
                      GXX(2,NB2)=GXX(2,NB2)+RDFRDF()
                      IF(NN == 2) GXX(2,NB2)=GXX(2,NB2)+RDFRDF()
                   ENDIF
                ENDDO
                !                 H1-H2
                DX=X(A+1)-X(B+2)
                DY=Y(A+1)-Y(B+2)
                DZ=Z(A+1)-Z(B+2)
                R2=(DX*DX)+(DY*DY)+(DZ*DZ)
                R2=SQRT(R2)
                NB2=INT(R2/DR + HALF)
                IF((NB2 <= NBIN).AND.(NB2 > 0)) THEN
                   GXX(3,NB2)=GXX(3,NB2)+RDFRDF()
                   IF(NN == 2) GXX(3,NB2)=GXX(3,NB2)+RDFRDF()
                ENDIF
                !                 all left to do is H1-H1 or H2-H2
                DX=X(N1+J)-X(N2+J)
                DY=Y(N1+J)-Y(N2+J)
                DZ=Z(N1+J)-Z(N2+J)
                R2=(DX*DX)+(DY*DY)+(DZ*DZ)
                R2=SQRT(R2)
                NB2=INT(R2/DR + HALF)
                IF((NB2 <= NBIN).AND.(NB2 > 0)) THEN
                   GXX(3,NB2)=GXX(3,NB2)+RDFRDF()
                   IF(NN == 2) GXX(3,NB2)=GXX(3,NB2)+RDFRDF()
                ENDIF
                !                 now swap N1 & N2 for the second round
                A=N2
                B=N1
             ENDDO
          ENDIF
       ENDIF
       IF(QDDIP.AND.(NB <= NBIN).AND.(NB > 0)) THEN
          !           dipole-dipole correlation
          GXX(4,NB)=GXX(4,NB) &
               +DDIPRDF(DIP(1,N1),DIP(2,N1),DIP(3,N1), &
               DIP(1,N2),DIP(2,N2),DIP(3,N2))
          IF(NN == 2) &
               GXX(4,NB)=GXX(4,NB) &
               +DDIPRDF(DIP(1,N2),DIP(2,N2),DIP(3,N2), &
               DIP(1,N1),DIP(2,N1),DIP(3,N1))
       ENDIF
       IF(QQDIP.AND.(NB <= NBIN).AND.(NB > 0)) THEN
          !           Ox-charge-dipole correlation
          !           CAUTION: we must consider a->b and a<-b if we find this pair
          !               or only once. if we find a-b and b-a: consider only once
          !           ch(N1)-dip(N2)
          !           TODO: CHECK SIGN OF DDX/Y/Z FOR Q->DIP
          GXX(6,NB)=GXX(6,NB) &
               +QDIPRDF(R,DDX,DDY,DDZ,DIP(1,N2),DIP(2,N2),DIP(3,N2))
          IF(NN == 2) THEN
             !              we find this pair only once, so also calculate:
             !              dip(N1)-ch(N2)
             !              TODO: CHECK SIGN OF DDX/Y/Z FOR Q->DIP
             !              invert distance vector
             DX=(MINONE)*DDX
             DY=(MINONE)*DDY
             DZ=(MINONE)*DDZ
             GXX(6,NB)=GXX(6,NB) &
                  +QDIPRDF(R,DX,DY,DZ,DIP(1,N1),DIP(2,N1),DIP(3,N1))
          ENDIF
       ENDIF
       IF(QHD.AND.(NB <= NBIN).AND.(NB > 0)) THEN
          !           dipole dipole h_D function:
          !
          !               3(mu_i.r)(mu_j.r)
          !               -----------------  -  mu_i.mu_j
          !                      r^2
          !
          !              (where mu_i and mu_j are unit vectors)
          !
          GXX(7,NB)=GXX(7,NB)+HDRDF(R,DDX,DDY,DDZ, &
               DIP(1,N1),DIP(2,N1),DIP(3,N1), &
               DIP(1,N2),DIP(2,N2),DIP(3,N2))

          IF(NN == 2) THEN
             DX=(MINONE)*DDX
             DY=(MINONE)*DDY
             DZ=(MINONE)*DDZ
             GXX(7,NB)=GXX(7,NB)+HDRDF(R,DX,DY,DZ, &
                  DIP(1,N2),DIP(2,N2),DIP(3,N2), &
                  DIP(1,N1),DIP(2,N1),DIP(3,N1))
          ENDIF
       ENDIF
    ELSE
       !        MINIMUM IMAGE REQUESTED
       XTMP=ZERO
       IF(QRDF) THEN
          !           ie RDF or user defined function
          XTMP=RDFRDF()
          IF(NN == 2) XTMP=XTMP+RDFRDF()
          CALL SETMIN(P1,P2,R,XTMP,MINI,LNMINI,LNMIN2)
          IF(QAWAT) THEN
             A=N1
             B=N2
             AA=P1
             BB=P2
             DO J=1,2
                DO I=1,2
                   !                    O-H
                   DX=X(A)-X(B+I)
                   DY=Y(A)-Y(B+I)
                   DZ=Z(A)-Z(B+I)
                   R2=(DX*DX)+(DY*DY)+(DZ*DZ)
                   R2=SQRT(R2)
                   XTMP=RDFRDF()
                   IF(NN == 2) XTMP=XTMP+RDFRDF()
                   CALL SETMIN(AA,BB+I,R2,XTMP,MINI,LNMINI,LNMIN2)
                ENDDO
                !                 H1-H2
                DX=X(A+1)-X(B+2)
                DY=Y(A+1)-Y(B+2)
                DZ=Z(A+1)-Z(B+2)
                R2=(DX*DX)+(DY*DY)+(DZ*DZ)
                R2=SQRT(R2)
                XTMP=RDFRDF()
                IF(NN == 2) &
                     XTMP=XTMP+RDFRDF()
                CALL SETMIN(AA+1,BB+2,R2,XTMP,MINI,LNMINI,LNMIN2)
                !                 all left to do is H1-H1 or H2-H2
                DX=X(N1+J)-X(N2+J)
                DY=Y(N1+J)-Y(N2+J)
                DZ=Z(N1+J)-Z(N2+J)
                R2=(DX*DX)+(DY*DY)+(DZ*DZ)
                R2=SQRT(R2)
                XTMP=RDFRDF()
                IF(NN == 2) &
                     XTMP=XTMP+RDFRDF()
                CALL SETMIN(AA+J,BB+J,R2,XTMP,MINI,LNMINI,LNMIN2)
                !                 now swap N1 & N2 for the second round
                A=N2
                B=N1
                AA=P2
                BB=P1
             ENDDO
          ENDIF
       ENDIF
       IF(QDDIP) THEN
          !           dipole-dipole correlation
          XTMP=DDIPRDF(DIP(1,N1),DIP(2,N1),DIP(3,N1), &
               DIP(1,N2),DIP(2,N2),DIP(3,N2))
          IF(NN == 2) &
               XTMP=XTMP &
               +DDIPRDF(DIP(1,N2),DIP(2,N2),DIP(3,N2), &
               DIP(1,N1),DIP(2,N1),DIP(3,N1))
          CALL SETMIN(P1,P2,R,XTMP,MINI,LNMINI,LNMIN2)
       ENDIF
       IF(QQDIP) THEN
          !           Ox-charge-dipole correlation
          !           ch(N1)-dip(N2)
          XTMP=QDIPRDF(R,DDX,DDY,DDZ,DIP(1,N2),DIP(2,N2),DIP(3,N2))
          IF(NN == 2) THEN
             !              we find this pair only once, so also calculate:
             !              dip(N1)-ch(N2)
             !              invert distance vector
             DX=(MINONE)*DDX
             DY=(MINONE)*DDY
             DZ=(MINONE)*DDZ
             XTMP=XTMP &
                  +QDIPRDF(R,DX,DY,DZ,DIP(1,N1),DIP(2,N1),DIP(3,N1))
          ENDIF
          CALL SETMIN(P1,P2,R,XTMP,MINI,LNMINI,LNMIN2)
       ENDIF
       IF(QHD) THEN
          !           dipole dipole h_D function:
          !
          !               3(mu_i.r)(mu_j.r)
          !               -----------------  -  mu_i.mu_j
          !                      r^2
          !
          !              (where mu_i and mu_j are unit vectors)
          !
          XTMP=HDRDF(R,DDX,DDY,DDZ, &
               DIP(1,N1),DIP(2,N1),DIP(3,N1), &
               DIP(1,N2),DIP(2,N2),DIP(3,N2))
          IF(NN == 2) THEN
             DX=(MINONE)*DDX
             DY=(MINONE)*DDY
             DZ=(MINONE)*DDZ
             XTMP=XTMP+HDRDF(R,DX,DY,DZ, &
                  DIP(1,N2),DIP(2,N2),DIP(3,N2), &
                  DIP(1,N1),DIP(2,N1),DIP(3,N1))
          ENDIF
          CALL SETMIN(P1,P2,R,XTMP,MINI,LNMINI,LNMIN2)
       ENDIF
    ENDIF
    !
    RETURN
  END SUBROUTINE FILGOO
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE FILGXY(N1,N2,P1,P2,R,DDX,DDY,DDZ,NBIN,DR, &
       QAWAT,QBWAT, &
       GXX,QRDF,QDDIP,QQDIP,QHD,QUSER,X,Y,Z, &
       DIP,QMINI,MINI,LNMINI,LNMIN2)
    !
    !     decide what to put into the bin if GXYDIS finds a pair N1-N2 worth
    !     to analyze. R is their distance, DDX/Y/Z the single distance
    !     components. NBIN is how many bins of spacing DR there are. XDIP=1
    !     mean RDF, =2 dipole-dipole correlation and =3 for charge-dipole.
    !     QAWAT/QBWAT are true if N1/N2 are the oxygens of tip3.
    !     GXY, GXH and GHH are the bins to be filled and X,Y and Z the
    !     coordinates
    !
    !     13. 3. 2003: change: put calculation of bin + property into
    !                  unified subroutines which are called from all
    !                  distance + fill - parts
    !
    !     25. 3. 2003: P1,P2 are the primary atom numbers no matter if N1 or N2
    !                  are primary or image
    !
    !     Tibor Rudas Jan 2003 - Jun 2003
    !
    !     28. 7. 2003: change binning to have bin nr I which prints out at a
    !                  distance of I*DR to contain pairs from I-(DR/2) to I+(DR/2)
    !                  tr
    !
  use stream
  use number
  use psf

    INTEGER N1,N2,P1,P2,NBIN
    real(chm_real) DDX,DDY,DDZ,R,DR,GXX(8,*)
    real(chm_real) X(*),Y(*),Z(*),DIP(4,*)
    LOGICAL QAWAT,QBWAT,QRDF,QDDIP,QQDIP,QHD,QUSER
    !
    !     new variables for distance-matrix (minimum image approach)
    LOGICAL QMINI
    INTEGER LNMINI,LNMIN2
    real(chm_real4) MINI(*)
    !
    INTEGER I,J,NB,NB2,A,B,AA,BB
    real(chm_real) DX,DY,DZ,R2,XTMP
    real(chm_real) X1D,Y1D,Z1D,L1D
    real(chm_real) X2D,Y2D,Z2D,L2D
    !
    !     solana like binning
    !tr      NB=INT(R/DR)+1
    !
    IF(QAWAT.AND.QBWAT) THEN
       !        i.e. we are calculating water-water -> reuse
       !        FILGOO
       CALL FILGOO(N1,N2,P1,P2,R,DDX,DDY,DDZ,NBIN, &
            DR,QAWAT,GXX,QRDF,QDDIP,QQDIP,QHD,QUSER, &
            X,Y,Z,1,DIP,QMINI,MINI,LNMINI,LNMIN2)
    ELSE
       IF(.NOT.QMINI) THEN
          !
          NB=INT(R/DR + HALF)
          !
          IF(QRDF) THEN
             !              i.e. RDF
             IF((NB <= NBIN).AND.(NB > 0)) &
                  GXX(1,NB)=GXX(1,NB)+RDFRDF()
             IF(QAWAT.OR.QBWAT) THEN
                !                 i.e. we are calculating X-wat -> arrange for
                !                     B to hold the water and calculate X-H1/H2
                !                     into GOH
                A=N1
                B=N2
                IF(QAWAT) THEN
                   A=N2
                   B=N1
                ENDIF
                DO I=1,2
                   !                    X-H
                   DX=X(A)-X(B+I)
                   DY=Y(A)-Y(B+I)
                   DZ=Z(A)-Z(B+I)
                   R2=(DX*DX)+(DY*DY)+(DZ*DZ)
                   R2=SQRT(R2)
                   !                    solana like binning
                   !TR                  NB=INT(R/DR)+1
                   NB2=INT(R2/DR + HALF)
                   IF((NB2 <= NBIN).AND.(NB2 > 0)) &
                        GXX(2,NB2)=GXX(2,NB2)+RDFRDF()
                ENDDO
                !              endif (qawat.or.qbwat)
             ENDIF
             !           endif (qrdf)
          ENDIF
          IF(QDDIP.AND.(NB <= NBIN).AND.(NB > 0)) THEN
             GXX(4,NB)=GXX(4,NB) &
                  +DDIPRDF(DIP(1,N1),DIP(2,N1),DIP(3,N1), &
                  DIP(1,N2),DIP(2,N2),DIP(3,N2))
          ENDIF
          IF(QQDIP.AND.(NB <= NBIN).AND.(NB > 0)) THEN
             !              i.e. we are calculating X-wat charge-dipole corr
             !                   -> arrange for B to hold the water
             !                   TODO: CHECK SIGN OF DDX/Y/Z FOR Q->DIP
             GXX(6,NB)=GXX(6,NB) &
                  +QDIPRDF(R,DDX,DDY,DDZ, &
                  DIP(1,N2),DIP(2,N2),DIP(3,N2))
          ENDIF
          IF(QHD.AND.(NB <= NBIN).AND.(NB > 0)) THEN
             !              dipole dipole h_D function:
             GXX(7,NB)=GXX(7,NB)+HDRDF(R,DDX,DDY,DDZ, &
                  DIP(1,N1),DIP(2,N1),DIP(3,N1), &
                  DIP(1,N2),DIP(2,N2),DIP(3,N2))
          ENDIF
          !        if(.not.QMINI)
       ELSE
          !           MINIMUM IMAGE REQUESTED
          XTMP=ZERO
          !
          IF(QRDF) THEN
             !              i.e. RDF
             XTMP=RDFRDF()
             CALL SETMIN(P1,P2,R,XTMP,MINI,LNMINI,LNMIN2)
             IF(QAWAT.OR.QBWAT) THEN
                !                    i.e. we are calculating X-wat -> arrange for
                !                       B to hold the water and calculate X-H1/H2
                !                       into GOH
                A=N1
                B=N2
                AA=P1
                BB=P2
                IF(QAWAT) THEN
                   A=N2
                   B=N1
                   AA=P2
                   BB=P1
                ENDIF
                DO I=1,2
                   !                       X-H
                   DX=X(A)-X(B+I)
                   DY=Y(A)-Y(B+I)
                   DZ=Z(A)-Z(B+I)
                   R2=(DX*DX)+(DY*DY)+(DZ*DZ)
                   R2=SQRT(R2)
                   XTMP=RDFRDF()
                   CALL SETMIN (AA,BB+I,R2,XTMP, &
                        MINI,LNMINI,LNMIN2)
                ENDDO
                !                 endif (qawat.or.qbwat)
             ENDIF
             !           endif (qrdf)
          ENDIF
          IF(QDDIP) THEN
             XTMP=DDIPRDF(DIP(1,N1),DIP(2,N1),DIP(3,N1), &
                  DIP(1,N2),DIP(2,N2),DIP(3,N2))
             CALL SETMIN(P1,P2,R,XTMP,MINI,LNMINI,LNMIN2)
          ENDIF
          IF(QQDIP) THEN
             !              i.e. we are calculating X-wat charge-dipole corr
             !                   -> arrange for B to hold the water
             !                   TODO: CHECK SIGN OF DDX/Y/Z FOR Q->DIP
             XTMP=QDIPRDF(R,DDX,DDY,DDZ, &
                  DIP(1,N2),DIP(2,N2),DIP(3,N2))
             CALL SETMIN(P1,P2,R,XTMP,MINI,LNMINI,LNMIN2)
          ENDIF
          IF(QHD) THEN
             !              dipole dipole h_D function:
             XTMP=HDRDF(R,DDX,DDY,DDZ, &
                  DIP(1,N1),DIP(2,N1),DIP(3,N1), &
                  DIP(1,N2),DIP(2,N2),DIP(3,N2))
             CALL SETMIN(P1,P2,R,XTMP,MINI,LNMINI,LNMIN2)
          ENDIF
          !        if(.not. QMINI)
       ENDIF
    ENDIF
    !
    RETURN
  END SUBROUTINE FILGXY
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE FILGPY(N,P,R,DDX,DDY,DDZ, &
       XP,YP,ZP,QP,XDP,YDP,ZDP,LDP, &
       NBIN,DR,QWAT, &
       GXX,QRDF,QDDIP,QQDIP,QHD,QUSER,X,Y,Z, &
       DIP,QMINI,MINI,LNMINI,LNMIN2)
    !
    !     decide what to put into the bin if GPYDIS finds P-N worth
    !     to analyze. R is the distance, DDX/Y/Z the single distance
    !     components. NBIN is how many bins of spacing DR there are. XDIP=1
    !     means RDF, =2 dipole-dipole correlation and =3 for charge-dipole.
    !     QWAT is true if N is the oxygen of tip3.
    !     GPY, GPH are the bins to be filled and X,Y and Z the coordinates
    !     X/Y/ZP are the coordinates of a point with charge QP which has
    !     a dipole moment of lenght LDP and unit vector X/Y/ZDP
    !
    !     13. 3. 2003: change: put calculation of bin + property into
    !                  unified subroutines which are called from all
    !                  distance + fill - parts
    !
    !     25. 3. 2003: P is the primary atom number no matter if N is a
    !                  primary or image
    !
    !     Tibor Rudas Jan 2003 - Jun 2003
    !
    !     28. 7. 2003: change binning to have bin nr I which prints out at a
    !                  distance of I*DR to contain pairs from I-(DR/2) to I+(DR/2)
    !                  tr
    !
    !     Sep. 2005: switch from XDIP to logicals to allow for simultaneous
    !                calculation of functions
    !
  use stream
  use number
  use psf

    INTEGER N,P,NBIN
    real(chm_real) R,DDX,DDY,DDZ,DR,GXX(8,*)
    real(chm_real) XP,YP,ZP,QP,XDP,YDP,ZDP,LDP
    real(chm_real) X(*),Y(*),Z(*),DIP(4,*)
    LOGICAL QWAT,QRDF,QDDIP,QQDIP,QHD,QUSER
    !
    !     new variables for distance-matrix (minimum image approach)
    LOGICAL QMINI
    INTEGER LNMINI,LNMIN2
    real(chm_real4) MINI(*)
    !
    INTEGER I,NB,NB2
    real(chm_real) DX,DY,DZ,R2,XTMP
    real(chm_real) X1D,Y1D,Z1D,L1D
    !
    !     solana like binning
    !tr      NB=INT(R/DR)+1
    NB=INT(R/DR + HALF)
    !
    IF(QRDF) THEN
       !        i.e. RDF
       IF((NB <= NBIN).AND.(NB > 0)) &
            GXX(1,NB)=GXX(1,NB)+RDFRDF()
       IF(QWAT) THEN
          !           i.e. we are calculating p-wat
          DO I=1,2
             !              P-H
             DX=XP-X(N+I)
             DY=YP-Y(N+I)
             DZ=ZP-Z(N+I)
             R2=(DX*DX)+(DY*DY)+(DZ*DZ)
             R2=SQRT(R2)
             NB2=INT(R2/DR + HALF)
             IF((NB2 <= NBIN).AND.(NB2 > 0)) &
                  GXX(2,NB2)=GXX(2,NB2)+RDFRDF()
          ENDDO
       ENDIF
    ENDIF
    IF(QDDIP.AND.(NB <= NBIN).AND.(NB > 0)) THEN
       !        i.e. we are calculating P-wat dipole-dipole corr
       GXX(4,NB)=GXX(4,NB) &
            +DDIPRDF(XDP,YDP,ZDP,DIP(1,N),DIP(2,N),DIP(3,N))
    ENDIF
    IF(QQDIP.AND.(NB <= NBIN).AND.(NB > 0)) THEN
       !        i.e. we are calculating P-wat charge-dipole corr
       !        TODO: CHECK SIGN OF DDX/Y/Z FOR Q->DIP
       GXX(6,NB)=GXX(6,NB) &
            +QDIPRDF(R,DDX,DDY,DDZ,DIP(1,N),DIP(2,N),DIP(3,N))
    ENDIF
    IF(QHD.AND.(NB <= NBIN).AND.(NB > 0)) THEN
       !        dipole dipole h_D function:
       !        see FILGOO for details
       GXX(7,NB)=GXX(7,NB)+HDRDF(R,DDX,DDY,DDZ, &
            XDP,YDP,ZDP,DIP(1,N),DIP(2,N),DIP(3,N))
    ENDIF
    !     
    RETURN
  END SUBROUTINE FILGPY
  !
  !
  !----------------------------------------------------------------------
  ! THE SINGLE ROUTINES WHICH FILL THE BINS
  !----------------------------------------------------------------------
  !
  FUNCTION RDFRDF() result(rdfrdf_1)
    !
    !     Add 1 (I know this is futile - but consistent...)
    !
    !     Tibor Rudas 13. 3. 2003 - 
    !
  use number

    real(chm_real) BIN,rdfrdf_1
    !
    RDFRDF_1=ONE
    !
    RETURN
  END FUNCTION RDFRDF
  !
  !----------------------------------------------------------------------
  !
  FUNCTION DDIPRDF(X1D,Y1D,Z1D,X2D,Y2D,Z2D) result(ddiprdf_1)
    !
    !     Compute an anonymous Dipole-dipole correlation function
    !
    !     Tibor Rudas 13. 3. 2003
    !

    real(chm_real) X1D,Y1D,Z1D,X2D,Y2D,Z2D,ddiprdf_1
    !
    real(chm_real)  XTMP
    !
    !     calculate Cos(theta) = (u1 . u2)
    !             (u1,u2 = unit vectors of the dipole moment)
    DDIPRDF_1=(X1D*X2D)+(Y1D*Y2D)+(Z1D*Z2D)
    !
    RETURN
  END FUNCTION DDIPRDF
  !
  !----------------------------------------------------------------------
  !
  FUNCTION QDIPRDF(R,DX,DY,DZ,X1D,Y1D,Z1D) result(QDIPRDF_1)
    !
    !     Compute an anonymous Distance-dipole correlation function
    !
    !     Tibor Rudas 13. 3. 2003 - 
    !

    real(chm_real) R,DX,DY,DZ,QDIPRDF_1
    real(chm_real) X1D,Y1D,Z1D
    !
    !     calculate Cos(theta) = (D . u)/R
    !             (D = distance vector, u = unit dipole moment, R = distance)
    QDIPRDF_1=((DX*X1D)+(DY*Y1D)+(DZ*Z1D))/R
    !
    RETURN
  END FUNCTION QDIPRDF
  !
  !----------------------------------------------------------------------
  !
  FUNCTION HDRDF(R,DX,DY,DZ,X1D,Y1D,Z1D,X2D,Y2D,Z2D) result(hdrdf_1)
    !
    !     Compute an anonymous h_D function
    !     X1D/Y1D/Z1D AND X2D/Y2D/Z2D are the corresponding unit dipole moments
    !
    !     Tibor Rudas 13. 3. 2003
    !

    real(chm_real) BIN,hdrdf_1
    real(chm_real) R,DX,DY,DZ
    real(chm_real) X1D,Y1D,Z1D,X2D,Y2D,Z2D
    !
    real(chm_real)  XTMP
    !
    !     calculate h_D, where h_D is:
    !
    !           3 ( u1 . D ) ( u2 . D )
    !     h_D = -----------------------  -  ( u1 . u2 )
    !                     R^2
    !
    !     (u1,u2 = unit dipole moments, D = distance vector, R = distance)
    XTMP=(X1D*DX)+(Y1D*DY)+(Z1D*DZ)
    XTMP=XTMP*((X2D*DX)+(Y2D*DY)+(Z2D*DZ))
    XTMP=(3*XTMP)/(R*R)
    XTMP=XTMP-((X1D*X2D)+(Y1D*Y2D)+(Z1D*Z2D))
    !
    !     and fill the bin
    HDRDF_1=XTMP
    !
    RETURN
  END FUNCTION HDRDF
  !
  !----------------------------------------------------------------------
  !
  FUNCTION USRRDF(R,DX,DY,DZ, &
       X1,Y1,Z1,X1D,Y1D,Z1D,L1D, &
       X2,Y2,Z2,X2D,Y2D,Z2D,L2D) result(usrrdf_1)
    !
    !     Compute a user supplied function for two particles R (DX/DY/DZ) apart
    !
    !     The passed variables are:
    !
    !     R           - distance of the two particles (N1 and N2)
    !     DX/DY/DZ    - distance vector between the two particles
    !     X1/Y1/Z1    - absolute(!) position of N1 in space
    !     X1D/Y1D/Z1D - unit vector of the dipole moment of N1
    !     L1D         - length of the dipole moment of particle N1
    !     X2/Y2/Z2    - absolute(!) position of N2 in space
    !     X2D/Y2D/Z2D - unit vector of the dipole moment of N2
    !     L2D         - length of the dipole moment of particle N2
    !
    !     Another note: depending on SetA and SetB this routine may
    !     be called multiply for one pair:
    !        if we have X-Y it will be called once with the first series as BIN
    !        if we have X-WATER it will be:
    !                  called once for X-OH2 with the first series as BIN
    !                  called for X-H1 and X-H2 with the second series as BIN
    !        and if we have WATER-WATER it will be:
    !                  called once for OH2(1)-OH2(2) with the first series (1x GOO)
    !                  called for OH2(1)-H1(2) OH2(1)-H2(2) OH2(2)-H1(1)
    !                         and OH2(2)-H2(1) with the second series (4x GOH)
    !                  called for H1(1)-H1(2) H1(1)-H2(2) H2(1)-H1(2) and
    !                         H2(1)-H2(2) with the third series (4x GHH)
    !     in all these calls the diole moments will remain the same while
    !     the other parameters (xyz, r, dxyz) will be adapted
    !
    !     So if you wanted a RDF (which is already implemented :) you would
    !     simply add 1 to NBIN (and divide series 2 and 3 by 4 after completion).
    !     And if you wanted a re-implementation of the dipole-dipole function
    !     you would calculate (dip_1 . dip_2) and add this to the series (if
    !     one or both sets are WATER then the second and third series will just
    !     hold multiples of the first series. computing these is admittedly an
    !     unnecessary overhead but this routine is written to be most flexible.)
    !
    !     Tibor Rudas
    !
  use number

    real(chm_real) R,DX,DY,DZ,usrrdf_1
    real(chm_real) X1,Y1,Z1,X1D,Y1D,Z1D,L1D
    real(chm_real) X2,Y2,Z2,X2D,Y2D,Z2D,L2D
    !
    real(chm_real)  XTMP
    !
    !     here comes the interesting part:
    !     calculate whatever property you have in mind from the passed
    !     parameters int XTMP and it will be added to the right bin below
    XTMP=ONE
    !
    !     and add to BIN
    USRRDF_1=XTMP
    !
    RETURN
  END FUNCTION USRRDF
  !
  !
  !----------------------------------------------------------------------
  ! HELPER ROUTINES
  !----------------------------------------------------------------------
  !
  SUBROUTINE SETMIN(N1,N2,R,VAL,MINI,LNMINI,LNMIN2)
    !
    !     This routine fills the LNMINI x LNMINI matrix MINI (which has
    !     LNMIN2=LNMINI^2 entries; LNMINI should (for now) be set to NATOM).
    !
    !     If N1 > N2 then the distance R will be stored in MINI(N1,N2) and the
    !     corresponding value VAL in MINI(N2,N1) (this makes sense since fortran
    !     is column ordered and we want to loop over all distances in the end).
    !
    !     MINI is only updated if the stored distance is 0 or larger than R
    !
    !     Tibor Rudas Mar 2003 - Jun 2003
    !
  use stream
  use number
    INTEGER N1,N2,LNMINI,LNMIN2
    real(chm_real) R,VAL
    real(chm_real4) :: MINI(LNMINI,LNMINI)
    !
    INTEGER A,B
    !
    IF((N1 > LNMINI).OR.(N2.GT.LNMINI)) &
         CALL WRNDIE(-5,'<SETMIN>','ATOM NUMBER PROBLEM !!!.')
    !
    A=MAX(N1,N2)
    B=MIN(N1,N2)
    !
    IF((MINI(A,B) <= ZERO).OR.(R < MINI(A,B))) THEN
       MINI(A,B)=R
       MINI(B,A)=VAL
    ENDIF
    !
    RETURN
  END SUBROUTINE SETMIN
  !

  !----------------------------------------------------------------------
  ! BELOW ARE THE BRUTE FORCE ALGORITHMS FOR GOO AND GXY
  !----------------------------------------------------------------------
  !
  SUBROUTINE GOODI2(NAT,NATIM,X,Y,Z,RMAX,NBIN,DR, &
       NSOLV,NSOLVI,SOLVL,SOLVSL,DIP,REVRES,IMATTR, &
       QRDF,QDDIP,QQDIP,QHD,QUSER, &
       GXX,DOIMAGES,QWAT,NN, &
       QMINI,MINI,LNMINI,LNMIN2)
    !
    !     alternative version built with the K.I.S.S. principle:
    !     simple double-loop, nothing fancy...
    !
    !     Tibor Rudas 5. 12. 2002 - 
    !
  use stream

    INTEGER NAT,NATIM,NBIN,NN
    INTEGER NSOLV,NSOLVI,SOLVL(*),SOLVSL(*),REVRES(*),IMATTR(*)
    real(chm_real)  X(*),Y(*),Z(*),RMAX,DR
    real(chm_real) GXX(8,*),DIP(4,*)
    LOGICAL DOIMAGES,QWAT
    LOGICAL QRDF,QDDIP,QQDIP,QHD,QUSER
    !
    !     new variables for distance-matrix (minimum image approach)
    LOGICAL QMINI
    INTEGER LNMINI,LNMIN2
    real(chm_real4) MINI(*)
    !
    INTEGER I,J,K,N,N2
    real(chm_real)  RMAX2,XX,YY,ZZ,DDX,DDY,DDZ,XTMP
    real(chm_real)  DDX2,DDY2,DDZ2
    !
    !
    !----------------------------------------------------------------------
    !     Executable code begins here
    !
    !---- Compute frequently used intermediate scalar results -------------
    !
    RMAX2 = RMAX * RMAX
    !
    !     go up to nsolv -> for images
    DO I = 1, NSOLV
       !
       !        Record often-used parameters for solute atom N
       N = SOLVL(I)
       XX = X(N)
       YY = Y(N)
       ZZ = Z(N)
       !
       !         Start this run of N2's (i.e. atoms in the surrounding cube)
       !------- primary -----------------------------------------------------
       DO J = (I+1), NSOLV
          N2 = SOLVL(J)
          IF(REVRES(N) /= REVRES(N2)) THEN
             DDX = X(N2) - XX
             DDX2=DDX*DDX
             IF(DDX2 <= RMAX2) THEN
                DDY = Y(N2) - YY
                DDY2=DDY*DDY
                IF(DDY2 <= RMAX2) THEN
                   DDZ = Z(N2) - ZZ
                   DDZ2=DDZ*DDZ
                   IF(DDZ2 <= RMAX2) THEN
                      XTMP=DDX2+DDY2+DDZ2
                      IF(XTMP <= RMAX2) THEN
                         XTMP=SQRT(XTMP)
                         CALL FILGOO(N,N2,N,N2,XTMP,DDX,DDY,DDZ,NBIN, &
                              DR,QWAT,GXX,QRDF,QDDIP,QQDIP,QHD,QUSER, &
                              X,Y,Z,2,DIP,QMINI,MINI,LNMINI,LNMIN2)
                      ENDIF
                   ENDIF
                ENDIF
             ENDIF
          ENDIF
       ENDDO
       !------- End primary --------------------------------------------------
       !------- image --------------------------------------------------------
       IF(DOIMAGES)THEN
          DO J = (NSOLV+1), NSOLVI
             N2 = SOLVL(J)
             K=IMATTR(N2)
             IF(REVRES(N) /= REVRES(K)) THEN
                DDX = X(N2) - XX
                DDX2=DDX*DDX
                IF(DDX2 <= RMAX2) THEN
                   DDY = Y(N2) - YY
                   DDY2=DDY*DDY
                   IF(DDY2 <= RMAX2) THEN
                      DDZ = Z(N2) - ZZ
                      DDZ2=DDZ*DDZ
                      IF(DDZ2 <= RMAX2) THEN
                         XTMP=DDX2+DDY2+DDZ2
                         IF(XTMP <= RMAX2) THEN
                            XTMP=SQRT(XTMP)
                            ! base counting of prim-img pairs on redundancy
                            ! of images, see comment in GOODIS
                            CALL FILGOO(N,N2,N,K,XTMP,DDX,DDY,DDZ, &
                                 NBIN,DR,QWAT,GXX,QRDF,QDDIP,QQDIP, &
                                 QHD,QUSER,X,Y,Z,NN, &
                                 DIP,QMINI,MINI,LNMINI,LNMIN2)
                         ENDIF
                      ENDIF
                   ENDIF
                ENDIF
             ENDIF
          ENDDO
          !        end if(doimages)
       ENDIF
       !------- End image ----------------------------------------------------
       !
       !     end loop over PRIMARY atoms (I)
    ENDDO
    !
    RETURN
  END SUBROUTINE GOODI2
  !     
  !----------------------------------------------------------------------
  !
  SUBROUTINE GXYDI2(NAT,NATIM,X,Y,Z,RMAX,NBIN,DR, &
       NSOLU,NSOLUI,SOLUL,SOLUSL, &
       NSOLV,NSOLVI,SOLVL,SOLVSL,DIP, &
       REVRES,IMATTR,GXX,QRDF,QDDIP,QQDIP,QHD,QUSER, &
       DOIMAGES,QAWAT,QBWAT,NN, &
       QMINI,MINI,LNMINI,LNMIN2)
    !
    !     alternative version built with the K.I.S.S. principle:
    !     simple double-loop, nothing fancy...
    !
    !     Tibor Rudas 16. 12. 2002 - 
    !
  use stream

    INTEGER NAT,NATIM,NBIN,NN
    INTEGER NSOLU,NSOLUI,SOLUL(*),SOLUSL(*)
    INTEGER NSOLV,NSOLVI,SOLVL(*),SOLVSL(*),REVRES(*),IMATTR(*)
    real(chm_real)  X(*),Y(*),Z(*),RMAX,DR
    real(chm_real) GXX(8,*),DIP(4,*)
    LOGICAL QRDF,QDDIP,QQDIP,QHD,QUSER
    LOGICAL DOIMAGES,QAWAT,QBWAT
    !
    !     new variables for distance-matrix (minimum image approach)
    LOGICAL QMINI
    INTEGER LNMINI,LNMIN2
    real(chm_real4) MINI(*)
    !
    INTEGER I,J,K,L,N,N2
    real(chm_real)  RMAX2,XX,YY,ZZ,DDX,DDY,DDZ,XTMP
    real(chm_real)  DDX2,DDY2,DDZ2
    LOGICAL QIM1,QIM2
    !
    !
    !----------------------------------------------------------------------
    !     Executable code begins here
    !
    !---- Compute frequently used intermediate scalar results -------------
    !
    RMAX2 = RMAX * RMAX
    !
    !     go up to nsolv -> for images
    DO I = 1, NSOLU
       !
       !        Record often-used parameters for solute atom N
       N = SOLUL(I)
       XX = X(N)
       YY = Y(N)
       ZZ = Z(N)
       !
       !        Start this run of N2's
       !------- primary -----------------------------------------------------
       DO J = 1, NSOLV
          N2 = SOLVL(J)
          IF(REVRES(N) /= REVRES(N2)) THEN
             DDX = X(N2) - XX
             DDX2=DDX*DDX
             IF(DDX2 <= RMAX2) THEN
                DDY = Y(N2) - YY
                DDY2=DDY*DDY
                IF(DDY2 <= RMAX2) THEN
                   DDZ = Z(N2) - ZZ
                   DDZ2=DDZ*DDZ
                   IF(DDZ2 <= RMAX2) THEN
                      XTMP=DDX2+DDY2+DDZ2
                      IF(XTMP <= RMAX2) THEN
                         XTMP=SQRT(XTMP)
                         CALL FILGXY(N,N2,N,N2,XTMP,DDX,DDY,DDZ,NBIN, &
                              DR,QAWAT,QBWAT, &
                              GXX,QRDF,QDDIP,QQDIP,QHD,QUSER,X,Y,Z, &
                              DIP,QMINI,MINI,LNMINI,LNMIN2)
                      ENDIF
                   ENDIF
                ENDIF
             ENDIF
          ENDIF
       ENDDO
       !------- End primary --------------------------------------------------
       !------- image --------------------------------------------------------
       IF(DOIMAGES)THEN
          DO J = (NSOLV+1), NSOLVI
             N2 = SOLVL(J)
             K=IMATTR(N2)
             IF(REVRES(N) /= REVRES(K)) THEN
                DDX = X(N2) - XX
                DDX2=DDX*DDX
                IF(DDX2 <= RMAX2) THEN
                   DDY = Y(N2) - YY
                   DDY2=DDY*DDY
                   IF(DDY2 <= RMAX2) THEN
                      DDZ = Z(N2) - ZZ
                      DDZ2=DDZ*DDZ
                      IF(DDZ2 <= RMAX2) THEN
                         XTMP=DDX2+DDY2+DDZ2
                         IF(XTMP <= RMAX2) THEN
                            XTMP=SQRT(XTMP)
                            CALL FILGXY(N,N2,N,K,XTMP,DDX,DDY,DDZ, &
                                 NBIN,DR,QAWAT,QBWAT, &
                                 GXX,QRDF,QDDIP,QQDIP,QHD,QUSER,X,Y,Z, &
                                 DIP,QMINI,MINI,LNMINI,LNMIN2)
                         ENDIF
                      ENDIF
                   ENDIF
                ENDIF
             ENDIF
          ENDDO
          !        end if(doimages)
       ENDIF
       !------- End image ----------------------------------------------------
       !
       !     end loop over PRIMARY atoms (I)
    ENDDO
    !
    !
    !     AND NOW CHECK IMAGES IF WANTED
    !
    IF(DOIMAGES) THEN
       DO I = (NSOLU+1), NSOLUI
          !
          !        Record often-used parameters for solute atom N
          N = SOLUL(I)
          L=IMATTR(N)
          XX = X(N)
          YY = Y(N)
          ZZ = Z(N)
          !
          !        Start this run of N2's
          !------- primary -----------------------------------------------------
          DO J = 1, NSOLV
             N2 = SOLVL(J)
             IF(REVRES(L) /= REVRES(N2)) THEN
                DDX = X(N2) - XX
                DDX2=DDX*DDX
                IF(DDX2 <= RMAX2) THEN
                   DDY = Y(N2) - YY
                   DDY2=DDY*DDY
                   IF(DDY2 <= RMAX2) THEN
                      DDZ = Z(N2) - ZZ
                      DDZ2=DDZ*DDZ
                      IF(DDZ2 <= RMAX2) THEN
                         XTMP=DDX2+DDY2+DDZ2
                         IF(XTMP <= RMAX2) THEN
                            XTMP=SQRT(XTMP)
                            CALL FILGXY(N,N2,L,N2,XTMP,DDX,DDY,DDZ, &
                                 NBIN,DR,QAWAT,QBWAT, &
                                 GXX,QRDF,QDDIP,QQDIP,QHD,QUSER,X,Y,Z, &
                                 DIP,QMINI,MINI,LNMINI,LNMIN2)
                         ENDIF
                      ENDIF
                   ENDIF
                ENDIF
             ENDIF
          ENDDO
          !------- End primary --------------------------------------------------
          !        end loop over PRIMARY atoms (I)
       ENDDO
       !     end of (doimages)
    ENDIF
    ! 
    return
  end SUBROUTINE GXYDI2
#endif /* (rdfsol)*/

  SUBROUTINE NULL_RDFSL2
    RETURN
  END SUBROUTINE NULL_RDFSL2

end module rdfsol_subs

