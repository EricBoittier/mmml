module helix
  use chm_kinds
  use dimens_fcm
  implicit none

  ! The helix axis index data are kept here
  !
  !   MXHLX  - Maximum atoms to define the helix axis
  !   NHLXAT - Actual number of atoms defining the helix
  !   INDHLX - Pointer array to atoms of the helix.
  !
  ! This data must remain in common due to the use of ZXMIN.
  !
  INTEGER,save :: NHLXAT
  integer,PARAMETER :: MXHLX=250
  INTEGER,dimension(mxhlx),save :: INDHLX
#if KEY_CONSHELIX==1
  logical :: LRAXM, LRAYM, LRAZM
  integer :: REFAT
#endif 

contains

  SUBROUTINE HLXSPC(COMLYN,COMLEN)
    !
    ! Setup for parsing
    !
    use psf
    use coord
    use memory
    use select
    !
    character(len=*) COMLYN
    INTEGER COMLEN
    integer,allocatable,dimension(:) :: ISLCT
    !
    call chmalloc('helix.src','HLXSPC','ISLCT',NATOM,intg=ISLCT)
    CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
    CALL HLXSP1(COMLYN,COMLEN,NATOM,ISLCT)
    RETURN
  END SUBROUTINE HLXSPC

  SUBROUTINE HLXSP1(COMLYN,COMLEN,NATOM,ISLCT)
    !
    ! Parse specification for helix axis calculation
    !
    use select

    character(len=*) COMLYN
    INTEGER COMLEN,NATOM
    INTEGER ISLCT(*)
    !
    INTEGER I,J
    !
    NHLXAT=NSELCT(NATOM,ISLCT)
    IF(NHLXAT  >  MXHLX) THEN
       CALL WRNDIE(-3,'HLXSPC','Too many atoms to specify helix')
       NHLXAT=MXHLX
    ENDIF
    !
    J=0
    DO I=1,NATOM
       IF(ISLCT(I)  == 1) THEN
          J=J+1
          IF(J  <=  MXHLX) INDHLX(J)=I
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE HLXSP1

  SUBROUTINE HELIX1(A,R0,NDIG)
    !
    ! Calculate the normalized axis (A) and the perpendicular vector (R0)
    ! from the origin to A of the cylinder most closely approximating a
    ! helix on which the NHLXAT atoms w/ numbers in INDHLX(I),I=1,NHLXAT.
    ! (ie. x(INDHLX(I)) is x-coordinate of atom i)
    ! NDIG returns the estimated number of significant digits in the answer
    ! or -1 if an error occurred.
    ! Lennart Nilsson June 1987.
    ! Algorithm by J. Aqvist Computers & Chemistry
    ! Vol. 10, pp97-99, (1986).
    !
  use stream

    real(chm_real) A(3),R0(3)
    INTEGER NDIG
    !
    INTEGER,parameter :: NP=6
    real(chm_real) V(NP),G(NP),W(3*NP),H(NP*(NP+1)/2),F
    INTEGER NSIG,MAXFCN,IOPT,IER,I,IOUT
!    EXTERNAL HELAX
    !
    IF(PRNLEV > 2) WRITE(OUTU,17)
17  FORMAT(/' Helix analysis using the Aqvist algorithm.',/ &
         '    Computers & Chemistry Vol.10, pp97-99, (1986).')
    !
    CALL HELGUESS(A,R0,V,NHLXAT,INDHLX)
    ! Call double precison IMSL routine to do the optimization job
    ! Options used (could possibly be set from outside?)
    ! Use identity as initial guess for Hessian
    IOPT=0
    ! Try to get five digits accuracy in parameter estimates
    NSIG=5
    ! Number of function evaluataions
    MAXFCN=2500
    IOUT=-1
    IF(PRNLEV >= 8) IOUT=OUTU
    CALL ZXMIN(HELAX,NP,NSIG,MAXFCN,IOPT,V,H,G,F,W,IER)
    !
    NDIG=W(3)
    IF(IER > 0 .AND. WRNLEV >= 2) THEN
       WRITE(OUTU,*) '%HELIX-ERROR: ZXMIN reports IER:',IER
       WRITE(OUTU,*) '   Number of significant digits:',INT(W(3))
       NDIG=-IER
    ENDIF
    DO I=1,3
       R0(I)=V(I)
       A(I)=V(I+3)
    ENDDO
    RETURN
  END SUBROUTINE HELIX1

  SUBROUTINE HELGUESS(A,R0,V,NHLXAT,INDHLX)
    !
    ! Make initial guess of helix axis and distance to origin
    !
  use psf
  use coord

    INTEGER NHLXAT,INDHLX(NHLXAT)
    real(chm_real) V(*),A(*),R0(*)
    !
    real(chm_real) FAK,XCM,YCM,ZCM
    INTEGER I
    LOGICAL QGUESS
    !
    IF((A(1)**2+A(2)**2+A(3)**2)  >  0.005) THEN
       ! If user provides an initial guess it should be almost a unit vector
       ! otherwise we try to guess an initial position
       qguess=.true.
       ! Initial guesses have been provided for us
       DO I=1,3
          V(I)=R0(I)
          V(I+3)=A(I)
       ENDDO
    ELSE
       qguess=.false.
       !  Assume the atoms are somewhat ordered
       !  Let (for now) the axis be given by the end atoms
       !
       V(4)=X(INDHLX(NHLXAT))-X(INDHLX(1))
       V(5)=Y(INDHLX(NHLXAT))-Y(INDHLX(1))
       V(6)=Z(INDHLX(NHLXAT))-Z(INDHLX(1))
    ENDIF
    FAK=SQRT(V(4)**2 + V(5)**2 + V(6)**2)
    V(4)=V(4)/FAK
    V(5)=V(5)/FAK
    V(6)=V(6)/FAK
    IF(QGUESS) RETURN
    ! Now get R0 as the perpendicular vector to a line parallell to A
    ! going thru the center of mass of the atoms
    !
    XCM=0.0
    YCM=0.0
    ZCM=0.0
    DO I=1,NHLXAT
       XCM=XCM+X(INDHLX(I))
       YCM=YCM+Y(INDHLX(I))
       ZCM=ZCM+Z(INDHLX(I))
    ENDDO
    XCM=XCM/NHLXAT
    YCM=YCM/NHLXAT
    ZCM=ZCM/NHLXAT
    FAK=XCM*V(4)+YCM*V(5)+ZCM*V(6)
    V(1)=XCM-FAK*V(4)
    V(2)=YCM-FAK*V(5)
    V(3)=ZCM-FAK*V(6)
    RETURN
  END SUBROUTINE HELGUESS

  SUBROUTINE HELAX(N,V,F)
    !
    !  Compute the objective function, including constraints,
    !  for the helix axis calculation; NO LAGARANGE MULTIPLIERS,
    !  just a simple penalty function to keep A normalized.
    !
    !  NOTE: ONLY MAIN COORDINATES CAN BE USED
    !        DISTANCES FROM CYLINDER SURFACE FOR THE SELECTED ATOMS
    !        ARE STORED IN WMAIN
    !
  use coord

    INTEGER N
    real(chm_real) V(*),F
    !
    real(chm_real) DAVE,XX,YY,ZZ,XA
    INTEGER I,J
    !
    DAVE=0.0
    DO J=1,NHLXAT
       I=INDHLX(J)
       XA=X(I)*V(4)+Y(I)*V(5)+Z(I)*V(6)
       XX=X(I)-V(1)-XA*V(4)
       YY=Y(I)-V(2)-XA*V(5)
       ZZ=Z(I)-V(3)-XA*V(6)
       WMAIN(I)=SQRT(XX**2+YY**2+ZZ**2)
       DAVE=DAVE+WMAIN(I)
    ENDDO
    DAVE=DAVE/NHLXAT
    XA=V(1)*V(4)+V(2)*V(5)+V(3)*V(6)
    F=10.0*XA**2+10.0*(V(4)**2+V(5)**2+V(6)**2-1)**2
    DO I=1,NHLXAT
       F=F+(WMAIN(INDHLX(I))-DAVE)**2
    ENDDO
    RETURN
  END SUBROUTINE HELAX

  SUBROUTINE HELIX2(NHELIX,ISLCT,JSLCT,NATOM,X,Y,Z)
    !
    !  This routine computes helix axis and helix comparison analysis as in
    !  Chothia, Levitt, and Richardson, "Helix to helix packing in proteins,"
    !  JMB 145, P215-250 (1981), doi:10.1016/0022-2836(81)90341-7
    !
    !     By Bernard R. Brooks   1991
    !  
  use number
  use consta
  use stream
  use vector
  use conshelix_fcm
  use param_store, only: set_param

    ! . Passed variables.
    INTEGER NHELIX
    INTEGER ISLCT(*),JSLCT(*)
    INTEGER NATOM
    real(chm_real) X(*),Y(*),Z(*)
    !
    ! . Local variables.
#if KEY_CONSHELIX==1
    INTEGER,parameter :: MAXSLT=MXSLT
#else /**/
    INTEGER,parameter :: MAXSLT=50
#endif 
    !
    real(chm_real) PITOT(3,2),AVEC(3,2),BVEC(3,2),EVEC(3,2),RBAR(3,2)
    real(chm_real) PVEC(3,MAXSLT-2,2),QVEC(3,2),RVEC(3,MAXSLT,2)
    real(chm_real) SVEC(3,MAXSLT-1,2),STOT,SNORM
    real(chm_real) TVEC(3,2),XVEC(3),YVEC(3),ZVEC(3),PTOT(3,2)
    !
    real(chm_real) U(9),SCR(24),AMOM(6)
    real(chm_real) U11,U12,U22,DIST,SVAL(2),DET,DOT,TAU(2),OMEGA,W1,W2
    INTEGER NISLCT,NJSLCT,NIAT(2),IPT,L2,L3,NIVEC(2)
    INTEGER I,K,L,M,KM
    !
    LOGICAL LPRINT
    !
#if KEY_CONSHELIX==1
    real(chm_real) PRVEC(3,MAXSLT,MXHEL)
    real(chm_real) AK(3),VZ(3),AKDVZ,THETA,PREANG
    real(chm_real) LBOUN,UBOUN,ADOT(MAXSLT)
    real(chm_real) RUPR,NRPT,PCA(3),ANG,TMPANG,AA(3),BB(3),ABSAA,ABSBB
    INTEGER LDOT(MXHEL),UDOT(MXHEL)
    INTEGER II
    !
    real(chm_real) RALPHA(3),IDVEC(3),DVEC(3),CALPHA(3),DVXCA(3)
    real(chm_real) TMP,MIDVEC,MCALPHA,DDVXCAAK,DVDOTCAL,RHOANG
    real(chm_real) TTT(3,MAXSLT),AMOMT(6)
    real(chm_real) UUU(3,3),EVVV(3)
#endif 

    LPRINT=PRNLEV >= 2
    !
    !    Chothia, Levitt, and Richardson
    IF(LPRINT) WRITE(OUTU,17)
17  FORMAT(/' Helix analysis using the Chothia, Levitt,', &
         ' and Richardson algorithm.',/ &
         '    J.Mol.Biol. Vol.145, pp215-250 (1981).') 
    !
    ! Count selected atoms and get indexing array
    NISLCT=0
    DO I=1,NATOM
       IF(ISLCT(I) == 1) THEN
          NISLCT=NISLCT+1
          RVEC(1,NISLCT,1)=X(I)
          RVEC(2,NISLCT,1)=Y(I)
          RVEC(3,NISLCT,1)=Z(I)
       ENDIF
       IF(NISLCT >= MAXSLT) THEN
          CALL WRNDIE(0,'<HELIX2>','Too many atoms selected.')
          RETURN
       ENDIF
    ENDDO
    !
    ! make sure there are at least 4 atoms selected in each set.
    IF(NISLCT < 4) THEN
       CALL WRNDIE(0,'<HELIX2>','Too few atoms selected')
       RETURN
    ENDIF
    !
    NIAT(1)=NISLCT
    NIVEC(1)=NISLCT-2
    !
#if KEY_CONSHELIX==1
    ! for dna and beta-hairpin 
    ! for dna should be even
    IF(lldna.or.ldnam.or.lbhpm.or.llbhp) THEN
       DO I=1,NIAT(1)/2
          IF(MOD(I,2) == 0) THEN
             K=(NIAT(1)-1)-(I-2)
             RVEC(1,I,1)=RVEC(1,K,1)
             RVEC(2,I,1)=RVEC(2,K,1)
             RVEC(3,I,1)=RVEC(3,K,1)
          ELSE
             RVEC(1,I,1)=RVEC(1,I,1)
             RVEC(2,I,1)=RVEC(2,I,1)
             RVEC(3,I,1)=RVEC(3,I,1)
          ENDIF
       ENDDO
       NIAT(1)=NISLCT/2
       NIVEC(1)=NIAT(1)-2
    ENDIF

    IF(LBHGM.OR.LLBHG) THEN
       IF(MOD(NIAT(1),2).NE.0) THEN
          WRITE(6,*) 'Number of selected atoms are not even'
          STOP
       ENDIF
       DO I=1,NIAT(1)
        TTT(1,I)=RVEC(1,I,1)
        TTT(2,I)=RVEC(2,I,1)
        TTT(3,I)=RVEC(3,I,1)
       ENDDO
       DO I=1,NIAT(1)
          IF(MOD(I,2) == 0) THEN
             K=NIAT(1)-(I-2)/2
             RVEC(1,I,1)=TTT(1,K)
             RVEC(2,I,1)=TTT(2,K)
             RVEC(3,I,1)=TTT(3,K)
          ELSE
             K=I-(I-1)/2
             RVEC(1,I,1)=TTT(1,K)
             RVEC(2,I,1)=TTT(2,K)
             RVEC(3,I,1)=TTT(3,K)
          ENDIF
       ENDDO
       NIAT(1)=NISLCT
       NIVEC(1)=NIAT(1)
    ENDIF
#endif 

    ! Count selected atoms and get indexing array
    IF(NHELIX == 2) THEN
       NJSLCT=0
       DO I=1,NATOM
          IF(JSLCT(I) == 1) THEN
             NJSLCT=NJSLCT+1
             RVEC(1,NJSLCT,2)=X(I)
             RVEC(2,NJSLCT,2)=Y(I)
             RVEC(3,NJSLCT,2)=Z(I)
          ENDIF
          IF(NJSLCT >= MAXSLT) THEN
             CALL WRNDIE(0,'<HELIX2>','Too many atoms selected.')
             RETURN
          ENDIF
       ENDDO
       !
       ! make sure there are at least 4 atoms selected in each set.
       IF(NJSLCT < 4) THEN
          CALL WRNDIE(0,'<HELIX2>','Too few atoms selected')
          RETURN
       ENDIF
       !
       NIAT(2)=NJSLCT
       NIVEC(2)=NJSLCT-2
       !
#if KEY_CONSHELIX==1
      ! for dna 
      ! for dna should be even
      IF(LLDNA.OR.LDNAM.OR.LBHPM.OR.LLBHP) THEN
         DO I=1,NIAT(2)/2
            IF(MOD(I,2) == 0) THEN
               K=(NIAT(2)-1)-(I-2)
               RVEC(1,I,2)=RVEC(1,K,2)
               RVEC(2,I,2)=RVEC(2,K,2)
               RVEC(3,I,2)=RVEC(3,K,2)
            ELSE
               RVEC(1,I,2)=RVEC(1,I,2)
               RVEC(2,I,2)=RVEC(2,I,2)
               RVEC(3,I,2)=RVEC(3,I,2)
            ENDIF
         ENDDO
         NIAT(2)=NJSLCT/2
         NIVEC(2)=NIAT(2)-2
      ENDIF

      IF(LBHGM.OR.LLBHG) THEN
         IF(MOD(NIAT(2),2).NE.0) THEN
            WRITE(6,*) 'Number of selected atoms are not even'
            STOP
         ENDIF
         DO I=1,NIAT(2)
            TTT(1,I)=RVEC(1,I,2)
            TTT(2,I)=RVEC(2,I,2)
            TTT(3,I)=RVEC(3,I,2)
         ENDDO
         DO I=1,NIAT(2)
           IF(MOD(I,2) == 0) THEN
              K=NIAT(2)-(I-2)/2
              RVEC(1,I,2)=TTT(1,K)
              RVEC(2,I,2)=TTT(2,K)
              RVEC(3,I,2)=TTT(3,K)
           ELSE
              K=I-(I-1)/2
              RVEC(1,I,2)=TTT(1,K)
              RVEC(2,I,2)=TTT(2,K)
              RVEC(3,I,2)=TTT(3,K)
           ENDIF
        ENDDO
        NIAT(2)=NJSLCT
        NIVEC(2)=NIAT(2)
      ENDIF
#endif 
    !
    ENDIF
    !
    ! do both helicies
    DO K=1,NHELIX
       !
#if KEY_CONSHELIX==1
       IF(LBHGM.OR.LLBHG) THEN
          ! Set up adjacent point vectors and compute average length.
          ! for bhairpin mode in general moment of inertia
          STOT=ZERO
          DO L=1,3
             DO I=1,NIVEC(K)-1
                SVEC(L,I,K)=RVEC(L,I+1,K)-RVEC(L,I,K)
                STOT=STOT+SVEC(L,I,K)**2
             ENDDO
          ENDDO
          SNORM=SQRT(STOT/(NIVEC(K)-1))

          ELSE
#endif 
       ! Set up adjacent point vectors and compute average length.
       STOT=ZERO
       DO L=1,3
          DO I=1,NIVEC(K)+1
             SVEC(L,I,K)=RVEC(L,I+1,K)-RVEC(L,I,K)
             STOT=STOT+SVEC(L,I,K)**2
          ENDDO
       ENDDO
       SNORM=SQRT(STOT/(NIVEC(K)+1))
       !
#if KEY_CONSHELIX==1
       ENDIF
#endif 
       !
       ! Check to make sure all vector lengths are roughly equal.
#if KEY_CONSHELIX==1
       ! skip helix2 warning message in case of dna and bhp mode
       IF(LLDNA.OR.LDNAM.OR.LBHPM.OR.LLBHP.OR.LBHGM.OR.LLBHG) THEN

       ELSE
#endif 
       DO I=1,NIVEC(K)+1
          STOT=ZERO
          DO L=1,3
             STOT=STOT+SVEC(L,I,K)**2
          ENDDO
          IF(ABS(SQRT(STOT)-SNORM)/SNORM > 0.1) THEN
             IF(WRNLEV >= 2) WRITE(OUTU,19) I,SQRT(STOT),SNORM
19           FORMAT(' HELIX2: Warning. Vector',I4, &
                  ' has a length more than 10% from the average.', &
                  2F14.5)
          ENDIF
       ENDDO
#if KEY_CONSHELIX==1
    ENDIF
#endif 
#if KEY_CONSHELIX==1
    IF(LBHGM.OR.LLBHG) THEN
       ! Actually, PVEC is not necessary in bhg mode
       !
       ! Set up axis normal direction vectors.
       DO L=1,3 
          PTOT(L,K)=ZERO
          DO I=1,NIVEC(K)-2
             PVEC(L,I,K)=SVEC(L,I+1,K)-SVEC(L,I,K)
             PTOT(L,K)=PTOT(L,K)+PVEC(L,I,K)
          ENDDO
       ENDDO
       !
       ! Get average vectors from a reference position.
       DO L=1,3
          PITOT(L,K)=PTOT(L,K)/(NIVEC(K)-2)
          DO I=1,NIVEC(K)-2
             PVEC(L,I,K)=PVEC(L,I,K)-PITOT(L,K)
          ENDDO
       ENDDO
    ELSE
#endif 
       !
       ! Set up axis normal direction vectors.
       DO L=1,3 
          PTOT(L,K)=ZERO
          DO I=1,NIVEC(K)
             PVEC(L,I,K)=SVEC(L,I+1,K)-SVEC(L,I,K)
             PTOT(L,K)=PTOT(L,K)+PVEC(L,I,K)
          ENDDO
       ENDDO
       !
       ! Get average vectors from a reference position.
       DO L=1,3
          PITOT(L,K)=PTOT(L,K)/NIVEC(K)
          DO I=1,NIVEC(K)
             PVEC(L,I,K)=PVEC(L,I,K)-PITOT(L,K)
          ENDDO
       ENDDO
       !
#if KEY_CONSHELIX==1
      ENDIF
#endif 
     !
#if KEY_CONSHELIX==1
     ! get projection vectors of pvec onto x-y plane
     ! DNA mode for bhairpin or dna structure
     !        using helical moment of inertia
     ! BHA mode for bhairpin using general moment of inertia
     IF(LBHGM.OR.LLBHG.OR.LBHPM.OR.LLBHP) THEN
        ! get average R vector. (average vector)
        DO L=1,3
           RBAR(L,K)=ZERO
           DO I=1,NIAT(K)
              RBAR(L,K)=RBAR(L,K)+RVEC(L,I,K)
           ENDDO
           RBAR(L,K)=RBAR(L,K)/NIAT(K)
        ENDDO

        IPT=1
        DO M=1,3
           DO L=M,3
              AMOMt(IPT)=ZERO
              DO I=1,NIAT(K)
                 AMOMt(IPT)=AMOMt(IPT)+(RVEC(L,I,K)-RBAR(L,K))*(RVEC(M,I,K)-RBAR(M,K))
              ENDDO
              IPT=IPT+1
           ENDDO
        ENDDO

        AMOM(1)=AMOMT(4)+AMOMT(6)
        AMOM(2)=-AMOMT(2)
        AMOM(3)=-AMOMT(3)
        AMOM(4)=AMOMT(1)+AMOMT(6)
        AMOM(5)=-AMOMT(5)
        AMOM(6)=AMOMT(1)+AMOMT(4)
     ELSE
#endif 
       !
       IPT=1
       DO M=1,3
          DO L=M,3
             AMOM(IPT)=ZERO
             DO I=1,NIVEC(K)
                AMOM(IPT)=AMOM(IPT)+PVEC(L,I,K)*PVEC(M,I,K)
             ENDDO
             IPT=IPT+1
          ENDDO
       ENDDO
       !
#if KEY_CONSHELIX==1
    ENDIF
#endif 
       IF(LPRINT) WRITE(OUTU,105) K,AMOM
105    FORMAT('   MOMENTS FOR HELIX',I3/3F16.8,/16X,2F16.8, &
            /32X,F16.8/)
       !
       CALL DIAGQ(3,3,AMOM,U,SCR(4),SCR(7),SCR(10),SCR(13),SCR(1), &
            SCR(16),SCR(19),SCR(22),0)
       !
       DO L=1,3
          AVEC(L,K)=U(L)
       ENDDO
       !
#if KEY_CONSHELIX==1
       DO I=1,9
          CU(I,K)=U(I)
       ENDDO
       DO I=1,3
          CEV(I,K)=SCR(I)
       ENDDO
#endif 
       !
       DOT=ZERO
       DO L=1,3
          DOT=DOT+U(L)*(RVEC(L,NIAT(K),K)-RVEC(L,1,K))
       ENDDO
       IF(DOT < ZERO) THEN
          DO L=1,3
             AVEC(L,K)=-U(L)
          ENDDO
       ENDIF
       !
#if KEY_CONSHELIX==1
       IF(LBHGM.OR.LLBHG.OR.LBHPM.OR.LLBHP) THEN
          IF(PRNLEV >= 6) THEN
             IF(DOT < ZERO) THEN
                WRITE(6,*) 'PAXIS changed'
             ELSE
                WRITE(6,*) 'PAXIS not changed'
             ENDIF
          ENDIF
          ! general MOI method ,in particular beta-hairpin
          IF(DOT < ZERO) THEN
             DO I=1,9
                !eigen vectors
                CU(I,K)=-U(I)
             ENDDO
             DO I=1,3
                !eigen values
                CEV(I,K)=CEV(I,K)
                !principal eigen vector
                AVEC(I,K)=-U(I)
             ENDDO
          ENDIF
       ELSE
          ! generally helix definition MOI dosen't change
          IF(DOT < ZERO) THEN
             DO I=1,9
                CU(I,K)=-CU(I,K)
             ENDDO
          ENDIF
       ENDIF
#endif 
       !
       IF(LPRINT) WRITE(OUTU,107) 'MOMENTS',K,(SCR(L),L=1,3)
       IF(LPRINT) WRITE(OUTU,107) 'AXIS   ',K,(AVEC(L,K),L=1,3)
107    FORMAT(5X,A,' for helix',I3,5F16.5/)
       !
       ! get average R vector.
       DO L=1,3
          RBAR(L,K)=ZERO
          DO I=1,NIAT(K)
             RBAR(L,K)=RBAR(L,K)+RVEC(L,I,K)
          ENDDO
          RBAR(L,K)=RBAR(L,K)/NIAT(K)
       ENDDO
       IF(LPRINT) WRITE(OUTU,107) 'CENTER ',K,(RBAR(L,K),L=1,3)
       ! get B (begin) vector
       DOT=ZERO
       DO L=1,3
          DOT=DOT+AVEC(L,K)*(RVEC(L,1,K)-RBAR(L,K))
       ENDDO
       DO L=1,3
          BVEC(L,K)=RBAR(L,K)+DOT*AVEC(L,K)
       ENDDO
       IF(LPRINT) WRITE(OUTU,107) 'BEGIN  ',K,(BVEC(L,K),L=1,3)
       ! get E (end) vector
       DOT=ZERO
       DO L=1,3
          DOT=DOT+AVEC(L,K)*(RVEC(L,NIAT(K),K)-RBAR(L,K))
       ENDDO
       DO L=1,3
          EVEC(L,K)=RBAR(L,K)+DOT*AVEC(L,K)
       ENDDO
       IF(LPRINT) WRITE(OUTU,107) 'END    ',K,(EVEC(L,K),L=1,3)
       !
#if KEY_CONSHELIX==1 /*consh*/
       IF(LOHEL(OCHNUM)) THEN
          DO L=1,3
             ! always k = 1
             CRVEC(L,K)=RBAR(L,K) ! center
             CBVEC(L,K)=BVEC(L,K) ! begin
             CEVEC(L,K)=EVEC(L,K) ! end
             CAVEC(L,K)=AVEC(L,K) ! axis

             DO I=1,NIAT(K)
                CVEC(L,I,K)=RVEC(L,I,K)
             ENDDO

             IF(LBHGM.OR.LLBHG) THEN
                DO I=1,NIVEC(K)-2
                   CPVEC(L,I,K)=PVEC(L,I,K)
                ENDDO
             ELSE
                DO I=1,NIVEC(K)
                   CPVEC(L,I,K)=PVEC(L,I,K)
                ENDDO
             ENDIF
          ENDDO
       ENDIF

       ! selected atom projected point vector onto axis
       ! # of selected atom :  niat(1) - 1st sel., niat(2) 2nd sel.
       DO I=1,NIAT(K)
          DOT=ZERO
          ! x,y,z (three) coordinate
          DO L=1,3
             ! a dot ( rm - r ) : inner product
             DOT=DOT+AVEC(L,K)*(RVEC(L,I,K)-RBAR(L,K))
          ENDDO

          ADOT(I)=DOT

          DO L=1,3
             PRVEC(L,I,K)=RBAR(L,K)+DOT*AVEC(L,K)
          ENDDO
       ENDDO

       IF(.NOT.QCONSH) THEN
          ! GET the boundary c_alpha
          DO I=1,NIAT(K)
            IF(ADOT(I) >= 0.D0) THEN
               LDOT(K)=I-1
               UDOT(K)=I
               EXIT
            ENDIF
          ENDDO

          DO I=1,NIAT(K)
             DOT=ZERO
             DO L=1,3
                DOT=DOT+(PRVEC(L,I,K)-RVEC(L,I,K))*(PRVEC(L,I,K)-RVEC(L,I,K))
             ENDDO
          ENDDO
          ! save tilt angle information to substitution value
          ! cos(T) = a_k (dot) v_z / (|a_k| |v_z|)
          DO L=1,3
             AK(L)=AVEC(L,K)
             IF(L == 3) THEN
                VZ(L)=1.D0
             ELSE
                VZ(L)=0.D0
             ENDIF
          ENDDO
          !         WRITE(6,*) ak(1),ak(2),ak(3),vz(1),vz(2),vz(3)
          ! akdvz
          ! = a_k (dot) v_z
          CALL DOTPR(AK,VZ,3,AKDVZ)
          ! UNIT vector a_k and v_z
          THETA=ACOS(AKDVZ)*RADDEG
      
          IF(PRNLEV >= 6) WRITE(OUTU,*) 'akdvz',AKDVZ,THETA
          call set_param('TANG',THETA)
          ! precession angle
          ! s=sqrt(x*x+y*y)
          ! x >= 0  phi = asin(y/s)
          ! x <  0  phi = pi-asin(y/s)
          ! or phi2 = atan (y/x)
          PREANG = atan2(AK(2), AK(1)) * RADDEG
          call set_param('PREA',PREANG)
          ! rise up per residue (d) and number of residue per turn (n) @hh/5.pdf and my note
          ! d_k = a_k (o) ( rmk - r1k ) / m --> d_k = a_k (o) ( rmk - r1k ) / ( m - 1 )
          DOT=ZERO
          DO L=1,3
             DOT=DOT+AVEC(L,K)*(RVEC(L,NIAT(K),K)-RVEC(L,1,K))
          ENDDO
          RUPR=DOT/(NIAT(K)-1)
          call set_param('RUPR',RUPR)
          ! n_k = m / ( sum_1^(m-1) theta_i,i+1) (rad) * 2PI 
          ANG=ZERO
          DO I=1,NIAT(K)-1
             DOT=ZERO
             II=I+1
             DO L=1,3
                DOT=DOT+(RVEC(L,II,K)-PRVEC(L,I,K))*AVEC(L,K)
             ENDDO
             DO L=1,3
                PCA(L)=RVEC(L,II,K)-DOT*AVEC(L,K)
                AA(L)=PCA(L)-PRVEC(L,I,K)
                BB(L)=RVEC(L,I,K)-PRVEC(L,I,K)
             ENDDO
             ABSAA=SQRT(AA(1)**2+AA(2)**2+AA(3)**2)
             ABSBB=SQRT(BB(1)**2+BB(2)**2+BB(3)**2)
             TMPANG = dot_product(AA/ABSAA, BB/ABSBB)
             if (TMPANG < MINONE) TMPANG = MINONE
             if (TMPANG > ONE) TMPANG = ONE
             ANG = ANG + ACOS(TMPANG)
          ENDDO
          NRPT=(NIAT(K)-1)/ANG*TWOPI
          call set_param('NRPT',NRPT)
          !if(.not.qconsh) THEN
      ENDIF
      ! projected vecotr, prvec
      IF(LOHEL(OCHNUM)) THEN
         DO l=1,3
            DO I=1,NIAT(K)
               CPRVEC(l,I,K)=PRVEC(l,I,K)
            ENDDO
         ENDDO
      ENDIF
      ! check if a helix is along to z axis
      TMP=AVEC(3,K)*1.D0
      IF(ABS(TMP) > 0.999999999D0) THEN
         LAZAXIS(OCHNUM)=.TRUE.
      ENDIF
      IF(.NOT.QCONSH) THEN
         ! Rotation angle calculation
         RALPHA(1)=RVEC(1,REFAT,K)
         RALPHA(2)=RVEC(2,REFAT,K)
         RALPHA(3)=RVEC(3,REFAT,K)

         IF(LRAXM) THEN
            DO I=1,3
               IF(I == 1) THEN
                  VZ(I)=1.D0
               ELSE
                  VZ(I)=0.D0
               ENDIF
            ENDDO
            ! RotY
         ELSEIF(LRAYM) THEN
            DO I=1,3
               IF(I == 2) THEN
                  VZ(I)=1.D0
               ELSE
                  VZ(I)=0.D0
               ENDIF
            ENDDO
            ! RotZ or default
         ELSE
            DO I=1,3
               IF(I == 3) THEN
                  VZ(I)=1.D0
               ELSE
                  VZ(I)=0.D0
               ENDIF
            ENDDO
         ENDIF
         !      
         IDVEC(1)=(AK(1)*AK(1)-1.D0)*VZ(1)+AK(1)*AK(2)*VZ(2)+AK(1)*AK(3)*VZ(3)
         IDVEC(2)=AK(2)*AK(1)*VZ(1)+(AK(2)*AK(2)-1.D0)*VZ(2)+AK(2)*AK(3)*VZ(3)
         IDVEC(3)=AK(3)*AK(1)*VZ(1)+AK(3)*AK(2)*VZ(2)+(AK(3)*AK(3)-1.D0)*VZ(3)
         ! RotX
         !   IDVEC(1)=AK(1)*AK(1)-1.D0
         !   IDVEC(2)=AK(1)*AK(2)
         !   IDVEC(3)=AK(1)*AK(3)
         ! RotY
         !   IDVEC(1)=AK(2)*AK(1)
         !   IDVEC(2)=AK(2)*AK(2)-1.D0
         !   IDVEC(3)=AK(2)*AK(3)
         ! RotZ (Default)
         !   IDVEC(1)=AK(3)*AK(1)
         !   IDVEC(2)=AK(3)*AK(2)
         !   IDVEC(3)=AK(3)*AK(3)-1.D0
         !
         TMP=0.D0
         DO I=1,3
            TMP=TMP+IDVEC(I)*IDVEC(I)
         ENDDO
         if (TMP == ZERO) then
            DVEC = ZERO
         else
            MIDVEC = SQRT(TMP)
            DVEC = IDVEC / MIDVEC
            TMP = ZERO
         endif
         DO I=1,3
            CALPHA(I)=RALPHA(I)-PRVEC(I,REFAT,K)
            TMP=TMP+CALPHA(I)*CALPHA(I)
         ENDDO
         MCALPHA=SQRT(TMP)
         ! UNIT vector of calpha
         DO I=1,3
            CALPHA(I)=CALPHA(I)/MCALPHA
         ENDDO
         ! sign of rho angle 
         ! 0 <= dvec(x)calpha (dot) ak <= 1 ( <= 90 degs)  (-) sign
         ! -1 <= dvec(x)calpha (dot) ak < 0 ( > 90 degs)  (+) sign
         !        ddvxcaak 
         CALL CROSS3(DVEC,CALPHA,DVXCA)
         CALL DOTPR(DVXCA,AK,3,DDVXCAAK)
         ! RHO angle
         ! cos(rho)= dvec (dot) calpha
         CALL DOTPR(DVEC,CALPHA,3,TMP)
         DVDOTCAL=TMP
         RHOANG=ACOS(TMP)*RADDEG
   
         IF(DDVXCAAK > ZERO) RHOANG=-RHOANG
   
         call set_param('ROTA',RHOANG)
         !axis and begin vector substitution
         ! first one
         ! axis
         call set_param('FXAV',AVEC(1,1))
         call set_param('FYAV',AVEC(2,1))
         call set_param('FZAV',AVEC(3,1))
         ! 2nd axis and 3rd axis
         call set_param('FXA2',CU(4,1))
         call set_param('FYA2',CU(5,1))
         call set_param('FZA2',CU(6,1))
         call set_param('FXA3',CU(7,1))
         call set_param('FYA3',CU(8,1))
         call set_param('FZA3',CU(9,1))
         ! begin
         call set_param('FXBE',BVEC(1,1))
         call set_param('FYBE',BVEC(2,1))
         call set_param('FZBE',BVEC(3,1))
         ! end
         call set_param('FXEN',EVEC(1,1))
         call set_param('FYEN',EVEC(2,1))
         call set_param('FZEN',EVEC(3,1))
         ! center
         call set_param('FXCE',RBAR(1,1))
         call set_param('FYCE',RBAR(2,1))
         call set_param('FZCE',RBAR(3,1))
         ! boundary
         CALL set_param('FLBO',LDOT(1))
         CALL set_param('FUBO',UDOT(1))
         !
         call set_param('FXPR',PRVEC(1,5,1))
         call set_param('FYPR',PRVEC(2,5,1))
         call set_param('FZPR',PRVEC(3,5,1))
      ENDIF
#endif /* (consh)*/
    !
    ENDDO
    ! end of helix loops
    !
#if KEY_CONSHELIX==1
    LDNAM=.FALSE.
    LBHPM=.FALSE.
    LBHGM=.FALSE.
#endif 
    IF(NHELIX /= 2) RETURN
    !
    ! solve for closest approach vector.
    W1=ZERO
    W2=ZERO
    U11=ZERO
    U22=ZERO
    U12=ZERO
    DO L=1,3
       W1=W1+(BVEC(L,1)-BVEC(L,2))*(EVEC(L,1)-BVEC(L,1))
       W2=W2+(BVEC(L,1)-BVEC(L,2))*(EVEC(L,2)-BVEC(L,2))
       U11=U11+(EVEC(L,1)-BVEC(L,1))*(EVEC(L,1)-BVEC(L,1))
       U22=U22+(EVEC(L,2)-BVEC(L,2))*(EVEC(L,2)-BVEC(L,2))
       U12=U12+(EVEC(L,2)-BVEC(L,2))*(EVEC(L,1)-BVEC(L,1))
    ENDDO
    DET=U11*U22-U12**2
    !
    IF(ABS(DET) < 0.00001) THEN
       DOT=ZERO
       DO L=1,3
          DOT=DOT+AVEC(L,1)*(RBAR(L,2)-RBAR(L,1))
       ENDDO
       DIST=ZERO
       DO L=1,3
          DIST=DIST+(RBAR(L,2)-RBAR(L,1)-DOT*AVEC(L,1))**2
       ENDDO
       DIST=SQRT(DIST)
       IF(LPRINT) WRITE(OUTU,137) DIST
137    FORMAT('   Helicies are parallel. Closest approach distance', &
            ' is;',F12.5)
#if KEY_CONSHELIX==1
       LPARL(OCHNUM)=.TRUE.

       IF(.NOT.QCONSH) THEN
          !
          !axis and begin vector substitution
          ! first one
          call set_param('FXAV',AVEC(1,1))
          call set_param('FYAV',AVEC(2,1))
          call set_param('FZAV',AVEC(3,1))
          ! 2nd axis and 3rd axis
          call set_param('FXA2',CU(4,1))
          call set_param('FYA2',CU(5,1))
          call set_param('FZA2',CU(6,1))
          call set_param('FXA3',CU(7,1))
          call set_param('FYA3',CU(8,1))
          call set_param('FZA3',CU(9,1))
          ! begin and end
          call set_param('FXBE',BVEC(1,1))
          call set_param('FYBE',BVEC(2,1))
          call set_param('FZBE',BVEC(3,1))
          call set_param('FXEN',EVEC(1,1))
          call set_param('FYEN',EVEC(2,1))
          call set_param('FZEN',EVEC(3,1))
          ! center
          call set_param('FXCE',RBAR(1,1))
          call set_param('FYCE',RBAR(2,1))
          call set_param('FZCE',RBAR(3,1))
          ! boundary
          CALL set_param('FLBO',LDOT(1))
          CALL set_param('FUBO',UDOT(1))
          ! second one
          call set_param('SXAV',AVEC(1,2))
          call set_param('SYAV',AVEC(2,2))
          call set_param('SZAV',AVEC(3,2))
          ! 2nd axis and 3rd axis
          call set_param('SXA2',CU(4,2))
          call set_param('SYA2',CU(5,2))
          call set_param('SZA2',CU(6,2))
          call set_param('SXA3',CU(7,2))
          call set_param('SYA3',CU(8,2))
          call set_param('SZA3',CU(9,2))
          ! begin and end
          call set_param('SXBE',BVEC(1,2))
          call set_param('SYBE',BVEC(2,2))
          call set_param('SZBE',BVEC(3,2))
          call set_param('SXEN',EVEC(1,2))
          call set_param('SYEN',EVEC(2,2))
          call set_param('SZEN',EVEC(3,2))
          ! center
          call set_param('SXCE',RBAR(1,2))
          call set_param('SYCE',RBAR(2,2))
          call set_param('SZCE',RBAR(3,2))
          ! boundary
          call set_param('SLBO',LDOT(2))
          call set_param('SUBO',UDOT(2))
          ! minimum distance
          call set_param('MIND',DIST)
       ENDIF
       !
#endif 
       !
       RETURN
    ENDIF
    !
    SVAL(1)=(W2*U12-W1*U22)/DET
    SVAL(2)=(W2*U11-W1*U12)/DET
    DO K=1,2
       IF(LPRINT) WRITE(OUTU,107) 'S**    ',K,SVAL(K)
    ENDDO
    !
    ! Put limits on SVAL and compute closest vector
    !
    DO K=1,2
       IF(SVAL(K) >= ONE .OR. SVAL(K) <= ZERO) THEN
          IF(SVAL(K) > ONE) SVAL(K)=ONE
          IF(SVAL(K) < ZERO) SVAL(K)=ZERO
          KM=3-K
          W1=ZERO
          U11=ZERO
          DO L=1,3
             TVEC(L,K)=BVEC(L,K)+SVAL(K)*(EVEC(L,K)-BVEC(L,K))
             W1=W1+(TVEC(L,K)-BVEC(L,KM))*(EVEC(L,KM)-BVEC(L,KM))
             U11=U11+(EVEC(L,KM)-BVEC(L,KM))*(EVEC(L,KM)-BVEC(L,KM))
          ENDDO
          W1=W1/U11
          IF(LPRINT) WRITE(OUTU,107) 'S(adj) ',KM,W1
          IF(W1 > ONE) W1=ONE
          IF(W1 < ZERO) W1=ZERO
          SVAL(KM)=W1
#if KEY_CONSHELIX==1
          LLIMIT(OCHNUM)=.TRUE.
#endif /* */
      ENDIF
    ENDDO
    !
    DO K=1,2
       IF(LPRINT) WRITE(OUTU,107) 'S      ',K,SVAL(K)
       DO L=1,3
          TVEC(L,K)=BVEC(L,K)+SVAL(K)*(EVEC(L,K)-BVEC(L,K))
       ENDDO
       IF(LPRINT) WRITE(OUTU,107) 'CLOSE  ',K,(TVEC(L,K),L=1,3)
    ENDDO
    !
#if KEY_CONSHELIX==1
    CSVAL(1)=SVAL(1)
    CSVAL(2)=SVAL(2)
#endif 
    !
    DIST=ZERO
    DO L=1,3
       XVEC(L)=TVEC(L,2)-TVEC(L,1)
       DIST=DIST+XVEC(L)**2
    ENDDO
    DIST=SQRT(DIST)
    !
    DOT=ZERO
    DO L=1,3
       DOT=DOT+AVEC(L,1)*AVEC(L,2)
    ENDDO
    OMEGA=ACOS(DOT)*RADDEG
#if KEY_CONSHELIX==1
    IF(LHMODC) THEN 
       IF(LPRINT) WRITE(OUTU,107) 'ANGLE  ',0,OMEGA 
    ELSE
       IF(LPRINT) WRITE(OUTU,107) 'ANGLEH ',0,180.d0-OMEGA 
    ENDIF
    ! HINGE ANGLE
    IF(LHMODC) THEN 
       call set_param('HANG',omega)
    ELSE
       call set_param('HANG',180.d0-omega)
    ENDIF
#endif 
    !
    IF(LPRINT) WRITE(OUTU,107) 'ANGLE  ',0,OMEGA 
    !
    IF(DIST < 0.00001) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,139)
139    FORMAT('   Helicies intersect. No closest approach vector.')
       RETURN
    ENDIF
    !
    DO L=1,3
       XVEC(L)=XVEC(L)/DIST
    ENDDO
    IF(LPRINT) WRITE(OUTU,107) 'XVEC   ',0,(XVEC(L),L=1,3)
    !
    DO L=1,3
       ZVEC(L)=AVEC(L,1)+AVEC(L,2)
    ENDDO
    !CC   IF(LPRINT) WRITE(OUTU,107) 'ZVEC*  ',0,(ZVEC(L),L=1,3)
#if KEY_CONSHELIX==1
    !IF(LPRINT) WRITE(OUTU,107) 'ZVEC*  ',0,(ZVEC(L),L=1,3)
#endif 
    !
    DOT=ZERO
    DO L=1,3
       L2=L+1
       IF(L2 > 3) L2=L2-3
       L3=L+2
       IF(L3 > 3) L3=L3-3
       YVEC(L)=XVEC(L2)*ZVEC(L3)-ZVEC(L2)*XVEC(L3)
       DOT=DOT+YVEC(L)**2
    ENDDO
    DOT=SQRT(DOT)
    DO L=1,3
       YVEC(L)=YVEC(L)/DOT
    ENDDO
    IF(LPRINT) WRITE(OUTU,107) 'YVEC   ',0,(YVEC(L),L=1,3),DOT
    !
    DOT=ZERO
    DO L=1,3
       L2=L+1
       IF(L2 > 3) L2=L2-3
       L3=L+2
       IF(L3 > 3) L3=L3-3
       ZVEC(L)=YVEC(L2)*XVEC(L3)-XVEC(L2)*YVEC(L3)
       DOT=DOT+ZVEC(L)**2
    ENDDO
    DOT=SQRT(DOT)
    DO L=1,3
       ZVEC(L)=ZVEC(L)/DOT
    ENDDO
    IF(LPRINT) WRITE(OUTU,107) 'ZVEC   ',0,(ZVEC(L),L=1,3),DOT
    !
    DO K=1,2
       !
       DOT=ZERO
       DO L=1,3
          DOT=DOT+XVEC(L)*AVEC(L,K)
       ENDDO
       TAU(K)=ASIN(DOT)*RADDEG
       IF(LPRINT) WRITE(OUTU,107) 'TAU    ',K,TAU(K)
       !
       DOT=ZERO
       DO L=1,3
          L2=L+1
          IF(L2 > 3) L2=L2-3
          L3=L+2
          IF(L3 > 3) L3=L3-3
          QVEC(L,K)=XVEC(L2)*AVEC(L3,K)-AVEC(L2,K)*XVEC(L3)
          DOT=DOT+QVEC(L,K)**2
       ENDDO
       DOT=SQRT(DOT)
       DO L=1,3
          QVEC(L,K)=QVEC(L,K)/DOT
       ENDDO
       !CC      IF(LPRINT) WRITE(OUTU,107) 'QVEC   ',K,(QVEC(L,K),L=1,3),DOT
       !
    ENDDO
    !
    DOT=ZERO
    DO L=1,3
       DOT=DOT+QVEC(L,1)*QVEC(L,2)
    ENDDO
    OMEGA=ACOS(DOT)*RADDEG
    !
    DOT=ZERO
    DO L=1,3
       L2=L+1
       IF(L2 > 3) L2=L2-3
       L3=L+2
       IF(L3 > 3) L3=L3-3
       DOT=DOT+XVEC(L)*(QVEC(L2,1)*QVEC(L3,2)-QVEC(L2,2)*QVEC(L3,1))
    ENDDO
    IF(DOT < ZERO) OMEGA=-OMEGA
    !
    IF(LPRINT) WRITE(OUTU,107) 'OMEGA  ',0,OMEGA 
    IF(LPRINT) WRITE(OUTU,107) 'DIST   ',0,DIST
    !
#if KEY_CONSHELIX==1
    ! NUMBER OF HELIX
    DO K=1,NHELIX
       DO L=1,3
          CRVEC(L,K)=RBAR(L,K) ! center
          CBVEC(L,K)=BVEC(L,K) ! begin
          CEVEC(L,K)=EVEC(L,K) ! end
          CAVEC(L,K)=AVEC(L,K) ! axis
          ! tvec (The close position vectors of two axis)
          CTVEC(L,K)=TVEC(L,K)
          DO I=1,NIAT(K)
             ! projected vector prvec
             CPRVEC(L,I,K)=PRVEC(L,I,K)
             ! selected atoms
             CVEC(L,I,K)=RVEC(L,I,K)
          ENDDO
          IF(LBHGM.OR.LLBHG) THEN
             DO I=1,NIVEC(K)-2
                CPVEC(L,I,K)=PVEC(L,I,K)
             ENDDO
          ELSE
             DO I=1,NIVEC(K)
                CPVEC(L,I,K)=PVEC(L,I,K)
             ENDDO
          ENDIF
       ENDDO
    ENDDO
    IF(.not.qconsh) THEN
       !axis and begin vector substitution
       ! first one
       call set_param('FXAV',AVEC(1,1))
       call set_param('FYAV',AVEC(2,1))
       call set_param('FZAV',AVEC(3,1))
       call set_param('FXBE',BVEC(1,1))
       call set_param('FYBE',BVEC(2,1))
       call set_param('FZBE',BVEC(3,1))
       call set_param('FXEN',EVEC(1,1))
       call set_param('FYEN',EVEC(2,1))
       call set_param('FZEN',EVEC(3,1))
       ! center
       call set_param('FXCE',RBAR(1,1))
       call set_param('FYCE',RBAR(2,1))
       call set_param('FZCE',RBAR(3,1))
       ! boundary
       CALL set_param('FLBO',LDOT(1))
       CALL set_param('FUBO',UDOT(1))
       ! second one
       call set_param('SXAV',AVEC(1,2))
       call set_param('SYAV',AVEC(2,2))
       call set_param('SZAV',AVEC(3,2))
       call set_param('SXBE',BVEC(1,2))
       call set_param('SYBE',BVEC(2,2))
       call set_param('SZBE',BVEC(3,2))
       call set_param('SXEN',EVEC(1,2))
       call set_param('SYEN',EVEC(2,2))
       call set_param('SZEN',EVEC(3,2))
       ! center
       call set_param('SXCE',RBAR(1,2))
       call set_param('SYCE',RBAR(2,2))
       call set_param('SZCE',RBAR(3,2))
       ! BOUNDARY
       call set_param('SLBO',LDOT(2))
       call set_param('SUBO',UDOT(2))
       ! CTVECTOR (minimum distance)
       call set_param('FXTV',CTVEC(1,1))
       call set_param('FYTV',CTVEC(2,1))
       call set_param('FZTV',CTVEC(3,1))
       call set_param('SXTV',CTVEC(1,2))
       call set_param('SYTV',CTVEC(2,2))
       call set_param('SZTV',CTVEC(3,2))
       ! minimum distance and crossing angle
       call set_param('MIND',DIST)
       call set_param('OMEG',OMEGA)
    ENDIF

#endif 
    !
    RETURN
  END SUBROUTINE HELIX2
end module helix

