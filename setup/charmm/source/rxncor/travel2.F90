module travelsub2
  ! TRAVEL file-names facilitate compilation with previous versions of CHARMM.

  !******************************************************************************
  !                                                                             *
  !           TReK: a program for Trajectory REfinement and Kinematics.         *
  !                                                                             *
  !                        Version 2.10 , July  5-2003.                         *
  !                                                                             *
  !******************************************************************************
  !         Please report problems or send suggestions to the author:           *
  !                                                                             *
  !                              Stefan Fischer                                 *
  !                         Tel. (49)6221-548879                                *
  !                e-mail: stefan.fischer@iwr.uni-heidelberg.de                 *
  !                                                                             *
  !            Check for application examples and bug-fixes under :             *
  !                                                                             *
  !            http://www.iwr.uni-heidelberg.de/iwr/biocomp/fischer             *
  !                                                                             *
  !******************************************************************************

contains

#if KEY_TRAVEL==1 /*travel_main*/

  SUBROUTINE ADDLST( LPAKED, SRTENE,SRTIDX, NMAXI )
    use chm_kinds
    implicit none

    LOGICAL   LPAKED
    INTEGER   SRTIDX(*), NMAXI
    real(chm_real)    SRTENE(*)

    IF (LPAKED) THEN
       CALL ADDBOT( SRTENE, SRTIDX, NMAXI, .FALSE.)
    ELSE
       SRTENE(1) = SRTENE(NMAXI)
       SRTIDX(1) = SRTIDX(NMAXI)
       NMAXI  = NMAXI - 1
       LPAKED = .TRUE.
       CALL ADDTOP( SRTENE, SRTIDX, NMAXI, .FALSE.)
    ENDIF

    RETURN
  END SUBROUTINE ADDLST

  SUBROUTINE UPDPRE(I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
       NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX, &
       IDNEXT,LSCAN1,LSCAN2,LSCAN3,SEGSTP,NATOM,VERBEU, &
       LENSEG,NEXMIN,PREMIN,XBAS,YBAS,ZBAS )


    use chm_kinds
    use chm_types
    use number
    implicit none


    LOGICAL   LPAKED,QUP(*),QDOWN(*)
    INTEGER   I, SRTIDX(*),NMAXI,STPMAX(*),NLINMN(*),SADMIN
    real(chm_real)    SRTENE(*),PTGRAD(*),SADGRD,PTENE(*),ENEMAX(*)

    LOGICAL   LSCAN1,LSCAN2,LSCAN3
    INTEGER   IDNEXT(*),NEXMIN(*),PREMIN(*)
    type(chm_array) XBAS(:),YBAS(:),ZBAS(:)
    INTEGER   NATOM,VERBEU
    real(chm_real)    LENSEG(*),SEGSTP(*)

    INTEGER   OLDSMX, I7000
    INTEGER, PARAMETER :: BIGINT=999999
    real(chm_real)    OLDEMX

    OLDSMX = STPMAX(I)
    OLDEMX = ENEMAX(I)

    CALL SGSCAN( 0,I,IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
         LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
         SRTENE,SRTIDX,SEGSTP,PTGRAD, &
         XBAS,YBAS,ZBAS, NATOM,VERBEU )

    IF (OLDSMX == 0) THEN
       IF (STPMAX(I) == 0) THEN
          NMAXI = NMAXI - 1
       ELSEIF (STPMAX(I) < 0) THEN
          IF (NLINMN(I) < 0) THEN
             SRTENE(NMAXI) = ENEMAX(I)
             SRTIDX(NMAXI) = I
             CALL ADDLST(LPAKED,SRTENE,SRTIDX,NMAXI )
          ELSE
             NMAXI = NMAXI - 1
          ENDIF
       ELSE

          call sub7000
       ENDIF
    ELSEIF (OLDSMX < 0) THEN
       IF (STPMAX(I) == 0) THEN
          NMAXI = NMAXI - 1
          IF (NLINMN(I) < 0) THEN
             CALL GETREM( OLDEMX,SRTENE,SRTIDX,NMAXI,.FALSE.)
          ENDIF
       ELSEIF (STPMAX(I) < 0) THEN
          IF (NLINMN(I) < 0) THEN
             SRTENE(NMAXI) = ENEMAX(I)
             SRTIDX(NMAXI) = I
             CALL ADDLST(LPAKED,SRTENE,SRTIDX,NMAXI )
             CALL GETREM( OLDEMX,SRTENE,SRTIDX,NMAXI,.FALSE.)
          ELSE
             NMAXI = NMAXI - 1
          ENDIF
       ELSE

          call sub7000
       ENDIF
    ELSE
       CALL UPDNEW( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX, &
            NLINMN,PTGRAD,SADGRD,SADMIN,ENEMAX )
       IF (OLDSMX /= BIGINT) THEN
          CALL GETREM( OLDEMX, SRTENE,SRTIDX, NMAXI,.FALSE.)
       ENDIF
    ENDIF

    RETURN

    !------------------- Contained recursive subroutines -----------------------
  contains

    subroutine sub7000
      IF (STPMAX(I) /= BIGINT) THEN
         CALL ADDLST(LPAKED,SRTENE,SRTIDX,NMAXI )
      ENDIF

      IF (NLINMN(I) < 0) THEN
         NLINMN(I) = ABS(NLINMN(I))
         IF (OLDSMX < 0) THEN
            CALL GETREM( OLDEMX, SRTENE,SRTIDX, NMAXI,.FALSE.)
         ENDIF
      ELSE
         CALL GETREM( PTENE(I), SRTENE, SRTIDX, NMAXI, .FALSE.)
      ENDIF
      return
    end subroutine sub7000
  END subroutine updpre


  SUBROUTINE UPDNEW( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX, &
       NLINMN,PTGRAD,SADGRD,SADMIN,ENEMAX )


    use chm_kinds
    use number
    implicit none

    LOGICAL   LPAKED
    INTEGER   I, SRTIDX(*),NMAXI,STPMAX(*),NLINMN(*),SADMIN
    real(chm_real)    SRTENE(*),PTGRAD(*),SADGRD,ENEMAX(*)

    INTEGER, PARAMETER :: BIGINT=999999


    IF ( STPMAX(I) <= 0 ) THEN
       IF (PTGRAD(I) <= SADGRD.AND.ABS(NLINMN(I)) >= SADMIN) THEN
          NLINMN(I) = -ABS(NLINMN(I))
          IF (STPMAX(I) < 0) THEN
             SRTENE(NMAXI) = ENEMAX(I)
             SRTIDX(NMAXI) = I
             CALL ADDLST(LPAKED,SRTENE,SRTIDX,NMAXI )
          ELSE
             NMAXI = NMAXI - 1
          ENDIF
       ELSE
          CALL ADDLST(LPAKED,SRTENE,SRTIDX,NMAXI )
       ENDIF
    ELSE
       NLINMN(I) = ABS(NLINMN(I))
       IF ( STPMAX(I) /= BIGINT ) THEN
          CALL ADDLST(LPAKED,SRTENE,SRTIDX,NMAXI )
       ENDIF
    ENDIF

    RETURN
  END subroutine updnew


  SUBROUTINE UPDEND( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
       NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX )


    use chm_kinds
    use number
    implicit none


    LOGICAL   LPAKED,QUP(*),QDOWN(*)
    INTEGER   I, SRTIDX(*),NMAXI,STPMAX(*),NLINMN(*),SADMIN
    real(chm_real)    SRTENE(*),PTGRAD(*),SADGRD,PTENE(*),ENEMAX(*)

    INTEGER, PARAMETER :: BIGINT=999999

    IF ( STPMAX(I) <= 0 ) THEN
       IF ( .NOT.QUP(I) ) THEN
          NLINMN(I) = ABS(NLINMN(I))
          IF (STPMAX(I) == 0) THEN
             STPMAX(I) = BIGINT
             IF (PTGRAD(I) > SADGRD .OR. &
                  ABS(NLINMN(I)) < SADMIN) THEN
                CALL GETREM( PTENE(I), SRTENE, SRTIDX, NMAXI, &
                     .FALSE.)
             ENDIF
          ELSE
             STPMAX(I) = -STPMAX(I)
             IF (PTGRAD(I) > SADGRD .OR. &
                  ABS(NLINMN(I)) < SADMIN) THEN
                NMAXI = NMAXI + 1
                SRTENE(NMAXI) = ENEMAX(I)
                SRTIDX(NMAXI) = I
                CALL ADDLST(LPAKED,SRTENE,SRTIDX,NMAXI )
                CALL GETREM( PTENE(I), SRTENE, SRTIDX, NMAXI, &
                     .FALSE.)
             ENDIF
          ENDIF
       ENDIF
    ELSEIF ( QUP(I).AND.QDOWN(I) ) THEN
       IF ( PTGRAD(I) <= SADGRD .AND. &
            ABS(NLINMN(I)) >= SADMIN)  NLINMN(I) = -ABS(NLINMN(I))
       IF (STPMAX(I) == BIGINT) THEN
          STPMAX(I) = 0
          IF (NLINMN(I) >= 0) THEN
             NMAXI = NMAXI + 1
             SRTENE(NMAXI) = PTENE(I)
             SRTIDX(NMAXI) = I
             CALL ADDLST(LPAKED,SRTENE,SRTIDX,NMAXI )
          ENDIF
       ELSEIF ( PTENE(I) > ENEMAX(I) ) THEN
          STPMAX(I) = -STPMAX(I)
          IF (NLINMN(I) >= 0) THEN
             NMAXI = NMAXI + 1
             SRTENE(NMAXI) = PTENE(I)
             SRTIDX(NMAXI) = I
             CALL ADDLST(LPAKED,SRTENE,SRTIDX,NMAXI )
             CALL GETREM( ENEMAX(I), SRTENE, SRTIDX, NMAXI, &
                  .FALSE.)
          ENDIF
       ELSE
          NLINMN(I) = ABS(NLINMN(I))
       ENDIF
    ELSE
       NLINMN(I) = ABS(NLINMN(I))
    ENDIF

    RETURN
  END subroutine updend


  SUBROUTINE SETIHI( IDXSAD,NPOINT,IDNEXT,PTGRAD,SADGRD,QUP,QDOWN, &
       NLINMN,SADMIN,PTENE )


    use chm_kinds
    implicit none

    LOGICAL   QUP(*),QDOWN(*)
    INTEGER   IDXSAD,IDNEXT(*),NPOINT,NLINMN(*),SADMIN
    real(chm_real)    PTENE(*),PTGRAD(*),SADGRD

    LOGICAL   QSADL
    INTEGER   J,IDUM3

    IDXSAD = 0
    J = 1
    DO IDUM3=2, NPOINT-1
       J = IDNEXT(J)

       QSADL = .FALSE.
       IF ((PTGRAD(J) <= SADGRD.AND.QUP(J).AND.QDOWN(J).AND. &
            ABS(NLINMN(J)) >= SADMIN) )  QSADL = .TRUE.

       IF ( NLINMN(J)  <  0 )  QSADL = .TRUE.

       IF ( QSADL )then
          if(idxsad == 0) then
             idxsad = j
          elseif(PTENE(J) > PTENE(IDXSAD) ) then
             IDXSAD = J
          endif
       endif

    ENDDO

    RETURN
  END subroutine setihi

  SUBROUTINE COUNTSD( NSADDL, NPOINT,IDNEXT,NLINMN )
    use chm_kinds
    implicit none

    INTEGER   NSADDL
    INTEGER   NPOINT,IDNEXT(*),NLINMN(*)

    INTEGER   J,IDX

    NSADDL = 0
    IDX = 1
    DO J=1,NPOINT
       IF (NLINMN(IDX) < 0)  NSADDL = NSADDL+1
       IDX    = IDNEXT(IDX)
    ENDDO

    RETURN
  END subroutine countsd

  SUBROUTINE LISTSD( NPOINT,IDNEXT,NLINMN, PTENE,PTGRAD )
    use chm_kinds
    use stream
#if KEY_PARALLEL==1
    use parallel        
#endif
    implicit none

    INTEGER   NPOINT,IDNEXT(*),NLINMN(*)
    real(chm_real)    PTENE(*),PTGRAD(*)

    INTEGER   J,IDX, NSADDL, ORDER, IDUM1
    INTEGER, PARAMETER :: MAXSAD=1000
    INTEGER   ISSORT(MAXSAD)
    real(chm_real)    ESSORT(MAXSAD)

    NSADDL = 0

    IDX = 1
    DO J=1,NPOINT
       IF (NLINMN(IDX) < 0) THEN
          IF (NSADDL >= MAXSAD) THEN
             CALL W0(' ')
             CALL WRNDIE(1,'<LISTSD>','Too many saddle-points.')
             exit
          ENDIF
          NSADDL = NSADDL+1
          ISSORT(NSADDL) = IDX
          ESSORT(NSADDL) = PTENE(IDX)
       ENDIF
       IDX    = IDNEXT(IDX)
    ENDDO

    IF (NSADDL == 0)  RETURN
    CALL DSORT1(ESSORT, ISSORT, NSADDL, .FALSE.)

    CALL W0(' ')
    CALL WI( ' Total number of saddle-points = ', NSADDL)
    CALL W0(' (either initially declared or refined to the')
    CALL W0('  specified criteria of SADGRD and SADMIN)')
    CALL W0('   They are (sorted by decreasing energy) :')
    CALL W0(' ------------------------------------------------')
    CALL W0('   N   IDX       Energy      rms(Grad)    LinMin')
    CALL W0(' ------------------------------------------------')

    DO J=1,NSADDL
       IDX = ISSORT(J)
       ORDER = POSITN(IDX, NPOINT,IDNEXT)
       IDUM1 = PRTLNM( NLINMN(IDX) )
#if KEY_PARALLEL==1
       IF (MYNOD == 0) THEN
#endif 
          WRITE(OUTU,1210)  ORDER,IDX,PTENE(IDX),PTGRAD(IDX), IDUM1
#if KEY_PARALLEL==1
       ENDIF
#endif 
    ENDDO

1210 FORMAT(I4,2X,I4,2X,1PG17.10E2,2X,1PE8.3E1,2X,I7)

    RETURN
  END subroutine listsd

  INTEGER FUNCTION PRTLNM( NLINMN )
    use chm_kinds
    use dimens_fcm
    use psf
    implicit none
    INTEGER       NLINMN
    INTEGER       NFIXED, IDIM

    CALL FIXED(NFIXED)
    IDIM  = 3*(NATOM - NFIXED)

    IF     ( ABS(NLINMN) < 2*IDIM ) THEN
       PRTLNM = ABS(NLINMN)
    ELSEIF ( ABS(NLINMN) < 3*IDIM ) THEN
       PRTLNM = ABS(NLINMN) - 2*IDIM
    ELSE
       PRTLNM = 999999
    ENDIF

    IF (NLINMN < 0)  PRTLNM = -PRTLNM

    RETURN
  END function prtlnm

  INTEGER FUNCTION REDLNM( NLINMN, SADMIN, NCYCLE )


    use chm_kinds
    use dimens_fcm
    use psf
    implicit none

    INTEGER       NLINMN, SADMIN, NCYCLE
    INTEGER       NFIXED, IDIM

    CALL FIXED(NFIXED)
    IDIM  = 3*(NATOM - NFIXED)

    IF     ( ABS(NLINMN) == 999999 ) THEN
       REDLNM = 3*IDIM
       IF (NLINMN < 0)  REDLNM = -REDLNM
    ELSE
       REDLNM = ABS(NLINMN)
       IF ( NLINMN < 0 .AND. &
            (REDLNM >= SADMIN .OR. NCYCLE == 0) )  REDLNM = -REDLNM
    ENDIF

    RETURN
  END function redlnm

  INTEGER FUNCTION POSITN(IDX, NPOINT,IDNEXT)
    use chm_kinds
    implicit none

    INTEGER   IDX, NPOINT,IDNEXT(*)

    INTEGER   J,NEXT

    POSITN = 0
    NEXT = 1
    DO J=1, NPOINT
       IF ( IDX == NEXT ) THEN
          POSITN = J
          RETURN
       ENDIF
       NEXT = IDNEXT(NEXT)
    ENDDO

    CALL WRNDIE(-1,'<POSITN>','Could not find IDX !')

    RETURN
  END function positn

  INTEGER FUNCTION PINDEX( N, NPOINT,IDNEXT)
    use chm_kinds
    implicit none

    INTEGER   N, NPOINT,IDNEXT(*)

    INTEGER   I

    IF (N > NPOINT.OR.N < 1) THEN
       PINDEX = 0
       CALL WRNDIE(1,'<PINDEX>','There are not that many path-points!')
    ELSE
       PINDEX = 1
       DO I=2,N
          PINDEX = IDNEXT(PINDEX)
       ENDDO
    ENDIF

    RETURN
  END function pindex

  FUNCTION PATHLN(NPOINT,IDNEXT,LENSEG)
    use chm_kinds
    use number
    implicit none

    real(chm_real)    LENSEG(*), pathln
    INTEGER   IDNEXT(*), NPOINT

    INTEGER   I,IDX

    PATHLN = ZERO
    IDX = 1

    DO I = 1,NPOINT-1
       PATHLN = PATHLN + LENSEG(IDX)
       IDX = IDNEXT(IDX)
    ENDDO

    RETURN
  END function pathln


  FUNCTION PARAMX(AA,BB,FA,F0,FB)
    use chm_kinds
    use number
    implicit none

    real(chm_real)     AA,BB,FA,F0,FB,paramx
    real(chm_real)     A,B

    A = ABS(AA)
    B = ABS(BB)


    PARAMX = HALF*(A*A*(F0-FB)+B*B*(FA-F0))/(A*(FB-F0)+B*(FA-F0))

    RETURN
  END FUNCTION PARAMX


  SUBROUTINE BRAKET( AX,BX,CX, FA,FB,FC, FUNC, MAGLIM, VERBEU, &
       PX,PY,PZ, SX,SY,SZ, DBX,DBY,DBZ, SNORM, &
       N , QMINI, I )
    use chm_kinds
    use number
    use dimens_fcm
    use deriv
    use exfunc
    use memory
    implicit none

    real(chm_real),allocatable,dimension(:) :: DCX
    real(chm_real),allocatable,dimension(:) :: DCY
    real(chm_real),allocatable,dimension(:) :: DCZ
    LOGICAL    QMINI
    INTEGER    N, VERBEU, I
    real(chm_real)     PX(*),PY(*),PZ(*), SX(*),SY(*),SZ(*)
    real(chm_real)     DBX(*),DBY(*),DBZ(*)
    real(chm_real)     AX,BX,CX,FA,FB,FC,FUNC,MAGLIM, SNORM

    LOGICAL    LINVER
    INTEGER    J, MAXSID
    real(chm_real)     DUM,R,Q,U,FU,ULIM,LASTU, AINI,BINI
    LOGICAL, PARAMETER :: QDERIV=.TRUE.

#if KEY_SINGLE==1
    real(chm_real), PARAMETER :: GOLD=1.618034E0
#else /**/
    real(chm_real), PARAMETER :: GOLD=1.618034D0
#endif 

    EXTERNAL  FUNC

    MAXSID = I
    I = 0
    AINI = AX
    BINI = BX
    LINVER = .FALSE.

    call chmalloc('travel2.src','BRAKET','DCX',N,crl=DCX)
    call chmalloc('travel2.src','BRAKET','DCY',N,crl=DCY)
    call chmalloc('travel2.src','BRAKET','DCZ',N,crl=DCZ)

    FB = FUNC(BX, PX,PY,PZ, SX,SY,SZ, N, QDERIV, .FALSE., ZERO)
    IF (QDERIV)  CALL COP3D( DX,DY,DZ, DBX,DBY,DBZ, N)

    IF (VERBEU >= 6) THEN
       CALL W0(' ')
       CALL W2R( 'Passed to BRAKET :  AX & FA =', AX,FA)
       CALL W2R( 'First energy call:  BX & FB =', BX,FB)
    ENDIF
    IF (.NOT.QMINI) THEN
       FA = -FA
       FB = -FB
    ENDIF

    IF (FB > FA .AND. MAXSID > 0) THEN
       DO J=1, MAXSID
          I = I+1
          CX = BX
          FC = FB
          BX = AX + (CX-AX)/(GOLD+ONE)
          FB = FUNC(BX, PX,PY,PZ, SX,SY,SZ, N, QDERIV, .FALSE., &
               ABS(CX-BX)*SNORM )
          IF (QDERIV)  CALL COP3D( DX,DY,DZ, DBX,DBY,DBZ, N)
          IF (.NOT.QMINI)  FB = -FB
          LASTU = BX
          IF (FB <= FA)  GOTO  2
       ENDDO
       I = 0
       IF (VERBEU >= 4) THEN
          CALL W0(' ')
          CALL W0('!!! Warning from BRAKET:  Failed to find extremum')
          CALL WI('!!! on side of initial BX. # of attempts =',MAXSID)
       ENDIF
    ENDIF

    IF (FB > FA) THEN
       DUM=AX
       AX=BX
       BX=DUM
       DUM=FB
       FB=FA
       FA=DUM
       LINVER = .TRUE.
    ENDIF

    CX=BX+GOLD*(BX-AX)
    I = I+1
    FC = FUNC(CX, PX,PY,PZ, SX,SY,SZ, N, QDERIV, .FALSE., &
         ABS(CX-BX)*SNORM )
    IF (QDERIV) &
         CALL COP3D( DX,DY,DZ, DCX,DCY,DCZ, N)
    IF (.NOT.QMINI)  FC = -FC
    LASTU = CX

1   IF (FB >= FC) THEN
       R=(BX-AX)*(FB-FC)
       Q=(BX-CX)*(FB-FA)
       IF ((ABS(R)  >  RBIG).OR.(ABS(Q)  >  RBIG)) THEN
          CALL W0(' ')
          CALL WI( ' After cycles I = ',I)
          CALL W0( ' BRAKET cannot find an extremum down'// &
               ' that direction : going to infinity !')
          I = 0
          GOTO  2
       ENDIF
       U =BX-((BX-CX)*Q-(BX-AX)*R)/(TWO*SIGN(MAX(ABS(Q-R),RSMALL),Q-R))
       ULIM=BX+MAGLIM*(CX-BX)
       IF ((BX-U)*(U-CX) > ZERO) THEN
          I = I+1
          FU = FUNC(U, PX,PY,PZ, SX,SY,SZ, N, QDERIV, .FALSE., &
               ABS(U-LASTU)*SNORM )
          IF (.NOT.QMINI)  FU = -FU
          LASTU = U
          IF (FU < FC) THEN
             AX=BX
             FA=FB
             BX=U
             FB=FU
             IF (QDERIV)  CALL COP3D( DX,DY,DZ, DBX,DBY,DBZ, N)
             GOTO  1
          ELSEIF (FU > FB) THEN
             CX=U
             FC=FU
             IF (QDERIV) &
                  CALL COP3D( DX,DY,DZ, DCX,DCY,DCZ, N)
             GOTO  1
          ENDIF
          U = CX+GOLD*(CX-BX)
          I = I+1
          FU = FUNC(U, PX,PY,PZ, SX,SY,SZ, N, QDERIV, .FALSE., &
               ABS(U-LASTU)*SNORM )
          IF (.NOT.QMINI)  FU = -FU
          LASTU = U
       ELSEIF ((CX-U)*(U-ULIM) > ZERO) THEN
          I = I+1
          FU = FUNC(U, PX,PY,PZ, SX,SY,SZ, N, QDERIV, .FALSE., &
               ABS(U-LASTU)*SNORM )
          IF (.NOT.QMINI)  FU = -FU
          LASTU = U
          IF (FU < FC) THEN
             BX=CX
             FB=FC
             CX=U
             FC=FU
             IF (QDERIV) THEN
                CALL COP3D(DCX,DCY,DCZ,DBX,DBY,DBZ,N)
                CALL COP3D( DX,DY,DZ, DCX,DCY,DCZ, N)
             ENDIF
             U = CX+GOLD*(CX-BX)
             I = I+1
             FU = FUNC(U, PX,PY,PZ, SX,SY,SZ, N, QDERIV, .FALSE., &
                  ABS(U-LASTU)*SNORM )
             IF (.NOT.QMINI)  FU = -FU
             LASTU = U
          ENDIF
       ELSEIF ((U-ULIM)*(ULIM-CX) >= ZERO) THEN
          U = ULIM
          I = I+1
          FU = FUNC(U, PX,PY,PZ, SX,SY,SZ, N, QDERIV, .FALSE., &
               ABS(U-LASTU)*SNORM )
          IF (.NOT.QMINI)  FU = -FU
          LASTU = U
       ELSE
          U = CX+GOLD*(CX-BX)
          I = I+1
          FU = FUNC(U, PX,PY,PZ, SX,SY,SZ, N, QDERIV, .FALSE., &
               ABS(U-LASTU)*SNORM )
          IF (.NOT.QMINI)  FU = -FU
          LASTU = U
       ENDIF
       AX=BX
       BX=CX
       CX=U
       FA=FB
       FB=FC
       FC=FU
       IF (QDERIV) THEN
          CALL COP3D(DCX,DCY,DCZ,DBX,DBY,DBZ,N)
          CALL COP3D( DX,DY,DZ, DCX,DCY,DCZ, N)
       ENDIF
       GOTO  1
    ENDIF

2   IF (.NOT.QMINI) THEN
       FA = -FA
       FB = -FB
       FC = -FC
    ENDIF

    IF (LINVER) THEN
       DUM=AX
       AX=CX
       CX=DUM
       DUM=FC
       FC=FA
       FA=DUM
       I = -I
    ENDIF

    IF (VERBEU >= 4.AND.I /= 0) THEN
       CALL W0(' ')
       CALL WI('    Found by BRAKET after cycles I = ',I)
       IF (AX == AINI.AND.( (AX-CX)*(CX-BINI) >= ZERO ) ) &
            CALL W0('    AX = initial AX, CX same side as initial BX.')
    ENDIF
    IF (VERBEU >= 5) THEN
       CALL W3R(' Braket AX,BX,CX  :', AX,BX,CX)
       CALL W3R(' F at these points:', FA,FB,FC)
    ENDIF

    call chmdealloc('travel2.src','BRAKET','DCX',N,crl=DCX)
    call chmdealloc('travel2.src','BRAKET','DCY',N,crl=DCY)
    call chmdealloc('travel2.src','BRAKET','DCZ',N,crl=DCZ)

    RETURN
  END subroutine braket


  SUBROUTINE ADDPRL( SERNM1,SERNM2, NEIBR1,NEIBR2, NEIBOR )

    use chm_kinds
    implicit none
    INTEGER  SERNM1,SERNM2, NEIBR1(*),NEIBR2(*), NEIBOR

    NEIBOR = NEIBOR + 1
    NEIBR1(NEIBOR) = SERNM1
    NEIBR2(NEIBOR) = SERNM2

    RETURN
  END SUBROUTINE ADDPRL

  SUBROUTINE DSORT1( R8, IND, N, QINCRS)

    use chm_kinds
    implicit none

    LOGICAL  QINCRS
    real(chm_real)   R8(*)
    INTEGER  IND(*), N, I

    DO I=2,N
       CALL ADDBOT( R8, IND, I, QINCRS)
    ENDDO

    RETURN
  END subroutine dsort1

  SUBROUTINE ADDTOP( R8, IND, N, QINCRS)

    use chm_kinds
    implicit none

    LOGICAL  QINCRS
    real(chm_real)   R8(*), NEWR8
    INTEGER  IND(*), GETIDX, N, I, IBASE, ITOP, NLEFT, IHALF

    IF (N <= 1)  RETURN

    NEWR8  =  R8(1)
    GETIDX = IND(1)
    IBASE  = 1
    ITOP   = N

    IF (QINCRS) THEN
       IF (R8(1) <= R8(2))  RETURN
261    NLEFT = ITOP-IBASE
       IF (NLEFT  ==  1) THEN
          IF (NEWR8  <  R8(ITOP) )  ITOP = ITOP-1
          GOTO  263
       ENDIF
       IHALF = IBASE + INT(NLEFT/2)
       IF (NEWR8  >  R8(IHALF) ) THEN
          IBASE = IHALF
       ELSE
          ITOP  = IHALF
       ENDIF
       GOTO  261

    ELSE
       IF (R8(1) >= R8(2))  RETURN
262    NLEFT = ITOP-IBASE
       IF (NLEFT  ==  1) THEN
          IF (NEWR8  >  R8(ITOP) )  ITOP = ITOP-1
          GOTO  263
       ENDIF

       IHALF = IBASE + INT(NLEFT/2)

       IF (NEWR8  <  R8(IHALF) ) THEN
          IBASE = IHALF
       ELSE
          ITOP  = IHALF
       ENDIF
       GOTO  262
    ENDIF

263 DO I = 1,(ITOP-1)
       R8(I) =  R8(I+1)
       IND(I) = IND(I+1)
    ENDDO

    R8(ITOP) = NEWR8
    IND(ITOP) = GETIDX
    RETURN
  END subroutine addtop

  SUBROUTINE GETREM( X, R8, IND, N, QINCRS)

    use chm_kinds
    implicit none

    LOGICAL  QINCRS
    real(chm_real)   R8(*), X
    INTEGER  IND(*), N

    INTEGER  IDX, J
    !      INTEGER  IDXSLD
    !      EXTERNAL IDXSLD

    IDX = IDXSLD( X, R8, N, QINCRS)
    IF ( IDX  >  0 ) THEN
       N = N - 1
       DO J=IDX, N
          R8(J) =  R8(J+1)
          IND(J) = IND(J+1)
       ENDDO
    ELSE
       CALL W0(' ')
       CALL W0(' !!! Stopping in routine GETREM :')
       CALL WR(' !!! Could not find the requested X= ', X )
       STOP
    ENDIF

    RETURN
  END subroutine getrem

  SUBROUTINE ADDBOT( R8, IND, N, QINCRS)

    use chm_kinds
    implicit none

    LOGICAL  QINCRS
    real(chm_real)   R8(*), NEWR8
    INTEGER  IND(*), GETIDX, N, I, IBASE, ITOP, NLEFT, IHALF

    IF (N <= 1)  RETURN

    NEWR8  =  R8(N)
    GETIDX = IND(N)
    IBASE  = 0
    ITOP   = N-1

    IF (QINCRS) THEN
       IF (R8(N) >= R8(N-1))  RETURN
251    NLEFT = ITOP-IBASE
       IF (NLEFT  ==  1) THEN
          IF (NEWR8  >  R8(ITOP) )  ITOP = ITOP+1
          GOTO  253
       ENDIF
       IHALF = IBASE + INT(NLEFT/2)
       IF (NEWR8  >  R8(IHALF) ) THEN
          IBASE = IHALF
       ELSE
          ITOP  = IHALF
       ENDIF
       GOTO  251

    ELSE
       IF (R8(N) <= R8(N-1))  RETURN
252    NLEFT = ITOP-IBASE
       IF (NLEFT  ==  1) THEN
          IF (NEWR8  <  R8(ITOP) )  ITOP = ITOP+1
          GOTO  253
       ENDIF

       IHALF = IBASE + INT(NLEFT/2)

       IF (NEWR8  <  R8(IHALF) ) THEN
          IBASE = IHALF
       ELSE
          ITOP  = IHALF
       ENDIF
       GOTO  252
    ENDIF

253 DO I = 1,(N-ITOP)
       R8(N-I+1) =  R8(N-I)
       IND(N-I+1) = IND(N-I)
    ENDDO

    R8(ITOP) = NEWR8
    IND(ITOP) = GETIDX
    RETURN
  END subroutine addbot

  SUBROUTINE REMSER( SERIAL, NEIBR1,NEIBR2, NEIBOR )

    use chm_kinds
    implicit none

    INTEGER  SERIAL, NEIBR1(*),NEIBR2(*), NEIBOR
    INTEGER  COPIED, HOLES, CH

    IF (NEIBOR < 0) THEN
       CALL WRNDIE(-5,'<REMSER>', &
            'REMSER was called with inadequate NEIBOR' )
    ENDIF

    COPIED = 0
    HOLES  = 1

1   IF (COPIED == NEIBOR)  RETURN

    CH = COPIED + HOLES
    IF (SERIAL == NEIBR1(CH) .OR. SERIAL == NEIBR2(CH)) THEN
       HOLES  = HOLES  + 1
       NEIBOR = NEIBOR - 1
       GOTO  1
    ENDIF

    COPIED = COPIED + 1
    NEIBR1(COPIED) = NEIBR1(CH)
    NEIBR2(COPIED) = NEIBR2(CH)
    GOTO  1

  END SUBROUTINE REMSER

  INTEGER FUNCTION IDXSLD( X, R8, N, QINCRS)

    use chm_kinds
    implicit none

    LOGICAL  QINCRS
    real(chm_real)   R8(*), X
    INTEGER  N, IBASE, ITOP, NLEFT, IHALF

    IDXSLD = 0

    IF (N  <  1)  RETURN

    IBASE  = 0
    ITOP   = N

    IF (QINCRS) THEN
1      NLEFT = ITOP-IBASE
       IF (NLEFT  ==  1) THEN
          IF (X  ==  R8(ITOP) )  IDXSLD = ITOP
          RETURN
       ENDIF

       IHALF = IBASE + INT(NLEFT/2)
       IF (X  >  R8(IHALF) ) THEN
          IBASE = IHALF
       ELSE
          ITOP  = IHALF
       ENDIF
       GOTO  1

    ELSE
2      NLEFT = ITOP-IBASE
       IF (NLEFT  ==  1) THEN
          IF (X  ==  R8(ITOP) )  IDXSLD = ITOP
          RETURN
       ENDIF

       IHALF = IBASE + INT(NLEFT/2)
       IF (X  <  R8(IHALF) ) THEN
          IBASE = IHALF
       ELSE
          ITOP  = IHALF
       ENDIF
       GOTO  2
    ENDIF

  END FUNCTION IDXSLD


  SUBROUTINE LNMINI( AX,BX,CX, FUNC,DFUNC, FB,DFB,DBX,DBY,DBZ, &
       STPTOL,TOLFUN,GRATOL, VERBEU, PX,PY,PZ, SX,SY,SZ,SNORM, &
       N, QMINI , LXEVAL, MODXIT, GNORM )


    use chm_kinds
    use number
    use dimens_fcm
    use deriv
    implicit none

    LOGICAL   QMINI
    INTEGER   N, VERBEU, LXEVAL, MODXIT
    real(chm_real)    DBX(*),DBY(*),DBZ(*),PX(*),PY(*),PZ(*), &
         SX(*),SY(*),SZ(*)
    real(chm_real)    AX,BX,CX,FUNC,DFUNC,FB,DFB,STPTOL,TOLFUN, &
         GRATOL,SNORM
    real(chm_real)    GNORM

    LOGICAL   OK1,OK2
    INTEGER   I
    real(chm_real)    A,B, V,W,X,U, FV,FW,FX,FU, DV,DW,DFX,DU, XM,  &
         TOL1,TOLFNC
    real(chm_real)    TOL2,STEP,PRVSTP,OLDSTP,STEP1,STEP2,U1,U2,  &
         INIPRJ
    real(chm_real)    GRDCOS,COSATB
    !      real(chm_real)    DDOT1,DSDOT1, GRDCOS,COSATB
    real(chm_real)    STNORM, LASTU, DDUM1

    EXTERNAL  FUNC,DFUNC!, DDOT1,DSDOT1

    IF ((BX-AX)*(CX-BX)  <=  ZERO) THEN
       CALL W0(' ')
       CALL W0(' LNMINI was called with inadequate AX,BX,CX :')
       CALL WR( ' AX =', AX)
       CALL WR( ' BX =', BX)
       CALL WR( ' CX =', CX)
       STOP
    ENDIF

    I = 0
    A=MIN(AX,CX)
    B=MAX(AX,CX)

    PRVSTP = ZERO
    TOL1   = STPTOL/SNORM
    TOL2   = TWO*TOL1

    X  = BX
    FX = FB
    DFX = DDOT1( SX,SY,SZ, DX,DY,DZ, N )/SNORM
    DFB = DFX

    IF (.NOT.QMINI) THEN
       FX = -FX
       DFX = -DFX
    ENDIF


    IF ( ABS(MODXIT) == 1 ) THEN
       GNORM = SQRT(DSDOT1( DX,DY,DZ, N ))
       GRDCOS = ZERO
       IF ( GNORM > ZERO )  GRDCOS = ABS(DFX/GNORM)
       IF ( GRDCOS <= GRATOL .AND. MODXIT > 0 ) THEN
          IF (VERBEU >= 4) THEN
             CALL W0(' ')
             CALL WI( ' After cycles I =', I)
             CALL W0(' LNMINI exit with satisfied GRATOLd.')
          ENDIF
          GOTO  3
       ENDIF
    ELSE
       DDUM1 = DDOT1( SX,SY,SZ, DBX,DBY,DBZ, N )
       IF (DDUM1 == ZERO) THEN
          CALL W0(' ')
          CALL W0('LNMINI> Warn: DBX has zero projection on SX.')
          CALL W0('LNMINI> Exiting LNMINI without minimizing !!')
          GOTO  3
       ENDIF
       INIPRJ = SNORM/DDUM1
       GRDCOS = ABS(DFX*INIPRJ)
       IF ( GRDCOS <= GRATOL .AND. MODXIT > 0 ) THEN
          IF (VERBEU >= 4) THEN
             CALL W0(' ')
             CALL WI( ' After cycles I =', I)
             CALL W0(' LNMINI exit with satisfied GRATOLd.')
          ENDIF
          GOTO  3
       ENDIF
    ENDIF

    CALL COP3D( DX,DY,DZ, DBX,DBY,DBZ, N)
    COSATB = GRDCOS

    V  = X
    W  = X
    FV = FX
    FW = FX
    DV = DFX
    DW = DFX

    XM=HALF*(A+B)

    IF ( MODXIT == 3 .AND. &
         (ABS(X-XM)+HALF*(B-A)) <= TOL2 ) THEN
       IF (VERBEU >= 4) THEN
          CALL W0(' ')
          CALL WI( ' After cycles I =', I)
          CALL W0(' LNMINI exit [1] with satisfied TOLSTeP.')
       ENDIF
       GOTO  5
    ENDIF
    loop4:DO I = 1,LXEVAL

       IF (ABS(PRVSTP) > TOL1) THEN
          STEP1 = TWO*(B-A)
          STEP2 = STEP1
          IF (DW /= DFX)  STEP1 = (W-X)*DFX/(DFX-DW)
          IF (DV /= DFX)  STEP2 = (V-X)*DFX/(DFX-DV)
          U1 = X+STEP1
          U2 = X+STEP2
          OK1 = ((A-U1)*(U1-B) > ZERO).AND.(DFX*STEP1 <= ZERO)
          OK2 = ((A-U2)*(U2-B) > ZERO).AND.(DFX*STEP2 <= ZERO)
          OLDSTP = PRVSTP
          PRVSTP = STEP

          IF (.NOT.(OK1.OR.OK2)) THEN
             GOTO  1
          ELSEIF (OK1.AND.OK2) THEN
             IF (ABS(STEP1) < ABS(STEP2)) THEN
                STEP = STEP1
             ELSE
                STEP = STEP2
             ENDIF
          ELSEIF (OK1) THEN
             STEP = STEP1
          ELSE
             STEP = STEP2
          ENDIF

          IF (ABS(STEP) > ABS(HALF*OLDSTP))  GOTO  1

          U = X+STEP
          STEP1 = SIGN(TOL1,XM-X)
          U1 = X+STEP1
          IF ( (U-A < TOL2.OR.B-U < TOL2) .AND. &
               ((A-U1)*(U1-B) > ZERO) .AND. (DFX*STEP1 <= ZERO) ) THEN
             STEP = STEP1
          ENDIF
          GOTO  2
       ENDIF

1      IF (DFX >= ZERO) THEN
          PRVSTP = A-X
       ELSE
          PRVSTP = B-X
       ENDIF
       STEP  = HALF*PRVSTP

2      STEP1 = SIGN(TOL1,STEP)
       U1 = X+STEP1

       IF ( (ABS(STEP) < TOL1) .AND. &
            ((A-U1)*(U1-B) > ZERO) .AND. (DFX*STEP1 <= ZERO) ) THEN
          U = U1

          IF (I == 1) THEN
             STNORM = MAX(ABS(AX-U),ABS(CX-U))*SNORM
          ELSE
             STNORM = ABS(U-LASTU)*SNORM
          ENDIF
          LASTU = U
          FU = FUNC(U, PX,PY,PZ, SX,SY,SZ, N, .TRUE., .FALSE., STNORM)

          IF (.NOT.QMINI)  FU = -FU
          IF ( ABS(MODXIT) == 3 .AND. (MODXIT > 0.OR.I > 1) .AND. &
               FU > FX ) THEN
             IF (VERBEU >= 4) THEN
                CALL W0(' ')
                CALL WI( ' After cycles I =', I)
                CALL W0(' LNMINI exit [2] with satisfied TOLSTeP.')
             ENDIF
             GOTO  5
          ENDIF
       ELSE
          STEP1 = STEP
          U = X+STEP1

          IF (I == 1) THEN
             STNORM = MAX(ABS(AX-U),ABS(CX-U))*SNORM
          ELSE
             STNORM = ABS(U-LASTU)*SNORM
          ENDIF
          LASTU = U
          FU = FUNC(U, PX,PY,PZ, SX,SY,SZ, N, .TRUE., .FALSE., STNORM)

          IF (.NOT.QMINI)  FU = -FU
       ENDIF

       DU = DFUNC(SX,SY,SZ,SNORM, N)
       IF (.NOT.QMINI)  DU = -DU

       IF (FX == ZERO) THEN
          TOLFNC = RSMALL
       ELSE
          TOLFNC = ABS(FX)*TOLFUN
       ENDIF

       IF ( ABS(MODXIT) == 1 ) THEN
          DDUM1 = DSDOT1( DX,DY,DZ, N )
          IF (DDUM1 > ZERO) THEN
             GNORM = SQRT(DDUM1)
             GRDCOS = ABS(DU/GNORM)
          ELSE
             GRDCOS = ZERO
          ENDIF
          IF ( GRDCOS <= GRATOL ) THEN
             IF (VERBEU >= 4) THEN
                CALL W0(' ')
                CALL WI( ' After cycles I =', I)
                CALL W0(' LNMINI exit with satisfied GRATOLd.')
             ENDIF
             GOTO  6
          ENDIF
       ELSE
          GRDCOS = ABS(DU*INIPRJ)
          IF ( GRDCOS <= GRATOL ) THEN
             IF (VERBEU >= 4) THEN
                CALL W0(' ')
                CALL WI( ' After cycles I =', I)
                CALL W0(' LNMINI exit with satisfied GRATOLd.')
             ENDIF
             GOTO  6
          ENDIF
       ENDIF

       IF ( ABS(MODXIT) == 3 .AND. ABS(FX-FU) <= TOLFNC .AND. &
            ABS(FW-FX) <= TOLFNC .AND. W /= X ) THEN
          IF (VERBEU >= 4) THEN
             CALL W0(' ')
             CALL WI( ' After cycles I =', I)
             CALL W0(' LNMINI exit with satisfied TOLFUNc.')
          ENDIF
          GOTO  6
       ENDIF

       IF (FU <= FX) THEN
          IF (U >= X) THEN
             A = X
          ELSE
             B = X
          ENDIF

          V  = W
          FV = FW
          DV = DW

          W  = X
          FW = FX
          DW = DFX

          X  = U
          FX = FU
          DFX = DU

          CALL COP3D( DX,DY,DZ, DBX,DBY,DBZ, N)
          COSATB = GRDCOS

       ELSE
          IF (U < X) THEN
             A = U
          ELSE
             B = U
          ENDIF
          IF (FU <= FW .OR. W == X) THEN
             V  = W
             FV = FW
             DV = DW
             W  = U
             FW = FU
             DW = DU
          ELSEIF (FU <= FV .OR. V == X .OR. V == W) THEN
             V  = U
             FV = FU
             DV = DU
          ENDIF
       ENDIF

       XM=HALF*(A+B)

       IF ( ABS(MODXIT) == 3 .AND. &
            (ABS(X-XM)+HALF*(B-A)) <= TOL2 ) THEN
          IF (VERBEU >= 4) THEN
             CALL W0(' ')
             CALL WI(' After cycles I =', I)
             CALL W0(' LNMINI exit [1] with satisfied TOLSTeP.')
          ENDIF
          GOTO  5
       ENDIF

    enddo loop4

    CALL W0(' ')
    CALL WI('LNMINI> Warn: reached maximum iterations LXEVAL=',LXEVAL)
    GOTO  5

6   IF (FU > FX)  GOTO  5
    X   =  U
    FX  = FU
    DFX = DU
3   CALL COP3D( DX,DY,DZ, DBX,DBY,DBZ, N)
    COSATB = GRDCOS

5   IF (AX < CX) THEN
       AX = MIN(A,B)
       CX = MAX(A,B)
    ELSE
       AX = MAX(A,B)
       CX = MIN(A,B)
    ENDIF

    IF (.NOT.QMINI) THEN
       FX = -FX
       DFX = -DFX
    ENDIF

    IF (VERBEU >= 4) &
         CALL W2R(' function & line/gradient cosine =', FX,COSATB)
    IF (VERBEU >= 5 .AND. I > 0 ) &
         CALL W2R(' last step-size & relative function change =', &
         ABS(STEP1)*SNORM , MAX(ABS(FX-FU),ABS(FX-FW))/ABS(FX) )

    BX  = X
    FB  = FX
    DFB = DFX

    RETURN
  END subroutine lnmini

  SUBROUTINE FIXED(NFIXED)


    use chm_kinds
    use dimens_fcm
    use psf
    use travel
    implicit none

    INTEGER   NFIX,NFIXED, I
    SAVE      NFIX

    IF (.NOT.QTRAV) THEN
       NFIX = 0
       DO I = 1, NATOM
          IF (IMOVE(I) > 0)  NFIX = NFIX + 1
       ENDDO
    ENDIF

    NFIXED = NFIX

    RETURN
  END subroutine fixed

  SUBROUTINE LISTMN( NPOINT,IDNEXT,NLINMN, PTENE,PTGRAD )


    use chm_kinds
    use number
    use stream
#if KEY_PARALLEL==1
    use parallel        
#endif
    implicit none

    INTEGER   NPOINT,IDNEXT(*),NLINMN(*)
    real(chm_real)    PTENE(*),PTGRAD(*)

    INTEGER   J,IDX, NLOCMN, ORDER, NLENPP
    INTEGER, PARAMETER :: MAXMIN=9
    INTEGER   ILSORT(MAXMIN)
    real(chm_real)    ELSORT(MAXMIN)
    real(chm_real)    SMALST, EE,EPREV,ENEXT, DDUM1

    DDUM1 = NPOINT-2
    NLENPP = NINT( DDUM1/(MAXMIN-1) )
    NLENPP = MAX(5,NLENPP)
    SMALST = TWO*RBIG
    NLOCMN = 1
    IDX    = 1

    DO J=1,NPOINT-2
       EPREV  = PTENE(IDX)
       IDX    = IDNEXT(IDX)
       EE     = PTENE(IDX)
       ENEXT  = PTENE(IDNEXT(IDX))

       IF ( EE < EPREV.AND.EE < ENEXT.AND.EE < SMALST ) THEN
          SMALST         = EE
          ELSORT(NLOCMN) = EE
          ILSORT(NLOCMN) = IDX
       ENDIF

       IF ( MOD(J,NLENPP) == 0 .AND. SMALST < RBIG   ) THEN
          SMALST = TWO*RBIG
          NLOCMN = NLOCMN + 1
       ENDIF
    ENDDO

    IF ( SMALST >= RBIG   ) NLOCMN = NLOCMN - 1

    IF (NLOCMN == 0)  RETURN
    CALL DSORT1(ELSORT, ILSORT, NLOCMN, .TRUE.)

    CALL W0(' ')
    CALL W0(' Lowest local energy-minima in several path-sections :')
    CALL WI(' Each path-section has a numb. of path-points =',NLENPP)
    CALL W0(' (the minima are sorted by increasing energy)')
    CALL W0(' ------------------------------------------------')
    CALL W0('   N   IDX       Energy      rms(Grad)    LinMin')
    CALL W0(' ------------------------------------------------')

    DO J=1,NLOCMN
       IDX = ILSORT(J)
       ORDER = POSITN(IDX, NPOINT,IDNEXT)
#if KEY_PARALLEL==1
       IF (MYNOD == 0) THEN
#endif 
          WRITE(OUTU,1210)  ORDER,IDX,PTENE(IDX),PTGRAD(IDX),NLINMN(IDX)
#if KEY_PARALLEL==1
       ENDIF
#endif 
    ENDDO

1210 FORMAT(I4,2X,I4,2X,1PG17.10E2,2X,1PE8.3E1,2X,I7)

    RETURN
  END subroutine listmn

  FUNCTION NORMAX( IATOM, SX,SY,SZ, NATOM )


    use chm_kinds
    use exfunc
    use memory
    implicit none

    real(chm_real),allocatable,dimension(:) :: IDUM1
    INTEGER    NATOM, IATOM
    real(chm_real)     SX(*),SY(*),SZ(*),normax

    call chmalloc('travel2.src','NORMAX','IDUM1',NATOM,crl=IDUM1)

    NORMAX = NORMX1( IATOM, SX,SY,SZ, NATOM, IDUM1 )

    call chmdealloc('travel2.src','NORMAX','IDUM1',NATOM,crl=IDUM1)

    RETURN
  END FUNCTION NORMAX

  FUNCTION NORMX1( IATOM, SX,SY,SZ, NATOM, DUM )


    use chm_kinds
    use number
    implicit none

    INTEGER    NATOM, IATOM
    real(chm_real)     SX(*),SY(*),SZ(*), DUM(*), normx1

    INTEGER    I

    NORMX1 = ZERO
    IATOM  = 0

    DO I = 1, NATOM
       DUM(I) = SX(I)*SX(I) + SY(I)*SY(I) + SZ(I)*SZ(I)
    ENDDO

    DO I = 1, NATOM
       IF ( DUM(I) > NORMX1 ) THEN
          NORMX1 = DUM(I)
          IATOM  = I
       ENDIF
    ENDDO

    NORMX1 = SQRT(NORMX1)

    RETURN
  END function normx1


  SUBROUTINE SERCH1(A,B,C, FA,FB,FC, FUNC, PX,PY,PZ, SX,SY,SZ,SNORM, &
       N, NSTEP, QMINI, QFA,QFB,QFC, LFOUND )



    use chm_kinds
    use number
    implicit none

    LOGICAL    QMINI, LFOUND, QFA,QFB,QFC
    INTEGER    N, NSTEP
    real(chm_real)     PX(*),PY(*),PZ(*), SX(*),SY(*),SZ(*)
    real(chm_real)     A,B,C,FA,FB,FC,FUNC,SNORM

    LOGICAL    LINVER, QUPA,QUPC,QAFIRS
    INTEGER    I,NSTEPA,NSTEPC
    real(chm_real)     STEP,STEPA,STEPC, A1,A2,C1,C2,X,  &
         FA1,FA2,FC1,FC2,FX
    real(chm_real)     FINTER, DDUM1, LASTX

    EXTERNAL  FUNC

    IF ((B-A)*(C-B)  <=  ZERO) THEN
       CALL W0(' ')
       CALL W0(' SERCH1 was called with inadequate A,B,C :')
       CALL WR( ' A =', A)
       CALL WR( ' B =', B)
       CALL WR( ' C =', C)
       STOP
    ENDIF

    IF ( .NOT. (QFA.OR.QFB.OR.QFC) ) THEN
       CALL W0(' ')
       CALL W0(' SERCH1 was called with inadequate QFA,QFB,QFC.')
       STOP
    ENDIF

    LFOUND = .TRUE.
    LINVER = .FALSE.

    IF (QFA) THEN
       FA = FUNC(A, PX,PY,PZ, SX,SY,SZ, N, .FALSE., .FALSE., ZERO )
       LASTX = A
    ENDIF
    IF (QFB) THEN
       DDUM1 = ZERO
       IF (QFA)  DDUM1 = ABS(B-A)*SNORM
       FB = FUNC(B, PX,PY,PZ, SX,SY,SZ, N, .FALSE., .FALSE., DDUM1 )
       LASTX = B
    ENDIF
    IF (QFC) THEN
       DDUM1 = ZERO
       IF (QFA)  DDUM1 = ABS(C-A)*SNORM
       IF (QFB)  DDUM1 = ABS(C-B)*SNORM
       FC = FUNC(C, PX,PY,PZ, SX,SY,SZ, N, .FALSE., .FALSE., DDUM1 )
       LASTX = C
    ENDIF

    IF (QMINI) THEN
       FA = -FA
       FB = -FB
       FC = -FC
    ENDIF

    IF ( (FB > FA) .AND. (FB > FC) )  GOTO  4

    IF (FA > FC) THEN
       LINVER = .TRUE.
       DDUM1=A
       A=C
       C=DDUM1
       DDUM1=FA
       FA=FC
       FC=DDUM1
    ENDIF

    STEP = (C-A)/MAX(NSTEP+1,1)
    NSTEPA = MAX( NINT((B-A)/STEP) - 1 , 0 )
    NSTEPC = MAX( NINT((C-B)/STEP) - 1 , 0 )
    STEPA = (A-B)/(NSTEPA+1)
    STEPC = (C-B)/(NSTEPC+1)

    QUPA   = .FALSE.
    QAFIRS = .FALSE.

    C2 =  B
    FC2 = FB
    A2 =  B
    FA2 = FB

    IF ( (FB <= FA) .AND. (FB <= FC) ) THEN
       IF ( ABS(C-B)  <  ABS(B-A) )  QAFIRS = .TRUE.
       QUPC   = .FALSE.
    ELSE
       FINTER = FA + (FC-FA)*(B-A)/(C-A)
       IF ((ABS(C-B) < ABS(B-A)).AND.(FB < FINTER)) QAFIRS=.TRUE.
       QUPC   = .TRUE.
       C1 =  A
       FC1 = FA
    ENDIF

    loop3: DO I = 1, MAX(NSTEPA,NSTEPC)

       IF (QAFIRS)  GOTO  2

1      IF (I  <=  NSTEPC) THEN
          X = B + I*STEPC
          FX = FUNC(X, PX,PY,PZ, SX,SY,SZ, N, .FALSE., .FALSE., &
               ABS(X-LASTX)*SNORM )
          LASTX = X
          IF (QMINI)  FX = -FX
          IF (QUPC .AND. (FX  <  FC2)) THEN
             A =  C1
             FA = FC1
             B =  C2
             FB = FC2
             C =  X
             FC = FX
             GOTO  4
          ENDIF
          IF (FX  >  FC) THEN
             A =  C2
             FA = FC2
             B =  X
             FB = FX
             GOTO  4
          ENDIF
          IF ((I == 1).AND.(.NOT.QAFIRS).AND.(FX < FB)) THEN
             QUPA = .TRUE.
             A1 =  X
             FA1 = FX
          ELSEIF (FX  >  FC2) THEN
             QUPC = .TRUE.
          ENDIF
          C1 =  C2
          FC1 = FC2
          C2 =  X
          FC2 = FX
       ENDIF

       IF (QAFIRS)  cycle loop3

2      IF (I  <=  NSTEPA) THEN
          X = B + I*STEPA
          FX = FUNC(X, PX,PY,PZ, SX,SY,SZ, N, .FALSE., .FALSE., &
               ABS(X-LASTX)*SNORM )
          LASTX = X
          IF (QMINI)  FX = -FX
          IF (QUPA .AND. (FX  <  FA2)) THEN
             A =  X
             FA = FX
             B =  A2
             FB = FA2
             C =  A1
             FC = FA1
             GOTO  4
          ENDIF
          IF ((FX  >  FA) .AND. (FX  >  FA2)) THEN
             B =  X
             FB = FX
             C =  A2
             FC = FA2
             GOTO  4
          ENDIF
          IF ((I == 1).AND.(QAFIRS).AND.(FX < FB)) THEN
             QUPC = .TRUE.
             C1 =  X
             FC1 = FX
          ELSEIF (FX  >  FA2) THEN
             QUPA = .TRUE.
          ENDIF
          A1 =  A2
          FA1 = FA2
          A2 =  X
          FA2 = FX
       ENDIF

       IF (QAFIRS)  GOTO  1

    enddo loop3
    LFOUND = .FALSE.

4   IF (QMINI) THEN
       FA = -FA
       FB = -FB
       FC = -FC
    ENDIF

    IF (LINVER) THEN
       DDUM1=A
       A=C
       C=DDUM1
       DDUM1=FA
       FA=FC
       FC=DDUM1
    ENDIF
    RETURN
  END subroutine serch1

  SUBROUTINE W0(TEXT)

    use chm_kinds

    use stream
#if KEY_PARALLEL==1
    use parallel
#endif 
    implicit none

    CHARACTER(len=*) TEXT

#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif 
       WRITE(OUTU,'(A)')  TEXT
#if KEY_PARALLEL==1
    ENDIF
#endif 

    RETURN
  END SUBROUTINE W0


  SUBROUTINE WR(TEXT,RR)

    use chm_kinds
    use stream
#if KEY_PARALLEL==1
    use parallel
#endif 
    implicit none

    CHARACTER(len=*) TEXT
    real(chm_real)        RR


#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif 
       WRITE(OUTU,'(A,1X,  1PG17.10E2 )')  TEXT, RR


#if KEY_PARALLEL==1
    ENDIF
#endif 

    RETURN
  END SUBROUTINE WR


  SUBROUTINE W2R(TEXT,R1,R2)

    use chm_kinds
    use stream
#if KEY_PARALLEL==1
    use parallel
#endif 
    implicit none

    CHARACTER(len=*) TEXT
    real(chm_real)        R1,R2

#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif 
       WRITE(OUTU,'( A,2(1X, 1PG17.10E2) )')  TEXT, R1,R2
#if KEY_PARALLEL==1
    ENDIF
#endif 

    RETURN
  END SUBROUTINE W2R


  SUBROUTINE W3R(TEXT,R1,R2,R3)

    use chm_kinds
    use stream
#if KEY_PARALLEL==1
    use parallel
#endif 
    implicit none

    CHARACTER(len=*) TEXT
    real(chm_real)        R1,R2,R3

#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif 
       WRITE(OUTU,'( A,3(1X, 1PG13.6E2) )')  TEXT, R1,R2,R3
#if KEY_PARALLEL==1
    ENDIF
#endif 

    RETURN
  END SUBROUTINE W3R


  SUBROUTINE WIR(TEXT,NN,R1)

    use chm_kinds
    use stream
#if KEY_PARALLEL==1
    use parallel
#endif 
    implicit none

    CHARACTER(len=*) TEXT
    INTEGER       NN
    real(chm_real)        R1

#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif 
       WRITE(OUTU,'( A, 1X,I12, 1X,1PG17.10E2 )')  TEXT, NN,R1
#if KEY_PARALLEL==1
    ENDIF
#endif 

    RETURN
  END SUBROUTINE WIR


  SUBROUTINE W3V(TEXT,R1,R2,R3, N)

    use chm_kinds
    use stream
#if KEY_PARALLEL==1
    use parallel
#endif 
    implicit none

    CHARACTER(len=*) TEXT
    real(chm_real)        R1(*),R2(*),R3(*)
    INTEGER       N, I


    IF (N > 10000) THEN
       CALL W0(' ')
       CALL W0('!!! Warning from W3V:  N is too large !!!')
    ENDIF

#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif 
       WRITE(OUTU,'(A)') TEXT
       WRITE(OUTU,1111)  ( R1(I), I = 1,N )
       WRITE(OUTU,1111)  ( R2(I), I = 1,N )
       WRITE(OUTU,1111)  ( R3(I), I = 1,N )
#if KEY_PARALLEL==1
    ENDIF
#endif 

1111 FORMAT( 10000(1PG12.6E1,1X) )

    RETURN
  END SUBROUTINE W3V


  SUBROUTINE WI(TEXT,NN)

    use chm_kinds
    use stream
#if KEY_PARALLEL==1
    use parallel
#endif 
    implicit none

    CHARACTER(len=*) TEXT
    INTEGER       NN

#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif 
       WRITE(OUTU,'(A,1X, I12 )')  TEXT, NN
#if KEY_PARALLEL==1
    ENDIF
#endif 

    RETURN
  END SUBROUTINE WI


  SUBROUTINE W2I(TEXT,N1,N2)

    use chm_kinds
    use stream
#if KEY_PARALLEL==1
    use parallel
#endif 
    implicit none

    CHARACTER(len=*) TEXT
    INTEGER       N1,N2

#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif 
       WRITE(OUTU,'( A,2(1X,I12) )')  TEXT, N1,N2
#if KEY_PARALLEL==1
    ENDIF
#endif 

    RETURN
  END SUBROUTINE W2I


  SUBROUTINE WL(TEXT,QQ)

    use chm_kinds
    use stream
#if KEY_PARALLEL==1
    use parallel
#endif 
    implicit none

    CHARACTER(len=*) TEXT
    LOGICAL       QQ

#if KEY_PARALLEL==1
    IF (MYNOD == 0) THEN
#endif 
       WRITE(OUTU,'(A,1X, L2 )')  TEXT, QQ
#if KEY_PARALLEL==1
    ENDIF
#endif 

    RETURN
  END SUBROUTINE WL

  FUNCTION ENERG( X,Y,Z, DX,DY,DZ, QFIRST,QSECND, STNORM )


    use chm_kinds
    use chm_types
    use number
    use dimens_fcm
    use energym
    use bases_fcm
    use contrl
    use psf
    use exfunc
    use memory
    use travel
#if KEY_PARALLEL==1
    use parallel
#endif 
    use heurist,only:updeci
    implicit none

    real(chm_real),allocatable,dimension(:) :: IDUM1
    real(chm_real),allocatable,dimension(:) :: IDUM2
    real(chm_real),allocatable,dimension(:) :: IDUM3
    real(chm_real),allocatable,dimension(:) :: IDUM4
    real(chm_real),allocatable,dimension(:) :: IDUM5
    real(chm_real),allocatable,dimension(:) :: IDUM6
    real(chm_real),allocatable,dimension(:) :: IDUM7
    LOGICAL   QFIRST, QSECND
    real(chm_real) :: X(:), Y(:), Z(:), DX(:), DY(:), DZ(:), STNORM, energ

    INTEGER   OLDINB, I
    real(chm_real)    XKEEP(natom),YKEEP(natom),ZKEEP(natom)

    IF (LSCAL) THEN
       XKEEP(1:natom) = X(1:natom)
       YKEEP(1:natom) = Y(1:natom)
       ZKEEP(1:natom) = Z(1:natom)
       CALL UNSTRE( X,Y,Z, NATOM )
    ENDIF

    call chmalloc('travel2.src','ENERG','IDUM1',NATOM,crl=IDUM1)
    call chmalloc('travel2.src','ENERG','IDUM2',NATOM,crl=IDUM2)
    call chmalloc('travel2.src','ENERG','IDUM3',NATOM,crl=IDUM3)
    call chmalloc('travel2.src','ENERG','IDUM4',NATOM,crl=IDUM4)
    call chmalloc('travel2.src','ENERG','IDUM5',NATOM,crl=IDUM5)
    call chmalloc('travel2.src','ENERG','IDUM6',NATOM,crl=IDUM6)
    call chmalloc('travel2.src','ENERG','IDUM7',NATOM,crl=IDUM7)

    OLDINB = INBFRQ
    IF (INBFRQ /= 0) THEN
       IF (STNORM < ZERO) THEN
          INBFRQ =  1
       ELSE
          INBFRQ = -1
       ENDIF
    ENDIF

    CALL UPDECI( 1,X,Y,Z,IDUM1,0,IDUM2,IDUM3,IDUM4,IDUM5,IDUM6,IDUM7 )
    INBFRQ = OLDINB

    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
    ENERG = EPROP(EPOT)

#if KEY_PARALLEL==1
    CALL VDGBR(DX,DY,DZ,1)
#endif 

    IF (LSCAL) THEN
       IF (QFIRST)  CALL SCALDERIV(NATOM)
       X(1:natom) = XKEEP(1:natom)
       Y(1:natom) = YKEEP(1:natom)
       Z(1:natom) = ZKEEP(1:natom)
    ENDIF

    call chmdealloc('travel2.src','ENERG','IDUM1',NATOM,crl=IDUM1)
    call chmdealloc('travel2.src','ENERG','IDUM2',NATOM,crl=IDUM2)
    call chmdealloc('travel2.src','ENERG','IDUM3',NATOM,crl=IDUM3)
    call chmdealloc('travel2.src','ENERG','IDUM4',NATOM,crl=IDUM4)
    call chmdealloc('travel2.src','ENERG','IDUM5',NATOM,crl=IDUM5)
    call chmdealloc('travel2.src','ENERG','IDUM6',NATOM,crl=IDUM6)
    call chmdealloc('travel2.src','ENERG','IDUM7',NATOM,crl=IDUM7)

    RETURN
  END function energ

  SUBROUTINE SCALDERIV(NATOM)


    use chm_kinds
    use dimens_fcm
    use deriv
    implicit none

    INTEGER   NATOM

    CALL UNSTRE( DX,DY,DZ, NATOM )

    RETURN
  END subroutine scalderiv

  SUBROUTINE UNSTRE( AX,AY,AZ, N )


    use chm_kinds
    use dimens_fcm
    use travel
    implicit none

    INTEGER   N, I
    real(chm_real)    AX(*),AY(*),AZ(*)

    IF (LSCAL) THEN
       AX(1:n) = AX(1:n)*XSCAL(1:n)
       AY(1:n) = AY(1:n)*YSCAL(1:n)
       AZ(1:n) = AZ(1:n)*ZSCAL(1:n)
    ENDIF

    RETURN
  END subroutine unstre

  SUBROUTINE STRECH( X,Y,Z, N )


    use chm_kinds
    use dimens_fcm
    use travel
    implicit none

    INTEGER   N, I
    real(chm_real)    X(*),Y(*),Z(*)

    IF (LSCAL) THEN
       X(1:n) = X(1:n)/XSCAL(1:n)
       Y(1:n) = Y(1:n)/YSCAL(1:n)
       Z(1:n) = Z(1:n)/ZSCAL(1:n)
    ENDIF

    RETURN
  END subroutine strech

  FUNCTION E1DIM(XX,PX,PY,PZ,SX,SY,SZ,N,QFIRST,QSECND,STNORM)
    use chm_kinds
    use dimens_fcm
    use coord
    use deriv
    implicit none

    LOGICAL   QFIRST, QSECND
    INTEGER   N
    real(chm_real)    XX, PX(*),PY(*),PZ(*), SX(*),SY(*),SZ(*), STNORM
    real(chm_real)    e1dim

    CALL DSUM2( X,Y,Z, PX,PY,PZ, XX, SX,SY,SZ, N )

    E1DIM = ENERG( X,Y,Z, DX,DY,DZ, QFIRST,QSECND, STNORM )

    RETURN
  END FUNCTION E1DIM

  FUNCTION DE1DIM(SX,SY,SZ,SNORM, N)


    use chm_kinds
    use dimens_fcm
    use deriv
    implicit none

    INTEGER   N
    real(chm_real)    SX(*),SY(*),SZ(*),SNORM, de1dim

    DE1DIM = DDOT1( SX,SY,SZ, DX,DY,DZ, N )/SNORM

    RETURN
  END FUNCTION DE1DIM


  SUBROUTINE ORIENT( IDX,XMOV,YMOV,ZMOV, XREF,YREF,ZREF, N, &
       NFIXED,IMGAXI,IMGROT,VERBEU, ISLCT )
    use chm_kinds
    use number
    use dimens_fcm
    use consta
    use exfunc
    use image
    use memory
    use corsubs,only:orintc
    implicit none

    integer,allocatable,dimension(:,:) :: IDUM2
    LOGICAL   IMGROT
    INTEGER   IDX, NFIXED,IMGAXI,VERBEU, N
    real(chm_real)    XMOV(*),YMOV(*),ZMOV(*), XREF(*),YREF(*),ZREF(*)
    INTEGER   ISLCT(*)

    LOGICAL   QVERBO
    INTEGER   I
    real(chm_real)    CX,CY,CZ, ANGLE
    real(chm_real),allocatable,dimension(:) :: dum1,dum3

    QVERBO = (VERBEU >= 3)

    IF (NFIXED > 0)  return

    IF (NTRANS == 0) THEN
       call chmalloc('travel2.src','ORIENT','DUM1',N,crl=DUM1)
       call chmalloc('travel2.src','ORIENT','IDUM2',2,N,intg=IDUM2)
       call chmalloc('travel2.src','ORIENT','DUM3',N,crl=DUM3)
       IF (IDX == 0) THEN
          CALL ORINTC(N, XMOV,YMOV,ZMOV, XREF,YREF,ZREF, &
               DUM1,.FALSE.,.FALSE. ,iDUM2,ISLCT, .FALSE., &
               DUM3,.FALSE.,.TRUE.)
       ELSE
          CALL ORINTC(N, XMOV,YMOV,ZMOV, XREF,YREF,ZREF, &
               DUM1,.FALSE., .TRUE. ,iDUM2,ISLCT, .FALSE., &
               DUM3,.FALSE.,QVERBO)
       ENDIF
       call chmdealloc('travel2.src','ORIENT','DUM1',N,crl=DUM1)
       call chmdealloc('travel2.src','ORIENT','IDUM2',2,N,intg=IDUM2)
       call chmdealloc('travel2.src','ORIENT','DUM3',N,crl=DUM3)
       RETURN
    ENDIF

    IF (IMGAXI == 0)  return

    CX = ZERO
    CY = ZERO
    CZ = ZERO

    IF (IMGROT) THEN
       IF        (IMGAXI == 1) THEN
          DO I = 1,N
             CX = CX + XMOV(I)
          ENDDO
          CX = CX/N
          DO I = 1,N
             XMOV(I) = XMOV(I) - CX
          ENDDO
          IF (QVERBO.OR.IDX == 0) THEN
             CALL W0(' ')
             CALL WR( ' <ORIENT> Point translated by CX =', CX)
          ENDIF
       ELSEIF (IMGAXI == 2) THEN
          DO I = 1,N
             CY = CY + YMOV(I)
          ENDDO
          CY = CY/N
          DO I = 1,N
             YMOV(I) = YMOV(I) - CY
          ENDDO
          IF (QVERBO.OR.IDX == 0) THEN
             CALL W0(' ')
             CALL WR( ' <ORIENT> Point translated by CY =', CY)
          ENDIF
       ELSEIF (IMGAXI == 3) THEN
          DO I = 1,N
             CZ = CZ + ZMOV(I)
          ENDDO
          CZ = CZ/N
          DO I = 1,N
             ZMOV(I) = ZMOV(I) - CZ
          ENDDO
          IF (QVERBO.OR.IDX == 0) THEN
             CALL W0(' ')
             CALL WR( ' <ORIENT> Point translated by CZ =', CZ)
          ENDIF
       ELSE
          CALL WRNDIE(-5,'<ORIENT>','Invalid value for IMGAXI !')
       ENDIF

    ELSE

       DO I = 1,N
          CX = CX + XMOV(I)
          CY = CY + YMOV(I)
          CZ = CZ + ZMOV(I)
       ENDDO
       CX = CX/N
       CY = CY/N
       CZ = CZ/N
       XMOV(1:n) = XMOV(1:n) - CX
       YMOV(1:n) = YMOV(1:n) - CY
       ZMOV(1:n) = ZMOV(1:n) - CZ
       IF (QVERBO.OR.IDX == 0) THEN
          CALL W0(' ')
          CALL WR( ' <ORIENT> Point translated by CX =', CX)
          CALL WR( ' <ORIENT> Point translated by CY =', CY)
          CALL WR( ' <ORIENT> Point translated by CZ =', CZ)
       ENDIF
    ENDIF

    IF (IDX /= 0)  then
       IF        (IMGAXI == 1) THEN
          ANGLE =     LSQFIT2D(YMOV,ZMOV, YREF,ZREF, N)
          CALL ROTA2D(ANGLE, YMOV,ZMOV, N )
       ELSEIF (IMGAXI == 2) THEN
          ANGLE =     LSQFIT2D(XMOV,ZMOV, XREF,ZREF, N)
          CALL ROTA2D(ANGLE, XMOV,ZMOV, N )
       ELSEIF (IMGAXI == 3) THEN
          ANGLE =     LSQFIT2D(XMOV,YMOV, XREF,YREF, N)
          CALL ROTA2D(ANGLE, XMOV,YMOV, N )
       ENDIF

       IF (QVERBO) THEN
          CALL W0(' ')
          CALL WR(' <ORIENT> Point rotated by ANGLE [deg] =', ANGLE*RADDEG)
       ENDIF

    endif
    RETURN
  END subroutine orient

  FUNCTION LSQFIT2D(XMOV,YMOV, XREF,YREF, N)
    use chm_kinds
    use number
    use consta
    implicit none

    INTEGER   N
    real(chm_real)    XMOV(*),YMOV(*), XREF(*),YREF(*),lsqfit2d

    INTEGER   I
    real(chm_real)    NOMI,DENO

    NOMI = ZERO
    DENO = ZERO

    DO I = 1,N
       NOMI = NOMI + XMOV(I)*YREF(I) - YMOV(I)*XREF(I)
       DENO = DENO + XMOV(I)*XREF(I) + YMOV(I)*YREF(I)
    end do

    LSQFIT2D = ATAN(NOMI/DENO)
    IF (DENO < ZERO)  LSQFIT2D = LSQFIT2D + PI

    RETURN
  END FUNCTION LSQFIT2D


  SUBROUTINE ROTA2D( ANGLE, X,Y, N )
    use chm_kinds
    implicit none

    INTEGER    N, I
    real(chm_real)     X(*),Y(*), ANGLE, COSA,SINA, DDUM1

    COSA = COS(ANGLE)
    SINA = SIN(ANGLE)

    DO I = 1,N
       DDUM1 = X(I)
       X(I) = COSA*X(I)   - SINA*Y(I)
       Y(I) = SINA*DDUM1 + COSA*Y(I)
    end do

    RETURN
  END SUBROUTINE ROTA2D

  SUBROUTINE CROSAD( NMAXP,NPOINT, XBAS,YBAS,ZBAS, NATOM,DDIM, &
       IDPREC,IDNEXT,IDXPRO, COMLYN,COMLEN, &
       XS0,YS0,ZS0, DUMYX,DUMYY,DUMYZ, VERBEU, &
       NFREEP,IFREEP, TOTUSD,SERIAL,NLINMN )


    use chm_kinds
    use chm_types
    use dimens_fcm
    use exfunc
    use number
    use consta
    use coord
    use deriv
    use stream
    use string
    implicit none


    type(chm_array) XBAS(:), YBAS(:), ZBAS(:)
    INTEGER      NPOINT,IDXPRO, NATOM, VERBEU
    INTEGER      IDPREC(*),IDNEXT(*)
    real(chm_real)       DUMYX(*),DUMYY(*),DUMYZ(*),  &
         XS0(*),YS0(*),ZS0(*),DDIM
    CHARACTER(len=*) COMLYN
    INTEGER      COMLEN

    INTEGER      NFREEP
    INTEGER      IFREEP(*)
    INTEGER      NLINMN(*)
    INTEGER      SERIAL(*)
    INTEGER      NMAXP
    INTEGER      TOTUSD


    LOGICAL      QREACT,LFOUND, QERROR
    INTEGER      NFIRST,NSTEP
    INTEGER      IFIRST,IPOINT,ICOUNT,I

    real(chm_real)       SFIRST,STEP,ANGLE,ALPHA, LFREQ,LCOUNT
    real(chm_real)       SMALST,LARGST,SMALAL,LARGAL
    real(chm_real)       NEWENE,PRVENE,ESADDL, S0NORM,DNORM

    INTEGER      IDX,ISAD,IDXNEW
    real(chm_real)       DDUM1,DDUM2,DDUM3,DDUM4,DDUM5,DDUM6

    IF (NPOINT /= 3) THEN
       CALL WRNDIE(-4,'<CROSAD>','Number of path-points must be = 3')
       RETURN
    ENDIF

    LFREQ  = GTRMF(COMLYN,COMLEN,'MIND', PT0001 )
    NFIRST = GTRMI(COMLYN,COMLEN,'MINS',     50 )
    ANGLE  = GTRMF(COMLYN,COMLEN,'ANGL', TWENTY )
    SFIRST = GTRMF(COMLYN,COMLEN,'FIRS', RBIG   )
    CALL XTRANE(COMLYN,COMLEN,'CROSAD')

    CALL W0(' ')
    CALL W0(' Barrier crossing mode parameters :')
    CALL W0(' ----------------------------------')
    CALL WR( ' Max. ANGLE [deg] between path & gradient= ', ANGLE)
    IF (SFIRST < RBIG  ) &
         CALL WR( ' Size of first step                      = ',SFIRST)
    CALL WI( ' Min. number of steps before saving point= ',NFIRST)
    CALL WR( ' Min. distance moved  before saving point= ', LFREQ)

    LFREQ  =  LFREQ*SQRT(DDIM)
    SFIRST = SFIRST*SQRT(DDIM)

    QREACT = .FALSE.

1   IF (QREACT) THEN
       ISAD = IDNEXT(1)
       IDX  = 1
       CALL W0(' ')
       CALL W0(' Stepping down the reactant side :')
       CALL W0(' ---------------------------------')
    ELSE
       ISAD = IDPREC(IDXPRO)
       IDX  = IDXPRO
       CALL W0(' ')
       CALL W0(' Stepping down the product side :')
       CALL W0(' ---------------------------------')
    ENDIF

    CALL DSUM2( XS0,YS0,ZS0, &
         XBAS(IDX)%a, YBAS(IDX)%a, ZBAS(IDX)%a, -ONE, &
         XBAS(ISAD)%a,YBAS(ISAD)%a,ZBAS(ISAD)%a, NATOM)

    CALL COP3D( XBAS(ISAD)%a,YBAS(ISAD)%a,ZBAS(ISAD)%a, &
         X,Y,Z, NATOM)
    ESADDL = ENERG( X,Y,Z, DX,DY,DZ, .FALSE., .FALSE., ZERO )

    S0NORM = SQRT( DSDOT1( XS0,YS0,ZS0, NATOM) )
    STEP   = MIN( SFIRST, S0NORM/TWENTY )
    NSTEP  = NINT( S0NORM/STEP )

    IF (VERBEU >= 3) THEN
       CALL WR( ' STEP for local max. search  = ', STEP/SQRT(DDIM))
       CALL WI( ' Number of steps in search   = ', NSTEP)
       CALL W0(' ')
    ENDIF

    LFOUND = .FALSE.
    DDUM1 = ZERO
    DDUM2 = ONE/THREE
    DDUM3 = ONE
    DDUM4 = ESADDL
    DDUM5 = ZERO
    DDUM6 = ZERO

    CALL SERCH1( DDUM1,DDUM2,DDUM3, DDUM4,DDUM5,DDUM6, &
         E1DIM,XBAS(ISAD)%a,YBAS(ISAD)%a,ZBAS(ISAD)%a, &
         XS0,YS0,ZS0, S0NORM, NATOM, &
         NSTEP, .FALSE., .FALSE.,.TRUE.,.TRUE., LFOUND )

    IF (LFOUND) THEN
       CALL DSUM2( X,Y,Z, &
            XBAS(ISAD)%a,YBAS(ISAD)%a,ZBAS(ISAD)%a, &
            DDUM2, XS0,YS0,ZS0, NATOM)
       NEWENE = ENERG( X,Y,Z, DX,DY,DZ, .TRUE., .FALSE., ZERO )

       CALL LNMINI( DDUM1,DDUM2,DDUM3, E1DIM,DE1DIM, &
            DDUM5,DDUM6, DUMYX,DUMYY,DUMYZ, &
            ZERO,ZERO, PT05, VERBEU, &
            XBAS(ISAD)%a,YBAS(ISAD)%a,ZBAS(ISAD)%a, &
            XS0,YS0,ZS0,S0NORM, NATOM,.FALSE.,100, &
            -1, DDUM4)

       NEWENE = DDUM5
       DNORM  = DDUM4
       CALL COP3D( DUMYX,DUMYY,DUMYZ, DX,DY,DZ, NATOM)
       CALL WR( ' Found a local max. @ fractional XS0 = ', DDUM2)
       CALL WR( ' Its E     =', NEWENE)
       CALL WR( ' and DNORM =', DNORM/SQRT(DDIM))

       DDUM2 = MIN( ONE, DDUM2 + STEP/S0NORM )
       CALL DSUM2( X,Y,Z, &
            XBAS(ISAD)%a,YBAS(ISAD)%a,ZBAS(ISAD)%a, &
            DDUM2, XS0,YS0,ZS0, NATOM)
       LCOUNT = ZERO
    ELSE
       CALL DSUM2( X,Y,Z, &
            XBAS(ISAD)%a,YBAS(ISAD)%a,ZBAS(ISAD)%a, &
            STEP/S0NORM, XS0,YS0,ZS0, NATOM)
       NEWENE = ENERG( X,Y,Z, DX,DY,DZ, .TRUE., .FALSE., ZERO )
       DNORM  = SQRT( DSDOT1( DX,DY,DZ, NATOM) )
       LCOUNT = STEP

       IF (NEWENE > ESADDL) THEN
          CALL W0(' No energy decrease in first step down saddle.')
          CALL W0(' !!!  Premature exit. Poor initial points  !!!')
          GOTO  110
       ENDIF
       CALL W0(' First step down saddle decreased E.')
       CALL WR( ' Its E     =', NEWENE)
       CALL WR( ' and DNORM =', DNORM/SQRT(DDIM))
    ENDIF

    CALL W0(' ')
    CALL W0(' Stepping down gradient, keeping ALPHA < ANGLE :')

    CALL COP3D( X,Y,Z, DUMYX,DUMYY,DUMYZ, NATOM)
    CALL COP3D( DX,DY,DZ, XS0,YS0,ZS0, NATOM)
    PRVENE = NEWENE
    S0NORM = DNORM
    SMALST =  RBIG
    SMALAL =  RBIG
    LARGST = -RBIG
    LARGAL = -RBIG
    ICOUNT = 0
    IPOINT = 0
    IFIRST = 0
50  ICOUNT = ICOUNT + 1

    I = 0
51  I = I + 1
    CALL DSUM2( X,Y,Z, DUMYX,DUMYY,DUMYZ, &
         -STEP/S0NORM, XS0,YS0,ZS0, NATOM)
    NEWENE = ENERG( X,Y,Z, DX,DY,DZ, .TRUE., .FALSE., ZERO )

    DNORM = SQRT( DSDOT1( DX,DY,DZ, NATOM) )
    ALPHA = DDOT1( DX,DY,DZ, XS0,YS0,ZS0, NATOM)
    ALPHA = ABS( RADDEG* ACOS(ALPHA/(S0NORM*DNORM)) )

    IF (VERBEU >= 3) THEN
       WRITE(OUTU,1002) ' I1= ',ICOUNT,' I2=',I,' STP= ',STEP/SQRT(DDIM), &
            ' ALF= ',ALPHA,' E= ',NEWENE,' G= ',DNORM/SQRT(DDIM)
1002   FORMAT(A5,I6,A4,I2,A6,1PE8.3E1,A6,1PE8.3E1,2(A4,1PE13.7E1))
    ENDIF

    IF (ALPHA > ANGLE) THEN
       IF (ALPHA > LARGAL.AND.ICOUNT > 1)  LARGAL = ALPHA
       STEP = STEP/TWO
       GOTO  51
    ENDIF

    LCOUNT = LCOUNT + STEP
    IF (ALPHA < SMALAL)  SMALAL = ALPHA
    IF (STEP < SMALST)   SMALST = STEP
    IF (STEP > LARGST)   LARGST = STEP
    IF (NEWENE < ESADDL) IFIRST = IFIRST + 1
    IF (NEWENE > PRVENE) CALL WI( &
         ' Warning: E increased. Expecting decrease. ICOUNT =', ICOUNT )

    IF (LCOUNT >= LFREQ.AND.IFIRST >= NFIRST) THEN
       IPOINT = IPOINT + 1
       LCOUNT = ZERO

       IF (IPOINT == 1.AND. .NOT.QREACT) THEN
          CALL COP3D( X,Y,Z,  XBAS(IDXPRO)%a, &
               YBAS(IDXPRO)%a,ZBAS(IDXPRO)%a,NATOM)
       ELSE

          IDXNEW = GETIDX( QERROR, NFREEP,IFREEP,NPOINT,NMAXP, &
               XBAS,YBAS,ZBAS, NATOM,TOTUSD,SERIAL,NLINMN )
          IF (QERROR)  RETURN

          CALL COP3D( X,Y,Z, &
               XBAS(NPOINT)%a,YBAS(NPOINT)%a,ZBAS(NPOINT)%a, &
               NATOM )

          IF (IPOINT > 1.AND.QREACT) THEN
             IDNEXT(NPOINT)   = NPOINT - 1
             IDPREC(NPOINT-1) = NPOINT
          ELSEIF (IPOINT > 1) THEN
             IDNEXT(IDXPRO) = NPOINT
             IDPREC(NPOINT) = IDXPRO
             IDXPRO = NPOINT
          ELSEIF (IPOINT == 1.AND.QREACT) THEN
             IDPREC(IDNEXT(1)) = NPOINT
             IDNEXT(NPOINT)    = IDNEXT(1)
          ENDIF
       ENDIF

       GOTO  100
    ENDIF

    CALL COP3D( X,Y,Z, DUMYX,DUMYY,DUMYZ, NATOM)
    PRVENE = NEWENE
    CALL COP3D( DX,DY,DZ, XS0,YS0,ZS0, NATOM)
    S0NORM = DNORM
    STEP   = STEP*SQRT(TWO)

    GOTO  50
100 CALL W0(' ')
    CALL WR( ' Smallest step  on this side = ', SMALST/SQRT(DDIM))
    CALL WR( ' Largest  step  on this side = ', LARGST/SQRT(DDIM))
    CALL WR( ' Smallest angle on this side = ', SMALAL)
    CALL WR( ' Largest  angle on this side = ', LARGAL)

110 IF (.NOT.QREACT) THEN
       QREACT = .TRUE.
       GOTO  1
    ENDIF

    CALL W0(' ')
    CALL W0(' Normal completion of reactive mode calculation.')

    CALL COP3D( &
         XBAS(NPOINT)%a,YBAS(NPOINT)%a,ZBAS(NPOINT)%a, &
         XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, NATOM )

    IDNEXT(1) = IDNEXT(NPOINT)
    IDPREC(IDNEXT(1)) = 1

    NFREEP = NFREEP + 1
    IFREEP(NFREEP) = NPOINT
    NPOINT = NPOINT - 1

    CALL WI( ' Total number of path-points, NPOINT = ', NPOINT)

    RETURN
  END SUBROUTINE CROSAD

  SUBROUTINE COP3D( FROMX, FROMY, FROMZ, TOX, TOY, TOZ , N )
    use chm_kinds
    implicit none

    INTEGER    N, I
    real(chm_real) FROMX(*),FROMY(*),FROMZ(*), TOX(*), TOY(*), TOZ(*)

    TOX(1:n) = FROMX(1:n)
    TOY(1:n) = FROMY(1:n)
    TOZ(1:n) = FROMZ(1:n)

    RETURN
  END SUBROUTINE COP3D


  FUNCTION DDOT1( AX,AY,AZ, BX,BY,BZ, N )
    use chm_kinds
    use exfunc
    use memory
    implicit none

    real(chm_real),allocatable,dimension(:) :: IDUMX
    real(chm_real),allocatable,dimension(:) :: IDUMY
    real(chm_real),allocatable,dimension(:) :: IDUMZ
    INTEGER    N
    real(chm_real) AX(*),AY(*),AZ(*), BX(*),BY(*),BZ(*),ddot1

    call chmalloc('travel2.src','DDOT1','IDUMX',N,crl=IDUMX)
    call chmalloc('travel2.src','DDOT1','IDUMY',N,crl=IDUMY)
    call chmalloc('travel2.src','DDOT1','IDUMZ',N,crl=IDUMZ)

    CALL COP3D(  AX,AY,AZ,IDUMX,IDUMY,IDUMZ, N)

    DDOT1 = DDOT2( IDUMX,IDUMY,IDUMZ,BX,BY,BZ, N )

    call chmdealloc('travel2.src','DDOT1','IDUMX',N,crl=IDUMX)
    call chmdealloc('travel2.src','DDOT1','IDUMY',N,crl=IDUMY)
    call chmdealloc('travel2.src','DDOT1','IDUMZ',N,crl=IDUMZ)

    RETURN
  END FUNCTION DDOT1

  FUNCTION DSDOT1( X,Y,Z, N )
    use chm_kinds
    use exfunc
    use memory
    implicit none

    real(chm_real),allocatable,dimension(:) :: IDUMX
    real(chm_real),allocatable,dimension(:) :: IDUMY
    real(chm_real),allocatable,dimension(:) :: IDUMZ
    INTEGER    N
    real(chm_real)     X(*),Y(*),Z(*), dsdot1

    call chmalloc('travel2.src','DSDOT1','IDUMX',N,crl=IDUMX)
    call chmalloc('travel2.src','DSDOT1','IDUMY',N,crl=IDUMY)
    call chmalloc('travel2.src','DSDOT1','IDUMZ',N,crl=IDUMZ)

    CALL COP3D(  X,Y,Z,IDUMX,IDUMY,IDUMZ, N)

    DSDOT1 = DDOT2(  IDUMX,IDUMY,IDUMZ,X,Y,Z, N )

    call chmdealloc('travel2.src','DSDOT1','IDUMX',N,crl=IDUMX)
    call chmdealloc('travel2.src','DSDOT1','IDUMY',N,crl=IDUMY)
    call chmdealloc('travel2.src','DSDOT1','IDUMZ',N,crl=IDUMZ)

    RETURN
  END FUNCTION DSDOT1

  FUNCTION DSDOT2( X,Y,Z, N )
    use chm_kinds
    implicit none

    INTEGER    N
    real(chm_real)     X(*),Y(*),Z(*), dsdot2

    DSDOT2 = DDOT2( X,Y,Z, X,Y,Z, N )

    RETURN
  END FUNCTION DSDOT2

  FUNCTION DDOT2( AX,AY,AZ, BX,BY,BZ, N )
    use chm_kinds
    use number
    implicit none

    INTEGER    N, I
    real(chm_real)     AX(*),AY(*),AZ(*), BX(*),BY(*),BZ(*), ddot2

    DDOT2 = ZERO

    DO I = 1,N
       AX(I) = AX(I)*BX(I)
       AY(I) = AY(I)*BY(I)
       AZ(I) = AZ(I)*BZ(I)
    enddo

    DO I = 1,N
       DDOT2 = DDOT2 + AX(I) + AY(I) + AZ(I)
    end do
    RETURN
  END FUNCTION DDOT2

  FUNCTION VANGLD( AX,AY,AZ,ANORM, SX,SY,SZ,SNORM, N )
    use chm_kinds
    use number
    use consta
    implicit none

    INTEGER    N
    real(chm_real) vangld
    real(chm_real) AX(*),AY(*),AZ(*), SX(*),SY(*),SZ(*), ANORM,SNORM

    IF (ANORM * SNORM <= ZERO) THEN
       CALL WRNDIE(1,'<VANGLD>','ANORM or SNORM = 0, cannot compute'// &
            ' angle between vectors.')
       VANGLD = -ONE
       RETURN
    ENDIF

    VANGLD = DDOT1( AX,AY,AZ, SX,SY,SZ, N )
    VANGLD = VANGLD/(ANORM*SNORM)
    IF ( VANGLD < -ONE )  VANGLD = -ONE
    IF ( VANGLD >  ONE )  VANGLD =  ONE
    VANGLD = RADDEG*ACOS( VANGLD )

    RETURN
  END FUNCTION VANGLD

  SUBROUTINE PROJKT( AX,AY,AZ,ANORM, SX,SY,SZ,SNORM, N )
    use chm_kinds
    implicit none

    INTEGER    N
    real(chm_real) AX(*),AY(*),AZ(*), SX(*),SY(*),SZ(*), ANORM,SNORM
    real(chm_real)     DOTPRO

    DOTPRO = DDOT1( AX,AY,AZ, SX,SY,SZ, N )
    CALL DSUM2( AX,AY,AZ, AX,AY,AZ, -DOTPRO/SNORM**2, SX,SY,SZ, N)
    ANORM = SQRT(ABS( ANORM**2 - (DOTPRO/SNORM)**2 ))
    RETURN
  END SUBROUTINE PROJKT

  SUBROUTINE MULTD2( A, X,Y,Z, N )
    use chm_kinds
    implicit none
    INTEGER    N, I
    real(chm_real)     A, X(n),Y(n),Z(n)

    X(1:n) = A*X(1:n)
    Y(1:n) = A*Y(1:n)
    Z(1:n) = A*Z(1:n)
    RETURN
  END SUBROUTINE MULTD2

  SUBROUTINE DSUM1( X,Y,Z, A, AX,AY,AZ, B, BX,BY,BZ, N )
    use chm_kinds
    implicit none

    INTEGER    N, I
    real(chm_real) X(*),Y(*),Z(*),BX(*),BY(*),BZ(*),AX(*),AY(*),AZ(*)
    real(chm_real)     A, B

    X(1:n) = A*AX(1:n) + B*BX(1:n)
    Y(1:n) = A*AY(1:n) + B*BY(1:n)
    Z(1:n) = A*AZ(1:n) + B*BZ(1:n)
    RETURN
  END SUBROUTINE DSUM1

  SUBROUTINE DSUM2( X,Y,Z, AX,AY,AZ, B, BX,BY,BZ, N )
    use chm_kinds
    implicit none

    INTEGER    N, I
    real(chm_real) X(*),Y(*),Z(*),BX(*),BY(*),BZ(*), AX(*),AY(*),AZ(*)
    real(chm_real)     B

    X(1:n) = AX(1:n) + B*BX(1:n)
    Y(1:n) = AY(1:n) + B*BY(1:n)
    Z(1:n) = AZ(1:n) + B*BZ(1:n)
    RETURN
  END subroutine dsum2

#if KEY_UNUSED==1 /*dsum3_unused*/

  SUBROUTINE DSUM3( X,Y,Z, AX,AY,AZ, BX,BY,BZ, N )

    use chm_kinds
    implicit none

    INTEGER    N, I
    real(chm_real) X(*),Y(*),Z(*),BX(*),BY(*),BZ(*), AX(*),AY(*),AZ(*)

    X(1:n) = AX(1:n) + BX(1:n)
    Y(1:n) = AY(1:n) + BY(1:n)
    Z(1:n) = AZ(1:n) + BZ(1:n)
    RETURN
  END subroutine dsum3

#endif /* (dsum3_unused)*/

  SUBROUTINE INII2(ARRAY,N,INTEG4)
    use chm_kinds
    implicit none

    INTEGER     N, I
    INTEGER     ARRAY(*), INTEG4

    ARRAY(1:n) = INTEG4
    RETURN
  END SUBROUTINE INII2

  SUBROUTINE CHKFIX( X1,Y1,Z1, X2,Y2,Z2, NMOVED )


    use chm_kinds
    use number
    use dimens_fcm
    use psf
    implicit none

    INTEGER    NMOVED, I
    real(chm_real)     X1(*),Y1(*),Z1(*), X2(*),Y2(*),Z2(*)

    NMOVED = 0

    DO  I = 1, NATOM
       IF ( (ABS(X2(I)-X1(I)) > TENM5 .OR. &
            ABS(Y2(I)-Y1(I)) > TENM5 .OR. &
            ABS(Z2(I)-Z1(I)) > TENM5   ) .AND. IMOVE(I) > 0) THEN
          NMOVED = NMOVED + 1
          X2(I) = X1(I)
          Y2(I) = Y1(I)
          Z2(I) = Z1(I)
       ENDIF
    end DO
    RETURN
  END SUBROUTINE CHKFIX

  SUBROUTINE SDPATH( NMAXP, NPOINT, XBAS,YBAS,ZBAS, NATOM,DDIM, &
       IDPREC,IDNEXT,IDXPRO, COMLYN,COMLEN, &
       XS0,YS0,ZS0, DUMYX,DUMYY,DUMYZ, VERBEU, &
       NFREEP,IFREEP,TOTUSD,SERIAL,NLINMN )
    use chm_kinds
    use chm_types
    use dimens_fcm
    use exfunc
    use number
    use consta
    use coord
    use deriv
    use energym
    use memory
    use stream
    use string

    implicit none
    real(chm_real),allocatable,dimension(:) :: DKX,DKY,DKZ,DOLDX,DOLDY,DOLDZ
    INTEGER      NMAXP,NPOINT,IDXPRO, NATOM, VERBEU
    INTEGER      IDPREC(*),IDNEXT(*)
    type(chm_array) XBAS(:), YBAS(:), ZBAS(:)
    real(chm_real) DUMYX(*),DUMYY(*),DUMYZ(*),  &
         XS0(*),YS0(*),ZS0(*),DDIM
    CHARACTER(len=*) COMLYN
    INTEGER      COMLEN

    INTEGER      NFREEP
    INTEGER      IFREEP(*)
    INTEGER      NLINMN(*)
    INTEGER      SERIAL(*)
    INTEGER      TOTUSD

    INTEGER      IMODE

    INTEGER      NREACT,NPROD
    INTEGER      IPOINT,ICOUNT, NREADY

    real(chm_real)       STEP, LFREQ,LCOUNT
    real(chm_real)       SMALST,LARGST, MINGRA, S0NORM
    LOGICAL      QREACT,QDONE,QWRONG,QREADY, QERROR

    INTEGER      BRKCYC,MODXIT, INICAL
    real(chm_real)       AX,BX,CX, FA,FB,FC,MAXANG,NEWENE,DNORM,ALPHA
    real(chm_real)       STPTOL,ENETOL,GRATOL,BRASCA,SFIRST,DNORM2

    INTEGER      ISAD, LSTECAL,IDXNEW
    real(chm_real)       DDUM1,DDUM2

    IF (NPOINT < 1) THEN
       CALL WRNDIE(-1,'<SDPATH>','Number of path-points must be > 0')
       RETURN
    ENDIF

    call chmalloc('travel2.src','SDPATH','DKX',NATOM,crl=DKX)
    call chmalloc('travel2.src','SDPATH','DKY',NATOM,crl=DKY)
    call chmalloc('travel2.src','SDPATH','DKZ',NATOM,crl=DKZ)
    call chmalloc('travel2.src','SDPATH','DOLDX',NATOM,crl=DOLDX)
    call chmalloc('travel2.src','SDPATH','DOLDY',NATOM,crl=DOLDY)
    call chmalloc('travel2.src','SDPATH','DOLDZ',NATOM,crl=DOLDZ)

    IMODE   = GTRMI(COMLYN,COMLEN,'MODE',  4 )

    IF        (IMODE == 1) THEN
       CALL W0(' ')
       CALL W0(' Descent mode 1 : path/gradient angle.')
       CALL W0(' --------------------------------------')
       MAXANG = GTRMF(COMLYN,COMLEN,'ANGL', THIRTY )
       SFIRST = GTRMF(COMLYN,COMLEN,'FIRS', TENM5 )
       BRASCA = GTRMF(COMLYN,COMLEN,'BRKS',  TWO**PT25 )
    ELSEIF (IMODE == 2) THEN
       CALL W0(' ')
       CALL W0(' Descent mode 2 : strict steepest-descent.')
       CALL W0(' -----------------------------------------')
       SFIRST = GTRMF(COMLYN,COMLEN,'FIRS', PT0001 )
       BRASCA = GTRMF(COMLYN,COMLEN,'BRKS',  TWO   )
       STPTOL = GTRMF(COMLYN,COMLEN,'TOLS', TENM5*TENM5 )
       ENETOL = GTRMF(COMLYN,COMLEN,'TOLE', PT001*PT0001 )
       GRATOL = GTRMF(COMLYN,COMLEN,'TOLG', PTONE )
       MODXIT = GTRMI(COMLYN,COMLEN,'EXIT',      3 )
    ELSEIF (IMODE == 3) THEN
       CALL W0(' ')
       CALL W0(' Descent mode 3 : loose steepest-descent.')
       CALL W0(' ----------------------------------------')
       SFIRST = GTRMF(COMLYN,COMLEN,'FIRS', PT0001 )
       BRASCA = GTRMF(COMLYN,COMLEN,'BRKS', TWO**PT25 )
    ELSEIF (IMODE == 4) THEN
       CALL W0(' ')
       CALL W0(' Descent mode 4 : conjugate-gradient descent.')
       CALL W0(' --------------------------------------------')
       SFIRST = GTRMF(COMLYN,COMLEN,'FIRS', PT0001 )
       BRASCA = GTRMF(COMLYN,COMLEN,'BRKS',  TWO   )
       STPTOL = GTRMF(COMLYN,COMLEN,'TOLS', TENM5*TENM5 )
       ENETOL = GTRMF(COMLYN,COMLEN,'TOLE', PT001*PT0001 )
       GRATOL = GTRMF(COMLYN,COMLEN,'TOLG', PTONE  )
       MODXIT = GTRMI(COMLYN,COMLEN,'EXIT',      3 )
    ELSE
       CALL WRNDIE(0,'<SDPATH>', &
            ' Required options : IMODE [1,2,3,4]')
    ENDIF

    LFREQ  = GTRMF(COMLYN,COMLEN,'SAVD', PT05   )
    NPROD  = INT( (NMAXP-NPOINT)/2 )
    NPROD  = GTRMI(COMLYN,COMLEN,'NPRO', NPROD  )
    NPROD  = MIN( NPROD, NMAXP-NPOINT )
    NREACT = NMAXP-NPOINT-1-NPROD
    NREACT = GTRMI(COMLYN,COMLEN,'NREA', NREACT )
    NREACT = MIN( NREACT, NMAXP-NPOINT-1-NPROD )
    MINGRA = GTRMF(COMLYN,COMLEN,'MING', PT001  )
    NREADY = GTRMI(COMLYN,COMLEN,'MINC', 50 )

    CALL XTRANE(COMLYN,COMLEN,'SDPATH')

    CALL W0(' ')
    CALL W0(' Steepest-Descent Path parameters :')
    CALL W0(' ----------------------------------')
    CALL WR( ' Distance between points saved           = ', LFREQ)
    CALL WI( ' On product  side, saving points up to   = ', NPROD)
    CALL WI( ' On reactant side, saving points up to   = ',NREACT)
    CALL WR( ' Stopping when GRMS in minimum less than = ',MINGRA)
    CALL WI( ' but not before doing a MINimum of Cycles= ',NREADY)
    CALL W0(' ')
    CALL W0(' Braketing and line-minimization parameters :')
    CALL W0(' --------------------------------------------')
    IF        (IMODE == 1) THEN
       CALL WR( ' MAX.ANGle [deg] between path & gradient=',MAXANG)
    ELSEIF (IMODE == 2.OR.IMODE == 4) THEN
       CALL WR( ' TOLSTeP          = ', STPTOL)
       CALL WR( ' GRATOLd          = ', GRATOL)
       CALL WR( ' ENETOLr          = ', ENETOL)
       CALL WI( ' eMODXITe         = ', MODXIT)
    ENDIF
    CALL WR( ' First step       = ', SFIRST)
    CALL WR( ' Step up-scale    = ', BRASCA)

    LFREQ  =  LFREQ*SQRT(DDIM)
    MINGRA = MINGRA*SQRT(DDIM)
    SFIRST = SFIRST*SQRT(DDIM)
    STPTOL = STPTOL*SQRT(DDIM)

    QREACT = .FALSE.

1   IF (QREACT.AND.NREACT > 0) THEN
       CALL W0(' ')
       CALL W0(' Stepping down the reactant side :')
       CALL W0(' ---------------------------------')
       IDXNEW = GETIDX( QERROR, NFREEP,IFREEP,NPOINT,NMAXP, &
            XBAS,YBAS,ZBAS, NATOM,TOTUSD,SERIAL,NLINMN )
       IF (QERROR)  RETURN

       CALL COP3D( XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, &
            XBAS(IDXNEW)%a,YBAS(IDXNEW)%a,ZBAS(IDXNEW)%a, &
            NATOM )
       IDNEXT(IDXNEW)    = IDNEXT(1)
       IDPREC(IDNEXT(1)) = IDXNEW
       ISAD = IDXNEW
    ELSEIF (.NOT.QREACT.AND.NPROD > 0) THEN
       CALL W0(' ')
       CALL W0(' Stepping down the product side :')
       CALL W0(' ---------------------------------')
       ISAD = IDXPRO
    ELSE
       GOTO  110
    ENDIF

    SMALST =  RBIG
    LARGST = -RBIG
    ICOUNT = 0
    LCOUNT = ZERO
    IPOINT = 0
    QDONE  = .FALSE.
    QWRONG = .FALSE.
    INICAL = ECALLS
    QREADY = .FALSE.

    CALL COP3D( XBAS(ISAD)%a,YBAS(ISAD)%a,ZBAS(ISAD)%a, &
         X,Y,Z, NATOM)
    CALL COP3D( X,Y,Z, DUMYX,DUMYY,DUMYZ, NATOM)
    FB = ENERG( X,Y,Z, DX,DY,DZ, .TRUE., .FALSE., ZERO )
    CALL COP3D( DX,DY,DZ, XS0,YS0,ZS0, NATOM)
    CALL COP3D(DX,DY,DZ, DOLDX,DOLDY,DOLDZ,NATOM)
    DNORM2 = DSDOT1( DX,DY,DZ, NATOM)
    S0NORM = SQRT( DSDOT1( XS0,YS0,ZS0, NATOM) )
    STEP = SFIRST/BRASCA

    IF (VERBEU >= 3) THEN
       CALL W0(' ')
       WRITE(OUTU,1001) 'I=',ICOUNT,' STEP=',ZERO,' E=',FB, &
            ' DE=',SQRT(DNORM2/DDIM),' ECALL=',0
    ENDIF

50  ICOUNT = ICOUNT + 1
    LSTECAL = ECALLS

    IF (IMODE == 1) THEN

51     CALL DSUM2( X,Y,Z, DUMYX,DUMYY,DUMYZ, &
            -STEP/S0NORM, XS0,YS0,ZS0, NATOM)
       NEWENE = ENERG( X,Y,Z, DX,DY,DZ, .TRUE., .FALSE., ZERO )

       DNORM2 = DSDOT1( DX,DY,DZ, NATOM)
       DNORM = SQRT(DNORM2)
       ALPHA = DDOT1( DX,DY,DZ, XS0,YS0,ZS0, NATOM)
       ALPHA = ABS( RADDEG* ACOS(ALPHA/(S0NORM*DNORM)) )

       IF (ALPHA > MAXANG.OR.NEWENE > FB) THEN
          STEP = STEP/TWO
          GOTO  51
       ENDIF

       LCOUNT = LCOUNT + STEP
       IF (STEP < SMALST)  SMALST = STEP
       IF (STEP > LARGST)  LARGST = STEP
       STEP = STEP*BRASCA

       CALL COP3D( X,Y,Z, DUMYX,DUMYY,DUMYZ, NATOM)
       CALL COP3D( DX,DY,DZ, XS0,YS0,ZS0, NATOM)
       FB = NEWENE

    ENDIF
    IF (IMODE == 2) THEN

       AX = ZERO
       BX = -(STEP*BRASCA)/S0NORM
       FA = FB
       BRKCYC = 0
       CALL BRAKET( AX,BX,CX, FA,FB,FC, E1DIM, FIVE, VERBEU, &
            DUMYX,DUMYY,DUMYZ, XS0,YS0,ZS0, &
            DKX,DKY,DKZ, S0NORM, &
            NATOM , .TRUE., BRKCYC)

       IF (CX > ZERO) THEN
          IF (LCOUNT > ZERO.AND.ICOUNT > 1) THEN
             QWRONG = .TRUE.
             GOTO  60
          ELSE
             CALL W0(' ')
             CALL W0(' Warning : braketing in the wrong direction.')
             GOTO  100
          ENDIF
       ENDIF

       IF ( BX == ZERO ) THEN
          CALL COP3D( XS0,YS0,ZS0, DX,DY,DZ, NATOM)
       ELSE
          CALL COP3D( DKX,DKY,DKZ, DX,DY,DZ, NATOM)
       ENDIF

       IF ( ABS(MODXIT) > 1 ) &
            CALL COP3D( XS0,YS0,ZS0, DKX,DKY,DKZ, NATOM)

       CALL LNMINI( AX,BX,CX, E1DIM,DE1DIM, FB,DDUM1, &
            DKX,DKY,DKZ, STPTOL, &
            ENETOL,GRATOL, VERBEU, DUMYX,DUMYY,DUMYZ, &
            XS0,YS0,ZS0,S0NORM, NATOM, .TRUE. , &
            20, MODXIT, DDUM2)

       STEP = ABS(BX)*S0NORM
       IF (STEP < SMALST)  SMALST = STEP
       IF (STEP > LARGST)  LARGST = STEP
       LCOUNT = LCOUNT + STEP

       CALL DSUM2( DUMYX,DUMYY,DUMYZ, DUMYX,DUMYY,DUMYZ, &
            -STEP/S0NORM, XS0,YS0,ZS0, NATOM)
       CALL COP3D( DKX,DKY,DKZ, XS0,YS0,ZS0, NATOM)

       DNORM2 = DSDOT1( DKX,DKY,DKZ, NATOM)

    ENDIF
    IF (IMODE == 3) THEN

52     CALL DSUM2( X,Y,Z, DUMYX,DUMYY,DUMYZ, &
            -STEP/S0NORM, XS0,YS0,ZS0, NATOM)
       NEWENE = ENERG( X,Y,Z, DX,DY,DZ, .TRUE., .FALSE., ZERO )

       IF (NEWENE > FB) THEN
          STEP = STEP/TWO
          GOTO  52
       ENDIF

       LCOUNT = LCOUNT + STEP
       IF (STEP < SMALST)  SMALST = STEP
       IF (STEP > LARGST)  LARGST = STEP
       STEP = STEP*BRASCA

       CALL COP3D( X,Y,Z, DUMYX,DUMYY,DUMYZ, NATOM)
       CALL COP3D( DX,DY,DZ, XS0,YS0,ZS0, NATOM)
       FB = NEWENE

       DNORM2 = DSDOT1( DX,DY,DZ, NATOM)

    ENDIF
    IF (IMODE == 4) THEN

       AX = ZERO
       BX = -(STEP*BRASCA)/S0NORM
       FA = FB
       BRKCYC = 0
       CALL BRAKET( AX,BX,CX, FA,FB,FC, E1DIM, FIVE, VERBEU, &
            DUMYX,DUMYY,DUMYZ, XS0,YS0,ZS0, &
            DKX,DKY,DKZ, S0NORM, &
            NATOM , .TRUE., BRKCYC)

       IF (CX > ZERO) THEN
          IF (LCOUNT > ZERO.AND.ICOUNT > 1) THEN
             QWRONG = .TRUE.
             GOTO  60
          ELSE
             CALL W0(' ')
             CALL W0(' Warning : braketing in the wrong direction.')
             GOTO  100
          ENDIF
       ENDIF

       IF ( BX == ZERO ) THEN
          CALL COP3D(DOLDX,DOLDY,DOLDZ,DX,DY,DZ,NATOM)
       ELSE
          CALL COP3D( DKX,DKY,DKZ, DX,DY,DZ, NATOM)
       ENDIF

       IF ( ABS(MODXIT) > 1 )  CALL COP3D( DOLDX,DOLDY, &
            DOLDZ,DKX,DKY,DKZ, NATOM )

       CALL LNMINI( AX,BX,CX, E1DIM,DE1DIM, FB,DDUM1, &
            DKX,DKY,DKZ, STPTOL, &
            ENETOL,GRATOL, VERBEU, DUMYX,DUMYY,DUMYZ, &
            XS0,YS0,ZS0,S0NORM, NATOM, .TRUE. , &
            20, MODXIT, DDUM2)

       STEP = ABS(BX)*S0NORM
       IF (STEP < SMALST)  SMALST = STEP
       IF (STEP > LARGST)  LARGST = STEP
       LCOUNT = LCOUNT + STEP

       CALL DSUM2( DUMYX,DUMYY,DUMYZ, DUMYX,DUMYY,DUMYZ, &
            -STEP/S0NORM, XS0,YS0,ZS0, NATOM)

       DDUM1 = DDOT2( DOLDX,DOLDY,DOLDZ, &
            DKX,DKY,DKZ, NATOM)
       CALL COP3D( DKX,DKY,DKZ, &
            DOLDX,DOLDY,DOLDZ, NATOM)
       DDUM2 = DSDOT2( DKX,DKY,DKZ, NATOM)
       CALL DSUM2( XS0,YS0,ZS0, DOLDX,DOLDY,DOLDZ, &
            (DDUM2-DDUM1)/DNORM2, XS0,YS0,ZS0, NATOM)
       DNORM2 = DDUM2

    ENDIF

    S0NORM = SQRT( DSDOT1( XS0,YS0,ZS0, NATOM) )

    IF (VERBEU >= 3) THEN
       WRITE(OUTU,1001) 'I=',ICOUNT,' STEP=',STEP/SQRT(DDIM),' E=',FB, &
            ' DE=',SQRT(DNORM2/DDIM),' ECALL=',ECALLS-LSTECAL
1001   FORMAT(A2,I5,A6,1PE13.7E1,A3,1PE13.7E1,A4,1PE13.7E1,A7,I3)
    ENDIF

    IF ( (ICOUNT >= NREADY) .OR. &
         (SQRT(DNORM2) > 1.1D0*MINGRA) )  QREADY = .TRUE.

    IF (SQRT(DNORM2) <= MINGRA.AND.QREADY)  QDONE = .TRUE.

60  IF ( QDONE .OR. LCOUNT >= LFREQ .OR. QWRONG) THEN
       IPOINT = IPOINT + 1
       LCOUNT = ZERO

       IDXNEW = GETIDX( QERROR, NFREEP,IFREEP,NPOINT,NMAXP, &
            XBAS,YBAS,ZBAS, NATOM,TOTUSD,SERIAL,NLINMN )
       IF (QERROR)  RETURN

       CALL COP3D( DUMYX,DUMYY,DUMYZ, &
            XBAS(IDXNEW)%a,YBAS(IDXNEW)%a,ZBAS(IDXNEW)%a, &
            NATOM )
       IF (QREACT) THEN
          IDNEXT(IDXNEW)   = NPOINT - 1
          IDPREC(NPOINT-1) = IDXNEW
       ELSE
          IDNEXT(IDXPRO) = IDXNEW
          IDPREC(IDXNEW) = IDXPRO
          IDXPRO = IDXNEW
       ENDIF

       IF (QDONE) THEN
          CALL W0(' ')
          CALL W0(' Minimum reached satisfies MINGRA.')
          GOTO  100
       ELSEIF ( (IPOINT == NREACT .AND.QREACT)  .OR. &
            (IPOINT == NPROD  .AND. .NOT.QREACT) ) THEN
          CALL W0(' ')
          CALL W0(' Added max. number of points on this side.')
          GOTO  100
       ELSEIF (QWRONG) THEN
          CALL W0(' ')
          CALL W0(' Warning : braketing in the wrong direction.')
          GOTO  100
       ENDIF
    ENDIF

    GOTO  50
100 CALL W0(' ')
    CALL WR( ' Smallest step  on this side = ', SMALST/SQRT(DDIM))
    CALL WR( ' Largest  step  on this side = ', LARGST/SQRT(DDIM))
    CALL WI( ' Number of points added IPOINT= ', IPOINT)
    CALL WI( ' Number of steps taken  ICOUNT= ', ICOUNT)
    CALL WI( ' Number of ENERGy calls       = ', ECALLS - INICAL)

110 IF (.NOT.QREACT) THEN
       QREACT = .TRUE.
       GOTO  1
    ENDIF

    CALL W0(' ')
    CALL W0(' <SDPATH> Normal exit.')

    IF (NREACT == 0)  GOTO  202

    CALL COP3D( &
         XBAS(NPOINT)%a,YBAS(NPOINT)%a,ZBAS(NPOINT)%a, &
         XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, NATOM )
    IDNEXT(1) = IDNEXT(NPOINT)
    IDPREC(IDNEXT(1)) = 1

    NFREEP = NFREEP + 1
    IFREEP(NFREEP) = NPOINT
    NPOINT = NPOINT - 1

202 CALL WI( ' Total number of path-points, NPOINT = ', NPOINT)

    call chmdealloc('travel2.src','SDPATH','DKX',NATOM,crl=DKX)
    call chmdealloc('travel2.src','SDPATH','DKY',NATOM,crl=DKY)
    call chmdealloc('travel2.src','SDPATH','DKZ',NATOM,crl=DKZ)
    call chmdealloc('travel2.src','SDPATH','DOLDX',NATOM,crl=DOLDX)
    call chmdealloc('travel2.src','SDPATH','DOLDY',NATOM,crl=DOLDY)
    call chmdealloc('travel2.src','SDPATH','DOLDZ',NATOM,crl=DOLDZ)

    RETURN
  END SUBROUTINE SDPATH


  ! sgscan moved from travel.src for circular dependencies

  SUBROUTINE SGSCAN( LLL,I,IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
       LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
       SRTENE,SRTIDX,SEGSTP,PTGRAD, &
       XBAS,YBAS,ZBAS, NATOM,VERBEU )
    use chm_kinds
    use chm_types
    use number
    use dimens_fcm
    use coord
    use deriv
    use exfunc
    use memory
    use stream
    use trek1
    implicit none

    real(chm_real),allocatable,dimension(:) :: IDUMX,IDUMY,IDUMZ
    LOGICAL   LSCAN1,LSCAN2,LSCAN3, QUP(*),QDOWN(*)
    INTEGER   LLL,I,IDNEXT(*),NMAXI, STPMAX(*),NEXMIN(*),PREMIN(*)
    INTEGER   SRTIDX(*), NATOM,VERBEU
    type(chm_array)   XBAS(:),YBAS(:),ZBAS(:)
    real(chm_real)    PTENE(*),LENSEG(*),ENEMAX(*),SRTENE(*),SEGSTP(*)
    real(chm_real)    PTGRAD(*)


    LOGICAL   QPRVUP
    INTEGER   IFL, NTERPL,STMPMX,SSECMX
    INTEGER   J,SMIN1,SMIN2,IDIM
    real(chm_real)    JRN,STPSIZ,ETMPMX,ESECMX,PRVENE,DDUM1,E,DDIM

    INTEGER, PARAMETER :: BIGINT=999999

    call chmalloc('travel.src','SGSCAN','IDUMX',NATOM,crl=IDUMX)
    call chmalloc('travel.src','SGSCAN','IDUMY',NATOM,crl=IDUMY)
    call chmalloc('travel.src','SGSCAN','IDUMZ',NATOM,crl=IDUMZ)

    IFL       = IDNEXT(I)
    STPMAX(I) =  BIGINT
    ENEMAX(I) = -RBIG
    QDOWN(I)  = .TRUE.
    QUP(IFL)  = .FALSE.

    IF (LSCAN3) THEN
       CALL DSUM2( IDUMX, IDUMY, IDUMZ, &
            XBAS(IFL)%a,YBAS(IFL)%a,ZBAS(IFL)%a, &
            -ONE, XBAS(I)%a,YBAS(I)%a,ZBAS(I)%a, NATOM)
       LENSEG(I) = SQRT( &
            DSDOT1(IDUMX,IDUMY,IDUMZ,NATOM) )
    ENDIF

    IF (INTERP >= 2) THEN
       SEGSTP(I) = MIN( STEPSZ, LENSEG(I)/INTERP )
    ELSE
       SEGSTP(I) = STEPSZ
    ENDIF
    IF (SEGSTP(I) < STEPLW) THEN
       SEGSTP(I) = STEPLW
       IF (VERBEU >= 2 .OR. LSCAN1) &
            CALL WI('          SGSCAN> SEGSTP(I) was limited by'// &
            ' STEPLW on segment I =', I )
    ENDIF

    NTERPL = MAX( NINT( LENSEG(I)/SEGSTP(I) ), 1 )
    STPSIZ = LENSEG(I)/NTERPL

    IF (.NOT.LSCAN3 .AND. NTERPL > 1) THEN
       CALL DSUM2( IDUMX, IDUMY, IDUMZ, &
            XBAS(IFL)%a,YBAS(IFL)%a,ZBAS(IFL)%a, &
            -ONE, XBAS(I)%a,YBAS(I)%a,ZBAS(I)%a, NATOM)
    ENDIF

    IF (VERBEU >= 1 .AND. LSCAN1) THEN
       WRITE(OUTU,1040) LLL, I, MAX(NTERPL-1,0), PTENE(I)
1040   FORMAT(/,' Segment',I5,4X,'Index',I5,4X, &
            'Interpolations=',I6,4X,'E=',F12.4,/)
    ENDIF

    STMPMX = -1
    SSECMX = -1
    ETMPMX = -RBIG
    ESECMX = -RBIG
    SMIN1  = -1
    SMIN2  = -1
    PRVENE = PTENE(I)
    QPRVUP = QUP(I)

    loop40:DO J=1, NTERPL
       IF (J < NTERPL) THEN
          JRN=J
          JRN=JRN/NTERPL
          CALL DSUM2( X,Y,Z, &
               XBAS(I)%a,YBAS(I)%a,ZBAS(I)%a,JRN, &
               IDUMX,IDUMY,IDUMZ, NATOM)
          DDUM1 = STPSIZ
          IF (J == 1)  DDUM1 = ZERO
          E = ENERG( X,Y,Z, DX,DY,DZ, .FALSE., .FALSE., DDUM1 )
       ELSE
          IF (LSCAN2) THEN
             CALL COP3D( &
                  XBAS(IFL)%a,YBAS(IFL)%a,ZBAS(IFL)%a, &
                  X,Y,Z, NATOM )
             DDUM1 = STPSIZ
             IF (J == 1)  DDUM1 = ZERO
             E = ENERG( X,Y,Z, DX,DY,DZ, .TRUE., .FALSE., DDUM1)
             CALL FIXED(IDIM)
             IDIM  = 3*(NATOM - IDIM)
             DDIM  = IDIM
             PTGRAD(IFL) = SQRT( DSDOT2(DX,DY,DZ,NATOM)/DDIM )
          ELSE
             E = PTENE(IFL)
          ENDIF
       ENDIF

       IF (E  <=  -RBIG  ) THEN
          CALL WRNDIE(-5,'<SGSCAN>','E is smaller than -RBIG  ')
       ENDIF

       IF (VERBEU >= 2 .AND. J < NTERPL .AND. LSCAN1) THEN
          WRITE(OUTU,1041) J, E
1041      FORMAT(27X,'Interpolation number',I5,4X,'E=',F12.4)
       ENDIF

       IF ((.NOT.QPRVUP) .AND. (PRVENE  <  E)) THEN
          IF (SMIN1  ==  -1) THEN
             SMIN1 = J-1
          ELSE
             SMIN2 = J-1
          ENDIF
       ENDIF

       IF (QPRVUP .AND. (PRVENE  >  E)) THEN
          IF (PRVENE  >  ETMPMX) THEN
             SSECMX = STMPMX
             ESECMX = ETMPMX
             STMPMX = J-1
             ETMPMX = PRVENE
          ELSEIF (PRVENE  >  ESECMX) THEN
             SSECMX = J-1
             ESECMX = PRVENE
          ENDIF
          QPRVUP = .FALSE.
       ELSEIF (PRVENE  <  E) THEN
          IF (J  ==  1)  QDOWN(I) = .FALSE.
          QPRVUP = .TRUE.
       ENDIF

       PRVENE = E
    enddo loop40

    QUP(IFL) = QPRVUP
    PTENE(IFL) = E
    IF        (SMIN1  ==  -1) THEN
       NEXMIN(I)   = 0
       PREMIN(IFL) = 0
    ELSEIF (SMIN2  ==  -1) THEN
       NEXMIN(I)   = SMIN1
       PREMIN(IFL) = SMIN1
    ELSE
       NEXMIN(I)   = SMIN1
       PREMIN(IFL) = SMIN2
    ENDIF

    IF (STMPMX  >  -1) THEN
       NMAXI         = NMAXI + 1
       SRTENE(NMAXI) = ETMPMX
       SRTIDX(NMAXI) = I
       STPMAX(I)     = STMPMX
       ENEMAX(I)     = ETMPMX
       IF ((STMPMX == 0).AND.(SSECMX > 0)) THEN
          STPMAX(I) = -SSECMX
          ENEMAX(I) =  ESECMX
       ENDIF
    ENDIF

    call chmdealloc('travel.src','SGSCAN','IDUMX',NATOM,crl=IDUMX)
    call chmdealloc('travel.src','SGSCAN','IDUMY',NATOM,crl=IDUMY)
    call chmdealloc('travel.src','SGSCAN','IDUMZ',NATOM,crl=IDUMZ)

    RETURN
  END subroutine sgscan

  ! getidx moved from adiab.src for circular dependency

  INTEGER FUNCTION GETIDX( QERROR, NFREEP,IFREEP, NPOINT,NMAXP, &
       XBAS,YBAS,ZBAS, NATOM,TOTUSD,SERIAL,NLINMN )
    use chm_kinds
    use chm_types
    use exfunc
    use memory
    implicit none

    LOGICAL   QERROR
    INTEGER   IFREEP(*),NFREEP, NPOINT,NMAXP, NATOM,TOTUSD
    INTEGER   SERIAL(*),NLINMN(*)
    type(chm_array) XBAS(:),YBAS(:),ZBAS(:)

    QERROR = .FALSE.

    IF (NFREEP  >  0) THEN
       GETIDX = IFREEP(NFREEP)
       NFREEP = NFREEP - 1
    ELSEIF (NPOINT  ==  NMAXP) THEN
       CALL W0(' ')
       CALL W0(' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
       CALL W0('   Aborting after reaching the maximum allowed')
       CALL W0('     number of path-points NPOINT = NMAXP')
       CALL W0(' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
       QERROR = .TRUE.
       RETURN
    ELSE
       GETIDX = NPOINT + 1
       call chmalloc('adiab.src','GETIDX','XBAS(GETIDX)',NATOM,crl=XBAS(GETIDX)%a)
       call chmalloc('adiab.src','GETIDX','YBAS(GETIDX)',NATOM,crl=YBAS(GETIDX)%a)
       call chmalloc('adiab.src','GETIDX','ZBAS(GETIDX)',NATOM,crl=ZBAS(GETIDX)%a)
    ENDIF
    NPOINT = NPOINT + 1
    TOTUSD = TOTUSD + 1
    SERIAL(GETIDX) = TOTUSD
    NLINMN(GETIDX) = 0

    RETURN
  END FUNCTION GETIDX

#else /*  (travel_main)*/
  SUBROUTINE NULL_TRAVEL
    return
  end SUBROUTINE NULL_TRAVEL
#endif /* (travel_main)*/

#if KEY_HFB==1 /*hfb*/
  !ivk
  subroutine cntslct(SLCT,NATOM,NSELR)
    !ivk Mon Jun 26 12:34:40 PDT 2006
    use chm_kinds
    use stream
    implicit none
    ! passed variables
    INTEGER SLCT(*), NATOM, NSELR
    ! local variables
    INTEGER I
    ! NSELR - the number of selected atoms
    NSELR = 0
    DO I=1,NATOM
       IF (SLCT(I) == 1) NSELR = NSELR + 1
    ENDDO
    return
  end subroutine cntslct

  subroutine pntslct(SLCT,AUXSLCT,NATOM,NSELR)
    ! IVK
    use chm_kinds
    use number
    implicit none
    ! passed variables
    INTEGER SLCT(*), AUXSLCT(*), NATOM, NSELR
    ! local variables
    INTEGER I, PNTR
    PNTR = 0
    DO I=1,NATOM
       IF(SLCT(I) == 1) THEN
          PNTR = PNTR + 1
          AUXSLCT(PNTR) = I
       ENDIF
    ENDDO

    IF(PNTR /= NSELR) THEN
       CALL WRNDIE(0,'<PNTSLCT>','SELECTION ERROR')
       ! This should never happen
    ENDIF

    return
  end subroutine pntslct
  !-----------------------------------------------------------------------
  SUBROUTINE initfrcp(XBAS,YBAS,ZBAS, &
       FPSLCT,NSELR,NATOM,NPOINT,SMASS)
    ! Initialize the PATH. This implies 
    ! 1) translation of all the replicas to the center of mass of 
    !    the given selection
    ! 2) finding the rotational matrices J(i) to best fit 
    !    the specified selections to the selection of the very first end
    !    point, this includes the very last replica (end point) as well
    ! 3) Applying the rotations to the complete vector of atomic coordinates 
    !    despite that the center of mass of the selection and the whole molecule 
    !    are different

    ! Used in the beginning of the path minimization and also in restart of
    ! conjugate gradient procedure
    ! WVCT[X,Y,Z] are the components of the raw working vectors (coordinates)
    use chm_kinds
    use chm_types
    use number
    use stream
    implicit none
    ! Passed variables
    INTEGER FPSLCT(*),NSELR,NATOM,NPOINT
    type(chm_array) XBAS(:),YBAS(:),ZBAS(:)
    ! Note that in this version SMASS is only NSELR long
    real(chm_real) SMASS(*)
    ! Local variables
    ! MROT - the rotation matrix
    real(chm_real) MROT(3,3)
    INTEGER REPA,REPB,GOFFSTA,GOFFSTB,PNTA,PNTB
    INTEGER ATOMPR(2,NSELR)
    INTEGER I,J
    ! Variables for the best fit
    LOGICAL LPRINTP
    real(chm_real) TMASS,RMST,RA(3,NSELR),RB(3,NSELR),DEVA(3,3),EVWID

    evwid = zero ! local & unset -- intent unclear

    LPRINTP = .FALSE.

    ! Rotation best fit (need to get the rotation matrix)
    ! The first replica serves as the reference point
    ! Setting up default
    !ivk      DO I=1,NSELR
    !ivk      WRITE(OUTU,11) FPSLCT(I),SMASS(I)
    !ivk  11  FORMAT('FPSLCT(I),SMASS(I)',I8,F16.3)
    !ivk      ENDDO

    DO I=1,NSELR
       ATOMPR(1,I)=FPSLCT(I)
       ATOMPR(2,I)=FPSLCT(I)
    ENDDO

    ! Translate REPA into the center of mass of the selection
    ! This is wierd, because HH is delcared as integer, but when
    ! it is passed to the function it becomes real(chm_real). 
    ! I wonder if this is legal. This is like accessing the array
    ! by pointers/reference.

    CALL RTRNSLSCM(1,XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, &
         SMASS, &
         FPSLCT,NSELR,NATOM,NPOINT)

    ! Loop over the remaining replicas
    DO J=2,NPOINT
       ! Do not forget the center of mass translation to the origin first
       ! Let's write a subroutine for that, since it is going to be special
       CALL RTRNSLSCM(J,XBAS(J)%a,YBAS(J)%a,ZBAS(J)%a, &
            SMASS, &
            FPSLCT,NSELR,NATOM,NPOINT)

       ! Can be replaced by standard BSTFT function, using DEVA instead of U
       CALL LAGBSTFT(XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, &
            XBAS(J)%a,YBAS(J)%a,ZBAS(J)%a, &
            ATOMPR,NSELR, &
            .FALSE.,.TRUE.,LPRINTP,SMASS,.FALSE.,SMASS, &
            TMASS,RMST,RA,RB,DEVA,EVWID,MROT)

       CALL RROTSCM(J,MROT,XBAS(J)%a,YBAS(J)%a,ZBAS(J)%a, &
            NATOM,NPOINT)
    ENDDO
    ! End loop over the remaining replicas
    RETURN
  END SUBROUTINE initfrcp
  !-----------------------------------------------------------------------
  SUBROUTINE RTRNSLSCM(N,VCTX,VCTY,VCTZ, &
       SMASS, &
       FPSLCT,NSELR,NATOM,NPOINT)
    ! IVK
    ! Mnemonics: Replica Translate into the center of mass of selection
    ! SVCT: super vector (all replicas)
    use chm_kinds
    use number
    use stream
    implicit none
    ! Passed variables
    INTEGER N,NSELR,NATOM,NPOINT
    INTEGER FPSLCT(*)
    real(chm_real) VCTX(*),VCTY(*),VCTZ(*)
    ! Note that in this version SMASS is only NSELR long
    real(chm_real) SMASS(*)
    ! Local variables
    INTEGER I,J,PNT
    real(chm_real) CMX,CMY,CMZ,TMASS

    IF(N > NPOINT) THEN
       CALL WI('Npoint Equals to',NPOINT)
       CALL WRNDIE(0,'<RTRNSLSCM>', &
            'Replica Index Exceeds Total Replicas')
    ENDIF

    ! Compute the center of mass(weight) of the selection first
    CMX = ZERO
    CMY = ZERO
    CMZ = ZERO
    TMASS = ZERO

    DO I=1,NSELR
       PNT = FPSLCT(I)
       CMX = CMX + VCTX(PNT)*SMASS(I)
       CMY = CMY + VCTY(PNT)*SMASS(I)
       CMZ = CMZ + VCTZ(PNT)*SMASS(I)
       TMASS = TMASS + SMASS(I)
    ENDDO
    CMX = CMX/TMASS
    CMY = CMY/TMASS
    CMZ = CMZ/TMASS

    ! Now that the center is found translate the whole replica 
    DO I=1,NATOM
       PNT = I
       !ivk       write(*,*)'TESTING RTRNSLSCM',N,I,PNT,VCTX(PNT) 
       VCTX(PNT)=VCTX(PNT)-CMX
       VCTY(PNT)=VCTY(PNT)-CMY
       VCTZ(PNT)=VCTZ(PNT)-CMZ
    ENDDO
    RETURN
  END SUBROUTINE RTRNSLSCM
  !-----------------------------------------------------------------------
  SUBROUTINE RROTSCM(N,U,VCTX,VCTY,VCTZ,NATOM,NPOINT)
    ! Mnemonics: Replica Rotate about the center of mass of the selection
    ! VCT: vector (all atoms)
    use chm_kinds
    use number
    implicit none
    ! Passed variables
    INTEGER N,NATOM,NPOINT
    real(chm_real) VCTX(*),VCTY(*),VCTZ(*)
    real(chm_real) U(3,*)
    ! Local variables
    INTEGER I,J,GOFFST,PNT
    real(chm_real) RSX,RSY,RSZ

    IF(N > NPOINT) THEN
       CALL WRNDIE(0,'<RROTSCM>','Replica Index Exceeds Total Replicas')
    ENDIF

    ! Apply rotation matrix U to the whole replica
    DO I=1,NATOM
       PNT = I
       RSX=U(1,1)*VCTX(PNT)+U(1,2)*VCTY(PNT)+U(1,3)*VCTZ(PNT)
       RSY=U(2,1)*VCTX(PNT)+U(2,2)*VCTY(PNT)+U(2,3)*VCTZ(PNT)
       RSZ=U(3,1)*VCTX(PNT)+U(3,2)*VCTY(PNT)+U(3,3)*VCTZ(PNT)
       VCTX(PNT)=RSX
       VCTY(PNT)=RSY
       VCTZ(PNT)=RSZ
    ENDDO
    RETURN
  END SUBROUTINE RROTSCM
  !-----------------------------------------------------------------------
  ! Have to modify the bestfit routine to be able to use the 
  ! rotation matrix, very little modification though

  SUBROUTINE LAGBSTFT(XA,YA,ZA,XB,YB,ZB,ATOMPR,NPAIR, &
       LNOROT,LNOTRN,LPRINTP,KCNSTR,QKCOPY,BMASS, &
       TMASS,RMST,RA,RB,DEVA,EVWID,U)
    !-----------------------------------------------------------------------
    ! Harmonic restraints with best fit.  Bestfit calculation routine.
    ! The matrix U will bestfit B to A: min(A-U*B)**2
    !
    !  XA,YA,ZA ->  Coordinates for atom set A
    !  XB,YB,ZB ->  Coordinates for atom set B (may be the same as A)
    !  ATOMPR(2,NPAIR) ->  The pointers into set A and B
    !  NPAIR    ->  The number of atoms pairs in the best fit analysis
    !  LNOROT   ->  flag to indicate no rotation in the best fit.
    !  LNOTRN   ->  flag to indicate no translation in the best fit.
    !  LPRINT   ->  Print flag (detailed results)
    !  KCNSTR   ->  Weighting array (set A atom based)
    !  QKCOPY   ->  Copy KCNSTR data to BMASS?
    !  BMASS    <-> Final weighting array! Civk: Note only NPAIR long!
    !  TMASS    <-  Total weight
    !  RA       <-  VA - <<VA>>
    !  RB       <-  VB - <<VB>>
    !  ROTM     <- the 3x3 rotation matrix
    !
    use chm_kinds
    use number
    use stream
    use corsubs, only: frotu
    implicit none
    !
    INTEGER NPAIR
    real(chm_real)  XA(*),YA(*),ZA(*),XB(*),YB(*),ZB(*)
    INTEGER ATOMPR(2,NPAIR)
    LOGICAL LNOROT,LNOTRN,LPRINTP,QKCOPY
    real(chm_real)  KCNSTR(*),TMASS,RMST
    real(chm_real)  U(3,3)
    !
    ! Temporary space arrays
    real(chm_real) BMASS(NPAIR)
    real(chm_real) RA(3,NPAIR),RB(3,NPAIR)
    real(chm_real) DEVA(3,3)
    !
    ! Local variables and arrays
    real(chm_real) EVA(3),ATOT,BTOT,EVWID
    real(chm_real) R(3,3)
    real(chm_real) CMA(3),CMB(3),CMC(3)
    real(chm_real) RMSV,ATMP,BTMP
    INTEGER K,KA,KB,I,J
    LOGICAL QEVW,LPRINT
    !
    LPRINT=LPRINTP
    IF(PRNLEV <= 2)LPRINT=.FALSE.
    ! IVK
    LPRINT=.FALSE.
    !
    TMASS=0.0
    DO K=1,NPAIR
       KA=ATOMPR(1,K)
       KB=ATOMPR(2,K)
       RA(1,K)=XA(KA)
       RA(2,K)=YA(KA)
       RA(3,K)=ZA(KA)
       RB(1,K)=XB(KB)
       RB(2,K)=YB(KB)
       RB(3,K)=ZB(KB)
       !IVK         IF(QKCOPY) BMASS(K)=KCNSTR(KA)
       !ivk Tue Jun 27 08:20:38 PDT 2006 noticed a problem here:
       ! BMASS is assumed to correspond to a selection already
       ! and not the whole coordinate set vector!
       TMASS=TMASS+BMASS(K)
    ENDDO
    !
    IF(TMASS < RSMALL) RETURN ! don't process nonpositive total weight.
    !
    IF(LPRINT) write(OUTU,56) ' The BMASS value:',BMASS
56  format(A/,(10X,3F12.5))

    ! Compute centers of mass (or weight)
    DO I=1,3
       CMA(I)=0.0
       CMB(I)=0.0
       CMC(I)=0.0
    ENDDO
    !
    ! Find the center of mass for both molecules
    IF(.NOT.LNOTRN) THEN
       DO K=1,NPAIR
          DO I=1,3
             CMA(I)=CMA(I)+RA(I,K)*BMASS(K)
             CMB(I)=CMB(I)+RB(I,K)*BMASS(K)
          ENDDO
       ENDDO

       DO I=1,3
          CMA(I)=CMA(I)/TMASS
          CMB(I)=CMB(I)/TMASS
          CMC(I)=CMA(I)-CMB(I)
       ENDDO
    ENDIF
    ! The centers of mass are found

    ! Translate the coordinates so that cm is the origin
    DO K=1,NPAIR
       DO I=1,3
          RA(I,K)=RA(I,K)-CMA(I)
          RB(I,K)=RB(I,K)-CMB(I)
       ENDDO
    ENDDO
    !
    IF(LPRINT) write(OUTU,56) ' The RA value:',RA
    IF(LPRINT) write(OUTU,56) ' The RB value:',RB
    !
    IF (LPRINT) THEN
       WRITE(OUTU,44) CMB
       WRITE(OUTU,45) CMA
       WRITE(OUTU,46) CMC
    ENDIF
44  FORMAT(' CENTER OF ATOMS BEFORE TRANSLATION',3F12.5)
45  FORMAT(' CENTER OF REFERENCE COORDINATE SET',3F12.5)
46  FORMAT(' NET TRANSLATION OF ROTATED ATOMS  ',3F12.5)
    !
    IF (LNOROT) THEN
       !
       ! Compute the mass weighted euclidean square of distance
       RMST=0.0
       DO K=1,NPAIR
          DO I=1,3
             RMST=RMST+(RA(I,K)-RB(I,K))**2*BMASS(K)
          ENDDO
       ENDDO
       !
    ELSE
       !
       !       compute rotation matrix from lagrangian
       !
       ATOT=0.0
       BTOT=0.0
       DO I=1,3
          DO J=1,3
             R(I,J)=0.0
          ENDDO
       ENDDO
       DO K=1,NPAIR
          DO I=1,3
             ATOT=ATOT + RA(I,K)*RA(I,K)*BMASS(K)
             BTOT=BTOT + RB(I,K)*RB(I,K)*BMASS(K)
             DO J=1,3
                R(I,J)=R(I,J) + RA(I,K)*RB(J,K)*BMASS(K)
             ENDDO
          ENDDO
       ENDDO
       !
       CALL FROTU(R,EVA,DEVA,U,EVWID,QEVW,LPRINT)
       !
       IF(LPRINT) write(6,56) ' The ATOT,BTOT and EV(*) values:', &
            ATOT,BTOT,EVA(1),EVA(2),EVA(3)
       !
       IF(QEVW) THEN
          ! Note: Do not use conventional (higher accuracy) method in case
          ! the width parameter (EVWID) is employed.     
          RMST=ATOT+BTOT-2.0*(EVA(1)+EVA(2)+EVA(3))
       ELSE
          RMST=0.0
          DO K=1,NPAIR
             ATMP=0.0
             DO I=1,3
                BTMP=RA(I,K)
                DO J=1,3
                   BTMP=BTMP-U(I,J)*RB(J,K)
                ENDDO
                ATMP=ATMP+BTMP**2
             ENDDO
             RMST=RMST+ATMP*BMASS(K)
          ENDDO
       ENDIF
       !
    ENDIF
    !
    ! Ok so here we stand and what is it we find? Clearly the coordinates of RB
    ! have not been rotated to best fit the coordinates of RA. Only RMST was 
    ! computed, but the coordiates left unchanged after all. 
    ! This is a bit unexpected, provided that the rotational matrix is
    ! no longer available after exiting this routine. Well I noticed that
    ! they use DEVA matrix which is available, so it may be still OK.

    IF(RMST  <  RSMALL) RMST=ZERO
    RMSV=SQRT(RMST/TMASS)
    IF(LPRINT) WRITE(OUTU,14) RMST,TMASS,RMSV
14  FORMAT(' TOTAL SQUARE DIFF IS',F12.4,'  DENOMINATOR IS',F12.4,/ &
         '       THUS RMS DIFF IS',F12.6)
    !
    RETURN
  END SUBROUTINE LAGBSTFT
  !-----------------------------------------------------------------------
  SUBROUTINE SETFPATH(XBAS,YBAS,ZBAS, &
       FPSLCT,NSELR,NATOM,NPOINT,NMAXP,ITRUNC, &
       LDENS, &
       XCP,YCP,ZCP, &
       XKEEP,YKEEP,ZKEEP, &
       XDUM,YDUM,ZDUM, &
       NQGRID)

    ! This routine sets the Fourier path and performs 
    ! Fourier transformation to obtain the corresponding coefficients for
    ! all coordinates of all selected atoms.

    use chm_kinds
    use chm_types
    use number
    use stream
    use consta
    use memory
    use exfunc
    implicit none
    ! Passed variables
    LOGICAL LDENS
    type(chm_array) XBAS(:),YBAS(:),ZBAS(:)
    INTEGER FPSLCT(*),NSELR,NATOM,NPOINT,NMAXP,ITRUNC
    INTEGER NQGRID
    real(chm_real) XCP(*),YCP(*),ZCP(*)
    real(chm_real) XKEEP(*),YKEEP(*),ZKEEP(*)
    real(chm_real) XDUM(*),YDUM(*),ZDUM(*)
    ! Local variables
    real(chm_real),allocatable,dimension(:) :: WAMPX
    real(chm_real),allocatable,dimension(:) :: WAMPY
    real(chm_real),allocatable,dimension(:) :: WAMPZ
    INTEGER TOFFSTK
    INTEGER REP1,REPN,REPJ,GOFFST1,GOFFSTN,GOFFSTJ
    INTEGER PNT1,PNTN,PNTJ,PNTK
    INTEGER I,J,K
    INTEGER NORDR,NGRID
    real(chm_real) alpha(NMAXP)
    real(chm_real) dalpha,PIalphaJ,fpsin
    real(chm_real) gridplngth(NQGRID),gridalpha(NQGRID)

    !ivk  Use [X,Y,Z]CP for a copy of current coordinates
    !ivk  Use [X,Y,Z]KEEP for the first replica
    !ivk  Use [X,Y,Z]DUM for the DIFFerence between the reactant and the product

    NGRID = NQGRID
    ! Initially let us try this option

    !ivk      NORDR = NPOINT
    IF (ITRUNC  >  0) THEN
       NORDR = ITRUNC
    ELSE
       NORDR = (NPOINT/2)/4*3
    ENDIF

    ! Now find the values of alpha that correspond to the equal
    ! length sectors for the given number of total replicas, NPOINT.

    ! I am not sure what is the best way to handle this transformation
    ! should I use the whole coordinate space and leave the rest as is
    ! or should I transform both spaces the reaction and the orthogonal
    ! space as well. I guess it would not hurt to do both, and in the
    ! end it might become useful, whereas if I just do the reactive space
    ! in the end it might turn out that I needed both spaces.
    ! So let's do both spaces, but compute the line integral only for the
    ! reactive space of cause, and so solve the equal segment equations
    ! for the reactive space only. Do the final output using the whole
    ! space however!

    ! Assuming the endpoints are already in the center of mass and best
    ! fit for the reactive subspace.


    call chmalloc('travel2.src','SETFPATH','WAMPX',NORDR*NATOM,crl=WAMPX)
    call chmalloc('travel2.src','SETFPATH','WAMPY',NORDR*NATOM,crl=WAMPY)
    call chmalloc('travel2.src','SETFPATH','WAMPZ',NORDR*NATOM,crl=WAMPZ)

    ! Compute the difference DR = R(1)-R(0); I call this DIFX(Y,Z)

    CALL COP3D(XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, &
         XKEEP,YKEEP,ZKEEP,NATOM)
    CALL COP3D(XBAS(NPOINT)%a, &
         YBAS(NPOINT)%a, &
         ZBAS(NPOINT)%a, &
         XDUM,YDUM,ZDUM,NATOM)

    DO I=1,NATOM
       XDUM(I)=XDUM(I)-XKEEP(I)
       YDUM(I)=YDUM(I)-YKEEP(I)
       ZDUM(I)=ZDUM(I)-ZKEEP(I)
    ENDDO

    dalpha = ONE/(REAL(NPOINT)-ONE)
    DO J=1,NPOINT
       CALL COP3D(XBAS(J)%a,YBAS(J)%a,ZBAS(J)%a, &
            XCP,YCP,ZCP,NATOM)
       alpha(J) = (REAL(J)-ONE)*dalpha
       DO I=1,NATOM
          XCP(I)=XCP(I)-(XKEEP(I)+XDUM(I)*alpha(J))
          YCP(I)=YCP(I)-(YKEEP(I)+YDUM(I)*alpha(J))
          ZCP(I)=ZCP(I)-(ZKEEP(I)+ZDUM(I)*alpha(J))
       ENDDO
       CALL COP3D(XCP,YCP,ZCP, &
            XBAS(J)%a,YBAS(J)%a,ZBAS(J)%a, &
            NATOM)
    ENDDO

    ! Now we are ready for Fourier transformation into the frequiency
    ! space using sin's only :). Let's use a simple trapezoid rule 
    ! to compute the Fourier coefficients.
    !CC
    !CC NEED TO DEFINE THE WAMP[X,Y,Z]
    !CC
    call pathFT(XBAS,YBAS,ZBAS, &
         WAMPX,WAMPY,WAMPZ, &
         XCP,YCP,ZCP, &
         alpha,NPOINT,NATOM,NORDR)

    IF (LDENS) THEN
       NPOINT = NMAXP
       dalpha = ONE/(REAL(NPOINT)-ONE)
       DO J=1,NPOINT
          alpha(J) = (REAL(J)-ONE)*dalpha
       ENDDO
    ENDIF

    call FPLineIntegral(XDUM,YDUM,ZDUM, &
         WAMPX,WAMPY,WAMPZ, &
         FPSLCT,NSELR,NPOINT,NATOM,NORDR, &
         NGRID,gridplngth,gridalpha)
    ! Now find the values of alpha that correspond to the equal
    ! length sectors for the given number of total replicas, NPOINT.

    call FPReparametrize(XDUM,YDUM,ZDUM, &
         WAMPX,WAMPY,WAMPZ, &
         FPSLCT,NSELR,NPOINT,NATOM,NORDR, &
         NGRID,gridplngth,gridalpha,alpha)

    ! This soubroutine provides us with the new set of alpha.
    ! At this point we should be ready to build new coordinates
    ! that correspond to equally spaced segments in reactive space. 
    ! Note that we messed up working coordinates WVCTX,(Y,Z),
    ! but kept the comparison set filled with the same coordinates
    ! untouched.

    call RebuildPathFT(XBAS,YBAS,ZBAS, &
         WAMPX,WAMPY,WAMPZ, &
         XCP,YCP,ZCP, &
         XKEEP,YKEEP,ZKEEP, &
         XDUM,YDUM,ZDUM, &
         alpha,NPOINT,NATOM,NORDR)

    ! This is it. This should give me a new reparametrized path in the
    ! Main coordinate set.
    call chmdealloc('travel2.src','SETFPATH','WAMPX',NORDR*NATOM,crl=WAMPX)
    call chmdealloc('travel2.src','SETFPATH','WAMPY',NORDR*NATOM,crl=WAMPY)
    call chmdealloc('travel2.src','SETFPATH','WAMPZ',NORDR*NATOM,crl=WAMPZ)

    RETURN
  END SUBROUTINE SETFPATH
  !-----------------------------------------------------------------------
  subroutine RebuildPathFT(XBAS,YBAS,ZBAS, &
       WAMPX,WAMPY,WAMPZ, &
       XCP,YCP,ZCP, &
       XKEEP,YKEEP,ZKEEP, &
       XDUM,YDUM,ZDUM, &
       alpha,NPOINT,NATOM,NORDR)

    use chm_kinds
    use chm_types
    use number
    use stream
    use consta
    implicit none
    type(chm_array) XBAS(:),YBAS(:),ZBAS(:)
    INTEGER NATOM,NPOINT,NORDR
    real(chm_real) WAMPX(*),WAMPY(*),WAMPZ(*)
    real(chm_real) XCP(*),YCP(*),ZCP(*)
    real(chm_real) XKEEP(*),YKEEP(*),ZKEEP(*)
    real(chm_real) XDUM(*),YDUM(*),ZDUM(*)
    real(chm_real) alpha(*)
    ! Local variables
    INTEGER I,J,K
    INTEGER TOFFSTK,PNTK
    real(chm_real) PIalphaJ,fpsin

    DO J=1,NPOINT
       PIalphaJ = PI*alpha(J)
       DO I=1,NATOM
          XCP(I)=XKEEP(I)+XDUM(I)*alpha(J)
          YCP(I)=YKEEP(I)+YDUM(I)*alpha(J)
          ZCP(I)=ZKEEP(I)+ZDUM(I)*alpha(J)
       ENDDO

       do K=1,NORDR !the number of terms in the Fourier series
          TOFFSTK=(K-1)*NATOM
          fpsin = sin(K*PIalphaJ)
          do I=1,NATOM
             PNTK = TOFFSTK + I
             XCP(I)=XCP(I)+fpsin*wampx(PNTK)
             YCP(I)=YCP(I)+fpsin*wampy(PNTK)
             ZCP(I)=ZCP(I)+fpsin*wampz(PNTK)
             !ivk          write(*,*)'Test Rebuild:',K,I,PNTK,wampx(PNTK)
          enddo
       enddo

       CALL COP3D(XCP,YCP,ZCP, &
            XBAS(J)%a,YBAS(J)%a,ZBAS(J)%a, NATOM)

    ENDDO

    RETURN
  END subroutine RebuildPathFT
  !-----------------------------------------------------------------------
  subroutine pathFT(XBAS,YBAS,ZBAS, &
       WAMPX,WAMPY,WAMPZ, &
       XCP,YCP,ZCP, &
       alpha,NPOINT,NATOM,NORDR)
    ! This routine does the Fourier transform of the nonlinear part 
    ! of the path, and then provides the corresponding amplitudes
    ! Need to decide how to store the amplitudes, since coordinate
    ! structure is different from amplitude in this implementation.

    use chm_kinds
    use chm_types
    use number
    use consta
    implicit none

    type(chm_array) XBAS(:),YBAS(:),ZBAS(:)
    real(chm_real) WAMPX(*),WAMPY(*),WAMPZ(*)
    real(chm_real) XCP(*),YCP(*),ZCP(*)
    real(chm_real) alpha(*)
    INTEGER NPOINT,NATOM,NORDR
    !ivk Locals
    real(chm_real) dalpha,fpsin,HALFfpsin,TWOdalpha,KPI
    integer i, j, k, l, PNTK
    integer TOFFSTK
    !ivk k - the number of elements in each vector
    ! TOFFSTK - term offset for order K
    ! Note that I don't have to carry all these crapy pointers
    ! that use computer time to be evaluated, more elegant way 
    ! would be to use a 2D array form, but for now let's just
    ! stick with this CHARMMing way of doing things.

    ! Make sure that the wamp(x,y,z) are empty = zero
    ! to avoid any unforseen problems

    !ivk      write(*,*)'Testing PathFT: NORDR',NORDR 
    do K=1,NORDR !the number of terms in the Fourier series
       TOFFSTK=(K-1)*NATOM
       do I=1,NATOM
          PNTK = TOFFSTK + I
          wampx(PNTK)=ZERO
          wampy(PNTK)=ZERO
          wampz(PNTK)=ZERO
       enddo
    enddo

    dalpha = ONE/(REAL(NPOINT)-ONE)

    ! Do the trapezoid quadrature

    do K=1,NORDR !the number of terms in the Fourier series
       TOFFSTK=(K-1)*NATOM
       KPI = K*PI
       do J=1,NPOINT !the number of replicas
          CALL COP3D(XBAS(J)%a,YBAS(J)%a,ZBAS(J)%a, &
               XCP,YCP,ZCP,NATOM)
          !ivk      write(*,*)'Testing PathFT: J,XCP(J)',J,XCP(J),XBAS(J)
          fpsin = sin(KPI*alpha(J))
          HALFfpsin = HALF*sin(KPI*alpha(J))
          !ivk      write(*,*)'Testing PathFT: alpha(J)',J,alpha(J),fpsin,HALFfpsin
          IF ((J  ==  1).or.(J  ==  NPOINT)) THEN
             do I=1,NATOM
                PNTK = TOFFSTK + I
                wampx(PNTK)=wampx(PNTK)+HALFfpsin*XCP(I)
                wampy(PNTK)=wampy(PNTK)+HALFfpsin*YCP(I)
                wampz(PNTK)=wampz(PNTK)+HALFfpsin*ZCP(I)
             enddo
          ELSE
             do I=1,NATOM
                PNTK = TOFFSTK + I
                wampx(PNTK)=wampx(PNTK)+fpsin*XCP(I)
                wampy(PNTK)=wampy(PNTK)+fpsin*YCP(I)
                wampz(PNTK)=wampz(PNTK)+fpsin*ZCP(I)
                !ivk      write(*,*)'Testing PathFT: XCP(I)',I,XCP(I)
                !ivk      write(*,*)'Testing PathFT: wampx(PNTK)',PNTK,wampx(PNTK)
             enddo
          ENDIF
       enddo
    enddo

    ! Normalize to get the correct coefficients, note that the period
    ! in alpha is one, so no need to do the division by total alpha. 

    TWOdalpha = TWO*dalpha
    do K=1,NORDR !the number of terms in the Fourier series
       TOFFSTK=(K-1)*NATOM
       do I=1,NATOM
          PNTK = TOFFSTK + I
          wampx(PNTK)=TWOdalpha*wampx(PNTK)
          wampy(PNTK)=TWOdalpha*wampy(PNTK)
          wampz(PNTK)=TWOdalpha*wampz(PNTK)
          !ivk      write(*,*)'Testing PathFT: wampx(PNTK)',K,I,PNTK,wampx(PNTK)
       enddo
    enddo

    ! At this point I should have all the amplitudes, and should be
    ! ready to do the line integral along the Fourier path. 
    ! The best way to do that is to choose a step in alpha so that
    ! one gets an integer number of intervals, (or the other way around)
    ! and then step by step generate all the points and compute all the 
    ! necessary variables to compute the line integral. The linear part of 
    ! the configurations removed for the FT should be added back during
    ! the integration. 
    ! Remember that I only need to integrate the line in the reactive space

    ! Warning: Upon exit from here I leave the working set of coordinates 
    ! messed up


    return
  end subroutine pathFT
  !-----------------------------------------------------------------------
  subroutine FPLineIntegral(DIFX,DIFY,DIFZ,WAMPX,WAMPY,WAMPZ, &
       FPSLCT,NSELR,NPOINT,NATOM,NORDR, &
       NGRID,plngth,alpha)
    use chm_kinds
    use number
    implicit none
    ! This function computes the length of the path in the reactive space
    ! as a function of alpha paramter. The length is computed via line 
    ! integral using extended trapezoid rule. This is the most convenient
    ! way for further bracketing and bisection root search necessary
    ! for the path reparametrization.

    real(chm_real) DIFX(*),DIFY(*),DIFZ(*),WAMPX(*),WAMPY(*),WAMPZ(*)
    real(chm_real) plngth(*),alpha(*)
    INTEGER FPSLCT(*),NSELR,NPOINT,NATOM,NORDR,NGRID

    ! Local variables
    INTEGER I,J,K
    real(chm_real) LINTGRNDJ,LINTGRL,dalpha
    ! Need selection stuff to compute the length in the reactive subspace

    LINTGRL = ZERO

    ! To save some flops let's multiply by dalpha in the end
    dalpha = ONE/(REAL(NGRID)-ONE)
    do J=1,NGRID !this should loop over the endpoints
       LINTGRNDJ = ZERO
       alpha(J) = (REAL(J)-ONE)*dalpha
       ! I can use the new function to compute the integrand here
       ! This might be more efficeint and fool-proof
       ! Call the FPLINTGRND function here!!!
       LINTGRNDJ=FPLINTGRND(DIFX,DIFY,DIFZ,WAMPX,WAMPY,WAMPZ, &
            FPSLCT,NSELR,NATOM,NORDR,alpha(J))
       IF (J  ==  1) THEN
          plngth(J) = ZERO
          LINTGRL = LINTGRL + HALF*LINTGRNDJ
       ELSEIF (J  ==  NGRID) THEN
          plngth(J) = (LINTGRL + HALF*LINTGRNDJ)*dalpha
          LINTGRL = LINTGRL + HALF*LINTGRNDJ
       ELSE
          plngth(J) = (LINTGRL + HALF*LINTGRNDJ)*dalpha
          LINTGRL = LINTGRL + LINTGRNDJ
       ENDIF
       !ivk      write(*,*)'Length',J,alpha(J),LINTGRNDJ,LINTGRL*dalpha,plngth(J)
       !ivk      write(*,*)'Length',J,alpha(J),LINTGRNDJ,LINTGRL*dalpha,plngth(J)
    enddo

    ! Now get the final value of the integral:

    LINTGRL = LINTGRL*dalpha

    ! Do consistency check to see if the final LINTGRL is the same as
    ! plngth(NGRID)

    return
  end subroutine FPLineIntegral
  !-----------------------------------------------------------------------
  subroutine FPReparametrize(DIFX,DIFY,DIFZ,WAMPX,WAMPY,WAMPZ, &
       FPSLCT,NSELR,NPOINT,NATOM,NORDR, &
       NGRID,plngth,alpha,solution)
    use chm_kinds
    use number
    use consta
    implicit none
    ! This function finds the parameter values that correspond to
    ! NPOINT-1 equally spaced (in the length) sectors in the reactive space.
    ! Use bracketing and bisection root search.

    real(chm_real) DIFX(*),DIFY(*),DIFZ(*),WAMPX(*),WAMPY(*),WAMPZ(*)
    real(chm_real) plngth(*),alpha(*),solution(*)
    !ivk upps, found another error Tue Jun 27 15:52:40 PDT 2006
    !ivk LMEPSLCT array (FPSLCT) was not passed as an array!
    INTEGER FPSLCT(*),NSELR,NPOINT,NATOM,NORDR,NGRID

    ! Local variables
    real(chm_real) fpcos,KPI,trgtplngth,dplngth,TPLNGTH
    real(chm_real) dalpha
    real(chm_real) FUNCK,FUNCKP,BRACKETKKP
    INTEGER I,J,K,REPJ,BRACKET(NPOINT)
    ! Need selection stuff to compute the length in the reactive space
    ! Get the total length of the pathway:
    TPLNGTH = plngth(NGRID)
    ! Using the fact the the length is monotonically increasing function
    ! of the paramter alpha bracket the regions corresponding to
    ! length(alpha) = TPLNGTH * (J-1)/(NPOINT-1)
    ! So our equation f(alpha) = length(I)-TPLNGTH * (J-1)/(NPOINT-1) = 0
    ! We already know the answers for the endpoints 1 and NPOINT
    dalpha = ONE/(REAL(NPOINT)-ONE)
    dplngth = TPLNGTH * dalpha
    do J=2,NPOINT-1
       trgtplngth = dplngth*(REAL(J)-ONE)
       do K=1,NGRID-1
          FUNCK = plngth(K)-trgtplngth  
          FUNCKP = plngth(K+1)-trgtplngth 
          BRACKETKKP = FUNCK*FUNCKP
          if (BRACKETKKP  <=  ZERO) then
             BRACKET(J)=K
             !ivk            write(*,*)'Testing FPReparametrize',J,K,FUNCK,FUNCKP
             ! Note: I can leave the loop once this is true
             ! Do I use break for that or what?
          endif
       enddo
    enddo

    ! Now I am ready to perform bisection root searches
    ! I need a function that would compute a small sector
    ! for the line integral, or may be just the value of its
    ! integrand. I could have used such a function before,
    ! consider rewriting the previous subrotine. OK rewrote it.
    do J=2,NPOINT-1
       REPJ = J
       trgtplngth = dplngth*(REAL(J)-ONE)
       !ivk         write(*,*)'I am about to call AlphBisect',J,solution(J)
       solution(J) = AlphBisect(DIFX,DIFY,DIFZ,WAMPX,WAMPY,WAMPZ, &
            FPSLCT,NSELR,NATOM,NORDR, &
            plngth,alpha,dalpha,trgtplngth, &
            BRACKET,REPJ)
       !ivk         write(*,*)'I just called AlphBisect',J,solution(J)
    enddo

    return
  end subroutine FPReparametrize
  !-----------------------------------------------------------------------
  FUNCTION FPLINTGRND(DIFX,DIFY,DIFZ,WAMPX,WAMPY,WAMPZ, &
       FPSLCT,NSELR,NATOM,NORDR,alpha)
    ! This function computes the length of the path in the reactive space
    ! as a function of alpha paramter. The length is computed via line 
    ! integral using extended trapezoid rule. This is the most convenient
    ! way for further bracketing and bisection root search necessary
    ! for the path reparametrization.
    use chm_kinds
    use number
    use consta
    implicit none

    real(chm_real) DIFX(*),DIFY(*),DIFZ(*),WAMPX(*),WAMPY(*),WAMPZ(*)
    real(chm_real) alpha,FPLINTGRND
    INTEGER FPSLCT(*),NSELR,NATOM,NORDR

    ! Local variables
    real(chm_real) wtmpx(NSELR),wtmpy(NSELR),wtmpz(NSELR)
    real(chm_real) fpcos,KPI,LINTGRND
    INTEGER I,J,K,PNTK,PNTJ,TOFFSTK
    ! Need selection stuff to compute the length in the reactive space

    ! To save some flops let's multiply by dalpha in the end
    LINTGRND = ZERO

    do I=1,NSELR
       wtmpx(I)=ZERO
       wtmpy(I)=ZERO
       wtmpz(I)=ZERO
    enddo

    do K=1,NORDR !the number of terms in the Fourier series
       TOFFSTK=(K-1)*NATOM
       KPI = K*PI
       fpcos = (KPI)*cos(KPI*alpha)
       do I=1,NSELR
          PNTK = TOFFSTK + FPSLCT(I)
          !ivk         write(*,*)'Testing function FPLINTGRND',PNTK,wampx(PNTK)
          wtmpx(I)=wtmpx(I)+fpcos*wampx(PNTK)
          wtmpy(I)=wtmpy(I)+fpcos*wampy(PNTK)
          wtmpz(I)=wtmpz(I)+fpcos*wampz(PNTK)
       enddo
    enddo
    ! Uppss found a very simple error, that did not show up anywhere
    ! well it did not show up because it basically added a straight line
    ! on top of whatever curve and since I was computing the length of
    ! the line via this function it was consistent throughout and
    ! clearly cancelled out.
    do I=1,NSELR
       PNTJ = FPSLCT(I)
       wtmpx(I)=wtmpx(I)+DIFX(PNTJ)
       wtmpy(I)=wtmpy(I)+DIFY(PNTJ)
       wtmpz(I)=wtmpz(I)+DIFZ(PNTJ)
       LINTGRND = LINTGRND + &
            wtmpx(I)*wtmpx(I) +  &
            wtmpy(I)*wtmpy(I) + &
            wtmpz(I)*wtmpz(I)
    enddo
    FPLINTGRND = SQRT(LINTGRND)
    !ivk         write(*,*)'Testing function FPLINTGRND',alpha,FPLINTGRND

    return
  end FUNCTION FPLINTGRND
  !-----------------------------------------------------------------------
  FUNCTION AlphBisect(DIFX,DIFY,DIFZ,WAMPX,WAMPY,WAMPZ, &
       FPSLCT,NSELR,NATOM,NORDR, &
       plngth,alpha,dalpha,trgtplngth, &
       BRACKET,REP)
    ! This function computes the length of the path in the reactive space
    ! as a function of alpha paramter. The length is computed via line
    ! integral using extended trapezoid rule. This is the most convenient
    ! way for further bracketing and bisection root search necessary
    ! for the path reparametrization.
    use chm_kinds
    use number
    use consta
    implicit none
    real(chm_real) AlphBisect
    real(chm_real) DIFX(*),DIFY(*),DIFZ(*),WAMPX(*),WAMPY(*),WAMPZ(*)
    real(chm_real) plngth(*),alpha(*)
    real(chm_real) dalpha,trgtplngth
    INTEGER FPSLCT(*),NSELR,NATOM,NORDR
    INTEGER BRACKET(*),REP

    ! Local variables
    ! vdalpha - vectorial dalpha
    real(chm_real) wtmpx(NSELR),wtmpy(NSELR),wtmpz(NSELR)
    real(chm_real) LINTGRND
    real(chm_real) AlphaLeft,AlphaRight,AlphaRoot,AlphaMid
    real(chm_real) LILeft,LIRight,LIMid,vdalpha
    real(chm_real) LIntgrndLeft,LIntgrndMid
    INTEGER I,J,K,PNTK,TOFFSTK
    INTEGER GridPointLeft,GridPointRight
    ! WARNING: Using a STATIC PARAMETER HERE
    ! Should make this available from the command line
    INTEGER, PARAMETER :: MAXIT = 40

    GridPointLeft = BRACKET(REP)
    GridPointRight = BRACKET(REP)+1

    AlphaLeft = alpha(GridPointLeft)
    AlphaRight = alpha(GridPointRight) 

    LILeft = plngth(GridPointLeft)-trgtplngth
    LIRight = plngth(GridPointRight)-trgtplngth

    LIntgrndLeft=FPLINTGRND(DIFX,DIFY,DIFZ,WAMPX,WAMPY,WAMPZ, &
         FPSLCT,NSELR,NATOM,NORDR,AlphaLeft)

    IF (LILeft*LIRight  <=  ZERO) THEN
       IF (LILeft  <=  ZERO) THEN
          AlphaRoot = AlphaLeft
          vdalpha = AlphaRight - AlphaLeft
       ELSE
          AlphaRoot = AlphaRight
          vdalpha = AlphaLeft - AlphaRight
       ENDIF

    ELSE
       CALL WRNDIE(0,'<BISECTSOLV>', &
            'The Root is not Bracketed')
    ENDIF

    ! Note that we use pretty primitive way of recomputing the 
    ! Line Integral here. We don't really care at this point 
    ! about terminating upon reaching some satisfactory threshold:

    DO I=1,MAXIT
       vdalpha=HALF*vdalpha
       AlphaMid = AlphaRoot + vdalpha
       LIntgrndMid=FPLINTGRND(DIFX,DIFY,DIFZ,WAMPX,WAMPY,WAMPZ, &
            FPSLCT,NSELR,NATOM,NORDR,AlphaMid)
       LIMid=LILeft &
            +HALF*(LIntgrndLeft+LIntgrndMid)*(AlphaMid-AlphaLeft)
       IF (LIMid  <=  ZERO) THEN
          AlphaRoot=AlphaMid
       ENDIF
    ENDDO

    AlphBisect = AlphaRoot
    !ivk        write(*,*)'Proverka:AlphaRoot',AlphaRoot

    return
  end FUNCTION AlphBisect
  !-----------------------------------------------------------------------
  SUBROUTINE InterpolateFP(XBAS,YBAS,ZBAS, &
       IDSTART,IDSTOP, &
       XCP,YCP,ZCP, &
       XKEEP,YKEEP,ZKEEP, &
       XDUM,YDUM,ZDUM, &
       FPSLCT,NSELR,NATOM,NPOINT,NMAXP,SMASS, &
       QERROR,NFREEP,IFREEP, &
       TOTUSD,SERIAL,NLINMN, &
       IDPREC,IDNEXT)
    use chm_kinds
    use chm_types
    use number
    use stream
    implicit none
    ! Passed variables
    type(chm_array) XBAS(:),YBAS(:),ZBAS(:)
    INTEGER IDSTART,IDSTOP
    INTEGER FPSLCT(*),NSELR,NATOM,NPOINT,NMAXP
    real(chm_real) XCP(*),YCP(*),ZCP(*)
    real(chm_real) XKEEP(*),YKEEP(*),ZKEEP(*)
    real(chm_real) XDUM(*),YDUM(*),ZDUM(*)
    real(chm_real) SMASS(*)
    LOGICAL QERROR
    INTEGER NFREEP,IFREEP(*)
    INTEGER TOTUSD,SERIAL(*),NLINMN(*)
    INTEGER IDPREC(*),IDNEXT(*)
    ! Local variables
    INTEGER ATOMPR(2,NSELR)
    INTEGER I,J
    INTEGER NINTRP,MOVETO,COPYTO,OLDNP,IDXNEW
    ! Variables for the best fit
    real(chm_real) TMASS,RMST,RA(3,NSELR),RB(3,NSELR),DEVA(3,3),EVWID
    real(chm_real) MROT(3,3)
    real(chm_real) alpha,dalpha
    LOGICAL LPRINTP

    LPRINTP = .FALSE.

    !ivk Let us do the translation and orientation for the replicas that
    !ivk we have read in. It could be more than just reactant and product.
    !ivk In fact initfrcp does that. So we have to be called from some other
    !ivk place, right after the inifrcp. The rest assumes that we have been
    !ivk moved into the center of mass and reoriented to bestfit the first
    !ivk structure.

    CALL COP3D(XBAS(IDSTART)%a, &
         YBAS(IDSTART)%a, &
         ZBAS(IDSTART)%a, &
         XKEEP,YKEEP,ZKEEP,NATOM)

    CALL COP3D(XBAS(IDSTOP)%a, &
         YBAS(IDSTOP)%a, &
         ZBAS(IDSTOP)%a, &
         XDUM,YDUM,ZDUM,NATOM)

    DO I=1,NATOM
       XDUM(I)=XDUM(I)-XKEEP(I)
       YDUM(I)=YDUM(I)-YKEEP(I)
       ZDUM(I)=ZDUM(I)-ZKEEP(I)
    ENDDO

    !ivk Now let's say we want to have final NPOINT = NMAXP
    !ivk so we need NMAXP - NPOINT new points

    ! Let's move the last strip to the end up to the maxpoint

    NINTRP = NMAXP - NPOINT

    ! Need to get the ids for up to NMAXP
    CALL W0('Increasing the numb. of points by interpolating')
    OLDNP = NPOINT
    DO J=1,NINTRP
       IDXNEW = GETIDX(QERROR, NFREEP,IFREEP,NPOINT,NMAXP, &
            XBAS,YBAS,ZBAS,NATOM,TOTUSD,SERIAL,NLINMN)

    ENDDO
    CALL WI('The new total number of path-points NPOINT =', NPOINT)

    IF (OLDNP < NMAXP) THEN
       MOVETO = NMAXP
       DO J=OLDNP,IDSTOP,-1
          CALL COP3D(XBAS(J)%a,YBAS(J)%a,ZBAS(J)%a, &
               XBAS(MOVETO)%a, &
               YBAS(MOVETO)%a, &
               ZBAS(MOVETO)%a, &
               NATOM)
          MOVETO = MOVETO - 1
       ENDDO
    ENDIF

    ! Let's bridge the gap with a line interpolation

    dalpha = ONE/(REAL(NINTRP+2)-ONE)

    COPYTO = IDSTART
    DO J=1,NINTRP+2
       alpha = (REAL(J)-ONE)*dalpha

       DO I=1,NATOM
          XCP(I)=XKEEP(I)+XDUM(I)*alpha
          YCP(I)=YKEEP(I)+YDUM(I)*alpha
          ZCP(I)=ZKEEP(I)+ZDUM(I)*alpha

       ENDDO
       CALL COP3D(XCP,YCP,ZCP, &
            XBAS(COPYTO)%a, &
            YBAS(COPYTO)%a, &
            ZBAS(COPYTO)%a, &
            NATOM)
       COPYTO = COPYTO + 1
    ENDDO

    !ivk Need to set the idnext to write coordinates out
    DO I=1,NPOINT
       IDPREC(I) = I - 1
       IDNEXT(I) = I + 1
    ENDDO

    !ivk Have to clean up the ends so we can pass TREK tests
    IDPREC(1) = 1
    IDNEXT(NPOINT) = NPOINT


    ! Not sure if I need to reorient the interpolated points
    ! If you want you can do this afterwards using the inifrcp again

    RETURN
  END SUBROUTINE InterpolateFP
  !-----------------------------------------------------------------------
  SUBROUTINE ActivateFPMax(XBAS,YBAS,ZBAS, &
       XCP,YCP,ZCP, &
       XKEEP,YKEEP,ZKEEP, &
       XDUM,YDUM,ZDUM, &
       FPSLCT,NSELR,NATOM,NPOINT,NMAXP,SMASS, &
       QERROR,NFREEP,IFREEP, &
       TOTUSD,SERIAL,NLINMN, &
       IDPREC,IDNEXT)
    use chm_kinds
    use chm_types
    use number
    use stream
    implicit none
    ! Passed variables
    type(chm_array) XBAS(:),YBAS(:),ZBAS(:)
    INTEGER FPSLCT(*),NSELR,NATOM,NPOINT,NMAXP
    real(chm_real) XCP(*),YCP(*),ZCP(*)
    real(chm_real) XKEEP(*),YKEEP(*),ZKEEP(*)
    real(chm_real) XDUM(*),YDUM(*),ZDUM(*)
    real(chm_real) SMASS(*)
    LOGICAL QERROR
    INTEGER NFREEP,IFREEP(*)
    INTEGER TOTUSD,SERIAL(*),NLINMN(*)
    INTEGER IDPREC(*),IDNEXT(*)
    ! Local variables
    INTEGER ATOMPR(2,NSELR)
    INTEGER I,J
    INTEGER NINTRP,MOVETO,COPYTO,OLDNP,IDXNEW

    !ivk Now let's say we want to have final NPOINT = NMAXP
    !ivk so we need NMAXP - NPOINT new points

    ! Let's move the last strip to the end up to the maxpoint

    NINTRP = NMAXP - NPOINT

    ! Need to get the ids for up to NMAXP
    CALL W0('Increasing the numb. of points by request')
    OLDNP = NPOINT
    DO J=1,NINTRP
       IDXNEW = GETIDX(QERROR, NFREEP,IFREEP,NPOINT,NMAXP, &
            XBAS,YBAS,ZBAS,NATOM,TOTUSD,SERIAL,NLINMN)

    ENDDO
    CALL WI('The new total number of path-points NPOINT =', NPOINT)

    !ivk Need to set the idnext to write coordinates out
    !ivk Not sure why this is only up to the NPOINT?
    !ivk that is because the NPOINT become NMAXP
    DO I=1,NPOINT
       IDPREC(I) = I - 1
       IDNEXT(I) = I + 1
    ENDDO

    IDPREC(1) = 1
    IDNEXT(NPOINT) = NPOINT
    !ivk This is a hack to preserve the NPOINT number
    NPOINT = OLDNP

    RETURN
  END SUBROUTINE ActivateFPMax
  !-----------------------------------------------------------------------
  SUBROUTINE PRCSSFPATH(XBAS,YBAS,ZBAS, &
       FPSLCT,NSELR,NATOM,NPOINT,NMAXP,ITRUNC, &
       LDENS,LFEND,LPRJCT,LPSTRUCT, &
       XCP,YCP,ZCP, &
       XKEEP,YKEEP,ZKEEP, &
       XDUM,YDUM,ZDUM, &
       SMASS,IRTYPE,RRFRC,RSTEP, &
       IPROFU,NBEADS,NQGRID)

    ! This routine sets the Fourier path and performs 
    ! Fourier transformation to obtain the corresponding coefficients for
    ! all coordinates of all selected atoms.

    ! Once again rememeber that NPOINT is now doubled

    use chm_kinds
    use chm_types
    use number
    use stream
    use consta
    use memory
    use exfunc
    implicit none
    ! Passed variables
    LOGICAL LDENS,LFEND,LPRJCT,LPSTRUCT
    type(chm_array) XBAS(:),YBAS(:),ZBAS(:)
    INTEGER FPSLCT(*),NSELR,NATOM,NPOINT,NMAXP,ITRUNC
    real(chm_real) XCP(*),YCP(*),ZCP(*)
    real(chm_real) XKEEP(*),YKEEP(*),ZKEEP(*)
    real(chm_real) XDUM(*),YDUM(*),ZDUM(*)
    real(chm_real) SMASS(*)
    real(chm_real) RRFRC,RSTEP
    INTEGER IRTYPE,IPROFU,NBEADS,NQGRID

    ! Local variables
    real(chm_real),allocatable,dimension(:) :: WAMPX
    real(chm_real),allocatable,dimension(:) :: WAMPY
    real(chm_real),allocatable,dimension(:) :: WAMPZ
    INTEGER TOFFSTK,CRDOFFST,AMPOFFST
    INTEGER REP1,REPN,REPJ,GOFFST1,GOFFSTN,GOFFSTJ
    INTEGER PNT1,PNTN,PNTI,PNTJ,PNTK
    INTEGER I,J,K
    INTEGER NORDR,NGRID,NPPNTS
    ! NPPNTS - the number of points in the path
    !ivk Effectively now NMAXP is double of what it should be anyway

    real(chm_real) alpha(NMAXP)
    real(chm_real) dalpha,PIalphaJ,fpsin,DBLRFRC,DBLFMASS
    real(chm_real) gridpwork(NQGRID),gridalpha(NQGRID)



    !ivk  Use [X,Y,Z]CP for a copy of current coordinates
    !ivk  Use [X,Y,Z]KEEP for the first replica
    !ivk  Use [X,Y,Z]DUM for the DIFFerence between the reactant and the product

    NORDR = ITRUNC

    NPPNTS = NBEADS

    NGRID = NQGRID

    ! I am not sure what is the best way to handle this transformation?
    ! Should I use only the RCS and leave the SCS as is, or should I transform 
    ! both spaces at together? I guess it would not hurt to do both unless 
    ! I run into some memory limitation.
    ! Let's do both spaces, but compute the line integral only for the
    ! RCS of cause, and so solve the equal segment equations
    ! for the RCS only. Do the final output using the complete space.

    ! We do not perform any transformations on the coordinates untill we are
    ! ready to generate the next reference path.

    ! I need double the number of storage for the Fourier amplitudes

    call chmalloc('travel2.src','PRCSSFPATH','WAMPX',NORDR*NATOM*2,crl=WAMPX)
    call chmalloc('travel2.src','PRCSSFPATH','WAMPY',NORDR*NATOM*2,crl=WAMPY)
    call chmalloc('travel2.src','PRCSSFPATH','WAMPZ',NORDR*NATOM*2,crl=WAMPZ)

    ! This takes care of the evolved beads (the first set of the coordinates)
    ! and prepares them for the Fourier transform.
    ! Compute the difference DR = R(1)-R(0); I call this DIFX(Y,Z)

    !ivk If we want to freeze the endpoints let's overwrite the evolved endpoints with
    !ivk the reference beads.
    if (LFEND) then

       CRDOFFST = NPPNTS

       PNT1 = CRDOFFST + 1
       PNTN = CRDOFFST + NPPNTS

       CALL COP3D(XBAS(PNT1)%a, &
            YBAS(PNT1)%a, &
            ZBAS(PNT1)%a, &
            XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a,NATOM)
       CALL COP3D(XBAS(PNTN)%a, &
            YBAS(PNTN)%a, &
            ZBAS(PNTN)%a, &
            XBAS(NPPNTS)%a, &
            YBAS(NPPNTS)%a, &
            ZBAS(NPPNTS)%a, &
            NATOM)

    endif

    CALL COP3D(XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, &
         XKEEP,YKEEP,ZKEEP,NATOM)

    CALL COP3D(XBAS(NPPNTS)%a, &
         YBAS(NPPNTS)%a, &
         ZBAS(NPPNTS)%a, &
         XDUM,YDUM,ZDUM,NATOM)

    CALL DSUM2(XDUM,YDUM,ZDUM, &
         XDUM,YDUM,ZDUM, &
         -ONE, &
         XKEEP,YKEEP,ZKEEP, &
         NATOM)

    dalpha = ONE/(REAL(NPPNTS)-ONE)
    DO J=1,NPPNTS
       CALL COP3D(XBAS(J)%a,YBAS(J)%a,ZBAS(J)%a, &
            XCP,YCP,ZCP,NATOM)
       alpha(J) = (REAL(J)-ONE)*dalpha
       DO I=1,NATOM
          XCP(I)=XCP(I)-(XKEEP(I)+XDUM(I)*alpha(J))
          YCP(I)=YCP(I)-(YKEEP(I)+YDUM(I)*alpha(J))
          ZCP(I)=ZCP(I)-(ZKEEP(I)+ZDUM(I)*alpha(J))
       ENDDO
       CALL COP3D(XCP,YCP,ZCP, &
            XBAS(J)%a,YBAS(J)%a,ZBAS(J)%a, &
            NATOM)
    ENDDO

    ! Now we are ready to do the Fourier transformation into the frequency
    ! domain. Let's use a simple trapezoid rule to compute the Fourier coefficients.

    !ivk This should work for both sets now that 
    !ivk I introduced offsets for both coordinates and the amplitudes

    CRDOFFST = 0
    AMPOFFST = 0

    call pathFToffst(XBAS,YBAS,ZBAS, &
         WAMPX,WAMPY,WAMPZ, &
         XCP,YCP,ZCP, &
         alpha,NPPNTS,NATOM,NORDR, &
         CRDOFFST,AMPOFFST)

    !ivk Now let us compute the mean forces acting on the beads
    !ivk To do that I need the coordinates of the evolved beads
    !ivk and then the coordinates of the reference beads
    !ivk How do I get the evolved beads back? 
    !ivk I have to recompute them, because I have
    !ivk subtracted the linear part to do the Fourier transform earlier.

    CRDOFFST = NPPNTS
    AMPOFFST = NATOM * NORDR
    DBLRFRC = TWO * RRFRC

    DO J=1,NPPNTS
       PNTJ = CRDOFFST + J
       CALL COP3D(XBAS(J)%a,YBAS(J)%a,ZBAS(J)%a, &
            XCP,YCP,ZCP,NATOM)
       DO I=1,NATOM
          XCP(I)=XCP(I)+(XKEEP(I)+XDUM(I)*alpha(J))
          YCP(I)=YCP(I)+(YKEEP(I)+YDUM(I)*alpha(J))
          ZCP(I)=ZCP(I)+(ZKEEP(I)+ZDUM(I)*alpha(J))
       ENDDO

       !ivk This restores the evolved beads coordinates

       CALL COP3D(XCP,YCP,ZCP, &
            XBAS(J)%a,YBAS(J)%a,ZBAS(J)%a, &
            NATOM)

       !ivk This shifts the reference beads into the evolved bead centered coordinate system
       !ivk That should be OK since we only use this difference for the forces

       CALL DSUM2(XBAS(PNTJ)%a,YBAS(PNTJ)%a,ZBAS(PNTJ)%a, &
            XBAS(PNTJ)%a,YBAS(PNTJ)%a,ZBAS(PNTJ)%a, &
            -ONE,XCP,YCP,ZCP,NATOM)

       !ivk This way we get the restored evolved beads in the first set 
       !ivk and the restraint forces in the second set - quite convenient! 

       !ivk Now depending on the IRTYPE I need to accordingly weight the difference
       !ivk to get at the forces, note: this should be done on selected atoms only

       IF (IRTYPE  ==  1) THEN

          !ivk the difference between the evolved beads and the reference beads should
          !ivk be multiplied by TWO * RRFRC * MASS(i)
          CALL COP3D(XBAS(PNTJ)%a, &
               YBAS(PNTJ)%a, &
               ZBAS(PNTJ)%a, &
               XCP,YCP,ZCP,NATOM)

          DO I=1,NSELR
             PNTI = FPSLCT(I)
             DBLFMASS = DBLRFRC*SMASS(I)
             XCP(PNTI)=XCP(PNTI)*DBLFMASS
             YCP(PNTI)=YCP(PNTI)*DBLFMASS
             ZCP(PNTI)=ZCP(PNTI)*DBLFMASS
          ENDDO

          CALL COP3D(XCP,YCP,ZCP, &
               XBAS(PNTJ)%a, &
               YBAS(PNTJ)%a, &
               ZBAS(PNTJ)%a, &
               NATOM)
       ENDIF

    ENDDO

    !ivk Now let's subtract the linear part from the forces (note however that
    !ivk the forces should not have any liner part when the endpoints are in
    !ivk their respective minima)
    !ivk I will have to get the forces with the linear part in them back 
    !ivk to be able to step into the next reference structure.

    PNT1 = CRDOFFST + 1
    PNTN = CRDOFFST + NPPNTS

    CALL COP3D(XBAS(PNT1)%a,YBAS(PNT1)%a,ZBAS(PNT1)%A, &
         XKEEP,YKEEP,ZKEEP,NATOM)
    CALL COP3D(XBAS(PNTN)%a,YBAS(PNTN)%a,ZBAS(PNTN)%a, &
         XDUM,YDUM,ZDUM,NATOM)

    CALL DSUM2(XDUM,YDUM,ZDUM, &
         XDUM,YDUM,ZDUM, &
         -ONE, &
         XKEEP,YKEEP,ZKEEP, &
         NATOM)

    DO J=1,NPPNTS
       PNTJ = CRDOFFST + J
       CALL COP3D(XBAS(PNTJ)%a,YBAS(PNTJ)%a,ZBAS(PNTJ)%a, &
            XCP,YCP,ZCP,NATOM)
       DO I=1,NATOM
          XCP(I)=XCP(I)-(XKEEP(I)+XDUM(I)*alpha(J))
          YCP(I)=YCP(I)-(YKEEP(I)+YDUM(I)*alpha(J))
          ZCP(I)=ZCP(I)-(ZKEEP(I)+ZDUM(I)*alpha(J))
       ENDDO
       CALL COP3D(XCP,YCP,ZCP, &
            XBAS(PNTJ)%a,YBAS(PNTJ)%a,ZBAS(PNTJ)%a, &
            NATOM)
    ENDDO

    !ivk This should give me the set of amplitudes for the forces

    call pathFToffst(XBAS,YBAS,ZBAS, &
         WAMPX,WAMPY,WAMPZ, &
         XCP,YCP,ZCP, &
         alpha,NPPNTS,NATOM,NORDR, &
         CRDOFFST,AMPOFFST)

    !ivk Now that I have the Fourier amplitudes for both the coordinates
    !ivk and the forces, and all the vectors required for the work integral setup,
    !ivk I am free to use the coordinate array sets.
    !ivk So let's just restore the forces back to original state and
    !ivk then make the step along the forces to make the new reference bead.
    !ivk This is the best place for this that I could find. 
    !ivk So let's just restore the forces back to original state first.

    DO J=1,NPPNTS
       PNTJ = CRDOFFST + J
       CALL COP3D(XBAS(PNTJ)%a,YBAS(PNTJ)%a,ZBAS(PNTJ)%a, &
            XCP,YCP,ZCP,NATOM)
       DO I=1,NATOM
          XCP(I)=XCP(I)+(XKEEP(I)+XDUM(I)*alpha(J))
          YCP(I)=YCP(I)+(YKEEP(I)+YDUM(I)*alpha(J))
          ZCP(I)=ZCP(I)+(ZKEEP(I)+ZDUM(I)*alpha(J))
       ENDDO

       !       IF (IRTYPE  ==  1) THEN
       !
       !ivk the difference between the evolved beads and the reference beads should
       !ivk be multiplied by TWO * RRFRC * MASS(i)
       !
       !          DO I=1,NSELR
       !             PNTI = FPSLCT(I)
       !             DBLFMASS = DBLRFRC*SMASS(I)
       !             XCP(PNTI)=XCP(PNTI)*DBLFMASS
       !             YCP(PNTI)=YCP(PNTI)*DBLFMASS
       !             ZCP(PNTI)=ZCP(PNTI)*DBLFMASS
       !          ENDDO
       !          
       !       ELSE
       !          RETURN                !No other restraint types implemented yet
       !       ENDIF

       !ivk At this point we have the force at a point J along the path
       !ivk stored at the XCP,YCP,ZCP. What we need is to project that
       !ivk force orthogonal to the path and then apply it with a certain
       !ivk step size to the corresponding evolved bead. This has to be postponed 
       !ivk until the work is computed.

       CALL COP3D(XCP,YCP,ZCP, &
            XBAS(PNTJ)%a,YBAS(PNTJ)%a,ZBAS(PNTJ)%a, &
            NATOM)

    ENDDO

    !ivk This restores the evolved beads coordinates

    !ivk At this point I am almost ready to compute the Work via 
    !ivk the generalized line integral of the second order 
    !ivk I need the force at the first bead, plus the difference
    !ivk of the forces between the end point beads
    !ivk I would also need the differences between the endpoint coordinates
    !ivk So let's decide how and get it here.
    !ivk here: XKEEP,YKEEP,ZKEEP
    !ivk the difference of the forces here: XDUM,YDUM,ZDUM
    !ivk I already have all those in order
    !ivk and let's put the difference between the endpoint 
    !ivk coordinates into XCP,YCP,ZCP

    CALL DSUM2(XCP,YCP,ZCP, &
         XBAS(NPPNTS)%a, &
         YBAS(NPPNTS)%a, &
         ZBAS(NPPNTS)%a, &
         -ONE, &
         XBAS(1)%a,YBAS(1)%A,ZBAS(1)%a, &
         NATOM)

    call WrkLineIntegral(XKEEP,YKEEP,ZKEEP, &
         XDUM,YDUM,ZDUM, &
         XCP,YCP,ZCP, &
         WAMPX,WAMPY,WAMPZ, &
         FPSLCT,NSELR,NPOINT,NATOM,NORDR, &
         NGRID,gridpwork,gridalpha)


    !ivk We can write out the work integral here
    IF (IPROFU  >  0) THEN
       IF (IPROFU  ==  6) THEN
          WRITE(IPROFU,77)
       ENDIF
       DO I=1,NGRID
          WRITE(IPROFU,'(2f12.5)') gridalpha(I),gridpwork(I)
       ENDDO
       IF (IPROFU  ==  6) THEN
          WRITE(IPROFU,78)
       ENDIF
    ENDIF
77  FORMAT('=============== Energy Profile Starts ===============')
78  FORMAT('=============== Energy Profile Ends =================')

    !ivk If someone wants to get the structures corresponding to the 
    !ivk energy profile (which is a reasonable thing to want) especially
    !ivk if one is trying to get the coordinates of all the intermediates
    !ivk and transition states based on the energy profile we should provide
    !ivk this option.
    !ivk Let's just say if this is requested we print these structures and 
    !ivk in the swollen trajectory and then quit since none of the following
    !ivk stuff will make any sense anyway.

    IF (LPSTRUCT) THEN
       NPOINT = NGRID
       !ivk  We still have [X,Y,Z]CP that holds the diff between react and product coordinates
       !ivk  but we do not have the reactant coordinates in the [X,Y,Z]KEEP 
       !ivk  (but rather the reactant force), so let's just copy the reactant coord there instead.
       !ivk  [X,Y,Z]DUM will be used as temporary space arrays.

       CALL COP3D(XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, &
            XKEEP,YKEEP,ZKEEP,NATOM)

       call RebuildPathFT(XBAS,YBAS,ZBAS, &
            WAMPX,WAMPY,WAMPZ, &
            XDUM,YDUM,ZDUM, &
            XKEEP,YKEEP,ZKEEP, &
            XCP,YCP,ZCP, &
            gridalpha,NPOINT,NATOM,NORDR)

       !ivk  Note: I swapped []DUM with []CP to get the right functionality.
       !ivk  If all goes well we should get the right amplitudes and get the
       !ivk  correct structures.

       !ivk  We do not wish to peform any operations beyond this poit
       RETURN
    ENDIF

    !ivk Assuming that the XCP,YCP,ZCP still hold the difference 
    !ivk between the endpoint coordinates.

    DO J=1,NPPNTS
       PNTJ = CRDOFFST + J
       if (LPRJCT) then

          CALL PathProjector(XBAS(PNTJ)%a, &
               YBAS(PNTJ)%a, &
               ZBAS(PNTJ)%a, &
               XCP,YCP,ZCP, &
               WAMPX,WAMPY,WAMPZ, &
               FPSLCT,NSELR,NATOM,NORDR,alpha(J))

       endif
       CALL DSUM2(XBAS(J)%a,YBAS(J)%a,ZBAS(J)%a, &
            XBAS(J)%a,YBAS(J)%a,ZBAS(J)%a, &
            -RSTEP, &
            XBAS(PNTJ)%a,YBAS(PNTJ)%a,ZBAS(PNTJ)%a, &
            NATOM)
    ENDDO

    !ivk The next step from here is to reorient and the redistribute the
    !ivk new reference beads.


    !ivk      IF (LDENS) THEN
    !ivk         NPOINT = NMAXP
    !ivk         dalpha = ONE/(REAL(NPOINT)-ONE)
    !ivk         DO J=1,NPOINT
    !ivk            alpha(J) = (REAL(J)-ONE)*dalpha
    !ivk         ENDDO
    !ivk      ENDIF

    ! Now find the values of alpha that correspond to the equal
    ! length sectors for the given number of total replicas, NPOINT.

    !ivk       write(*,*)'I am about to call FPReparametrize'
    !ivk      call FPReparametrize(XDUM,YDUM,ZDUM,
    !ivk     $                     HEAP_(WAMPX),HEAP_(WAMPY),HEAP_(WAMPZ),
    !ivk     $                     FPSLCT,NSELR,NPOINT,NATOM,NORDR,
    !ivk     $                     NGRID,gridplngth,gridalpha,alpha)
    !ivk       write(*,*)'I just called FPReparametrize'

    ! This soubroutine provides us with the new set of alpha.
    ! At this point we should be ready to build new coordinates
    ! that correspond to equally spaced segments in reactive space. 
    ! Note that we messed up working coordinates WVCTX,(Y,Z),
    ! but kept the comparison set filled with the same coordinates
    ! untouched.

    !ivk       write(*,*)'I am about to call RebuildPathFT'
    !ivk      call RebuildPathFT(HPCRD,XBAS,YBAS,ZBAS,
    !ivk     $                   HEAP_(WAMPX),HEAP_(WAMPY),HEAP_(WAMPZ),
    !ivk     $                   XCP,YCP,ZCP,
    !ivk     $                   XKEEP,YKEEP,ZKEEP,
    !ivk     $                   XDUM,YDUM,ZDUM,
    !ivk     $                   alpha,NPOINT,NATOM,NORDR)
    !ivk       write(*,*)'I just called RebuildPathFT'

    ! This is it. This should give me a new reparametrized path in the
    ! Main coordinate set.
    ! I do not like calling the actual HEAP_ in the same function that gets HEAP_
    ! passed under a mask of HPCRD
    !ivk      FREE HEAP_!!!!
    call chmdealloc('travel2.src','PRCSSFPATH','WAMPX',NORDR*NATOM*2,crl=WAMPX)
    call chmdealloc('travel2.src','PRCSSFPATH','WAMPY',NORDR*NATOM*2,crl=WAMPY)
    call chmdealloc('travel2.src','PRCSSFPATH','WAMPZ',NORDR*NATOM*2,crl=WAMPZ)

    RETURN
  END SUBROUTINE PRCSSFPATH
  !-----------------------------------------------------------------------
  subroutine pathFToffst(XBAS,YBAS,ZBAS, &
       WAMPX,WAMPY,WAMPZ, &
       XCP,YCP,ZCP, &
       alpha,NPOINT,NATOM,NORDR, &
       CRDOFFST,AMPOFFST)
    ! This routine does the Fourier transform of the nonlinear part 
    ! of the path, and then provides the corresponding amplitudes
    ! Need to decide how to store the amplitudes, since coordinate
    ! structure is different from amplitude in this implementation.

    use chm_kinds
    use chm_types
    use number
    use consta
    use stream
    implicit none

    type(chm_array) XBAS(:),YBAS(:),ZBAS(:)
    real(chm_real) WAMPX(*),WAMPY(*),WAMPZ(*)
    real(chm_real) XCP(*),YCP(*),ZCP(*)
    real(chm_real) alpha(*)
    INTEGER NPOINT,NATOM,NORDR
    INTEGER CRDOFFST,AMPOFFST
    !ivk Locals
    real(chm_real) dalpha,fpsin,HALFfpsin,TWOdalpha,KPI
    integer i, j, k, l, PNTK,PNTJ
    integer TOFFSTK
    !ivk k - the number of elements in each vector
    ! TOFFSTK - term offset for order K
    ! Note that I don't have to carry all these crapy pointers
    ! that use computer time to be evaluated, more elegant way 
    ! would be to use a 2D array form, but for now let's just
    ! stick with this CHARMMing way of doing things.

    if(prnlev >=7) write(outu,'(a,i8)') &
         'CRDOFFST1',CRDOFFST
    IF ((CRDOFFST  /=  0) .and. (CRDOFFST  /=  NPOINT)) THEN 
       if(prnlev >=7) write(outu,'(a,i8)') &
            'CRDOFFST',CRDOFFST
       CALL WRNDIE(0,'<pathFToffst>', &
            'Not allowed CRDOFFST value')
    ENDIF

    IF ((AMPOFFST  /=  0) .and. (AMPOFFST  /=  NORDR * NATOM)) THEN 
       CALL WRNDIE(0,'<pathFToffst>', &
            'Not allowed AMPOFFST value')
    ENDIF

    !ivk NEED A CHECK for the VALUES of the CRDOFFST and AMPOFFST
    !ivk      write(*,*)'Testing PathFT: NORDR',NORDR 

    ! Make sure that the wamp(x,y,z) are empty = zero
    ! to avoid any unforseen problems

    do K=1,NORDR !the number of terms in the Fourier series
       TOFFSTK=(K-1)*NATOM
       do I=1,NATOM
          PNTK = AMPOFFST + TOFFSTK + I
          wampx(PNTK)=ZERO
          wampy(PNTK)=ZERO
          wampz(PNTK)=ZERO
       enddo
    enddo

    dalpha = ONE/(REAL(NPOINT)-ONE)

    ! Do the trapezoid quadrature

    do K=1,NORDR !the number of terms in the Fourier series
       TOFFSTK=(K-1)*NATOM
       KPI = K*PI
       do J=1,NPOINT !the number of replicas
          PNTJ = CRDOFFST + J
          CALL COP3D(XBAS(PNTJ)%a, &
               YBAS(PNTJ)%a, &
               ZBAS(PNTJ)%a, &
               XCP,YCP,ZCP,NATOM)
          fpsin = sin(KPI*alpha(J))
          HALFfpsin = HALF*sin(KPI*alpha(J))
          IF ((J  ==  1).or.(J  ==  NPOINT)) THEN
             do I=1,NATOM
                PNTK = AMPOFFST + TOFFSTK + I
                wampx(PNTK)=wampx(PNTK)+HALFfpsin*XCP(I)
                wampy(PNTK)=wampy(PNTK)+HALFfpsin*YCP(I)
                wampz(PNTK)=wampz(PNTK)+HALFfpsin*ZCP(I)
             enddo
          ELSE
             do I=1,NATOM
                PNTK = AMPOFFST + TOFFSTK + I
                wampx(PNTK)=wampx(PNTK)+fpsin*XCP(I)
                wampy(PNTK)=wampy(PNTK)+fpsin*YCP(I)
                wampz(PNTK)=wampz(PNTK)+fpsin*ZCP(I)
             enddo
          ENDIF
       enddo
    enddo

    ! Normalize to get the correct coefficients, note that the period
    ! in alpha is one, so no need to do the division by total alpha. 

    TWOdalpha = TWO*dalpha
    do K=1,NORDR !the number of terms in the Fourier series
       TOFFSTK=(K-1)*NATOM
       do I=1,NATOM
          PNTK = AMPOFFST + TOFFSTK + I
          wampx(PNTK)=TWOdalpha*wampx(PNTK)
          wampy(PNTK)=TWOdalpha*wampy(PNTK)
          wampz(PNTK)=TWOdalpha*wampz(PNTK)
       enddo
    enddo

    ! At this point I should have all the amplitudes, and should be
    ! ready to do the line integral along the Fourier path. 
    ! The best way to do that is to choose a step in alpha so that
    ! one gets an integer number of intervals, (or the other way around)
    ! and then step by step generate all the points and compute all the 
    ! necessary variables to compute the line integral. The linear part of 
    ! the configurations removed for the FT should be added back during
    ! the integration. 
    ! Remember that I only need to integrate the line in the reactive space

    ! Warning: Upon exit from here I leave the working set of coordinates 
    ! messed up


    return
  end subroutine pathFToffst
  !-----------------------------------------------------------------------
  subroutine WrkLineIntegral(FRCRX,FRCRY,FRCRZ, &
       FDIFX,FDIFY,FDIFZ, &
       DIFX,DIFY,DIFZ, &
       WAMPX,WAMPY,WAMPZ, &
       FPSLCT,NSELR,NPOINT,NATOM,NORDR, &
       NGRID,pwork,alpha)
    use chm_kinds
    use number
    use stream
    implicit none
    ! This function computes the length of the path in the reactive space
    ! as a function of alpha paramter. The length is computed via line 
    ! integral using extended trapezoid rule. This is the most convenient
    ! way for further bracketing and bisection root search necessary
    ! for the path reparametrization.
    !ivk Memo: FRCRX - ForceReactantX
    !ivk Memo: FDIFX - DifferenceForce(Product-Reactant)X
    !ivk Memo: DIFX - DifferenceCoord(Product-Reactant)X

    real(chm_real) FRCRX(*),FRCRY(*),FRCRZ(*), &
         FDIFX(*),FDIFY(*),FDIFZ(*)
    real(chm_real) DIFX(*),DIFY(*),DIFZ(*),WAMPX(*),WAMPY(*),WAMPZ(*)
    real(chm_real) pwork(*),alpha(*)
    INTEGER FPSLCT(*),NSELR,NPOINT,NATOM,NORDR,NGRID

    ! Local variables
    INTEGER I,J,K
    real(chm_real) LINTGRNDJ,LINTGRL,dalpha
    ! Need selection stuff to compute the length in the reactive subspace

    LINTGRL = ZERO

    ! To save some flops let's multiply by dalpha in the end
    dalpha = ONE/(REAL(NGRID)-ONE)
    do J=1,NGRID !this should loop over the endpoints
       LINTGRNDJ = ZERO
       alpha(J) = (REAL(J)-ONE)*dalpha
       ! I can use the new function to compute the integrand here
       ! This might be more efficeint and fool-proof
       ! Call the FPLINTGRND function here!!!
       !ivk         LINTGRNDJ=-worklintgrnd(FRCRX,FRCRY,FRCRZ,
       LINTGRNDJ=worklintgrnd(FRCRX,FRCRY,FRCRZ, &
            FDIFX,FDIFY,FDIFZ, &
            DIFX,DIFY,DIFZ, &
            WAMPX,WAMPY,WAMPZ, &
            FPSLCT,NSELR,NATOM,NORDR,alpha(J))
       IF (J  ==  1) THEN
          pwork(J) = ZERO
          LINTGRL = LINTGRL + HALF*LINTGRNDJ
       ELSEIF (J  ==  NGRID) THEN
          pwork(J) = (LINTGRL + HALF*LINTGRNDJ)*dalpha
          LINTGRL = LINTGRL + HALF*LINTGRNDJ
       ELSE
          pwork(J) = (LINTGRL + HALF*LINTGRNDJ)*dalpha
          LINTGRL = LINTGRL + LINTGRNDJ
       ENDIF
    enddo

    ! Now get the final value of the integral:

    LINTGRL = LINTGRL*dalpha

    do J=1,NGRID !this should loop over the endpoints
       if(prnlev >= 7) write(outu,'(a,i5,f12.5)') &
            'J,Work ',J,pwork(J)
    enddo

    ! Do consistency check to see if the final LINTGRL is the same as
    ! plngth(NGRID)

    return
  end subroutine WrkLineIntegral
  !-----------------------------------------------------------------------
  FUNCTION worklintgrnd(FRCRX,FRCRY,FRCRZ, &
       FDIFX,FDIFY,FDIFZ, &
       DIFX,DIFY,DIFZ, &
       WAMPX,WAMPY,WAMPZ, &
       FPSLCT,NSELR,NATOM,NORDR,alpha)
    ! This function computes the length of the path in the reactive space
    ! as a function of alpha paramter. The length is computed via line 
    ! integral using extended trapezoid rule. This is the most convenient
    ! way for further bracketing and bisection root search necessary
    ! for the path reparametrization.
    use chm_kinds
    use number
    use consta
    implicit none

    real(chm_real) worklintgrnd
    real(chm_real) FRCRX(*),FRCRY(*),FRCRZ(*), &
         FDIFX(*),FDIFY(*),FDIFZ(*)
    real(chm_real) DIFX(*),DIFY(*),DIFZ(*),WAMPX(*),WAMPY(*),WAMPZ(*)
    real(chm_real) alpha
    INTEGER FPSLCT(*),NSELR,NATOM,NORDR

    ! Local variables
    real(chm_real) wtmpx(NSELR),wtmpy(NSELR),wtmpz(NSELR)
    real(chm_real) ftmpx(NSELR),ftmpy(NSELR),ftmpz(NSELR)
    real(chm_real) fpcos,fpsin,KPI,LINTGRND
    INTEGER I,J,K,PNTK,PNTJ,TOFFSTK,AMPOFFST
    ! Need selection stuff to compute the length in the reactive space

    ! To save some flops let's multiply by dalpha in the end
    LINTGRND = ZERO

    do I=1,NSELR
       wtmpx(I)=ZERO
       wtmpy(I)=ZERO
       wtmpz(I)=ZERO

       ftmpx(I)=ZERO
       ftmpy(I)=ZERO
       ftmpz(I)=ZERO
    enddo

    AMPOFFST = NATOM * NORDR
    do K=1,NORDR !the number of terms in the Fourier series
       TOFFSTK=(K-1)*NATOM
       KPI = K*PI
       fpcos = (KPI)*cos(KPI*alpha)
       fpsin = sin(KPI*alpha)
       do I=1,NSELR
          PNTK = TOFFSTK + FPSLCT(I)
          PNTJ = AMPOFFST + TOFFSTK + FPSLCT(I)

          wtmpx(I)=wtmpx(I)+fpcos*wampx(PNTK)
          wtmpy(I)=wtmpy(I)+fpcos*wampy(PNTK)
          wtmpz(I)=wtmpz(I)+fpcos*wampz(PNTK)

          ftmpx(I)=ftmpx(I)+fpsin*wampx(PNTJ)
          ftmpy(I)=ftmpy(I)+fpsin*wampy(PNTJ)
          ftmpz(I)=ftmpz(I)+fpsin*wampz(PNTJ)
       enddo
    enddo

    do I=1,NSELR
       PNTJ = FPSLCT(I)
       wtmpx(I)=wtmpx(I)+DIFX(PNTJ)
       wtmpy(I)=wtmpy(I)+DIFY(PNTJ)
       wtmpz(I)=wtmpz(I)+DIFZ(PNTJ)

       ftmpx(I)=ftmpx(I)+FRCRX(PNTJ)+FDIFX(PNTJ)*alpha
       ftmpy(I)=ftmpy(I)+FRCRY(PNTJ)+FDIFY(PNTJ)*alpha
       ftmpz(I)=ftmpz(I)+FRCRZ(PNTJ)+FDIFZ(PNTJ)*alpha
       LINTGRND = LINTGRND + &
            wtmpx(I)*ftmpx(I) +  &
            wtmpy(I)*ftmpy(I) + &
            wtmpz(I)*ftmpz(I)
    enddo
    worklintgrnd = LINTGRND
    !ivk         write(*,*)'Testing function worklintgrnd',
    !ivk $                 alpha,worklintgrnd

    return
  end FUNCTION worklintgrnd
  !-----------------------------------------------------------------------
  SUBROUTINE PathProjector(FRCX,FRCY,FRCZ, &
       DIFX,DIFY,DIFZ, &
       WAMPX,WAMPY,WAMPZ, &
       FPSLCT,NSELR,NATOM,NORDR,alpha)
    ! This subroutine computes the projection of the mean force on the
    ! path curve first and then computes its orthogonal component 
    ! in the reactive space as a function of alpha paramter. 
    use chm_kinds
    use number
    use consta
    implicit none

    real(chm_real) FRCX(*),FRCY(*),FRCZ(*)
    real(chm_real) DIFX(*),DIFY(*),DIFZ(*),WAMPX(*),WAMPY(*),WAMPZ(*)
    real(chm_real) alpha
    INTEGER FPSLCT(*),NSELR,NATOM,NORDR

    ! Local variables
    real(chm_real) wtmpx(NSELR),wtmpy(NSELR),wtmpz(NSELR)
    real(chm_real) fpcos,KPI,dotCRVCRV,dotCRVFRC,prjctr
    INTEGER I,K,PNTK,PNTI,TOFFSTK
    ! Need selection stuff to compute the projections in the reactive space

    do I=1,NSELR
       wtmpx(I)=ZERO
       wtmpy(I)=ZERO
       wtmpz(I)=ZERO
    enddo

    do K=1,NORDR !the number of terms in the Fourier series
       TOFFSTK=(K-1)*NATOM
       KPI = K*PI
       fpcos = (KPI)*cos(KPI*alpha)
       do I=1,NSELR
          PNTK = TOFFSTK + FPSLCT(I)
          wtmpx(I)=wtmpx(I)+fpcos*wampx(PNTK)
          wtmpy(I)=wtmpy(I)+fpcos*wampy(PNTK)
          wtmpz(I)=wtmpz(I)+fpcos*wampz(PNTK)
       enddo
    enddo

    do I=1,NSELR
       PNTI = FPSLCT(I)
       wtmpx(I)=wtmpx(I)+DIFX(PNTI)
       wtmpy(I)=wtmpy(I)+DIFY(PNTI)
       wtmpz(I)=wtmpz(I)+DIFZ(PNTI)

       dotCRVCRV = dotCRVCRV + &
            wtmpx(I)*wtmpx(I) + &
            wtmpy(I)*wtmpy(I) + &
            wtmpz(I)*wtmpz(I)

       dotCRVFRC = dotCRVFRC + &
            wtmpx(I)*FRCX(PNTI) + &
            wtmpy(I)*FRCY(PNTI) + &
            wtmpz(I)*FRCZ(PNTI)
    enddo

    !ivk Now we have the required dot products and are ready to assemble the
    !ivk orthogonal projection of the forces to the path curve(CRV)

    prjctr = dotCRVFRC/dotCRVCRV
    do I=1,NSELR
       PNTI = FPSLCT(I)
       FRCX(PNTI)=FRCX(PNTI)-wtmpx(I)*prjctr
       FRCY(PNTI)=FRCY(PNTI)-wtmpy(I)*prjctr
       FRCZ(PNTI)=FRCZ(PNTI)-wtmpz(I)*prjctr
    enddo
    return
  end SUBROUTINE PathProjector
  !-----------------------------------------------------------------------
  !ivk
#endif /* (hfb)*/



end module travelsub2

