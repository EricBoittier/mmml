 SUBROUTINE EABFORINTC(NAT,X,Y,Z,XCOMP,YCOMP,ZCOMP,AMASS,LMASS,LRMS, &
       ATOMIN,ISLCT,LWEIG,WMAIN,LNORO,LPRINT)
    !-----------------------------------------------------------------------
    !  THIS ROUTINE IS CALLED BY THE ORIENT COMMAND AND ROTATES THE
    !  MOLECULE SO THAT IT SITS ON THE ORIGIN WITH NO OFF-DIAGONAL
    !  MOMENTS. LMASS CAUSES A MASS WEIGHTING TO BE DONE.
    !  LRMS=.TRUE. WILL CAUSE THE ROTATION TO BE WRT THE OTHER SET.
    !
    !     By Bernard R. Brooks   1982

  use number
  use stream
  use param_store, only: set_param
  use corsubs

  implicit none

    INTEGER NAT
    real(chm_real) X(*),Y(*),Z(*),XCOMP(*),YCOMP(*),ZCOMP(*),WMAIN(*)
    real(chm_real) AMASS(*)
    LOGICAL LMASS,LRMS
    INTEGER ATOMIN(2,*)
    INTEGER ISLCT(*)
    LOGICAL LWEIG,LNORO,LPRINT

    real(chm_real) U(9),RN(3),PHI,EV(3)
    real(chm_real) XC,YC,ZC,XN,YN,ZN
    INTEGER NPR,N,I,J,NG
    LOGICAL LPRNT

    N=NAT
    LPRNT = (LPRINT .AND. PRNLEV >= 2)

    IF(LRMS) THEN
       NPR=0
       DO I=1,N
          IF(ISLCT(I) == 1) THEN
             NPR=NPR+1
             ATOMIN(1,NPR)=I
             ATOMIN(2,NPR)=I
          ENDIF
       ENDDO
       !
       CALL EABFROTLSQ(XCOMP,YCOMP,ZCOMP,N,X,Y,Z,N,ATOMIN,NPR,LMASS,AMASS,AMASS,LWEIG,WMAIN,LNORO,LPRNT)
       !
    ELSE
       IF(LPRNT) WRITE(OUTU,15)
15     FORMAT(/' ORIENT THE COORDINATES TO ALIGN WITH AXIS'/)
       !
       ! Process best translation
       !
       CALL LSQP2(NAT,X,Y,Z,AMASS,LMASS, &
            ISLCT,LWEIG,WMAIN,LNORO,LPRINT,XC,YC,ZC,U,EV)
       !
       IF(LPRNT) WRITE(OUTU,25) XC,YC,ZC
25     FORMAT(' CENTER OF ATOMS BEFORE TRANSLATION',3F12.5)
       !
       call set_param('XCEN',ZERO)
       call set_param('YCEN',ZERO)
       call set_param('ZCEN',ZERO)
       XC=-XC
       YC=-YC
       ZC=-ZC
       call set_param('XMOV',XC)
       call set_param('YMOV',YC)
       call set_param('ZMOV',ZC)
       DO I=1,N
          X(I)=X(I)+XC
          Y(I)=Y(I)+YC
          Z(I)=Z(I)+ZC
       ENDDO
       !
       IF(LNORO) RETURN
       !
       ! Process best rotation
       !
       CALL FNDROT(U,RN,PHI,LPRNT)
       call set_param('THET',PHI)
       QAXISC=.TRUE.
       AXISCX= ZERO
       AXISCY= ZERO
       AXISCZ= ZERO
       AXISR = ONE
       AXISX = RN(1)
       AXISY = RN(2)
       AXISZ = RN(3)
       call set_param('XAXI',AXISX)
       call set_param('YAXI',AXISY)
       call set_param('ZAXI',AXISZ)
       call set_param('RAXI',AXISR)
       call set_param('XCEN',AXISCX)
       call set_param('YCEN',AXISCY)
       call set_param('ZCEN',AXISCZ)
       !
       DO I=1,N
          XN=U(1)*X(I)+U(2)*Y(I)+U(3)*Z(I)
          YN=U(4)*X(I)+U(5)*Y(I)+U(6)*Z(I)
          ZN=U(7)*X(I)+U(8)*Y(I)+U(9)*Z(I)
          X(I)=XN
          Y(I)=YN
          Z(I)=ZN
       ENDDO
       !
    ENDIF
    !
    RETURN
  END SUBROUTINE EABFORINTC

  SUBROUTINE EABFROTLSQ(X1,Y1,Z1,NATOM1,X2,Y2,Z2,NATOM2,ATOMPR,NPAIR,LMASS,AMASS1,AMASS2,QWGHT,KWMULT,LNOROT,LPRINT)
    !-----------------------------------------------------------------------
    !     THE PROGRAM ROTATES COORDINATE SET 2 RESULTING IN A 2 SUCH THAT
    !     THE SUM OF THE SQUARE OF THE DISTANCE BETWEEN EACH COORDINATE IN 1
    !     AND 2 IS A MINIMUM.
    !     THE LEAST SQUARE MINIMIZATION IS DONE ONLY WITH RESPECT TO THE
    !     ATOMS REFERRED TO IN THE PAIR ARRAY. THE ROTATION MATRIX THAT IS
    !     CALCULATED IS APPLIED TO THE ENTIRE SECOND COORDINATE SET.
    !     BERNARD R. BROOKS
    !
  use chm_kinds
  use exfunc
  use number
  use memory
  use param_store, only: set_param
  use corsubs

    implicit none
    !
    INTEGER NATOM1,NATOM2,NPAIR
    real(chm_real) X1(*),Y1(*),Z1(*),X2(*),Y2(*),Z2(*)
    real(chm_real) AMASS1(*),AMASS2(*)
    real(chm_real) KWMULT(*)
    LOGICAL LMASS,QWGHT,LNOROT,LPRINT
    INTEGER ATOMPR(2,*)
    real(chm_real),allocatable,dimension(:) :: mass
    !
    call chmalloc('rotlsq.src','ROTLSQ','MASS',NPAIR,crl=MASS)
    CALL PKMASS(AMASS1,AMASS2,MASS,ATOMPR,NPAIR,LMASS,QWGHT,KWMULT)
    CALL EABFROTLS1(X1,Y1,Z1,X2,Y2,Z2,NATOM2,ATOMPR,NPAIR,MASS,LPRINT,LNOROT)
    call chmdealloc('rotlsq.src','ROTLSQ','MASS',NPAIR,crl=MASS)

    RETURN
  END SUBROUTINE EABFROTLSQ

  SUBROUTINE EABFROTLS1(XA,YA,ZA,XB,YB,ZB,NATOMRMSD,ATOMPR,NPAIR,BMASS,LPRINTP,LNOROT)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE DOES THE ACTUAL ROTATION OF B TO MATCH A. THE NEW
    !     ARRAY B IS RETURNED IN X.
    !     THIS ROUTINE WRITTEN BY B. BROOKS , ADAPTED FROM ACTA CRYST
    !     (1976) A32,922 W. KABSCH
    !
  use chm_kinds
  use number
  use stream
  use consta
  use param_store, only: set_param
  use corsubs
  use dimens_fcm
  use eabfsrc
    implicit none
    !
    INTEGER NPAIR,NATOMRMSD
    real(chm_real) BMASS(NPAIR)
    real(chm_real) XA(*),YA(*),ZA(*),XB(*),YB(*),ZB(*)
    INTEGER ATOMPR(2,NPAIR)
    LOGICAL LPRINTP,LNOROT
    !
    !
    real(chm_real) R(9),U(9),EVA(3),DEVA(3,3)
    real(chm_real) RN(3), PHI
    real(chm_real) CMXA,CMYA,CMZA,CMXB,CMYB,CMZB,CMXC,CMYC,CMZC
    real(chm_real) XI,YI,ZI,XJ,YJ,ZJ
    real(chm_real) TMASS,RMST,RMSV,CST,FACT,RSHIFT
    INTEGER K,KA,KB,I
    LOGICAL LPRINT2, QEVW, LPRINT
    !
    LPRINT=LPRINTP
    IF(PRNLEV <= 2)LPRINT=.FALSE.
    !
    LPRINT2=(LPRINT.AND.(PRNLEV > 6))
    !
    CMXA=0.0
    CMYA=0.0
    CMZA=0.0
    CMXB=0.0
    CMYB=0.0
    CMZB=0.0
    TMASS=0.0
    DO K=1,NPAIR
       KA=ATOMPR(1,K)
       KB=ATOMPR(2,K)
       IF (XA(KA) /= ANUM .AND. XB(KB) /= ANUM) THEN
          CMXA=CMXA+XA(KA)*BMASS(K)
          CMYA=CMYA+YA(KA)*BMASS(K)
          CMZA=CMZA+ZA(KA)*BMASS(K)
          CMXB=CMXB+XB(KB)*BMASS(K)
          CMYB=CMYB+YB(KB)*BMASS(K)
          CMZB=CMZB+ZB(KB)*BMASS(K)
          TMASS=TMASS+BMASS(K)
       ENDIF
    ENDDO
    CMXA=CMXA/TMASS
    CMYA=CMYA/TMASS
    CMZA=CMZA/TMASS
    CMXB=CMXB/TMASS
    CMYB=CMYB/TMASS
    CMZB=CMZB/TMASS
    DO K=1,NATOMRMSD
       IF(XB(K) /= ANUM) THEN
          XB(K)=XB(K)-CMXB
          YB(K)=YB(K)-CMYB
          ZB(K)=ZB(K)-CMZB
       ENDIF
    ENDDO
    !
    CMXC=CMXA-CMXB
    CMYC=CMYA-CMYB
    CMZC=CMZA-CMZB
    call set_param('XMOV',CMXC)
    call set_param('YMOV',CMYC)
    call set_param('ZMOV',CMZC)
    IF (LPRINT) THEN
       WRITE(OUTU,44) CMXB,CMYB,CMZB
       WRITE(OUTU,45) CMXA,CMYA,CMZA
       WRITE(OUTU,46) CMXC,CMYC,CMZC
    ENDIF
44  FORMAT(' CENTER OF ATOMS BEFORE TRANSLATION',3F12.5)
45  FORMAT(' CENTER OF REFERENCE COORDINATE SET',3F12.5)
46  FORMAT(' NET TRANSLATION OF ROTATED ATOMS  ',3F12.5)
    !
    IF (LNOROT) THEN
       !
       !       USE A UNIT ROTATION MATRIX. NO ROTATION IS SPECIFIED
       !
       DO K=1,NATOMRMSD
          IF (XB(K) /= ANUM) THEN
             XB(K)=XB(K)+CMXA
             YB(K)=YB(K)+CMYA
             ZB(K)=ZB(K)+CMZA
          ENDIF
       ENDDO
       !
    ELSE
       !
       !       COMPUTE ROTATION MATRIX FROM LAGRANGIAN
       !
       DO I=1,9
          R(I)=0.0
       ENDDO
       DO K=1,NPAIR
          KA=ATOMPR(1,K)
          KB=ATOMPR(2,K)
          IF (XA(KA) /= ANUM .AND. XB(KB) /= ANUM) THEN
             XI=XB(KB)*BMASS(K)
             YI=YB(KB)*BMASS(K)
             ZI=ZB(KB)*BMASS(K)
             XJ=XA(KA)-CMXA
             YJ=YA(KA)-CMYA
             ZJ=ZA(KA)-CMZA
             R(1)=R(1)+XI*XJ
             R(2)=R(2)+XI*YJ
             R(3)=R(3)+XI*ZJ
             R(4)=R(4)+YI*XJ
             R(5)=R(5)+YI*YJ
             R(6)=R(6)+YI*ZJ
             R(7)=R(7)+ZI*XJ
             R(8)=R(8)+ZI*YJ
             R(9)=R(9)+ZI*ZJ
          ENDIF
       ENDDO
       !
       CALL FROTU(R,EVA,DEVA,U,ZERO,QEVW,LPRINT2)
       !
       ! rotate/translate the atoms in set B to match set A.
       DO K=1,NATOMRMSD
          IF (XB(K) /= ANUM) THEN
             CMXC=U(1)*XB(K)+U(4)*YB(K)+U(7)*ZB(K)+CMXA
             CMYC=U(2)*XB(K)+U(5)*YB(K)+U(8)*ZB(K)+CMYA
             ZB(K)=U(3)*XB(K)+U(6)*YB(K)+U(9)*ZB(K)+CMZA
             XB(K)=CMXC
             YB(K)=CMYC
          ENDIF
       ENDDO
       !
       IF (LPRINT) WRITE(OUTU,55) U
55     FORMAT(' ROTATION MATRIX',3(/1X,3F12.6))
       CALL FNDROT(U,RN,PHI,LPRINT)
       !
       ! Compute center of rotation (if it's significant)
       IF(ABS(PHI) >= ONE) THEN
          IF(ABS(PHI) > 179.0) THEN
             CST=MINONE
          ELSE
             CST=COS(DEGRAD*PHI)
          ENDIF
          XI=CMXB-(U(1)*CMXB+U(4)*CMYB+U(7)*CMZB)
          YI=CMYB-(U(2)*CMXB+U(5)*CMYB+U(8)*CMZB)
          ZI=CMZB-(U(3)*CMXB+U(6)*CMYB+U(9)*CMZB)
          XI=XI+CMXA-(U(1)*CMXA+U(2)*CMYA+U(3)*CMZA)
          YI=YI+CMYA-(U(4)*CMXA+U(5)*CMYA+U(6)*CMZA)
          ZI=ZI+CMZA-(U(7)*CMXA+U(8)*CMYA+U(9)*CMZA)
          FACT=HALF/(ONE-CST)
          XI=XI*FACT
          YI=YI*FACT
          ZI=ZI*FACT
          FACT=RN(1)*(CMXA+CMXB)+RN(2)*(CMYA+CMYB)+RN(3)*(CMZA+CMZB)
          XI=XI+HALF*FACT*RN(1)
          YI=YI+HALF*FACT*RN(2)
          ZI=ZI+HALF*FACT*RN(3)
          RSHIFT=RN(1)*(CMXA-CMXB)+RN(2)*(CMYA-CMYB)+RN(3)*(CMZA-CMZB)
          IF (LPRINT) THEN
             WRITE(OUTU,65) XI,YI,ZI,RSHIFT
65           FORMAT(' CENTER OF ROTATION ',3F10.6,'  SHIFT IS',F10.6/)
          ENDIF
       ELSE
          !         don't bother for very small rotations...
          XI=ZERO
          YI=ZERO
          ZI=ZERO
          RSHIFT=ZERO
       ENDIF
       !
       ! Set substitution variables
       call set_param('THET',PHI)
       call set_param('SHIFT',RSHIFT)
       QAXISC=.TRUE.
       AXISCX= XI
       AXISCY= YI
       AXISCZ= ZI
       AXISR = ONE
       AXISX = RN(1)
       AXISY = RN(2)
       AXISZ = RN(3)
       call set_param('XAXI',AXISX)
       call set_param('YAXI',AXISY)
       call set_param('ZAXI',AXISZ)
       call set_param('RAXI',AXISR)
       call set_param('XCEN',AXISCX)
       call set_param('YCEN',AXISCY)
       call set_param('ZCEN',AXISCZ)
    ENDIF
    !
    RMST=0.0
    DO K=1,NPAIR
       KA=ATOMPR(1,K)
       KB=ATOMPR(2,K)
       IF (XA(KA) /= ANUM .AND. XB(KB) /= ANUM) RMST=RMST+ &
            BMASS(K)*((XB(KB)-XA(KA))**2+(YB(KB)-YA(KA))**2+ &
            (ZB(KB)-ZA(KA))**2)
    ENDDO
    RMSV=SQRT(RMST/TMASS)
    !eABF
    eabf_rms_tmp=RMSV
    call set_param('RMS ',RMSV)
    IF(LPRINT) WRITE(OUTU,14) RMST,TMASS,RMSV
14  FORMAT(' TOTAL SQUARE DIFF IS',F12.4,'  DENOMINATOR IS',F12.4,/ &
         '       THUS RMS DIFF IS',F12.6)
    !

    !eABF
    !Here, we want to rotate the target selection so that below we can get the RMSD for the target... after we get the RMSD
    !compute the transpose of the rotation matrix to move the atoms back...


    !Rotate & Translate Atoms Back to original positions
    DO K=1,NATOMRMSD
          IF (XB(K) /= ANUM) THEN
             XB(K)=XB(K)-CMXA
             YB(K)=YB(K)-CMYA
             ZB(K)=ZB(K)-CMZA
             CMXC=U(1)*XB(K)+U(2)*YB(K)+U(3)*ZB(K)
             CMYC=U(4)*XB(K)+U(5)*YB(K)+U(6)*ZB(K)
             ZB(K)=U(7)*XB(K)+U(8)*YB(K)+U(9)*ZB(K)
             XB(K)=CMXC
             YB(K)=CMYC
             XB(K)=XB(K)+CMXB
             YB(K)=YB(K)+CMYB
             ZB(K)=ZB(K)+CMZB
          ENDIF
    ENDDO
    U_copy(1)=U(1)
    U_copy(2)=U(2)
    U_copy(3)=U(3)
    U_copy(4)=U(4)
    U_copy(5)=U(5)
    U_copy(6)=U(6)
    U_copy(7)=U(7)
    U_copy(8)=U(8)
    U_copy(9)=U(9)
    com_x_ref=CMXB
    com_y_ref=CMYB
    com_z_ref=CMZB

    RETURN
  END SUBROUTINE EABFROTLS1

SUBROUTINE EABFRMSDOTF(curr_rmsd_ref,curr_cv)
  use chm_kinds
  use number
  use stream
  use consta
  use param_store, only: set_param
  use corsubs
  use dimens_fcm
  use coord
  use coordc
  use contrl
  use psf
  use eabfsrc
  implicit none
  integer :: i,ii,curr_cv,curr_rmsd_ref
  real(chm_real) :: CMXC,CMYC

if(MDSTEP .EQ. 0 .OR. eabf_rst .EQ. 1)then
if(curr_rmsd_ref .EQ. 0)then
   !Fill Pair Array for RMS fitting
   ALLOCATE (pair_array(2,NAtom))
   ALLOCATE (rot_selct(1:NAtom))
   pair_array(:,:)=0
   rot_selct(:)=0
endif
endif
   pair_array(:,:)=0
   rot_selct(:)=0
   !Set CHARMM COMParison set to current atom position unless its a selection. Also set the atoms to be used to calculate the rotation via rot_selct
   do i=1,NAtom
      XCOMP(i)=X(i)
      YCOMP(i)=Y(i)
      ZCOMP(i)=Z(i)
      do ii=1,cv_natom_lists(curr_cv,0)
         if(cv_selection_lists(curr_cv,0,ii) .EQ. i)then
            rot_selct(i)=1
            XCOMP(i)=initial_rmsd_refs_x(curr_rmsd_ref,ii)
            YCOMP(i)=initial_rmsd_refs_y(curr_rmsd_ref,ii)
            ZCOMP(i)=initial_rmsd_refs_z(curr_rmsd_ref,ii)
         endif
      enddo
   enddo

   CALL EABFORINTC(NAtom,X,Y,Z,XCOMP,YCOMP,ZCOMP,AMASS,.TRUE.,.TRUE.,pair_array,rot_selct,.FALSE.,WMAIN,.FALSE.,.FALSE.)

   !Call below gets the rotation matrix (which is called U_copy below) that aligns the current configuration to the reference. However, within this call the current coordinates are rotated back to their position upon entry. Following the call, we rotate the reference coordinates to the current coordinates as to not interfere with the dynamics (not rotating the entire system, instead rotate the reference)
   do i=1,cv_natom_lists(curr_cv,0)
      CMXC=U_copy(1)*initial_rmsd_refs_x(curr_rmsd_ref,i)+U_copy(2)*initial_rmsd_refs_y(curr_rmsd_ref,i)+U_copy(3)*initial_rmsd_refs_z(curr_rmsd_ref,i)
      CMYC=U_copy(4)*initial_rmsd_refs_x(curr_rmsd_ref,i)+U_copy(5)*initial_rmsd_refs_y(curr_rmsd_ref,i)+U_copy(6)*initial_rmsd_refs_z(curr_rmsd_ref,i)
      rmsd_refs_z(curr_rmsd_ref,i)=U_copy(7)*initial_rmsd_refs_x(curr_rmsd_ref,i)+U_copy(8)*initial_rmsd_refs_y(curr_rmsd_ref,i)+U_copy(9)*initial_rmsd_refs_z(curr_rmsd_ref,i)
      rmsd_refs_x(curr_rmsd_ref,i)=CMXC
      rmsd_refs_y(curr_rmsd_ref,i)=CMYC
      rmsd_refs_x(curr_rmsd_ref,i)=rmsd_refs_x(curr_rmsd_ref,i)+com_x_ref
      rmsd_refs_y(curr_rmsd_ref,i)=rmsd_refs_y(curr_rmsd_ref,i)+com_y_ref
      rmsd_refs_z(curr_rmsd_ref,i)=rmsd_refs_z(curr_rmsd_ref,i)+com_z_ref
   enddo
   eabf_rms(curr_cv)=eabf_rms_tmp
END SUBROUTINE

