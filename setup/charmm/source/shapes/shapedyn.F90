#if KEY_SHAPES==1 /*main*/
SUBROUTINE SHAKBODY(X,Y,Z,XREF,YREF,ZREF,AMASS,LMASS,LDYNA, &
     NATOM,IMOVE,ISLCT,NUMSH,SHPTYP)
  !
  ! This routine does "shake" for rigid bodies and allows holonomic
  ! constraints for both dynamics and minimization.
  !
  !     By Bernard R. Brooks   February, 1998
  !
  use chm_kinds
  use number
  use vector
  use stream
  use consta
  use parallel
  !
  implicit none
  !
  INTEGER NATOM,NUMSH
  real(chm_real) X(NATOM),Y(NATOM),Z(NATOM)
  real(chm_real) XREF(NATOM),YREF(NATOM),ZREF(NATOM)
  real(chm_real) AMASS(NATOM)
  LOGICAL LMASS,LDYNA
  INTEGER IMOVE(NATOM),ISLCT(NATOM)
  CHARACTER(len=4) SHPTYP(NUMSH)
  !
  INTEGER I,J,K,IS,NBAD,NITER
  INTEGER ATFRST,ATLAST
  real(chm_real) AM,TOL,DOT,VAL1,VAL2,MASSH,ANGMM,THETA,TSQ
  real(chm_real) CMO(3),CMN(3),ANGMV(3),INERT(3,3),RINERT(3,3), &
       ROTM(3,3)
  real(chm_real) CX,CY,CZ,PX,PY,PZ,RX,RY,RZ
  real(chm_real) T(3,3),T2(3,3),ROTV(3),ROTW(3),LT(3),SCR(21), &
       AMOM(6)
  real(chm_real) AMXX,AMXY,AMXZ,AMYY,AMYZ,AMZZ
  LOGICAL OK
  !
#if KEY_DEBUG==1
  write(outu,234) ' entering SHAKBODY'
234 format(A)
  write(outu,235) is,'    X=    ',X
  !     write(outu,235) is,'    Y=    ',Y
  !     write(outu,235) is,'    Z=    ',Z
  !     write(outu,235) is,'    XREF= ',XREF
  !     write(outu,235) is,'    YREF= ',YREF
  !     write(outu,235) is,'    ZREF= ',ZREF
235 format(I5,A,9F20.10)
#endif 
  !
  TOL=TENM14
  !
  ATFRST=1
  ATLAST=NATOM
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
  !     Define the atom bounds for this processor.
  IF(LDYNA) THEN
     ATFRST=1+IPARPT(MYNOD)
     ATLAST=IPARPT(MYNODP)
  ENDIF
#endif 
#endif 
  !
  ! Note: Although some steps seem inefficient, extra work is done
  ! to preserve accuracy for very small steps and very small rotations
  !
  DO IS=1,NUMSH
     IF(SHPTYP(IS).NE.'RIGI') GOTO 800
     MASSH=ZERO
     DO I=1,3
        CMO(I)=ZERO
        CMN(I)=ZERO
        ANGMV(I)=ZERO
     ENDDO
     AMXX=ZERO
     AMXY=ZERO
     AMXZ=ZERO
     AMYY=ZERO
     AMYZ=ZERO
     AMZZ=ZERO
     !
     ! Compute center of mass and weight of each body
     !
     NBAD=0
     DO I=ATFRST,ATLAST
        IF(ISLCT(I).EQ.IS) THEN
           IF(IMOVE(I).NE.0) NBAD=NBAD+1
           IF(LMASS) THEN
              AM=AMASS(I)
           ELSE
              AM=ONE
           ENDIF
           MASSH=MASSH+AM
           CMO(1)=CMO(1)+AM*XREF(I)
           CMO(2)=CMO(2)+AM*YREF(I)
           CMO(3)=CMO(3)+AM*ZREF(I)
           CMN(1)=CMN(1)+AM*X(I)             
           CMN(2)=CMN(2)+AM*Y(I)
           CMN(3)=CMN(3)+AM*Z(I)
        ENDIF
     ENDDO
     !
     IF(NBAD.GT.0) THEN
        CALL WRNDIE(-2,'<SHAKEBODY>', &
             'Atoms with IMOVE(I).NE.0 found in a rigid object -ignored')
        SHPTYP(IS)='NONE' ! remove this shape as a rigid object
        GOTO 800
     ENDIF
     IF(MASSH.LE.TOL) GOTO 800 ! ignore bodies with no mass
     !                                 ! or those on other processors
     !
#if KEY_PARALLEL==1
     IF(LDYNA) THEN
        ! Optional check to make sure that shapes don't cross processor
        ! boundaries.  This check really should be done only once
        ! (i.e. somewhare else...) at the beginning of dynamics/minimization
        NBAD=0
        DO I=1,ATFRST-1
           IF(ISLCT(I).EQ.IS) NBAD=NBAD+1
        ENDDO
        DO I=ATLAST+1,NATOM
           IF(ISLCT(I).EQ.IS) NBAD=NBAD+1
        ENDDO
        IF(NBAD.GT.0) THEN
           CALL WRNDIE(-2,'<SHAKEBODY>', &
                'Atoms of a rigid body cross processor boundaries -ignored')
           SHPTYP(IS)='NONE' ! remove this shape as a rigid object
           !           This problem can be avoided by making the object one group.
           GOTO 800
        ENDIF
     ENDIF
#endif 
     !
     DO J=1,3
        CMO(J)=CMO(J)/MASSH
        CMN(J)=CMN(J)/MASSH
     ENDDO
     !
#if KEY_DEBUG==1
     write(outu,235) is,'    MASSH=',MASSH
     write(outu,235) is,'    CMO=  ',CMO
     write(outu,235) is,'    CMN=  ',CMN
#endif 
     !
     ! compute angular momentum and moments of inertia for each body
     !
     DO I=ATFRST,ATLAST
        IF(ISLCT(I).EQ.IS) THEN
           IF(LMASS) THEN
              AM=AMASS(I)
           ELSE
              AM=ONE
           ENDIF
           RX=XREF(I)-CMO(1)
           RY=YREF(I)-CMO(2)
           RZ=ZREF(I)-CMO(3)
           AMXX=AMXX+AM*RX*RX
           AMXY=AMXY+AM*RX*RY
           AMXZ=AMXZ+AM*RX*RZ
           AMYY=AMYY+AM*RY*RY
           AMYZ=AMYZ+AM*RY*RZ
           AMZZ=AMZZ+AM*RZ*RZ
           !
           CX=X(I)-RX-CMN(1)
           CY=Y(I)-RY-CMN(2)
           CZ=Z(I)-RZ-CMN(3)
           PX=CY*RZ-CZ*RY
           PY=CZ*RX-CX*RZ
           PZ=CX*RY-CY*RX
           ANGMV(1)=ANGMV(1)+AM*PX
           ANGMV(2)=ANGMV(2)+AM*PY
           ANGMV(3)=ANGMV(3)+AM*PZ
        ENDIF
     ENDDO
     !
     INERT(1,1)= AMYY+AMZZ
     INERT(2,1)=-AMXY     
     INERT(3,1)=-AMXZ     
     INERT(1,2)=-AMXY     
     INERT(2,2)= AMXX+AMZZ
     INERT(3,2)=-AMYZ
     INERT(1,3)=-AMXZ     
     INERT(2,3)=-AMYZ     
     INERT(3,3)= AMYY+AMXX
     !
     ! Calculate the inverse of the moment of inertia symmetric tensor
     !
     CALL INVT33(RINERT, INERT, OK)
     IF(.NOT.OK) THEN
        !         the molecule must be linear... use a different method
        AMOM(1)=INERT(1,1)
        AMOM(2)=INERT(2,1)
        AMOM(3)=INERT(3,1)
        AMOM(4)=INERT(2,2)
        AMOM(5)=INERT(3,2)
        AMOM(6)=INERT(3,3)
        !         diagonalize the matrix
        CALL DIAGQ(3,3,AMOM,ROTM,SCR(4),SCR(7),SCR(10),SCR(13),LT, &
             SCR(16),SCR(19),SCR(1),0)
        DO I=1,3
           DO J=1,3
              T(J,I)=ZERO
           ENDDO
           IF(LT(I).LT.TENM5) THEN
              T(I,I)=ONE
           ELSE
              T(I,I)=ONE/LT(I)
           ENDIF
        ENDDO
        CALL MULMXN(T2,ROTM,3,3,T,3,3)
        CALL TRANSPS(T,T2,3,3)
        CALL MULMXN(RINERT,ROTM,3,3,T,3,3)
        CALL MULMXN(T2,ROTM,3,3,T,3,3)
     ENDIF
     !
     ! Get the initial guess for the rotation matrix (assume uncoupled)
     !
     CALL MULMXN(ROTW,RINERT,3,3,ANGMV,3,1)
     !
#if KEY_DEBUG==1
     CALL DOTPR(ANGMV,ANGMV,3,ANGMM)
     ANGMM=SQRT(ANGMM)      ! magnitude of the angular mom.
     write(outu,235) is,'    ANGMV=',ANGMV
     write(outu,235) is,'    ANGMM=',ANGMM
     write(outu,235) is,'    INERT=',INERT
     write(outu,235) is,'   RINERT=',RINERT
     write(outu,235) is,'    ROTW= ',ROTW
#endif 
     !
     !-----------------------------------------------------------------------
     ! Iterative solution (non-linear problem) starts here
     !    (3-4 iterations will typically be required)
     !
     NITER=0
200  CONTINUE
     !
     CALL DOTPR(ROTW,ROTW,3,THETA)
     THETA=SQRT(THETA)      ! angle of the current rotation
     !
     IF(THETA.GT.HALF*PI) THEN
        CALL WRNDIE(0,'<SHAKEBODY>', &
             'Deviation too large for convergence - using 60 deg. rot.')
        THETA=THIRD*PI
        NITER=-1  ! do not do any iterations (it's pointless)
     ENDIF
     !
     ! Generate the rotation matrix (from R and theta)
     !
     TSQ=THETA*THETA
     IF(THETA.GT.PT01) THEN
        !         do a direct calculation
        VAL1=SIN(THETA)/THETA
        VAL2=(ONE-COS(THETA))/TSQ
     ELSE
        !         do a taylor expansion of above for small angles
        VAL1=ONE-TSQ*SIXTH*(ONE-PT05*TSQ)
        VAL2=HALF*(ONE-TSQ*(ONE-TSQ/THIRTY)/TWELVE)
     ENDIF
     !
#if KEY_DEBUG==1
     !      write(outu,235) is,' SIN(T)/T=',val1
     !      write(outu,235) is,'    THETA=',THETA*180./pi
#endif 
     !
     T(3,2)= ROTW(1)
     T(2,3)=-ROTW(1)
     T(1,3)= ROTW(2)
     T(3,1)=-ROTW(2)
     T(2,1)= ROTW(3)
     T(1,2)=-ROTW(3)
     T(1,1)=ZERO
     T(2,2)=ZERO
     T(3,3)=ZERO
     DO I=1,3
        DO J=1,3
           ROTM(J,I)=ZERO
        ENDDO
        ROTM(I,I)=ONE
     ENDDO
     CALL ADDCTV(ROTM,T,9,VAL1)
     CALL MULNXN(T2,T,T,3)
     CALL ADDCTV(ROTM,T2,9,VAL2)
     !
     ! calculate the angular momentum if this rotation matrix is used
     !
     LT(1)=AMXY*ROTM(1,3)+AMYY*ROTM(2,3)+AMYZ*ROTM(3,3) &
          -AMXZ*ROTM(1,2)-AMYZ*ROTM(2,2)-AMZZ*ROTM(3,2)
     LT(2)=AMXZ*ROTM(1,1)+AMYZ*ROTM(2,1)+AMZZ*ROTM(3,1) &
          -AMXX*ROTM(1,3)-AMXY*ROTM(2,3)-AMXZ*ROTM(3,3)
     LT(3)=AMXX*ROTM(1,2)+AMXY*ROTM(2,2)+AMXZ*ROTM(3,2) &
          -AMXY*ROTM(1,1)-AMYY*ROTM(2,1)-AMYZ*ROTM(3,1)
     !
     ! calculate the current error in the angular momentum
     !
     DO J=1,3
        LT(J)=ANGMV(J)+LT(J)
     ENDDO
     !
     ! check for convergence
     !
     IF(NITER.LT.0) GOTO 400    ! deviation too large exit
     NITER=NITER+1
     IF(NITER.GT.100) GOTO 300  ! too many iterations exit
     CALL DOTPR(LT,LT,3,DOT)
     IF(DOT.LT.TOL) GOTO 400    ! standard iteration exit

#if KEY_DEBUG==1
     write(outu,235) is,'    LTM= ',LT
#endif 
     !
     ! calculate the change in the rotation matrix needed to remove the error
     !  (to optimize convergence use Rdel = R**(1/2) * I**-1 * ERRvec)
     !
     CALL MULMXN(ROTV,RINERT,3,3,LT,3,1)
     CALL MULMXN(LT,ROTV,1,3,ROTM,3,3)
     !
     DO J=1,3
        ROTW(J)=ROTW(J)+(LT(J)+ROTV(J))*HALF
     ENDDO
     !
     GOTO 200
     !
     !  end of iteration
     !-----------------------------------------------------------------------
     !
300  CONTINUE  ! abnormal iteration exit
     CALL WRNDIE(0,'<SHAKEBODY>', &
          'Convergence not reached in 100 iterations')
400  CONTINUE  ! normal iteration exit
     !
#if KEY_DEBUG==1
     IF(THETA.GT.TOL) call normall(ROTW,3)
     write(outu,235) is,'final   ROTW= ',ROTW
     write(outu,235) is
#endif 
     !
     ! now, put the atoms where they should be...
     !
     DO I=ATFRST,ATLAST
        IF(ISLCT(I).EQ.IS) THEN 
           RX=XREF(I)-CMO(1)
           RY=YREF(I)-CMO(2)
           RZ=ZREF(I)-CMO(3)
           CX=RX*ROTM(1,1)+RY*ROTM(2,1)+RZ*ROTM(3,1)
           CY=RX*ROTM(1,2)+RY*ROTM(2,2)+RZ*ROTM(3,2)
           CZ=RX*ROTM(1,3)+RY*ROTM(2,3)+RZ*ROTM(3,3)
           X(I)=CX+CMN(1)
           Y(I)=CY+CMN(2)
           Z(I)=CZ+CMN(3)
        ENDIF
     ENDDO

800  CONTINUE  ! exit shape (for wrong or bad shapes)
  ENDDO ! DO(IS)
  !
  RETURN
END SUBROUTINE SHAKBODY

SUBROUTINE SHAKBODYF(DX,DY,DZ,XREF,YREF,ZREF,AMASS,LMASS, &
     NATOM,IMOVE,ISLCT,NUMSH,SHPTYP)
  !
  ! This routine does "shake" force removal for rigid bodies
  !
  !     By Bernard R. Brooks   February, 1998
  !
  use chm_kinds
  use number
  use exfunc
  use stream
  use consta
  !
  implicit none
  !
  INTEGER NATOM,NUMSH
  real(chm_real) DX(NATOM),DY(NATOM),DZ(NATOM)
  real(chm_real) XREF(NATOM),YREF(NATOM),ZREF(NATOM)
  real(chm_real) AMASS(NATOM)
  LOGICAL LMASS
  INTEGER IMOVE(NATOM),ISLCT(NATOM)
  CHARACTER(len=4) SHPTYP(NUMSH)
  !
  INTEGER I,J,K,IS,NBAD
  real(chm_real) AM,TOL,DOT,MASSH,ANGMM,THETA
  real(chm_real) CMO(3),CMN(3),ANGMV(3),INERT(3,3),RINERT(3,3), &
       ROTM(3,3)
  real(chm_real) CX,CY,CZ,PX,PY,PZ,RX,RY,RZ
  real(chm_real) T(3,3),T2(3,3),ROTW(3),LT(3),SCR(21),AMOM(6)
  real(chm_real) AMXX,AMXY,AMXZ,AMYY,AMYZ,AMZZ
  LOGICAL OK
  !
#if KEY_DEBUG==1
  write(outu,234) ' entering SHAKBODYF'
234 format(A)
  write(outu,235) is,'    X=    ',X
  !     write(outu,235) is,'    Y=    ',Y
  !     write(outu,235) is,'    Z=    ',Z
  !     write(outu,235) is,'    XREF= ',XREF
  !     write(outu,235) is,'    YREF= ',YREF
  !     write(outu,235) is,'    ZREF= ',ZREF
235 format(I5,A,9F20.10)
#endif 
  !
  TOL=TENM14
  !
  ! Note: Although some steps seem inefficient, extra work is done
  ! to preserve accuracy for very small steps and very small rotations
  !
  DO IS=1,NUMSH
     IF(SHPTYP(IS).NE.'RIGI') GOTO 800
     MASSH=ZERO
     DO I=1,3
        CMO(I)=ZERO
        CMN(I)=ZERO
        ANGMV(I)=ZERO
     ENDDO
     AMXX=ZERO
     AMXY=ZERO
     AMXZ=ZERO
     AMYY=ZERO
     AMYZ=ZERO
     AMZZ=ZERO
     !
     ! Compute center of mass and weight and total force of each body
     !
     NBAD=0
     DO I=1,NATOM
        IF(ISLCT(I).EQ.IS) THEN
           IF(IMOVE(I).NE.0) NBAD=NBAD+1
           IF(LMASS) THEN
              AM=AMASS(I)
           ELSE
              AM=ONE
           ENDIF
           MASSH=MASSH+AM
           CMO(1)=CMO(1)+AM*XREF(I)
           CMO(2)=CMO(2)+AM*YREF(I)
           CMO(3)=CMO(3)+AM*ZREF(I)
           CMN(1)=CMN(1)+DX(I)             
           CMN(2)=CMN(2)+DY(I)
           CMN(3)=CMN(3)+DZ(I)
        ENDIF
     ENDDO
     !
     IF(NBAD.GT.0) THEN
        CALL WRNDIE(-2,'<SHAKEBODY>', &
             'Atoms with IMOVE(I).NE.0 found in a rigid object -ignored')
        SHPTYP(IS)='NONE'
        GOTO 800
     ENDIF
     IF(MASSH.LE.TOL) GOTO 800 ! ignore bodies with no mass
     !
     DO J=1,3
        CMO(J)=CMO(J)/MASSH   ! c.m. position
        CMN(J)=CMN(J)/MASSH   ! c.m. acceleration
     ENDDO
     !
#if KEY_DEBUG==1
     write(outu,235) is,'    MASSH=',MASSH
     write(outu,235) is,'    CMO=  ',CMO
     write(outu,235) is,'    CMN=  ',CMN
#endif 
     !
     ! compute torque and moments of inertia for each body
     !
     DO I=1,NATOM
        IF(ISLCT(I).EQ.IS) THEN
           IF(LMASS) THEN
              AM=AMASS(I)
           ELSE
              AM=ONE
           ENDIF
           RX=XREF(I)-CMO(1)
           RY=YREF(I)-CMO(2)
           RZ=ZREF(I)-CMO(3)
           AMXX=AMXX+AM*RX*RX
           AMXY=AMXY+AM*RX*RY
           AMXZ=AMXZ+AM*RX*RZ
           AMYY=AMYY+AM*RY*RY
           AMYZ=AMYZ+AM*RY*RZ
           AMZZ=AMZZ+AM*RZ*RZ
           !
           CX=DX(I)-CMN(1)*AM
           CY=DY(I)-CMN(2)*AM
           CZ=DZ(I)-CMN(3)*AM
           PX=CY*RZ-CZ*RY
           PY=CZ*RX-CX*RZ
           PZ=CX*RY-CY*RX
           ANGMV(1)=ANGMV(1)+PX
           ANGMV(2)=ANGMV(2)+PY
           ANGMV(3)=ANGMV(3)+PZ
        ENDIF
     ENDDO
     !
     INERT(1,1)= AMYY+AMZZ
     INERT(2,1)=-AMXY     
     INERT(3,1)=-AMXZ     
     INERT(1,2)=-AMXY     
     INERT(2,2)= AMXX+AMZZ
     INERT(3,2)=-AMYZ
     INERT(1,3)=-AMXZ     
     INERT(2,3)=-AMYZ     
     INERT(3,3)= AMYY+AMXX
     !
     ! Calculate the inverse of the moment of inertia symmetric tensor
     !
     CALL INVT33(RINERT, INERT, OK)
     IF(.NOT.OK) THEN
        !         the molecule is linear... use a different method
        AMOM(1)=INERT(1,1)
        AMOM(2)=INERT(2,1)
        AMOM(3)=INERT(3,1)
        AMOM(4)=INERT(2,2)
        AMOM(5)=INERT(3,2)
        AMOM(6)=INERT(3,3)
        !         diagonalize the matrix
        CALL DIAGQ(3,3,AMOM,ROTM,SCR(4),SCR(7),SCR(10),SCR(13),LT, &
             SCR(16),SCR(19),SCR(1),0)
        DO I=1,3
           DO J=1,3
              T(J,I)=ZERO
           ENDDO
           IF(LT(I).LT.TENM5) THEN
              T(I,I)=ONE
           ELSE
              T(I,I)=ONE/LT(I)
           ENDIF
        ENDDO
        CALL MULMXN(T2,ROTM,3,3,T,3,3)
        CALL TRANSPS(T,T2,3,3)
        CALL MULMXN(RINERT,ROTM,3,3,T,3,3)
        CALL MULMXN(T2,ROTM,3,3,T,3,3)
     ENDIF
     !
     ! Get the angular acceleration vector (and corresponding tensor)
     !
     CALL MULMXN(ROTW,RINERT,3,3,ANGMV,3,1)
     !
     ROTM(3,2)= ROTW(1)
     ROTM(2,3)=-ROTW(1)
     ROTM(1,3)= ROTW(2)
     ROTM(3,1)=-ROTW(2)
     ROTM(2,1)= ROTW(3)
     ROTM(1,2)=-ROTW(3)
     ROTM(1,1)=ZERO
     ROTM(2,2)=ZERO
     ROTM(3,3)=ZERO
     !
#if KEY_DEBUG==1
     CALL DOTPR(ANGMV,ANGMV,3,ANGMM)
     ANGMM=SQRT(ANGMM)      ! magnitude of the torque.
     write(outu,235) is,'    ANGMV=',ANGMV
     write(outu,235) is,'    ANGMM=',ANGMM
     write(outu,235) is,'    INERT=',INERT
     write(outu,235) is,'   RINERT=',RINERT
     write(outu,235) is,'    ROTW= ',ROTW
     CALL DOTPR(ROTW,ROTW,3,THETA)
     THETA=SQRT(THETA)     ! angular acceleration magnitude
     write(outu,235) is,'    THETA=',THETA*180./pi
     write(outu,235) is,'final   ROTM= ',ROTM
     write(outu,235) is
#endif 
     !
     ! now, put the forces on the atoms...
     !
     DO I=1,NATOM
        IF(ISLCT(I).EQ.IS) THEN 
           IF(LMASS) THEN
              AM=AMASS(I)
           ELSE
              AM=ONE
           ENDIF
           RX=XREF(I)-CMO(1)
           RY=YREF(I)-CMO(2)
           RZ=ZREF(I)-CMO(3)
           CX=RX*ROTM(1,1)+RY*ROTM(2,1)+RZ*ROTM(3,1)
           CY=RX*ROTM(1,2)+RY*ROTM(2,2)+RZ*ROTM(3,2)
           CZ=RX*ROTM(1,3)+RY*ROTM(2,3)+RZ*ROTM(3,3)
           DX(I)=(CX+CMN(1))*AM
           DY(I)=(CY+CMN(2))*AM
           DZ(I)=(CZ+CMN(3))*AM
        ENDIF
     ENDDO
800  CONTINUE  ! exit shape (for wrong or bad shapes)
  ENDDO ! DO(IS)
  !
  RETURN
END SUBROUTINE SHAKBODYF
#endif /* (main)*/
SUBROUTINE NULL_SHKBOD
  RETURN
END SUBROUTINE NULL_SHKBOD

