module lupcns_module
  implicit none
contains
#if KEY_RXNCOR==1  
  SUBROUTINE LUPCNS(X,Y,Z,XR,YR,ZR,DX,DY,DZ,AMASS,NATOM,XC,YC,ZC)
    !-----------------------------------------------------------------------
    !     Set up LUP constraints and orthogonalize gradient to them
    !
    !     The input data are:
    !     X,Y,Z - current coordinates
    !     DX,DY,DZ - current energy gradient
    !     XR,YR,ZR - vector of path direction
    !     AMASS - atomic masses
    !     NATOM - number of atoms
    !
    !     Work arrays:
    !     XC,YC,ZC - constraint coefficient arrays
    !     WORK - storage array
    !     NLUPC - number of LUP constraints, passed from lupcom.f90
    !
    !     K. Kuczera,  Lawrence, KS 08-Mar-97
    !
    use chm_kinds
    use number
    use dimens_fcm
    use exfunc
    use stream
    use lupcom
    implicit none
    !
    !     Passed variables.
    !
    INTEGER NATOM
    real(chm_real) X(NATOM),Y(NATOM),Z(NATOM), &
         XR(NATOM),YR(NATOM),ZR(NATOM)
    real(chm_real) DX(NATOM),DY(NATOM),DZ(NATOM),AMASS(NATOM)
    real(chm_real) XC(NLUPC,NATOM),YC(NLUPC,NATOM),ZC(NLUPC,NATOM)
    !
    !     Local variables.
    !
    INTEGER I,J,K,I1
    real(chm_real)  S
    !
    ! ... Set up the initial linear constraint coefficients
    !
    DO I=1,NLUPC
       DO J=1,NATOM
          XC(I,J) =  ZERO
          YC(I,J) =  ZERO
          ZC(I,J) =  ZERO
       END DO
    END DO
    !
    DO J=1,NATOM
       ! ... CM translations  Mj*Rj
       XC(1,J) = AMASS(J)
       YC(2,J) = AMASS(J)
       ZC(3,J) = AMASS(J)
       ! ... CM rotations (Eckart) Mj*(R0j x (Rj - R0j))
       YC(4,J) =  -AMASS(J)*Z(J)
       ZC(4,J) =   AMASS(J)*Y(J)
       XC(5,J) =   AMASS(J)*Z(J)
       ZC(5,J) =  -AMASS(J)*X(J)
       XC(6,J) =  -AMASS(J)*Y(J)
       YC(6,J) =   AMASS(J)*X(J)
       ! ... Path tangent vector
       XC(7,J) = XR(J)
       YC(7,J) = YR(J)
       ZC(7,J) = ZR(J)
    END DO    ! J=1,NATOM

    !
    ! ... Orthonormalize the constraint coefficient vectors
    !
    DO I=1,NLUPC
       ! ... normalize vector I
       S=ZERO
       DO J=1,NATOM
          S = S + XC(I,J)**2 + YC(I,J)**2 + ZC(I,J)**2
       END DO
       S = SQRT(S)
       !
       IF(S < TENM5) THEN
          WRITE(OUTU,*) ' LUPCNS> ****Error : cannot normalize vector',I
          CALL WRNDIE(-4,'LUPCNS',' Null vector encountered')
       END IF
       !
       DO J=1,NATOM
          XC(I,J) = XC(I,J)/S
          YC(I,J) = YC(I,J)/S
          ZC(I,J) = ZC(I,J)/S
       END DO
       ! ... orthogonalize vectors I+1, I+2, ..., NLUPC to vector I
       I1 = I + 1
       S = ZERO
       DO K=I1,NLUPC
          !
          DO J=1,NATOM
             S = S + XC(I,J)*XC(K,J) + YC(I,J)*YC(K,J) + ZC(I,J)*ZC(K,J)
          END DO
          !
          DO J=1,NATOM
             XC(K,J) = XC(K,J) - S*XC(I,J)
             YC(K,J) = YC(K,J) - S*YC(I,J)
             ZC(K,J) = ZC(K,J) - S*ZC(I,J)
          END DO
          !
       END DO    ! K=I1,NLUPC
       !
    END DO      ! I=1,NLUPC
    !
    ! ... Remove component parallel to constraints from energy gradient
    !
    DO I=1,NLUPC
       !
       S = ZERO
       DO J=1,NATOM
          S = S + XC(I,J)*DX(J) + YC(I,J)*DY(J) + ZC(I,J)*DZ(J)
       END DO
       !
       DO J=1,NATOM
          DX(J) = DX(J) - S*XC(I,J)
          DY(J) = DY(J) - S*YC(I,J)
          DZ(J) = DZ(J) - S*ZC(I,J)
       END DO
       !
    END DO
    !
    RETURN
  END SUBROUTINE LUPCNS
#endif /* KEY_RXNCOR */
end module lupcns_module
