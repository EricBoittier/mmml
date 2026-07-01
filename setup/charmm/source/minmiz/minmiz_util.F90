module minmiz_util_module
  implicit none
contains
  
SUBROUTINE MINTRJ(NCALLS,NSTEPS,STEP)
  !-----------------------------------------------------------------------
  !     Writes trajectory frame of minimization steps
  !
#if KEY_CHEQ==1
  use cheq,only:qcg     
#endif

  use chm_kinds
  use dimens_fcm
  use reawri
  use psf
  use coord
  use cvio
  use deriv
  use dynio
  use number
  use ctitla
  !
  implicit none
  !
  INTEGER NCALLS,NSTEPS
  real(chm_real) STEP,T(1)
  !
  INTEGER NAT3
  !
  T(1)=ZERO
  IF((NSAVC > 0).AND.(IUNCRD >= 0) &
       .AND.(MOD(NCALLS,NSAVC) == 0)) THEN
     NAT3=3*NATOM
     CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
          CG,QCG,                                & 
#endif
          NATOM, (/ 0 /), NATOM,NCALLS,NCALLS,NAT3,STEP,NSAVC, &
          NSTEPS,TITLEA,NTITLA,IUNCRD,.FALSE.,.FALSE., (/ 0 /), .FALSE., (/ ZERO /))
  ENDIF
  IF((NSAVX > 0).AND.(IUNXYZ >= 0).AND.(MOD(NCALLS,NSAVX) == 0)) &
       CALL WRXYZ(IUNXYZ,X,Y,Z,T,T,T,DX,DY,DZ,NCALLS)
  !
  RETURN
END SUBROUTINE MINTRJ

end module minmiz_util_module
