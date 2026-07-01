#if KEY_SHELL==1 || KEY_RDFSOL==1 || KEY_PROTO==1 || KEY_CORSOL==1 /*shlsel-rdfsol-proto-corsol*/
!
!     SHELL, RDFSOL, PROTO and CORSOL use this function
!
SUBROUTINE SHLSEL(ST,STLEN,LIST,NLIST,WFLAGS)
  !
  !     fills LIST with all atoms (NLIST in total) selected by the first
  !     atom selection in string ST of lenght STLEN.
  !     WFLAGS is a work array
  !
  !     Tibor Rudas Oct 2002 - Nov 2002
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use coord
  use psf
  use select

  implicit none
  !
  character(len=*) ST
  INTEGER STLEN,LIST(*),NLIST,WFLAGS(*),NA
  !
  !
  INTEGER I
  !
  !     select & fill the WFLAGS array 
  CALL SELCTA(ST,STLEN,WFLAGS,X,Y,Z,WMAIN,.TRUE.)
  !
  !     fill the numbers of the selected atoms into LIST
  NLIST=0
  DO I=1,NATOM
     IF(WFLAGS(I).EQ.1) THEN
        NLIST=NLIST+1
        LIST(NLIST)=I
     ENDIF
  ENDDO
  !
  RETURN
END SUBROUTINE SHLSEL
!
#endif /* (shlsel-rdfsol-proto-corsol)*/
!----------------------------------------------------------------------
!
SUBROUTINE GWDIP(N1,X,Y,Z,CH,XD,YD,ZD,LEND,QNORM)
  !
  !     this routine computes the dipole of water N1
  !     The components of the dipole UNIT vector are returned in
  !     XD, YD and ZD. LEND is the length of the vector.
  !     (this breaks the genericness of CDIPOLE but this should be
  !      much more efficient if we _know_ it is TIP3 water...)
  !
  !     Tibor Rudas Jan 2003 - Jun 2003
  !
  use chm_kinds
  use consta
  use dimens_fcm
  use exfunc
  use number
  implicit none
  !
  INTEGER N1
  real(chm_real) X(*),Y(*),Z(*),CH(*)
  real(chm_real) XD,YD,ZD,LEND
  LOGICAL QNORM
  !
  INTEGER I,K
  !
#if KEY_RDFSOL==1 || KEY_CORSOL==1 /*rdfsol-corsol*/
  !
  XD=ZERO
  YD=ZERO
  ZD=ZERO
  !
  DO I=1,3
     K=N1+I-1
     XD=XD+(X(K)*CH(K))
     YD=YD+(Y(K)*CH(K))
     ZD=ZD+(Z(K)*CH(K))
  ENDDO
  XD=XD*DEBYEC
  YD=YD*DEBYEC
  ZD=ZD*DEBYEC
  !
  !     calc lenght and unit vector
  LEND=(XD*XD)+(YD*YD)+(ZD*ZD)
  LEND=DSQRT(LEND)
  IF(QNORM) THEN
     !        Normalize only if requested
     XD=XD/LEND
     YD=YD/LEND
     ZD=ZD/LEND
  ENDIF
  !
  return
#endif /* (rdfsol-corsol)*/
end SUBROUTINE GWDIP
subroutine shlsel_dummy()
  RETURN
end subroutine shlsel_dummy

