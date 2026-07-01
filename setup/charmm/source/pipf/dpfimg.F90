#if KEY_PIPF==1 /*pipf_main*/
SUBROUTINE DPFIMG(IFRSTAPP,NATOMPP,JNBPP,INBLOPP,  &
     IFRSTAIP,NATOMIP,JNBIP,INBLOIP,IMATTR,CG, &
     MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
     LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
     TXWWPP,TXWWIP,DMUIND,PFCTOF, &
     ALP,DPFAC,NPDAMP,QPFEX &
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES POLARIZATION ENERGIES AND FORCES
  !
  !     IFRSTA - first atom to look at in INBLO
  !     NATOM  - last atom to look at in INBLO (number of atoms)
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     MAXROW  - offset for 1-4 interaction 
  !     LELEC - logical flags used in BNBND.FCM
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     ALP - polarizability
  !     DPFAC - damping factor
  !     NPDAMP - damping option
  !     QPFEX - tag to exclude 1-4 polarization
  !----------------------------------------------------------------------
  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use exfunc
  use energym
#if KEY_PBOUND==1
  use pbound     
#endif

  implicit none
#if KEY_PBOUND==1
  real(chm_real) CORR     
#endif

  !
  INTEGER IFRSTAPP,NATOMPP,IFRSTAIP,NATOMIP
  INTEGER JNBPP(*),JNBIP(*)
  INTEGER INBLOPP(*),INBLOIP(*)
  INTEGER IMATTR(*)
  real(chm_real) CG(*)
  INTEGER MAXROW
  INTEGER IAC(*),ITC(*)
  !
  !     INTEGER IOFF(*)
  !
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) CTONNB,CTOFNB,EPS,E14FAC
  real(chm_real) DMUIND(3,*),TXWWPP(6,*),TXWWIP(6,*)
  !
  INTEGER I,J,K,NB,ITEMP,JPR,NPR
  INTEGER I1,J1,IC,IACI
  real(chm_real) R1,R2,R5,DXI,DYI,DZI,CGI,CGJ,DPR1,DPR2,DPR3
  real(chm_real) TXXX,TYXX,TYYX,TZXX,TZYX,TZZX,TYYY,TZYY,TZZY,TZZZ
  real(chm_real) E14M1,E14F,CGT,CGT2
  !
  real(chm_real) ALP(*),DPFAC,U3,AU3,FLMD5,FLMD7,R3
  INTEGER NPDAMP
  !
  real(chm_real) RIJ,PFCTOF
  !
  LOGICAL QPFEX

  ! DO BLOCK EXPANSION OF CODE TO IMPROVE EFFICIENCY
  IF(PFCTOF.LE.0.0D0 .AND. NPDAMP.EQ.0) THEN
#undef PIPF_CTOF
#undef PIPF_DAMP
#include "dpfimg.inc"
  ELSE IF(PFCTOF.GT.0.0D0 .AND. NPDAMP.EQ.0) THEN
#define PIPF_CTOF 1
#include "dpfimg.inc"
#undef PIPF_CTOF
  ELSE IF(PFCTOF.LE.0.0D0 .AND. NPDAMP.EQ.1) THEN
#define PIPF_DAMP 1
#include "dpfimg.inc"
  ELSE
#define PIPF_CTOF 1
#include "dpfimg.inc"
#undef PIPF_CTOF
#undef PIPF_DAMP
  ENDIF
  RETURN
END SUBROUTINE DPFIMG

#endif /* (pipf_main)*/
subroutine dpfimf_dummy()
  return
end subroutine dpfimf_dummy

