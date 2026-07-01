module icfix
  use chm_kinds
  use dimens_fcm

#if KEY_TSM==1
  !     Information for internal coordinate fixed constraints
  !
  !     Purpose: to store the information needed to do ic fixed constraints
  !
  !     Variable    Purpose
  !
  !     MAXICF      Maximum number of ic constraints
  !     NICF        Number of ic constraints in use
  !     IICF        Flag; 0 => NICF == 0, 1 => NICF > 0
  !     ICFTYP      Type of ic: 1 => distance, 2 => angle, 3 => dihedral
  !     ICFATN      Matrix of the atom numbers of the atoms involved
  !                 in ic constraints
  !     SVAL        Initial value of constrained ic
  !     DS          Vector of differences of refence and actual ic's
  !     SMF         Matrix of Wilson s vectors at t = h (molecule-fixed axes)
  !     GCOEF       Sum of mass-weighted projections, SMF*SMF
  !     ICFADJ      Array of logicals indicating whether or not the i.c.'s
  !                 have been adjusted
  !     ANYADJ      Logical indicating whether or not any i.c. was adjusted
  !     MXLENS      Maximum length of s vector araays (4*MAXICF)
  !     TOLI        Tolerances for ic resetting
  !     MAXI        Maximum number of iterations for ic resetting
  !
  INTEGER,PARAMETER :: MAXICF=50,MXLENS=4*MAXICF
  INTEGER NICF,IICF,ICFTYP(maxicf),ICFATN(4,maxicf),MAXI
  real(chm_real) :: SVAL(maxicf),TOLI(maxicf),SMF(3,mxlens),GCOEF(maxicf),DS(maxicf)
  LOGICAL ICFADJ(maxicf),ANYADJ

#endif 
  !
end module icfix

