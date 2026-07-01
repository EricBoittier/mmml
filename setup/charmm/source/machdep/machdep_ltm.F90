module machdep
  use chm_kinds
  use dimens_fcm
  !CHARMM Element source/fcm/machdep.fcm 1.1
  !
  ! MACHDEP.FCM:  machine dependent parallel processing information
  !
  !     maximum number of cpus
  !     minvec is the minimum vector length (scalar/vector breakeven point)

  INTEGER, PARAMETER :: MAXCPU=16,MINVEC=6

  !     number of bytes per element of temporary jnbt array.
  !     for Convex - integer*4
  integer,PARAMETER :: NBYTES=4
  !     maxjnbt: storage used for temporary jnbt arrays.

  INTEGER NCPU,OLDCPU,MAXJNBT    !,JNBT(2,MAXCPU)
  real(chm_real) NBFACT

  ! QFASTNB: logical if FASTER >0; use fast non-bond if true else don't
  ! QFLUSH : flush output, restart files, and trajectories if .true.

  LOGICAL QFASTNB,QFLUSH

  !
end module machdep

