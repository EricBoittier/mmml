module dimb
  use chm_kinds
  use dimens_fcm
  implicit none
  !
  !     Diagonalization using DIMB algorithm
  !
  !     Variable    Purpose
  !
  !     MNBCMP      maximum number of interactions for a given atom
  !     QDW         logical flag for double windowing
  !     PINBCM      pointer into JNBCMP for pair list interactions
  !     PJNBCM      second atoms of interacting pairs
  !     LENCMP      maximum number of elements in JNBCMP
  !     LENDSK      maximum number of ij pairs in DD1CMP when LDISK is .true.
  !     PDD1CM      compact second derivatives of the potential energy
  !
  !     ITER        ?
  !     IPAR1       ?
  !     IPAR2       ?
  !     NFSAV       ?
  !     QDISK       ?
  !     QCMPCT      ?
  !
#if KEY_DIMB==1 /*dimbfcm*/
  INTEGER,PARAMETER :: NPARMX=1000,MNBCMP=300,LENDSK=200000

  integer,PARAMETER :: IJXXCM=1,IJXYCM=2,IJXZCM=3,IJYXCM=4,IJYYCM=5
  integer,PARAMETER :: IJYZCM=6,IJZXCM=7,IJZYCM=8,IJZZCM=9
  integer,PARAMETER :: IIXXCM=1,IIXYCM=2,IIXZCM=3,IIYYCM=4
  integer,PARAMETER :: IIYZCM=5,IIZZCM=6
  integer,PARAMETER :: JJXXCM=1,JJXYCM=2,JJXZCM=3,JJYYCM=4
  integer,PARAMETER :: JJYZCM=5,JJZZCM=6

  INTEGER ITER,IPAR1,IPAR2,NFSAV,LENCMP
  integer,allocatable,dimension(:) :: PINBCM, PJNBCM
  real(chm_real),allocatable,dimension(:) :: PDD1CM
  LOGICAL QDISK,QDW,QCMPCT

contains

  subroutine dimb_init()
    !=======================================================================
    ! DIMB.FCM
    qdisk=.false.
    qdw=.false.
    qcmpct=.false.
    return
  end subroutine dimb_init

#endif /* (dimbfcm)*/
end module dimb

