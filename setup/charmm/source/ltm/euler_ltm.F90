module euler
  use chm_kinds
  use dimens_fcm
  implicit none
  !CHARMM Element source/fcm/euler.fcm 1.1
  !
  !     Arrays and variables for Euler minimization and dynamics
  !
  !     Variables:
  !
  !     QEULER      Do Langevin/Implicit Euler dynamics
  !     QEHARM      Include Euler restraints in subsequent energy calls
  !
  !     KEHARM      Pointer to force constant array
  !     RXHARM      Pointer to X reference array
  !     RYHARM      Pointer to Y reference array
  !     RZHARM      Pointer to Z reference array
  !     IHHARM      Atom array of ones for restrain set number
  !
  !     JCALDY      Pointer to the indice of the step 
  !     NZH         estimate of the number of nonzeros of the Hessian.
  !
  !     QXN         OPTION FOR THE CANDIDATES:
  !                 IF (TRUE)  X = XN
  !                 IF (FALSE) X = XNo + e R(N+1)
  !
#if KEY_TNPACK==1 /*eulerfcm*/
   real(chm_real),allocatable,dimension(:) :: KEHARM
   real(chm_real),allocatable,dimension(:) :: RXHARM
   real(chm_real),allocatable,dimension(:) :: RYHARM
   real(chm_real),allocatable,dimension(:) :: RZHARM
   integer,allocatable,dimension(:) :: IHHARM
  integer,allocatable,dimension(:) :: IAHES
  integer,allocatable,dimension(:) :: JAHES
  real(chm_real),allocatable,dimension(:) :: AHES

  LOGICAL QEULER, QEHARM, QESTRT, QXN, QLOC1
  INTEGER JCALDY, INUDYN
  real(chm_real)  LIEFF, GNOMI
  INTEGER IEULER
!  INTEGER JAHES, IAHES, AHES, SIZE1, NZHE, IEULER
  
  integer,PARAMETER :: SIZE1 = 18361, NZHE = 3000000 !cutnb 12.0

contains
  subroutine euler_init()
    use number,only:one
    qeuler=.false.
    qeharm=.false.
    lieff=one
    return
  end subroutine euler_init

#endif /* (eulerfcm)*/
end module euler

