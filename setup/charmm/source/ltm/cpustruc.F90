  module cpustruc
  use chm_kinds
  use chm_types
  implicit none
!NUMHOOD number of cpu neighborhoods
!MYNHOOD integer corresponding to neighborhood (group of cpus)
!IMYNHDP  MYNHOOD + 1
!HOODCPULST  !list of cpus in each neighborhood
!HOODCPUHI
!HOODCPUMNY

!communicators:
!HOODCOM communicator for all cpus within a neighborhood
!HEADCOM communicator for all head nodes in the system (inter-neighborhood communicator)
!BASECOM communicator for base (non-head) nodes within a neighborhood
!
!NEIGHMAPP(NUMNOD) map of original ranks (MPI_COMM_WORLD) to neighborhoods (both indexed from 1)
!LOC2GLBP !map of local (neighborhood) rankp to global rankp
!HEADRANK(NUMHOOD)  map of neighborhoods-> head node rank (MPI_COMM_WORLD rank of head nodes)
!MYKEY is rank within local group (neighborhood)
!MYKEYP = MYKEY + 1
!IMYKEYP = MYKEYP, integer (not int4)
!MYKEYB is rank within base communicator
!MYKEYBP = MYKEYB + 1
!IMYKEYBP = MYKEYBP, integer (not int4)
!NUMNODH = # nodes in neighborhood
!RNUMNODH real type
!NUMNODHI4 int4 type
! QCPUONES  true if CPUONES,CPUIDENT are set up
 
  integer,allocatable,dimension(:),save :: OTHERCPUS  !node map array (global rank of all other cpus)
      integer,allocatable,dimension(:),save :: CPUIDENT  !node map array (= cpu)
      integer,allocatable,dimension(:),save :: CPUONES  !node map array (= 1)
      integer,allocatable,dimension(:),save :: OTHERCPU_LN  !global rank of other cpus in my neighborhood
      integer,allocatable,dimension(:),target,save :: OTHERCPU_HD  !neighborhood rank of other cpus in my neighborhood
      integer,allocatable,dimension(:),target,save :: RNDOTHMP_HD
      integer,allocatable,dimension(:),target,save ::NEIGHMAPP 
 integer,allocatable,dimension(:),save :: LOC2GLBP !map of local (neighborhood) rankp to global rankp
      character(len=7),save :: cellordtyp  !name of order type
      logical,save :: QWRSPACMAP=.false.  !if writing out spatial map of neighborhood/cells
      integer,allocatable,dimension(:),save :: HOODCPULST,HOODCPUHI,HOODCPUMNY
      integer(chm_int4),allocatable,dimension(:), save:: headrank
      REAL(chm_real),save :: RNUMNODH
      integer(chm_int4),save :: MYNHOOD,MYKEY,MYKEYP,HOODCOM,HEADCOM,BASECOM,NUMNODHI4, &
       MYKEYB,MYKEYBP
      integer,save :: NUMHOOD=0,IMYKEYP=0,IMYNHDP=0,NUMNODH=0,IMYKEYBP=0
      integer,save ::  TSNTRIAL,TSNATOMX
      logical,save :: TSQACTUAL,QCPUONES=.false.
  end module cpustruc

