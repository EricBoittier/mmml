module ensemble
  use chm_kinds
  use dimens_fcm
! stuff for ensemble code
! description of important vbls:
!
! General ENSEMBLE variables
! --------------------------
! WHOIAM: number of each node according to MPI
! NENSEM: number of copies started by MPI
! MAXENS: A fixed(!) maximum for NENSEM
! ENSBUF, ENSBFL: buffer, buffer length for MPI broadcast of
!       ENSEMBLE messages from each node.
!
! Replica exchange variables
! --------------------------
! T2REP, REP2P: map of temp. number to replica number, inverse
! T2REPO: output unit for map of temp. number to replica number
! REP2TO: output for inverse of above map.
! ENSFRQ: frequency in dynamics steps to attempt replica swapping
! FLIPFLOP: tells ensemble which replicas we are trying to
!           swap each time
! ENSATT: statistics of attempted swaps
! ENSSUC: statistics of successful swaps
! JREX: logical to tell whether replicas have been set up for
!       replica exchange.
! JRSWAP: is replica exchange actually turned on at the moment?
! ENSTEM: list of replica temperatures sorted in ascending order
! ENSDB: lookup table of beta_i-beta_j factors
! ENSMYT: temperature of present replica 
! ensswmethod: 0 = try one swap per try (default)
!               1 = try to swap all temperature pairs
!
! the following are for code which is still in testing...
! ENSDX,ENSDY,ENSDZ,ENSH,ENSEXPU,QPARPRM,QENSEXP,SWAPC,JENSC,
! ENSEBETA, EXPOFF
!
#if KEY_ENSEMBLE==1
  real(chm_real),allocatable,dimension(:) :: ensbuf
  real(chm_real),allocatable,dimension(:) :: ensdx
  real(chm_real),allocatable,dimension(:) :: ensdy
  real(chm_real),allocatable,dimension(:) :: ensdz
  real(chm_real),allocatable,dimension(:) :: ensh
  integer whoiam,nensem,t2repo,rep2to
  integer ensemble_layers,current_ens_layer
  integer,parameter :: maxensemble_layers=1
  integer,dimension(maxensemble_layers) :: comm_ensemble,comm_ens_index
  integer swbuf
  integer ensfrq,ensnsw
  integer ensexpu
  integer comm_master
  logical slave_ens
  integer,parameter :: maxens=4096, ensbfl=120
  integer,parameter :: maxswp=(maxens+1)*maxens/2
  integer t2rep(maxens),rep2t(maxens),ensatt(maxswp),enssuc(maxswp)
  integer ensisw(maxswp),ensjsw(maxswp)
  integer repswp(maxens)
  logical jrex,jrswap,jensc,qparprm,qensexp,swapc,qexpbw
  integer ensswmethod
  logical ensas,lmasternode,mastermaster
  real(chm_real) enstem(maxens),ensdb(maxens),ensmyt,ensebeta,expoff(maxens)
  integer,save :: old_mynod,ensmasternod
  logical,save :: ensemble_verbose

contains

  subroutine ensprint(a1,a2)
    use parallel,only:mynod
    character(len=*),intent(in) :: a1,a2
    character(len=80) :: fmt
    character(len=80) :: filnam

    if(.not. ensemble_verbose) return
    write(filnam,'("prn",i3.3)')old_mynod
    open(action="write",file=filnam,unit=80,position="APPEND")

    fmt='(a," e",i2," n",i2," w",i2," >>> ",a)'
    write(80,fmt) a1,whoiam,mynod,old_mynod,a2
    close(unit=80)
    return
  end subroutine ensprint
#endif 
end module ensemble

