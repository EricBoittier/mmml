! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! TSP.MOD -- solution of traveling salesman problem by Monte-Carlo simulated annealing
!=====================================================================================
!
module tsp
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
 use stream
 use number
 use clcg_mod, only: random; use reawri, only: iseed
 implicit none
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
 private
!
 public tsp_anneal_path
 private shiftpath
 public tsp_path_len

!======================================================================================
 contains
!
  function tsp_anneal_path(D, route, nmove, itemp_, ftemp_, dtemp_, gettemp_) result(newroute)
!
!
!
  real(chm_real), intent(in) :: D(:,:) ! matrix of distances between all vertices
  integer, intent(in) :: route(:) ! initial route
  real(chm_real), optional, intent(inout) :: itemp_, ftemp_, dtemp_ ! initial and final temperatures for annealing, and the temperature decrement
  logical, optional, intent(in) :: gettemp_ ! flag that indicates whether to estimate initial temperature
  integer, intent(in) :: nmove ! number of Monte Carlo moves
!
  integer, pointer :: newroute(:)
  real(chm_real), parameter :: mintemp = 0.01d0 ; ! default minimum temperature
  integer, parameter :: gettemp_niter = 1000 ! number of iterations to guess initial temperature
  real(chm_real), parameter :: gettemp_defrac=0.2d0 ! fraction of energy difference for computing temperature
!
! locals
  character(len=len("TSP_ANNEAL_PATH>") ),parameter::whoami="TSP_ANNEAL_PATH>";!macro
  integer :: i,n
  real(chm_real) :: itemp, ftemp, dtemp, temp
  logical :: gettemp
  integer, pointer, dimension(:) :: minroute, r ! minimum path, temporary path
  real(chm_real) :: mind, maxd, dist, newd, dd
  integer :: ind1, ind2
!
!
  nullify(newroute)
!==== consistency checks
  n=size(route)
  if (n.lt.4) then
   call wrndie(0,whoami,trim('NUMBER OF VERTICES MUST BE AT LEAST FOUR. NOTHING DONE.'))
   return
  elseif ( size(D,1).ne.n .or. size(D,2).ne.n ) then
   call wrndie(0,whoami,trim('DISTANCE MATRIX MUST BE SQUARE WITH SIZE EQUAL TO THE NUMBER OF VERTICES. NOTHING DONE.'))
   return
  elseif (nmove .lt. 0) then
   call wrndie(0,whoami,trim('NUMBER OF MONTE CARLO MOVES CANNOT BE NEGATIVE. NOTHING DONE.'))
   return
  endif
! see if we have sufficient information to determine temperature schedule
!
  if (present(gettemp_)) then
   gettemp=gettemp_
   if (.not.gettemp) then
    if ( .not. present(itemp_) ) then
     call wrndie(0,whoami,trim('MUST SPECIFY INITIAL TEMPERATURE OR SET "GETTEMP". NOTHING DONE.'));
     return
    elseif (itemp_.eq.-abs(anum)) then
     gettemp=.true.
    elseif (itemp_.lt.0) then
      call wrndie(0,whoami,trim('INITIAL TEMPERATURE CANNOT BE LESS THAN ZERO. NOTHING DONE.'));
      return
    else
     itemp=itemp_
    endif
   endif ! .not. gettemp
  else ! gettemp is not supplied
! check if itemp is supplied, if not, set gettemp tp true
   if (present(itemp_)) then
    if (itemp_.lt.0) then
      call wrndie(0,whoami,trim('INITIAL TEMPERATURE CANNOT BE LESS THAN ZERO. NOTHING DONE.'));
      return
    endif
    gettemp=.false.
   else ! itemp not supplied
    gettemp=.true.
   endif ! present (itemp)
  endif ! present(gettemp)
!=========================== now, if gettemp is set, get the initial temperature
  allocate(r(n)); r=route; ! original route
!
  dist=zero; do i=2,n; dist=dist+(D(r(i-1),r(i)));;enddo
  if (gettemp) then
   mind=dist
   maxd=dist
   do i=1, gettemp_niter
    ind1=1 + ( INT(random(iseed)*(n-2)) + 1 ); ind2=1 + ( INT(random(iseed)*(n-3)) + 1 ); if (ind2.ge.ind1) ind2=ind2+(1);
    dd= (D(r(ind2),r(ind1-1))+D(r(ind2),r(ind1+1))+D(r(ind1),r(ind2-1))+D(r(ind1),r(ind2+1)))-(D(r(ind1),r(ind1-1))+D(r(ind1),r(ind1+1))+D(r(ind2),r(ind2-1))+D(r(ind2),r(ind2+1)))+max(0,min(1,2-abs(ind1-ind2)))*two*D(r(ind1),r(ind2))
    newd=dist+dd
    if (newd<mind) then ; mind=newd; elseif (newd>maxd) then ; maxd=newd; endif
   enddo
   itemp = (maxd-mind)*gettemp_defrac
!
   if (present(itemp_)) itemp_=itemp ! pass itemp to calling subroutine
  endif ! gettemp
!
!================== ftemp
  if (present(ftemp_)) then
   if(ftemp_.lt.zero) then
! special case : -1 means determine automatically
    if (ftemp_.eq.-abs(anum)) then
     ftemp_=mintemp
    else
     call wrndie(0,whoami,trim('FINAL TEMPERATURE CANNOT BE LESS THAN ZERO. NOTHING DONE.'));
     return;
    endif
   elseif (ftemp_.gt.itemp) then
    call wrndie(0,whoami,trim('FINAL TEMPERATURE IS GREATER THAN INITIAL TEMPERATURE.'))
   endif
   ftemp=ftemp_
  else
   ftemp=mintemp
  endif ! present(ftemp)
! dtemp :
  if (present(dtemp_)) then
   if (dtemp_.eq.-abs(anum)) then ! special case -- assign automatically
    dtemp_=(itemp - ftemp) / nmove * 1.2d0
   elseif ( dtemp_*nmove .lt. (itemp-ftemp)) then;
    call wrndie(0,whoami,trim('NOT ENOUGH ITERATIONS TO ANNEAL TO ZERO FOR GIVEN DeltaT'));
   endif
   dtemp=dtemp_
  else
   dtemp=(itemp - ftemp) / nmove * 1.2d0
  endif
!=========================================================
! perform annealing
  allocate(minroute(n))
  r=route; ! reset iroute if it was modified to compute temperature
  minroute=r;
  mind=dist;
!
  temp=itemp;
  do i=1,nmove
   if (mod(i,2).eq.0) then
     call shiftpath(D,n,r, ind1, ind2)
   else
     ind1=1 + ( INT(random(iseed)*(n-2)) + 1 ); ind2=1 + ( INT(random(iseed)*(n-3)) + 1 ); if (ind2.ge.ind1) ind2=ind2+(1);
   endif
   dd= (D(r(ind2),r(ind1-1))+D(r(ind2),r(ind1+1))+D(r(ind1),r(ind2-1))+D(r(ind1),r(ind2+1)))-(D(r(ind1),r(ind1-1))+D(r(ind1),r(ind1+1))+D(r(ind2),r(ind2-1))+D(r(ind2),r(ind2+1)))+max(0,min(1,2-abs(ind1-ind2)))*two*D(r(ind1),r(ind2))
!
   if ( dd .le. zero ) then
    r(ind1)=ieor(r(ind1),r(ind2)) ; r(ind2)=ieor(r(ind1),r(ind2)) ; r(ind1)=ieor(r(ind1),r(ind2)) ! swap vertices
    dist=dist+(dd);;
    if (dist.lt.mind) then; mind=dist; minroute=r; endif
!
   elseif ( exp ( - dd/temp ) .gt. random(iseed) ) then ! Metropolis criterion
    r(ind1)=ieor(r(ind1),r(ind2)) ; r(ind2)=ieor(r(ind1),r(ind2)) ; r(ind1)=ieor(r(ind1),r(ind2)) ! swap vertices
    dist=dist+(dd);;
    if (dist.lt.mind) then; mind=dist; minroute=r; endif
   endif
! change temperature (annealing)
   temp=max(temp-dtemp,ftemp)
  enddo ! nmove
!
  if(associated(r))deallocate(r)
  newroute=>minroute; nullify(minroute);
!
  end function tsp_anneal_path
!========================================================
 subroutine shiftpath(D,n,r,indi,indk)
  real(chm_real), intent(in) :: D(n,n)
  integer, intent(in) :: n, r(n)
  integer, intent(out) :: indi, indk
!
  integer :: rinv(n) ! inverse map
  integer :: idir, indj, ind1, ind2, i, indk1(1)
  real(chm_real) :: dmax, Di(n)
!
  indi=0 ; indk=0 ;
  if (n.lt.4) return;
!
  rinv=0; do i=1,n ; rinv(r(i))=i ; enddo ! inverse map
!
! choose random vertex between 2 : n-1
  indi=1 + ( INT(random(iseed)*(n-2)) + 1 )
! choose an adjacent vertex randomly, which will define an edge
  idir= INT(random(iseed)*2)*2-1
  indj= indi + idir ; ! (indi, indj) correspond to an edge
! get absolute vertex indices
  ind1=r(indi);
  ind2=r(indj);
!find a vertex that is closest to indj and that is not indi
  Di=D(ind2,:)
  dmax=maxval(Di) + one
  dmax=maxval(Di) + two
  Di(ind2)=dmax-one ; Di(ind1)=dmax ; Di(r(1))=dmax ; Di(r(n))=dmax ; ! exclude corresponding indices from minimization below
                                                                        ! BUT if all lengths are equal choose ind2 (only safe option)
  indk1 = rinv(minloc(Di)); ! index in r of the vertex that is closest to the vertex r(indi)
  indk=indk1(1) ! minloc needs an array
!
 end subroutine shiftpath
!========================================================
 function tsp_path_len(route,D,n) result(dist)
 integer :: n, i
 integer, intent(in) :: route(n)
 real(chm_real), intent(in) :: D(n,n)
 real(chm_real) :: dist
  dist=zero; do i=2,n; dist=dist+(D(route(i-1),route(i)));;enddo
 end function tsp_path_len
!========================================================
#endif /* automatically protect all code */
end module tsp
