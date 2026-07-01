! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
! Plastic Network Model.
!
! Author: Jingzhi Pu & Paul Maragakis
!
! Date: Fri Oct 31 15:02:20 EST 2003
! Date: Thu Nov 6 15:23:49 EST 2003
! Date: Wed Dec 3, 2003 (increased maxcon to 150)
! Date: Fri Apr 2, 2004 (consistency sweep: fortran indexing)
! Date: June 2007 (interfaced to CHARMM)
! overhauled by VO in Aug. 2013 :! bug fixes, parallelization, generalization to an arbitrary number of networks,
! multiple simultaneous plastic networks (e.g. corrsponding to different proteins/domains); NOTE : parallelization may not help in some cases
!---------------------------------------------------------------
module pnm
!
#if KEY_PNM==1 /* (pnm_main) */
  use chm_kinds
  use number
  use consta
!







#if KEY_STRINGM==1 /*  VO can use either string method routines or local routines (below) */
  use ivector
  use rvector
#endif

!
  implicit none
  private
!


#if KEY_STRINGM==0 /*  VO: otherwise can use string method aux subroutines */

!========================= auxiliary vector types
!====================================================
      type int_vector
       integer, dimension(:), pointer :: i
       integer :: length ! length of the vector
       integer :: last ! index of last element
       logical :: initialized=.false. ! has the vector been initialized
!
      end type int_vector
!====================================================
      type real_vector
       real(chm_real), dimension(:), pointer :: r
       integer :: length ! length of the vector
       integer :: last ! index of last element
       logical :: initialized=.false. ! has the vector been initialized
!
      end type real_vector
      integer, parameter, private :: expand_incr=50
#endif
!========================= end auxiliary vector types
!
  type enet ! elastic network type
   type(int_vector) :: nodes ! pnm node list (no connectivity)
   type(int_vector) :: bonds ! pnm bond list (interacting pairs in tandem)
   type(real_vector) :: x ! x-coordinate
   type(real_vector) :: y ! y-coordinate
   type(real_vector) :: z ! z-coordinate
   type(real_vector) :: fx ! gradient of energy w.r.t x
   type(real_vector) :: fy ! y
   type(real_vector) :: fz ! z
   type(real_vector) :: econt ! energy decomposition array
   type(real_vector) :: r0 ! equilibrium bond lengths (single list corresponding to bonds)
   real(chm_real) :: k ! force constant
   real(chm_real) :: kb ! bonded force constant
   real(chm_real) :: knb ! nonbonded force constant
   real(chm_real) :: cutoff ! cutoff for network
   real(chm_real) :: emin ! energy value at minimum
   real(chm_real) :: eps ! mixing constant
   logical :: initialized
  end type enet
!
  logical :: qpnm ! pnm active
!
  type(enet), private, save, pointer :: networks(:) ! array of elastic network type
  integer :: num_enm ! number of networks
  integer :: max_num_enm ! max number of networks : allocated at initialization
  logical, private :: initialized=.false. ! whether the module has been initialized (possibly with no networks)
  logical, public :: calc_para=.true. ! whether to calculate forces in parallel
  real(chm_real), private, save, pointer :: pnmene(:,:)=>null() ! matrix of pairwise mixing coefficeints
!
  integer, private, save, pointer :: models(:)=>null() ! index of separate pnm models into the networks array
  logical, private, save, pointer :: qexp(:)=>null() ! array of flags indicating whether the exponential PNM version is used
  real(chm_real), private, save, pointer :: beta(:)=>null() ! array of PNM temperatures (only for the exponential version)
  integer, private :: num_models ! number of pnm models
!
!
  character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
  public pnm_main
  public pnm_add
  public pnm_ene
  contains
!===================================================================
   subroutine pnm_main(comlyn,comlen) ! parser
   use string
   use stream
!
   character(len=*) :: comlyn
   integer :: comlen
! local variables
   character(len=8) :: keyword
   character(len=len("PNM_MAIN>") ),parameter::whoami="PNM_MAIN>";!macro
   integer :: i, j
   real(chm_real) :: t
!
!
   keyword=nexta8(comlyn,comlen)
!====================================================================
   if (( keyword(1:4).eq.'DONE'(1:4) )) then
     call pnm_done()
!====================================================================
   elseif (( keyword(1:4).eq.'INIT'(1:4) )) then
     keyword=nexta8(comlyn,comlen)
     i=-1
     j=len(keyword)
     call trima(keyword, j)
     if (j.gt.0) i=decodi(keyword, j)
     if (i.le.0) then
      i=2
     write(info,600) whoami,i ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
 600 format(/A,' VALID NUMBER OF PNM NETWORKS NOT SPECIFIED. USING ',I6)
     endif ! i<=2
     call pnm_init(i)
!====================================================================
   elseif ( ( keyword(1:3).eq.'ADD'(1:3) )) then
     call pnm_add(comlyn, comlen) ! add new rtmd restraint
!====================================================================
! specify parallel calculation options (lifted from string code)
   elseif ( ( keyword(1:4).eq.'PARA'(1:4) )) then
     keyword=nexta8(comlyn,comlen)
     select case(keyword)
       case('YES','ON','TRUE','T','yes','on','true','t')
        keyword='ENABLED '; calc_para=.true.
        write(info,7009) whoami, keyword ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
       case('NO','OFF','FALSE','F','no','off','false','f')
        keyword='DISABLED' ; calc_para=.false.
        write(info,7009) whoami, keyword ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
       case default
        call wrndie(0,whoami,trim('UNKNOWN OPTION SPECIFIED'))
     end select
 7009 format(' ',A,':', ' PARALLEL COMPUTATION OF FORCES ',A)
!====================================================================
! specify exponential version
   elseif (( keyword(1:3).eq.'EXP'(1:3) )) then
     if (.not.initialized) call pnm_init() ! default initialization
     keyword=nexta8(comlyn,comlen)
     select case(keyword)
       case('YES','ON','TRUE','T','yes','on','true','t')
        qexp(num_models)=.true.
        t=gtrmf(COMLYN, COMLEN, 'TEMP', 300d0)
        if (t.le.RSMALL) then
         call wrndie(0,whoami,trim(' PNM TEMPERATURE MUST BE GREATER THAN ZERO. ABORT.'))
         return
        else
         beta(num_models)=one/(kboltz*t)
        endif
        write(info,7010) whoami, itoa(num_models), ftoa(t); if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
       case('NO','OFF','FALSE','F','no','off','false','f')
        qexp(num_models)=.false.
        write(info,7011) whoami, itoa(num_models) ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
       case default
        call wrndie(0,whoami,trim('NO VALID "EXP" OPTION SPECIFIED'))
     end select
 7010 format(' ',A,': MODEL #',A,' WILL USE EXPONENTIAL PNM WITH T=',A,'K.')
 7011 format(' ',A,': MODEL #',A,' WILL USE STANDARD PNM.')
!====================================================================
   elseif ( ( keyword(1:4).eq.'NEWM'(1:4) )) then ! new model
    if (.not.initialized) call pnm_init() ! default initialization
    ! can only add a new model if the present model has at least one network !
    if ( num_enm .ge. models(num_models) ) then
! make sure there will be enough room to add network(s) to the model
     if (num_enm.ge.max_num_enm) then
       call wrndie(0,whoami,trim(' MAXIMUM NUMBER OF ALLOWED NETWORKS EXCEEDED (REINITIALIZE).'))
     else
       num_models=num_models+1
       models(num_models)=num_enm+1
       write(info,601) whoami, itoa(num_models) ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
 601 format(' ',A,': WILL CREATE A NEW PNM MODEL (#',A,').')
     endif ! num_enm > max
    endif ! initialized
!====================================================================
   elseif ( ( keyword(1:4).eq.'HELP'(1:4) )) then ! print short help screen
 write(info,'(2A)')&
& whoami, ' _______________________________________________________________________'&
& , whoami, ' DESCRIPTION: Plastic Network Model'&
& , whoami, ' _______________________________________________________________________'&
& , whoami, ' SYNTAX:'&
& , whoami, ' PNM [{ INITialize int }] | // initialize with the specified maximum total no. of ENMs'&
& , whoami, '     [{ DONE }]           | // finalize'&
& , whoami, '     [{ NEWModel }]       | // start a new PNM'&
& , whoami, '     [{ EXP <on|true|t|yes|off|false|f|no> [TEMP real] }]| //turn on/off exponential version and set temperature'&
& , whoami, '     [{ ADD [FORC real  // Add ENM to current PNM ; ENM force constant'&
& , whoami, '            [CUT real]  // ENM cutoff (Ang)'&
& , whoami, '            [ZERO real] // ENM minimum energy value'&
& , whoami, '            [PMIX real] // ENM mixing coefficient'&
& , whoami, '            [atom-selection] // ENM atom selection'&
& , whoami, '            [REMO atom-selection atom-selection] //ENM : interactions between these selections are off'&
& , whoami, '            [COMP] }]     | ENM : take reference structure from comparison set// '&
& , whoami, '            [ { PARA <on|true|t|yes|off|false|f|no> } ] // parallelization on/off'&
& , whoami, ' _______________________________________________________________________'
 if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';;
!====================================================================
   else
    write(info(1),*)'UNRECOGNIZED SUBCOMMAND: ',keyword;call wrndie(0,whoami,trim(info(1)))
   endif
!
   end subroutine pnm_main
!====================================================================
   subroutine pnm_done()
   use stream
!====================================================================
   character(len=len("PNM_DONE>") ),parameter::whoami="PNM_DONE>";!macro
   integer :: i
   type(enet), pointer :: enm
!
   if (.not.initialized) return
!
   if(associated(pnmene))deallocate(pnmene)
   if(associated(models))deallocate(models)
   if(associated(qexp))deallocate(qexp)
   if(associated(beta))deallocate(beta)
!
   do i=1, num_enm ! over all networks
    enm=>networks(i)
    call int_vector_done(enm%nodes)
    call int_vector_done(enm%bonds)
    call real_vector_done(enm%x)
    call real_vector_done(enm%y)
    call real_vector_done(enm%z)
    call real_vector_done(enm%fx)
    call real_vector_done(enm%fy)
    call real_vector_done(enm%fz)
    call real_vector_done(enm%r0)
    call real_vector_done(enm%econt)
   enddo
   if(associated(networks))deallocate(networks)
!
   num_enm=0
   num_models=0
   initialized=.false.
!
   write(info,99) whoami ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
 99 format(' ',A,': PNM IS OFF.')
!
   end subroutine pnm_done
!
!===========================================================================
   subroutine pnm_init(n_)
   use string
   use number
   use stream
!===========================================================================
   character(len=len("PNM_INIT>") ),parameter::whoami="PNM_INIT>";!macro
   integer, optional :: n_
   integer :: i, n
   type(enet), pointer :: enm
!
   if (initialized) call pnm_done()
   if (present(n_)) then ; n=n_ ; else ; n=2; endif
!
   if (n.gt.0) then
     allocate(pnmene(n,n)) ! allocate mixing coefficients
     allocate(networks(n)) ! allocate network data storage
     allocate(models(n+1)) ! allocate models array; this is `overdimensioned` for technical simplicity in pnm_ene
     allocate(qexp(n)) ! exponential PNM flags
     allocate(beta(n)) ! temperatures for exponential PNM
     do i=1, n
      enm=>networks(i)
!associate(nodes=>networks(i)%nodes, r0=>networks(i)%r0, xn=>networks(i)%x, yn=>networks(i)%y, zn=>networks(i)%z)
! initialize components
      call int_vector_init(enm%nodes)
      call int_vector_init(enm%bonds)
      call real_vector_init(enm%x)
      call real_vector_init(enm%y)
      call real_vector_init(enm%z)
      call real_vector_init(enm%fx)
      call real_vector_init(enm%fy)
      call real_vector_init(enm%fz)
      call real_vector_init(enm%r0)
      call real_vector_init(enm%econt)
      enm%k=zero
      enm%kb=zero
      enm%knb=zero
      enm%cutoff=zero
      enm%emin=zero
      enm%eps=zero
      enm%initialized=.false.
!end associate
     enddo
     num_enm=0
     max_num_enm=n
     pnmene=zero ! initialize mixing matrix
     models=0 ! initialize to zero
     qexp=.false.! off by default
     num_models=1
     models(num_models)=1 ! index of the first pnm model in the networks array
!
     write(info,101) whoami, itoa(n) ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
 101 format(' ',A,': INITIALIZED PNM MODULE WITH ',A,' NETWORKS (MAX.)')
     initialized=.true.
    else
     write(info(1),*)' PNM MODULE INVOKED WITH AN INVALID NUMBER OF NETWORKS(',itoa(n),').';call wrndie(0,whoami,trim(info(1)))
     return
    endif
!
   end subroutine pnm_init
!===========================================================================
  subroutine pnm_add(comlyn, comlen)
!
! add a single elastic network
!
  use dimens_fcm
  use consta
  use coord; use coordc
  use number
  use string
  use stream
  use select, only : selcta
  use psf
  character(len=len("PNM_ADD>") ),parameter::whoami="PNM_ADD>";!macro
  integer, pointer, dimension(:) :: islct, jslct, kslct

  character(len=*) :: comlyn
  integer :: comlen
!
  real(chm_real) :: dist2, dref2, dx, dy, dz
  integer :: i, j, k, ii, jj, kk, l, inet
  logical :: qrem, qcomp
!
  type(enet), pointer :: enm
!



!
  if (.not.initialized) then
   call wrndie(0,whoami,trim(' PNM MODULE NOT INITIALIZED. INITIALIZING WITH DEFAULTS.'))
   call pnm_init()
  endif
!
  if (num_enm.ge.max_num_enm) then
    call wrndie(0,whoami,trim(' MAXIMUM NUMBER OF ALLOWED NETWORKS EXCEEDED (REINITIALIZE).'))
    return
  endif
!
! add network
  inet=num_enm+1 ! network index
! set network parameters
  enm=>networks(inet);
  enm%k =gtrmf(COMLYN, COMLEN, 'FORC', ONE) ! generic force constant
  enm%kb =gtrmf(COMLYN, COMLEN, 'FCBD', enm%k) ! bonded atoms
  enm%knb =gtrmf(COMLYN, COMLEN, 'FCNB', enm%k) ! non-bonded atoms
  enm%cutoff=gtrmf(COMLYN, COMLEN, 'CUT', TEN) ! cutoff for this
  enm%emin =gtrmf(COMLYN, COMLEN, 'ZERO', ZERO) ! equilibrium energy
  enm%eps =gtrmf(COMLYN, COMLEN, 'PMIX', HALF) ! enm mixing constant
!
  write(info,*) 'PNM_ADD :   FORCE   V_0   Cutoff   Epsilon'
  write(info(2),*) '------------------------------------------'
  write(info(3),10)'           ',enm%k, enm%emin, enm%cutoff, enm%eps
  if(prnlev.ge. 2) write(OUTU,'(A)') pack(info,info.ne.'');info='';
!
10 FORMAT(A,4F7.2)
! PNM node selection
  allocate(islct(natom), jslct(natom), kslct(natom))
  islct=0; jslct=0; kslct=0;

! main selection
  call SELCTA(COMLYN,COMLEN,islct,X,Y,Z,WMAIN,.TRUE.)






! auxiliary selections to generate exclusions between two sub-selections
  qrem = ( indxa(COMLYN, COMLEN, 'REMO') .gt. 0)
  if (qrem) then

   call SELCTA(COMLYN,COMLEN,jslct,X,Y,Z,WMAIN,.TRUE.)
   call SELCTA(COMLYN,COMLEN,kslct,X,Y,Z,WMAIN,.TRUE.)
  endif
!
! use selection array to add node indices and coordinates
  qcomp = (indxa(COMLYN, COMLEN, 'COMP') .gt. 0) ! whether to take reference coordinates from comparison set
  if (qcomp) then
   write(info,104) whoami, 'COMP'
   do i=1,natom
    if (islct(i) .eq. 1) then
!=============================================
     l=int_vector_add(enm%nodes,i)
     l=real_vector_add(enm%x,xcomp(i))
     l=real_vector_add(enm%y,ycomp(i))
     l=real_vector_add(enm%z,zcomp(i))
     l=real_vector_add(enm%fx,zero); l=real_vector_add(enm%fy,zero); l=real_vector_add(enm%fz,zero); l=real_vector_add(enm%econt,zero);
    endif
   enddo
  else
   write(info,104) whoami, 'MAIN'
   do i=1,natom
    if (islct(i) .eq. 1) then
     l=int_vector_add(enm%nodes,i)
     l=real_vector_add(enm%x,x(i))
     l=real_vector_add(enm%y,y(i))
     l=real_vector_add(enm%z,z(i))
     l=real_vector_add(enm%fx,zero); l=real_vector_add(enm%fy,zero); l=real_vector_add(enm%fz,zero); l=real_vector_add(enm%econt,zero)
!=============================================
    endif
   enddo
  endif ! qcomp
 104 format(' ',A,': EQUILIBRIUM GEOMETRY TAKEN FROM ',A,' SET.')
!
  write(info(2),105) whoami, itoa(enm%nodes%last)
  if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
 105 format(' ',A,': ',A,' ENM ATOMS FOUND.')
!
! compute connectivity
  dref2=enm%cutoff**2 ! for faster comparison
! O(N^2) loop to find connectivity
!
  do j=1, enm%nodes%last
    do k=j+1, enm%nodes%last
! check for node connection exclusion;
! fetch atom indices
     jj=enm%nodes%i(j) ! atom index of first node
     kk=enm%nodes%i(k) ! atom index of second node
     if ( .not. ( (jslct(jj).eq.1 .and. kslct(kk).eq.1) .or. (jslct(kk).eq.1 .and. kslct(jj).eq.1) ) ) then ! no exclusion
! distance between points i and j
        dx=enm%x%r(j) - enm%x%r(k); dy=enm%y%r(j) - enm%y%r(k); dz=enm%z%r(j) - enm%z%r(k);
        dist2 = dx**2 + dy**2 + dz**2
        if (dist2.le.dref2) then
! add to the network connectivity list
! index negative if nodes are covalently bonded
         do II = 1,NBOND ! over all bonds
          if ((JJ .EQ. IB(II) .AND. KK .EQ. IB(JJ)) .OR. (KK .EQ. IB(II) .AND. JJ .EQ. IB(JJ))) then
           kk=-kk; exit
          endif
         enddo
! update bond list (note that adding node and _NOT_ atom indices
         l=int_vector_add(enm%bonds,j); l=int_vector_add(enm%bonds, sign(k,kk) );
! equilibrium distance value
         l=real_vector_add(enm%r0,sqrt(dist2));
        endif ! dist <= dref
     endif ! no exclusion
    enddo ! k-nodes
  enddo ! jnodes
!
! set additional interaction terms
  do i=1, num_enm
   pnmene(i,inet) = half * (enm%eps + networks(i)%eps )
   pnmene(inet,i) = pnmene(i, inet)
  enddo
!
  enm%initialized=.true.
  num_enm=inet ! increment counter
! free arrays
  if(associated(ISLCT))deallocate(ISLCT)
  if(associated(JSLCT))deallocate(JSLCT)
  if(associated(KSLCT))deallocate(KSLCT)
!
  end subroutine pnm_add
!----------------------------------------------------------------
  subroutine enm_ene(DEU,X,Y,Z,INET)
  use consta
  use number
!
#if KEY_PARALLEL==1
  use parallel, only : numnod, mynod
!
  logical :: qgrp
#endif
!
! Subroutine that returns the elastic network forces of the network (INET)
  real(chm_real), dimension(*) :: X, Y, Z
  integer :: inet ! network index
  integer :: ibeg, iend
  integer :: i, j, ii, jj, iii, jjj
  real(chm_real) :: deu, dist, dref, ddx, ddy, ddz
  real(chm_real) :: d, kf
  type(enet), pointer :: enm
  real(chm_real), pointer, dimension(:) :: dx, dy, dz, econt ! pointers to gradient arrays
!
  if (inet.gt.size(networks)) return
!
  enm=>networks(inet)
!
  dx=>enm%fx%r ; dy=>enm%fy%r ; dz=>enm%fz%r
  econt=>enm%econt%r ! energy decomposition array
  deu=zero;
  do i=1, enm%nodes%last ; dx(i)=zero; dy(i)=zero ; dz(i)=zero; econt(i)=zero ; enddo
!
#if KEY_PARALLEL==1
!
  qgrp=calc_para.and.numnod.gt.1
  if (qgrp) then
   j=ceiling(one*enm%r0%last/numnod) ! bonds / cpu
   ibeg=min(j*mynod,enm%r0%last-1) + 1 ! index of first bond for this cpu
   iend=ibeg - 1 + max(0,min(j,enm%r0%last-j*mynod )) ! index of last bond for this cpu
  else ! not qgrp
   ibeg=1; iend=enm%r0%last
  endif ! qgrp
!
#else
  ibeg=1; iend=enm%r0%last
#endif
!
  do i=ibeg, iend ! over all bonds on this CPU
   jj=2*i; ii=jj-1 ; ! indices into list of pairs
   ii=enm%bonds%i( ii ) ! index of first node
   jj=enm%bonds%i( jj ) ! index of second node
   if (jj.lt.0) then ; kf=enm%kb ; else ; kf=enm%knb ; endif ! ( negative j-index corresponds to a bonded pair )
   jj=abs(jj);
   iii=enm%nodes%i(ii) ! atom index of first node
   jjj=enm%nodes%i(jj) ! atom index of second node
   ddx=(X(iii)-X(jjj)); ddy=(Y(iii)-Y(jjj)); ddz=(Z(iii)-Z(jjj));
   dist=sqrt( ddx**2 + ddy**2 + ddz**2 ) ! distance between nodes i and j
!
   dref=dist-enm%r0%r(i) ! difference from equilibrium distance
!
  
!
!======= Update total energy ====================
   d=kf * dref**2
   deu=deu+d
!======= Update the energy decomposition array ==
   d=fourth*d; ! split equally between two nodes
   econt(ii)=econt(ii) + d ; econt(jj)=econt(jj) + d ;
!==============================================
! Update forces (DX = Grad_x = -f_x in CHARMM convention)
   d=kf*dref/( max ( abs(dist),RSMALL ) ); ! protect from overflow
   dx(ii)=dx(ii) + d * ddx ; dy(ii)=dy(ii) + d * ddy; dz(ii)=dz(ii) + d * ddz
   dx(jj)=dx(jj) - d * ddx ; dy(jj)=dy(jj) - d * ddy; dz(jj)=dz(jj) - d * ddz
!======= forces update =======
  enddo ! i: bondlist
!
!end associate
!
  deu = half * deu + enm%emin ! elastic energy
!
 
!
  end subroutine enm_ene
!-------------------------------------------------------------
  subroutine pnm_ene(EU,X,Y,Z,DX,DY,DZ,QECONT,ECONT,NATOM)
! The plastic network model (PNM) energy routine.
! Original Author: Paul Maragakis
! overhaul by VO 2013
! NOTE : parallelization assumes that all coodinates are known
  use number
  use consta
  use string
!
!
#if KEY_PARALLEL==1
  use parallel, only : gcomb, numnod, mynod, mynodp, iparpt
!
  integer :: afirst, alast
  logical :: qgrp
  real(chm_real) :: pnmeneg(num_enm) ! reduced energies
  real(chm_real), dimension(:,:), pointer :: fc ! atomic pnm force arrays
#endif
!
!
  character(len=len("PNM_ENE>") ),parameter::whoami="PNM_ENE>";!macro
  real(chm_real) :: eu, deu ! total energy, energy from a particular PNM
  integer :: natom
  real(chm_real), dimension(natom) :: X, Y, Z, DX, DY, DZ ! coordinates and forces
  logical :: qecont ! decomposition flag
  real(chm_real) :: econt(*) ! decomposition array
!
  integer :: i, j, k, ii
  integer ::imodel, emodel, num_enm_this ! beginning and ending indices of ENMs in the pnm model, number of ENMs in PNM
  real(chm_real) :: dref
!
!
  real(chm_real), pointer, dimension(:) :: fx, fy, fz, edecomp ! short-hand pointers
! variables for diagonalization
  real(chm_real), pointer, dimension(:,:) :: M, evec ! copy of interaction matrix, eigenvectors
  real(chm_real), pointer, dimension(:) :: eval ! eigenvalues
  real(chm_real) :: deval(num_enm) ! force prefactor for individual enms
!
  type(enet), pointer :: enm
!
  if (num_enm.le.0) return
  if (.not.initialized) then
   call wrndie(0,whoami,trim(' PNM MODULE NOT INITIALIZED. NOTHING DONE.'))
   return
  endif
!
  eu=zero
!
! note : num_models is less than or equal to num_enm (equality with one ENM per model)
  do j=1, num_enm ! over all networks (in all models)
!
#if KEY_PARALLEL==1
   call enm_ene(pnmeneg(j),X,Y,Z,j)
#else
   call enm_ene(pnmene(j,j),X,Y,Z,j)
#endif
!
  enddo ! j: over all networks
! reduce energies in parallel
! compute energy of the plastic network
! diagonalize interaction matrix and take the lowest eigenvalue
! using the default diagonalized in CHARMM, which may only work for symmetric
! matrices; this is OK as long as the interaction matrix is kept symmetric
! for the exponential version of the model, diagonalization is not needed (see below)
#if KEY_PARALLEL==1
!
  qgrp=calc_para .and.numnod.gt.1
  if (qgrp) &
 & call gcomb(pnmeneg, num_enm)
  do j=1, num_enm ; pnmene(j,j)=pnmeneg(j); enddo ! update diagonal entries (energies)
!
  if (qecont) then ; allocate(fc(natom,4)) ; else ; allocate(fc(natom,3)); endif ; fc=zero ! initialize force arrays
!
#endif
!
! now loop over all models and diagonalize
  do k=1, num_models ! over all pnm models
   imodel=models(k) ! network index of first model
   emodel=models(k+1)-1 ; if (emodel.le.0) emodel=num_enm ! network index of second model
   num_enm_this=emodel-imodel+1; ! number of ENMs in this PNM
!
   if (qexp(k)) then ! code for exponential version
!=========== exponential PNM
    deu=zero ; dref=zero
!=========== compute in a numerically stable way for large energies
! 1) find minimum energy
    ii=imodel ! location of minimum
    deu=pnmene(ii,ii)
    do j=imodel+1, emodel
     if (deu.gt.pnmene(j,j)) then ; ii=j ; deu=pnmene(j,j) ; endif
    enddo
! 2) compute correction to minimum energy
    do j=imodel, emodel
      deval(j)=exp( - beta(k) * (pnmene(j,j)-deu) ) ! relative to minimum
      dref = dref + deval(j)
    enddo
    deu = deu - log ( dref ) / beta(k)
    deval(imodel:emodel) = deval(imodel:emodel) / dref ! gradient contribution coefficients for all except ii

   else
!============ standard PNM
    allocate(M(num_enm_this, num_enm_this), evec(num_enm_this, num_enm_this), eval(num_enm_this))
    M=pnmene(imodel:emodel,imodel:emodel); ! copy part of interaction matrix corresponding to this PNM
! diagonalize matrix
! diagq is inaccurate; using diagrs
    call diagrs('FULL',num_enm_this,M,eval,evec)
!
    ii=1; deu=eval(ii); do i=2, num_enm_this ; if ( eval(i) .lt. deu ) then ; ii=i ; deu=eval(ii) ; endif ; enddo ! scan all evals to find lowest
! compute corresponding eigenvalue (energy) derivatives w.r.t individual ENM energies (diagonal matrix components);
    deval(imodel:emodel) = evec (:, ii)**2 ;
! make sure eigenvectors are normalized to unity:
    dref = sum(deval(imodel:emodel)) ; if (dref .gt. RSMALL) deval(imodel:emodel)=deval(imodel:emodel) / dref ;
! deallocate arrays
    deallocate(M, evec, eval)
   endif ! qexp
   eu=eu+deu ! add this network`s energy contribution to total PNM energy
  enddo ! over all models
!



! apply forces
! all CPUs do this (even in parallel, because in that case only partial forces are computed by each CPU)
!
  if (qecont) then
   do j=1,num_enm ! over all networks
    enm=>networks(j)
    fx=>enm%fx%r; fy=>enm%fy%r; fz=>enm%fz%r; edecomp=>enm%econt%r;
    do i=1, enm%nodes%last
     ii=enm%nodes%i(i) ! index
!
#if KEY_PARALLEL==1
     fc(ii,1) = fc(ii,1) + deval(j) * fx(i);
     fc(ii,2) = fc(ii,2) + deval(j) * fy(i);
     fc(ii,3) = fc(ii,3) + deval(j) * fz(i);
     fc(ii,4) = fc(ii,4) + edecomp(i);
#else
     dx(ii) = dx(ii) + deval(j) * fx(i);
     dy(ii) = dy(ii) + deval(j) * fy(i);
     dz(ii) = dz(ii) + deval(j) * fz(i);
     econt(ii) = econt(ii) + edecomp(i);
#endif
!
    enddo ! over nodes
   enddo ! over networks
  else
   do j=1,num_enm ! over all networks
    enm=>networks(j)
    fx=>enm%fx%r; fy=>enm%fy%r; fz=>enm%fz%r;
    do i=1, enm%nodes%last
     ii=enm%nodes%i(i)
!
#if KEY_PARALLEL==1
     fc(ii,1) = fc(ii,1) + deval(j) * fx(i);
     fc(ii,2) = fc(ii,2) + deval(j) * fy(i);
     fc(ii,3) = fc(ii,3) + deval(j) * fz(i);
!
#else
     dx(ii) = dx(ii) + deval(j) * fx(i)
     dy(ii) = dy(ii) + deval(j) * fy(i)
     dz(ii) = dz(ii) + deval(j) * fz(i)
#endif
!
    enddo ! over nodes
   enddo ! over networks
  endif
!
!
#if KEY_PARALLEL==1 /* (parallel) */
#if KEY_PARAFULL==1
  afirst=1+IPARPT(MYNOD)
  alast =IPARPT(MYNODP)
#else
  afirst=1
  alast=natom
#endif
! reduce gradients and scatter
  if (qgrp) &
 & call vdgsum(fc(:,1), fc(:,2), fc(:,3),0)
! add to main force array
  do i=afirst, alast
   dx(i)=dx(i)+fc(i,1);
   dy(i)=dy(i)+fc(i,2);
   dz(i)=dz(i)+fc(i,3);
  enddo
  if(associated(fc))deallocate(fc)
!
 if (mynod.gt.0) eu=zero ! energies will be reduced outside of this routine
!
#endif /*(parallel) */
!
  end subroutine pnm_ene
!
#if KEY_STRINGM==0 /*  VO: otherwise can use string method aux subroutines */
!========================= auxiliary vector subroutines follow
!====================================================
       subroutine int_vector_init( v, expand_incr_ )
       type (int_vector) :: v
       integer, optional :: expand_incr_
       integer :: length
       if (v%initialized) return
       if (present(expand_incr_)) then
        length=max(expand_incr_, expand_incr)
       else
        length=expand_incr
       endif
       allocate(v%i(length))
       v%i=0
       v%length=length
       v%last=0
       v%initialized=.true.
       end subroutine int_vector_init
!====================================================
       subroutine int_vector_done( v )
       type (int_vector) :: v
       if (associated(v%i)) deallocate(v%i)
       v%length=0
       v%last=0
       v%initialized=.false.
       end subroutine int_vector_done
!====================================================
       subroutine int_vector_expand( v, expand_incr_ )
       type (int_vector) :: v
       integer, optional :: expand_incr_
       integer :: newlength
       integer, dimension(:), allocatable :: p
!
       if (.not.v%initialized) then
        call int_vector_init(v)
       else
! assume length is valid
        if (present(expand_incr_)) then
         newlength=v%length+max(expand_incr_, expand_incr)
        else
         newlength=v%length+expand_incr
        endif
        allocate(p(newlength)) ! new memory
        p(1:v%length)=v%i ! copy old data
        deallocate(v%i) ! delete old data
        allocate(v%i(newlength))
        v%i = p ! copy data
        deallocate(p)
        v%length=newlength
       endif
       end subroutine int_vector_expand
!====================================================
       function int_vector_add( v,i ) ! add a new element to the list (not necessarily unique)
! and return its index
       type (int_vector) :: v
       integer :: i
       integer :: j, int_vector_add
!
       if (.not.v%initialized) call int_vector_init(v)
! add element to the list
       if (v%last.eq.v%length) call int_vector_expand(v)
       j=v%last+1
       v%i(j)=i
       v%last=j
       int_vector_add=j
       end function int_vector_add
!====================================================
       subroutine real_vector_init( v, expand_incr_ )
       type (real_vector) :: v
       integer, optional :: expand_incr_
       integer :: length
       if (v%initialized) return
       if (present(expand_incr_)) then
        length=max(expand_incr_, expand_incr)
       else
        length=expand_incr
       endif
       allocate(v%r(length))
       v%r=0
       v%length=length
       v%last=0
       v%initialized=.true.
       end subroutine real_vector_init
!====================================================
       subroutine real_vector_done( v )
       type (real_vector) :: v
       if (associated(v%r)) deallocate(v%r)
       v%length=0
       v%last=0
       v%initialized=.false.
       end subroutine real_vector_done
!====================================================
       subroutine real_vector_expand( v, expand_incr_ )
       type (real_vector) :: v
       integer, optional :: expand_incr_
       integer :: newlength
       real(chm_real), dimension(:), allocatable :: p
!
       if (.not.v%initialized) then
        call real_vector_init(v)
       else
! assume length is valid
        if (present(expand_incr_)) then
         newlength=v%length+max(expand_incr_, expand_incr)
        else
         newlength=v%length+expand_incr
        endif
        allocate(p(newlength)) ! new memory
        p(1:v%length)=v%r ! copy old data
        deallocate(v%r) ! delete old data
        allocate(v%r(newlength))
        v%r = p ! copy data
        deallocate(p)
        v%length=newlength
       endif
       end subroutine real_vector_expand
!====================================================
       function real_vector_add( v,i ) ! add a new element to the list (not necessarily unique)
! and return its index
       type (real_vector) :: v
       real(chm_real) :: i
       integer :: j, real_vector_add
!
       if (.not.v%initialized) call real_vector_init(v)
! add element to the list
       if (v%last.eq.v%length) call real_vector_expand(v)
       j=v%last+1
       v%r(j)=i
       v%last=j
       real_vector_add=j
       end function real_vector_add
#endif
!========================= done with auxiliary vector subroutines
#endif /*(pnm_main) */
!======================================================================
end module pnm
