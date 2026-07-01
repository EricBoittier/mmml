! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
! zero-temperature string module
!
      module sm0k ! string-method-at-0-K
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      use chm_kinds
!
      private
!
! VARIABLES
      character(len=132), public :: sm0k_info(21) = (/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer

 !ccccc initialization flag
      logical, public, save :: sm0k_initialized=.false.
!ccccc number of replicas on the string
      integer, save :: nstring=-1
      integer, save :: mestring=-1
!ccccc GENERAL VARS
      logical, save :: repa_initialized
      integer, save :: norient, nmove
! interpolation methods
      integer, parameter :: linear=1, spline=2, bspline=3, dst=4, &
     & linear_exact=5
!
      integer, save :: interp_method=0,orient=0
      logical, save :: qstat_orient=.false.
      integer, save :: orient_mass=0, repa_mass=0
      integer, save :: iterations=1 ! maximum interpolation iterations
      real(chm_real), save :: def=1.1d0 ! interpolation tolerance
      real(chm_real), save :: dst_cutoff=1.0d0 ! wavenumber truncation parameter for DST
!
! arrays
      real(chm_real), save, pointer :: &
     & rcurrent_m(:,:), rref_o(:,:), rcurrent_o(:,:)
! arclength and curvature
      real(chm_real), save, pointer :: ds(:), curv(:) ! unavailable at first iteration
! orientation weights -- for 0-temp. string
      real(chm_real), save, pointer :: orientWeights(:), repaWeights(:)
      integer, save, pointer :: iatom_o(:), iatom_m(:), &
     & iatom_free_o(:), iatom_free_m(:)
      integer, save, pointer :: iatom_f(:)
      integer, save :: nfix_bckl
      logical, save, pointer :: fixed_o(:), fixed_m(:), fixed_s(:)
!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccc STATISTICS VARS cccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
      integer, save :: stat_iteration_counter=0 ! how many times 'stat' has been called
      logical, save :: output_rmsd0=.false., &
     & output_dsdt=.false., &
     & output_arclength=.false., & ! output options
     & output_curvature=.false., &
     & output_rmsd_ave=.false. ! rmsd wrt to the average structure
      logical, save :: stat_rmsd_mass=.false. ! should the statistics routine do mass-weighted RMSD?
      logical, save :: stat_initialized=.false.
!
      character(len=80), save :: rmsd0_fname='', dsdt_fname='', s_fname='', &
     & rmsd_ave_fname='', c_fname='' ! output names
!
      integer, save :: nstat
      integer, save :: rmsd0_funit=-1, dsdt_funit=-1, s_funit=-1, &
     & rmsd_ave_funit=-1, c_funit=-1


      logical, save :: output_energy=.false.
      character(len=80), save :: energy_fname=''
      integer, parameter :: enmax=100
      character(len=4), save :: energy_names(enmax) ! arrays large enough to
      integer, save :: energy_indices(enmax) ! indices into the EPROP array
      integer, save :: num_energy_terms=0
      integer, save :: energy_flen=0

      integer, save :: num_average_samples=0 ! number of samples in the average set
      integer, save :: rmsd0_flen=0, dsdt_flen=0, s_flen=0, rmsd_ave_flen=0, c_flen
! arrays
      real(chm_real), save, pointer, dimension(:,:) :: &
     & rold_s, rave_s, rcurrent_s, rcomp_s, rold_o, rave_o, rcomp_o
      real(chm_real), save, pointer :: statWeights(:)
      integer, save, pointer :: iatom_s(:), iatom_free_s(:)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! SUBROUTINES
!
      public sm0k_repa
      public sm0k_stat
      public sm0k_confcons
      public sm0k_chirality
      public sm0k_fixed_atoms ! used by ftsm

      ! for sm0k_main
      public sm0k_done
      public sm0k_init
      public sm0k_interpolate
      public sm0k_repa_init
      public sm0k_stat_init
      
      contains

      subroutine sm0k_init()
      use stream
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux;
#endif
      use mpi
      use param_store, only: set_param
      use parallel, only: psnd4, psnd8
!
      implicit none
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
      logical :: qroot, qslave
      integer :: ierror
      character(len=len("SM0K_INIT>") ),parameter::whoami="SM0K_INIT>";!macro
!
! do a basic communicator check:
      if (ME_LOCAL.eq.0.and.ME_STRNG.eq.MPI_UNDEFINED) then
        write(sm0k_info, 111) whoami, ME_GLOBAL, whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
 111 FORMAT(A, ' WORLD REPLICA ',I5, ' HAS ZERO GROUP ID', &
     & /,A,' BUT INVALID STRING ID (MAY BE OK).')
      elseif (ME_STRNG.ne.MPI_UNDEFINED.and. &
     & (ME_LOCAL.ne.0.or.MPI_COMM_LOCAL.eq.MPI_COMM_NULL)) then
        write(sm0k_info, 112) whoami, ME_GLOBAL, whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
 112 FORMAT(A, ' WORLD REPLICA ',I5, ' HAS A VALID STRING ID', &
     & /,A,' BUT A NONZERO GROUP ID. ABORTING.')
       return
      endif
!
      qroot=ME_STRNG.ne.MPI_UNDEFINED
      qslave=ME_LOCAL.ne.MPI_UNDEFINED ! (also includes roots)
!
      if (sm0k_initialized) then
       if (qroot) then
        if (ME_STRNG.eq.0) then
          write(sm0k_info,'(2A)') &
     & whoami, ' SM0K ALREADY INITIALIZED. CALL "DONE" TO CLEAN UP.' ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        endif
       endif
       return
      endif
!
      nstring=1 ! safe (hopefully) default
      mestring=-1 ! safe (hopefully) default
!
      if (qroot) then
        nstring=SIZE_STRNG
        mestring=ME_STRNG
      endif
! broadcast string size to all slave nodes
#if (KEY_INTEGER8==1)
      call PSND8(nstring,1)
#endif
#if (KEY_INTEGER8==1)
      call PSND8(mestring,1)
#endif
#if (KEY_INTEGER8==0)
      call PSND4(nstring,1)
#endif
#if (KEY_INTEGER8==0)
      call PSND4(mestring,1)
#endif
! set environment variables
      call set_param('NSTRING',nstring)
      call set_param('MESTRING',mestring)
!
      if (qroot) then
        if (ME_STRNG.eq.0) then
          write(sm0k_info,'(2A,I5, A)') &
     & whoami, ' FOUND ',nstring,' REPLICAS.' ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        endif
      endif
! store fixed atom indices
      iatom_f=>sm0k_fixed_atoms()
      nfix_bckl=size(iatom_f)


!
      sm0k_initialized=.true.
!
      end subroutine sm0k_init
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine sm0k_done()
      use stream
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux;
#endif
      use mpi
      use param_store, only: set_param

      implicit none
!
      character(len=len("SM0K_DONE>") ),parameter::whoami="SM0K_DONE>";!macro
!
      if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then
       write(sm0k_info,'(2A,I5, A)') whoami, ' CLEANING UP.' ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
      endif
      sm0k_initialized=.false.
      nstring=-1
      mestring=-1
!

! deallocate fixed atom indices
      nfix_bckl=-1;
! set envorinment variable
      call set_param('NSTRING',nstring)
      call set_param('MESTRING',mestring)

      if(associated(iatom_f))deallocate(iatom_f)
      if(associated(ds))deallocate(ds)
      if(associated(curv))deallocate(curv)
!
      end subroutine sm0k_done
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

!
      function sm0k_fixed_atoms()
      use stream
      use dimens_fcm
      use psf
      use tsmh, only : backls ! TSM common blocks included because minimizer DOF depend on TSM
      use tsms_mod, only : qtsm
      use coord
      implicit none
!
      integer, pointer :: sm0k_fixed_atoms(:)
      integer, pointer :: backlist(:)
      integer :: nfix, i, j
      logical :: qfree
      allocate(backlist(natom)); backlist=0 ! need to initialize, otherwise nfix might be wrong
! try to acommodate TSM
#if (KEY_TSM==1)
      if (qtsm) then
! j=bpert(backls)
! backlist=heap(j:j+natom-1) ! heap must have of integer type (it was when this line was written)
        backlist=backls
      endif
#endif
! construct indexing that excludes fixed and backlist atoms
      nfix = count(abs(imove(1:natom))+abs(backlist(1:natom)).gt.0)
      allocate(sm0k_fixed_atoms(nfix))
!
      nfix=0
      do i=1,natom
       qfree=imove(i).eq.0 ! see egrad1.src routines
#if (KEY_TSM==1)
       if (qtsm) qfree=qfree.and.backlist(i).eq.0
#endif
       if (.not.qfree) then
        nfix=nfix+1
        sm0k_fixed_atoms(nfix)=i
       endif
      enddo
!
      deallocate(backlist)
      end function sm0k_fixed_atoms

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine sm0k_repa_init(COMLYN, COMLEN)
! initialize string reparametrization
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      use stream
      use dimens_fcm
      use string
      use coord; use coordc
      use select, only : selcta, selrpn, nselct; use psf

#if (KEY_TSM==1) /*  TSM common blocks included because minimizer DOF depend on TSM */
      use tsmh, only:backls
#endif
#if (KEY_TSM==1)
      use tsms_mod, only:qtsm
#endif

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
!
      CHARACTER(LEN=*) :: COMLYN
      integer :: COMLEN
!
      character(len=22) :: methods(5)
      character(len=8) :: keyword
      data methods &
     & / 'LINEAR','CUBIC SPLINE','B-SPLINE','DST','LINEAR EXACT'/
! selection array
      integer :: i ,j, mlen
      integer :: qlinear, qspline, qbspline, qdst, qlinear_exact

      integer, pointer :: backlist(:) ! for nominal compatibility with TSM
      integer, pointer :: ifixed(:) ! fixed atom indices
      integer :: imode
      logical :: qfree



      integer, pointer :: iselct(:), jselct(:)
!
      character(len=len("SM0K_REPA_INIT>") ),parameter::whoami="SM0K_REPA_INIT>";!macro
!
      if (.not.sm0k_initialized) call sm0k_init()
!
! reset variables
      qspline=0
      qbspline=0
      qlinear=0
      qdst=0
      qlinear_exact=0
      dst_cutoff=0.0
      interp_method=0
      orient=0
      orient_mass=0 ! will the orientation use mass weighting?
      repa_mass=0 ! will the reparametrization use mass weighting
      nmove=0
      norient=0
      num_average_samples=0
!
! deallocate arrays
      if(associated(fixed_o))deallocate(fixed_o)
      if(associated(fixed_m))deallocate(fixed_m)
      if(associated(iatom_o))deallocate(iatom_o)
      if(associated(iatom_m))deallocate(iatom_m)
      if(associated(iatom_free_o))deallocate(iatom_free_o)
      if(associated(iatom_free_m))deallocate(iatom_free_m)
      if(associated(rref_o))deallocate(rref_o)
      if(associated(rcurrent_o))deallocate(rcurrent_o)
      if(associated(rcurrent_m))deallocate(rcurrent_m)
      if(associated(orientWeights))deallocate(orientWeights)
      if(associated(repaWeights))deallocate(repaWeights)
      if(associated(ds))deallocate(ds) ! deallocate arclength+curavture arrays
      if(associated(curv))deallocate(curv) ! deallocate curvature array
!
      repa_initialized=.false.
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (indxa(comlyn, comlen, 'LINE').gt.0) then
       qlinear=1
       interp_method=linear
      endif
      if ((indxa(comlyn, comlen, 'CSPL').gt.0).or. &
     & (indxa(comlyn, comlen, 'SPLI').gt.0)) then
       qspline=1
       interp_method=spline
      endif
      if (indxa(comlyn, comlen, 'BSPL').gt.0) then
       qbspline=1
       interp_method=bspline
      endif
      if (indxa(comlyn, comlen, 'DST').gt.0) then
       qdst=1
       interp_method=dst
! did the user specify filter cutoff?
       dst_cutoff=gtrmf(comlyn, comlen, 'WNCT', -1.0d0)
       if (dst_cutoff.lt.0.0) then
        write(sm0k_info,664) whoami, whoami ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
 664 FORMAT(A,' DST REQUESTED BUT FILTER CUTOFF', &
     & A, ' NOT SPECIFIED.',/,' WILL USE 0.500')
        dst_cutoff=0.5
       endif
      endif
      if (indxa(comlyn, comlen, 'LIN2').gt.0) then
       qlinear_exact=1
       interp_method=linear_exact
      endif
!ccccccc CHECK FOR MULTIPLE OPTIONS
      if ((qspline+qlinear+qbspline+qdst+qlinear_exact) .eq. 0) then
       write(sm0k_info,665) whoami, whoami ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
 665 FORMAT(A,' INTERPOLATION METHOD NOT SPECIFIED.',/, &
     & A,' WILL USE LINEAR INTERPOLATION.')
       interp_method=linear
      elseif ((qdst+qspline+qlinear+qbspline+qlinear_exact) .gt. 1)then
       call wrndie(0,whoami,trim('TOO MANY INTERPOLATION OPTIONS.'))
      endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! did the user specify an interpolation tolerance?
      if (interp_method.ne.linear_exact) then ! options below are invalid for exact interpolation
       def=gtrmf(comlyn, comlen, 'DEFI', 1.1d0)
       if (def.lt.1.0) then
         call wrndie(0,whoami,trim('INTERPOLATION TOLERANCE MUST BE >= 1. EXITING.'))
         return
       endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! did the user specify a maximum number of iterations?
       iterations=gtrmi(comlyn, comlen, 'ITER', 10)
      else
       def=0d0
       iterations=0
      endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! check for orientation options: must appear after 'ORIE'
! process selection
!

      allocate(iselct(natom), jselct(natom)) ; iselct=0 ! 1: select all atoms by default; 0: select no atoms
      allocate(backlist(natom))
      backlist=0 ! setting this to zero is the simplest way to provide correct default behavior (see below)

!
      i=indxa(comlyn, comlen, 'ORIE')
      if (i.gt.0) then ! only if the ORIE directive exists
! selection text taken from corman.src
       orient=1 ! NOTE : this implies that orientation is off by default !
       j=indx(comlyn, comlen, 'SELE', 4)
       if (j.gt.0.and.j.lt.i) then ! sele occurs before orie
        call wrndie(0,whoami,trim('ATOM SELECTION MUST BE SPECIFIED AFTER ORIE.'))
        return
       endif
!

       IMODE=0
!
       CALL SELRPN(COMLYN,COMLEN,iselct,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0) THEN
        CALL WRNDIE(0,whoami,'ATOM SELECTION ERROR')
        RETURN
       ENDIF




      endif ! orie specified
!
! check for mass weighting
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      j=1
      do while (j.gt.0)
       mlen=comlen
       j=indxa(comlyn, comlen, 'MASS')
       if ( (orient.eq.1).and.(j.ge.i) ) then
        orient_mass=1
       else if (j.gt.0) then
        repa_mass=1
        i=i-(mlen-comlen) ! how much the string has shrunk
       endif
      enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! check which atoms will be moved by reparametrization (default: all atoms)
! process selection
      i=indxa(comlyn, comlen, 'MOVE')
      if (i.gt.0) then ! only if the MOVE directive exists
       j=indx(comlyn, comlen, 'SELE', 4)
       if (j.gt.0.and.j.lt.i) then ! sele occurs before orie
        call wrndie(0,whoami,trim('ATOM SELECTION MUST BE SPECIFIED AFTER MOVE.'))
        return
       endif
!

       IMODE=0
       CALL SELRPN(COMLYN,COMLEN,jselct,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0) THEN
        CALL WRNDIE(0,whoami,'ATOM SELECTION ERROR')
        RETURN
       ENDIF
      else
       jselct=1 ! reparametrization will move all atoms by default
      endif ! move specified
!
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! now process the selections
! 3/11 provide compatibility with fixed atoms and nominal compatibility with TSM backlists
!
      norient=count( iselct(1:natom).gt.0 ) ! total number of orientation atoms
!
      if (norient.eq.0) then
        call wrndie(0,whoami,trim('NO ATOMS SELECTED FOR ORIENTATION. WILL NOT ORIENT.'))
        orient=0 ! invalid or missing selection for orientation
       elseif (norient.lt.3) then
        call wrndie(0,whoami,trim('FEWER THAN THREE ATOMS SELECTED FOR ORIENTATION. WILL NOT ORIENT.'))
        orient=0
      endif
!
      nmove=count( jselct(1:natom).gt.0 ) ! total number of moving atoms (may be overcounted -- see below)
!
! try to accommodate TSM
#if (KEY_TSM==1)
      if (qtsm) then
! j=bpert(backls)
! backlist=heap(j:j+natom-1) ! heap must have of integer type (it was when this line was written)
        backlist=backls
      endif
#endif
! check that the number of fixed atoms hasn`t changed since initialization; if it has, issue warning
      ifixed=>sm0k_fixed_atoms()
! the problem is that ifort sometimes crashes when arrays of unequal length are compared
      qfree=size(ifixed).eq.nfix_bckl ; if (qfree) qfree=all(ifixed.eq.iatom_f)
!
      if (.not.qfree) then
       call wrndie(0,whoami,trim('FIXED ATOM ARRAY CHANGED AFTER LAST STRING COMMAND.'))
       deallocate(iatom_f)
       iatom_f=>ifixed
       nfix_bckl=size(iatom_f)
       nullify(ifixed)
      else
       deallocate(ifixed)
      endif
!
! allocate space for various arrays atom array
      allocate(iatom_o(norient), iatom_m(nmove), &
     & iatom_free_o(norient), iatom_free_m(nmove), & ! these index lists are for use with minimizer arrays (fixed atoms removed)
     & fixed_o(norient), fixed_m(nmove), & ! these flags indicate that the atom will be absent from minimizer array
     & rref_o(norient,3),rcurrent_o(norient,3), &
     & rcurrent_m(nmove,3), &
     & orientWeights(norient), repaWeights(nmove))
!
      iatom_o=0; iatom_m=0;
      iatom_free_o=0; iatom_free_m=0;
      rref_o=0d0
      rcurrent_o=0d0
      rcurrent_m=0d0
      orientWeights=1d0
      repaWeights=1d0
! build various index arrays
!
!
      norient=0
      nmove=0
!
      j=-2 ! free atom index; want the valid indices to be 1,4,7,..etc (see below)
      do i=1,natom
       qfree=imove(i).eq.0 ! see egrad1.src routines
#if (KEY_TSM==1)
       if (qtsm) qfree=qfree.and.backlist(i).eq.0
#endif
       if (qfree) j=j+3 ! increment free index -- atom is movable and will be moved by minimization
! orientation indices
       if (iselct(i).gt.0) then
        norient=norient+1
        iatom_o(norient)=i
        if (qfree) then
         iatom_free_o(norient)=j
         fixed_o(norient)=.false.
        else
         iatom_free_o(norient)=-999 ! unknown index because the minimization routine will not supply the coordinate
         fixed_o(norient)=.true. ! this flag will tepp sm0k repa not to use free index
        endif
       endif
! moving indices
       if (jselct(i).gt.0) then
        if (.not.qfree) then ! cannot (well, should not) move fixed atoms: warn but skip entry & continue
#if (KEY_TSM==1)
         write(sm0k_info(1),*)'ATOM ',i,' IS FIXED OR IN A TSM BACKLIST. SKIPPING.';call wrndie(0,whoami,trim(sm0k_info(1)))
#endif
#if (KEY_TSM==0)
         write(sm0k_info(1),*)'ATOM ',i,' IS FIXED. SKIPPING.';call wrndie(0,whoami,trim(sm0k_info(1)))
#endif
        else
         nmove=nmove+1
         iatom_m(nmove)=i
         iatom_free_m(nmove)=j
         fixed_m(nmove)=.false.
        endif
       endif
!
      enddo
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      if (nmove.eq.0) then ! this will occur if all of the atoms selected are fixed
        call wrndie(0,whoami,trim('NO ATOMS SELECTED FOR REPARAMETRIZATION. CANNOT CONTINUE.'))
        if(associated(iselct))deallocate(iselct)
        if(associated(jselct))deallocate(jselct)
        if(associated(backlist))deallocate(backlist)
        return
      endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! done with selections
!!!!!!!!! process mass-weighting
      if (orient_mass.eq.1) then
       do i=1,norient
        orientWeights(i)=amass(iatom_o(i))*orientWeights(i) ! these weights are for the Best-Fit routine
! orientWeights(i)=sqrt(amass(iatom_o(i)))*orientWeights(i)
       enddo
!
       do i=1, nmove
! repaWeights(i)=sqrt(amass(iatom_m(i)))*repaWeights(i) ! these weights are essentially for multiplications
        repaWeights(i)=amass(iatom_m(i))*repaWeights(i)
       enddo
      endif
!!!!!!
! normalize orientWeights;
      orientWeights=orientWeights/sum(orientWeights)
! unnecessary -- repaWeights are normalized in interpolation routine
      repaWeights=repaWeights/sum(repaWeights)
!
! print summary
!!!!!! reparametrization summary !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      mlen=len(methods(interp_method))
call trima(methods(interp_method), mlen)
      if (interp_method.eq.linear_exact) then
        write(sm0k_info,666) whoami,methods(interp_method)(1:mlen)
      else
        write(sm0k_info,667) whoami,methods(interp_method)(1:mlen),whoami, &
     & def
      endif
      if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
 666 format(A,' WILL REPARAMETRIZE STRING USING ',A,' INTERPOLATION')
 667 format(A,' WILL REPARAMETRIZE STRING USING ',A,/, &
     &A,' INTERPOLATION TO WITHIN MAX(DS)/MIN(DS) < ',F7.3,' TOLERANCE')
      if (iterations.gt.0) then ; write(sm0k_info,668) whoami, iterations ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info=''; ; endif
 668 format(A,' WITH A MAXIMUM OF ',I3,' ITERATIONS')
      if(interp_method.eq.dst) then ; write(sm0k_info,6680) whoami,dst_cutoff*100.0 ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info=''; ; endif
 6680 format(A,' DST INTERPOLATION WILL USE THE LOWER ',F8.4, &
     & '% OF WAVENUMBERS')
!
      write(keyword,'(I8)') nmove ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
      mlen=len(keyword)
      call trima(keyword, mlen)
      write(sm0k_info,669) whoami, keyword(1:mlen) ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
 669 format(A,' INTERPOLATION IS BASED ON ',A,' CARTESIAN COORDINATES')
      if (repa_mass.eq.1) then ; write(sm0k_info,672) whoami ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info=''; ; endif
 672 format(A,' INTERPOLATION WILL USE MASS-WEIGHTING')
!
!!!!!! orientation summary !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      if (orient.eq.1) then
        write(keyword,'(I8)') norient ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        mlen=len(keyword)
        call trima(keyword, mlen)
        write(sm0k_info,670) whoami, keyword(1:mlen) ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
 670 format(A,' STRING WILL BE ORIENTED BASED ON ',A,' ATOMS')
        if (orient_mass.eq.1) then ; write(sm0k_info,671) whoami ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info=''; ; endif
 671 format(A,' ORIENTATION WILL USE MASS-WEIGHTING')
      endif ! orient
!
! initialize arclength array
      if (.not.associated(ds)) then
       allocate(ds(nstring-1))
       ds=0.0d0
      endif
! initialize curvature array
      if (.not.associated(curv)) then
       allocate(curv(nstring-2))
       curv=0.0d0
      endif
!
      repa_initialized=.true.
!
! deallocate temporary variables
      if(associated(iselct))deallocate(iselct)
      if(associated(jselct))deallocate(jselct)
       if(associated(backlist))deallocate(backlist)
!
      end subroutine sm0k_repa_init
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine sm0k_repa(n,var)
! zero-temperature:reparametrization is based
! on Cartesian coordinates
! note: var is assumed to contain coordinates of free atoms (CHARMM SD minimizer convention) in [x1,y1,z1,x2,y2,z2...] triplet format
!
      use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3
      use stream
#if (KEY_MULTICOM==1)
      use multicom_aux;
#endif
      use number
      use dimens_fcm
      use coord; use coordc
      use string
      use mpi
      use psf
!
      implicit none
!ccccccccccccccccccccccccccccccccccccccc
      integer :: n ! NOTE: arrays as optional arguments can be a dangerous feature of F90
      real(chm_real), optional :: var(*) ! ideally, n should give the dimension of n, but we live in an imperfect world
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
      integer :: stat(MPI_STATUS_SIZE)
      integer :: ierror
      integer :: me, ncpu
      integer :: i, j
      real(chm_real) :: t
      logical :: qroot, qslave, qmanual
! aux arrays
      real(chm_real) :: weights(nmove,3) ! for reparametrization
      real(chm_real) :: u(3,3)=RESHAPE( (/1,0,0,0,1,0,0,0,1/), (/3,3/) ) ! rotation matrix
      real(chm_real) :: rcurrent_com(3)=(/0d0,0d0,0d0/) ! COM vector
!
! interface to reparametrization routine
! needed because of assumed shape array below
!
      interface
        subroutine interp_driver_sci(rin,rout,wgt,n, &
     & interp_method,tol,max_iterations,d_arclength, curvature, &
     & dst_cutoff, dr,r_bc_0, r_bc_1)
        use stream
        use chm_kinds
        implicit none
        integer :: n
        real(chm_real) :: rin(n), rout(n), wgt(n)
        integer, intent(in) :: interp_method
        integer :: max_iterations
        real(chm_real) :: tol, d_arclength(:), curvature(:)
        real(chm_real), optional :: dst_cutoff
        real(chm_real), optional :: dr(n) ,r_bc_0(n), r_bc_1(n)
        end subroutine interp_driver_sci
!
        subroutine interp_driver_sci_root(rin,rout,wgt,n, &
     & interp_method,tol,max_iterations,d_arclength, curvature, &
     & dst_cutoff, dr,r_bc_0, r_bc_1)
        use stream
        use chm_kinds
        implicit none
        integer :: n
        real(chm_real) :: rin(n), rout(n), wgt(n)
        integer, intent(in) :: interp_method
        integer :: max_iterations
        real(chm_real) :: tol, d_arclength(:), curvature(:)
        real(chm_real), optional :: dst_cutoff
        real(chm_real), optional :: dr(n) ,r_bc_0(n), r_bc_1(n)
        end subroutine interp_driver_sci_root
!
        subroutine interp_linear_exact(rin,rout,wgt,n, &
     & d_arclength, curvature, &
     & drout, &
     & r_bc_0, r_bc_1)
        use chm_kinds
        implicit none
        integer :: n
        real(chm_real) :: rin(n), rout(n), wgt(n)
        real(chm_real) :: d_arclength(:), curvature(:)
        real(chm_real), optional :: drout(n) ! optional computation of tangent
        real(chm_real) , optional :: r_bc_0(n), r_bc_1(n) ! optional fixed bc data
       end subroutine interp_linear_exact
!
      end interface
!
      character(len=len("SM0K_REPA>") ),parameter::whoami="SM0K_REPA>";!macro
!
      if (.not.sm0k_initialized) call sm0k_init()
      qmanual=(n.eq.0)
!
      qroot =MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.gt.1
      qslave=MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1
!
! check if the user has made an initialization call
      if (.not.repa_initialized) then
       call wrndie(0,whoami,trim('NO REPARAMETRIZATION OPTIONS SELECTED. NOTHING DONE.'))
       return
      endif
!
! load coordinates
! manual reparametrization: use PSF indices
      if (qmanual) then
! orientation
       do i=1,norient
        j=iatom_o(i)
        rcurrent_o(i,1)=x(j); rcurrent_o(i,2)=y(j); rcurrent_o(i,3)=z(j)
       enddo
! moving
       do i=1,nmove
        j=iatom_m(i)
        if (imove(j).ne.0) then ; call wrndie(0,whoami,trim('ATTEMPED TO MOVE A FIXED ATOM. THIS IS MOST LIKELY BAD.')) ; endif
        rcurrent_m(i,1)=x(j); rcurrent_m(i,2)=y(j); rcurrent_m(i,3)=z(j)
       enddo
      else ! `automatic`, i.e. called from a minimization routine
! orientation atoms
       do i=1,norient
        if (fixed_o(i)) then ! grab coordinates from main coordinate array
         j=iatom_o(i)
         rcurrent_o(i,1)=x(j)
         rcurrent_o(i,2)=y(j)
         rcurrent_o(i,3)=z(j);
        else ! grab coordinates provided by minimizer
         j=iatom_free_o(i) ! x-index (y- z- indices follow)
         rcurrent_o(i,1)=var(j);j=j+1
         rcurrent_o(i,2)=var(j);j=j+1
         rcurrent_o(i,3)=var(j)
        endif
       enddo
! moving atoms
       do i=1,nmove
! here, we should be able to assume that the moving atom coords are passed in
        j=iatom_free_m(i) ! x-index (y- z- indices follow)
        rcurrent_m(i,1)=var(j);j=j+1
        rcurrent_m(i,2)=var(j);j=j+1
        rcurrent_m(i,3)=var(j)
       enddo
      endif ! qmanual
!
      if (orient.eq.1) then
! translate to centroid
       rcurrent_com=matmul(transpose(rcurrent_o), orientWeights)
!
       rcurrent_m(:,1)=rcurrent_m(:,1)-rcurrent_com(1)
       rcurrent_m(:,2)=rcurrent_m(:,2)-rcurrent_com(2)
       rcurrent_m(:,3)=rcurrent_m(:,3)-rcurrent_com(3)
!
       rcurrent_o(:,1)=rcurrent_o(:,1)-rcurrent_com(1)
       rcurrent_o(:,2)=rcurrent_o(:,2)-rcurrent_com(2)
       rcurrent_o(:,3)=rcurrent_o(:,3)-rcurrent_com(3)
      endif
!
! statistics: save current structure
      if (stat_initialized) then
        if (output_dsdt.or.output_rmsd_ave) then
! also need to deal with fixed atoms
         do i=1,nstat
          if (qmanual.or.fixed_s(i)) then ! grab coordinates from main coordinate array
           j=iatom_s(i)
           rold_s(i,1)=x(j)
           rold_s(i,2)=y(j)
           rold_s(i,3)=z(j)
          else
           j=iatom_free_s(i) ! x-index (y- z- indices follow)
           rold_s(i,1)=var(j);j=j+1
           rold_s(i,2)=var(j);j=j+1
           rold_s(i,3)=var(j)
          endif
         enddo
! stat orientation atoms
         if (qstat_orient) then
          rold_o=rcurrent_o ! COM-free
          rold_s(:,1)=rold_s(:,1) - rcurrent_com(1)
          rold_s(:,2)=rold_s(:,2) - rcurrent_com(2)
          rold_s(:,3)=rold_s(:,3) - rcurrent_com(3)
         endif
        endif ! dsdt or rmsd_ave
! update running average
        if (output_rmsd_ave) then
         t=1.0d0*num_average_samples/(num_average_samples+1)
         if (qstat_orient) then
          if (num_average_samples.gt.0) then
           call RMSBestFit(rold_o, rave_o, orientWeights, u)
           u=transpose(u)
           rold_o=matmul(rold_o, u)
           rold_s=matmul(rold_s, u)
          endif
          rave_o=t*rave_o+(1.0d0-t)*rold_o
         endif ! qstat_orient
         rave_s=t*rave_s+(1.0d0-t)*rold_s
         num_average_samples=num_average_samples+1
        endif
      endif
!
      if (qroot) then
       if (orient.eq.1) then
!ccccccccccc take care of orientation ccccccc
! send/receive orientation structure
        me=ME_STRNG
        ncpu=SIZE_STRNG
        if (me.gt.0) then
         call mpi_recv(rref_o,3*norient,mpifloat,me-1,1, &
     & MPI_COMM_STRNG, stat,ierror)
! orient rcurrent based on rref
! 12.09: using RTMD orientation routines
! no checking for undefined coordinates here
         call RMSBestFit(rcurrent_o,rref_o,orientWeights,u)
! transform current structure to overlap with reference
! (if orientation is off, u=I)
         u=transpose(u)
         rcurrent_o=matmul(rcurrent_o, u)
         rcurrent_m=matmul(rcurrent_m, u)
        endif
        if (me.lt.ncpu-1) then
         call mpi_send(rcurrent_o,norient*3,mpifloat,me+1,1, &
     & MPI_COMM_STRNG, ierror)
        endif ! me
       endif ! orient
!cccccccccccccc now call the appropriate interpolation subroutine
       if (repa_mass.eq.1) then
        weights(:,1)=repaWeights(:)! need 3 sets for x-,y-,z- coords
        weights(:,2)=repaWeights(:)
        weights(:,3)=repaWeights(:)
       else
        weights=1.0d0
       endif
!
! call by name
       if (interp_method.eq.linear_exact) then
        call interp_linear_exact(RIN=rcurrent_m,ROUT=rcurrent_m, &
     & WGT=weights,N=3*nmove, D_ARCLENGTH=ds,CURVATURE=curv)
       else
        call interp_driver_sci(RIN=rcurrent_m,ROUT=rcurrent_m, &
     & WGT=weights,N=3*nmove, INTERP_METHOD=interp_method,TOL=def, &
     & MAX_ITERATIONS=iterations,D_ARCLENGTH=ds,CURVATURE=curv, &
     & DST_CUTOFF=dst_cutoff)
       endif
!
       if (orient.eq.1) then
        u=transpose(u)
        rcurrent_m=matmul(rcurrent_m, u) ! rotate back
! restore original COM
        rcurrent_m(:,1)=rcurrent_m(:,1)+rcurrent_com(1)
        rcurrent_m(:,2)=rcurrent_m(:,2)+rcurrent_com(2)
        rcurrent_m(:,3)=rcurrent_m(:,3)+rcurrent_com(3)
!
       endif ! orient
      endif ! root
!
! broadcast coordinates to slaves
      if (qslave) then
#if (KEY_SINGLE==1)
       call PSND4r(rcurrent_m,nmove*3)
#endif
#if (KEY_SINGLE==0)
       call PSND8m(rcurrent_m,nmove*3)
#endif
      endif
!
! copy back moving atoms
!
      if (qmanual) then
       do i=1,nmove
        j=iatom_m(i)
        x(j)=rcurrent_m(i,1)
        y(j)=rcurrent_m(i,2)
        z(j)=rcurrent_m(i,3)
       enddo
      else
       do i=1,nmove
        if (fixed_m(i)) cycle ! do not return fixed atoms (should not be used)
        j=iatom_free_m(i) ! x-index (y- z- indices follow)
        var(j)=rcurrent_m(i,1);j=j+1
        var(j)=rcurrent_m(i,2);j=j+1
        var(j)=rcurrent_m(i,3)
       enddo
      endif
!
      end subroutine sm0k_repa
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine sm0k_confcons(from_sd_,var) ! optionally, accepts variables directly from sd minimizer; no fixed atom support yet
! conformational consistency check within SM0K
!
      use mpi
      use stream
      use multicom_aux;
      use coord; use coordc
      use psf
!
      use confcons, only : confcons_check
      use sm_config, only : string_noprint
!
      implicit none
!
      logical, optional :: from_sd_
      logical :: from_sd
      real(chm_real), optional, intent(inout) :: var(*)
!
      real(chm_real), pointer :: rref_(:,:), r_(:,:)
!
      logical :: qroot, qprint
      integer :: ierror, errnum, errors(SIZE_STRNG), i, j
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
      integer :: stat(MPI_STATUS_SIZE)
!
!
      character(len=len("SM0K_CONFCONS>") ),parameter::whoami="SM0K_CONFCONS>";!macro
!
      if (.not.sm0k_initialized) call sm0k_init()
!
      qroot =MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.gt.1
      qprint=MPI_COMM_STRNG.ne.MPI_COMM_NULL
      qprint=qprint.and.(ME_STRNG.eq.0)
!
!
      if (present(from_sd_)) then ; from_sd=from_sd_ ; else ; from_sd=.false. ; endif
      if (from_sd) then
       if (present(var)) then
        allocate(r_(natom,3))
        j=1
        do i=1,natom
         r_(i,1)=var(j);j=j+1
         r_(i,2)=var(j);j=j+1
         r_(i,3)=var(j);j=j+1
        enddo
       else
       call wrndie(0,whoami,trim(' NO S/D COORDINATES FOUND. ABORT.'))
       return
       endif
      endif ! from_sd
!
      allocate(rref_(natom,3))
!
      if (mestring.gt.0) then
! receive reference structure
       if (qroot) call mpi_recv(rref_,3*natom,mpifloat,mestring-1,1,MPI_COMM_STRNG,stat,ierror)
! call confcons -- all processors enter
       if (from_sd) then
        errnum=confcons_check( r__=r_, rref__=rref_ )
       else
        errnum=confcons_check( rref__=rref_ )
       endif
      endif ! mestring>0
!
      if (mestring.lt.nstring-1) then
       if (qroot) then
        if (from_sd) then
         rref_=r_
        else
         rref_(:,1)=x(1:natom) ; rref_(:,2)=y(1:natom) ; rref_(:,3)=z(1:natom)
        endif
! send reference structure
         call mpi_send(rref_,3*natom,mpifloat,mestring+1,1,MPI_COMM_STRNG, ierror)
       endif ! qroot
      endif ! mestring
!
! collect number of errors into array
!
      if (qroot) call mpi_gather(errnum,1,mpiint,errors,1,mpiint,0,MPI_COMM_STRNG,ierror)
! print errors summary
      if (qprint.and..not.string_noprint) then
       write(sm0k_info,'(2A)') whoami, '___________________________________________________________',&
& whoami, ' REPLICA, NUMBER OF INCONSISTENCIES',&
& whoami, '___________________________________________________________'
       if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
       do i=2, SIZE_STRNG
        write(sm0k_info,'(A,I5,", ",I5)') whoami, i, errors(i)
       if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
       enddo
       write(sm0k_info,'(2A)') whoami, '___________________________________________________________'
       write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
      endif
!
      if (from_sd) then
       j=1
       do i=1,natom
        var(j)=r_(i,1);j=j+1
        var(j)=r_(i,2);j=j+1
        var(j)=r_(i,3);j=j+1
       enddo
      endif
!
      if(associated(rref_))deallocate(rref_)
      if(associated(r_))deallocate(r_)
!
      end subroutine sm0k_confcons
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine sm0k_chirality(from_sd_,var) ! optionally, accepts variables directly from sd minimizer; no fixed atom support yet
! this is a plain wrapper around the chirality checker routine
!
      use stream
      use mpi
      use multicom_aux;
      use coord; use coordc
      use psf
!
      use chirality, only : chirality_check
      use sm_config, only : string_noprint
!
      implicit none
!
      logical, optional :: from_sd_
      logical :: from_sd
      real(chm_real), optional, intent(inout) :: var(*)
!
      real(chm_real), pointer :: r_(:,:)
!
      logical :: qroot, qprint
      integer :: ierror, errnum, errors(SIZE_STRNG), i, j
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
!
      character(len=len("SM0K_CHIRALITY>") ),parameter::whoami="SM0K_CHIRALITY>";!macro
!
      if (.not.sm0k_initialized) call sm0k_init()
!
      qroot =MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.gt.1
      qprint=MPI_COMM_STRNG.ne.MPI_COMM_NULL
      qprint=qprint.and.(ME_STRNG.eq.0)
!
!
      if (present(from_sd_)) then ; from_sd=from_sd_ ; else ; from_sd=.false. ; endif
      if (from_sd) then
       if (present(var)) then
        allocate(r_(natom,3))
        j=1
        do i=1,natom
         r_(i,1)=var(j);j=j+1
         r_(i,2)=var(j);j=j+1
         r_(i,3)=var(j);j=j+1
        enddo
        errnum=chirality_check( r__=r_ )
       else
       call wrndie(0,whoami,trim(' NO S/D COORDINATES FOUND. ABORT.'))
       return
       endif
      else
        errnum=chirality_check()
      endif ! from_sd
!
       if (from_sd) then
       else
       endif
!
! collect number of errors into array
!
      if (qroot) call mpi_gather(errnum,1,mpiint,errors,1,mpiint,0,MPI_COMM_STRNG,ierror)
! print errors summary
      if (qprint.and..not.string_noprint) then
       write(sm0k_info,'(2A)') whoami, '___________________________________________________________',&
& whoami, ' REPLICA, NUMBER OF ERRORS',&
& whoami, '___________________________________________________________'
       if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
       do i=1, SIZE_STRNG
        write(sm0k_info,'(A,I5,", ",I5)') whoami, i, errors(i)
       if(prnlev.ge. 3) write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
       enddo
       write(sm0k_info,'(2A)') whoami, '___________________________________________________________'
       write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
      endif
!
      if (from_sd) then
       j=1
       do i=1,natom
        var(j)=r_(i,1);j=j+1
        var(j)=r_(i,2);j=j+1
        var(j)=r_(i,3);j=j+1
       enddo
      endif
!
      if(associated(r_))deallocate(r_)
!
      end subroutine sm0k_chirality
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine sm0k_interpolate(comlyn, comlen)
! given a collection of n string replicas, interpolate onto a finer/coarser path
!
      use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3
      use stream
      use mpi
      use string
      use dimens_fcm
      use number
!
!
!
#if (KEY_MULTICOM==1)
      use multicom_aux;
#endif
      use select, only : selcta, selrpn, nselct; use psf
      use cvio, only : writcv, readcv
      use coorio_mod, only : cwrite, cread
      use ctitla

      implicit none
      character(len=len("SM0K_INTERPOLATE>") ),parameter::whoami="SM0K_INTERPOLATE>";!macro
!
      CHARACTER(LEN=*) :: COMLYN
      integer :: COMLEN
!
! local declarations
      integer, parameter :: linear=1, spline=2, bspline=3
      integer :: ifile, ofile, num_rep_in, num_rep_out
      integer :: len_cor_in, len_cor_out, &
     & length, mlen, interp_method
      integer :: ibegin, istep
      integer :: i,j,k, orient=0, repa_mass=0, orient_mass=0, norient
!
      real(chm_real), pointer :: rin_all(:,:,:), dr(:,:,:), rout_all(:,:,:)
      real(chm_real), pointer :: rr(:),rr_out(:),ds(:),s(:),t(:),rrpp(:)
      real(chm_real) :: dum
!
      character(len=80), pointer :: fname_cor_in(:), fname_cor_out(:)
      character(len=80) :: name_cor_in, name_cor_out, dummy
      character(len=20) :: methods(4), method, form
      character(len=8) :: keyword
!
      logical :: qdcd, qcor, qbase, qlist , qprint
!
!
      integer :: moder, modew, imode
      logical :: lresid=.false.
! compatibility variables for coordinate reading/writing
      real(chm_real) :: xdum(natom+1), ydum(natom+1), zdum(natom+1), &
     & wdum(natom+1)
      integer :: icntrl(20)
      integer :: iselct(natom), ifreea(natom), pairs(2,natom)
      character, parameter :: tab=char(9)
!================== dcd reading v
      character(len=80) :: title(maxtit)
      real*4 :: trash4(natom) ! scratch array for ugly routine
      real(chm_real) :: trash8(natom) ! scratch array for ugly routine
! some dummy vars for coordinate read
      integer :: nfile, istats, ndof, begin_, stop_, istep_, &
     & skip_, nsavv_, satoms, ntitle
      real(chm_real) :: delta
      logical :: qdim4, qcg
!================== dcd reading ^
!
      real(chm_real), pointer :: orient_weights(:), weight(:)
      integer :: ierror, offset
      real(chm_real) :: u(3,3) ! rotation matrix
      real(chm_real) :: r_com(3)=(/0d0, 0d0, 0d0/)
      integer, pointer :: stringatoms(:)
!
!
      interface ! to linear interpolation routine
       subroutine linear_interp(xin,yin,nin,xout,yout,nout,dydxout)
       use stream
       use chm_kinds
       implicit none
       integer :: nin, nout
       real(chm_real) :: xin(nin), yin(nin), xout(nout), yout(nout)
       real(chm_real), optional :: dydxout(nout) ! tangent computation
       real(chm_real) :: dydx(nout)
       end subroutine linear_interp
      end interface
!
      data methods/ 'LINEAR','CUBIC SPLINE','B-SPLINE','DST'/
!
      if (.not.sm0k_initialized) call sm0k_init()
!
      qprint=MPI_COMM_STRNG.ne.MPI_COMM_NULL
      qprint=qprint.and.(ME_STRNG.eq.0)
! get interpolation specifications
! interpolation type
!
      interp_method=0
      method=gtrma(comlyn, comlen, 'METH')
      length=len(method)
      call trima(method, length)
      if (length.ge.4) then
       if (( method(1:4).eq.'LINE'(1:4) )) then
        interp_method=linear
       elseif (( method(1:4).eq.'BSPL'(1:4) )) then
        interp_method=bspline
       elseif (( method(1:4).eq.'SPLI'(1:4) )) then
        interp_method=spline
       endif
      endif
! print summary
      if (qprint) then
       if (interp_method.gt.0) then
        length=len(methods(interp_method))
call trima(methods(interp_method), length)
        write(sm0k_info,6770) whoami, methods(interp_method)(1:length) ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
 6770 format(/A,' WILL INTERPOLATE USING ',A,' INTERPOLATION')
       else
        if (length.gt.0) then
         write(sm0k_info,6771) whoami, method(1:length), whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
 6771 format(/A,' UNRECOGNIZED INTERPOLATION METHOD: ',A,'.',/, &
     & A, ' WILL INTERPOLATE USING LINEAR INTERPOLATION')
        else
         write(sm0k_info,6772) whoami, whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
 6772 format(/A,' UNSPECIFIED INTERPOLATION METHOD.',/, &
     & A, ' WILL INTERPOLATE USING LINEAR INTERPOLATION')
        endif ! length
       endif ! interp_method
      endif ! qprint
      if (interp_method.eq.0) interp_method=linear ! choose linear interpolation as default
! process other options ccccccccccccccccccccccccccccccccccccccccccccccc
! number of input replicas
      if (indx(comlyn, comlen, 'NIN', 3).gt.0) then
       num_rep_in=gtrmi(comlyn, comlen, 'NIN', 0)
       if (num_rep_in.le.0) then
        if (qprint) then ; write(sm0k_info, 6781) whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info=''; ; endif
 6781 format(A,' NUMBER OF INPUT REPLICAS MUST BE > 0. NOTHING DONE.')
        return
       else
        if (qprint) then
          write(sm0k_info,6783) whoami, num_rep_in ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        endif
 6783 format(A,' INITIAL STRING RESOLUTION: ', I5, ' REPLICAS.')
       endif ! num_rep_in<=0
      else
        if (qprint) then ; write(sm0k_info, 6784) whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info=''; ; endif
 6784 format(A,' NUMBER OF INPUT REPLICAS UNSPECIFIED NOTHING DONE.')
         return
      endif ! indx('NIN')
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! number of output replicas
      if (indx(comlyn, comlen, 'NOUT', 4).gt.0) then
       num_rep_out=gtrmi(comlyn, comlen, 'NOUT', 0)
       if (num_rep_out.le.0) then
        if (qprint) then ; write(sm0k_info, 6782) whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info=''; ; endif
 6782 format(A,' NUMBER OF OUTPUT REPLICAS MUST BE > 0. NOTHING DONE.')
        return
       else
        if (qprint) then
          write(sm0k_info,6785) whoami, num_rep_out ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        endif
 6785 format(A,' INTERPOLATED STRING RESOLUTION: ', I5, ' REPLICAS.')
       endif ! num_rep_in<=0
      else
        if (qprint) then ; write(sm0k_info, 6786) whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info=''; ; endif
 6786 format(A,' NUMBER OF OUTPUT REPLICAS UNSPECIFIED NOTHING DONE.')
         return
      endif ! indx('NIN')
!=======================================================================
! get input coordinate file sm0k_info
      qcor=indx(comlyn, comlen, 'CRIN', 4).gt.0
      qdcd=indx(comlyn, comlen, 'DCDIN', 5).gt.0
!
      if (qcor) then
       if (qdcd) then ; call wrndie(0,whoami,trim("CANNOT SPECIFY BOTH 'CRIN' and 'DCDIN'. NOTHING DONE"));
        i=indxa(comlyn, comlen, 'CRIN');i=indxa(comlyn, comlen, 'DCDIN'); return;
       endif
       keyword='CRIN' ; i=4;
      elseif (qdcd) then
       keyword='DCDIN' ; i=5;
      endif
      len_cor_in=-1
!
      if (qcor .or. qdcd) then ; call gtrmwa(comlyn, comlen, keyword, i, name_cor_in, 80, len_cor_in); endif
      if (len_cor_in.le.0) then
       if (qprint) then ; write(sm0k_info, 6789) whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info=''; ; endif
 6789 format(A,' INPUT COORDINATE FILE NAME UNSPECIFIED. NOTHING DONE.')
       return
      endif
!
      if (qdcd) then ! parse trajectory options
       ibegin = gtrmi(comlyn, comlen, 'BEGI', 1);
       istep = gtrmi(comlyn, comlen, 'STEP', 1);
       if (ibegin.lt.1) then
        call wrndie(0,whoami,trim('BEGIN LESS THAN 1. RESETTING TO 1')); ibegin=1
       endif
!
       if (istep.lt.1) then
        call wrndie(0,whoami,trim('STEP LESS THAN 1. RESETTING TO 1')); istep=1
       endif
      endif
!
! get output coordinate file sm0k_info
      qlist=indx(comlyn, comlen, 'OUTLIST', 7).gt.0 ! formatted file with explicit filenames
      qbase=indx(comlyn, comlen, 'OUTBASE', 7).gt.0 ! base name with index appended
!===========================
      if (qlist) then
       if (qbase) then ; call wrndie(0,whoami,trim("CANNOT SPECIFY BOTH 'OUTLIST' and 'OUTBASE'. NOTHING DONE"));
        i=indxa(comlyn, comlen, 'OUTLIST');i=indxa(comlyn, comlen, 'OUTBASE'); return;
       endif
       keyword='OUTLIST' ; i=7;
      elseif (qbase) then
       keyword='OUTBASE' ; i=7;
      endif
      len_cor_out=-1
      if (qlist.or.qbase) then ; call gtrmwa(comlyn, comlen, keyword, i, name_cor_out, 80, len_cor_out) ; endif
      if (len_cor_out.le.0) then
       if (qprint) then ; write(sm0k_info, 6790) whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info=''; ; endif
 6790 format(A,' OUTPUT COORDINATE FILE NAME UNSPECIFIED. NOTHING DONE.')
       return
      endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! parse file format spec. (same for both input/output)c
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! when reading from a dcd file, the formatting below applies only to the output files
      lresid=.false.
      if (indxa(comlyn, comlen, 'PDB').gt.0) then
        moder=-1
        modew=4
        if (indxa(comlyn, comlen, 'RESI').gt.0) lresid=.true.
        form='FORMATTED'
      elseif ( (indxa(comlyn, comlen, 'FILE').gt.0).or. &
     & (indxa(comlyn, comlen, 'UNFO').gt.0)) then
        moder=0
        modew=1
        form='UNFORMATTED'
      elseif ( (indxa(comlyn, comlen, 'CARD').gt.0).or. &
     & (indxa(comlyn, comlen, 'FORM').gt.0)) then
        moder=1
        modew=2
        form='FORMATTED'
      else ! default
        moder=1
        modew=2
        form='FORMATTED'
      endif
!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! write summary ( same code as in SMCV interpolation )
      if (qprint) then ! note: using qprint as a root node flag, too (should change this)
!ccccc get coordinate file names
       ifile=-1 ! a valid unit number will be assigned by __OPEN_FILE
       ofile=-1
!==========================
       if (qdcd) then
        call open_file(ifile, name_cor_in(1:len_cor_in), 'UNFORMATTED', 'READ')
         write(sm0k_info,6791) whoami, name_cor_in(1:len_cor_in) ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
         write(sm0k_info,67910) whoami, itoa(ibegin), itoa(istep); write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
 6791 format(A,' COORDINATE SETS WILL BE READ FROM THE FILE ',A )
 67910 format(A,' STARTING AT FRAME ',A,' WITH STRIDE OF ',A,'.')
       else
        call open_file(ifile, name_cor_in(1:len_cor_in), 'FORMATTED', 'READ')
        allocate(fname_cor_in(num_rep_in))
!
        do j=1, num_rep_in
         read(ifile,'(A80)') fname_cor_in(j)
        enddo
        call VCLOSE(ifile, 'KEEP', ierror)
!
         write(sm0k_info,6792) whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
 6792 format(A,' COORDINATE SETS WILL BE READ FROM THE FOLLOWING FILES:' )
!
        do j=1, num_rep_in
         write(sm0k_info,'(A1,I5," ",A80)') tab, j, fname_cor_in(j) ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        enddo
       endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       allocate(fname_cor_out(num_rep_out))
       if (qlist) then
        call open_file(ofile, name_cor_out(1:len_cor_out), 'FORMATTED', 'READ')
        do j=1, num_rep_out
         read(ofile,'(A80)') fname_cor_out(j)
        enddo
        call VCLOSE(ofile, 'KEEP', ierror)
!============================================================
       elseif (qbase) then
        offset=gtrmi(comlyn, comlen, 'OUTI', 0); ! output coordinate name offset
        do j=1, num_rep_out
         k=min(len_cor_out, len(fname_cor_out(j)) - (len(itoa(num_rep_out-1))+4) ) ! determine maximum length to avoid buffer overrun
         fname_cor_out(j)=name_cor_out(1:k)//itoa(j-1+offset)//'.cor'
        enddo
       endif ! qlist
!
       write(sm0k_info,6793) whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
 6793 format(A,' COORDINATE SETS WILL BE WRITTEN TO THE FOLLOWING FILES:' )
!
       do j=1, num_rep_out
        write(sm0k_info,'(A1,I5," ",A80)') tab, j, fname_cor_out(j) ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
       enddo
!
      endif ! qprint
!
!cccccccccccccccccccccc parse weighting/orientation options
! check for orientation options: must appear after 'ORIE'
! process selection
      i=indxa(comlyn, comlen, 'ORIE')
      if (i.gt.0) then ! only if the ORIE directive exists
! selection text taken from corman.src
       orient=1
       j=indx(comlyn, comlen, 'SELE', 4)
       if (j.gt.0.and.j.lt.i) then ! sele occurs before orie
        call wrndie(0,whoami,trim('ATOM SELECTION MUST BE SPECIFIED AFTER ORIE.'))
         return
       endif
!
!
       IMODE=0
       CALL SELRPN(COMLYN,COMLEN,ISELCT,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,xdum,ydum,zdum,.TRUE.,1,wdum)
       IF(IMODE.NE.0) THEN
        CALL WRNDIE(0,whoami,'ATOM SELECTION ERROR')
        RETURN
       ENDIF
       norient=count(iselct.gt.0)
!
!
       if (norient.eq.0) then
        call wrndie(0,whoami,trim('NO ATOMS SELECTED FOR ORIENTATION. WILL NOT ORIENT.'))
        orient=0 ! invalid or missing selection for orientation
       elseif (norient.lt.3) then
        call wrndie(0,whoami,trim('FEWER THAN FOUR ATOMS SELECTED FOR ORIENTATION. WILL NOT ORIENT.'))
        orient=0
       endif
!
      endif ! orie specified
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! check for mass weighting
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      j=1
      do while (j.gt.0)
       mlen=comlen
       j=indxa(comlyn, comlen, 'MASS') ! s
       if ( (orient.eq.1).and.(j.ge.i) ) then
        orient_mass=1
       else if (j.gt.0) then
        repa_mass=1
        i=i-(mlen-comlen) ! how much the string has shrunk
       endif
      enddo
!
!
! root node performs the interpolation
! however, other nodes are included for compatibility with CHARMM routines
!
      if (qprint) then ! using qprint as a root flag -- not great
!
       if (repa_mass.eq.1) then ; write(sm0k_info,6720) whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info=''; ; endif
 6720 format(A,' INTERPOLATION WILL USE MASS-WEIGHTING')
!
       if (orient.eq.1) then
!
        allocate(orient_weights(natom))
!
        orient_weights=(amass*orient_mass+(1d0-orient_mass))*iselct
! nomalize weights
        dum=sum(orient_weights)
        if (abs(dum).gt.RSMALL) then
         dum=one/dum
         orient_weights=dum*orient_weights
        endif
!
! print a summary
        write(keyword,'(I8)') norient
        mlen=len(keyword)
        call trima(keyword, mlen)
        write(sm0k_info,6700) whoami, keyword(1:mlen) ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
 6700 format(A,' STRING WILL BE ORIENTED BASED ON ',A,' ATOMS')
        if (orient_mass.eq.1) then ; write(sm0k_info,6710) whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info=''; ; endif
 6710 format(A,' ORIENTATION WILL USE MASS-WEIGHTING')
       endif ! orient == 1
!ccccccccccccccccccccccccc do work cccccccccccccccccccccccccccc
       write(sm0k_info,6974) whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
 6974 format(A,' READING COORDINATES')
! allocate memory for ALL replicas (may not be the best solution if
! the structure is very large; however, with 0K string, you will want
! to exclude water
!
      endif ! qprint
! coordinate arrays input & output
!
      if(associated(rin_all))deallocate(rin_all)
      allocate(rin_all(natom,3,num_rep_in))
      rin_all=anum ! initialize
! allocate dr to use for dummy coords below
      allocate(dr(natom,3,1))
!
      if(associated(rout_all))deallocate(rout_all)
      allocate(rout_all(natom,3,num_rep_out))
      rout_all=anum
!
      iselct=1. ! CHARMM compatibility
!
      if (qdcd) then
! most of the code lifted from ftsm
       if (qprint) then
!
! call trajectory reader
!
        istats=1
        qcg=.false.
        qdim4=.false.
        begin_=0 ! note begin <=0 forces a strange "reset" with begin=istep (which is zero below); this is to support trajectories
                 ! made with VMD
        skip_=1
        stop_=0
        ntitle=0
        istep_=0
        allocate(stringatoms(natom))
        do i=1, ibegin-1 ! skip frames
         call readcv(dr(:,1,1), dr(:,2,1), dr(:,3,1), &
#if (KEY_CHEQ==1)
     & trash8, qcg, &
#endif
     & trash4, natom, &
     & stringatoms, satoms, ifile, 1, ifile, nfile, &
     & istep_, istats, ndof, delta, begin_, stop_, skip_, &
     & nsavv_, 'CORD', 'CORD', title, ntitle, qdim4, trash8, .false.)
! fix stop value (ad hoc)
         if (i.eq.1) then ; stop_ = begin_ + skip_ * ( ibegin - 1 + num_rep_in * istep ) ; if (istats.lt.0) istats=2; endif
!
        enddo ! skip frames
       endif ! ME_STRNG==0
      endif
!
      do j=1, num_rep_in ! input coordinate sets
!
       if (qdcd) then
        call readcv(rin_all(:,1,j), rin_all(:,2,j), rin_all(:,3,j), &
#if (KEY_CHEQ==1)
     & trash8, qcg, &
#endif
     & trash4, natom, &
     & stringatoms, satoms, ifile, 1, ifile, nfile, &
     & istep_, istats, ndof, delta, begin_, stop_, skip_, &
     & nsavv_, 'CORD', 'CORD', title, ntitle, qdim4, trash8, .false.)
! fix stop value (ad hoc)
        if (j+ibegin.eq.2) then ; stop_ = begin_ + skip_ * ( ibegin - 1 + num_rep_in * istep ) ; if (istats.lt.0) istats=2; endif
! write(0,*) j+ibegin, stop_, begin_, skip_, istep, ibegin, skip_ * ( ibegin - 1 + num_rep_in * istep )
!====================================================================================
! skip requested frames (stride), unless we just read the final frame we need
        if(j.lt.num_rep_in) then
         do k=1, istep-1
           call readcv(dr(:,1,1), dr(:,2,1), dr(:,3,1), &
#if (KEY_CHEQ==1)
     & trash8, qcg, &
#endif
     & trash4, natom, &
     & stringatoms, satoms, ifile, 1, ifile, nfile, &
     & istep_, istats, ndof, delta, begin_, stop_, skip_, &
     & nsavv_, 'CORD', 'CORD', title, ntitle, qdim4, trash8, .false.)
         enddo
        endif ! j not last
!====================================================================================
!
       else ! coordinate files
!
        if (qprint) then
         length=len_trim(fname_cor_in(j))
         dummy=fname_cor_in(j)(1:length)
         ifile=-1 ; call open_file(ifile, dummy, form, 'READ')
        endif
!
        dummy=''
        call cread(ifile, titleb, ntitlb, icntrl, &
     & rin_all(:,1,j), rin_all(:,2,j), rin_all(:,3,j), &
     & wdum, natom, moder, iselct, &
     & 0, res, nres, atype, ibase, 1, ifreea, &
     & segid, resid, nictot, nseg, lresid, .false., &
     & dummy, 80, 0, .false.)
        if (qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ; endif
       endif ! qdcd
      enddo
!
      if (qdcd.and.qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ; endif ! close dcd file
      if(associated(stringatoms))deallocate(stringatoms)
!
! check for undefined coordinates
      if (qprint) then
       if (any(rin_all.eq.anum)) then
        call wrndie(0,whoami,trim('WARNING: SOME INPUT COORDINATES ARE UNDEFINED AFTER READING.'))
       endif
      endif
!========================================================================================
! orient structures if requested
!
      if (qprint) then
        if (orient.eq.1) then
!
! translate first set to centroid
!
        j=1
        r_com=matmul(orient_weights, rin_all(:,:,j))
        rin_all(:,1,j)=rin_all(:,1,j)-r_com(1)
        rin_all(:,2,j)=rin_all(:,2,j)-r_com(2)
        rin_all(:,3,j)=rin_all(:,3,j)-r_com(3)
!
! set up pairs
! pairs = RESHAPE( (/ (i, i=1,natom), &
! & (i, i=1,natom) /), (/ 2,natom /), order=(/ 2,1 /) )
! orient x based on the previous replica
         do j=2,num_rep_in
! translate next set to centroid
          r_com=matmul(orient_weights, rin_all(:,:,j))
          rin_all(:,1,j)=rin_all(:,1,j)-r_com(1)
          rin_all(:,2,j)=rin_all(:,2,j)-r_com(2)
          rin_all(:,3,j)=rin_all(:,3,j)-r_com(3)
!
          call RMSBestFit(rin_all(:,:,j-1), rin_all(:,:,j), orient_weights, u) ! superpose r_(j-i) onto r_j
          rin_all(:,:,j)=matmul(rin_all(:,:,j),u) ! apply transpose (=inverse) of u to r_j
! old CHARMM routine:
! call rotls1( &
! & rin_all(1,1,j-1), rin_all(1,2,j-1), rin_all(1,3,j-1), &
! & rin_all(1,1,j), rin_all(1,2,j), rin_all(1,3,j), &
! & natom,pairs,natom, &
! & orient_weights, &
! & .false.,.false.)
         enddo ! j
        endif ! orient
!
        allocate(weight(natom))
!
        if (repa_mass.eq.1) then
         weight=amass(1:natom)
        else
         weight=1d0
        endif
! nomalize weights
        dum=sum(weight)
        if (abs(dum).gt.RSMALL) then
         dum=one/dum
         weight=dum*weight
        endif
!
! do the actual interpolation -- simple, not self-consistent
! allocate memory
        if(associated(dr))deallocate(dr); allocate(dr(natom,3,num_rep_in-1))
        if(associated(ds))deallocate(ds); allocate(ds(num_rep_in-1))
        if(associated(s))deallocate(s); allocate(s(num_rep_in))
        if(associated(t))deallocate(t); allocate(t(num_rep_out))
        if(associated(rr))deallocate(rr); allocate(rr(num_rep_in))
        if(associated(rr_out))deallocate(rr_out); allocate(rr_out(num_rep_out))
        if(associated(rrpp))deallocate(rrpp); allocate(rrpp(num_rep_in))
!
! compute arclength
        dr=rin_all(:,:,2:num_rep_in)-rin_all(:,:,1:num_rep_in-1)
        s(1)=0
        do i=1,num_rep_in-1
         ds(i)=sqrt(sum(matmul( (transpose(dr(:,:,i)))**2, weight**2)))
         s(i+1)=s(i)+ds(i)
        enddo
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! normalize arclength
        do i=1,num_rep_in
         s(i)=s(i)/s(num_rep_in)
        enddo
!ccccccccccccccccccccccccc
! create uniform array
        do i=1,num_rep_out
         t(i)=1.0d0*(i-1)/(num_rep_out-1)
        enddo
!cccccccccccccc now interpolate variables cccccc
        if (interp_method.eq.spline) then
         do i=1,natom
          do j=1,3
           rr=rin_all(i,j,:)
           call spline_cubic_set(num_rep_in,s,rr,0,0,0,0,rrpp)
           do k=1,num_rep_out
            call spline_cubic_val(num_rep_in,s,rr,rrpp,t(k), &
     & rout_all(i,j,k),dum,dum)
           enddo
          enddo
         enddo
        elseif (interp_method.eq.bspline) then
         do i=1,natom
          do j=1,3
           rr=rin_all(i,j,:)
           do k=1,num_rep_out
            call spline_b_val(num_rep_in,s,rr,t(k),rout_all(i,j,k))
           enddo
          enddo
         enddo
        elseif (interp_method.eq.linear) then
         do i=1,natom
          do j=1,3
           rr=rin_all(i,j,:)
           call linear_interp(s,rr,num_rep_in,t,rr_out,num_rep_out)
           rout_all(i,j,:)=rr_out
          enddo
         enddo
        endif ! interp_method
! check for undefined coordinates
        if (any(rout_all.eq.anum)) then
         call wrndie(0,whoami,trim('WARNING: SOME COORDINATES ARE UNDEFINED AFTER INTERPOLATION.'))
        endif
!
!cccccccccccc write file cccccccccccccccccccccccc
!
        wdum=1. ! reset weighting array (can modify this if needed)
!
        ofile=-1 ! open_file will assign a unit #
        do j=1,num_rep_out
!cccccccccccc write new coordinate file
         length=len_trim(fname_cor_out(j))
         dummy=fname_cor_out(j)(1:length)
         call open_file(ofile, dummy, form, 'WRITE')
!
         if (ntitla+1 .lt. maxtit) &
     & write(titlea(ntitla+1),'(A,I5,A,I5)') &
     & '* REPLICA ',j,' OF ',num_rep_out
!
         call cwrite(ofile,TITLEA,min(NTITLA+1,maxtit),icntrl, &
     & rout_all(:,1,j), rout_all(:,2,j), rout_all(:,3,j), &
     & wdum,res,atype,ibase, &
     & NRES,NATOM,iselct,modew,0,0,.false.)
         call VCLOSE(ofile, 'KEEP', ierror)
!
        enddo ! loop over new coordinate sets
!
        deallocate(rr, rr_out, dr, s, t, ds, rrpp)
      endif ! qprint
!
      if(associated(rin_all))deallocate(rin_all)
      if(associated(rout_all))deallocate(rout_all)
      if(associated(fname_cor_in))deallocate(fname_cor_in)
      if(associated(fname_cor_out))deallocate(fname_cor_out)
      if(associated(weight))deallocate(weight)
      if(associated(orient_weights))deallocate(orient_weights)
!
      end subroutine sm0k_interpolate
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine sm0k_stat_init(comlyn, comlen)
! statistics initialization subroutine
      use stream
      use dimens_fcm
      use string
      use coord; use coordc
#if (KEY_MULTICOM==1)
      use multicom_aux;
#endif
      use select, only : selcta, selrpn, nselct; use psf
      use number
! use psf
!
      use energym
!
#if (KEY_TSM==1)
      use tsmh ! TSM common blocks included because minimizer DOF depend on TSM
      use tsms_mod, only:qtsm
#endif
#if (KEY_MPI==1)
      use mpi
#endif
!ccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
!
      CHARACTER(len=*) :: COMLYN
      integer :: COMLEN
!
      character(len=80) :: rform, dform, sform, fcform, cform
      integer :: klen=0, ierror, i, j
      logical :: found
      logical :: qprint, qfree
!
      integer, pointer :: iselct(:)
      integer, pointer :: backlist(:) ! for nominal compatibility with TSM
      integer, pointer :: ifixed(:)
      integer :: imode
!
      real(chm_real) :: d, com(3)
!
      character(len=8) :: keyword
      character(len=len("SM0K_STAT_INIT>") ),parameter::whoami="SM0K_STAT_INIT>";!macro
!
! interface
! function sm0k_fixed_atoms()
! integer, pointer :: sm0k_fixed_atoms(:)
! end function sm0k_fixed_atoms
! end interface
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (.not.sm0k_initialized) call sm0k_init()
!
      qprint=(ME_STRNG.eq.0).and.(MPI_COMM_STRNG.ne.MPI_COMM_NULL)
! begin
! reset iteration counter
! did the user specify it?
      stat_iteration_counter=gtrmi(comlyn, comlen, 'COUN', -1)
      stat_iteration_counter=max(stat_iteration_counter,0)
      if (stat_iteration_counter.gt.0) then
       if (qprint) then ; write(sm0k_info,639) whoami, stat_iteration_counter ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info=''; ; endif
 639 format(A,' SETTING ITERATION COUNTER TO ',I7)
      endif
!
      energy_fname=''
      output_energy=.false.
!
      if (rmsd0_funit.ge.0) then ; call VCLOSE(rmsd0_funit, 'KEEP', ierror) ; endif
      rmsd0_fname=''
      output_rmsd0=.false.
      if (dsdt_funit.ge.0) then ; call VCLOSE(dsdt_funit, 'KEEP', ierror) ; endif
      dsdt_fname=''
      output_dsdt=.false.
!
      if (c_funit.ge.0) then ; call VCLOSE(c_funit, 'KEEP', ierror) ; endif
      c_fname=''
      output_curvature=.false.
!
      if (s_funit.ge.0) then ; call VCLOSE(s_funit, 'KEEP', ierror) ; endif
      s_fname=''
      output_arclength=.false.
!
! memory allocation
      if(associated(fixed_s))deallocate(fixed_s)
      if(associated(iatom_s))deallocate(iatom_s)
      if(associated(iatom_free_s))deallocate(iatom_free_s)
      if(associated(rold_s))deallocate(rold_s)
      if(associated(rave_s))deallocate(rave_s)
      if(associated(rcurrent_s))deallocate(rcurrent_s)
      if(associated(rcomp_s))deallocate(rcomp_s)
      if(associated(statWeights))deallocate(statWeights)
      if(associated(rcomp_o))deallocate(rcomp_o)
      if(associated(rold_o))deallocate(rold_o)
      if(associated(rave_o))deallocate(rave_o)
      nstat=0
      qstat_orient=.false.
!
! first check that the number of fixed atoms hasn`t changed since initialization; if it has, warn and quit
      ifixed=>sm0k_fixed_atoms()
      qfree=size(ifixed).eq.nfix_bckl
      if (qfree) qfree=all(ifixed.eq.iatom_f)
      if (.not.qfree) then
       call wrndie(0,whoami,trim('FIXED ATOM ARRAY CHANGED AFTER LAST STRING COMMAND.'))
       deallocate(iatom_f)
       iatom_f=>ifixed
       nfix_bckl=size(iatom_f)
       nullify(ifixed)
      else
       deallocate(ifixed)
      endif
!
!cccccccccc first process energy terms (ZTS ONLY) ccccccccccccc
      keyword=nexta8(comlyn,comlen)
      if (( keyword(1:4).eq.'ENER'(1:4) )) then
        output_energy=.true.
! get nergy file name
        call gtrmwa(COMLYN, COMLEN, 'ENAM', 4, energy_fname, 80, energy_flen)
!
! get energy file name
! now parse until the 'end' keyword
        keyword=nexta8(comlyn,comlen)
        klen=len(keyword)
        call trima(keyword, klen)
        do while (keyword(1:3).ne.'END'.and.klen.gt.0)
! assume that it is a energy parameter; locate it in the property/term arrays:
!
         found=.false.
! 1) search properties
         i=0
         do while (.not.found.and.i.lt.lenenp)
          i=i+1
! 10/21/2012 : allow extra characters in keyword for backward compatibility (e.g. ANGLes -- the last two ignored)
          j=max(1,min(klen,len_trim(ceprop(i)))) ! empty strings give a false 'found'
          if ( (keyword(1:j).EQ.ceprop(i)(1:j)) ) then
           found=.true. ; klen=j ; exit
          endif
         enddo
!
         if (.not.found) then
! 2) search energy terms
          i=0
          do while (.not.found.and.i.lt.lenent)
           i=i+1
! 10/21/2012 : allow extra characters in keyword for backward compatibility (e.g. ANGLes -- the last two ignored)
           j=max(1,min(klen,len_trim(ceterm(i))))
           if ( (keyword(1:j).EQ.ceterm(i)(1:j)) ) then
            found=.true. ; klen=j ; exit
           endif
          enddo
          i=-i ! this is a 'trick' to keep distinguish between indices into eprop and eterm
         endif
!
         if (.not.found) then ! if still not found, warn
           write(sm0k_info(1),*)'ENERGY VALUE ',KEYWORD(1:klen),' NOT FOUND';call wrndie(0,whoami,trim(sm0k_info(1)))
         else
          num_energy_terms=num_energy_terms+1
          if (num_energy_terms.gt.enmax) then
           call wrndie(0,whoami,trim('NUMBER OF REQUESTED ENERGY TERMS TOO LARGE. ABORTING.'))
           return
          else
           energy_indices(num_energy_terms)=i
           energy_names(num_energy_terms)=keyword(1:klen)
          endif
         endif
! read the next keyword
         keyword=nexta8(comlyn,comlen)
         klen=len(keyword)
         call trima(keyword, klen)
        enddo ! lop over all keywords
!ccccccccccc print summary
        if (qprint) then
         if (energy_flen.gt.0) then
          write(sm0k_info,662 ) whoami,whoami,energy_fname(1:energy_flen)
         else
          write(sm0k_info,663 ) whoami
         endif
         write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
         write(sm0k_info,664) whoami,(energy_names(i),i=1,num_energy_terms)
         write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        endif
 662 format(A,' WILL WRITE STRING ENERGY TO FILES ',/A,' ',A,'[I].DAT,'&
     &, ' AT EACH ITERATION INDEX I.')
 663 format(A,' WILL WRITE STRING ENERGY TO STDOUT.')
 664 format(A,' THE FOLLOWING KEYWORDS', &
     & ' WILL BE USED:',100(' ',A5),/)
!
      endif ! energy terms
!
!cccccccccccccccccccccc deal with RMSD comparison options cccccccccccccccccccccccccccccccc
!
      allocate(iselct(natom), backlist(natom))
      iselct=1 ! select all atoms by default
      backlist=0 ! setting this to zero is the simplest way to provide correct default behavior (see below)
!
!
! is there an atom selection ?
!ccccccccccccccccc first process the RMSD-related commands
      j=indx(comlyn, comlen, 'SELE', 4)
      if (j.gt.0) then
       imode=0
       CALL SELRPN(COMLYN,COMLEN,iselct,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN) !
       IF(IMODE.NE.0) THEN
        CALL WRNDIE(0,whoami,'ATOM SELECTION ERROR')
        RETURN
       ENDIF
      else
       iselct=1
      endif
      nstat=count( iselct(1:natom).gt.0 )
!
!*************************************************************************
      if (nstat.eq.0) then
        call wrndie(0,whoami,trim('NO ATOMS SELECTED FOR RMSD COMPUTATION. WILL NOT COMPUTE RMSD.'))
        output_rmsd0=.false.
        output_dsdt=.false.
        output_rmsd_ave=.false.
      else
! determine whether structures are to be oriented before comparison
       qstat_orient=(indxa(comlyn, comlen, 'ORIE').gt.0)
       if (qstat_orient) then
        if (repa_initialized.and.norient.gt.0) then
         if (qprint) then ; write(sm0k_info,638) whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info=''; ; endif
 638 format(A,' RMSD CALCULATIONS WILL INCLUDE ORIENTATION.')
        else
      call wrndie(0,whoami,trim('REPARAMETRIZATION DISABLED OR NO ORIENTATION ATOMS FOUND. WILL NOT ORIENT.'))
         qstat_orient=.false.
        endif ! repa_initialized
       endif ! qstat_orient
!
!!!!!!!!!!!!!! RMSD from static structure in comp
       if (indxa(comlyn, comlen, 'RMSD').gt.0) then ! request for RMSD
        output_rmsd0=.true.
!
        if (.not.associated(rcurrent_s)) allocate(rcurrent_s(nstat,3))
        if (.not.associated(rcomp_s)) allocate(rcomp_s(nstat,3))
        if (qstat_orient) then
         if (.not.associated(rcomp_o)) allocate(rcomp_o(norient,3))
        endif ! qstat_orient
!
        call gtrmwa(COMLYN, COMLEN, 'RNAM', 4, rmsd0_fname, 80, rmsd0_flen)
        if (rmsd0_flen.eq.0) then
         call wrndie(0,whoami,trim('NO RMSD FILE NAME SPECIFIED. WILL WRITE TO STDOUT.'))
         rmsd0_funit=outu
        else
         if (indxa(comlyn, comlen, 'RAPP').gt.0) then ! APPEND?
           rform='APPEND'
         else
           rform='WRITE'
         endif
         if (qprint) then
          rmsd0_funit=-1
          call open_file(rmsd0_funit, rmsd0_fname, 'FORMATTED', rform)
         endif
        endif
!ccccccccccc print summary
        if (qprint) then
         if (rmsd0_flen.gt.0) then
          write(sm0k_info,660 ) whoami,rmsd0_fname(1:rmsd0_flen)
         else
          write(sm0k_info,661 ) whoami
         endif
         write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        endif
 660 format(A,' WILL WRITE STRING RMSD TO FILE ',A)
 661 format(A,' WILL WRITE STRING RMSD TO STDOUT.')
!
       endif ! RMSD
!!!!!!!!!!!!!! RMSD from structure at the previous step (zts/fts)
       if (indxa(comlyn, comlen, 'DELS').gt.0) then
        output_dsdt=.true.
!
        if (.not.associated(rold_s)) then ; allocate(rold_s(nstat,3)) ; rold_s=zero ; endif ! for storing "old" coords
        if (.not.associated(rcurrent_s)) allocate(rcurrent_s(nstat,3))
        if (qstat_orient) then
         if (.not.associated(rold_o)) then ; allocate(rold_o(norient,3)) ; rold_o=zero ; endif
        endif
!
        call gtrmwa(COMLYN, COMLEN, 'DNAM', 4, dsdt_fname, 80, dsdt_flen)
        if (dsdt_flen.eq.0) then
         call wrndie(0,whoami,trim('NO DELS FILE NAME SPECIFIED. WILL WRITE TO STDOUT.'))
         dsdt_funit=outu
        else
         if (indxa(comlyn, comlen, 'DAPP').gt.0) then ! APPEND?
           dform='APPEND'
         else
           dform='WRITE'
         endif
         if (qprint) then
          dsdt_funit=-1
          call open_file(dsdt_funit, dsdt_fname, 'FORMATTED', dform)
         endif
        endif
!ccccccccccc print summary
        if (qprint) then
         if (dsdt_flen.gt.0) then
          write(sm0k_info,650 ) whoami,dsdt_fname(1:dsdt_flen)
         else
          write(sm0k_info,651 ) whoami
         endif
         write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        endif
 650 format(A,' WILL WRITE STRING RMSD(I,I+1) TO FILE ',A)
 651 format(A,' WILL WRITE STRING RMSD(I,I+1) TO STDOUT.')
!
       endif ! rmsd0
!!!!!!!!!!!!!! RMSD from average structure (zts/fts)
       if (indxa(comlyn, comlen, 'RMSA').gt.0) then
        output_rmsd_ave=.true.
!
        if (.not.associated(rave_s)) allocate(rave_s(nstat,3)) ! for storing average coords
        if (.not.associated(rold_s)) allocate(rold_s(nstat,3)) ! for storing "old" coords
        rave_s=zero
        if (.not.associated(rcurrent_s)) allocate(rcurrent_s(nstat,3))
        if (qstat_orient) then
         if (.not.associated(rold_o)) allocate(rold_o(norient,3))
         if (.not.associated(rave_o)) allocate(rave_o(norient,3))
         rave_o=zero
        endif
!
        call gtrmwa(COMLYN, COMLEN, 'RANM', 4, rmsd_ave_fname, 80, rmsd_ave_flen)
        if (rmsd_ave_flen.eq.0) then
         call wrndie(0,whoami,trim('NO RMSA FILE NAME SPECIFIED. WILL WRITE TO STDOUT.'))
         rmsd_ave_funit=outu
        else
         if (indxa(comlyn, comlen, 'RAAP').gt.0) then ! APPEND?
           rform='APPEND'
         else
           rform='WRITE'
         endif
         if (qprint) then
          rmsd_ave_funit=-1
          call open_file(rmsd_ave_funit, rmsd_ave_fname, 'FORMATTED', rform)
         endif
        endif
!ccccccccccc print summary
        if (qprint) then
         if (rmsd_ave_flen.gt.0) then
          write(sm0k_info,6500 ) whoami,rmsd_ave_fname(1:rmsd_ave_flen)
         else
          write(sm0k_info,6510 ) whoami
         endif
         write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        endif
 6500 format(A,' WILL WRITE STRING RMSD FROM AVERAGE STRUC. TO FILE ',A)
 6510 format(A,' WILL WRITE STRING RMSD FROM AVERAGE STRUC. TO STDOUT.')
       endif ! rmsd_ave
!
       if (output_rmsd_ave.or.output_rmsd0.or.output_dsdt) then
! populate iatom array
        if (.not.associated(fixed_s)) allocate(fixed_s(nstat)) ! psf indices
        if (.not.associated(iatom_s)) allocate(iatom_s(nstat)) ! psf indices
        if (.not.associated(iatom_free_s)) allocate(iatom_free_s(nstat)) ! free indices for atoms with minimizer
        nstat=0
        iatom_s=0; iatom_free_s=0;
!
!
#if (KEY_TSM==1)
        if (qtsm) then
! j=bpert(backls)
! backlist=heap(j:j+natom-1) ! heap must have of integer type (it was when this line was written)
          backlist=backls
        endif
#endif
!
        j=-2 ! free atom index
        do i=1,natom
         qfree=imove(i).eq.0 ! see egrad1.src routines
#if (KEY_TSM==1)
         if (qtsm) qfree=qfree.and.backlist(i).eq.0
#endif
         if (qfree) j=j+3 ! indices increase in increments of 3: 1,4,...
! orientation indices
         if (iselct(i).gt.0) then
          nstat=nstat+1
          iatom_s(nstat)=i
          if (qfree) then
           iatom_free_s(nstat)=j
           fixed_s(nstat)=.false.
          else
           iatom_free_s(nstat)=anum ! unknown index because the minimization routine will not supply the coordinate
           fixed_s(nstat)=.true. ! this flag will tepp sm0k stat not to use free index
          endif
         endif
        enddo ! all atoms
!
!
        if (output_rmsd0) then
         do i=1, nstat
           j=iatom_s(i)
           rcomp_s(i,1)=xcomp(j);
           rcomp_s(i,2)=ycomp(j);
           rcomp_s(i,3)=zcomp(j);
         enddo
! orientation atoms
         if (qstat_orient) then
          do i=1, norient
           j=iatom_o(i)
           rcomp_o(i,1)=xcomp(j);
           rcomp_o(i,2)=ycomp(j);
           rcomp_o(i,3)=zcomp(j);
          enddo
! subtract COM
          com=matmul(transpose(rcomp_o), orientWeights)
          rcomp_o(:,1)=rcomp_o(:,1)-com(1)
          rcomp_o(:,2)=rcomp_o(:,2)-com(2)
          rcomp_o(:,3)=rcomp_o(:,3)-com(3)
!
          rcomp_s(:,1)=rcomp_s(:,1)-com(1)
          rcomp_s(:,2)=rcomp_s(:,2)-com(2)
          rcomp_s(:,3)=rcomp_s(:,3)-com(3)
!
         endif ! qstat_orient
        endif
!
        if (.not.(associated(statWeights))) allocate(statWeights(nstat))
        statWeights=1d0
!
! use mass-weighting in RMSD computation?
!
        stat_rmsd_mass=(indxa(comlyn, comlen, 'MASS').gt.0)
        if (stat_rmsd_mass) then
         if (qprint) then ; write(sm0k_info,640) whoami ; write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info=''; ; endif
         do i=1,nstat
          statWeights(i)=amass(iatom_s(i))*statWeights(i)
         enddo
        endif ! stat_rmsd_mass
!
        d=sum(statWeights)
        if (abs(d).gt.RSMALL) then
         d=1d0/d
         statWeights=d*statWeights
        endif
!
 640 format(A, ' WILL USE MASS WEIGHTING IN RMSD CALCULATIONS.')
       endif ! output_ ...
      endif ! nstat .eq. 0
!
!!!!!!!!!!!!!! ARCLENGTH
      if (indxa(comlyn, comlen, 'ARCL').gt.0) then
        output_arclength=.true.
        call gtrmwa(COMLYN, COMLEN, 'ANAM', 4, s_fname, 80, s_flen)
        if (s_flen.eq.0) then
         call wrndie(0,whoami,trim('STRING LENGTH FILE NAME NOT SPECIFIED. WILL WRITE TO STDOUT.'))
         s_funit=outu
        else
         if (indxa(comlyn, comlen, 'AAPP').gt.0) then ! APPEND?
           sform='APPEND'
         else
           sform='WRITE'
         endif
         if (qprint) then
          s_funit=-1
          call open_file(s_funit, s_fname, 'FORMATTED', sform)
         endif
        endif
!ccccccccccc print summary
        if (qprint) then
         if (s_flen.gt.0) then
          write(sm0k_info,652) whoami,s_fname(1:s_flen)
         else
          write(sm0k_info,653) whoami
         endif
         write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        endif
 652 format(A,' WILL WRITE STRING LENGTH TO FILE ',A)
 653 format(A,' WILL WRITE STRING LENGTH TO STDOUT.')
!
      endif ! ARCLENGTH
!!!!!!!!!!!!!! CURVATURE
      if (indxa(comlyn, comlen, 'CURV').gt.0) then
        output_curvature=.true.
        call gtrmwa(COMLYN, COMLEN, 'CVNM', 4, c_fname, 80, c_flen)
        if (c_flen.eq.0) then
         call wrndie(0,whoami,trim('CURVATURE FILE NAME NOT SPECIFIED. WILL WRITE TO STDOUT.'))
         c_funit=outu
        else
         if (indxa(comlyn, comlen, 'CAPP').gt.0) then ! APPEND?
           cform='APPEND'
         else
           cform='WRITE'
         endif
         if (qprint) then
          c_funit=-1
          call open_file(c_funit, c_fname, 'FORMATTED', cform)
         endif
        endif
!ccccccccccc print summary
        if (qprint) then
         if (c_flen.gt.0) then
          write(sm0k_info,6521) whoami,c_fname(1:c_flen)
         else
          write(sm0k_info,6531) whoami
         endif
         write(OUTU,'(A)') pack(sm0k_info,sm0k_info.ne.'');sm0k_info='';
        endif
 6521 format(A,' WILL WRITE CURVATURE TO FILE ',A)
 6531 format(A,' WILL WRITE CURVATURE TO STDOUT.')
!
      endif ! CURVATURE
! if we got this far, we are probably OK
      stat_initialized=.true.
!
      end subroutine sm0k_stat_init
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine sm0k_stat(n,var)
! statistics subroutine
      use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3
      use stream
      use dimens_fcm
      use string
      use coord; use coordc
!
      use psf
      use energym
      use eutil, only : gete
#if (KEY_MULTICOM==1)
      use multicom_aux;
#endif
      use number
      use mpi
!
      implicit none
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
      integer :: n
      real(chm_real), optional :: var(*)
! locals
      character(len=8) :: keyword
      real(chm_real) :: u(3,3), com(3)
!
      integer :: klen, i, j, k, ifile
!
! for output
      real(chm_real) :: energy_values(enmax,SIZE_STRNG)! store data from all procs
      real(chm_real) :: energy_values_me(enmax)
      integer :: fmt_e_len
      character(len=80) :: fmt_ene
!
      integer :: me, ierror
      character(len=80) :: fmt
      integer :: fmt_len
      character(len=8) :: dummy
      logical :: found
      logical :: qprint, qroot, qmanual
      real(chm_real) :: rmsd0, rmsd0_all(SIZE_STRNG), dsdt, dsdt_all(SIZE_STRNG)
!
      character(len=len("SM0K_STAT>") ),parameter::whoami="SM0K_STAT>";!macro
!
      qroot=MPI_COMM_STRNG.ne.MPI_COMM_NULL
      qprint=qroot.and.ME_STRNG.eq.0
      qmanual=(n.eq.0)
!cccccccccccccccccccccc begin ccccccccccccccccccccccccccccc
! check if the user has made an initialization call
      if (.not.sm0k_initialized) call sm0k_init()
!
      if (.not.stat_initialized) then
       call wrndie(0,whoami,trim('NO OUTPUT OPTIONS SELECTED. NOTHING DONE'))
       return
      endif
!
      stat_iteration_counter=stat_iteration_counter+1
! define number format string for output
!
      if (qroot) then
       write(fmt,*) SIZE_STRNG
       fmt_len=len(fmt)
       call trima(fmt, fmt_len)
      endif
!
      me=mestring
!
!
      if (output_energy) then
       write(fmt_ene,*) num_energy_terms
       fmt_e_len=len(fmt_ene)
       call trima(fmt_ene, fmt_e_len)
!
! call energy
! NOTE: this routine needs to be called after "update"
       if (qmanual) call gete(x,y,z,x,y,z,0) ! is everything calculated in this routine?
! otherwise, the minimizer has already called energy routines (off by one iteration, but this should be insignificant)
       if (qroot) then
        if (energy_flen.eq.0) then
         call wrndie(0,whoami,trim('NO ENERGY FILE NAME SPECIFIED. WILL WRITE TO STDOUT.'))
         ifile=outu
        else
         if (qprint) then
          write(dummy,'(I8)') stat_iteration_counter
          k=len(dummy)
          call trima(dummy, k)
          ifile=-1
          energy_fname(energy_flen+1:energy_flen+k+4)=dummy(1:k)//'.dat'
          call open_file(ifile, energy_fname(1:energy_flen+k+4), 'FORMATTED', 'WRITE')
          energy_fname(energy_flen+1:)=''
         endif
        endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! assume file is open
! loop over energy terms
        do i=1, num_energy_terms
          j=energy_indices(i)
          if (j.gt.0) then
           energy_values_me(i)=eprop(j)
          elseif (j.lt.0) then
           energy_values_me(i)=eterm(-j)
          endif
        enddo
!ccccccccccccccccc
! gather data and write file
! call mpi_gather(energy_values(1,me+1),enmax,mpifloat
! & ,energy_values,enmax,mpifloat,0,
! & MPI_COMM_STRNG, ierror)
! some compilers require mpi_in_place when the input and output buffere are the same
! mpi_in_place, however, behaves strangely on some systems, so I am using
! the additional array `energy_values_me`
        call mpi_gather(energy_values_me,enmax,mpifloat &
     & ,energy_values,enmax,mpifloat,0, &
     & MPI_COMM_STRNG, ierror)
!
        if (qprint) then
         if (ifile.ne.outu) then
          write(ifile,'("%",'//fmt_ene(1:fmt_e_len)//'(A19))')          &
     & (energy_names(i),i=1,num_energy_terms)
          do j=1,SIZE_STRNG
           write(ifile,'(I5," ",'//fmt_ene(1:fmt_e_len)//'F20.10)')     &
     & j,(energy_values(i,j),i=1,num_energy_terms)
          enddo
          call VCLOSE(ifile, 'KEEP', ierror)
         else
      write(ifile,'("ENERGY> ",'//fmt_ene(1:fmt_e_len)//'(," ",A19))')  &
     & (energy_names(i),i=1,num_energy_terms)
          do j=1,SIZE_STRNG
          write(ifile,'("ENERGY> ",I5," ",'//                           &
     & fmt_ene(1:fmt_e_len)//'F20.10)')                       &
     & j,(energy_values(i,j),i=1,num_energy_terms)
          enddo
         endif ! ifile
        endif ! qprint
! 6650 format('ENERGY> ',100(' ',A19))
! 6660 format('ENERGY> ',I5,' ',100F20.10)
! 665 format('%',100(A19))
! 666 format(I5,' ',100F20.10)
!
       endif ! qroot
      endif ! output energy
!
!
      if (output_rmsd0.or.output_rmsd_ave.or.output_dsdt) then
       do i=1, nstat
         if (qmanual.or.fixed_s(i)) then
          j=iatom_s(i)
          rcurrent_s(i,1)=x(j)
          rcurrent_s(i,2)=y(j)
          rcurrent_s(i,3)=z(j)
         else
          j=iatom_free_s(i);
          rcurrent_s(i,1)=var(j);j=j+1
          rcurrent_s(i,2)=var(j);j=j+1
          rcurrent_s(i,3)=var(j)
         endif
       enddo ! nstat
! orientation atoms
       if (qstat_orient) then
        do i=1,norient
          if (qmanual.or.fixed_o(i)) then ! grab coordinates from main coordinate array
           j=iatom_o(i)
           rcurrent_o(i,1)=x(j)
           rcurrent_o(i,2)=y(j)
           rcurrent_o(i,3)=z(j)
          else ! grab coordinates provided by minimizer
           j=iatom_free_o(i) ! x-index (y- z- indices follow)
           rcurrent_o(i,1)=var(j);j=j+1
           rcurrent_o(i,2)=var(j);j=j+1
           rcurrent_o(i,3)=var(j)
          endif
        enddo ! norient
        com=matmul(transpose(rcurrent_o), orientWeights)
        rcurrent_o(:,1)=rcurrent_o(:,1)-com(1)
        rcurrent_o(:,2)=rcurrent_o(:,2)-com(2)
        rcurrent_o(:,3)=rcurrent_o(:,3)-com(3)
        rcurrent_s(:,1)=rcurrent_s(:,1)-com(1)
        rcurrent_s(:,2)=rcurrent_s(:,2)-com(2)
        rcurrent_s(:,3)=rcurrent_s(:,3)-com(3)
       endif ! qstat_orient
      endif ! output rmsd
!
!ccccccccccccccccccccccccccccccccccccc
      if (output_rmsd0) then
        if (qroot) then
! reference structure is in the comparison set
         if (qstat_orient) then
           call RMSBestFit(rcurrent_o,rcomp_o,orientWeights,u)
! transform current structure to overlap with reference
! (if orientation is off, u=I)
           u=transpose(u)
           rcurrent_o=matmul(rcurrent_o, u) ! need to rotate both (see below)
           rcurrent_s=matmul(rcurrent_s, u)
         endif
         rmsd0=rmsd(rcurrent_s, rcomp_s, statWeights)
!
! gather !
         call mpi_gather(rmsd0,1,mpifloat &
     & ,rmsd0_all,1,mpifloat,0, &
     & MPI_COMM_STRNG, ierror)
         if (qprint) then ! root writes
           if (rmsd0_funit.eq.outu) then
      write(rmsd0_funit,'("RMSD0> ",I5," ",'//fmt(1:fmt_len)//'F11.5)') &
     & stat_iteration_counter, &
     & (rmsd0_all(i),i=1,SIZE_STRNG)
           else
            write(rmsd0_funit,'(I5," ",'//fmt(1:fmt_len)//'F11.5)')     &
     & stat_iteration_counter, &
     & (rmsd0_all(i),i=1,SIZE_STRNG)
           endif
! flush unit: close and reopen
           call VCLOSE(rmsd0_funit, 'KEEP', ierror)
           call open_file(rmsd0_funit, rmsd0_fname, 'FORMATTED', 'APPEND')
! done
         endif ! qprint
        endif ! qroot
      endif
! 667 format(I5,' ',100F11.5)
! 6670 format('RMSD0> ',I5,' ',100F11.5)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_dsdt) then
        if (qroot) then
         if (repa_initialized) then ! proceed only if rold is defined
          if (qstat_orient) then
           call RMSBestFit(rcurrent_o,rold_o,orientWeights,u)
           u=transpose(u)
           rcurrent_o=matmul(rcurrent_o, u)
           rcurrent_s=matmul(rcurrent_s, u)
          endif
          dsdt=rmsd(rcurrent_s, rold_s, statWeights)
! gather !
          call mpi_gather(dsdt,1,mpifloat &
     & ,dsdt_all,1,mpifloat,0, &
     & MPI_COMM_STRNG, ierror)
          if (qprint) then ! root writes
           if (dsdt_funit.eq.outu) then
      write(dsdt_funit,'("DLEN> ",I5," ",'//fmt(1:fmt_len)//'F11.5)')   &
     & stat_iteration_counter, &
     & (dsdt_all(i),i=1,SIZE_STRNG)
           else
            write(dsdt_funit,'(I5," ",'//fmt(1:fmt_len)//'F11.5)')      &
     & stat_iteration_counter, &
     & (dsdt_all(i),i=1,SIZE_STRNG)
           endif
! flush unit: close and reopen
           call VCLOSE(dsdt_funit, 'KEEP', ierror)
           call open_file(dsdt_funit, dsdt_fname, 'FORMATTED', 'APPEND')
! done
          endif ! qprint
         else ! repa_initialized
          call wrndie(0,whoami,trim('NO REPARAMETRIZATION OPTIONS SELECTED. SKIPPING DSDT.'))
         endif
        endif ! qroot
! 6680 format('DLEN> ',I5,' ',100F11.5)
! 668 format(I5,' ',100F11.5)
      endif ! dsdt
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_rmsd_ave) then
        if (qroot) then
         if (repa_initialized) then ! proceed only if rave defined
          if (qstat_orient) then
           call RMSBestFit(rcurrent_o,rave_o,orientWeights,u)
           u=transpose(u)
           rcurrent_o=matmul(rcurrent_o, u)
           rcurrent_s=matmul(rcurrent_s, u)
          endif
          dsdt=rmsd(rcurrent_s, rave_s, statWeights)
! gather !
          call mpi_gather(rmsd0,1,mpifloat &
     & ,rmsd0_all,1,mpifloat,0, &
     & MPI_COMM_STRNG, ierror)
          if (qprint) then ! root writes
           if (rmsd_ave_funit.eq.outu) then
            write(rmsd_ave_funit,'("RMSD_AVE> ",I5," ",'//              &
     & fmt(1:fmt_len)//'F11.5)')                                   &
     & stat_iteration_counter, &
     & (rmsd0_all(i),i=1,SIZE_STRNG)
           else
            write(rmsd_ave_funit,'(I5," ",'//fmt(1:fmt_len)//'F11.5)')  &
     & stat_iteration_counter, &
     & (rmsd0_all(i),i=1,SIZE_STRNG)
           endif
          endif
! 6681 format('RMSD_AVE> ',I5,' ',100F11.5)
! 6682 format(I5,' ',100F11.5)
! flush unit: close and reopen
          call VCLOSE(rmsd_ave_funit, 'KEEP', ierror)
          call open_file(rmsd_ave_funit, rmsd_ave_fname, 'FORMATTED', 'APPEND')
        else ! repa_initialized
          call wrndie(0,whoami,trim('NO REPARAMETRIZATION OPTIONS SELECTED. SKIPPING RMSD_AVE.'))
        endif ! repa_initialized
       endif ! qroot
! done
      endif ! RMSD from average structure
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_arclength) then
       if (qprint) then
        if (repa_initialized) then ! proceed only if arclength defined
         if (s_funit.eq.outu) then
      write(s_funit,'("ARCL> ",I5," ",'//fmt(1:fmt_len)//'F11.5)')      &
     & stat_iteration_counter, (ds(i),i=1,SIZE_STRNG-1)
         else
          write(s_funit,'(I5," ",'//fmt(1:fmt_len)//'F11.5)')           &
     & stat_iteration_counter, (ds(i),i=1,SIZE_STRNG-1)
         endif
! flush unit: close and reopen
         call VCLOSE(s_funit, 'KEEP', ierror)
         call open_file(s_funit, s_fname, 'FORMATTED', 'APPEND')
! done
        else
          call wrndie(0,whoami,trim('NO REPARAMETRIZATION OPTIONS SELECTED. SKIPPING ARCLENGTH.'))
        endif
       endif ! qprint
! 669 format(I5,' ',100F11.5)
! 6690 format('ARCL> ',I5,' ',100F11.5)
      endif ! output_arclength
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_curvature) then
       if (qprint) then
        if (repa_initialized) then ! proceed only if arclength defined
         if (c_funit.eq.outu) then
      write(c_funit,'("CURV> ",I5," ",'//fmt(1:fmt_len)//'F11.5)')      &
     & stat_iteration_counter, (curv(i),i=1,SIZE_STRNG-2)
         else
          write(c_funit,'(I5," ",'//fmt(1:fmt_len)//'F11.5)')           &
     & stat_iteration_counter, (curv(i),i=1,SIZE_STRNG-2)
         endif
! flush unit: close and reopen
         call VCLOSE(c_funit, 'KEEP', ierror)
         call open_file(c_funit, c_fname, 'FORMATTED', 'APPEND')
! done
        else
          call wrndie(0,whoami,trim('NO REPARAMETRIZATION OPTIONS SELECTED. SKIPPING CURVATURE.'))
        endif
       endif ! me
! 6692 format(I5,' ',100F11.5)
! 6691 format('CURV> ',I5,' ',100F11.5)
      endif ! output_curvature
!
      end subroutine sm0k_stat
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#endif
#endif /* automatically protect all code */
      end module sm0k
