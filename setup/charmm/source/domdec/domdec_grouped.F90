module domdec_grouped
  ! *
  ! * Module for grouped atoms:
  ! *
  ! * -bonds, angles, dihedrals, etc.
  ! * -lone pairs
  ! * - anisotropy
  ! *
#if KEY_DOMDEC==1
  use chm_kinds
  use nblist_types,only:intarray_t
  implicit none
  private

  ! --------------------------------------------------------------------------------------------
  ! For internal use within this module only:
  integer, allocatable, dimension(:,:) :: ngroupedtbl_th
  type(intarray_t), target, allocatable, dimension(:,:) :: groupedtbl_th
  ! --------------------------------------------------------------------------------------------

  ! Logical flag that determines if grouped atoms are used or not
  logical q_grouped

  ! Cutoff for grouped interactions
  real(chm_real) rcut_grouped

  ! Grouped structure
  type grouped_struct_t
     integer, allocatable, dimension(:) :: ngrouped
     integer, allocatable, dimension(:) :: grouped
     integer, allocatable, dimension(:) :: groupedstart
   contains
     procedure :: init
     procedure :: uninit
  end type grouped_struct_t

  type(grouped_struct_t) :: grouped_struct

  ! temporary list used by build_groupedtbl_kernel
  ! Dimensions:
  ! tmplist(1:3,0:nthread-1)%array(1:ngroup)
  ! ntmplist(1:3,0:nthread-1)
  type(intarray_t), allocatable, dimension(:,:) :: tmplist
  integer, allocatable, dimension(:,:) :: ntmplist

  ! Public variables
  public q_grouped, rcut_grouped

  ! Public subroutines
  public init_grouped, uninit_grouped
  public build_groupedtbl
  public calc_rcut_grouped

#endif

contains

#if KEY_DOMDEC==1
  ! *
  ! * Initialization for grouped_struct_t.
  ! *
  subroutine init(self, natom, grouped_size)
    use memory
    class(grouped_struct_t), intent(inout) :: self
    integer, intent(in) :: natom, grouped_size

    ! groupedstart
    if (allocated(self%groupedstart)) then
       if (size(self%groupedstart) /= natom) then
          call chmdealloc('domdec_grouped.src','grouped_struct_t%init',&
               'groupedstart',size(self%groupedstart),intg=self%groupedstart)
       endif
    endif
    if (.not.allocated(self%groupedstart)) then
          call chmalloc('domdec_grouped.src','grouped_struct_t%init',&
               'groupedstart',natom,intg=self%groupedstart)
    endif

    ! ngrouped
    if (allocated(self%ngrouped)) then
       if (size(self%ngrouped) /= natom) then
          call chmdealloc('domdec_grouped.src','grouped_struct_t%init',&
               'ngrouped',size(self%ngrouped),intg=self%ngrouped)
       endif
    endif
    if (.not.allocated(self%ngrouped)) then
          call chmalloc('domdec_grouped.src','grouped_struct_t%init',&
               'ngrouped',natom,intg=self%ngrouped)
    endif

    ! grouped
    if (allocated(self%grouped)) then
       if (size(self%grouped) /= grouped_size) then
          call chmdealloc('domdec_grouped.src','grouped_struct_t%init',&
               'grouped',size(self%grouped),intg=self%grouped)
       endif
    endif
    if (.not.allocated(self%grouped)) then
          call chmalloc('domdec_grouped.src','grouped_struct_t%init',&
               'grouped',grouped_size,intg=self%grouped)
    endif

    return
  end subroutine init

  ! *
  ! * Uninitialization for grouped_struct_t
  ! *
  subroutine uninit(self)
    use memory,only:chmdealloc
    class(grouped_struct_t), intent(inout) :: self

    ! groupedstart
    if (allocated(self%groupedstart)) then
       call chmdealloc('domdec_grouped.src','grouped_struct_t%uninit',&
            'groupedstart',size(self%groupedstart),intg=self%groupedstart)
    endif

    ! ngrouped
    if (allocated(self%ngrouped)) then
       call chmdealloc('domdec_grouped.src','grouped_struct_t%uninit',&
            'ngrouped',size(self%ngrouped),intg=self%ngrouped)
    endif

    ! grouped
    if (allocated(self%grouped)) then
       call chmdealloc('domdec_grouped.src','grouped_struct_t%uninit',&
            'grouped',size(self%grouped),intg=self%grouped)
    endif

    return
  end subroutine uninit

  ! *
  ! * Initializes grouped atoms
  ! *
  subroutine init_grouped()
    use domdec_bonded_types,only:ngrouptype
    use domdec_bonded,only:get_bonded_num, set_bonded_alloc_len, bonded_associate_tbl
    use domdec_aniso,only:get_aniso_num, set_aniso_alloc_len, aniso_associate_tbl
    use domdec_common,only:nthread
    use memory
    implicit none
    integer i, j
    integer alloc_len(ngrouptype)

    q_grouped = .false.
    if (get_bonded_num() > 0) q_grouped = .true.
    if (get_aniso_num() > 0) q_grouped = .true.

    if (.not.q_grouped) return
    
    ! groupedtbl_th
    if (allocated(groupedtbl_th)) then
       if (size(groupedtbl_th,2) < nthread) then
          do i=0,nthread-1
             do j=1,ngrouptype
                if (allocated(groupedtbl_th(j,i)%array)) then
                   call chmdealloc('domdec_grouped.src','init_grouped','groupedtbl_th(j,i)%array',&
                        size(groupedtbl_th(j,i)%array),intg=groupedtbl_th(j,i)%array)
                endif
             enddo
          enddo
          deallocate(groupedtbl_th)
       endif
    endif

    if (.not.allocated(groupedtbl_th)) then
       allocate(groupedtbl_th(1:ngrouptype,0:nthread-1))
    endif

    do i=0,nthread-1
       call set_bonded_alloc_len(alloc_len)
       call set_aniso_alloc_len(alloc_len)
       if (i == 0) alloc_len(1:ngrouptype) = nthread*alloc_len(1:ngrouptype)
       ! NOTE: +16 is added here to avoid cache trashing among the threads
       alloc_len(1:ngrouptype) = alloc_len(1:ngrouptype) + 16
       do j=1,ngrouptype
          alloc_len(j) = max(1,alloc_len(j))
          if (allocated(groupedtbl_th(j,i)%array) .and. &
               size(groupedtbl_th(j,i)%array) < alloc_len(j)) then
             call chmdealloc('domdec_grouped.src','init_grouped','groupedtbl_th(j,i)%array',&
                  size(groupedtbl_th(j,i)%array),intg=groupedtbl_th(j,i)%array)
          endif
          if (.not.allocated(groupedtbl_th(j,i)%array)) then
             call chmalloc('domdec_grouped.src','init_grouped','groupedtbl_th(j,i)%array',&
                  alloc_len(j),intg=groupedtbl_th(j,i)%array)
          endif
       enddo
    enddo

    ! ngroupedtbl_th
    if (allocated(ngroupedtbl_th)) then
       if (size(ngroupedtbl_th,2) < nthread) then
          call chmdealloc('domdec_grouped.src','init_grouped','ngroupedtbl_th',&
               size(ngroupedtbl_th,1),size(ngroupedtbl_th,2),intg=ngroupedtbl_th)
       endif
    endif

    if (.not.allocated(ngroupedtbl_th)) then
       call chmalloc('domdec_grouped.src','init_grouped','ngroupedtbl_th',&
            ngrouptype+16,nthread,lbou2=0,intg=ngroupedtbl_th)
    endif

    call bonded_associate_tbl(groupedtbl_th(:,0))
    call aniso_associate_tbl(groupedtbl_th(:,0))

    call build_grouped_struct()

    return
  end subroutine init_grouped

  ! *
  ! * Uninitializes grouped atoms
  ! *
  subroutine uninit_grouped()
    use domdec_common,only:nthread
    use domdec_bonded_types,only:ngrouptype
    use memory,only:chmdealloc
    implicit none
    integer i, j

    ! groupedtbl_th
    if (allocated(groupedtbl_th)) then
       do i=0,nthread-1
          do j=1,ngrouptype
             if (allocated(groupedtbl_th(j,i)%array)) then
                call chmdealloc('domdec_grouped.src','uninit_grouped','groupedtbl_th(j,i)%array',&
                     size(groupedtbl_th(j,i)%array),intg=groupedtbl_th(j,i)%array)
             endif
          enddo
       enddo
       deallocate(groupedtbl_th)
    endif

    ! ngroupedtbl_th
    if (allocated(ngroupedtbl_th)) then
       call chmdealloc('domdec_grouped.src','uninit_grouped','ngroupedtbl_th',&
            size(ngroupedtbl_th,1),size(ngroupedtbl_th,2),intg=ngroupedtbl_th)
    endif

    call grouped_struct%uninit()

    ! tmplist
    if (allocated(tmplist)) then
       do i=1,3
          do j=0,nthread-1
             call chmdealloc('domdec_grouped.src','uninit_grouped','tmplist(i,j)%array',&
               size(tmplist(i,j)%array),intg=tmplist(i,j)%array)
          enddo
       enddo
       deallocate(tmplist)
    endif

    ! ntmplist
    if (allocated(ntmplist)) then
       call chmdealloc('domdec_grouped.src','uninit_grouped','ntmplist',&
            size(ntmplist,1),size(ntmplist,2),intg=ntmplist)
    endif

    return
  end subroutine uninit_grouped

  ! *
  ! * Builds grouped_struct
  ! *
  subroutine build_grouped_struct()
    use stream,only:outu
    use memory,only:chmalloc, chmdealloc
    use psf,only:natom
    use domdec_bonded_types,only:ll_type, ngrouptype, TYPE_HYPER, TYPE_ANISO
    use domdec_bonded,only:get_bonded_ind, get_bonded_num, build_bonded_ll
    use domdec_aniso,only:get_aniso_ind, get_aniso_num, build_aniso_ll
    implicit none
    ! Variables
    integer i, j, k, d, ll_ind
    integer ngrouped_sum, grouped_size, ind(1:8), nind
    integer groupedll_data_len
    type(ll_type), allocatable, dimension(:) :: groupedll_data
    integer, allocatable, dimension(:,:) :: groupedll_head, ntmp

    call chmalloc('domdec_grouped.src','build_grouped_struct',&
         'ntmp',natom,ngrouptype,intg=ntmp)
    call chmalloc('domdec_grouped.src','build_grouped_struct',&
         'groupedll_head',natom,ngrouptype,intg=groupedll_head)

    groupedll_data_len = 0
    groupedll_data_len = groupedll_data_len + get_bonded_num()
    groupedll_data_len = groupedll_data_len + get_aniso_num()

    allocate(groupedll_data(groupedll_data_len))

    groupedll_head(:,:) = 0
    ntmp(:,:) = 0
    grouped_size = 0
    ll_ind = 0
    call build_bonded_ll(ntmp, grouped_size, ll_ind, groupedll_head, groupedll_data)
    call build_aniso_ll(ntmp, grouped_size, ll_ind, groupedll_head, groupedll_data)

    if (ll_ind /= sum(ntmp)) then
       call wrndie(-5,'<domdec_grouped>','ll_end /= sum(ntmp) in build_grouped_struct')
    endif

    call grouped_struct%init(natom, grouped_size)
    grouped_struct%groupedstart(1:natom) = 0

    k = 1
    do i=1,natom
       ! Number of grouped interactions for atom i
       grouped_struct%ngrouped(i) = sum(ntmp(i,:))
       ! Start of the grouped -list for atom i
       grouped_struct%groupedstart(i) = k
       ! Construct grouped -list
       do d=1,ngrouptype
          ll_ind = groupedll_head(i,d)
          do j=1,ntmp(i,d)
             if (ll_ind == 0) then
                write (outu,'(a,3i4)') 'i,d,j=',i,d,j
                call wrndie(-5,'<domdec_grouped>','error in linked list in build_grouped_struct')
             endif
             ! Store type
             grouped_struct%grouped(k) = d
             k = k + 1
             ! Store index
             grouped_struct%grouped(k) = groupedll_data(ll_ind)%ind
             k = k + 1
             ! Retrieve and store atom indices
             if (d <= TYPE_HYPER) then
                call get_bonded_ind(groupedll_data(ll_ind)%ind, d, ind, nind)
             else if (d == TYPE_ANISO) then
                call get_aniso_ind(groupedll_data(ll_ind)%ind, d, ind, nind)
             else
                call wrndie(-5,'<domdec_grouped>','build_grouped_struct, invalid type d')
             endif

             if (nind > 0) then
                grouped_struct%grouped(k:k+nind-1) = ind(1:nind)
                k = k + nind
             endif

             ll_ind = groupedll_data(ll_ind)%next
          enddo
       enddo
    enddo    

    if (k-1 /= grouped_size) then
       write (outu,'(a,2i8)') 'k-1,grouped_size=',k-1,grouped_size
       call wrndie(-5,'<domdec_grouped>','build_grouped_struct: invalid grouped size')
    endif

    deallocate(groupedll_data)
    call chmdealloc('domdec_grouped.src','build_grouped_struct',&
         'groupedll_head',natom,ngrouptype,intg=groupedll_head)
    call chmdealloc('domdec_grouped.src','build_grouped_struct',&
         'ntmp',natom,ngrouptype,intg=ntmp)

    return
  end subroutine build_grouped_struct

  ! *
  ! * Checks if group ii belongs to this box
  ! * ind(1:nind) = indices
  ! *
  integer function group_check_box(ind, nind)
    use groupxfast,only:groupcenter, invgroup, grouploc
    use domdec_common,only:homezone
    implicit none
    ! Input / Output
    integer, intent(in) :: ind(:), nind
    ! Variables
    integer i, j, gind, loc

    group_check_box = 0
    loc = 0
    do i=1,nind
       j = ind(i)
       if (homezone(j) == 0) return
       gind = invgroup(j)
       loc = ior(loc, grouploc(gind))
    enddo

    if (loc == b'111') group_check_box = 1

    return
  end function group_check_box

  ! *
  ! * Outputs:
  ! * tmplist(1:ntmplist(1), 1) = solutes in homebox (max zonelist(1) number)
  ! * tmplist(1:ntmplist(2), 2) = solvents in homebox (max zonelist(1) number)
  ! * tmplist(1:ntmplist(3), 3) = solutes in import volume (max zonelist(8)-zonelist(1)+1 number)
  ! *
  subroutine build_tmplist(zonelist, groupl, group, nthread, ntmplist, tmplist)
    use groupxfast,only:group_out
    use nblist_util,only:reduce_intarray
    implicit none
    ! Input / Output
    integer, intent(in) :: zonelist(8), groupl(:), group(:), nthread
    integer, intent(inout) :: ntmplist(1:3,0:nthread-1)
    type(intarray_t), intent(inout) :: tmplist(1:3,0:nthread-1)
    ! Functions
    integer omp_get_thread_num
    ! Variables
    integer tid, jg, igrp, g, itype, is, iq
    integer ntmplist_loc(1:3)

!$omp parallel private(tid, ntmplist_loc)
#ifdef _OPENMP
    tid = omp_get_thread_num()
#else
    tid = 0
#endif

    ntmplist_loc(1:3) = 0

    ! Homezone: include both solutes and solvents
!$omp do schedule(static) private(jg, igrp, g, is, iq, itype)
    do jg=1,zonelist(1)
       igrp = groupl(jg)
       g = group(igrp)
       call group_out(g, itype)
       itype = itype + 1
       ntmplist_loc(itype) = ntmplist_loc(itype) + 1
       tmplist(itype,tid)%array(ntmplist_loc(itype)) = g
    enddo
!$omp end do

    ! Import volume: include only solutes
!$omp do schedule(static) private(jg, igrp, g, is, iq, itype)
    do jg=zonelist(1)+1,zonelist(8)
       igrp = groupl(jg)
       g = group(igrp)
       call group_out(g, itype)
       if (itype == 0) then
          ntmplist_loc(3) = ntmplist_loc(3) + 1
          tmplist(3,tid)%array(ntmplist_loc(3)) = g
       endif
    enddo
!$omp end do

    ntmplist(1:3,tid) = ntmplist_loc(1:3)
!$omp end parallel

    ! Combine tmplist(1:3,0:nthread-1) into tmplist(1:3,0)
    call reduce_intarray(3, nthread, ntmplist, tmplist)

    return
  end subroutine build_tmplist

  ! *
  ! * Builds grouped table kernel
  ! *
  subroutine build_groupedtbl_kernel(zonelist, groupl, group, storage_size, &
       groupedstart, ngrouped, grouped, nsolute_home, solute_home_list, nsolvent, solvent_list, &
       nsolute_impvol, solute_impvol_list, ngroupedtbl, groupedtbl)
    use groupxfast,only:group_out
    implicit none
    ! Input / Output
    integer, intent(in) :: zonelist(8), groupl(:), group(:), storage_size(:)
    integer, intent(in) :: groupedstart(:), ngrouped(:), grouped(:)
    integer, intent(in) :: nsolute_home, solute_home_list(:)
    integer, intent(in) :: nsolvent, solvent_list(:)
    integer, intent(in) :: nsolute_impvol, solute_impvol_list(:)
    integer, intent(inout) :: ngroupedtbl(:)
    type(intarray_t), intent(inout) :: groupedtbl(:)
    ! Variables
    integer jg, igrp, is, iq, itype, i, j, k, bt, bi, ind(8), nind
    integer g, plus

    ! Solutes in home box

!$omp do schedule(static)
    do jg=1,nsolute_home
       call group_out(solute_home_list(jg), is, iq, itype)
       do i=is,iq
          j = groupedstart(i)
          do k=1,ngrouped(i)
             bt = grouped(j)
             bi = grouped(j+1)
             j = j + 2
             nind = storage_size(bt)
             ind(1:nind) = grouped(j:j+nind-1)
             plus = group_check_box(ind, nind)
             groupedtbl(bt)%array(ngroupedtbl(bt)+1) = bi
             ngroupedtbl(bt) = ngroupedtbl(bt) + plus
             j = j + nind
          enddo
       enddo
    enddo
!$omp end do

    ! Solutes in import volume

!$omp do schedule(static)
    do jg=1,nsolute_impvol
       call group_out(solute_impvol_list(jg), is, iq, itype)
       do i=is,iq
          j = groupedstart(i)
          do k=1,ngrouped(i)
             bt = grouped(j)
             bi = grouped(j+1)
             j = j + 2
             nind = storage_size(bt)
             ind(1:nind) = grouped(j:j+nind-1)
             plus = group_check_box(ind, nind)
             groupedtbl(bt)%array(ngroupedtbl(bt)+1) = bi
             ngroupedtbl(bt) = ngroupedtbl(bt) + plus
             j = j + nind
          enddo
       enddo
    enddo
!$omp end do

    ! Solvents
!$omp do schedule(static)
    do jg=1,nsolvent
       call group_out(solvent_list(jg), is, iq, itype)
       do i=is,iq
          j = groupedstart(i)
          do k=1,ngrouped(i)
             bt = grouped(j)
             bi = grouped(j+1)
             ngroupedtbl(bt) = ngroupedtbl(bt) + 1
             groupedtbl(bt)%array(ngroupedtbl(bt)) = bi
             j = j + 2
          enddo
       enddo
    enddo
!$omp end do

    return
  end subroutine build_groupedtbl_kernel

  ! *
  ! * Builds grouped interaction tables
  ! *
  ! * If q_use_fixed_removed = true, we use grouped_struct_fixed_removed (where groups that are
  ! * constrained by holonomic constraints are removed), otherwise we use grouped_struct_all
  ! *
  ! * NOTE: called after update_groupl is done
  ! * NOTE: requires that homezone values are up to date
  ! *
  subroutine build_groupedtbl()
    use nblist_util,only:reduce_intarray
    use groupxfast,only:group, group_out
    use domdec_common,only:zonelist, groupl, nthread
    use domdec_bonded_types,only:ngrouptype, grouptype_storage_size
    use domdec_bonded,only:set_bonded_ntbl
#if KEY_DOMDEC_GPU==1
    use domdec_util_gpu_mod,only:range_start, range_stop
    use domdec_common,only:q_gpu
#endif
    use domdec_aniso,only:set_aniso_ntbl
    implicit none
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num
#endif
    ! Variables
    integer i, j, k, jg, ll
    integer ind(8), nind, bt, bi
    integer igrp, is, iq, itype
    integer tid

    ! Allocate & reallocate memory (ntmplist, tmplist)
    call alloc_realloc(nthread, zonelist(1), zonelist(8)-zonelist(1)+1)

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('q_grouped')
#endif

    if (q_grouped) then

#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_start('build_tmplist')
#endif
       call build_tmplist(zonelist, groupl, group, nthread, ntmplist, tmplist)
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_stop()
#endif

#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_start('build_groupedtbl_kernel')
#endif

!$omp parallel private(tid)
#ifdef _OPENMP
       tid = omp_get_thread_num()
#else
       tid = 0
#endif
       ngroupedtbl_th(1:ngrouptype,tid) = 0
       call build_groupedtbl_kernel(zonelist, groupl, group, grouptype_storage_size, &
            grouped_struct%groupedstart, grouped_struct%ngrouped, &
            grouped_struct%grouped, &
            ntmplist(1,0), tmplist(1,0)%array, &
            ntmplist(2,0), tmplist(2,0)%array, &
            ntmplist(3,0), tmplist(3,0)%array, &
            ngroupedtbl_th(:,tid), groupedtbl_th(:,tid))
!$omp end parallel
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_stop()
#endif

#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_start('reduce')
#endif
       call reduce_intarray(ngrouptype, nthread, ngroupedtbl_th, groupedtbl_th)
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_stop()
#endif

       call set_bonded_ntbl(ngroupedtbl_th(:,0))
       call set_aniso_ntbl(ngroupedtbl_th(:,0))

    endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

    return
  contains

    subroutine alloc_realloc(nthread, ngroup_home, ngroup_impvol)
      use memory
      implicit none
      ! Input
      integer, intent(in) :: nthread, ngroup_home, ngroup_impvol
      ! Variables
      logical q_realloc
      integer i, j
      integer ngroup(3)

      ngroup(1:3) = (/ ngroup_home, ngroup_home, ngroup_impvol /)

      ! ntmplist
      if (allocated(ntmplist)) then
         if (size(ntmplist,2) < nthread) then
            call chmdealloc('domdec_grouped.src','build_groupedtbl','ntmplist',&
                 size(ntmplist,1),size(ntmplist,2),intg=ntmplist)
         endif
      endif

      if (.not.allocated(ntmplist)) then
         call chmalloc('domdec_grouped.src','build_groupedtbl','ntmplist',&
              3,nthread,lbou2=0,intg=ntmplist)
      endif

      ! tmplist
      if (allocated(tmplist)) then
         q_realloc = .false.
         if (size(tmplist,2) < nthread) q_realloc = .true.
         do i=1,3
            do j=0,nthread-1
               if (size(tmplist(i,j)%array) < ngroup(i)) q_realloc = .true.
            enddo
         enddo
         if (q_realloc) then
            do i=1,3
               do j=0,nthread-1
                  call chmdealloc('domdec_grouped.src','build_groupedtbl','tmplist(i,j)%array',&
                       size(tmplist(i,j)%array),intg=tmplist(i,j)%array)
               enddo
            enddo
            deallocate(tmplist)
         endif
      endif

      if (.not.allocated(tmplist)) then
         allocate(tmplist(1:3,0:nthread-1))
         do i=1,3
            do j=0,nthread-1
               call chmalloc('domdec_grouped.src','build_groupedtbl','tmplist(i,j)%array',&
                    int(ngroup(i)*1.4),intg=tmplist(i,j)%array)
            enddo
         enddo
      endif

      return
    end subroutine alloc_realloc

  end subroutine build_groupedtbl

  ! *
  ! * Calculates the bonded interaction cut-off
  ! * (= maximum distance between atoms in any bond)
  ! * Requires groupcenter calculated for all groups
  ! *
  subroutine calc_rcut_grouped(x, y, z, rcut)
    use number,only:zero, half
    use psf
    use image,only:xucell
#if KEY_LONEPAIR==1
    use lonepr,only:numlp, lpnhost, lphptr, lphost
#endif
    use aniso_fcm,only:naniso, lstani1, lstani2, lstani3, lstani4
    use groupxfast,only:grouprad, groupbox, groupcenter, invgroup, ngroup
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    real(chm_real), intent(out) :: rcut
    ! Variables
    integer ii, ind(8)
#if KEY_LONEPAIR==1
    integer n, ipt
#endif
    real(chm_real) dxyz(3), hxucell(3)

    hxucell(1:3) = half*xucell(1:3)

    ! Calculate group radii
    do ii=1,ngroup
       grouprad(ii) = groupbox(1,ii)**2 + groupbox(2,ii)**2 + groupbox(3,ii)**2
    enddo

    rcut = zero

    do ii=1,nbond
       if (min(ib(ii),jb(ii)) <= 0) cycle
       ind(1) = ib(ii)
       ind(2) = jb(ii)
       call calc_rcut_max(2)
    enddo

    do ii=1,ntheta
       if (min(it(ii),jt(ii),kt(ii)) <= 0) cycle
       ind(1) = it(ii)
       ind(2) = jt(ii)
       ind(3) = kt(ii)
       call calc_rcut_max(3)
    enddo

    do ii=1,nphi
       if (min(ip(ii),jp(ii),kp(ii),lp(ii)) <= 0) cycle
       ind(1) = ip(ii)
       ind(2) = jp(ii)
       ind(3) = kp(ii)
       ind(4) = lp(ii)
       call calc_rcut_max(4)
    enddo

    do ii=1,nimphi
       if (min(ip(ii),jp(ii),kp(ii),lp(ii)) <= 0) cycle
       ind(1) = ip(ii)
       ind(2) = jp(ii)
       ind(3) = kp(ii)
       ind(4) = lp(ii)
       call calc_rcut_max(4)
    enddo

#if KEY_CMAP==1
    do ii=1,ncrterm
       if (min(i1ct(ii),j1ct(ii),k1ct(ii),l1ct(ii),&
            i2ct(ii),j2ct(ii),k2ct(ii),l2ct(ii)) <= 0) cycle
       ind(1) = i1ct(ii)
       ind(2) = j1ct(ii)
       ind(3) = k1ct(ii)
       ind(4) = l1ct(ii)
       ind(5) = i2ct(ii)
       ind(6) = j2ct(ii)
       ind(7) = k2ct(ii)
       ind(8) = l2ct(ii)
       call calc_rcut_max(8)
    enddo
#endif 

#if KEY_LONEPAIR==1
    do ii=1,numlp
       n = lpnhost(ii)
       ipt = lphptr(ii)
       ind(1:n+1) = lphost(ipt:ipt+n)
       call calc_rcut_max(n+1)
    enddo
#endif

    do ii=1,naniso
       ind(1) = lstani1(ii)
       ind(2) = lstani1(ii) + 1
       ind(3) = lstani2(ii)
       ind(4) = lstani3(ii)
       ind(5) = lstani4(ii)
       call calc_rcut_max(5)
    enddo

    ! We assume maximum 20% stretch
    rcut = 1.20d0*sqrt(rcut)

    return

  contains
    subroutine pbc(dxyz)
      implicit none
      real(chm_real) dxyz(3)

      do while (dxyz(1) > hxucell(1))
         dxyz(1) = dxyz(1) - xucell(1)
      enddo
      
      do while (dxyz(2) > hxucell(2))
         dxyz(2) = dxyz(2) - xucell(2)
      enddo

      do while (dxyz(3) > hxucell(3))
         dxyz(3) = dxyz(3) - xucell(3)
      enddo

      return
    end subroutine pbc

    subroutine calc_rcut_max(nind)
      implicit none
      integer, intent(in) :: nind
      integer i, j
      integer gind(8)

      ! Get group index
      do i=1,nind
         gind(i) = invgroup(ind(i))
      enddo

      do i=1,nind-1
         do j=i+1,nind
            if (gind(i) /= gind(j)) then
               ! Separate groups
               dxyz(1:3) = abs(groupcenter(1:3,gind(i)) - groupcenter(1:3,gind(j)))
               call pbc(dxyz)
               rcut = max(rcut,sum(dxyz(1:3)**2)+grouprad(gind(i))+grouprad(gind(j)))
            else
               ! Same group
               dxyz(1:3) = (/ x(ind(i)) - x(ind(j)), y(ind(i)) - y(ind(j)), z(ind(i)) - z(ind(j)) /)
               dxyz(1:3) = abs(dxyz(1:3))
               call pbc(dxyz)
               rcut = max(rcut,sum(dxyz(1:3)**2))
            endif
         enddo
      enddo

      return
    end subroutine calc_rcut_max

  end subroutine calc_rcut_grouped

#endif

end module domdec_grouped
