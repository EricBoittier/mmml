module domdec_shake

#if KEY_DOMDEC==1
  use chm_kinds
  use dimens_fcm
  use nblist_types,only:intarray_t
  implicit none
  private
  
  type pair_t
     sequence
     integer indi, indj
     real(chm_real) massi, massj
     real(chm_real) constr
  end type pair_t

  type trip_t
     sequence
     integer indi, indj, indk
     real(chm_real) mass1, mass2, mass3, mass4, mass5
     real(chm_real) constr1, constr2
  end type trip_t

  type quad_t
     sequence
     integer indi, indj, indk, indl
     real(chm_real) mass1, mass2, mass3, mass4, mass5, mass6, mass7
     real(chm_real) constr1, constr2, constr3
  end type quad_t

  type water_t
     sequence
     integer indO, indH1, indH2
  end type water_t

  ! Quick-to-built table 
  integer, allocatable, dimension(:,:) :: nshaketbl
  type(intarray_t), target, allocatable, dimension(:,:) :: shaketbl

  ! For each atom, shake type and index
  integer, allocatable, dimension(:) :: shaketype, shakeind

  ! Total tables (includes shakes in the entire system)
  integer shakepair_len, shaketrip_len, shakequad_len, shakewater_len
  type(pair_t), allocatable, dimension(:) :: shakepair_tbl
  type(trip_t), allocatable, dimension(:) :: shaketrip_tbl
  type(quad_t), allocatable, dimension(:) :: shakequad_tbl
  type(water_t), allocatable, dimension(:) :: shakewater_tbl

  ! Final shake tables with direct indexing
  integer nshakepair, nshaketrip, nshakequad, nshakewater
  integer, pointer, dimension(:) :: shakepair, shaketrip, shakequad, shakewater
  integer, allocatable, dimension(:) :: shakewater_ind
  integer, allocatable, dimension(:) :: shakepair_ind, shaketrip_ind, shakequad_ind
  real(chm_real), allocatable, dimension(:) :: shakepair_constr, shakepair_mass
  real(chm_real), allocatable, dimension(:) :: shaketrip_constr, shaketrip_mass
  real(chm_real), allocatable, dimension(:) :: shakequad_constr, shakequad_mass

  ! .true. if shake is used
  logical q_shake

  ! Oxygen and hydrogen masses and squared distances
  real(chm_real) mO, mH, rOHsq, rHHsq

  ! Public variables
  public nshakepair, nshaketrip, nshakequad, nshakewater
  public shakepair_ind, shakepair_constr, shakepair_mass
  public shaketrip_ind, shaketrip_constr, shaketrip_mass
  public shakequad_ind, shakequad_constr, shakequad_mass
  public shakewater_ind
  public q_shake, mO, mH, rOHsq, rHHsq

  ! Public subroutines
  public init_shake, uninit_shake
  public build_shaketbl

contains

  ! *
  ! * Allocates & reallocates shake
  ! *
  subroutine init_shake(nsh1_loc, nsh2_loc, nsh3_loc, nstwat_loc, numwater_loc, &
       bshkgp_loc, shkapr_loc, hmassi, hmassj, ammi, constr)
    use psf,only:natom
    use memory,only:chmalloc,chmdealloc
    use domdec_common,only:nthread
    use parallel,only:mynod, mynodp, numnod, comm_charmm
    use mpi
    use number,only:zero
    implicit none
    ! Input
    integer, intent(in) :: nsh1_loc, nsh2_loc, nsh3_loc
    integer, intent(in) :: nstwat_loc, numwater_loc
    integer, intent(in) :: bshkgp_loc(:), shkapr_loc(:,:)
    real(chm_real), intent(in) :: hmassi(:), hmassj(:), ammi(:), constr(:)
    ! Variables
    !integer start
    integer, allocatable, dimension(:,:) :: num_list
    integer, allocatable, dimension(:) :: nrecv, disp
    integer num_send(4)
    integer nsh1, nsh2, nsh3, nstwat, numwater
    type(pair_t), allocatable, dimension(:) :: shakepair_loc
    type(trip_t), allocatable, dimension(:) :: shaketrip_loc
    type(quad_t), allocatable, dimension(:) :: shakequad_loc
    type(water_t), allocatable, dimension(:) :: shakewater_loc
    real(chm_real), allocatable, dimension(:) :: buffer
    integer i, j, alloc_len(4)
    integer sizeof_pair_t, sizeof_trip_t, sizeof_quad_t, sizeof_water_t
    integer pair_base, trip_base, quad_base, water_base
    integer ierror

    !------------------------------------------------------------------------------------
    call chmalloc('domdec_shake.src','init_shake','num_list',4,numnod,intg=num_list)
    call chmalloc('domdec_shake.src','init_shake','nrecv',numnod,intg=nrecv)
    call chmalloc('domdec_shake.src','init_shake','disp',numnod,intg=disp)
    ! Combine counts
    num_send(1:4) = (/ nsh1_loc, nsh2_loc, nsh3_loc, numwater_loc /)
    call mpi_allgather(num_send, 4, mpi_integer, num_list, 4, mpi_integer, &
         comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_shake>','init_shake: Error in mpi_allgather')
    endif

    ! Calculate total number of shake constraints
    nsh1     = sum(num_list(1,1:numnod))
    nsh2     = sum(num_list(2,1:numnod))
    nsh3     = sum(num_list(3,1:numnod))
    numwater = sum(num_list(4,1:numnod))

    !------------------------------------------------------------------------------------

    ! Allocate space for (shakepair_tbl, shaketrip_tbl, shakequad_tbl, shakewater_tbl)

    shakepair_len = max(1, nsh1)
    shaketrip_len = max(1, nsh2)
    shakequad_len = max(1, nsh3)
    shakewater_len = max(1, numwater)
    
    ! shakepair_tbl
    if (allocated(shakepair_tbl)) then
       if (size(shakepair_tbl) /= shakepair_len) then
          deallocate(shakepair_tbl)
       endif
    endif
    if (.not.allocated(shakepair_tbl)) then
       allocate(shakepair_tbl(shakepair_len))
    endif

    ! shaketrip_tbl
    if (allocated(shaketrip_tbl)) then
       if (size(shaketrip_tbl) /= shaketrip_len) then
          deallocate(shaketrip_tbl)
       endif
    endif
    if (.not.allocated(shaketrip_tbl)) then
       allocate(shaketrip_tbl(shaketrip_len))
    endif
    
    ! shakequad_tbl
    if (allocated(shakequad_tbl)) then
       if (size(shakequad_tbl) /= shakequad_len) then
          deallocate(shakequad_tbl)
       endif
    endif
    if (.not.allocated(shakequad_tbl)) then
       allocate(shakequad_tbl(shakequad_len))
    endif

    ! shakewater_tbl
    if (allocated(shakewater_tbl)) then
       if (size(shakewater_tbl) /= shakewater_len) then
          deallocate(shakewater_tbl)
       endif
    endif
    if (.not.allocated(shakewater_tbl)) then
       allocate(shakewater_tbl(shakewater_len))
    endif

    ! shaketbl
    if (allocated(shaketbl)) then
       if (size(shaketbl,2) < nthread) then
          do i=0,nthread-1
             do j=1,4
                if (allocated(shaketbl(j,i)%array)) then
                   call chmdealloc('domdec_shake.src','init_shake','shaketbl(j,i)%array',&
                        size(shaketbl(j,i)%array),intg=shaketbl(j,i)%array)
                endif
             enddo
          enddo
          deallocate(shaketbl)
       endif
    endif

    if (.not.allocated(shaketbl)) then
       allocate(shaketbl(1:4,0:nthread-1))
    endif

    do i=0,nthread-1
       ! NOTE: we are allocating too much space here, should be nsh1*4/ndirect or so
       alloc_len(1:4) = (/ shakepair_len, shaketrip_len, shakequad_len, shakewater_len /)
       if (i == 0) alloc_len(1:4) = nthread*alloc_len(1:4)       
       do j=1,4
          if (allocated(shaketbl(j,i)%array)) then
             if (size(shaketbl(j,i)%array) < alloc_len(j)) then
                call chmdealloc('domdec_shake.src','init_shake','shaketbl(j,i)%array',&
                     size(shaketbl(j,i)%array),intg=shaketbl(j,i)%array)
             endif
          endif
          if (.not.allocated(shaketbl(j,i)%array)) then
             call chmalloc('domdec_shake.src','init_shake','shaketbl(j,i)%array',&
                  alloc_len(j),intg=shaketbl(j,i)%array)
          endif
       enddo
    enddo

    ! Associate pointers
    nullify(shakepair, shaketrip, shakequad, shakewater)
    shakepair  => shaketbl(1,0)%array
    shaketrip  => shaketbl(2,0)%array
    shakequad  => shaketbl(3,0)%array
    shakewater => shaketbl(4,0)%array

    ! nshaketbl
    if (allocated(nshaketbl)) then
       if (size(nshaketbl,2) < nthread) then
          call chmdealloc('domdec_shake.src','init_shake','nshaketbl',&
               size(nshaketbl,1),size(nshaketbl,2),intg=nshaketbl)
       endif
    endif

    if (.not.allocated(nshaketbl)) then
       call chmalloc('domdec_shake.src','init_shake','nshaketbl',&
            4,nthread,lbou2=0,intg=nshaketbl)
    endif

    ! Allocate shaketype and shakeind (both size natom)
    if (allocated(shaketype)) then
       if (size(shaketype) /= natom) then
          call chmdealloc('domdec_shake.src','init_shake',&
               'shaketype',size(shaketype),intg=shaketype)
          call chmdealloc('domdec_shake.src','init_shake',&
               'shakeind',size(shakeind),intg=shakeind)
       endif
    endif

    if (.not.allocated(shaketype)) then
       call chmalloc('domdec_shake.src','init_shake',&
            'shaketype',natom,intg=shaketype)
       call chmalloc('domdec_shake.src','init_shake',&
            'shakeind',natom,intg=shakeind)
    endif

    allocate(shakepair_loc(nsh1_loc))
    allocate(shaketrip_loc(nsh2_loc))
    allocate(shakequad_loc(nsh3_loc))
    allocate(shakewater_loc(numwater_loc))

    if (mynodp > 1) then
       pair_base = sum(num_list(1,1:mynodp-1))
       trip_base = sum(num_list(2,1:mynodp-1))
       quad_base = sum(num_list(3,1:mynodp-1))
       water_base = sum(num_list(4,1:mynodp-1))
    else
       pair_base = 0
       trip_base = 0
       quad_base = 0
       water_base = 0
    endif

    call build_shake_struct(nsh1_loc, nsh2_loc, nsh3_loc, nstwat_loc, numwater_loc, &
         bshkgp_loc, shkapr_loc, hmassi, hmassj, ammi, constr, &
         pair_base, trip_base, quad_base, water_base, &
         shaketype, shakeind, shakepair_loc, shaketrip_loc, shakequad_loc, shakewater_loc)

    ! Combine (shaketype, shakeind, shakepair_tbl, shaketrip_tbl, shakequad_tbl, shakewater_tbl)
    ! among nodes

    ! Add test bits to shaketype
    do i=1,natom
       if (shaketype(i) /= 0) then
          shaketype(i) = ior(shaketype(i), ishft(mynodp,3))
       endif
    enddo

    call mpi_allreduce(mpi_in_place, shaketype, natom, mpi_integer, mpi_sum, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_shake>','init_shake: Error in mpi_allreduce')
    endif

    ! Check and remove test bit from shaketype
    do i=1,natom
       if (shakeind(i) /= 0) then
          if (ishft(shaketype(i),-3) /= mynodp) then
             call wrndie(-5,'<domdec_shake>','Error combining shaketype')
          endif
       endif
       shaketype(i) = iand(shaketype(i), B'111')
    enddo

    call mpi_allreduce(mpi_in_place, shakeind, natom, mpi_integer, mpi_sum, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_shake>','init_shake: Error in mpi_allreduce')
    endif

    sizeof_pair_t = sizeof(shakepair_tbl(1))
    sizeof_trip_t = sizeof(shaketrip_tbl(1))
    sizeof_quad_t = sizeof(shakequad_tbl(1))
    sizeof_water_t = sizeof(shakewater_tbl(1))

    ! shakepair_tbl
    nrecv(1:numnod) = num_list(1,1:numnod)*sizeof_pair_t
    disp(1) = 0
    do i=2,numnod
       disp(i) = disp(i-1) + nrecv(i-1)
    enddo
    call mpi_allgatherv(shakepair_loc, num_list(1,mynodp)*sizeof_pair_t, mpi_byte, &
         shakepair_tbl, nrecv, disp, mpi_byte, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_shake>','init_shake: Error in mpi_allgatherv')
    endif

    ! shaketrip_tbl
    nrecv(1:numnod) = num_list(2,1:numnod)*sizeof_trip_t
    disp(1) = 0
    do i=2,numnod
       disp(i) = disp(i-1) + nrecv(i-1)
    enddo
    call mpi_allgatherv(shaketrip_loc, num_list(2,mynodp)*sizeof_trip_t, mpi_byte, &
         shaketrip_tbl, nrecv, disp, mpi_byte, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_shake>','init_shake: Error in mpi_allgatherv')
    endif

    ! shakequad_tbl
    nrecv(1:numnod) = num_list(3,1:numnod)*sizeof_quad_t
    disp(1) = 0
    do i=2,numnod
       disp(i) = disp(i-1) + nrecv(i-1)
    enddo
    call mpi_allgatherv(shakequad_loc, num_list(3,mynodp)*sizeof_quad_t, mpi_byte, &
         shakequad_tbl, nrecv, disp, mpi_byte, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_shake>','init_shake: Error in mpi_allgatherv')
    endif

    ! shakewater_tbl
    nrecv(1:numnod) = num_list(4,1:numnod)*sizeof_water_t
    disp(1) = 0
    do i=2,numnod
       disp(i) = disp(i-1) + nrecv(i-1)
    enddo
    call mpi_allgatherv(shakewater_loc, num_list(4,mynodp)*sizeof_water_t, mpi_byte, &
         shakewater_tbl, nrecv, disp, mpi_byte, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_shake>','init_shake: Error in mpi_allgatherv')
    endif
    
    deallocate(shakepair_loc)
    deallocate(shaketrip_loc)
    deallocate(shakequad_loc)
    deallocate(shakewater_loc)

    call chmdealloc('domdec_shake.src','init_shake','num_list',size(num_list,1),size(num_list,2),&
         intg=num_list)
    call chmdealloc('domdec_shake.src','init_shake','nrecv',size(nrecv),intg=nrecv)
    call chmdealloc('domdec_shake.src','init_shake','nrecv',size(disp),intg=disp)

    !---------------------------------------------
    ! Combine water O and H masses and distances
    !---------------------------------------------
    call chmalloc('domdec_shake.src','init_shake','buffer',numnod,crl=buffer)

    call mpi_allgather(mO, 1, mpi_real8, buffer, 1, mpi_real8, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_shake>','init_shake: Error calling mpi_alltoall')
    endif
    do i=1,numnod
       if (mO == zero .and. buffer(i) /= zero) mO = buffer(i)
       if (mO /= zero .and. buffer(i) /= zero .and. mO /= buffer(i)) then
          call wrndie(-5,'<domdec_shake>','init_shake: Water O masses do not agree')
       endif
    enddo
       
    call mpi_allgather(mH, 1, mpi_real8, buffer, 1, mpi_real8, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_shake>','init_shake: Error calling mpi_alltoall')
    endif
    do i=1,numnod
       if (mH == zero .and. buffer(i) /= zero) mH = buffer(i)
       if (mH /= zero .and. buffer(i) /= zero .and. mH /= buffer(i)) then
          call wrndie(-5,'<domdec_shake>','init_shake: Water H masses do not agree')
       endif
    enddo
    
    call mpi_allgather(rOHsq, 1, mpi_real8, buffer, 1, mpi_real8, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_shake>','init_shake: Error calling mpi_alltoall')
    endif
    do i=1,numnod
       if (rOHsq == zero .and. buffer(i) /= zero) rOHsq = buffer(i)
       if (rOHsq /= zero .and. buffer(i) /= zero .and. rOHsq /= buffer(i)) then
          call wrndie(-5,'<domdec_shake>','init_shake: Water O-H distances do not agree')
       endif
    enddo
    
    call mpi_allgather(rHHsq, 1, mpi_real8, buffer, 1, mpi_real8, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_shake>','init_shake: Error calling mpi_alltoall')
    endif
    do i=1,numnod
       if (rHHsq == zero .and. buffer(i) /= zero) rHHsq = buffer(i)
       if (rHHsq /= zero .and. buffer(i) /= zero .and. rHHsq /= buffer(i)) then
          call wrndie(-5,'<domdec_shake>','init_shake: Water H-H distances do not agree')
       endif
    enddo
    
    call chmdealloc('domdec_shake.src','init_shake','buffer',numnod,crl=buffer)

    if (numwater > 0) then
       if (mO == zero .or. mH == zero .or. rOHsq == zero .or. rHHsq == zero) then
          call wrndie(-5,'<domdec_shake>',&
               'init_shake: Unable to determine O/H masses and distances')
       endif
    endif

    return
  end subroutine init_shake

  ! *
  ! * Builds shake lists for each atom
  ! * Only the atom with the lowest index has the list
  ! * NOTE: only called once in initialization
  ! *
  subroutine build_shake_struct(nsh1, nsh2, nsh3, nstwat, numwater, &
       bshkgp, shkapr, hmassi, hmassj, ammi, constr, &
       pair_base, trip_base, quad_base, water_base, &
       shaketype, shakeind, shakepair_tbl, shaketrip_tbl, shakequad_tbl, &
       shakewater_tbl)
    use stream,only:outu
    use psf,only:natom, amass
    use groupxfast,only:invgroup
    use number,only:zero
    implicit none
    ! Input / Output
    integer, intent(in) :: nsh1, nsh2, nsh3, nstwat, numwater
    integer, intent(in) :: bshkgp(:), shkapr(:,:)
    real(chm_real), intent(in) :: hmassi(:), hmassj(:), ammi(:), constr(:)
    integer, intent(in) :: pair_base, trip_base, quad_base, water_base
    integer, intent(out) :: shaketype(:), shakeind(:)
    type(pair_t), intent(out) :: shakepair_tbl(:)
    type(trip_t), intent(out) :: shaketrip_tbl(:)
    type(quad_t), intent(out) :: shakequad_tbl(:)
    type(water_t), intent(out) :: shakewater_tbl(:)
    ! Variables
    integer ii, ll, t, i2
    integer ierror

    shaketype(1:natom) = 0
    shakeind(1:natom) = 0

    ! Bonds
    do ii=1,nsh1
       ll = bshkgp(ii)
       t = min(shkapr(1,ll),shkapr(2,ll))
       if (shaketype(t) /= 0) then
          call wrndie(-5,'<domdec_shake>','Overwriting shake list at bond')
       endif
       shaketype(t) = 1
       shakeind(t) = pair_base + ii
       shakepair_tbl(ii)%indi = shkapr(1,ll)
       shakepair_tbl(ii)%indj = shkapr(2,ll)
       shakepair_tbl(ii)%massi = hmassi(ll)
       shakepair_tbl(ii)%massj = hmassj(ll)
       shakepair_tbl(ii)%constr = constr(ll)
       if (invgroup(shkapr(1,ll)) /= invgroup(shkapr(2,ll))) then
          write (outu,'(a,2i6,a,2i6)') 'groups:',&
               invgroup(shkapr(1,ll)),invgroup(shkapr(2,ll)),' atoms:',&
               shkapr(1,ll),shkapr(2,ll)
          call wrndie(-5,'<domdec_shake>',&
               'build_shake_struct: Shake cannot cross group boundaries')
       endif
    enddo

    ! Angles
    i2 = 2*nsh1 + 1
    do ii=1,nsh2
       ll = bshkgp(nsh1 + ii)
       t = min(shkapr(1,ll),shkapr(2,ll),shkapr(2,ll+1))
       if (shaketype(t) /= 0) then
          call wrndie(-5,'<domdec_shake>','Overwriting shake list at angle')
       endif
       shaketype(t) = 2
       shakeind(t) = trip_base + ii
       shaketrip_tbl(ii)%indi = shkapr(1,ll)
       shaketrip_tbl(ii)%indj = shkapr(2,ll)
       shaketrip_tbl(ii)%indk = shkapr(2,ll+1)
       shaketrip_tbl(ii)%mass1 = ammi(i2)
       shaketrip_tbl(ii)%mass2 = ammi(i2+1)
       shaketrip_tbl(ii)%mass3 = ammi(i2+2)
       shaketrip_tbl(ii)%mass4 = ammi(i2+3)
       shaketrip_tbl(ii)%mass5 = ammi(i2+4)
       shaketrip_tbl(ii)%constr1 = constr(ll)
       shaketrip_tbl(ii)%constr2 = constr(ll+1)
       if (invgroup(shkapr(1,ll)) /= invgroup(shkapr(2,ll)) .or. &
            invgroup(shkapr(1,ll)) /= invgroup(shkapr(2,ll+1)) .or. &
            invgroup(shkapr(2,ll)) /= invgroup(shkapr(2,ll+1))) then
          write (outu,'(a,3i6)') 'groups:',&
               invgroup(shkapr(1,ll)),invgroup(shkapr(2,ll)),&
               invgroup(shkapr(2,ll+1))
          call wrndie(-5,'<domdec_shake>',&
               'build_shake_struct: Shake cannot cross group boundaries')
       endif
       i2 = i2 + 5
    enddo

    ! Dihedrals
    i2 = 2*nsh1 + 5*nsh2 + 1
    do ii=1,nsh3
       ll = bshkgp(nsh1 + nsh2 + ii)
       t = min(shkapr(1,ll),shkapr(2,ll),shkapr(2,ll+1),shkapr(2,ll+2))
       if (shaketype(t) /= 0) then
          call wrndie(-5,'<domdec_shake>','Overwriting shake list at dihedrals')
       endif
       shaketype(t) = 3
       shakeind(t) = quad_base + ii
       shakequad_tbl(ii)%indi = shkapr(1,ll)
       shakequad_tbl(ii)%indj = shkapr(2,ll)
       shakequad_tbl(ii)%indk = shkapr(2,ll+1)
       shakequad_tbl(ii)%indl = shkapr(2,ll+2)
       shakequad_tbl(ii)%mass1 = ammi(i2)
       shakequad_tbl(ii)%mass2 = ammi(i2+1)
       shakequad_tbl(ii)%mass3 = ammi(i2+2)
       shakequad_tbl(ii)%mass4 = ammi(i2+3)
       shakequad_tbl(ii)%mass5 = ammi(i2+4)
       shakequad_tbl(ii)%mass6 = ammi(i2+5)
       shakequad_tbl(ii)%mass7 = ammi(i2+6)
       shakequad_tbl(ii)%constr1 = constr(ll)
       shakequad_tbl(ii)%constr2 = constr(ll+1)
       shakequad_tbl(ii)%constr3 = constr(ll+2)
       if (invgroup(shkapr(1,ll)) /= invgroup(shkapr(2,ll)) .or. &
            invgroup(shkapr(1,ll)) /= invgroup(shkapr(2,ll+1)) .or. &
            invgroup(shkapr(1,ll)) /= invgroup(shkapr(2,ll+2)) .or. &
            invgroup(shkapr(2,ll)) /= invgroup(shkapr(2,ll+1)) .or. &
            invgroup(shkapr(2,ll)) /= invgroup(shkapr(2,ll+2)) .or. &
            invgroup(shkapr(2,ll+1)) /= invgroup(shkapr(2,ll+2))) then
          write (outu,'(a,4i6)') 'groups:',&
               invgroup(shkapr(1,ll)),invgroup(shkapr(2,ll)),&
               invgroup(shkapr(2,ll+1)),invgroup(shkapr(2,ll+2))
          call wrndie(-5,'<domdec_shake>',&
               'build_shake_struct: Shake cannot cross group boundaries')
       endif
       i2 = i2 + 7
    enddo

    ! Get O and H masses and distances
    if (numwater > 0) then
       mO = amass(shkapr(1,nstwat))
       mH = amass(shkapr(2,nstwat))
       rOHsq = constr(nstwat)
       rHHsq = constr(nstwat+2)
    else
       mO = zero
       mH = zero
       rOHsq = zero
       rHHsq = zero
    endif

    ! Solvents
    do ii = 1, numwater
       ll = 3*(ii-1) + nstwat
       t = min(shkapr(1,ll),shkapr(2,ll),shkapr(2,ll+1))
       if (shaketype(t) /= 0) then
          call wrndie(-5,'<domdec_shake>','Overwriting shake list at solvents')
       endif
       shaketype(t) = 4
       shakeind(t) = water_base + ii
       shakewater_tbl(ii)%indO  = shkapr(1,ll)
       shakewater_tbl(ii)%indH1 = shkapr(2,ll)
       shakewater_tbl(ii)%indH2 = shkapr(2,ll+1)
       if (rOHsq /= constr(ll) .or. rOHsq /= constr(ll+1)) then
          call wrndie(-5,'<domdec_shake>','build_shake_struct: Water O-H distances do not agree')
       endif
    enddo

    return
  end subroutine build_shake_struct

  ! *
  ! * Builds shake tables
  ! *
  ! * NOTE: called after update_groupl is done
  ! * NOTE: requires that homezone values are up to date
  ! *
  subroutine build_shaketbl()
    use nblist_util,only:reduce_intarray
    use groupxfast,only:group, group_out
    use domdec_common,only:zonelist, groupl, nthread
#if KEY_DOMDEC_GPU==1
    use domdec_util_gpu_mod,only:range_start, range_stop
    use domdec_common,only:q_gpu
#endif
    implicit none
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num
#endif
    ! Variables
    integer tid, jg, igrp, is, iq, i, bt, bi, ll

    if (.not.q_shake) return

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('build_shaketbl')
#endif

!$omp parallel private(tid)
#ifdef _OPENMP
    tid = omp_get_thread_num()
#else
    tid = 0
#endif
    nshaketbl(1:4,tid) = 0
    ! Home box, both solutes and solvents
!$omp do schedule(static) private(jg, igrp, is, iq, i, bt, bi)
    do jg=1,zonelist(1)
       igrp = groupl(jg)
       call group_out(group(igrp), is, iq)
       ! Automatically add, no check required
       do i=is,iq
          bt = shaketype(i)
          bi = shakeind(i)
          if (bt > 0) then
             nshaketbl(bt,tid) = nshaketbl(bt,tid) + 1
             shaketbl(bt,tid)%array(nshaketbl(bt,tid)) = bi
          endif
       enddo
    enddo
!$omp end do
!$omp end parallel

    call reduce_intarray(4, nthread, nshaketbl, shaketbl)
    
    nshakepair = nshaketbl(1, 0)
    nshaketrip = nshaketbl(2, 0)
    nshakequad = nshaketbl(3, 0)
    nshakewater = nshaketbl(4, 0)

    ! Allocate / reallocate (shakepair_, shaketrip_, shakequad_, shakewater_)
    call alloc_realloc(nshakepair, nshaketrip, nshakequad, nshakewater)

    !
    ! Change into direct accessing 
    !
!$omp parallel do schedule(static) private(i, ll)
    do i=1,nshakewater
       ll = shakewater(i)
       shakewater_ind(3*i-2) = shakewater_tbl(ll)%indO
       shakewater_ind(3*i-1) = shakewater_tbl(ll)%indH1
       shakewater_ind(3*i)   = shakewater_tbl(ll)%indH2
    enddo
!$omp end parallel do

!$omp parallel do schedule(static) private(i, ll)
    do i=1,nshakepair
       ll = shakepair(i)
       shakepair_ind(i*2-1)  = shakepair_tbl(ll)%indi
       shakepair_ind(i*2)    = shakepair_tbl(ll)%indj
       shakepair_mass(i*2-1) = shakepair_tbl(ll)%massi
       shakepair_mass(i*2)   = shakepair_tbl(ll)%massj
       shakepair_constr(i)   = shakepair_tbl(ll)%constr
    enddo
!$omp end parallel do

!$omp parallel do schedule(static) private(i, ll)
    do i=1,nshaketrip
       ll = shaketrip(i)
       shaketrip_ind(i*3-2)    = shaketrip_tbl(ll)%indi
       shaketrip_ind(i*3-1)    = shaketrip_tbl(ll)%indj
       shaketrip_ind(i*3)      = shaketrip_tbl(ll)%indk
       shaketrip_mass(i*5-4)   = shaketrip_tbl(ll)%mass1
       shaketrip_mass(i*5-3)   = shaketrip_tbl(ll)%mass2
       shaketrip_mass(i*5-2)   = shaketrip_tbl(ll)%mass3
       shaketrip_mass(i*5-1)   = shaketrip_tbl(ll)%mass4
       shaketrip_mass(i*5)     = shaketrip_tbl(ll)%mass5
       shaketrip_constr(i*2-1) = shaketrip_tbl(ll)%constr1
       shaketrip_constr(i*2)   = shaketrip_tbl(ll)%constr2
    enddo
!$omp end parallel do

!$omp parallel do schedule(static) private(i, ll)
    do i=1,nshakequad
       ll = shakequad(i)
       shakequad_ind(i*4-3)    = shakequad_tbl(ll)%indi
       shakequad_ind(i*4-2)    = shakequad_tbl(ll)%indj
       shakequad_ind(i*4-1)    = shakequad_tbl(ll)%indk
       shakequad_ind(i*4)      = shakequad_tbl(ll)%indl
       shakequad_mass(i*7-6)   = shakequad_tbl(ll)%mass1
       shakequad_mass(i*7-5)   = shakequad_tbl(ll)%mass2
       shakequad_mass(i*7-4)   = shakequad_tbl(ll)%mass3
       shakequad_mass(i*7-3)   = shakequad_tbl(ll)%mass4
       shakequad_mass(i*7-2)   = shakequad_tbl(ll)%mass5
       shakequad_mass(i*7-1)   = shakequad_tbl(ll)%mass6
       shakequad_mass(i*7)     = shakequad_tbl(ll)%mass7
       shakequad_constr(i*3-2) = shakequad_tbl(ll)%constr1
       shakequad_constr(i*3-1) = shakequad_tbl(ll)%constr2
       shakequad_constr(i*3)   = shakequad_tbl(ll)%constr3
    enddo
!$omp end parallel do

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

    return

  contains
    subroutine alloc_realloc(nshakepair, nshaketrip, nshakequad, nshakewater)
      use memory,only:chmalloc, chmdealloc
      implicit none
      ! Input
      integer, intent(in) :: nshakepair, nshaketrip, nshakequad, nshakewater
      ! Variables
      integer nshakepair_new, nshaketrip_new, nshakequad_new, nshakewater_new

      nshakepair_new = min(int(1.5*nshakepair), shakepair_len)
      nshaketrip_new = min(int(1.5*nshaketrip), shaketrip_len)
      nshakequad_new = min(int(1.5*nshakequad), shakequad_len)
      nshakewater_new = min(int(1.5*nshakewater), shakewater_len)

      ! (shakewater_ind)
      if (allocated(shakewater_ind)) then
         if (size(shakewater_ind) < nshakewater*3) then
            call chmdealloc('domdec_shake.src','build_shaketbl','shakewater_ind',&
                 size(shakewater_ind),intg=shakewater_ind)
         endif
      endif
      
      if (.not.allocated(shakewater_ind)) then
         call chmalloc('domdec_shake.src','build_shaketbl','shakewater_ind',&
              nshakewater_new*3,intg=shakewater_ind)
      endif

      ! (shakepair_ind, shakepair_constr, shakepair_mass)
      if (allocated(shakepair_ind)) then
         if (size(shakepair_ind) < nshakepair*2) then
            call chmdealloc('domdec_shake.src','build_shaketbl','shakepair_ind',&
                 size(shakepair_ind),intg=shakepair_ind)
            call chmdealloc('domdec_shake.src','build_shaketbl','shakepair_constr',&
                 size(shakepair_constr),crl=shakepair_constr)
            call chmdealloc('domdec_shake.src','build_shaketbl','shakepair_mass',&
                 size(shakepair_mass),crl=shakepair_mass)
         endif
      endif

      if (.not.allocated(shakepair_ind)) then
         call chmalloc('domdec_shake.src','build_shaketbl','shakepair_ind',&
              nshakepair_new*2,intg=shakepair_ind)
         call chmalloc('domdec_shake.src','build_shaketbl','shakepair_constr',&
              nshakepair_new*1,crl=shakepair_constr)
         call chmalloc('domdec_shake.src','build_shaketbl','shakepair_mass',&
              nshakepair_new*2,crl=shakepair_mass)
      endif

      ! (shaketrip_ind, shaketrip_constr, shaketrip_mass)
      if (allocated(shaketrip_ind)) then
         if (size(shaketrip_ind) < nshaketrip*3) then
            call chmdealloc('domdec_shake.src','build_shaketbl','shaketrip_ind',&
                 size(shaketrip_ind),intg=shaketrip_ind)
            call chmdealloc('domdec_shake.src','build_shaketbl','shaketrip_constr',&
                 size(shaketrip_constr),crl=shaketrip_constr)
            call chmdealloc('domdec_shake.src','build_shaketbl','shaketrip_mass',&
                 size(shaketrip_mass),crl=shaketrip_mass)
         endif
      endif

      if (.not.allocated(shaketrip_ind)) then
         call chmalloc('domdec_shake.src','build_shaketbl','shaketrip_ind',&
              nshaketrip_new*3,intg=shaketrip_ind)
         call chmalloc('domdec_shake.src','build_shaketbl','shaketrip_constr',&
              nshaketrip_new*2,crl=shaketrip_constr)
         call chmalloc('domdec_shake.src','build_shaketbl','shaketrip_mass',&
              nshaketrip_new*5,crl=shaketrip_mass)
      endif

      ! (shakequad_ind, shakequad_constr, shakequad_mass)
      if (allocated(shakequad_ind)) then
         if (size(shakequad_ind) < nshakequad*4) then
            call chmdealloc('domdec_shake.src','build_shaketbl','shakequad_ind',&
                 size(shakequad_ind),intg=shakequad_ind)
            call chmdealloc('domdec_shake.src','build_shaketbl','shakequad_constr',&
                 size(shakequad_constr),crl=shakequad_constr)
            call chmdealloc('domdec_shake.src','build_shaketbl','shakequad_mass',&
                 size(shakequad_mass),crl=shakequad_mass)
         endif
      endif

      if (.not.allocated(shakequad_ind)) then
         call chmalloc('domdec_shake.src','build_shaketbl','shakequad_ind',&
              nshakequad_new*4,intg=shakequad_ind)
         call chmalloc('domdec_shake.src','build_shaketbl','shakequad_constr',&
              nshakequad_new*3,crl=shakequad_constr)
         call chmalloc('domdec_shake.src','build_shaketbl','shakequad_mass',&
              nshakequad_new*7,crl=shakequad_mass)
      endif

      return
    end subroutine alloc_realloc

  end subroutine build_shaketbl

  ! *
  ! * Deallocates shake
  ! *
  subroutine uninit_shake()
    use memory,only:chmdealloc
    use domdec_common,only:nthread
    implicit none
    integer i, j

    ! shakepair_tbl
    if (allocated(shakepair_tbl)) then
       deallocate(shakepair_tbl)
    endif

    ! shaketrip_tbl
    if (allocated(shaketrip_tbl)) then
       deallocate(shaketrip_tbl)
    endif
    
    ! shakequad_tbl
    if (allocated(shakequad_tbl)) then
       deallocate(shakequad_tbl)
    endif

    ! shakewater_tbl
    if (allocated(shakewater_tbl)) then
       deallocate(shakewater_tbl)
    endif

    if (allocated(shaketbl)) then
       do i=0,nthread-1
          do j=1,4
             if (allocated(shaketbl(j,i)%array)) then
                call chmdealloc('domdec_shake.src','uninit_shake','shaketbl(j,i)%array',&
                     size(shaketbl(j,i)%array),intg=shaketbl(j,i)%array)
             endif
          enddo
       enddo
       deallocate(shaketbl)
       nullify(shakepair, shaketrip, shakequad, shakewater)
    endif

    if (allocated(nshaketbl)) then
       call chmdealloc('domdec_shake.src','uninit_shake','nshaketbl',&
            size(nshaketbl,1),size(nshaketbl,2),intg=nshaketbl)
    endif

    if (allocated(shaketype)) then
       call chmdealloc('domdec_shake.src','uninit_shake',&
            'shaketype',size(shaketype),intg=shaketype)
       call chmdealloc('domdec_shake.src','uninit_shake',&
            'shakeind',size(shakeind),intg=shakeind)
    endif

    ! (shakewater_ind)
    if (allocated(shakewater_ind)) then
       call chmdealloc('domdec_shake.src','uninit_shake','shakewater_ind',&
            size(shakewater_ind),intg=shakewater_ind)
    endif

    ! (shakepair_ind, shakepair_constr, shakepair_mass)
    if (allocated(shakepair_ind)) then
       call chmdealloc('domdec_shake.src','uninit_shake','shakepair_ind',&
            size(shakepair_ind),intg=shakepair_ind)
       call chmdealloc('domdec_shake.src','uninit_shake','shakepair_constr',&
            size(shakepair_constr),crl=shakepair_constr)
       call chmdealloc('domdec_shake.src','uninit_shake','shakepair_mass',&
            size(shakepair_mass),crl=shakepair_mass)
    endif
    
    ! (shaketrip_ind, shaketrip_constr, shaketrip_mass)
    if (allocated(shaketrip_ind)) then
       call chmdealloc('domdec_shake.src','uninit_shake','shaketrip_ind',&
            size(shaketrip_ind),intg=shaketrip_ind)
       call chmdealloc('domdec_shake.src','uninit_shake','shaketrip_constr',&
            size(shaketrip_constr),crl=shaketrip_constr)
       call chmdealloc('domdec_shake.src','uninit_shake','shaketrip_mass',&
            size(shaketrip_mass),crl=shaketrip_mass)
    endif
    
    ! (shakequad_ind, shakequad_constr, shakequad_mass)
    if (allocated(shakequad_ind)) then
       call chmdealloc('domdec_shake.src','uninit_shake','shakequad_ind',&
            size(shakequad_ind),intg=shakequad_ind)
       call chmdealloc('domdec_shake.src','uninit_shake','shakequad_constr',&
            size(shakequad_constr),crl=shakequad_constr)
       call chmdealloc('domdec_shake.src','uninit_shake','shakequad_mass',&
            size(shakequad_mass),crl=shakequad_mass)
    endif

    return
  end subroutine uninit_shake

#endif

end module domdec_shake
