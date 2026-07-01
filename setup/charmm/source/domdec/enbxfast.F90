module enbxfast

#if KEY_DOMDEC==1 /*domdec_main*/

  use chm_kinds
  use dimens_fcm
  use nblist_types,only:nb_pair_t,nblist_t,dpsp_t
  implicit none
  private

  enum, bind(C)
     enumerator :: NONE = 0
     ! Van der Waals
     enumerator :: VSH = 1, VSW = 2, VFSW = 3, VGSH = 4, VIPS = 6
     ! Electrostatic
     enumerator :: EWALD=101, CSHIFT=102, CFSWIT=103, CSHFT=104, CSWIT=105, RSWIT=106
     enumerator :: RSHFT=107, RSHIFT=108, RFSWIT=109, GSHFT=110, EIPS=121
  end enum

  integer, save :: vdwmodel = NONE   ! Current VdW force model setting
  integer, save :: elecmodel = NONE  ! Current electrostatic force model setting

#if KEY_DOMDEC_GPU==1
  ! .true. if the homezone nonbonded forces are already calculated
  logical :: q_home_calculated = .false.
#endif
  ! .true. if call to enbcalc_xfast_home originated from the main loop in dynamc.src
  logical :: q_in_dynamc_main_loop = .false.

  ! VdW parameters in a packed form, regular and 1-4 interaction version
  type(dpsp_t) :: vdwparam, vdwparam14

  ! Public subroutines
  public init_nblist, enbcalc_xfast, calc_virial, test_build_pftable
  public init_vdwparam, enbcalc_xfast_home
  public calc_virial_noshift
  public uninit_vdwparam, uninit_nblist

  ! Public variables
  public elecmodel,vdwmodel, vdwparam14, vdwparam
  public q_in_dynamc_main_loop

contains

  ! *
  ! * Launches the home non-bonded calculation for GPUs
  ! * NOTE: This only does computation when called within the
  ! *       main dynamic loop in dynamc.src (Actual call is in nbonds/heurist.src)
  ! *       This is in contrast to calls from "energy"
  ! *       that will not perform computation here since in these cases
  ! *       q_in_dynamc_main_loop = .false.
  ! *
  subroutine enbcalc_xfast_home()
#if KEY_DOMDEC_GPU==1
    use reawri,only:qcnstp
    use domdec_common,only:q_gpu, q_test, q_print_output, boxx, boxy, boxz
    use enb_core_gpu_mod,only:enb_tilex_gpu
    use domdec_util_gpu_mod,only:clear_force_virial_energy_gpu
#if KEY_BLOCK==1
    use lambdam,only:qmld, nblock, ninter, bixlam, blcoep, msld_setblcoef
    use domdec_block,only:set_block_params_to_gpu
#endif
#endif
    implicit none
#if KEY_DOMDEC_GPU==1
    logical q_calc_energy, q_calc_virial
    integer vdwmodel, elecmodel
    
    if (q_gpu .and. q_in_dynamc_main_loop) then
#if KEY_BLOCK==1
       if (qmld) then
          ! Set block parameters
          call msld_setblcoef(nblock,ninter,bixlam,blcoep)
          ! Copy block parameters (fullblcoep and bixlam) to GPU
          call set_block_params_to_gpu()
       endif
#endif
       q_calc_energy = q_print_output .or. q_test
       q_calc_virial = q_print_output .or. q_test .or. qcnstp
#if KEY_BLOCK==1
       ! For Lambda-dynamics, we always calculate energies as they're required in biflam/biflam2 computation
       ! if (qmld) q_calc_energy = .true.
       ! Let's not do that, as it wastes about 25% computation time.
       ! Rewrote domdec_gpu/CudaDirectForce_util.h to calculate only
       ! necessary forces. -RLH 2018-01-23
#endif
       ! Clear forces and optionally virial and energies
       call clear_force_virial_energy_gpu(q_calc_energy, q_calc_virial)
       ! Calculate non-bonded forces on GPU
       q_home_calculated = .true.
       ! In NPT box sizes change at every time step
       call enb_tilex_gpu(1, q_calc_energy, q_calc_virial, boxx, boxy, boxz)
    endif
#endif

    return
  end subroutine enbcalc_xfast_home
  
  ! *
  ! * Calculates non-bonded interactions using the neighborlists given in nblist
  ! *
  subroutine enbcalc_xfast(x, y, z, charge, forcex, forcey, forcez, &
       coulpot, vdwpot, excoulpot, lused)
    use number
    use psf,only:natom
    use memory
    use parallel,only:mynod,numnod
    use energym,only:qeterm,vdw,elec,ewexcl,qdyncall, ewksum
    use reawri,only:qcnstp
    use domdec_common,only:zonelist,groupl,natoml_tot,q_sort_groups,energy_time,q_single
    use groupxfast,only:group
    use nblist_util,only:pack_vdwtype, init_array
    use nblist_types,only:nblist, nb_pair
    use domdec_local,only:xyzq_loc, force_loc, q_force_loc_used, pack_xyz, pack_xyzq, scoordtab, &
         sforce, vdwtype_loc
    !use inbnd,only:ctonnb,ctofnb
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu, q_test, q_print_output, boxx, boxy, boxz
    use enb_core_gpu_mod,only:enb_tilex_gpu
    use domdec_util_gpu_mod,only:clear_force_virial_energy_gpu, copy_vdwtype_to_gpu, &
         range_start, range_stop
#if KEY_BLOCK==1
    use domdec_block,only:set_block_params_to_gpu
#endif    
#endif
#if KEY_BLOCK==1
    use lambdam,only:qmld, biflam, biflam2, nblock, ninter, bixlam, blcoep, msld_setblcoef
    use domdec_block,only:biflam_loc, biflam2_loc
#endif
    use mpi
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: x(*), y(*), z(*), charge(*)
    real(chm_real), intent(inout) :: forcex(*), forcey(*), forcez(*)
    real(chm_real), intent(out) :: coulpot, vdwpot, excoulpot
    logical, intent(out) :: lused
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num 
#endif
    ! Variables
    real(chm_real) sforce_loc(3*27), coulpot_loc, vdwpot_loc, excoulpot_loc
#if KEY_DOMDEC_GPU==1
    logical q_calc_energy, q_calc_virial
#endif 
    integer nnnb, tid
    ! For timing
    real(chm_real) time_start, time_stop, elapsed_time

!    call timer_stop(T_dir)

    call init_vdwparam(.false.)

    vdwpot = zero
    coulpot = zero
!    excoulpot = zero

    nnnb = 0

#if KEY_DOMDEC_GPU==1
    if (q_gpu) then
       ! Calculate non-bonded forces on GPU
       q_calc_energy = (.not.qdyncall) .or. q_print_output .or. q_test
       q_calc_virial = (.not.qdyncall) .or. q_print_output .or. q_test .or. qcnstp
#if KEY_BLOCK==1
       ! For Lambda-dynamics, we always calculate energies as they're required in biflam/biflam2 computation
       ! if (qmld) q_calc_energy = .true.
       ! Let's not do that, as it wastes about 25% computation time.
       ! Rewrote domdec_gpu/CudaDirectForce_util.h to calculate only
       ! necessary forces. -RLH 2018-01-23
#endif
       if (.not.q_home_calculated) then
#if KEY_BLOCK==1
          if (qmld) then
             ! Set block parameters
             call msld_setblcoef(nblock, ninter, bixlam, blcoep)
             ! Copy block parameters (fullblcoep and bixlam) to GPU
             call set_block_params_to_gpu()
          endif
#endif
          ! Clear forces and optionally virial and energies
          call clear_force_virial_energy_gpu(q_calc_energy, q_calc_virial)
          call enb_tilex_gpu(1, q_calc_energy, q_calc_virial, boxx, boxy, boxz)
       endif
       call enb_tilex_gpu(2, q_calc_energy, q_calc_virial, boxx, boxy, boxz)

       q_home_calculated = .false.

       !-----------------------------------------------------------------------
       !### APH 10/24/14 Force zeroing moved to happen at unpack_reduce_forces
       !-----------------------------------------------------------------------
       ! Zero local (per thread) force arrays
       ! NOTE: These arrays are used for bonded and 1-4 interaction calculations on the CPU
       !if (q_gpu) call range_start('zero_force_loc')  
       !call zero_force_loc()
       !if (q_gpu) call range_stop()  

    else
#endif 
       if (.not.nblist%q_pack) then
          call wrndie(-5,'<enbxfast>','Must use packed coordinates')
       endif

       ! Pack and recenter coordinates
       ! NOTE: array xyzq_loc is already initialized in domdec_local
       
       ! NOTE: for CPU version, zero_force_loc() was called in 
       !       build_local_coord() / update_local_coord()
       
       !    call timer_start(T_dir)
       
       time_start = mpi_wtime()

       sforce_loc = zero
       vdwpot_loc = zero
       coulpot_loc = zero
       excoulpot_loc = zero
#if KEY_BLOCK==1
       if (qmld) then
          biflam_loc = zero
          biflam2_loc = zero
       endif
#endif
!$omp parallel private(tid) &
!$omp& reduction(+:nnnb, sforce_loc, vdwpot_loc, coulpot_loc, excoulpot_loc)
#ifdef _OPENMP
       tid = omp_get_thread_num()
#else
       tid = 0
#endif
       call enbcalc_xfast_cpu_loop(qeterm(vdw), qeterm(elec), qeterm(ewexcl), nblist, nb_pair, &
            vdwtype_loc, vdwparam, vdwparam14, xyzq_loc, force_loc(tid)%array, sforce_loc, &
            coulpot_loc, vdwpot_loc, excoulpot_loc, &
            scoordtab, nnnb, q_single)
!$omp end parallel
       q_force_loc_used = .true.
       sforce = sforce + sforce_loc
       vdwpot = vdwpot + vdwpot_loc
       coulpot = coulpot + coulpot_loc
       excoulpot = excoulpot + excoulpot_loc
#if KEY_BLOCK==1
       if (qmld) then
          biflam = biflam + biflam_loc
          biflam2 = biflam2 + biflam2_loc
       endif
#endif

       time_stop = mpi_wtime()
       elapsed_time = time_stop - time_start
       energy_time = energy_time + elapsed_time

       !   call timer_stop(T_dir)
       
#if KEY_DOMDEC_GPU==1
    endif  
#endif

    lused = .true.

!    call timer_start(T_dir)

    return
  end subroutine enbcalc_xfast

  ! *
  ! * Non-bonded force calculations for CPU
  ! *
  subroutine enbcalc_xfast_cpu_loop(q_vdw, q_elec, q_ewexcl, nblist, nb_pair, vdwtype, &
       vdwparam, vdwparam14, xyzq, force, sforce, coulpot, vdwpot, excoulpot, &
       scoordtab, nnnb, q_single)
    use number,only:zero
    use enb_core,only:enb_uu_coul_vdw, enb_uu_vdw, enb_uv_coul_vdw, enb_uv_vdw, enb_vv
    use domdec_local_types,only:xyzq_dpsp_t
    use nblist_types,only:nblist_t, nb_pair_t, dpsp_t, uu, uv, vv
    !use new_timer,only:timer_start, timer_stop, T_uu, T_uv, T_vv
    implicit none
    ! Input / Output
    logical, intent(in) :: q_vdw, q_elec, q_ewexcl
    type(nblist_t), intent(in) :: nblist
    type(nb_pair_t), intent(in) :: nb_pair(:)
    integer, intent(in) :: vdwtype(:)
    type(dpsp_t), intent(in) :: vdwparam, vdwparam14
    type(xyzq_dpsp_t), intent(in) :: xyzq
    real(chm_real), intent(inout) :: force(:), sforce(:), coulpot, vdwpot, excoulpot
    type(dpsp_t), intent(in) :: scoordtab
    integer, intent(inout) :: nnnb
    logical, intent(in) :: q_single
    ! Variables
    real(chm_real) tmp
    integer i

    do i=1,nblist%n

       if (nblist%pair(i)%ni > 0) then
          if (nblist%pair(i)%itype == uu) then
             !call timer_start(T_uu)  
             if (nblist%pair(i)%lcoul .and. nblist%pair(i)%lvdw) then
                ! uuc
                if (q_vdw .or. q_elec) then
                   call enb_uu_coul_vdw(nblist%pair(i), nb_pair, vdwtype, vdwparam, &
                        xyzq, force, sforce, coulpot, vdwpot, &
                        scoordtab, nnnb, q_single)
                endif
             elseif (nblist%pair(i)%lcoul) then
                call wrndie(-5,'<enbxfast>','enbcalc_xfast: calling an invalid force loop')
             else
                ! uu
                if (q_vdw) then
                   call enb_uu_vdw(nblist%pair(i), nb_pair, vdwtype, vdwparam, &
                        xyzq, force, sforce, vdwpot, &
                        scoordtab, nnnb, q_single)
                endif
             endif
             !call timer_stop(T_uu)  
          elseif (nblist%pair(i)%itype == uv) then
             !call timer_start(T_uv)  
             if (nblist%pair(i)%lcoul .and. nblist%pair(i)%lvdw) then
                ! uvc
                if (q_vdw .or. q_elec) then
                   call enb_uv_coul_vdw(nblist%pair(i), nb_pair, vdwtype, vdwparam, &
                        xyzq, force, sforce, coulpot, vdwpot, &
                        scoordtab, nnnb, q_single)
                endif
             else
                ! uv
                if (q_vdw) then
                   call enb_uv_vdw(nblist%pair(i), nb_pair, vdwtype, vdwparam, &
                        xyzq, force, sforce, vdwpot, &
                        scoordtab, nnnb, q_single)
                endif
             endif
             !call timer_stop(T_uv)  
          elseif (nblist%pair(i)%itype == vv) then
             ! vv
             !call timer_start(T_vv)  
             if (q_vdw .or. q_elec) then
                call enb_vv(nblist%pair(i), nb_pair, vdwtype, vdwparam, &
                     xyzq, force, sforce, coulpot, vdwpot, &
                     scoordtab, nnnb, q_single)
             endif
             !call timer_stop(T_vv)  
          endif
       endif
    enddo

    return
  end subroutine enbcalc_xfast_cpu_loop

  ! *
  ! * Calculate statistics for non-bonded list
  ! *
  subroutine nblist_stat(ni, startj, nmin, nmax, nnb)
    implicit none
    ! Input / Output
    integer ni, startj(*), nmin, nmax, nnb
    ! Variables
    integer i, n

    nmin = 1000000000
    nmax = 0
    nnb = 0

    do i=1,ni
       n = startj(i+1) - startj(i)
       nnb = nnb + n
       nmin = min(nmin, n)
       nmax = max(nmax, n)
    enddo

  end subroutine nblist_stat

  ! *
  ! * Initializes lookup tables for non-bonded interactions
  ! *
  subroutine init_nb_pair(num_nb_pair, ppang)
    use nblist_types,only:nb_pair, nblist, TYPE_PAIR
    implicit none
    ! Input / Output
    integer, intent(out) :: num_nb_pair
    integer, intent(in) :: ppang

    if (nblist%type == TYPE_PAIR) then
       !   Coulomb   | VdW
       nb_pair(1)%lcoul  = .true.
       nb_pair(1)%lvdw   = .true.
       nb_pair(1)%type14 = 0
       !      -      | VdW
       nb_pair(2)%lcoul  = .false.
       nb_pair(2)%lvdw   = .true.
       nb_pair(2)%type14 = 0
       !      -      | 1-4 VdW
       nb_pair(3)%lcoul  = .false.
       nb_pair(3)%lvdw   = .true.
       nb_pair(3)%type14 = 1
       !   Coulomb   | -
       nb_pair(4)%lcoul  = .true.
       nb_pair(4)%lvdw   = .false.
       nb_pair(4)%type14 = 0
       num_nb_pair = 4
    else
       num_nb_pair = 0
    endif

    call build_lookup_tables(num_nb_pair, nb_pair, ppang)

    return
  end subroutine init_nb_pair

  ! *
  ! * Initializes lookup tables for 1-4 interactions and exclusions
  ! *
  subroutine init_inex14_pair(ppang)
    use nblist_types,only:inex14_pair
    implicit none
    ! Input
    integer, intent(in) :: ppang

    ! 1-4 interaction Coulomb | 1-4 interaction VdW
    inex14_pair(1)%lcoul  = .true.
    inex14_pair(1)%lvdw   = .true.
    inex14_pair(1)%type14 = 1
    ! 1-4 exclusion Coulomb | -
    inex14_pair(2)%lcoul  = .true.
    inex14_pair(2)%lvdw   = .true.
    inex14_pair(2)%type14 = 2

    call build_lookup_tables(2, inex14_pair, ppang)

    return
  end subroutine init_inex14_pair

  ! *
  ! * Builds lookup tables for bunch of pair interactions
  ! *
  subroutine build_lookup_tables(n, pair, ppang)
    use number,only:zero, one
    use energym,only:qeterm, vdw, elec, ewexcl
    use inbnd,only:cutnb, ctonnb, ctofnb, e14fac
    use ewald_1m,only:kappa
#if KEY_LJPME==1
    use energym,only: ljexc
#endif
    implicit none
    ! Input / Output
    integer, intent(in) :: n, ppang
    type(nb_pair_t), intent(inout) :: pair(:)
    ! Variables
    real(chm_real) qscale
    real(chm_real) vscale
    logical qvterm_val, qeterm_val
    integer vdwmodel, elecmodel
    integer i

    ! Get cutoff options
    call get_force_model(vdwmodel, elecmodel)

    do i=1,n
       qscale = one
       vscale = one
       qeterm_val = qeterm(elec)
       if (pair(i)%lcoul .and. pair(i)%type14 == 1) qscale = e14fac  ! 1-4 interaction
       if (pair(i)%lcoul .and. pair(i)%type14 == 2) then
          ! 1-4 exclusion
          qscale = zero
          ! If ewald exclusions are turned off, make sure the lookup table gets filled with
          ! zeros by setting qeterm_val = .false.
          if (.not.qeterm(ewexcl)) qeterm_val = .false.
       endif
       qvterm_val = qeterm(vdw)
#if KEY_LJPME==1
       if(pair(i)%type14 == 2) then
           !Exclusion, but we always subtract the multiplicative term
           qvterm_val = qeterm(ljexc)
       endif
#endif
       pair(i)%ctofnb = ctofnb
       pair(i)%ctonnb = ctonnb
       pair(i)%vdw_type = 1
       if ( pair(i)%type14 ==1 .and. vdwmodel == VIPS ) vscale = zero
       if ( pair(i)%type14 ==2) vscale = zero
       call build_pftable(pair(i)%pftable_dp, pair(i)%pftable_sp, ppang, &
            cutnb, pair(i)%ctonnb, pair(i)%ctofnb, pair(i)%lcoul, &
            pair(i)%lvdw, pair(i)%n, pair(i)%h, qscale, vscale, &
            vdwmodel, elecmodel, qeterm(vdw), qeterm_val, kappa)
    enddo

    return
  end subroutine build_lookup_tables

  ! *
  ! * Uninitializes nblist and lookup tables
  ! *
  subroutine uninit_nblist()
    use memory,only:chmdealloc
    use nblist_types,only:nblist, nb_pair, inex14_pair
#if KEY_DOMDEC_GPU==1
    use nblist_tilex_sort,only:uninit_nblist_tilex_sort
    use nblist_tilex,only:uninit_nblist_tilex
#endif
    implicit none
    ! Variables
    integer i
    integer ierror

#if KEY_DOMDEC_GPU==1
    call uninit_nblist_tilex_sort()
    call uninit_nblist_tilex(nblist%tilex1, nblist%tilex2)
#endif

    if (allocated(nblist%pair)) then
       do i=1,size(nblist%pair)
          if (allocated(nblist%pair(i)%indi)) then
             call chmdealloc('enbxfast.src','uninit_nblist','indi',size(nblist%pair(i)%indi),&
                  intg=nblist%pair(i)%indi)
          endif
          if (allocated(nblist%pair(i)%startj)) then
             call chmdealloc('enbxfast.src','uninit_nblist','startj',size(nblist%pair(i)%startj),&
                  intg=nblist%pair(i)%startj)
          endif
          if (allocated(nblist%pair(i)%indj)) then
             call chmdealloc('enbxfast.src','uninit_nblist','indj',size(nblist%pair(i)%indj),&
                  intg=nblist%pair(i)%indj)
          endif
          if (allocated(nblist%pair(i)%iscoord)) then
             call chmdealloc('enbxfast.src','uninit_nblist','iscoord',size(nblist%pair(i)%iscoord),&
                  intg=nblist%pair(i)%iscoord)
          endif
       enddo
       deallocate(nblist%pair, stat=ierror)
       if (ierror /= 0) call wrndie(-5,'<enbxfast>',&
            'Error deallocating memory for nblist%pair')
    endif

    do i=1,size(nb_pair)
       if (allocated(nb_pair(i)%pftable_dp)) then
          call chmdealloc('enbxfast.src','uninit_nblist','pftable_dp',&
               size(nb_pair(i)%pftable_dp),crl=nb_pair(i)%pftable_dp)
       endif
       if (allocated(nb_pair(i)%pftable_sp)) then
          call chmdealloc('enbxfast.src','uninit_nblist','pftable_sp',&
               size(nb_pair(i)%pftable_sp),cr4=nb_pair(i)%pftable_sp)
       endif
    enddo

    do i=1,size(inex14_pair)
       if (allocated(inex14_pair(i)%pftable_dp)) then
          call chmdealloc('enbxfast.src','uninit_nblist','pftable_dp',&
               size(inex14_pair(i)%pftable_dp),crl=inex14_pair(i)%pftable_dp)
       endif
       if (allocated(inex14_pair(i)%pftable_sp)) then
          call chmdealloc('enbxfast.src','uninit_nblist','pftable_sp',&
               size(inex14_pair(i)%pftable_sp),cr4=inex14_pair(i)%pftable_sp)
       endif
    enddo

    return
  end subroutine uninit_nblist

  ! *
  ! * Initialize nblist, calculates lookup tables
  ! *
  subroutine init_nblist(div)
    use energym,only:qeterm, vdw, elec, ewexcl
    use stream,only:outu, prnlev
    use number,only:zero
    use memory
    use psf,only:natom, ngrp
    use inbnd,only:cutnb, ctonnb, ctofnb, e14fac
    use ewald_1m,only:kappa
    use nblist_pair,only:iuu, iuuc, iuv, iuvc, ivv
    use nblist_types,only:TYPE_NONE, TYPE_PAIR, TYPE_TILEX, uu, uv, vv, nblist, nb_pair
    use domdec_common,only:natoml_tot, nthread, q_gpu, ppang
#if KEY_DOMDEC_GPU==1
    use enb_core_gpu_mod,only:set_nonbond_param_gpu
#endif 
#if KEY_BLOCK==1
    use nblist_pair,only:iuuc_block
    use lambdam,only:qmld
#endif 
    implicit none
    ! Input / Output
    integer div
    ! Variables
    integer ni_max, nj_max
    real(chm_real) qscale
    integer i
    integer ierror
    integer vdwmodel_new      ! New VdW force model setting
    integer elecmodel_new     ! New electrostatic force model setting
    ! Save values, initialized to zero so that the nblist structure is always made in the first call
    real(chm_real), save :: cutnb_save=zero, ctonnb_save=zero, ctofnb_save=zero, kappa_save=zero
    logical, save :: qeterm_vdw_save, qeterm_elec_save, qeterm_ewexcl_save, qmld_save=.false.
    integer, save :: type_save = TYPE_NONE, ppang_save
    logical qeterm_val
    integer num_nb_pair

    if (q_gpu) then
       nblist%type = TYPE_TILEX
    else
       nblist%type = TYPE_PAIR
    endif

    ! Get (possibly new) cutoff options
    call get_force_model(vdwmodel_new, elecmodel_new)

!!$    ! Check if we need to re-calculate lookup tables
!!$    if (cutnb_save == cutnb .and. ctonnb_save == ctonnb .and. ctofnb_save == ctofnb .and. &
!!$         kappa_save == kappa .and. &
!!$         (qeterm_vdw_save .eqv. qeterm(vdw)) .and. &
!!$         (qeterm_elec_save .eqv. qeterm(elec)) .and. &
!!$         (qeterm_ewexcl_save .eqv. qeterm(ewexcl)) .and. &
!!$#if KEY_BLOCK==1
!!$         (qmld_save .eqv. qmld) .and. &
!!$#endif 
!!$         vdwmodel == vdwmodel_new .and. &
!!$         elecmodel == elecmodel_new .and. &
!!$         type_save == nblist%type .and. &
!!$         ppang_save == ppang) then
!!$       ! All settings are equal => No need to re-calculate => return
!!$       return
!!$    endif

    ! Get cutoff options
    call get_force_model(vdwmodel, elecmodel)

    ! Set Non-bonded force parameters to GPU
#if KEY_DOMDEC_GPU==1
    if (q_gpu) then
       call set_nonbond_param_gpu(kappa, ctofnb, ctonnb, e14fac, &
            vdwmodel, elecmodel, qeterm(vdw), qeterm(elec))
    endif
#endif
    
    nblist%q_pack = .true.

    ! Build lookup tables for non-bonded interactions
    call init_nb_pair(num_nb_pair, ppang)

    ! Build lookup tables for 1-4 interactions & exclusions
    call init_inex14_pair(ppang)

    ! Calculate how many nonbond lists we need
    if (nblist%type == TYPE_TILEX) then
       nblist%n = 0

       if (allocated(nblist%tilex1)) then
          if (size(nblist%tilex1) < nthread) then
             do i=0,size(nblist%tilex1)-1
                if (associated(nblist%tilex1(i)%ientry)) then
                   deallocate(nblist%tilex1(i)%ientry)
                   nullify(nblist%tilex1(i)%ientry)
                endif
             enddo
             deallocate(nblist%tilex1)
          endif
       endif

       if (allocated(nblist%tilex2)) then
          if (size(nblist%tilex2) < nthread) then
             do i=0,size(nblist%tilex2)-1
                if (associated(nblist%tilex2(i)%ientry)) then
                   deallocate(nblist%tilex2(i)%ientry)
                   nullify(nblist%tilex2(i)%ientry)
                endif
             enddo
             deallocate(nblist%tilex2)
          endif
       endif

       if (.not.allocated(nblist%tilex1)) then
          allocate(nblist%tilex1(0:nthread-1))
          do i=0,nthread-1
             nullify(nblist%tilex1(i)%ientry)
          enddo
       endif

       if (.not.allocated(nblist%tilex2)) then
          allocate(nblist%tilex2(0:nthread-1))
          do i=0,nthread-1
             nullify(nblist%tilex2(i)%ientry)
          enddo
       endif

    else
       nblist%n = 5
#if KEY_BLOCK==1
       if (qmld) nblist%n = 6
#endif 
    endif

    if (nblist%type == TYPE_PAIR) then
       ! Allocate & reallocate memory for nblist
       if (allocated(nblist%pair)) then
          if (size(nblist%pair) < nblist%n) then
             deallocate(nblist%pair, stat=ierror)
             if (ierror /= 0) call wrndie(-5,'<enbxfast>',&
                  'Error deallocating memory for nblist%pair')
          endif
       endif
       if (.not.allocated(nblist%pair)) then
          allocate(nblist%pair(nblist%n),stat=ierror)
          if (ierror /= 0) call wrndie(-5,'<enbxfast>',&
               'Error allocating memory for nblist%pair')
       endif

       do i=1,nblist%n
          if (i == 1) then
             ! Solute - Solute with Coulomb and VdW
             nblist%pair(i)%lcoul = .true.
             nblist%pair(i)%lvdw = .true.
             nblist%pair(i)%lblock = .false.
             nblist%pair(i)%type14 = 0
             nblist%pair(i)%itype = uu
             nblist%pair(i)%name = 'U-U with Coulomb and VdW'
             iuuc = i
          elseif (i == 2) then
             ! Solute - Solute with VdW
             nblist%pair(i)%lcoul = .false.
             nblist%pair(i)%lvdw = .true.
             nblist%pair(i)%lblock = .false.
             nblist%pair(i)%type14 = 0
             nblist%pair(i)%itype = uu
             nblist%pair(i)%name = 'U-U with VdW'
             iuu = i
          elseif (i == 3) then
             ! Solute - Solvent with Coulomb and VdW
             nblist%pair(i)%lcoul = .true.
             nblist%pair(i)%lvdw = .true.
             nblist%pair(i)%lblock = .false.
             nblist%pair(i)%type14 = 0
             nblist%pair(i)%itype = uv
             nblist%pair(i)%name = 'U-V with Coulomb and VdW'
             iuvc = i
          elseif (i == 4) then
             ! Solute - Solvent with VdW
             nblist%pair(i)%lcoul = .false.
             nblist%pair(i)%lvdw = .true.
             nblist%pair(i)%lblock = .false.
             nblist%pair(i)%type14 = 0
             nblist%pair(i)%itype = uv
             nblist%pair(i)%name = 'U-V with VdW'
             iuv = i
          elseif (i == 5) then
             ! Solvent - Solvent with Coulomb and VdW
             nblist%pair(i)%lcoul = .true.
             nblist%pair(i)%lvdw = .true.
             nblist%pair(i)%lblock = .false.
             nblist%pair(i)%type14 = 0
             nblist%pair(i)%itype = vv
             nblist%pair(i)%name = 'V-V'
             ivv = i
#if KEY_BLOCK==1 /*block*/
          elseif (qmld .and. i == 6) then
             ! #####################################
             ! # Rest of the lists are for BLOCK
             ! #####################################
             ! Solute - Solute with Coulomb and VdW
             nblist%pair(i)%lcoul = .true.
             nblist%pair(i)%lvdw = .true.
             nblist%pair(i)%lblock = .true.
             nblist%pair(i)%type14 = 0
             nblist%pair(i)%itype = uu
             nblist%pair(i)%name = 'BLOCK U-U with Coulomb and VdW'
             iuuc_block = i
#endif /* (block)*/
          endif
       enddo

       do i=1,nblist%n
          nblist%pair(i)%nb_pair_ind = get_nb_pair_ind(num_nb_pair, nblist%pair(i)%lcoul, &
               nblist%pair(i)%lvdw, nblist%pair(i)%type14)
          
          if (nblist%pair(i)%nb_pair_ind == -1) then
             call wrndie(-5,'<enbxfast>',&
                  'Could not find appropriate Non-bonded pair interaction type')
          endif
          
          if (.not.allocated(nblist%pair(i)%indi)) then
             ! Estimate the number of entries needed in the neighbor lists
             ni_max = natom/(nblist%n*div)
             nj_max = natom/(nblist%n*div)
             nblist%pair(i)%ni_max = ni_max
             nblist%pair(i)%nj_max = nj_max
             call chmalloc('enb.src','init_nblist','indi',nblist%pair(i)%ni_max,&
                  intg=nblist%pair(i)%indi)
             call chmalloc('enb.src','init_nblist','startj',nblist%pair(i)%ni_max+1,&
                  intg=nblist%pair(i)%startj)
             call chmalloc('enb.src','init_nblist','indj',nblist%pair(i)%nj_max,&
                  intg=nblist%pair(i)%indj)
             call chmalloc('enb.src','init_nblist','iscoord',nblist%pair(i)%ni_max,&
                  intg=nblist%pair(i)%iscoord)
          endif
       enddo
    endif

    ! Save the values of cutnb, ctonnb, and ctofnb, for which the tables were calculated
    cutnb_save = cutnb
    ctonnb_save = ctonnb
    ctofnb_save = ctofnb
    kappa_save = kappa
    qeterm_vdw_save = qeterm(vdw)
    qeterm_elec_save = qeterm(elec)
    qeterm_ewexcl_save = qeterm(ewexcl)
#if KEY_BLOCK==1
    qmld_save = qmld
#endif 
    type_save = nblist%type
    ppang_save = ppang

    return
  end subroutine init_nblist

  ! *
  ! * Searches for the correct non-bonded pair interaction type from nb_pair(1:6)
  ! * Returns integer 1-6, or -1 on error
  ! *
  ! *
  integer function get_nb_pair_ind(n, lcoul, lvdw, type14)
    use nblist_types,only:nb_pair
    implicit none
    ! Input
    integer, intent(in) :: n
    logical, intent(in) :: lcoul, lvdw
    integer, intent(in) :: type14
    ! Variables
    integer i

    get_nb_pair_ind = -1

    do i=1,n
       if ((nb_pair(i)%lcoul .eqv. lcoul) .and. &
            (nb_pair(i)%lvdw .eqv. lvdw) .and. &
            (nb_pair(i)%type14 == type14)) then
          get_nb_pair_ind = i
          return
       endif
    enddo

    return
  end function get_nb_pair_ind

  ! *
  ! * Initialize vdwparam array
  ! * If q_redo = .true. redoes the arrays
  ! *
  subroutine init_vdwparam(q_redo_in)
    use nb_module,only:ccnba,ccnbb,lccnba
    use memory
    use domdec_common,only:q_single
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu
    use domdec_util_gpu_mod,only:copy_vdwparam_to_gpu, copy_vdwparam14_to_gpu
#endif 
    implicit none
    ! Input
    logical, intent(in) :: q_redo_in
    ! Variables
    integer nitcc2
#if KEY_DOMDEC_GPU==1
    real(chm_real4), allocatable, dimension(:) :: tmp  
#endif
    logical q_redo
    logical, save :: vdwparam_on_gpu = .false.

    q_redo = q_redo_in

    if (lccnba == 0) call wrndie(-5,'<enbxfast>','init_vdwparam: lccnba is zero')
    nitcc2 = lccnba / 2

    if (q_single) then
       if (allocated(vdwparam%sp)) then
          if (size(vdwparam%sp) /= lccnba) then
             call chmdealloc('enbxfast.src','init_vdwparam','vdwparam%sp',&
                  size(vdwparam%sp),cr4=vdwparam%sp)
             q_redo = .true.
          endif
       endif
       if (allocated(vdwparam14%sp)) then
          if (size(vdwparam14%sp) /= lccnba) then
             call chmdealloc('enbxfast.src','init_vdwparam','vdwparam14%sp',&
                  size(vdwparam14%sp),cr4=vdwparam14%sp)
             q_redo = .true.
          endif
       endif
    else
       if (allocated(vdwparam%dp)) then
          if (size(vdwparam%dp) /= lccnba) then
             call chmdealloc('enbxfast.src','init_vdwparam','vdwparam%dp',&
                  size(vdwparam%dp),crl=vdwparam%dp)
             q_redo = .true.
          endif
       endif
       if (allocated(vdwparam14%dp)) then
          if (size(vdwparam14%dp) /= lccnba) then
             call chmdealloc('enbxfast.src','init_vdwparam','vdwparam14%dp',&
                  size(vdwparam14%dp),crl=vdwparam14%dp)
             q_redo = .true.
          endif
       endif
    endif

    if (q_single) then
       if (.not.allocated(vdwparam%sp)) then
          call chmalloc('enbxfast.src','init_vdwparam','vdwparam%sp',lccnba,cr4=vdwparam%sp)
       endif

       if (.not.allocated(vdwparam14%sp)) then
          call chmalloc('enbxfast.src','init_vdwparam','vdwparam14%sp',lccnba,cr4=vdwparam14%sp)
       endif

       if (q_redo) then
          vdwparam%sp(1:lccnba-1:2) = ccnbb(1:nitcc2)
          vdwparam%sp(2:lccnba:2) = ccnba(1:nitcc2)
          vdwparam_on_gpu = .false.
          vdwparam14%sp(1:lccnba-1:2) = ccnbb(nitcc2+1:lccnba)
          vdwparam14%sp(2:lccnba:2) = ccnba(nitcc2+1:lccnba)
       endif
    else
       if (.not.allocated(vdwparam%dp)) then
          call chmalloc('enbxfast.src','init_vdwparam','vdwparam%dp',lccnba,crl=vdwparam%dp)
       endif

       if (.not.allocated(vdwparam14%dp)) then
          call chmalloc('enbxfast.src','init_vdwparam','vdwparam14%dp',lccnba,crl=vdwparam14%dp)
       endif

       if (q_redo) then
          vdwparam%dp(1:lccnba-1:2) = ccnbb(1:nitcc2)
          vdwparam%dp(2:lccnba:2) = ccnba(1:nitcc2)
          vdwparam_on_gpu = .false.
          vdwparam14%dp(1:lccnba-1:2) = ccnbb(nitcc2+1:lccnba)
          vdwparam14%dp(2:lccnba:2) = ccnba(nitcc2+1:lccnba)
       endif
    endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu .and. .not.vdwparam_on_gpu) then
       call chmalloc('enbxfast.src','init_vdwparam','tmp',lccnba,cr4=tmp)
       tmp(1:lccnba-1:2) = ccnbb(1:nitcc2)
       tmp(2:lccnba:2) = ccnba(1:nitcc2)
       call copy_vdwparam_to_gpu(lccnba, tmp)
       tmp(1:lccnba-1:2) = ccnbb(nitcc2+1:lccnba)
       tmp(2:lccnba:2) = ccnba(nitcc2+1:lccnba)
       call copy_vdwparam14_to_gpu(lccnba, tmp)
       vdwparam_on_gpu = .true.
       call chmdealloc('enbxfast.src','init_vdwparam','tmp',lccnba,cr4=tmp)
    endif
#endif

    return
  end subroutine init_vdwparam

  ! *
  ! * Deallocates memory allocated in init_vdwparam
  ! *
  subroutine uninit_vdwparam()
    use memory,only:chmdealloc
    implicit none

    ! vdwparam%sp
    if (allocated(vdwparam%sp)) then
       call chmdealloc('enbxfast.src','uninit_vdwparam','vdwparam%sp',&
            size(vdwparam%sp),cr4=vdwparam%sp)
    endif

    ! vdwparam14%sp
    if (allocated(vdwparam14%sp)) then
       call chmdealloc('enbxfast.src','uninit_vdwparam','vdwparam14%sp',&
            size(vdwparam14%sp),cr4=vdwparam14%sp)
    endif

    if (allocated(vdwparam%dp)) then
       call chmdealloc('enbxfast.src','uninit_vdwparam','vdwparam%dp',&
            size(vdwparam%dp),crl=vdwparam%dp)
    endif

    if (allocated(vdwparam14%dp)) then
       call chmdealloc('enbxfast.src','uninit_vdwparam','vdwparam14%dp',&
            size(vdwparam14%dp),crl=vdwparam14%dp)
    endif

    return
  end subroutine uninit_vdwparam

  ! *
  ! * Determines the VdW and electrostatic force model types
  ! *
  subroutine get_force_model(vdwmodel, elecmodel)
    use ewald_1m,only:lewald
    use inbnd,only:lcons, lshft, lfswt, lvfswt, lvshft, legrom, lvgrom, leips, lvips
    implicit none
    ! Input / Output
    integer, intent(out) :: vdwmodel, elecmodel

!    if (legrom) call wrndie(-5,'<ENBXFAST>',&
!         'GROMACS TRUNCATION METHOD NOT IMPLEMENTED YET')

    elecmodel = NONE
    vdwmodel = NONE

    ! Electrostatic force model
    if (.not.lcons .and.      lshft .and.      lfswt .and. .not. legrom) then
       elecmodel = RSHIFT
    endif

    if (.not.lcons .and. .not.lshft .and.      lfswt .and. .not. legrom) then
       elecmodel = RFSWIT
    endif

    if (.not.lcons .and.      lshft .and. .not.lfswt .and. .not. legrom) then
       elecmodel = RSHFT
    endif
    
    if (.not.lcons .and. .not.lshft .and. .not.lfswt .and. .not. legrom) then
       elecmodel = RSWIT
    endif

    if (lcons .and.      lshft .and.      lfswt .and. .not. legrom) then
       elecmodel = CSHIFT
    endif

    if (lcons .and. .not.lshft .and.      lfswt .and. .not. legrom) then
       elecmodel = CFSWIT
    endif

    if (lcons .and.      lshft .and. .not.lfswt .and. .not. legrom) then
       elecmodel = CSHFT
    endif

    if (lcons .and. .not.lshft .and. .not.lfswt .and. .not. legrom) then
       elecmodel = CSWIT
    endif

    if (legrom) then
       elecmodel = GSHFT
    endif

    if (leips) then
        elecmodel = EIPS
    endif

    ! NOTE: Ewald overrides all other electrostatic model settings
    if (lewald) then
       elecmodel = EWALD
    endif

    ! VdW force model
    if (lvfswt .and. .not. lvgrom) then
       vdwmodel = VFSW
    endif

    if (.not.lvfswt .and. lvshft .and. .not. lvgrom) then
       vdwmodel = VSH
    endif

    if (.not.lvfswt .and. .not.lvshft .and. .not. lvgrom) then
       vdwmodel = VSW
    endif

    if (lvgrom) then
       vdwmodel = VGSH
    endif

    if (lvips) then
        vdwmodel = VIPS
    endif

    return
  end subroutine get_force_model

  ! *
  ! * Determines the Coulomb and VdW cutoff options
  ! *
  subroutine print_force_model(vdwmodel, elecmodel)
    use stream
    implicit none
    ! Input
    integer, intent(in) :: vdwmodel, elecmodel
    ! Variables
    character(20) cstr, vstr

    if (prnlev > 2) then
       if (elecmodel == RSHIFT) then
          write (cstr,'(a)') 'rshift'
       elseif (elecmodel == RFSWIT) then
          write (cstr,'(a)') 'rfswit'
       elseif (elecmodel == RSHFT) then
          write (cstr,'(a)') 'rshft'
       elseif (elecmodel == RSWIT) then
          write (cstr,'(a)') 'rswit'
       elseif (elecmodel == CSHIFT) then
          write (cstr,'(a)') 'cshift'
       elseif (elecmodel == CFSWIT) then
          write (cstr,'(a)') 'cfswit'
       elseif (elecmodel == CSHFT) then
          write (cstr,'(a)') 'cshft'
       elseif (elecmodel == CSWIT) then
          write (cstr,'(a)') 'cswit'
       elseif (elecmodel == GSHFT) then
          write (cstr,'(a)') 'gshft'
       elseif (elecmodel == EIPS) then
          write (cstr,'(a)') 'eips'
       elseif (elecmodel == EWALD) then
          write (cstr,'(a)') 'ewald'
       elseif (elecmodel == NONE) then
          write (cstr,'(a)') 'NONE'
       else
          call wrndie(-5,'<enbxfast>','print_force_model, invalid electrostatic model')
       endif

       if (vdwmodel == VFSW) then
          write (vstr,'(a)') 'lvfsw'
       elseif (vdwmodel == VSH) then
          write (vstr,'(a)') 'lvsh'
       elseif (vdwmodel == VSW) then
          write (vstr,'(a)') 'lvsw'
       elseif (vdwmodel == VGSH) then
          write (vstr,'(a)') 'vgsh'
       elseif (vdwmodel == VIPS) then
          write (vstr,'(a)') 'vips'
       elseif (vdwmodel == NONE) then
          write (vstr,'(a)') 'NONE'
       else
          call wrndie(-5,'<enbxfast>','print_force_model, invalid VdW model')
       endif

       write (outu,'(4a)') 'Coulomb: ',trim(cstr), ' VdW: ',trim(vstr)
    endif

    return
  end subroutine print_force_model

  ! *
  ! * Tests build_pftable -routine for internal consistency
  ! * e.g. makes sure force is the derivative of the energy
  ! *
  subroutine test_build_pftable()
    use stream
    use memory
    use number
    use domdec_common,only:ppang
    use nbips,only:rips
    implicit none
    ! Parameters
    integer, parameter, dimension(5) :: vdw_models = (/ VSH, VSW, VFSW, VGSH, VIPS/)
    integer, parameter, dimension(11) :: elec_models = (/ EWALD, CSHIFT, CFSWIT, CSHFT, CSWIT, &
         RSWIT, RSHFT, RSHIFT, RFSWIT, GSHFT, EIPS/)
    ! Variables
    integer vdwmodel, elecmodel
    real(chm_real), allocatable, dimension(:) :: pftable_dp
    real(chm_real4), allocatable, dimension(:) :: pftable_sp
    integer n
    real(chm_real) cutnb, ctonnb, ctofnb, h, rips_save
    real(chm_real) re_max
    real(chm_real) hfunc, refunc, refunc_max, d2_apx
    real(chm_real) r1, f1, d1, r2, f2, d2, d1_apx
    real(chm_real) kappa, qscale
    real(chm_real) Acoef, Bcoef, Ccoef, Constr, Denom
    real(chm_real) Aconst, Bconst, Cconst, Dconst, Eaddr, dvc
    real(chm_real) GAconst, GBcoef
    real(chm_real) k6, k12, dv6, dv12
    real(chm_real) GA6, GB6, GC6, GA12, GB12, GC12
    logical lcoul, lvdw
    logical coul_opts(8), vdw_opts(3)
    integer ict, icut, i, j, k
    logical ok

    re_max = zero
    refunc_max = zero

    hfunc = 1.0d-7

    cutnb = 12.0d0
    kappa = 0.32d0
    qscale = 1.4d0
    
    ! save rips for restore
    rips_save = rips
    
    do ict=1,2
       
       if (ict == 1) then
          ctonnb = 10.0d0
          ctofnb = 11.0d0
       else
          ctonnb = 11.0d0
          ctofnb = 10.0d0
       endif
       
       ! Test sw() and dsw()
       do icut=1,3
          if (icut == 1) then
             r1 = min(ctonnb,ctofnb)/2.0d0
          elseif (icut == 2) then
             r1 = min(ctonnb,ctofnb) + abs(ctofnb-ctonnb)/2.0d0
          else
             r1 = max(ctonnb,ctofnb) + (cutnb-max(ctonnb,ctofnb))/2.0d0
          endif
          r2 = r1 + hfunc
          f1 = sw(r1, ctonnb, ctofnb)
          f2 = sw(r2, ctonnb, ctofnb)
          d1 = dsw(r1, ctonnb, ctofnb)
          d2 = dsw(r2, ctonnb, ctofnb)
          d1_apx = (f2-f1)/hfunc
          if (d1 /= zero) then
             refunc = abs(d1_apx - d1)/abs(d1)
          else
             refunc = abs(d1_apx - d1)
          endif
          refunc_max = max(refunc, refunc_max)
          if (refunc > 1.0d-6) then
             write (outu,'(a,3f12.6)') 'r1, f1, d1=',r1, f1, d1
             write (outu,'(a,3f12.6)') 'r2, f2, d2=',r2, f2, d2
             write (outu,'(a,f12.6,e12.4)') 'd1_apx, refunc=',d1_apx, refunc
             call wrndie(-5,'<enbxfast>',&
                  'test_build_pftable: error in functional form of sw() / dsw()')
          endif
       enddo

       ! Loop through all possible electrostatic and VdW cut options
      
       ! initialize IPS for elec vdw calculation
       rips=-one
       call ipsset(cutnb,ctofnb,.true.,.true.,.true.,.false.)
       
       do i=1,size(elec_models)
          elecmodel = elec_models(i)
          ! Test functional form of coulomb()
          call calc_coulomb_constants(.true., ctonnb, ctofnb, &
               Aconst, Bconst, Cconst, Dconst, dvc, &
               Acoef, Bcoef, Ccoef, Denom, Constr, Eaddr, &
               GAconst, GBcoef)
          do icut=1,3
             if (icut == 1) then
                r1 = min(ctonnb,ctofnb)/2.0d0
             elseif (icut == 2) then
                r1 = min(ctonnb,ctofnb) + abs(ctofnb-ctonnb)/2.0d0
             else
                r1 = max(ctonnb,ctofnb) + (cutnb-max(ctonnb,ctofnb))/2.0d0
             endif
             r2 = r1 + hfunc

             r1 = ctonnb/2.0d0
             r2 = r1 + hfunc
             call electrostatic(r1, f1, d1, ctofnb, ctonnb, elecmodel, kappa, qscale, &
                  Acoef, Bcoef, Ccoef, Constr, Denom, Aconst, Bconst, Cconst, Dconst, Eaddr, dvc, &
                  GAconst, GBcoef)
             call electrostatic(r2, f2, d2, ctofnb, ctonnb, elecmodel, kappa, qscale, &
                  Acoef, Bcoef, Ccoef, Constr, Denom, Aconst, Bconst, Cconst, Dconst, Eaddr, dvc, &
                  GAconst, GBcoef)
             d1_apx = (f2-f1)/hfunc
             refunc = abs(d1_apx - d1)/abs(d1)
             refunc_max = max(refunc, refunc_max)

             if (refunc > 1.0d-6) then
                call print_force_model(vdw_models(1), elecmodel)
                write (outu,'(a,3e20.12)') 'r1, f1, d1=',r1, f1, d1
                write (outu,'(a,3e20.12)') 'r2, f2, d2=',r2, f2, d2
                write (outu,'(a,f20.12,e20.12)') 'd1_apx, refunc=',d1_apx, refunc
                call wrndie(-5,'<enbxfast>',&
                     'test_build_pftable: error in functional form of coulomb()')
             endif
          enddo

          do j=1,size(vdw_models)
             vdwmodel = vdw_models(j)
             ! Test functional form of vdw_attraction() and vdw_repulsive()
             if (i == 1) then
                call calc_vdw_constants(.true., ctonnb, ctofnb, k6, k12, dv6, dv12, &
                     GA6, GB6, GC6, GA12, GB12, GC12)
                do icut=1,3
                   if (icut == 1) then
                      r1 = min(ctonnb,ctofnb)/2.0d0
                   elseif (icut == 2) then
                      r1 = min(ctonnb,ctofnb) + abs(ctofnb-ctonnb)/2.0d0
                   else
                      r1 = max(ctonnb,ctofnb) + (cutnb-max(ctonnb,ctofnb))/2.0d0
                   endif
                   r2 = r1 + hfunc

                   call vdw_attraction(r1, f1, d1, ctonnb, ctofnb, one, vdwmodel, k6, dv6, &
                        GA6, GB6, GC6)
                   call vdw_attraction(r2, f2, d2, ctonnb, ctofnb, one, vdwmodel, k6, dv6, &
                        GA6, GB6, GC6)
                   d1_apx = (f2-f1)/hfunc
                   if (d1 .eq. zero) then
                      refunc = 0
                   else
                      refunc = abs(d1_apx - d1) / abs(d1)
                   end if
                   refunc_max = max(refunc, refunc_max)
                   if (refunc > 1.0d-6) then
                      call print_force_model(vdwmodel, elecmodel)
                      write (outu,'(a,3f20.12)') 'r1, f1, d1=',r1, f1, d1
                      write (outu,'(a,3f20.12)') 'r2, f2, d2=',r2, f2, d2
                      write (outu,'(a,f20.12,e20.12)') 'd1_apx, refunc=',d1_apx, refunc
                      call wrndie(-5,'<enbxfast>',&
                           'test_build_pftable: error in functional form of vdw_attraction()')
                   endif

                   call vdw_repulsion(r1, f1, d1, ctonnb, ctofnb, one, vdwmodel, k12, dv12, &
                        GA12, GB12, GC12)
                   call vdw_repulsion(r2, f2, d2, ctonnb, ctofnb, one, vdwmodel, k12, dv12, &
                        GA12, GB12, GC12)
                   d1_apx = (f2-f1)/hfunc
                   if (d1 .eq. zero) then
                      refunc = 0
                   else
                      refunc = abs(d1_apx - d1) / abs(d1)
                   end if
                   refunc_max = max(refunc, refunc_max)
                   if (refunc > 1.0d-6) then
                      call print_force_model(vdwmodel, elecmodel)
                      write (outu,'(a,3f12.6)') 'r1, f1, d1=',r1, f1, d1
                      write (outu,'(a,3f12.6)') 'r2, f2, d2=',r2, f2, d2
                      write (outu,'(a,f12.6,e12.4)') 'd1_apx, refunc=',d1_apx, refunc
                      call wrndie(-5,'<enbxfast>',&
                           'test_build_pftable: error in functional form of vdw_repulsion()')
                   endif
                enddo
             endif

             do k=1,3
                if (k == 1) then
                   lcoul = .true.
                   lvdw = .false.
                elseif (k == 2) then
                   lcoul = .false.
                   lvdw = .true.
                else
                   lcoul = .true.
                   lvdw = .true.
                endif

                call build_pftable(pftable_dp, pftable_sp, ppang, cutnb, ctonnb, ctofnb, &
                     lcoul, lvdw, n, h, one, one, &
                     vdwmodel, elecmodel, .true., .true., kappa)
                call test_pftable_dp(pftable_dp, re_max)
                if (.not.ok) then
                   write (outu,'(a,3i2)') 'i,j,k=',i,j,k
                   call print_force_model(vdwmodel, elecmodel)
                   call wrndie(-5,'<enbxfast>','Energy/Force consistency check failed')
                endif
             enddo
          enddo
       enddo
    enddo

    if (allocated(pftable_dp)) then
       call chmdealloc('enbxfast.src','test_build_pftable','pftable_dp',&
            size(pftable_dp),crl=pftable_dp)
    endif

    if (allocated(pftable_sp)) then
       call chmdealloc('enbxfast.src','test_build_pftable','pftable_sp',&
            size(pftable_sp),cr4=pftable_sp)
    endif

    if (prnlev >= 2) then
       write (outu,'(a,2e12.4)') 'test_build_pftable passed OK with re_max, refunc_max=',&
            re_max,refunc_max
    endif

    ! restore IPS for elec vdw calculation
    if(rips_save > zero)then
      rips=-one
      call ipsset(cutnb,rips_save,.true.,.true.,.true.,.false.)
    endif
    return

  contains
    subroutine test_pftable_dp(pftable, re_max)
      implicit none
      ! Input
      real(chm_real), intent(in), allocatable, dimension(:) :: pftable
      real(chm_real), intent(out) :: re_max
      ! Variables
      integer ii, multt, start_tt, tt
      real(chm_real) r, dr
      real(chm_real) v1, v2, v3, d1, d2, d3
      real(chm_real) d2_apx, re
      
      if (lcoul .and. lvdw) then
         multt = 12
      elseif (lvdw) then
         multt = 8
      elseif (lcoul) then
         multt = 12
      else
         call wrndie(-5,'<enbxfast>','test_pftable_dp: either coulomb or vdw flag must be set true')
      endif

#if KEY_LJPME==1
      multt = multt + 4
#endif

      ok = .true.
      re_max = zero

      dr = h/1000000.0d0

      do ii=20,n-1
         r = ii*h

         ! Only test within ctofnb
         if (r + half*dr < ctofnb) then
            do start_tt=0,multt-4,4
               call eval_pftable_dp(pftable, start_tt, r - half*dr, multt, v1, d1)
               call eval_pftable_dp(pftable, start_tt, r          , multt, v2, d2)
               call eval_pftable_dp(pftable, start_tt, r + half*dr, multt, v3, d3)

               d2_apx = (v3-v1)/dr
               if (d2 .eq. zero) then
                  re = 0
               else
                  re = abs((d2_apx - d2)/d2)
               end if
               re_max = max(re, re_max)

               if (re > 1.0e-2) then
                  write (outu,'(a,i3)') 'multt=',multt
                  write (outu,'(a,i6,2f20.8)') 'ii,r,re=',ii,r,re
                  write (outu,'(a,f20.5)') 'd2_apx=',d2_apx
                  write (outu,'(a,2f20.5)') 'v1,d1=',v1,d1
                  write (outu,'(a,2f20.5)') 'v2,d2=',v2,d2
                  write (outu,'(a,2f20.5)') 'v3,d3=',v3,d3
                  ok = .false.
                  return
               endif
            enddo
         endif
      enddo

      return
    end subroutine test_pftable_dp

    ! *
    ! * Returns value (v) and derivative (d) from the lookup table
    ! * start_tt = 0: Coulomb
    ! * start_tt = 4: VdW attraction
    ! * start_tt = 8: VdW repulsion
    ! *
    subroutine eval_pftable_dp(pftable, start_tt, r, multt, v, d)
      implicit none
      ! Input / Output
      real(chm_real), intent(in), allocatable, dimension(:) :: pftable
      real(chm_real), intent(in) :: r
      integer, intent(in) :: multt, start_tt
      real(chm_real), intent(out) :: v, d
      ! Variables
      real(chm_real) a0, a1, a2, a3, ept
      integer tt

      tt = int(r/h)
      ept = r/h - real(tt, kind=8)
      tt = multt*(tt - 1) + 1 + start_tt
      
      a0 = pftable(tt)             ! a0
      a1 = pftable(tt+1)           ! a1        
      a2 = pftable(tt+2)*ept       ! a2*ep         
      a3 = pftable(tt+3)*ept*ept   ! a3*ep^2
      
      v = (a0 + (a1 + a2 + a3)*ept)
      d = (a1 + two*a2 + three*a3)/h

      return
    end subroutine eval_pftable_dp

  end subroutine test_build_pftable

  ! *
  ! * Builds potential-force lookup table
  ! *
  subroutine build_pftable(pftable_dp, pftable_sp, ppangst, cutnb, ctonnb, ctofnb, &
       lcoul, lvdw, n, h, qscale, vscale, &
       vdwmodel, elecmodel, qeterm_vdw, qeterm_elec, kappa)
    use memory
    use number,only:zero, one, two, three
#if KEY_LJPME==1
    use pmeutil, only: qljpme
#endif
    implicit none
    ! Input / Output
    real(chm_real), intent(inout), allocatable, dimension(:) :: pftable_dp
    real(chm_real4), intent(inout), allocatable, dimension(:) :: pftable_sp
    integer, intent(in) :: ppangst       ! points per Angstrom
    real(chm_real), intent(in) :: cutnb, ctonnb, ctofnb  ! Cut-off distances
    logical, intent(in) :: lcoul, lvdw   ! true/false flags for Coulomb and VdW interactions
    integer, intent(out) :: n            ! Number of bins in the table
    real(chm_real), intent(out) :: h     ! Bin width
    real(chm_real), intent(in) :: qscale ! Coulomb interaction scaling, accounts for 1-4 pairs
    real(chm_real), intent(in) :: vscale ! vdw scaling, accounts for 1-4 pairs
    integer, intent(in) :: vdwmodel, elecmodel  ! VdW and electrostatic force model settings
    logical, intent(in) :: qeterm_vdw, qeterm_elec
    real(chm_real), intent(in) :: kappa
    ! Variables
    integer i, t, icut, pftable_inc_coul, pftable_inc_vdw, pftable_inc, pftable_inc_ljpme
    real(chm_real) r1, r2, f1, f2, d1, d2
    real(chm_real) Aconst, Bconst, Cconst, Dconst, dvc
    real(chm_real) Acoef, Bcoef, Ccoef, Denom, Constr, Eaddr
    real(chm_real) GAconst, GBcoef
    real(chm_real) k6, k12, dv6, dv12
    real(chm_real) GA6, GB6, GC6, GA12, GB12, GC12

    n = int((cutnb + 6.0d0)*real(ppangst)) + 1
    h = one/real(ppangst)

    ! ctofnb = non-bonded cut-off distance
    ! ctonnb = non-bonded switch-on distance

    call calc_coulomb_constants(lcoul, ctonnb, ctofnb, &
         Aconst, Bconst, Cconst, Dconst, dvc, &
         Acoef, Bcoef, Ccoef, Denom, Constr, Eaddr, &
         GAconst, GBcoef)

    call calc_vdw_constants(lvdw, ctonnb, ctofnb, k6, k12, dv6, dv12, &
                     GA6, GB6, GC6, GA12, GB12, GC12)

    ! Set table entry sizes
    if (lcoul .and. lvdw) then
       pftable_inc_coul = 4
       pftable_inc_vdw = 8
    elseif (lvdw) then
       pftable_inc_coul = 0
       pftable_inc_vdw = 8
    elseif (lcoul) then
       ! NOTE: there are no separate force loops for Coulomb interaction only, therefore
       !       we include the VdW here too put make sure the parameters are all zero
       pftable_inc_coul = 4
       pftable_inc_vdw = 8
    else
       call wrndie(-5,'<ENB>','EITHER COULOMB OR VDW FLAG MUST BE SET TRUE')
    endif
#if KEY_LJPME==1
    pftable_inc_ljpme = 4
#else
    pftable_inc_ljpme = 0
#endif
    ! Total table entry size
    pftable_inc = pftable_inc_coul + pftable_inc_vdw + pftable_inc_ljpme

    if (allocated(pftable_dp)) call chmdealloc('enbxfast.src','build_pftable','pftable_dp',&
         pftable_inc*n,crl=pftable_dp)
    call chmalloc('enbxfast.src','build_pftable','pftable_dp',pftable_inc*n,crl=pftable_dp)
    if (allocated(pftable_sp)) call chmdealloc('enbxfast.src','build_pftable','pftable_sp',&
         pftable_inc*n,cr4=pftable_sp)
    call chmalloc('enbxfast.src','build_pftable','pftable_sp',pftable_inc*n,cr4=pftable_sp)

    t = 1

    ! The interaction is cut at and beyond icut
    icut = ctofnb/h

    ! First fill tables with zeros. This makes sure no unintended interactions are included
    pftable_dp(1:pftable_inc*n) = zero
    pftable_sp(1:pftable_inc*n) = real(0.0,kind=chm_real4)

    do i=1,n
       r1 = i*h
       r2 = (i+1)*h
       if (lcoul) then
          ! Coulomb interaction
          if (qeterm_elec) then
             call electrostatic(r1, f1, d1, ctofnb, ctonnb, elecmodel, kappa, qscale, &
                  Acoef, Bcoef, Ccoef, Constr, Denom, Aconst, Bconst, Cconst, Dconst, Eaddr, dvc, &
                  GAconst, GBcoef)
             call electrostatic(r2, f2, d2, ctofnb, ctonnb, elecmodel, kappa, qscale, &
                  Acoef, Bcoef, Ccoef, Constr, Denom, Aconst, Bconst, Cconst, Dconst, Eaddr, dvc, &
                  GAconst, GBcoef)
             if (i >= icut) then
                f1 = zero
                d1 = zero
                f2 = zero
                d2 = zero
             endif
             pftable_dp(t) = f1
             pftable_dp(t+1) = h*d1
             pftable_dp(t+2) = three*(f2 - f1) - two*h*d1 - h*d2
             pftable_dp(t+3) = two*(f1 - f2) + h*d1 + h*d2
          endif
          pftable_sp(t:t+3) = real(pftable_dp(t:t+3),kind=chm_real4)
       endif
       t = t + pftable_inc_coul
       if (lvdw) then
          ! VdW attraction
          if (qeterm_vdw) then
             ! MGL: why not stick these calls in an else block below?
             ! then you can yank out one of the if tests in the 
             ! vdw_attraction/etc. routines, and get rid of a bunch of
             ! calls to the vdw_attraction/etc. routines wholesale.
             call vdw_attraction(r1, f1, d1, ctonnb, ctofnb, vscale, vdwmodel, k6, dv6, &
                  GA6, GB6, GC6)
             call vdw_attraction(r2, f2, d2, ctonnb, ctofnb, vscale, vdwmodel, k6, dv6, &
                  GA6, GB6, GC6)
             if (i >= icut) then
                f1 = zero
                d1 = zero
                f2 = zero
                d2 = zero
             endif
             pftable_dp(t) = f1
             pftable_dp(t+1) = h*d1
             pftable_dp(t+2) = three*(f2 - f1) - two*h*d1 - h*d2
             pftable_dp(t+3) = two*(f1 - f2) + h*d1 + h*d2
             ! VdW repulsion
             call vdw_repulsion(r1, f1, d1, ctonnb, ctofnb, vscale, vdwmodel, k12, dv12, &
                        GA12, GB12, GC12)
             call vdw_repulsion(r2, f2, d2, ctonnb, ctofnb, vscale, vdwmodel, k12, dv12, &
                        GA12, GB12, GC12)
             if (i >= icut) then
                f1 = zero
                d1 = zero
                f2 = zero
                d2 = zero
             endif
             pftable_dp(t+4) = f1
             pftable_dp(t+5) = h*d1
             pftable_dp(t+6) = three*(f2 - f1) - two*h*d1 - h*d2
             pftable_dp(t+7) = two*(f1 - f2) + h*d1 + h*d2
          endif
          pftable_sp(t:t+7) = real(pftable_dp(t:t+7),kind=chm_real4)
       endif
       t = t + pftable_inc_vdw
#if KEY_LJPME==1
       ! We need the multiplicative terms here.  No check for lvdw here because there's
       ! always a multiplicative correction term for excluded and present pairs.
       if (qeterm_vdw .and. qljpme) then
          call vdw_ljpme_grid(r1, f1, d1, ctonnb, ctofnb, vdwmodel, vscale)
          call vdw_ljpme_grid(r2, f2, d2, ctonnb, ctofnb, vdwmodel, vscale)
          if (i >= icut) then
             f1 = zero
             d1 = zero
             f2 = zero
             d2 = zero
          endif
          pftable_dp(t) = f1
          pftable_dp(t+1) = h*d1
          pftable_dp(t+2) = three*(f2 - f1) - two*h*d1 - h*d2
          pftable_dp(t+3) = two*(f1 - f2) + h*d1 + h*d2
          pftable_sp(t:t+3) = real(pftable_dp(t:t+3),kind=chm_real4)
       endif
       t = t + pftable_inc_ljpme
#endif
    enddo

    return
  end subroutine build_pftable

  ! *
  ! * Determines Coulomb constants
  ! *
  subroutine calc_coulomb_constants(lcoul, ctonnb, ctofnb, &
       Aconst, Bconst, Cconst, Dconst, dvc, &
       Acoef, Bcoef, Ccoef, Denom, Constr, Eaddr, &
       GAconst, GBcoef )
    use number,only:zero, one, two, three, five, six, eight, twelve
    implicit none
    ! Input / Output
    logical, intent(in) :: lcoul
    real(chm_real), intent(in) :: ctonnb, ctofnb
    real(chm_real), intent(out) :: Aconst, Bconst, Cconst, Dconst, dvc
    real(chm_real), intent(out) :: Acoef, Bcoef, Ccoef, Denom, Constr, Eaddr
    real(chm_real), intent(out) :: GAconst, GBcoef
    ! Variables
    real(chm_real) g, ctonnb2, ctofnb2, ctonnb4, ctofnb4, ctofnb5

    if (lcoul) then
       ! Constants for Coulomb
       ctonnb2 = ctonnb**2
       ctofnb2 = ctofnb**2
       ctonnb4 = ctonnb**4
       ctofnb4 = ctofnb**4
       ctofnb5 = ctofnb**5
       g = (ctofnb2 - ctonnb2)**3
       Aconst = ctofnb4*(ctofnb2 - three*ctonnb2)/g
       Bconst = six*ctofnb2*ctonnb2/g
       Cconst = -(ctonnb2 + ctofnb2)/g
       Dconst = two/(five*g)
       dvc = eight*(ctonnb2*ctofnb2*(ctofnb-ctonnb) - (ctofnb**5 - ctonnb**5)/five)/g
       GAconst = 5.0/(3.0*ctofnb)
       GBcoef = 5.0/(3.0*ctofnb**4)
       if (ctonnb < ctofnb) then
          Denom = one/(ctofnb2 - ctonnb2)**3
          Acoef = ctofnb4*(ctofnb2 - three*ctonnb2)*Denom
          Bcoef = six*ctofnb2*ctonnb2*Denom
          Ccoef = -three*(ctonnb2 + ctofnb2)*Denom
          Constr = two*Bcoef*log(ctofnb) - Acoef/ctofnb2 + Ccoef*ctofnb2 + Denom*ctofnb4
          Eaddr = (twelve*ctonnb2*ctofnb2*log(ctofnb/ctonnb) - &
               three*(ctofnb4 - ctonnb4))*Denom
       else
          Eaddr = -one/ctofnb2
       endif
    else
       Aconst = zero
       Bconst = zero
       Cconst = zero
       Dconst = zero
       dvc = zero
       Denom = zero
       Acoef = zero
       Bcoef = zero
       Ccoef = zero
       Constr = zero
       Eaddr = zero
       GAconst = zero
       GBcoef = zero
    endif

    return
  end subroutine calc_coulomb_constants

  ! *
  ! * Determines VdW constaints
  ! *
  subroutine calc_vdw_constants(lvdw, ctonnb, ctofnb, k6, k12, dv6, dv12, &
                     GA6, GB6, GC6, GA12, GB12, GC12)
    use number,only:zero,one
    implicit none
    ! Input / Output
    logical, intent(in) :: lvdw
    real(chm_real), intent(in) :: ctonnb, ctofnb
    real(chm_real), intent(out) :: k6, k12, dv6, dv12
    real(chm_real), intent(out) :: GA6, GB6, GC6, GA12, GB12, GC12
    real(chm_real) ctofmcton

    if (lvdw) then
       ! Constants for VdW
       if (ctonnb < ctofnb) then
          k6 = ctofnb**3/(ctofnb**3 - ctonnb**3)
          k12 = ctofnb**6/(ctofnb**6 - ctonnb**6)
          dv6 = -one/(ctonnb*ctofnb)**3
          dv12 = -one/(ctonnb*ctofnb)**6
       else
          dv6 = -one/(ctofnb)**6
          dv12 = -one/(ctofnb)**12
       endif
       ctofmcton = ctofnb - ctonnb
       GA6  = - 6.0*(10*ctofnb - 7*ctonnb)/(ctofnb**8*ctofmcton**2)
       GB6  =   6.0*( 9*ctofnb - 7*ctonnb)/(ctofnb**8*ctofmcton**3)
       GC6  = 1/ctofnb**6 - (GA6*ctofmcton**3)/3.0 - (GB6*ctofmcton**4)/4.0
       GA12 = - 12.0*(16*ctofnb - 13*ctonnb)/(ctofnb**14*ctofmcton**2)
       GB12 =   12.0*(15*ctofnb - 13*ctonnb)/(ctofnb**14*ctofmcton**3)
       GC12 = 1/ctofnb**12 - (GA12*ctofmcton**3)/3.0 - (GB12*ctofmcton**4)/4.0
    else
       k6 = zero
       k12 = zero
       dv6 = zero
       dv12 = zero
       GA6  = zero
       GB6  = zero
       GC6  = zero
       GA12 = zero
       GB12 = zero
       GC12 = zero
    endif

    return
  end subroutine calc_vdw_constants

#if KEY_LJPME==1
  ! *
  ! * Calculates the LJPME grid correction (-1/r^6), i.e. the term computed in reciprocal
  ! * space assuming geometric mean combination rules.
  ! * r = distance
  ! * e = energy value
  ! * d = derivative
  ! *
  subroutine vdw_ljpme_grid(r, e, d, ctonnb, ctofnb, vdwmodel, vscale)
    use number,only: zero, one, two, six
    use pmeutil, only: dkappa, dopcut
    implicit none
    ! Input / Output
    real(chm_real), intent(out) :: e, d
    real(chm_real), intent(in) :: r, ctonnb, ctofnb, vscale
    integer, intent(in) :: vdwmodel
    ! Variables
    real(chm_real) r2, kr2, kr4, kr6, kr8, expterm

    r2 = r*r
    if(vdwmodel == VSH) then
       kr2 = dkappa*dkappa*r2
       kr4 = kr2*kr2
       kr6 = kr4*kr2
       kr8 = kr6*kr2
       expterm = exp(-kr2)
       e = (one - (one + kr2 + kr4/two)*expterm)/r**6 - vscale*dopcut/ctofnb**6
       d = -six*(one - (one + kr2 + kr4/two + kr6/six)*expterm)/r**7
    else if(vdwmodel == VSW) then
       kr2 = dkappa*dkappa*r2
       kr4 = kr2*kr2
       kr6 = kr4*kr2
       kr8 = kr6*kr2
       expterm = exp(-kr2)
       e = sw(r2,ctonnb**2,ctofnb**2)*(one - (one + kr2 + kr4/two)*expterm)/r**6
       d = -six*sw(r2,ctonnb**2,ctofnb**2)*(one - (one + kr2 + kr4/two + kr6/six)*expterm)/r**7 &
           +  two*dsw(r2,ctonnb**2,ctofnb**2)*(one - (one + kr2 + kr4/two)*expterm)/r**5
    else
       e = zero
       d = zero
    endif

    return
  end subroutine vdw_ljpme_grid
#endif /* KEY_LJPME */
  
  ! *
  ! * Calculates the VdW attraction (-1/r^6)
  ! * r = distance
  ! * e = energy value
  ! * d = derivative
  ! *
  subroutine vdw_attraction(r, e, d, ctonnb, ctofnb, vscale, vdwmodel, k6, dv6, &
                        GA6, GB6, GC6)
    use number,only:zero, one, two, six
#if KEY_LJPME==1
    use pmeutil, only: qljpme
#endif
    use nbips 
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: r
    real(chm_real), intent(out) :: e, d
    real(chm_real), intent(in) :: ctonnb, ctofnb
    real(chm_real), intent(in) :: vscale
    integer, intent(in) :: vdwmodel
    real(chm_real), intent(in) :: k6, dv6
    real(chm_real), intent(in) :: GA6, GB6, GC6
    ! Variables
    real(chm_real) r2
    real(chm_real) u2,u4,u6r,pvc,dpvc,sig2,sig6,enevc
    !real(chm_real) ga, gb, gc, ctofmcton

    r2 = r*r
    if (vdwmodel == VSH) then
#if KEY_LJPME==1
        if(qljpme) then
            e = -(one/r**6 - one/ctofnb**6)
            d = six/r**7
        else
#endif
            e = -(one/r**6 + r**6/ctofnb**12 - two/ctofnb**6)
            d = six/r**7 - six*r**5/ctofnb**12
#if KEY_LJPME==1
        endif
#endif
       e = vscale*e
       d = vscale*d
    elseif (vdwmodel == VSW) then
       e = -sw(r2,ctonnb**2,ctofnb**2)/r**6
       d = six*sw(r2,ctonnb**2,ctofnb**2)/r**7 - two*dsw(r2,ctonnb**2,ctofnb**2)/r**5
       e = vscale*e
       d = vscale*d
    elseif (vdwmodel == VFSW) then
       if (r < ctofnb) then
          if (r > ctonnb) then
             e = -k6*(r**(-3) - ctofnb**(-3))**2
             d = six*k6*(r**(-3) - ctofnb**(-3))/r**4
          else
             e = -(one/r**6 + dv6)
             d = six/r**7
          endif
       else
          e = zero
          d = zero
       endif
       e = vscale*e
       d = vscale*d
    elseif (vdwmodel == VGSH) then
       ! MGL: Find out when to use "six" instead of 6, etc.
       if (r < ctofnb) then
          !ctofmcton = ctofnb - ctonnb
          !ga = - 6.0*(10*ctofnb - 7*ctonnb)/(ctofnb**8*ctofmcton**2)
          !gb =   6.0*( 9*ctofnb - 7*ctonnb)/(ctofnb**8*ctofmcton**3)
          !gc = 1/ctofnb**6 - (ga*ctofmcton**3)/3.0 - (gb*ctofmcton**4)/4.0
          !write(*,*) 'MGL G6',ga,GA6,gb,GB6,gc,GC6
          if (r > ctonnb) then
             ! MGL: Move calculation of a, b, c to
             ! calc_vdw_constants.
             !e = -(r**(-6) - (ga*(r-ctonnb)**3)/3.0 - (gb*(r-ctonnb)**4)/4.0 - gc)
             !d = six/r**7 + ga*(r-ctonnb)**2 + gb*(r-ctonnb)**3
             e = -(r**(-6) - (GA6*(r-ctonnb)**3)/3.0 - (GB6*(r-ctonnb)**4)/4.0 - GC6)
             d = six/r**7 + GA6*(r-ctonnb)**2 + GB6*(r-ctonnb)**3
          else
             !e = - r**(-6) + gc
             e = - r**(-6) + GC6
             d = six/r**7
          endif
          !write(*,*) 'MGL V6 ga ',ga,' gb ',gb,' gc ',gc
          !write(*,*) 'MGL ctonnb ',ctonnb,' ctofnb ',ctofnb
          !write(*,*) 'MGL e ',e,' d ',d,' r ',r
       else
          e = zero
          d = zero
       endif
       e = vscale*e
       d = vscale*d       
    elseif (vdwmodel == VIPS) then
       u2 = r*r*rips2r
       u4 = u2*u2
       u6r = one/(u4*u2) 
       pvc=vscale*u6r+aipsvc(0)+u2*(aipsvc(1)+u2*(aipsvc(2)+u2*(aipsvc(3) &
            +u2*(aipsvc(4)+u4*(aipsvc(5)+u4*aipsvc(6))))))-pipsvcc
       dpvc=-vscale*six*u6r+u2*(bipsvc(1)+u2*(bipsvc(2)+u2*(bipsvc(3) &
            +u2*(bipsvc(4)+u4*(bipsvc(5)+u4*bipsvc(6))))))
       !sgshsq=rsclf(i)*rsclf(j)*cnba(ic)
       sig2=rips2r
       sig6=sig2*sig2*sig2
       enevc = -sig6
       e = enevc*pvc
       d = enevc*dpvc/r
    endif

    return
  end subroutine vdw_attraction

  ! *
  ! * Calculates the VdW repulsion (1/r^12)
  ! * r = distance
  ! * e = energy value
  ! * d = derivative
  ! *
  subroutine vdw_repulsion(r, e, d, ctonnb, ctofnb, vscale, vdwmodel, k12, dv12, &
                        GA12, GB12, GC12)
    use number,only:zero, one, two, three, twelve
#if KEY_LJPME==1
    use pmeutil, only: qljpme
#endif
    use nbips
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: r
    real(chm_real), intent(out) :: e, d
    real(chm_real), intent(in) :: ctonnb, ctofnb
    real(chm_real), intent(in) :: vscale
    integer, intent(in) :: vdwmodel
    real(chm_real), intent(in) :: k12, dv12
    real(chm_real), intent(in) :: GA12, GB12, GC12
    ! Variables
    real(chm_real) r2
    real(chm_real) u2,u4,u12r,pva,dpva,sig2,sig6,sig12,eneva
    !real(chm_real) ga, gb, gc, ctofmcton
    r2 = r*r
    if (vdwmodel == VSH) then
#if KEY_LJPME==1
       if(qljpme) then
          e = one/r**12 - one/ctofnb**12
          d = -twelve/r**13
       else
#endif
          e = one/r**12 + two*r**6/ctofnb**18 - three/ctofnb**12
          d = -twelve/r**13 + twelve*r**5/ctofnb**18
#if KEY_LJPME==1
       endif
#endif
       e = vscale*e
       d = vscale*d
    elseif (vdwmodel == VSW) then
       e = sw(r2,ctonnb**2,ctofnb**2)/r**12
       d = -twelve*sw(r2,ctonnb**2,ctofnb**2)/r**13 + two*dsw(r2,ctonnb**2,ctofnb**2)/r**11
       e = vscale*e
       d = vscale*d
    elseif (vdwmodel == VFSW) then
       if (r < ctofnb) then
          if (r > ctonnb) then
             e = k12*(r**(-6) - ctofnb**(-6))**2
             d = -twelve*k12*(r**(-6) - ctofnb**(-6))/r**7
          else
             e = one/r**12 + dv12
             d = -twelve/r**13
          endif
       else
          e = zero
          d = zero
       endif
       e = vscale*e
       d = vscale*d
    elseif (vdwmodel == VGSH) then
       ! MGL: Find out when to use "six" instead of 6, etc.
       if (r < ctofnb) then
          !ctofmcton = ctofnb - ctonnb
          !ga = - 12.0*(16*ctofnb - 13*ctonnb)/(ctofnb**14*ctofmcton**2)
          !gb =   12.0*(15*ctofnb - 13*ctonnb)/(ctofnb**14*ctofmcton**3)
          !gc = 1/ctofnb**12 - (ga*ctofmcton**3)/3.0 - (gb*ctofmcton**4)/4.0
          !write(*,*) 'MGL G12',ga,GA12,gb,GB12,gc,GC12
          if (r > ctonnb) then
             ! MGL: Move calculation of a, b, c to
             ! calc_vdw_constants.
             
             !e = r**(-12) - (ga*(r-ctonnb)**3)/3.0 - (gb*(r-ctonnb)**4)/4.0 - gc
             !d = -(twelve/r**13 + ga*(r-ctonnb)**2 + gb*(r-ctonnb)**3)
             e = r**(-12) - (GA12*(r-ctonnb)**3)/3.0 - (GB12*(r-ctonnb)**4)/4.0 - GC12
             d = -(twelve/r**13 + GA12*(r-ctonnb)**2 + GB12*(r-ctonnb)**3)
          else
             !e = r**(-12) - gc
             e = r**(-12) - GC12
             d = - twelve/r**13
          endif
          ! print out the terms here to double-check. 
          ! First, check with Python to make sure the code is shifting correctly
          ! for c12 = 1, and that I'm getting cton and ctof correct.
          !write(*,*) 'MGL V12 ga ',ga,' gb ',gb,' gc ',gc
          !write(*,*) 'MGL ctonnb ',ctonnb,' ctofnb ',ctofnb
          !write(*,*) 'MGL e ',e,' d ',d,' r ',r
       else 
          e = zero
          d = zero
       endif
       e = vscale*e
       d = vscale*d
    elseif (vdwmodel == VIPS) then
       u2 = r*r*rips2r
       u4 = u2*u2
       u12r = (one/(u4*u2))**2 
       pva=vscale*u12r+aipsva(0)+u2*(aipsva(1)+u2*(aipsva(2)+u2*(aipsva(3) &
            +u4*(aipsva(4)+u4*(aipsva(5)+u4*aipsva(6))))))-pipsvac
       dpva=-vscale*twelve*u12r+u2*(bipsva(1)+u2*(bipsva(2)+u2*(bipsva(3) &
            +u4*(bipsva(4)+u4*(bipsva(5)+u4*bipsva(6))))))
       !sgshsq=rsclf(i)*rsclf(j)*cnba(ic)
       sig2=rips2r
       sig6=sig2*sig2*sig2
       sig12=sig6*sig6
       eneva = sig12
       e = eneva*pva
       d = eneva*dpva/r
    endif

    return
  end subroutine vdw_repulsion

  ! *
  ! * Calculates the electrostatic interaction
  ! * r = distance
  ! * e = energy value
  ! * d = derivative
  ! *
  subroutine electrostatic(r, e, d, ctofnb, ctonnb, elecmodel, kappa, qscale, &
       Acoef, Bcoef, Ccoef, Constr, Denom, Aconst, Bconst, Cconst, Dconst, Eaddr, dvc, &
       GAconst, GBcoef)
    use consta,only:pi
    use number,only:one, two, three, four, five, eight, zero
    use nbips
    implicit none
    
#ifdef __PGI
    interface
       ! use C math lib until PGI ships this part of F08
       function erfc(x) bind(c)
         use, intrinsic :: iso_c_binding
         real(c_double) :: erfc
         real(c_double), intent(in), value :: x
       end function erfc
    end interface
#endif
    
    ! Input / Output
    real(chm_real), intent(in) :: r
    real(chm_real), intent(out) :: e, d
    real(chm_real), intent(in) :: ctofnb, ctonnb
    integer, intent(in) :: elecmodel
    real(chm_real), intent(in) :: kappa, qscale
    real(chm_real), intent(in) :: Acoef, Bcoef, Ccoef, Constr, Denom
    real(chm_real), intent(in) :: Aconst, Bconst, Cconst, Dconst, Eaddr, dvc
    real(chm_real), intent(in) :: GAconst, GBcoef
    ! Variables
    real(chm_real) r2, r3, spi
    real(chm_real) ctofnb2, ctonnb2
    real(chm_real) ctofnb4, ctofnb5
    !real(chm_real) ga, gb, gc, ctofmcton
    real(chm_real) u1,u2,p2,dpe,enep,pe
    
    ctofnb2 = ctofnb*ctofnb
    ctonnb2 = ctonnb*ctonnb

    r2 = r*r
    
    if (elecmodel == EWALD) then
       e = (erfc(kappa*r) + qscale - one)/r
       d = -two*kappa*exp(-(kappa*r)**2)/(sqrt(pi)*r) - erfc(kappa*r)/r2 - (qscale - one)/r2
    elseif (elecmodel == CSHIFT) then
       e = qscale/r*(one - two*r/ctofnb + r2/ctofnb2)
       d = qscale/r*(r/ctofnb2 - one/r)
    elseif (elecmodel == CFSWIT) then
       if (r <= ctonnb) then
          e = qscale*(one/r + dvc)
          d = -qscale/r2
       else
          e = qscale*(Aconst*(one/r - one/ctofnb) + Bconst*(ctofnb - r) + &
               Cconst*(ctofnb**3 - r**3) + &
               Dconst*(ctofnb**5 - r**5))
          d = -qscale*(Aconst/r2 + Bconst + three*Cconst*r2 + five*Dconst*r2**2)
       endif
    elseif (elecmodel == CSHFT) then
       ! Shift 1/r energy
       e = qscale/r*(one - r2/ctofnb2)**2
       d = -qscale*(one/r2*(one - r2/ctofnb2)**2 + four/ctofnb2*(one - r2/ctofnb2))
    elseif (elecmodel == CSWIT) then
       ! Switch 1/r energy
       e = qscale/r*sw(r2,ctonnb2,ctofnb2)
       d = -qscale/r2*sw(r2,ctonnb2,ctofnb2) + two*qscale*dsw(r2,ctonnb2,ctofnb2)
    elseif (elecmodel == RSWIT) then
       ! Switch 1/r^2 energy
       e = qscale/r2*sw(r2,ctonnb2,ctofnb2)
       d = two*qscale*(-one/(r2*r)*sw(r2,ctonnb2,ctofnb2) + dsw(r2,ctonnb2,ctofnb2)/r)
    elseif (elecmodel == RSHFT) then
       ! Shift 1/r^2 energy
       e = qscale/r2*(one - r2/ctofnb2)**2
       d = -qscale*(two/r**3*(one - r2/ctofnb2)**2 + &
            four/(r*ctofnb2)*(one - r2/ctofnb2))
    elseif (elecmodel == RSHIFT) then
       ! Shift 1/r^2 force with (r/rc -1)
       e = qscale/r2*(one - two*r/ctofnb + r2/ctofnb2)
       !          d = qscale/r2*(r/ctofnb - one)
       d = qscale/r2*two*(one/ctofnb - one/r)
    elseif (elecmodel == RFSWIT) then
       ! Switch 1/r^2 force
       if (r <= ctonnb) then
          e = qscale*(one/r2 + Eaddr)
          d = -two*qscale/(r2*r)
       else
          e = qscale*(Acoef/r2 - two*Bcoef*log(r) - r2*(Ccoef + r2*Denom) + Constr)
          d = -two*qscale*(Acoef/(r*r2) + Bcoef/r + Ccoef*r + 2*Denom*r2*r)
       endif
    elseif (elecmodel == GSHFT) then
       ! GROMACS style shift 1/r^2 force
       ! MGL special casing ctonnb=0 might speed this up
       ! NOTE THAT THIS EXPLICITLY ASSUMES ctonnb = 0
       
       if (r < ctofnb) then
          ctofnb4 = ctofnb2*ctofnb2
          ctofnb5 = ctofnb4*ctofnb
          e = qscale*(one/r - GAconst + r*r2*GBcoef - r2*r2/ctofnb5)
          d = -qscale*(one/r2 - 5.0*r2/ctofnb4 +4*r2*r/ctofnb5)
          !ctofmcton = ctofnb - ctonnb
          !ga = - (5*ctofnb - 2*ctonnb)/(ctofnb**3*ctofmcton**2)
          !gb =   (4*ctofnb - 2*ctonnb)/(ctofnb**3*ctofmcton**3)
          !gc = 1/ctofnb - (ga*ctofmcton**3)/3.0 - (gb*ctofmcton**4)/4.0
          !if (r <= ctonnb) then
          !   e = qscale*(one/r + gc)
          !   d = -qscale/r2
          !else
          !   e = qscale*(one/r - (ga*(r-ctonnb)**3)/3.0 + (gb*(r-ctonnb)**4)/4.0 - gc)
          !   d = -qscale*(one/r2 + ga*(r-ctonnb)**2 + gb*(r-ctonnb)**3)
          !endif
       else
          ! MGL ask AP why this clause isn't in the others
          e = zero
          d = zero
       endif
    elseif (elecmodel == EIPS) then
      ! Constant dielectric IPS
      ! Will clean this up
      u1 =  r*ripsr
      u2=u1*u1
      pe = qscale/u1 + u2*(aipse(1)+u2*(aipse(2)+u2*(aipse(3)+u2*(aipse(4)+u2*(aipse(5)+u2*aipse(6))))))-pipsec 
      dpe = -qscale/u1 + u2*(bipse(1)+u2*(bipse(2)+u2*(bipse(3)+u2*(bipse(4)+u2*(bipse(5)+u2*bipse(6))))))

      e = ripsr*pe
      d = ripsr*dpe/r
    endif
    
    return
  end subroutine electrostatic

  ! *
  ! * Switching function from CHARMM J. Comp. Chem. 1983 paper
  ! *
  real(chm_real) function sw(x, xon, xoff)
    use number,only:zero,one,two,three
    implicit none
    ! Input
    real(chm_real) x, xon, xoff

    if (x <= xon) then
       sw = one
    elseif (x > xoff) then
       sw = zero
    else
       sw = (xoff-x)**2*(xoff + two*x - three*xon)/(xoff-xon)**3
    endif

  end function sw

  ! *
  ! * Derivative of switching function from CHARMM J. Comp. Chem. 1983 paper
  ! *
  real(chm_real) function dsw(x, xon, xoff)
    use number,only:zero,six
    implicit none
    ! Input
    real(chm_real) x, xon, xoff

    if (x <= xon) then
       dsw = zero
    elseif (x > xoff) then
       dsw = zero
    else
       dsw = six*(xoff-x)*(xon-x)/(xoff-xon)**3
    endif

  end function dsw

  ! *
  ! * Calculates virial without shifting the atoms. This virial contribution only
  ! * takes into account forces that are acting within the periodic box
  ! * (i.e. periodic boundary conditions are NOT taken into account)
  ! *
  subroutine calc_virial_noshift(vpress, x, y, z, forcex, forcey, forcez, &
       n_atomlist, atomlist)
    use number,only:zero,three
    implicit none
    ! Input / Output
    real(chm_real), intent(out) :: vpress(9)
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    real(chm_real), intent(in) :: forcex(*), forcey(*), forcez(*)
    integer, intent(in) :: n_atomlist, atomlist(:)
    ! Variables
    real(chm_real) vxx, vxy, vxz, vyx, vyy, vyz, vzx, vzy, vzz
    integer i, j

    vxx = zero
    vxy = zero
    vxz = zero
    vyx = zero
    vyy = zero
    vyz = zero
    vzx = zero
    vzy = zero
    vzz = zero

    do j=1,n_atomlist
       i = atomlist(j)
       vxx = vxx + x(i)*forcex(i)
       vxy = vxy + x(i)*forcey(i)
       vxz = vxz + x(i)*forcez(i)
       
       vyx = vyx + y(i)*forcex(i)
       vyy = vyy + y(i)*forcey(i)
       vyz = vyz + y(i)*forcez(i)
       
       vzx = vzx + z(i)*forcex(i)
       vzy = vzy + z(i)*forcey(i)
       vzz = vzz + z(i)*forcez(i)
    enddo

    vpress(1) = -(vxx)
    vpress(2) = -(vxy)
    vpress(3) = -(vxz)

    vpress(4) = -(vyx)
    vpress(5) = -(vyy)
    vpress(6) = -(vyz)

    vpress(7) = -(vzx)
    vpress(8) = -(vzy)
    vpress(9) = -(vzz)

    return
  end subroutine calc_virial_noshift

  ! *
  ! * Calculates the virial tensor due to non-bonded interactions
  ! * If sflag = .true. calculates also the shift part of the virial
  ! *
  subroutine calc_virial(vprop, vpress, x, y, z, forcex, forcey, forcez, &
       groupl, ngrpl, sflag)
    use groupxfast,only:group,groupsh,group_out
    use number,only:zero,three
    use domdec_local,only:scoordtab, sforce
    use domdec_common,only:boxx, boxy, boxz
    implicit none
    ! Input / Output
    real(chm_real), intent(out) :: vprop, vpress(9)
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    real(chm_real), intent(in) :: forcex(*), forcey(*), forcez(*)
    integer, intent(in) :: groupl(*), ngrpl
    logical, intent(in) :: sflag
    ! Variables
    real(chm_real) sxx, sxy, sxz, syx, syy, syz, szx, szy, szz
    real(chm_real) vxx, vxy, vxz, vyx, vyy, vyz, vzx, vzy, vzz
    real(chm_real) xd, yd, zd
    integer i, ii, ig, is, iq

    vxx = zero
    vxy = zero
    vxz = zero
    vyx = zero
    vyy = zero
    vyz = zero
    vzx = zero
    vzy = zero
    vzz = zero

!$omp parallel do private(ig, i, is, iq, xd, yd, zd, ii) &
!$omp&            reduction(+:vxx, vxy, vxz, vyx, vyy, vyz, vzx, vzy, vzz)
    do ig=1,ngrpl
       i = groupl(ig)
       call group_out(group(i), is, iq)
       xd = groupsh(1,i)*boxx
       yd = groupsh(2,i)*boxy
       zd = groupsh(3,i)*boxz
       do ii=is,iq
          vxx = vxx + (x(ii)+xd)*forcex(ii)
          vxy = vxy + (x(ii)+xd)*forcey(ii)
          vxz = vxz + (x(ii)+xd)*forcez(ii)
       
          vyx = vyx + (y(ii)+yd)*forcex(ii)
          vyy = vyy + (y(ii)+yd)*forcey(ii)
          vyz = vyz + (y(ii)+yd)*forcez(ii)
       
          vzx = vzx + (z(ii)+zd)*forcex(ii)
          vzy = vzy + (z(ii)+zd)*forcey(ii)
          vzz = vzz + (z(ii)+zd)*forcez(ii)
       enddo
    enddo
!$omp end parallel do

    sxx = zero
    sxy = zero
    sxz = zero
    syx = zero
    syy = zero
    syz = zero
    szx = zero
    szy = zero
    szz = zero

    if (sflag) then
       do i=1,81,3    ! 81 = 27*3
          sxx = sxx + scoordtab%dp(i)*sforce(i)
          sxy = sxy + scoordtab%dp(i)*sforce(i+1)
          sxz = sxz + scoordtab%dp(i)*sforce(i+2)

          syx = syx + scoordtab%dp(i+1)*sforce(i)
          syy = syy + scoordtab%dp(i+1)*sforce(i+1)
          syz = syz + scoordtab%dp(i+1)*sforce(i+2)

          szx = szx + scoordtab%dp(i+2)*sforce(i)
          szy = szy + scoordtab%dp(i+2)*sforce(i+1)
          szz = szz + scoordtab%dp(i+2)*sforce(i+2)
       enddo
    endif
    
    vpress(1) = -(vxx + sxx)
    vpress(2) = -(vxy + sxy)
    vpress(3) = -(vxz + sxz)

    vpress(4) = -(vyx + syx)
    vpress(5) = -(vyy + syy)
    vpress(6) = -(vyz + syz)

    vpress(7) = -(vzx + szx)
    vpress(8) = -(vzy + szy)
    vpress(9) = -(vzz + szz)

    vprop = (vpress(1) + vpress(5) + vpress(9))/three

    return
  end subroutine calc_virial

#endif /* (domdec_main)*/

end module enbxfast

