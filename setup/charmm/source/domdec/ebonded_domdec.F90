module ebonded_domdec

#if KEY_DOMDEC==1 /* domdec_main */

  use chm_kinds
  use dimens_fcm
#if KEY_DOMDEC_GPU==1
  use iso_c_binding
#endif
  ! For ebonded_domdec_kernel.inc
  use domdec_bonded_types,only:bondlist_t, anglelist_t, dihelist_t
  use domdec_local_types,only:xyzq_dp_t, xyzq_sp_t
  use consta,only:pi
  implicit none
  private

  ! ANGLE style
  logical :: q_gromacs_style = .false., q_charmm_style = .false.

#if KEY_DOMDEC_GPU==1
  ! Determines wether the bonded lists are current or not.
  ! If .false., we copy the bonded lists to GPU
  logical q_lists_current
  logical q_calc_bond, q_calc_angle, q_calc_ureyb, q_calc_dihe, q_calc_imdihe
#endif  

  ! Public subroutines
  public ebond_domdec, eureyb_domdec, eangle_domdec, edihe_domdec, eimdihe_domdec, e14_domdec
  public thole_ex14_domdec
#if KEY_DOMDEC_GPU==1
  public ebonded_gpu
#endif

#endif /* domdec_main */

contains
  
#if KEY_DOMDEC_GPU==1
  ! *
  ! * All bonded interactions evaluated on the GPU
  ! * This subroutine launches the kernel(s)
  ! *
  subroutine ebonded_gpu()
    use domdec_common,only:boxx, boxy, boxz, q_test, q_print_output
    use domdec_bonded,only:nbondtbl, bondlist, nangletbl, anglelist, nangletbl, ureyblist, &
         ndihetbl, dihelist, nimdihetbl, imdihelist
    use bonded_gpu_mod,only:setup_bonded_list_gpu, calc_bonded_gpu
    use energym,only:qdyncall
    use reawri,only:qcnstp
    implicit none
    ! Variables
    logical q_calc_energy, q_calc_virial

    if (.not.q_lists_current) then
       call setup_bonded_list_gpu(nbondtbl, bondlist, nangletbl, anglelist, nangletbl, ureyblist, &
            ndihetbl, dihelist, nimdihetbl, imdihelist)
       q_lists_current = .true.
    endif

    q_calc_energy = (.not.qdyncall) .or. q_print_output .or. q_test
    q_calc_virial = (.not.qdyncall) .or. q_print_output .or. q_test .or. qcnstp
    call calc_bonded_gpu(boxx, boxy, boxz, q_calc_energy, q_calc_virial, q_calc_bond, &
         q_calc_angle, q_calc_ureyb, q_calc_dihe, q_calc_imdihe)

    q_calc_bond = .false.
    q_calc_angle = .false.
    q_calc_ureyb = .false.
    q_calc_dihe = .false.
    q_calc_imdihe = .false.

    return
  end subroutine ebonded_gpu

#endif

#if KEY_DOMDEC==1 /* domdec_main */
  ! *
  ! * Bonds
  ! *
  subroutine ebond_domdec(ib, jb, icb, cbb, cbc, epot)
    use number,only:zero, one
    use domdec_common,only:q_single
    use domdec_bonded,only:nbondtbl, bondtbl, bondlist, q_bondlist_current, &
         build_bondlist_ps, build_bondlist_pd, realloc_bondlist
    use domdec_local,only:scoordtab, force_loc, q_force_loc_used, sforce, xyzq_loc, glo2loc_ind
    use stream,only:prnlev, outu
#if KEY_BLOCK==1
    use lambdam,only:qmld, qsobo, soboa, sobob, qldm_scalecons, mldbondi, mldbondj, mldbonds, msld_get_scale, msld_lambdaforce, biflam, biflam2
    use block_ltm,only:nblock
    use block_fcm,only:qnobo
    use domdec_bonded_block,only:nbondtbl_block, bondtbl_block_pos, bondtbl_block, &
         lower_triangle_index, build_bondlist_block_ps, build_bondlist_block_pd
    use domdec_block,only:biflam_loc, biflam2_loc
#endif
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu, gpu_code_version
#endif
    implicit none
    ! Input / Output
    integer, intent(in) :: ib(:), jb(:), icb(:)
    real(chm_real), intent(in) :: cbb(:), cbc(:)
    real(chm_real), intent(inout) :: epot
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num
#endif
    ! Variables
#if KEY_BLOCK==1
    real(chm_real) fscale, epot_block, efsoft_block
    integer ibl, jbl, k
#endif
    integer tid
    real(chm_real) epot_loc, sforce_loc(3*27)

    if (.not.q_bondlist_current) then
       call realloc_bondlist(nbondtbl)
#if KEY_DOMDEC_GPU==1
       q_lists_current = .false.
#endif
#if KEY_BLOCK==1
       if (qmld .and. ((.not.qnobo) .or. qsobo .or. qldm_scalecons)) then
          if (q_single) then
             call build_bondlist_block_ps(nbondtbl, bondtbl, ib, jb, icb, &
                  mldbondi, mldbondj, mldbonds, qsobo, &
                  glo2loc_ind, xyzq_loc%sp, bondlist, &
                  nbondtbl_block, bondtbl_block_pos, bondtbl_block)
          else
             call build_bondlist_block_pd(nbondtbl, bondtbl, ib, jb, icb, &
                  mldbondi, mldbondj, mldbonds, qsobo, &
                  glo2loc_ind, xyzq_loc%dp, bondlist, &
                  nbondtbl_block, bondtbl_block_pos, bondtbl_block)
          endif
       else
#endif
          if (q_single) then
             call build_bondlist_ps(nbondtbl, bondtbl, ib, jb, icb, &
                  glo2loc_ind, xyzq_loc%sp, bondlist)
          else
             call build_bondlist_pd(nbondtbl, bondtbl, ib, jb, icb, &
                  glo2loc_ind, xyzq_loc%dp, bondlist)
          endif
#if KEY_BLOCK==1
       endif
#endif
       q_bondlist_current = .true.
    endif

    if (q_single) then
       if (prnlev > 6) write (outu,'(a)') ' ebond_domdec: calling ebond_domdec_loop_ps'
    else
       if (prnlev > 6) write (outu,'(a)') ' ebond_domdec: calling ebond_domdec_loop_pd'
    endif

    sforce_loc = zero
    epot_loc = zero

#if KEY_BLOCK==1
    if (qmld .and. ((.not.qnobo) .or. qsobo .or. qldm_scalecons)) then
       biflam_loc = zero
       biflam2_loc = zero
!$omp parallel private(tid, ibl, jbl, k, fscale, epot_block, efsoft_block) &
!$omp&         reduction(+:epot_loc, sforce_loc, biflam_loc, biflam2_loc)
#ifdef _OPENMP
       tid = omp_get_thread_num()
#else
       tid = 0
#endif
       do jbl=1,nblock
          do ibl=jbl,nblock
             k = lower_triangle_index(ibl, jbl)
             ! Get correct scaling factor for interaction between blocks ibl and jbl
             call msld_get_scale(ibl, jbl, fscale)
             if (q_single) then
                call ebond_domdec_loop_ps(nbondtbl_block(k), &
                     bondlist(bondtbl_block_pos(k):bondtbl_block_pos(k+1)-1), &
                     cbb, cbc, xyzq_loc%sp, scoordtab%sp, real(fscale,kind=chm_real4), &
                     epot_block, force_loc(tid)%array, sforce_loc)
             else
                call ebond_domdec_loop_pd(nbondtbl_block(k), &
                     bondlist(bondtbl_block_pos(k):bondtbl_block_pos(k+1)-1), &
                     cbb, cbc, xyzq_loc%dp, scoordtab%dp, fscale, &
                     epot_block, force_loc(tid)%array, sforce_loc)
             endif
             if (fscale /= one .and. fscale > zero) then
                call msld_lambdaforce(ibl, jbl, epot_block, biflam_loc, biflam2_loc)
             endif
             epot_loc = epot_loc + epot_block*fscale

             ! soft bonds
             if (qsobo) then
                k = k + lower_triangle_index(nblock, nblock)
                ! Get correct scaling factor for interaction between blocks ibl and jbl
                call msld_get_scale(ibl, jbl, fscale)
                if (q_single) then
                   call esbond_domdec_loop_ps(nbondtbl_block(k), &
                        bondlist(bondtbl_block_pos(k):bondtbl_block_pos(k+1)-1), &
                        cbb, cbc, xyzq_loc%sp, scoordtab%sp, real(fscale**sobob,kind=chm_real4), real(soboa,kind=chm_real4), &
                        epot_block, efsoft_block, force_loc(tid)%array, sforce_loc)
                else
                   call esbond_domdec_loop_pd(nbondtbl_block(k), &
                        bondlist(bondtbl_block_pos(k):bondtbl_block_pos(k+1)-1), &
                        cbb, cbc, xyzq_loc%dp, scoordtab%dp, fscale**sobob, soboa, &
                        epot_block, efsoft_block, force_loc(tid)%array, sforce_loc)
                endif
                if (fscale /= one .and. fscale > zero) then
                   call msld_lambdaforce(ibl, jbl, sobob*(fscale**(sobob-1))*(epot_block+efsoft_block), biflam_loc, biflam2_loc)
                endif
                epot_loc = epot_loc + epot_block*(fscale**sobob)
             endif
          enddo
       enddo
!$omp end parallel
       biflam = biflam + biflam_loc
       biflam2 = biflam2 + biflam2_loc
       q_force_loc_used = .true.
    else
#endif
#if KEY_DOMDEC_GPU==1
       if (q_gpu .and. gpu_code_version==2) then
          q_calc_bond = .true.
          return
       endif
#endif
!$omp parallel private(tid) reduction(+:epot_loc, sforce_loc)
#ifdef _OPENMP
       tid = omp_get_thread_num()
#else
       tid = 0
#endif
       if (q_single) then
          call ebond_domdec_loop_ps(nbondtbl, bondlist, cbb, cbc, xyzq_loc%sp, scoordtab%sp, &
               1.0_chm_real4, epot_loc, force_loc(tid)%array, sforce_loc)
       else
          call ebond_domdec_loop_pd(nbondtbl, bondlist, cbb, cbc, xyzq_loc%dp, scoordtab%dp, &
               1.0_chm_real, epot_loc, force_loc(tid)%array, sforce_loc)
       endif
!$omp end parallel
       q_force_loc_used = .true.       
#if KEY_BLOCK==1
    endif
#endif

    epot = epot + epot_loc
    sforce = sforce + sforce_loc

    !call write_bond_data(nbondtbl, bondlist, cbb, cbc, 'bondlist.txt', 'bondcoef.txt')

    return
  end subroutine ebond_domdec

  ! *
  ! * Writes bond data in to a file
  ! *
  subroutine write_bond_data(n, bondlist, cbb, cbc, filename_list, filename_coef)
    use domdec_bonded_types,only:bondlist_t
    implicit none
    ! Input
    integer, intent(in) :: n
    type(bondlist_t), intent(in) :: bondlist(:)
    real(chm_real), intent(in) :: cbb(:), cbc(:)
    character(*), intent(in) :: filename_list, filename_coef
    ! Variables
    integer i, ncoef

    ncoef = 0
    open(120,file=filename_list)
    do i=1,n
       write (120,'(2i8,2i4)') bondlist(i)%i, bondlist(i)%j, &
            bondlist(i)%itype, bondlist(i)%ishift
       ncoef = max(ncoef, bondlist(i)%itype)
    enddo
    close(120)

    open(120,file=filename_coef)
    do i=1,ncoef
       write (120,'(2e28.20)') cbb(i), cbc(i)
    enddo
    close(120)

    return
  end subroutine write_bond_data

  ! *
  ! * Urey-Bradley
  ! *
  subroutine eureyb_domdec(ib, jb, icb, cbb, cbc, epot)
    use number,only:zero, one
    use domdec_common,only:q_single
    use domdec_bonded,only:nangletbl, angletbl, ureyblist, q_ureyblist_current, &
         build_bondlist_ps, build_bondlist_pd, realloc_ureyblist
    use domdec_local,only:scoordtab, force_loc, q_force_loc_used, sforce, xyzq_loc, glo2loc_ind
    use stream,only:prnlev,outu
#if KEY_BLOCK==1
    use lambdam,only:qmld, qsobo, soboa, sobob, qldm_scalecons, msld_get_scale, msld_lambdaforce, mldureyc, mldureyi, mldureyj, mldureys, biflam, biflam2
    use block_ltm,only:nblock
    use block_fcm,only:qnoub
    use domdec_bonded_block,only:nureybtbl_block, ureybtbl_block_pos, ureybtbl_block, &
         lower_triangle_index, build_bondlist_block_ps, build_bondlist_block_pd
    use domdec_block,only:biflam_loc, biflam2_loc
#endif
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu, gpu_code_version
#endif
    implicit none
    ! Input / Output
    integer, intent(in) :: ib(:), jb(:), icb(:)
    real(chm_real), intent(in) :: cbb(:), cbc(:)
    real(chm_real), intent(inout) :: epot
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num
#endif
    ! Variables
#if KEY_BLOCK==1
    real(chm_real) fscale, epot_block, efsoft_block
    integer ibl, jbl, k
#endif
    integer tid
    real(chm_real) epot_loc, sforce_loc(3*27)

    if (.not.q_ureyblist_current) then
       call realloc_ureyblist(nangletbl)
#if KEY_DOMDEC_GPU==1
       q_lists_current = .false.
#endif
#if KEY_BLOCK==1
       if (qmld .and. ((.not.qnoub) .or. qsobo .or. qldm_scalecons)) then
          if (q_single) then
             call build_bondlist_block_ps(nangletbl, angletbl, ib, jb, icb, &
                  mldureyi, mldureyj, mldureys, qsobo, &
                  glo2loc_ind, xyzq_loc%sp, ureyblist, &
                  nureybtbl_block, ureybtbl_block_pos, ureybtbl_block)
          else
             call build_bondlist_block_pd(nangletbl, angletbl, ib, jb, icb, &
                  mldureyi, mldureyj, mldureys, qsobo, &
                  glo2loc_ind, xyzq_loc%dp, ureyblist, &
                  nureybtbl_block, ureybtbl_block_pos, ureybtbl_block)
          endif
       else
#endif
          if (q_single) then
             call build_bondlist_ps(nangletbl, angletbl, ib, jb, icb, &
                  glo2loc_ind, xyzq_loc%sp, ureyblist)
          else
             call build_bondlist_pd(nangletbl, angletbl, ib, jb, icb, &
                  glo2loc_ind, xyzq_loc%dp, ureyblist)
          endif
#if KEY_BLOCK==1
       endif
#endif
       q_ureyblist_current = .true.
    endif

    if (q_single) then
       if (prnlev > 6) write (outu,'(a)') ' eureyb_domdec: calling ebond_domdec_loop_ps'
    else
       if (prnlev > 6) write (outu,'(a)') ' eureyb_domdec: calling ebond_domdec_loop_pd'
    endif

    sforce_loc = zero
    epot_loc = zero
#if KEY_BLOCK==1
    if (qmld .and. ((.not.qnoub) .or. qsobo .or. qldm_scalecons)) then
       biflam_loc = zero
       biflam2_loc = zero
!$omp parallel private(tid, ibl, jbl, k, fscale, epot_block, efsoft_block) &
!$omp&         reduction(+:epot_loc, sforce_loc, biflam_loc, biflam2_loc)
#ifdef _OPENMP
       tid = omp_get_thread_num()
#else
       tid = 0
#endif
       do jbl=1,nblock
          do ibl=jbl,nblock
             k = lower_triangle_index(ibl, jbl)
             ! Get correct scaling factor for interaction between blocks ibl and jbl
             call msld_get_scale(ibl, jbl, fscale)
             if (q_single) then
                call ebond_domdec_loop_ps(nureybtbl_block(k), &
                     ureyblist(ureybtbl_block_pos(k):ureybtbl_block_pos(k+1)-1), &
                     cbb, cbc, xyzq_loc%sp, scoordtab%sp, &
                     real(fscale,kind=chm_real4), epot_block, force_loc(tid)%array, sforce_loc)
             else
                call ebond_domdec_loop_pd(nureybtbl_block(k), &
                     ureyblist(ureybtbl_block_pos(k):ureybtbl_block_pos(k+1)-1), &
                     cbb, cbc, xyzq_loc%dp, scoordtab%dp, &
                     fscale, epot_block, force_loc(tid)%array, sforce_loc)
             endif
             if (fscale /= one .and. fscale > zero) then
                call msld_lambdaforce(ibl, jbl, epot_block, biflam_loc, biflam2_loc)
             endif
             epot_loc = epot_loc + epot_block*fscale

             ! soft ureys
             if (qsobo) then
                k = k + lower_triangle_index(nblock, nblock)
                ! Get correct scaling factor for interaction between blocks ibl and jbl
                call msld_get_scale(ibl, jbl, fscale)
                if (q_single) then
                   call esbond_domdec_loop_ps(nureybtbl_block(k), &
                        ureyblist(ureybtbl_block_pos(k):ureybtbl_block_pos(k+1)-1), &
                        cbb, cbc, xyzq_loc%sp, scoordtab%sp, &
                        real(fscale**sobob,kind=chm_real4), real(soboa,kind=chm_real4), epot_block, efsoft_block, force_loc(tid)%array, sforce_loc)
                else
                   call esbond_domdec_loop_pd(nureybtbl_block(k), &
                        ureyblist(ureybtbl_block_pos(k):ureybtbl_block_pos(k+1)-1), &
                        cbb, cbc, xyzq_loc%dp, scoordtab%dp, &
                        fscale**sobob, soboa, epot_block, efsoft_block, force_loc(tid)%array, sforce_loc)
                endif
                if (fscale /= one .and. fscale > zero) then
                   call msld_lambdaforce(ibl, jbl, sobob*(fscale**(sobob-1))*(epot_block+efsoft_block), biflam_loc, biflam2_loc)
                endif
                epot_loc = epot_loc + epot_block*(fscale**sobob)
             endif
          enddo
       enddo
!$omp end parallel
       q_force_loc_used = .true.
       biflam = biflam + biflam_loc
       biflam2 = biflam2 + biflam2_loc
    else
#endif
#if KEY_DOMDEC_GPU==1
       if (q_gpu .and. gpu_code_version==2) then
          q_calc_ureyb = .true.
          return
       endif
#endif
!$omp parallel private(tid) reduction(+:epot_loc, sforce_loc)
#ifdef _OPENMP
       tid = omp_get_thread_num()
#else
       tid = 0
#endif
       if (q_single) then
          call ebond_domdec_loop_ps(nangletbl, ureyblist, cbb, cbc, xyzq_loc%sp, scoordtab%sp, &
               1.0_chm_real4, epot_loc, force_loc(tid)%array, sforce_loc)
       else
          call ebond_domdec_loop_pd(nangletbl, ureyblist, cbb, cbc, xyzq_loc%dp, scoordtab%dp, &
               1.0_chm_real, epot_loc, force_loc(tid)%array, sforce_loc)
       endif
!$omp end parallel
       q_force_loc_used = .true.
#if KEY_BLOCK==1
    endif
#endif

    epot = epot + epot_loc
    sforce = sforce + sforce_loc
    
    !call write_bond_data(nangletbl, ureyblist, cbb, cbc, 'ureyblist.txt', 'ureybcoef.txt')

    return
  end subroutine eureyb_domdec

  ! *
  ! * Angles
  ! *
  subroutine eangle_domdec(it, jt, kt, ict, ctb, ctc, epot)
    use number,only:zero, one
    use domdec_common,only:q_single
    use domdec_bonded,only:nangletbl, angletbl, anglelist, q_anglelist_current, &
         build_anglelist_ps, build_anglelist_pd, realloc_anglelist
    use domdec_local,only:scoordtab, force_loc, q_force_loc_used, sforce, xyzq_loc, glo2loc_ind
    use stream,only:prnlev,outu
#if KEY_BLOCK==1
    use lambdam,only:qmld, qsobo, sobon, qldm_scalecons, msld_get_scale, msld_lambdaforce, mldangc, mldangi, mldangj, mldangs, biflam, biflam2
    use block_ltm,only:nblock
    use block_fcm,only:qnoan
    use domdec_bonded_block,only:nangletbl_block, angletbl_block_pos, angletbl_block, &
         lower_triangle_index, build_anglelist_block_ps, build_anglelist_block_pd
    use domdec_block,only:biflam_loc, biflam2_loc
#endif
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu, gpu_code_version
#endif
    implicit none
    ! Input / Output
    integer, intent(in) :: it(:), jt(:), kt(:), ict(:)
    real(chm_real), intent(in) :: ctb(:), ctc(:)
    real(chm_real), intent(inout) :: epot
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num
#endif
    ! Variables
#if KEY_BLOCK==1
    real(chm_real) fscale, epot_block
    integer ibl, jbl, k
#endif
    integer tid
    real(chm_real) epot_loc, sforce_loc(3*27)

    if (.not.q_anglelist_current) then
       call realloc_anglelist(nangletbl)
#if KEY_DOMDEC_GPU==1
       q_lists_current = .false.
#endif
#if KEY_BLOCK==1
       if (qmld .and. ((.not.qnoan) .or. qsobo .or. qldm_scalecons)) then
          if (q_single) then
             call build_anglelist_block_ps(nangletbl, angletbl, it, jt, kt, ict, ctb, &
                  mldangi, mldangj, mldangs, qsobo, &
                  glo2loc_ind, xyzq_loc%sp, anglelist, &
                  nangletbl_block, angletbl_block_pos, angletbl_block, &
                  q_gromacs_style, q_charmm_style)
          else
             call build_anglelist_block_pd(nangletbl, angletbl, it, jt, kt, ict, ctb, &
                  mldangi, mldangj, mldangs, qsobo, &
                  glo2loc_ind, xyzq_loc%dp, anglelist, &
                  nangletbl_block, angletbl_block_pos, angletbl_block, &
                  q_gromacs_style, q_charmm_style)
          endif
       else
#endif
          if (q_single) then
             call build_anglelist_ps(nangletbl, angletbl, it, jt, kt, ict, ctb, &
                  glo2loc_ind, xyzq_loc%sp, anglelist, q_gromacs_style, q_charmm_style)
          else
             call build_anglelist_pd(nangletbl, angletbl, it, jt, kt, ict, ctb, &
                  glo2loc_ind, xyzq_loc%dp, anglelist, q_gromacs_style, q_charmm_style)
          endif
#if KEY_BLOCK==1
       endif
#endif
       q_anglelist_current = .true.
    endif

    if (.not.q_charmm_style .and. .not.q_gromacs_style) then
       call wrndie(-5,'<ebonded_domdec>','Undefined ANGLE style')
    endif
    
    if (prnlev > 6) then
       if (q_single) then
          if (q_charmm_style .and. q_gromacs_style) then
             write (outu,'(a)') ' eangle_domdec: calling eangle_domdec_loop_charmm_gromacs_ps'
          elseif (q_charmm_style) then
             write (outu,'(a)') ' eangle_domdec: calling eangle_domdec_loop_charmm_ps'
          else
             write (outu,'(a)') ' eangle_domdec: calling eangle_domdec_loop_gromacs_ps'
          endif
       else
          if (q_charmm_style .and. q_gromacs_style) then
             write (outu,'(a)') ' eangle_domdec: calling eangle_domdec_loop_charmm_gromacs_pd'
          elseif (q_charmm_style) then
             write (outu,'(a)') ' eangle_domdec: calling eangle_domdec_loop_charmm_pd'
          else
             write (outu,'(a)') ' eangle_domdec: calling eangle_domdec_loop_gromacs_pd'
          endif
       endif
    endif

    sforce_loc = zero
    epot_loc = zero

#if KEY_BLOCK==1
    if (qmld .and. ((.not.qnoan) .or. qsobo .or. qldm_scalecons)) then
       biflam_loc = zero
       biflam2_loc = zero
!$omp parallel private(tid, ibl, jbl, k, fscale, epot_block) &
!$omp&         reduction(+:epot_loc, sforce_loc, biflam_loc, biflam2_loc)
#ifdef _OPENMP
       tid = omp_get_thread_num()
#else
       tid = 0
#endif
       do jbl=1,nblock
          do ibl=jbl,nblock
             k = lower_triangle_index(ibl, jbl)
             ! Get correct scaling factor for interaction between blocks ibl and jbl
             call msld_get_scale(ibl, jbl, fscale)
             if (q_single) then
                if (q_charmm_style .and. q_gromacs_style) then
                   call eangle_domdec_loop_charmm_gromacs_ps(nangletbl_block(k), &
                        anglelist(angletbl_block_pos(k):angletbl_block_pos(k+1)-1), &
                        ctb, ctc, xyzq_loc%sp, scoordtab%sp, &
                        real(fscale,kind=chm_real4), epot_block, force_loc(tid)%array, sforce_loc)
                elseif (q_charmm_style) then
                   call eangle_domdec_loop_charmm_ps(nangletbl_block(k), &
                        anglelist(angletbl_block_pos(k):angletbl_block_pos(k+1)-1), &
                        ctb, ctc, xyzq_loc%sp, scoordtab%sp, &
                        real(fscale,kind=chm_real4), epot_block, force_loc(tid)%array, sforce_loc)
                else
                   call eangle_domdec_loop_gromacs_ps(nangletbl_block(k), &
                        anglelist(angletbl_block_pos(k):angletbl_block_pos(k+1)-1), &
                        ctb, ctc, xyzq_loc%sp, scoordtab%sp, &
                        real(fscale,kind=chm_real4), epot_block, force_loc(tid)%array, sforce_loc)
                endif
             else
                if (q_charmm_style .and. q_gromacs_style) then
                   call eangle_domdec_loop_charmm_gromacs_pd(nangletbl_block(k), &
                        anglelist(angletbl_block_pos(k):angletbl_block_pos(k+1)-1), &
                        ctb, ctc, xyzq_loc%dp, scoordtab%dp, &
                        fscale, epot_block, force_loc(tid)%array, sforce_loc)
                elseif (q_charmm_style) then
                   call eangle_domdec_loop_charmm_pd(nangletbl_block(k), &
                        anglelist(angletbl_block_pos(k):angletbl_block_pos(k+1)-1), &
                        ctb, ctc, xyzq_loc%dp, scoordtab%dp, &
                        fscale, epot_block, force_loc(tid)%array, sforce_loc)
                else
                   call eangle_domdec_loop_gromacs_pd(nangletbl_block(k), &
                        anglelist(angletbl_block_pos(k):angletbl_block_pos(k+1)-1), &
                        ctb, ctc, xyzq_loc%dp, scoordtab%dp, &
                        fscale, epot_block, force_loc(tid)%array, sforce_loc)
                endif
             endif
             if (fscale /= one .and. fscale > zero) then
                call msld_lambdaforce(ibl, jbl, epot_block, biflam_loc, biflam2_loc)
             endif
             epot_loc = epot_loc + epot_block*fscale

             ! soft angles
             if (qsobo) then
                k = k + lower_triangle_index(nblock, nblock)
                ! Get correct scaling factor for interaction between blocks ibl and jbl
                call msld_get_scale(ibl, jbl, fscale)
                if (q_single) then
                   if (q_charmm_style .and. q_gromacs_style) then
                      call eangle_domdec_loop_charmm_gromacs_ps(nangletbl_block(k), &
                           anglelist(angletbl_block_pos(k):angletbl_block_pos(k+1)-1), &
                           ctb, ctc, xyzq_loc%sp, scoordtab%sp, &
                           real(fscale**sobon,kind=chm_real4), epot_block, force_loc(tid)%array, sforce_loc)
                   elseif (q_charmm_style) then
                      call eangle_domdec_loop_charmm_ps(nangletbl_block(k), &
                           anglelist(angletbl_block_pos(k):angletbl_block_pos(k+1)-1), &
                           ctb, ctc, xyzq_loc%sp, scoordtab%sp, &
                           real(fscale**sobon,kind=chm_real4), epot_block, force_loc(tid)%array, sforce_loc)
                   else
                      call eangle_domdec_loop_gromacs_ps(nangletbl_block(k), &
                           anglelist(angletbl_block_pos(k):angletbl_block_pos(k+1)-1), &
                           ctb, ctc, xyzq_loc%sp, scoordtab%sp, &
                           real(fscale**sobon,kind=chm_real4), epot_block, force_loc(tid)%array, sforce_loc)
                   endif
                else
                   if (q_charmm_style .and. q_gromacs_style) then
                      call eangle_domdec_loop_charmm_gromacs_pd(nangletbl_block(k), &
                           anglelist(angletbl_block_pos(k):angletbl_block_pos(k+1)-1), &
                           ctb, ctc, xyzq_loc%dp, scoordtab%dp, &
                           fscale**sobon, epot_block, force_loc(tid)%array, sforce_loc)
                   elseif (q_charmm_style) then
                      call eangle_domdec_loop_charmm_pd(nangletbl_block(k), &
                           anglelist(angletbl_block_pos(k):angletbl_block_pos(k+1)-1), &
                           ctb, ctc, xyzq_loc%dp, scoordtab%dp, &
                           fscale**sobon, epot_block, force_loc(tid)%array, sforce_loc)
                   else
                      call eangle_domdec_loop_gromacs_pd(nangletbl_block(k), &
                           anglelist(angletbl_block_pos(k):angletbl_block_pos(k+1)-1), &
                           ctb, ctc, xyzq_loc%dp, scoordtab%dp, &
                           fscale**sobon, epot_block, force_loc(tid)%array, sforce_loc)
                   endif
                endif
                if (fscale /= one .and. fscale > zero) then
                   call msld_lambdaforce(ibl, jbl, sobon*(fscale**(sobon-1))*epot_block, biflam_loc, biflam2_loc)
                endif
                epot_loc = epot_loc + epot_block*(fscale**sobon)
             endif
          enddo
       enddo
!$omp end parallel
       biflam = biflam + biflam_loc
       biflam2 = biflam2 + biflam2_loc
       q_force_loc_used = .true.
    else
#endif
#if KEY_DOMDEC_GPU==1
       if (q_gpu .and. gpu_code_version==2) then
          q_calc_angle = .true.
          return
       endif
#endif
!$omp parallel private(tid) reduction(+:epot_loc, sforce_loc)
#ifdef _OPENMP
       tid = omp_get_thread_num()
#else
       tid = 0
#endif
       if (q_single) then
          if (q_charmm_style .and. q_gromacs_style) then
             call eangle_domdec_loop_charmm_gromacs_ps(nangletbl, anglelist, ctb, ctc, xyzq_loc%sp,&
                  scoordtab%sp, 1.0_chm_real4, epot_loc, force_loc(tid)%array, sforce_loc)
          elseif (q_charmm_style) then
             call eangle_domdec_loop_charmm_ps(nangletbl, anglelist, ctb, ctc, xyzq_loc%sp, &
                  scoordtab%sp, 1.0_chm_real4, epot_loc, force_loc(tid)%array, sforce_loc)
          else
             call eangle_domdec_loop_gromacs_ps(nangletbl, anglelist, ctb, ctc, xyzq_loc%sp, &
                  scoordtab%sp, 1.0_chm_real4, epot_loc, force_loc(tid)%array, sforce_loc)
          endif
       else
          if (q_charmm_style .and. q_gromacs_style) then
             call eangle_domdec_loop_charmm_gromacs_pd(nangletbl, anglelist, ctb, ctc, xyzq_loc%dp,&
                  scoordtab%dp, 1.0_chm_real, epot_loc, force_loc(tid)%array, sforce_loc)
          elseif (q_charmm_style) then
             call eangle_domdec_loop_charmm_pd(nangletbl, anglelist, ctb, ctc, xyzq_loc%dp,&
                  scoordtab%dp, 1.0_chm_real, epot_loc, force_loc(tid)%array, sforce_loc)
          else
             call eangle_domdec_loop_gromacs_pd(nangletbl, anglelist, ctb, ctc, xyzq_loc%dp,&
                  scoordtab%dp, 1.0_chm_real, epot_loc, force_loc(tid)%array, sforce_loc)
          endif
       endif
!$omp end parallel
       q_force_loc_used = .true.
#if KEY_BLOCK==1
    endif
#endif

    epot = epot + epot_loc
    sforce = sforce + sforce_loc

    !call write_angle_data(nangletbl, anglelist, ctb, ctc, 'anglelist.txt', 'anglecoef.txt')

    return
  end subroutine eangle_domdec

  ! *
  ! * Writes angle data in to a file
  ! *
  subroutine write_angle_data(n, anglelist, ctb, ctc, filename_list, filename_coef)
    use domdec_bonded_types,only:anglelist_t
    implicit none
    ! Input
    integer, intent(in) :: n
    type(anglelist_t), intent(in) :: anglelist(:)
    real(chm_real), intent(in) :: ctb(:), ctc(:)
    character(*), intent(in) :: filename_list, filename_coef
    ! Variables
    integer i, ncoef

    ncoef = 0
    open(120,file=filename_list)
    do i=1,n
       write (120,'(3i8,3i4)') anglelist(i)%i, anglelist(i)%j, anglelist(i)%k, &
            anglelist(i)%itype, anglelist(i)%ishift1, anglelist(i)%ishift2
       ncoef = max(ncoef, anglelist(i)%itype)
    enddo
    close(120)

    open(120,file=filename_coef)
    do i=1,ncoef
       write (120,'(2e28.20)') ctb(i), ctc(i)
    enddo
    close(120)

    return
  end subroutine write_angle_data

  ! *
  ! * Dihedrals
  ! *
  subroutine edihe_domdec(ip, jp, kp, lp, icp, cpd, cpc, cpsin, cpcos, epot)
    use number,only:zero, one
    use domdec_common,only:q_single
    use domdec_bonded,only:ndihetbl, dihetbl, dihelist, q_dihelist_current, &
         build_dihelist_ps, build_dihelist_pd, realloc_dihelist
    use domdec_local,only:scoordtab, force_loc, q_force_loc_used, sforce, xyzq_loc, glo2loc_ind
    use stream,only:prnlev,outu
#if KEY_BLOCK==1
    use lambdam,only:qmld, qsobo, sobon, qldm_scalecons, msld_get_scale, msld_lambdaforce, mlddihc, mlddihi, mlddihj, mlddihs, biflam, biflam2
    use block_ltm,only:nblock
    use block_fcm,only:qnoph
    use domdec_bonded_block,only:ndihetbl_block, dihetbl_block_pos, dihetbl_block, &
         lower_triangle_index, build_dihelist_block_ps, build_dihelist_block_pd
    use domdec_block,only:biflam_loc, biflam2_loc
#endif
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu, gpu_code_version
#endif
    implicit none
    ! Input / Output
    integer, intent(in) :: ip(:), jp(:), kp(:), lp(:), icp(:), cpd(:)
    real(chm_real), intent(in) :: cpc(:), cpsin(:), cpcos(:)
    real(chm_real), intent(inout) :: epot
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num
#endif
    ! Variables
#if KEY_BLOCK==1
    real(chm_real) fscale, epot_block
    integer ibl, jbl, k
#endif
    integer tid
    integer nlinear
    real(chm_real) epot_loc, sforce_loc(3*27)

    if (.not.q_dihelist_current) then
       call realloc_dihelist(ndihetbl)
#if KEY_DOMDEC_GPU==1
       q_lists_current = .false.
#endif
#if KEY_BLOCK==1
       if (qmld .and. ((.not.qnoph) .or. qsobo .or. qldm_scalecons)) then
          if (q_single) then
             call build_dihelist_block_ps(ndihetbl, dihetbl, ip, jp, kp, lp, icp, &
                  mlddihi, mlddihj, mlddihs, qsobo, &
                  glo2loc_ind, xyzq_loc%sp, dihelist, &
                  ndihetbl_block, dihetbl_block_pos, dihetbl_block)
          else
             call build_dihelist_block_pd(ndihetbl, dihetbl, ip, jp, kp, lp, icp, &
                  mlddihi, mlddihj, mlddihs, qsobo, &
                  glo2loc_ind, xyzq_loc%dp, dihelist, &
                  ndihetbl_block, dihetbl_block_pos, dihetbl_block)
          endif
       else
#endif
          if (q_single) then
             call build_dihelist_ps(ndihetbl, dihetbl, ip, jp, kp, lp, icp, &
                  glo2loc_ind, xyzq_loc%sp, dihelist)
          else
             call build_dihelist_pd(ndihetbl, dihetbl, ip, jp, kp, lp, icp, &
                  glo2loc_ind, xyzq_loc%dp, dihelist)
          endif
#if KEY_BLOCK==1
       endif
#endif
       q_dihelist_current = .true.
    endif

    if (q_single) then
       if (prnlev > 6) write (outu,'(a)') ' edihe_domdec: calling edihe_domdec_loop_ps'
    else
       if (prnlev > 6) write (outu,'(a)') ' edihe_domdec: calling edihe_domdec_loop_pd'
    endif

    sforce_loc = zero
    epot_loc = zero
    nlinear = 0

#if KEY_BLOCK==1
    if (qmld .and. ((.not.qnoph) .or. qsobo .or. qldm_scalecons)) then
       biflam_loc = zero
       biflam2_loc = zero
!$omp parallel private(tid, ibl, jbl, k, fscale, epot_block) &
!$omp&         reduction(+:epot_loc, sforce_loc, nlinear, biflam_loc, biflam2_loc)
#ifdef _OPENMP
       tid = omp_get_thread_num()
#else
       tid = 0
#endif
       do jbl=1,nblock
          do ibl=jbl,nblock
             k = lower_triangle_index(ibl, jbl)
             ! Get correct scaling factor for interaction between blocks ibl and jbl
             call msld_get_scale(ibl, jbl, fscale)
             epot_block = zero
             if (q_single) then
                call edihe_domdec_loop_ps(ndihetbl_block(k), &
                     dihelist(dihetbl_block_pos(k):dihetbl_block_pos(k+1)-1), &
                     cpd, cpc, cpsin, cpcos, xyzq_loc%sp, scoordtab%sp, &
                     real(fscale,kind=chm_real4), epot_block, force_loc(tid)%array, &
                     sforce_loc, nlinear)
             else
                call edihe_domdec_loop_pd(ndihetbl_block(k), &
                     dihelist(dihetbl_block_pos(k):dihetbl_block_pos(k+1)-1), &
                     cpd, cpc, cpsin, cpcos, xyzq_loc%dp, scoordtab%dp, &
                     fscale, epot_block, force_loc(tid)%array, &
                     sforce_loc, nlinear)
             endif
             if (fscale /= one .and. fscale > zero) then
                call msld_lambdaforce(ibl, jbl, epot_block, biflam_loc, biflam2_loc)
             endif
             epot_loc = epot_loc + epot_block*fscale

             ! soft dihedrals
             if (qsobo) then
                k = k + lower_triangle_index(nblock, nblock)
                ! Get correct scaling factor for interaction between blocks ibl and jbl
                call msld_get_scale(ibl, jbl, fscale)
                epot_block = zero
                if (q_single) then
                   call edihe_domdec_loop_ps(ndihetbl_block(k), &
                        dihelist(dihetbl_block_pos(k):dihetbl_block_pos(k+1)-1), &
                        cpd, cpc, cpsin, cpcos, xyzq_loc%sp, scoordtab%sp, &
                        real(fscale**sobon,kind=chm_real4), epot_block, force_loc(tid)%array, &
                        sforce_loc, nlinear)
                else
                   call edihe_domdec_loop_pd(ndihetbl_block(k), &
                        dihelist(dihetbl_block_pos(k):dihetbl_block_pos(k+1)-1), &
                        cpd, cpc, cpsin, cpcos, xyzq_loc%dp, scoordtab%dp, &
                        fscale**sobon, epot_block, force_loc(tid)%array, &
                        sforce_loc, nlinear)
                endif
                if (fscale /= one .and. fscale > zero) then
                   call msld_lambdaforce(ibl, jbl, sobon*(fscale**(sobon-1))*epot_block, biflam_loc, biflam2_loc)
                endif
                epot_loc = epot_loc + epot_block*(fscale**sobon)
             endif
          enddo
       enddo
!$omp end parallel
       biflam = biflam + biflam_loc
       biflam2 = biflam2 + biflam2_loc
       q_force_loc_used = .true.
    else
#endif
#if KEY_DOMDEC_GPU==1
       if (q_gpu .and. gpu_code_version==2) then
          q_calc_dihe = .true.
          return
       endif
#endif
!$omp parallel private(tid) reduction(+:epot_loc, sforce_loc, nlinear)
#ifdef _OPENMP
       tid = omp_get_thread_num()
#else
       tid = 0
#endif
       if (q_single) then
          call edihe_domdec_loop_ps(ndihetbl, dihelist, cpd, cpc, cpsin, cpcos, &
               xyzq_loc%sp, scoordtab%sp, 1.0_chm_real4, epot_loc, force_loc(tid)%array, &
               sforce_loc, nlinear)
       else
          call edihe_domdec_loop_pd(ndihetbl, dihelist, cpd, cpc, cpsin, cpcos, &
               xyzq_loc%dp, scoordtab%dp, 1.0_chm_real, epot_loc, force_loc(tid)%array, &
               sforce_loc, nlinear)
       endif
!$omp end parallel
       q_force_loc_used = .true.
#if KEY_BLOCK==1
    endif
#endif

    epot = epot + epot_loc
    sforce = sforce + sforce_loc

    if (nlinear > 0 .and. wrnlev >= 5) then
       write (outu,'(a,i5,a)') ' edihe_domdec: Warning: ',nlinear,&
            ' dihedrals almost linear, derivatives may be affected'
    endif

    !call write_dihe_data(ndihetbl, dihelist, cpd, cpc, cpsin, cpcos, &
    !     'dihelist.txt', 'dihecoef.txt')

    return
  end subroutine edihe_domdec

  ! *
  ! * Writes dihe data in to a file
  ! *
  subroutine write_dihe_data(n, dihelist, cpd, cpc, cpsin, cpcos, filename_list, filename_coef)
    use domdec_bonded_types,only:dihelist_t
    implicit none
    ! Input
    integer, intent(in) :: n
    type(dihelist_t), intent(in) :: dihelist(:)
    integer, intent(in) :: cpd(:)
    real(chm_real), intent(in) :: cpc(:), cpsin(:), cpcos(:)
    character(*), intent(in) :: filename_list, filename_coef
    ! Variables
    integer i, ncoef

    ncoef = 0
    open(120,file=filename_list)
    do i=1,n
       write (120,'(4i8,4i4)') dihelist(i)%i, dihelist(i)%j, dihelist(i)%k, dihelist(i)%l, &
            dihelist(i)%itype, dihelist(i)%ishift1, dihelist(i)%ishift2, dihelist(i)%ishift3
       ncoef = max(ncoef, dihelist(i)%itype)
    enddo
    close(120)

    open(120,file=filename_coef)
    do i=1,ncoef
       write (120,'(i4,3e28.20)') cpd(i), cpc(i), cpsin(i), cpcos(i)
    enddo
    close(120)

    return
  end subroutine write_dihe_data


  ! *
  ! * Improper dihedrals
  ! *
  subroutine eimdihe_domdec(im, jm, km, lm, ici, cid, cic, cisin, cicos, epot)
    use number,only:zero, one
    use domdec_common,only:q_single
    use domdec_bonded,only:nimdihetbl, imdihetbl, imdihelist, q_imdihelist_current, &
         build_dihelist_ps, build_dihelist_pd, realloc_imdihelist
    use domdec_local,only:scoordtab, force_loc, q_force_loc_used, sforce, xyzq_loc, glo2loc_ind
    use stream,only:prnlev,outu
#if KEY_BLOCK==1
    use lambdam,only:qmld, qsobo, sobon, qldm_scalecons, msld_get_scale, msld_lambdaforce, mldimpc, mldimpi, mldimpj, mldimps, biflam, biflam2
    use block_ltm,only:nblock
    use block_fcm,only:qnoim
    use domdec_bonded_block,only:nimdihetbl_block, imdihetbl_block_pos, imdihetbl_block, &
         lower_triangle_index, build_dihelist_block_ps, build_dihelist_block_pd
    use domdec_block,only:biflam_loc, biflam2_loc
#endif
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu, gpu_code_version
#endif
    implicit none
    ! Input / Output
    integer, intent(in) :: im(:), jm(:), km(:), lm(:), ici(:), cid(:)
    real(chm_real), intent(in) :: cic(:), cisin(:), cicos(:)
    real(chm_real), intent(inout) :: epot
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num
#endif
    ! Variables
#if KEY_BLOCK==1
    real(chm_real) fscale, epot_block
    integer ibl, jbl, k
#endif
    integer tid
    integer nlinear, nbent
    real(chm_real) epot_loc, sforce_loc(3*27)

    if (.not.q_imdihelist_current) then
       call realloc_imdihelist(nimdihetbl)
#if KEY_DOMDEC_GPU==1
       q_lists_current = .false.
#endif
#if KEY_BLOCK==1
       if (qmld .and. ((.not.qnoim) .or. qsobo .or. qldm_scalecons)) then
          if (q_single) then
             call build_dihelist_block_ps(nimdihetbl, imdihetbl, im, jm, km, lm, ici, &
                  mldimpi, mldimpj, mldimps, qsobo, &
                  glo2loc_ind, xyzq_loc%sp, imdihelist, &
                  nimdihetbl_block, imdihetbl_block_pos, imdihetbl_block)
          else
             call build_dihelist_block_pd(nimdihetbl, imdihetbl, im, jm, km, lm, ici, &
                  mldimpi, mldimpj, mldimps, qsobo, &
                  glo2loc_ind, xyzq_loc%dp, imdihelist, &
                  nimdihetbl_block, imdihetbl_block_pos, imdihetbl_block)
          endif
       else
#endif
          if (q_single) then
             call build_dihelist_ps(nimdihetbl, imdihetbl, im, jm, km, lm, ici, &
                  glo2loc_ind, xyzq_loc%sp, imdihelist)
          else
             call build_dihelist_pd(nimdihetbl, imdihetbl, im, jm, km, lm, ici, &
                  glo2loc_ind, xyzq_loc%dp, imdihelist)
          endif
#if KEY_BLOCK==1
       endif
#endif
       q_imdihelist_current = .true.
    endif

    if (q_single) then
       if (prnlev > 6) write (outu,'(a)') ' eimdihe_domdec: calling eimdihe_domdec_loop_ps'
    else
       if (prnlev > 6) write (outu,'(a)') ' eimdihe_domdec: calling eimdihe_domdec_loop_pd'
    endif

    sforce_loc = zero
    epot_loc = zero
    nlinear = 0
    nbent = 0

#if KEY_BLOCK==1
    if (qmld .and. ((.not.qnoim) .or. qsobo .or. qldm_scalecons)) then
       biflam_loc = zero
       biflam2_loc = zero
!$omp parallel private(tid, ibl, jbl, k, fscale, epot_block) &
!$omp&         reduction(+:epot_loc, sforce_loc, nlinear, nbent, biflam_loc, biflam2_loc)
#ifdef _OPENMP
       tid = omp_get_thread_num()
#else
       tid = 0
#endif
       do jbl=1,nblock
          do ibl=jbl,nblock
             k = lower_triangle_index(ibl, jbl)
             ! Get correct scaling factor for interaction between blocks ibl and jbl
             call msld_get_scale(ibl, jbl, fscale)
             if (q_single) then
                call eimdihe_domdec_loop_ps(nimdihetbl_block(k), &
                     imdihelist(imdihetbl_block_pos(k):imdihetbl_block_pos(k+1)-1),&
                     cid, cic, cisin, cicos, xyzq_loc%sp, scoordtab%sp, &
                     real(fscale,kind=chm_real4), epot_block, force_loc(tid)%array, &
                     sforce_loc, nlinear, nbent)
             else
                call eimdihe_domdec_loop_pd(nimdihetbl_block(k), &
                     imdihelist(imdihetbl_block_pos(k):imdihetbl_block_pos(k+1)-1),&
                     cid, cic, cisin, cicos, xyzq_loc%dp, scoordtab%dp, &
                     fscale, epot_block, force_loc(tid)%array, &
                     sforce_loc, nlinear, nbent)
             endif
             if (fscale /= one .and. fscale > zero) then
                call msld_lambdaforce(ibl, jbl, epot_block, biflam_loc, biflam2_loc)
             endif
             epot_loc = epot_loc + epot_block*fscale

             ! soft impropers
             if (qsobo) then
                k = k + lower_triangle_index(nblock, nblock)
                ! Get correct scaling factor for interaction between blocks ibl and jbl
                call msld_get_scale(ibl, jbl, fscale)
                if (q_single) then
                   call eimdihe_domdec_loop_ps(nimdihetbl_block(k), &
                        imdihelist(imdihetbl_block_pos(k):imdihetbl_block_pos(k+1)-1),&
                        cid, cic, cisin, cicos, xyzq_loc%sp, scoordtab%sp, &
                        real(fscale**sobon,kind=chm_real4), epot_block, force_loc(tid)%array, &
                        sforce_loc, nlinear, nbent)
                else
                   call eimdihe_domdec_loop_pd(nimdihetbl_block(k), &
                        imdihelist(imdihetbl_block_pos(k):imdihetbl_block_pos(k+1)-1),&
                        cid, cic, cisin, cicos, xyzq_loc%dp, scoordtab%dp, &
                        fscale**sobon, epot_block, force_loc(tid)%array, &
                        sforce_loc, nlinear, nbent)
                endif
                if (fscale /= one .and. fscale > zero) then
                   call msld_lambdaforce(ibl, jbl, sobon*(fscale**(sobon-1))*epot_block, biflam_loc, biflam2_loc)
                endif
                epot_loc = epot_loc + epot_block*(fscale**sobon)
             endif
          enddo
       enddo
!$omp end parallel
       biflam = biflam + biflam_loc
       biflam2 = biflam2 + biflam2_loc
       q_force_loc_used = .true.
    else
#endif
#if KEY_DOMDEC_GPU==1
       if (q_gpu .and. gpu_code_version==2) then
          q_calc_imdihe = .true.
          return
       endif
#endif
!$omp parallel private(tid) reduction(+:epot_loc, sforce_loc, nlinear, nbent)
#ifdef _OPENMP
       tid = omp_get_thread_num()
#else
       tid = 0
#endif
       if (q_single) then
          call eimdihe_domdec_loop_ps(nimdihetbl, imdihelist, cid, cic, cisin, cicos, &
               xyzq_loc%sp, scoordtab%sp, 1.0_chm_real4, epot_loc, force_loc(tid)%array, &
               sforce_loc, nlinear, nbent)
       else
          call eimdihe_domdec_loop_pd(nimdihetbl, imdihelist, cid, cic, cisin, cicos, &
               xyzq_loc%dp, scoordtab%dp, 1.0_chm_real, epot_loc, force_loc(tid)%array, &
               sforce_loc, nlinear, nbent)
       endif
!$omp end parallel
       q_force_loc_used = .true.
#if KEY_BLOCK==1
    endif
#endif

    epot = epot + epot_loc
    sforce = sforce + sforce_loc

    if (nlinear > 0 .and. wrnlev >= 5) then
       write (outu,'(a,i8,a)') ' eimdihe_domdec: Warning: ',nlinear,&
            ' dihedrals almost linear, derivatives may be affected'
    endif

    if (nbent > 0 .and. wrnlev >= 5) then
       write (outu,'(a,i8,a)') ' eimdihe_domdec: Warning: ',nbent,&
            ' bent improper torsion angles'
    endif

    !call write_dihe_data(nimdihetbl, imdihelist, cid, cic, cisin, cicos, &
    !     'imdihelist.txt', 'imdihecoef.txt')

    return
  end subroutine eimdihe_domdec

!!$  ! *
!!$  ! * CMAP
!!$  ! *
!!$  subroutine ecmap_domdec(i1p, j1p, k1p, l1p, i2p, j2p, k2p, l2p, icpt, cpd, cpc, cpsin, cpcos, epot)
!!$    use number,only:zero, one
!!$    use domdec_common,only:q_single
!!$    use domdec_bonded,only:ncmaptbl, cmaptbl, cmaplist, q_cmaplist_current, &
!!$         build_cmaplist_ps, build_cmaplist_pd
!!$    use domdec_local,only:scoordtab, force_loc, sforce, xyzq_loc, glo2loc_ind
!!$    use stream,only:prnlev,outu
!!$##IF BLOCK
!!$    use lambdam,only:qmld
!!$##ENDIF
!!$    implicit none
!!$    ! Input / Output
!!$    integer, intent(in) :: i1p(:), j1p(:), k1p(:), l1p(:), i2p(:), j2p(:), k2p(:), l2p(:)
!!$    integer, intent(in) :: icp(:), cpd(:)
!!$    real(chm_real), intent(in) :: cpc(:), cpsin(:), cpcos(:)
!!$    real(chm_real), intent(out) :: epot
!!$    ! Functions
!!$    integer omp_get_thread_num  !##OPENMP
!!$    ! Variables
!!$    integer tid
!!$    integer nlinear
!!$    real(chm_real) epot_loc, sforce_loc(3*27)
!!$
!!$    if (.not.q_cmaplist_current) then
!!$       if (q_single) then
!!$          call build_cmaplist_ps(ncmaptbl, cmaptbl, ip, jp, kp, lp, icp, &
!!$               glo2loc_ind, xyzq_loc%sp, cmaplist)
!!$       else
!!$          call build_cmaplist_pd(ncmaptbl, cmaptbl, ip, jp, kp, lp, icp, &
!!$               glo2loc_ind, xyzq_loc%dp, cmaplist)
!!$       endif
!!$       q_cmaplist_current = .true.
!!$    endif
!!$
!!$    if (q_single) then
!!$       if (prnlev > 6) write (outu,'(a)') ' ecmap_domdec: calling ecmap_domdec_loop_ps'
!!$    else
!!$       if (prnlev > 6) write (outu,'(a)') ' ecmap_domdec: calling ecmap_domdec_loop_pd'
!!$    endif
!!$
!!$    epot_loc = zero
!!$    sforce_loc = zero
!!$!$omp parallel private(tid) reduction(+:epot_loc, sforce_loc)
!!$    tid = 0                     !##.not.OPENMP
!!$    tid = omp_get_thread_num()  !##OPENMP
!!$    if (q_single) then
!!$       call ecmap_domdec_loop_ps(ncmaptbl, cmaplist, cpd, cpc, cpsin, cpcos, &
!!$            xyzq_loc%sp, scoordtab%sp, 1.0_chm_real4, epot_loc, force_loc(tid)%array, &
!!$            sforce_loc, nlinear)
!!$    else
!!$       call ecmap_domdec_loop_pd(ncmaptbl, cmaplist, cpd, cpc, cpsin, cpcos, &
!!$            xyzq_loc%dp, scoordtab%dp, 1.0_chm_real, epot_loc, force_loc(tid)%array, &
!!$            sforce_loc, nlinear)
!!$    endif
!!$!$omp end parallel
!!$
!!$    epot = epot + epot_loc
!!$    sforce = sforce + sforce_loc
!!$
!!$    return
!!$  end subroutine ecmap_domdec

  ! *
  ! * 1-4 interactions & exclusions
  ! *
  subroutine e14_domdec(q_vdw, q_ewald, q_ewexcl, &
#if KEY_LJPME==1
                        qljpme, qljexc, eljexc, &
#endif
  vdwpot, coulpot, excoulpot)
    use number,only:zero, one
    use domdec_common,only:q_single, q_gpu, divide_thread_work
    use domdec_bonded,only:nin14tbl, in14tbl, in14list, in14i, in14j, &
         nex14tbl, ex14tbl, ex14list, ex14i, ex14j, &
         build_xx14list_ps, build_xx14list_pd, q_in14list_current, q_ex14list_current, &
         realloc_in14list, realloc_ex14list
    use domdec_local,only:scoordtab, force_loc, q_force_loc_used, sforce, xyzq_loc, glo2loc_ind, &
         vdwtype_loc
    use enbfix14,only:ewald14_ps, ewald14_pd, ewald14_soft_ps, ewald14_soft_pd
    use enbxfast,only:vdwparam14,vdwparam
    use stream,only:prnlev, outu
#if KEY_LJPME==1
    use enbfix14,only:ljpme14_ps, ljpme14_pd
#endif
#if KEY_BLOCK==1
    use lambdam,only:qmld, msld_get_scale, msld_lambdaforce, biflam, biflam2, bixlam, iqldm_pme, iqldm_softcore
    use block_ltm,only:nblock, iblckp
    use domdec_bonded_block,only:nin14tbl_block, in14tbl_block_pos, &
         nex14tbl_block, ex14tbl_block_pos, &
         lower_triangle_index, build_xx14list_block_ps, build_xx14list_block_pd
    use domdec_block,only:biflam_loc, biflam2_loc
#endif
#if KEY_DOMDEC_GPU==1
    use domdec_util_gpu_mod,only:range_start, range_stop, copy_14_list_to_gpu, calc_14_force_gpu
    use domdec_common,only:q_gpu, q_test, q_print_output, boxx, boxy, boxz
    use energym,only:qdyncall
    use reawri,only:qcnstp
#if KEY_BLOCK==1
    use domdec_util_gpu_mod,only:copy_14_block_pos_to_gpu
#endif
#endif
    use nb_module,only:lccnba
    implicit none
    ! Input / Output
    logical, intent(in) :: q_vdw, q_ewald, q_ewexcl
    real(chm_real), intent(inout) :: vdwpot, coulpot, excoulpot
#if KEY_LJPME==1
    logical, intent(in) :: qljpme, qljexc
    real(chm_real), intent(inout) :: eljexc
    real(chm_real) eljexc_loc
#endif
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num
#endif
    ! Variables
#if KEY_BLOCK==1
    real(chm_real) fscale, fscalex, dummy1, dummy2, dummy3, vdwpot_block, coulpot_block, excoulpot_block, dpotdfs_block
    integer ibl, jbl, k
#endif
#if KEY_DOMDEC_GPU==1
    logical q_calc_energy, q_calc_virial
    integer vdwmodel, elecmodel
#endif
    real(chm_real) vdwpot_loc, coulpot_loc, excoulpot_loc, sforce_loc(3*27)
    integer in14_start, in14_end, ex14_start, ex14_end
    integer tid

    if (.not.q_in14list_current) then
       call realloc_in14list(nin14tbl)
#if KEY_BLOCK==1
       if (qmld) then
          if (q_single) then
             call build_xx14list_block_ps(nin14tbl, in14tbl, in14i, in14j, iblckp, &
                  glo2loc_ind, xyzq_loc%sp, in14list, nin14tbl_block, in14tbl_block_pos)
          else
             call build_xx14list_block_pd(nin14tbl, in14tbl, in14i, in14j, iblckp, &
                  glo2loc_ind, xyzq_loc%dp, in14list, nin14tbl_block, in14tbl_block_pos)
          endif
       else
#endif
          if (q_single) then
             call build_xx14list_ps(nin14tbl, in14tbl, in14i, in14j, &
                  glo2loc_ind, xyzq_loc%sp, in14list)
          else
             call build_xx14list_pd(nin14tbl, in14tbl, in14i, in14j, &
                  glo2loc_ind, xyzq_loc%dp, in14list)
          endif
#if KEY_BLOCK==1
       endif
#endif
    endif

    if (.not.q_ex14list_current) then
       call realloc_ex14list(nex14tbl)
#if KEY_BLOCK==1
       if (qmld) then
          if (q_single) then
             call build_xx14list_block_ps(nex14tbl, ex14tbl, ex14i, ex14j, iblckp, &
                  glo2loc_ind, xyzq_loc%sp, ex14list, nex14tbl_block, ex14tbl_block_pos)
          else
             call build_xx14list_block_pd(nex14tbl, ex14tbl, ex14i, ex14j, iblckp, &
                  glo2loc_ind, xyzq_loc%dp, ex14list, nex14tbl_block, ex14tbl_block_pos)
          endif
       else
#endif
          if (q_single) then
             call build_xx14list_ps(nex14tbl, ex14tbl, ex14i, ex14j, &
                  glo2loc_ind, xyzq_loc%sp, ex14list)
          else
             call build_xx14list_pd(nex14tbl, ex14tbl, ex14i, ex14j, &
                  glo2loc_ind, xyzq_loc%dp, ex14list)
          endif
#if KEY_BLOCK==1
       endif
#endif
    endif

    ! Copy 1-4 lists to GPU
#if KEY_DOMDEC_GPU==1
    if (q_gpu .and. (.not.q_in14list_current .or. .not.q_ex14list_current)) then
       call copy_14_list_to_gpu(nin14tbl, nex14tbl, in14list, ex14list)
#if KEY_BLOCK==1
       if (qmld) then
          call copy_14_block_pos_to_gpu(in14tbl_block_pos, ex14tbl_block_pos)
       endif
#endif
    endif
#endif

    ! Mark lists as current
    q_in14list_current = .true.
    q_ex14list_current = .true.

#if KEY_DOMDEC_GPU==1
    if (q_gpu) then
       ! Perform 1-4 computation on GPU
       q_calc_energy = (.not.qdyncall) .or. q_print_output .or. q_test
       q_calc_virial = (.not.qdyncall) .or. q_print_output .or. q_test .or. qcnstp
#if KEY_BLOCK==1
       ! For Lambda-dynamics, we always calculate energies as they're required in biflam/biflam2 computation
       if (qmld) q_calc_energy = .true.
#endif
       call calc_14_force_gpu(q_calc_energy, q_calc_virial, boxx, boxy, boxz)
    else
#endif
       ! Perform 1-4 computation on CPU
       vdwpot_loc = zero
       coulpot_loc = zero
       excoulpot_loc = zero
       sforce_loc = zero
#if KEY_LJPME==1
       eljexc_loc = zero
#endif

#if KEY_BLOCK==1
       if (qmld) then
          biflam_loc = zero
          biflam2_loc = zero
!$omp parallel private(tid, ibl, jbl, k, fscale, fscalex, in14_start, in14_end, ex14_start, ex14_end, &
!$omp&                 vdwpot_block, coulpot_block, excoulpot_block, dpotdfs_block) &
!$omp&         reduction(+:vdwpot_loc, coulpot_loc, excoulpot_loc, sforce_loc, biflam_loc, biflam2_loc)
#ifdef _OPENMP
          tid = omp_get_thread_num()
#else
          tid = 0
#endif
          do jbl=1,nblock
             do ibl=jbl,nblock
                k = lower_triangle_index(ibl, jbl)
                ! Get correct scaling factor for interaction between blocks ibl and jbl
                call msld_get_scale(ibl, jbl, fscale)
                if (ibl==jbl .and. iqldm_pme==2) then
                   fscalex=fscale*fscale
                else if (iqldm_pme<=2) then
                   fscalex=fscale
                else if (iqldm_pme==3) then
                   fscalex=bixlam(ibl)*bixlam(jbl)
                else
                   ! Shouldn't reach this option
                   fscalex=fscale
                endif
                call divide_thread_work(nin14tbl_block(k), in14_start, in14_end)
                call divide_thread_work(nex14tbl_block(k), ex14_start, ex14_end)
                vdwpot_block = zero
                coulpot_block = zero
                excoulpot_block = zero
                dpotdfs_block = zero
                if (iqldm_softcore==2) then
                if (q_single) then
                   call ewald14_soft_ps(q_vdw, q_ewald, q_ewexcl, &
                        in14_start, in14_end, &
                        in14list(in14tbl_block_pos(k):in14tbl_block_pos(k+1)-1), &
                        ex14_start, ex14_end, &
                        ex14list(ex14tbl_block_pos(k):ex14tbl_block_pos(k+1)-1), &
                        xyzq_loc%sp, scoordtab%sp, vdwtype_loc, &
                        vdwparam14%sp, vdwparam%sp, real(fscale,kind=chm_real4), &
                        real(fscalex,kind=chm_real4), &
                        force_loc(tid)%array, sforce_loc, &
                        vdwpot_block, coulpot_block, excoulpot_block, dpotdfs_block)
                else
                   call ewald14_soft_pd(q_vdw, q_ewald, q_ewexcl, &
                        in14_start, in14_end, &
                        in14list(in14tbl_block_pos(k):in14tbl_block_pos(k+1)-1), &
                        ex14_start, ex14_end, &
                        ex14list(ex14tbl_block_pos(k):ex14tbl_block_pos(k+1)-1), &
                        xyzq_loc%dp, scoordtab%dp, vdwtype_loc, &
                        vdwparam14%dp, vdwparam%dp, fscale, &
                        fscalex, &
                        force_loc(tid)%array, sforce_loc, &
                        vdwpot_block, coulpot_block, excoulpot_block, dpotdfs_block)
                endif
                else
                if (q_single) then
                   call ewald14_ps(q_vdw, q_ewald, q_ewexcl, &
                        in14_start, in14_end, &
                        in14list(in14tbl_block_pos(k):in14tbl_block_pos(k+1)-1), &
                        ex14_start, ex14_end, &
                        ex14list(ex14tbl_block_pos(k):ex14tbl_block_pos(k+1)-1), &
                        xyzq_loc%sp, scoordtab%sp, vdwtype_loc, &
                        vdwparam14%sp, vdwparam%sp, real(fscale,kind=chm_real4), &
                        real(fscalex,kind=chm_real4), &
                        force_loc(tid)%array, sforce_loc, &
                        vdwpot_block, coulpot_block, excoulpot_block)
                else
                   call ewald14_pd(q_vdw, q_ewald, q_ewexcl, &
                        in14_start, in14_end, &
                        in14list(in14tbl_block_pos(k):in14tbl_block_pos(k+1)-1), &
                        ex14_start, ex14_end, &
                        ex14list(ex14tbl_block_pos(k):ex14tbl_block_pos(k+1)-1), &
                        xyzq_loc%dp, scoordtab%dp, vdwtype_loc, &
                        vdwparam14%dp, vdwparam%dp, fscale, &
                        fscalex, &
                        force_loc(tid)%array, sforce_loc, &
                        vdwpot_block, coulpot_block, excoulpot_block)
                endif
                endif
                if (fscale /= one .and. fscale > zero) then
                   call msld_lambdaforce(ibl, jbl, vdwpot_block, biflam_loc, biflam2_loc)
                   call msld_lambdaforce(ibl, jbl, coulpot_block, biflam_loc, biflam2_loc)
                   if (ibl==jbl .and. iqldm_pme==2) then
                      call msld_lambdaforce(ibl, jbl, 2*fscale*excoulpot_block, biflam_loc, biflam2_loc)
                   else if (iqldm_pme<=2) then
                      call msld_lambdaforce(ibl, jbl, excoulpot_block, biflam_loc, biflam2_loc)
                   else if (iqldm_pme/=3) then
                      ! shouldn't reach this option
                      call msld_lambdaforce(ibl, jbl, excoulpot_block, biflam_loc, biflam2_loc)
                   endif
                endif
                if (iqldm_pme==3) then
                   if (ibl>1) then
                      biflam_loc(ibl)=biflam_loc(ibl)+bixlam(jbl)*excoulpot_block
                   endif
                   if (jbl>1) then
                      biflam_loc(jbl)=biflam_loc(jbl)+bixlam(ibl)*excoulpot_block
                   endif
                endif
                vdwpot_loc = vdwpot_loc + vdwpot_block*fscale
                coulpot_loc = coulpot_loc + coulpot_block*fscale
                excoulpot_loc = excoulpot_loc + excoulpot_block*fscalex
             enddo
          enddo
!$omp end parallel
          biflam = biflam + biflam_loc
          biflam2 = biflam2 + biflam2_loc
          q_force_loc_used = .true.
       else
#endif
!$omp parallel private(tid, in14_start, in14_end, ex14_start, ex14_end) &
!$omp&         reduction(+:vdwpot_loc, coulpot_loc, excoulpot_loc, sforce_loc)
#ifdef _OPENMP
          tid = omp_get_thread_num()
#else
          tid = 0
#endif
          call divide_thread_work(nin14tbl, in14_start, in14_end)
          call divide_thread_work(nex14tbl, ex14_start, ex14_end)
          if (q_single) then
             call ewald14_ps(q_vdw, q_ewald, q_ewexcl, in14_start, in14_end, in14list, &
                  ex14_start, ex14_end, ex14list, xyzq_loc%sp, scoordtab%sp, vdwtype_loc, &
                  vdwparam14%sp, vdwparam%sp, 1.0_chm_real4, &
                  1.0_chm_real4, &
                  force_loc(tid)%array, sforce_loc, vdwpot_loc, coulpot_loc, excoulpot_loc)
          else
             call ewald14_pd(q_vdw, q_ewald, q_ewexcl, in14_start, in14_end, in14list, &
                  ex14_start, ex14_end, ex14list, xyzq_loc%dp, scoordtab%dp, vdwtype_loc, &
                  vdwparam14%dp, vdwparam%dp, 1.0_chm_real, &
                  1.0_chm_real, &
                  force_loc(tid)%array, sforce_loc, vdwpot_loc, coulpot_loc, excoulpot_loc)
          endif
!$omp end parallel

#if KEY_LJPME==1
!$omp parallel private(tid, in14_start, in14_end, ex14_start, ex14_end) &
!$omp&         reduction(+:eljexc_loc, vdwpot_loc, sforce_loc)
#ifdef _OPENMP
          tid = omp_get_thread_num()
#else
          tid = 0
#endif
          call divide_thread_work(nin14tbl, in14_start, in14_end)
          call divide_thread_work(nex14tbl, ex14_start, ex14_end)
          if (q_single) then
             call ljpme14_ps(qljpme, q_vdw, qljexc, &
                  in14_start, in14_end, in14list, ex14_start, ex14_end, ex14list, xyzq_loc%sp, scoordtab%sp, &
                  vdwtype_loc, vdwparam14%sp, 1.0_chm_real4, &
                  force_loc(tid)%array, sforce_loc, vdwpot_loc, eljexc_loc)
          else
             call ljpme14_pd(qljpme, q_vdw, qljexc, &
                  in14_start, in14_end, in14list, ex14_start, ex14_end, ex14list, xyzq_loc%dp, scoordtab%dp, &
                  vdwtype_loc, vdwparam14%dp, 1.0_chm_real, &
                  force_loc(tid)%array, sforce_loc, vdwpot_loc, eljexc_loc)
          endif
!$omp end parallel
#endif
          q_force_loc_used = .true.
#if KEY_BLOCK==1
       endif
#endif
       
       vdwpot = vdwpot + vdwpot_loc
       coulpot = coulpot + coulpot_loc
       excoulpot = excoulpot + excoulpot_loc
       sforce = sforce + sforce_loc
#if KEY_LJPME==1
       eljexc = eljexc + eljexc_loc
#endif

#if KEY_DOMDEC_GPU==1
    endif
#endif
    
    !call write_14_data(nin14tbl, nex14tbl, lccnba, in14list, ex14list, vdwparam14%dp, &
    !     'in14list.txt', 'ex14list.txt', 'vdwparam14.txt')

    return
  end subroutine e14_domdec

  ! *
  ! * Writes dihe data in to a file
  ! *
  subroutine write_14_data(nin14list, nex14list, nvdwparam14, in14list, ex14list, vdwparam14, &
       filename_in14, filename_ex14, filename_vdw14)
    use domdec_bonded_types,only:list14_t
    implicit none
    ! Input
    integer, intent(in) :: nin14list, nex14list, nvdwparam14
    type(list14_t), intent(in) :: in14list(:), ex14list(:)
    real(chm_real), intent(in) :: vdwparam14(:)
    character(*), intent(in) :: filename_in14, filename_ex14, filename_vdw14
    ! Variables
    integer i

    open(120,file=filename_in14)
    do i=1,nin14list
       write (120,'(4i8,4i4)') in14list(i)%i, in14list(i)%j, in14list(i)%ishift
    enddo
    close(120)

    open(120,file=filename_ex14)
    do i=1,nex14list
       write (120,'(4i8,4i4)') ex14list(i)%i, ex14list(i)%j, ex14list(i)%ishift
    enddo
    close(120)

    open(120,file=filename_vdw14)
    write (120,'(i6)') nvdwparam14
    do i=1,nvdwparam14
       write (120,'(e28.20)') vdwparam14(i)
    enddo
    close(120)

    return
  end subroutine write_14_data

  ! *
  ! * Thole 1-4 exclusions
  ! *
  subroutine thole_ex14_domdec(coulpot)
    use number,only:zero
    use domdec_common,only:q_single, q_gpu, divide_thread_work
    use domdec_bonded,only:ex14tholetbl, nex14tholetbl, ex14tholelist, ex14tholei, ex14tholej, &
         ex14thole_aa, build_ex14tholelist_ps, build_ex14tholelist_pd, q_ex14tholelist_current
    use domdec_local,only:scoordtab, force_loc, q_force_loc_used, sforce, xyzq_loc, glo2loc_ind, &
         vdwtype_loc
    use enbfix14,only:thole14_ps, thole14_pd
#if KEY_BLOCK==1
    use lambdam,only:qmld
#endif
    implicit none
    ! Input / Output
    real(chm_real), intent(inout) :: coulpot
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num
#endif
    ! Variables
    real(chm_real) coulpot_loc, sforce_loc(3*27)
    integer istart, iend
    integer tid

    if (.not.q_ex14tholelist_current) then
#if KEY_BLOCK==1
       if (qmld) then
          call wrndie(-5,'<ebonded_domdec>','BLOCK not implemented for Thole')
          !if (q_single) then
          !call build_ex14list_block_ps(nex14tbl, ex14tbl, ex14i, ex14j, iblckp, &
          !     glo2loc_ind, xyzq_loc%sp, ex14list, nex14tbl_block, ex14tbl_block_pos)
          !else
          !call build_ex14list_block_pd(nex14tbl, ex14tbl, ex14i, ex14j, iblckp, &
          !     glo2loc_ind, xyzq_loc%dp, ex14list, nex14tbl_block, ex14tbl_block_pos)
          !endif
       else
#endif
          if (q_single) then
             call build_ex14tholelist_ps(nex14tholetbl, ex14tholetbl, ex14tholei, ex14tholej, &
                  ex14thole_aa, glo2loc_ind, xyzq_loc%sp, ex14tholelist)
          else
             call build_ex14tholelist_pd(nex14tholetbl, ex14tholetbl, ex14tholei, ex14tholej, &
                  ex14thole_aa, glo2loc_ind, xyzq_loc%dp, ex14tholelist)
          endif
#if KEY_BLOCK==1
       endif
#endif
       q_ex14tholelist_current = .true.
    endif

    coulpot_loc = zero
    sforce_loc = zero

!$omp parallel private(tid, istart, iend) reduction(+:coulpot_loc, sforce_loc)
#ifdef _OPENMP
    tid = omp_get_thread_num()
#else
    tid = 0
#endif
    call divide_thread_work(nex14tholetbl, istart, iend)
    if (q_single) then
       call thole14_ps(istart, iend, ex14tholelist, xyzq_loc%sp, scoordtab%sp, &
            force_loc(tid)%array, sforce_loc, coulpot_loc)
    else
       call thole14_pd(istart, iend, ex14tholelist, xyzq_loc%dp, scoordtab%dp, &
            force_loc(tid)%array, sforce_loc, coulpot_loc)
    endif
!$omp end parallel
    q_force_loc_used = .true.
    
    coulpot = coulpot + coulpot_loc
    sforce = sforce + sforce_loc

    return
  end subroutine thole_ex14_domdec

  !######################################################################################
  !                                         KERNELS
  !######################################################################################

#define BOND_LOOP
#define DOUBLE_PREC
#define KERNEL_NAME ebond_domdec_loop_pd
#include "ebonded_domdec_kernel.inc"
#undef BOND_LOOP
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define BOND_SOFT_LOOP
#define DOUBLE_PREC
#define KERNEL_NAME esbond_domdec_loop_pd
#include "ebonded_domdec_kernel.inc"
#undef BOND_SOFT_LOOP
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define ANGLE_LOOP
#define DOUBLE_PREC
#define CHARMM_STYLE
#define KERNEL_NAME eangle_domdec_loop_charmm_pd
#include "ebonded_domdec_kernel.inc"
#undef ANGLE_LOOP
#undef DOUBLE_PREC
#undef CHARMM_STYLE
#undef KERNEL_NAME

#define ANGLE_LOOP
#define DOUBLE_PREC
#define GROMACS_STYLE
#define KERNEL_NAME eangle_domdec_loop_gromacs_pd
#include "ebonded_domdec_kernel.inc"
#undef ANGLE_LOOP
#undef DOUBLE_PREC
#undef GROMACS_STYLE
#undef KERNEL_NAME

#define ANGLE_LOOP
#define DOUBLE_PREC
#define CHARMM_STYLE
#define GROMACS_STYLE
#define KERNEL_NAME eangle_domdec_loop_charmm_gromacs_pd
#include "ebonded_domdec_kernel.inc"
#undef ANGLE_LOOP
#undef DOUBLE_PREC
#undef CHARMM_STYLE
#undef GROMACS_STYLE
#undef KERNEL_NAME

#define DIHE_LOOP
#define DOUBLE_PREC
#define KERNEL_NAME edihe_domdec_loop_pd
#include "ebonded_domdec_kernel.inc"
#undef DIHE_LOOP
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define IMDIHE_LOOP
#define DOUBLE_PREC
#define KERNEL_NAME eimdihe_domdec_loop_pd
#include "ebonded_domdec_kernel.inc"
#undef IMDIHE_LOOP
#undef DOUBLE_PREC
#undef KERNEL_NAME

!------------- Single precision ----------------

#define BOND_LOOP
#define SINGLE_PREC
#define KERNEL_NAME ebond_domdec_loop_ps
#include "ebonded_domdec_kernel.inc"
#undef BOND_LOOP
#undef SINGLE_PREC
#undef KERNEL_NAME

#define BOND_SOFT_LOOP
#define SINGLE_PREC
#define KERNEL_NAME esbond_domdec_loop_ps
#include "ebonded_domdec_kernel.inc"
#undef BOND_SOFT_LOOP
#undef SINGLE_PREC
#undef KERNEL_NAME

#define ANGLE_LOOP
#define SINGLE_PREC
#define CHARMM_STYLE
#define KERNEL_NAME eangle_domdec_loop_charmm_ps
#include "ebonded_domdec_kernel.inc"
#undef ANGLE_LOOP
#undef SINGLE_PREC
#undef CHARMM_STYLE
#undef KERNEL_NAME

#define ANGLE_LOOP
#define SINGLE_PREC
#define GROMACS_STYLE
#define KERNEL_NAME eangle_domdec_loop_gromacs_ps
#include "ebonded_domdec_kernel.inc"
#undef ANGLE_LOOP
#undef SINGLE_PREC
#undef GROMACS_STYLE
#undef KERNEL_NAME

#define ANGLE_LOOP
#define SINGLE_PREC
#define CHARMM_STYLE
#define GROMACS_STYLE
#define KERNEL_NAME eangle_domdec_loop_charmm_gromacs_ps
#include "ebonded_domdec_kernel.inc"
#undef ANGLE_LOOP
#undef SINGLE_PREC
#undef CHARMM_STYLE
#undef GROMACS_STYLE
#undef KERNEL_NAME

#define DIHE_LOOP
#define SINGLE_PREC
#define KERNEL_NAME edihe_domdec_loop_ps
#include "ebonded_domdec_kernel.inc"
#undef DIHE_LOOP
#undef SINGLE_PREC
#undef KERNEL_NAME

#define IMDIHE_LOOP
#define SINGLE_PREC
#define KERNEL_NAME eimdihe_domdec_loop_ps
#include "ebonded_domdec_kernel.inc"
#undef IMDIHE_LOOP
#undef SINGLE_PREC
#undef KERNEL_NAME

#endif /* domdec_main */


end module ebonded_domdec
