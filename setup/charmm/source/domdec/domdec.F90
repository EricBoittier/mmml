module domdec


  ! *
  ! * Domain decomposition initialization routines
  ! *
#if KEY_DOMDEC==1 /*domdec_main*/
  use, intrinsic :: iso_c_binding
  use chm_kinds
  use dimens_fcm
  implicit none
  private

  logical q_split_set, q_split_new

#if KEY_DOMDEC_GPU==1
  ! GPU id that is set with "GPUId" option and can be used to override default GPU selection
  integer :: gpuid = -999
#endif
  
  ! Public subroutines
  public domdec_com, allocate_domdec, build_groupl, init_domdec1, init_domdec2

  interface
    subroutine get_simd_version(simd_compile, simd_cpu) bind(C)
      import
      integer(c_int), intent(out) :: simd_compile, simd_cpu
    end subroutine get_simd_version

    subroutine get_vendor_id(vendor_id) bind(C)
      import
      integer(c_int), intent(out) :: vendor_id
    end subroutine get_vendor_id
 end interface

contains

  ! *
  ! * Checks SIMD version
  ! *
  subroutine domdec_check_simd()
    use domdec_common,only:simd_version, SIMD_NONE, SIMD_SSE, &
         SIMD_AVX, SIMD_AVX2, SIMD_AVX_FMA, SIMD_AVX2_FMA, &
         cpu_vendor, CPU_UNKNOWN, CPU_INTEL, CPU_AMD
    implicit none
    integer(c_int) :: simd_compile, simd_cpu, vendor_id
    call get_simd_version(simd_compile, simd_cpu)
    if (simd_compile >= 20 .and. simd_cpu >= 20) then
       call wrndie(-5,'<domdec>','Intel MIC not supported')
    elseif (simd_compile >= 6 .and. simd_cpu >= 9) then
       simd_version = SIMD_AVX2_FMA
    elseif (simd_compile >= 6 .and. simd_cpu >= 8) then
       simd_version = SIMD_AVX2
    elseif (simd_compile >= 6 .and. simd_cpu >= 7) then
       simd_version = SIMD_AVX_FMA
    elseif (simd_compile >= 6 .and. simd_cpu >= 6) then
       simd_version = SIMD_AVX
    elseif (simd_compile >= 2 .and. simd_cpu >= 2) then
       simd_version = SIMD_SSE
    endif
    if (simd_cpu >= 6 .and. simd_compile < 6) then
       call wrndie(4,'<domdec>','CPU supports AVX, compile with AVX support for performance gain')
    elseif (simd_cpu >= 2 .and. simd_compile < 2) then
       call wrndie(4,'<domdec>','CPU supports SSE, compile with SSE support for performance gain')
    endif
    call get_vendor_id(vendor_id)
    if (vendor_id == 0) then
       cpu_vendor = CPU_UNKNOWN
    elseif (vendor_id == 1) then
       cpu_vendor = CPU_INTEL
    elseif (vendor_id == 2) then
       cpu_vendor = CPU_AMD
    endif
  end subroutine domdec_check_simd

  ! *
  ! * Parse Doman Decomposition command line
  ! *
  subroutine domdec_com(comlyn, comlen)
#if KEY_MMFF==1
    use ffieldm, only: ffield, mmff
#endif
    use number
    use string,only:nexta4, nexti, curra4
    use stream
    use parallel,only:numnod,mynod,comm_charmm
    use coord,only:x,y,z
    use fast,only:lfast,faster
    use domdec_common,only:q_domdec, nx, ny, nz, ppang, q_sort_groups, q_test, q_single, &
         q_gpu, simd_version
    use domdec_dlb,only:q_load_balance
    use domdec_dr_common,only:ndir_set
#if KEY_DOMDEC_GPU==1
    use domdec_gpu_mod,only:stop_device, stop_mdsim, is_device_on
    use domdec_local,only:uninit_pinned_memory
    use domdec_common,only:gpu_code_version
    use nblist_tilex,only:uninit_pinned_nblist_tilex
#endif /* domdec_gpu */
    implicit none
    ! Input / Output
    character(len=*) comlyn
    integer comlen
    ! Variables
    character(len=4) wrd
    integer i
#if KEY_DOMDEC_GPU==1
    logical q_device_on
    integer gpuid_new
#endif

#if KEY_MMFF==1
    if (ffield .eq. mmff) call wrndie(-2, '<domdec.src>', 'DOMDEC is incompatible with MMFF')
#endif

    q_split_set = .false.
    q_test = .false.
    ndir_set = .false.
    q_domdec = .true.

    do while (len(comlyn) > 0 .and. comlen > 0)
       wrd = curra4(comlyn,comlen)
       if (wrd == 'DLB ') then
          wrd = nexta4(comlyn,comlen)   ! Remove the command read by curra4() above
          wrd = nexta4(comlyn,comlen)
          if (wrd == 'ON  ') then
             q_load_balance = .true.
          elseif (wrd == 'OFF ') then
             q_load_balance = .false.
          else
             call wrndie(-5,'<domdec.src>','Invalid value for DLB, must be ON or OFF')
          endif
       elseif (wrd == 'OFF ') then
          wrd = nexta4(comlyn,comlen)   ! Remove the command read by curra4() above
          q_domdec = .false.
       elseif (wrd == 'PPAN') then
          wrd = nexta4(comlyn,comlen)   ! Remove the command read by curra4() above
          ppang = nexti(comlyn,comlen)
       elseif (wrd == 'NDIR') then
          wrd = nexta4(comlyn,comlen)   ! Remove the command read by curra4() above
          nx = nexti(comlyn,comlen)
          ny = nexti(comlyn,comlen)
          nz = nexti(comlyn,comlen)
          ndir_set = .true.
       elseif (wrd == 'SING') then
          wrd = nexta4(comlyn,comlen)   ! Remove the command read by curra4() above
          q_single = .true.
       elseif (wrd == 'DOUB') then
          wrd = nexta4(comlyn,comlen)   ! Remove the command read by curra4() above
          q_single = .false.
       elseif (wrd == 'SORT') then
          wrd = nexta4(comlyn,comlen)   ! Remove the command read by curra4() above
          q_sort_groups = .true.
       elseif (wrd == 'TEST') then
          wrd = nexta4(comlyn,comlen)   ! Remove the command read by curra4() above
          q_test = .true.
       elseif (wrd == 'SPLI') then
          wrd = nexta4(comlyn,comlen)   ! Remove the command read by curra4() above
          wrd = nexta4(comlyn,comlen)
          if (wrd == 'ON  ') then
             if (numnod > 1) then
                q_split_new = .true.
                q_split_set = .true.
             else
                ! We can't set split = ON because there's only a single node, throw a warning
                call wrndie(5,'<domdec.src>','Cannot set SPLIT ON when using a single MPI node')
             endif
          elseif (wrd == 'OFF ') then
             q_split_new = .false.
             q_split_set = .true.
          else
             call wrndie(-5,'<domdec.src>','Invalid value for parameter SPLIt, must be ON or OFF')
          endif
#if KEY_DOMDEC_GPU==1
       elseif (wrd == 'GPUI') then
          wrd = nexta4(comlyn,comlen)   ! Remove the command read by curra4() above
          gpuid_new = nexti(comlyn,comlen)
          if (gpuid_new < -2) gpuid_new = -2
          call is_device_on(q_device_on)
          if (q_device_on .and. gpuid_new /= gpuid) then
             ! GPU ID has changed, warn the user that he/she needs to turn DOMDEC_GPU off before any
             ! ID change has effect
             call wrndie(2,'<domdec.src>','GPUId setting change only has effect after GPU OFF/GPU ON cycle')
          endif
          gpuid = gpuid_new
#endif
       elseif (wrd == 'GPU ') then
#if KEY_DOMDEC_GPU==0
          call wrndie(-5,'<domdec.src>',&
               'In order to use the GPU option, code must be compiled with DOMDEC_GPU')
#endif
          wrd = nexta4(comlyn,comlen)   ! Remove the command read by curra4() above
          wrd = nexta4(comlyn,comlen)
          if (wrd == 'ON  ') then
             q_load_balance = .false.   ! Switch off DLB
             q_gpu = .true.
             q_single = .true.
#if KEY_DOMDEC_GPU==1
             gpu_code_version = 1
#endif
          elseif (wrd == 'ONLY') then
             q_load_balance = .false.   ! Switch off DLB
             q_gpu = .true.
             q_single = .true.
#if KEY_DOMDEC_GPU==1
             gpu_code_version = 2
#endif
          elseif (wrd == 'OFF ') then
#if KEY_DOMDEC_GPU==1
             if (q_gpu) then
                ! Turn off GPU computation
                call uninit_pinned_memory()
                call uninit_pinned_nblist_tilex()
                call stop_mdsim()
                call stop_device()
             endif
             gpu_code_version = 1
#endif
             q_gpu = .false.
             q_single = .false.
          else
             call wrndie(-5,'<domdec.src>','Invalid value for parameter GPU, must be ON or OFF')
          endif
       else
          ! This is not a valid DOMDEC command => exit parsing
          exit
       endif
    enddo

#if KEY_DOMDEC_GPU==1
    if (q_gpu .and. .not.q_single) then
       call wrndie(-5,'<domdec>','Cannot use DOUBle option with DOMDEC GPU')
    endif
#endif
    
    if (ppang <= 0) then
       call wrndie(-5,'<domdec>','PPANg parameter must be positive integer')
    endif

    if (.not.q_domdec) then
       call uninit_domdec()
    endif

    return
  end subroutine domdec_com

  ! *
  ! * Initializes domdec. Called from gener/update.src
  ! * (x, y, z) is the coordinate set. Must be the same on all cores!
  ! *
  subroutine init_domdec1(x, y, z)
    use stream,only:outu, prnlev
    use number,only:one
    use groupxfast,only:make_groups
    use pme_module,only:qpme
#if KEY_LJPME==1
    use psf, only: ljc6
    use pmeutil,only:qljpme, nfft1, nfft2, nfft3, forder, dfftx, dffty, dfftz, dorder
#endif
    use enbxfast,only:init_nblist, test_build_pftable
    use domdec_d2d_comm,only:init_d2d_comm, transfer_coord
    use domdec_dr_common,only:q_direct_node, nrecip, ndir_set, determine_direct_split, &
         init_direct_recip, start_split_direct_recip, stop_split_direct_recip
    use domdec_common,only:nx, ny, nz, nthread, frx, fry, frz, q_ortho, ndirect, set_box, &
         q_test, q_gpu, simd_version, SIMD_NONE, SIMD_SSE, SIMD_AVX, SIMD_AVX_FMA, SIMD_AVX2, &
         SIMD_AVX2_FMA, cpu_vendor, CPU_UNKNOWN, CPU_INTEL, CPU_AMD, calc_min_nx_ny_nz
    use image,only:xucell,xtlabc
    use parallel,only:numnod,mynod
    use nblist_util,only:test_cumsum, test_concat_buffer, test_bucket_sort
    use domdec_local,only:build_local_coord, build_local_vdwtype
    use psf,only:cg
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu, gpu_code_version, boxx, boxy, boxz
    use domdec_dr_common,only:q_recip_node, q_direct_node
    use domdec_gpu_mod,only:set_q_test_gpu, start_device, start_mdsim, stop_mdsim
    use pmeutil,only:nfft1, nfft2, nfft3, forder
    use ewald_1m,only:kappa
    use param,only:cbb, cbc, ctb, ctc, ctub, ctuc, cpd, cpc, cpsin, cpcos, cid, cic, cisin, cicos,&
         ncb, nct, ncp, nci
    use psf,only:natom
    use bonded_gpu_mod,only:setup_bonded_coef_gpu
    use bases_fcm,only:bnbnd
    use enb_core_gpu_mod,only:set_box_size_gpu
#if KEY_BLOCK==1
    use lambdam,only:qmld, isitemld, iqldm_softcore, iqldm_pme
    use block_ltm,only:nblock
    use domdec_util_gpu_mod,only:copy_isitemld_to_gpu
#endif
#endif
#if KEY_LONEPAIR
    use domdec_lonepair,only:init_lonepr, q_lonepr, setup_lonepr_comm
    use lonepr,only:numlp, lpnhost, lphptr, lphost
#endif
    use nblist_builder,only:init_nsxfast
    use colfft_util,only:colfft_util_init
    use reawri,only:iseed
    implicit none
    ! Input
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    ! Variables
    integer min_nx, min_ny, min_nz
    character(64) cpustr
#if KEY_DOMDEC_GPU==1
    logical q_recip_on_gpu, q_bonded_on_gpu, q_nblist_on_gpu
#if KEY_BLOCK==1
    integer numblock
    integer iqldm_use_softcore
    integer iqldm_use_pme
#endif
#endif
    
    call init_direct_recip(qpme, q_split_set, q_split_new)
    q_split_set = .false.
    
    if(( xucell(4) == 90._chm_real) .and. &
         ( xucell(5) == 90._chm_real) .and. &
         ( xucell(6) == 90._chm_real) ) then
       q_ortho = .true.
       call set_box()
    else
       q_ortho = .false.
       call calc_latvec()
       call wrndie(-5,"<domdec> domdec_com: non-orthorhombic", &
            "Current domdec implementation only supports orhtorhombic boxes")
    endif

    ! NOTE: both recip and direct nodes have to call make_groups!
    call make_groups(x, y, z)

    ! Get the minimum number of nodes for each direction
    call calc_min_nx_ny_nz(min_nx, min_ny, min_nz)

    ! No NDIR set, set nx, ny, nz
    if (.not.ndir_set) then
       call determine_direct_split(min_nx, min_ny, min_nz, ndirect, nx, ny, nz)
       ndir_set = .true.
    endif

    if (nx > 1 .and. nx < min_nx) then
       write (outu,'(a)') 'Try using more MPI nodes or set NDIR manually:'
       write (outu,'(a,i3,a)') 'x-direction must have (a) 1 node or (b) at least ',min_nx,' nodes'
       call wrndie(-5,'<domdec>','DOMDEC node number limitation violated')
    endif

    if (ny > 1 .and. ny < min_ny) then
       write (outu,'(a)') 'Try using more MPI nodes or set NDIR manually:'
       write (outu,'(a,i3,a)') 'y-direction must have (a) 1 node or (b) at least ',min_ny,' nodes'
       call wrndie(-5,'<domdec>','DOMDEC node number limitation violated')
    endif

    if (nz > 1 .and. nz < min_nz) then
       write (outu,'(a)') 'Try using more MPI nodes or set NDIR manually:'
       write (outu,'(a,i3,a)') 'z-direction must have (a) 1 node or (b) at least ',min_nz,' nodes'
       call wrndie(-5,'<domdec>','DOMDEC node number limitation violated')
    endif

    !---------------------------------------------------
#if KEY_DOMDEC_GPU==1
    if (q_gpu) then
       ! Turn on GPU computation
       ! NOTE: At this point we know which node is direct vs. recip
       !       (this was determined in "init_direct_recip")
       call start_device(prnlev, mynod, numnod, gpuid)
       q_recip_on_gpu = qpme .and. (gpu_code_version==2)
       q_bonded_on_gpu = (gpu_code_version==2)
       q_nblist_on_gpu = (gpu_code_version==2)
       numblock = 0
       iqldm_use_softcore = 0
       iqldm_use_pme = 0
#if KEY_BLOCK==1
       if (qmld) numblock = nblock
       if (qmld) iqldm_use_softcore = iqldm_softcore
       if (qmld) iqldm_use_pme = iqldm_pme
#endif
       call stop_mdsim()
       call start_mdsim(ndirect, q_recip_on_gpu, q_bonded_on_gpu, q_nblist_on_gpu, &
            q_direct_node, q_recip_node, nx, ny, nz, natom, bnbnd%iblo14, bnbnd%inb14, &
            nfft1, nfft2, nfft3, forder, kappa, numblock, iqldm_use_softcore, iqldm_use_pme, iseed)
       call set_box_size_gpu(boxx, boxy, boxz)
       if (q_direct_node .and. q_bonded_on_gpu) then
          ! NOTE: CMAP is not yet supported. CMAP is computed using the old CHARMM subroutines
          call setup_bonded_coef_gpu(ncb, cbb, cbc, nct, ctb, ctc, nct, ctub, ctuc, &
               ncp, cpd, cpc, cpsin, cpcos, nci, cid, cic, cisin, cicos)
       endif
       ! Set q_test -flag for the GPU code
       call set_q_test_gpu(q_test)
#if KEY_BLOCK==1
       if (qmld) then
          ! Set isitemld on GPU
          call copy_isitemld_to_gpu(isitemld)
       endif
#endif
    endif
#endif
    !---------------------------------------------------
    
    ! Test cumulative summation routines
    if (q_test) then
       call test_cumsum()
       call test_concat_buffer()
       call test_bucket_sort()
    endif

    call domdec_check_simd()
    if (prnlev > 2) then
       if (cpu_vendor == CPU_UNKNOWN) then
          write (cpustr,'(a)') 'Unkown CPU'
       elseif (cpu_vendor == CPU_INTEL) then
          write (cpustr,'(a)') 'Intel CPU'
       elseif (cpu_vendor == CPU_AMD) then
          write (cpustr,'(a)') 'AMD CPU'
       endif
#if KEY_DOMDEC_GPU==1
       if (q_gpu) then
          if (gpu_code_version == 2) then
             if (simd_version == SIMD_SSE .or. simd_version == SIMD_AVX_FMA .or. &
                  simd_version == SIMD_AVX .or. simd_version == SIMD_AVX2_FMA .or. &
                  simd_version == SIMD_AVX2) then
                write (outu,'(a,a)') trim(cpustr),&
                     ' | Using CUDA for most force computations and SSE elsewhere'
             else
                write (outu,'(a,a)') trim(cpustr),&
                     ' | Using CUDA for most force computations and Fortran elsewhere'
             endif
          else
             if (simd_version == SIMD_SSE .or. simd_version == SIMD_AVX_FMA .or. &
                  simd_version == SIMD_AVX .or. simd_version == SIMD_AVX2_FMA .or. &
                  simd_version == SIMD_AVX2) then
                write (outu,'(a,a)') trim(cpustr),&
                     ' | Using CUDA version of non-bonded force loops and SSE elsewhere'
             else
                write (outu,'(a,a)') trim(cpustr),&
                     ' | Using CUDA version of non-bonded force loops and Fortran elsewhere'
             endif
          endif
       else
#endif
          if (simd_version == SIMD_SSE) then
             write (outu,'(a,a)') trim(cpustr),' | Using SSE version of non-bonded force loops'
          elseif (simd_version == SIMD_AVX_FMA) then
             write (outu,'(a,a)') trim(cpustr),&
                  ' | Using AVX version of non-bonded force loops (FMA)'
          elseif (simd_version == SIMD_AVX) then
             write (outu,'(a,a)') trim(cpustr),' | Using AVX version of non-bonded force loops'
          elseif (simd_version == SIMD_AVX2_FMA) then
             write (outu,'(a,a)') trim(cpustr),&
                  ' | Using AVX version of non-bonded force loops (AVX2+FMA)'
          elseif (simd_version == SIMD_AVX2) then
             write (outu,'(a,a)') trim(cpustr),&
                  ' | Using AVX version of non-bonded force loops (AVX2)'
          else
             write (outu,'(a,a)') trim(cpustr),' | Using Fortran version of non-bonded force loops'
          endif
       endif
#if KEY_DOMDEC_GPU==1
    endif
#endif
           
    if (q_direct_node) then
       ! Initialize nonbonded list, calculate lookup tables
       call init_nblist(ndirect)
    endif

    ! Run lookup table unit tests
    if (q_test) then
       call test_build_pftable()
    endif
    
    if (prnlev > 2) then
       write (outu,'(a,3i3)') 'Initializing DOMDEC with NDIR = ',nx,ny,nz
    endif
       
#ifdef _OPENMP
    if (prnlev > 2) then
       write (outu,'(a,i3)') 'Number of threads per MPI node = ',nthread
    endif
#endif
       
    ! Determine fractional sub-box sizes
    frx = one / real(nx)
    fry = one / real(ny)
    frz = one / real(nz)

    ! Allocate memory for domdec
    call allocate_domdec()
              
    ! Initialize Direct-Direct communications
    call init_d2d_comm(x, y, z)

    if (q_direct_node) then
       ! Build groupl
       ! NOTE: Assumes all nodes have all coordinates!
       call build_groupl(x, y, z)
       ! Transfer coordinates among nodes
       call start_split_direct_recip()
       call transfer_coord(x, y, z, .true.)
       call build_local_coord(x, y, z, cg&
#if KEY_LJPME==1
         ,ljc6 &
#endif
       )
       call build_local_vdwtype()
       ! Initializes lone pairs
#if KEY_LONEPAIR
       call init_lonepr(numlp, lpnhost, lphptr, lphost)
       if (q_lonepr .and. q_direct_node) then
          call setup_lonepr_comm(numlp, lpnhost, lphptr, lphost)
       endif
#endif
       ! Initialize non-bonded builder
       call init_nsxfast()
       call stop_split_direct_recip
    endif

    ! Initialize reciprocal calculation (i.e. column FFT)
    if (qpme) then
#if KEY_DOMDEC_GPU==1
       if (.not.q_gpu .or. (q_gpu .and. .not.q_recip_on_gpu)) then
#endif
          call colfft_util_init(nrecip)

#if KEY_DOMDEC_GPU==1
       endif
#endif
    endif

    return
  end subroutine init_domdec1

  ! *
  ! * Initializes domdec. Called from gener/update.src
  ! *
  subroutine init_domdec2()
    use domdec_dr_common,only:q_direct_node
    use domdec_bonded,only:init_bonded
    use domdec_grouped,only:init_grouped, build_groupedtbl
    use domdec_aniso,only:init_aniso
#if KEY_BLOCK==1
    use lambdam,only:qmld
    use domdec_bonded_block,only:init_bonded_block
    use domdec_block,only:init_block
#endif
    use enbxfast,only:init_vdwparam
    use domdec_local,only:build_local_vdwtype
    use domdec_cons,only:init_cons
    use domdec_shake,only:init_shake, build_shaketbl, q_shake
    use fstshk,only:nsh1, nsh2, nsh3, bshkgp, nstwat, numwater, hmassi, hmassj, ammi
    use shake,only:qfshake, qshake, shkapr, constr
    use mpi,only:mpi_logical, mpi_lor, mpi_success
    use parallel,only:comm_charmm
    implicit none
    integer ierror
    logical qfshake_all, qshake_all

    ! Initialize constraints / restraints
    call init_cons()

    ! ------------------------------------------------------------------------------------------
    ! Initialize grouped atoms (anything where atoms are needed as a group on a single node):
    ! -bonds, angles, dihedrals, etc.
    ! -lone pairs
    ! -anisotropy
    ! ------------------------------------------------------------------------------------------
    ! Initializes bonded interactions
    call init_bonded()

    call init_aniso()

    call init_grouped()
    ! ------------------------------------------------------------------------------------------

    ! Initialize shake, but first combine qfshake with OR (this is done because OLD CHARMM
    ! can have different values for qfshake for different nodes if the number of atoms is
    ! small

    call mpi_allreduce(qfshake, qfshake_all, 1, mpi_logical, mpi_lor, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec>','init_domdec2: Error calling mpi_allreduce (1)')
    endif

    call mpi_allreduce(qshake, qshake_all, 1, mpi_logical, mpi_lor, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec>','init_domdec2: Error calling mpi_allreduce (2)')
    endif

    if (qfshake_all) then
       q_shake = .true.
       call init_shake(nsh1, nsh2, nsh3, nstwat, numwater, bshkgp, shkapr, &
            hmassi, hmassj, ammi, constr)
    else if (qshake_all) then
       call wrndie(-5,'<domdec>','DOMDEC only supports fast SHAKE')
    else
       q_shake = .false.
    endif
    
#if KEY_BLOCK==1
    if (qmld) then
       ! Initialize domdec block arrays
       call init_block()
       ! Initializes bonded block
       call init_bonded_block()
    endif
#endif

    if (q_direct_node) then
       ! Build grouped lists
       call build_groupedtbl()

       ! Build shake lists
       call build_shaketbl()
    endif

    if (q_direct_node) then
       ! ccnba and ccnbb have changed => update domdec vdwparam
       call init_vdwparam(.true.)
       ! iacnb has changed => redo domdec vdwtype
       call build_local_vdwtype()
    endif

    return
  end subroutine init_domdec2

  ! *
  ! * Uninitializes domdec
  ! *
  subroutine uninit_domdec
    use domdec_d2d_comm,only:uninit_d2d_comm
    use domdec_dr_common,only:uninit_dr_common
    use domdec_d2r_comm,only:uninit_d2r_comm
    use domdec_r2d_comm,only:uninit_r2d_comm
    use domdec_bonded,only:uninit_bonded
    use domdec_shake,only:uninit_shake
    use domdec_cons,only:uninit_cons
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu
    use domdec_gpu_mod,only:stop_device,stop_mdsim
    use domdec_local,only:uninit_pinned_memory
#endif
#if KEY_BLOCK==1
    use domdec_bonded_block,only:uninit_bonded_block
    use domdec_block,only:uninit_block
#endif
    use domdec_local,only:uninit_local_coord, uninit_local_vdwtype
    use nblist_builder,only:uninit_nsxfast
    use enbxfast,only:uninit_vdwparam, uninit_nblist
    use domdec_grouped,only:uninit_grouped
#if KEY_LONEPAIR==1
    use domdec_lonepair,only:uninit_lonepr
#endif
    use domdec_aniso,only:uninit_aniso
    implicit none

    call deallocate_domdec()

    ! Bonded
    call uninit_bonded()

    ! Lone pairs
#if KEY_LONEPAIR==1
    call uninit_lonepr()
#endif

    ! anisotropy
    call uninit_aniso()

    ! BLOCK
#if KEY_BLOCK==1
    call uninit_block()
    call uninit_bonded_block()
#endif

    ! Grouped
    call uninit_grouped()

    ! SHAKE
    call uninit_shake()

    ! Direct - Direct Communication
    call uninit_d2d_comm()

    ! Direct - Recip Communication
    call uninit_dr_common()

    ! Direct - Recip Communication
    call uninit_d2r_comm()

    ! Direct - Recip Communication
    call uninit_r2d_comm()

    ! CONS
    call uninit_cons()
    
    ! Local coordinate and vdwtype
    call uninit_local_coord()
    call uninit_local_vdwtype()

    ! Non-bonded Neighborlist builder
    call uninit_nsxfast()

    ! Non-bonded force calculation
    call uninit_nblist()
    call uninit_vdwparam()

    ! Turn off GPU computation
#if KEY_DOMDEC_GPU==1
    if (q_gpu) then
       call uninit_pinned_memory()
       call stop_mdsim()
       call stop_device()
       q_gpu = .false.
    endif
#endif

    return
  end subroutine uninit_domdec

  ! *
  ! * Deallocate memory for Domain Decomposition
  ! *
  subroutine deallocate_domdec()
    use psf,only:natom
    use memory,only:chmdealloc
    use domdec_d2d_comm,only:deallocate_copy
    use domdec_common,only:groupl, atoml, homezone, nodeind, q_cons_node
    implicit none

    q_cons_node = .false.

    ! Deallocate buffers for copy_to_all -routines
    ! NOTE: allocate_copy was called from one of copy_to_all/copy_to_root routines
    call deallocate_copy()

    if (allocated(groupl)) then
       call chmdealloc('domdec.src','deallocate_domdec','groupl',size(groupl),intg=groupl)
    endif

    if (allocated(atoml)) then
       call chmdealloc('domdec.src','deallocate_domdec','atoml',size(atoml),intg=atoml)
    endif

    if (allocated(homezone)) then
       call chmdealloc('domdec.src','deallocate_domdec','homezone',size(homezone),intg=homezone)
    endif

    if (allocated(nodeind)) then
       call chmdealloc('domdec.src','deallocate_domdec','nodeind',&
            size(nodeind,1),size(nodeind,2),size(nodeind,3),intg=nodeind)
    endif

    return
  end subroutine deallocate_domdec

  ! *
  ! * Allocate memory for Domain Decomposition
  ! *
  subroutine allocate_domdec()
    use psf,only:natom
    use groupxfast,only:ngroup
    use number,only:two
    use memory,only:chmalloc,chmrealloc
    use stream,only:outu
    use parallel,only:mynod,numnod
    use domdec_common,only:atoml, groupl, homezone, nx, ny, nz, nodeind, homeix, homeiy, homeiz, &
         nneigh, neighlist, nodeindfunc, neighind
    use domdec_d2d_comm,only:calc_max_atom_group
    use domdec_dr_common,only:mynod_split
    implicit none
    ! Variables
    integer i, j, k, l
    integer ix, iy, iz
    integer ixt, iyt, izt
    integer nxt, nyt, nzt
    integer ixl, iyl, izl
    integer ixh, iyh, izh
    integer, allocatable, dimension(:) :: tmpbuf
    real(chm_real) tmp(7,8)
    integer oldtypes(0:1), blockcounts(0:1), offsets(0:1), extent
    integer is, iq
    integer max_atom_box, max_grp_box
    logical q_realloc

    call calc_max_atom_group(max_atom_box, max_grp_box)

    ! If atoml is already allocated => do reallocation, otherwise do allocation
    q_realloc = allocated(atoml)

    if (q_realloc) then
       call chmrealloc('domdec.src','allocate_domdec','atoml',max_atom_box,intg=atoml)
       call chmrealloc('domdec.src','allocate_domdec','groupl',max_grp_box,intg=groupl)
    else
       call chmalloc('domdec.src','allocate_domdec','atoml',max_atom_box,intg=atoml)
       call chmalloc('domdec.src','allocate_domdec','groupl',max_grp_box,intg=groupl)
    endif

    if (q_realloc) then
       call chmrealloc('domdec.src','allocate_domdec','homezone',natom,intg=homezone)
    else
       call chmalloc('domdec.src','allocate_domdec','homezone',natom,intg=homezone)
    endif

    ! Initializes the nodeind that is used in the nodeindfunc function
    ! Also, initializes the home box index (homeix, homeiy, homeiz)
    if (q_realloc) then
       call chmrealloc('domdec.src','allocate_domdec','nodeind',nx,ny,nz,intg=nodeind)
    else
       call chmalloc('domdec.src','allocate_domdec','nodeind',nx,ny,nz,intg=nodeind)
    endif
    i = 0
    do iz=1,nz
       do iy=1,ny
          do ix=1,nx
             nodeind(ix,iy,iz) = i
             if (i == mynod_split) then
                homeix = ix
                homeiy = iy
                homeiz = iz
             endif
             i = i + 1
          enddo
       enddo
    enddo

    ! ###########################################################################

    ! Initialize tables used in update_groupl
    nxt = min(3,nx)
    nyt = min(3,ny)
    nzt = min(3,nz)
    nneigh = nxt*nyt*nzt - 1
    ixl = 1 - anint(real(nxt)/two)
    ixh = ixl + nxt-1
    iyl = 1 - anint(real(nyt)/two)
    iyh = iyl + nyt-1
    izl = 1 - anint(real(nzt)/two)
    izh = izl + nzt-1
    k = 0
    do ix=ixl,ixh
       do iy=iyl,iyh
          do iz=izl,izh
             if (ix /= 0 .or. iy /= 0 .or. iz /= 0) then
                k = k + 1
                neighind(ix,iy,iz) = k
                neighlist(k) = nodeindfunc(ix+homeix,iy+homeiy,iz+homeiz)
             endif
          enddo
       enddo
    enddo
    if (nneigh /= k) then
       call wrndie(-5,'<domdec>','Error in nneigh')
    endif
    if (nxt < 3) then
       neighind(-1,:,:) = neighind(1,:,:)
    endif
    if (nyt < 3) then
       neighind(:,-1,:) = neighind(:,1,:)
    endif
    if (nzt < 3) then
       neighind(:,:,-1) = neighind(:,:,1)
    endif

    return
  end subroutine allocate_domdec

  ! *
  ! * Builds group list for this node's home box
  ! * NOTE: requires coordinates to be current at every node
  !
  subroutine build_groupl(x, y, z)
    use memory
    use number
    use psf,only:natom
    use image,only:xtlabc
    use groupxfast,only:ngroup, calc_groupbox, group, group_out, groupcenter
    use mpi
    use inbnd,only:cutnb
    use domdec_grouped,only:q_grouped, rcut_grouped, calc_rcut_grouped
    use domdec_bonded,only:check_home_box
    use domdec_d2d_comm
    use domdec_common,only:homezone, zonelist, q_ortho, frx, fry, frz, &
         zonelist_atom, groupl, natoml, atoml, q_test, set_box
    use domdec_dlb,only:calc_min_nodefr,rcut_grouped_dlb,print_dlb_info,&
         q_load_balance,q_load_balance_x,q_load_balance_y,q_load_balance_z
    use stream,only:outu,prnlev
    implicit none
    ! Input / Output
    real(chm_real) x(*), y(*), z(*)
    ! Variables
    integer i, igroup, iatom
    integer is, iq, j
    real(chm_real) xf, yf, zf
    real(chm_real) xtlinv(6)
    logical ok
    logical inhome
    real(chm_real) min_nodefr_x, min_nodefr_y, min_nodefr_z

    call set_box()

    if (q_ortho) then
    else
       call invt33s(xtlinv,xtlabc,ok)
    endif

    igroup = 0
    iatom = 0
    homezone(1:natom) = 0
    do i=1,ngroup
       ! Calculate groups centers and box size
       call calc_groupbox(i, x, y, z)

       call put_into_centerbox(groupcenter(1,i), groupcenter(2,i), groupcenter(3,i), &
            xf, yf, zf)

       call check_home_box(xf, yf, zf, inhome)

       if (inhome) then
          call group_out(group(i), is, iq)
          igroup = igroup + 1
          if (igroup > size(groupl)) then
             call chmrealloc('domdec.src','build_groupl','groupl',&
                  int(2*igroup),intg=groupl)
          endif
          groupl(igroup) = i
          ! Copy coordinates to local array
          if (iatom + iq-is+1 > size(atoml)) then
             call chmrealloc('domdec.src','build_groupl','atoml',&
                  int(2*(iatom+iq-is+1)),intg=atoml)
          endif
          do j=1,iq-is+1
             atoml(iatom+j) = j-1+is
          enddo
          iatom = iatom + iq-is+1
          homezone(is:iq) = 1
       endif
    enddo

    zonelist(1) = igroup
    zonelist_atom(1) = iatom
    natoml = iatom

    ! Calculate bonded interaction cut-off (used in transfer_coord)
    call calc_rcut_grouped(x, y, z, rcut_grouped)

    if (rcut_grouped > cutnb) then
       write (outu,'(a,2f10.2)') 'rcut_grouped, cutnb=',rcut_grouped,cutnb
       call wrndie(-5,'<domdec>','grouped cut-off larger than non-bonded cut-off')
    endif

    rcut_grouped_dlb = rcut_grouped
    call calc_min_nodefr(min_nodefr_x, min_nodefr_y, min_nodefr_z)

    if (min_nodefr_x > frx) then
       q_load_balance_x = .false.
       if (prnlev > 5) &
            write (outu,'(a)') 'build_groupl: Switching Dynamic Load Balancing OFF in X direction'
    endif

    if (min_nodefr_y > fry) then
       q_load_balance_y = .false.
       if (prnlev > 5) &
            write (outu,'(a)') 'build_groupl: Switching Dynamic Load Balancing OFF in Y direction'
    endif

    if (min_nodefr_z > frz) then
       q_load_balance_z = .false.
       if (prnlev > 5) &
            write (outu,'(a)') 'build_groupl: Switching Dynamic Load Balancing OFF in Z direction'
    endif

    q_load_balance = q_load_balance_x .or. q_load_balance_y .or. q_load_balance_z

    call print_dlb_info()

    if (q_test) then
       call test_build_groupl()
    endif

    return
  end subroutine build_groupl

  ! *
  ! * Test for build_groupl()
  ! *
  subroutine test_build_groupl()
    use domdec_d2d_comm,only:test_groupl_atoml
    use stream,only:outu,prnlev
    implicit none

    if (test_groupl_atoml()) then
       if (prnlev > 2) then
          write (outu,'(a)') 'test_build_groupl OK'
       endif
    else
       call wrndie(-5,'<domdec>','test_build_groupl FAILED')
    endif

    return
  end subroutine test_build_groupl

  ! *
  ! * Calculate triclinic simulation box vectors
  ! *
  subroutine calc_triclinic_vec()
    use number
    use image
    implicit none
    ! Variables
    real(chm_real) k(3), l(3), m(3)

    ! Define original lattice vectors k, l, m
    k = (/ xtlabc(1), xtlabc(2), xtlabc(4) /)
    l = (/ xtlabc(2), xtlabc(3), xtlabc(5) /)
    m = (/ xtlabc(4), xtlabc(5), xtlabc(6) /)

!    tricdir(1:3) = .false.

    if (k(2) /= zero .or. k(3) /= zero) then
       call wrndie(-5,"<domdec.src>","x-direction cannot be triclinic")
    endif

    if (l(1) /= zero .or. l(3) /= zero) then
!       tricdir(2) = .true.
    endif

    if (m(1) /= zero .or. m(2) /= zero) then
!       tricdir(3) = .true.
    endif
    

    return
  end subroutine calc_triclinic_vec

  ! *
  ! * Calculate lattice vectors for the rectangular box
  ! *
  subroutine calc_latvec()
    use number,only:rsmall,zero,one,minone,half
    use image,only:xtlabc
    use parallel,only:mynod
    use domdec_common,only:hboxx, hboxy, hboxz, xdisp, ydisp, zdisp, box_a, box_b, box_c
    use vector
    implicit none
    ! Input / Output
    ! Variables
    real(chm_real) k(3), l(3), m(3), u(3), v(3), w(3), t(3), rv(3)
    real(chm_real) ut(3), vt(3), wt(3), kt(3), lt(3), mt(3)
    real(chm_real) rm(3,3)
    real(chm_real) f, g, e, cosa
    real(chm_real) boxx, boxy, boxz

    ! Define original lattice vectors k, l, m
    k = (/ xtlabc(1), xtlabc(2), xtlabc(4) /)
    l = (/ xtlabc(2), xtlabc(3), xtlabc(5) /)
    m = (/ xtlabc(4), xtlabc(5), xtlabc(6) /)

    if (mynod == 0) write (*,'(a)') 'original:'
    if (mynod == 0) write (*,'(a,3f12.6)') 'k=',k
    if (mynod == 0) write (*,'(a,3f12.6)') 'l=',l
    if (mynod == 0) write (*,'(a,3f12.6)') 'm=',m

!!$    ! Check that |k| >= |l| >= |m|
!!$    call dotpr(k,k,3,f)
!!$    call dotpr(l,l,3,g)
!!$    call dotpr(m,m,3,e)
!!$    if (g - f < rsmall .or. e - g < rsmall .or. e - f < rsmall) &
!!$         call wrndie(-5,"<domdec.src>", "k,l,m not in correct order")

    ! Calculate orthogonal lattice vectors u, v, w using Gram-Schmidt process
    u = k

    t = k
    call normall(t,3)     ! t = normalized k
    call dotpr(l,t,3,f)   ! f = l . t
    v = l - f*t

    call dotpr(m,u,3,f)   ! f = m . u
    call dotpr(u,u,3,g)   ! g = u . u
    e = f / g             ! e = m . u / u . u
    call dotpr(m,v,3,f)   ! f = m . v
    call dotpr(v,v,3,g)   ! g = v . v
    f = f / g             ! f = m . v / v . v
    w = m - e*u - f*v

    ! The orthorhombic box sizes are now given by the length of u, v, and w
    call dotpr(u,u,3,boxx)
    call dotpr(v,v,3,boxy)
    call dotpr(w,w,3,boxz)
    boxx = sqrt(boxx)
    boxy = sqrt(boxy)
    boxz = sqrt(boxz)
    hboxx = half*boxx
    hboxy = half*boxy
    hboxz = half*boxz
    if (mynod == 0) write (*,'(a,3f12.6)') 'box=',boxx,boxy,boxz

!    if (mynod == 0) write (*,'(a,9f8.3)') 'k,l,m=',k,l,m
    call calc_vol(k,l,m,f)
!    if (mynod == 0) write (*,'(a,f12.6)') 'vol(klm)=',f

!    if (mynod == 0) write (*,'(a,9f8.3)') 'u,v,w=',u,v,w
    call calc_vol(u,v,w,f)
!    if (mynod == 0) write (*,'(a,f12.6)') 'vol(uvw)=',f

    ! Rotate vectors such that u is along x, v is along y, and w is along z axis
    !
    ! First, rotate u to be along x axis:
    ! -Calculate unit vector rv = u x x, that is perpendicular to the plane
    ! defined by u and x.
    ! -Calculate angle cosa = u . x / |u|
    ! -Rotate u, v, w around rv by amount arccos(cosa)
    !
    t = (/ one, zero, zero /)  ! t = x (unit vector)
    call cross3(u,t,rv)        ! rv = u x x
    call normall(rv,3)         ! rv = normalized u x x
    call dotpr(u,t,3,cosa)     ! u . x
    call dotpr(u,u,3,f)        ! f = u . u
    cosa = cosa/sqrt(f)        ! cosa = u . x / norm(u)

    call rot_matrix(cosa, rv, rm)
    ut = u
    vt = v
    wt = w
    ! u, v, w are the new rotated vectors
    call mul_mat_vec(rm, ut, u)
    call mul_mat_vec(rm, vt, v)
    call mul_mat_vec(rm, wt, w)
    kt = k
    lt = l
    mt = m
    call mul_mat_vec(rm, kt, k)
    call mul_mat_vec(rm, lt, l)
    call mul_mat_vec(rm, mt, m)

    ! Remove any rounding errors
    u(2:3) = zero
    v(1) = zero

    if (mynod == 0) write (*,'(a)') 'After first rotation:'
    if (mynod == 0) write (*,'(a,9f8.3)') 'u,v,w=',u,v,w
    if (mynod == 0) write (*,'(a,9f8.3)') 'k,l,m=',k,l,m
    call calc_vol(u,v,w,f)
    if (mynod == 0) write (*,'(a,f12.6)') 'vol(uvw)=',f

    ! Second, rotate v to be along the y axis:
    ! -Calculate the angle cosa between v the y axis
    ! -Rotate around -x axis by cosa
    !
    if (abs(v(3)) > rsmall) then
       t = (/ zero, one, zero /)   ! t = y (unit vector)
       call dotpr(v,t,3,cosa)
       call dotpr(v,v,3,f)         ! f = v . v
       cosa = cosa/sqrt(f)         ! cosa = v . y / norm(v)
       if (mynod == 0) write (*,'(a,f12.6)') 'cosa=',cosa
       ! BUG: rv could be either sign ! rv = (/ minone, zero, zero /)  ! rv = -x (unit vector)
       call cross3(v,t,rv)        ! rv = u x y
       call normall(rv,3)         ! rv = normalized u x y
       call rot_matrix(cosa, rv, rm)
       vt = v
       wt = w
       ! u, v, w are the new rotated vectors
       ! NOTE: we rotate around u, so no need to rotate it
       call mul_mat_vec(rm, vt, v)
       call mul_mat_vec(rm, wt, w)
       kt = k
       lt = l
       mt = m
       call mul_mat_vec(rm, kt, k)
       call mul_mat_vec(rm, lt, l)
       call mul_mat_vec(rm, mt, m)
    endif

    v(3) = zero
    w(1:2) = zero

    if (mynod == 0) write (*,'(a)') 'After second rotation:'
    if (mynod == 0) write (*,'(a,9f8.3)') 'u,v,w=',u,v,w
    if (mynod == 0) write (*,'(a,9f8.3)') 'k,l,m=',k,l,m
    call calc_vol(u,v,w,f)
    if (mynod == 0) write (*,'(a,f12.6)') 'vol(uvw)=',f

    if (abs(k(2)) > rsmall .or. abs(k(3)) > rsmall .or. abs(l(3)) > rsmall) &
         call wrndie(-5,"<domdec.src>", "k or l are not correctly rotated")

    k(2:3) = zero
    l(3) = zero

    if (k(1) < zero .or. l(2) < zero .or. m(3) < zero) &
         call wrndie(-5,"<domdec.src>", "k, l, or m have wrong sign")

!!$    if (abs(l(1)) > half*k(1) .or. abs(m(1)) > half*k(1) .or. abs(m(2)) > half*l(2)) &
!!$         call wrndie(-5,"<domdec.src>", "k, l, m have wrong lengths")

!!$    if (abs(l(1)) > half*k(1)) write (*,*) '1'
!!$    if (abs(m(1)) > half*k(1)) write (*,*) '2'
!!$    if (abs(m(2)) > half*l(2)) write (*,*) '3'

    ! Check orthogonality
    call dotpr(u,v,3,f)
    if (abs(f) > rsmall)  call wrndie(-5,"<domdec.src>", "u and v not orthogonal")
    call dotpr(u,w,3,f)
    if (abs(f) > rsmall)  call wrndie(-5,"<domdec.src>", "u and w not orthogonal")
    call dotpr(w,v,3,f)
    if (abs(f) > rsmall)  call wrndie(-5,"<domdec.src>", "w and v not orthogonal")

    call dotpr(u,u,3,f)
    f = sqrt(f)
    call dotpr(v,v,3,g)
    g = sqrt(g)
    call dotpr(w,w,3,e)
    e = sqrt(e)
    if (mynod == 0) write (*,'(a,4f12.6)') 'u=',u,f
    if (mynod == 0) write (*,'(a,4f12.6)') 'v=',v,g
    if (mynod == 0) write (*,'(a,4f12.6)') 'w=',w,e
    call calc_vol(u,v,w,f)
    if (mynod == 0) write (*,'(a,f12.6)') 'vol(uvw)=',f

    call dotpr(k,k,3,f)
    f = sqrt(f)
    call dotpr(l,l,3,g)
    g = sqrt(g)
    call dotpr(m,m,3,e)
    e = sqrt(e)
    if (mynod == 0) write (*,'(a,4f12.6)') 'k=',k,f
    if (mynod == 0) write (*,'(a,4f12.6)') 'l=',l,g
    if (mynod == 0) write (*,'(a,4f12.6)') 'm=',m,e
    call calc_vol(k,l,m,f)
    if (mynod == 0) write (*,'(a,f12.6)') 'vol(klm)=',f

    ! Set box displacement vectors
    xdisp(1) = k(1)
    xdisp(2) = l(1)
    xdisp(3) = m(1)
    ydisp(1) = l(2)
    ydisp(2) = m(2)
    zdisp = m(3)

    box_a(1:3) = k(1:3)
    box_b(1:3) = l(1:3)
    box_c(1:3) = m(1:3)

    return
  end subroutine calc_latvec

  ! *
  ! * Translates coordinates x, y, z into the center box
  ! * Returns: xf, yf, zf in center box fractional coordinates
  ! *
  subroutine put_into_centerbox(x, y, z, xf, yf, zf)
    use number
    use domdec_common,only:q_ortho, invx, invy, invz
    implicit none
    ! Input / Output
    real(chm_real) x, y, z, xf, yf, zf
    ! Variables
    integer, parameter :: maxboxind = 100
    integer i, j, k
    logical foundbox

    if (q_ortho) then
       xf = x*invx + half
       yf = y*invy + half
       zf = z*invz + half
       xf = xf - floor(xf)
       yf = yf - floor(yf)
       zf = zf - floor(zf)
    else
       foundbox = .false.
       do i=0,maxboxind
          do j=-i,i
             do k=-i,i
                call tryshiftbox(j,k,i)
                if (foundbox) return
             enddo
          enddo
          
          do j=-i,i
             do k=-i+1,i-1
                call tryshiftbox(j,i,k)
                if (foundbox) return
             enddo
          enddo
          
          do j=-i+1,i-1
             do k=-i+1,i-1
                call tryshiftbox(i,k,j)
                if (foundbox) return
             enddo
          enddo
          
       enddo
       call wrndie(-5,"<domdec.src>", "Unable to find centerbox")
    endif

    return

  contains
    subroutine tryshiftbox(n1, n2, n3)
      use domdec_common,only:xdisp, ydisp, zdisp
      implicit none
      ! Input / Output
      integer n1, n2, n3
      ! Variables
      real(chm_real) xs, ys, zs
      
      xs = n1*xdisp(1) + n2*xdisp(2) + n3*xdisp(3)
      ys = n2*ydisp(1) + n3*ydisp(2)
      zs = n3*zdisp

      xf = x + xs
      yf = y + ys
      zf = z + zs
      if (in_centerbox(xf, yf, zf)) then
         xf = xf*invx + half
         yf = yf*invy + half
         zf = zf*invz + half
         foundbox = .true.
         return
      endif

      xf = x - xs
      yf = y - ys
      zf = z - zs
      if (in_centerbox(xf, yf, zf)) then
         xf = xf*invx + half
         yf = yf*invy + half
         zf = zf*invz + half
         foundbox = .true.
         return
      endif

      return
    end subroutine tryshiftbox

  end subroutine put_into_centerbox

  ! *
  ! * Returns .true. if coordinates x, y, z is in the centerbox
  ! *
  function in_centerbox(x, y, z) result(inflag)
    use domdec_common,only:hboxx, hboxy, hboxz
    implicit none
    ! Input / Output
    real(chm_real) x, y, z
    logical inflag
    
    if (x > -hboxx .and. x < hboxx .and. y > -hboxy .and. y < hboxy .and. &
         z > -hboxz .and. z < hboxz) then
       inflag = .true.
    else
       inflag = .false.
    endif
    
    return
  end function in_centerbox

  ! *
  ! * Calculates |k . (l x m)|
  ! *
  subroutine calc_vol(k,l,m,vol)
    use vector
    implicit none
    ! Input / Output
    real(chm_real) k(3), l(3), m(3), vol
    ! Variables
    real(chm_real) lxm(3)
    
    call cross3(l,m,lxm)
    call dotpr(k,lxm,3,vol)
    vol = abs(vol)

    return
  end subroutine calc_vol

  ! *
  ! * Multiplies 3x3 matrix with 3x1 vector: c = Ab
  ! *
  subroutine mul_mat_vec(A, b, c)
    implicit none
    ! Input / Output
    real(chm_real) A(3,3), b(3), c(3)

    c(1) = sum(A(1:3,1)*b(1:3))
    c(2) = sum(A(1:3,2)*b(1:3))
    c(3) = sum(A(1:3,3)*b(1:3))

    return
  end subroutine mul_mat_vec

  ! *
  ! * Calculates rotation matrix rm around vector rv by angle acos(cosa)
  ! *
  subroutine rot_matrix(cosa, rv, rm)
    use number,only:one
    implicit none
    ! Input / Output
    real(chm_real) cosa, rv(3), rm(3,3)
    ! Variables
    real(chm_real) sina
    integer, save :: ncall = 0
    ncall = ncall + 1

    if (ncall == 1) then
       sina = sqrt(one - cosa**2)
    else
       sina = sqrt(one - cosa**2)
    endif

    rm(1,1) = cosa + rv(1)**2*(one - cosa)
    rm(2,2) = cosa + rv(2)**2*(one - cosa)
    rm(3,3) = cosa + rv(3)**2*(one - cosa)

    rm(2,1) = rv(1)*rv(2)*(one -cosa) - rv(3)*sina
    rm(3,1) = rv(1)*rv(3)*(one -cosa) + rv(2)*sina

    rm(1,2) = rv(2)*rv(1)*(one -cosa) + rv(3)*sina
    rm(3,2) = rv(2)*rv(3)*(one -cosa) - rv(1)*sina

    rm(1,3) = rv(3)*rv(1)*(one -cosa) - rv(2)*sina
    rm(2,3) = rv(3)*rv(2)*(one -cosa) + rv(1)*sina

    return
  end subroutine rot_matrix

#endif /*domdec_main*/
end module domdec
