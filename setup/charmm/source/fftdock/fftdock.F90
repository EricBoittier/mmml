module fftdock
  use chm_kinds
  use dimens_fcm
  use, intrinsic :: iso_c_binding

  implicit none

  !   XGridCenter   => Geometric center of grid in X-dimension
  !   YGridCenter   => Geometric center of grid in Y-dimension
  !   ZGridCenter   => Geometric center of grid in Z-dimension
  !   XGridLen      => Extent of grid about the origin in X-dimension
  !   YGridLen      => Extent of grid about the origin in Y-dimension
  !   ZGridLen      => Extent of grid about the origin in Z-dimension
  !   XGridMin      => Minimum X coordinate of potential grid
  !   YGridMin      => Minimum Y coordinate of potential grid
  !   ZGridMin      => Minimum Z coordinate of potential grid
  !   DGrid         => Spacing between grid points

  !   VdwEmax       => Emax for vdw potential
  !   ElecReplEmax  => Emax for repulsive electrostatic potential
  !   ElecAttrEmax  => Emax for attractive electrostatic potential

  logical :: &
       flag_fftdock_clear = .false., &
       flag_init_fft_plans = .false., &
       flag_init_lig_conformers = .false., &
       flag_init_lig_grid = .false., &
       flag_init_lig_rotamers = .false., &
       flag_init_potential_grid = .false., &
       flag_init_quaternions = .false., &
       flag_init_quaternions_from_file = .false., &
       flag_add_random = .false., &
       flag_read_probe = .false., &
       is_ocl_session_active = .false.

#if KEY_FFTDOCK == 1

  !! Grid information
  integer :: ElecMode, num_select_atoms, NumGrid, &
             XGridNum, YGridNum, ZGridNum
  real(chm_real) DGrid, GridForce
  real(chm_real) rcta, rctb, hmax
  real(chm_real) XGridLen, YGridLen, ZGridLen
  real(chm_real) XGridCenter, YGridCenter, ZGridCenter
  real(chm_real4) DGrid4, RIJ
  real(chm_real4) XGridMin, YGridMin, ZGridMin
  real(chm_real4) VdwEmax, ElecReplEmax, ElecAttrEmax
  real(chm_real4) fa, fb, gmax, Dielec, CCELEC_charmm4
  real(chm_real), allocatable, dimension(:) :: GridRadii
  real(chm_real4), allocatable, dimension(:), target :: GridPot
  real(chm_real4), allocatable, dimension(:), target :: GridRadii4
  real(chm_real4), allocatable, dimension(:), target :: Used_GridPot
  real(chm_real4), allocatable, dimension(:), target :: SelectAtomsParameters

  !! Ligand (conformer) information
  integer num_lig_conformers, num_lig_atoms, num_low_energy_rotamer_per_conformer
  real(chm_real4), allocatable, dimension(:,:,:) :: lig_conformer_coors
  real(chm_real4), allocatable, dimension(:,:,:) :: lig_low_energy_coors
  real(chm_real4), allocatable, dimension(:) :: lig_low_energy_energy
  integer, allocatable, dimension(:) :: flag_select_lig
  integer :: num_quaternions, batch_size, quaternions_file_unit
  real(chm_real4), allocatable, dimension(:,:) :: quaternions
  real(chm_real4), allocatable, dimension(:,:,:) :: lig_rotamer_coors
  real(chm_real4), allocatable, dimension(:,:) :: lig_rotamer_min_coors
  real(chm_real4), allocatable, dimension(:,:) :: lig_rotamer_max_coors
  real(chm_real4), allocatable, dimension(:), target :: EnergyGrid
  real(chm_real4), allocatable, dimension(:) :: energy_rotamer
  integer, allocatable, dimension(:) :: energy_rotamer_sort_idx
  real(chm_real), allocatable, dimension(:,:) :: Lig_NB
  integer idx_conformer

  !! Ligand grid information
  real(chm_real4), allocatable, dimension(:), target :: LigGrid
  Logical, allocatable, dimension(:) :: flag_vdw_grid_used
  Logical, allocatable, dimension(:,:) :: flag_vdw_grid_atom
  integer, allocatable, dimension(:) :: idx_vdw_grid_used
  integer num_vdw_grid_used

  !! R2C / C2R
  type(c_ptr), target :: potential_fft_R2C_plan, lig_fft_R2C_plan, fft_C2R_plan

  !! GPU memory pointers
  type(c_ptr), target :: d_LigGrid, d_LigGrid_FFT
  type(c_ptr), target :: d_GridPot, d_GridPot_FFT
  type(c_ptr), target :: d_LigSum, d_LigSum_FFT

  !! Ligand grid used
  real(chm_real4),allocatable,dimension(:),target :: d_LigandParams
  real(chm_real4),allocatable,dimension(:),target :: d_LigandRotamerCoors
  real(chm_real4),allocatable,dimension(:),target :: d_LigandRotamerMinCoors

#if KEY_OPENCL == 1
  type(c_ptr) :: ocl_context, ocl_queue
#endif /* OPENCL */

  interface
#if KEY_CUDA == 1
     SUBROUTINE make_fft_R2C_plan(xdim, ydim, zdim, batch_size, plan) &
          bind(c, name = 'make_cufft_R2C_plan')
       use, intrinsic :: iso_c_binding
       implicit none
       integer(c_int), value :: xdim, ydim, zdim, batch_size
       type(c_ptr), value :: plan
     END SUBROUTINE make_fft_R2C_plan

     SUBROUTINE make_fft_C2R_plan(xdim, ydim, zdim, batch_size, plan) &
          bind(c, name = 'make_cufft_C2R_plan')
       use, intrinsic :: iso_c_binding
       implicit none
       integer(c_int), value :: xdim, ydim, zdim, batch_size
       type(c_ptr), value :: plan
     END SUBROUTINE make_fft_C2R_plan

     SUBROUTINE allocate_GPU_id(gpu_id) bind(c, name = 'allocate_GPU_id')
       use, intrinsic :: iso_c_binding
       implicit none
       integer(c_int), value :: gpu_id
     END SUBROUTINE allocate_GPU_id

     SUBROUTINE destroy_fft_plan(plan) bind(c, name = 'destroy_cufft_plan')
       use, intrinsic :: iso_c_binding
       implicit none
       type(c_ptr), value :: plan
     END SUBROUTINE destroy_fft_plan

     SUBROUTINE rigid_FFT_dock(xdim, ydim, zdim, &
          batch_size, idx_batch, &
          num_rotamers, num_grid, &
          potential_R2C_plan, lig_R2C_plan, C2R_plan, &
          grid_potential, LigGrid, EnergyGrid, &
          d_LigGrid_Fort, d_LigGrid_FFT_Fort, &
          d_GridPot_Fort, d_GridPot_FFT_Fort, &
          d_LigSum_Fort, d_LigSum_FFT_Fort) bind(c, name = 'rigid_FFT_dock')
       use, intrinsic :: iso_c_binding
       implicit none
       integer(c_int), value :: xdim, ydim, zdim
       integer(c_int), value :: batch_size, idx_batch
       integer(c_int), value :: num_rotamers, num_grid
       type(c_ptr), value :: potential_R2C_plan, lig_R2C_plan, C2R_plan
       type(c_ptr), value :: grid_potential, LigGrid, EnergyGrid
       type(c_ptr), value :: d_LigGrid_Fort, d_LigGrid_FFT_Fort
       type(c_ptr), value :: d_GridPot_Fort, d_GridPot_FFT_Fort
       type(c_ptr), value :: d_LigSum_Fort, d_LigSum_FFT_Fort
     END SUBROUTINE rigid_FFT_dock

     SUBROUTINE batchFFT(xdim, ydim, zdim, batch_size, &
          plan, grid_potential) bind(c, name = 'batchFFT')
       use, intrinsic :: iso_c_binding
       implicit none
       integer(c_int), value :: xdim, ydim, zdim, batch_size
       type(c_ptr), value :: plan
       type(c_ptr), value :: grid_potential
     END SUBROUTINE batchFFT

     SUBROUTINE clean_FFTDock_GPU(d_LigGrid_Fort, d_LigGrid_FFT_Fort, &
          d_GridPot_Fort, d_GridPot_FFT_Fort, &
          d_LigSum_Fort, d_LigSum_FFT_Fort) bind(c, name = 'clean_FFTDock_GPU')
       use, intrinsic :: iso_c_binding
       implicit none
       type(c_ptr), value :: d_LigGrid_Fort, d_LigGrid_FFT_Fort
       type(c_ptr), value :: d_GridPot_Fort, d_GridPot_FFT_Fort
       type(c_ptr), value :: d_LigSum_Fort, d_LigSum_FFT_Fort
     END SUBROUTINE clean_FFTDock_GPU

     SUBROUTINE calcLigGrid(BatchIdx, BatchSize, NumQuaterions, NumAtoms,&
          NumVdwGridUsed, DGrid, XGridNum, YGridNum, &
          ZGridNum,SelectAtomsParameters, LigGrid, LigRotamerCoors, &
          LigRotamerMinCoors, d_LigGrid_F) bind(c,name='calcLigGrid')
       use, intrinsic :: iso_c_binding
       implicit none
       integer(c_int), value :: BatchIdx, BatchSize, NumQuaterions, NumAtoms,&
            NumVdwGridUsed, XGridNum, YGridNum, ZGridNum
       real(c_float), value :: DGrid
       type(c_ptr), value :: SelectAtomsParameters, LigGrid, &
            LigRotamerCoors, LigRotamerMinCoors, d_LigGrid_F
     END SUBROUTINE calcLigGrid

     SUBROUTINE calcPotGrid(NumGrid, NumAtoms, DGrid, &
          XGridNum, YGridNum, ZGridNum, XMin, YMin, ZMin, &
          Fa, Fb, Gmax, &
          VdwEmax, ElecAttrEmax, ElecReplEmax, CCELEC, ElecMode, Dielec, &
          SelectAtomsParameters, GridPot, GridRadii) &
          bind(c, name = 'calcPotGrid')
       use, intrinsic :: iso_c_binding
       integer(c_int), value :: XGridNum, YGridNum, ZGridNum, NumAtoms, &
            NumGrid, ElecMode
       real(c_float), value :: DGrid, Fa, Fb, Gmax, &
            XMin,YMin,ZMin,ElecReplEmax,ElecAttrEmax,VdwEmax,CCELEC, Dielec
       type(c_ptr), value :: SelectAtomsParameters, GridPot, GridRadii
     END SUBROUTINE calcPotGrid

#elif KEY_OPENCL == 1 /* KEY_CUDA */
     function calcPotGrid(device, ctx, q, NumGrid, NumAtoms, DGrid, &
          XGridNum, YGridNum, ZGridNum, XMin, YMin, ZMin, &
          Fa, Fb, Gmax, &
          VdwEmax, ElecAttrEmax, ElecReplEmax, CCELEC, ElecMode, Dielec, &
          SelectAtomsParameters, GridPot, GridRadii) &
          result(status) &
          bind(c, name = 'calcPotGrid')
       use, intrinsic :: iso_c_binding, only: c_int, c_float, c_ptr
       implicit none
       type(c_ptr), value :: device, ctx, q  ! OpenCL data structures
       integer(c_int), value :: XGridNum, YGridNum, ZGridNum, NumAtoms, &
            NumGrid, ElecMode
       real(c_float), value :: DGrid, Fa, Fb, Gmax, &
            XMin,YMin,ZMin,ElecReplEmax,ElecAttrEmax,VdwEmax,CCELEC, Dielec
       type(c_ptr), value :: SelectAtomsParameters, GridPot, GridRadii
       integer(c_int) :: status  ! return value
     end function calcPotGrid

     subroutine calcLigGrid(ocl_dev_id, ocl_context, ocl_queue, &
          BatchIdx, BatchSize, &
          NumQuaterions, NumAtoms, NumVdwGridUsed, &
          DGrid, XGridNum, YGridNum, ZGridNum, &
          SelectAtomsParameters, &
          LigRotamerCoors, LigRotamerMinCoors, d_LigGrid_F) &
          bind(c,name='calcLigGrid')
       use, intrinsic :: iso_c_binding, only: c_int, c_float, c_ptr
       implicit none
       integer(c_int), value :: BatchIdx, BatchSize, NumQuaterions, NumAtoms,&
            NumVdwGridUsed, XGridNum, YGridNum, ZGridNum
       real(c_float), value :: DGrid
       real(c_float), dimension(:) :: &
            SelectAtomsParameters, &
            LigRotamerCoors, LigRotamerMinCoors
       type(c_ptr), value :: ocl_dev_id, ocl_context, ocl_queue
       type(c_ptr) :: d_LigGrid_F
     end subroutine calcLigGrid

     subroutine rigid_fft_dock(ocl_dev_id, ctx, q, &
          xdim, ydim, zdim, &
          batch_size, idx_batch, &
          num_rotamers, num_grid, &
          potential_r2c_plan, lig_r2c_plan, c2r_plan, &
          grid_potential, energygrid, &
          d_liggrid_fort, d_liggrid_fft_fort, &
          d_gridpot_fort, d_gridpot_fft_fort, &
          d_ligsum_fort, d_ligsum_fft_fort) bind(c)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_float
       implicit none
       integer(c_int), value :: &
            xdim, ydim, zdim, &
            batch_size, idx_batch, &
            num_rotamers, num_grid
       real(c_float), dimension(:) :: grid_potential
       type(c_ptr), value :: EnergyGrid, &
            ocl_dev_id, ctx, q, &
            potential_r2c_plan, lig_r2c_plan, c2r_plan
       type(c_ptr) :: &
            d_liggrid_fort, d_liggrid_fft_fort, &
            d_gridpot_fort, d_gridpot_fft_fort, &
            d_ligsum_fort, d_ligsum_fft_fort
     end subroutine rigid_fft_dock

     subroutine init_fft() bind(c)
       implicit none
     end subroutine init_fft

     subroutine tear_down_fft() bind(c)
       implicit none
     end subroutine tear_down_fft

     subroutine destroy_fft_plan(plan) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr) :: plan
     end subroutine destroy_fft_plan

     subroutine make_fft_r2c_plan(ocl_context, ocl_queue, &
          xdim, ydim, zdim, &
          batch_size, out_plan) &
          bind(c)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       integer(c_int), value :: xdim, ydim, zdim, batch_size
       type(c_ptr), value :: ocl_context, ocl_queue
       type(c_ptr) :: out_plan
     end subroutine make_fft_r2c_plan

     subroutine make_fft_c2r_plan(ocl_context, ocl_queue, &
          xdim, ydim, zdim, &
          batch_size, out_plan) &
          bind(c)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       integer(c_int), value :: xdim, ydim, zdim, batch_size
       type(c_ptr), value :: ocl_context, ocl_queue
       type(c_ptr) :: out_plan
     end subroutine make_fft_c2r_plan

     subroutine clean_fftdock_gpu(d_liggrid_fort, d_liggrid_fft_fort, &
          d_gridpot_fort, d_gridpot_fft_fort, &
          d_ligsum_fort, d_ligsum_fft_fort) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr) :: &
            d_liggrid_fort, d_liggrid_fft_fort, &
            d_gridpot_fort, d_gridpot_fft_fort, &
            d_ligsum_fort, d_ligsum_fft_fort
     end subroutine clean_fftdock_gpu
#endif  /* KEY_CUDA */
  end interface

CONTAINS

#if KEY_OPENCL == 1 && KEY_CUDA == 0
  subroutine choose_opencl_device(dev_id)
    use opencl_main_mod, only: ocl_device_select
    implicit none
    integer, intent(in) :: dev_id
    call ocl_device_select(dev_id)
  end subroutine choose_opencl_device

  subroutine fftdock_device_init(gpuid)
    use stream, only: outu, prnlev
    use opencl_main_mod, only: &
         ocl_is_initialized, &
         ocl_devices, selected_device, &
         ocl_device_init, ocl_device_max_mem_get, &
         ocl_device_show_current, &
         ocl_begin_session

    implicit none

    ! input args
    integer, intent(in) :: gpuid

    ! local vars
    integer :: status

    if (.not. ocl_is_initialized) then
       status = ocl_device_init(ocl_devices)
    end if

    if ((gpuid .le. 0) .and. (.not. c_associated(selected_device))) then
       status = ocl_device_max_mem_get(ocl_devices, selected_device)
    end if

    if ((gpuid .gt. 0) .and. (.not. c_associated(selected_device))) then
       call choose_opencl_device(gpuid)
    end if

    if (.not. is_ocl_session_active) then
       status = ocl_begin_session(selected_device, ocl_context, ocl_queue)
       is_ocl_session_active = .true.
    end if

    if (prnlev >= 2) then
       write(outu, '(a)') 'Device selected:'
       call ocl_device_show_current()
    end if
  end subroutine fftdock_device_init
#endif /* KEY_OPENCL == 1 && KEY_CUDA == 0 */

  SUBROUTINE fft_dock_set(comlyn, comlen)
    use consta
    use coord
    use memory
    use number
    use param
    use psf
    use select, only: atmsel
    use stream
    use string

    use, intrinsic :: iso_c_binding, only: c_ptr

#if KEY_OPENCL == 1 && KEY_CUDA == 0
    use opencl_main_mod, only: &
         selected_device, &
         ocl_end_session
#endif

    implicit none

    character(len=*) comlyn
    integer comlen
    Logical flag_PGEN, flag_save_lig_conformer, flag_READ_PotGrid, flag_coor, &
        flag_include_identity_quaternions, flag_clear, flag_quaternion_file_unit
    logical flag_Write_PotGrid,flag_hbond,Form
    logical flag_read_probe_from_file
    integer out_Potgrid_Unit, probes_unit
    integer num_grid_points, tot_num_grid_points
    integer num_parameters
    integer num_batch
    integer gpuid, GridUnit
    integer tmp_I, tmp_J, tmp_K, tmp_idx, min_I, min_J, min_K, &
            idx_batch, tmp_io, HBatm, idx
    type(c_ptr) :: plan
    real(chm_real4) tmp_r
    real(chm_real) tmp_radii
    integer, allocatable, dimension(:) :: HDonor, Hacceptor, flag_select
    character tmp_a1, tmp_a2
    character(len=1000) tmp_a

    integer status

    flag_PGEN = .False.
    flag_coor = .False.
    flag_hbond = .False.
    flag_Read_PotGrid = .False.
    flag_Write_PotGrid = .False.
    flag_save_lig_conformer = .False.
    flag_read_probe_from_file = .False.
    flag_include_identity_quaternions = .False.

    !! Pass logical flags
    flag_PGEN = (IndxA(comLyn, comLen, 'PGEN').GT.0)
    flag_coor = (IndxA(comLyn, comLen, 'COOR').GT.0)
    flag_fftdock_clear = (IndxA(comLyn, comLen, 'CLEA').GT.0)
    if (flag_fftdock_clear) then
       if (prnLev .ge. 2) then
          write(outu,'(a)') "<FFTDOCK> FFTG CLEAR sets initialization flags to FALSE for"
          write(outu,'(a)') "          fft_plans, lig_grid, potential_grid, lig_conformers, lig_rotamers"
          write(outu,'(a)') "          quaternions, quaternions_from_file, aff_random"
       endif
       flag_init_lig_conformers = .False.
       flag_init_fft_plans = .true.
       flag_init_lig_grid = .false.
       flag_init_lig_rotamers = .false.
       flag_init_potential_grid = .false.
       flag_init_quaternions = .false.
       flag_init_quaternions_from_file = .false.
       flag_add_random = .false.
    endif
    flag_hbond = (IndxA(comLyn, comLen, 'GRHB').GT.0)
    flag_READ_PotGrid = (IndxA(comLyn, comLen, 'READ').GT.0)
    flag_Write_PotGrid = (IndxA(comLyn, comLen, 'PWRI').GT.0)
    flag_save_lig_conformer = (IndxA(comLyn, comLen, 'LCON').GT.0)
    flag_read_probe_from_file = (IndxA(comLyn, comLen, 'RADI').GT.0)
    flag_include_identity_quaternions = (IndxA(comLyn, comLen, 'IQUA').GT.0)
    flag_add_random = (IndxA(comLyn, comLen, 'RNQU').GT.0)  ! multiply quaternions by random quaternion
    if (flag_add_random) then
       if (prnlev > 2) then
          write(outu,'(a)')"<FFTDOCK> RNQU key chosen, quaternions will be multiplied by a random rotation"
       endif
    endif

    !! get GPUID, default is 0
    gpuid = GTrmI(comlyn,comlen,'GPUI', 0)

    ! write protein grid to file
    if (flag_Write_PotGrid) then
       if(.not.flag_init_potential_grid) then
          CALL WRNDIE(-5, '<FFTDOCK>', "FFT dock potential has not been initialized.")
       endif
       Form = .false.
       if (IndxA(comLyn,comLen,'UNFO').GT.0) Form = .False.
       if (IndxA(comLyn,comLen,'FORM').GT.0) Form = .True.
       out_Potgrid_Unit = GTrmI(comlyn, comlen, 'UNIT', OUTU)
       call Write_Potential_Grid(XGridCenter, YGridCenter, ZGridCenter, &
            XGridLen, YGridLen, ZGridLen, DGrid, GridForce, NumGrid, &
            XGridNum, YGridNum, ZGridNum, GridPot, GridRadii, out_Potgrid_Unit, &
            flag_hbond, Form)
    endif

    ! read probes radii
    if (flag_read_probe_from_file) then
        probes_unit = GTrmI(comlyn, comlen, 'UNIT', -1)
        call Read_Probes_Radii(probes_unit)
        flag_read_probe = .True.
    endif

    ! generate protein grid using GPU
    if (flag_PGEN) then
        !! Check probe radii
        If (.not. flag_read_probe) &
           call WrnDie(1,'fft_dock_set', &
           'Probes radii has not been read yet.')

        !! Prepare parameters
        XGridCenter = GTrmf(comlyn,comlen, 'XCEN', Zero )
        YGridCenter = GTrmf(comlyn,comlen, 'YCEN', Zero )
        ZGridCenter = GTrmf(comlyn,comlen, 'ZCEN', Zero )
        XGridLen = GTrmf(comlyn,comlen, 'XMAX', Zero )
        YGridLen = GTrmf(comlyn,comlen, 'YMAX', Zero )
        ZGridLen = GTrmf(comlyn,comlen, 'ZMAX', Zero )
        RCTA = GTrmf(comlyn,comlen, 'RCTA', 0.0 )
        RCTB = GTrmf(comlyn,comlen, 'RCTB', 1.0 )
        HMAX = GTrmf(comlyn,comlen, 'HMAX', -1.0 )
        DGrid = GTrmf(comlyn,comlen, 'DGRI', Half )
        ElecReplEmax = GTrmf(comlyn,comlen, 'MAXE', 150.0 )
        ElecAttrEmax = GTrmf(comlyn,comlen, 'MINE', -300.0 )
        VdwEmax = GTrmf(comlyn,comlen, 'EMAX', 20.0 )
        GridForce = GTrmf(comlyn,comlen, 'FORC', 300.0 )
        Dielec = GTrmf(comlyn,comlen, 'EPSI', 1.0 )
        XGridNum = XGridLen / DGrid + 2
        YGridNum = YGridLen / DGrid + 2
        ZGridNum = ZGridLen / DGrid + 2
        XGridMin = XGridCenter - XGridLen/2.0
        YGridMin = YGridCenter - YGridLen/2.0
        ZGridMin = ZGridCenter - ZGridLen/2.0
        RIJ = (RCTA - RCTB) * (RCTA - RCTB)
        IF (RIJ .eq. 0) Then
           IF (PrnLev .ge. 2) Write(OutU, '(a)') &
              'Please check h-bond grid potential setup'
           RIJ = 1
           HMax = 0 ! This will set all energy value to be 0
        EndIf
        GMAX = HMAX
        FA = -4 * HMAX / RIJ
        FB = 0.5 * (RCTA + RCTB)
        DGrid4 = DGrid
        CCELEC_charmm4 = CCELEC_charmm
        num_grid_points = XGridNum * YGridNum * ZGridNum
        tot_num_grid_points = num_grid_points * NumGrid
        call chmalloc('fftdock.src','fft_dock_set','GridRadii4',&
             NumGrid - 3, cr4 = GridRadii4)
        GridRadii4 = GridRadii

        !! Set up hydrogen bond donor & acceptor index
        call chmalloc('fft_dock.src', 'fft_dock_set', 'HDonor', &
                      natom, intg = HDonor)
        call chmalloc('fft_dock.src', 'fft_dock_set', 'HDonor', &
                      natom, intg = HAcceptor)
        HDonor = 0
        HAcceptor = 0
        If (NDON.GT.0.and.NDON.LE.NATOM) Then
           Do HBatm = 1, NDON
              HDonor(IHD1(HBatm)) = 1
           Enddo
        Endif
        If (NACC.GT.0.and.NACC.LE.NATOM) Then
           Do HBatm = 1, NACC
              HAcceptor(IACC(HBatm)) = 1
           Enddo
        Endif

        !! Build selected atoms parameters array
        !! Parameters order for each atom: X,Y,Z,Epsilon,VDWRadii,Charge,Hdonor,Hacceptor
        idx=1
        num_parameters=8
        call chmalloc('fft_dock.src', 'fft_dock_set', 'flag_select', &
                       natom, intg = flag_select)
        num_select_atoms = AtmSel(comlyn, comlen, flag_select, .FALSE.)
        call chmalloc('fft_dock.src', 'fft_dock_set', &
            'SelectAtomsParameters', num_select_atoms*num_parameters, &
            cr4 = SelectAtomsParameters)
        DO tmp_i = 1, natom
           IF (flag_select(tmp_i) > 0) THEN
           SelectAtomsParameters((idx-1)*num_parameters+1)=X(tmp_i)
           SelectAtomsParameters((idx-1)*num_parameters+2)=Y(tmp_i)
           SelectAtomsParameters((idx-1)*num_parameters+3)=Z(tmp_i)
           SelectAtomsParameters((idx-1)*num_parameters+4)=EFF(ITC(IAC(tmp_i))) !well depth
           SelectAtomsParameters((idx-1)*num_parameters+5)=VDWR(ITC(IAC(tmp_i))) !vdw radii
           SelectAtomsParameters((idx-1)*num_parameters+6)=CG(tmp_i) !charge
           SelectAtomsParameters((idx-1)*num_parameters+7)=HDonor(tmp_i) !donor
           SelectAtomsParameters((idx-1)*num_parameters+8)=HAcceptor(tmp_i) !acceptor
           idx=idx+1
           Endif
        Enddo

        !! Output Information
        If (PrnLev .ge. 2) Then
           write(outu,'(a)') "<FFTDOCK> Generate protein grid using GPU"
           write(outu,'(a,f12.5,a,f12.5,a,f12.5)') "Grid center x= ",XGridCenter,&
               " y= ",YGridCenter," z= ",ZGridCenter
           write(outu,'(a,f12.5,a,f12.5,a,f12.5)') "Grid length x= ",XGridLen,&
               " y= ",YGridLen," z= ",ZGridLen
           write(outu,'(a,i5,a,i5,a,i5)') "Grid points number x= ",XGridNum,&
               " y= ",YGridNum," z= ",ZGridNum
           write(outu,'(a,f12.5,a,f12.5,a,f12.5)') "Grid minimum dimension x=",&
               XGridMin, " y= ",YGridMin," z= ",ZGridMin
           write(outu,'(a,f12.5,a,f12.5,a,f12.5)') "Hydrogen bond grid rcta=",&
               rcta, " rctb= ",rctb," cutoff= ",gmax
           write(outu,'(a,f12.5)') "Grid space = ", DGrid4
           write(outu,'(a,i10)') "Number of grid points per type = ", num_grid_points
           write(outu,'(a,f12.5,a)') "Grid force constant = ", GridForce," kcal/mol/A^2"
           write(outu,'(a,f12.5)') "Electrostatics softcore repulsion emax  = ", &
               ElecReplEmax
           write(outu,'(a,f12.5)') "Electrostatics softcore attraction emax  = ", &
               ElecAttrEmax
           write(outu,'(a,f12.5)') "Van der Waals softcore emax  = ", VdwEmax
           write(outu,'(a)') "  Probe       Radii"
           do tmp_i = 1, NumGrid - 3
               write(outu,'(i7,f12.6)') tmp_i, GridRadii4(tmp_i)
           enddo
           write(outu,'(a,i5)') "Total number of grid types", NumGrid
           write(outu,'(a,i10)') "Total number of grid points = ", tot_num_grid_points
           write(outu,'(a,i10)') "Total number of atoms used for protein grid = ",idx-1
        Endif

        IF (IndxA(comLyn,comLen,'RDIE').GT.0) THEN
            If (PrnLev .ge. 2) Write(outu,'(a,f12.5)') &
               "Using radial dependent dielectric constant = ", Dielec
            ElecMode = 1
        ELSE IF (IndxA(comLyn,comLen,'CDIE').GT.0) THEN
            If (PrnLev .ge. 2) Write(outu,'(a,f12.5)') &
               "Using constant dielectric constant = ",Dielec
            ElecMode = 0
        ElSE
            ElecMode = 0
            If (PrnLev .ge. 2) Write(outu,'(a,f12.5)') &
               "Using constant dielectric constant = ",Dielec
        Endif
        !! Allocate grid pot array
        If (allocated(GridPot)) then
           call chmdealloc('fft_dock.src', 'fft_dock_set', 'GridPot', &
                            tot_num_grid_points, cr4 = GridPot)
        endif
        call chmalloc('fft_dock.src', 'fft_dock_set', 'GridPot', &
                       tot_num_grid_points, cr4 = GridPot)
        GridPot(:) = 0.0D0

#if KEY_CUDA == 1
        call allocate_gpu_id(gpuid)
#elif KEY_OPENCL == 1
        call fftdock_device_init(gpuid)
#endif
        call generate_potential_grid_gpu(NumGrid, num_select_atoms, DGrid4, &
             XGridNum, YGridNum, ZGridNum, XGridMin, YGridMin, ZGridMin, &
             Fa, Fb, Gmax, VdwEmax, ElecAttrEmax, ElecReplEmax, CCELEC_charmm4, &
             ElecMode, Dielec, SelectAtomsParameters, GridPot, GridRadii4, gpuid)

        !! deallocate charmm array
        call chmdealloc('fft_dock.src', 'fft_dock_set', 'HDonor', &
                         natom, intg = HDonor)
        call chmdealloc('fft_dock.src', 'fft_dock_set', 'HDonor', &
                         natom, intg = HAcceptor)
        call chmdealloc('fftdock.src','fft_dock_set','GridRadii4',&
                         NumGrid - 3, cr4 = GridRadii4)
        call chmdealloc('fft_dock.src', 'fft_dock_set', 'flag_select', &
                          natom, intg = flag_select)
        call chmdealloc('fft_dock.src', 'fft_dock_set', &
            'SelectAtomsParameters', num_select_atoms*num_parameters, &
            cr4 = SelectAtomsParameters)

        !! Grid generate successful
        flag_init_potential_grid = .TRUE.
    endif

    !! read potential grid
    if (flag_READ_PotGrid) THEN
       if (flag_init_potential_grid) THEN
          CALL WRNDIE(-5, '<FFTDOCK>', "FFT dock potential has been initialized.")
       endif
       GridUnit = GTrmI(comlyn, comlen, 'UNIT', OutU)
       Form = .False.
       Form = (IndxA(comLyn,comLen,'FORM').GT.0)
       If (GridUnit .eq. OutU) Form = .True.
       call Read_Potential_Grid(GridUnit, flag_hbond, Form)
       flag_init_potential_grid = .TRUE.
    endif

    !! save ligand conformation and do FFT dock
    if (flag_save_lig_conformer) THEN
       idx_conformer = GTrmI(comlyn, comlen, 'ICON', 1)
#if KEY_CUDA == 1
       print *, "allocate gpu" 
       call allocate_gpu_id(gpuid)
#elif KEY_OPENCL == 1
       call wrndie(-5, '<FFTDOCK>', &
       'FFTDOCK is not support on the OpenCL platform yet, please use CUDA instead.')
       call fftdock_device_init(gpuid)
#endif
       print *, "flag_init_quaternions_from_file ", &
        flag_init_quaternions_from_file
       if (.not. flag_init_quaternions_from_file) then
          quaternions_file_unit = GTrmI(comlyn, comlen, 'QUAU', -1)
          if (Quaternions_file_unit /= -1) THEN
             num_quaternions = 0

             !! count the number of quaternions in the provided file
             DO
                READ(quaternions_file_unit,'(a)',iostat = tmp_io) tmp_a
                IF (tmp_io/=0) EXIT
                if (tmp_a(1:1) /= '*') then
                   num_quaternions = num_quaternions + 1
                end if
             END DO
             write(outu, '(a,i5)')"FFTDOCK - num of orientations: ", num_quaternions

             !! read the quaternions
             !! First check if already allocated
             if (allocated(quaternions)) then                
                CALL chmdealloc('fftdock.src', 'fft_dock_set', 'quaternions', &
                     size(quaternions,1),size(quaternions,2),cr4 = quaternions)
             endif
                
             CALL chmalloc('fftdock.src', 'fft_dock_set', 'quaternions', num_quaternions, 4, cr4 = quaternions)
             rewind(Quaternions_file_unit)

             DO
                READ(quaternions_file_unit,'(a)',iostat = tmp_io) tmp_a
                if (tmp_a(1:1) /= '*') then
                   backspace quaternions_file_unit
                   exit
                end if
             END DO

             do tmp_i = 1, num_quaternions
                read(Quaternions_file_unit, *) quaternions(tmp_i, 1), quaternions(tmp_i, 2), quaternions(tmp_i, 3), quaternions(tmp_i, 4)
             end do

             flag_init_quaternions_from_file = .True.
             flag_init_quaternions = .True.
          else
             num_quaternions = GTrmI(comlyn, comlen, 'NQUA', 1)
             if (.not. flag_init_quaternions) then
                if (allocated(quaternions)) then                
                   CALL chmdealloc('fftdock.src', 'fft_dock_set', 'quaternions', &
                        size(quaternions,1), size(quaternions,2),cr4 = quaternions)
                endif
                CALL chmalloc('fftdock.src', 'fft_dock_set', 'quaternions', num_quaternions, 4, cr4 = quaternions)
                flag_init_quaternions = .True.
             end if
          end if
       end if

       batch_size = GTrmI(comlyn, comlen, 'SIZB', 100)
       num_batch = num_quaternions / batch_size
       if (num_batch * batch_size < num_quaternions) THEN
          num_batch = num_batch + 1
       end if

       write(outu, '(a,i5)')"FFTDOCK - batch_size: ", batch_size
       write(outu, '(a,i5)')"FFTDOCK - num_batch: ", num_batch

       call Save_Lig_Conformer(comlyn, comlen)
       call Generate_Lig_Rotamers(idx_conformer, num_quaternions, flag_include_identity_quaternions)
       call Calc_Lig_Rotamer_Min_Coors()
       call Calc_Lig_Rotamer_Max_Coors()

       do idx_batch = 1, num_batch
          !write(outu, '(a,i5)')"FFTDOCK - idx_batch: ", idx_batch
          !open(unit=10,file="LigGridGPU.dat",form="unformatted")
          call Generate_Lig_Grid_GPU(idx_batch)
          !write(10) LigGrid
          !close(10)
          !open(unit=10,file="LigGrid.dat",form="unformatted")
          !call Generate_Lig_Grid(idx_batch)
          !write(10) LigGrid
          !close(10)
          if (.not. flag_init_fft_plans) then
             flag_init_fft_plans = .True.
#if KEY_CUDA == 1  /* KEY_CUDA */
             Call make_fft_R2C_plan(XGridNum, YGridNum, ZGridNum, &
                  num_vdw_grid_used + 1, c_loc(potential_fft_R2C_plan))
             call make_fft_R2C_plan(XGridNum, YGridNum, ZGridNum, &
                  (num_vdw_grid_used + 1) * batch_size, c_loc(lig_fft_R2C_plan))
             call make_fft_C2R_plan(XGridNum, YGridNum, ZGridNum, &
                  batch_size, c_loc(fft_C2R_plan))
#elif KEY_OPENCL == 1  /* KEY_CUDA */
             call init_fft()
             Call make_fft_R2C_plan(ocl_context, ocl_queue, &
                  XGridNum, YGridNum, ZGridNum, &
                  num_vdw_grid_used + 1, potential_fft_R2C_plan)
             call make_fft_R2C_plan(ocl_context, ocl_queue, &
                  XGridNum, YGridNum, ZGridNum, &
                  (num_vdw_grid_used + 1) * batch_size, lig_fft_R2C_plan)
             call make_fft_C2R_plan(ocl_context, ocl_queue, &
                  XGridNum, YGridNum, ZGridNum, &
                  batch_size, fft_C2R_plan)
#endif /* KEY_CUDA */
          end if

#if KEY_CUDA == 1  /* KEY_CUDA */
          call rigid_FFT_dock(XGridNum, YGridNum, ZGridNum, &
               batch_size, idx_batch, &
               num_quaternions, num_vdw_grid_used + 1, &
               c_loc(potential_fft_R2C_plan), &
               c_loc(lig_fft_R2C_plan), &
               c_loc(fft_C2R_plan), &
               c_loc(Used_GridPot), c_loc(LigGrid), c_loc(EnergyGrid), &
               c_loc(d_LigGrid), c_loc(d_LigGrid_FFT), &
               c_loc(d_GridPot), c_loc(d_GridPot_FFT), &
               c_loc(d_LigSum), c_loc(d_LigSum_FFT))
#elif KEY_OPENCL == 1  /* KEY_CUDA */
          call rigid_FFT_dock(selected_device, ocl_context, ocl_queue, &
               XGridNum, YGridNum, ZGridNum, &
               batch_size, idx_batch, &
               num_quaternions, num_vdw_grid_used + 1, &
               potential_fft_R2C_plan, &
               lig_fft_R2C_plan, &
               fft_C2R_plan, &
               Used_GridPot, c_loc(EnergyGrid), &
               d_LigGrid, d_LigGrid_FFT, &
               d_GridPot, d_GridPot_FFT, &
               d_LigSum, d_LigSum_FFT)
#endif /* KEY_CUDA */
          call Translate_Lig_Rotamer(lig_rotamer_coors, batch_size, idx_batch, num_lig_atoms, &
               lig_rotamer_min_coors, lig_rotamer_max_coors, &
               XGridNum, YGridNum, ZGridNum, DGrid, &
               XGridMin, YGridMin, ZGridMin,&
               EnergyGrid, Energy_rotamer)
       end do
       call Sort_Lig_Rotamer(num_quaternions, energy_rotamer, energy_rotamer_sort_idx)
       call Keep_Low_Energy_Rotamer(idx_conformer)

#if KEY_CUDA == 1  /* KEY_CUDA */
       call clean_FFTDock_GPU(c_loc(d_LigGrid), c_loc(d_LigGrid_FFT), c_loc(d_GridPot), &
            c_loc(d_GridPot_FFT), c_loc(d_LigSum), c_loc(d_LigSum_FFT))
#elif KEY_OPENCL == 1  /* KEY_CUDA */
       call clean_FFTDock_GPU(d_LigGrid, d_LigGrid_FFT, &
            d_GridPot, d_GridPot_FFT, &
            d_LigSum, d_LigSum_FFT)
#endif /* KEY_CUDA */
    end if

    if ( flag_coor ) THEN
       tmp_I = GTrmI(comlyn, comlen, 'ICON', 1)
       tmp_J = GTrmI(comlyn, comlen, 'IROT', 1)
       call Copy_Lig_Coor(tmp_I, tmp_J)
    endif

    if ( flag_fftdock_clear ) THEN
       flag_init_potential_grid = .False.
       flag_init_lig_grid = .False.
       if (flag_init_fft_plans) then
#if KEY_CUDA == 1  /* KEY_CUDA */
          call destroy_fft_plan(c_loc(potential_fft_R2C_plan))
          call destroy_fft_plan(c_loc(lig_fft_R2C_plan))
          call destroy_fft_plan(c_loc(fft_C2R_plan))
#elif KEY_OPENCL == 1  /* KEY_CUDA */
          call destroy_fft_plan(potential_fft_R2C_plan)
          call destroy_fft_plan(lig_fft_R2C_plan)
          call destroy_fft_plan(fft_C2R_plan)
          call tear_down_fft()
#endif  /* KEY_CUDA */
          flag_init_fft_plans = .false.
       end if  ! if flat_init_fft_plans
#if KEY_CUDA == 0 && KEY_OPENCL == 1  /* KEY_CUDA */
       if (is_ocl_session_active) then
          status = ocl_end_session(ocl_context, ocl_queue)
          is_ocl_session_active = .false.
       end if  ! if is_ocl_session_active
#endif
    endif  ! if flag_fftdock_clear
  END SUBROUTINE fft_dock_set

  SUBROUTINE Copy_Lig_Coor(idx_conf, idx_rota)
    use psf ,only :  natom
    use coord, only : X, Y, Z
    use param_store, only: set_param
    use stream

    implicit None
    integer idx_conf, idx_rota
    integer tmp_I, tmp_J, tmp_K
    real(chm_real) tmp_r

    tmp_J = 1
    do tmp_I = 1, natom
       if (flag_select_lig(tmp_I) > 0) THEN
          X(tmp_I) = lig_low_energy_coors((idx_conf-1)*num_low_energy_rotamer_per_conformer+idx_rota, tmp_J, 1)
          Y(tmp_I) = lig_low_energy_coors((idx_conf-1)*num_low_energy_rotamer_per_conformer+idx_rota, tmp_J, 2)
          Z(tmp_I) = lig_low_energy_coors((idx_conf-1)*num_low_energy_rotamer_per_conformer+idx_rota, tmp_J, 3)
          tmp_J = tmp_J + 1
       endif
    enddo

    tmp_r = lig_low_energy_energy((idx_conf-1)*num_low_energy_rotamer_per_conformer+idx_rota)
    call set_param("FFTE", tmp_r)

    Write(OutU, '(a, i5, a, i5, a)') ' Coordinates of the ', idx_conf, 'th conforamtion and the ', idx_rota, 'th rotation from FFTDOCK are copied into the main coordinates.'

  END SUBROUTINE Copy_Lig_Coor

  SUBROUTINE Keep_Low_Energy_Rotamer(idx_conf)
    implicit None

    integer idx_conf
    integer tmp_I, tmp_J, tmp_idx

    do tmp_I = 1, num_low_energy_rotamer_per_conformer
       tmp_idx = energy_rotamer_sort_idx(tmp_I)
       lig_low_energy_energy((idx_conf-1)*num_low_energy_rotamer_per_conformer + tmp_I) = energy_rotamer(tmp_idx)
       ! print *, lig_low_energy_energy((idx_conf-1)*num_low_energy_rotamer_per_conformer + tmp_I)
       do tmp_J = 1, num_lig_atoms
          lig_low_energy_coors((idx_conf-1)*num_low_energy_rotamer_per_conformer + tmp_I, tmp_J, 1) = lig_rotamer_coors(tmp_idx, tmp_J, 1)
          lig_low_energy_coors((idx_conf-1)*num_low_energy_rotamer_per_conformer + tmp_I, tmp_J, 2) = lig_rotamer_coors(tmp_idx, tmp_J, 2)
          lig_low_energy_coors((idx_conf-1)*num_low_energy_rotamer_per_conformer + tmp_I, tmp_J, 3) = lig_rotamer_coors(tmp_idx, tmp_J, 3)
       enddo
    enddo

  END SUBROUTINE Keep_Low_Energy_Rotamer

  SUBROUTINE Sort_Lig_Rotamer(num_rotamer, energy_rotamer, energy_rotamer_sort_idx)
    use memory

    implicit None
    real(chm_real4), dimension(:) :: energy_rotamer
    integer, dimension(:) :: energy_rotamer_sort_idx
    integer num_rotamer
    real(chm_real4), allocatable, dimension(:) :: tmp_energy_rotamer
    integer tmp_I, tmp_J, tmp_K, tmp_L

    real(chm_real4) tmp_r
    CALL chmalloc('fftdock.src', 'fft_dock_set', 'tmp_energy_rotamer', num_quaternions, cr4 = tmp_energy_rotamer)

    do tmp_I = 1, num_rotamer
       energy_rotamer_sort_idx(tmp_I) = tmp_I
       tmp_energy_rotamer(tmp_I) = energy_rotamer(tmp_I)
    enddo

    do tmp_I = 2, num_rotamer
       do tmp_J = 1, tmp_I - 1
          tmp_K = tmp_I - tmp_J
          if ( tmp_energy_rotamer(tmp_K + 1) < tmp_energy_rotamer(tmp_K) ) THEN
             tmp_r = tmp_energy_rotamer(tmp_K + 1)
             tmp_energy_rotamer(tmp_K + 1) = tmp_energy_rotamer(tmp_K)
             tmp_energy_rotamer(tmp_K) = tmp_r

             tmp_L = energy_rotamer_sort_idx(tmp_K + 1)
             energy_rotamer_sort_idx(tmp_K + 1) = energy_rotamer_sort_idx(tmp_K)
             energy_rotamer_sort_idx(tmp_K) = tmp_L
          endif
       enddo
    enddo
    CALL chmdealloc('fftdock.src', 'fft_dock_set', 'tmp_Energy_rotamer', num_quaternions, cr4 = tmp_energy_rotamer)
  END SUBROUTINE SORT_LIG_ROTAMER

  SUBROUTINE Translate_Lig_Rotamer(lig_rotamer_coors, batch_size, idx_batch, num_atoms, &
                                   lig_rotamer_min_coors, lig_rotamer_max_coors, &
                                   XGridNum, YGridNum, ZGridNum, DGrid, &
                                   XGridMin, YGridMin, ZGridMin,&
                                   EnergyGrid, Energy_rotamer)

    implicit None
    real(chm_real4), dimension(:,:,:) :: lig_rotamer_coors
    integer batch_size, num_atoms, idx_batch
    real(chm_real4), dimension(:,:) :: lig_rotamer_min_coors
    real(chm_real4), dimension(:,:) :: lig_rotamer_max_coors

    integer XGridNum, YGridNum, ZGridNum
    real(chm_real4) XGridMin, YGridMin, ZGridMin
    real(chm_real) DGrid

    real(chm_real4), dimension(:) :: EnergyGrid
    real(chm_real4), dimension(:) :: energy_rotamer

    integer min_I, min_J, min_K
    integer tmp_I, tmp_J, tmp_K, tmp_L, tmp_M, tmp_base_idx, tmp_idx

    do tmp_M = 1, batch_size
       tmp_L = (idx_batch - 1) * batch_size + tmp_M
       if (tmp_L > num_quaternions) THEN
          exit
       endif

       tmp_base_idx = (tmp_M - 1) * XGridNum * YGridNum * ZGridNum
       energy_rotamer(tmp_L) = EnergyGrid(tmp_base_idx + 1)

       min_I = 1
       min_J = 1
       min_K = 1

       do tmp_I = 1, XGridNum - (int((lig_rotamer_max_coors(tmp_L,1) - lig_rotamer_min_coors(tmp_L,1))/DGrid) + 1)
          do tmp_J = 1, YGridNum - (int((lig_rotamer_max_coors(tmp_L,2) - lig_rotamer_min_coors(tmp_L,2))/DGrid) + 1)
             do tmp_K = 1, ZGridNum - (int((lig_rotamer_max_coors(tmp_L,3) - lig_rotamer_min_coors(tmp_L,3))/DGrid) + 1)
                tmp_idx = ((tmp_I - 1) * YGridNum + tmp_J - 1) * ZGridNum + tmp_K
                tmp_idx = tmp_base_idx + tmp_idx
                if (EnergyGrid(tmp_idx) < energy_rotamer(tmp_L)) then
                   energy_rotamer(tmp_L) = EnergyGrid(tmp_idx)
                   min_I = tmp_I
                   min_J = tmp_J
                   min_K = tmp_K
                endif
             enddo
          enddo
       enddo

       do tmp_I = 1, num_lig_atoms
          lig_rotamer_coors(tmp_L, tmp_I, 1) = lig_rotamer_coors(tmp_L, tmp_I, 1) - lig_rotamer_min_coors(tmp_L, 1) + XGridMin + (min_I - 1) * DGrid
          lig_rotamer_coors(tmp_L, tmp_I, 2) = lig_rotamer_coors(tmp_L, tmp_I, 2) - lig_rotamer_min_coors(tmp_L, 2) + YGridMin + (min_J - 1) * DGrid
          lig_rotamer_coors(tmp_L, tmp_I, 3) = lig_rotamer_coors(tmp_L, tmp_I, 3) - lig_rotamer_min_coors(tmp_L, 3) + ZGridMin + (min_K - 1) * DGrid
       enddo
    enddo
  END SUBROUTINE Translate_Lig_Rotamer


  Function Get_Idx_Nearest_Radii(query_radii, radii_array, len_array) result(min_idx)
    implicit None

    real(chm_real) query_radii
    real(chm_real), allocatable, dimension(:) :: radii_array
    integer len_array

    real(chm_real) min_diff
    integer tmp_i, min_idx
    min_diff = abs(query_radii - radii_array(1))
    min_idx = 1
    do tmp_i = 2, len_array
       if (abs(query_radii - radii_array(tmp_i)) < min_diff) THEN
          min_diff = abs(query_radii - radii_array(tmp_i))
          min_idx = tmp_i
       endif
    enddo
  END Function Get_Idx_Nearest_Radii

  !! save the current main coordinates
  SUBROUTINE Save_Lig_Conformer(comlyn, comlen)
    use psf
    use string
    use number
    use select
    use memory
    use coord
    use param

    implicit None
    character(len=*) comlyn
    integer comlen

    integer idx_lig_atom, tmp_I
    real(chm_real4) tmp_eps, tmp_half_rmin


    print *, "flag_init_lig_conformers ", flag_init_lig_conformers
    if ( .not. flag_init_lig_conformers) THEN
       flag_init_lig_conformers = .True.
       if (allocated(flag_select_lig)) then
          CALL chmdealloc('fftdock.src', 'save_lig_conformer', 'flag_select_lig', &
               size(flag_select_lig), intg = flag_select_lig)
       endif
       CALL chmalloc('fftdock.src', 'save_lig_conformer', 'flag_select_lig', natom, intg = flag_select_lig)
       num_lig_atoms = AtmSel(comlyn, comlen, flag_select_lig, .FALSE.)
       num_lig_conformers = GTrmI(comlyn, comlen, 'NCON', 0)
       num_low_energy_rotamer_per_conformer = GTrmI(comlyn, comlen, 'NROK', 1)
       if (allocated(lig_conformer_coors)) then
          CALL chmdealloc('fftdock.src', 'save_lig_conformer', 'lig_conformer_coors', &
               size(lig_conformer_coors,1), size(lig_conformer_coors,2), size(lig_conformer_coors,3), cr4 = lig_conformer_coors)
       endif
       CALL chmalloc('fftdock.src', 'save_lig_conformer', 'lig_conformer_coors', num_lig_conformers, num_lig_atoms, 3, cr4 = lig_conformer_coors)
       if (allocated(lig_low_energy_coors)) then
          CALL chmdealloc('fftdock.src', 'save_lig_conformer', 'lig_conformer_coors', &
               size(lig_low_energy_coors,1), size(lig_low_energy_coors,2), size(lig_low_energy_coors,3), cr4 = lig_low_energy_coors)
       endif
       CALL chmalloc('fftdock.src', 'save_lig_conformer', 'lig_conformer_coors', &
            num_lig_conformers * num_low_energy_rotamer_per_conformer, &
             num_lig_atoms, 3, cr4 = lig_low_energy_coors)
        if (allocated(lig_low_energy_energy)) then
           CALL chmdealloc('fftdock.src', 'save_lig_conformer', 'lig_conformer_coors', &
                size(lig_low_energy_energy), cr4 = lig_low_energy_energy)
        endif
       CALL chmalloc('fftdock.src', 'save_lig_conformer', 'lig_conformer_coors', num_lig_conformers * num_low_energy_rotamer_per_conformer, &
            cr4 = lig_low_energy_energy)
       if (allocated(Lig_NB)) then
          CALL chmdealloc('fftdock.src', 'Save_Lig_Conformer', 'LigCharge', size(Lig_NB,1),size(Lig_NB,2), crl = Lig_NB)
       endif
       CALL chmalloc('fftdock.src', 'Save_Lig_Conformer', 'LigCharge', num_lig_atoms, 3, crl = Lig_NB)
    endif

    idx_lig_atom = 1
    do tmp_I = 1, natom
       if (flag_select_lig(tmp_I) > 0) THEN
          lig_conformer_coors(idx_conformer, idx_lig_atom, 1) = X(tmp_I)
          lig_conformer_coors(idx_conformer, idx_lig_atom, 2) = Y(tmp_I)
          lig_conformer_coors(idx_conformer, idx_lig_atom, 3) = Z(tmp_I)

          Lig_NB(idx_lig_atom, 1) = CG(tmp_I)
          tmp_eps = -EFF(ITC(IAC(tmp_I)))
          tmp_half_rmin = VDWR(ITC(IAC(tmp_I)))
          Lig_NB(idx_lig_atom, 2) = tmp_eps
          Lig_NB(idx_lig_atom, 3) =  tmp_half_rmin
          idx_lig_atom = idx_lig_atom + 1
       endif
    enddo


  END SUBROUTINE Save_Lig_Conformer

  !! generate rotamers for a conformer
  SUBROUTINE Generate_Lig_Rotamers(idx_conf, num_rotamer, flag_identity_quaternions)
    use memory
    implicit none
    integer idx_conf, num_rotamer
    integer atom_i, rotamer_i
    logical flag_identity_quaternions
    if ( .not. flag_init_lig_rotamers) THEN
       flag_init_lig_rotamers = .True.
       !       CALL chmalloc('fftdock.src', 'fft_dock_set', 'quaternions', num_rotamer, 4, cr4 = quaternions)
       if (allocated(lig_rotamer_coors)) then          
          CALL chmdealloc('fftdock.src', 'fft_dock_set', 'lig_rotamer_coors', &
               size(lig_rotamer_coors,1),size(lig_rotamer_coors,2), size(lig_rotamer_coors,3), cr4 = lig_rotamer_coors)
       endif
       CALL chmalloc('fftdock.src', 'fft_dock_set', 'lig_rotamer_coors', num_rotamer, num_lig_atoms, 3, cr4 = lig_rotamer_coors)
       if (allocated(lig_rotamer_min_coors)) then          
          CALL chmdealloc('fftdock.src', 'fft_dock_set', 'lig_rotamer_min_coors', &
               size(lig_rotamer_min_coors,1),size(lig_rotamer_min_coors,2), cr4 = lig_rotamer_min_coors)
       endif
       CALL chmalloc('fftdock.src', 'fft_dock_set', 'lig_rotamer_min_coors', num_rotamer, 3, cr4 = lig_rotamer_min_coors)
       if (allocated(lig_rotamer_max_coors)) then          
          CALL chmdealloc('fftdock.src', 'fft_dock_set', 'lig_rotamer_max_coors', &
               size(lig_rotamer_max_coors,1),size(lig_rotamer_max_coors,2), cr4 = lig_rotamer_max_coors)
       endif
       CALL chmalloc('fftdock.src', 'fft_dock_set', 'lig_rotamer_max_coors', num_rotamer, 3, cr4 = lig_rotamer_max_coors)
       if (allocated(Energy_rotamer)) then
          CALL chmdealloc('fftdock.src', 'fft_dock_set', 'Energy_rotamer', size(Energy_rotamer), cr4 = Energy_rotamer)
       endif
       CALL chmalloc('fftdock.src', 'fft_dock_set', 'Energy_rotamer', num_rotamer, cr4 = Energy_rotamer)
       if (allocated(energy_rotamer_sort_idx)) then
          CALL chmdealloc('fftdock.src', 'fft_dock_set', 'Energy_rotamer', size(energy_rotamer_sort_idx), intg = energy_rotamer_sort_idx)
       endif
       CALL chmalloc('fftdock.src', 'fft_dock_set', 'Energy_rotamer', num_rotamer, intg = energy_rotamer_sort_idx)

       ! energy grid only holds energies in one batch
       if (allocated(EnergyGrid)) then
          CALL chmdealloc('fftdock.src', 'fft_dock_set', 'EnergyGrid',size(EnergyGrid), cr4 = EnergyGrid)
       endif
       CALL chmalloc('fftdock.src', 'fft_dock_set', 'EnergyGrid', batch_size*XGridNum*YGridNum*ZGridNum, cr4 = EnergyGrid)
       EnergyGrid(:) = 0

       do rotamer_i = 1, num_rotamer
          energy_rotamer_sort_idx(rotamer_i) = rotamer_i
       enddo

    endif
    call Sample_Quaterions(num_rotamer, flag_identity_quaternions)
    do rotamer_i = 1, num_rotamer
       do atom_i = 1, num_lig_atoms
          call Rotate(quaternions(rotamer_i,:), &
               lig_conformer_coors(idx_conf, atom_i, :), &
               lig_rotamer_coors(rotamer_i, atom_i, :))
       enddo
    enddo
  END SUBROUTINE Generate_Lig_Rotamers

  !! calculate the minimum coordinates for each rotamer
  SUBROUTINE Calc_Lig_Rotamer_Min_Coors()
    implicit None
    integer idx_quaterions, idx_atom

    do idx_quaterions = 1, num_quaternions
       lig_rotamer_min_coors(idx_quaterions, 1) = lig_rotamer_coors(idx_quaterions, 1, 1)
       lig_rotamer_min_coors(idx_quaterions, 2) = lig_rotamer_coors(idx_quaterions, 1, 2)
       lig_rotamer_min_coors(idx_quaterions, 3) = lig_rotamer_coors(idx_quaterions, 1, 3)
       do idx_atom = 2, num_lig_atoms
          if (lig_rotamer_coors(idx_quaterions, idx_atom, 1) < lig_rotamer_min_coors(idx_quaterions, 1)) then
             lig_rotamer_min_coors(idx_quaterions, 1) = lig_rotamer_coors(idx_quaterions, idx_atom, 1)
          endif
          if (lig_rotamer_coors(idx_quaterions, idx_atom, 2) < lig_rotamer_min_coors(idx_quaterions, 2)) then
             lig_rotamer_min_coors(idx_quaterions, 2) = lig_rotamer_coors(idx_quaterions, idx_atom, 2)
          endif
          if (lig_rotamer_coors(idx_quaterions, idx_atom, 3) < lig_rotamer_min_coors(idx_quaterions, 3)) then
             lig_rotamer_min_coors(idx_quaterions, 3) = lig_rotamer_coors(idx_quaterions, idx_atom, 3)
          endif
       enddo
    enddo
  END SUBROUTINE Calc_Lig_Rotamer_Min_Coors

  !! calculate the maximum coordinates for each rotamer
  SUBROUTINE Calc_Lig_Rotamer_Max_Coors()
    implicit None
    integer idx_quaterions, idx_atom

    do idx_quaterions = 1, num_quaternions
       lig_rotamer_max_coors(idx_quaterions, 1) = lig_rotamer_coors(idx_quaterions, 1, 1)
       lig_rotamer_max_coors(idx_quaterions, 2) = lig_rotamer_coors(idx_quaterions, 1, 2)
       lig_rotamer_max_coors(idx_quaterions, 3) = lig_rotamer_coors(idx_quaterions, 1, 3)
       do idx_atom = 2, num_lig_atoms
          if (lig_rotamer_coors(idx_quaterions, idx_atom, 1) > lig_rotamer_max_coors(idx_quaterions, 1)) then
             lig_rotamer_max_coors(idx_quaterions, 1) = lig_rotamer_coors(idx_quaterions, idx_atom, 1)
          endif
          if (lig_rotamer_coors(idx_quaterions, idx_atom, 2) > lig_rotamer_max_coors(idx_quaterions, 2)) then
             lig_rotamer_max_coors(idx_quaterions, 2) = lig_rotamer_coors(idx_quaterions, idx_atom, 2)
          endif
          if (lig_rotamer_coors(idx_quaterions, idx_atom, 3) > lig_rotamer_max_coors(idx_quaterions, 3)) then
             lig_rotamer_max_coors(idx_quaterions, 3) = lig_rotamer_coors(idx_quaterions, idx_atom, 3)
          endif
       enddo
    enddo
  END SUBROUTINE Calc_Lig_Rotamer_Max_Coors

  !! generate uniformly distributed quaternions
  SUBROUTINE Sample_Quaterions(num_rotamer, flag_identity_quaternions)
    use clcg_mod

    implicit none
    real(chm_real4) PI, sigma1, sigma2, theta1, theta2
    real s
    integer num_rotamer
    integer tmp_i, iseed
    logical flag_identity_quaternions
    real(chm_real4) :: random_qua(4), tmp_qua(4)

    PI=4.D0*DATAN(1.D0)


    iseed = 1

    if (.not. flag_init_quaternions_from_file) then
       do tmp_i = 1, num_rotamer
          s = random(iseed)
          sigma1 = sqrt(1-s)
          sigma2 = sqrt(s)

          theta1 = 2 * PI * random(iseed)
          theta2 = 2 * PI * random(iseed)

          quaternions(tmp_i, 1) = cos(theta2) * sigma2
          quaternions(tmp_i, 2) = sin(theta1) * sigma1
          quaternions(tmp_i, 3) = cos(theta1) * sigma1
          quaternions(tmp_i, 4) = sin(theta2) * sigma2
       enddo
    else
       s = random(iseed)
       sigma1 = sqrt(1-s)
       sigma2 = sqrt(s)

       theta1 = 2 * PI * random(iseed)
       theta2 = 2 * PI * random(iseed)

       random_qua(1) = cos(theta2) * sigma2
       random_qua(2) = sin(theta1) * sigma1
       random_qua(3) = cos(theta1) * sigma1
       random_qua(4) = sin(theta2) * sigma2

       if (flag_add_random) then
          do tmp_i = 1, num_rotamer
             call QuaternionMultiply(quaternions(tmp_i, :), random_qua, tmp_qua)
             quaternions(tmp_i, :) = tmp_qua(:)
          enddo
       endif
    endif

    if ( flag_identity_quaternions) then
        do tmp_i = 1, num_rotamer
           quaternions(tmp_i,1) = 1
           quaternions(tmp_i,2) = 0
           quaternions(tmp_i,3) = 0
           quaternions(tmp_i,4) = 0
       enddo
    endif

  END SUBROUTINE Sample_Quaterions

  !! multiply two quaternions. (this operation is not communitive)
  SUBROUTINE QuaternionMultiply(lq, rq, res)
    implicit None
    real(chm_real4), dimension(:) :: lq, rq, res

    res(1) = lq(1)*rq(1) - lq(2)*rq(2) - lq(3)*rq(3) - lq(4)*rq(4)
    res(2) = lq(1)*rq(2) + lq(2)*rq(1) + lq(3)*rq(4) - lq(4)*rq(3)
    res(3) = lq(1)*rq(3) - lq(2)*rq(4) + lq(3)*rq(1) + lq(4)*rq(2)
    res(4) = lq(1)*rq(4) + lq(2)*rq(3) - lq(3)*rq(2) + lq(4)*rq(1)
  END SUBROUTINE QuaternionMultiply

  !! rotate a 3d vector using a quaternion
  SUBROUTINE Rotate(quaternion, icoor, fcoor)
    implicit None

    real(chm_real4), dimension(1:4) :: quaternion
    real(chm_real4), dimension(1:3) :: icoor, fcoor
    real(chm_real4), dimension(1:4) :: quaternionConj
    real(chm_real4), dimension(1:4) :: qicoor
    real(chm_real4), dimension(1:4) :: tmp1, tmp2

    quaternionConj(1) = quaternion(1)
    quaternionConj(2) = -quaternion(2)
    quaternionConj(3) = -quaternion(3)
    quaternionConj(4) = -quaternion(4)

    qicoor(1) = 0;
    qicoor(2) = icoor(1);
    qicoor(3) = icoor(2);
    qicoor(4) = icoor(3);

    CALL QuaternionMultiply(quaternion, qicoor, tmp1)
    CALL QuaternionMultiply(tmp1, quaternionConj, tmp2)

    fcoor(1) = tmp2(2)
    fcoor(2) = tmp2(3)
    fcoor(3) = tmp2(4)
  END SUBROUTINE Rotate

  FUNCTION distance(x1, y1, z1, x2, y2, z2) result(r)
    real(chm_real4) :: r
    real(chm_real4), intent(in) :: x1, y1, z1
    real(chm_real), intent(in) :: x2, y2, z2
    r = sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
  END FUNCTION distance

  SUBROUTINE Read_Potential_Grid(GridUnit, flag_hbond, Form)
    use grid_dock, only: ReadGrid
    use string
    use stream
    use ctitla
    use memory

    implicit None
    integer gridunit, num_grid_points, ngrid0
    logical form, flag_hbond
    integer tmp_i, tmp_j, tmp_k, tmp_l, tmp_idx
    real(chm_real),allocatable,dimension(:,:,:,:) :: tmp_grid_pot

    !  Read binary file of Grid Potentials
    If (.not. Form) Then
       Write(OutU,'(a,i4)') &
            ' Grid potentials read from binary file on unit', &
            GridUnit
       Call Rdtitl(TitleB, NtitlB, GridUnit, -1)
       Call Wrtitl(TitleB, NtitlB, OutU, 0)
       Read(GridUnit) NumGrid, XGridNum, YGridNum, ZGridNum
       Read(GridUnit) XGridCenter, YGridCenter, ZGridCenter
       Read(GridUnit) XGridLen, YGridLen, ZGridLen, DGrid, GridForce
    Else
       Write(OutU,'(a,i4)') &
            ' Grid potentials read from formatted file on unit', &
            GridUnit
       Call Rdtitl(TitleB, NtitlB, GridUnit, 0)
       Call Wrtitl(TitleB, NtitlB, OutU, 0)
       Read(GridUnit,'(4(i5,1x))') NumGrid, XGridNum, YGridNum, ZGridNum
       Read(GridUnit,'(3(f12.5,1x))') XGridCenter, YGridCenter, ZGridCenter
       Read(GridUnit,'(5(f12.5,1x))') XGridLen, YGridLen, ZGridLen, DGrid, GridForce
    endif

    XGridMin = XGridCenter - XGridLen/2.0
    YGridMin = YGridCenter - YGridLen/2.0
    ZGridMin = ZGridCenter - ZGridLen/2.0

    !  Write data for read parameters
    If (PrnLev >= 2) then
         if (flag_hbond) then
            Write(OutU,'(a,i4,a)') &
            ' GridSetUp: Grid potentials will be set-up for', &
            NumGrid-3,' atom types plus electrostatics plus hydrogen bond grids'
         else
            Write(OutU,'(a,i4,a)') &
            ' GridSetUp: Grid potentials will be set-up for', &
            NumGrid-1,' atom types plus electrostatics'
         endif
    endif
    If (PrnLev >= 2) Write(OutU,'(a,i4)') &
         ' GridSetUp: and read from unit', GridUnit
    If (PrnLev >= 2) Write(OutU,'(a,3f10.5)') &
         ' GridSetUp: Grid centered at', &
         XGridCenter, YGridCenter, ZGridCenter
    If (PrnLev >= 2) Write(OutU,'(a,3f10.5)') &
         ' GridSetUp: Hydrogen bond grids', &
         Rcta, Rctb, Hmax
    If (PrnLev >= 2) Write(OutU,'(3(a,f10.5,a,f10.5/))') &
         ' GridSetUp: Grid runs from (X)', &
         -XGridLen/2+XGridCenter,' -', XGridLen/2+XGridCenter, &
         ' GridSetUp: Grid runs from (Y)', &
         -YGridLen/2+YGridCenter,' -', YGridLen/2+YGridCenter, &
         ' GridSetUp: Grid runs from (Z)', &
         -ZGridLen/2+ZGridCenter,' -', ZGridLen/2+ZGridCenter
    If (PrnLev >= 2) Write(OutU,'(a,f10.5)') &
         ' GridSetUp: With a grid spacing of', DGrid
    If (PrnLev >= 2) Write(OutU,'(a,f10.3,a)') &
         ' GridSetUp: Force constant at grid edge set to ', &
         GridForce,' kcal/mol/A^2'

    call chmalloc('fftdock.src','read_potential_grid','tmp_grid_pot',NumGrid, XGridNum, YGridNum, ZGridNum,crl = tmp_grid_pot)

    if(allocated(GridRadii)) then
       call chmdealloc('fftdock.src','read_potential_grid','GridRadii',size(GridRadii),crl=GridRadii)
       !deallocate(GridRadii)
    endif

    if (flag_hbond) then
       call chmalloc('fftdock.src','read_potential_grid','GridRadii',NumGrid-3,crl=GridRadii)
       NGrid0 = NumGrid - 3
    else
       call chmalloc('fftdock.src','read_potential_grid','GridRadii',NumGrid-1,crl=GridRadii)
       NGrid0 = NumGrid - 1
    endif

    call ReadGrid(tmp_grid_pot, GridRadii, NGrid0, &
         NumGrid, XGridNum, YGridNum, ZGridNum, &
         GridUnit, Form)

    num_grid_points = NumGrid * XGridNum * YGridNum * ZGridNum

    if(allocated(GridPot)) then
       call chmdealloc('fftdock.src', 'Read_Potential_Grid', 'GridPot', size(GridPot), cr4 = GridPot)
       !deallocate(GridPot)
    endif
    call chmalloc('fftdock.src', 'Read_Potential_Grid', 'GridPot', num_grid_points, cr4 = GridPot)
    GridPot(:) = 0.0

    do tmp_l = 1, NumGrid
       do tmp_i = 1, XGridNum
          do tmp_j = 1, YGridNum
             do tmp_k = 1, ZGridNum
                tmp_idx = (((tmp_l - 1) * XGridNum + tmp_i - 1) * YGridNum + tmp_j - 1) * ZGridNum + tmp_k
                GridPot(tmp_idx) = tmp_grid_pot(tmp_l, tmp_i, tmp_j, tmp_k)
             enddo
          enddo
       enddo
    enddo
    call chmdealloc('fftdock.src','read_potential_grid','tmp_grid_pot',&
         size(tmp_grid_pot,1),size(tmp_grid_pot,2),size(tmp_grid_pot,3),&
         size(tmp_grid_pot,4),crl = tmp_grid_pot)
    !deallocate(tmp_grid_pot)
  END SUBROUTINE Read_Potential_Grid

  SUBROUTINE Write_Potential_Grid(XGridCenter, YGridCenter, ZGridCenter, &
             XGridLen, YGridLen, ZGridLen, DGrid, GridForce, NumGrid, &
             XGridNum, YGridNum, ZGridNum, GridPot, GridRadii, &
             grid_potential_unit, flag_hbond, form)
      use grid_dock, only: WriteGrid
      use string
      use stream
      use ctitla
      use memory
      integer grid_potential_unit,num_grid_points
      logical form, flag_hbond
      integer tmp_i, tmp_j, tmp_k, tmp_l, tmp_idx
      integer NumGrid, XGridNum, YGridNum, ZGridNum
      real(chm_real) XGridCenter, YGridCenter, ZGridCenter
      real(chm_real) XGridLen, YGridLen, ZGridLen
      real(chm_real) DGrid, GridForce
      real(chm_real),allocatable,dimension(:,:,:,:) :: tmp_grid_pot
      real(chm_real4),allocatable,dimension(:),target :: GridPot
      real(chm_real),allocatable,dimension(:) :: GridRadii

      if (flag_hbond) then
         call chmalloc('fftdock.src','write_potential_grid','tmp_grid_pot',&
             NumGrid, XGridNum, YGridNum, ZGridNum,crl = tmp_grid_pot)
         num_grid_points = NumGrid * XGridNum * YGridNum * ZGridNum
         do tmp_l = 1, NumGrid
           do tmp_i = 1, XGridNum
             do tmp_j = 1, YGridNum
               do tmp_k = 1, ZGridNum
               tmp_idx = (((tmp_l - 1) * XGridNum + tmp_i - 1) * YGridNum &
                   + tmp_j - 1) * ZGridNum + tmp_k
               tmp_grid_pot(tmp_l, tmp_i, tmp_j, tmp_k) = GridPot(tmp_idx)
               enddo
             enddo
           enddo
         enddo
      else
         NumGrid = NumGrid - 2
         call chmalloc('fftdock.src','write_potential_grid','tmp_grid_pot',&
             NumGrid, XGridNum, YGridNum, ZGridNum,crl = tmp_grid_pot)
         num_grid_points = NumGrid * XGridNum * YGridNum * ZGridNum
         do tmp_l = 1, NumGrid - 1
           do tmp_i = 1, XGridNum
             do tmp_j = 1, YGridNum
               do tmp_k = 1, ZGridNum
               tmp_idx = (((tmp_l - 1) * XGridNum + tmp_i - 1) * YGridNum &
                   + tmp_j - 1) * ZGridNum + tmp_k
               tmp_grid_pot(tmp_l, tmp_i, tmp_j, tmp_k) = GridPot(tmp_idx)
               enddo
             enddo
           enddo
         enddo
         do tmp_i = 1, XGridNum
           do tmp_j = 1, YGridNum
             do tmp_k = 1, ZGridNum
             tmp_idx = (((NumGrid + 1) * XGridNum + tmp_i - 1) * YGridNum &
                 + tmp_j - 1) * ZGridNum + tmp_k
             tmp_grid_pot(NumGrid, tmp_i, tmp_j, tmp_k) = GridPot(tmp_idx)
             enddo
           enddo
         enddo
      endif

      call WriteGrid(XGridCenter, YGridCenter, ZGridCenter, &
          XGridLen, YGridLen, ZGridLen, DGrid, &
          GridForce, NumGrid, XGridNum, YGridNum, ZGridNum, &
          tmp_grid_pot, GridRadii, grid_potential_unit, flag_hbond, form)
      call chmdealloc('fftdock.src','write_potential_grid','tmp_grid_pot',&
           size(tmp_grid_pot,1),size(tmp_grid_pot,2),size(tmp_grid_pot,3),&
           size(tmp_grid_pot,4),crl = tmp_grid_pot)
      !deallocate(tmp_grid_pot)
  END SUBROUTINE Write_Potential_Grid

  SUBROUTINE Read_Probes_Radii(probes_file_unit)
      use string
      use memory
      use stream
      use number
      implicit none

      integer probes_file_unit
      integer num_probes
      integer tmp_io
      integer tmp_i
      logical flag_hbond
      character(len=1000) tmp_a

      num_probes = 0
      do
        read(probes_file_unit,'(a)',iostat = tmp_io) tmp_a
        if (tmp_io/=0) exit
        if (tmp_a(1:1) /= '*') then
          num_probes = num_probes + 1
        endif
      end do
      write(outu, '(a,i5)')"<FFTDOCK> - num of probes read: ", num_probes

      if(allocated(GridRadii)) then
         call chmdealloc('fftdock.src', 'Read_Probes_Radii', 'GridRadii', size(GridRadii), crl = GridRadii)
         !deallocate(GridRadii)
      end if
      call chmalloc('fftdock.src', 'Read_Probes_Radii', 'GridRadii', num_probes, crl = GridRadii)
      rewind(probes_file_unit)
      do
          read(probes_file_unit, *,iostat = tmp_io) tmp_a
          if (tmp_a(1:1) /= '*') then
              backspace probes_file_unit
              exit
          endif
      end do

      do tmp_i = 1, num_probes
          read(probes_file_unit, *) GridRadii(tmp_i)
      enddo
      write(outu,'(a)') "  Probe       Radii"
      do tmp_i = 1, num_probes
          write(outu,'(i7,f12.6)') tmp_i,GridRadii(tmp_i)
      enddo

      ! total number of grids = num_probes + 2xHbondGrid + 1xElecGrid
      NumGrid = num_probes + 3
  END SUBROUTINE Read_Probes_Radii

#if KEY_CUDA == 1 || KEY_OPENCL == 1
  SUBROUTINE Generate_Potential_Grid_Gpu(NumGrid, num_select_atoms, DGrid4, &
            XGridNum, YGridNum, ZGridNum, XGridMin, YGridMin, ZGridMin, &
            Fa, Fb, Gmax, VdwEmax, ElecAttrEmax, ElecReplEmax, CCELEC_charmm4, &
            ElecMode, Dielec, SelectAtomsParameters, GridPot, GridRadii4, gpuid)
    use stream
    use chm_kinds
    use string
    use struc
    use, intrinsic :: iso_c_binding, only: c_associated

#if KEY_CUDA == 0 && KEY_OPENCL == 1
    use opencl_main_mod, only: selected_device
#endif

    implicit None
    integer :: ElecMode, num_select_atoms, NumGrid, &
               XGridNum, YGridNum, ZGridNum, gpuid
    real(chm_real4) :: XGridMin, YGridMin, ZGridMin, DGrid4
    real(chm_real4) :: VdwEmax, ElecReplEmax, ElecAttrEmax
    real(chm_real4) :: Fa, Fb, gmax, Dielec, CCELEC_charmm4
    real(chm_real4),allocatable,dimension(:),target :: GridPot
    real(chm_real4),allocatable,dimension(:),target :: SelectAtomsParameters
    real(chm_real4),allocatable,dimension(:),target :: GridRadii4

    integer status  ! error status for calcPotGrid

    !external GPU code
#if KEY_CUDA == 1
    call calcPotGrid(NumGrid, num_select_atoms,DGrid4,&
         XGridNum, YGridNum, ZGridNum, XGridMin,YGridMin,ZGridMin,&
         Fa, Fb, Gmax, &
         VdwEmax,ElecAttrEmax,ElecReplEmax,CCELEC_charmm4,ElecMode, Dielec, &
         c_loc(SelectAtomsParameters),c_loc(GridPot),c_loc(GridRadii4))
#elif KEY_OPENCL == 1 /* KEY_CUDA */
    status = calcPotGrid(selected_device, ocl_context, ocl_queue, &
         NumGrid, num_select_atoms,DGrid4, &
         XGridNum, YGridNum, ZGridNum, XGridMin,YGridMin,ZGridMin,&
         Fa, Fb, Gmax, &
         VdwEmax,ElecAttrEmax,ElecReplEmax,CCELEC_charmm4,ElecMode, Dielec, &
         c_loc(SelectAtomsParameters),c_loc(GridPot),c_loc(GridRadii4))
    if (status .ne. 0) then
       call wrndie(-5, '<FFTDOCK>', &
            'Something went wrong while generating the potential grid.')
    end if
#endif /* KEY_CUDA */
  END SUBROUTINE Generate_Potential_Grid_GPU

  SUBROUTINE Generate_Lig_Grid_GPU(idx_batch)
    use string
    use stream
    use memory

#if KEY_CUDA == 0 && KEY_OPENCL == 1
    use opencl_main_mod, only: selected_device
#endif

    implicit none

    integer tmp_i, tmp_j, tmp_k, tmp_idx, tmp_x
    integer idx_batch, idx_quaternions_in_batch, num_batch
    integer NumVdwGrid, num_parameters
    logical flag_hbond
    real(chm_real4) DGrid4
    integer status

    num_parameters = 4

    !----- whether or not include extra hbond grids
    if (flag_hbond) then
       NumVdwGrid = NumGrid - 3
    else
       NumVdwGrid = NumGrid - 1
    endif

    if ( .not. flag_init_lig_grid ) then
        write(outu,'(a)') "<FFTDock> Initializing ligand grid"
        write(outu,'(a,i5)') "Total number of grids = ",NumGrid
        flag_init_lig_grid = .True.

        if(allocated(flag_vdw_grid_used)) then
           call chmdealloc('fftdock.src', 'Generate_Lig_Grid_GPU', 'flag_vdw_grid_used',&
                size(flag_vdw_grid_used), log = flag_vdw_grid_used)
           !deallocate(flag_vdw_grid_used)
        endif
        call chmalloc('fftdock.src', 'Generate_Lig_Grid_GPU', 'flag_vdw_grid_used',&
            NumVdwGrid, log = flag_vdw_grid_used)
        flag_vdw_grid_used(:) = .False.

        if(allocated(flag_vdw_grid_atom)) then
           call chmdealloc('fftdock.src', 'Generate_Lig_Grid_GPU', 'flag_vdw_grid_atom',&
                size(flag_vdw_grid_atom,1),size(flag_vdw_grid_atom,1), log = flag_vdw_grid_atom)
           !deallocate(flag_vdw_grid_atom)
        endif
        call chmalloc('fftdock.src', 'Generate_Lig_Grid_GPU', 'flag_vdw_grid_atom',&
            NumVdwGrid, num_lig_atoms, log = flag_vdw_grid_atom)
        flag_vdw_grid_atom(:,:) = .False.

        do  tmp_i = 1, num_lig_atoms
            tmp_j = Get_Idx_Nearest_Radii(Lig_NB(tmp_i, 3), GridRadii, NumVdwGrid)
            flag_vdw_grid_used(tmp_j) = .True.
            flag_vdw_grid_atom(tmp_j, tmp_i) = .True.
        end do

        num_vdw_grid_used = 0
        do tmp_i = 1, NumVdwGrid
            if (flag_vdw_grid_used(tmp_i)) THEN
                num_vdw_grid_used = num_vdw_grid_used + 1
            endif
        enddo

        if(allocated(idx_vdw_grid_used)) then
           call chmdealloc('fftdock.src', 'Generate_Lig_Grid_GPU', 'idx_vdw_grid_used',&
                size(idx_vdw_grid_used), intg = idx_vdw_grid_used)
           !deallocate(idx_vdw_grid_used)
        endif
        call chmalloc('fftdock.src', 'Generate_Lig_Grid_GPU', 'idx_vdw_grid_used',&
            num_vdw_grid_used, intg = idx_vdw_grid_used)
        write(outu,'(a,i5)') "Types of VDW grid used  = ",num_vdw_grid_used

        tmp_j = 1
        do tmp_i = 1, NumVdwGrid
            if (flag_vdw_grid_used(tmp_i)) THEN
                idx_vdw_grid_used(tmp_j) = tmp_i
                tmp_j = tmp_j + 1
            endif
        enddo

        if(allocated(LigGrid)) then
           call chmdealloc('fftdock.src', 'fft_dock_set', 'LigGrid', size(LigGrid), cr4 = LigGrid)
           !deallocate(LigGrid)
        endif
        call chmalloc('fftdock.src', 'fft_dock_set', 'LigGrid',&
            batch_size*(num_vdw_grid_used + 1)*XGridNum*YGridNum*ZGridNum,&
            cr4 = LigGrid)
        ! copy the used vdw grid from GridPot to Use_GridPot
        if(allocated(Used_GridPot)) then
           call chmdealloc('fftdock.src', 'Generate_Lig_Grid_GPU', 'Used_GridPot',&
                size(Used_GridPot), cr4 = Used_GridPot)
           !deallocate(Used_GridPot)
        endif
        call chmalloc('fftdock.src', 'Generate_Lig_Grid_GPU', 'Used_GridPot',&
            (num_vdw_grid_used + 1)*XGridNum*YGridNum*ZGridNum, cr4 = Used_GridPot)

        ! print *, "num_vdw_grid_used:", num_vdw_grid_used

        ! Vdw grids
        Used_GridPot(:) = 0.0
        tmp_K = XGridNum*YGridNum*ZGridNum
        do tmp_i = 1, num_vdw_grid_used
            do tmp_j = 1, tmp_K
                Used_GridPot((tmp_i - 1)*tmp_K + tmp_j) = &
                    GridPot((idx_vdw_grid_used(tmp_i) - 1)*tmp_K + tmp_j)
            enddo
        enddo

        ! Electrostatics grid
        do tmp_j = 1, tmp_K
        Used_GridPot(num_vdw_grid_used*tmp_K + tmp_j) = &
            GridPot((NumGrid - 1)*tmp_K + tmp_j)
        enddo

        if(allocated(d_LigandParams)) then
           call chmdealloc('fftdock.src', 'Generate_Lig_Grid_GPU', 'd_LigandParams', &
                size(d_LigandParams), cr4 = d_LigandParams)
           !deallocate(d_LigandParams)
        endif
        call chmalloc('fftdock.src', 'Generate_Lig_Grid_GPU', 'd_LigandParams', &
            4*num_lig_atoms, cr4 = d_LigandParams)

        if(allocated(d_LigandRotamerCoors)) then
            call chmdealloc('fftdock.src', 'Generate_Lig_Grid_GPU', 'lig_rotamer_coors', &
                 size(d_LigandRotamerCoors), cr4 = d_LigandRotamerCoors)
            !deallocate(d_LigandRotamerCoors)
        endif
        call chmalloc('fftdock.src', 'Generate_Lig_Grid_GPU', 'lig_rotamer_coors', &
            batch_size*num_lig_atoms*3, cr4 = d_LigandRotamerCoors)

        if(allocated(d_LigandRotamerMinCoors)) then
           call chmdealloc('fftdock.src', 'Generate_Lig_Grid_GPU', 'lig_rotamer_min_coors', &
                size(d_LigandRotamerMinCoors), cr4 = d_LigandRotamerMinCoors)
           !deallocate(d_LigandRotamerMinCoors)
        endif
        call chmalloc('fftdock.src', 'Generate_Lig_Grid_GPU', 'lig_rotamer_min_coors', &
            batch_size*3, cr4 = d_LigandRotamerMinCoors)
        d_LigandParams(:) = 0
        d_LigandRotamerCoors(:) = 0
        d_LigandRotamerMinCoors(:) = 0

        ! copy ligand parameters
        do  tmp_i = 1, num_lig_atoms
            tmp_j = Get_Idx_Nearest_Radii(Lig_NB(tmp_i, 3), GridRadii, NumVdwGrid)
            tmp_x = 0 !vdw grid index of atom tmp_i in the used_GridPot
            do tmp_k = 1,tmp_j
                if (flag_vdw_grid_used(tmp_k)) then
                    tmp_x = tmp_x + 1
                end if
            enddo
            !print *,"radii=",Lig_NB(tmp_i, 3),"idx=",tmp_j,"real idx=",tmp_x-1
            !charge
            d_LigandParams(num_parameters*(tmp_i-1)+1) = Lig_NB(tmp_i,1)
            !epsilon
            d_LigandParams(num_parameters*(tmp_i-1)+2) = Lig_NB(tmp_i,2)
            !rmin/2.0
            d_LigandParams(num_parameters*(tmp_i-1)+3) = Lig_NB(tmp_i,3)
            !index of vdw grids starting from 0
            d_LigandParams(num_parameters*(tmp_i-1)+4) = tmp_x-1
        enddo
    endif

    num_batch = num_quaternions / batch_size
    if (num_batch * batch_size < num_quaternions) THEN
        num_batch = num_batch + 1
    endif
    write(outu,'(a,i5,a,i5)') "BatchId / Total = ",idx_batch," / ",num_batch
    !copy data to current batch
    do tmp_i = 1, batch_size
        idx_quaternions_in_batch  = (idx_batch-1)*batch_size + tmp_i
        if (idx_quaternions_in_batch .le. num_quaternions) then
            do tmp_j = 1, num_lig_atoms
                tmp_idx = (tmp_i-1)*num_lig_atoms*3 + (tmp_j-1)*3
                do tmp_k = 1, 3
                d_LigandRotamerCoors(tmp_idx + tmp_k) = &
                    lig_rotamer_coors(idx_quaternions_in_batch, tmp_j, tmp_k)
                enddo
            enddo
        end if
        tmp_idx = (tmp_i - 1)*3
        do tmp_k = 1, 3
            d_LigandRotamerMinCoors(tmp_idx + tmp_k) = &
                lig_rotamer_min_coors(idx_quaternions_in_batch, tmp_k)
        enddo
    enddo

    LigGrid(:) = 0
    DGrid4 = DGrid
    ! external GPU code
#if KEY_CUDA == 1
    call calcLigGrid(idx_batch, batch_size, &
         num_quaternions, num_lig_atoms, num_vdw_grid_used, &
         DGrid4, &
         XGridNum, YGridNum, ZGridNum, &
         c_loc(d_LigandParams), c_loc(LigGrid), &
         c_loc(d_LigandRotamerCoors), c_loc(d_LigandRotamerMinCoors), &
         c_loc(d_LigGrid))
#elif KEY_OPENCL == 1 /* KEY_CUDA */
    call calcLigGrid(selected_device, ocl_context, ocl_queue, &
         idx_batch, batch_size, &
         num_quaternions, num_lig_atoms, num_vdw_grid_used, dgrid4, &
         xgridnum, ygridnum, zgridnum, &
         d_ligandparams, d_ligandrotamercoors, d_ligandrotamermincoors, &
         d_liggrid)
#endif /* KEY_CUDA */
  END SUBROUTINE Generate_Lig_Grid_GPU
#endif /* KEY_CUDA == 1 || KEY_OPENCL == 1 */
#endif /* KEY_FFTDOCK */
end module fftdock
