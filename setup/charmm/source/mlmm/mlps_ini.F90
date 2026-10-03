! MLPS ini
module mlps_ini
#if KEY_MLMM==1
    use chm_kinds,     only: chm_real
    use iso_c_binding, only: c_char, c_int, c_double, c_null_char
    use number
    use stream,        only: OUTU, PRNLEV
    use psf,           only: natom
#if KEY_MLPTORCH == 1
    use mlps_abi,      only: charmm_add_dummy, charmm_add_tani, charmm_add_dpmm
#endif
    use mlps_abi,      only: charmm_add_pyth, charmm_add_pyth_custom

    implicit none    
    !===============================================================
    ! Compile-time constants
    !===============================================================
    integer, parameter :: mlps_pt_max = 512
    integer, parameter :: mlmax       = 500   ! max number of ML atoms

    logical :: mlps_debug = .true.

    !===============================================================
    ! ML (QM) selection info (Fortran / CHARMM side)
    !===============================================================
    logical :: mlps_use = .false.

    integer :: mlnum        = 0      ! number of ML regions (0 or 1)
    integer :: mlnm2        = 0      ! total number of ML atoms in compact list
    integer :: ml_mm_max    = 5000   ! hard cap for MM buffer allocation
    integer :: ml_num_mm    = 0      ! maximum MM input slots for DPMM
    integer :: natom_cache  = 0      ! cached natom for masks / C mirroring

    ! Region -> atom list
    integer, allocatable :: ml_ipt(:)    ! pointer to first atom in ml_idx
    integer, allocatable :: ml_inm(:)    ! number of atoms in region

    ! Compact ML lists
    integer, allocatable :: ml_idx(:)    ! global atom index (1..natom)
    integer, allocatable :: ml_qmZid(:)  ! atomic number Z for each ML atom
    integer, allocatable :: ml_qmSid(:)  ! contiguous species id for DPMM

    ! natom-sized mask: 1 if atom is in ML region, 0 otherwise
    integer, allocatable :: ml_imask(:)

    !===============================================================
    ! Backend / model file info
    !
    ! MODL : TorchScript model file for TANI/DPMM
    ! SPEC : optional Python socket specification file for PYTH
    ! RUNF : Python socket runner script for PYTH
    !===============================================================
    logical :: mlps_pt_use   = .false.
    logical :: mlps_type_use = .false.
    logical :: mlps_spec_use = .false.
    logical :: mlps_runf_use = .false.

    integer :: use_gpu = -1   ! GPU id; -1 means not requested

    integer :: all_ml_charge = 0        ! total charge of ML region (for UMA/MACE-OMOL)
    integer :: all_ml_multiplicity = 1  ! total multiplicity of ML region (for UMA/MACE-OMOL)

    ! MLP type or interface selection flags; -1 means not requested
    integer :: with_pyth = 0
    integer :: with_libtorch = 0

    integer :: with_dummy = 0

    integer :: with_tani = 0
    integer :: with_uma  = 0
    integer :: with_mace = 0
    integer :: with_dpmm = 0

    logical :: mlps_charge_use = .false.
    logical :: mlps_mult_use   = .false.
    logical :: mlps_mxmm_use   = .false.
    logical :: mlps_cutoff_use = .false.
    logical :: mlps_deps_use   = .false.
    logical :: mlps_scale_use  = .false.
    
    integer :: mlps_pt_len   = 0
    integer :: mlps_type_len = 0
    integer :: mlps_spec_len = 0 ! optional for PYTH
    integer :: mlps_runf_len = 0 ! runner script for PYTH

    character(len=mlps_pt_max) :: mlps_ptname   = ''
    character(len=mlps_pt_max) :: mlps_type   = ''    
    character(len=mlps_pt_max) :: mlps_specname = ''  ! optional for PYTH
    character(len=mlps_pt_max) :: mlps_runfname = ''  ! runner script for PYTH

    character(kind=c_char), allocatable :: mlps_ptname_c(:)
    character(kind=c_char), allocatable :: mlps_type_c(:)
    character(kind=c_char), allocatable :: mlps_specname_c(:) ! optional for PYTH
    character(kind=c_char), allocatable :: mlps_runfname_c(:) ! runner script for PYTH

    !===============================================================
    ! C-side mirrors
    !===============================================================
    integer(c_int) :: mlnum_c       = 0_c_int
    integer(c_int) :: mlnm2_c       = 0_c_int
    integer(c_int) :: ml_mm_max_c   = 5000_c_int
    integer(c_int) :: ml_num_mm_c   = 0_c_int
    integer(c_int) :: natom_cache_c = 0_c_int

    integer(c_int) :: use_gpu_c = -1_c_int

    integer(c_int) :: all_ml_charge_c = 0_c_int
    integer(c_int) :: all_ml_multiplicity_c = 1_c_int

    integer(c_int) :: with_pyth_c = 0_c_int
    integer(c_int) :: with_libtorch_c = 0_c_int

    integer(c_int) :: with_dummy_c = 0_c_int

    integer(c_int) :: with_tani_c = 0_c_int
    integer(c_int) :: with_uma_c = 0_c_int
    integer(c_int) :: with_mace_c = 0_c_int
    integer(c_int) :: with_dpmm_c = 0_c_int

    integer(c_int), allocatable :: ml_ipt_c(:)
    integer(c_int), allocatable :: ml_inm_c(:)
    integer(c_int), allocatable :: ml_idx_c(:)
    integer(c_int), allocatable :: ml_qmZid_c(:)
    integer(c_int), allocatable :: ml_qmSid_c(:)
    integer(c_int), allocatable :: ml_imask_c(:)

    !===============================================================
    ! ML energy / gradient arrays (Fortran and C mirrors)
    !===============================================================
    real(c_double) :: E_mlps_c = 0.0d0

    real(chm_real) :: mlps_cutoff = 0.0_chm_real

    real(c_double), allocatable :: mlx_mlps_c(:),  mly_mlps_c(:),  mlz_mlps_c(:)
    real(c_double), allocatable :: mldx_mlps_c(:), mldy_mlps_c(:), mldz_mlps_c(:)

    integer(c_int), allocatable :: mmidx_mlps_c(:)
    real(c_double), allocatable :: mmx_mlps_c(:),  mmy_mlps_c(:),  mmz_mlps_c(:),  mmcg_mlps_c(:)
    real(c_double), allocatable :: mmdx_mlps_c(:), mmdy_mlps_c(:), mmdz_mlps_c(:)

    !===============================================================
    ! Polarization component options, energy and gradient types
    !===============================================================

    integer :: with_pol_op_1 = 0 ! for induced polarization energy/gradient type 1 ! pol-op1
    real(chm_real), allocatable :: ml_atom_pol(:) ! atomic polarization values for each atom type ! pol-op1
    real(chm_real) :: dielec_eps = 1.0_chm_real ! dielectric constant for induced polarization energy/gradient type 1 ! pol-op1

    real(chm_real) :: ml_pol_energy = 0.0_chm_real ! induced polarization energy for type 1 ! pol-op1

    real(chm_real), allocatable :: mlx_mlps(:),  mly_mlps(:),  mlz_mlps(:)
    real(chm_real), allocatable :: mldx_pol_mlps(:), mldy_pol_mlps(:), mldz_pol_mlps(:) ! induced polarization gradients for type 1 ! pol-op1

    integer, allocatable :: mmidx_mlps(:)
    real(chm_real), allocatable :: mmx_mlps(:),  mmy_mlps(:),  mmz_mlps(:),  mmcg_mlps(:)
    real(chm_real), allocatable :: mmdx_pol_mlps(:), mmdy_pol_mlps(:), mmdz_pol_mlps(:)

    !===============================================================
    ! Scale energy and gradients
    !===============================================================
    real(chm_real) :: mlps_ene_scale = 1.0_chm_real


    !===============================================================
    ! Error handling and state flags
    !===============================================================
    character(len=1024) :: error_message   = ''
    logical :: mlps_qerror = .false.
    integer(c_int) :: mlps_ini_abi_err_c = 0_c_int

    !===============================================================
    ! Turn off MLMM
    !===============================================================
    logical :: mlps_off = .false.


contains


    !===============================================================
    ! Helper: build a null-terminated C string from Fortran string
    !===============================================================
    subroutine to_c_string(fstr, flen, cstr)
        implicit none
        character(len=*), intent(in) :: fstr
        integer,          intent(in) :: flen
        character(kind=c_char), allocatable, intent(inout) :: cstr(:)

        integer :: i, n

        if (mlps_qerror) return

        if (allocated(cstr)) deallocate(cstr)

        if (flen < 0) then
            error_message = ' MLMM> Internal error: negative C-string length.'
            mlps_qerror = .true.
            return
        end if

        if (flen > len(fstr)) then
            error_message = ' MLMM> Internal error: C-string length exceeds Fortran string length.'
            mlps_qerror = .true.
            return
        end if

        n = flen + 1
        allocate(cstr(n))

        do i = 1, flen
            cstr(i) = char(iachar(fstr(i:i)), kind=c_char)
        end do
        cstr(n) = c_null_char
    end subroutine to_c_string


    !===============================================================
    ! Helper: read a filename/script token, strip quotes, and convert
    ! to a null-terminated C string.
    !===============================================================
    subroutine read_clean_c_string(comlyn, comlen, label, fname, flen, cstr)
        use string
        implicit none

        character(len=*), intent(inout) :: comlyn
        integer,          intent(inout) :: comlen
        character(len=*), intent(in)    :: label
        character(len=*), intent(inout) :: fname
        integer,          intent(inout) :: flen
        character(kind=c_char), allocatable, intent(inout) :: cstr(:)

        character(len=mlps_pt_max) :: cleaned
        integer :: i, j

        if (mlps_qerror) return
        call nextwd(comlyn, comlen, fname, len(fname), flen)

        j = 0
        cleaned = ' '
        do i = 1, flen
            if (fname(i:i) /= '"' .and. fname(i:i) /= "'") then
                j = j + 1
                if (j > len(cleaned)) then
                    error_message = ' MLMM> '//trim(label)//' value is too long.'
                    mlps_qerror = .true.
                    return
                end if
                cleaned(j:j) = fname(i:i)
            end if
        end do

        fname = cleaned
        flen  = j

        if (flen <= 0) then
            error_message = ' MLMM> '//trim(label)//' requires a non-empty value.'
            mlps_qerror = .true.
            return
        end if

        call to_c_string(fname, flen, cstr)
    end subroutine read_clean_c_string


    !===============================================================
    ! Debug helper: convert a null-terminated C char array into
    ! an allocatable Fortran string
    !===============================================================
    subroutine cstr_to_fstr(carr, fstr)
        use iso_c_binding, only: c_char, c_null_char
        implicit none

        character(kind=c_char), dimension(:), intent(in) :: carr
        character(len=:), allocatable, intent(out)       :: fstr

        integer :: i, n
        if (mlps_qerror) return
        n = 0
        do i = 1, size(carr)
            if (carr(i) == c_null_char) exit
            n = n + 1
        end do

        if (n <= 0) then
            allocate(character(len=0) :: fstr)
        else
            allocate(character(len=n) :: fstr)
            do i = 1, n
                fstr(i:i) = achar(iachar(carr(i)))
            end do
        end if
    end subroutine cstr_to_fstr


    !===============================================================
    ! Deallocate C-side ML coordinate/gradient buffers
    !===============================================================
    subroutine deallocate_mlps_ml_buffer_array_c()
        implicit none

        if (allocated(mlx_mlps_c))  deallocate(mlx_mlps_c)
        if (allocated(mly_mlps_c))  deallocate(mly_mlps_c)
        if (allocated(mlz_mlps_c))  deallocate(mlz_mlps_c)
        if (allocated(mldx_mlps_c)) deallocate(mldx_mlps_c)
        if (allocated(mldy_mlps_c)) deallocate(mldy_mlps_c)
        if (allocated(mldz_mlps_c)) deallocate(mldz_mlps_c)

    end subroutine deallocate_mlps_ml_buffer_array_c


    !===============================================================
    ! Polarzation component: pol-op1: deallocate polarization gradient buffers for type 1
    ! Deallocate CHM-REAL and C-side ML coordinate/gradient buffers
    !===============================================================
    subroutine deallocate_mlps_pol_buffer_array_chmreal()
        implicit none
        if (mlps_qerror) return
        ! deallocate polarization gradient buffers for type 1
        if (allocated(mlx_mlps))  deallocate(mlx_mlps)
        if (allocated(mly_mlps))  deallocate(mly_mlps)
        if (allocated(mlz_mlps))  deallocate(mlz_mlps)
        if (allocated(mldx_pol_mlps)) deallocate(mldx_pol_mlps)
        if (allocated(mldy_pol_mlps)) deallocate(mldy_pol_mlps)
        if (allocated(mldz_pol_mlps)) deallocate(mldz_pol_mlps)
    end subroutine deallocate_mlps_pol_buffer_array_chmreal    


    !===============================================================
    ! Deallocate CHM REAL Fortran-side MM coordinate/gradient buffers ! pol-op1
    !===============================================================
    subroutine deallocate_mlps_mm_pol_buffer_array_chmreal()
        implicit none
        if (mlps_qerror) return
        if (allocated(mmx_mlps))   deallocate(mmx_mlps)
        if (allocated(mmy_mlps))   deallocate(mmy_mlps)
        if (allocated(mmz_mlps))   deallocate(mmz_mlps)
        if (allocated(mmidx_mlps)) deallocate(mmidx_mlps)
        if (allocated(mmcg_mlps))  deallocate(mmcg_mlps)
        if (allocated(mmdx_pol_mlps))  deallocate(mmdx_pol_mlps)
        if (allocated(mmdy_pol_mlps))  deallocate(mmdy_pol_mlps)
        if (allocated(mmdz_pol_mlps))  deallocate(mmdz_pol_mlps)
    end subroutine deallocate_mlps_mm_pol_buffer_array_chmreal

    !===============================================================
    ! Deallocate C-side MM coordinate/gradient buffers
    !===============================================================
    subroutine deallocate_mlps_mm_buffer_array_c()
        implicit none
        if (mlps_qerror) return
        if (allocated(mmx_mlps_c))   deallocate(mmx_mlps_c)
        if (allocated(mmy_mlps_c))   deallocate(mmy_mlps_c)
        if (allocated(mmz_mlps_c))   deallocate(mmz_mlps_c)
        if (allocated(mmidx_mlps_c)) deallocate(mmidx_mlps_c)
        if (allocated(mmcg_mlps_c))  deallocate(mmcg_mlps_c)
        if (allocated(mmdx_mlps_c))  deallocate(mmdx_mlps_c)
        if (allocated(mmdy_mlps_c))  deallocate(mmdy_mlps_c)
        if (allocated(mmdz_mlps_c))  deallocate(mmdz_mlps_c)
    end subroutine deallocate_mlps_mm_buffer_array_c  

    !===============================================================
    ! Allocate / deallocate ML-related arrays and reset module state
    !===============================================================
    subroutine mlps_memory_allocate(natom, qallocate)
        implicit none
        integer, intent(in) :: natom
        logical, intent(in) :: qallocate

        integer :: ier, tmp

        if (mlps_qerror) return
        if (qallocate) mlps_off = .false.
        mlps_use = .false.

        ier         = 0
        mlnum       = 0
        mlnm2       = 0
        ml_mm_max   = 5000
        ml_num_mm   = 0
        natom_cache = natom

        use_gpu = -1

        all_ml_charge = 0
        all_ml_multiplicity = 1

        with_pyth = 0
        with_libtorch = 0

        with_dummy = 0

        with_tani = 0
        with_uma  = 0
        with_mace = 0
        with_dpmm = 0

        mlps_charge_use = .false.
        mlps_mult_use   = .false.
        mlps_mxmm_use   = .false.
        mlps_cutoff_use = .false.
        mlps_deps_use   = .false.
        mlps_scale_use  = .false.

        with_pol_op_1 = 0 ! for induced polarization energy/gradient type 1 ! pol-op1
        dielec_eps = 1.0_chm_real ! dielectric constant for induced polarization energy/gradient type 1 ! pol-op1

        mlps_pt_use   = .false.
        mlps_type_use = .false.
        mlps_spec_use = .false.
        mlps_runf_use = .false.

        mlps_pt_len   = 0
        mlps_type_len = 0
        mlps_spec_len = 0
        mlps_runf_len = 0

        mlps_ptname   = ''
        mlps_type     = ''
        mlps_specname = ''
        mlps_runfname = ''

        E_mlps_c = 0.0d0

        ml_pol_energy = 0.0_chm_real ! induced polarization energy for type 1 ! pol-op1

        mlps_cutoff = 0.0_chm_real
        mlps_ene_scale = 1.0_chm_real

        error_message   = ''
        mlps_qerror = .false.
        mlps_ini_abi_err_c = 0_c_int

        if (allocated(mlps_ptname_c))   deallocate(mlps_ptname_c)
        if (allocated(mlps_type_c))     deallocate(mlps_type_c)
        if (allocated(mlps_specname_c)) deallocate(mlps_specname_c)
        if (allocated(mlps_runfname_c)) deallocate(mlps_runfname_c)

        if (allocated(ml_ipt))   deallocate(ml_ipt)
        if (allocated(ml_inm))   deallocate(ml_inm)
        if (allocated(ml_imask)) deallocate(ml_imask)
        if (allocated(ml_idx))   deallocate(ml_idx)
        if (allocated(ml_qmZid)) deallocate(ml_qmZid)
        if (allocated(ml_qmSid)) deallocate(ml_qmSid)
        if (allocated(ml_atom_pol)) deallocate(ml_atom_pol) ! for atomic polarization list type 1 ! pol-op1

        if (allocated(ml_ipt_c))   deallocate(ml_ipt_c)
        if (allocated(ml_inm_c))   deallocate(ml_inm_c)
        if (allocated(ml_imask_c)) deallocate(ml_imask_c)
        if (allocated(ml_idx_c))   deallocate(ml_idx_c)
        if (allocated(ml_qmZid_c)) deallocate(ml_qmZid_c)
        if (allocated(ml_qmSid_c)) deallocate(ml_qmSid_c)

        call deallocate_mlps_ml_buffer_array_c()
        call deallocate_mlps_mm_buffer_array_c()
        call deallocate_mlps_pol_buffer_array_chmreal()
        call deallocate_mlps_mm_pol_buffer_array_chmreal()

        if (qallocate) then
            allocate(ml_ipt(mlmax),   stat=tmp); if (tmp /= 0) ier = tmp
            allocate(ml_inm(mlmax),   stat=tmp); if (tmp /= 0) ier = tmp
            allocate(ml_imask(natom), stat=tmp); if (tmp /= 0) ier = tmp
            allocate(ml_idx(mlmax),   stat=tmp); if (tmp /= 0) ier = tmp
            allocate(ml_qmZid(mlmax), stat=tmp); if (tmp /= 0) ier = tmp
            allocate(ml_qmSid(mlmax), stat=tmp); if (tmp /= 0) ier = tmp
            allocate(ml_atom_pol(mlmax), stat=tmp); if (tmp /= 0) ier = tmp ! for atomic polarization list type 1 ! pol-op1

            if (ier /= 0) then
                if(prnlev >= 2) write(OUTU,*) ' MLMM> Allocation error in mlps_memory_allocate.'
                error_message = ' MLMM> Failed to allocate ML arrays.'
                mlps_qerror = .true.
                return
            else if (mlps_debug) then
                if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> Allocated ML arrays; natom=', natom, ', mlmax=', mlmax
            end if

            ml_ipt   = 0
            ml_inm   = 0
            ml_imask = 0
            ml_idx   = 0
            ml_qmZid = 0
            ml_qmSid = 0
            ml_atom_pol = 0.0_chm_real ! for atomic polarization list type 1 ! pol-op1
        end if
    end subroutine mlps_memory_allocate


    !===============================================================
    ! DPMM Helper: sort integer array in ascending order (first n)
    !===============================================================
    subroutine sort_int_ascending(arr, n)
        implicit none
        integer, intent(inout) :: arr(:)
        integer, intent(in)    :: n

        integer :: i, j, key
        if (mlps_qerror) return
        if (n <= 1) return

        do i = 2, n
            key = arr(i)
            j   = i - 1

            do while (j >= 1)
                if (arr(j) <= key) exit
                arr(j+1) = arr(j)
                j = j - 1
            end do

            arr(j+1) = key
        end do
    end subroutine sort_int_ascending


    !===============================================================
    ! DPMM Helper: build contiguous DPMM species IDs from atomic numbers
    !===============================================================
    subroutine mlps_build_dpmm_species_map()
        implicit none

        integer, allocatable :: unique_zids(:)
        integer :: i, k, z, num_unique
        logical :: found
        if (mlps_qerror) return
        if (mlnm2 <= 0) return

        ml_qmSid(1:mlnm2) = 0

        allocate(unique_zids(mlnm2))
        num_unique = 0

        do i = 1, mlnm2
            z = ml_qmZid(i)
            found = .false.

            do k = 1, num_unique
                if (unique_zids(k) == z) then
                    found = .true.
                    exit
                end if
            end do

            if (.not. found) then
                num_unique = num_unique + 1
                unique_zids(num_unique) = z
            end if
        end do

        call sort_int_ascending(unique_zids, num_unique)

        do i = 1, mlnm2
            z = ml_qmZid(i)
            found = .false.

            do k = 1, num_unique
                if (unique_zids(k) == z) then
                    ml_qmSid(i) = k - 1
                    found = .true.
                    exit
                end if
            end do

            if (.not. found) then
                if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: Failed to map atomic number to DPMM species id.'
                error_message = ' MLMM> Failed to build DPMM species map.'
                mlps_qerror = .true.
                return
            end if
        end do

        if (mlps_debug) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> DPMM: unique Z values in ML region:'
            do k = 1, num_unique
                if(prnlev >= 2) write(OUTU,'(A,I5,A,I5)') '   ', unique_zids(k), ' mapped to ', k-1
            end do

            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> DPMM: first few ml_qmZid and ml_qmSid values:'
            do i = 1, min(mlnm2, 10)
                if(prnlev >= 2) write(OUTU,'(A,I5,A,I5,A,I5)') '   idx=', ml_idx(i), ' Z=', ml_qmZid(i), ' S=', ml_qmSid(i)
            end do
        end if

        deallocate(unique_zids)
    end subroutine mlps_build_dpmm_species_map


    !===============================================================
    ! Build ml_atom_pol with mlnm2_c size corresponding to ml_qmZid from provided list of atomic numbers and polarization values.
    ! Atomic Polarization List: build an array of atomic polarization for each atom type
    ! to be mapped from a given list of atomic number to polarization values.
    ! pol-op1
    !===============================================================
    subroutine mlps_build_qm_atom_polarization()
        implicit none

        integer :: i, z
        real(chm_real) :: pol_value
        if (mlps_qerror) return
        if (mlnm2 <= 0) return

        if (mlnm2 > mlmax) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: mlnm2 exceeds allocated ml_atom_pol size mlmax.'
            if(prnlev >= 2) write(OUTU,*) ' MLMM>        mlnm2=', mlnm2, ' mlmax=', mlmax
            error_message = ' MLMM> Failed to build QM atom polarization array.'
            mlps_qerror = .true.
            return
        end if

        ml_atom_pol(1:mlnm2) = 0.0_chm_real

        do i = 1, mlnm2
            z = ml_qmZid(i)

            select case (z)

            case (1)
                ! https://doi.org/10.1080/00268976.2018.1535143
                pol_value = 0.6678847927952  ! 4.50711 a.u. ! 0.6678847927952 angstrom*3

            case (6)
                ! https://doi.org/10.1080/00268976.2018.1535143
                pol_value = 1.6744872343 ! 11.3 a.u. ! 1.6744872343 angstrom*3

            case (7)
                ! https://doi.org/10.1080/00268976.2018.1535143
                pol_value = 1.0965668614 ! 7.4 a.u. !1.0965668614  angstrom*3

            case (8)
                ! https://doi.org/10.1080/00268976.2018.1535143
                pol_value = 0.7853789683 ! 5.3 a.u.!0.7853789683 angstrom*3

            case (12)
                ! https://doi.org/10.1080/00268976.2018.1535143
                pol_value = 10.5507514232 ! 71.2 a.u. !10.5507514232 angstrom*3

            case (15)
                ! https://doi.org/10.1080/00268976.2018.1535143
                pol_value = 3.704617775 ! 25 a.u. !3.704617775 angstrom*3

            case default
                if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: No dummy polarization value defined for atomic number.'
                if(prnlev >= 2) write(OUTU,*) ' MLMM>        idx=', ml_idx(i), ' Z=', z
                error_message = ' MLMM> Failed to map atomic number to polarization value.'
                mlps_qerror = .true.
                return

            end select

            ml_atom_pol(i) = pol_value
        end do

        if (mlps_debug) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> QM atom polarization values:'
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> first few ml_qmZid and ml_atom_pol values:'

            do i = 1, min(mlnm2, 10)
                if(prnlev >= 2) write(OUTU,'(A,I5,A,I5,A,F12.6)') &
                    '   idx=', ml_idx(i), ' Z=', ml_qmZid(i), ' pol=', ml_atom_pol(i)
            end do
        end if
    end subroutine mlps_build_qm_atom_polarization    

    
    !===============================================================
    ! Sync Fortran INTEGER arrays to C-side c_int arrays
    !===============================================================
    subroutine mlps_sync_c()
        implicit none

        integer :: i, ierr
        if (mlps_qerror) return
        mlnum_c       = int(mlnum,       kind=c_int)
        mlnm2_c       = int(mlnm2,       kind=c_int)
        ml_mm_max_c   = int(ml_mm_max,   kind=c_int)
        ml_num_mm_c   = int(ml_num_mm,   kind=c_int)
        natom_cache_c = int(natom_cache, kind=c_int)

        use_gpu_c = int(use_gpu, kind=c_int)

        all_ml_charge_c = int(all_ml_charge, kind=c_int)
        all_ml_multiplicity_c = int(all_ml_multiplicity, kind=c_int)

        with_pyth_c = int(with_pyth, kind=c_int)
        with_libtorch_c = int(with_libtorch, kind=c_int)

        with_dummy_c = int(with_dummy, kind=c_int)

        with_tani_c = int(with_tani, kind=c_int)
        with_uma_c  = int(with_uma,  kind=c_int)
        with_mace_c = int(with_mace, kind=c_int)
        with_dpmm_c = int(with_dpmm, kind=c_int)

        if (allocated(ml_ipt_c))   deallocate(ml_ipt_c)
        if (allocated(ml_inm_c))   deallocate(ml_inm_c)
        if (allocated(ml_imask_c)) deallocate(ml_imask_c)
        if (allocated(ml_idx_c))   deallocate(ml_idx_c)
        if (allocated(ml_qmZid_c)) deallocate(ml_qmZid_c)
        if (allocated(ml_qmSid_c)) deallocate(ml_qmSid_c)

        ierr = 0
        allocate(ml_ipt_c(mlmax),         stat=ierr)
        if (ierr == 0) allocate(ml_inm_c(mlmax),         stat=ierr)
        if (ierr == 0) allocate(ml_imask_c(natom_cache), stat=ierr)
        if (ierr == 0) allocate(ml_idx_c(mlnm2),         stat=ierr)
        if (ierr == 0) allocate(ml_qmZid_c(mlnm2),       stat=ierr)
        if (ierr == 0) allocate(ml_qmSid_c(mlnm2),       stat=ierr)

        if (ierr /= 0) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: Allocation failure in mlps_sync_c.'
            error_message = ' MLMM> Could not allocate C-side MLPS arrays.'
            mlps_qerror = .true.
            return
        end if

        do i = 1, mlmax
            ml_ipt_c(i) = int(ml_ipt(i), kind=c_int)
            ml_inm_c(i) = int(ml_inm(i), kind=c_int)
        end do

        do i = 1, natom_cache
            ml_imask_c(i) = int(ml_imask(i), kind=c_int)
        end do

        do i = 1, mlnm2
            ml_idx_c(i)   = int(ml_idx(i),   kind=c_int)
            ml_qmZid_c(i) = int(ml_qmZid(i), kind=c_int)
            ml_qmSid_c(i) = int(ml_qmSid(i), kind=c_int)
        end do

        if (mlps_debug) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> Finished mlps_sync_c():'
            if(prnlev >= 2) write(OUTU,*) '            mlnum_c        = ', mlnum_c
            if(prnlev >= 2) write(OUTU,*) '            mlnm2_c        = ', mlnm2_c
            if(prnlev >= 2) write(OUTU,*) '            natom_cache_c  = ', natom_cache_c
            if(prnlev >= 2) write(OUTU,*) '            use_gpu_c      = ', use_gpu_c
            if(prnlev >= 2) write(OUTU,*) '            with_pyth_c    = ', with_pyth_c
            if(prnlev >= 2) write(OUTU,*) '            with_libtorch_c = ', with_libtorch_c
            if(prnlev >= 2) write(OUTU,*) '            with_dummy_c   = ', with_dummy_c
            if(prnlev >= 2) write(OUTU,*) '            with_tani_c    = ', with_tani_c
            if(prnlev >= 2) write(OUTU,*) '            with_uma_c     = ', with_uma_c
            if(prnlev >= 2) write(OUTU,*) '            with_mace_c    = ', with_mace_c
            if(prnlev >= 2) write(OUTU,*) '            with_dpmm_c    = ', with_dpmm_c
            if(prnlev >= 2) write(OUTU,*) '            ml_mm_max_c    = ', ml_mm_max_c
            if(prnlev >= 2) write(OUTU,*) '            ml_num_mm_c    = ', ml_num_mm_c
        end if
    end subroutine mlps_sync_c


    subroutine allocate_mlps_ml_buffer_array_c(n_ml)
        implicit none
        integer, intent(in) :: n_ml

        integer :: ierr
        if (mlps_qerror) return
        if (n_ml <= 0) then
            error_message = ' MLMM> ML buffer allocation requires a positive ML atom count.'
            mlps_qerror = .true.
            return
        end if

        call deallocate_mlps_ml_buffer_array_c()

        ierr = 0
        allocate(mlx_mlps_c(n_ml),  stat=ierr)
        if (ierr == 0) allocate(mly_mlps_c(n_ml),  stat=ierr)
        if (ierr == 0) allocate(mlz_mlps_c(n_ml),  stat=ierr)
        if (ierr == 0) allocate(mldx_mlps_c(n_ml), stat=ierr)
        if (ierr == 0) allocate(mldy_mlps_c(n_ml), stat=ierr)
        if (ierr == 0) allocate(mldz_mlps_c(n_ml), stat=ierr)

        if (ierr /= 0) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: Allocation failure in allocate_mlps_ml_buffer_array_c.'
            error_message = ' MLMM> Could not allocate C-side MLPS buffer arrays.'
            mlps_qerror = .true.
            return
        end if

        mlx_mlps_c  = 0.0d0
        mly_mlps_c  = 0.0d0
        mlz_mlps_c  = 0.0d0
        mldx_mlps_c = 0.0d0
        mldy_mlps_c = 0.0d0
        mldz_mlps_c = 0.0d0
    end subroutine allocate_mlps_ml_buffer_array_c


    subroutine allocate_mlps_mm_buffer_array_c(n_mm)
        implicit none
        integer, intent(in) :: n_mm

        integer :: ierr
        if (mlps_qerror) return
        if (n_mm <= 0) then
            error_message = ' MLMM> MM buffer allocation requires a positive MM neighbor count.'
            mlps_qerror = .true.
            return
        end if

        call deallocate_mlps_mm_buffer_array_c()

        ierr = 0
        allocate(mmx_mlps_c(n_mm),    stat=ierr)
        if (ierr == 0) allocate(mmy_mlps_c(n_mm),    stat=ierr)
        if (ierr == 0) allocate(mmz_mlps_c(n_mm),    stat=ierr)
        if (ierr == 0) allocate(mmidx_mlps_c(n_mm),  stat=ierr)
        if (ierr == 0) allocate(mmcg_mlps_c(n_mm),   stat=ierr)
        if (ierr == 0) allocate(mmdx_mlps_c(n_mm),   stat=ierr)
        if (ierr == 0) allocate(mmdy_mlps_c(n_mm),   stat=ierr)
        if (ierr == 0) allocate(mmdz_mlps_c(n_mm),   stat=ierr)

        if (ierr /= 0) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: Allocation failure in allocate_mlps_mm_buffer_array_c.'
            error_message = ' MLMM> Could not allocate C-side MM buffer arrays.'
            mlps_qerror = .true.
            return
        end if

        mmx_mlps_c   = 0.0d0
        mmy_mlps_c   = 0.0d0
        mmz_mlps_c   = 0.0d0
        mmcg_mlps_c  = 0.0d0
        mmidx_mlps_c = 0
        mmdx_mlps_c  = 0.0d0
        mmdy_mlps_c  = 0.0d0
        mmdz_mlps_c  = 0.0d0
    end subroutine allocate_mlps_mm_buffer_array_c


    !===============================================================
    ! allocate/ deallocate MLPS polarization gradient buffers for type 1 ! pol-op1
    ! both chm_real and c_double versions are allocated for Fortran and C sides
    !===============================================================
    subroutine allocate_mlps_pol_buffer_array_chmreal(n_ml)
        implicit none
        integer, intent(in) :: n_ml

        integer :: ierr
        if (mlps_qerror) return
        if (n_ml <= 0) then
            error_message = ' MLMM> Polarization energy buffer allocation requires a positive ML atom count.'
            mlps_qerror = .true.
            return
        end if

        call deallocate_mlps_pol_buffer_array_chmreal()

        ierr = 0
        allocate(mlx_mlps(n_ml),  stat=ierr)
        if (ierr == 0) allocate(mly_mlps(n_ml),  stat=ierr)
        if (ierr == 0) allocate(mlz_mlps(n_ml),  stat=ierr)
        if (ierr == 0) allocate(mldx_pol_mlps(n_ml), stat=ierr)
        if (ierr == 0) allocate(mldy_pol_mlps(n_ml), stat=ierr)
        if (ierr == 0) allocate(mldz_pol_mlps(n_ml), stat=ierr)

        if (ierr /= 0) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: Allocation failure in allocate_mlps_pol_buffer_array_chmreal.'
            error_message = ' MLMM> Could not allocate polarizationbuffer arrays.'
            mlps_qerror = .true.
            return
        end if

        mlx_mlps   = 0.0_chm_real
        mly_mlps   = 0.0_chm_real
        mlz_mlps   = 0.0_chm_real
        mldx_pol_mlps   = 0.0_chm_real
        mldy_pol_mlps   = 0.0_chm_real
        mldz_pol_mlps   = 0.0_chm_real
    end subroutine allocate_mlps_pol_buffer_array_chmreal

    subroutine allocate_mlps_mm_pol_buffer_array_chmreal(n_mm)
        implicit none
        integer, intent(in) :: n_mm

        integer :: ierr
        if (mlps_qerror) return
        if (n_mm <= 0) then
            error_message = ' MLMM> MM buffer allocation requires a positive MM neighbor count.'
            mlps_qerror = .true.
            return
        end if

        call deallocate_mlps_mm_pol_buffer_array_chmreal()

        ierr = 0
        allocate(mmx_mlps(n_mm),    stat=ierr)
        if (ierr == 0) allocate(mmy_mlps(n_mm),    stat=ierr)
        if (ierr == 0) allocate(mmz_mlps(n_mm),    stat=ierr)
        if (ierr == 0) allocate(mmidx_mlps(n_mm),  stat=ierr)
        if (ierr == 0) allocate(mmcg_mlps(n_mm),   stat=ierr)
        if (ierr == 0) allocate(mmdx_pol_mlps(n_mm),   stat=ierr)
        if (ierr == 0) allocate(mmdy_pol_mlps(n_mm),   stat=ierr)
        if (ierr == 0) allocate(mmdz_pol_mlps(n_mm),   stat=ierr)


        if (ierr /= 0) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: Allocation failure in allocate_mlps_mm_pol_buffer_array_chmreal.'
            error_message = ' MLMM> Could not allocate CHM REAL MM buffer arrays.'
            mlps_qerror = .true.
            return
        end if

        mmx_mlps   = 0.0_chm_real
        mmy_mlps   = 0.0_chm_real
        mmz_mlps   = 0.0_chm_real
        mmcg_mlps  = 0.0_chm_real
        mmidx_mlps = 0
        mmdx_pol_mlps  = 0.0_chm_real
        mmdy_pol_mlps  = 0.0_chm_real
        mmdz_pol_mlps  = 0.0_chm_real
    end subroutine allocate_mlps_mm_pol_buffer_array_chmreal

    !===============================================================
    ! Parse MLMM (MLPS) keywords:
    !
    ! Common Required:
    !   MLSL
    ! Common Optional:
    !   GPUI
    !
    ! Decide Interface backend:
    !   LIBT, PYTH
    !
    ! Decide Model Type:
    !   TANI, UMA1, MACE, DPMM, <NONE>
    !   (LIBT only supports TANI and DPMM)
    !   (PYTH only supports TANI, UMA1, MACE, and <none meaning custom with RUNF and SPEC>)
    ! Model Specification:
    !   TANI Requires: MODL
    !   DPMM Requires: MODL, MXMM
    !   UMA1 Requires: MODL, CHAR, MULT
    !   MACE Requires: MODL, CHAR, MULT
    !   <none> Requires: RUNF, SPEC
    ! Optional Additional Polarization Energy For TANI, UMA1, MACE:
    !   POL1
    !     POL1 requires: CUTF, DEPS
    !===============================================================   
    subroutine mlps_setops(comlyn, comlen)
        use dimens_fcm
        use psf
        use coord
        use select
        use string
        use linkatom, only: findel
        use rtf,      only: atct

        implicit none

        integer,          intent(inout) :: comlen
        character(len=*), intent(inout) :: comlyn

        character(len=4)  :: wd
        character(len=6)  :: ele
        character(len=32) :: mxmm_word, gpu_word, cutoff_word, charge_word, mult_word, scale_word
        character(len=32) :: deps_word ! pol-op1 dielectric constant   

        integer        :: i, nml, nread, ios
        real(chm_real) :: znum
        if (mlps_qerror) return
        do while (comlen > 0)
            wd = nexta4(comlyn, comlen)

            if (mlps_debug) then
                if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> token = "', wd, '"  (remaining comlen=', comlen, ')'
            end if

            select case (wd)

            !------------------------------------------------------
            ! MLSL: build natom-sized mask and compact ML lists
            !------------------------------------------------------
            case ('MLSL')
                if (mlnum >= 1) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> The ML region is already selected. Only one ML region is allowed.'
                else
                    call selcta(comlyn, comlen, ml_imask, x, y, z, wmain, .true.)

                    mlnum     = 1
                    mlnm2     = 0
                    ml_ipt(1) = 1
                    ml_inm(1) = 0

                    do i = 1, natom
                        if (ml_imask(i) == 1) then
                            if (mlnm2 >= mlmax) then
                                error_message = ' MLMM> Number of selected ML atoms exceeds mlmax.'
                                mlps_qerror = .true.
                                return
                            end if

                            mlnm2 = mlnm2 + 1
                            nml   = mlnm2

                            ml_idx(nml) = i

                            call findel(atct(iac(i)), amass(i), i, ele, znum, .true.)
                            ml_qmZid(nml) = int(znum)

                            ml_inm(1) = ml_inm(1) + 1
                        end if
                    end do

                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ML region selection completed with ', ml_inm(1), ' atoms.'

                    if (mlps_debug) then
                        if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> MLSL: mlnm2=', mlnm2
                        if (mlnm2 > 0) then
                            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> MLSL: first few ML atoms:'
                            do i = 1, min(mlnm2, 10)
                                if(prnlev >= 2) write(OUTU,'(A,I5,A,I7,A,I4)') '   j=', i, ' idx=', ml_idx(i), ' Z=', ml_qmZid(i)
                            end do
                        end if
                    end if
                end if

            !------------------------------------------------------
            ! MODL: model file path for TANI/DPMM
            !------------------------------------------------------
            case ('MODL')
                if (mlps_pt_use) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ML model already specified. Only one ML model is allowed.'
                else
                    call read_clean_c_string(comlyn, comlen, 'MODL', &
                                             mlps_ptname, mlps_pt_len, mlps_ptname_c)

                    mlps_pt_use = .true.
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ML model specified: ', &
                                   trim(adjustl(mlps_ptname(1:mlps_pt_len)))
                end if

            !------------------------------------------------------
            ! SPEC: optional Python socket specification file
            !------------------------------------------------------
            case ('SPEC')
                if (mlps_spec_use) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> SPEC file already specified. Only one SPEC is allowed.'
                else
                    call read_clean_c_string(comlyn, comlen, 'SPEC', &
                                             mlps_specname, mlps_spec_len, mlps_specname_c)

                    mlps_spec_use = .true.
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> Python socket SPEC file: ', &
                                   trim(adjustl(mlps_specname(1:mlps_spec_len)))
                end if

            !------------------------------------------------------
            ! RUNF: Python socket runner script
            !------------------------------------------------------
            case ('RUNF')
                if (mlps_runf_use) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> RUNF script already specified. Only one RUNF is allowed.'
                else
                    call read_clean_c_string(comlyn, comlen, 'RUNF', &
                                             mlps_runfname, mlps_runf_len, mlps_runfname_c)

                    mlps_runf_use = .true.
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> Python socket RUNF script: ', &
                                   trim(adjustl(mlps_runfname(1:mlps_runf_len)))
                end if

            !------------------------------------------------------
            ! MXMM: maximum MM input slots, including real+padded atoms
            !------------------------------------------------------
            case ('MXMM')
                if(prnlev >= 2) write(OUTU,*) ' MLMM> MXMM keyword is considered for DPMM.'

                call nextwd(comlyn, comlen, mxmm_word, len(mxmm_word), nread)

                if (nread <= 0) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: MXMM requires an integer value.'
                    error_message = ' MLMM> Missing value for MXMM.'
                    mlps_qerror = .true.
                    return
                end if

                read(mxmm_word(1:nread), *, iostat=ios) ml_num_mm
                if (ios /= 0) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: Invalid MXMM value: ', trim(mxmm_word(1:nread))
                    error_message = ' MLMM> Failed to read MXMM value.'
                    mlps_qerror = .true.
                    return
                end if

                if (ml_num_mm <= 0) then
                    error_message = ' MLMM> MXMM must be a positive integer.'
                    mlps_qerror = .true.
                    return
                end if

                if (ml_num_mm > ml_mm_max) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> WARNING: Specified MXMM value exceeds ml_mm_max=', ml_mm_max
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> Setting ml_num_mm to ml_mm_max.'
                    ml_num_mm = ml_mm_max
                end if

                mlps_mxmm_use = .true.

                if(prnlev >= 2) write(OUTU,*) ' MLMM> maximum MM input slots set to ', ml_num_mm
                if(prnlev >= 2) write(OUTU,*) ' MLMM> real MM atoms will be padded up to this size when needed.'

            !------------------------------------------------------
            ! CUTF: MM neighbor cutoff for DPMM
            !------------------------------------------------------
            case ('CUTF')
                if(prnlev >= 2) write(OUTU,*) ' MLMM> CUTF keyword is considered for DPMM.'

                call nextwd(comlyn, comlen, cutoff_word, len(cutoff_word), nread)

                if (nread <= 0) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: CUTF requires a real value.'
                    error_message = ' MLMM> Missing value for CUTF.'
                    mlps_qerror = .true.
                    return
                end if

                read(cutoff_word(1:nread), *, iostat=ios) mlps_cutoff
                if (ios /= 0) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: Invalid CUTF value: ', trim(cutoff_word(1:nread))
                    error_message = ' MLMM> Failed to read CUTF value.'
                    mlps_qerror = .true.
                    return
                end if

                if (mlps_cutoff <= 0.0_chm_real) then
                    error_message = ' MLMM> CUTF must be a positive real value.'
                    mlps_qerror = .true.
                    return
                end if

                mlps_cutoff_use = .true.

                if(prnlev >= 2) write(OUTU,*) ' MLMM> MM cutoff distance set to ', mlps_cutoff

            !------------------------------------------------------
            ! GPUI: requested GPU id
            !------------------------------------------------------
            case ('GPUI')
                call nextwd(comlyn, comlen, gpu_word, len(gpu_word), nread)

                if (nread <= 0) then
                    error_message = ' MLMM> ERROR: GPUI requires an integer GPU id.'
                    mlps_qerror = .true.
                    return
                end if

                read(gpu_word(1:nread), *, iostat=ios) use_gpu
                if (ios /= 0) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: Invalid GPU ID value: ', trim(gpu_word(1:nread))
                    error_message = ' MLMM> Failed to read GPUI value.'
                    mlps_qerror = .true.
                    return
                end if

                if (use_gpu < 0) then
                    error_message = ' MLMM> GPUI must be a non-negative integer.'
                    mlps_qerror = .true.
                    return
                end if

                if(prnlev >= 2) write(OUTU,*) ' MLMM> GPU acceleration requested with GPU ID = ', use_gpu
                if(prnlev >= 2) write(OUTU,*) ' MLMM> BLaDE will ignore GPUId keyword and default to GPU ID requested for BLaDE acceleration'
            
            !------------------------------------------------------
            ! SCALe: optional scaling factor for all energy/gradient contributions from ML region
            !------------------------------------------------------
            case ('SCAL')
                call nextwd(comlyn, comlen, scale_word, len(scale_word), nread)

                if (nread <= 0) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: SCAL requires a real value.'
                    error_message = ' MLMM> Missing value for SCAL.'
                    mlps_qerror = .true.
                    return
                end if

                read(scale_word(1:nread), *, iostat=ios) mlps_ene_scale
                if (ios /= 0) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: Invalid SCAL value: ', trim(scale_word(1:nread))
                    error_message = ' MLMM> Failed to read SCAL value.'
                    mlps_qerror = .true.
                    return
                end if

                mlps_scale_use = .true.
                if(prnlev >= 2) write(OUTU,*) ' MLMM> The total energy/gradient contributions from the ML region will be scaled by ', mlps_ene_scale

            !------------------------------------------------------        
            ! CHARge: optional charge for UMA1 and MACE
            !------------------------------------------------------
            case ('CHAR')
                call nextwd(comlyn, comlen, charge_word, len(charge_word), nread)

                if (nread <= 0) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: CHAR requires an integer value.'
                    error_message = ' MLMM> Missing value for CHAR.'
                    mlps_qerror = .true.
                    return
                end if

                read(charge_word(1:nread), *, iostat=ios) all_ml_charge
                if (ios /= 0) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: Invalid CHAR value: ', trim(charge_word(1:nread))
                    error_message = ' MLMM> Failed to read CHAR value.'
                    mlps_qerror = .true.
                    return
                end if
                mlps_charge_use = .true.
                if(prnlev >= 2) write(OUTU,*) ' MLMM> The total charge for the ML region is set to ', all_ml_charge

            !------------------------------------------------------        
            ! MULTiplicity: optional multiplicity for UMA1 and MACE
            !------------------------------------------------------
            case ('MULT')
                call nextwd(comlyn, comlen, mult_word, len(mult_word), nread)

                if (nread <= 0) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: MULT requires an integer value.'
                    error_message = ' MLMM> Missing value for MULT.'
                    mlps_qerror = .true.
                    return
                end if

                read(mult_word(1:nread), *, iostat=ios) all_ml_multiplicity
                if (ios /= 0) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: Invalid MULT value: ', trim(mult_word(1:nread))
                    error_message = ' MLMM> Failed to read MULT value.'
                    mlps_qerror = .true.
                    return
                end if
                mlps_mult_use = .true.
                if(prnlev >= 2) write(OUTU,*) ' MLMM> The total multiplicity for the ML region is set to ', all_ml_multiplicity

            !------------------------------------------------------
            ! DEPS: Dielectric constant for induced polarization energy/gradient type 1 ! pol-op1
            !------------------------------------------------------
            case ('DEPS')
                if(prnlev >= 2) write(OUTU,*) ' MLMM> DEPS keyword is considered for PYTH with POL1 and TANI with POL1.'

                call nextwd(comlyn, comlen, deps_word, len(deps_word), nread)

                if (nread <= 0) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: DEPS requires a real value.'
                    error_message = ' MLMM> Missing value for DEPS.'
                    mlps_qerror = .true.
                    return
                end if

                read(deps_word(1:nread), *, iostat=ios) dielec_eps
                if (ios /= 0) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> ERROR: Invalid DEPS value: ', trim(deps_word(1:nread))
                    error_message = ' MLMM> Failed to read DEPS value.'
                    mlps_qerror = .true.
                    return
                end if

                if (dielec_eps <= 0.0_chm_real) then
                    error_message = ' MLMM> DEPS must be a positive real value.'
                    mlps_qerror = .true.
                    return
                end if

                mlps_deps_use = .true.
                if(prnlev >= 2) write(OUTU,*) ' MLMM> The dielectric constant for induced polarization is set to ', dielec_eps
            !------------------------------------------------------

            !------------------------------------------------------
            !POL1: Induced polarization energy/gradient type 1 ! pol-op1
            !------------------------------------------------------
            case ('POL1')
                with_pol_op_1 = 1
                if(prnlev >= 2) write(OUTU,*) ' MLMM> POL1 induced-polarization correction requested.'
                if(prnlev >= 2) write(OUTU,*) ' MLMM> POL1 buffers will be allocated after validation.'



            ! MODEL TYPES
            !------------------------------------------------------
            ! TANI: built-in TorchANI/TANI model type
            !------------------------------------------------------
            case ('TANI')
                mlps_type = 'TANI'
                mlps_type_use = .true.
                mlps_type_len = 4
                call to_c_string(mlps_type, mlps_type_len, mlps_type_c)
                with_tani = 1

                if(prnlev >= 2) write(OUTU,*) ' MLMM> ML model type set to TorchANI/TANI.'

            !------------------------------------------------------
            ! UMA1/UMA: built-in UMA Python model type
            ! CHARMM keyword may be UMA1; Python receives model_type="UMA".
            !------------------------------------------------------
            case ('UMA1', 'UMA')
                mlps_type = 'UMA'
                mlps_type_use = .true.
                mlps_type_len = 3
                call to_c_string(mlps_type, mlps_type_len, mlps_type_c)
                with_uma = 1

                if(prnlev >= 2) write(OUTU,*) ' MLMM> ML model type set to Universal Model for Atoms (UMA).'

            !------------------------------------------------------
            ! MACE: built-in MACE Python model type
            !------------------------------------------------------
            case ('MACE')
                mlps_type = 'MACE'
                mlps_type_use = .true.
                mlps_type_len = 4
                call to_c_string(mlps_type, mlps_type_len, mlps_type_c)
                with_mace = 1

                if(prnlev >= 2) write(OUTU,*) ' MLMM> ML model type set to MACE backend (MACE).'

            !------------------------------------------------------
            ! DPMM: Deep Potential MM LibTorch model type
            !------------------------------------------------------
            case ('DPMM')
                mlps_type = 'DPMM'
                mlps_type_use = .true.
                mlps_type_len = 4
                call to_c_string(mlps_type, mlps_type_len, mlps_type_c)
                with_dpmm = 1

                if(prnlev >= 2) write(OUTU,*) ' MLMM> ML model type set to Deep Potential MM (DPMM).'

            !------------------------------------------------------
            ! DUMM: dummy model type for testing
            !------------------------------------------------------
            case ('TEST', 'DUMM')
                mlps_type = 'DUMMY'
                mlps_type_use = .true.
                mlps_type_len = 5
                call to_c_string(mlps_type, mlps_type_len, mlps_type_c)
                with_dummy = 1

                if(prnlev >= 2) write(OUTU,*) ' MLMM> ML model type set to DUMMY (DUMM).'


            !------------------------------------------------------
            ! PYTH: finalizer for Python socket ML-only backend
            ! Supports UMA
            !------------------------------------------------------
            case ('PYTH')
                with_pyth =  1
                if(prnlev >= 2) write(OUTU,*) ' MLMM> ML model inference will be carried out via Python socket communication (ML-only ) (PYTH).'
            !------------------------------------------------------
            ! LIBTorch: finalizer for LibTorch C++ backend
            ! Support TorchANI (TANI) and Deep Potential MM (DPMM) TorchScript models
            !------------------------------------------------------
            case ('LIBT')
                with_libtorch = 1
                if(prnlev >= 2) write(OUTU,*) ' MLMM> ML model inference will be carried out via LibTorch C++ backend (LIBT).'

            case ('CLEA', 'OFF ')
                if(prnlev >= 2) write(OUTU,*) ' MLMM> ML model inference is disabled (OFF). '
                if(prnlev >= 2) write(OUTU,*) ' MLMM> Allocated MLMM memory buffers will be deallocated and reset.'
                mlps_off = .true.
                mlps_use = .false.
                call mlps_memory_allocate(natom, .false.)
                return


            !------------------------------------------------------
            ! Unknown MLPS keyword
            !------------------------------------------------------
            case default
                if (len_trim(wd) > 0) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> WARNING: Unknown MLPS token "', trim(wd), '" ignored.'
                end if
            end select
        end do

        if (mlps_debug) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> Exiting mlps_setops.'
        end if
    end subroutine mlps_setops

    !=======================================================================
    ! Validate MLMM MLPS setops reqiurements and exit with error if any are not met.
    !=======================================================================
    subroutine mlps_validate_setops()
        implicit none

        integer :: n_model_types
        if (mlps_qerror) return
        if (mlps_off) return
        ! Common option checks.
        if (mlnm2 <= 0) then
            error_message = ' MLMM> No ML atoms selected. MLSL keyword must be specified.'
            mlps_qerror = .true.
            return
        end if

        if (with_libtorch == 1 .and. with_pyth == 1) then
            error_message = ' MLMM> LibTorch and Python socket backends are mutually exclusive. Only one backend can be used.'
            mlps_qerror = .true.
            return
        end if

        if (with_libtorch /= 1 .and. with_pyth /= 1) then
            error_message = ' MLMM> No MLPS backend specified. Either LIBT or PYTH must be specified.'
            mlps_qerror = .true.
            return
        end if

        n_model_types = with_dummy + with_tani + with_uma + with_mace + with_dpmm

        if (n_model_types > 1) then
            error_message = ' MLMM> Multiple ML model types specified. Only one type can be used at a time.'
            mlps_qerror = .true.
            return
        end if

        if (with_libtorch == 1) then
            if (n_model_types == 0) then
                error_message = ' MLMM> LibTorch backend requires a model type: TANI or DPMM.'
                mlps_qerror = .true.
                return
            end if

            if (with_uma == 1 .or. with_mace == 1) then
                error_message = ' MLMM> LibTorch backend does not support UMA or MACE model types. Only TANI and DPMM are supported.'
                mlps_qerror = .true.
                return
            end if

            if (with_tani == 1 .or. with_dpmm == 1) then
                if(prnlev >= 2) write(OUTU,*) ' MLMM> LibTorch backend requires a model-tag as a TorchScript model file path specified with MODL keyword.'
                if(prnlev >= 2) write(OUTU,*) ' MLMM> Example: MODL "/path/to/model.pt"'
                if (.not. mlps_pt_use) then
                    error_message = ' MLMM> LibTorch backend requires a model-tag as a file path specified with MODL keyword.'
                    mlps_qerror = .true.
                    return
                end if
            end if
        end if

        if (with_pyth == 1) then

            if (with_dummy == 1) then
                ! Dummy requires neither MODL nor RUNF/SPEC.
                if (.not. allocated(mlps_ptname_c)) then
                    call to_c_string('', 0, mlps_ptname_c)
                end if
            end if

            if (with_dpmm == 1) then
                error_message = ' MLMM> Python socket backend does not support DPMM model type. Only TANI, UMA, MACE, or custom RUNF/SPEC are supported.'
                mlps_qerror = .true.
                return
            end if

            if (with_tani == 1 .or. with_uma == 1 .or. with_mace == 1) then
                if (.not. mlps_pt_use) then
                    error_message = ' MLMM> Python socket backend requires a local model/checkpoint path specified with MODL.'
                    mlps_qerror = .true.
                    return
                end if
            elseif ( with_dummy == 0 .and. with_tani == 0 .and. with_uma == 0 .and. with_mace == 0 .and. with_dpmm == 0 ) then
                ! Custom PYTH mode: no built-in model type. User supplies external RUNF and SPEC.
                if (.not. mlps_runf_use) then
                    error_message = ' MLMM> Custom PYTH backend requires a Python runner script specified with RUNF keyword.'
                    mlps_qerror = .true.
                    return
                end if
                if (.not. mlps_spec_use) then
                    if(prnlev >= 2) write(OUTU,*) ' MLMM> Custom PYTH backend supports a SPEC file specified with SPEC keyword, if needed.'
                    ! SPEC is optional for Python socket backends.  If absent,
                    ! pass an empty C string to the C++ launcher.
                    if ((with_pyth == 1) .and. .not. allocated(mlps_specname_c)) then
                        call to_c_string('', 0, mlps_specname_c)
                    end if
                end if
            end if

            if (with_uma == 1 .or. with_mace == 1) then
                if (.not. mlps_charge_use .or. .not. mlps_mult_use) then
                    error_message = ' MLMM> UMA and MACE model types require both CHAR and MULT keywords.'
                    mlps_qerror = .true.
                    return
                end if
            end if
        end if

        if (with_dpmm == 1) then
            if (.not. mlps_mxmm_use) then
                error_message = ' MLMM> DPMM requires MXMM to define the maximum MM neighbor slots.'
                mlps_qerror = .true.
                return
            end if
            if (.not. mlps_cutoff_use) then
                error_message = ' MLMM> DPMM requires CUTF to define the MM neighbor cutoff.'
                mlps_qerror = .true.
                return
            end if
        end if

        if (with_pol_op_1 == 1) then
            if (with_dpmm == 1) then
                error_message = ' MLMM> POL1 option is not compatible with DPMM model type.'
                mlps_qerror = .true.
                return
            end if
            if (.not. mlps_deps_use) then
                error_message = ' MLMM> POL1 option requires DEPS keyword.'
                mlps_qerror = .true.
                return
            end if
            if (.not. mlps_cutoff_use) then
                error_message = ' MLMM> POL1 option requires CUTF keyword.'
                mlps_qerror = .true.
                return
            end if
            if (dielec_eps <= 0.0_chm_real) then
                error_message = ' MLMM> POL1 option requires a positive dielectric constant.'
                mlps_qerror = .true.
                return
            end if
            if (mlps_cutoff <= 0.0_chm_real) then
                error_message = ' MLMM> POL1 option requires a positive cutoff distance.'
                mlps_qerror = .true.
                return
            end if
        end if
    end subroutine mlps_validate_setops


    !=======================================================================
    ! Allocate backend buffers and build derived metadata after validation.
    ! This keeps MLPS keyword order flexible: MLSL/MXMM/CUTF/POL1/model type
    ! may appear in any order, as long as the final validated command is valid.
    !=======================================================================
    subroutine mlps_finalize_after_validation()
        implicit none

        integer :: pol_mm_slots
        if (mlps_qerror) return
        if (mlps_off) return
        if (mlnm2 <= 0) then
            error_message = ' MLMM> Internal error: finalize called with no ML atoms.'
            mlps_qerror = .true.
            return
        end if

        if (with_dpmm == 1) then
            call mlps_build_dpmm_species_map()
        end if

        ! Every ML backend needs compact ML coordinate/gradient buffers.
        call allocate_mlps_ml_buffer_array_c(mlnm2)

        if (with_dpmm == 1) then
            call allocate_mlps_mm_buffer_array_c(ml_num_mm)
        else
            call deallocate_mlps_mm_buffer_array_c()
        end if

        if (with_pol_op_1 == 1) then
            call mlps_build_qm_atom_polarization()
            call allocate_mlps_pol_buffer_array_chmreal(mlnm2)

            ! POL1 needs MM neighbor storage but does not require MXMM.
            ! If MXMM was not specified, use the module hard cap.
            if (ml_num_mm <= 0) then
                ml_num_mm = ml_mm_max
            end if
            pol_mm_slots = ml_num_mm

            call allocate_mlps_mm_pol_buffer_array_chmreal(pol_mm_slots)

            if (mlps_debug) then
                if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> POL1 finalized: mlnm2=', mlnm2, ' mm_slots=', pol_mm_slots
            end if
        else
            call deallocate_mlps_pol_buffer_array_chmreal()
            call deallocate_mlps_mm_pol_buffer_array_chmreal()
        end if
    end subroutine mlps_finalize_after_validation

    !=======================================================================
    ! FULL SETUP: allocate memory, parse MLPS keywords and call C-side setup
    !=======================================================================
    subroutine mlps_setup(comlyn, comlen)
        use dimens_fcm
        use psf, only: natom
#if KEY_PARALLEL == 1
        use parallel, only: mynod, comm_charmm
        use mpi_f08, only: MPI_LOGICAL, MPI_INTEGER
#endif
        implicit none

        character(len=*), intent(inout) :: comlyn
        integer,          intent(inout) :: comlen
        integer :: ierr

        if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> CAUTION!!: MLMM Module Requested. Use BLOCK Module to scale all ML-ML atom interactions by 0.0. Refer MLMM documentation for details.'
        if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> CAUTION!!: MLMM Module Requested. Use SHAKE Module to remove all ML-ML SHAKE constraints. Refer MLMM documentation for details.'
        if (mlps_use .and. mlps_debug) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> Reinitializing existing MLPS state.'
        end if

#if KEY_PARALLEL == 1
     if (mynod == 0) then
#endif

        if (mlps_qerror) goto 1010

        call mlps_memory_allocate(natom, .true.)

        if (mlps_debug) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> Entering mlps_setops with comlen=', comlen
        end if

        call mlps_setops(comlyn, comlen)
        call mlps_validate_setops()
        call mlps_finalize_after_validation()

        if (mlps_off) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> MLMM module is turned off.'
            goto 1010
        end if

        if (mlps_debug .and. .not. mlps_off) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> Returned from mlps_setops. comlen=', comlen
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> mlnum        = ', mlnum, ', mlnm2 = ', mlnm2
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> mlps_pt_use  = ', mlps_pt_use
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> mlps_type_use = ', mlps_type_use
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> mlps_spec_use= ', mlps_spec_use
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> mlps_runf_use= ', mlps_runf_use
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> use_gpu      = ', use_gpu
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> with_dummy   = ', with_dummy
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> with_tani    = ', with_tani
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> with_uma     = ', with_uma
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> with_mace    = ', with_mace
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> with_dpmm    = ', with_dpmm
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> with_pyth    = ', with_pyth
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> with_libtorch = ', with_libtorch
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> ml_mm_max    = ', ml_mm_max
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> ml_num_mm    = ', ml_num_mm
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> with_pol_op_1 = ', with_pol_op_1
            if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> dielec_eps = ', dielec_eps
        end if

        if (mlnm2 > 0) then
            call mlps_sync_c()
        else
            if (mlps_debug) then
                if(prnlev >= 2) write(OUTU,*) ' MLMM-DBG> No ML atoms selected; skipping mlps_sync_c.'
            end if
        end if


        if (with_libtorch == 1 .and. with_dummy == 1) then
#if KEY_MLPTORCH==1
            call charmm_add_dummy( with_dummy_c, use_gpu_c, mlnm2_c, ml_idx_c, ml_qmZid_c, ml_imask_c, &
                natom_cache_c, mlps_ini_abi_err_c)
            if (mlps_ini_abi_err_c /= 0_c_int) then
                error_message = ' MLMM> LibTorch/DUMMY setup failed.'
                mlps_qerror = .true.
                goto 1010
            else
                mlps_use = .true.
            end if            
#endif
        else if (with_libtorch == 1 .and. with_tani == 1) then
#if KEY_MLPTORCH==1
            if(prnlev >= 2) write(OUTU,*) ' MLMM> TorchANI model: periodic_table_index=True should be used when preparing the model file.'
            if(prnlev >= 2) write(OUTU,*) ' MLMM> TorchANI model: Only Supports Neutral Molecules. Charged Molecules are not supported.'
            if (all_ml_charge /= 0 .or. all_ml_multiplicity /= 1) then
                error_message = ' MLMM> TorchANI model: Charged Molecules are not supported. Please set CHAR=0 and MULT=1 for neutral molecules.'
                mlps_qerror = .true.
                goto 1010
            endif
            call charmm_add_tani(with_tani_c, use_gpu_c, mlps_ptname_c, mlnm2_c, &
                                 ml_idx_c, ml_qmZid_c, ml_imask_c, natom_cache_c, mlps_ini_abi_err_c)
            if (mlps_ini_abi_err_c /= 0_c_int) then
                error_message = ' MLMM> LibTorch/TorchANI Setup Failed. Hint: Please verify Atom Selection or Model-Tag is a valid path'
                mlps_qerror = .true.
                goto 1010
            else
                mlps_use = .true.
            end if
#endif
        else if (with_libtorch == 1 .and. with_dpmm == 1) then
#if KEY_MLPTORCH==1
            call charmm_add_dpmm(with_dpmm_c, use_gpu_c, mlps_ptname_c, mlnm2_c, ml_num_mm_c, &
                                 ml_idx_c, ml_qmSid_c, ml_imask_c, natom_cache_c, mlps_ini_abi_err_c)
            if (mlps_ini_abi_err_c /= 0_c_int) then
                error_message = ' MLMM> LibTorch/DPMM Setup Failed. Hint: Please verify Atom Selection or Model-Tag is a valid path'
                mlps_qerror = .true.
                goto 1010
            else
                mlps_use = .true.
            end if
#endif

        else if (with_pyth == 1 .and. (with_dummy == 1 .or. with_tani == 1 .or. with_uma == 1 .or. with_mace == 1)) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM> Python socket backend PYTH initialized on rank 0.'
            call charmm_add_pyth(with_pyth_c, use_gpu_c, mlps_type_c, mlps_ptname_c, all_ml_charge_c, &
                                     all_ml_multiplicity_c, mlnm2_c, ml_idx_c, ml_qmZid_c, ml_imask_c, natom_cache_c, mlps_ini_abi_err_c)

            if (mlps_ini_abi_err_c /= 0_c_int) then
                error_message = ' MLMM> PYTHSocket/Model Setup Failed. Hint: Please verify Atom Selection or Model-Tag syntax is valid (eg. repo/local:<model>:<option>:<optional-path>)'
                mlps_qerror = .true.
                goto 1010
            else
                mlps_use = .true.
            end if

        else if (with_pyth == 1 .and. with_dummy == 0 .and. with_tani == 0 .and. with_uma == 0 .and. with_mace == 0 .and. with_dpmm == 0) then
            if(prnlev >= 2) write(OUTU,*) ' MLMM> Python socket backend PYTH initialized on rank 0.'
            call charmm_add_pyth_custom(with_pyth_c, use_gpu_c, mlps_specname_c, mlps_runfname_c, mlnm2_c, &
                                     ml_idx_c, ml_qmZid_c, ml_imask_c, natom_cache_c, mlps_ini_abi_err_c)
            if (mlps_ini_abi_err_c /= 0_c_int) then
                error_message = ' MLMM> PYTHSocket/Custom-Model Setup Failed. Hint: Please verify Atom Selection or RUNF/SPEC file paths are valid and accessible.'
                mlps_qerror = .true.
                goto 1010
            else
                mlps_use = .true.
            end if

        end if

1010 continue
#if KEY_PARALLEL==1
    end if
    call MPI_Bcast(mlps_qerror, 1, MPI_LOGICAL, 0,comm_charmm, ierr)
    call MPI_Bcast(mlps_off, 1, MPI_LOGICAL, 0, comm_charmm, ierr)
    call MPI_Bcast(mlps_use, 1, MPI_LOGICAL, 0,comm_charmm, ierr)
    call MPI_Bcast(with_pol_op_1, 1, MPI_INTEGER, 0,comm_charmm, ierr)
    call MPI_Bcast(mlnm2, 1, MPI_INTEGER, 0,comm_charmm, ierr)
    call MPI_Bcast(with_pyth, 1, MPI_INTEGER, 0,comm_charmm, ierr)
    call MPI_Bcast(with_libtorch, 1, MPI_INTEGER, 0,comm_charmm, ierr)
    call MPI_Bcast(with_dummy, 1, MPI_INTEGER, 0, comm_charmm, ierr)
    call MPI_Bcast(with_tani, 1, MPI_INTEGER, 0,comm_charmm, ierr)
    call MPI_Bcast(with_uma, 1, MPI_INTEGER, 0,comm_charmm, ierr)
    call MPI_Bcast(with_mace, 1, MPI_INTEGER, 0,comm_charmm, ierr)
    call MPI_Bcast(with_dpmm, 1, MPI_INTEGER, 0,comm_charmm, ierr)
    if (mlps_qerror) then
        if (mynod == 0) then
            call wrndie(-5, '', trim(error_message))
        else
            call wrndie(-5, '', ' MLMM> Fatal error during MLMM setup on rank 0. See rank 0 output for details.')
        end if
    end if
#else
    if (mlps_qerror) call wrndie(-5, '', trim(error_message))
#endif

    end subroutine mlps_setup

#endif
end module mlps_ini
