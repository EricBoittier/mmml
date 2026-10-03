!> API for accessing BLOCK facility configuration data
!>
!> This module provides C-bound functions for reading and writing
!> BLOCK facility configuration data including coefficient matrix,
!> LDIN parameters, and various settings.
!>
!> These functions complement api_lambdata and api_msldata which
!> collect dynamics trajectory data. This module provides access
!> to the setup/configuration state.
module api_blockdata
  implicit none

#if KEY_BLOCK == 1
contains

  !> @brief Check if BLOCK facility is active
  !> @return 1 if active, 0 otherwise
  function blockdata_is_active() bind(c) result(is_active)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: nblock
    implicit none
    integer(c_int) :: is_active

    is_active = 0
    if (nblock > 0) is_active = 1
  end function blockdata_is_active

  !> @brief Get number of blocks
  !> @return Number of blocks (nblock)
  function blockdata_get_nblock() bind(c) result(out_nblock)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: nblock
    implicit none
    integer(c_int) :: out_nblock

    out_nblock = nblock
  end function blockdata_get_nblock

  !> @brief Get number of biasing potentials
  !> @return Number of biasing potentials (nbiasv)
  function blockdata_get_nbiasv() bind(c) result(out_nbiasv)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: nbiasv
    implicit none
    integer(c_int) :: out_nbiasv

    out_nbiasv = nbiasv
  end function blockdata_get_nbiasv

  !> @brief Get coefficient matrix dimensions
  !> @param[out] nrows Number of rows
  !> @param[out] ncols Number of columns
  subroutine blockdata_coef_get_dims(nrows, ncols) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: nblock
    implicit none
    integer(c_int), intent(out) :: nrows, ncols

    nrows = nblock
    ncols = nblock
  end subroutine blockdata_coef_get_dims

  !> @brief Get coefficient matrix (flattened row-major)
  !> @param[out] coefs Flattened coefficient matrix (nblock x nblock)
  subroutine blockdata_coef_get(coefs) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use chm_kinds, only: chm_real
    use lambdam, only: nblock, fullblcoep
    implicit none
    real(c_double), dimension(*), intent(out) :: coefs
    integer :: i, j, idx

    idx = 1
    do i = 1, nblock
      do j = 1, nblock
        coefs(idx) = real(fullblcoep(i, j), c_double)
        idx = idx + 1
      end do
    end do
  end subroutine blockdata_coef_get

  !> @brief Set a single coefficient value
  !> @param[in] i Block index i (1-based)
  !> @param[in] j Block index j (1-based)
  !> @param[in] value Coefficient value
  subroutine blockdata_coef_set(i, j, value) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use chm_kinds, only: chm_real
    use lambdam, only: fullblcoep
#if KEY_BLADE==1
    use blade_main, only: system_dirty
#endif
#if KEY_OPENMM==1
    use omm_ctrl, only: omm_system_changed
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    integer(c_int), value, intent(in) :: i, j
    real(c_double), value, intent(in) :: value

    fullblcoep(i, j) = real(value, chm_real)

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
  end subroutine blockdata_coef_set

  !> @brief Get lambda values for all blocks (bixlam)
  !> @param[out] lambdas Array of lambda values (size nblock)
  subroutine blockdata_lambda_get(lambdas) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use chm_kinds, only: chm_real
    use lambdam, only: nblock, bixlam
    implicit none
    real(c_double), dimension(*), intent(out) :: lambdas
    integer :: i

    do i = 1, nblock
      lambdas(i) = real(bixlam(i), c_double)
    end do
  end subroutine blockdata_lambda_get

  !> @brief Get lambda-squared values for all blocks (bixlam^2)
  !> @param[out] lamsq Array of lambda^2 values (size nblock)
  subroutine blockdata_lamsq_get(lamsq) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use chm_kinds, only: chm_real
    use lambdam, only: nblock, bixlam
    implicit none
    real(c_double), dimension(*), intent(out) :: lamsq
    integer :: i

    do i = 1, nblock
      lamsq(i) = real(bixlam(i) * bixlam(i), c_double)
    end do
  end subroutine blockdata_lamsq_get

  !> @brief Get LDIN parameters for a specific block
  !> @param[in] block_id Block number (1-based)
  !> @param[out] lam0 Current lambda^2 value (bixlam^2)
  !> @param[out] vel Lambda velocity (bivlam)
  !> @param[out] mass Lambda mass (bimlam)
  !> @param[out] bias Biasing energy (bielam)
  !> @param[out] friction Friction coefficient (biblam)
  subroutine blockdata_ldin_get(block_id, lam0, vel, mass, bias, friction) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use chm_kinds, only: chm_real
    use lambdam, only: bixlam, bivlam, bimlam, bielam, biblam
    implicit none
    integer(c_int), value, intent(in) :: block_id
    real(c_double), intent(out) :: lam0, vel, mass, bias, friction

    lam0 = real(bixlam(block_id) * bixlam(block_id), c_double)
    vel = real(bivlam(block_id), c_double)
    mass = real(bimlam(block_id), c_double)
    bias = real(bielam(block_id), c_double)
    friction = real(biblam(block_id), c_double)
  end subroutine blockdata_ldin_get

  !> @brief Set LDIN parameters for a specific block
  !> @param[in] block_id Block number (1-based)
  !> @param[in] lam0 Lambda^2 value (lambda = sqrt(lam0))
  !> @param[in] vel Lambda velocity (bivlam)
  !> @param[in] mass Lambda mass (bimlam)
  !> @param[in] bias Biasing energy (bielam)
  !> @param[in] friction Friction coefficient (biblam)
  subroutine blockdata_ldin_set(block_id, lam0, vel, mass, bias, friction) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use chm_kinds, only: chm_real
    use lambdam, only: nblock, bixlam, bivlam, bimlam, bielam, biblam
#if KEY_BLADE==1
    use blade_main, only: system_dirty
#endif
#if KEY_OPENMM==1
    use omm_ctrl, only: omm_system_changed
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    integer(c_int), value, intent(in) :: block_id
    real(c_double), value, intent(in) :: lam0, vel, mass, bias, friction

    if (block_id < 1 .or. block_id > nblock) return
    if (lam0 < 0.0d0) return
    if (.not. allocated(bixlam) .or. .not. allocated(bivlam) .or. &
        .not. allocated(bimlam) .or. .not. allocated(bielam) .or. &
        .not. allocated(biblam)) return

    bixlam(block_id) = sqrt(real(lam0, chm_real))
    bivlam(block_id) = real(vel, chm_real)
    bimlam(block_id) = real(mass, chm_real)
    bielam(block_id) = real(bias, chm_real)
    biblam(block_id) = real(friction, chm_real)

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
  end subroutine blockdata_ldin_set

  !> @brief Get FFIX (fixed lambda) flags for all blocks
  !> @param[out] flags Array of flags (size nblock): 1=fixed, 0=dynamic
  !> @param[out] status 0=success, -1=qlfix not allocated
  subroutine blockdata_ffix_get(flags, status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: nblock, qlfix
    implicit none
    integer(c_int), dimension(*), intent(out) :: flags
    integer(c_int), intent(out) :: status
    integer :: i

    if (.not. allocated(qlfix)) then
      status = -1
      return
    endif

    do i = 1, nblock
      if (qlfix(i)) then
        flags(i) = 1
      else
        flags(i) = 0
      endif
    end do
    status = 0
  end subroutine blockdata_ffix_get

  !> @brief Set FFIX (fixed lambda) flag for a specific block
  !> @param[in] block_id Block number (1-based)
  !> @param[in] is_fixed 1=fixed, 0=dynamic
  !> @param[out] status 0=success, -1=qlfix not allocated, -2=invalid block_id
  subroutine blockdata_ffix_set(block_id, is_fixed, status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: nblock, qlfix
#if KEY_BLADE==1
    use blade_main, only: system_dirty
#endif
#if KEY_OPENMM==1
    use omm_ctrl, only: omm_system_changed
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    integer(c_int), value, intent(in) :: block_id, is_fixed
    integer(c_int), intent(out) :: status

    if (.not. allocated(qlfix)) then
      status = -1
      return
    endif

    if (block_id < 1 .or. block_id > nblock) then
      status = -2
      return
    endif

    qlfix(block_id) = (is_fixed /= 0)
    status = 0

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
  end subroutine blockdata_ffix_set

  !> @brief Get friction coefficients for all blocks (biblam array)
  !> @param[out] frictions Array of friction values (size nblock)
  !> @param[out] status 0=success, -1=biblam not allocated
  subroutine blockdata_friction_get(frictions, status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use chm_kinds, only: chm_real
    use lambdam, only: nblock, biblam
    implicit none
    real(c_double), dimension(*), intent(out) :: frictions
    integer(c_int), intent(out) :: status
    integer :: i

    if (.not. allocated(biblam)) then
      status = -1
      return
    endif

    do i = 1, nblock
      frictions(i) = real(biblam(i), c_double)
    end do
    status = 0
  end subroutine blockdata_friction_get

  !> @brief Set friction coefficient for a specific block (biblam)
  !> @param[in] block_id Block number (1-based)
  !> @param[in] friction Friction coefficient value
  subroutine blockdata_friction_set(block_id, friction) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use chm_kinds, only: chm_real
    use lambdam, only: nblock, biblam
#if KEY_BLADE==1
    use blade_main, only: system_dirty
#endif
#if KEY_OPENMM==1
    use omm_ctrl, only: omm_system_changed
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    integer(c_int), value, intent(in) :: block_id
    real(c_double), value, intent(in) :: friction

    if (.not. allocated(biblam)) return
    if (block_id < 1 .or. block_id > nblock) return

    biblam(block_id) = real(friction, chm_real)

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
  end subroutine blockdata_friction_set

  !> @brief Get bias parameters for a specific bias
  !> @param[in] bias_id Bias index (1-based)
  !> @param[out] block_i First block index
  !> @param[out] block_j Second block index
  !> @param[out] cls Bias class
  !> @param[out] reup Upper reference value
  !> @param[out] rlow Lower reference value
  !> @param[out] kbias Bias force constant
  !> @param[out] pbias Bias power
  subroutine blockdata_bias_get(bias_id, block_i, block_j, cls, &
                                 reup, rlow, kbias, pbias) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use chm_kinds, only: chm_real
    use lambdam, only: ibvidi, ibvidj, ibclas, irreup, irrlow, ikbias, ipbias
    implicit none
    integer(c_int), value, intent(in) :: bias_id
    integer(c_int), intent(out) :: block_i, block_j, cls, pbias
    real(c_double), intent(out) :: reup, rlow, kbias

    block_i = ibvidi(bias_id)
    block_j = ibvidj(bias_id)
    cls = ibclas(bias_id)
    reup = real(irreup(bias_id), c_double)
    rlow = real(irrlow(bias_id), c_double)
    kbias = real(ikbias(bias_id), c_double)
    pbias = ipbias(bias_id)
  end subroutine blockdata_bias_get

  !> @brief Get temperature for lambda dynamics
  !> @return Temperature (tbld)
  function blockdata_get_temperature() bind(c) result(temp)
    use, intrinsic :: iso_c_binding, only: c_double
    use lambdam, only: tbld
    implicit none
    real(c_double) :: temp

    temp = real(tbld, c_double)
  end function blockdata_get_temperature

  !> @brief Set temperature for lambda dynamics
  !> @param[in] temp Temperature value
  subroutine blockdata_set_temperature(temp) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use chm_kinds, only: chm_real
    use lambdam, only: tbld
#if KEY_BLADE==1
    use blade_main, only: system_dirty
#endif
#if KEY_OPENMM==1
    use omm_ctrl, only: omm_system_changed
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    real(c_double), value, intent(in) :: temp

    tbld = real(temp, chm_real)

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
  end subroutine blockdata_set_temperature

  !> @brief Check if lambda dynamics is enabled
  !> @return 1 if enabled, 0 otherwise
  function blockdata_qldm_enabled() bind(c) result(enabled)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: qldm
    implicit none
    integer(c_int) :: enabled

    enabled = 0
    if (qldm) enabled = 1
  end function blockdata_qldm_enabled

  !> @brief Check if theta mode is enabled
  !> @return 1 if enabled, 0 otherwise
  function blockdata_theta_enabled() bind(c) result(enabled)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: Qthetadm
    implicit none
    integer(c_int) :: enabled

    enabled = 0
    if (Qthetadm) enabled = 1
  end function blockdata_theta_enabled

  !> @brief Check if Langevin coupling is enabled
  !> @return 1 if enabled, 0 otherwise
  function blockdata_langevin_enabled() bind(c) result(enabled)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: ILALDM
    implicit none
    integer(c_int) :: enabled

    enabled = 0
    if (ILALDM) enabled = 1
  end function blockdata_langevin_enabled

  !> @brief Get MSLD number of sites
  !> @return Number of MSLD sites (nsitemld)
  function blockdata_get_nsites() bind(c) result(nsites)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: nsitemld
    implicit none
    integer(c_int) :: nsites

    nsites = nsitemld
  end function blockdata_get_nsites

  !> @brief Get site assignment for each block
  !> @param[out] sites Array of site IDs (size nblock)
  subroutine blockdata_sites_get(sites) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: nblock, isitemld
    implicit none
    integer(c_int), dimension(*), intent(out) :: sites
    integer :: i

    do i = 1, nblock
      sites(i) = isitemld(i)
    end do
  end subroutine blockdata_sites_get

  !> @brief Get soft-core state
  !> @param[out] mode 0=off, 1=on, 2=w14
  function blockdata_softcore_mode() bind(c) result(mode)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: iqldm_softcore
    implicit none
    integer(c_int) :: mode

    mode = iqldm_softcore
  end function blockdata_softcore_mode

  !> @brief Get PME handling mode for MSLD
  !> @return PME mode: 0=off, 1=NN, 2=EX, 3=ON
  function blockdata_pme_mode() bind(c) result(mode)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: iqldm_pme
    implicit none
    integer(c_int) :: mode

    mode = iqldm_pme
  end function blockdata_pme_mode

  !> @brief Get FNEX factor for MSLD
  !> @return FNEX exponential factor
  function blockdata_get_fnex() bind(c) result(fnex)
    use, intrinsic :: iso_c_binding, only: c_double
    use lambdam, only: fnexp_factor
    implicit none
    real(c_double) :: fnex

    fnex = real(fnexp_factor, c_double)
  end function blockdata_get_fnex

  !> @brief Get pH value for constant-pH MD
  !> @return pH value (PHVAL)
  function blockdata_get_ph() bind(c) result(ph)
    use, intrinsic :: iso_c_binding, only: c_double
    use lambdam, only: PHVAL
    implicit none
    real(c_double) :: ph

    ph = real(PHVAL, c_double)
  end function blockdata_get_ph

  !> @brief Set pH value for constant-pH MD
  !> @param[in] ph pH value
  subroutine blockdata_set_ph(ph) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use chm_kinds, only: chm_real
    use lambdam, only: PHVAL
#if KEY_BLADE==1
    use blade_main, only: system_dirty
#endif
#if KEY_OPENMM==1
    use omm_ctrl, only: omm_system_changed
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    real(c_double), value, intent(in) :: ph

    PHVAL = real(ph, chm_real)

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
  end subroutine blockdata_set_ph

  !> @brief Check if MSLD is active (qmld flag)
  !> @return 1 if MSLD active, 0 otherwise
  !> @note This indicates whether fullblcoep array is allocated
  function blockdata_msld_active() bind(c) result(active)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: qmld
    implicit none
    integer(c_int) :: active

    active = 0
    if (qmld) active = 1
  end function blockdata_msld_active

  !> @brief Check if fullblcoep coefficient array is allocated
  !> @return 1 if allocated, 0 otherwise
  !> @note Use this before calling blockdata_coef_get to prevent segfaults
  function blockdata_fullblcoep_allocated() bind(c) result(alloc)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: fullblcoep
    implicit none
    integer(c_int) :: alloc

    alloc = 0
    if (allocated(fullblcoep)) alloc = 1
  end function blockdata_fullblcoep_allocated

  !> @brief Safe version of coefficient matrix getter with status check
  !> @param[out] coefs Flattened coefficient matrix (nblock x nblock)
  !> @param[out] status 0=success, -1=not allocated, -2=MSLD not active
  !> @note Only succeeds if MSLD is active and fullblcoep is allocated
  subroutine blockdata_coef_get_safe(coefs, status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use chm_kinds, only: chm_real
    use lambdam, only: nblock, fullblcoep, qmld
    implicit none
    real(c_double), dimension(*), intent(out) :: coefs
    integer(c_int), intent(out) :: status
    integer :: i, j, idx

    ! Check if MSLD is active
    if (.not. qmld) then
      status = -2
      return
    endif

    ! Check if array is allocated
    if (.not. allocated(fullblcoep)) then
      status = -1
      return
    endif

    ! Safe to copy data
    idx = 1
    do i = 1, nblock
      do j = 1, nblock
        coefs(idx) = real(fullblcoep(i, j), c_double)
        idx = idx + 1
      end do
    end do
    status = 0
  end subroutine blockdata_coef_get_safe

  !> @brief Check if basic BLOCK is active (QBLOCK flag from block_ltm)
  !> @return 1 if BLOCK active, 0 otherwise
  function blockdata_qblock_active() bind(c) result(active)
    use, intrinsic :: iso_c_binding, only: c_int
    use block_ltm, only: QBLOCK
    implicit none
    integer(c_int) :: active

    active = 0
    if (QBLOCK) active = 1
  end function blockdata_qblock_active

  !> @brief Get number of interactions (triangular matrix size)
  !> @return NINTER = nblock*(nblock+1)/2
  function blockdata_get_ninter() bind(c) result(ninter_out)
    use, intrinsic :: iso_c_binding, only: c_int
    use block_ltm, only: NINTER
    implicit none
    integer(c_int) :: ninter_out

    ninter_out = NINTER
  end function blockdata_get_ninter

  !> @brief Check if BLCOEP array is allocated
  !> @return 1 if allocated, 0 otherwise
  function blockdata_blcoep_allocated() bind(c) result(alloc)
    use, intrinsic :: iso_c_binding, only: c_int
    use block_ltm, only: BLCOEP
    implicit none
    integer(c_int) :: alloc

    alloc = 0
    if (allocated(BLCOEP)) alloc = 1
  end function blockdata_blcoep_allocated

  !> @brief Get BLCOEP triangular coefficient matrix (basic BLOCK)
  !> @param[out] coefs Triangular matrix as 1D array (size ninter)
  !> @param[out] status 0=success, -1=not allocated, -2=BLOCK not active
  !> @note BLCOEP uses triangular indexing: idx = i*(i-1)/2 + j where i >= j
  subroutine blockdata_blcoep_get(coefs, status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use chm_kinds, only: chm_real
    use block_ltm, only: QBLOCK, NINTER, BLCOEP
    implicit none
    real(c_double), dimension(*), intent(out) :: coefs
    integer(c_int), intent(out) :: status
    integer :: i

    ! Check if BLOCK is active
    if (.not. QBLOCK) then
      status = -2
      return
    endif

    ! Check if array is allocated
    if (.not. allocated(BLCOEP)) then
      status = -1
      return
    endif

    ! Copy triangular matrix data
    do i = 1, NINTER
      coefs(i) = real(BLCOEP(i), c_double)
    end do
    status = 0
  end subroutine blockdata_blcoep_get

  !> @brief Get a single coefficient from BLCOEP triangular matrix
  !> @param[in] i Block index i (1-based, i >= j)
  !> @param[in] j Block index j (1-based)
  !> @return Coefficient value, or -999.0 if invalid/not allocated
  function blockdata_blcoep_get_ij(i, j) bind(c) result(coef)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use chm_kinds, only: chm_real
    use block_ltm, only: QBLOCK, NBLOCK, BLCOEP
    implicit none
    integer(c_int), value, intent(in) :: i, j
    real(c_double) :: coef
    integer :: ii, jj, idx

    coef = -999.0d0

    if (.not. QBLOCK) return
    if (.not. allocated(BLCOEP)) return

    ! Ensure ii >= jj for triangular indexing
    if (i >= j) then
      ii = i
      jj = j
    else
      ii = j
      jj = i
    endif

    ! Validate indices
    if (ii < 1 .or. ii > NBLOCK .or. jj < 1 .or. jj > NBLOCK) return

    ! Calculate triangular index: idx = ii*(ii-1)/2 + jj
    idx = ii * (ii - 1) / 2 + jj
    coef = real(BLCOEP(idx), c_double)
  end function blockdata_blcoep_get_ij

  !> @brief Get block assignment for each atom (IBLCKP array)
  !> @param[out] assignments Array of block IDs (size natom)
  !> @param[in] natom Number of atoms
  !> @param[out] status 0=success, -1=not allocated, -2=BLOCK not active
  subroutine blockdata_get_atom_blocks(assignments, natom, status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use block_ltm, only: QBLOCK, IBLCKP
    implicit none
    integer(c_int), dimension(*), intent(out) :: assignments
    integer(c_int), value, intent(in) :: natom
    integer(c_int), intent(out) :: status
    integer :: i

    if (.not. QBLOCK) then
      status = -2
      return
    endif

    if (.not. allocated(IBLCKP)) then
      status = -1
      return
    endif

    do i = 1, natom
      assignments(i) = IBLCKP(i)
    end do
    status = 0
  end subroutine blockdata_get_atom_blocks

  !> @brief Set block assignment for multiple atoms (SelectAtoms compatible)
  !> @param[in] atom_ids Array of atom indices (1-based)
  !> @param[in] block_id Block to assign
  !> @param[in] count Number of atoms in array
  !> @param[out] status 0=success, -1=IBLCKP not allocated
  !> @note Auto-expands NBLOCK if block_id > current NBLOCK
  subroutine blockdata_set_atom_blocks(atom_ids, block_id, count, status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use block_ltm, only: NBLOCK, IBLCKP
#if KEY_BLADE==1
    use blade_main, only: system_dirty
#endif
#if KEY_OPENMM==1
    use omm_ctrl, only: omm_system_changed
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    integer(c_int), dimension(*), intent(in) :: atom_ids
    integer(c_int), value, intent(in) :: block_id, count
    integer(c_int), intent(out) :: status
    integer :: i

    ! Check if IBLCKP is allocated
    if (.not. allocated(IBLCKP)) then
      status = -1
      return
    endif

    ! Auto-expand NBLOCK if needed
    if (block_id > NBLOCK) NBLOCK = block_id

    ! Set block assignments
    do i = 1, count
      IBLCKP(atom_ids(i)) = block_id
    end do

    status = 0

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
  end subroutine blockdata_set_atom_blocks

  !> @brief Set bias parameters for a specific bias
  !> @param[in] bias_id Bias index (1-based)
  !> @param[in] block_i First block index
  !> @param[in] block_j Second block index
  !> @param[in] cls Bias class (1-12)
  !> @param[in] ref Reference value
  !> @param[in] cforce Force constant
  !> @param[in] npower Power/exponent
  subroutine blockdata_bias_set(bias_id, block_i, block_j, cls, &
                                 ref, cforce, npower) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use chm_kinds, only: chm_real
    use lambdam, only: nblock, nbiasv, ibvidi, ibvidj, ibclas, irreup, ikbias, ipbias
#if KEY_BLADE==1
    use blade_main, only: system_dirty
#endif
#if KEY_OPENMM==1
    use omm_ctrl, only: omm_system_changed
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    integer(c_int), value, intent(in) :: bias_id, block_i, block_j, cls, npower
    real(c_double), value, intent(in) :: ref, cforce

    if (bias_id < 1 .or. bias_id > nbiasv) return
    if (block_i < 1 .or. block_i > nblock) return
    if (block_j < 1 .or. block_j > nblock) return
    if (cls < 1 .or. cls > 12) return
    if (npower < 0) return
    if (.not. allocated(ibvidi) .or. .not. allocated(ibvidj) .or. &
        .not. allocated(ibclas) .or. .not. allocated(irreup) .or. &
        .not. allocated(ikbias) .or. .not. allocated(ipbias)) return
    if (bias_id > size(ibvidi) .or. bias_id > size(ibvidj) .or. &
        bias_id > size(ibclas) .or. bias_id > size(irreup) .or. &
        bias_id > size(ikbias) .or. bias_id > size(ipbias)) return

    ! Set bias parameters
    ibvidi(bias_id) = block_i
    ibvidj(bias_id) = block_j
    ibclas(bias_id) = cls
    irreup(bias_id) = real(ref, chm_real)
    ikbias(bias_id) = real(cforce, chm_real)
    ipbias(bias_id) = npower

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
  end subroutine blockdata_bias_set

  !> @brief Set MSLD site assignments for blocks
  !> @param[in] sites Array of site IDs for each block (1-based)
  !> @param[in] count Number of blocks
  !> @note Requires isitemld to already be allocated by active MSLD state
  subroutine blockdata_sites_set(sites, count) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use lambdam, only: nblock, isitemld, nsitemld
#if KEY_BLADE==1
    use blade_main, only: system_dirty
#endif
#if KEY_OPENMM==1
    use omm_ctrl, only: omm_system_changed
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    integer(c_int), dimension(*), intent(in) :: sites
    integer(c_int), value, intent(in) :: count
    integer :: i, max_site

    if (.not. allocated(isitemld)) return
    if (count < 1 .or. count /= nblock) return
    if (count > size(isitemld)) return

    max_site = 0
    do i = 1, count
      if (sites(i) < 0) return
      isitemld(i) = sites(i)
      if (sites(i) > max_site) max_site = sites(i)
    end do
    nsitemld = max_site

#if KEY_BLADE==1
    system_dirty = .true.
#endif
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
  end subroutine blockdata_sites_set

  !> @brief Read the state needed for pH/MSLD label exchange
  !> @param[out] ph Current pH label
  !> @param[out] temperature Lambda-dynamics temperature
  !> @param[out] lambdas Current lambda occupancies
  !> @param[out] biases Current BIELAM bias vector
  !> @param[out] masses Lambda masses (protocol invariant)
  !> @param[out] frictions Lambda frictions (protocol invariant)
  !> @param[out] fixed FFIX flags (protocol invariant)
  !> @param[out] sites MSLD site assignments (protocol invariant)
  !> @param[in] count Size of the lambda and bias buffers
  !> @param[out] status 0=success, -1=wrong size, -2=MSLD state unavailable
  subroutine blockdata_ph_rex_get(ph, temperature, lambdas, biases, masses, &
       frictions, fixed, sites, count, status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    use chm_kinds, only: chm_real
    use lambdam, only: qmld, nblock, ninter, phval, tbld, bixlam, bielam, &
         bimlam, biblam, qlfix, isitemld, blcoep, msld_setblcoef
    implicit none
    real(c_double), intent(out) :: ph, temperature
    real(c_double), dimension(*), intent(out) :: lambdas, biases, masses, frictions
    integer(c_int), dimension(*), intent(out) :: fixed, sites
    integer(c_int), value, intent(in) :: count
    integer(c_int), intent(out) :: status
    integer :: i

    ph = 0.0d0
    temperature = 0.0d0
    status = -2

    if (count /= nblock .or. count < 1) then
      status = -1
      return
    endif
    if (.not. qmld .or. .not. allocated(bixlam) .or. &
        .not. allocated(bielam) .or. .not. allocated(bimlam) .or. &
        .not. allocated(biblam) .or. .not. allocated(qlfix) .or. &
        .not. allocated(isitemld) .or. .not. allocated(blcoep)) return

    ! BLaDE returns current theta at the end of each dynamics segment, but
    ! BIXLAM is otherwise refreshed only when lambda output is written.
    call msld_setblcoef(nblock, ninter, bixlam, blcoep)

    ph = real(phval, c_double)
    temperature = real(tbld, c_double)
    do i = 1, count
      lambdas(i) = real(bixlam(i), c_double)
      biases(i) = real(bielam(i), c_double)
      masses(i) = real(bimlam(i), c_double)
      frictions(i) = real(biblam(i), c_double)
      fixed(i) = merge(1_c_int, 0_c_int, qlfix(i))
      sites(i) = int(isitemld(i), c_int)
    enddo
    status = 0
  end subroutine blockdata_ph_rex_get

  !> @brief Apply a pH label and its complete BIELAM vector atomically
  !> @param[in] ph New pH label
  !> @param[in] biases Complete BIELAM bias vector for that pH
  !> @param[in] count Size of the bias buffer
  !> @param[out] status 0=success, -1=wrong size, -2=MSLD state unavailable,
  !>                    -3=non-finite input, -4=live BLaDE synchronization failed
  subroutine blockdata_ph_rex_set(ph, biases, count, status) bind(c)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    use chm_kinds, only: chm_real
    use lambdam, only: qmld, nblock, phval, bielam
#if KEY_BLADE==1
    use blade_main, only: blade_sync_ph_bias_from_values
#endif
#if KEY_OPENMM==1
    use omm_ctrl, only: omm_system_changed
#endif
#if KEY_DOMDEC==1
    use domdec_common, only: domdec_system_changed
#endif
    implicit none
    real(c_double), value, intent(in) :: ph
    real(c_double), dimension(*), intent(in) :: biases
    integer(c_int), value, intent(in) :: count
    integer(c_int), intent(out) :: status
    integer :: i

    status = -2
    if (count /= nblock .or. count < 1) then
      status = -1
      return
    endif
    if (.not. qmld .or. .not. allocated(bielam)) return
    if (.not. ieee_is_finite(ph)) then
      status = -3
      return
    endif
    do i = 1, count
      if (.not. ieee_is_finite(biases(i))) then
        status = -3
        return
      endif
    enddo

#if KEY_BLADE==1
    ! Update a clean live BLaDE system before changing the CHARMM copy so a
    ! rejected synchronization leaves the pH label unchanged.
    if (blade_sync_ph_bias_from_values(biases, count) == 0_c_int) then
      status = -4
      return
    endif
#endif

    phval = real(ph, chm_real)
    do i = 1, count
      bielam(i) = real(biases(i), chm_real)
    enddo

#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
    status = 0
  end subroutine blockdata_ph_rex_set

#endif /* KEY_BLOCK */
end module api_blockdata
