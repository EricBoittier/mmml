module omm_nbopts
   use chm_kinds
   implicit none

   type, public :: omm_nbopts_t
      real(chm_real) :: rcut
      real(chm_real) :: switchdist
      real(chm_real) :: rf_diel
      real(chm_real) :: box(3)
      real(chm_real) :: alpha       ! kappa for ewald and PME
      real(chm_real) :: torsion_lambda
      integer :: nbxmode
      integer :: kmax(3)
      integer :: mesh_dim(3)
      integer :: block_changed
      logical :: periodic           ! use pbcs
      logical :: use_ewald          ! ewald option (non-pme)
      logical :: use_pme            ! pme option
      logical :: use_lrc            ! vdw long range correction
      logical :: use_eten           ! Go model eten form
      logical :: use_etsr           ! Go model etsr form  V_ETSR=V_ETEN / (1 + (2r/3sigma)^12)
      logical :: use_elec_swit      ! switch option
      logical :: use_elec_fswit     ! fswitch option
      logical :: use_elec_fshift    ! fshift option
      logical :: use_elec_rxnfld    ! omm reaction field option
      logical :: use_vdw_ommswit    ! omm vdW switching option
      logical :: use_vdw_swit       ! vswitch option
      logical :: use_vdw_fswit      ! fswitch option
      logical :: use_omm_vdw        ! composite logical (see below)
      logical :: use_omm_elec       ! composite logical (see below)
      logical :: use_gbsaobc2       ! logical liked to OpenMM GBSAOBC2
      logical :: use_gbsw           ! logical liked to OpenMM GBSW
      logical :: use_gbmv           ! logical liked to OpenMM GBMV
      logical :: use_mses           ! logical liked to OpenMM MSES AN EXAMPLE OF OPENMM PLUGIN
      logical :: use_block          ! logical indicating block was called
      logical :: use_soft_omm       ! logical indicating fot vdW for block is true
      logical :: qtor_repex         ! Torsion angle repd is turned on/off
      character(len=20) :: omm_platform
      character(len=20) :: omm_deviceid
      character(len=20) :: omm_precision
   end type omm_nbopts_t

contains

   !> Imports nonbonded options from CHARMM into an omm_nbopts_t.
   type(omm_nbopts_t) function current_nbopts() result(opts)
      use omm_glblopts
      use inbnd, only: NBXMOD
      use ewald_1m, only: KAPPA
      use ewald, only: LEWALD, KMAXX, KMAXY, KMAXZ
      use pme_module, only: QPME
      use image, only: XUCELL, XTLTYP
      use pmeutil, only: NFFT1, NFFT2, NFFT3
      use inbnd, only: QETEN, QETSR,     & ! GOMODEL 
           CTOFNB, CTONNB,               & ! cutoff and switching distance
           LCONS, LFSWT, LSHFT, LEGROM,  & ! electrostatics
           LVDW, LVFSWT, LVSHFT, LVGROM, & ! vdw
           LOMMSWI, LOMMRXN, OMM_RXNFLD_DIELECTRIC, &
           LOMMGB
#if KEY_LRVDW==1
      use inbnd, only: LLRVDW  
#endif
      use inbnd, only: LOMMGB, qgbsw_omm, qgbmv_omm, qmses_omm ! AN EXAMPLE OF OPENMM PLUGIN
#if KEY_BLOCK==1
      use block_ltm, only: qblock, nblckcalls, qsoft_omm
#endif
      implicit none
      ! TODO move fallback decls to inbnd
#if KEY_LRVDW==0
      logical, parameter :: LLRVDW = .false.  
#endif
      character(len=100) :: charmm_plugin_dir_name

      real*8 :: openmm_version_number
      integer :: i1, i2
      character(len=10) :: omm_version, version

      opts%alpha = KAPPA
      opts%kmax = [KMAXX, KMAXY, KMAXZ]

      opts%rcut = CTOFNB
      opts%switchdist = CTONNB

      opts%use_elec_swit = LCONS .and. .not. &
           ( LFSWT .or. LSHFT .or. LEWALD .or. LEGROM .or. opts%rcut >= 990 )
      opts%use_elec_fswit = LCONS .and. LFSWT .and. .not. &
           ( LSHFT .or. LEWALD .or. LEGROM .or. opts%rcut >= 990 )
      opts%use_elec_fshift = LCONS .and. LSHFT .and. LFSWT .and. .not. &
           ( LEWALD .or. LEGROM .or. opts%rcut >= 990 )

      ! use electrostatic omm rxn field
      opts%use_elec_rxnfld = LCONS .and. LOMMRXN .and. .not. &
           ( LFSWT .or. LEWALD .or. LEGROM )
      ! use OpenMM vdW switching
      opts%use_vdw_ommswit = lommswi

      opts%use_vdw_swit = LVDW .and. .not. &
           ( LVGROM .or. LVFSWT .or. LVSHFT.or. opts%rcut >= 990  &
           .or. lommswi )
      opts%use_vdw_fswit = LVFSWT .and. LVDW .and. .not. &
           ( LVGROM .or. opts%rcut >= 990 &
           .or. lommswi)

      opts%rf_diel = omm_rxnfld_dielectric ! moved default setup to nbutil.src/inbnd_ltm.src
      opts%box = XUCELL(1:3)
      opts%nbxmode = NBXMOD
      opts%mesh_dim = [NFFT1, NFFT2, NFFT3]
      opts%periodic = XTLTYP /= '    '
      opts%use_ewald = LEWALD
      opts%use_pme = QPME .and. LEWALD
      opts%use_lrc = LLRVDW
      opts%use_eten = QETEN
      opts%use_etsr = QETSR
      opts%use_gbsaobc2 = LOMMGB
      opts%use_gbsw     = qgbsw_omm
      opts%use_gbmv     = qgbmv_omm
      opts%use_mses     = qmses_omm ! AN EXAMPLE OF OPENMM PLUGIN

      opts%use_vdw_swit = opts%use_vdw_swit &
           .and. .not. ( (opts%rcut <= opts%switchdist) .and. &
           ( opts%use_ewald .or. opts%use_elec_rxnfld ) )

      opts%use_omm_vdw = opts%use_vdw_ommswit .or. &
           .not. ( opts%use_vdw_swit .or. opts%use_vdw_fswit )

      opts%use_omm_elec = (opts%use_ewald .or. &
         opts%use_elec_rxnfld .or. opts%rcut >= 990 )

      opts%use_elec_swit = opts%use_elec_swit .and. .not. opts%use_omm_elec

#if KEY_BLOCK==1
      opts%use_block = qblock
      opts%block_changed = nblckcalls
      opts%use_soft_omm = qsoft_omm
#endif
      opts%qtor_repex = qtor_repex
      opts%torsion_lambda = torsion_lambda
!      opts%qtor_repex = opts%qtor_repex .and. opts%use_block

#if KEY_OPENMM==1
      call OpenMM_Platform_getOpenMMVersion(omm_version)
      i1 = index(omm_version,".")
      i2 = index(omm_version(i1+1:len(trim(omm_version))),".")
      if(i2>0) then
         version = trim( omm_version(1:i2-1) & 
              // omm_version(i2+1:len(trim(omm_version))) )
         omm_version = version
      endif
      opts%omm_platform = omm_platform
      opts%omm_precision = omm_precision
      opts%omm_deviceid = omm_deviceid
#endif

   end function current_nbopts

   logical function same_nbopts(a, b) result(same)
     implicit none
     type(omm_nbopts_t), intent(in) :: a, b
     real(chm_real), parameter :: NOCUT = 990.0
     real(chm_real), parameter :: RTOL = 1.0d-3

      same = (a%rcut >= NOCUT .and. b%rcut >= NOCUT) &
            .or. abs(a%rcut - b%rcut) < RTOL
      same = same .and. abs(a%switchdist - b%switchdist) < RTOL

      same = same .and. (a%use_eten .eqv. b%use_eten)
      same = same .and. (a%use_etsr .eqv. b%use_etsr)
      same = same .and. (a%nbxmode == b%nbxmode)

      same = same .and. (a%periodic .eqv. b%periodic)
      if (a%periodic) then
         ! don't compare box sizes, Context allows change
         same = same .and. (a%use_lrc .eqv. b%use_lrc)
         same = same .and. (a%use_pme .eqv. b%use_pme)
         same = same .and. (a%use_ewald .eqv. b%use_ewald)
         if (a%use_ewald) then
            same = same .and. (a%alpha == b%alpha)
            same = same .and. all(a%kmax == b%kmax)
         endif
         if (a%use_pme) then
            same = same .and. (a%alpha == b%alpha)
            same = same .and. all(a%mesh_dim == b%mesh_dim)
         endif
      else
         same = same .and. a%rf_diel == b%rf_diel
      endif

      same = same .and. (a%use_elec_swit .eqv. b%use_elec_swit)
      same = same .and. (a%use_elec_fswit .eqv. b%use_elec_fswit)
      same = same .and. (a%use_elec_fshift .eqv. b%use_elec_fshift)
      same = same .and. (a%use_elec_rxnfld .eqv. b%use_elec_rxnfld)

      same = same .and. (a%use_vdw_ommswit .eqv. b%use_vdw_ommswit)
      same = same .and. (a%use_vdw_swit .eqv. b%use_vdw_swit)
      same = same .and. (a%use_vdw_fswit .eqv. b%use_vdw_fswit)
      same = same .and. (a%use_omm_vdw .eqv. b%use_omm_vdw)
      same = same .and. (a%use_omm_elec .eqv. b%use_omm_elec)
      same = same .and. (a%use_gbsaobc2 .eqv. b%use_gbsaobc2)
      same = same .and. (a%use_gbsw .eqv. b%use_gbsw)
      same = same .and. (a%use_gbmv .eqv. b%use_gbmv)
      same = same .and. (a%use_mses .eqv. b%use_mses) ! AN EXAMPLE OF OPENMM PLUGIN

      same = same .and. (a%use_block .eqv. b%use_block)
      same = same .and. (a%use_soft_omm .eqv. b%use_soft_omm)
      same = same .and. (a%qtor_repex .eqv. b%qtor_repex)
      same = same .and. (a%block_changed == b%block_changed)

      ! Do these checks belong in a subroutine of omm_glblopts?
      ! hack: omm_glblopts%omm_platform needs fixing
      if (trim(a%omm_platform) /= '' .and. trim(b%omm_platform) /= '') &
           same = same .and. (a%omm_platform == b%omm_platform)

      if (trim(a%omm_precision) /= '' .and. trim(b%omm_precision) /= '') &
           same = same .and. (a%omm_precision == b%omm_precision)

      if (trim(a%omm_deviceid) /= '' .and. trim(b%omm_deviceid) /= '') &
           same = same .and. (a%omm_deviceid == b%omm_deviceid)
   end function same_nbopts

   subroutine print_real_diff(name, a_value, b_value)
     use stream, only: outu
     use chm_kinds, only: chm_real
     implicit none
     character(len=*), intent(in) :: name
     real(chm_real), intent(in) :: a_value, b_value

     write(outu, '(a, x, a, /, a, x, a, x, f8.4, /, a, x, a, x, f8.4)') &
          name, 'differed', &
          name, 'a:', a_value, &
          name, 'b:', b_value
   end subroutine print_real_diff

   subroutine print_int_diff(name, a_value, b_value)
     use stream, only: outu
     implicit none
     character(len=*), intent(in) :: name
     integer, intent(in) :: a_value, b_value

     write(outu, '(a, x, a, /, a, x, a, x, i4, /, a, x, a, x, i4)') &
          name, 'differed', &
          name, 'a:', a_value, &
          name, 'b:', b_value
   end subroutine print_int_diff

   subroutine print_logical_diff(name, a_value, b_value)
     use stream, only: outu
     implicit none
     character(len=*), intent(in) :: name
     logical, intent(in) :: a_value, b_value

     write(outu, '(a, x, a, /, a, x, a, x,l5, /, a, x, a, x, l5)') &
          name, 'differed', &
          name, 'a:', a_value, &
          name, 'b:', b_value
   end subroutine print_logical_diff

   subroutine print_char_diff(name, a_value, b_value)
     use stream, only: outu
     use chm_kinds, only: chm_real
     implicit none
     character(len=*), intent(in) :: name
     character(len=*), intent(in) :: a_value, b_value

     write(outu, '(a, x, a, /, a, x, a, x, a, /, a, x, a, x, a)') &
          name, 'differed', &
          name, 'a:', a_value, &
          name, 'b:', b_value
   end subroutine print_char_diff

   subroutine print_nbopts_diff(a, b)
     use stream, only: outu
     implicit none

     type(omm_nbopts_t), intent(in) :: a, b

     logical :: same
     real(chm_real), parameter :: &
          NOCUT = 990.0,          &
          RTOL = 1.0d-3
     
     write(outu, '(a)') 'Differences between nonbonded options a and b'
     if ((a%rcut < NOCUT .or. b%rcut < NOCUT) &
         .and. abs(a%rcut - b%rcut) > RTOL)  &
         call print_real_diff('rcut', a%rcut, b%rcut)

     if (abs(a%switchdist - b%switchdist) > RTOL) &
          call print_real_diff('switchdist', a%switchdist, b%switchdist)

     if (a%use_eten .neqv. b%use_eten) &
          call print_logical_diff('use_eten', a%use_eten, b%use_eten)     

     if (a%use_etsr .neqv. b%use_etsr) &
          call print_logical_diff('use_etsr', a%use_etsr, b%use_etsr)

     if (a%nbxmode /= b%nbxmode) &
          call print_int_diff('nbxmode', a%nbxmode, b%nbxmode)

     if (a%periodic .neqv. b%periodic) &
          call print_logical_diff('periodic', a%periodic, b%periodic)

     if (a%periodic .or. b%periodic) then
        if (a%use_lrc .neqv. b%use_lrc) &
             call print_logical_diff('use_lrc', a%use_lrc, b%use_lrc)

        if (a%use_pme .neqv. b%use_pme) &
             call print_logical_diff('use_pme', a%use_pme, b%use_pme)
        if (a%use_pme .or. b%use_pme) then
           if (a%alpha /= b%alpha) &
                call print_real_diff('alpha', a%alpha, b%alpha)
           if (.not. all(a%mesh_dim == b%mesh_dim)) then
              call print_int_diff('mesh_dim(1)', a%mesh_dim(1), b%mesh_dim(1))
              call print_int_diff('mesh_dim(2)', a%mesh_dim(2), b%mesh_dim(2))
              call print_int_diff('mesh_dim(3)', a%mesh_dim(3), b%mesh_dim(3))
           end if
        end if

        if (a%use_ewald .neqv. b%use_ewald) &
             call print_logical_diff('use_ewald', a%use_ewald, b%use_ewald)
        if (a%use_ewald .or. b%use_ewald) then
           if (a%alpha /= b%alpha) &
                call print_real_diff('alpha', a%alpha, b%alpha)
           if (.not. all(a%kmax == b%kmax)) then
              call print_int_diff('kmax(1)', a%kmax(1), b%kmax(1))
              call print_int_diff('kmax(2)', a%kmax(2), b%kmax(2))
              call print_int_diff('kmax(3)', a%kmax(3), b%kmax(3))
           end if
        end if
     else
        if(a%rf_diel /= b%rf_diel) &
             call print_real_diff('rf_diel', a%rf_diel, b%rf_diel)
     endif

     if (a%use_elec_swit .neqv. b%use_elec_swit) &
          call print_logical_diff('use_elec_swit', a%use_elec_swit, b%use_elec_swit)
     if (a%use_elec_fswit .neqv. b%use_elec_fswit) &
          call print_logical_diff('use_elec_fswit', a%use_elec_fswit, b%use_elec_fswit)
     if (a%use_elec_fshift .neqv. b%use_elec_fshift) &
          call print_logical_diff('use_elec_fshift', a%use_elec_fshift, b%use_elec_fshift)
     if (a%use_elec_rxnfld .neqv. b%use_elec_rxnfld) &
          call print_logical_diff('use_elec_rxnfld', a%use_elec_rxnfld, b%use_elec_rxnfld)

     if (a%use_vdw_ommswit .neqv. b%use_vdw_ommswit) &
          call print_logical_diff('use_vdw_ommswit', a%use_vdw_ommswit, b%use_vdw_ommswit)
     if (a%use_vdw_swit .neqv. b%use_vdw_swit) &
          call print_logical_diff('use_vdw_swit', a%use_vdw_swit, b%use_vdw_swit)
     if (a%use_vdw_fswit .neqv. b%use_vdw_fswit) &
          call print_logical_diff('use_vdw_fswit', a%use_vdw_fswit, b%use_vdw_fswit)
     if (a%use_omm_vdw .neqv. b%use_omm_vdw) &
          call print_logical_diff('use_omm_vdw', a%use_omm_vdw, b%use_omm_vdw)
     if (a%use_omm_elec .neqv. b%use_omm_elec) &
          call print_logical_diff('use_omm_elec', a%use_omm_elec, b%use_omm_elec)
     if (a%use_gbsaobc2 .neqv. b%use_gbsaobc2) &
          call print_logical_diff('use_gbsaobc2', a%use_gbsaobc2, b%use_gbsaobc2)
     if (a%use_gbsw .neqv. b%use_gbsw) &
          call print_logical_diff('use_gbsw', a%use_gbsw, b%use_gbsw)
     if (a%use_gbmv .neqv. b%use_gbmv) &
          call print_logical_diff('use_gbmv', a%use_gbmv, b%use_gbmv)
     if (a%use_mses .neqv. b%use_mses) & ! AN EXAMPLE OF OPENMM PLUGIN
          call print_logical_diff('use_mses', a%use_mses, b%use_mses) ! AN EXAMPLE OF OPENMM PLUGIN

     if (a%use_block .neqv. b%use_block) &
          call print_logical_diff('use_block', a%use_block, b%use_block)
     if (a%qtor_repex .neqv. b%qtor_repex) &
          call print_logical_diff('qtor_repex', a%qtor_repex, b%qtor_repex)
     if (a%block_changed /= b%block_changed) &
          call print_int_diff('block_changed', a%block_changed, b%block_changed)

     if (a%omm_platform /= b%omm_platform) &
          call print_char_diff('omm_platform', a%omm_platform, b%omm_platform)
     if (a%omm_precision /= b%omm_precision) &
          call print_char_diff('omm_precision', a%omm_precision, b%omm_precision)
     if (a%omm_deviceid /= b%omm_deviceid) &
          call print_char_diff('omm_deviceid', a%omm_deviceid, b%omm_deviceid)
   end subroutine print_nbopts_diff
end module omm_nbopts

