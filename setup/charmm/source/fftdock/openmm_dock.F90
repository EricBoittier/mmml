module openmm_dock
  use OpenMM
  use chm_kinds
  implicit none
#if KEY_FFTDOCK == 1 && KEY_OPENMM ==1
  !! objects used by OpenMM
  type(OpenMM_System), save :: system
#if OMM_VER < 82
  type(OpenMM_LangevinIntegrator), save :: langevin_integrator
#else
  type(OpenMM_LangevinMiddleIntegrator), save :: langevin_integrator
#endif /* OMM_VER */
  type(OpenMM_VerletIntegrator), save :: verlet_integrator
  type(OpenMM_Platform), save :: platform
  type(OpenMM_Context), save :: context
  type(OpenMM_Vec3Array), save :: position
  type(OpenMM_State), save :: state

  !! data for grid potential
  real(chm_real) XGridCenter, YGridCenter, ZGridCenter
  real(chm_real) XGridLen, YGridLen, ZGridLen
  real(chm_real) XGridMin, YGridMin, ZGridMin
  real(chm_real) XGridMax, YGridMax, ZGridMax
  real(chm_real) DGrid, GridForce
  integer NumGrid, XGridNum, YGridNum, ZGridNum
  real(chm_real),allocatable,dimension(:) :: SoftGridPot
  real(chm_real),allocatable,dimension(:) :: HardGridPot
  real(chm_real),allocatable,dimension(:) :: GridRadii
  type(OpenMM_DoubleArray), save :: grid_potentials(21)
  integer :: softGridUnit, hardGridUnit
  logical :: Form
  real(chm_real8) :: soft_p, hard_p, emax, mine, maxe, eps

  !! atom selections for both fixed and flexible parts of the system
  integer, allocatable, dimension(:) :: fix_select
  integer, allocatable, dimension(:) :: flex_select
  integer, allocatable, dimension(:) :: omm_select
  integer, allocatable, dimension(:) :: hdonor_select
  integer, allocatable, dimension(:) :: hacceptor_select
  integer, allocatable, dimension(:) :: charmmIdx2ommIdx
  integer num_fix_atom, num_flex_atom
  integer ::  num_copy_flex

  !! force groups
  integer :: force_group_bond = 1
  integer :: force_group_angle = 2
  integer :: force_group_torsion = 3
  integer :: force_group_improper = 4
  integer :: force_group_cmaps = 5
  integer :: force_group_vdw_grid = 6
  integer :: force_group_elec_grid = 7
  integer :: force_group_nonbond = 8
  integer :: force_group_external = 9
  integer :: force_group_hdonor_grid = 10
  integer :: force_group_hacceptor_grid = 11

  !! simulated annealing parameters
  integer :: num_steps, heat_frq
  real(chm_real) :: start_temp, end_temp, incr_temp

CONTAINS

  subroutine openmm_dock_set(comlyn, comlen)
    use psf
    use coord
    use select
    use string
    use number, only: one
    use stream, only: outu, prnlev
    use memory, only: chmalloc, chmdealloc
    implicit none

    character(len=*) comlyn
    integer comlen
    logical :: flag_create = .FALSE.
    logical :: flag_set_position_from_main = .FALSE.
    logical :: flag_copy_position_to_main = .FALSE.
    logical :: flag_simulated_anneling = .FALSE.
    logical :: flag_read_grid_potentials = .FALSE.
    logical :: flag_print_energy = .FALSE.
    logical :: flag_minimize = .FALSE.
    logical :: flag_change_grid_softness = .FALSE.
    logical :: flag_hbond = .FALSE.
    logical :: flag_no_fix_atom = .FALSE.
    logical :: flag_clear = .FALSE.
    logical :: err
    integer :: i, HBatm, idx_copy, NGrid0, num_grid_points

    flag_clear = (IndxA(comLyn, comlen, 'CLEA').GT.0)
    flag_hbond = (IndxA(comLyn, comlen, 'GRHB').GT.0)
    flag_create = (IndxA(comLyn, comlen, 'BUIL').GT.0)
    flag_minimize = (IndxA(comLyn, comlen, 'MINI').GT.0)
    flag_print_energy = (IndxA(comLyn, comlen, 'ENER').GT.0)
    flag_simulated_anneling = (IndxA(comLyn, comlen, 'SIAN').GT.0)
    flag_change_grid_softness = (IndxA(comLyn, comlen, 'CGRS').GT.0)
    flag_read_grid_potentials = (IndxA(comLyn, comlen, 'GRID').GT.0)
    flag_copy_position_to_main = (IndxA(comLyn, comlen, 'COOR').GT.0)
    flag_set_position_from_main = (IndxA(comLyn, comlen, 'SETC').GT.0)

    !! Read in grid potential
    if (flag_read_grid_potentials) THEN
       softGridUnit = GTrmI(comlyn, comlen, 'UNIS', OutU)
       hardGridUnit = GTrmI(comlyn, comlen, 'UNIH', OutU)
       Form = .False.
       Form = (IndxA(comlyn, comlen, 'FORM').GT.0)
       If (softGridUnit .eq. OutU) Form = .True.
       call Read_Grid_Potentials(softGridUnit, hardGridUnit, flag_hbond, Form)
    endif

    !! Build OpenMM docking system
    if (flag_create) THEN
       !---------------------------Allocate arrays--------------------------
       call chmalloc("openmm_dock.src", "openmm_dock_set", 'omm_select', natom, intg = omm_select)
       call chmalloc("openmm_dock.src", "openmm_dock_set", 'fix_select', natom, intg = fix_select)
       call chmalloc("openmm_dock.src", "openmm_dock_set", 'flex_select', natom, intg = flex_select)
       call chmalloc("openmm_dock.src", "openmm_dock_set", 'hdonor_select', natom, intg = hdonor_select)
       call chmalloc("openmm_dock.src", "openmm_dock_set", 'hacceptor_select', natom, intg = hacceptor_select)
       call chmalloc("openmm_dock.src", "openmm_dock_set", 'charmmIdx2ommIdx', &
                     natom, intg = charmmIdx2ommIdx)
       omm_select(:) = 0
       fix_select(:) = 0
       flex_select(:) = 0
       hdonor_select(:) = 0
       hacceptor_select(:) = 0

       !---------------------------Read in system--------------------------
       num_copy_flex = GTrmI(comlyn, comlen, 'NCOPY', 1)
       flag_no_fix_atom = (IndxA(comLyn, comlen, 'NOFI').GT.0)
       if (flag_no_fix_atom) THEN
          call selcta(comlyn, comlen, flex_select, X, Y, Z, WMAIN, .TRUE.)
          fix_select(:) = 0
       else
          call selctd(comlyn, comlen, fix_select, flex_select, X, Y, Z, WMAIN, .TRUE., err)
       endif

       !---------------------Flexible and fix atom index--------------------------
       num_fix_atom = 0
       num_flex_atom = 0
       do i = 1, nAtom
          if (fix_select(i) == 1) then
             num_fix_atom = num_fix_atom + 1
             omm_select(i) = 1
          endif
          if (flex_select(i) == 1) then
             num_flex_atom = num_flex_atom + 1
             omm_select(i) = 1
          endif
       enddo
       If (PrnLev .ge. 2) Write(OutU, '(a,i10)') &
          " Number of fixed atoms = ", num_fix_atom
       If (PrnLev .ge. 2) Write(OutU, '(a,i10)') &
          " Number of flexible atoms = ", num_flex_atom

       !--------------Hydrogen bond donor and acceptor index-----------------------
       ! Whether or not use hydrogen bond in energy calculation
       if (flag_hbond) then
         If (PrnLev .ge. 2) Write(OutU, '(a)') ' Hydrogen bond grids will be used'
         if (NACC.GT.0.and.NACC.LE.NATOM) then
            do HBatm = 1, NACC
               hacceptor_select(IACC(HBatm)) = 1
            enddo
         end if

         if (NDON.GT.0.and.NDON.LE.NATOM) then
            do HBatm = 1, NDON
               hdonor_select(IHD1(HBatm)) = 1
            enddo
         endif
       else
         If (PrnLev .ge. 2) Write(OutU, '(a)') ' Hydrogen bond grids will not be used'
       endif

       call Create_System(omm_select, fix_select, flex_select, hdonor_select, &
                          hacceptor_select, charmmIdx2ommIdx, num_copy_flex, &
                          num_fix_atom, num_flex_atom, flag_hbond)
       call Create_Context()
       call Print_Energy(flag_hbond)
    endif

    !! Change grid softness
    if (flag_change_grid_softness) THEN
       soft_p = GTrmf(comlyn, comlen, 'SOFT', 1.0D0)
       hard_p = GTrmf(comlyn, comlen, 'HARD', 0.0D0)
       emax = GTrmf(comlyn, comlen, 'EMAX', 1.0D0)
       mine = GTrmf(comlyn, comlen, 'MINE', -1.0D0)
       maxe = GTrmf(comlyn, comlen, 'MAXE', 1.0D0)
       eps = GTrmf(comlyn, comlen, 'EPS', 1.0D0)
       call Change_Grid_Softness(soft_p, hard_p, emax, mine, maxe, eps)
    endif

    !! Print OpenMM docking system energy
    if (flag_print_energy) THEN
       call Print_Energy(flag_hbond)
    endif

    if (flag_minimize) THEN
       call Minimize(comlyn, comlen)
    endif

    !! Set flex part positions based on main coor
    if (flag_set_position_from_main) THEN
       idx_copy = GTrmI(comlyn, comlen, 'IDXC', 1)
       call set_positions(flex_select, charmmIdx2ommIdx, num_flex_atom, idx_copy)
       call OpenMM_Context_getState(context, OpenMM_State_Energy, OpenMM_False, state)
    endif

    !! Copy flex part position to main coor
    if (flag_copy_position_to_main) THEN
       idx_copy = GTrmI(comlyn, comlen, 'IDXC', 1)
       call copy_positions_to_main(flex_select, charmmIdx2ommIdx, num_flex_atom, idx_copy)
    endif

    !! Run simulated annealing
    if (flag_simulated_anneling) THEN
       num_steps = GTrmI(comlyn, comlen, 'NSTE', 10000)
       heat_frq = GTrmI(comlyn, comlen, 'NHRQ', 50)
       start_temp = GTrmf(comlyn, comlen, 'FIRS', 500.0)
       end_temp = GTrmf(comlyn, comlen, 'FINA', 100.0)
       incr_temp = GTrmf(comlyn, comlen, 'INCT', 1.0)
       call Run_Simulated_Anneling(num_steps, heat_frq, start_temp, end_temp, incr_temp)
    endif

    !! Clear OpenMM docking system
    if (flag_clear) Then
       num_grid_points = NumGrid * XGridNum * YGridNum * ZGridNum
       if (flag_hbond) then
          NGrid0 = NumGrid - 3
       else
          NGrid0 = NumGrid - 1
       endif

       call chmdealloc("openmm_dock.src", "openmm_dock_set", 'omm_select', natom, intg = omm_select)
       call chmdealloc("openmm_dock.src", "openmm_dock_set", 'fix_select', natom, intg = fix_select)
       call chmdealloc("openmm_dock.src", "openmm_dock_set", 'flex_select', natom, intg = flex_select)
       call chmdealloc("openmm_dock.src", "openmm_dock_set", 'hdonor_select', natom, intg = hdonor_select)
       call chmdealloc("openmm_dock.src", "openmm_dock_set", 'hacceptor_select', natom, intg = hacceptor_select)
       call chmdealloc("openmm_dock.src", "openmm_dock_set", 'charmmIdx2ommIdx', natom, intg = charmmIdx2ommIdx)
       call chmdealloc("openmm_dock.src", "openmm_dock_set", 'SoftGridPot', num_grid_points, crl = SoftGridPot)
       call chmdealloc("openmm_dock.src", "openmm_dock_set", 'HardGridPot', num_grid_points, crl = HardGridPot)
       call chmdealloc("openmm_dock.src", "openmm_dock_set", 'GridRadii', NGrid0, crl = GridRadii)

       call OpenMM_System_destroy(system)
       call OpenMM_Context_destroy(context)
    endif

  end subroutine openmm_dock_set

  SUBROUTINE Change_Grid_Softness(soft_p, hard_p, emax, mine, maxe, eps)
    use string
    use stream

    implicit None
    real(chm_real8) :: soft_p, hard_p, emax, mine, maxe, eps

    emax = emax * OpenMM_KJPerKcal
    mine = mine * OpenMM_KJPerKcal
    maxe = maxe * OpenMM_KJPerKcal
    eps = eps / OpenMM_NmPerAngstrom

    call OpenMM_Context_setParameter(context, "soft_p", soft_p)
    call OpenMM_Context_setParameter(context, "hard_p", hard_p)
    call OpenMM_Context_setParameter(context, "Emax", emax)
    call OpenMM_Context_setParameter(context, "mine", mine)
    call OpenMM_Context_setParameter(context, "maxe", maxe)
    call OpenMM_Context_setParameter(context, "eps", eps)

  END SUBROUTINE Change_Grid_Softness

#ifdef __PGI
  integer function irand()
    implicit none
    integer, parameter :: low = 1, high = 2147483647
    real :: rand_real = 0

    call random_number(rand_real)
    irand = low + floor((high + 1 - low) * rand_real)
  end function irand
#endif

  SUBROUTINE Run_Simulated_Anneling(num_steps, heat_frq, start_temp, end_temp, incr_temp)
    use string
    use stream
#ifdef __INTEL_COMPILER
    use ifport, only: irand
#endif /* __INTEL_COMPILER */

    implicit None
    integer :: num_steps, heat_frq
    integer num_temps, num_steps_per_temp, tmp_i
    real(chm_real) :: start_temp, end_temp, incr_temp, frictioncoef
    real(chm_real) :: delta_temp
    real(chm_real) :: current_temp

    frictioncoef = 10

    ! ! set friction coefficient for Langevin dynamics
    ! frictioncoef = GTrmf(comlyn,comlen, 'FRIC', 10.0)
    ! call OpenMM_LangevinIntegrator_setFriction(langevin_integrator, frictioncoef)

    ! temperatures
    current_temp = start_temp
    call OpenMM_Context_setVelocitiesToTemperature(context, current_temp, irand())

    num_temps = int(num_steps / heat_frq)
    do tmp_i = 1, num_temps
       ! call OpenMM_LangevinIntegrator_setTemperature(langevin_integrator, current_temp)
       ! call OpenMM_LangevinIntegrator_step(langevin_integrator, heat_frq)
       call OpenMM_VerletIntegrator_step(verlet_integrator, heat_frq)
       if ((start_temp >= end_temp .and. current_temp > end_temp) .or. &
           (start_temp <= end_temp .and. current_temp < end_temp)) then
          current_temp = current_temp + incr_temp
          call OpenMM_Context_setVelocitiesToTemperature(context, current_temp, irand())
       endif
    enddo

  END SUBROUTINE Run_Simulated_Anneling

  SUBROUTINE copy_positions_to_main(flex_select, charmmIdx2ommIdx, num_flex_atom, idx_copy)
    use psf
    use coord
    use OpenMM

    implicit None
    real*8 vec(3)
    integer tmp_i, tmp_j, tmp_k
    integer idx_copy, num_flex_atom
    integer, allocatable, dimension(:) :: flex_select, charmmIdx2ommIdx

    call OpenMM_Context_getState(context, OpenMM_State_Positions, OpenMM_False, state)
    call OpenMM_State_getPositions(state, position)

    do tmp_i = 1, nAtom
       if (flex_select(tmp_i) == 1) THEN
          tmp_j = charmmIdx2ommIdx(tmp_i) + (idx_copy - 1) * num_flex_atom
          call OpenMM_Vec3Array_get(position, tmp_j + 1, vec)
          X(tmp_i) = vec(1) / OpenMM_NmPerAngstrom
          Y(tmp_i) = vec(2) / OpenMM_NmPerAngstrom
          Z(tmp_i) = vec(3) / OpenMM_NmPerAngstrom
       endif
    enddo
  END SUBROUTINE copy_positions_to_main

  SUBROUTINE set_positions(flex_select, charmmIdx2ommIdx, num_flex_atom, idx_copy)
    use psf
    use coord
    use OpenMM

    implicit None
    real*8 vec(3)
    integer tmp_i, tmp_j, tmp_k
    integer idx_copy, num_flex_atom
    integer, allocatable, dimension(:) :: flex_select, charmmIdx2ommIdx

    do tmp_i = 1, nAtom
       if (flex_select(tmp_i) == 1) THEN
          tmp_j = charmmIdx2ommIdx(tmp_i) + (idx_copy - 1) * num_flex_atom
          call xyz2vec3(vec, X(tmp_i)*OpenMM_NmPerAngstrom, Y(tmp_i)*OpenMM_NmPerAngstrom, Z(tmp_i)*OpenMM_NmPerAngstrom)
          call OpenMM_Vec3Array_set(position, tmp_j + 1, vec)
       endif
    enddo
    call OpenMM_Context_setPositions(context, position)
  END SUBROUTINE set_positions

  SUBROUTINE Create_Context()
    use OpenMM
    use omm_main, only: get_plugins

    implicit None

    real*8 temperature, frictioncoef, stepsize
    integer tmp_i, tmp_j, tmp_k

    character(len=10) :: &
         env_string = '', &
         platform_name = ''

    include "plugin_locs.f90"

    temperature = 300
    frictioncoef = 10
    stepsize = 0.0015

    call get_plugins(OPENMM_PLUGIN_DIR)

    ! tmp_i = OpenMM_Platform_getNumPlatforms()
    ! print *, "Num of avaiable platform: ", tmp_i

    ! call OpenMM_LangevinIntegrator_create(langevin_integrator, &
    !      temperature, frictioncoef, stepsize)

    call OpenMM_VerletIntegrator_create(verlet_integrator, stepsize)

    call getenv('OPENMM_PLATFORM', env_string)
    if (trim(env_string) /= '') then ! Try to use user specific platform
       call OpenMM_Platform_getPlatformByName(env_string, platform)
       call OpenMM_Platform_getName(platform, platform_name)

       ! call OpenMM_Context_create_2(context, system, &
       !      transfer(langevin_integrator, OpenMM_Integrator(0)), &
       !      platform)

       call OpenMM_Context_create_2(context, system, &
            transfer(verlet_integrator, OpenMM_Integrator(0)), &
            platform)


    else ! Let OpenMM Context choose best platform.
       ! call OpenMM_Context_create(context, system, &
       !      transfer(langevin_integrator, OpenMM_Integrator(0)))

       call OpenMM_Context_create(context, system, &
            transfer(verlet_integrator, OpenMM_Integrator(0)))

       call OpenMM_Context_getPlatform(context, platform)
    end if
    call OpenMM_Context_setPositions(context, position)
  END SUBROUTINE Create_Context

  SUBROUTINE Print_Energy(flag_hbond)
    use OpenMM
    use stream, only: outu
    use param_store, only: set_param
    implicit None

    real(chm_real) :: energy_bond, energy_angle
    real(chm_real) :: energy_torsion, energy_improper
    real(chm_real) :: energy_cmaps
    real(chm_real) :: energy_vdw_grid, energy_elec_grid, energy_nonbond
    real(chm_real) :: energy_hdonor_grid, energy_hacceptor_grid
    real(chm_real) :: energy_external

    real(chm_real) :: energy_total
    logical :: flag_hbond

    call OpenMM_Context_getState(context, OpenMM_State_Energy, OpenMM_False, state)
    energy_total = OpenMM_State_getPotentialEnergy(state) * OpenMM_KcalPerKJ

    call OpenMM_Context_getState_2(context, OpenMM_State_Energy, OpenMM_False, 2**force_group_bond, state)
    energy_bond = OpenMM_State_getPotentialEnergy(state) * OpenMM_KcalPerKJ

    call OpenMM_Context_getState_2(context, OpenMM_State_Energy, OpenMM_False, 2**force_group_angle, state)
    energy_angle = OpenMM_State_getPotentialEnergy(state) * OpenMM_KcalPerKJ

    call OpenMM_Context_getState_2(context, OpenMM_State_Energy, OpenMM_False, 2**force_group_torsion, state)
    energy_torsion = OpenMM_State_getPotentialEnergy(state) * OpenMM_KcalPerKJ

    call OpenMM_Context_getState_2(context, OpenMM_State_Energy, OpenMM_False, 2**force_group_improper, state)
    energy_improper = OpenMM_State_getPotentialEnergy(state) * OpenMM_KcalPerKJ

    call OpenMM_Context_getState_2(context, OpenMM_State_Energy, OpenMM_False, 2**force_group_cmaps, state)
    energy_cmaps = OpenMM_State_getPotentialEnergy(state) * OpenMM_KcalPerKJ

    call OpenMM_Context_getState_2(context, OpenMM_State_Energy, OpenMM_False, 2**force_group_vdw_grid, state)
    energy_vdw_grid = OpenMM_State_getPotentialEnergy(state) * OpenMM_KcalPerKJ

    call OpenMM_Context_getState_2(context, OpenMM_State_Energy, OpenMM_False, 2**force_group_elec_grid, state)
    energy_elec_grid = OpenMM_State_getPotentialEnergy(state) * OpenMM_KcalPerKJ

    call OpenMM_Context_getState_2(context, OpenMM_State_Energy, OpenMM_False, 2**force_group_nonbond, state)
    energy_nonbond = OpenMM_State_getPotentialEnergy(state) * OpenMM_KcalPerKJ

    call OpenMM_Context_getState_2(context, OpenMM_State_Energy, OpenMM_False, 2**force_group_external, state)
    energy_external = OpenMM_State_getPotentialEnergy(state) * OpenMM_KcalPerKJ

    if (flag_hbond) then
       call OpenMM_Context_getState_2(context, OpenMM_State_Energy, OpenMM_False, 2**force_group_hdonor_grid, state)
       energy_hdonor_grid = OpenMM_State_getPotentialEnergy(state) * OpenMM_KcalPerKJ

       call OpenMM_Context_getState_2(context, OpenMM_State_Energy, OpenMM_False, 2**force_group_hacceptor_grid, state)
       energy_hacceptor_grid = OpenMM_State_getPotentialEnergy(state) * OpenMM_KcalPerKJ
    endif

    write(outu, '(a, x, f15.7)') "Energy total:", energy_total
    write(outu, '(a, x, f15.7)') "Energy bond:", energy_bond
    write(outu, '(a, x, f15.7)') "Energy angle:", energy_angle
    write(outu, '(a, x, f15.7)') "Energy torsion:", energy_torsion
    write(outu, '(a, x, f15.7)') "Energy improper:", energy_improper
    write(outu, '(a, x, f15.7)') "Energy cmaps:", energy_cmaps
    write(outu, '(a, x, f15.7)') "Energy vdw_grid:", energy_vdw_grid
    write(outu, '(a, x, f15.7)') "Energy elec_grid:", energy_elec_grid
    write(outu, '(a, x, f15.7)') "Energy nonbond:", energy_nonbond
    write(outu, '(a, x, f15.7)') "Energy external:", energy_external
    if (flag_hbond) then
       write(outu, '(a, x, f15.7)') "Energy hdonor_grid:", energy_hdonor_grid
       write(outu, '(a, x, f15.7)') "Energy hacceptor_grid:", energy_hacceptor_grid
    endif

    call set_param("OMMDETOT", energy_total)
    call set_param("OMMDBOND", energy_bond)
    call set_param("OMMDANGL", energy_angle)
    call set_param("OMMDDIHE", energy_torsion)
    call set_param("OMMDIMPR", energy_improper)
    call set_param("OMMDCMAP", energy_cmaps)
    call set_param("OMMDVDWG", energy_vdw_grid)
    call set_param("OMMDELEG", energy_elec_grid)
    call set_param("OMMDNBON", energy_nonbond)
    call set_param("OMMDEXTE", energy_external)
    if (flag_hbond) then
       call set_param("OMMDHDOG", energy_hdonor_grid)
       call set_param("OMMDHACG", energy_hacceptor_grid)
    endif

  END SUBROUTINE Print_Energy

  SUBROUTINE Minimize(comlyn, comlen)
    use, intrinsic::iso_c_binding, only: c_null_ptr
    use OpenMM
    use string

    implicit None

    character(len=*) comlyn
    integer comlen

    integer :: num_steps
    num_steps = GTrmI(comlyn, comlen, 'NSTE', 1000)
    call OpenMM_LocalEnergyMinimizer_Minimize(context, 10.0d+0, num_steps &
#if OMM_VER > 80
         , transfer(c_null_ptr, OpenMM_MinimizationReporter(0)) &
#endif /* OMM_VER > 80 */
)

  END SUBROUTINE Minimize

  SUBROUTINE Create_System(omm_select, fix_select, flex_select, hdonor_select, &
                           hacceptor_select, charmmIdx2ommIdx, num_copy_flex, &
                           num_fix_atom, num_flex_atom, flag_hbond)
    use OpenMM
    use memory
    use psf
    use coord
    use select
    use stream, only : outu
    use number, only : zero, one
    use string

    implicit none
    real*8 temperature, frictioncoef, stepsize
    real*8 vec(3)
    logical :: flag_hbond
    integer, allocatable, dimension(:) :: fix_select
    integer, allocatable, dimension(:) :: flex_select
    integer, allocatable, dimension(:) :: omm_select
    integer, allocatable, dimension(:) :: hdonor_select
    integer, allocatable, dimension(:) :: hacceptor_select
    integer, allocatable, dimension(:) :: charmmIdx2ommIdx
    integer :: num_fix_atom, num_flex_atom, num_copy_flex
    integer :: i, j, k, tmp_i, tmp_j, tmp_k

    call OpenMM_System_Create(system)
    call OpenMM_Vec3Array_create(position, num_fix_atom + num_flex_atom*num_copy_flex)

    !! add fixed particles to the system
    j = 0
    do i = 1, nAtom
       if (fix_select(i) == 1) then
          tmp_I = OpenMM_System_addParticle(system, zero)
          charmmIdx2ommIdx(i) = j
          call xyz2vec3(vec, X(i)*OpenMM_NmPerAngstrom, Y(i)*OpenMM_NmPerAngstrom, Z(i)*OpenMM_NmPerAngstrom)
          call OpenMM_Vec3Array_set(position, j+1, vec)
          j = j + 1
       end if
    end do
    !! add several copies of flexible particles to the system
    do k = 1, num_copy_flex
       do i = 1, nAtom
          if (flex_select(i) == 1) then
             tmp_I = OpenMM_System_addParticle(system, amass(i))
             call xyz2vec3(vec, X(i)*OpenMM_NmPerAngstrom, Y(i)*OpenMM_NmPerAngstrom, Z(i)*OpenMM_NmPerAngstrom)
             call OpenMM_Vec3Array_set(position, j+1, vec)
             if (k == 1) then
                charmmIdx2ommIdx(i) = j
             end if
             j = j + 1
          end if
       end do
    end do
    Write(OutU, '(a,i10,i10)') " Number of particles", j, OpenMM_System_getNumParticles(system)

    !! add bond force to the system
    call add_bonds(system, fix_select, num_fix_atom, &
         flex_select, num_flex_atom, &
         charmmIdx2ommIdx, num_copy_flex)

    !! add angle force to the system
    call add_angles(system, fix_select, num_fix_atom, &
         flex_select, num_flex_atom, &
         charmmIdx2ommIdx, num_copy_flex)

    !! add torsion force to the system
    call add_torsions(system, fix_select, num_fix_atom, &
         flex_select, num_flex_atom, &
         charmmIdx2ommIdx, num_copy_flex)

    !! add improper force to the system
    call add_impropers(system, fix_select, num_fix_atom, &
         flex_select, num_flex_atom, &
         charmmIdx2ommIdx, num_copy_flex)

    !! add cmap force to the system
    call add_cmaps(system, fix_select, num_fix_atom, &
         flex_select, num_flex_atom, &
         charmmIdx2ommIdx, num_copy_flex)

    ! ! !! add custom nonbond force to the system
    ! ! call add_custom_nonbonds(system, fix_select, num_fix_atom, &
    ! !      flex_select, num_flex_atom, &
    ! !      charmmIdx2ommIdx, num_copy_flex)

    !! add grid nonbonded force between fixed part and flex part
    call add_vdw_grid_potential(system, fix_select, num_fix_atom, &
                               flex_select, num_flex_atom, &
                               charmmIdx2ommIdx, num_copy_flex, flag_hbond)

    !! add grid nonbonded force between fixed part and flex part
    call add_elec_grid_potential(system, fix_select, num_fix_atom, &
                                 flex_select, num_flex_atom, &
                                 charmmIdx2ommIdx, num_copy_flex)

    if (flag_hbond) then
       !! add hydrogen donor grid force between fixed part and flex part
       !! acceptor atom
       call add_hdonor_grid_potential(system, fix_select, num_fix_atom, &
                                  flex_select, num_flex_atom, &
                                  charmmIdx2ommIdx, num_copy_flex)

       !! add hydrogen acceptor grid force between fixed part and flex part
       !! donor atom
       call add_hacceptor_grid_potential(system, fix_select, num_fix_atom, &
                                  flex_select, num_flex_atom, &
                                  charmmIdx2ommIdx, num_copy_flex)
    endif

    !! add nonbonded force within flex part, excluding 1-2,1-3,1-4 pairs
    call add_nonbonds(system, fix_select, num_fix_atom, &
                      flex_select, num_flex_atom, &
                      charmmIdx2ommIdx, num_copy_flex)

    !! add 1-4 nonbond force to the system. It includes 1-4 nonbonded force
    !! within flex part and 1-4 nonbonded force between fix and flex parts
    call add_14_nonbonds(system, fix_select, num_fix_atom, &
         flex_select, num_flex_atom, &
         charmmIdx2ommIdx, num_copy_flex)

    !! add external force to keep the ligand inside the grid
    call add_external(system, fix_select, num_fix_atom, &
         flex_select, num_flex_atom, &
         charmmIdx2ommIdx, num_copy_flex)


  END SUBROUTINE Create_System

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!! external force to keep ligand inside the grid  !!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  SUBROUTINE add_external(system, fix_select, num_fix_atom, &
                          flex_select, num_flex_atom, &
                          charmmIdx2ommIdx, num_copy_flex)
    use psf, only : nAtom
    implicit None

    type(OpenMM_System) :: system
    type(OpenMM_CustomExternalForce) :: external_force
    type(OpenMM_DoubleArray) :: params
    integer, dimension(:), intent(in) :: fix_select
    integer, dimension(:), intent(in) :: flex_select
    integer, dimension(:), intent(in) :: charmmIdx2ommIdx
    integer, intent(in) :: num_copy_flex, num_fix_atom, num_flex_atom

    integer :: tmp_i, tmp_j, tmp_k, tmp_particle_idx

    character(len=1024) :: energy_expression

    energy_expression = "0.5*external_k*(step(xmin-x)*(x-xmin)^2 + &
                                        &step(x-xmax)*(x-xmax)^2 + &
                                        &step(ymin-y)*(y-ymin)^2 + &
                                        &step(y-ymax)*(y-ymax)^2 + &
                                        &step(zmin-z)*(z-zmin)^2 + &
                                        &step(z-zmax)*(z-zmax)^2);"

    call OpenMM_CustomExternalForce_Create(external_force, energy_expression)
    tmp_k =  OpenMM_CustomExternalForce_addGlobalParameter(external_force, "external_k", GridForce*OpenMM_KJPerKcal/(OpenMM_NmPerAngstrom**2))
    tmp_k =  OpenMM_CustomExternalForce_addGlobalParameter(external_force, "xmin", XGridMin * OpenMM_NmPerAngstrom)
    tmp_k =  OpenMM_CustomExternalForce_addGlobalParameter(external_force, "ymin", YGridMin * OpenMM_NmPerAngstrom)
    tmp_k =  OpenMM_CustomExternalForce_addGlobalParameter(external_force, "zmin", ZGridMin * OpenMM_NmPerAngstrom)
    tmp_k =  OpenMM_CustomExternalForce_addGlobalParameter(external_force, "xmax", XGridMax * OpenMM_NmPerAngstrom)
    tmp_k =  OpenMM_CustomExternalForce_addGlobalParameter(external_force, "ymax", YGridMax * OpenMM_NmPerAngstrom)
    tmp_k =  OpenMM_CustomExternalForce_addGlobalParameter(external_force, "zmax", ZGridMax * OpenMM_NmPerAngstrom)

    call OpenMM_DoubleArray_create(params, 0)

    do tmp_i = 1, nAtom
       if (flex_select(tmp_i) == 1) THEN
          do tmp_j = 1, num_copy_flex
             tmp_particle_idx = charmmIdx2ommIdx(tmp_i) + (tmp_j - 1) * num_flex_atom
             tmp_k = OpenMM_CustomExternalForce_addParticle(external_force, tmp_particle_idx, params)
          enddo
       endif
    enddo

    call OpenMM_Force_setForceGroup(transfer(external_force, OpenMM_Force(0)), force_group_external)
    tmp_i = OpenMM_System_addForce(system, transfer(external_force, OpenMM_Force(0)))

  END SUBROUTINE add_external


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!! Bond Force (including Urey-Bradley force)  !!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine add_bonds(system, fix_select, num_fix_atom, &
                      flex_select, num_flex_atom, &
                      charmmIdx2ommIdx, num_copy_flex)

    !! data structure for bond force
    use psf, only: NBOND, IB, JB
    use code, only: ICB
    use param, only: CBB, CBC

    !! data structure for Urey-Bradley bond force
    use psf, only: NTHETA, IT, JT, KT
    use code, only: ICT
    use param, only: CTUB, CTUC

    use number, only: zero

    type(OpenMM_System) :: system
    type(OpenMM_HarmonicBondForce) :: bonds
    integer, dimension(:), intent(in) :: fix_select
    integer, dimension(:), intent(in) :: flex_select
    integer, dimension(:), intent(in) :: charmmIdx2ommIdx
    integer, intent(in) :: num_copy_flex, num_fix_atom, num_flex_atom
    integer :: i_atom, j_atom, k_atom, idx_bond, tmp_i, tmp_j, tmp_k, tmp_l, junk_i, tmp_idx
    real*8 :: r, k
    call OpenMM_HarmonicBondForce_Create(bonds)

    !! setup bond force
    do idx_bond = 1, NBOND
       i_atom = IB(idx_bond)
       j_atom = JB(idx_bond)
       tmp_idx = ICB(idx_bond)
       r = CBB(tmp_idx) * OpenMM_NmPerAngstrom
       k = CBC(tmp_idx) * OpenMM_KJPerKcal / (OpenMM_NmPerAngstrom**2)

       if (k == 0) then
          cycle
       end if

       !! if the bond is between two atoms which are not in openmm system
       if (omm_select(i_atom) == 0 .or. omm_select(j_atom) == 0) THEN
          cycle
       endif

       !! if the bond is between two fixed atoms in openmm system
       if (fix_select(i_atom) == 1 .and. fix_select(j_atom) == 1) then
          cycle
       end if

       tmp_i = charmmIdx2ommIdx(i_atom)
       tmp_j = charmmIdx2ommIdx(j_atom)

       do tmp_k = 1, num_copy_flex
          if (flex_select(i_atom) == 1) then
             tmp_i = charmmIdx2ommIdx(i_atom) + (tmp_k - 1) * num_flex_atom
          end if
          if (flex_select(j_atom) == 1) then
             tmp_j = charmmIdx2ommIdx(j_atom) + (tmp_k - 1) * num_flex_atom
          end if
          junk_i = OpenMM_HarmonicBondForce_addBond(bonds, &
                   tmp_i, tmp_j, r, 2*k)
       end do
    end do

    !! setup Urey-Bradley bond force
    do idx_bond = 1, NTHETA
       i_atom = IT(idx_bond)
       j_atom = JT(idx_bond)
       k_atom = KT(idx_bond)
       tmp_idx = ICT(idx_bond)
       r = CTUB(tmp_idx) * OpenMM_NmPerAngstrom
       k = CTUC(tmp_idx) * OpenMM_KJPerKcal / (OpenMM_NmPerAngstrom**2)

       if (k == zero) then
          cycle
       end if

       !! if the bond is between two atoms which are not in openmm system
       if (omm_select(i_atom) == 0 .or. omm_select(j_atom) == 0) THEN
          cycle
       endif

       !! if the angle is between fixed atoms, skip it
       if (fix_select(i_atom) == 1 .and. &
           fix_select(k_atom) == 1) then
          cycle
       end if

       tmp_i = charmmIdx2ommIdx(i_atom)
       tmp_k = charmmIdx2ommIdx(k_atom)

       do tmp_l = 1, num_copy_flex
          if (flex_select(i_atom) == 1) then
             tmp_i = charmmIdx2ommIdx(i_atom) + (tmp_l - 1) * num_flex_atom
          end if
          if (flex_select(k_atom) == 1) then
             tmp_k = charmmIdx2ommIdx(k_atom) + (tmp_l - 1) * num_flex_atom
          end if
          junk_i = OpenMM_HarmonicBondForce_addBond(bonds, &
                   tmp_i, tmp_k, r, 2*k)
       end do
    end do

    call OpenMM_Force_setForceGroup(transfer(bonds, OpenMM_Force(0)), force_group_bond)
    tmp_i = OpenMM_System_addForce(system, transfer(bonds, OpenMM_Force(0)))
  end subroutine add_bonds

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!  Angle force   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine add_angles(system, fix_select, num_fix_atom, &
                        flex_select, num_flex_atom, &
                        charmmIdx2ommIdx, num_copy_flex)
    !! data structure for angle force
    use psf, only: NTHETA, IT, JT, KT
    use code, only: ICT
    use param, only: CTB, CTC
    use number, only: zero

    type(OpenMM_System) :: system
    type(OpenMM_HarmonicAngleForce) :: angles
    integer, dimension(:), intent(in) :: fix_select
    integer, dimension(:), intent(in) :: flex_select
    integer, dimension(:), intent(in) :: charmmIdx2ommIdx
    integer, intent(in) :: num_copy_flex, num_fix_atom, num_flex_atom
    integer :: i_atom, j_atom, k_atom, idx_angle, tmp_i, tmp_j, tmp_k, tmp_l, junk_i
    real*8 :: theta, k

    call OpenMM_HarmonicAngleForce_Create(angles)

    !! setup angle force
    do idx_angle = 1, NTHETA
       i_atom = IT(idx_angle)
       j_atom = JT(idx_angle)
       k_atom = KT(idx_angle)
       tmp_i = ICT(idx_angle)
       theta = CTB(tmp_i)
       k = CTC(tmp_i) * OpenMM_KJPerKcal

       if (k == zero) then
          cycle
       end if

       !! if the angle is between atoms which are not in openmm system, skip it
       if (omm_select(i_atom) == 0 .or. &
           omm_select(j_atom) == 0 .or. &
           omm_select(k_atom) == 0) then
          cycle
       end if

       !! if the angle is between fixed atoms, skip it
       if (fix_select(i_atom) == 1 .and. &
           fix_select(j_atom) == 1 .and. &
           fix_select(k_atom) == 1) then
          cycle
       end if

       !! if the angle atoms involve flex atoms
       tmp_i = charmmIdx2ommIdx(i_atom)
       tmp_j = charmmIdx2ommIdx(j_atom)
       tmp_k = charmmIdx2ommIdx(k_atom)

       do tmp_l = 1, num_copy_flex
          if (flex_select(i_atom) == 1) then
             tmp_i = charmmIdx2ommIdx(i_atom) + (tmp_l - 1) * num_flex_atom
          end if
          if (flex_select(j_atom) == 1) then
             tmp_j = charmmIdx2ommIdx(j_atom) + (tmp_l - 1) * num_flex_atom
          end if
          if (flex_select(k_atom) == 1) then
             tmp_k = charmmIdx2ommIdx(k_atom) + (tmp_l - 1) * num_flex_atom
          end if
          junk_i = OpenMM_HarmonicAngleForce_addAngle(angles, tmp_i, tmp_j, tmp_k, theta, 2*k)
       end do
    end do
    call OpenMM_Force_setForceGroup(transfer(angles, OpenMM_Force(0)), force_group_angle)
    tmp_l = OpenMM_System_addForce(system, transfer(angles, OpenMM_Force(0)))
  end subroutine add_angles

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!  torsion force   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine add_torsions(system, fix_select, num_fix_atom, &
                          flex_select, num_flex_atom, &
                          charmmIdx2ommIdx, num_copy_flex)
    !! data structure for torsion force
    use psf, only: NPHI, IP, JP, KP, LP
    use code, only: ICP
    use param, only: CPB, CPC, CPD
    use number, only: zero

    type(OpenMM_System) :: system
    type(OpenMM_PeriodicTorsionForce) :: torsions
    integer, dimension(:), intent(in) :: fix_select
    integer, dimension(:), intent(in) :: flex_select
    integer, dimension(:), intent(in) :: charmmIdx2ommIdx
    integer, intent(in) :: num_copy_flex, num_fix_atom, num_flex_atom
    integer :: i_atom, j_atom, k_atom, l_atom, idx_torsion, tmp_i, tmp_j, tmp_k, tmp_l, tmp_m, junk_i, tmp_idx, iper
    real*8 :: phi, k

    if (NPHI <= 0) return
    call OpenMM_PeriodicTorsionForce_Create(torsions)

    !! setup torsion force
    do idx_torsion = 1, NPHI
       i_atom = IP(idx_torsion)
       j_atom = JP(idx_torsion)
       k_atom = abs(KP(idx_torsion))
       l_atom = abs(LP(idx_torsion))

       tmp_idx = ICP(idx_torsion)

       if (tmp_idx == 0) cycle

       !! if the angle is between atoms which are not in openmm system, skip it
       if (omm_select(i_atom) == 0 .or. &
           omm_select(j_atom) == 0 .or. &
           omm_select(k_atom) == 0 .or. &
           omm_select(l_atom) == 0) then
          cycle
       end if

       !! if the torsion is between fixed atoms, skip it
       if (fix_select(i_atom) == 1 .and. &
           fix_select(j_atom) == 1 .and. &
           fix_select(k_atom) == 1 .and. &
           fix_select(l_atom) == 1) then
          cycle
       end if

       !! if the torsion atoms involve flex atoms
       tmp_i = charmmIdx2ommIdx(i_atom)
       tmp_j = charmmIdx2ommIdx(j_atom)
       tmp_k = charmmIdx2ommIdx(k_atom)
       tmp_l = charmmIdx2ommIdx(l_atom)

       do tmp_m = 1, num_copy_flex
          if (flex_select(i_atom) == 1) then
             tmp_i = charmmIdx2ommIdx(i_atom) + (tmp_m - 1) * num_flex_atom
          end if
          if (flex_select(j_atom) == 1) then
             tmp_j = charmmIdx2ommIdx(j_atom) + (tmp_m - 1) * num_flex_atom
          end if
          if (flex_select(k_atom) == 1) then
             tmp_k = charmmIdx2ommIdx(k_atom) + (tmp_m - 1) * num_flex_atom
          end if
          if (flex_select(l_atom) == 1) then
             tmp_l = charmmIdx2ommIdx(l_atom) + (tmp_m - 1) * num_flex_atom
          end if

          tmp_idx = ICP(idx_torsion)

          do
             iper = CPD(tmp_idx)
             phi = CPB(tmp_idx)
             k = CPC(tmp_idx) * OpenMM_KJPerKcal
             junk_i = OpenMM_PeriodicTorsionForce_addTorsion(torsions, &
                  tmp_i, tmp_j, tmp_k, tmp_l, abs(iper), phi, k)
             if (iper >= 0) exit
             tmp_idx = tmp_idx + 1
          end do
       end do
    end do
    call OpenMM_Force_setForceGroup(transfer(torsions, OpenMM_Force(0)), force_group_torsion)
    junk_i = OpenMM_System_addForce(system, transfer(torsions, OpenMM_Force(0)))
  end subroutine add_torsions

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!  improper force   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine add_impropers(system, fix_select, num_fix_atom, &
                           flex_select, num_flex_atom, &
                           charmmIdx2ommIdx, num_copy_flex)
    !! data structure for torsion force
    use psf, only: NIMPHI, IM, JM, KM, LM
    use code, only: ICI
    use param, only: CIB, CIC, CPD
    use number, only: zero

    type(OpenMM_System) :: system
    type(OpenMM_CustomTorsionForce) :: impropers
    type(OpenMM_DoubleArray) :: params
    integer, dimension(:), intent(in) :: fix_select
    integer, dimension(:), intent(in) :: flex_select
    integer, dimension(:), intent(in) :: charmmIdx2ommIdx
    integer, intent(in) :: num_copy_flex, num_fix_atom, num_flex_atom
    integer :: i_atom, j_atom, k_atom, l_atom, idx_torsion, tmp_i, tmp_j, tmp_k, tmp_l, tmp_m, junk_i, tmp_idx, iper
    real*8 :: p1, p2

    if (NIMPHI <= 0) return
    call OpenMM_CustomTorsionForce_Create(impropers, &
         "k * (min(diff, 2*pi-diff)^2); diff = abs(theta - theta0); pi=3.141592653589793")
    junk_i = OpenMM_CustomTorsionForce_addPerTorsionParameter(impropers, 'k')
    junk_i = OpenMM_CustomTorsionForce_addPerTorsionParameter(impropers, 'theta0')

    call OpenMM_DoubleArray_create(params, 2)

    !! setup torsion force
    do idx_torsion = 1, NIMPHI
       i_atom = IM(idx_torsion)
       j_atom = JM(idx_torsion)
       k_atom = abs(KM(idx_torsion))
       l_atom = abs(LM(idx_torsion))

       tmp_idx = ICI(idx_torsion)

       if (tmp_idx == 0) cycle

       !! if the angle is between atoms which are not in openmm system, skip it
       if (omm_select(i_atom) == 0 .or. &
           omm_select(j_atom) == 0 .or. &
           omm_select(k_atom) == 0 .or. &
           omm_select(l_atom) == 0) then
          cycle
       end if

       !! if the torsion is between fixed atoms, skip it
       if (fix_select(i_atom) == 1 .and. &
           fix_select(j_atom) == 1 .and. &
           fix_select(k_atom) == 1 .and. &
           fix_select(l_atom) == 1) then
          cycle
       end if

       !! if the torsion atoms involve flex atoms
       tmp_i = charmmIdx2ommIdx(i_atom)
       tmp_j = charmmIdx2ommIdx(j_atom)
       tmp_k = charmmIdx2ommIdx(k_atom)
       tmp_l = charmmIdx2ommIdx(l_atom)

       do tmp_m = 1, num_copy_flex
          if (flex_select(i_atom) == 1) then
             tmp_i = charmmIdx2ommIdx(i_atom) + (tmp_m - 1) * num_flex_atom
          end if
          if (flex_select(j_atom) == 1) then
             tmp_j = charmmIdx2ommIdx(j_atom) + (tmp_m - 1) * num_flex_atom
          end if
          if (flex_select(k_atom) == 1) then
             tmp_k = charmmIdx2ommIdx(k_atom) + (tmp_m - 1) * num_flex_atom
          end if
          if (flex_select(l_atom) == 1) then
             tmp_l = charmmIdx2ommIdx(l_atom) + (tmp_m - 1) * num_flex_atom
          end if

          p1 = CIC(tmp_idx) * OpenMM_KJPerKcal
          p2 = CIB(tmp_idx)

          call OpenMM_DoubleArray_set(params, 1, p1)
          call OpenMM_DoubleArray_set(params, 2, p2)

          junk_i = OpenMM_CustomTorsionForce_addTorsion(impropers, &
               tmp_i, tmp_j, tmp_k, tmp_l, params)
       end do
    end do
    call OpenMM_Force_setForceGroup(transfer(impropers, OpenMM_Force(0)), force_group_improper)
    junk_i = OpenMM_System_addForce(system, transfer(impropers, OpenMM_Force(0)))
  end subroutine add_impropers

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!  cmap force   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine add_cmaps(system, fix_select, num_fix_atom, &
                       flex_select, num_flex_atom, &
                       charmmIdx2ommIdx, num_copy_flex)
    use psf
    use code, only: ICCT
    use cmapm
    use omm_bonded, only: transform_cmap

    type(OpenMM_System), intent(inout) :: system
    type(OpenMM_CMAPTorsionForce) :: cmaps
    type(OpenMM_DoubleArray) :: omm_map
    integer, dimension(:), intent(in) :: fix_select
    integer, dimension(:), intent(in) :: flex_select
    integer, dimension(:), intent(in) :: charmmIdx2ommIdx
    integer, intent(in) :: num_copy_flex, num_fix_atom, num_flex_atom
    integer :: i_atom1, j_atom1, k_atom1, l_atom1, &
               i_atom2, j_atom2, k_atom2, l_atom2, &
               tmp_i_atom1, tmp_j_atom1, tmp_k_atom1, tmp_l_atom1, &
               tmp_i_atom2, tmp_j_atom2, tmp_k_atom2, tmp_l_atom2, &
               idx_map, idx_term, &
               tmp_i, tmp_j, tmp_k, tmp_l, tmp_m, junk_i, tmp_idx, map_size

    if (NCTP <= 0) return
    call OpenMM_CMAPTorsionForce_create(cmaps)

    do idx_map = 1, NCTP
       map_size = MCTP(idx_map)%grid(1)%len1
       call OpenMM_DoubleArray_create(omm_map, map_size**2)
       call transform_cmap(omm_map, map_size, MCTP(idx_map)%grid(1)%a)
       junk_i = OpenMM_CMAPTorsionForce_addMap(cmaps, map_size, omm_map)
       call OpenMM_DoubleArray_destroy(omm_map)
    end do

    do idx_term = 1, NCRTERM
       idx_map = ICCT(idx_term)
       i_atom1 = I1CT(idx_term)
       j_atom1 = J1CT(idx_term)
       k_atom1 = K1CT(idx_term)
       l_atom1 = L1CT(idx_term)
       i_atom2 = I2CT(idx_term)
       j_atom2 = J2CT(idx_term)
       k_atom2 = K2CT(idx_term)
       l_atom2 = L2CT(idx_term)


       if (omm_select(i_atom1) == 0 .or. &
           omm_select(j_atom1) == 0 .or. &
           omm_select(k_atom1) == 0 .or. &
           omm_select(l_atom1) == 0 .or. &
           omm_select(i_atom2) == 0 .or. &
           omm_select(j_atom2) == 0 .or. &
           omm_select(k_atom2) == 0 .or. &
           omm_select(l_atom2) == 0) then
          cycle
       end if

       if (fix_select(i_atom1) == 1 .and. &
           fix_select(j_atom1) == 1 .and. &
           fix_select(k_atom1) == 1 .and. &
           fix_select(l_atom1) == 1 .and. &
           fix_select(i_atom2) == 1 .and. &
           fix_select(j_atom2) == 1 .and. &
           fix_select(k_atom2) == 1 .and. &
           fix_select(l_atom2) == 1) then
          cycle
       end if

       tmp_i_atom1 = charmmIdx2ommIdx(i_atom1)
       tmp_j_atom1 = charmmIdx2ommIdx(j_atom1)
       tmp_k_atom1 = charmmIdx2ommIdx(k_atom1)
       tmp_l_atom1 = charmmIdx2ommIdx(l_atom1)

       tmp_i_atom2 = charmmIdx2ommIdx(i_atom2)
       tmp_j_atom2 = charmmIdx2ommIdx(j_atom2)
       tmp_k_atom2 = charmmIdx2ommIdx(k_atom2)
       tmp_l_atom2 = charmmIdx2ommIdx(l_atom2)

       do tmp_m = 1, num_copy_flex
          if (flex_select(i_atom1) == 1) then
             tmp_i_atom1 = charmmIdx2ommIdx(i_atom1) + (tmp_m - 1) * num_flex_atom
          end if
          if (flex_select(j_atom1) == 1) then
             tmp_j_atom1 = charmmIdx2ommIdx(j_atom1) + (tmp_m - 1) * num_flex_atom
          end if
          if (flex_select(k_atom1) == 1) then
             tmp_k_atom1 = charmmIdx2ommIdx(k_atom1) + (tmp_m - 1) * num_flex_atom
          end if
          if (flex_select(l_atom1) == 1) then
             tmp_l_atom1 = charmmIdx2ommIdx(l_atom1) + (tmp_m - 1) * num_flex_atom
          end if

          if (flex_select(i_atom2) == 1) then
             tmp_i_atom2 = charmmIdx2ommIdx(i_atom2) + (tmp_m - 1) * num_flex_atom
          end if
          if (flex_select(j_atom2) == 1) then
             tmp_j_atom2 = charmmIdx2ommIdx(j_atom2) + (tmp_m - 1) * num_flex_atom
          end if
          if (flex_select(k_atom2) == 1) then
             tmp_k_atom2 = charmmIdx2ommIdx(k_atom2) + (tmp_m - 1) * num_flex_atom
          end if
          if (flex_select(l_atom2) == 1) then
             tmp_l_atom2 = charmmIdx2ommIdx(l_atom2) + (tmp_m - 1) * num_flex_atom
          end if
          junk_i = OpenMM_CMAPTorsionForce_addTorsion(cmaps, idx_map, &
               tmp_i_atom1, tmp_j_atom1, tmp_k_atom1, tmp_l_atom1, &
               tmp_i_atom2, tmp_j_atom2, tmp_k_atom2, tmp_l_atom2)
       end do
    end do
    if (OpenMM_CMAPTorsionForce_getNumTorsions(cmaps) > 0 ) then
       call OpenMM_Force_setForceGroup(transfer(cmaps, OpenMM_Force(0)), force_group_cmaps)
       junk_i = OpenMM_System_addForce(system, transfer(cmaps, OpenMM_Force(0)))
    end if
  end subroutine add_cmaps

  SUBROUTINE add_elec_grid_potential(system, fix_select, num_fix_atom, &
                                    flex_select, num_flex_atom, &
                                    charmmIdx2ommIdx, num_copy_flex)
    use memory
    use psf, only: natom, CG
    implicit none

    type(OpenMM_System), intent(inout) :: system
    integer, dimension(:), intent(in) :: fix_select
    integer, dimension(:), intent(in) :: flex_select
    integer, dimension(:), intent(in) :: charmmIdx2ommIdx
    integer, intent(in) :: num_copy_flex, num_fix_atom, num_flex_atom

    type(OpenMM_Continuous3DFunction) :: soft_grid_function
    type(OpenMM_Continuous3DFunction) :: hard_grid_function
    type(OpenMM_DoubleArray) :: grid_values
    type(OpenMM_DoubleArray) :: tmp_grid_values

    type(OpenMM_CustomCompoundBondForce) :: grid_force
    character(len=1024) :: energy_expression

    integer tmp_i, tmp_j, tmp_k, tmp_l
    character(len=2) :: tmp_c

    type(OpenMM_IntArray) :: tmp_particle_idx
    type(OpenMM_DoubleArray) :: tmp_param

    call OpenMM_DoubleArray_Create(grid_values, XGridNum*YGridNum*ZGridNum)

    do tmp_i = 1, XGridNum
       do tmp_j = 1, YGridNum
          do tmp_k = 1, ZGridNum
             call OpenMM_DoubleArray_Set(grid_values, &
                                         ((tmp_k - 1)*YGridNum + tmp_j - 1)*XGridNum + tmp_i, &
                                         SoftGridPot((((NumGrid - 1)*XGridNum + tmp_i - 1)*YGridNum + tmp_j - 1)*ZGridNum + tmp_k))
          enddo
       enddo
    enddo
    call OpenMM_Continuous3DFunction_Create(soft_grid_function, &
                                            XGridNum, YGridNum, ZGridNum, &
                                            grid_values, &
                                            XGridMin*OpenMM_NmPerAngstrom, XGridMax*OpenMM_NmPerAngstrom, &
                                            YGridMin*OpenMM_NmPerAngstrom, YGridMax*OpenMM_NmPerAngstrom, &
                                            ZGridMin*OpenMM_NmPerAngstrom, ZGridMax*OpenMM_NmPerAngstrom &
#if OMM_VER >= 75
                                            , 0 &
#endif
                                            )

    do tmp_i = 1, XGridNum
       do tmp_j = 1, YGridNum
          do tmp_k = 1, ZGridNum
             call OpenMM_DoubleArray_Set(grid_values, &
                                         ((tmp_k - 1)*YGridNum + tmp_j - 1)*XGridNum + tmp_i, &
                                         HardGridPot((((NumGrid - 1)*XGridNum + tmp_i - 1)*YGridNum + tmp_j - 1)*ZGridNum + tmp_k))
          enddo
       enddo
    enddo
    call OpenMM_Continuous3DFunction_Create(hard_grid_function, &
                                            XGridNum, YGridNum, ZGridNum, &
                                            grid_values, &
                                            XGridMin*OpenMM_NmPerAngstrom, XGridMax*OpenMM_NmPerAngstrom, &
                                            YGridMin*OpenMM_NmPerAngstrom, YGridMax*OpenMM_NmPerAngstrom, &
                                            ZGridMin*OpenMM_NmPerAngstrom, ZGridMax*OpenMM_NmPerAngstrom &
#if OMM_VER >= 75
                                            , 0 &
#endif
)

    !! make grid forces
    energy_expression = "(soft_p*soft_elec(x1,y1,z1) + hard_p*hard_elec(x1,y1,z1))*charge"
    call OpenMM_CustomCompoundBondForce_Create(grid_force, 1, trim(energy_expression))
    tmp_i = OpenMM_CustomCompoundBondForce_AddTabulatedFunction( &
                       grid_force, "soft_elec", &
                       transfer(soft_grid_function, OpenMM_TabulatedFunction(0)))
    tmp_i = OpenMM_CustomCompoundBondForce_AddTabulatedFunction( &
                       grid_force, "hard_elec", &
                       transfer(hard_grid_function, OpenMM_TabulatedFunction(0)))
    tmp_i = OpenMM_CustomCompoundBondForce_AddGlobalParameter(grid_force, "soft_p", 1.0D0)
    tmp_i = OpenMM_CustomCompoundBondForce_AddGlobalParameter(grid_force, "hard_p", 0.0D0)
    tmp_i = OpenMM_CustomCompoundBondForce_AddPerBondParameter(grid_force, "charge")

    !! add particles
    call OpenMM_DoubleArray_Create(tmp_param, 1)
    call OpenMM_IntArray_Create(tmp_particle_idx, 1)
    do tmp_i = 1, nAtom
       if (omm_select(tmp_i) == 0) THEN
          cycle
       endif

       if (fix_select(tmp_i) == 1) THEN
          cycle
       endif

       if (flex_select(tmp_i) == 1) THEN
          call OpenMM_DoubleArray_Set(tmp_param, 1, CG(tmp_i)*OpenMM_KJPerKcal)
          do tmp_j = 1, num_copy_flex
             call OpenMM_IntArray_Set(tmp_particle_idx, 1, charmmIdx2ommIdx(tmp_i) + (tmp_j - 1)*num_flex_atom)
             tmp_l = OpenMM_CustomCompoundBondForce_AddBond(grid_force, tmp_particle_idx, tmp_param)
          enddo
       endif
    enddo

    !! add force to system
    call OpenMM_Force_setForceGroup(transfer(grid_force, OpenMM_Force(0)), force_group_elec_grid)
    tmp_i =  OpenMM_System_addForce(system, transfer(grid_force, OpenMM_Force(0)))

  END SUBROUTINE add_elec_grid_potential

  !-----------------------------------------------------------------------------
  ! Begin of computing energy and force term for hydrogen bond grid
  ! Added by Yujin Wu
  ! At 2020/06/16
  !-----------------------------------------------------------------------------

  SUBROUTINE add_hdonor_grid_potential(system, fix_select, num_fix_atom, &
                                    flex_select, num_flex_atom, &
                                    charmmIdx2ommIdx, num_copy_flex)
    use memory
    use psf, only: natom
    implicit none

    type(OpenMM_System), intent(inout) :: system
    integer, dimension(:), intent(in) :: fix_select
    integer, dimension(:), intent(in) :: flex_select
    integer, dimension(:), intent(in) :: charmmIdx2ommIdx
    integer, intent(in) :: num_copy_flex, num_fix_atom, num_flex_atom

    type(OpenMM_Continuous3DFunction) :: soft_grid_function
    type(OpenMM_Continuous3DFunction) :: hard_grid_function
    type(OpenMM_DoubleArray) :: grid_values
    type(OpenMM_DoubleArray) :: tmp_grid_values

    type(OpenMM_CustomCompoundBondForce) :: grid_force
    character(len=1024) :: energy_expression

    integer tmp_i, tmp_j, tmp_k, tmp_l
    character(len=2) :: tmp_c

    type(OpenMM_IntArray) :: tmp_particle_idx
    type(OpenMM_DoubleArray) :: tmp_param

    call OpenMM_DoubleArray_Create(grid_values, XGridNum*YGridNum*ZGridNum)

    do tmp_i = 1, XGridNum
       do tmp_j = 1, YGridNum
          do tmp_k = 1, ZGridNum
             call OpenMM_DoubleArray_Set(grid_values, &
                                         ((tmp_k - 1)*YGridNum + tmp_j - 1)*XGridNum + tmp_i, &
                                         SoftGridPot((((NumGrid - 3)*XGridNum + tmp_i - 1)*YGridNum + tmp_j - 1)*ZGridNum + tmp_k))
          enddo
       enddo
    enddo
    call OpenMM_Continuous3DFunction_Create(soft_grid_function, &
                                            XGridNum, YGridNum, ZGridNum, &
                                            grid_values, &
                                            XGridMin*OpenMM_NmPerAngstrom, XGridMax*OpenMM_NmPerAngstrom, &
                                            YGridMin*OpenMM_NmPerAngstrom, YGridMax*OpenMM_NmPerAngstrom, &
                                            ZGridMin*OpenMM_NmPerAngstrom, ZGridMax*OpenMM_NmPerAngstrom &
#if OMM_VER >= 75
                                            , 0 &
#endif
)

    do tmp_i = 1, XGridNum
       do tmp_j = 1, YGridNum
          do tmp_k = 1, ZGridNum
             call OpenMM_DoubleArray_Set(grid_values, &
                                         ((tmp_k - 1)*YGridNum + tmp_j - 1)*XGridNum + tmp_i, &
                                         HardGridPot((((NumGrid - 3)*XGridNum + tmp_i - 1)*YGridNum + tmp_j - 1)*ZGridNum + tmp_k))
          enddo
       enddo
    enddo
    call OpenMM_Continuous3DFunction_Create(hard_grid_function, &
                                            XGridNum, YGridNum, ZGridNum, &
                                            grid_values, &
                                            XGridMin*OpenMM_NmPerAngstrom, XGridMax*OpenMM_NmPerAngstrom, &
                                            YGridMin*OpenMM_NmPerAngstrom, YGridMax*OpenMM_NmPerAngstrom, &
                                            ZGridMin*OpenMM_NmPerAngstrom, ZGridMax*OpenMM_NmPerAngstrom &
#if OMM_VER >= 75
                                            , 0 &
#endif
)

    !! make grid forces
    energy_expression = "(soft_p*soft_hdonor(x1,y1,z1) + hard_p*hard_hdonor(x1,y1,z1))*hacceptor"
    call OpenMM_CustomCompoundBondForce_Create(grid_force, 1, trim(energy_expression))
    tmp_i = OpenMM_CustomCompoundBondForce_AddTabulatedFunction( &
                       grid_force, "soft_hdonor", &
                       transfer(soft_grid_function, OpenMM_TabulatedFunction(0)))
    tmp_i = OpenMM_CustomCompoundBondForce_AddTabulatedFunction( &
                       grid_force, "hard_hdonor", &
                       transfer(hard_grid_function, OpenMM_TabulatedFunction(0)))
    tmp_i = OpenMM_CustomCompoundBondForce_AddGlobalParameter(grid_force, "soft_p", 1.0D0)
    tmp_i = OpenMM_CustomCompoundBondForce_AddGlobalParameter(grid_force, "hard_p", 0.0D0)
    tmp_i = OpenMM_CustomCompoundBondForce_AddPerBondParameter(grid_force, "hacceptor")

    !! add particles
    call OpenMM_DoubleArray_Create(tmp_param, 1)
    call OpenMM_IntArray_Create(tmp_particle_idx, 1)
    do tmp_i = 1, nAtom
       if (omm_select(tmp_i) == 0) THEN
          cycle
       endif

       if (fix_select(tmp_i) == 1) THEN
          cycle
       endif

       if (flex_select(tmp_i) == 1) THEN
          call OpenMM_DoubleArray_Set(tmp_param, 1, hacceptor_select(tmp_i)*OpenMM_KJPerKcal)
          do tmp_j = 1, num_copy_flex
             call OpenMM_IntArray_Set(tmp_particle_idx, 1, charmmIdx2ommIdx(tmp_i) + (tmp_j - 1)*num_flex_atom)
             tmp_l = OpenMM_CustomCompoundBondForce_AddBond(grid_force, tmp_particle_idx, tmp_param)
          enddo
       endif
    enddo

    !! add force to system
    call OpenMM_Force_setForceGroup(transfer(grid_force, OpenMM_Force(0)), force_group_hdonor_grid)
    tmp_i =  OpenMM_System_addForce(system, transfer(grid_force, OpenMM_Force(0)))

  END SUBROUTINE add_hdonor_grid_potential

  SUBROUTINE add_hacceptor_grid_potential(system, fix_select, num_fix_atom, &
                                    flex_select, num_flex_atom, &
                                    charmmIdx2ommIdx, num_copy_flex)
    use memory
    use psf, only: natom
    implicit none

    type(OpenMM_System), intent(inout) :: system
    integer, dimension(:), intent(in) :: fix_select
    integer, dimension(:), intent(in) :: flex_select
    integer, dimension(:), intent(in) :: charmmIdx2ommIdx
    integer, intent(in) :: num_copy_flex, num_fix_atom, num_flex_atom

    type(OpenMM_Continuous3DFunction) :: soft_grid_function
    type(OpenMM_Continuous3DFunction) :: hard_grid_function
    type(OpenMM_DoubleArray) :: grid_values
    type(OpenMM_DoubleArray) :: tmp_grid_values

    type(OpenMM_CustomCompoundBondForce) :: grid_force
    character(len=1024) :: energy_expression

    integer tmp_i, tmp_j, tmp_k, tmp_l
    character(len=2) :: tmp_c

    type(OpenMM_IntArray) :: tmp_particle_idx
    type(OpenMM_DoubleArray) :: tmp_param

    call OpenMM_DoubleArray_Create(grid_values, XGridNum*YGridNum*ZGridNum)

    do tmp_i = 1, XGridNum
       do tmp_j = 1, YGridNum
          do tmp_k = 1, ZGridNum
             call OpenMM_DoubleArray_Set(grid_values, &
                                         ((tmp_k - 1)*YGridNum + tmp_j - 1)*XGridNum + tmp_i, &
                                         SoftGridPot((((NumGrid - 2)*XGridNum + tmp_i - 1)*YGridNum + tmp_j - 1)*ZGridNum + tmp_k))
          enddo
       enddo
    enddo
    call OpenMM_Continuous3DFunction_Create(soft_grid_function, &
                                            XGridNum, YGridNum, ZGridNum, &
                                            grid_values, &
                                            XGridMin*OpenMM_NmPerAngstrom, XGridMax*OpenMM_NmPerAngstrom, &
                                            YGridMin*OpenMM_NmPerAngstrom, YGridMax*OpenMM_NmPerAngstrom, &
                                            ZGridMin*OpenMM_NmPerAngstrom, ZGridMax*OpenMM_NmPerAngstrom &
#if OMM_VER >= 75
                                            , 0 &
#endif
)

    do tmp_i = 1, XGridNum
       do tmp_j = 1, YGridNum
          do tmp_k = 1, ZGridNum
             call OpenMM_DoubleArray_Set(grid_values, &
                                         ((tmp_k - 1)*YGridNum + tmp_j - 1)*XGridNum + tmp_i, &
                                         HardGridPot((((NumGrid - 2)*XGridNum + tmp_i - 1)*YGridNum + tmp_j - 1)*ZGridNum + tmp_k))
          enddo
       enddo
    enddo
    call OpenMM_Continuous3DFunction_Create(hard_grid_function, &
                                            XGridNum, YGridNum, ZGridNum, &
                                            grid_values, &
                                            XGridMin*OpenMM_NmPerAngstrom, XGridMax*OpenMM_NmPerAngstrom, &
                                            YGridMin*OpenMM_NmPerAngstrom, YGridMax*OpenMM_NmPerAngstrom, &
                                            ZGridMin*OpenMM_NmPerAngstrom, ZGridMax*OpenMM_NmPerAngstrom &
#if OMM_VER >= 75
                                            , 0 &
#endif
)

    !! make grid forces
    energy_expression = "(soft_p*soft_hacceptor(x1,y1,z1) + hard_p*hard_hacceptor(x1,y1,z1))*hdonor"
    call OpenMM_CustomCompoundBondForce_Create(grid_force, 1, trim(energy_expression))
    tmp_i = OpenMM_CustomCompoundBondForce_AddTabulatedFunction( &
                       grid_force, "soft_hacceptor", &
                       transfer(soft_grid_function, OpenMM_TabulatedFunction(0)))
    tmp_i = OpenMM_CustomCompoundBondForce_AddTabulatedFunction( &
                       grid_force, "hard_hacceptor", &
                       transfer(hard_grid_function, OpenMM_TabulatedFunction(0)))
    tmp_i = OpenMM_CustomCompoundBondForce_AddGlobalParameter(grid_force, "soft_p", 1.0D0)
    tmp_i = OpenMM_CustomCompoundBondForce_AddGlobalParameter(grid_force, "hard_p", 0.0D0)
    tmp_i = OpenMM_CustomCompoundBondForce_AddPerBondParameter(grid_force, "hdonor")

    !! add particles
    call OpenMM_DoubleArray_Create(tmp_param, 1)
    call OpenMM_IntArray_Create(tmp_particle_idx, 1)
    do tmp_i = 1, nAtom
       if (omm_select(tmp_i) == 0) THEN
          cycle
       endif

       if (fix_select(tmp_i) == 1) THEN
          cycle
       endif

       if (flex_select(tmp_i) == 1) THEN
          call OpenMM_DoubleArray_Set(tmp_param, 1, hdonor_select(tmp_i)*OpenMM_KJPerKcal)
          do tmp_j = 1, num_copy_flex
             call OpenMM_IntArray_Set(tmp_particle_idx, 1, charmmIdx2ommIdx(tmp_i) + (tmp_j - 1)*num_flex_atom)
             tmp_l = OpenMM_CustomCompoundBondForce_AddBond(grid_force, tmp_particle_idx, tmp_param)
          enddo
       endif
    enddo

    !! add force to system
    call OpenMM_Force_setForceGroup(transfer(grid_force, OpenMM_Force(0)), force_group_hacceptor_grid)
    tmp_i =  OpenMM_System_addForce(system, transfer(grid_force, OpenMM_Force(0)))

  END SUBROUTINE add_hacceptor_grid_potential

  !-----------------------------------------------------------------------------
  ! End of computing energy and force term for hydrogen bond grid
  ! Added by Yujin Wu
  ! At 2020/06/16
  !-----------------------------------------------------------------------------

  SUBROUTINE add_vdw_grid_potential(system, fix_select, num_fix_atom, &
                                 flex_select, num_flex_atom, &
                                 charmmIdx2ommIdx, num_copy_flex, flag_hbond)
    use memory
    use stream, only: outu
    use psf, only: natom, IAC
    use param, only: ITC, VDWR, EFF
    use fftdock, only: Get_Idx_Nearest_Radii
    implicit none

    type(OpenMM_System), intent(inout) :: system
    integer, dimension(:), intent(in) :: fix_select
    integer, dimension(:), intent(in) :: flex_select
    integer, dimension(:), intent(in) :: charmmIdx2ommIdx
    integer, intent(in) :: num_copy_flex, num_fix_atom, num_flex_atom

    integer, allocatable, dimension(:) :: idx_vdw_grid_atom
    logical, allocatable, dimension(:) :: flag_vdw_grid_used
    integer, allocatable, dimension(:) :: idx_vdw_grid_used
    integer, allocatable, dimension(:) :: idx_vdw_grid_to_position

    integer :: num_vdw_grid_used = 0

    type(OpenMM_Continuous3DFunction), allocatable, dimension(:) :: soft_grid_functions
    type(OpenMM_Continuous3DFunction), allocatable, dimension(:) :: hard_grid_functions
    type(OpenMM_DoubleArray) :: grid_values
    type(OpenMM_DoubleArray) :: tmp_grid_values

    type(OpenMM_CustomCompoundBondForce), allocatable, dimension(:) :: grid_forces
    character(len=1024) :: energy_expression

    integer tmp_i, tmp_j, tmp_k, tmp_l
    character(len=2) :: tmp_c

    type(OpenMM_IntArray) :: tmp_particle_idx
    type(OpenMM_DoubleArray) :: tmp_param

    logical :: flag_hbond
    integer :: NumVdwGrid

    if (flag_hbond) then
       NumVdwGrid = NumGrid - 3
    else
       NumVdwGrid = NumGrid - 1
    EndIf
    Write(OutU, '(a,i4)') " Number of vdw grids = ", NumVdwGrid

    CALL chmalloc('openmm_dock.src', 'add_grid_potential', 'flag_vdw_grid_used', NumVdwGrid, log = flag_vdw_grid_used)
    flag_vdw_grid_used(:) = .FALSE.
    CALL chmalloc('openmm_dock.src', 'add_grid_potential', 'idx_vdw_grid', natom, intg = idx_vdw_grid_atom)
    idx_vdw_grid_atom(:) = 0

    do tmp_i = 1, nAtom
       if (flex_select(tmp_i) == 1) THEN
          tmp_j = Get_Idx_Nearest_Radii(VDWR(ITC(IAC(tmp_I))), GridRadii, NumVdwGrid)
          idx_vdw_grid_atom(tmp_i) = tmp_j
          flag_vdw_grid_used(tmp_J) = .TRUE.
       endif
    enddo

    num_vdw_grid_used = 0
    do tmp_i = 1, NumVdwGrid
       if (flag_vdw_grid_used(tmp_i)) THEN
          num_vdw_grid_used = num_vdw_grid_used + 1
       endif
    enddo

    ! print *, "num_vdw_grid_used:", num_vdw_grid_used
    CALL chmalloc('openmm_dock.src', 'add_grid_potential', 'idx_vdw_grid_used', num_vdw_grid_used, intg = idx_vdw_grid_used)
    CALL chmalloc('openmm_dock.src', 'add_grid_potential', 'idx_vdw_grid_to_position', NumVdwGrid, intg = idx_vdw_grid_to_position)
    idx_vdw_grid_to_position(:) = 0

    tmp_j = 1
    do tmp_i = 1, NumVdwGrid
       if (flag_vdw_grid_used(tmp_i)) THEN
          idx_vdw_grid_used(tmp_j) = tmp_i
          idx_vdw_grid_to_position(tmp_i) = tmp_j
          tmp_j = tmp_j + 1
       endif
    enddo

    !! make Continuous3DFunctions
    allocate(soft_grid_functions(num_vdw_grid_used))
    allocate(hard_grid_functions(num_vdw_grid_used))

    call OpenMM_DoubleArray_Create(grid_values, XGridNum*YGridNum*ZGridNum)

    do tmp_l = 1, num_vdw_grid_used
       !! soft grid potential
       do tmp_i = 1, XGridNum
          do tmp_j = 1, YGridNum
             do tmp_k = 1, ZGridNum
                call OpenMM_DoubleArray_Set(grid_values, &
                                           ((tmp_k - 1)*YGridNum + tmp_j - 1)*XGridNum + tmp_i, &
                                           SoftGridPot((((idx_vdw_grid_used(tmp_l) - 1)*XGridNum + tmp_i - 1)*YGridNum + tmp_j - 1)*ZGridNum + tmp_k))
             enddo
          enddo
       enddo
       call OpenMM_Continuous3DFunction_Create(soft_grid_functions(tmp_l), &
                                              XGridNum, YGridNum, ZGridNum, &
                                              grid_values, &
                                              XGridMin*OpenMM_NmPerAngstrom, XGridMax*OpenMM_NmPerAngstrom, &
                                              YGridMin*OpenMM_NmPerAngstrom, YGridMax*OpenMM_NmPerAngstrom, &
                                              ZGridMin*OpenMM_NmPerAngstrom, ZGridMax*OpenMM_NmPerAngstrom &
#if OMM_VER >= 75
                                            , 0 &
#endif
)

       !! hard grid potential
       do tmp_i = 1, XGridNum
          do tmp_j = 1, YGridNum
             do tmp_k = 1, ZGridNum
                call OpenMM_DoubleArray_Set(grid_values, &
                                           ((tmp_k - 1)*YGridNum + tmp_j - 1)*XGridNum + tmp_i, &
                                           HardGridPot((((idx_vdw_grid_used(tmp_l) - 1)*XGridNum + tmp_i - 1)*YGridNum + tmp_j - 1)*ZGridNum + tmp_k))
             enddo
          enddo
       enddo
       call OpenMM_Continuous3DFunction_Create(hard_grid_functions(tmp_l), &
                                              XGridNum, YGridNum, ZGridNum, &
                                              grid_values, &
                                              XGridMin*OpenMM_NmPerAngstrom, XGridMax*OpenMM_NmPerAngstrom, &
                                              YGridMin*OpenMM_NmPerAngstrom, YGridMax*OpenMM_NmPerAngstrom, &
                                              ZGridMin*OpenMM_NmPerAngstrom, ZGridMax*OpenMM_NmPerAngstrom &
#if OMM_VER >= 75
                                            , 0 &
#endif
)

    enddo

    !! make grid forces
    allocate(grid_forces(num_vdw_grid_used))
    do tmp_l = 1, num_vdw_grid_used
       write(tmp_c, '(I0)') tmp_l
       energy_expression = "(soft_p * soft_vdw"//trim(tmp_c)//"(x1,y1,z1) + hard_p * hard_vdw"//trim(tmp_c)//"(x1, y1, z1))*sqrt_eps"
       ! print *, trim(energy_expression)
       call OpenMM_CustomCompoundBondForce_Create(grid_forces(tmp_l), 1, trim(energy_expression))
       tmp_i = OpenMM_CustomCompoundBondForce_AddTabulatedFunction( &
                       grid_forces(tmp_l), "soft_vdw"//trim(tmp_c), &
                       transfer(soft_grid_functions(tmp_l), OpenMM_TabulatedFunction(0)))
       tmp_i = OpenMM_CustomCompoundBondForce_AddTabulatedFunction( &
                       grid_forces(tmp_l), "hard_vdw"//trim(tmp_c), &
                       transfer(hard_grid_functions(tmp_l), OpenMM_TabulatedFunction(0)))
       tmp_i = OpenMM_CustomCompoundBondForce_AddGlobalParameter(grid_forces(tmp_l), "soft_p", 1.0D0)
       tmp_i = OpenMM_CustomCompoundBondForce_AddGlobalParameter(grid_forces(tmp_l), "hard_p", 0.0D0)
       tmp_i = OpenMM_CustomCompoundBondForce_AddPerBondParameter(grid_forces(tmp_l), "sqrt_eps")
    enddo

    !! add particles
    call OpenMM_DoubleArray_Create(tmp_param, 1)
    call OpenMM_IntArray_Create(tmp_particle_idx, 1)
    do tmp_i = 1, nAtom
       if (omm_select(tmp_i) == 0) THEN
          cycle
       endif

       if (fix_select(tmp_i) == 1) THEN
          cycle
       endif

       if (flex_select(tmp_i) == 1) THEN
          call OpenMM_DoubleArray_Set(tmp_param, 1, sqrt(-EFF(ITC(IAC(tmp_i)))) * OpenMM_KJPerKcal)
          do tmp_j = 1, num_copy_flex
             call OpenMM_IntArray_Set(tmp_particle_idx, 1, charmmIdx2ommIdx(tmp_i) + (tmp_j - 1)*num_flex_atom)
             tmp_l = OpenMM_CustomCompoundBondForce_AddBond(grid_forces(idx_vdw_grid_to_position(idx_vdw_grid_atom(tmp_i))), tmp_particle_idx, tmp_param)
          enddo
       endif
    enddo

    !! add force to system
    do tmp_l = 1, num_vdw_grid_used
       call OpenMM_Force_setForceGroup(transfer(grid_forces(tmp_l), OpenMM_Force(0)), force_group_vdw_grid)
       tmp_i =  OpenMM_System_addForce(system, transfer(grid_forces(tmp_l), OpenMM_Force(0)))
    enddo

  END SUBROUTINE add_vdw_grid_potential



  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!  Custom Nonbonded force   !!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine add_custom_nonbonds(system, fix_select, num_fix_atom, &
                         flex_select, num_flex_atom, &
                         charmmIdx2ommIdx, num_copy_flex)
    use psf, only: natom, IAC, NBOND, IB, JB, CG
    use param, only: ITC, VDWR, EFF
    use bases_fcm, only: BNBND

    type(OpenMM_System), intent(inout) :: system
    type(OpenMM_CustomNonbondedForce) :: nonbonds
    type(OpenMM_DoubleArray) :: params
    type(OpenMM_BondArray) :: bonds
    integer, dimension(:), intent(in) :: fix_select
    integer, dimension(:), intent(in) :: flex_select
    integer, dimension(:), intent(in) :: charmmIdx2ommIdx
    integer, intent(in) :: num_copy_flex, num_fix_atom, num_flex_atom
    integer i_atom, j_atom, junk_i, tmp_i_atom, tmp_j_atom,  idx, tmp_k, tmp_l
    integer, allocatable, dimension(:) :: tmp1_select, tmp2_select
    real*8 p1, p2, p3

    real*8 switching_distance, cutoff_distance

    type(OpenMM_IntSet) :: all_atom_idx, flex_atom_idx

    call OpenMM_CustomNonbondedForce_create(nonbonds, &
         "0 * 4*epsilon*((sigma/r)^12 - (sigma/r)^6) + 138.935456*charge/r; sigma = 0.5*(sigma1+sigma2); epsilon= sqrt(epsilon1*epsilon2); charge = charge1 * charge2")
    junk_i = OpenMM_CustomNonbondedForce_addPerParticleParameter(nonbonds, "epsilon")
    junk_i = OpenMM_CustomNonbondedForce_addPerParticleParameter(nonbonds, "sigma")
    junk_i = OpenMM_CustomNonbondedForce_addPerParticleParameter(nonbonds, "charge")
    call OpenMM_DoubleArray_create(params, 3)

    !! add fixed particles
    do i_atom = 1, natom
       if (fix_select(i_atom) == 1) then
          idx = ITC(IAC(i_atom))
          p1 = -OpenMM_KJPerKcal * EFF(idx)
          p2 = OpenMM_SigmaPerVdwRadius * VDWR(idx) * OpenMM_NmPerAngstrom
          p3 = CG(i_atom)
          call OpenMM_DoubleArray_set(params, 1, p1)
          call OpenMM_DoubleArray_set(params, 2, p2)
          call OpenMM_DoubleArray_set(params, 3, p3)
          junk_i = OpenMM_CustomNonbondedForce_addParticle(nonbonds, params)
       end if
    end do

    !! add multiple copies of flexible particles
    do tmp_k = 1, num_copy_flex
       do i_atom = 1, natom
          if (flex_select(i_atom) == 1) then
          idx = ITC(IAC(i_atom))
          p1 = -OpenMM_KJPerKcal * EFF(idx)
          p2 = OpenMM_SigmaPerVdwRadius * VDWR(idx) * OpenMM_NmPerAngstrom
          p3 = CG(i_atom)
          call OpenMM_DoubleArray_set(params, 1, p1)
          call OpenMM_DoubleArray_set(params, 2, p2)
          call OpenMM_DoubleArray_set(params, 3, p3)
          junk_i = OpenMM_CustomNonbondedForce_addParticle(nonbonds, params)
          end if
       end do
    end do

    cutoff_distance = 9.0
    switching_distance = 8.8
    call OpenMM_CustomNonbondedForce_setUseSwitchingFunction(nonbonds, OpenMM_True)
    call OpenMM_CustomNonbondedForce_setSwitchingDistance(nonbonds, switching_distance)
    call OpenMM_CustomNonbondedForce_setCutoffDistance(nonbonds, cutoff_distance)

    !! generate bond array used for adding nonbond exclusion
    call OpenMM_BondArray_create(bonds, 0)
    do tmp_k = 1, NBOND
       i_atom = IB(tmp_k)
       j_atom = JB(tmp_k)
       if (fix_select(i_atom) == 1 .and. fix_select(j_atom) == 1) then
          cycle
       endif
       tmp_i_atom = charmmIdx2ommIdx(i_atom)
       tmp_j_atom = charmmIdx2ommIdx(j_atom)
       do tmp_l = 1, num_copy_flex
          if (flex_select(i_atom) == 1) then
             tmp_i_atom = charmmIdx2ommIdx(i_atom) + (tmp_l - 1) * num_flex_atom
          end if
          if (flex_select(j_atom) == 1) then
             tmp_j_atom = charmmIdx2ommIdx(j_atom) + (tmp_l - 1) * num_flex_atom
          end if
          call OpenMM_BondArray_append(bonds, tmp_i_atom, tmp_j_atom)
       end do
    end do

    ! tmp1_select is the flag for fixed atoms which are one bond awary from a flex atom
    allocate(tmp1_select(natom))
    tmp1_select(:) = 0
    do tmp_k = 1, NBOND
       i_atom = IB(tmp_k)
       j_atom = JB(tmp_k)
       if (flex_select(i_atom) == 1 .and. fix_select(j_atom) == 1) then
          tmp1_select(j_atom) = 1
       end if
       if (flex_select(j_atom) == 1 .and. fix_select(i_atom) == 1) then
          tmp1_select(i_atom) = 1
       end if
    end do

    ! tmp2_select is the flag for fixed atoms which are two bonds awary from a flex atom
    allocate(tmp2_select(natom))
    tmp2_select(:) = 0
    do tmp_k = 1, NBOND
       i_atom = IB(tmp_k)
       j_atom = JB(tmp_k)
       if ((fix_select(i_atom) == 1 .and. tmp1_select(j_atom) == 1)) then
          call OpenMM_BondArray_append(bonds, charmmIdx2ommIdx(i_atom), charmmIdx2ommIdx(j_atom))
          tmp2_select(i_atom) = 1
       end if
       if ((fix_select(j_atom) == 1 .and. tmp1_select(i_atom) == 1)) then
          call OpenMM_BondArray_append(bonds, charmmIdx2ommIdx(i_atom), charmmIdx2ommIdx(j_atom))
          tmp2_select(j_atom) = 1
       end if
    end do

    do tmp_k = 1, NBOND
       i_atom = IB(tmp_k)
       j_atom = JB(tmp_k)
       if ((fix_select(i_atom) == 1 .and. tmp2_select(j_atom) == 1)) then
          call OpenMM_BondArray_append(bonds, charmmIdx2ommIdx(i_atom), charmmIdx2ommIdx(j_atom))
       end if
       if ((fix_select(j_atom) == 1 .and. tmp2_select(i_atom) == 1)) then
          call OpenMM_BondArray_append(bonds, charmmIdx2ommIdx(i_atom), charmmIdx2ommIdx(j_atom))
       end if
    end do

    call OpenMM_CustomNonbondedForce_createExclusionsFromBonds(nonbonds, bonds, 3)

    !! add interaction group
    do tmp_k = 1, num_copy_flex
       call OpenMM_IntSet_create(all_atom_idx)
       call OpenMM_IntSet_create(flex_atom_idx)

       do i_atom = 1, num_fix_atom
          call OpenMM_IntSet_insert(all_atom_idx, i_atom - 1)
       end do

       do i_atom = 1, num_flex_atom
          call OpenMM_IntSet_insert(flex_atom_idx, num_fix_atom + (tmp_k - 1) * num_flex_atom + i_atom - 1)
          call OpenMM_IntSet_insert(all_atom_idx, num_fix_atom + (tmp_k - 1) * num_flex_atom + i_atom - 1)
       end do

       junk_i = OpenMM_CustomNonbondedForce_addInteractionGroup(nonbonds, all_atom_idx, flex_atom_idx)
       call OpenMM_IntSet_destroy(flex_atom_idx)
       call OpenMM_IntSet_destroy(all_atom_idx)
    end do
    junk_i = OpenMM_System_addForce(system, transfer(nonbonds, OpenMM_Force(0)))
  end subroutine add_custom_nonbonds


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!  1-4 nonbonded force   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine add_14_nonbonds(system, fix_select, num_fix_atom, &
                         flex_select, num_flex_atom, &
                         charmmIdx2ommIdx, num_copy_flex)
    use bases_fcm, only: BNBND
    use psf, only: natom, IAC, MAXATC, CG
    use param, only: ITC, VDWR, EFF
    type(OpenMM_System), intent(inout) :: system
    type(OpenMM_CustomBondForce) :: bonds_14
    type(OpenMM_DoubleArray) :: params
    integer, dimension(:), intent(in) :: fix_select
    integer, dimension(:), intent(in) :: flex_select
    integer, dimension(:), intent(in) :: charmmIdx2ommIdx
    integer, intent(in) :: num_copy_flex, num_fix_atom, num_flex_atom
    integer :: i_atom, j_atom, istrt, iend, ipair, tmp_i_atom, tmp_j_atom, i_copy
    integer :: ikind, jkind
    integer :: itype, jtype
    real(chm_real) :: tmp_epsilon, tmp_rmin, tmp_cg
    integer :: junk_i

    call OpenMM_CustomBondForce_create(bonds_14, &
         "step(r-rc)*epsilon*((rmin/r)^12 - 2*(rmin/r)^6) + step(rc-r)*(Emax/2 + fc * (r-rc)) + &
          select(charge, step(r-elec_rc)*138.935456*charge/(eps*r^2) + step(elec_rc-r)*(cute/2 + elec_fc * (r-elec_rc)), 0); &
          fc=12*epsilon*(rmin_over_rc_6/rc - rmin_over_rc_6^2/rc);&
          rc=rmin/rmin_over_rc_6^(1.0/6); &
          rmin_over_rc_6 = 1 + sqrt(1 + Emax/2/epsilon); &
          elec_fc = -2*138.935456*charge/(eps*elec_rc^3); &
          elec_rc = sqrt(2*138.935456*abs(charge)/(abs(cute)*eps)); &
          cute = step(charge)*maxe + step(-charge)*mine;")

    junk_i = OpenMM_CustomBondForce_addGlobalParameter(bonds_14, "Emax", 0.6D0)
    junk_i = OpenMM_CustomBondForce_addGlobalParameter(bonds_14, "maxe", 1.0D0)
    junk_i = OpenMM_CustomBondForce_addGlobalParameter(bonds_14, "mine", -1.0D0)
    junk_i = OpenMM_CustomBondForce_addGlobalParameter(bonds_14, "eps", 1.0D0)
    junk_i = OpenMM_CustomBondForce_addPerBondParameter(bonds_14, "epsilon")
    junk_i = OpenMM_CustomBondForce_addPerBondParameter(bonds_14, "rmin")
    junk_i = OpenMM_CustomBondForce_addPerBondParameter(bonds_14, "charge")
    call OpenMM_DoubleArray_create(params, 3)

    istrt = 1
    do i_atom = 1, natom
       if (i_atom > 1) istrt = max(BNBND%IBLO14(i_atom - 1) + 1, 1)
       iend = BNBND%IBLO14(i_atom)
       do ipair = istrt, iend
          j_atom = abs(BNBND%INB14(ipair))
          if (BNBND%INB14(ipair) < 0) then
             ikind = IAC(i_atom)
             jkind = IAC(j_atom)
             itype = ITC(ikind) + MAXATC
             jtype = ITC(jkind) + MAXATC
             tmp_epsilon = sqrt(abs(EFF(itype) * EFF(jtype)))
             tmp_rmin = VDWR(itype) + VDWR(jtype)
             tmp_cg = CG(i_atom) * CG(j_atom)

             if (omm_select(i_atom) == 0 .or. omm_select(j_atom) == 0) THEN
                cycle
             endif
             if (fix_select(i_atom) == 1 .and. fix_select(j_atom) == 1) then
                cycle
             end if

             tmp_i_atom = charmmIdx2ommIdx(i_atom)
             tmp_j_atom = charmmIdx2ommIdx(j_atom)
             do i_copy = 1, num_copy_flex
                if (flex_select(i_atom) == 1) then
                   tmp_i_atom = charmmIdx2ommIdx(i_atom) + &
                        (i_copy - 1) * num_flex_atom
                end if
                if (flex_select(j_atom) == 1) then
                   tmp_j_atom = charmmIdx2ommIdx(j_atom) + &
                        (i_copy - 1) * num_flex_atom
                end if
                call OpenMM_DoubleArray_set(params, 1, tmp_epsilon * OpenMM_KJPerKcal)
                call OpenMM_DoubleArray_set(params, 2, tmp_rmin * OpenMM_NmPerAngstrom)
                call OpenMM_DoubleArray_set(params, 3, tmp_cg)
                junk_i = OpenMM_CustomBondForce_addBond(bonds_14, tmp_i_atom, tmp_j_atom, params)
             end do
          end if
       end do
    end do
    call OpenMM_Force_setForceGroup(transfer(bonds_14, OpenMM_Force(0)), force_group_nonbond)
    junk_i = OpenMM_System_addForce(system, transfer(bonds_14, OpenMM_Force(0)))
  end subroutine add_14_nonbonds

  SUBROUTINE add_nonbonds(system, fix_select, num_fix_atom, &
                          flex_select, num_flex_atom, &
                          charmmIdx2ommIdx, num_copy_flex)
    use bases_fcm, only: BNBND
    use psf, only: natom, IAC, MAXATC, CG
    use param, only: ITC, VDWR, EFF
    type(OpenMM_System), intent(inout) :: system
    type(OpenMM_CustomBondForce) :: bonds
    type(OpenMM_DoubleArray) :: params
    integer, dimension(:), intent(in) :: fix_select
    integer, dimension(:), intent(in) :: flex_select
    integer, dimension(:), intent(in) :: charmmIdx2ommIdx
    integer, intent(in) :: num_copy_flex, num_fix_atom, num_flex_atom
    integer :: i_atom, j_atom, istrt, iend, ipair, tmp_i_atom, tmp_j_atom, i_copy
    integer :: ikind, jkind
    integer :: itype, jtype
    integer :: junk_i, tmp_i
    logical :: flag_excl
    real(chm_real) :: tmp_epsilon, tmp_rmin, tmp_cg

    call OpenMM_CustomBondForce_create(bonds, &
         "step(r-rc)*epsilon*((rmin/r)^12 - 2*(rmin/r)^6) + step(rc-r)*(Emax/2 + fc * (r-rc)) + &
          select(charge, step(r-elec_rc)*138.935456*charge/(eps*r^2) + step(elec_rc-r)*(cute/2 + elec_fc * (r-elec_rc)), 0); &
          fc=12*epsilon*(rmin_over_rc_6/rc - rmin_over_rc_6^2/rc);&
          rc=rmin/rmin_over_rc_6^(1.0/6); &
          rmin_over_rc_6 = 1 + sqrt(1 + Emax/2/epsilon); &
          elec_fc = -2*138.935456*charge/(eps*elec_rc^3); &
          elec_rc = sqrt(2*138.935456*abs(charge)/(abs(cute)*eps)); &
          cute = step(charge)*maxe + step(-charge)*mine;")

    junk_i = OpenMM_CustomBondForce_addGlobalParameter(bonds, "Emax", 0.6D0)
    junk_i = OpenMM_CustomBondForce_addGlobalParameter(bonds, "maxe", 1.0D0)
    junk_i = OpenMM_CustomBondForce_addGlobalParameter(bonds, "mine", -1.0D0)
    junk_i = OpenMM_CustomBondForce_addGlobalParameter(bonds, "eps", 1.0D0)
    junk_i = OpenMM_CustomBondForce_addPerBondParameter(bonds, "epsilon")
    junk_i = OpenMM_CustomBondForce_addPerBondParameter(bonds, "rmin")
    junk_i = OpenMM_CustomBondForce_addPerBondParameter(bonds, "charge")
    call OpenMM_DoubleArray_create(params, 3)

    do i_atom = 1, nAtom - 1
       if (omm_select(i_atom) == 0) THEN
          cycle
       endif

       if (fix_select(i_atom) == 1) THEN
          do j_atom = 1, nAtom
             if (flex_select(j_atom) == 1) then
                if ( .not. test_excl(i_atom, j_atom, BNBND%IBLO14, BNBND%INB14)) THEN
                   ikind = IAC(i_atom)
                   jkind = IAC(j_atom)
                   itype = ITC(ikind)
                   jtype = ITC(jkind)
                   tmp_epsilon = sqrt(abs(EFF(itype) * EFF(jtype)))
                   tmp_rmin = VDWR(itype) + VDWR(jtype)
                   tmp_cg = CG(i_atom) * CG(j_atom)
                   do i_copy = 1, num_copy_flex
                      tmp_i_atom = charmmIdx2ommIdx(i_atom)
                      tmp_j_atom = charmmIdx2ommIdx(j_atom) + (i_copy - 1)*num_flex_atom
                      call OpenMM_DoubleArray_set(params, 1, tmp_epsilon * OpenMM_KJPerKcal)
                      call OpenMM_DoubleArray_set(params, 2, tmp_rmin * OpenMM_NmPerAngstrom)
                      call OpenMM_DoubleArray_set(params, 3, tmp_cg)
                      junk_i = OpenMM_CustomBondForce_addBond(bonds, tmp_i_atom, tmp_j_atom, params)
                   enddo
                endif
             endif
          enddo
       endif

       if (flex_select(i_atom) == 1) then
          do j_atom = i_atom + 1, nAtom
             if (flex_select(j_atom) == 1) then
                if ( .not. test_excl(i_atom, j_atom, BNBND%IBLO14, BNBND%INB14)) THEN
                   ikind = IAC(i_atom)
                   jkind = IAC(j_atom)
                   itype = ITC(ikind)
                   jtype = ITC(jkind)
                   tmp_epsilon = sqrt(abs(EFF(itype) * EFF(jtype)))
                   tmp_rmin = VDWR(itype) + VDWR(jtype)
                   tmp_cg = CG(i_atom) * CG(j_atom)
                   do i_copy = 1, num_copy_flex
                      tmp_i_atom = charmmIdx2ommIdx(i_atom) + (i_copy - 1)*num_flex_atom
                      tmp_j_atom = charmmIdx2ommIdx(j_atom) + (i_copy - 1)*num_flex_atom
                      call OpenMM_DoubleArray_set(params, 1, tmp_epsilon * OpenMM_KJPerKcal)
                      call OpenMM_DoubleArray_set(params, 2, tmp_rmin * OpenMM_NmPerAngstrom)
                      call OpenMM_DoubleArray_set(params, 3, tmp_cg)
                      junk_i = OpenMM_CustomBondForce_addBond(bonds, tmp_i_atom, tmp_j_atom, params)
                   enddo
                endif
             endif
          enddo
       endif
    enddo
    call OpenMM_Force_setForceGroup(transfer(bonds, OpenMM_Force(0)), force_group_nonbond)
    junk_i = OpenMM_System_addForce(system, transfer(bonds, OpenMM_Force(0)))
  END SUBROUTINE add_nonbonds

  FUNCTION test_excl(i_atom, j_atom, IBLO14, INB14) result(flag)
    use psf, only: natom
    use bases_fcm, only: BNBND
    implicit None

    integer :: i_atom, j_atom
    integer :: istrt, iend, ipair
    integer, dimension(:) :: IBLO14, INB14
    logical :: flag
    integer :: max_idx, min_idx
    flag = .FALSE.

    min_idx = min(i_atom, j_atom)
    max_idx = max(i_atom, j_atom)
    istrt = 1
    if (min_idx > 1) istrt = max(BNBND%IBLO14(min_idx - 1) + 1, 1)
    iend = BNBND%IBLO14(min_idx)
    do ipair = istrt, iend
       if ( max_idx == abs(BNBND%INB14(ipair))) THEN
          flag = .TRUE.
       endif
    enddo

    if (i_atom == j_atom) flag = .TRUE.

  END FUNCTION test_excl

  SUBROUTINE print_excl_list()
    use stream, only: outu
    use bases_fcm, only: BNBND
    use psf, only: natom

    implicit None

    integer :: istrt, iend, ipair, i_atom, j_atom

    write(outu, '(i5)') BNBND%IBLO14
    write(outu, '(i5)') BNBND%INB14
    write(outu, '(a)') "++++++++++++++++++"
    istrt = 1
    do i_atom = 1, natom
       if (i_atom > 1) istrt = max(BNBND%IBLO14(i_atom - 1) + 1, 1)
       iend = BNBND%IBLO14(i_atom)
       do ipair = istrt, iend
          j_atom = BNBND%INB14(ipair)
          write(outu, '(i5, x, i5)') i_atom, j_atom
       enddo
    enddo
  END SUBROUTINE print_excl_list

  SUBROUTINE Read_Grid_Potentials(softGridUnit, hardGridUnit, flag_hbond, Form)
    use grid_dock, only: ReadGrid
    use string
    use stream
    use ctitla
    use memory

    implicit None

    integer softGridUnit, hardGridUnit, num_grid_points
    logical form, flag_hbond
    integer tmp_i, tmp_j, tmp_k, tmp_l, tmp_idx, ngrid0
    real(chm_real),allocatable,dimension(:,:,:,:) :: tmp_grid_pot

    real(chm_real) tmp_XGridCenter, tmp_YGridCenter, tmp_ZGridCenter
    real(chm_real) tmp_XGridLen, tmp_YGridLen, tmp_ZGridLen
    real(chm_real) tmp_XGridMin, tmp_YGridMin, tmp_ZGridMin
    real(chm_real) tmp_XGridMax, tmp_YGridMax, tmp_ZGridMax
    real(chm_real) tmp_DGrid, tmp_GridForce
    integer tmp_NumGrid, tmp_XGridNum, tmp_YGridNum, tmp_ZGridNum

    !  Read binary file of Grid Potentials
    If (.not. Form) Then
       !! soft grid potential
       If(Prnlev .ge.2) Write(OutU,'(a,i4)') &
            ' Soft grid potentials read from binary file on unit', &
            softGridUnit
       Call Rdtitl(TitleB, NtitlB, softGridUnit, -1)
       Call Wrtitl(TitleB, NtitlB, OutU, 0)
       Read(softGridUnit) NumGrid, XGridNum, YGridNum, ZGridNum
       Read(softGridUnit) XGridCenter, YGridCenter, ZGridCenter
       Read(softGridUnit) XGridLen, YGridLen, ZGridLen, DGrid, GridForce

       !! hard grid potential
       If(Prnlev .ge.2) Write(OutU,'(a,i4)') &
            ' Hard grid potentials read from binary file on unit', &
            hardGridUnit
       Call Rdtitl(TitleB, NtitlB, hardGridUnit, -1)
       Call Wrtitl(TitleB, NtitlB, OutU, 0)
       Read(hardGridUnit) tmp_NumGrid, tmp_XGridNum, tmp_YGridNum, tmp_ZGridNum
       Read(hardGridUnit) tmp_XGridCenter, tmp_YGridCenter, tmp_ZGridCenter
       Read(hardGridUnit) tmp_XGridLen, tmp_YGridLen, tmp_ZGridLen, tmp_DGrid, tmp_GridForce

    Else
       !! soft grid potential
       Write(OutU,'(a,i4)') &
            ' Soft grid potentials read from formatted file on unit', &
            softGridUnit
       Call Rdtitl(TitleB, NtitlB, softGridUnit, 0)
       Call Wrtitl(TitleB, NtitlB, OutU, 0)
       Read(softGridUnit,'(4(i5,1x))') NumGrid, XGridNum, YGridNum, ZGridNum
       Read(softGridUnit,'(3(f12.5,1x))') XGridCenter, YGridCenter, ZGridCenter
       Read(softGridUnit,'(5(f12.5,1x))') XGridLen, YGridLen, ZGridLen, DGrid, GridForce

       !! hard grid potential
       Write(OutU,'(a,i4)') &
            ' Hard grid potentials read from formatted file on unit', &
            hardGridUnit
       Call Rdtitl(TitleB, NtitlB, hardGridUnit, 0)
       Call Wrtitl(TitleB, NtitlB, OutU, 0)
       Read(hardGridUnit,'(4(i5,1x))') tmp_NumGrid, tmp_XGridNum, tmp_YGridNum, tmp_ZGridNum
       Read(hardGridUnit,'(3(f12.5,1x))') tmp_XGridCenter, tmp_YGridCenter, tmp_ZGridCenter
       Read(hardGridUnit,'(5(f12.5,1x))') tmp_XGridLen, tmp_YGridLen, tmp_ZGridLen, tmp_DGrid, tmp_GridForce

    endif

    !! check that soft and hard grids have the same setup parameters
    if (tmp_NumGrid /= NumGrid .OR. &
        tmp_XGridNum /= XGridNum .OR. &
        tmp_YGridNum /= YGridNum .OR. &
        tmp_ZGridNum /= ZGridNum) then
        call WRNDIE(-2, "OMMDOCK Read Potential", "Soft and hard grid potential have different num of grid points!")
     endif

     if (abs(tmp_XGridCenter-XGridCenter) > 1e-4 .OR. &
         abs(tmp_YGridCenter-YGridCenter) > 1e-4 .OR. &
         abs(tmp_ZGridCenter-ZGridCenter) > 1e-4 .OR. &
         abs(tmp_XGridLen-XGridLen) > 1e-4 .OR. &
         abs(tmp_YGridLen-YGridLen) > 1e-4 .OR. &
         abs(tmp_ZGridLen-ZGridLen) > 1e-4 .OR. &
         abs(tmp_DGrid - DGrid) > 1e-4 .OR. &
         abs(tmp_GridForce - GridForce) > 1e-4 &
         ) THEN
        call WRNDIE(-2, "OMMDOCK Read Potential", "Soft and hard grid potential have positios or length!")
     endif

    XGridMin = XGridCenter - XGridLen/2.0
    YGridMin = YGridCenter - YGridLen/2.0
    ZGridMin = ZGridCenter - ZGridLen/2.0

    XGridMax = XGridCenter + XGridLen/2.0
    YGridMax = YGridCenter + YGridLen/2.0
    ZGridMax = ZGridCenter + ZGridLen/2.0

    !  Write data for read parameters
    If (PrnLev .ge. 2 .and. flag_hbond) Write(OutU,'(a,i4,a)') &
         ' GridSetUp: soft and hard Grid potentials will be set-up for', &
         NumGrid-3,' atom types , plus 2 hyrogen bonds grid, plus electrostatics'
    If (PrnLev .ge. 2 .and. .not. flag_hbond) Write(OutU,'(a,i4,a)') &
         ' GridSetUp: soft and hard Grid potentials will be set-up for', &
         NumGrid-1,' atom types , plus electrostatics'
    If (PrnLev .ge. 2) Write(OutU,'(a,i4, a, i4)') &
         ' GridSetUp: and read from unit', softGridUnit, ' and unit', hardGridUnit
    If (PrnLev .ge. 2) Write(OutU,'(a,3f10.5)') &
         ' GridSetUp: Grid centered at', &
         XGridCenter, YGridCenter, ZGridCenter
    If (PrnLev .ge. 2) Write(OutU,'(3(a,f10.5,a,f10.5/))') &
         ' GridSetUp: Grid runs from (X)', &
         -XGridLen/2+XGridCenter,' -', XGridLen/2+XGridCenter, &
         ' GridSetUp: soft Grid runs from (Y)', &
         -YGridLen/2+YGridCenter,' -', YGridLen/2+YGridCenter, &
         ' GridSetUp: soft Grid runs from (Z)', &
         -ZGridLen/2+ZGridCenter,' -', ZGridLen/2+ZGridCenter
    If (PrnLev .ge. 2) Write(OutU,'(a,f10.5)') &
         ' GridSetUp: With a grid spacing of', DGrid
    If (PrnLev .ge. 2) Write(OutU,'(a,f10.3,a)') &
         ' GridSetUp: Force constant at grid edge set to ', &
         GridForce,' kcal/mol/A^2'

    call chmalloc('openmm_dock.src','read_potential_grid','tmp_grid_pot',NumGrid, XGridNum, YGridNum, ZGridNum,crl = tmp_grid_pot)

    If (flag_hbond) then
       NGrid0 = NumGrid - 3
    else
       NGrid0 = NumGrid - 1
    Endif
    call chmalloc('openmm_dock.src','read_potential_grid','GridRadii',NGrid0,crl=GridRadii)

    !! read soft grid values
    Call ReadGrid(tmp_grid_pot, GridRadii, NGrid0, &
         NumGrid, XGridNum, YGridNum, ZGridNum, &
         softGridUnit, Form)
    num_grid_points = NumGrid * XGridNum * YGridNum * ZGridNum
    call chmalloc('openmm_dock.src', 'Read_Potential_Grid', 'SoftGridPot', num_grid_points, crl = SoftGridPot)
    SoftGridPot(:) = 0.0
    do tmp_l = 1, NumGrid
       do tmp_i = 1, XGridNum
          do tmp_j = 1, YGridNum
             do tmp_k = 1, ZGridNum
                tmp_idx = (((tmp_l - 1) * XGridNum + tmp_i - 1) * YGridNum + tmp_j - 1) * ZGridNum + tmp_k
                SoftGridPot(tmp_idx) = tmp_grid_pot(tmp_l, tmp_i, tmp_j, tmp_k)
             enddo
          enddo
       enddo
    enddo

    !! read hard grid values
    Call ReadGrid(tmp_grid_pot, GridRadii, NGrid0, &
         NumGrid, XGridNum, YGridNum, ZGridNum, &
         hardGridUnit, Form)
    num_grid_points = NumGrid * XGridNum * YGridNum * ZGridNum
    call chmalloc('openmm_dock.src', 'Read_Potential_Grid', 'HardGridPot', num_grid_points, crl = hardGridPot)
    hardGridPot(:) = 0.0
    do tmp_l = 1, NumGrid
       do tmp_i = 1, XGridNum
          do tmp_j = 1, YGridNum
             do tmp_k = 1, ZGridNum
                tmp_idx = (((tmp_l - 1) * XGridNum + tmp_i - 1) * YGridNum + tmp_j - 1) * ZGridNum + tmp_k
                hardGridPot(tmp_idx) = tmp_grid_pot(tmp_l, tmp_i, tmp_j, tmp_k)
             enddo
          enddo
       enddo
    enddo
  END SUBROUTINE Read_Grid_Potentials

  subroutine xyz2vec3(vec, x, y, z)
    real*8, intent(out) :: vec(3)
    real(chm_real), intent(in) :: x, y, z
    vec(1) = x
    vec(2) = y
    vec(3) = z
  end subroutine xyz2vec3
#endif /* KEY_FFTDOCK == 1 && KEY_OPENMM ==1  */
end module openmm_dock
