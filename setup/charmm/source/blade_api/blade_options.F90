module blade_options_module
  implicit none

  interface
     subroutine blade_init_run(system) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr), value :: system
     end subroutine blade_init_run

     subroutine blade_dest_run(system) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr), value :: system
     end subroutine blade_dest_run

     subroutine blade_add_run_flags(system, &
       gamma, &
       beta_ewald, &
       r_cut, &
       r_switch, &
       vdw_fswitch, &
       elec_pme, &
       grid_space, &
       gridx, &
       gridy, &
       gridz, &
       order_ewald, &
       shake_tol) bind(c)

       use, intrinsic :: iso_c_binding, only: c_ptr, c_long, c_int, c_double
       implicit none

       type(c_ptr), value :: system

       real(c_double), value :: gamma, beta_ewald, r_cut, r_switch, grid_space
       integer(c_int), value :: vdw_fswitch, elec_pme
       integer(c_int), value :: gridx, gridy, gridz, order_ewald
       real(c_double), value :: shake_tol
     end subroutine blade_add_run_flags

     subroutine blade_add_run_dynopts(system, &
       step, &
       step0, &
       nsteps, &
       dt, &
       t, &
       freq_npt, &
       vol_fluc, &
       pressure) bind(c)

       use, intrinsic :: iso_c_binding, only: c_ptr, c_long, c_int, c_double
       implicit none

       type(c_ptr), value :: system

       integer(c_int), value :: step, step0
       integer(c_int), value :: nsteps
       real(c_double), value :: dt, t
       integer(c_int), value :: freq_npt
       real(c_double), value :: vol_fluc, pressure
     end subroutine blade_add_run_dynopts
  end interface
end module blade_options_module
