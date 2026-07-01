module blade_param_module
  implicit none

  interface
     subroutine blade_init_parameters(system) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr

       implicit none

       type(c_ptr), value :: system
     end subroutine blade_init_parameters

     subroutine blade_dest_parameters(system) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr

       implicit none

       type(c_ptr), value :: system
     end subroutine blade_dest_parameters

     subroutine blade_add_parameter_atoms(system, name, mass) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_char, c_double

       implicit none

       type(c_ptr), value :: system
       character(len=1, kind=c_char) :: name(*)
       real(c_double), value :: mass
     end subroutine blade_add_parameter_atoms

     subroutine blade_add_parameter_bonds(system, t1, t2, kb, b0) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_char, c_double

       implicit none

       type(c_ptr), value :: system
       character(len=1, kind=c_char) :: t1(*), t2(*)
       real(c_double), value :: kb, b0
     end subroutine blade_add_parameter_bonds

     subroutine blade_add_parameter_angles(system, t1, t2, t3, &
          kangle, angle0, kureyb, ureyb0) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_char, c_double

       implicit none

       type(c_ptr), value :: system
       character(len=1, kind=c_char) :: t1(*), t2(*), t3(*)
       real(c_double), value :: kangle, angle0, kureyb, ureyb0
     end subroutine blade_add_parameter_angles

     subroutine blade_add_parameter_dihes(system, t1, t2, t3, t4, &
          kdih, ndih, dih0) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_char, c_double, c_int

       implicit none

       type(c_ptr), value :: system
       character(len=1, kind=c_char), dimension(*) :: t1, t2, t3, t4
       real(c_double), value :: kdih, dih0
       integer(c_int), value :: ndih
     end subroutine blade_add_parameter_dihes

     subroutine blade_add_parameter_imprs(system, t1, t2, t3, t4, &
          kimp, nimp, imp0) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_char, c_double, c_int

       implicit none

       type(c_ptr), value :: system
       character(len=1, kind=c_char), dimension(*) :: t1, t2, t3, t4
       real(c_double), value :: kimp, imp0
       integer(c_int), value :: nimp
     end subroutine blade_add_parameter_imprs

     subroutine blade_add_parameter_cmaps(system, &
          t1, t2, t3, t4, &
          t5, t6, t7, t8, &
          ngrid) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_char, c_int

       implicit none

       type(c_ptr), value :: system
       character(len=1, kind=c_char), dimension(*) :: &
            t1, t2, t3, t4, &
            t5, t6, t7, t8
       integer(c_int), value :: ngrid
     end subroutine blade_add_parameter_cmaps

     subroutine blade_add_parameter_cmaps_fill(system, &
          t1, t2, t3, t4, &
          t5, t6, t7, t8, &
          i, j, kcmapij) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_char, c_double, c_int

       implicit none

       type(c_ptr), value :: system
       character(len=1, kind=c_char), dimension(*) :: &
            t1, t2, t3, t4, &
            t5, t6, t7, t8
       integer(c_int), value :: i, j
       real(c_double), value :: kcmapij
     end subroutine blade_add_parameter_cmaps_fill

     subroutine blade_add_parameter_nbonds(system, &
          t1, eps, sig, eps14, sig14, e14fac, combine) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_char, c_double, c_int

       implicit none

       type(c_ptr), value :: system
       character(len=1, kind=c_char) :: t1(*)
       real(c_double), value :: eps, sig, eps14, sig14, e14fac
       integer(c_int), value :: combine
     end subroutine blade_add_parameter_nbonds

     subroutine blade_add_parameter_nbfixs(system, &
          t1, t2, eps, sig, eps14, sig14) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_char, c_double

       implicit none

       type(c_ptr), value :: system
       character(len=1, kind=c_char) :: t1(*), t2(*)
       real(c_double), value :: eps, sig, eps14, sig14
     end subroutine blade_add_parameter_nbfixs
  end interface
end module blade_param_module
