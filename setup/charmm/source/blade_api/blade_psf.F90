module blade_psf_module
  implicit none

  interface
     subroutine blade_init_structure(system) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr), value :: system
     end subroutine blade_init_structure

     subroutine blade_add_atom(system, atom_idx, seg_name, res_idx, res_name, &
          atom_name, atom_type_name, charge, mass) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char, c_double

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: atom_idx
       character(kind=c_char, len=1), dimension(*) :: &
            seg_name, res_idx, res_name, atom_name, atom_type_name
       real(c_double), value :: charge, mass
     end subroutine blade_add_atom
     
     subroutine blade_add_bond(system, i, j) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: i, j
     end subroutine blade_add_bond
     
     subroutine blade_add_angle(system, i, j, k) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: i, j, k
     end subroutine blade_add_angle

     subroutine blade_add_dihe(system, i, j, k, l) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: i, j, k, l
     end subroutine blade_add_dihe

     subroutine blade_add_impr(system, i, j, k, l) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: i, j, k, l
     end subroutine blade_add_impr

     subroutine blade_add_cmap(system, i1, j1, k1, l1, &
          i2, j2, k2, l2) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: &
            i1, j1, k1, l1, &
            i2, j2, k2, l2
     end subroutine blade_add_cmap

     subroutine blade_add_virt2(system, v, h1, h2, dist, scale) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: v, h1, h2
       real(c_double), value :: dist, scale
     end subroutine blade_add_virt2

     subroutine blade_add_virt3(system, v, h1, h2, h3, dist, theta, phi) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: v, h1, h2, h3
       real(c_double), value :: dist, theta, phi
     end subroutine blade_add_virt3

     subroutine blade_add_shake(system, shake_h_bond) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: system
       integer(c_int), value :: shake_h_bond
     end subroutine blade_add_shake

     subroutine blade_add_noe(system, i, j, rmin, kmin, rmax, kmax, rpeak, rswitch, nswitch) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: i, j
       real(c_double), value :: rmin, kmin, rmax, kmax, rpeak, rswitch, nswitch
     end subroutine blade_add_noe

     subroutine blade_add_harmonic(system, i, k, x0, y0, z0, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: system
       integer(c_int), value :: i
       real(c_double), value :: k, x0, y0, z0, n
     end subroutine blade_add_harmonic
  end interface
end module blade_psf_module
