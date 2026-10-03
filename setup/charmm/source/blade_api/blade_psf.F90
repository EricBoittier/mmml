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

     subroutine blade_add_noe(system, i, j, rmin, kmin, rmax, kmax, rpeak, rswitch, nswitch, c0x, c0y, c0z, is_pnoe) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double, c_bool

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: i, j
       real(c_double), value :: rmin, kmin, rmax, kmax, rpeak, rswitch, nswitch, c0x, c0y, c0z
       logical(c_bool), value :: is_pnoe
     end subroutine blade_add_noe

     subroutine blade_add_harmonic(system, i, k, x0, y0, z0, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: system
       integer(c_int), value :: i
       real(c_double), value :: k, x0, y0, z0, n
     end subroutine blade_add_harmonic

     subroutine blade_add_boRest(system, i, j, kr, r0, lambdaBlock) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: system
       integer(c_int), value :: i, j, lambdaBlock
       real(c_double), value :: kr, r0
     end subroutine blade_add_boRest

     subroutine blade_add_anRest(system, i, j, k, kt, t0, lambdaBlock) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: system
       integer(c_int), value :: i, j, k, lambdaBlock
       real(c_double), value :: kt, t0
     end subroutine blade_add_anRest

     subroutine blade_add_diRest(system, i, j, k, l, kphi, nphi, phi0, width, lambdaBlock) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: system
       integer(c_int), value :: i, j, k, l, nphi, lambdaBlock
       real(c_double), value :: kphi, phi0, width
     end subroutine blade_add_diRest

     ! eemlp eeresd resd abi
     subroutine blade_add_resd(system, i1, i2, j1, j2, ci, cj, rdist, kdist) bind(c)
      use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
      implicit none
      type(c_ptr), value :: system
      integer(c_int), value :: i1, i2, j1, j2
      real(c_double), value :: kdist, rdist, ci, cj
     end subroutine blade_add_resd

#if KEY_MLMM==1 && KEY_MLPTORCH==1
     ! eemlp torchani in blade
     subroutine blade_add_tani(system,tani_in_use,tani_pt_name,tani_pt_nml, &
      tani_in_mlidx,tani_in_mlZid,tani_in_mlmaskid,tani_in_natoms) bind(C, name="blade_tani_internal_setup")
       use, intrinsic :: iso_c_binding, only: c_ptr, c_char, c_int
       implicit none
       type(c_ptr), value :: system
       integer(c_int), value, intent(in) :: tani_in_use
       character(kind=c_char), dimension(*), intent(in) :: tani_pt_name
       integer(c_int), value, intent(in) :: tani_pt_nml
       integer(c_int), value, intent(in) :: tani_in_natoms
       integer(c_int), dimension(*), intent(in) :: tani_in_mlidx
       integer(c_int), dimension(*), intent(in) :: tani_in_mlZid
       integer(c_int), dimension(*), intent(in) :: tani_in_mlmaskid
     end subroutine blade_add_tani
#endif     
  end interface
end module blade_psf_module
