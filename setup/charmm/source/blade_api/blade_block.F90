! void blade_init_msld(System *system,int nblocks);
! void blade_dest_msld(System *system);
! void blade_add_msld_atomassignment(System *system,int atomIdx,int blockIdx);
! void blade_add_msld_initialconditions(System *system,int blockIdx,int siteIdx,double theta0,double thetaVelocity,double thetaMass,double fixBias,double blockCharge);
! void blade_add_msld_termscaling(System *system,bool scaleBond,bool scaleUrey,bool scaleAngle,bool scaleDihe,bool scaleImpr,bool scaleCmap);
! void blade_add_msld_flags(System *system,bool useSoftCore,bool useSoftCore14,int msldEwaldType,double kRestraint,double softBondRadius,double softBondExponent,double softNotBondExponent);
! void blade_add_msld_charges(System *system,double kChargeRestraint1,double kChargeRestraint2,double kChargeRestraint3,double q0ChargeRestraint3,double wChargeRestraint3);
! void blade_add_msld_bias(System *system,int i,int j,int type,double l0,double k,int n);
! void blade_add_msld_thetacollbias(System *system,int sites,int i,double k,double n);
! void blade_add_msld_thetaindebias(System *system,int sites,int i,double k);
! void blade_set_msld_thetaedgebias(System *system,double k,double N,double alpha,double phi);
! void blade_set_msld_piecewise_constraint(System *system,int do_pwi,double w,double k);
! void blade_add_msld_softbond(System *system,int i,int j);
! void blade_add_msld_atomrestraint(System *system);
! void blade_add_msld_atomrestraint_element(System *system,int i);
! int blade_sync_msld_bias(System *system,double *bias,int count);

module blade_block_module
  implicit none

  interface
     subroutine blade_init_msld(system, nblocks) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: nblocks
     end subroutine blade_init_msld

     subroutine blade_dest_msld(system) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr

       implicit none

       type(c_ptr), value :: system
     end subroutine blade_dest_msld

     subroutine blade_add_msld_atomassignment(system, atomIdx, blockIdx) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: atomIdx, blockIdx
     end subroutine blade_add_msld_atomassignment

     subroutine blade_add_msld_initialconditions(system, blockIdx, siteIdx, theta0, thetaVelocity, thetaMass, fixBias, blockCharge) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: blockIdx, siteIdx
       real(c_double), value :: theta0, thetaVelocity, thetaMass, fixBias, blockCharge
     end subroutine blade_add_msld_initialconditions

     subroutine blade_add_msld_termscaling(system, scaleBond, scaleUrey, scaleAngle, scaleDihe, scaleImpr, scaleCmap) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: scaleBond, scaleUrey, scaleAngle, scaleDihe, scaleImpr, scaleCmap
     end subroutine blade_add_msld_termscaling

     subroutine blade_add_msld_flags(system, gamma, fnex, temperature, &
          useSoftCore, useSoftCore14, msldEwaldType, kRestraint, &
          softBondRadius, softBondExponent, softNotBondExponent, fix) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double

       implicit none

       type(c_ptr), value :: system
       real(c_double), value :: gamma, fnex, temperature
       integer(c_int), value :: useSoftCore, useSoftCore14 ! Cast to logical to integer to bool
       integer(c_int), value :: msldEwaldType
       real(c_double), value :: kRestraint, softBondRadius, softBondExponent, softNotBondExponent
       integer(c_int), value :: fix
     end subroutine blade_add_msld_flags

     subroutine blade_add_msld_charges(system, kChargeRestraint1, kChargeRestraint2, kChargeRestraint3, q0ChargeRestraint3, wChargeRestraint3) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_double

       implicit none

       type(c_ptr), value :: system
       real(c_double), value :: kChargeRestraint1, kChargeRestraint2, kChargeRestraint3, q0ChargeRestraint3, wChargeRestraint3
     end subroutine blade_add_msld_charges

     subroutine blade_add_msld_bias(system, i, j, type, l0, k, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: i, j, type, n
       real(c_double), value :: l0, k
     end subroutine blade_add_msld_bias

     subroutine blade_add_msld_thetacollbias(system, sites, i, k, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: sites, i
       real(c_double), value :: k, n
     end subroutine blade_add_msld_thetacollbias

     subroutine blade_add_msld_thetaindebias(system, sites, i, k) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: sites, i
       real(c_double), value :: k
     end subroutine blade_add_msld_thetaindebias

     subroutine blade_set_msld_thetaedgebias(system, k, N, alpha, phi) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_double

       implicit none

       type(c_ptr), value :: system
       real(c_double), value :: k, N, alpha, phi
     end subroutine blade_set_msld_thetaedgebias

     subroutine blade_set_msld_piecewise_constraint(system, do_pwi, w, k) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: do_pwi
       real(c_double), value :: w, k
     end subroutine blade_set_msld_piecewise_constraint

     subroutine blade_add_msld_block_friction(system, blockIdx, friction) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: blockIdx
       real(c_double), value :: friction
     end subroutine blade_add_msld_block_friction

     subroutine blade_add_msld_softbond(system, i, j) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: i, j
     end subroutine blade_add_msld_softbond

     subroutine blade_add_msld_atomrestraint(system) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr

       implicit none

       type(c_ptr), value :: system
     end subroutine blade_add_msld_atomrestraint

     subroutine blade_add_msld_atomrestraint_element(system, i) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: i
     end subroutine blade_add_msld_atomrestraint_element

     subroutine blade_set_msld_block_fixed(system, blockIdx, fixed) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       type(c_ptr), value :: system
       integer(c_int), value :: blockIdx, fixed
     end subroutine blade_set_msld_block_fixed

     integer(c_int) function blade_sync_msld_bias(system, bias, count) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_double, c_int

       implicit none

       type(c_ptr), value :: system
       real(c_double) :: bias(*)
       integer(c_int), value :: count
     end function blade_sync_msld_bias
  end interface
end module blade_block_module
