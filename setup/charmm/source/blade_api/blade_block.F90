! void blade_init_msld(System *system,int nblocks);
! void blade_dest_msld(System *system);
! void blade_add_msld_atomassignment(System *system,int atomIdx,int blockIdx);
! void blade_add_msld_initialconditions(System *system,int blockIdx,int siteIdx,double theta0,double thetaVelocity,double thetaMass,double fixBias,double blockCharge);
! void blade_add_msld_termscaling(System *system,bool scaleBond,bool scaleUrey,bool scaleAngle,bool scaleDihe,bool scaleImpr,bool scaleCmap);
! void blade_add_msld_flags(System *system,bool useSoftCore,bool useSoftCore14,int msldEwaldType,double kRestraint,double kChargeRestraint,double softBondRadius,double softBondExponent,double softNotBondExponent);
! void blade_add_msld_bias(System *system,int i,int j,int type,double l0,double k,int n);
! void blade_add_msld_thetabiascoll(System *system,int sites,int i,double k,double n);
! void blade_add_msld_thetabiasinde(System *system,int sites,int i,double k);
! void blade_add_msld_softbond(System *system,int i,int j);
! void blade_add_msld_atomrestraint(System *system);
! void blade_add_msld_atomrestraint_element(System *system,int i);

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

     subroutine blade_add_msld_flags(system, gamma, fnex, useSoftCore, useSoftCore14, msldEwaldType, kRestraint, kChargeRestraint, softBondRadius, softBondExponent, softNotBondExponent, fix) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double

       implicit none

       type(c_ptr), value :: system
       real(c_double), value :: gamma, fnex
       integer(c_int), value :: useSoftCore, useSoftCore14 ! Cast to logical to integer to bool
       integer(c_int), value :: msldEwaldType
       real(c_double), value :: kRestraint, kChargeRestraint, softBondRadius, softBondExponent, softNotBondExponent
       integer(c_int), value :: fix
     end subroutine blade_add_msld_flags

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
  end interface
end module blade_block_module
