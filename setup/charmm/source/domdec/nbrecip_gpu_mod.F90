module nbrecip_gpu_mod
#if KEY_DOMDEC_GPU==1
  use, intrinsic :: iso_c_binding
  implicit none
  public

  interface
     subroutine calc_recip_gpu(h_boxx, h_boxy, h_boxz, &
          h_q_calc_energy, h_q_calc_virial) bind(C,name="calc_recip_gpu")
       import
       real(c_double), intent(in) :: h_boxx, h_boxy, h_boxz
       logical(c_int), intent(in) :: h_q_calc_energy, h_q_calc_virial
     end subroutine calc_recip_gpu

     subroutine calc_recip_block_gpu(h_boxx, h_boxy, h_boxz, &
          h_q_calc_energy, h_q_calc_virial) bind(C,name="calc_recip_block_gpu")
       import
       real(c_double), intent(in) :: h_boxx, h_boxy, h_boxz
       logical(c_int), intent(in) :: h_q_calc_energy, h_q_calc_virial
     end subroutine calc_recip_block_gpu
     
     subroutine read_recip_energy_gpu(h_ewksum, h_ewself) &
          bind(C,name="read_recip_energy_gpu")
       import
       real(c_double), intent(inout) :: h_ewksum, h_ewself
     end subroutine read_recip_energy_gpu

     subroutine read_recip_virial_gpu(h_ewvirial) &
          bind(C,name="read_recip_virial_gpu")
       import
       real(c_double), intent(inout) :: h_ewvirial(9)
     end subroutine read_recip_virial_gpu

  end interface

#endif
end module nbrecip_gpu_mod
