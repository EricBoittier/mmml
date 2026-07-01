module enb_core_gpu_mod
#if KEY_DOMDEC_GPU==1 /*domdec_gpu*/
  use, intrinsic :: iso_c_binding
  implicit none

  private

  ! Public CUDA subroutines
  public enb_tilex_gpu, set_nonbond_param_gpu, set_box_size_gpu, read_nonbond_energy_gpu
  public read_direct_virial_gpu
  
  interface

     subroutine enb_tilex_gpu(h_whichlist, h_q_calc_energy, h_q_calc_virial, boxx, boxy, boxz) &
          bind(C,name="enb_tilex_gpu")
       import
       integer(c_int), intent(in) :: h_whichlist
       logical(c_int), intent(in) :: h_q_calc_energy, h_q_calc_virial
       real(c_double), intent(in) :: boxx, boxy, boxz
     end subroutine enb_tilex_gpu

     subroutine set_nonbond_param_gpu(h_kappa, h_roff, h_ron, h_e14fac, &
          h_vdwmodel, h_elecmodel, h_qeterm_vdw, h_qeterm_elec) &
          bind(C,name="set_nonbond_param_gpu")
       import
       real(c_double), intent(in) :: h_kappa
       real(c_double), intent(in) :: h_roff, h_ron, h_e14fac
       integer(c_int), intent(in) :: h_vdwmodel, h_elecmodel
       logical(c_int), intent(in) :: h_qeterm_vdw, h_qeterm_elec
     end subroutine set_nonbond_param_gpu

     subroutine set_box_size_gpu(h_boxx, h_boxy, h_boxz) &
          bind(C,name="set_box_size_gpu")
       import
       real(c_double), intent(in) :: h_boxx, h_boxy, h_boxz
     end subroutine set_box_size_gpu

     subroutine read_nonbond_energy_gpu(h_vdwpot, h_coulpot, h_exclpot) &
          bind(C,name="read_nonbond_energy_gpu")
       import
       real(c_double), intent(inout) :: h_vdwpot, h_coulpot, h_exclpot
     end subroutine read_nonbond_energy_gpu
     
     subroutine read_direct_virial_gpu(h_vir, h_virtensor) &
          bind(C,name="read_direct_virial_gpu")
       import
       real(c_double), intent(inout) :: h_vir, h_virtensor(*)
     end subroutine read_direct_virial_gpu

  end interface

contains
#endif /* (domdec_gpu)*/
end module enb_core_gpu_mod


