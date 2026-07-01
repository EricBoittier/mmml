module bonded_gpu_mod
#if KEY_DOMDEC_GPU==1
  use, intrinsic :: iso_c_binding
  use domdec_bonded_types,only:bondlist_t, anglelist_t, dihelist_t
  implicit none
  public

  interface
     subroutine setup_bonded_coef_gpu(nbond, cbb, cbc, ntheta, ctb, ctc, nureyb, ctub, ctuc, &
          nphi, cpd, cpc, cpsin, cpcos, nimphi, cid, cic, cisin, cicos) &
          bind(C,name="setup_bonded_coef_gpu")
       import
       integer(c_int), intent(in) :: nbond, ntheta, nureyb, nphi, nimphi
       integer(c_int), intent(in) :: cpd(*), cid(*)
       real(c_double), intent(in) :: cbb(*), cbc(*), ctb(*), ctc(*), ctub(*), ctuc(*), &
            cpc(*), cpsin(*), cpcos(*), cic(*), cisin(*), cicos(*)
     end subroutine setup_bonded_coef_gpu

     subroutine setup_bonded_list_gpu(nbondtbl, bondlist, nangletbl, anglelist, nureybtbl, ureyblist, &
          ndihetbl, dihelist, nimdihetbl, imdihelist) &
          bind(C,name="setup_bonded_list_gpu")
       import
       integer(c_int), intent(in) :: nbondtbl, nangletbl, nureybtbl, ndihetbl, nimdihetbl
       type(bondlist_t), intent(in) :: bondlist(*), ureyblist(*)
       type(anglelist_t), intent(in) :: anglelist(*)
       type(dihelist_t), intent(in) :: dihelist(*), imdihelist(*)
     end subroutine setup_bonded_list_gpu

     subroutine calc_bonded_gpu(boxx, boxy, boxz, q_calc_energy, q_calc_virial, q_calc_bond, &
          q_calc_angle, q_calc_ureyb, q_calc_dihe, q_calc_imdihe) bind(C,name="calc_bonded_gpu")
       import
       real(c_double), intent(in) :: boxx, boxy, boxz
       logical(c_int), intent(in) :: q_calc_energy, q_calc_virial, q_calc_bond, &
            q_calc_angle, q_calc_ureyb, q_calc_dihe, q_calc_imdihe
     end subroutine calc_bonded_gpu

     subroutine read_bonded_energy_gpu(energy_bond, energy_angle, energy_ureyb, &
          energy_dihe, energy_imdihe) bind(C,name="read_bonded_energy_gpu")
       import
       real(c_double), intent(inout) :: energy_bond, energy_angle, energy_ureyb, &
            energy_dihe, energy_imdihe
     end subroutine read_bonded_energy_gpu
     
  end interface
#endif
end module bonded_gpu_mod
