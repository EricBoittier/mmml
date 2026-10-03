! ML/MM ABI
module mlps_abi
  implicit none

#if KEY_MLMM==1

  interface

#if KEY_MLPTORCH==1
    !=============================================================
    ! DUMMY TEST C++ / LibTorch interface ! type_libt_dummy.cxx
    !=============================================================
    subroutine charmm_add_dummy(dummy_in_use, dummy_use_gpu, dummy_in_nml, &
                               dummy_in_mlidx, dummy_in_mlZid, dummy_in_mlmaskid, &
                               dummy_in_natoms, dummy_out_setup_err) &
      bind(C, name="charmm_dummy_internal_setup")
      use, intrinsic :: iso_c_binding, only: c_int
      implicit none
    
      integer(c_int), value, intent(in) :: dummy_in_use
      integer(c_int), value, intent(in) :: dummy_use_gpu
    
      integer(c_int), value, intent(in) :: dummy_in_nml
      integer(c_int), value, intent(in) :: dummy_in_natoms
    
      integer(c_int), dimension(*), intent(in) :: dummy_in_mlidx
      integer(c_int), dimension(*), intent(in) :: dummy_in_mlZid
      integer(c_int), dimension(*), intent(in) :: dummy_in_mlmaskid
    
      integer(c_int), intent(out) :: dummy_out_setup_err
    end subroutine charmm_add_dummy
    
    
    subroutine charmm_dummy_force(E_dummy_c, ml_x_c, ml_y_c, ml_z_c, &
                                  ml_dx_c, ml_dy_c, ml_dz_c) &
      bind(C, name="charmm_dummy_internal_force")
      use, intrinsic :: iso_c_binding, only: c_double
      implicit none
    
      real(c_double), intent(out) :: E_dummy_c
    
      real(c_double), dimension(*), intent(in) :: ml_x_c
      real(c_double), dimension(*), intent(in) :: ml_y_c
      real(c_double), dimension(*), intent(in) :: ml_z_c
    
      real(c_double), dimension(*), intent(out) :: ml_dx_c
      real(c_double), dimension(*), intent(out) :: ml_dy_c
      real(c_double), dimension(*), intent(out) :: ml_dz_c
    end subroutine charmm_dummy_force

    !=============================================================
    ! TANI C++ / LibTorch interface ! type_libt_tani.cxx
    !=============================================================
    subroutine charmm_add_tani(tani_in_use, tani_use_gpu, tani_pt_name, tani_pt_nml, &
                               tani_in_mlidx, tani_in_mlZid, tani_in_mlmaskid, tani_in_natoms, tani_out_setup_err) &
      bind(C, name="charmm_tani_internal_setup")
      use, intrinsic :: iso_c_binding, only: c_char, c_int
      implicit none

      integer(c_int), value, intent(in) :: tani_in_use
      integer(c_int), value, intent(in) :: tani_use_gpu
      character(kind=c_char), dimension(*), intent(in) :: tani_pt_name

      integer(c_int), value, intent(in) :: tani_pt_nml
      integer(c_int), value, intent(in) :: tani_in_natoms

      integer(c_int), dimension(*), intent(in) :: tani_in_mlidx
      integer(c_int), dimension(*), intent(in) :: tani_in_mlZid
      integer(c_int), dimension(*), intent(in) :: tani_in_mlmaskid

      integer(c_int), intent(out) :: tani_out_setup_err
    end subroutine charmm_add_tani


    subroutine charmm_tani_force(E_tani_c, ml_x_c, ml_y_c, ml_z_c, &
                                 ml_dx_c, ml_dy_c, ml_dz_c) &
      bind(C, name="charmm_tani_internal_force")
      use, intrinsic :: iso_c_binding, only: c_double
      implicit none

      real(c_double), intent(out) :: E_tani_c

      real(c_double), dimension(*), intent(in)  :: ml_x_c
      real(c_double), dimension(*), intent(in)  :: ml_y_c
      real(c_double), dimension(*), intent(in)  :: ml_z_c

      real(c_double), dimension(*), intent(out) :: ml_dx_c
      real(c_double), dimension(*), intent(out) :: ml_dy_c
      real(c_double), dimension(*), intent(out) :: ml_dz_c
    end subroutine charmm_tani_force


    !=============================================================
    ! DPMM C++ / LibTorch interface ! type_libt_dpmm.cxx
    !=============================================================
    subroutine charmm_add_dpmm(dpmm_in_use, dpmm_use_gpu, dpmm_pt_name, &
                               dpmm_pt_nml, dpmm_pt_nmm_max, &
                               dpmm_in_mlidx, dpmm_in_mlSid, &
                               dpmm_in_mlmaskid, dpmm_in_natoms, dpmm_out_setup_err) &
      bind(C, name="charmm_dpmm_internal_setup")
      use, intrinsic :: iso_c_binding, only: c_char, c_int
      implicit none

      integer(c_int), value, intent(in) :: dpmm_in_use
      integer(c_int), value, intent(in) :: dpmm_use_gpu
      character(kind=c_char), dimension(*), intent(in) :: dpmm_pt_name

      integer(c_int), value, intent(in) :: dpmm_pt_nml
      integer(c_int), value, intent(in) :: dpmm_pt_nmm_max
      integer(c_int), value, intent(in) :: dpmm_in_natoms

      integer(c_int), dimension(*), intent(in) :: dpmm_in_mlidx
      integer(c_int), dimension(*), intent(in) :: dpmm_in_mlSid
      integer(c_int), dimension(*), intent(in) :: dpmm_in_mlmaskid

      integer(c_int), intent(out) :: dpmm_out_setup_err
    end subroutine charmm_add_dpmm


    subroutine charmm_dpmm_force(E_dpmm_c, mm_count_c, &
                                 ml_x_c, ml_y_c, ml_z_c, &
                                 mm_cg_c, mm_x_c, mm_y_c, mm_z_c, &
                                 ml_dx_c, ml_dy_c, ml_dz_c, &
                                 mm_dx_c, mm_dy_c, mm_dz_c) &
      bind(C, name="charmm_dpmm_internal_force")
      use, intrinsic :: iso_c_binding, only: c_double, c_int
      implicit none

      real(c_double), intent(out) :: E_dpmm_c

      ! Actual number of MM neighbors in this force call.
      ! C++ pads internally up to dpmm_pt_nmm_max cached at setup.
      integer(c_int), value, intent(in) :: mm_count_c

      real(c_double), dimension(*), intent(in)  :: ml_x_c
      real(c_double), dimension(*), intent(in)  :: ml_y_c
      real(c_double), dimension(*), intent(in)  :: ml_z_c

      real(c_double), dimension(*), intent(in)  :: mm_cg_c
      real(c_double), dimension(*), intent(in)  :: mm_x_c
      real(c_double), dimension(*), intent(in)  :: mm_y_c
      real(c_double), dimension(*), intent(in)  :: mm_z_c

      real(c_double), dimension(*), intent(out) :: ml_dx_c
      real(c_double), dimension(*), intent(out) :: ml_dy_c
      real(c_double), dimension(*), intent(out) :: ml_dz_c

      real(c_double), dimension(*), intent(out) :: mm_dx_c
      real(c_double), dimension(*), intent(out) :: mm_dy_c
      real(c_double), dimension(*), intent(out) :: mm_dz_c
    end subroutine charmm_dpmm_force
#endif


    !=============================================================
    ! PYTHON SOCKET interface, embedded runner version ! type_pyth.cxx
    !
    ! Generic ML-only backend through type_pyth_embedded.cxx.
    !
    ! No SPEC file.
    ! No external RUNF .py file.
    !
    ! Setup contract:
    !
    !   model_type        : "uma", "mace", "tani", "dummy"
    !   model_name        : local model/checkpoint path from MODL
    !   pyth_charge       : total charge, used by UMA/MACE if needed
    !   pyth_multiplicity : spin multiplicity, used by UMA/MACE if needed
    !
    !=============================================================
    subroutine charmm_add_pyth(pyth_in_use, pyth_use_gpu, &
                               model_type, model_name, &
                               pyth_charge, pyth_multiplicity, &
                               pyth_pt_nml, &
                               pyth_in_mlidx, pyth_in_mlZid, &
                               pyth_in_mlmaskid, pyth_in_natoms, pyth_out_setup_err) &
      bind(C, name="charmm_pyth_internal_setup")
      use, intrinsic :: iso_c_binding, only: c_char, c_int
      implicit none

      integer(c_int), value, intent(in) :: pyth_in_use
      integer(c_int), value, intent(in) :: pyth_use_gpu

      character(kind=c_char), dimension(*), intent(in) :: model_type
      character(kind=c_char), dimension(*), intent(in) :: model_name

      integer(c_int), value, intent(in) :: pyth_charge
      integer(c_int), value, intent(in) :: pyth_multiplicity

      integer(c_int), value, intent(in) :: pyth_pt_nml
      integer(c_int), value, intent(in) :: pyth_in_natoms

      integer(c_int), dimension(*), intent(in) :: pyth_in_mlidx
      integer(c_int), dimension(*), intent(in) :: pyth_in_mlZid
      integer(c_int), dimension(*), intent(in) :: pyth_in_mlmaskid

      integer(c_int), intent(out) :: pyth_out_setup_err
    end subroutine charmm_add_pyth


    subroutine charmm_pyth_force(E_pyth_c, ml_x_c, ml_y_c, ml_z_c, &
                                 ml_dx_c, ml_dy_c, ml_dz_c) &
      bind(C, name="charmm_pyth_internal_force")
      use, intrinsic :: iso_c_binding, only: c_double
      implicit none

      real(c_double), intent(out) :: E_pyth_c

      real(c_double), dimension(*), intent(in)  :: ml_x_c
      real(c_double), dimension(*), intent(in)  :: ml_y_c
      real(c_double), dimension(*), intent(in)  :: ml_z_c

      real(c_double), dimension(*), intent(out) :: ml_dx_c
      real(c_double), dimension(*), intent(out) :: ml_dy_c
      real(c_double), dimension(*), intent(out) :: ml_dz_c
    end subroutine charmm_pyth_force


    !=============================================================
    ! PYTHON SOCKET custom external-runner interface ! type_pyth_custom.cxx
    ! NEED external RUNF .py file.
    ! OPTIONAL SPEC file.
    !=============================================================
    subroutine charmm_add_pyth_custom(pyth_in_use, pyth_use_gpu, &
                                      spec_name, runf_name, pyth_pt_nml, &
                                      pyth_in_mlidx, pyth_in_mlZid, &
                                      pyth_in_mlmaskid, pyth_in_natoms, pythcstm_out_setup_err) &
      bind(C, name="charmm_pyth_internal_setup_custom")
      use, intrinsic :: iso_c_binding, only: c_char, c_int
      implicit none

      integer(c_int), value, intent(in) :: pyth_in_use
      integer(c_int), value, intent(in) :: pyth_use_gpu

      character(kind=c_char), dimension(*), intent(in) :: spec_name
      character(kind=c_char), dimension(*), intent(in) :: runf_name

      integer(c_int), value, intent(in) :: pyth_pt_nml
      integer(c_int), value, intent(in) :: pyth_in_natoms

      integer(c_int), dimension(*), intent(in) :: pyth_in_mlidx
      integer(c_int), dimension(*), intent(in) :: pyth_in_mlZid
      integer(c_int), dimension(*), intent(in) :: pyth_in_mlmaskid

      integer(c_int), intent(out) :: pythcstm_out_setup_err
    end subroutine charmm_add_pyth_custom


    subroutine charmm_pyth_force_custom(E_pyth_c, ml_x_c, ml_y_c, ml_z_c, &
                                        ml_dx_c, ml_dy_c, ml_dz_c) &
      bind(C, name="charmm_pyth_internal_force_custom")
      use, intrinsic :: iso_c_binding, only: c_double
      implicit none

      real(c_double), intent(out) :: E_pyth_c

      real(c_double), dimension(*), intent(in)  :: ml_x_c
      real(c_double), dimension(*), intent(in)  :: ml_y_c
      real(c_double), dimension(*), intent(in)  :: ml_z_c

      real(c_double), dimension(*), intent(out) :: ml_dx_c
      real(c_double), dimension(*), intent(out) :: ml_dy_c
      real(c_double), dimension(*), intent(out) :: ml_dz_c
    end subroutine charmm_pyth_force_custom

  end interface

#endif

end module mlps_abi