! MLPS ene
module mlps_ene
#if KEY_MLMM==1
  implicit none
 contains
  subroutine calc_mlps_force(E_mlps, x, y, z, dx_mlps, dy_mlps, dz_mlps)
    use chm_kinds
    use dimens_fcm
    use iso_c_binding, only: c_double, c_int
    use mlps_ini
    use mlps_abi, only: charmm_pyth_force, charmm_pyth_force_custom
#if KEY_MLPTORCH==1
    use mlps_abi, only: charmm_dummy_force, charmm_tani_force, charmm_dpmm_force
#endif
#if KEY_PARALLEL==1
    use parallel
    use mpi_f08, only: MPI_LOGICAL, MPI_INTEGER
#endif
    use psf, only: cg
    use consta, only: CCELEC_charmm

    implicit none

    real(chm_real), intent(out) :: E_mlps
    real(chm_real), dimension(*), intent(in)    :: x, y, z
    real(chm_real), dimension(*), intent(inout) :: dx_mlps, dy_mlps, dz_mlps

    ! Coulomb gradient subtraction variables
    integer :: ia, ib
    real(chm_real) :: qml, qmm
    real(chm_real) :: dxij, dyij, dzij, r2, rinv, rinv3, fac

    integer :: i, j, mm_count
    integer(c_int) :: mm_count_c

#if KEY_PARALLEL==1
    integer :: ierr
    if (mynod == 0) then ! eemlp
#endif

    if (mlps_qerror) goto 1010

    E_mlps = 0.0_chm_real

    !------------------------------------------------------------
    ! Compact ML coordinates for TANI, DPMM, PYTH
    !------------------------------------------------------------
    if (with_libtorch == 1 .or. with_pyth == 1 ) then
        mlx_mlps_c = 0.0_c_double
        mly_mlps_c = 0.0_c_double
        mlz_mlps_c = 0.0_c_double
        do i = 1, mlnm2
            mlx_mlps_c(i) = real(x(ml_idx(i)), kind=c_double)
            mly_mlps_c(i) = real(y(ml_idx(i)), kind=c_double)
            mlz_mlps_c(i) = real(z(ml_idx(i)), kind=c_double)
        end do
    end if

    !------------------------------------------------------------
    ! Compact MM neighbor list (for DPMM)
    !------------------------------------------------------------
    if (with_dpmm == 1 .and. ml_num_mm > 0) then
        mm_count = 0
        mmcg_mlps_c = 0.0_c_double
        mmx_mlps_c = 0.0_c_double
        mmy_mlps_c = 0.0_c_double
        mmz_mlps_c = 0.0_c_double

        do i = 1, natom_cache
            if (ml_imask(i) /= 1) then
                do j = 1, mlnm2
                    if ((x(i)-x(ml_idx(j)))**2 + (y(i)-y(ml_idx(j)))**2 + (z(i)-z(ml_idx(j)))**2 < mlps_cutoff**2) then
                        if (mm_count >= ml_num_mm) then
                            error_message = ' MLMM> Number of MM neighbors within cutoff exceeds MXMM.'
                            mlps_qerror = .true.
                            goto 1010
                        end if

                        mm_count = mm_count + 1
                        mmidx_mlps_c(mm_count) = i
                        mmcg_mlps_c(mm_count)  = real(cg(i), kind=c_double)
                        mmx_mlps_c(mm_count)   = real(x(i),  kind=c_double)
                        mmy_mlps_c(mm_count)   = real(y(i),  kind=c_double)
                        mmz_mlps_c(mm_count)   = real(z(i),  kind=c_double)
                        exit
                    end if
                end do
            end if
        end do
    else
        mm_count = 0
    end if

    mm_count_c = int(mm_count, kind=c_int)

    !------------------------------------------------------------
    ! Call model backends
    !------------------------------------------------------------
    if (with_libtorch == 1 .and. with_dummy == 1) then
#if KEY_MLPTORCH==1
        call charmm_dummy_force(E_mlps_c, mlx_mlps_c, mly_mlps_c, mlz_mlps_c, &
                                mldx_mlps_c, mldy_mlps_c, mldz_mlps_c)
        E_mlps = mlps_ene_scale * real(E_mlps_c, kind=chm_real)

        do i = 1, mlnm2
            dx_mlps(ml_idx(i)) = dx_mlps(ml_idx(i)) + (mlps_ene_scale * real(mldx_mlps_c(i), kind=chm_real))
            dy_mlps(ml_idx(i)) = dy_mlps(ml_idx(i)) + (mlps_ene_scale * real(mldy_mlps_c(i), kind=chm_real))
            dz_mlps(ml_idx(i)) = dz_mlps(ml_idx(i)) + (mlps_ene_scale * real(mldz_mlps_c(i), kind=chm_real))
        end do
#endif      
    else if (with_libtorch == 1 .and. with_tani == 1) then
#if KEY_MLPTORCH==1
        call charmm_tani_force(E_mlps_c, mlx_mlps_c, mly_mlps_c, mlz_mlps_c, &
                               mldx_mlps_c, mldy_mlps_c, mldz_mlps_c)

        E_mlps = mlps_ene_scale * real(E_mlps_c, kind=chm_real)

        do i = 1, mlnm2
            dx_mlps(ml_idx(i)) = dx_mlps(ml_idx(i)) + (mlps_ene_scale * real(mldx_mlps_c(i), kind=chm_real))
            dy_mlps(ml_idx(i)) = dy_mlps(ml_idx(i)) + (mlps_ene_scale * real(mldy_mlps_c(i), kind=chm_real))
            dz_mlps(ml_idx(i)) = dz_mlps(ml_idx(i)) + (mlps_ene_scale * real(mldz_mlps_c(i), kind=chm_real))
        end do
#endif
    else if (with_libtorch == 1 .and. with_dpmm == 1) then
#if KEY_MLPTORCH==1
        call charmm_dpmm_force(E_mlps_c, mm_count_c, &
                               mlx_mlps_c, mly_mlps_c, mlz_mlps_c, &
                               mmcg_mlps_c, mmx_mlps_c, mmy_mlps_c, mmz_mlps_c, &
                               mldx_mlps_c, mldy_mlps_c, mldz_mlps_c, &
                               mmdx_mlps_c, mmdy_mlps_c, mmdz_mlps_c)

        E_mlps = mlps_ene_scale * real(E_mlps_c, kind=chm_real)

        do i = 1, mlnm2
            dx_mlps(ml_idx(i)) = dx_mlps(ml_idx(i)) + (mlps_ene_scale * real(mldx_mlps_c(i), kind=chm_real))
            dy_mlps(ml_idx(i)) = dy_mlps(ml_idx(i)) + (mlps_ene_scale * real(mldy_mlps_c(i), kind=chm_real))
            dz_mlps(ml_idx(i)) = dz_mlps(ml_idx(i)) + (mlps_ene_scale * real(mldz_mlps_c(i), kind=chm_real))
        end do

        do i = 1, mm_count
            dx_mlps(mmidx_mlps_c(i)) = dx_mlps(mmidx_mlps_c(i)) + (mlps_ene_scale * real(mmdx_mlps_c(i), kind=chm_real))
            dy_mlps(mmidx_mlps_c(i)) = dy_mlps(mmidx_mlps_c(i)) + (mlps_ene_scale * real(mmdy_mlps_c(i), kind=chm_real))
            dz_mlps(mmidx_mlps_c(i)) = dz_mlps(mmidx_mlps_c(i)) + (mlps_ene_scale * real(mmdz_mlps_c(i), kind=chm_real))
        end do
#endif
    else if (with_pyth == 1) then
        if (with_dummy == 1 .or. with_uma == 1 .or. with_mace == 1 .or. with_tani == 1) then
            call charmm_pyth_force(E_mlps_c, mlx_mlps_c, mly_mlps_c, mlz_mlps_c, &
                               mldx_mlps_c, mldy_mlps_c, mldz_mlps_c)
        elseif (with_dummy == 0 .and. with_uma == 0 .and. with_mace == 0 .and. with_tani == 0 .and. with_dpmm == 0) then
            call charmm_pyth_force_custom(E_mlps_c, mlx_mlps_c, mly_mlps_c, mlz_mlps_c, &
                                      mldx_mlps_c, mldy_mlps_c, mldz_mlps_c)
        end if                        

        E_mlps = mlps_ene_scale * real(E_mlps_c, kind=chm_real)

        do i = 1, mlnm2
            dx_mlps(ml_idx(i)) = dx_mlps(ml_idx(i)) + (mlps_ene_scale * real(mldx_mlps_c(i), kind=chm_real))
            dy_mlps(ml_idx(i)) = dy_mlps(ml_idx(i)) + (mlps_ene_scale * real(mldy_mlps_c(i), kind=chm_real))
            dz_mlps(ml_idx(i)) = dz_mlps(ml_idx(i)) + (mlps_ene_scale * real(mldz_mlps_c(i), kind=chm_real))
        end do
    end if

    if (with_dpmm == 1) then
        !------------------------------------------------------------
        ! Remove the classical CHARMM Coulomb gradient for ML-MM pairs
        ! that are already represented by the DPMM model.
        !
        ! E_coul = CCELEC_charmm * q_ml * q_mm / r
        !
        ! Let:
        !   Rij = R_ml - R_mm
        !   fac = CCELEC_charmm * q_ml * q_mm / r^3
        !
        ! Then the classical Coulomb gradients are:
        !   dE/dR_ml = -fac * Rij
        !   dE/dR_mm = +fac * Rij
        !
        ! Since CHARMM dx/dy/dz store gradients, subtracting the
        ! classical ML-MM Coulomb contribution means adding:
        !   -dE/dR_ml = +fac * Rij
        !   -dE/dR_mm = -fac * Rij
        !------------------------------------------------------------
        do i = 1, mlnm2
            ia  = ml_idx(i)
            qml = real(cg(ia), kind=chm_real)

            do j = 1, mm_count
                ib  = mmidx_mlps_c(j)
                qmm = real(cg(ib), kind=chm_real)

                dxij = x(ia) - x(ib)
                dyij = y(ia) - y(ib)
                dzij = z(ia) - z(ib)

                r2 = dxij*dxij + dyij*dyij + dzij*dzij

                if (r2 > 1.0e-20_chm_real) then
                    rinv  = 1.0_chm_real / sqrt(r2)
                    rinv3 = rinv * rinv * rinv
                    fac   = real(CCELEC_charmm, kind=chm_real) * qml * qmm * rinv3

                    dx_mlps(ia) = dx_mlps(ia) + fac * dxij
                    dy_mlps(ia) = dy_mlps(ia) + fac * dyij
                    dz_mlps(ia) = dz_mlps(ia) + fac * dzij

                    dx_mlps(ib) = dx_mlps(ib) - fac * dxij
                    dy_mlps(ib) = dy_mlps(ib) - fac * dyij
                    dz_mlps(ib) = dz_mlps(ib) - fac * dzij
                end if
            end do
        end do

    end if

1010    continue
#if KEY_PARALLEL==1
    end if
    call MPI_Bcast(mlps_qerror, 1, MPI_LOGICAL, 0,comm_charmm, ierr)
    if (mlps_qerror) then
        !if (mynod == 0) then
        !    call wrndie(-5, '', trim(error_message))
        !else
            call wrndie(-5, '', ' MLMM> Fatal error during MLMM energy/gradient calculation on rank 0. See previous messages for details.')
        !end if
    end if
#else
    if (mlps_qerror) call wrndie(-5, '', trim(error_message))
#endif
    return
  end subroutine calc_mlps_force
#endif
end module mlps_ene

