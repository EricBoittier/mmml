module mlps_pol_ene
#if KEY_MLMM==1  /*MLPS POL*/
  implicit none
 contains
  subroutine calc_mlps_pol_force(E_mlps_pol, x, y, z, dx_pol_mlps, dy_pol_mlps, dz_pol_mlps)
    use chm_kinds
    use dimens_fcm
    use mlps_ini
#if KEY_PARALLEL==1
    use parallel
    use mpi_f08, only: MPI_LOGICAL, MPI_INTEGER
#endif
    use psf, only: cg
    use consta, only: CCELEC_charmm

    implicit none

    real(chm_real), intent(out) :: E_mlps_pol
    real(chm_real), dimension(*), intent(in)    :: x, y, z
    real(chm_real), dimension(*), intent(inout) :: dx_pol_mlps, dy_pol_mlps, dz_pol_mlps

    ! other local vars
    logical :: within_cutoff
    real(chm_real) :: cut2, r2ij

    integer :: i, j, mm_count
    integer :: ia, ib
    real(chm_real) :: q_mm
    real(chm_real) :: ex, ey, ez, field2, edotr
    real(chm_real) :: dxij, dyij, dzij, r2, rinv, rinv3, rinv5
    real(chm_real) :: fac, cc, coef, alpha_i, cfac, grad_prefac
    real(chm_real) :: gxij, gyij, gzij



#if KEY_PARALLEL==1
    integer :: ierr
    if (mynod == 0) then ! eemlp
#endif

    if (mlps_qerror) goto 1010

    if (real(dielec_eps, kind=chm_real) <= 0.0_chm_real) then
        error_message = ' MLMM> dielec_eps must be positive for POL1.'
        mlps_qerror = .true.
        goto 1010
    end if

    cc = real(CCELEC_charmm, kind=chm_real)
    coef = 0.5_chm_real / real(dielec_eps, kind=chm_real)
    cut2 = real(mlps_cutoff, kind=chm_real)*real(mlps_cutoff, kind=chm_real)
    
    E_mlps_pol = 0.0_chm_real
    !------------------------------------------------------------
    ! Compact ML coordinates for POL1 calculation
    !------------------------------------------------------------
    if (with_pol_op_1 == 1) then
        mlx_mlps = 0.0_chm_real
        mly_mlps = 0.0_chm_real
        mlz_mlps = 0.0_chm_real
        do i = 1, mlnm2
            mlx_mlps(i) = real(x(ml_idx(i)), kind=chm_real)
            mly_mlps(i) = real(y(ml_idx(i)), kind=chm_real)
            mlz_mlps(i) = real(z(ml_idx(i)), kind=chm_real)
        end do
    end if

    !------------------------------------------------------------
    ! Compact unique MM neighbor list for POL1
    !
    ! This builds the UNION of all MM atoms within mlps_cutoff
    ! of at least one ML/QM atom.
    !
    ! Each MM atom is added at most once because the outer loop is
    ! over atom index i, and we exit after the first matching ML atom.
    !------------------------------------------------------------
    if ((with_pol_op_1 == 1) .and. ml_num_mm > 0) then

        mm_count = 0

        mmidx_mlps = 0
        mmcg_mlps  = 0.0_chm_real
        mmx_mlps   = 0.0_chm_real
        mmy_mlps   = 0.0_chm_real
        mmz_mlps   = 0.0_chm_real

        do i = 1, natom_cache

            ! Skip ML/QM atoms. Keep only MM/environment atoms.
            if (ml_imask(i) == 1) cycle

            within_cutoff = .false.

            do j = 1, mlnm2

                dxij = real(x(i), kind=chm_real) - mlx_mlps(j)
                dyij = real(y(i), kind=chm_real) - mly_mlps(j)
                dzij = real(z(i), kind=chm_real) - mlz_mlps(j)

                r2ij = dxij*dxij + dyij*dyij + dzij*dzij

                if (r2ij < cut2) then
                    within_cutoff = .true.
                    exit
                end if

            end do

            if (within_cutoff) then

                if (mm_count >= ml_num_mm) then
                    error_message = ' MLMM> Number of unique MM neighbors within cutoff exceeds MXMM.'
                    mlps_qerror = .true.
                    goto 1010
                end if

                mm_count = mm_count + 1

                mmidx_mlps(mm_count) = i
                mmcg_mlps(mm_count)  = real(cg(i), kind=chm_real)
                mmx_mlps(mm_count)   = real(x(i),  kind=chm_real)
                mmy_mlps(mm_count)   = real(y(i),  kind=chm_real)
                mmz_mlps(mm_count)   = real(z(i),  kind=chm_real)

            end if

        end do

        !------------------------------------------------------------
        ! Compute POL1 energy and gradient
        !------------------------------------------------------------
        do i = 1, mlnm2
        
            ia = ml_idx(i)
            alpha_i = ml_atom_pol(i)
        
            if (alpha_i == 0.0_chm_real) cycle
        
            ex = 0.0_chm_real
            ey = 0.0_chm_real
            ez = 0.0_chm_real
        
            ! First pass: total MM electric field at QM atom ia
            do j = 1, mm_count
            
                q_mm = mmcg_mlps(j)
            
                dxij = x(ia) - mmx_mlps(j)
                dyij = y(ia) - mmy_mlps(j)
                dzij = z(ia) - mmz_mlps(j)
            
                r2 = dxij*dxij + dyij*dyij + dzij*dzij
            
                if (r2 > 1.0e-20_chm_real) then
                    rinv  = 1.0_chm_real / sqrt(r2)
                    rinv3 = rinv * rinv * rinv
                
                ! Unswitched Field
                    fac = cc * q_mm * rinv3
                    !
                    ex = ex + fac * dxij
                    ey = ey + fac * dyij
                    ez = ez + fac * dzij


                end if
            
            end do
        
            field2 = ex*ex + ey*ey + ez*ez
        
            ! Energy: E_D = alpha |E|^2 / (2 eps)
            E_mlps_pol = E_mlps_pol - coef * alpha_i * field2
        
            ! Second pass: gradient dE_D/dR
            do j = 1, mm_count
            
                ib   = mmidx_mlps(j)
                q_mm = mmcg_mlps(j)
            
                dxij = x(ia) - mmx_mlps(j)
                dyij = y(ia) - mmy_mlps(j)
                dzij = z(ia) - mmz_mlps(j)
            
                r2 = dxij*dxij + dyij*dyij + dzij*dzij
            
                if (r2 > 1.0e-20_chm_real ) then
                    rinv  = 1.0_chm_real / sqrt(r2)
                    rinv3 = rinv * rinv * rinv
                    rinv5 = rinv3 * rinv * rinv
                
                    cfac = cc * q_mm
                
                    edotr = ex*dxij + ey*dyij + ez*dzij
                
                    grad_prefac = 2.0_chm_real * coef * alpha_i * cfac
                    ! Equivalent to: grad_prefac = alpha_i * cfac / eps
                    ! negative of distortion energy
                    ! 
                    gxij = - grad_prefac * &
                           (ex*rinv3 - 3.0_chm_real*dxij*edotr*rinv5)
                
                    gyij = - grad_prefac * &
                           (ey*rinv3 - 3.0_chm_real*dyij*edotr*rinv5)
                
                    gzij = - grad_prefac * &
                           (ez*rinv3 - 3.0_chm_real*dzij*edotr*rinv5)
                
                    ! Gradient on QM atom ia
                    dx_pol_mlps(ia) = dx_pol_mlps(ia) + gxij
                    dy_pol_mlps(ia) = dy_pol_mlps(ia) + gyij
                    dz_pol_mlps(ia) = dz_pol_mlps(ia) + gzij
                
                    ! Compensating gradient on MM atom ib
                    dx_pol_mlps(ib) = dx_pol_mlps(ib) - gxij
                    dy_pol_mlps(ib) = dy_pol_mlps(ib) - gyij
                    dz_pol_mlps(ib) = dz_pol_mlps(ib) - gzij
                end if
            
            end do
        
        end do

    end if

1010    continue
#if KEY_PARALLEL==1
    end if
    call MPI_Bcast(mlps_qerror, 1, MPI_LOGICAL, 0,comm_charmm, ierr)
    if (mlps_qerror) then
        if (mynod == 0) then
            call wrndie(-5, '', trim(error_message))
        else
            call wrndie(-5, '', ' MLMM> Fatal error during MLMM energy/gradient calculation on rank 0. See previous messages for details.')
        end if
    end if
#else
    if (mlps_qerror) call wrndie(-5, '', trim(error_message))
#endif
  end subroutine calc_mlps_pol_force

#endif /*MLPS POL*/
end module mlps_pol_ene

