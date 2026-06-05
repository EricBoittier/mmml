module colfft

  use chm_kinds
  use nblist_types,only:intarray_t
  use colfft_types,only:q_grid_t
  use colfft_func, only: grad_sumrc_ps
  implicit none
  private

  integer kbot,ktop,jbot,jtop

  integer, allocatable, dimension(:) :: fr1, fr2, fr3

  ! Double precision versions
  real(chm_real), allocatable, dimension(:,:) :: theta1,theta2,theta3
  real(chm_real), allocatable, dimension(:,:) :: dtheta1,dtheta2,dtheta3
  real(chm_real), allocatable, dimension(:) :: m1_tbl_dp, m2_tbl_dp, m3_tbl_dp
  real(chm_real), allocatable, dimension(:) :: prefac1_dp, prefac2_dp, prefac3_dp

  ! Single precision versions
  real(chm_real4), allocatable, dimension(:,:) :: theta1_sp,theta2_sp,theta3_sp
  real(chm_real4), allocatable, dimension(:,:) :: dtheta1_sp,dtheta2_sp,dtheta3_sp
  real(chm_real4), allocatable, dimension(:) :: m1_tbl_sp,m2_tbl_sp,m3_tbl_sp
  real(chm_real4), allocatable, dimension(:) :: prefac1_sp, prefac2_sp, prefac3_sp

  integer, save :: forder_save=0, nfft1_save=0, nfft2_save=0, nfft3_save=0
  logical, save :: q_use_single_save = .false.

  real(chm_real4), allocatable, dimension(:) :: xf, yf, zf, chargef
  real(chm_real4) :: ccelecf

  ! Data structure for charge grid
  ! Local charge grid for each thread
  type(q_grid_t), target, allocatable, dimension(:) :: q_grid_loc

  ! Global grid
  type(q_grid_t), target :: q_grid_glo(0:0)

  ! Pointer to charge grid
  type(q_grid_t), pointer, dimension(:) :: q_grid_p

  ! Atom list for each thread
  type(intarray_t), allocatable, dimension(:) :: atomlist_thread

  ! List of thread IDs for each atom
  integer, allocatable, dimension(:) :: thread_id_list

  ! Number of atoms for each thread
  type(intarray_t), allocatable, dimension(:) :: natom_thread

  ! Mapping grid -> thread id
  integer, allocatable, dimension(:) :: grid2tx, grid2ty, grid2tz
  integer grid2tx_lo, grid2ty_lo, grid2tz_lo
  integer grid2tx_hi, grid2ty_hi, grid2tz_hi
  ! Saved parameter values
  integer :: grid2t_nthread = 0, grid2t_forder = 0

  ! Number of threads in each coordinate direction
  integer ntx, nty, ntz

  ! Public subroutines
  public do_colfft, colfft_init, colfft_uninit, filter_grid_atom

  ! Public variables
  !public colfft_allocated

!!$  interface fill_bspline1
!!$     module procedure fill_bspline1_pd
!!$     module procedure fill_bspline1_ps
!!$  end interface

  !===========================================================================
contains
  !===========================================================================

  ! *
  ! * Calculate column fft
  ! *
  subroutine do_colfft(n_grid_atom, grid_atom, natom, &
       nfftdim1c, nfftdim2c, nfftdim3c, &
       eksum, eqcor, eutil, &
       qeksum, qeqcor, qeutil, &
#if KEY_LJPME==1
       do_ljpme, &
#endif
       x, y, z, dx, dy, dz, cg, cgtot, virial, kappa)
    use pmeutil,only:nfft1,nfft2,nfft3,forder,get_fftdims,&
         load_bsp_mod,mxyslabs,mxzslabs,&
         bsp_mod1,bsp_mod2,bsp_mod3
    use memory
#if KEY_CHEQ==1
    use cheq, only:qcg, dch, sumdch, ddch
#endif
    use dimens_fcm
    use exfunc
    use number
    use energym,only:eprop, volume, qdyncall
    use reawri,only:qcnstp
    use inbnd
    use consta
    use stream
    use image,only:xnsymm
    use domdec_common,only:q_use_single
#if KEY_DOMDEC==1
    use domdec_common,only:natoml, atoml, q_gpu, q_test
#endif
#if KEY_DOMDEC_GPU==1
    use domdec_util_gpu_mod,only:range_start, range_stop
#endif
    implicit none
    ! Input / Output
    integer, intent(in) :: n_grid_atom, grid_atom(:)
    integer,intent(in) :: natom
    integer,intent(in) :: nfftdim1c,nfftdim2c,nfftdim3c
    real(chm_real),intent(in)::kappa
    real(kind=chm_real),intent(out) ::  eksum,eqcor,eutil,virial(6)
    logical,intent(in) :: qeksum,qeqcor,qeutil
    real(kind=chm_real),dimension(1:natom),intent(in) ::  x,y,z,cg
    real(kind=chm_real),dimension(1:natom),intent(inout) :: dx,dy,dz
    real(kind=chm_real),intent(in) ::  cgtot
    !
    !-----------------------------------------------------------------------
    !     This routine calculates non bonded interaction energies and
    !     forces via the Particle Mesh Ewald Summation
    !     original code by Tom Darden, implemented into CHARMM
    !     by Scott Feller and Bernie Brooks, NIH, Jan-1996
    !     Parallel 3D fft routines from Michael Crowley, Pittsburgh
    !     Supercomputer Center
    !     Column fft version mike crowley 2008
    !-----------------------------------------------------------------------

#if KEY_CHEQ==1
    real(kind=chm_real) ::   hij
#endif
    real(kind=chm_real) ::   vircor(6),eslf,cfact
    real(kind=chm_real) ::   rdum(3),htmp(9),hwrk(9)
    integer siztheta, sizdtheta, siz_q, siz_f
    integer nfftdim1, nfftdim2, nfftdim3
    integer i, ii, ipt, itrans
    real(kind=chm_real) ::  tx,ty,tz,rs,s2,erfcx,drfc,ene,qd(1), &
         dxf,dyf,dzf
#if KEY_PMEPLSMA==1
    real(kind=chm_real) ::  factor,ee_plasma
#endif
    integer ierr
    integer atfrst,atlast,ia
#if KEY_DOMDEC==1
    integer n
#endif
    logical q_calc_energy_virial
#if KEY_LJPME==1
    logical, intent(in) :: do_ljpme
    logical order_changed
#endif
#if KEY_DOMDEC_GPU==1
    real(chm_real), dimension(3,3) :: recip_dp
#endif

    nfftdim1=nfftdim1c
    nfftdim2=nfftdim2c
    nfftdim3=nfftdim3c
#if KEY_PARASCAL==1
#error "Illegal parallel compile options"
#endif
    if(xnsymm > 1 ) then
       write(outu,*)" XNSYMM = ",xnsymm
       call wrndie(-5,"PME_column","no symmetry operations for column FFT yet")
    endif
  
    virial=zero
    eqcor=0.0
    eutil=0.0

    !-------------------------------------------------------------------
    ! INPUT
    !      NFFT1,NFFT2,NFFT3 are the (integer) dimensions of the charge grid array
    !      NATOM is number of atoms
    !      FORDER is the order of B-spline interpolation
    !      x,y,z:   atomic coords
    !      CG  atomic charges
    !      XTLINV=recip: array of reciprocal unit cell vectors
    !      VOLUME: the volume of the unit cell
    !      KAPPA=ewald_coeff:   ewald convergence parameter
    !      NFFT1,NFFT2,NFFT3: the dimensions of the charge grid array
    ! OUTPUT
    !      siz_Q=3d charge grid array
    !      sizfftab is permanent 3d fft table storage
    !      sizffwrk is temporary 3d fft work storage
    !      siztheta is size of arrays theta1-3 dtheta1-3
    !      EKSUM= eer:  ewald reciprocal or k-space  energy
    !      dx,dy,dz: forces incremented by k-space sum
    !      EWVIRIAL=virial:  virial due to k-space sum (valid for atomic scaling;
    !                rigid molecule virial needs a correction term not
    !                computed here
    !
    !   All heap pointers are the integer names of the real(kind=chm_real)
    !   variables that will be filled
    !
    !   Get memory for scratch arrays, free them after summation.
    !
    !====================================================
    call get_fftdims(nfftdim1,nfftdim2,nfftdim3,i,ii)

    siztheta  = natom*xnsymm*forder
    sizdtheta = (natom+1)*forder
    siz_f = natom*xnsymm

    siz_q = max(2*nfftdim1*nfftdim2*mxyslabs, &
         2*nfftdim1*nfftdim3*mxzslabs)

    !====================================================
    if(qeksum) then

       if (nfft1 /= nfft1_save .or. nfft2 /= nfft2_save .or. nfft3 /= nfft3_save .or. &
            forder /= forder_save .or. q_use_single() .neqv. q_use_single_save) then
          call load_bsp_mod()
          nfft1_save = nfft1
          nfft2_save = nfft2
          nfft3_save = nfft3
          forder_save = forder
          q_use_single_save = q_use_single()
          ! Make a local copy of (bsp_mod1, bsp_mod2, bsp_mod3) to
          ! (prefac1_dp, prefac2_dp, prefac3_dp) or (prefac1_sp, prefac2_sp, prefac3_sp),
          ! depending on precision model
          if (.not.q_use_single()) then
             ! Double precision
             prefac1_dp(1:nfft1) = bsp_mod1(1:nfft1)
             prefac2_dp(1:nfft2) = bsp_mod2(1:nfft2)
             prefac3_dp(1:nfft3) = bsp_mod3(1:nfft3)
          else
             ! Single precision
             prefac1_sp(1:nfft1) = bsp_mod1(1:nfft1)
             prefac2_sp(1:nfft2) = bsp_mod2(1:nfft2)
             prefac3_sp(1:nfft3) = bsp_mod3(1:nfft3)
          endif
       endif
          
#if KEY_DOMDEC_GPU==1
       if (q_gpu) then
          q_calc_energy_virial = ((.not.qdyncall).or.qcnstp) .or. q_test
       else
#endif
          q_calc_energy_virial = .true.
#if KEY_DOMDEC_GPU==1
       endif
#endif

       call spatial_do_pmesh_kspace(n_grid_atom, grid_atom, natom, x, y, z, cg, &
            kappa, eprop(volume), eksum, dx, dy, dz, virial, &
#if KEY_LJPME==1
            do_ljpme, &
#endif
            q_calc_energy_virial)

    endif

    return
  end subroutine do_colfft

  !------------------------------------------------------------------
  !           SPATIAL  DO_PMESH_KSPACE
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  subroutine spatial_do_pmesh_kspace(n_grid_atom, grid_atom, natom, x, y, z, charge, &
       kappa, volume, &
       eer, dx, dy, dz, virial, &
#if KEY_LJPME==1
       do_ljpme, &
#endif
       q_calc_energy_virial)
    use pmeutil,only:nfft1, nfft2, nfft3, forder
    use colfft_util,only:backward_rc_fft, forward_rc_fft
    use new_timer,only: T_fft, T_scsum, T_bspl, t_grads, t_fillg, &
         timer_start, timer_stop, timer_stpstrt
    use exfunc
    use stream
    use parallel
    use dimens_fcm
    use image
    use number,only:zero
    use consta, only: ccelec
    use domdec_common,only:q_use_single, nthread
#if KEY_DOMDEC==1
    use domdec_common,only:q_gpu, q_test
#endif
#if KEY_DOMDEC_GPU==1
    use domdec_util_gpu_mod,only:range_start, range_stop
#endif
    use colfft_func,only:grad_sumrc_ps, grad_sumrc_pd, scalar_sumrc_ps, scalar_sumrc_pd, &
         reduce_charge_grid_ps, reduce_charge_grid_pd, &
#if KEY_LJPME==1
         scalar_sumrc_lj_ps, scalar_sumrc_lj_pd, &
#endif
#if KEY_DOMDEC==1
         test_fill_bspline_ps, test_fill_bspline_pd, &
         test_spread_charge_grid_ps, test_spread_charge_grid_pd, &
         test1_scalar_sumrc_ps, test1_scalar_sumrc_pd, &
         test2_scalar_sumrc_ps, test2_scalar_sumrc_pd, &
         test_replicate_grid_borders_ps, test_replicate_grid_borders_pd, &
         test1_grad_sumrc, test2_grad_sumrc_ps, test2_grad_sumrc_pd, &
#endif
         spread_charge_grid_ps, spread_charge_grid_pd, fill_bspline_ps, fill_bspline_pd
    ! INPUT
    !       natom:  number of atoms
    !       x,y,z   atomic coords
    !       charge  atomic charges
    !      integer,parameter :: real8 = selected_real_kind(15,300)
    integer, intent(in) :: n_grid_atom, grid_atom(:)
    integer, intent(in) :: natom
    real(chm_real), intent(in), dimension(natom) :: x,y,z
    real(chm_real), intent(in), dimension(natom) :: charge
    real(chm_real), intent(in) :: kappa,volume
    logical, intent(in) :: q_calc_energy_virial
    ! OUTPUT
    !       eer:  ewald reciprocal or k-space  energy
    !       dx,dy,dz forces incremented by k-space sum
    real(chm_real), intent(inout), dimension(natom) :: dx,dy,dz
    real(chm_real), intent(inout), dimension(6) :: virial
    real(chm_real), intent(out) :: eer
#if KEY_LJPME==1
    logical, intent(in) :: do_ljpme
#endif
    ! Parameters
    real(chm_real4), parameter :: zero_sp = 0.0_chm_real4
    ! Variables
    integer i, j, k
    logical lorthog
    real(chm_real4), dimension(3,3) :: recip_sp
    real(chm_real), dimension(3,3) :: recip_dp

    call timer_start(T_FILLG)

    ! Calculate reciprocal matrix
    if (q_use_single()) then
       call calc_recip_ps(recip_sp)
       lorthog = &
            (recip_sp(1,2) == zero_sp .and. recip_sp(1,3) == zero_sp).and. &
            (recip_sp(2,1) == zero_sp .and. recip_sp(2,3) == zero_sp).and. &
            (recip_sp(3,1) == zero_sp .and. recip_sp(3,2) == zero_sp )
    else
       call calc_recip_pd(recip_dp)
       lorthog = &
            (recip_dp(1,2) == zero .and. recip_dp(1,3) == zero).and. &
            (recip_dp(2,1) == zero .and. recip_dp(2,3) == zero).and. &
            (recip_dp(3,1) == zero .and. recip_dp(3,2) == zero )
    endif

    if (q_use_single()) then
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_start('xf,yf,zf')
#endif
!$omp parallel do private(i, j)
       do i=1,n_grid_atom
          j = grid_atom(i)
          xf(j) = x(j)
          yf(j) = y(j)
          zf(j) = z(j)
          chargef(j) = charge(j)
       enddo
!$omp end parallel do
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_stop()
#endif
       ccelecf = ccelec
    endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('fill_bspline')
#endif
    if (q_use_single()) then
       call fill_bspline_ps(n_grid_atom, grid_atom, &
            forder, nfft1, nfft2, nfft3, &
            xf, yf, zf, chargef, recip_sp, lorthog, &
            fr1, fr2, fr3, theta1_sp, theta2_sp, theta3_sp, dtheta1_sp, &
            dtheta2_sp, dtheta3_sp, &
            grid2tx_lo, grid2ty_lo, grid2tz_lo, &
            grid2tx_hi, grid2ty_hi, grid2tz_hi, &
            grid2tx, grid2ty, grid2tz, &
            nthread, natom_thread, thread_id_list, atomlist_thread)
    else
       call fill_bspline_pd(n_grid_atom, grid_atom, &
            forder, nfft1, nfft2, nfft3, &
            x, y, z, charge, recip_dp, lorthog, &
            fr1, fr2, fr3, theta1, theta2, theta3, &
            dtheta1, dtheta2, dtheta3, &
            grid2tx_lo, grid2ty_lo, grid2tz_lo, &
            grid2tx_hi, grid2ty_hi, grid2tz_hi, &
            grid2tx, grid2ty, grid2tz, &
            nthread, natom_thread, thread_id_list, atomlist_thread)
    endif
#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

#if KEY_DOMDEC==1
    if (q_test) then
       if (q_use_single()) then
          call test_fill_bspline_ps(n_grid_atom, grid_atom, &
               natom, forder, xf, yf, zf, chargef, recip_sp, nfft2, nfft3, &
               fr1, fr2, fr3, &
               theta1_sp, theta2_sp, theta3_sp, dtheta1_sp, dtheta2_sp, dtheta3_sp, &
               grid2tx_lo, grid2ty_lo, grid2tz_lo, &
               grid2tx_hi, grid2ty_hi, grid2tz_hi, &
               grid2tx, grid2ty, grid2tz, &
               nthread, natom_thread, thread_id_list, atomlist_thread, q_grid_p)
       else
          call test_fill_bspline_pd(n_grid_atom, grid_atom, &
               natom, forder, x, y, z, charge, recip_dp, nfft2, nfft3, &
               fr1, fr2, fr3, &
               theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, &
               grid2tx_lo, grid2ty_lo, grid2tz_lo, &
               grid2tx_hi, grid2ty_hi, grid2tz_hi, &
               grid2tx, grid2ty, grid2tz, &
               nthread, natom_thread, thread_id_list, atomlist_thread, q_grid_p)
       endif
    endif
#endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('spread_charge_grid')
#endif
    if (q_use_single()) then
       call spread_charge_grid_ps(n_grid_atom, grid_atom, xf, yf, zf, chargef, &
            theta1_sp, theta2_sp, theta3_sp, &
            recip_sp, lorthog, &
            fr1, fr2, fr3,  &
            forder, nfft1, nfft2, nfft3,  &
            nthread, natom_thread, atomlist_thread, q_grid_p)
    else
       call spread_charge_grid_pd(n_grid_atom, grid_atom, x, y, z, charge, &
            theta1, theta2, theta3, &
            recip_dp, lorthog, &
            fr1, fr2, fr3,  &
            forder, nfft1, nfft2, nfft3,  &
            nthread, natom_thread, atomlist_thread, q_grid_p)
    endif
#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('reduce_charge_grid')
#endif

    if (q_use_single()) then
!$omp parallel
       call reduce_charge_grid_ps(nthread, ntx, nty, ntz, forder, q_grid_p, q_grid_glo(0))
!$omp end parallel
    else
!$omp parallel
       call reduce_charge_grid_pd(nthread, ntx, nty, ntz, forder, q_grid_p, q_grid_glo(0))
!$omp end parallel
    endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

#if KEY_DOMDEC==1
    if (q_test) then
       if (q_use_single()) then
          call test_spread_charge_grid_ps(n_grid_atom, grid_atom, &
               fr1, fr2, fr3, theta1_sp, theta2_sp, theta3_sp, &
               q_grid_glo(0), chargef, forder, nfft2, nfft3)
       else
          call test_spread_charge_grid_pd(n_grid_atom, grid_atom, &
               fr1, fr2, fr3, theta1, theta2, theta3, &
               q_grid_glo(0), charge, forder, nfft2, nfft3)
       endif
    endif
#endif

    call timer_stpstrt(T_fillg,T_FFT)

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('backward FFT')
#endif
    if (q_use_single()) then
       call backward_rc_fft(q_grid_glo(0)%array_sp, q_grid_glo(0)%tot_size)
    else
       call backward_rc_fft(q_grid_glo(0)%array_dp, q_grid_glo(0)%tot_size)
    endif
#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

    call timer_stpstrt(T_FFT,T_SCSUM)

    !           -------------SCALAR SUM------------------

#if KEY_DOMDEC==1
    if (q_test) then
       if (q_use_single()) then
          call test1_scalar_sumrc_ps(q_grid_glo(0)%tot_size, q_grid_glo(0)%array_sp)
       else
          call test1_scalar_sumrc_pd(q_grid_glo(0)%tot_size, q_grid_glo(0)%array_dp)
       endif
    endif
#endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('Scalar sum')
#endif
#if KEY_LJPME==1
    if (do_ljpme) then
       if (q_use_single()) then
          call scalar_sumrc_lj_ps(q_grid_glo(0)%tot_size, q_grid_glo(0)%array_sp, &
               kappa, volume, recip_sp, &
               prefac1_sp, prefac2_sp, prefac3_sp, &
               nfft1, nfft2, nfft3, eer, virial, &
               q_calc_energy_virial)
       else
          call scalar_sumrc_lj_pd(q_grid_glo(0)%tot_size, q_grid_glo(0)%array_dp, &
               kappa, volume, recip_dp, &
               prefac1_dp, prefac2_dp, prefac3_dp, &
               nfft1, nfft2, nfft3, eer, virial, &
               q_calc_energy_virial)
       endif
    else
#endif
       if (q_use_single()) then
          call scalar_sumrc_ps(q_grid_glo(0)%tot_size, q_grid_glo(0)%array_sp, &
               kappa, volume, recip_sp, &
               prefac1_sp, prefac2_sp, prefac3_sp, &
               m1_tbl_sp, m2_tbl_sp, m3_tbl_sp, &
               nfft1, nfft2, nfft3, eer, virial, lorthog, &
               q_calc_energy_virial)
       else
          call scalar_sumrc_pd(q_grid_glo(0)%tot_size, q_grid_glo(0)%array_dp, &
               kappa, volume, recip_dp, &
               prefac1_dp, prefac2_dp, prefac3_dp, &
               m1_tbl_dp, m2_tbl_dp, m3_tbl_dp, &
               nfft1, nfft2, nfft3, eer, virial, lorthog, &
               q_calc_energy_virial)
       endif
#if KEY_LJPME==1
    endif
#endif
#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

#if KEY_DOMDEC==1
    if (q_test) then
       if (q_use_single()) then
          call test2_scalar_sumrc_ps(nfft1, nfft2, nfft3, kappa, volume, &
               prefac1_sp, prefac2_sp, prefac3_sp, recip_sp, &
               q_grid_glo(0)%tot_size, q_grid_glo(0)%array_sp, &
               eer, virial, q_calc_energy_virial)
       else
          call test2_scalar_sumrc_pd(nfft1, nfft2, nfft3, kappa, volume, &
               prefac1_dp, prefac2_dp, prefac3_dp, recip_dp, &
               q_grid_glo(0)%tot_size, q_grid_glo(0)%array_dp, &
               eer, virial, q_calc_energy_virial)
       endif
    endif
#endif

    !-----------FFT FORWARD--------------------

    call timer_stpstrt(T_scsum,T_FFT)

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('Forward FFT')
#endif
    if (q_use_single()) then
       call forward_rc_fft(q_grid_glo(0)%array_sp, q_grid_glo(0)%tot_size)
    else
       call forward_rc_fft(q_grid_glo(0)%array_dp, q_grid_glo(0)%tot_size)
    endif
#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

    call timer_stpstrt(T_FFT,T_GRADS)

    !-----------SPATIAL GRAD SUM--------------------

#if KEY_DOMDEC==1
    if (q_test) then
       call test1_grad_sumrc(natom, dx, dy, dz)
    endif
#endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('Grad sum')
#endif

#if KEY_LJPME==1
    if (do_ljpme) then
       if (q_use_single()) then
          call grad_sumrc_ps(n_grid_atom, grid_atom,  &
               chargef, recip_sp, &
               theta1_sp, theta2_sp, theta3_sp,  &
               dtheta1_sp, dtheta2_sp, dtheta3_sp,  &
               dx, dy, dz, fr1, fr2, fr3, &
               forder, nfft1, nfft2, nfft3,  &
               q_grid_glo(0)%tot_size, q_grid_glo(0)%array_sp, 2.0_chm_real4)
       else
          call grad_sumrc_pd(n_grid_atom, grid_atom,  &
               charge, recip_dp, &
               theta1, theta2, theta3,  &
               dtheta1, dtheta2, dtheta3,  &
               dx, dy, dz, fr1, fr2, fr3, &
               forder, nfft1, nfft2, nfft3,  &
               q_grid_glo(0)%tot_size, q_grid_glo(0)%array_dp, 2.0_chm_real8)
       endif
    else
#endif
       if (q_use_single()) then
          call grad_sumrc_ps(n_grid_atom, grid_atom,  &
               chargef, recip_sp, &
               theta1_sp, theta2_sp, theta3_sp,  &
               dtheta1_sp, dtheta2_sp, dtheta3_sp,  &
               dx, dy, dz, fr1, fr2, fr3, &
               forder, nfft1, nfft2, nfft3,  &
               q_grid_glo(0)%tot_size, q_grid_glo(0)%array_sp, ccelecf)
       else
          call grad_sumrc_pd(n_grid_atom, grid_atom,  &
               charge, recip_dp, &
               theta1, theta2, theta3,  &
               dtheta1, dtheta2, dtheta3,  &
               dx, dy, dz, fr1, fr2, fr3, &
               forder, nfft1, nfft2, nfft3,  &
               q_grid_glo(0)%tot_size, q_grid_glo(0)%array_dp, ccelec)
       endif
#if KEY_LJPME==1
    endif
#endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

#if KEY_DOMDEC==1
    if (q_test) then
       if (q_use_single()) then
          call test_replicate_grid_borders_ps(q_grid_glo(0)%array_sp, forder, nfft2, nfft3)
          call test2_grad_sumrc_ps(forder, n_grid_atom, grid_atom, &
               fr1, fr2, fr3, nfft1, nfft2, nfft3, recip_sp, chargef, &
               theta1_sp, theta2_sp, theta3_sp, dtheta1_sp, dtheta2_sp, dtheta3_sp, &
               q_grid_glo(0)%tot_size, q_grid_glo(0)%array_sp, dx, dy, dz)
       else
          call test_replicate_grid_borders_pd(q_grid_glo(0)%array_dp, forder, nfft2, nfft3)
          call test2_grad_sumrc_pd(forder, n_grid_atom, grid_atom, &
               fr1, fr2, fr3, nfft1, nfft2, nfft3, recip_dp, charge, &
               theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, &
               q_grid_glo(0)%tot_size, q_grid_glo(0)%array_dp, dx, dy, dz)
       endif
    endif
#endif

    call timer_stop(T_GRADS)
    
    return
  end subroutine spatial_do_pmesh_kspace

  ! *
  ! * Sets recip
  ! *
  subroutine calc_recip_pd(recip)
    use image,only:xtlabc
    use prssre,only:getvol
    use energym,only:volume,eprop
#if KEY_PBOUND==1
    use pbound,only:qboun,boxinv,boyinv,bozinv,pbound_getvol
#endif
    implicit none
    ! Input / Output
    real(chm_real),intent(out) :: recip(3,3)
    ! Parameters
    real(chm_real), parameter :: zero = 0.0_chm_real
    ! Variables
    real(chm_real) xtlinv(6)
    logical ok

#if KEY_PBOUND==1
    if (qboun) then
       ! APH: Assumes orthorhombic box
       call pbound_getvol(eprop(volume))
       xtlinv(1) = boxinv
       xtlinv(2) = zero
       xtlinv(3) = boyinv
       xtlinv(4) = zero
       xtlinv(5) = zero
       xtlinv(6) = bozinv
       ok = .true.
    else
#endif
       call getvol(eprop(volume))
       call invt33s(xtlinv,xtlabc,ok)
#if KEY_PBOUND==1
    endif
#endif

    if (.not.ok) then
       call wrndie(-5,'<colfft>','calc_recip_pd: problem with recip calculation')
    endif

    recip(1,1) = xtlinv(1)
    recip(2,2) = xtlinv(3)
    recip(3,3) = xtlinv(6)
    recip(1,2) = xtlinv(2)
    recip(2,1) = xtlinv(2)
    recip(1,3) = xtlinv(4)
    recip(3,1) = xtlinv(4)
    recip(2,3) = xtlinv(5)
    recip(3,2) = xtlinv(5)

    return
  end subroutine calc_recip_pd

  ! *
  ! * Sets recip
  ! *
  subroutine calc_recip_ps(recip)
    use image,only:xtlabc
    use prssre,only:getvol
    use energym,only:volume,eprop
#if KEY_PBOUND==1
    use pbound,only:qboun,boxinv,boyinv,bozinv,pbound_getvol
#endif
    implicit none
    ! Input / Output
    real(chm_real4),intent(out) :: recip(3,3)
    ! Parameters
    real(chm_real4), parameter :: zero = 0.0_chm_real4
    ! Variables
    real(chm_real) xtlinv(6)
    logical ok

#if KEY_PBOUND==1
    if (qboun) then
       ! APH: Assumes orthorhombic box
       call pbound_getvol(eprop(volume))
       xtlinv(1) = boxinv
       xtlinv(2) = zero
       xtlinv(3) = boyinv
       xtlinv(4) = zero
       xtlinv(5) = zero
       xtlinv(6) = bozinv
       ok = .true.
    else
#endif
       call getvol(eprop(volume))
       call invt33s(xtlinv,xtlabc,ok)
#if KEY_PBOUND==1
    endif
#endif

    if (.not.ok) then
       call wrndie(-5,'<colfft>','calc_recip_ps: problem with recip calculation')
    endif

    recip(1,1) = xtlinv(1)
    recip(2,2) = xtlinv(3)
    recip(3,3) = xtlinv(6)
    recip(1,2) = xtlinv(2)
    recip(2,1) = xtlinv(2)
    recip(1,3) = xtlinv(4)
    recip(3,1) = xtlinv(4)
    recip(2,3) = xtlinv(5)
    recip(3,2) = xtlinv(5)

    return
  end subroutine calc_recip_ps

  ! *
  ! * Filter atoms into grid_atom -list
  ! *
  subroutine filter_grid_atom(natom, nfft2, nfft3, forder, x, y, z, n_grid_atom, grid_atom)
#if KEY_PARALLEL==1
    use parallel,only:mynod,numnod
    use colfft_util,only:coord_to_grid,get_spatial_limits,YZ_X_PARTITION, &
         filter_atom,q_ydim_periodic,q_zdim_periodic
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
#endif
    use memory
    implicit none
    ! Input / Output
    integer, intent(in) :: natom, nfft2, nfft3, forder
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    integer, intent(out) :: n_grid_atom, grid_atom(:)
    ! Variables
    integer n
#if KEY_PARALLEL==1
    integer ydim, zdim, ymin, ymax, zmin, zmax
    integer xgridmin,xgridmax,ygridmin,ygridmax,zgridmin,zgridmax
    real(chm_real) fr2n, fr3n
    real(chm_real), dimension(3,3) :: recip
#endif

#if KEY_DOMDEC==1
    if (q_domdec) then
       call wrndie(-5,'<colfft>','filter_grid_atom: DOMDEC should not end up here!')
    endif
#endif

#if KEY_PARALLEL==1
    call get_spatial_limits(YZ_X_PARTITION, &
         xgridmin,xgridmax,ygridmin,ygridmax,  &
         zgridmin,zgridmax,mynod)

    call calc_recip_pd(recip)

    if (q_ydim_periodic(forder)) then
       ydim = nfft2
    else
       ydim = nfft2*4          ! Make sure periodic boundaries are NOT used
    endif

    if (q_zdim_periodic(forder)) then
       zdim = nfft3
    else
       zdim = nfft3*4          ! Make sure periodic boundaries are NOT used
    endif

    ymin = ygridmin
    ymax = ygridmax
    zmin = zgridmin
    zmax = zgridmax
#endif

    n_grid_atom = 0

    ! code for filtering atoms. In single proc mode, do all atoms
#if KEY_PARALLEL==1
    if (numnod > 1) then
       do n = 1,natom
          call coord_to_grid(x(n), y(n), z(n), recip, fr2n, fr3n)
          if (.not.filter_atom(int(fr2n), forder, ymin, ymax, ydim) .or. &
               .not.filter_atom(int(fr3n), forder, zmin, zmax, zdim)) cycle
          n_grid_atom = n_grid_atom + 1
          grid_atom(n_grid_atom) = n
       end do
    else
#endif
       n_grid_atom = natom
       do n=1,natom
          grid_atom(n) = n
       enddo
#if KEY_PARALLEL==1
    endif
#endif

    return
  end subroutine filter_grid_atom

  ! *
  ! * Uninitialize column fft
  ! *
  subroutine colfft_uninit()
    use colfft_util,only:colfft_util_uninit
    implicit none

!!$    if (.not.colfft_allocated) then
!!$       call wrndie(-2,'<colfft>','colfft_uninit: colfft not allocated, returning')
!!$       return
!!$    endif

    call colfft_util_uninit()

    call deallocate_colfft()

    forder_save=0
    nfft1_save=0
    nfft2_save=0
    nfft3_save=0

!!$    colfft_allocated = .false.

    return
  end subroutine colfft_uninit

  ! *
  ! * Allocate & reallocates arrays
  ! *
  subroutine colfft_init(natom, n_grid_atom)
    use parallel,only:numnod
    use domdec_common,only:q_use_single, nthread
    use pmeutil,only:nfft1, nfft2, nfft3, forder,&
                    bsp_mod1, bsp_mod2, bsp_mod3
    use memory
    implicit none
    ! Input / Output
    integer, intent(in) :: natom, n_grid_atom
    ! Variables
    integer natom_alloc
    integer w1, w2, w3
    integer t, len
    
    if (forder < 3) call wrndie(-5,'<colfft>','colfft_init: Must use PME order >= 3')

    natom_alloc = min(natom, int(1.5*n_grid_atom))

    call allocate_q_grid(forder, nthread, q_use_single())

    w1 = nfft1/2 + 1
    w2 = nfft2/2 + 1
    w3 = nfft3/2 + 1

    ! (m1_tbl_dp, m2_tbl_dp, m3_tbl_dp)
    if (allocated(m1_tbl_dp)) then
       if (size(m1_tbl_dp) /= w1*2+1 .or. &
            size(m2_tbl_dp) /= w2*2+1 .or. &
            size(m3_tbl_dp) /= w3*2+1 .or. q_use_single()) then
          call chmdealloc('colfft.src','colfft_init','m1_tbl_dp',&
               size(m1_tbl_dp),crl=m1_tbl_dp)
          call chmdealloc('colfft.src','colfft_init','m2_tbl_dp',&
               size(m2_tbl_dp),crl=m2_tbl_dp)
          call chmdealloc('colfft.src','colfft_init','m3_tbl_dp',&
               size(m3_tbl_dp),crl=m3_tbl_dp)
       endif
    endif

    ! (m1_tbl_sp, m2_tbl_sp, m3_tbl_sp)
    if (allocated(m1_tbl_sp)) then
       if (size(m1_tbl_sp) /= w1*2+1 .or. &
            size(m2_tbl_sp) /= w2*2+1 .or. &
            size(m3_tbl_sp) /= w3*2+1 .or. .not.q_use_single()) then
          call chmdealloc('colfft.src','colfft_init','m1_tbl_sp',&
               size(m1_tbl_sp),cr4=m1_tbl_sp)
          call chmdealloc('colfft.src','colfft_init','m2_tbl_sp',&
               size(m2_tbl_sp),cr4=m2_tbl_sp)
          call chmdealloc('colfft.src','colfft_init','m3_tbl_sp',&
               size(m3_tbl_sp),cr4=m3_tbl_sp)
       endif
    endif

    ! (fr1, fr2, fr3)
    if (allocated(fr1)) then
       if (size(fr1) < n_grid_atom) then
          call chmdealloc('colfft.src','colfft_init','fr1',size(fr1),intg=fr1)
          call chmdealloc('colfft.src','colfft_init','fr2',size(fr2),intg=fr2)
          call chmdealloc('colfft.src','colfft_init','fr3',size(fr3),intg=fr3)
       endif
    endif

    if (.not.allocated(fr1)) then
       call chmalloc('colfft.src','colfft_init','fr1',natom_alloc,intg=fr1)
       call chmalloc('colfft.src','colfft_init','fr2',natom_alloc,intg=fr2)
       call chmalloc('colfft.src','colfft_init','fr3',natom_alloc,intg=fr3)
    endif

    ! (theta1, theta2, theta3)
    if (allocated(theta1)) then
       if (size(theta1,1) /= forder .or. size(theta1,2) < n_grid_atom .or. q_use_single()) then
          call chmdealloc('colfft.src','colfft_init','theta1',&
               size(theta1,1),size(theta1,2),crl=theta1)
          call chmdealloc('colfft.src','colfft_init','theta2',&
               size(theta2,1),size(theta2,2),crl=theta2)
          call chmdealloc('colfft.src','colfft_init','theta3',&
               size(theta3,1),size(theta3,2),crl=theta3)
          call chmdealloc('colfft.src','colfft_init','dtheta1',&
               size(dtheta1,1),size(dtheta1,2),crl=dtheta1)
          call chmdealloc('colfft.src','colfft_init','dtheta2',&
               size(dtheta2,1),size(dtheta2,2),crl=dtheta2)
          call chmdealloc('colfft.src','colfft_init','dtheta3',&
               size(dtheta3,1),size(dtheta3,2),crl=dtheta3)
       endif
    endif

    ! (theta1_sp, theta2_sp, theta3_sp)
    if (allocated(theta1_sp)) then
       if (size(theta1_sp,1) /= forder .or. size(theta1_sp,2) < n_grid_atom .or. &
            .not.q_use_single()) then
          call chmdealloc('colfft.src','colfft_init','theta1_sp',&
               size(theta1_sp,1),size(theta1_sp,2),cr4=theta1_sp)
          call chmdealloc('colfft.src','colfft_init','theta2_sp',&
               size(theta2_sp,1),size(theta2_sp,2),cr4=theta2_sp)
          call chmdealloc('colfft.src','colfft_init','theta3_sp',&
               size(theta3_sp,1),size(theta3_sp,2),cr4=theta3_sp)
          call chmdealloc('colfft.src','colfft_init','dtheta1_sp',&
               size(dtheta1_sp,1),size(dtheta1_sp,2),cr4=dtheta1_sp)
          call chmdealloc('colfft.src','colfft_init','dtheta2_sp',&
               size(dtheta2_sp,1),size(dtheta2_sp,2),cr4=dtheta2_sp)
          call chmdealloc('colfft.src','colfft_init','dtheta3_sp',&
               size(dtheta3_sp,1),size(dtheta3_sp,2),cr4=dtheta3_sp)
       endif
    endif

    if (.not.q_use_single()) then
       ! Double precision

       if(.not.allocated(m1_tbl_dp)) then
          call chmalloc('colfft.src','colfft_init','m1_tbl_dp',w1*2+1,lbou=-w1,crl=m1_tbl_dp)
          call chmalloc('colfft.src','colfft_init','m2_tbl_dp',w2*2+1,lbou=-w2,crl=m2_tbl_dp)
          call chmalloc('colfft.src','colfft_init','m3_tbl_dp',w3*2+1,lbou=-w3,crl=m3_tbl_dp)
       end if

       if (.not.allocated(theta1)) then
          call chmalloc('colfft.src','colfft_init','theta1',forder,natom_alloc,&
               crl=theta1)
          call chmalloc('colfft.src','colfft_init','theta2',forder,natom_alloc,&
               crl=theta2)
          call chmalloc('colfft.src','colfft_init','theta3',forder,natom_alloc,&
               crl=theta3)
          call chmalloc('colfft.src','colfft_init','dtheta1',forder,natom_alloc,&
               crl=dtheta1)
          call chmalloc('colfft.src','colfft_init','dtheta2',forder,natom_alloc,&
               crl=dtheta2)
          call chmalloc('colfft.src','colfft_init','dtheta3',forder,natom_alloc,&
               crl=dtheta3)
       endif

    else
       ! Single precision

       if(.not.allocated(m1_tbl_sp)) then
          call chmalloc('colfft.src','colfft_init','m1_tbl_sp',&
               w1*2+1,lbou=-w1,cr4=m1_tbl_sp)
          call chmalloc('colfft.src','colfft_init','m2_tbl_sp',&
               w2*2+1,lbou=-w2,cr4=m2_tbl_sp)
          call chmalloc('colfft.src','colfft_init','m3_tbl_sp',&
               w3*2+1,lbou=-w3,cr4=m3_tbl_sp)
       end if

       if (.not.allocated(theta1_sp)) then
          call chmalloc('colfft.src','colfft_init','theta1_sp',forder,natom_alloc,&
               cr4=theta1_sp)
          call chmalloc('colfft.src','colfft_init','theta2_sp',forder,natom_alloc,&
               cr4=theta2_sp)
          call chmalloc('colfft.src','colfft_init','theta3_sp',forder,natom_alloc,&
               cr4=theta3_sp)
          call chmalloc('colfft.src','colfft_init','dtheta1_sp',forder,natom_alloc,&
               cr4=dtheta1_sp)
          call chmalloc('colfft.src','colfft_init','dtheta2_sp',forder,natom_alloc,&
               cr4=dtheta2_sp)
          call chmalloc('colfft.src','colfft_init','dtheta3_sp',forder,natom_alloc,&
               cr4=dtheta3_sp)
       endif

    endif

    ! bsp_mod1
    if (allocated(bsp_mod1)) then
       if (size(bsp_mod1) /= nfft1+1) then
          call chmdealloc('colfft.src','colfft_init','bsp_mod1',size(bsp_mod1),crl=bsp_mod1)
       endif
    endif

    if (.not.allocated(bsp_mod1)) then
       call chmalloc('colfft.src','colfft_init','bsp_mod1',nfft1+1,crl=bsp_mod1)
    endif

    ! bsp_mod2
    if (allocated(bsp_mod2)) then
       if (size(bsp_mod2) /= nfft2+1) then
          call chmdealloc('colfft.src','colfft_init','bsp_mod2',size(bsp_mod2),crl=bsp_mod2)
       endif
    endif

    if (.not.allocated(bsp_mod2)) then
       call chmalloc('colfft.src','colfft_init','bsp_mod2',nfft2+1,crl=bsp_mod2)
    endif

    ! bsp_mod3
    if (allocated(bsp_mod3)) then
       if (size(bsp_mod3) /= nfft3+1) then
          call chmdealloc('colfft.src','colfft_init','bsp_mod3',size(bsp_mod3),crl=bsp_mod3)
       endif
    endif

    if (.not.allocated(bsp_mod3)) then
       call chmalloc('colfft.src','colfft_init','bsp_mod3',nfft3+1,crl=bsp_mod3)
    endif

    if (.not.q_use_single()) then
       ! Double precision
       if (allocated(prefac1_dp)) then
          if (size(prefac1_dp) /= nfft1) then
             call chmdealloc('colfft.src','colfft_init','prefac1_dp',&
                  size(prefac1_dp),crl=prefac1_dp)
          endif
       endif
       if (allocated(prefac2_dp)) then
          if (size(prefac2_dp) /= nfft2) then
             call chmdealloc('colfft.src','colfft_init','prefac2_dp',&
                  size(prefac2_dp),crl=prefac2_dp)
          endif
       endif
       if (allocated(prefac3_dp)) then
          if (size(prefac3_dp) /= nfft3) then
             call chmdealloc('colfft.src','colfft_init','prefac3_dp',&
                  size(prefac3_dp),crl=prefac3_dp)
          endif
       endif
       if (allocated(prefac1_sp)) then
          call chmdealloc('colfft.src','colfft_init','prefac1_sp',&
               size(prefac1_sp),cr4=prefac1_sp)
       endif
       if (allocated(prefac2_sp)) then
          call chmdealloc('colfft.src','colfft_init','prefac2_sp',&
               size(prefac2_sp),cr4=prefac2_sp)
       endif
       if (allocated(prefac3_sp)) then
          call chmdealloc('colfft.src','colfft_init','prefac3_sp',&
               size(prefac3_sp),cr4=prefac3_sp)
       endif
       if (.not.allocated(prefac1_dp)) then
          call chmalloc('colfft.src','colfft_init','prefac1_dp',&
               nfft1,crl=prefac1_dp)
       endif
       if (.not.allocated(prefac2_dp)) then
          call chmalloc('colfft.src','colfft_init','prefac2_dp',&
               nfft2,crl=prefac2_dp)
       endif
       if (.not.allocated(prefac3_dp)) then
          call chmalloc('colfft.src','colfft_init','prefac3_dp',&
               nfft3,crl=prefac3_dp)
       endif
    else

       ! Single precision
       if (allocated(prefac1_sp)) then
          if (size(prefac1_sp) /= nfft1) then
             call chmdealloc('colfft.src','colfft_init','prefac1_sp',&
                  size(prefac1_sp),cr4=prefac1_sp)
          endif
       endif
       if (allocated(prefac2_sp)) then
          if (size(prefac2_sp) /= nfft2) then
             call chmdealloc('colfft.src','colfft_init','prefac2_sp',&
                  size(prefac2_sp),cr4=prefac2_sp)
          endif
       endif
       if (allocated(prefac3_sp)) then
          if (size(prefac3_sp) /= nfft3) then
             call chmdealloc('colfft.src','colfft_init','prefac3_sp',&
                  size(prefac3_sp),cr4=prefac3_sp)
          endif
       endif
       if (allocated(prefac1_dp)) then
          call chmdealloc('colfft.src','colfft_init','prefac1_dp',&
               size(prefac1_dp),crl=prefac1_dp)
       endif
       if (allocated(prefac2_dp)) then
          call chmdealloc('colfft.src','colfft_init','prefac2_dp',&
               size(prefac2_dp),crl=prefac2_dp)
       endif
       if (allocated(prefac3_dp)) then
          call chmdealloc('colfft.src','colfft_init','prefac3_dp',&
               size(prefac3_dp),crl=prefac3_dp)
       endif
       if (.not.allocated(prefac1_sp)) then
          call chmalloc('colfft.src','colfft_init','prefac1_sp',&
               nfft1,cr4=prefac1_sp)
       endif
       if (.not.allocated(prefac2_sp)) then
          call chmalloc('colfft.src','colfft_init','prefac2_sp',&
               nfft2,cr4=prefac2_sp)
       endif
       if (.not.allocated(prefac3_sp)) then
          call chmalloc('colfft.src','colfft_init','prefac3_sp',&
               nfft3,cr4=prefac3_sp)
       endif

    endif

    ! (xf, yf, zf, chargef)
    if (q_use_single()) then

       if (allocated(xf)) then
          if (size(xf) /= natom) then
             call chmdealloc('colfft.src','colfft_init','xf',size(xf),cr4=xf)
             call chmdealloc('colfft.src','colfft_init','yf',size(yf),cr4=yf)
             call chmdealloc('colfft.src','colfft_init','zf',size(zf),cr4=zf)
             call chmdealloc('colfft.src','colfft_init','chargef',size(chargef),cr4=chargef)
          endif
       endif

       if (.not.allocated(xf)) then
          call chmalloc('colfft.src','colfft_init','xf',natom,cr4=xf)
          call chmalloc('colfft.src','colfft_init','yf',natom,cr4=yf)
          call chmalloc('colfft.src','colfft_init','zf',natom,cr4=zf)
          call chmalloc('colfft.src','colfft_init','chargef',natom,cr4=chargef)
       endif

    endif

    ! thread_id_list
    if (allocated(thread_id_list)) then
       if (size(thread_id_list) < n_grid_atom) then
          call chmdealloc('colfft.src','colfft_init','thread_id_list',&
               size(thread_id_list),intg=thread_id_list)
       endif
    endif

    if (.not.allocated(thread_id_list)) then
       call chmalloc('colfft.src','colfft_init','thread_id_list',&
            ceiling(real(n_grid_atom)*1.5),intg=thread_id_list)
    endif

    ! atomlist_thread
    if (allocated(atomlist_thread)) then
       if (size(atomlist_thread) < nthread) then
          do t=0,size(atomlist_thread)-1
             call chmdealloc('colfft.src','colfft_init','atomlist_thread%array',&
                  size(atomlist_thread(t)%array),intg=atomlist_thread(t)%array)
          enddo
          deallocate(atomlist_thread)
       endif
    endif

    if (.not.allocated(atomlist_thread)) then
       allocate(atomlist_thread(0:nthread-1))
       do t=0,nthread-1
          call chmalloc('colfft.src','colfft_init','atomlist_thread%array',&
               ceiling(real(n_grid_atom)/real(nthread)*1.5)+1,intg=atomlist_thread(t)%array)
       enddo
    endif

    ! natom_thread(0:nthread)%array(0:nthread+15)
    if (allocated(natom_thread)) then
       if (size(natom_thread) < nthread+1) then
          do t=0,size(natom_thread)-1
             call chmdealloc('colfft.src','colfft_init','natom_thread(t)%array',&
                  size(natom_thread(t)%array),intg=natom_thread(t)%array)
          enddo
       endif
       deallocate(natom_thread)
    endif

    if (.not.allocated(natom_thread)) then
       allocate(natom_thread(0:nthread))
       do t=0,nthread
          call chmalloc('colfft.src','colfft_init','natom_thread',&
               nthread+16,lbou=0,intg=natom_thread(t)%array)
       enddo
    endif

    return
  end subroutine colfft_init

  ! *
  ! * Deallocate arrays
  ! *
  subroutine deallocate_colfft()
    use memory
    implicit none
    integer t

    call deallocate_q_grid()

    if (allocated(m1_tbl_dp)) then
       call chmdealloc('colfft.src','deallocate_colfft','m1_tbl_dp',&
            size(m1_tbl_dp),crl=m1_tbl_dp)
       call chmdealloc('colfft.src','deallocate_colfft','m2_tbl_dp',&
            size(m2_tbl_dp),crl=m2_tbl_dp)
       call chmdealloc('colfft.src','deallocate_colfft','m3_tbl_dp',&
            size(m3_tbl_dp),crl=m3_tbl_dp)
    endif

    if (allocated(m1_tbl_sp)) then
       call chmdealloc('colfft.src','deallocate_colfft','m1_tbl_sp',&
            size(m1_tbl_sp),cr4=m1_tbl_sp)
       call chmdealloc('colfft.src','deallocate_colfft','m2_tbl_sp',&
            size(m2_tbl_sp),cr4=m2_tbl_sp)
       call chmdealloc('colfft.src','deallocate_colfft','m3_tbl_sp',&
            size(m3_tbl_sp),cr4=m3_tbl_sp)
    endif

    if (allocated(fr1)) then
       call chmdealloc('colfft.src','deallocate_colfft','fr1',size(fr1),intg=fr1)
       call chmdealloc('colfft.src','deallocate_colfft','fr2',size(fr2),intg=fr2)
       call chmdealloc('colfft.src','deallocate_colfft','fr3',size(fr3),intg=fr3)
    endif

    if (allocated(theta1)) then
       call chmdealloc('colfft.src','deallocate_colfft','theta1',&
            size(theta1,1),size(theta1,2),crl=theta1)
       call chmdealloc('colfft.src','deallocate_colfft','theta2',&
            size(theta2,1),size(theta2,2),crl=theta2)
       call chmdealloc('colfft.src','deallocate_colfft','theta3',&
            size(theta3,1),size(theta3,2),crl=theta3)
       call chmdealloc('colfft.src','deallocate_colfft','dtheta1',&
            size(dtheta1,1),size(dtheta1,2),crl=dtheta1)
       call chmdealloc('colfft.src','deallocate_colfft','dtheta2',&
            size(dtheta2,1),size(dtheta2,2),crl=dtheta2)
       call chmdealloc('colfft.src','deallocate_colfft','dtheta3',&
            size(dtheta3,1),size(dtheta3,2),crl=dtheta3)
    endif

    if (allocated(theta1_sp)) then
       call chmdealloc('colfft.src','deallocate_colfft','theta1_sp',&
            size(theta1_sp,1),size(theta1_sp,2),cr4=theta1_sp)
       call chmdealloc('colfft.src','deallocate_colfft','theta2_sp',&
            size(theta2_sp,1),size(theta2_sp,2),cr4=theta2_sp)
       call chmdealloc('colfft.src','deallocate_colfft','theta3_sp',&
            size(theta3_sp,1),size(theta3_sp,2),cr4=theta3_sp)
       call chmdealloc('colfft.src','deallocate_colfft','dtheta1_sp',&
            size(dtheta1_sp,1),size(dtheta1_sp,2),cr4=dtheta1_sp)
       call chmdealloc('colfft.src','deallocate_colfft','dtheta2_sp',&
            size(dtheta2_sp,1),size(dtheta2_sp,2),cr4=dtheta2_sp)
       call chmdealloc('colfft.src','deallocate_colfft','dtheta3_sp',&
            size(dtheta3_sp,1),size(dtheta3_sp,2),cr4=dtheta3_sp)
    endif

    if (allocated(prefac1_sp)) then
       call chmdealloc('colfft.src','deallocate_colfft','prefac1_sp',&
            size(prefac1_sp),cr4=prefac1_sp)
    endif
    if (allocated(prefac2_sp)) then
       call chmdealloc('colfft.src','deallocate_colfft','prefac2_sp',&
            size(prefac2_sp),cr4=prefac2_sp)
    endif
    if (allocated(prefac3_sp)) then
       call chmdealloc('colfft.src','deallocate_colfft','prefac3_sp',&
            size(prefac3_sp),cr4=prefac3_sp)
    endif

    if (allocated(prefac1_dp)) then
       call chmdealloc('colfft.src','deallocate_colfft','prefac1_dp',&
            size(prefac1_dp),crl=prefac1_dp)
    endif
    if (allocated(prefac2_dp)) then
       call chmdealloc('colfft.src','deallocate_colfft','prefac2_dp',&
            size(prefac2_dp),crl=prefac2_dp)
    endif
    if (allocated(prefac3_dp)) then
       call chmdealloc('colfft.src','deallocate_colfft','prefac3_dp',&
            size(prefac3_dp),crl=prefac3_dp)
    endif

    ! (xf, yf, zf, chargef)
    if (allocated(xf)) then
       call chmdealloc('colfft.src','allocate_colfft','xf',size(xf),cr4=xf)
       call chmdealloc('colfft.src','allocate_colfft','yf',size(yf),cr4=yf)
       call chmdealloc('colfft.src','allocate_colfft','zf',size(zf),cr4=zf)
       call chmdealloc('colfft.src','allocate_colfft','chargef',size(chargef),cr4=chargef)
    endif

    ! thread_id_list
    if (allocated(thread_id_list)) then
       call chmdealloc('colfft.src','deallocate_colfft','thread_id_list',&
            size(thread_id_list),intg=thread_id_list)
    endif

    ! natomlist_thread
    if (allocated(atomlist_thread)) then
       do t=0,size(atomlist_thread)-1
          call chmdealloc('colfft.src','deallocate_colfft','atomlist_thread%array',&
               size(atomlist_thread(t)%array),intg=atomlist_thread(t)%array)
       enddo
       deallocate(atomlist_thread)
    endif
    
    ! natom_thread
    if (allocated(natom_thread)) then
       do t=0,size(natom_thread)-1
          call chmdealloc('colfft.src','deallocate_colfft','natom_thread(t)%array',&
               size(natom_thread(t)%array),intg=natom_thread(t)%array)
       enddo
       deallocate(natom_thread)
    endif

    return
  end subroutine deallocate_colfft

  ! *
  ! * Allocate & Initialize charge grid(s)
  ! * NOTE: In case of single thread, we only allocate the global grid
  ! *
  subroutine allocate_q_grid(forder, nthread, q_single)
    use stream,only:outu
    use parallel,only:mynod
    use colfft_util,only:get_spatial_limits, get_spatial_sizes, &
         YZ_X_PARTITION, XY_Z_PARTITION, ZX_Y_PARTITION, &
#if KEY_FFTW==1 || KEY_MKL==1
         reset_fftw_plans, &
#endif
         ny_box, nz_box
#if KEY_DOMDEC==1
    use domdec_common,only:q_test
#endif
    use memory,only:chmalloc, chmdealloc
    implicit none
    ! Input / Output
    integer, intent(in) :: forder, nthread
    logical, intent(in) :: q_single
    ! Variables
    logical q_redo_grid2t, q_changed
    integer nfftx, nffty, nfftz
    integer xgridlo, ygridlo, zgridlo, xgridhi, ygridhi, zgridhi
    integer xgridsize, ygridsize, zgridsize
    integer xlo, ylo, zlo, xhi, yhi, zhi
    integer xsize, ysize, zsize
    integer tx, ty, tz, ngridp, min_ngridp
    integer t, i, istart, iend
    integer x0, x1, y0, y1, z0, z1
    integer q_size

    ! Check if forder has changed
    q_changed = (forder /= grid2t_forder)

    ! Save forder value
    grid2t_forder = forder

    call get_spatial_limits(YZ_X_PARTITION, x0, x1, y0, y1, z0, z1, mynod)

    call set_q_grid_param(0, 0, 0, x0, y0, z0, x1, y1, z1, 1, 1, 1, &
         forder, ny_box, nz_box, q_grid_glo(0))

    ! Determine max global charge grid size
    call get_spatial_sizes(ZX_Y_PARTITION, ygridsize, zgridsize, xgridsize, mynod)
    q_size = xgridsize*ygridsize*zgridsize
    call get_spatial_sizes(XY_Z_PARTITION, zgridsize, xgridsize, ygridsize, mynod)
    q_size = max(q_size, xgridsize*ygridsize*zgridsize)
    call get_spatial_sizes(YZ_X_PARTITION, xgridsize, ygridsize, zgridsize, mynod)
    q_size = max(q_size, xgridsize*ygridsize*zgridsize)

    ! Allocate q_grid_glo(0) to size q_size and set (xsize, ysize, zsize) manually
    call alloc_realloc_q_grid_array(q_size, q_single, q_grid_glo(0), q_changed)
    q_grid_glo(0)%xsize = xgridsize
    q_grid_glo(0)%ysize = ygridsize
    q_grid_glo(0)%zsize = zgridsize

    ! Grid point region                       Writing region ( +(forder-1) )
    ! For periodic systems:
    ! fr1i = x0...x1                       -> x0...x1 + forder-1
    ! fr2i = y0-(forder-1)...y1-(forder-1) -> y0-(forder-1)...y1
    ! fr3i = z0-(forder-1)...z1-(forder-1) -> z0-(forder-1)...z1
    ! For non-periodic systems:
    ! fr1i = x0...x1                    -> x0...x1 + forder-1
    ! fr2i = y0...y1                    -> y0...y1 + forder-1
    ! fr3i = z0...z1                    -> z0...z1 + forder-1

    ! Set node grid point region
    xgridlo = x0
    xgridhi = x1
    ygridlo = y0
    ygridhi = y1
    zgridlo = z0
    zgridhi = z1
    if (ny_box > 1) ygridlo = ygridlo - (forder-1)
    if (nz_box > 1) zgridlo = zgridlo - (forder-1)

    ! Set node writing region
    xlo = xgridlo
    xhi = xgridhi + (forder-1)
    ylo = ygridlo
    yhi = ygridhi + (forder-1)
    zlo = zgridlo
    zhi = zgridhi + (forder-1)

    xgridsize = xgridhi - xgridlo + 1
    ygridsize = ygridhi - ygridlo + 1
    zgridsize = zgridhi - zgridlo + 1

    xsize = xhi - xlo + 1
    ysize = yhi - ylo + 1
    zsize = zhi - zlo + 1

    q_redo_grid2t = q_changed

    grid2tx_lo = xgridlo
    grid2ty_lo = ygridlo
    grid2tz_lo = zgridlo

    grid2tx_hi = xgridhi
    grid2ty_hi = ygridhi
    grid2tz_hi = zgridhi

    ! grid2tx
    if (allocated(grid2tx)) then
       if (size(grid2tx) < xgridsize) then
          call chmdealloc('colfft.src','allocate_q_grid','grid2tx',&
               size(grid2tx),intg=grid2tx)
       endif
    endif

    if (.not.allocated(grid2tx)) then
       call chmalloc('colfft.src','allocate_q_grid','grid2tx',&
            xgridsize,lbou=0,intg=grid2tx)
       q_redo_grid2t = .true.
    endif

    ! grid2ty
    if (allocated(grid2ty)) then
       if (size(grid2ty) < ygridsize) then
          call chmdealloc('colfft.src','allocate_q_grid','grid2ty',&
               size(grid2ty),intg=grid2ty)
       endif
    endif

    if (.not.allocated(grid2ty)) then
       call chmalloc('colfft.src','allocate_q_grid','grid2ty',&
            ygridsize,lbou=0,intg=grid2ty)
       q_redo_grid2t = .true.
    endif

    ! grid2tz
    if (allocated(grid2tz)) then
       if (size(grid2tz) < zgridsize) then
          call chmdealloc('colfft.src','allocate_q_grid','grid2tz',&
               size(grid2tz),intg=grid2tz)
       endif
    endif

    if (.not.allocated(grid2tz)) then
       call chmalloc('colfft.src','allocate_q_grid','grid2tz',&
            zgridsize,lbou=0,intg=grid2tz)
       q_redo_grid2t = .true.
    endif

    if (grid2t_nthread /= nthread) q_redo_grid2t = .true.

    ! Redo (grid2tx, grid2ty, grid2tz) if neccessary
    if (q_redo_grid2t) then

       nfftx = x1 - x0 + 1
       nffty = y1 - y0 + 1
       nfftz = z1 - z0 + 1

       ! Initialize variables
       ntx = 0
       nty = 0
       ntz = 0
       min_ngridp = (nfftx+forder+1)*(nffty+forder+1)*(nfftz+forder+1)*2

       ! Loop through all possible number of thread numbers (tx, ty, tz)
       do tx=1,nthread
          ! Only allow values of tx that divide nthread evenly
          if (mod(nthread,tx) == 0) then
             do ty=1,nthread
                if (mod(nthread,tx*ty) == 0 .and. tx*ty <= nthread) then
                   tz = nthread/(tx*ty)
                   ! Calculate the approximate number of grid points for
                   ! thread numbers (tx, ty, tz)
                   ngridp = (ceiling(real(nfftx)/real(tx) + forder))&
                        *(ceiling(real(nffty)/real(ty) + forder))&
                        *(ceiling(real(nfftz)/real(tz) + forder))
                   if (ngridp < min_ngridp .or. (ngridp == min_ngridp .and. &
                        (tx < ntx .or. (tx == ntx .and. ty < nty))) ) then
                      min_ngridp = ngridp
                      ntx = tx
                      nty = ty
                      ntz = tz
                   endif
                endif
             enddo
          endif
       enddo

!!$       ! -------------------------------
!!$       ! APH: TEMPORARY - REMOVE THIS!
!!$       if (nthread == 8) then
!!$          ntx = 1
!!$          nty = 4
!!$          ntz = 2
!!$       endif
!!$       ! -------------------------------

       if (ntx*nty*ntz /= nthread) then
          write (outu,'(a,4i3)') 'ntx,nty,ntz,nthread=',ntx,nty,ntz,nthread
          call wrndie(-5,'<colfft>','Unable to find appropriate thread division')
       endif

       if (nthread > 1) then
          if (mynod == 0) write (outu,'(a,3i3)') 'COLFFT thread division ntx,nty,ntz=',ntx,nty,ntz
       endif

       grid2tx = -1
       grid2ty = -1
       grid2tz = -1

       do t=0,ntz-1
          istart = z0 - grid2tz_lo + t*nfftz/ntz
          iend = z0 - grid2tz_lo + (t+1)*nfftz/ntz - 1
          if (t == 0 .and. nz_box > 1) then
             istart = istart - (forder-1)
          endif
          grid2tz(istart:iend) = t
       enddo
       if (grid2tz_lo < z0) then
          grid2tz(0:z0 - grid2tz_lo - 1) = 0
       endif

       do t=0,nty-1
          istart = y0 - grid2ty_lo + t*nffty/nty
          iend = y0 - grid2ty_lo + (t+1)*nffty/nty - 1
          if (t == 0 .and. ny_box > 1) then
             istart = istart - (forder-1)
          endif
          grid2ty(istart:iend) = t*ntz
       enddo
       if (grid2ty_lo < y0) then
          grid2ty(0:y0 - grid2ty_lo - 1) = 0
       endif

       do t=0,ntx-1
          istart = x0 - grid2tx_lo + t*nfftx/ntx
          iend = x0 - grid2tx_lo + (t+1)*nfftx/ntx - 1
          grid2tx(istart:iend) = t*ntz*nty
       enddo

       ! Deallocate & Allocate q_grid_loc
       if (allocated(q_grid_loc)) then
          if (size(q_grid_loc) /= nthread) then
             call dealloc_q_grid_local(q_grid_loc, 0, size(q_grid_loc)-1)
          endif
       endif

       if (nthread > 1) then
          if (.not.allocated(q_grid_loc)) then
             allocate(q_grid_loc(0:nthread-1))
          endif

          ! Deallocate & Allocate q_grid_loc(0:nthread-1)%array_dp/sp
          call alloc_realloc_q_grid_local(x0, y0, z0, x1, y1, z1, ntx, nty, ntz, &
               forder, ny_box, nz_box, q_single, q_grid_loc, q_changed)
       endif

       grid2t_nthread = nthread

#if KEY_DOMDEC==1
       if (q_test) then
          call test_allocate_q_grid(xgridlo, xgridhi, ygridlo, ygridhi, zgridlo, zgridhi)
       endif
#endif

    endif

    if (q_changed) then
#if KEY_FFTW==1 || KEY_MKL==1
       call reset_fftw_plans()
#endif
    endif

    ! Set q_grid_p
    ! NOTE: q_grid_p points to the local grid, if it exists
    nullify(q_grid_p)
    if (nthread > 1) then
       q_grid_p => q_grid_loc
    else
       q_grid_p => q_grid_glo
    endif

    return
  end subroutine allocate_q_grid

  ! *
  ! * Allocate & reallocate local grid (with threads)
  ! *
  subroutine alloc_realloc_q_grid_local(x0, y0, z0, x1, y1, z1, ntx, nty, ntz, &
       forder, ny_box, nz_box, q_single, q_grid, q_changed)
    implicit none
    ! Input / Output
    integer, intent(in) :: x0, y0, z0, x1, y1, z1
    integer, intent(in) :: ntx, nty, ntz
    integer, intent(in) :: forder
    integer, intent(in) :: ny_box, nz_box
    logical, intent(in) :: q_single
    type(q_grid_t), intent(inout) :: q_grid(0:ntx*nty*ntz-1)
    logical, intent(inout) :: q_changed
    ! Variables
    integer tx, ty, tz, t, q_size

    ! Deallocate & Allocate q_grid(t)%array_dp/sp
    do tx=0,ntx-1
       do ty=0,nty-1
          do tz=0,ntz-1
             t = tz + ty*ntz + tx*ntz*nty
             call set_q_grid_param(tx, ty, tz, x0, y0, z0, x1, y1, z1, ntx, nty, ntz, &
                  forder, ny_box, nz_box, q_grid(t))
             q_size = q_grid(t)%xsize*q_grid(t)%ysize*q_grid(t)%zsize
             call alloc_realloc_q_grid_array(q_size, q_single, q_grid(t), q_changed)
          enddo
       enddo
    enddo

    return
  end subroutine alloc_realloc_q_grid_local

  ! *
  ! * Allocate & reallocate q_grid%array_sp/dp
  ! *
  subroutine alloc_realloc_q_grid_array(q_size, q_single, q_grid, q_changed)
    use memory
    implicit none
    ! Input / Output
    integer, intent(in) :: q_size
    logical, intent(in) :: q_single
    type(q_grid_t), intent(inout) :: q_grid
    logical, intent(inout) :: q_changed

    if (allocated(q_grid%array_sp)) then
       if (size(q_grid%array_sp) /= q_size .or. .not.q_single) then
          call chmdealloc('colfft.src','alloc_realloc_q_grid_array','q_grid%array_sp',&
               size(q_grid%array_sp),cr4=q_grid%array_sp)
          q_changed = .true.
       endif
    endif

    if (allocated(q_grid%array_dp)) then
       if (size(q_grid%array_dp) /= q_size .or. q_single) then
          call chmdealloc('colfft.src','alloc_realloc_q_grid_array','q_grid%array_dp',&
               size(q_grid%array_dp),crl=q_grid%array_dp)
          q_changed = .true.
       endif
    endif

    if (q_single) then
       if (.not.allocated(q_grid%array_sp)) then
          call chmalloc('colfft.src','alloc_realloc_q_grid_array','q_grid%array_sp',&
               q_size,lbou=0,cr4=q_grid%array_sp)
          q_changed = .true.
       endif
    else
       if (.not.allocated(q_grid%array_dp)) then
          call chmalloc('colfft.src','alloc_realloc_q_grid_array','q_grid%array_dp',&
               q_size,lbou=0,crl=q_grid%array_dp)
          q_changed = .true.
       endif
    endif

    q_grid%tot_size = q_size

    return
  end subroutine alloc_realloc_q_grid_array

  ! *
  ! * Deallocates q_grid_local
  ! *
  subroutine dealloc_q_grid_local(q_grid, lb, ub)
    use memory,only:chmdealloc
    ! Input / Output
    type(q_grid_t), allocatable, dimension(:), intent(inout) :: q_grid
    integer, intent(in) :: lb, ub
    ! Variables
    integer i

    if (allocated(q_grid)) then
       do i=lb,ub
          call dealloc_q_grid_array(q_grid(i))
       enddo
       deallocate(q_grid)
    endif

    return
  end subroutine dealloc_q_grid_local

  ! *
  ! * Deallocates q_grid_array_sp/dp
  ! *
  subroutine dealloc_q_grid_array(q_grid)
    use memory,only:chmdealloc
    ! Input / Output
    type(q_grid_t), intent(inout) :: q_grid

    if (allocated(q_grid%array_sp)) then
       call chmdealloc('colfft.src','dealloc_q_grid_array','q_grid%array_sp',&
            size(q_grid%array_sp),cr4=q_grid%array_sp)
    endif
    if (allocated(q_grid%array_dp)) then
       call chmdealloc('colfft.src','dealloc_q_grid_array','q_grid%array_dp',&
            size(q_grid%array_dp),crl=q_grid%array_dp)
    endif

    return
  end subroutine dealloc_q_grid_array

  ! *
  ! * Sets the parameters for q_grid_t -structure
  ! *
  subroutine set_q_grid_param(tx, ty, tz, x0, y0, z0, x1, y1, z1, ntx, nty, ntz, &
       forder, ny_box, nz_box, q_grid)
    implicit none
    ! Input / Output
    integer, intent(in) :: tx, ty, tz
    integer, intent(in) :: x0, y0, z0, x1, y1, z1
    integer, intent(in) :: ntx, nty, ntz
    integer, intent(in) :: forder, ny_box, nz_box
    type(q_grid_t), intent(inout) :: q_grid
    ! Variables
    integer  nfftx, nffty, nfftz

    ! Region boundary (x0...x1) (y0...y1) (z0...z1)
    nfftx = x1 - x0 + 1
    nffty = y1 - y0 + 1
    nfftz = z1 - z0 + 1

    q_grid%tx = tx
    q_grid%ty = ty
    q_grid%tz = tz
    ! Set region boundaries for thread t
    q_grid%x0 = x0 + tx*nfftx/ntx
    q_grid%x1 = x0 + (tx+1)*nfftx/ntx - 1
    q_grid%y0 = y0 + ty*nffty/nty
    q_grid%y1 = y0 + (ty+1)*nffty/nty - 1
    q_grid%z0 = z0 + tz*nfftz/ntz
    q_grid%z1 = z0 + (tz+1)*nfftz/ntz - 1
    ! Set grid point boundaries for thread t
    q_grid%xgridlo = q_grid%x0
    q_grid%xgridhi = q_grid%x1
    q_grid%ygridlo = q_grid%y0
    q_grid%ygridhi = q_grid%y1
    q_grid%zgridlo = q_grid%z0
    q_grid%zgridhi = q_grid%z1
    if (ny_box > 1) then
       q_grid%ygridlo = q_grid%ygridlo - (forder-1)
    endif
    if (nz_box > 1) then
       q_grid%zgridlo = q_grid%zgridlo - (forder-1)
    endif
    ! Set region writing boundaries for thread t
    q_grid%xlo = q_grid%xgridlo
    q_grid%xhi = q_grid%xgridhi + (forder-1)
    q_grid%ylo = q_grid%ygridlo
    q_grid%yhi = q_grid%ygridhi + (forder-1)
    q_grid%zlo = q_grid%zgridlo
    q_grid%zhi = q_grid%zgridhi + (forder-1)
    ! Calculate grid size based on the writing boundaries
    q_grid%xsize = q_grid%xhi - q_grid%xlo + 1
    q_grid%ysize = q_grid%yhi - q_grid%ylo + 1
    q_grid%zsize = q_grid%zhi - q_grid%zlo + 1

    return
  end subroutine set_q_grid_param

  ! *
  ! * Tests allocate_q_grid -subroutine
  ! *
  subroutine test_allocate_q_grid(xgridlo, xgridhi, ygridlo, ygridhi, zgridlo, zgridhi)
    use stream,only:outu
    use domdec_common,only:nthread
    use parallel,only:mynod
    implicit none
    ! Input
    integer, intent(in) :: xgridlo, xgridhi, ygridlo, ygridhi, zgridlo, zgridhi
    ! Variables
    logical q_tt_found
    integer i, ix, iy, iz, t, tt

    if (grid2tx_lo /= xgridlo .or. grid2ty_lo /= ygridlo .or. grid2tz_lo /= zgridlo .or. &
         grid2tx_hi /= xgridhi .or. grid2ty_hi /= ygridhi .or. grid2tz_hi /= zgridhi) then
       call wrndie(-5,'<colfft>','test_allocate_q_grid: incorrect settings for lo/hi boundaries')
    endif

    if (lbound(grid2tx,1) /= 0 .or. ubound(grid2tx,1) /= grid2tx_hi-grid2tx_lo .or. &
         lbound(grid2ty,1) /= 0 .or. ubound(grid2ty,1) /= grid2ty_hi-grid2ty_lo .or. &
         lbound(grid2tz,1) /= 0 .or. ubound(grid2tz,1) /= grid2tz_hi-grid2tz_lo) then
       call wrndie(-5,'<colfft>','test_allocate_q_grid: incorrect settings for lbound/ubound)')
    endif

    ! Make sure (grid2tx, grid2ty, grid2tz) are filled in
    do i=0,grid2tx_hi-grid2tx_lo
       if (grid2tx(i) < 0) then
          call wrndie(-5,'<colfft>','test_allocate_q_grid: grid2tx < 0, allocate_q_grid FAILED')
       endif
    enddo

    do i=0,grid2ty_hi-grid2ty_lo
       if (grid2ty(i) < 0) then
          call wrndie(-5,'<colfft>','test_allocate_q_grid: grid2ty < 0, allocate_q_grid FAILED')
       endif
    enddo

    do i=0,grid2tz_hi-grid2tz_lo
       if (grid2tz(i) < 0) then
          call wrndie(-5,'<colfft>','test_allocate_q_grid: grid2tz < 0, allocate_q_grid FAILED')
       endif
    enddo

    if (allocated(q_grid_loc)) then
       ! Go through grid points and make sure (grid2tx, grid2ty, grid2tz)
       ! matches settings in q_grid_loc
       do ix=xgridlo,xgridhi
          do iy=ygridlo,ygridhi
             do iz=zgridlo,zgridhi
                t = grid2tx(ix-grid2tx_lo) + grid2ty(iy-grid2ty_lo) + grid2tz(iz-grid2tz_lo)
                if (t < 0 .or. t > nthread-1) then
                   call wrndie(-5,'<colfft>','test_allocate_q_grid: grid2t(x,y,z) set failure')
                endif
                if (ix < q_grid_loc(t)%xgridlo .or. ix > q_grid_loc(t)%xgridhi .or. &
                     iy < q_grid_loc(t)%ygridlo .or. iy > q_grid_loc(t)%ygridhi .or. &
                     iz < q_grid_loc(t)%zgridlo .or. iz > q_grid_loc(t)%zgridhi) then
                   write (outu,'(a,i3)') 't=',t
                   write (outu,'(a,3i4)') 'ix,iy,iz=',ix,iy,iz
                   write (outu,'(a,2i4)') 'xgridlo,xgridhi=',&
                        q_grid_loc(t)%xgridlo,q_grid_loc(t)%xgridhi
                   write (outu,'(a,2i4)') 'ygridlo,ygridhi=',&
                        q_grid_loc(t)%ygridlo,q_grid_loc(t)%ygridhi
                   write (outu,'(a,2i4)') 'zgridlo,zgridhi=',&
                        q_grid_loc(t)%zgridlo,q_grid_loc(t)%zgridhi
                   call wrndie(-5,'<colfft>',&
                        'test_allocate_q_grid: grid2t(x,y,z) does not match q_grid_loc')
                endif
                q_tt_found = .false.
                do tt=0,nthread-1
                   if (ix >= q_grid_loc(tt)%x0 .and. ix <= q_grid_loc(tt)%x1 .and. &
                        iy >= q_grid_loc(tt)%y0 .and. iy <= q_grid_loc(tt)%y1 .and. &
                        iz >= q_grid_loc(tt)%z0 .and. iz <= q_grid_loc(tt)%z1) then
                      q_tt_found = .true.
                      exit
                   endif
                enddo
                if (q_tt_found .and. tt /= t) then
                   write (outu,'(a,5i4)') 'ix,iy,iz,tt,t=',ix,iy,iz,tt,t
                   write (outu,'(a,9i4)') 't, x0,x1=',t,q_grid_loc(t)%x0,q_grid_loc(t)%x1,&
                        q_grid_loc(t)%y0,q_grid_loc(t)%y1,q_grid_loc(t)%z0,q_grid_loc(t)%z1
                   write (outu,'(a,9i4)') 'tt,x0,x1=',tt,q_grid_loc(tt)%x0,q_grid_loc(tt)%x1,&
                        q_grid_loc(tt)%y0,q_grid_loc(tt)%y1,q_grid_loc(tt)%z0,q_grid_loc(tt)%z1
                   write (outu,'(a,2i4)') 'xgridlo,xgridhi=',xgridlo,xgridhi
                   write (outu,'(a,2i4)') 'ygridlo,ygridhi=',ygridlo,ygridhi
                   write (outu,'(a,2i4)') 'zgridlo,zgridhi=',zgridlo,zgridhi
                   write (outu,'(a,10i4)') 'grid2tx=',grid2tx(0:grid2tx_hi-grid2tx_lo)
                   write (outu,'(a,10i4)') 'grid2ty=',grid2ty(0:grid2ty_hi-grid2ty_lo)
                   write (outu,'(a,10i4)') 'grid2tz=',grid2tz(0:grid2tz_hi-grid2tz_lo)
                   call wrndie(-5,'<colfft>','test_allocate_q_grid: ERROR, found a closer thread')
                endif
             enddo
          enddo
       enddo
    endif

    if (mynod == 0) write (outu,'(a)') 'test_allocate_q_grid OK'

    return
  end subroutine test_allocate_q_grid

  ! *
  ! * Deallocate (grid2tx, grid2ty, gridtz) lists
  ! *
  subroutine deallocate_q_grid()
    use memory,only:chmdealloc
    implicit none
    ! Variables
    integer t

    ! grid2tx
    if (allocated(grid2tx)) then
       call chmdealloc('colfft.src','deallocate_q_grid','grid2tx',&
            size(grid2tx),intg=grid2tx)
    endif

    ! grid2ty
    if (allocated(grid2ty)) then
       call chmdealloc('colfft.src','deallocate_q_grid','grid2ty',&
            size(grid2ty),intg=grid2ty)
    endif

    ! grid2tz
    if (allocated(grid2tz)) then
       call chmdealloc('colfft.src','deallocate_q_grid','grid2tz',&
            size(grid2tz),intg=grid2tz)
    endif

    ! q_grid_loc
    call dealloc_q_grid_local(q_grid_loc, 0, size(q_grid_loc)-1)

    ! q_grid_glo
    call dealloc_q_grid_array(q_grid_glo(0))

    return
  end subroutine deallocate_q_grid

end module colfft
