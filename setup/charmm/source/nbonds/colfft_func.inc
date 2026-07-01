#define PASTE(a) a
#define ADD_PREC(name) PASTE(name)PASTE(_)COLFFT_PREC

! this code relies on the C Preprocessor
! # define COLFFT_PREC to be either
! PS => SINGLE and set SINGLEP <= 1
! PD => DOUBLE and set DOUBLEP <= 1

!  ##EXP P0 sp dp .when. EXPAND (exp_precision)
!  ##PASS1 PS sp
!  ##PASS2 PD dp
!  ##EXEND

subroutine ADD_PREC(grad_sumrc)(n_grid_atom, grid_atom,  &
     charge, recip_in, &
     theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, &
     dx, dy, dz, fr1, fr2, fr3,  &
     forder, nfft1, nfft2, nfft3, q_spat_len, q_spat, scale_fac)
  use domdec_common,only:divide_thread_work
#if KEY_DOMDEC==1
  use domdec_common,only:q_gpu, simd_version, SIMD_NONE
#endif 
  use colfft_util,only:get_spatial_limits, get_halo_limits, get_spatial_sizes, &
       YZ_X_PARTITION
  ! MSLDPME ->
#if KEY_BLOCK==1
  use lambdam,only:iqldm_pme
  use colfft_kernel, only: &
       ADD_PREC(gather_force_kernel_fortran), &
       ADD_PREC(gather_force_kernel_block_fortran)
#else
  ! <- MSLDPME
  use colfft_kernel,only: ADD_PREC(gather_force_kernel_fortran)
  ! MSLDPME ->
#endif
  ! <- MSLDPME
#if KEY_DOMDEC_GPU==1
  use domdec_util_gpu_mod,only:range_start, range_stop  
#endif
  use parallel,only:mynod
  implicit none
  ! Input / Output
  integer, intent(in) :: n_grid_atom, grid_atom(:)
  integer,intent(in) :: forder,nfft1,nfft2,nfft3
  integer, intent(in) :: q_spat_len
#if DOUBLEP == 1
  real(chm_real),intent(inout) :: q_spat(0:q_spat_len-1)
  real(chm_real),intent(in),dimension(:,:) ::  &
       theta1,theta2,theta3,dtheta1,dtheta2,dtheta3
  real(chm_real), intent(in) :: charge(:)
  real(chm_real), intent(in) :: recip_in(3,3)
  real(chm_real), intent(in) :: scale_fac
#else /**/
  real(chm_real4),intent(inout) :: q_spat(0:q_spat_len-1)
  real(chm_real4),intent(in),dimension(:,:) ::  &
       theta1,theta2,theta3,dtheta1,dtheta2,dtheta3
  real(chm_real4), intent(in) :: charge(:)
  real(chm_real4), intent(in) :: recip_in(3,3)
  real(chm_real4), intent(in) :: scale_fac
#endif 
  integer, intent(in) :: fr1(:), fr2(:), fr3(:)
  real(chm_real),intent(inout),dimension(:):: dx,dy,dz
  ! Parameters
#if DOUBLEP == 1
  real(chm_real), parameter :: zero = 0.0_chm_real
#else /**/
  real(chm_real4), parameter :: zero = 0.0_chm_real4
#endif 
  ! Variables
#if DOUBLEP == 1
  real(chm_real) dnfft1, dnfft2, dnfft3, recip(3,3)
#else /**/
  real(chm_real4) dnfft1, dnfft2, dnfft3, recip(3,3)
#endif 
  integer xgridmin,xgridmax,ygridmin,ygridmax,zgridmin,zgridmax
  integer xhalomin,xhalomax,yhalomin,yhalomax,zhalomin,zhalomax
  integer xgridsize, ygridsize, zgridsize
  integer istart, iend

  dnfft1 = nfft1
  dnfft2 = nfft2
  dnfft3 = nfft3

  dnfft1 = dnfft1*scale_fac
  dnfft2 = dnfft2*scale_fac
  dnfft3 = dnfft3*scale_fac

  recip(1,1:3) = recip_in(1,1:3)*dnfft1
  recip(2,1:3) = recip_in(2,1:3)*dnfft2
  recip(3,1:3) = recip_in(3,1:3)*dnfft3

  call get_spatial_limits(YZ_X_PARTITION, &
       xgridmin,xgridmax,ygridmin,ygridmax,  &
       zgridmin,zgridmax,mynod)

  call get_halo_limits(YZ_X_PARTITION, &
       xhalomin,xhalomax,yhalomin,yhalomax,  &
       zhalomin,zhalomax,mynod)

  call get_spatial_sizes(YZ_X_PARTITION,xgridsize,ygridsize,zgridsize,mynod)

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('replicate_grid_borders')
#endif 

  call ADD_PREC(replicate_grid_borders)(q_spat_len, q_spat, xgridmin, xgridmax, ygridmin, ygridmax, &
       zgridmin, zgridmax, xhalomin, yhalomin, zhalomin, xgridsize, ygridsize, &
       forder, nfft2, nfft3)

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()
#endif 

#if SINGLEP==1
  if (forder == 4 &
#if KEY_DOMDEC==1
       .and. simd_version /= SIMD_NONE &  
#endif
       ) then
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('gather_force_kernel4u_sse')
#endif 
     !$omp parallel private(istart, iend)
     call divide_thread_work(n_grid_atom, istart, iend)
     call gather_force_kernel4u_sse(istart, iend, grid_atom(1), &
          fr1(1), fr2(1), fr3(1), &
          yhalomin, zhalomin, &
          xgridsize, ygridsize, &
          recip(1,1), charge(1), &
          theta1(1,1), theta2(1,1), theta3(1,1), &
          dtheta1(1,1), dtheta2(1,1), dtheta3(1,1), &
          q_spat(0), dx(1), dy(1), dz(1))
     !$omp end parallel
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif 
  else
#endif 
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('ADD_PREC(gather_force_kernel_fortran)')
#endif 
     !$omp parallel private(istart, iend)
     call divide_thread_work(n_grid_atom, istart, iend)
     ! MSLDPME ->
#if KEY_BLOCK==1
     if (iqldm_pme>0) then
        call ADD_PREC(gather_force_kernel_block_fortran)(forder, istart, iend, grid_atom, &
             fr1, fr2, fr3, &
             yhalomin, zhalomin, &
             xgridsize, ygridsize, &
             recip, &
             charge, &
             theta1, theta2, theta3, &
             dtheta1, dtheta2, dtheta3, &
             q_spat_len, q_spat, &
             dx, dy, dz)
     else
#endif
        ! <- MSLDPME
        call ADD_PREC(gather_force_kernel_fortran)(forder, istart, iend, grid_atom, &
             fr1, fr2, fr3, &
             yhalomin, zhalomin, &
             xgridsize, ygridsize, &
             recip, &
             charge, &
             theta1, theta2, theta3, &
             dtheta1, dtheta2, dtheta3, &
             q_spat_len, q_spat, &
             dx, dy, dz)
        ! MSLDPME ->
#if KEY_BLOCK==1
     endif
#endif
     ! <- MSLDPME
     !$omp end parallel
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif 
#if SINGLEP==1
  endif  
#endif

  return
end subroutine ADD_PREC(grad_sumrc)

! *
! * Replicates border (of width forder-1) contributions to outside the grid
! * 
! * Right borders:
! * Replicated for q_ydir/q_zdir = .true.
! * Set to zero for q_ydir/q_zdir = .false.
! *
! * Left borders:
! * All set to zero
! *
subroutine ADD_PREC(replicate_grid_borders)(qlen, q, xgridmin, xgridmax, ygridmin, ygridmax, &
     zgridmin, zgridmax, xhalomin, yhalomin, zhalomin, xgridsize, ygridsize, &
     forder, nfft2, nfft3)
  use parallel,only:mynod
  implicit none
  ! Input / Output
  integer, intent(in) :: qlen
#if DOUBLEP == 1
  real(chm_real), intent(inout) :: q(0:qlen-1)
#elif SINGLEP == 1
  real(chm_real4), intent(inout) :: q(0:qlen-1)
#endif 
  integer, intent(in) :: xgridmin, xgridmax, ygridmin, ygridmax, zgridmin, zgridmax
  integer, intent(in) :: xhalomin, yhalomin, zhalomin
  integer, intent(in) :: xgridsize, ygridsize
  integer, intent(in) :: forder, nfft2, nfft3
  ! Parameters
#if DOUBLEP == 1
  real(chm_real), parameter :: zero = 0.0_chm_real
#elif SINGLEP == 1
  real(chm_real4), parameter :: zero = 0.0_chm_real4
#endif 
  ! Variables
  integer xygridsize
  integer xdim, ydim, zdim
  integer xhalo, yhalo, zhalo
  integer i, j, k
  integer ind, ind0
  logical q_ydir, q_zdir

  xygridsize = xgridsize*ygridsize

  q_ydir = .false.
  q_zdir = .false.
  if (ygridmin == 0 .and. ygridmax == nfft2-1) q_ydir = .true.
  if (zgridmin == 0 .and. zgridmax == nfft3-1) q_zdir = .true.
  ! ind = index in the border region
  ! ind0 = index in the grid region

  xdim = xgridmax-xgridmin+1
  ydim = ygridmax-ygridmin+1
  zdim = zgridmax-zgridmin+1

  xhalo = xgridmin-xhalomin
  yhalo = ygridmin-yhalomin
  zhalo = zgridmin-zhalomin

  !$omp parallel private(k, j, i, ind, ind0)

  ! xz plate
  if (q_ydir) then
     ! Replicate
     !$omp do schedule(static)
     do k=zhalo,zhalo + zgridmax-zgridmin
        do j=yhalo,yhalo + forder-2
           do i=xhalo,xhalo + xgridmax-xgridmin
              ind  = i + (j+ydim)*xgridsize + k*xygridsize
              ind0 = i +        j*xgridsize + k*xygridsize
              q(ind) = q(ind0)
           enddo
        enddo
     enddo
     !$omp end do
  else
     ! Set to zero
     ! Right border
     !$omp do schedule(static)
     do k=zhalo,zhalo + zgridmax-zgridmin
        do j=yhalo,yhalo + forder-2
           do i=xhalo,xhalo + xgridmax-xgridmin
              ind  = i + (j+ydim)*xgridsize + k*xygridsize
              q(ind) = zero
           enddo
        enddo
     enddo
     !$omp end do
     ! Left border
     !$omp do schedule(static)
     do k=zhalo,zhalo + zgridmax-zgridmin
        do j=yhalo - (forder-1),yhalo - 1
           do i=xhalo,xhalo + xgridmax-xgridmin
              ind  = i + j*xgridsize + k*xygridsize
              q(ind) = zero
           enddo
        enddo
     enddo
     !$omp end do
  endif

  ! xy plate
  if (q_zdir) then
     ! Replicate
     !$omp do schedule(static)
     do k=zhalo,zhalo + forder-2
        do j=yhalo,yhalo + ygridmax-ygridmin
           do i=xhalo,xhalo + xgridmax-xgridmin
              ind  = i + j*xgridsize + (k+zdim)*xygridsize
              ind0 = i + j*xgridsize +        k*xygridsize
              q(ind) = q(ind0)
           enddo
        enddo
     enddo
     !$omp end do
  else
     ! Set to zero
     ! Right border
     !$omp do schedule(static)
     do k=zhalo,zhalo + forder-2
        do j=yhalo,yhalo + ygridmax-ygridmin
           do i=xhalo,xhalo + xgridmax-xgridmin
              ind  = i + j*xgridsize + (k+zdim)*xygridsize
              q(ind) = zero
           enddo
        enddo
     enddo
     !$omp end do
     ! Left border
     !$omp do schedule(static)
     do k=zhalo - (forder-1),zhalo - 1
        do j=yhalo,yhalo + ygridmax-ygridmin
           do i=xhalo,xhalo + xgridmax-xgridmin
              ind  = i + j*xgridsize + k*xygridsize
              q(ind) = zero
           enddo
        enddo
     enddo
     !$omp end do
  endif

  ! Replicate
  ! yz plate
  !$omp do schedule(static)
  do k=zhalo,zhalo + zgridmax-zgridmin
     do j=yhalo,yhalo + ygridmax-ygridmin
        do i=xhalo,xhalo + forder-2
           ind  = i+xdim + j*xgridsize + k*xygridsize
           ind0 = i      + j*xgridsize + k*xygridsize
           q(ind) = q(ind0)
        enddo
     enddo
  enddo
  !$omp end do

  ! yz column
  if (q_ydir .and. q_zdir) then
     ! Replicate
     !$omp do schedule(static)
     do k=zhalo,zhalo + forder-2
        do j=yhalo,yhalo + forder-2
           do i=xhalo,xhalo + xgridmax-xgridmin
              ind  = i + (j+ydim)*xgridsize + (k+zdim)*xygridsize
              ind0 = i +        j*xgridsize +        k*xygridsize
              q(ind) = q(ind0)
           enddo
        enddo
     enddo
     !$omp end do
  else
     ! Set to zero
     ! Right border
     !$omp do schedule(static)
     do k=zhalo,zhalo + forder-2
        do j=yhalo,yhalo + forder-2
           do i=xhalo,xhalo + xgridmax-xgridmin
              ind  = i + (j+ydim)*xgridsize + (k+zdim)*xygridsize
              q(ind) = zero
           enddo
        enddo
     enddo
     !$omp end do
     if (.not.q_zdir) then
        ! Left border
        !$omp do schedule(static)
        do k=zhalo - (forder-1),zhalo - 1
           do j=yhalo,yhalo + forder-2
              do i=xhalo,xhalo + xgridmax-xgridmin
                 ind  = i + (j+ydim)*xgridsize + k*xygridsize
                 q(ind) = zero
              enddo
           enddo
        enddo
        !$omp end do
     endif
     if (.not.q_ydir) then
        ! Left border
        !$omp do schedule(static)
        do k=zhalo,zhalo + forder-2
           do j=yhalo - (forder-1),yhalo - 1
              do i=xhalo,xhalo + xgridmax-xgridmin
                 ind  = i + j*xgridsize + (k+zdim)*xygridsize
                 q(ind) = zero
              enddo
           enddo
        enddo
        !$omp end do
     endif
     if (.not.q_zdir .and. .not.q_ydir) then
        ! Left border
        !$omp do schedule(static)
        do k=zhalo - (forder-1),zhalo - 1
           do j=yhalo - (forder-1),yhalo - 1
              do i=xhalo,xhalo + xgridmax-xgridmin
                 ind  = i + j*xgridsize + k*xygridsize
                 q(ind) = zero
              enddo
           enddo
        enddo
        !$omp end do
     endif
  endif

  ! xy column
  if (q_ydir) then
     ! Replicate
     !$omp do schedule(static)
     do k=zhalo,zhalo + zgridmax-zgridmin
        do j=yhalo,yhalo + forder-2
           do i=xhalo,xhalo + forder-2
              ind  = i+xdim + (j+ydim)*xgridsize + k*xygridsize
              ind0 = i      +        j*xgridsize + k*xygridsize
              q(ind) = q(ind0)
           enddo
        enddo
     enddo
     !$omp end do
  else
     ! Set to zero
     ! Right border
     !$omp do schedule(static)
     do k=zhalo,zhalo + zgridmax-zgridmin
        do j=yhalo,yhalo + forder-2
           do i=xhalo,xhalo + forder-2
              ind  = i+xdim + (j+ydim)*xgridsize + k*xygridsize
              q(ind) = zero
           enddo
        enddo
     enddo
     !$omp end do
     ! Left border
     !$omp do schedule(static)
     do k=zhalo,zhalo + zgridmax-zgridmin
        do j=yhalo - (forder-1),yhalo - 1
           do i=xhalo,xhalo + forder-2
              ind  = i+xdim + j*xgridsize + k*xygridsize
              q(ind) = zero
           enddo
        enddo
     enddo
     !$omp end do
  endif

  ! xz column
  if (q_zdir) then
     ! Replicate
     !$omp do schedule(static)
     do k=zhalo,zhalo + forder-2
        do j=yhalo,yhalo + ygridmax-ygridmin
           do i=xhalo,xhalo + forder-2
              ind  = i+xdim + j*xgridsize + (k+zdim)*xygridsize
              ind0 = i      + j*xgridsize +        k*xygridsize
              q(ind) = q(ind0)
           enddo
        enddo
     enddo
     !$omp end do
  else
     ! Set to zero
     ! Right border
     !$omp do schedule(static)
     do k=zhalo,zhalo + forder-2
        do j=yhalo,yhalo + ygridmax-ygridmin
           do i=xhalo,xhalo + forder-2
              ind  = i+xdim + j*xgridsize + (k+zdim)*xygridsize
              q(ind) = zero
           enddo
        enddo
     enddo
     !$omp end do
     ! Left border
     !$omp do schedule(static)
     do k=zhalo - (forder-1),zhalo - 1
        do j=yhalo,yhalo + ygridmax-ygridmin
           do i=xhalo,xhalo + forder-2
              ind  = i+xdim + j*xgridsize + k*xygridsize
              q(ind) = zero
           enddo
        enddo
     enddo
     !$omp end do
  endif

  ! xyz corner
  if (q_ydir .and. q_zdir) then
     ! Replicate
     !$omp do schedule(static)
     do k=zhalo,zhalo + forder-2
        do j=yhalo,yhalo + forder-2
           do i=xhalo,xhalo + forder-2
              ind  = i+xdim + (j+ydim)*xgridsize + (k+zdim)*xygridsize
              ind0 = i      +        j*xgridsize +        k*xygridsize
              q(ind) = q(ind0)
           enddo
        enddo
     enddo
  else
     ! Set zero
     ! Right border
     !$omp do schedule(static)
     do k=zhalo,zhalo + forder-2
        do j=yhalo,yhalo + forder-2
           do i=xhalo,xhalo + forder-2
              ind  = i+xdim + (j+ydim)*xgridsize + (k+zdim)*xygridsize
              q(ind) = zero
           enddo
        enddo
     enddo
     !$omp end do
     ! Left border
     if (.not.q_zdir) then
        !$omp do schedule(static)
        do k=zhalo - (forder-1),zhalo - 1
           do j=yhalo,yhalo + forder-2
              do i=xhalo,xhalo + forder-2
                 ind  = i+xdim + (j+ydim)*xgridsize + k*xygridsize
                 q(ind) = zero
              enddo
           enddo
        enddo
        !$omp end do
     endif
     if (.not.q_ydir) then
        !$omp do schedule(static)
        do k=zhalo,zhalo + forder-2
           do j=yhalo - (forder-1),yhalo - 1
              do i=xhalo,xhalo + forder-2
                 ind  = i+xdim + j*xgridsize + (k+zdim)*xygridsize
                 q(ind) = zero
              enddo
           enddo
        enddo
        !$omp end do
     endif
     if (.not.q_zdir .and. .not.q_ydir) then
        !$omp do schedule(static)
        do k=zhalo - (forder-1),zhalo - 1
           do j=yhalo - (forder-1),yhalo - 1
              do i=xhalo,xhalo + forder-2
                 ind  = i+xdim + j*xgridsize + k*xygridsize
                 q(ind) = zero
              enddo
           enddo
        enddo
        !$omp end do
     endif
  endif

  !$omp end parallel

  return
end subroutine ADD_PREC(replicate_grid_borders)

subroutine ADD_PREC(scalar_sumrc_lj)(size_q, q, kappa, volume, recip, &
     prefac1, prefac2, prefac3,  &
     nfft1, nfft2, nfft3, &
     eer, virial, q_calc_energy_virial)
  use colfft_util,only:get_fft_limits,get_fft_sizes,XY_Z_PARTITION
  use parallel,only:mynod
  use consta,only:pi
  ! Input / Output
  integer,intent(in) :: size_q, nfft1, nfft2, nfft3
#if DOUBLEP == 1
  real(chm_real), intent(in) :: recip(3,3)
  real(chm_real),intent(inout) :: q(0:size_q-1)
  real(chm_real),intent(in) :: prefac1(nfft1),prefac2(nfft2),prefac3(nfft3)
#elif SINGLEP == 1
  real(chm_real4), intent(in) :: recip(3,3)
  real(chm_real4),intent(inout) :: q(0:size_q-1)
  real(chm_real4),intent(in) :: prefac1(nfft1),prefac2(nfft2),prefac3(nfft3)
#endif 
  real(chm_real),intent(in) :: kappa,volume
  real(chm_real),intent(out) :: eer,virial(6)
  logical, intent(in) :: q_calc_energy_virial
  ! Parameters
#if DOUBLEP == 1
  real(chm_real), parameter :: zero = 0.0_chm_real, half = 0.5_chm_real
#elif SINGLEP == 1
  real(chm_real4), parameter :: zero = 0.0_chm_real4, half = 0.5_chm_real4
#endif 
  ! Variables
  real(chm_real), save :: energy
#if DOUBLEP == 1
  real(chm_real) fac, piv_inv, bfac, fac1, fac2, fac3, p, ps, &
         prefac12s_piv_inv, prefac12_piv_inv, tmp1, tmp1s, &
         mhat1s, mhat2s, mhat3s, &
         mhat1, mhat2, mhat3, &
         expterm, expterms, erfcterm, erfcterms, &
         expfac, expfacs, b, bs, &
         eterm, vterm, eterms, vterms, struc2, m, mmm, msqs, ms, mmms, msq
#elif SINGLEP == 1
  real(chm_real4) fac, piv_inv, bfac, fac1, fac2, fac3, p, ps, &
         prefac12s_piv_inv, prefac12_piv_inv, tmp1, tmp1s, &
         mhat1s, mhat2s, mhat3s, &
         mhat1, mhat2, mhat3, &
         expterm, expterms, erfcterm, erfcterms, &
         expfac, expfacs, b, bs, &
         eterm, vterm, eterms, vterms, struc2, m, mmm, msqs, ms, mmms, msq
#endif 
  integer nf1, nf2, nf3, nff
  integer xgridmin, xgridmax, ygridmin, ygridmax, zgridmin, zgridmax
  integer xgridsize, ygridsize, zgridsize, xzgridsize
  integer k3_start,k3_stop,k1s,k2s,k3s,m1s,m2s,m3s
  integer k2q,k1, k2, k3, m1, m2, m3
  integer index, index2

  piv_inv = pi**(1.5d0) / (6d0*volume)

  fac = pi*pi/(kappa*kappa)
  nff = nfft1*nfft2
  nf1 = nfft1/2 + mod(nfft1,2)
  nf2 = nfft2/2 + mod(nfft2,2)
  nf3 = nfft3/2 + mod(nfft3,2)

  energy = zero
  virial = zero

  ! prefactors
  bfac = PI / kappa
  fac1 = 2d0*PI**(3.5d0)
  fac2 = kappa**3
  fac3 = -2d0*kappa*PI*PI

  call get_fft_limits(XY_Z_PARTITION,zgridmin,zgridmax,xgridmin,xgridmax, &
       ygridmin,ygridmax,mynod)

  call get_fft_sizes(XY_Z_PARTITION,zgridsize,xgridsize,ygridsize,mynod)
  xzgridsize = xgridsize*zgridsize

!$omp do schedule(static) reduction(+:energy, virial)
    do k2q = 1, ygridmax-ygridmin+1
       k2 = k2q + ygridmin
       k2s = mod(nfft2-k2+1,nfft2) + 1
       m2 = k2 - 1
       if (k2 > nf2) m2 = m2 - nfft2
       m2s = k2s - 1
       if (k2s > nf2) m2s = m2s - nfft2

       index2 = (k2q-1)*xzgridsize

       ! Loop over x
       do k1 = xgridmin+1, xgridmax+1
          k1s = nfft1 - k1 + 2
          m1 = k1 - 1
          if ( k1 > nf1 ) m1 = m1 - nfft1
          m1s = k1s - 1
          if ( k1s > nf1 ) m1s = m1s - nfft1

          prefac12_piv_inv = piv_inv*prefac2(k2)*prefac1(k1)

          ! We don't want to skip the k==0 term for DPME
          k3_start = zgridmin+1

          index = 2*(k3_start-1 + (k1-xgridmin-1)*zgridsize + index2)

          ! Loop over z
          if(k1 <= 1 .or. k1 > nfft1)then
             ! k3 <= nf3
             do k3 = k3_start,nf3
                m3 = k3 - 1

                mhat1 = recip(1,1)*m1+recip(1,2)*m2+recip(1,3)*m3
                mhat2 = recip(2,1)*m1+recip(2,2)*m2+recip(2,3)*m3
                mhat3 = recip(3,1)*m1+recip(3,2)*m2+recip(3,3)*m3

                msq = mhat1*mhat1 + mhat2*mhat2 + mhat3*mhat3
                m = sqrt(msq)
                b = m * bfac
                expfac = -b*b
                mmm = m * msq
                expterm = exp(expfac)
                erfcterm = erfc(b)

                struc2 = q(index)**2 + q(index+1)**2

                p = prefac12_piv_inv*prefac3(k3)
                eterm = p*(fac1*erfcterm*mmm + expterm*(fac2 + fac3*msq))
                vterm = p*struc2*(3.0d0*(fac1*erfcterm*m + fac3*expterm))
                tmp1 = struc2*eterm
                energy = energy - tmp1
                q(index:index+1) = -eterm*q(index:index+1)
                index = index + 2

                virial(1) = virial(1) - (vterm*mhat1*mhat1 + tmp1)
                virial(2) = virial(2) - (vterm*mhat1*mhat2)
                virial(3) = virial(3) - (vterm*mhat1*mhat3)
                virial(4) = virial(4) - (vterm*mhat2*mhat2 + tmp1)
                virial(5) = virial(5) - (vterm*mhat2*mhat3)
                virial(6) = virial(6) - (vterm*mhat3*mhat3 + tmp1)
             enddo
             ! k3 > nf3
             do k3 = nf3+1,zgridmax+1
                m3 = k3 - 1 - nfft3

                mhat1 = recip(1,1)*m1+recip(1,2)*m2+recip(1,3)*m3
                mhat2 = recip(2,1)*m1+recip(2,2)*m2+recip(2,3)*m3
                mhat3 = recip(3,1)*m1+recip(3,2)*m2+recip(3,3)*m3

                msq = mhat1*mhat1 + mhat2*mhat2 + mhat3*mhat3
                m = sqrt(msq)
                b = m * bfac
                expfac = -b*b
                mmm = m * msq
                expterm = exp(expfac)
                erfcterm = erfc(b)

                struc2 = q(index)**2 + q(index+1)**2

                p = prefac12_piv_inv*prefac3(k3)
                eterm = p*(fac1*erfcterm*mmm + expterm*(fac2 + fac3*msq))
                vterm = p*struc2*(3.0d0*(fac1*erfcterm*m + fac3*expterm))
                tmp1 = struc2*eterm
                energy = energy - tmp1
                q(index:index+1) = -eterm*q(index:index+1)
                index = index + 2

                virial(1) = virial(1) - (vterm*mhat1*mhat1 + tmp1)
                virial(2) = virial(2) - (vterm*mhat1*mhat2)
                virial(3) = virial(3) - (vterm*mhat1*mhat3)
                virial(4) = virial(4) - (vterm*mhat2*mhat2 + tmp1)
                virial(5) = virial(5) - (vterm*mhat2*mhat3)
                virial(6) = virial(6) - (vterm*mhat3*mhat3 + tmp1)
             enddo
          else
             prefac12s_piv_inv = piv_inv*prefac2(k2s)*prefac1(k1s)
             ! k3 <= nf3
             do k3 = k3_start,nf3
                m3 = k3 - 1
                mhat1 = recip(1,1)*m1+recip(1,2)*m2+recip(1,3)*m3
                mhat2 = recip(2,1)*m1+recip(2,2)*m2+recip(2,3)*m3
                mhat3 = recip(3,1)*m1+recip(3,2)*m2+recip(3,3)*m3

                k3s = mod(nfft3-k3+1,nfft3)+1
                m3s = k3s - 1
                if (k3s > nf3) m3s = k3s - 1 - nfft3

                mhat1s = recip(1,1)*m1s+recip(1,2)*m2s+recip(1,3)*m3s
                mhat2s = recip(2,1)*m1s+recip(2,2)*m2s+recip(2,3)*m3s
                mhat3s = recip(3,1)*m1s+recip(3,2)*m2s+recip(3,3)*m3s

                msqs = mhat1s*mhat1s + mhat2s*mhat2s + mhat3s*mhat3s
                msq  = mhat1*mhat1   + mhat2*mhat2   + mhat3*mhat3
                m = sqrt(msq)
                ms = sqrt(msqs)
                b = m * bfac
                bs = ms * bfac
                expfac = -b*b
                expfacs = -bs*bs
                mmm = m * msq
                mmms = ms * msqs
                expterm = exp(expfac)
                expterms = exp(expfacs)
                erfcterm = erfc(b)
                erfcterms = erfc(bs)

                struc2 = q(index)**2 + q(index+1)**2

                p = prefac12_piv_inv*prefac3(k3)
                ps = prefac12s_piv_inv*prefac3(k3s)
                eterm = p*(fac1*erfcterm*mmm + expterm*(fac2 + fac3*msq))
                eterms = ps*(fac1*erfcterms*mmms + expterms*(fac2 + fac3*msqs))
                vterm = p*struc2*(3.0d0*(fac1*erfcterm*m + fac3*expterm))
                vterms = ps*struc2*(3.0d0*(fac1*erfcterms*ms + fac3*expterms))
                tmp1 = struc2*eterm
                tmp1s = struc2*eterms
                energy = energy - tmp1 - tmp1s
                q(index:index+1) = -eterm*q(index:index+1)
                index = index + 2

                virial(1) = virial(1) - (vterm*mhat1*mhat1 + tmp1) &
                                      - (vterms*mhat1s*mhat1s + tmp1s)
                virial(2) = virial(2) - (vterm*mhat1*mhat2) &
                                      - (vterms*mhat1s*mhat2s)
                virial(3) = virial(3) - (vterm*mhat1*mhat3) &
                                      - (vterms*mhat1s*mhat3s)
                virial(4) = virial(4) - (vterm*mhat2*mhat2 + tmp1) &
                                      - (vterms*mhat2s*mhat2s + tmp1s)
                virial(5) = virial(5) - (vterm*mhat2*mhat3) &
                                      - (vterms*mhat2s*mhat3s)
                virial(6) = virial(6) - (vterm*mhat3*mhat3 + tmp1) &
                                      - (vterms*mhat3s*mhat3s + tmp1s)
             enddo
             ! k3 > nf3
             do k3 = nf3+1,zgridmax+1
                m3 = k3 - 1 - nfft3
                mhat1 = recip(1,1)*m1+recip(1,2)*m2+recip(1,3)*m3
                mhat2 = recip(2,1)*m1+recip(2,2)*m2+recip(2,3)*m3
                mhat3 = recip(3,1)*m1+recip(3,2)*m2+recip(3,3)*m3

                k3s = mod(nfft3-k3+1,nfft3)+1
                m3s = k3s - 1
                if (k3s > nf3) m3s = k3s - 1 - nfft3

                mhat1s = recip(1,1)*m1s+recip(1,2)*m2s+recip(1,3)*m3s
                mhat2s = recip(2,1)*m1s+recip(2,2)*m2s+recip(2,3)*m3s
                mhat3s = recip(3,1)*m1s+recip(3,2)*m2s+recip(3,3)*m3s

                msqs = mhat1s*mhat1s + mhat2s*mhat2s + mhat3s*mhat3s
                msq  = mhat1*mhat1   + mhat2*mhat2   + mhat3*mhat3
                m = sqrt(msq)
                ms = sqrt(msqs)
                b = m * bfac
                bs = ms * bfac
                expfac = -b*b
                expfacs = -bs*bs
                mmm = m * msq
                mmms = ms * msqs
                expterm = exp(expfac)
                expterms = exp(expfacs)
                erfcterm = erfc(b)
                erfcterms = erfc(bs)

                struc2 = q(index)**2 + q(index+1)**2

                p = prefac12_piv_inv*prefac3(k3)
                ps = prefac12s_piv_inv*prefac3(k3s)
                eterm = p*(fac1*erfcterm*mmm + expterm*(fac2 + fac3*msq))
                eterms = ps*(fac1*erfcterms*mmms + expterms*(fac2 + fac3*msqs))
                vterm = p*struc2*(3.0d0*(fac1*erfcterm*m + fac3*expterm))
                vterms = ps*struc2*(3.0d0*(fac1*erfcterms*ms + fac3*expterms))
                tmp1 = struc2*eterm
                tmp1s = struc2*eterms
                energy = energy - tmp1 - tmp1s
                q(index:index+1) = -eterm*q(index:index+1)
                index = index + 2

                virial(1) = virial(1) - (vterm*mhat1*mhat1 + tmp1) &
                                      - (vterms*mhat1s*mhat1s + tmp1s)
                virial(2) = virial(2) - (vterm*mhat1*mhat2) &
                                      - (vterms*mhat1s*mhat2s)
                virial(3) = virial(3) - (vterm*mhat1*mhat3) &
                                      - (vterms*mhat1s*mhat3s)
                virial(4) = virial(4) - (vterm*mhat2*mhat2 + tmp1) &
                                      - (vterms*mhat2s*mhat2s + tmp1s)
                virial(5) = virial(5) - (vterm*mhat2*mhat3) &
                                      - (vterms*mhat2s*mhat3s)
                virial(6) = virial(6) - (vterm*mhat3*mhat3 + tmp1) &
                                      - (vterms*mhat3s*mhat3s + tmp1s)
             enddo
          endif
       enddo
    enddo
!$omp end do

  if (q_calc_energy_virial) then
     last_energy = energy
     last_virial = virial
  endif

  eer = last_energy
  virial = last_virial

  return
end subroutine ADD_PREC(scalar_sumrc_lj)



subroutine ADD_PREC(scalar_sumrc)(size_q, q, kappa, volume, recip, &
     prefac1, prefac2, prefac3,  &
     m1_tbl, m2_tbl, m3_tbl, &
     nfft1, nfft2, nfft3, &
     eer, virial, lorthog, q_calc_energy_virial)
  use colfft_util,only:get_fft_limits,get_fft_sizes,XY_Z_PARTITION
  use colfft_kernel,only: &
       ADD_PREC(scalar_sumrc_kernel_o0), &
       ADD_PREC(scalar_sumrc_kernel_o1)
  use parallel,only:mynod
  use consta,only:pi
  ! Input / Output
  integer,intent(in) :: size_q, nfft1, nfft2, nfft3
#if DOUBLEP == 1
  real(chm_real), intent(in) :: recip(3,3)
  real(chm_real),intent(inout) :: q(0:size_q-1)
  real(chm_real),intent(in) :: prefac1(nfft1),prefac2(nfft2),prefac3(nfft3)
  real(chm_real),intent(out) :: m1_tbl(-(nfft1/2+1):(nfft1/2+1))
  real(chm_real),intent(out) :: m2_tbl(-(nfft2/2+1):(nfft2/2+1))
  real(chm_real),intent(out) :: m3_tbl(-(nfft3/2+1):(nfft3/2+1))
#elif SINGLEP == 1
  real(chm_real4), intent(in) :: recip(3,3)
  real(chm_real4),intent(inout) :: q(0:size_q-1)
  real(chm_real4),intent(in) :: prefac1(nfft1),prefac2(nfft2),prefac3(nfft3)
  real(chm_real4),intent(out) :: m1_tbl(-(nfft1/2+1):(nfft1/2+1))
  real(chm_real4),intent(out) :: m2_tbl(-(nfft2/2+1):(nfft2/2+1))
  real(chm_real4),intent(out) :: m3_tbl(-(nfft3/2+1):(nfft3/2+1))
#endif 
  real(chm_real),intent(in) :: kappa,volume
  real(chm_real),intent(out) :: eer,virial(6)
  logical, intent(in) :: lorthog, q_calc_energy_virial
  ! Parameters
#if DOUBLEP == 1
  real(chm_real), parameter :: zero = 0.0_chm_real, half = 0.5_chm_real
#elif SINGLEP == 1
  real(chm_real4), parameter :: zero = 0.0_chm_real4, half = 0.5_chm_real4
#endif 
  ! Variables
  real(chm_real) energy
#if DOUBLEP == 1
  real(chm_real) fac, piv_inv
#elif SINGLEP == 1
  real(chm_real4) fac, piv_inv
#endif 
  integer nf1, nf2, nf3, nff
  integer xgridmin, xgridmax, ygridmin, ygridmax, zgridmin, zgridmax
  integer xgridsize, ygridsize, zgridsize, xzgridsize

  piv_inv = 1.d0/(pi*volume)
  fac = pi*pi/(kappa*kappa)
  nff = nfft1*nfft2
  nf1 = nfft1/2 + mod(nfft1,2)
  nf2 = nfft2/2 + mod(nfft2,2)
  nf3 = nfft3/2 + mod(nfft3,2)

  if (lorthog) then
     call ADD_PREC(calc_m_tbl)(recip(1,1), fac, nfft1, m1_tbl)
     call ADD_PREC(calc_m_tbl)(recip(2,2), fac, nfft2, m2_tbl)
     call ADD_PREC(calc_m_tbl)(recip(3,3), fac, nfft3, m3_tbl)
  endif

  energy = zero

  !........Insist that Q(1,1,1) is set to 0 (true already for neutral)

#if KEY_PARALLEL==1
  if (mynod == 0) then  
#endif
     q(0:1) = zero
#if KEY_PARALLEL==1
  endif  
#endif

  call get_fft_limits(XY_Z_PARTITION,zgridmin,zgridmax,xgridmin,xgridmax, &
       ygridmin,ygridmax,mynod)

  call get_fft_sizes(XY_Z_PARTITION,zgridsize,xgridsize,ygridsize,mynod)
  xzgridsize = xgridsize*zgridsize

  if (lorthog) then
     ! Orthogonal box
     !$omp parallel
     call ADD_PREC(scalar_sumrc_kernel_o1)(nfft1, nfft2, nfft3, nf1, nf2, nf3, &
          xgridmin, xgridmax, ygridmin, ygridmax, zgridmin, zgridmax, &
          zgridsize, xzgridsize, &
          fac, piv_inv, prefac1, prefac2, prefac3, recip, &
          m1_tbl, m2_tbl, m3_tbl, &
          size_q, q, energy, virial, q_calc_energy_virial)
     !$omp end parallel
  else
     ! Non-orthogonal box
     !$omp parallel
     call ADD_PREC(scalar_sumrc_kernel_o0)(nfft1, nfft2, nfft3, nf1, nf2, nf3, &
          xgridmin, xgridmax, ygridmin, ygridmax, zgridmin, zgridmax, &
          zgridsize, xzgridsize, &
          fac, piv_inv, prefac1, prefac2, prefac3, recip, &
          size_q, q, energy, virial, q_calc_energy_virial)
     !$omp end parallel
  endif

  if (q_calc_energy_virial) then
     last_energy = half*energy
     last_virial = half*virial
  endif

  eer = last_energy
  virial = last_virial

  return
end subroutine ADD_PREC(scalar_sumrc)

! *
! * Calculates on of the "m_tbl"s: (m1_tbl, m2_tbl, m3_tbl)
! *
subroutine ADD_PREC(calc_m_tbl)(recip, fac, nfft, m_tbl)
  implicit none
  ! Input / Output
#if DOUBLEP == 1
  real(chm_real), intent(in) :: recip, fac
#else /**/
  real(chm_real4), intent(in) :: recip, fac
#endif 
  integer, intent(in) :: nfft
#if DOUBLEP == 1
  real(chm_real), intent(out) :: m_tbl(-(nfft/2+1):(nfft/2+1))
#else /**/
  real(chm_real4), intent(out) :: m_tbl(-(nfft/2+1):(nfft/2+1))
#endif 
  ! Variables
#if DOUBLEP == 1
  real(chm_real) recipsq
#else /**/
  real(chm_real4) recipsq
#endif 
  integer i

  recipsq = recip*recip

  do i = -(nfft/2 + 1), nfft/2 + 1
     m_tbl(i) = exp(-fac*i*i*recipsq)
  end do

  return
end subroutine ADD_PREC(calc_m_tbl)

subroutine ADD_PREC(spread_charge_grid)(n_grid_atom, grid_atom, &
     x, y, z, charge,  &
     theta1, theta2, theta3, &
     recip, lorthog, &
     fr1, fr2, fr3,  &
     forder, nfft1, nfft2, nfft3, &
     nthread, natom_thread, atomlist_thread, q_grid)
  !---------------------------------------------------------------------
  ! INPUT:
  !      charge: the array of atomic charges
  !      theta1,theta2,theta3: the spline coeff arrays
  !      fr1,fr2,fr3 the scaled and shifted fractional coords
  !      nfft1,nfft2,nfft3: the charge grid dimensions
  !      nfftdim1,nfftdim2,nfftdim3: physical charge grid dims
  !      order: the order of spline interpolation
  ! OUTPUT:
  !      Q the charge grid
  !---------------------------------------------------------------------
  use memory
  use exfunc
  use parallel
#if KEY_DOMDEC==1
  use domdec_common,only:q_gpu 
#endif
  use domdec_common,only:simd_version, SIMD_NONE
  ! MSLDPME ->
#if KEY_BLOCK==1
  use lambdam,only:iqldm_pme
  use colfft_kernel,only: &
       ADD_PREC(spread_charge_kernel), &
       ADD_PREC(spread_charge_kernel_block)
#else
  ! <- MSLDPME
  use colfft_kernel,only: ADD_PREC(spread_charge_kernel)
  ! MSLDPME ->
#endif
  ! <- MSLDPME
#if KEY_DOMDEC_GPU==1
  use domdec_util_gpu_mod,only:range_start, range_stop  
#endif
  use colfft_types,only:q_grid_t
  use nblist_types,only:intarray_t
  ! Input / Output
  integer, intent(in) :: n_grid_atom, grid_atom(:)
  integer, intent(in) :: forder, nfft1, nfft2, nfft3
  integer, intent(in),dimension(:) :: fr1, fr2, fr3
  logical, intent(in) :: lorthog
#if DOUBLEP == 1
  real(chm_real),intent(in),dimension(:) :: x, y, z
  real(chm_real),intent(in),dimension(:) :: charge
  real(chm_real),intent(in),dimension(:,:) :: theta1, theta2, theta3
  real(chm_real), intent(in) :: recip(3,3)
#elif SINGLEP == 1
  real(chm_real4),intent(in),dimension(:) :: x, y, z
  real(chm_real4),intent(in),dimension(:) :: charge
  real(chm_real4),intent(in),dimension(:,:) :: theta1, theta2, theta3
  real(chm_real4), intent(in) :: recip(3,3)
#endif 
  integer, intent(in) :: nthread
  type(intarray_t), intent(in) :: natom_thread(0:nthread)
  type(intarray_t), intent(inout) :: atomlist_thread(0:nthread-1)
  type(q_grid_t), pointer, intent(inout) :: q_grid(:)
  ! Parameters
#if DOUBLEP == 1
  real(chm_real), parameter :: zero = 0.0_chm_real
#elif SINGLEP == 1
  real(chm_real4), parameter :: zero = 0.0_chm_real4
#endif 
  ! Functions
#ifdef _OPENMP
  integer omp_get_thread_num
#endif 
  ! Variables
  integer tid

  !$omp parallel private(tid)

#ifdef _OPENMP
  tid = omp_get_thread_num()
#else /**/
  tid = 0
#endif 

  !$omp master
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('Zero q_grid_loc')  
#endif
  !$omp end master    

  !........Zero the Charge grids
#if DOUBLEP == 1
  q_grid(tid)%array_dp(0:q_grid(tid)%tot_size-1) = zero
#else /**/
  q_grid(tid)%array_sp(0:q_grid(tid)%tot_size-1) = zero
#endif 

  !$omp master
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()  
#endif
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('spread_charge_kernel')  
#endif
  !$omp end master

  !$omp barrier
  ! MSLDPME ->
#if KEY_BLOCK==1
  if (iqldm_pme>0) then
     call ADD_PREC(spread_charge_kernel_block)(natom_thread(nthread)%array(tid), &
          atomlist_thread(tid)%array, &
          grid_atom, forder, &
          fr1, fr2, fr3, &
          q_grid(tid)%xlo, q_grid(tid)%ylo, q_grid(tid)%zlo, &
          q_grid(tid)%xsize, q_grid(tid)%xsize*q_grid(tid)%ysize, &
          charge, theta1, theta2, theta3, &
          q_grid(tid)%xsize*q_grid(tid)%ysize*q_grid(tid)%zsize, &
#if DOUBLEP == 1
          q_grid(tid)%array_dp &  
#endif
#if SINGLEP == 1
          q_grid(tid)%array_sp &  
#endif
          )
  else
#endif
     ! <- MSLDPME
#if SINGLEP==1
     if (forder == 4 .and. simd_version /= SIMD_NONE) then
        call spread_charge_kernel4u_sse(natom_thread(nthread)%array(tid), &
             atomlist_thread(tid)%array(1), &
             grid_atom(1), &
             fr1(1), fr2(1), fr3(1), &
             q_grid(tid)%xlo, q_grid(tid)%ylo, q_grid(tid)%zlo, &
             q_grid(tid)%xsize, q_grid(tid)%xsize*q_grid(tid)%ysize, &
             charge(1), theta1(1,1), theta2(1,1), theta3(1,1), &
             q_grid(tid)%array_sp(0))
     else
#endif 
        call ADD_PREC(spread_charge_kernel)(natom_thread(nthread)%array(tid), &
             atomlist_thread(tid)%array, &
             grid_atom, forder, &
             fr1, fr2, fr3, &
             q_grid(tid)%xlo, q_grid(tid)%ylo, q_grid(tid)%zlo, &
             q_grid(tid)%xsize, q_grid(tid)%xsize*q_grid(tid)%ysize, &
             charge, theta1, theta2, theta3, &
             q_grid(tid)%xsize*q_grid(tid)%ysize*q_grid(tid)%zsize, &
#if DOUBLEP == 1
             q_grid(tid)%array_dp &  
#endif
#if SINGLEP == 1
             q_grid(tid)%array_sp &  
#endif
             )
#if SINGLEP==1
     endif  
#endif
     ! MSLDPME ->
#if KEY_BLOCK==1
  endif
#endif
  ! <- MSLDPME

  !$omp end parallel

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()  
#endif

  return
end subroutine ADD_PREC(spread_charge_grid)

! *
! * Combine charge grids
! *
subroutine ADD_PREC(reduce_charge_grid)(nthread, ntx, nty, ntz, forder, q_grid_loc, q_grid)
  use stream,only:outu
  use colfft_util,only:ny_box, nz_box
  use colfft_types,only:q_grid_t
#if KEY_DOMDEC_GPU==1
  use domdec_util_gpu_mod,only:range_start, range_stop
  use domdec_common,only:q_gpu
#endif 
  implicit none
  ! Input / Output
  integer, intent(in) :: nthread, ntx, nty, ntz, forder
  type(q_grid_t), pointer, intent(inout) :: q_grid_loc(:)
  type(q_grid_t), intent(inout) :: q_grid
  ! Functions
#ifdef _OPENMP
  integer omp_get_thread_num
#endif 
  ! Variables
  integer tid
  integer x0, x1, y0, y1, z0, z1
  integer xsize_loc, ysize_loc, zsize_loc, xysize_loc
  integer xsize, ysize, zsize, xysize
  integer x, y, z, ix, iy, iz, tx, ty, tz, t
  integer xlo, ylo, zlo
  integer xlo_loc, ylo_loc, zlo_loc
  integer ix_lo, iy_lo, iz_lo
  integer i, j

#ifdef _OPENMP
  tid = omp_get_thread_num()
#else /**/
  tid = 0
#endif 

  x0 = q_grid_loc(tid)%x0
  x1 = q_grid_loc(tid)%x1
  y0 = q_grid_loc(tid)%y0
  y1 = q_grid_loc(tid)%y1
  z0 = q_grid_loc(tid)%z0
  z1 = q_grid_loc(tid)%z1
  xlo_loc = q_grid_loc(tid)%xlo
  ylo_loc = q_grid_loc(tid)%ylo
  zlo_loc = q_grid_loc(tid)%zlo
  xsize_loc = q_grid_loc(tid)%xsize
  ysize_loc = q_grid_loc(tid)%ysize
  zsize_loc = q_grid_loc(tid)%zsize
  xysize_loc = xsize_loc*ysize_loc

  xlo = q_grid%xlo
  ylo = q_grid%ylo
  zlo = q_grid%zlo
  xsize = q_grid%xsize
  ysize = q_grid%ysize
  zsize = q_grid%zsize
  xysize = xsize*ysize

  if (nthread > 1) then
     !$omp master
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('main block')  
#endif
     !$omp end master

     ! First write the main part of the block
     ! Each thread takes care of block: (x0..x1) (y0..y1) (z0..z1)
     !$omp barrier
     do z=z0,z1
        do y=y0,y1
           !dec$ ivdep
           do x=x0,x1
              j = (x-xlo_loc) + (y-ylo_loc)*xsize_loc + (z-zlo_loc)*xysize_loc
              i = (x-xlo)     + (y-ylo)*xsize         + (z-zlo)*xysize
#if SINGLEP==1
              q_grid%array_sp(i) = q_grid_loc(tid)%array_sp(j)  
#endif
#if DOUBLEP==1
              q_grid%array_dp(i) = q_grid_loc(tid)%array_dp(j)  
#endif
           enddo
        enddo
     enddo
     !$omp master
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()  
#endif
     !$omp end master
  endif

  ix_lo = -1
  iy_lo = -1
  iz_lo = -1

  !$omp master
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('boundaries')  
#endif
  !$omp end master

  ! Second, add in contributions from the boundaries
  ! Get contributions from surrounding threads:
  ! These are the threads in the negative direction from the current thread
  do ix=ix_lo,0
     do iy=iy_lo,0
        do iz=iz_lo,0
           if (ix == 0 .and. iy == 0 .and. iz == 0) cycle
           tx = q_grid_loc(tid)%tx + ix
           ty = q_grid_loc(tid)%ty + iy
           tz = q_grid_loc(tid)%tz + iz
           if (tx < 0) tx = tx + ntx
           if (ty < 0 .and. ny_box == 1) ty = ty + nty
           if (tz < 0 .and. nz_box == 1) tz = tz + ntz
           if (ty < 0 .or. tz < 0) cycle
           ! t = thread id of the source
           t = tz + ty*ntz + tx*ntz*nty

           if (ix == -1) then
              x0 = q_grid_loc(t)%x1 + 1
              x1 = q_grid_loc(t)%x1 + forder - 1
           else
              x0 = q_grid_loc(t)%x0
              x1 = q_grid_loc(t)%x1
           endif
           if (iy == -1) then
              y0 = q_grid_loc(t)%y1 + 1
              y1 = q_grid_loc(t)%y1 + forder - 1
           else
              y0 = q_grid_loc(t)%y0
              y1 = q_grid_loc(t)%y1
           endif
           if (iz == -1) then
              z0 = q_grid_loc(t)%z1 + 1
              z1 = q_grid_loc(t)%z1 + forder - 1
           else
              z0 = q_grid_loc(t)%z0
              z1 = q_grid_loc(t)%z1
           endif
           xsize_loc = q_grid_loc(t)%xsize
           ysize_loc = q_grid_loc(t)%ysize
           xysize_loc = xsize_loc*ysize_loc

           xlo_loc = q_grid_loc(t)%xlo
           ylo_loc = q_grid_loc(t)%ylo
           zlo_loc = q_grid_loc(t)%zlo

           if (q_grid_loc(tid)%tx + ix < 0) then
              ! X direction has been wrapped using periodic boundaries
              ! Set xlo = x0 - xhalo
              xlo = x0 - (q_grid%x0 - q_grid%xlo)
           else
              xlo = q_grid%xlo
           endif

           if (q_grid_loc(tid)%ty + iy < 0) then
              ! Y direction has been wrapped using periodic boundaries
              ! Set ylo = y0 - yhalo
              ylo = y0 - (q_grid%y0 - q_grid%ylo)
           else
              ylo = q_grid%ylo
           endif

           if (q_grid_loc(tid)%tz + iz < 0) then
              ! Z direction has been wrapped using periodic boundaries
              ! Set zlo = z0 - zhalo
              zlo = z0 - (q_grid%z0 - q_grid%zlo)
           else
              zlo = q_grid%zlo
           endif

#if DOUBLEP == 1
           if ((x1-xlo_loc) + (y1-ylo_loc)*xsize_loc + (z1-zlo_loc)*xysize_loc > &
                size(q_grid_loc(t)%array_dp)-1 .or. &
                (x1-xlo) + (y1-ylo)*xsize + (z1-zlo)*xysize > &
                size(q_grid%array_dp)-1) then
              call wrndie(-5,'<colfft_func>','ADD_PREC(reduce_charge_grid) has a problem')
           endif
#else /**/
           if ((x1-xlo_loc) + (y1-ylo_loc)*xsize_loc + (z1-zlo_loc)*xysize_loc > &
                size(q_grid_loc(t)%array_sp)-1 .or. &
                (x1-xlo) + (y1-ylo)*xsize + (z1-zlo)*xysize > &
                size(q_grid%array_sp)-1) then
              write (outu,'(2i10)') &
                   (x1-xlo_loc) + (y1-ylo_loc)*xsize_loc + (z1-zlo_loc)*xysize_loc, &
                   size(q_grid_loc(t)%array_sp)-1
              write (outu,'(2i10)') &
                   (x1-xlo) + (y1-ylo)*xsize + (z1-zlo)*xysize, &
                   size(q_grid%array_sp)-1
              call wrndie(-5,'<colfft_func>','ADD_PREC(reduce_charge_grid) has a problem')
           endif
#endif 

           do z=z0,z1
              do y=y0,y1
                 !dec$ ivdep
                 do x=x0,x1
                    j = (x-xlo_loc) + (y-ylo_loc)*xsize_loc + (z-zlo_loc)*xysize_loc
                    i = (x-xlo)     + (y-ylo)*xsize         + (z-zlo)*xysize
#if SINGLEP==1
                    q_grid%array_sp(i) = q_grid%array_sp(i) + q_grid_loc(t)%array_sp(j)  
#endif
#if DOUBLEP==1
                    q_grid%array_dp(i) = q_grid%array_dp(i) + q_grid_loc(t)%array_dp(j)  
#endif
                 enddo
              enddo
           enddo
        enddo
     enddo
  enddo
  !$omp barrier

  !$omp master
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()  
#endif
  !$omp end master

  return
end subroutine ADD_PREC(reduce_charge_grid)

! *
! * Calculates fr1, fr2, fr3, theta1, theta2, theta3, dtheta1, dtheta2, dtheta3
! *
subroutine ADD_PREC(fill_bspline)(n_grid_atom, grid_atom, &
     forder, nfft1, nfft2, nfft3, x, y, z, charge, recip, lorthog, &
     fr1, fr2, fr3, theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, &
     grid2tx_lo, grid2ty_lo, grid2tz_lo, &
     grid2tx_hi, grid2ty_hi, grid2tz_hi, &
     grid2tx, grid2ty, grid2tz, &
     nthread, natom_thread, thread_id_list, atomlist_thread)
  use memory
  use domdec_common,only:divide_thread_work, simd_version, SIMD_NONE
  use colfft_kernel,only: ADD_PREC(fill_bspline_kernel_fortran)
  use colfft_util,only:coord_to_grid, q_ydim_periodic, q_zdim_periodic
#if KEY_DOMDEC_GPU==1
  use domdec_util_gpu_mod,only:range_start, range_stop  
#endif
  use nblist_types,only:intarray_t
  implicit none
  ! Input / Output
  integer, intent(in) :: n_grid_atom, grid_atom(:)
  integer,intent(in) :: forder,  nfft1, nfft2, nfft3
#if DOUBLEP == 1
  real(chm_real), intent(in), dimension(:) :: x, y, z, charge
  real(chm_real), intent(in), dimension(3,3) :: recip
#else /**/
  real(chm_real4), intent(in), dimension(:) :: x, y, z, charge
  real(chm_real4), intent(in), dimension(3,3) :: recip
#endif 
  logical, intent(in) :: lorthog
  integer,intent(out),dimension(:) :: fr1,fr2,fr3
#if DOUBLEP == 1
  real(chm_real),intent(out),dimension(:,:) :: theta1,theta2,theta3
  real(chm_real),intent(out),dimension(:,:) :: dtheta1,dtheta2,dtheta3
#elif SINGLEP == 1
  real(chm_real4),intent(out),dimension(:,:) :: theta1,theta2,theta3
  real(chm_real4),intent(out),dimension(:,:) :: dtheta1,dtheta2,dtheta3
#endif 
  integer, intent(in) :: grid2tx_lo, grid2ty_lo, grid2tz_lo
  integer, intent(in) :: grid2tx_hi, grid2ty_hi, grid2tz_hi
  integer, intent(in) :: grid2tx(0:grid2tx_hi-grid2tx_lo)
  integer, intent(in) :: grid2ty(0:grid2ty_hi-grid2ty_lo)
  integer, intent(in) :: grid2tz(0:grid2tz_hi-grid2tz_lo)
  integer, intent(in) :: nthread
  type(intarray_t), intent(inout) :: natom_thread(0:nthread)
  integer, intent(inout) :: thread_id_list(:)
  type(intarray_t), intent(inout) :: atomlist_thread(0:nthread-1)
  ! Parameters
#if DOUBLEP == 1
  real(chm_real), parameter :: two = 2.0_chm_real, half = 0.5_chm_real
#elif SINGLEP == 1
  real(chm_real4), parameter :: two = 2.0_chm_real4, half = 0.5_chm_real4
#endif 
  ! Functions
#ifdef _OPENMP
  integer omp_get_thread_num  
#endif
  ! Variables
#if DOUBLEP == 1
  real(chm_real) recip_orthog(3)
#elif SINGLEP == 1
  real(chm_real4) recip_orthog(3)
#endif 
  integer ydim, zdim
  integer tid, jstart, jend, t, tt

  ! Set (ydim, zdim)
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

  !$omp parallel private(tid, jstart, jend)
#ifdef _OPENMP
  tid = omp_get_thread_num()
#else /**/
  tid = 0
#endif 
  call divide_thread_work(n_grid_atom, jstart, jend)

  natom_thread(tid)%array(0:nthread-1) = 0

  if (lorthog .and. forder == 4 .and. simd_version /= SIMD_NONE) then
     ! Use fast SSE version
     recip_orthog(1) = recip(1,1)
     recip_orthog(2) = recip(2,2)
     recip_orthog(3) = recip(3,3)
     call ADD_PREC(fill_bspline4_kernel_sse)(jstart, jend, grid_atom(1), &
          x(1), y(1), z(1), charge(1), recip_orthog(1), &
          ydim, zdim, nfft1, nfft2, nfft3, fr1(1), fr2(1), fr3(1), &
          theta1(1,1), theta2(1,1), theta3(1,1), dtheta1(1,1), dtheta2(1,1), dtheta3(1,1), &
          grid2tx(0), grid2ty(0), grid2tz(0), grid2tx_lo, grid2ty_lo, grid2tz_lo, &
          natom_thread(tid)%array(0), thread_id_list(1))
  else
     ! Use slow general version
     call ADD_PREC(fill_bspline_kernel_fortran)(jstart, jend, grid_atom, &
          x, y, z, charge, recip, &
          forder, ydim, zdim, &
          nfft1, nfft2, nfft3, &
          fr1, fr2, fr3, &
          theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, &
          grid2tx_lo, grid2ty_lo, grid2tz_lo, &
          grid2tx_hi, grid2ty_hi, grid2tz_hi, &
          grid2tx, grid2ty, grid2tz, &
          nthread, &
          natom_thread(tid)%array, &
          thread_id_list)
  endif

  !$omp end parallel

  !$omp parallel private(tid, jstart, jend)
#ifdef _OPENMP
  tid = omp_get_thread_num()
#else /**/
  tid = 0
#endif 
  call divide_thread_work(n_grid_atom, jstart, jend)
  call build_atomlist_thread(nthread, tid, jstart, jend, natom_thread, thread_id_list, &
       atomlist_thread)
  !$omp end parallel

!!$    call build_atomlist_thread_safe(nthread, n_grid_atom, natom_thread, thread_id_list, &
!!$         atomlist_thread)

  return
end subroutine ADD_PREC(fill_bspline)

!  ##ENDEX (exp_precision)
!  ##EXP P0 sp dp .when. EXP (exp_precision)
!  ##PASS1 PS sp
!  ##PASS2 PD dp
!  ##EXEND

#if KEY_DOMDEC==1 /*domdec*/

! *
! * Second part of test for grad_sumrc_ps, computes (dx, dy, dz) using the Fortran
! * code and compares the result
! *
subroutine ADD_PREC(test2_grad_sumrc)(forder, n_grid_atom, grid_atom, &
     fr1, fr2, fr3, nfft1, nfft2, nfft3, &
     recip_in, charge, &
     theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, &
     qlen, q, dx, dy, dz)
  use stream,only:outu
  use parallel,only:mynod
  use memory,only:chmdealloc
  use colfft_kernel, only: ADD_PREC(gather_force_kernel_fortran)
  use colfft_util,only:get_halo_limits, get_spatial_sizes, YZ_X_PARTITION
  use consta,only:ccelec
  implicit none
  ! Input / Output
  integer, intent(in) :: forder, n_grid_atom, grid_atom(:)
  integer, intent(in) :: fr1(:), fr2(:), fr3(:), nfft1, nfft2, nfft3
  integer, intent(in) :: qlen
#if DOUBLEP == 1
  real(chm_real), intent(in) :: recip_in(3,3)
  real(chm_real), intent(in) :: charge(:)
  real(chm_real), intent(in) :: theta1(:,:), theta2(:,:), theta3(:,:)
  real(chm_real), intent(in) :: dtheta1(:,:), dtheta2(:,:), dtheta3(:,:)
  real(chm_real), intent(in) :: q(0:qlen-1)
#else /**/
  real(chm_real4), intent(in) :: recip_in(3,3)
  real(chm_real4), intent(in) :: charge(:)
  real(chm_real4), intent(in) :: theta1(:,:), theta2(:,:), theta3(:,:)
  real(chm_real4), intent(in) :: dtheta1(:,:), dtheta2(:,:), dtheta3(:,:)
  real(chm_real4), intent(in) :: q(0:qlen-1)
#endif 
  real(chm_real), intent(in) :: dx(:), dy(:), dz(:)
  ! Parameters
#if DOUBLEP == 1
  real(chm_real), parameter :: errtol = 1.0e-5_chm_real
#else /**/
  real(chm_real), parameter :: errtol = 1.0e-4_chm_real
#endif 
  ! Variables
#if DOUBLEP == 1
  real(chm_real) dnfft1, dnfft2, dnfft3, recip(3,3)
#else /**/
  real(chm_real4) dnfft1, dnfft2, dnfft3, recip(3,3)
#endif 
  integer xhalomin,xhalomax,yhalomin,yhalomax,zhalomin,zhalomax
  integer xsize, ysize, zsize
  integer i, j

  dnfft1 = nfft1
  dnfft2 = nfft2
  dnfft3 = nfft3

  dnfft1 = dnfft1*ccelec
  dnfft2 = dnfft2*ccelec
  dnfft3 = dnfft3*ccelec

  recip(1,1:3) = recip_in(1,1:3)*dnfft1
  recip(2,1:3) = recip_in(2,1:3)*dnfft2
  recip(3,1:3) = recip_in(3,1:3)*dnfft3

  call get_halo_limits(YZ_X_PARTITION, &
       xhalomin, xhalomax, yhalomin, yhalomax, zhalomin, zhalomax, mynod)

  call get_spatial_sizes(YZ_X_PARTITION, xsize, ysize, zsize, mynod)

  call ADD_PREC(gather_force_kernel_fortran)(forder, 1, n_grid_atom, grid_atom, &
       fr1, fr2, fr3, &
       yhalomin, zhalomin, &
       xsize, ysize, &
       recip, &
       charge, &
       theta1, theta2, theta3, &
       dtheta1, dtheta2, dtheta3, &
       qlen, q, &
       dx_orig, dy_orig, dz_orig)

  do i=1,n_grid_atom
     j = grid_atom(i)
     if (abs(dx(j) - dx_orig(j)) > errtol .or. &
          abs(dy(j) - dy_orig(j)) > errtol .or. &
          abs(dz(j) - dz_orig(j)) > errtol) then
        write (outu,'(a,2i8)') 'i,j=',i,j
        write (outu,'(a,3g12.6)') 'dx,dy,dz               =',dx(j),dy(j),dz(j)
        write (outu,'(a,3g12.6)') 'dx_orig,dy_orig,dz_orig=',dx_orig(j),dy_orig(j),dz_orig(j)
        write (outu,'(a,3g12.4)') 'diff                   =',&
             abs(dx(j) - dx_orig(j)), abs(dy(j) - dy_orig(j)), abs(dz(j) - dz_orig(j))
        call wrndie(-5,'<colfft_func>',&
             'ADD_PREC(test2_grad_sumrc): forces calculated incorrectly')
     endif
  enddo

  call chmdealloc('colfft_func.src','ADD_PREC(test2_grad_sumrc)','dx_orig',size(dx_orig),crl=dx_orig)
  call chmdealloc('colfft_func.src','ADD_PREC(test2_grad_sumrc)','dy_orig',size(dy_orig),crl=dy_orig)
  call chmdealloc('colfft_func.src','ADD_PREC(test2_grad_sumrc)','dz_orig',size(dz_orig),crl=dz_orig)

  return
end subroutine ADD_PREC(test2_grad_sumrc)

! *
! * Test spread_charge_grid
! *
subroutine ADD_PREC(test_spread_charge_grid)(n_grid_atom, grid_atom, &
     fr1, fr2, fr3, theta1, theta2, theta3, &
     q_grid, charge, forder, nfft2, nfft3)
  use stream,only:outu
  use memory
  use colfft_util,only:get_spatial_limits, YZ_X_PARTITION
  use parallel,only:mynod, numnod
  use colfft_types,only:q_grid_t
  ! MSLDPME ->
#if KEY_BLOCK==1
  use lambdam,only:bixlam,iqldm_pme
  use block_ltm,only:iblckp
#endif
  ! <- MSLDPME
  implicit none
  ! Input / Output
  integer, intent(in) :: n_grid_atom, grid_atom(:)
  integer, intent(in) :: fr1(:), fr2(:), fr3(:)
#if DOUBLEP == 1
  real(chm_real), intent(in) :: theta1(:,:), theta2(:,:), theta3(:,:)
  real(chm_real), intent(in) :: charge(:)
#elif SINGLEP == 1
  real(chm_real4), intent(in) :: theta1(:,:), theta2(:,:), theta3(:,:)
  real(chm_real4), intent(in) :: charge(:)
#endif 
  type(q_grid_t), intent(in) :: q_grid
  integer, intent(in) :: forder, nfft2, nfft3
  ! Parameters
#if DOUBLEP == 1
  real(chm_real), parameter :: zero = 0.0_chm_real, one = 1.0_chm_real
#elif SINGLEP == 1
  real(chm_real4), parameter :: zero = 0.0_chm_real4, one = 1.0_chm_real4
#endif 
  ! Variables
#if DOUBLEP == 1
  real(chm_real), allocatable, dimension(:) :: qcheck
  real(chm_real) chargen, theta3v, prod, qval
#elif SINGLEP == 1
  real(chm_real4), allocatable, dimension(:) :: qcheck
  real(chm_real4) chargen, theta3v, prod, qval
#endif 
  integer qcheck_len
  integer x0, x1, y0, y1, z0, z1
  integer xlo, ylo, zlo, xsize, ysize
  integer xlen, ylen, zlen
  integer i, j, k, n, ind, indc
  integer fr1i, fr2i, fr3i
  integer ix, iy, iz
  integer ith1, ith2, ith3
  logical q_ydir, q_zdir, test_ok

  ! Get the grid region which we are testing
  call get_spatial_limits(YZ_X_PARTITION, x0, x1, y0, y1, z0, z1, mynod)

  xlo = q_grid%xlo
  ylo = q_grid%ylo
  zlo = q_grid%zlo
  xsize = q_grid%xsize
  ysize = q_grid%ysize

  q_ydir = .false.
  q_zdir = .false.
  if (y0 == 0 .and. y1 == nfft2-1) q_ydir = .true.
  if (z0 == 0 .and. z1 == nfft3-1) q_zdir = .true.

  xlen = x1-x0+1
  ylen = y1-y0+1
  zlen = z1-z0+1
  qcheck_len = (x1-x0+1)*(y1-y0+1)*(z1-z0+1)

#if DOUBLEP == 1
  call chmalloc('colfft_func.src','ADD_PREC(test_spread_charge_grid)','qcheck',&
       qcheck_len,lbou=0,crl=qcheck)
#elif SINGLEP == 1
  call chmalloc('colfft_func.src','ADD_PREC(test_spread_charge_grid)','qcheck',&
       qcheck_len,lbou=0,cr4=qcheck)
#endif 

  ! Calculate qcheck
  qcheck(0:qcheck_len-1) = zero

  do i = 1,n_grid_atom
     n = grid_atom(i)

     fr1i = fr1(i)
     fr2i = fr2(i)
     fr3i = fr3(i)

     ! MSLDPME ->
#if KEY_BLOCK==1
     if (iqldm_pme>0) then
        chargen = bixlam(iblckp(n)) * charge(n)
     else
#endif
        ! <- MSLDPME
        chargen = charge(n)
        ! MSLDPME ->
#if KEY_BLOCK==1
     endif
#endif
     ! <- MSLDPME

     do ith3 = 1,forder
        theta3v = theta3(ith3,i)*chargen
        do ith2 = 1,forder
           prod = theta2(ith2,i)*theta3v
           do ith1 = 1,forder
              ix = fr1i + ith1 - 1
              iy = fr2i + ith2 - 1
              iz = fr3i + ith3 - 1

              if (ix < x0) then
                 write (outu,'(a)') 'ADD_PREC(test_spread_charge_grid): ix < x0'
                 stop
              endif

              if (ix > x1) ix = ix - xlen

              if (iy >= nfft2) iy = iy - nfft2
              if (iz >= nfft3) iz = iz - nfft3

              if (iy < y0) then
                 if (q_ydir) then
                    write (outu,'(a)') 'ADD_PREC(test_spread_charge_grid): iy < y0'
                    stop
                 endif
                 cycle
              endif

              if (iz < z0) then
                 if (q_zdir) then
                    write (outu,'(a)') 'ADD_PREC(test_spread_charge_grid): iz < z0'
                    stop
                 endif
                 cycle
              endif

              if (iy > y1) then
                 if (q_ydir) then
                    iy = iy - ylen
                 else
                    cycle
                 endif
              endif

              if (iz > z1) then
                 if (q_zdir) then
                    iz = iz - zlen
                 else
                    cycle           ! system not periodic in z => cycle
                 endif
              endif

              ind = (ix-x0) + (iy-y0)*xlen + (iz-z0)*xlen*ylen
              qcheck(ind) = qcheck(ind) + theta1(ith1,i)*prod
           enddo
        enddo
     enddo
  enddo

  ! Compare q vs. qcheck
  test_ok = .true.
  do k=z0,z1
     do j=y0,y1
        do i=x0,x1
           ind  = (i-xlo) + (j-ylo)*xsize + (k-zlo)*xsize*ysize
           indc = (i-x0)  + (j-y0)*xlen   + (k-z0)*xlen*ylen
#if SINGLEP==1
           qval = q_grid%array_sp(ind)  
#endif
#if DOUBLEP==1
           qval = q_grid%array_dp(ind)  
#endif
           if (abs(qval - qcheck(indc)) > 1.0e-5) then
              write (outu,'(a,4i3)') 'mynod,i,j,k=',mynod,i,j,k
              write (outu,'(a,2g13.5,2i8)') 'q,qcheck,ind,indc=',&
                   qval,qcheck(indc),ind,indc
              test_ok = .false.
              goto 666
           endif
        enddo
     enddo
  enddo

666 continue

#if DOUBLEP == 1
  call chmdealloc('colfft_func.src','ADD_PREC(test_spread_charge_grid)','qcheck',&
       qcheck_len,crl=qcheck)
#elif SINGLEP == 1
  call chmdealloc('colfft_func.src','ADD_PREC(test_spread_charge_grid)','qcheck',&
       qcheck_len,cr4=qcheck)
#endif 

  if (test_ok) then
     if (mynod == 0) write (outu,'(a)') 'ADD_PREC(test_spread_charge_grid) OK'
  else
     call wrndie(-5,'<colfft_func>','ADD_PREC(test_spread_charge_grid) FAILED')
  endif

  return
end subroutine ADD_PREC(test_spread_charge_grid)

! *
! * Makes a copy of q(0:*) for the testing routine. Allocates q_check_ps/pd
! *
subroutine ADD_PREC(test1_scalar_sumrc)(size_q, q)
  use memory,only:chmalloc
  implicit none
  ! Input / Output
  integer, intent(in) :: size_q
#if DOUBLEP == 1
  real(chm_real), intent(in) :: q(0:*)
#else /**/
  real(chm_real4), intent(in) :: q(0:*)
#endif 

#if DOUBLEP == 1
  call chmalloc('colfft_func.src','ADD_PREC(test1_scalar_sumrc)','q_check',&
       size_q,lbou=0,crl=q_check_pd)
#else /**/
  call chmalloc('colfft_func.src','ADD_PREC(test1_scalar_sumrc)','q_check',&
       size_q,lbou=0,cr4=q_check_ps)
#endif 

  ADD_PREC(q_check)(0:size_q-1) = q(0:size_q-1)

  return
end subroutine ADD_PREC(test1_scalar_sumrc)

! *
! * Test scalar_sumrc. Deallocated q_check_ps/pd
! *
subroutine ADD_PREC(test2_scalar_sumrc)(nfft1, nfft2, nfft3, &
     kappa, volume, prefac1, prefac2, prefac3, recip, &
     size_q, q, energy, virial, q_calc_energy_virial)
  use stream,only:outu
  use parallel,only:mynod
  use memory,only:chmdealloc
  use colfft_kernel,only: ADD_PREC(scalar_sumrc_kernel_orig_o0)
  use colfft_util,only:get_fft_limits, get_fft_sizes, XY_Z_PARTITION
  use consta,only:pi
  implicit none
  ! Input / Output
  integer, intent(in) :: nfft1, nfft2, nfft3
  integer, intent(in) :: size_q
  real(chm_real),intent(in) :: kappa,volume
#if DOUBLEP == 1
  real(chm_real), intent(in) :: prefac1(nfft1),prefac2(nfft2),prefac3(nfft3)
  real(chm_real), intent(in) :: recip(3,3)
! #if orthog == 1
!   real(chm_real), intent(in) :: m1_tbl(-(nfft1/2+1):(nfft1/2+1))
!   real(chm_real), intent(in) :: m2_tbl(-(nfft2/2+1):(nfft2/2+1))
!   real(chm_real), intent(in) :: m3_tbl(-(nfft3/2+1):(nfft3/2+1))
! #endif 
  real(chm_real), intent(in) :: q(0:*)
#else /**/
  real(chm_real4), intent(in) :: prefac1(nfft1),prefac2(nfft2),prefac3(nfft3)
  real(chm_real4), intent(in) :: recip(3,3)
! #if orthog == 1
!   real(chm_real4), intent(in) :: m1_tbl(-(nfft1/2+1):(nfft1/2+1))
!   real(chm_real4), intent(in) :: m2_tbl(-(nfft2/2+1):(nfft2/2+1))
!   real(chm_real4), intent(in) :: m3_tbl(-(nfft3/2+1):(nfft3/2+1))
! #endif 
  real(chm_real4), intent(in) :: q(0:*)
#endif 
  real(chm_real), intent(in) :: energy, virial(6)
  logical, intent(in) :: q_calc_energy_virial
  ! Parameters
  real(chm_real), parameter :: zerod = 0.0_chm_real, half = 0.5_chm_real
#if DOUBLEP == 1
  real(chm_real), parameter :: etol = 1.0e-6_chm_real, vtol = 1.0e-6_chm_real, &
       qtol = 1.0e-9_chm_real
#else /**/
  real(chm_real), parameter :: etol = 1.0e-5_chm_real, vtol = 1.0e-5_chm_real, &
       qtol = 1.0e-8_chm_real
#endif 
#if DOUBLEP == 1
  real(chm_real), parameter :: zero = 0.0_chm_real
#else /**/
  real(chm_real4), parameter :: zero = 0.0_chm_real4
#endif 
  ! Variables
  real(chm_real) :: energy_check, virial_check(6)
#if DOUBLEP == 1
  real(chm_real) fac, piv_inv
#elif SINGLEP == 1
  real(chm_real4) fac, piv_inv
#endif 
  integer nf1, nf2, nf3, nff
  integer xgridmin, xgridmax, ygridmin, ygridmax, zgridmin, zgridmax
  integer xgridsize, ygridsize, zgridsize, xzgridsize
  integer i

  piv_inv = 1.d0/(pi*volume)
  fac = pi*pi/(kappa*kappa)
  nff = nfft1*nfft2
  nf1 = nfft1/2 + mod(nfft1,2)
  nf2 = nfft2/2 + mod(nfft2,2)
  nf3 = nfft3/2 + mod(nfft3,2)

  call get_fft_limits(XY_Z_PARTITION,zgridmin,zgridmax,xgridmin,xgridmax, &
       ygridmin,ygridmax,mynod)

  call get_fft_sizes(XY_Z_PARTITION,zgridsize,xgridsize,ygridsize,mynod)
  xzgridsize = xgridsize*zgridsize

  energy_check = zerod
  virial_check(1:6) = zerod

#if KEY_PARALLEL==1
  if (mynod == 0) then  
#endif
     ADD_PREC(q_check)(0:1) = zero
#if KEY_PARALLEL==1
  endif  
#endif

  call ADD_PREC(scalar_sumrc_kernel_orig_o0)(nfft1, nfft2, nfft3, nf1, nf2, nf3, &
       xgridmin, xgridmax, ygridmin, ygridmax, zgridmin, zgridmax, &
       zgridsize, xzgridsize, &
       fac, piv_inv, prefac1, prefac2, prefac3, recip, &
       ADD_PREC(q_check), energy_check, virial_check)

  energy_check = half*energy_check
  virial_check(1:6) = half*virial_check(1:6)

  if (q_calc_energy_virial) then
     if (abs(energy-energy_check) > etol) then
        write (outu,'(a,2f12.6,g12.4)') 'energy,energy_check,diff=',&
             energy,energy_check,abs(energy-energy_check)
        call wrndie(-5,'<colfft_func>',&
             'ADD_PREC(test2_scalar_sumrc): energy calculated incorrectly')
     endif

     do i=1,6
        if (abs(virial(i)-virial_check(i)) > vtol) then
           write (outu,'(a,i2,2f12.6,g12.4)') 'i,virial,virial_check,diff=',&
                i,virial(i),virial_check(i),abs(virial(i)-virial_check(i))
           call wrndie(-5,'<colfft_func>',&
                'ADD_PREC(test2_scalar_sumrc): virial calculated incorrectly')
        endif
     enddo
  endif

  do i=0,size_q-1
     if (abs(q(i) - ADD_PREC(q_check)(i)) > qtol) then
        write (outu,'(a,2i8,2f12.6,g12.4)') 'i,size_q,q,q_check,diff=',&
             i,size_q,q(i), ADD_PREC(q_check)(i), abs(q(i) - ADD_PREC(q_check)(i))
        call wrndie(-5,'<colfft_func>',&
             'ADD_PREC(test2_scalar_sumrc): q calculated incorrectly')
     endif
  enddo

#if DOUBLEP == 1
  call chmdealloc('colfft_func.src','ADD_PREC(test2_scalar_sumrc)','q_check_pd',&
       size(q_check_pd),crl=q_check_pd)
#else /**/
  call chmdealloc('colfft_func.src','ADD_PREC(test2_scalar_sumrc)','q_check_ps',&
       size(q_check_ps),cr4=q_check_ps)
#endif 

  if (mynod == 0) write (outu,'(a)') 'ADD_PREC(test_scalar_sumrc) OK'

  return
end subroutine ADD_PREC(test2_scalar_sumrc)

! *
! * Tests fill_bspline -subroutine
! *
subroutine ADD_PREC(test_fill_bspline)(n_grid_atom, grid_atom, &
     natom, forder, x, y, z, charge, recip, nfft2, nfft3, &
     fr1, fr2, fr3, theta1, theta2, theta3, dtheta1, dtheta2, dtheta3, &
     grid2tx_lo, grid2ty_lo, grid2tz_lo, &
     grid2tx_hi, grid2ty_hi, grid2tz_hi, &
     grid2tx, grid2ty, grid2tz, &
     nthread, natom_thread, thread_id_list, atomlist_thread, q_grid)
  use stream,only:outu
  use parallel,only:mynod
  use colfft_util,only:coord_to_grid, q_ydim_periodic, q_zdim_periodic
  use colfft_types,only:q_grid_t
  use nblist_types,only:intarray_t
  implicit none
  ! Input / Output
  integer, intent(in) :: n_grid_atom, grid_atom(:)
  integer,intent(in) :: natom, forder
#if DOUBLEP == 1
  real(chm_real), intent(in), dimension(natom) :: x, y, z, charge
  real(chm_real), intent(in), dimension(3,3) :: recip
#else /**/
  real(chm_real4), intent(in), dimension(natom) :: x, y, z, charge
  real(chm_real4), intent(in), dimension(3,3) :: recip
#endif 
  integer,intent(in) :: nfft2, nfft3
  integer,intent(in),dimension(natom) :: fr1,fr2,fr3
#if DOUBLEP == 1
  real(chm_real),intent(in),dimension(forder,natom) :: theta1,theta2,theta3
  real(chm_real),intent(in),dimension(forder,natom) :: dtheta1,dtheta2,dtheta3
#elif SINGLEP == 1
  real(chm_real4),intent(in),dimension(forder,natom) :: theta1,theta2,theta3
  real(chm_real4),intent(in),dimension(forder,natom) :: dtheta1,dtheta2,dtheta3
#endif 
  integer, intent(in) :: grid2tx_lo, grid2ty_lo, grid2tz_lo
  integer, intent(in) :: grid2tx_hi, grid2ty_hi, grid2tz_hi
  integer, intent(in) :: grid2tx(0:grid2tx_hi-grid2tx_lo)
  integer, intent(in) :: grid2ty(0:grid2ty_hi-grid2ty_lo)
  integer, intent(in) :: grid2tz(0:grid2tz_hi-grid2tz_lo)
  integer, intent(in) :: nthread
  type(intarray_t), intent(in) :: natom_thread(0:nthread)
  integer, intent(in) :: thread_id_list(:)
  type(intarray_t), intent(in) :: atomlist_thread(0:nthread-1)
  type(q_grid_t), pointer, intent(in) :: q_grid(:)
  ! Parameters
#if DOUBLEP == 1
  real(chm_real), parameter :: zero = 0.0_chm_real, two = 2.0_chm_real, half = 0.5_chm_real
  real(chm_real), parameter :: theta_tol = 1.0e-10_chm_real
#elif SINGLEP == 1
  real(chm_real4), parameter :: zero = 0.0_chm_real4, two = 2.0_chm_real4, half = 0.5_chm_real4
  real(chm_real4), parameter :: theta_tol = 1.0e-6_chm_real4
#endif 
  ! Variables
#if DOUBLEP == 1
  real(chm_real) fr1n, fr2n, fr3n
  real(chm_real) w1, w2, w3
  real(chm_real) theta1v(forder), theta2v(forder), theta3v(forder)
  real(chm_real) dtheta1v(forder), dtheta2v(forder), dtheta3v(forder)
#elif SINGLEP == 1
  real(chm_real4) fr1n, fr2n, fr3n
  real(chm_real4) w1, w2, w3
  real(chm_real4) theta1v(forder), theta2v(forder), theta3v(forder)
  real(chm_real4) dtheta1v(forder), dtheta2v(forder), dtheta3v(forder)
#endif 
  integer ydim, zdim
  integer natom_thread_check(0:nthread-1), thread_id
  integer fr1i, fr2i, fr3i, dfr2, dfr3
  integer i, j, n, k

  ! Set (ydim, zdim)
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

  natom_thread_check(0:nthread-1) = 0

  do i = 1,n_grid_atom
     n = grid_atom(i)

     call coord_to_grid(x(n), y(n), z(n), recip, fr1n, fr2n, fr3n)

     fr1i = int(fr1n)
     fr2i = int(fr2n)
     fr3i = int(fr3n)
     ! fr1i = 0...nfft1-1
     ! fr2i = 0...nfft2-1
     ! fr3i = 0...nfft3-1

     w1 = fr1n - fr1i
     w2 = fr2n - fr2i
     w3 = fr3n - fr3i

     ! Apply periodic boundaries
     dfr2 = ((fr2i + forder - 1)/ydim)*ydim
     dfr3 = ((fr3i + forder - 1)/zdim)*zdim
     fr2i = fr2i - dfr2
     fr3i = fr3i - dfr3
     ! For periodic systems:
     ! fr1i = 0...nfft1-1
     ! fr2i = -(forder-1)...nfft2-forder
     ! fr3i = -(forder-1)...nfft3-forder
     ! For non-periodic systems:
     ! fr1i = 0...nfft1-1
     ! fr2i = 0...nfft2-1
     ! fr3i = 0...nfft3-1

     if (fr1(i) /= fr1i .or. fr2(i) /= fr2i .or. fr3(i) /= fr3i .or. &
          fr1i < grid2tx_lo .or. fr2i < grid2ty_lo .or. fr3i < grid2tz_lo .or. &
          fr1i > grid2tx_hi .or. fr2i > grid2ty_hi .or. fr3i > grid2tz_hi) then
        write (outu,'(a,3i8)') 'i,n,n_grid_atom=',i,n,n_grid_atom
        write (outu,'(a,3i6)') 'fr1i,  fr2i,  fr3i  =',fr1i,fr2i,fr3i
        write (outu,'(a,3i6)') 'fr1(i),fr2(i),fr3(i)=',fr1(i),fr2(i),fr3(i)
        call wrndie(-5,'<colfft_func>',&
             'ADD_PREC(test_fill_bspline): fr1, fr2, fr3 is calculated incorrectly')
     endif

     thread_id = grid2tx(fr1i-grid2tx_lo) + grid2ty(fr2i-grid2ty_lo) + grid2tz(fr3i-grid2tz_lo)
     if (thread_id < 0 .or. thread_id > nthread-1 .or. &
          thread_id_list(i) < 0 .or. thread_id_list(i) > nthread-1 .or. &
          thread_id_list(i) /= thread_id) then
        call wrndie(-5,'<colfft_func>',&
             'ADD_PREC(test_fill_bspline): thread_id_list is setup incorrectly')
     endif
     natom_thread_check(thread_id) = natom_thread_check(thread_id) + 1

     if (fr1i < q_grid(thread_id)%xgridlo .or. &
          fr1i > q_grid(thread_id)%xgridhi .or. &
          fr2i < q_grid(thread_id)%ygridlo .or. &
          fr2i > q_grid(thread_id)%ygridhi .or. &
          fr3i < q_grid(thread_id)%zgridlo .or. &
          fr3i > q_grid(thread_id)%zgridhi) then
        write (outu,'(a,3i4,a,3i4,a,3i4)') 'fr1,fr2,fr3=',fr1i,fr2i,fr3i, &
             ' low limit=',&
             q_grid(thread_id)%xgridlo,&
             q_grid(thread_id)%ygridlo, &
             q_grid(thread_id)%zgridlo, &
             ' high limit=',&
             q_grid(thread_id)%xgridhi, &
             q_grid(thread_id)%ygridhi, &
             q_grid(thread_id)%zgridhi
        call wrndie(-5,'<colfft_func>','ADD_PREC(test_fill_bspline): fr1, fr2, fr3 out of bounds')
     endif

     call ADD_PREC(calc_bspline)(w1,w2,w3,forder,&
          theta1v,theta2v,theta3v,dtheta1v,dtheta2v,dtheta3v)

     do j=1,forder
        if (abs(theta1v(j) - theta1(j,i)) > theta_tol .or. &
             abs(theta2v(j) - theta2(j,i)) > theta_tol .or. &
             abs(theta3v(j) - theta3(j,i)) > theta_tol .or. &
             abs(dtheta1v(j) - dtheta1(j,i)) > theta_tol .or. &
             abs(dtheta2v(j) - dtheta2(j,i)) > theta_tol .or. &
             abs(dtheta3v(j) - dtheta3(j,i)) > theta_tol) then
           write (outu,'(a,i8,i2)') 'i,j=',i,j
           write (outu,'(a,3f12.8)') 'theta1,theta2,theta3   =',&
                theta1(j,i),theta2(j,i),theta3(j,i)
           write (outu,'(a,3f12.8)') 'theta1v,theta2v,theta3v=',&
                theta1v(j),theta2v(j),theta3v(j)
           write (outu,'(a,3g18.8)') 'diff=',abs(theta1v(j) - theta1(j,i)),&
                abs(theta2v(j) - theta2(j,i)), abs(theta3v(j) - theta3(j,i))
           write (outu,'(a,3f12.8)') 'dtheta1,dtheta2,dtheta3   =',&
                dtheta1(j,i),dtheta2(j,i),dtheta3(j,i)
           write (outu,'(a,3f12.8)') 'dtheta1v,dtheta2v,dtheta3v=',&
                dtheta1v(j),dtheta2v(j),dtheta3v(j)
           write (outu,'(a,3g18.8)') 'diff=',abs(dtheta1v(j) - dtheta1(j,i)),&
                abs(dtheta2v(j) - dtheta2(j,i)), abs(dtheta3v(j) - dtheta3(j,i))
           call wrndie(-5,'<colfft_func>',&
                'ADD_PREC(test_fill_bspline): theta or dtheta setup incorrectly')
        endif
     enddo
  enddo

  do i=0,nthread-1
     if (natom_thread(nthread)%array(i) /= natom_thread_check(i)) then
        write (outu,'(a,3i8)') 'i,natom_thread(nthread)%array(i),natom_thread_check(i)',&
             i,natom_thread(nthread)%array(i),natom_thread_check(i)
        call wrndie(-5,'<colfft_func>',&
             'ADD_PREC(test_fill_bspline): natom_thread is setup incorrectly')
     endif

     ! Test atomlist_thread(tid)%array(1:natom_thread(nthread)%array(tid))
     do j=1,natom_thread(nthread)%array(i)
        k = atomlist_thread(i)%array(j)
        if (k < 1 .or. k > natom) then
           write (outu,'(a,3i6)') 'i,j,k=',i,j,k
           call wrndie(-5,'<colfft_func>',&
                'ADD_PREC(test_fill_bspline): atomlist_thread incorrect')
        endif
     enddo

  enddo

  if (mynod == 0) write (outu,'(a)') 'ADD_PREC(test_fill_bspline) OK'

  return
end subroutine ADD_PREC(test_fill_bspline)

! *
! * Test routine for "replicate_grid_borders".
! *
subroutine ADD_PREC(test_replicate_grid_borders)(q, forder, nfft2, nfft3)
  use stream,only:outu
  use colfft_util,only:get_spatial_limits, get_halo_limits, get_spatial_sizes, YZ_X_PARTITION
  use parallel,only:mynod
  implicit none
  ! Input / Output
#if DOUBLEP == 1
  real(chm_real), intent(inout) :: q(0:*)
#elif SINGLEP == 1
  real(chm_real4), intent(inout) :: q(0:*)
#endif 
  integer, intent(in) :: forder, nfft2, nfft3
  ! Parameters
#if DOUBLEP == 1
  real(chm_real), parameter :: zero = 0.0_chm_real, one = 1.0_chm_real
#elif SINGLEP == 1
  real(chm_real4), parameter :: zero = 0.0_chm_real4, one = 1.0_chm_real4
#endif 
  ! Variables
  integer xgridmin,xgridmax,ygridmin,ygridmax,zgridmin,zgridmax
  integer xhalomin,xhalomax,yhalomin,yhalomax,zhalomin,zhalomax
  integer xgridsize, ygridsize, zgridsize, xygridsize
  integer i, j, k, ind
  integer i0, j0, k0, ind0
  logical q_ydir, q_zdir, test_ok

  call get_spatial_limits(YZ_X_PARTITION, &
       xgridmin,xgridmax,ygridmin,ygridmax,  &
       zgridmin,zgridmax,mynod)

  call get_halo_limits(YZ_X_PARTITION, &
       xhalomin,xhalomax,yhalomin,yhalomax,  &
       zhalomin,zhalomax,mynod)

  call get_spatial_sizes(YZ_X_PARTITION, xgridsize, ygridsize, zgridsize, mynod)
  xygridsize = xgridsize*ygridsize

  q_ydir = .false.
  q_zdir = .false.
  if (ygridmin == 0 .and. ygridmax == nfft2-1) q_ydir = .true.
  if (zgridmin == 0 .and. zgridmax == nfft3-1) q_zdir = .true.

  test_ok = .true.
  do k=zgridmin,zhalomax
     do j=ygridmin,yhalomax
        do i=xgridmin,xhalomax
           i0 = i
           j0 = j
           k0 = k
           if (i0 > xgridmax) i0 = i0 - (xgridmax - xgridmin + 1)
           if (j0 > ygridmax) j0 = j0 - (ygridmax - ygridmin + 1)
           if (k0 > zgridmax) k0 = k0 - (zgridmax - zgridmin + 1)
           ind  = i-xhalomin  + (j-yhalomin)*xgridsize  + (k-zhalomin)*xygridsize
           ind0 = i0-xhalomin + (j0-yhalomin)*xgridsize + (k0-zhalomin)*xygridsize
           if (q_ydir .and. q_zdir) then
              if (ind /= ind0 .and. q(ind) /= q(ind0)) then
                 test_ok = .false.
                 goto 667
              endif
           elseif (q_ydir) then
              if (k /= k0) then
                 if (q(ind) /= zero) then
                    test_ok = .false.
                    goto 667
                 endif
              elseif (ind /= ind0) then
                 if (q(ind) /= q(ind0)) then
                    test_ok = .false.
                    goto 667
                 endif
              endif
           elseif (q_zdir) then
              if (j /= j0) then
                 if (q(ind) /= zero) then
                    test_ok = .false.
                    goto 667
                 endif
              elseif (ind /= ind0) then
                 if (q(ind) /= q(ind0)) then
                    test_ok = .false.
                    goto 667
                 endif
              endif
           else
              if ((j /= j0 .or. k /= k0)) then
                 if (q(ind) /= zero) then
                    test_ok = .false.
                    goto 667
                 endif
              elseif (ind /= ind0) then
                 if (q(ind) /= q(ind0)) then
                    test_ok = .false.
                    goto 667
                 endif
              endif
           endif
        enddo
     enddo
  enddo

  goto 668

667 continue
  write (outu,'(a,4i3,2g13.5,2i6)') 'mynod,i,j,k,q,ind',&
       mynod,i,j,k,q(ind),q(ind0),ind,ind0
  write (outu,'(a,4i3)') 'mynod,i0,j0,k0=',mynod,i0,j0,k0
668 continue

  if (test_ok) then
     if (mynod == 0) write (outu,'(a)') 'test_replicate_grid_borders OK'
  else
     call wrndie(-5,'<colfft_func>','test_replicate_grid_borders FAILED')
  endif

  return
end subroutine ADD_PREC(test_replicate_grid_borders)

! *
! * Fills arrays with b-splines. Slow reference code.
! *
subroutine ADD_PREC(calc_bspline)(w1,w2,w3,order,array1,array2,array3,darray1,darray2,darray3)
  !---------- use standard b-spline recursions: see doc file
  ! Input / Output
  integer, intent(in) :: order
#if DOUBLEP == 1
  real(chm_real), intent(in) :: w1,w2,w3
  real(chm_real), intent(out) :: array1(order),darray1(order)
  real(chm_real), intent(out) :: array2(order),darray2(order)
  real(chm_real), intent(out) :: array3(order),darray3(order)
#elif SINGLEP == 1
  real(chm_real4), intent(in) :: w1,w2,w3
  real(chm_real4), intent(out) :: array1(order),darray1(order)
  real(chm_real4), intent(out) :: array2(order),darray2(order)
  real(chm_real4), intent(out) :: array3(order),darray3(order)
#endif 
  ! Parameters
#if DOUBLEP == 1
  real(chm_real), parameter :: zero = 0.0_chm_real, one = 1.0_chm_real, &
       half = 0.5_chm_real, two = 2.0_chm_real, three = 3.0_chm_real, third = one/three
#elif SINGLEP == 1
  real(chm_real4), parameter :: zero = 0.0_chm_real4, one = 1.0_chm_real4, &
       half = 0.5_chm_real4, two = 2.0_chm_real4, three = 3.0_chm_real4, third = one/three
#endif 
  ! Variables
#if DOUBLEP == 1
  real(chm_real) div
#elif SINGLEP == 1
  real(chm_real4) div
#endif 
  integer j, k

  !--- do linear case
  select case (order)
  case (4)

     array1(4) = zero
     array2(4) = zero
     array3(4) = zero
     array1(2) = w1
     array2(2) = w2
     array3(2) = w3
     array1(1) = one - w1
     array2(1) = one - w2
     array3(1) = one - w3

     !--- compute standard b-spline recursion
     array1(3) = half*w1*array1(2)
     array2(3) = half*w2*array2(2)
     array3(3) = half*w3*array3(2)

     array1(2) = half*((w1+one)*array1(1) + (two-w1)*array1(2))
     array2(2) = half*((w2+one)*array2(1) + (two-w2)*array2(2))
     array3(2) = half*((w3+one)*array3(1) + (two-w3)*array3(2))

     array1(1) = half*(one-w1)*array1(1)
     array2(1) = half*(one-w2)*array2(1)
     array3(1) = half*(one-w3)*array3(1)

     !--- perform standard b-spline differentiation
     darray1(1) = -array1(1)
     darray2(1) = -array2(1)
     darray3(1) = -array3(1)
     do j = 2,4
        darray1(j) = array1(j-1) - array1(j)
        darray2(j) = array2(j-1) - array2(j)
        darray3(j) = array3(j-1) - array3(j)
     enddo

     !--- one more recursion
     array1(4) = third*w1*array1(3)
     array2(4) = third*w2*array2(3)
     array3(4) = third*w3*array3(3)
     do j = 1,2
        array1(4-j) = third*((w1+j)*array1(3-j) + (4-j-w1)*array1(4-j))
        array2(4-j) = third*((w2+j)*array2(3-j) + (4-j-w2)*array2(4-j))
        array3(4-j) = third*((w3+j)*array3(3-j) + (4-j-w3)*array3(4-j))
     enddo

     array1(1) = third*(one-w1)*array1(1)
     array2(1) = third*(one-w2)*array2(1)
     array3(1) = third*(one-w3)*array3(1)

  case default
     !--- compute standard b-spline recursion

     array1(order) = zero
     array2(order) = zero
     array3(order) = zero
     array1(2) = w1
     array2(2) = w2
     array3(2) = w3
     array1(1) = one - w1
     array2(1) = one - w2
     array3(1) = one - w3

     do k = 3,order-1
        div = one / (k-1)
        array1(k) = div*w1*array1(k-1)
        array2(k) = div*w2*array2(k-1)
        array3(k) = div*w3*array3(k-1)
        do j = 1,k-2
           array1(k-j) = div*((w1+j)*array1(k-j-1) + (k-j-w1)*array1(k-j))
           array2(k-j) = div*((w2+j)*array2(k-j-1) + (k-j-w2)*array2(k-j))
           array3(k-j) = div*((w3+j)*array3(k-j-1) + (k-j-w3)*array3(k-j))
        enddo
        array1(1) = div*(one-w1)*array1(1)
        array2(1) = div*(one-w2)*array2(1)
        array3(1) = div*(one-w3)*array3(1)
     enddo

     !--- perform standard b-spline differentiation
     darray1(1) = -array1(1)
     darray2(1) = -array2(1)
     darray3(1) = -array3(1)
     do j = 2,order
        darray1(j) = array1(j-1) - array1(j)
        darray2(j) = array2(j-1) - array2(j)
        darray3(j) = array3(j-1) - array3(j)
     enddo

     !--- one more recursion
     div = one / (order-1)
     array1(order) = div*w1*array1(order-1)
     array2(order) = div*w2*array2(order-1)
     array3(order) = div*w3*array3(order-1)
     do j = 1,order-2
        array1(order-j) = div*((w1+j)*array1(order-j-1) + (order-j-w1)*array1(order-j))
        array2(order-j) = div*((w2+j)*array2(order-j-1) + (order-j-w2)*array2(order-j))
        array3(order-j) = div*((w3+j)*array3(order-j-1) + (order-j-w3)*array3(order-j))
     enddo

     array1(1) = div*(one-w1)*array1(1)
     array2(1) = div*(one-w2)*array2(1)
     array3(1) = div*(one-w3)*array3(1)

  end select

  return
end subroutine ADD_PREC(calc_bspline)

#else /* (domdec)*/
subroutine ADD_PREC(dummy)
  return
end subroutine ADD_PREC(dummy)
#endif /* (domdec)*/

!  ##ENDEX (exp_precision)
