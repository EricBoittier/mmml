module colfft_util
  ! most of the use statements are here to satisfy dependency scanning
  ! because colfft_util does not get scanned for dependencies
  use chm_kinds
  use consta
#if KEY_DOMDEC_GPU == 1
  use domdec_common
  use domdec_util_gpu_mod
#endif
#if KEY_FFTW==1
  use fftw3
#endif
  use memory
  use new_timer
  use parallel
  use pmeutil
  use stream
  
#if KEY_MKL==1
  use, intrinsic :: iso_c_binding
#endif
  
  implicit none

  private

#if KEY_MKL==1
  include 'fftw3.f'
  include 'fftw3_mkl.f'
  ! This part copied from:
  ! https://software.intel.com/en-us/articles/how-to-set-number-of-users-threads-while-using-intel-mkl-fftw3-wrappers-3
!DIR$ ATTRIBUTES ALIGN : 8 :: fftw3_mkl
  COMMON/fftw3_mkl/ignore(4),mkl_dft_number_of_user_threads,ignore2(7)
  INTEGER*4 :: ignore, mkl_dft_number_of_user_threads, ignore2 
  BIND (c) :: /fftw3_mkl/
#endif 

!IF COLFFT (columnfft)
  integer,parameter :: NDIRECTIONS = 3
  integer,parameter :: X_DIRECTION = 1
  integer,parameter :: Y_DIRECTION = 2
  integer,parameter :: Z_DIRECTION = 3
  integer,dimension(1:NDIRECTIONS) :: fftdim
  integer,parameter :: CONTIGUOUS_INFIMUM = 0
  integer,parameter :: NRECT_COORS=2
  integer,parameter :: FIRST_RECT_COOR=1
  integer,parameter :: SECOND_RECT_COOR=2
  integer,parameter ::   NPARTITIONS    = 3
  integer,parameter ::   YZ_X_PARTITION = 1
  integer,parameter ::   ZX_Y_PARTITION = 2
  integer,parameter ::   XY_Z_PARTITION = 3

  ! dec = decomposed direction
  ! con = contiguous direction
  !
  ! gridmin, gridmax = min/max of global grid coordinates
  ! halomin, halomax = min/max of halo
  !
  ! size = size in real numbers
  ! csize = size in complex numbers
  integer, allocatable, dimension(:,:,:) :: dec_gridmin, dec_gridmax
  integer, allocatable, dimension(:,:,:) :: dec_halomin, dec_halomax
  integer, allocatable, dimension(:,:,:) :: dec_size, dec_csize
  integer   con_gridmin( NPARTITIONS )
  integer   con_gridmax( NPARTITIONS )
  integer   con_halomin( NPARTITIONS )
  integer   con_halomax( NPARTITIONS )
  integer   con_size( NPARTITIONS )
  integer   con_csize( NPARTITIONS )
  integer   con_fftdim( NPARTITIONS )
  integer,parameter :: NCOMMGROUPS    = 2
  integer,parameter :: YZ_X_COMM_ZX_Y = 1
  integer,parameter :: XY_Z_COMM_ZX_Y = 2
  integer, allocatable, dimension(:,:) :: comm_group_begin,comm_group_end
  integer,dimension( NCOMMGROUPS) :: comm_group_stride
  integer,dimension( NCOMMGROUPS ) ::  preserved_coor,altered_coor
  integer,parameter ::   TRANSFORM_BACKWARD = -1, TRANSFORM_FORWARD = 1
  real(chm_real), dimension(:),pointer :: recv_buffer_pd, send_buffer_pd
  real(chm_real), dimension(:),allocatable :: transposed_data_pd
  real(chm_real4), dimension(:),pointer :: recv_buffer_ps, send_buffer_ps
  real(chm_real4), dimension(:),allocatable :: transposed_data_ps
#if KEY_FFTW==1 || KEY_MKL==1
#else /**/
  real(chm_real), dimension(:),allocatable,save :: &
       fft_table_1, fft_table_2, fft_table_3, &
       alpha_rcfft, beta_rcfft
  real(chm_real), dimension(:,:),allocatable,save :: tmp_rc
#endif 

#if KEY_PARALLEL==1
  ! Communication buffers for transpose and ftranspose routines
  ! These are allocated in colfft_util_init
  integer(chm_int4), allocatable, dimension(:) :: pe_list, recv_request, recv_offset, send_request
  integer(chm_int4), allocatable, dimension(:,:) :: recv_status, send_status
#endif 

#if KEY_FFTW==1 || KEY_MKL==1
#if KEY_FFTW==1
  type(C_PTR) fftw_xf_plan, fftw_xb_plan
  type(C_PTR) fftw_yf_plan, fftw_yb_plan
  type(C_PTR) fftw_zf_plan, fftw_zb_plan
#else /**/
  integer*8 fftw_xf_plan, fftw_xb_plan
  integer*8 fftw_yf_plan, fftw_yb_plan
  integer*8 fftw_zf_plan, fftw_zb_plan
#endif 
  integer :: plan_type = 0   ! 1=double, 2=single
  integer :: n_fftw_xf_plan = 0, n_fftw_xb_plan = 0
  integer :: n_fftw_yf_plan = 0, n_fftw_yb_plan = 0
  integer :: n_fftw_zf_plan = 0, n_fftw_zb_plan = 0
  integer(c_int), parameter :: fftw_opt = FFTW_ESTIMATE
  public reset_fftw_plans
#endif 

  integer,parameter :: int_mpi = 4
  integer ny_box, nz_box
  integer :: numtasks_save = -1, nthread_save = -1
  logical :: q_single_save = .false.

  ! tile buffer for transpose_yzx and transpose_zxy -subroutines
  ! NOTE: tilebuf_pd and tilebuf_ps start from 0
  integer, parameter :: tiledim = 64
  real(chm_real), allocatable, dimension(:) :: tilebuf_pd
  real(chm_real4), allocatable, dimension(:) :: tilebuf_ps

  ! Public subroutines
  public check_fft_limit_overlap, backward_rc_fft, forward_rc_fft
  public colfft_util_init, colfft_util_uninit, coord_to_grid
  public get_fft_limits, get_spatial_limits, get_fft_sizes, get_spatial_sizes
  public get_halo_limits, filter_atom
  public calc_column_num

  ! Public functions
  public q_ydim_periodic, q_zdim_periodic

  ! Public variables
  public ny_box, nz_box
  public YZ_X_PARTITION,ZX_Y_PARTITION,XY_Z_PARTITION

  interface coord_to_grid
     module procedure coord3_to_grid3
     module procedure coord3_to_grid3_ps
     module procedure coord3_to_grid2
     module procedure coord3_to_grid2_ps
  end interface

  interface backward_rc_fft
     module procedure backward_rc_fft_pd
     module procedure backward_rc_fft_ps
  end interface

  interface forward_rc_fft
     module procedure forward_rc_fft_pd
     module procedure forward_rc_fft_ps
  end interface

contains

  ! *
  ! * Returns .true. if y fft dimension is periodic
  ! *
  logical function q_ydim_periodic(forder)
    use parallel,only:mynod
    implicit none
    ! Input
    integer, intent(in) :: forder
    ! Variables
    integer xgridmin,xgridmax,ygridmin,ygridmax,zgridmin,zgridmax

    call get_spatial_limits(YZ_X_PARTITION, &
         xgridmin,xgridmax,ygridmin,ygridmax,  &
         zgridmin,zgridmax,mynod)

    if (ny_box == 1 .or. ygridmin >= forder-1) then
       q_ydim_periodic = .false.
    else
       q_ydim_periodic = .true.
    endif

    return
  end function q_ydim_periodic

  ! *
  ! * Returns .true. if z fft dimension is periodic
  ! *
  logical function q_zdim_periodic(forder)
    use parallel,only:mynod
    implicit none
    ! Input
    integer, intent(in) :: forder
    ! Variables
    integer xgridmin,xgridmax,ygridmin,ygridmax,zgridmin,zgridmax

    call get_spatial_limits(YZ_X_PARTITION, &
         xgridmin,xgridmax,ygridmin,ygridmax,  &
         zgridmin,zgridmax,mynod)

    if (nz_box == 1 .or. zgridmin >= forder-1) then
       q_zdim_periodic = .false.
    else
       q_zdim_periodic = .true.
    endif

    return
  end function q_zdim_periodic

  !-----------------------------------------------------------------------
  !           GET_FFT_LIMITS
  !-----------------------------------------------------------------------
  Subroutine get_fft_limits(partition, &
       gridmin0,gridmax0,gridmin1,gridmax1,gridmin2,gridmax2, &
       mytaskid  )
    implicit none
    integer,intent(in) :: mytaskid,partition
    integer,intent(out) :: gridmin0,gridmax0,gridmin1, &
         gridmax1,gridmin2,gridmax2
    gridmin0 = con_gridmin( partition )
    gridmax0 = con_gridmax( partition )
    gridmin1 = dec_gridmin(1,mytaskid,partition)
    gridmax1 = dec_gridmax(1,mytaskid,partition)
    gridmin2 = dec_gridmin(2,mytaskid,partition)
    gridmax2 = dec_gridmax(2,mytaskid,partition)
  end Subroutine get_fft_limits
  
  ! *
  ! * Similar to get_fft_limits but returns the spatial (not fft) limits
  ! *
  Subroutine get_spatial_limits(partition, &
       gridmin0,gridmax0,gridmin1,gridmax1,gridmin2,gridmax2, &
       mytaskid  )
    implicit none
    integer,intent(in) :: mytaskid,partition
    integer,intent(out) :: gridmin0,gridmax0,gridmin1, &
         gridmax1,gridmin2,gridmax2
    gridmin0 = con_gridmin( partition )
    if (partition == YZ_X_PARTITION) then
       gridmax0 = 2*con_gridmax( partition ) - 1
    else
       gridmax0 = con_gridmax( partition )
    endif
    gridmin1 = dec_gridmin(1,mytaskid,partition)
    gridmax1 = dec_gridmax(1,mytaskid,partition)
    gridmin2 = dec_gridmin(2,mytaskid,partition)
    gridmax2 = dec_gridmax(2,mytaskid,partition)
  end Subroutine get_spatial_limits

  ! *
  ! * Similar to halo (spatial) limits
  ! *
  subroutine get_halo_limits(partition, &
       halomin0,halomax0,halomin1,halomax1,halomin2,halomax2, &
       mytaskid  )
    implicit none
    integer,intent(in) :: mytaskid,partition
    integer,intent(out) :: halomin0,halomax0,halomin1,halomax1,halomin2,halomax2
    halomin0 = con_halomin( partition )
    halomax0 = con_halomax( partition )
    halomin1 = dec_halomin(1,mytaskid,partition)
    halomax1 = dec_halomax(1,mytaskid,partition)
    halomin2 = dec_halomin(2,mytaskid,partition)
    halomax2 = dec_halomax(2,mytaskid,partition)
  end subroutine get_halo_limits

  ! *
  ! * Returns fft grid sizes
  ! *
  subroutine get_fft_sizes(partition, gridsize0, gridsize1, gridsize2, mytaskid)
    implicit none
    ! Input / Output
    integer, intent(in) :: partition, mytaskid
    integer, intent(out) :: gridsize0, gridsize1, gridsize2

    gridsize0 = con_csize( partition )
    gridsize1 = dec_csize(1,mytaskid,partition)
    gridsize2 = dec_csize(2,mytaskid,partition)

    return
  end subroutine get_fft_sizes

  ! *
  ! * Returns spatial grid sizes
  ! *
  subroutine get_spatial_sizes(partition, gridsize0, gridsize1, gridsize2, mytaskid)
    implicit none
    ! Input / Output
    integer, intent(in) :: partition, mytaskid
    integer, intent(out) :: gridsize0, gridsize1, gridsize2

    gridsize0 = con_size( partition )
    gridsize1 = dec_size(1,mytaskid,partition)
    gridsize2 = dec_size(2,mytaskid,partition)

    return
  end subroutine get_spatial_sizes

  ! *
  ! * Filter atom for a single coordinate direction:
  ! * Returns .true. if atom is within the fft grid spacing
  ! * Returns .false. otherwise
  ! *
  ! * Applies periodic boundary conditions if cdim is set to grid dimension
  ! *
  ! * For example for y-direction, accept atoms that have:
  ! * y <= ymin <= yp .or. y <= ymax <= yp .or.   (box crosses grid boundary)
  ! * ymin <= y <= ymax .or. ymin <= yp <= ymax   (box within grid boundaries)
  ! *
  logical function filter_atom(c, forder, cmin, cmax, cdim)
    implicit none
    ! Input
    integer, intent(in) :: c, forder, cmin, cmax, cdim
    ! Variables
    integer ct, dc, cp

    ! Apply periodic boundary conditions, result in temporary variable ct
    dc = ((c + forder - 1)/cdim)*cdim
    ct = c - dc

    cp = ct + forder - 1

    filter_atom = ((cmin >= ct .and. cmin <= cp) .or. (cmax >= ct .and. cmax <= cp) .or. &
         (ct >= cmin .and. ct <= cmax) .or. (cp >= cmin .and. cp <= cmax))

    return
  end function filter_atom

  ! *
  ! * Deallocates data needed for fft and transposes
  ! *
  subroutine deallocate_fft_data()
    use memory,only:chmdealloc
    implicit none

    if (allocated(transposed_data_pd)) then
       call chmdealloc('colfft_util.src','deallocate_fft_data','transposed_data_pd',&
            size(transposed_data_pd),crl=transposed_data_pd)
    endif

    if (associated(recv_buffer_pd)) then
       call chmdealloc('colfft_util.src','deallocate_fft_data','recv_buffer_pd',&
            size(recv_buffer_pd),mcrlp=recv_buffer_pd)
    endif

    if (associated(send_buffer_pd)) then
       call chmdealloc('colfft_util.src','deallocate_fft_data','send_buffer_pd',&
            size(send_buffer_pd),mcrlp=send_buffer_pd)
    endif

    if (allocated(transposed_data_ps)) then
       call chmdealloc('colfft_util.src','deallocate_fft_data','transposed_data_ps',&
            size(transposed_data_ps),cr4=transposed_data_ps)
    endif

    if (associated(recv_buffer_ps)) then
       call chmdealloc('colfft_util.src','deallocate_fft_data','recv_buffer_ps',&
            size(recv_buffer_ps),mcr4p=recv_buffer_ps)
    endif

    if (associated(send_buffer_ps)) then
       call chmdealloc('colfft_util.src','deallocate_fft_data','send_buffer_ps',&
            size(send_buffer_ps),mcr4p=send_buffer_ps)
    endif

#if KEY_FFTW==1 || KEY_MKL==1
#else /**/
    if (allocated(fft_table_1)) then
       call chmdealloc('colfft_util.src','deallocate_fft_data','fft_table_1',&
            size(fft_table_1),crl=fft_table_1)
    endif

    if (allocated(fft_table_2)) then
       call chmdealloc('colfft_util.src','deallocate_fft_data','fft_table_2',&
            size(fft_table_2),crl=fft_table_2)
    endif

    if (allocated(fft_table_3)) then
       call chmdealloc('colfft_util.src','deallocate_fft_data','fft_table_3',&
            size(fft_table_3),crl=fft_table_3)
    endif

    if (allocated(alpha_rcfft)) then
       call chmdealloc('colfft_util.src','deallocate_fft_data','alpha_rcfft',&
            size(alpha_rcfft),crl=alpha_rcfft)
    endif

    if (allocated(beta_rcfft)) then
       call chmdealloc('colfft_util.src','deallocate_fft_data','beta_rcfft',&
            size(beta_rcfft),crl=beta_rcfft)
    endif

    if (allocated(tmp_rc)) then
       call chmdealloc('colfft_util.src','deallocate_fft_data','tmp_rc',&
            size(tmp_rc,1),size(tmp_rc,2),crl=tmp_rc)
    endif
#endif 

    return
  end subroutine deallocate_fft_data

#define COLFFT_PREC PS
#define SINGLEP 1
#include <colfft_util.inc>
#undef SINGLEP
#undef COLFFT_PREC

#define COLFFT_PREC PD
#define DOUBLEP 1
#include <colfft_util.inc>
#undef DOUBLEP
#undef COLFFT_PREC

  !-----------------------------------------------------------------------
  !           FFT_UNINIT
  !-----------------------------------------------------------------------
  subroutine colfft_util_uninit()
    use memory,only:chmdealloc
    implicit none

    numtasks_save = -1

    if (allocated(dec_gridmin)) then
       call chmdealloc('colfft_util.src','colfft_util_uninit','dec_gridmin',&
            size(dec_gridmin,1), size(dec_gridmin,2), size(dec_gridmin,3),intg=dec_gridmin)
    endif
    
    if (allocated(dec_gridmax)) then
       call chmdealloc('colfft_util.src','colfft_util_uninit','dec_gridmax',&
            size(dec_gridmax,1), size(dec_gridmax,2), size(dec_gridmax,3),intg=dec_gridmax)
    endif

    if (allocated(dec_halomin)) then
       call chmdealloc('colfft_util.src','colfft_util_uninit','dec_halomin',&
            size(dec_halomin,1), size(dec_halomin,2), size(dec_halomin,3),intg=dec_halomin)
    endif

    if (allocated(dec_halomax)) then
       call chmdealloc('colfft_util.src','colfft_util_uninit','dec_halomax',&
            size(dec_halomax,1), size(dec_halomax,2), size(dec_halomax,3),intg=dec_halomax)
    endif

    if (allocated(dec_size)) then
       call chmdealloc('colfft_util.src','colfft_util_uninit','dec_size',&
            size(dec_size,1), size(dec_size,2), size(dec_size,3),intg=dec_size)
    endif

    if (allocated(dec_csize)) then
       call chmdealloc('colfft_util.src','colfft_util_uninit','dec_csize',&
            size(dec_csize,1), size(dec_csize,2), size(dec_csize,3),intg=dec_csize)
    endif

    if (allocated(comm_group_begin)) then
       call chmdealloc('colfft_util.src','colfft_util_uninit','comm_group_begin',&
            size(comm_group_begin,1),size(comm_group_begin,2),intg=comm_group_begin)
    endif

    if (allocated(comm_group_end)) then
       call chmdealloc('colfft_util.src','colfft_util_uninit','comm_group_end',&
            size(comm_group_end,1),size(comm_group_end,2),intg=comm_group_end)
    endif

    call deallocate_transpose()

    call deallocate_fft_data()

#if KEY_FFTW==1 || KEY_MKL==1
    call reset_fftw_plans()
#endif 

    call dealloc_tilebuf()

    return
  end subroutine colfft_util_uninit

  ! *
  ! * Calculates the number of columns
  ! *
  subroutine calc_column_num(numtot, nfft1, nfft2, num1, num2)
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec, nx, ny, nz, q_split
#endif 
    implicit none
    ! Input / Output
    integer, intent(in) :: numtot, nfft1, nfft2
    integer, intent(out) :: num1, num2

    if (numtot <= 0) then
       call wrndie(-5, 'calc_column_num','Total number of columns must be positive')
    endif

#if KEY_DOMDEC==1
    if (q_split .or. .not.q_domdec) then
#endif 
       ! APH NOTE: This simple formula maximizes the "squareness" of the columns
       num1 = max(1,ceiling( sqrt(real(numtot*nfft1)/real(nfft2))))
       num2 = numtot/num1
       do while (num1*num2 /= numtot)
          num1 = num1 - 1
          num2 = numtot/num1
       enddo
#if KEY_DOMDEC==1
    else
       ! For NO SPLIT, imitate the way direct nodes are split
       num1 = ny*nx
       num2 = nz
    endif
#endif 

    if (num1*num2 /= numtot) then
       call wrndie(-5, 'calc_column_num', 'Error setting up FFT columns')
    endif

    return
  end subroutine calc_column_num

  ! *
  ! * Converts coordinates (x, y, z) into grid coordinates (fr1, fr2, fr3)
  ! *
  subroutine coord3_to_grid3(x, y, z, recip, fr1, fr2, fr3)
    use number,only:half, two
    use pmeutil,only:nfft1,nfft2,nfft3
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: x, y, z, recip(3,3)
    real(chm_real), intent(out) :: fr1, fr2, fr3
    ! Variables
    real(chm_real) w1, w2, w3

    w1 = x*recip(1,1) + y*recip(2,1) + z*recip(3,1) + two
    w2 = x*recip(1,2) + y*recip(2,2) + z*recip(3,2) + two
    w3 = x*recip(1,3) + y*recip(2,3) + z*recip(3,3) + two

    fr1 = nfft1*(w1 - (anint(w1) - half))
    fr2 = nfft2*(w2 - (anint(w2) - half))
    fr3 = nfft3*(w3 - (anint(w3) - half))

    return
  end subroutine coord3_to_grid3

  ! *
  ! * Converts coordinates (x, y, z) into grid coordinates (fr1, fr2, fr3)
  ! *
  subroutine coord3_to_grid3_ps(x, y, z, recip, fr1, fr2, fr3)
    use pmeutil,only:nfft1,nfft2,nfft3
    implicit none
    ! Input / Output
    real(chm_real4), intent(in) :: x, y, z, recip(3,3)
    real(chm_real4), intent(out) :: fr1, fr2, fr3
    ! Parameters
    real(chm_real4), parameter :: two_sp = 2.0_chm_real4, half_sp = 0.5_chm_real4
    ! Variables
    real(chm_real4) w1, w2, w3

    w1 = x*recip(1,1) + y*recip(2,1) + z*recip(3,1) + two_sp
    w2 = x*recip(1,2) + y*recip(2,2) + z*recip(3,2) + two_sp
    w3 = x*recip(1,3) + y*recip(2,3) + z*recip(3,3) + two_sp

    fr1 = nfft1*(w1 - (anint(w1) - half_sp))
    fr2 = nfft2*(w2 - (anint(w2) - half_sp))
    fr3 = nfft3*(w3 - (anint(w3) - half_sp))

    return
  end subroutine coord3_to_grid3_ps

  ! *
  ! * Converts coordinates (x, y, z) into grid coordinates (fr2, fr3)
  ! *
  subroutine coord3_to_grid2_ps(x, y, z, recip, fr2, fr3)
    use pmeutil,only:nfft2,nfft3
    implicit none
    ! Input / Output
    real(chm_real4), intent(in) :: x, y, z, recip(3,3)
    real(chm_real4), intent(out) :: fr2, fr3
    ! Parameters
    real(chm_real4), parameter :: two_sp = 2.0_chm_real4, half_sp = 0.5_chm_real4
    ! Variables
    real(chm_real4) w2, w3

    w2 = x*recip(1,2) + y*recip(2,2) + z*recip(3,2) + two_sp
    w3 = x*recip(1,3) + y*recip(2,3) + z*recip(3,3) + two_sp

    fr2 = nfft2*(w2 - (anint(w2) - half_sp))
    fr3 = nfft3*(w3 - (anint(w3) - half_sp))

    return
  end subroutine coord3_to_grid2_ps

  ! *
  ! * Converts coordinates (x, y, z) into grid coordinates (fr2, fr3)
  ! *
  subroutine coord3_to_grid2(x, y, z, recip, fr2, fr3)
    use number,only:half, two
    use pmeutil,only:nfft2,nfft3
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: x, y, z, recip(3,3)
    real(chm_real), intent(out) :: fr2, fr3
    ! Variables
    real(chm_real) w2, w3

    w2 = x*recip(1,2) + y*recip(2,2) + z*recip(3,2) + two
    w3 = x*recip(1,3) + y*recip(2,3) + z*recip(3,3) + two

    fr2 = nfft2*(w2 - (anint(w2) - half))
    fr3 = nfft3*(w3 - (anint(w3) - half))

    return
  end subroutine coord3_to_grid2

  ! *
  ! * Make sure left halo of the first column does not overlap with the grid
  ! *
  subroutine check_halo_overlap(nfft, grades, forder, gmin1, gmax1)
    implicit none
    ! Input
    integer, intent(in) :: nfft, grades, forder, gmin1, gmax1
    ! Variables
    integer lhp   ! left halo border position

    lhp = -forder + 1 + nfft

    if (grades > 1 .and. lhp >= gmin1 .and. lhp <= gmax1 ) then
       call wrndie(-5,'<colfft>',&
            'colfft left halo overlaps with the grid: Choose less fft nodes or make grid larger')
    endif

    return
  end subroutine check_halo_overlap


  !-----------------------------------------------------------------------
  !           FFT_INIT
  !-----------------------------------------------------------------------
  subroutine colfft_util_init(numtasks)
    use pmeutil,only:nfft1, nfft2, nfft3, forder
!    use consta
    use psf,only:natom
    use memory,only:chmalloc, chmdealloc
    use stream,only:outu, prnlev
!    use parallel,only:mynod
    use domdec_common,only:q_use_single, nthread
    implicit none
    ! Input
    integer, intent(in) :: numtasks
    ! Variables
    Integer maxfftdim, size_fft_table, size_fft_work
    integer, dimension(1:NDIRECTIONS) :: fftdatalen
    Integer, allocatable, dimension(:) :: xinf, xsup, yinf, ysup, zinf, zsup
    integer, allocatable, dimension(:,:) :: master_gridmin, master_gridmax
    integer, allocatable, dimension(:,:) :: master_halomin, master_halomax
    integer, allocatable, dimension(:,:) :: master_size, master_csize

    if ( mod( nfft1, 2 ) /= 0 ) then
       call wrndie(-5, "<colfft_util>", "fft_init: nfft1 must be even for RealComplex FFT")
    endif

    call realloc_tilebuf(q_use_single())

    if (fftdim(X_DIRECTION) /= nfft1/2 .or. fftdim(Y_DIRECTION) /= nfft2 .or. &
         fftdim(Z_DIRECTION) /= nfft3 .or. numtasks_save /= numtasks .or. &
         nthread_save /= nthread) then

       numtasks_save = numtasks
       nthread_save = nthread

#if KEY_MKL==1
       ! This is done in order to make MKL execute functions thread safe
       mkl_dft_number_of_user_threads = nthread
#endif 

       ! Allocate / Reallocate (dec_gridmin, dec_gridmax, dec_halomin, dec_halomax,
       ! dec_size, dec_csize, comm_group_begin, comm_group_end)
       call alloc_realloc(nrect_coors, numtasks, npartitions, ncommgroups)
       
       fftdim(X_DIRECTION) = nfft1/2
       fftdim(Y_DIRECTION) = nfft2
       fftdim(Z_DIRECTION) = nfft3

       !--- For RC fft, data after x FFT contains half the size of
       !--- the original FFT plus the nyquist frequency so the datlen
       !--- and thus the number of fft's for the y and z is not the same
       !--- as the fftdim for x. The datalen value is the size of the data
       !--- array in each direction.
       fftdatalen(X_DIRECTION) = nfft1/2 + 1
       fftdatalen(Y_DIRECTION) = nfft2
       fftdatalen(Z_DIRECTION) = nfft3
       
       maxfftdim = max( fftdim(X_DIRECTION), fftdim(Y_DIRECTION), fftdim(Z_DIRECTION) )
       size_fft_table = 3*( 4*maxfftdim + 15 )
       size_fft_work  = 3*( 4*maxfftdim + 15 )

       call calc_column_num(numtasks, nfft2, nfft3, ny_box, nz_box)
       
       if (prnlev > 2) then
          write (outu,'(a,i3,a,i4)') 'Splitting recip cores into (y by z): ',ny_box,' by ',nz_box
       endif
       
       ! Allocate temporary storage
       call chmalloc('colfft_util.src','colfft_util_init','master_gridmin',&
            3,numtasks,lbou2=0,intg=master_gridmin)
       call chmalloc('colfft_util.src','colfft_util_init','master_gridmax',&
            3,numtasks,lbou2=0,intg=master_gridmax)
       call chmalloc('colfft_util.src','colfft_util_init','master_halomin',&
            3,numtasks,lbou2=0,intg=master_halomin)
       call chmalloc('colfft_util.src','colfft_util_init','master_halomax',&
            3,numtasks,lbou2=0,intg=master_halomax)
       call chmalloc('colfft_util.src','colfft_util_init','master_size',&
            3,numtasks,lbou2=0,intg=master_size)
       call chmalloc('colfft_util.src','colfft_util_init','master_csize',&
            3,numtasks,lbou2=0,intg=master_csize)
       
       call chmalloc('colfft_util.src','colfft_util_init','xinf',numtasks,intg=xinf)
       call chmalloc('colfft_util.src','colfft_util_init','yinf',numtasks,intg=yinf)
       call chmalloc('colfft_util.src','colfft_util_init','zinf',numtasks,intg=zinf)
       call chmalloc('colfft_util.src','colfft_util_init','xsup',numtasks,intg=xsup)
       call chmalloc('colfft_util.src','colfft_util_init','ysup',numtasks,intg=ysup)
       call chmalloc('colfft_util.src','colfft_util_init','zsup',numtasks,intg=zsup)
       
       !--- Comm groups are the processor groups for each of the transposes
       !--- This call will set up the first, last, and stride of processors for
       !--- each processor in each transpose
       
       call create_comm_groups( ny_box, nz_box )
       
       !--- Define the start and stop of y and z for each y and z  section
       call divide_interval( fftdim(Y_DIRECTION), ny_box, yinf, ysup )
       call divide_interval( fftdim(Z_DIRECTION), nz_box, zinf, zsup )
       
       ! Couple of sanity checks:
       ! Make sure the grid is larger than the spline size
       if (nfft1 < forder .or. nfft2 < forder .or. nfft3 < forder) then
          call wrndie(-5,'<colfft>','Grid size smaller than spline size(order)')
       endif
       ! Make sure halo does not overlap with the grid
       call check_halo_overlap(fftdim(Y_DIRECTION), ny_box, forder, yinf(1), ysup(1))
       call check_halo_overlap(fftdim(Z_DIRECTION), nz_box, forder, zinf(1), zsup(1))
       ! Make sure none of the fft dimensions becomes zero (more checks later in the code)
       if (minval(ysup(1:ny_box) - yinf(1:ny_box)) < 0) then
          call wrndie(-5,'<colfft>',&
               'FFT length in Y direction becomes zero: use less reciprocal nodes or larger FFT grid')
       endif
       if (minval(zsup(1:nz_box) - zinf(1:nz_box)) < 0) then
          call wrndie(-5,'<colfft>',&
               'FFT length in Z direction becomes zero: use less reciprocal nodes or larger FFT grid')
       endif
       
       !--- Define how the data moves in the transposes
       !---  in the YZ_X_COMM_ZX_Y transpose, the z coordinate is preserved.
       !---     That means that the z coord is unchanged for each point and
       !---     only the x and y coord are transposing. In this case the 
       !---     Z coord is the second rectangular dimension of the rectangle
       !---     on the y-z face where the spatial decomposition is defined.
       !---   Data for the first part of the transpose is contiguous in
       !---     the x direction and in full x-vectors. 
       !---   After the transpose,
       !---     data is contiuguous in the y-direction in full y vectors,
       !---     y will be the contiguous direction and x will be the first
       !---     rectangular coordinate.
       preserved_coor( YZ_X_COMM_ZX_Y ) = SECOND_RECT_COOR
       altered_coor( YZ_X_COMM_ZX_Y )   = FIRST_RECT_COOR
       !--- Define as above but for the other transpose where the x coord is 
       !---     preserved and the y and z are transposing.
       !--- In this case, the First and Second rectangular coord are x and z
       !---     before transpose and are x and y afterward.
       preserved_coor( XY_Z_COMM_ZX_Y ) = FIRST_RECT_COOR
       altered_coor( XY_Z_COMM_ZX_Y )   = SECOND_RECT_COOR
       
       !--- Create Partition
       !--- this routine will fill in the following information for the specified
       !--- partition (global to this module)
       !---   dec_gridmin and dec_gridmax for both "inner" and "outer"
       !---                                     grades, meaning altered_coor and
       !---                                     preserved_coor resp. for this 
       !---                                     partition or even better,
       !---                                     2nd and 3rd coord
       !---   con_gridmin and con_gridmax for this partition or 1st coord
       call create_partition( YZ_X_PARTITION, &
            altered_coor(YZ_X_COMM_ZX_Y), ny_box, yinf, ysup, &
            preserved_coor(YZ_X_COMM_ZX_Y), nz_box, zinf, zsup, &
            CONTIGUOUS_INFIMUM, fftdatalen(X_DIRECTION)-1, &
            fftdim(X_DIRECTION) )
       call set_halo_size(YZ_X_PARTITION, forder, &
            altered_coor(YZ_X_COMM_ZX_Y), ny_box, &
            preserved_coor(YZ_X_COMM_ZX_Y), nz_box)
       
       call divide_interval(fftdatalen(X_DIRECTION),ny_box,xinf,xsup)
       if (minval(xsup(1:ny_box) - xinf(1:ny_box)) < 0) then
          call wrndie(-5,'<colfft>',&
               'FFT length in X direction becomes zero: use less reciprocal nodes or larger FFT grid')
       endif
       call create_partition(ZX_Y_PARTITION, &
            preserved_coor(XY_Z_COMM_ZX_Y), ny_box, xinf, xsup, &
            preserved_coor(YZ_X_COMM_ZX_Y), nz_box, zinf, zsup, &
            CONTIGUOUS_INFIMUM, fftdim(Y_DIRECTION) - 1, &
            fftdim(Y_DIRECTION) )
       call set_halo_size(ZX_Y_PARTITION, forder, &
            preserved_coor(XY_Z_COMM_ZX_Y), ny_box, &
            preserved_coor(YZ_X_COMM_ZX_Y), nz_box)
       
       call divide_interval( fftdim(Y_DIRECTION), nz_box, yinf, ysup )
       if (minval(ysup(1:nz_box) - yinf(1:nz_box)) < 0) then
          call wrndie(-5,'<colfft>',&
               'FFT length in Y direction becomes zero: use less reciprocal nodes or larger FFT grid')
       endif
       call create_partition(XY_Z_PARTITION, &
            preserved_coor(XY_Z_COMM_ZX_Y), ny_box, xinf, xsup, &
            altered_coor(XY_Z_COMM_ZX_Y), nz_box, yinf, ysup, &
            CONTIGUOUS_INFIMUM, fftdim(Z_DIRECTION) - 1, &
            fftdim(Z_DIRECTION) )
       call set_halo_size(XY_Z_PARTITION, forder, &
            preserved_coor(XY_Z_COMM_ZX_Y), ny_box, &
            altered_coor(XY_Z_COMM_ZX_Y), nz_box)
       
       ! Deallocate temporary storage
       call chmdealloc('colfft_util.src','colfft_util_init','xinf',numtasks,intg=xinf)
       call chmdealloc('colfft_util.src','colfft_util_init','yinf',numtasks,intg=yinf)
       call chmdealloc('colfft_util.src','colfft_util_init','zinf',numtasks,intg=zinf)
       call chmdealloc('colfft_util.src','colfft_util_init','xsup',numtasks,intg=xsup)
       call chmdealloc('colfft_util.src','colfft_util_init','ysup',numtasks,intg=ysup)
       call chmdealloc('colfft_util.src','colfft_util_init','zsup',numtasks,intg=zsup)
       
       call chmdealloc('colfft_util.src','colfft_util_init','master_gridmin',&
            3,numtasks,intg=master_gridmin)
       call chmdealloc('colfft_util.src','colfft_util_init','master_gridmax',&
            3,numtasks,intg=master_gridmax)
       call chmdealloc('colfft_util.src','colfft_util_init','master_halomin',&
            3,numtasks,intg=master_halomin)
       call chmdealloc('colfft_util.src','colfft_util_init','master_halomax',&
            3,numtasks,intg=master_halomax)
       call chmdealloc('colfft_util.src','colfft_util_init','master_size',&
            3,numtasks,intg=master_size)
       call chmdealloc('colfft_util.src','colfft_util_init','master_csize',&
            3,numtasks,intg=master_csize)
       
#if KEY_FFTW==1 || KEY_MKL==1
       call reset_fftw_plans()
#endif 
    elseif (q_single_save .neqv. q_use_single()) then
       ! If precision changed => reset FFTW plans
#if KEY_FFTW==1 || KEY_MKL==1
       call reset_fftw_plans()
#endif 
    endif

    q_single_save = q_use_single()

    return
    
  contains

    subroutine alloc_realloc(nrect_coors, numtasks, npartitions, ncommgroups)
      use memory,only:chmalloc, chmdealloc
      implicit none
      ! Input
      integer, intent(in) :: nrect_coors, numtasks, npartitions, ncommgroups

      ! dec_gridmin
      if (allocated(dec_gridmin)) then
         if (size(dec_gridmin,1) /= nrect_coors .or. size(dec_gridmin,2) /= numtasks .or. &
              size(dec_gridmin,3) /= npartitions) then
            call chmdealloc('colfft_util.src','colfft_util_init','dec_gridmin',size(dec_gridmin,1), &
                 size(dec_gridmin,2),size(dec_gridmin,3),intg=dec_gridmin)
         endif
      endif
      if (.not.allocated(dec_gridmin)) then
         call chmalloc('colfft_util.src','colfft_util_init','dec_gridmin',nrect_coors,numtasks, &
              npartitions,lbou2=0,intg=dec_gridmin)
      endif

      ! dec_gridmax
      if (allocated(dec_gridmax)) then
         if (size(dec_gridmax,1) /= nrect_coors .or. size(dec_gridmax,2) /= numtasks .or. &
              size(dec_gridmax,3) /= npartitions) then
            call chmdealloc('colfft_util.src','colfft_util_init','dec_gridmax',size(dec_gridmax,1), &
                 size(dec_gridmax,2),size(dec_gridmax,3),intg=dec_gridmax)
         endif
      endif
      if (.not.allocated(dec_gridmax)) then
         call chmalloc('colfft_util.src','colfft_util_init','dec_gridmax',nrect_coors,numtasks, &
              npartitions,lbou2=0,intg=dec_gridmax)
      endif
      
      ! dec_halomin
      if (allocated(dec_halomin)) then
         if (size(dec_halomin,1) /= nrect_coors .or. size(dec_halomin,2) /= numtasks .or. &
              size(dec_halomin,3) /= npartitions) then
            call chmdealloc('colfft_util.src','colfft_util_init','dec_halomin',size(dec_halomin,1), &
                 size(dec_halomin,2),size(dec_halomin,3),intg=dec_halomin)
         endif
      endif
      if (.not.allocated(dec_halomin)) then
         call chmalloc('colfft_util.src','colfft_util_init','dec_halomin',nrect_coors,numtasks, &
              npartitions,lbou2=0,intg=dec_halomin)
      endif

      ! dec_halomax
      if (allocated(dec_halomax)) then
         if (size(dec_halomax,1) /= nrect_coors .or. size(dec_halomax,2) /= numtasks .or. &
              size(dec_halomax,3) /= npartitions) then
            call chmdealloc('colfft_util.src','colfft_util_init','dec_halomax',size(dec_halomax,1), &
                 size(dec_halomax,2),size(dec_halomax,3),intg=dec_halomax)
         endif
      endif
      if (.not.allocated(dec_halomax)) then
         call chmalloc('colfft_util.src','colfft_util_init','dec_halomax',nrect_coors,numtasks, &
              npartitions,lbou2=0,intg=dec_halomax)
      endif

      ! dec_size
      if (allocated(dec_size)) then
         if (size(dec_size,1) /= nrect_coors .or. size(dec_size,2) /= numtasks .or. &
              size(dec_size,3) /= npartitions) then
            call chmdealloc('colfft_util.src','colfft_util_init','dec_size',nrect_coors,numtasks, &
                 npartitions,intg=dec_size)
         endif
      endif
      if (.not.allocated(dec_size)) then
         call chmalloc('colfft_util.src','colfft_util_init','dec_size',nrect_coors,numtasks, &
              npartitions,lbou2=0,intg=dec_size)
      endif

      ! dec_csize
      if (allocated(dec_csize)) then
         if (size(dec_csize,1) /= nrect_coors .or. size(dec_csize,2) /= numtasks .or. &
              size(dec_csize,3) /= npartitions) then
            call chmdealloc('colfft_util.src','colfft_util_init','dec_csize',nrect_coors,numtasks, &
                 npartitions,intg=dec_csize)
         endif
      endif
      if (.not.allocated(dec_csize)) then
         call chmalloc('colfft_util.src','colfft_util_init','dec_csize',nrect_coors,numtasks, &
              npartitions,lbou2=0,intg=dec_csize)
      endif

      ! comm_group_begin
      if (allocated(comm_group_begin)) then
         if (size(comm_group_begin,1) /= numtasks .or. size(comm_group_begin,2) /= ncommgroups) then
            call chmdealloc('colfft_util.src','colfft_util_init','comm_group_begin',&
                 size(comm_group_begin,1),size(comm_group_begin,2),intg=comm_group_begin)
         endif
      endif
      if (.not.allocated(comm_group_begin)) then
         call chmalloc('colfft_util.src','colfft_util_init','comm_group_begin',numtasks,ncommgroups,&
              lbou=0,intg=comm_group_begin)
      endif

      ! comm_group_end
      if (allocated(comm_group_end)) then
         if (size(comm_group_end,1) /= numtasks .or. size(comm_group_end,2) /= ncommgroups) then
            call chmdealloc('colfft_util.src','colfft_util_init','comm_group_end',&
                 size(comm_group_end,1),size(comm_group_end,2),intg=comm_group_end)
         endif
      endif
      if (.not.allocated(comm_group_end)) then
         call chmalloc('colfft_util.src','colfft_util_init','comm_group_end',numtasks,ncommgroups,&
              lbou=0,intg=comm_group_end)
      endif

      return
    end subroutine alloc_realloc

  end subroutine colfft_util_init

  ! *
  ! * Allocates / reallocates tilebuf_pd and tilebuf_ps
  ! *
  subroutine realloc_tilebuf(q_single)
    use domdec_common,only:nthread
    use memory,only:chmalloc,chmdealloc
    implicit none
    ! Input / Output
    logical, intent(in) :: q_single

    if (q_single) then
       if (allocated(tilebuf_pd)) then
          call chmdealloc('colfft_util.src','realloc_tilebuf','tilebuf_pd',&
               size(tilebuf_pd),crl=tilebuf_pd)
       endif
       if (.not.allocated(tilebuf_ps)) then
          call chmalloc('colfft_util.src','realloc_tilebuf','tilebuf_ps',&
               2*nthread*tiledim*tiledim,lbou=0,cr4=tilebuf_ps)
       endif
    else
       if (allocated(tilebuf_ps)) then
          call chmdealloc('colfft_util.src','realloc_tilebuf','tilebuf_ps',&
               size(tilebuf_ps),cr4=tilebuf_ps)
       endif
       if (.not.allocated(tilebuf_pd)) then
          call chmalloc('colfft_util.src','realloc_tilebuf','tilebuf_pd',&
               2*nthread*tiledim*tiledim,lbou=0,crl=tilebuf_pd)
       endif
    endif

    return
  end subroutine realloc_tilebuf

  ! *
  ! * Deallocates tilebuf_pd and tilebuf_ps
  ! *
  subroutine dealloc_tilebuf()
    use memory,only:chmdealloc
    implicit none

    if (allocated(tilebuf_pd)) then
       call chmdealloc('colfft_util.src','dealloc_tilebuf','tilebuf_pd',&
            size(tilebuf_pd),crl=tilebuf_pd)
    endif

    if (allocated(tilebuf_ps)) then
       call chmdealloc('colfft_util.src','dealloc_tilebuf','tilebuf_ps',&
            size(tilebuf_ps),cr4=tilebuf_ps)
    endif

    return
  end subroutine dealloc_tilebuf

  ! *
  ! * Check for overlapping grid limits
  ! *
  logical function check_fft_limit_overlap(bot, top, n, nfft)
    implicit none
    ! Input
    integer, intent(in) :: bot(*), top(*), n, nfft
    ! Variables
    integer i, bot_ext(2), botv
    integer xgridmin, xgridmax, ygridmin, ygridmax, zgridmin, zgridmax

    check_fft_limit_overlap = .false.

    ! Extend bottom
    bot_ext(1) = bot(1) + nfft
    if (n > 1) then
       bot_ext(2) = bot(2) + nfft
    else
       bot_ext(2) = bot_ext(1) + nfft
    endif

    do i=1,n
       if (i+2 > n) then
          botv = bot_ext(i+2-n)
       else
          botv = bot(i+2)
       endif
       if (top(i) >= botv) check_fft_limit_overlap = .true.
    enddo

    return
  end function check_fft_limit_overlap

  !-----------------------------------------------------------------------
  !           CREATE_COMM_GROUPS
  !-----------------------------------------------------------------------
  Subroutine create_comm_groups( inner_grades, outer_grades )
    integer,intent(in) :: inner_grades
    integer,intent(in) :: outer_grades
    integer group_inf
    integer group_sup
    integer i
    integer j
    integer processor
    comm_group_stride(YZ_X_COMM_ZX_Y) = 1
    processor = 0
    do i = 1, outer_grades
       group_inf = processor
       group_sup = group_inf + comm_group_stride(YZ_X_COMM_ZX_Y) * &
            ( inner_grades - 1 )
       do j = 1, inner_grades
          comm_group_begin(processor, YZ_X_COMM_ZX_Y) = group_inf
          comm_group_end(processor, YZ_X_COMM_ZX_Y) = group_sup
          processor = processor + comm_group_stride(YZ_X_COMM_ZX_Y)
       enddo
    enddo
    comm_group_stride(XY_Z_COMM_ZX_Y) = inner_grades
    do j = 1, inner_grades
       processor = j - 1
       group_inf = processor
       group_sup = group_inf + comm_group_stride(XY_Z_COMM_ZX_Y) * &
            ( outer_grades - 1 )
       do i = 1, outer_grades
          comm_group_begin(processor, XY_Z_COMM_ZX_Y) = group_inf
          comm_group_end(processor, XY_Z_COMM_ZX_Y) = group_sup
          processor = processor + comm_group_stride(XY_Z_COMM_ZX_Y)
       enddo
    enddo
  End subroutine create_comm_groups

  ! *
  ! * Prints partition info on screen
  ! *
  subroutine print_partition_info(partition, inner_grades, outer_grades, recip_node)
    use stream,only:outu
    implicit none
    ! Input
    integer, intent(in) :: partition, inner_grades, outer_grades, recip_node
    ! Variables
    integer i, j, cpu, ind(3)

    if (partition == YZ_X_PARTITION) then
       write (outu,'(a,i3)') 'YZ_X_PARTITION info, recip_node=',recip_node
    elseif (partition == XY_Z_PARTITION) then
       write (outu,'(a,i3)') 'XY_Z_PARTITION info, recip_node=',recip_node
    elseif (partition == ZX_Y_PARTITION) then
       write (outu,'(a,i3)') 'ZX_Y_PARTITION info, recip_node=',recip_node
    endif

    write (outu,'(a,2i4)') 'con_gridmin/max =',con_gridmin(partition),con_gridmax(partition)
    write (outu,'(a,2i4)') 'con_halomin/max =',con_halomin(partition),con_halomax(partition)
    write (outu,'(a,2i4)') 'con_size/csize  =',con_size(partition),con_csize(partition)

    if (partition == YZ_X_PARTITION) then
       write (outu,'(a)') '                          Y       Z'
    elseif (partition == XY_Z_PARTITION) then
       write (outu,'(a)') '                          X       Y'
    elseif (partition == ZX_Y_PARTITION) then
       write (outu,'(a)') '                          Z       X'
    endif

    cpu = 0
    do i=1,outer_grades
       do j=1,inner_grades

          write (outu,'(i3,a,4i4)') cpu,': dec_gridmin/max=',&
               dec_gridmin(1,cpu,partition),dec_gridmax(1,cpu,partition),&
               dec_gridmin(2,cpu,partition),dec_gridmax(2,cpu,partition)

          write (outu,'(i3,a,4i4)') cpu,': dec_halomin/max=',&
               dec_halomin(1,cpu,partition),dec_halomax(1,cpu,partition),&
               dec_halomin(2,cpu,partition),dec_halomax(2,cpu,partition)

          write (outu,'(i3,a,4i4)') cpu,': dec_size/csize =',&
               dec_size(1,cpu,partition),dec_csize(1,cpu,partition),&
               dec_size(2,cpu,partition),dec_csize(2,cpu,partition)

          cpu = cpu + 1
       enddo
    enddo

    return
  end subroutine print_partition_info

  !-----------------------------------------------------------------------
  !           CREATE_PARTITION
  !-----------------------------------------------------------------------
  Subroutine create_partition( partition, &
       inner_coor, inner_grades, inner_min, inner_max, &
       outer_coor, outer_grades, outer_min, outer_max, &
       local_min, local_max,local_fftdim )
    use chm_kinds
    use pmeutil,only:forder
    implicit none
    integer,intent(in) :: partition,inner_coor, inner_grades, &
         outer_coor, outer_grades
    integer,intent(in),dimension(inner_grades) :: inner_min,inner_max
    integer,intent(in),dimension(outer_grades) :: outer_min,outer_max
    integer,intent(in) :: local_min, local_max, local_fftdim
    !----- local --------------------
    integer :: i,j,processor

    processor = 0
    do i = 1, outer_grades
       do j = 1, inner_grades
          dec_gridmin(inner_coor,processor,partition)=inner_min(j)
          dec_gridmax(inner_coor,processor,partition)=inner_max(j)

          dec_gridmin(outer_coor,processor,partition)=outer_min(i)
          dec_gridmax(outer_coor,processor,partition)=outer_max(i)

          processor = processor + 1
       enddo
    enddo
    con_gridmin(partition) = local_min
    con_gridmax(partition) = local_max
    con_fftdim(partition) = local_fftdim
    return
  End subroutine create_partition

  ! *
  ! * Set halomin/halomax and size/csize
  ! * NOTE: halo only exists for YZ_X_PARTITION -partition
  ! *
  subroutine set_halo_size(partition, forder, inner_coor, inner_grades, &
       outer_coor, outer_grades)
    implicit none
    ! Input
    integer, intent(in) :: partition, forder
    integer, intent(in) :: inner_coor, inner_grades, outer_coor, outer_grades
    ! Variables
    integer i, j, cpu, halo

    if (partition == YZ_X_PARTITION) then
       halo = forder - 1
    else
       halo = 0
    endif

    cpu = 0
    do i = 1, outer_grades
       do j = 1, inner_grades
          if (inner_grades > 1) then
             dec_halomin(inner_coor,cpu,partition)=&
                  dec_gridmin(inner_coor,cpu,partition) - halo
          else
             dec_halomin(inner_coor,cpu,partition)=&
                  dec_gridmin(inner_coor,cpu,partition)
          endif
          dec_halomax(inner_coor,cpu,partition)=&
               dec_gridmax(inner_coor,cpu,partition) + halo

          if (outer_grades > 1) then
             dec_halomin(outer_coor,cpu,partition)=&
                  dec_gridmin(outer_coor,cpu,partition) - halo
          else
             dec_halomin(outer_coor,cpu,partition)=&
                  dec_gridmin(outer_coor,cpu,partition)
          endif

          dec_halomax(outer_coor,cpu,partition)=&
               dec_gridmax(outer_coor,cpu,partition) + halo

          if (partition == ZX_Y_PARTITION .or. partition == XY_Z_PARTITION) then
             dec_csize(inner_coor,cpu,partition) = &
                  dec_halomax(inner_coor,cpu,partition) - &
                  dec_halomin(inner_coor,cpu,partition) + 1
             dec_size(inner_coor,cpu,partition) = &
                  dec_csize(inner_coor,cpu,partition)*2
          else
             dec_size(inner_coor,cpu,partition) = &
                  dec_halomax(inner_coor,cpu,partition) - &
                  dec_halomin(inner_coor,cpu,partition) + 1
             dec_size(inner_coor,cpu,partition) = &
                  ((dec_size(inner_coor,cpu,partition) - 1)/2 + 1)*2
             dec_csize(inner_coor,cpu,partition) = &
                  dec_size(inner_coor,cpu,partition)
          endif


          dec_size(outer_coor,cpu,partition) = &
               dec_halomax(outer_coor,cpu,partition) - &
               dec_halomin(outer_coor,cpu,partition) + 1
          dec_size(outer_coor,cpu,partition) = &
               ((dec_size(outer_coor,cpu,partition) - 1)/2 + 1)*2
          dec_csize(outer_coor,cpu,partition) = dec_size(outer_coor,cpu,partition)


          cpu = cpu + 1
       enddo
    enddo
    
    if (partition == YZ_X_PARTITION) then
       con_halomin(partition) = con_gridmin(partition)
       con_halomax(partition) = con_gridmax(partition)*2-1 + halo
       con_size(partition) = con_halomax(partition) - con_halomin(partition) + 1
       con_size(partition) = ((con_size(partition)-1)/2+1)*2
       con_csize(partition) = con_size(partition)/2
    else
       con_halomin(partition) = con_gridmin(partition)
       con_halomax(partition) = con_gridmax(partition)
       con_size(partition) = con_halomax(partition) - con_halomin(partition) + 1
       con_size(partition) = ((con_size(partition)-1)/2+1)*2
       con_csize(partition) = con_size(partition)
    endif

    return
  end subroutine set_halo_size

  !-----------------------------------------------------------------------
  !           DIVIDE_INTERVAL
  !-----------------------------------------------------------------------
  Subroutine divide_interval( npoints, ngrades, inf, sup )
    implicit none
    integer,intent(in)  :: npoints
    integer,intent(in)  :: ngrades
    integer,intent(out) :: inf(ngrades)
    integer,intent(out) :: sup(ngrades)
    integer i
    integer increment
    integer residue
    increment = npoints / ngrades
    residue   = mod( npoints, ngrades )
    sup(ngrades) = npoints - 1
    if(ngrades > 1 ) then
       do i = ngrades - 1, 1, -1
          sup(i) = sup(i + 1) - increment
          if ( residue > 0 ) then
             residue = residue - 1
             sup(i)  = sup(i) - 1
          endif
          inf(i + 1)  = sup(i) + 1
       enddo
    endif
    inf(1) = sup(1) - increment + 1
  End subroutine divide_interval

#if KEY_FFTW==1 || KEY_MKL==1
#else /**/
  !-----------------------------------------------------------------------
  !           FFT_1D_REALCOMPLEX
  !-----------------------------------------------------------------------
  Subroutine fft_1d_realcomplex( isign, length, data, table,ifac )
    use pmeutil,only:cfftf,cfftb
    use number
    implicit none
    integer,intent(in) :: isign,length
    integer,dimension(1:15) :: ifac
    real(chm_real) data(0:*)
    real(chm_real), dimension(1:*) :: table
    real(chm_real) a
    real(chm_real) b
    real(chm_real) c
    real(chm_real) d
    integer i

    if ( isign == TRANSFORM_BACKWARD ) then
       do i = 0, length - 1
          tmp_rc(1,i)= data(2*i)
          tmp_rc(2,i)= data(2*i+1)
       enddo
       call cfftf(length, tmp_rc(1,0), table,ifac)
       do i = 1, length - 1
          a =  Half*(tmp_rc(1,i)+tmp_rc(1,length-i))
          b =  Half*(tmp_rc(2,i)-tmp_rc(2,length-i))
          c =  Half*(tmp_rc(2,i)+tmp_rc(2,length-i))
          d = -Half*(tmp_rc(1,i)-tmp_rc(1,length-i))
          data(2*i)   = a + alpha_rcfft(i)*c + beta_rcfft(i)*d
          data(2*i+1) = b + alpha_rcfft(i)*d - beta_rcfft(i)*c
       enddo
       data(0)  = tmp_rc(1,0) + tmp_rc(2,0)
       data(1)  = zero
       data(2*length) = tmp_rc(1,0) - tmp_rc(2,0)
       data(2*length+1) = zero
    else if ( isign == TRANSFORM_FORWARD ) then
       do i = 1, length - 1
          a =   data(2*i) +  data(2*(length-i))
          b =   data(2*i+1) - data(2*(length-i)+1)
          c =   data(2*i+1) + data(2*(length-i)+1)
          d =   data(2*i) -  data(2*(length-i))
          tmp_rc(1,i) = a - alpha_rcfft(i)*c - beta_rcfft(i)*d
          tmp_rc(2,i) = b + alpha_rcfft(i)*d - beta_rcfft(i)*c
       enddo
       tmp_rc(1,0) = data(0) + data(2*length)
       tmp_rc(2,0) = data(0) - data(2*length)
       call cfftb(length, tmp_rc(1,0), table,ifac)
       do i = 0, length - 1
          data(2*i)   =   tmp_rc(1,i)
          data(2*i+1) =   tmp_rc(2,i)
       enddo
    endif
  End subroutine fft_1d_realcomplex

  !-----------------------------------------------------------------------
  !           FFT_1D_CC
  !-----------------------------------------------------------------------
  Subroutine fft_1d_cc( isign, length, fft_table, ddata,ifac )
    use pmeutil,only:cfftf,cfftb
    implicit none
    real(chm_real), dimension(1:*) :: fft_table
    integer :: isign,length,ifac(15)
    real(chm_real) :: ddata(2*length)
    if ( isign == TRANSFORM_BACKWARD ) then
       call cfftf( length, ddata, fft_table,ifac )
    else if ( isign == TRANSFORM_FORWARD ) then
       call cfftb( length, ddata, fft_table,ifac )
    endif
    return
  End subroutine fft_1d_cc
#endif 

!!$  !-----------------------------------------------------------------------
!!$  !           LOCATION function
!!$  !-----------------------------------------------------------------------
!!$  integer function location( k_length, k_inf, k, &
!!$       j_length, j_inf, j, &
!!$       i_length, i_inf, i )
!!$    implicit none
!!$    integer  k_length
!!$    integer  k_inf
!!$    integer  k
!!$    integer  j_length
!!$    integer  j_inf
!!$    integer  j
!!$    integer  i_length
!!$    integer  i_inf
!!$    integer  i
!!$    location = j_length * i_length * ( k - k_inf ) + &
!!$         i_length * ( j - j_inf ) + &
!!$         ( i - i_inf )
!!$  End function location

  ! *
  ! * Deallocates temporary buffers needed for transpose() and ftranspose() -subroutines
  ! *
  subroutine deallocate_transpose()
#if KEY_PARALLEL==1
    use mpi,only:mpi_status_size  
#endif
    use memory
    implicit none

#if KEY_PARALLEL==1
    if (allocated(pe_list)) then
       call chmdealloc('colfft_util.src','deallocate_transpose','pe_list',&
            size(pe_list),ci4=pe_list)
    endif

    if (allocated(recv_request)) then
       call chmdealloc('colfft_util.src','deallocate_transpose','recv_request',&
            size(recv_request),ci4=recv_request)
    endif

    if (allocated(recv_offset)) then
       call chmdealloc('colfft_util.src','deallocate_transpose','recv_offset',&
            size(recv_offset),ci4=recv_offset)
    endif

    if (allocated(send_request)) then
       call chmdealloc('colfft_util.src','deallocate_transpose','send_request',&
            size(send_request),ci4=send_request)
    endif

    if (allocated(recv_status)) then
       call chmdealloc('colfft_util.src','deallocate_transpose','recv_status',&
            MPI_STATUS_SIZE,size(recv_status,2),ci4=recv_status)
    endif

    if (allocated(send_status)) then
       call chmdealloc('colfft_util.src','deallocate_transpose','send_status',&
            MPI_STATUS_SIZE,size(send_status,2),ci4=send_status)
    endif
#endif 

    return
  end subroutine deallocate_transpose

#if KEY_FFTW==1 || KEY_MKL==1
  ! *
  ! * Deletes a single plan
  ! *
  subroutine delete_plan(plan, n_plan)
    implicit none
    ! Input / Output
#if KEY_FFTW==1
    type(c_ptr), intent(inout) :: plan
#else /**/
    integer*8, intent(inout) :: plan
#endif 
    integer, intent(out) :: n_plan

    if (n_plan > 0) then
       if (plan_type == 1) then
          call dfftw_destroy_plan(plan)
       elseif (plan_type == 2) then
#if KEY_COLFFT_NOSP==0
          call sfftw_destroy_plan(plan)
#else /**/
          call wrndie(-5,'<colfft_util>','FFTW not compiled in single precision')
#endif 
       else
          call wrndie(-5,'<colfft_util>','delete_plan: Invalid plan_type')
       endif
    endif
    n_plan = 0

    return
  end subroutine delete_plan

  ! *
  ! * Deletes all FFTW plans
  ! *
  subroutine reset_fftw_plans()
    implicit none

    call delete_plan(fftw_xf_plan, n_fftw_xf_plan)
    call delete_plan(fftw_xb_plan, n_fftw_xb_plan)

    call delete_plan(fftw_yf_plan, n_fftw_yf_plan)
    call delete_plan(fftw_yb_plan, n_fftw_yb_plan)

    call delete_plan(fftw_zf_plan, n_fftw_zf_plan)
    call delete_plan(fftw_zb_plan, n_fftw_zb_plan)

  end subroutine reset_fftw_plans
#endif 

  ! *
  ! * Translates
  ! *
  subroutine translate_q_real(q, tx, ty, tz)
    use parallel,only:mynod
    use memory
    implicit none
    ! Input / Output
    real(chm_real), intent(inout) :: q(0:*)
    integer, intent(in) :: tx, ty, tz
    ! Variables
    integer xmin, xmax, ymin, ymax, zmin, zmax, xsize, ysize, zsize
    integer xhalomin, xhalomax, yhalomin, yhalomax, zhalomin, zhalomax
    integer xhalo, yhalo, zhalo
    integer ix, iy, iz, ox, oy, oz
    integer ind_in, ind_out
    real(chm_real), allocatable, dimension(:) :: qtmp

    call get_spatial_limits(YZ_X_PARTITION,xmin,xmax,ymin,ymax,zmin,zmax,mynod)
    call get_halo_limits(YZ_X_PARTITION,xhalomin,xhalomax,yhalomin,yhalomax,zhalomin,zhalomax,mynod)
    call get_spatial_sizes(YZ_X_PARTITION, xsize, ysize, zsize, mynod)

    call chmalloc('colfft_util.src','translate_q_real','qtmp',xsize*ysize*zsize,lbou=0,crl=qtmp)

    qtmp(0:xsize*ysize*zsize-1) = q(0:xsize*ysize*zsize-1)

    xhalo = xmin - xhalomin
    yhalo = ymin - yhalomin
    zhalo = zmin - zhalomin

    do iz=0,zmax-zmin
       do iy=0,ymax-ymin
          do ix=0,xmax-xmin
             ind_in = ix+xhalo + (iy+yhalo)*xsize + (iz+zhalo)*xsize*ysize
             ox = ix + tx
             oy = iy + ty
             oz = iz + tz
             if (ox < 0) ox = ox + xmax-xmin+1
             if (oy < 0) oy = oy + ymax-ymin+1
             if (oz < 0) oz = oz + zmax-zmin+1
             if (ox >= xmax-xmin+1) ox = ox - xmax-xmin+1
             if (oy >= ymax-ymin+1) oy = oy - ymax-ymin+1
             if (oz >= zmax-zmin+1) oz = oz - zmax-zmin+1
             ind_out = ox+xhalo + (oy+yhalo)*xsize + (oz+zhalo)*xsize*ysize
             q(ind_out) = qtmp(ind_in)
          enddo
       enddo
    enddo

    call chmdealloc('colfft_util.src','translate_q_real','qtmp',xsize*ysize*zsize,crl=qtmp)

    return
  end subroutine translate_q_real

!ENDIF (columnfft)
end module colfft_util
