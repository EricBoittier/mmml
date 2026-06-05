#define PASTE(a) a
#define ADD_PREC(name) PASTE(name)PASTE(_)COLFFT_PREC

! this code relies on the C Preprocessor
! # define COLFFT_PREC to be either
! PS => SINGLE and set SINGLEP <= 1
! PD => DOUBLE and set DOUBLEP <= 1

!  ##EXP P0 sp dp .when. EXP (exp_precision)
!  ##PASS1 PS sp
!  ##PASS2 PD dp
!  ##EXEND

! *
! * Allocates temporary buffers needed for transpose() and ftranspose() -subroutines
! * nx = size of the fft transpose in x-direction
! *
subroutine ADD_PREC(allocate_transpose)()
#if KEY_PARALLEL==1
  use mpi,only:mpi_status_size  
#endif
  use memory
  implicit none
  ! Variables
  integer numtasks

  numtasks = ny_box*nz_box

#if KEY_PARALLEL==1
  if (allocated(pe_list)) then
     if (size(pe_list) < numtasks) then
        call deallocate_transpose()
     endif
  endif

  if (.not.allocated(pe_list)) then
     call chmalloc('colfft_util.src','allocate_transpose','pe_list',numtasks,ci4=pe_list)
     call chmalloc('colfft_util.src','allocate_transpose','recv_request',numtasks,&
          ci4=recv_request)
     call chmalloc('colfft_util.src','allocate_transpose','recv_offset',numtasks,lbou=0,&
          ci4=recv_offset)
     call chmalloc('colfft_util.src','allocate_transpose','send_request',numtasks,&
          ci4=send_request)
     call chmalloc('colfft_util.src','allocate_transpose','recv_status',MPI_STATUS_SIZE,&
          numtasks,ci4=recv_status)
     call chmalloc('colfft_util.src','allocate_transpose','send_status',MPI_STATUS_SIZE,&
          numtasks,ci4=send_status)
  endif
#endif 

  return
end subroutine ADD_PREC(allocate_transpose)

! *
! * Allocates data needed for fft and transposes
! *
subroutine ADD_PREC(allocate_fft_data)(data_len)
  use pmeutil,only:nfft1, nfft2, nfft3, cffti, ifac1, ifac2, ifac3
  use memory,only:chmalloc, chmdealloc
  use consta,only:twopi   /* (FFTW .or. MKL) */
  implicit none
  ! Input
  integer, intent(in) :: data_len
  ! Variables
  real(chm_real) theta,pi2n
  integer i

  ! Check if we should re-allocate
#if DOUBLEP == 1
  !--------------------------------------------------------------------------------
  ! Double precision
  !--------------------------------------------------------------------------------
  ! De-allocate single precision data
  if (allocated(transposed_data_ps)) then
     call chmdealloc('colfft_util.src','allocate_fft_data','transposed_data_ps',&
          size(transposed_data_ps),cr4=transposed_data_ps)
     call chmdealloc('colfft_util.src','allocate_fft_data','recv_buffer_ps',&
          size(recv_buffer_ps),mcr4p=recv_buffer_ps)
     call chmdealloc('colfft_util.src','allocate_fft_data','send_buffer_ps',&
          size(send_buffer_ps),mcr4p=send_buffer_ps)
  endif

  if (allocated(transposed_data_pd)) then
     if (size(transposed_data_pd) < data_len) then
        call chmdealloc('colfft_util.src','allocate_fft_data','transposed_data_pd',&
             size(transposed_data_pd),crl=transposed_data_pd)
        call chmdealloc('colfft_util.src','allocate_fft_data','recv_buffer_pd',&
             size(recv_buffer_pd),mcrlp=recv_buffer_pd)
        call chmdealloc('colfft_util.src','allocate_fft_data','send_buffer_pd',&
             size(send_buffer_pd),mcrlp=send_buffer_pd)
#if KEY_FFTW==1 || KEY_MKL==1
        call reset_fftw_plans()
#endif 
     endif
  endif

  if (.not.allocated(transposed_data_pd)) then
     call chmalloc('colfft_util.src','allocate_fft_data','transposed_data_pd',&
          data_len,lbou=0,crl=transposed_data_pd)
     call chmalloc('colfft_util.src','allocate_fft_data','recv_buffer_pd',&
          data_len,lbou=1,mcrlp=recv_buffer_pd)
     call chmalloc('colfft_util.src','allocate_fft_data','send_buffer_pd',&
          data_len,lbou=1,mcrlp=send_buffer_pd)
  endif

#elif SINGLEP == 1
  !--------------------------------------------------------------------------------
  ! Single precision
  !--------------------------------------------------------------------------------
  ! De-allocate double precision data
  if (allocated(transposed_data_pd)) then
     call chmdealloc('colfft_util.src','allocate_fft_data','transposed_data_pd',&
          size(transposed_data_pd),crl=transposed_data_pd)
     call chmdealloc('colfft_util.src','allocate_fft_data','recv_buffer_pd',&
          size(recv_buffer_pd),mcrlp=recv_buffer_pd)
     call chmdealloc('colfft_util.src','allocate_fft_data','send_buffer_pd',&
          size(send_buffer_pd),mcrlp=send_buffer_pd)
  endif

  if (allocated(transposed_data_ps)) then
     if (size(transposed_data_ps) < data_len) then
        call chmdealloc('colfft_util.src','allocate_fft_data','transposed_data_ps',&
             size(transposed_data_ps),cr4=transposed_data_ps)
        call chmdealloc('colfft_util.src','allocate_fft_data','recv_buffer_ps',&
             size(recv_buffer_ps),mcr4p=recv_buffer_ps)
        call chmdealloc('colfft_util.src','allocate_fft_data','send_buffer_ps',&
             size(send_buffer_ps),mcr4p=send_buffer_ps)
#if KEY_FFTW==1 || KEY_MKL==1
        call reset_fftw_plans()
#endif 
     endif
  endif

  if (.not.allocated(transposed_data_ps)) then
     call chmalloc('colfft_util.src','allocate_fft_data','transposed_data_ps',&
          data_len,lbou=0,cr4=transposed_data_ps)
     call chmalloc('colfft_util.src','allocate_fft_data','recv_buffer_ps',&
          data_len,lbou=1,mcr4p=recv_buffer_ps)
     call chmalloc('colfft_util.src','allocate_fft_data','send_buffer_ps',&
          data_len,lbou=1,mcr4p=send_buffer_ps)
  endif
#endif 

#if KEY_FFTW==1 || KEY_MKL==1
#else /**/
  if (allocated(fft_table_1)) then
     if ((size(fft_table_1) /= fftdim(X_DIRECTION)*4+15 .or. &
          size(fft_table_2) /= fftdim(Y_DIRECTION)*4+15 .or. &
          size(fft_table_2) /= fftdim(Z_DIRECTION)*4+15 .or. &
          size(alpha_rcfft) /= nfft1+1 .or. size(beta_rcfft) /= nfft1+1 .or. &
          size(tmp_rc,2) /= fftdim(X_DIRECTION))) then
        call chmdealloc('colfft_util.src','allocate_fft_data','fft_table_1',&
             size(fft_table_1),crl=fft_table_1)
        call chmdealloc('colfft_util.src','allocate_fft_data','fft_table_2',&
             size(fft_table_2),crl=fft_table_2)
        call chmdealloc('colfft_util.src','allocate_fft_data','fft_table_3',&
             size(fft_table_3),crl=fft_table_3)

        call chmdealloc('colfft_util.src','allocate_fft_data','alpha_rcfft',&
             size(alpha_rcfft),crl=alpha_rcfft)
        call chmdealloc('colfft_util.src','allocate_fft_data','beta_rcfft',&
             size(beta_rcfft),crl=beta_rcfft)
        call chmdealloc('colfft_util.src','allocate_fft_data','tmp_rc',&
             size(tmp_rc,1),size(tmp_rc,2),crl=tmp_rc) 
     endif
  endif

  if (.not.allocated(fft_table_1)) then
     call chmalloc('colfft_util.src','allocate_fft_data','fft_table_1',&
          fftdim(X_DIRECTION)*4+15,crl=fft_table_1)
     call chmalloc('colfft_util.src','allocate_fft_data','fft_table_2',&
          fftdim(Y_DIRECTION)*4+15,crl=fft_table_2)
     call chmalloc('colfft_util.src','allocate_fft_data','fft_table_3',&
          fftdim(Z_DIRECTION)*4+15,crl=fft_table_3)

     call cffti(fftdim(X_DIRECTION),fft_table_1,ifac1)
     call cffti(fftdim(Y_DIRECTION),fft_table_2,ifac2)
     call cffti(fftdim(Z_DIRECTION),fft_table_3,ifac3)

     call chmalloc('colfft_util.src','allocate_fft_data','alpha_rcfft',&
          nfft1+1,lbou=0,crl=alpha_rcfft)
     call chmalloc('colfft_util.src','allocate_fft_data','beta_rcfft',&
          nfft1+1,lbou=0,crl=beta_rcfft)
     call chmalloc('colfft_util.src','allocate_fft_data','tmp_rc',&
          2,fftdim(X_DIRECTION),lbou2=0,crl=tmp_rc)

     pi2n=TWOPI/nfft1
     do i=0,fftdim(X_DIRECTION)-1
        theta=pi2n*i
        alpha_rcfft(i) = cos(theta)
        beta_rcfft(i)  = sin(theta)
     end do
  endif
#endif 

  return
end subroutine ADD_PREC(allocate_fft_data)

!-----------------------------------------------------------------------
!           Backward_RC_FFT
!-----------------------------------------------------------------------
subroutine ADD_PREC(backward_rc_fft)(data, data_len)
  use new_timer,only:timer_start,timer_stop,T_fftcomm
#if KEY_DOMDEC_GPU==1
  use domdec_util_gpu_mod,only:range_start, range_stop   
  use domdec_common,only:q_gpu    
#endif
  implicit none
  ! Input / Output
  integer, intent(in) :: data_len
#if DOUBLEP == 1
  real(chm_real), intent(inout) :: data(0:data_len-1)
#elif SINGLEP == 1
  real(chm_real4), intent(inout) :: data(0:data_len-1)
#endif 
  ! Variables
  integer xgridmin,xgridmax,ygridmin,ygridmax,zgridmin,zgridmax
!!$    character(20) filename
  integer i

#if SINGLEP == 1 && KEY_COLFFT_NOSP == 1
  call wrndie(-5,'<colfft_util>','FFTW not compiled in single precision') 
#endif

  call ADD_PREC(allocate_fft_data)(data_len)

!!$    call translate_q_real(data, -3, -3, -3)

!!$    if (numnod == 1) then
!!$       call ADD_PREC(write_q_real)(data, YZ_X_PARTITION, 'q_real.txt')
!!$    else
!!$       write (filename,'(a,i1,a)') 'q_real_',mynod,'.txt'
!!$       call ADD_PREC(write_q_real)(data, YZ_X_PARTITION, trim(filename))
!!$    endif

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('x_fft')  
#endif
  call ADD_PREC(x_fft)(TRANSFORM_BACKWARD, data)
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()  
#endif

!!$    if (numnod == 1) then
!!$       call ADD_PREC(write_q_complex)(data, YZ_X_PARTITION, 'q_comp1.txt')
!!$    else
!!$       write (filename,'(a,i1,a)') 'q_comp1_',mynod,'.txt'
!!$       call ADD_PREC(write_q_complex)(data, YZ_X_PARTITION, trim(filename))
!!$    endif

  !
  ! Transpose data(x,y,z) -> transposed_data(y,z,x)
  !
  call timer_start(T_FFTCOMM)
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('transpose')  
#endif
  call ADD_PREC(transpose)(data, YZ_X_PARTITION, &
                           ADD_PREC(transposed_data), ZX_Y_PARTITION, &
                           YZ_X_COMM_ZX_Y, ADD_PREC(recv_buffer), &
                           ADD_PREC(send_buffer))
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()  
#endif
  call timer_stop(T_FFTCOMM)

!!$    if (numnod == 1) then
!!$       call write_q_complex(transposed_data, ZX_Y_PARTITION, 'q_comp2.txt')
!!$    else
!!$       write (filename,'(a,i1,a)') 'q_comp2_',mynod,'.txt'
!!$       call write_q_complex(transposed_data, ZX_Y_PARTITION, trim(filename))
!!$    endif

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('y_fft')  
#endif
  call ADD_PREC(y_fft)(TRANSFORM_BACKWARD, ADD_PREC(transposed_data))
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()  
#endif

!!$    if (numnod == 1) then
!!$       call write_q_complex(transposed_data, ZX_Y_PARTITION, 'q_comp3.txt')
!!$    else
!!$       write (filename,'(a,i1,a)') 'q_comp3_',mynod,'.txt'
!!$       call write_q_complex(transposed_data, ZX_Y_PARTITION, trim(filename))
!!$    endif

  !
  ! Transpose: transposed_data(y,z,x) -> data(z,x,y)
  !
  call timer_start(T_FFTCOMM)
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('transpose')  
#endif
  call ADD_PREC(transpose)(ADD_PREC(transposed_data), ZX_Y_PARTITION, &
                       data, XY_Z_PARTITION, &
                       XY_Z_COMM_ZX_Y, ADD_PREC(recv_buffer), &
                       ADD_PREC(send_buffer))
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()  
#endif
  call timer_stop(T_FFTCOMM)                       

!!$    if (numnod == 1) then
!!$       call write_q_complex(data, XY_Z_PARTITION, 'q_comp4.txt')
!!$    else
!!$       write (filename,'(a,i1,a)') 'q_comp4_',mynod,'.txt'
!!$       call write_q_complex(data, XY_Z_PARTITION, trim(filename))
!!$    endif

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('z_fft')  
#endif
  call ADD_PREC(z_fft)(TRANSFORM_BACKWARD, data)
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()  
#endif

!!$    if (numnod == 1) then
!!$       call ADD_PREC(write_q_complex)(data, XY_Z_PARTITION, 'q_comp5.txt')
!!$    else
!!$       write (filename,'(a,i1,a)') 'q_comp5_',mynod,'.txt'
!!$       call ADD_PREC(write_q_complex)(data, XY_Z_PARTITION, trim(filename))
!!$    endif

  return
end subroutine ADD_PREC(backward_rc_fft)

!-----------------------------------------------------------------------
!           Forward_RC_FFT
!-----------------------------------------------------------------------
subroutine ADD_PREC(forward_rc_fft)(data, data_len)
  use new_timer,only:timer_start,timer_stop,T_FFTcomm  
#if KEY_DOMDEC_GPU==1
  use domdec_util_gpu_mod,only:range_start, range_stop  
#endif
#if KEY_DOMDEC_GPU==1
  use domdec_common,only:q_gpu                       
#endif
!!$    use parallel,only:mynod, numnod
  implicit none
  ! Input / Output
  integer, intent(in) :: data_len
#if DOUBLEP == 1
  real(chm_real), intent(inout) :: data(0:data_len-1)
#elif SINGLEP == 1
  real(chm_real4), intent(inout) :: data(0:data_len-1)
#endif 
!!$    character(20) filename

  call ADD_PREC(allocate_fft_data)(data_len)

!!$    if (numnod == 1) then
!!$       call ADD_PREC(write_q_complex)(data, XY_Z_PARTITION, 'q_comp6.txt')
!!$    else
!!$       write (filename,'(a,i1,a)') 'q_comp6_',mynod,'.txt'
!!$       call ADD_PREC(write_q_complex)(data, XY_Z_PARTITION, trim(filename))
!!$    endif

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('z_fft')  
#endif
  call ADD_PREC(z_fft)( TRANSFORM_FORWARD, data )
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()  
#endif

!!$    if (numnod == 1) then
!!$       call ADD_PREC(write_q_complex)(data, XY_Z_PARTITION, 'q_comp7.txt')
!!$    else
!!$       write (filename,'(a,i1,a)') 'q_comp7_',mynod,'.txt'
!!$       call ADD_PREC(write_q_complex)(data, XY_Z_PARTITION, trim(filename))
!!$    endif

  !
  ! Transpose data(z,x,y) -> transposed_data(y,z,x)
  !
  call timer_start(T_FFTCOMM)     
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('ftranspose')  
#endif
  call ADD_PREC(ftranspose)(data, XY_Z_PARTITION, ADD_PREC(transposed_data), &
                            ZX_Y_PARTITION, XY_Z_COMM_ZX_Y, &
                            ADD_PREC(recv_buffer), &
                            ADD_PREC(send_buffer))
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()  
#endif
  call timer_stop(T_FFTCOMM)        

!!$    if (numnod == 1) then
!!$       call write_q_complex(transposed_data, ZX_Y_PARTITION, 'q_comp8.txt')
!!$    else
!!$       write (filename,'(a,i1,a)') 'q_comp8_',mynod,'.txt'
!!$       call write_q_complex(transposed_data, ZX_Y_PARTITION, trim(filename))
!!$    endif

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('y_fft')  
#endif
  call ADD_PREC(y_fft)(TRANSFORM_FORWARD, ADD_PREC(transposed_data))
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()  
#endif

!!$    if (numnod == 1) then
!!$       call ADD_PREC(write_q_complex)(ADD_PREC(transposed_data), ZX_Y_PARTITION, 'q_comp9.txt')
!!$    else
!!$       write (filename,'(a,i1,a)') 'q_comp9_',mynod,'.txt'
!!$       call ADD_PREC(write_q_complex)(ADD_PREC(transposed_data), ZX_Y_PARTITION, trim(filename))
!!$    endif

  !
  ! Transpose: transposed_data(y,z,x) -> data(x,y,z)
  !
  call timer_start(T_FFTCOMM)        
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('ftranspose')  
#endif
  call ADD_PREC(ftranspose)(ADD_PREC(transposed_data), ZX_Y_PARTITION, data, &
                            YZ_X_PARTITION, YZ_X_COMM_ZX_Y, &
                            ADD_PREC(recv_buffer), ADD_PREC(send_buffer))
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()  
#endif
  call timer_stop(T_FFTCOMM)     

!!$    if (numnod == 1) then
!!$       call ADD_PREC(write_q_complex)(data, YZ_X_PARTITION, 'q_comp10.txt')
!!$    else
!!$       write (filename,'(a,i1,a)') 'q_comp10_',mynod,'.txt'
!!$       call ADD_PREC(write_q_complex)(data, YZ_X_PARTITION, trim(filename))
!!$    endif

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('x_fft')  
#endif
  call ADD_PREC(x_fft)(TRANSFORM_FORWARD, data)
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()  
#endif

!!$    if (numnod == 1) then
!!$       call ADD_PREC(write_q_real)(data, YZ_X_PARTITION, 'q_real2.txt')
!!$    else
!!$       write (filename,'(a,i1,a)') 'q_real2_',mynod,'.txt'
!!$       call ADD_PREC(write_q_real)(data, YZ_X_PARTITION, trim(filename))
!!$    endif

  return
end subroutine ADD_PREC(forward_rc_fft)

! *
! * Writes real valued q on file
! *
subroutine ADD_PREC(write_q_real)(q, partition, filename)
  use stream,only:outu
  use parallel,only:mynod
  implicit none
  ! Input / Output
#if DOUBLEP == 1
  real(chm_real), intent(in) :: q(0:*)
#else /**/
  real(chm_real4), intent(in) :: q(0:*)
#endif 
  integer, intent(in) :: partition
  character(len=*), intent(in) :: filename
  ! Variables
  integer xmin, xmax, ymin, ymax, zmin, zmax, xsize, ysize, zsize
  integer xhalomin, xhalomax, yhalomin, yhalomax, zhalomin, zhalomax
  integer xhalo, yhalo, zhalo
  integer x, y, z, ind, n

  call get_spatial_limits(partition,xmin,xmax,ymin,ymax,zmin,zmax,mynod)
  call get_halo_limits(partition,xhalomin,xhalomax,yhalomin,yhalomax,zhalomin,zhalomax,mynod)
  call get_spatial_sizes(partition, xsize, ysize, zsize, mynod)

  xhalo = xmin - xhalomin
  yhalo = ymin - yhalomin
  zhalo = zmin - zhalomin

  write (outu,'(a,5i4)') filename,xmax-xmin+1,ymax-ymin+1,zmax-zmin+1,xsize,ysize

  n = 0
  open(121,file=filename)
  do z=0,zmax-zmin
     do y=0,ymax-ymin
        do x=0,xmax-xmin
           n = n + 1
           ind = x+xhalo + (y+yhalo)*xsize + (z+zhalo)*xsize*ysize
           write (121,'(e26.18)') q(ind)
        enddo
     enddo
  enddo
  close(121)

  return
end subroutine ADD_PREC(write_q_real)

! *
! * Writes complex valued q on file
! *
subroutine ADD_PREC(write_q_complex)(q, partition, filename)
  use stream,only:outu
  use parallel,only:mynod
  implicit none
  ! Input / Output
#if DOUBLEP == 1
  real(chm_real), intent(in) :: q(0:*)
#else /**/
  real(chm_real4), intent(in) :: q(0:*)
#endif 
  integer, intent(in) :: partition
  character(len=*), intent(in) :: filename
  ! Variables
  integer xmin, xmax, ymin, ymax, zmin, zmax, xsize, ysize, zsize
  integer xhalomin, xhalomax, yhalomin, yhalomax, zhalomin, zhalomax
  integer xhalo, yhalo, zhalo
  integer x, y, z, ind, n

  if (partition == ZX_Y_PARTITION) then
     call get_fft_limits(partition,xmin,xmax,zmin,zmax,ymin,ymax,mynod)
     call get_halo_limits(partition,xhalomin,xhalomax,zhalomin,zhalomax,yhalomin,yhalomax,mynod)
     call get_fft_sizes(partition, xsize, zsize, ysize, mynod)
  else
     call get_fft_limits(partition,xmin,xmax,ymin,ymax,zmin,zmax,mynod)
     call get_halo_limits(partition,xhalomin,xhalomax,yhalomin,yhalomax,zhalomin,zhalomax,mynod)
     call get_fft_sizes(partition, xsize, ysize, zsize, mynod)
  endif

  xhalo = xmin - xhalomin
  yhalo = ymin - yhalomin
  zhalo = zmin - zhalomin

  if (xhalo /= 0) then
     call wrndie(-5,'<colfft_util>','write_q_complex: Error in xhalo')
  endif

  write (outu,'(a,5i4)') filename,xmax-xmin+1,ymax-ymin+1,zmax-zmin+1,xsize,ysize

  n = 0
  open(121,file=filename)
  do z=0,zmax-zmin
     do y=0,ymax-ymin
        do x=0,xmax-xmin
           n = n + 1
           ind = 2*(x + (y+yhalo)*xsize + (z+zhalo)*xsize*ysize)
           write (121,'(2e26.18)') q(ind:ind+1)
        enddo
     enddo
  enddo
  close(121)

  return
end subroutine ADD_PREC(write_q_complex)

!-----------------------------------------------------------------------
!           X_FFT
!-----------------------------------------------------------------------
subroutine ADD_PREC(x_fft)(isign, data)
  use memory
  use pmeutil,only:ifac1,nfft2,nfft3
  use parallel,only:mynod
  implicit none
  ! Input / Output
  integer, intent(in) :: isign
#if DOUBLEP == 1
  real(chm_real), intent(inout) :: data(0:*)
#elif SINGLEP == 1
  real(chm_real4), intent(inout) :: data(0:*)
#endif 
  ! Variables
#if KEY_FFTW==1 || KEY_MKL==1
  integer(C_INT) :: rank, n(1), howmany, inembed(1), istride, idist, onembed(1), ostride, odist
#if DOUBLEP == 1
  real(chm_real), allocatable, dimension(:) :: data_save
#elif SINGLEP == 1
  real(chm_real4), allocatable, dimension(:) :: data_save
#endif 
#endif 
  integer min1, max1, size1, len1, fftdim1, halo1, halo2, halo3
  integer min2, max2, len2, size2
  integer min3, max3, len3, size3
  integer i, j, k, index0, index1

  min1 = con_gridmin(YZ_X_PARTITION)
  max1 = con_gridmax(YZ_X_PARTITION)
  len1 = max1 - min1 + 1
  size1 = con_size(YZ_X_PARTITION)
  fftdim1 = con_fftdim(YZ_X_PARTITION)
  halo1 = con_gridmin(YZ_X_PARTITION) - con_halomin(YZ_X_PARTITION)

  min2 = dec_gridmin(1, mynod, YZ_X_PARTITION)
  max2 = dec_gridmax(1, mynod, YZ_X_PARTITION)
  len2 = max2 - min2 + 1
  size2 = dec_size(1, mynod, YZ_X_PARTITION)
  halo2 = dec_gridmin(1, mynod, YZ_X_PARTITION) - dec_halomin(1, mynod, YZ_X_PARTITION)

  min3 = dec_gridmin(2, mynod, YZ_X_PARTITION)
  max3 = dec_gridmax(2, mynod, YZ_X_PARTITION)
  len3 = max3 - min3 + 1
  size3 = dec_size(2, mynod, YZ_X_PARTITION)
  halo3 = dec_gridmin(2, mynod, YZ_X_PARTITION) - dec_halomin(2, mynod, YZ_X_PARTITION)

#if KEY_FFTW==1 || KEY_MKL==1
  if (isign == TRANSFORM_BACKWARD ) then
     ! Create Real-to-Complex plan

     if (n_fftw_xf_plan /= len2) then
        n_fftw_xf_plan = len2

#if DOUBLEP == 1
        plan_type = 1
#else /**/
        plan_type = 2
#endif 

        rank = 1
        n = fftdim1*2
        howmany = n_fftw_xf_plan
        !in = data (real)
        inembed = size1
        istride = 1
        idist = size1
        !out = data (complex)
        onembed = size1/2
        ostride = 1
        odist = size1/2

        index0 = halo1 + (0+halo3)*size1*size2 + size1*halo2
        index1 = halo1 + (1+halo3)*size1*size2 + size1*halo2 - 1

#if DOUBLEP == 1
        call chmalloc('colfft_util.src','x_fft','data_save',&
             index1-index0+1,lbou=index0,crl=data_save)
#elif SINGLEP == 1
        call chmalloc('colfft_util.src','x_fft','data_save',&
             index1-index0+1,lbou=index0,cr4=data_save)
#endif 
        data_save(index0:index1) = data(index0:index1)

#if DOUBLEP == 1
        call dfftw_plan_many_dft_r2c(fftw_xf_plan, rank, n, howmany, data(index0), &
             inembed, istride, idist, data(index0), onembed, ostride, odist, fftw_opt)
#elif SINGLEP == 1
#if KEY_COLFFT_NOSP==0
        call sfftw_plan_many_dft_r2c(fftw_xf_plan, rank, n, howmany, data(index0), &
             inembed, istride, idist, data(index0), onembed, ostride, odist, fftw_opt)
#endif 
#endif 

        data(index0:index1) = data_save(index0:index1)

#if DOUBLEP == 1
        call chmdealloc('colfft_util.src','x_fft','data_save',&
             index1-index0+1,crl=data_save)
#elif SINGLEP == 1
        call chmdealloc('colfft_util.src','x_fft','data_save',&
             index1-index0+1,cr4=data_save)
#endif 

     endif

     !$omp parallel do schedule(static) private(k, index0)
     do k = 0, len3-1
        index0 = halo1 + (k+halo3)*size1*size2 + size1*halo2
#if DOUBLEP == 1
        call dfftw_execute_dft_r2c(fftw_xf_plan, data(index0), data(index0))
#elif SINGLEP == 1
#if KEY_COLFFT_NOSP==0
        call sfftw_execute_dft_r2c(fftw_xf_plan, data(index0), data(index0))
#endif 
#endif 
     enddo
     !$omp end parallel do

  else
     ! Create Complex-to-Real plan

     if (n_fftw_xb_plan /= len2) then
        n_fftw_xb_plan = len2

#if DOUBLEP == 1
        plan_type = 1
#else /**/
        plan_type = 2
#endif 

        rank = 1
        n = fftdim1*2
        howmany = n_fftw_xb_plan
        !in = data (complex)
        inembed = size1/2
        istride = 1
        idist = size1/2
        !out = tmpr (real)
        onembed = size1
        ostride = 1
        odist = size1

        index0 = halo1 + (0+halo3)*size1*size2 + size1*halo2
        index1 = halo1 + (1+halo3)*size1*size2 + size1*halo2 - 1

#if DOUBLEP == 1
        call chmalloc('colfft_util.src','x_fft','data_save',&
             index1-index0+1,lbou=index0,crl=data_save)
#elif SINGLEP == 1
        call chmalloc('colfft_util.src','x_fft','data_save',&
             index1-index0+1,lbou=index0,cr4=data_save)
#endif 
        data_save(index0:index1) = data(index0:index1)

#if DOUBLEP == 1
        call dfftw_plan_many_dft_c2r(fftw_xb_plan, rank, n, howmany, data(index0), &
             inembed, istride, idist, data(index0), onembed, ostride, odist, fftw_opt)
#elif SINGLEP == 1
#if KEY_COLFFT_NOSP==0
        call sfftw_plan_many_dft_c2r(fftw_xb_plan, rank, n, howmany, data(index0), &
             inembed, istride, idist, data(index0), onembed, ostride, odist, fftw_opt)
#endif 
#endif 

        data(index0:index1) = data_save(index0:index1)
#if DOUBLEP == 1
        call chmdealloc('colfft_util.src','x_fft','data_save',&
             index1-index0+1,crl=data_save)
#elif SINGLEP == 1
        call chmdealloc('colfft_util.src','x_fft','data_save',&
             index1-index0+1,cr4=data_save)
#endif 

     endif

     !$omp parallel do schedule(static) private(k, index0)
     do k = 0, len3-1
        index0 = halo1 + (k+halo3)*size1*size2 + size1*halo2
#if DOUBLEP == 1
        call dfftw_execute_dft_c2r(fftw_xb_plan, data(index0), data(index0))
#elif SINGLEP == 1
#if KEY_COLFFT_NOSP==0
        call sfftw_execute_dft_c2r(fftw_xb_plan, data(index0), data(index0))
#endif 
#endif 
     enddo
     !$omp end parallel do

  endif
#else /**/
  ! Native CHARMM FFT
  ! APH 9/18/2014
  ! NOTE: No OpenMP parallelization here because the native implementation is
  !       not thread safe. To make it thread safe, we need to give each thread
  !       its own (fft_table_1, fft_table_2, fft_table_3) -lists.
  do k = min3, max3
     do j = min2, max2
        index0 = halo1 + (k-min3+halo3)*size1*size2 + (j-min2+halo2)*size1
#if DOUBLEP == 1
        call fft_1d_realcomplex( isign, fftdim1, data( index0 ), &
             fft_table_1 ,ifac1)
#elif SINGLEP == 1
        call wrndie(-5,'<colfft_util>','x_fft native FFT not implemented in single precision')
#endif 
     enddo
  enddo
#endif 
end subroutine ADD_PREC(x_fft)

!-----------------------------------------------------------------------
!           Y_FFT
!-----------------------------------------------------------------------
subroutine ADD_PREC(y_fft)(isign, data)
  use memory
  use pmeutil,only:ifac2,nfft2,nfft3
  use parallel,only:mynod
  implicit none
  ! Input / Output
  integer, intent(in) :: isign
#if DOUBLEP == 1
  real(chm_real), intent(inout) :: data(0:*)
#elif SINGLEP == 1
  real(chm_real4), intent(inout) :: data(0:*)
#endif 
  ! Variables
#if KEY_FFTW==1 || KEY_MKL==1
  integer(C_INT) :: rank, n(1), howmany, inembed(1), istride, idist, onembed(1), ostride, odist
#if DOUBLEP == 1
  real(chm_real), allocatable, dimension(:) :: data_save
#elif SINGLEP == 1
  real(chm_real4), allocatable, dimension(:) :: data_save
#endif 
#endif 
  integer min1, max1, len1, size1, halo1
  integer min2, max2, len2, size2
  integer min3, max3, len3, size3
  integer i,j,k,index

  min1 = con_gridmin(ZX_Y_PARTITION)
  max1 = con_gridmax(ZX_Y_PARTITION)
  size1 = con_csize(ZX_Y_PARTITION)
  len1 = max1 - min1 + 1
  halo1 = con_gridmin(ZX_Y_PARTITION) - con_halomin(ZX_Y_PARTITION)

  min2 = dec_gridmin(2, mynod, ZX_Y_PARTITION)
  max2 = dec_gridmax(2, mynod, ZX_Y_PARTITION)
  size2 = dec_csize(2, mynod, ZX_Y_PARTITION)
  len2 = max2 - min2 + 1

  min3 = dec_gridmin(1, mynod, ZX_Y_PARTITION)
  max3 = dec_gridmax(1, mynod, ZX_Y_PARTITION)
  size3 = dec_csize(1, mynod, ZX_Y_PARTITION)
  len3 = max3 - min3 + 1

#if KEY_FFTW==1 || KEY_MKL==1
  ! Create Complex-to-Complex plan
  if (isign == TRANSFORM_BACKWARD ) then

     if (n_fftw_yf_plan /= len2) then
        n_fftw_yf_plan = len2

#if DOUBLEP == 1
        plan_type = 1
#else /**/
        plan_type = 2
#endif 

        rank = 1
        n = len1
        howmany = n_fftw_yf_plan
        !in = data
        inembed = size1
        istride = 1
        idist = size1
        !out = data
        onembed = size1
        ostride = 1
        odist = size1

        index = size1*size2*2
#if DOUBLEP == 1
        call chmalloc('colfft_util.src','y_fft','data_save',&
             index,lbou=0,crl=data_save)
#elif SINGLEP == 1
        call chmalloc('colfft_util.src','y_fft','data_save',&
             index,lbou=0,cr4=data_save)
#endif 
        data_save(0:index-1) = data(0:index-1)

#if DOUBLEP == 1
        call dfftw_plan_many_dft(fftw_yf_plan, rank, n, howmany, data, inembed, &
             istride, idist, data, onembed, ostride, odist, FFTW_FORWARD, &
             fftw_opt)
#elif SINGLEP == 1
#if KEY_COLFFT_NOSP==0
        call sfftw_plan_many_dft(fftw_yf_plan, rank, n, howmany, data, inembed, &
             istride, idist, data, onembed, ostride, odist, FFTW_FORWARD, &
             fftw_opt)
#endif 
#endif 

        data(0:index-1) = data_save(0:index-1)
#if DOUBLEP == 1
        call chmdealloc('colfft_util.src','y_fft','data_save',&
             index,crl=data_save)
#elif SINGLEP == 1
        call chmdealloc('colfft_util.src','y_fft','data_save',&
             index,cr4=data_save)
#endif 

     endif

     !$omp parallel do schedule(static) private(k, index)
     do k = 0, len3-1
        index = k*size1*size2*2
#if DOUBLEP == 1
        call dfftw_execute_dft(fftw_yf_plan, data(index), data(index))
#elif SINGLEP == 1
#if KEY_COLFFT_NOSP==0
        call sfftw_execute_dft(fftw_yf_plan, data(index), data(index))
#endif 
#endif 
     enddo
     !$omp end parallel do

  else

     if (n_fftw_yb_plan /= len2) then
        n_fftw_yb_plan = len2

#if DOUBLEP == 1
        plan_type = 1
#else /**/
        plan_type = 2
#endif 

        rank = 1
        n = len1
        howmany = n_fftw_yb_plan
        !in = data
        inembed = size1
        istride = 1
        idist = size1
        !out = data
        onembed = size1
        ostride = 1
        odist = size1

        index = size1*size2*2
#if DOUBLEP == 1
        call chmalloc('colfft_util.src','y_fft','data_save',&
             index,lbou=0,crl=data_save)
#elif SINGLEP == 1
        call chmalloc('colfft_util.src','y_fft','data_save',&
             index,lbou=0,cr4=data_save)
#endif 
        data_save(0:index-1) = data(0:index-1)

#if DOUBLEP == 1
        call dfftw_plan_many_dft(fftw_yb_plan, rank, n, howmany, data, inembed, &
             istride, idist, data, onembed, ostride, odist, FFTW_BACKWARD, &
             fftw_opt)
#elif SINGLEP == 1
#if KEY_COLFFT_NOSP==0
        call sfftw_plan_many_dft(fftw_yb_plan, rank, n, howmany, data, inembed, &
             istride, idist, data, onembed, ostride, odist, FFTW_BACKWARD, &
             fftw_opt)
#endif 
#endif 

        data(0:index-1) = data_save(0:index-1)
#if DOUBLEP == 1
        call chmdealloc('colfft_util.src','y_fft','data_save',&
             index,crl=data_save)
#elif SINGLEP == 1
        call chmdealloc('colfft_util.src','y_fft','data_save',&
             index,cr4=data_save)
#endif 

     endif

     !$omp parallel do schedule(static) private(k, index)
     do k = 0, len3-1
        index = k*size1*size2*2
#if DOUBLEP == 1
        call dfftw_execute_dft(fftw_yb_plan, data(index), data(index))
#elif SINGLEP == 1
#if KEY_COLFFT_NOSP==0
        call sfftw_execute_dft(fftw_yb_plan, data(index), data(index))
#endif 
#endif 
     enddo
     !$omp end parallel do

  endif
#else /**/
  ! Native CHARMM FFT
  ! APH 9/18/2014
  ! NOTE: No OpenMP parallelization here because the native implementation is
  !       not thread safe. To make it thread safe, we need to give each thread
  !       its own (fft_table_1, fft_table_2, fft_table_3) -lists.
  do i = min3, max3
     do k = min2, max2
        index = 2*((i-min3)*size1*size2 + (k-min2)*size1)
#if DOUBLEP == 1
        call fft_1d_cc( isign, len1, fft_table_2, data( index ), &
             ifac2)
#elif SINGLEP == 1
        call wrndie(-5,'<colfft_util>','y_fft native FFT not implemented in single precision')
#endif 
     enddo
  enddo
#endif 
end subroutine ADD_PREC(y_fft)

!-----------------------------------------------------------------------
!           Z_FFT
!-----------------------------------------------------------------------
subroutine ADD_PREC(z_fft)(isign, data)
  use memory
  use pmeutil,only:ifac3,nfft2,nfft3
  use parallel,only:mynod
  implicit none
  ! Input / Output
  integer, intent(in) :: isign
#if DOUBLEP == 1
  real(chm_real), intent(inout) :: data(0:*)
#elif SINGLEP == 1
  real(chm_real4), intent(inout) :: data(0:*)
#endif 
  ! Variables
#if KEY_FFTW==1 || KEY_MKL==1
  integer(C_INT) :: rank, n(1), howmany, inembed(1), istride, idist, onembed(1), ostride, odist
#if DOUBLEP == 1
  real(chm_real), allocatable, dimension(:) :: data_save
#elif SINGLEP == 1
  real(chm_real4), allocatable, dimension(:) :: data_save
#endif 
#endif 
  integer min3, max3, len3, size3
  integer min1, max1, len1, size1
  integer i,index,j,k
  integer min2, max2, len2, size2

  min1 = con_gridmin(XY_Z_PARTITION)
  max1 = con_gridmax(XY_Z_PARTITION)
  size1 = con_csize(XY_Z_PARTITION)
  len1 = max1 - min1 + 1

  min2 = dec_gridmin(1, mynod, XY_Z_PARTITION)
  max2 = dec_gridmax(1, mynod, XY_Z_PARTITION)
  size2 = dec_csize(1, mynod, XY_Z_PARTITION)
  len2 = max2 - min2 + 1

  min3 = dec_gridmin(2, mynod, XY_Z_PARTITION)
  max3 = dec_gridmax(2, mynod, XY_Z_PARTITION)
  size3 = dec_csize(2, mynod, XY_Z_PARTITION)
  len3 = max3 - min3 + 1

#if KEY_FFTW==1 || KEY_MKL==1
  ! Create Complex-to-Complex plan
  if (isign == TRANSFORM_BACKWARD ) then

     if (n_fftw_zf_plan /= len2) then
        n_fftw_zf_plan = len2

#if DOUBLEP == 1
        plan_type = 1
#else /**/
        plan_type = 2
#endif 

        rank = 1
        n = len1
        howmany = n_fftw_zf_plan
        !in = data
        inembed = size1
        istride = 1
        idist = size1
        !out = data
        onembed = size1
        ostride = 1
        odist = size1

        index = 2*size1*size2
#if DOUBLEP == 1
        call chmalloc('colfft_util.src','z_fft','data_save',&
             index,lbou=0,crl=data_save)
#elif SINGLEP == 1
        call chmalloc('colfft_util.src','z_fft','data_save',&
             index,lbou=0,cr4=data_save)
#endif 
        data_save(0:index-1) = data(0:index-1)

#if DOUBLEP == 1
        call dfftw_plan_many_dft(fftw_zf_plan, rank, n, howmany, data, inembed, &
             istride, idist, data, onembed, ostride, odist, FFTW_FORWARD, &
             fftw_opt)
#elif SINGLEP == 1
#if KEY_COLFFT_NOSP==0
        call sfftw_plan_many_dft(fftw_zf_plan, rank, n, howmany, data, inembed, &
             istride, idist, data, onembed, ostride, odist, FFTW_FORWARD, &
             fftw_opt)
#endif 
#endif 

        data(0:index-1) = data_save(0:index-1)
#if DOUBLEP == 1
        call chmdealloc('colfft_util.src','z_fft','data_save',&
             index,crl=data_save)
#elif SINGLEP == 1
        call chmdealloc('colfft_util.src','z_fft','data_save',&
             index,cr4=data_save)
#endif 
     endif

     !$omp parallel do schedule(static) private(k, index)
     do k = 0, len3-1
        index = 2*k*size1*size2
#if DOUBLEP == 1
        call dfftw_execute_dft(fftw_zf_plan, data(index), data(index))
#elif SINGLEP == 1
#if KEY_COLFFT_NOSP==0
        call sfftw_execute_dft(fftw_zf_plan, data(index), data(index))
#endif 
#endif 
     enddo
     !$omp end parallel do

  else

     if (n_fftw_zb_plan /= len2) then
        n_fftw_zb_plan = len2

#if DOUBLEP == 1
        plan_type = 1
#else /**/
        plan_type = 2
#endif 

        rank = 1
        n = len1
        howmany = n_fftw_zb_plan
        !in = data
        inembed = size1
        istride = 1
        idist = size1
        !out = data
        onembed = size1
        ostride = 1
        odist = size1

        index = 2*size1*size2
#if DOUBLEP == 1
        call chmalloc('colfft_util.src','z_fft','data_save',&
             index,lbou=0,crl=data_save)
#elif SINGLEP == 1
        call chmalloc('colfft_util.src','z_fft','data_save',&
             index,lbou=0,cr4=data_save)
#endif 
        data_save(0:index-1) = data(0:index-1)

#if DOUBLEP == 1
        call dfftw_plan_many_dft(fftw_zb_plan, rank, n, howmany, data, inembed, &
             istride, idist, data, onembed, ostride, odist, FFTW_BACKWARD, &
             fftw_opt)
#elif SINGLEP == 1
#if KEY_COLFFT_NOSP==0
        call sfftw_plan_many_dft(fftw_zb_plan, rank, n, howmany, data, inembed, &
             istride, idist, data, onembed, ostride, odist, FFTW_BACKWARD, &
             fftw_opt)
#endif 
#endif 

        data(0:index-1) = data_save(0:index-1)
#if DOUBLEP == 1
        call chmdealloc('colfft_util.src','z_fft','data_save',&
             index,crl=data_save)
#elif SINGLEP == 1
        call chmdealloc('colfft_util.src','z_fft','data_save',&
             index,cr4=data_save)
#endif 
     endif

     !$omp parallel do schedule(static) private(k, index)
     do k = 0, len3-1
        index = 2*k*size1*size2
#if DOUBLEP == 1
        call dfftw_execute_dft(fftw_zb_plan, data(index), data(index))
#elif SINGLEP == 1
#if KEY_COLFFT_NOSP==0
        call sfftw_execute_dft(fftw_zb_plan, data(index), data(index))
#endif 
#endif 
     enddo
     !$omp end parallel do

  endif
#else /**/
  ! Native CHARMM FFT
  ! APH 9/18/2014
  ! NOTE: No OpenMP parallelization here because the native implementation is
  !       not thread safe. To make it thread safe, we need to give each thread
  !       its own (fft_table_1, fft_table_2, fft_table_3) -lists.
  do j = min3, max3
     do i = min2, max2
        index = 2*((j-min3)*size1*size2 + (i-min2)*size1)
#if DOUBLEP == 1
        call fft_1d_cc( isign, len1, fft_table_3, data( index ), &
             ifac3 )
#elif SINGLEP == 1
        call wrndie(-5,'<colfft_util>','z_fft native FFT not implemented in single precision')
#endif 
     enddo
  enddo
#endif 
end subroutine ADD_PREC(z_fft)

!-----------------------------------------------------------------------
!         Communicate and transpose fft data
!-----------------------------------------------------------------------
subroutine ADD_PREC(transpose)(data, from_partition, transposed_data, &
     to_partition, comm_group, recv_buffer, send_buffer)
  use parallel
#if KEY_PARALLEL==1
  use mpi              
#endif
#if KEY_DOMDEC_GPU==1
  !    use domdec_util_gpu_mod,only:range_start, range_stop  
#endif
#if KEY_DOMDEC_GPU==1
  !    use domdec_common,only:q_gpu                       
#endif
  implicit none
  ! Input / Output
#if DOUBLEP == 1
  real(chm_real), intent(in) :: data(0:*)
  real(chm_real), intent(out) :: transposed_data(0:*)
  real(chm_real), intent(inout) :: recv_buffer(0:*), send_buffer(0:*)
#elif SINGLEP == 1
  real(chm_real4), intent(in) :: data(0:*)
  real(chm_real4), intent(out) :: transposed_data(0:*)
  real(chm_real4), intent(inout) :: recv_buffer(0:*), send_buffer(0:*)
#endif 
  integer, intent(in) :: from_partition
  integer, intent(in) :: to_partition
  integer, intent(in) :: comm_group
  ! Parameters
#if KEY_PARALLEL==1
  integer,parameter :: int_mpi = 4
#if DOUBLEP == 1
  integer,parameter :: mpi_real_type = mpi_real8
#elif SINGLEP == 1
  integer,parameter :: mpi_real_type = mpi_real4
#endif 
#endif 
  ! Variables
  integer alt_c
  integer alt_from_min,alt_from_max,alt_from_len
  integer alt_r_to_min,alt_r_to_max,alt_r_to_len
  integer alt_s_from_min,alt_s_from_max,alt_s_from_len
  integer alt_to_min,alt_to_max,alt_to_len
  integer alt_from_size, alt_to_size, con_from_size, con_to_size, pre_from_size
  integer begin
  integer cg
  integer con_from_min,con_from_max,con_from_len
  integer con_to_min,con_to_max,con_to_len
  integer endd
  integer i,ito,ifrom
  integer j,jto,jfrom
  integer k,kto,kfrom
  integer p
  integer pre_c,pre_from_min,pre_from_max,pre_from_len
  integer proc_offset
  integer alt_from_halo, alt_to_halo, con_from_halo, con_to_halo, pre_from_halo
  integer pre_to_size, pre_to_halo
#if KEY_PARALLEL==1
  integer(kind=int_mpi),parameter :: TRANSPOSE_TAG=83
  integer(kind=int_mpi) :: sender,receiver,recv_counter,send_counter,irecv
  integer(kind=int_mpi) :: mpi_ierr,buf_size,ipe
  integer step
#endif 
#if DOUBLEP == 1
  real(chm_real), allocatable, dimension(:) :: data_check
#elif SINGLEP == 1
  real(chm_real4), allocatable, dimension(:) :: data_check
#endif 

  alt_c        = altered_coor(comm_group)
  alt_from_min = dec_gridmin(alt_c, mynod, from_partition)
  alt_from_max = dec_gridmax(alt_c, mynod, from_partition)
  alt_from_size= dec_csize(alt_c, mynod, from_partition)
  alt_from_len = alt_from_max - alt_from_min + 1

  alt_from_halo = dec_halomin(alt_c, mynod, from_partition)

  alt_to_min   = dec_gridmin(alt_c, mynod, to_partition)
  alt_to_max   = dec_gridmax(alt_c, mynod, to_partition)
  alt_to_size  = dec_csize(alt_c, mynod, to_partition)
  alt_to_len   = alt_to_max - alt_to_min + 1

  alt_to_halo = dec_halomin(alt_c, mynod, to_partition)

  begin        = comm_group_begin(mynod, comm_group)
  cg           = comm_group
  con_from_min = con_gridmin(from_partition)
  con_from_max = con_gridmax(from_partition)
  con_from_size= con_csize(from_partition)
  con_from_len = con_from_max - con_from_min + 1

  con_from_halo = con_halomin(from_partition)

  con_to_min   = con_gridmin(to_partition)
  con_to_max   = con_gridmax(to_partition)
  con_to_size  = con_csize(to_partition)
  con_to_len   = con_to_max - con_to_min + 1

  con_to_halo = con_halomin(to_partition)

  endd         = comm_group_end(mynod, comm_group)
  p            = mynod
  pre_c        = preserved_coor(comm_group)
  pre_from_min = dec_gridmin(pre_c, mynod, from_partition)
  pre_from_max = dec_gridmax(pre_c, mynod, from_partition)
  pre_from_size= dec_csize(pre_c, mynod, from_partition)
  pre_from_len = pre_from_max - pre_from_min + 1

  pre_from_halo = dec_halomin(pre_c, mynod, from_partition)

  pre_to_halo = dec_halomin(pre_c, mynod, to_partition)
  pre_to_size= dec_csize(pre_c, mynod, to_partition)

#if KEY_PARALLEL==1
  step         = comm_group_stride( comm_group )  
#endif

  call ADD_PREC(allocate_transpose)()

#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_start('barrier')  
#endif
!!$    call mpi_barrier(comm_charmm, mpi_ierr)
#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_stop()  
#endif

#if KEY_PARALLEL==1
#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_start('mpi recv')  
#endif
  proc_offset = 0
  recv_counter = 0
  do sender = begin, endd, step
     if ( sender  /=  mynod ) then
        recv_counter = recv_counter + 1
        pe_list(recv_counter)=sender
        recv_offset( sender ) = proc_offset
        buf_size = alt_to_len * pre_from_len * &
             ( dec_gridmax(alt_c, sender, from_partition) - &
             dec_gridmin(alt_c, sender, from_partition) + 1 )
        call MPI_IRECV( recv_buffer( 2*recv_offset( sender ) ), &
             2*buf_size, &
             mpi_real_type, &
             sender, &
             TRANSPOSE_TAG, &
             COMM_CHARMM, &
             recv_request( recv_counter ), &
             mpi_ierr )
        proc_offset = proc_offset + buf_size
     endif
  enddo
#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_stop()  
#endif
#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_start('mpi send')  
#endif
  proc_offset = 0
  send_counter = 0
  do receiver = begin, endd, step
     if ( receiver  /=  mynod ) then
        alt_r_to_min = dec_gridmin(alt_c,receiver,to_partition)
        alt_r_to_max = dec_gridmax(alt_c,receiver,to_partition)
        alt_r_to_len = alt_r_to_max - alt_r_to_min + 1
        !$omp parallel do schedule(static) private(k, j, i, kto, kfrom, jto, jfrom)
        do k = pre_from_min, pre_from_max
           kto  = (k - pre_from_min) * alt_from_len + proc_offset
           kfrom= (k - pre_from_halo) * alt_from_size * con_from_size
           do j = alt_from_min, alt_from_max
              jto   = kto   + (j - alt_from_min) &
                   -alt_r_to_min*pre_from_len*alt_from_len
              jfrom = kfrom + (j - alt_from_halo) * con_from_size &
                   - con_from_halo
              do i = alt_r_to_min, alt_r_to_max
                 send_buffer( 2*(jto+ i*pre_from_len*alt_from_len) ) = data( 2*(jfrom  + i) )
                 send_buffer( 2*(jto+ i*pre_from_len*alt_from_len) +1) = data( 2*(jfrom  + i) +1)
              enddo
           enddo
        enddo
        !$omp end parallel do
        send_counter = send_counter + 1
        buf_size = pre_from_len * alt_from_len * alt_r_to_len
        !write (0,*) 'mynod,2*buf_size=',mynod,2*buf_size
        !          call MPI_SEND( send_buffer( 2*proc_offset ), 2*buf_size, &
        !               mpi_real_type,receiver,TRANSPOSE_TAG,COMM_CHARMM, mpi_ierr )
        call MPI_ISEND( send_buffer( 2*proc_offset ), 2*buf_size, &
             mpi_real_type,receiver,TRANSPOSE_TAG,COMM_CHARMM, &
             send_request( send_counter ), mpi_ierr )
        proc_offset = proc_offset + buf_size
     endif
  enddo
#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_stop()  
#endif
!!$
#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_start('mpi wait recv')  
#endif
  call mpi_waitall(recv_counter,recv_request,mpi_statuses_ignore,mpi_ierr)
#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_stop()  
#endif
!!$
#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_start('mpi wait send')  
#endif
  call MPI_Waitall(send_counter,send_request,mpi_statuses_ignore,mpi_ierr)
#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_stop()  
#endif
#endif 

#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_start('local')  
#endif
  ! Local transpose
!!$    call ADD_PREC(transpose_local_zyx)(alt_to_max-alt_to_min+1, alt_from_max-alt_from_min+1, &
!!$         pre_from_max-pre_from_min+1, data, &
!!$         (/ alt_to_min-con_from_halo, alt_from_min-alt_from_halo, pre_from_min-pre_from_halo /), &
!!$         (/ con_from_size, alt_from_size, pre_from_size /), &
!!$         transposed_data, &
!!$         (/ alt_to_min-alt_to_halo, alt_from_min-con_to_halo, pre_from_min-pre_to_halo /), &
!!$         (/ alt_to_size, con_to_size, pre_to_size /))
  call ADD_PREC(transpose_yzx)(alt_to_min-con_from_halo, alt_from_min-alt_from_halo, &
       pre_from_min-pre_from_halo, &
       alt_from_min-con_to_halo,  pre_from_min-pre_to_halo, alt_to_min-alt_to_halo,&
       alt_to_max-alt_to_min+1, alt_from_max-alt_from_min+1, pre_from_max-pre_from_min+1, &
       con_from_size, alt_from_size, data, &
       con_to_size, pre_to_size, &
       transposed_data)

#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_stop()  
#endif

!!$!$cmp parallel
!!$!$cmp do schedule(static) private(j, i, kto, kfrom, jto, jfrom)
!!$    do k = pre_from_min, pre_from_max
!!$       kto   = (k - pre_to_halo)*con_to_size
!!$       kfrom = (k - pre_from_halo)*alt_from_size*con_from_size
!!$       do j = alt_from_min, alt_from_max
!!$          jto   = kto + (j - con_to_halo)- &
!!$               alt_to_halo*pre_to_size*con_to_size
!!$          jfrom = kfrom + (j - alt_from_halo) * con_from_size &
!!$               - con_from_halo
!!$          do i = alt_to_min, alt_to_max
!!$             transposed_data( 2*(jto+i*pre_to_size*con_to_size) ) = data( 2*(jfrom + i) )
!!$             transposed_data( 2*(jto+i*pre_to_size*con_to_size) + 1) = data( 2*(jfrom + i) + 1)
!!$          enddo
!!$       enddo
!!$    enddo
!!$!$cmp end do
!!$!$cmp end parallel

#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_start('copy')  
#endif
#if KEY_PARALLEL==1
  do irecv=1,recv_counter
!!$       call mpi_waitany(recv_counter,recv_request,ipe, &
!!$            recv_status(:,irecv), &
!!$            mpi_ierr)
     ipe = irecv    ! APH: added
     sender=pe_list(ipe)
     alt_s_from_min = dec_gridmin(alt_c,sender,from_partition)
     alt_s_from_max = dec_gridmax(alt_c,sender,from_partition)
     alt_s_from_len = alt_s_from_max - alt_s_from_min + 1
     !$omp parallel do schedule(static) private(i, k, j, ito, ifrom, kto, kfrom)
     do i = 0,alt_to_len-1
        ito   = i*pre_to_size*con_to_size
        ifrom = i*pre_from_len*alt_s_from_len + recv_offset(sender)
        do k = 0, pre_from_len-1
           kto   = ito + k*con_to_size
           kfrom = ifrom + k*alt_s_from_len
           do j = alt_s_from_min, alt_s_from_max
              transposed_data( 2*(kto + j - con_to_halo) ) = &
                   recv_buffer( 2*(kfrom + j - alt_s_from_min) )
              transposed_data( 2*(kto + j - con_to_halo) + 1) = &
                   recv_buffer( 2*(kfrom + j - alt_s_from_min) + 1)
           enddo
        enddo
     enddo
     !$omp end parallel do
  enddo
!!$    call MPI_Waitall(send_counter,send_request,send_status,mpi_ierr)
#endif 
#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_stop()  
#endif
end subroutine ADD_PREC(transpose)

! *
! * Transpose 3d array locally, data_in(x,y,z) -> data_out(y,z,x).
! * Loops are in order z,y,x
! * 
! * Size of the transpose region
! * nx, ny, nz
! *
! * Start position of the transpose region (in order x, y, z)
! * start_in(1:3)
! * start_out(1:3)
! *
! * Dimensions of the arrays (in order x, y, z)
! * size_in(1:3)
! * size_out(1:3)
! *
subroutine ADD_PREC(transpose_local_zyx)(nx, ny, nz, data_in, start_in, size_in, &
     data_out, start_out, size_out)
  implicit none
  ! Input / Output
  integer, intent(in) :: nx, ny, nz
  integer, intent(in) :: size_in(3), start_in(3)
  integer, intent(in) :: size_out(3), start_out(3)
#if DOUBLEP == 1
  real(chm_real), intent(in) :: data_in(0:*)
  real(chm_real), intent(out) :: data_out(0:*)
#elif SINGLEP == 1
  real(chm_real4), intent(in) :: data_in(0:*)
  real(chm_real4), intent(out) :: data_out(0:*)
#endif 
  ! Variables
  integer out_pos, in_pos, out_pos_add
  integer zpos_in, zypos_in, zpos_out, zypos_out
  integer x, y, z

  out_pos_add = 2*size_out(2)*size_out(3)
  !$omp parallel do schedule(static) private(z, y, x, zpos_in, zpos_out, zypos_in, zypos_out, &
  !$omp&                                     out_pos, in_pos)
  do z=0,nz-1
     zpos_in  = (z + start_in(3))*size_in(1)*size_in(2)
     zpos_out = (z + start_out(3))*size_out(2)
     do y=0,ny-1
        zypos_in  = zpos_in + (y + start_in(2))*size_in(1) + start_in(1)
        zypos_out = zpos_out + (y + start_out(2)) + start_out(1)*size_out(2)*size_out(3)
        out_pos = 2*zypos_out
        in_pos = 2*zypos_in
        do x=0,nx-1
!!$             data_out(2*(zypos_out + x*size_out(2)*size_out(3))) = data_in(2*(zypos_in + x))
!!$             data_out(2*(zypos_out + x*size_out(2)*size_out(3))+1) = data_in(2*(zypos_in + x)+1)
           data_out(out_pos) = data_in(in_pos)
           data_out(out_pos+1) = data_in(in_pos+1)
           out_pos = out_pos + out_pos_add
           in_pos = in_pos + 2
        enddo
     enddo
  enddo
  !$omp end parallel do

  return
end subroutine ADD_PREC(transpose_local_zyx)

! *
! * New faster tiled transpose
! *
subroutine ADD_PREC(transpose_yzx)(src_x0, src_y0, src_z0, dst_x0, dst_y0, dst_z0, xlen, ylen, zlen, &
     xsize_in, ysize_in, data_in, xsize_out, ysize_out, data_out)
  implicit none
  ! Input / Output
  integer, intent(in) :: src_x0, src_y0, src_z0, dst_x0, dst_y0, dst_z0, xlen, ylen, zlen
  integer, intent(in) :: xsize_in, ysize_in, xsize_out, ysize_out
#if DOUBLEP == 1
  real(chm_real), intent(in) :: data_in(0:*)
  real(chm_real), intent(out) :: data_out(0:*)
#elif SINGLEP == 1
  real(chm_real4), intent(in) :: data_in(0:*)
  real(chm_real4), intent(out) :: data_out(0:*)
#endif 
  ! Functions
  integer omp_get_thread_num
  ! Variables
  integer tilebuf_pos, ntilex, ntiley, ntile, tile, tmp, z, tiley, tilex
  integer xstart, xend, ystart, yend, y, src_pos, dst_pos, x

  ntilex = (xlen-1)/tiledim+1
  ntiley = (ylen-1)/tiledim+1
  ntile = ntilex*ntiley*zlen
  !$omp parallel private(tilebuf_pos)
#ifdef _OPENMP
  tilebuf_pos = omp_get_thread_num()*tiledim*tiledim
#else /**/
  tilebuf_pos = 0
#endif 
  !$omp do schedule(static) private(tile, tmp, z, tiley, tilex, xstart, xend, ystart, yend, y, &
  !$omp&                            src_pos, dst_pos, x)
  do tile=0,ntile-1
     ! Calculate position (tilex, tiley, z)
     tmp = tile
     z = tmp/(ntilex*ntiley)
     tmp = tmp - z*(ntilex*ntiley)
     tiley = tmp/ntilex
     tilex = tmp - tiley*ntilex
     ! Calculate start and end
     xstart = tilex*tiledim
     xend   = min((tilex+1)*tiledim, xlen)
     ystart = tiley*tiledim
     yend   = min((tiley+1)*tiledim, ylen)
     ! Read in data
     do y=ystart,yend-1
        src_pos = (xstart+src_x0 + (y+src_y0 + (z+src_z0)*ysize_in)*xsize_in)*2
        dst_pos = ((y-ystart)*tiledim + tilebuf_pos)*2
        do x=xstart,xend-1
           ADD_PREC(tilebuf)(dst_pos)     = data_in(src_pos)
           ADD_PREC(tilebuf)(dst_pos + 1) = data_in(src_pos + 1)
           dst_pos = dst_pos + 2
           src_pos = src_pos + 2
        enddo
     enddo
     ! Write out data
     do x=xstart,xend-1
        src_pos = ((x-xstart) + tilebuf_pos)*2
        dst_pos = (ystart+dst_x0 + (z+dst_y0 + (dst_z0 + x)*ysize_out)*xsize_out)*2
        do y=ystart,yend-1
           data_out(dst_pos)     = ADD_PREC(tilebuf)(src_pos)
           data_out(dst_pos + 1) = ADD_PREC(tilebuf)(src_pos + 1)
           dst_pos = dst_pos + 2
           src_pos = src_pos + 2*tiledim
        enddo
     enddo
  enddo
  !$omp end do
  !$omp end parallel

  return
end subroutine ADD_PREC(transpose_yzx)

!-----------------------------------------------------------------------
!           FTRANSPOSE
!-----------------------------------------------------------------------
subroutine ADD_PREC(ftranspose)(data, from_partition, transposed_data, &
     to_partition, comm_group, recv_buffer, send_buffer)
  use parallel
#if KEY_PARALLEL==1
  use mpi       
#endif
#if KEY_DOMDEC_GPU==1
  !    use domdec_util_gpu_mod,only:range_start, range_stop  
#endif
#if KEY_DOMDEC_GPU==1
  !    use domdec_common,only:q_gpu                       
#endif
  implicit none
  ! Input / Output
#if DOUBLEP == 1
  real(chm_real), intent(in) :: data(0:*)
  real(chm_real), intent(out) :: transposed_data(0:*)
  real(chm_real), intent(inout) :: recv_buffer(0:*), send_buffer(0:*)
#elif SINGLEP == 1
  real(chm_real4), intent(in) :: data(0:*)
  real(chm_real4), intent(out) :: transposed_data(0:*)
  real(chm_real4), intent(inout) :: recv_buffer(0:*), send_buffer(0:*)
#endif 
  integer, intent(in) :: from_partition, to_partition
  integer, intent(in) :: comm_group
  ! Parameters
  integer,parameter :: int_mpi = 4
#if KEY_PARALLEL==1
#if DOUBLEP == 1
  integer,parameter :: mpi_real_type = mpi_real8
#elif SINGLEP == 1
  integer,parameter :: mpi_real_type = mpi_real4
#endif 
#endif 
  ! Variables
  integer alt_c
  integer alt_from_min, alt_from_max, alt_from_len
  integer alt_r_to_min, alt_r_to_max, alt_r_to_len
  integer alt_s_from_min, alt_s_from_max, alt_s_from_len
  integer alt_to_min, alt_to_max, alt_to_len
  integer begin,endd,cg
  integer con_from_min, con_from_max, con_from_len
  integer con_to_min, con_to_max, con_to_len
  integer alt_from_size, alt_to_size, con_from_size, con_to_size, pre_from_size
  integer i,ifrom,ito
  integer j,jfrom,jto
  integer k,kfrom,kto
  integer p
  integer pre_c
  integer pre_from_min, pre_from_max, pre_from_len
  integer proc_offset
  integer con_to_halo, con_from_halo, alt_from_halo, pre_from_halo, alt_to_halo
  integer pre_to_min, pre_to_halo, pre_to_size
#if KEY_PARALLEL==1
  integer step
  !    integer trans_timer,waitall_timer,waitany_timer,postrcv_timer, &
  !         send_timer,packsend_timer
  integer(kind=int_mpi),parameter :: TRANSPOSE_TAG=83
  integer(kind=int_mpi) :: sender,receiver,recv_counter,send_counter,irecv
  integer(kind=int_mpi) :: mpi_ierr,size,ipe
#endif 

  alt_c        = altered_coor( comm_group )
  alt_from_min = dec_gridmin(alt_c, mynod, from_partition)
  alt_from_max = dec_gridmax(alt_c, mynod, from_partition)
  alt_from_size= dec_csize(alt_c, mynod, from_partition)
  alt_from_len = alt_from_max - alt_from_min + 1
  alt_from_halo= dec_halomin(alt_c, mynod, from_partition)
  alt_to_min   = dec_gridmin(alt_c, mynod, to_partition)
  alt_to_max   = dec_gridmax(alt_c, mynod, to_partition)
  alt_to_size  = dec_csize(alt_c, mynod, to_partition)
  alt_to_len   = alt_to_max - alt_to_min + 1
  alt_to_halo  = dec_halomin(alt_c, mynod, to_partition)
  begin        = comm_group_begin( mynod, comm_group )
  cg           = comm_group
  con_from_min = con_gridmin( from_partition )
  con_from_max = con_gridmax( from_partition )
  con_from_size = con_csize( from_partition )
  con_from_len = con_from_max - con_from_min + 1
  con_from_halo= con_halomin(from_partition)
  con_to_min   = con_gridmin( to_partition )
  con_to_max   = con_gridmax( to_partition )
  con_to_size  = con_csize( to_partition )
  con_to_len   = con_to_max - con_to_min + 1
  con_to_halo  = con_halomin(to_partition)
  endd         = comm_group_end( mynod, comm_group )
  p            = mynod
  pre_c        = preserved_coor( comm_group )
  pre_from_min = dec_gridmin(pre_c, mynod, from_partition)
  pre_from_max = dec_gridmax(pre_c, mynod, from_partition)
  pre_from_size= dec_csize(pre_c, mynod, from_partition)
  pre_from_len = pre_from_max - pre_from_min + 1
  pre_from_halo= dec_halomin(pre_c, mynod, from_partition)
  pre_to_min   = dec_gridmin(pre_c, mynod, to_partition)
  pre_to_halo  = dec_halomin(pre_c, mynod, to_partition)
  pre_to_size  = dec_csize(pre_c, mynod, to_partition)
#if KEY_PARALLEL==1
  step         = comm_group_stride( comm_group )  
#endif

  call ADD_PREC(allocate_transpose)()

#if KEY_PARALLEL==1
  proc_offset = 0
  recv_counter = 0

  !------ Post receives ---------------------------------------
  do sender = begin, endd, step
     if ( sender  /=  mynod ) then
        recv_counter = recv_counter + 1
        pe_list(recv_counter)=sender
        recv_offset( sender ) = proc_offset
        size = alt_to_len * pre_from_len * &
             ( dec_gridmax(alt_c, sender, from_partition) - &
             dec_gridmin(alt_c, sender, from_partition) + 1 )
        call MPI_IRECV( recv_buffer( 2*recv_offset( sender ) ), 2*size, &
             mpi_real_type, sender, TRANSPOSE_TAG, COMM_CHARMM, &
             recv_request( recv_counter ), mpi_ierr )
        proc_offset = proc_offset + size
     endif
  enddo


  !------  Send data ---------------------------------------------
  proc_offset = 0
  send_counter = 0

  do receiver = begin, endd, step
     if ( receiver  /=  mynod ) then
        alt_r_to_min = dec_gridmin(alt_c,receiver,to_partition)
        alt_r_to_max = dec_gridmax(alt_c,receiver,to_partition)
        alt_r_to_len = alt_r_to_max - alt_r_to_min + 1
        !$omp parallel do schedule(static) private(j, k, i, jto, jfrom, kto, kfrom)
        do j = alt_from_min, alt_from_max
           jto = j - alt_from_min + proc_offset
           jfrom = (j - alt_from_halo)* pre_from_size * con_from_size
           do k = pre_from_min, pre_from_max
              kto = jto + &
                   (k-pre_from_min)*alt_r_to_len*alt_from_len &
                   -alt_r_to_min*alt_from_len
              kfrom = jfrom + (k - pre_from_halo)* &
                   con_from_size
              do i = alt_r_to_min, alt_r_to_max
                 send_buffer(2*(kto + i*alt_from_len))   = data(2*(kfrom+i))
                 send_buffer(2*(kto + i*alt_from_len)+1) = data(2*(kfrom+i)+1)
              enddo
           enddo
        enddo
        !$omp end parallel do
        send_counter = send_counter + 1
        size = pre_from_len * alt_from_len * alt_r_to_len
        !          call MPI_SEND( send_buffer( 2*proc_offset ), 2*size, &
        !               mpi_real_type, receiver, TRANSPOSE_TAG,COMM_CHARMM, mpi_ierr )
        call MPI_ISEND( send_buffer( 2*proc_offset ), 2*size, &
             mpi_real_type, receiver, TRANSPOSE_TAG,COMM_CHARMM, &
             send_request( send_counter ), mpi_ierr )
        proc_offset = proc_offset + size
     endif
  enddo
  call mpi_waitall(recv_counter,recv_request,mpi_statuses_ignore,mpi_ierr)
  call MPI_Waitall(send_counter,send_request,mpi_statuses_ignore,mpi_ierr)
#endif 

  !------  Transpose local data --------------------------------
#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_start('local')  
#endif
  call ADD_PREC(ftranspose_local_yxz)(alt_to_max-alt_to_min+1, pre_from_max-pre_from_min+1, &
       alt_from_max-alt_from_min+1, data, &
       (/ alt_to_min-con_from_halo, pre_from_min-pre_from_halo, alt_from_min-alt_from_halo /), &
       (/ con_from_size, pre_from_size, alt_from_size /), &
       transposed_data, &
       (/ alt_to_min-alt_to_halo, pre_from_min-pre_to_halo, alt_from_min-con_to_halo /), &
       (/ alt_to_size, pre_to_size, con_to_size /))
!!$!$cmp parallel
!!$!$cmp do schedule(static) private(k, i, jto, jfrom, kto, kfrom)
!!$    do j = alt_from_min, alt_from_max
!!$       jto   = j - con_to_halo
!!$       jfrom = (j - alt_from_halo)*pre_from_size*con_from_size
!!$       do k = pre_from_min, pre_from_max
!!$          kto   = jto + (k-pre_to_halo)*alt_to_size*con_to_size &
!!$               -alt_to_halo*con_to_size
!!$          kfrom = jfrom + (k-pre_from_halo)*con_from_size - con_from_halo
!!$          do i = alt_to_min, alt_to_max
!!$             transposed_data( 2*(kto+i*con_to_size)) = data(2*(kfrom + i))
!!$             transposed_data( 2*(kto+i*con_to_size)+1) = data(2*(kfrom + i)+1)
!!$          enddo
!!$       enddo
!!$    enddo
!!$!$cmp end do
!!$!$cmp end parallel
#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_stop()  
#endif

#if KEY_PARALLEL==1
  !------  Receive and process data --------------------------------
#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_start('copy')  
#endif
  do irecv=1,recv_counter
!!$       call mpi_waitany(recv_counter,recv_request,ipe, &
!!$            recv_status(:,irecv), &
!!$            mpi_ierr)
     ipe = irecv    ! APH: added
     sender=pe_list(ipe)
     size = alt_to_len * pre_from_len * &
          ( dec_gridmax(alt_c, sender, from_partition) - &
          dec_gridmin(alt_c, sender, from_partition) + 1 )
     alt_s_from_min = dec_gridmin(alt_c, sender, from_partition)
     alt_s_from_max = dec_gridmax(alt_c, sender, from_partition)
     alt_s_from_len = alt_s_from_max - alt_s_from_min + 1
     !$omp parallel do schedule(static) private(k, i, j, kto, kfrom, ito, ifrom)
     do k = pre_from_min, pre_from_max
        kto   = (k-pre_to_halo)*alt_to_size*con_to_size
        kfrom = recv_offset( sender ) + &
             (k-pre_from_min)*alt_to_len*alt_s_from_len
        do i = alt_to_min, alt_to_max
           ito   = kto   + (i -alt_to_halo)*con_to_size - con_to_halo
           ifrom = kfrom+(i -alt_to_min)*alt_s_from_len
           do j=alt_s_from_min,alt_s_from_max
              transposed_data( 2*(ito + j) ) &
                   = recv_buffer(2*(ifrom+j-alt_s_from_min))
              transposed_data( 2*(ito + j)+1) &
                   = recv_buffer(2*(ifrom+j-alt_s_from_min)+1)
           enddo
        enddo
     enddo
     !$omp end parallel do
  enddo
#if KEY_DOMDEC_GPU==1
!!$    if (q_gpu) call range_stop()  
#endif

  !------  Clean up with Waitall -------------------------------
!!$    call MPI_Waitall(send_counter,send_request,send_status,mpi_ierr)
#endif 

  return

end subroutine ADD_PREC(ftranspose)

! *
! * Transpose 3d array locally, data_in(x,y,z) -> data_out(z,x,y).
! * Loops are in order y,x,z
! * 
! * Size of the transpose region
! * nx, ny, nz
! *
! * Start position of the transpose region (in order x, y, z)
! * start_in(1:3)
! * start_out(1:3)
! *
! * Dimensions of the arrays (in order x, y, z)
! * size_in(1:3)
! * size_out(1:3)
! *
subroutine ADD_PREC(ftranspose_local_yxz)(nx, ny, nz, data_in, start_in, size_in, &
     data_out, start_out, size_out)
  implicit none
  ! Input / Output
  integer, intent(in) :: nx, ny, nz
  integer, intent(in) :: size_in(3), start_in(3)
  integer, intent(in) :: size_out(3), start_out(3)
#if DOUBLEP == 1
  real(chm_real), intent(in) :: data_in(0:*)
  real(chm_real), intent(out) :: data_out(0:*)
#elif SINGLEP == 1
  real(chm_real4), intent(in) :: data_in(0:*)
  real(chm_real4), intent(out) :: data_out(0:*)
#endif
  ! Variables
  integer out_pos, in_pos, in_pos_add
  integer ypos_in, yxpos_in, ypos_out, yxpos_out
  integer x, y, z

  in_pos_add = 2*size_in(1)*size_in(2)
  !$omp parallel do schedule(static) private(y, x, z, ypos_in, ypos_out, yxpos_in, yxpos_out, &
  !$omp&                                     out_pos, in_pos)
  do y=0,ny-1
     ypos_in = (y + start_in(2))*size_in(1)
     ypos_out = (y + start_out(2))*size_out(3)*size_out(1)
     do x=0,nx-1
        yxpos_in = ypos_in + (x+start_in(1)) + start_in(3)*size_in(1)*size_in(2)
        yxpos_out = ypos_out + (x+start_out(1))*size_out(3) + start_out(3)
        out_pos = 2*(yxpos_out)
        in_pos = 2*(yxpos_in)
        do z=0,nz-1
!!$             data_out(2*(yxpos_out + z))   = data_in(2*(yxpos_in + z*size_in(1)*size_in(2)))
!!$             data_out(2*(yxpos_out + z)+1) = data_in(2*(yxpos_in + z*size_in(1)*size_in(2))+1)
           data_out(out_pos)   = data_in(in_pos)
           data_out(out_pos+1) = data_in(in_pos+1)
           out_pos = out_pos + 2
           in_pos = in_pos + in_pos_add
        enddo
     enddo
  enddo
  !$omp end parallel do

  return
end subroutine ADD_PREC(ftranspose_local_yxz)

! ##ENDEX (exp_precision)
