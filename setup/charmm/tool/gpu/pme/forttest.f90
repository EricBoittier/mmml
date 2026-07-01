program forttest
  implicit none
!  integer(kind=4),parameter :: n = 10000
!  integer(kind=4),parameter :: n = 17491
  integer(kind=4),parameter :: n = 23558 
!  integer(kind=4),parameter :: n = 41993
!  integer(kind=4),parameter :: n = 78537
!  integer(kind=4),parameter :: nmesh = 16
  integer(kind=4),parameter :: nmesh = 64
!  integer(kind=4),parameter :: nmesh = 90
! 16384  32768 65536
  integer(kind=4),parameter :: nat = 1
  integer(kind=4) :: natchangeflag
  integer(kind=4) :: periodicflag
  integer(kind=4) :: atype(1:n)
  integer(kind=4) :: tblno, step, num_calc
  real(kind=8) :: x(1:3,1:n)
  real(kind=8) :: q(1:n)
  real(kind=8) :: xq(4,1:n)
  real(kind=8) :: gvec(4,1:n)
  real(kind=8) :: bsp_mod(nmesh)
  real(kind=8) :: recip(9)
  real(kind=8) :: force(1:3,1:n)
  real(kind=8) :: rscale(1:nat*nat)
  real(kind=8) :: gscale(1:nat*nat)
  real(kind=8) :: xmax
  real(kind=8) :: dom_width(3)
  real(kind=8) :: alpha, radius_cut
  real(kind=8) :: rho, interval, volume
  real(kind=8) :: pot(n)
  real(kind=8) :: stress(3,3)

  integer(kind=4) :: i, j, k, i1, i2, i3, j1, j2, j3, k1, k2, k3, ntmp
  integer(kind=4) :: cnt
  real(kind=8) :: tmp, time1, time2, fcc(1:3,1:4)
  character(len=10) :: ctmp10
  character(len=8) :: ctmp8
  character(len=5) :: ctmp5
  integer(kind=4) :: value(8)
  integer(kind=4) :: order

  order=6
  natchangeflag = 0
  periodicflag = 1
  atype(1:n) = 1
  tblno = 3
  rscale(1) = 1.d0
  gscale(1) = 4.d0
  alpha = 0.31d0;
  rho = 0.100d0 ! water charge
!  rho = 0.035d0 ! water vdw
  force(1:3,1:n) = 0.d0
  volume = dble(n)/rho
  xmax = volume**(1.d0/3.d0)
  recip(1) = 1.d0/xmax
  recip(5) = 1.d0/xmax
  recip(9) = 1.d0/xmax
  recip(2) = 0.d0
  recip(3) = 0.d0
  recip(4) = 0.d0
  recip(6) = 0.d0
  recip(7) = 0.d0
  recip(8) = 0.d0

  dom_width(1:3) = xmax
  fcc(1:3,1) = 0.d0
  fcc(1:3,2:4) = 1.d0
  fcc(1,2) = 0.d0
  fcc(2,3) = 0.d0
  fcc(3,4) = 0.d0
  i1 = int((dble(n)*0.25d0)**(1.d0/3.d0))+1
  i2 = (i1)**3*4
  tmp = xmax/dble(i1)
  i3 = 0
  interval = tmp
  do j1 = 1, i1
     do j2 = 1, i1
        do j3 = 1, i1
           do k1 = 1, 4
              i3 = i3 + 1
              if (i3 > n) cycle
              x(1,i3) = tmp * dble(j1) + fcc(1,k1) * tmp * 0.5d0
              x(2,i3) = tmp * dble(j2) + fcc(2,k1) * tmp * 0.5d0
              x(3,i3) = tmp * dble(j3) + fcc(3,k1) * tmp * 0.5d0
           end do
        end do
     end do
  end do

  do i = 1, n
     call random_number(tmp)
     x(1,i) = x(1,i) + tmp*0.1d0
!     x(1,i) = tmp * xmax
     call random_number(tmp)
     x(2,i) = x(2,i) + tmp*0.1d0
!     x(2,i) = tmp * xmax
     call random_number(tmp)
     x(3,i) = x(3,i) + tmp*0.1d0
!     x(3,i) = tmp * xmax
     call random_number(tmp)
     q(i) = tmp
  end do
  do i = 1, n
     call random_number(tmp)
     gvec(1,i) = tmp
     call random_number(tmp)
     gvec(2,i) = tmp
     call random_number(tmp)
     gvec(3,i) = tmp
     call random_number(tmp)
     gvec(4,i) = tmp
  end do
  do i = 1, nmesh
     call random_number(tmp)
     bsp_mod(i) = tmp
  end do
  tmp = 0.d0
  do i = 1, n
     tmp = tmp + q(i)
  end do
  tmp = tmp / dble(n)
  do i = 1, n
     q(i) = q(i) - tmp
  end do
  xq(1,1:n) = x(1,1:n)
  xq(2,1:n) = x(2,1:n)
  xq(3,1:n) = x(3,1:n)
  xq(4,1:n) = q(1:n)

  call date_and_time(ctmp8,ctmp10,ctmp5,value)
  time1 = dble(value(5)) * 3600d0 + dble(value(6)) * 60.d0 + dble(value(7)) + dble(value(8)) * 0.001d0

  force(1:3,1:n) = 0.d0
!  step = 100
  step = 10
  write(6,'(a,f10.2,a,f10.2,a,i8,a,i8)') &
       & ' xmax:',xmax, ' interval:',interval, ' particlenum:', n, ' step:', step
  num_calc = 1
  radius_cut = 10.d0

! for removing first time overhead
  call gpupme_(n,x,q,force,pot,nmesh,nmesh,nmesh,alpha,bsp_mod,&
       & bsp_mod,bsp_mod,volume,recip,alpha,order,-1,stress)
  force(1:3,1:n) = 0.d0
  call date_and_time(ctmp8,ctmp10,ctmp5,value)
  time1 = dble(value(5)) * 3600d0 + dble(value(6)) * 60.d0 + dble(value(7)) + dble(value(8)) * 0.001d0

  do j = 1, step
!     call gpuvdwcutoffdble_(&
!     call gpuvdwcutoff_(&
!          & x,n,atype,nat,gscale,rscale,tblno,dom_width,&
!          & periodicflag,natchangeflag,force,pot,0,1,n,radius_cut)
!     call gpuvdwdirect_(&
!          & x,n,atype,nat,gscale,rscale,tblno,dom_width,&
!          & periodicflag,natchangeflag,force,pot,0,1,n)
!     call gpuvdwcell_(&
!          & x,n,atype,nat,gscale,rscale,tblno,dom_width,&
!          & periodicflag,natchangeflag,force,pot,0,radius_cut,1,num_calc)
!     call gpuewredirect_(&
!          & x,n,q,alpha,tblno,dom_width,periodicflag,&
!          & natchangeflag,force,pot,0,1,n)
!     call gpuewrecutoff_(&
!          & x,n,q,alpha,tblno,dom_width,periodicflag,&
!          & natchangeflag,force,pot,0,1,n,radius_cut)
!     call gpuewrecell_(&
!          & x,n,q,alpha,tblno,dom_width,periodicflag,&
!          & natchangeflag,force,pot,0,1,n,30.d0,8,num_calc,1)
!          & xq,n,gvec,n,force)
     force(1:3,1:n) = 0.d0
     pot(1:n) = 0.d0
     call gpupme_(n,x,q,force,pot,nmesh,nmesh,nmesh,alpha,bsp_mod,&
          & bsp_mod,bsp_mod,volume,recip,alpha,order,-1,stress)
!  do i = 1, n
!     x(1,i) = x(1,i) + force(1,i)*10.d0
!     x(2,i) = x(2,i) + force(2,i)*10.d0
!     x(3,i) = x(3,i) + force(3,i)*10.d0
!  end do
  end do

  call date_and_time(ctmp8,ctmp10,ctmp5,value)
  time2 = dble(value(5)) * 3600d0 + dble(value(6)) * 60.d0 + dble(value(7)) + dble(value(8)) * 0.001d0

  write(6,'(a,f10.4,a,f10.4,a)') ' end. (time:',time2-time1,' sec, ',(time2-time1)/dble(step),' sec/step)'
  do i=1,10
     write(6,*) 'out(1,',i,'):',force(1,i)
     write(6,*) 'out(2,',i,'):',force(2,i)
     write(6,*) 'out(3,',i,'):',force(3,i)
     write(6,*) 'out(4,',i,'):',pot(i)
  enddo
  ntmp = (int(dble(n-1)/dble(256))+1)*256
  write(6,*) 'num_calc',num_calc
  write(6,'(a,f9.3,a,f8.3)') &
       & ' GFlops:',dble(ntmp)**2*dble(step)*38.d0/(time2-time1)/1000000000.d0,&
       & ' (cell):',dble(num_calc)*dble(step)*38.d0/(time2-time1)/1000000000.d0
  write(6,'(a,f9.3,a,f8.3)') &
       & ' GCalps:',dble(ntmp)**2*dble(step)/(time2-time1)/1000000000.d0,&
       & ' (cell):',dble(num_calc)*dble(step)/(time2-time1)/1000000000.d0

  stop
end program forttest
