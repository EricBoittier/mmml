      subroutine dispersion_egr(nn,x,gr,boxsiz,nlat,period) 
      use sccdftbsrc, A=> Adis, B=>Bdis, C=>Cdis,
     $    r0=> r0dis, rv=>rvdis
      implicit real*8(A-H,O-Z) 
CQC1  include 'maxima.inc'
!cccccccc
      integer nlat(3)
      real*8 x(3,NNDIM),gr(3,NNDIM)
      real*8 boxsiz(3,3)
      logical period
!ccccccccc common
CQC1  real*8 C6(NNDIM,NNDIM),Rvdw(NNDIM,NNDIM)
CQC1  real*8 A,B,C,r0,rv
CQC1  real*8 Edis
CQC1  logical dispers
!cccccccc local
      real*8 Catan,xm,rr
      real*8 C1,dgr,r,r2,dif(3) 
      real*8 conv,invpi
      real*8 h1,h2,h3,fdamp,g612
!ccccccccccccccccc
CQC1  common /disper/ Edis, dispers
CQC1  common /dispertmp/ A,B,C,r0,rv,C6,Rvdw
      conv= 0.529177/27.211
      invpi=1.0/3.14
!ccccccccccccccccccccccc 
! we caculate gradient gr, not force! 
! dE/dx = -F_x
       Edis = 0.0
       do j = 1,nn
         do i = 1,3
           gr(i,j) = 0.0
         end do
       end do
!
! calculate pairwise contribution i-j
! all values in eV and Angstr.
! convert r in Angstr
! Rvdw is already in Angstr
! then convert energy Edis from eV in H
!
!
        if (period) then
      do i=1,nn
       do j=1,nn
        rr = C/Rvdw(i,j)**A
        h3 = rv*0.5*Rvdw(i,j)**6

       DO nu =-nlat(1),nlat(1) 
       DO nv = -nlat(2),nlat(2) 
       DO nw = -nlat(3),nlat(3) 
!       DO nu =-1,1
!       DO nv = -1,1
!       DO nw = -1,1
         dif(1) = x(1,i) - 
     c    (x(1,j) +nu*boxsiz(1,1)+nv*boxsiz(2,1)+nw*boxsiz(3,1))
         dif(2) = x(2,i) - 
     c    (x(2,j) +nu*boxsiz(1,2)+nv*boxsiz(2,2)+nw*boxsiz(3,2))
         dif(3) = x(3,i) - 
     c    (x(3,j) +nu*boxsiz(1,3)+nv*boxsiz(2,3)+nw*boxsiz(3,3))
         r2 = dif(1)**2 + dif(2)**2 + dif(3)**2
          r = sqrt(r2)*0.52917
!
       if (r.ne.0.0) then
        call dis_e(r,rr,h3,C1)
        Edis = Edis + 0.5*C1*C6(i,j)/27.211
        do m=1,3
          xm =dif(m)*0.52917
         call dis_gr(r,rr,xm,h3,C1,dgr)
         gr(m,i) = gr(m,i) + dgr*C6(i,j)
!        gr(m,j) = gr(m,j) - dgr*C6(i,j)
       enddo
      endif !r.ne.0.0
!
      enddo !nw
      enddo !nv
      enddo !nu
      enddo !j
      enddo !i  

      else !period
      do i=1,nn
       do j=1,i-1
        rr = C/Rvdw(i,j)**A
        h3 = rv*0.5*Rvdw(i,j)**6
         r2 = 0.0
         do n = 1,3 
          dif(n) = x(n,i) - x(n,j)
          r2 = r2 + dif(n)**2
         enddo 
        r = sqrt(r2)*0.52917
        call dis_e(r,rr,h3,C1)
        Edis = Edis + C1*C6(i,j)/27.211
!
        do m=1,3
        xm =dif(m)*0.52917
        call dis_gr(r,rr,xm,h3,C1,dgr) 
        gr(m,i) = gr(m,i) + dgr*C6(i,j)
        gr(m,j) = gr(m,j) - dgr*C6(i,j)
       enddo
 
      enddo !j
      enddo !i
      endif !period
      return
      end
!
       subroutine dis_e(r,rr,h3,C1)
       use sccdftbsrc, A=> Adis, B=>Bdis, C=>Cdis,
     $    r0=> r0dis, rv=>rvdis
CQC1   include 'maxima.inc'
CQC1   real*8 A,B,C,r0,rv,Catan,C1,dgr,r,conv,invpi
       real*8 Catan,C1,dgr,r,conv,invpi
       real*8 h1,h2,h3,fdamp,g612,rr
CQC1   common /dispertmp/ A,B,C,r0,rv,C6,Rvdw
       conv= 0.529177/27.211
       invpi=1.0/3.14
       if(C.ge.0.0) then
!       old scaling
        Catan=  invpi*atan(A*(r-r0-rv))
        C1 =   -B/
     c   ((r-rv)**6 + (0.5-Catan)*C*(r-r0-rv)**2 )
       else
!       C is negative!! rv switches R**12, scaling function as in JCP!
        h1 = rr*r**A
        h2 = exp(h1)
        fdamp = 1 - h2
        g612 = h3/r**12 - 1/r**6
        C1 = g612*fdamp**B
      endif
!      write(*,*) C1/27.211
!        write(*,'(5I4,2F12.8)') i,j,nu,nv,nw,r,C1/27.211
!       return
       end
!
       subroutine dis_gr(r,rr,xm,h3,C1,dgr)
       use sccdftbsrc, A=> Adis, B=>Bdis, C=>Cdis,
     $    r0=> r0dis, rv=>rvdis

CQC1   include 'maxima.inc'
CQC1   real*8 C6(NNDIM,NNDIM),Rvdw(NNDIM,NNDIM)
CQC1   real*8 A,B,C,r0,rv,Catan,C1,dgr,r,conv,invpi
       real*8 Catan,C1,dgr,r,conv,invpi
       real*8 h1,h2,h3,fdamp,g612,xm,rr
CQC1   common /dispertmp/ A,B,C,r0,rv,C6,Rvdw
       conv= 0.529177/27.211
       invpi=1.0/3.14
         if(C.ge.0.0)then
          Catan=  invpi*atan(A*(r-r0-rv))
          dgr=   conv*B*
     c   ((6*xm*(-rv + r)**5)/r -
     c   (0.3184713375796178*A*C*xm*(-r0-rv + r)**2)/
     c   (r*(1 + A**2*(-r0-rv + r)**2)) + 
     c   (2*C*xm*(-r0-rv + r)*
     c   (0.5 - Catan))/r)/
     c   (  ((-rv + r)**6 + 
     c   C*(-r0-rv + r)**2*
     c   (0.5 - Catan))**2)
        else
        h1 = rr*r**A
        h2 = exp(h1)
        fdamp = 1 - h2
        g612 = h3/r**12 - 1/r**6
       dgr =   conv*(xm/r)*(
     c  g612*B*(fdamp**(B-1.0d0))*(-h2)*A*rr*r**(A-1.0d0)
     c  + (fdamp**B)*(6/r**7 - 12*h3/r**13) 
     c     )
        endif 
        return
        end
!
!
 

