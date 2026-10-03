!------------------------------------------------!
SUBROUTINE EXECUTE_SPLINE_INTERPOLATION()
!------------------------------------------------!
use eabfsrc
use reawri
use contrl
use consta
use coord
use parallel

implicit none
integer :: i,p,n


!Find Correct Bin
bin_num=nint(pos_particle)
!Assign Interploation Start Bin
i_start_lambd=bin_num-(sorder/2)
!Assign Interpolation End Bin
i_end_lambd=bin_num+(sorder/2)
!Convert Start Stop for Smaller Array
sp_gap=0
if(i_end_lambd .GT. sorder+2)then
   sp_gap=bin_num-(sorder/2)+1
   bin_num=(sorder/2)+1
endif
!Loop Through Interpolation Orders
spline_weight(:,:,:)=0.0
do p=2,sorder
   do i=i_start_lambd,i_end_lambd
      !Calculate 2nd Order Weghts
      if(p .EQ. 2)then
         do n=0,sorder-2
            call calc_u_lambd(i,n,u_lambd)
            if(u_lambd .LE. 2)then
            if(u_lambd .GE. 0)then
               call calc_m2_weight(u_lambd,n_lambd)
               spline_weight(i-sp_gap,p,n)=n_lambd
            endif
            endif
         enddo
      endif
      !Calculate Weight to Desired Order
      if(p .GT. 2)then
         do n=0,sorder-p
            call calc_u_lambd(i,n,u_lambd)
            if(u_lambd .LE. p)then
            if(u_lambd .GE. 0)then
               call calc_weight(u_lambd,p,spline_weight(i-sp_gap,p-1,n),spline_weight(i-sp_gap,p-1,n+1),n_lambd)
               spline_weight(i-sp_gap,p,n)=n_lambd
            endif
            endif
         enddo
      endif
   enddo
enddo

END SUBROUTINE

!------------------------------------------------!
SUBROUTINE CALC_U_LAMBD(dum1,dum2,dum3)
!------------------------------------------------!
use eabfsrc
use reawri
use contrl
use consta
use coord
use parallel

implicit none

integer :: dum1,dum2
real(chm_real) :: t,q,dum3

if(spline_center_flag .EQ. 1)then
   q=pos_particle-(float((ABS(dum1))))+1.0
   if(dum1 .LT. 0)then
      q=pos_particle+(float((ABS(dum1))))+1.0
   endif
endif
if(spline_center_flag .EQ. 0)then
   q=pos_particle-((float((ABS(dum1))))-0.5)+1.0
   if(dum1 .LT. 0)then
      q=pos_particle+((float((ABS(dum1))))+0.5)+1.0
   endif
endif
t=q+(0.5*(sorder-2))
dum3=t-dum2
END SUBROUTINE

!------------------------------------------------!
SUBROUTINE CALC_M2_WEIGHT(dum4,dum5)
!------------------------------------------------!
use eabfsrc
use reawri
use contrl
use consta
use coord
use parallel

implicit none

real(chm_real) :: dum4,dum5

dum5=(1.0-ABS(dum4-1.0))
END SUBROUTINE

!------------------------------------------------!
SUBROUTINE CALC_WEIGHT(dum6,dum7,dum8,dum9,dum10)
!------------------------------------------------!
use eabfsrc
use reawri
use contrl
use consta
use coord
use parallel

implicit none

integer :: dum7
real(chm_real) ::t,q,dum6,dum8,dum9,dum10

t=(dum6/(((float(dum7))-1.0)))*(dum8)
q=(((float(dum7))-dum6)/((float(dum7))-1.0))*(dum9)
dum10=q+t
END SUBROUTINE
