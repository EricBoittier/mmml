!------------------------------------------------------
subroutine WRITE_EABF_RESTART()
!------------------------------------------------------
use eabfsrc
use dimens_fcm
use exfunc
use number
use stream
use psf
use coord
use coordc
use deriv
use reawri
use consta
use contrl
use chm_kinds
use memory
use string
use parallel
implicit none
integer :: i,k

OPEN(UNIT=1339, FILE='eabf_dyn.rst')
OPEN(UNIT=1347, FILE='eabf_fl.rst')
OPEN(UNIT=1357, FILE='eabf_me_fl.rst')
do i=0,num_particles-1
   !Write Particle Dynamics Restart Information
   write(1339,'(I14,2x,I8,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14)') MDSTEP+restart_step,i,lambda_particle(i),xi_cv(i),arnc(i),ax_0(i),arf_0(i),av_half_c(i),ax_c(i),arnr(i),af_c(i),arf_c(i),av_half_f(i),ax_f(i),alpha_c(i),alpha_r(i),beta_r(i)
   call flush(1339)
   !Write Force Sample Information
   do k=0,lambda_bin_num(i)
      write(1347,'(I8,2x,I8,2x,G21.14,2x,G21.14)') i,k,bin_sum(i,k),bin_count(i,k)
      call flush(1347)
      write(1357,'(I8,2x,I8,2x,G21.14,2x,G21.14)') i,k,particle_bin_sum(i,k),particle_bin_count(i,k)
      call flush(1357)
   enddo
enddo
CLOSE (1339)
CLOSE (1347)
CLOSE (1357)
END SUBROUTINE

!------------------------------------------------------
subroutine READ_EABF_RESTART()
!------------------------------------------------------
use eabfsrc
use dimens_fcm
use exfunc
use number
use stream
use psf
use coord
use coordc
use deriv
use reawri
use consta
use contrl
use chm_kinds
use memory
use string
use parallel
implicit none
real(chm_real) :: summ,count,lambd,xi,rnc,x_0,rf_0,v_half_c,x_c,rnr,f_c,rf_c,v_half_f,x_f,alph_c,alph_r,betaa_r
integer :: i,k,istat,curr_step

write(*,"(a)") "Reading eABF Restart Info..."
restart_step=0
if(eabf_rst .EQ. 1)then
!Read F Surface Restart
 do
   READ(1347,FMT=*,IOSTAT=istat)i,k,summ,count
   IF( istat < 0) EXIT
     bin_sum(i,k)=summ
     bin_count(i,k)=count
     bin_avg(i,k)=0.0
     if(count .GT. 0)then
        bin_avg(i,k)=summ/count
     endif
enddo
CLOSE (1347,STATUS='KEEP',IOSTAT=I)

!Read F "Memory Erasure" Surface Restart
 do
   READ(1357,FMT=*,IOSTAT=istat)i,k,summ,count
   IF( istat < 0) EXIT
     particle_bin_sum(i,k)=summ
     particle_bin_count(i,k)=count
     particle_bin_avg(i,k)=0.0
     if(count .GT. 0)then
        particle_bin_avg(i,k)=summ/count
     endif
enddo
CLOSE (1357,STATUS='KEEP',IOSTAT=I)

 do
  READ(1339,FMT=*,IOSTAT=istat)curr_step,i,lambd,xi,rnc,x_0,rf_0,v_half_c,x_c,rnr,f_c,rf_c,v_half_f,x_f,alph_c,alph_r,betaa_r
   IF( istat < 0) EXIT
   restart_step=curr_step
   lambda_particle(i)=lambd
   xi_cv(i)=xi
   arnc(i)=rnc
   ax_0(i)=x_0
   arf_0(i)=rf_0
   av_half_c(i)=v_half_c
   ax_c(i)=x_c
   arnr(i)=rnr
   af_c(i)=f_c
   arf_c(i)=rf_c
   av_half_f(i)=v_half_f
   ax_f(i)=x_f
   alpha_c(i)=alph_c
   alpha_r(i)=alph_r
   beta_r(i)=betaa_r
enddo
CLOSE (1339,STATUS='KEEP',IOSTAT=I)
endif
END SUBROUTINE
