!------------------------------------------------!
SUBROUTINE WRITE_EABF() 
!------------------------------------------------!
use eabfsrc
use reawri
use contrl
use consta
use coord
use energym
use param_store

if(mod(MDSTEP,print_frq) .EQ. 0)then

   !1 CV
   if(eabf_cv_count .EQ. 1)then
      write(777,'(I14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14)') mdstep+restart_step,xi_cv(0),lambda_particle(0),particle_dudl(0),dfdl(0),EPROP(EPOT),eabf_eterm
   endif
   !2 CV
   if(eabf_cv_count .EQ. 2)then
      write(777,'(I14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14)') mdstep+restart_step,xi_cv(0),lambda_particle(0),particle_dudl(0),xi_cv(1),lambda_particle(1),particle_dudl(1),EPROP(EPOT),eabf_eterm
   endif
   !3 CV
   if(eabf_cv_count .EQ. 3)then
      write(777,'(I14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14)') mdstep+restart_step,xi_cv(0),lambda_particle(0),particle_dudl(0),xi_cv(1),lambda_particle(1),particle_dudl(1),xi_cv(2),lambda_particle(2),particle_dudl(2),EPROP(EPOT),eabf_eterm
   endif
   !4 CV
   if(eabf_cv_count .EQ. 4)then
      write(777,'(I14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14)') mdstep+restart_step,xi_cv(0),lambda_particle(0),particle_dudl(0),xi_cv(1),lambda_particle(1),particle_dudl(1),xi_cv(2),lambda_particle(2),particle_dudl(2),xi_cv(3),lambda_particle(3),particle_dudl(3),EPROP(EPOT),eabf_eterm
   endif
   call flush(777)
endif

END SUBROUTINE
