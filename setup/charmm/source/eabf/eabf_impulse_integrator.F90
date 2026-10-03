!------------------------------------------------!
 SUBROUTINE IMPULSE_INTEGRATOR()
!------------------------------------------------!

!====Impulse Integrator for Langevin Dynamics====!
!Reference "An impulse integrator for Langevin Dynamics" Skeel et al 2002 for details!

use eabfsrc
use clcg_mod, only:bmgaus
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
use energym
implicit none
integer :: e,e2,ee,ee2,eee,i,ii,k,lambd_start_bin,lambd_end_bin,spline_lambd,t,curve_flag,bin_flag,left_brace,right_brace,g_lambd_bin,total_counts,force_cv
real(chm_real) :: w_f,w_r,a,b,c,f_0,v_0,x_0,rf_0,v_half_f,function_value,flatten_value,sx_ref,sy_ref,sz_ref,tma,&
                  duodXA,duodYA,duodZA,duodXB,duodYB,duodZB,duoldxa,duoldya,duoldza,duoldxb,duoldyb,duoldzb,max_force,&
                  R,prob_deriv,fe_rms,p_ent,p_1,p_2,p_12,particle_dudl_back,new_counter

if(MDSTEP .EQ. 0 .OR. eabf_rst .EQ. 1)then
lambda_weight_sum=0.0
particle_dudl_back=0.0
R=0.0019872041  !kcal/mol*K
   do i=0,num_particles-1
      !-----------------------------------------------------------!
      !---------Initiation of Propagation for Particles-----------!
      !-----------------------------------------------------------!
      w_f=((EXP((-1.0)*lambda_frict(i)*DELTA))-(1.0)+(lambda_frict(i)*DELTA))/(lambda_frict(i)*DELTA*(1-EXP((-1.0)*lambda_frict(i)*DELTA)))
      !Reverse Step Weight
      w_r=1.0-w_f
      !Covariance Matrix Elements
      particle_a(i)=(KBOLTZ*lambda_temps(i)*((2*(w_f**2)*lambda_frict(i)*DELTA)+w_f-w_r))/lambda_mass(i)
      particle_b(i)=(KBOLTZ*lambda_temps(i)*(2*w_f*w_r*lambda_frict(i)*DELTA+w_r-w_f))/lambda_mass(i)
      particle_c(i)=(KBOLTZ*lambda_temps(i)*((2*(w_r**2)*lambda_frict(i)*DELTA)+w_f-w_r))/lambda_mass(i)
      !Initial Random Number
      arnc(i)=BMGAUS(1.0d0,ISEED)
      !Initial Force
      f_0=0.0
      !Initial Velocity
      v_0=0.0
      !Initial Position
      ax_0(i)=xi_cv(i)
      lambda_particle(i)=xi_cv(i)
      !Initial Alpha Parameter
      alpha_c(i)=SQRT(particle_a(i))
      !Initial Random Force Forward Step
      arf_0(i)=alpha_c(i)*arnc(i)
      !Velocity at Half Step
      av_half_c(i)=(EXP((-1.0)*(lambda_frict(i)*DELTA)/(2.0)))*((v_0)+((DELTA*w_f*f_0)/lambda_mass(i))+(arf_0(i)))
      !Initial Position Propagation
      ax_c(i)=ax_0(i)+((((1.0)-EXP((-1.0)*lambda_frict(i)*DELTA))/(lambda_frict(i)*EXP(((-1.0)*lambda_frict(i)*DELTA)/2)))*av_half_c(i))
      lambda_particle(i)=ax_c(i)
      enddo
endif

!Read Restart Information If Necessary
if(eabf_rst .EQ. 1)then
   call READ_EABF_RESTART()
   call WRITE_EABF()
endif
!Restart Read Complete

if(MDSTEP .GT. 0)then
eabf_eterm=0.0
   !---------------------------------------------------------!
   !-----------------Particle Propagation--------------------!
   !---------------------------------------------------------!
   do i=0,num_particles-1
      !Begin Propagation
      !Forward Step Weight
      w_f=((EXP((-1.0)*lambda_frict(i)*DELTA))-(1.0)+(lambda_frict(i)*DELTA))/(lambda_frict(i)*DELTA*((1.0)-EXP((-1.0)*lambda_frict(i)*DELTA)))
      !Reverse Step Weight
      w_r=(1.0)-w_f
      !Covariance Matrix Elements
      particle_a(i)=(KBOLTZ*lambda_temps(i)*((2*(w_f**2)*lambda_frict(i)*DELTA)+w_f-w_r))/lambda_mass(i)
      particle_b(i)=(KBOLTZ*lambda_temps(i)*(2*w_f*w_r*lambda_frict(i)*DELTA+w_r-w_f))/lambda_mass(i)
      particle_c(i)=(KBOLTZ*lambda_temps(i)*((2*(w_r**2)*lambda_frict(i)*DELTA)+w_f-w_r))/lambda_mass(i)
      !Reverse Alpha Value
      alpha_r(i)=alpha_c(i)
      !Random Number Update
      arnr(i)=arnc(i)
      !Current Random Number
      arnc(i)=BMGAUS(1.0d0,ISEED)
      !Force at Step n
      particle_dudl_back=particle_dudl(i)
      particle_dudl(i)=(lambda_k(i)*(lambda_particle(i)-xi_cv(i)))+dfdl(i)
      eabf_eterm=eabf_eterm+(0.5*lambda_k(i)*((lambda_particle(i)-xi_cv(i))**2.0))
      af_c(i)=(-1.0)*particle_dudl(i)
      !Beta Parameter Reverse
      beta_r(i)=particle_b(i)/alpha_r(i)
      !Current Alpha Parameter
      alpha_c(i)=SQRT(particle_a(i)+particle_c(i)-(beta_r(i)**2))
      !Current Random Force
      arf_c(i)=(beta_r(i)*arnr(i))+alpha_c(i)*arnc(i)
      !Current Velocity Plus Half Step
      av_half_f(i)=(EXP(((-1.0)*lambda_frict(i)*DELTA)/(2.0)))*((EXP(((-1.0)*lambda_frict(i)*DELTA)/(2.0))*(av_half_c(i)))+((DELTA*af_c(i))/lambda_mass(i))+arf_c(i))
      !Position Propagation Forward
      ax_f(i)=ax_c(i)+((((1.0)-EXP((-1.0)*lambda_frict(i)*DELTA))/(lambda_frict(i)*EXP(((-1.0)*lambda_frict(i)*DELTA)/(2.0))))*av_half_f(i))
      !Update Lambda Position
      lambda_particle(i)=ax_f(i)
      !Position Update for Propagation
      ax_c(i)=ax_f(i)
      !Velocity Update for Propagation
      av_half_c(i)=av_half_f(i)
      !Propagation complete
      !---------------------------------------------------------!
      !--------------------Sample Collection--------------------!
      !---------------------------------------------------------!
      !Bin the force along CV
      poslambda=lambda_particle(i)/lambda_bin_width(i)
      bin_num=ceiling(poslambda)
      if(bin_num .LE. lambda_bin_num(i))then
      if(bin_num .GT. 0)then
         particle_bin_sum(i,bin_num)=particle_bin_sum(i,bin_num)+particle_dudl(i)-dfdl(i)
         particle_bin_count(i,bin_num)=particle_bin_count(i,bin_num)+1.0
         particle_bin_avg(i,bin_num)=particle_bin_sum(i,bin_num)/particle_bin_count(i,bin_num)
         !All Data
         bin_sum(i,bin_num)=bin_sum(i,bin_num)+particle_dudl(i)-dfdl(i)
         bin_count(i,bin_num)=bin_count(i,bin_num)+1.0
         bin_avg(i,bin_num)=bin_sum(i,bin_num)/bin_count(i,bin_num)
      endif
      endif

   !Particle Loop enddo
   enddo

      !------------------------------------------------------!
      !----------Sample Collection Complete------------------!
      !------------------------------------------------------!
   !---------------------------------------------------------!
   !-----------------Propagation Complete--------------------!
   !---------------------------------------------------------!
endif

!Integrate Along CV for On the Fly FE estimate
if(mod(MDSTEP+restart_step,recursion_frq) .EQ. 0 .OR. eabf_rst .EQ. 1)then
   if(mod(MDSTEP,print_frq*100) .EQ. 0 .OR. eabf_rst .EQ. 1)then
      open(unit=123,file='eabf_fl.dat')
      !open(unit=223,file='eabf_me_fl.dat')
   endif
   particle_free_energy(:,:)=0.0
   free_energy(:,:)=0.0
   do i=0,num_particles-1
      particle_FES_min(i)=999.9
      fe_min=999.9
      do k=1,lambda_bin_num(i)
         if(particle_bin_count(i,k) .LT. sample_cut)then
            particle_free_energy(i,k)=particle_free_energy(i,k-1)
         endif
         if(particle_bin_count(i,k) .GE. sample_cut)then
            particle_free_energy(i,k)=particle_free_energy(i,k-1)+((particle_bin_avg(i,k))*(1.0/lambda_bin_num(i)))
         endif
         if(particle_free_energy(i,k) .LT. particle_FES_min(i))then
            particle_FES_min(i)=particle_free_energy(i,k)
         endif
         !All Data
         if(bin_count(i,k) .LT. sample_cut)then
            free_energy(i,k)=free_energy(i,k-1)
         endif
         if(bin_count(i,k) .GE. sample_cut)then
            free_energy(i,k)=free_energy(i,k-1)+(bin_avg(i,k)*(1.0/lambda_bin_num(i)))
         endif
         if(free_energy(i,k) .LT. fe_min)then
            fe_min=free_energy(i,k)
         endif
      enddo
      !Set 0th bin equal to bin 1 (since we ceiling the sample collection there's no data here)
      free_energy(i,0)=free_energy(i,1)
      bin_avg(i,0)=bin_avg(i,1)
      particle_free_energy(i,0)=particle_free_energy(i,1)
      particle_bin_avg(i,0)=particle_bin_avg(i,1)
      if(mod(MDSTEP,print_frq*100) .EQ. 0 .OR. eabf_rst .EQ. 1)then
         !Write Surface
         do k=0,lambda_bin_num(i)
            free_energy(i,k)=free_energy(i,k)-fe_min
         enddo
         do k=0,lambda_bin_num(i)
            particle_free_energy(i,k)=particle_free_energy(i,k)-particle_FES_min(i)
            !write(223,'(I8,2x,G21.14,2x,G21.14,2x,G21.14)'),i,(k+0.0)/lambda_bin_num(i),particle_free_energy(i,k),particle_bin_avg(i,k)
            !write(423,'(I8,2x,I14,2x,G21.14,2x,G21.14,2x,G21.14)'),i,MDSTEP+restart_step,(k+0.0)/lambda_bin_num(i),particle_free_energy(i,k),particle_bin_avg(i,k)
            !All Data
            poslambda=lambda_particle(i)/lambda_bin_width(i)
            bin_num=ceiling(poslambda)
            write(123,'(I8,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14)') i,(k+0.0)/lambda_bin_num(i),free_energy(i,k),bin_avg(i,k),bin_count(i,k),lambda_avg(i,k),alph_array(k)
            !write(323,'(I8,2x,I14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14,2x,G21.14)'),i,MDSTEP+restart_step,(k+0.0)/lambda_bin_num(i),free_energy(i,k),bin_avg(i,k),bin_count(i,k),lambda_particle(i),free_energy(i,bin_num),0.0
         enddo
         write(123,'(G21.14)') "  "
         write(123,'(G21.14)') "  "
         !write(223,'(G21.14)'),"  "
         !write(223,'(G21.14)'),"  "
         !write(323,'(G21.14)'),"  "
         !write(423,'(G21.14)'),"  "
         !flush(323)
         !flush(423)
      endif
   enddo
   if(mod(MDSTEP,print_frq*100) .EQ. 0 .OR. eabf_rst .EQ. 1)then
      close(123)
      !close(223)
   endif
!Set Restart flag to 0
eabf_rst=0
endif
!---------------------------------------------------------!
!---------------------------------------------------------!
!---------------------------------------------------------!

!---------------------------------------------------------!
!--------------------Force Calculation--------------------!
!---------------------------------------------------------!
if(MDSTEP .GT. recursion_frq .OR. eabf_rst .EQ. 1)then
   !Calculate Force to Apply to CV
   do i=0,num_particles-1
      particle_force(i)=0.0
      if(lambda_particle(i) .GE. 0.0)then
      if(lambda_particle(i) .LE. 1.0)then
         poslambda=lambda_particle(i)/lambda_bin_width(i)
         pos_particle=poslambda
         !Sets force without use of spline
         sorder=0
         if(sorder .EQ. 0)then
            particle_force(i)=particle_bin_avg(i,ceiling(poslambda))
            !Set Scaling of Force based off number of samples collected
            f_alpha(i)=particle_bin_count(i,ceiling(poslambda))/sample_cut
            if(f_alpha(i) .GT. 1.0)then
               f_alpha(i)=1.0
            endif
         endif
      endif
      endif
      dfdl(i)=particle_force(i)*(-1.0)*f_alpha(i)
      !Boundary
      if(xi_cv(i) .LT. 0.0)then
         dfdl(i)=(1.0)*(lambd_bnd*(xi_cv(i)-0.0))
      endif
      if(xi_cv(i) .GT. 1.0)then
         dfdl(i)=(1.0)*(lambd_bnd*(xi_cv(i)-1.0))
      endif
      !Send force to CV
      if(cv_array(i) .EQ. 1)then
         NatomA=cv_natom_lists(i,0)
         do ii=1,cv_natom_lists(i,0)
            ListA(ii)=cv_selection_lists(i,0,ii)
         enddo
         NatomB=cv_natom_lists(i,1)
         do ii=1,cv_natom_lists(i,1)
            ListB(ii)=cv_selection_lists(i,1,ii)
         enddo
         call DISTRIBUTE_FORCE(i)
      endif
      if(cv_array(i) .EQ. 2)then
         NatomA=cv_natom_lists(i,0)
         do ii=1,cv_natom_lists(i,0)
            ListA(ii)=cv_selection_lists(i,0,ii)
         enddo
         call DISTRIBUTE_FORCE_2(rmsd_ref_count,i)
         rmsd_ref_count=rmsd_ref_count+1
      endif
      if(cv_array(i) .EQ. 3)then
         NatomA=cv_natom_lists(i,0)
         do ii=1,cv_natom_lists(i,0)
            ListA(ii)=cv_selection_lists(i,0,ii)
         enddo
         NatomB=cv_natom_lists(i,1)
         do ii=1,cv_natom_lists(i,1)
            ListB(ii)=cv_selection_lists(i,1,ii)
         enddo
         NatomC=cv_natom_lists(i,2)
         do ii=1,cv_natom_lists(i,2)
            ListC(ii)=cv_selection_lists(i,2,ii)
         enddo
         NatomD=cv_natom_lists(i,3)
         do ii=1,cv_natom_lists(i,3)
            ListD(ii)=cv_selection_lists(i,3,ii)
         enddo
         call DISTRIBUTE_FORCE_3(i)
      endif
      if(cv_array(i) .EQ. 4)then
         NatomA=cv_natom_lists(i,0)
         do ii=1,cv_natom_lists(i,0)
            ListA(ii)=cv_selection_lists(i,0,ii)
         enddo
         call DISTRIBUTE_FORCE_4(i)
      endif
   enddo
!---------------------------------------------------------!
!------------------Force Calc Complete--------------------!
!---------------------------------------------------------!
endif


!-------------------------------------------------------!
!---------------CZAR Free energy Estimator--------------!
!-------------------------------------------------------!
if(mod(MDSTEP,print_frq*100) .EQ. 0)then
sorder=4
open(unit=888,file='eabf_czar_fl.dat')
do i=0,num_particles-1
   czar_bin_avg(:,:)=0.0
   czar_free_energy(:,:)=0.0
   fe_min=999.9
   do k=1,lambda_bin_num(i)
      prob_deriv=0.0
      pos_particle=k+0.0
      spline_center_flag=1
      call execute_spline_interpolation()
      lambd_start_bin=i_start_lambd
      lambd_end_bin=i_end_lambd
      !Calculate Interpolated Values
      do ii=lambd_start_bin,lambd_end_bin
         spline_lambd=ii
         if(ii .GT. lambda_bin_num(i))then
            spline_lambd=lambda_bin_num(i)
         endif
         if(ii .LT. 0)then
            spline_lambd=0
         endif
         lambd_weight(ii)=spline_weight(ii-sp_gap,sorder,0)
         if(bin_count(i,spline_lambd) .GT. 0)then
            function_value=log(bin_count(i,spline_lambd)/MDSTEP)
         endif
         if(bin_count(i,spline_lambd) .EQ. 0)then
            function_value=0.0
         endif
         prob_deriv=prob_deriv+((spline_weight(ii-sp_gap,sorder-1,0)-spline_weight(ii-sp_gap,sorder-1,1))*(function_value)*(1.0/lambda_bin_width(i)))
      enddo
      !Compute CZAR Force
      R=0.0019872041  !kcal/mol*K
      czar_bin_avg(i,k)=((-1.0*R*lambda_temps(i))*prob_deriv)+bin_avg(i,k)
      czar_free_energy(i,k)=czar_free_energy(i,k-1)+(czar_bin_avg(i,k)*(1.0/lambda_bin_num(i)))
      if(czar_free_energy(i,k) .LT. fe_min)then
         fe_min=czar_free_energy(i,k)
      endif
   enddo
   !Set 0th bin equal to bin 1 (since we ceiling the sample collection there's no data here)
   czar_free_energy(i,0)=czar_free_energy(i,1)
   czar_bin_avg(i,0)=czar_bin_avg(i,1)
   do k=0,lambda_bin_num(i)
      write(888,'(I8,2x,G21.14,2x,G21.14,2x,G21.14)') i,(k+0.0)/lambda_bin_num(i),czar_free_energy(i,k)-fe_min,czar_bin_avg(i,k)
      flush(888)
   enddo

enddo
close(888)
endif
!-------------------------------------------------------!
!-------------------------------------------------------!
!-------------------------------------------------------!

!Set Restart flag back to 0
eabf_rst=0

END SUBROUTINE
