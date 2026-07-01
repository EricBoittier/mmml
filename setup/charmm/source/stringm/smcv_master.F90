! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! CV_MASTER.MOD
!
! MODULE TO PERFORM SEVERAL VECTOR OPERATIONS ON ALL CV
! (1) set z=theta(x) [notation as in Maragliano, Fisher, Vanden-Eijnden & Ciccotti 06]
! (2) compute restraining force
! (3) compute M matrix (weights)
! (5) list currently defined CV
! (4) calculate theta(x) for Voronoi calculations
      module smcv_master
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      use cv_common
      use cv_frames, only: frames_reset_calculate, frames, frames_initialized
      use cv_quaternion, only: quat_reset_calculate, quat, quat_initialized
      use cv_types
      use sm_config
! when adding new cv types, include the corresponding modules below
      use cv_posi_com ! Cartesian positions of centers of mass
      use cv_dihe_com ! Dihedral angle between COM of 4 atom groups
      use cv_dist_com ! Distance between COM of 2 atom groups
      use cv_angle_com ! Angle between COM of 3 atom groups
      use cv_anglvec ! angle defined based on 4 points
      use cv_qcomp ! Components of orientation quaternion
      use cv_cvrms ! RMSD of selected CV from target values (specified in ref set)
      use cv_rmsd ! RMSD of selected atom coordinates from target values
      use cv_drmsd ! RMSD difference of selected atom coordinates from two target structures
      use cv_proj ! projection onto vector between two reference strutures
      use ivector
!
      implicit none
! listing of subroutines:
!
      public smcv_main ! driver subroutine
      public smcv_addforce
      public smcv_fill
      public smcv_add_hist ! computes and stores theta_i(x), and M_ij for each i and j
      public smcv_voronoi_compute
      public smcv_voronoi_smart_update ! version that ensures that MD replicas stay within the corresponding cells
      public smcv_compute_wgt ! compute CV weights for interpolation and RMSD
      public smcv_compute_M ! compute M matrix
      public smcv_list ! list CV weights
      public smcv_compute_frames_quat_para ! compute frames+quaternions in parallel
      public smcv_repl_exchange ! attempt exchange of neighboring replicas
      public smcv_gradcv_dot_dr
      public smcv_dcv_dot_gradcv
!
      contains
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_main(x,y,z,xcomp,ycomp,zcomp, &
     & mass,fx,fy,fz,iteration)
! called from dynamics
      use stream
      use multicom_aux;
      use mpi
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
      real(chm_real) :: x(:), y(:), z(:), &
     & xcomp(:), ycomp(:), zcomp(:), &
     & mass(:), &
     & fx(:), fy(:), fz(:)
      integer :: iteration ! MD iteration
! locals
      real(chm_real) :: s
      logical :: qgrp
      integer :: ierror
      character(len=len("SMCV_MAIN>") ),parameter::whoami="SMCV_MAIN>";!macro
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      qgrp=(MPI_COMM_LOCAL.ne.MPI_COMM_NULL) &
     & .and.(SIZE_LOCAL.gt.1)
!
      if (.not.cv_common_grad_initialized) call cv_common_grad_init() ! allocate cv%grad array (for derivatives)
      call frames_reset_calculate(.true.) ! make sure that frames recalculate their axes + derivatives
      call quat_reset_calculate(.true.) ! make sure that quaternions recalculate their values + derivatives
!
      if (.not.cv_common_weights_initialized) then
       call wrndie(0,whoami,trim(' CV WEIGHTS NOT INITIALIZED. WILL COMPUTE FROM M^-1(X)'))
       call smcv_compute_wgt(x,y,z,mass)
      endif
! for off-path sampling, determine tangent vector
      if (planar_on.and..not.cv_common_dz_initialized) then
       call wrndie(0,whoami,trim(' PATH TANGENT NOT INITIALIZED. WILL COMPUTE FROM CV.'))
       call cv_common_compute_dr()
      endif
! for off-path sampling simulations, need to compute dimensional k from koffp
! some parts of the code require a dimensional k, e.g. fe computation, dfdz;
! may be recoded in the future
! NOTE: this operation will destroy the current contents of the cv%k array
      if (planar_on.and..not.cv_common_k_initialized) then
       call wrndie(0,whoami,trim(' FORCE CONSTANTS FOR PLANAR SAMPLING UNDEFINED.'))
        write(info, 6765) whoami, whoami, whoami ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6765 format(/A, 'COMPUTING FORCE CONSTANTS FOR RESTRAINED ',/, &
     & A, 'DYNAMICS BY SCALING KPAR WITH CV WEIGHTS.',/, &
     & A, 'OVERWRITING PREVIOUSLY DEFINED FORCE CONSTANTS.')
       call cv_common_compute_k()
      endif
!
      restraint_force_on=.false.
!
      if (restrained_on) then ! impose restraints
       if (.not.unrestrained_on .or. & ! if unrestrained dynamics is off
     & (unrestrained_on.and. & ! or: unrestrained dyn. is on, BUT we are equilibrating w/ res.
     & ((iteration-unrestrained_eq0).lt.unrestrained_eq_steps))) then ! first equilibrate with restraints on, then release restraints
!
        restraint_force_on=.true.
!
        if (restrained_eq_steps.gt.0) then ; s=min(1.0d0*(iteration-restrained_eq0)/restrained_eq_steps,1d0) ; else ; s=1d0 ; endif !restrained equilibration
        call smcv_addforce(x,y,z,mass,fx,fy,fz,s)
!
       endif ! not untrestrained_on
      endif ! restrained_on
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! the following only gets executed if iteration > olditeration;
! this is because CHARMM executes frequent 'restarts' at the least frequency
! that is common to all output counters; a restart will require two calls to smcv_master;
! to avoid duplicating statistics + evolution etc (since the step # is the same!) I keep track
! of the iteration counter, and proceed only if the iteration counter has increased.
      if (iteration.gt.olditeration) then
       if (hist_freq.gt.0) then
!
! write(6,*) iteration, restrained_eq0, evolve_nskip
!
        if ( mod(iteration,hist_freq).eq.0 ) then
         call smcv_add_hist(x,y,z,mass, &
     & (iteration-restrained_eq0.gt.evolve_nskip)) ! add slice; the last argument is a flag
! indicating whether to update running averages
        endif
       endif
!===================== EVOLUTION ========================
       if (evolve_cv_on.and.evolve_freq.gt.0) then
        if (mod(iteration,evolve_freq).eq.0) then ! evolve string
         if (.not.string_noprint) then
          write(info,'(2A)') whoami,' EVOLVING STRING.' ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
!=================== take care of M tensor statistics
! need to add contributions to M, if running in parallel (reduce short-term average)
          if (qgrp.and.calc_Mtensor_para) then
! will broadcast all rows, but only num_cv columns
           call MPI_ALLREDUCE(cv%M(1,1,1),cv%M(1,1,2),max_cv_common*cv%num_cv, &
     & mpifloat, MPI_SUM, MPI_COMM_LOCAL, ierror)
          else ! just copy
           cv%M(1:cv%num_cv,1:cv%num_cv,2)=cv%M(1:cv%num_cv,1:cv%num_cv,1)
          endif ! qgrp
!
!========== type of of evolution
!===== EXPONENTIAL
         if (evolve_expo_on) then ! exponential averaging
           call cv_common_evolve_expo(evolve_expo_mem, evolve_nskip)
!====== AVERAGE
         elseif (evolve_aver_on) then ! just add the current theta values to the running average
           num_ave_cv_samples=num_ave_cv_samples + 1
           call cv_common_evolve_expo( 1d0-1d0/num_ave_cv_samples, evolve_nskip )
!===== SMOOTHED PATH
         elseif (evolve_smooth_on) then
           call cv_common_smooth_hist(evolve_smooth_d, evolve_nskip) ! evolve by unrestrained dynamics
!===== DEFAULT IS REGULAR SMCV
         else
          call cv_common_evolve_smcv(evolve_step) ! Euler string evolution
         endif
         if (unrestrained_on) unrestrained_eq0=iteration ! unrestrained dynamics
         restrained_eq0=iteration
! call cv_common_compute_wgt() ! compute weight from M
! update Voronoi cell centers
! write(6,*) 'ITERATION' ,iteration
         if (cv_common_voronoi_initialized) then
! recompute M inverse (which together with path centers defines the cells)
          call cv_common_compute_Minv()
          call smcv_voronoi_smart_update(x, y, z, xcomp, ycomp, zcomp, &
     & mass, iteration)
! & call smcv_voronoi_smart_update(xcomp, ycomp, zcomp,
! & dx, dy, dz, mass, iteration)
         endif ! voronoi_initialized
        endif ! evolve_freq
       endif ! evolve_cv_on
!
       if (repa_on.and.repa_freq.gt.0) then
         if (mod(iteration,repa_freq).eq.0) then
           if (.not.string_noprint) then
            write(info,'(2A)') whoami,' REPARAMETRIZING STRING.' ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
           endif
           call smcv_repa() ! reparameterize string
           unrestrained_eq0=iteration
           restrained_eq0=iteration
! update Voronoi cell centers
           if (cv_common_voronoi_initialized) & ! update voronoi info after path centers change !
! & call smcv_voronoi_smart_update(xcomp, ycomp, zcomp,
! & dx, dy, dz, mass, iteration)
     & call smcv_voronoi_smart_update(x, y, z, xcomp, ycomp, zcomp,&
     & mass, iteration)
         endif
       endif ! repa_on
!
       if (repl_x_on.and.repl_x_freq.gt.0) then
        if (mod(iteration, repl_x_freq).eq.0) then
         if (.not.string_noprint) then
          write(info,'(2A)') whoami,' ATTEMPTING EXCHANGE OF NEIGHBORING REPLICAS.'
          if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
         call smcv_repl_exchange(x, y, z, mass, iteration &
     & + rextime_offset)
        endif
       endif
!
       if (stat_on.and.stat_freq.gt.0) then
         if (mod(iteration,stat_freq).eq.0) then
           write(info,'(2A)') whoami,' CALLING STRING STATISTICS.' ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
           call smcv_stat() ! output statistics
         endif
       endif ! stat_on
      endif ! iteration > olditeration
! update internal iteration counter
      olditeration=iteration
!
      end subroutine smcv_main
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_addforce(x,y,z,mass,fx,fy,fz,t)
      use stream
      use multicom_aux;
      use consta
      use mpi
!
      real(chm_real) :: x(:), y(:), z(:), mass(:), & ! mass(size(x,1)) ,
     & fx(:), fy(:), fz(:)
      real(chm_real), optional :: t
! local
      real(chm_real) :: dummy
! slerp
      real(chm_real) :: q0(4),q1(4),q12(4),W
      integer :: k
!
      integer :: i, j, qbeg, qend, ibeg, iend, ierror
      real(chm_real) :: s, one_m_s, r, fpre, f, fpara, fperp, qnorm, qdp
      logical :: deriv, addforce, addforce2, calctheta, qgrp
      character(len=len("SMCV_ADDFORCE>") ),parameter::whoami="SMCV_ADDFORCE>";!macro
!
      qgrp=(MPI_COMM_LOCAL.ne.MPI_COMM_NULL) &
     & .and.(SIZE_LOCAL.gt.1)
!
! if running in parallel (SIZE_LOCAL>1), force recalculation + communication here
      if (qgrp) call smcv_compute_frames_quat_para(x,y,z,mass)
!
      if (present(t)) then
       if (t.ge.0d0.and.t.le.1d0) then ; s=t ; else ; s=1d0 ; endif
      else
       s=1d0
      endif
      one_m_s=1d0-s
!
      calctheta=.true. ; deriv=.true. ; addforce2=.true.
!
      qgrp=qgrp.and.calc_cv_para
!
      if (qgrp) then
       ibeg=cv_send_displ(ME_LOCAL+1)+1
       iend=cv_send_displ(ME_LOCAL+1)+cv_send_count(ME_LOCAL+1)
      else
       ibeg=1
       iend=cv%num_cv
      endif
!
      fpre=0d0 ! initialize force prefactor
      if (planar_on.or.qgrp) then
       addforce2=.false. ! do not add force on first pass with planar sampling
                        ! do _not_ addforce after first pass if
                        ! cvs are computed in parallel, because I first need to communicate their values + gradients !
      endif
! update current z-restraints, compute non-eq work, and add new forces
! do i=1, cv%num_cv ! loop over all CV
      do i=ibeg, iend ! loop over CV assigned to this slave in the group
       r=(s*cv%r(i,main)+one_m_s*cv%r(i,comp))
!
       select case(cv%type(i))
! EXPERIMENTAL:
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! quaternion slerp (sloppy)
       case(qcomp_1)
        qbeg=i; qend=qbeg+3; qnorm=0d0; qdp=0d0; k=0
        do j=qbeg, qend;
         k=k+1
         q1(k)=cv%r(j,main); q0(k)=cv%r(j,comp);
        enddo
        q12(1)=q0(1)*q1(1)+q0(2)*q1(2)+q0(3)*q1(3)+q0(4)*q1(4) ! cos W
        if (q12(1).lt.0) q1=-q1 ! take shortest interpolation path
        q12(2)=q0(1)*q1(2)-q1(1)*q0(2) - (q0(3)*q1(4)-q0(4)*q1(3))
        q12(3)=q0(1)*q1(3)-q1(1)*q0(3) - (q0(4)*q1(2)-q0(2)*q1(4))
        q12(4)=q0(1)*q1(4)-q1(1)*q0(4) - (q0(2)*q1(3)-q0(3)*q1(2))
! interpolate
        W=acos(q12(1))
        q12(1)=cos(s*W)
        W=sin(s*W)/sin(W)
        q12(2:4)=W*q12(2:4)
! compute new quaternion
        q1(1)=q0(1)*q12(1)-q0(2)*q12(2)-q0(3)*q12(3)-q0(4)*q12(4)
        q1(2)=q0(1)*q12(2)+q12(1)*q0(2) + (q0(3)*q12(4)-q0(4)*q12(3))
        q1(3)=q0(1)*q12(3)+q12(1)*q0(3) + (q0(4)*q12(2)-q0(2)*q12(4))
        q1(4)=q0(1)*q12(4)+q12(1)*q0(4) + (q0(2)*q12(3)-q0(3)*q12(2))
!
        qnorm=1./sqrt(q1(1)**2+q1(2)**2+q1(3)**2+q1(4)**2)
        q1=q1*qnorm
!
        r=q1(1)
! write(601,*) r, cv%r(i,instant)
        dummy=(r-cv%r(i,zcur))
!
       case(qcomp_2)
        qbeg=i-1; qend=qbeg+3; qnorm=0d0; qdp=0d0; k=0
        do j=qbeg, qend;
         k=k+1
         q1(k)=cv%r(j,main); q0(k)=cv%r(j,comp);
        enddo
        q12(1)=q0(1)*q1(1)+q0(2)*q1(2)+q0(3)*q1(3)+q0(4)*q1(4) ! cos W
        if (q12(1).lt.0) q1=-q1 ! take shortest interpolation path
        q12(2)=q0(1)*q1(2)-q1(1)*q0(2) - (q0(3)*q1(4)-q0(4)*q1(3))
        q12(3)=q0(1)*q1(3)-q1(1)*q0(3) - (q0(4)*q1(2)-q0(2)*q1(4))
        q12(4)=q0(1)*q1(4)-q1(1)*q0(4) - (q0(2)*q1(3)-q0(3)*q1(2))
! interpolate
        W=acos(q12(1))
        q12(1)=cos(s*W)
        W=sin(s*W)/sin(W)
        q12(2:4)=W*q12(2:4)
! compute new quaternion
        q1(1)=q0(1)*q12(1)-q0(2)*q12(2)-q0(3)*q12(3)-q0(4)*q12(4)
        q1(2)=q0(1)*q12(2)+q12(1)*q0(2) + (q0(3)*q12(4)-q0(4)*q12(3))
        q1(3)=q0(1)*q12(3)+q12(1)*q0(3) + (q0(4)*q12(2)-q0(2)*q12(4))
        q1(4)=q0(1)*q12(4)+q12(1)*q0(4) + (q0(2)*q12(3)-q0(3)*q12(2))
!
        qnorm=1./sqrt(q1(1)**2+q1(2)**2+q1(3)**2+q1(4)**2)
        q1=q1*qnorm
!
        r=q1(2)
! write(601,*) r, cv%r(i,instant)
        dummy=(r-cv%r(i,zcur))
!
       case(qcomp_3)
        qbeg=i-2; qend=qbeg+3; qnorm=0d0; qdp=0d0; k=0
        do j=qbeg, qend;
         k=k+1
         q1(k)=cv%r(j,main); q0(k)=cv%r(j,comp);
        enddo
        q12(1)=q0(1)*q1(1)+q0(2)*q1(2)+q0(3)*q1(3)+q0(4)*q1(4) ! cos W
        if (q12(1).lt.0) q1=-q1 ! take shortest interpolation path
        q12(2)=q0(1)*q1(2)-q1(1)*q0(2) - (q0(3)*q1(4)-q0(4)*q1(3))
        q12(3)=q0(1)*q1(3)-q1(1)*q0(3) - (q0(4)*q1(2)-q0(2)*q1(4))
        q12(4)=q0(1)*q1(4)-q1(1)*q0(4) - (q0(2)*q1(3)-q0(3)*q1(2))
! interpolate
        W=acos(q12(1))
        q12(1)=cos(s*W)
        W=sin(s*W)/sin(W)
        q12(2:4)=W*q12(2:4)
! compute new quaternion
        q1(1)=q0(1)*q12(1)-q0(2)*q12(2)-q0(3)*q12(3)-q0(4)*q12(4)
        q1(2)=q0(1)*q12(2)+q12(1)*q0(2) + (q0(3)*q12(4)-q0(4)*q12(3))
        q1(3)=q0(1)*q12(3)+q12(1)*q0(3) + (q0(4)*q12(2)-q0(2)*q12(4))
        q1(4)=q0(1)*q12(4)+q12(1)*q0(4) + (q0(2)*q12(3)-q0(3)*q12(2))
!
        qnorm=1./sqrt(q1(1)**2+q1(2)**2+q1(3)**2+q1(4)**2)
        q1=q1*qnorm
!
        r=q1(3)
! write(601,*) r, cv%r(i,instant)
        dummy=(r-cv%r(i,zcur))
!
       case(qcomp_4)
        qbeg=i-3; qend=qbeg+3; qnorm=0d0; qdp=0d0; k=0
        do j=qbeg, qend;
         k=k+1
         q1(k)=cv%r(j,main); q0(k)=cv%r(j,comp);
        enddo
        q12(1)=q0(1)*q1(1)+q0(2)*q1(2)+q0(3)*q1(3)+q0(4)*q1(4) ! cos W
        if (q12(1).lt.0) q1=-q1 ! take shortest interpolation path
        q12(2)=q0(1)*q1(2)-q1(1)*q0(2) - (q0(3)*q1(4)-q0(4)*q1(3))
        q12(3)=q0(1)*q1(3)-q1(1)*q0(3) - (q0(4)*q1(2)-q0(2)*q1(4))
        q12(4)=q0(1)*q1(4)-q1(1)*q0(4) - (q0(2)*q1(3)-q0(3)*q1(2))
! interpolate
        W=acos(q12(1))
        q12(1)=cos(s*W)
        W=sin(s*W)/sin(W)
        q12(2:4)=W*q12(2:4)
! compute new quaternion
        q1(1)=q0(1)*q12(1)-q0(2)*q12(2)-q0(3)*q12(3)-q0(4)*q12(4)
        q1(2)=q0(1)*q12(2)+q12(1)*q0(2) + (q0(3)*q12(4)-q0(4)*q12(3))
        q1(3)=q0(1)*q12(3)+q12(1)*q0(3) + (q0(4)*q12(2)-q0(2)*q12(4))
        q1(4)=q0(1)*q12(4)+q12(1)*q0(4) + (q0(2)*q12(3)-q0(3)*q12(2))
!
        qnorm=1./sqrt(q1(1)**2+q1(2)**2+q1(3)**2+q1(4)**2)
        q1=q1*qnorm
!
        r=q1(4)
! write(601,*) r, cv%r(i,instant)
! write(601,*) '%'
        dummy=(r-cv%r(i,zcur))
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccc EXPERIMENTAL ABOVE: quaternion slerp
!
       case(dihe_com)
         dummy=(r-cv%r(i,zcur))
         dummy=modulo(dummy,TWOPI)
         if (dummy.gt.PI) dummy=dummy-TWOPI ! pick the smallest angular displacement
       case(angle_com) ! cannot tell between theta/-theta
         dummy=(abs(r)-abs(cv%r(i,zcur)))
       case default
         dummy=(r-cv%r(i,zcur))
       end select
!
! compute nonequilibrium work -- u
       cv%work=cv%work+cv%r(i,forces2)*dummy ! update work;
       cv%r(i,zcur)=r ! update current restrains
!ccccccccccccccccccccccccc ADD FORCES cccccccccccccccccccc
       addforce=addforce2.and.cv%active(i)
!
       select case(cv%type(i))
          case(posi_com_x, posi_com_y, posi_com_z);
           call cv_posi_com_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce)
          case(qcomp_1, qcomp_2, qcomp_3, qcomp_4);
           call cv_qcomp_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce)
          case(angle_com);
           call cv_angle_com_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce)
          case(anglvec);
           call cv_anglvec_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce)
          case(dihe_com);
           call cv_dihe_com_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce)
          case(dist_com);
           call cv_dist_com_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce)
          case(rmsd);
           call cv_rmsd_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce)
          case(drmsd);
           call cv_drmsd_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce)
          case(proj);
           call cv_proj_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce)
          case(cvrms); ! may not be correct if running in parallel in case slave cvs on a different node -- so skip here & force compute below
          if (.not. qgrp) call cv_cvrms_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce)
          case default
           call wrndie(0,whoami,trim('UNKNOWN CV SPECIFIED.'))
       end select
! update force prefactor (planar sampling)
       fpre=fpre+cv%r(i,forces2)*cv%r(i,dz)/cv%weight(i) ! for planar force; incorrect for inactive cv or cvrms; also obsolete
      enddo ! over all cv
!
! communicate cv values + gradients using custom types
      if (qgrp) then ! send zcur, instant, forces2
       call mpi_allgatherv(cv%r(ibeg,1),cv_send_count(ME_LOCAL+1), &
     & MPI_CV_TYPE3, &
     & cv%r, cv_send_count, cv_send_displ, &
     & MPI_CV_TYPE3, MPI_COMM_LOCAL, ierror)
! send all gradients in one message
       call mpi_allgatherv(cv%grad(ibeg,1,1,1), &
     & cv_send_count(ME_LOCAL+1), MPI_GRAD_TYPE, &
     & cv%grad, cv_send_count, cv_send_displ, &
     & MPI_GRAD_TYPE, MPI_COMM_LOCAL, ierror)
      endif
!
! write(0,*) 'ME_LOCAL: ',ME_LOCAL, 'SIZE_LOCAL:', SIZE_LOCAL
! call MPI_BARRIER(MPI_COMM_GLOBAL, ierror)
! write(600+ME_GLOBAL, *) cv%r(1:cv%num_cv,forces2)
! close(600+ME_GLOBAL)
! write(600+ME_GLOBAL, *) cv%grad(10,:,1,1)
! write(700+ME_GLOBAL, *) trash(10,:,1,1)
! close(700+ME_GLOBAL)
! stop
! second loop for planar sampling or parallelization within group
      if (planar_on.or.qgrp) then
       calctheta=.false. ! already calculated above
       deriv=.false. ! already calculated above
! deriv=.true.
       addforce2=.true.
!
       do i=1, cv%num_cv ! loop over all CV
!ccccccccccccccccccccccccc ADD FORCES cccccccccccccccccccccccccccccccccccc
! f=fpre*cv%r(i,dz)*cv%weight(i) ! `external` force passed in; NOTE the scaling by weight
        if (planar_on) then ! obsolete
         fpara=fpre*cv%r(i,dz)*cv%weight(i) ! force acting parallel to dz
         fperp=(cv%r(i,forces2)-fpara)/cv%kpara*cv%kperp ! force perpendicular to dz
         f=fpara+fperp
        else ! qgrp -- just use the force computed above
         f=cv%r(i,forces2)
        endif
!
        addforce=addforce2.and.cv%active(i)
        select case(cv%type(i))
          case(posi_com_x, posi_com_y, posi_com_z);
           call cv_posi_com_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce,f)
          case(qcomp_1, qcomp_2, qcomp_3, qcomp_4);
           call cv_qcomp_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce,f)
          case(angle_com);
           call cv_angle_com_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce,f)
          case(anglvec);
           call cv_anglvec_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce,f)
          case(dihe_com);
           call cv_dihe_com_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce,f)
          case(rmsd);
           call cv_rmsd_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce,f)
          case(drmsd);
           call cv_drmsd_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce,f)
          case(proj);
           call cv_proj_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce,f)
          case(dist_com);
           call cv_dist_com_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce,f)
          case(cvrms); ! force computation of restraint value & derivative here (see above for comments)
           call cv_cvrms_calc(i,x,y,z,mass,fx,fy,fz, &
     & qgrp, qgrp, addforce,f)
          case default
           call wrndie(0,' SMCV_ADDFORCE>',trim('UNKNOWN CV SPECIFIED.'))
        end select
       enddo ! over all cv
      endif ! planar_on
!
      end subroutine smcv_addforce
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_add_hist(x,y,z,mass,update_average)
! this is a driver to routines
! that calculates the M matrix and cv time series
      use stream
      use multicom_aux;
      use consta
      use number
      use mpi
!
      real(chm_real) :: x(:), y(:), z(:), mass(:) ! mass(size(x,1))
      logical :: update_average ! whether the current dataset should be included with the cv running average
! (it is always added to history)
      integer :: i, j, k, imap, ibeg, iend, kbeg, kend, ierror
      real(chm_real) :: u, v, t, t1, dot_pr
      logical :: deriv, calctheta, qgrp, qgrp1
      integer :: cvlist(cv%num_cv) ! for fast M tensor computation
      integer :: cvlistsize, ii, jj
      character(len=len("SMCV_ADD_HIST>") ),parameter::whoami="SMCV_ADD_HIST>";!macro
      real(chm_real), pointer :: M(:,:)
!
      qgrp=(MPI_COMM_LOCAL.ne.MPI_COMM_NULL) &
     & .and.(SIZE_LOCAL.gt.1)
!
! if running in parallel (SIZE_LOCAL>1), force recalculation + communication here
      if (qgrp) call smcv_compute_frames_quat_para(x,y,z,mass)
!
      if (.not.cv_common_grad_initialized) call cv_common_grad_init() ! allocate cv%grad array (for derivatives)
! loop over all cv and make calls by type
      if (cv%num_cv.eq.0) then
       call wrndie(0,whoami,trim('NO CV DEFINED.'))
      endif
!
      calctheta=.true. ! compute cv values
      deriv=.true. ! compute derivatives
      j=cv_common_add_hist() ! determine index into the history array
! also update running average
      t=0d0 ! default behavior does not modify runinng average (see below)
!
! write(6,*) 'hist index', j
! write(600+whoiam,*) update_average, j
!
      if (update_average) then
       cv%num_run_ave=cv%num_run_ave+1
       t=1.0d0/(cv%num_run_ave);
      endif
      t1=1d0-t
      if (restraint_force_on) then ! if this flag is set, then cv and derivatives are already known
       do i=1, cv%num_cv
        v=cv%r(i,instant)
        cv%r(i,j)=v
! VO 3.24.09: correct for angle average
        u=cv%r(i,runave)
        v=u-v
!
        select case(cv%type(i));
         case(dihe_com, angle_com, anglvec);
          v=modulo(v,TWOPI)
          if (v.gt.PI) v=v-TWOPI
        end select
!
        cv%r(i,runave)=u-t*v
       enddo
      else ! need to compute
!
       qgrp1=qgrp.and.calc_cv_para
       if (qgrp1) then
        ibeg=cv_send_displ(ME_LOCAL+1)+1
        iend=cv_send_displ(ME_LOCAL+1)+cv_send_count(ME_LOCAL+1)
       else
        ibeg=1
        iend=cv%num_cv
       endif
!
       do i=ibeg, iend
        select case(cv%type(i))
         case(posi_com_x, posi_com_y, posi_com_z);
          call cv_posi_com_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.) ! force arguments are dummies
         case(qcomp_1, qcomp_2, qcomp_3, qcomp_4);
          call cv_qcomp_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(angle_com);
          call cv_angle_com_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(anglvec);
          call cv_anglvec_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(dihe_com);
          call cv_dihe_com_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(rmsd);
          call cv_rmsd_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(drmsd);
          call cv_drmsd_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(proj);
          call cv_proj_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(dist_com);
          call cv_dist_com_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(cvrms);
          if (.not.qgrp1) then
          call cv_cvrms_calc(i,x,y,z,mass,x,y,z,calctheta,deriv,.false.)
          else
           cycle ! continue without updating average since it would be incorrect
          endif
         case default
         call wrndie(0,whoami,trim('UNKNOWN CV SPECIFIED.'))
        end select
! update running average
! add history
        v=cv%r(i,instant)
        cv%r(i,j)=v
! VO 3.24.09: correct for angle average
        u=cv%r(i,runave)
        v=u-v
!
        select case(cv%type(i));
         case(dihe_com, angle_com, anglvec);
          v=modulo(v,TWOPI)
          if (v.gt.PI) v=v-TWOPI
        end select
!
        cv%r(i,runave)=u-t*v
       enddo ! loop over cv
!
       if (qgrp1) then
! communicate cv values + gradients using custom types
        call mpi_allgatherv(cv%r(ibeg,1),cv_send_count(ME_LOCAL+1), &
     & MPI_CV_TYPE3I, &
     & cv%r, cv_send_count, cv_send_displ, &
     & MPI_CV_TYPE3I, MPI_COMM_LOCAL, ierror)
        call mpi_allgatherv(cv%grad(ibeg,1,1,1), &
     & cv_send_count(ME_LOCAL+1),MPI_GRAD_TYPE, &
     & cv%grad, cv_send_count, cv_send_displ, &
     & MPI_GRAD_TYPE, MPI_COMM_LOCAL, ierror)
!
! cvrms fix below:
!*************************************************************************
        do i=1, cv%num_cv
         if (cv%type(i).eq.cvrms) then
          call cv_cvrms_calc(i,x,y,z,mass,x,y,z,calctheta,deriv,.false.)
! update running average
          v=cv%r(i,instant)
          cv%r(i,j)=v
          u=cv%r(i,runave)
          v=u-v
          cv%r(i,runave)=u-t*v
         endif
        enddo
!*************************************************************************
       endif ! qgrp1
!
! also need to add history for non-local indices (could also gather history on root before output, but is this not faster?)
       cv%r(1:cv%num_cv, j)=cv%r(1:cv%num_cv, instant)
!
      endif ! restraint_force_on
!
! write(600+ME_GLOBAL,*) cv%r(1:cv%num_cv,1:2)
! close(600+ME_GLOBAL)
! write(700+ME_GLOBAL,*) cv%gradx(59,:,2)
! write(700+ME_GLOBAL,*) cv%gradx(60,:,2)
! write(700+ME_GLOBAL,*) cv%grad
! close(700+ME_GLOBAL)
! stop
!
! now update M = < grad(theta_i) . grad(theta_j) >
      imap=cv%amap%last ! last index into the atom map
!
! it is not clear how to parallelize the loop below best,
! and whether it will make a significant difference,
! since the number of CV should be relatively small in most cases;
! one possibility is to split up [1..imap], (as currently done)
! this implies that M is _always_ partial, i.e. global M is computed for
! use in evolution, etc;
!
      if (qgrp.and.calc_Mtensor_para) then
       kbeg=imap_displ(ME_LOCAL+1)+1
       kend=kbeg+imap_count(ME_LOCAL+1)-1
      else
       kbeg=1
       kend=imap
      endif
!***** compute M tensor **************
! original (slow) code : note that the execution time is independent of the sparsity of M
! this code is simple and adequate if the frequency of calling this routine is small, as in the SMCV
      if (.not. calc_Mtensor_fast) then ! original (brute force) method
       do j=1,cv%num_cv
        do i=1,j
         dot_pr=0d0
! do k=1, imap
         do k=kbeg, kend
          dot_pr = dot_pr + &
     & cv%gradx(j,k,2)*cv%gradx(i,k,2) + &
     & cv%grady(j,k,2)*cv%grady(i,k,2) + &
     & cv%gradz(j,k,2)*cv%gradz(i,k,2)
         enddo
         cv%M(i,j,1)=t1*cv%M(i,j,1) + t*dot_pr
         cv%M(j,i,1)=cv%M(i,j,1) ! note: grad(:,:,2) is scaled by mass,
                                ! so M, too, includes this scaling
                                ! this means that M is dimensional; z evolution is in unscaled variables
        enddo
       enddo
      else ! faster method for sparse matrices
!
! new code that relies on the inverse map that for every atom gives the CVs that depend on it
! the speedup compared to the simple code depends on the number of CV that make use of a particular atom;
! the code is ideal if there are many cv, but each depends on a couple of atoms only;
! this way, cvlistsize will be small, and the two inner loops will be fast
       allocate(M(cv%num_cv,cv%num_cv)); M=zero
! do k=1, imap
       do k=kbeg, kend
        cvlistsize=cv%amap%v(k)%last
        cvlist(1:cvlistsize)=cv%amap%v(k)%i(1:cvlistsize) ! note that the cvs in the cvlist should be in increasing order because the map is updated as CV are added
        do j=1,cvlistsize
         jj=cvlist(j)
         do i=1, j
          ii=cvlist(i)
          M(ii,jj)=M(ii,jj)+ &
     & cv%gradx(jj,k,2)*cv%gradx(ii,k,2) + &
     & cv%grady(jj,k,2)*cv%grady(ii,k,2) + &
     & cv%gradz(jj,k,2)*cv%gradz(ii,k,2)
         enddo ! i
        enddo ! j
       enddo ! k
!
       do j=1,cv%num_cv
        do i=1,j
         cv%M(i,j,1)= t1 * cv%M(i,j,1) + t * M(i,j)
         cv%M(j,i,1)= cv%M(i,j,1) ! note: grad(:,:,2) is scaled by mass,
        enddo
       enddo
       deallocate(M)
      endif
!
! write(600+ME_GLOBAL,'(2F15.10)') cv%M(1:cv%num_cv,1:cv%num_cv,1)
! write(600+ME_GLOBAL,'(I3, 151F15.10)')
! & (cv%amap%i(i), (cv%gradx(j,i,1), i=1,imap), j=1,cv%num_cv)
! write(600+whoiam,'(2F15.10)') cv%grady(1:cv%num_cv,:,2)
! write(600+whoiam,'(2F15.10)') cv%gradz(1:cv%num_cv,:,2)
!
      end subroutine smcv_add_hist
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_repl_exchange(x,y,z,mass,itime)
! attempt to swap CV that correspond to two adjacent replicas
      use sm_var, only: nstring, mestring
      use multicom, only: multicom_permute_string_ranks
!
      use stream
      use multicom_aux;
      use consta
! __DEP_FILES ! in the case of CHARMM, should use a 'clean' way to open files (future?)
      use string
      use mpi
      use clcg_mod, only: random; use reawri, only: iseed
      use reawri
      use param_store, only: set_param
      use parallel, only: psnd4, psnd8

      implicit none
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
      real(chm_real) :: x(:), y(:), z(:), mass(:) ! mass(size(x,1))
      integer :: itime
!
      integer :: i, j, ibeg, iend, ierror, stat(MPI_STATUS_SIZE)
      logical :: calctheta, deriv, qgrp
!
      integer :: which ! replica with which the exchange was attempted
      logical :: success ! whether the exchange attempt was successful
      integer :: nodelist(nstring) ! holds new string replica order after excahnge attempt
!
      real(chm_real) :: r_send(1:cv%num_cv, 1:5) ! cv values to send to partner
      real(chm_real) :: r_recv(1:cv%num_cv, 1:5) ! cv values to receive from partner
      real(chm_real) :: dE_me, dE
!
      character(len=150) :: fnames(5) ! for storing output file names
      character(len=150) :: new_fnames(5) ! for storing swapped file names
      logical :: openun(5) ! , qform, qwrite
      integer :: oldiol
      integer :: nfiles
!
! real(chm_real) :: dEG(nstring) !aa
!
      character(len=len("SMCV_REPL_EXCHANGE>") ),parameter::whoami="SMCV_REPL_EXCHANGE>";!macro
!
      if (.not.cv_common_rex_initialized) call cv_common_rex_init()
!
      qgrp=(MPI_COMM_LOCAL.ne.MPI_COMM_NULL) &
     & .and.(SIZE_LOCAL.gt.1)
! compute instantaneous CV values, if needed
!
! if running in parallel (SIZE_LOCAL>1), force recalculation + communication here
      if (qgrp) call smcv_compute_frames_quat_para(x,y,z,mass)
!
! loop over all cv and make calls by type
      if (cv%num_cv.eq.0) then
       call wrndie(0,whoami,trim('NO CV DEFINED.'))
      endif
!
      calctheta=.true. ! compute cv values
      deriv=.false. ! compute derivatives
      dE=0d0
      success=.false.
!
      if (.not.restraint_force_on) then ! if this flag is set, then cv and derivatives are already known
! need to compute
       if (qgrp.and.calc_cv_para) then
        ibeg=cv_send_displ(ME_LOCAL+1)+1
        iend=cv_send_displ(ME_LOCAL+1)+cv_send_count(ME_LOCAL+1)
       else
        ibeg=1
        iend=cv%num_cv
       endif
!
       do i=ibeg, iend
        select case(cv%type(i))
         case(posi_com_x, posi_com_y, posi_com_z);
          call cv_posi_com_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.) ! force arguments are dummies
         case(qcomp_1, qcomp_2, qcomp_3, qcomp_4);
          call cv_qcomp_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(angle_com);
          call cv_angle_com_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(anglvec);
          call cv_anglvec_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(dihe_com);
          call cv_dihe_com_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(rmsd);
          call cv_rmsd_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(drmsd);
          call cv_drmsd_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(proj);
          call cv_proj_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(dist_com);
          call cv_dist_com_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case(cvrms);
          call cv_cvrms_calc(i,x,y,z,mass,x,y,z,calctheta, &
     & deriv,.false.)
         case default
         call wrndie(0,whoami,trim('UNKNOWN CV SPECIFIED.'))
        end select
       enddo ! loop over cv
!
       if (qgrp.and.calc_cv_para) then
! communicate cv values + gradients using custom types
        call mpi_allgatherv(cv%r(ibeg,1),cv_send_count(ME_LOCAL+1), &
     & MPI_CV_TYPE3I, &
     & cv%r, cv_send_count, cv_send_displ, &
     & MPI_CV_TYPE3I, MPI_COMM_LOCAL, ierror)
        call mpi_allgatherv(cv%grad(ibeg,1,1,1), &
     & cv_send_count(ME_LOCAL+1),MPI_GRAD_TYPE, &
     & cv%grad, cv_send_count, cv_send_displ, &
     & MPI_GRAD_TYPE, MPI_COMM_LOCAL, ierror)
       endif
!
      endif ! .not. restraint_force_on
!
! determine exchange partner
      if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
       if (ME_STRNG.eq.0) which=INT(random(iseed)*2d0) ! either 0 or 1
       call MPI_BCAST(which, 1, mpiint, 0, MPI_COMM_STRNG, ierror) ! string root broadcasts to all replicas
! determine whether swapping w. left (-1) or w. right (+1) neighbor & calculate rank of neighbor
       which=ME_STRNG + (mod(ME_STRNG + which, itwo)*2 - 1)
! if which == 0, then: -1, 2, 1, 4, 3, ...
! if which == 1, then: 1, 0, 3, 2, ...
! communicate CV:
       if ((which.ge.0).and.(which.lt.SIZE_STRNG)) then ! endpoint(s) do nothing !
! 1) : save
        r_send(:,1)=cv%r(1:cv%num_cv,main)
        r_send(:,2)=cv%r(1:cv%num_cv,comp)
        r_send(:,3)=cv%r(1:cv%num_cv,ref)
        r_send(:,4)=cv%r(1:cv%num_cv,ref2)
        r_send(:,5)=cv%r(1:cv%num_cv,zcur)
! 2) send/receive
        call MPI_SENDRECV(r_send, 5*cv%num_cv, mpifloat, &
     & which, which, r_recv, 5*cv%num_cv, mpifloat, &
     & which, ME_STRNG, MPI_COMM_STRNG, stat, ierror)
! 3) calculate string energies (code adopted from cv_common_evolve_smcv)
        call cv_common_rex_compute_dE(r_recv(:,1), dE_me)
! 4) combine energies from two replicas:
        call MPI_SENDRECV(dE_me, 1, mpifloat, &
     & which, which, dE, 1, mpifloat, &
     & which, ME_STRNG, MPI_COMM_STRNG, stat, ierror)
        dE=dE+dE_me
! 5) apply Metropolis criterion
        if (dE.le.0d0) then
         success=.true.
        else
! the higher-rank replica draws random number
! this may not be correct because the random numbers will not come from the same sequence;
! may change this later
         if (which.lt.ME_STRNG) then
          success=(random(iseed).le.exp(-cv%rex_beta*dE))
! send to othe replica
          call MPI_SEND(success, 1, mpibool, which, which, &
     & MPI_COMM_STRNG, ierror)
         else
          call MPI_RECV(success, 1, mpibool, which, ME_STRNG, &
     & MPI_COMM_STRNG, stat, ierror)
         endif ! which lt ME_STRNG
        endif ! apply Metropolis
!
       endif ! which .ge.0 ...
! all root nodes continue (success=false for idle node(s)) :
!
       if (success) then
        call MPI_ALLGATHER(which, 1, mpiint, &
     & nodelist, 1, mpiint, MPI_COMM_STRNG, ierror)
!
! make entry in REX log (only lower-rank replica does this)
        if (ME_STRNG.lt.which) then
          j=ME_STRNG ! possible integer typecast
          i=int_vector_add(cv%rex_log, j) ! this replica
          i=int_vector_add(cv%rex_log, which) ! exchanged with this replica
          i=int_vector_add(cv%rex_log, itime) ! at this time
        endif ! success
!
!********************************************************************************************
! swap restart & traj file info; otherwise restart files will correspond to wrong replica
!#ifdef 1
! oldiol=iolev
! iolev=1 ! so that vinqre works
!#endif
        nfiles=2
! can add others here
        if (iunwri.gt.0) &
! CHARMM VINQUIRE gives problems, did not bother to debug, since that code is obsolete anyway
     & INQUIRE(UNIT=iunwri, OPENED=openun(1), NAME=fnames(1))
! & CALL VINQRE('UNIT',fnames(1),i,j,
! & OPENUN(1),QFORM,QWRITE,iunwri)
        if (iuncrd.gt.0) &
     & INQUIRE(UNIT=iuncrd, OPENED=openun(2), NAME=fnames(2))
! & CALL VINQRE('UNIT',fnames(2),i,j,
! & OPENUN(2),QFORM,QWRITE,iuncrd)
! aa
! write(600+ME_STRNG,*) iunwri, fnames(1), iuncrd, fnames(2)
!
!
        i=nfiles*len(fnames(1)) ! length of broadcast buffer
        if ( iunwri .gt. 0 .or. iuncrd .gt. 0 )&
         call MPI_SENDRECV(fnames, i, MPI_BYTE, &
     & which, which, new_fnames, i, MPI_BYTE, &
     & which, ME_STRNG, MPI_COMM_STRNG, stat, ierror)
! write(600+ME_STRNG,*) iunwri, new_fnames(1),
! & iuncrd, new_fnames(2)
! close(600+ME_STRNG)
! assuming that the restart file is formatted (might change this later)

        if (iunwri.gt.0.and.openun(1)) then
         close(iunwri)
         i=len(new_fnames(1))
         call trima(new_fnames(1), i)
         open(UNIT=iunwri, FILE=new_fnames(1)(1:i), FORM='FORMATTED', &
     & STATUS='OLD', ACCESS='SEQUENTIAL')
        endif
! assuming that dcd file is unformatted
        if (iuncrd.gt.0.and.openun(2)) then
         close(iuncrd)
         i=len(new_fnames(2))
         call trima(new_fnames(2), i)
         open(UNIT=iuncrd, FILE=new_fnames(2)(1:i), FORM='UNFORMATTED', &
! & STATUS='OLD', ACCESS='APPEND')
     & STATUS='OLD', POSITION='APPEND')
        endif





!
!#ifdef 1
! iolev=oldiol
!#endif
! done with swap output file info
!********************************************************************************************
       else ! success
        call MPI_ALLGATHER(ME_STRNG, 1, mpiint, &
     & nodelist, 1, mpiint, MPI_COMM_STRNG, ierror)
       endif ! success
! aa
! dE=exp(-cv%rex_beta*dE)
! call MPI_ALLGATHER(dE, 1, mpifloat,
! & dEG, 1, mpifloat, MPI_COMM_STRNG, ierror)
!
      endif ! MPI_COMM_STRNG
! from here on all nodes continue:
! broadcast success to all slave nodes
      if (qgrp) then
       call PSND4(success,1)

#if (KEY_INTEGER8==0)
       call PSND4(nodelist,nstring) 
#endif
#if (KEY_INTEGER8==1)
       call PSND8(nodelist,nstring) 
#endif



! broadcast new cv to slaves
       if (success) then

#if (KEY_SINGLE==1)
        call PSND4r(r_recv,cv%num_cv*5) 
#endif
#if (KEY_SINGLE==0)
        call PSND8m(r_recv,cv%num_cv*5) 
#endif



       endif
      endif ! qgrp
!
! 6) if move was successful, perform actual exchange of CV
      if (success) then
        cv%r(1:cv%num_cv,main)=r_recv(:,1)
        cv%r(1:cv%num_cv,comp)=r_recv(:,2) ! might comment this line so that restraint change gradually (should not be necessary in principle!)
        cv%r(1:cv%num_cv,ref) =r_recv(:,3)
        cv%r(1:cv%num_cv,ref2)=r_recv(:,4)
        cv%r(1:cv%num_cv,zcur)=r_recv(:,5)
      endif
!
! write(600+ME_GLOBAL, *) ME_STRNG
! if replica order has changed, switch communicator
! if (ME_GLOBAL.eq.0) write(600,*) nodelist !aa
! if (ME_GLOBAL.eq.0) write(600,*) dEG !aa
! if (ME_GLOBAL.eq.0) write(600,*) '***************', cv%rex_beta !aa
! close(600)
!
! modify replica map (assumes that only adjacent switches are possible)
      j=1
      do while (j.lt.nstring)
        if (nodelist(j).gt.nodelist(j+1)) then ! node numbers start at 0
          i=cv%rex_map(j)
          cv%rex_map(j)=cv%rex_map(j+1)
          cv%rex_map(j+1)=i
          j=j+1
        endif
        j=j+1
      enddo
!
      if (any(nodelist.ne.(/ (i, i=0,nstring-1) /))) &
     & call multicom_permute_string_ranks(nodelist+1) ! special-purpose routine to reorder ranks in order of string replicas
! added 1 because in multicom node indices start from 1
      if (success) then
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) mestring=ME_STRNG
! broadcast string size to all slave nodes

#if (KEY_INTEGER8==0)
       call PSND4(mestring,1) 
#endif
#if (KEY_INTEGER8==1)
       call PSND8(mestring,1) 
#endif



       call set_param('MESTRING',mestring)
      endif
! write(600+ME_GLOBAL, *) ME_STRNG
! close(600+ME_GLOBAL)
!
      end subroutine smcv_repl_exchange
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! two auxiliary subroutines for projected components (experimental, see rtmd)
! both subroutines assume that the CV gradients are known and valid
      subroutine smcv_gradcv_dot_dr(rx,ry,rz,c)
! for now, this routine is not parallel, since it should execute pretty fast
      use stream
      real(chm_real) :: rx(:), ry(:), rz(:)
      integer, optional :: c
!
      integer :: col
!
      integer :: i
      real(chm_real) :: dcv
      character(len=len("SMCV_GRADCV_DOT_DR>") ),parameter::whoami="SMCV_GRADCV_DOT_DR>";!macro
!
      if (.not.cv_common_grad_initialized) call cv_common_grad_init() ! allocate cv%grad array (for derivatives)
!
      if (present(c)) then ; col=c ; else ; col=zold ; endif ! no bounds check so be careful
!
! loop over all cv and make calls by type
      if (cv%num_cv.eq.0) then
       call wrndie(0,whoami,trim('NO CV DEFINED.'))
      endif
      do i=1, cv%num_cv
       select case(cv%type(i))
! case(posi_com_x, posi_com_y, posi_com_z);
! dcv=cv_posi_com_grad_dot_dr(i,rx,ry,rz)
! case(qcomp_1, qcomp_2, qcomp_3, qcomp_4);
! dcv=cv_qcomp_grad_dot_dr(i,rx,ry,rz)
! case(angle_com);
! dcv=cv_angle_com_grad_dot_dr(i,rx,ry,rz)
! case(anglvec);
! dcv=cv_anglvec_grad_dot_dr(i,rx,ry,rz)
        case(dihe_com);
         dcv=cv_dihe_com_grad_dot_dr(i,rx,ry,rz)
! case(dist_com);
! dcv=cv_dist_com_grad_dot_dr(i,rx,ry,rz)
! case(cvrms);
! dcv=cv_cvrms_grad_dot_dr(i,rx,ry,rz)
        case default; call wrndie(0,whoami,trim('UNKNOWN CV SPECIFIED.'))
       end select
       cv%r(i,col)=dcv
      enddo ! loop over cv
!
      end subroutine smcv_gradcv_dot_dr
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_dcv_dot_gradcv(rx,ry,rz,c)
! for now, this routine is not parallel, since it should execute pretty fast
      use stream
      real(chm_real) :: rx(:), ry(:), rz(:)
      integer, optional :: c
!
      integer :: col
      real(chm_real) :: dummy
!
      integer :: i
      character(len=len("SMCV_DCV_DOT_GRADCV>") ),parameter::whoami="SMCV_DCV_DOT_GRADCV>";!macro
!
      if (.not.cv_common_grad_initialized) call cv_common_grad_init() ! allocate cv%grad array (for derivatives)
!
      if (present(c)) then ; col=c ; else ; col=zold ; endif ! no bounds check so be careful
!
! loop over all cv and make calls by type
      if (cv%num_cv.eq.0) then
       call wrndie(0,whoami,trim('NO CV DEFINED.'))
      endif
      do i=1, cv%num_cv
       dummy=cv%r(i,col)
       select case(cv%type(i))
! case(posi_com_x, posi_com_y, posi_com_z);
! call cv_posi_com_dcv_dot_grad(i,rx,ry,rz,dummy)
! case(qcomp_1, qcomp_2, qcomp_3, qcomp_4);
! call cv_qcomp_dcv_dot_grad(i,rx,ry,rz,col)
! case(angle_com);
! call cv_angle_com_dcv_dot_grad(i,rx,ry,rz,dummy)
! case(anglvec);
! call cv_anglvec_dcv_dot_grad(i,rx,ry,rz,dummy)
        case(dihe_com);
         call cv_dihe_com_dcv_dot_grad(i,rx,ry,rz,dummy)
! case(dist_com);
! call cv_dist_com_dcv_dot_grad(i,rx,ry,rz,dummy)
! case(cvrms);
! call cv_cvrms_dcv_dot_grad(i,rx,ry,rz,dummy)
        case default; call wrndie(0,whoami,trim('UNKNOWN CV SPECIFIED.'))
       end select
      enddo ! loop over cv
!
      end subroutine smcv_dcv_dot_gradcv
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_fill(x,y,z,mass,c)
! for now, this routine is not parallel, since it should not be called often
      use stream
      real(chm_real) :: x(:), y(:), z(:), mass(:) ! mass(size(x,1))
      integer, optional :: c
!
      integer :: i
      logical :: calcz, deriv
      character(len=len("SMCV_FILL>") ),parameter::whoami="SMCV_FILL>";!macro
!
      if (.not.cv_common_grad_initialized) call cv_common_grad_init() ! allocate cv%grad array (for derivatives)
!
      calcz=.true.
      deriv=.true.
! loop over all cv and make calls by type
      if (cv%num_cv.eq.0) then
       call wrndie(0,whoami,trim('NO CV DEFINED.'))
      endif
      do i=1, cv%num_cv
       select case(cv%type(i))
        case(posi_com_x, posi_com_y, posi_com_z);
         call cv_posi_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,.false.) ! force arguments dummy
        case(qcomp_1, qcomp_2, qcomp_3, qcomp_4);
         call cv_qcomp_calc(i,x,y,z,mass,x,y,z,calcz,deriv,.false.)
        case(angle_com);
         call cv_angle_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,.false.)
        case(anglvec);
         call cv_anglvec_calc(i,x,y,z,mass,x,y,z,calcz,deriv,.false.)
        case(dihe_com);
         call cv_dihe_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,.false.)
        case(rmsd);
         call cv_rmsd_calc(i,x,y,z,mass,x,y,z,calcz,deriv,.false.)
        case(drmsd);
         call cv_drmsd_calc(i,x,y,z,mass,x,y,z,calcz,deriv,.false.)
        case(proj);
         call cv_proj_calc(i,x,y,z,mass,x,y,z,calcz,deriv,.false.)
        case(dist_com);
         call cv_dist_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,.false.)
        case(cvrms);
         call cv_cvrms_calc(i,x,y,z,mass,x,y,z,calcz,deriv,.false.)
        case default; call wrndie(0,whoami,trim('UNKNOWN CV SPECIFIED.'))
       end select
! now cv%r(i,instant) is filled
       if (present(c)) then
        call cv_common_fill(i,cv%r(i,instant),c)
       else
        call cv_common_fill(i,cv%r(i,instant))
       endif
      enddo ! loop over cv
!
      end subroutine smcv_fill
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      function smcv_test_grad_fd(x,y,z,mass,h)
      use stream
      real(chm_real) :: x(:), y(:), z(:), mass(:) ! mass(size(x,1))
      real(chm_real) :: h
      real(chm_real), pointer :: error(:,:), smcv_test_grad_fd(:,:)
      real(chm_real) :: grad(cv%num_cv,cv%amap%last,3) ! temporary gradient array
      integer :: i, j, jj
      real(chm_real) :: dummy
!
      character(len=len("SMCV_TEST_GRAD_FD>") ),parameter::whoami="SMCV_TEST_GRAD_FD>";!macro
!
      allocate(error(cv%num_cv,3)); smcv_test_grad_fd=>error
      error=0d0
!
      if (cv%num_cv.eq.0) then
       call wrndie(0,whoami,trim(' NO CV DEFINED.'))
       return
      endif
!
      if (h.eq.0d0) then
       call wrndie(0,whoami,trim(' COORDINATE PERTURBATION ZERO.'))
       return
      endif
!
! compute cv values and derivatives analytically
! force recalculation of frames and quaternions
      call frames_reset_calculate(.true.)
      call quat_reset_calculate(.true.)
      call smcv_fill(x,y,z,mass,main)
!
! copy gradients (they will be overwritten by repeated calls to smcv_fill)
      grad(:,:,1)=cv%grad(1:cv%num_cv, 1:cv%amap%last, 1, 1)
      grad(:,:,2)=cv%grad(1:cv%num_cv, 1:cv%amap%last, 1, 2)
      grad(:,:,3)=cv%grad(1:cv%num_cv, 1:cv%amap%last, 1, 3)
!
! frames and quaternions will be calculated below by each processor as needed by CV
! loop over all coordinates and compute finite differences
      do jj=1, cv%amap%last
       j=cv%amap%i(jj) ! psf index
! x-derivatives *******************************************
       dummy=x(j) ; x(j)=dummy-h
! force recalculation of frames and quaternions
       call frames_reset_calculate(.true.)
       call quat_reset_calculate(.true.)
       call smcv_fill(x,y,z,mass,zcur) ! overwrite zcur cv value array (warn about this elsewhere)
       x(j)=dummy+h
! force recalculation of frames and quaternions
       call frames_reset_calculate(.true.)
       call quat_reset_calculate(.true.)
       call smcv_fill(x,y,z,mass,zold) ! overwrite zold cv value array (warn about this elsewhere)
       x(j)=dummy ! restore correct value
! compute largest absolute error
       do i=1, cv%num_cv
        error(i,1) = max (error(i,1), &
     & abs(0.5d0/h*(cv%r(i,zold)-cv%r(i,zcur))-grad(i,jj,1))) ! note: technically, will need to worry about periodicity in the finite difference
       enddo
! y-derivatives *******************************************
       dummy=y(j) ; y(j)=dummy-h
! force recalculation of frames and quaternions
       call frames_reset_calculate(.true.)
       call quat_reset_calculate(.true.)
       call smcv_fill(x,y,z,mass,zcur)
       y(j)=dummy+h
! force recalculation of frames and quaternions
       call frames_reset_calculate(.true.)
       call quat_reset_calculate(.true.)
       call smcv_fill(x,y,z,mass,zold)
       y(j)=dummy ! restore correct value
! compute largest absolute error
       do i=1, cv%num_cv
        error(i,2) = max (error(i,2), &
     & abs(0.5/h*(cv%r(i,zold)-cv%r(i,zcur))-grad(i,jj,2)))
       enddo
! z-derivatives ******************************************
       dummy=z(j) ; z(j)=dummy-h
! force recalculation of frames and quaternions
       call frames_reset_calculate(.true.)
       call quat_reset_calculate(.true.)
       call smcv_fill(x,y,z,mass,zcur)
       z(j)=dummy+h
! force recalculation of frames and quaternions
       call frames_reset_calculate(.true.)
       call quat_reset_calculate(.true.)
       call smcv_fill(x,y,z,mass,zold)
       z(j)=dummy ! restore correct value
! compute largest absolute error
       do i=1, cv%num_cv
        error(i,3) = max (error(i,3), &
     & abs(0.5/h*(cv%r(i,zold)-cv%r(i,zcur))-grad(i,jj,3)))
       enddo
      enddo ! over all atom map
!
! since gradients are last calculated based on perturbed values,
! recalculate them again
      call frames_reset_calculate(.true.)
      call quat_reset_calculate(.true.)
      call smcv_fill(x,y,z,mass,main)
!
      end function smcv_test_grad_fd
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      function smcv_test_parallel(x,y,z,mass)
      use multicom_aux;
      use mpi
      use stream
!
      real(chm_real) :: x(:), y(:), z(:), mass(:) ! mass(size(x,1))
      real(chm_real), pointer :: error(:,:), smcv_test_parallel(:,:)
      real(chm_real) :: grad(cv%num_cv,cv%amap%last,3) ! temporary gradient array
!
      integer :: i
      logical :: qcv, qfr, qqt
      logical :: qgrp
      real(chm_real) :: dummy(size(x,1))
!
      character(len=len("SMCV_TEST_PARALLEL>") ),parameter::whoami="SMCV_TEST_PARALLEL>";!macro
!
      allocate(error(cv%num_cv,4)) ! first column contains the CV values; then maximum derivative error (x,y,z)
      smcv_test_parallel=>error
!
      if (cv%num_cv.eq.0) then
       call wrndie(0,whoami,trim('NO CV DEFINED.'))
       return
      endif
!
      qgrp=(MPI_COMM_LOCAL.ne.MPI_COMM_NULL) &
     & .and.(SIZE_LOCAL.gt.1)
      if (.not. qgrp) then ! quit if cannot run in parallel
       call wrndie(0,whoami,trim(' CANNOT PERFORM TEST ON 1-PROCESSOR GROUPS'))
       return
      endif
! save values & force a serial calculation
      qcv=calc_cv_para; calc_cv_para=.false.
      qfr=calc_fr_para; calc_fr_para=.false.
      qqt=calc_qt_para; calc_qt_para=.false.
!
! will use the addforce routine to reuse code and to take advantage of its flexibility
! 1) compute serially
      call frames_reset_calculate(.true.)
      call quat_reset_calculate(.true.)
      call smcv_addforce(x,y,z,mass,dummy,dummy,dummy)
! copy cv
      cv%r(1:cv%num_cv, zold)=cv%r(1:cv%num_cv, instant)
! copy gradients (they will be overwritten by repeated calls to smcv_fill)
      grad(:,:,1)=cv%grad(1:cv%num_cv, 1:cv%amap%last, 1, 1)
      grad(:,:,2)=cv%grad(1:cv%num_cv, 1:cv%amap%last, 1, 2)
      grad(:,:,3)=cv%grad(1:cv%num_cv, 1:cv%amap%last, 1, 3)
! 2) compute in (fully) parallel
      calc_cv_para=.true.
      calc_fr_para=.true.
      calc_qt_para=.true.
!
      call frames_reset_calculate(.true.)
      call quat_reset_calculate(.true.)
      call smcv_addforce(x,y,z,mass,dummy,dummy,dummy)
! compute largest absolute error
      do i=1, cv%num_cv
        error(i,1) = &
     & abs(cv%r(i, zold)-cv%r(i, instant)) ! technically _should_ account for periodicity
        error(i,2) = &
     & maxval ( abs (cv%grad(i,1:cv%amap%last,1,1)-grad(i,:,1)))
        error(i,3) = &
     & maxval ( abs (cv%grad(i,1:cv%amap%last,1,2)-grad(i,:,2)))
        error(i,4) = &
     & maxval ( abs (cv%grad(i,1:cv%amap%last,1,3)-grad(i,:,3)))
      enddo
! restore original parameters
      calc_cv_para=qcv
      calc_fr_para=qfr
      calc_qt_para=qqt
! reinitialize equlibrium work variables (addforce modifies them)
      call cv_common_neq_work_init()
!
      end function smcv_test_parallel
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_voronoi_whereami(x,y,z,mass)
      use stream
!
       real(chm_real) :: x(:), y(:), z(:), mass(:) ! mass(size(x,1))
! locals
      logical :: calcz, deriv, addforce
      integer :: i
      real(chm_real) :: rtemp(cv%num_cv) ! for storing current CV values (theta(x))
!
      character(len=len("SMCV_VORONOI_WHEREAMI>") ),parameter::whoami="SMCV_VORONOI_WHEREAMI>";!macro
! do some work
!
      calcz=.true.; deriv=.false. ; addforce=.false.
!
      call frames_reset_calculate(.false.) ! make sure that frames recalculate their axes (but not gradients)
      call quat_reset_calculate(.false.)
!
      do i=1, cv%num_cv
       select case(cv%type(i))
!
        case(posi_com_x, posi_com_y, posi_com_z);
         call cv_posi_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(qcomp_1, qcomp_2, qcomp_3, qcomp_4);
         call cv_qcomp_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(angle_com);
         call cv_angle_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(anglvec);
         call cv_anglvec_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(dihe_com);
         call cv_dihe_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(rmsd);
         call cv_rmsd_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(drmsd);
         call cv_drmsd_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(proj);
         call cv_proj_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(dist_com);
         call cv_dist_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(cvrms);
         call cv_cvrms_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case default
         call wrndie(0,whoami,trim('UNKNOWN CV SPECIFIED.'))
       end select
       rtemp(i)=cv%r(i,instant)
      enddo
!
      cv%voronoi_whereami=cv_common_voronoi_compute(rtemp)
!
      end subroutine smcv_voronoi_whereami
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      function smcv_voronoi_compute(x,y,z,mass,itime)
! not parallelized yet; have to decide what level of parallelization is time-efficient
      use sm_var, only: nstring
      use lu ! for computing FE
      use stream
      use multicom_aux;
      use mpi
      use clcg_mod, only: random; use reawri, only: iseed
      use parallel, only: psnd4, psnd8
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
      real(chm_real) :: x(:), y(:), z(:)
      real(chm_real) :: mass(:) ! assumed shape
! real(chm_real) :: mass(size(x,1)) ! assumed size
      integer :: itime ! timestep -- needed by new version of Voronoi
! local var.
      integer :: i, j, k, l, m, which, me
      integer*4 :: ierror, m_
      integer*4 :: length(nstring-1)
      integer*4 :: request(nstring-1)
      logical :: smcv_voronoi_compute ! returns false if theta_i(x) does not correspond to z_i
      logical :: calcz, deriv, addforce
      real(chm_real) :: rtemp(cv%num_cv) ! for storing current CV values (theta(x))
!
      integer, pointer :: vtemp(:), vtemp2(:) ! for gathering Voronoi stats; this is an upper bound
      integer*4 :: stat(MPI_STATUS_SIZE)
      logical :: voronoi_update
      logical :: success, qgrp, qstring, ready(nstring-1), ok
      real(chm_real) :: P_accept_cross
! for computing FE
! real(chm_real) :: flux(nstring, nstring), ! normalized probability flux
! & prob(nstring), ! probability of being in a Voronoi cell
! & netflux(nstring) ! net probability flux into a Voronoi cell
! integer :: permut(nstring), d, code ! for LU decomposition routine
!
      character(len=len("SMCV_VORONOI_COMPUTE>") ),parameter::whoami="SMCV_VORONOI_COMPUTE>";!macro
! loop over all cv and make calls by type
      if (cv%num_cv.eq.0) then
       call wrndie(0,whoami,trim('NO CV DEFINED.'))
      endif
!
      qstring=(MPI_COMM_STRNG.ne.MPI_COMM_NULL) &
     & .and.(SIZE_STRNG.gt.1)
      qgrp=(MPI_COMM_LOCAL.ne.MPI_COMM_NULL) &
     & .and.(SIZE_LOCAL.gt.1)
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! pool crossing+occupancy
      voronoi_update=voronoi_allow_cross.and.(voronoi_update_freq.gt.0)
      if (voronoi_update) &
     & voronoi_update=(mod(itime,voronoi_update_freq).eq.0)
!
      if (voronoi_update) then
! only roots
         if (qstring) then
! pack
          allocate(vtemp(3*nstring*(2*nstring+1))) ! upper bound
          k=0
          do i=1, nstring
! do j=1, 2*nstring+1 ! send everything
           do j=2*nstring+1, 2*nstring+1 ! send just the occupancy: this means FE cannot be computed
            l=cv%voronoi_data(i,j,1)
            if (l.gt.0) then
             k=k+1; vtemp(k)=i; k=k+1; vtemp(k)=j; k=k+1; vtemp(k)=l
            endif
           enddo
          enddo
!
          if (ME_STRNG.ne.0) then
           call MPI_ISEND(vtemp, k, & ! send local attempts, accepts + occupancy to root packed in temp
     & mpiint, 0, ME_STRNG, MPI_COMM_STRNG, request(ME_STRNG), ierror)
          else ! rank 0
! wait for all messages to begin sending
           ready=.false.
           do while (any(.not.ready))
            do m=1,nstring-1
             if (.not.ready(m)) then
              ok=.false. ! this is necessary
              m_=m ! cast
              call MPI_IPROBE(m_, m_, MPI_COMM_STRNG, ok, stat, ierror)
              if (ok) then
! get message length
                call MPI_Get_count(stat,mpiint,length(m),ierror)
! write(600+ME_STRNG,*) ok, ready(m), m, length(m), stat
                ready(m)=.true.
              endif ! ok
             endif ! .not.ready(m)
            enddo ! m=1,nstring-1
           enddo ! now have all the lengths
! close(600+ME_STRNG) !aa
! begin receiving
           l=k+sum(length)
           allocate(vtemp2(l))
           do m=1,nstring-1
            l=k+sum(length(1:m-1))+1
            call MPI_IRECV(vtemp2(l), length(m), &
     & mpiint, m, m, MPI_COMM_STRNG, request(m), ierror)
           enddo ! m
! copy vtemp into vtemp2
           vtemp2(1:k)=vtemp(1:k)
          endif ! ME_STRNG
         endif ! MPI_COMM_STRNG
      endif ! voro_update
! do some work
!
      calcz=.true.; deriv=.false. ; addforce=.false.
!
      call frames_reset_calculate(.false.) ! make sure that frames recalculate their axes (but not gradients)
      call quat_reset_calculate(.false.)
!
      do i=1, cv%num_cv
       select case(cv%type(i))
!
        case(posi_com_x, posi_com_y, posi_com_z);
         call cv_posi_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(qcomp_1, qcomp_2, qcomp_3, qcomp_4);
         call cv_qcomp_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(angle_com);
         call cv_angle_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(anglvec);
         call cv_anglvec_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(dihe_com);
         call cv_dihe_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(rmsd);
         call cv_rmsd_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(drmsd);
         call cv_drmsd_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(proj);
         call cv_proj_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(dist_com);
         call cv_dist_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case(cvrms);
         call cv_cvrms_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
        case default
         call wrndie(0,whoami,trim('UNKNOWN CV SPECIFIED.'))
       end select
       rtemp(i)=cv%r(i,instant)
      enddo
!
! update
      if (voronoi_update) then
! root waits for all messages; then sends back concatenated array
       if (qstring) then
        if (ME_STRNG.eq.0) then
! wait for all messages to arrive
         call MPI_WAITALL(nstring-1, request, MPI_STATUSES_IGNORE, ierror)
! now send received array to all cpus:
         do m=1, nstring-1
           call MPI_ISEND(vtemp2, k+sum(length), & ! send local attempts, accepts + occupancy to root packed in temp
     & mpiint, m, m, MPI_COMM_STRNG, request(m), ierror)
         enddo
        else ! other roots
! first make sure previous send was successful
         call MPI_WAIT(request(ME_STRNG), stat, ierror)
! test for message from root:
         ok=.false.
         do while (.not.ok)
          call MPI_IPROBE(0, ME_STRNG, MPI_COMM_STRNG, ok, stat, ierror)
         enddo
! a message is ready to be received : begin
! get message length
         call MPI_Get_count(stat,mpiint,length(1),ierror)
         l=length(1)
! begin receiving
         allocate(vtemp2(l))
         call MPI_IRECV(vtemp2, l, mpiint, &
     & 0, ME_STRNG, MPI_COMM_STRNG, request(ME_STRNG), ierror)
        endif ! ME
       endif ! string root
      endif ! voro_update
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! do some more work
      which=cv_common_voronoi_compute(rtemp)
      me=cv%voronoi_whereami
!
      if (voronoi_update) then
        if (qstring) then
         if (ME_STRNG.ne.0) call MPI_WAIT(request(ME_STRNG), stat, ierror) ! make sure message is received
! unpack message (all string roots do this)
         cv%voronoi_data(:,:,2)=cv%voronoi_data(:,:,3) ! statistics from previous runs (if any)
         k=0
         do while (k.lt.l)
           k=k+1; i=vtemp2(k); k=k+1; j=vtemp2(k); k=k+1;
           cv%voronoi_data(i,j,2)=cv%voronoi_data(i,j,2)+vtemp2(k)
         enddo
         if (ME_STRNG.eq.0) call MPI_WAITALL(nstring-1, request, &
     & MPI_STATUSES_IGNORE, ierror)
         deallocate(vtemp, vtemp2)
        endif ! string roots
!
        if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.eq.1) &
     & cv%voronoi_data(:,:,2)=cv%voronoi_data(:,:,1)
!
      endif ! voronoi_update
!
!
      if (which.lt.0) then
        smcv_voronoi_compute=.false. ! cutoff exceeded ; reflect ; do not update log
      elseif (which.eq.me) then ! stayed in the same cell as before
       cv%voro_occupancy(which)=cv%voro_occupancy(which)+1 ! update occupancy log
       smcv_voronoi_compute=.true.
      else ! crossed into a different cell
       cv%cross_attempt(me,which)=cv%cross_attempt(me,which)+1
!
       if (voronoi_allow_cross.and.itime.gt.voronoi_nocross_ini) then
! decide whether to allow the crossing
! roots decide
        if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
! do i=1,nstring
! if (cv%voro_occupancyG(i).eq.0) cv%voro_occupancyG(i)=1
! enddo
! dummy=1d0/sum(cv%voro_occupancyG)
!************************* acceptance criterion ********************
! P_accept_cross=
! & (sum(1d0*cv%cross_acceptG(:,me)/cv%voro_occupancyG(:) )*
! & cv%voro_occupancyG(me) ! note: cross_accept(i,i)=0
! & -(sum(cv%cross_acceptG(me,:))-cv%cross_acceptG(me,which)))/
! & (max(cv%cross_attemptG(me,which),1))
!********************************************************************
! & ( sum(1d0*cv%cross_attemptG(:,me)/cv%voro_occupancyG(:))-
! & sum(1d0*cv%cross_attemptG(me,:))/cv%voro_occupancyG(me) )
! & /
! & ( sum(1d0*cv%cross_attemptG(:,which)/cv%voro_occupancyG(:))-
! & sum(1d0*cv%cross_attemptG(which,:))/cv%voro_occupancyG(which))
!********************************************************************
! & sum(1d0*cv%cross_attemptG(:,me)-cv%cross_attemptG(me,:))/
! & sum(1d0*cv%cross_attemptG(:,which)-cv%cross_attemptG(which,:))
!********************************************************************
! compute free energy from crossing attemps
! do i=2,nstring
! flux(i,:)=
! & 1d0*cv%cross_attemptG(:,i)/cv%voro_occupancyG(:)
! flux(i,i)=
! & -1d0*sum(cv%cross_attemptG(i,:))/cv%voro_occupancyG(i)
! if (flux(i,i-1).eq.0d0) flux(i,i-1)=
! & 1d0/cv%voro_occupancyG(i-1) ! crude regularization
! if (flux(i-1,i).eq.0d0) flux(i-1,i)=
! & 1d0/cv%voro_occupancyG(i)
! enddo
! netflux(2:nstring)=0d0;
! netflux(1)=1d0 ! boundary condition (p(1)=1)
! flux(1,1)=1d0; flux(1,2:nstring)=0d0 ! boundary condition
! prob=netflux;
! call ludcmp(flux, nstring, permut, d, code)
! call lubksb(flux, nstring, permut, prob)
!
         P_accept_cross=1d0 &
     & * (2-abs(which-me)) & ! allow crosses only between adjacent cells (ad hoc)
! & *prob(me)/prob(which) ! free energy estimate
! & *exp(10d0*(cv%voro_occupancyG(me)/cv%voro_occupancyG(which)-1d0))
     & *(cv%voro_occupancyG(me)/max(cv%voro_occupancyG(which),1))
!********************************************************************
         if (P_accept_cross.ge.1) then
          success=.true.
         else
          success=(random(iseed).le.P_accept_cross)
         endif
        endif ! string roots
! write(600+ME_STRNG,*) P_accept_cross, success, ' * ', prob
! broadcast to slaves
        if (qgrp) call PSND4(success,1)
! if successful, update
        if (success) then
         cv%cross_accept(me,which)=cv%cross_accept(me,which)+1
         cv%voro_occupancy(which)=cv%voro_occupancy(which)+1
         cv%voronoi_whereami=which
        else
         cv%voro_occupancy(me)=cv%voro_occupancy(me)+1
        endif
!
        smcv_voronoi_compute=success
       else ! crossing not allowed, so reflect
        smcv_voronoi_compute=.false.
        cv%voro_occupancy(me)=cv%voro_occupancy(me)+1 ! update occupancy log
       endif ! voronoi_allow_cross
! update voronoi log.
       j=ME_STRNG+1 ! possible typecast
       i=int_vector_add(cv%voro_log, j) ! replica ID
       i=int_vector_add(cv%voro_log, me) ! from this cell
       i=int_vector_add(cv%voro_log, which) ! into this cell
       i=int_vector_add(cv%voro_log, cv%voronoi_whereami) ! now in this cell
       i=int_vector_add(cv%voro_log, itime+vtime_offset) ! at this time
      endif ! which<0
!
      end function smcv_voronoi_compute
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! subroutine smcv_voronoi_smart_update(xcomp, ycomp, zcomp, dx, dy, dz,
! & mass, iteration)
      subroutine smcv_voronoi_smart_update(x, y, z, xcomp, ycomp, zcomp,&
     & mass, iteration)
! currently not parallelized; I assume that it will be called infrequently (e.g. 1 in 20 or more iter.)
      use stream
      use multicom_aux;
      use mpi
! real(chm_real) :: dx(:), dy(:), dz(:), xcomp(:), ycomp(:), zcomp(:)
      real(chm_real) :: x(:), y(:), z(:), xcomp(:), ycomp(:), zcomp(:)
      real(chm_real) :: mass(:) ! mass(size(x,1))
      integer :: i, iteration
      logical :: calcz, deriv, addforce
! local var.
      real(chm_real) :: rnew(cv%num_cv) ! for storing current CV values (theta(x))
      real(chm_real) :: rold(cv%num_cv) ! for storing additional (old) CV values (theta(xcomp))
      real(chm_real) :: A(3,3,frames%num_frames), o(3,frames%num_frames), &
     & q(4,quat%num_quat)
      character(len=len("SMCV_VORONOI_SMART_UPDATE>") ),parameter::whoami="SMCV_VORONOI_SMART_UPDATE>";!macro
! loop over all cv and make calls by type
      if (cv%num_cv.eq.0) then
       call wrndie(0,whoami,trim('NO CV DEFINED.'))
      endif
! compute theta(x) -- note this is possibly redundant!
      calcz=.true. ; deriv=.false. ; addforce=.false.
!
      if ( (hist_freq.le.0) .or. &
     & (hist_freq.gt.0).and.(mod(iteration,hist_freq).ne.0) ) then ! cv%r(:,instant) was not computed in smcv_addhist
       do i=1, cv%num_cv
! compute rnew
        select case(cv%type(i))
!
         case(posi_com_x, posi_com_y, posi_com_z);
          call cv_posi_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
         case(qcomp_1, qcomp_2, qcomp_3, qcomp_4);
          call cv_qcomp_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
         case(dihe_com);
          call cv_dihe_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
         case(rmsd);
          call cv_rmsd_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
         case(drmsd);
          call cv_drmsd_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
         case(proj);
          call cv_proj_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
         case(angle_com);
         call cv_angle_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
         case(anglvec);
          call cv_anglvec_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
         case(dist_com);
          call cv_dist_com_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
         case(cvrms);
          call cv_cvrms_calc(i,x,y,z,mass,x,y,z,calcz,deriv,addforce)
         case default
          call wrndie(0,whoami,trim('UNKNOWN CV SPECIFIED.'))
        end select
        rnew(i)=cv%r(i,instant)
      enddo
      else
       rnew=cv%r(1:cv%num_cv,instant)
      endif
! compute rold
! save frames
      if (frames_initialized) then
       A=frames%r(:,:,1:frames%num_frames)
       o=frames%o(:,1:frames%num_frames)
       call frames_reset_calculate(.false.) ! make sure that frames recalculate their axes
      endif
! save quaternions
      if (quat_initialized) then
       q=quat%q(:,1:quat%num_quat)
       call quat_reset_calculate(.false.)
      endif
!
      do i=1, cv%num_cv
        select case(cv%type(i))
!
         case(posi_com_x, posi_com_y, posi_com_z);
          call cv_posi_com_calc(i,xcomp,ycomp,zcomp,mass, &
     & xcomp,ycomp,zcomp,calcz,deriv,addforce)
         case(qcomp_1, qcomp_2, qcomp_3, qcomp_4);
          call cv_qcomp_calc(i,xcomp,ycomp,zcomp,mass, &
     & xcomp,ycomp,zcomp,calcz,deriv,addforce)
!
         case(dihe_com);
          call cv_dihe_com_calc(i,xcomp,ycomp,zcomp,mass, &
     & xcomp,ycomp,zcomp,calcz,deriv,addforce)
         case(rmsd);
          call cv_rmsd_calc(i,xcomp,ycomp,zcomp,mass, &
     & xcomp,ycomp,zcomp,calcz,deriv,addforce)
         case(drmsd);
          call cv_drmsd_calc(i,xcomp,ycomp,zcomp,mass, &
     & xcomp,ycomp,zcomp,calcz,deriv,addforce)
         case(proj);
          call cv_proj_calc(i,xcomp,ycomp,zcomp,mass, &
     & xcomp,ycomp,zcomp,calcz,deriv,addforce)
         case(angle_com);
          call cv_angle_com_calc(i,xcomp,ycomp,zcomp,mass, &
     & xcomp,ycomp,zcomp,calcz,deriv,addforce)
         case(anglvec);
          call cv_anglvec_calc(i,xcomp,ycomp,zcomp,mass, &
     & xcomp,ycomp,zcomp,calcz,deriv,addforce)
         case(dist_com);
          call cv_dist_com_calc(i,xcomp,ycomp,zcomp,mass, &
     & xcomp,ycomp,zcomp,calcz,deriv,addforce)
         case(cvrms);
          call cv_cvrms_calc(i,xcomp,ycomp,zcomp,mass, &
     & xcomp,ycomp,zcomp,calcz,deriv,addforce)
         case default
          call wrndie(0,whoami,trim('UNKNOWN CV SPECIFIED.'))
        end select
        rold(i)=cv%r(i,instant)
        cv%r(i,instant)=rnew(i) ! restore correct instantaneous values (note, if frames are present, they will have incorrect axes)
      enddo
! restore frames
      if (frames_initialized) then
       frames%r(:,:,1:frames%num_frames)=A
       frames%o(:,1:frames%num_frames)=o
      endif
! restore quaternions
      if (quat_initialized) then
       quat%q(:,1:quat%num_quat)=q
      endif
!
! call the base update routine
! all routines call
      if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL) then
       call cv_common_voronoi_smart_update(rnew,rold)
      endif
!
      end subroutine smcv_voronoi_smart_update
!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_list()
      use stream
      use multicom_aux;
      use mpi
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
! local
      integer :: i
      logical :: qprint
      character(len=len("SMCV_LIST>") ),parameter::whoami="SMCV_LIST>";!macro
!
      qprint=MPI_COMM_STRNG.ne.MPI_COMM_NULL &
     & .and.ME_STRNG.eq.0
!
      do i=1, cv%num_cv
       if (qprint) then ; write(info,666) whoami,i ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 666 format(A,' CV # ',I8,':')
       select case (cv%type(i))
         case (posi_com_x, posi_com_y, posi_com_z);
          call cv_posi_com_list(i) ! one routine handles all
         case (qcomp_1, qcomp_2, qcomp_3, qcomp_4);
          call cv_qcomp_list(i) ! one routine handles all
         case (dihe_com); call cv_dihe_com_list(i)
         case (rmsd); call cv_rmsd_list(i)
         case (drmsd); call cv_drmsd_list(i)
         case (proj); call cv_proj_list(i)
         case (angle_com); call cv_angle_com_list(i)
         case (anglvec); call cv_anglvec_list(i)
         case (dist_com); call cv_dist_com_list(i)
         case (cvrms); call cv_cvrms_list(i)
!
         case default
          call wrndie(0,whoami,trim('UNEXPECTED CV SPECIFIED.'))
       end select
      enddo ! over all cv
      end subroutine smcv_list
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_compute_wgt(x,y,z,mass) ! compute CV weights from current coordinates
                                              ! note that this routine clears all history
!
      real(chm_real) :: x(:), y(:), z(:), mass(:) ! mass(size(x,1))
!
      if (.not. cv_common_Minv_initialized) &
     & call smcv_compute_M(x,y,z,mass,.true.) ! compute M and M inverse
      call cv_common_compute_wgt() ! compute weights from M inverse
!
      end subroutine smcv_compute_wgt
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_compute_M(x,y,z,mass,inverse) ! compute matrix M from current coordinates
      use stream
      use mpi
      use multicom_aux;
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
      real(chm_real) :: x(:), y(:), z(:), mass(:) ! mass(size(x,1))
      logical :: inverse ! should inverse be computed?
      logical :: qgrp
      integer :: ierror
!
      qgrp=(MPI_COMM_LOCAL.ne.MPI_COMM_NULL) &
     & .and.(SIZE_LOCAL.gt.1)
!
      call cv_common_clear_hist() ! clear history
      cv%num_run_ave=0 ! number of slices in running averages
      call frames_reset_calculate(.true.) ! for projected dynamics
      call quat_reset_calculate(.true.)
      call smcv_add_hist(x,y,z,mass,.true.) ! .true. -- need to update average (for the calculation of M))
! need to add contributions to M, if running in parallel (reduce)
!
      if (qgrp.and.calc_Mtensor_para) then
       call MPI_ALLREDUCE(cv%M(1,1,1),cv%M(1,1,2),max_cv_common*cv%num_cv,&
     & mpifloat, MPI_SUM, MPI_COMM_LOCAL, ierror)
      else
       cv%M(1:cv%num_cv,1:cv%num_cv,2)=cv%M(1:cv%num_cv,1:cv%num_cv,1)
      endif
! set the M long-term average to the same thing
       cv%M(1:cv%num_cv,1:cv%num_cv,4)=cv%M(1:cv%num_cv,1:cv%num_cv,2)
! compute M inverse (from long term average) ; assume that M is regular
      if (inverse) call cv_common_compute_Minv(inverse_LU)
! clean up !
      call cv_common_clear_hist() ! clear history
      cv%num_run_ave=0 ! number of slices in running averages
      end subroutine smcv_compute_M
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      function smcv_test_Minv(x,y,z,mass) result(diff)
!
      real(chm_real), intent (in) :: x(:), y(:), z(:), mass(:) ! mass(size(x,1))
      real(chm_real) :: diff
      real(chm_real) :: temp(cv%num_cv, cv%num_cv)
      logical :: qlu
      qlu=inverse_LU
      inverse_LU=.true.
      call smcv_compute_M(x,y,z,mass,.true.)
      temp=cv%M(1:cv%num_cv,1:cv%num_cv,3)
      inverse_LU=.false.
      call smcv_compute_M(x,y,z,mass,.true.)
      diff=maxval(abs(temp-cv%M(1:cv%num_cv,1:cv%num_cv,3)))
      inverse_LU=qlu
      end function smcv_test_Minv
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_compute_frames_quat_para(x,y,z,mass)
! utility routine to synchronize frame and quaternion components
      use cv_frames
      use cv_quaternion
      use mpi
      use multicom_aux;
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
      real(chm_real) :: x(:), y(:), z(:), mass(:) ! mass(size(x,1))
! local vars
      integer :: ibeg, iend, i, ierror
      integer*4 :: qfac, ffac
!
! aardvark
! real(chm_real), pointer :: trash(:,:,:,:,:)
!
      if (MPI_COMM_LOCAL.eq.MPI_COMM_NULL) return
!
! note: for efficiency, should compute in parallel only if #frames/quaternions is fairly sizeable
!
! 1) frames
      if (calc_fr_para) then
       if (any(frames%recalculate)) then
!
        ibeg=fr_send_displ(ME_LOCAL+1)+1
        iend=fr_send_displ(ME_LOCAL+1)+fr_send_count(ME_LOCAL+1)
!
        do i=ibeg,iend
         call frames_calc(i,x,y,z,mass,.true.)
        enddo
! pack frame values into gradient array to make just one comm. call:
! these are the frame array dimensions (from cv_frames.src)
! dimension(3,nfr), pointer :: o ! frame origin
! dimension(3,3,nfr), pointer :: r ! frame vectors
! dimension(3,imap), pointer :: grado ! gradient vector of frame origin w.r.t. x y or z (the same) ;
! dimension(3,3,imap,3,nfr), pointer :: gradr ! combined gradient array (with some extra space for `packed` communication)
! a little complicated, c`est la vie
        frames%gradr(:,1,1,4,ibeg:iend) =frames%o(:,ibeg:iend)
        frames%gradr(1,2,:,4,ibeg:iend) =frames%grado(:,ibeg:iend)
        frames%gradr(:,3,2:4,4,ibeg:iend)=frames%r(:,:,ibeg:iend) ! note that is it safe to assume that cv%amap%last>=4,
! since a frame with 3 atom coords can never be defined
! uniquely
! write(0,*) me_group, frames%gradr(:,1,1,4,ibeg:iend)
! write(0,*) me_group, frames%o(:,ibeg:iend) !aa
! stop
! scale displacements/counts
        ffac=9*cv%amap%last*4
! communicate
! allocate(trash(3,3,cv%amap%last,4, frames%num_frames)) !aa
! aa - DANGEROUS BC TYPE IS WRONG (INT8)
! call MPI_ALLGATHERV(frames%o(1,ibeg), fr_send_count(ME_LOCAL+1)*
! & cv%amap%last, mpifloat, frames%o,
! & fr_send_count*cv%amap%last, fr_send_displ*cv%amap%last,
! & mpifloat, MPI_COMM_LOCAL, ierror)
        call mpi_allgatherv(frames%gradr(1,1,1,1,ibeg), &
     & fr_send_count(ME_LOCAL+1)*ffac, mpifloat, &
     & frames%gradr,fr_send_count*ffac, fr_send_displ*ffac, &
     & mpifloat, MPI_COMM_LOCAL, ierror)
! call mpi_allgatherv(frames%gradr(1,1,1,1,ibeg),
! & fr_send_count(ME_LOCAL+1)*ffac, mpifloat,
! & trash,fr_send_count*ffac, fr_send_displ*ffac,
! & mpifloat, MPI_COMM_LOCAL, ierror)
! unpack frame values
        ibeg=1
        iend=frames%num_frames
        frames%o(:,ibeg:iend)=frames%gradr(:,1,1,4,ibeg:iend)
! frames%o(:,ibeg:iend)=trash(:,1,1,4,ibeg:iend) !aa
        frames%grado(:,ibeg:iend)=frames%gradr(1,2,:,4,ibeg:iend)
        frames%r(:,:,ibeg:iend)=frames%gradr(:,3,2:4,4,ibeg:iend) ! note that is it safe to assume that cv%amap%last>=4,
! change recalculate flags (cheat)
        frames%recalculate=.false.
        frames%recalculate_grad=.false.
!
! deallocate(trash) !aa
!
! write(700+ME_LOCAL,*)'par: ', frames%r(:,:,1:frames%num_frames)
! write(800+ME_LOCAL,*)'par: ', frames%o(:,1:frames%num_frames)
! write(900+ME_LOCAL,*)
! & frames%gradrx(:,:,:,1:frames%num_frames)
! close(700+ME_LOCAL)
! close(800+ME_LOCAL)
! close(900+ME_LOCAL)
! second loop for planar sampling or parallelization within group
! write(0,*) ME_LOCAL, fr_send_count(ME_LOCAL+1)*ffac,
! & fr_send_displ(ME_LOCAL+1)*ffac
! write(0,*) SIZE(frames%gradr)
! stop !aa
! write(0,*) fr_send_displ*ffac,fr_send_count*ffac
!
!
!
!
!
       endif
      endif
!
! write(0,*) 'done inside quat para'
! return! stop
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (calc_qt_para) then
       if (any(quat%recalculate)) then
! 2) quaternions
        ibeg=qt_send_displ(ME_LOCAL+1)+1
        iend=qt_send_displ(ME_LOCAL+1)+qt_send_count(ME_LOCAL+1)
!
        do i=ibeg,iend
         call quat_calc(i,x,y,z,mass,.true.)
        enddo
!
! write(0,*) ME_LOCAL, ibeg, iend
! stop
! pack quaternion values into gradient array to make just one comm. call:
        quat%gradq(:,1,4,ibeg:iend)=quat%q(:,ibeg:iend)
! scale displacements/counts
        qfac=4*cv%amap%last*4
! communicate
        call mpi_allgatherv(quat%gradq(1,1,1,ibeg), &
     & qt_send_count(ME_LOCAL+1)*qfac, mpifloat, &
     & quat%gradq,qt_send_count*qfac, qt_send_displ*qfac, &
     & mpifloat, MPI_COMM_LOCAL, ierror)
! unpack quaternion values
! write(0,*) ME_LOCAL, ibeg, iend
! write(600+ME_LOCAL) quat%gradq
! close(600+ME_LOCAL)
! stop
        ibeg=1
        iend=quat%num_quat
        quat%q(:,ibeg:iend)=quat%gradq(:,1,4,ibeg:iend)
! change recalculate flags (cheat)
        quat%recalculate=.false.
        quat%recalculate_grad=.false.
       endif
      endif
!
      end subroutine smcv_compute_frames_quat_para
!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#endif
#endif /* automatically protect all code */
      end module smcv_master
