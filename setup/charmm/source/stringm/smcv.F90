#if (KEY_STRINGM==1) /*  automatically protect all code */
! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!CHARMM Element source/stringm/smcv.src $Revision: 1.4 $
!
! string code
!
#if (KEY_STRINGM==1)
!
      SUBROUTINE smcv(COMLYN,COMLEN)
!----------------------------------------------------------------------
! command parser for the string method
!----------------------------------------------------------------------
      use sm_var
      use cv_frames, only: frames_init, frames_list, frames_done, &
     & frames_read_local, frames_read_global, &
     & frames_print_local, frames_print_global, &
     & frames_calc, frames, frames_initialized, &
     & frames_reset_calculate, &
     & frames_calc_align_comp
      use cv_quaternion, only: quat_reset_calculate, quat_done
!
      use sm_config
      use smcv_master, only: smcv_fill, smcv_compute_wgt, smcv_add_hist,&
     & smcv_list, smcv_compute_M, smcv_voronoi_whereami, &
     & smcv_test_grad_fd, smcv_test_parallel, smcv_test_Minv
      use ftsm_var, only : ftsm_initialized
!
      use stream
      use dimens_fcm
      use string
      use coord; use coordc
      use psf
      use number
      use ctitla

      use dcntrl_mod, only : dynopt

#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use consta
      use mpi
      use cvio, only : writcv, readcv; use coorio_mod, only : cwrite, cread; use ctitla
      use cv_common ! to prevent cv from getting masked, put this last
      use parallel, only: psnd4, psnd8

      implicit none
!
      CHARACTER(len=*) :: COMLYN
      integer :: comlen
!
! local variables
!
      character(len=8) :: keyword
!

      integer :: ivver, ivv2, iorig, ileap ! for dynamics
      integer :: islct(natom)
      integer :: ifreea(natom)
      integer :: oldiol
! compatibility variables for coordinate reading/writing
      real(chm_real) :: xdum(natom+1), ydum(natom+1), zdum(natom+1), &
     & wdum(natom+1)
      integer :: icntrl(20)

!
      integer :: ifile
      integer :: i,j,n
      integer*4 :: ierror, impi, jmpi, kmpi
      real(chm_real) :: k,w,gam,step,expo_memory,zval
      integer :: num_ave_samples, irep
      integer :: c1, c2, klen, delta, nskip, scol=0, totcol
      integer :: ind, all, ibeg, iend
      character(len=80) :: fname
      integer :: flen
      logical :: ok
      character(len=20) :: fixbc
! integer*4, pointer, dimension(:) :: temp1, temp2 ! for MPI_GRAD_TYPE (alt)
      integer :: me, maxcv
!
! for interpolation
!
      character(len=20) :: methods(5), method, form
      character(len=80) :: name_cv_in, name_cv_out, name_cor_in, name_cor_out
      character(len=80) :: dummy
      character(len=80), pointer :: fname_cor_in(:), fname_cor_out(:)
      data methods/ 'LINEAR','CUBIC SPLINE','B-SPLINE','DST','LINEAR_EXACT'/
!
      integer :: int_method, length, num_rep_in, num_rep_out, &
     & len_cv_in, len_cv_out, len_cor_in, len_cor_out, ofile

      integer :: moder, modew
      logical :: lresid



!
      logical :: interp_cv, inte_get_coor, qcomp
      real(chm_real), pointer :: inte_rmsd(:,:), rtemp(:,:)
      real(chm_real) :: voro_cut, repl_x_temp
      logical :: min_rmsd, voronoi_check_map
      integer :: which(1)
      logical :: qroot, qslave , qprint
! tests
      real(chm_real), pointer :: fd_error(:,:) ! FD gradient computation
!
      integer(kind=MPI_ADDRESS_KIND) :: lb, extent
!
      character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
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
! interface to frames_align_string routine (needed since routine is not in a module and I use optional args
      interface
       subroutine frames_align_string(x,y,z,mass,min_rmsd,ind)
        use stream
        implicit none
        real(chm_real) :: x(:), y(:), z(:), mass(:)
        logical, optional :: min_rmsd
        integer, optional :: ind ! frame index
       end subroutine frames_align_string
!
       subroutine frame_align_rmsd(x,y,z,mass,ind)
        use stream
        implicit none
        real(chm_real) :: x(:), y(:), z(:), mass(:)
        integer, optional :: ind ! frame index
       end subroutine frame_align_rmsd
!
       subroutine frame_align_voro(x,y,z,mass,ind)
        use stream
        implicit none
        real(chm_real) :: x(:), y(:), z(:), mass(:)
        integer, optional :: ind ! frame index
       end subroutine frame_align_voro
!
       subroutine smcv_init(maxcv)
        implicit none
        integer, optional :: maxcv
       end subroutine smcv_init
!
       function sm_get_column(cmd_, l, qcoltag, missing) result(C)
        implicit none
        character(len=*) :: cmd_
        integer :: l, missing
        logical :: qcoltag
        integer :: C
       end function sm_get_column
!
      end interface
!
      character(len=len("SMCV>") ),parameter::whoami="SMCV>";!macro
!
      keyword=nexta8(comlyn,comlen)
!
      qroot=MPI_COMM_STRNG.ne.MPI_COMM_NULL
      qslave=((MPI_COMM_LOCAL.ne.MPI_COMM_NULL).and.SIZE_LOCAL.gt.1)
      qprint=qroot.and.ME_STRNG.eq.0
!
! check for ftsm initialization; quit if initialized
!
      if (ftsm_initialized) then
       call wrndie(0,whoami,trim(' FTSM IS ON AND CANNOT BE USED WITH SMCV. NOTHING DONE.'))
       return
      endif
!
#if (KEY_MPI==1)
      if (SIZE_STRNG.gt.1) call MPI_BARRIER(MPI_COMM_STRNG,ierror) 
#endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (( keyword(1:4).eq.'INIT'(1:4) )) then
        maxcv=gtrmi(comlyn, comlen, 'MAXCV', -999)
        if (maxcv.eq.-999 ) then ; call smcv_init() ; else ; call smcv_init(maxcv) ; endif
        return
      endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (.not.smcv_initialized) then
        call smcv_init()
      endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (( keyword(1:4).eq.'DONE'(1:4) )) then
        call smcv_done()
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'REPA'(1:4) )) then
       if (comlen.gt.0) then ! this is an initialization call!
        call smcv_repa_init(comlyn, comlen)
       else
        call smcv_repa()
       endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'STAT'(1:4) )) then
       if (comlen.gt.0) then ! this is an initialization call!
        call smcv_stat_init(comlyn, comlen)
       else
        call smcv_stat()
       endif ! call statistics
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'DYNA'(1:4) )) then
       ivver=indxa(comlyn, comlen, 'VVER')
       ivv2=indxa(comlyn, comlen, 'VV2')
       iorig=indxa(comlyn, comlen, 'ORIG')
       ileap=indxa(comlyn, comlen, 'LEAP')
       if ((ivver+ivv2+iorig).gt.0) then
        call wrndie(0,whoami,trim('ONLY LEAP-FROG DYNAMICS ARE SUPPORTED. NOTHING DONE'))
        return
       endif
! force LEAP DYNAMICS
       call joinwd(comlyn, mxcmsz, comlen, 'LEAP ', 5)
!ccccccccccccccc PARSE OTHER DYNAMICS OPTIONS
       voronoi_hist_on=(indxa(comlyn, comlen, 'VORO').gt.0)
       if (voronoi_hist_on) then
        voronoi_allow_cross=(indxa(comlyn, comlen, 'VCRS').gt.0)
        if (voronoi_allow_cross) then
         voronoi_update_freq=gtrmi(comlyn, comlen, 'VCRF', 0)
         if (voronoi_update_freq.le.0) then
          call wrndie(0,whoami,trim('MUST SPECIFY POSITIVE VCRF. VORONOI CELL CROSSING DISABLED.'))
          voronoi_allow_cross=.false.
         elseif (indx(comlyn, comlen, 'VINI', 4).gt.0) then ! if vini is present
          voronoi_nocross_ini=gtrmi(comlyn, comlen, 'VINI', 0) ! get it
          if (voronoi_nocross_ini.le.0) then
           call wrndie(0,whoami,trim('NONPOSITIVE VINI SPECIFIED. WILL SET TO ZERO.'))
           voronoi_nocross_ini=0
          endif ! voronoi_nocross_ini>0
         else
          voronoi_nocross_ini=0
         endif ! voronoi_nocross_ini present
        endif ! voronoi_allow_cross
!
! initialize Voronoi data
        if (.not.cv_common_voronoi_initialized) &
     & call cv_common_voronoi_init()
! standard V. calculation case -- no crossing
        compute_whereami=.false.
        if (.not.voronoi_allow_cross) then
! create standard map (unless map is present)

         if (.not.any(cv%voronoi_map.ne.-1)) then
          cv%voronoi_map=(/ (i, i=1, nstring) /)
          compute_whereami=.true. ! will be computed by dynamc routine
         endif
        endif
!
        voronoi_check_map=(indxa(comlyn, comlen, 'CHCK').gt.0)
!
! compute whereami
!
        if (voronoi_check_map) then
         if (qprint) then
          write(info, 660) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
 660 FORMAT(A,' CHECKING VORONOI MAP AGAINST CURRENT COORDINATES.')
!
         compute_whereami=.false.
         call smcv_voronoi_whereami(x,y,z,amass)
!
         if (any(cv%voronoi_map.ne.-1)) then
           me=cv%voronoi_map(mestring+1)
! compare me and whereami:
           if (qroot) then
            if(SIZE_STRNG.gt.1) then
             call MPI_ALLREDUCE(me.eq.cv%voronoi_whereami, ok, &
     & 1, MPI_LOGICAL, MPI_LAND, MPI_COMM_STRNG, ierror)
            else
             ok=me.eq.cv%voronoi_whereami
            endif
           endif ! qroot
           if (qslave) then
            call PSND4(ok,1)
           endif
           if (.not.ok) then
            call wrndie(0,whoami,trim('VORONOI MAP INCONSISTENT WITH CURRENT COORDINATES. ABORTING.'))
            return
           endif ! .not. ok
         else ! voronoi map invalid (or was not read); proceed anyway using current whereami
          call wrndie(0,whoami,trim('VORONOI MAP CONTAINS INVALID ENTRIES.'))
         endif ! voronoi_map.ne.-1
!
        else
         cv%voronoi_whereami=cv%voronoi_map(mestring+1)
        endif ! voronoi_check_map
!
       endif ! voronoi_hist_on
! reset internal interation counter for smcv_master
       olditeration=0
!
       repa_on=(indxa(comlyn, comlen, 'REPA').gt.0)
       if (repa_on) repa_freq=gtrmi(comlyn, comlen, 'REPF', 0)
!
       hist_freq=gtrmi(comlyn, comlen, 'HISF', 0)
!
       stat_on=(indxa(comlyn, comlen, 'STAT').gt.0)
       if (stat_on) stat_freq=gtrmi(comlyn, comlen, 'STAF', 0)
!
       evolve_cv_on=(indxa(comlyn, comlen, 'EVOL').gt.0)
       if (evolve_cv_on) then
        evolve_freq=gtrmi(comlyn, comlen, 'EVOF', 0)
        evolve_nskip=gtrmi(comlyn, comlen, 'EVOS', 0)
! express in terms of history frequency
        if (hist_freq.gt.0) evolve_nskip=evolve_nskip/hist_freq
!
        evolve_step=gtrmf(comlyn, comlen, 'EVST', 0.0d0) ! evolution step
! ----- types of evolution
        evolve_smooth_on=(indxa(comlyn, comlen, 'SMOO').gt.0) ! smooth trajectory
        if (evolve_smooth_on) then
         evolve_smooth_d=gtrmi(comlyn, comlen, 'EVOD', 1) ! smoothing filter
        endif
!
        evolve_expo_on=(indxa(comlyn, comlen, 'EXPO').gt.0) ! use exponential convolution
        if (evolve_expo_on) then
         evolve_expo_mem=gtrmf(comlyn, comlen, 'MEMO', 0.99d0)
        endif
!
        evolve_bd_on=(indxa(comlyn, comlen, 'BD').gt.0) ! use brownian dynamics (M not used); for T-accelerated sampling
        if (evolve_bd_on) then
! evolve step specified above (will modify this section later)
         evolve_bd_T=gtrmf(comlyn, comlen, 'TEMP', 0d0)
        endif
!
        evolve_aver_on=(indxa(comlyn, comlen, 'AVER').gt.0) ! z=mean(theta)
        if (evolve_aver_on) then
         num_ave_cv_samples=gtrmi(comlyn, comlen, 'NAVE', 0) ! initial number of samples in the averages
! setting this large will dampen initial fluctuations
        endif
       endif
!
       restrained_on=(indxa(comlyn, comlen, 'RSTR').gt.0)
       if (restrained_on) then
!
        steering_on=( (indxa(comlyn, comlen, 'SMD').gt.0) .or. (indxa(comlyn, comlen, 'STEE').gt.0) )
        if (.not.steering_on) call cv_common_copy(main, comp)
!
        restrained_eq_steps=gtrmi(comlyn, comlen, 'REEQ', 0)
        restrained_eq0=0
! for off-path sampling
        planar_on=(indxa(comlyn, comlen, 'PLAN').gt.0) ! restraint applied parallel to the path (this is obsolete)
       endif
!
       unrestrained_on=(indxa(comlyn, comlen, 'URES').gt.0)
       if (unrestrained_on) then
        unrestrained_eq_steps=gtrmi(comlyn, comlen, 'UREQ', 0)
        unrestrained_eq0=0
        restrained_eq0=0
! currently unrestrained dynamics works as a preliminary step for restrained dynamics, therefore :
        if (.not.restrained_on) then
         call wrndie(0,whoami,trim('UNRESTRAINED (EXPLORATION) DYNAMICS REQUIRES EQUILIBRATION WITH RESTRAINTS ("RSTR"). TURNING OFF.'));
         unrestrained_on=.false.
        endif
       endif
       repl_x_on=(indxa(comlyn, comlen, 'REX').gt.0)
       if (repl_x_on) then
        repl_x_freq=gtrmi(comlyn, comlen, 'REXF', 0)
        repl_x_temp=gtrmf(comlyn, comlen, 'REXT', 0d0)
!
        if (repl_x_freq.le.0) then
          call wrndie(0,whoami,trim('MUST SPECIFY POSITIVE REXF. REPLICA EXCHANGE IS OFF.'))
          repl_x_on=.false.
!
        elseif (repl_x_temp.le.0) then
          call wrndie(0,whoami,trim('MUST SPECIFY POSITIVE REXT. REPLICA EXCHANGE IS OFF.'))
          repl_x_on=.false.
        elseif (voronoi_hist_on) then
          call wrndie(0,whoami,trim('REPLICA EXCHANGE INCOMPATIBLE WITH V. TESSELATION. REX IS OFF.'))
          repl_x_on=.false.
        elseif (evolve_cv_on.and.mod(repl_x_freq,evolve_freq).gt.0) then
          call wrndie(0,whoami,trim('REPLICA SWAP ATTEMPT FREQ. NOT A MULTIPLE OF EVOLUTION FREQ.'))
        elseif (repa_on.and.mod(repl_x_freq,repa_freq).gt.0) then
          call wrndie(0,whoami,trim('REPLICA SWAP ATTEMPT FREQ. NOT A MULTIPLE OF REPA. FREQ.'))
        else ! OK
          call cv_common_rex_set_temp(repl_x_temp)
        endif
       endif ! repl_x_on
!
       if (repa_on.or.evolve_cv_on.or.repl_x_on) then ! decrease output
         string_noprint=(indxa(comlyn, comlen, 'NOPR').gt.0)
       endif
!--------------- DONE PARSING DYNAMICS OPTIONS -----
! print summary
!cccccccccccccccccc STRING METHOD OPTIONS cccccccccccccccccccccc
       if (qprint) then
        WRITE (info,'(2A)') &
     & whoami, ' STRING METHOD ENABLED.' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        if (evolve_cv_on) then
            WRITE (info,'(/,2A,/,2A,I7,A)') &
     & whoami, ' STRING EVOLUTION ENABLED.', &
     & whoami, ' WILL EVOLVE AFTER EVERY ', &
     & evolve_freq,' ITERATIONS.' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
            WRITE (info,'(2A,I7,A)') &
     & whoami, ' THE FIRST', evolve_nskip, &
     & ' ITERATIONS AFTER STRING EVOLUTION WILL NOT CONTRIBUTE TO AVERAGES.' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
! type of evolution
            i=0;
            if (evolve_expo_on) i=i+1
            if (evolve_bd_on) i=i+1
            if (evolve_aver_on) i=i+1
            if (evolve_smooth_on) i=i+1
!
            if (i.gt.1) then
             call wrndie(0,whoami,trim('MORE THAN ONE EVOLUTION SCHEME REQUESTED. WILL USE SMCV.'))
             evolve_expo_on=.false.
             evolve_aver_on=.false.
             evolve_smooth_on=.false.
             evolve_bd_on=.false.
            endif
!
            if (evolve_expo_on) then
               write(info,671) whoami, whoami, evolve_expo_mem ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 671 format(A,' CV EVOLUTION WILL BE OF THE FORM:',/, &
     & A,' Z(N+1)=A*Z(N)+(1-A)*<THETA>, A=',F9.5,'.')
            elseif (evolve_aver_on) then
               write(info,6710) whoami, whoami, num_ave_cv_samples ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6710 format(A,' CV EVOLUTION WILL BE OF THE FORM:',/, &
     &A,' Z(N+1)=AVERAGE_1^N(THETA).  INITIAL NUMBER OF SAMPLES IS ',I5,'.')
            elseif (evolve_smooth_on) then
               write(info,672) whoami, whoami, evolve_smooth_d ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 672 format(A,' WILL EVOLVE CV BY SMOOTHING MD TRAJECTORY',/, &
     & A,' USING FILTER WIDTH D=',F8.3)
            elseif (evolve_bd_on) then
               write(info,6720) whoami, whoami, evolve_step, evolve_bd_T ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6720 format(A,' WILL EVOLVE CV USING BD ADVANCEMENT',/, &
     & A,' AT T=',F8.3,' WITH STEP=',F11.5)
            else
               write(info,673) whoami, whoami, evolve_step ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 673 format(A,' WILL EVOLVE CV USING SMCV ADVANCEMENT ',/, &
     & A,' WITH STEP=',F11.5)
            endif ! evolve_expo
        endif ! evolve_cv_on
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if (repa_on) then
          WRITE (info,666) whoami, repa_freq ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 666 format(A,' WILL REPARAMETRIZE STRING AFTER EVERY ',I7, &
     & ' ITERATIONS.')
        endif ! repa_on
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if (hist_freq.gt.0) then ; write(info,667) whoami, hist_freq ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 667 format(A,' WILL SAVE CV VALUES AFTER EVERY ', I7, ' ITERATIONS.')
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if (restrained_on) then
            WRITE (info,'(2A)') &
     & whoami, ' WILL USE RESTRAINED DYNAMICS.' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
!
            if (steering_on) then
             write(info(1), '(2A)') &
             whoami, ' WILL NOT INITIALIZE COMPARISON CV SET TO SIMULATE STEERED DYNAMICS (SMD).'
             write(info(2), '(2A)') &
             whoami, ' USER MUST INITIALIZE MAIN SET TO FINAL CV VALUES'
             write(info(3), '(2A)') &
             whoami, ' AND COMPARISON SET TO INITIAL CV VALUES'
             write(info(4), '(2A,I7,A)') &
             whoami, ' STEERING SPEED IS GOVERNED BY "REEQ"(=',restrained_eq_steps,')'
             write(OUTU,'(A)') pack(info,info.ne.'');info='';
            endif
!
            if (unrestrained_on) then
             WRITE (info,'(2A)') &
     & whoami, ' WILL USE UNRESTRAINED (EXPLORATION) DYNAMICS.' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
             unrestrained_eq0=0
             WRITE (info,669) whoami, unrestrained_eq_steps ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 669 format(A,' WILL EQUILIBRATE UNDER CV RESTRAINTS FOR ', I7, ' STEPS.')
            endif ! unrestrained_on
!
            write(info,665) whoami, restrained_eq_steps ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
            restrained_eq0=0
 665 format(A, ' WILL ADJUST TO NEW RESTRAINTS OVER ', I7, ' STEPS.')
        endif ! restrained
!
!
        if (planar_on) then
            write (info,'(2A)') whoami, &
     & ' WILL RESTRAIN SYSTEM IN PLANE PERPENDICULAR TO PATH.' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if (stat_on) then
            write(info,668) whoami, stat_freq ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 668 format(A, ' WILL OUTPUT STRING STATISTICS EVERY ', &
     & I7, ' STEPS.')
        endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if (repl_x_on) then
            write(info,691) whoami, whoami, repl_x_freq, repl_x_temp ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 691 format(A, ' WILL ATTEMPT TO EXCHANGE NEIGHBORING REPLICAS ',/ &
     & A, ' ONCE IN EVERY ',I6,' ITERATIONS AT ',F8.3, ' K.')
        endif ! repl_x_on
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if (voronoi_hist_on) then
            write(info,670) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 670 format(A, ' WILL COMPUTE FREE ENERGY ALONG STRING ', &
     & 'USING VORONOI TESSELLATION.' )
         if (voronoi_allow_cross) then
          write(info,601) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          write(info,602) whoami, whoami, voronoi_update_freq ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          if (voronoi_nocross_ini.gt.0) then
           write(info, 603) whoami, whoami, voronoi_nocross_ini ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          endif
 601 format(A, ' WILL ALLOW REPLICAS TO CROSS BETWEEN V. CELLS.')
 602 format(A, ' WILL UPDATE CROSSING STATISTICS ONCE IN EVERY',/, &
     & A, I6, ' ITERATIONS.')
 603 format(A, ' WILL DISALLOW CROSSING DURING THE INITIAL ',/,A,I6, &
     & ' ITERATIONS.')
         endif
         if (restrained_on) then
          write(info,'(2A,/2A)') &
     & whoami, ' STRING DYNAMICS SHOULD BE USED WITH CAUTION', &
     & whoami, ' DURING VORONOI FE COMPUTATION.' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
         if (evolve_cv_on.or.repa_on) then
          if (.not.voronoi_allow_cross) then
           write(info,'(2A,/2A)') &
     & whoami, ' PERFORMING STRING EVOLUTION / REPARAMETRIZATION ', &
     & whoami, ' DURING VORONOI FE TESSELLATION IS AN EXTENSION ' , &
     & whoami, ' TO THE STRING METHOD BECAUSE THE M TENSOR IS NOT ', &
     & whoami, ' COMPUTED AT THE STRING IMAGE. YOU HAVE BEEN WARNED.' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          else
           write(info,'(2A,/2A,/2A)') &
     & whoami, ' STRING EVOLUTION AND REPARAMETRIZATION CANNOT', &
     & whoami, ' BE USED IF VORONOI CELL CROSSING IS ALLOWED.', &
     & whoami, ' VORONOI CELL CROSSING WILL BE OFF.' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
           voronoi_allow_cross=.false.
          endif ! voronoi_allow_cross
         endif ! evolve_cv_on
        endif ! voronoi_hist_on
       endif ! root writes
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! check cv values for NAN
!
       if (restrained_on) then
        if (any(cv%r(1:cv%num_cv,main).eq.anum)) then
         write(info(1),*)'MAIN CV SET APPEARS TO BE UNDEFINED ON GLOBAL RANK ',ME_GLOBAL,'. BEWARE.';call wrndie(0,whoami,trim(info(1)));
        elseif (any(cv%r(1:cv%num_cv,comp).eq.anum)) then
         write(info(1),*)'COMPARISON/OLD CV SET APPEARS TO BE UNDEFINED ON GLOBAL RANK ',ME_GLOBAL,'. BEWARE.';call wrndie(0,whoami,trim(info(1)));
        endif
       endif ! restrained on
!
! turn on string for dynamics
       smcv_on=.true.
! call dynamics parser
       call dynopt(comlyn, comlen)
!cccccc turn off string for regular dynamics ccccccc
       smcv_on=.false.
       repa_on=.false. ! turn off after dynamics because SM0K also uses this flag; therefore a subsequent minimization would call reparametrization
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'ADD '(1:4) )) then ! add CV
! call MPI_COMM_RANK(MPI_COMM_WORLD, i,ierror) ! aa
! write(600+ME_GLOBAL,*) i,ME_GLOBAL, MPI_COMM_LOCAL,
! & ME_LOCAL, SIZE_LOCAL
! call MPI_BARRIER(MPI_COMM_GLOBAL, ierror)
! stop
!
       call smcv_add(comlyn, comlen) ! this routine deals with the various CV
!
! (re)compute cv index limits (for parallelization) after each addition,
! because cv%num_cv changes
!
       if (SIZE_LOCAL.gt.0) then
        if (.not. cv_common_initialized) call cv_common_init() ! make sure cv%num_cv is defined
!
        j=ceiling(1.0d0*cv%num_cv/SIZE_LOCAL) ! max. number of CV assigned to slave node
        n=ceiling(1.0d0*cv%amap%last/SIZE_LOCAL) ! max. number of amap indices assigned to slave node
!
        do i=1,SIZE_LOCAL
         cv_send_displ(i)=min((i-1)*j,cv%num_cv-1) ! cannot exceed num_cv
         cv_send_count(i)=max(0,min(j,cv%num_cv-j*(i-1))) ! how many CV I will send to CPU i
! atom map partitioning (for parallel computation of M
!
         imap_displ(i)=min((i-1)*n,cv%amap%last-1)
         imap_count(i)=max(0,min(n,cv%amap%last-n*(i-1)))
        enddo
       endif ! SIZE
! IGNORE COMMENTS BELOW
! have to do some cheating: in order to communicate, namely, I assume that the (local) number of CV is the
! same on each processor (to that some cpus might be sending "zeros" -- cv%r that is outside of the used bounds)
! this also means that j*SIZE_LOCAL has to be less than or equal to than max_cv_common.
! basically, I only use cv_send_count(1)=j from the above array
! if (SIZE_LOCAL*j.gt.max_cv_common) then
! if (qroot) then
! if (ME_STRNG.eq.0)
! & write(info,'(2A,I5,1A,/,2A,I5,A)' ) whoami,
! ' CV STORAGE SPACE EXCEEDED. PARALLELIZATION REQUIRES ',
! & SIZE_LOCAL*j, ' ELEMENTS', whoami, ', BUT ONLY ', max_cv_common,
! & ' ARE ALLOCATED.'
! endif
! call wrndie(-4, whoami, ' CV STORAGE SPACE EXCEEDED.')
! endif
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccc
! this type will change when the auxiliary atom array is resized
! I could define the type in grad_init (only once would be required)
! but I don`t want to contaminate cv_common module (it would need to know sm_config)
       if (MPI_GRAD_TYPE.ne.MPI_DATATYPE_NULL) &
     & call mpi_type_free(MPI_GRAD_TYPE, ierror)
       if (MPI_GRAD_TYPE_.ne.MPI_DATATYPE_NULL) &
     & call mpi_type_free(MPI_GRAD_TYPE_, ierror)
!
       impi=6*cv%amap%last
       jmpi=1
       kmpi=6*cv%num_cv
       call mpi_type_vector(impi,jmpi,kmpi, & ! impi=6 because we are taking 3 gradients and both grad arrays (see cv_common)
     & mpifloat,MPI_GRAD_TYPE_,ierror)
!
! indexed version
! i=6*cv%amap%last
! allocate(temp1(i), temp2(i))
! temp1=(/ (1, j=1, i) /) ! block sizes
! temp2=(/ ( (j-1)*cv%num_cv, j=1, i) /) ! offsets from zero
! call mpi_type_indexed(i, temp1, temp2,
! & mpifloat, MPI_GRAD_TYPE_, ierror)
! deallocate(temp1, temp2)
!
       lb=0
       extent=sizeofreal
       call mpi_type_create_resized(MPI_GRAD_TYPE_,lb,extent, &
     & MPI_GRAD_TYPE, ierror)
       call mpi_type_commit(MPI_GRAD_TYPE, ierror)
!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'CLEA'(1:4) )) then ! initialize
       if (qprint) then ; write(info,6666) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6666 format(/A,' WILL REMOVE ALL CV, FRAMES, AND QUATERNIONS.')
       call quat_done()
       call cv_common_done()
       call frames_done() ! frames rely on cv%amap, so must also be initialized
       call cv_common_init()
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccc ADDITIONAL VORONOI OPTIONS (nowhere else to put them)
      elseif (( keyword(1:4).eq.'VORO'(1:4) )) then
!*****************************************************************************************
       if (indx(comlyn, comlen, 'VCUT', 4).gt.0) then
        voro_cut=gtrmf(comlyn, comlen, 'VCUT', zero)
! replica spec
        irep=gtrmi(comlyn, comlen, 'REP', -1)
        if (voro_cut.le.zero) then
          call wrndie(0,whoami,trim('VCUT MUST BE POSITIVE. NOT SET.'))
        else
         if (irep.lt.0.or.irep.ge.nstring) then
          if (qprint) then ; write(info, 6779) whoami, whoami, voro_cut ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
          call cv_common_voronoi_set_cutoff(voro_cut)
         else
          if (qprint) then ; write(info, 6780) whoami, irep, voro_cut ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
          if (mestring.eq.irep) call cv_common_voronoi_set_cutoff(voro_cut)
         endif ! irep
         if (qprint) then
          write(info,'(2A,/2A,F11.7,A)') &
     & whoami,' STRING WILL BE RESTRICTED TO STAY WITHIN THE WEIGHTED', &
     & whoami,' DISTANCE ',voro_cut,' OF THE CELL CENTERS DURING DYNAMICS.' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
!
 6779 format(A,' REPLICA NUMBER INVALID OR UNSPECIFIED.',/A,' WILL SET VORONOI TUBE CUTOFF ', &
     & '  TO ',F7.3,' ON ALL REPLICAS.')
 6780 format(A,' WILL SET VORONOI TUBE CUTOFF ON REPLICA ',I5,' TO ',F7.3,'.')
        endif ! voro_cut > 0
!
       endif ! VCUT
! process other voronoi commands
! get command
       if (comlen .ge. 4) then ! process additional commands
        keyword=nexta8(comlyn,comlen)
! voronoi map commands cccccccccccccccccccccccccccccccccccccccccccccccccc
        if (( keyword(1:4).eq.'VMAP'(1:4) )) then
         if (indxa(comlyn, comlen, 'CALC').gt.0) then
          if (qprint) then ; write(info,6010) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6010 format(A,' WILL CALCULATE VORONOI MAP FROM MAIN COORDINATES.')
          call smcv_voronoi_whereami(x,y,z,amass)
! put 'whereami' into the map
          if (qroot.and.SIZE_STRNG.gt.1) then
           call MPI_ALLGATHER(cv%voronoi_whereami, 1, mpiint, &
     & cv%voronoi_map, 1, mpiint, MPI_COMM_STRNG, ierror)
          else
           cv%voronoi_map(mestring+1)=cv%voronoi_whereami
          endif
          if (qslave) then
#if (KEY_INTEGER8==0)
          call PSND4(cv%voronoi_map,nstring) 
#endif
#if (KEY_INTEGER8==1)
          call PSND8(cv%voronoi_map,nstring) 
#endif

!
          endif
! print map cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         elseif ((indxa(comlyn, comlen, 'PRIN').gt.0) .or. (indxa(comlyn, comlen, 'WRIT').gt.0)) then
          ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
          call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: FNAME will be UPPER CASE
          if (qroot) then
           if (flen.gt.0) then
            if (qprint) then

             oldiol=iolev
             iolev=1

             call open_file(ifile, fname, 'FORMATTED', 'WRITE')
             write(info,6011) whoami, fname(1:flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
            endif
 6011 format(A,' WRITING VORONOI MAP TO FILE ',A,'.')
            call cv_common_print_voro_map(ifile)
            if (qprint) then
             call VCLOSE(ifile, 'KEEP', ierror)

             iolev=oldiol

            endif
           else
            call wrndie(0,whoami,trim('FILE NAME NOT SPECIFIED. NOTHING DONE.'))
           endif ! flen
          endif ! qroot
! read map ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         elseif (indxa(comlyn, comlen, 'READ').gt.0) then
          ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
          call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: FNAME will be UPPER CASE
          if (flen.GT.0) then
            if (qprint) then

             oldiol=iolev
             iolev=1

             call open_file(ifile, fname, 'FORMATTED', 'WRITE')
             write(info,6013) whoami, fname(1:flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
            endif
!
 6013 format(A,' READING VORONOI MAP FROM FILE ',A,'.')
            call cv_common_read_voro_map(ifile)
            if (qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ;

             iolev=oldiol

            endif
          else
            call wrndie(0,whoami,trim('FILE NAME NOT SPECIFIED. NOTHING DONE.'))
          endif ! flen
!ccc reset map cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         elseif (indxa(comlyn, comlen, 'CLEA').gt.0) then
          if (associated(cv%voronoi_map)) cv%voronoi_map=-ione
         endif ! 'CALC'
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        elseif (( keyword(1:4).eq.'READ'(1:4) )) then
! read "restart" file that contains (1) crossing_attempt (2) crossing_accepts (3) occupancy
         ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
         call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: FNAME will be UPPER CASE
         if (flen.GT.0) then
          if (qprint) then

           oldiol=iolev
           iolev=1

           call open_file(ifile, fname, 'FORMATTED', 'WRITE')
           write(info,6014) whoami, fname(1:flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          endif
 6014 format(A,' READING VORONOI CROSSING DATA FROM FILE ',A,'.')
          call cv_common_read_voro_data(ifile)
          if (qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ;

           iolev=oldiol

          endif
         else
          call wrndie(0,whoami,trim('FILE NAME NOT SPECIFIED. NOTHING DONE.'))
         endif ! flen
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        elseif ((( keyword(1:4).eq.'PRIN'(1:4) )) .or. (( keyword(1:4).eq.'WRIT'(1:4) ))) then
         ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
         call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: FNAME will be UPPER CASE
         if (flen.gt.0) then
           if (qprint) then

            oldiol=iolev
            iolev=1

            call open_file(ifile, fname, 'FORMATTED', 'WRITE')
            write(info,6015) whoami, fname(1:flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6015 format(A,' WRITING VORONOI CROSSING DATA TO FILE ',A,'.')
           endif
           call cv_common_print_voro_data(ifile)
           if (qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ;

            iolev=oldiol

           endif
         else
            call wrndie(0,whoami,trim('FILE NAME NOT SPECIFIED. NOTHING DONE.'))
         endif ! flen
        endif ! VMAP
       endif ! VCUT
!cccccccccccccccccccc FRAMES PARSER ccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'FRAM'(1:4) )) then ! frames parser
! get frames command
       keyword=nexta8(comlyn,comlen)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       if (( keyword(1:4).eq.'CLEA'(1:4) )) then ! initialize
        if (qprint) then ; write(info,6667) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6667 format(/A,' WILL REMOVE ALL LOCAL FRAMES.')
        call frames_done()
        call frames_init()
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif (( keyword(1:4).eq.'RESE'(1:4) )) then ! initialize
        if (qprint) then ; write(info,6650) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6650 format(/A,' WILL FORCE RECALCULATION OF FRAME AXES.')
        call frames_reset_calculate(.true.)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif (( keyword(1:4).eq.'FILL'(1:4) )) then ! compute frame axes values from current coordinates
!
        qcomp=(indxa(comlyn, comlen, 'COMP').gt.0) ! compute CV from comp set?
!
        if (frames%num_frames.lt.1) then
         call wrndie(0,whoami,trim('NO FRAMES DEFINED. NOTHING DONE.'))
        else
! first process special option: ALIGn
! frame axes will be calculated based on the main set, but `consistently` with the comparison set;
! specifically: the frame axes are permuted such that the rotation matrix associated with transforming one frame into another is the closest
! to the best-fit-RMSD rotation matrix
         if (indxa(comlyn, comlen, 'ALIG').gt.0) then
          if (any(xcomp.eq.anum)) then
           call wrndie(0,whoami,trim('MAIN X SET HAS UNDEFINED VALUES. NOTHING DONE.'))
          elseif (any(x.eq.anum)) then
           call wrndie(0,whoami,trim('COMPARISON X SET HAS UNDEFINED VALUES. NOTHING DONE.'))
          else
           if (qcomp) then
            if (qprint) then ; write(info,6651) whoami, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6651 format(/A,' WILL CALCULATE FRAME AXES FROM', &
     & ' COMPARISON COORDINATES USING', &
     & /A,' BEST-FIT ALIGNMENT WITH MAIN COORDINATES.')
            do i=1, frames%num_frames
             call frames_calc_align_comp( &
     & i,xcomp,ycomp,zcomp,x,y,z,amass,.true.)
            enddo
           else ! qcomp
            if (qprint) then ; write(info,6652) whoami, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6652 format(/A,' WILL CALCULATE FRAME AXES FROM', &
     & ' MAIN COORDINATES USING BEST-FIT', &
     & /A,' ALIGNMENT WITH COMPARISON COORDINATES.')
            do i=1, frames%num_frames
             call frames_calc_align_comp( &
     & i,x,y,z,xcomp,ycomp,zcomp,amass,.true.)
            enddo
           endif ! qcomp
           call frames_reset_calculate(.true.) ! make sure that next time frames_calc is called we recalculate axes (to be safe)
          endif ! xcomp
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! process regular fill
         elseif (qcomp) then
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
          if (any(xcomp.eq.anum)) then
           call wrndie(0,whoami,trim('COMPARISON X SET HAS UNDEFINED VALUES. NOTHING DONE.'))
          else
           if (qprint) then ; write(info,6656) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6656 format(/A,' WILL CALCULATE FRAME AXES FROM COMPARISON COORDINATES.')
           do i=1, frames%num_frames
            call frames_calc(i,xcomp,ycomp,zcomp,amass,.true.)
           enddo
           call frames_reset_calculate(.true.) ! make sure that next time frames_calc is called we recalculate axes (to be safe)
          endif ! xcomp.eq.anum
         else ! qcomp false -- use main coords
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
          if (any(x.eq.anum)) then
           call wrndie(0,whoami,trim('MAIN X SET HAS UNDEFINED VALUES. NOTHING DONE.'))
          else
           if (qprint) then ; write(info,6668) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6668 format(/A,' WILL CALCULATE FRAME AXES FROM MAIN COORDINATES.')
           do i=1, frames%num_frames
            call frames_calc(i,x,y,z,amass,.true.)
           enddo
           call frames_reset_calculate(.true.) ! make sure that next time frames_calc is called we recalculate axes (to be safe)
          endif ! x.eq.anum
         endif ! qcomp
        endif ! num_frames < 1
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif ((( keyword(1:4).eq.'PRIN'(1:4) )).or.(( keyword(1:4).eq.'WRIT'(1:4) ))) then
! can write both a local and a global file
! local is specified with 'ALL'; global is the default
        all=indxa(comlyn, comlen, 'ALL') ! all replicas print
! prepare file
!-----------------------------------------------------------------------------
        ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
        call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: FNAME will be UPPER CASE
!---------------------------------- OPEN FILE --------------------------------
        if (qroot) then
         if (all.gt.0.or.qprint) then

          oldiol=iolev
          iolev=0 ! trick to open file on all nodes

          if (flen.gt.0) then ; call open_file(ifile, fname, 'FORMATTED', 'WRITE') ; endif
          if (ifile .eq. -1) ifile=outu ! write to output stream
         endif
!---------------------------- assume file is open, write -------------------------
         if (qprint) then ; write(info,6669) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6669 format(/A,' WRITING LOCAL FRAME AXES.')
         if (all.eq.0) then ; call frames_print_global(ifile) ;
         else ; call frames_print_local(ifile) ; endif
         if (all.gt.0.or.qprint) then
          if (flen.gt.0) call VCLOSE(ifile, 'KEEP', ierror)

          iolev=oldiol

         endif
        endif ! qroot
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif (( keyword(1:4).eq.'READ'(1:4) )) then
! can read from both a local and a global file
! local is specified with 'ALL'; default is global
        all=indxa(comlyn, comlen, 'ALL') ! all replicas read
! prepare file
        ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
        call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: flen will be UPPER CASE
        if (qroot) then
!cccccccccccccccccccccccccccc OPEN FILE ccccccccccccccccccccccc
         if (all.gt.0.or.qprint) then

          oldiol=iolev
          iolev=0 ! open file on all processors

          if (flen.gt.0) then ; call open_file(ifile, fname, 'FORMATTED', 'READ') ; endif
         endif
         if (ifile .eq. -1) then
          Ifile=istrm ! read from input file

          call rdtitl(titleb,ntitlb,ifile,0) ! 0 = card format

         endif
!cccccccccccccccccc assume file is open, read ccccccccccccccccccc
         if (qprint) then ; write(info,6670) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6670 format(A,' READING LOCAL FRAME AXES.')
         if (all.gt.0) then ; call frames_read_local(ifile) ;
         else ; call frames_read_global(ifile) ; endif
         if (all.gt.0.or.qprint) then
          if (flen.gt.0) then ; call VCLOSE(ifile, 'KEEP', ierror) ; endif

          iolev=oldiol

         endif
        endif ! qroot
! send to slaves
        if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and. &
     & SIZE_LOCAL.gt.1) &
     & call MPI_BCAST(frames%r(:,:,:), frames%num_frames*9, &
     & mpifloat, 0, MPI_COMM_LOCAL, ierror)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif (( keyword(1:3).eq.'ADD'(1:3) )) then
        call smcv_frame_add(comlyn, comlen)
!
! (re)compute frame index limits (for parallelization) after each addition
!
        if (SIZE_LOCAL.gt.0) then
         if (.not.frames_initialized) call frames_init() ! make sure frames%num_frames is defined
!
         j=ceiling(1.0d0*frames%num_frames/SIZE_LOCAL) ! max. number of frames assigned to slave node
         n=ceiling(1.0d0*cv%amap%last/SIZE_LOCAL) ! max. number of amap indices assigned to slave node
!
         do i=1,SIZE_LOCAL
          fr_send_displ(i)=min((i-1)*j,frames%num_frames-1) ! cannot exceed num_cv
          fr_send_count(i)=max(0,min(j,frames%num_frames-j*(i-1))) ! how many CV I will send to CPU i
! atom map partitioning (for parallel computation of M
!
          imap_displ(i)=min((i-1)*n,cv%amap%last-1)
          imap_count(i)=max(0,min(n,cv%amap%last-n*(i-1)))
         enddo
        endif
!cccc !aa
!
! write(0,*) ME_LOCAL, fr_send_displ(ME_LOCAL+1),
! & fr_send_count(ME_LOCAL+1),frames%num_frames
! stop
!
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif (( keyword(1:4).eq.'LIST'(1:4) )) then ! list frames
        if (qprint) then ; write(info,6671) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6671 format(/A,' WILL LIST LOCAL FRAMES.')
       call frames_list() ! list local frames
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif (( keyword(1:4).eq.'ALIG'(1:4) )) then ! iteratively invert frame vectors (v -> -v)
! to guess the best alignment along string &
! (optional) to mimimize DIST(z,theta(x))
        min_rmsd=(indxa(comlyn, comlen, 'RMSD').gt.0) ! look for optimal RMSD alignment : i.e. minimize DIST(z,theta(x)) ?
!
        if (qprint) then ; write(info,6672) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6672 format(/A,' WILL ALIGN LOCAL FRAMES.')
        if (indxa(comlyn, comlen, 'VORO').gt.0) then
          if (qprint) then ; write(info,6673) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6673 format(A,' WILL FIT THE VORONOI MAP.')
          call frame_align_voro(x,y,z,amass)
        else
          if (min_rmsd.and.qprint) then ; write(info,6674) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6674 format(A,' WILL CHECK RMSD(Z,THETA[X]).')
          call frames_align_string(x,y,z,amass,min_rmsd) ! subroutine moved to sm_util to resolve dependency problems
        endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       else
            write(info(1),*)'UNRECOGNIZED SUBCOMMAND: ',keyword;call wrndie(0,whoami,trim(info(1)))
       endif ! frames parser
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'FILL'(1:4) )) then ! fill CV values from current coordinates
!
       qcomp=(indxa(comlyn, comlen, 'COMP').gt.0)
!
       if (qcomp) then
        if (any(xcomp.eq.anum)) then
         call wrndie(0,whoami,trim('COMPARISON X SET HAS UNDEFINED VALUES. NOTHING DONE.'))
        else
         if (qprint) then ; write(info,6657) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6657 format(/A,' WILL OBTAIN CV VALUES FROM COMPARISON COORDINATES.')
! check for column spec
         c1=sm_get_column(comlyn, comlen, qcoltag=.true., missing=-1)
         if (c1.gt.0) then
          if (qprint) then ; write(info,6661) whoami, c1 ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
          call smcv_fill(xcomp,ycomp,zcomp,amass,c1)
          call quat_reset_calculate(.true.)
          call frames_reset_calculate(.true.)
          call cv_common_unwrap_angles(c1) ! in case they are present
         else
          if (qprint) then ; write(info,6662) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
          call smcv_fill(xcomp,ycomp,zcomp,amass)
          call quat_reset_calculate(.true.)
          call frames_reset_calculate(.true.)
          call cv_common_unwrap_angles() ! in case they are present
         endif ! c1
        endif ! x.eq.anum
       else ! ~qcomp -- use main coirdinates
        if (any(x.eq.anum)) then
         call wrndie(0,whoami,trim('MAIN X SET HAS UNDEFINED VALUES. NOTHING DONE.'))
        else
         if (qprint) then ; write(info,6660) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6660 format(/A,' WILL OBTAIN CV VALUES FROM MAIN COORDINATES.')
! check for column spec
         c1=sm_get_column(comlyn, comlen, qcoltag=.true., missing=-1)
         if (c1.gt.0) then
          if (qprint) then ; write(info,6661) whoami, c1 ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6661 format(/A,' WILL FILL COLUMN ',I3,'.')
          call smcv_fill(x,y,z,amass,c1)
          call quat_reset_calculate(.true.)
          call frames_reset_calculate(.true.)
          call cv_common_unwrap_angles(c1) ! in case they are present
         else
          if (qprint) then ; write(info,6662) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6662 format(/A,' WILL FILL DEFAULT COLUMN.')
          call smcv_fill(x,y,z,amass)
          call quat_reset_calculate(.true.)
          call frames_reset_calculate(.true.)
          call cv_common_unwrap_angles() ! in case they are present
         endif ! c1
        endif ! x.eq.anum
       endif ! qcomp
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'TEST'(1:4) )) then !
       if (indxa(comlyn, comlen, 'GRAD').gt.0) then ! finite-difference gradient test
! check fd spec
        step=gtrmf(comlyn, comlen, 'STEP', finite_difference_d)
        if (qprint) then ; write(info, 7001) whoami,whoami,step,whoami,whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 7001 format(/A,' WILL TEST GRADIENTS USING FINITE DIFFERENCES', &
     & /A,' USING DX = DY = DZ = ',F15.9,'.', &
     & /A,' MAIN COORDINATE SET MUST BE DEFINED.', &
     & /A,' WILL OVERWRITE "MAIN", "ZCUR", AND "ZOLD" CV ARRAYS')
        if (any(x.eq.anum)) then
         call wrndie(0,whoami,trim('MAIN X SET HAS UNDEFINED VALUES. NOTHING DONE.'))
        else
         fd_error=>smcv_test_grad_fd(x,y,z,amass,step)
         if (qprint) then
          write(info,7002) whoami, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 7002 format(/A,' CV#, DGRAD_X_MAX, DGRAD_Y_MAX, DGRAD_Z_MAX', &
     & /A,' ==========================================')
          do i=1, cv%num_cv
         write(info,'(A," ",I5," ",3(F15.9," "))')whoami,i,fd_error(i,:) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
! write(info,*), i, fd_error(i,:)
          enddo
         endif ! qprint
! decide whether the test was passed
         zval=abs(maxval(fd_error))
         if (zval.lt.abs(step)*1d0) then
          write(info,7003) whoami, zval, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         else
          write(info,7004) whoami, zval, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          call wrndie(0,whoami,trim('FINITE DERIVATIVE TEST FAILED.'))
         endif ! report test result
 7003 format(/A, ' THE MAXIMUM GRADIENT ierror IS ',F15.9,', ', &
     & /A, ' WHICH IS SMALLER THAN STEP. TEST PASSED.')
 7004 format(/A, ' THE MAXIMUM GRADIENT ierror IS ',F15.9,', ', &
     & /A, ' WHICH IS NO SMALLER THAN STEP. TEST FAILED.')
         deallocate(fd_error) ! smcv_test_grad returns a pointer to an array of abs errors
        endif
       endif ! grad
!
       if (indxa(comlyn, comlen, 'PARA').gt.0) then ! parallel communication test
        if (qprint) then ; write(info, 7005) whoami,whoami,whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 7005 format(/A,' WILL COMPARE PARALLEL AND SERIAL CV COMPUTATION', &
     & /A,' MAIN COORDINATE SET MUST BE DEFINED.', &
     & /A,' WILL OVERWRITE "ZCUR" CV ARRAY')
        if (any(x.eq.anum)) then
         call wrndie(0,whoami,trim('MAIN X SET HAS UNDEFINED VALUES. NOTHING DONE.'))
        else
         fd_error=>smcv_test_parallel(x,y,z,amass) ! use the same array as above
         if (qprint) then
          write(info,7006) whoami, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 7006 format(/A,' CV#, DCV, DGRAD_X_MAX, DGRAD_Y_MAX, DGRAD_Z_MAX', &
     & /A,' ===============================================')
          do i=1, cv%num_cv
         write(info,'(A," ",I5," ",4(F15.9," "))')whoami,i,fd_error(i,:) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
! write(info,*), i, fd_error(i,:)
          enddo
         endif ! qprint
! decide whether the test was passed
         zval=abs(maxval(fd_error))
         if (zval.lt.parallel_tolerance) then
          write(info,7007) whoami, zval, whoami, parallel_tolerance ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         else
          write(info,7008) whoami, zval, whoami, parallel_tolerance ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          call wrndie(0,whoami,trim('PARALLEL COMPUTATION TEST FAILED.'))
         endif ! report test result
 7007 format(/A, ' THE MAXIMUM ierror IS ',E12.5,', ', &
     & /A, ' WHICH IS SMALLER THAN ',E12.5,'. TEST PASSED.')
 7008 format(/A, ' THE MAXIMUM ierror IS ',E12.5,', ', &
     & /A, ' WHICH IS NO SMALLER THAN ',E12.5,'. TEST FAILED.')
         deallocate(fd_error) ! smcv_test_grad returns a pointer to an array of abs errors
        endif
       endif ! para
!
       if (indxa(comlyn, comlen, 'MINV').gt.0) then ! finite-difference gradient test
        if (qprint) then ; write(info, 7010) whoami,whoami,whoami, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 7010 format(/A,' WILL COMPARE M TENSOR INVERSE COMPUTATION', &
     & /A,' USING LU DECOMPOSITION AND MULTIDIAGONAL', &
     & /A,' MATRIX INVERSION.' &
     & /A,' MAIN COORDINATE SET MUST BE DEFINED.')
        if (any(x.eq.anum)) then
         call wrndie(0,whoami,trim('MAIN X SET HAS UNDEFINED VALUES. NOTHING DONE.'))
        else
         zval=smcv_test_Minv(x,y,z,amass)
         if (zval.lt.parallel_tolerance) then
          write(info,7011) whoami, zval, whoami, parallel_tolerance ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         else
          write(info,7012) whoami, zval, whoami, parallel_tolerance ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          call wrndie(0,whoami,trim('M INVERSE TEST FAILED.'))
         endif ! report test result
!
 7011 format(/A, ' THE MAXIMUM DIFFERENCE IS ',E12.5,', ', &
     & /A, ' WHICH IS SMALLER THAN ',E12.5,'. TEST PASSED.')
 7012 format(/A, ' THE MAXIMUM DIFFERENCE IS ',E12.5,', ', &
     & /A, ' WHICH IS NO SMALLER THAN ',E12.5,'. TEST FAILED.')
        endif
       endif
! other tests will go below this line
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! specify parallel CV calculation options
      elseif (( keyword(1:4).eq.'PARA'(1:4) )) then
       do while (comlen .gt. 1)
        keyword=nexta8(comlyn,comlen)
        select case(keyword)
         case('QUAT');
          keyword=nexta8(comlyn,comlen)
          select case(keyword)
           case('YES','ON','TRUE','T','yes','on','true','t')
            keyword='ENABLED '; calc_qt_para=.true.
            if (qprint) then ; write(info,7009) whoami, 'QUATERNIONS', keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           case('NO','OFF','FALSE','F','no','off','false','f')
            keyword='DISABLED' ; calc_qt_para=.false.
            if (qprint) then ; write(info,7009) whoami, 'QUATERNIONS', keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           case default
            call wrndie(0,whoami,trim('UNKNOWN OPTION SPECIFIED FOR "QUAT"'))
          end select
         case('FRAM');
          keyword=nexta8(comlyn,comlen)
          select case(keyword)
           case('YES','ON','TRUE','T','yes','on','true','t')
            keyword='ENABLED '; calc_fr_para=.true.
            if (qprint) then ; write(info,7009) whoami, 'FRAMES', keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           case('NO','OFF','FALSE','F','no','off','false','f')
            keyword='DISABLED' ; calc_fr_para=.false.
            if (qprint) then ; write(info,7009) whoami, 'FRAMES', keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           case default
            call wrndie(0,whoami,trim('UNKNOWN OPTION SPECIFIED FOR "FRAM"'))
          end select
         case('COLV');
          keyword=nexta8(comlyn,comlen)
          select case(keyword)
           case('YES','ON','TRUE','T','yes','on','true','t')
            keyword='ENABLED '; calc_cv_para=.true.
            if (qprint) then ; write(info,7009) whoami, 'CV', keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           case('NO','OFF','FALSE','F','no','off','false','f')
            keyword='DISABLED' ; calc_cv_para=.false.
            if (qprint) then ; write(info,7009) whoami, 'CV', keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           case default
            call wrndie(0,whoami,trim('UNKNOWN OPTION SPECIFIED FOR "COLV"'))
          end select
         case('MMAT');
          keyword=nexta8(comlyn,comlen)
          select case(keyword)
           case('YES','ON','TRUE','T','yes','on','true','t')
            keyword='ENABLED '; calc_Mtensor_para=.true.
            if (qprint) then ; write(info,7009) whoami, 'M TENSOR', keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           case('NO','OFF','FALSE','F','no','off','false','f')
            keyword='DISABLED' ; calc_Mtensor_para=.false.
            if (qprint) then ; write(info,7009) whoami, 'M TENSOR', keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           case default
            call wrndie(0,whoami,trim('UNKNOWN OPTION SPECIFIED FOR "MMAT"'))
          end select
         case('VORO');
          keyword=nexta8(comlyn,comlen)
          select case(keyword)
           case('YES','ON','TRUE','T','yes','on','true','t')
            keyword='ENABLED '; calc_voronoi_para=.true.
            if (qprint) then ; write(info,7009) whoami, 'VORONOI NORM', keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           case('NO','OFF','FALSE','F','no','off','false','f')
            keyword='DISABLED' ; calc_voronoi_para=.false.
            if (qprint) then ; write(info,7009) whoami, 'VORONOI NORM', keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           case default
            call wrndie(0,whoami,trim('UNKNOWN OPTION SPECIFIED FOR "VORO"'))
          end select
         case default
          call wrndie(0,whoami,trim('UNKNOWN OPTION SPECIFIED FOR "PARA"'))
        end select
       enddo ! comlen
 7009 format(/A, ' PARALLEL COMPUTATION OF ',A,' ',A)
!
      elseif (( keyword(1:4).eq.'MINV'(1:4) )) then
       keyword=nexta8(comlyn,comlen)
       select case(keyword)
        case('LU','lu')
          keyword='LU'; inverse_LU=.true.
          if (qprint) then ; write(info,7013) whoami, keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
        case('DIAG','diag')
          keyword='MULTDIAG' ; inverse_LU=.false.
          if (qprint) then ; write(info,7013) whoami, keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
        case default
          call wrndie(0,whoami,trim('UNKNOWN MATRIX INVERSION OPTION SPECIFIED.'))
       end select
 7013 format(/A, ' MATRIX INVERSION WILL USE ',A,' ROUTINES.')
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:3).eq.'FIX'(1:3) )) then ! tell cv_posi about fixed "virtual" replicas
       if (qprint) then ; write(info, 6665) whoami, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
       fixed_bc_0=(indxa(comlyn, comlen, 'FIRS').gt.0)
       fixed_bc_1=(indxa(comlyn, comlen, 'LAST').gt.0)
       if (fixed_bc_0) then
         fixbc=' '
         flen=1
       else
         fixbc=' NOT '
         flen=5
       endif
       if (qprint) then ; write(info,6663) whoami, fixbc(1:flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
       if (fixed_bc_1) then
         fixbc=' '
         flen=1
       else
         fixbc=' NOT '
         flen=5
       endif
       if (qprint) then ; write(info,6664) whoami, fixbc(1:flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
       call cv_common_set_bc(fixed_bc_0, fixed_bc_1)
 6663 format(/A,' FIRST REPLICA OF STRING WILL',A,'BE FIXED.')
 6664 format(A,' LAST REPLICA OF STRING WILL',A,'BE FIXED.'/)
 6665 format(A,' WARNING: SETTING BC REQUIRES REINITIALIZATION.',/, &
     & A,' ALL CV DATA WILL BE ERASED.')
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif ((( keyword(1:4).eq.'PRIN'(1:4) )) .or. (( keyword(1:4).eq.'WRIT'(1:4) ))) then
! can write both a local and a global file
! local is specified with 'ALL'; global is the default
       all=indxa(comlyn, comlen, 'ALL') ! all replicas print
! prepare file
!-----------------------------------------------------------------------------
       ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
       call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: FNAME will be UPPER CASE
!---------------------------------- OPEN FILE --------------------------------
       if (qroot) then
        if (all.gt.0.or.qprint) then

         oldiol=iolev
         iolev=0 ! trick to open file on all nodes

         if (flen.gt.0) then ; call open_file(ifile, fname, 'FORMATTED', 'WRITE') ; endif
        endif
        if (ifile .eq. -1) ifile=outu ! write to output stream
!---------------------------- assume file is open, write -------------------------
! check for column spec
        c1=sm_get_column(comlyn, comlen, qcoltag=.true., missing=-1)
        if (c1.gt.0) then
         if (qprint) then ; write(info,6679) whoami, c1 ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6679 format(/A,' WRITING COORDINATES FROM COLUMN ',I3)
         if (all.eq.0) then ; call cv_common_print_global(ifile, c1) ;
         else ; call cv_common_print_local(ifile,c1) ; endif
        else
         if (qprint) then ; write(info,6689) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6689 format(/A,' WRITING COORDINATES FROM DEFAULT COLUMN.')
         if (all.eq.0) then ; call cv_common_print_global(ifile) ;
         else ; call cv_common_print_local(ifile) ; endif
        endif ! c1
        if (all.gt.0.or.qprint) then
         if (flen.gt.0) then ; call VCLOSE(ifile, 'KEEP', ierror) ; endif

         iolev=oldiol

        endif
       endif ! qroot
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'READ'(1:4) )) then
! can read from both a local and a global file
! local is specified with 'ALL'; default is global
! can also read a specific column: specify SCOL x
       all=indxa(comlyn, comlen, 'ALL') ! all replicas read
       if (indx(comlyn, comlen, 'SCOL', 4).gt.0) then
        scol=gtrmi(comlyn, comlen, 'SCOL', 0)
        if (scol.ge.1) then ! need to know total number of replicas in CV file
         if (all.gt.0) then
          call wrndie(0,whoami,trim('ALL AND SCOL CANNOT BOTH BE SPECIFIED.'))
          return
         endif
         totcol=gtrmi(comlyn, comlen, 'TCOL', 0)
         if (totcol.le.0) then
          call wrndie(0,whoami,trim('MUST PROVIDE TOTAL NUMBER OF COLUMNS IN CV DATA FILE.'))
          return
         endif
        else ! scol.ge.1
         call wrndie(0,whoami,trim('SCOL MUST BE A POSITIVE INTEGER.'))
         return
        endif
       endif ! SCOL present
!
! prepare file
       ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
       call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: flen will be UPPER CASE
! check for column spec
       c1=sm_get_column(comlyn, comlen, qcoltag=.true., missing=-1)
       if (qroot) then
!cccccccccccccccccccccccccccc OPEN FILE ccccccccccccccccccccccc
        if (all.gt.0.or.qprint) then

         oldiol=iolev
         iolev=0 ! open file on all processors

         if (flen.gt.0) then ; call open_file(ifile, fname, 'FORMATTED', 'READ') ; endif
        endif
        if(ifile .eq. -1) then
         ifile=istrm ! read from input file

         call rdtitl(titleb,ntitlb,ifile,0) ! 0 = card format

        endif
!cccccccccccccccccc assume file is open, read ccccccccccccccccccc
        if (c1.gt.0) then ! column spec
         if (qprint) then ; write(info,6699) whoami, c1 ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6699 format(A,' READING COORDINATES INTO COLUMN ',I3)
         if (all.gt.0) then ; call cv_common_read_local(ifile, c1) ;
         elseif (scol.ge.1) then
          call cv_common_read_local_from_global(ifile, totcol, scol, c1)
         else; call cv_common_read_global(ifile,c1) ; endif
        else
         if (qprint) then ; write(info,6709) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6709 format(A,' READING COORDINATES INTO DEFAULT COLUMN.')
         if (all.gt.0) then ; call cv_common_read_local(ifile) ;
         elseif (scol.ge.1) then
          call cv_common_read_local_from_global(ifile, totcol, scol)
         else ; call cv_common_read_global(ifile) ; endif
        endif
        if (all.gt.0.or.qprint) then
         if (flen.gt.0) then ; call VCLOSE(ifile, 'KEEP', ierror) ; endif

         iolev=oldiol

        endif
       endif ! qroot
!
! broadcast to slaves
       if (c1.lt.0) c1=main ! guess what the "default column" is
       if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1) then
! call MPI_BCAST(cv%r(1,c1), cv%num_cv, mpifloat,
! & 0, MPI_COMM_LOCAL, ierror)
        call PSND8(cv%r(1,c1),cv%num_cv)
! broadcast BC
        if (cv_common_fixed_0_bc.eq.1) then

#if (KEY_SINGLE==1)
         call PSND4(cv%r_bc_0,cv%num_cv) 
#endif
#if (KEY_SINGLE==0)
         call PSND8(cv%r_bc_0,cv%num_cv) 
#endif



        endif
        if (cv_common_fixed_1_bc.eq.1) then

#if (KEY_SINGLE==1)
         call PSND4(cv%r_bc_1,cv%num_cv) 
#endif
#if (KEY_SINGLE==0)
         call PSND8(cv%r_bc_1,cv%num_cv) 
#endif



        endif
       endif
! unwrap angles if present
       call cv_common_unwrap_angles(c1)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'SWAP'(1:4) )) then ! swap two columns
! read column spec
        c1=sm_get_column(comlyn, comlen, qcoltag=.false., missing=-1)
        c2=sm_get_column(comlyn, comlen, qcoltag=.false., missing=-1)
        if (qprint) then ; write(info,6729) whoami, c1, c2 ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6729 format(/A,' WILL SWAP COLUMNS ',I3,' AND ',I3,' ')
        call cv_common_swap(c1,c2)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'COPY'(1:4) )) then ! copy form c1 to c2
! read column spec
        c1=sm_get_column(comlyn, comlen, qcoltag=.false., missing=-1)
        c2=sm_get_column(comlyn, comlen, qcoltag=.false., missing=-1)
        if (qprint) then ; write(info,6739) whoami, c1, c2 ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6739 format(/A,' WILL COPY COLUMN ',I3,' TO ',I3,' ')
        call cv_common_copy(c1,c2)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'HIST'(1:4) )) then ! parse history commands
        keyword=nexta8(comlyn,comlen)
        if (( keyword(1:3).eq.'ADD'(1:3) )) then ! save current CV values to history
         if (any(x.eq.anum)) then
          call wrndie(0,whoami,trim('MAIN X SET HAS UNDEFINED VALUES. NOTHING DONE.'))
         else
          if (qprint) then ; write(info,674) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 674 format(A,' WILL ADD CV VALUES FROM MAIN COOR. SET INTO HISTORY.')
! last argument tells routine to add the calculated cv/derivative values to the history
          call smcv_add_hist(x,y,z,amass,.true.)
         endif
        elseif ((( keyword(1:4).eq.'PRIN'(1:4) )).or.(( keyword(1:4).eq.'WRIT'(1:4) ))) then ! print history
! can write both a local and a global file
! local is specified with 'ALL'; global is the default
         all=indxa(comlyn, comlen, 'ALL') ! all replicas print
! prepare file
!-----------------------------------------------------------------------------
         ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
         call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: FNAME will be UPPER CASE
! check for other spec
         nskip=gtrmi(comlyn, comlen, 'SKIP', 0) ! number of entries to skip
!---------------------------------- OPEN FILE --------------------------------
         if (qroot) then

          oldiol=iolev
          if (all.gt.0) iolev=0 ! trick to open file on all nodes

!
          if (flen.gt.0) then ; call open_file(ifile, fname, 'FORMATTED', 'WRITE') ; endif
          if (ifile .eq. -1) ifile=outu ! write to output stream
!---------------------------- assume file is open, write -------------------------
          if (qprint) then ; write(info,676) whoami, nskip ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 676 format(A,' WRITING CV HISTORY. SKIPPING ',I5,' ENTRIES.')
          if (all.eq.0) then;call cv_common_print_hist_global(ifile,nskip)
          else ; call cv_common_print_hist_local(ifile,nskip) ; endif
          if (flen.gt.0) then ; call VCLOSE(ifile, 'KEEP', ierror) ; endif

          iolev=oldiol

         endif ! qroot
        elseif (( keyword(1:4).eq.'SMOO'(1:4) )) then ! smooth history
! look for other spec.
         delta=gtrmi(comlyn, comlen, 'DELT', 10) ! filter width
         nskip=gtrmi(comlyn, comlen, 'SKIP', 0) ! number of entries to skip
!
         if (qprint) then ; write(info,675) whoami, delta, nskip ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 675 format(A,' SMOOTHING CV HISTORY. FILTER WIDTH =',I5,'.', &
     & / ,' SKIPPING ',I5,' ENTRIES.')
         call cv_common_smooth_hist(delta,nskip)
        elseif (( keyword(1:4).eq.'EXPO'(1:4) )) then ! CONVOLUTION W/ EXPONENTIAL
! look for other spec.
         expo_memory=gtrmf(comlyn, comlen, 'MEMO', 0.999d0) ! memory in the exp. conv. kernel
         nskip=gtrmi(comlyn, comlen, 'SKIP', 0) ! number of entries to skip
!
         if (qprint) then ; write(info,701) whoami, expo_memory, nskip ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 701 format(A,' EVOLVING CV: Z(N+1)=A*Z(N)+(1-A)*<THETA>, A=',F7.3,'.',&
     & / ,' SKIPPING ',I5,' ENTRIES.')
         call cv_common_evolve_expo(expo_memory,nskip)
        endif
! done parsing 'HIST'
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'EVOL'(1:4) )) then ! evolve string using average force (SMCV)
! look for other spec: dt and
        step=gtrmf(comlyn, comlen, 'STEP', 0.0d0) ! evolution step
        if (qprint) then
         if (step.eq.0.0d0) then
          call wrndie(0,whoami,trim('CV EVOLUTION STEP ZERO OR UNSPECIFIED.'))
         endif
         write(info,677) whoami, step
 677 format(A,' EVOLVING CV USING AVERAGE FORCE. STEP =',F7.3,'.')
        endif ! qprint
        call cv_common_evolve_smcv(step)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:3).eq.'SET'(1:3) )) then ! modify k,w,g,dt
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! first check for global options: num_ave_samples
        if (indx(comlyn, comlen, 'NAVE', 4).gt.0) then
          num_ave_samples=gtrmi(comlyn, comlen, 'NAVE', -1)
          if (num_ave_samples.gt.0) then
           call cv_common_set_ave_samples(num_ave_samples)
           if (qprint) then ; write(info,6748) whoami, num_ave_samples ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6748 format(A,' SETTING NUMBER OF SAMPLES IN THE AVERAGE SET TO ',I7)
          else
           if (qprint) then ; write(info,6749) whoami, num_ave_samples; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6749 format(A,' INVALID NUMBER OF SAMPLES SPECIFIED: ',I7)
          endif
        endif
! set k parallel to path (for off-path dynamics)
        if (indx(comlyn, comlen, 'KPAR', 4).gt.0) then
          k=gtrmf(comlyn, comlen, 'KPAR', -1d0)
          if (k.ge.0d0) then
           call cv_common_set_kpara(k)
           if (qprint) then ; write(info,6756) whoami, k; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6756 format(A,' SETTING PARALLEL FORCE CONSTANT TO ',F11.5)
          else
           if (qprint) then ; write(info,6757) whoami, k; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6757 format(A,' INVALID FORCE CONSTANT SPECIFIED: ',F11.5)
          endif
        endif ! kpara
! set k perpendicular to path (for off-path dynamics)
        if (indx(comlyn, comlen, 'KPRP', 4).gt.0) then
          k=gtrmf(comlyn, comlen, 'KPRP', -1d0)
          if (k.ge.0d0) then
           call cv_common_set_kperp(k)
           if (qprint) then ; write(info,6746) whoami, k; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6746 format(A,' SETTING PERPENDICULAR FORCE CONSTANT TO ',F11.5)
          else
           if (qprint) then ; write(info,6747) whoami, k; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6747 format(A,' INVALID FORCE CONSTANT SPECIFIED: ',F11.5)
          endif
        endif ! kperp
!
! to set k,w,g can specify atom index, or ' ALL ' to apply to all CV
! process CV selection
        ind=gtrmi(comlyn, comlen, 'IND', 0)
        all=indxa(comlyn, comlen, 'ALL')
        if (all.gt.0) then ! will loop over all cv
         ibeg=1
         iend=cv%num_cv
         if (qprint) then ; write(info,6750) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6750 format(A,' ALL CV INDICES SELECTED.')
        elseif (ind.gt.0.and.ind.le.cv%num_cv) then
         ibeg=ind
         iend=ind
         if (qprint) then ; write(info,6751) whoami, ind ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6751 format(A,' CV INDEX ',I5,' SELECTED.')
        else ! no indices specified
         call wrndie(0,whoami,trim(' INVALID CV INDEX SPECIFIED'))
         ibeg=0
         iend=-1
        endif
!
        if (iend.gt.0) then ! skip this for invalid indices
         if (indx(comlyn, comlen, 'FORC', 4).gt.0) then
          k=gtrmf(comlyn, comlen, 'FORC', 0.0d0)
          if (qprint) then ; write(info,6752) whoami, k ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6752 format(A,' WILL SET K TO ',F11.5,'.')
          do i=ibeg,iend
           call cv_common_set_k(i,k)
          enddo
         endif
!
         if (indx(comlyn, comlen, 'GAMM', 4).gt.0) then
          gam=gtrmf(comlyn, comlen, 'GAMM', 1.0d0)
          if (qprint) then ; write(info,6753) whoami, gam ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6753 format(A,' WILL SET GAMMA TO ',F7.3,'.')
          do i=ibeg,iend
           call cv_common_set_g(i,gam)
          enddo
         endif
!
         if (indx(comlyn, comlen, 'WEIG', 4).gt.0) then ! weighting by a real
          w=gtrmf(comlyn, comlen, 'WEIG', -1.0d0)
          if (qprint) then ; write(info,6755) whoami,w ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6755 format(A,' WILL SET WEIGHT TO ',F7.3,'.')
          do i=ibeg,iend
           call cv_common_set_w(i,w)
          enddo
         endif
!
         if (indx(comlyn, comlen, 'ZVAL', 4).gt.0) then ! weighting by a real
          zval=gtrmf(comlyn, comlen, 'ZVAL', -1.0d0)
! check replica spec
          irep=gtrmi(comlyn, comlen, 'REP', -1)
          if (irep.lt.0.or.irep.ge.nstring) then
           call wrndie(0,whoami,trim('REPLICA NUMBER INVALID OR UNSPECIFIED.'))
          else
! check column spec
           c1=sm_get_column(comlyn, comlen, qcoltag=.true., missing=-1)
           if (c1.gt.0) then
            if (qprint) then
             write(info,6774) whoami, irep,c1, zval
             write(OUTU,'(A)') pack(info,info.ne.'');info='';
            endif
 6774 format(A,' WILL SET REPLICA ',I5,' CV VALUE IN COLUMN ', &
     & I3, ' TO ',F7.3,'.')
            if (mestring.eq.irep) then ;do i=ibeg,iend
                                       call cv_common_set_r(i,zval,c1)
                                      enddo; endif
           else
            if (qprint) then ; write(info,6773) whoami, irep, zval ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6773 format(A,' WILL SET REPLICA ',I5,' CV VALUE IN DEFAULT COLUMN TO '&
     & ,F7.3,'.')
            if (mestring.eq.irep) then ;do i=ibeg,iend
                                       call cv_common_set_r(i,zval)
                                      enddo; endif
           endif ! colspec
          endif ! irep
         endif ! zval
!
        endif ! iend.gt.0
! done with 'SET' parsing
!cccccccccccccccccccccccccccccccccccc M matrix cccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'MMAT'(1:4) )) then
        if (indxa(comlyn, comlen, 'CALC').gt.0) then
          if (qprint) then ; write(info,6854) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6854 format(A,' COMPUTING INSTANTANEOUS M(X) FROM ATOMIC COORDINATES.')
          call smcv_compute_M(x,y,z,amass,.true.) ! compute M and M inverse
! print
        elseif ((indxa(comlyn, comlen, 'PRIN').gt.0) .or. (indxa(comlyn, comlen, 'WRIT').gt.0)) then
! if running in parallel, combine partial M entries
          if (qslave) then
! call MPI_ALLREDUCE(cv%M(1,1,1),cv%M(1,1,2),max_cv_common*cv%num_cv, ! will broadcast all rows, but only num_cv columns
! & mpifloat, MPI_SUM, MPI_COMM_LOCAL, ierror)
          else ! qslave
! cv%M(1:cv%num_cv,1:cv%num_cv,2)=cv%M(1:cv%num_cv,1:cv%num_cv,1)
          endif ! qslave
! check for inverse spec (inverse stored in 3)
          if (indxa(comlyn, comlen, 'INV').gt.0) then
            ind=3
            keyword='INVERSE '; klen=8
            call cv_common_compute_Minv(inverse_LU)
          else
            ind=4 ! long-term average
            keyword=' '; klen=0
          endif
!
          ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
          call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: FNAME will be UPPER CASE
          if (qroot) then
           if (flen.gt.0) then
            if (qprint) then

             oldiol=iolev
             iolev=1

             call open_file(ifile, fname, 'FORMATTED', 'WRITE')
             write(info,6859) whoami, keyword(1:klen), fname(1:flen)
             write(OUTU,'(A)') pack(info,info.ne.'');info='';
            endif
 6859 format(A,' WRITING M TENSOR (LONG-TERM AVERAGE) ',A,'TO FILE ',A,'.')
            call cv_common_print_M_global(ifile, IND=ind)
            if (qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ;

             iolev=oldiol

            endif
           else
            call wrndie(0,whoami,trim('FILE NAME NOT SPECIFIED. NOTHING DONE.'))
           endif ! flen
          endif ! qroot
! read cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        elseif (indxa(comlyn, comlen, 'READ').gt.0) then
! check for inverse spec (inverse stored in 3)
          if (indxa(comlyn, comlen, 'INV').gt.0) then
            ind=3
            keyword='INVERSE '; klen=8
          else
            ind=4 ! long-term average
            keyword=' '; klen=0
          endif
!
          ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
          call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: FNAME will be UPPER CASE
          if (flen.gt.0) then
            if (qprint) then

             oldiol=iolev
             iolev=1

             call open_file(ifile, fname, 'FORMATTED', 'WRITE')
             write(info,6858) whoami, keyword(1:klen), fname(1:flen)
             write(OUTU,'(A)') pack(info,info.ne.'');info='';
            endif
 6858 format(A,' READING M TENSOR (LONG- & SHORT-TERM AVERAGE) ',A,'FROM FILE ',A,'.')
            call cv_common_read_M_global(ifile, ind)
            if (qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ;

             iolev=oldiol

            endif
          else
            call wrndie(0,whoami,trim('FILE NAME NOT SPECIFIED. NOTHING DONE.'))
          endif ! flen
! change calculation algorithm ccccccccccccccccccccccccccccccccccccc
        elseif (indxa(comlyn, comlen, 'FAST').gt.0) then
          keyword=nexta8(comlyn,comlen)
          select case(keyword)
           case('YES','ON','TRUE','T','yes','on','true','t')
            keyword='ENABLED '; calc_Mtensor_fast=.true.
            if (qprint) then ; write(info,7014) whoami, keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           case('NO','OFF','FALSE','F','no','off','false','f')
            keyword='DISABLED' ; calc_Mtensor_fast=.false.
            if (qprint) then ; write(info,7014) whoami, keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           case default
            call wrndie(0,whoami,trim('UNKNOWN OPTION SPECIFIED FOR "FAST"'))
          end select
 7014 format(/A,' SPARSE MATRIX ROUTINE FOR M TENSOR COMPUTATION ',A)
        endif
!cccccccccccccccccccccccccccccccccccc CV WEIGHTS cccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'WEIG'(1:4) )) then
        if (indxa(comlyn, comlen, 'CALC').gt.0) then
          if (qprint) then ; write(info,6754) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6754 format(A,' COMPUTING CV WEIGHTS FROM METRIC TENSOR M(X).')
          call smcv_compute_wgt(x,y,z,amass)
! print
        elseif ((indxa(comlyn, comlen, 'PRIN').gt.0).or.(indxa(comlyn, comlen, 'WRIT').gt.0)) then
! process output options
          ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
          call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: FNAME will be UPPER CASE
          if (qprint) then
            if (flen.gt.0) then
             write(info,6759) whoami, fname(1:flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6759 format(A,' WRITING CV WEIGHTS TO FILE ',A,'.')

             oldiol=iolev
             iolev=1

             call open_file(ifile, fname, 'FORMATTED', 'WRITE')
            else
             if (ifile .eq. -1) ifile=outu ! write to output stream
             write(info,6758) whoami, ifile; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6758 format(A,' WRITING CV WEIGHTS TO UNIT ',I5)
            endif
            call cv_common_print_wgt(ifile) ! only root node writes
            if (flen.gt.0) then ; call VCLOSE(ifile, 'KEEP', ierror) ;

             iolev=oldiol

            endif
          endif ! qprint
! read
        elseif (indxa(comlyn, comlen, 'READ').gt.0) then
! prepare file
          ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
          call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: flen will be UPPER CASE
          if (qprint) then
           if (flen.gt.0) then
            write(info,6761) whoami, fname(1:flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6761 format(A,' READING CV WEIGHTS FROM FILE ',A,'.')

            oldiol=iolev
            iolev=1

            call open_file(ifile, fname, 'FORMATTED', 'READ')
           else
            if (ifile .eq. -1) ifile=istrm ! read from input
            write(info,6760) whoami, ifile ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6760 format(A,' READING CV WEIGHTS FROM UNIT ',I5)

             if (ifile.eq.istrm) call rdtitl(titleb,ntitlb,ifile,0) ! 0 = card format

           endif ! flen
          endif ! qprint
          call cv_common_read_wgt(ifile) ! root and slave nodes enter
          if (qprint.and.flen.gt.0) then ; call VCLOSE(ifile, 'KEEP', ierror) ;

           iolev=oldiol

          endif
        endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'KPAR'(1:4) )) then
       if (indx(comlyn, comlen, 'SET', 3).gt.0) then
         k=gtrmf(comlyn, comlen, 'SET', -1d0)
         if (k.ge.0d0) then
          call cv_common_set_kpara(k)
          if (qprint) then ; write(info,6763) whoami, k ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6763 format(A,' SETTING PARALLEL FORCE CONSTANT TO ',F11.5)
         else
          if (qprint) then ; write(info,6764) whoami, k ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6764 format(A,' INVALID FORCE CONSTANT SPECIFIED: ',F11.5)
         endif
       endif ! set
!
       if (indxa(comlyn, comlen, 'CALC').gt.0) then
        if (qprint) then ; write(info, 6765) whoami, whoami, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6765 format(/A, ' COMPUTING FORCE CONSTANTS FOR RESTRAINED ',/, &
     & A, ' DYNAMICS BY SCALING KPAR WITH CV WEIGHTS.',/, &
     & A, ' OVERWRITING PREVIOUSLY DEFINED FORCE CONSTANTS.')
        call cv_common_compute_k()
       endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'TANV'(1:4) )) then ! print/read/compute tangent to path
       if (indxa(comlyn, comlen, 'CALC').gt.0) then ! calculate
        if (qprint) then ; write(info,6766) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6766 format(/A,' WILL COMPUTE TANGENT TO PATH.')
        call cv_common_compute_dr()
!
       elseif (indxa(comlyn, comlen, 'READ').gt.0) then ! read from file
! prepare file
        ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
        call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: flen will be UPPER CASE
!cccccccccccccccccccccccccccc OPEN FILE ccccccccccccccccccccccc
        if (qprint) then
         if (flen.gt.0) then ; call open_file(ifile, fname, 'FORMATTED', 'READ') ;

          oldiol=iolev
          iolev=1

         endif
         if(ifile .eq. -1) then
          ifile=istrm ! read from input file

          call rdtitl(titleb,ntitlb,ifile,0) ! 0 = card format

         endif ! ifile
!cccccccccccccccccc assume file is open, read ccccccccccccccccccc
         write(info,6767) whoami
 6767 format(A,' READING VECTORS TANGENT TO PATH.')
        endif ! qprint
!
        call cv_common_read_dr(ifile) ! roots and slaves
!
        if (qprint.and.flen.gt.0) then ; call VCLOSE(ifile, 'KEEP', ierror) ;

         iolev=oldiol

        endif
!
       elseif ((indxa(comlyn, comlen, 'PRIN').gt.0).or.(indxa(comlyn, comlen, 'WRIT').gt.0)) then ! print to file
! prepare file
        ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
        call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: FNAME will be UPPER CASE
        if (qprint) then
!---------------------------------- OPEN FILE --------------------------------
         if (flen.gt.0) then ; call open_file(ifile, fname, 'FORMATTED', 'WRITE')

          oldiol=iolev
          iolev=1

         endif
         if (ifile .eq. -1) ifile=outu ! write to output stream
!---------------------------- assume file is open, write -------------------------
         write(info,6768) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6768 format(/A,' WRITING VECTORS TANGENT TO PATH.')
        endif ! qprint
        if (qroot) call cv_common_print_dr(ifile)
        if (qprint.and.flen.gt.0) then ; call VCLOSE(ifile, 'KEEP', ierror)

         iolev=oldiol

        endif
       endif ! TANV
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'LIST'(1:4) )) then ! list CV
       if (qprint) then ; write(info,6762) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6762 format(/A,' WILL LIST CV.')
       call smcv_list() ! this routine deals with the various CV
! write(0,*) 'ME_LOCAL: ',ME_LOCAL, 'SIZE_LOCAL:', SIZE_LOCAL
! write(600+ME_LOCAL, *) cv%r(1:cv%num_cv,1:main_offset)
! close(600+ME_LOCAL)
! write(700+ME_LOCAL, *) frames%r(3,3,1:frames%num_frames)
! close(700+ME_LOCAL)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! CV interpolation
      elseif (( keyword(1:4).eq.'INTE'(1:4) )) then
! note: this is done serially
! get specifications

        if (qprint) then ; oldiol=iolev ; iolev=1 ; endif

!ccccccccccccc should the new cv file be interpolated from the CV old file?
        interp_cv=(indxa(comlyn, comlen, 'INTERPCV').gt.0)
        if (qprint) then
         if (interp_cv) then
          write(info, 6801) whoami
         else
          write(info, 6802) whoami
         endif
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif
!
 6801 format(A,' WILL OBTAIN NEW CV VALUES BY INTERPOLATION.')
 6802 format(A,' NEW CV VALUES WILL BE READ FROM FILE.')
! interpolation type
        if (interp_cv) then
         int_method=0
         method=gtrma(comlyn, comlen, 'METH')
         length=len(method)
         call trima(method, length)
         if (length.ge.4) then
           if (( method(1:4).eq.'LINE'(1:4) )) then
             int_method=linear
           elseif (( method(1:4).eq.'BSPL'(1:4) )) then
             int_method=bspline
           elseif (( method(1:4).eq.'SPLI'(1:4) )) then
             int_method=spline
           elseif (( method(1:4).eq.'LIN2'(1:4) )) then
             int_method=linear_exact
           endif
         endif
! print summary
         if (qprint) then
           if (int_method.gt.0) then
             length=len(methods(int_method))
             call trima(methods(int_method), length)
             write(info,6770) whoami, methods(int_method)(1:length)
 6770 format(/A,' WILL INTERPOLATE CV USING ',A,' INTERPOLATION')
           else
             if (length.gt.0) then
               write(info,6771) whoami, method(1:length), whoami
 6771 format(/A,' UNRECOGNIZED INTERPOLATION METHOD: ',A,'.',/, &
     & A, ' WILL INTERPOLATE CV USING LINEAR INTERPOLATION')
             else
              write(info,6772) whoami, whoami
 6772 format(/A,' UNSPECIFIED INTERPOLATION METHOD.',/, &
     & A, ' WILL INTERPOLATE CV USING LINEAR INTERPOLATION')
             endif ! length
             write(OUTU,'(A)') pack(info,info.ne.'');info='';
           endif ! int_method
         endif ! prnlev
         if (int_method.eq.0) int_method=linear ! choose linear interpolation as default
        endif ! interp_cv
! process other options ccccccccccccccccccccccccccccccccccccccccccccccc
!
        if (indx(comlyn, comlen, 'NIN', 3).gt.0) then
          num_rep_in=gtrmi(comlyn, comlen, 'NIN', 0)
          if (num_rep_in.le.0) then
            if (qprint) then ; write(info, 6781) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6781 format(A,' NUMBER OF INPUT REPLICAS MUST BE > 0. ',/, &
     & ' NOTHING DONE.')

            iolev=oldiol

            return
          else
            call gtrmwa(comlyn, comlen, 'CVIN', 4, name_cv_in, 80, len_cv_in)
            if (len_cv_in.le.0) then
              if (qprint) then ; write(info, 6782) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6782 format(A,' INPUT CV FILE NAME UNSPECIFIED.',/, &
     & ' NOTHING DONE.')

              iolev=oldiol

              return
            else
              if (qprint) then ; write(info,6783) &
     & whoami, num_rep_in, whoami, name_cv_in(1:len_cv_in) ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6783 format(A,' INITIAL STRING RESOLUTION: ', I5, ' REPLICAS.',/, &
     & A,' INPUT CV FILE IS ', A)
            endif ! len_cv_in<=0
          endif ! num_rep_in<=0
        else
          if (qprint) then ; write(info, 6784) whoami, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6784 format(A,' NUMBER OF INPUT REPLICAS UNSPECIFIED',/, &
     & A,' NOTHING DONE.')

          iolev=oldiol

          return
        endif ! indx('NIN')
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
! process output CV specification
        if (indx(comlyn, comlen, 'NOUT', 4).gt.0) then
          num_rep_out=gtrmi(comlyn, comlen, 'NOUT', 0)
          if (num_rep_out.le.0) then
            if (qprint) then ; write(info, 6785) whoami, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6785 format(A,' NUMBER OF OUTPUT REPLICAS MUST BE > 0. ',/,A, &
     & ' NOTHING DONE.')

            iolev=oldiol

            return
          else
            call gtrmwa(comlyn, comlen, 'CVOUT', 5, name_cv_out, 80, len_cv_out)
            if (len_cv_out.le.0) then
              if (qprint) then ; write(info, 6786) whoami, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6786 format(A,' OUTPUT CV FILE NAME UNSPECIFIED.',/,A, &
     & ' NOTHING DONE.')

              iolev=oldiol

              return
            else
              if (qprint) then
               write(info,6787) whoami, num_rep_out, whoami, name_cv_out(1:len_cv_out)
               write(OUTU,'(A)') pack(info,info.ne.'');info='';
              endif
 6787 format(A,' OUTPUT STRING RESOLUTION: ', I5, ' REPLICAS.',/, &
     & A,' OUPUT CV FILE IS ', A)
            endif ! len_cv_out
          endif ! num_rep_out
        else ! num_rep_out
          if (qprint) then ; write(info, 6788) whoami, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6788 format(A,' NUMBER OF OUTPUT REPLICAS UNSPECIFIED',/, &
     & A,' NOTHING DONE.')

          iolev=oldiol

          return
        endif ! indx('NOUT')
!ccccccccccccc coordinate file specification
        inte_get_coor=(indxa(comlyn, comlen, 'COOR').gt.0)
        if (inte_get_coor) then ! look for input and output coordinate files
          if (qprint) then ; write(info,6800) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6800 format(A,' WILL GENERATE REPLICA COORDINATE SETS.')
          inte_get_coor=.true.
! will also interpolate coordinates
          call gtrmwa(comlyn, comlen, 'CRIN', 4, name_cor_in, 80, len_cor_in) ! text file which contains a list of file names (6/20/2011)
          if (len_cor_in.le.0) then
            if (qprint) then ; write(info, 6789) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6789 format(A,' INPUT COORDINATES FILE NAME UNSPECIFIED.',/, &
     & '  NOTHING DONE.')

            iolev=oldiol

            return
          endif
!
          call gtrmwa(comlyn, comlen, 'CROUT', 5, name_cor_out, 80, len_cor_out)
          if (len_cor_out.le.0) then
            if (qprint) then ; write(info, 6790) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6790 format(A,' OUTPUT COORDINATES FILE NAME UNSPECIFIED.',/, &
     & ' NOTHING DONE.')

            iolev=oldiol

            return
          endif ! len_cor_out
! parse file format spec. (same for both input/output)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

          lresid=.false.

          if (indxa(comlyn, comlen, 'PDB').gt.0) then

            moder=-1
            modew=4
            if (indxa(comlyn, comlen, 'RESI').gt.0) lresid=.true.



            form='FORMATTED'

          elseif ( (indxa(comlyn, comlen, 'FILE').gt.0).or. &
     & (indxa(comlyn, comlen, 'UNFO').gt.0)) then
            moder=0
            modew=1
            form='UNFORMATTED'

          elseif ( (indxa(comlyn, comlen, 'CARD').gt.0).or. &
     & (indxa(comlyn, comlen, 'FORM').gt.0)) then

            moder=1
            modew=2



            form='FORMATTED'
          else ! default

            moder=1
            modew=2



            form='FORMATTED'
          endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! write summary
          if (qprint) then ! note: using qprint as a root node flag, too (should change this)
!ccccc get coordinate file names
           ifile=-1 ! a valid unit number will be assigned by __OPEN_FILE
           ofile=-1
           call open_file(ifile, name_cor_in(1:len_cor_in), 'FORMATTED', 'READ')
           allocate(fname_cor_in(num_rep_in))
!
           do j=1, num_rep_in
            read(ifile,*) fname_cor_in(j)
           enddo
           call VCLOSE(ifile, 'KEEP', ierror)
!
           write(info,6791) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6791 format(A,' COORDINATE SETS WILL BE READ FROM', &
     & ' THE FOLLOWING FILES:' )
!
           do j=1, num_rep_in
            write(info,'(A1,I5," ",A80)') char(9), j, fname_cor_in(j) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
           enddo
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
           call open_file(ofile, name_cor_out(1:len_cor_out), 'FORMATTED', 'READ')
!
           allocate(fname_cor_out(num_rep_out))
!
           do j=1, num_rep_out
            read(ifile,'(A80)') fname_cor_out(j)
           enddo
           call VCLOSE(ofile, 'KEEP', ierror)
!
           write(info,6793) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6793 format(A,' COORDINATE SETS WILL BE WRITTEN TO THE FOLLOWING FILES:' )
!
           do j=1, num_rep_out
            write(info,'(A1,I5," ",A80)') char(9), j, fname_cor_out(j) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
           enddo
!
          endif ! qprint
        endif ! 'COOR'
!
        if (.not.(interp_cv.or.inte_get_coor)) then ! nothing to do
         write(info,'(A," NOTHING TO DO")') whoami
         write(OUTU,'(A)') pack(info,info.ne.'');info='';

         iolev=oldiol

         return
        endif
!ccccccccccccccccccccccccc do work
! interpolate CV first
! compute cv weights, if needed
        if (.not.cv_common_weights_initialized) then
         call wrndie(0,whoami,trim('CV WEIGHTS NOT INITIALIZED. WILL COMPUTE FROM M(X)'))
         call smcv_compute_wgt(x,y,z,amass)
        endif
!
        if (interp_cv) then
         if (qprint) then ! only the head node does this
! open CV files
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! now prepare cv units and call interpolation routine
          ifile=-1 ! a valid unit number will be assigned by __OPEN_FILE
          ofile=-1
          call open_file(ifile, name_cv_in, 'FORMATTED', 'READ')
          call open_file(ofile, name_cv_out, 'FORMATTED', 'WRITE')
!
          call cv_common_interpolate(ifile, ofile, num_rep_in, num_rep_out,&
     & int_method)
!
          call VCLOSE(ifile, 'KEEP', ierror)
          call VCLOSE(ofile, 'KEEP', ierror)
         endif ! qprint
        endif ! interp_cv
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! process coordinate files, if requested
! this code is fairly slow; things are complicated by providing compatibility with frames, which means that
! every distance computation, essentially, required an 'optimal' rearrengement of frame vectors
! this is provided by routine `frame_align_rmsd`
        if (inte_get_coor) then
! rmsd array
         if(associated(inte_rmsd))deallocate(inte_rmsd)
         allocate(inte_rmsd(num_rep_out,num_rep_in))
! cv array
         if(associated(rtemp))deallocate(rtemp)
         allocate(rtemp(max_cv_common,num_rep_out))
!
! (re-)load new cv file and store cv in rtemp
         do j=1, num_rep_out
           if (qprint) then
            call open_file(ifile, name_cv_out, 'FORMATTED', 'READ')
           endif
           call cv_common_read_local_from_global(ifile, num_rep_out, &
     & j, comp) ! this needs to be run in parallel
           rtemp(:,j)=cv%r(:,comp)
         enddo
!
         if (qprint) then ; write(info,6974) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6974 format(A,' READING COORDINATE FILES')
!

         islct=1.

         do j=1, num_rep_in
! open file
            length=len_trim(fname_cor_in(j))
            dummy=fname_cor_in(j)(1:length)
!
            if (qroot) then ; call open_file(ifile, dummy, form, 'READ') ; endif
            dummy=''
            if (mestring.eq.0) then ! need a whole group to read correctly

             call cread(ifile, titleb, ntitlb, icntrl, xcomp, ycomp, &
     & zcomp, wcomp, natom, moder, islct, &
     & 0, res, nres, atype, ibase, 1, ifreea, &
     & segid, resid, nictot, nseg, lresid, .false., &
     & dummy, 80, 0, .false.)






             if (qroot) then ; call VCLOSE(ifile, 'KEEP', ierror) ; endif
! compute "distance" to cv values
             do i=1, num_rep_out
              cv%r(:,main)=rtemp(:,i) ! CV into main set
              if (frames_initialized) &
     & call frame_align_rmsd(xcomp, ycomp, zcomp, amass) ! calculate optimal frame axes in the sense of minimal rmsd
! compute the instantaneous CV realizations
              call smcv_fill(xcomp, ycomp, zcomp, amass, comp) ! works in series or parallel
! compute RMSD:
              inte_rmsd(i,j)=cv_common_rmsd(comp,main)
             enddo
            endif ! mestring
! write(600,*) inte_rmsd(:,j)
         enddo ! j=1,num_rep_in
!
! reload new cv file
         do j=1, num_rep_out
           which=minloc(inte_rmsd(j,:)) ! which index corresponds to the smallest rmsd (ds)
! open the corresponding file and save under new file name
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! open file
           length=len_trim(fname_cor_in(which(1)))
           dummy=fname_cor_in(which(1))(1:length)
!
           if (mestring.eq.0) then
            if (qroot) then ; call open_file(ifile, dummy, form, 'READ') ; endif

            dummy=''
            call cread(ifile, titleb, ntitlb, icntrl, xcomp, ycomp, &
     & zcomp, wcomp, natom, moder, islct, &
     & 0, res, nres, atype, ibase, 1, ifreea, &
     & segid, resid, nictot, nseg, lresid, .false., &
     & dummy, 80, 0, .false.)
            if (ntitla+1 .lt. maxtit) &
     & write(titlea(ntitla+1),'(A,I5,A,I5)') &
     & '* REPLICA ',j,' OF ',num_rep_out






            if (qroot) then ; call VCLOSE(ifile, 'KEEP', ierror) ; endif
!cccccccccccc now write the same file cccccccccccccccccccccccc
! open file
            length=len_trim(fname_cor_out(j))
            dummy=fname_cor_out(j)(1:length)
!
            if (qroot) then ; call open_file(ofile, dummy, form, 'WRITE') ; endif
!

            call CWRITE(ofile,TITLEA,min(NTITLA+1,maxtit),ICNTRL, &
     & xcomp,ycomp,zcomp,wcomp,res,atype,ibase, &
     & NRES,NATOM,islct,modew,0,0,.false.)






            if (qroot) then ; call VCLOSE(ofile, 'KEEP', ierror) ; endif
!
           endif ! mestring
         enddo ! loop over new coordinate sets
         if(associated(fname_cor_in))deallocate(fname_cor_in)
         if(associated(fname_cor_out))deallocate(fname_cor_out)
        endif ! inte_get_coor
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

        iolev=oldiol

      else
            write(info(1),*)'UNRECOGNIZED SUBCOMMAND: ',keyword;call wrndie(0,whoami,trim(info(1)))
      endif
      end subroutine smcv
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_init(maxcv)
      use sm_var
      use sm_config
      use cv_common, only: cv_common_initialized, cv_common_init, main, &
     & comp, zcur, instant, runave, forces2, max_cv_common
! , only:smcv_initialized, nstring, mestring,
! & cv_send_displ,cv_send_count
      use stream
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use number
      use param_store, only: set_param
      use parallel, only: psnd4, psnd8
!
      implicit none
!
   character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
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
      integer*4 :: ierror
      logical :: qroot, qslave
      integer*4 :: temp1(3), temp2(3) ! for communication
      integer, optional :: maxcv
!
      integer(kind=MPI_ADDRESS_KIND) :: lb, extent
!
      character(len=len("SMCV_INIT>") ),parameter::whoami="SMCV_INIT>";!macro
!
! do a basic communicator check:
      if (ME_LOCAL.eq.0.and.ME_STRNG.eq.MPI_UNDEFINED) then
        write(info, 111) whoami, ME_GLOBAL, whoami
 111 FORMAT(A, ' WORLD REPLICA ',I5, ' HAS ZERO GROUP ID', &
     & /,A,' BUT INVALID STRING ID (MAY BE OK).')
        write(OUTU,'(A)') pack(info,info.ne.'');info='';
      elseif (ME_STRNG.ne.MPI_UNDEFINED.and. &
     & (ME_LOCAL.ne.0.or.MPI_COMM_LOCAL.eq.MPI_COMM_NULL)) then
        write(info, 112) whoami, ME_GLOBAL, whoami
 112 FORMAT(A, ' WORLD REPLICA ',I5, ' HAS A VALID STRING ID', &
     & /,A,' BUT A NONZERO GROUP ID. ABORTING.')
        write(OUTU,'(A)') pack(info,info.ne.'');info='';
       return
      endif
!
      qroot=ME_STRNG.ne.MPI_UNDEFINED
      qslave=ME_LOCAL.ne.MPI_UNDEFINED ! (also includes roots)
!
      if (smcv_initialized) then
       if (qroot) then
        if (ME_STRNG.eq.0) then
          write(info,'(2A)') &
     & whoami, ' SMCV ALREADY INITIALIZED. CALL "DONE" TO CLEAN UP.'
          write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif ! ME_STRNG
       endif ! qroot
       return
      endif
!
      nstring=1 ! safe (hopefully) default
      mestring=-1 ! safe (hopefully) default
!
      if (qroot) then
        nstring=SIZE_STRNG
        mestring=ME_STRNG
      endif
! broadcast string size to all slave nodes
#if (KEY_INTEGER8==0)
      call PSND4(nstring,1) 
#endif
#if (KEY_INTEGER8==0)
      call PSND4(mestring,1) 
#endif
#if (KEY_INTEGER8==1)
      call PSND8(nstring,1) 
#endif
#if (KEY_INTEGER8==1)
      call PSND8(mestring,1) 
#endif
! set envorinment variables
      call set_param('NSTRING',nstring)
      call set_param('MESTRING',mestring)
!
      if (qroot) then
        if (ME_STRNG.eq.0) then
          write(info,'(2A,I5, A)') &
     & whoami, ' FOUND ',nstring,' REPLICAS.'
          write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif
      endif
!
      smcv_initialized=.true.
!
      if (.not.cv_common_initialized) then
       if (present(maxcv)) then
        if (maxcv.gt.0) then
         if (qroot.and.ME_STRNG.eq.0) then
          write(info,'(2A,I5, A)') &
     & whoami, ' WILL INITIALIZE SMCV WITH STORAGE FOR AT MOST ',maxcv,' CV.'
          write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif ! root prints
         call cv_common_init(maxcv)
        else
         call wrndie(0,whoami,trim('NEGATIVE NUMBER OF CV REQUESTED, USING PREVIOUS VALUE (OR DEFAULTS).'))
         call cv_common_init() ! default
        endif
       else ! present maxcv
        call cv_common_init() ! default
       endif
      endif
! allocate index arrays for cv index limits (parallelization)
      allocate(cv_send_displ(SIZE_LOCAL), & ! cv
     & cv_send_count(SIZE_LOCAL))
      allocate(fr_send_displ(SIZE_LOCAL), & ! frames
     & fr_send_count(SIZE_LOCAL))
      allocate(qt_send_displ(SIZE_LOCAL), & ! quat
     & qt_send_count(SIZE_LOCAL))
      allocate(imap_displ(SIZE_LOCAL), & ! imap
     & imap_count(SIZE_LOCAL))
! initialize
      cv_send_displ=0
      cv_send_count=0
      fr_send_displ=0
      fr_send_count=0
      qt_send_displ=0
      qt_send_count=0
      imap_displ=0
      imap_count=0
!
! define/update derived MPI types:
! these special types are for communicating CV;
! I chose to index cv%r in a way that is not really
! suitable for communication/parallelization, hence the special
! types with custom blocks/strides/extents
!ccccccccccccccccc two cv values (main+comp) ccccccccccccccccc
!
! call mpi_type_indexed(2,(/1,1/),
! & (/max_cv_common*(main-1),
! & max_cv_common*(comp-1)/),
! & mpifloat, MPI_CV_TYPE2_, ierror)
      temp1=(/ione,ione,izero/)
      temp2=(/max_cv_common*(main-1), &
     & max_cv_common*(comp-1),izero/)
      call mpi_type_indexed(2, temp1, temp2, &
     & mpifloat, MPI_CV_TYPE2_, ierror)
! corresponding resized type (modified extent)
      lb=0
      extent=sizeofreal
      call mpi_type_create_resized(MPI_CV_TYPE2_,lb,extent, &
     & MPI_CV_TYPE2, ierror)
      call mpi_type_commit(MPI_CV_TYPE2, ierror)
!ccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccc three cv values cccccccccccccc
! call mpi_type_indexed(3,(/1,1,1/),
! & (/max_cv_common*(zcur-1),
! & max_cv_common*(instant-1),
! & max_cv_common*(forces2-1)/), ! strides: i.e. take 1 element from zcur, 1 from inst., 1 from forces2
! & mpifloat, MPI_CV_TYPE3_, ierror)

      temp1=(/1,1,1/)
      temp2=(/max_cv_common*(zcur-1), &
     & max_cv_common*(instant-1), &
     & max_cv_common*(forces2-1)/)
      call mpi_type_indexed(3, temp1, temp2, &
     & mpifloat, MPI_CV_TYPE3_, ierror)

! write(0,*) temp2 !aa
! stop

! corresponding resized type
      lb=0
      extent=sizeofreal
      call mpi_type_create_resized(MPI_CV_TYPE3_,lb,extent, &
     & MPI_CV_TYPE3, ierror)
      call mpi_type_commit(MPI_CV_TYPE3, ierror)
!ccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccc three cv values (different from above) cccccccccccccc
! call mpi_type_indexed(3,(/1,1,1/),
! & (/max_cv_common*(runave-1),
! & max_cv_common*(instant-1),
! & max_cv_common*(forces2-1)/),
! & mpifloat, MPI_CV_TYPE3I_, ierror)

      temp1=(/1,1,1/)
      temp2=(/max_cv_common*(runave-1), &
     & max_cv_common*(instant-1), &
     & max_cv_common*(forces2-1)/)
      call mpi_type_indexed(3, temp1, temp2, &
     & mpifloat, MPI_CV_TYPE3I_, ierror)

! corresponding resized type (note change of extent)
      lb=0
      extent=sizeofreal
      call mpi_type_create_resized(MPI_CV_TYPE3I_,lb,extent, &
     & MPI_CV_TYPE3I, ierror)
      call mpi_type_commit(MPI_CV_TYPE3I, ierror)
!
      MPI_GRAD_TYPE =MPI_DATATYPE_NULL
      MPI_GRAD_TYPE_=MPI_DATATYPE_NULL
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccc
      end subroutine smcv_init
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_done()
      use cv_common, only: cv_common_done
      use sm_var,only: smcv_initialized, nstring, mestring
      use sm_config,only: cv_send_displ, cv_send_count, &
     & fr_send_displ,fr_send_count, &
     & imap_displ,imap_count, &
     & qt_send_displ,qt_send_count, &
     & MPI_CV_TYPE2, MPI_CV_TYPE2_, &
     & MPI_CV_TYPE3, MPI_CV_TYPE3_, &
     & MPI_CV_TYPE3I, MPI_CV_TYPE3I_, &
     & MPI_GRAD_TYPE, MPI_GRAD_TYPE_
!
      use stream
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use param_store, only: set_param
!
      implicit none
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      integer*4 :: ierror
!
      character(len=len("SMCV_DONE>") ),parameter::whoami="SMCV_DONE>";!macro
!
      if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then
        write(info,'(2A,I5, A)') whoami, ' CLEANING UP.' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
      endif
!
      call cv_common_done()
      nstring=-1
      mestring=-1
!

! set envorinment variable
      call set_param('NSTRING',nstring)
      call set_param('MESTRING',mestring)

!
! deallocate index arrays for cv index limits (parallelization)
      if (smcv_initialized) then
       deallocate(cv_send_displ,cv_send_count, &
     & fr_send_displ,fr_send_count, &
     & imap_displ,imap_count, &
     & qt_send_displ,qt_send_count)
! free MPI_TYPES
       if (MPI_CV_TYPE2.ne.MPI_DATATYPE_NULL) &
     & call mpi_type_free(MPI_CV_TYPE2,ierror);
       if (MPI_CV_TYPE2_.ne.MPI_DATATYPE_NULL) &
     & call mpi_type_free(MPI_CV_TYPE2_,ierror);
!
       if (MPI_CV_TYPE3.ne.MPI_DATATYPE_NULL) &
     & call mpi_type_free(MPI_CV_TYPE3,ierror);
       if (MPI_CV_TYPE3_.ne.MPI_DATATYPE_NULL) &
     & call mpi_type_free(MPI_CV_TYPE3_,ierror);
!
       if (MPI_CV_TYPE3I.ne.MPI_DATATYPE_NULL) &
     & call mpi_type_free(MPI_CV_TYPE3I,ierror);
       if (MPI_CV_TYPE3I_.ne.MPI_DATATYPE_NULL) &
     & call mpi_type_free(MPI_CV_TYPE3I_,ierror);
!
       if (MPI_GRAD_TYPE.ne.MPI_DATATYPE_NULL) &
     & call mpi_type_free(MPI_GRAD_TYPE,ierror);
       if (MPI_GRAD_TYPE_.ne.MPI_DATATYPE_NULL) &
     & call mpi_type_free(MPI_GRAD_TYPE_,ierror);
      endif
!
! what else ?
!
      smcv_initialized=.false.
!
      end subroutine smcv_done
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_repa_init(COMLYN, COMLEN)
! initialize string reparametrization
!
      use sm_var
      use stream
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
!
      character(len=*) :: comlyn
      integer :: comlen
!
      character(len=20) :: methods(5)
      data methods/ 'LINEAR','CUBIC SPLINE','B-SPLINE','DST','LINEAR_EXACT'/
! selection array
      integer :: qlinear, qspline, qbspline, qdst, qlinear_exact
      integer :: mlen
!
! declare functions here
!
      logical :: qroot, qprint
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      character(len=len("SMCV_REPA_INIT>") ),parameter::whoami="SMCV_REPA_INIT>";!macro
!
      qroot=MPI_COMM_STRNG.ne.MPI_COMM_NULL
      qprint=qroot.and.ME_STRNG.eq.0
! begin
! reset variables
      qspline=0
      qbspline=0
      qlinear=0
      qdst=0
      qlinear_exact=0
      dst_cutoff=0d0
      interp_method=0
      repa_initialized=.false.
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (indxa(comlyn, comlen, 'LINE').gt.0) then
       qlinear=1
       interp_method=linear
      endif
      if ((indxa(comlyn, comlen, 'CSPL').gt.0).or. &
     & (indxa(comlyn, comlen, 'SPLI').gt.0)) then
       qspline=1
       interp_method=spline
      endif
      if (indxa(comlyn, comlen, 'BSPL').gt.0) then
       qbspline=1
       interp_method=bspline
      endif
      if (indxa(comlyn, comlen, 'DST').gt.0) then
       qdst=1
       interp_method=dst
! did the user specify filter cutoff?
       dst_cutoff=gtrmf(comlyn, comlen, 'WNCT', -1.0d0)
       if (dst_cutoff.lt.0.0d0) then
        if (qprint) then ; write(info,664) whoami, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 664 FORMAT(A,' DST REQUESTED BUT FILTER CUTOFF', &
     & A, ' NOT SPECIFIED.',/,' WILL USE 0.500')
        dst_cutoff=0.5d0
       endif
      endif
      if (indxa(comlyn, comlen, 'LIN2').gt.0) then
       qlinear_exact=1
       interp_method=linear_exact
      endif
!ccccccc CHECK FOR MULTIPLE OPTIONS
      if ((qspline+qlinear+qbspline+qdst+qlinear_exact) .eq. 0) then
       if (qprint) then ; write(info,665) whoami, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 665 FORMAT(A,' INTERPOLATION METHOD NOT SPECIFIED.',/, &
     & A,' WILL USE LINEAR INTERPOLATION.')
       interp_method=linear
      elseif ((qspline+qlinear+qbspline+qdst+qlinear_exact) .gt. 1) then
       call wrndie(0,whoami,trim('TOO MANY INTERPOLATION OPTIONS.'))
      endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! did the user specify a tolerance?
      if (interp_method.ne.linear_exact) then ! options below are invalid for exact interpolation
       def=gtrmf(comlyn, comlen, 'DEFI', 1.1d0)
       if (def.lt.1.0d0) then
         call wrndie(0,whoami,trim('INTERPOLATION TOLERANCE MUST BE >= 1.'))
! return
       endif
! did the user specify a maximum number of iterations?
       iterations=gtrmi(comlyn, comlen, 'ITER', 10)
      else
       def=0d0
       iterations=0
      endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! print summary
      if (qprint) then
       mlen=len(methods(interp_method))
       if (interp_method.eq.linear_exact) then
        write(info,666) whoami,methods(interp_method)(1:mlen)
       else
        write(info,667) whoami,methods(interp_method)(1:mlen),whoami,&
     & def
       endif
       write(OUTU,'(A)') pack(info,info.ne.'');info='';
 666 format(A,' WILL REPARAMETRIZE STRING USING ',A,' INTERPOLATION')
 667 format(A,' WILL REPARAMETRIZE STRING USING ',A,/, &
     &A,' INTERPOLATION TO WITHIN MAX(DS)/MIN(DS) < ',F7.3,' TOLERANCE')
       if (iterations.gt.0) then ; write(info,668) whoami, iterations ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 668 format(A,' WITH A MAXIMUM OF ',I5,' ITERATIONS')
       if(interp_method.eq.dst) then ; write(info,6680) whoami,dst_cutoff*100.0 ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6680 format(A,' DST INTERPOLATION WILL USE THE LOWER ',F8.4, &
     & '% OF WAVENUMBERS')
      endif
!
! initialize arclength array
      if (.not.associated(ds)) then
       allocate(ds(nstring-1))
       ds=0.0
      endif
! initialize curvature array
      if (.not.associated(curv)) then
       allocate(curv(nstring-2))
       curv=0.0
      endif
!
      repa_initialized=.true.
!
      end subroutine smcv_repa_init
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_stat_init(comlyn, comlen)
!
      use sm_var
      use sm_config,only: vtime_offset, rextime_offset
      use cv_common,only:cv_common_neq_work_init, cv_common_rex_read_map
!
      use stream
      use dimens_fcm
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      implicit none
!====================================================================
      CHARACTER(LEN=*) :: COMLYN
      integer :: comlen
!
      integer :: wtag_len, rex_flen_old , ierror

      integer :: oldiol

!
      character(len=80) :: rex_fname_old
      character(len=len("SMCV_STAT_INIT>") ),parameter::whoami="SMCV_STAT_INIT>";!macro
!
      logical :: qroot, qprint
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      qroot=MPI_COMM_STRNG.ne.MPI_COMM_NULL
      qprint=qroot.and.ME_STRNG.eq.0

      if (qprint) then; oldiol=iolev; iolev=0; endif

! begin
! reset iteration counter
! did the user specify it?
      stat_iteration_counter=gtrmi(comlyn, comlen, 'COUN', -1)
      stat_iteration_counter=max(stat_iteration_counter,0)
      if (stat_iteration_counter.gt.0) then
       if (qprint) then ; write(info,639) whoami, stat_iteration_counter ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 639 format(A,' SETTING ITERATION COUNTER TO ',I7)
      endif
!
      cv_fname=''
      output_cv=.false.

      forces_fname=''
      output_forces=.false.
!
      voronoi_fname=''
      output_voronoi_hist=.false.
      output_voronoi_log=.false.
      output_voronoi_map=.false.
!
      rmsd_ave_fname=''
      output_rmsd_ave=.false.
!
      rmsd0_fname=''
      output_rmsd0=.false.

      dsdt_fname=''
      output_dsdt=.false.
!
      c_fname=''
      output_curvature=.false.
!
      s_fname=''
      output_arclength=.false.
!
      fe_fname=''
      output_fe=.false.
!
      work_fname=''
      work_tag=''
      output_work=.false.
!
      rex_fname_old=''
      rex_fname=''
      output_rex_log=.false.
      output_rex_map=.false.
!
      wgt_fname=''
      output_wgt=.false.
!
      M_fname=''
      output_M=.false.

!ccccccccccccccccc first process the RMSD-related commands
!!!!!!!!!!!!!! RMSD from static structure in comp (zts/fts)
      if (indxa(comlyn, comlen, 'RMSD').gt.0) then ! request for RMSD
       output_rmsd0=.true.
       call gtrmwa(COMLYN, COMLEN, 'RNAM', 4, rmsd0_fname, 80, rmsd0_flen)
       if (rmsd0_flen.eq.0) then
         call wrndie(0,whoami,trim('NO RMSD FILE NAME SPECIFIED. WILL WRITE TO STDOUT.'))
         rmsd0_funit=outu
       else
         if (indxa(comlyn, comlen, 'RAPP').gt.0) then ! APPEND?
           rform='APPEND'
         else
           rform='WRITE'
         endif
       endif
!ccccccccccc print summary
       if (qprint) then
         if (rmsd0_flen.gt.0) then
          write(info,660 ) whoami,rmsd0_fname(1:rmsd0_flen)
         else
          write(info,661 ) whoami
         endif
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
       endif
 660 format(A,' WILL WRITE STRING RMSD TO FILE ',A)
 661 format(A,' WILL WRITE STRING RMSD TO STDOUT.')
!
      endif !! RMSD
!!!!!!!!!!!!!! RMSD from structure at the previous step (zts/fts)
      if (indxa(comlyn, comlen, 'DELS').gt.0) then
        output_dsdt=.true.
        call gtrmwa(COMLYN, COMLEN, 'DNAM', 4, dsdt_fname, 80, dsdt_flen)
        if (dsdt_flen.eq.0) then
         call wrndie(0,whoami,trim('NO DELS FILE NAME SPECIFIED. WILL WRITE TO STDOUT.'))
         dsdt_funit=outu
        else
         if (indxa(comlyn, comlen, 'DAPP').gt.0) then ! APPEND?
           dform='APPEND'
         else
           dform='WRITE'
         endif
        endif
!ccccccccccc print summary
        if (qprint) then
         if (dsdt_flen.gt.0) then
          write(info,650 ) whoami,dsdt_fname(1:dsdt_flen)
         else
          write(info,651 ) whoami
         endif
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif
 650 format(A,' WILL WRITE STRING RMSD(I,I+1) TO FILE ',A)
 651 format(A,' WILL WRITE STRING RMSD(I,I+1) TO STDOUT.')
!
      endif !
!!!!!!!!!!!!!! RMSD from average structure (zts/fts)
      if (indxa(comlyn, comlen, 'RMSA').gt.0) then
        output_rmsd_ave=.true.
        call gtrmwa(COMLYN, COMLEN, 'RANM', 4, rmsd_ave_fname, 80, rmsd_ave_flen)
        if (rmsd_ave_flen.eq.0) then
         call wrndie(0,whoami,trim('NO RMSA FILE NAME SPECIFIED. WILL WRITE TO STDOUT.'))
         rmsd_ave_funit=outu
        else
         if (indxa(comlyn, comlen, 'RAAP').gt.0) then ! APPEND?
           raform='APPEND'
         else
           raform='WRITE'
         endif
        endif
!ccccccccccc print summary
        if (qprint) then
         if (rmsd_ave_flen.gt.0) then
          write(info,6500 ) whoami,rmsd_ave_fname(1:rmsd_ave_flen)
         else
          write(info,6510 ) whoami
         endif
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif
 6500 format(A,' WILL WRITE STRING RMSD FROM AVERAGE STRUC. TO FILE ',A)
 6510 format(A,' WILL WRITE STRING RMSD FROM AVERAGE STRUC. TO STDOUT.')
!
! set number of samples in the average ( to continue a calculation )
        num_average_samples=max(gtrmi(comlyn, comlen, 'NAVE', 0),0)
! same value for cv
        call cv_set_ave_samples(num_average_samples)
        if (qprint) then ; write(info,6511) whoami, num_average_samples ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6511 format(A,' SETTING NUMBER OF SAMPLES IN THE AVERAGE SET TO ',I7)
!
      endif !
!!!!!!!!!!!!!! ARCLENGTH
      if (indxa(comlyn, comlen, 'ARCL').gt.0) then
        output_arclength=.true.
        call gtrmwa(COMLYN, COMLEN, 'ANAM', 4, s_fname, 80, s_flen)
        if (s_flen.eq.0) then
         call wrndie(0,whoami,trim('STRING LENGTH FILE NAME NOT SPECIFIED. WILL WRITE TO STDOUT.'))
         s_funit=outu
        else
         if (indxa(comlyn, comlen, 'AAPP').gt.0) then ! APPEND?
           sform='APPEND'
         else
           sform='WRITE'
         endif
        endif
!ccccccccccc print summary
        if (qprint) then
         if (s_flen.gt.0) then
          write(info,652) whoami,s_fname(1:s_flen)
         else
          write(info,653) whoami
         endif
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif
 652 format(A,' WILL WRITE STRING LENGTH TO FILE ',A)
 653 format(A,' WILL WRITE STRING LENGTH TO STDOUT.')
!
      endif ! ARCLENGTH
!!!!!!!!!!!!!! CURVATURE
      if (indxa(comlyn, comlen, 'CURV').gt.0) then
        output_curvature=.true.
        call gtrmwa(COMLYN, COMLEN, 'CVNM', 4, c_fname, 80, c_flen)
        if (c_flen.eq.0) then
         call wrndie(0,whoami,trim('CURVATURE FILE NAME NOT SPECIFIED. WILL WRITE TO STDOUT.'))
         c_funit=outu
        else
         if (indxa(comlyn, comlen, 'CAPP').gt.0) then ! APPEND?
           cform='APPEND'
         else
           cform='WRITE'
         endif
        endif
!ccccccccccc print summary
        if (qprint) then
         if (c_flen.gt.0) then
          write(info,6521) whoami,c_fname(1:c_flen)
         else
          write(info,6531) whoami
         endif
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif
 6521 format(A,' WILL WRITE CURVATURE TO FILE ',A)
 6531 format(A,' WILL WRITE CURVATURE TO STDOUT.')
!
      endif ! CURVATURE
!!!!!!!!!!!!!! FREE ENERGY
      if (indxa(comlyn, comlen, 'FREE').gt.0) then
        output_fe=.true.
        call gtrmwa(COMLYN, COMLEN, 'FENM', 4, fe_fname, 80, fe_flen)
        if (fe_flen.eq.0) then
         call wrndie(0,whoami,trim('NO F.E. FILE NAME SPECIFIED. WILL WRITE TO STDOUT.'))
         fe_funit=outu
        else
         if (indxa(comlyn, comlen, 'FAPP').gt.0) then ! APPEND?
           feform='APPEND'
         else
           feform='WRITE'
         endif
        endif
!ccccccccccc print summary cccccccccccccccccccccccccccccccccccccc
        if (qprint) then
         if (fe_flen.gt.0) then
          write(info,6520) whoami,fe_fname(1:fe_flen)
         else
          write(info,6530) whoami
         endif
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif
 6520 format(A,' WILL WRITE FREE ENERGY TO FILE ',A)
 6530 format(A,' WILL WRITE FREE ENERGY TO STDOUT.')
!
      endif ! F.E.
!ccccccccccccccccccccccccccc NONEQ. WORK cccccccccccccccccccccccc
      if (indxa(comlyn, comlen, 'WORK').gt.0) then
        output_work=.true.
        call cv_common_neq_work_init() ! initialize force and position arrays
        call gtrmwa(COMLYN, COMLEN, 'WKNM', 4, work_fname, 80, work_flen)
        if (work_flen.eq.0) then
         call wrndie(0,whoami,trim('NO F.E. FILE NAME SPECIFIED. WILL WRITE TO STDOUT.'))
         work_funit=outu
        else
         if (indxa(comlyn, comlen, 'WKAP').gt.0) then ! APPEND?
          wkform='APPEND'
         else
          wkform='WRITE'
         endif
        endif
! specify tag that identifies the work calculated with a particular process
        call gtrmwa(COMLYN, COMLEN, 'WTAG', 4, work_tag, 8, wtag_len)
        if (wtag_len.eq.0) then
         call wrndie(0,whoami,trim('WORK TAG NOT SPECIFIED.'))
        endif
!ccccccccccc print summary
        if (qprint) then
         if (work_flen.gt.0) then
          write(info,6523) whoami,work_fname(1:work_flen)
         else
          write(info,6533) whoami
         endif
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif
 6523 format(A,' WILL WRITE NON-EQ. WORK TO FILE ',A)
 6533 format(A,' WILL WRITE NON-EQ. WORK TO STDOUT.')
!
      endif ! F.E.
!cccccccccc process CV output options ccccccccccccccccccccccc
      if (indxa(comlyn, comlen, 'COLV').gt.0) then
! get nergy file name
        call gtrmwa(COMLYN, COMLEN, 'CNAM', 4, cv_fname, 80, cv_flen)
!ccccccccccc print summary
        if (cv_flen.gt.0) then
         output_cv=.true.
         if (qprint) then
           write(info,6620 ) whoami,cv_fname(1:cv_flen)
           write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
         if (indxa(comlyn, comlen, 'CVAP').gt.0) then ! APPEND?
           cvform='APPEND'
         else
           cvform='WRITE'
         endif
        else
          call wrndie(0,whoami,trim('NO FILE NAME GIVEN. WILL NOT WRITE CV.'))
        endif
 6620 format(A,' WILL WRITE CV TIME SERIES TO FILE ',A,'.')
!
      endif ! cv output
!ccccccccccccccccccccccc output weights cccccccccccccccccccccc
      if (indxa(comlyn, comlen, 'WEIG').gt.0) then
! get nergy file name
        call gtrmwa(COMLYN, COMLEN, 'WTNM', 4, wgt_fname, 80, wgt_flen)
!ccccccccccc print summary
        if (wgt_flen.gt.0) then
         output_wgt=.true.
         if (qprint) then
          write(info,6621) whoami,wgt_fname(1:wgt_flen)
          write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
         if (indxa(comlyn, comlen, 'WTAP').gt.0) then ! APPEND?
           wgtform='APPEND'
         else
           wgtform='WRITE'
         endif
        else
          call wrndie(0,whoami,trim('NO FILE NAME GIVEN. WILL NOT WRITE CV WEIGHTS.'))
        endif
 6621 format(A,' WILL WRITE CV WEIGHTS TO FILE ',A,'.')
!
      endif ! cv output
!cccccccccccc process Voronoi histogram output options ccccccccccc
      voronoi_flen=0
      if (indxa(comlyn, comlen, 'VORO').gt.0) then
! get file name
        call gtrmwa(COMLYN, COMLEN, 'VNAM', 4, voronoi_fname, 80, voronoi_flen)
!ccccccccccc print summary
        if (voronoi_flen.gt.0) then
         output_voronoi_hist=.true.
         if (qprint) then
          write(info,6622) whoami,voronoi_fname(1:voronoi_flen)
         endif
        else
         call wrndie(0,whoami,trim('NO FILE NAME GIVEN. WILL NOT WRITE VORONOI HISTOGRAMS.'))
        endif
 6622 format(A,' WILL WRITE VORONOI HISTOGRAMS TO FILE ',A,'.DAT')
!
      endif ! voronoi histograms
!cccccccccccccc voronoi map cccccccccccccccccccccccccccccccccccccccc
      if (indxa(comlyn, comlen, 'VMAP').gt.0) then
! get file name
        if (voronoi_flen.eq.0) then
         call gtrmwa(COMLYN, COMLEN, 'VNAM', 4, voronoi_fname, 80, voronoi_flen)
        endif
!
        if (voronoi_flen.gt.0) then
         output_voronoi_map=.true.
         if (qprint) then
          write(info,6627) whoami,voronoi_fname(1:voronoi_flen)
          write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
        else
          call wrndie(0,whoami,trim('NO FILE NAME GIVEN. WILL NOT WRITE VORONOI MAP.'))
        endif
 6627 format(A,' WILL WRITE VORONOI MAP TO FILE ',A,'.MAP')
!
      endif ! voronoi map
!cccccccccccccc voronoi log cccccccccccccccccccccccccccccccccccccccc
      if (indxa(comlyn, comlen, 'VLOG').gt.0) then
! get file name
        if (voronoi_flen.eq.0) then
         call gtrmwa(COMLYN, COMLEN, 'VNAM', 4, voronoi_fname, 80, voronoi_flen)
        endif
! check for timestep offset
        vtime_offset=gtrmi(comlyn, comlen, 'VOFF', 0);
        if (vtime_offset.gt.0) then
         if (qprint) then ; write(info,6624) whoami, vtime_offset ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6624 format(A,' WILL OFFSET STEP COUNTER IN VORONOI LOG BY ',I10)
        endif
!
        if (voronoi_flen.gt.0) then
         output_voronoi_log=.true.
         if (qprint) then
          write(info,6623) whoami,voronoi_fname(1:voronoi_flen)
          write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
         if (indxa(comlyn, comlen, 'VLAP').gt.0) then ! APPEND?
           vlform='APPEND'
         else
           vlform='WRITE'
         endif ! vlap
!
        else
          call wrndie(0,whoami,trim('NO FILE NAME GIVEN. WILL NOT WRITE VORONOI LOG.'))
        endif ! voronoi_flen.gt.0
 6623 format(A,' WILL WRITE VORONOI LOG TO BINARY FILE ',A,'.DAT')
!
      endif ! complete voronoi log
!cccccccccccc replica exchange map cccccccccccccc
      rex_flen=0
      if (indxa(comlyn, comlen, 'REXM').gt.0) then
! get file name
        call gtrmwa(COMLYN, COMLEN, 'RXNM', 4, rex_fname, 80, rex_flen)
! check if user specified an custom map (e.g. from an older run)
        call gtrmwa(COMLYN, COMLEN, 'RXOL', 4, rex_fname_old, 80, rex_flen_old)
!
        if (rex_flen.gt.0) then
         output_rex_map=.true.
         if (qprint) then
          write(info,6721) whoami,rex_fname(1:rex_flen)
          write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
         if (rex_flen_old.gt.0) then
          if (qprint) then
            write(info,6722) whoami,rex_fname_old(1:rex_flen_old) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
            rex_funit=-1
            call open_file(rex_funit, rex_fname_old(1:rex_flen_old), 'FORMATTED', 'READ')
          endif
          call cv_common_rex_read_map(rex_funit)
          if (qprint) then ;call VCLOSE(rex_funit, 'KEEP', ierror) ; endif
         endif
!
        else
          call wrndie(0,whoami,trim('NO FILE NAME GIVEN. WILL NOT WRITE REPLICA EXCHANGE MAP.'))
        endif
 6721 format(A,' WILL WRITE REPLICA EXCHANGE MAP TO FILE ',A,'.MAP')
 6722 format(A,' WILL RESTART FROM REPLICA EXCHANGE MAP IN FILE ',A)
!
      endif ! replica exchange map
!cccccccccccccc replica exchange log cccccccccccccccccccccccccccccccccccccccc
      if (indxa(comlyn, comlen, 'REXL').gt.0) then
! get file name
        if (rex_flen.eq.0) then
         call gtrmwa(COMLYN, COMLEN, 'RXNM', 4, rex_fname, 80, rex_flen)
        endif
! check for timestep offset
        rextime_offset=gtrmi(comlyn, comlen, 'ROFF', 0);
        if (rextime_offset.gt.0) then
         if (qprint) then ; write(info,6724) whoami, whoami,rextime_offset ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6724 format(A,' WILL OFFSET STEP COUNTER IN REPLICA EXCHANGE LOG BY ' &
     & /,A,' ',I10)
        endif
!
        if (rex_flen.gt.0) then
         output_rex_log=.true.
         if (qprint) then
           write(info,6723) whoami,whoami,rex_fname(1:rex_flen)
           write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
         if (indxa(comlyn, comlen, 'RXAP').gt.0) then ! APPEND?
           rxlform='APPEND'
         else
           rxlform='WRITE'
         endif ! rxap
        else
          call wrndie(0,whoami,trim('NO FILE NAME GIVEN. WILL NOT WRITE REPLICA EXCHANGE LOG.'))
        endif ! rex_flen.gt.0
 6723 format(A,' WILL WRITE REPLICA EXCHANGE LOG TO FILE ',/, &
     & A,' ',A,'.DAT')
!
      endif ! replica exchange log
!cccccccccccccccccc process forces output options cccccccccccccccccc
      if (indxa(comlyn, comlen, 'FORC').gt.0) then
! get nergy file name
        call gtrmwa(COMLYN, COMLEN, 'FCNM', 4, forces_fname, 80, forces_flen)
!ccccccccccc print summary
        if (forces_flen.gt.0) then
         output_forces=.true.
         if (qprint) then
          write(info,6625) whoami,forces_fname(1:forces_flen)
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
         if (indxa(comlyn, comlen, 'FCAP').gt.0) then ! APPEND?
           fform='APPEND'
         else
           fform='WRITE'
         endif
        else
         call wrndie(0,whoami,trim('NO FILE NAME GIVEN. WILL NOT WRITE AVERAGE FORCE.'))
        endif
 6625 format(A,' WILL WRITE AVERAGE FORCE TO FILE ',A,'.')
      endif ! forces
!ccccccccccccccccc process M matrix output options ccccccccccccccccc
      if (indxa(comlyn, comlen, 'MMAT').gt.0) then
!
        call gtrmwa(COMLYN, COMLEN, 'MNAM', 4, M_fname, 80, M_flen)
        if (M_flen.gt.0) then
         output_M=.true.
         if (qprint) then ; write(info,6626) whoami,M_fname(1:M_flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
        else
         call wrndie(0,whoami,trim('NO FILE NAME GIVEN. WILL NOT WRITE M TENSOR.'))
        endif
 6626 format(A,' WILL WRITE SHORT-TERM AVERAGED TENSOR M TO FILE ',A,'.')
      endif
!

      if (qprint) iolev=oldiol

! if we got this far, we are probably OK
      stat_initialized=.true.
!
      end subroutine smcv_stat_init
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_stat()
! output statistics for SMCV
      use sm_var
      use sm_config, only: calc_cv_para, calc_Mtensor_para , izero, ione, itwo, ithree, ifour
      use cv_common
      use stream
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use string
      use mpi ! deal with other platforms later (never)
      use number
      use parallel, only: psnd4, psnd8
!
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
      integer :: ierror, i, ifile, fmt_r_len
      real(chm_real) :: rmsd0, rmsd0_all(nstring), dsdt, dsdt_all(nstring)
      real(chm_real) :: mywork, allwork(nstring)
      character(len=8) :: work_tags(nstring)
      character(len=80) :: fmt, fmt_real, fmt_int ! format strings for output
      integer :: oldiol
      character(len=len("SMCV_STAT>") ),parameter::whoami="SMCV_STAT>";!macro
!
      logical :: qroot, qprint, qgrp
!
      interface
       subroutine smcv_init(maxcv)
        implicit none
        integer, optional :: maxcv
       end subroutine smcv_init
      end interface
!
      qroot=MPI_COMM_STRNG.ne.MPI_COMM_NULL
      qprint=qroot.and.ME_STRNG.eq.0
      qgrp=MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1
!
! ad hoc fix for REX
      if (qprint) then ; oldiol=iolev; iolev=0; endif
!cccccccccccccccccccccc begin ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! check if the user has made an initialization call
      if (.not.smcv_initialized) call smcv_init()
      if (.not.stat_initialized) then
       call wrndie(0,whoami,trim('NO OUTPUT OPTIONS SELECTED. NOTHING DONE'))
       return
      endif
!
      stat_iteration_counter=stat_iteration_counter+1
! define number format string for output
!
      write(fmt_real,*) nstring
      fmt_r_len=len(fmt_real)
      call trima(fmt_real, fmt_r_len)
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_rmsd0) then
! weighting is taken care of in cv_common class
          rmsd0=cv_common_rmsd(ione,ithree) ! we will keep the initial reference coords in col. 3
          if (qroot) call mpi_gather(rmsd0,1,mpifloat & ! heads communicate
     & ,rmsd0_all,1,mpifloat,0, &
     & MPI_COMM_STRNG, ierror)
          if (qprint) then ! root writes
           if (rmsd0_funit.eq.outu) then
            fmt='("RMSD0> ",I5," ",'//fmt_real(1:fmt_r_len)//'F15.5)'
           else
            rmsd0_funit=-1
            call open_file(rmsd0_funit, rmsd0_fname, 'FORMATTED', rform)
            fmt='(I5," ",'//fmt_real(1:fmt_r_len)//'F15.5)'
           endif
           write(rmsd0_funit,fmt) stat_iteration_counter, &
     & (rmsd0_all(i),i=1,nstring)
!
           if (rmsd0_funit.ne.outu) then
            call VCLOSE(rmsd0_funit, 'KEEP', ierror)
           endif
          endif ! qprint
          rform='APPEND'
      endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_dsdt) then
! weighting is taken care of in cv_common class
          dsdt=cv_common_rmsd(ione,itwo) ! we will keep the previous coords in col. 2
          if (qroot) call mpi_gather(dsdt,1,mpifloat &
     & ,dsdt_all,1,mpifloat,0, &
     & MPI_COMM_STRNG, ierror)
          if (qprint) then ! root writes
           if (dsdt_funit.eq.outu) then
            fmt='("DLEN> ",I5," ",'//fmt_real(1:fmt_r_len)//'F15.5)'
           else
            dsdt_funit=-1
            call open_file(dsdt_funit, dsdt_fname, 'FORMATTED', dform)
            fmt='(I5," ",'//fmt_real(1:fmt_r_len)//'F15.5)'
           endif
           write(dsdt_funit,fmt) stat_iteration_counter, &
     & (dsdt_all(i),i=1,nstring)
! flush unit: close and reopen
           if (dsdt_funit.ne.outu) then
            call VCLOSE(dsdt_funit, 'KEEP', ierror)
           endif
          endif ! qprint
          dform='APPEND'
      endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_rmsd_ave) then ! rmsd with respect to the running average structure
! weighting is taken care of in cv_common class
! call to update running average
          call cv_common_update_ave()
          rmsd0=cv_common_rmsd(ione,ifour) ! we will keep the average coords in col. 4
          if (qroot) call mpi_gather(rmsd0,1,mpifloat &
     & ,rmsd0_all,1,mpifloat,0, &
     & MPI_COMM_STRNG, ierror)
          if (qprint) then ! root writes
           if (rmsd_ave_funit.eq.outu) then
            fmt='("RMSD_AVE> ",I5," ",'//fmt_real(1:fmt_r_len)//'F15.5)'
           else
            rmsd_ave_funit=-1
            call open_file(rmsd_ave_funit, rmsd_ave_fname, 'FORMATTED', raform)
            fmt='(I5," ",'//fmt_real(1:fmt_r_len)//'F15.5)'
           endif
           write(rmsd_ave_funit,fmt) stat_iteration_counter, &
     & (rmsd0_all(i),i=1,nstring)
! flush unit: close and reopen
           if (rmsd_ave_funit.ne.outu) then
            call VCLOSE(rmsd_ave_funit, 'KEEP', ierror)
           endif
          endif ! qprint
          raform='APPEND'
      endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_arclength) then
       if (qprint) then
         write(fmt_int,'(I5)') stat_iteration_counter
         if (s_funit.eq.outu) then
          fmt='("ARCL> '//fmt_int(1:5)//' ",'                           &
     & //fmt_real(1:fmt_r_len)//'F15.5)'
         else
          s_funit=-1
          call open_file(s_funit, s_fname, 'FORMATTED', sform)
          fmt='("'//fmt_int(1:5)//' ",'//fmt_real(1:fmt_r_len)//'F15.5)'
         endif
         call cv_common_print_ds(s_funit, fmt)
! flush unit: close and reopen
         if (s_funit.ne.outu) then
          call VCLOSE(s_funit, 'KEEP', ierror)
         endif
! done
       endif ! qprint
       sform='APPEND'
      endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_curvature) then
       if (qprint) then
         write(fmt_int,'(I5)') stat_iteration_counter
         if (c_funit.eq.outu) then
          fmt='("CURV> '//fmt_int(1:5)//' ",'                           &
     & //fmt_real(1:fmt_r_len)//'F11.5)'
         else
          c_funit=-1
          call open_file(c_funit, c_fname, 'FORMATTED', cform)
          fmt='("'//fmt_int(1:5)//' ",'//fmt_real(1:fmt_r_len)//'F11.5)'
         endif
         call cv_common_print_curvature(c_funit, fmt)
! flush unit: close and reopen
         if (c_funit.ne.outu) then
          call VCLOSE(c_funit, 'KEEP', ierror)
! done
         endif
       endif ! qprint
       cform='APPEND'
      endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_fe) then
       if (qprint) then
         write(fmt_int,'(I5)') stat_iteration_counter
         if (fe_funit.eq.outu) then
          fmt='("FE> '//fmt_int(1:5)//' ",'                             &
     & //fmt_real(1:fmt_r_len)//'F15.5)'
         else
          fe_funit=-1
          call open_file(fe_funit, fe_fname, 'FORMATTED', feform)
          fmt='("'//fmt_int(1:5)//' ",'//fmt_real(1:fmt_r_len)//'F15.5)'
         endif
         call cv_common_print_feav(fe_funit, fmt)
! flush unit: close and reopen
         if (fe_funit.ne.outu) then
          call VCLOSE(fe_funit, 'KEEP', ierror)
! done
         endif
       endif ! qprint
       feform='APPEND'
      endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_work) then
! gather work
       mywork=cv_common_neq_get_work()
! if running in parallel, need to reduce work from slave nodes
       if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1 &
     & .and.calc_cv_para) then
        call MPI_REDUCE(mywork, allwork(1), 1, mpifloat, &
     & MPI_SUM, 0, MPI_COMM_LOCAL, ierror) ! reduce on all group roots
        mywork=allwork(1)
       endif
! gather work from all nodes into one output buffer
       if (qroot) then
        call mpi_gather(mywork, 1, mpifloat, &
     & allwork, 1, mpifloat, &
     & 0, MPI_COMM_STRNG, ierror)
        call mpi_gather(work_tag, 8, MPI_BYTE, &
     & work_tags, 8, MPI_BYTE, 0, MPI_COMM_STRNG, &
     & ierror)
       endif
! write
       if (qprint) then
         write(fmt_int,'(I5)') stat_iteration_counter
         if (work_funit.eq.outu) then
          fmt='("WORK> '//fmt_int(1:5)//' ",A8,F15.5)'
         else
          work_funit=-1
          call open_file(work_funit, work_fname, 'FORMATTED', wkform)
          fmt='("'//fmt_int(1:5)//' ",A8,F15.5)'
         endif
         do i=1,nstring
          write(work_funit,fmt) work_tags(i), allwork(i)
         enddo
! flush unit: close and reopen
         if (work_funit.ne.outu) then
          call VCLOSE(work_funit, 'KEEP', ierror)
         endif
! done
       endif ! qprint
       wkform='APPEND'
      endif ! output_work
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_voronoi_hist) then ! output voronoi data
        if (voronoi_flen.eq.0) then
         call wrndie(0,whoami,trim('NO FILE NAME SPECIFIED. WILL NOT WRITE VORONOI DATA.'))
        else
         if (qprint) then
          ifile=-1
          voronoi_fname(voronoi_flen+1:voronoi_flen+4)='.dat'
          call open_file(ifile, voronoi_fname(1:voronoi_flen+4), 'FORMATTED', 'WRITE')
          voronoi_fname(voronoi_flen+1:)=''
         endif
         call cv_common_print_voro_data(ifile) ! all root processes enter
         if (qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ; endif
        endif
      endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_voronoi_log) then
       if (voronoi_flen.eq.0) then
         call wrndie(0,whoami,trim('NO FILE NAME SPECIFIED. WILL NOT WRITE VORONOI LOG.'))
       else
         if (qprint) then
           vlog_funit=-1
           voronoi_fname(voronoi_flen+1:voronoi_flen+4)='.log'
           call open_file(vlog_funit, voronoi_fname(1:voronoi_flen+4), 'UNFORMATTED', vlform)
           voronoi_fname(voronoi_flen+1:)=''
         endif
         vlform='APPEND'
         if (qroot) call cv_common_voronoi_print_log(vlog_funit)
! flush unit:
         if (qprint) then ; call VCLOSE(vlog_funit, 'KEEP', ierror) ; endif
       endif
      endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_voronoi_map) then ! output voronoi map
        if (voronoi_flen.eq.0) then
         call wrndie(0,whoami,trim('NO FILE NAME SPECIFIED. WILL NOT WRITE VORONOI MAP.'))
        else
! put 'whereami' into the map
          if (qroot.and.SIZE_STRNG.gt.1) then
           call MPI_ALLGATHER(cv%voronoi_whereami, 1, mpiint, &
     & cv%voronoi_map, 1, mpiint, MPI_COMM_STRNG, ierror)
          else
           cv%voronoi_map(mestring+1)=cv%voronoi_whereami
          endif
          if (qgrp) then
#if (KEY_INTEGER8==0)
           call PSND4(cv%voronoi_map,nstring) 
#endif
#if (KEY_INTEGER8==1)
           call PSND8(cv%voronoi_map,nstring) 
#endif

          endif ! qgrp
!
         if (qroot) then
          ifile=-1
          voronoi_fname(voronoi_flen+1:voronoi_flen+4)='.map'
          if (qprint) then
           call open_file(ifile, voronoi_fname(1:voronoi_flen+4), 'FORMATTED', 'WRITE')
           voronoi_fname(voronoi_flen+1:)=''
          endif
          call cv_common_print_voro_map(ifile)
          if (qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ; endif
         endif ! qroot
        endif
      endif
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_rex_map) then ! output replica exchange map
        if (rex_flen.eq.0) then
         call wrndie(0,whoami,trim('NO FILE NAME SPECIFIED. WILL NOT WRITE REPLICA EXCHANGE MAP.'))
        else
         if (qprint) then
          rex_funit=-1
          rex_fname(rex_flen+1:rex_flen+4)='.map'
          call open_file(rex_funit, rex_fname(1:rex_flen+4), 'FORMATTED', 'WRITE')
          rex_fname(rex_flen+1:)=''
!
          call cv_common_rex_print_map(rex_funit)
!
          call VCLOSE(rex_funit, 'KEEP', ierror)
         endif ! qprint
        endif
      endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_rex_log) then
       if (rex_flen.eq.0) then
         call wrndie(0,whoami,trim('NO FILE NAME SPECIFIED. WILL NOT WRITE REPLICA EXCHANGE LOG.'))
       else
        if (qprint) then
         rex_funit=-1
         rex_fname(rex_flen+1:rex_flen+4)='.dat' ! append to name
         call open_file(rex_funit, rex_fname(1:rex_flen+4), 'FORMATTED', rxlform)
         rex_fname(rex_flen+1:)='' ! erase extension
        endif
        rxlform='APPEND'
!
        call cv_common_rex_print_log(rex_funit)
! flush unit:
        if (qprint) then ; call VCLOSE(rex_funit, 'KEEP', ierror) ; endif
       endif
      endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_cv) then
        if (qprint) then
         cv_funit=-1
         call open_file(cv_funit, cv_fname, 'FORMATTED', cvform)
         write(cv_funit,'("% ",I8)') stat_iteration_counter ! % is a MATLAB comment
        endif
        cvform='APPEND'
        call cv_common_print_global(cv_funit)
! flush unit:
        if (qprint) then ; call VCLOSE(cv_funit, 'KEEP', ierror) ; endif
      endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_forces) then
       if (qprint) then
        forces_funit=-1
        call open_file(forces_funit, forces_fname, 'FORMATTED', fform);
        write(forces_funit,'("% ",I8)') stat_iteration_counter ! % is a MATLAB comment
       endif
       fform='APPEND'
       call cv_common_print_forces(forces_funit)
! flush unit:
       if (qprint) then ; call VCLOSE(forces_funit, 'KEEP', ierror) ; endif
      endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_wgt) then
       if (qprint) then
         wgt_funit=-1
         call open_file(wgt_funit, wgt_fname, 'FORMATTED', wgtform);
         write(wgt_funit,'("% ",I8)') stat_iteration_counter ! % is a MATLAB comment
         call cv_common_print_wgt(wgt_funit)
! flush unit:
         call VCLOSE(wgt_funit, 'KEEP', ierror)
       endif
       wgtform='APPEND'
      endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (output_M) then
! if running in parallel, combine partial M entries
       if (qgrp.and.calc_Mtensor_para) then
!
! print the short-term average (gather on root)
! broadcast all rows, but only num_cv columns
        call MPI_REDUCE(cv%M(1,1,1),cv%M(1,1,2),max_cv_common*cv%num_cv, &
     & mpifloat, MPI_SUM, 0, MPI_COMM_LOCAL, ierror)
       else ! qgrp
        cv%M(1:cv%num_cv,1:cv%num_cv,2)=cv%M(1:cv%num_cv,1:cv%num_cv,1)
       endif ! qgrp
!
       if (qroot) then
        if (M_flen.eq.0) then
         call wrndie(0,whoami,trim('NO FILE NAME SPECIFIED. WILL NOT WRITE M TENSOR.'))
        else
         ifile=-1
         call open_file(ifile, M_fname(1:M_flen), 'FORMATTED', 'WRITE')
         call cv_common_print_M_global(ifile,IND=2) ! print the short-term average (long-term average by default)
         call VCLOSE(ifile, 'KEEP', ierror)
        endif ! M_flen
       endif ! qroot
      endif ! output_M
! ad hoc fix for REX

      if (qprint) iolev=oldiol

!
      end subroutine smcv_stat
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine smcv_repa()
! this is routine is just a wrapper, so that any subroutine can call a "default" repa.
      use sm_var,only: interp_method, iterations, def, dst_cutoff, &
     & repa_initialized
      use cv_common,only: cv_common_repa
      use stream
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
!
      implicit none
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
! local variables
      character(len=len("SMCV_REPA>") ),parameter::whoami="SMCV_REPA>";!macro
      logical :: qprint
! check if the user has made an initialization call
!
      qprint=(MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0)
!
      if (.not.repa_initialized) then
       call wrndie(0,whoami,trim('NO REPARAMETRIZATION OPTIONS SELECTED. NOTHING DONE.'))
       return
      endif
      if (qprint) then ; write(info,690) whoami ; if(prnlev.ge. 5) write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 690 format(/A,' CALLING STRING REPARAMETRIZATION.')
      call cv_common_repa(interp_method,def,iterations,dst_cutoff)
!
      end subroutine smcv_repa
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine cv_set_ave_samples(n)
      use cv_common,only: cv_common_set_ave_samples
      use stream
      implicit none
      integer :: n
      call cv_common_set_ave_samples(n)
      end subroutine cv_set_ave_samples
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#endif
#endif /* automatically protect all code */
