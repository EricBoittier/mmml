! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
! finite-temperature string method
      module ftsm ! finite-temperature string method
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      use chm_kinds
      use ftsm_var
      use ftsm_voronoi
      use ftsm_compute
      use ftsmv2_compute
      use ftsm_stats
      use ftsm_io
      use ftsm_util
      use ftsm_rep
      use ftsm_min
      use ftsm_connect
      use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3 ! , only : RMSBestFit, rmsd
      implicit none
!
      private
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! SUBROUTINES
!
      public ftsm_parse
      public ftsm_main
      private ftsm_init
      private ftsm_done
      private ftsm_list_atoms
      private ftsm_test_grad_fd
      private ftsm_test_parallel
      private ftsm_evolve
      private ftsm_repl_exchange
!
      contains
!
      subroutine ftsm_parse(COMLYN,COMLEN)
      use ftsm_rex, only: ftsm_rex_set_temp
      use ftsm_voronoi, only: ftsm_voronoi_map, ftsm_voronoi_initialized, ftsm_voronoi_whereami
!----------------------------------------------------------------------
! command parser for the finite temperature string
!----------------------------------------------------------------------
      use stream
      use string
      use number
      use multicom_aux;
      use consta
      use mpi
      use select, only : selcta, selrpn, nselct; use psf
      use coord; use coordc
      use dimens_fcm
!
      use ctitla

        use dcntrl_mod, only : dynopt


! need BNBND, BIMAG for ftsm_mini
      use bases_fcm, only : BNBND, BIMAG
      use parallel, only: psnd4, psnd8

!======================================================================
      character(len=*) :: comlyn
      integer :: comlen
! local variables
 integer :: ivver, ivv2, iorig, ileap, error ! for dynamics
      integer :: klen
!
      character(len=len("FTSM>") ),parameter::whoami="FTSM>";!macro
      character(len=16) :: keyword
      character(len=80) :: fname
      real(chm_real) :: zval, k, step, voro_cut
      integer :: ifile, c1, c2, qcor, qdcd, flen, &
     & num_ave_samples, irep, i, a, &
     & ierror, me
      real(chm_real) :: t, omt
!

      integer :: oldiol



      real(chm_real), pointer :: fd_error(:,:)
!
      logical :: qroot, qslave, qprint, qcomp, voronoi_check_map, ok, &
     & qangstrom
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
      interface
      function sm_get_column(cmd_, l, qcoltag, missing) result(C)
       implicit none
       character(len=*) :: cmd_
       integer :: l, missing
       logical :: qcoltag
       integer :: C
      end function sm_get_column
      end interface
!
      keyword=nexta8(comlyn,comlen)
!
      qroot=MPI_COMM_STRNG.ne.MPI_COMM_NULL
      qslave=((MPI_COMM_LOCAL.ne.MPI_COMM_NULL).and.SIZE_LOCAL.gt.1)
      qprint=qroot.and.ME_STRNG.eq.0
!
! check for smcv initialization; quit if initialized
      if (smcv_initialized) then
       call wrndie(0,whoami,trim(' SMCV IS ON AND CANNOT BE USED WITH FTSM. NOTHING DONE.'))
       return
      endif
!===================================================================
      if (( keyword(1:4).eq.'INIT'(1:4) )) then
!
! NOTE : ftsm_init does not get command line, so will process here before calling ftsm_init
!
       if ((indxa(comlyn, comlen, 'VER1').gt.0).or.(indxa(comlyn, comlen, 'V1').gt.0)) then
        if (qprint) then
         write(info,'(2A)') whoami, 'FTSM VERSION 1 WILL BE USED TO SAMPLE HYPERPPLANES.'
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif
        qver2=.false. ! use ver2 unless v1 requested
       elseif ((indxa(comlyn, comlen, 'VER2').gt.0).or.(indxa(comlyn, comlen, 'V2').gt.0)) then
        qver2=.true.
       endif
!===== ! check for distance scaling
       ftsm_scaledist = .not.(indxa(comlyn, comlen, 'NOSC').gt.0) ! distance scaling is on by default
!===== for now, unscaled distances are only supported in version 2
       if ( (.not.qver2) .and. (.not.ftsm_scaledist) ) then ! version 1 must have ftsm_scaledist=1
         call wrndie(0,whoami,trim(' ABSOLUTE (UNSCALED) DISTANCES ARE ONLY SUPPORTED IN FTSM VERSION 2. TURNING SCALING ON.'))
         ftsm_scaledist=.true.
       endif
!
       if (.not.ftsm_scaledist.and.qprint) then
        write(info,'(2A)') whoami, ' USING ABSOLUTE (UNSCALED) DISTANCE UNITS.'
        write(OUTU,'(A)') pack(info,info.ne.'');info='';
       endif
!
       if (indxa(comlyn, comlen, 'NOCO').gt.0) then
        if (qprint) then
         write(info,'(2A)') whoami, ' USING FTSM IN ATOMIC CARTESIAN COORDINATES.'
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif
        ftsm_com_on=.false.
       elseif (indxa(comlyn, comlen, 'COM').gt.0) then
        ftsm_com_on=.true.
       endif
!
       if (ftsm_com_on.and.qprint) then
         write(info,'(2A)') whoami, ' USING FTSM IN CENTER-OF-MASS COORDINATES. (M-TENSOR/JACOBIAN CALCULATIONS ARE NOT IMPLEMENTED).'
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
       endif
!
       call ftsm_init()
       return
      endif
!===================================================================
      if (.not.ftsm_initialized) then
        call ftsm_init()
      endif
!===================================================================
      if (( keyword(1:4).eq.'DONE'(1:4) )) then
        call ftsm_done()
!===================================================================
      elseif (( keyword(1:3).eq.'ADD'(1:3) )) then
       if (ftsm_com_on) then
        call ftsm_add_com(comlyn, comlen)
       else
        call wrndie(0,whoami,trim('"ADD" COMMAND ONLY WORKS WITH FTSM IN COM COORDINATES.'))
       endif
!===================================================================
      elseif (( keyword(1:3).eq.'CLEA'(1:3) )) then
       if (ftsm_com_on) then
        call ftsm_clear_com(comlyn, comlen)
       else
        call wrndie(0,whoami,trim('"CLEAR" COMMAND ONLY WORKS WITH FTSM IN COM COORDINATES.'))
       endif
!===================================================================
      elseif (( keyword(1:4).eq.'REPA'(1:4) )) then
       if (comlen.gt.0) then ! this is an initialization call!
        call ftsm_repa_init(comlyn, comlen)
       else
        if (ftsm_check(qorient)) call ftsm_repa()
       endif
!===================================================================
      elseif (( keyword(1:4).eq.'MINI'(1:4) )) then
       if (ftsm_com_on) then
        call wrndie(0,whoami,trim('IMAGE MINIMIZATION IS NOT SUPPORTED FOR FTSM IN COM COORDINATES.'))
       else
        if (comlen.gt.0 .or. .not. ftsm_mini_initialized ) then ! this is an initialization call
         call ftsm_mini_init(comlyn, comlen)
        else
         if (ftsm_check(qorient)) then
          if (qorient.and.any(x(iatom_o(1:norient)).eq.anum)) then
          call wrndie(0,whoami,trim('MAIN ORIENTATION X-SET HAS UNDEFINED VALUES. NOTHING DONE.'))
          else
           call ftsm_mini(x(1:natom), y(1:natom), z(1:natom) &
     & ,WMAIN(1:natom), BNBND, BIMAG &
     & )
          endif ! qorient
         endif ! ftsm_check
        endif ! comlen
       endif ! ftsm_com_on
!===================================================================
      elseif (( keyword(1:4).eq.'RECO'(1:4) )) then
       if (comlen.gt.0 .or. .not. ftsm_connect_initialized ) then ! process options or initialize
        if (.not.((associated(r_o).and.associated(orientWeights).or..not.qorient).and.associated(r_f).and.associated(forcedWeights))) then ! splitting to reduce line length
        call wrndie(0,whoami,trim('SOME FTSM COORDINATES NOT DEFINED. USE "FILL" or "READ". NOTHING DONE.')); return;
        endif
        call ftsm_reconnect_parse(comlyn, comlen)
       else
        if (ftsm_check(qorient)) call ftsm_reconnect(-ione) ! pass phony time
       endif
!===================================================================
      elseif (( keyword(1:4).eq.'ORIE'(1:4) )) then
! perform a best-fit orientation of all instantaneous coordinates using
! specified coordinates for orientation;
! all replica coordinates will be modified; the first replica
! will have its COM removed (computed using selected atoms);
! and the other replicas will in addition be each superposed onto the
! previous replica. I note that this function belongs more naturally
! with the ZTS, and may be moved there.
!
! this _may_ be useful for ftsm without bestfit orientation
       call ftsm_orient(comlyn, comlen)
!===================================================================
      elseif (( keyword(1:4).eq.'STAT'(1:4) )) then
       if (comlen.gt.0) then ! this is an initialization call!
        call ftsm_stat_init(comlyn, comlen)
       else
        call ftsm_stat()
       endif
!===================================================================
      elseif (( keyword(1:4).eq.'DYNA'(1:4) )) then
!ccccc will assume that other distributions specify dynamics elsewhere ccccccc
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
!================= CHECK THAT COORDINATE ARRAYS AND WEIGHTS HAVE BEEN ALLOCATED
       if (.not.((associated(r_o).and.associated(orientWeights).or..not.qorient).and.associated(r_f).and.associated(forcedWeights))) then
         call wrndie(0,whoami,trim('SOME FTSM COORDINATES NOT DEFINED. USE "FILL" or "READ". NOTHING DONE.')); return;
       endif
!===================== PARSE DYNAMICS OPTIONS
! code from SMCV
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
        if (.not.ftsm_voronoi_initialized) call ftsm_voronoi_init()
! standard V. calculation case -- no crossing
        compute_whereami=.false.
        if (.not.voronoi_allow_cross) then
! create standard map (unless map is present)
         if (all(ftsm_voronoi_map.eq.-1)) then
          ftsm_voronoi_map=(/ (i, i=1, nstring) /)
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
         call ftsm_voronoi_whereami_compute(x,y,z)
!
         if (all(ftsm_voronoi_map.ne.-1)) then ! does the map have valid entries
           me=ftsm_voronoi_map(mestring+1)
! compare me and whereami:
           if (qroot) then
            if(SIZE_STRNG.gt.1) then
             call MPI_ALLREDUCE(me.eq.ftsm_voronoi_whereami, ok, &
     & 1, mpibool, MPI_LAND, MPI_COMM_STRNG, ierror)
            else
             ok=me.eq.ftsm_voronoi_whereami
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
         endif ! ftsm_voronoi_map.ne.-1
!
        else
         ftsm_voronoi_whereami=ftsm_voronoi_map(mestring+1)
        endif ! voronoi_check_map
!
       endif ! voronoi_hist_on
! reset internal interation counter for ftsm_master
       olditeration=0
!
       update_on=(indxa(comlyn, comlen, 'UPDA').gt.0)
       if (update_on) then
        update_freq=gtrmi(comlyn, comlen, 'UPDF', 0)
        if (update_freq.le.0) then
         call wrndie(0,whoami,trim('UPDATE FREQUENCY INVALID OR UNSPECIFIED. WILL NOT UPDATE.'))
         update_on=.false.
        else
         repa_on=(indxa(comlyn, comlen, 'REPA').gt.0)
         ftsm_mini_on=(indxa(comlyn, comlen, 'MINI').gt.0)
         ftsm_reconnect_on=(indxa(comlyn, comlen, 'RECO').gt.0)
        endif
       endif
!
       stat_on=(indxa(comlyn, comlen, 'STAT').gt.0)
       if (stat_on) then
        stat_freq=gtrmi(comlyn, comlen, 'STAF', 0)
        if (stat_freq.le.0) then
         call wrndie(0,whoami,trim('STATISTICS FREQUENCY INVALID OR UNSPECIFIED.'))
         stat_on=.false.
        endif
       endif ! stat_on
!
       evolve_ftsm_on=(indxa(comlyn, comlen, 'EVOL').gt.0)
       if (evolve_ftsm_on) then
        evolve_freq=gtrmi(comlyn, comlen, 'EVOF', 0)
        if (evolve_freq.le.0) then
         call wrndie(0,whoami,trim('EVOLUTION FREQUENCY INVALID OR UNSPECIFIED. WILL NOT EVOLVE.'))
         evolve_ftsm_on=.false.
        endif
       endif ! evolve_ftsm_on
!
       if (evolve_ftsm_on) then ! still on (see above)
        evolve_nskip=gtrmi(comlyn, comlen, 'EVOS', 0)
!
! ----- types of evolution
!
        evolve_expo_on=(indxa(comlyn, comlen, 'EXPO').gt.0) ! use exponential convolution
        if (evolve_expo_on) then
         evolve_expo_mem=gtrmf(comlyn, comlen, 'MEMO', 0.999d0)
        endif
!
        evolve_aver_on=(indxa(comlyn, comlen, 'AVER').gt.0) ! r_ref=mean(r_inst)
        if (evolve_aver_on) then
         num_evolve_samples=0
         max_evolve_samples=0
! setting this large will dampen initial fluctuations
         if (indx(comlyn, comlen, 'NAVE', 4).gt.0) then
          num_ave_samples=gtrmi(comlyn, comlen, 'NAVE', -1)
          if (num_ave_samples.ge.0) then
            num_evolve_samples=num_ave_samples
          else
           call wrndie(0,whoami,trim('INVALID NUMBER OF SAMPLES SPECIFIED. WILL SET TO ZERO.'))
          endif ! num_samples
         endif ! NAVE
!
         if (indx(comlyn, comlen, 'MAXAVE', 6).gt.0) then
          num_ave_samples=gtrmi(comlyn, comlen, 'MAXAVE', -1)
          if (num_ave_samples.gt.0) then
            max_evolve_samples=num_ave_samples
          else
  call wrndie(0,whoami,trim('INVALID MAXIMUM NUMBER OF SAMPLES SPECIFIED. WILL SET TO ZERO.'))
          endif ! num_samples
         endif ! MAXAVE
        endif ! evolve_aver
!
        i=0
        if (evolve_expo_on) i=i+1
        if (evolve_aver_on) i=i+1
!
        if (i.gt.1) then
         call wrndie(0,whoami,trim('MORE THAN ONE EVOLUTION SCHEME REQUESTED. WILL USE EXPO.'))
         evolve_expo_on=.true.
         evolve_aver_on=.false.
        endif
!
        if (i.eq.0) then
         call wrndie(0,whoami,trim('EVOLUTION SCHEME UNSPECIFIED. WILL USE EXPO.'))
         evolve_expo_on=.true.
         evolve_aver_on=.false.
        endif
       endif ! evolve_ftsm_on
!
       if (update_on.and..not.(evolve_ftsm_on.or.repa_on.or.ftsm_mini_on)) then
        call wrndie(0,whoami,trim('EVOLUTION, REPARAMETRIZATION AND MINIMIZATION ARE ALL DISABLED. UPDATE IS OFF.'))
        update_on=.false.
       endif
!
       restrained_on=(indxa(comlyn, comlen, 'RSTR').gt.0)
       if (restrained_on) then
        restrained_eq_steps=gtrmi(comlyn, comlen, 'REEQ', 0)
        if (restrained_eq_steps.lt.0) then
          call wrndie(0,whoami,trim('REEQ CANNOT BE NEGATIVE. WILL SET TO ZERO.'))
          restrained_eq_steps=0
        endif
        restrained_eq0=0
!
! scale force constants and restraint reference values (if necessary)
        if (.not.ftsm_scaledist.or. &
& qkpara_angstrom.or.qkperp_angstrom.or.qdperp_angstrom.or.qdperpf_angstrom) then
         call ftsm_compute_arcl_curv(linear)
!writE(0,*) 'DARCL : ', ds ! aa
         zval = sum(ds) / ( max( nstring-1 , 1 ) ) * sqrt3 ; ! sqrt(3) makes the arclength correspond to the RMSD in 3D
! broadcast zval to slaves
         if (qslave) then
          call mpi_bcast(zval,1,mpifloat,0,MPI_COMM_LOCAL,ierror)
         endif
!
         if (.not.ftsm_scaledist) then ! need to compute reference distance from string length
          if (mestring.eq.0) then ; dpar0=zero ; else ; dpar0 = zval ; endif ; dpar0i=dpar0 ! left endpoint uses one sided tangent (see ftsm14 paper)
          if (qprint) then ; write(info,6713) whoami, ftoa(zval)
          write(OUTU,'(A)') pack(info,info.ne.'');info=''; ;
          endif
 6713 format(A,' AVERAGE COMPUTED ARCLENGTH INCREMENT IS ',A,' ANG.')
          ftsm_reset_dpar0f=.true.
!
         else ! options valid for scaled version
          zval=two*zval ! gives 2L/(M-1) , i.e. || q-p ||
!
          if (qkpara_angstrom) then
           fname='PARALLEL FORCE CONSTANT (KPAR)' ; kpara = kpara * zval**2 ;
           if (qprint) then ; write(info,6712) whoami, trim(fname), ftoa(zval), ftoa(kpara)
           write(OUTU,'(A)') pack(info,info.ne.'');info=''; ;
           endif
           qkpara_angstrom=.false. ! make sure no additional scaling will occur if there are multiple runs per initialization
          endif
          if (qkperp_angstrom) then
           fname='PERPENDICULAR FORCE CONSTANT (KPRP)' ; kperp = kperp * zval**2 ;
           if (qprint) then ; write(info,6712) whoami, trim(fname), ftoa(zval), ftoa(kperp)
           write(OUTU,'(A)') pack(info,info.ne.'');info='';
           endif
           qkperp_angstrom=.false.
          endif
          if (qdperp_angstrom) then
           fname='REFERENCE TUBE WIDTH (DPRP)' ; dperp0 = dperp0 / zval ; dperp0i = dperp0i / zval ;
           if (qprint) then ; write(info,6712) whoami, trim(fname), ftoa(zval), ftoa(dperp0)
           write(OUTU,'(A)') pack(info,info.ne.'');info='';
           endif
           qdperp_angstrom=.false.
          endif
          if (qdperpf_angstrom) then
           fname='FINAL REFERENCE TUBE WIDTH (DPRPF)' ; dperp0f = dperp0f / zval ;
           if (qprint) then ; write(info,6712) whoami, trim(fname), ftoa(zval), ftoa(dperp0f)
           write(OUTU,'(A)') pack(info,info.ne.'');info='';
           endif
           qdperpf_angstrom=.false.
          endif
         endif ! .not. ftsm_scaledist
!
 6712 format(A,' SCALING ',A,' BY 2 x INTER-REPLICA DS (',A,' ANG). NEW VALUE IS ',A)
        endif ! ftsm_scaledist .or. ...
!
       endif ! if restrained_on
!
! VO 4/14 : taken from SMCV to allow transient unrestrained dynamics for more aggressive path exploration
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
!
       repl_x_on=(indxa(comlyn, comlen, 'REX').gt.0)
       if (repl_x_on) then
        repl_x_freq=gtrmi(comlyn, comlen, 'REXF', 0)
        repl_x_temp=gtrmf(comlyn, comlen, 'REXT', 0d0)
!
        if (repl_x_freq.le.0) then
          call wrndie(0,whoami,trim('MUST SPECIFY POSITIVE REXF. REPLICA EXCHANGE IS OFF.'))
          repl_x_on=.false.
        elseif (repl_x_temp.le.0) then
          call wrndie(0,whoami,trim('MUST SPECIFY POSITIVE REXT. REPLICA EXCHANGE IS OFF.'))
          repl_x_on=.false.
        else
          call ftsm_rex_set_temp(repl_x_temp)
        endif
       endif ! repl_x_on
!
       if (update_on.or.repl_x_on) then ! decrease output
         string_noprint=(indxa(comlyn, comlen, 'NOPR').gt.0)
       endif
!--------------- DONE PARSING DYNAMICS OPTIONS -----
! print summary
!cccccccccccccccccc STRING METHOD OPTIONS cccccccccccccccccccccc
       if (qprint) then
        WRITE (info,'(2A)') whoami, ' FINITE TEMPERATURE STRING METHOD ENABLED.'; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        if (evolve_ftsm_on) then
            WRITE (info,'(/,2A,/,2A,I7,A)') &
     & whoami, ' STRING EVOLUTION ENABLED.', &
     & whoami, ' WILL EVOLVE AFTER EVERY ', &
     & evolve_freq,' ITERATIONS.'; write(OUTU,'(A)') pack(info,info.ne.'');info='';
            if (update_on.and.evolve_nskip.gt.0) then
             WRITE (info,'(2A,I7,A)') &
     & whoami, ' THE FIRST', evolve_nskip, &
     & ' ITERATIONS AFTER IMAGE UPDATE WILL NOT CONTRIBUTE TO STRING EVOLUTION.';
             write(OUTU,'(A)') pack(info,info.ne.'');info='';
            endif
            if (evolve_expo_on) then
               write(info,671) whoami, whoami, evolve_expo_mem ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 671 format(A,' STRING EVOLUTION WILL BE OF THE FORM:',/, &
     & A,' R(N+1)=A*R(N)+(1-A)*RINST, A=',F9.5,'.')
            elseif (evolve_aver_on) then
               write(info,6710) whoami, whoami, num_evolve_samples ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6710 format(A,' CV EVOLUTION WILL BE OF THE FORM:',/, &
     & A,' R(N)=AVERAGE_1^{N}(RINST).  INITIAL NUMBER OF SAMPLES IS ', &
     & I9,'.')
             if (max_evolve_samples.gt.0) &
     & write(info, 6711) whoami, max_evolve_samples ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6711 format(A, ' ONLY THE MOST RECENT ', I9,' SAMPLES WILL BE USED.')
            endif
        endif ! evolve_ftsm_on
!===================================================================
        if (update_on) then
          WRITE (info,666) whoami, update_freq ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 666 format(A,' WILL UPDATE IMAGES AFTER EVERY ',I7,' ITERATIONS.')
         if (ftsm_mini_on) then
 669 format(A,' WILL MINIMIZE STRING DURING UPDATE ')
          WRITE (info,669) whoami
          write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif ! mini
         if (repa_on) then
 667 format(A,' WILL REPARAMETRIZE STRING DURING UPDATE ')
          WRITE (info,667) whoami
          write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
        endif ! update_on
!===================================================================
        if (restrained_on) then
         WRITE (info,'(2A)') whoami, ' WILL USE RESTRAINED DYNAMICS.'
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
!
         if (unrestrained_on) then
          WRITE (info,'(2A)') whoami, ' WILL USE UNRESTRAINED (EXPLORATION) DYNAMICS.'
          write(OUTU,'(A)') pack(info,info.ne.'');info='';
          unrestrained_eq0=0
          WRITE (info,664) whoami, unrestrained_eq_steps ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 664 format(A,' WILL EQUILIBRATE UNDER CV RESTRAINTS FOR ', I7, ' STEPS.')
         endif ! unrestrained_on
!
         if (restrained_eq_steps.gt.0) then
          write(info,665) whoami, restrained_eq_steps
          write(OUTU,'(A)') pack(info,info.ne.'');info='';
 665 format(A, ' WILL ADJUST TO NEW RESTRAINTS OVER ',I11, ' STEPS.')
         endif
! proj_on is necessary to compute free energies
         if (proj_on) then
          write (info,'(2A)') whoami,' WILL RESTRAIN SYSTEM TO PLANE PERPENDICULAR TO PATH'
          zval=kpara
         else
          write (info,'(2A)') whoami,' WILL RESTRAIN SYSTEM TO PATH IMAGE (FE/MFPT CANNOT BE COMPUTED)'
          zval=krms
         endif ! proj_on
          if (ftsm_scaledist) then
            write(info(2),'(2A)') whoami, ' WITH HARMONIC FORCE CONSTANT K='//ftoa(zval)//' KCAL/MOL (SCALED DISTANCE)'
          else
            write(info(2),'(2A)') whoami, ' WITH HARMONIC FORCE CONSTANT K='//ftoa(zval)//' KCAL/MOL/ANG^2'
          endif
        write(OUTU,'(A)') pack(info,info.ne.'');info='';
        else
!
        endif !restrained
!===================================================================
        if (stat_on) then
            write(info,668) whoami, stat_freq ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 668 format(A, ' WILL OUTPUT STRING STATISTICS AFTER EVERY ',I7, ' STEPS.')
        endif
!===================================================================
        if (repl_x_on) then
            write(info,691) whoami, whoami, repl_x_freq, repl_x_temp
            write(OUTU,'(A)') pack(info,info.ne.'');info='';
 691 format(A, ' WILL ATTEMPT TO EXCHANGE NEIGHBORING REPLICAS ',/ &
     & A, ' ONCE IN EVERY ',I6,' ITERATIONS AT ',F11.3, ' K.')
        endif ! repl_x_on
!===================================================================
        if (voronoi_hist_on) then
            write(info,670) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 670 format(A, ' WILL COMPUTE FREE ENERGY ALONG STRING USING VORONOI TESSELLATION.' )
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
         if (update_on) then
          if (.not.voronoi_allow_cross) then
           write(info,'(2A,/2A)') &
     & whoami, ' STRING UPDATE DURING VORONOI FE COMPUTATION IS', &
     & whoami, ' EXPERIMENTAL AND SHOULD BE USED WITH CAUTION ' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          else
           write(info,'(2A,/2A,/2A)') &
     & whoami, ' STRING CANNOT BE UPDATED ON THE FLY', &
     & whoami, ' IF VORONOI CELL CROSSING IS ALLOWED.', &
     & whoami, ' VORONOI CELL CROSSING WILL BE OFF.' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
           voronoi_allow_cross=.false.
          endif ! voronoi_allow_cross
         endif ! update_on
        endif ! voronoi_hist_on
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       endif ! qprint
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       call ftsm_save_com() ! recompute COM in case weights changed
       call ftsm_swap_bc(.true.) ! update boundary replicas (with new COM-free structures)
! turn on string for dynamics
       ftsm_on=.true.
       ftsm_ini_iteration = -ione ! will be computed by ftsm_main
! call dynamics parser
       call dynopt(comlyn, comlen)
!cccccc turn off string for regular dynamics ccccccc
       ftsm_on=.false.
       repa_on=.false. ! turn off after dynamics because SM0K also uses this flag; therefore a subsequent minimization would call reparametrization
!===================================================================================
!ccccccccccccccccccc ADDITIONAL VORONOI OPTIONS cccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'VORO'(1:4) )) then
! get voronoi command
       keyword=nexta8(comlyn,comlen)
       if (( keyword(1:4).eq.'VMAP'(1:4) )) then
        if (indxa(comlyn, comlen, 'CALC').gt.0) then
          if (qprint) then ; write(info,6010) whoami ;
          write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6010 format(A,' WILL CALCULATE VORONOI MAP FROM MAIN COORDINATES.')
          if (.not.((associated(r_o).and.associated(orientWeights).or..not.qorient).and.associated(r_f).and.associated(forcedWeights))) then
           call wrndie(0,whoami,trim('SOME FTSM COORDINATES NOT DEFINED. USE "FILL" or "READ". NOTHING DONE.')); return;
          endif
          call ftsm_voronoi_whereami_compute(x,y,z)
! put 'whereami' into the map
          if (qroot.and.SIZE_STRNG.gt.1) then
           call MPI_ALLGATHER(ftsm_voronoi_whereami, 1, mpiint, &
     & ftsm_voronoi_map, 1, mpiint, MPI_COMM_STRNG, ierror)
          else
           ftsm_voronoi_map(mestring+1)=ftsm_voronoi_whereami
          endif
          if (qslave) then
#if (KEY_INTEGER8==0)
          call PSND4(ftsm_voronoi_map,nstring) 
#endif
#if (KEY_INTEGER8==1)
          call PSND8(ftsm_voronoi_map,nstring) 
#endif


!
          endif
! print cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        elseif ((indxa(comlyn, comlen, 'PRIN').gt.0 ) .or. (indxa(comlyn, comlen, 'WRIT').gt.0)) then
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
            call ftsm_voronoi_print_map(ifile)
            if (qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ;

             iolev=oldiol

            endif
           else
            call wrndie(0,whoami,trim('FILE NAME NOT SPECIFIED. NOTHING DONE.'))
           endif ! flen
          endif ! qroot
! read ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        elseif (indxa(comlyn, comlen, 'READ').gt.0) then
          ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
          call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: FNAME will be UPPER CASE
          if (flen.gt.0) then
            if (qprint) then

             oldiol=iolev
             iolev=1

             call open_file(ifile, fname, 'FORMATTED', 'WRITE')
             write(info,6013) whoami, fname(1:flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
            endif
!
 6013 format(A,' READING VORONOI MAP FROM FILE ',A,'.')
            call ftsm_voronoi_read_map(ifile)
            if (qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ;

             iolev=oldiol

            endif
           else
            call wrndie(0,whoami,trim('FILE NAME NOT SPECIFIED. NOTHING DONE.'))
           endif ! flen
! clear ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        elseif (indxa(comlyn, comlen, 'CLEA').gt.0) then
           if (associated(ftsm_voronoi_map)) ftsm_voronoi_map=-ione
        endif ! 'CALC'
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif (( keyword(1:4).eq.'READ'(1:4) )) then
! read "restart" file that contains (1) crossing_attempt (2) crossing_accepts (3) occupancy
         ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
         call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: FNAME will be UPPER CASE
         if (flen.gt.0) then
          if (qprint) then

           oldiol=iolev
           iolev=1

           call open_file(ifile, fname, 'FORMATTED', 'WRITE')
           write(info,6014) whoami, fname(1:flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          endif
 6014 format(A,' READING VORONOI CROSSING DATA FROM FILE ',A,'.')
          call ftsm_voronoi_read_data(ifile)
          if (qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ;

           iolev=oldiol

          endif
         else
          call wrndie(0,whoami,trim('FILE NAME NOT SPECIFIED. NOTHING DONE.'))
         endif ! flen
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       elseif ( (( keyword(1:4).eq.'PRIN'(1:4) )) .or. (( keyword(1:4).eq.'WRIT'(1:4) ))) then
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
           call ftsm_voronoi_print_data(ifile)
           if (qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ;

            iolev=oldiol

           endif
         else
            call wrndie(0,whoami,trim('FILE NAME NOT SPECIFIED. NOTHING DONE.'))
         endif ! flen
       endif ! VMAP
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'FILL'(1:4) )) then ! compute ftsm coordinates from current instantaneous coordinates
!
       if (ftsm_com_on .and. (.not.((associated(r_o).and.associated(orientWeights).or..not.qorient).and.associated(r_f).and.associated(forcedWeights)))) call ftsm_coor_wgt_alloc(amass,natom)
!
       c1=sm_get_column(comlyn, comlen, qcoltag=.true., missing=center) ! look after 'COL'
!
       qcomp=(indxa(comlyn, comlen, 'COMP').gt.0)
!
       if (qcomp) then
        write(info,6657) whoami, c1
 6657 format(/A,' WILL CALCULATE FTSM COORDINATES IN COLUMN ',I3,' FROM COMPARISON COORDINATES.')
        call ftsm_fill(xcomp,ycomp,zcomp, c1)
       else ! ~qcomp -- use main coordinates
        write(info,6660) whoami, c1
 6660 format(/A,' WILL CALCULATE FTSM COORDINATES IN COLUMN ',I3,' FROM MAIN COORDINATES.')
        call ftsm_fill(x,y,z, c1)
       endif ! qcomp
       if (qprint) then
        write(OUTU,'(A)') pack(info,info.ne.'');info='';
       endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'CALC'(1:4) )) then ! calculate parallel and perpendicular distances
       if (ftsm_com_on .and. (.not.((associated(r_o).and.associated(orientWeights).or..not.qorient).and.associated(r_f).and.associated(forcedWeights)))) call ftsm_coor_wgt_alloc(amass,natom)
       if (.not.((associated(r_o).and.associated(orientWeights).or..not.qorient).and.associated(r_f).and.associated(forcedWeights))) then
        call wrndie(0,whoami,trim('SOME FTSM COORDINATES NOT DEFINED. USE "FILL" or "READ". NOTHING DONE.')); return;
       endif
!
       qcomp=(indxa(comlyn, comlen, 'COMP').gt.0)
!
       if (qcomp) then
        if (qprint) then ; write(info,6658) whoami, 'COMPARISON' ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
        if (qorient.and.any(xcomp(iatom_o(1:norient)).eq.anum)) then
         call wrndie(0,whoami,trim('COMPARISON ORIENTATION X-SET HAS UNDEFINED VALUES. NOTHING DONE.'))
        else
         if (qver2.and.proj_on) then ; call ftsmv2_calc(xcomp,ycomp,zcomp,.false., t=one); else ; call ftsmv1_calc(xcomp,ycomp,zcomp,.false., t=one); endif
        endif
       else ! ~qcomp -- use main coordinates
        if (qprint) then ; write(info,6670) whoami, 'MAIN' ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
        if (qorient.and.any(x(iatom_o(1:norient)).eq.anum)) then
         call wrndie(0,whoami,trim('MAIN ORIENTATION X-SET HAS UNDEFINED VALUES. NOTHING DONE.'))
        else
         if (qver2.and.proj_on) then ; call ftsmv2_calc(x,y,z, .false., t=one); else ; call ftsmv1_calc(x,y,z, .false., t=one); endif
        endif
       endif ! qcomp
 6670 format(/A,' WILL COMPUTE DISTANCE FROM STRING IMAGE USING ',A,' COORDINATES.')
!=======================================================================================
! the option below will be useful to perform usual charmm operations on the STRING (if desired)
      elseif (( keyword(1:4).eq.'LIFT'(1:4) )) then ! force string into current coordinates
       if (ftsm_com_on) then
        call wrndie(0,whoami,trim('LIFTING FROM STRING IS NOT SUPPORTED FOR FTSM IN COM COORDINATES.'))
        return
       else
!
        qcomp=(indxa(comlyn, comlen, 'COMP').gt.0)
!
        if (qcomp) then
         if (qprint) then ; write(info,6658) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6658 format(/A,' WILL COPY STRING INTO COMPARISON COORDINATES.')
         if (qorient.and.any(xcomp(iatom_o(1:norient)).eq.anum)) then
          call wrndie(0,whoami,trim('COMPARISON ORIENTATION X-SET HAS UNDEFINED VALUES. NOTHING DONE.'))
         else
          call ftsm_lift(xcomp,ycomp,zcomp)
         endif
        else ! ~qcomp -- use main coordinates
         if (qprint) then ; write(info,6661) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6661 format(/A,' WILL COPY STRING INTO MAIN COORDINATES.')
         if (qorient.and.any(x(iatom_o(1:norient)).eq.anum)) then
          call wrndie(0,whoami,trim('MAIN ORIENTATION X-SET HAS UNDEFINED VALUES. NOTHING DONE.'))
         else
          call ftsm_lift(x,y,z)
         endif
        endif ! qcomp
       endif ! ftsm_com_on
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'TEST'(1:4) )) then !
       if (.not.((associated(r_o).and.associated(orientWeights).or..not.qorient).and.associated(r_f).and.associated(forcedWeights))) then
        call wrndie(0,whoami,trim('SOME FTSM COORDINATES NOT DEFINED. USE "FILL" or "READ". NOTHING DONE.')); return;
       endif
       if (indxa(comlyn, comlen, 'GRAD').gt.0) then ! finite-difference gradient test
! check fd spec
        step=gtrmf(comlyn, comlen, 'STEP', finite_difference_d)
        if (qprint) then
         write(info, 7001) whoami,whoami,step,whoami,whoami
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
 7001 format(/A,' WILL TEST GRADIENTS USING FINITE DIFFERENCES', &
     & /A,' USING DX = DY = DZ = ',F15.9,'.', &
     & /A,' MAIN COORDINATE SET MUST BE DEFINED.', &
     & /A,' WILL OVERWRITE FORCE/GRAD ARRAYS')
        endif
        if (any(x.eq.anum)) then
         call wrndie(0,whoami,trim('MAIN X SET HAS UNDEFINED VALUES. NOTHING DONE.'))
        else
         fd_error=>ftsm_test_grad_fd(x,y,z,step)
! write(me_global+100,*) fd_error
! if (qprint) then
          if (proj_on) then
           write(info,7002) whoami, whoami, whoami, whoami
 7002 format(/A,'  TOP LINE:    NORMALIZED PROJECTION ONTO PATH', &
     & /A,'  BOTTOM LINE: DISTANCE PERPENDICULAR TO PATH', &
     & /A,'  DGRAD_X_MAX, DGRAD_Y_MAX, DGRAD_Z_MAX', &
     & /A,' ============================================================================')
           if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
           write(info(1),'(A," ============================================================================")') whoami
           write(info(2),'(A,"  STRING RANK:",I5,", LOCAL RANK:",I5,", GLOBAL RANK:",I5,"")') whoami, mestring, ME_LOCAL, ME_GLOBAL
           do i=1,2
            write(info(i+2),'(A,3'//real_format//'F15.9)')                   &
     & whoami,fd_error(i,:)
           enddo
           i=3
          else ! projection off
           write(info,7013) whoami, whoami, whoami ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
 7013 format(/A,'  DISTANCE TO PATH POINT:', &
     & /A,'  DGRAD_X_MAX, DGRAD_Y_MAX, DGRAD_Z_MAX', &
     & /A,' ============================================================================')
           write(info(1),'(A," ============================================================================")') whoami
           write(info(2),'(A,"  STRING RANK:",I5,", LOCAL RANK:",I5,", GLOBAL RANK:",I5,"" )') whoami, mestring, ME_LOCAL, ME_GLOBAL
           i=2
           write(info(i+1),'(A,3'//real_format//'F15.9)')                   &
     & whoami,fd_error(1,:)
          endif ! projection
!
! write(OUTU,'(A)') pack(info,info.ne.'');info='';
!
! endif ! qprint
! decide whether the test was passed
         zval=abs(maxval(fd_error))
         if (zval.lt.abs(step)*one) then
          write(info(i+2:i+3),7003) whoami, zval, whoami
         else
          write(info(i+2:i+3),7004) whoami, zval, whoami
          call wrndie(0,whoami,trim('FINITE DIFFERENCE TEST FAILED.'))
         endif ! report test result
!
         write(info(i+4),'(A," ============================================================================")') whoami
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
 7003 format(A, '  THE MAXIMUM GRADIENT ERROR IS ',F15.9,', ', &
     & /A, '  WHICH IS SMALLER THAN STEP. TEST PASSED.')
 7004 format(A, '  THE MAXIMUM GRADIENT ERROR IS ',F15.9,', ', &
     & /A, '  WHICH IS NO SMALLER THAN STEP. TEST FAILED.')
         if (associated(fd_error)) deallocate(fd_error) ! test_grad returns a pointer to an array of abs errors
        endif
       endif ! grad
!
       if (indxa(comlyn, comlen, 'PARA').gt.0) then ! parallel communication test
        if (qprint) write(info, 7005) whoami,whoami,whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 7005 format(/A,' WILL COMPARE PARALLEL AND SERIAL FORCE COMPUTATION', &
     & /A,' MAIN COORDINATE SET MUST BE DEFINED.', &
     & /A,' WILL OVERWRITE FORCE/GRAD ARRAYS')
        if (any(x.eq.anum)) then
         call wrndie(0,whoami,trim(' MAIN X SET HAS UNDEFINED VALUES. NOTHING DONE.'))
        else
         fd_error=>ftsm_test_parallel(x,y,z) ! use the same array as above
         if (qprint) then
          if (proj_on) then
           write(info,7006) whoami, whoami, whoami, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 7006 format(/A,'  TOP:    NORMALIZED PROJECTION ONTO PATH', &
     & /A,'  BOTTOM: DISTANCE PERPENDICULAR TO PATH', &
     & /A,'  DGRAD_X_MAX, DGRAD_Y_MAX, DGRAD_Z_MAX, VALUE', &
     & /A,' ============================================================================')
           do i=1, 2
            write(info,'(A,4'//real_format//'F15.9)')                   &
     & whoami,fd_error(i,:) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
           enddo
          else ! not proj_on
           write(info,7010) whoami, whoami, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 7010 format(/A,'  DISTANCE TO PATH POINT:', &
     & /A,'  DGRAD_X_MAX, DGRAD_Y_MAX, DGRAD_Z_MAX, VALUE', &
     & /A,' ============================================================================')
            write(info,'(A,4'//real_format//'F15.9)')                   &
     & whoami,fd_error(1,:) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          endif ! proj_on
         endif ! qprint
! decide whether the test was passed
         zval=abs(maxval(fd_error))
         if (zval.lt.parallel_tolerance) then
          write(info,7007) whoami, zval, whoami, parallel_tolerance ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         else
          write(info,7008) whoami, zval, whoami, parallel_tolerance ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          call wrndie(0,whoami,trim('PARALLEL COMPUTATION TEST FAILED.'))
         endif ! report test result
 7007 format(/A, ' THE MAXIMUM ERROR IS ',E12.5,', ', &
     & /A, ' WHICH IS SMALLER THAN ',E12.5,'. TEST PASSED.')
 7008 format(/A, ' THE MAXIMUM ERROR IS ',E12.5,', ', &
     & /A, ' WHICH IS NO SMALLER THAN ',E12.5,'. TEST FAILED.')
         if (associated(fd_error)) deallocate(fd_error) ! pointer to an array of abs errors
        endif
       endif ! para
!
! other tests will go below this line
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! specify parallel calculation options
      elseif (( keyword(1:4).eq.'PARA'(1:4) )) then
       do while (comlen .gt. 1)
        keyword=nexta8(comlyn,comlen)
        select case(keyword)
           case('YES','ON','TRUE','T','yes','on','true','t')
            keyword='ENABLED '; calc_bestfit_grad_para=.true. ; calc_voronoi_para=.true.
            if (qprint) then
             write(info,7009) whoami, 'FORCES AND VORONOI DISTANCES', keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
            endif
           case('NO','OFF','FALSE','F','no','off','false','f')
            keyword='DISABLED' ; calc_bestfit_grad_para=.false. ; calc_voronoi_para=.false.
            if (qprint) then
             write(info,7009) whoami, 'FORCES AND VORONOI DISTANCES', keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
            endif
           case default
            call wrndie(0,whoami,trim('UNKNOWN OPTION SPECIFIED'))
          end select
       enddo ! comlen
 7009 format(/A, ' PARALLEL COMPUTATION OF ',A,' ',A)
!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! specify whether to use custom allgather hypercube
      elseif (( keyword(1:4).eq.'COMM'(1:4) )) then
       do while (comlen .gt. 1)
        keyword=nexta8(comlyn,comlen)
        select case(keyword)
           case('allg','ALLG','ALLGATHE','allgathe')
            keyword='ALLGATHER'; allgather_method=allgather_
            if (qprint) then
             write(info,7012) whoami, keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
            endif
           case('hycu','HYCU','HYPER','hyper','hcube','HCUBE')
            keyword='HYPERCUBE' ; allgather_method=hypercube_
            if (qprint) then
             write(info,7012) whoami, keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
            endif
           case('gatherb','GATHERB','gather','GATHER','bcast','BCAST')
            keyword='GATHER + BCAST' ; allgather_method=gather_bcast_
            if (qprint) then
             write(info,7012) whoami, keyword ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
            endif
           case default
            call wrndie(0,whoami,trim('UNKNOWN OPTION SPECIFIED'))
          end select
       enddo ! comlen
 7012 format(/A, ' WILL COMMUNICATE FTSM FORCES USING ',A)
!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif ( (( keyword(1:4).eq.'PRIN'(1:4) )) .or. (( keyword(1:4).eq.'WRIT'(1:4) )) ) then
       if (.not.((associated(r_o).and.associated(orientWeights).or..not.qorient).and.associated(r_f).and.associated(forcedWeights))) then
        call wrndie(0,whoami,trim('SOME FTSM COORDINATES NOT DEFINED. USE "FILL" or "READ". NOTHING DONE.')); return;
       endif
! can write both coordinate files and a global dcd file
! local is specified with 'COR'; global is the default

       qcor=indxa(comlyn, comlen, 'COR'); qcor = min(qcor,1)



       qdcd=indxa(comlyn, comlen, 'DCD'); qdcd = min(qdcd,1)
!
       if ((qcor+qdcd) .gt. 1) then
        call wrndie(0,whoami,trim(' MORE THAN ONE OUTPUT FORMAT REQUESTED. WILL USE DCD.'))
        qcor=0; qdcd=1;
       endif
!
! prepare file
!-----------------------------------------------------------------------------
       ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
       CALL GTRMWD(COMLYN,COMLEN,'NAME',4,FNAME,80,FLEN)
! note: FNAME will be UPPER CASE
!---------------------------------- OPEN FILE --------------------------------
       if (qroot) then

        oldiol=iolev

        if (qdcd.eq.0) then ! no dcd -- local write

         iolev=1 ! open file on all nodes

         if (flen.gt.0) call open_file(ifile, fname, 'FORMATTED', 'WRITE')
        else
         if (qprint) then ! write one dcd file (root does this)

          iolev=1 ! open file on root (which may not be zero if rex is used)

          if (flen.gt.0) call open_file(ifile, fname, 'UNFORMATTED', 'WRITE') ! open binary fle for DCD
         endif
! broadcast ifile so that all roots know whether handle is valid
         call mpi_bcast(ifile,1,mpiint,0,MPI_COMM_STRNG,ierror)
        endif

        if (ifile .eq. -1) then
         if (qdcd .eq. 0 ) ifile=outu ! write to output stream (rather dirty, but keep for now)
        endif

        if (ifile.ge.0) then
!---------------------------- assume file is open, write -------------------------
! check for column spec
         c1=sm_get_column(comlyn, comlen, .true., -1)
         if (c1.gt.0) then
          if (qprint) then ; write(info,6679) whoami, c1 ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6679 format(/A,' WRITING COORDINATES FROM COLUMN ',I3)
          if (qdcd.gt.0) then ; call ftsm_write_dcd(IFILE=ifile,COL=c1) ;
          else ; call ftsm_write_cor(ifile,c1) ; endif
         else
          if (qprint) then ; write(info,6689) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6689 format(/A,' WRITING COORDINATES FROM DEFAULT COLUMN.')
          if (qdcd.gt.0) then ; call ftsm_write_dcd(IFILE=ifile) ;
          else ; call ftsm_write_cor(ifile) ; endif
         endif ! c1
         if (qdcd.eq.0.or.qprint) then
          if (flen.gt.0) call VCLOSE(ifile, 'KEEP', error)
         endif
        endif ! ifile

        iolev=oldiol

       endif ! qroot
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'READ'(1:4) )) then
       if (ftsm_com_on .and. (.not.((associated(r_o).and.associated(orientWeights).or..not.qorient).and.associated(r_f).and.associated(forcedWeights)))) call ftsm_coor_wgt_alloc(amass,natom)
       if (.not.((associated(r_o).and.associated(orientWeights).or..not.qorient).and.associated(r_f).and.associated(forcedWeights))) then
        call wrndie(0,whoami,trim('SOME FTSM COORDINATES NOT DEFINED. USE "FILL" or "READ". NOTHING DONE.')); return;
       endif
! can read from both coordinate files and a global dcd file (see above)
! can also read a frame in the DCD: specify FRAM for frame;
       qcor=indxa(comlyn, comlen, 'COR'); qcor = min(qcor,1)
       qdcd=indxa(comlyn, comlen, 'DCD'); qdcd = min(qdcd,1)
!
       if ((qcor+qdcd) .gt. 1) then
        call wrndie(0,whoami,trim('MORE THAN ONE INPUT FORMAT REQUESTED. WILL USE DCD.'))
        qcor=0; qdcd=1;
       endif
!
! prepare file
       ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
       call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: flen will be UPPER CASE
! check for column spec (which coordinate set to read into)
       c1=sm_get_column(comlyn, comlen, .true., 0)
!cccccccccccccccccccccccccccc OPEN FILE ccccccccccccccccccccccc
       if (qroot) then

        oldiol=iolev

        if (qdcd.eq.0) then

         iolev=1 ! open file on all processors

         if (flen.gt.0) call open_file(ifile, fname, 'FORMATTED', 'READ')
        else
         if (qprint) then ! binary dcd file

          iolev=1

          if (flen.gt.0) call open_file(ifile, fname, 'UNFORMATTED', 'READ') ! open binary fle for DCD
         endif
! broadcast ifile so that all roots know whether handle is valid
         call mpi_bcast(ifile,1,mpiint,0,MPI_COMM_STRNG,ierror)
        endif

        if(ifile .le. -1 ) then
         if (qdcd.eq.0 ) then
          ifile=istrm ! read local files from input file
          call rdtitl(titleb,ntitlb,ifile,0) ! 0 = card format
         endif ! qdcd
        endif ! ifile
!
        iolev=oldiol

       endif ! qroot
!cccccccccccccccccc assume file is open, read ccccccccccccccccccc
! broadcast ifile so that all slaves also know whether handle is valid
! because they need to enter read_cor
       if (qslave) then ; call mpi_bcast(ifile,1,mpiint,0,MPI_COMM_LOCAL,ierror) ; endif
       if (ifile.ge.0) then
        if (c1.gt.0) then ! column spec
         if (qprint) then ; write(info,6699) whoami, c1 ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6699 format(A,' READING COORDINATES INTO COLUMN ',I3)
         if (qdcd.gt.0) then ; if (qroot) call ftsm_read_dcd(ifile, c1);
         else; call ftsm_read_cor(ifile,c1) ; endif
       else
         if (qprint) then ; write(info,6709) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6709 format(A,' READING COORDINATES INTO DEFAULT COLUMN.')
         if (qdcd.gt.0) then ; if (qroot) call ftsm_read_dcd(ifile);
         else ; call ftsm_read_cor(ifile) ;
         endif
        endif ! c1
       endif
!cccccccccccccccc close file ccccccccccccccccccccccccccccccccccccc
       if (qroot) then
        if (qdcd.eq.0.or.qprint) then
         if (flen.gt.0) call VCLOSE(ifile, 'KEEP', error)
        endif ! qdcd

        iolev=oldiol

       endif ! qroot
!
! broadcast to slaves (although cread routine will send coords to slaves, too)
       if (c1.le.0) c1=center ! assume "default column"
       if (c1.eq.center) then
        call ftsm_swap_bc(.true.)
        r_f(:,:,left_old:right_old)=r_f(:,:,left:right)
        r_f(:,:,center_new)=r_f(:,:,center)
        if (qorient.and.qdiffrot) then
         r_o(:,:,left_old:right_old)=r_o(:,:,left:right)
         r_o(:,:,center_new)=r_o(:,:,center)
        endif
       else ! swap_bc above will send to slaves
        if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1) then

#if (KEY_SINGLE==1) /*  forcing coordinates */
         call PSND4r(r_f(:,:,c1),3*nforced) 
#endif
#if (KEY_SINGLE==0)
         call PSND8m(r_f(:,:,c1),3*nforced) 
#endif



         if (qorient.and.qdiffrot) then

#if (KEY_SINGLE==1) /*  send orientation coordinates (only if distinct from forcing) */
          call PSND4r(r_o(:,:,c1),3*norient) 
#endif
#if (KEY_SINGLE==0)
          call PSND8m(r_o(:,:,c1),3*norient) 
#endif



         endif
        endif
       endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'SWAP'(1:4) )) then ! swap two columns
        if (.not.((associated(r_o).and.associated(orientWeights).or..not.qorient).and.associated(r_f).and.associated(forcedWeights))) then
         call wrndie(0,whoami,trim('SOME FTSM COORDINATES NOT DEFINED. USE "FILL" or "READ". NOTHING DONE.')); return;
        endif
! read column spec
        c1=sm_get_column(comlyn, comlen, qcoltag=.false., missing=-1)
        c2=sm_get_column(comlyn, comlen, qcoltag=.false., missing=-1)
        if (qprint) then ; write(info,6729) whoami, c1, c2 ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6729 format(/A,' WILL SWAP COLUMNS ',I3,' AND ',I3,' ')
        call ftsm_swap(c1, c2)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'COPY'(1:4) )) then ! copy form c1 to c2
        if (.not.((associated(r_o).and.associated(orientWeights).or..not.qorient).and.associated(r_f).and.associated(forcedWeights))) then
         call wrndie(0,whoami,trim('SOME FTSM COORDINATES NOT DEFINED. USE "FILL" or "READ". NOTHING DONE.')); return;
        endif
! read column spec
        c1=sm_get_column(comlyn, comlen, qcoltag=.false., missing=-1)
        c2=sm_get_column(comlyn, comlen, qcoltag=.false., missing=-1)
        if (qprint) then ; write(info,6739) whoami, c1, c2 ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6739 format(/A,' WILL COPY COLUMN ',I3,' TO ',I3,' ')
        call ftsm_copy(c1,c2)
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'SET '(1:4) )) then ! modify k, etc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if ( ( indx(comlyn, comlen, 'ORIE', 4).gt.0 ) .or. ( indx(comlyn, comlen, 'RMSD', 4) ).gt.0 ) then
! inlined selection functionality moved to subroutine
         call ftsm_add_atomic_set(comlyn, comlen)
!!==============================================================
! scaling output macro

 6758 format(A,'( LENGTH SCALE ASSUMED TO BE IN ANGSTROMS AND WILL BE SCALED INTERNALLY.)')
 6759 format(A,'( LENGTH SCALE ASSUMED TO BE SCALED BY TWICE THE INTER-REPLICA DISTANCE.)')
!
!==============================================================
! set k parallel to path (for off-path dynamics)
        elseif (indx(comlyn, comlen, 'KPAR', 4).gt.0) then
          k=gtrmf(comlyn, comlen, 'KPAR', -1d0)
! check for distance scaling option: if set the force constant is specified as u[k] = kcal/ mol /A^2
! otherwise, distance is dimensionless (scaled by twice the interimage distance in the default method), so u[k] = kcal/mol
! note that distance scaling only makes sense with ftsm_scaledist=.false., i.e. scaled version of the code
          if (ftsm_scaledist) qkpara_angstrom = indxa(comlyn, comlen, 'ANG').gt.0
!
          if (k.ge.0d0) then
           kpara=k
           if (qprint) then
            write(info,6756) whoami, k ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
            if (ftsm_scaledist) then
             if (qkpara_angstrom) then;write(info,6758)whoami;else;write(info,6759)whoami;endif;write(OUTU,'(A)') pack(info,info.ne.'');info='';;
            endif
           endif
 6756 format(A,' SETTING PARALLEL RESTRAINT FORCE CONSTANT TO ',F11.5)
          else
           if (qprint) then ; write(info,6757) whoami, k ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6757 format(A,' NEGATIVE FORCE CONSTANT SPECIFIED: ',F11.5, '. NOT SET.')
          endif
!==============================================================
! set k perpendicular to path (for off-path dynamics)
        elseif (indx(comlyn, comlen, 'KPRP', 4).gt.0) then
          k=gtrmf(comlyn, comlen, 'KPRP', -one)
          if (ftsm_scaledist) qkperp_angstrom = indxa(comlyn, comlen, 'ANG').gt.0
!
          if (k.ge.0d0) then
           kperp=k
           if (qprint) then
            write(info,6746) whoami, k ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
            if (ftsm_scaledist) then
             if (qkperp_angstrom) then;write(info,6758)whoami;else;write(info,6759)whoami;endif;write(OUTU,'(A)') pack(info,info.ne.'');info='';;
            endif
           endif
 6746 format(A,' SETTING PERPENDICULAR RESTRAINT FORCE CONSTANT TO ' &
     & ,F11.5)
          else
           if (qprint) then ; write(info,6757) whoami, k ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
          endif
!==============================================================
        elseif (indx(comlyn, comlen, 'KRMS', 4).gt.0) then
          k=gtrmf(comlyn, comlen, 'KRMS', -one)
          if (k.ge.0d0) then
           krms=k
           if (qprint) then ; write(info,6752) whoami, k ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6752 format(A,' SETTING RMSD RESTRAINT FORCE CONSTANT TO ' &
     & ,F11.5)
          else
           if (qprint) then ; write(info,6757) whoami, k ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
          endif
!==============================================================
        elseif (indxa(comlyn, comlen, 'MASS').gt.0) then ! mass-weighting
          if (ftsm_com_on .and. (.not.((associated(r_o).and.associated(orientWeights).or..not.qorient).and.associated(r_f).and.associated(forcedWeights)))) call ftsm_coor_wgt_alloc(amass,natom)
!
          keyword=nexta8(comlyn,comlen)
          klen=len(keyword)
          call trima(keyword, klen)



          select case(keyword(1:klen))
           case('YES','ON','TRUE','T','yes','on','true','t')
            if (qprint) then ; write(info,8001) whoami, 'SET FROM ATOM MASSES' ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
            call ftsm_set_weights(amass, natom) ! send masses
           case('NO','OFF','FALSE','F','no','off','false','f')
            if (qprint) then ; write(info,8001) whoami, 'WILL BE UNIFORM' ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
            call ftsm_set_weights( (/ (1d0, i=1,natom) /), natom) ! uniform

           case('WMAIN', 'wmain')
            if (qprint) then ; write(info,8001) whoami, 'SET FROM WMAIN ARRAY' ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
            call ftsm_set_weights(wmain, natom) ! send masses
           case('WCOMP', 'wcomp')
            if (qprint) then ; write(info,8001) whoami, 'SET FROM WCOMP ARRAY' ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
            call ftsm_set_weights(wcomp, natom) ! send masses
           case default
            call wrndie(0,whoami,trim('UNKNOWN OPTION SPECIFIED'))
          end select
 8001 format(A,' WEIGHTS ',A,'.')
!==============================================================
        elseif (indxa(comlyn, comlen, 'PROJ').gt.0) then
          keyword=nexta8(comlyn,comlen)
          klen=len(keyword)
          call trima(keyword, klen)
          select case(keyword(1:klen))
           case('YES','ON','TRUE','T','yes','on','true','t')
            proj_on=.true.
            if (qprint) then ; write (info,'(2A)') whoami, &
     & ' WILL RESTRAIN SYSTEM TO PLANE PERPENDICULAR TO PATH.' ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
            if (nstring.eq.1) then
             call wrndie(0,whoami,trim('PROJECTION IS ILL-DEFINED BECAUSE ONLY ONE STRING IMAGE IS PRESENT.'))
            endif
           case('NO','OFF','FALSE','F','no','off','false','f')
            proj_on=.false.
            if (qprint) then ; write (info,'(2A)') whoami, &
     & ' WILL RESTRAIN SYSTEM TO PATH IMAGE.'//                         &
     & ' (FE/MFPT CANNOT BE COMPUTED).' ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           case default
            call wrndie(0,whoami,trim('UNKNOWN OPTION SPECIFIED'))
          end select
!==============================================================
        elseif (indx(comlyn, comlen, 'DPAR', 4).gt.0) then ! normalized distance parallel to vector between neighboring replicas at which this system is restrained
          zval=gtrmf(comlyn, comlen, 'DPAR', -1.0d0)
! check replica spec
          irep=gtrmi(comlyn, comlen, 'REP', -1)
          if (irep.lt.0.or.irep.ge.nstring) then
           if (qprint) then ; write(info, 6773) whoami, whoami, zval ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           dpar0=zval
          else
           if (qprint) then ; write(info,6774) whoami, irep, zval ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           if (mestring.eq.irep) dpar0=zval ! note: permitting any value
          endif ! irep
 6773 format(A,' REPLICA NUMBER INVALID OR UNSPECIFIED.', &
     & /A,' WILL SET RERERENCE VALUE FOR PARALLEL RESTRAINT ', &
     & '  TO ',F7.3,' ON ALL REPLICAS.')
 6774 format(A,' WILL SET RERERENCE VALUE FOR PARALLEL RESTRAINT ', &
     & 'ON REPLICA ',I5,' TO ',F7.3,'.')
        ! DPAR
!==============================================================
        elseif (indx(comlyn, comlen, 'DPRPF', 5).gt.0) then ! final perpendicular distance at which the replicas are restrained
          qangstrom=.false.
          if (ftsm_scaledist) qangstrom = indxa(comlyn, comlen, 'ANG').gt.0 ! scaled/unscaled units
!
          zval=gtrmf(comlyn, comlen, 'DPRPF', anum)
          if (zval.eq.anum) then
           call wrndie(0,whoami,trim(' FINAL OFFSET DISTANCE FOR PERPENDICULAR RESTRAINT NOT SPECIFIED.'))
          else
! check replica spec
           irep=gtrmi(comlyn, comlen, 'REP', -1)
           if (irep.lt.0.or.irep.ge.nstring) then
            qdperpf_angstrom=qangstrom
            if (qprint) then
             write(info, 67761) whoami, whoami, zval ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
             if (ftsm_scaledist) then
              if (qangstrom) then;write(info,6758)whoami;else;write(info,6759)whoami;endif;write(OUTU,'(A)') pack(info,info.ne.'');info='';;
             endif
            endif
            dperp0f=zval
           else
            if (qprint) then ; write(info,67762) whoami, irep, zval ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ;
             if (ftsm_scaledist) then
              if (qangstrom) then;write(info,6758)whoami;else;write(info,6759)whoami;endif;write(OUTU,'(A)') pack(info,info.ne.'');info='';;
             endif
            endif
            if (mestring.eq.irep) then ; dperp0f=zval ; qdperpf_angstrom=qangstrom ; endif ! note: permitting any value
           endif ! irep
67761 format(A,' REPLICA NUMBER INVALID OR UNSPECIFIED.', &
     & /A,' WILL SET FINAL OFFSET DISTANCE FOR PERPENDICULAR RESTRAINT', &
     & ' TO ',E10.3,' ON ALL REPLICAS.')
67762 format(A,' WILL SET FINAL OFFSET DISTANCE FOR PERPENDICULAR RESTRAINT ',&
     & 'ON REPLICA ',I5,' TO ',E10.3,'.')
          endif ! zval valid
!==============================================================
        elseif (indx(comlyn, comlen, 'DPRPITER', 8).gt.0) then ! number of iterations to adjust to final perpendicular restraint value
          i=gtrmi(comlyn, comlen, 'DPRPITER', nint(anum))
!
          if (i.eq.nint(anum)) then
           call wrndie(0,whoami,trim(' NUMBER OF ADJUSTMENT ITERATIONS FOR PERPENDICULAR RESTRAINT NOT SPECIFIED.'))
           dperp_adjust_iter=0
          else
! check replica spec
           irep=gtrmi(comlyn, comlen, 'REP', -1)
           if (irep.lt.0.or.irep.ge.nstring) then
            if (qprint) then ; write(info, 67763) whoami, whoami, i ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
            dperp_adjust_iter=i
           else
            if (qprint) then ; write(info,67764) whoami, irep, i ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
            if (mestring.eq.irep) dperp_adjust_iter=i
           endif ! irep
67763 format(A,' REPLICA NUMBER INVALID OR UNSPECIFIED.', &
     & /A,' WILL SET NUMBER OF ADJUSTMENT ITERATIONS FOR PERPENDICULAR RESTRAINT', &
     & ' TO ',I9,' ON ALL REPLICAS.')
67764 format(A,' WILL SET NUMBER OF ADJUSTMENT ITERATIONS FOR PERPENDICULAR RESTRAINT ',&
     & 'ON REPLICA ',I5,' TO ',I9,'.')
          endif ! i valid
!==============================================================
        elseif (indx(comlyn, comlen, 'DPRP', 4).gt.0) then ! (initial) distance perpendicular to vector between neighboring replicas
          qangstrom=.false.
          if (ftsm_scaledist) qangstrom = indxa(comlyn, comlen, 'ANG').gt.0 ! scaled/unscaled units
          zval=gtrmf(comlyn, comlen, 'DPRP', anum)
!
          if (zval.eq.anum) then
           call wrndie(0,whoami,trim(' OFFSET DISTANCE FOR PERPENDICULAR RESTRAINT NOT SPECIFIED.'))
          else
! check replica spec
           irep=gtrmi(comlyn, comlen, 'REP', -1)
           if (irep.lt.0.or.irep.ge.nstring) then
            qdperp_angstrom=qangstrom
            if (qprint) then ; write(info, 67765) whoami, whoami, zval ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
             if (ftsm_scaledist) then
              if (qdperp_angstrom) then;write(info,6758)whoami;else;write(info,6759)whoami;endif;write(OUTU,'(A)') pack(info,info.ne.'');info='';;
             endif
            endif
            dperp0=zval ; dperp0i=dperp0
           else
            if (qprint) then ; write(info,67766) whoami, irep, zval ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
             if (ftsm_scaledist) then
              if(qangstrom) then;write(info,6758)whoami;else;write(info,6759)whoami;endif;write(OUTU,'(A)') pack(info,info.ne.'');info='';;
             endif
            endif
            if (mestring.eq.irep) then ; dperp0=zval ; dperp0i=dperp0 ; qdperp_angstrom=qangstrom ; endif ! note: permitting any value
           endif ! irep
67765 format(A,' REPLICA NUMBER INVALID OR UNSPECIFIED.', &
     & /A,' WILL SET OFFSET DISTANCE FOR PERPENDICULAR RESTRAINT', &
     & ' TO ',E10.3,' ON ALL REPLICAS.')
67766 format(A,' WILL SET OFFSET DISTANCE FOR PERPENDICULAR RESTRAINT ',&
     & 'ON REPLICA ',I5,' TO ',E10.3,'.')
          endif ! zval valid
!==============================================================
        elseif (indx(comlyn, comlen, 'DRMS', 4).gt.0) then ! RMS distance between simulation and reference structure
          zval=gtrmf(comlyn, comlen, 'DRMS', -one)
! check replica spec
          irep=gtrmi(comlyn, comlen, 'REP', -1)
          if (irep.lt.0.or.irep.ge.nstring) then
           if (qprint) then ; write(info, 6777) whoami, whoami, zval ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           drms0=zval
          else
           if (qprint) then ; write(info,6778) whoami, irep, zval ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           if (mestring.eq.irep) drms0=zval ! note: permitting any value
          endif ! irep
 6777 format(A,' REPLICA NUMBER INVALID OR UNSPECIFIED.', &
     & /A,' WILL SET REFERENCE VALUE FOR RMSD RESTRAINT ', &
     & '  TO ',F7.3,' ON ALL REPLICAS.')
 6778 format(A,' WILL SET REFERENCE VALUE FOR RMSD RESTRAINT ', &
     & 'ON REPLICA ',I5,' TO ',F7.3,'.')
!
! check upper boundary spec
          qrms_upper_bound = ( indxa(comlyn, comlen, 'UPPE').gt.0 )
          if (qrms_upper_bound.and.qprint) then
           write(info,6781) whoami
           write(OUTU,'(A)') pack(info,info.ne.'');info='';
          endif
 6781 format(A,' RESTRAINT WILL BE APPLIED ONLY IF RMSD EXCEEDS REFERENCE VALUE.')
!
!==============================================================
        elseif (indx(comlyn, comlen, 'VCUT', 4).gt.0) then
         voro_cut=gtrmf(comlyn, comlen, 'VCUT', zero)
! replica spec
         irep=gtrmi(comlyn, comlen, 'REP', -1)
         if (voro_cut.le.zero) then
          call wrndie(0,whoami,trim('VCUT MUST BE POSITIVE. NOT SET.'))
         else
          if (irep.lt.0.or.irep.ge.nstring) then
           if (qprint) then ; write(info, 6779) whoami, whoami, voro_cut ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           call ftsm_voronoi_set_cutoff(voro_cut)
          else
           if (qprint) then ; write(info, 6780) whoami, irep, voro_cut ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
           if (mestring.eq.irep) call ftsm_voronoi_set_cutoff(voro_cut)
          endif ! irep
 6779 format(A,' REPLICA NUMBER INVALID OR UNSPECIFIED.',/A,' WILL SET VORONOI TUBE CUTOFF ', &
     & '  TO ',F7.3,' ON ALL REPLICAS.')
 6780 format(A,' WILL SET VORONOI TUBE CUTOFF ON REPLICA ',I5,' TO ',F7.3,'.')
         endif ! voro_cut > 0
!==============================================================
        elseif (indx(comlyn, comlen, 'FLIM', 4).gt.0) then
         ftsm_flim_coeff=gtrmf(comlyn, comlen, 'FLIM', ftsm_flim_coeff)
         if (ftsm_flim_coeff.gt.0) then
          if (qprint) then ; write(info, 6782) whoami, ftsm_flim_coeff ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
         endif
 6782 format(A,' SET FORCE LIMIT COEFFICIENT TO ',F7.3,'(HOPEFULLY, YOU KNOW WHAT YOU ARE DOING).')
        endif
! done with 'SET' parsing
!==================================== M tensor ======================================
      elseif (( keyword(1:4).eq.'MMAT'(1:4) )) then
!
       if (ftsm_com_on) then
        call wrndie(0,whoami,trim('FTSM IN COM COORDINATES DOES NOT CURRENTLY SUPPORT M TENSOR CALCULATION.'))
        return
       endif
!
       if (ftsm_check(qorient)) then
        if (indxa(comlyn, comlen, 'CALC').gt.0) then
!
         ok=.true.
         if (qorient) then
          if(any(r_o(:,:,center).eq.anum)) then
           call wrndie(0,whoami,trim('ORIENTATION COORDINATES HAVE UNDEFINED VALUES. NOTHING DONE.'))
           ok=.false.
          endif
         endif
!
         if (ok) then
          if (qprint) then ; write(info,6854) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6854 format(A,' COMPUTING M TENSOR FROM ATOMIC COORDINATES.')
          call ftsm_calc_M(x,y,z,amass,one)
!
          num_M_samples=num_M_samples+1
          omt=one/num_M_samples
          t=one-omt
          Mtensor(:,:,:,:,2) = t * Mtensor(:,:,:,:,2) + omt * Mtensor(:,:,:,:,1) ! add to running average
!
         endif
!
        elseif (indxa(comlyn, comlen, 'CLEA').gt.0) then
          if (qprint) then ; write(info,6855) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6855 format(A,' CLEARING M TENSOR DATA.')
          num_M_samples=0;
          Mtensor=zero; do a=1,3; do i=1, nforced ; Mtensor(a,a,i,i,:)=one ; enddo ; enddo! reset to I
        endif
       endif
!
!==================================== Jacobian ======================================
      elseif (( keyword(1:4).eq.'JACO'(1:4) )) then
!
       if (ftsm_com_on) then
        call wrndie(0,whoami,trim('FTSM IN COM COORDINATES DOES NOT CURRENTLY SUPPORT JACOBIAN CALCULATION.'))
        return
       endif
!
       if (ftsm_check(qorient)) then
        if (indxa(comlyn, comlen, 'CALC').gt.0) then
!
         ok=.true.
         if (qorient) then
          if(any(r_o(:,:,center).eq.anum)) then
           call wrndie(0,whoami,trim('ORIENTATION COORDINATES HAVE UNDEFINED VALUES. NOTHING DONE.'))
           ok=.false.
          endif
         endif
!
         if (ok) then
          if (qprint) then ; write(info,6856) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6856 format(A,' COMPUTING APPROXIMATE JACOBIAN FROM ATOMIC COORDINATES.')
          call ftsm_calc_J(x,y,z,one)
!
          num_J_samples=num_J_samples+1
          omt=one/num_J_samples
          t=one-omt
          Jacobian(2) = t * Jacobian(2) + omt * Jacobian(1) ! add to running average
         endif ! not ok
!
        elseif (indxa(comlyn, comlen, 'CLEA').gt.0) then
          if (qprint) then ; write(info,6857) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6857 format(A,' CLEARING JACOBIAN.')
          num_J_samples=0; Jacobian=one
        endif
       endif
!
!cccccccccccccccccccccccccccccccccccc CV WEIGHTS cccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'LIST'(1:4) )) then ! list forcing and orientation atom groups
       if (qprint) then
        if (ftsm_com_on) then
         write(info,6763) whoami
        else
         write(info,6762) whoami
        endif
        write(OUTU,'(A)') pack(info,info.ne.'');info='';
       endif
 6762 format(/A,' WILL LIST REPLICA ATOMS.')
 6763 format(/A,' WILL LIST REPLICA ATOM GROUPS.')
       if (ftsm_com_on .and. (.not.((associated(r_o).and.associated(orientWeights).or..not.qorient).and.associated(r_f).and.associated(forcedWeights)))) then
          call ftsm_coor_wgt_alloc(amass,natom)
          ! qorient is set in this routine; we need it to know whether groups are identical or different
       end if
       call ftsm_list_atoms()
!=====================================================================================
      else
            write(info(1),*)'UNRECOGNIZED SUBCOMMAND: ',keyword;call wrndie(0,whoami,trim(info(1)))
      endif
!
      end subroutine ftsm_parse
!===================================================================
      subroutine ftsm_init()
      use stream
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use number
      use ivector; use ivector_list; use rvector; use rvector_list
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

      integer :: ierror
      logical :: qroot, qslave, qprint
!
      character(len=len("FTSM_INIT>") ),parameter::whoami="FTSM_INIT>";!macro
!
! do a basic communicator check:
      if (ME_LOCAL.eq.0.and.ME_STRNG.eq.MPI_UNDEFINED) then
        write(info, 111) whoami, ME_GLOBAL, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 111 FORMAT(A, ' WORLD REPLICA ',I5, ' HAS ZERO GROUP ID', &
     & /,A,' BUT INVALID STRING ID (MAY BE OK).')
      elseif (ME_STRNG.ne.MPI_UNDEFINED.and. &
     & (ME_LOCAL.ne.0.or.MPI_COMM_LOCAL.eq.MPI_COMM_NULL)) then
        write(info, 112) whoami, ME_GLOBAL, whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 112 FORMAT(A, ' WORLD REPLICA ',I5, ' HAS A VALID STRING ID', &
     & /,A,' BUT A NONZERO GROUP ID. ABORTING.')
       return
      endif
!
      qroot=ME_STRNG.ne.MPI_UNDEFINED
      qslave=ME_LOCAL.ne.MPI_UNDEFINED ! (also includes roots)
      qprint=qroot.and.ME_STRNG.eq.0
!
      if (ftsm_initialized) then
       if (qprint) then
          write(info,'(2A)') &
     & whoami, ' FTSM ALREADY INITIALIZED. CALL "DONE" TO CLEAN UP.' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';;
       endif ! qroot
       return
      endif
      nstring=1
      mestring=-1
!
      if (qroot) then
        nstring=SIZE_STRNG
        mestring=ME_STRNG
      endif

! broadcast string size to all slave nodes
#if (KEY_INTEGER8==1)
      call PSND8(nstring,1) 
#endif
#if (KEY_INTEGER8==1)
      call PSND8(mestring,1) 
#endif
!
#if (KEY_INTEGER8==0)
      call PSND4(nstring,1) 
#endif
#if (KEY_INTEGER8==0)
      call PSND4(mestring,1) 
#endif
! set envorinment variable
      call set_param('NSTRING',nstring)
      call set_param('MESTRING',mestring)




!
! disabling procedure pointer due to limited compiler compatibility
! if (qver2) then ; ftsm_calc=>ftsmv2_calc ; else ; ftsm_calc=>ftsmv1_calc ; endif
!
      if (qprint) then
          write(info,'(2A,I5, A)') &
     & whoami, ' FOUND ',nstring,' REPLICAS.'
          if (qver2) then
           write(info(2),'(2A)') whoami, ' FTSM VERSION 2 WILL BE USED TO SAMPLE HYPERPLANES.' ;
          endif
          write(OUTU,'(A)') pack(info,info.ne.'');info='';
      endif
!
      MPI_RTMD_TYPE =MPI_DATATYPE_NULL
      MPI_RTMD_TYPE_=MPI_DATATYPE_NULL
! initialize free energy arrays
      if(associated(fe))deallocate(fe)
      if(associated(feav))deallocate(feav)
      if(associated(ds))deallocate(ds)
      if(associated(curv))deallocate(curv)
      allocate(fe(nstring), feav(nstring), &
     & ds(nstring-1), curv(nstring-2))
!
      allocate(iatoms_a)
      call int_vector_init(iatoms_a)
      if (ftsm_com_on) then
       allocate(iatoms_o, iatoms_f, wgts_o, wgts_f)
       call int_vlist_init(iatoms_o)
       call int_vlist_init(iatoms_f)
! call real_vlist_init(wgts_o)
! call real_vlist_init(wgts_f)
      endif
!
      fe=0d0; feav=0d0; avforce=0d0; ds=0d0; curv=0d0
!
      num_evolve_samples=0
      num_force_samples=0
      num_M_samples=0
! set default restraint positions
! these will be overwritten later if distances are unscaled (ftsm_scaledist=f)
!
      if (ftsm_scaledist) then
       if (mestring.eq.0) then
        dpar0=zero
       elseif (mestring.eq.nstring-1) then
        dpar0=one
       else
        dpar0=half
       endif
      else
       dpar0=anum
       ftsm_reset_dpar0f=.false.
      endif
!
      dpar0i=dpar0 ; dpar0f=anum ;
!
      dperp0=one ; dperp0i=dperp0 ; dperp0f=anum ; dperp_adjust_iter=0
      drms0=zero
!
      qorient=.false.
      qdiffrot=.false.
!
      nullify(Mtensor)
!
      restrained_eq0=0 ; unrestrained_eq0=0
!
      ftsm_initialized=.true.
!
      end subroutine ftsm_init
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine ftsm_done()
      use ftsm_rex, only: ftsm_rex_done
      use ftsm_voronoi, only : ftsm_voronoi_done
!
      use stream
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use param_store, only: set_param
!
      character(len=len("FTSM_DONE>") ),parameter::whoami="FTSM_DONE>";!macro
!
      if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then
       write(info,'(2A,I5, A)') whoami, ' CLEANING UP.' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
      endif
!
      nstring=-1
      mestring=-1

! set envorinment variable
      call set_param('NSTRING',nstring)
      call set_param('MESTRING',mestring)

!
! deallocate arrays
!
      nforced=0
      if(associated(r_f))deallocate(r_f)
      if(associated(iatom_f))deallocate(iatom_f)
      if(associated(iatom_o))deallocate(iatom_o)
      if(associated(orientWeights))deallocate(orientWeights)
      if(associated(forcedWeights))deallocate(forcedWeights)
      if(associated(Mtensor))deallocate(Mtensor)
!
      if (qdiffrot) then
       if(associated(r_o))deallocate(r_o)
       if(associated(iatom_both))deallocate(iatom_both)
      endif
!
      if (ftsm_com_on) then
       if (associated(iatoms_f)) then
        if (iatoms_f%initialized) call int_vlist_done(iatoms_f)
        deallocate(iatoms_f)
       endif
!
       if (associated(iatoms_o)) then
        if (iatoms_o%initialized) call int_vlist_done(iatoms_o)
        deallocate(iatoms_o)
       endif
!
       if (associated(wgts_f)) then
        if (wgts_f%initialized) call real_vlist_done(wgts_f)
        deallocate(wgts_f)
       endif
!
       if (qdiffrot) then
        if (associated(wgts_o)) then
         if (wgts_o%initialized) call real_vlist_done(wgts_o)
         deallocate(wgts_o)
        endif
       else
        nullify(wgts_o)
       endif
!
      endif ! ftsm_com_on
!
      nullify(r_o)
      nullify(iatom_o) ! will not be associated with ftsm_com_on, but does not matter
      norient=0
      nboth=0
!
      nany=0
!
      if (associated(iatoms_a)) then
       if (iatoms_a%initialized) call int_vector_done(iatoms_a)
       deallocate(iatoms_a)
      endif
!
      if(associated(rcom))deallocate(rcom)
      if(associated(ds))deallocate(ds)
      if(associated(curv))deallocate(curv)
      if(associated(fe))deallocate(fe)
      if(associated(feav))deallocate(feav)
!
      num_evolve_samples=0
      num_force_samples=0
      num_M_samples=0
!
      qdiffrot=.false.
      qorient=.false.
      qver2=.false.
!
      call ftsm_define_rtmd_type() ! deallocate MPI type for transmitting forces (this is the effect when norient=0)
!
      call ftsm_rex_done()
!
      call ftsm_mini_done() ! deallocate minimization structures (if any)
!
      call ftsm_voronoi_done()
!
      ftsm_initialized=.false.
!
      end subroutine ftsm_done
!=========================================================================
      subroutine ftsm_list_atoms()
      use stream
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use chutil, only : atomid
!
      integer :: i, j
      character(len=8) :: sid, rid, ren, ac
      character(len=len("FTSM_LIST_ATOMS>") ),parameter::whoami="FTSM_LIST_ATOMS>";!macro
!
      if (ftsm_compute_qdiffrot) call ftsm_qdiffrot()
!
      if (ME_STRNG.eq.0) then
!
        if (norient.gt.0) then ; write(info,'(A)') tab//'ORIENTATION ATOMS:'
        write(OUTU,'(A)') pack(info,info.ne.'');info='';; endif
        do j=1, norient;
         if (ftsm_com_on) then
          write(info,666) whoami, j ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 666 format(A,' COM GROUP # ',I8,':')
          do i=1, iatoms_o%v(j)%last
           call atomid(iatoms_o%v(j)%i(i), sid, rid, ren, ac)
           write(info,667) tab,j, iatoms_o%v(j)%i(i), sid, rid, ren, ac; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          enddo ! i
         else ! ftsm_com_on
          call atomid(iatom_o(j), sid, rid, ren, ac)
           write(info,667) tab, j, iatom_o(j), sid, rid, ren, ac; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
        enddo ! j
!
        if (qdiffrot) then
         if (nforced.gt.0) then ; write(info,'(A)') tab//'FORCING ATOMS:'; 
         write(OUTU,'(A)') pack(info,info.ne.'');info='';; endif
         do j=1, nforced;
          if (ftsm_com_on) then
           write(info,666) whoami, j ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
           do i=1, iatoms_f%v(j)%last
            call atomid(iatoms_f%v(j)%i(i), sid, rid, ren, ac)
            write(info,667) tab, j, iatoms_f%v(j)%i(i), sid, rid, ren, ac; write(OUTU,'(A)') pack(info,info.ne.'');info='';
           enddo
          else ! ftsm_com_on
           call atomid(iatom_f(j), sid, rid, ren, ac)
           write(info,667) tab, j, iatom_f(j), sid, rid, ren, ac; write(OUTU,'(A)') pack(info,info.ne.'');info='';
          endif ! ftsm_com_on
         enddo ! j
        else
         write(info,'(A)') tab//'FORCING AND ORIENTATION ATOMS ARE THE SAME' 
         write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif ! qdiffrot
      endif ! ME_STRING
!
 667 format(A,2I8,' ',4A)
!
      end subroutine ftsm_list_atoms
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! VO 1/2013 modifying function call to include string minimization option
      subroutine ftsm_main(x,y,z,xcomp,ycomp,zcomp,fx,fy,fz,mass,iteration & ! include comparison set for voronoi calculations

     & , wmain, nbond_data, image_data & ! to be passed on to ftsm_mini

     & )
!
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use stream
      use number
      use chm_types, only : nonbondDataStructure, imageDataStructure
      use tsp ! traveling salesman module
!
      real(chm_real), dimension(:) :: x, y, z, xcomp, ycomp, zcomp, fx, fy, fz, mass
      integer :: iteration ! MD iteration
!

! CHARMM - dependent energy evaluation routines/vars
      real(chm_real) :: wmain(:)
      type(nonbondDataStructure) :: nbond_data
      type(imageDataStructure) :: image_data

! locals
      real(chm_real) :: s=zero, t, omt ! , zval
      logical :: qgrp
!
      character(len=len("FTSM_MAIN>") ),parameter::whoami="FTSM_MAIN>";!macro
!==========================================================================
      if (ftsm_ini_iteration.lt.0) ftsm_ini_iteration=iteration
!
      qgrp=(MPI_COMM_LOCAL.ne.MPI_COMM_NULL).and.(SIZE_LOCAL.gt.1)
!==========================================================================
! evolution of dperp, if requested
      if (dperp_adjust_iter.gt.0) then
       if (dperp0f.ne.anum) then
        s=one*(iteration-ftsm_ini_iteration)/dperp_adjust_iter ; s=min(max(s,zero),one); ! limit s to range [0,1]
        dperp0 = s * dperp0f + (one-s) * dperp0i
       endif
      endif
! aa
! write(me_global+600,*) dperp0, dperp0i, dperp0f, dperp_adjust_iter
!
      restraint_force_on=.false.
!
      if (restrained_on) then ! impose restraints
!
       if (.not.unrestrained_on .or. & ! if unrestrained dynamics is off
     & (unrestrained_on.and. & ! or: unrestrained dyn. is on, BUT we are equilibrating w/ res.
     & ((iteration-unrestrained_eq0).lt.unrestrained_eq_steps))) then ! first equilibrate with restraints on, then release restraints
!
        restraint_force_on=.true.
!
! limit s to range [0,1], even though calc works fine with s>1
        if (restrained_eq_steps.gt.0) then
         s=one*(iteration-restrained_eq0)/restrained_eq_steps ; s=min(max(s,zero),one)
        else
         s=one
        endif
!aa
! write(600+ME_GLOBAL,*) iteration, restrained_eq0, restrained_eq_steps, s ; ! close(600+ME_GLOBAL)
!
! evolution of dpar if necessary (used during string evolution in unscaled version of code)
        if (dpar0f.ne.anum) then
         dpar0 = s * dpar0f + (one-s) * dpar0i
        endif
!
        if (qver2.and.proj_on) then ; call ftsmv2_calc(x,y,z,.true.,s); else ; call ftsmv1_calc(x,y,z,.true.,s); endif ! compute gradients
        call ftsm_addforce(fx,fy,fz,s.ge.one) ! add restraint forces to global force arrays
!
       endif ! unrestrained_on
      endif ! restrained_on
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! the following only gets executed if iteration > olditeration;
! this is because CHARMM executes frequent 'restarts' at the least frequency
! that is common to all output counters; a restart will require two calls to this routine;
! to avoid duplicating statistics + evolution etc (since the step # is the same!) I keep track
! of the iteration counter, and proceed only if the iteration counter has increased.
      if (iteration.gt.olditeration) then
       if (evolve_ftsm_on.and.evolve_freq.gt.0) then
!
        if ( mod(iteration,evolve_freq).eq.0 .and. &
     & (iteration-restrained_eq0.gt.evolve_nskip)) then
! note that e_nskip is relevant even with restraints off, i.e. not protected by "restrained_on"; when restraints are off we are still using
! restrained_eq0 (which gets reset after each update)
         call ftsm_evolve(x,y,z) ;
!------------------------------------------------------------------------
! compute M tensor : NB, currently computed as a statistic only; no functional role (no compelling reason to do otherwise)
         if (output_M) then
          call ftsm_calc_M(x,y,z,mass,s) ! computes instantaneous M (Mtensor(:,:,:,:,1))
          num_M_samples=num_M_samples+1
          omt=one/num_M_samples
          t=one-omt
          Mtensor(:,:,:,:,2) = t * Mtensor(:,:,:,:,2) + omt * Mtensor(:,:,:,:,1) ! add to running average
!aa
! if (ME_STRNG.eq.0) then
! write(700+ME_STRNG,*) '% ',s, olditeration, iteration, evolve_freq
! write(700+ME_STRNG,'(30F12.6)') ( (Mtensor(:,a,:,k,1), a=1,3), k=1,nforced ) ; close(700+me_strng)
! write(800+ME_STRNG,'(30F12.6)') ( (Mtensor(:,a,:,k,2), a=1,3), k=1,nforced ) ; close(800+me_strng)
! endif
! call MPI_BARRIER(MPI_COMM_GLOBAL,i)
! stop
!aa
         endif ! output_M
!
! compute approximate Jacobian : NB, currently computed as a statistic only; no functional role
         if (output_J) then
          call ftsm_calc_J(x,y,z,s) ! computes instantaneous J
          num_J_samples=num_J_samples+1
          omt=one/num_J_samples
          t=one-omt
          Jacobian(2) = t * Jacobian(2) + omt * Jacobian(1) ! add to running average
         endif ! output_J
        endif ! evolve_freq
       endif ! evolve
!------------------------------------------------------------------------
       if (update_on.and.update_freq.gt.0) then
        if (mod(iteration,update_freq).eq.0) then
         if (.not.string_noprint) then
          write(info,'(2A,I10)') whoami,' UPDATING STRING AT STEP ',iteration ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
! VO 4.2014 : see if image reconnection requested
         if (ftsm_reconnect_on) then
          if (.not.string_noprint) then
             write(info,'(2A)') whoami,' RECONNECTING IMAGES TO MINIMIZE PATH LENGTH.' ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
          endif
          call ftsm_reconnect(iteration)
         endif
!
         if (proj_on) then
! save old reference coordinates
          r_f(:,:,left_old:right_old)=r_f(:,:,left:right)
          r_f(:,:,center)=r_f(:,:,center_new)
          if (qorient.and.qdiffrot) then
           r_o(:,:,left_old:right_old)=r_o(:,:,left:right)
           r_o(:,:,center)=r_o(:,:,center_new)
! make sure orientation coordinates are current
           call ftsm_update_overlap_coor(ione) ! r_f --> r_o
          endif
         else ! not proj_on (voronoi tessellation should also be covered in this case)
! save old reference coordinates and switch to new reference coordinates
          if (.not.string_noprint) then
           write(info,'(2A)') whoami,' UPDATING REFERENCE IMAGES.'
           if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
          endif
          r_f(:,:,center_old)=r_f(:,:,center)
          r_f(:,:,center)=r_f(:,:,center_new)
          if (qorient.and.qdiffrot) then
           r_o(:,:,center_old)=r_o(:,:,center)
           r_o(:,:,center)=r_o(:,:,center_new)
! make sure orientation coordinates are current
           call ftsm_update_overlap_coor(ione) ! r_f --> r_o
          endif
         endif ! proj_on
!
! VO 1.2013 : see if image minimization is requested
         if (ftsm_mini_on) then
            if (.not.string_noprint) then
             write(info,'(2A)') whoami,' MINIMIZING IMAGE ENERGY.' ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
            endif
            call ftsm_mini(x, y, z &

     & ,wmain, nbond_data, image_data &
     & ,fx, fy, fz & ! to make sure forces are not overwritten

     & )
         endif ! ftsm_mini_on
! see if reparametrization requested
         if (repa_on) then
          if (.not.string_noprint) then
             write(info,'(2A)') whoami,' REPARAMETRIZING IMAGES.'
             if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
          endif
          call ftsm_repa(.not.proj_on) ! reparametrize string, proj_on: do not broadcast to slaves b/c ftsm_swap_bc ; removes COM
         else
! recompute and remove centers of mass (which will change due to repa)
           call ftsm_save_com()
         endif ! repa_on
! update reference coordinates
         if (proj_on) then
          if (.not.string_noprint) then
           write(info,'(2A)') whoami,' UPDATING NEIGHBOR IMAGES.' ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
          endif
          call ftsm_swap_bc(.true.) ! VO changed from false to true 1/2013 (why was it false? problem only in parallel)
         endif ! proj_on
!------------------------------------------------------------------------
! Voronoi tessellation:
! smart-update voronoi data: if update too aggressive and some replicas end up in a wrong v. cell, string coords will be rolled back;
         if (voronoi_hist_on) call ftsm_voronoi_smart_update(x,y,z,xcomp,ycomp,zcomp)
!
! reset arrays for updating reference structure
         r_f(:,:,center_new)=r_f(:,:,center)
         if (qorient.and.qdiffrot) r_o(:,:,center_new)=r_o(:,:,center)
!
         if (unrestrained_on) unrestrained_eq0=iteration
         restrained_eq0=iteration
!
! if we are working in unscaled coordinates, string length changes due to evolution/repa/mini; therefore we need to update the reference distance
! however, this is not so simple because the string adjusts gradually to avoid instability; this means that we have to adjust dpara gradually also
! to do this, I will introduce dpar0f in analogy with dperp0f
! NOTE that I will not pass dpar0i/f and dperp0i/f with rex data; 1 : this would give too many features; 2 : dpar0i/f should typically be the same on all im
         if (.not.ftsm_scaledist) then
! zval= sum(ds) / ( max( nstring-1 , 1 ) ) * sqrt3 ; ! sqrt(3) makes the arclength correspond to the RMSD in 3D (what ftsm_compute uses)
! write(info,6713) whoami, ftoa(zval); if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info=''; ;
! 6713 format(A,' AVERAGE COMPUTED ARCLENGTH INCREMENT IS ',A,' ANG.')
! if (mestring.gt.0) dpar0f = zval ! set final target for dpar ; left endpoint stays at zero
! dpar0i=dpar0 ; ! current target becomes initial target
          ftsm_reset_dpar0f=.true. ! instead of the above, compute the reference location in ftsmv2_calc
         endif ! scaledist
!
        endif ! update_on
       endif ! update_on
!
       if (repl_x_on.and.repl_x_freq.gt.0) then
        if (mod(iteration, repl_x_freq).eq.0) then
         if (.not.string_noprint) then
          write(info,'(2A)') whoami,' ATTEMPTING EXCHANGE OF NEIGHBORING REPLICAS.'
          if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
         call ftsm_repl_exchange(x, y, z, iteration)
        endif
       endif
!
       if (stat_on.and.stat_freq.gt.0) then
         if (mod(iteration,stat_freq).eq.0) then
           write(info,'(2A)') whoami,' CALLING STRING STATISTICS.'
           if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
           call ftsm_stat() ! output statistics
         endif
       endif ! stat_on
      endif ! iteration > olditeration
! update internal iteration counter
      olditeration=iteration
!
      end subroutine ftsm_main
!===========================================================================
      subroutine ftsm_evolve(x,y,z)
      use number
      real(chm_real) :: x(:), y(:), z(:)
      integer :: ind, i, j, k
      real(chm_real) :: t, omt, d
      real(chm_real) :: u (3,3)= Id3
      real(chm_real), pointer :: r_com(:), ow(:), wgts(:)
      real(chm_real), pointer, dimension(:,:) :: roi, roc, rfi, rfc
      integer, pointer :: inds(:)
!
      roi=>r_o(:,:,instant)
      ow=>orientWeights
      rfi=>r_f(:,:,instant)
      r_com=>rcom(:,instant)
!
      roc=>r_o(:,:,center_new)
      rfc=>r_f(:,:,center_new)
!
      if (evolve_aver_on) then
       if (num_evolve_samples.lt.max_evolve_samples.or. &
     & max_evolve_samples.le.0) &
     & num_evolve_samples=num_evolve_samples+1
       omt=1d0/num_evolve_samples
       t=1d0-omt
      elseif (evolve_expo_on) then
       t=evolve_expo_mem
       omt=1d0-t
      endif
!
      if (.not. restraint_force_on) then ! load coordinates unless restraints on, in which case they are loaded
!
!
!
!
!
!
! | macro ftsm_load_fcor.def
! v
! ==========================
  if (ftsm_com_on) then
   do k=1, nforced
    rfi(k,:)=zero
    inds=>iatoms_f%v(k)%i
    wgts=>wgts_f%v(k)%r
    do i=1, iatoms_f%v(k)%last
     ind=inds(i)
     d=wgts(i)
     !
     rfi(k,1)=rfi(k,1)+(d*x(ind));
     rfi(k,2)=rfi(k,2)+(d*y(ind));
     rfi(k,3)=rfi(k,3)+(d*z(ind));
    enddo ! i
   enddo ! k
  else
   do k=1, nforced
    ind=iatom_f(k)
    !
    rfi(k,1)=x(ind)
    rfi(k,2)=y(ind)
    rfi(k,3)=z(ind)
   enddo
  endif
!===========================
! ^
! | macro ftsm_load_fcor.def
!
       if (qorient) then
        if (qdiffrot) then
!
!
!
!
!
!
! | macro ftsm_load_fcor.def
! v
! ==========================
  if (ftsm_com_on) then
   do k=1, norient
    roi(k,:)=zero
    inds=>iatoms_o%v(k)%i
    wgts=>wgts_o%v(k)%r
    do i=1, iatoms_o%v(k)%last
     ind=inds(i)
     d=wgts(i)
     !
     roi(k,1)=roi(k,1)+(d*x(ind));
     roi(k,2)=roi(k,2)+(d*y(ind));
     roi(k,3)=roi(k,3)+(d*z(ind));
    enddo ! i
   enddo ! k
  else
   do k=1, norient
    ind=iatom_o(k)
    !
    roi(k,1)=x(ind)
    roi(k,2)=y(ind)
    roi(k,3)=z(ind)
   enddo
  endif
!===========================
! ^
! | macro ftsm_load_fcor.def
        endif ! qdiffrot (otherwise rfi and roi point to the same thing)
!
! translate forced atoms to centroid
        r_com(:)=0d0;
        do j=1,3 ; do k=1, norient;
          r_com(j) = r_com(j)+ow(k)*roi(k,j)
        enddo ; enddo
!
        rfi(:,1)=rfi(:,1)-r_com(1)
        rfi(:,2)=rfi(:,2)-r_com(2)
        rfi(:,3)=rfi(:,3)-r_com(3)
!
        if (qdiffrot) then
         roi(:,1)=roi(:,1)-r_com(1)
         roi(:,2)=roi(:,2)-r_com(2)
         roi(:,3)=roi(:,3)-r_com(3)
        endif ! qdiffrot
!
       endif ! qorient
      endif ! .not. restrained on
!
      if (qorient) then ! orient w.r.t. center image
       call RMSBestFit(roc,roi,ow,u) ! superpose roc onto roi
       rfc = t * rfc + omt * matmul(rfi, u) ! apply transpose (=inverse) of u to rfi
      else
! evolve image using instantaneous structure
       rfc = t * rfc + omt *rfi
      endif
! NOTE that if the forcing set overlaps with orientation set, we also need to
! update some atom coords in the orientation set; this is done elsewhere to save CPU time
! NOTE also that rfi is not rotated
      end subroutine ftsm_evolve
!================================================================
      subroutine ftsm_orient(comlyn, comlen)
! based on sm0k_repa_init
      use stream
      use dimens_fcm
      use string
      use coord; use coordc
      use select, only : selcta, selrpn, nselct; use psf
      use number
      use multicom_aux;
      use sm0k, only:sm0k_fixed_atoms
      use parallel, only: psnd4, psnd8
!
      implicit none
!
      character(len=*) :: comlyn
      integer :: comlen
!
      integer, pointer :: iselct(:), ifixed(:)
      integer :: imode
      real(chm_real), pointer :: rcurrent_o(:,:), rref_o(:,:), ow(:) ! coordinates and orientation weights
      integer :: norient
      real(chm_real) :: orient_mass, wsum
      real(chm_real) :: u(3,3)=Id3
      real(chm_real) :: rcurrent_com(3)=(/zero,zero,zero/) ! COM vector
      integer :: i, j, me, ncpu, ierror, stat(MPI_STATUS_SIZE)
      logical :: qroot, qslave
!
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
      character(len=len("FTSM_ORIENT>") ),parameter::whoami="FTSM_ORIENT>";!macro
!
!===== begin
      qroot =MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.gt.1
      qslave=MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1
!
      allocate(iselct(natom)) ; iselct=1
       IMODE=0
      CALL SELRPN(COMLYN,COMLEN,iselct,NATOM,1,IMODE, &
     & .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
     & .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
      if (IMODE.NE.0) then
       call wrndie(0,whoami,trim('ATOM SELECTION ERROR.'))
       deallocate(iselct)
       return
      endif
! check for mass weighting
      if (indxa(comlyn, comlen, 'MASS').gt.0) then ; orient_mass=one; else ; orient_mass=zero; endif
!
      norient=count( iselct(1:natom).gt.0 ) ! total number of orientation atoms
!
      if (norient.eq.0) then
        call wrndie(0,whoami,trim('NO ATOMS SELECTED FOR ORIENTATION. NOTHING DONE.'))
        if(associated(iselct))deallocate(iselct)
       return
      elseif (norient.lt.3) then
        call wrndie(0,whoami,trim('FEWER THAN THREE ATOMS SELECTED FOR ORIENTATION. NOTHING DONE.'))
        if(associated(iselct))deallocate(iselct)
       return
      endif
!
        write(info,670) whoami, itoa(norient)
        if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info='';
 670 format(A,' INSTANTANEOUS (STRING) COORDINATES WILL BE ORIENTED BASED ON ',A,' ATOMS')
        if (orient_mass.eq.1) then ;
        write(info,671) whoami ; if(prnlev.ge. 3) write(OUTU,'(A)') pack(info,info.ne.'');info=''; ;
        endif
 671 format(A,' ORIENTATION WILL USE MASS-WEIGHTING')
!=== check for fixed atoms (allowed, but warrant a warning)
      ifixed=>sm0k_fixed_atoms()
      if (size(ifixed).gt.0) then
       call wrndie(0,whoami,trim('FIXED ATOMS FOUND. ALL COORDINATES WILL BE MODIFIED BY ORIENTATION.'))
      endif
      deallocate(ifixed)
!
      allocate(rcurrent_o(norient,3), rref_o(norient,3), ow(norient))
!
      rcurrent_com=zero
      wsum=zero
      j=1
      do i=1, natom
       if (iselct(i).gt.0) then
        rcurrent_o(j,1)=x(i); rcurrent_o(j,2)=y(i); rcurrent_o(j,3)=z(i);
        ow(j)= one-orient_mass + amass(i) * orient_mass
        rcurrent_com = rcurrent_com + ow(j)*rcurrent_o(j,:)
        wsum=wsum+ow(j)
!
        j=j+1;
       endif
      enddo
! aa
! write(700+ME_STRNG,*) ow, wsum, norient
! write(700+ME_STRNG,*) orient_mass, rcurrent_com
! write(700+ME_STRNG,*) rcurrent_o
! write(700+ME_STRNG,*) natom,size(x), x(1:natom)
! write(700+ME_STRNG,*) natom,size(y), y(1:natom)
! write(700+ME_STRNG,*) natom,size(z), y(1:natom)
! close(700+ME_STRNG)
      if (wsum.gt.RSMALL) then ; wsum=one/wsum ; else ; wsum=1; endif
      ow=ow*wsum
      rcurrent_com=rcurrent_com*wsum
! subtract COM : note that all nodes compute it, which is why we do not need to broadcast it below
       rcurrent_o(:,1)=rcurrent_o(:,1)-rcurrent_com(1)
       rcurrent_o(:,2)=rcurrent_o(:,2)-rcurrent_com(2)
       rcurrent_o(:,3)=rcurrent_o(:,3)-rcurrent_com(3)
!
! compute rotation matrices
      if (qroot) then
! send/receive orientation structure
        me=ME_STRNG
        ncpu=SIZE_STRNG
        if (me.gt.0) then
         call mpi_recv(rref_o,3*norient,mpifloat,me-1,1, &
     & MPI_COMM_STRNG, stat, ierror)
! orient rcurrent based on rref
! no checking for undefined coordinates here
         call RMSBestFit(rref_o,rcurrent_o,ow,u) ! superpose rref onto rcurrent
! transform current structure to overlap with reference
         rref_o=matmul(rcurrent_o, u) ! this should be safer than overwriting rcurrent_o
        else
         rref_o=rcurrent_o
        endif ! me
        if (me.lt.ncpu-1) then
         call mpi_send(rref_o,norient*3,mpifloat,me+1,1, &
     & MPI_COMM_STRNG, ierror)
        endif ! me
      endif ! root
!
! broadcast COM and rotation matrix to slaves
!
      if (qslave) then
#if (KEY_SINGLE==1)
       call PSND4r(u,9) 
#endif
#if (KEY_SINGLE==0)
       call PSND8m(u,9) 
#endif
      endif
! perform rotation of all coordinates
!
!aa write(600+me,*) u, rcurrent_com
! close(600+me)
      do i=1, natom
! reuse rref array
        rref_o(1,1)=x(i)-rcurrent_com(1);
        rref_o(1,2)=y(i)-rcurrent_com(2);
        rref_o(1,3)=z(i)-rcurrent_com(3);
        rref_o(2,:)=matmul(rref_o(1,:),u); ! rotate and place into rref(2) -- there must be at least 3 o. atoms
        x(i)=rref_o(2,1)
        y(i)=rref_o(2,2)
        z(i)=rref_o(2,3)
      enddo
!
! done !
      if(associated(iselct))deallocate(iselct)
      if(associated(rcurrent_o))deallocate(rcurrent_o)
      if(associated(rref_o))deallocate(rref_o)
      if(associated(ow))deallocate(ow)
!
      end subroutine ftsm_orient
!=================================================================
      subroutine ftsm_repl_exchange(x,y,z,itime)
! attempt to swap restraints that correspond to two adjacent replicas
      use multicom, only: multicom_permute_string_ranks
      use ftsm_rex, only: ftsm_rex_init, rex_initialized, rex_map, &
     & rex_log, rex_beta, rex_string_datatype, rex_string_data_mpi
      use ivector, only: int_vector_add
!
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use consta
      use number
      use clcg_mod, only: random; use reawri, only: iseed
      use reawri
      use string
      use mpi
      use param_store, only: set_param
      use parallel, only: psnd4, psnd8

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
      real(chm_real) :: x(:), y(:), z(:) ! mass(size(x,1))
      integer :: itime
!
      integer :: i, j, stat(MPI_STATUS_SIZE)
      integer*4 :: ierror
      integer(KIND=MPI_ADDRESS_KIND) :: lb, extent
      logical :: deriv, qendpoint, qgrp, qvalid
!
      integer :: which ! replica with which the exchange was attempted
      logical :: success ! whether the exchange attempt was successful
      integer :: nodelist(nstring) ! holds new string replica order after exchange attempt
! integer :: itype ! MPI_INTEGER type
      integer :: ndata, nfiles
!
! variables for exchanging string-dependent properties
      type(rex_string_datatype) :: rex_string_data, rex_string_data_new
      real(chm_real) :: dE_me, dE, s, dpar_ori, dperp_ori, drms_ori, fac
!
      character(len=200) :: fnames(5) ! for storing output file names
      character(len=200) :: new_fnames(5) ! for storing swapped file names
      logical :: openun(5)=.false., qform, qwrite
      integer :: oldiol
!
      real(chm_real), pointer, dimension(:,:,:) :: r_f2, r_o2, r_f3, r_o3
!
      character(len=len("FTSM_REPL_EXCHANGE>") ),parameter::whoami="FTSM_REPL_EXCHANGE>";!macro
!
      if (.not.rex_initialized) call ftsm_rex_init()
!
      if (.not.ftsm_check(qorient)) return
      if ( ( .not. restrained_on ) & ! restraints are off
& .or. ( restrained_on .and. unrestrained_on .and. ( (itime-unrestrained_eq0) .ge. unrestrained_eq_steps ) ) ) & ! unrestrained exploration
& return
!
      qgrp=(MPI_COMM_LOCAL.ne.MPI_COMM_NULL) &
     & .and.(SIZE_LOCAL.gt.1)
!
      deriv=.false. ! do not compute derivatives
      dE=zero
      success=.false.
!
! determine exchange partner
      if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
       if (ME_STRNG.eq.0) which=INT(random(iseed)*2d0) ! either 0 or 1
       call MPI_BCAST(which, 1, mpiint, 0, MPI_COMM_STRNG, ierror) ! string root broadcasts to all replicas
! determine whether swapping w. left (-1) or w. right (+1) neighbor & calculate rank of neighbor
       which=ME_STRNG + (mod(ME_STRNG + which, itwo)*itwo - ione)
! if which == 0, then: -1, 2, 1, 4, 3, ...
! if which == 1, then: 1, 0, 3, 2, ...
! communicate:
       qvalid=(which.ge.0).and.(which.lt.SIZE_STRNG)
       if (qvalid) then
! store reference values in dummy
! need to store all replica-dependent parameters
        rex_string_data%dpar0 = dpar0;
        rex_string_data%dperp0= dperp0;
        rex_string_data%drms0 = drms0;
        rex_string_data%qrms_upper_bound=qrms_upper_bound
        rex_string_data%kpara = kpara;
        rex_string_data%kperp = kperp;
        rex_string_data%krms = krms;
        rex_string_data%evolve_expo_mem = evolve_expo_mem;
        rex_string_data%num_evolve_samples = num_evolve_samples;
        rex_string_data%avforce = avforce
        rex_string_data%ftsm_mini_on = ftsm_mini_on
        rex_string_data%evolve_expo_on = evolve_expo_on
        rex_string_data%evolve_aver_on = evolve_aver_on
!
! send/receive
! allocate storage for new restraints
        allocate(r_f2(nforced,3,num_sets))
        ndata=3*(nforced*9) ! number of reals to send
        call MPI_SENDRECV(r_f, ndata, mpifloat, & ! send almost everything
     & which, which, r_f2, ndata, mpifloat, & ! put into the same array, starting at position 11
     & which, ME_STRNG, MPI_COMM_STRNG, stat, ierror)
        if (qorient) then
         if (qdiffrot) then ! orientation atoms
! allocate storage for new restraints
          allocate(r_o2(nforced,3,num_sets))
          ndata=27*norient
          call MPI_SENDRECV(r_o, ndata, mpifloat, &
     & which, which, r_o2, ndata, mpifloat, &
     & which, ME_STRNG, MPI_COMM_STRNG, stat, ierror)
         else
          r_o2=>r_f2
         endif ! qdiffrot
        endif ! qorient
! also exchange string image properties
        call MPI_SENDRECV(rex_string_data, 1, rex_string_data_mpi, which, which, &
     & rex_string_data_new, 1, rex_string_data_mpi, which, ME_STRNG,&
     & MPI_COMM_STRNG, stat, ierror)
! NOTE: in the above communication it is essential to have certain sets adjacent, as indicated by numbering in fts_var; do not break this
! calculate new string energies
        dpar_ori=dpar; dperp_ori=dperp; drms_ori=drms; ! first, save this replica`s projection values
! swap arrays:
        r_f3=>r_f; r_o3=>r_o ! save in case move is rejected
        r_f=>r_f2; r_o=>r_o2 ! point to new array
! call calculation
! consider the possibility that equilibration is underway:
        if (restrained_eq_steps.gt.0) then
         s=one*(itime-restrained_eq0)/restrained_eq_steps ; s=min(one,max(zero,s))
        else
         s=one
        endif ! restrained equilibration is on
!
        deriv=.false. ! skip derivative calculation
        if (qver2.and.proj_on) then ; call ftsmv2_calc(x,y,z,deriv,s); else ; call ftsmv1_calc(x,y,z,deriv,s); endif ! will compute new dpar, dperp, drms
!
! calculate energies (adapted from _addforce):
!
        if (proj_on) then
! restraint force parallel to string
         qendpoint=(which.eq.0.or.which.eq.nstring-1)
         if (qendpoint) then ; fac=half ; else ; fac=one ; endif
! compare energies (NOTE: in this version of REX, I am sending reference values, NOT coordinates)
! new energy
         dE_me = rex_string_data_new%kpara * fac * fac * ( dpar-rex_string_data_new%dpar0 )**2 & ! scale down the force constant of endpoints (one for d, one for gradients)
     & + rex_string_data_new%kperp * fac * max ( fac * dperp - rex_string_data_new%dperp0, zero )**2 ! ignore negative values; dperp0 criterion in inner-replica d-metric
! old energy
         qendpoint=(mestring.eq.0.or.mestring.eq.nstring-1)
         if (qendpoint) then ; fac=half ; else ; fac=one ; endif
!
         dE_me = dE_me &
     & - kpara * fac * fac * ( dpar_ori-dpar0 )**2 &
     & - kperp * fac * max ( fac * dperp_ori - dperp0, zero )**2
        else ! .not. proj_on
         dE_me = zero ;
         if (rex_string_data_new%qrms_upper_bound) then ; dE_me = max ( drms-rex_string_data_new%drms0, zero ) **2 ;
         else ; dE_me = ( drms-rex_string_data_new%drms0 ) **2 ;
         endif
!
         if (qrms_upper_bound) then ; dE_me = dE_me - max ( drms_ori-drms0, zero ) **2 ;
         else ; dE_me = dE_me - ( drms_ori-drms0 ) **2 ;
         endif
! do not forget to scale by force constant (half applied below)
         dE_me = rex_string_data_new%krms * dE_me
        endif
!
! combine energies from two replicas:
        call MPI_SENDRECV(dE_me, 1, mpifloat, &
     & which, which, dE, 1, mpifloat, &
     & which, ME_STRNG, MPI_COMM_STRNG, stat, ierror)
        dE = half * ( dE+dE_me )
! 5) apply Metropolis criterion
        if (dE.le.0d0) then
         success=.true.
        else
! the higher-rank replica draws random number
! this may not be correct because the random numbers will not come from the same sequence;
! may change this later
         if (which.lt.ME_STRNG) then
          success=(random(iseed).le.exp(-rex_beta*dE))
! send to othe replica
          call MPI_SEND(success, 1, mpibool, which, which, &
     & MPI_COMM_STRNG, ierror)
         else
          call MPI_RECV(success, 1, mpibool, which, ME_STRNG, &
     & MPI_COMM_STRNG, stat, ierror)
         endif ! which lt ME_STRNG
        endif ! apply Metropolis
!
       endif ! qvalid ...
! all root nodes continue (success=false for idle node(s)) :
       if (success) then
        call MPI_ALLGATHER(which, 1, mpiint, &
     & nodelist, 1, mpiint, MPI_COMM_STRNG, ierror)
!
! make entry in REX log (only lower-rank replica does this)
        if (ME_STRNG.lt.which) then
          j=ME_STRNG ! possible cast i4=>i8 accommodates in8 compilations
          i=int_vector_add(rex_log, j) ! this replica
          i=int_vector_add(rex_log, which) ! exchanged with this replica
          i=int_vector_add(rex_log, itime + rextime_offset) ! at this time
        endif ! ME_STRNG
!
!********************************************************************************************
! swap restart & traj file info; otherwise restart files will correspond to wrong replica
!#ifdef 1
! oldiol=iolev
! iolev=1 ! so that vinqre works
!#endif
!
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
! write(600+ME_STRNG,*) iunwri, new_fnames(1), &
! & iuncrd, new_fnames(2), openun(1:2)
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
!#ifdef 1
! iolev=oldiol
!#endif
! done with swap output file info
!********************************************************************************************
!
         deallocate(r_f3); nullify(r_f2);
         if (qorient) then
          nullify(r_o2);
          if (qdiffrot) then ; deallocate(r_o3);
          else ; nullify(r_o3); endif
         endif ! qorient
       else ! success ( move rejected )
        call MPI_ALLGATHER(ME_STRNG, 1, mpiint, &
     & nodelist, 1, mpiint, MPI_COMM_STRNG, ierror)
!
! move rejected, so restore string
!
        if (qvalid) then
         r_f=>r_f3; deallocate(r_f2); nullify(r_f3);
         if (qorient) then
          r_o=>r_o3; nullify(r_o3);
          if (qdiffrot) then ; deallocate(r_o2);
          else ; nullify(r_o2); endif
         endif ! qorient
        endif ! qvalid
       endif ! success
! aa
! dE=exp(-rex_beta*dE)
! call MPI_ALLGATHER(dE, 1, MPI_DOUBLE_PRECISION,
! & dEG, 1, MPI_DOUBLE_PRECISION, MPI_COMM_STRNG, bug)
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
! broadcast new reference to slaves (what about dperp0, dpar0, drms0?)
       if (success) then
#if (KEY_SINGLE==1)
        call PSND4r(r_f,27*nforced) 
#endif
#if (KEY_SINGLE==0)
        call PSND8m(r_f,27*nforced) 
#endif
        if (qorient.and.qdiffrot) then ;
#if (KEY_SINGLE==1)
         call PSND4r(r_o,27*norient) ; 
#endif
#if (KEY_SINGLE==0)
         call PSND8m(r_o,27*norient) ; 
#endif
        endif
!
! broadcast reference values
! command below does not work in NERSC with pathscale
! call mpi_bcast(rex_string_data_new,1,rex_string_data_mpi,0,MPI_COMM_LOCAL,ierror) ! broadcast to slaves
        call mpi_type_get_extent(rex_string_data_mpi, lb, extent, ierror)
         call PSND4r(rex_string_data_new,extent/4)
!
       endif ! success
      endif ! qgrp
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
          i=rex_map(j)
          rex_map(j)=rex_map(j+1)
          rex_map(j+1)=i
          j=j+1
        endif
        j=j+1
      enddo
!
      if (any(nodelist.ne.(/ (i, i=0,nstring-1) /))) &
     & call multicom_permute_string_ranks(nodelist+1) ! special-purpose routine to reorder ranks in order of string replicas
! added 1 because in multicom node indices start from 1
      if (success) then
! finish updating reference values
       dpar0 =rex_string_data_new%dpar0
       dperp0=rex_string_data_new%dperp0
       drms0 =rex_string_data_new%drms0
       kpara =rex_string_data_new%kpara
       kperp =rex_string_data_new%kperp
       krms =rex_string_data_new%krms
       qrms_upper_bound =rex_string_data_new%qrms_upper_bound
       evolve_expo_mem =rex_string_data_new%evolve_expo_mem
       num_evolve_samples=rex_string_data_new%num_evolve_samples
       avforce =rex_string_data_new%avforce
       ftsm_mini_on =rex_string_data_new%ftsm_mini_on
       evolve_expo_on =rex_string_data_new%evolve_expo_on
       evolve_aver_on =rex_string_data_new%evolve_aver_on
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) mestring=ME_STRNG
! broadcast string size to all slave nodes
       if (qgrp) then
#if (KEY_INTEGER8==0)
        call PSND4(mestring,1) 
#endif
#if (KEY_INTEGER8==1)
        call PSND8(mestring,1) 
#endif
        call set_param('MESTRING',mestring)

       endif ! qgrp
      endif ! success
! write(600+ME_GLOBAL, *) ME_STRNG
! close(600+ME_GLOBAL)
!
      end subroutine ftsm_repl_exchange
!==========================================================================================
#endif

#endif /* automatically protect all code */
      end module ftsm
!
