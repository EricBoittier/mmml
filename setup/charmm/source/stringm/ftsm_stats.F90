! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
! statistics module for the finite-temperature string method
      module ftsm_stats
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      private
      public ftsm_stat_init
      public ftsm_stat
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      contains
!============================================================
      subroutine ftsm_stat_init(comlyn, comlen)
!
      use ftsm_var
      use ftsm_rex, only: ftsm_rex_read_map
      use ftsm_connect, only: ftsm_reconnect_read_map
      use stream
      use dimens_fcm
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
!=============================================================
      character(len=len("FTSM_STAT_INIT>") ),parameter::whoami="FTSM_STAT_INIT>";!macro
      CHARACTER(LEN=*) :: COMLYN
      integer :: COMLEN
!
      character(len=80) :: rex_fname_old, connect_fname_old
      integer :: rex_flen_old, connect_flen_old
 integer :: oldiol, error
!
      logical :: qroot, qprint
!==========================================================
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
      forces_fname=''
      output_forces=.false.
!
      rmsd0_fname=''
      output_rmsd0=.false.
!
      c_fname=''
      output_curvature=.false.
!
      s_fname=''
      output_arclength=.false.
! distance to string (dpar & dperp, or drms, depending on proj_on)
      dist_fname=''
      output_dist=.false.
!
      fe_fname=''
      output_fe=.false.
      fe_curvature=.false.
      fe_curv=0d0
!
      centers_fname=''
      output_centers=.false.
!
      rex_fname_old=''
      rex_fname=''
      output_rex_log=.false.
      output_rex_map=.false.
!
      voronoi_fname=''
      output_voronoi_hist=.false.
      output_voronoi_log=.false.
      output_voronoi_map=.false.
!
      connect_fname_old=''
      connect_fname=''
      output_connect_log=.false.
      output_connect_map=.false.
!
      M_fname=''
      output_M=.false.
!
      J_fname=''
      output_J=.false.
!
!ccccccccccccccccc first process the RMSD-related commands
!============== RMSD from static structure ===============
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
!cccccccccccc print summary
       if (qprint) then
         if (rmsd0_flen.gt.0) then
          write(info,660 ) whoami,rmsd0_fname(1:rmsd0_flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         else
          write(info,661 ) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
       endif
 660 format(A,' WILL WRITE STRING RMSD TO FILE ',A)
 661 format(A,' WILL WRITE STRING RMSD TO STDOUT.')
!
      endif !! RMSD
!======================= ARCLENGTH
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
          write(info,652) whoami,s_fname(1:s_flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         else
          write(info,653) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
        endif
 652 format(A,' WILL WRITE STRING LENGTH TO FILE ',A)
 653 format(A,' WILL WRITE STRING LENGTH TO STDOUT.')
!
      endif ! ARCLENGTH
!======================================= CURVATURE
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
          write(info,6521) whoami,c_fname(1:c_flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         else
          write(info,6531) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
        endif
 6521 format(A,' WILL WRITE CURVATURE TO FILE ',A)
 6531 format(A,' WILL WRITE CURVATURE TO STDOUT.')
!
      endif ! CURVATURE
!======================================= DISTANCE TO STRING IMAGES
      if (indxa(comlyn, comlen, 'DIST').gt.0) then
        output_dist=.true.
        call gtrmwa(COMLYN, COMLEN, 'DNAM', 4, dist_fname, 80, dist_flen)
        if (dist_flen.eq.0) then
         call wrndie(0,whoami,trim('DISTANCE TO STRING FILE NAME NOT SPECIFIED. WILL WRITE TO STDOUT.'))
         dist_funit=outu
        else
         if (indxa(comlyn, comlen, 'DAPP').gt.0) then
           distform='APPEND'
         else
           distform='WRITE'
         endif
        endif
!ccccccccccc print summary
        if (qprint) then
         if (dist_flen.gt.0) then
          write(info,6522) whoami, dist_fname(1:dist_flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         else
          write(info,6523) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
        endif
 6522 format(A,' WILL WRITE DISTANCE TO STRING TO FILE ',A)
 6523 format(A,' WILL WRITE DISTANCE TO STRING TO STDOUT.')
!
      endif ! DISTANCE TO STRING
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
!
        fe_curvature=(indxa(comlyn, comlen, 'NOCV').le.0)
!ccccccccccc print summary cccccccccccccccccccccccccccccccccccccc
        if (qprint) then
         if (fe_flen.gt.0) then
          write(info,6520) whoami,fe_fname(1:fe_flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         else
          write(info,6530) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
         if (.not.fe_curvature) then
          write(info,6540) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
        endif
 6520 format(A,' WILL WRITE FREE ENERGY TO FILE ',A)
 6530 format(A,' WILL WRITE FREE ENERGY TO STDOUT.')
 6540 format(A,' FREE ENERGY PROFILE WILL EXCLUDE CONTRIBUTIONS FROM STRING CURVATURE.')
!
      endif ! F.E.
!cccccccccc process PATH CENTERS output options ccccccccccccccccccccccc
      if (indxa(comlyn, comlen, 'CENT').gt.0) then
! get file name
        call gtrmwa(COMLYN, COMLEN, 'CNAM', 4, centers_fname, 80, centers_flen)
!cccccccccccc print summary
        if (centers_flen.gt.0) then
         output_centers=.true.
         if (qprint) then
          write(info,6620 ) whoami,centers_fname(1:centers_flen) ;write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
         if (indxa(comlyn, comlen, 'CEAP').gt.0) then ! APPEND?
           cenform='APPEND' ! note: if appending, should not duplicate DCD header !
         else
           cenform='WRITE'
         endif
        else
          call wrndie(0,whoami,trim('NO FILE NAME GIVEN. WILL NOT WRITE PATH CENTERS.'))
        endif
 6620 format(A,' WILL WRITE PATH CENTERS TO FILE ',A,'.')
!
      endif ! centers output
!ccccccccccccc replica exchange map cccccccccccccc
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
          write(info,6721) whoami,rex_fname(1:rex_flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
         if (rex_flen_old.gt.0) then
          if (qprint) then
             write(info,6722) whoami,rex_fname_old(1:rex_flen_old) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
             rex_funit=-1
             call open_file(rex_funit, rex_fname_old(1:rex_flen_old), 'FORMATTED', 'READ')
          endif
!
          call ftsm_rex_read_map(rex_funit)
!
          if (qprint) call VCLOSE(rex_funit, 'KEEP', error)
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
        if (rex_flen.eq.0) then ! in the case that name was not read above
         call gtrmwa(COMLYN, COMLEN, 'RXNM', 4, rex_fname, 80, rex_flen)
        endif
! check for timestep offset
        rextime_offset=gtrmi(comlyn, comlen, 'ROFF', 0);
        if (rextime_offset.gt.0) then
         if (qprint) then ; write(info,6725) whoami, whoami,rextime_offset ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6725 format(A,' WILL OFFSET STEP COUNTER IN REPLICA EXCHANGE LOG BY ' &
     & /,A,' ',I10)
        endif
!
        if (rex_flen.gt.0) then
         output_rex_log=.true.
         if (qprint) then
          write(info,6723) whoami,whoami,rex_fname(1:rex_flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
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
!ccccccccccccc path connectivity map cccccccccccccc
      connect_flen=0
      if (indxa(comlyn, comlen, 'RECM').gt.0) then
! get file name
        call gtrmwa(COMLYN, COMLEN, 'RCNM', 4, connect_fname, 80, connect_flen)
! check if user specified an custom map (e.g. from an older run)
        call gtrmwa(COMLYN, COMLEN, 'RCOL', 4, connect_fname_old, 80, connect_flen_old)
!
        if (connect_flen.gt.0) then
         output_connect_map=.true.
         if (qprint) then
          write(info,6726) whoami,connect_fname(1:connect_flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
         if (connect_flen_old.gt.0) then
          if (qprint) then
             write(info,6727) whoami,connect_fname_old(1:connect_flen_old) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
             connect_funit=-1
             call open_file(connect_funit, connect_fname_old(1:connect_flen_old), 'FORMATTED', 'READ')
          endif
!
          call ftsm_reconnect_read_map(connect_funit)
!
          if (qprint) call VCLOSE(connect_funit, 'KEEP', error)
         endif
!
        else
          call wrndie(0,whoami,trim('NO FILE NAME GIVEN. WILL NOT WRITE REPLICA EXCHANGE MAP.'))
        endif
 6726 format(A,' WILL WRITE PATH CONNECTIVITY MAP TO FILE ',A,'.MAP')
 6727 format(A,' WILL RESTART FROM PATH CONNECTIVITY MAP IN FILE ',A)
!
      endif ! path connectivity map
!cccccccccccccc replica exchange log cccccccccccccccccccccccccccccccccccccccc
      if (indxa(comlyn, comlen, 'RECL').gt.0) then
! get file name
        if (connect_flen.eq.0) then ! in the case that name was not read above
         call gtrmwa(COMLYN, COMLEN, 'RCNM', 4, connect_fname, 80, connect_flen)
        endif
! check for timestep offset
        connect_offset=gtrmi(comlyn, comlen, 'RCOF', 0);
        if (connect_offset.gt.0) then
         if (qprint) then ; write(info,6728) whoami, whoami,connect_offset ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
 6728 format(A,' WILL OFFSET STEP COUNTER IN PATH CONNECTIVITY LOG BY ' &
     & /,A,' ',I10)
        endif
!
        if (connect_flen.gt.0) then
         output_connect_log=.true.
         if (qprint) then
          write(info,6729) whoami,connect_fname(1:connect_flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
         if (indxa(comlyn, comlen, 'RCAP').gt.0) then ! APPEND?
           connectform='APPEND'
         else
           connectform='WRITE'
         endif ! rcap
        else
          call wrndie(0,whoami,trim('NO FILE NAME GIVEN. WILL NOT WRITE PATH CONNECTIVITY LOG.'))
        endif ! connect_flen.gt.0
 6729 format(A,' WILL WRITE PATH CONNECTIVITY LOG TO FILE ',A,'.DAT')
!
      endif ! path connectivity log
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
          write(OUTU,'(A)') pack(info,info.ne.'');info='';
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
 6623 format(A,' WILL WRITE VORONOI LOG TO BINARY FILE ',A,'.LOG')
!
      endif ! complete voronoi log
!cccccccccccccccccc process forces output options cccccccccccccccccc
      if (indxa(comlyn, comlen, 'FORC').gt.0) then
! get nergy file name
        call gtrmwa(COMLYN, COMLEN, 'FCNM', 4, forces_fname, 80, forces_flen)
!ccccccccccc print summary
        if (forces_flen.gt.0) then
         output_forces=.true.
         if (qprint) then
          write(info,6625) whoami,forces_fname(1:forces_flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif
         if (indxa(comlyn, comlen, 'FCAP').gt.0) then ! APPEND?
           fform='APPEND'
         else
           fform='WRITE'
         endif
         avforce=zero;
        else
         call wrndie(0,whoami,trim('NO FILE NAME GIVEN. WILL NOT WRITE AVERAGE FORCE.'))
        endif
 6625 format(A,' WILL WRITE AVERAGE FORCE TO FILE ',A,'.')
      endif ! forces
!ccccccccccccccccc process M matrix output options ccccccccccccccccc
      if (indxa(comlyn, comlen, 'MMAT').gt.0) then
!
       if (ftsm_com_on) then
        call wrndie(0,whoami,trim('FTSM IN COM COORDINATES DOES NOT CURRENTLY SUPPORT M TENSOR CALCULATION.'))
       else
!
        call gtrmwa(COMLYN, COMLEN, 'MNAM', 4, M_fname, 80, M_flen)
        if (M_flen.gt.0) then
         output_M=.true.
         if (qprint) then ; write(info,6626) whoami,M_fname(1:M_flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
        else
         call wrndie(0,whoami,trim('NO FILE NAME GIVEN. WILL NOT WRITE M TENSOR.'))
        endif
 6626 format(A,' WILL WRITE M TENSOR TO FILE ',A,'.')
       endif
!
      endif
!ccccccccccccccccc process Jacobian output options ccccccccccccccccc
      if (indxa(comlyn, comlen, 'JACO').gt.0) then
!
       if (ftsm_com_on) then
        call wrndie(0,whoami,trim('FTSM IN COM COORDINATES DOES NOT CURRENTLY SUPPORT JACOBIAN CALCULATION.'))
        return
       else
!
        call gtrmwa(COMLYN, COMLEN, 'JNAM', 4, J_fname, 80, J_flen)
        if (J_flen.gt.0) then
         output_J=.true.
         if (qprint) then ; write(info,6628) whoami,J_fname(1:J_flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif
        else
         call wrndie(0,whoami,trim('NO FILE NAME GIVEN. WILL NOT WRITE JACOBIAN.'))
        endif
 6628 format(A,' WILL WRITE JACOBIAN TO FILE ',A,'.')
       endif ! jaco
      endif
!===================================================================

      if (qprint) iolev=oldiol

!
! if we got this far, we are probably OK
      stat_initialized=.true.
!
      end subroutine ftsm_stat_init
!======================================================================
      subroutine ftsm_stat()
      use ftsm_var
      use ftsm_rex, only: ftsm_rex_print_map, ftsm_rex_print_log
      use ftsm_connect, only: ftsm_reconnect_print_map, ftsm_reconnect_print_log
      use ftsm_util, only: ftsm_update_overlap_coor
      use ftsm_voronoi
      use ftsm_io
      use ftsm_compute
      use stream
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use string
      use mpi
      use number
      use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3
      use parallel, only: psnd4, psnd8
!
      integer :: a, i, j, k, fmt_len, ifile
!

      integer :: oldiol

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
      character(len=80) :: fmt_real, fmt_int, fmt
      real(chm_real) :: projdist(3) ! dpar, dperp, drms
      real(chm_real) :: u (3,3)= Id3
      real(chm_real) :: rmsd0, rmsd0_all(nstring), fc_all(3,nstring)
      real(chm_real), pointer :: M_all(:,:,:,:,:) ! Mtensor combined from all images
      real(chm_real), pointer :: J_all(:) ! combined jacobian
!
      integer :: ierror
      character(len=len("FTSM_STAT>") ),parameter::whoami="FTSM_STAT>";!macro
!
      logical :: qroot, qprint, qgrp
!
      qroot=MPI_COMM_STRNG.ne.MPI_COMM_NULL
      qprint=qroot.and.ME_STRNG.eq.0
      qgrp=MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1
!
! ad hoc fix for REX :
! when string ranks are permuted, it might happen that a new root is silent
      if (qprint) then ; oldiol=iolev; iolev=0; endif
!ccccccccccccccccccccccc begin ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! check if the user has made an initialization call
      if (.not.ftsm_initialized) then
       call wrndie(0,whoami,trim('FTSM NOT INITIALIZED. NOTHING DONE.'))
       return
      endif
!
      if (.not.stat_initialized) then
       call wrndie(0,whoami,trim('NO OUTPUT OPTIONS SELECTED. NOTHING DONE'))
       return
      endif
!
      stat_iteration_counter=stat_iteration_counter+1
      if (qroot) then
! define number format strings for output
       write(fmt_int,'(I5)') stat_iteration_counter
       write(fmt_real,*) nstring
       fmt_len=len(fmt_real)
       call trima(fmt_real, fmt_len)
      endif
!=================================================================================
      if (output_rmsd0) then
! compute rmsd
       if (qorient) then
        if (qdiffrot) call ftsm_update_overlap_coor(ione) ! just in case, make sure orientation coordiantes are up-to-date
        call RMSBestFit( r_o(:,:,ref), r_o(:,:,center), &
     & orientWeights, u ) ! superpose ref onto center
       endif
       rmsd0=rmsd( r_f(:,:,ref), matmul ( r_f(:,:,center), u ), & ! rotate center using transpose of u
     & forcedWeights ) ! note: I am assuming that COMs have been removed, which should be true
!
          if (qroot) call mpi_gather(rmsd0,1,MPI_DOUBLE_PRECISION & ! heads communicate
     & ,rmsd0_all,1,MPI_DOUBLE_PRECISION,0, &
     & MPI_COMM_STRNG, ierror)
          if (qprint) then ! root writes
           if (rmsd0_funit.eq.outu) then
            fmt='("RMSD0> '//fmt_int(1:5)//' ",'                        &
     & //fmt_real(1:fmt_len)//real_format//')'
           else
            rmsd0_funit=-1
            call open_file(rmsd0_funit, rmsd0_fname, 'FORMATTED', rform)
            fmt='("'//fmt_int(1:5)//' ",'                               &
     & //fmt_real(1:fmt_len)//real_format//')'
           endif
           write(rmsd0_funit,fmt) (rmsd0_all(i),i=1,nstring)
!
           if (rmsd0_funit.ne.outu) then
            call VCLOSE(rmsd0_funit, 'KEEP', ierror)
           endif
          endif ! qprint
          rform='APPEND'
      endif
!===========================================================================
      if (output_arclength) then
       if (qprint) then
        if (repa_initialized) then ! proceed only if arclength defined
!
         if (s_funit.eq.outu) then
          fmt='("ARCL> '//fmt_int(1:5)//' ",'                           &
     & //fmt_real(1:fmt_len)//real_format//')'
         else
          s_funit=-1
          call open_file(s_funit, s_fname, 'FORMATTED', sform)
          fmt='("'//fmt_int(1:5)//' ",'//fmt_real(1:fmt_len)            &
     & //real_format//')'
         endif
!
         write(s_funit, fmt) ds * sqrt3 ! correction factor for consistency with atomic RMSD
! flush unit: close and reopen later
         if (s_funit.ne.outu) then
          call VCLOSE(s_funit, 'KEEP', ierror)
         endif
        else ! repa
          call wrndie(0,whoami,trim('NO REPARAMETRIZATION OPTIONS SELECTED. SKIPPING ARCLENGTH.'))
        endif
       endif ! qprint
       sform='APPEND'
      endif ! output_arclength
!===========================================================================
      if (output_curvature) then
       if (qprint) then
        if (repa_initialized) then ! curvature computed by repa routines
!
         if (c_funit.eq.outu) then
          fmt='("CURV> '//fmt_int(1:5)//' ",'                           &
     & //fmt_real(1:fmt_len)//real_format//')'
         else
          c_funit=-1
          call open_file(c_funit, c_fname, 'FORMATTED', cform)
          fmt='("'//fmt_int(1:5)//' ",'//fmt_real(1:fmt_len)            &
     & //real_format//')'
         endif
!
         write(s_funit, fmt) curv / sqrt(3d0) ! correction factor for consistency with atomic RMSD
! flush unit: close and reopen later
         if (c_funit.ne.outu) then
          call VCLOSE(c_funit, 'KEEP', ierror)
         endif
        else
          call wrndie(0,whoami,trim('NO REPARAMETRIZATION OPTIONS SELECTED. SKIPPING CURVATURE.'))
        endif
       endif
       cform='APPEND'
      endif ! output_curvature
!===========================================================================
      if (output_dist) then ! projection distance variables
! gather on root processor
       if (qroot) then
        projdist(1)=dpar; projdist(2)=dperp; projdist(3)=drms
! reuse fc_all
        call mpi_gather(projdist,3,mpifloat, & ! heads communicate
     & fc_all,3,mpifloat,0, &
     & MPI_COMM_STRNG, ierror)
!
        fmt='('//fmt_real(1:fmt_len)//real_format//')'
        if (qprint) then
         if (dist_funit.eq.outu) then
          write(dist_funit,'("PROJECTION DISTANCE> ",I8)') stat_iteration_counter
         else
          dist_funit=-1
          call open_file(dist_funit, dist_fname, 'FORMATTED', distform)
          write(dist_funit,'("% ",I8)') stat_iteration_counter ! % is a MATLAB comment
         endif
!
! print
         if (proj_on) then
          write(dist_funit, fmt) fc_all(1,:) ! parallel distance component
          write(dist_funit, fmt) fc_all(2,:) ! perpendicular distance component
         else
          write(dist_funit, fmt) fc_all(3,:) ! total distance
         endif
!
! flush unit: close and reopen later
         if (dist_funit.ne.outu) &
     & call VCLOSE(dist_funit, 'KEEP', ierror)
        endif ! qprint
        distform='APPEND'
       endif ! qroot
!
      endif ! output_dist
!===========================================================================
      if (output_fe) then
! compute free energy
       call ftsm_compute_fe_fd()
       if (qprint) then
         if (fe_funit.eq.outu) then
          fmt='("FE> '//fmt_int(1:5)//' ",'                             &
     & //fmt_real(1:fmt_len)//real_format//')'
         else
          fe_funit=-1
          call open_file(fe_funit, fe_fname, 'FORMATTED', feform)
          fmt='("'//fmt_int(1:5)//' ",'//fmt_real(1:fmt_len)            &
     & //real_format//')'
         endif
!
! print
         write(fe_funit, fmt) fe
! flush unit: close and reopen later
         if (fe_funit.ne.outu) then
          call VCLOSE(fe_funit, 'KEEP', ierror)
         endif
       endif ! qprint
       feform='APPEND'
      endif
!===========================================================================
      if (output_rex_map) then ! output replica exchange map
       if (rex_flen.eq.0) then
        call wrndie(0,whoami,trim('NO FILE NAME SPECIFIED. WILL NOT WRITE REPLICA EXCHANGE MAP.'))
       else
        if (qprint) then
         rex_funit=-1
         rex_fname(rex_flen+1:rex_flen+4)='.map' ! append to name
         call open_file(rex_funit, rex_fname(1:rex_flen+4), 'FORMATTED', 'WRITE')
         rex_fname(rex_flen+1:)='' ! erase extension
         call ftsm_rex_print_map(rex_funit) ! all processes enter
!
         call VCLOSE(rex_funit, 'KEEP', ierror)
        endif ! qprint
       endif
      endif
!===========================================================================
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
        call ftsm_rex_print_log(rex_funit)
! flush unit:
        if (qprint) call VCLOSE(rex_funit, 'KEEP', ierror)
       endif
      endif
!===========================================================================
      if (output_connect_map) then ! output ftsm path connectivity map
       if (connect_flen.eq.0) then
        call wrndie(0,whoami,trim('NO FILE NAME SPECIFIED. WILL NOT WRITE PATH CONNECTIVITY MAP.'))
       else
        if (qprint) then
         connect_funit=-1
         connect_fname(connect_flen+1:connect_flen+4)='.map' ! append to name
         call open_file(connect_funit, connect_fname(1:connect_flen+4), 'FORMATTED', 'WRITE')
         connect_fname(connect_flen+1:)='' ! erase extension
         call ftsm_reconnect_print_map(connect_funit) ! all processes enter
!
         call VCLOSE(connect_funit, 'KEEP', ierror)
        endif ! qprint
       endif
      endif
!===========================================================================
      if (output_connect_log) then
       if (connect_flen.eq.0) then
         call wrndie(0,whoami,trim('NO FILE NAME SPECIFIED. WILL NOT WRITE PATH CONNECTIVITY LOG.'))
       else
        if (qprint) then
         connect_funit=-1
         connect_fname(connect_flen+1:connect_flen+4)='.dat' ! append to name
         call open_file(connect_funit, connect_fname(1:connect_flen+4), 'FORMATTED', connectform)
         connect_fname(connect_flen+1:)='' ! erase extension
        endif
        call ftsm_reconnect_print_log(connect_funit) ! all processes can call (and clear their logs)
        if (qprint) then
         call VCLOSE(connect_funit, 'KEEP', ierror)
        endif
        connectform='APPEND'
       endif
      endif
!===========================================================================
      if (output_centers) then
        if (qprint) then
         centers_funit=-1
         call open_file(centers_funit, centers_fname, 'UNFORMATTED', cenform)
        endif
! write current centers as a trajectory frame (adopted from write_dcd)
!--------------------------------------------------------------------
! header will be written only if IBEG=1, so be careful, or will have a corrupt file
! however, if header missing, can cat file to another trajectory file w header (e.g. "cat path1.dcd path2.dcd > path1-2.dcd" )
! VO 10.2012 : "cheat" by looking at whether the append option is set
        if (cenform.eq.'APPEND') then ; i=(stat_iteration_counter-1) * nstring + 1 ; else ; i=1 ; endif
        call ftsm_write_dcd(IFILE=centers_funit, &
     & IBEG=i, &
! & IBEG=(stat_iteration_counter-1) * nstring + 1, &
! & IEND=2**31-1) ! largest 4-byte integer
     & IEND= ione*(-1 + 2**30 + 2**30)) ! possible i4 => i8 cast
! NOTE: should write the correct number of records to header at the end of calculation
!--------------------------------------------------------------------
! flush unit:
        if (qprint) call VCLOSE(centers_funit, 'KEEP', ierror)
        cenform='APPEND'
      endif
!===========================================================================
      if (output_forces) then ! NOTE: these are the forces acting on the projection variables
! gather on root processor
       if (qroot) then
        call mpi_gather(avforce,3,MPI_DOUBLE_PRECISION, & ! heads communicate
     & fc_all,3,MPI_DOUBLE_PRECISION,0, &
     & MPI_COMM_STRNG, ierror)
!
        fmt='('//fmt_real(1:fmt_len)//real_format//')'
        if (qprint) then
         if (forces_funit.eq.outu) then
          write(forces_funit,'("FORCES> ",I8)') stat_iteration_counter ! % is a MATLAB comment
         else
          forces_funit=-1
          call open_file(forces_funit, forces_fname, 'FORMATTED', fform)
          write(forces_funit,'("% ",I8)') stat_iteration_counter ! % is a MATLAB comment
         endif
!
! print
         write(forces_funit, fmt) fc_all(1,:) ! parallel forces
         if (fe_curvature) write(forces_funit, fmt) fc_all(2,:) ! curvature forces
         write(forces_funit, fmt) fc_all(3,:) ! perpendicular forces
!
! flush unit: close and reopen later
         if (forces_funit.ne.outu) &
     & call VCLOSE(forces_funit, 'KEEP', ierror)
        endif ! qprint
        fform='APPEND'
       endif ! qroot
!
      endif ! output_force
!===========================================================================
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
         call ftsm_voronoi_print_data(ifile) ! all root processes enter
         if (qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ; endif
        endif
      endif
!===========================================================================
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
         if (qroot) call ftsm_voronoi_print_log(vlog_funit)
! flush unit:
         if (qprint) then ; call VCLOSE(vlog_funit, 'KEEP', ierror) ; endif
       endif
      endif
!===========================================================================
      if (output_voronoi_map) then ! output voronoi map
        if (voronoi_flen.eq.0) then
         call wrndie(0,whoami,trim('NO FILE NAME SPECIFIED. WILL NOT WRITE VORONOI MAP.'))
        else
! put 'whereami' into the map
          if (qroot.and.SIZE_STRNG.gt.1) then
! GATHER +BCAST
           call MPI_GATHER(ftsm_voronoi_whereami, 1, mpiint, ftsm_voronoi_map, 1, mpiint, 0, MPI_COMM_STRNG, ierror)
           call mpi_bcast(ftsm_voronoi_map,nstring,mpiint,0,MPI_COMM_STRNG,ierror)
          else
           ftsm_voronoi_map(mestring+1)=ftsm_voronoi_whereami
          endif
          if (qgrp) then
#if (KEY_INTEGER8==0)
           call PSND4(ftsm_voronoi_map,nstring) 
#endif
#if (KEY_INTEGER8==1)
           call PSND8(ftsm_voronoi_map,nstring) 
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
          call ftsm_voronoi_print_map(ifile)
          if (qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ; endif
         endif ! qroot
        endif
      endif
!===========================================================================
      if (output_M) then
! if running in parallel, combine partial M entries
       if (qroot) then
        if (M_flen.eq.0) then
         call wrndie(0,whoami,trim('NO FILE NAME SPECIFIED. WILL NOT WRITE M TENSOR.'))
        else
         ifile=-1
         call open_file(ifile, M_fname(1:M_flen), 'FORMATTED', 'WRITE')
! essentially duplicated code from cv_common
         i=9*nforced*nforced
         allocate(M_all(3, 3, nforced, nforced, nstring))
!
         if (SIZE_STRNG.gt.1) then
          call MPI_GATHER(Mtensor(1,1,1,1,2),i,mpifloat, & ! gather all tensors on root for output
     & M_all, i,mpifloat, &
     & 0,MPI_COMM_STRNG,ierror)
         else
          M_all(:,:,:,:,1)=Mtensor(:,:,:,:,2);
         endif
!
         write(fmt,*) 3*nforced
         if (ME_STRNG.eq.0) then ! root writes
          do j=1,nstring
           write(ifile,'("% ",I5)') j
           write(ifile,'('//fmt//real_format//')') ( (M_all(:,a,:,k,j), a=1,3), k=1,nforced )
          enddo
         endif ! ME
         deallocate(M_all)
         call VCLOSE(ifile, 'KEEP', ierror)
        endif ! M_flen
       endif ! qroot
      endif ! output_M
!===========================================================================
      if (output_J) then
! if running in parallel, combine partial J entries
       if (qroot) then
        if (J_flen.eq.0) then
         call wrndie(0,whoami,trim('NO FILE NAME SPECIFIED. WILL NOT WRITE M TENSOR.'))
        else
         ifile=-1
         call open_file(ifile, J_fname(1:J_flen), 'FORMATTED', 'WRITE')
! essentially duplicated code from above
         i=1
         allocate(J_all(nstring))
!
         if (SIZE_STRNG.gt.1) then
          call MPI_GATHER(Jacobian(2),i,mpifloat, & ! gather on root for output
     & J_all, i,mpifloat, &
     & 0,MPI_COMM_STRNG, ierror)
         else
          Jacobian(1)=Jacobian(2);
         endif
!
         if (ME_STRNG.eq.0) then ! root writes
          write(fmt,*) nstring
          write(ifile,'('//fmt//real_format//')') J_all
         endif ! ME
         deallocate(J_all)
         call VCLOSE(ifile, 'KEEP', ierror)
        endif ! J_flen
       endif ! qroot
      endif ! output_J
!
! ad hoc fix for REX
      if (qprint) iolev=oldiol

!===========================================================================
! reset force averages -- relevant for f.e.
      num_force_samples=0
!
      end subroutine ftsm_stat
!
!===========================================================================
#endif

#endif /* automatically protect all code */
      end module ftsm_stats
!
