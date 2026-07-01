module abpo
! Module for Adaptively Biased Path Optimization
#if KEY_ABPO==1
   use chm_kinds
   use dimens_fcm
   use abpo_ltm
   use abpo_collvar
   use ensemble, only:whoiam, old_mynod, nensem, lmasternode, comm_master
   !
   implicit none
   private
   public :: abpo_setup, abpo_cntrl, abpo_ener
   type(cv_ptr_wrapper), allocatable, dimension(:) :: cv_array
   real(kind=chm_real), allocatable, dimension(:, :) :: path, iD2_path
   real(kind=chm_real), allocatable, dimension(:) :: current_cv
   real(kind=chm_real), allocatable, dimension(:) :: cv_force
   real(kind=chm_real), allocatable, dimension(:, :) :: norm 
   real(kind=chm_real), allocatable, dimension(:, :) :: mean 
   real(kind=chm_real), allocatable, dimension(:, :) :: tot_mean 
   real(kind=chm_real), allocatable, dimension(:) :: h
   real(kind=chm_real), allocatable, dimension(:) :: tot_h
   real(kind=chm_real), allocatable, dimension(:, :) :: abpo_iD, abpo_iD2
   integer, allocatable, dimension(:) :: cnt
   integer, allocatable, dimension(:) :: tot_cnt
   real(kind=chm_real), allocatable, dimension(:) :: dh
   real(kind=chm_real), allocatable, dimension(:) :: delta
   real(kind=chm_real), allocatable, dimension(:) :: d_delta
   real(kind=chm_real) :: abpo_b, abpo_c, moll, sigma, min_cnt 
   real(kind=chm_real) :: abpo_temp, abpo_rbeta, rtube, ftube
   real(kind=chm_real) :: smooth_strength, hdl
   integer, parameter :: max_n_point = 50000
   integer, parameter :: n_sigma = 3
   integer, parameter :: cvu = 70, dcdu = 71, rstu = 72, prstu = 73
   integer :: n_point, n_cv, bcyc, ecyc, current_lambda, dfrq
   integer :: block_steps, max_nblock, iblock, abpo_d, cvfrq
   integer :: n_span, n_span2, abpo_step, D_step, D_nsteps
   logical :: q_debug = .false.
   logical :: q_cbias = .false.
   logical :: q_D = .false.
   logical :: q_loop = .false.
   logical :: q_rest = .false.
   character(len=80) :: cycdir, repdir

   contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Public procedures                      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine abpo_setup(comlyn, comlen)
      ! Parse command line keywords to
      !  define various parameters
      !  and the collective variables
         use string
         use stream
         use memory 
         use consta
         use number
         use ensemble, only: ensprint
         !
         character(len=*) :: comlyn
         integer :: comlen
         !
         character(len=4) :: cmd
         !
         cmd = nexta4(comlyn, comlen)
         select case (cmd)
         case ('SETC')
            ! Define collective variables
            call parse_cv(comlyn, comlen, cv_array, n_cv)
            q_abpo = .true.
         case ('DTNS')
            ! Set up D tensor evaluation
            if (.not. q_abpo) call wrndie(-2, '<abpo_setup>', &
               'Collective variables not defined yet.')
            call abpo_pre_evalD(comlyn, comlen)
         case ('OPTI')
            ! Set up path optimization
            if (.not. q_abpo) call wrndie(-2, '<abpo_setup>', &
               'Collective variables not defined yet.')
            call abpo_pre_opti(comlyn, comlen)
         case default
            if (prnlev >= 3) write(outu, *) 'Cannot recognize the subcommand ', cmd
            call wrndie(0, '<abpo_setup>', &
               'Invalid subcommand. Check syntax.')
         end select
      end subroutine abpo_setup

      subroutine abpo_cntrl(comlyn, comlen)
      ! Control the dynamics to 
      !  evaluate the D matrix OR
      !  carry out path optimization cycles
         use number
         use stream
         use memory
         use dcntrl_mod, only: dynopt
         use rndnum
         use comand, only: comlyn_save, comlen_save, sav_comlyn
         use coord, only: x, y, z
         use machio,only:vopen
         use mpi
         !
         character(len=*) comlyn
         integer comlen
         !
         integer :: ic, ib
         integer :: i, j, ierr
         real(kind=chm_real) :: dist
         real(kind=chm_real), dimension(n_cv, n_cv) :: sum_D
         character(len=80) :: dcdfile, rstfile, prstfile, cvfile
         logical :: qrst, qerr, qopen, dummyq, done
         !
         q_abpo_ener = .true.
         if (q_D) then
            ! Evaluate the D_matrix if q_D is set
            ! Create directory for the current task
            call newdir('eval_d', old_mynod == 0)
            ! Reset D_step, abpo_iD
            abpo_iD = ZERO
            D_step = 0
            ! Setup output files
            write(dcdfile, '("eval_d/rep", I3.3, ".dcd")') whoiam
            write(rstfile, '("eval_d/rep", I3.3, ".rst")') whoiam
            ! iolev is tested inside vopen 
            call vopen(dcdu, dcdfile, 'UNFORMATTED', 'WRITE', qerr, 0)
            call vopen(rstu, rstfile, 'FORMATTED', 'WRITE', qerr, 0)
            ! Call dynopt to run dynamics
            call sav_comlyn(comlyn, comlen)
            call replace(comlyn_save, comlen_save, 'IUNC', dcdu)
            call replace(comlyn_save, comlen_save, 'IUNW', rstu)
            call replace(comlyn_save, comlen_save, 'NSTE', D_nsteps)
            call dynopt(comlyn_save, comlen_save)
            call vclose(dcdu, 'KEEP', qerr)
            call vclose(rstu, 'KEEP', qerr)
            ! Post-process the D matrix and invert it
            if (lmasternode) &
               call asum_comm(abpo_iD, sum_D, comm_master, n_cv*n_cv)
               !call mpi_reduce(abpo_iD, sum_D, n_cv*n_cv, mpi_real8, mpi_sum, &
               !   0, comm_master, ierr)
            do i = 1, n_cv
               do j = i + 1, n_cv
                  sum_D(j, i) = sum_D(i, j)
               end do
            end do
            abpo_iD = sum_D / (D_step * nensem) 
            call gauss_inv(abpo_iD, n_cv)
            ! No need to broadcast to all nodes because 
            !  path optimization cycles won't follow
            if (old_mynod == 0) &
               call save_path(abpo_iD, n_cv, 'invd', '.')
         else ! q_D
            ! Enter the loop of path optimization cycles 
            qrst = .false.
            do ic = bcyc, ecyc
               cycdir = ' '
               write(cycdir, '("./cyc",I3.3)') ic
               repdir = ' '
               write(repdir, '("./cyc",I3.3,"/rep",I3.3)') ic, whoiam
               if (q_rest) then
                  if (prnlev >= 3) write(outu, '(a, i2)') &
                     "abpo_cntrl: Restarting ABPO path optimization, Cycle", ic
                  qrst = .true.
                  write(rstfile, '(A, "/block", I3.3, ".rst")') &
                     trim(repdir), iblock
               else
                  if (prnlev >= 3) write(outu, '(a, i2)') &
                     "abpo_cntrl: Initiating ABPO path optimization, Cycle", ic
                  ! Create directory for the current cycle
                  call newdir(cycdir, old_mynod == 0)
                  call newdir(repdir, lmasternode)
                  ! Reset abpo_step, mean, histograms
                  abpo_step = 1
                  h = ZERO
                  dh = ZERO
                  cnt = 0
                  mean = ZERO
                  iblock = 0
               end if
               done = .false.
               do i = 1, n_point
                  iD2_path(i, :) = matmul(abpo_iD2, path(i, :))
               end do
               ! Evaluate current cv and lambda values 
               do i = 1, n_cv
                  call cv_array(i)%p%cv_eval(current_cv(i), x, y, z)
               end do
               call update_lambda(current_cv, current_lambda, dist, .true.)
               ! Enter the loop of blocks of ABP dynamics
               do 
                  ! Combine the sampling from all reps to check whether to finish 
                  !  the current cycle
                  if (lmasternode) then
                     if (old_mynod == 0) then
                        call asum_comm(h, tot_h, comm_master, n_point)
                        if (prnlev >= 3) write(outu, *) 'ABPO_CNTRL: Minimun slice count is', &
                           minval(tot_h)
                        if (minval(tot_h) >= min_cnt) done = .true.
                     else
                        call asum_comm(h, h, comm_master, n_point)
                     end if
                  end if
                  call psnd4_world(done, 1)
                  if (done) then
                     if (prnlev >= 3) write(outu, *) 'ABPO_CNTRL: The threshold is reached. &
                        Finishing the current cycle.'
                     exit
                  else if (iblock >= max_nblock) then
                     if (prnlev >= 3) write(outu, *) 'ABPO_CNTRL: The threshold is not reached, &
                        but MNBLocks is reached. Finishing the current cycle.'
                     exit
                  else
                     if (prnlev >= 3) write(outu, *) 'ABPO_CNTRL: The threshold is not reached. &
                        Continuing with the current cycle.'
                  end if
                  ! Increment iblock
                  iblock = iblock + 1
                  ! Setup output files
                  prstfile = rstfile
                  write(rstfile, '(A, "/block", I3.3, ".rst")') &
                     trim(repdir), iblock
                  write(dcdfile, '(A, "/block", I3.3, ".dcd")') &
                     trim(repdir), iblock
                  write(cvfile, '(A, "/block", I3.3, ".cv")') &
                     trim(repdir), iblock
                  ! iolev is tested inside vopen 
                  call vopen(dcdu, dcdfile, 'UNFORMATTED', 'WRITE', qerr, 0)
                  call vopen(rstu, rstfile, 'FORMATTED', 'WRITE', qerr, 0)
                  if (cvfrq > 0) call vopen(cvu, cvfile, 'FORMATTED', 'WRITE', qerr, 0)
                  ! Call dynopt to run dynamics
                  call sav_comlyn(comlyn, comlen)
                  call replace(comlyn_save, comlen_save, 'IUNC', dcdu)
                  call replace(comlyn_save, comlen_save, 'IUNW', rstu)
                  call replace(comlyn_save, comlen_save, 'NSTE', block_steps)
                  call replace(comlyn_save, comlen_save, 'ISVF', block_steps)
                  if (qrst) then
                     call vopen(prstu, prstfile, 'FORMATTED', 'READ', qerr, 0)
                     call replace(comlyn_save, comlen_save, 'IUNR', prstu)
                     call rmv(comlyn_save, comlen_save, 'STAR', dummyq)
                     call append(comlyn_save, comlen_save, 'REST')
                     call rmvni(comlyn_save, comlen_save, 'ISEE', nrand, dummyq)
                  end if
                  call dynopt(comlyn_save, comlen_save)
                  call vclose(dcdu, 'KEEP', qerr)
                  call vclose(rstu, 'KEEP', qerr)
                  if (qrst) call vclose(prstu, 'KEEP', qerr)
                  if (cvfrq > 0) call vclose(cvu, 'KEEP', qerr)
                  ! Save snapshot and histograms
                  call save_snap()
                  call save_h()
               end do
               ! Combine info from all nodes, and
               !  save the principal curve, smoothed principal curve, and PMF
               if (lmasternode) then
                  tot_mean = mean
                  call psnd8_comm(comm_master, tot_mean, n_point*n_cv)
                  call shift_path(mean, tot_mean)
                  tot_mean = mean
                  call fold(tot_mean, n_point)
                  call save_path(tot_mean, n_point, 'mean', repdir)
                  call save_pmf(h, cnt, n_point, 'pmf', repdir)
                  do i = 1, n_point
                     mean(i, :) = mean(i, :) * cnt(i)
                  end do
                  call asum_comm(mean, tot_mean, comm_master, n_point*n_cv)
                  call isum_comm(cnt, tot_cnt, comm_master, n_point)
               end if
               if (old_mynod == 0) then
                  do i = 1, n_point
                     if (tot_cnt(i) == 0) then
                        mean(i, :) = ZERO
                     else
                        mean(i, :) = tot_mean(i, :) / tot_cnt(i)
                     end if
                  end do
                  call fold(mean, n_point)
                  call save_path(mean, n_point, 'mean', cycdir)
                  call smooth(mean, n_point, smooth_strength, tot_cnt, path) 
                  call interpolate(mean, n_point)
                  call save_path(path, n_point, 'smean', cycdir)
                  call save_pmf(tot_h, tot_cnt, n_point, 'pmf', cycdir)
               end if
               ! Broadcast the new path to all nodes
               call psnd8_world(path, n_point * n_cv)
               call build_path()
               q_rest = .false.
               qrst = .true.
            end do
         end if ! q_D
         q_abpo_ener = .false.
         comlyn = ' '
         comlen = 0
      end subroutine abpo_cntrl

      subroutine abpo_ener(x, y, z, dx, dy, dz)
      ! Called by the old_ENERGY subroutine
      ! If q_D is set
      !     evaluate the D matrix
      ! If q_D is not set
      !     evaluate and apply the biasing force
         real(kind=chm_real) x(*), y(*), z(*), dx(*), dy(*), dz(*)
         !
         if (q_D) then
            if (mod(abpo_step, dfrq) == 0) call eval_D(x, y, z)
         else
            call apply_bias(x, y, z, dx, dy, dz)
         end if
         abpo_step = abpo_step + 1
      end subroutine abpo_ener

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Private procedures                      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine abpo_pre_evalD(comlyn, comlen)
      ! Parse command line keywords to
      !  define various parameters
      !  and the collective variables
         use string
         use stream
         use memory 
         use consta
         use number
         use ensemble, only: ensprint
         !
         character(len=*) :: comlyn
         integer :: comlen
         !
         integer :: i
         !
         if (prnlev >= 3) write(outu,*) "abpo_pre_evalD: Preparing for D tensor evaluation" 
         q_D = .true.
         q_debug = checque(comlyn, 'DEBU')
         dfrq = gtrmi(comlyn, comlen, 'DFRQ', 100)
         D_nsteps = gtrmi(comlyn, comlen, 'DSTE', 1000)
         call chmalloc('abpo.src', 'abpo_pre_evalD', 'abpo_iD', &
            n_cv, n_cv, crl=abpo_iD)
      end subroutine abpo_pre_evalD

      subroutine abpo_pre_opti(comlyn, comlen)
      ! Parse command line keywords to
      !  define various parameters
      !  and the collective variables
         use string
         use stream
         use memory 
         use consta
         use number
         use ensemble, only: ensprint
         !
         character(len=*) :: comlyn
         integer :: comlen
         !
         integer :: i
         logical :: q
         character(len=100) :: keys
         !
         q_D = .false.
         ! Parse general params from cmd line 
         call trime(comlyn, comlen)
         q_rest = checque(comlyn, 'REST')
         bcyc = gtrmi(comlyn, comlen, 'BCYC', 1)
         ecyc = gtrmi(comlyn, comlen, 'ECYC', bcyc)
         q_debug = checque(comlyn, 'DEBU')
         write(cycdir, '("./cyc",I3.3)') bcyc
         if (q_rest) then 
            if (prnlev >= 3) write(outu,*) &
               "abpo_pre_opti: Invoking ABPO path optimization in RESTART mode." 
            ! Restart run, load params from path.snap
            call load_snap_pars()
            ! Parse params that can be overwritten from cmd line
            min_cnt = gtrmf(comlyn, comlen, 'MINC', min_cnt)
            max_nblock = gtrmi(comlyn, comlen, 'MNBL', max_nblock)
            block_steps = gtrmi(comlyn, comlen, 'BSTE', block_steps)
            smooth_strength = gtrmf(comlyn, comlen, 'SMOO', smooth_strength)
            cvfrq = gtrmi(comlyn, comlen, 'CVFR', cvfrq)
            ! Ignore params that cannot be overwritten
            keys = ''
            call rmv(comlyn, comlen, 'CBIA', q)
            if (q) keys = trim(keys) // ' CBIA'
            call rmv(comlyn, comlen, 'LOOP', q)
            if (q) keys = trim(keys) // ' LOOP'
            call rmvni(comlyn, comlen, 'NPNT', 1, q)
            if (q) keys = trim(keys) // ' NPNT'
            call rmvf(comlyn, comlen, 'MOLL', q)
            if (q) keys = trim(keys) // ' MOLL'
            call rmvf(comlyn, comlen, 'BFCT', q)
            if (q) keys = trim(keys) // ' BFCT'
            call rmvf(comlyn, comlen, 'PRED', q)
            if (q) keys = trim(keys) // ' PRED'
            call rmvf(comlyn, comlen, 'TEMP', q)
            if (q) keys = trim(keys) // ' TEMP'
            call rmvf(comlyn, comlen, 'RTUB', q)
            if (q) keys = trim(keys) // ' RTUB'
            call rmvf(comlyn, comlen, 'FTUB', q)
            if (q) keys = trim(keys) // ' FTUB'
            if (len(keys) > 0 .and. prnlev >= 3) write(outu, '(a, a)') &
                  "Warning: The following keywords are ignored for &
                  restart run: ", trim(keys)
         else
            if (prnlev >= 3) write(outu,*) &
               "abpo_pre_opti: Envoking ABPO path optimization in START mode." 
            ! Fresh start, parse everything from cmd line
            min_cnt = gtrmf(comlyn, comlen, 'MINC', 10.)
            max_nblock = gtrmi(comlyn, comlen, 'MNBL', 20)
            block_steps = gtrmi(comlyn, comlen, 'BSTE', 1000)
            smooth_strength = gtrmf(comlyn, comlen, 'SMOO', 0.05)
            cvfrq = gtrmi(comlyn, comlen, 'CVFR', 100)
            q_cbias = checque(comlyn, 'CBIA')
            q_loop = checque(comlyn, 'LOOP')
            n_point = gtrmi(comlyn, comlen, 'NPNT', 100)
            moll = gtrmf(comlyn, comlen, 'MOLL', 0.05)
            sigma = n_point * moll
            abpo_b = gtrmf(comlyn, comlen, 'BFCT', 0.8)
            abpo_d = gtrmi(comlyn, comlen, 'PRED', 1000)
            abpo_c = abpo_d * 1.0 / n_point
            abpo_temp = gtrmf(comlyn, comlen, 'TEMP', -1.0)
            rtube = gtrmf(comlyn, comlen, 'RTUB', 1.0)
            ftube = gtrmf(comlyn, comlen, 'FTUB', 1.0)
            ! Make sure the parameters are not insane
            if (bcyc > ecyc) call wrndie(-2, '<abpo_pre_opti>', &
               'ECYC cannot be smaller than BCYC.')
            if (n_point <= 2) call wrndie(-2, '<abpo_pre_opti>', &
               'NPNT is too small, use a greater value.')
            if (sigma <= 1) call wrndie(-2, '<abpo_pre_opti>', &
               'MOLL is too small, try use a greater value.')
            if (sigma > n_point/n_sigma/2) call wrndie(0, '<abpo_pre_opti>', &
               'MOLL is too large, try use a smaller value.')
            if (abpo_rbeta == -1.0) then
               call wrndie(5, '<abpo_pre_opti>', &
                  'Temperature is not specified for ABPO, set to 300 K.')
               abpo_temp = 300.0
            end if
            if (min_cnt <= ZERO) call wrndie(-2, '<abpo_pre_opti>', &
               'Invalid value for MINC.')
         end if
         if (prnlev >= 3) write(outu,40) &
            bcyc,          ecyc,          n_point, &
            abpo_b,        abpo_d,        moll, &
            rtube,         ftube,         abpo_temp, &
            min_cnt,       max_nblock,    block_steps, &   
            smooth_strength,  cvfrq,      q_loop 
40       format('  BCYCle =',I9,1X, '  ECYCle =',I9,1X, '    NPNT =',I9/ &
            '    BFCT =',F9.3,1X, '    PRED =',I9,1X, '    MOLL =',F9.3/ &
            '   RTUBe =',F9.3,1X, '   FTUBe =',F9.3,1X, '    TEMP =',F9.3/ &
            '  MINCnt =',F9.3,1X, ' MNBLock =',I9,1X, '   BSTEp =',I9/ &
            '  SMOOth =',F9.3,1X, '   CVFRq =',I9,1X, '   LOOP =',L9) 
         abpo_rbeta = KBOLTZ * abpo_temp
         n_span = ceiling(n_sigma * sigma)
         n_span2 = n_span * 2 + 1
         ! Allocate storage arrays
         call chmalloc('abpo.src', 'abpo_pre_opti', 'path', &
            n_point, n_cv, crl=path)
         call chmalloc('abpo.src', 'abpo_pre_opti', 'iD2_path', &
            n_point, n_cv, crl=iD2_path)
         if (.not. allocated(abpo_iD)) &
            call chmalloc('abpo.src', 'abpo_pre_opti', 'abpo_iD', &
               n_cv, n_cv, crl=abpo_iD)
         call chmalloc('abpo.src', 'abpo_pre_opti', 'abpo_iD2', &
            n_cv, n_cv, crl=abpo_iD2)
         call chmalloc('abpo.src', 'abpo_pre_opti', 'current_cv', &
            n_cv, crl=current_cv)
         call chmalloc('abpo.src', 'abpo_pre_opti', 'cv_force', &
            n_cv, crl=cv_force)
         call chmalloc('abpo.src', 'abpo_pre_opti', 'norm', &
            n_point, n_cv, crl=norm)
         call chmalloc('abpo.src', 'abpo_pre_opti', 'mean', &
            n_point, n_cv, crl=mean)
         call chmalloc('abpo.src', 'abpo_pre_opti', 'cnt', &
            n_point, intg=cnt)
         call chmalloc('abpo.src', 'abpo_pre_opti', 'h', &
            n_point, crl=h)
         if (old_mynod == 0) then   
            call chmalloc('abpo.src', 'abpo_pre_opti', 'tot_cnt', &
               n_point, intg=tot_cnt)
            call chmalloc('abpo.src', 'abpo_pre_opti', 'tot_h', &
               n_point, crl=tot_h)
         end if
         if (lmasternode) &
            call chmalloc('abpo.src', 'abpo_pre_opti', 'tot_mean', &
               n_point, n_cv, crl=tot_mean)
         call chmalloc('abpo.src', 'abpo_pre_opti', 'dh', &
            n_point, crl=dh)
         call chmalloc('abpo.src', 'abpo_pre_opti', 'delta', &
            n_span2, crl=delta)
         call chmalloc('abpo.src', 'abpo_pre_opti', 'd_delta', &
            n_span2, crl=d_delta)
         ! Load iD
         call load_iD() ! need to precede load_initial
         abpo_iD2 = abpo_iD
         call matsqrt(abpo_iD2, n_cv)
         if (q_rest) then
            ! Restart run, load path data from path.snap
            call load_snap_data()
         else
            ! Fresh start, load initial path
            call load_initial()
         end if
         abpo_step = 1
      end subroutine abpo_pre_opti

      subroutine apply_bias(x, y, z, dx, dy, dz)
      ! Calculate the ABPO force and add to dx, dy, dz
      ! Update ABPO histograms and abpo_step
         use number
         use stream
         real(kind=chm_real) x(*), y(*), z(*), dx(*), dy(*), dz(*)
         !
         real(kind=chm_real) :: dVb, dist, weight, dist_t
         integer :: i, j, k
         !
         ! Evaluate the cv values 
         do i = 1, n_cv
            call cv_array(i)%p%cv_eval(current_cv(i), x, y, z)
         end do
         ! Update lambda 
         call update_lambda(current_cv, current_lambda, dist, .true.)
         ! Apply wall potential in the cv space
         if (dist > rtube .or. &
            current_lambda == 1 .or. current_lambda == n_point) then
            cv_force = current_cv - path(current_lambda, :)
            call shift_diff(cv_force)
            if (current_lambda == 1 .or. current_lambda == n_point) &
               dist_t = abs(dot_product(cv_force, norm(current_lambda, :)))
            if (dist > rtube) then
               cv_force = matmul(abpo_iD, cv_force)
               cv_force = ftube * cv_force * (dist - rtube) / dist
            end if
         else
            cv_force = ZERO
         end if
         if (q_debug .and. prnlev >= 3) &
            write(outu, *) 'wall', dist, cv_force
         ! Apply biasing potential in the cv space
         if (.not. q_cbias) then
            dVb = abpo_b / (1 - abpo_b) * abpo_rbeta &
               * dh(current_lambda) / (h(current_lambda) + abpo_c)
         else
            dVb = abpo_rbeta &
               * dh(current_lambda) / (h(current_lambda) + abpo_c)
         end if
         do i = 1, n_cv
            cv_force(i) = cv_force(i) + dVb * norm(current_lambda, i)
         end do
         if (q_debug .and. prnlev >= 3) &
            write(outu, *) old_mynod, abpo_step, current_lambda, &
               current_cv, cv_force, h(current_lambda), dh(current_lambda)
         if (lmasternode .and. cvfrq > 0 .and. mod(abpo_step, cvfrq) == 0) &
            write(cvu, *) current_cv
         ! Apply the wall and biasing potential in the configuration space
         do i = 1, n_cv
            call cv_array(i)%p%cv_ener(x, y, z, dx, dy, dz, cv_force(i))
         end do
         ! Special treatment for the endpoints
         if (current_lambda /= 1 .and. current_lambda /= n_point &
            .or. dist_t < hdl .or. q_loop) then
            call shift_point(current_cv, mean(current_lambda, :))
            mean(current_lambda, :) = &
               (mean(current_lambda, :) * cnt(current_lambda) & 
               + current_cv) / (cnt(current_lambda) + 1) 
            cnt(current_lambda) = cnt(current_lambda) + 1
            if (.not. q_cbias) then
               do i = 1, n_span2
                  j = current_lambda - n_span - 1 + i
                  if (q_loop) then
                     j = mod(j+n_point-1, n_point) + 1
                     k = i
                  else if (j < 1) then
                     ! reflect at the left edge of the first cell
                     j = 1 - j 
                     k = n_span2 + 1 - i
                  else if (j > n_point) then
                     ! reflect at the right edge of the last cell
                     j = 2 * n_point + 1 - j 
                     k = n_span2 + 1 - i
                  else
                     k = i
                  end if
                  h(j) = h(j) + delta(k)
                  dh(j) = dh(j) + d_delta(k)
               end do
            else
               weight = h(current_lambda) / maxval(h)
               do i = 1, n_span2
                  j = current_lambda - n_span - 1 + i
                  if (q_loop) j = mod(j+n_point-1, n_point) + 1
                  if (j >= 1 .and. j <= n_point) then 
                     h(j) = h(j) + delta(i) * weight
                     dh(j) = dh(j) + d_delta(i) * weight
                  end if
               end do
            end if
         end if
      end subroutine apply_bias

      subroutine eval_D(x, y, z)
      ! Evaluate the instantaneous value of the D matrix
      !  and update abpo_iD
         use psf
         use number
         !
         real(kind=chm_real) :: x(*), y(*), z(*)
         !
         real(kind=chm_real), dimension(natom) :: dxi, dyi, dzi
         real(kind=chm_real), dimension(natom) :: dxj, dyj, dzj
         real(kind=chm_real) :: d
         integer :: i, j
         !
         do i = 1, n_cv
            dxi = ZERO
            dyi = ZERO
            dzi = ZERO
            call cv_array(i)%p%cv_ener(x, y, z, dxi, dyi, dzi, ONE)
            dxi = dxi / amass(1:natom)
            dyi = dyi / amass(1:natom)
            dzi = dzi / amass(1:natom)
            do j = i, n_cv
               dxj = ZERO
               dyj = ZERO
               dzj = ZERO
               call cv_array(j)%p%cv_ener(x, y, z, dxj, dyj, dzj, ONE)
               d = dot_product(dxi, dxj) + dot_product(dyi, dyj) &
                  + dot_product(dzi, dzj)
               abpo_iD(i, j) = abpo_iD(i, j) + d
               ! the lower half of D is left to post-processing
            end do
         end do
         D_step = D_step + 1
      end subroutine eval_D

      subroutine load_iD()
      !Reading the inversed D tensor from invd.dat'
         use stream
         use machio,only:vopen
         use mpi
         integer :: i, j, u
         logical :: qerr
         !
         u = 100
         if (old_mynod == 0) then
            call vopen(u, 'invd.dat', 'FORMATTED', 'READ', qerr, 0)
            if (qerr) call wrndie(-2, '<load_iD>', 'Error opening invd.dat.')
            do i = 1, n_cv
               read(u, *, err=51) (abpo_iD(i, j), j = 1, n_cv)
            end do
            call vclose(u, 'KEEP', qerr)
         end if
         call psnd8_world(abpo_iD, n_cv * n_cv)
         if (q_debug .and. prnlev >= 3) &
            write(outu, *) 'abpo_iD', abpo_iD
         return
         !
51       call wrndie(-2, '<load_iD>', &
            'Error or EOF on reading invd.dat. Check the format.')
         call vclose(u, 'KEEP', qerr)
         return
      end subroutine load_iD

      subroutine load_initial()
      ! Load the initial path into a temporary space
      !  and interpolate it to fill THE path
         use machio,only:vopen
         use stream
         !
         integer :: tmp_n, icv, ipoint, u
         real(kind=chm_real), dimension(max_n_point, n_cv) :: p
         character(len=80) :: filename
         logical :: qerr
         !
         u = 100
         if (bcyc /= 1) then
            write(filename, '("./cyc", I3.3, "/smean.dat")') bcyc-1
         else
            filename = 'init.dat'
         end if
         if (old_mynod == 0) then
            if (prnlev >= 3) write(outu, *) 'load_initial: Reading initial path from ', &
               trim(filename)
            call vopen(u, filename, 'FORMATTED', 'READ', qerr, 0)
            if (qerr) call wrndie(-2, '<load_initial>', &
               'Error opening '//trim(filename))
            do ipoint = 1, max_n_point + 1
               if (ipoint == max_n_point + 1) &
                  call wrndie(-1, '<load_initial>', &
                     'Maximun number of points exceded.')
               read(u, *, end=60, err=61) (p(ipoint, icv), icv = 1, n_cv)
               tmp_n = ipoint
            end do
            call vclose(u, 'KEEP', qerr)
60          if (prnlev >= 3) write(outu,350) tmp_n
350         format('load_initial: A total of', i3, ' points are read')
            call interpolate(p(1:tmp_n, :), tmp_n)
            call smooth(path, n_point, smooth_strength)
            call interpolate(path, n_point)
         end if
         call psnd8_world(path, n_point * n_cv)
         ! Interpolate the initial path
         call build_path()
         return
         !
61       call wrndie(-2, '<load_initial>', &
            'Error EOF on initial path reading. Check the format.')
         call vclose(u, 'KEEP', qerr)
      end subroutine load_initial

      subroutine shift_diff(d)
      ! Fold the difference vector d,
      !  assuming size(d) == n_cv
         real(kind=chm_real), dimension(n_cv) :: d
         !
         real(kind=chm_real) :: period
         integer :: i
         !
         do i = 1, n_cv
            period = cv_array(i)%p%period
            if (period > 0.0) &
               d(i) = d(i) - floor((d(i) + period / 2) / period) * period
         end do
      end subroutine shift_diff

      subroutine shift_point(p, refp)
      ! Find the image of p closest to refp,
      !  assuming size(p) == size(refp) == n_cv
         real(kind=chm_real), dimension(n_cv) :: p, refp
         !
         real(kind=chm_real), dimension(n_cv) :: d
         !
         d = p - refp
         call shift_diff(d)
         p = refp + d
      end subroutine shift_point

      subroutine shift_path(p, refp)
      ! Shift all points of p wrt refp
      !  assuming both p and refp are of dimension (n_point, n_cv)
         real(kind=chm_real), dimension(n_point, n_cv) :: p, refp
         !
         integer :: i
         !
         do i = 1, n_point
            call shift_point(p(i, :), refp(i, :))
         end do
      end subroutine shift_path

      subroutine fold(p, npnt)
      ! Fold path p (center each point wrt zero), 
      !  assuming size(p) == (npnt, n_cv)
         use number
         !
         integer :: npnt
         real(kind=chm_real), dimension(npnt, n_cv) :: p
         !
         real(kind=chm_real), dimension(n_cv) :: refp
         integer :: i
         !
         refp = ZERO
         do i = 1, npnt
            call shift_point(p(i, :), refp)
         end do
      end subroutine fold

      subroutine unfold(p, npnt)
      ! Unfold path p, 
      !  assuming size(p) == (npnt, n_cv)
         integer :: npnt
         real(kind=chm_real), dimension(npnt, n_cv) :: p
         !
         integer :: i
         !
         do i = 2, npnt
            call shift_point(p(i, :), p(i - 1, :))
         end do
      end subroutine unfold

      real(kind=chm_real) function distance(p0, p1)
      ! Calculate the shifted distance between p0 and p1,
      !  assuming size(p0) == size(p1) == n_cv
         real(kind=chm_real), dimension(n_cv) :: p0, p1
         !
         real(kind=chm_real), dimension(n_cv) :: d
         integer :: i
         !
         d = p0 - p1
         call shift_diff(d)
         distance = sqrt(dotD(d, d))
         return 
      end function distance

      real(kind=chm_real) function fast_distance(p, iD2_p, i)
      ! Calculate the shifted distance between p and path point i,
      !  use the pre-transformed path to reduce the number of 
      !  calls to matmul
         real(kind=chm_real), dimension(n_cv) :: p, iD2_p
         integer :: i
         !
         real(kind=chm_real), dimension(n_cv) :: d0, d
         real(kind=chm_real) :: dist
         !
         d0 = p - path(i, :)
         d = d0
         call shift_diff(d)
         if (any(d /= d0)) then
            p = path(i, :) + d
            iD2_p = matmul(abpo_iD2, p)
         end if
         d = iD2_p - iD2_path(i, :)
         fast_distance = sqrt(dot_product(d, d))
         return 
      end function fast_distance

      real function dotD(p0, p1)
      ! Calculate the dot product between p0 and p1 wrt the D tensor
      !  assuming size(p0) == size(p1) == n_cv
         real(kind=chm_real), dimension(1, n_cv) :: p0
         real(kind=chm_real), dimension(n_cv, 1) :: p1
         !
         real(kind=chm_real), dimension(n_cv, 1) :: t
         real(kind=chm_real), dimension(1, 1) :: rtn
         !
         t = matmul(abpo_iD, p1)
         rtn = matmul(p0, t)
         dotD = rtn(1, 1)
      end function dotD

      subroutine update_lambda(p, lambda, dist, global)
      ! Update lambda to lambda(p), dist to distance(p, path(lambda(p))),
      !  use global search if global is True
         real(kind=chm_real), dimension(n_cv) :: p
         integer :: lambda 
         real(kind=chm_real) :: dist
         logical :: global
         !
         real(kind=chm_real), dimension(n_cv) :: iD2_p
         integer :: i
         real(kind=chm_real) :: d, d_l, d_r
         !
         i = lambda
         iD2_p = matmul(abpo_iD2, p)
         if (global .or. i < 1 .or. i > n_point) then
            lambda = 1
            dist = fast_distance(p, iD2_p, 1)
            !dist = distance(p, path(1,:))
            do i = 2, n_point
               d = fast_distance(p, iD2_p, i)
               !d = distance(p, path(i,:))
               if (d < dist) then
                  lambda = i
                  dist = d
               end if
            end do
         else
            d = fast_distance(p, iD2_p, i)
            if (i > 1) d_l = fast_distance(p, iD2_p, i-1)
            if (i == 1 .or. d_l > d) then
               do
                  if (i == n_point) exit
                  d_r = fast_distance(p, iD2_p, i+1)
                  if (d_r > d) exit
                  i = i + 1
                  d = d_r
               end do
            else
               do
                  if (i == 1) exit
                  d_l = fast_distance(p, iD2_p, i-1)
                  if (d_l > d) exit
                  i = i - 1
                  d = d_l
               end do
            end if
            lambda = i
            dist = d
         end if
         return
      end subroutine update_lambda

      subroutine interpolate(oldpath, oldn)
      ! Interpolate oldpath to fill THE path,  
         use number
         !
         real(kind=chm_real), dimension(oldn, n_cv) :: oldpath
         integer :: oldn
         !
         real(kind=chm_real), dimension(oldn+1, n_cv) :: oldp
         real(kind=chm_real), dimension(oldn+1) :: l
         real(kind=chm_real) :: l_path, dl, d
         integer :: i, j, nstop0, nstop1
         !
         if (q_loop) then
            nstop0 = oldn + 1
            nstop1 = n_point + 1
         else
            nstop0 = oldn
            nstop1 = n_point
         end if
         ! Make a copy of oldpath
         do i = 1, nstop0
            oldp(i, :) = oldpath(mod((i-1), oldn)+1, :)
         end do
         call unfold(oldp(1:nstop0, :), nstop0)
         l(1) = ZERO
         do i = 2, nstop0
            l(i) = l(i-1) + &
               distance(oldp(i-1, :), oldp(i, :))
         end do
         dl = l(nstop0) / (nstop1 - 1)
         path(1, :) = oldp(1, :)
         i = 2
         j = 2
         do
            if (j == nstop1) exit
            if (l(i) > dl * (j - 1)) then
               d = l(i) - l(i-1)
               d = (l(i) - dl * (j - 1)) / d
               path(j, :) = d * oldp(i-1, :) + (1-d) * oldp(i, :)
               j = j + 1
            else
               i = i + 1
            end if
         end do
         if (.not. q_loop) path(n_point, :) = oldp(oldn, :)
         call fold(path, n_point)
      end subroutine interpolate

      subroutine smooth(p, npnt, strength, cnt, refp)
      ! Smooth path with a Gaussian filter
      ! strength gives the ratio between sigma and npnt
      ! assuming size(p(1)) == n_cv
         use number
         use memory
         !
         integer :: npnt
         real(kind=chm_real), dimension(npnt, n_cv) :: p
         real(kind=chm_real) :: strength
         integer, dimension(npnt), optional :: cnt
         real(kind=chm_real), dimension(npnt, n_cv), optional :: refp
         !
         integer, dimension(npnt) :: weight
         real(kind=chm_real), dimension(npnt, n_cv) :: p_cp
         real(kind=chm_real), dimension(2*npnt-1) :: w
         real(kind=chm_real) :: sigma, d, tw
         integer, parameter :: n_sigma = 2
         integer :: n_span, i, j, k
         !
         if (present(cnt)) then
            weight = cnt
         else
            weight = 1
         end if
         sigma = npnt * strength
         n_span = min((npnt - 1) / 2, ceiling(n_sigma*sigma))
         w(n_span + 1) = ONE 
         do i = 1, n_span
            d = exp(-(i / sigma)**2 / 2)
            w(n_span + 1 + i) = d
            w(n_span + 1 - i) = d
         end do
         if (present(refp)) then
            do i = 1, npnt
               if (weight(i) == 0) then
                  p(i, :) = refp(i, :)
                  weight(i) = 1 
               end if
            end do
         end if
         call unfold(p, npnt)
         p_cp = p
         p = ZERO
         if (.not. q_loop) then
            p(1, :) = p_cp(1, :)
            p(npnt, :) = p_cp(npnt, :)
            do i = 2, npnt - 1
               tw = ZERO
               do j = 1, n_span * 2 + 1
                  k = i - n_span - 1 + j
                  if (k >= 1 .and. k <= npnt) then 
                     p(i, :) = p(i, :) + p_cp(k, :) * w(j) * weight(k)
                     tw = tw + w(j) * weight(k)
                  end if
               end do
               p(i, :) = p(i, :) / tw
            end do
         else
            do i = 1, npnt
               tw = ZERO
               do j = 1, n_span * 2 + 1
                  k = i - n_span - 1 + j
                  k = mod(k+npnt-1, npnt) + 1
                  call shift_point(p_cp(k, :), p(i, :))
                  p(i, :) = p(i, :) * tw + p_cp(k, :) * w(j) * weight(k)
                  tw = tw + w(j) * weight(k)
                  p(i, :) = p(i, :) / tw
               end do
            end do
         end if
      end subroutine smooth

      subroutine build_path()
      ! Fill norm, delta and d_delta, set hdl
      !  Should be called each time path is loaded or updated
         use number
         use stream
         !
         integer :: i, nstop
         real(kind=chm_real) :: d, t_delta
         real(kind=chm_real) :: l_path, dl
         !
         if (q_loop) then
            nstop = n_point + 1
         else
            nstop = n_point
         end if
         l_path = ZERO
         do i = 2, nstop
            l_path = l_path + &
               distance(path(mod(i-2, n_point)+1, :), path(mod(i-1, n_point)+1, :))
         end do
         dl = l_path / (nstop - 1)
         hdl = dl / 2
         if (.not. q_loop) then
            norm(1, :) = path(2, :) - path(1, :)
            norm(n_point, :) = path(n_point, :) - path(n_point-1, :)
         else
            norm(1, :) = (path(2, :) - path(n_point, :)) / 2
            norm(n_point, :) = (path(1, :) - path(n_point-1, :)) / 2
            call shift_diff(norm(1, :))
            call shift_diff(norm(n_point, :))
         end if
         do i = 2, n_point-1
            norm(i, :) = (path(i+1, :) - path(i-1, :)) / 2
         end do
         ! norm is the gradient of the arc length parameter in the cv space
         ! norm = dot(iD, n) / sqrt(dotD(n, n))
         ! where n is any vector tangent to the path 
         do i = 1, n_point
            norm(i, :) = matmul(norm(i, :), abpo_iD) &
               / sqrt(dotD(norm(i, :), norm(i, :)))
            if (q_debug .and. prnlev >= 3) &
               write(outu, *) 'norm', i, norm(i, :)
         end do
         delta(n_span + 1) = ONE 
         t_delta = ONE
         ! d_delta is defined as such that dh is the derivative
         !  of h wrt the arc length (which is wrt iD)
         d_delta(n_span + 1) = ZERO
         do i = 1, n_span
            d = exp(-(i / sigma)**2 / 2)
            delta(n_span + 1 + i) = d
            delta(n_span + 1 - i) = d
            t_delta = t_delta + d * 2
            d = d * i / dl / sigma**2
            d_delta(n_span + 1 + i) = -d
            d_delta(n_span + 1 - i) = d
         end do
         do i = 1, n_span2
            delta(i) = delta(i) / t_delta
            d_delta(i) = d_delta(i) / t_delta
         end do
         if (q_debug .and. prnlev >= 3) write(outu, *) 'delta', delta
         if (q_debug .and. prnlev >= 3) write(outu, *) 'd_delta', d_delta
      end subroutine build_path

      subroutine save_path(path, npnt, prefix, dir)
      ! Save path to file
         use machio,only:vopen
         real(kind=chm_real), dimension(npnt, n_cv) :: path
         integer :: npnt
         character(len=*) :: prefix, dir
         !
         integer :: i, u
         character(len=80) :: filename
         logical :: qerr
         !
         u = 100
         filename = trim(dir)//'/'//trim(prefix)//'.dat'
         call vopen(u, filename, 'FORMATTED', 'WRITE', qerr, 0)
         do i = 1, npnt
            write(u, *) path(i, :)
         end do
         call vclose(u, 'KEEP', qerr)
      end subroutine save_path

      subroutine save_pmf(h, cnt, npnt, prefix, dir)
      ! Calculate and save the pmf to file
         use machio,only:vopen
         use number
         !
         real(kind=chm_real), dimension(npnt) :: h
         integer, dimension(npnt) :: cnt
         integer :: npnt
         character(len=*) :: prefix, dir
         !
         real(kind=chm_real), dimension(npnt) :: pmf
         integer :: i, u
         character(len=80) :: filename
         logical :: qerr
         !
         pmf = - abpo_rbeta * (log(cnt + abpo_c) + &
            abpo_b / (ONE - abpo_b) * log(h + abpo_c))
         pmf = pmf - minval(pmf)
         u = 100
         filename = trim(dir)//'/'//trim(prefix)//'.dat'
         call vopen(u, filename, 'FORMATTED', 'WRITE', qerr, 0)
         do i = 1, n_point
            write(u, *) pmf(i)
         end do
         call vclose(u, 'KEEP', qerr)
      end subroutine save_pmf

      subroutine save_h()
      ! Save abp infor to file
         use machio,only:vopen
         character(len=3) :: s
         integer :: i, u
         character(len=80) :: filename
         logical :: qerr
         !
         u = 100 
         write(s, '(I3.3)') iblock
         filename = trim(repdir)//'/h.'//s//'.dat'
         if (lmasternode) then
            call vopen(u, filename, 'FORMATTED', 'WRITE', qerr, 0)
            if (.not. q_cbias) then
               write(u, *) '#step, h, dh, dVb, real_h' 
               do i = 1, n_point
                  write(u, *) i, h(i)/abpo_step, dh(i)/abpo_step, &
                     abpo_b / (1 - abpo_b) * abpo_rbeta * &
                     dh(i) / (h(i) + abpo_c), &
                     cnt(i) * 1.0 / abpo_step
               end do
            else
               write(u, *) '#step, h, dh, dVb, real_h' 
               do i = 1, n_point
                  write(u, *) i, h(i)/abpo_step, dh(i)/abpo_step, &
                     abpo_rbeta * dh(i) / (h(i) + abpo_c), &
                     cnt(i) * 1.0 / abpo_step
               end do
            end if
            call vclose(u, 'KEEP', qerr)
         end if
      end subroutine save_h

      subroutine save_snap()
      ! Save the parameters and the current path data into 
      !  repdir/path.snap
         use machio,only:vopen
         use stream
         !
         character(len=80) :: filename, rowfmt1, rowfmt2, rowfmt3
         integer :: u, i, j, rep
         logical :: qerr
         !
         filename = trim(cycdir)//'/path.snap'
         u = 100
         write(rowfmt1,'(A,I4,A)') '(', n_point, '(1X,ES14.7))'
         write(rowfmt2,'(A,I4,A)') '(', n_point, '(1X,I9))'
         write(rowfmt3,'(A,I4,A)') '(', n_cv, '(1X,ES14.7))'
         if (prnlev >= 3) write(outu, '(a)') &
            "Saving the current ABPO status to "//trim(filename) 
         if (old_mynod == 0) then   
            call vopen(u, filename, 'FORMATTED', 'WRITE', qerr, 0)
            if (qerr) call wrndie(-2, '<save_snap>', 'Error opening '//trim(filename))
            write(u, '(A4,1X,I9)') 'MNBL', max_nblock
            write(u, '(A4,1X,I9)') 'PNBL', iblock
            write(u, '(A4,1X,I9)') 'BSTE', block_steps
            write(u, '(A4,1X,I9)') 'NPNT', n_point
            write(u, '(A4,1X,I9)') 'PRED', abpo_d
            write(u, '(A4,1X,I9)') 'CVFR', cvfrq
            write(u, '(A4,1X,L9)') 'LOOP', q_loop
            write(u, '(A4,1X,L9)') 'CBIA', q_cbias
            write(u, '(A4,1X,ES14.7)') 'MINC', min_cnt
            write(u, '(A4,1X,ES14.7)') 'SMOO', smooth_strength
            write(u, '(A4,1X,ES14.7)') 'MOLL', moll
            write(u, '(A4,1X,ES14.7)') 'BFCT', abpo_b
            write(u, '(A4,1X,ES14.7)') 'TEMP', abpo_temp
            write(u, '(A4,1X,ES14.7)') 'RTUB', rtube
            write(u, '(A4,1X,ES14.7)') 'FTUB', ftube
            write(u, *)
            write(u, '(A4)') 'PATH'
            do i = 1, n_point
               write(u, fmt=rowfmt3) (path(i,j), j=1,n_cv)
            end do
            call vclose(u, 'KEEP', qerr)
         end if
         do rep = 0, nensem-1
            call psync_world()
            if (lmasternode .and. whoiam == rep) then
               call vopen(u, filename, 'FORMATTED', 'APPEND', qerr, 0)
               if (qerr) call wrndie(-2, '<save_snap>', &
                  'Error opening '//trim(filename))
               write(u, *)
               write(u, '("rep",I3.3)') whoiam
               write(u, '(A4,1X,I9)') 'step', abpo_step
               write(u, '(A4)') 'h'
               write(u, fmt=rowfmt1) (h(i), i=1,n_point)
               write(u, '(A4)') 'dh'
               write(u, fmt=rowfmt1) (dh(i), i=1,n_point)
               write(u, '(A4)') 'cnt'
               write(u, fmt=rowfmt2) (cnt(i), i=1,n_point)
               write(u, '(A4)') 'mean'
               do i = 1, n_point
                  write(u, fmt=rowfmt3) (mean(i,j), j=1,n_cv)
               end do
               call vclose(u, 'KEEP', qerr)
            end if
         end do
         return
      end subroutine save_snap

      subroutine load_snap_pars()
      ! Load ABPO parameters from dir/path.snap
         use machio,only:vopen
         use stream
         !
         character(len=80) :: filename, line, rowfmt3
         character(len=4) :: key
         integer :: u, i, j, iolev_cp
         logical :: qerr
         !
         u = 100
         filename = trim(cycdir)//'/path.snap'
         write(rowfmt3,'(A,I4,A)') '(', n_cv, '(1X,ES14.7))'
         iolev_cp = iolev
         iolev = 1
         call vopen(u, filename, 'FORMATTED', 'READ', qerr, 0)
         iolev = iolev_cp
         if (qerr) call wrndie(-2, '<load_snap_pars>', 'Error opening '//trim(filename))
         if (prnlev >= 3) write(outu, '(a, i3)') &
            "Loading ABPO parameters from path.snap for cycle ", bcyc 
         do 
            read (u, '(A)') line
            read (line, '(A4)') key
            select case (key)
               case ('MNBL')
                  read (line(6:), '(I9)', err=200) max_nblock
               case ('PNBL')
                  read (line(6:), '(I9)', err=200) iblock
               case ('BSTE')
                  read (line(6:), '(I9)', err=200) block_steps
               case ('NPNT')
                  read (line(6:), '(I9)', err=200) n_point
               case ('PRED')
                  read (line(6:), '(I9)', err=200) abpo_d
               case ('CVFR')
                  read (line(6:), '(I9)', err=200) cvfrq
               case ('LOOP')
                  read (line(6:), '(L9)', err=200) q_loop
               case ('CBIA')
                  read (line(6:), '(L9)', err=200) q_cbias
               case ('MINC')
                  read (line(6:), '(ES14.7)', err=200) min_cnt
               case ('SMOO')
                  read (line(6:), '(ES14.7)', err=200) smooth_strength
               case ('MOLL')
                  read (line(6:), '(ES14.7)', err=200) moll
               case ('BFCT')
                  read (line(6:), '(ES14.7)', err=200) abpo_b
               case ('TEMP')
                  read (line(6:), '(ES14.7)', err=200) abpo_temp
               case ('RTUB')
                  read (line(6:), '(ES14.7)', err=200) rtube
               case ('FTUB')
                  read (line(6:), '(ES14.7)', err=200) ftube
               case ('')
                  exit
               case default      
                  goto 200
            end select
         end do
         sigma = n_point * moll
         abpo_c = abpo_d * 1.0 / n_point
         call vclose(u, 'KEEP', qerr)
         return
200      call wrndie(-2, '<load_snap_pars>', &
            trim(filename)//' is in wrong format')
         call vclose(u, 'KEEP', qerr)
      end subroutine load_snap_pars

      subroutine load_snap_data()
      ! Load ABPO path data from dir/path.snap
         use machio,only:vopen
         use stream
         !
         character(len=80) :: filename, line, rowfmt1, rowfmt2, rowfmt3
         character(len=6) :: repid
         integer :: u, i, j, iolev_cp
         logical :: qerr
         !
         u = 100
         filename = trim(cycdir)//'/path.snap'
         write(rowfmt1,'(A,I4,A)') '(', n_point, '(1X,ES14.7))'
         write(rowfmt2,'(A,I4,A)') '(', n_point, '(1X,I9))'
         write(rowfmt3,'(A,I4,A)') '(', n_cv, '(1X,ES14.7))'
         iolev_cp = iolev
         iolev = 1
         call vopen(u, filename, 'FORMATTED', 'READ', qerr, 0)
         iolev = iolev_cp
         if (qerr) call wrndie(-2, '<load_snap_data>', 'Error opening '//trim(filename))
         if (prnlev >= 3) write(outu, '(a, i3)') &
            "Loading ABPO path data from path.snap for cycle ", bcyc 
         do 
            read (u, '(A)') line
            if (line(:4) == 'PATH') exit
         end do
         do i = 1, n_point
            read(u, fmt=rowfmt3, err=201) (path(i,j), j=1,n_cv)
         end do
         write(repid, '("rep",I3.3)') whoiam
         do 
            read (u, '(A)') line
            if (line(:6) == repid) exit
         end do
         read(u, '(A5,1X,I9)') abpo_step
         read(u, '(A)') line ! load h
         read(u, fmt=rowfmt1, err=201) (h(i), i=1,n_point)
         read(u, '(A)') line ! load dh
         read(u, fmt=rowfmt1, err=201) (dh(i), i=1,n_point)
         read(u, '(A)') line ! load cnt
         read(u, fmt=rowfmt2, err=201) (cnt(i), i=1,n_point)
         read(u, '(A)') line ! load mean
         do i = 1, n_point
            read(u, fmt=rowfmt3, err=201) (mean(i,j), j=1,n_cv)
         end do
         call vclose(u, 'KEEP', qerr)
         call build_path()
         return
201      call wrndie(-2, '<load_snap_data>', &
            trim(filename)//' is in wrong format')
         call vclose(u, 'KEEP', qerr)
      end subroutine load_snap_data

      subroutine replace(comlyn, comlen, key, val)
      ! Replace the value of key in comlyn with val 
         use string
         !
         character(len=*) :: comlyn, key
         integer :: comlen, val
         !
         character(len=80) :: tmps
         integer :: tmpi
         !
         tmpi = gtrmi(comlyn, comlen, key, -1)
         write(tmps, '(" ", A4, " ", I10)') key, val
         comlyn(comlen+1:) = trim(tmps)
         comlen = len(trim(comlyn))
      end subroutine replace

      subroutine rmv(comlyn, comlen, key, q)
      ! Remove key from comlyn
         use string
         !
         character(len=*) :: comlyn, key
         integer :: comlen
         logical :: q
         !
         q = checque(comlyn, key)
         comlen = len(trim(comlyn))
      end subroutine rmv

      subroutine rmvf(comlyn, comlen, key, q)
      ! Remove key and real from comlyn
         use string
         !
         character(len=*) :: comlyn, key
         integer :: comlen, i
         logical :: q
         !
         real :: t
         !
         i = index(comlyn, key)
         if (i /= 0) then
            q = .true.
            t = gtrmf(comlyn, comlen, key, 0.)
         else
            q = .false.
         endif
      end subroutine rmvf

      subroutine rmvni(comlyn, comlen, key, n, q)
      ! Remove key and n following integers from comlyn
         use string
         !
         character(len=*) :: comlyn, key
         integer :: comlen, n
         logical :: q
         !
         integer :: i
         integer, dimension(n) :: is, dis
         !
         i = index(comlyn, key)
         if (i /= 0) then
            q = .true.
            call gtrmim(n, comlyn, comlen, key, dis, is, q)
         else
            q = .false.
         end if
      end subroutine rmvni

      subroutine append(comlyn, comlen, key)
      ! Append key to comlyn
         use string
         !
         character(len=*) :: comlyn, key
         integer :: comlen
         !
         comlyn(comlen+2:) = trim(key)
         comlen = len(trim(comlyn))
      end subroutine append

      subroutine newdir(dir, q)
      ! Make a new directory 
         character(len=*) :: dir
         logical :: q
         !
         logical :: ex
         !
         if (q) then
            inquire(file=dir, exist=ex)
            if (ex) call wrndie(-2, '<abpo_cntrl>', &
                     'Directory '//trim(dir)//' already exists')
            call system('mkdir '//dir)
         end if
         call psync_world()
      end subroutine newdir

      subroutine gauss_inv(a, n)       
      ! Invert matrix a of dimension n*n by Gauss method
         integer :: n
         real(kind=chm_real), dimension(n, n) :: a
         ! 
         real(kind=chm_real), dimension(n, n) :: b
         real(kind=chm_real), dimension(n) :: temp
         real(kind=chm_real) :: c, d
         integer :: i, j, k, m, imax(1), ipvt(n)
         ! 
         b = a
         ipvt = (/ (i, i = 1, n) /)
         do k = 1, n
            imax = maxloc(abs(b(k:n, k)))
            m = k - 1 + imax(1)
            if (m /= k) then
               ipvt( (/m,k/) ) = ipvt( (/k,m/) )
               b((/m,k/), :) = b((/k,m/), :)
            end if
            d = 1 / b(k,k)

            temp = b(:, k)
            do j = 1, n
               c = b(k, j) * d
               b(:, j) = b(:, j) - temp * c
               b(k, j) = c
            end do
            b(:, k) = temp * (-d)
            b(k, k) = d
         end do
         a(:, ipvt) = b
      end subroutine gauss_inv

      subroutine matsqrt(a, n)       
      ! Calculate the square root of an n*n matrix
         integer :: n
         real(kind=chm_real), dimension(n, n) :: a
         !
         real(kind=chm_real), dimension(n, n) :: v
         real(kind=chm_real), dimension(n) :: d
         integer :: i
         ! 
         call jacobi(a, n, d, v)
         d = sqrt(d)
         do i = 1, n
            a(:, i) = d(i) * v(:, i)
         end do
         a = transpose(a)
      end subroutine matsqrt

      subroutine jacobi(a, n, d, v)
      ! Diagonalize an n*n symmetric matrix a 
      ! Note that a is changed 
         use stream
         !
         integer :: n
         real(kind=chm_real), dimension(n, n) ::  a, v
         real(kind=chm_real), dimension(n) :: d
         !
         integer :: i, j, ip, iq
         real(kind=chm_real) :: sm, g, h, c, s, t
         real(kind=chm_real) :: tau, theta, tresh
         real(kind=chm_real), dimension(n) :: b, z
         ! set v = I
         do ip = 1, n
            do iq = 1, n
               v(ip,iq) = 0.
            end do
            v(ip,ip)= 1.
         end do
         ! set b = d = diag(a)
         ! set z = 0
         do ip = 1, n
           b(ip) = a(ip,ip)
           d(ip) = b(ip)
           z(ip) = 0.
         end do
         ! enter iteration
         do i = 1, 50
            ! sm = sum of abs of off-diagonal of a
            sm = 0.
            do ip = 1, n-1
               do iq = ip+1, n
                  sm = sm + abs(a(ip,iq))
               end do
            end do
            if (sm .eq. 0.) return
            ! set threshold
            if (i .lt. 4) then
               tresh = 0.2 * sm / n**2
            else
               tresh = 0.
            end if
            ! scan each row
            do ip = 1, n-1
               ! scan each off-diagonal element
               do iq = ip+1, n
                  g = 100. * abs(a(ip,iq))
                  if ((i .gt. 4) .and. &
                     (abs(d(ip))+g .eq. abs(d(ip))) .and. &
                     (abs(d(iq))+g .eq. abs(d(iq)))) then
                     ! if a(p,p) >> a(p,q), set a(p,q) to 0
                     a(ip,iq) = 0.
                  else if (abs(a(ip,iq)) .gt. tresh) then
                     ! if a(p,q) > tresh, do the rotation
                     h = d(iq) - d(ip)
                     ! calculate t = tan(alpha)
                     if (abs(h)+g .eq. abs(h)) then
                        ! treat the special case where h >> a(p,q)
                        t = a(ip,iq) / h
                     else
                        ! treat the normal case
                        theta = 0.5 * h / a(ip,iq)
                        t = abs(theta) + sqrt(1.+theta**2)
                        if (theta .gt. 0.) t = -t
                     endif
                     ! calculate c = cos(alpha), s = sin(alpha)  
                     c = 1. / sqrt(1+t**2)
                     s = t * c
                     tau = s / (1.+c)
                     h = t * a(ip,iq)
                     z(ip) = z(ip) - h
                     z(iq) = z(iq) + h
                     d(ip) = d(ip) - h
                     d(iq) = d(iq) + h
                     ! rotate the upper half of a
                     a(ip,iq) = 0.
                     do j = 1, ip-1
                        g = a(j,ip)
                        h = a(j,iq)
                        a(j,ip) = g - s*(h+g*tau)
                        a(j,iq) = h + s*(g-h*tau)
                     end do
                     do j = ip+1, iq-1
                        g = a(ip,j)
                        h = a(j,iq)
                        a(ip,j) = g - s*(h+g*tau)
                        a(j,iq) = h + s*(g-h*tau)
                     end do
                     do j = iq+1, n
                        g = a(ip,j)
                        h = a(iq,j)
                        a(ip,j) = g - s*(h+g*tau)
                        a(iq,j) = h + s*(g-h*tau)
                     end do
                     ! update v
                     do j = 1, n
                        g = v(j,ip)
                        h = v(j,iq)
                        v(j,ip) = g - s*(h+g*tau)
                        v(j,iq) = h + s*(g-h*tau)
                     end do
                  endif
               end do
            end do
            ! update b, d, reset z
            do ip = 1, n
               b(ip) = b(ip) + z(ip)
               d(ip) = b(ip)
               z(ip) = 0.
            end do
         end do
         if (prnlev >= 3) write(outu, *) 'too many iterations in jacobi' 
         return
      end subroutine jacobi

      subroutine asum_comm(a, w, comm, length)
         !-----------------------------------------------------------------------
         ! this routine performs on node 0 broadcast to all other nodes
         ! and receive from node 0 on all other nodes.
         ! usually called after read on node 0. for real(chm_real) precision arrays.
         !
         use chm_kinds
         !
         use dimens_fcm
         use parallel
#if KEY_CMPI==0
         use mpi      
#endif
         !
         integer, intent(in) :: length, comm
         real(chm_real), intent(inout), dimension(length) :: a, w
         integer :: status, nod0
         ! 
         nod0 = 0
         call mpi_reduce(a, w, length, mpi_real8, mpi_sum, &
               nod0, comm, status)
         return
      end subroutine asum_comm 

      subroutine isum_comm(a, w, comm, length)
         !-----------------------------------------------------------------------
         ! this routine performs on node 0 broadcast to all other nodes
         ! and receive from node 0 on all other nodes.
         ! usually called after read on node 0. for real(chm_real) precision arrays.
         !
         use chm_kinds
         !
         use dimens_fcm
         use parallel
#if KEY_CMPI==0
         use mpi      
#endif
         !
         integer, intent(in) :: length, comm
         integer, intent(inout), dimension(length) :: a, w
         integer :: status, nod0
         ! 
         nod0 = 0
         call mpi_reduce(a, w, length, mpi_integer4, mpi_sum, &
               nod0, comm, status)
         return
      end subroutine isum_comm 

#endif /* ABPO*/
end module abpo


