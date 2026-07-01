module machutil


  !---------------------------------------------------------------------
  use chm_kinds
  use stream
  use number
  implicit none
  !RCZ 92/09/17 - is there any difference between ELATIM and ELATM ???
  !               cannot we keep just one of them and get rid of TT ???
  real(chm_real) CPUTIM,TT,CPUT8
  real(chm_real) ELATM
  !  INTEGER NPFLTS,IODIR,IOBUF
  !
#if KEY_GNU==1
  real(chm_real) ZEROT,timval
  !      EXTERNAL XSECNDS
  integer(chm_int8) :: count,rate,maxcount
  REAL CPUREF
  REAL CPUT
  REAL TREF !, TARRAY(2), ETIME
#elif KEY_UNIX==1 || KEY_OSX==1
  REAL CPUREF
  REAL CPUT,ZEROT
  REAL TREF 
#endif 
  public eclock

contains

  !-----------------------------------------------------------------------
  !                            Initialize_timers
  !-----------------------------------------------------------------------
  subroutine Initialize_Timers
    use startup

    !     INITIALIZE ALL TIMERS

#if KEY_GNU==1
    call cpu_time(cpuref)
    zerot = zero
    call system_clock(count,rate,maxcount)
    timval=count
    if(rate.ne.0)zerot=timval/rate-zerot
#elif KEY_UNIX==1 || KEY_OSX==1
    call cpu_time(cpuref)
    zerot = 0.0
    zerot = secnds(zerot)
#else /**/
#error  'Unrecognized machine type in Initialize_Timers'
#endif 
    call timrb
    return
  end subroutine initialize_timers


  !-----------------------------------------------------------------------
  !                            Machdep Init
  !-----------------------------------------------------------------------
  subroutine machdep_init()
    use machdep
    use param_store, only: set_param

    implicit none

    integer :: lcpu
    call getcpu(ncpu,lcpu)
    oldcpu = ncpu
    call set_param('NCPU',ncpu)
    qfastnb = .false.
    call set_param('FNBL',0)
    qflush = .true.
    call set_param('FLUSH',1)
    nbfact = onept5
    call set_param('NBFACT',onept5)
    return
  end subroutine machdep_init
  
! BIOVIA Code Start : Fix for Windows
#if KEY_WIN32 == 1 || KEY_WIN64 == 1
  INTEGER FUNCTION DCLOCK
    DCLOCK=0
    RETURN
  END FUNCTION DCLOCK
#endif
! BIOVIA Code End

  SUBROUTINE TIMRB
#if KEY_GNU==1
100 CONTINUE
    !      TREF = ETIME(TARRAY)
    call cpu_time(tref)
#elif KEY_UNIX==1 || KEY_OSX==1

100 CONTINUE

    call cpu_time(tref)
#else /**/
#error  'Unrecognized machine type in TIMRB'
#endif 
    RETURN
  end SUBROUTINE TIMRB

  !--e-------------------------------------------------------------------
  subroutine TIMRE
    !-----------------------------------------------------------------------
    !     TIMRE obtains process statistics and subtracts the beginning
    !     of interval statistics recorded by TIMRB.  The incremental values
    !     are written to Fortran unit OUTU.
    !
    use param_store, only: set_param
    implicit none
#if KEY_GNU==1
    !
    !      CPUT = ETIME(TARRAY) - TREF
    call cpu_time(cput)
    cput=cput-tref
    !      ELATM = XSECNDS(ZEROT)
    call system_clock(count,rate,maxcount)
    timval=count
    elatm=zero
    if(rate.ne.0)elatm=timval/rate-zerot
#elif KEY_UNIX==1 || KEY_OSX==1
    !
    !      CPUT = ETIME(TARRAY) - TREF
    call cpu_time(cput)
    cput=cput-tref
    ELATM = SECNDS(ZEROT)
    !RCZ 91/12/04
    !     to make outputs between different machines easier comparable;
    !     probably whole ENTRY TIMRE should we rewritten with just
    !     single write statement and single format
#else /**/
#error  'Unrecognized machine type in TIMRE'
#endif 
    CPUT8=CPUT
    call set_param('CPUTIME',CPUT8)
    call set_param('ELATIME',ELATM)
    IF(PRNLEV.GE.2) WRITE(OUTU,99) CPUT,ELATM
99  FORMAT(' CPU TIME=',F11.2,' ELAPSED TIME=',F11.2)
    !
    RETURN
  end subroutine TIMRE
  !
  SUBROUTINE WRTTIM(STRING)
    !
    !     Write out timing information.
    !
    !
    CHARACTER(len=*) STRING
    !
    IF(PRNLEV.GE.2) WRITE(OUTU,'(2A)') ' WRTTIM> ',STRING
    CALL TIMRE
    CALL TIMRB
    !
    RETURN
  END SUBROUTINE WRTTIM
  !---------------------------------------------------------------------
  subroutine JOBDAT(ELATIM,CPUTIM,IOBUF,IODIR,NPFLTS)
    !-----------------------------------------------------------------------
    real(chm_real) :: elatim,cputim
    integer iobuf,iodir,npflts
#if KEY_GNU==1
    call system_clock(count,rate,maxcount)
    timval=count
    elatim=zero
    if(rate.ne.0)elatim=timval/rate-zerot
    !      ELATIM = XSECNDS(ZEROT)
    !      CPUTIM = ETIME(TARRAY) - CPUREF
    CALL CPU_TIME(CPUTIM)
    CPUTIM = CPUTIM - CPUREF
    IOBUF = 0
    IODIR = 0
    NPFLTS = 0
#elif KEY_UNIX==1 || KEY_OSX==1
    ELATIM = SECNDS(ZEROT)
    !      CPUTIM = ETIME(TARRAY) - CPUREF
    call cpu_time(cputim)
    CPUTIM = cputim - CPUREF
    IOBUF = 0
    IODIR = 0
    NPFLTS = 0
#else /**/
#error  'Unrecognized machine type in JOBDAT'
#endif 
    IOBUF = 0
    IODIR = 0
    NPFLTS = 0
    RETURN
  end subroutine JOBDAT

  FUNCTION ECLOCK() result(eclck)
    !-----------------------------------------------------------------------
    !
    ! Timing routine which ususally returns elapsed time
    !
    real(chm_real) :: eclck
    real(chm_real) TIM   !old::: ,DCLOCK
    integer(chm_int8) count,rate,maxcount
    !
    ! call standard fortran here...
    ! old:::      TIM=DCLOCK()
    ! maybe we should check for maxcount
    ! maxcount=0 => no clock available on this machine
    ! use integer*8 for better precision on some compilers
    !
    call system_clock(count,rate,maxcount)
    tim=count
    if(rate.ne.0) then
       ECLCK=TIM/rate
    else
       eclck=zero
    endif
    !
    RETURN
  END FUNCTION ECLOCK

  SUBROUTINE DAYTIM(month, day, year, hour, minute, second)
    !----------------------------------------------------------------------
    !     THIS ROUTINE RETURNS THE DAY AND TIME OF DAY
    !
    use chm_kinds
    implicit none
    integer month, day, year, hour, minute, second
#if KEY_GNU==1 || KEY_OSX==1
    character(len=8) date
    character(len=10) time

    call date_and_time(date, time)
    read(date, '(2x, i2, i2, i2)') year, month, day
    read(time, '(i2, i2, i2)') hour, minute, second
#elif KEY_UNIX==1
    CHARACTER(len=8) A
    ! --- UNIX IDATE(dd,mm,yy)
    CALL IDATE(month, day, year)
    ! --- UNIX TIME(A) : A=hh:mm:ss
    CALL TIME(A)
    READ (A,1) hour, minute, second
1   FORMAT(I2,1X,I2,1X,I2)
#else /**/
#error  'Unrecognized machine type in DAYTIM'
#endif 
    RETURN
  END SUBROUTINE DAYTIM

  SUBROUTINE DIE
    !----------------------------------------------------------------------
    !     Terminates execution with a traceback. We use a VMS call to do
    !     this because this is faster than generating a error numerically.
    !     The numeric error usually has more subroutine levels to trace down
    !     subroutine names. The error code for termination is
    !     44, SYSTEM-F-ABORT
    !
    !     Author: Robert Bruccoleri
    !
    ! ---------- compiler directive :: do not move :: upper case external
    use chm_kinds
    use stream
#if KEY_MPI==1
    use mpi     
#endif
    use cstuff, only: stack_trace

    implicit none

#if KEY_MPI==1
    INTEGER STATUS    
#endif
    CHARACTER(len=40) TRBACK
    INTEGER      IDUM
    !
! BIOVIA Code Start
#if KEY_LICENSE==1
    ! release charmm license
    call release_licenses()
#endif
! BIOVIA Code End
    !
    !CC##IF NIH ! make it available as per the 1999 CHARMM meeting
#if KEY_NOSKULL==0
    IF(WRNLEV.GE.2) THEN
       WRITE(OUTU,9910)
9910   FORMAT(/, &
            /'                                                  ', &
            /'                            /---------\           ', &
            /'                           /           \          ', &
            /'                          /             \         ', &
            /'                         /               \        ', &
            /'                         !  XXXX   XXXX  !        ', &
            /'                         !  XXXX   XXXX  !        ', &
            /'                         !  XXX     XXX  !        ', &
            /'                         !       X       !        ')
       WRITE(OUTU,9911)
9911   FORMAT( &
            '                          --\   XXX   /--         ', &
            /'                           ! !  XXX  ! !          ', &
            /'                           ! !       ! !          ', &
            /'                           ! I I I I I !          ', &
            /'                           !  I I I I  !          ', &
            /'                            \         /           ', &
            /'                             --     --            ', &
            /'                               \---/              ')
       WRITE(OUTU,9912)
9912   FORMAT('                        XXX             XXX       ', &
            /'                       XXXX             XXXX      ', &
            /'                       XXXXX           XXXXX      ', &
            /'                          XXX         XXX         ', &
            /'                            XXX     XXX           ', &
            /'                               XXXXX              ')
       WRITE(OUTU,9913)
9913   FORMAT('                              XXX XXX             ', &
            /'                            XXX     XXX           ', &
            /'                          XXX         XXX         ', &
            /'                       XXXXX           XXXXX      ', &
            /'                       XXXX             XXXX      ', &
            /'                        XXX             XXX       ', &
            /'                                                  ', &
            /'                                                  ')
    ENDIF
#endif 
    !CC##ENDIF ! NIH
    !
#if KEY_NIH==1
    IF(WRNLEV.GE.2) WRITE(OUTU,201)

#ifdef __hpux
    CALL U_STACK_TRACE()
#else /* __hpux */
    call stack_trace()
#endif /* __hpux */
    
#else /* KEY_NIH */
    IF(WRNLEV.GE.2) WRITE(OUTU,202)

    CALL PRTSTATS('DIE')
#endif /* KEY_NIH */
    
    CALL EXIT(1)
    TRBACK = 'YES'
#if KEY_OSX==1 /*OSX*/
    IF (TRBACK.EQ.' ') THEN
    IF (WRNLEV.GE.2) WRITE(OUTU,202)
    ELSE
    IF (WRNLEV.GE.2) WRITE(OUTU,201)
    ENDIF
#else /* (OSX)*/
    IF (WRNLEV.GE.2) WRITE(OUTU,202)
#endif /* (OSX)*/
    ! print out termination status
    CALL PRTSTATS('DIE')
#if KEY_MPI==1
    CALL MPI_ABORT(MPI_COMM_WORLD,MPI_ERR_ARG,STATUS)
    CALL EXIT(-1)
#else /**/
    STOP
#endif /*  MPI */
    RETURN
201 FORMAT(' Execution terminated due to the detection of a', &
         ' fatal error.'/' A traceback will now be generated.')
202 FORMAT(' Execution terminated due to the detection of a', &
         ' fatal error.')
  END SUBROUTINE DIE

  SUBROUTINE CSYSTEM(COMLYN,COMLEN)
    use stream
    use param_store, only: set_param
    use cstuff, only: fsystem
    
! BIOVIA Code Start
#if KEY_IMPI==1
    use ifport
#endif
! BIOVIA Code End

    implicit none

    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    INTEGER STATUS

    !AvdV moved the "Invoked:" line out of the cstuff.c to here,
    !     so we can skip it if prnlev<=0
    !
    !     Some parallel and non-UNIX machines don't support SYSTem comand
    !
    !
    !     All the rest comes here
    !
    !      STATUS = FSYSTEM(COMLYN,COMLEN)
    !  Only pass characters 2:comlen-1 assuming command protected from CHARMM
    !  interpreter by beginning and ending double quotes.
    !AvdV added prnlev statement

    IF(PRNLEV.GE.0) WRITE(OUTU,'("Invoking: ",A)') COMLYN(3:COMLEN-1)

    ! system command might have some side effect output to stdout
    ! so flush stdout before calling csystem
    flush(outu)
! BIOVIA Code Start
#if KEY_IMPI==1
    STATUS = SYSTEM(COMLYN(3:COMLEN-1))
#else
! BIOVIA Code End
    STATUS = FSYSTEM(COMLYN(3:COMLEN-1), COMLEN-3)
! BIOVIA Code Start
#endif
! BIOVIA Code End
    CALL set_param('SYSSTAT', STATUS)
    COMLEN = 0
  END SUBROUTINE CSYSTEM

  SUBROUTINE GETCPU(NTHREAD,NTHREAD_MAX)
    !-----------------------------------------------------------------------
    !     For compatibility with all machines.
    !     Should replace with a routine that returns
    !     Number of available cpu's in nthread.
    !     NTHREAD: number of processors available to the process.
    !     NTHREAD_MAX: hardware number of processors.
    !
    INTEGER NTHREAD,NTHREAD_MAX
    !
    NTHREAD=1
    NTHREAD_MAX=1
    RETURN
  END SUBROUTINE GETCPU

  ! Dummy setcpu. If machine has capability of setting
  ! number of processors at run time, substitute routine
  ! for this one. Convex version is in convex.c.
  !
  INTEGER FUNCTION SETCPU(NCPU)
    !-----------------------------------------------------------------------
    INTEGER NCPU
    SETCPU = NCPU
    RETURN
  END FUNCTION SETCPU

  INTEGER FUNCTION LOCDIF(A,B,NMATCH)
    !----------------------------------------------------------------------
    !     This routine returns the difference inlocation
    !     words between the arguments A and B.
    !     NMATCH gives the number of address locations per HEAP word
    !
    !     By B.R. Brooks
    !
    use chm_kinds
    implicit none
    real(chm_real) :: A(*),B(*)
    INTEGER NMATCH,LOC
    !
#if KEY_UNIX==1 || KEY_GNU==1 || KEY_OSX==1
    LOCDIF=LOC(A)-LOC(B)
    NMATCH=4
#else /**/
#error  'Unrecognized machine type in LOCDIF'
#endif 
    RETURN
  END FUNCTION LOCDIF

#if KEY_GNU==1 || KEY_OSX==1 /*itime*/
  subroutine itime(iarray)
    !-----------------------------------------------------------------------
    ! returns the current time, i.e., hour, minute and second in IARRAY
    !-----------------------------------------------------------------------
    implicit none
    integer, intent(out) :: iarray(3)

    integer i
    character(len=10) time

    call date_and_time(time=time)
    read(time, '(i2, i2, i2)') (iarray(i), i=1, 3)
  end subroutine itime

  subroutine idate(month, day, year)
    !-----------------------------------------------------------------------
    ! returns the current month, day and year
    !-----------------------------------------------------------------------
    implicit none
    integer, intent(out) :: month, day, year

    character(len=8) date
    call date_and_time(date)
    read(date, '(i4, i2, i2)') year, month, day
  end subroutine idate
#endif /* (itime)  GNU OSX*/

#if KEY_GNU==1 || KEY_OSX==1 /*secnds*/
  REAL FUNCTION SECNDS(TREF)
    REAL TREF
    !
    !     This is a replacement for the Vax system routine secnds for
    !     unix machines.  It uses a unix system call to get the time and
    !     then converts to the form returned by the vax secnds.
    !           J. Kottalam 9/3/90
    !
    INTEGER TCUR
    INTEGER IARRAY(3)
    CALL ITIME(IARRAY)
    TCUR = ( IARRAY(1)*60+IARRAY(2) )*60 + IARRAY(3)
    SECNDS = TCUR - TREF
    RETURN
  END FUNCTION SECNDS
#endif /* (secnds)*/

#if KEY_GNU==1 || KEY_OSX==1 /*wcpu*/
  !-----------------------------------------------------------------------
  ! This is a reminder:
  !   The last subroutine in EACH source file written in this dialect
  !   (pre-flecs) MUST be a routine that gets included for ALL the
  !   platforms. Otherwise we get 'extra' comments at the end of the '.f'
  !   file and Fortran assumes it is a main program.
  !           Happy pre-flecs'ing!
  !-----------------------------------------------------------------------
  SUBROUTINE WCPU(IRWAIT)
    !-----------------------------------------------------------------------
    !     This routine will wait for a fixed number of seconds.
    !
    use chm_kinds
    use stream
#if KEY_PARALLEL==1
    use dimens_fcm
    use parallel
#endif 
    real(chm_real) A
    INTEGER IRWAIT
    !
    !
    INTEGER I, J, N
    !
#if KEY_PARALLEL==1
#endif 
    RETURN
  END SUBROUTINE WCPU
#endif /* (wcpu)*/
  subroutine machutil_dummy
    return
  end subroutine machutil_dummy

end module machutil

