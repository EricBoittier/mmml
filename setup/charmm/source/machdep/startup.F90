module startup
  use chm_kinds
  implicit none

contains

  !----------------------------------------------------------------------
  !     CHARMM STARTUP MACHINE DEPENDENT CODE
  !----------------------------------------------------------------------

  SUBROUTINE Startup_machine_dependent_code

    use dimens_fcm
    use number
    use machdep

    integer lcpu

    ! Startup for all parallel code.
    call parstrt

!MFC-- This needs to be called at the ensemble command
!MFC-- will remove this code if it works
#if KEY_ENSEMBLE==1
    call ensini     
#endif
    return
  end subroutine startup_machine_dependent_code

SUBROUTINE ARGUMT(WANTQUA)
  !----------------------------------------------------------------------
  ! Process command line parameters to set CHARMM parameters
  !----------------------------------------------------------------------
  !
  !   Process command line arguments
  !
  !   Command line arguments can be used for:
  !     - to set CHARMM parameters;
  !     - to set CHM_DATA environment variable (CHARMM data directory);
  !     - to change the current (working) directory;
  !
  !   Syntax:
  !     -h or -help  give help message and quit charmm
  !
  !     N:value or N=value, where N - one-character parameter name,
  !            - parameter N will get the value 'value';
  !
  !     -i <file>   or -input <file>
  !              specify input to come from a file rather than standard input
  !     -o <file>   or -output <file>
  !              specify input to come from a file rather than standard input
  !     -chsize N
  !              (charmm size) specify number of atoms to allocate arrays
  !     -prevrandom
  !     -prevclcg
  !     -set name value - set environment variable
  !     For PVM version only:
  !     -n <numproc> has been parsed by pvmfbegin. We must ignore it here.
  !        -- Stephen Fleischman 5/11/94.
  !        -- Mike Crowley 2011
  !
  use cmdpar,only:parins,numpar,toklen,toknam,vallen,valnam
  use dimens_fcm
  use stream
  use string, only: cnvtuc
  use rndnum
  use parallel
  ! VO string v
#if KEY_MULTICOM==1 && KEY_MPI==1
  use mpi          
#endif
#if KEY_MULTICOM==1
  use multicom_aux 
#endif
  ! VO string ^

  implicit none
  
  logical wantqua

  ! local
  integer narg,iarg,arglen,ind,iargc,ichar,ierr
  integer dirlen, oldsyn
  integer :: chsize_arg
  integer, parameter :: maxlen=120, mxargs=300
  character(len=maxlen) arg, argdir
  character(len=10) varname
  !
  !
  !     default to 'no quanta communications'
  wantqua=.false.
  !
  iarg=1
  oldsyn=0
  qcmdin=.false.
  qcmdout=.false.
  !     Process arguments (main loop: while (iarg<=MXARGS .and. arg^=' ') )

  qoldrandom = .false.
  qbrokenclcg = .false.

  call mgetarg(iarg, arg, arglen)
  args_loop: do while(arg(1:1) /= ' ' )
     ind = max(index(arg(1:arglen), ":"), index(arg(1:arglen), "="))

     if(prnlev >= 6) write(outu,25) arg(1:arglen)
25   FORMAT(' Processing passed argument "',A,'"')


     parameter_test: if (ind > 0) then       ! install parameter
        call cnvtuc(arg,arglen)
        ierr =  parins( arg(1:ind-1),ind-1, arg(ind+1:arglen),arglen-ind)
        if(ierr < 0) then
           IF(PRNLEV >= 2) CALL WrnDie(0,'<ARGUMT>', &
                'Failed to install parameter from command line')
        endif

     else parameter_test

        argument_cases: select case (arg(1:len_trim(arg)))

        case ('-input','-i') argument_cases
           IARG=IARG+1
           ARG=' '
           CALL MGETARG(IARG, ARG, ARGLEN)
           IF (ARG(1:1) == ' ') CALL WRNDIE(-5,'<CHANDLE>', &
                'Unexpected end of arguments after "-input"')
           qcmdin=.true.
           lcinfil=arglen
           cinfile(1:lcinfil)=arg(1:lcinfil)
           open(unit=5,file=cinfile(1:lcinfil))

        case('-output','-o') argument_cases
           IARG=IARG+1
           ARG=' '
           CALL MGETARG(IARG, ARG, ARGLEN)
           IF (ARG(1:1) == ' ') CALL WRNDIE(-5,'<CHANDLE>', &
                'Unexpected end of arguments after "-output"')
           qcmdout=.true.
           lcoutfil=arglen
           coutfile(1:lcoutfil)=arg(1:lcoutfil)
           open(unit=6,file=coutfile(1:lcoutfil))

        case('-help','-h') argument_cases
           if(mynod == 0) then
              write(outu,'(a)')'CHARMM>'
              write(outu,'(a)')'CHARMM> available command line arguments:'
              write(outu,'(a)') &
                   'CHARMM> <varname>=<value>  or  <varname>:<value>  '
              write(outu,'(a)') &
                   'CHARMM>                  sets @ variable in charmm to value'

              write(outu,'(a)') &
                   'CHARMM> -h, -help    This text.'
              write(outu,'(a)') &
                   'CHARMM> -input, -i f get input script from file f.'
              write(outu,'(a)') &
                   'CHARMM> -output,-o f put output to file f.'
              write(outu,'(a)') &
                   'CHARMM> -prevclcg    Compatibility mode with previous CLCG.'
              write(outu,'(a)') &
                   'CHARMM> -prevrandom  Compatibility mode with previous RANDOM.'
              write(outu,'(a)') &
                   'CHARMM> -chsize N    Allocate arrays to hold up to N atoms.'
              write(outu,'(a)')'CHARMM>'
           endif

           stop

        case('-prevrandom') argument_cases
           qoldrandom = .true.

        case('-prevclcg') argument_cases
           qbrokenclcg = .true.

! UHF MP2 in parallel needs this! Uncomment appropriate
! for DDIMPI:         case('-scr') argument_cases
! for DDIMPI:            !        these arguments were parsed for MPICH library
! for DDIMPI:            !        they have already been dealt with in MPI_INIT.
! for DDIMPI:            IARG=IARG+1
! for DDIMPI:            ARG=' '
! for DDIMPI:            CALL MGETARG(IARG, ARG, ARGLEN)
! for DDIMPI:            IF (ARG(1:1) == ' ') CALL WRNDIE(-5,'<CHANDLE>', &
! for DDIMPI:                 'Unexpected end of arguments after "-scr"')

! for DDI:         case('-ddi')  argument_cases
! for DDI:            !        these arguments were parsed for DDI library
! for DDI:            !        they have already been dealt with in ddikick
! for DDI:            !        Ignore the rest of them ??
! for DDI:            IARG=IARG+1
! for DDI:            ARG=' '
! for DDI:            CALL MGETARG(IARG, ARG, ARGLEN)
! for DDI:            IF (ARG(1:1) == ' ') CALL WRNDIE(-5,'<CHANDLE>', &
! for DDI:                 'Unexpected end of arguments after "-ddi"')

#if KEY_MPI==1
        case('-np') argument_cases
           !        this argument was parsed to get the number of processors for MPI.
           !        get the next argument (number of processors) and ignore both since
           !        they have already been dealt with in MPI_INIT.
           IARG=IARG+1
           ARG=' '
           CALL MGETARG(IARG, ARG, ARGLEN)
           IF (ARG(1:1) == ' ') CALL WRNDIE(-5,'<CHANDLE>', &
                'Unexpected end of arguments after "-np"')
           !     ELSE IF(ARG(1:5) == '-p4pg') THEN
        case('-p4pg')  argument_cases
           !        these arguments were parsed for MPICH library
           !        they have already been dealt with in MPI_INIT.
           IARG=IARG+1
           ARG=' '
           CALL MGETARG(IARG, ARG, ARGLEN)
           IF (ARG(1:1) == ' ') CALL WRNDIE(-5,'<CHANDLE>', &
                'Unexpected end of arguments after "-p4pg"')
           !     ELSE IF(ARG(1:5) == '-p4wd') THEN
        case('-p4wd') argument_cases
           !        this argument was parsed for MPICH library
           !        they have already been dealt with in MPI_INIT.
           IARG=IARG+1
           ARG=' '
           CALL MGETARG(IARG, ARG, ARGLEN)
           IF (ARG(1:1) == ' ') CALL WRNDIE(-5,'<CHANDLE>', &
                'Unexpected end of arguments after "-p4wd"')
#endif /*  MPI*/
           !     elseIF((ARG(1:7) == '-chsize'))THEN
        case('-chsize') argument_cases
           IARG=IARG+1
           ARG=' '
           CALL MGETARG(IARG, ARG, ARGLEN)
           IF (ARG(1:1) == ' ') CALL WRNDIE(-5,'<CHANDLE>', &
                'Unexpected end of arguments after "-chsize"')
           read(arg,'(i20)') chsize_arg
           call set_chsize(chsize_arg)

        case default argument_cases
#if KEY_MPI==1 || KEY_PARALLEL==1
              !     The problem for parameters is not easy solvable in case of
              !     MPICH library so for now we just ignore the errors.
              !     -np, -p4pg and -p4wd will not report the error, but 
              !     some combination of mpirun options are still hard to deal with
              CALL WRNDIE(5,'<CHANDLE>', &
                   'Unrecognized argument "'//ARG(1:ARGLEN)//'"')
#else /**/
              CALL WRNDIE(-5,'<CHANDLE>', &
                   'Unrecognized argument "'//ARG(1:ARGLEN)//'"')
#endif 
        end select argument_cases
     endif parameter_test

     iarg=iarg+1
     if (iarg > mxargs) then
        CALL WRNDIE(0,'<CHANDLE>', &
             'Too many command line arguments')
     endif
     call mgetarg(iarg, arg, arglen)
  enddo args_loop
  !
  !     end of loop by arguments

  !
  
! BIOVIA Code Start
#if KEY_LICENSE==1
  if (mynod .eq. 0) then
     call checkout_license_startup()
  endif
#endif
! BIOVIA Code End

#if KEY_PARALLEL==1 || KEY_ENSEMBLE==1
#if KEY_MULTICOM==1 && KEY_ENSEMBLE==0 /* (mcom)  multicom and .and..not.ENSEMBLE  use MPI for global broadcast */
  ! VO string v
      CALL MPI_BCAST(NUMPAR,1,MPI_INTEGER,0,MPI_COMM_GLOBAL,IND)
      IF(NUMPAR > 0) THEN
       CALL MPI_BCAST(TOKLEN,NUMPAR,MPI_INTEGER,0,MPI_COMM_GLOBAL,IND)
       CALL MPI_BCAST(VALLEN,NUMPAR,MPI_INTEGER,0,MPI_COMM_GLOBAL,IND)
       CALL MPI_BCAST(TOKNAM,NUMPAR*len(TOKNAM(1)),&
     &     MPI_BYTE,0,MPI_COMM_GLOBAL,IND)
       CALL MPI_BCAST(VALNAM,NUMPAR*len(VALNAM(1)),&
     &     MPI_BYTE,0,MPI_COMM_GLOBAL,IND)
      ENDIF
#else /* (mcom) */
!-- ##IF .not.PARALLEL .not.ENSEMBLE
  CALL PSND4(NUMPAR,1)
  IF(NUMPAR > 0) THEN
     CALL PSND4(TOKLEN,NUMPAR)
     CALL PSND4(VALLEN,NUMPAR)
     CALL PSNDC(TOKNAM,NUMPAR)
     CALL PSNDC(VALNAM,NUMPAR)
  ENDIF
!-- ##ELSE
!-- !MFC--- I think this is wrong, parallel ensemble should never send to world
!--   CALL PSND4_WORLD(NUMPAR,1)
!--   IF(NUMPAR > 0) THEN
!--      CALL PSND4_WORLD(TOKLEN,NUMPAR)
!--      CALL PSND4_WORLD(VALLEN,NUMPAR)
!--      CALL PSNDC_WORLD(TOKNAM,NUMPAR)
!--      CALL PSNDC_WORLD(VALNAM,NUMPAR)
!--   ENDIF
!-- ##ENDIF
#endif /*(mcom)  VO string ^ */
#endif 
  RETURN
END SUBROUTINE ARGUMT


SUBROUTINE MGETARG(IARG, ARG, ARGLEN)
  !-----------------------------------------------------------------------
  !     get one argument from the command line
  use,intrinsic :: iso_fortran_env
  use string
  INTEGER IARG, ARGLEN
  CHARACTER(len=*) ARG
  INTEGER, PARAMETER :: MAXLEN=120

  INTEGER*4 IARG4

  !
  ARG=' '

#if KEY_UNIX==1 /*machtype*/

#if KEY_GFORTRAN==1 || KEY_EM64T==1 /*f2003*/
  call get_command_argument(iarg,arg)
#elif KEY_PATHSCALE==1 /*f2003*/
  iarg4 = iarg
  call get_command_argument(iarg4,arg)
#else /* (f2003)*/
  ! PGI, osx/ifort, gnu/ifort
  CALL GETARG(IARG,ARG)
#endif /* (f2003)*/

  ARGLEN=MAXLEN
  CALL TRIME(ARG,ARGLEN)
#endif /* (machtype)*/
  RETURN
END SUBROUTINE MGETARG

! Gets the username associated with the current process and
!   places it in USERNAM
! -- mikem -- 07/22/92 -->
! getlog() doesn't work, if you are not attached to a terminal (like
! CHARMm started from QUANTA.) Neither 'whoami' does.
! However, getunamef in cstuff.c works just fine, so call it
subroutine getnam(out_name)
#ifdef __INTEL_COMPILER
  use ifport, only: getlog
#endif
  use cstuff, only: get_username
  
  implicit none
  
  character(len=*) out_name
  character(len=32) name
  integer len_name, max_len, error

  name = ' '
  len_name = 32
  max_len = len(out_name)

#if KEY_STATIC != 1
  call getlog(name)
#endif
  
  error = 0
  if (trim(name) .eq. '') then
     error = get_username(name, len_name)
  end if

  if (error .ne. 0) then
     call wrndie(0, '<GETNAM>', 'failed to get username')
  end if
  
  if (len_name > 0) then
     len_name = min(len_name, max_len)
     out_name(1:len_name) = name(1:len_name)
  end if

  if (len_name < max_len) then
     out_name((len_name + 1):max_len) = ' '
  end if
end subroutine getnam

SUBROUTINE SYSID(OSNAME)
  !----------------------------------------------------------------------
  !     GETS SYSTEM IDENTIFICATION (MACHINE, OS NAME AND NUMBER)
  !     AND PLACES IT IN THE STRING OSNAME.
  !
  !     AUTHOR: LENNART NILSSON MAY 1984
  !
#if KEY_GNU==1
  use parallel          
#endif
#if KEY_MPI==1 && KEY_PARCMD==1
  use mpi                  
#endif
  implicit none

#if KEY_NONETWORK==0 /*net*/
  integer k
  interface
    subroutine uninf(inf, ll, lhostnm, lhn) bind (c)
      use iso_c_binding, only: c_char, c_int
      character(kind=c_char, len=1), intent(out), dimension(80) :: inf, lhostnm
      integer(c_int), intent(inout) :: ll, lhn
    end subroutine uninf
  end interface
#endif

  CHARACTER(len=80) :: OSNAME

#if KEY_OSX==1
  ! try to get it from environment variable
  CALL GETENV('SYSID', OSNAME)
  IF(OSNAME  ==  ' ') OSNAME='MACINTOSH OSX'
#elif KEY_GNU==1

  ! getting gnu/iris
  character, dimension(80) :: inf_array
  character(len=80) :: INF
  INTEGER :: LL = 50, I   ! not i*4 : MH09
  character, dimension(80) :: lhostnm_array
  character(len=80) :: LHOSTNM
  integer :: lhn = 80
#if KEY_PARCMD==1
#if KEY_PARALLEL==1
  INTEGER NP(MAXNODE),IP(MAXNODE),ST    
#endif
#endif
  ! BIOVIA Code Start
  LL = 50
  lhn = 80
  ! BIOVIA Code End
  osname(1:80)=" "
  inf(1:80)=" "
  inf(1:6)="nohost"
#if KEY_NONETWORK==0 /*net*/
  CALL UNINF(INF_array, LL, LHOSTNM_array, LHN)
  write(inf, '(80a)') (inf_array(k), k = 1, 80)
  write(lhostnm, '(80a)') (lhostnm_array(k), k = 1, 80)
#endif /* (net)*/

  OSNAME(1:LL)=INF(1:LL)

#if KEY_PARALLEL==1 /*parallel*/
  IF((LL < 45).AND.(NUMNOD > 1))THEN
     WRITE(OSNAME(LL+1:LL+6),'(''[+'',I3,'']'')')NUMNOD-1
  ENDIF

#if KEY_PARCMD==1 /*par_cmd*/
  DO I=1,MAXNODE
     PARHOST(I)(1:80)=' '
     PARHLEN(I)=0
     NP(I)=80
     IP(I)=(I-1)*80
  ENDDO
  PARHOST(MYNODP)(1:LHN)=LHOSTNM(1:LHN)
  PARHLEN(MYNODP)=LHN
  CALL IGCOMB(PARHLEN,NUMNOD)
  !     No such routine in paral?.src files, so must be protected:
#if KEY_MPI==1
  CALL MPI_ALLGATHERV(PARHOST(MYNODP),80,MPI_BYTE,PARHOST, &
       NP,IP,MPI_BYTE,MPI_COMM_WORLD,ST)
#endif 
#endif /* (par_cmd)*/
  !
#endif /* (parallel)*/
#else /**/
#error  'Unrecognized machine type in SYSID'
  osname(1:80)=" "
  OSNAME='UNKNOWN'
#endif 
  RETURN
END SUBROUTINE SYSID

end module startup

