#if KEY_ENSEMBLE==1 /*ensemble_main*/

!----------------------------------------------------------------------
!            ENSINI
!----------------------------------------------------------------------
subroutine ensini
  !----------------------------------------------------------------------
  ! Initialize ensemble when charmm is started
  !----------------------------------------------------------------------
  use chm_kinds
  use stream
  use ensemble
  use parallel, only: mynod,numnod,comm_charmm
  use param_store, only: set_param
  use mpi

  implicit none
  
  integer l
  integer ierror
  character(len=ensbfl) tbuf

  
  ensemble_verbose = .false.
  !ensemble_verbose = .true.
  slave_ens = .false.
  old_mynod=mynod
  ensemble_layers=0
  ! Set WHOIAM and NENSEM for now
  !   these will be modified when/if the user calls ENSEMBLE NENSEM command
  whoiam = mynod
  nensem = numnod
  lmasternode=.true.
  call mpi_barrier(comm_charmm, ierror)

  comm_master = comm_charmm
  if(ensemble_verbose)then
     write(tbuf, '(a,i5,a)') &
          ' ENSEMBLE>    REPLICA NODE ', WHOIAM, ' STARTED!'
     call ensprn(outu,tbuf,len_trim(tbuf))

     write(tbuf,'(3i5)')prnlev,wrnlev,iolev
     call ensprint("  ENSINI>>  plev,wlev,iolev: ",tbuf)
  endif
  if (nensem > maxens) then
     if (iolev > 0) write (outu,'(a)') &
          ' ENSINI> TOO MANY REPLICAS!', &
          ' ENSINI> INCREASE MAXENS IN ensemble.src'
     CALL WRNDIE(0, '<ENSINI>', 'TOO MANY REPLICAS REQUESTED')
  endif

  call set_param('WHOIAM',whoiam)
  call set_param('NENSEM',nensem)
  jrex = .false.
  jrswap = .false.
  qensexp = .false.
  ensswmethod = 0
  ensas = .false.

  !     this is ignored:
!MFC-- Huh?
  swapc = .true.
  t2repo = -1
  rep2to = -1
  ensexpu = -1

  if (iolev > 0 .and. ensemble_verbose) then
#if KEY_STRINGM==1
#if KEY_ABPO==1
     write(outu,'(a)')  &
          ' ENSINI> ALL REPLICAS PRESENTLY USE ONE TEMPERATURE', &
          ' ENSINI> USE "ENSEmble EXCHange"', &
          ' ENSINI> TO SET UP REPLICA EXCHANGE.', &
          ' ENSINI> USE ENSEmble STRIng [...]', &
          ' ENSINI> TO USE THE STRING METHOD.', &
          ' ENSINI> USE ENSEmble ABPO [...]', &
          ' ENSINI> TO USE THE ABPO METHOD.'
#else /**/
     write(outu,'(a)')  &
          ' ENSINI> ALL REPLICAS PRESENTLY USE ONE TEMPERATURE', &
          ' ENSINI> USE "ENSEmble EXCHange"', &
          ' ENSINI> TO SET UP REPLICA EXCHANGE.', &
          ' ENSINI> USE ENSEmble STRIng [...]', &
          ' ENSINI> TO USE THE STRING METHOD.'
#endif 
#else /**/
#if KEY_ABPO==1
     write(outu,'(a)')  &
          ' ENSINI> ALL REPLICAS PRESENTLY USE ONE TEMPERATURE', &
          ' ENSINI> USE "ENSEmble EXCHange"', &
          ' ENSINI> TO SET UP REPLICA EXCHANGE.', &
          ' ENSINI> USE ENSEmble ABPO [...]', &
          ' ENSINI> TO USE THE ABPO METHOD.'
#else /**/
     write(outu,'(a)')  &
          ' ENSINI> ALL REPLICAS PRESENTLY USE ONE TEMPERATURE', &
          ' ENSINI> USE "ENSEmble EXCHange"', &
          ' ENSINI> TO SET UP REPLICA EXCHANGE.'
#endif 
#endif 
  endif
  return
end subroutine ensini


!----------------------------------------------------------------------
!          ENSFIN
!----------------------------------------------------------------------
subroutine ensfin
  !----------------------------------------------------------------------
  ! clean up on exit
  !----------------------------------------------------------------------
  use chm_kinds
  use stream
  use ensemble
  use dimens_fcm
  use psf
  use mpi
  use parallel,only:comm_charmm,mynod,numnod
  use parallel_groups,only:current_comm_index,allcomms
#if KEY_PINS==1 /*(partial inf. swapping)*/
  use pins, only:lpins,iswfin
#endif /* (pins) */
  
  implicit none

  integer ierror
  character(len=ensbfl) tbuf

#if KEY_PINS==1 /*(partial inf. swapping)*/
  if (lpins) call iswfin()
#endif /* (pins) */

!  do while(allcomms(comm_ens_index(current_ens_layer))%parent_index > 0 ) 
!     current_comm_index = allcomms(comm_ens_index(current_ens_layer))%parent_index
!  enddo
  do while(allcomms(current_comm_index)%parent_index > 0 ) 
     current_comm_index = allcomms(current_comm_index)%parent_index
  enddo
  comm_charmm=allcomms(current_comm_index)%comm
  mynod      =allcomms(current_comm_index)%mynod
  numnod     =allcomms(current_comm_index)%numnod

  call mpi_barrier(comm_charmm,ierror)

  !------------ Close open charmm.out files --------------------------
  

  !------------ Shut down communicators -------------------------------
  !--- comm_ensemble will be the one to shut down and all its children


  !------------ All done, charmm_comm is reset, if verbose, print message ----
  if(ensemble_verbose)then
     write(tbuf, '(a,i5,a)') &
          ' ENSEMBLE>    REPLICA NODE ', WHOIAM, ' STOPPING!'
     call ensprn(outu,tbuf,len_trim(tbuf))
  endif
  return 
end subroutine ensfin

!----------------------------------------------------------------------
!            ENS_GLOBAL_BARRIER
!----------------------------------------------------------------------
subroutine ens_global_barrier(ierror)
  use mpi
  use ensemble
  use parallel,only:comm_charmm,mynod
  implicit none
  integer,intent(out) :: ierror
  integer :: ier
  character(len=80) :: chint

  write(chint,'(2i4)')comm_charmm,mynod
  call ensprint("   ENS_SYNC>> calling rep barrier with communicator",chint)
  call mpi_barrier(comm_charmm,ierror)
  call ensprint("   ENS_SYNC>>  "," after rep barrier")

  if(lmasternode) call mpi_barrier(comm_master,ierror)
  call ensprint("   ENS_SYNC>>  "," after master barrier")
  ierror = ier + ierror
  return
end subroutine ens_global_barrier


!----------------------------------------------------------------------
!            ENSPRN
!----------------------------------------------------------------------
subroutine ensprn(unum,message,l)
  !-----------------------------------------------------------------------
  ! Print a message for each process, _via_ root. 
  !     UNUM: unit to print to
  !     MESSAGE: message to write for this node
  !     L: length of message
  !-----------------------------------------------------------------------
  use chm_kinds
  use stream
  use ensemble
  use parallel, only:mynod,comm_charmm
  use mpi
  implicit none

  character(len=*) message
  integer,intent(in) :: unum, l
  ! local
  integer i
  integer, parameter :: biglen=ensbfl*maxens
  character(len=biglen) msgarray
  integer ierror,lenst,stat(mpi_status_size)

  if(l == 0) return
  if(lmasternode)then
     if(nensem == 1) then
        write(unum, '(a)') message
     else
        lenst = l   !len_trim(message(1:1))
     endif
     if (whoiam == 0 ) write(unum, '(a)') message(1:lenst)
     do i=1,nensem-1
        if(whoiam == 0) then
           call mpi_recv(message,lenst,mpi_byte,i,i, &
                comm_master,stat,ierror)
           write(unum, '(a)') message(1:lenst)
           
        else
           call mpi_send(message,lenst,mpi_byte,0,i, &
                comm_master,ierror)
        endif
     enddo
  endif
  call gflush(unum)
  return
end subroutine ensprn

!----------------------------------------------------------------------
!            ENSCMD
!----------------------------------------------------------------------
subroutine enscmd(comlyn,comlen)
  !----------------------------------------------------------------------
  !     parse command line and call subcommands, namely
  !           OPEN
  !           CLOSE
  !           SEED
  !     etc...
  !----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use string
  use stream
  use dimens_fcm
  use coord
  use parallel,only:mynod
  use evb_mod,only:evbsetup
#if KEY_ABPO==1
  use abpo, only:abpo_setup  
#endif
  implicit none

  character(len=*) comlyn
  integer comlen,ierror

  if (indxa(comlyn,comlen,'OPEN')  >  0) then
     call ensopn(comlyn,comlen)
  else if (indxa(comlyn,comlen,'CLOSE')  >  0) then
     call ensclo(comlyn,comlen)
  else if (indxa(comlyn,comlen,'SEED')  >  0) then
     call enseed(comlyn,comlen)
  else if (indxa(comlyn,comlen,'PRSEED')  >  0) then
     call ensdpr(comlyn,comlen)
  else if (indxa(comlyn,comlen,'EXPAVG')  >  0) then
     call eavgsetup(comlyn,comlen)
  else if (indxa(comlyn,comlen,'EXCH')  >  0) then
     call ensrex(comlyn,comlen)
  else if (indxa(comlyn,comlen,'WRIT')  >  0) then
     call enswri(comlyn,comlen)
  else if (indxa(comlyn,comlen,'EVB') > 0) then
      call evbsetup(comlyn, comlen)
#if KEY_STRINGM==1
  else if (indxa(comlyn,comlen,'STRI')  >  0) then
     CALL parse_string_commands(comlyn,comlen) ! parse string commands
#endif 
#if KEY_ABPO==1
  else if (indxa(comlyn,comlen,'ABPO')  >  0) then
     call abpo_setup(comlyn,comlen) ! parse abpo commands
#endif 
  else if (indxa(comlyn,comlen,'SWON')  >  0) then
     jrswap=.true.
     if (iolev > 0)  &
          write(outu,'(a)') &
          'SWAPPING OF TEMPERATURE REPLICAS ENABLED'
  else if (indxa(comlyn,comlen,'SWOFF')  >  0) then
     jrswap=.false.
     if (iolev > 0)  &
          write(outu,'(a)')  &
          'SWAPPING OF TEMPERATURE REPLICAS DISABLED'
  else if (indxa(comlyn,comlen,'INFO')  >  0) then
     call ensinf()

  else if (indxa(comlyn,comlen,'NENSEM')  >  0) then
!     call ensini
     call ensnensem(comlyn,comlen)

  elseif (indxa(comlyn,comlen,'SYNC')  >  0) then
     call ens_sync(comlyn,comlen)

  else
     call wrndie(0, '<ENSPAR>', 'UNRECOGNIZED SUBCOMMAND')
  endif
  call ensprint("  ENSCMD>> returning "," ")
  return
end subroutine enscmd

!----------------------------------------------------------------------
!          ENSDPR
!----------------------------------------------------------------------
subroutine ensdpr(comlyn,comlen)
  !----------------------------------------------------------------------
  !     PRINT RANDOM SEED FOR EACH NODE
  !----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use dimens_fcm
  use consta
  use stream
  use string
  use number
  use reawri
  use parallel, only: mynod
  implicit none

  character(len=*) comlyn
  integer comlen
  ! local ...
  integer rseed,i,j
  integer seeds(maxens),myseed
  character(len=ensbfl) tbuf
  real(chm_real) r

  if (mynod == 0) then
     rseed=gtrmi(comlyn,comlen,'ROOT',-1)
     if (iolev > 0) then
        write(outu,'(a)') ' ENSEED> RANDOM SEEDS FROM EACH NODE: '
     end if
     write(tbuf,'(a,i3,a,i20)') ' ENSEMBLE> REPLICA NODE ', WHOIAM, ' HAS ISEED = ', ISEED
     call ensprn(outu,tbuf,len(tbuf))
  endif

  return
end subroutine ensdpr

!----------------------------------------------------------------------
!          ENSEED
!----------------------------------------------------------------------
subroutine enseed(comlyn,comlen)
  !----------------------------------------------------------------------
  !     The root node takes the random seed given on the command line
  !     and generates random seeds for all the other nodes, which are
  !     placed in the ?myseed variable on each node
  !     WARNING: THIS IS NOT A GOOD IDEA!
  !----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use dimens_fcm
  use consta
  use stream
  use string
  use number
  use clcg_mod
  use param_store, only: set_param
  use parallel, only: mynod,comm_charmm
  implicit none

  character(len=*) comlyn
  integer comlen
  ! local ...
  integer rseed,i,j
  integer seeds(maxens),myseed
  character(len=ensbfl) tbuf
  real(chm_real) r

  rseed=gtrmi(comlyn,comlen,'ROOT',-1)
  if (iolev > 0) then
     WRITE(OUTU,'(A)') ' ENSEED> ASSIGNING SEEDS AUTOMATICALLY'
     WRITE(OUTU,'(A)') ' ENSEED> THIS IS A BAD IDEA'
     WRITE(OUTU,'(A)') ' ENSEED> RATHER SUPPLY YOUR OWN SEEDS!'
  end if
  if (whoiam == 0 .and. mynod == 0) then
     seeds(1) = rseed
     do i=2,nensem
        r = random(rseed)
        seeds(i) = rseed
     end do
  endif

  call ens_bcast_all(seeds,nensem)
  myseed = seeds(whoiam+1)
  call set_param('MYSEED',myseed)
  write(tbuf,'(a,i3,a,i20)') &
       ' ENSEMBLE> REPLICA NODE ', WHOIAM, ' HAS SEED ', MYSEED
  call ensprn(outu,tbuf,len(tbuf))
  return
end subroutine enseed

!----------------------------------------------------------------------
!          ENSINQ
!----------------------------------------------------------------------
subroutine ensinq(mode,name,maxlen,length,qopen,qform,qwrite, &
     qens,unit)
  !----------------------------------------------------------------------
  !     taken from VINQRE()
  !
  !     file inquiry by file name or FORTRAN unit
  !     Flag QOPEN indicates whether file or unit is "open".
  !     Flag QFORM indicates whether file was opened formatted.
  !     Flag QWRITe indicates whether file was opened write-access.
  !     For inquiry by unit MAXLEN has to be specified (max length of NAME
  !     and LENGTH returns with the length of NAME.
  !     For inquiry by file the two names INPUT and OUTPUT are reserved
  !     for the standard input and output channels 5 and OUTU.
  !
  use chm_kinds
  use stream
  use dimens_fcm
  use machio
  use string
  use parallel
  use ensemble,only:comm_master,lmasternode
  implicit none
  character(len=mxcmsz) ename
  integer elen
  character(len=*) mode, name
  integer maxlen, length
  logical qopen, qform, qwrite, qens
  integer unit
  integer qshare(4),i

  qopen=.true.
  length=0

  if (mode == 'FILE'.and.name == 'INPUT') then
     unit=5
  else if (mode == 'FILE' .and. name == 'OUTPUT') then
     unit=outu
  else if (mode == 'FILE') then
     ename = name
     call expnam(ename,elen,qopen)
     qopen = .not.qopen
     unit = -1
     if(qopen) then
        inquire(file=ename(1:elen),opened=qopen,number=unit)
        if (unit <= 0.or.unit > 99) qopen=.false.
     endif
  else if (mode == 'UNIT') then
     inquire(unit=unit,opened=qopen,name=name)
     length=maxlen
     call trime(name,length)

  endif

  !     if file is open then get QFORM and QWRITE flags
  !     ... and QENS
  if (qopen.and.iolev > 0) then
     if ((ifreeu(unit) == +1).or.(ifreeu(unit) == +7)) then
        qform=.true.
        qwrite=.true.
     else if ((ifreeu(unit) == +10).or.(ifreeu(unit) == +70)) then
        qform=.true.
        qwrite=.false.
     else if ((ifreeu(unit) == -1).or.(ifreeu(unit) == -7)) then
        qform=.false.
        qwrite=.true.
     else if ((ifreeu(unit) == -10).or.(ifreeu(unit) == -70)) then
        qform=.false.
        qwrite=.false.
     else if (ifreeu(unit) == 0) then
        qform=.false.
        qwrite=.false.
     endif
     if ((mod(ifreeu(unit),7) == 0).and.(ifreeu(unit) /= 0)) then
        qens = .true.
     else
        qens = .false.
     endif
  endif
  !     if single rather than parallel input only root knows file
  !     status
  qshare(1:4) = 0

  if (iolev > 0) then
     if (qopen) qshare(1)  = 1
     if (qform) qshare(2)  = 1
     if (qwrite) qshare(3) = 1
     if (qens) qshare(4)  = 1
  endif
!MFC-- Need to check this one out, probably should not be world.
!  if(lmasternode)call psnd4_comm(comm_master,qshare,4)
!  call psnd4_comm(comm_charmm,qshare,4)
!HH: the following line causes error when ensinq is not called on all
!  processors, commented out by HH
!  call ens_bcast_all(qshare,4)
  qopen=.false.
  qform=.false.
  qwrite=.false.
  qens=.false.
  if (qshare(1) == 1) qopen = .true.
  if (qshare(2) == 1) qform = .true.
  if (qshare(3) == 1) qwrite = .true.
  if (qshare(4) == 1) qens = .true.
  return
end subroutine ensinq

!----------------------------------------------------------------------
!          ENSDMP
!----------------------------------------------------------------------
subroutine ensdmp()
  !-----------------------------------------------------------------------
  !     dumps coordinate files to "node0.pdb" "node1.pdb" etc on a 
  !     crash
  !-----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use dimens_fcm
  use stream
  use string
  use coord
  use psf
  use ctitla
  use machio
  use memory
  use coorio_mod,only:cwrite
  use parallel, only: mynod

  implicit none
  integer,allocatable,dimension(:) :: sel
  integer, parameter :: mxfile=128
  character(len=mxfile) filenm, junknm, tmp
  integer crashu,flen
  integer icntrl(20)
  integer i
  logical qopen,qform,qwrite,qens

  if (mynod == 0) then
     crashu = 50
     write(filenm, '(a,i2,a)') &
          'node',whoiam,'.pdb'
     if(filenm(5:5) == ' ') filenm(5:5)='0'
     flen=strlng(filenm)
     call ensinq('UNIT',junknm,mxfile,flen,qopen,qform,qwrite,qens, &
          crashu)
     if (qopen) then
        if(wrnlev >= 2.and.iolev > 0) write(outu,'(2a)') &
             ' ENSDMP> Unit already open.', &
             ' The old file will be closed first.'
        close(unit=crashu)   
        ifreeu(crashu) = 0
     endif
     OPEN(UNIT=CRASHU,FILE=FILENM,FORM='FORMATTED',STATUS='UNKNOWN', &
          ACCESS='SEQUENTIAL')
     call chmalloc('ensemble.src','ensdmp','sel',natom+1,intg=sel)
     sel(1:natom) = 1
     call cwrite(crashu,titlea,ntitla,icntrl,x,y,z,wmain, &
          res,atype,ibase,nres,natom,sel,4,0,0,.false.)
     
     close(unit=crashu)
     call chmdealloc('ensemble.src','ensdmp','sel',natom+1,intg=sel)
  endif

  return 
end subroutine ensdmp

!----------------------------------------------------------------------
!          ENSOPN
!----------------------------------------------------------------------
subroutine ensopn(comlyn,comlen)
  !-----------------------------------------------------------------------
  ! Opens 1 file per replica process
  ! TODO: Include checking to make sure each replica's file name is
  ! different (when opening with write access).
  !-----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use dimens_fcm
  use consta
  use stream
  use string
  use number
  use machio
  use parallel,only:mynod
  implicit none

  character(len=*) comlyn
  integer comlen

  integer mxpref, mxsuff
  integer flen, l, unum, unum2, ierror, i
  logical err, qopen, qform, qwrite, qens
#if KEY_UNIX==1
  integer, parameter :: mxfile=128
#else /**/
  integer, parameter :: mxfile=256
#endif 
  integer ipt
  integer, parameter :: fmlen=30,stslen=20,faccl=20
  character(len=mxfile) filex, junknm
  character(len=fmlen) formt
  character(len=stslen) fstat
  character(len=faccl) facc
  character(len=ensbfl) tbuf

 if(iolev < 0 )then
     call ensprint("   ENSOPN>> "," not opening file")
     comlyn=""
     comlen=0

  else
     unum=gtrmi(comlyn,comlen,'UNIT',-1)
     if (unum < 0) then
        call wrndie(0,'<OPNLGU>','NO UNIT NUMBER SPECIFIED')
        return
     endif
     
     ! filename
     call gtrmwa(comlyn, comlen, 'NAME', 4, filex, mxfile, flen)
     if (flen <= 0) then
        call wrndie(0, '<ENSOPN>', 'NO FILE NAME GIVEN')
     endif
     call ensprint("  ENSOPN>> " ,filex)

     ! remove quotes
     ipt=0
     do i=1,flen
        if(filex(i:i) /= '"') then
           ipt=ipt+1
           filex(ipt:ipt)=filex(i:i)
        endif
     enddo
     filex = filex(1:ipt)
     flen=ipt
     
     !     convert file name to lowercase
     if(lower) call cnvtlc(filex,flen)
     call ensprint("   ENSOPN>> filename",filex)
     
     !     unit already open?
     call ensinq('UNIT',junknm,mxfile,flen,qopen,qform,qwrite,qens, &
          unum)
     if (qopen) then
        if(wrnlev >= 2.and.iolev > 0) write(outu,'(2a)') &
             ' ENSOPN> Unit already open.', &
             ' The old file will be closed first.'
        close(unit=unum)   
        ifreeu(unum) = 0
     endif
     
     !     file already in use
     call ensinq('FILE',filex,i,i,qopen,qform,qwrite,qens,unum2)
     if (qopen) then
        if(wrnlev >= 2.and.iolev > 0) write(outu,'(a,/,2a)') &
             ' ENSOPN> ***** WARNING ***** another unit is already ', &
             '         assigned to the file -', &
             ' it will be disconnected first.'
        close(unit=unum2)   
        ifreeu(unum2) = 0
     endif
     
     ! format
     if (indxa(comlyn,comlen,'UNFO') > 0) then
        formt='UNFORMATTED'
     else if (indxa(comlyn,comlen,'FILE') > 0) then
        FORMT='UNFORMATTED'
     ELSE IF (INDXA(COMLYN,COMLEN,'FORM') > 0) THEN
        FORMT='FORMATTED'
     ELSE IF (INDXA(COMLYN,COMLEN,'CARD') > 0) THEN
        FORMT='FORMATTED'
     else
        call wrndie(1,'<OPNLGU>', &
             'NO FORMATTING SPECIFICATION, IT WILL BE OPENED UNFORMATTED')
        FORMT='UNFORMATTED'
     endif
     
     ! status
     call gtrmwa(comlyn, comlen, 'STAT', 4, fstat, stslen, l)
     if (l <= 0) then
        fstat = 'UNKNOWN'
     endif
     
     ! access
     facc='READ'
     if (indxa(comlyn,comlen,'APPE') > 0) then
        facc='APPEND'
     ELSE IF (INDXA(COMLYN,COMLEN,'READ') > 0) THEN
        FACC='READ'
     ELSE IF (INDXA(COMLYN,COMLEN,'WRIT') > 0) THEN
        FACC='WRITE'
     ELSE
        FACC='READ'
     endif

     tbuf(1:len(tbuf))=" "
     write (tbuf, '(a,I3)') ' ENSEMBLE>   REPLICA NODE ', WHOIAM
     call ensprint(tbuf(1:len_trim(tbuf))," ") 
     tbuf(1:len(tbuf))=" "
     write (tbuf, '(a26,a54)') ' ENSEMBLE>   OPENING FILE ', FILEX
     call ensprint(tbuf(1:len_trim(tbuf))," ") 
     tbuf(1:len(tbuf))=" "
     WRITE (tbuf, '(A,I3)')  ' ENSEMBLE>   ON UNIT ', UNUM
     call ensprint(tbuf(1:len_trim(tbuf))," ") 
     tbuf(1:len(tbuf))=" "
     WRITE (tbuf, '(A,A15,A,A10)')  ' ENSEMBLE>   WITH FORMAT ',  &
          FORMT, ' AND ACCESS ', FACC
     call ensprint(tbuf(1:len_trim(tbuf))," ") 

     ! open it
     IF (FACC == 'APPEND') THEN
        OPEN(UNIT=UNUM,FILE=FILEX,FORM=FORMT,STATUS='OLD', &
             ACCESS='SEQUENTIAL')
     ELSE IF (FACC == 'READ') THEN
        OPEN(UNIT=UNUM,FILE=FILEX,FORM=FORMT,STATUS='OLD', &
             ACCESS='SEQUENTIAL')
     ELSE IF (FACC == 'WRITE') THEN
        OPEN(UNIT=UNUM,FILE=FILEX,FORM=FORMT,STATUS='UNKNOWN', &
             ACCESS='SEQUENTIAL')
     END IF

     ! update ifreeu array
     INQUIRE(FILE=FILEX,OPENED=QOPEN,NUMBER=UNUM)
     IF (.NOT. QOPEN) THEN
        CALL WRNDIE(0, '<ENSOPN>', 'Could not open file')
     ELSE
        !
        !     put appropriate code in IFREEU array to play nicely
        !     with rest of charmm:
        !        +70 ensemble read formatted
        !        +10 read formatted
        !        +7  ensemble write/append formatted
        !        +1  write/append formatted
        !        -1  write/append unformatted
        !        -7  ensemble write/append unformatted
        !        -10 read unformatted
        !        -70 ensemble read unformatted
        !       i.e. ifreeu(unum)%7 tells whether we have ensemble file
        IF (FORMT == 'FORMATTED') THEN
           IFREEU(UNUM)=7
        ELSE
           IFREEU(UNUM)=-7
        ENDIF
        IF (FACC == 'READ') IFREEU(UNUM)=IFREEU(UNUM)*10
     endif
  endif
  call ensprint("   ENSOPN>> returning "," ")

  return
end subroutine ensopn

!----------------------------------------------------------------------
!          ENSCLO
!----------------------------------------------------------------------
subroutine ensclo(comlyn,comlen)
  !-----------------------------------------------------------------------
  ! Closes replica output
  !-----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use stream
  use dimens_fcm
  use string
  use machio
  use mpi
  use parallel,only:comm_charmm
  implicit none

  character(len=*) comlyn
  integer comlen
  integer unum, ierror
  character(len=ensbfl) tbuf

  ! words
  if (iolev > 0) then
     write(outu,'(a80)') comlyn
  endif
  unum=gtrmi(comlyn,comlen,'UNIT',-1)
  write (tbuf, '(a,i3,a,i3)') ' ENSEMBLE>   REPLICA NODE ',  &
       whoiam,  &
       ' CLOSING UNIT ', UNUM
  call ensprn(outu,tbuf,len(tbuf))

  ! action
  close(unit=unum)   
  ifreeu(unum) = 0
  call mpi_barrier(comm_charmm, ierror)

  return
end subroutine ensclo

!----------------------------------------------------------------------
!          ENSREX
!----------------------------------------------------------------------
subroutine ensrex(comlyn,comlen)
  !-----------------------------------------------------------------------
  ! Set up replica exchange -parses CHARMM command options for
  ! "ENSEMBLE EXCHANGE ..."
  !-----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use stream
  use dimens_fcm
  use string
  use consta
  use number,only: zero
  use memory
  use psf
  use parallel
  use param_store, only: set_param
#if KEY_PINS==1 /*(partial inf. swapping)*/
  use pins, only:lpins,iswparse,iswprintrex,iswini
#endif /* (pins) */

  implicit none
  !
  character(len=*) comlyn
  integer comlen
  !
  integer i,j,mapu,ii,swrules,tmpi
  logical jautot
  real(chm_real) lowt,tgrad,pswap,tmpt,a,b,c,deff

  ! words
  deff=zero
  jrex=.true.
  jrswap=.true.
  jensc=.true.
  ensas=.false.
  t2repo=gtrmi(comlyn,comlen,'T2RE',-1)
  rep2to=gtrmi(comlyn,comlen,'REP2',-1)
  ensfrq=gtrmi(comlyn,comlen,'FREQ',0)
  ensswmethod=gtrmi(comlyn,comlen,'SWME',0)
  call ens_bcast_all(ensswmethod,1)

#if KEY_PINS==1 /*(partial inf. swapping)*/
  lpins = indxa(comlyn,comlen,'PINS') .gt. 0
  if (lpins) then
    if (ensswmethod/=0)then
      call wrndie(-5,'<ENSREX>','Using partial infinite swapping (PINS) is not compatible with SWME other than 0')
    else
      call iswparse(comlyn,comlen)
    endif
  endif
#endif /* (pins) */

  mapu=gtrmi(comlyn,comlen,'MAPU',-1)
  swrules=gtrmi(comlyn,comlen,'RULE',-1)

!
!   if (iolev > 0) then
!      write(outu,'(a,i5,a)') ' ENSREX> RUNNING', &
!           nensem, ' REPLICA(S)'
  if (iolev > 0) then
     write(outu,'(a,i5,a)') ' ENSREX> RUNNING', &
          nensem, ' REPLICA(S)'
     write(outu,'(a)') ' ENSREX> SWITCHING REPLICAS EVERY '
     write(outu,'(a,i8,a)') ' ENSREX> ', ENSFRQ, ' DYNAMICS STEPS'
#if KEY_PINS==1 /*(partial inf. swapping)*/
    if (lpins) call iswprintrex()
#endif /* (pins) */
  endif
  
  if (indxa(comlyn, comlen, 'NOSC')  >  0) jensc=.false.
  if (indxa(comlyn, comlen, 'ASVE')  >  0) ensas=.true.
  i = indxa(comlyn, comlen, 'SWAPC')
  if (indxa(comlyn, comlen, 'NOSC')  >  0) jensc=.false.
  if (indxa(comlyn, comlen, 'ASVE')  >  0) ensas=.true.
  i = indxa(comlyn, comlen, 'SWAPC')
  if (iolev > 0) write(outu,'(a)') &
       ' ENSREX> WILL SWAP COORDINATES BETWEEN REPLICAS'
  call chmalloc('ensemble.src','ENSREX','ENSBUF',NATOM,crl=ENSBUF)
  !
  jautot = (INDXA(COMLYN, COMLEN, 'AUTO')  >  0)
  if (jautot) then
     LOWT=GTRMF(COMLYN,COMLEN,'LOWT',DEFF)
     TGRAD=GTRMF(COMLYN,COMLEN,'TGRAD',DEFF)
     PSWAP=GTRMF(COMLYN,COMLEN,'PSWAP',DEFF)
     !          ... set up temperature series automagically...  
     enstem(1)=lowt
     do i=2,nensem
        a=tgrad
        b=(-2.0*tgrad+log(pswap)*kboltz)*enstem(i-1)
        c=tgrad*enstem(i-1)**2
        enstem(i)=(-b+sqrt(b**2-4.0*a*c))/(2.0*a)
     enddo
  else
     do i=1,nensem
        tmpt = nextf(comlyn,comlen)
        enstem(i)=tmpt
        !write(6,*) "Read temperature ",tmpt," for rank ",i
     enddo
  endif

  if (mapu > 0) then
     if (iolev > 0) then
        do i=1,nensem
           read(mapu,*) ii,rep2t(i)
           t2rep(rep2t(i))=i
        enddo
     endif
     call ens_bcast_all(rep2t,nensem)
     call ens_bcast_all(t2rep,nensem)
  else 
     do i=1,nensem
        t2rep(i)=i
        rep2t(i)=i
     enddo
  endif

  if (swrules > 0) then
     !        read allowed swap rules from unit swrules
     if (iolev > 0) then
        read(swrules,*) ensnsw
        do i=1,ensnsw
           read(swrules,*) ensisw(i),ensjsw(i)
           if (ensisw(i) > ensjsw(i)) then
              tmpi=ensisw(i)
              ensisw(i)=ensjsw(i)
              ensjsw(i)=tmpi
           endif
        enddo
     endif
     call ens_bcast_all(ensnsw,1)
     call ens_bcast_all(ensisw,ensnsw)
     call ens_bcast_all(ensjsw,ensnsw)
  else 
     !        setup default swaps between 'adjacent' replicas
     ensnsw=nensem-1
     do i=1,ensnsw
        ensisw(i)=i
        ensjsw(i)=i+1
     enddo
#if KEY_PINS==1 /*(partial inf. swapping)*/
      if(lpins) then
        !  initialize infinite swapping
        call iswini()
        !ensnsw = nensem
      endif
#endif /* (pins) */
  endif

  if (iolev > 0) then
     write(outu,'(a)')' ENSREX> THE FOLLOWING ARE ALLOWED EXCHANGES'
     do i=1,ensnsw
        write(outu,'(a,i4,a,i4)') ' ENSREX> ',ENSISW(I),' <---> ', &
             ENSJSW(I)
     enddo
  endif

  ensatt(1:ensnsw)=0
  enssuc(1:ensnsw)=0

  repswp(1:nensem)=0
  do i=1,nensem
     if (whoiam == (i-1)) then
        ensmyt=enstem(rep2t(i)) 
        call set_param('ensmyt',ensmyt)
     endif
  enddo
  ensmyt=enstem(rep2t(whoiam+1))  !MFC does this do the same thing as above loop? 
  call set_param('ENSMYT',ensmyt)

  return
end subroutine ensrex

!----------------------------------------------------------------------
!          ENSWRI
!----------------------------------------------------------------------
subroutine enswri(comlyn,comlen)
  !     ... now redundant
  !-----------------------------------------------------------------------
  ! Write out replica target temperatures (for restart)
  !-----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use stream
  use dimens_fcm
  use string
  use parallel, only:mynod
  implicit none

  character(len=*) comlyn
  integer comlen

  integer unum,i

  if (mynod == 0) then
     UNUM=GTRMI(COMLYN,COMLEN,'UNIT',-1)
     if (iolev <= 0) return
     if (unum < 1)  call wrndie(-3,'<ENSWRI>', &
          'COULD NOT OPEN UNIT FOR OUTPUT.')
     do i=1,nensem
        write(unum,'(i5,i5)') i, rep2t(i)
     enddo
  endif

  return 
end subroutine enswri

!----------------------------------------------------------------------
!          ENSCHK
!----------------------------------------------------------------------
subroutine enschk()
  !-----------------------------------------------------------------------
  ! Check that all replicas set up OK before starting dynamics
  !-----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use stream
  use dimens_fcm
  use parallel, only:mynod
  implicit none
  integer i

  if (mynod == 0) then
     if (.not.jrex) return
     do i=1,nensem
        if (enstem(i) < 1.0) then
           call wrndie(-3,'<enschk>', &
                'TEMPERATURE NOT SET FOR SOME REPLICAS, QUITTING.')
        endif
     enddo
  endif

  return
end subroutine enschk

!----------------------------------------------------------------------
!          ENSINF
!----------------------------------------------------------------------
subroutine ensinf()
  !-----------------------------------------------------------------------
  ! Print replica exchange information
  !-----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use stream
  use dimens_fcm
  use parallel, only:mynod
  implicit none
  !
  integer i
  integer unum, ierror
  character(len=ensbfl) tbuf
  !
  if (iolev > 0) then
     write(outu,'(a)') &
          ' ENSINF> **************************************************'
     write(outu,'(a)') &
          ' ENSINF> CURRENT REPLICA EXCHANGE INFO:'
     write(outu,'(a,i5)') &
          ' ENSINF>           NUMBER OF REPLICAS = ', NENSEM
     if (jrswap) then
        write(outu,'(a)') &
             ' ENSINF>   REPLICA SWAPPING ENABLED'
     else
        write(outu,'(a)') &
             ' ENSINF>   REPLICA SWAPPING DISABLED'
     endif
     if (jensc) then
        write(outu,'(a)') &
             ' ENSINF>   TEMPERATURE SCALING ENABLED'
     else
        write(outu,'(a)') &
             ' ENSINF>   TEMPERATURE SCALING DISABLED'
     endif
     if (ensas) then
        write(outu,'(a)') &
             ' ENSINF>   TEMPERATURE ASSIGNING ENABLED'
     else
        write(outu,'(a)') &
             ' ENSINF>   TEMPERATURE ASSIGNING DISABLED'
     endif
     write(outu,'(a,i8)') &
          ' ENSINF>   FREQUENCY FOR ATTEMPTING SWAPS = ', ENSFRQ
     write(outu,'(a,i3)') &
          ' ENSINF>     WRITING REPLICA MAP TO UNIT = ', T2REPO
     write(outu,'(a,i3)') &
          ' ENSINF> WRITING TEMPERATURE MAP TO UNIT = ', REP2TO
     do i=1,nensem
        write(outu,'(a,i5,a,f8.3)') ' ENSINF> REPLICA ',I-1, &
             ' HAS TEMPERATURE ', enstem(rep2t(i))
     enddo
     write(outu,'(a)') &
          ' ENSINF> **************************************************'
     write(outu,'(a)') &
          ' ENSINF> AND NOW, A WORD FROM EACH REPLICA ...'
  endif
  if (mynod == 0) &
       write (tbuf, '(a,i5,a,f8.3)') ' ENSINF>   REPLICA NODE ',  &
            whoiam,  &
            ': MY TEMPERATURE IS ', ENSMYT
  call ensprn(outu,tbuf,len(tbuf))

  return
end subroutine ensinf

!----------------------------------------------------------------------
!          ENSNENSEM
!----------------------------------------------------------------------
subroutine ensnensem(comlyn,comlen)
  !-----------------------------------------------------------------------
  ! Set the number of replicas NENSEM
  !-----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use stream
  use dimens_fcm
  use string
  use memory
  use parallel
  use parallel_groups
  use param_store, only: set_param
  use mpi

  implicit none

  character(len=*) comlyn
  integer comlen

  integer i,j,k,l,node, startnode, endnode, numnod_orig
  integer ens_group, old_comm_charmm
  integer,save :: orig_comm_charmm,orig_charmm_group,orig_mynod
  integer new_charmm_group, new_comm_charmm, new_comm_charmm_index
  integer master_group,     master_comm,      comm_master_index
  integer ierror,status, everb
  integer charmm_group

  integer, allocatable, dimension(:) :: nodes, masternodes
  logical, save :: already_called = .false.
  character(len=14),dimension(0:maxens) :: chmout
  character(len=80) :: chint
  logical :: eof

  orig_comm_charmm=comm_charmm
  orig_mynod=mynod
  !-------- For purposes of being able to turn ensemble off, need a new communicator
  !-------- Make a new comm_charmm complete copy of current comm_charmm (identical)
  !-------- When turning ensemble off, need to kill all child communicators
  !-------- Killing this copy with the machinery in paralgroups, we can eliminate
  !--------   all child communicators and be left where we started before ensemble
  call set_ens_nensem_and_verbosity         !contained sub
  call setup_ens_comm_charmm     !contained sub

  call create_ens_communicators  !contained sub
  call open_chmout               !contained sub

  !---- Reset the printlevel/warnlevel/iolev for new group
  if(mynod /= 0) plnod0=0
  if(mynod /= 0) then
     prnlev=-1
     wrnlev=-5
     iolev=-1
  else
     call mpi_bcast(prnlev,1,mpi_integer,0,comm_master,ierror)
     call mpi_bcast(wrnlev,1,mpi_integer,0,comm_master,ierror)
     call mpi_bcast( iolev,1,mpi_integer,0,comm_master,ierror)
     call mpi_bcast(plnod0,1,mpi_integer,0,comm_master,ierror)
  endif

  call set_param('WHOIAM',whoiam)
  call set_param('NENSEM',nensem)
  call set_param('MYNODE',mynod)
  call set_param('NUMNODE',numnod)
  do i=1,maxnode
     inode(i)=mod(i,numnod)
  enddo
  call ensprint("   ENSNENSEM>> Testing outu unit","6")
  call cube(mynod,numnod,ippmap)

  !     Check priority of the main process and assign the same on the
  !     rest of processes
  !MFC  WTF??? got to figure this one out.....
#if KEY_GNU==1 || KEY_OSX==1
  i=0
  if(mynod == 0) call getppr(i)
  call psnd4(i,1)
  if(mynod /= 0) call setppr(i)
#endif 
  call ens_setup_stream_input()
  return

contains
  !----------------------------------------------------------------------
  subroutine setup_ens_comm_charmm
    call mpi_comm_group(COMM_CHARMM, orig_charmm_group, ierror)
    call chmalloc('ensemble.src','ensnensem','nodes',numnod,intg=nodes)
    do i=1,numnod
       nodes(i)=i-1
    enddo
    call mpi_group_incl(orig_charmm_group, numnod, nodes, new_charmm_group, ierror)
    call mpi_comm_create(orig_comm_charmm, new_charmm_group, new_COMM_CHARMM, ierror)
    call comm_save(orig_comm_charmm,new_comm_charmm,new_charmm_group, &
         mynod,numnod,comm_charmm_index,new_comm_charmm_index)
    comm_charmm_index = new_comm_charmm_index
    call chmdealloc('ensemble.src','ensnensem','nodes',numnod,intg=nodes)
    write(chint,'(i4)')new_comm_charmm
    call ensprint("setup_ens_comm_charmmcomm_ensemble",chint)

    numnod_orig = numnod
    numnod = numnod/nensem            ! numnod will now = number of cores per ensemble
    whoiam = mynod/numnod
    call check_and_allocate_node_tables   !internal subroutine

    !------------ Now set the comm_charmm to this child copy of the original
    current_ens_layer = ensemble_layers
    comm_ensemble(current_ens_layer) = new_comm_charmm
    comm_ens_index(current_ens_layer) = new_comm_charmm_index
    comm_charmm = new_comm_charmm
    old_comm_charmm=comm_charmm
    old_mynod=mynod
    call mpi_comm_group(COMM_CHARMM, ens_group, ierror)


    return
  end subroutine setup_ens_comm_charmm

  !----------------------------------------------------------------------
  subroutine create_ens_communicators
    !-- Set up groups. Make group for each set of processors (master group and member rep).
    call setup_ens_groups          !contained sub

    !--- Create COMM_CHARMM communicator
    !---    Redefine comm_charmm to be the one replica set of processors that
    !---    this processor belongs to, as defined above in mpi_group_incl

    call mpi_comm_create(comm_charmm, new_charmm_group, new_COMM_CHARMM, ierror)
    call comm_save(old_comm_charmm,new_comm_charmm,new_charmm_group, &
         mynod,numnod,comm_charmm_index,new_comm_charmm_index)

    write(chint,'(i4)')new_comm_charmm
    call ensprint("setup_ens_comm_charmm comm_charmm rep",chint)

    !--- Create COMM_MASTER communicator is the communicator 
    !---        between the master node only of each replica
    call mpi_comm_create(comm_charmm, master_group, COMM_MASTER, ierror)
    write(chint,'(i4)')comm_master
    call ensprint("setup_ens_comm_charmm comm_master",chint)

    ensmasternod=-1
    if(lmasternode) then
       call mpi_comm_rank(comm_master,ensmasternod,ierror)
    endif

    call chmdealloc('ensemble.src','ensnensem','nodes',numnod,intg=nodes)
    call chmdealloc('ensemble.src','ensnensem','masternodes',nensem,intg=masternodes)

    comm_charmm = new_comm_charmm

    if (numnod > 1) then
       call mpi_comm_rank(comm_charmm, mynod, ierror)
    else
       mynod = 0
    endif

    write(chint,'(2i4)')mynod,comm_charmm
    call ensprint("ENSNENSEM create_ens_communicators mynod, comm_charmm rep",chint)

    if (mynod /= 0) slave_ens = .true.
    mynodp = mynod + 1
    mynodg = mynod
    numnodg = numnod
    plnod0=prnlev

    mastermaster = (whoiam == 0 .and. mynod == 0)
    return
  end subroutine create_ens_communicators

  !----------------------------------------------------------------------
  subroutine setup_ens_groups
    !-- Set up groups. Make group for each set of processors.
    !-- First fill arrays with list of processors for each group and the master group
    node = 0
    lmasternode=.false.
    do i=1,nensem
       startnode = node
       do j=1,numnod
          nodes(j) = node
          node = node + 1
       enddo
       endnode = node - 1
       !-- Make a new group for a replica if this node is part of the group responsible
       !--  for that replica
       ! include this mynod to group charmm_group, if startnode <= mynod <= endnode
       if (mynod >= startnode .and. mynod <= endnode ) then
          call mpi_group_incl(ens_group, numnod, nodes, new_charmm_group, ierror)
       endif
       masternodes(i) = startnode
       lmasternode = lmasternode .or. (startnode == mynod)
    enddo
    master_node = 0
    call mpi_group_incl(ens_group, nensem, masternodes, master_group, ierror)
    return
  end subroutine setup_ens_groups

  !----------------------------------------------------------------------
  subroutine open_chmout
    logical opn
    !--- fill in default stdout file names------
    outloop: do i=0,9
       do j = 0,9
          do k = 0,9
             l = 100*i + 10*j + k
             if( l > maxens)exit outloop
             write(chmout(l),'("charmm.out.",3i1)')i,j,k
          enddo
       enddo
    enddo outloop

    if(mynod == 0 .and. whoiam > 0 ) then
       call ensprint("   ENSEMBLE>> opening file ",chmout(whoiam))
       inquire(unit=6,opened=opn)
       if(opn)then
!          inquire(unit=6,name=chmout(whoiam))
          call ensprint("   ENSEMBLE>> unit 6 already open: ",chmout(whoiam))
          close(6)
       endif
       open (unit=6, file=chmout(whoiam),status="REPLACE")
       call ensprint("   ENSEMBLE>> opened file ",chmout(whoiam))
    endif
    !    write(chint,'("6",3i6)')whoiam,mynod,numnod
    !    call ensprint("   ENSNENSEM>> Testing outu unit",chint(1:len_trim(chint)))
    if(iolev > 0 )write(outu,*)"Testing outu ",whoiam,mynod
    return
  end subroutine open_chmout

  !----------------------------------------------------------------------
  subroutine check_and_allocate_node_tables
    if (numnod*nensem /= numnod_orig) then
       write(outu,*)"ENSEMBLE>> ERROR numnod,nensem ",numnod,nensem
       CALL WRNDIE(-3,'<ENSNENSEM>', &
            'nensem must be factor of number of nodes (mod(numnod,nensem)=0).')
    endif
    
    IF (numnod < 1) CALL WRNDIE(-3,'<ENSNENSEM>', 'NUMNOD CANNOT BE SMALLER THAN ONE.')

    IF (nensem > maxens) then
       write(outu,*)"ENSEMBLE>  Maxens = ",maxens
       CALL WRNDIE(-3,'<ENSNENSEM>', &
            'Need to redimension chmout for more than maxens ensemble members')
    endif
    ensemble_layers = ensemble_layers + 1
    if(ensemble_layers > maxensemble_layers) &
         CALL WRNDIE(-3,'<ENSNENSEM>ensemble.src', &
         'Too many ensemble layers (ensemble nested too many times).')
         
    call chmalloc('ensemble.src','ensnensem','nodes',numnod,intg=nodes)
    call chmalloc('ensemble.src','ensnensem','masternodes',nensem,intg=masternodes)
  end subroutine check_and_allocate_node_tables

  !----------------------------------------------------------------------
  subroutine set_ens_nensem_and_verbosity
    if (iolev >= 0) then
       nensem = nexti(comlyn,comlen)
       !HH: changed 2 to 1 to enable the use of a single replica
       !if (nensem < 2) call wrndie(-3,'<ENSNENSEM>', 'Invalid number of ensemble copies.')
       if (nensem < 1) call wrndie(-3,'<ENSNENSEM>', 'Invalid number of ensemble copies.')
       everb=indxra(comlyn,comlen,'VERB',4,.true.) ! index of word verb* if found
    endif

    call psnd4(nensem,1)
    call psnd4(everb,1)
    ensemble_verbose = everb>0
  end subroutine set_ens_nensem_and_verbosity

end subroutine ensnensem


!----------------------------------------------------------------------
!          ENS_SETUP_STREAM_INPUT
!----------------------------------------------------------------------
subroutine ens_setup_stream_input()
  use chm_kinds
  use dimens_fcm,only:mxcmsz
  use stream
  use string,only:cnvtuc
  use comand,only:comlyn_save,comlen_save
  use ensemble,only:lmasternode,ensprint,ensmasternod,comm_master
  use mpi

  implicit none
  character(len=mxcmsz) :: line,linesav
  character(len=80) :: chint
  integer :: linelen
  logical keepreading,eof
  integer ierror


  !--- Reset input file reading -- Open file for each master and read through
  !---     the ensemble command
  if(ensmasternod == 0)then
     if(.not. qcmdin) &
          call wrndie("ENS_setup_stream<ensemble.src>", &
          "Input must NOT be stdin, use -i or -input")
  endif
  if(lmasternode) then
     call mpi_bcast(lcinfil,1,mpi_integer,0,comm_master,ierror)
     call mpi_bcast(cinfile,lcinfil,mpi_character,0,comm_master,ierror)
  endif
  write(chint,'(i4)')ensmasternod
  call ensprint("   ENS_setup_stream>> ensmasternod ",chint )
  if(ensmasternod>0)then
     open(unit=5,file=cinfile(1:lcinfil))
     rewind(5)
     call ensprint("   ENS_setup_stream>> opening input ",cinfile(1:lcinfil) )
     write(chint,'(i6)')istrm
     call ensprint("   ENS_setup_stream>> istrm ",chint )
     line(1:mxcmsz) = " "
     linesav(1:comlen_save) = comlyn_save(1:comlen_save)
     call cnvtuc(linesav,comlen_save)
     keepreading = .true.
     do while(keepreading)
        eof=.false.
!MFC        read(5,'(a)')line
        !-- setting qcmprs false so it will not broadcast ---
        call rdcmnd(line,mxcmsz,linelen,istrm,eof,.false.,.false., &
             'ENS_setup_stream read> ')
        write(chint,*)eof
        call ensprint("   ENS_setup_stream>> eof ",chint(1:len_trim(chint)) )
!MFC        if(eof)call mpi_abort(mpi_comm_world,-1,status)
        
        call ensprint("   ENS_setup_stream>> line returned",line(1:len_trim(line)) )
        call cnvtuc(line,len_trim(line))
        !        keepreading = .not. ( indx(line,len_trim(line),"ENSE",4) > 0 )
        keepreading = .not. ( line(1:comlen_save) == linesav(1:comlen_save) )
     enddo
          call ensprint("   ENS_setup_stream>> found ensemble line ",line(1:len_trim(line)))
  endif

  call ensprint("   ENS_setup_stream>> leaving "," ")


  return
end subroutine ens_setup_stream_input

!----------------------------------------------------------------------
!          ENS_SYNC
!----------------------------------------------------------------------
subroutine ens_sync(comlyn,comlen)
  !-----------------------------------------------------------------------
  ! Synchronises all the replica
  !-----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use stream
  use dimens_fcm
  use string
  use memory
  use parallel
  use parallel_groups
  use mpi
  use comand,only:comlyn_save,comlen_save
  implicit none

  character(len=*) comlyn
  character(len=4) wrd
  integer comlen,ierror
  character(len=80) :: chint

  wrd=nexta4(comlyn,comlen)
  
  call ens_global_barrier(ierror)

  return

end subroutine ens_sync



!----------------------------------------------------------------------
!          ENSTAT
!----------------------------------------------------------------------
subroutine enstat()
  !-----------------------------------------------------------------------
  ! Print replica exchange statistics at end of run
  !-----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use stream
  use dimens_fcm
  use mpi
  use number,only:zero
  implicit none

  integer i
  integer unum, ierror
  character(len=ensbfl) tbuf
  real(chm_real) ratio

  if (jrex.and.(iolev > 0)) then
     write(outu,'(a)') &
          ' ENSTAT> **************************************************'
     write(outu,'(a)') &
          ' ENSTAT> REPLICA EXCHANGE STATISTICS:'
     write(outu,'(a,i5)') &
          ' ENSINF>         NUMBER OF REPLICAS = ', NENSEM
     write(outu,'(a12,a10,a10,a10)') &
          'TRANSITION', 'ATTEMPTS', 'SUCCESSES', 'RATIO'
     do i=1,ensnsw
        if (ensatt(i) /= 0) then
           ratio = real(enssuc(i))/real(ensatt(i))
        else
           ratio = zero
        endif
        write(outu,'(2x,i3,a4,i3,2x,i10,i10,f10.3)') &
             ensisw(i),'<-->',ensjsw(i),ensatt(i),enssuc(i), &
             ratio
     enddo
     write(outu,'(a)') &
          ' ENSINF> **************************************************'
     call gflush(outu)
  endif
  return
end subroutine enstat

!----------------------------------------------------------------------
!          ENSSWL
!----------------------------------------------------------------------
subroutine ensswl(xold,yold,zold,xnew,ynew,znew,xcomp,ycomp,zcomp,natom)
  !-----------------------------------------------------------------------
  ! Attempt replica swap -- swapping coordinates between nodes
  ! for leapfrog dynamics (dynamc.src integrator)
  !-----------------------------------------------------------------------
  use chm_kinds
  use chm_types
  use ensemble
  use stream
  use dimens_fcm
  use number
  use reawri
  use nose_mod
  use energym
  use consta
  use coord
  use deriv
  use bases_fcm
  use image
  use comand
  use clcg_mod
  use parallel
  use mpi
#if KEY_PINS==1 /*(partial inf. swapping)*/
  use pins, only:lpins,isw_ensswl
#endif /* (pins) */

  implicit none
  integer natom
  real(chm_real),dimension(natom) :: xold,yold,zold,xnew,ynew,znew,xcomp,ycomp,zcomp
  !
  real(chm_real) tmp6(6),pist(3),pistmp(3)
  integer ierror,i,j,k,ri,rj,partner,tmpi,ii,s
  integer, parameter :: myenel=4
  real(chm_real) myene(myenel),ensene(myenel*maxens)
  real(chm_real) eii,eij,ejj,eji,cdel,rnum,scafac,oldt,db
  real(chm_real) p_i,v_i,p_j,v_j
  real(chm_real) pivi,pivj,pjvi,pjvj
  logical qswp,sendfirst,debug
  integer atfrst,atlast
  integer swtmpi(maxswp), swtmpj(maxswp), ssave(maxswp), kk
  real(chm_real) randtbl(maxswp), tempd
  integer tempi,status
  logical swapped

  !  If I is this node and J is potential swap partner, then:
  !  MYENE(1) = EI(XI)
  !  MYENE(2) = EI(XJ)

  debug=.false.
  !  debug=.true.

  !Wraping routine for vector distributed global broadcast
  call vdgbr(xcomp,ycomp,zcomp,1)

#if KEY_PINS==1 /*(partial inf. swapping)*/
  !if infinite swapping enabled use a different version of this subroutine, isw_ensswl(...), from pins.src
  if(lpins) then
    if(prnlev.ge.2)then
      write(outu,'(a)') "ENSSWL> Infinite swapping use detected in subroutine ensswl, &
             redirecting to isw_ensswl(...) from pins.src"
    endif
    call isw_ensswl(xold,yold,zold,xnew,ynew,znew,xcomp,ycomp,zcomp,natom)
    return
  endif
#endif /* (pins) */

  !     ROOT decides which nodes will attempt a swap
  setswap: if (mastermaster) then
     write(outu,'(a)') ' ******************************************************'
     write(outu,'(a)') ' **   Testing whether replicas need to be swapped ...**'

    ! attempt one swap between 2 replicas i and j, the swap is chosen randomly with s
     if (ensswmethod == 0) then
        s=int(random(iseed)*(ensnsw))+1
        i=ensisw(s)
        j=ensjsw(s)
        write(outu,'(a,i3,a,i3)')  &
             ' ATTEMPTING TO SWAP REPLICAS ', I, ' AND ', J 
        call flush(outu)
        repswp(1:nensem)=0
        repswp(i)=j
        repswp(j)=i
        ensatt(s) = ensatt(s)+1
     else
        
        do k=1,ensnsw
           randtbl(k) = random(iseed)
           ssave(k) = k
        enddo
        
        swapped = .true.
        do while (swapped)
           do k=1,ensnsw-1
              if (randtbl(k) > randtbl(k+1)) then
                 tempd = randtbl(k)
                 randtbl(k) = randtbl(k+1)
                 randtbl(k+1) = tempd
                 tempi = ssave(k)
                 ssave(k) = ssave(k+1)
                 ssave(k+1) = tempi
                 swapped = .true.
              else
                 swapped = .false.
              endif
           enddo
        enddo

        do k=1,ensnsw
           swtmpi(k) = ensisw(ssave(k))
           swtmpj(k) = ensjsw(ssave(k))
           ensatt(ssave(k)) = ensatt(ssave(k)) + 1
        enddo

     endif
  endif setswap

  kk = ensnsw
  if (ensswmethod /= 0) kk = 1

  swaploop: do while (kk <= ensnsw)
     if (ensswmethod == 1) then
        s = ssave(kk)
     
        if (mastermaster) then
           repswp(1:nensem) = 0
           i = swtmpi(kk)
           j = swtmpj(kk)
           repswp(i)=j
           repswp(j)=i
           write(outu,'(a,i3,a,i3)') ' ATTEMPTING TO SWAP REPLICAS ', I, ' AND ', J
           call flush(outu)
        endif
     endif

     !     ROOT broadcasts swap array to other nodes
     call ens_bcast_all(repswp,nensem)
     !     TRIAL NODES evaluate old and new energies 
     if (repswp(whoiam+1) /= 0) then
        partner=repswp(whoiam+1)
        sendfirst=(partner > (whoiam+1))
        !        must evaluate energy in main coordinates to 
        !        accommmodate fast energy routines
        atfrst=1
        atlast=natom
        x(atfrst:atlast)=xcomp(atfrst:atlast)
        y(atfrst:atlast)=ycomp(atfrst:atlast)
        z(atfrst:atlast)=zcomp(atfrst:atlast)
        !        energy of local coords in local energy function
        !        (just to be on the safe side, re-evaluate energy)
        call update(comlyn,comlen,x,y,z,wmain,.false.,.false.,.true., &
             .false.,.true.,0,0,0,0,0,0,0)
        call energy(x,y,z,dx,dy,dz,bnbnd,bimag,0)
        
        ! store in myene(1) the potential energy before swap
        myene(1)=eprop(epot)
        !        swap current coordinates
        if (debug) then
           write (outu,'(a14,2i3," x(1) = ",f12.6)') 'before: rep ',whoiam+1,mynod,x(1)
           call flush(outu)
        endif
        !        swap main coordinates
        call ensscv(x,ensbuf,natom,partner-1,sendfirst)
        call ensscv(y,ensbuf,natom,partner-1,sendfirst)
        call ensscv(z,ensbuf,natom,partner-1,sendfirst)
        if (debug) then
           write (outu,'(a14,2i3," x(1) = ",f12.6)') 'after: rep ',whoiam+1,mynod,x(1)
           call flush(outu)
        endif
        !        energy of remote coords in local energy function
        call update(comlyn,comlen,x,y,z,wmain,.false.,.false.,.true., &
             .false.,.true.,0,0,0,0,0,0,0)
        call energy(x,y,z,dx,dy,dz,bnbnd,bimag,0)
        
        ! store in myene(2) the potential energy after swap
        myene(2)=eprop(epot)
        if (debug) then
           write (outu,'(a,i3,a,e12.5)') 'node ', whoiam+1,  &
                ' olde = ', myene(1)
           write (outu,'(a,i3,a,e12.5)') 'node ', whoiam+1,  &
                ' newe = ', myene(2)
           call flush(outu)
        endif

     endif

     !     ALL NODES share energy arrays (though only above data will be used)

     call ens_global_barrier(ierror)
     if (mynod == 0) then
        call mpi_allgather( &
             myene, myenel,mpi_double_precision, &
             ensene,myenel,mpi_double_precision, &
             comm_master,ierror)
     endif
     !     ROOT decides if TRIAL NODES will swap
     if (ensmasternod == 0) then
        qswp=.false.
        eii=ensene((i-1)*myenel+1)
        eij=ensene((i-1)*myenel+2)
        ejj=ensene((j-1)*myenel+1)
        eji=ensene((j-1)*myenel+2)
        write(outu,'(a,e12.5)') 'eii (kcal/mol) = ', eii
        write(outu,'(a,e12.5)') 'eij (kcal/mol) = ', eij
        write(outu,'(a,e12.5)') 'ejj (kcal/mol) = ', ejj
        write(outu,'(a,e12.5)') 'eji (kcal/mol) = ', eji
        call flush(outu)

        db = (eij-eii)/(enstem(i)*kboltz) &
             +(eji-ejj)/(enstem(j)*kboltz)
        write(outu,'(a,f12.6)') 'db = ', db
        write(outu,'(a,f12.6)') 'exp(-db) = ', exp(-db)
        call flush(outu)
        if (db <= zero) then
           qswp=.true.
        else
           rnum = random(iseed)
           write(outu,'(a,f12.6)') 'rnum = ', rnum
           if (rnum <= dexp(-db)) then
              qswp=.true.
           else
              qswp=.false.
           endif
        endif
        if (qswp) then
           write(outu,'(a,i3,a,i3)')  &
                'swapping replicas for nodes ', i, &
                ' and ', j
           call flush(outu)
           !              this just tracks which rep is at which temp for post-analysis
           ri=t2rep(i)
           rj=t2rep(j)
           t2rep(i)=rj
           t2rep(j)=ri
           rep2t(ri)=j
           rep2t(rj)=i
           enssuc(s)=enssuc(s)+1
           repswp(i)=-j
           repswp(j)=-i
        else
           repswp(i)=j
           repswp(j)=i
        endif
     endif

     !     ROOT broadcasts swap array to other nodes
     if(lmasternode)then 
        call psnd4_comm(comm_master,repswp,nensem)
        call psnd4_comm(comm_master,t2rep,nensem)
        call psnd4_comm(comm_master,rep2t,nensem)
        write(outu,'(a,10i4)') 'rep2t ', rep2t(1:nensem)
        write(outu,'(a,10i4)') 't2rep ', t2rep(1:nensem)
        call flush(outu)
     endif
     call psnd4_comm(comm_charmm,repswp,nensem)

     call psnd4_comm(comm_charmm,rep2t,nensem)
     call psnd4_comm(comm_charmm,t2rep,nensem)

     !     TRIAL NODES swap coordinates if success
     !     else also swap other dynamics arrays
     if (repswp(whoiam+1) /= 0) then
        partner=abs(repswp(whoiam+1))
        if (partner > (whoiam+1)) then
           sendfirst=.true.
        else
           sendfirst=.false.
        endif
        if (repswp(whoiam+1) < 0) then
           !           ... success
           !           swap coordinates used in leapfrog algorithm
           call ensscv(xold,ensbuf,natom,partner-1,sendfirst)
           call ensscv(yold,ensbuf,natom,partner-1,sendfirst)
           call ensscv(zold,ensbuf,natom,partner-1,sendfirst)
           call ensscv(xnew,ensbuf,natom,partner-1,sendfirst)
           call ensscv(ynew,ensbuf,natom,partner-1,sendfirst)
           call ensscv(znew,ensbuf,natom,partner-1,sendfirst)
           call ensscv(xcomp,ensbuf,natom,partner-1,sendfirst)
           call ensscv(ycomp,ensbuf,natom,partner-1,sendfirst)
           call ensscv(zcomp,ensbuf,natom,partner-1,sendfirst)
           !           update nb
           call update(comlyn,comlen,xcomp,ycomp,zcomp,wmain, &
                .false.,.false.,.true., &
                .false.,.true.,0,0,0,0,0,0,0)
           oldt=enstem(partner)
           scafac=sqrt(ensmyt/oldt)
           if (jensc) then
              do k=1,natom
                 xnew(k)=(two*scafac-one)*xold(k)+scafac*(xnew(k)-xold(k))
                 ynew(k)=(two*scafac-one)*yold(k)+scafac*(ynew(k)-yold(k))
                 znew(k)=(two*scafac-one)*zold(k)+scafac*(znew(k)-zold(k))
              enddo
           endif

        else
           call update(comlyn,comlen,xcomp,ycomp,zcomp,wmain, &
                .false.,.false., &
                .true.,.false.,.true.,0,0,0,0,0,0,0)
        endif
     endif

     kk = kk + 1
  enddo swaploop
  

  return
end subroutine ensswl

!----------------------------------------------------------------------
!          ENSSCV
!----------------------------------------------------------------------
subroutine ensscv(array,tmpv,natom,partner,sendfirst)
  !-----------------------------------------------------------------------
  !     client-to-client array swapping for coordinate version of
  !     replica exchange
  !-----------------------------------------------------------------------
  ! In Parallel Ensemble, Send/Receive is between nodes with same mynod:
  ! 0   1   2  ... numnod-1
  ! |   |   |         |
  ! 0   1   2  ... numnod-1
  use chm_kinds
  use ensemble
  use stream
  use dimens_fcm
  use number
  use reawri
  use nose_mod
  use energym
  use parallel, only:mynod,numnod,comm_charmm
  use mpi
  implicit none
  integer,intent(in) :: natom,partner
  real(chm_real),dimension(natom) :: array,tmpv
  logical,intent(in) :: sendfirst

  integer ierror,stat(mpi_status_size)

  if (sendfirst) then
     if(lmasternode)then
        call mpi_send(array,natom,mpi_double_precision,partner,10, &
             comm_master,ierror)
        call mpi_recv(tmpv,natom,mpi_double_precision,partner,11, &
             comm_master,stat,ierror)
     endif
  ELSE
     if(lmasternode)then
        call mpi_recv(tmpv,natom,mpi_double_precision,partner,10, &
             comm_master,stat,ierror)
        call mpi_send(array,natom,mpi_double_precision,partner,11, &
             comm_master,ierror)
     endif
  endif
  call psnd8_comm(comm_charmm,tmpv,natom)

     array=tmpv
     return
end subroutine ensscv

!----------------------------------------------------------------------
!          ENSOUT
!----------------------------------------------------------------------
SUBROUTINE ENSOUT()
  !-----------------------------------------------------------------------
  ! write rex output
  !-----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use stream
  use dimens_fcm
  implicit none
  integer i
  if (mastermaster .and. iolev >= 0) then
     if (rep2to > 0) then
        write(rep2to,'(10i5)') rep2t(1:nensem)
        call gflush(rep2to)
     endif
     if (t2repo > 0) then 
        write(t2repo,'(10i5)') t2rep(1:nensem)
        call gflush(t2repo)
     endif
  endif
  return
end subroutine ensout

!----------------------------------------------------------------------
!          ENSAVE
!----------------------------------------------------------------------
SUBROUTINE ENSAVE(PARMS,BUFF,PARML)
  !-----------------------------------------------------------------------
  ! Simple linear ensemble averaging ...
  !-----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use mpi
  implicit none

  real(chm_real) PARMS(*), BUFF(*)
  INTEGER PARML,IERROR
  INTEGER I,M
  real(chm_real) AVE
  IF (NENSEM == 1) RETURN
  CALL MPI_BARRIER(MPI_COMM_WORLD, IERROR)
  CALL MPI_ALLGATHER(PARMS(1),PARML,MPI_DOUBLE_PRECISION,BUFF(1), &
       PARML,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,IERROR)
  DO I=1,PARML
     AVE = 0.0
     DO M=1,NENSEM
        AVE = AVE + BUFF((M-1)*PARML+I)
     ENDDO
     PARMS(I) = AVE/REAL(NENSEM)
  ENDDO
  RETURN
END SUBROUTINE ENSAVE

!----------------------------------------------------------------------
!          ENSAV3
!----------------------------------------------------------------------
SUBROUTINE ENSAV3(PARMS,BUFF,PARML)
  !-----------------------------------------------------------------------
  ! Third power (noe) averaging
  !-----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use mpi
  implicit none
  real(chm_real) PARMS(*), BUFF(*)
  INTEGER PARML,IERROR
  INTEGER I,M
  real(chm_real) AVE
  IF (NENSEM == 1) RETURN
  CALL MPI_BARRIER(MPI_COMM_WORLD, IERROR)
  CALL MPI_ALLGATHER(PARMS(1),PARML,MPI_DOUBLE_PRECISION,BUFF(1), &
       PARML,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,IERROR)
  DO I=1,PARML
     AVE = 0.0
     DO M=1,NENSEM
        AVE = AVE + (BUFF((M-1)*PARML+I))**(-3.0)
     ENDDO
     PARMS(I) = (AVE/REAL(NENSEM))**(-1.0/3.0)
  ENDDO
  RETURN
END SUBROUTINE ENSAV3

!----------------------------------------------------------------------
!          ENSAV6
!----------------------------------------------------------------------
SUBROUTINE ENSAV6(PARMS,BUFF,PARML)
  !-----------------------------------------------------------------------
  ! Sixth power (noe) averaging
  !-----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use mpi
  implicit none
  real(chm_real) PARMS(*), BUFF(*)
  INTEGER PARML,IERROR
  INTEGER I,M
  real(chm_real) AVE
  IF (NENSEM == 1) RETURN
  CALL MPI_BARRIER(MPI_COMM_WORLD, IERROR)
  CALL MPI_ALLGATHER(PARMS(1),PARML,MPI_DOUBLE_PRECISION,BUFF(1), &
       PARML,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,IERROR)
  DO I=1,PARML
     AVE = 0.0
     DO M=1,NENSEM
        AVE = AVE + (BUFF((M-1)*PARML+I))**(-6.0)
     ENDDO
     PARMS(I) = (AVE/REAL(NENSEM))**(-1.0/6.0)
  ENDDO
  RETURN
END SUBROUTINE ENSAV6

!----------------------------------------------------------------------
!          ENSS2
!----------------------------------------------------------------------
SUBROUTINE ENSS2(BXIJ,BYIJ,BZIJ,XIJBUF,YIJBUF,ZIJBUF,NPARM,RIJ,S2)
  !-----------------------------------------------------------------------
  ! Calculate order parameter over ensemble
  ! Takes r_ij vector components (BXIJ etc for this replica) and calculates
  ! S2 over ensemble. Also calculates components of forces, and returns
  ! this in BXIJ etc.
  !-----------------------------------------------------------------------
  use chm_kinds
  use ensemble
  use mpi
  implicit none
  real(chm_real) BXIJ(*),BYIJ(*),BZIJ(*),XIJBUF(*),YIJBUF(*),ZIJBUF(*)
  real(chm_real) RIJ(*),S2(*)
  INTEGER NPARM
  !     local vbls
  INTEGER IERROR,I,M
  real(chm_real) SX2,SY2,SZ2,SXY,SXZ,SYZ,XIJ,YIJ,ZIJ
  !
  IF (NENSEM == 1) RETURN
  CALL MPI_BARRIER(MPI_COMM_WORLD, IERROR)
  CALL MPI_ALLGATHER(BXIJ(1),NPARM,MPI_DOUBLE_PRECISION,XIJBUF(1), &
       NPARM,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,IERROR)
  CALL MPI_ALLGATHER(BYIJ(1),NPARM,MPI_DOUBLE_PRECISION,YIJBUF(1), &
       NPARM,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,IERROR)
  CALL MPI_ALLGATHER(BZIJ(1),NPARM,MPI_DOUBLE_PRECISION,ZIJBUF(1), &
       NPARM,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,IERROR)
  do i=1,nparm
     sx2 = 0.0
     sy2 = 0.0
     sz2 = 0.0
     sxy = 0.0
     sxz = 0.0
     syz = 0.0
     do m=1,nensem
        xij = xijbuf((m-1)*nparm+i)
        yij = yijbuf((m-1)*nparm+i)
        zij = zijbuf((m-1)*nparm+i)
        sx2 = sx2 + xij**2
        sy2 = sy2 + yij**2
        sz2 = sz2 + zij**2
        sxy = sxy + xij*yij
        sxz = sxz + xij*zij
        syz = syz + yij*zij
     end do
     sx2 = sx2 / real(nensem)
     sy2 = sy2 / real(nensem)
     sz2 = sz2 / real(nensem)
     sxy = sxy / real(nensem)
     sxz = sxz / real(nensem)
     syz = syz / real(nensem)
     !        order parameter:
     s2(i) = (1.5/rij(i)**4)*(sx2**2+sy2**2+sz2**2+2.0*sxy**2 &
          + 2.0*sxz**2 + 2.0*syz**2) - 0.5
     !        components of force:
     xij = bxij(i)
     yij = byij(i)
     zij = bzij(i)
     bxij(i) = sx2*xij+sxy*yij+sxz*zij
     byij(i) = sy2*yij+sxy*xij+syz*zij
     bzij(i) = sz2*zij+sxz*xij+syz*yij
  enddo
  return
end subroutine enss2

!----------------------------------------------------------------------
!          EAVGSETUP
!----------------------------------------------------------------------
subroutine eavgsetup(comlyn,comlen)
  !-----------------------------------------------------------------------
  ! the nodes share their forces and energy and and do exp(-H)=exp(-HA)+exp(-HB)
  ! averaging
  !
  ! ... setup & allocate storage at beginning
  !
  use chm_kinds
  use dimens_fcm
  use string
  use stream
  use psf
  use ensemble
  use number
  use memory
  implicit none
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  !
  ! local:
  INTEGER BUFLEN,I
  LOGICAL JOFFSET
  !
  QENSEXP = .true.
  BUFLEN = NATOM*NENSEM
  IF (IOLEV > 0) THEN
     WRITE(OUTU,'(A)') ' WILL DO EXPONENTIAL AVERAGING OF ENERGY'
  ENDIF
  !     write output?
  ENSEXPU=GTRMI(COMLYN,COMLEN,'UNIT',-1)
  IF ((IOLEV > 0).AND.(ENSEXPU > 0)) THEN
     WRITE(OUTU,'(A,I5)') ' WRITING EXP ENERGY OUTPUT TO UNIT', &
          ENSEXPU
  ENDIF
  !     set up beta for averaging
  ENSEBETA=GTRMF(COMLYN,COMLEN,'BETA',MINONE)
  IF (IOLEV > 0) THEN
     WRITE(OUTU,'(A,F12.5)') ' WEIGHTING ENERGIES USING BETA = ', &
          ENSEBETA
  ENDIF
  IF (INDXA(COMLYN, COMLEN, 'OFFSET')  >  0) THEN
     JOFFSET =.TRUE.
  ELSE
     JOFFSET =.FALSE.
  ENDIF
  IF (JOFFSET) THEN
     DO I=1,NENSEM
        EXPOFF(I) = NEXTF(COMLYN,COMLEN)
     ENDDO
  ELSE
     DO I=1,NENSEM
        EXPOFF(I) = 0.0
     ENDDO
  ENDIF
  IF (IOLEV > 0) THEN
     WRITE(OUTU,'(A)') ' ENERGY OFFSETS: '
     DO I=1,NENSEM
        WRITE(OUTU,'(A,I3,A,F12.5)') ' NODE = ', I-1,  &
             ' OFFSET = ', EXPOFF(I)
     ENDDO
  ENDIF
  IF (INDXA(COMLYN, COMLEN, 'BOLTZ')  >  0) THEN
     QEXPBW =.TRUE.
     IF (IOLEV > 0) THEN
        WRITE(OUTU,'(A)') 'USING ALTERNATIVE EXPONENTIAL AVE:'
        WRITE(OUTU,'(A)') '         exp(-2BUa)+exp(-2BUb)'
        WRITE(OUTU,'(A)') 'exp(-BU)=---------------------'
        WRITE(OUTU,'(A)') '         exp(-BUa)+exp(-BUb)'
     ENDIF
  ELSE
     QEXPBW =.FALSE.
     IF (IOLEV > 0) THEN
        WRITE(OUTU,'(A)') 'USING STANDARD EXPONENTIAL AVE:'
        WRITE(OUTU,'(A)') 'exp(-BU)=exp(-BUa)+exp(-BUb)'
     ENDIF
  ENDIF
  !     allocate space for MPI receive buffers
  call chmalloc('ensemble.src','EAVGSETUP','ENSDX',BUFLEN,crl=ENSDX)
  call chmalloc('ensemble.src','EAVGSETUP','ENSDY',BUFLEN,crl=ENSDy)
  call chmalloc('ensemble.src','EAVGSETUP','ENSDZ',BUFLEN,crl=ENSDZ)
  call chmalloc('ensemble.src','EAVGSETUP','ENSH',BUFLEN,crl=ENSH)
  !
  return
end subroutine eavgsetup

!----------------------------------------------------------------------
!          ENSEXPAVG
!----------------------------------------------------------------------
subroutine ensexpavg(qx, qy, qz, dxk,dyk,dzk,hk,natom)
  !-----------------------------------------------------------------------
  ! the nodes share their forces and energy and and do 
  ! exp(-H)=exp(-HA)+exp(-HB) averaging
  !
  ! ... calculate energy & forces at each time step

  use chm_kinds
  use dimens_fcm
  use stream
  use ensemble

  implicit none
  real(kind = chm_real) :: qx(*), qy(*), qz(*)
  real(chm_real) :: dxk(*),dyk(*),dzk(*),hk
  integer natom
  ! local:
  integer buflen
  !
  buflen = natom*nensem
 
  if(qensexp) then 
    call ensexpavg2(dxk,dyk,dzk,hk,natom,ensdx,ensdy,ensdz,ensh,buflen) 
  endif

  return

end subroutine ensexpavg

!----------------------------------------------------------------------
!          ENSEXPAVG2
!----------------------------------------------------------------------
subroutine ensexpavg2(dxk,dyk,dzk,hk,natom, &
     rcvdx,rcvdy,rcvdz,rcvh,buflen)
  !-----------------------------------------------------------------------
  ! the nodes share their forces and energy and and do exp(-H)=exp(-HA)+exp(-HB)
  ! averaging
  !
  ! ... calculate energy & forces at each time step
  !
  use chm_kinds
  use stream
  use ensemble
  use contrl
  use number
  use mpi
  implicit none
  !
  real(chm_real) DXK(*),DYK(*),DZK(*),RCVDX(*),RCVDY(*),RCVDZ(*)
  real(chm_real) RCVH(*),HK
  INTEGER NATOM,BUFLEN
  !     local
  INTEGER IERROR,I,M
  real(chm_real) EXPSUM,EXP2SUM,DXI,DYI,DZI,MINE
  real(chm_real) DXI2,DYI2,DZI2
  real(chm_real) TMPH(1)
  real(chm_real) EXPWT(MAXENS)
  !
  TMPH(1) = HK
  CALL MPI_BARRIER(MPI_COMM_WORLD, IERROR)
  CALL MPI_ALLGATHER(DXK(1),NATOM,MPI_DOUBLE_PRECISION,RCVDX(1), &
       NATOM,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,IERROR)
  CALL MPI_ALLGATHER(DYK(1),NATOM,MPI_DOUBLE_PRECISION,RCVDY(1), &
       NATOM,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,IERROR)
  CALL MPI_ALLGATHER(DZK(1),NATOM,MPI_DOUBLE_PRECISION,RCVDZ(1), &
       NATOM,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,IERROR)
  CALL MPI_ALLGATHER(TMPH(1),1,MPI_DOUBLE_PRECISION,RCVH(1), &
       1,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,IERROR)
  !     first calculate mean to use as an offset to avoid numerical
  !     errors in exp(-beta U) for very small (large negative) U.
  MINE=RCVH(1)+EXPOFF(1)
  DO I=2,NENSEM
     MINE = MIN(MINE,RCVH(I)+EXPOFF(I))
  ENDDO
  !     convert entries HK(I) to EXP(-beta*HK(I))
  EXPSUM=0.0
  EXP2SUM=0.0
  DO I=1,NENSEM
     EXPWT(I)=DEXP(-ENSEBETA*(RCVH(I)+EXPOFF(I)-MINE))
     EXPSUM=EXPSUM+EXPWT(I)
     IF (QEXPBW) EXP2SUM=EXP2SUM+(EXPWT(I))**2
  ENDDO
  !     calculate forces
  IF (QEXPBW) THEN
     !        write (outu,*) 'DOING EXPBW'
     !        call flush(outu)
     DO I=1,NATOM
        DXI = 0.0
        DYI = 0.0
        DZI = 0.0
        DXI2 = 0.0
        DYI2 = 0.0
        DZI2 = 0.0
        DO M=1,NENSEM
           DXI2=DXI2+RCVDX((M-1)*NATOM+I)*(EXPWT(M)**2)
           DYI2=DYI2+RCVDY((M-1)*NATOM+I)*(EXPWT(M)**2)
           DZI2=DZI2+RCVDZ((M-1)*NATOM+I)*(EXPWT(M)**2)
           DXI =DXI +RCVDX((M-1)*NATOM+I)*EXPWT(M)
           DYI =DYI +RCVDY((M-1)*NATOM+I)*EXPWT(M)
           DZI =DZI +RCVDZ((M-1)*NATOM+I)*EXPWT(M)
        ENDDO
        DXK(I) = TWO*DXI2/EXP2SUM-DXI/EXPSUM
        DYK(I) = TWO*DYI2/EXP2SUM-DYI/EXPSUM
        DZK(I) = TWO*DZI2/EXP2SUM-DZI/EXPSUM
     ENDDO
     HK = DLOG(EXP2SUM/EXPSUM)/(-ENSEBETA)+MINE
  ELSE
     DO I=1,NATOM
        DXI = 0.0
        DYI = 0.0
        DZI = 0.0
        DO M=1,NENSEM
           DXI=DXI+RCVDX((M-1)*NATOM+I)*EXPWT(M)
           DYI=DYI+RCVDY((M-1)*NATOM+I)*EXPWT(M)
           DZI=DZI+RCVDZ((M-1)*NATOM+I)*EXPWT(M)
        ENDDO
        DXK(I) = DXI/EXPSUM
        DYK(I) = DYI/EXPSUM
        DZK(I) = DZI/EXPSUM
     ENDDO
     HK = DLOG(EXPSUM)/(-ENSEBETA)+MINE
  ENDIF
  !     write output ...
  IF ((IOLEV >= 0).AND.(ENSEXPU > 0)) THEN
     IF(MOD(MDSTEP,NPRINT) == 0) THEN
        WRITE(ENSEXPU,'(I10$)') MDSTEP
        DO I=1,NENSEM
           WRITE(ENSEXPU,'(1X,F12.5$)') RCVH(I)
        ENDDO
        WRITE(ENSEXPU,'(1X,F12.5)') HK
     ENDIF
  ENDIF

  return
end subroutine ensexpavg2

!----------------------------------------------------------------------
!          ERDCMD
!----------------------------------------------------------------------
subroutine erdcmd(comlyn,mxcms2,comlen,unit,eof, &
     qcmprs,qprint,echost)
  !-----------------------------------------------------------------------
  !     parallel input version of rdcmnd
  use chm_kinds
  use dimens_fcm
  use cmdpar,only:parse1
  use string
  use stream
  use rtf,only:ucase


  implicit none
  integer mxcms2,comlen,unit
  character(len=*) comlyn
  !
  integer enrlen
  integer, parameter :: enrmax=20
  character(len=enrmax) enrst
  integer cardln
  integer, parameter :: mxcard=200
  character(len=mxcard) card
  character(len=1) stemp
  integer wdlen,iend,ipar,i,j
  logical eof,qprint,qcmprs
  character(len=*) echost
  character(len=1) hyphen/'-'/,exclmk/'!'/,atmark/'@'/,sdblq/'"'/
  character(len=1) sques/'?'/
  !
  if(eof) return

  !--- clean up the previous command line in case of something left
  comlyn=' '
  comlen=0
  if(unit < 0) then
     eof=.true.
     return
  endif

  if(qprint .and. prnlev >= 3) write(outu,'("  ")')

  loop0: do while(.true.)
     iend=1
     loop1: do while(iend == 1)
        read(unit,'(a)',end=9) card
        cardln=mxcard
        call trime(card,cardln)
        if(cardln == 0) cardln=1
        if(qprint.and.prnlev >= 3) write(outu,'(1x,a8,3x,a)') echost,card(1:cardln)
        iend=indx(card,cardln,exclmk,1)
     enddo loop1    ! if (iend == 1) goto 1
     
     if (iend /= 0) then
        cardln=iend-1
        call trime(card,cardln)
     endif
     
#if KEY_CFF==1
     if (ucase) then      
#endif
        call cnvtuc(card,cardln)
#if KEY_CFF==1
     endif      
#endif

     if(qcmprs) call cmprst(card,cardln)
     if(cardln == 0) exit loop0
     if(card(cardln:cardln) /= hyphen) exit loop0
     if(cardln == 1) cycle loop0
     if(comlen+cardln-1 > mxcms2) then
        call wrndie(-1,'<RDCMND>','Command line too long: truncated.')
     endif
     call addst(comlyn,mxcms2,comlen,card,cardln-1)
  enddo loop0

  if(comlen+cardln > mxcms2) then
     call wrndie(-1,'<RDCMND>','Command line too long: truncated.')
  endif
  call addst(comlyn,mxcms2,comlen,card,cardln)

  if(qcmprs) ffour=comlyn(1:4)
  call parse1(comlyn,mxcms2,comlen,qprint)
  if(comlen == 0)return

  !     Before returning the string make any energy
  !     substitutions that may be required.

2020 FORMAT(' ERDCMD substituted energy or value "',80A1)
2030 FORMAT(' ERDCMD: can not substitute energy "',80A1)

  ipar=indx(comlyn,comlen-1,sques,1)
  loop400: do while(ipar > 0)
     call copsub(enrst,enrmax,enrlen,comlyn,ipar+1, &
          min(comlen,enrmax+ipar))
     call subenr(wdlen,enrlen,enrst,enrmax)
     if (enrlen > 0) then
        if (qprint .and. prnlev >= 3) then
           write(outu,2020) (comlyn(j:j),j=ipar,ipar+wdlen),sdblq, &
                ' ','t','o',' ',sdblq,(enrst(j:j),j=1,enrlen),sdblq
        endif
        call copsub(scrtch,scrmax,scrlen,comlyn,ipar+wdlen+1,comlen)
        comlen=ipar-1
        call addst(comlyn,mxcms2,comlen,enrst,enrlen)
        call addst(comlyn,mxcms2,comlen,scrtch,scrlen)
        ipar=indx(comlyn,comlen-1,sques,1)
     else
        !          fix may-90/ln, we rather want the whole parameter name
        if(wrnlev >= 2) write(outu,2030) &
             (comlyn(j:j),j=ipar,ipar+wdlen),sdblq
        ipar=0
     endif
  enddo loop400

  !--- Trim the command line before return to the caller
  call trima(comlyn,comlen)
  return
9 eof=.true.

  return
end subroutine erdcmd

!=======================================================================
!     The routines below were lifted from the parallel code
!=======================================================================
!----------------------------------------------------------------------
!          PSYNC_ENS
!----------------------------------------------------------------------
subroutine psync_ens()
  !-----------------------------------------------------------------------
  !
  !     This is a wrapper routine for global sync.
  !
  use chm_kinds
  !
  use dimens_fcm
  use ensemble
#if KEY_CMPI==0
  use mpi    
#endif
  implicit none
  integer status

#if KEY_PARINFNTY==1
  if(qinfinity) return                 
#endif
  if(nensem == 1) return
  if (slave_ens) return
#if KEY_CMPI==1
  call cmpi_barrier(comm_master,status)
#endif 

  return
end subroutine psync_ens

!----------------------------------------------------------------------
!          ENS_BCAST_MASTERS
!----------------------------------------------------------------------
subroutine ens_bcast_masters(array, length)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other MASTER nodes
  ! and receive from node 0 on all other nodes.
  ! Usually called after read on node 0. For single precision arrays.

  use chm_kinds
  use dimens_fcm
  use ensemble
  use mpi

  implicit none
  integer,intent(in) :: length
  integer,dimension(length),intent(inout) :: array(*)

  integer status

  if(nensem == 1) return
  if (.not.lmasternode) return
  call mpi_bcast(array,4*length,mpi_byte,0,comm_master,status)
  return
end subroutine ens_bcast_masters

!----------------------------------------------------------------------
!          ENS_BCAST_ALL
!----------------------------------------------------------------------
subroutine ens_bcast_all(array, length)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other MASTER nodes
  ! Then broadcast to all nodes in each copy.
  use chm_kinds
  use dimens_fcm
  use ensemble
  use mpi    
  use parallel,only:comm_charmm
  implicit none
  integer,intent(in) :: length
  integer,intent(inout),dimension(length) :: array
  
  integer status
  
  if(nensem == 1) return
  if(lmasternode) then
     call mpi_bcast(array,4*length,mpi_byte,0,comm_master,status)
  endif
  call mpi_bcast(array,4*length,mpi_byte,0,comm_charmm,status)
  return
end subroutine ens_bcast_all

!----------------------------------------------------------------------
!          PSND8_ENS
!----------------------------------------------------------------------
subroutine psnd8_ens(array, length)
  !-----------------------------------------------------------------------
  ! This routine performs on node 0 broadcast to all other nodes
  ! and receive from node 0 on all other nodes.
  ! Usually called after read on node 0. For real(chm_real) precision arrays.

  use chm_kinds
  use dimens_fcm
  use ensemble
  use mpi
  implicit none
  real(chm_real) array(*)
  integer length

  integer status

  if(nensem == 1) return
  if (slave_ens) return
  call mpi_bcast(array,8*length,mpi_byte,0,comm_master,status)
  return
end subroutine psnd8_ens




#else /* (ensemble_main)*/


!====================== NO ENSEMBLES ... ==============================
subroutine enscmd(comlyn,comlen)
  !----------------------------------------------------------------------
  use stream
  character(len=*) comlyn
  integer comlen
  write (outu, '(a)') 'NO ENSEMBLE CODE COMPILED'
  return
end subroutine enscmd

subroutine ensfin
  return
end subroutine ensfin

subroutine enschk
  return
end subroutine enschk

#endif /* (ensemble_main)*/

