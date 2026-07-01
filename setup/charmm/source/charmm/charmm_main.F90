PROGRAM CHARMM
  !
  !      Chemistry at HARvard Macromolecular Mechanics
  !      -            ---     -              -
  !
  !      Version 48 - Developmental Version (c48b1) - August 15, 2023
  !
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !                                                                      C
  !      COPYRIGHT(c) 1984-2023                                          C
  !      President and Fellows of Harvard College                        C
  !                                                                      C
  !      All rights reserved                                             C
  !                                                                      C
  !      This copyright includes all source files and documentation.     C
  !                                                                      C
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !
  !      Present and past developers:
  !      Ioan Andricioaei
  !      Georgios Archontis
  !      Jay L. Banks
  !      Christian Bartels
  !      Robert Best
  !      Arnaud Blondel
  !      Stefan Boresch
  !      John Brady
  !      Bernard R. Brooks
  !      Charles L. Brooks III
  !      Robert E. Bruccoleri
  !      Axel Brunger
  !      Amedeo Caflisch
  !      Leo Caves
  !      Jhih-Wei Chu
  !      Michael Crowley
  !      Qiang Cui
  !      Ryszard Czerminski
  !      Aaron R. Dinner
  !      Ron Elber
  !      Marcus Elstner
  !      Jeff Evanseck
  !      Michael Feig
  !      Scott Feller
  !      Martin J. Field
  !      Stefan Fischer
  !      Stephen H. Fleischman
  !      Jiali Gao
  !      Michael Garrahan
  !      Bruce Gelin
  !      Garrett Goh
  !      Urs Haberthuer
  !      Thomas A. Halgren
  !      Sergio A. Hassan
  !      Milan Hodoscek
  !      Wonmuk Hwang (hwm)
  !      Antti-Pekka Hynninen
  !      Toshiko Ichiye
  !      Wonpil Im
  !      Mary E. Karpen
  !      Jennifer Knight
  !      Jeyapandian Kottalam
  !      Krzysztof Kuczera
  !      Themis Lazaridis
  !      Michael S. Lee
  !      Paul Lyne
  !      Jianpeng Ma
  !      Alex Mackerell
  !      Paul Maragakis
  !      Markus Meuwly
  !      Robert Nagle
  !      Benjamin Tim Miller
  !      Tom Ngo
  !      Lennart Nilsson
  !      Barry D. Olafson
  !      Victor Ovchinnikov
  !      Emanuele Paci
  !      Richard W. Pastor
  !      David Perahia
  !      Robert J. Petrella
  !      B. Montgomery Pettitt
  !      Carol B. Post
  !      Zingzhi Pu
  !      Walter E. Reiher III
  !      Daniel R. Roe
  !      Benoit Roux
  !      Michael Schaefer
  !      Jana Shen
  !      Paul Sherwood
  !      Tom Simonson
  !      Jeremy Smith
  !      David J. States
  !      Peter J. Steinbach
  !      Roland Stote
  !      John Straub
  !      Sundaramoothi Swaminathan
  !      Walter Thiel
  !      Bruce Tidor
  !      Douglas J. Tobias
  !      Don G. Truhlar
  !      Arjan van der Vaart
  !      Richard M. Venable
  !      Herman van Vlijmen
  !      Joanna Wiorkiewicz
  !      Masa Watanabe
  !      Youngdo Won
  !      Lee H. Woodcock
  !      Thomas B. Woolf
  !      Xiongwu Wu
  !      Wei Yang
  !      Darrin M. York
  !      William S. Young
  !
  !      Developed under the overall direction of Martin Karplus,
  !      Department of Chemistry & Chemical Biology, Harvard University,
  !      12 Oxford Street, Cambridge, MA 02138
  !
  !      Before including in a publication any data calculated using
  !      CHARMM, please contact Martin Karplus at the address above to get
  !      the appropriate publication(s) to be referenced.
  !
  !      Refer to the documentation for information and usage.
  !
  !      DIMENSIONING INFORMATION - SEE PARTICULAR .FCM FILES
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use bases_fcm
  use comand
  use ctitla
  use image
  use inbnd
  use param_store, only: param_store_init, set_param
  use param
  use pert
  use psf
  use startup
  use stream
  use string
  use timerm
  use new_timer,only:init_timers,timer_start,t_total              
  use parallel
#ifdef KEY_MPI
  use mpi
#endif
  use repdstr
#if KEY_REPDSTR==1
  use REPDSTRMOD                           
#endif
  use intcor_module,only:initialize_icr_structs
  use machutil,only:Initialize_timers
  use machio,only:vopen
  use vangle_mm, only: ptrini
#if KEY_CFF==1
  use rtf,only: ucase    
#endif
  use usermod,only: usrini
  use cmdpar,only:cmdpar_init
#if KEY_ENSEMBLE==1
  use ensemble     
#endif
#if KEY_GAMUS==1
  use gamusmodule,only:gamusinit 
#endif
#if KEY_DHDGB==1
!AP/MF
  use dhdgb
  use derivdhdgb
#endif

  ! work-around for intel 18 compiler bug affecting fsystem
#if (__INTEL_COMPILER == 1800) && defined(_OPENMP)
  use ifport, only: setenvqq
#endif
  
  implicit none

  logical quantaok    

  !     The following are local variables.
  integer   istart
  logical   eof,error
  logical   lused,ok,oktmp
  logical   wantqu
  logical qrdcmd,get_next_cmd

  !--mfc temporary variables
  integer :: j,ierr
  character(len=4) :: winit

  ! work-around for intel 18 compiler bug affecting fsystem
#if (__INTEL_COMPILER == 1800) && defined(_OPENMP)
  error = setenvqq("KMP_INIT_AT_FORK=FALSE")
#endif
  
  !
  !*********************************************************
  !     END DECLARATIONS -- EXECUTION BEGINS
  !*********************************************************
  !
  !     Set I/O units and zero mscpar totals; default to lower case file names
  outu=poutu
  prnlev=5
  lower=.true.
  bomlev=0
  iolev=1
  wrnlev=5

  call param_store_init()

  call set_param('BOMLEV',bomlev)
  call set_param('WRNLEV',wrnlev)
  call set_param('PRNLEV',prnlev)
  call set_param('IOLEV',iolev)
#if KEY_CFF==1
  ucase = .true. 
#endif

  !     Start times and do machine specific startup.
  call Startup_machine_dependent_code !used to be called from jobini
  call Initialize_timers   ! used to be jobini

  !     Get the CHARMM command line arguments and initialize for different
  !     platforms different variables. Check if CHARMM should communicate
  !     with QUANTA or not
  call cmdpar_init
  call argumt(wantqu)
  call set_dimens()

  call init_timers()
  call timer_start(T_total)

  !     open standard input on unit 5 and standard output on unit OUTU.
  call vopen(5,'$$INITIAL$$',' ',' ',error,0)

  !     print a header for the output.
  call header

  !     Initialize the index common blocks

  quantaok=.false.

  !     Open the input file and read title for run
  !     attention: 'call header' have already been done by now
  if (quantaok) then
     nstrm=0
     istrm=-1
     eof=.false.
  else
     nstrm=1
     istrm=5
     jstrm(nstrm)=istrm
     eof=.false.
     call tryoro(istrm,'FORMATTED')
     ntitla=0
     call rdtitl(titlea,ntitla,istrm,0)
  endif

  call initialize_icr_structs()
#if KEY_TSM==1
  call tsminit(.false.)       
#endif
  call iniall

  !     Initialize local variables
  call getpref()
  comlen=0
  altlen=0
  istart=1
  ! checking for dimension chsize command or related max.. sizes
  do while(comlen == 0)
     call rdcmnd(comlyn,mxcmsz,comlen,istrm,eof,.true.,.true., 'CHARMM> ')
#if KEY_ENSEMBLE==1
     call sav_comlyn(comlyn,comlen)     
#endif
  enddo
  call parse_size(comlyn,comlen,qrdcmd)



  call allocate_all
  !     Initialize the rest, simply because I cannot figure out right 
  !     now whether the stuff in gtnbct needs the allocation first.
  !     Eventually the following lines will probably move into  iniall
  !     above.
  j=4
  winit='INIT'
  call gtnbct(winit,j,bnbnd)


  !=======================================================================
  ! call user defined startup routine -- mfc pulled this from iniall
  call usrini
  !=======================================================================

  !     Initialize pointers
  call ptrini

  !=======================================================================
  !     START OF MAIN COMMAND LOOP
  !=======================================================================
  !     before reading the next command, make sure that the variables have not
  !     exceeded their bounds.

  main_cmd_loop: do while(.true.)
     ok = check_for_exceed_bounds()  !internal function
     !     Check for unparsed junk of the last command.
     if(qrdcmd) CALL XTRANE(COMLYN,COMLEN,'CHARMM')

     get_next_cmd=.true.
     miscom_cmd_loop:do while(get_next_cmd)

        !     If a timelimit has passed, execute the alternate commandline
        call chklim
        if (atlim) then
           !     Reset deadlines (otherwise we could have an infinite loop here)
              cpulim=0.0
              deadhr=-1
              if(prnlev >= 2) write(outu,*) &
                   'CHARMM: ',LIMTYP,' timelimit(s) exceeded'
              if (altlen > 0) then
                 if(prnlev >= 2) then
                    WRITE(OUTU,*) 'CHARMM: Executing alternate command'
                    WRITE(OUTU,*) 'ALTCOM: ',ALTCOM(1:ALTLEN)
                 endif
                 call copyst(comlyn,mxcmsz,comlen,altcom,altlen)
                 goto 120
              endif
              if(prnlev >= 2) write(outu,*) 'CHARMM: Terminating execution'
              call stopch('limit reached')
        endif

        !---------------------------------------------------------------
        !     main loop command reader
        if(qrdcmd) then
#if KEY_ENSEMBLE==1
           call ensprint(" CHM>> main loop"," ") 
#endif
           call rdcmnd(comlyn,mxcmsz,comlen,istrm,eof,.true.,.true., &
                'CHARMM> ')
#if KEY_ENSEMBLE==1
           call sav_comlyn(comlyn,comlen)     
#endif
#if KEY_ENSEMBLE==1
           call ensprint(" CHM>>>>> ",comlyn(1:len_trim(comlyn))) 
#endif
!           call psync_world
        endif

        qrdcmd = .true.
        
120     continue

! --- Start BIOVIA ---
! Check out a license for this command
#if KEY_LICENSE==1
      IF(MYNOD .EQ. 0) THEN
         CALL checkout_license_command(COMLYN,COMLEN)
      ENDIF
#endif
! --- End BIOVIA --- 

        ! See if it is one of the miscellaneous commands...
        call miscom(comlyn,mxcmsz,comlen,lused)

        get_next_cmd = ( lused .and. (.not. eof) ) 
     end do miscom_cmd_loop

     eoftest: if (eof) then
        !     If we run out of stuff on a particular stream, pop to the
        !     previous stream. quit when there are no more streams.
140     call ppstrm(ok)

#if KEY_REPDSTR==1
        !     This is excuted by everyone, so we can broadcast here if process 0
        !     wants to finish. Always restore the local parallel setup: this is safe
        if(qrepdstr)then
           call psetglob
           call psnd4(ok,1)
           call psetloc
        endif
#endif 

        if (.not.(ok)) then
           call stopch('END OF FILE')
        endif
        eof=.false.
        !     Don't loop back to read from a script if no script
        if(quantaok.and.istrm == 0) goto 140
        cycle main_cmd_loop
     endif eoftest

     ! parse the command
#    if KEY_REPDSTR==1
     if (nrepdstr.lt.2.or.repd_inside_if_block.eq.0) then
       ! No REPD, only 1 replica, or not in an 'if' block. Just execute the command.
       call maincomx(comlyn, comlen, lused)
     else if ( nrepdstr.gt.1.and. &
               repd_inside_if_block.gt.0.and. &
               repd_if_block_stat(repd_inside_if_block).eq.1 ) then
       ! Inside a REPD 'if' block and stat is OK. Execute the command.
       call maincomx(comlyn, comlen, lused)
     endif
#    else
     call maincomx(comlyn,comlen,lused)
#    endif

  end do main_cmd_loop
  !--------------------------------------------------------------------

contains
  logical function check_for_exceed_bounds() result(ok)
    ok= natc <= maxatc .and. ncb <= maxcb .and. &
         nct <= maxct .and. ncp <= maxcp .and. nci <= maxci .and. &
         nch <= maxch .and. ncn <= maxcn &
#ifndef KEY_RESIZE
         .and. natc <= maxatc .and. ncb <= maxcb .and. &
         nct <= maxct .and. ncp <= maxcp .and. nci <= maxci .and. &
         nch <= maxch .and. ncn <= maxcn &
#if KEY_CMAP==1
         .and. ncrterm <= maxcrt &
#endif
#endif  /* KEY_RESIZE */    
         ;  ! needed to close ok=
    IF (.NOT.(OK)) THEN
       IF(WRNLEV.GE.2) THEN
          ! Write out the list of conditions so that one can see what
          ! actually caused the error.
#ifndef KEY_RESIZE
          write(outu,'(a17,l5,2i10)')  'NSEG <= MAXSEG   ',NSEG <= MAXSEG   ,NSEG,MAXSEG 
          write(outu,'(a17,l5,2i10)')  'NATOM <= MAXA    ',NATOM <= MAXA    ,NATOM,MAXA 
          write(outu,'(a17,l5,2i10)')  'NBOND <= MAXB    ',NBOND <= MAXB    ,NBOND,MAXB 
          write(outu,'(a17,l5,2i10)')  'NTHETA <= MAXT   ',NTHETA <= MAXT   ,NTHETA,MAXT 
          write(outu,'(a17,l5,2i10)')  'NPHI <= MAXP     ',NPHI <= MAXP     ,NPHI,MAXP 
          write(outu,'(a17,l5,2i10)')  'NIMPHI <= MAXIMP ',NIMPHI <= MAXIMP ,NIMPHI,MAXIMP 
          write(outu,'(a17,l5,2i10)')  'NNB <= MAXNB     ',NNB <= MAXNB     ,NNB,MAXNB 
          write(outu,'(a17,l5,2i10)')  'NDON <= MAXPAD   ',NDON <= MAXPAD   ,NDON,MAXPAD 
          write(outu,'(a17,l5,2i10)')  'NACC <= MAXPAD   ',NACC <= MAXPAD   ,NACC,MAXPAD 
          write(outu,'(a17,l5,2i10)')  'NRES <= MAXRES   ',NRES <= MAXRES   ,NRES,MAXRES 
#endif
          write(outu,'(a17,l5,2i10)')  'NATC <= MAXATC   ',NATC <= MAXATC   ,NATC,MAXATC 
          write(outu,'(a17,l5,2i10)')  'NCB <= MAXCB     ',NCB <= MAXCB     ,NCB,MAXCB 
          write(outu,'(a17,l5,2i10)')  'NCT <= MAXCT     ',NCT <= MAXCT     ,NCT,MAXCT 
          write(outu,'(a17,l5,2i10)')  'NCP <= MAXCP     ',NCP <= MAXCP     ,NCP,MAXCP 
          write(outu,'(a17,l5,2i10)')  'NCI <= MAXCI     ',NCI <= MAXCI     ,NCI,MAXCI 
          write(outu,'(a17,l5,2i10)')  'NCH <= MAXCH     ',NCH <= MAXCH     ,NCH,MAXCH 
          write(outu,'(a17,l5,2i10)')  'NCN <= MAXCN     ',NCN <= MAXCN     ,NCN,MAXCN
#ifndef KEY_RESIZE
#if KEY_CMAP==1
          write(outu,'(a17,l5,2i10)')'NCRTERM <= MAXCRT  ', &      
#endif
#if KEY_CMAP==1
               NCRTERM <= MAXCRT,NCRTERM,MAXCRT                    
#endif
#endif          
       ENDIF
       CALL wrndie(-4,'<charmm_main.src>charmm','Bounds exceeded, try DIMENSION command.')
    ENDIF
70  FORMAT(//10X,'****  ERROR  **** A COUNTER VARIABLE HAS', &
         ' EXCEEDED ITS MAXIMUM ALLOWABLE VALUE.', &
         /16X,'EXECUTION WILL BE TERMINATED.', &
         /16X,'THE CURRENT COUNTER VARIABLES AND THEIR ', &
         'MAXIMUM ALLOWED VALUES ARE:', &
#ifndef KEY_RESIZE         
         /14X,'  NSEG = ',I8,' MAXSEG = ',I8,'  NATOM = ',I8, &
         '   MAXA = ',I8,'   NBOND = ',I8,'  MAXB = ',I8, &
         /14X,' NTHETA = ',I8,'   MAXT = ',I8,'   NPHI = ',I8, &
         '   MAXP = ',I8,'  NIMPHI = ',I8,' MAXIMP = ',I8, &
#if KEY_CMAP==1
         /14X,' NCRTERM = ',I8,' MAXCRT = ',I8,                 & 
#endif
         /14X,'    NNB = ',I8,'  MAXNB = ',I8,'   NDON = ',I8, &
         ' MAXPAD = ',I8,'    NACC = ',I8,' MAXPAD = ',I8, &
#endif         
         /14X,'   NRES = ',I8,'  MAXRES = ',I8,'  NATC = ',I8, &
         ' MAXATC = ',I8,'     NCB = ',I8,'  MAXCB = ',I8, &
         /14X,'    NCT = ',I8,'   MAXCT = ',I8,'   NCP = ',I8, &
         '  MAXCP = ',I8,'     NCI = ',I8,'  MAXCI = ',I8, &
         /14X,'     NCH = ',I8,'   MAXCH = ',I8,'   NCN = ',I8, &
         '  MAXCN = ',I8)
  end function check_for_exceed_bounds

END PROGRAM CHARMM
