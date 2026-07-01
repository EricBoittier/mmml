#if (KEY_STRINGM==1) /*  automatically protect all code */
! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
#if (KEY_STRINGM==1)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      SUBROUTINE sm_main(COMLYN,COMLEN)
      use string
      use sm0k_main_mod, only: sm0k_main
      use ftsm, only: ftsm_parse
      implicit none
!-----------------------------------------------
! calls string method parsers
!-----------------------------------------------
      character(len=*) :: comlyn
      integer :: comlen
!
! local variables
!
      character(len=8) :: keyword
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      character(len=len("SM_MAIN>") ),parameter::whoami="SM_MAIN>";!macro
!
      keyword=nexta8(comlyn,comlen)
! there are two parsers, 0K , CV , and FTSM
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (( keyword(1:4).eq.'ZERO'(1:4) )) then
        call sm0k_main(comlyn, comlen) ! SM0K parser
      elseif (( keyword(1:4).eq.'SM0K'(1:4) )) then
        call sm0k_main(comlyn, comlen) ! SM0K parser
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'COLV'(1:4) )) then
        call smcv(comlyn, comlen) ! SMCV parser
      elseif (( keyword(1:4).eq.'SMCV'(1:4) )) then
        call smcv(comlyn, comlen) ! SMCV parser
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'FTSM'(1:4) )) then
        call ftsm_parse(comlyn, comlen) ! FTSM parser
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'OPEN'(1:4) )) then
        call sm_open(comlyn, comlen, .true.) ! open a file on each replica
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      elseif (( keyword(1:4).eq.'CLOS'(1:4) )) then
        call sm_close(comlyn, comlen) ! close a file on each replica
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      else
            write(info(1),*)'UNRECOGNIZED SUBCOMMAND: ',keyword;call wrndie(0,whoami,trim(info(1)))
      endif
!
      end subroutine sm_main
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       SUBROUTINE SM_OPEN(COMLYN,COMLEN, loud)
!-----------------------------------------------------------------------
! Opens 1 file per replica process
!-----------------------------------------------------------------------
! much of this code is duplicated from the main charmm open routine (vopen)
       use sm_var, only: smcv_initialized
       use sm0k, only: sm0k_initialized
       use ftsm_var, only: ftsm_initialized
!
       use dimens_fcm ; use consta ; use stream ; use machio ; use exfunc ; use parallel ; use string ; use number ;
       use mpi
       use multicom_aux;
!
       implicit none
!
       character(len=*) :: COMLYN
       integer :: comlen
!********************** CHARMM *************************************************
       integer MXFILE, MXPREF, MXSUFF
       integer FLEN, L, UNUM, UNUM2, I
       LOGICAL ERR, QOPEN, QFORM, QWRITE, QENS
!
#if (KEY_UNIX==1) || (KEY_OS2==1)
       PARAMETER (MXFILE=128)
#else
       PARAMETER (MXFILE=256)
#endif
       integer FMLEN, IPT, STSLEN, FACCL
       PARAMETER (FMLEN=11,STSLEN=8,FACCL=7)
       CHARACTER*(MXFILE) FILEX, JUNKNM, FNAM2
       CHARACTER*(FMLEN) FORMT
       CHARACTER*(STSLEN) FSTAT
       CHARACTER*(FACCL) FACC
!
       integer :: oldiol
!********************** END CHARMM *********************************************



!
       logical :: loud
       integer :: ierror
       character(len=len("SM_OPEN>") ),parameter::whoami="SM_OPEN>";!macro
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
       if (.not.(smcv_initialized &
     & .or.sm0k_initialized &
     & .or.ftsm_initialized)) then
        if (ME_GLOBAL.eq.0) &
        call wrndie(0,whoami,trim(' STRING METHOD NOT INITIALIZED. NOTHING DONE.'))
        return
       endif
!
       if (ME_STRNG.ne.MPI_UNDEFINED) then ! only roots work
! unum
        UNUM=gtrmi(COMLYN, COMLEN, 'UNIT', -1)
        IF (UNUM.LT.0) THEN
         call wrndie(0,whoami,trim(' NO UNIT NUMBER SPECIFIED'))

         return
        elseif (unum.gt.size(ifreeu)) then
         write(info(1),*)' MAXIMUM UNIT NUMBER IS LIMITED TO SIZE(IFREEU)=',itoa(size(ifreeu)),'. ABORT.';call wrndie(0,whoami,trim(info(1)))
         return

        ENDIF
! filename
        call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FILEX, MXFILE, FLEN)
        IF (FLEN.LE.0) THEN
         call wrndie(0,whoami,trim('NO FILE NAME GIVEN'))
         return
        ENDIF

!********************** CHARMM *************************************************
! remove quotes
        IPT=0
        DO I=1,FLEN
          IF(FILEX(I:I).NE.'"') THEN
             IPT=IPT+1
             FILEX(IPT:IPT)=FILEX(I:I)
          ENDIF
        ENDDO
        FILEX = FILEX(1:IPT)
        FLEN=IPT
!
        IF(LOWER) CALL CNVTLC(FILEX,FLEN)
!
        fnam2=filex
        call expnam(filex, flen, err)
! modify iolev so that vinquire works
        oldiol=iolev
        iolev=1
!
! unit already open?
        CALL VINQRE('UNIT',JUNKNM,MXFILE,I,QOPEN,QFORM,QWRITE,UNUM)
        IF (QOPEN) THEN
         IF(WRNLEV.GE.2.AND.IOLEV.GT.0.and.loud) WRITE(OUTU,'(2A)') &
     & whoami//' Unit already open.',                            &
     & ' The old file will be closed first.'
         CLOSE(UNIT=UNUM)
         IFREEU(UNUM) = 0
        ENDIF
! file already in use
        CALL VINQRE('FILE',FILEX,I,I,QOPEN,QFORM,QWRITE,UNUM2)
        IF (QOPEN) THEN
         IF(WRNLEV.GE.2.AND.IOLEV.GT.0.and.loud) WRITE(OUTU,'(A,/,2A)') &
     & whoami//' ***** WARNING ***** another unit is already ',      &
     & '         assigned to the file -', &
     & ' it will be disconnected first.'
         CLOSE(UNIT=UNUM2)
         IFREEU(UNUM2) = 0
        ENDIF
!
!********************** END CHARMM *************************************************

! format
        IF (indxa(COMLYN, COMLEN, 'UNFO').GT.0) THEN
         FORMT='UNFORMATTED'
        ELSE IF (indxa(COMLYN, COMLEN, 'FILE').GT.0) THEN
         FORMT='UNFORMATTED'
        ELSE IF (indxa(COMLYN, COMLEN, 'FORM').GT.0) THEN
         FORMT='FORMATTED'
        ELSE IF (indxa(COMLYN, COMLEN, 'CARD').GT.0) THEN
         FORMT='FORMATTED'
        ELSE
 call wrndie(0,whoami,trim('NO FORMAT SPECIFIED, WILL USE "UNFORMATTED"'))
         FORMT='UNFORMATTED'
        ENDIF
!

!************************** CHARMM *************************************************
! status
        CALL GTRMWA(COMLYN, COMLEN, 'STAT', 4, FSTAT, STSLEN, L)
        IF (L.LE.0) THEN
            FSTAT = 'UNKNOWN'
        ENDIF
!********************** END CHARMM *************************************************

! access
        IF (indxa(COMLYN, COMLEN, 'APPE').GT.0) THEN
         FACC='APPEND'
        ELSE IF (indxa(COMLYN, COMLEN, 'READ').GT.0) THEN
         FACC='READ'
        ELSE IF (indxa(COMLYN, COMLEN, 'WRIT').GT.0) THEN
         FACC='WRITE'
        ELSE
          FACC='READ'
        ENDIF

!************************** CHARMM *************************************************
        flen=mxfile
        call trima(fnam2,flen)
        if (loud) then
         do i=1, SIZE_STRNG
          if (i-1.eq.ME_STRNG) then
          WRITE (outu, '(2A,I3,3A,I3,4A)') whoami,' REPLICA: ',ME_STRNG,&
     & ', FILE: ',fnam2(1:FLEN),', UNIT: ',UNUM,', FORMAT: ',FORMT, &
     & ', ACCESS: ', FACC
          endif
         enddo
        endif ! loud
! if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) !##MPI
! & call MPI_BARRIER(MPI_COMM_STRNG, IERROR) !##MPI
! open it
        IF (FACC.EQ.'APPEND') THEN
         OPEN(UNIT=UNUM,FILE=FILEX,FORM=FORMT,STATUS='OLD', &
     & ACCESS='SEQUENTIAL')
        ELSE IF (FACC.EQ.'READ') THEN
         OPEN(UNIT=UNUM,FILE=FILEX,FORM=FORMT,STATUS='OLD', &
     & ACCESS='SEQUENTIAL')
        ELSE IF (FACC.EQ.'WRITE') THEN
         OPEN(UNIT=UNUM,FILE=FILEX,FORM=FORMT,STATUS='UNKNOWN', &
     & ACCESS='SEQUENTIAL')
        END IF
! update ifreeu array
!
        INQUIRE(FILE=FILEX,OPENED=QOPEN,NUMBER=UNUM)
        IF (.NOT. QOPEN) THEN
          CALL WRNDIE(0, whoami, 'Could not open file')
        ELSE
!
! put appropriate code in IFREEU array to play nicely
! with rest of charmm; use +/8, 80
! +80 string read formatted
! +10 read formatted
! +8 string write/append formatted
! +1 write/append formatted
! -1 write/append unformatted
! -8 string write/append unformatted
! -10 read unformatted
! -80 string read unformatted
! i.e. ifreeu(unum)%8 tells whether we have a string file
         IF (FORMT.EQ.'FORMATTED') THEN
           IFREEU(UNUM)=8
         ELSE
           IFREEU(UNUM)=-8
         ENDIF
         IF (FACC.EQ.'READ') IFREEU(UNUM)=IFREEU(UNUM)*10
        ENDIF
! restore iolev
        iolev=oldiol
       endif ! ME_STRNG undefined
! propagate changes to slaves
!
       call PSNDC(COMLYN,1) ! updated command line (since slaves did not read)
#if (KEY_INTEGER8==0)
       call PSND4(COMLEN,1) 
#endif
#if (KEY_INTEGER8==1)
       call PSND8(COMLEN,1) 
#endif
! call PSND4(IFREE(UNUM),1) ! slaves should know nothing about the open file
!********************** END CHARMM *************************************************






!
       END SUBROUTINE SM_OPEN
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine sm_close(comlyn, comlen)
!
       use sm_var, only: smcv_initialized
       use sm0k, only: sm0k_initialized
       use ftsm_var, only: ftsm_initialized
!
       use dimens_fcm ; use consta ; use stream ; use machio ; use exfunc ; use parallel ; use string ; use number ;
       use mpi
       use multicom_aux;
!
       implicit none
!
       CHARACTER(len=*) :: COMLYN
       integer :: COMLEN
!
       integer :: UNUM
!
       character(len=len("SM_CLOSE>") ),parameter::whoami="SM_CLOSE>";!macro
!
       if (.not.(smcv_initialized &
     & .or.sm0k_initialized &
     & .or.ftsm_initialized)) then
        if (ME_GLOBAL.eq.0) &
        call wrndie(0,whoami,trim(' STRING METHOD NOT INITIALIZED. NOTHING DONE.'))
        return
       endif
!
       UNUM=gtrmi(COMLYN, COMLEN, 'UNIT', -1)
!
       if(ME_STRNG.ne.MPI_UNDEFINED) then ! only ensemble heads work
!

!**********************************************
! if ((MOD(IFREEU(UNUM),8).EQ.0).AND.
        if (IFREEU(UNUM).NE.0) THEN
         CLOSE(UNIT=UNUM)
         IFREEU(UNUM) = 0
        endif
!**********************************************



       endif
!
       end subroutine sm_close
!**********************************************************************

#endif
#endif /* automatically protect all code */
