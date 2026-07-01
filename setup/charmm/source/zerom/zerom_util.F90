module zutil
use chm_types
#if KEY_PARALLEL==1
use parallel,only: MYNODP,NUMNOD 
#endif
implicit none
real(chm_real) :: etime,ctime,etime2,ctime2

 contains

  subroutine rdzconf(FSTCNF,TMPDOF,TMPVAL,COMPLD, &
     COMPLV,SDOFRE,SVALUE,STMPDF,ALLDDF,LODOFL, &
     HIDOFL,LOCONF,HICONF,MSTDOF,MSTDFV,RDZUNI, &
     WRZUNI,SSNAME, &
     ALILST) !temporary
  !------------------------------------------
  !
  !  reads a zero module conformer file
  !
  use chm_kinds
  use zdata_mod
  use ztypes
  use zstruc,only : CSR,CSW
  use stream
  use energym
  use dimens_fcm
  use coord
  use nbndcc_util
  use parallel,only: MYNODP !temporary
  use, intrinsic :: iso_fortran_env
  use memory
  use zcs,only: cs_copy
  implicit none

  ! variables passed for allocation
!  INTEGER FSTCNF(*)  !list of first conformers of each ss
!  INTEGER TMPDOF(*) !temporary storage of conformer dofs
!  real(chm_real)  TMPVAL(*) !temp storage of dof values for a conformer 
!  INTEGER COMPLD(*) !the complete (first) comformer of a subspace
!  real(chm_real) COMPLV(*) !the dof values of complete conformer 
!  INTEGER SDOFRE(*) !temporary storage of sorted dofs 
!  real(chm_real) SVALUE(*)  !temp storage of sorted dof values
!  INTEGER STMPDF(*) !temp storage for sort of dofs
!  INTEGER ALLDDF(*) !list of all distinct dofs encountered in read
!  INTEGER LODOFL(*) !lo pointer into master dof list
!  INTEGER HIDOFL(*) !hi pointer into master dof list
!  INTEGER LOCONF(*) !lo pointer into conformer list
!  INTEGER HICONF(*) !hi pointer into conformer list
!  INTEGER MSTDOF(*) !master dof list
!  real(chm_real)  MSTDFV(*) !master dof value list
!  INTEGER SSNAME(*)
   INTEGER RDZUNI,WRZUNI  !unit numbers

   integer,dimension(:) :: FSTCNF,TMPDOF,COMPLD,SDOFRE,STMPDF,ALLDDF,LODOFL,HIDOFL, &
    LOCONF,HICONF,MSTDOF,SSNAME  
   real(chm_real),dimension(:) :: TMPVAL,COMPLV,SVALUE,MSTDFV 

  INTEGER,dimension(:) :: ALILST !temporary
  !
  ! local variables
  INTEGER RSUBSP,RCONFO,RDOFRE
  INTEGER NCURDF,NEWVAL
  integer :: NEWVALW 
  real(chm_real) RVALUE
  INTEGER DALINE,I,KK,J,II,III
  INTEGER LSTCNF
  INTEGER COUNTF,COUNTO
  LOGICAL STRDFC,DOLAST
  INTEGER PNDOFR,PPNDOR,NCMPDF
  INTEGER PRCONF,SAVESS
  INTEGER LOCCNT,DCONFO,DSUBSP
  LOGICAL QONE,FOUND
  INTEGER FSTCNF2
  CHARACTER (len=200) :: LINE
  real(chm_real) :: SAVENER,RENERGY
  !
  !
  !    Note FSTCNF is a local array
  do I = 1,NSUBME
     FSTCNF(I) = 0
  enddo
  DALINE = 0
  DSUBSP = 0
  DCONFO = -1  !local conformer count; NCONFO is global count of conformers (not reset on each call)
  PRCONF = 0
  NALLDF = 0
  STRDFC = .FALSE.
  FSTCNF2 = 0
  SAVENER = -1.0D0   !initial value
  SAVESS = -1
  !      IF (QONE) THEN
  !      OPEN (UNIT =1, FILE='dataa',FORM = 'FORMATTED',
  !     & STATUS = 'OLD')
  !      OPEN (UNIT=WRZUNI, FILE='echodata',FORM = 'FORMATTED',
  !     & STATUS = 'NEW')
  !      else
  !      OPEN (UNIT =3, FILE='datab',FORM = 'FORMATTED',
  !     & STATUS = 'OLD')
  !      ENDIF
  DOLAST = .FALSE.
  read_loop: do while (.NOT.DOLAST)
     read(UNIT=RDZUNI,FMT='(A)',END=11) LINE
     if(len_trim(LINE) == 0) then
#if KEY_PARALLEL==1
      if(MYNODP.eq.1) then 
#endif
       write(6,*) 'blank line in conformer file'
#if KEY_PARALLEL==1
      endif 
#endif
!      WRITE(6,*) 'cycling read_loop'
      cycle read_loop
     endif 
     !       if(QONE) THEN
     ! orig        READ(UNIT=RDZUNI,FMT=10,END=11) RSUBSP,RCONFO,RDOFRE,RVALUE
!     READ(UNIT=RDZUNI,FMT=*,END=11) RSUBSP,RCONFO,RDOFRE,RVALUE
     READ(LINE,FMT=*) RSUBSP,RCONFO,RDOFRE,RVALUE,RENERGY
     !     & ' RDOFRE ',RDOFRE,' RVALUE ',RVALUE 
     !       else
     !        READ(UNIT=3,FMT=10,END=11) RSUBSP,RCONFO,RDOFRE,RVALUE
     !       ENDIF
5    CONTINUE
     DALINE = DALINE+1
     !       WRITE(UNIT=2,FMT=10) RSUBSP,RCONFO,RDOFRE,RVALUE
10   FORMAT(I14,I14,I14,F14.7,F14.7)
     if(RDOFRE.EQ.0) THEN
#if KEY_PARALLEL==1
       if(MYNODP.eq.1) then  
#endif
        WRITE(6,'(2X,A26)') 'MISSING DEGREE OF FREEDOM'
        WRITE(6,'(2X,A4,1X,I14,1X,A13)') 'LINE',DALINE, &
             'OF INPUT FILE'
        WRITE(6,*) 'RSUBSP ',RSUBSP,' RCONFO ',RCONFO,' RDOFRE ',RDOFRE,' RVALUE ',RVALUE
#if KEY_PARALLEL==1
       endif 
#endif
        GOTO 20
     ENDIF
     ! ******************************************************************
     ! section for changing conformers: *********************************
     ! ******************************************************************
     if(((RCONFO.NE.PRCONF).AND.(RCONFO.NE.0)).OR.DOLAST) THEN
        !      if changing conformers
        !         WRITE(6,*) 'CHANGING CONFORMER SECTION '
        !         if(DOLAST) 
        !     & WRITE(6,*) 'WE ARE DOING THE LAST ONE'
        IF((RCONFO.NE.(PRCONF+1)).AND.(.NOT.DOLAST).AND. &
             (.NOT.QMISSG)) THEN
           WRITE(6,'(2X,A30)') 'CONFORMER NUMBERS OUT OF ORDER'
           !         WRITE(6,*) 'RCONFO ',RCONFO,' PRCONF ',PRCONF
           WRITE(6,'(2X,A4,1X,I14,1X,A13)') 'LINE',DALINE, &
                'OF INPUT FILE'
           CALL WRNDIE(-5,'<RDZCONF>', &
                ' CONFORMERS OUT OF ORDER OR MISSING--try ZREAD MISSing')
        ENDIF
        !     if we are not at conformer # 1 of the whole list,
        !     sort and store previous conformer
        IF(PRCONF.NE.0) THEN  
           PNDOFR = NCURDF  !store previous number of dof
           IF(PNDOFR.EQ.1) THEN
              SVALUE(1) = TMPVAL(1)
              SDOFRE(1) = TMPDOF(1)
           else 
              CALL INDIXX(PNDOFR,TMPDOF,STMPDF)
              do I = 1,PNDOFR
                 KK = STMPDF(I)
                 !           WRITE(6,*) 'SORTING ARRAY'
                 SVALUE(I) = TMPVAL(KK)
                 SDOFRE(I) = TMPDOF(KK)
!                 WRITE(6,*) 'count ',I,' SDOFRE(I) ',SDOFRE(I),' SVALUE ',SVALUE(I)
                 IF(I.GE.2) THEN
                    IF(SDOFRE(I).EQ.SDOFRE(I-1)) THEN
                       !     test for duplicates
                       WRITE(6,'(2X,A20)') 'DUPLICATE DOF BEFORE'
                       WRITE(6,'(2X,A4,1X,I14,1X,A13)') 'LINE',DALINE, &
                            'OF INPUT FILE'
                       GOTO 20
                    ENDIF
                 ENDIF
              enddo
           ENDIF
        ENDIF
        ! if the previous conformer was not the first of a subspace, compare it to 
        ! the complete (first) conformer
        !  First compare it to the complete conformer, check for extraneous dofs
        !  If dofs match, save dof and value and update complete conformer values
        NEWVAL = 0
        NEWVALW = 0 
!        WRITE(6,*) 'PRCONF ',PRCONF,' FSTCNF2 ',FSTCNF2
        if (PRCONF.GT.FSTCNF2) THEN
           COUNTF = 1
           COUNTO = 1
!           do I =1,NCMPDF
!                         WRITE(6,*) 'TEST2 I ',I,' COMPLD ',COMPLD(I),  &
!                   ' SDOFRE ',SDOFRE(I)
!           enddo
           !          WRITE(6,*) ' GOT TO SIX.2'
           do WHILE ((COUNTF.LE.NCMPDF).AND.(COUNTO.LE.PNDOFR))
              !     loop over dofs of previous conformer
              IF(COMPLD(COUNTF).EQ.SDOFRE(COUNTO)) THEN !the dof's match
                 !              WRITE(6,*) 'COUNTF ',COUNTF,' COMPLD ',COMPLD(COUNTF),
                 !     & ' SDOFRE ',SDOFRE(COUNTO)
                 !              WRITE(6,*) 'COUNTF ',COUNTF,' COMPLV ',COMPLV(COUNTF),
                 !     & ' SVALUE ',SVALUE(COUNTO)
!need this for random access to conformer info (uncompressed data)
                 NMASTRW = NMASTRW + 1
                 CSR%MSTDF(NMASTRW) = SDOFRE(COUNTO)
                 CSR%MSTDV(NMASTRW) = SVALUE(COUNTO)
                 NEWVALW = NEWVALW + 1
                 IF(NMASTRW.GT.NVALME) THEN  !should we augment NVALME if parallel
                       WRITE(6,*)  &
                            'LENGTH OF INPUT CONFORMER FILE ',NMASTRW, &
                            ' EXCEEDS ALLOTTED MEMORY ',NVALME
                       CALL WRNDIE(-5,'<RDZCONF>', &
                            'CONFORMER FILE TOO LONG. INCREASE ZMEMOry NVALue')
                 ENDIF
!                 WRITE(6,*) '>>TEST2MYNODP ',MYNODP,' incr NMASTRW to ',NMASTRW,' CSR%MSTDF is ',CSR%MSTDF(NMASTRW)
!
                 if(COMPLV(COUNTF).NE.SVALUE(COUNTO)) THEN
                    !                WRITE(6,*) ' GOT TO SIX.5'
                    NMASTR = NMASTR + 1
                    NEWVAL = NEWVAL + 1
                    MSTDOF(NMASTR) = SDOFRE(COUNTO)
                    MSTDFV(NMASTR) = SVALUE(COUNTO)
!                    if(MYNODP.eq.2) then
!                 WRITE(6,*) '>>TEST2MYNODP ',MYNODP,' incr NMASTR to ',NMASTR,' MSTDOF is ',MSTDOF(NMASTR)
!                 endif

                    if(NMASTR.GT.NVALME) THEN
                       WRITE(6,*)  &
                            'LENGTH OF INPUT CONFORMER FILE ',NMASTR, &
                            ' EXCEEDS ALLOTTED MEMORY ',NVALME  
                       CALL WRNDIE(-5,'<RDZCONF>', &
                            'CONFORMER FILE TOO LONG. INCREASE ZMEMOry NVALue')
                    endif
                    if(WRZUNI.GT.0) THEN
                       if(.NOT.QSSDIS) THEN
                          WRITE(WRZUNI,10) NSUBSP,NCONFO+1,MSTDOF(NMASTR), &
                               MSTDFV(NMASTR),SAVENER
!                          WRITE(MYNODP+400,10) NSUBSP,NCONFO+1,MSTDOF(NMASTR), &
!                               MSTDFV(NMASTR),SAVENER  !temporary
                       else
                          WRITE(WRZUNI,10) RSUBSP,NCONFO+1,MSTDOF(NMASTR), &
                               MSTDFV(NMASTR),SAVENER
                       endif
                    endif  !if writing out conformers
                    !                 WRITE(6,*) ' GOT TO SIX.6'
                    COMPLV(COUNTF) = SVALUE(COUNTO)
                    !     update the complete conformer
                    !         WRITE(6,*) 'NOTEQUAL STORE! ',NMASTR,' ODOF ',MSTDOF(NMASTR)
                    !      WRITE(6,*)  ' OVAL ',MSTDFV(NMASTR),COMPLV(COUNTF),SVALUE(COUNTO)
                 endif
                 COUNTO = COUNTO + 1
                 COUNTF = COUNTF + 1
              else if (COMPLD(COUNTF).GT.SDOFRE(COUNTO)) THEN
                 WRITE(6,'(2X,A27)') 'FIRST CONFORMER IN SUBSPACE'
                 WRITE(6,'(I14,1X,A9,1X,I14)') RSUBSP,'LACKS DOF', &
                      SDOFRE(COUNTO)
                 WRITE(6,'(2X,A12,1X,I14,1X,A13)') 'BEFORE LINE',DALINE, &
                      'OF INPUT FILE'
                 GOTO 20
              else  !the dof in the other conformer is larger
                 IF (COUNTF.EQ.NCMPDF) THEN
                    WRITE(6,'(2X,A27)') 'FIRST CONFORMER IN SUBSPACE'
                    WRITE(6,'(I14,1X,A9,1X,I14)') RSUBSP,'LACKS DOF', &
                         SDOFRE(COUNTO)
                    WRITE(6,'(2X,A12,1X,I14,1X,A13)') 'BEFORE LINE',DALINE, &
                         'OF INPUT FILE'
                    GOTO 20
                 ENDIF
                 COUNTF = COUNTF + 1
              endif ! if dofs match, if values match, etc.
           enddo ! loop over degrees of freedom from prev conf
           !          WRITE(6,*) ' GOT TO SIX.10'
           if (NEWVAL.NE.0) THEN
              LOCCNT = LOCCNT + 1
              DCONFO = DCONFO + 1
              NCONFO = NCONFO + 1
              HIDOFL(NCONFO) = NMASTR
              LODOFL(NCONFO) = NMASTR - NEWVAL + 1
! uncompressed
              CSR%HIDOF(NCONFO) = NMASTRW
!              write(6,*) 'NMASTRW ',NMASTRW,' NEWVALW ',NEWVALW
          
              CSR%LODOF(NCONFO) = NMASTRW - NEWVALW + 1
              CSR%ENERG(NCONFO) = SAVENER

!              WRITE(6,*) '>>TEST2MYNODP CSR%LODOF ', CSR%LODOF(NCONFO),' NEWVALW ',NEWVALW,' NMASTRW ', &
!                  NMASTRW
! 
              IF (NCONFO.GT.NCNFME) THEN
                 CALL WRNDIE(-5,'<RDZCONF>', &
                      'CONFORMERS EXCEED ALLOCATED MEMORY (ZMEM NCON)')
              ENDIF
              !            if(.NOT.DOLAST) 
              !     &     WRITE(6,*) 'INCREMENTING DCONFO AT A TO ',DCONFO
           else
              WRITE(6,'(2X,A30)')  &
                   'EMPTY OR REPEAT CONFORMER BEFORE'
              WRITE(6,'(2X,A4,1X,I14,1X,A23,I14)') 'LINE',DALINE, &
                   'OF INPUT FILE, SUBSPACE',RSUBSP
              WRITE(6,'(2X,A15,1X,I14,1X,A11)') 'CONFORMER ',PRCONF, &
                   'DISREGARDED'
           endif ! if conformer is empty or not
           !
        else if (PRCONF.NE.0) THEN !if previous conformer was the first
           ! else if the previous conformer was number 1 of subspace, store
           do I=1,NCURDF
              ! store this complete (first) conformer
              COMPLV(I) = SVALUE(I)
              COMPLD(I) = SDOFRE(I)
              NMASTR = NMASTR + 1
              NEWVAL = NEWVAL + 1
              MSTDOF(NMASTR) = SDOFRE(I)
              MSTDFV(NMASTR) = SVALUE(I)
!      WRITE(6,*) '>>TEST2MYNODP ',MYNODP,' incr NMASTR late to ',NMASTR,' MSTDOF is ',MSTDOF(NMASTR)
!uncompressed data
              NMASTRW = NMASTRW + 1
              NEWVALW = NEWVALW + 1
              CSR%MSTDF(NMASTRW) = SDOFRE(I)
              CSR%MSTDV(NMASTRW) = SVALUE(I)
!   WRITE(6,*) '>>TEST2MYNODP ',MYNODP,' incr NMASTRW late to ',NMASTRW,' CSR%MSTDF is ',CSR%MSTDF(NMASTR)
!
              if(NMASTR.GT.NVALME) THEN
                 WRITE(6,*) &
                      'LENGTH OF INPUT CONFORMER FILE ',NMASTR, &
                      ' EXCEEDS ALLOTTED MEMORY ',NVALME
                 CALL WRNDIE(-5,'<RDZCONF>', &
                      'CONFORMER FILE TOO LONG. INCREASE ZMEMOry NVALue')
              endif
              !
              if(WRZUNI.GT.0) THEN !if writing out conformers
                 if(.NOT.QSSDIS) THEN
                    WRITE(WRZUNI,10) NSUBSP,NCONFO+1,MSTDOF(NMASTR), &
                         MSTDFV(NMASTR),SAVENER
                 else
                    WRITE(WRZUNI,10) RSUBSP,NCONFO+1,MSTDOF(NMASTR), &
                         MSTDFV(NMASTR),SAVENER
                 ENDIF
              endif
              ! check against previous dofs on definition list
              do J = 1,NALLDF
                 if((ALLDDF(J).EQ.SDOFRE(I)).AND. &
                      (.NOT.QREDUN)) THEN
                    WRITE(6,'(2X,A27)') 'REDUNDANT DEGREE OF FREEDOM'
                    WRITE(6,'(2X,A7)') 'IN CONFORMER LIST'
                    WRITE(6,'(2X,A8,1X,I14,1X,A3,I10)')  &
                         'SUBSPACE',RSUBSP,'DOF',SDOFRE(I)
                    WRITE(6,'(2X,A4,1X,I14,1X,A13)') 'LINE',DALINE, &
                         'OF INPUT FILE'
                    CALL WRNDIE(-5,'<RDZCONF>', &
                         'SAME DOF IN MULT SUBSPs--ALIASES? Try ZREAD REDUndancy')
                 ENDIF
              enddo
              !  add dofs to master list of dofs
              NALLDF = NALLDF + 1
              if(NALLDF.GT.DDEFME) THEN
                 WRITE(OUTU,'(A)')  &
                      '   NUMBER OF DOFs IN CONFORMER FILE (INCLUDING REDUNDANCIES)'
                 WRITE(OUTU,'(A)')  &
                      '   EXCEED ALLOCATED MEMORY (ZMEM NDOF)'
                 CALL WRNDIE(-5,'<RDZCONF>', &
                      'DOFs EXCEED ALLOCATED MEMORY')
              ENDIF
              ALLDDF(NALLDF)=SDOFRE(I)
              if(ALLDDF(NALLDF).GT.NTHDDF) THEN
                 WRITE(6,*) 'NTHDDF ',NTHDDF,' ALLDDF ',ALLDDF(NALLDF)
                 WRITE(OUTU,'(A18,I8,A18)') &
                      'DEGREE OF FREEDOM ',ALLDDF(NALLDF),' HAS NO DEFINITION'
              ENDIF
           enddo !loop over dofs from prev conformer
           IF (NEWVAL.GT.0) THEN
              DCONFO = DCONFO + 1
              NCONFO = NCONFO + 1
              HIDOFL(NCONFO) = NMASTR
              LODOFL(NCONFO) = NMASTR - NEWVAL + 1
! uncompressed
              CSR%HIDOF(NCONFO) = NMASTRW
              CSR%LODOF(NCONFO) = NMASTRW - NEWVALW + 1
              CSR%ENERG(NCONFO) = SAVENER
!              WRITE(6,*) '>>TEST2MYNODP CSR%LODOF ', CSR%LODOF(NCONFO),' NEWVALW ',NEWVALW,' NMASTRW ', &
!                  NMASTRW
!
              IF (NCONFO.GT.NCNFME) THEN
                 CALL WRNDIE(-5,'<RDZCONF>', &
                      'CONFORMERS EXCEED ALLOCATED MEMORY (ZMEM NCON)')
              ENDIF
              !            if(.NOT.DOLAST)
              !     &     WRITE(6,*) 'INCREMENTING DCONFO AT B TO ',DCONFO
              LOCCNT = LOCCNT + 1
              NCMPDF = NCURDF !store number of dofs in complete conf
           else
              WRITE(6,'(2X,A30)') 'EMPTY CONFORMER BEFORE'
              WRITE(6,'(2X,A4,1X,I14,1X,A23,I14)') 'LINE',DALINE, &
                   'OF INPUT FILE, SUBSPACE',RSUBSP
              WRITE(6,'(2X,A46)')  &
                   'FIRST CONFORMER OF A SUBSPACE CANNOT BE EMPTY'
              GOTO 20
           ENDIF !if the conformer is empty or not
        else if ((PRCONF.EQ.0).AND.(.NOT.DOLAST)) THEN
           !     first conformer in file
           DCONFO = DCONFO + 1
           !           WRITE(6,*) 'INCREMENTING DCONFO AT C TO ',DCONFO
        ENDIF !if previous conf was first of subspace or not
        NCURDF = 0 !reset local dof counter
        !         WRITE(6,*) 'DCONFO ',DCONFO,' RCONFO ',RCONFO
        if((RCONFO.NE.(PRCONF+1)).AND.(.NOT.DOLAST).AND. &
             (.NOT.QMISSG)) THEN
           WRITE(6,'(2X,A30)') 'CONFORMER NUMBERS OUT OF ORDER'
           WRITE(6,'(2X,A4,1X,I14,1X,A13)') 'LINE',DALINE, &
                'OF INPUT FILE'
           CALL WRNDIE(-5,'<RDZCONF>', &
                ' CONFORMERS OUT OF ORDER OR MISSING--try ZREAD MISSing')
        ENDIF
        !
        ! ****************************************************************************
        ! end of section for changing conformers **************************************
        ! ****************************************************************************
     else !if not changing conformers
        IF ((RSUBSP.NE.SAVESS).AND.(RSUBSP.NE.0).AND. &
             (.NOT.DOLAST)) THEN 
           WRITE(6,'(2X,A32)') 'ERROR: SAME CONFORMER NUMBER FOR'
           WRITE(6,'(2X,A19)') 'DIFFERENT SUBSPACES'
           WRITE(6,'(2X,A4,1X,I14,1X,A13)') 'LINE',DALINE, &
                'OF INPUT FILE'
           GOTO 20
        ENDIF
        if((RENERGY.ne.SAVENER).and.(NCURDF.gt.1)) then
#if KEY_PARALLEL==1
         if(MYNODP.eq.1) then 
#endif
          WRITE(6,*) 'WARNING: energies differ within same conformer'
#if KEY_PARALLEL==1
         endif 
#endif
        endif
     ENDIF !whether changing conformers or not
     ! ********* process subspaces **********************************************
     if(((RSUBSP.NE.0).AND.(RSUBSP.NE.SAVESS)).OR. &
          (DOLAST)) THEN !if new subspace
        if(DSUBSP.NE.0) THEN
           if(QSSDIS) THEN
              HICONF(SAVESS) = NCONFO
              LOCONF(SAVESS) = NCONFO - LOCCNT + 1
              !       WRITE(6,*) 'QSSDIS ',QSSDIS,' SAVESS ',SAVESS,
              !     & ' LOCF ',LOCONF(SAVESS),' HICF ',HICONF(SAVESS)
           else !no disorder in subspace names
              HICONF(NSUBSP) = NCONFO
              LOCONF(NSUBSP) = NCONFO - LOCCNT + 1
              !       WRITE(6,*) 'QSSDIS ',QSSDIS,' NSUBSP ',NSUBSP,
              !     & ' LOCF ',LOCONF(NSUBSP),' HICF ',HICONF(NSUBSP)
           ENDIF
        ENDIF
        if(.NOT.DOLAST) THEN 
           STRDFC = .FALSE.
           if(RSUBSP.LT.0) THEN
              WRITE(6,'(2X,A24)') 'BAD SUBSPACE NUMBER AT'
              WRITE(6,'(2X,A4,1X,I14,1X,A13)') 'LINE',DALINE, &
                   'OF INPUT FILE'
              CALL WRNDIE(-5,'<RDZCONF>', &
                   'BAD SUBSPACE NUMBER')
           ENDIF !RSUBSP not less than 0
           if(RSUBSP.GT.NSUBDE) THEN
              WRITE(6,'(4X,A9,I8,1X,A25)') 'SUBSPACE ',RSUBSP, &
                   'STRUCTURE IS NOT DEFINED '
              CALL WRNDIE(-5,'<RDZCONF>', &
                   'BAD SUBSPACE NUMBER')
           ENDIF
           DSUBSP = DSUBSP + 1
           NSUBSP = NSUBSP + 1
           ! check to see if subspace already read in
           FOUND = .FALSE.
           if(QSSDIS) THEN
              III = 1
              do WHILE((.NOT.FOUND).AND.(III.LT.NSUBSP)) 
                 !           WRITE(6,*) 'RSUBSP ',RSUBSP,' TESTED SS ',
                 !     & SSNAME(III)
                 if(SSNAME(III).EQ.RSUBSP) THEN
                    FOUND = .TRUE.
                    WRITE(6,*) 'SUBSPACE ',RSUBSP, &
                         ' READ IN X 2 WITH DISORDER FLAG SSDI ON'
                    WRITE(6,*) 'ORDER OR RENUMBER SUBSPACES'
                    CALL WRNDIE(-5,'<RDZCONF>', &
                         'SUBSPACES CANNOT BE READ IN TWICE IF SSDI')
                 ENDIF
                 III = III + 1
              enddo
           ENDIF
           if(.NOT.FOUND) SSNAME(NSUBSP) = RSUBSP
           if(NSUBSP.GT.NSUBME) THEN
              CALL WRNDIE(-5,'<RDZCONF>', &
                   'SUBSPACES EXCEED ALLOCATED MEMORY (ZMEM NSUB)')
           ENDIF
           LOCCNT = 0
           if((RSUBSP.NE.DSUBSP).AND.(.NOT.QSSDIS)) THEN
              WRITE(6,'(2X,A29)') 'SUBSPACE NUMBERS OUT OF ORDER'
              WRITE(6,'(2X,A4,1X,I14,1X,A13)') 'LINE',DALINE, &
                   'OF INPUT FILE'
              CALL WRNDIE(-5,'<RDZCONF>', &
                   'SUBSP NUMBERS OUT OF ORDER--try ZREAd SSDISorder')
           ENDIF ! if rsubsp .ne. nsubsp
           if(RCONFO.EQ.0) THEN 
              WRITE(6,'(2X,A37,1X,I14)') &
                   'NO CONFORMER DESIGNATION FOR SUBSPACE',RSUBSP
              WRITE(6,'(2X,A4,1X,I14,1X,A13)') 'LINE',DALINE, &
                   'OF INPUT FILE'
              GOTO 20 
           else if (.NOT.STRDFC) THEN
              FSTCNF2=RCONFO
              STRDFC=.TRUE.
           ENDIF
           SAVESS = RSUBSP
        ENDIF ! if not dolast
     ENDIF  !if new subspace
     NCURDF = NCURDF + 1
     TMPDOF(NCURDF)=RDOFRE
     TMPVAL(NCURDF)=RVALUE
     SAVENER = RENERGY
!     WRITE(6,*) 'NCURDF ',NCURDF,' TMPDOF(NCURDF) ',TMPDOF(NCURDF),' SAVENER ',SAVENER
     if(RCONFO.NE.0) PRCONF = RCONFO
     !       WRITE(6,*) 'ASSIGNING TMPs'
     !       WRITE(6,*) 'NCURDF ',NCURDF,' TDOF ',TMPDOF(NCURDF),
     !     &  ' TVAL ',TMPVAL(NCURDF) 
  end do read_loop    !loop over datafile    
11 if(.NOT.DOLAST) THEN
     DOLAST = .TRUE.
     GOTO 5
  ENDIF
20 CONTINUE
  !      NSUBSP = DSUBSP
  do I = 1,NSUBSP
     WRITE(6,'(4X,A8,1X,I10,1X,A24,1X,I10)') 'READ IN ', &
          HICONF(I) - LOCONF(I) + 1,'CONFORMERS FOR SUBSPACE ',I
  enddo
!        do I = 1,NCONFO
!         WRITE(6,*) 'CONFORMER ',I,' LODOFL ',LODOFL(I),  &
!        ' DOF ',MSTDOF(LODOFL(I)),' VAL ',MSTDFV(LODOFL(I))
!         WRITE(6,*) 'CONFORMER ',I,' HIDOFL ',HIDOFL(I),  &
!        ' DOF ',MSTDOF(HIDOFL(I)),' VAL ',MSTDFV(HIDOFL(I))
!         WRITE(6,*) 'CONFORMER ',I,' CSR%LODOF ',CSR%LODOF(I),  &
!        ' DOF ',CSR%MSTDF(CSR%LODOF(I)),' VAL ',CSR%MSTDV(CSR%LODOF(I))
!         WRITE(6,*) 'CONFORMER ',I,' CSR%HIDOF ',CSR%HIDOF(I),  &
!        ' DOF ',CSR%MSTDF(CSR%HIDOF(I)),' VAL ',CSR%MSTDV(CSR%HIDOF(I))
!        enddo
  !      if(QONE) THEN
  CLOSE(RDZUNI)
  !      else
  !         CLOSE(UNIT=3)
  !      ENDIF 
  !
  !      do II = 1,5
  !       WRITE(6,*) 'LINE END',II,' MSTDOF ',MSTDOF(II),
  !     & ' NMASTR ',NMASTR
  !      enddo
  !      WRITE(6,*) 'AFTER READING CONFORMERS'
  !      do KK = 1,NSUBDE
  !        WRITE(6,*) 'SUBSPACE ',KK,' REFERENCE ',ALILST(KK)
  !      enddo
!   WRITE(6,*) 'MYNODP ',MYNODP,' END OF RDZCONF CSR%MSTDF(1) is ',CSR%MSTDF(1)

!make copies of LOCONF, HICONF conformers just for consistency
    CSR%LOCNF(1:NSUBSP) = LOCONF(1:NSUBSP)
    CSR%HICNF(1:NSUBSP) = HICONF(1:NSUBSP)

!    write(6,*) 'in rdzconf CSR%LODOF(1) is ',CSR%LODOF(1)

!    do II = 1,NCONFO
!     write(6,*) II,' >>>CSR%ENERG ',CSR%ENERG(II)
!    enddo
!make working copies of the list

  RETURN
end subroutine rdzconf
! ------------------------------------------------------
! ------------------------------------------------------

 subroutine random_array(CHILD,NP,NC,PARENT,INSEEDAR,OUTSEEDAR,IFAC)
! randomly selects NC integers out of NP integers and returns result in CHILD array.  
! If PARENT array is specified, pulls integers out of the PARENT array 
 integer,dimension(:),intent(out) :: CHILD  !child array, which results from selection
 integer,intent(in) :: NP,NC   !number of elements in PARENT and CHILD arrays
 integer,dimension(:),intent(in),optional :: INSEEDAR  !if present, used as seed array
 integer,dimension(:),intent(out),optional :: OUTSEEDAR !if present, values of current seed array
 real*8,intent(in),optional :: IFAC !if present, fraction of PARENT size above which
      !algorithm will select NP - NC elements (instead of NC) and then result the inverse
 integer,dimension(:),intent(in),optional :: PARENT  !parent array from which values are taken, if present
! otherwise, values taken from positive integers: 1,2,3,...NP

 real(chm_real) :: OUTREAL,RAND,FAC
 integer :: II,NOK,JJ,MAXTRIAL,SEL,NTARG
 logical :: QREV
 integer :: FLAG
 integer,dimension(:),allocatable :: SELFLAG

 MAXTRIAL = 10*NP

 if(present(INSEEDAR)) then
   call random_seed(put=INSEEDAR)
 endif

 if(allocated(SELFLAG)) deallocate(SELFLAG)
 allocate(SELFLAG(NP))
 SELFLAG = 0
 FAC = 0.6666666666666666666D0
 if(present(IFAC)) FAC = IFAC

 if(NC.GT.REAL(NP)*FAC) then
  QREV = .true.
  NTARG = NP - NC
 else
  QREV = .false.
  NTARG = NC
 endif
! WRITE(6,*) 'NC ',NC,' NP ',NP,' expression ',REAL(NP)*FAC,' FAC ',FAC,' QREV ',QREV

 NOK = 0
 do II = 1,MAXTRIAL
  call random_number(harvest=RAND)
  SEL = int(RAND*NP) + 1
  IF(SELFLAG(SEL).EQ.0) then
   NOK = NOK + 1
   SELFLAG(SEL) = 1
  endif
  if(NOK.eq.NTARG) exit
 enddo

 NOK = 0
! WRITE(6,*) 'QREV is ',QREV
 do II = 1,NP
  FLAG = SELFLAG(II)
  if(((FLAG.eq.1).and.(.not.QREV)) .or. ((FLAG.eq.0).and.(QREV))) then
    NOK = NOK + 1
    if(present(PARENT)) then
     CHILD(NOK) = PARENT(II)
    else
     CHILD(NOK) = II
    endif
  endif
 enddo

 if(present(OUTSEEDAR)) then
  call random_seed(get=OUTSEEDAR)
 endif

 end subroutine random_array
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------

 subroutine set_randseed(ISEED)
 implicit none

 integer,intent(in) :: ISEED
 integer(chm_int8) :: INT8,INTG,IMOD
 integer :: SDSZ
 integer,allocatable,dimension(:) :: SEED
 integer :: II,JJ
! 
 call random_seed(size = SDSZ)
 allocate(SEED(SDSZ))

 INTG = mod(ISEED,2147483647) !limits size 
!
 IMOD=104729 ! VO to avoid type mismatch in mod() on 32-bit systems
!
 do II = 1,SDSZ
  INT8 = II*II
  SEED(II) = mod(INT8*INTG,IMOD)
  if(SEED(II).LE.0) then
   CALL WRNDIE(-5,'<SET_RANDSEED>', &
    'BAD SEED ')
  endif
 enddo
 call random_seed(put=SEED)  !sets the seed for fortran rand num gen

 end subroutine set_randseed
!--------------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------------
 subroutine get_randseed(SEEDAR)
 implicit none
 integer,dimension(:),allocatable,intent(out) :: SEEDAR
 integer :: SDSZ
 
 call random_seed(size = SDSZ)
 allocate(SEEDAR(SDSZ))

 call random_seed(get=SEEDAR)

 end subroutine get_randseed
!--------------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------------
 subroutine print_randseed
 implicit none
!locals
 integer,dimension(:),allocatable :: SEEDAR
 integer :: SDSZ

 call random_seed(size = SDSZ)
 allocate(SEEDAR(SDSZ))

 call random_seed(get=SEEDAR)

#if KEY_PARALLEL==1
 if(MYNODP.eq.1) then 
#endif
  WRITE(6,*) 'CURRENT SEED: ',SEEDAR
#if KEY_PARALLEL==1
 endif 
#endif
 end subroutine print_randseed
!--------------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------------

 subroutine trim_conf(MYECUT,NSSTAG,CONFMR,MSTDOF,MSTDFV,MYPOTEN,RMST1,OUTLNCNT,BNDWDTH)
#if KEY_PARALLEL==1
 use parallel,only: MYNODP 
#endif
 use memory
 use ztypes,only: CONFSET
 implicit none

 real(chm_real),intent(in) :: MYECUT
 real(chm_real),intent(in),optional :: BNDWDTH !width of band above minimum
 integer,dimension(:),allocatable,intent(inout) :: NSSTAG,CONFMR,MSTDOF  
 real(chm_real),dimension(:),allocatable,intent(inout) :: MSTDFV,MYPOTEN
 real(chm_real),dimension(:),allocatable,intent(inout),optional :: RMST1
 integer,intent(out) :: OUTLNCNT

!locals
 integer,dimension(:),allocatable :: NSSTAG_TMP,CONFMR_TMP,MSTDOF_TMP  
 real(chm_real),dimension(:),allocatable :: MSTDFV_TMP,MYPOTEN_TMP,RMST1_TMP
 integer :: NCOUNT,NCOUNT2,II,LASTCNF,THISCNF
 real(chm_real) :: ENER,EMIN,MYVCUT
 type(confset) :: CSTMP

!************
!    WRITE(6,*) 'MYNODP ',MYNODP,' LOCMIN ',LOCMIN(MYNODP),' GLOBMIN ',GLOBMIN
!    call parstoperr('<ZMERGE>','printed out values of GLOBAL MIN')

    OUTLNCNT = size(NSSTAG)
    if((OUTLNCNT.ne.size(CONFMR)).or.(OUTLNCNT.ne.size(MSTDOF)).or.(OUTLNCNT.ne.size(MSTDFV)).or. &
       (OUTLNCNT.ne.size(MYPOTEN))) then
#if KEY_PARALLEL==1
      if(MYNODP.eq.1) then 
#endif
       WRITE(6,*) 'WARNING: <trimconf> dataset columns with unequal lengths'
#if KEY_PARALLEL==1
      endif  
#endif
    endif

    if(allocated(NSSTAG_TMP)) call chmdealloc('zerom2.src','ZMERGE','NSSTAG_TMP',OUTLNCNT,intg = NSSTAG_TMP)
    call chmalloc('zerom2.src','ZMERGE','NSSTAG_TMP',OUTLNCNT,intg=NSSTAG_TMP)
    
    if(allocated(CONFMR_TMP)) call chmdealloc('zerom2.src','ZMERGE','CONFMR_TMP',OUTLNCNT,intg = CONFMR_TMP)
    call chmalloc('zerom2.src','ZMERGE','CONFMR_TMP',OUTLNCNT,intg=CONFMR_TMP)
   
    if(allocated(MSTDOF_TMP)) call chmdealloc('zerom2.src','ZMERGE','MSTDOF_TMP',OUTLNCNT,intg = MSTDOF_TMP)
    call chmalloc('zerom2.src','ZMERGE','MSTDOF_TMP',OUTLNCNT,intg=MSTDOF_TMP)

    if(allocated(MSTDFV_TMP)) call chmdealloc('zerom2.src','ZMERGE','MSTDFV_TMP',OUTLNCNT,crl = MSTDFV_TMP)
    call chmalloc('zerom2.src','ZMERGE','MSTDFV_TMP',OUTLNCNT,crl=MSTDFV_TMP)

    if(allocated(MYPOTEN_TMP)) call chmdealloc('zerom2.src','ZMERGE','MYPOTEN_TMP',OUTLNCNT,crl = MYPOTEN_TMP)
    call chmalloc('zerom2.src','ZMERGE','MYPOTEN_TMP',OUTLNCNT,crl=MYPOTEN_TMP)

    if(present(RMST1)) then
     if(OUTLNCNT.ne.size(RMST1)) then
#if KEY_PARALLEL==1
      if(MYNODP.eq.1) then 
#endif
       WRITE(6,*) 'WARNING: <trimconf> dataset column (RMST1) has wrong length'
#if KEY_PARALLEL==1
      endif  
#endif
     endif

     OUTLNCNT = size(RMST1)
     if(allocated(RMST1_TMP)) call chmdealloc('zerom2.src','ZMERGE','RMST1_TMP',OUTLNCNT,crl = RMST1_TMP)
     call chmalloc('zerom2.src','ZMERGE','RMST1_TMP',OUTLNCNT,crl=RMST1_TMP)
    endif

    NCOUNT = 0
    EMIN = 1.0D99
    do II = 1,OUTLNCNT
     ENER = MYPOTEN(II)
     if(ENER.LT.EMIN) EMIN = ENER
     if(ENER.LT.MYECUT) then
      NCOUNT = NCOUNT + 1
     endif
!make copies
     NSSTAG_TMP(II)=  NSSTAG(II)
     CONFMR_TMP(II) = CONFMR(II)
     MSTDOF_TMP(II) = MSTDOF(II)
     MSTDFV_TMP(II) = MSTDFV(II)
     MYPOTEN_TMP(II) = MYPOTEN(II)
     RMST1_TMP(II) = RMST1(II)
    enddo
    if(present(BNDWDTH)) MYVCUT = BNDWDTH + EMIN 
! reallocate original arrays
    if(allocated(NSSTAG)) call chmdealloc('zerom2.src','ZMERGE','NSSTAG',NCOUNT,intg = NSSTAG)
    call chmalloc('zerom2.src','ZMERGE','NSSTAG',NCOUNT,intg=NSSTAG)
    if(allocated(CONFMR)) call chmdealloc('zerom2.src','ZMERGE','CONFMR',NCOUNT,intg = CONFMR)
    call chmalloc('zerom2.src','ZMERGE','CONFMR',NCOUNT,intg=CONFMR)
    if(allocated(MSTDOF)) call chmdealloc('zerom2.src','ZMERGE','MSTDOF',NCOUNT,intg = MSTDOF)
    call chmalloc('zerom2.src','ZMERGE','MSTDOF',NCOUNT,intg=MSTDOF)
    if(allocated(MSTDFV)) call chmdealloc('zerom2.src','ZMERGE','MSTDFV',NCOUNT,crl = MSTDFV)
    call chmalloc('zerom2.src','ZMERGE','MSTDFV',NCOUNT,crl=MSTDFV)
    if(allocated(MYPOTEN)) call chmdealloc('zerom2.src','ZMERGE','MYPOTEN',NCOUNT,crl = MYPOTEN)
    call chmalloc('zerom2.src','ZMERGE','MYPOTEN',NCOUNT,crl=MYPOTEN)
    if(present(RMST1)) then
     if(allocated(RMST1)) call chmdealloc('zerom2.src','ZMERGE','RMST1',NCOUNT,crl = RMST1)
     call chmalloc('zerom2.src','ZMERGE','RMST1',NCOUNT,crl=RMST1)
     RMST1 = 0
    endif

    NSSTAG = 0
    CONFMR = 0
    MSTDOF = 0
    MSTDFV = 0
    MYPOTEN = 0

!    WRITE(6,*) 'MYNODP ',MYNODP,' NCOUNT ',NCOUNT,' OUTLNCNT ',OUTLNCNT

    NCOUNT2 =0
    LASTCNF = 0
    THISCNF = 0
    do II = 1,OUTLNCNT
     ENER = MYPOTEN_TMP(II)
     if(ENER.LT.MYECUT) then
      if((.not.present(BNDWDTH)).or.ENER.LT.MYVCUT) then  !band cutoff
       NCOUNT2 = NCOUNT2 + 1
       if(CONFMR_TMP(II).NE.LASTCNF)  THISCNF = THISCNF + 1
!     if(MYNODP.eq.1) then
!      endif
       NSSTAG(NCOUNT2) = NSSTAG_TMP(II)
       CONFMR(NCOUNT2) = THISCNF
       MSTDOF(NCOUNT2) = MSTDOF_TMP(II)
       MSTDFV(NCOUNT2) = MSTDFV_TMP(II)
       MYPOTEN(NCOUNT2) = ENER 
       RMST1(NCOUNT2) = RMST1_TMP(II)
       LASTCNF = CONFMR_TMP(II)
      endif
     endif
    enddo

    OUTLNCNT = NCOUNT2

 end subroutine trim_conf

 subroutine randsel(NSUBS,CSI, &
 NRANDAR_,ZRNDECUT,QZRNDECUT,ZRNDVTOL,QZRNDVCT,INSEEDAR,OUTSEEDAR,CSO)
 ! makes random selections of conformers and filters with energy cutoffs
 ! uses intrinsic fortran random number generator  --RJP

 use exfunc, only: ORDER,ORDER5
#if KEY_PARALLEL==1
 use nbndcc_utilb,only: parstoperr 
#endif
 use zcs,only: cs_alloc,cs_init,cs_write,cs_copy
 use ztypes,only : CONFSET
 use memory
#if KEY_PARALLEL==1
 use parallel,only: MYNODP 
#endif
 type(confset),intent(inout) :: CSI  !input data structure (conformer set) 
 type(confset),intent(out),optional :: CSO !output data structure 
 integer,intent(in) :: NSUBS
 integer,dimension(:),intent(in) :: NRANDAR_
 integer,dimension(:),intent(in),optional :: INSEEDAR !seed array
 integer,dimension(:),intent(out),optional :: OUTSEEDAR !seed array
 real(chm_real),intent(in) :: ZRNDECUT,ZRNDVTOL
 logical,intent(in) :: QZRNDECUT,QZRNDVCT
! integer,dimension(:),allocatable,intent(out) :: CSO%LOCNF,HICONFR,MSTDOFR,CSO%LODOF,CSO%HIDOF
! real(chm_real), dimension(:),allocatable,intent(out) :: CSO%MSTDV,CSO%ENERG
 
!locals
 type(confset) :: CST !output data structure 
 integer,dimension(:),allocatable,save :: seed
 integer :: II, SUBSPA, SIZESS,JJ,NN,NUMBER,NSEL,CONFSZ,LINE,OLDCNF,MSDLLN
 real(chm_real) :: RAND
 integer,save :: pass=0
 integer,dimension(:),allocatable :: SELECTED,IBRR
 integer :: NCONF,SSSIZE,TOTSEL,CNF,LOCNF,TOTDOF
 logical :: QDOSEED,QVERBOSE=.false.

! need to eliminate doubles.

  if(QVERBOSE) then
   write(6,*) 'size of CSI%HICNF top of randsel',size(CSI%HICNF)
   write(6,*) 'size of CSI%LOCNF top of randsel',size(CSI%LOCNF)
   write(6,*) 'size of CSI%MSTDF top of randsel',size(CSI%MSTDF)
   write(6,*) 'size of CSI%MSTDV top of randsel',size(CSI%MSTDV)
  endif
  QDOSEED = .false.
  if((present(INSEEDAR).and..not.present(OUTSEEDAR)).or.(present(OUTSEEDAR).and..not.present(INSEEDAR))) then
    call WRNDIE(-5,'<randsel>',' INSEED and OUTSEED are neither both absent or both present')
  else if(present(INSEEDAR).or.present(OUTSEEDAR)) then
    QDOSEED = .true. 
  endif
! set seeds
 pass = pass + 1
! if(pass.eq.1) then
!  call random_seed(size = NN) 
!  allocate(seed(NN))
!  NUMBER = 1
!  NUMBER = MYNODP  !PARALLEL
!  do II = 1,NN
!   SEED(II) = (NUMBER+II)*1000  
!  enddo
!  call random_seed(put=seed)
!  WRITE(6,*) 'MYNODP is ',MYNODP,' seed is ',SEED
! endif

!  WRITE(6,*) 'INSIDE RANDSEL NSUBS ',NSUBS,' NRANDAR_(1) ',NRANDAR_(1)

! calculate total number of conformers
 TOTSEL = 0
 TOTDOF = 0
 DO II = 1,NSUBS
  LOCNF = CSI%LOCNF(II)
  if(QVERBOSE) then 
    WRITE(6,*) 'for subspace  LOCONF is ',LOCNF,' CSI%HIDOF(LOCNF) ', CSI%HIDOF(LOCNF), &
    ' CSI%LODOF(LOCNF) ',CSI%LODOF(LOCNF)
  endif
  CONFSZ = CSI%HIDOF(LOCNF) - CSI%LODOF(LOCNF) + 1
  if(NRANDAR_(II).GT.0) then  !if conformers being randomized
!  WRITE(6,*) 'SUBS ',II,' CONFSZ ',CONFSZ,' NCONF ',NRANDAR_(II)
   TOTSEL = TOTSEL + NRANDAR_(II)
   TOTDOF = TOTDOF + NRANDAR_(II)*CONFSZ
  if(QVERBOSE) WRITE(6,*) 'for subspace ',II,' NRANDAR is ',NRANDAR_(II),' CONFSZ ',CONFSZ,' TOTDOF is ',TOTDOF
  else
!  WRITE(6,*) 'SUBS ',II,' CONFSZ ',CONFSZ,' NCONF ',(CSI%HICNF(II) - CSI%LOCNF(II) + 1)
    TOTSEL = TOTSEL + (CSI%HICNF(II) - CSI%LOCNF(II) + 1)
    TOTDOF = TOTDOF + (CSI%HICNF(II) - CSI%LOCNF(II) + 1)*CONFSZ
  endif
 ENDDO
! WRITE(6,*) 'MYNODP ',MYNODP,' TOTSEL IS ',TOTSEL,' TOTDOF ',TOTDOF

! allocate space
  if(QVERBOSE) write(6,*) 'before allocation NSUBS ',NSUBS,' TOTSEL ',TOTSEL,' TOTDOF ',TOTDOF
 call cs_alloc(CST,NSUBS,TOTSEL,TOTDOF,'zerom2.src','RANDSEL') 
  if(QVERBOSE) then
   write(6,*) 'size of CST%HICNF after alloc in randsel',size(CST%HICNF)
   write(6,*) 'size of CST%LOCNF after alloc in randsel',size(CST%LOCNF)
   write(6,*) 'size of CST%MSTDF after alloc in randsel',size(CST%MSTDF)
   write(6,*) 'size of CST%MSTDV after alloc in randsel',size(CST%MSTDV)
  endif

 call cs_init(CST)  !initialize

  if(QVERBOSE) write(6,*) 'size of CSI%MSTDF top2 of randsel',size(CSI%MSTDF)
 NCONF = 0
 LINE = 0
! WRITE(6,*) 'size CSI%LOCNF ',size(CSI%LOCNF),' size CSI%HICNF ',size(CSI%HICNF)
 do II = 1,NSUBS
    !      WRITE(6,*) 'FOR PASSED SS ',II,' SUBSPA = ',SUBSPA,
    !     & ' CSI%HICNF ',CSI%HICNF(SUBSPA)
   SIZESS = CSI%HICNF(II) - CSI%LOCNF(II) + 1
#if KEY_PARALLEL==1
   if(MYNODP.eq.1) then 
#endif
     WRITE(6,'(5X,A14,1X,I14)') 'SUBSPACE ',II,' NUMb of CONF = ',SIZESS
#if KEY_PARALLEL==1
   endif 
#endif
   NSEL = NRANDAR_(II)
   if(allocated(SELECTED)) deallocate (SELECTED,IBRR)
   allocate(SELECTED(NSEL),IBRR(NSEL))
   if((NSEL.LT.SIZESS).and.(NSEL.GT.0)) then
    if(QDOSEED) then 
     call random_array(SELECTED,SIZESS,NSEL,INSEEDAR,OUTSEEDAR)
    else
     call random_array(SELECTED,SIZESS,NSEL)
    endif

!    do JJ = 1,NSEL !loop over number of random selections
!!     call random_number(harvest=RAND)
!!     SELECTED(JJ) = int(RAND*SIZESS) + 1
!     if(MYNODP.eq.1) then !PARALLEL
!      WRITE(6,*) 'MYNODP ',MYNODP,' JJ ',JJ,' SELECTED ',SELECTED(JJ),' SIZESS ',SIZESS
!     endif !PARALLEL
!!     IBRR(JJ) = JJ
!    enddo
    call qisort(SELECTED,NSEL,IBRR)
    do JJ = 1,NSEL
     NCONF = NCONF + 1
     OLDCNF = SELECTED(JJ) + CSI%LOCNF(II) - 1
     do MSDLLN = CSI%LODOF(OLDCNF),CSI%HIDOF(OLDCNF)
!       if(MYNODP.eq.1) then
!         WRITE(6,*) 'OLDCONF is ',OLDCNF,' MSDLLN ',MSDLLN
!       endif
       LINE = LINE + 1
       if(MSDLLN.LT.1) then
         write(6,*) 'size of CSI%LODOF ',size(CSI%LODOF)
         write(6,*) 'line in zutil is ',line,' OLDCNF ',OLDCNF,' MSDLLN ',MSDLLN
       endif
       CST%MSTDF(LINE) = CSI%MSTDF(MSDLLN)
       CST%MSTDV(LINE) = CSI%MSTDV(MSDLLN)
     enddo 
     CST%ENERG(NCONF) = CSI%ENERG(OLDCNF)
     CST%HIDOF(NCONF) = LINE
     CST%LODOF(NCONF) = LINE - (CSI%HIDOF(OLDCNF) - CSI%LODOF(OLDCNF)) 
!if(MYNODP.eq.1) then
!     WRITE(6,*) '>>SELEMYNODP ',MYNODP,' JJ ',JJ,' SORTED SELECTED ',SELECTED(JJ),' IBRR ',IBRR(JJ)
!endif
    enddo
    CST%LOCNF(II) = NCONF - NSEL + 1
    CST%HICNF(II) = NCONF
   else !no randomization
#if KEY_PARALLEL==1
    if(MYNODP.eq.1) then 
#endif
      WRITE(6,*) 'WARNING <RANDSEL>: NOT ENOUGH CONFORMERS, SUBSPACE ',II,' USING ALL ',SIZESS
#if KEY_PARALLEL==1
    endif 
#endif
    do JJ = CSI%LOCNF(II),CSI%HICNF(II)
     NCONF = NCONF + 1
     do MSDLLN = CSI%LODOF(JJ),CSI%HIDOF(JJ)
       LINE = LINE + 1
       CST%MSTDF(LINE) = CSI%MSTDF(MSDLLN)
       CST%MSTDV(LINE) = CSI%MSTDV(MSDLLN)
     enddo
     CST%ENERG(NCONF) = CSI%ENERG(JJ) 
     CST%HIDOF(NCONF) = LINE
     CST%LODOF(NCONF) = LINE - (CSI%HIDOF(OLDCNF) - CSI%LODOF(OLDCNF))
    enddo
#if KEY_PARALLEL==1
    if(MYNODP.eq.1) then  
#endif
    WRITE(6,*) 'WARNING: FOR SS ',II,' No randomization required '
#if KEY_PARALLEL==1
    endif 
#endif
    CST%LOCNF(II) = NCONF - (CSI%HICNF(II)-CSI%LOCNF(II))
    CST%HICNF(II) = NCONF
   endif
!   if(MYNODP.eq.1) then 
!   WRITE(6,*) 'MYNODP ',MYNODP,' SUBSPACE is ',II,' RAND is ',RAND
!   WRITE(6,*) ' LINES in new CST%HIDOF ',LINE,' TOTSEL ',TOTSEL,' TOTDOF ',TOTDOF
!   endif
!   WRITE(MYNODP+10,*) RAND
 enddo
! if(MYNODP.eq.1) then
!  call cs_write(CST,NSUBS,MYNODP+600)
! endif
! WRITE(6,*) 'MYNODP ',MYNODP,' end of RANDSEL '
!  call parstoperr('<RANDSEL>',' stopping after print')
  if(QVERBOSE) then
   write(6,*) 'size of CST%HICNF bottom of randsel',size(CST%HICNF)
   write(6,*) 'size of CST%LOCNF bottom of randsel',size(CST%LOCNF)
   write(6,*) 'size of CST%MSTDF bottom of randsel',size(CST%MSTDF)
   write(6,*) 'size of CST%MSTDV bottom of randsel',size(CST%MSTDV)
  endif
  if(present(CSO)) then
    call cs_copy(CST,CSO)
  else
    call cs_copy(CST,CSI)
  endif
end subroutine randsel
!--------------------------------------
!--------------------------------------
subroutine loc_timer(string)
   use new_timer, only: seconds
   implicit none

character(len=*),intent(in) :: string

#if KEY_PARALLEL==1
    if((MYNODP.eq.1).or.(MYNODP.eq.NUMNOD).or.(MYNODP.eq.4)) then
#endif
     call seconds(etime,ctime)
#if KEY_PARALLEL==1
     WRITE(6,*) '>>time MYNODP ',MYNODP,string,' etime ',etime,' ctime ',ctime
    endif
#else
    WRITE(6,*) '>>time ',string,' etime ',etime,' ctime ',ctime
#endif
end subroutine loc_timer
 
!----------------------------------------------------------------------------------------------
!----------------------------------------------------------------------------------------------
 end module zutil
