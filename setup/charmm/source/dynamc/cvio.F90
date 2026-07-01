module cvio
  use chm_kinds
  implicit none
  LOGICAL, PRIVATE :: QCHECK=.TRUE., QRPICO, QWPICO=.FALSE.
  real(chm_real), private :: TSET

contains

SUBROUTINE READCV(X,Y,Z, &
#if KEY_CHEQ==1
     CG,QCG,                          & 
#endif
     TEMP,NATOM,FREEAT,NFREAT, &
     FIRSTU,NUNIT,IUNIT,NFILE, &
     ISTEP,ISTATS,NDEGF,DELTA, &
     BEGIN,STOP,SKIP,NSAVV,HDR1,HDR2, &
     TITLE,NTITLE,DIM4,FDIM,Q_PLL)
  !
  !             THIS IS USED TO READ IN COORDINATES OR VELOCITIES DURING
  !     THE ANALYSIS OF DYNAMICS OUTPUT. VARIOUS OTHER VARIABLES STORED IN
  !     THE ICNTRL ARRAY ARE ALSO READ IN. NOTE THAT FREEAT WILL BE READ
  !     IF NFREAT IS NOT EQUAL TO NATOM.
  !             ISTATS IS A FLAG WITH THE FOLLOWING FUNCTIONS:
  !
  !             ON CALL
  !             1 - OPEN A NEW UNIT FOR READING THE INFORMATION
  !             2 - USE THE UNIT THAT IS ALREADY OPEN ON IUNIT
  !             3 - SAME AS 2, BUT CALC. IFILE FROM ISTEP & SKIP
  !
  !             ON RETURN
  !            -1 - THE REQUESTED INFORMATION HAS BEEN READ
  !             1 - NOT DONE READING, BUT THIS FILE IS FINISHED
  !             2 - NOT DONE READING, AND THIS FILE IS IS NOT DONE.
  !     HDR1 AND HDR2 ARE OPTIONS FOR THE FILE HEADER THAT IS READ.
  !     RECORDS WILL BE USED IF MOD(ISTEP,SKIP)=0 AND BEGIN<=ISTEP<=STOP.
  !     DJS 1/25/81  !
  !     Authors: S. Swaminathan
  !              David Perahia
  !              Dave States
  !              Robert Bruccoleri
  !
  !    Q_PLL    mfc added logical variable to signal whether
  !             the calling routine is being done in parallel or not.
  !             When calling routine is not parallel, master hangs
  !             trying to send data to slaves that are not receiving.
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use stream
  use image
#if KEY_FLUCQ==1
  use flucq     
#endif
  use machio,only:vopen
  use memory
  use number,only:zero
#if KEY_MULTICOM==1
  use multicom_aux                                    
#endif
  use param_store, only: set_param
  use parallel,only:psnd4,psnd8

  real(chm_real),allocatable,dimension(:,:,:) :: TRANSF
  INTEGER NATOM
  real(chm_real) X(NATOM),Y(NATOM),Z(NATOM)
#if KEY_CHEQ==1
  real(chm_real) CG(NATOM)
  LOGICAL QCG
#endif 

  REAL(chm_real4) TEMP(NATOM)
  REAL(chm_real8) DUMXTL(6)
  INTEGER FREEAT(*),NFREAT,FIRSTU,NUNIT,IUNIT,NFILE
  INTEGER ISTEP,ISTATS
  INTEGER NDEGF,NSAVV
  real(chm_real) DELTA
  INTEGER BEGIN,STOP,SKIP
  character(len=4) HDR1,HDR2
  character(len=*) TITLE(*)
  INTEGER NTITLE
  LOGICAL DIM4,Q_PLL
  real(chm_real) FDIM(NATOM)
#if KEY_CHEQ==1
  Logical :: QHAS_CHGS
  INTEGER IHAS_CHGS
#endif 
  INTEGER ONFREA,LASTU,IFILE,SKIP_1

  LOGICAL QCRYS_1,QDIM4_1
  !
  INTEGER ICNTRL(20),natrec
  INTEGER NINTVL,NTOT,MULTIP,LFILE,IWARN
  INTEGER NPREV,NFINAL,NEWSTE,I
  real(chm_real) JUNK, TIME
  REAL(chm_real4) DELTA4
  character(len=4) HDRR
  ! BIOVIA Code Start : Bug fix
  LOGICAL RDFST,RDFRE,QCRYS,QDIM4
  ! BIOVIA Code End
  !ln...To enble to read coordinates at multiples of skip FROM actual
  !     first coordinate set in file, 27-Jul-95
  INTEGER ISTEP1
  !ln...9507
  integer int8traju
  !
  !     save all local variables between calls to READCV
  SAVE

  ! variables added to handle premature end of file error
  integer :: io_status
  character(len=128) :: iunit_filename
  character(len=256) :: io_error_msg

  ! Gruesome hack alert:: treat istats=3 the same as istats=2, except calculate
  ! ifile from istep and nskip, y'know, like a sane person would, instead of relying
  ! on ifile being saved accross subroutine calls.
  if(istats.eq.3) then
     istats=2
     ifile=(istep/skip)-1 ! we add 1 later, so subtrace it here
  endif
  !!write(outu,'(a,5i6)') 'TIM DEBUG CVIO> BEGIN,ISTEP,STOP,IFILE,NFILE = ', BEGIN, ISTEP, STOP, IFILE, NFILE
  !
  int8traju=0

  ioblock: IF(IOLEV > 0 &
#if KEY_MULTICOM==1 /*  VO stringm communication */
 &                      .or. (ME_LOCAL.eq.0) &      
#endif
 &                      ) THEN
     RDFST=.FALSE.
     IF(SKIP <= 0) THEN
        IF(WRNLEV >= 2) WRITE(OUTU,2007) SKIP
2007    FORMAT(' ***** WARNING ***** from READCV -- SKIP, ',I6, &
             ' was not positive.'/' It will be set to 1.')
        SKIP=1
     ENDIF
     IF(ISTATS == 1) GOTO 100
     IF(ISTATS == 2) GOTO 200
     IF(WRNLEV >= 2) WRITE(OUTU,908) ISTATS
908  FORMAT(' ERROR in READCV -- Bad ISTATS,',I12,' passed.')
     CALL DIEWRN(-3)
     GOTO 9000
     !
     !     read a new file
     !
100  CONTINUE
     !  --- Test to see if the unit is open, if not, try to open it ------
     !  --- Usually users will have opened the files before this call ----
     !  ---   so it usually does nothing. --------------------------------
     CALL TRYORO(IUNIT,'UNFORMATTED')
     IF(reallow)THEN    
        REWIND(IUNIT)
     ENDIF              
     ! Don't let GTICNT print for now, to make testing easier
     CALL GTICNT(IUNIT,HDRR,ICNTRL,.FALSE.,.FALSE.,.FALSE.)
     ! check the headers to see if they match.
     !
     IF(HDRR /= HDR1.AND.HDRR /= HDR2) THEN
        IF(WRNLEV >= 2) WRITE(OUTU,903) HDRR,HDR1,HDR2
903     FORMAT(' *****  ERROR  ***** HEADERS DO NOT MATCH',3(6X,A4))
        CALL DIEWRN(-1)
     ENDIF

     icntsv(1:20)=icntrl(1:20)   

     CALL RDTITL(TITLE,NTITLE,IUNIT,-1)
        READ(IUNIT) NATREC
     NFILE=ICNTRL(1)
     ISTEP=ICNTRL(2)
     !yw...When incorporate LN9507, ISTEP1 = ISTEP?
     ISTEP1=ISTEP
     !yw...9507
     NINTVL=ICNTRL(3)
     ONFREA=NFREAT
     NFREAT=NATREC-ICNTRL(9)
     QCRYS=(ICNTRL(11) == 1)
     QDIM4=(ICNTRL(12) == 1)
     IF(IUNIT == FIRSTU)THEN
       QCRYS_1=QCRYS
       QDIM4_1=QDIM4
     ELSE
       IF(WRNLEV >= 2)THEN
         IF(QCRYS .NEQV. QCRYS_1) WRITE(OUTU,'(A,I5)') &
          'Presence of crystal data in first file does not match IUNIT:',IUNIT
         IF(QDIM4 .NEQV. QDIM4_1) WRITE(OUTU,'(A,I5)') &
          'Presence of 4D data in first file does not match IUNIT:',IUNIT
       ENDIF  
     ENDIF
#if KEY_CHEQ==1
     QHAS_CHGS=(ICNTRL(13) == 1)
     IF (PRNLEV >= 2.AND.QCG) &
          WRITE(OUTU,*)"--- QCG IS TRUE in READCV."
     IF(PRNLEV >= 2.AND.QHAS_CHGS)THEN
        IF(QCG) THEN
           WRITE(OUTU,'(/,/,A,/)') &
                "-- CHEQ DYNA FILE, CHARGES READ BUT IGNORED (CHEQ is OFF) --"
        ELSE
           WRITE(OUTU,'(/,/,A,/)') &
                "-- CHEQ DYNA FILE, CHARGE READ AND UPDATED ALONG WITH XYZ. --"
        ENDIF
     ENDIF
#else /**/
     if(icntrl(13) == 1)then
        write(outu,913)
913     format("This trajectory has CHEQ data but charmm not", &
             " compiled with CHEQ")
        call wrndie(-4,'<READCV>', &
             'CHEQ not compiled in, traj written with CHEQ')
     endif
#endif 
     
     !
     IF(PRNLEV >= 2) THEN
       WRITE(OUTU,26) IUNIT,(ICNTRL(I),I=1,4)
26     FORMAT(/' READING TRAJECTORY FROM UNIT',I4,/ &
          '   NUMBER OF COORDINATE SETS IN FILE:',I12,/ &
          '   NUMBER OF PREVIOUS DYNAMICS STEPS:',I12,/ &
          '   FREQUENCY FOR SAVING COORDINATES: ',I12,/ &
          '   NUMBER OF STEPS FOR CREATION RUN: ',I12,/)
     ENDIF
    !
     RDFRE=.FALSE.
     IF(NATREC /= NATOM) THEN
        IF (NFREAT == NATOM) THEN
           IF(PRNLEV >= 2) WRITE(OUTU,904)
904        FORMAT(' READCV> READING ONLY THE FREE ATOMS FROM', &
                ' THE TRAJECTORY')
           RDFRE=.TRUE.
        ELSE
           IF(WRNLEV >= 2) WRITE(OUTU,905) NATOM,NATREC
905        FORMAT(' *****  ERROR  ***** NO. OF ATOMS DO NOT MATCH',2I8)
           CALL DIEWRN(-4)
        ENDIF
     ENDIF
     !
     IF(NFREAT /= NATREC) READ(IUNIT) (FREEAT(I),I=1,NFREAT)
     !
     IF(.NOT.QCHECK)THEN 
     ! BEGIN and STOP from commandline are not meaningful here, and SKIP has to
     ! be relative to each file
        BEGIN=-1
        STOP=-1
     ENDIF  
     IF (IUNIT == FIRSTU ) THEN !!!.AND. NUNIT > 1) THEN
        ! Save initial value for comparison with later files when .NOT.QCHECK
        SKIP_1=SKIP
        IF(.NOT. QCHECK)THEN
           CALL WRNDIE(1,'<READCV>', &
           'NO CHECKING THAT TRAJECTORY FILES MATCH')
        ENDIF
        LASTU=FIRSTU+NUNIT-1
        NTOT=0
        CALL WRTITL(TITLE,NTITLE,OUTU,1)
        !
        ! check SKIP value, and try to accommodate NOCHECK for each file
        !
        IF(MOD(SKIP_1,NINTVL) /= 0 ) THEN
           MULTIP=GCD(SKIP_1,NINTVL)
           IF(QCHECK)THEN
              SKIP=SKIP_1*NINTVL/MULTIP
              IF(WRNLEV >= 2) WRITE(OUTU,2000) SKIP_1,NINTVL,SKIP
2000          FORMAT(' *****  WARNING  ***** SKIP=',I6,' WAS NOT A', &
                ' MULTIPLE OF THE FILE INTERVAL=',I6/ &
                ' IT HAS BEEN RESET TO',I6)
           ELSE  IF(SKIP_1 < NINTVL)THEN
              SKIP=NINTVL
              IF(WRNLEV >= 2) WRITE(OUTU,'(/A,I6/)') & 
                 '*** Reading all frames - SKIP has been set to:',SKIP
           ENDIF 
        ENDIF

        !
        ! check BEGIN value
        !
        IF(BEGIN <= 0) THEN
           IF(SKIP == 1) THEN
              I=ISTEP1
           ELSE
              I=((ISTEP1-1)/SKIP+1)*SKIP
           ENDIF
           IF(WRNLEV >= 2) WRITE(OUTU,2001) BEGIN,I
2001         FORMAT(' *****  WARNING  ***** BEGIN=',I12, &
                ' Was not specified. It has been set to:',I12)
           BEGIN=I
        ENDIF
        !ln
        IF(BEGIN < ISTEP1) THEN
           IF(WRNLEV >= 2) WRITE(OUTU,2004) BEGIN,ISTEP1
2004       FORMAT(' *****  WARNING  ***** BEGIN=',I12, &
                '  Was specified to a value smaller than ISTEP1=',I12)
           CALL WRNDIE(-2,'<READCV>','BEGIN was specified too small.')
           BEGIN=ISTEP1
        ENDIF
        !ln
        IF(MOD((BEGIN-ISTEP1),NINTVL) /= 0) THEN
           I=(((BEGIN-ISTEP1)/NINTVL)+1)*NINTVL + ISTEP1
           IF(WRNLEV >= 2) WRITE(OUTU,2002) BEGIN,ISTEP1,I
2002       FORMAT(' *****  WARNING  ***** BEGIN-ISTEP1=',I12,' -',I12, &
                '  Was not a multiple of nintvl',/, &
                '  Begin has been reset to',i12)
           CALL WRNDIE(1,'<READCV>','BEGIN was invalid and reset.')
           BEGIN=I
        ENDIF
        !
        ! check STOP value
        !
        IF(STOP >= 1) THEN
           IF(MOD((STOP-BEGIN),SKIP) /= 0) THEN
              I=((STOP-BEGIN)/SKIP)*SKIP + BEGIN
              IF(WRNLEV >= 2) WRITE(OUTU,2003) STOP,BEGIN,I
2003          FORMAT(' *****  WARNING  ***** STOP-BEGIN=',I12, &
                   ' -',I8,' was not a multiple of skip',/, &
                   ' STOP has been reset to:',I12)
              STOP=I
           ENDIF
           !ln...9507
           IF(BEGIN > STOP) THEN
              IF(WRNLEV >= 2) WRITE(OUTU,2005) BEGIN,STOP
2005          FORMAT(' *****  WARNING  ***** BEGIN',I12, &
                   ' IS GREATER THAN STOP',I12,/, &
                   ' NO DATA WILL BE READ')
              ISTATS=-1
              GOTO 9000
           ENDIF
        ENDIF
        !
        NSAVV=ICNTRL(5)
        NDEGF=ICNTRL(8)
        LFILE=SKIP/NINTVL
        ! store bytes of icntrl(10) in delta4
        delta4 = transfer(icntrl(10), delta4)
        DELTA=DELTA4
        ! convert to AKMA flag; assumes ps units in file hdr
        IF (QRPICO) THEN
          DELTA=DELTA/TIMFAC
        ! set explicitly if requested via TSET (ps)
        ELSE IF (TSET>0.D0) THEN
          DELTA=TSET/TIMFAC
        ENDIF
        !mu...7/93, M.E. Karpen, needed to correctly read fixed atoms
        !     from first frame of data when the first frame desired (BEGIN)
        !     is greater than first frame of data (ICNTRL(2)).
        RDFST = .TRUE.
        !mu...
     ELSE
        !
        ! check other files for consistency
        !
        IWARN=-1
        IF(.NOT. QCHECK)THEN
          !
          ! check SKIP value, try to accommodate NOCHECK for each file
          !
          SKIP=SKIP_1          
          IF(MOD(SKIP_1,NINTVL) /= 0 ) THEN
             MULTIP=GCD(SKIP_1,NINTVL)
             IF(SKIP_1 <NINTVL )THEN
               SKIP=NINTVL
               IF(WRNLEV >= 2) WRITE(OUTU,'(/A,I6/)') & 
                   '*** Reading all frames. SKIP set to',SKIP
             ELSE
               SKIP=SKIP_1*NINTVL/MULTIP
               IF(WRNLEV >= 2) WRITE(OUTU,2000) SKIP_1,NINTVL,SKIP
             ENDIF 
          ENDIF

          !
          ! check BEGIN value
          !
          IF(BEGIN <= 0) THEN
             IF(SKIP == 1) THEN
                I=ISTEP1
             ELSE
                I=((ISTEP1-1)/SKIP+1)*SKIP
             ENDIF
             IF(WRNLEV >= 2) WRITE(OUTU,2001) BEGIN,I
             BEGIN=I
          ENDIF
          LFILE=SKIP/NINTVL
          IWARN=1
        ENDIF
        IF (NPREV /= ISTEP) THEN
           IF(WRNLEV >= 2) WRITE(OUTU,906) IUNIT,ISTEP,NPREV
906        FORMAT(' *****  ERROR  ***** FILES DO NOT MERGE',/ &
                3X,'UNIT',I4,' BEGINS AT ',I12,' INSTEAD OF ',I12)
           CALL DIEWRN(IWARN)
           ISTEP=NPREV
        ENDIF
        !
        IF (ONFREA /= NFREAT) THEN
           IF(WRNLEV >= 2) WRITE(OUTU,907)
907        FORMAT(' *****  ERROR  ***** IN READCV --'/ &
                ' Number of fixed atoms does not match.')
           CALL DIEWRN(IWARN)
        ENDIF
     ENDIF
     !
     IFILE=0
     NFINAL=(NFILE-1)*NINTVL+ISTEP
     ISTATS=2
     NPREV=NFINAL+NINTVL
     ISTEP=ICNTRL(2)-NINTVL
     !
     ! set some substitution parameters based on this new file
     CALL set_param('NFILE',ICNTRL(1))
     CALL set_param('START',ICNTRL(2))
     CALL set_param('SKIP', ICNTRL(3))
     CALL set_param('NSTEP',ICNTRL(4))
     CALL set_param('NDEGF',ICNTRL(8))
     call set_param('DELTA',DELTA)
     !
     ! continue reading an existing file
     !
200  CONTINUE
     NEWSTE=ISTEP+NINTVL
     IF(BEGIN > NEWSTE) GOTO 205
     !ln...9507
     !ln      IF (MOD(NEWSTE,SKIP) == 0) GOTO 210
     IF (MOD((NEWSTE-BEGIN),SKIP) == 0) GOTO 210
     !ln...9507
205  CONTINUE
     IF (RDFST.AND..NOT.RDFRE) THEN
        !
        ! Make sure we always get the first coordinate set since it contains
        ! the fixed atoms.
        !
           IF(QCRYS) then
        ! and this info is alwasy real*8 even if code is single precision
              READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) DUMXTL
              XTLABC=DUMXTL
           endif
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) TEMP
           X(1:natom)=TEMP(1:natom)
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) TEMP
           Y(1:natom)=TEMP(1:natom)
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) TEMP
           Z(1:natom)=TEMP(1:natom)
        IF(QDIM4) THEN
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) TEMP
           IF(DIM4) THEN
              FDIM(1:natom)=TEMP(1:natom)
           ENDIF
        ENDIF
#if KEY_CHEQ==1
        IF (QHAS_CHGS) THEN
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) TEMP
           IF (QCG) THEN
              CG(1:natom)=TEMP(1:natom)
           ENDIF
        ENDIF
#endif 
#if KEY_FLUCQ==1
        IF(QFLUC) CALL FQCREA(IUNIT,TEMP) 
#endif
     ELSE
        IF(QCRYS) READ(IUNIT) JUNK
        READ (IUNIT) JUNK
        READ (IUNIT) JUNK
        READ (IUNIT) JUNK
        IF(QDIM4) READ (IUNIT) JUNK
#if KEY_FLUCQ==1
        IF(QFLUC) READ (IUNIT) JUNK   
#endif
#if KEY_CHEQ==1
        IF (QHAS_CHGS) READ (IUNIT) JUNK   
#endif
     ENDIF
     RDFST=.FALSE.
     ISTEP=NEWSTE
     IFILE=IFILE+1
     !
     IF (IFILE < NFILE) GOTO 200
     !ln is this really necessary? L.Nilsson, Oct-96, wanting to reuse files
     !      CALL VCLOSE(IUNIT,'KEEP',ERROR)
     IUNIT=IUNIT+1
     IF(IUNIT > LASTU) THEN
     !LNI. Something is wrong, bail out     
        CALL WRNDIE(-1,'<READCV>','Trying to open file beyond LASTU')
        GOTO 300
     ENDIF
     ISTATS=1
     GOTO 100
     !
210  CONTINUE
     ISTEP=NEWSTE
     IFILE=IFILE+1
     !
     IF(QCRYS)  then
        READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) DUMXTL
        XTLABC=DUMXTL
     endif
     IF (RDFRE.AND.IFILE == 1) THEN
        READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) &
             (TEMP(I),I=1,NATREC)
        DO I=1,NATOM
           X(I)=TEMP(FREEAT(I))
        ENDDO
        READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) &
             (TEMP(I),I=1,NATREC)
        DO I=1,NATOM
           Y(I)=TEMP(FREEAT(I))
        ENDDO
        READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) &
             (TEMP(I),I=1,NATREC)
        DO I=1,NATOM
           Z(I)=TEMP(FREEAT(I))
        ENDDO
        IF(QDIM4) THEN
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) &
                (TEMP(I),I=1,NATREC)
           IF(DIM4) THEN
              DO I=1,NATOM
                 FDIM(I)=TEMP(FREEAT(I))
              ENDDO
           ENDIF
        ENDIF
#if KEY_CHEQ==1
        IF (QHAS_CHGS) THEN
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) TEMP
           IF (QCG) THEN
              CG(1:natom)=TEMP(1:natom)
           ENDIF
        ENDIF
#endif 
#if KEY_FLUCQ==1
        IF(QFLUC) CALL FQCREA(IUNIT,TEMP)  
#endif
        !
     ELSE IF (IFILE == 1.OR.NFREAT.EQ.NATOM) THEN
           !!WRITE(OUTU,'(a,i5,a,i9)') 'TIM DEBUG CVIO> READ FROM UNIT ',IUNIT,' OFFSET ',FTELL(IUNIT)
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) TEMP
           X(1:natom)=TEMP(1:natom)
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) TEMP
           Y(1:natom)=TEMP(1:natom)
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) TEMP
           Z(1:natom)=TEMP(1:natom)
        IF(QDIM4) THEN
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) TEMP
           IF(DIM4) THEN
              FDIM(1:natom)=TEMP(1:natom)
           ENDIF
        ENDIF
#if KEY_CHEQ==1
        IF (QHAS_CHGS) THEN
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) TEMP
           IF (QCG) THEN
              CG(1:natom)=TEMP(1:natom)
           ENDIF
        ENDIF
#endif 
#if KEY_FLUCQ==1
        IF(QFLUC) CALL FQCREA(IUNIT,TEMP)  
#endif
        !
     ELSE
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) &
                (TEMP(I),I=1,NFREAT)
           DO I=1,NFREAT
              X(FREEAT(I))=TEMP(I)
           ENDDO
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) &
                (TEMP(I),I=1,NFREAT)
           DO I=1,NFREAT
              Y(FREEAT(I))=TEMP(I)
           ENDDO
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) &
                (TEMP(I),I=1,NFREAT)
           DO I=1,NFREAT
              Z(FREEAT(I))=TEMP(I)
           ENDDO
        IF(QDIM4) THEN
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) &
                (TEMP(I),I=1,NFREAT)
           IF(DIM4) THEN
              DO I=1,NFREAT
                 FDIM(FREEAT(I))=TEMP(I)
              ENDDO
           ENDIF
        ENDIF
#if KEY_CHEQ==1
        IF (QHAS_CHGS) THEN
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) &
                (TEMP(I),I=1,NATREC)
           READ (IUNIT, IOSTAT=io_status, err=3000, end=3000) TEMP
           IF (QCG) THEN
              CG(1:natom)=TEMP(1:natom)
           ENDIF
        ENDIF
#endif 
#if KEY_FLUCQ==1
        IF(QFLUC) CALL FQCREA(IUNIT,TEMP)  
#endif
     ENDIF
     !
     NTOT=NTOT+1
     !
     IF((NFILE-IFILE) < LFILE .AND.IUNIT == LASTU) GOTO 300
     IF (ISTEP == STOP) GOTO 300
     IF (IFILE < NFILE) GOTO 9000
     !ln, Oct-96 wanting to reuse files
     !      CALL VCLOSE(IUNIT,'KEEP',ERROR)
     IUNIT=IUNIT+1
     IF(IUNIT > LASTU) THEN
     !LNI. Something is wrong, bail out     
        CALL WRNDIE(-1,'<READCV>','Trying to open file beyond LASTU')
        GOTO 300
     ENDIF
     ISTATS=1
     GOTO 9000
     !
300  CONTINUE
     !ln, Oct-96 wanting to reuse files
     !     CALL VCLOSE(IUNIT,'KEEP',ERROR)
     IF (STOP > 0.AND.STOP /= ISTEP) THEN
        IF(WRNLEV >= 2) WRITE(OUTU,3004) ISTEP,STOP
3004    FORMAT(' *****  WARNING  ***** ONLY',I8,' OF THE',I8, &
             ' REQUESTED STEPS WERE IN THE INPUT FILE(S)')
     ENDIF
     STOP=ISTEP
     ISTATS=-1

     IF(PRNLEV >= 2) WRITE(OUTU,902) NTOT,HDRR,NUNIT,FIRSTU, &
          BEGIN,STOP,SKIP,DELTA
902  FORMAT(/I5,1X,A4,' RECORDS READ FROM',I4, &
          ' UNITS STARTING WITH UNIT',I4,/ &
          ' RUNNING FROM STEP',I8,' TO ',I8, &
          ' SKIPPING',I5,' STEPS BETWEEN RECORDS'/ &
          ' Time step was ',1PG14.7,' AKMA time units.')
9000 CONTINUE
     !
     IF(QCRYS .AND. XDIM > 0) THEN
        IF (ICNTRL(20) < 22) THEN
           !     for version numbers  <  22 we have XUCELL data in XTLABC.
           xucell(1:6) = xtlabc(1:6)
           CALL XTLAXS(XTLABC,XUCELL)
        ENDIF
     ENDIF
  ENDIF ioblock
#if KEY_PARALLEL==1 /*pll*/
  if(q_pll)then
!     CALL PSNDC(HDR,1)
 !    CALL PSND4(ICNTRL,20)
     CALL PSND4(NTOT,1)
     CALL PSND4(ISTEP,1)
     CALL PSND4(ISTATS,1)
     CALL PSND4(NDEGF,1)
     CALL PSND4(NSAVV,1)
     CALL PSND4(NFREAT,1)
     CALL PSND4(NFILE,1)
     CALL PSND4(BEGIN,1)
     CALL PSND4(STOP,1)
     CALL PSND4(SKIP,1)
     CALL PSND8(DELTA,1)
     CALL PSND4(FREEAT,NFREAT)
     CALL PSND8(X,NATOM)
     CALL PSND8(Y,NATOM)
     CALL PSND8(Z,NATOM)
     CALL PSND4(QCRYS,1)
     CALL PSND4(QDIM4,1)
#if KEY_CHEQ==1
     IF(QCG)THEN
        IHAS_CHGS=0
        IF(QHAS_CHGS)IHAS_CHGS=1
        CALL PSND4(IHAS_CHGS,1)
        IF (IHAS_CHGS == 1) CALL PSND8(CG,NATOM)
     ENDIF
#endif 
     CALL PSND8(XTLABC,6)
     IF(QDIM4 .AND. DIM4) CALL PSND8(FDIM,NATOM)
  endif
#endif /* (pll)*/
  !
  IF(DIM4 .AND. .NOT.QDIM4) FDIM(1:NATOM)=zero
  !
  IF(QCRYS .AND. XDIM > 0) THEN
     xucold(1:6) = xucell(1:6)
     CALL XTLLAT(XUCELL,XTLABC)
     CALL XTLMSR(XUCELL)
     !     Recompute the images from the crystal transformations.
     call chmalloc('cvio.src','READCV','TRANSF',3,4,XNSYMM,crl=TRANSF)
     CALL IMFILL(TRANSF,.FALSE.)
     call chmdealloc('cvio.src','READCV','TRANSF',3,4,XNSYMM,crl=TRANSF)
  ENDIF
  CALL set_param('NTOT',NTOT)
  CALL set_param('STEP',ISTEP)
  TIME=DELTA*ISTEP*TIMFAC
  call set_param('TIME',TIME)

  RETURN

! Error handling for a CV that ends prematurely
3000    inquire(IUNIT, name=iunit_filename)
        write(io_error_msg, "(A, I0, A, A, A, I0, A, I0, A)") &
           "READ errorno ", io_status, " for ", trim(iunit_filename), ": ", &
           NFILE, " frames expected but ", NTOT, " frames read"
        call WRNDIE(-5, "<READCV>", io_error_msg)
END SUBROUTINE READCV

INTEGER FUNCTION GCD(IN,JN)
  !
  !     THIS FUNCTION RETURNS THE GREATEST COMMON DENOMINATOR OF IN AND
  !     JN.
  !
  !     Author: Dave States
  !
  use chm_kinds
  implicit none
  INTEGER IN,JN
  !
  INTEGER I,J,K
  !
  I=ABS(IN)
  K=ABS(JN)
  IF(I < K) THEN
     J=I
     I=K
     K=J
  ENDIF
  !  10 IF(K /= 0) THEN
  DO WHILE (K /= 0)
     J=I
     I=K
     K=MOD(J,I)
  ENDDO
  !       GOTO 10
  !     ENDIF
  GCD=I
  RETURN
END FUNCTION GCD

SUBROUTINE WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
     CG,QCG,                              & 
#endif
     NATOM,FREEAT,NFREAT,NPRIV,ISTEP,NDEGF, &
     DELTA,NSAVC,NSTEP,TITLE,NTITLE,IUNCRD,QVEL, &
     QINCT,JCNTRL,DIM4,FDIM)
  !
  !     WRITES A SET OF COORDINATES FOR A SINGLE DYNAMICS STEP. THE FORMAT
  !     FOR THE TRAJECTORY FILE VARIES WITH WHETHER ANY ATOMS ARE FIXED.
  !     ICNTRL(9) STORES THE NUMBER OF FIXED ATOMS WHICH WILL BE ZERO FOR
  !     ALL PREVIOUS TRAJECTORY FILES SO COMPATIBILITY IS ASSURED.
  !
  !     Authors: S. Swaminathan
  !              Robert Bruccoleri
  !
  use chm_kinds
  use dimens_fcm
  use version
  use consta
  use stream
  use image
  use parallel    ! mh050712
  use repdstr     ! mh050712
  !
#if KEY_STRINGM==1 /*  VO stringm */
#if KEY_MULTICOM==1
  use multicom_aux                 
#endif
#if KEY_MPI==1
  use mpi                          
#endif
#endif
  !
#if KEY_FLUCQ==1
  use flucq
#endif 
  use cveloci_mod
  implicit none
  INTEGER NATOM,FREEAT(*),NFREAT,NPRIV,ISTEP,NDEGF
  real(chm_real) X(NATOM),Y(NATOM),Z(NATOM)
#if KEY_CHEQ==1
  real(chm_real) CG(NATOM)
  LOGICAL QCG
#endif 
  real(chm_real) DELTA
  INTEGER NSAVC,NSTEP
  character(len=*) TITLE(*)
  INTEGER NTITLE,IUNCRD,MXYZ
  LOGICAL QVEL
  INTEGER JCNTRL(:)
  LOGICAL QINCT,DIM4
  real(chm_real)  FDIM(NATOM)
  INTEGER ICNTRL(20)
  integer IFILE,NFILE,I,J,K,LNFREAT,M
  LOGICAL ERROR,QCRYS,QDIM4
  REAL(chm_real4) DELTA4
  character(len=4) HDR,HDRC,HDRV
  PARAMETER (HDRC='CORD',HDRV='VELD')
  !
#if KEY_STRINGM==1 /*  string method fix, VO, 1.10 */
  integer :: oldiol     
#endif
  !
  ! save all variables between calls to WRITCV
  SAVE
  !
  IF(IUNCRD < 0) RETURN
#if KEY_REPDSTR==1
  IF(QREPDSTR) THEN
     IF(MYNOD > 0) RETURN
  ELSE
     IF(IOLEV < 0) RETURN
  ENDIF
#else /**/
  IF(IOLEV < 0 &
#if KEY_MULTICOM==1 /*  VO stringm */
  &            .and. (ME_LOCAL.ne.0) &     
#endif
  &            ) RETURN
#endif 

  LNFREAT=NFREAT
#if KEY_CVELOCI==1
  !     This loop is for cases when an atom is fixed but has a constant velocity
  !     This means it moves and it should be present in the trajectory file
  !     so we append it to the FREEAT list.
  ! NOTE: VMD works too!
  K=1
  M=0
  DO I=1,NATOM
     IF(I == FREEAT(K))THEN
        K=K+1
     ELSE
        !          Atom I is fixed, check if it is in FCVEL
        DO J=1,NCVEL
           IF(I == FCVEL(J))THEN
              M=M+1
              !                Append it to FREEAT
              FREEAT(NFREAT+M)=I
              GOTO 9988
           ENDIF
        ENDDO
9988    CONTINUE
     ENDIF
  ENDDO
  LNFREAT=NFREAT+M
#endif 
  IFILE=ISTEP/NSAVC
  NFILE=NSTEP/NSAVC

  IF(QINCT) THEN
     icntrl(1:20)=jcntrl(1:20)
     QCRYS=ICNTRL(11) == 1
     QDIM4=ICNTRL(12) == 1
     IF(QDIM4 .AND. .NOT.DIM4) CALL WRNDIE(-4,'<WRITCV>', &
          '4D write requested, but no 4D data povided')
  ELSE
  !  XTLTYP is 'FAKE' if called from MERGEO w/o crystal being setup
  !  and the input trajectory has crystal information
     QCRYS=(XTLTYP /= '    ')
     QDIM4=DIM4
  ENDIF
  !
  IF (IFILE <= 0) THEN
     !       ERROR IN INITIATION
     IF(WRNLEV >= 2) WRITE(OUTU,44) NATOM,NFREAT,NPRIV,ISTEP, &
          NSAVC,NSTEP,NTITLE,IUNCRD
44   FORMAT(' ** ERROR IN WRITCV INITIATION **',/ &
          '  NATOM,NFREAT,NPRIV,ISTEP,', &
          'NSAVC,NSTEP,NTITLE,IUNCRD'/,12I6)
     CALL DIEWRN(-3)
  ELSE IF (IFILE == 1) THEN
     IF(.NOT.QINCT) THEN
        icntrl(1:20)=0
        ICNTRL(1)=NFILE
        ICNTRL(2)=NPRIV
        ICNTRL(3)=NSAVC
        ICNTRL(4)=NSTEP
        IF(QVEL) ICNTRL(5)=NSAVC
        ICNTRL(8)=NDEGF
        ICNTRL(9)=NATOM-LNFREAT
        IF(QWPICO) DELTA=DELTA*TIMFAC
        DELTA4=DELTA
        ! store bytes of delta 4 in icntrl(10)
        icntrl(10) = transfer(delta4, icntrl(10))
        IF(QCRYS) ICNTRL(11)=1
        IF(DIM4)  ICNTRL(12)=1
#if KEY_CHEQ==1
        IF (QCG) ICNTRL(13)=1                        
#endif
        IF(.NOT.QCHECK) ICNTRL(14)=1  
        IF(VERNUM <= 0 .OR. VERNUM > 10000) &
           CALL WRNDIE(-2,'<WRITCV>', 'Internal error. Illegal VERNUM.')
        ICNTRL(20)=VERNUM
     ENDIF
     !
     IF (QVEL) THEN
        HDR=HDRV
     ELSE
        HDR=HDRC
     ENDIF
     WRITE(IUNCRD) HDR,ICNTRL
     CALL WRTITL(TITLE,NTITLE,IUNCRD,-1)
     WRITE(IUNCRD) natom
     IF (LNFREAT /= NATOM) WRITE(IUNCRD) (FREEAT(I),I=1,LNFREAT)
#if KEY_SINGLE==1
     IF(QCRYS) WRITE(IUNCRD) XTLABC
     WRITE(IUNCRD) X
     WRITE(IUNCRD) Y
     WRITE(IUNCRD) Z
     IF(DIM4) WRITE(IUNCRD) FDIM
#if KEY_CHEQ==1
     IF (QCG) WRITE(IUNCRD) CG                     
#endif
#if KEY_FLUCQ==1
     IF(QFLUC) CALL FQCWRI(IUNCRD)                  
#endif
  ELSE IF (LNFREAT  ==  NATOM) THEN
     ! Write all coordinates
     IF(QCRYS) WRITE(IUNCRD) XTLABC
     WRITE(IUNCRD) X
     WRITE(IUNCRD) Y
     WRITE(IUNCRD) Z
     IF(DIM4) WRITE(IUNCRD) FDIM
#if KEY_CHEQ==1
     IF (QCG) WRITE(IUNCRD) CG                     
#endif
#if KEY_FLUCQ==1
     IF(QFLUC) CALL FQCWRI(IUNCRD)                  
#endif
  ELSE
     ! Write free coordinates only
     IF(QCRYS) WRITE(IUNCRD) XTLABC
     WRITE(IUNCRD) (X(FREEAT(I)),I=1,LNFREAT)
     WRITE(IUNCRD) (Y(FREEAT(I)),I=1,LNFREAT)
     WRITE(IUNCRD) (Z(FREEAT(I)),I=1,LNFREAT)
     IF(DIM4) WRITE(IUNCRD) (FDIM(FREEAT(I)),I=1,LNFREAT)
#if KEY_CHEQ==1
     IF (QCG) WRITE(IUNCRD) CG                    
#endif
#if KEY_FLUCQ==1
     IF(QFLUC) CALL FQCWRI(IUNCRD)                 
#endif
#else /**/
     IF(QCRYS) WRITE(IUNCRD) DBLE(XTLABC)
     WRITE(IUNCRD) (SNGL(X(I)),I=1,NATOM)
     WRITE(IUNCRD) (SNGL(Y(I)),I=1,NATOM)
     WRITE(IUNCRD) (SNGL(Z(I)),I=1,NATOM)
     IF(DIM4) WRITE(IUNCRD) (SNGL(FDIM(I)),I=1,NATOM)
#if KEY_CHEQ==1
     IF (QCG) WRITE(IUNCRD) (SNGL(CG(I)),I=1,NATOM)    
#endif
#if KEY_FLUCQ==1
     IF(QFLUC) CALL FQCWRI(IUNCRD)                      
#endif
  ELSE IF (LNFREAT  ==  NATOM) THEN
     ! Write all coordinates
     IF(QCRYS) WRITE(IUNCRD) DBLE(XTLABC)
     WRITE(IUNCRD) (SNGL(X(I)),I=1,NATOM)
     WRITE(IUNCRD) (SNGL(Y(I)),I=1,NATOM)
     WRITE(IUNCRD) (SNGL(Z(I)),I=1,NATOM)
     IF(DIM4) WRITE(IUNCRD) (SNGL(FDIM(I)),I=1,NATOM)
#if KEY_CHEQ==1
     IF (QCG) WRITE(IUNCRD) (SNGL(CG(I)),I=1,NATOM)    
#endif
#if KEY_FLUCQ==1
     IF(QFLUC) CALL FQCWRI(IUNCRD)                      
#endif
  ELSE
     ! Write free coordinates only
     IF(QCRYS) WRITE(IUNCRD) DBLE(XTLABC)
     WRITE(IUNCRD) (SNGL(X(FREEAT(I))),I=1,LNFREAT)
     WRITE(IUNCRD) (SNGL(Y(FREEAT(I))),I=1,LNFREAT)
     WRITE(IUNCRD) (SNGL(Z(FREEAT(I))),I=1,LNFREAT)
     IF(DIM4) WRITE(IUNCRD) (SNGL(FDIM(FREEAT(I))),I=1,LNFREAT)
#if KEY_CHEQ==1
     IF (QCG) WRITE(IUNCRD) (SNGL(CG(I)),I=1,NATOM)    
#endif

#if KEY_FLUCQ==1
     IF(QFLUC) CALL FQCWRI(IUNCRD)                      
#endif
#endif 
  ENDIF
  !
  !++  LN MOD /APR 90
  !     Make sure everything is put on disk (which is needed on some
  !     machines in case of a job crash
  CALL SAVEIT(IUNCRD)
  !--
  IF(IFILE < NFILE) RETURN
  IF(PRNLEV >= 2) THEN
     IF (QVEL) THEN
        WRITE(OUTU,102) NSTEP/NSAVC,ICNTRL(2),NSAVC,IUNCRD
        !C??      WRITE(OUTU,102) (ICNTRL(I),I=1,3),IUNCRD
     ELSE
        WRITE(OUTU,101) (ICNTRL(I),I=1,3),IUNCRD
     ENDIF
102  FORMAT(/2X,I5,'   VELOCITY SETS STARTING FROM',/, &
          5X,'STEP NO ',I8,'   FOR EVERY ',I5,'  STEPS',/, &
          5X,'WRITTEN ON UNIT',I5,/)
101  FORMAT(/2X,I5,'   COORDINATE SETS STARTING FROM',/, &
          5X,'STEP NO ',I8,'   FOR EVERY ',I5,'  STEPS',/, &
          5X,'WRITTEN ON UNIT',I5,/)
  ENDIF

#if KEY_STRINGM==1 /*  VO : stringm fix */
  oldiol=iolev
  if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) iolev=0
#endif
  CALL VCLOSE(IUNCRD,'KEEP',ERROR)
#if KEY_STRINGM==1 /*  VO */
  iolev=oldiol                        
#endif
  !
  RETURN
END SUBROUTINE WRITCV

SUBROUTINE DYNFOR
  !
  ! reads in specified sections of a trajectory on files
  ! FILEs and writes them onto a formatted file (i.e. for transport
  ! between machines)
  !
  !  DYNAmics FORMat   FIRStunit <int>  NUNIt <int> BEGIn <int>
  !                    SKIP <int>  STOP <int>       OUTPut <int>
  !                    OFFSet <int>  SCALe <int>    MODE <FORTRAN-FORMAT>
  !
  ! The defaults for OFFSet, SCALe and MODE are:
  !     OFFSet=600 SCALE=10000 and MODE=12Z6
  !
  ! Axel Brunger, 16-FEB-83
  ! =======================
  !
  use chm_kinds
  use dimens_fcm
  use number
  use psf
  use comand
  use coord
  use ctitla
  use stream
  use string
  use image
  use version
  use memory

  implicit none

#if KEY_CHEQ==1
! BIOVIA Code Start : Bug fix
!  INTEGER IHAS_CHGS
! BIOVIA Code End
  LOGICAL QCG_LOCAL
#endif 
  ! local
  INTEGER I, NFILE, ISTATS, NDEGF, IUNIT
  INTEGER NUNIT, NFIRST, OUTPUT, SKIP, BEGIN, STOP, ISTEP
  real(chm_real) DELTA
  LOGICAL QFIRST
  INTEGER NFREAT, NSAVV, PASSED, NSET
  character(len=4) SFORM
  character(len=6) FORM
  real(chm_real) OFFSET, SCALE
  character(len=4) HDRC,HDRD,HDR
  PARAMETER (HDRC='CORD',HDRD='VELD')
  INTEGER ICNTRL(20)
  integer int8traju
  INTEGER JCNTRL(20)
  ! BIOVIA Code Start : Bug fix
!  REAL(chm_real4) DELTA4
  ! BIOVIA Code End
  real(chm_real4),allocatable,dimension(:) :: TEMP
  integer,allocatable,dimension(:) :: FREEAT

  !
  ! begin
  call chmalloc('cvio.src','DYNFOR','TEMP',  NATOM,cr4=TEMP)
  call chmalloc('cvio.src','DYNFOR','FREEAT',NATOM,intg=FREEAT)
  !
  ! parsing
  NFIRST=GTRMI(COMLYN,COMLEN,'FIRS',-1)
  NUNIT=GTRMI(COMLYN,COMLEN,'NUNI',1)
  BEGIN=GTRMI(COMLYN,COMLEN,'BEGI',0)
  STOP=GTRMI(COMLYN,COMLEN,'STOP',0)
  SKIP=GTRMI(COMLYN,COMLEN,'SKIP',1)
  OUTPUT=GTRMI(COMLYN,COMLEN,'OUTP',6)
  SFORM=GTRMA(COMLYN,COMLEN,'MODE')
  IF (SFORM == ' ') SFORM='12Z6'
  SCALE=10000.0D0
  SCALE=GTRMF(COMLYN,COMLEN,'SCAL',SCALE)
  OFFSET=800.0D0
  OFFSET=GTRMF(COMLYN,COMLEN,'OFFS',OFFSET)
  !
  !
  NFREAT=NATOM
  DO I=1,NATOM
     FREEAT(I)=I
  ENDDO
  !
  FORM='('//SFORM//')'
  PASSED=0
  IUNIT=NFIRST
  ISTATS=1
  QFIRST=.TRUE.
  !
  if (reallow) then
     REWIND(IUNIT)
     int8traju=0  
     READ(IUNIT)HDR,ICNTRL
     REWIND(IUNIT)

     jcntrl(1:20)=icntrl(1:20)
#if KEY_CHEQ==1
     QCG_LOCAL = icntrl(13) /= 0                        
#endif
  else  
#if KEY_CHEQ==1
     QCG_LOCAL=.TRUE.        
#endif
  endif
400 IF(ISTATS >= 0) THEN
     CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
          CG,QCG_LOCAL,                         & 
#endif
          TEMP,NATOM,FREEAT,NFREAT,NFIRST,NUNIT,IUNIT, &
          NFILE,ISTEP,ISTATS,NDEGF,DELTA, &
          BEGIN,STOP,SKIP,NSAVV,HDRC,HDRD, &
          TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
     !
     !
     IF (QFIRST) THEN
        icntrl(1:20)=icntsv(1:20)
        jcntrl(1:20)=icntsv(1:20)
#if KEY_CHEQ==1
        QCG_LOCAL = icntrl(13) /= 0      
#endif
     ENDIF

     !
     PASSED=PASSED+1
     ! Insulate the writes from the slave nodes
     IF(IOLEV  >  0)THEN
        IF (QFIRST) THEN
           QFIRST=.FALSE.
           !
           IF(PRNLEV >= 2) CALL WRTITL(TITLEB,NTITLB,OUTU,+2)
           WRITE(OUTPUT,'(/2I8,A)') NTITLB,VERNUM,' !NTITLE  VERNUM'
           WRITE(OUTPUT,'(A)') (TITLEB(I),I=1,NTITLB)
           WRITE(OUTPUT,'(5Z16)')JCNTRL
           !
           IF (STOP > 0) THEN
              NSET=(STOP-BEGIN)/SKIP+1
           ELSE
              NSET=NFILE
           ENDIF
           WRITE(OUTPUT,'(1X,A4,1X,4I10,G14.6)') &
                HDRC, BEGIN, SKIP, NSET, NATOM, DELTA
           WRITE(OUTPUT,'(G14.6,G14.6,A)') SCALE, OFFSET, SFORM
           WRITE(OUTPUT,'(10I8)') NFREAT, (FREEAT(I),I=1,NFREAT)
           IF(ICNTRL(11) == 1)WRITE(OUTPUT,'(3D24.16)') XTLABC
           WRITE(OUTPUT,'(A)') HDRC
           WRITE(OUTPUT,FORM) &
                (INT((X(I)+OFFSET)*SCALE), &
                INT((Y(I)+OFFSET)*SCALE), &
                INT((Z(I)+OFFSET)*SCALE), I=1,NATOM)
#if KEY_CHEQ==1
           IF (QCG_LOCAL) WRITE(OUTPUT,FORM) &
                (INT((CG(I)+OFFSET)*SCALE), I=1,NATOM)
#endif 
        ELSE
           IF(ICNTRL(11) == 1)WRITE(OUTPUT,'(3D24.16)') XTLABC
           WRITE(OUTPUT,'(A)') HDRC
           WRITE(OUTPUT,FORM) &
                (INT((X(FREEAT(I))+OFFSET)*SCALE), &
                INT((Y(FREEAT(I))+OFFSET)*SCALE), &
                INT((Z(FREEAT(I))+OFFSET)*SCALE), I=1,NFREAT)
#if KEY_CHEQ==1
           IF (QCG_LOCAL) WRITE(OUTPUT,FORM) &
                (INT((CG(I)+OFFSET)*SCALE), I=1,NATOM)
#endif 
        ENDIF
     ENDIF
     GOTO 400
  ENDIF
  !
  IF(PRNLEV >= 2) THEN
     WRITE(OUTU,'(A)')' DYNFOR: last coordinate set disposed to main'
     WRITE(OUTU,'(A,I8,A)') ' DYNFOR: ',PASSED, &
          ' coordinate sets copied'
  ENDIF
  !
  call chmdealloc('cvio.src','DYNFOR','TEMP',  NATOM,cr4=TEMP)
  call chmdealloc('cvio.src','DYNFOR','FREEAT',NATOM,intg=FREEAT)
  RETURN
END SUBROUTINE DYNFOR

SUBROUTINE DYNUFO
  !
  ! converts portable dynamics trajectory into binary format
  !
  !    DYNAmics UNFOrmat  INPUt <int>  OUTPut <int>
  !
  ! Axel Brunger, 16-FEB-83
  ! =======================

#if KEY_CHEQ==1
  use cheq,only: qcg  
#endif

  use chm_kinds
  use dimens_fcm
  use number
  use comand
  use psf
  use coord
  use ctitla
  use stream
  use string
  use image
  use version
  use memory
  implicit none
#if KEY_CHEQ==1
  INTEGER CCG
  LOGICAL QHAS_CHGS
#endif 
  ! local
  LOGICAL QFIRST
  INTEGER NSET, ISET, NATOMX, INPUT, OUTPUT, ISTART, NSAVC, I
  INTEGER NFREAT, PASSED
  real(chm_real) DELTA
  character(len=4) HDR, SFORM
  character(len=6) FORM
  real(chm_real) OFFSET, SCALE
  character(len=80) LINE
  INTEGER ICNTRL(20),VERN

  LOGICAL QINCT
  integer jcntrl(20)
  integer,allocatable,dimension(:) :: FREEAT,IX,IY,IZ
  ! begin
  !
  ! parsing
  OUTPUT=GTRMI(COMLYN,COMLEN,'OUTP',-1)
  INPUT=GTRMI(COMLYN,COMLEN,'INPU',-1)
  IF(IOLEV < 0) RETURN
  !
  IF(OUTPUT < 0) THEN
     CALL WRNDIE(-3,'<DYNUFO>','No OUTPut unit specified')
     RETURN
  ENDIF
  IF(INPUT < 0) THEN
     CALL WRNDIE(-3,'<DYNUFO>','No INPUt unit specified')
     RETURN
  ENDIF
  !
  ! Note: Old format is no longer supported
  ! As with dynamics restart files, FORMAT and UNFORMAT using
  ! same CHARMM version  - BRB 7/23/98
  !
  READ(INPUT,'(/2I8)',ERR=910) NTITLB,VERN
  IF(VERN < 26) GOTO 910  ! don't accept an old file
  !
  READ(INPUT,'(A)',ERR=920) (TITLEB(I),I=1,NTITLB)
  IF(PRNLEV >= 2) CALL WRTITL(TITLEB,NTITLB,OUTU,+1)
  READ(INPUT,'(5Z16)') JCNTRL
  icntrl(1:20)=jcntrl(1:20)
#if KEY_CHEQ==1
  QHAS_CHGS = jcntrl(13) /= 0
  write(outu,*)"<DYNUFO> QHAS_CHGS: ",QHAS_CHGS,qcg
#else /**/
  if(jcntrl(13) /= 0)then
     write(outu,913)
913  format("This trajectory has CHEQ data but charmm not", &
          " compiled with CHEQ")
     call wrndie(-4,'<READCV>', &
          'CHEQ not compiled in, traj written with CHEQ')
  endif
#endif 
  !
  QFIRST=.TRUE.
  QINCT=.TRUE.
  PASSED=0
  !
  READ(INPUT,'(1X,A4,1X,4I10,G14.6)',ERR=930) &
       HDR, ISTART, NSAVC, NSET, NATOMX, DELTA
  IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A4,1X,4I7,G14.6)') &
       HDR, ISTART, NSAVC, NSET, NATOMX, DELTA
  IF (NATOM /= NATOMX) THEN
     CALL WRNDIE(-1,'<DYNUFO>','PSF - NATOM mismatch')
  ENDIF
  READ(INPUT,'(G14.6,G14.6,A)',ERR=940) SCALE, OFFSET, SFORM
  FORM='('//SFORM//')'
  !
  call chmalloc('cvio.src','DYNUFO','FREEAT',NATOM,intg=FREEAT)
  call chmalloc('cvio.src','DYNUFO','IX',NATOM,intg=IX)
  call chmalloc('cvio.src','DYNUFO','IY',NATOM,intg=IY)
  call chmalloc('cvio.src','DYNUFO','IZ',NATOM,intg=IZ)

  DO ISET=1,NSET
     IF (QFIRST) THEN
        QFIRST=.FALSE.
        READ(INPUT,'(10I8)',END=9999,ERR=950) &
             NFREAT, (FREEAT(I),I=1,NFREAT)
        IF(ICNTRL(11) == 1) &
             READ(INPUT,'(3D24.16)',END=9999,ERR=980) XTLABC
        READ(INPUT,'(A)',END=9999,ERR=960) HDR
        READ(INPUT,FORM,END=9999,ERR=970) &
             (IX(I),IY(I),IZ(I), I=1,NATOM)
        X(1:natom)=IX(1:natom)/SCALE-OFFSET
        Y(1:natom)=IY(1:natom)/SCALE-OFFSET
        Z(1:natom)=IZ(1:natom)/SCALE-OFFSET
#if KEY_CHEQ==1
        if(QHAS_CHGS)then
           READ(INPUT,FORM,END=9999,ERR=970) &
                (IX(I), I=1,NATOM)
           !     Borrow IX array here.
           DO I=1,NATOM
              CG(I)=IX(I)/SCALE-OFFSET
           ENDDO
        ENDIF
#endif 
     ELSE
        IF(ICNTRL(11) == 1)  &
             READ(INPUT,'(3D24.16)',END=9999,ERR=980) XTLABC
        READ(INPUT,'(A)',END=9999,ERR=960) HDR
        READ(INPUT,FORM,END=9999,ERR=970) &
             (IX(FREEAT(I)),IY(FREEAT(I)), &
             IZ(FREEAT(I)), I=1,NFREAT)
        DO I=1,NFREAT
           X(FREEAT(I))=IX(FREEAT(I))/SCALE-OFFSET
           Y(FREEAT(I))=IY(FREEAT(I))/SCALE-OFFSET
           Z(FREEAT(I))=IZ(FREEAT(I))/SCALE-OFFSET
        ENDDO
#if KEY_CHEQ==1
        if(QHAS_CHGS)then
           READ(INPUT,FORM,END=9999,ERR=970) &
                (IX(I), I=1,NATOM)
           !     Borrow IX array here.
           DO I=1,NATOM
              CG(I)=IX(I)/SCALE-OFFSET
           ENDDO
        ENDIF
#endif 
     ENDIF
     !
     PASSED=PASSED+1
     !
     CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
          CG,QCG,                                  & 
#endif
          NATOM,FREEAT,NFREAT,ISTART,PASSED*NSAVC,0, &
          DELTA,NSAVC,NSET*NSAVC,TITLEB,NTITLB,OUTPUT, &
          .FALSE.,QINCT,ICNTRL,.FALSE., (/ ZERO /))
     !
  ENDDO
  !     end of do loop : DO ISET=1,NSET
  !
  IF(PRNLEV >= 2) THEN
     WRITE(OUTU,'(A,I8,A)') ' DYNUFO: ',PASSED, &
          ' coordinate sets copied'
     WRITE(OUTU,'(A)') ' DYNUFO: main coordinates set to last point'
  ENDIF
  !
  call chmdealloc('cvio.src','DYNUFO','FREEAT',NATOM,intg=FREEAT)
  call chmdealloc('cvio.src','DYNUFO','IX',NATOM,intg=IX)
  call chmdealloc('cvio.src','DYNUFO','IY',NATOM,intg=IY)
  call chmdealloc('cvio.src','DYNUFO','IZ',NATOM,intg=IZ)
  RETURN
910 IF(WRNLEV > 2) WRITE(OUTU,'(A)') &
       ' ERROR::DYNUFO:CANT READ NTITLE VALUE OR VERSION NUMBER'
  GOTO 1000
920 IF(WRNLEV > 2) WRITE(OUTU,'(A)') ' ERROR::DYNUFO:CANT READ TITLE'
  GOTO 1000
930 IF(WRNLEV > 2) WRITE(OUTU,'(A)') &
       ' ERROR::DYNUFO:CANT READ HEADER AND INTEGER VALUES'
  GOTO 1000
940 IF(WRNLEV > 2) WRITE(OUTU,'(A)') &
       ' ERROR::DYNUFO:CANT READ FLOATING VALUES'
  GOTO 1000
950 IF(WRNLEV > 2) WRITE(OUTU,'(A)') &
       ' ERROR::DYNUFO:CANT READ FREE ATOMS'
  GOTO 1000
960 IF(WRNLEV > 2) WRITE(OUTU,'(A)') &
       ' ERROR::DYNUFO:CANT READ HEADER'
  GOTO 1000
970 IF(WRNLEV > 2) WRITE(OUTU,'(A)') &
       ' ERROR::DYNUFO:CANT READ COORDINATES'
  GOTO 1000
980 IF(WRNLEV > 2) WRITE(OUTU,'(A)') &
       ' ERROR::DYNUFO:CANT READ CELL DATA'
  !
1000 BACKSPACE(UNIT=INPUT)
  READ(INPUT,'(A80)') LINE
  IF(WRNLEV > 2) WRITE(OUTU,'(A80)') LINE
  CALL WRNDIE(-3,'<DYNUFO>','Error during read')
  call chmdealloc('cvio.src','DYNUFO','FREEAT',NATOM,intg=FREEAT)
  call chmdealloc('cvio.src','DYNUFO','IX',NATOM,intg=IX)
  call chmdealloc('cvio.src','DYNUFO','IY',NATOM,intg=IY)
  call chmdealloc('cvio.src','DYNUFO','IZ',NATOM,intg=IZ)
  RETURN
9999 CALL WRNDIE(-1,'<DYNUFO>','End of file during read')
  call chmdealloc('cvio.src','DYNUFO','FREEAT',NATOM,intg=FREEAT)
  call chmdealloc('cvio.src','DYNUFO','IX',NATOM,intg=IX)
  call chmdealloc('cvio.src','DYNUFO','IY',NATOM,intg=IY)
  call chmdealloc('cvio.src','DYNUFO','IZ',NATOM,intg=IZ)
  RETURN
END SUBROUTINE DYNUFO

SUBROUTINE GTICNT(IUNIT,HDR,ICNTRL,QPRINT,QREWIND,QCOMM)
  !
  ! Extract ICNTRL information from dynamics trajectory file
  ! on unit IUNIT (which is REWOUND).
  !  HDR      type of trajectory (CORD or VELD)
  !  ICNTRL   the whole ICNTRL array
  !
  !  Lennart Nilsson, Karolinska Institutet, NOV-87
  !  Print and substitution added - BRB, NIH, MAR-98
  !  QPRINT false to suppress printing, QREWIND tells to rewind.
  !  QCOMM flas to suppress parallel communication
  use chm_kinds
  use consta
  use stream
  use machio,only:vopen
#if KEY_MULTICOM==1 /*  VO stringm */
  use multicom_aux
#endif
  use param_store, only: set_param
  use parallel,only:psnd4,psnd8
  implicit none

  INTEGER ICNTRL(20),IUNIT
  LOGICAL, OPTIONAL:: QPRINT,QREWIND,QCOMM
  integer int8traju
  character(len=4) HDR 
  character(len=16) conv
  character(len=256) fname
  !
  INTEGER I
  real(chm_real) DELTA
  REAL(chm_real4) DELTA4
  LOGICAL LREWIND,LPRINT,LCOMM,QERR
  !
  IF(PRESENT(QREWIND))THEN
     LREWIND=QREWIND
  ELSE
     LREWIND=.FALSE.
  ENDIF
  IF(PRESENT(QPRINT))THEN
     LPRINT=QPRINT
  ELSE
     LPRINT =.FALSE.
  ENDIF
  IF(PRESENT(QCOMM))THEN
     LCOMM=QCOMM
  ELSE
     LCOMM=.TRUE.
  ENDIF
  IF( .NOT. REALLOW) LREWIND=.FALSE.  

  IF(IOLEV > 0                      &
#if KEY_MULTICOM==1 /*  VO stringm */
 &             .or. (ME_LOCAL.eq.0) &         
#endif
 &              ) THEN
     int8traju=0     !assume 32 bit ints
     IF (LREWIND) REWIND(IUNIT)
     READ(IUNIT) HDR,ICNTRL
#if KEY_NOCONVERT==0 && __PGI == 0 && KEY_G95==0 && KEY_PATHSCALE==0
     ! simple check of endianness based on value of the version number in ICNTRL(20),
     ! which should be a positive integer, assumed to be less than 10000; this means that
     ! in a normal 32bit integer representaion either the first two
     ! or the last two bytes are all zero.
     ! The CONVERT specifier to OPEN/INQUIRE is supported by gfortran and ifort, 
     ! but not by xlf, which apparently simply ignores it.
     IF(ICNTRL(20) < 0 .OR. ICNTRL(20) > 10000)THEN
     ! try to open file with opposite endianness
        INQUIRE(IUNIT,CONVERT=CONV,NAME=FNAME)
        SELECT CASE(CONV)
        CASE('LITTLE_ENDIAN')
           CONV='BIG_ENDIAN'
        CASE('BIG_ENDIAN')
           CONV='LITTLE_ENDIAN'
        CASE('NATIVE')
           IF(reallow)THEN    
              CLOSE(IUNIT)
              CONV='BIG_ENDIAN'
              CALL VOPEN(IUNIT,FNAME,'UNFORMATTED','READ',QERR,0,CONV) 
              READ(IUNIT) HDR,ICNTRL
              IF(ICNTRL(20) < 0 .OR. ICNTRL(20) > 10000) CONV='LITTLE_ENDIAN'
           ENDIF               
        END SELECT
     ! try again with the opposite endianness
        IF(reallow)THEN    
          CLOSE(IUNIT)
          CALL VOPEN(IUNIT,FNAME,'UNFORMATTED','READ',QERR,0,CONV) 
          READ(IUNIT) HDR,ICNTRL 
          IF(ICNTRL(20) < 0 .OR. ICNTRL(20) > 10000) THEN
             CALL WRNDIE(-2,'<READCV>', 'Apparent problem with endianness')
          ENDIF
        ENDIF               
     ENDIF
     ! now the file is either opened with correct endianness,
     ! or we are here with a -2 warning,
     ! or rewind was not allowed and we have the incorrect endianness,
     ! which seems absurd.
#endif
     ! store bytes of icntrl(10) in delta4
     delta4 = transfer(icntrl(10), delta4)
  ENDIF

  IF (LREWIND) REWIND(IUNIT)
  DELTA=DELTA4*TIMFAC ! switch from AKMA time to picoseconds - BRB
  !
#if KEY_PARALLEL==1
  IF(LCOMM)THEN
    CALL PSNDC(HDR,1)
    CALL PSND4(ICNTRL,20)
    CALL PSND8(delta,1)
  ENDIF
#endif 

  ! set some substitution parameters
  CALL set_param('NFILE',ICNTRL(1))
  CALL set_param('START',ICNTRL(2))
  CALL set_param('SKIP', ICNTRL(3))
  CALL set_param('NSTEP',ICNTRL(4))
  CALL set_param('NDEGF',ICNTRL(8))
  call set_param('DELTA',DELTA)
  !

  IF(PRNLEV < 2 .OR. (.NOT. LPRINT .AND. PRNLEV <6)) RETURN
  !
     WRITE(OUTU,26) IUNIT,(ICNTRL(I),I=1,4)
26   FORMAT(/' READING TRAJECTORY FROM UNIT',I4,/ &
          '   NUMBER OF COORDINATE SETS IN FILE:',I12,/ &
          '   NUMBER OF PREVIOUS DYNAMICS STEPS:',I12,/ &
          '   FREQUENCY FOR SAVING COORDINATES: ',I12,/ &
          '   NUMBER OF STEPS FOR CREATION RUN: ',I12,/)
     WRITE(OUTU,27) ICNTRL(8),ICNTRL(9),DELTA
27   FORMAT( &
          '   NUMBER OF DEGREES OF FREEDOM:     ',I12,/ &
          '   NUMBER OF FIXED ATOMS:            ',I12,/ &
          '   THE INTEGRATION TIME STEP (PS):',F12.4)
     IF(ICNTRL(11) == 1) THEN
        WRITE(OUTU,28) &
             '   THE FILE CONTAINS CRYSTAL UNIT CELL DATA'
     ELSE
        WRITE(OUTU,28) &
             '   THE FILE DOES NOT CONTAIN CRYSTAL UNIT CELL DATA'
     ENDIF
     IF(ICNTRL(12) == 1) THEN
        WRITE(OUTU,28) &
             '   THE FILE CONTAINS 4-DIMENSIONAL COORDINATE DATA'
     ELSE
        WRITE(OUTU,28) &
             '   THE FILE DOES NOT CONTAIN 4-D DATA'
     ENDIF
     IF(ICNTRL(14) == 1) THEN
        WRITE(OUTU,28) ' '
        WRITE(OUTU,28) '   THE FILE MAY CONTAIN NON-CONTIGUOUS COORDINATE DATA.'
        WRITE(OUTU,28) '   HEADER AND OTHER META INFORMATION MAY BE INCORRECT,' 
        WRITE(OUTU,28) '   AND THE FILE INAPPROPRIATE FOR TIMEDEPENDENT ANALYSIS'
        WRITE(OUTU,28) ' '
28   FORMAT(A)
  ENDIF
  RETURN
END SUBROUTINE GTICNT

SUBROUTINE TRJSPC(COMLYN,COMLEN, &
     NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP)
  !-----------------------------------------------------------------------
  !     Parse trajectory specification:
  !     NUNIT = number of dynamics files to process (1)
  !     FIRSTU = first unit number                  (-1)
  !     NBEGIN = first step number                  (0)
  !     NSKIP =  step increment                     (1)
  !     NSTOP = last step number                    (0)
  !
  !     Lennart Nilsson, Karolinska Institutet, NOV-87
  !     Lennart Nilsson, Karolinska Institutet, MAR-98: 
  !           changed FIRSTU default to -1
  !     LNI March 2011, parse NOCHeck to set QCHECK=.FALSE. 
  !         to turn off strict trajectory checking
  !
  ! Additions for timestep field fixes, Rick Venable, NIH 2022
  !     TSET = set timestep to value in ps, converted to AKMA
  !            initialized to -1.0, input must be > 0.0
  !     RPICO = read timestep in ps, convert to AKMA (.false.)
  !     WPICO = convert AKMA to ps for writing (.false.)
  !
  use string, only:gtrmi,gtrmf,indxa
  character(len=*) COMLYN
  INTEGER COMLEN,NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP
  !
  !
  NUNIT= GTRMI(COMLYN,COMLEN,'NUNI',1)
  FIRSTU= GTRMI(COMLYN,COMLEN,'FIRS',-1)
  NBEGIN= GTRMI(COMLYN,COMLEN,'BEGI',0)
  NSKIP= GTRMI(COMLYN,COMLEN,'SKIP',-1)
  IF(NSKIP  ==  -1) NSKIP=GTRMI(COMLYN,COMLEN,'NSKI',1)
  NSTOP= GTRMI(COMLYN,COMLEN,'STOP',0)
  TSET=GTRMF(COMLYN,COMLEN,'TSET',-1.0)
  QCHECK=INDXA(COMLYN,COMLEN,'NOCH') == 0
  ! flags to read and/or write timestep field in picosec
  QRPICO=.FALSE.
  QWPICO=.FALSE.
  QRPICO=INDXA(COMLYN,COMLEN,'RPIC') > 0
  ! QWPICO=INDXA(COMLYN,COMLEN,'WPIC') > 0
  RETURN
END SUBROUTINE TRJSPC
LOGICAL FUNCTION GET_TRAJ_CHK()
  GET_TRAJ_CHK=QCHECK
RETURN
END FUNCTION GET_TRAJ_CHK
SUBROUTINE SET_TRAJ_CHK(BOOLEAN)
LOGICAL BOOLEAN
  QCHECK=BOOLEAN
RETURN
END SUBROUTINE SET_TRAJ_CHK

! writes frame to a simplified trajectory file opened
! for DIRECT I/O
subroutine writsimp(x,y,z,len,unum,istep,nstep,nsavc)
   real(chm_real),intent(in) :: x(*),y(*),z(*)
   integer,intent(in)        :: len,unum,istep,nstep,nsavc

! BIOVIA Code Start : Bug fix
   integer                   :: i,ifile,nfile
   logical                   :: error
   integer,save              :: recno = 0
! BIOVIA Code End

   write(unum,rec=recno+1) (sngl(x(i)),i=1,len)
   write(unum,rec=recno+2) (sngl(y(i)),i=1,len)
   write(unum,rec=recno+3) (sngl(z(i)),i=1,len)
   recno = recno + 3

   ifile=istep/nsavc
   nfile=nstep/nsavc

   if(ifile < nfile) return
   call vclose(unum,'KEEP',error)
   recno = 0

end subroutine writsimp

end module cvio

