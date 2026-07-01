module traj_mod
  use chm_kinds
  use chm_types
  implicit none
  !CHARMM Element source/fcm/traj.fcm 1.1
  !
  !  Common file to hold temporary trajectory data
  !
  !  Note: this file now contains only saved local variables. - BRB
  !
  ! VARAIBLES FOR INPUT TRAJECTORY FILE
  !
  !  NFREAT  - NUMBER OF ATOMS FREE TO MOVE
  !  FIRSTU  - I/O UNIT FOR FIRST READ FILE
  !  NUNIT   - NUMBER OF TRAJECTORY READ FILES
  !  UNIT    - CURRENT READ UNIT
  !  IFILE   - NUMBER OF FRAMES IN THE FIRST READ FILE
  !  ISTEP   - CURRENT READ STEP NUMBER (GLOBAL BY SKIP)
  !  ISTATS  - CURRENT READ STATUS (-1 MEANS ALL DONE)
  !  NBEGN   - FIRST STEP TO READ (0=READ FROM BEGINNING)
  !  NSTOP   - LAST STEP TO READ  (0=READ TO THE END)
  !  SKIP    - READING STEP INTERVAL (GLOBAL)
  !  QREAD   - FLAG FOR ACTIVE READING
  !
  !               *             *          *
  INTEGER NFREAT,FIRSTU,NUNIT,UNIT,IFILE
  !               *     *
  INTEGER ISTEP,ISTATS
  !                          *
  INTEGER NBEGN,NSTOP,SKIP
  !                                    *
  integer,allocatable,dimension(:) :: FREEAT
  REAL(CHM_REAL4),ALLOCATABLE,DIMENSION(:):: TEMP
#if KEY_CHEQ==1
  INTEGER         FREECG,CGNEW              
#endif
  !
  LOGICAL QREAD
  !
  ! VARIABLES FOR OUTPUT TRAJECTORY FILE
  !
  !  NPRIV   - STEP NUMBER OF FIRST FRAME TO WRITE
  !  OUTSTP  - CURRENT OUTPUT STEP NUMBER (LOCAL TO FILE)
  !  NSAVC   - OUTPUT FRAME STEP INCREMENT (GLOBAL)
  !  NSTEP   - TOTAL NUMBER OF STEPS PER OUTPUT FILE
  !  TRAJU   - CURRENT OUTPUT UNIT NUMBER
  !  NTRAJU  - NUMBER OF OUTPUT FILES (OR LAST ONE)
  !  NFROUT  - NUMBER OF FREE ATOMS IN OUTPUT TRAJECTORY
  !  NFILE   - TOTAL NUMBER OF FRAMES PER OUTPUT FILE
  !  QEXPND  - FLAG TO REMOVE ALL FIXED ATOMS FROM THE OUTPUT
  !  QWRITE  - FLAG FOR ACTIVE WRITING
  !
  INTEGER NPRIV,OUTSTP,NSAVC,NSTEP,TRAJU,NTRAJU
  INTEGER NFROUT,NFILE
  LOGICAL QEXPND,QWRITE
  !
  ! VARIABLES UNCHANGED, BUT IN BOTH
  !
  INTEGER NDEGF,RNSAVV
  real(chm_real) DELTA,WDELTA
  !
  !
    CHARACTER(len=4),private :: COORHD,VELHD,HDR
    INTEGER,private :: IUNIT,ICNTRL(20)
    LOGICAL,private :: QVEL
    DATA QVEL/.FALSE./
    !
    DATA COORHD,VELHD/'CORD','VELD'/

  
contains
  
  !CHARMM Element source/io/trajio.src 1.1
  SUBROUTINE TRAJIO
    !
    !     Merges or breaks up a dynamics coordinate or velocity trajectory
    !     into different numbers of units. The syntax of the command is
    !
    !     TRAJectory {read-spec} {write-spec}
    !
    ! read-spec:==  [IREAd unit]  [NREAd int]  [SKIP int]
    !
    ! write-spec:== [IWRIte unit] [NWRIte int] [NFILE int] [EXPAnd]
    !                 [NOTHer int]    [DELTa real]  [SKIP int]
    !
    !  IREAd  - first unit to read from  (default: do not read)
    !  NREAd  - number of units to read from (default:1)
    !  SKIP   - skip value for both reading and writing (default:1)
    !  IWRIte - first unit to write to (default: do not write)
    !  NWRIte - number of units to write to (default:1)
    !  NFILe  - number of frames on each output file (default: total)
    !  EXPAnd - flag to free fixed atoms in copying (only if reading)
    !  NOTHer - number of frames in previous files (if not reading) (d:0)
    !  DELTa  - output delta value (if not reading) (default:0.001)
    !
#if KEY_CHEQ==1
  use cheq,only:qcg,cpcgimg    
#endif
#if KEY_FLUCQ==1
  use flucqm,only:fqcfor       
#endif
  use memory
  use chm_kinds
  use dimens_fcm
  use comand
  use ctitla
  use dynio
  use cvio,only:trjspc,gticnt
  use consta,only: timfac
  use exfunc
  use psf
  use stream
  use string
  use bases_fcm
#if KEY_FLUCQ==1
  use flucq      
#endif
  use cstran_mod,only:mkfrat
  use number

#if KEY_BLOCK==1
!ldm
  use lambdam,only:msld_preprocessld
!    integer :: iprint
!    real(chm_real) :: itemp, cutlo, cuthi
! LDM
#endif

  ! kluge to enable skip of rdtitle below
  ! Local new variable qread_title
  logical qnotitle

    qread=.false.
    qwrite=.false.

#if KEY_BLOCK==1
!ldm
    ! process multi-site lambda-dynamics options
    IF(INDXA(COMLYN,COMLEN,'LAMB').GT.0) THEN
       call msld_preprocessld(comlyn,comlen)
       return
    endif
! LDM
#endif 
    ! process query option
    IF(INDXA(COMLYN,COMLEN,'QUER').GT.0) THEN
       IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
       CALL GTICNT(IUNIT,HDR,ICNTRL,.TRUE.,.TRUE.)
       RETURN
    ENDIF
    !
    IF(QREAD) THEN
       CALL WRNDIE(0,'<TRAJIO>', &
            'INITIATING NEW TRAJECTORY I/O WHILE STILL READING OLD ONE')
    ENDIF
    IF(QWRITE) THEN
       CALL WRNDIE(0,'<TRAJIO>', &
            'INITIATING NEW TRAJECTORY I/O WHILE STILL WRITING OLD ONE')
    ENDIF
    !
    !     READ TRAJ OPTIONS
    !  First try to parse the standard syntax
    !  LNI March 1998
    !
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,NBEGN,SKIP,NSTOP)
    !
    !  Be nice and see if user may have tried the "old" syntax:
    NUNIT=GTRMI(COMLYN,COMLEN,'NREA',NUNIT)
    FIRSTU=GTRMI(COMLYN,COMLEN,'IREA',FIRSTU)
    !
    !     WRITE TRAJ OPTIONS
    !
    TRAJU=GTRMI(COMLYN,COMLEN,'IWRI',-1)
    NTRAJU=GTRMI(COMLYN,COMLEN,'NWRI',1)
    NFILE=GTRMI(COMLYN,COMLEN,'NFIL',0)
    QEXPND=(INDXA(COMLYN,COMLEN,'EXPA').GT.0)
    QVEL=(INDXA(COMLYN,COMLEN,'VELO').GT.0)
    NPRIV=GTRMI(COMLYN,COMLEN,'NOTH',0)
    ! init when routine is called
    DELTA=-1.d0
    WDELTA=-1.d0
    WDELTA=GTRMF(COMLYN,COMLEN,'DELT',WDELTA)
    ! convert to AKMA
    WDELTA=WDELTA/TIMFAC
    ! kluge for pycharmm
    qnotitle = ( indxa(comlyn,comlen,'NOTI') > 0 )
    !
    QREAD=(FIRSTU.GT.0)
    QWRITE=(TRAJU.GT.0)
    !
    if(.not.allocated(freeat)) then 
       call chmalloc('trajio.src','TRAJIO','FREEAT',NATOM,intg=FREEAT)
    else
       call chmdealloc('trajio.src','TRAJIO','FREEAT',NATOM,intg=FREEAT)
       call chmalloc('trajio.src','TRAJIO','FREEAT',NATOM,intg=FREEAT)
    endif
    IF(QREAD) THEN
       !       set up to read a trajectory
       IF(PRNLEV.GE.2) WRITE(OUTU,220) FIRSTU,NUNIT,SKIP
220    FORMAT(' TRAJ: INITIATING READ OF A TRAJECTORY, OPTIONS;'/, &
            '    FIRSTU = ',I3,' NUNIT = ',I3,' SKIP = ',I5)
       !
       !  allocate space for both reading and writing
       if(.not. allocated(temp)) then
          call chmalloc('trajio.src','TRAJIO','TEMP',NATOM,cr4=TEMP)
       else
          call chmdealloc('trajio.src','TRAJIO','TEMP',NATOM,cr4=TEMP)
          call chmalloc('trajio.src','TRAJIO','TEMP',NATOM,cr4=TEMP)
       endif
       !
       OUTSTP=0
       UNIT=FIRSTU
       ISTATS=1
       !
       NPRIV=-2
    ELSE
       !       set up dummy read options (for writing)
       IF(QEXPND) THEN
          NFREAT=NATOM
       ELSE
          call MKFRAT(IMOVE,NATOM,FREEAT,NFREAT)
       ENDIF
       ISTEP=(NPRIV+1)*SKIP
       ISTATS=1
       NDEGF=NATOM*3-6
       RNSAVV=0
       NPRIV=-1
    ENDIF
    !
    IF(QWRITE) THEN
       !       write a trajectory
       !
       IF(NFILE.LE.0 .AND. .NOT.QREAD) THEN
          CALL WRNDIE(-1,'<TRAJIO>','NFILE MUST BE SPECIFIED.')
          RETURN
       ENDIF
       !
       TRAJU=TRAJU-1
       NTRAJU=TRAJU+NTRAJU
       ! kluge for pycharmm
       if ( .not. qnotitle ) CALL RDTITL(TITLEA,NTITLA,ISTRM,0)
    ENDIF
    !
    RETURN
  end SUBROUTINE TRAJIO
    !
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !
  subroutine REATRJ(X,Y,Z)
#if KEY_CHEQ==1
  use cheq,only:qcg,cpcgimg    
#endif
#if KEY_FLUCQ==1
  use flucqm,only:fqcfor       
#endif
  use memory
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use cvio
  use comand
  use ctitla
  use exfunc
  use psf
  use stream
  use bases_fcm
#if KEY_FLUCQ==1
  use flucq      
#endif
  use image
  use image_routines_module, only: transo

  implicit none
  
    real(chm_real) X(*),Y(*),Z(*)

    !     UNLESS(QREAD)
    IF(.NOT.QREAD) &
         CALL WRNDIE(-1,'<REATRJ>','INPUT TRAJECTORY NOT SET UP.')
    !     FIN ! IF
    !
    !      REPEAT UNTIL (ISTATS .LT. 0)
    CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
         CG,QCG,                                   & 
#endif
         TEMP, &
         NATOM,FREEAT,NFREAT, &
         FIRSTU,NUNIT,UNIT,IFILE, &
         ISTEP,ISTATS,NDEGF,DELTA, &
         NBEGN,NSTOP,SKIP,RNSAVV,COORHD,VELHD, &
         TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
    ! . Construct coordinates for all image atoms.
    !
    IF(NTRANS.GT.0) THEN
       ! . Construct the coordinates.
       CALL TRANSO(X, Y, Z, &
            (/ zero /), (/ zero /), (/ zero /), &
            .FALSE., .FALSE., &
            (/ zero /), &
            NATOM,NTRANS, &
            IMTRNS,BIMAG%IMATPT,BIMAG%IMATTR, &
            NOROT,NATIM &
#if KEY_FLUCQ==1
            ,QFLUC,CG,FQCFOR    & 
#endif
            )
#if KEY_CHEQ==1
       ! Make sure the image charges are right
       IF (QCG) CALL CPCGIMG(BIMAG%IMATTR)
#endif 
    ENDIF
    !
    IF(NPRIV.LT.0) NPRIV=-1
    IF(ISTATS.GE.0) RETURN
    QREAD=.FALSE.
    IF(.NOT.QWRITE) &
       call chmdealloc('trajio.src','TRAJIO','freeat',NATOM,intg=freeat)
       call chmdealloc('trajio.src','TRAJIO','TEMP',NATOM,cr4=TEMP)
    RETURN
  end subroutine REATRJ
    !
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !
  subroutine WRTTRJ(X,Y,Z)
#if KEY_CHEQ==1
  use cheq,only:qcg,cpcgimg    
#endif
#if KEY_FLUCQ==1
  use flucqm,only:fqcfor       
#endif
  use memory
  use chm_kinds
  use dimens_fcm
  use number
  use consta,only: timfac
  use cvio
  use comand
  use ctitla
  use exfunc
  use psf
  use stream
  use bases_fcm
#if KEY_FLUCQ==1
  use flucq      
#endif
  use image
    real(chm_real) X(*),Y(*),Z(*)


    IF(.NOT.QWRITE) &
         CALL WRNDIE(-1,'<WRTTRJ>','OUTPUT TRAJECTORY NOT SET UP.')
    IF(NPRIV.EQ.-2) &
         CALL WRNDIE(-1,'<WRTTRJ>','MUST READ A FRAME BEFORE FIRST WRITE')
    !
    IF(OUTSTP.EQ.0) THEN
       IF(NFILE.EQ.0) NFILE=IFILE*NUNIT
       TRAJU=TRAJU+1
       IF(NPRIV.LT.0) THEN
          NPRIV=ISTEP
          NSAVC=SKIP
          NSTEP=NFILE*SKIP
          IF(QEXPND) THEN
             NFROUT=NATOM
          ELSE
             NFROUT=NFREAT
          ENDIF
       ELSE
          NPRIV=NPRIV+NSTEP
       ENDIF
       !
       IF(PRNLEV.GE.2) WRITE(OUTU,201) SKIP,NPRIV,NSTEP,TRAJU
201    FORMAT(' TRAJ: WRITING NEXT TRAJECTORY FILE,  OPTIONS;'/, &
            '  SKIP= ',I6,' NPRIV= ',I6,' NSTEP= ',I6,' TRAJU= ',I6,/)
       !
    ENDIF
    !
    IF(WDELTA < 0.d0) THEN
      IF (DELTA > 0.d0) THEN
        WDELTA = DELTA
      ELSE
        WDELTA = 0.001/TIMFAC
      ENDIF
    ENDIF
    OUTSTP=OUTSTP+SKIP
    !
    CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
         CG,QCG,                                      & 
#endif
         NATOM,FREEAT,NFROUT,NPRIV,OUTSTP,NDEGF, &
         WDELTA,NSAVC,NSTEP,TITLEA,NTITLA,TRAJU,QVEL, &
         .FALSE., (/ 0 /), .FALSE., (/ ZERO /))
    !
    IF (OUTSTP.EQ.NSTEP) OUTSTP=0
    !
    IF(OUTSTP.GT.0 .OR. TRAJU.LT.NTRAJU) RETURN
    QWRITE=.FALSE.
    call chmdealloc('trajio.src','TRAJIO','freeat',NATOM,intg=freeat)
    RETURN
  END SUBROUTINE WRTTRJ

end module traj_mod

