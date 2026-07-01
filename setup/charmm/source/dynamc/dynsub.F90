SUBROUTINE TMERGE(COMLYN,COMLEN,TITLEB,NTITLB)
  !-----------------------------------------------------------------------
  !     Merges or breaks up a dynamics coordinate or velocity trajectory
  !     into different numbers of units. The syntax of the command is
  !
  !
  !     MERGE [ COOR ] [FIRSTU unit-number] [NUNIT integer] [SKIP integer]
  !           [BEGIN int] [STOP int]   [ NOCHeck ]
  !     [ VEL  ]       [OUTPUTU unit-number] [NFILE integer] [ NOCRystal ]
  !     [ DRAW ]             [first-atom-selection] 
  !
  !               [ SUBSET MEMSSU integer OUTPU integer NUNSS integer ]
  !
  !     [ ORIEnt  [MASS] [WEIGht] [NOROt] [PRINT] second-atom-selection ]
  !
  !     [ XFLUct ] [ UNFOld ] [ REPACK ] [RECEnter second-atom-selection ]
  !     [ SIMPle ]
  !     [ RTOTem  EXCU unit-number NEXChange integer NRPL integer NREPeat integer ]
  !
  !     Subcommand for RECEnter:
  !     RECEnter second-atom-selection -
  !     NAMD LATTice integer -
  !     VE11 real VE21 real VE31 real -
  !     VE12 real VE22 real VE32 real -
  !     VE13 real VE23 real VE33 real
  !
  !     NAMD : use namd way of lattice transformation with
  !            real-space lattice vector (VEC1, VEC2, VEC3), upto the
  !            lattice vector of defined in LATTice integer
  !
  !     The various numbers are identical in semantics to those used for
  !     rotating a coordinate trajectory with respect to given set of
  !     coordinates in the COOR ORIEnt command.
  !
  !     The first atom selection selects which atoms are to be written out.
  !     A new PSF will be needed to read the merged trajectory is only a
  !     subset of atoms are written. This option is incompatible with
  !     fixed atoms.
  !
  !     The SUBSET keyword splits the input into NUNSS subset files
  !     based on user criteria; a membership list is read from MEMSSU.
  !     The membership list format is identical to that produced by the
  !     CLUSter command in CORREL.  [ Rick Venable March 2008 ]
  !
  !     REPACK added for trajectories w/o image centering; repacks
  !     orthogonal unit cells (CUBI,TETR,ORTH), translation only
  !
  !     SIMPLE added to handle direct I/O cases
  !
  !     The current main coordinates will be destroyed.
  !     The second atom selection is used if the ORIENT option is spcified
  !     which will orient each coordinate set wrt to the comparison
  !     coordinates.  This option is not for use with velocities.
  !
  !     Authors: Robert Bruccoleri
  !     Tim Miller      - Added handling of "simplified" trajectory
  !                       files and merging of per replica into
  !                       per temperature trajectories (RTOT keyword);
  !                       Dec 2012.
  !     Lennart Nilsson - Added BEGIN/STOP, ORIENT, SELECT
  !                       More educated NFILE estimate, October 1996
  !                       RECEnter, April 1998 (may or may not work
  !                       with XFLUct& UNFOld; uses same selection as
  !                       ORIENT - now a +1 warning, but perhaps OK?)
  !                       Pass CRYSTAL data even if CRYSTAL is not setup,
  !                       unless NOCRystal is specified in command; Jan 2011
  !     Rick Venable    - Unit cell corrections, XFLUCT & UNFOLD; Nov97
  !

#if KEY_CHEQ==1
  use cheq,only:qcg                    
#endif

  use chm_kinds
  use dimens_fcm
  use memory
  use number
  use cvio
  use psf
  use coordc
  use image
  use select
  use stream
  use string
  use exchangefile
  use consta,only : PI
  use param_store, only: get_param

  implicit none
  integer,allocatable,dimension(:) :: SSLST
  real(chm_real),allocatable,dimension(:) :: X
  real(chm_real),allocatable,dimension(:) :: Y
  real(chm_real),allocatable,dimension(:) :: Z
#if KEY_CHEQ==1
  real(chm_real),allocatable,dimension(:) :: CCG   
#endif
  real(chm_real),allocatable,dimension(:) :: XF
  real(chm_real),allocatable,dimension(:) :: YF
  real(chm_real),allocatable,dimension(:) :: ZF
  integer,allocatable,dimension(:) :: FREEAT
  real(chm_real4),allocatable,dimension(:) :: ITEMP
  integer,allocatable,dimension(:) :: ISLCT
  integer,allocatable,dimension(:) :: JSLCT
  integer,allocatable,dimension(:,:) :: ATOMIN
  character(len=*) COMLYN
  INTEGER COMLEN,NTITLB
  character(len=*) TITLEB(*)

  !
  INTEGER NFREAT,FIRSTU,NUNIT,UNIT,IFILE
  INTEGER ISTEP,ISTATS,NDEGF,NSEL
  real(chm_real) DELTA, XTLAVG(6)
  INTEGER NBEGN,NSTOP,SKIP,RNSAVV
  INTEGER NPRIV,OUTSTP,NSAVC,NSTEP,TRAJU
  INTEGER ICNTRL(20),NSTP,NBEG,NSKP,IUNIT,NFI,NCRYS
  INTEGER OUTPTU,NFILE,OLDLST,I,IWARN, J, K
  LOGICAL QORIENT,QSEL,QDRAW,QXFLUCT,QUNFOLD,QFIRST,QCENTER,QCINT,QREPACK,QOUTSIM
  LOGICAL LMASS,LWEIG,LNORO,LPRINT,LRMS,ERR,QSUBSET,QNOCHK,QNOCRYS,QRPIMG,QRTOT
  ! subset control variables
  INTEGER MAXSS,KSUBSTP,EXCUN,NEXCHANGE,NREPL,NREPEAT
  PARAMETER (MAXSS=1024)
  INTEGER ISSFRM(MAXSS),NSSFRM(MAXSS), MEMSSU,NUNSS,NSSREC

  ! for NAMD lattice transformation
  logical :: QCENTER_namd
  integer :: icnt,jcnt
  integer :: NTRANS_keep
  real(chm_real),parameter :: halfPi=half*PI

  !
  CHARACTER(LEN=4) OLDXTL
  character(len=4) COORHD,VELHD,HDR,HDRR
  DATA COORHD,VELHD/'CORD','VELD'/
  !
  IF(IOLEV < 0) RETURN
  !
  HDR=COORHD
  QDRAW=.FALSE.
  IF (INDXA(COMLYN,COMLEN,'COOR')  >  0) THEN
     HDR=COORHD
  ELSE IF (INDXA(COMLYN,COMLEN,'VEL ')  >  0) THEN
     HDR=VELHD
  ELSE IF (INDXA(COMLYN,COMLEN,'DRAW')  >  0) THEN
     QDRAW=.TRUE.
  ENDIF
  !
  QUNFOLD=(INDXA(COMLYN,COMLEN,'UNFO') > 0)
  QXFLUCT=(INDXA(COMLYN,COMLEN,'XFLU') > 0)
  QOUTSIM=(INDXA(COMLYN,COMLEN,'SIMP') > 0)
  QRTOT=(INDXA(COMLYN,COMLEN,'RTOT') > 0)
  IF(QRTOT) THEN
     EXCUN=GTRMI(COMLYN,COMLEN,'EXCU',-1)
     NEXCHANGE=GTRMI(COMLYN,COMLEN,'NEXC',-1)
     NREPL=GTRMI(COMLYN,COMLEN,'NRPL',-1)
     NREPEAT=GTRMI(COMLYN,COMLEN,'NREP',1)
     IF(EXCUN.LE.0) &
        CALL WRNDIE(-6,'<TMERGE>','CANNOT CONVERT REPLICA TO TEMP WITHOUT EXCHANGES')
     IF(NREPL.LE.0) &
        CALL WRNDIE(-6,'<TMERGE>','NUMBER OF REPLICAS IS NOT VALID')
     IF(NEXCHANGE.LE.0) &
        CALL WRNDIE(-6,'<TMERGE>','NEED A VALID NUMBER OF EXCHANGES')

     CALL CREATE_EXFILE(EXCUN,NEXCHANGE,NREPL,NREPEAT)
     CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,NBEGN,SKIP,NSTOP)
     OUTPTU=GTRMI(COMLYN,COMLEN,'OUTP',61)
     CALL MERGETM(NEXCHANGE,NREPL,NREPEAT,FIRSTU,OUTPTU,NUNIT,NBEGN,SKIP,NSTOP,HDR)
     CALL DELETE_EXFILE()
     RETURN
  ENDIF

  IF (QUNFOLD.OR.QXFLUCT) THEN
     IF (XNSYMM == 0) THEN
        CALL WRNDIE(-3,'<TMERGE> ', &
             'CRYStal required for XFLUct, UNFOld')
     ELSEIF (XNSYMM > 1) THEN
        CALL WRNDIE(-3,'<TMERGE> ', &
             'Only translation supported for XFLUct, UNFOld')
     ELSE
        ! store the initial unit cell data
        DO I=1,6
           XTLAVG(I) = XUCELL(I)
        ENDDO
     ENDIF
  ENDIF
  !
  QREPACK=(INDXA(COMLYN,COMLEN,'REPA') > 0)
  IF (QREPACK) THEN
     IF (XNSYMM == 0) THEN ! error if no images
        CALL WRNDIE(-3,'<TMERGE> ','images required for REPAck')
     ENDIF
     QRPIMG=(INDXA(COMLYN,COMLEN,'IMAG') > 0)
     IF (QRPIMG) THEN
       IF(PRNLEV >= 2) WRITE(OUTU,*) &
          '  REPACK enabled, using image transformations'
     ELSE
       IF (XNSYMM > 1) THEN ! translation only
         CALL WRNDIE(-3,'<TMERGE> ', &
           'Only translation supported for REPAck w/o image trans.')
       ELSEIF (ZERO/=(XTLABC(2)+XTLABC(4)+XTLABC(5))) THEN ! cosine sum
         CALL WRNDIE(-3,'<TMERGE> ', &
           'Orthogonal cell required for REPAck w/o image trans.')
       ELSE
         IF(PRNLEV >= 2) WRITE(OUTU,*) '  REPACK enabled, orthogonal mode'
       ENDIF
     ENDIF
  ENDIF

  QCENTER=(INDXA(COMLYN,COMLEN,'RECE') > 0)
  QCENTER_namd=(INDXA(COMLYN,COMLEN,'NAMD').GT.0)
  IF(QCENTER)THEN
     IF(NTRANS == 0)THEN
        CALL WRNDIE(-2,'<TMERGE> ', &
             'IMAGes required for RECEnter')
        ! user insists, but no transformations so reset flag
        QCENTER=.FALSE.        
     ENDIF
     ! Perhaps cannot do this AND the XFLUct, UNFOld stuff at the same time?
     IF(QUNFOLD.OR.QXFLUCT) THEN
        CALL WRNDIE(-2,'<TMERGE> ', &
             'RECEnter not tested with XFLUct/UNFOld')
     ENDIF
  ENDIF

  if(QCENTER.and.QCENTER_namd) then
     ILATT=GTRMI(COMLYN,COMLEN,'LATT',0)     ! number of lattice vectors in each direction.
     lattice_vector(1,1)=GTRMF(COMLYN,COMLEN,'VE11',zero)
     lattice_vector(2,1)=GTRMF(COMLYN,COMLEN,'VE21',zero)
     lattice_vector(3,1)=GTRMF(COMLYN,COMLEN,'VE31',zero)

     lattice_vector(1,2)=GTRMF(COMLYN,COMLEN,'VE12',zero)
     lattice_vector(2,2)=GTRMF(COMLYN,COMLEN,'VE22',zero)
     lattice_vector(3,2)=GTRMF(COMLYN,COMLEN,'VE32',zero)

     lattice_vector(1,3)=GTRMF(COMLYN,COMLEN,'VE13',zero)
     lattice_vector(2,3)=GTRMF(COMLYN,COMLEN,'VE23',zero)
     lattice_vector(3,3)=GTRMF(COMLYN,COMLEN,'VE33',zero)
     NumLattice=(2*ILATT+1)*(2*ILATT+1)*(2*ILATT+1)    ! total number of lattice sums.
     if(NumLattice.gt.MAXTRN) NumLattice=MAXTRN
     NTRANS_keep=NTRANS
     NTRANS=NumLattice

!    now recompute IMTRNS values.
!    icnt=0
!    do i=-ILATT,ILATT
!       do j=-ILATT,ILATT
!          do k=-ILATT,ILATT
!             icnt=icnt+1
!             vtmp(1:3)=real(i)*lattice_vector(1:3,1)+    &
!                       real(j)*lattice_vector(1:3,2)+    &
!                       real(k)*lattice_vector(1:3,3)
!             jcnt=12*(icnt-1)
!             IMTRNS(1,icnt) = one
!             IMTRNS(2:4,icnt) =zero
!             IMTRNS(5,icnt) = one
!             IMTRNS(6:8,icnt) = zero
!             IMTRNS(9,icnt) = one
!             IMTRNS(10:12,icnt)=vtmp(1:3)
!          end do
!       end do
!    end do
  end if

  ! set up for subset creation
  QSUBSET=(INDXA(COMLYN,COMLEN,'SUBS') > 0)
  IF(QSUBSET) THEN
     MEMSSU = GTRMI(COMLYN,COMLEN,'MEMS',-1)
     IF (MEMSSU < 1) CALL WRNDIE(-3, &
          '<TMERGE> ','MEMSSU, unit number for member list required')
     NUNSS  = GTRMI(COMLYN,COMLEN,'NUNS',-1)
     IF (NUNSS < 1.OR.NUNSS > MAXSS) CALL WRNDIE(-3, &
          '<TMERGE> ','NUNSS, invalid number of SUBSET units')
     CALL SSMEMCHK(MEMSSU,NUNSS,NSSREC,ISSFRM,NSSFRM)
     call chmalloc('dynsub.src','TMERGE','SSLST',NSSREC,intg=SSLST)

     CALL SSGETMEM(MEMSSU,NUNSS,NSSREC,ISSFRM,NSSFRM,SSLST)
     IF (PRNLEV > 2) WRITE(OUTU,920) NUNSS,NSSREC, &
          (NSSFRM(I),I=1,NUNSS)
     KSUBSTP = 0
  ENDIF
920 FORMAT('<TMERGE> Creating ',I2,' subsets from ',I9,' frames', &
       /,'Member counts:',/,(8I9))
  !
  ! Space allocation
  !
  call chmalloc('dynsub.src','TMERGE','X',NATOM,crl=X)
  call chmalloc('dynsub.src','TMERGE','Y',NATOM,crl=Y)
  call chmalloc('dynsub.src','TMERGE','Z',NATOM,crl=Z)
#if KEY_CHEQ==1
  call chmalloc('dynsub.src','TMERGE','CCG',NATOM,crl=CCG)    
#endif
  call chmalloc('dynsub.src','TMERGE','XF',NATOM,crl=XF)
  call chmalloc('dynsub.src','TMERGE','YF',NATOM,crl=YF)
  call chmalloc('dynsub.src','TMERGE','ZF',NATOM,crl=ZF)
  call chmalloc('dynsub.src','TMERGE','FREEAT',NATOM,intg=FREEAT)
  call chmalloc('dynsub.src','TMERGE','ITEMP',NATOM,cr4=ITEMP)
  call chmalloc('dynsub.src','TMERGE','ISLCT',NATOM,intg=ISLCT)
  !      islct(1:natom) = 0
  call chmalloc('dynsub.src','TMERGE','JSLCT',NATOM,intg=JSLCT)
  !      jslct(1:natom) = 0
  call chmalloc('dynsub.src','TMERGE','ATOMIN',2,NATOM,intg=ATOMIN)
  !
  ! File information
  !
  CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,NBEGN,SKIP,NSTOP)
  OUTPTU=GTRMI(COMLYN,COMLEN,'OUTP',61)
  NFILE=GTRMI(COMLYN,COMLEN,'NFIL',0)
  QNOCHK=.NOT. GET_TRAJ_CHK()
  IF(PRNLEV >= 2 .AND. QNOCHK) WRITE(OUTU,*) & 
    ' *** MERGE WITHOUT CHECKING THAT FILES ARE CONTIGUOUS ***' 
  !
  ! Now let READCV fix our choices, and also try to make a reasonable
  ! estimate for NFILE
  NBEG=NBEGN
  NSKP=SKIP
  NSTP=NSTOP
  !
  ! Open the first of our specified files
  ISTATS=1
  UNIT=FIRSTU
  CALL READCV(X,Y,Z,  &
#if KEY_CHEQ==1
       CCG,QCG,                            & 
#endif
       ITEMP, &
       NATOM,FREEAT,NFREAT, &
       FIRSTU,NUNIT,UNIT,IFILE, &
       ISTEP,ISTATS,NDEGF,DELTA, &
       NBEG,NSTP,NSKP,RNSAVV,HDR,HDR, &
       TITLEB,NTITLB,.FALSE., (/ ZERO /), .false.)
  !
  ! READCV should not close, and should also rewind files
  ! NBEG and NSKP should now be commensurate with the file structure
  !
  ! check if input traj contains CRYSTAL records
  QCINT=(ICNTRL(11) == 1)
  !
  ! we also need to get some information from the last unit, or in case NOCHeck
  ! is specified, we need to go through all files, trying to accommodate SKIP...
  !
  QNOCRYS=.FALSE.
  IF(QNOCHK)THEN
     NFI=0
     NCRYS=0
     DO IUNIT=FIRSTU,FIRSTU+NUNIT-1
        CALL GTICNT(IUNIT,HDRR,ICNTRL,.FALSE.,.TRUE.,.FALSE.)
  ! Assume ICNTRL(11)== 1 if crystal info present, 0 if not   
        NCRYS=NCRYS+ICNTRL(11)
        IF( NUNIT==1 .OR. SKIP > ICNTRL(3))THEN
           ! Hm, not sure this will always work
           NFI=NFI + ICNTRL(1) * GCD(SKIP,ICNTRL(3)) / SKIP
        ELSE
           ! We will read all frames
           NFI=NFI + ICNTRL(1)
        ENDIF
     ENDDO 
     IF(NCRYS /= 0 .AND. NCRYS /= NUNIT)THEN
        IF(PRNLEV >= 2) WRITE(OUTU,*) &
         ' *** Files differ regarding CRYSTAL information.'
        QNOCRYS=.TRUE.
     ENDIF
  ! Set up very simple structure in this case
     NPRIV=1
     NSAVC=1
     NSTEP=NFI
  ELSE    
    IUNIT=FIRSTU+NUNIT-1
    CALL GTICNT(IUNIT,HDRR,ICNTRL,.FALSE.,.TRUE.,.FALSE.)
    NSTP=ICNTRL(2)+ICNTRL(4)-ICNTRL(3)
  ENDIF
  ! 
  ! make sure also the STOP step is commensurate,
  ! taking into account that the user may have some say here
  !
  IF(NSTOP  >  0)THEN
     IF(NSTOP > NSTP)THEN
        IF(PRNLEV >= 2) WRITE(OUTU,*)  &
             'NSTOP IS TOO LARGE, HAS BEEN RESET TO:',NSTP
     ELSE    
        NSTP=NSTOP
     ENDIF
  ENDIF
  IF(MOD((NSTP-NBEG),NSKP) /= 0) THEN
     NSTP=((NSTP-NBEG)/NSKP)*NSKP + NBEG
  ENDIF
  !
  ! Number of coordinate sets to actually process:
  !
  IF(QNOCHK .AND. NFILE > 0 .AND. NFILE /= NFI) THEN
    IF(PRNLEV >= 2)THEN
       WRITE(OUTU,*) & 
          'With NOCHeck only one output file is written'
       WRITE(OUTU,*) 'NFILE reset to:', NFI
    ENDIF
    NFILE=NFI
  ELSE IF(.NOT. QNOCHK)THEN
    NFI=(NSTP-NBEG)/NSKP + 1
  ENDIF
  IF(NFILE  <=  0) NFILE=NFI
  !
  ! Set the values that will be used on reading also
  !
  NBEGN=NBEG
  SKIP=NSKP
  NSTOP=NSTP
  OLDXTL=XTLTYP
  !
!
! If crystal not setup, and input traj has crystal info, and 
! user has not explicitly asked to turn this off
! then we fake it for now to allow crystal 
! information to be written to the new trajectory. 
  QNOCRYS= (INDXA(COMLYN,COMLEN,'NOCR') > 0) .OR. QNOCRYS
  IF(QNOCRYS .AND. (QCINT .OR. XNSYMM /= 0) )THEN
     IWARN=-1
     IF(QNOCHK) IWARN=1
     CALL WRNDIE(IWARN,'<TMERGE>','CRYSTAL DATA WILL NOT BE WRITTEN')
     XTLTYP='    '
  ELSE IF(QCINT .AND. XNSYMM == 0)THEN
     XTLTYP='FAKE'
  ENDIF
 
  QORIENT=(INDXA(COMLYN,COMLEN,'ORIE') > 0)
  IF(QORIENT) THEN
     IF(QCENTER)THEN
        CALL WRNDIE(1,'<TMERGE> ', &
             'RECEnter USES SAME SELECTION AS ORIEnt!!!')
     ENDIF
     !
     IF (HDR  ==  COORHD) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,'(A)') &
             ' Orienting each frame wrt comparison coordinates'
        LMASS=(INDXA(COMLYN,COMLEN,'MASS') > 0)
        LWEIG=(INDXA(COMLYN,COMLEN,'WEIG') > 0)
        LNORO=(INDXA(COMLYN,COMLEN,'NORO') > 0)
        LPRINT=(INDXA(COMLYN,COMLEN,'PRIN') > 0)
        LRMS=.TRUE.
        !
     ELSE
        QORIENT=.FALSE.
        IF(WRNLEV >= 2) THEN
           WRITE(OUTU,'(A)') &
                ' *** WARNING *** MERGE CANNOT ORIENT VELOCITIES!'
           WRITE(OUTU,'(A)') &
                '                 MERGE proceeds w/o orienting'
        ENDIF
     ENDIF
  ENDIF

  ! subset courtesy messages; velocity, fixed atoms error
  IF (QSUBSET) THEN
     IF (QORIENT.AND.PRNLEV >= 2) THEN
        WRITE(OUTU,'(A)') '<TMERGE> ORIEnt ignored with SUBSET'
     ELSE IF (QCENTER.AND.PRNLEV >= 2) THEN
        WRITE(OUTU,'(A)') '<TMERGE> RECEnter ignored with SUBSET'
     ELSE IF (QUNFOLD.AND.PRNLEV >= 2) THEN
        WRITE(OUTU,'(A)') '<TMERGE> UNFOld ignored with SUBSET'
     ELSE IF (QXFLUCT.AND.PRNLEV >= 2) THEN
        WRITE(OUTU,'(A)') '<TMERGE> XFLUct ignored with SUBSET'
     ELSE IF (HDR == VELHD) THEN
        CALL WRNDIE(-3,'<TMERGE> ', &
             'SUBSET does not support VELOcities')
     ELSE IF (NFREAT /= NATOM) THEN
        CALL WRNDIE(-3,'<TMERGE> ', &
             'SUBSET does not support FIXed atoms')
     ENDIF
  ENDIF

  IF (HDR  ==  COORHD) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,'(A)') &
          ' Merging a coordinate trajectory'
  ELSE
     IF(PRNLEV >= 2) WRITE(OUTU,'(A)') &
          ' Merging a velocity trajectory'
  ENDIF
  IF(PRNLEV >= 2) WRITE(OUTU,220) FIRSTU,NUNIT,SKIP,OUTPTU, &
       NFILE,NBEGN,NSTOP
220 FORMAT(' FIRSTU=',I4,' NUNIT=',I4,' SKIP=',I6,/, &
       ' OUTPUTU=',I4,' NFILE=',I12,' BEGIN=',I12,' STOP=',I12)
  !
  !     Now perform the merge operation.
  !
  CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT, &
       XCOMP,YCOMP,ZCOMP,WCOMP,.TRUE.,ERR)
  !
  !     The second call to SELCTA should return a selection of atoms
  !     to be used with the orient command, or the recenter command.
  !
  NSEL=NSELCT(natom,ISLCT)
  IF(NSEL  <=  0) THEN
     IF(WRNLEV >= 2) WRITE(OUTU,'(A)') &
          '<TMERGE> NO ATOMS SELECTED FOR OUTPUT'
     GOTO 900
  ENDIF
  QSEL=(NSEL < NATOM .AND. NSEL  >  0)
  OUTSTP=0
  UNIT=FIRSTU
  ISTATS=1
  TRAJU=OUTPTU - 1
  !
300 CONTINUE
  CALL READCV(X,Y,Z,                      &
#if KEY_CHEQ==1
       CCG,QCG,                    & 
#endif
       ITEMP,                      &
       NATOM,FREEAT,NFREAT, &
       FIRSTU,NUNIT,UNIT,IFILE, &
       ISTEP,ISTATS,NDEGF,DELTA, &
       NBEGN,NSTOP,SKIP,RNSAVV,HDR,HDR, &
       TITLEB,NTITLB,.FALSE., (/ ZERO /), .false.)
  !brb...19-Jul-94 Comment out
  !      IF(NFREAT /= NATOM .AND. QSEL) THEN
  !        CALL WRNDIE(-3,'<TMERGE> ','CANNOT BOTH SELECT AND FIX ATOMS')
  !        GOTO 900
  !      ENDIF
  IF ( OUTSTP  ==  0 ) THEN
     IF ( NFILE  ==  0 ) NFILE=IFILE*NUNIT
     TRAJU=TRAJU+1
     IF(.NOT.QNOCHK)THEN
       NPRIV=ISTEP
       NSAVC=SKIP
       NSTEP=NFILE*SKIP
     ENDIF 
     IF(PRNLEV >= 2) WRITE(OUTU,320) SKIP,NBEGN,NSTEP,TRAJU
  ENDIF
320 FORMAT(//3X,' SKIP =',I6,' NBEGN =',I12, &
       6X,' NSTEP =',I12,' TRAJU=',I6,//)
  OUTSTP=OUTSTP+NSAVC
  IF (QDRAW) THEN
#if KEY_NOGRAPHICS==1 || KEY_NODISPLAY==1
     CALL WRNDIE(-5,'<TMERGE>','Draw feature not compiled.')
#else /**/
     CALL DRAWIT(-1,X,Y,Z,XCOMP,YCOMP,ZCOMP)
#endif 
  ELSE IF (QSUBSET) THEN
     IF(QOUTSIM) &
        CALL WRNDIE(-3,'<TMERGE>','SUBSET IS NOT COMPATIBLE WITH SIMPLIFIED TRAJECTORIES')

     KSUBSTP = KSUBSTP + 1
     CALL TSUBSET(QSEL,ISLCT,X,Y,Z, &
#if KEY_CHEQ==1
          CCG,                                       & 
#endif
          FREEAT,NFREAT,SSLST,KSUBSTP,QNOCHK, &
          QCINT,NDEGF,TITLEB,NTITLB,OUTPTU,ISSFRM,NSSFRM,MAXSS)
  ELSE
     QFIRST=.FALSE.
     IF (ISTEP == NBEG) QFIRST=.TRUE.
     if(QCENTER .and. QCENTER_namd) then
        ! if namd. NAMD (version 2.5 or higher) uses a different sequence for the cell info.
        ! Here is the hack of the trajectory according to the dcdplugin.c code for vmd.
        ! namd   uses A gamma B beta alpha C, alpha, beta, gamma are their cosines ( -1 <= ... <= 1.0), whereas
        ! charmm uses A B C alpha beta gamma in xucell array.
        xucell(1) = xtlabc(1)  ! A
        xucell(2) = xtlabc(3)  ! B
        xucell(3) = xtlabc(6)  ! C
        xucell(4) = 90.0d0 - asin(xtlabc(5))*90.0d0/halfPi  ! alpha
        xucell(5) = 90.0d0 - asin(xtlabc(4))*90.0d0/halfPi  ! beta
        xucell(6) = 90.0d0 - asin(xtlabc(2))*90.0d0/halfPi  ! gamma
        call XTLAXS(XTLABC,XUCELL)
        lattice_vector(1,1)=xtlabc(1)
        lattice_vector(2,1)=xtlabc(2)
        lattice_vector(3,1)=xtlabc(4)
        lattice_vector(1,2)=xtlabc(2)
        lattice_vector(2,2)=xtlabc(3)
        lattice_vector(3,2)=xtlabc(5)
        lattice_vector(1,3)=xtlabc(4)
        lattice_vector(2,3)=xtlabc(5)
        lattice_vector(3,3)=xtlabc(6)
     end if
     CALL MERGEO(HDR,QSEL,ISLCT,QORIENT,JSLCT, &
          ATOMIN,X,Y,Z, &
#if KEY_CHEQ==1
          CCG,                                       & 
#endif
          XF,YF,ZF,XTLAVG, &
          FREEAT,NFREAT, &
          LMASS,LWEIG,LNORO,LPRINT,LRMS,QXFLUCT,QUNFOLD,QFIRST,QCENTER, &
          QREPACK,QOUTSIM,QRPIMG,NPRIV,OUTSTP,NDEGF,DELTA, &
          NSAVC,NSTEP,TITLEB,NTITLB,TRAJU)
     IF (OUTSTP  ==  NSTEP) OUTSTP=0
  ENDIF
  IF (ISTATS >= 0) GOTO 300
  !
900 CONTINUE

  ! namkh:
  ! for NAMD lattice transformation
  if(QCENTER.and.QCENTER_namd) NTRANS=NTRANS_keep

  call chmdealloc('dynsub.src','TMERGE','X',NATOM,crl=X)
  call chmdealloc('dynsub.src','TMERGE','Y',NATOM,crl=Y)
  call chmdealloc('dynsub.src','TMERGE','Z',NATOM,crl=Z)
#if KEY_CHEQ==1
  call chmdealloc('dynsub.src','TMERGE','CCG',NATOM,crl=CCG)    
#endif
  call chmdealloc('dynsub.src','TMERGE','XF',NATOM,crl=XF)
  call chmdealloc('dynsub.src','TMERGE','YF',NATOM,crl=YF)
  call chmdealloc('dynsub.src','TMERGE','ZF',NATOM,crl=ZF)
  call chmdealloc('dynsub.src','TMERGE','FREEAT',NATOM,intg=FREEAT)
  call chmdealloc('dynsub.src','TMERGE','ITEMP',NATOM,cr4=ITEMP)
  call chmdealloc('dynsub.src','TMERGE','ISLCT',NATOM,intg=ISLCT)
  call chmdealloc('dynsub.src','TMERGE','JSLCT',NATOM,intg=JSLCT)
  call chmdealloc('dynsub.src','TMERGE','ATOMIN',2,NATOM,intg=ATOMIN)

  IF (QSUBSET) THEN
     call chmdealloc('dynsub.src','TMERGE','SSLST',NSSREC,intg=SSLST)
  ENDIF

  call get_param('NTOT', i)
  IF(I /= NFI) CALL WRNDIE(-1,'<TMERGE>', &
     'NFILES IS INCORRECT IN  OUTPUT FILE HEADER')
  XTLTYP=OLDXTL
  CALL SET_TRAJ_CHK(.TRUE.)
  RETURN
END SUBROUTINE TMERGE

SUBROUTINE MERGEO(HDR,QSEL,ISLCT,QORIENT,JSLCT,ATOMIN, &
     XX,YY,ZZ, &
#if KEY_CHEQ==1
     CCG,                                        & 
#endif
     XF,YF,ZF,XTLAVG,FREEAT,NFREAT, &
     LMASS,LWEIG,LNORO,LPRINT,LRMS, QXFLUCT,QUNFOLD,QFIRST,QCENTER, &
     QREPACK,QSIMOUT,QRPIMG,NPRIV,OUTSTP,NDEGF,DELTA, &
     NSAVC,NSTEP,TITLEB,NTITLB,TRAJU)
  !-----------------------------------------------------------------------
  !     Does the actual output, with proper orientations and selections
  !     as the case may be.
  !

#if KEY_CHEQ==1
  use cheq,only:qcg                 
#endif

  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use psf
  use bases_fcm
  use coordc
  use coord
  use cvio
  use image
  use image_routines_module, only: imcent
  use corman2
  use corsubs,only:orintc
  use chutil,only:qhavcrd
  !
  implicit none
  !
  character(len=*) HDR,TITLEB(*)
  LOGICAL QSEL,QORIENT,QCENTER
  INTEGER ISLCT(*),JSLCT(*),ATOMIN(2,*)
  INTEGER NFREAT,FREEAT(*),NPRIV,OUTSTP,NDEGF
  INTEGER NSAVC,NTITLB,TRAJU
  real(chm_real) XX(*),YY(*),ZZ(*)
#if KEY_CHEQ==1
  real(chm_real) CCG(*)
#endif 
  real(chm_real) XF(*),YF(*),ZF(*), XI,YI,ZI, CSCL(3,3)
  real(chm_real) DELTA, XTLAVG(6), XCINV(6),XHAVG(6)
  LOGICAL LMASS,LWEIG,LNORO,LPRINT,LRMS
  !
  INTEGER I,NAT,NSTEP,NF
  LOGICAL QVEL, QXFLUCT, QUNFOLD, LCHK, QFIRST, QREPACK, QRPIMG
  LOGICAL QSIMOUT
  !
  INTEGER NJSEL
  character(len=4) COORHD
  DATA COORHD/'CORD'/
  !
  QVEL=HDR /= COORHD
  !
  IF ((QXFLUCT.OR.QUNFOLD) .AND. (HDR == COORHD)) THEN
     ! convert from symmetric to fractional coords
     CALL XTLLAT(XUCELL,XTLABC)
     CALL CONCOR('SYMM','FRAC',XX,YY,ZZ,NATOM,ISLCT,XUCELL)
     IF (QUNFOLD) THEN
        ! just store the fractional coords for the first frame
        IF (QFIRST) THEN
           GOTO 120
        ELSE
           ! unfold; remove image centering operations by translation
           ! Allen & Tildesley algorithm applied to fractional coords
           DO I=1,NATOM
              XI = XX(I) - XF(I)
              YI = YY(I) - YF(I)
              ZI = ZZ(I) - ZF(I)
              XI = XI - ANINT(XI)
              YI = YI - ANINT(YI)
              ZI = ZI - ANINT(ZI)
              XX(I) = XF(I) + XI
              YY(I) = YF(I) + YI
              ZZ(I) = ZF(I) + ZI
           ENDDO
        ENDIF
     ENDIF
     ! preserve the corrected fractional coords for the next step
120  CONTINUE
     DO I=1,NATOM
        XF(I) = XX(I)
        YF(I) = YY(I)
        ZF(I) = ZZ(I)
     ENDDO
     ! convert from fractional back to Cartesian symmetric
     IF (QXFLUCT) THEN
        ! remove unit cell fluctuations by scaling to average unit cell
        CALL CONCOR('FRAC','SYMM',XX,YY,ZZ,NATOM,ISLCT,XTLAVG)
        ! replace frame shape matrix with average cell shape matrix
        CALL XTLAXS(XHAVG,XTLAVG)
        DO I=1,6
           XTLABC(I) = XHAVG(I)
        ENDDO
     ELSE
        ! simple unfolding, no changes to XUCELL or XTLABC
        CALL CONCOR('FRAC','SYMM',XX,YY,ZZ,NATOM,ISLCT,XUCELL)
     ENDIF
  ! repack, with or w/o image trans.; w/o limited to orthogonal, translation
  ELSE IF (QREPACK.AND.(HDR == COORHD)) THEN
    IF (QRPIMG) THEN
      CALL IMCENT(IMXCEN,IMYCEN,IMZCEN,BIMAG%IMCENF, &
           NTRANS,IMTRNS,IMNAME,XX,YY,ZZ,0, &
           (/ ZERO /), (/ ZERO /), (/ ZERO /), &
           (/ ZERO /), (/ ZERO /), (/ ZERO /), &
           .false., &
           NumLattice,ILATT,lattice_vector)
    ELSE
      CALL REPACK(IMXCEN,IMYCEN,IMZCEN,BIMAG%IMCENF,XX,YY,ZZ,XTLABC)
    ENDIF
  ENDIF
  !
  IF(QCENTER)THEN
     ! find center of geometry of second selection of atoms
     IMXCEN=ZERO
     IMYCEN=ZERO
     IMZCEN=ZERO
     NJSEL=0
     DO I=1,NATOM
        IF(JSLCT(I) /= 0)THEN
           IMXCEN=IMXCEN+XX(I)
           IMYCEN=IMYCEN+YY(I)
           IMZCEN=IMZCEN+ZZ(I)
           NJSEL=NJSEL+1
        ENDIF
     ENDDO
     IF(NJSEL > 0)THEN
        IMXCEN=IMXCEN/NJSEL
        IMYCEN=IMYCEN/NJSEL
        IMZCEN=IMZCEN/NJSEL
     ENDIF
     CALL IMCENT(IMXCEN,IMYCEN,IMZCEN,BIMAG%IMCENF, &
          NTRANS,IMTRNS,IMNAME,XX,YY,ZZ,0, &
          (/ ZERO /), (/ ZERO /), (/ ZERO /), &
          (/ ZERO /), (/ ZERO /), (/ ZERO /), &
          .false., &
          NumLattice,ILATT,lattice_vector)
  ENDIF
  !
  !     Orient (for now) all atoms wrt reference coordinates, using second
  !     selection as reference, rotate (that's the purpose of it), minimal output
  !
  IF(QORIENT) THEN
     !lni Check that comparison coordinates are present if ORIEnt is requested
     IF(QFIRST .AND. .NOT. QHAVCRD(NATOM,JSLCT,XCOMP)) THEN
        CALL WRNDIE(1,'<MERGEO>', &
             'NO COMP. COORD. FOR ORIENT! USING FIRST FRAME.')
        DO I =1,NATOM
           XCOMP(I)=XX(I)
           YCOMP(I)=YY(I)
           ZCOMP(I)=ZZ(I)
        ENDDO
     ENDIF
     CALL ORINTC(NATOM,XX,YY,ZZ,XCOMP,YCOMP,ZCOMP, &
          AMASS,LMASS,LRMS,ATOMIN,JSLCT,LWEIG,WMAIN, &
          LNORO,LPRINT)
  ENDIF
  !
  !     Only output those coordinates that we really want...
  !
  IF (QSEL) THEN
     NAT=0
     DO I=1,NATOM
        IF(ISLCT(I)  /=  0) THEN
           NAT=NAT+1
           X(NAT)=XX(I)
           Y(NAT)=YY(I)
           Z(NAT)=ZZ(I)
#if KEY_CHEQ==1
           IF(QCG) CG(NAT)=CCG(I)                           
#endif
        ENDIF
     ENDDO
     NF=MAX(1,NDEGF-3*(NATOM-NAT))
     IF(QSIMOUT) THEN
        CALL WRITSIMP(X,Y,Z,NAT,TRAJU,OUTSTP,NSTEP,NSAVC)
     ELSE
     CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
          CG,QCG,                                   & 
#endif
          NAT,FREEAT,NAT,NPRIV,OUTSTP,NF, &
          DELTA,NSAVC,NSTEP,TITLEB,NTITLB,TRAJU,QVEL, &
          .FALSE., (/ 0 /), .FALSE., (/ ZERO /))
     ENDIF
  ELSE
     IF(QSIMOUT) THEN
        CALL WRITSIMP(XX,YY,ZZ,NATOM,TRAJU,OUTSTP,NSTEP,NSAVC)
  ELSE
     CALL WRITCV(XX,YY,ZZ, &
#if KEY_CHEQ==1
          CCG,QCG,                                  & 
#endif
          NATOM,FREEAT,NFREAT,NPRIV,OUTSTP,NDEGF, &
          DELTA,NSAVC,NSTEP,TITLEB,NTITLB,TRAJU,QVEL, &
          .FALSE., (/ 0 /), .FALSE., (/ ZERO /))
  ENDIF
  ENDIF
  !
  RETURN
END SUBROUTINE MERGEO

SUBROUTINE TSUBSET(QSEL,ISELCT,XX,YY,ZZ, &
#if KEY_CHEQ==1
     CCG,                                               & 
#endif
     FREEAT,NFREAT,SUBLST,TSTP,QNOCHK,QCRYS,NDEGF,TTL,NTTL,SSU, &
     ISSFRM,NSSFRM,MXSS)
  !-----------------------------------------------------------------------
  ! writes each input frame to the appropriate subset file
  !-----------------------------------------------------------------------
#if KEY_CHEQ==1
  use cheq,only:qcg                 
#endif

  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use version
  use psf
  use coord
  use cvio

  implicit none
  LOGICAL QSEL,QCRYS,QCTL,QNOCHK
  INTEGER ISELCT(*),FREEAT(*),NFREAT,SUBLST(*),TSTP,NDEGF
  INTEGER NTTL,SSU,NUNSS,NSSREC,ISSFRM(*),NSSFRM(*),ICTL(20)
  INTEGER IU, NPRV, NAT, NF, ISTP, NSTP, ISKP, ISS, I, MXSS
#if KEY_CHEQ==1
  real(chm_real) CCG(*)                     
#endif
  real(chm_real) XX(*),YY(*),ZZ(*), DELTA
  character(len=*) TTL(*)
  REAL(chm_real4)  DEL4

  ! psf for NATOM and CG array; coord for X,Y,Z arrays

  ! set DELTA and SKIP to 1, NPRIV to 0
  DELTA = ONE
  ISKP = 1
  NPRV = 1
  NDEGF = 6 + 3*NATOM
  ! get the subset index, compute the unit number
  ISS = SUBLST(TSTP)
  IF(ISS < 1 .OR. ISS > MXSS) CALL WRNDIE(-3,'<TSUBSET> ', &
       'Bad subset number, must be 1..MAXSS inclusive')
  IU = SSU + ISS - 1
  ! update the step number for the subset, read subset total steps
  ISTP = ISSFRM(ISS) + 1
  ISSFRM(ISS) = ISTP
  NSTP = NSSFRM(ISS)

  ! set up the control array
  QCTL=.TRUE.
  DO I=1,20
     ICTL(I)=0
  ENDDO
  ICTL(1)=NSTP
  ICTL(2)=NPRV
  ICTL(3)=ISKP
  ICTL(4)=NSTP
  DEL4 = REAL(DELTA)
  ! store bytes of delta in ictl(10)
  ictl(10) = transfer(delta, ictl(10))
  IF (QCRYS) ICTL(11)=1
#if KEY_CHEQ==1
  IF (QCG) ICTL(13)=1                                  
#endif
  IF (QNOCHK) ICTL(14) =1
  IF(VERNUM <= 0 .OR. VERNUM > 10000) &
     CALL WRNDIE(-2,'<TSUBSET>', 'Internal error. Illegal VERNUM.')
  ICTL(20)=VERNUM

  ! write the frame, w. or w/o atom selection

  IF (QSEL) THEN
     NAT=0
     DO I=1,NATOM
        IF(ISELCT(I)  /=  0) THEN
           NAT=NAT+1
           X(NAT)=XX(I)
           Y(NAT)=YY(I)
           Z(NAT)=ZZ(I)
#if KEY_CHEQ==1
           IF(QCG) CG(NAT)=CCG(I)                           
#endif
        ENDIF
     ENDDO
     NF=MAX(1,NDEGF-3*(NATOM-NAT))
     ICTL(8)=NF
     CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
          CG,QCG,                                   & 
#endif
          NAT,FREEAT,NAT,NPRV,ISTP,NF, &
          DELTA,ISKP,NSTP,TTL,NTTL,IU,.FALSE., &
          QCTL,ICTL,.FALSE., (/ ZERO /))
  ELSE
     ICTL(8)=NDEGF
     CALL WRITCV(XX,YY,ZZ, &
#if KEY_CHEQ==1
          CCG,QCG,                                  & 
#endif
          NATOM,FREEAT,NFREAT,NPRV,ISTP,NDEGF, &
          DELTA,ISKP,NSTP,TTL,NTTL,IU,.FALSE., &
          QCTL,ICTL,.FALSE., (/ ZERO /))
  ENDIF

  RETURN
END SUBROUTINE TSUBSET

SUBROUTINE SSMEMCHK(MEMSSU,NUNSS,NSSREC,ISSFRM,NSSFRM)
  !-----------------------------------------------------------------------
  ! check membership list file for subset information, initialize
  !   MEMSSU  in; unit number for membership list; read header
  !   NUNSS   in; number of units opened for subset output; verify
  !   NSSREC  out; number of subset records, i.e. number of frames
  !   ISSFRM  out; array; current frame indices for each subset; init 0
  !   NSSFRM  out; array; number of frames for each subset; init 0
  !-----------------------------------------------------------------------
  INTEGER MEMSSU,NUNSS,NSSREC,ISSFRM(*),NSSFRM(*), I,J
  character(len=256) S
  DO I=1,4
     READ(MEMSSU,'(A)',ERR=901) S
  ENDDO
  READ(MEMSSU,'(A)',ERR=901) S
  I=INDEX(S,':')
  READ(S(I+1:I+8),'(I8)') NSSREC
  READ(MEMSSU,'(A)',ERR=901) S
  I=INDEX(S,':')
  READ(S(I+1:I+8),'(I8)') J
  IF(J /= NUNSS) CALL WRNDIE(-3,'<SSMEMCHK> ', &
       'NUNSS does not match the number of subsets (Clusters)')
  DO I=1,NUNSS
     ISSFRM(I) = 0
     NSSFRM(I) = 0
  ENDDO
  READ(MEMSSU,'(A)',ERR=901) S
  READ(MEMSSU,'(A)',ERR=901) S
  READ(MEMSSU,'(A)',ERR=901) S
  READ(S(1:16),'(2I8)') I,J
  IF (J == 0) THEN
     NSSREC = NSSREC - 1
  ELSE
     BACKSPACE(MEMSSU)
  ENDIF
  RETURN
901 CALL WRNDIE(-3,'<SSMEMCHK> ','member list file header error')
  RETURN
END SUBROUTINE SSMEMCHK

SUBROUTINE SSGETMEM(MEMSSU,NUNSS,NSSREC,ISSFRM,NSSFRM,NSSL)
  !-----------------------------------------------------------------------
  ! read membership list file for subset information, fill arrays
  !   MEMSSU  in; unit number for membership list
  !   NUNSS   in; number of units opened for subset output
  !   NSSREC  in; number of subset records, i.e. number of frames
  !   ISSFRM  in; array; current frame indices for each subset
  !   NSSFRM  out; array; number of frames for each subset
  !   NSSL    out; array; the membership list for each frame
  !-----------------------------------------------------------------------
  INTEGER MEMSSU,NUNSS,NSSREC,ISSFRM(*),NSSFRM(*)
  INTEGER NSSL(NSSREC),I,J
  character(len=256) S
  DO I=1,NSSREC
     READ(MEMSSU,'(A)',ERR=901,END=901) S
     READ(S(1:8),'(I8)') J
     NSSL(I) = J
     NSSFRM(J) = NSSFRM(J) + 1
  ENDDO
  RETURN
901 CALL WRNDIE(-3,'<SSGETMEM> ','member list file error')
  RETURN
END SUBROUTINE SSGETMEM

SUBROUTINE REPACK(IMXCEN,IMYCEN,IMZCEN,IMCENF,X,Y,Z,XTLABC)
  !
  use chm_kinds
  use dimens_fcm
  use number
  use psf
  implicit none
  real(chm_real) :: IMXCEN,IMYCEN,IMZCEN,X(*),Y(*),Z(*),XTLABC(6)
  integer :: IMCENF(*)
  !
  REAL(chm_real) :: xa,xb,xc
  INTEGER :: MODE
  INTEGER :: ISEG,IMIN,IS,IQ,I,ITRAN,IRES,IAT
  !
  ! unit cell data
  xa = xtlabc(1)
  xb = xtlabc(3)
  xc = xtlabc(6)
  !
  ! process segments
  MODE=1
  DO ISEG=1,NSEG
     IMIN=5
     IS=IBASE(NICTOT(ISEG)+1)+1
     IQ=IBASE(NICTOT(ISEG+1)+1)
     DO I=IS,IQ
        IF(IMCENF(I).LT.IMIN) IMIN=IMCENF(I)
     ENDDO
     IF(IMIN.EQ.MODE) THEN
       CALL MOVEATOMS(IMXCEN,IMYCEN,IMZCEN,X,Y,Z,XA,XB,XC,IS,IQ)
     ENDIF
  ENDDO
  ! 
  ! process residues
  MODE=2
  DO IRES=1,NRES
     IMIN=5
     IS=IBASE(IRES)+1
     IQ=IBASE(IRES+1)
     DO I=IS,IQ
        IF(IMCENF(I).LT.IMIN) IMIN=IMCENF(I)
     ENDDO
     IF(IMIN.EQ.MODE) THEN
       CALL MOVEATOMS(IMXCEN,IMYCEN,IMZCEN,X,Y,Z,XA,XB,XC,IS,IQ)
     ENDIF
  ENDDO
  !
  ! process groups
  MODE=3
  DO IRES=1,NGRP
     IMIN=5
     IS=IGPBS(IRES)+1
     IQ=IGPBS(IRES+1)
     DO I=IS,IQ
        IF(IMCENF(I).LT.IMIN) IMIN=IMCENF(I)
     ENDDO
     IF(IMIN.EQ.MODE) THEN
       CALL MOVEATOMS(IMXCEN,IMYCEN,IMZCEN,X,Y,Z,XA,XB,XC,IS,IQ)
     ENDIF
  ENDDO
  !
  ! process atoms
  MODE=4
  DO IAT=1,NATOM
     IS=IAT
     IQ=IAT
     IF(IMCENF(IAT).EQ.MODE) THEN
       CALL MOVEATOMS(IMXCEN,IMYCEN,IMZCEN,X,Y,Z,XA,XB,XC,IS,IQ)
     ENDIF
  ENDDO
  RETURN

END SUBROUTINE REPACK

SUBROUTINE MOVEATOMS(IMXCEN,IMYCEN,IMZCEN,X,Y,Z,XA,XB,XC,IS,IQ)
  ! move atoms based on center of geometry
  use chm_kinds
  use number
  implicit none
  real(chm_real) :: IMXCEN,IMYCEN,IMZCEN,X(*),Y(*),Z(*),XA,XB,XC
  integer :: i,is,iq
  real(chm_real) :: ha,hb,hc,cx,cy,cz,tx,ty,tz,cnt

  ! half
  ha = xa/two
  hb = xb/two
  hc = xc/two
  ! get center, shift by half
  cx = zero
  cy = zero
  cz = zero
  cnt = real(1+iq-is)
  do i = is,iq
    cx = cx + x(i)
    cy = cy + y(i)
    cz = cz + z(i)
  enddo
  cx = ha + cx/cnt
  cy = hb + cy/cnt
  cz = hc + cz/cnt
  ! translation of center
  tx = imxcen + (modulo(cx,xa)-cx)
  ty = imycen + (modulo(cy,xb)-cy)
  tz = imzcen + (modulo(cz,xc)-cz)
  ! apply to atoms
  do i = is,iq
    x(i) = x(i) + tx
    y(i) = y(i) + ty
    z(i) = z(i) + tz
  enddo
  return
END SUBROUTINE MOVEATOMS

SUBROUTINE MERGETM(NEXCHANGE,NREPL,NREPEAT,FIRSTU,OUTPTU,NUNIT,NBEGN,NSKIP,NSTOP,HDR)
   USE CHM_KINDS
   USE PSF
   USE CVIO
   USE MEMORY
   USE EXCHANGEFILE,ONLY:MAP_TOTEMPSPACE
   USE CTITLA
#if KEY_CHEQ==1
   USE CHEQ,ONLY:QCG  
#endif
   USE NUMBER
   USE STREAM,ONLY:OUTU,PRNLEV

   implicit none

   INTEGER,INTENT(IN)          :: NEXCHANGE,NREPL,NREPEAT,FIRSTU,OUTPTU,NUNIT
   INTEGER,INTENT(INOUT)       :: NBEGN,NSKIP
   INTEGER,INTENT(INOUT)       :: NSTOP
   CHARACTER(LEN=4),INTENT(IN) :: HDR

   REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:)     :: X,Y,Z,CCG,XF,YF,ZF
   REAL(CHM_REAL4),ALLOCATABLE,DIMENSION(:)    :: TEMP
   INTEGER,ALLOCATABLE,DIMENSION(:)            :: FREEAT,ISTEPS,ISTATS
   REAL(CHM_REAL)                              :: DELTA
   INTEGER                                     :: NFREAT,I,J,N,A
   INTEGER                                     :: IFILE,NDEGF
   INTEGER                                     :: INUN,TIMOUTU,RNSAVV,OUTSTP
   LOGICAL                                     :: QVEL
   integer nsavc, nstep, npriv

   CALL CHMALLOC('dynsub.src','MERGETM','X',NATOM,CRL=X)
   CALL CHMALLOC('dynsub.src','MERGETM','Y',NATOM,CRL=Y)
   CALL CHMALLOC('dynsub.src','MERGETM','Z',NATOM,CRL=Z)
   CALL CHMALLOC('dynsub.src','MERGETM','XF',NATOM,CRL=XF)
   CALL CHMALLOC('dynsub.src','MERGETM','YF',NATOM,CRL=YF)
   CALL CHMALLOC('dynsub.src','MERGETM','ZF',NATOM,CRL=ZF)
   CALL CHMALLOC('dynsub.src','MERGETM','CCG',NATOM,CRL=CCG)
   CALL CHMALLOC('dynsub.src','MERGETM','TEMP',NATOM,CR4=TEMP)
   CALL CHMALLOC('dynsub.src','MERGETM','FREEAT',NATOM,INTG=FREEAT)
   CALL CHMALLOC('dynsub.src','MERGETM','ISTEPS',NREPL,INTG=ISTEPS)
   CALL CHMALLOC('dynsub.src','MERGETM','ISTATS',NREPL,INTG=ISTATS)

   OUTSTP=0
   N=NEXCHANGE/NREPEAT
   ISTEPS(1:NREPL)=0
   ISTATS(1:NREPL)=2

   ! Get the header??!
   DO I=1,NREPL
      INUN=FIRSTU+I-1
      ISTATS(1:NREPL)=1
      CALL READCV(X,Y,Z,  &
#if KEY_CHEQ==1
                  CCG,QCG,                  & 
#endif
                  TEMP, &
                  NATOM,FREEAT,NFREAT, &
                  INUN,1,INUN,IFILE, &
                  ISTEPS(I),ISTATS(I),NDEGF,DELTA, &
                  NBEGN,NSTOP,NSKIP,RNSAVV,HDR,HDR, &
                  TITLEB,NTITLB,.FALSE., (/ ZERO /), .false.)
   ENDDO

   
   ISTEPS(1:NREPL)=NSKIP
   ISTATS(1:NREPL)=1
   NSAVC=NSKIP        !! Let's HOPE the skip is the same for all of the files
   NSTEP=IFILE*NSKIP  !! And likewise for NSTEP (but we test for this)
   N=IFILE            !! And also for the number of steps in each file!
   NSTOP=NSTEP        !! We know we just have one file

   DO I=1,N
      OUTSTP=OUTSTP+NSAVC
      DO J=1,NREPL
         INUN=FIRSTU+J-1
         !!WRITE(OUTU,'(A,I3,A,I10,X,I10)') 'TIM DEBUG CVIO> BEFORE ISTEPS,ISTATS(',J,') = ', ISTEPS(J),ISTATS(J)
         CALL READCV(X,Y,Z,  &
#if KEY_CHEQ==1
                     CCG,QCG,                  & 
#endif
                     TEMP, &
                     NATOM,FREEAT,NFREAT, &
                     INUN,1,INUN,IFILE, &
                     ISTEPS(J),ISTATS(J),NDEGF,DELTA, &
                     NBEGN,NSTOP,NSKIP,RNSAVV,HDR,HDR, &
                     TITLEB,NTITLB,.FALSE., (/ ZERO /), .false.)
         ISTATS(J)=3 ! gory hack alert
         !!WRITE(OUTU,'(A,I3,A,I10,X,I10)') 'TIM DEBUG CVIO> AFTER ISTEPS,ISTATS(',J,') = ', ISTEPS(J),ISTATS(J)

         IF(IFILE*NSKIP.NE.NSTEP) THEN
            CALL WRNDIE(-3,'<MERGETM>','INCONSISTENT NUMBER OF STEPS')
         ENDIF

         QVEL=(HDR /= 'CORD')
         CALL MAP_TOTEMPSPACE(OUTSTP,J,A)
         TIMOUTU=OUTPTU+A-1
         NPRIV=ISTEPS(J)

         CALL WRITCV(X,Y,Z,   &
#if KEY_CHEQ==1
                     CCG,QCG,     &  
#endif
                     NATOM,FREEAT,NFREAT,NPRIV,OUTSTP,NDEGF, &
                     DELTA,NSAVC,NSTEP,TITLEB,NTITLB,TIMOUTU,QVEL, &
                     .FALSE., (/ 0 /), .FALSE., (/ ZERO /))

        IF(PRNLEV.GT.6) & 
           WRITE(OUTU,'(A,I9,A,I3,A,I3)') 'MERGETM> STEP ',I,' FROM UN ',INUN,' TO UN ',TIMOUTU
      ENDDO
   ENDDO

   CALL CHMDEALLOC('dynsub.src','MERGETM','X',NATOM,CRL=X)
   CALL CHMDEALLOC('dynsub.src','MERGETM','Y',NATOM,CRL=Y)
   CALL CHMDEALLOC('dynsub.src','MERGETM','Z',NATOM,CRL=Z)
   CALL CHMDEALLOC('dynsub.src','MERGETM','XF',NATOM,CRL=XF)
   CALL CHMDEALLOC('dynsub.src','MERGETM','YF',NATOM,CRL=YF)
   CALL CHMDEALLOC('dynsub.src','MERGETM','ZF',NATOM,CRL=ZF)
   CALL CHMDEALLOC('dynsub.src','MERGETM','CCG',NATOM,CRL=CCG)
   CALL CHMDEALLOC('dynsub.src','MERGETM','TEMP',NATOM,CR4=TEMP)
   CALL CHMDEALLOC('dynsub.src','MERGETM','FREEAT',NATOM,INTG=FREEAT)
   CALL CHMDEALLOC('dynsub.src','MERGETM','ISTEPS',NREPL,INTG=ISTEPS)
   CALL CHMDEALLOC('dynsub.src','MERGETM','ISTATS',NREPL,INTG=ISTATS)

END SUBROUTINE MERGETM

