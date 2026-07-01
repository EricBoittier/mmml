module univ
  use chm_kinds
  implicit none
  !
  ! UNIV.FCM
  !    DATA USED TO ALLOW UNIVERSAL COORDINATE FILE I/O
  !
  !      INTEGER FMTMX - MAX NUMBER OF TRANSLATIONS
  !
  ! SEGID (SEGMENT IDENTIFIERS)
  !      INTEGER FMTSII - STARING COLUMN FOR SEGID
  !      INTEGER FMTSIF - LAST COLUMN FOR SEGID
  !      INTEGER FMTNSI - NUMBER OF TRANSLATIONS
  !      CHARACTER(len=8) FMTSIC(*) - EXTERNAL SEGID NAMES
  !      CHARACTER(len=8) FMTSIM(*) - INTERNAL SEGID NAMES
  !
  ! RESID (RESIDUE IDENTIFIERS)
  !      INTEGER FMTRII - STARTING COLUMN FOR RESID
  !      INTEGER FMTRIF - LAST COLUMN FOR RESID
  !      INTEGER FMTNRI - NUMBER OF TRANSLATIONS
  !      CHARACTER(len=8) FMTRIC(*) - EXTERNAL RESID NAMES
  !      CHARACTER(len=8) FMTRIM(*) - INTERNAL RESID NAMES
  !      CHARACTER(len=8) FMTRIS(*) - SELECTED SEGID NAMES FOR RESID TRANSLATIONS
  !
  ! IRES (GLOBAL RESIDUE NUMBERS)
  !      INTEGER FMTIRI - STARTING COLUMN FOR IRES
  !      INTEGER FMTIRF - LAST COLUMN FOR IRES
  !
  ! RESN (RESIDUE TYPES)
  !      INTEGER FMTRNI - STARTING FIELD FOR RESN
  !      INTEGER FMTRNF - LAST COLUMN FOR RESN
  !      INTEGER FMTNRN - NUMBER OF RESN TRANSLATIONS
  !      CHARACTER(len=8) FMTRNC(*) - EXTERNAL RESN NAMES
  !      CHARACTER(len=8) FMTRNM(*) - INTERNAL RESN NAMES
  !      CHARACTER(len=8) FMTRNS(*) - SELECTED SEGID NAMES FOR RESN TRANSLATIONS
  !
  ! TYPE (ATOM IDENTIFIERS)
  !      INTEGER FMTATI - STARTING COLUMN FOR ATOM FIELD
  !      INTEGER FMTATF - LAST COLUMN FOR ATOM FIELD
  !      INTEGER FMTNAT - NUMBER OF ATOM TRANSLATIONS
  !      CHARACTER(len=8) FMTATC(*) - EXTERNAL ATOM NAMES
  !      CHARACTER(len=8) FMTATM(*) - INTERNAL ATOM NAMES
  !      CHARACTER(len=8) FMTATS(*) - SELECTED SEGID NAMES FOR ATOM TRANSLATIONS
  !      CHARACTER(len=8) FMTATR(*) - SELECTED RESID NAMES FOR ATOM TRANSLATIONS
  !      CHARACTER(len=8) FMTATN(*) - SELECTED RESN  NAMES FOR ATOM TRANSLATIONS
  !
  ! ISEQ (GLOBAL ATOM NUMBERS)
  !      INTEGER FMTISI - STARTING COLUMN FOR ISEQ
  !      INTEGER FMTISF - LAST COLUMN FOR ISEQ
  !
  ! X,Y,Z,W (COORDINATES AND WEIGHTING ARRAY)
  !      INTEGER FMTIXI - STARTING COLUMN FOR X   
  !      INTEGER FMTIXF - LAST COLUMN FOR X   
  !      INTEGER FMTIYI - STARTING COLUMN FOR Y   
  !      INTEGER FMTIYF - LAST COLUMN FOR Y   
  !      INTEGER FMTIZI - STARTING COLUMN FOR Z   
  !      INTEGER FMTIZF - LAST COLUMN FOR Z   
  !      INTEGER FMTIWI - STARTING COLUMN FOR W   
  !      INTEGER FMTIWF - LAST COLUMN FOR W   
  !
  ! PICK OPTION
  !      INTEGER FMTNSL - NUMBER OF SELECTIONS
  !      INTEGER FMTSLI(*) - STARTING COLUMN FOR SELECTION
  !      INTEGER FMTSLF(*) - LAST COLUMN FOR SELECTION
  !      CHARACTER(len=8) FMTSLC(*) - CHARACTER STRINGS FOR SELECTIONS
  !
  ! EXCLUDE OPTION
  !      INTEGER FMTNEX - NUMBER OF EXCLUSIONS
  !      INTEGER FMTEXI(*) - STARTING COLUMN FOR SELECTION
  !      INTEGER FMTEXF(*) - LAST COLUMN FOR SELECTION
  !      CHARACTER(len=8) FMTEXC(*) - CHARACTER STRINGS FOR EXCLUSIONS
  !
  ! TITLE OPTION
  !      INTEGER FMTNTL - NUMBER OF TITLE KEYWORDS
  !      INTEGER FMTTII(*) - STARTING COLUMN FOR TITLE KEYWORDS
  !      INTEGER FMTTIF(*) - LAST COLUMN FOR TITLE KEYWORDS
  !      CHARACTER(len=8) FMTTIT(*) - TITLE KEYWORDS
  !
  !
  INTEGER,PARAMETER :: FMTMX=50
  ! SEGID
  INTEGER FMTSII,FMTSIF,FMTNSI
  character(len=8),dimension(fmtmx) :: FMTSIC,FMTSIM
  ! RESID
  INTEGER FMTRII,FMTRIF,FMTNRI
  character(len=8),dimension(fmtmx) :: FMTRIC,FMTRIM,FMTRIS
  ! IRES
  INTEGER FMTIRI,FMTIRF
  ! RESN
  INTEGER FMTRNI,FMTRNF,FMTNRN
  character(len=8),dimension(fmtmx) :: FMTRNC,FMTRNM,FMTRNS
  ! ATOMS
  INTEGER FMTATI,FMTATF,FMTNAT
  character(len=8),dimension(fmtmx) :: FMTATC,FMTATM,FMTATS,FMTATR,FMTATN
  ! ISEQ
  INTEGER FMTISI,FMTISF
  ! X,Y,Z,W
  INTEGER FMTIXI,FMTIXF
  INTEGER FMTIYI,FMTIYF
  INTEGER FMTIZI,FMTIZF
  INTEGER FMTIWI,FMTIWF
  ! PICK OPTION
  INTEGER FMTNSL
  integer,dimension(fmtmx) :: FMTSLI,FMTSLF
  character(len=8),dimension(fmtmx) :: FMTSLC
  ! EXCLUDE OPTION
  INTEGER FMTNEX
  integer,dimension(fmtmx) :: FMTEXI,FMTEXF
  character(len=8),dimension(fmtmx) :: FMTEXC
  ! TITLE OPTION
  INTEGER FMTNTL
  integer,dimension(fmtmx) :: FMTTII,FMTTIF
  character(len=8),dimension(fmtmx) :: FMTTIT
  !


contains


  SUBROUTINE CREADU(IUNIT,X,Y,Z,WMAIN,NATOM, &
       ISLCT,RES,NRES,ATYPE,IBASE,SEGID,RESID,NICTOT,NSEG, &
       LRSID,LFREE,IOFFS)
    !
    !     COORDINATE READING ROUTINES CARD READING SECTION MODIFIED TO
    !     MAP COORDINATES BY THE SEQUENCE NUMBER, RESIDUE TYPE, AND ATOM
    !     TYPE IN THE INPUT FILE, AND TO CHECK FOR SEQUENCE CONSISTENCY:
    !
    !     sequence errors are warnings.
    !     missing coordinates result in warnings.
    !     multiple cooridnates result in warnings.
    !     atoms not present are ignored.
    !     Residues out of range result in warnings.
    !
    !     by Bernard R. Brooks   1987
    !
    use chm_kinds
    use dimens_fcm
    use exfunc
    use stream
    use string
    use ctitla
    use chutil,only:matom,initia
#if KEY_STRINGM==1 /*  VO */
    use machio, only: ifreeu
    use multicom_aux
#endif
    !
    implicit none
    INTEGER IUNIT
    real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
    INTEGER NATOM
    INTEGER IBASE(*),ISLCT(*)
    INTEGER NRES,NSEG
    CHARACTER(len=*) RESID(*),SEGID(*),RES(*),ATYPE(*)
    INTEGER NICTOT(*),IOFFS
    LOGICAL LRSID,LFREE
    !
    LOGICAL DONE
    !
    character(len=8) :: RESIN,ATOMIN,SID,RID
    character(len=8) :: ATOMN4
    CHARACTER(len=180) PDBLIN
    INTEGER ERRCNT
    real(chm_real)  XIN,YIN,ZIN,WIN
    LOGICAL QATOM
    INTEGER IJ
    INTEGER NSLCT,NMULT,NRNG,NSEQM,NDESL,ISEQ,IRES,ISEG
    INTEGER ISTP,IPOINT,NMISS,I,IDX(10)
    !
#if KEY_STRINGM==1 /*  VO stringm v */
    logical :: qstr
    common /replicaio/ qstr ! need global variable
#endif
    !     START OF CODE
    !
    NSLCT=0
    DO I=1,NATOM
       IF(ISLCT(I).EQ.1) NSLCT=NSLCT+1
    ENDDO
    IF(NSLCT.EQ.0) THEN
       CALL WRNDIE(1,'<COORIO>','ZERO ATOMS SPECIFIED IN SELECTION')
       RETURN
    ENDIF
    !
    IF(NSLCT.LT.NATOM .AND. PRNLEV.GE.2) WRITE(OUTU,127)
127 FORMAT(' INFO: A SUBSET OF TOTAL ATOMS WILL BE READ.'/)
    !
    ERRCNT=0
    !
    CALL TRYORO(IUNIT,'FORMATTED')
    !
    NTITLB=0
    !
    NRNG=0
    NSEQM=0
    NMULT=0
    NDESL=0
    DONE=.FALSE.
    !
#if KEY_STRINGM==1
    qstr=iunit.le.size(ifreeu) ! protect against OOB on ifreeu
    if (qstr) qstr=((MOD(IFREEU(IUNIT),8).eq.0).and.(IFREEU(IUNIT).ne.0).and.(ME_LOCAL.eq.0))
#endif
    !
    !     free field input, sort the starting column numbers
    IF (LFREE) CALL SORTP(10,IDX,ORDER,FMTSII,1,0,0,0,0,0,0)
    !
1000 CONTINUE
    !     Begin Procedure READ-LINE-AND-UNFORMAT
    !
1100 CONTINUE
    IF(IOLEV.GT.0 &
#if KEY_STRINGM==1 /*  VO */
&                 .or.qstr & 
#endif
&                          ) READ(IUNIT,'(A)',END=83,ERR=83) PDBLIN
#if KEY_PARALLEL==1
!MFC--- I think this is wrong, parallel ensemble should never send to world
#if KEY_PARALLEL==1 && KEY_ENSEMBLE==1
!--    CALL PSNDC_WORLD(PDBLIN,1) 
#endif
    CALL PSNDC(PDBLIN,1) 
#endif 
    IF(PDBLIN.EQ.'END-OF-FILE') GOTO 61
    IF (PDBLIN(1:3).EQ.'END') GOTO 61
    GOTO 86
83  CONTINUE
#if KEY_PARALLEL==1
    PDBLIN='END-OF-FILE'
!MFC--- I think this is wrong, parallel ensemble should never send to world
#if KEY_PARALLEL==1 && KEY_ENSEMBLE==1
!--    CALL PSNDC_WORLD(PDBLIN,1) 
#endif
    CALL PSNDC(PDBLIN,1) 
#endif 
    GOTO 61
86  CONTINUE
    !
    QATOM=.FALSE.
    DO I=1,FMTNTL
       IF(FMTTIT(I).EQ.PDBLIN(FMTTII(I):FMTTIF(I))) THEN
          IF(NTITLB.LT.MAXTIT) THEN
             NTITLB=NTITLB+1
             TITLEB(NTITLB)=PDBLIN
          ENDIF
          GOTO 150
       ENDIF
    ENDDO
    !
    !     reject excluded lines
    DO I=1,FMTNEX
       IF(FMTEXC(I).EQ.PDBLIN(FMTEXI(I):FMTEXF(I))) GOTO 150
    ENDDO
    !
    !     accept only lines that match the selection characters
    IF (FMTNSL.GT.0) THEN
       QATOM=.FALSE.
       DO I=1,FMTNSL
          IF(FMTSLC(I).EQ.PDBLIN(FMTSLI(I):FMTSLF(I))) QATOM=.TRUE.
       ENDDO
    ELSE
       QATOM=.TRUE.
    ENDIF
    !
150 CONTINUE
    IF (.NOT.QATOM) GOTO 1100
    !
    ISEQ=0
    IRES=0
    RESIN=' '
    ATOMIN=' '
    XIN=0.0
    YIN=0.0
    ZIN=0.0
    SID=' '
    RID=' '
    WIN=0.0
    !
    IF (LFREE) THEN
       !       free field input, sort the starting column numbers
       DO I=1,10
          IJ=LEN(PDBLIN)
          CALL TRIME(PDBLIN,IJ)
          !
          IF(IJ.EQ.0) GOTO 732
          IF (IDX(I).EQ.1) THEN
             SID=NEXTA8(PDBLIN,IJ)
          ELSE IF (IDX(I).EQ.2) THEN
             RID=NEXTA8(PDBLIN,IJ)
          ELSE IF (IDX(I).EQ.3) THEN
             IRES=NEXTI(PDBLIN,IJ)+IOFFS
          ELSE IF (IDX(I).EQ.4) THEN
             RESIN=NEXTA8(PDBLIN,IJ)
          ELSE IF (IDX(I).EQ.5) THEN
             ATOMIN=NEXTA8(PDBLIN,IJ)
          ELSE IF (IDX(I).EQ.6) THEN
             ISEQ=NEXTI(PDBLIN,IJ)
          ELSE IF (IDX(I).EQ.7) THEN
             XIN=NEXTF(PDBLIN,IJ)
          ELSE IF (IDX(I).EQ.8) THEN
             YIN=NEXTF(PDBLIN,IJ)
          ELSE IF (IDX(I).EQ.9) THEN
             ZIN=NEXTF(PDBLIN,IJ)
          ELSE IF (IDX(I).EQ.10) THEN
             WIN=NEXTF(PDBLIN,IJ)
          ENDIF
       ENDDO
       CALL TRIME(PDBLIN,IJ)
       IF(IJ.GT.0) CALL XTRANE(PDBLIN,IJ,'CREAD')
732    CONTINUE
    ELSE
       !
       !
       !       READ(PDBLIN,'(6X,I5,1X,A4,1X,A4,2X,A4,3X,3F8.3,6X,F6.2,6X,A4)')
       !       &      ISEQ,ATOMIN,RESIN,RID,XIN,YIN,ZIN,WIN,SID
       !       QATOM=.TRUE.
       !
       !       READ(IUNIT,40,ERR=61,END=61) ISEQ,IRES,RESIN,ATOMIN,XIN,YIN,ZIN,
       !       1    SID,RID,WIN
       !       40 FORMAT(2I5,2(1X,A4),3F10.5,1X,A4,1X,A4,2F10.5)
       !
       !       process segid
       IF(FMTSII.GT.0) THEN
          SID=PDBLIN(FMTSII:FMTSIF)
          IF(SID.EQ.' ') SID='NONE'
          IJ=8
          CALL TRIMA(SID,IJ)
          DO I=1,FMTNSI
             IF(SID.EQ.FMTSIC(I)) THEN
                SID=FMTSIM(I)
                GOTO 312
             ENDIF
          ENDDO
       ENDIF
312    CONTINUE
       !
       !       process resid
       IF(FMTRII.GT.0) THEN
          RID=PDBLIN(FMTRII:FMTRIF)
          IF(RID.EQ.' ') RID='NONE'
          IJ=8
          CALL TRIMA(RID,IJ)
          DO I=1,FMTNRI
             IF(RID.EQ.FMTRIC(I)) THEN
                IF(FMTRIS(I).EQ.' ' .OR. FMTRIS(I).EQ.SID) THEN
                   RID=FMTRIM(I)
                   GOTO 323
                ENDIF
             ENDIF
          ENDDO
       ENDIF
323    CONTINUE
       !
       !       process ires
       IF(FMTIRI.GT.0) THEN
          IJ=FMTIRF-FMTIRI+1
          IRES=DECODI(PDBLIN(FMTIRI:FMTIRF),IJ)
          IRES=IRES+IOFFS
          !         IF(PRNLEV.GE.2) WRITE(6,88) PDBLIN(FMTIRI:FMTIRF),IRES
          !         88  FORMAT(' >>>',A,'<<<',I10)
       ENDIF
       !
       !       process residue type
       IF(FMTRNI.GT.0) THEN
          RESIN=PDBLIN(FMTRNI:FMTRNF)
          IJ=8
          CALL TRIMA(RESIN,IJ)
          DO I=1,FMTNRN
             IF(RESIN.EQ.FMTRNC(I)) THEN
                IF(FMTRNS(I).EQ.' ' .OR. FMTRNS(I).EQ.SID) THEN
                   RESIN=FMTRNM(I)
                   GOTO 333
                ENDIF
             ENDIF
          ENDDO
       ENDIF
333    CONTINUE
       !
       !       process ATOM line
       !
       IF(FMTATI.GT.0) THEN
          ATOMIN=PDBLIN(FMTATI:FMTATF)
          IF(ATOMIN.EQ.' ') ATOMIN='NONE'
          IJ=8
          CALL TRIMA(ATOMIN,IJ)
          DO I=1,FMTNAT
             IF(ATOMIN.EQ.FMTATC(I)) THEN
                IF (FMTATS(I).NE.' ' .AND. FMTATS(I).NE.SID) THEN
                ELSE IF (FMTATR(I).NE.' ' .AND. FMTATR(I).NE.RID) THEN
                ELSE IF (FMTATN(I).NE.' ' .AND. FMTATN(I).NE.RESIN) THEN
                ELSE 
                   ATOMIN=FMTATM(I)
                   GOTO 343
                ENDIF
             ENDIF
          ENDDO
       ENDIF
343    CONTINUE
       !
       !       process iseq
       IF(FMTISI.GT.0) THEN
          IJ=FMTISF-FMTISI+1
          ISEQ=DECODI(PDBLIN(FMTISI:FMTISF),IJ)
       ENDIF
       !
       !       process XIN
       IF(FMTIXI.GT.0) THEN
          IJ=FMTIXF-FMTIXI+1
          XIN=DECODF(PDBLIN(FMTIXI:FMTIXF),IJ)
       ENDIF
       !
       !       process YIN
       IF(FMTIYI.GT.0) THEN
          IJ=FMTIYF-FMTIYI+1
          YIN=DECODF(PDBLIN(FMTIYI:FMTIYF),IJ)
       ENDIF
       !
       !       process ZIN
       IF(FMTIZI.GT.0) THEN
          IJ=FMTIZF-FMTIZI+1
          ZIN=DECODF(PDBLIN(FMTIZI:FMTIZF),IJ)
       ENDIF
       !
       !       process WIN
       IF(FMTIWI.GT.0) THEN
          IJ=FMTIWF-FMTIWI+1
          WIN=DECODF(PDBLIN(FMTIWI:FMTIWF),IJ)
       ENDIF
       !
       !       finished parsing line values
       !
    ENDIF
    !
    !     get the IRES value if segid,resid values are used.
    !
    IF(LRSID) THEN
       !       get residue number from resid and segid fields
       ISEG=1
       DO WHILE (SEGID(ISEG).NE.SID)
          ISEG=ISEG+1
          IF(ISEG.GT.NSEG) GOTO 888
       ENDDO
       IRES=NICTOT(ISEG)+1
       ISTP=NICTOT(ISEG+1)
       IF(IRES.GT.ISTP) GOTO 888
       DO WHILE (RESID(IRES).NE.RID)
          IRES=IRES+1
          IF(IRES.GT.ISTP) GOTO 888
       ENDDO
       GOTO 889
888    IRES=-99999999
    ENDIF
889 CONTINUE
    !
    !     End Procedure READ-LINE-AND-UNFORMAT
    !
    !     the sequence matches the psf
    !     the atom type is located properly
    !     no coordinates are multiply defined
    !     that the coordinates are within the desired interval
    !
    IF(IRES.LT.1.OR.IRES.GT.NRES) THEN
       NRNG=NRNG+1
       IF (NRNG.LT.5 .AND. WRNLEV.GE.2) WRITE(OUTU,82) &
            SID,RID,IRES,RESIN,ATOMIN
82     FORMAT(/' **** WARNING **** IN CREAD. COULD NOT FIND RESIDUE.' &
            ,/ &
            '  SEGID=',A8,' RESID=',A8,'  IRES=',I7,' RESNAME= ',A8, &
            ' TYPE= ',A8)
       GOTO 60
    ENDIF
    !
    IF(RES(IRES).NE.RESIN) THEN
       NSEQM=NSEQM+1
       IF(NSEQM.LE.5 .AND. WRNLEV.GE.2) WRITE(OUTU,85) &
            IRES,RES(IRES),RESIN
85     FORMAT(/' SEQUENCE MISMATCH AT',I5,' PSF= ',A8,' INPUT= ',A8)
    ENDIF
    !
    ATOMN4=ATOMIN
    IPOINT=MATOM(IRES,ATOMN4,ATYPE,IBASE,IRES,IRES,.FALSE.)
    IF (IPOINT.LT.0) THEN
       ERRCNT=ERRCNT+1
       IF (ERRCNT.LT.10 .AND. WRNLEV.GE.2) WRITE(OUTU,45) &
            ISEQ,IRES,RESID(IRES),RESIN,ATOMIN
45     FORMAT(' ****  WARNING  **** UNABLE TO FIND ATOM.',/ &
            ' INDEX=',I5,' IRES=',I5,' RESID=',A8,' RES=',A8,' ATOM=',A8)
    ELSE IF (IPOINT.LT.1.OR.IPOINT.GT.NATOM) THEN
       NRNG=NRNG+1
       IF(NRNG.LE.5 .AND. WRNLEV.GE.2) WRITE(OUTU,50) &
            ISEQ,IRES,RESIN,ATOMIN,IPOINT
50     FORMAT(' *****  WARNING  ***** COORDINATES OUT OF INTERVAL', &
            ' IGNORED',2I5,2(1X,A8),'  MAPPED TO:',I5)
    ELSE IF (ISLCT(IPOINT).EQ.1) THEN
       IF (INITIA(IPOINT,X,Y,Z)) NMULT=NMULT+1
       X(IPOINT)=XIN
       Y(IPOINT)=YIN
       Z(IPOINT)=ZIN
       WMAIN(IPOINT)=WIN
    ELSE
       NDESL=NDESL+1
    ENDIF
60  CONTINUE
    IF (.NOT.DONE) GOTO 1000
    !
61  CONTINUE
    !
    IF(NTITLB.EQ.0) THEN
       NTITLB=1
       TITLEB(1)='NONE'
    ENDIF
    CALL WRTITL(TITLEB,NTITLB,OUTU,1)
    !
    !     CHECK TO SEE THAT ALL THE DESIRED COORDINATES WERE FOUND
    !
    IRES=1
    NMISS=0
    DO I=1,NATOM
       IF (I.GT.IBASE(IRES+1)) IRES=IRES+1
       IF(ISLCT(I).EQ.1) THEN
          IF (.NOT.(INITIA(I,X,Y,Z))) THEN
             NMISS=NMISS+1
             IF (NMISS.LT.6 .AND. WRNLEV.GE.2) WRITE(OUTU,65) &
                  I,IRES,RES(IRES),ATYPE(I)
65           FORMAT(' ** WARNING ** NO COORDINATES FOR',2I5,2(1X,A8))
          ENDIF
       ENDIF
    ENDDO
    !
    IF(NMISS.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,74) NMISS
74  FORMAT(/' ** TOTAL OF',I5,' MISSING COORDINATES')
    IF (ERRCNT.GT.5 .AND. WRNLEV.GE.2) WRITE(OUTU,75) ERRCNT
75  FORMAT(/' ** TOTAL OF',I5,' ERRORS DURING COORDINATE ', &
         'READING **')
    IF (NMULT.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,55) NMULT
55  FORMAT(/' ** WARNING ** MULTIPLE COORDINATES FOR ',I5,' ATOMS.')
    IF(NDESL.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,77) NDESL
77  FORMAT(/' ** MESSAGE **',I5,' COORDINATES WERE IGNORED ', &
         'BECAUSE OF THE SPECIFIED SELECTION.')
    IF (NRNG.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,78) NRNG
78  FORMAT(/' ** MESSAGE **',I5,' COORDINATES IN THE INPUT FILE', &
         ' WERE OUTSIDE THE SPECIFIED SEQUENCE RANGE.')
    IF(NMISS+ERRCNT+NMULT.GT.0) CALL DIEWRN(2)
    IF (NSEQM.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,79) NSEQM
79  FORMAT(/' ** WARNING **',I5,' COORDINATES HAD A SEQUENCE', &
         ' MISMATCH.')
    RETURN
    !
    !======================================================================
    !
  END SUBROUTINE CREADU

  SUBROUTINE RUNIVF(IUNIT)
    !
    !     PDB    - setup brookhaven format (default)
    !     CHARMM - setup standard CHARMM format
    !     AMBER  - setup standard AMBER  format
    !     UNKNown - setup null format (everything must be specified)
    !
    !     SEGID start length
    !     RESID start length
    !     TYPE  start length
    !     RESN  start length
    !     IRES  start length
    !     ISEQ  start length
    !
    !     X     start length
    !     Y     start length
    !     Z     start length
    !     W     start length
    !
    !     PICK  start length  string
    !     EXCL  start length  string
    !     TITL  start length  string
    !
    !     TRANslate { SEGID ext-segid int-segid                         }
    !     { RESID ext-resid int-resid match-segid             }
    !     { RESN  ext-resn  int-resn  match-segid             }
    !     { TYPE  ext-type  int-type  match-resn  match-segid }
    !
    !     END
    !
    !C
    use chm_kinds
    use dimens_fcm
    use comand
    use exfunc
    use stream
    use string
    use ctitla

    implicit none
    !
    INTEGER IUNIT
    !
    !
    CHARACTER(len=4) WRD
    LOGICAL DONE,EOF
    !
    DONE=.FALSE.
    EOF=.FALSE.
    !
    IF(IUNIT.LT.-1) THEN
       !       PROCESS-CHARMM-FORMAT-COMMAND
       CALL CHRMFMT
       RETURN
    ENDIF
    !
    CALL TRYORO(IUNIT,'FORMATTED')
    CALL RDTITL(TITLEB,NTITLB,IUNIT,0)
    IF(PRNLEV.GE.2) WRITE(OUTU,13) IUNIT
13  FORMAT(/5X,'READING UNIVERSAL COORDINATE', &
         ' FORMAT SPECIFICATION FROM UNIT',I5/)
    CALL WRTITL(TITLEB,NTITLB,OUTU,1)
    !
9000 CONTINUE
    CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE., &
         .TRUE.,'RUNIVF> ')
    WRD=NEXTA4(COMLYN,COMLEN)
    IF (WRD.EQ.'    ') THEN
    ELSE IF (WRD.EQ.'PDB ') THEN
       !       Begin Procedure PROCESS-PDB-FORMAT-COMMAND
       !       SEGID
       FMTSII=22
       FMTSIF=22
       FMTNSI=0
       !       character(len=8) :: FMTSIC,FMTSIM
       !       RESID
       FMTRII=23
       FMTRIF=26
       FMTNRI=0
       !       character(len=8) :: FMTRIC,FMTRIM,FMTRIS
       !       IRES
       FMTIRI=23
       FMTIRF=26
       !       RESN
       FMTRNI=18
       FMTRNF=21
       FMTNRN=0
       !       character(len=8) :: FMTRNC,FMTRNM,FMTRNS
       !       ATOMS
       FMTATI=13
       FMTATF=16
       FMTNAT=0
       !       character(len=8) :: FMTATC,FMTATM,FMTATS,FMTATR,FMTATN
       !       ISEQ
       FMTISI=7
       FMTISF=11
       !       X,Y,Z,W
       FMTIXI=31
       FMTIXF=38
       FMTIYI=39
       FMTIYF=46
       FMTIZI=47
       FMTIZF=54
       FMTIWI=55
       FMTIWF=60
       !       PICK OPTION
       FMTNSL=1
       FMTSLI(1)=1
       FMTSLF(1)=6
       FMTSLC(1)='ATOM'
       !       EXCLUDE OPTION
       FMTNEX=0
       FMTEXI(1)=0
       FMTEXF(1)=0
       FMTEXC(1)='JUNK'
       !       TITLE OPTION
       FMTNTL=2
       FMTTII(1)=1
       FMTTIF(1)=6
       FMTTIT(1)='HEADER'
       FMTTII(2)=1
       FMTTIF(2)=6
       FMTTIT(2)='COMPND'
       !       End Procedure PROCESS-PDB-FORMAT-COMMAND
    ELSE IF (WRD.EQ.'CHAR') THEN
       !       PROCESS-CHARMM-FORMAT-COMMAND
       CALL CHRMFMT
    ELSE IF (WRD.EQ.'AMBE') THEN
       !       Begin Procedure PROCESS-AMBER-FORMAT-COMMAND
       !       SEGID
       FMTSII=0
       FMTSIF=0
       FMTNSI=0
       !       character(len=8) :: FMTSIC,FMTSIM
       !       RESID
       FMTRII=0
       FMTRIF=0
       FMTNRI=0
       !       character(len=8) :: FMTRIC,FMTRIM,FMTRIS
       !       IRES
       FMTIRI=0
       FMTIRF=0
       !       RESN
       FMTRNI=0
       FMTRNF=0
       FMTNRN=0
       !       character(len=8) :: FMTRNC,FMTRNM,FMTRNS
       !       ATOMS
       FMTATI=0
       FMTATF=0
       FMTNAT=0
       !       character(len=8) :: FMTATC,FMTATM,FMTATS,FMTATR,FMTATN
       !       ISEQ
       FMTISI=0
       FMTISF=0
       !       X,Y,Z,W
       FMTIXI=0
       FMTIXF=0
       FMTIYI=0
       FMTIYF=0
       FMTIZI=0
       FMTIZF=0
       FMTIWI=0
       FMTIWF=0
       !       PICK OPTION
       FMTNSL=1
       FMTSLI(1)=0
       FMTSLF(1)=0
       FMTSLC(1)='JUNK'
       !       EXCLUDE OPTION
       FMTNEX=1
       FMTEXI(1)=0
       FMTEXF(1)=0
       FMTEXC(1)='JUNK'
       !       TITLE OPTION
       FMTNTL=1
       FMTTII(1)=0
       FMTTIF(1)=0
       FMTTIT(1)='JUNK'
       !       End Procedure PROCESS-AMBER-FORMAT-COMMAND
    ELSE IF (WRD.EQ.'UNKN') THEN
       !       Begin Procedure PROCESS-UNKNOWN-FORMAT-COMMAND
       !       SEGID
       FMTSII=0
       FMTSIF=0
       FMTNSI=0
       !       character(len=8) :: FMTSIC,FMTSIM
       !       RESID
       FMTRII=0
       FMTRIF=0
       FMTNRI=0
       !       character(len=8) :: FMTRIC,FMTRIM,FMTRIS
       !       IRES
       FMTIRI=0
       FMTIRF=0
       !       RESN
       FMTRNI=0
       FMTRNF=0
       FMTNRN=0
       !       character(len=8) :: FMTRNC,FMTRNM,FMTRNS
       !       ATOMS
       FMTATI=0
       FMTATF=0
       FMTNAT=0
       !       character(len=8) :: FMTATC,FMTATM,FMTATS,FMTATR,FMTATN
       !       ISEQ
       FMTISI=0
       FMTISF=0
       !       X,Y,Z,W
       FMTIXI=0
       FMTIXF=0
       FMTIYI=0
       FMTIYF=0
       FMTIZI=0
       FMTIZF=0
       FMTIWI=0
       FMTIWF=0
       !       PICK OPTION
       FMTNSL=0
       FMTSLI(1)=0
       FMTSLF(1)=0
       FMTSLC(1)='JUNK'
       !       EXCLUDE OPTION
       FMTNEX=0
       FMTEXI(1)=0
       FMTEXF(1)=0
       FMTEXC(1)='JUNK'
       !       TITLE OPTION
       FMTNTL=0
       FMTTII(1)=0
       FMTTIF(1)=0
       FMTTIT(1)='JUNK'
       !       End Procedure PROCESS-UNKNOWN-FORMAT-COMMAND
    ELSE IF (WRD.EQ.'SEGI') THEN
       !       Begin Procedure PROCESS-SEGID-COMMAND
       FMTSII=NEXTI(COMLYN,COMLEN)
       FMTSIF=NEXTI(COMLYN,COMLEN)+FMTSII-1
       FMTNSI=0
       !       End Procedure PROCESS-SEGID-COMMAND
    ELSE IF (WRD.EQ.'RESI') THEN
       !       Begin Procedure PROCESS-RESID-COMMAND
       FMTRII=NEXTI(COMLYN,COMLEN)
       FMTRIF=NEXTI(COMLYN,COMLEN)+FMTRII-1
       FMTNRI=0
       !       End Procedure PROCESS-RESID-COMMAND
    ELSE IF (WRD.EQ.'TYPE') THEN
       !       Begin Procedure PROCESS-TYPE-COMMAND
       FMTATI=NEXTI(COMLYN,COMLEN)
       FMTATF=NEXTI(COMLYN,COMLEN)+FMTATI-1
       FMTNAT=0
       !       End Procedure PROCESS-TYPE-COMMAND
    ELSE IF (WRD.EQ.'RESN') THEN
       !       Begin Procedure PROCESS-RESN-COMMAND
       FMTRNI=NEXTI(COMLYN,COMLEN)
       FMTRNF=NEXTI(COMLYN,COMLEN)+FMTRNI-1
       FMTNRN=0
       !       End Procedure PROCESS-RESN-COMMAND
    ELSE IF (WRD.EQ.'IRES') THEN
       !       Begin Procedure PROCESS-IRES-COMMAND
       FMTIRI=NEXTI(COMLYN,COMLEN)
       FMTIRF=NEXTI(COMLYN,COMLEN)+FMTIRI-1
       !       End Procedure PROCESS-IRES-COMMAND
    ELSE IF (WRD.EQ.'ISEQ') THEN
       !       Begin Procedure PROCESS-IESQ-COMMAND
       FMTISI=NEXTI(COMLYN,COMLEN)
       FMTISF=NEXTI(COMLYN,COMLEN)+FMTISI-1
       !       End Procedure PROCESS-IESQ-COMMAND
    ELSE IF (WRD.EQ.'X   ') THEN
       !       Begin Procedure PROCESS-X-COMMAND
       FMTIXI=NEXTI(COMLYN,COMLEN)
       FMTIXF=NEXTI(COMLYN,COMLEN)+FMTIXI-1
       !       End Procedure PROCESS-X-COMMAND
    ELSE IF (WRD.EQ.'Y   ') THEN
       !       Begin Procedure PROCESS-Y-COMMAND
       FMTIYI=NEXTI(COMLYN,COMLEN)
       FMTIYF=NEXTI(COMLYN,COMLEN)+FMTIYI-1
       !       End Procedure PROCESS-Y-COMMAND
    ELSE IF (WRD.EQ.'Z   ') THEN
       !       Begin Procedure PROCESS-Z-COMMAND
       FMTIZI=NEXTI(COMLYN,COMLEN)
       FMTIZF=NEXTI(COMLYN,COMLEN)+FMTIZI-1
       !       End Procedure PROCESS-Z-COMMAND
    ELSE IF (WRD.EQ.'W   ') THEN
       !       Begin Procedure PROCESS-W-COMMAND
       FMTIWI=NEXTI(COMLYN,COMLEN)
       FMTIWF=NEXTI(COMLYN,COMLEN)+FMTIWI-1
       !       End Procedure PROCESS-W-COMMAND
    ELSE IF (WRD.EQ.'PICK') THEN
       !       Begin Procedure PROCESS-PICK-COMMAND
       FMTNSL=FMTNSL+1
       FMTSLI(FMTNSL)=NEXTI(COMLYN,COMLEN)
       FMTSLF(FMTNSL)=NEXTI(COMLYN,COMLEN)+FMTSLI(FMTNSL)-1
       CALL TRIMA(COMLYN,COMLEN)
       FMTSLC(FMTNSL)=COMLYN
       !       End Procedure PROCESS-PICK-COMMAND
    ELSE IF (WRD.EQ.'EXCL') THEN
       !       Begin Procedure PROCESS-EXCLUDE-COMMAND
       FMTNEX=FMTNEX+1
       FMTEXI(FMTNEX)=NEXTI(COMLYN,COMLEN)
       FMTEXF(FMTNEX)=NEXTI(COMLYN,COMLEN)+FMTEXI(FMTNEX)-1
       CALL TRIMA(COMLYN,COMLEN)
       FMTEXC(FMTNEX)=COMLYN
       !       End Procedure PROCESS-EXCLUDE-COMMAND
    ELSE IF (WRD.EQ.'TITL') THEN
       !       Begin Procedure PROCESS-TITLE-COMMAND
       FMTNTL=FMTNTL+1
       FMTTII(FMTNTL)=NEXTI(COMLYN,COMLEN)
       FMTTIF(FMTNTL)=NEXTI(COMLYN,COMLEN)+FMTTII(FMTNTL)-1
       CALL TRIMA(COMLYN,COMLEN)
       FMTTIT(FMTNTL)=COMLYN
       !       End Procedure PROCESS-TITLE-COMMAND
    ELSE IF (WRD.EQ.'TRAN') THEN
       !       Begin Procedure PROCESS-TRANSLATE-COMMAND
       WRD=NEXTA4(COMLYN,COMLEN)
       IF (WRD.EQ.'    ') THEN
       ELSE IF (WRD.EQ.'SEGI') THEN
          FMTNSI=FMTNSI+1
          FMTSIC(FMTNSI)=NEXTA4(COMLYN,COMLEN)
          FMTSIM(FMTNSI)=NEXTA4(COMLYN,COMLEN)
       ELSE IF (WRD.EQ.'RESI') THEN
          FMTNRI=FMTNRI+1
          FMTRIC(FMTNRI)=NEXTA4(COMLYN,COMLEN)
          FMTRIM(FMTNRI)=NEXTA4(COMLYN,COMLEN)
          FMTRIS(FMTNRI)=NEXTA4(COMLYN,COMLEN)
       ELSE IF (WRD.EQ.'TYPE') THEN
          FMTNAT=FMTNAT+1
          FMTATC(FMTNAT)=NEXTA4(COMLYN,COMLEN)
          FMTATM(FMTNAT)=NEXTA4(COMLYN,COMLEN)
          FMTATN(FMTNAT)=NEXTA4(COMLYN,COMLEN)
          FMTATS(FMTNAT)=NEXTA4(COMLYN,COMLEN)
          FMTATR(FMTNAT)=NEXTA4(COMLYN,COMLEN)
       ELSE IF (WRD.EQ.'RESN') THEN
          FMTNRN=FMTNRN+1
          FMTRNC(FMTNRN)=NEXTA4(COMLYN,COMLEN)
          FMTRNM(FMTNRN)=NEXTA4(COMLYN,COMLEN)
          FMTRNS(FMTNRN)=NEXTA4(COMLYN,COMLEN)
       ELSE 
          IF(WRNLEV.GE.2) WRITE(OUTU,39) WRD
          CALL DIEWRN(0)
       ENDIF
       !       End Procedure PROCESS-TRANSLATE-COMMAND
    ELSE IF (WRD.EQ.'END ') THEN
       IF(.TRUE.) RETURN
    ELSE 
       IF(WRNLEV.GE.2) WRITE(OUTU,39) WRD
39     FORMAT(' ** UNRECOGNIZED OPERATION "',A4,'" IN RUNIVF **'/)
       CALL DIEWRN(0)
    ENDIF
    !
    IF (.NOT.DONE) GOTO 9000
    RETURN
    !--------------------------------------------
  END SUBROUTINE RUNIVF

  SUBROUTINE CHRMFMT
    !     TO PROCESS-CHARMM-FORMAT-COMMAND
    use chm_kinds
    use dimens_fcm
    implicit none
    !
    !     SEGID
    FMTSII=52
    FMTSIF=55
    FMTNSI=0
    !     character(len=8) :: FMTSIC,FMTSIM
    !     RESID
    FMTRII=57
    FMTRIF=60
    FMTNRI=0
    !     character(len=8) :: FMTRIC,FMTRIM,FMTRIS
    !     IRES
    FMTIRI=6
    FMTIRF=10
    !     RESN
    FMTRNI=12
    FMTRNF=15
    FMTNRN=0
    !     character(len=8) :: FMTRNC,FMTRNM,FMTRNS
    !     ATOMS
    FMTATI=17
    FMTATF=20
    FMTNAT=0
    !     character(len=8) :: FMTATC,FMTATM,FMTATS,FMTATR,FMTATN
    !     ISEQ
    FMTISI=1
    FMTISF=5
    !     X,Y,Z,W
    FMTIXI=21
    FMTIXF=30
    FMTIYI=31
    FMTIYF=40
    FMTIZI=41
    FMTIZF=50
    FMTIWI=61
    FMTIWF=70
    !     PICK OPTION
    FMTNSL=0
    FMTSLI(1)=0
    FMTSLF(1)=0
    !     character(len=8) :: FMTSLC
    !     EXCLUDE OPTION
    FMTNEX=1
    FMTEXI(1)=21
    FMTEXF(1)=30
    FMTEXC(1)=' '
    !     TITLE OPTION
    FMTNTL=1
    FMTTII(1)=1
    FMTTIF(1)=1
    FMTTIT(1)='*'
    RETURN
  END SUBROUTINE CHRMFMT
end module univ

