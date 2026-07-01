module corsol_mod
  use chm_kinds
  implicit none

contains
#if KEY_CORSOL==1 /*corsol*/
  SUBROUTINE CORSOL
    
    !             CORSOL is very much alike the original CORREL module (from
    !     which the framework was reused) but differs inasmuch as series are
    !     usually interactions between a solute and a solvent (up to now only
    !     TIP3 is implemented!) and are filled for a whole array of atoms.

    !     Tibor Rudas 12. 02. 2003 - 

    use chm_kinds
    use dimens_fcm
    use exfunc

    use comand
    use image
    use psf
    use stream
    use coord
    use coordc
    use memory
    use string

    implicit none

    INTEGER :: MAXSER,MAXTOT
    INTEGER :: QSIZE
    LOGICAL :: DOIMAGES

    MAXSER=GTRMI(COMLYN,COMLEN,'MAXS',2)
    MAXTOT=GTRMI(COMLYN,COMLEN,'MAXT',512)
    QSIZE=GTRMI(COMLYN,COMLEN,'MAXA',100)

    ! parse options and generate lists, unless not asked not to do so
    ! sb+ln add testing for presence of NOUPdate keyword
    IF(.NOT. (INDXA(COMLYN,COMLEN,'NOUP')  > 0) )THEN
       CALL UPDATE(COMLYN, COMLEN, X,Y,Z,WMAIN,.TRUE., &
            .TRUE.,.TRUE.,.TRUE.,.TRUE.,0,0,0,0,0,0,0)
    ENDIF


    DOIMAGES=.FALSE.
    IF(NATIM > NATOM) DOIMAGES=.TRUE.

    CALL CORSO1(MAXSER,MAXTOT,QSIZE,DOIMAGES)


    RETURN
  END SUBROUTINE CORSOL

  !
  !----------------------------------------------------------------------
  !

  SUBROUTINE CORSO1(MAXSER,MAXTOT,QSIZE,DOIMAGES)

    !             This is the work routine it interprets commands given in
    !     the commandline, fills in series description.
    !
    !     THE TIMESERIES DATA STRUCTURE
    !     MXNTOT - Maximum number of time entries
    !     NSERIE - Number of time series to compute
    !     SNAME(NSERIE) - Name of series
    !     ICLASS(NSERIE) - class name for each series
    !     STOT(NSERIE) - Number of entries in time series
    !     VELCOD(NSERIE) - Logical flag for time series from velocities
    !     SSKIP(NSERIE) - SKIP value for the time series
    !     SDELTA(NSERIE) - time interval between 1/SKIP steps.
    !     SOFFST(NSERIE) - time of first element.
    !     GECOD(NSERIE) - Flag specifying geometry as opposed to energy
    !     VECCOD(NSERIE) - Vector code (1 or 3)
    !     SAVEG(NSERIE) - Average value for time series
    !     SFLCT(NSERIE) - Fluctuation value for each series
    !     SERPT(NSERIE) - Pointer to start of QAT atom descriptor
    !     SERNQ(NSERIE) - Number of atoms for this series
    !     SERVAL(NSERIE) - Number of primary atoms for this series
    !     TQ(MXNTOT,NSERIE+1) - Time series values
    !     IUNMAX(NSERIE) - unit no for output of INOMAX max contributions
    !     INOMAX(NSERIE) - nr of maximum contributions to write out
    !
    !     SERSET - is a NSRST long list with the starting number of
    !              each individual series
    !
    !     SERDAT
    !
    !     Tibor Rudas 12. 02. 2003 - 

    use chm_kinds
    use chm_types
    use dimens_fcm
    use exfunc
    use number
    use comand
    use consta
    use psf
    use coord
    use coordc
    use stream
    use string
    use ctitla
    use memory
    use image
    use cvio,only: trjspc
#if KEY_SHELL==1
    use shell     
#endif
    use ancsol_mod,only:ancsol,serdat_type
    use select
    use memory

    implicit none

    ! passed arguments:
    integer :: maxser,maxtot,qsize
    logical :: doimages

    ! allocatable data-structure
    integer,allocatable,dimension(:)   :: islct,jslct
    integer,allocatable,dimension(:)   :: velcod,gecod,veccod,stot,sskip,serpt,sernq
    integer,allocatable,dimension(:)   :: serset,serncr,iunmax,inomax,qat

    integer,allocatable,dimension(:,:) :: atomin
    integer,allocatable,dimension(:)   :: seract

    real(chm_real),allocatable,dimension(:,:) :: tq
    real(chm_real),allocatable,dimension(:) :: serval,sdelta,soffst,saveg,sflct

    logical,allocatable,dimension(:) :: lfft

    character(len=8),allocatable,dimension(:) :: sname,iclass

    ! additional series data
    type(serdat_type),dimension(maxser) :: serdat,sercor,serlst


    ! ---------

    character(len=4) SID,RID
    INTEGER TBT

    INTEGER GECD,VECCD,SERP,SERN,NCORR
    real(chm_real) SERVL
    character(len=4) NAME,ICLAS
    LOGICAL MASSWT
    !
    INTEGER NSERIE,ISERIE,JSERIE,NQ,NSERX,NSERXI
    INTEGER VELCD,NUNIT,FIRSTU,BEGIN,STOP,SKIP
    INTEGER VELCD2,NUNIT2,FIRSTU2,BEGIN2,STOP2,SKIP2
    INTEGER OLDT2E
    INTEGER NTOT,NPRI2,CRSCOD
    INTEGER NAVEG
    !      INTEGER I800
    character(len=4) MANCOD
    INTEGER IPN,OLDCOR,LFA
    INTEGER SFA,SFB,SFC,SFD,SFAH,SFBH
    INTEGER I,J,K,L,N,IS,IQ
    INTEGER ICODE,NSRST
    INTEGER IUNMX,INOMX
    real(chm_real) FACTOR,FACTR2,DELTA,OFFST
    LOGICAL LUSED,EOF,OK,ERR
    LOGICAL QDIFF,QFFT,QLTC,QNNORM,QFOLD,QRAMP,QSWIT,QORIENT,QMASS
    !
    INTEGER WDMAX,WDLEN
    PARAMETER (WDMAX=80)
    character(len=(WDMAX)) WD
    integer maxsr, icorr
    !
    ! QCRD/QVEL/QBOTH: reading COOR/VEL or both? QAFFT: is at least one FFT
    !                                                   series used?
    LOGICAL QCRD,QVEL,QBOTH,QAFFT

    INTEGER,parameter :: NSPN=2
    CHARACTER(len=2),dimension(nspn) :: SPN=(/'P1','P2'/)

    INTEGER,PARAMETER :: MARK = -99999999



    maxsr = maxser + 1
    icorr = maxsr

    call chmalloc('corsol.src','CORSO1','ISLCT',NATOM,intg=ISLCT)
    call chmalloc('corsol.src','CORSO1','JSLCT',NATOM,intg=JSLCT)
    call chmalloc('corsol.src','CORSO1','TQ',MAXTOT,maxsr,crl=TQ)

    call chmalloc('corsol.src','CORSO1','VELCOD',maxsr,intg=VELCOD)
    call chmalloc('corsol.src','CORSO1','GECOD',maxsr,intg=GECOD)
    call chmalloc('corsol.src','CORSO1','VECCOD',maxsr,intg=VECCOD)
    call chmalloc('corsol.src','CORSO1','STOT',maxsr,intg=STOT)
    call chmalloc('corsol.src','CORSO1','SSKIP',maxsr,intg=SSKIP)
    call chmalloc('corsol.src','CORSO1','SERPT',maxsr,intg=SERPT)
    call chmalloc('corsol.src','CORSO1','SERNQ',maxsr,intg=SERNQ)

    call chmalloc('corsol.src','CORSO1','SERSET',maxser,intg=SERSET)
    call chmalloc('corsol.src','CORSO1','SERNCR',maxser,intg=SERNCR)
    call chmalloc('corsol.src','CORSO1','IUNMAX',maxser,intg=IUNMAX)
    call chmalloc('corsol.src','CORSO1','INOMAX',maxser,intg=INOMAX)

    call chmalloc('corsol.src','CORSO1','QAT',qsize,intg=QAT)

    call chmalloc('corsol.src','CORSO1','SERVAL',maxsr,crl=SERVAL)
    call chmalloc('corsol.src','CORSO1','SDELTA',maxsr,crl=SDELTA)
    call chmalloc('corsol.src','CORSO1','SOFFST',maxsr,crl=SOFFST)
    call chmalloc('corsol.src','CORSO1','SAVEG',maxsr,crl=SAVEG)
    call chmalloc('corsol.src','CORSO1','SFLCT',maxsr,crl=SFLCT)

    call chmalloc('corsol.src','CORSO1','lfft',  MAXSR,log=lfft)

    call chmalloc('corsol.src','CORSO1','sname', MAXSR,ch8=sname)
    call chmalloc('corsol.src','CORSO1','iclass',MAXSR,ch8=iclass)

    QCRD=.FALSE.
    QVEL=.FALSE.
    QBOTH=.FALSE.
    QAFFT=.FALSE.
    LUSED=.FALSE.
    EOF=.FALSE.
    NQ=0
    NSERIE=0
    NSRST=0
    !
    !     set up data for the correlation function.
    SNAME(ICORR)='CORR'
    ICLASS(ICORR)='CORR'
    STOT(ICORR)=0
    VELCOD(ICORR)=0
    SSKIP(ICORR)=1
    SDELTA(ICORR)=0.0
    SOFFST(ICORR)=0.0
    GECOD(ICORR)=0
    VECCOD(ICORR)=1
    SERVAL(ICORR)=0.0
    LFFT(ICORR)=.TRUE.
    SERNQ(ICORR)=0
    SERPT(ICORR)=MARK
    !
100 CONTINUE
    CALL XTRANE(COMLYN,COMLEN,'CORSOL')

    lused=.true.
    loop10: do while(lused .and. .not. eof)
       CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE., &
            .TRUE.,'CORSOL> ')
       CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
    enddo loop10
    !
    eof_block: IF(EOF) THEN
       IF(NSTRM  ==  1) then
          call corso1_deallocate
          RETURN
       endif
       CALL PPSTRM(OK)
       EOF=.FALSE.
       GOTO 100
    ENDIF eof_block
    WD=NEXTA4(COMLYN,COMLEN)
    IF(WD == '    ') GOTO 100
    !
    wd_block: IF(WD == 'ENTE') THEN
       ! PROCESS-ENTER-COMMAND
       CALL NEXTWD(COMLYN,COMLEN,WD,WDMAX,WDLEN)
       IF(WDLEN > 4) CALL WRNDIE(0,'<CORSOL>', &
            'ENTER name too long, truncated to 4 characters.')
       NAME=NEXTA4(WD,WDLEN)
       !
       WD=NEXTA4(COMLYN,COMLEN)
       !
       OK=.TRUE.
       VECCD=1
       GECD=1
       !TRD         IF(INDXA(COMLYN,COMLEN,'GEOM') > 0) GECD=1
       !TRD         IF(INDXA(COMLYN,COMLEN,'ENER') > 0) GECD=2
       SERVL=zero
       MASSWT=(INDXA(COMLYN,COMLEN,'FFT ') > 0)
       MASSWT=(MASSWT.OR.(INDXA(COMLYN,COMLEN,'FFT') > 0))
       IF(MASSWT) QAFFT=.TRUE.
       NCORR=GTRMI(COMLYN,COMLEN,'NCOR',0)
       IF((.NOT.MASSWT).AND.((NCORR <= 0).OR.(NCORR > MAXTOT))) &
            NCORR=MAXTOT
       SERP=NQ+1
       SERN=0
       !
       !        output of maximum contributions?
       IUNMX=GTRMI(COMLYN,COMLEN,'IMAX',0)
       INOMX=GTRMI(COMLYN,COMLEN,'NMAX',0)
       !
       ICODE=-1
       ICLAS=WD
       wd2_blk: IF(WD == 'DDIP') THEN
          !           DIPOLE DIPOLE CORRELATION WITH WATERS
          QCRD=.TRUE.
          GECD=0
          VECCD=3
          ICODE=1
       ELSE IF(WD == 'VACF') THEN   wd2_blk
          !           VELOCITY AUTO CORRELATION FUNCTION
          QVEL=.TRUE.
          GECD=0
          !           remember: VECCD gives the dimensionality of the RESULT (i.e
          !           the correlation function: in this case: 1D correlation function
          !           and 1D for the integral = diffusion coefficient)
          VECCD=2
          ICODE=0
          IF(INDXA(COMLYN,COMLEN,'CHAR') > 0) SERVL=one
       ELSE IF(WD == 'WREO') THEN   wd2_blk
          !           WATER SELF REORIENTATION CORRELATION FUNCTION
          QCRD=.TRUE.
          GECD=0
          VECCD=1
          ICODE=0
          IF(INDXA(COMLYN,COMLEN,'NORM') > 0) SERVL=one
          IF(INDXA(COMLYN,COMLEN,'P2') > 0)   SERVL=two
#if KEY_SHELL==1 /*shell*/
       ELSE IF(WD == 'SDDP') THEN   wd2_blk
          !           DIPOLE DIPOLE CORRELATION WITH WATERS IN A SHELL
          !           which shell do we want (default=first)
          IF(QSHELL) THEN
             QCRD=.TRUE.
             !             which shell do we want (default=first)
             GECD=GTRMI(COMLYN,COMLEN,'SHEL',1)
             !              or do we want bulk (overrides)
             IF(INDXA(COMLYN,COMLEN,'BULK') > 0) GECD=-1
             !              check for sanity
             IF(GECD > NSHL) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,870) GECD,NSHL
870             FORMAT(' ** WARNING ** from CORSOL -- Shell, ' &
                     ,I6,' is not defined.'/' It will be set to ' &
                     ,I6,'.')
                GECD=NSHL
             ENDIF
          ELSE
             CALL WRNDIE(-1,'<CORSOL>','SHELL is not set up...')
          ENDIF
          VECCD=3
          ICODE=1
       ELSE IF(WD == 'SATM') THEN   wd2_blk
          !           atom in shell x?
          IF(QSHELL) THEN
             QCRD=.TRUE.
             VECCD=1
             ICODE=0
             !              which shell do we want (default=first)
             GECD=GTRMI(COMLYN,COMLEN,'SHEL',1)
             !              or do we want bulk (overrides)
             IF(INDXA(COMLYN,COMLEN,'BULK') > 0) GECD=-1
             !              check for sanity
             IF(GECD > NSHL) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,870) GECD,NSHL
                GECD=NSHL
             ENDIF
          ELSE
             CALL WRNDIE(-1,'<CORSOL>','SHELL is not set up...')
          ENDIF
       ELSE IF(WD == 'SATX') THEN   wd2_blk
          !           atom in shell x? - cross correlation
          IF(QSHELL) THEN
             QCRD=.TRUE.
             ICODE=0
             !              which shell do we want (default=first)
             GECD=GTRMI(COMLYN,COMLEN,'SHL1',1)
             !              or do we want bulk (overrides)
             IF(INDXA(COMLYN,COMLEN,'BULK') > 0) GECD=-1
             !              check for sanity
             IF(GECD > NSHL) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,870) GECD,NSHL
                GECD=NSHL
             ENDIF
             !
             !              and the other shell?
             SERVL=GTRMI(COMLYN,COMLEN,'SHL2',2)
             !              or do we want bulk (overrides)
             IF(INDXA(COMLYN,COMLEN,'BULK') > 0) SERVL=-one
             !              check for sanity
             IF(SERVL > NSHL) THEN
                IF(GECD == NSHL) THEN
                   ! reset it to bulk
                   IF(WRNLEV >= 2) WRITE(OUTU,871) INT(SERVL)
                   SERVL=-one
871                FORMAT(' ** WARNING ** from CORSOL -- Shell, ' &
                        ,I6,' is not defined.'/' It will be set to BULK.')
                ELSE
                   ! reset to last shell
                   IF(WRNLEV >= 2) WRITE(OUTU,870) INT(SERVL),NSHL
                   SERVL=NSHL
                ENDIF
             ENDIF

             ! tr: TEMP! output of maximum contribution crashes sometimes
             !           and may give wrong output - disabled for now
             IF(IUNMX > 0) THEN
                CALL WRNDIE(-1,'<CORSOL>','IMAX/NMAX currently disabled for SATX.')
             ENDIF
          ELSE
             CALL WRNDIE(-1,'<CORSOL>','SHELL is not set up...')
          ENDIF
       ELSE IF(WD == 'MRTI') THEN   wd2_blk
          !           mean residence time (shell)
          IF(QSHELL) THEN
             QCRD=.TRUE.
             VECCD=1
             ICODE=0
             !              which shell do we want (default=first)
             GECD=GTRMI(COMLYN,COMLEN,'SHEL',1)
             !              or do we want bulk (overrides)
             IF(INDXA(COMLYN,COMLEN,'BULK') > 0) GECD=-1
             !              check for sanity
             IF(GECD > NSHL) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,870) GECD,NSHL
                GECD=NSHL
             ENDIF
          ELSE
             CALL WRNDIE(-1,'<CORSOL>','SHELL is not set up...')
          ENDIF
#else /* (shell)*/
       ELSE IF((WD == 'SDDP').OR.(WD.EQ.'SATM').OR. &   wd2_blk
          (WD == 'SATX').OR.(WD.EQ.'MRTI')) THEN
          CALL WRNDIE(0,'<CORSOL>', &
               'SHELL is not compiled...')
#endif /* (shell)*/
       ELSE   wd2_blk
          CALL WRNDIE(0,'<CORSOL>', &
               'Unrecognized time series specification')
          OK=.FALSE.
       ENDIF wd2_blk
       !
       icode_blk: IF(ICODE >= 0) THEN
          CALL INTIC3(COMLYN,COMLEN,ICODE,NQ,QAT, &
               QSIZE,SERN,GECD,SERVL)
       ENDIF icode_blk
       !
       !        one cannot calculate coor & velocity series simulatneously
       IF(QCRD.AND.QVEL) &
            CALL WRNDIE(-5,'<CORSOL>', &
            'Cannot fill COOR and VEL series simultanously!')
       !
       !        output of DDIP series for multiple atoms at once only up to 20
       IF((ICLAS == 'DDIP').AND.(SERVL > 20)) &
            CALL WRNDIE(-1,'<CORSOL>', &
            'More than 20 primary atoms not implemented!')
       !
       IF(NSERIE+VECCD > MAXSER) THEN
          CALL WRNDIE(0,'<CORSOL>', &
               'Maximum number of time series exceeded')
          IF(PRNLEV >= 2) WRITE(OUTU,'(A)')' ENTER command ignored'
          GOTO 100
       ENDIF
       !
       K=1
       IF(ICLAS == 'DDIP') K=SERVL
       !
       IF(NSERIE+(VECCD*K) > MAXSER) THEN
          CALL WRNDIE(0,'<CORSOL>', &
               'Maximum number of time series exceeded')
          IF(PRNLEV >= 2) WRITE(OUTU,'(A)')' ENTER command ignored'
          GOTO 100
       ENDIF
       !
       N=NSERIE
       DO L=1,K
          DO I=1,VECCD
             N=N+1
             ICLASS(N)=ICLAS
             VELCOD(N)=0
             IF(VECCD > 1) THEN
                GECOD(N)=I+GECD
             ELSE
                GECOD(N)=GECD
             ENDIF
             VECCOD(N)=0
             SERVAL(N)=SERVL
             LFFT(N)=MASSWT
             STOT(N)=0
             SSKIP(N)=1
             SDELTA(N)=0.0
             SOFFST(N)=0.0
             SAVEG(N)=0.0
             SFLCT(N)=0.0
             SERPT(N)=SERP
             SERNQ(N)=SERN
             SNAME(N)=NAME
             IUNMAX(N)=IUNMX
             INOMAX(N)=INOMX
             DO J=1,MAXTOT
                TQ(J,N)=0.0
             ENDDO
          ENDDO
       ENDDO
       VECCOD(NSERIE+1)=(VECCD*K)
       !
       !        set pointers for the data-structure to 0 (for testing)
       NSRST=NSRST+1
       SERSET(NSRST)=NSERIE+1
       SERNCR(NSRST)=NCORR
       serdat(nsrst)%col=0
       serdat(nsrst)%row=0
       serdat(nsrst)%spc=0
       serdat(nsrst)%pt  => null()
       serdat(nsrst)%ipt => null()
       sercor(nsrst)%col=0
       sercor(nsrst)%row=0
       sercor(nsrst)%spc=0
       sercor(nsrst)%pt  => null()
       sercor(nsrst)%ipt => null()
       serlst(nsrst)%col=0
       serlst(nsrst)%row=0
       serlst(nsrst)%spc=0
       serlst(nsrst)%pt  => null()
       serlst(nsrst)%ipt => null()
       !
       NSERIE=NSERIE+(VECCD*K)
       SERPT(NSERIE+1)=NQ+1
       !
    ELSE IF(WD == 'TRAJ') THEN
       !
       IF(.NOT.DOIMAGES.AND.QCRD) THEN
          WRITE(OUTU,'(A)') &
               ' *** WARNING *** Filling series without images'
          WRITE(OUTU,'(A)') &
               '                 Some series may be incorrect !'
       ENDIF
       !
       ! get-time-series
       CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT,X,Y,Z,WMAIN,.TRUE.,ERR)
       CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,BEGIN,SKIP,STOP)
       VELCD=0
       IF(INDXA(COMLYN,COMLEN,'VELO') > 0) VELCD=1
       !
       ! parse rms best fit options
       QORIENT=(INDXA(COMLYN,COMLEN,'ORIE') > 0)
       IF(QORIENT) THEN
          IF(PRNLEV >= 2) WRITE(OUTU,'(A)') &
               ' Orienting each frame wrt comparison coordinates'
          QMASS=(INDXA(COMLYN,COMLEN,'MASS') > 0)
       ENDIF
       !
       call chmalloc('corsol.src','CORSO1','ATOMIN',2,NATOM,intg=ATOMIN)
       call chmalloc('corsol.src','CORSO1','SERACT',MAXSER,intg=SERACT)
       !
       CALL ANCSOL(X,Y,Z,XCOMP,YCOMP,ZCOMP, &
            TQ,MAXTOT,NTOT,NUNIT,FIRSTU, &
            BEGIN,STOP,SKIP,NUNIT2,FIRSTU2, &
            BEGIN2,STOP2,SKIP2,DELTA,VELCD, &
            NSERIE,SNAME,STOT,ICLASS,GECOD, &
            SERVAL,LFFT,SAVEG,SFLCT, &
            SERPT,SERNQ, &
            IUNMAX,INOMAX, &
#if KEY_SHELL==1
            NSHL,NASHL,NSHBLK,SHLLST,SHBLKL, &  
#endif
#if KEY_SHELL==1
            QSHLUP,QSHIMG,QSHELL, & 
#endif
            QAT,ISLCT,ATOMIN,WMAIN, &
            QORIENT,QMASS,JSLCT, &
            MAXSER,NSRST,SERSET,SERDAT,SERCOR,SERLST,SERNCR, &
            SERACT,DOIMAGES,QBOTH,QAFFT)
       !
       call chmdealloc('corsol.src','CORSO1','ATOMIN',2,NATOM,intg=ATOMIN)
       call chmdealloc('corsol.src','CORSO1','SERACT',MAXSER,intg=SERACT)
       !
       !        save series specs pertaining to time series just read.
       DO I=1,NSRST
          ISERIE=SERSET(I)
          IF(STOT(ISERIE) == 0) THEN
             DO J=1,VECCOD(ISERIE)
                JSERIE=ISERIE+J-1
                STOT(ISERIE)=SERNCR(I)
                IF(NTOT < SERNCR(I)) STOT(ISERIE)=NTOT
                SSKIP(JSERIE)=1
                SDELTA(JSERIE)=DELTA*SKIP
                SOFFST(JSERIE)=0.0D0
                VELCOD(JSERIE)=VELCD
             ENDDO
          ENDIF
       ENDDO
       !
    ELSE IF(WD == 'READ') THEN
       ! PROCESS-READ-COMMAND
       !        get-series-number
       if( .not. func800()) goto 100
810    CONTINUE
       !
       ISERIE=I
       CALL READTS(ISERIE,NSERX,.FALSE., &
            MAXSER,MAXTOT,ICLASS,VELCOD,GECOD, &
            VECCOD,SERVAL,LFFT,STOT,SSKIP,SDELTA,SOFFST, &
            SAVEG,SFLCT,SERPT,SERNQ,SNAME,TQ,QSIZE,QAT)
       !        process-edit-command
       GOTO 700
       !
    ELSE IF(WD == 'WRIT' .OR. WD.EQ.'PRIN') THEN
       ! process-writ-command

       !        get-series-number
       if( .not. func800()) goto 100

815    CONTINUE
       !
       ISERIE=I
       CALL WRITS2(ISERIE,NSERX, &
            MAXSER,MAXTOT,ICLASS,VELCOD,GECOD, &
            VECCOD,SERVAL,LFFT,STOT,SSKIP,SDELTA,SOFFST, &
            SAVEG,SFLCT,SERPT,SERNQ,SNAME,TQ,QSIZE,QAT)
       !
    ELSE IF(WD == 'EDIT') THEN
       ! process-edit-command
       !        get-series-number
       if( .not. func800()) goto 100

820    CONTINUE
       GOTO 700
       !
       !
    ELSE IF(WD == 'SHOW') THEN
       ! process-show-command
       !        get-series-number
       if( .not. func800()) goto 100

825    CONTINUE
       !
       ISERIE=I
       CALL SHOWTS(ISERIE,NSERX, &
            MAXSER,MAXTOT,ICLASS,VELCOD,GECOD, &
            VECCOD,SERVAL,LFFT,STOT,SSKIP,SDELTA,SOFFST, &
            SAVEG,SFLCT,SERPT,SERNQ,SNAME,TQ,QSIZE,QAT)
       !
       !CC
       IF(PRNLEV >= 4) THEN
          write(outu,423) ' Atom pointers for all time series:'
          write(outu,423) ' SERPT:',(serpt(j),j=1,nserie)
          write(outu,423) ' SERNQ:',(sernq(j),j=1,nserie)
          write(outu,423) ' QAT  :',(qat(j),j=1,nq)
423       format(A,10I6/,(7X,10I6))
       ENDIF
       !CC
       !
    ELSE IF(WD == 'END ') THEN
       call corso1_deallocate
       RETURN
    ELSE
       CALL WRNDIE(0,'<CORSOL>','UNRECOGNIZED COMMAND')
    ENDIF wd_block
    GOTO 100
    !
    !=======================================================================
    ! to process-edit-spec
700 CONTINUE
    ISERIE=I
    VECCD=GTRMI(COMLYN,COMLEN,'VECC',MARK)
    IF(VECCD /= MARK) THEN
       NSERX=VECCD
       IF(ISERIE+NSERX-1 > MAXSER) NSERX=MAXSER-ISERIE+1
       IF(NSERX <= 0) NSERX=1
    ENDIF
    J=GTRMI(COMLYN,COMLEN,'INDE',MARK)
    IF(J == MARK) THEN
       IS=ISERIE
       IQ=ISERIE+NSERX-1
    ELSE
       IS=ISERIE+J-1
       IQ=IS
    ENDIF
    ICLAS=GTRMA(COMLYN,COMLEN,'CLAS')
    GECD=GTRMI(COMLYN,COMLEN,'SECO',MARK)
    NTOT=GTRMI(COMLYN,COMLEN,'TOTA',MARK)
    SKIP=GTRMI(COMLYN,COMLEN,'SKIP',MARK)
    DELTA=GTRMF(COMLYN,COMLEN,'DELT',ANUM)
    OFFST=GTRMF(COMLYN,COMLEN,'OFFS',ANUM)
    SERVL=GTRMF(COMLYN,COMLEN,'VALU',ANUM)
    NAME=GTRMA(COMLYN,COMLEN,'NAME')
    DO JSERIE=IS,IQ
       IF(ICLAS /= ' ') ICLASS(JSERIE)=ICLAS
       IF(GECD /= MARK) GECOD(JSERIE)=GECD
       IF(NTOT /= MARK) STOT(JSERIE)=NTOT
       IF(SKIP /= MARK) SSKIP(JSERIE)=SKIP
       IF(DELTA /= ANUM) SDELTA(JSERIE)=DELTA
       IF(OFFST /= ANUM) SOFFST(JSERIE)=OFFST
       IF(SERVL /= ANUM) SERVAL(JSERIE)=SERVL
       IF(VECCD /= MARK) VECCOD(JSERIE)=0
       IF(NAME /= ' ') SNAME(JSERIE)=NAME
    ENDDO
    IF(VECCD /= MARK) VECCOD(ISERIE)=NSERX
    !        process-show-command
    CALL SHOWTS(ISERIE,NSERX, &
         MAXSER,MAXTOT,ICLASS,VELCOD,GECOD, &
         VECCOD,SERVAL,LFFT,STOT,SSKIP,SDELTA,SOFFST, &
         SAVEG,SFLCT,SERPT,SERNQ,SNAME,TQ,QSIZE,QAT)
    GOTO 100 
    !
    !                end of subroutine statements
    !      ======================================================================= 
    !
    !
    !     ======================================================================= 
    !              internal subroutines

  contains

    logical function func800() result(sub800ok) 
      !       to get-series-number
      CALL NEXTWD(COMLYN,COMLEN,WD,WDMAX,WDLEN)
      IF(EQSTA(WD,WDLEN,'ALL')) THEN
         I=1
         NSERX=NSERIE
      ELSE IF(EQSTA(WD,MIN(4,WDLEN),'CORR')) THEN
         I=ICORR
         NSERX=1
      ELSE
         I=0
         DO J=1,NSERIE
            N=4
            CALL TRIME(SNAME(J),N)
            IF(EQST(WD,WDLEN,SNAME(J),N)) THEN
               I=J
               GOTO 250
            ENDIF
         ENDDO
         IF(PRNLEV >= 2) WRITE(OUTU,'(3A)') '"',WD(1:WDLEN), &
              '" doesnt exist. Command ignored.'
         CALL WRNDIE(0,'<CORSOL>','Time series not found')

         sub800ok=.false.
         return
250      CONTINUE
         NSERX=VECCOD(I)
         IF(NSERX <= 0) NSERX=1
      ENDIF
      sub800ok=.true.
      return
    end function func800
    !
    subroutine corso1_deallocate
      call chmdealloc('corsol.src','CORSO1','ISLCT',NATOM,intg=ISLCT)
      call chmdealloc('corsol.src','CORSO1','JSLCT',NATOM,intg=JSLCT)
      call chmdealloc('corsol.src','CORSO1','TQ',MAXTOT,maxsr,crl=TQ)

      call chmdealloc('corsol.src','CORSO1','VELCOD',maxsr,intg=VELCOD)
      call chmdealloc('corsol.src','CORSO1','GECOD',maxsr,intg=GECOD)
      call chmdealloc('corsol.src','CORSO1','VECCOD',maxsr,intg=VECCOD)
      call chmdealloc('corsol.src','CORSO1','STOT',maxsr,intg=STOT)
      call chmdealloc('corsol.src','CORSO1','SSKIP',maxsr,intg=SSKIP)
      call chmdealloc('corsol.src','CORSO1','SERPT',maxsr,intg=SERPT)
      call chmdealloc('corsol.src','CORSO1','SERNQ',maxsr,intg=SERNQ)

      call chmdealloc('corsol.src','CORSO1','SERSET',maxser,intg=SERSET)
      call chmdealloc('corsol.src','CORSO1','SERNCR',maxser,intg=SERNCR)
      call chmdealloc('corsol.src','CORSO1','IUNMAX',maxser,intg=IUNMAX)
      call chmdealloc('corsol.src','CORSO1','INOMAX',maxser,intg=INOMAX)

      call chmdealloc('corsol.src','CORSO1','QAT',qsize,intg=QAT)

      call chmdealloc('corsol.src','CORSO1','SERVAL',maxsr,crl=SERVAL)
      call chmdealloc('corsol.src','CORSO1','SDELTA',maxsr,crl=SDELTA)
      call chmdealloc('corsol.src','CORSO1','SOFFST',maxsr,crl=SOFFST)
      call chmdealloc('corsol.src','CORSO1','SAVEG',maxsr,crl=SAVEG)
      call chmdealloc('corsol.src','CORSO1','SFLCT',maxsr,crl=SFLCT)

      call chmdealloc('corsol.src','CORSO1','lfft',  MAXSR,log=lfft)

      call chmdealloc('corsol.src','CORSO1','sname', MAXSR,ch8=sname)
      call chmdealloc('corsol.src','CORSO1','iclass',MAXSR,ch8=iclass)
    end subroutine  corso1_deallocate
  END    SUBROUTINE CORSO1

  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE INTIC3(COMLYN,COMLEN,ICODE,NQ,QAT,QSIZE,SERN, &
       GECD,SERVL)
    !
    !     THIS ROUTINE INTERPRETS THE ATOMS SPECIFIED FOR A TIME SERIES
    !     ALSO, CODE VALUES ARE PUT IN IF ENERGY IS DESIRED.
    !
  use chm_kinds
  use stream
  use memory
    implicit none
    integer,allocatable,dimension(:) :: ZAT
    character(len=*) COMLYN
    INTEGER COMLEN,ICODE,NQ
    INTEGER QAT(*)
    INTEGER QSIZE,SERN,GECD
    real(chm_real) SERVL
    !
    !
    INTEGER NQ1
    !
    call chmalloc('corsol.src','INTIC3','ZAT',QSIZE-NQ,intg=ZAT)
    CALL INTIC4(COMLYN,COMLEN,ICODE,NQ1,QAT(NQ+1),QSIZE-NQ,SERN,GECD, &
         ZAT,SERVL)
    call chmdealloc('corsol.src','INTIC3','ZAT',QSIZE-NQ,intg=ZAT)
    NQ=NQ+NQ1
    RETURN
  END SUBROUTINE INTIC3

  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE INTIC4(COMLYN,COMLEN,ICODE,NQ,QAT,QSIZE,SERN,GECD, &
       ZAT,SERVL)
    !
    !     THIS ROUTINE INTERPRETS THE ATOMS SPECIFIED FOR A TIME SERIES
    !     ALSO, CODE VALUES ARE PUT IN IF ENERGY IS DESIRED.
    !
  use chm_kinds
  use dimens_fcm
  use psf
  use param
  use stream
  use memory
  use string

    implicit none
    !
    integer,allocatable,dimension(:) :: ISLCT
    character(len=*) COMLYN
    INTEGER COMLEN,ICODE,NQ
    INTEGER QAT(*),ZAT(*)
    INTEGER QSIZE,SERN,GECD
    real(chm_real) SERVL
    !
    !
    INTEGER ICD,NZAT,MAX,I,J,NS
    !
    NZAT=0
    SERN=0
    NQ=0
    CALL TRIME(COMLYN,COMLEN)
    call chmalloc('corsol.src','INTIC4','ISLCT',NATOM,intg=ISLCT)
    !
    !     now snatch the first selection from the COMLYN
    CALL SHLSEL(COMLYN,COMLEN,ZAT,NZAT,ISLCT)
    !
    IF(NZAT == 0) &
         CALL WRNDIE(-5,'<INTIC>', &
         'No solvent/solute atoms selected!.')
    !
    MAX=NZAT
    IF(MAX > QSIZE) THEN
       CALL WRNDIE(-5,'<INTIC>','Overflow of atom list.')
       RETURN
    ENDIF
    !
    SERN=NZAT
    NQ=NZAT
    DO I=1,SERN
       QAT(I)=ZAT(I)
    ENDDO
    !
    !     and now get the second selection if there is one
    !     (DDIP needs 2 selections, SATM olny one)
    IF(ICODE >= 1) THEN
       SERVL=NQ
       CALL SHLSEL(COMLYN,COMLEN,ZAT,NZAT,ISLCT)
       DO I=1,NZAT
          QAT(NQ+I)=ZAT(I)
       ENDDO
       NQ=NQ+NZAT
       SERN=SERN+NZAT
    ENDIF
    !
    call chmdealloc('corsol.src','INTIC4','ISLCT',NATOM,intg=ISLCT)
    !
    RETURN
  END SUBROUTINE INTIC4
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE WRITS2(ISERIE,NSER, &
       MAXSER,MAXTOT,ICLASS,VELCOD,GECOD, &
       VECCOD,SERVAL,LMASS,STOT,SSKIP,SDELTA,SOFFST, &
       SAVEG,SFLCT,SERPT,SERNQ,SNAME,TQ,QSIZE,QAT)
    !
    !
    !     This routine writes time series or time series plots.
    !     ... a little modification for CORSol output
    !
    !       { ALL                }             [ FILE          ]
    ! WRITe { time-series-name   } unit-spec   [ CARD          ]
    !       {CORRelation-function}             [ PLOT          ]
    !                                          [ DUMB [ TIME ] ]
    !
    !      By  Bernard R. Brooks    18-Oct-1984
    !      modified: Tibor Rudas 26-Feb-2003
    !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use comand
  use consta
  use ctitla
  use stream
  use string
  use chutil,only:atomid
    implicit none
    INTEGER ISERIE,NSER,MAXSER,MAXTOT
    character(len=*) ICLASS(*)
    INTEGER VELCOD(*),GECOD(*),VECCOD(*)
    real(chm_real) SERVAL(*)
!    Integer SERVAL(*)
    LOGICAL LMASS(*)
    INTEGER STOT(*),SSKIP(*)
    real(chm_real) SDELTA(*),SOFFST(*)
    real(chm_real) SAVEG(*),SFLCT(*)
    INTEGER SERPT(*),SERNQ(*)
    character(len=*) SNAME(*)
    real(chm_real) TQ(MAXTOT,MAXSER)
    INTEGER QSIZE
    INTEGER QAT(*)
    !
    !
    INTEGER ISTOP,VECCD,IUNIT,I,J,NTOT
    real(chm_real) DEL
    LOGICAL LTIME, ERROR
    character(len=4) SEGID,RESID,RESNAM,ATNAM
    INTEGER PT,ATOM,i1
    !
    ISTOP=ISERIE+NSER-1
    IF(ISTOP > MAXSER) ISTOP=MAXSER
    IF(ISTOP < ISERIE) ISTOP=ISERIE
    VECCD=ISTOP-ISERIE+1
    DEL=SSKIP(ISERIE)*SDELTA(ISERIE)
    !
    IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
    IF(IUNIT < 0) THEN
       CALL WRNDIE(0,'<WRITS2>','NO UNIT SPECIFIED')
       RETURN
    ENDIF
    !
    CALL RDTITL(TITLEA,NTITLA,ISTRM,0)
    IF(OUTU == IUNIT .AND. PRNLEV < 2) RETURN
    IF(OUTU /= IUNIT .AND.  IOLEV < 0) RETURN
    !
    IF(INDXA(COMLYN,COMLEN,'CARD') > 0) THEN
       IF((ICLASS(ISERIE) == 'DDIP').OR. &
            (ICLASS(ISERIE) == 'SDDP')) THEN
          !           do output of primary atoms
          PT=2
          i1=serval(iserie)
          DO I=1,i1
             ATOM=QAT(SERPT(ISERIE)+I-1)
             CALL ATOMID(ATOM,SEGID,RESID,RESNAM,ATNAM)
             WRITE(IUNIT,850) PT,PT+2,SEGID,RESID,RESNAM,ATNAM
             PT=PT+3
          ENDDO
       ENDIF
       ATOM=SERNQ(ISERIE)-SERVAL(ISERIE)
       WRITE(IUNIT,851) ATOM
850    FORMAT('# rows ',I5,' - ',I5,4(1X,A4))
851    FORMAT('# and ',I20,' secondary atoms')
       ! process-card-write
       !
       CALL WRTITL(TITLEA,NTITLA,IUNIT,0)
       WRITE(IUNIT,25) VECCD
25     FORMAT(' NSER:',I4)
       WRITE(IUNIT,35) (SNAME(I),I=ISERIE,ISTOP)
35     FORMAT(' NAMES: ',10(4X,A4,4X))
       WRITE(IUNIT,45) (STOT(I),I=ISERIE,ISTOP)
45     FORMAT(' TOTALS:',10(I12))
       WRITE(IUNIT,50) (VECCOD(I),I=ISERIE,ISTOP)
50     FORMAT(' VECCOD:',10(I12))
       WRITE(IUNIT,55) (ICLASS(I),I=ISERIE,ISTOP)
55     FORMAT(' CLASS: ',10(4X,A4,4X))
       WRITE(IUNIT,65) (VELCOD(I),I=ISERIE,ISTOP)
65     FORMAT(' VELCOD:',10(I12))
       WRITE(IUNIT,75) (SSKIP(I),I=ISERIE,ISTOP)
75     FORMAT(' SKIP:  ',10(I12))
       WRITE(IUNIT,85) (SDELTA(I),I=ISERIE,ISTOP)
85     FORMAT(' DELTA: ',10(F12.6))
       WRITE(IUNIT,90) (SOFFST(I),I=ISERIE,ISTOP)
90     FORMAT(' OFFST: ',10(F12.6))
       WRITE(IUNIT,95) (GECOD(I),I=ISERIE,ISTOP)
95     FORMAT(' GECOD: ',10(I12))
       WRITE(IUNIT,105) (SERVAL(I),I=ISERIE,ISTOP)
105    FORMAT(' VALUE: ',10(F12.6))
       NTOT=0
       DO I=ISERIE,ISTOP
          IF(STOT(I) > NTOT) NTOT=STOT(I)
       ENDDO
       DO I=1,NTOT
          WRITE(IUNIT,115) I,(TQ(I,J),J=ISERIE,ISTOP)
115       FORMAT(I10,3X,60(1X,G17.11))
       ENDDO
       !
    ELSE IF(INDXA(COMLYN,COMLEN,'DUMB') > 0) THEN
       IF((ICLASS(ISERIE) == 'DDIP').OR. &
            (ICLASS(ISERIE) == 'SDDP')) THEN
          !           do output of primary atoms
          PT=2
          i1=SERVAL(ISERIE)
          DO I=1,i1
             ATOM=QAT(SERPT(ISERIE)+I-1)
             CALL ATOMID(ATOM,SEGID,RESID,RESNAM,ATNAM)
             WRITE(IUNIT,850) PT,PT+2,SEGID,RESID,RESNAM,ATNAM
             PT=PT+3
          ENDDO
       ENDIF
       ATOM=SERNQ(ISERIE)-SERVAL(ISERIE)
       WRITE(IUNIT,851) ATOM
       ! process-dumb-write
       !
       LTIME=(INDXA(COMLYN,COMLEN,'TIME') > 0)
       NTOT=0
       DO I=ISERIE,ISTOP
          IF(NTOT < STOT(I)) NTOT=STOT(I)
       ENDDO
       DO I=1,NTOT
          IF(LTIME) THEN
             WRITE(IUNIT,145) (I-1)*DEL+SOFFST(ISERIE), &
                  (TQ(I,J),J=ISERIE,ISTOP)
          ELSE
             WRITE(IUNIT,145) (TQ(I,J),J=ISERIE,ISTOP)
          ENDIF
145       FORMAT(1X,31(G22.10))
       ENDDO
       !
    ELSE IF(INDXA(COMLYN,COMLEN,'PLOT') > 0) THEN
       ! process-plot-write
       !
       NTOT=STOT(ISERIE)
       WRITE(IUNIT) TITLEA(1)(1:80)
       WRITE(IUNIT) NTOT
       WRITE(IUNIT) ((I-1)*DEL+SOFFST(ISERIE),I=1,NTOT)
       WRITE(IUNIT) (TQ(I,ISERIE),I=1,NTOT)
       !
    ELSE
       ! process-binary-write
       !     IF(INDXA(COMLYN,COMLEN,'FILE') > 0)
       CALL WRTITL(TITLEA,NTITLA,IUNIT,-1)
       WRITE(IUNIT) VECCD
       WRITE(IUNIT) (SNAME(I),I=ISERIE,ISTOP)
       WRITE(IUNIT) (STOT(I),I=ISERIE,ISTOP)
       WRITE(IUNIT) (VECCOD(I),I=ISERIE,ISTOP)
       WRITE(IUNIT) (ICLASS(I),I=ISERIE,ISTOP)
       WRITE(IUNIT) (VELCOD(I),I=ISERIE,ISTOP)
       WRITE(IUNIT) (SSKIP(I),I=ISERIE,ISTOP)
       WRITE(IUNIT) (SDELTA(I),I=ISERIE,ISTOP)
       WRITE(IUNIT) (SOFFST(I),I=ISERIE,ISTOP)
       WRITE(IUNIT) (GECOD(I),I=ISERIE,ISTOP)
       WRITE(IUNIT) (SERVAL(I),I=ISERIE,ISTOP)
       DO I=ISERIE,ISTOP
          WRITE(IUNIT) (TQ(J,I),J=1,STOT(I))
       ENDDO
       !
    ENDIF
    IF(IUNIT /= OUTU) CALL VCLOSE(IUNIT,'KEEP',ERROR)
    !
    !----------------------------------------------------------------------
    !
    return
  end SUBROUTINE WRITS2
#endif /* (corsol)*/

  SUBROUTINE NULL_CORSOL
    RETURN
  END SUBROUTINE NULL_CORSOL

end module corsol_mod

